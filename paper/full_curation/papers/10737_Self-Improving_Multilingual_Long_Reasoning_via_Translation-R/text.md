# arXiv:2602.05940v1[cs.CL]5Feb2026

## Self-Improving Multilingual Long Reasoning via Translation-Reasoning Integrated Training

Junxiao Liu1, Zhijun Wang1, Yixiao Li1, Zhejian Lai1, Liqian Huang2 Xin Huang3∗, Xue Han3, Junlan Feng3, Shujian Huang1* 1 National Key Laboratory for Novel Software Technology, Nanjing University 2 University of Tübingen 3 China Mobile Communications Company Limited Research Institute

{junxiaoliu,wangzj,liyixiao,laizj}@smail.nju.edu.cn, liqian.huang@student.uni-tuebingen.de, {huangxin,hanxuejt}@cmjt.chinamobile.com,fengjunlan@chinamobile.com, huangsj@nju.edu.cn

### Abstract

Long reasoning models often struggle in multilingual settings: they tend to reason in English for non-English questions; when constrained to reasoning in the question language, accuracies drop substantially. The struggle is caused by the limited abilities for both multilingual question understanding and multilingual reasoning. To address both problems, we propose TRIT (Translation-Reasoning Integrated Training), a self-improving framework that integrates the training of translation into multilingual reasoning. Without external feedback or additional multilingual data, our method jointly enhances multilingual question understanding and response generation. On MMATH, our method outperforms multiple baselines by an average of 7 percentage points, improving both answer correctness and language consistency. Further analysis reveals that integrating translation training improves cross-lingual question alignment by over 10 percentage points and enhances translation quality for both mathematical questions and general-domain text, with gains up to 8.4 COMET points on FLORES200.1

### 1 Introduction

Long reasoning models (LRMs), typically trained through reinforcement learning from verifiable rewards (RLVR) (DeepSeek-AI et al., 2025), have achieved strong performance on complex reasoning tasks under the "think-then-answer" paradigm (Yang et al., 2025; OpenAI et al., 2025).

However, such capabilities are not the same for different languages: when the input questions are non-English, LRMs often tend to reason in English, i.e. inconsistent language usage; forcing models to reason in the question language typically leads to a pronounced performance drop accompanied

*Corresponding author. 1Code and data are available at https://github.com/

NJUNLP/TRIT

by degenerative repetition, indicating poor multilingual reasoning (Qi et al., 2025; Wang et al., 2025). Furthermore, when reasoning is constrained to a single language, models still exhibit a substantial performance gap between questions expressed in English and non-English, suggesting biases in question understanding (Ko et al., 2025; Kang et al., 2026).

Previous work leverages external evaluators to align multilingual reasoning traces with English (e.g. M-Thinker (Zhang et al., 2025) and MAPO (She et al., 2024)). These approaches pay little attention to the problem in question understanding. However, when the question is not correctly understood, models may reason in the wrong direction from the start. In these cases, aligning reasoning traces may not be effective in fixing the misunderstanding. Moreover, they typically require separate feedback models to guide generation, thereby introducing substantial computational training overhead.

In this paper, we propose TRIT (TranslationReasoning Integrated Training), a self-improving reinforcement learning framework that integrates the training of translation with multilingual reasoning. TRIT jointly improves multilingual question understanding and reasoning, without external feedback or additional multilingual data (Figure 1).

More specifically, our framework consists of two stages. Firstly, the model is trained to improve its ability to answer English questions in the target language (cross-lingual reasoning). The cross-lingual reasoning ability also serves for an accuracy-based filtering: only questions the model can reliably solve in the target language proceed to the subsequent stage.

Secondly, the model is trained to (1) translate English questions into the target language (translation), and (2) solve the translated questions with the target language (target language reasoning). If the translated question cannot be solved, it

indicates a translation problem rather than a reasoning capability issue, since the model has already demonstrated the ability to solve the question in cross-lingual reasoning. In this way, we use the reasoning performance to provide rewards for the translation training, thus avoiding using any external feedback or resources. Both reasoning tasks enjoy verifiable rewards. All tasks are jointly optimized via reinforcement learning.

We evaluate our method on models with diverse multilingual capabilities. Experiments on MMATH show that our approach substantially improves performance, outperforming baselines by 7 percentage points on average while achieving near-perfect language consistency. Further analyses reveal that using reasoning accuracy as a proxy signal for translation quality improves translation both in-domain (mathematical questions) and out-of-domain (general text), with gains up to 8.4 (COMET) on FLORES-200. Translation training improves representation similarity between English and non-English questions by over 10 percentage points at best, suggesting an enhanced question alignment and understanding.

### 2 Related Work

While large language models demonstrate strong reasoning capabilities in English, their multilingual reasoning performance remains weaker (Qi et al., 2025; Wang et al., 2025; Chen et al., 2024). Existing attempts to improve multilingual reasoning have mainly relied on supervised fine-tuning with translated chain-of-thought data (Chen et al., 2024), or on preference optimization and reinforcement learning to explicitly encourage multilingual chains of thought to align with English trajectories (She et al., 2024; Park et al., 2025; Hwang et al., 2025; Zhang et al., 2025). These approaches largely overlook differences in how models understand questions across languages.

Prior work shows that even when the reasoning language is fixed to a single language (e.g., English, Korean), performance can still vary substantially with the language of the input question (Ko et al., 2025; Kang et al., 2026), which suggests that multilingual question understanding remains inadequate. To address this, QAlign (Zhu et al., 2024) trains translation and reasoning in two separate stages: first training question translation, then training English reasoning. However, this pipeline relies on English reasoning to solve non-English questions,

without directly enhancing the model’s native multilingual reasoning capability.

### 3 Methods

We propose TRIT, a reinforcement learning framework that jointly enhances multilingual question understanding and reasoning without external feedback or additional multilingual data.

#### 3.1 Reward Modeling

To encourage correct, language-consistent, and nonrepetitive responses, we design a reward function with four components:

- • Accuracy reward (racc): racc = 1 if the answer is correct, otherwise 0.
- • Language consistency reward (rlang): We use langdetect2 to verify that the reasoning

trace is in the target language. rlang = 1 if consistent, otherwise 0.

- • Repetition penalty (rrep): We detect degenerate repetition at sentence and n-gram levels

(details in Appendix A). rrep = 1 if no repetition, otherwise 0.

- • Format reward (rfmt): rfmt = 1 if the output follows the <think>...</think> format, otherwise 0.

We adopt a compositional reward structure where correctness is rewarded only when all quality constraints are satisfied. More specifically, the model receives positive reward for correct answers only if the output is well-formed (rfmt = 1), language-consistent (rlang = 1), and free of repetition (rrep = 1). This design ensures high-quality responses across all dimensions.

 

1, if C ∧ (racc = 1), 0.1, if C ∧ (racc = 0), 0, otherwise,

rfinal =



C = (rfmt = 1 ∧ rlang = 1 ∧ rrep = 1).

#### 3.2 Translation-Reasoning Integrated Training Framework

As shown in Algorithm 1, TRIT consists of two components. The first, Cross-Lingual Reasoning, identifies English questions that can be reliably

2https://github.com/Mimino666/langdetect

|If you have 20 apples and give away 7, how many are left?<br><br>Qen 𝟏𝟑<br><br><think>(Japanese reasoning chain)…\boxed{13}<br><br><think>(Japanese reasoning chain)…\boxed{12}<br><br>JA<br><br>Cross-Lingual Reasoning Translation<br><br>𝒓𝒂𝒗𝒈 ≥ θ<br><br>𝒓𝒂𝒗𝒈 < θ<br><br>Filtered Out<br><br>あなたは 20 個のリンゴを持って いて、7 個あげたら（give away）、残りはいくつですか？<br><br>あなたはリンゴを20個持っていま す。7個もらったら（receive）、 残りはいくつですか？<br><br><think>(Japanese reasoning<br><br>chain)…\boxed{13}<br><br><think>(Japanese reasoning<br><br>chain)…\boxed{14}<br><br><br><think>(Japanese reasoning chain)…\boxed{27}<br><br><think>(English reasoning chain)…\boxed{27}<br><br>Target Language Reasoning<br><br>Target Language Question<br><br>𝑨𝒄𝒄 > 𝟎<br><br>rtrans=1<br><br>𝑨𝒄𝒄 = 𝟎<br><br>rtrans=0<br><br>Qen_filtered<br><br>only filtered question’s cross-lingual reasoning data are used for training<br><br>(Qen, Rtgt, rreasoning)<br><br>all translation data are used for training<br><br>(Qen_filtered, Qtgt, rtrans)<br><br>Q: Question R: Response A: Answer r: reward<br><br>(Qtgt, Rtgt, rreasoning)<br><br>only correct translation’s reasoning data are used for training<br><br>GRPO Training<br><br>data movement reward feedback<br><br>…<br><br>…<br><br>…<br><br>…<br><br>Qtgt<br><br><think>(English reasoning chain)…\boxed{13}<br><br><think>(Japanese reasoning chain)…\boxed{13}<br><br>あなたがリンゴを20個持っていて、 そこから7個受け取る（receive） と、残りはいくつになりますか？|
|---|

- Figure 1: The Framework of TRIT. Our framework consists of two stages: Cross-Lingual Reasoning filters questions by accuracy threshold θ, and Translation-Reasoning Integration & Feedback trains both translation and target-language reasoning using filtered questions (Translation errors are denoted with red color, which results in wrong reasoning results, and get 0 as rtrans).

3.2.2 Translation-Reasoning Integration & Feedback

solved in the target language to ensure accurate feedback. The second, Translation–Reasoning Integration & Feedback, forms a closed loop where translation and reasoning mutually improve the model’s multilingual reasoning ability.

After filtering questions in the cross-lingual reasoning stage, we train the model to accurately translate them into the target language within <Translation>...</Translation> tags. Translation quality is evaluated through a two-step process. First, we apply basic quality checks: translations violating language or format constraints receive rtrans = 0 and are excluded from further processing. Second, for valid translations, we use a deferred reward mechanism based on downstream reasoning performance.

#### 3.2.1 Cross-Lingual Reasoning

We train the model to answer English questions in the target language. To establish initial crosslingual reasoning capability, we perform cold-start training on a small set of supervised cross-lingual examples. RLVR is then performed together with the other tasks.

To ensure that the model correctly captures the semantics of the original English questions and to avoid attributing the model’s reasoning errors to translation quality in later stages, we use an accuracy-based filtering. Only questions the model can currently solve proceed to subsequent stages. Concretely, we prompt the model to answer English questions directly in the target language using language-specific instructions (Figure 8), and compute a final reward rfinal for each response. We compute each question’s average reward ravg and include only those with ravg ≥ θ in the next phase.

More specifically, we train target-language reasoning by prompting the model to solve the translated questions in the target language. For each translated question, we compute the average reasoning accuracy (Acc) of sampled reasoning paths. If Acc > 0, indicating that the translation preserves key semantics, we assign rtrans = 1; otherwise, rtrans = 0. This design creates a closed loop: translation provides multilingual question data for reasoning, while reasoning accuracy provides reward signals for translation quality. This mutual feedback enables self-improvement without external feedback.

The training strengthens the model’s crosslingual reasoning over time. As the model improves, more questions satisfy the accuracy criterion, ensuring stable training across a broader data distribution.

In addition to the cross-lingual reasoning data collected in the first stage, we collect two types of training data in this stage. For translation training, we keep all translation data pairs(every En-

Algorithm 1 TRIT Training Algorithm

- 1: Input: English questions Qen, target language Ltgt, threshold θ
- 2: for each training iteration do
- 3: Initialize Dcross, Dtrans, Dtgt ← ∅; Qfiltered ← ∅
- 4: // Phase 1: Cross-lingual Reasoning
- 5: for qen ∈ Qen do
- 6: Sample {oi}Gi=1 ∼ πθ(·|qen, Ltgt); Compute ravg = 1 G

G i=1 rifinal

- 7: if ravg ≥ θ then
- 8: Qfiltered ← Qfiltered ∪ {qen}
- 9: Dcross ← Dcross ∪ {(qen, oi, rifinal)}Gi=1
- 10: end if
- 11: end for
- 12: // Phase 2: Translation-Reasoning Integration & Feedback
- 13: for qen ∈ Qfiltered do
- 14: Sample {tj}Kj=1 ∼ πθ(·|qen, Ltgt); Set rjtrans ← pending (or 0 if invalid)
- 15: for valid tj do
- 16: Sample {oi}Gi=1 ∼ πθ(·|tj, Ltgt); Compute Acc = G1 i riacc

- 17: rjtrans ← I(Acc > 0); Add to Dtgt if Acc > 0
- 18: end for
- 19: Dtrans ← Dtrans ∪ {(qen, tj, rjtrans)}
- 20: end for
- 21: Train with GRPO on Dcross ∪ Dtrans ∪ Dtgt; Update πθ
- 22: end for

glish question paired with its translation). For target-language reasoning training, we only collect question-response pairs from correctly translated questions (Acc > 0). This filtering prevents pairing mistranslated questions with answers, which would provide misleading training signals.

#### 3.3 Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) (Shao et al., 2024) has been widely adopted for RL training to enhance LLM ability. For each question sampled from Q, GRPO samples a group of responses {oi}Gi=1. Specifically, the objective function is formulated as follows:

JGRPO(θ) = E q ∼ P(Q), {oi}Gi=1 ∼ πθold(O | q)

|oi|

G

1 |oi|

1 G

min ρi,t(θ)Aˆi,t,

t=1

i=1

clip(ρi,t(θ),1 − ϵ, 1 + ϵ)Aˆi,t − β DKL(πθ ∥πref)}.

(1)

where ρi,t(θ) = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t) denotes the im-

portance sampling ratio. The advantage term Aˆi,t is derived by standardizing the rewards within each

group:

ri − mean({r1,...,rG}) std({r1,...,rG})

Aˆi,t =

(2)

By estimating the baseline directly from group statistics, GRPO obviates the necessity of an explicit value network, and mitigates the variance of the advantage estimation. We apply GRPO to optimize all training data in TRIT. For each data type (cross-lingual reasoning, translation, and targetlanguage reasoning), we use the sampled response groups to compute advantages within each group, and accumulate the GRPO loss across all data.

### 4 Experiments

#### 4.1 Experiment Setup

Backbone Models. We evaluate our framework on three models with diverse multilingual capabilities: DeepSeek-Distill-Qwen-1.5B, Qwen3-1.7B, and Qwen3-4B. DeepSeek-Distill-Qwen-1.5B represents a model with weaker multilingual reasoning and translation abilities, while the Qwen3 family provides strong, state-of-the-art models. This diversity allows us to assess the robustness and generality of our framework.

Benchmarks and Evaluation Metrics. We evaluate multilingual reasoning on MMATH, which contains problems of varying difficulty from AIME24, AIME25, CNMO, and MATH500, with multilingual versions of all questions. We report the macro average across subsets as the final score.

Following M-Thinker (Zhang et al., 2025), we use three metrics: Language Consistency (LC) measures whether the reasoning trace is in the question language; Accuracy (Acc) evaluates response correctness; and and LC&Acc measures the percentage of responses that are both correct and language-consistent, serving as our primary metric.

Baselines We compare against the following baselines:

- • Prompt Control: (Wang et al., 2025) Appends language-control instructions at inference time without parameter updates. Please refer to Figure 9 for detailed prompt.
- • SFT: Fine-tunes on supervised data

(Questiontgt,Responsetgt) generated by Qwen3-32B, where both questions and responses are in the target language.

| |FR|PT|JA|KO|TH|Non-EN|EN|
|---|---|---|---|---|---|---|---|
|Methods|lc&acc acc lc<br><br>|lc&acc acc lc<br><br>|lc&acc acc lc<br><br>|lc&acc acc lc<br><br>|lc&acc acc lc<br><br>|ALL-AVG|lc&acc|
|DeepSeek-Distill-1.5B Prompt Control SFT Naive RL SLC RL<br><br>M-Thinker ⇒ Iter-1<br><br>M-Thinker ⇒ Iter-2 External-Translation TRIT TRIT⇒Iter2<br><br><br>|6.3 34.8 30.9 10.1 34.4 47.8 22.7 22.7 98.8 2.0 46.5 6.7 36.4 36.4 99.4 35.6 35.6 99.8<br><br>39.5 39.9 99.7<br><br>40.6 40.6 99.9 45.1 45.1 99.9 49.0 49.0 99.9<br><br><br>|10.3 34.4 48.9 15.1 30.1 67.7 24.2 24.5 98.9 0.0 45.3 0.0<br><br>38.0 38.1 99.5 33.9 34.3 99.6 41.2 41.3 99.5 40.6 40.6 99.9<br><br>39.9 39.9 99.9 44.8 44.9 99.9<br><br><br>|0.3 30.4 3.5<br><br>1.8 31.3 19.1 11.5 11.5 97.6<br><br><br>0.0 40.6 0.0 22.7 22.7 99.9 30.1 30.1 99.9 36.4 36.4 100.0<br><br>29.8 29.8 99.6<br><br>30.4 30.4 99.6 39.1 39.1 99.9<br><br><br>|0.1 32.2 1.3 0.5 28.5 4.5 10.1 10.3 94.9 0.0 39.7 0.0 0.0 38.9 6.7<br><br>23.6 23.7 99.4<br><br>29.8 32.8 86.0<br><br>24.1 24.1 99.8 22.3 22.3 99.7<br><br>30.9 30.9 99.9<br><br><br><br><br>|0.4 24.4 11.9 0.4 22.6 18.7 9.6 9.6 99.5 0.0 37.4 0.0 23.6 23.7 99.5 25.7 26.0 99.6 30.2 30.5 99.7<br><br>28.1 28.1 99.9<br><br>29.7 29.7 99.9 37.3 37.3 99.9<br><br><br>|3.5 5.6 15.6 0.4 24.1 29.8 35.4 32.6 33.5 40.2<br><br>|42.9 42.6 38.5 47.6 48.4 38.9 37.6 46.1 45.1 50.7<br><br>|
|Qwen3-1.7B Prompt Control SFT Naive RL SLC RL M-Thinker External-Translation TRIT<br><br>|0.0 42.8 0.0 0.0 45.1 0.0 35.0 37.2 96.9 0.0 50.5 0.0 40.6 40.6 99.9 42.0 42.0 99.9 46.0 46.0 99.9 48.5 48.5 99.8<br><br>|0.0 43.3 0.0 2.0 42.7 6.7 36.4 36.5 99.4 0.0 51.1 0.0 41.3 41.3 99.9 45.3 43.1 99.9 49.0 49.0 100.0<br><br>49.4 49.4 99.9<br><br>|0.0 40.7 0.0 2.0 39.9 6.7 25.6 25.6 99.4 0.0 46.4 0.0 32.0 32.0 99.7 34.0 34.0 99.8 40.2 40.2 100.0<br><br>43.8 43.8 99.9<br><br>|0.0 41.2 0.0 2.0 42.6 6.7 24.6 25.0 99.1 0.0 45.9 0.0 34.4 34.4 100.0 31.1 31.1 99.9 39.0 39.0 99.9 38.5 38.5 99.7<br><br>|0.0 40.0 0.0 6.0 38.2 20.0<br><br>25.5 25.6 99.2 0.0 46.8 0.0 35.0 35.0 99.9 34.0 34.0 99.8 39.2 39.2 100.0<br><br>42.6 42.6 99.9<br><br>|0.0 2.4 29.4 0.0 36.7 37.3 42.7 44.5<br><br>|41.7 42.1 34.4 54.5 39.7 47.4 50.6 53.3<br><br>|
|Qwen3-4B Prompt Control SFT Naive RL SLC RL M-Thinker External-Translation TRIT<br><br>|0.0 53.2 0.0 2.6 53.7 5.6 37.5 38.0 99.3 0.0 65.1 0.0 60.9 60.9 100.0 60.8 60.8 100.0<br><br>63.4 63.4 99.9<br><br>64.6 64.6 100.0<br><br><br>|0.0 53.3 0.0 3.6 55.2 4.8 38.3 38.3 99.5 0.0 64.3 0.0 63.2 63.2 100.0<br><br>60.5 60.5 99.7<br><br>61.2 61.2 99.9 65.2 65.2 99.9<br><br><br>|0.0 51.8 0.0 0.0 54.5 0.0 25.6 25.6 99.2 0.0 60.4 0.0<br><br>51.8 51.7 99.7<br><br>51.9 51.9 100.0 55.5 55.5 99.9 58.1 58.1 100.0<br><br><br>|0.0 52.3 0.0 0.0 52.6 0.1 25.2 25.2 99.1 0.0 62.7 0.0 48.9 48.9 99.8 52.9 53.0 99.9<br><br>55.2 55.2 99.8<br><br>55.2 55.2 100.0<br><br><br>|0.0 50.9 0.0 0.9 51.7 3.2 19.9 19.9 99.9 0.0 62.3 0.0 53.0 53.0 99.9 53.3 53.3 99.9<br><br>58.5 58.5 100.0 57.7 57.7 100.0<br><br>|0.0 1.4 29.3 0.0 55.6 55.9 58.8 60.2<br><br>|51.4 51.7 46.7 65.8 39.7 25.2 52.1 61.0<br><br>|

Table 1: Main results on MMATH. We evaluate on five in-domain languages (FR, PT, JA, KO, TH) and one out-of-domain language (EN). TRIT consistently outperforms all baselines across different backbone models. LC&Acc is our primary metric. Best results in bold.

- • Naive RL: Optimizes only response correct-

ness using the accuracy reward (racc), without enforcing language consistency.

- • SLC-RL: (Mistral-AI et al., 2025) Adds a soft language reward (0.1) to Naive RL when the response matches the target language.
- • M-Thinker: Uses language consistency and cross-lingual thinking alignment rewards with an external model to align multilingual reasoning traces with English.
- • External-Translation: Employs an external translation model (DeepSeek-V3.2-Exp) to supply high-quality translations. The training focuses exclusively on reasoning (crosslingual and target-language) rather than translation learning.

All experiments use training data constructed from DAPO-MATH-17K (Yu et al., 2025). Training data construction and other implementation details are provided in Appendix B.

- 4.2 Experiment Results

TRIT substantially improves multilingual reasoning performance across all models. As

shown in Table 1, TRIT consistently outperforms all baselines across models with varying multilingual capabilities, from the weaker DeepSeekDistill-Qwen-1.5B to the stronger Qwen3 family. On average across three backbones, TRIT improves over SLC-RL by more than 7 percentage points, with the largest gain on DeepSeek-DistillQwen-1.5B (from 24.1% to 33.5%). On the Qwen3 models, TRIT outperforms M-Thinker by approximately 5 percentage points on average. Language consistency reaches nearly 100% across all settings.

TRIT also improves out-of-domain English performance. On Qwen3-1.7B, English accuracy increases from 41.7% to 53.3%, approaching Naive RL (54.5%), which explicitly optimizes for accuracy without language constraints. This suggests that training the model to understand questions consistently across languages improves its fundamental question-comprehension ability, leading to better reasoning even in English.

Notably, M-Thinker yields only limited improvements on the Qwen3 models, showing only marginal gains over SLC-RL. We attribute this to reward saturation: when baseline CTA is already high (e.g., 93% on Qwen3-1.7B), the CTA

reward provides limited discriminative signal for further optimization. In contrast, TRIT optimizes at the question level through translation-reasoning integration, providing an additional optimization dimension that remains effective even on wellaligned models. Detailed analysis is provided in Appendix D.

TRIT also outperforms External-Translation. While external translations provide high-quality target-language questions, they do not teach the model to align its internal understanding across languages. In contrast, TRIT trains the model to generate translations itself, forcing it to learn consistent question representations across languages. This question-level alignment means the model interprets semantically equivalent questions similarly regardless of language, leading to more robust and consistent reasoning. Our MEXA analysis (Section 5.2) confirms this: TRIT improves crosslingual question alignment by over 10 percentage points at best compared to External-Translation.

TRIT supports iterative training for continual improvement. To compare with M-Thinker’s iterative approach, we run one additional RL iteration on DeepSeek-Distill-Qwen-1.5B, improving overall performance from 33.5% to 40.2%. Importantly, low-resource languages continue to improve substantially—Japanese, Korean, and Thai gain over 7 percentage points on average— demonstrating that TRIT can bootstrap multilingual capabilities even in low-resource settings. This sustained improvement reveals TRIT’s potential for scaling to truly resource-scarce languages where traditional supervised approaches struggle due to limited training data.

### 5 Analysis

- 5.1 Self-Improvement of Translation and Generalization

A key aspect of our approach is to use reasoning accuracy as a proxy signal for translation quality. As validated in Appendix C, reasoning accuracy positively correlates with translation quality, making it a reliable proxy signal. To verify whether TRIT improves translation ability, we conduct evaluations both in-domain (MATH500) and out-of-domain (FLORES-200).

In-domain translation quality. To assess translation quality improvements, we compare translations from backbone and TRIT-trained models on

MATH500 using DeepSeek-V3.2-Exp as a judge. As shown in Figure 2(a), TRIT-trained models produce preferred translations across all backbones. The improvements are particularly pronounced for models with weaker initial capabilities: Qwen31.7B achieves a 3.3:1 win-to-loss ratio (51% win vs 16% loss), while DeepSeek-Distill-Qwen-1.5B shows a 2.2:1 ratio. For Qwen3-4B, which already possesses strong translation capabilities, the improvements are more modest (40% win vs 21% loss), suggesting that reasoning-based feedback is most effective when baseline translation quality leaves more room for improvement.

These results confirm that using reasoning accuracy as a proxy signal effectively improves question translation quality. The pattern of stronger gains for weaker models aligns with our expectation: when baseline translation is already high-quality, the reasoning feedback provides less discriminative signal for further optimization. Even strong models benefit from the translation-reasoning integration, demonstrating the robustness of our approach.

Out-of-domain generalization. To examine whether translation improvements generalize beyond mathematics, we evaluate both backbone and TRIT-trained models on the complete FLORES200 benchmark(Team et al., 2022) using COMET as the metric. Figure 2(b) shows that TRIT’s translation improvements transfer substantially to general-domain text. DeepSeek-Distill-Qwen-1.5B, with the weakest baseline translation capability, achieves the largest gain of 8.4 COMET points. Qwen3-1.7B and Qwen3-4B, which already possess stronger translation abilities, improve by 2.2 and 1.5 COMET points respectively.

Notably, these improvements emerge despite TRIT being trained exclusively on mathematical questions, demonstrating that reasoning-based feedback develops translation skills generalizing beyond the mathematical domain. Consistent gains across in-domain and out-of-domain evaluations confirm the applicability of our approach.

#### 5.2 Multilingual Question Alignment

A core contribution of our method is training question translation to induce question-level crosslingual alignment. To verify whether TRIT improves alignment, we use MEXA (Kargaran et al., 2025), which measures cosine similarity between hidden representations of English and targetlanguage question pairs across model layers.

###### Win Tie Lose Base Ours

(a) In-Domain Evolution on MATH500

(b) Cross-Domain Generalization on Flores200

100

| |23.7<br><br>15.7 21.1| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| |52.4 51.2<br><br>40.2| | | |
| | | | | |
| | | | | |

| |88.7<br><br>+1.5| | | |
|---|---|---|---|---|
| |83.2<br><br>+2.2 85.5 87.2| | | |
| | | | | |
| | | | | |
| |51.7| | | |
| |43.4<br><br>+8.4| | | |
| | | | | |

90

80

Percentage(%)

80

60

COMET

70

40

60

20

50

0

40

DeepSeek-Distill-1.5B Qwen3-1.7B Qwen3-4B

DeepSeek-Distill-1.5B Qwen3-1.7B Qwen3-4B

- Figure 2: Evolution of translation quality. (a) In-domain evaluation on MATH500 (Win/Tie/Lose rates vs. Base). (b) Cross-domain generalization on Flores200 (Comet Scores).

0 5 10 15 20 25

Layer

0.65

0.70

0.75

0.80

0.85

0.90

0.95

Similarity

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

DeepSeek-Distill-Qwen-1.5B

TRIT

ET

- Figure 3: Cross-lingual question alignment across model layers (DeepSeek-Distill-Qwen-1.5B). Layerwise cosine similarity between English and targetlanguage question representations for TRIT and External-Translation (ET, without translation training).

Method Constrained Flexible

SLC-RL 36.7 48.0 TRIT 44.5 52.1

Table 2: Performance comparison (LC&Acc, %) between constrained (reasoning in question language) and flexible (reasoning in any language) settings. Experiments conducted on Qwen3-1.7B.

level alignment contributes to better multilingual reasoning performance.

#### 5.3 Evaluation on Flexible Reasoning Setting

We further investigate TRIT’s effectiveness in a more flexible setting: models can reason in any language but must provide final answers in the target language. This setting relaxes the reasoning language constraint, allowing models to choose the reasoning language based on their capabilities.

We sample 100 question pairs from MMATH and compute layer-wise similarity for both TRIT and External-Translation. As shown in Figure 3, TRIT achieves substantially higher alignment across layers, with improvements particularly pronounced in later layers. For example, DeepSeekDistill-Qwen-1.5B’s final-layer similarity increases from 62.7% to 78.6% (15.9 percentage points). Qwen3-4B shows a similar pattern (Figure 7).

As shown in Table 2, TRIT achieves 52.1% in the flexible setting, a 4.1 percentage point improvement over SLC-RL’s 48.0%. Notably, while the improvement margin is smaller compared to the constrained setting (36.7% vs 44.5%, an improvement of 7.8 percentage points), it remains substantial. This result demonstrates that TRIT enhances multilingual question understanding through translation training, and this improvement does not depend on specific reasoning language constraints. In other words, even when models can freely choose their reasoning language, TRIT-trained models exhibit significantly improved comprehension of multilingual questions, enabling consistent performance gains under different constraint conditions.

These results demonstrate that translation training drives question-level alignment. ExternalTranslation uses high-quality translations but does not train the model to generate them, leaving the model without aligned cross-lingual question representations. In contrast, TRIT’s translation training teaches the model to preserve semantics across languages, inducing aligned representations. This increased alignment coincides with the reasoning improvements in Table 1, suggesting that question-

###### 44.5

- 40

- 41

- 42

- 43

- 44

- 45

42.9

###### LC&Acc(%)

41.7

41.6

40.2

0 1/6 1/3 1/2 2/3

Stage-1 Threshold (θ)

- Figure 4: Impact of Stage 1 Filtering Threshold (θ) on Final Multilingual Reasoning Performance

#### 5.4 Sensitivity Analysis of Filtering Thresholds

In the cross-lingual reasoning stage, we filter questions based on their average reward rfinal across sampled responses. A question is retained for subsequent training only if rfinal ≥ θ. This filtering mechanism aims to reduce noise in translation evaluation: when θ is too low, the model may fail at reasoning due to limited capability, causing highquality translations to be incorrectly penalized for reasoning failures rather than translation errors. Conversely, when θ is too high, fewer samples pass the filter, and the retained questions tend to be easier (i.e., those the model can solve more reliably), potentially reducing training signal diversity.

To determine the optimal threshold, we evaluate five candidates on Qwen3-1.7B as a representative model: θ ∈ {0,1/6,1/3,1/2,2/3}. As shown in Figure 4, performance increases from 41.6% to 44.5% as θ rises from 0 to 1/3, but drops to 40.2% at θ = 2/3. To understand this pattern, we conduct a noise analysis (detailed in Appendix E). We use DeepSeek-V3.2-Exp to evaluate translation quality and measure the false negative rate—the proportion of high-quality translations incorrectly assigned low rewards due to reasoning failures. When θ increases from 0 to 1/3, the false negative rate decreases sharply from 38.8% to 7.5%. However, further increasing θ to 1/2 yields only marginal improvement (7.5% to 5.8%) while substantially reducing the number of training samples. Based on this analysis, we set θ = 1/3 for all experiments, which achieves the best performance while maintaining low noise and sufficient training data.

Method LC&Acc(%) TRIT(Full) 44.5 Ablation: removing training data

w/o Cross-lingual Reasoning Data 37.4 w/o Self-translation Data 41.8 w/o Target-language Reasoning Data 36.3

Ablation: design choice English Reasoning for Filtering 42.1

Table 3: Ablation study. Removing training data types (upper) and filtering strategy comparison (lower).

#### 5.5 Ablation Study

To assess the contribution of each training data type, we conduct ablation experiments where we retain the full training pipeline but exclude specific data types from parameter updates: (1) crosslingual reasoning data, (2) translation data, (3) target-language reasoning data. In addition, we evaluate a key design variant: (4) using Englishonly filtering instead of cross-lingual filtering. Results are shown in Table 3.

Necessity of core reasoning types. Removing either cross-lingual or target-language reasoning data degrades performance: from 44.5% to 37.4% and 36.3% respectively. The large drop when removing target-language reasoning data reflects a distribution shift: the model is trained primarily on cross-lingual reasoning (English questions → target-language responses) and translation, but evaluated on target-language-only reasoning (targetlanguage questions → target-language responses). Without explicit training on this distribution, the model struggles to transfer its capabilities effectively.

Removing cross-lingual reasoning data also causes substantial degradation. Without this component, the model’s cross-lingual reasoning capability develops more slowly, resulting in fewer questions passing the accuracy-based filter and reducing the available training data for translation and targetlanguage reasoning.

Role of self-translation training. Removing self-translation data reduces accuracy by 2.7 percentage points (44.5% → 41.8%). While more modest than removing reasoning data, this degradation demonstrates the importance of translation training for question-level alignment. As shown in Figure 3 (Section 5.2), translation training substantially improves cross-lingual question alignment, helping the model develop unified semantic representations

across languages. Without this alignment, multilingual reasoning performance suffers.

Cross-lingual vs. English-only filtering. A key design choice in our framework is using crosslingual reasoning (rather than English-only reasoning) to filter questions before translation training. To validate this design, we compare against an intuitive alternative: filtering based on whether the model can solve questions correctly in English. Results show that English-only filtering reduces performance to 42.1%, a 2.4 percentage point drop from our approach.

This degradation stems from increased noise in translation feedback. English-only filtering assumes that if a model can solve a question in English, it can also solve it in the target language, but this assumption often fails. The model may lack sufficient target-language reasoning capability even with a perfect translation, leading to reasoning failures that are incorrectly attributed to translation quality. As detailed in Appendix E, English-only filtering increases the false negative rate from 7.5% to 13.8%. This noisier feedback signal weakens translation policy optimization and degrades overall performance.

### 6 Conclusion

We propose TRIT, a self-improving framework that integrates translation training with multilingual reasoning through reinforcement learning. Without external feedback or additional multilingual data, TRIT creates a closed loop where translation and reasoning mutually improve each other. Experiments show that TRIT significantly enhances multilingual reasoning performance while maintaining high language consistency. The translation improvements extend beyond the in domain to general-domain text. Critically, integrating translation training substantially improves cross-lingual question alignment. By jointly optimizing translation and reasoning, TRIT improves both multilingual question understanding and reasoning capabilities, offering a promising direction for building more capable multilingual reasoning systems.

### Limitations

While our work demonstrates the effectiveness of TRIT for improving multilingual reasoning, several limitations remain. First, our experiments are conducted on five target languages, which do not fully cover the diversity of multilingual settings. TRIT

does not rely on annotated multilingual data and uses only English questions as the training source, making the framework straightforward to extend to additional languages without modifying the core pipeline. Second, due to computational constraints, we evaluate our method on models up to 4B parameters. While larger models are not explored in this work, we expect TRIT to remain effective at larger scales, as the translation–reasoning integration is model-agnostic, aiming to improve multilingual question alignment.

### Acknowledgments

We would like to thank the anonymous reviewers for their insightful comments. Shujian Huang and Xin Huang are the co-corresponding authors. This work is supported by National Science Foundation of China (No. 62376116), research project of Nanjing University-China Mobile Joint Institute (NJ20250038), the Fundamental Research Funds for the Central Universities (No. 2024300507).

### References

Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Dongmei Zhang, and Jia Li. 2024. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. Preprint, arXiv:2310.20246.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, and 181 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Jaedong Hwang, Kumar Tanmay, Seok-Jin Lee, Ayush Agrawal, Hamid Palangi, Kumar Ayush, Ila Fiete, and Paul Pu Liang. 2025. Learn globally, speak locally: Bridging the gaps in multilingual reasoning. Preprint, arXiv:2507.05418.

Deokhyung Kang, Seonjeong Hwang, Daehui Kim, Hyounghun Kim, and Gary Geunbae Lee. 2026. Why do multilingual reasoning gaps emerge in reasoning language models? Preprint, arXiv:2510.27269.

Amir Hossein Kargaran, Ali Modarressi, Nafiseh Nikeghbal, Jana Diesner, François Yvon, and Hinrich Schütze. 2025. Mexa: Multilingual evaluation of english-centric llms via cross-lingual alignment. Preprint, arXiv:2410.05873.

Hyunwoo Ko, Guijin Son, and Dasol Choi. 2025. Understand, solve and translate: Bridging the multilingual mathematical reasoning gap. Preprint, arXiv:2501.02448.

Mistral-AI, :, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, and 82 others. 2025. Magistral. Preprint, arXiv:2506.10910.

OpenAI, :, Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, Jerry Tworek, Lorenz Kuhn, Lukasz Kaiser, Mark Chen, Max Schwarzer, Mostafa Rohaninejad, Nat McAleese, and 7 others. 2025. Competitive programming with large reasoning models. Preprint, arXiv:2502.06807.

Cheonbok Park, Jeonghoon Kim, Joosung Lee, Sanghwan Bae, Jaegul Choo, and Kang Min Yoo. 2025. Cross-lingual collapse: How language-centric foundation models shape reasoning in large language models. Preprint, arXiv:2506.05850.

Jirui Qi, Shan Chen, Zidi Xiong, Raquel Fernández, Danielle Bitterman, and Arianna Bisazza. 2025. When models reason in your language: Controlling thinking language comes at the cost of accuracy. In Findings of the Association for Computational Linguistics: EMNLP 2025, page 20279–20296. Association for Computational Linguistics.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Shuaijie She, Wei Zou, Shujian Huang, Wenhao Zhu, Xiang Liu, Xiang Geng, and Jiajun Chen. 2024. Mapo: Advancing multilingual reasoning through multilingual alignment-as-preference optimization. Preprint, arXiv:2401.06838.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, and 20 others. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672.

Yiming Wang, Pei Zhang, Jialong Tang, Haoran Wei, Baosong Yang, Rui Wang, Chenshu Sun, Feitong Sun, Jiran Zhang, Junxuan Wu, Qiqian Cang, Yichang Zhang, Fei Huang, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Polymath: Evaluating mathematical reasoning in multilingual contexts. Preprint, arXiv:2504.18428.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao

Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Xue Zhang, Yunlong Liang, Fandong Meng, Songming Zhang, Kaiyu Huang, Yufeng Chen, Jinan Xu, and Jie Zhou. 2025. Think natively: Unlocking multilingual reasoning with consistency-enhanced reinforcement learning. Preprint, arXiv:2510.07300.

Wenhao Zhu, Shujian Huang, Fei Yuan, Shuaijie She, Jiajun Chen, and Alexandra Birch. 2024. Question translation training for better multilingual reasoning. Preprint, arXiv:2401.07817.

### A Model Repetition Analysis

We observe a pervasive issue of degenerate repetition when guiding the model to reason in the target language, which substantially undermines the readability and practical usability of the generated outputs. Notably, such repetition is not necessarily tied to incorrect answers: even when the model reaches the correct answer, repeated segments in the reasoning trace can still make the output difficult to understand. Even worse, we find that repetition can escalate over iterative training if no targeted suppression mechanism is applied, leading to a pronounced degradation in response quality (Figure 10 provides a representative example). This observation motivates incorporating an explicit repetition penalty (Rrep) into our reward function, so that repetition is discouraged during training and response quality is improved.

We design a repetition detection scheme that combines (n)-gram statistics with line-level matching, and use it both for reward computation during training and for quality evaluation at test time. Concretely, we apply two criteria:

- 1. n-gram–based detection We tokenize the text and enumerate n-grams with n = 20, counting their occurrences. If any n-gram appears at least 20 times, we further verify the presence of contiguous repeated spans using a suffixarray construction and the longest common prefix (LCP) algorithm.
- 2. Line-level detection We split the text into lines. If any line contains at least 20 tokens

Model Correct Answer Rep. (%) M-Thinker

- M-Thinker-Iter1 3.3
- M-Thinker-Iter2 43.3 TRIT(Ours)

- TRIT-Iter1 3.6
- TRIT-Iter2 1.4

- Table 4: Repetition rate among correct answers during iterative training on the MMATH Japanese subset. This metric reflects output quality when the model produces correct results.

and occurs at least 6 times, we flag the output as exhibiting line-level repetition.

- A response is marked as repetitive if it satisfies either criterion. By setting appropriate thresholds for repetition frequency (≥6) and minimum span length (≥20 tokens), this method effectively identifies degenerate repetition while reducing false positives from legitimate linguistic patterns.

To validate the effectiveness of the repetition penalty, we compare M-Thinker and TRIT on the Japanese subset of MMATH and track how repetition escalates under iterative training. We focus on repetition among correct answers, as this metric better reflects output quality when correctness is already satisfied.

As shown in Table 4, M-Thinker exhibits severe quality degradation across iterations: from Iter1 to Iter2, the repetition rate among correct answers spikes from 3.3% to 43.3%. This indicates that, without explicit quality constraints, iterative training can substantially exacerbate degenerate repetition—even when the model produces the correct answer, the reasoning trace becomes dominated by repeated content, severely harming readability and utility.

In contrast, TRIT, which incorporates the repetition penalty, maintains high-quality generations throughout iteration. The repetition rate among correct answers decreases from 3.6% at Iter1 to 1.4% at Iter2, suggesting that the repetition penalty continues to provide effective regularization during iterative training, preserving response quality while improving accuracy.

- B Implementation Details B.1 Data Construction

We construct training data from DAPO-MATH17K (Yu et al., 2025) for five target languages

(French, Portuguese, Japanese, Korean, Thai). Important: The external translations mentioned below are used only for constructing baseline datasets and evaluation benchmarks, not for TRIT training itself.

To enable multilingual training, we translate the original English questions into target languages using DeepSeek-V3.2-Exp and verify translation quality with Qwen3-32B. We prepare three datasets:

- • Cold-start dataset: Generated by Qwen3-8B through cross-lingual reasoning (Questionen

→ Responsetgt), used for warm-up training to establish the cross-lingual reasoning pattern. No translation is used—the model directly answers English questions in the target language. We retain 3,000 samples per language after filtering for language consistency and correctness.

- • SFT dataset: Generated by Qwen3-32B for the supervised fine-tuning baseline

(Questiontgt → Responsetgt). We retain 3,000 validated samples per language after the same filtering process.

- • RL dataset: For reinforcement learning training, we collect 3,000 English questions per language: 2,000 questions with baseline accuracy below 0.5 (challenging but solvable) and 1,000 randomly sampled questions with zero accuracy (harder cases). This mixture ensures diverse difficulty levels for effective RL training.

Summary: TRIT requires only English questions and learns to translate during training. External translations are used solely for baseline construction and evaluation.

#### B.2 Training Configuration

We train all models using the AdamW optimizer. Cold-start stage: we use a batch size of 64 with a learning rate of 1×10−5 for 2 epochs, together with linear warmup and a cosine learning-rate schedule. Reinforcement learning stage: we use a global batch size of 512, a mini-batch size of 64, and a learning rate of 1 × 10−6, with a KL-divergence penalty coefficient β = 0.001. To balance sampling efficiency and training stability, we sample 6 responses per question for both cross-lingual reasoning and target-language reasoning, and 4 translation candidates for the translation task. The maxi-

mum sequence length is 8,192 tokens for all experiments.

### C Alignment Analysis of Translation Quality and Reasoning Accuracy

Win

| |
|---|

(a) Moderate Differences ( Acc > 0.2)

30.4% 5.3%

64.3%

Lose

| |
|---|

| |
|---|

Tie

(b) Critical Failures (Acc > 0 vs. Acc = 0)

16.1% 7.9%

75.9%

Figure 5: Translation quality correlates with reasoning accuracy. Distribution of translation quality (Win/Lose/Tie judged by DeepSeek-V3.2) for question pairs with (a) moderate accuracy differences (∆Acc >

- 0.2) and (b) critical failures (Acc = 0 vs. Acc > 0). Better translations consistently correspond to higher reasoning accuracy.

To examine how translation quality affects mathematical reasoning, we translated MATH500 questions into multiple versions and analyzed their impact on model performance. We first considered samples where reasoning accuracy differed by more than 0.2 across translations, ensuring that the lower-accuracy version still yielded at least one correct answer. As shown in Figure 5(a), even when the model has basic problem-solving ability, highaccuracy translations achieve a higher quality win rate (64%) than low-accuracy ones (30%), indicating that translation quality can influence reasoning stability.

We also analyzed extreme cases where one translation yields 0 accuracy while another yields non-zero accuracy. Figure 5(b) shows that highaccuracy translations achieve a win rate of 76% compared to 16% for low-accuracy translations, highlighting that precise translation of key information is critical for enabling successful reasoning.

- Figure 6 further illustrates a representative ex-

ample: the original English question specifies a parallelogram; the high-accuracy translation preserves this detail, while the low-accuracy translation weakens it to a “quadrilateral,” resulting in information loss and reduced answer accuracy.

### D Why M-Thinker Failed

In experiments, we observe that M-Thinker does not yield consistent performance gains on the

Source Question (English):

The coordinates of a parallelogram are (5, 3), (6, 8), (7, 4) and (x, y) and x > 7. What is the value of x + y?

###### Imprecise Translation (Failure):

“사각형(Quadrilateral)의 좌표는 (5, 3), (6, 8), (7, 4)와 (x, y)이며, x > 7입니다. x + y의 값을 구 하세요.”

###### Precise Translation (Success):

“평행사변형(Parallelogram)의 좌표는 (5, 3), (6, 8), (7, 4)와 (x, y)이며, x > 7의 조건을 만족합니다. x + y의 값을 구하세요.”

Figure 6: Case study on semantic precision in translation. The imprecise translation generalizes the specific term Parallelogram into a generic Quadrilateral, resulting in the loss of parallel constraints. In contrast, the precise translation preserves the exact geometric definition, enabling the correct solution.

Qwen3 family. To better understand this phenomenon, we analyze the issue from the perspective of the model’s initial cross-lingual thinking alignment.

We evaluate cross-lingual reasoning-trace consistency on MMATH using the CTA score for models trained with different methods. Concretely, we randomly sample English questions and retain those for which the model produces at least one correct answer in English, together with their corresponding multilingual responses. We then use the evaluation prompt provided by M-Thinker and compute a consistency score between the multilingual reasoning trace and the English reasoning trace using an external judge, DeepSeek-V3.2-Exp.

As shown in Table 5, the baseline CTA score of Qwen3-1.7B is already 0.93, indicating that its cross-lingual reasoning consistency is high at initialization. After M-Thinker training, the CTA score slightly decreases to 0.923, whereas TRIT increases it to 0.947. This comparison highlights a key difference between the two approaches. MThinker explicitly optimizes cross-lingual chain-ofthought consistency via a CTA reward; however, when the baseline consistency is already around 0.93, the CTA reward is near-saturated for most samples and varies only minimally, making the reward signal poorly discriminative and providing little guidance for further optimization. In contrast, TRIT optimizes at the level of question understanding: translation training encourages the model to

###### Method CTA Score

Baseline (Qwen3-1.7B) 0.930 M-Thinker 0.923 TRIT (Ours) 0.947

- Table 5: Cross-lingual thinking alignment (CTA) analysis. We measure CTA scores on MMATH using DeepSeek-V3.2-Exp as the judge. The baseline Qwen3-

- 1.7B already exhibits high CTA (0.93), leaving little room for M-Thinker’s trace-alignment optimization. TRIT improves CTA through question-level alignment, demonstrating an alternative optimization pathway.

Setting False-negative Rate (%) Before training TRIT(θ = 1/3) 7.5 θ = 1/6 11.8 θ = 1/2 5.8 θ = 0 (no filtering) 38.8 English reasoning 13.5 After training Ours (θ = 1/3) 3.6

- Table 6: False-negative rates of semantically correct translations under different cross-lingual filtering configurations. A false negative refers to a correct translation incorrectly penalized due to target-language reasoning failure.

align its understanding of target-language questions with their English counterparts. As question representations become more aligned across languages, the resulting reasoning processes also become more consistent, allowing TRIT to improve CTA without directly optimizing the reasoning-trace alignment objective.

Overall, these results suggest that M-Thinker’s explicit trace-alignment strategy can suffer from reward saturation when starting from a highly aligned backbone, whereas TRIT introduces an additional optimization dimension through questionlevel alignment and continues to improve multilingual reasoning even when baseline cross-lingual consistency is already high.

### E Noise Analysis of Deferred Reasoning Feedback

One of the core design choices in our framework is to use target-language reasoning accuracy as a delayed supervisory signal for evaluating the quality of self-generated translations. While well motivated in principle, this mechanism can introduce false-negative noise when reasoning failures are mistakenly attributed to translation errors, causing

the model to penalize semantically faithful translations. In this section, we quantify the magnitude of this false-negative risk and analyze how crosslingual filtering effectively controls it.

We compare false-negative rates across different cross-lingual filtering thresholds ((0), (1/6), (1/3), and (1/2)), as well as a variant that replaces targetlanguage filtering with English-only reasoningbased filtering.

Before training, under the default setting (θ = 1/3), the false-negative rate is 7.5%, indicating that although target-language reasoning accuracy is not a perfect indicator, it can still serve as a reasonably reliable proxy for translation quality. In contrast, removing cross-lingual filtering (θ = 0) causes the false-negative rate to surge to 38.8%, suggesting that without filtering the causal linkage between translation quality and downstream reasoning accuracy is severely compromised. Introducing filtering markedly reduces false negatives: the rate drops to 11.8% at θ = 1/6 and further to 7.5% at θ = 1/3, confirming the necessity of cross-lingual filtering.

Replacing target-language filtering with Englishonly reasoning increases the false-negative rate to 13.8%. This is because solving a question in English does not guarantee that the model can solve the same question in the target language; such capability mismatch weakens the filter and admits more cases where reasoning failures are incorrectly attributed to translation errors. Increasing the threshold to θ = 1/2 reduces the false-negative rate to

5.8%, but the gain over θ = 1/3 (7.5%) is modestonly 1.7 percentage points. Together with the overall performance drop at θ = 1/2 in Figure 4, these results suggest that θ = 1/3 offers the best tradeoff between controlling false-negative noise and retaining sufficient training samples.

More importantly, after TRIT training, the falsenegative rate under the default setting drops from 7.5% to 3.6%. We attribute this improvement primarily to stronger target-language reasoning, which allows the model to solve more questions when the translation is semantically faithful and thus reduces cases where reasoning failures are mistakenly attributed to translation errors. This indicates that the integrating training mechanism can progressively mitigate false-negative noise over time, creating a positive feedback loop.

### F Additional Figures

###### Qwen3-4B

0.9

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

| |
|---|

| |
|---|

| |
|---|

0.8

| |
|---|

| |
|---|

Similarity

0.7

0.6

0.5

TRIT

| |
|---|

ET

0.4

0 5 10 15 20 25 30 35

Layer

- Figure 7: Cross-lingual question alignment for Qwen3-4B. Similar to DeepSeek-Distill-Qwen-1.5B (Figure 3), TRIT achieves higher alignment than External-Translation (ET), particularly in later layers.

[Figure 1]

##### Figure 8: Multilingual reasoning instructions. We use language-specific prompts to instruct the model to reason step-by-step in the question language and place the final answer within \\boxed{}. All prompts are semantically equivalent translations requesting step-by-step reasoning and formatted output.

[Figure 2]

##### Figure 9: Two language control strategies. Left: Language prefixes (e.g., <think>\nOkay) prepended to the input to guide the model to respond in the corresponding language. We use it in data construction. Right: Explicit language instruction prompts that directly instruct the model to think and answer in the target language. We use it in Prompt Control baseline.

Source Question: 求列 (an) の a99 の値： ai+1 = 1−1a

, i ≥ 1, 件 a3 = a1

i

###### Answer (Full, with Repetition):

1 1 − ai

, i ≥ 1

ai+1 =

です。件で a3 = a1 ですですですですですですですですですですですですですですですですですですですで すですですですですですですですですですです a99 を求めます。

まず、列の定義を分析します：

1 1 − ai

ai+1 =

です。件 a3 = a1 ですですですですですですですですですですですですですですですですですですですです ですですですですですですですですです a1 です ですですですですですですですですですですですですで すですですですですですですですですですですですです。

, a2 = 1−1a

a3 = 1−1a

です ですですですですですですですですですですですですですですですですですで すですですです。

2

1

a3 = a1 =⇒ α2 − α + 1 = 0 です ですですですですですですですですですですですですですですです。 判別式: D = (−1)2 − 4 · 1 · 1 = −3 です ですですですですですですですですです。 解:

1 ± i√3 2

α =

列の周期: a1 = α, a2 = 1−1α, a3 = 1−1a

です ですですですですですですですですです。

2

周期3: a1 = α, a2 = α, a3 = α a9 = a3 = α a99 = (α3)3 = (−1)3 = −1 した、a99 =

|−1|
|---|

です ですですですですですですですですですですです。

- Figure 10: Case study on excessive repetition in reasoning. The answer is mathematically correct, but intermediate steps contain massive repeated words (ですですです...), which heavily reduces readability.

You are a professional translator. Task: Translate the given English question into {language} with zero loss of mathematical or logical meaning. Output format (MUST keep tags): <TRANSLATION> {language} text only </TRANSLATION> Rules:

- 1. Re-state any numbers, symbols, units exactly as they appear.
- 2. Keep LaTeX ($...$) unchanged; do not translate inside $...$.
- 3. If the question contains multiple-choice items (A), (B), ... keep the same labels.
- 4. Use natural {language} wording, but stay one-to-one faithful to the original semantics.
- 5. Reply with ONLY the tagged translation—no explanations. <ENGLISH QUESTION> {question} </ENGLISH QUESTION> Please translate the above English question into {language}.

- Figure 11: Translation prompt template used in TRIT to generate semantically faithful translations while preserving mathematical notation and formatting.

