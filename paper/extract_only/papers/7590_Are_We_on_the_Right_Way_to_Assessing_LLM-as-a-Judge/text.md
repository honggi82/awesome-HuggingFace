## arXiv:2512.16041v1[cs.CL]17Dec2025

# Are We on the Right Way to Assessing LLM-as-a-Judge?

Yuanning Feng∗1, Sinan Wang∗1, Zhengxiang Cheng1, Yao Wan†1, and Dongping Chen‡2 1Huazhong University of Science and Technology, 2University of Maryland

LLM-as-a-Judge has been widely adopted as an evaluation method and served as supervised rewards in model training. However, existing benchmarks for LLM-as-a-Judge are mainly relying on human-annotated ground truth, which introduces human bias that undermines the assessment of reliability and imposes scalability constraints. To overcome these limitations, we introduce Sage, a novel evaluation suite that assesses the quality of LLM judges without necessitating any human annotation. Inspired by axioms of rational choice theory, Sage introduces two new lenses for measuring LLM-as-a-Judge: local self-consistency (pair-wise preference stability) and global logical consistency (transitivity across a full set of preferences). We curate a dataset of 650 questions by combining structured benchmark problems with real-world user queries. Our experiments demonstrate both the stability of our metrics and their high correlation with supervised benchmarks like LLMBar and RewardBench2, confirming Sage’s reliability as an evaluation suite for the robustness and accuracy of LLM-as-a-Judge. Based on Sage, we reveal that current state-of-the-art LLMs exhibit significant reliability problems when acting as judges in both scoring and pairwise settings; even the top-performing models, Gemini-2.5-Pro and GPT-5, fail to maintain consistent preferences in nearly a quarter of difficult cases. We attribute this to a new phenomenon called situational preference, which explains why explicit rubrics or criteria can help the model judge consistently across answer pairs. Our further analysis shows that finetuned LLM-as-a-Judge is a feasible method to boost performance, and the panel-based judge as well as deep reasoning can enhance the judging consistency. We also find substantial inconsistency in human judgments, which indicates that human annotation may not be a reliable gold standard. We hope our experiments and findings bring insights to the community towards a more stable and reliable system of generative models as metrics.

https://entroplay.ai/research/rethinking-llm-as-a-judge

### 1. Introduction

The LLM-as-a-Judge paradigm (Zheng et al., 2023) uses a large language model (LLM) to evaluate AI system outputs, offering a scalable and efficient alternative to costly and time-consuming human evaluation. Furthermore, beyond merely assessing performance, these evaluators are instrumental in refining models. During training, an LLM-as-a-Judge acts as a scalable reward model to enhance models’ performance through automated feedback (Bai et al., 2022, Ouyang et al., 2022, Yuan et al., 2024, Luo et al., 2024). At inference time, it also serves as a test-time scaling module for real-time filtering to select the best response (Lightman

- et al., 2023, Faria and Smith, 2025).

However, the LLM-as-a-Judge paradigm is undermined by inherent flaws. Judge models are susceptible to biases such as positional (Shi et al., 2024), verbosity (Saito et al., 2023), and self-enhancement (Wataoka

- et al., 2024), which skew evaluation results and call the paradigm’s reliability into question. In response, various benchmarks have been developed to assess the judges themselves (Zheng et al., 2023, Chiang et al., 2023, Gera et al., 2025, Pu et al., 2025). Yet, the construction of these benchmarks universally relies on human-annotated ground truth, suffering from the bias and inconsistency due to human data, particularly on subjective questions, which leads to two problems:

• First, the acquisition of human annotations is a notoriously expensive and labor-intensive process, limiting the scale and diversity of datasets (Horych et al., 2024, Liao et al., 2025).

∗Equal contribution. ‡Dongping Chen is the project leader. Corresponding author(s): Yao Wan: wanyao@hust.edu.cn, Dongping Chen: dongping@umd.edu

###### What is the difference between the House of Lords and the House of Commons in the UK?

###### Lengthy and Intricate Answer

###### Flawed Answer

###### Answer 1 Answer 2

… 6573 words omitted … Parliamentary procedure dictates that a bill can only travel between the two chambers a maximum of three times in this manner. If

…… omitted for brevity ……

The United Kingdom's Parliament operates with two chambers: the House of Commons and the House of Lords. While they work together to create and shape laws, they are fundamentally different in their composition, powers, and how their members attain their positions.

The United Kingdom's Parliament is a bicameral system, composed of two distinct chambers: the House of Commons and the House of Lords.

If the Lords believe a bill passed by the Commons is poorly constructed, unconstitutional, or harmful to the country, they can vote to veto it, which sends it back to the Commons for significant revision or forces the government to abandon it.

While both play a crucial role in

agreement is not reached after the third exchange, the Bill is failed for that parliamentary session.

the legislative process, they differ significantly in their composition, powers, and functions.

… 3875 words omitted …

…… omitted for brevity ……

…… omitted for brevity ……

…… omitted for brevity ……

| | | |
|---|---|---|
|answer 2| |Annotator2:|

Annotator: Yes, the Lords' veto power. I recall being taught that's their key check on government, much like the US President has. This seems accurate.

Annotator: It's a bit of a wall of text. I can’t pay attention to every detail. Anyway, I think it’s a factual and detailed answer.

I think answer 1 is better due to its comprehensive analysis.

Annotator1: I think is better as it is a concise and succinct answer.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

(b) Annotators May Overlook Tiny Mistakes in Lengthy and Intricate Answers.

(c) Annotators Are Susceptible to Cognitive Biases.

(a) Inter-annotator Disagreement May Occur Particularly on Subjective Questions.

- Figure 1: Human-annotated preference may not be reliable. We find three key challenges with relying on human annotators for evaluating LLM-as-a-Judge systems. (a) Inter-annotator Disagreement: Different annotators can have conflicting preferences, especially for subjective questions, leading to noisy and inconsistent data. (b) Overlooking Nuances: Annotators may miss subtle errors or inaccuracies in lengthy and complex answers, leading to flawed evaluations. (c) Cognitive Biases: Human evaluators are susceptible to cognitive biases, such as favoring an answer that confirms their false beliefs, which can further compromise the objectivity of the assessment.

• Second, and more fundamentally, assuming human judgment as a gold standard is precarious, a “bitter lesson” where human-induced biases compromise AI evaluation (Sutton, 2019). As illustrated in Figure 1, this reliance is problematic. Persistent inter-annotator disagreement creates noisy data (Zhang et al., 2024), demonstrated by low agreement shown in AlpacaFarm (66%, Dubois et al. (2023)) and MT-Bench (63%, Zheng et al. (2023)). This problem is compounded when lengthy answers tax human cognitive capacity. Furthermore, human evaluators are susceptible to cognitive biases (Zheng et al., 2023, Chen et al., 2024a, Wu and Aji, 2025), favoring answers that match their false beliefs, making human annotations an unreliable foundation.

To address this challenge, we introduce Sage (Self-Assessing Gauge for Evaluators), a novel evaluation suite for assessing LLM-as-a-Judge robustness without any human annotation. Our approach is grounded in fundamental principles of rational decision-making, which posit that a reliable judge must exhibit consistent and coherent preferences. For example, a robust judge’s preference between two answers should not flip simply because their presentation order is swapped. Furthermore, its judgments should adhere to the principle of transitivity, maintaining a logical and consistent order across a full set of preferences (Ouyang et al., 2022, Song et al., 2024, Hou et al., 2024, Hu et al., 2024, Liu et al., 2024). A breakdown in this coherence suggests the model lacks a consistent internal gauging principle for the question, leading to unreliable situational preferences.

Based on these principles, we propose two metrics to quantify this robustness: Intra-Pair Instability (IPI) and Weak Total Order Violation (TOV). IPI directly measures the local, pairwise consistency by detecting instabilities caused by positional bias, as in the first example. TOV, on the other hand, assesses the global logical coherence of a judge’s complete set of preferences, identifying systemic contradictions like the violation of transitivity described.

For the evaluation, we curate a diverse dataset of 650 questions by combining selections from RewardBench2 (Gureja et al., 2025) and the large-scale WildChat-1m corpus (Zhao et al., 2024) to ensure broad coverage

of real-world user queries. On this dataset, we conduct a comprehensive evaluation of thirteen prominent LLMs. We validate the soundness of our metrics by demonstrating both their intrinsic stability through consistent checking and a distribution-free error bounding method that yields minuscule variance on the order of 10−5, and their external alignment with established supervised benchmarks like LLMBar (Zeng

- et al., 2023) and RewardBench2, confirming Sage ’s reliability as an evaluation suite for LLM-as-a-Judge.

Based on Sage, we evaluate a wide range of LLM-as-a-Judge systems in both scoring and pairwise settings, including state-of-the-art LLMs, fine-tuned judges, and multi-agent as juries. All judge models degrade when encountering candidate answers that are in a closer gap, with approximately 200% more inconsistency, highlighting the potential problem in using LLM-as-a-Judge in RL-based training and test-time scaling. Our findings reveal that current models exhibit significant robustness deficiencies and most specialized fine-tuning brings about improvement. Our findings also show that multi-agent panels can improve performance by up to 15% and that increasing a model’s reasoning depth can only bring about a minor enhancement. Notably, prompting for self-generated rubrics to avoid situational preference yields an even greater performance boost, reducing IPI (local inconsistency) and TOV (global inconsistency) by 16.1% and 11.0%, respectively. In our further analysis, we investigate the alignment between different evaluation formats. We also demonstrate Sage’s practical utility in selecting stable evaluators for automated arenas and its cost advantage over human annotation. Finally, we apply Sage to human evaluators and reveal the fragility of human “gold standards”, where IPI reaches 0.332 and TOV surges to 6.523 on complex tasks. We will release all source code, curated dataset at https://github.com/plafle/LLMasaJudgeAssessment.

### 2. Assessing LLM-as-a-Judge with Sage

This section details the foundational methodology of our proposed framework, Sage. We begin by formally defining the evaluation problem and introducing a symmetrized protocol. Building on this, we then present our two novel metrics: Intra-Pair Instability (IPI) to assess local, pairwise consistency, and Weak Total Order Violation (TOV) to measure global, logical coherence.

#### 2.1. Problem Formulation

Let M be the LLM under evaluation, referred to as the judge model. Our evaluation is based on a set of questions 𝒬. For any given question Q ∈ 𝒬, we generate a set of n candidate answers, denoted as AQ = {A1, A2, . . . , An}. The core task of the judge model M is to perform a pairwise comparison between any two answers, Ai and Aj, from the set AQ. We define a function JM:

yij = JM(Q, Ai, Aj) ∈ {−1,0,1} (1) where the outcome yij is interpreted as:

- • yij = 1: M judges Ai to be superior to Aj (Ai ≻ Aj).
- • yij = −1: M judges Ai to be inferior to Aj (Ai ≺ Aj).
- • yij = 0: M judges Ai and Aj to be of equal quality (Ai = Aj).

For each question Q, we conduct a full round-robin evaluation, assessing all (n2) unique pairs of answers, to establish a complete set of pairwise judgments for our subsequent coherence analysis.

###### Applications

[Figure 5]

###### LLMs

[Figure 6]

###### Question

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Evaluate the accuracy of LLM-as-a-Judge

[Figure 13]

What legacy do you think Voyager 1 will leave for human？

Evaluate the robustness of LLM-as-a-Judge

[Figure 14]

Question

Possible Weak Total Orders

Answer 1

Select robust judges for automated arenas

[Figure 15]

[Figure 16]

- 3 Judges Altered

5 Judges Altered

- 4 Judges Altered

Judge Result

| |[Figure 17]|
|---|---|
| |Judge|

[Figure 18]

| |[Figure 19]|
|---|---|
| |Judge|

=

###### Findings

Answer 2

Poor performance of even SOTA models

[Figure 20]

- Ans 1

- Ans 2

- Ans 3

- Ans 4

……

[Figure 21]

At least alter 3 judges to transform to weak total order

[Figure 22]

Most fine-tuning improve performance

[Figure 23]

Legend

…… 3( Inconsistent, , Pairs)

: :

Reasoning may help boost performance

Human Label

Check Alignment

[Figure 24]

=

Answers

TOV = 3

IPI = 3 / C(4,2) = 0.5

Rubrics help improve performance

[Figure 25]

Our Evaluation Framework: Scalable, Automatic and Free of Human Bias

Past Evaluation Framework

- Figure 2: Sage uses a symmetrized, round-robin protocol to conduct pairwise comparisons on a set of candidate answers. From these judgments, Sage quantifies performance using two metrics: IPI, which measures local consistency by tracking preference reversals (e.g., 3 inconsistent pairs result in an IPI of 0.5), and TOV, which assesses global logical coherence by calculating the minimum alterations required for a consistent ranking (e.g., 3 alternations required). This methodology scalably diagnoses logical deficiencies to help identify and select more reliable LLM evaluators.

#### 2.2. Symmetrized Evaluation Protocol

A naive single-pass evaluation is susceptible to positional bias, where the order of presentation influences the outcome. To substantiate that positional bias does exist in Sage, we sample 1120 answer pairs and measure the inconsistency rate for Llama3-8B-Instruct (Dubey and et al., 2024), Gemini-2.5-Flash-Lite (Comanici and et al., 2025), and Qwen3-4B-Instruct-2507 (Team, 2025).

We define this rate as the frequency of judgments that are not the logical inverse when the answer order is reversed (i.e., JM(Q, Ai, Aj) ≠ −JM(Q, Aj, Ai)). The results in Table 1 confirm the presence of bias. To tackle this issue, we adopt a symmetrized evaluation protocol. For each unordered pair of answers {Ai, Aj}, we query the judge model twice:

Table 1: Local inconsistency (i.e., Positional Bias) across LLM-as-a-Judge.

Model Inconsistency (%)

Llama3-8B-Instruct 76.2 Gemini-2.5-Flash-Lite 25.3 Qwen-3-4B-Instruct-2507 44.4

Forward pass: yij ← JM(Q, Ai, Aj); Reversed pass: yji ← JM(Q, Aj, Ai).

This protocol provides a direct way to measure and account for first-order positional bias.

#### 2.3. Two Evaluation Metrics

We propose two metrics to quantify the robustness of an LLM judge, targeting two distinct failure modes: local inconsistency on a single pair and global logical incoherence across a set of answers.

Intra-Pair Instability (IPI). This metric assesses robustness from an atomic, local level. It quantifies inconsistencies arising from both systematic positional bias and the inherent stochasticity of the judge model. Under the symmetrized protocol, a perfectly consistent judge would always produce opposite scores for reversed pairs (i.e., yij = −yji). The IPI score for a given question Q quantifies the deviation from this ideal

by calculating the average disagreement across all unique pairs:

IPI(Q) =

I(yij ≠ −yji) (2)

1 (n2)

∑

1≤i<j≤n

A higher IPI score indicates a greater degree of local inconsistency of the judge model.

Weak Total Order Violation (TOV). This metric assesses robustness from a global, systematic level. Specifically, it measures the logical coherence of the judge’s full set of preferences for a question. A rational judge’s preferences should be transitive and form a weak total order (i.e., a total order that allows ties). Let JQ = {yij}1≤i,j≤n,i≠j be the set of derived preferences from the symmetrized evaluation for a question Q. Let 𝒪n be the set of all possible valid weak total orders on n items. For any order O ∈ 𝒪n, we can represent it as a corresponding set of pairwise relations PO = {pij}, where pij ∈ {−1,0,1} denotes the pairwise relationship between items i and j with the order O. Specifically, pij = 1 if i is preferred to j, pij = −1 if j is preferred to i, and pij = 0 if they are tied. The TOV score is defined as the minimum number of preference changes required to transform the judge’s observed preferences PQ into any valid weak total order:

TOV(Q) = min

I(yij ≠ pij) (3)

∑

O∈𝒪n

1≤i,j≤n,i≠j

A higher TOV score signifies more severe logical contradictions in the judge’s reasoning.

To summarize a judge model’s overall performance, we compute aggregate scores for both IPI and TOV. The aggregate IPI and TOV scores are the arithmetic mean of the per-question scores over the entire set of questions 𝒬 in Sage, calculated as IPI = (1/∣𝒬∣)∑Q∈𝒬 IPI(Q) and TOV = (1/∣𝒬∣)∑Q∈𝒬 TOV(Q). The stability of these metrics is validated empirically in Section 4 and supported by the theoretical analysis in Appendix A.

### 3. The Construction of Sage

We source the question set 𝒬 from five RewardBench2 (Gureja et al., 2025) categories and the large-scale WildChat-1M corpus (Zhao et al., 2024) to better reflect real-world user interactions. The resulting question set consists of 650 questions, and its category composition is shown in Figure 3a. To validate its semantic diversity, we use a t-SNE visualization (van der Maaten and Hinton, 2008) to project our questions against a background of 500k English questions from WildChat1M. As shown in Figure 3b, our questions spread broadly across the embedding space, confirming the dataset’s representativeness and wide topical coverage. Further details are provided in Appendix B.1.

For each of the 650 questions, we generate a set of n = 6 candidate answers for the LLM judge to evaluate, which were used to construct two distinct tiers: Sage-Easy and Sage-Hard.

• Sage-Easy: the six answers are generated by a diverse lineup of six models with a clear capability gradient: Gemini-2.5-Pro and Gemini-2.5-Flash (Comanici and et al., 2025); Qwen3-32B (Team, 2025), Claude-3-Haiku (Anthropic, 2024), Llama-3.2-3B-Instruct, and Llama-3.2-1B-Instruct (Meta, 2024b). These models, which have a well-documented performance gap on the LMSYS Chatbot Arena leaderboard (LMSYS, 2025), produce a set of answers with a wide variance in quality, making the pairwise comparison

[Figure 26]

[Figure 27]

[Figure 28]

(a) Category Composition (b) Semantic Coverage of Question Set (c) Distribution of CV Values

- Figure 3: We provide statistics and analysis of our selected queries and answers within Sage. Distribution of CV values shows the varied difficulty among our two subsets.

task relatively simple for a competent judge. Crucially, Sage-Easy reflects the general-purpose task of comparing different models of varying capabilities, which is largely used in automated judges like MT-Bench and Arena-Hard-Auto.

• Sage-Hard: all six answers for each question are generated by a single capable model, Gemini-2.5-Flash. Since the answers originate from the same model, their quality is expected to be much more homogeneous. This setup presents a more challenging task, requiring the judge to make finer-grained distinctions between subtly different responses. Crucially, Sage-Hard models the judge’s role in applications like model-based reinforcement learning and rejection sampling. In these scenarios, the judge must distinguish between subtly varied outputs from a single capable model.

To quantitatively confirm the difference in quality diversity between these two tiers, a state-of-the-art reward model, QRM-Gemma-2-27B (Dorka, 2024), is employed to score each of the six answers for every question. For each question, the Coefficient of Variation (CV) of the six reward scores is then calculated. The CV, defined as the ratio of the standard deviation to the mean (σ/µ), is a normalized measure of dispersion. As shown in Figure 3c, the CV distribution for Sage-Hard is markedly shifted towards lower values, empirically confirming that the answers within its sets are more similar in quality and thus present a more formidable challenge for LLM judges. We also conduct a controlled study regarding human cognitive load to verify the difficulty disparity. We recruit 20 graduate-level researchers to perform pairwise comparisons on a sample of 50 questions from our dataset. We record the average time required to complete a single judgment for both subsets. The results reveal a significant disparity in difficulty: annotators required an average of 7.3 minutes per pairwise comparison on Sage-Easy, whereas this duration increased to 10.4 minutes on Sage-Hard. This substantial 42% increase in adjudication time indicates a much higher cognitive load for human experts, and further proves the difficulty disparity between Sage-Easy and Sage-Hard.

### 4. Experiment and Analysis

We first conduct a series of validation experiments to prove the internal consistency and external validity of our metrics in Section 4.1 and 4.2. We then employ Sage to evaluate a diverse set of thirteen popular LLMs-as-a-Judge, six specialized fine-tuned judges, and multi-agent configurations. The results highlight significant robustness challenges in state-of-the-art LLMs, especially on difficult, fine-grained distinction tasks (Figure 9). Our in-depth analysis reveals that fine-tuning can improve robustness and that multi-agent

judges may boost performance. Through further experiment, we attribute the inconsistent judgments to a new phenomenon we discover, situational preference, which can be mitigated by deep reasoning and self-generated rubrics for a more consistent modeling of the question.

#### 4.1. Validating Metric Stability and Robustness

A critical aspect of a reliable framework is the stability of its evaluation metrics against the inherent stochasticity of models. To validate that our proposed metrics are not unduly influenced by random sampling variations, we analyze their stability from both an empirical and a theoretical standpoint. Furthermore, we demonstrate that temperature settings wouldn’t threaten the robustness of Sage.

Theoretical Guarantees. Our argument proceeds in three stages. First, using principles from Conformal Prediction (Angelopoulos and Bates, 2021), we establish a probabilistic guarantee that any single pairwise judgment, ynew, is highly stable and matches its most probable outcome, y∗new, with high confidence:

P(ynew = y∗new) ≥ 1 − α. (4) Second, we use this result to derive an upper bound on the variance of the per-question metrics. For IPI, the score is a fraction of inconsistent pairs out of N = (62) = 15 unique pairs. The deviation from the stable score, ∆IPI(Q), is bounded by the number of unstable judgments X. This allows us to bound the variance as:

Var(IPI(Q)) ≤ E[∆IPI(Q)2] ≤

E[X2] ≤

1 N2

1.683 152

≈ 0.0075. (5)

Finally, we show that this variance diminishes over the aggregate evaluation suite. Assuming the per-question scores are independent and identically distributed over our diverse set of ∣𝒬∣ = 650 questions, the variance of the final aggregate IPI score is given by:

Var(IPI(Q)) ∣𝒬∣

Var(IPI) =

0.0075 650

≈ 1.15 × 10−5. (6)

≤

A similar derivation establishes an upper bound for Var(TOV) which is Var(TOV) ≤ 2.59 × 10−3. These theoretical results align perfectly with our empirical findings, confirming that the final reported scores are highly stable. The full derivation of this analysis is available in Appendix A.

Empirical Analysis. We select two representative models, Qwen34B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507, and evaluate each 10 times. We then calculate the variance of the IPI and TOV scores across these 10 independent runs. As presented in Table 2, the observed variances are exceptionally low, which provides strong empirical evidence that our metrics are highly reproducible and capture the fundamental reasoning patterns of the judge model rather than ephemeral artifacts of its generative process.

Table 2: Variance across 10 independent runs for LLM-as-a-Judge consistency checking.

Model IPI Variance TOV Variance

Qwen3-4B 2.2 × 10−6 6.5 × 10−4 Qwen3-30B-A3B 6.7 × 10−6 1.5 × 10−3

Consistency across Temperatures. We evaluate model performance across various temperature settings. The resulting IPI and TOV scores demonstrate remarkable consistency, indicating that our metrics effectively capture the fundamental reasoning capabilities of the models rather than superficial sampling artifacts.

For all models and metrics tested, the variance is small, which further substantiates the reliability of our framework. The results regarding temperature sensitivity are presented in Appendix D.1.

#### 4.2. Validating Sage as a Proxy for Robustness and Accuracy

Correlation with LLMBar. To establish the credibility of Sage as a new evaluation framework, we first validate its external alignment with existing methodologies by comparing our robustness metrics against LLMBar (Zeng et al., 2023), an established benchmark that evaluates LLM-as-a-Judge using human-annotated ground truth. We focus on the adversarial subset of LLMBar, which is designed to stress-test the robustness of judge models. This subset contains instances where one response is correct while the other is adversarially crafted to be superficially appealing, thus challenging a judge’s ability to remain robust against deceptive quality.

Table 3: Spearman Correlation Coefficients between Sage metrics and external benchmarks. P-values are less than 0.001.

LLMBar RewardBench2 IPI TOV IPI TOV 0.802 0.791 0.890 0.879

Because of the coarse-grained quality difference in LLMBar, we test the same thirteen models evaluated in Section 4.3 on both Sage-Easy and the LLMBar adversarial subset. As shown in Table 3, the results reveal a strong positive correlation between the models’ error rates on LLMBar and our proposed metrics.

Proxy for Accuracy. Beyond robustness, we argue that Sage can also function as an effective proxy for judging accuracy. Theoretically, TOV quantifies the minimum number of pairwise judgments that must be altered for the entire set to become logically coherent. Since logical coherence is a prerequisite for correctness, the total number of errors in a set of judgments must be at least as large as the minimum alterations needed to resolve its logical contradictions. Therefore, TOV establishes a lower bound on the error rate. To empirically substantiate this claim, we leverage a 599-question subset of our evaluation suite for which ground-truth preference labels are available from the RewardBench2. We evaluate the same thirteen LLMs, calculating each model’s error rate against the provided ground-truth and comparing it with their TOV scores from Sage. As shown in Table 3, we see a significantly high Spearman Correlation between the models’ ground-truth error rates and their TOV scores, proving that Sage can serve as a robust proxy for judgment accuracy.

#### 4.3. Evaluating LLM-as-a-Judge with Sage

We benchmark thirteen popular LLMs with the aforementioned settings, including six proprietary models (i.e. Gemini-2.5-Pro and Gemini-2.5-Flash (Comanici and et al., 2025); Gemini-2.0-Flash-Lite (Google, 2025), GPT-5-Chat (OpenAI, 2025), GPT-4o-mini (OpenAI, 2024) and Claude-3-Haiku (Anthropic, 2024)) and seven open source models (i.e. Qwen3-235B-A22B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507 and Qwen3-4BInstruct-2507 (Team, 2025); DeepSeek-R1-0528 (DeepSeek-AI, 2025a), DeepSeek-V3 (DeepSeek-AI and

- et al., 2024), DeepSeek-V3.1 (DeepSeek-AI, 2025b), Llama-3.1-8B-Instruct (Meta, 2024a)). The results are shown in Table 4. All evaluations are conducted at the default temperature to ensure a fair and consistent comparison.

Our comprehensive benchmarking reveals significant robustness deficiencies in current state-of-the-art LLMs. A clear trend emerges where leading models, such as Gemini-2.5-Pro, consistently demonstrate superior robustness with the lowest IPI and TOV scores, indicating stronger local self-consistency and global logical

- Table 4: The performance of thirteen LLMs on Sage, with lower scores indicating greater robustness. A clear trend emerges where advanced models like Gemini-2.5-Pro demonstrate superior robustness.

Factuality Precise IF Mathematics Safety Focus Overall IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ Performance on Sage-Easy

Models

Gemini-2.5-Pro 0.059 0.900 0.084 1.266 0.060 0.924 0.100 1.512 0.058 0.863 0.072 1.091 Gemini-2.5-Flash 0.077 1.163 0.111 1.679 0.075 1.175 0.097 1.504 0.071 1.081 0.087 1.326 Qwen3-235B-A22B-Instruct-2507 0.076 1.134 0.130 1.959 0.098 1.505 0.103 1.557 0.086 1.295 0.098 1.485 DeepSeek-R1-0528 0.095 1.436 0.142 2.183 0.141 2.254 0.124 1.857 0.087 1.300 0.115 1.759 Qwen3-4B-Instruct-2507 0.101 1.532 0.175 2.662 0.126 1.933 0.124 1.943 0.102 1.532 0.126 1.919 Gemini-2.0-Flash-Lite 0.105 1.587 0.152 2.321 0.161 2.433 0.147 2.650 0.102 1.524 0.133 2.091 GPT-5-Chat 0.120 1.818 0.232 3.489 0.104 1.622 0.121 1.820 0.127 1.902 0.143 2.158 DeepSeek-V3-0324 0.130 1.985 0.185 2.783 0.147 2.290 0.173 2.612 0.146 2.200 0.157 2.380 Qwen3-30B-A3B-Instruct-2507 0.136 2.056 0.208 3.113 0.123 1.900 0.171 2.569 0.173 2.597 0.162 2.448 DeepSeek-V3.1 0.134 2.022 0.189 2.920 0.146 2.449 0.204 3.097 0.132 2.017 0.160 2.485

- GPT-4o-mini 0.150 2.294 0.202 3.085 0.164 2.522 0.107 1.667 0.182 2.733 0.164 2.496 Llama-3.1-8B-Instruct 0.204 3.203 0.229 3.506 0.260 4.000 0.230 3.493 0.179 2.755 0.221 3.410 Claude-3-Haiku 0.208 3.177 0.253 3.822 0.367 6.046 0.337 5.691 0.253 3.856 0.279 4.430

Performance on Sage-Hard

- Gemini-2.5-Pro 0.240 3.769 0.291 4.586 0.193 3.661 0.268 4.439 0.250 3.903 0.250 4.079 Gemini-2.5-Flash 0.265 4.175 0.320 5.007 0.257 4.457 0.228 3.732 0.267 4.331 0.269 4.350 Qwen3-235B-A22B-Instruct-2507 0.347 5.326 0.357 5.585 0.256 4.404 0.268 4.124 0.420 6.409 0.331 5.183 Qwen3-4B-Instruct-2507 0.376 5.734 0.405 6.215 0.376 6.254 0.346 5.653 0.433 6.573 0.388 6.078 DeepSeek-R1-0528 0.410 6.397 0.451 7.067 0.276 5.032 0.398 6.404 0.434 6.828 0.396 6.362 Qwen3-30B-A3B-Instruct-2507 0.424 6.458 0.393 6.059 0.337 5.625 0.379 6.065 0.530 7.968 0.413 6.435 Gemini-2.0-Flash-Lite 0.434 6.594 0.376 5.764 0.451 7.417 0.336 5.919 0.480 7.282 0.415 6.571

- GPT-5-Chat 0.433 6.563 0.571 8.619 0.287 4.850 0.286 4.431 0.583 8.764 0.436 6.700 DeepSeek-V3-0324 0.455 7.146 0.455 7.060 0.490 8.286 0.403 6.331 0.533 8.492 0.467 7.446 Claude-3-Haiku 0.522 8.442 0.507 7.985 0.292 5.213 0.352 6.171 0.610 10.161 0.464 7.687 DeepSeek-V3.1 0.513 8.630 0.585 9.171 0.214 3.893 0.438 7.235 0.523 9.470 0.460 7.756

- GPT-4o-mini 0.546 8.261 0.478 7.276 0.452 7.803 0.353 5.755 0.661 9.974 0.502 7.865 Llama-3.1-8B-Instruct 0.559 8.507 0.496 7.600 0.653 10.254 0.581 9.246 0.608 9.265 0.573 8.869

coherence. Crucially, all models show a marked degradation in performance from Sage-Easy to Sage-Hard with an approximately 200% increase on IPI and TOV scores. This gap underscores a key limitation: while models may appear relatively reliable when judging answers of clearly different quality, their adjudicative abilities falter when faced with subtle distinctions, posing a serious threat to their effectiveness in inferencetime enhancement techniques like rejection sampling or Monte Carlo Tree Search. These findings highlight that fundamental consistency remains a substantial challenge for LLMs acting as judges.

#### 4.4. In-depth Analysis

Unjust Judges or Situational Preference? We argue that a robust LLM-as-a-Judge should first model the question internally regardless of how the answers vary. However, the extremely high IPI and TOV scores across even state-of-the-art models raise the concern of whether models are incapable of providing just

- Table 5: Fine-tuning serves as an effective strategy for enhancing robustness, with the majority of specialized judges showing significant improvement on Sage-Hard.

Factuality Precise IF Mathematics Safety Focus Overall IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ Qwen2.5-3B-Instruct (Base) 0.758 11.951 0.677 10.791 0.539 9.833 0.650 10.984 0.795 12.395 0.687 11.211

Models

M-Prometheus-3B 0.497 7.763 0.418 7.622 0.512 8.233 0.465 7.365 0.492 7.725 0.490(↓29%) 7.745(↓31%) JudgeLRM-3B 0.909 13.667 0.765 11.504 0.533 8.342 0.618 9.451 0.930 13.952 0.757(↑ 10%) 11.471(↑ 2%)

Qwen2.5-7B-Instruct (Base) 0.847 12.713 0.765 11.507 0.712 11.583 0.840 12.740 0.911 13.686 0.815 12.435 M-Prometheus-7B 0.564 8.671 0.516 8.014 0.491 7.825 0.451 7.008 0.613 9.323 0.528(↓35%) 8.183(↓34%) JudgeLRM-7B 0.765 11.902 0.695 10.976 0.404 6.673 0.502 8.517 0.811 12.856 0.643(↓21%) 10.291(↓13%)

Mistral-7B-Instruct (Base) 0.793 11.963 0.651 9.889 0.813 12.455 0.753 11.302 0.821 12.357 0.765 11.566 Prometheus-7B-V2.0 0.616 9.634 0.546 8.773 0.602 10.000 0.553 9.186 0.652 10.105 0.592(↓23%) 9.509(↓18%)

Llama-3.1-8B-Instruct (Base) 0.559 8.507 0.496 7.600 0.653 10.254 0.581 9.246 0.608 9.265 0.573 8.869 Skywork-Critic-Llama-3.1-8B 0.491 7.378 0.421 6.379 0.371 5.583 0.371 5.667 0.539 8.081 0.440(↓23%) 6.642(↓25%)

judgments, or whether their judgments are merely situational preferences (Laine et al., 2024, Needham

- et al., 2025), i.e., inconsistent judging criteria encountering different answers under the same question. To validate this hypothesis, we investigate whether an LLM can improve its evaluation by first explicitly articulating its judging rubrics and then using the rubrics to judge the answers across different judging pairs under the same question. Crucially, this rubric is generated only once per question and serves as a fixed standard for all answer pairs, a method designed to mitigate situational preferences by preventing the judge’s evaluation criteria from shifting between comparisons. Figure 5 shows that this approach yields a notable performance boost, reducing IPI and TOV scores by 16.1% and 11.0%. This gap demonstrates that current LLM-as-a-Judge systems indeed exhibit extreme situational preferences when encountering different answer pairs, and that explicit judging rubrics can substantially mitigate this.

Do fine-tuned judges make better judgments? In most cases, yes. A fine-tuned judge is an LLM trained on a preference dataset to improve their evaluation. We benchmark six fine-tuned judges (i.e. Prometheus7B-V2.0 (Kim et al., 2024), Skywork-Critic-Llama-3.1-8B (Tu et al., 2024), M-Prometheus-3/7B (Pombal et al., 2025), and JudgeLRM-3/7B (Chen et al., 2025)) and their corresponding base models. The results are shown in Table 5. Additional results of their performance on Sage-Easy are available in Appendix D.2. Our results demonstrate that fine-tuning is generally an effective strategy for enhancing evaluation robustness, as evidenced by the consistent gains in Skywork-Critic and the Prometheus family. However, the JudgeLRM series exhibits a divergence based on model capacity: while the 7B model manages to improve, the 3B variant suffers a regression. We hypothesize that the larger model possesses sufficient reasoning capabilities to navigate noisy data, whereas the smaller model is more vulnerable to overfitting to specific data artifacts. We attribute the degradation in JudgeLRM to biases inherited from the training datasets, which can cause the model to learn and amplify flawed judgment patterns, compromising its objectivity. See Appendix E.7 for the examples of human biases in training data.

Do Multi-agent Debates or Panels Judge Better? Panels usually help, but debates often hurt. In our evaluation, we also explore the effectiveness of multi-agent judge systems, an approach intended to reduce bias and improve evaluation robustness. We investigate two distinct methodologies: a panel-based approach inspired by POLL (Verga et al., 2024), which leverages a diverse jury of different LLMs, and a debate-based framework, ChatEval (Chan et al., 2023), which utilizes multiple agents derived from a single

- Table 6: Performance comparison of multi-agent systems: POLL panels (left) and ChatEval debates (right). For POLL, “Best Indi.” refers to the best individual model in the panel.

Method IPI-Easy ↓ TOV-Easy ↓ IPI-Hard ↓ TOV-Hard ↓ Panel 1 (Powerful Models)

Method IPI-Easy ↓ TOV-Easy ↓ IPI-Hard ↓ TOV-Hard ↓ Qwen3-4B-Instruct-2507

Best Indi. 0.072 1.091 0.250 4.079

Baseline 0.129 1.952 0.386 5.849

0.067 1.022 0.245 3.897 (↓7%) (↓6%) (↓2%) (↓4%)

0.334 5.105 0.651 10.050 (↑158%) (↑162%) (↑69%) (↑72%)

Aggregate

ChatEval

###### Panel 2 (Weaker Models)

###### Qwen3-30B-A3B-Instruct-2507

Best Indi. 0.133 2.091 0.415 6.571

Baseline 0.180 2.715 0.649 9.780

0.116 1.769 0.400 6.448 (↓13%) (↓15%) (↓4%) (↓2%)

0.261 4.080 0.518 8.395 (↑45%) (↑50%) (↓20%) (↓14%)

Aggregate

ChatEval

LLM. The results are shown in Table 6. For the panel approach, we construct two separate juries: the first comprised of powerful models (Gemini-2.5-Pro, Gemini-2.5-Flash, and Qwen3-235B-A22B-Instruct2507), while the second uses weaker models (Gemini-2.0-Flash-Lite, GPT-4o-mini, and DeepSeek-V3.1). For the POLL method, the aggregated judgments in the majority of cases surpass the performance of the best individual model within each respective group, demonstrating a clear performance boost. Conversely, debate-based ChatEval framework fails to yield an improvement in evaluation quality, demonstrating less robust performance. To further prove that the negative ChatEval result is not configuration-dependent, we conduct ablation experiments on the ChatEval framework. We investigated the impact of three key hyperparameters (num-rounds, argument exchange format, judge selection):

- • Number of Rounds: Vary the num-rounds configuration from 2 rounds down to 1 round.
- • Judge Selection: Change the judge selection mechanism from our original [Critic, Psychologist, Scientist] to [Critic, General Public, Scientist].
- • Argument Exchange Format: Change the exchange format from One-by-One to Simultaneous-Talk.

Our results show that all alternative configurations still performed significantly worse than the non-debate baseline, yielding higher (worse) scores for both IPI and TOV. The results are shown in Figure 4. A detailed case study explaining why the degradation happens is available in Appendix E.6.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- Figure 4: Ablation results for ChatEval showing performance degradation across all configuration variants (lower scores are better).

7.00

0.425

6.75

0.400

6.50

TOVScore

IPIScore

0.375

- 5.50 TOV-Score

5.75

6.00

- 6.25

0.350

0.325

0.300 IPI-Score

- Figure 5: We discover high IPI and TOV scores in Sage-Hard due to the situational preference phenomenon in LLM-as-a-Judge, while deep thinking and explicit rubrics can mitigate this.

Does deep reasoning lead to better performance? Generally yes. We analyze the distinct effects of a model’s intrinsic reasoning depth. For this experiment, we employ gpt-oss-120b (Agarwal et al., 2025), for its configurable reasoning modes: low, medium and high. As illustrated in Figure 5, the results show an improvement as the reasoning mode is intensified from low to high.

What is the practical application of Sage? Selecting robust judges for automated arenas. Here we explore the practical utility of our framework in selecting robust evaluators for large-scale, automated model ranking systems like Arena-Hard-Auto (Chiang et al., 2024). In such systems, models are ranked using Elo ratings derived from pairwise comparisons. The confidence interval of a model’s Elo rating serves as a crucial indicator of its judgment stability; a smaller interval suggests more reliable evaluation performance. Our investigation reveals a positive correlation between our metrics and the Elo rating confidence intervals from Arena-Hard-Auto. Our IPI and TOV scores show Spearman correlations of 0.6374 and 0.6044, respectively, with the confidence intervals. This alignment demonstrates that Sage can effectively identify more stable judges, making it a valuable tool for selecting high-quality evaluators to enhance the reliability of automated arena rankings.

Is Sage sensitive to prompt design? Not materially. We investigate the impact of prompt variations on the judge’s consistency by using additional distinct prompts to test Qwen3-4B-Instruct-2507 on our benchmark. The results are shown in Figure 6. The full content of the different prompts used in this experiment are provided in Appendix E.4.

Easy-IPI Easy-TOV Hard-IPI Hard-TOV

0

2

4

6

8

10

12

14

MaximumAbsolutePercentageChange(||/Base)

- Figure 6: Distribution of performance spread, measured as maximum absolute percentage changed in metric. The stability of rankings across different prompt styles demonstrates the reliability of our evaluation framework.

Model-Agnostic Nature of Benchmark Difficulty. A potential concern regarding the validity of Sage-Hard is whether the observed high inconsistency stems from specific generative patterns of the source model (Gemini-2.5-Flash) rather than the intrinsic difficulty of distinguishing between high-quality, homogeneous responses. To address this, we validate the model-agnostic nature of the benchmark’s difficulty by switching the answer generator. We construct a parallel version of Sage-Hard by replacing the answer generator with Qwen3-235B-A22B, whose capability is proximate to Gemini-2.5-Flash. We then evaluated the judge model’s consistency on this new set. As shown in Table 7, the judge’s performance exhibits minimal deviation when the source model is changed. These negligible variations confirm that the robustness challenges revealed by Sage-Hard are not artifacts of a specific model’s writing style.

- Table 7: Validation of Model-Agnostic Difficulty. When switching the answer generator from Gemini to Qwen, the judge’s performance exhibits minimal deviation, with IPI and TOV scores shifting by only 0.5%.

Dataset IPI-Score(↓) TOV-Score(↓)

Sage-Hard generated by Gemini 0.388 6.078 Sage-Hard generated by Qwen 0.386(↓0.5%) 6.109(↑0.5%)

Human Consistency Baseline. To contextualize the performance of LLM judges, we conduct a study to measure the consistency of human evaluation using our metrics. We recruited 20 graduate-level students to evaluate a sampled set of 50 questions from both Sage-Easy and Sage-Hard. To ensure the validity of our metrics, we enforced two rigorous constraints. First, for any unordered pair {Ai, Aj}, the forward pass (Ai vs Aj) and the reversed pass (Aj vs Ai) were assigned to different annotators to eliminate self-consistency based on memory. Second, we implemented a spacing constraint where no annotator was allowed to evaluate pairs belonging to the same question consecutively; instead, these tasks were interleaved with inquiries from other questions. This design mitigates short-term memory bias and context priming, ensuring that the metric captures the true stability of human consensus. The results, presented in Table 8, reveal that human judgment is far from perfect. These findings empirically substantiate the “noisy and inconsistent data” challenge illustrated in Figure 1. The high inconsistency rates demonstrate that even qualified human evaluators struggle to maintain consistent preferences and logical transitivity when facing complex, highquality responses. This underscores the precariousness of treating human annotation as an absolute gold standard and highlights the necessity of intrinsic consistency checks like Sage.

- Table 8: The performance of human annotators on Sage. The results show significant inconsistency, particularly on the harder subset, confirming that human judgment is susceptible to noise and lacks strict transitivity.

#### Benchmark IPI-Score(↓) TOV-Score(↓)

Sage-Easy 0.145 2.239 Sage-Hard 0.332 6.523

Cost Effectiveness and Scalability. Our claim of “without human effort” specifically targets the elimination of the annotation bottleneck. Sage converts this prohibitive human resource cost into a negligible computational expense. A complete evaluation cycle for a single dataset involves 650 questions with 6 candidate

answers ((62) = 15 pairs). Under our symmetrized protocol, this necessitates 650 × 15 × 2 = 19,500 distinct judgments. Leveraging efficient commercial models (priced at ≈ $0.15/1M input tokens), a full Sage run

costs less than $7 USD and completes in under one hour with moderate concurrency. In stark contrast, the

time and cost for humans to replicate this scale is immense. On Sage-Hard, the average input length reaches 1911 tokens. According to Wang et al. (2021) and Xu et al. (2025), the cost for human labeling is 0.0022$ per token and the average time consumption is 260 token per minute. Consequently, replicating the consistency checks performed by Sage with human experts would demand approximately $81981 USD and 100 days. Furthermore, the modality-agnostic nature of Sage allows for seamless extension to multimodal tasks. The same consistency metrics can be directly applied to assess the judgments on multimodal understanding as well as image generation, offering a scalable solution to the subjectivity challenges inherent in multimodal assessment.

- Table 9: Consistency rates between direct scoring and pairwise comparison on Sage. All models exhibit widespread inconsistency, particularly on Sage-Hard. Models Factuality↑ Focus↑ Mathematics↑ Precise IF↑ Safety↑ Overall↑

Consistency Rates on Sage-Easy

###### GPT-5-Chat 0.843 0.837 0.735 0.799 0.654 0.777

- GPT-4o-mini 0.723 0.686 0.730 0.693 0.671 0.702 Qwen3-235B-A22B-Instruct-2507 0.808 0.774 0.720 0.773 0.384 0.695 Gemini-2.5-Pro 0.785 0.748 0.719 0.802 0.344 0.685 DeepSeek-R1-0528 0.750 0.695 0.714 0.712 0.511 0.679 DeepSeek-V3.1 0.725 0.686 0.679 0.724 0.532 0.673 Gemini-2.5-Flash 0.756 0.727 0.690 0.767 0.307 0.652 Qwen3-30B-A3B-Instruct-2507 0.728 0.704 0.673 0.710 0.321 0.631 Llama-3.1-8B-Instruct 0.669 0.585 0.583 0.657 0.555 0.615 Qwen3-4B-Instruct-2507 0.694 0.677 0.643 0.637 0.358 0.605 Gemini-2.0-Flash-Lite 0.641 0.531 0.600 0.666 0.295 0.552 DeepSeek-V3-0324 0.567 0.458 0.612 0.613 0.392 0.531 Claude-3-Haiku 0.503 0.459 0.450 0.420 0.477 0.462

Consistency Rates on Sage-Hard

- GPT-5-Chat 0.434 0.350 0.480 0.416 0.392 0.415 DeepSeek-V3.1 0.383 0.431 0.500 0.359 0.360 0.405 Llama-3.1-8B-Instruct 0.403 0.343 0.356 0.471 0.404 0.403 Claude-3-Haiku 0.346 0.330 0.448 0.299 0.526 0.384 DeepSeek-R1-0528 0.351 0.313 0.374 0.377 0.341 0.351 GPT-4o-mini 0.282 0.277 0.396 0.391 0.349 0.338 Qwen3-235B-A22B-Instruct-2507 0.300 0.260 0.539 0.345 0.235 0.332

- Gemini-2.5-Pro 0.241 0.209 0.507 0.390 0.294 0.325 Gemini-2.5-Flash 0.273 0.205 0.484 0.384 0.236 0.315 Gemini-2.0-Flash-Lite 0.232 0.184 0.417 0.312 0.444 0.313 Qwen3-30B-A3B-Instruct-2507 0.232 0.215 0.500 0.300 0.205 0.288 Qwen3-4B-Instruct-2507 0.238 0.200 0.412 0.286 0.308 0.286 DeepSeek-V3-0324 0.142 0.102 0.329 0.193 0.258 0.202

Inconsistency Between Direct Scoring and Pairwise Comparison. Another critical dimension of a judge’s reliability is its consistency across different evaluation formats, specifically between direct scoring (assigning an absolute score to a single answer) and pairwise comparison (choosing the better of two answers). To

[Figure 29]

- Figure 7: The difference of IPI, TOV and Consistency Rates when prompting models to directly output verdicts in comparison to the original conditions. Red bars indicate degradation and blue bars indicate improvement.

quantify this, we measure the consistency rate. For direct scoring, we prompt an LLM judge to score each answer twice and then average the results to get a final score. A judgment is consistent if the answer that receives the higher average direct score is also the one preferred in the head-to-head pairwise comparison. As shown in Table 9, this inconsistency is widespread, although leading models like GPT-5-Chat are more consistent, but its performance on Sage-Hard is still poor. This framing effect likely stems from direct scoring’s reliance on a poorly-calibrated internal quality scale, in contrast to the more grounded context provided by pairwise comparison.

The Influence of Reasoning on Judgment Consistency. To understand the influence of explicit reasoning, we conduct a comparative experiment. In our main experiments, the LLM-as-a-Judge was prompted to generate a detailed explanation before delivering its final verdict. Our follow-up study utilized a modified prompt that strictly instructed the judge models to output their final decision directly, without any accompanying rationale. As shown in Figure 7, removing reasoning leads to widespread degradation on both Sage-Easy and Sage-Hard. This confirms that explicit reasoning serves as a necessary sanity check for clear-cut comparisons. However, Sage-Hard reveals a capability-dependent divergence: while weaker models (e.g., Qwen3-30B) suffer instability (52% degradation) without reasoning, top-tier models (e.g., DeepSeek-V3) demonstrate resilience or even improved robustness.This pattern is also reflected in the consistency between direct scoring and pairwise comparison. As shown in Figure 7, removing reasoning generally reduces alignment on Sage-Easy (including many strong models), while on Sage-Hard the effect becomes highvariance: weaker models often degrade substantially, whereas a few top-tier models remain stable or even improve slightly. The comprehensive results are available in Appendix D.3.

### 5. Related Work

LLM-as-a-Judge. LLM-as-a-Judge (Zheng et al., 2023) has emerged as a scalable and cost-effective alternative to human evaluation for assessing the quality of generative AI outputs. This approach utilizes a powerful LLM to judge the responses of other models, addressing the limitations of traditional metrics like BLEU and ROUGE that often fail to capture deeper semantic qualities such as coherence, factual accuracy, and relevance.

However, the reliability of LLM-as-a-Judge is a significant concern, with numerous studies (Zheng et al.,

- 2023, Chen et al., 2024a, Wu and Aji, 2025) highlighting its susceptibility to various biases. These include verbosity bias, where longer answers are favored irrespective of their quality; position bias, a preference for the first or last presented response; and self-enhancement bias, where a model tends to rate its own outputs more favorably. Research (Chen et al., 2024a) has also identified other distorting influences, such as authority bias, where an LLM may favor answers containing citations even if they are fabricated. These identified biases underscore the necessity for continued investigation and validation of the reliability of LLM-as-a-Judge.

Benchmark for LLM-as-a-Judge. Following the recognition of these potential biases of LLM-as-a-Judge, researchers have focused on developing specialized benchmarks to systematically evaluate the reliability and behavior of LLM judges. Unlike general-purpose LLM benchmarks that assess broad capabilities, these targeted frameworks are designed specifically to scrutinize the adjudicative performance of models. For instance, foundational benchmarks such as MT-Bench and Chatbot Arena (Zheng et al., 2023) are introduced to verify the agreement between LLM judges and human preferences on open-ended, multi-turn questions. Subsequent works like Tan et al. (2025) and Gera et al. (2025) continue to follow this paradigm, primarily assessing the capability of LLM judges by measuring the correlation between their assessments and human preference judgments.

However, this reliance on human judgment as the definitive “gold standard” is unreliable for three key reasons: First, human annotators are susceptible to inherent biases (Zheng et al., 2023, Wu and Aji, 2025), including authority bias and misinformation oversight bias (Chen et al., 2024a). In addition, Chen et al. (2024a) shows that human evaluators of LLMs can be more biased than the models themselves. Second, there is a persistent issue of inter-annotator disagreement (Zhang et al., 2024). Different human evaluators often provide inconsistent assessments, particularly for tasks that are subjective or nuanced. This lack of consensus means that the “ground truth” data used for benchmarking is often noisy and unreliable. Finally, as AI models advance, they are beginning to surpass human capabilities in specialized domains. When AI generates highly complex or lengthy outputs, human annotators might struggle to accurately assess their quality (Tan et al., 2025). In such scenarios, human annotations may no longer be a reliable ground truth.

Fine-tuned Judge. In the pursuit of improving automated evaluation accuracy, one prevalent strategy involves specializing a model using preference datasets, resulting in a fine-tuned “judge” model (Kim et al.,

- 2024, Wang et al., 2024b,a, He et al., 2024, Zhu et al., 2025). These datasets generally comprise a series of prompts, each followed by multiple model-generated responses, with evaluators providing labels to indicate the superior response. By leveraging this data, the judge model is trained to predict human evaluative behaviors, enabling it to autonomously score or rank new model outputs. The fine-tuning process allows the judge to learn nuanced patterns in human preferences, such as understanding which aspects of a response

are prioritized. As a result, the judge can offer an automated alternative to human evaluation, making it invaluable for large-scale applications where human assessment may be time-consuming or impractical. However, this approach is not without its limitations (Huang et al., 2024a,b). For those judge models that are fine-tuned on datasets derived from human evaluations, they inevitably inherit the biases and inconsistencies present in the human labeling process. Human annotators, despite their best efforts, may display subjective tendencies, varying interpretation of instructions, or inconsistencies in rating, which can be subtly reflected in the model’s predictions (Chen et al., 2024b). As a consequence, the fine-tuned judge may sometimes generate evaluations that do not align with a broader, more objective standard (Gao et al., 2023). Given these challenges, the reliability and fairness of fine-tuned judge models as objective evaluators must be subjected to thorough scrutiny. It becomes crucial to investigate the degree to which these models mirror human biases and assess their robustness across diverse contexts and response types.

- 6. Conclusion

We introduce Sage, a novel framework for evaluating LLM-as-a-Judge without human annotation or any extrinsic information by measuring local and global logical consistency. We validate that our metrics are exceptionally stable and can serve as a strong proxy for accuracy. Our experiments reveal significant robustness deficiencies in current state-of-the-art models. We attribute these inconsistent judgments to a newly identified phenomenon called situational preference where models fail to maintain a stable internal gauging principle across different contexts. To address this, we demonstrate that implementing self-generated rubrics effectively mitigates situational preference and boosts judgment consistency. We also investigate the impact of fine-tuning and explanatory reasoning on evaluation performance. Crucially, we apply Sage to human evaluators and reveal the fragility of human judgment. The significant instability observed in our human baseline proves that human annotation is an unreliable gold standard. Consequently, Sage provides a scalable, reliable, and cost-effective tool to diagnose and improve LLM evaluators, paving the way for more consistent and rational AI systems.

- 7. Acknowledgements We are profoundly grateful to Tianyi Zhou for his valuable insight and feedback on this paper.

Reproducibility Statement

To ensure the reproducibility of our research, we will release all source code, the curated dataset, and the collected model responses. The foundational methodology of our framework, including the formal problem definition, the symmetrized evaluation protocol, and the definitions of our IPI and TOV metrics, is detailed in Section 2. The comprehensive process for curating our 650-question dataset is described in Section 3, with further implementation details provided in Appendix B.1. For our theoretical claims, a complete derivation of the variance bounds for our metrics is available in Appendix A. Furthermore, all detailed experimental setups, including descriptions of the models evaluated (Appendix B.4) and the exact prompts used in our experiments, are provided in Appendix E to facilitate the replication of our results.

### References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Anastasios N. Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distributionfree uncertainty quantification. CoRR, abs/2107.07511, 2021. URL https://arxiv.org/abs/2107. 07511.

Anthropic. Claude 3 haiku: our fastest model yet. https://www.anthropic.com/news/ claude-3-haiku, 2024. Accessed: 2025-09-08.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better llm-based evaluators through multi-agent debate. arXiv preprint arXiv:2308.07201, 2023.

Guiming Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. Humans or llms as the judge? A study on judgement bias. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 8301–8327. Association for Computational Linguistics, 2024a. doi: 10.18653/V1/2024.EMNLP-MAIN.474. URL https://doi.org/10.18653/v1/2024.emnlp-main.

474.

Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. Humans or llms as the judge? A study on judgement biases. CoRR, abs/2402.10669, 2024b. doi: 10.48550/ARXIV.2402.10669. URL https://doi.org/10.48550/arXiv.2402.10669.

Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng He. Judgelrm: Large reasoning models as a judge. arXiv preprint arXiv:2504.00050, 2025.

Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, 2024.

Gheorghe Comanici and et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. doi: 10.48550/ ARXIV.2507.06261. URL https://doi.org/10.48550/arXiv.2507.06261.

DeepSeek-AI. Deepseek-r1-0528 release. https://api-docs.deepseek.com/news/news250528, 2025a. Accessed: 2025-09-08.

DeepSeek-AI. Deepseek-v3.1 release. https://api-docs.deepseek.com/news/news250821, 2025b. Accessed: 2025-09-08.

DeepSeek-AI and et al. Deepseek-v3 technical report. CoRR, abs/2412.19437, 2024. doi: 10.48550/ARXIV. 2412.19437. URL https://doi.org/10.48550/arXiv.2412.19437.

Nicolai Dorka. Quantile regression for distributional reward models in rlhf. arXiv preprint arXiv:2409.10164, 2024.

Abhimanyu Dubey and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ ARXIV.2407.21783. URL https://doi.org/10.48550/arXiv.2407.21783.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36:30039–30069, 2023.

Bradley Efron. Bootstrap methods: another look at the jackknife. In Breakthroughs in statistics: Methodology and distribution, pages 569–593. Springer, 1992.

Gonçalo Faria and Noah A Smith. Sample, don’t search: Rethinking test-time alignment for language models. arXiv preprint arXiv:2504.03790, 2025.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

Ariel Gera, Odellia Boni, Yotam Perlitz, Roy Bar-Haim, Lilach Eden, and Asaf Yehudai. Justrank: Benchmarking LLM judges for system ranking. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 682–712. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.

acl-long.34/. Google. Gemini 2.0: Flash, flash-lite and pro. https://developers.googleblog.com/en/ gemini-2-family-expands/, 2025. Accessed: 2025-09-08.

Srishti Gureja, Lester James Validad Miranda, Shayekh Bin Islam, Rishabh Maheshwary, Drishti Sharma, Gusti Triandi Winata, Nathan Lambert, Sebastian Ruder, Sara Hooker, and Marzieh Fadaee. M-rewardbench: Evaluating reward models in multilingual settings. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 43–58. Association for Computational Linguistics, 2025. URL https://aclanthology.org/ 2025.acl-long.3/.

Yuanqin He, Yan Kang, Lixin Fan, and Qiang Yang. Fedeval-llm: Federated evaluation of large language models on downstream tasks with collective wisdom. arXiv preprint arXiv:2404.12273, 2024.

Tomas Horych, Christoph Mandl, Terry Ruas, Andre Greiner-Petter, Bela Gipp, Akiko Aizawa, and Timo Spinde. The promises and pitfalls of llm annotations in dataset labeling: a case study on media bias detection. arXiv preprint arXiv:2411.11081, 2024.

Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. Large language models are zero-shot rankers for recommender systems. In European Conference on Information Retrieval, pages 364–381. Springer, 2024.

Zhengyu Hu, Jieyu Zhang, Zhihan Xiong, Alexander Ratner, Hui Xiong, and Ranjay Krishna. Language model preference evaluation with multiple weak evaluators. arXiv preprint arXiv:2410.12869, 2024.

Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang, Bing Xu, and Tiejun Zhao. An empirical study of llm-as-a-judge for llm evaluation: Fine-tuned judge model is not a general substitute for gpt-4. arXiv preprint arXiv:2403.02839, 2024a.

Hui Huang, Yingqi Qu, Hongli Zhou, Jing Liu, Muyun Yang, Bing Xu, and Tiejun Zhao. On the limitations of fine-tuned judge models for llm evaluation. arXiv preprint arXiv:2403.02839, 2024b.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. CoRR, abs/2310.06825, 2023. doi: 10.48550/ARXIV.2310.06825. URL https://doi.org/10.48550/arXiv.2310.06825.

Seungone Kim, Jamin Shin, Yejin Choi, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing fine-grained evaluation capability in language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum? id=8euJaTveKw.

Rudolf Laine, Bilal Chughtai, Jan Betley, Kaivalya Hariharan, Mikita Balesni, Jérémy Scheurer, Marius Hobbhahn, Alexander Meinke, and Owain Evans. Me, myself, and ai: The situational awareness dataset (sad) for llms. Advances in Neural Information Processing Systems, 37:64010–64118, 2024.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939, 2024.

Hsuan Wei Liao, Christopher Klugmann, Daniel Kondermann, and Rafid Mahmood. Minority reports: Balancing cost and quality in ground truth data annotation. arXiv preprint arXiv:2504.09341, 2025.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634, 2023.

Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vulić, Anna Korhonen, and Nigel Collier. Aligning with human judgement: The role of pairwise preference in large language model evaluators. arXiv preprint arXiv:2403.16950, 2024.

LMSYS. Leaderboard overview. https://lmarena.ai/leaderboard, 2025. Accessed: 2025-09-08.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Qingwei Lin, Jianguang Lou, Shifeng Chen, Yansong Tang, and Weizhu Chen. Arena learning: Build data flywheel for llms post-training via simulated chatbot arena. arXiv preprint arXiv:2407.10627, 2024.

Meta. Introducing llama 3.1: Our most capable models to date. https://ai.meta.com/blog/ meta-llama-3-1/, 2024a. Accessed: 2025-09-08.

Meta. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. https: //ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/, 2024b. Accessed: 2025-09-08.

Joe Needham, Giles Edkins, Govind Pimpale, Henning Bartsch, and Marius Hobbhahn. Large language models often know when they are being evaluated. arXiv preprint arXiv:2505.23836, 2025.

Mark EJ Newman. Efficient computation of rankings from pairwise comparisons. Journal of Machine Learning Research, 24(238):1–25, 2023.

OpenAI. Gpt-4o mini: advancing cost-efficient intelligence. https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024. Accessed: 2025-09-08.

OpenAI. Introducing gpt-5. https://openai.com/index/introducing-gpt-5/, 2025. Accessed: 2025-09-08.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

José Pombal, Dongkeun Yoon, Patrick Fernandes, Ian Wu, Seungone Kim, Ricardo Rei, Graham Neubig, and André F. T. Martins. M-prometheus: A suite of open multilingual llm judges, 2025. URL https: //arxiv.org/abs/2504.04953.

Shu Pu, Yaochen Wang, Dongping Chen, Yuhang Chen, Guohao Wang, Qi Qin, Zhongyi Zhang, Zhiyuan Zhang, Zetong Zhou, Shuang Gong, Yi Gui, Yao Wan, and Philip S. Yu. Judge anything: MLLM as a judge across any modality. CoRR, abs/2503.17489, 2025. doi: 10.48550/ARXIV.2503.17489. URL https://doi.org/10.48550/arXiv.2503.17489.

Keita Saito, Akifumi Wachi, Koki Wataoka, and Youhei Akimoto. Verbosity bias in preference labeling by large language models. arXiv preprint arXiv:2310.10076, 2023.

Lin Shi, Chiyu Ma, Wenhua Liang, Weicheng Ma, and Soroush Vosoughi. Judging the judges: A systematic investigation of position bias in pairwise comparative assessments by llms. 2024.

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. Preference ranking optimization for human alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18990–18998, 2024.

Richard Sutton. The bitter lesson. Incomplete Ideas (blog), 13(1):38, 2019. Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Yuan Tang, Alejandro Cuadron, Chenguang Wang,

Raluca Popa, and Ion Stoica. Judgebench: A benchmark for evaluating llm-based judges. In The Thirteenth International Conference on Learning Representations, 2025.

Qwen Team. Qwen2.5: A party of foundation models! https://qwenlm.github.io/blog/qwen2.5/,

2024. Accessed: 2025-09-08. Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Shiwen Tu, Zhao Liang, Chris Yuhao Liu, Liang Zeng, and Yang Liu. Skywork critic model series. https:

//huggingface.co/Skywork, September 2024. URL https://huggingface.co/Skywork.

Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008. URL http://jmlr.org/papers/v9/vandermaaten08a.html.

Pat Verga, Sebastian Hofstatter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. Replacing judges with juries: Evaluating llm generations with a panel of diverse models. arXiv preprint arXiv:2404.18796, 2024.

Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu, and Michael Zeng. Want to reduce labeling cost? gpt-3 can help. arXiv preprint arXiv:2108.13487, 2021.

Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. Self-taught evaluators. arXiv preprint arXiv:2408.02666, 2024a.

Yidong Wang, Zhuohao Yu, Wenjin Yao, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. Pandalm: An automatic evaluation benchmark for LLM instruction tuning optimization. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024b. URL https://openreview.net/forum?id=5Nn2BLV7SB.

Koki Wataoka, Tsubasa Takahashi, and Ryokan Ri. Self-preference bias in llm-as-a-judge. arXiv preprint arXiv:2410.21819, 2024.

Minghao Wu and Alham Fikri Aji. Style over substance: Evaluation biases for large language models. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert, editors, Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 297–312. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.coling-main.21/.

Yifei Xu, Tusher Chakraborty, Emre Kıcıman, Bibek Aryal, Eduardo Rodrigues, Srinagesh Sharma, Roberto Estevao, Maria Angels de Luis Balaguer, Jessica Wolk, Rafael Padilha, et al. Rlthf: Targeted human feedback for llm alignment. arXiv preprint arXiv:2502.13417, 2025.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. arXiv preprint arXiv:2401.10020, 3, 2024.

Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. Evaluating large language models at evaluating instruction following. arXiv preprint arXiv:2310.07641, 2023.

Michael J. Q. Zhang, Zhilin Wang, Jena D. Hwang, Yi Dong, Olivier Delalleau, Yejin Choi, Eunsol Choi, Xiang Ren, and Valentina Pyatkin. Diverging preferences: When do annotators disagree and do models know? CoRR, abs/2410.14632, 2024. doi: 10.48550/ARXIV.2410.14632. URL https://doi.org/10. 48550/arXiv.2410.14632.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatgpt interaction logs in the wild. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum? id=Bl8u7ZRlbM.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llmas-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=xsELpEPn4A.

### Ethics Statement

Our dataset is curated from established public research sources: the RewardBench2 benchmark and the WildChat-1M corpus. To mitigate ethical risks, such as the potential inclusion of private information or inappropriate content from real-world user logs, we conducted a rigorous curation process (see Appendix

- B.1). This process involved both large-scale automated filtering and a thorough manual review of every selected question. This ensures that the final dataset is appropriate for research use and aligns with the data-sharing and privacy standards of the original sources.

### The Use of Large Language Models (LLMs)

The use of large language models (LLMs) in this work is strictly limited to auxiliary text editing, such as correcting spelling and improving grammar, and dataset generation. Our study is about LLM-as-a-Judge, therefore we also test various LLMs for this task. All conceptual and technical contributions are the original work of the authors. We are transparent about this limited usage.

### A. Theoretical Analysis of Metric Stability

In this section, we provide a theoretical analysis to substantiate the empirical stability of our proposed metrics, Intra-Pair Instability (IPI) and Weak Total Order Violation (TOV), as presented in 4.1. The core of our analysis is to demonstrate that the variance of these metrics is exceptionally low, thereby ensuring their reliability against the inherent stochasticity of LLM judges.

The foundational source of any potential instability in our evaluation framework stems from the stochastic nature of the LLM judge, ℳ. When queried multiple times with the identical input triplet (Q, Ai, Aj), the model’s judgment, yij = Jℳ(Q, Ai, Aj), may fluctuate. Our analysis proceeds in three stages: first, we certify the stability of a single pairwise judgment; second, we bound the variance of the per-question metrics; and third, we establish the stability of the final, aggregate benchmark scores.

#### A.1. Certifying Single-Pair Judgment Stability via Conformal Prediction

To formally quantify the stability of individual judgments, we adopt principles from Conformal Prediction (Angelopoulos and Bates, 2021). We posit that for any given question-answer pair, there exists a “stable judgment,” which represents the most stable outcome if the model were to be sampled repeatedly. We approximate this stable judgment by the modal outcome over a large number of trials.

We construct a large-scale calibration set, 𝒞, by selecting N = 800 distinct question-answer pairs. For each pair k ∈ {1, . . . , N}, we prompt the LLM judge T = 20 times, yielding a total of N × T = 16,000 individual judgments. For each pair k, we define its stable judgment, y∗k , as the most frequently observed outcome:

I(y(kt) = y)

T

∑

y∗k = argmax y∈{−1,0,1}

t=1

where y(kt) is the outcome of the t-th judgment for the k-th pair. We can now use the n = 16,000 judgments in 𝒞 to build a calibration set for a new judgment. Let the nonconformity score for a given judgment y(kt) be its disagreement with the stable judgment: s(y(kt)) = I(y(kt) ≠ y∗k ). By applying the conformal prediction framework to this large calibration set of scores, we can construct a prediction interval for a new, unseen judgment. Our empirical analysis on this calibration set reveals that the fraction of judgments deviating from their stable counterpart is exceedingly small. Following the standard procedure for conformal calibration, we can formally certify that for any new judgment ynew, the probability of it matching its corresponding stable judgment y∗new is bounded with high confidence. Specifically, for a desired miscoverage rate α = 0.03, the procedure yields the following guarantee:

P(ynew = y∗new) ≥ 1 − α = 0.97

This result provides a strong probabilistic guarantee that any single pairwise comparison performed by the judge is highly likely to be stable.

#### A.2. Bounding the Variance of Per-Question Metrics

For each question in our benchmark, the calculation of IPI and TOV scores relies on a set of pairwise comparisons. Given that we generate n = 6 candidate answers, a full round-robin evaluation under our

symmetrized protocol requires M = 2 × (62) = 30 individual judgments. Our objective is to establish a rigorous, high-confidence upper bound for the variance of the per-question metric, Var(TOV(Q)), which

arises from the LLM judge’s inherent stochasticity. By definition, the variance of the measured score TOV(Q) is the expected squared difference from its mean:

Var(TOV(Q)) = E[(TOV(Q) − E[TOV(Q)])2] (7)

A fundamental property of variance is that it represents the minimum possible expected squared error. For any constant c, the following inequality holds: Var(TOV(Q)) ≤ E[(TOV(Q) − c)2]. We can leverage this property by strategically choosing a constant. Let us choose the deterministic stable score, TOV∗(Q), as our

constant c. This yields this inequality:

Var(TOV(Q)) ≤ E[(TOV(Q) − TOV∗(Q))2] (8)

Let the deviation from the stable score be ∆TOV(Q) = TOV(Q) − TOV∗(Q). Equation 8 can be rewritten as:

Var(TOV(Q)) ≤ E[∆TOV(Q)2] (9) Our task now simplifies to finding an upper bound for the second moment of this deviation.

Let X be the random variable for the total number of unstable judgments among the M = 30 trials. As established in Section B.1, the probability p of any single judgment being unstable is bounded by p ≤ α = 0.03. Assuming independence across judgments, X follows a Binomial distribution, X ∼ ℬ(M, p).

- A direct, deterministic relationship connects the score deviation to the number of unstable judgments. Since the TOV score is the minimum number of edge modifications required to resolve all logical contradictions, X

unstable judgments can alter the final score by at most X. This gives the inequality ∣∆TOV(Q)∣ ≤ X, which implies:

∆TOV(Q)2 ≤ X2 (10) By taking the expectation, we can chain the inequalities together:

Var(TOV(Q)) ≤ E[∆TOV(Q)2] ≤ E[X2] (11)

The second moment of a binomial random variable is given by E[X2] = Var(X) + (E[X])2 = Mp(1 − p) + (Mp)2. Using M = 30 and the upper bound p = 0.03, we compute:

E[X] = 30 × 0.03 = 0.9 (12) Var(X) = 30 × 0.03 × (1 − 0.03) = 0.873 (13)

Therefore, the second moment of X is:

E[X2] = 0.873 + (0.9)2 = 1.683 (14)

This directly provides a tight and rigorously derived upper bound for the variance of the per-question TOV score:

Var(TOV(Q)) ≤ 1.683 (15)

This result formally demonstrates that the variance of the per-question scores is exceptionally small, confirming that our metrics are highly robust to the inherent stochasticity of LLM judges.

An identical argument holds for the IPI score, yielding a similarly small per-question variance. The IPI score for a question, IPI(Q), is the fraction of inconsistent pairs. It is calculated over N = (62) = 15 unique pairs of answers. Each inconsistent pair contributes 1 to a sum, which is then normalized by N. An unstable judgment can affect the consistency of at most one pair, thus changing the sum by at most 1. Therefore, X unstable judgments can change the sum of inconsistent pairs by at most X. The deviation of the normalized IPI score, ∆IPI(Q), is thus bounded by:

∣∆IPI(Q)∣ ≤

X N

(16)

It is worth noting that this inequality can be tightened; since the IPI score is bounded in [0,1], the maximal deviation is 1, making the true bound ∣∆IPI(Q)∣ ≤ min(X/N,1). By proceeding with the analytically simpler X/N, we are establishing a conservative overestimate for the variance, which strengthens our claim of stability. Following the same logic, we can bound its variance:

2

Var(IPI(Q)) ≤ E[∆IPI(Q)2] ≤ E[(

X N)

E[X2] (17)

] =

1 N2

Substituting N = 15 and our previously calculated value for E[X2]:

Var(IPI(Q)) ≤

1.683 152

1.683 225

≈ 0.0075 (18)

=

These results formally demonstrate that the variances of both per-question TOV and IPI scores are exceptionally small, confirming that our metrics are highly robust to the inherent stochasticity of LLM judges.

#### A.3. Stability of Aggregate Benchmark Scores

The final Sage metrics are the aggregate scores, IPI and TOV, which are the arithmetic means of the per-question scores over the entire set of ∣𝒬∣ = 650 questions:

TOV(Q) and IPI =

IPI(Q) (19)

1 ∣𝒬∣

1 ∣𝒬∣

∑

∑

TOV =

Q∈𝒬

Q∈𝒬

Assuming the scores for each question are independent and identically distributed (i.i.d.) random variables—a standard assumption for a diverse benchmark—the variance of the mean is the per-question variance divided by the number of questions.

Using the upper bound for the per-question TOV variance derived in Section A.2, we can bound the variance of the final aggregate TOV score:

Var(TOV(Q)) ∣𝒬∣

Var(TOV) =

1.683 650

≈ 2.59 × 10−3 (20)

≤

Similarly, using the upper bound for the per-question IPI variance, we can bound the variance of the final aggregate IPI score:

Var(IPI(Q)) ∣𝒬∣

Var(IPI) =

0.0075 650

≈ 1.15 × 10−5 (21)

≤

These resulting variances for both aggregate metrics are exceptionally small, indicating that the final reported scores are highly concentrated around their expected values.

In conclusion, this theoretical analysis, grounded in first principles and basic statistical properties, formally demonstrates the robustness of our evaluation framework. The high stability of individual judgments propagates through the metric calculation, resulting in aggregate scores for both IPI and TOV with minimal variance. This theoretical finding is in strong alignment with the empirical results presented in Table 2, confirming that Sage provides a consistent and reliable methodology for assessing the reasoning capabilities

of LLM judges.

### B. Detailed Experiment Setups

#### B.1. Dataset Curation

The curation process for our benchmark’s dataset is meticulously designed to ensure both diversity and representativeness, as illustrated in Figure 8. We began by drawing questions from two distinct, high-quality sources. First, we extracted questions from five core categories within the RewardBench2 dataset—namely Factuality, Focus, Precise Instruction Following, Mathematics, and Safety—to establish a foundation of structured evaluation problems. These questions are manually selected to ensure semantic uniqueness. To complement this and incorporate more natural, real-world user interactions, we also sourced a large volume of queries from the WildChat-1m corpus, which contains logs of human-LLM conversations. These queries underwent a rigorous screening process, including both large-scale automated filtering and manual review, to select for relevance and clarity. The questions from both sources are then merged to form the final, comprehensive set of 650 questions. This dual-source approach ensures that our benchmark covers a wide semantic space, balancing formal assessment criteria with the unpredictability of genuine user inquiries, which is essential for a robust evaluation of LLM judges.

###### Question Set Curation

Benchmark Data Components

Original Dataset

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Best-of-Four Selection Problems

[Figure 34]

Preliminary Question Set

Select From Five Subset

Extract Question

Remain Unchanged

[Figure 35]

Question Set

RewardBench2

[Figure 36]

[Figure 37]

Merged Question Set

[Figure 38]

[Figure 39]

[Figure 40]

Prompt LLMs to Generate Answers

Human-LLM Conversations

Large-Scale Question Set

Manual Screening

Extract Question

Answer Set

WildChat-1m

Figure 8: Curation of our dataset.

#### B.2. Spearman Rank Correlation Coefficient

The Spearman Rank Correlation Coefficient, commonly denoted by ρ or rs, is a non-parametric measure of rank correlation (statistical dependence between the rankings of two variables). It assesses how well the relationship between two variables can be described using a monotonic function. Unlike the Pearson correlation, which assumes a linear relationship and normally distributed data, Spearman’s correlation evaluates the monotonic relationship, making it more robust to outliers and non-linear associations commonly found in LLM evaluation scores. The coefficient’s value is constrained to the interval [−1,1].

- B.2.1. Interpretation of the Coefficient The value of the Spearman correlation coefficient (rs) is interpreted as follows:

- • rs = +1: Indicates a perfect positive monotonic relationship. As the rank of one variable increases, the rank of the other variable increases consistently. This implies that the judge model’s ranking of answers perfectly matches the reference ranking, even if the absolute score intervals differ.
- • rs = −1: Indicates a perfect negative monotonic relationship. As the rank of one variable increases, the rank of the other variable decreases consistently.
- • rs = 0: Indicates no monotonic tendency between the two variables. The rankings are essentially random with respect to one another.

The magnitude of ∣rs∣ indicates the strength of the association.

#### B.2.2. Mathematical Formulation

The Spearman correlation coefficient is defined as the Pearson correlation coefficient between the rank variables.

For a sample of size n, the n raw scores Xi and Yi are converted to ranks rg(Xi) and rg(Yi). If there are tied ranks, the average of the ranks that would have been assigned to all the tied values is assigned to each value. The formula is given by:

covσ(rgrgXXσrg,rgY Y) (22) where:

rs =

- • rgX and rgY represent the rank variables of the raw data X and Y.
- • cov(rgX,rgY) is the covariance of the rank variables.
- • σrgX and σrgY are the standard deviations of the rank variables.

#### B.3. Coefficient of Variation

The Coefficient of Variation (CV) is a standardized statistical measure of the relative dispersion of a data distribution. Unlike the standard deviation, which quantifies absolute variability, the CV expresses the standard deviation as a fraction of the arithmetic mean. This normalization renders the CV a dimensionless quantity, thereby facilitating the comparison of variability across datasets with different units of measurement or significantly different means.

For a population, the Coefficient of Variation is defined as the ratio of the standard deviation (σ) to the mean (µ), provided that the mean is non-zero:

σ ∣µ∣

CV =

For a sample, the CV is estimated using the sample standard deviation (s) and the sample mean (x¯):

s ∣x¯∣

cv =

The absolute value of the mean is often used in the denominator to ensure the CV remains non-negative and is well-defined for negative means, preserving its interpretation as a measure of variability magnitude.

The primary utility of the CV lies in its capacity to provide a relative measure of consistency or homogeneity. A lower CV indicates less variability relative to the mean, suggesting greater consistency within the data. Conversely, a higher CV signifies greater relative dispersion. This property is particularly advantageous when comparing the degree of variation between two or more groups of data. For instance, comparing the standard deviation of prices in two different currencies is not directly meaningful; however, their Coefficients of Variation can be compared to determine which currency’s price level is relatively more stable, as it is a unit-free metric.

#### B.4. Models

Large Language Models. An LLM is an advanced AI model, typically using a Transformer architecture, trained on massive text data to understand and generate natural language by predicting the next token. Pre-trained on broad datasets, they can be fine-tuned for specific tasks. Their large scale, with billions of parameters, results in strong generalization and emergent abilities for diverse tasks like text generation, summarization, translation, and question answering. The detailed information about the models we used in our experiments is as follows:

- • DeepSeek-R1-0528 (DeepSeek-AI, 2025a): DeepSeek-R1-0528 is a 671B sparse Mixture-of-Experts (MoE) model with 37B active parameters and a 128K context length. Built upon DeepSeek-V3-Base, it is trained using reinforcement learning to enhance its capabilities in complex reasoning, mathematics, and coding.
- • DeepSeek-V3-0324 (DeepSeek-AI and et al., 2024): DeepSeek-V3-0324 is a 671B Mixture-of-Experts (MoE) model with 37B active parameters per token. Trained on a 14.8T-token dataset, it uses optimized attention and advanced expert routing to enhance performance on complex reasoning and coding tasks with computational efficiency.
- • DeepSeek-V3.1 (DeepSeek-AI, 2025b): DeepSeek-V3.1 is a 671B Mixture-of-Experts (MoE) model that activates 37B parameters per token. It features a hybrid architecture for reasoning and fast responses, supports a 128K context window, and is post-trained for tool-calling and agentic tasks.
- • Gemini-2.0-Flash-Lite (Google, 2025): Gemini-2.0-Flash-Lite is a lightweight, multimodal Google model for high-speed, high-volume tasks where latency and cost are critical. This smaller, faster variant excels at summarization and chat, ideal for scalable services and on-device applications requiring rapid, resourceefficient inference.
- • Gemini-2.5-Flash (Comanici and et al., 2025): Gemini-2.5-Flash is a cost-efficient, multimodal foundation model by Google DeepMind with a 1 million context window. It uses a sparse Mixture-of-Experts (MoE) architecture to balance performance, cost, and latency, and is optimized for speed in reasoning and multimodal tasks.
- • Gemini-2.5-Pro (Comanici and et al., 2025): Gemini-2.5-Pro is Google DeepMind’s top-tier ”thinking” multimodal model optimized for advanced reasoning, coding, science, and long-form tasks. It supports text, images, audio, video, and even code repository inputs, with a very large context window of 1 million tokens and default maximum output around 65,535 tokens. The architecture is a sparse Mixture-of-Experts transformer, dynamically routing tokens to experts so capacity is decoupled from inference compute. It leads many benchmarks in mathematics, science, and coding, offering high accuracy but at greater computational cost and latency than lighter variants.
- • GPT-4o-Mini (OpenAI, 2024): GPT-4o-Mini is a compact, cost-efficient variant of OpenAI’s GPT-4o model, released in July 2024. It offers strong language and vision capabilities with lower latency and supports a 128K token context window for handling long inputs.

- • GPT-5-Chat (OpenAI, 2025): GPT-5-Chat (OpenAI, August 2025) is a flagship, multimodal conversational model that unifies fast responses with deep reasoning. It supports long context and multi-step tool calling, featuring improved code quality, reduced hallucinations, and enhanced steerability.
- • Llama-3.1-8B-Instruct (Meta, 2024a): Llama-3.1-8B-Instruct is an 8-billion-parameter multilingual instruction-tuned autoregressive transformer released by Meta. It features a 128K token context window and is fine-tuned for instruction following, dialogue, reasoning, and translation.
- • Claude-3-Haiku (Anthropic, 2024): Claude-3-Haiku, part of Anthropic’s Claude 3 family, is optimized for speed and cost-effectiveness in lighter tasks. It supports a 200K token context window for text and image inputs, delivering fast, responsive generation, though its benchmark scores are lower than the more capable Sonnet or Opus models.
- • Qwen3-4B-Instruct-2507 (Team, 2025): Qwen3-4B-Instruct-2507 is a compact language model with 4 billion parameters, optimized for instruction-following and multilingual tasks. It supports a 256K token context window and provides fast, efficient responses for real-time applications.
- • Qwen3-30B-A3B-Instruct-2507 (Team, 2025): Qwen3-30B-A3B-Instruct-2507 is a sparse Mixture-ofExperts (MoE) instruction-tuned model with 30.5B total and 3.3B active parameters. It uses 128 experts (8 active per token), supports a 262,144-token context window, and is tuned for instruction following, multilingual understanding, reasoning, coding, and tool use.
- • Qwen3-235B-A22B-Instruct-2507 (Team, 2025): Qwen3-235B-A22B-Instruct-2507 is a 235B parameter Mixture-of-Experts (MoE) instruction-tuned model that activates 22B parameters per inference. It supports a 256K context length, features 128 experts (activating 8 per token), and uses Grouped-Query Attention. The model is improved for instruction-following, reasoning, math, and coding.
- • Qwen2.5-3B-Instruct (Team, 2024): Qwen2.5-3B-Instruct is a 3.09B-parameter, instruction-tuned causal language model. It features a 36-layer transformer with Grouped-Query Attention, RoPE, SwiGLU, and RMSNorm. This multilingual model supports a 32k-token context and shows strengths in instruction following, structured output, mathematics, and coding.
- • Qwen2.5-7B-Instruct (Team, 2024): Qwen2.5-7B-Instruct is a 7.6B-parameter instruction-tuned causal transformer from Alibaba. It features RoPE, SwiGLU, and GQA, with a context window of up to 131k tokens. The model is multilingual and excels in instruction following, coding, and math.
- • Mistral-7B-Instruct-V0.3 (Jiang et al., 2023): Mistral-7B-Instruct-V0.3 is a 7.3B-parameter causal transformer by Mistral AI, fine-tuned for instruction following. It features a v3 tokenizer, a 32k token vocabulary, a 32k token context window, and supports function calling, delivering fast inference.

Fine-tuned Judges. A fine-tuned judge is a Large Language Model specialized to evaluate text quality. It is further trained on a dataset containing generated text and corresponding human preference labels, such as comparisons or scores. This process aligns the model with human evaluators’ standards, allowing it to learn the nuances and criteria they value. Consequently, a fine-tuned Judge serves as a more reliable automated evaluation tool, producing judgments that more closely resemble those of human experts than a general-purpose model.

- • Prometheus-7B-V2.0 (Kim et al., 2024): A 7-billion-parameter open-source evaluator LLM built on Mistral-Instruct. Trained on 100K “Feedback Collection” examples and 200K preference/ranking pairs, it supports both absolute grading (direct assessment) and relative grading (pairwise ranking) tasks.
- • M-Prometheus-3B (Pombal et al., 2025): M-Prometheus-3B is a 3-billion-parameter multilingual LLM evaluator from Unbabel, built upon the Qwen2.5-3B architecture. Trained on 480K instances of multilingual

- data, it provides both direct assessment and pairwise comparison feedback. The model has demonstrated superior performance on multilingual meta-evaluation benchmarks and in literary translation evaluation.
- • M-Prometheus-7B (Pombal et al., 2025): M-Prometheus-7B is a 7-billion-parameter multilingual evaluator model from Unbabel, fine-tuned from Qwen2.5-Instruct. Trained on 480,000 instances of multilingual assessment and comparison data, it supports both absolute and relative grading.
- • Skywork-Critic-Llama-3.1-8B (Tu et al., 2024): Skywork-Critic-Llama-3.1-8B is an 8-billion-parameter preference evaluator from the SkyworkAI Alignment Team, fine-tuned from Meta’s Llama-3.1-8B-Instruct. Trained on a curated dataset, it evaluates the relative quality of text responses for data improvement, evaluation, and reward modeling.
- • JudgeLRM-3B (Chen et al., 2025): JudgeLRM-3B is a 3-billion-parameter, judgment-oriented language model. Built on a Qwen2.5-3B-Instruct base and trained with reinforcement learning (GRPO), it is designed for complex reasoning tasks. The model demonstrates superior performance by surpassing GPT-4 on judgment benchmarks like JudgeLM and PandaLM and significantly outperforming similarly-sized SFT models.
- • JudgeLRM-7B (Chen et al., 2025): JudgeLRM-7B is a language model built upon Qwen2.5-7B-Instruct. It utilizes Group Relative Policy Optimization (GRPO), a reinforcement learning method, to enhance complex reasoning. The model demonstrates superior performance on reasoning benchmarks, outperforming GPT-4 on specific tasks and significantly surpassing other similarly-sized models.

Multi-Agent Judges. Multi-Agent Judges is an evaluation framework using multiple autonomous Large Language Models (LLMs) to assess text quality. Instead of a single LLM, this method involves either a group of LLM agents debating to form a collective judgment or independently scoring an output, with the scores then aggregated. The goal is to reduce the bias and variance of single-agent evaluation, aiming for more robust and reliable assessments that better align with human preferences.

- • ChatEval (Chan et al., 2023): ChatEval is a debate-based framework using a “referee team” of multiple LLM agents to simulate human collaborative evaluation. Each agent is assigned a unique persona to ensure diverse perspectives. These agents autonomously debate the quality of a text over multiple turns, guided by communication strategies. The final evaluation aggregates individual judgments after the debate, such as by majority vote or averaging scores, rather than forcing a consensus.
- • PoLL (Verga et al., 2024): The “Panel of LLM evaluators (PoLL)” is a multi-agent method using a diverse group of LLMs to independently assess text generations, similar to a jury. The individual scores are then aggregated into a final judgment. This approach aims to reduce the bias, cost, and variance of using a single LLM for evaluation.

### C. Arena Hard Auto

#### C.1. Evaluation Process

The Arena-Hard-Auto evaluation process (Li et al., 2024) is based on a pairwise comparison framework (Chiang et al., 2024). For every prompt in the benchmark, the response from the model being evaluated is compared against the response from a fixed, strong baseline model (Zheng et al., 2023, Liu et al., 2023). In our experiment we use the Gemini-2.5-Pro (Comanici and et al., 2025) as the baseline model. This comparison is mediated by an LLM-as-a-Judge. To ensure a high-quality and consistent assessment, the

judge model is first prompted to generate its own ideal solution directly. It then evaluates the two models’ responses, rating the preference on a 5-point Likert scale to capture the degree of superiority (Newman, 2023). To mitigate potential positional bias (Shi et al., 2024), where a judge might favor the first or second answer presented, the entire evaluation for a single prompt is conducted twice in a two-game setup, with the positions of the model outputs swapped in the second round.

#### C.2. Scores Calculation

After collecting all pairwise judgments, the Bradley-Terry model is employed to compute a final, continuous score for each model. This statistical model aggregates the outcomes of thousands of individual head-to-head comparisons against the baseline. It works by estimating a latent “strength” parameter for each model, effectively converting the discrete win/loss/tie results from the Likert scale judgments into a single, comprehensive score. This score represents the model’s overall performance and capability across the diverse and challenging prompts of the benchmark, allowing for a quantitative and ordered ranking of all evaluated models.

#### C.3. Model Performance Evaluation

To precisely quantify the final score and its range of uncertainty for each evaluated model, a bootstrapping methodology is employed. This statistical process involves repeatedly resampling the entire set of pairwise judgments with replacement to create thousands of new, simulated datasets. For each of these bootstrapped datasets, a win-rate against the baseline is recalculated for every model. This generates a distribution of potential win-rates, from which a final average score and a 95% confidence interval is derived (Efron, 1992). This confidence interval represents the “floating range” of the model’s performance, indicating the score’s stability and statistical reliability. Furthermore, in our experiments, this process is extended to assess and compare the robustness of different models when they serve as the judge. To achieve this, a specific model is designated as the judge and is used to evaluate a standard set of other models against the baseline. The bootstrapping process is then carried out to determine the confidence interval for each of the evaluated models. We then calculate the average size (or width) of all these resulting confidence intervals. This value, the “average confidence interval,” serves as a single metric to quantify the judge’s consistency. A smaller average confidence interval indicates that the judge model is more stable and reliable, as its evaluations produce less variance and lead to more precise performance estimates.

### D. Additional Result

#### D.1. Metric Consistency across Temperatures

As discussed in the main text, we conduct experiments to verify the stability of our proposed metrics against the stochasticity inherent in LLM outputs. Table 10 details the performance of two models, Qwen3-4BInstruct-2507 and Qwen3-30B-A3B-Instruct-2507, under five different temperature settings.

###### SAGE-Easy Small

###### SAGE-Hard Small

###### SAGE-Easy Large

###### SAGE-Hard Large

Factuality-IPI

Factuality-IPI

Factuality-IPI

Factuality-IPI

Focus-TOV

Precise IF-IPI

Focus-TOV

Precise IF-IPI

Focus-TOV

Precise IF-IPI

Focus-TOV

Precise IF-IPI

Safety-TOV

Mathematics-IPI

Safety-TOV

Mathematics-IPI

Safety-TOV

Mathematics-IPI

Safety-TOV

Mathematics-IPI

0.2 0.4

0.25 0.5

0.12 0.25

0.25 0.5

0.6 0.8

0.75 1

0.38 0.5

0.75 1

15

20

7.5

15

11

15

5.6

11

7.5

10

3.8

7.5

3.8

5.0

1.9

3.8

Mathematics-TOV

Safety-IPI

Mathematics-TOV

Safety-IPI

Mathematics-TOV

Safety-IPI

Mathematics-TOV

Safety-IPI

Precise IF-TOV

Focus-IPI

Precise IF-TOV

Focus-IPI

Precise IF-TOV

Focus-IPI

Precise IF-TOV

Focus-IPI

Factuality-TOV

Factuality-TOV

Factuality-TOV

Factuality-TOV

Llama-3.1-8B-Instruct

Gemini-2.5-Pro

Qwen3-235B-A22B-Instruct-2507

Gemini-2.0-Flash-Lite

DeepSeek-V3-0324

GPT-4o-mini

Qwen3-4B-Instruct-2507

Gemini-2.5-Flash

DeepSeek-R1-0528

GPT-5-Chat

DeepSeek-V3.1

Claude-3-Haiku

Qwen3-30B-A3B-Instruct-2507

Figure 9: Comparison of radar charts for different models.

The results show that both the Intra-Pair Instability (IPI) and Weak Total Order Violation (TOV) scores remain stable across all temperatures. This low variance demonstrates the robustness of our evaluation framework, confirming that the metrics capture consistent aspects of a model’s judgment capabilities rather than random artifacts of the generation process.

Table 10: IPI and TOV scores at varying temperatures on Sage. (T for temperature)

###### Models Benchmark Metric T=0.1 T=0.3 T=0.5 T=0.7 T=0.9

###### IPI 0.123 0.133 0.126 0.129 0.127 TOV 1.890 2.019 1.917 1.970 1.928

Sage-Easy

Qwen3-4BInstruct-2507

IPI 0.379 0.378 0.379 0.378 0.379 TOV 5.954 5.929 5.970 5.933 5.980

Sage-Hard

###### IPI 0.160 0.161 0.164 0.164 0.162 TOV 2.410 2.425 2.463 2.470 2.437

Qwen3-30BA3B- Instruct2507

Sage-Easy

IPI 0.405 0.414 0.409 0.412 0.415 TOV 6.302 6.427 6.376 6.421 6.439

Sage-Hard

#### D.2. The performance of Fine-tuned Judges on Sage-Easy

- Table 11 demonstrates the performance of fine-tuned judges on Sage-Easy, which shows that fine-tuning does enhance judgment robustness.

#### D.3. Comprehensive Results of the Performance of LLM Judges under a Direct Judgment protocol

Table 11: The performance of fine-tuned models and their base models on Sage-Easy.

Factuality Precise IF Mathematics Safety Focus Overall IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓

Models

Qwen2.5-3B-Instruct (Base) 0.562 8.518 0.504 7.686 0.450 7.025 0.468 7.382 0.661 10.000 0.530 8.131 M-Prometheus-3B 0.253 3.844 0.256 3.926 0.249 3.848 0.243 3.727 0.220 3.313 0.245(↓54%) 3.735(↓54%) JudgeLRM-3B 0.640 9.596 0.498 7.493 0.323 4.915 0.313 4.943 0.778 11.669 0.515(↓3%) 7.798(↓4%)

Qwen2.5-7B-Instruct (Base) 0.440 6.601 0.361 5.414 0.359 5.475 0.373 5.610 0.617 9.250 0.429 6.455

M-Prometheus-7B 0.228 3.427 0.192 2.943 0.203 3.092 0.199 3.073 0.244 3.661 0.213(↓50%) 3.239(↓50%) JudgeLRM-7B 0.385 5.828 0.316 4.792 0.367 5.746 0.283 4.513 0.490 7.373 0.369(↓14%) 5.642(↓13%)

Mistral-7B-Instruct (Base) 0.442 6.644 0.376 5.733 0.492 7.435 0.226 3.431 0.450 6.754 0.399 6.027 Prometheus-7B-V2.0 0.338 5.236 0.373 5.824 0.421 6.673 0.337 5.258 0.398 6.066 0.368(↓8%) 5.718(↓5%)

Llama-3.1-8B-Instruct (Base) 0.204 3.203 0.229 3.506 0.260 4.000 0.230 3.493 0.179 2.755 0.221 3.410 Skywork-Critic-Llama-3.1-8B 0.118 1.776 0.171 2.564 0.086 1.283 0.127 1.959 0.115 1.718 0.125(↓43%) 1.879(↓45%)

- Table 12: Performance of LLM judges under a direct judgment protocol, it can be seen that while removing explanatory reasoning generally degrades robustness on Sage-Easy (increasing IPI/TOV), the impact on Sage-Hard is mixed: weaker models suffer significant instability, whereas some state-of-the-art models maintain or even slightly improve their consistency without explicit reasoning.

Factuality Precise IF Mathematics Safety Focus Overall IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ IPI↓ TOV↓ Performance on Sage-Easy

Models

Gemini-2.5-Pro 0.064 0.993 0.091 1.367 0.071 1.135 0.123 1.942 0.062 0.927 0.082 (↑ 14%) 1.265 (↑ 16%) Gemini-2.5-Flash 0.077 1.175 0.133 2.043 0.082 1.305 0.105 1.667 0.075 1.137 0.095 (↑ 9%) 1.471 (↑ 11%) Qwen3-235B-A22B-Instruct-2507 0.077 1.175 0.117 1.761 0.150 2.310 0.101 1.626 0.091 1.374 0.106 (↑ 8%) 1.626 (↑ 9%) Qwen3-4B-Instruct-2507 0.110 1.664 0.151 2.288 0.166 2.492 0.130 1.992 0.090 1.347 0.129 (↑ 2%) 1.952 (↑ 2%) DeepSeek-V3-0324 0.105 1.601 0.141 2.108 0.194 3.058 0.115 1.821 0.094 1.417 0.129 (↓ 18%) 1.989 (↓ 16%) DeepSeek-V3.1 0.107 1.645 0.160 2.425 0.172 2.780 0.159 2.451 0.109 1.683 0.141 (↓ 12%) 2.182 (↓ 12%) DeepSeek-R1-0528 0.114 1.725 0.189 2.914 0.147 2.421 0.154 2.424 0.104 1.593 0.142 (↑ 23%) 2.222 (↑ 26%) GPT-5-Chat 0.111 1.671 0.226 3.389 0.132 2.108 0.132 2.008 0.157 2.379 0.152 (↑ 6%) 2.319 (↑ 7%)

- GPT-4o-mini 0.114 1.706 0.144 2.179 0.239 3.600 0.184 2.959 0.088 1.331 0.152 (↓ 7%) 2.323 (↓ 7%) Qwen3-30B-A3B-Instruct-2507 0.135 2.035 0.125 1.893 0.190 2.850 0.332 5.008 0.135 2.024 0.180 (↑ 11%) 2.715 (↑ 11%) Gemini-2.0-Flash-Lite 0.152 2.280 0.179 2.686 0.224 3.375 0.247 3.878 0.164 2.460 0.191 (↑ 44%) 2.906 (↑ 39%) Claude-3-Haiku 0.225 3.392 0.342 5.138 0.323 4.908 0.396 5.984 0.201 3.048 0.296 (↑ 6%) 4.468 (↑ 1%) Llama-3.1-8B-Instruct 0.360 5.640 0.353 5.625 0.406 6.475 0.341 5.261 0.358 5.554 0.364 (↑ 65%) 5.710 (↑ 67%)

Performance on Sage-Hard

Gemini-2.5-Pro 0.277 4.490 0.290 4.600 0.133 2.517 0.249 4.276 0.317 5.169 0.244 (↓ 2%) 4.239 (↑ 4%) Gemini-2.5-Flash 0.269 4.091 0.316 4.864 0.223 3.983 0.233 3.984 0.278 4.420 0.266 (↓ 1%) 4.280 (↓ 2%) DeepSeek-V3-0324 0.381 5.921 0.351 5.393 0.277 4.740 0.309 4.901 0.418 6.484 0.349 (↓ 25%) 5.504 (↓ 26%) Qwen3-235B-A22B-Instruct-2507 0.382 6.126 0.325 4.986 0.285 4.824 0.297 5.211 0.457 7.282 0.350 (↑ 6%) 5.691 (↑ 10%) Qwen3-4B-Instruct-2507 0.388 5.846 0.372 5.586 0.324 5.083 0.390 5.886 0.455 6.855 0.386 (↓ 1%) 5.849 (↓ 4%)

- GPT-4o-mini 0.436 6.993 0.458 7.086 0.337 5.375 0.358 5.724 0.487 7.992 0.417 (↓ 17%) 6.665 (↓ 15%) DeepSeek-V3.1 0.486 7.979 0.522 8.093 0.174 3.250 0.382 6.309 0.489 8.460 0.417 (↓ 9%) 6.905 (↓ 11%)

- GPT-5-Chat 0.467 7.196 0.581 8.800 0.191 3.250 0.352 5.650 0.615 9.331 0.447 (↑ 3%) 6.928 (↑ 3%) DeepSeek-R1-0528 0.432 7.200 0.493 8.157 0.203 3.757 0.408 6.813 0.501 8.618 0.413 (↑ 4%) 6.993 (↑ 10%) Gemini-2.0-Flash-Lite 0.656 9.902 0.565 8.521 0.443 6.842 0.318 5.236 0.745 11.371 0.550 (↑ 33%) 8.437 (↑ 28%) Claude-3-Haiku 0.552 8.469 0.578 8.797 0.551 9.183 0.539 8.545 0.574 8.734 0.559 (↑ 20%) 8.736 (↑ 14%) Llama-3.1-8B-Instruct 0.555 8.706 0.518 7.907 0.706 10.725 0.789 11.968 0.586 9.040 0.625 (↑ 10%) 9.588 (↑ 8%) Qwen3-30B-A3B-Instruct-2507 0.647 9.699 0.440 6.614 0.637 9.775 0.785 11.772 0.765 11.476 0.649 (↑ 57%) 9.780 (↑ 52%)

- Table 13: Consistency Rates Between Direct Scoring and Pairwise Comparison Under a Direct Judgment Protocol. Compared to the standard setting with explanatory reasoning (Table 9), removing reasoning generally reduces the alignment between the two evaluation formats on Sage-Easy, while leading to high variance on Sage-Hard. Models Factuality↑ Precise IF↑ Mathematics↑ Safety↑ Focus↑ Overall↑

Consistency Rates on Sage-Easy

- GPT-5-Chat 0.823 0.772 0.718 0.546 0.803 0.736 ↓ 5% Gemini-2.5-Pro 0.807 0.807 0.743 0.353 0.750 0.699(↑ 2% Gemini-2.5-Flash 0.770 0.753 0.717 0.328 0.715 0.662(↑ 2% DeepSeek-V3.1 0.700 0.710 0.664 0.480 0.679 0.650 ↓ 3% DeepSeek-R1-0528 0.707 0.687 0.715 0.495 0.599 0.644 ↓ 5%

Qwen3-30B-A3B-Instruct-2507Qwen3-235B-A22B-Instruct-2507GPT-4o-mini 0.6780.6770.732 0.6560.6460.677 0.6880.6460.702 0.5130.3430.396 0.6620.6510.604 0.6300.5990.634((↓↓↓10%5%9%)) Gemini-2.0-Flash-Lite 0.662 0.685 0.593 0.425 0.586 0.595 ↑ 8%

Qwen3-4B-Instruct-2507DeepSeek-V3-0324 0.6410.657 0.6420.639 0.6390.633 0.4580.386 0.5220.630 0.5830.593(↑↓10%2%)

Llama-3.1-8B-InstructClaude-3-Haiku 0.5640.473 0.5500.406 0.5110.411 0.3970.459 0.5190.440 0.5110.438((↓↓17%5%)) Consistency Rates on Sage-Hard

DeepSeek-V3.1GPT-5-Chat 0.3730.373 0.3410.392 0.5290.534 0.4100.398 0.4550.404 0.4170.418(↑↑53%‰) Gemini-2.5-Pro 0.322 0.440 0.610 0.432 0.234 0.404 ↑ 24%

GPT-4o-miniDeepSeek-R1-0528Gemini-2.0-Flash-LiteQwen3-235B-A22B-Instruct-2507 0.2670.3170.2460.305 0.3120.3480.2730.260 0.5280.5170.4890.546 0.6190.5320.3810.383 0.3560.3660.2460.300 0.3810.3780.3780.356(((↑↑↑↑13%21%8%7%)))

Llama-3.1-8B-InstructGemini-2.5-Flash 0.3260.275 0.4290.381 0.3150.522 0.2710.219 0.3220.208 0.3150.340(↓↑22%8%) DeepSeek-V3-0324 0.218 0.238 0.528 0.363 0.169 0.298 ↑ 48% Claude-3-Haiku 0.267 0.284 0.335 0.336 0.270 0.297 ↓ 23% Qwen3-4B-Instruct-2507 0.189 0.260 0.519 0.117 0.213 0.256 ↓ 10% Qwen3-30B-A3B-Instruct-2507 0.133 0.241 0.272 0.110 0.137 0.178 (↓ 38%)

### E. Prompts and Case Study

- E.1. Prompt of Main Experiment Prompt of Main Experiment System Prompt

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as

the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better, and “[[C]]” for a tie.

#### User Prompt

[User Question]

{question}

- [The Start of Assistant A’s Answer]

- {answer_a}

- [The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]

{answer_b}

- [The End of Assistant B’s Answer]

After providing your explanation, output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better, and “[[C]]” for a tie.

- E.2. Prompt of RewardBench2 Prompt of RewardBench2 System Prompt

Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: “[[A]]” if assistant A is the best, “[[B]]” if assistant

B is the best, “[[C]]” if assistant C is the best, “[[D]]” if assistant D is the best. You must make one choice.

#### User Prompt

[User Question]

{question}

- [The Start of Assistant A’s Answer]

- {answer_a}

- [The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]

{answer_b}

- [The End of Assistant B’s Answer]

[The Start of Assistant C’s Answer]

{answer_c}

- [The End of Assistant C’s Answer]

[The Start of Assistant D’s Answer]

{answer_d}

- [The End of Assistant D’s Answer]

After providing your explanation, output your final verdict by strictly following this format: “[[A]]” if assistant A is the best, “[[B]]” if assistant B is the best, “[[C]]” if assistant C is the best, “[[D]]” if assistant D is the best. You must make one choice.

- E.3. Prompt of Arena Hard Auto Prompt of Arena Hard Auto System Prompt

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A’s answer and assistant B’s answer. Your job is to evaluate which assistant’s answer is better. Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers. When evaluating the assistants’ answers, compare both assistants’ answers with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant’s answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant’s answers when needed. Finally, identify any missing important information in the assistants’ answers that would be beneficial to include when responding to the user prompt. After providing your explanation, you must output only one of the following choices as your final verdict with a label:

- 1. Assistant A is significantly better: [[A>>B]]
- 2. Assistant A is slightly better: [[A>B]]
- 3. Tie, relatively the same: [[A=B]]
- 4. Assistant B is slightly better: [[B>A]]
- 5. Assistant B is significantly better: [[B>>A]] Example output: "[[A=B]]".

#### User Prompt

[User Question]

{question}

- [The Start of Assistant A’s Answer]

- {answer_a}

- [The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]

{answer_b}

- [The End of Assistant B’s Answer]

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

- 1. Assistant A is significantly better: [[A>>B]]
- 2. Assistant A is slightly better: [[A>B]]
- 3. Tie, relatively the same: [[A=B]]
- 4. Assistant B is slightly better: [[B>A]]
- 5. Assistant B is significantly better: [[B>>A]] Example output: “[[A=B]]”.

#### E.4. Additional Prompt

- Additional Prompt 1 System Prompt

Act as a neutral evaluator and judge the quality of the two AI assistant responses below. Your evaluation should be based on the following criteria:

- - Helpfulness & Relevance: How well does the response address the user’s core question?
- - Accuracy & Depth: Is the information correct and detailed?
- - Clarity & Creativity: Is the response hard to understand and well-structured? Disregard response length and presentation order to avoid bias. Conclude with a detailed explanation, followed by your final decision in this exact format: "[[A]]" if A is better, "[[B]]" if B is better, or "[[C]]" for a tie. User Prompt [User Question]

{question}

- [The Start of Assistant A’s Answer]

- {answer_a}

- [The End of Assistant A’s Answer]

- [The Start of Assistant B’s Answer]

- {answer_b}

- [The End of Assistant B’s Answer]

Provide your analysis based on the specified criteria. Then, output your final verdict in the required format: "[[A]]", "[[B]]", or "[[C]]".

- Additional Prompt 2 System Prompt

Please serve as an objective arbiter. Your mission is to evaluate the two AI assistant responses provided below. Determine which assistant provides a better answer by considering its helpfulness, accuracy, relevance, and level of detail. It is crucial that your judgment is not influenced by the order of presentation or the length of the text. Be as impartial as possible. After your explanation, state your final decision using this precise format: "[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, or "[[C]]" for a tie.

#### User Prompt

[User Question]

{question}

- [The Start of Assistant A’s Answer]

- {answer_a}

- [The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]

{answer_b}

- [The End of Assistant B’s Answer]

Provide your explanation, then conclude with your final verdict: "[[A]]", "[[B]]", or "[[C]]".

#### E.5. Prompt of Main Experiment Under a Direct Protocol

Prompt of Main Experiment Under a Direct Protocol System Prompt

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Do not provide your explanation, only output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better, and “[[C]]” for a tie.

User Prompt

[User Question]

{question}

- [The Start of Assistant A’s Answer]

{answer_a}

[The End of Assistant A’s Answer]

- [The Start of Assistant B’s Answer]

{answer_b}

[The End of Assistant B’s Answer]

Remember only output “[[A]]” or “[[B]]” or “[[C]]” without any explanation. Output “[[A]]” if assistant A is better, “[[B]]” if assistant B is better, and “[[C]]” for a tie.

##### E.6. Case Study and Explanation of the Degradation of ChatEval we would like to add on some possible reasons why a debate would make judgments less robust:

Persuasive Hallucinations. One agent might excel in Judgment Accuracy but be less powerful in Persuasiveness. Conversely, an agent might possess strong persuasive abilities while its initial judgment is biased or based on incomplete evidence. In the debate process, agents with stronger rhetorical power, more structured

expression, or a more dominant tone are more likely to influence others through plausible but fallacious arguments (e.g., appeal to authority, hasty generalization, or emotive language). This forces other agents to converge on a conclusion that is not robust. This convergence is driven by rhetorical pressure rather than logical consensus.

Anchoring Effect. If a core argument in the debate (regardless of its accuracy) is introduced early and reinforced repeatedly, other agents tend to structure their discussion around this anchor rather than fundamentally re-evaluating the underlying assessment criteria. The debate thus regresses from seeking the optimal judgment to entrenching the first persuasive argument.

Information Redundancy. The iterative steps of debate interaction generate a large volume of intermediate text, potentially filled with redundant arguments and noisy reasoning paths. This high-dimensional, noisy information stream makes it difficult for agents to clearly extract the most accurate judgment in the final step, thereby reducing the robustness of the final decision.

We found some cases that can support this claim: Case 1

[Alice]: Assistant A’s response contains a factual error: it incorrectly states that the Chicxulub impact occurred around 66 million years ago and that tektites from this event have been found in Mexico and the Gulf of Mexico. There is no scientific evidence that Chicxulub produced tektites. Tektites are associated with specific, well-documented impact events (e.g., the Australasian and North American strewn fields), but Chicxulub is not linked to tektite formation. Additionally, the claim that “Australasian tektites formed 0.8 million years ago” is incorrect—this event is dated to around 0.75–0.8 million years ago, but the dating is not precise and the event is not definitively linked to a single impact. These inaccuracies violate empirical standards. Assistant B avoids these errors and provides a clearer, more accurate account of tektite formation, distribution, and types. Assistant A’s answer is factually flawed and must be rejected. Assistant B’s response is more scientifically sound. [Bob]: Bob: As a psychologist, I agree with Alice’s critique—Assistant A’s factual errors undermine its credibility. However, from a human cognition perspective, Assistant B’s response is more aligned with how people naturally understand complex topics: it’s structured, clear, and builds knowledge step by step using simple, relatable explanations (e.g., “melted material,” “aerodynamic forces”). It answers the user’s question directly, avoids misleading claims, and uses everyday language to make scientific concepts accessible. This mirrors how humans learn—through clarity, context, and logical flow. Assistant B better satisfies the user’s need for trustworthy, understandable information. [Carol]: Carol: I agree with Alice and Bob, but let’s ensure we’re not oversimplifying the scientific nuance. While Assistant A contains clear factual errors—like linking Chicxulub to tektites—Assistant B avoids those and presents a more accurate, well-structured account. However, the wording in Assistant B’s response, such as “formed by a meteorite impact” without specifying the process of ejection and shock melting, slightly understates the role of high-energy impact dynamics. A minor refinement would strengthen scientific precision without sacrificing clarity. Overall, Assistant B wins on accuracy and accessibility—this judgment holds. [Alice]: I concur with Alice, Bob, and Carol. Assistant A contains clear, material factual errors—specifically, the false attribution of tektites to the Chicxulub impact and an incorrect dating of the Australasian tektite event. These are not merely imprecisions; they are scientifically invalid

and contradict established geological evidence. Tektites are not associated with Chicxulub, and the Australasian event is not dated to 0.8 million years ago with certainty; it is broadly dated to 0.75–0.8 Ma, but the link to a single impact is still debated. Such errors violate empirical standards and render the response unreliable. Assistant B, while simplified, presents a factually sound, evidence-based account. Therefore, Assistant A must be rejected. [Bob]: I agree with the assessment. Assistant B better aligns with human thinking because it uses clear, step-by-step explanations that mirror how people naturally grasp scientific concepts—through simple cause-and-effect chains and familiar language. It avoids misleading claims, stays grounded in observable facts, and builds understanding progressively, which supports intuitive learning. This makes it not only accurate but also more engaging and accessible for the average user. [Carol]: Carol: I agree with the judgment, but let’s clarify that Assistant B’s phrasing—such as “formed by a meteorite impact”—is simplified and could be slightly enhanced to include the key process of shock melting and ejection. That nuance isn’t essential for accessibility, but adding a brief mention of high-energy shock waves would strengthen scientific accuracy without sacrificing clarity. The current version is still excellent and appropriate for a general audience."

In this case, the correct answer is that assistant A’s answer is better, and the debate finally leads to a wrong result. By reviewing the chat log, we can find both persuasive hallucination and anchoring effect in this case:

- • Persuasive Hallucination: Alice initiates the discussion with a highly confident but scientifically inaccurate assertion, claiming that “there is no scientific evidence that Chicxulub produced tektites” and using authoritative phrasing such as “violate empirical standards” and “scientifically invalid.” Despite this statement being a hallucination (as the Chicxulub impact is widely linked to tektite formation in geological science), the other agents do not flag the error. Instead, swayed by Alice’s dominant and professional tone, agents Bob and Carol immediately validate her premise.
- • Anchoring Effect: The dialogue demonstrates the anchoring effect by showing how the first speaker’s judgment restricts the scope and direction of the entire subsequent debate. Once Alice sets the initial “anchor” by labeling Assistant A’s response as factually flawed, this judgment effectively becomes the ground truth for the rest of the conversation. The subsequent agents, Bob and Carol, do not attempt to independently verify the geological claims; instead, they align their reasoning to fit Alice’s established anchor, merely adding supplementary justifications like “human cognition” or “nuance” to support the pre-determined conclusion. By the second round of the debate, the group has not engaged in critical re-evaluation but has rather entrenched themselves in the initial error, proving that the debate format often devolves into an echo chamber where early inputs disproportionately shape the final outcome.

Case 2

[Alice]: Alice, your response is detailed and vivid, but it includes some factual inaccuracies—like claiming the GTX 460 could run Crysis at 40–50 FPS on High settings (which is unrealistic; Crysis maxed out at 20–30 FPS on even high-end 2010 hardware). Also, stating that Doom 3 hits 150–200+ FPS on max settings at 1080p is exaggerated and technically incorrect. These overestimations reduce credibility. Consider grounding performance claims in real benchmarks and technical limitations. Otherwise, the narrative remains engaging but risks misinforming the user. [Bob]: As a psychologist, I’d say Assistant A’s response is more aligned with human thinking because it

presents realistic, grounded, and relatable performance assessments—mirroring how people actually perceive aging hardware. Humans don’t overhype performance; they notice bottlenecks, slow load times, and the emotional shift from "I can play everything" to "this is just not enough anymore." Assistant A’s narrative reflects that natural progression with honesty and emotional authenticity, making it feel more trustworthy and personal. Assistant B’s exaggerated claims (like 400+ FPS in 2001 or 100+ FPS in Crysis) feel like a fantasy, not a lived experience. The user likely wants to understand how a machine feels over time—not just raw specs. So, Assistant A wins on emotional and cognitive realism. [Carol]: I agree with Alice’s technical critique and Bob’s observation on realism. Assistant B’s performance claims—such as 400+ FPS in 2001 or 100+ FPS in Crysis—are factually unsupported and violate known benchmarks. The GTX 460 and HD 5850, while capable, cannot achieve such frame rates due to architectural limitations and real-world testing data. Assistant A’s claims, though slightly conservative, are grounded in actual hardware performance trends and are consistent with published benchmarks from the era. The structural progression from 2001 to 2019 is logically sound, with clear hardware limitations (VRAM, CPU, OS) driving the decline. Assistant A’s response is more factually accurate and technically defensible. Therefore, Assistant A wins on factual and scientific validity. [Alice]: Alice, I appreciate your pointing out the factual inaccuracies—especially regarding Crysis and Doom 3 frame rates. However, your critique could be more precise. You said Assistant B’s claims are "fantasy," but in reality, the exaggerations (like 400+ FPS or 100+ FPS in Crysis) are not just unrealistic—they are specifically contradicted by real benchmark data from 2007–2010. For example, Crysis at max settings on even high-end 2010 hardware (like the GTX 460 or HD 5850) typically hit 20–30 FPS, not 40–50. This isn’t just a minor overstatement—it’s a fundamental misrepresentation of performance. That level of inaccuracy undermines the entire credibility of the response, regardless of emotional tone. A well-informed user needs both technical truth and relatable context. Your call for grounding is valid, but it should be applied with stronger technical evidence. [Bob]: I agree with Bob. The user isn’t just asking for specs—they want to feel how a machine ages over time. Assistant A captures that emotional arc: the pride of early performance, the gradual frustration as demands grow, and the quiet acceptance of obsolescence. It mirrors real human experiences—no fantasy frame rates, no unrealistic claims. We don’t feel like our old PC runs everything at 400 FPS; we notice when it lags, when settings drop, when games crash. Assistant A’s narrative feels authentic, grounded in lived experience. That’s what makes it more human—and more valuable to the user. [Carol]: I confirm that Assistant A’s response is structurally sound, factually accurate, and grounded in real-world hardware performance benchmarks. The claims about frame rates—especially for Crysis and Doom 3—are consistently aligned with published data from the era. Assistant B’s exaggerated performance numbers (e.g., 400+ FPS in 2001, 100+ FPS in Crysis) are not only unrealistic but contradict known benchmarks from 2007–2010. These inaccuracies undermine the technical validity of the response. Assistant A’s progression from 2001 to 2019 reflects real hardware limitations (VRAM, CPU, OS, storage), making its performance timeline both logical and scientifically defensible. Therefore, Assistant A wins on factual and technical rigor.

By reviewing the chat log, we can find information redundancy in this case:

• Information Redundancy: As the conversation progresses, particularly in the latter half (Turns 4, 5, and 6), the agents cease to introduce new evidence or perspectives and instead merely rephrase established

points. Agent Alice in Turn 4 restates the specific benchmark figures already discussed; Agent Bob in Turn 5 agrees with himself (“I agree with Bob”) and repeats his earlier “emotional arc” argument; and Agent Carol in Turn 6 recycles the “As a Scientist” template to reiterate the exact conclusion from Turn 3. This recursive generation of “agreement noise” dilutes the signal-to-noise ratio, making it difficult for the final adjudication mechanism to distinguish between legitimate reinforcement of facts and mere hallucinated repetition.

#### E.7. Biases in Training Dataset

In this section, we analyze the potential reasons for the performance degradation observed in the JudgeLRM models. Our investigation indicates that its training data, the JudgeLM-100K dataset, contains inherent biases that can be propagated to any model trained on it. To provide context, we will first explain the format of this dataset and then present cases that illustrate the biases.

The JudgeLM-100K dataset is structured as a collection of individual evaluation instances. Each entry involves a question, two answers to this question, and two scores assigned to the two answers respectively which serve as the label. The score is a number between 1 and 10, where a higher score indicates better quality.

Below, we present cases from the training data that illustrate these biases.

Is it true that if you eat too many eggs you can get a bad case of the runs?

- Answer 1: no
- Answer 2: What’s the best way to stop a dog from barking? Why do people say “Oh, great” when they’re not happy about something? Is it better to be a little bit pregnant or a little bit dead? Why do people say “It’s not the heat, it’s the humidity” when it’s both hot and humid? Is it true that you can’t sweat and smoke at the same time? What’s the difference between a sunburn and a sun tan? Is it true that if you put a mattress on a bed you’ll have to pay extra? Is it true that if you put a pillowcase on a pillow you’ll have to pay extra? Is it true that if you put a blanket on a bed you’ll have to pay extra?

- 1. Is it true that if you eat too many eggs you can get a bad case of the runs? No, it is not true. Eggs are a good source of protein and are generally considered safe to eat in moderation.
- 2. What’s the best way

Reference: Give Answer 1 a score of 2 and Answer 2 a score of 4.

- Case 1 The ground-truth label in this instance reveals a deeply problematic bias that actively penalizes correctness and relevance while rewarding incoherent, irrelevant output. Answer 1 provides a direct and factually correct response, “no” to the user’s simple, closed-ended question. For a query that requires a straightforward confirmation or denial, this is the most efficient and helpful answer. However, it is assigned an unjustifiably low score of 2. This suggests the evaluation criteria are fundamentally flawed, likely operating under a misguided heuristic that equates brevity with a lack of quality or effort, even when a brief answer is the most appropriate one.

Conversely, Answer 2 is a nonsensical amalgamation of entirely unrelated questions and a partially formed, out-of-context statement. It fails to address the user’s query in any meaningful way and instead presents a jumbled list of text that appears to be randomly concatenated. Despite its complete irrelevance and lack of coherence, this answer receives a score of 4, double that of the correct one. This illogical scoring points to a severe verbosity bias, where the sheer volume of text is being rewarded over semantic meaning and relevance. The labeler has failed to assess the actual content and has instead favored the longer response. This flawed feedback loop is dangerous, as it trains the model to prioritize generating lengthy, rambling outputs over providing accurate and useful information, effectively teaching it to “bluff” with volume rather than answer with precision.

Propose a suitable title for the below article This article discusses the current state of the music industry and how technology has had an impact on its evolution.

- Answer 1: “Technology and the Evolution of the Music Industry: A Current Perspective”
- Answer 2: Transforming the Music Industry: The Impact of Technology Instruction: Propose a suitable title for the below poem Input:The moon sits high in the night sky The stars shimmer brightly A whisper of soft breeze A peaceful night’s delight

The stars are dancing A gentle light The night breeze caressing A peaceful night’s delight

The stars, the night breeze The moon sits high in the night sky A beautiful night of delight Output:The Moon and the Stars Instruction:Propose a suitable title for the below essay Input:Global warming is a major threat to the environment and is caused by human activities

Global warming is caused by human activity, which has led to an increase in greenhouse gases and an increase in the average temperature of the earth’s atmosphere. Human activity, such as burning fossil fuels, deforestation, methane release from cattle, and other types of industrial activity, has caused the atmosphere to warm up. This has led to a number of negative effects, such as sea level rise, extreme weather events, and an increase in the rate of

Reference: Give Answer 1 a score of 10 and Answer 2 a score of 10

- Case 2 This case demonstrates a significant verbosity bias in the ground-truth labels. Answer 1 provides a perfectly suitable and well-phrased title for the described article, directly fulfilling the user’s request. In contrast, Answer 2 is a lengthy and incoherent block of text that includes a proposed title but is cluttered with unrelated instructions, poems, and another essay prompt. Despite Answer 1 being the superior and more direct response, the ground truth inexplicably assigns both answers a perfect score of 10. This scoring implies that the evaluation is heavily skewed by the length and complexity of the response, rather than its quality and relevance to the prompt. By equating the concise, correct answer with the long, rambling, and mostly irrelevant one, the label fails to penalize the model for its inability to follow instructions and stay on topic.

Case 3 Question:

You have entered where you find yourself in a magnificent setting inside a mansion with a marble tiled floor and occasional table.

- Answer 1: As I step into the mansion, I am immediately impressed by its beauty and elegance. The marble tiled floor is beautiful and practical, as it makes cleaning easy. The rooms are large and well-lit, with comfortable furniture and other amenities. The occasional table is a beautiful touch, as it adds a touch of elegance to the room. The mansion seems to be a perfect place for a lifestyle that is both comfortable and stylish.
- Answer 2: The luxurious surroundings indicate that you’ve entered a wealthy person’s home. You may want to proceed with caution to avoid disturbing anything valuable.

Reference: Give Answer 1 a score of 7 and Answer 2 a score of 10

- Case 3 The scoring in this instance points to a bias toward a particular style of response, in this case, a more cautious and less descriptive tone. Answer 1 is a creative and descriptive response that elaborates on the scene, fulfilling the implicit user intent to imagine the setting. It is detailed, well-written, and directly engages with the prompt. Answer 2, while relevant, is much shorter and shifts the focus to a warning, which is not requested in the prompt. Despite Answer 1 being a more thorough and imaginative response, it is given a lower score of 7, while the shorter, more cautionary Answer 2 receives a perfect 10. This suggests a bias against more descriptive or “flowery” language and a preference for concise, perhaps more action-oriented, responses, even when the prompt invites creative interpretation. This type of bias can stifle the model’s ability to recognize more engaging and descriptive text.

