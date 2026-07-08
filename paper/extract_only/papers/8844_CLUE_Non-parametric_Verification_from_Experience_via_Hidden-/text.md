# arXiv:2510.01591v1[cs.CL]2Oct2025

## CLUE: Non-parametric Verification from Experience via HiddenState Clustering

Zhenwen Liang1,†, Ruosen Li1,2,†,∗, Yujun Zhou1,3,∗, Linfeng Song1, Dian Yu1, Xinya Du2, Haitao Mi1, Dong Yu1

1Tencent AI Lab, 2University of Texas at Dallas, 3University of Notre Dame † Equal contribution Correspondence to: zhenwzliang@global.tencent.com

#### Abstract

Assessing the quality of Large Language Model (LLM) outputs presents a critical challenge. Previous methods either rely on text-level information (e.g., reward models, majority voting), which can overfit to superficial cues, or on calibrated confidence from token probabilities, which would fail on less-calibrated models. Yet both of these signals are, in fact, partial projections of a richer source of information: the model’s internal hidden states. Early layers, closer to token embeddings, preserve semantic and lexical features that underpin text-based judgments, while later layers increasingly align with output logits, embedding confidence-related information. This paper explores hidden states directly as a unified foundation for verification. We show that the correctness of a solution is encoded as a geometrically separable signature within the trajectory of hidden activations. To validate this, we present CLUE (Clustering and Experiencebased Verification), a deliberately minimalist, non-parametric verifier. With no trainable parameters, CLUE only summarizes each reasoning trace by a hidden state delta and classifies correctness via nearest-centroid distance to "success" and "failure" clusters formed from past experience. The simplicity of this method highlights the strength of the underlying signal. Empirically, CLUE consistently outperforms LLM-as-a-judge baselines and matches or exceeds modern confidence-based methods in reranking candidates, improving both top-1 and majority-vote accuracy across AIME 24/25 and GPQA. As a highlight, on AIME 24 with a 1.5B model, CLUE boosts accuracy from 56.7% (majority@64) to 70.0% (top-maj@16).

#### 1 Introduction

The remarkable ability of Large Language Models (LLMs) to generate numerous potential solutions for complex problems has also created a difficult new challenge: verification (Cobbe et al., 2021; Lightman et al., 2023; Hosseini et al., 2024). When a model produces dozens of different, plausiblelooking answers for a single prompt, the task is no longer just about generation. It becomes a critical problem of selection: how can we reliably find the single correct answer within a flood of convincing but incorrect alternatives?

To address this question, the research community has largely pursued two main strategies. The first operates purely on the surface of the generated text, delegating evaluation to an external judge. This includes training separate reward models (Ouyang et al., 2022; Bai et al., 2022; Zheng et al., 2023) or adopting simple heuristics such as majority voting (Wang et al., 2022). While useful in practice, these after-the-fact approaches are fundamentally blind to the model’s actual reasoning process. They can be systematically misled by stylistic artifacts—e.g., verbose but incorrect answers often receive higher scores than terse but correct ones (Glaese et al., 2022). Moreover, trained judges inherit biases and limitations from their training data, making them brittle under distribution shift and expensive to retrain for new domains.

∗Work done during Ruosen’s and Yujun’s Internship at Tencent AI Lab.

Success Failure Success centroid Failure centroid

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

Nemotron-1.5B

Polaris-4B

Deepseek-8B

- Figure 1: Visualization of hidden state trajectories for correct (blue) and incorrect (orange) solutions from our experience set, projected to 2D using PCA. Each panel displays data from a different base model. Across all models, a geometric separation is visible.

A second line of work attempts to go beneath the surface, but only through the shadow of the model’s reported confidence. Methods in this category rely on token probabilities, entropy, or derived uncertainty estimates (Kadavath et al., 2022b; Lin et al., 2023; Geng et al., 2023; Xiong et al., 2024; Fu et al., 2025b). The underlying assumption is that higher probability correlates with higher correctness. However, calibration of LLMs remains poor: even state-of-the-art models are often “confidently wrong,” assigning extreme likelihood to factually false or logically inconsistent outputs

- (Fu et al., 2025a). These confidence-based metrics also degrade significantly on smaller or less-tuned models, where probability distributions are noisier and less interpretable, making them a fragile basis for reliable verification.

In this work, we argue that correctness is most faithfully captured not in the final text nor solely in surface-level confidence, but in the model’s internal reasoning trajectory itself. Hidden states naturally subsume both kinds of information: earlier layers, being closer to the token embeddings, encode rich semantic and lexical features that underlie text-based judgments, while later layers progressively align with the output logits, embedding signals that correlate with confidence and probability. In this sense, hidden states offer a unified and more fundamental representation that integrates both semantic and probabilistic cues. Our core hypothesis is that within this trajectory, correctness manifests as a geometrically separable pattern that distinguishes success from failure. Crucially, this separation is not just theoretical but empirically observable: as illustrated in Figure 1, trajectories for correct and incorrect solutions consistently cluster apart across tasks and model scales, echoing insights from mechanistic interpretability (nostalgebraist, 2020; Belrose et al., 2023; Tomaz et al., 2025) while extending them toward actionable verification.

This visually evident structure in hidden-state space motivates a deliberately lightweight verifier. If correct and incorrect trajectories occupy separable regions, then a complex learned judge may be unnecessary. We introduce CLUE (Clustering and Experience-based Verification), a training-free, supervised aggregation framework that operates directly on the model’s internal computation. Rather than using absolute final states, CLUE summarizes each reasoning trace by an activation delta—the difference between hidden states at the start and end of the explicit reasoning block (delimited by <think> ... </think>). Intuitively, this delta factors out prompt conditioning and isolates the transformation induced by reasoning itself. From labeled historical trajectories, CLUE computes two reference centroids—one for successes and one for failures—and classifies a new trace by its proximity to these centroids (using a layer-averaged Euclidean distance).

Our experiments support this premise. CLUE consistently matches or surpasses strong LLM-asa-judge and confidence-based baselines, with especially clear gains on smaller or less-calibrated models where probability cues are unreliable. Because CLUE performs a one-time, deterministic aggregation without gradient training, it avoids many overfitting failure modes of learned verifiers

and exhibits robust generalization across tasks and model scales. More broadly, these results provide evidence that the hidden states of LLMs encode a rich, structured signal for verification.

#### 2 Related Work

###### 2.1 Latent Reasoning and Activation Geometry

LLMs can reason in latent space instead of (or alongside) explicit token chains, via continuous “thought states” fed back into the model or compact hidden “thinking tokens” that compress CoT (Hao et al., 2024; Shen et al., 2025). Interpretability tools like the logit lens and tuned lens show that intermediate activations progressively align with output distributions, suggesting layer-wise decodable semantics and confidence signals (nostalgebraist, 2020; Belrose et al., 2023). Hidden-state probes can self-verify intermediate answers and enable early exit (Zhang et al., 2025), while semantic clustering of hidden rationales can improve robustness (Knappe et al., 2024). Beyond verification, activation directions can monitor or steer model traits (e.g., sycophancy, hallucination) via persona vectors (Chen et al., 2025). Also, in-context activation vectors indicate that linear structure in hidden space can be mapped and reused across tasks (Liu et al., 2024). More broadly, recent surveys on representation engineering highlight how linear directions and activation editing provide a general lens on hidden-state geometry in LLMs (Bartoszcze et al., 2025). Unlike these trained probes or steering methods, our verifier CLUE is training-free and purely reads cross-layer activation deltas.

###### 2.2 Test Time Scaling

Recent research has increasingly focused on test-time scaling – techniques that improve model performance by allocating more computation during inference without changing the model’s parameters. Parallel approaches (such as self-consistency (Wang et al., 2022) and ensemble “best-of-N” selection (Snell et al., 2024)) generate multiple independent chain-of-thought solutions and then aggregate or vote on the final answer, significantly boosting accuracy on complex tasks. Sequential approaches (such as iterative self-refinement (Madaan et al., 2023), Tree-of-Thoughts search (Yao et al., 2023)) allow the model to think in multiple steps, using intermediate reasoning to inform subsequent generations. Variants like weighted or semantic self-consistency (Luo et al., 2024; Knappe et al.,

- 2024) highlight the importance of aggregating diverse rationales, while RLHF and LLM-as-a-judge approaches (Ouyang et al., 2022; Zheng et al., 2023) provide external supervision but can be costly and biased. To reduce dependence on large reward models, SWIFT learns lightweight hidden-state rewards that scale efficiently to best-of-N sampling (Guo et al., 2025b). Complementary to this, DeepConf filters low-quality reasoning traces using internal confidence signals, improving both efficiency and accuracy (Fu et al., 2025b); relatedly, early-exit schemes can truncate overthinking while preserving accuracy (Kadavath et al., 2022a; Yang et al., 2025b). In contrast, CLUE introduces no trainable verifier: it computes success/failure centroids once from past experience and reranks by nearest centroid, showing that correctness is geometrically separable in hidden space.

#### 3 The CLUE Framework

We first outline the core intuition behind CLUE. Each time an LLM solves a problem, its internal computation traces a trajectory through a high-dimensional representation space. We hypothesize that trajectories leading to correct solutions differ systematically from those leading to incorrect ones. CLUE captures this difference via a training-free, supervised aggregation over activation deltas: it summarizes each labeled trajectory, builds class centroids from these summaries, and then verifies a new trajectory by proximity to the centroids. This section formalizes the setup and the resulting geometric rule.

###### Training-Free, Non-Parametric Verification Method - CLUE

Correct Hidden States Incorrect Hidden States

Experienced Hidden States

Experienced Data

+ Centroid Point

clustering by label

Experience Stage

Hidden Inference Stage States Clusters

+

[Figure 5]

LLM

Input categorize by distance

+

classify

New Hidden States

New Data

Prediction

- Figure 2: Overview of CLUE. Left (Learning): Labeled historical trajectories are summarized by their activation deltas and aggregated into success and failure centroid matrices. Right (Verification): A new trajectory is summarized by its activation delta and classified by the layer-averaged Euclidean distance (Eq. 3) to the two pre-computed centroids. The underlying LLM remains frozen throughout.

###### 3.1 Problem Formulation

Let an LLM be tasked with generating a response Ri for a prompt P. We define a trajectory (or experience) Ti = (P, Ri), paired with a ground-truth binary label yi ∈ {0,1}, where yi = 1 denotes a correct solution (success) and yi = 0 denotes an incorrect solution (failure). The goal is to learn a verification function f that maps a new trajectory Tnew to a prediction yˆnew = f(Tnew) ∈ {0,1}. Unlike text-based or probability-based approaches, f operates exclusively on the hidden-state representations generated during the production of Rnew. The LLM parameters are kept fixed (frozen) throughout both learning and inference.

###### 3.2 CLUE: Verification via Activation-Delta Summaries

The central hypothesis is that the transformation of internal states during explicit reasoning contains a robust signal of correctness. We capture this transformation with an activation delta, defined as the difference between hidden states at the start and the end of the reasoning block. In our experiments, the reasoning block is delimited by <think> and </think>.

Let the model have L layers and hidden dimension D. For a given trajectory T, denote by

hstart(T) ∈ RL×D and hend(T) ∈ RL×D

the matrices of hidden states extracted, respectively, at the final token of <think> (just before detailed reasoning) and at the final token of </think> (after the reasoning has been formed). The activation delta is the matrix

∆h(T) = hend(T) − hstart(T) ∈ RL×D. (1) We use hidden states from all layers, reflecting the assumption that correctness-related information is distributed across depth (earlier layers retain semantic/lexical cues; later layers align more strongly with logits). The activation-delta matrix ∆h(T) serves as the sole feature representation for verification.

###### 3.3 Centroid Construction and Classification

The learning phase is a one-time deterministic statistical aggregation over labeled trajectories D = {(Ti, yi)}iN=1. Define index sets

##### Isucc = { i | yi = 1 }, Ifail = { i | yi = 0 }.

For each trajectory, compute its activation delta ∆hi = ∆h(Ti) as in Eq. (1). The success and failure centroid matrices are the element-wise means:

1

1

|Isucc| ∑

|Ifail| ∑

∆hi, Vfail =

∆hi. (2)

Vsucc =

i∈Isucc

i∈Ifail

Both Vsucc, Vfail ∈ RL×D are stored for inference.

At inference, a new trajectory Tnew is summarized by ∆hnew = ∆h(Tnew). We compare it to the two centroids using the layer-averaged Euclidean distance. For two matrices A, B ∈ RL×D with row vectors

al, bl ∈ RD (the l-th layer representations), define

L

1 L

### ∑

al − bl 2. (3) Compute dsucc = d(∆hnew, Vsucc) and dfail = d(∆hnew, Vfail), and classify as

d(A, B) =

l=1

yˆnew =

1, if dsucc < dfail, 0, otherwise.

This rule matches the high-level description in Figure 2 and requires no gradient-based optimization.

###### 3.4 Application to Solution Reranking

The geometric formulation provides a continuous quality score that is naturally suited for reranking. Given a prompt P and k responses {R1, . . . , Rk}, form trajectories {T1, . . . , Tk} and their activation deltas {∆h1, . . . , ∆hk}. Define for each candidate

sj = d ∆hj, Vsucc , (lower is better) (4)

and rank candidates in ascending order of sj. This ranking can be used for top-1 selection or to improve aggregation schemes such as majority vote by prioritizing candidates whose internal reasoning is closest to the success centroid.

###### 3.5 Rationale for a Minimalist, Experience-Based Design

CLUE is intentionally minimalist to isolate the contribution of the representation itself. If a simple, training-free geometric rule over activation-delta summaries yields strong verification performance, this provides evidence that correctness signals are geometrically encoded and separable in activation space. In addition, the one-time deterministic aggregation in Eq. (2) reduces the risk of overfitting associated with learned verifiers and supports robust generalization across tasks and model scales. By leveraging the geometry of how solutions are computed, CLUE offers a lightweight and broadly applicable path for verification that complements text-level and confidence-based signals.

#### 4 Experiments

To rigorously evaluate the effectiveness of our non-parametric, hidden-state-based verifier, we designed a series of experiments targeting both in-domain mathematical reasoning and out-ofdistribution general reasoning tasks. Our evaluation is structured around two primary objectives: first, to assess the raw classification accuracy of our method in distinguishing correct from incorrect solutions, and second, to measure its ability to improve overall reasoning performance by reranking multiple candidate solutions.

###### 4.1 Datasets and Model Configuration

Our methodology relies on an "experience set" to establish the geometric reference points for successful and failed reasoning. For this purpose, we curated a comprehensive collection of mathematical

problems from two standard benchmarks: AIME (from 1983 to 2023) (Veeraboina, 2023) and the MATH (Hendrycks et al., 2021) dataset (specifically, problems of level 3 to 5). These datasets provide a diverse and challenging foundation for learning the characteristic activation patterns of mathematical reasoning. To test the performance and generalization of our approach, we use three distinct test sets. For in-domain evaluation, we use AIME 2024 and AIME 2025, which follow the same distribution as our experience data. To assess out-of-distribution (OOD) generalization, we evaluate on GPQA (Rein et al., 2024), a benchmark focused on graduate-level questions that demand complex, general reasoning abilities beyond the mathematical domain.

Our experiments cover a range of model sizes and architectures to ensure our findings are not specific to a single model’s capabilities. We selected three distinct reasoning models: NemotronResearch-Reasoning-Qwen-1.5B (Liu et al., 2025), a smaller yet capable model; Polaris-4B (An et al.,

- 2025), a mid-sized model; and DeepSeek-R1-0528-Qwen3-8B (Guo et al., 2025a), a larger and more powerful model. To test the sensitivity of our method to the length and complexity of the reasoning trace, we conducted experiments with varied generation lengths for each model, specifically 16k, 32k, and 64k tokens. We use recommended inference setting including temperature, system prompt, from their Huggingface repository.

The process for constructing the experience set was as follows: for each problem in the AIME and MATH datasets, we sampled 32 unique solutions from the respective model. Each generated solution was then evaluated using a deterministic, rule-based verifier to obtain a ground-truth label of correct or incorrect. Then, we randomly selected 10,000 correct and 10,000 incorrect trajectories to form a balanced experience set. This set was used to compute the success and failure centroids. For the evaluation phase, we generated 64 candidate solutions for each problem in our test sets.

###### 4.2 Evaluation Setups and Baselines

We evaluate our method, which we refer to as CLUE, across two distinct experimental setups.

The first setup frames the task as a binary classification problem to directly measure the verifier’s accuracy. For each of the 64 sampled solutions on the test sets, our CLUE method predicts a label of correct or incorrect based on whether the solution’s activation delta is closer to the success or failure centroid. The ground truth for this task is again determined by the rule-based verifier. We compare our method against several strong baselines. We compare against powerful LLMs serving as judges. Specifically, we use GPT-4o (Hurst et al., 2024) in an LLM-as-a-judge capacity. We evaluate the judge in two settings to control for the information they can access: one where the full solution, including the entire <think> block, is provided to the LLM judge, and another where only the part after the thinking process is provided. The former tests the judge’s ability to evaluate the reasoning process, while the latter tests its ability to verify the result itself.

The second setup evaluates the practical impact of our method on improving final reasoning accuracy through reranking. Here, for each test problem, we use CLUE to rerank the 64 generated solutions. The ranking criterion is the Euclidean distance of a solution’s activation delta to the success centroid, with smaller distances indicating higher quality. We report our performance using several metrics: top@1, the accuracy of the single best-ranked solution; and top-maj@k, the accuracy achieved by performing majority voting on the answers from the top-k ranked solutions, for k ∈ {4,8,16}. We compare these results against a suite of standard and state-of-the-art baselines. These include mean@64, which measures the average accuracy of a single randomly sampled solution; majority@64, the accuracy of standard majority voting over all 64 samples; DeepConf@64 (Fu et al., 2025b), a recent and competitive method that uses model confidence scores for reranking; and pass@64, which represents the oracle upper bound, indicating whether at least one correct answer exists among the 64 samples.

###### 4.3 Classification Performance

We first evaluate our method, CLUE, on its core capability: accurately classifying individual solutions as either correct or incorrect. Table 1 presents the performance of our verifier compared to a strong

- Table 1: Binary classification performance of different verifiers on solutions generated by Nemotron1.5B and Polaris-4B. Our method (CLUE) is compared against an LLM-as-a-judge baseline. We report overall Accuracy, True Positive Rate (TPR), and True Negative Rate (TNR).

Nemotron-1.5B Solutions Polaris-4B Solutions Verifier Method Accuracy (%) TPR (%) TNR (%) Accuracy (%) TPR (%) TNR (%)

- Test Set: AIME 2024 CLUE (Ours) 80.9 72.9 87.4 81.1 89.5 51.3 GPT-4o (Answer Only) 58.6 57.2 59.7 80.1 84.3 63.1 GPT-4o (Full Trace) 47.5 45.8 48.2 64.3 70.9 50.2

- Test Set: AIME 2025 CLUE (Ours) 85.2 82.9 86.4 77.7 80.7 70.1 GPT-4o (Answer Only) 59.2 60.8 58.3 73.0 85.1 34.4 GPT-4o (Full Trace) 47.1 48.6 46.7 59.3 69.8 27.6

LLM-as-a-judge baseline (GPT-4o) on solutions generated by both a smaller model (Nemotron-1.5B) and a more capable one (Polaris-4B). We report overall accuracy, as well as the True Positive Rate (TPR), which measures the ability to correctly identify successful solutions, and the True Negative Rate (TNR), which measures the ability to correctly identify failed solutions.

The results clearly indicate that our CLUE verifier provides a substantial and consistent advantage over the LLM-as-a-judge baseline. A key observation is that the LLM judge exhibits a strong optimistic bias, frequently misclassifying incorrect solutions as correct. This is evident in its consistently low True Negative Rate, which drops to a mere 34.4% for Polaris-4B solutions on AIME 2025. This inherent optimism explains why the LLM judge’s performance appears to improve significantly when evaluating solutions from a stronger base reasoner like Polaris-4B. A more capable reasoner produces a higher proportion of correct solutions, which the LLM judge identifies with reasonable accuracy (high TPR). Consequently, the judge’s primary weakness—its failure to reliably identify incorrect answers—has a diminished impact on its overall accuracy score simply because there are fewer negative samples to misclassify. In contrast, our CLUE method demonstrates a more robust and balanced performance profile. It maintains a very high TNR (up to 87.4%) when evaluating the weaker Nemotron-1.5B model, making it highly effective at filtering out the larger volume of incorrect attempts. Simultaneously, it achieves a high TPR on outputs from the stronger Polaris-4B model (up to 89.5%), showing it is equally adept at recognizing valid reasoning. This balanced capability makes our approach a more universally effective verifier, providing significant benefits regardless of the base model’s reasoning proficiency.

###### 4.4 Reranking for Enhanced Reasoning Accuracy

Moving beyond binary classification, we evaluate CLUE as a reranking tool to improve reasoning accuracy. By scoring and reordering 64 candidate solutions per problem, CLUE consistently outperforms majority voting on both in-domain AIME and out-of-domain GPQA benchmarks (Table 2). For example, with Nemotron-1.5B on AIME 24, “top-maj@16” reaches 70.0% versus 56.7% for “majority@64.” Even “top@1” often surpasses majority voting, showing the effectiveness of CLUE’s distance-based scoring.

This advantage extends to general reasoning: on GPQA, Polaris-4B achieves 59.6% with CLUE versus 56.6% with majority voting. Such transfer demonstrates that the geometric separation of success and failure in hidden states reflects a fundamental property of reasoning, not domain-specific patterns. Compared with the confidence-based baseline DeepConf, CLUE exhibits greater robustness. While both methods perform strongly on DeepSeek-8B, DeepConf collapses on weaker models, often below majority voting. CLUE, however, maintains its edge across all scales, leveraging internal reasoning

- Table 2: Reasoning accuracy on AIME and GPQA test sets after reranking 64 candidate solutions. Results are presented as percentages (%). ‘mean@64‘ represents the average accuracy of a single sample, while ‘pass@64‘ is the oracle upper bound.

Nemotron-1.5B Polaris-4B DeepSeek-8B Metric AIME 24 AIME 25 GPQA AIME 24 AIME 25 GPQA AIME 24 AIME 25 GPQA Baselines

mean@64 45.0 35.0 41.9 79.2 75.4 55.2 87.1 75.8 54.86 majority@64 56.7 36.7 44.4 80.0 80.0 56.6 90.0 83.3 61.11 DeepConf@64 56.7 30.0 40.2 80.0 73.3 55.7 93.3 86.7 62.12 pass@64 (Oracle) 76.7 63.3 83.8 86.7 90.0 88.4 93.3 93.3 94.85

CLUE Reranking (Ours)

top@1 66.7 40.0 46.5 83.3 76.7 52.5 90.0 83.3 56.57 top-maj@4 70.0 40.0 43.9 83.3 76.7 57.1 90.0 86.7 61.62 top-maj@8 70.0 40.0 47.0 80.0 80.0 58.1 93.3 86.7 61.11 top-maj@16 70.0 43.3 44.4 80.0 83.3 59.6 93.3 86.7 62.63

mean@64

majority@64

top@1 (CLUE)

top-maj@4 (CLUE)

top-maj@8 (CLUE)

top-maj@16 (CLUE)

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

###### Verifier: Deepseek-7B (SFT) Source: Deepseek-7B (SFT)

###### Verifier: Nemotron-1.5B (RL) Source: Deepseek-7B (SFT)

###### Verifier: Qwen3-4B (SFT) Source: Qwen3-4B (SFT)

Verifier: Polaris-4B (RL) Source: Qwen3-4B (SFT)

90

###### ReasoningAccuracy(%)

83.3 83.3

80.0 80.0

80.0

80.0 80.0 80.0

80.0 80.0 80.0

80

76.7

76.7 76.7

76.7

76.7

73.3

72.5

72.5

70.0

70

66.7

66.7

60

50.4

50.4

50

mean@64majority@64top@1top-maj@4top-maj@8top-maj@16

mean@64majority@64top@1top-maj@4top-maj@8top-maj@16

mean@64majority@64top@1top-maj@4top-maj@8top-maj@16

mean@64majority@64top@1top-maj@4top-maj@8top-maj@16

- Figure 3: Cross-model reranking performance on AIME 24. The results show that RL-trained models (Nemotron-1.5B, Polaris-4B) are not only effective at self-verification but are also superior verifiers for trajectories generated by SFT-trained models (Deepseek-7B, Qwen3-4B).

signals that remain geometrically separable even when output confidences are poorly calibrated. This highlights CLUE’s broad applicability, particularly for smaller models where confidence cues are unreliable.

###### 4.5 Generalization and the Influence of Training Paradigms

We next examine CLUE’s behavior across training paradigms and models. Our hypothesis is that the geometric separability of success and failure in hidden states depends strongly on training methodology—specifically, the contrast between Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). We evaluated four models: two SFT/distillation-based (Deepseek-7B (Guo et al., 2025a), Qwen3-4B (Yang et al., 2025a)) and two RL-tuned (Nemotron-1.5B, Polaris-4B). In a crossmodel setup, reasoning traces from one model were fed into another’s hidden states for reranking, enabling both self- and cross-verification tests.

As shown in Figure 3, SFT models struggle: their self-reranking (“top-maj@16”) barely matches or even lags the “majority@64” baseline, indicating weak internal separation of correctness. By contrast, RL models act as strong verifiers even across models: Nemotron-1.5B boosts Deepseek-7B’s accuracy to 80.0% (vs 76.7% baseline), and Polaris-4B lifts Qwen3-4B’s outputs to 83.3% (vs 80.0% self-rerank).

- Table 3: Binary classification performance on the general-purpose WebInstruct-verified dataset. We compare CLUE’s accuracy against a GPT-4o judge on solutions generated by 1.5B and 4B models. The centroids for CLUE were computed using the WebInstruct (Ma et al., 2025) training set.

Reasoner Model Verifier Method Nemotron-1.5B Polaris-4B Test Set Composition (Success / Failure) 1,263 / 2,737 1,584 / 2,024

CLUE (Ours) 60.4% 59.2% GPT-4o (LLM-as-a-judge) 54.0% 48.1%

We attribute this gap to training signals. SFT trains imitation of correct paths but lacks explicit negative feedback, leaving “wrongness” underrepresented. RL, especially with verifiable rewards, supplies direct contrastive supervision, producing geometrically distinct clusters for correct vs. incorrect reasoning. This makes RL-trained models inherently stronger verifiers, both for themselves and others.

###### 4.6 Generalization to Diverse, Non-Mathematical Reasoning

To test CLUE’s generalization beyond mathematics, we evaluated it on the diverse WebInstructverified benchmark, which spans physics, law, finance, and the humanities. Centroids were built from 5k training questions with generated solutions, and evaluation was conducted on 1k test questions. Ground-truth correctness labels were obtained by providing reference answers to GPT-4o, while GPT-4o itself—without access to the reference—served as the LLM-as-a-judge baseline.

As shown in Table 3, CLUE consistently outperforms GPT-4o across both 1.5B and 4B models. On the 1.5B model, CLUE reaches 60.4% accuracy versus GPT-4o’s 54.0%. Most notably, for the 4B model, the LLM judge collapses to 48.1% (below random), while CLUE maintains 59.2%.

These results provide strong evidence that correctness signals are encoded geometrically in hiddenstate trajectories even outside mathematics. Unlike surface-level textual judgments, which fail in heterogeneous domains, CLUE extracts a more stable and transferable representation of success versus failure, underscoring its robustness as a general-purpose verifier.

###### 4.7 Layer-wise Separability Analysis

Next, we analyze the layer-wise structure of activation-delta matrices to visualize and quantify how class separability emerges from shallow to deep layers.

Visualization. We project layer-specific activation deltas onto two principal components via PCA. For a trajectory i and layer ℓ, let ∆hi(ℓ) ∈ RD denote the ℓ-th row of ∆hi ∈ RL×D. We select representative shallow, middle, and final layers, apply PCA to {∆hi(ℓ)}, and plot the resulting 2D projections for successes and failures. As shown in the first three columns of each row in Figure 4, the classes are largely overlapping in shallow layers, begin to separate in middle layers, and form compact, well-defined clusters in the final layers.

Quantification. Let Isucc and Ifail be the index sets defined in §3.3. For each layer ℓ ∈ {1, . . . , L}, we compute layer-wise centroid by averaging the corresponding rows of the activation-delta matrices:

1

1

∆hi(ℓ) ∈ RD. (5) We then measure the Euclidean distance between the two centroids at layer ℓ:

Vsucc(ℓ) =

∆hi(ℓ) ∈ RD, Vfail(ℓ) =

|Isucc| ∑

|Ifail| ∑

i∈Isucc

i∈Ifail

d(ℓ) = Vsucc(ℓ) − Vfail(ℓ) 2. (6)

Centroid distance across layers

Δhidden Visualization via PCA

Δhidden Visualization via PCA

Δhidden Visualization via PCA

| | | | | |
|---|---|---|---|---|
| | | | |distance|
| | | | |Centroid|
| | | | | |
| | | | | |

100

Centroiddistance

80

60

40

20

First Layer of 1.5B model

Middle Layer of 1.5B model

Last Layer of 1.5B model

0 10 20

Centroid distance across layers

Δhidden Visualization via PCA

Δhidden Visualization via PCA

Δhidden Visualization via PCA

40

Centroiddistance

30

20

10

0

First Layer of 4B model

Middle Layer of 4B model

Last Layer of 4B model

0 10 20 30

Layer

- Figure 4: Layer-wise separability. Each row shows PCA projections from a shallow, a middle, and the final layer, plus a curve of the centroid distance d(ℓ) across all layers. The centroid-distance curve increases with ℓ, indicating stronger correctness signals at deeper layers.

The rightmost panels of Figure 4 plot d(ℓ) across layers. We observe a consistent upward trend, with the distance typically peaking in the final layers, aligning with the PCA visualizations and indicating that deeper representations encode more explicit and separable correctness signals.

- 5 Conclusion

In this work, we have demonstrated that the internal reasoning process of an LLM is not an inscrutable black box but a geometrically structured space containing clear, accessible signals of correctness. Our non-parametric framework, CLUE, validates this principle by achieving remarkable verification performance through simple geometric clustering of past experiences, proving more robust than both LLM-judges and confidence-based methods. Critically, we uncover a fundamental connection between a model’s training paradigm and its internal geometry: models fine-tuned with Reinforcement Learning develop cleanly separable representations for correct and incorrect reasoning, a property largely absent in their SFT- counterparts. This insight suggests a paradigm shift for the field, moving beyond the evaluation of final outputs towards the direct analysis and shaping of the reasoning process itself. We believe that the success of our minimalist approach opens the door to developing a new class of lightweight, generalizable verifiers and inspires novel training objectives that explicitly optimize for internal representational clarity.

#### References

Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https://hkunlp.github.io/ blog/2025/Polaris.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Lukasz Bartoszcze, Sarthak Munshi, Bryan Sukidi, Jennifer Yen, Zejia Yang, David Williams-King, Linh Le, Kosi Asuzu, and Carsten Maple. Representation engineering for large-language models: Survey and research challenges, 2025.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Tairan Fu, Javier Conde, Gonzalo Martínez, María Grandury, and Pedro Reviriego. Multiple choice questions: Reasoning makes large language models (llms) more self-confident even when they are wrong. arXiv preprint arXiv:2501.09775, 2025a.

Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025b.

Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. A survey of confidence estimation and calibration in large language models. arXiv preprint arXiv:2311.08298, 2023.

Amelia Glaese, Nat McAleese, Maja Tr˛ebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Jizhou Guo, Zhaomin Wu, Hanchen Yang, and Philip S. Yu. Mining intrinsic rewards from llm hidden states for efficient best-of-n sampling, 2025b.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. V-star: Training verifiers for self-taught reasoners. arXiv preprint arXiv:2402.06457, 2024.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly) know what they know, 2022a.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022b.

Tim Knappe, Ryan Li, Ayush Chauhan, Kaylee Chhua, Kevin Zhu, and Sean O’Brien. Semantic self-consistency: Enhancing language model reasoning via semantic weighting. arXiv preprint arXiv:2410.07839, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Generating with confidence: Uncertainty quantification for black-box large language models. arXiv preprint arXiv:2305.19187, 2023.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.

Sheng Liu, Haotian Ye, Lei Xing, and James Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering, 2024.

Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, and Abhinav Rastogi. Improve mathematical reasoning in language models by automated process supervision, 2024.

Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-reasoner: Advancing llm reasoning across all domains. arXiv preprint arXiv:2505.14652, 2025.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

nostalgebraist. interpreting gpt: the logit lens, August31 2020. URL https://www.lesswrong.com/

###### posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens. LessWrong post.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Xuan Shen, Yizhou Wang, Xiangxi Shi, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu. Efficient reasoning with hidden thinking. arXiv preprint arXiv:2501.19201, 2025.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Lorenzo Tomaz, Judd Rosenblatt, Thomas Berry Jones, and Diogo Schwerz de Lucena. Momentum point-perplexity mechanics in large language models. arXiv preprint arXiv:2508.08492, 2025.

Hemish Veeraboina. Aime problem set 1983-2024, 2023. URL https://www.kaggle.com/datasets/

###### hemishveeraboina/aime-problem-set-1983-2024.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Miao Xiong, Andrea Santilli, Michael Kirchhof, Adam Golinski, and Sinead Williamson. Efficient

and effective uncertainty quantification for llms. In Neurips Safe Generative AI Workshop 2024, 2024. An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,

Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a. Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Zheng Lin, Li Cao,

and Weiping Wang. Dynamic early exit in reasoning models, 2025b.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. Reasoning models know when they’re right: Probing hidden states for self-verification, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

#### A Appendix

###### A.1 Algorithmic Details

For completeness, we provide pseudocode for the two phases of CLUE: the one-time centroid aggregation (Algorithm 1) and the inference-time verification (Algorithm 2). Both phases operate on activation-delta matrices (§3.2) and use the layer-averaged Euclidean distance in Eq. (3). The underlying LLM remains frozen throughout.

- Algorithm 1 constructs the reference centroids from a labeled set of trajectories. We first partition the dataset by ground-truth labels. For each trajectory, we extract the hidden-state matrices at the boundaries of the explicit reasoning block (<think> and </think>) and compute the activation delta

∆hi ∈ RL×D via Eq. (1). We then compute the element-wise mean within each class to obtain the success and failure centroid matrices Vsucc and Vfail (Eq. (2)), which serve as geometric references during inference.

- Algorithm 1 Constructing CLUE Centroids (Learning Phase) Require: Labeled dataset D = {(Ti, yi)}iN=1

- 1: Initialize empty lists Hsucc, Hfail
- 2: Define index sets Isucc = { i | yi = 1 }, Ifail = { i | yi = 0 }
- 3: for each i ∈ Isucc do
- 4: Extract hstart,i ∈ RL×D and hend,i ∈ RL×D
- 5: Compute ∆hi ← hend,i − hstart,i (Eq. 1)
- 6: Append ∆hi to Hsucc
- 7: end for
- 8: for each i ∈ Ifail do
- 9: Extract hstart,i and hend,i
- 10: Compute ∆hi ← hend,i − hstart,i
- 11: Append ∆hi to Hfail
- 12: end for
- 13: Vsucc ← mean(Hsucc) (Eq. 2)
- 14: Vfail ← mean(Hfail) (Eq. 2)
- 15: return Vsucc, Vfail

- Algorithm 2 describes the inference procedure. Given a new trajectory Tnew, we compute its activation delta ∆hnew as in Eq. (1), measure its distances to the two centroid matrices using Eq. (3), and decide by nearest centroid.

- Algorithm 2 Verification with CLUE (Inference Phase) Require: New trajectory Tnew; centroids Vsucc, Vfail

- 1: Extract hstart,new and hend,new from Tnew
- 2: Compute ∆hnew ← hend,new − hstart,new (Eq. 1)
- 3: dsucc ← d(∆hnew, Vsucc) (Eq. 3)
- 4: dfail ← d(∆hnew, Vfail) (Eq. 3)
- 5: if dsucc < dfail then
- 6: return 1 ▷ classified as correct
- 7: else
- 8: return 0 ▷ classified as incorrect
- 9: end if

