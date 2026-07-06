### arXiv:2511.13254v1[cs.CL]17Nov2025

# Souper-Model: How Simple Arithmetic Unlocks State-of-the-Art LLM Performance

Shalini Maiti1,3†, Amar Budhiraja2†, Bhavul Gauri2, Gaurav Chaurasia2, Anton Protopopov2, Alexis Audran-Reiss2, Michael Slater2, Despoina Magka2, Tatiana Shavrina2, Roberta Raileanu2,3∗, Yoram Bachrach2

1Meta SuperIntelligence Labs, 2FAIR at Meta, 3University College London †Equal contribution, ∗Work done while working at Meta

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their training remains resource- and time-intensive, requiring massive compute power and careful orchestration of training procedures. Model souping—the practice of averaging weights from multiple models of the same architecture—has emerged as a promising pre- and post-training technique that can enhance performance without expensive retraining.

In this paper, we introduce Soup Of Category Experts (SoCE), a principled approach for model souping that utilizes benchmark composition to identify optimal model candidates and applies non-uniform weighted averaging to maximize performance. Contrary to previous uniform-averaging approaches, our method leverages the observation that benchmark categories often exhibit low inter-correlations in model performance. SoCE identifies "expert" models for each weakly-correlated category cluster and combines them using optimized weighted averaging rather than uniform weights. We demonstrate that the proposed method improves performance and robustness across multiple domains, including multilingual capabilities, tool calling, and math and achieves state-of-the-art results on the Berkeley Function Calling Leaderboard.

Date: November 18, 2025 Correspondence: Shalini Maiti at shalinimaiti@meta.com, Amar Budhiraja at amarbudhiraja@meta.com Code: https://github.com/facebookresearch/llm_souping

#### 1 Introduction

Large Language Models (LLMs) have emerged as transformative technologies, demonstrating remarkable capabilities across diverse domains from natural language understanding to code generation and tool use (Brown et al., 2020; Comanici et al., 2025; Dubey et al., 2024; Roziere et al., 2023; Schick et al., 2023; Touvron et al., 2023a,b; Achiam et al., 2023; Team et al., 2023). The training paradigms for these foundational models typically involve massivescale pretraining on diverse corpora followed by supervised fine-tuning and reinforcement learning from human feedback (Ouyang et al., 2022; Bai et al., 2022). However, current training procedures remain extremely resource- and time-intensive, often requiring large amounts of compute power and careful orchestration of training data mixtures to achieve desired capabilities to achieve (Hoffmann et al., 2022; Touvron et al., 2023b; Zhang et al., 2022).

Recent advances in model souping—the practice of averaging weights from multiple models of the same architecture—have shown promising results as a post-training technique to enhance performance without requiring expensive retraining. Prior work has demonstrated that uniform averaging of model weights can lead to improved performance across various tasks Wortsman et al. (2022) and mitigate issues like catastrophic forgetting Kleiman et al. (2025). These findings suggest that model souping offers a computationally efficient alternative to traditional approaches that rely on modifying training data recipes or extensive retraining procedures.

In this paper, we demonstrate that carefully designed souping techniques can achieve state-of-the-art performance by considering benchmark composition and employing non-uniform weighting strategies. In particular, our approach addresses two key limitations of existing techniques: the arbitrary selection of models for souping and the assumption that uniform weighting is optimal. The contributions of this paper are thus summarized as follows:

- 1. Automated Checkpoint Souping: We introduce Soup Of Category Experts (SoCE), a novel model souping technique that leverages benchmark composition through an automatic category-aware expert selection mechanism. Unlike prior approaches that rely on uniform averaging Wortsman et al. (2022), SoCE exploits the observation that benchmark categories often exhibit low inter-correlations in model performance to identify “expert” models for each weakly-correlated category cluster, and aggregate their expertise using non-uniform weighted averaging to maximize overall performance.
- 2. State-of-the-Art Performance: We demonstrate the efficiency of the proposed method across diverse domains, including state-of-the-art results for the Berkeley Function Calling Leaderboard (Patil et al.). Our approach consistently outperforms existing baselines, validating the effectiveness of category-specific model souping.
- 3. Higher Model Consistency: We perform a large-scale empirical analysis to show that model souping enhances performance consistency across benchmark categories. Souped models exhibit significantly higher Pearson correlations between category performances across model populations compared to their unsouped counterparts, indicating improved robustness and coherence across diverse task types.

#### 2 Related Work

While model merging or averaging techniques are not new (Regent’s et al. (1996) describe one of the first implementations), we can outline the current emerged methodologies applied to LLMs for efficiency and generalization improvements.

Souping: Model Souping has been explored in the machine learning literature in multiple contexts Jang et al. (2025); Jaiswal et al. (2023); Kleiman et al. (2025); Wortsman et al. (2022); Yu et al. (2024); Zimmer et al. (2023). Wortsman et al. (2022) show how using souping on finetuned models can lead to improvements over the individual models where individual models are variants of different hyperparameters of the same base models. They proposed three strategies for souping: Uniform Souping (where all models are weighted equally), Greedy Souping (add 1 model at a time in decreasing order of performance) and Learned-Souping (soup weights are learned via gradient descent). They show the most promising results via Greedy Souping technique. Jang et al. (2025) is the closest to our work. They improve upon the work by Wortsman et al. (2022) using a geometric insight of the proximity to pre-trained models to reduce the number of models required to soup to achieve better results than greedy souping techniques. We take this further by utilizing the insights from the correlation of metrics as well as co-operative game theory within a benchmark to formalize the selection of the best candidates and the proportion of souping.

In a related study Yu et al. (2024), the authors take several homologous (same architecture and parameters) models and perform deactivation and reactivation of neurons and merge the newly derived models to achieve better models compared to the original input candidates. In another line of work, model souping has been used in continual learning to address catastrophic forgetting during finetuning of LLMs Kleiman et al. (2025).

Automatic model merging. Several works explore automatic model merging techniques: authors in Yang et al. (2024) show that we can use unsupervised methods to discover optimal merging parameters based on entropy minimization. The technique is mostly applied to classification tasks with ViT models, leading to gains in the multi-task learning domain for classification models, not yet applied to LLMs. Akiba et al. (2025) apply evolutionary algorithms to model merging, finding that effective combinations of open-source models can be discovered automatically, leading to higher-performing combinations of multimodal capabilities.

Model averaging and merging techniques have gained significant traction, and various studies have collectively demonstrated that this approach: 1) can result in improved performance for post-trained LLMs; 2) can be directed automatically;

###### 3) can leverage the landscape of the open-source derivative models of the same architecture.

#### 3 Methodology: Soup Of Category Experts

In this section, we present our method for model selection and weighting the selected models with the benchmark composition in mind.

The fundamental insight underlying our approach is that benchmark performance across categories exhibits heterogeneous correlation patterns. Different models demonstrate varying expertise across benchmark categories, with some

categories being strongly correlated while others remain weakly correlated or even negatively correlated in terms of cross-model performance. To illustrate this phenomenon, we analyze the Berkeley Function Calling Leaderboard (BFCL) (Patil et al.), which comprises multiple categories, including multi-turn function calling, irrelevance detection, and function calling across different programming languages (Java, Javascript, etc.). Figure 1 presents a correlation heatmap showing Pearson correlation coefficients between category performances across all models on the leaderboard, where darker regions indicate higher correlations. The heatmap reveals strong positive correlations (dark green regions) between related categories and weak to negative correlations (light green regions) between unrelated categories. For instance, multi-turn categories exhibit high inter-correlation (0.96-0.98), indicating that models proficient in one multi-turn task typically excel across all multi-turn scenarios. Conversely, weak correlation (0.07) exists between Multi-turn-base (where the model is evaluated on Multi-turn function calling aspect) and Live Accuracy (where the model is evaluated on real-world function-calling prompts collected by users) categories, suggesting these represent distinct competency domains. We present our method in relation to the BFCL Patil et al. benchmark, but this can be extended without any loss of generalisation to other benchmarks too, as observed from the results reported in Table 2.

Positive Correlation

##### +1

Overall Acc

[Figure 1]

Non-Live AST Acc Non-Live Simple AST

Non-Live Multiple AST

Non-Live Parallel AST Non-Live Parallel Multiple AST

Live Acc Live Simple AST

Live Multiple AST

0

Live Parallel AST Live Parallel Multiple AST

Multi Turn Acc Multi Turn Base

Multi Turn Miss Func Multi Turn Miss Param Multi Turn Long Context Relevance Detection

Irrelevance Detection -1

IrrelevanceDetectionRelevanceDetectionMultiTurnLongContextMultiTurnMissParamMultiTurnMissFuncLiveParallelMultipleASTMultiTurnBaseMultiTurnAccNon-LiveParallelMultipleASTLiveParallelASTLiveMultipleASTLiveSimpleASTNon-LiveParallelASTNon-LiveMultipleASTLiveAccNon-LiveSimpleASTNon-LiveASTAccOverallAcc

Negative Correlation

- Figure 1 Pearson Correlation of model performance from BFCL leaderboard

Our proposed method, Soup Of Category Experts (SoCE), exploits these correlation patterns to strategically select and weight models for souping. The core principle is to identify expert models for each weakly-correlated category cluster and aggregate them using optimized weighted averaging to combine complementary expertise. Algorithm 1 formalizes this procedure, which comprises four key steps: (1) correlation analysis to identify weakly-correlated category pairs,

(2) expert model selection for each category based on performance rankings, (3) weight optimization to maximize aggregate performance, and (4) weighted model souping to produce the final combined model. For weight optimization, we perform a search over a uniform set of weights. We iterate over all combinations in the weight space with the highest weight 0.9 and lowest of 0.1 for each model with step size of 0.1. We also add a special case of equal weighing of the candidates to compare uniform souping.

Algorithm 1 Soup Of Category Experts (SoCE) Require: Benchmark dataset D with categories {C1,C2,...,Ck} Require: Set of candidate models M = {M1,M2,...,Mn} Require: Correlation threshold τ for identifying low-correlation categories Ensure: Souped model Msoup

- 1: Step 1: Analysis
- 2: for each pair of categories (Ci,Cj) where i ̸= j do
- 3: Compute Pearson correlation ρi,j between (P1i,...,Pni) and (P1j,...,Pnj) across all models and category performances
- 4: end for
- 5: L = {Ck | ∃l such that |ρk,l| < τ or |ρl,k| < τ}
- 6: Step 2: Expert Model Selection
- 7: for each category Ci ∈ L do
- 8: for each model Mj ∈ M do
- 9: Pji = Performance(Mj,Ci)
- 10: end for
- 11: Select expert model Mi∗ = arg maxM

j∈M Pji

- 12: end for
- 13: Step 3: Weight Optimization
- 14: Generate weights w = {w1,...,wl} such that li=1 wi = 1 and l = |L|
- 15: Iterate over uniform set of weight combinations to identify the best souped model:
- 16: w∗ = arg maxw ki=1 Performance lj=1 wj · Mj∗,Ci
- 17: Step 4: Model Souping
- 18: Create souped model: Msoup = ki=1 wi∗ · Mi∗
- 19: return Msoup

#### 4 Experiments

###### 4.1 Benchmarks and Baselines

We conduct comprehensive experiments to evaluate the effectiveness of our proposed Soup Of Category Experts (SoCE) methodology across three diverse benchmarks that span different LLM capabilities:

- 1. Berkeley Function Calling Leaderboard (BFCL) Patil et al.: evaluates tool calling and function invocation capabilities of LLMs across multiple categories including multi-turn interactions, irrelevance detection, and cross-language function calling.
- 2. Multilingual Grade School Math Benchmark (MGSM) (Shi et al., 2022): assesses mathematical reasoning abilities across multiple languages, testing both computational skills and cross-lingual generalization.
- 3. ∞-Bench (Zhang et al., 2024b): evaluates long-context processing capabilities, testing models’ ability to maintain coherence and extract information from extended sequences. We use a subset containing 3 categories.
- 4. FLORES-101 (Goyal et al., 2021): Measures translation quality and multilingual understanding across a diverse set of language pairs. We use a subset containing translations in 18 languages to and from English. These are only used for ablation studies. We will refer to this subset at FLORES-36.

For all benchmarks, we compare individual models with the following model soups:

|Model<br><br>|BFCL Accuracy|
|---|---|
|xLAM-2-70b<br><br>(Prabhakar et al., 2025) CoALM-70B<br><br>(Acikgoz et al., 2025) watt-tool-70B<br><br>(gm8.ai, 2024a) functionary-medium-70B (MeetKai, 2024)<br><br>|78.56% 54.49% 73.57% 62.32%|
|Uniform Souping (Wortsman et al., 2022)<br><br>|68.33%|
|Uniform Souping with SoCE Model Selection|78.40%|
|SoCE (Proposed Method)<br><br>|80.68%|

(a) 70 billion parameters models

|Model<br><br>|BFCL Accuracy|
|---|---|
|xLAM-2-8b<br><br>(Prabhakar et al., 2025) ToolACE-2-8B<br><br>(Liu et al., 2024) watt-tool-8B<br><br>(gm8.ai, 2024b) BitAgent-8B<br><br>(bittensor, 2024) CoALM-8B<br><br>(Acikgoz et al., 2025)<br><br>|72.37% 68.73% 67.79% 67.49% 54.52%|
|Uniform Souping (Wortsman et al., 2022)<br><br>|69.80%|
|Uniform Souping with SoCE Model Selection|74.01%|
|SoCE (Proposed Method)|76.50%|

(b) 8 billion parameter models

- Table 1 BFCL Performance of 8 billion and 70 billion parameter models. The first 4 entries are the ingredient models and the last 3 are Souped Models. Uniform Souping refers to the baseline where all models are combined together with the same weight, Uniform Souping with model selection only combines the selected models using the anti-correlation criterion and SoCE is the proposed method that augments Uniform Souping with model selection with weights optimization.

- 1. Uniform Souping (All Candidate Models) Wortsman et al. (2022): Baseline approach uniformly averaging all candidate models.
- 2. Uniform Souping with SoCE Model Selection: Uniform weighting applied to our strategically selected models based on anti-correlated categories.
- 3. SoCE (Weighted Souping with Model Selection) : Complete proposed methodology with both strategic model candidate selection and optimized weighting.

This experiment design enables targeted ablation studies: comparing (1) vs (2) isolates our candidate selection benefits, (2) vs (3) quantifies weight optimization gains, and (1) vs (3) demonstrates overall SoCE performance improvements. We also compare individual models, to quantify improvements of souping over models some of which are considered state-of-the-art in their capabilities.

- 4.2 Numerical Takeaways

###### 4.2.1 Benchmark Performance

We compare souping for two sets of models on BFCL, i.e., 70 billion and 8 billion parameter dense models. For 70B models, we identified a total of 4 model candidates from the official leaderboard and employed the proposed technique, SoCE, on these models. SoCE achieved 80.68% accuracy, establishing a new state-of-the-art with a 2.7% improvement over the previous best-performing individual model, xLAM-2-70b-fc-r (Prabhakar et al., 2025) (78.56%). The optimal configuration utilized xLAM-2-70b-fc-r, CoALM-70B (Acikgoz et al., 2025), and watt-tool-70B gm8.ai (2024a)(weight:

- 0.3), with 0.5, 0.2 and 0.3 weights, respectively. For 8B models, SoCE achieved 76.50% accuracy, surpassing the previous state-of-the-art within the 8B model size, xLAM-2-8b-fc-r (Prabhakar et al., 2025) by 5.7% relative, with optimal weights of 0.7, 0.2, and 0.1 for xLAM-2-8b-fc-r, ToolACE-2-8B (Liu et al., 2024), and watt-tool-8B (gm8.ai, 2024b) respectively. We also show the results on the ablation of candidate selection by comparing Uniform Souping (all candidate models) Wortsman et al. (2022) with Uniform Souping with SoCE model selection. The results show that model selection for both 70B and 8B boosts performance. We further compare Uniform Souping with SoCE model selection with SoCE to understand the impact of weight optimization on models in the soup, and it can be seen that it leads to a relative improvement of 2.28% for 70 billion model and 3.44% for 8 billion model.

We present the results on the Multilingual Grade School Math (MGSM) Benchmark (Shi et al., 2022) in Table 2a. Similar to BFCL, we observed that SoCE performs better than the candidate models and uniform souping. We consider four open weight models having 6.74 billion parameters for model souping: MetaMathOctopus-7B (Chen et al., 2023), MetaMathOctopus-MAPO-DPO-7B (She et al., 2024), MathOctopus-MAPO-DPO-7B (She et al., 2024), and Mathoctopus-Parallel-7B (She et al., 2024). We present the results for the uniform souping (Wortsman et al., 2022), uniform souping with SoCE candidate selection, and SoCE in Table 2a. Uniform souping, considering all candidates, leads to performance regression compared to the best candidate models.

|Model<br><br>|MGSM Accuracy|
|---|---|
|MetaMathOctopus-7B (Chen et al., 2023) MetaMathOctopus-MAPODPO-7B (She et al., 2024) MathOctopus-MAPO-DPO-7B (She et al., 2024) Mathoctopus-Parallel-7B (She et al., 2024)<br><br>|41.9% 50.9% 39.0% 35.5%|
|Uniform Souping (Wortsman et al., 2022)<br><br>|47%|
|Uniform Souping with SoCEModel Selection|47.8%|
|SoCE (Proposed Method)|51.7%|

(a) Results for MGSM Benchmark: First 4 rows are ingredient models and the last 3 rows are souped models. It can be seen that uniform souping of all four models leads to performance regression compared to MetaMathOctopus-MAPO and uniform soup with candidate selection reduces this regression. Weight tuning in SoCE improves the final performance by

- 1.57% relative compared to the best baseline model.

|Model<br><br>|∞Bench Accuracy|
|---|---|
|Model Candidate 1<br>Model Candidate 2<br>Model Candidate 3<br>Model Candidate 4<br>Model Candidate 5<br>|27.24% 24.87%<br><br>26.72%<br><br>27.24%<br><br><br>27.44%<br><br>|
|Uniform Soup (Wortsman et al., 2022)<br><br>|27.44%|
|Uniform Soup with SoCEModel Selection|27.85%|
|SoCE(Proposed Method)<br><br>|28.0%|

(b) ∞-Bench accuracy of LLAMA 3 70B architecture based models. It can be seen that model selection (corresponding to uniform souping with SoCE Model Selection row) and weight tuning (corresponding to SoCErow) leads to improvements. Overall, SoCE improves the best models candidate by 0.66%.

- Table 2 Results of SoCE on the MGSM and ∞-Bench benchmarks.

We also perform similar ablations to ∞-Bench Zhang et al. (2024a) by training 5 Llama 3 architecture checkpoints (Dubey et al., 2024) of 70 billion parameters with variations of the same data mix to understand if souping is useful beyond tool calling and math for LLMs. We present these results in Table 2b. We can see that even though the candidate models have similar performance because of being trained on variants of a single data mix, model souping is still helpful in improving performance. We see that uniform souping does not regress performance, but uniform souping with SoCE model selection improves performance by 1.15%. Further, SoCE led to a 2.05% lift in performance compared to the best model candidate, also demonstrating that both weight tuning and candidate selection plays a role in performance improvement.

###### 4.2.2 Large Scale Model Analysis: BFCL, FLORES-36 and ∞Bench

We conducted model souping and evaluation experiments on a comprehensive set of candidate checkpoints across the MGSM, BFCL, FLORES-36, and ∞Bench benchmarks to systematically investigate the effects of souping. Our analysis yielded the following key findings:

- • Increased Linear Correlation Across Categories after Souping: Following souping, the performance metrics across different categories exhibit a marked increase in linear correlation. As illustrated in Figure 2, the top row presents the Pearson correlation coefficients of checkpoint performance prior to souping, while the bottom row shows the correlations post-souping for BFCL, FLORES-36, and ∞Bench, respectively. These results indicate that souping leads to a more consistent and linearly correlated performance profile across categories.
- • Consistent Performance Gains Across categories: We observe higher average gains across the majority of

[Figure 2]

[Figure 3]

[Figure 4]

CorrelatedAnti-correlatedUn-correlated

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

(a) BFCL (b) MGSM

(c) ∞Bench

- Figure 2 Intra-benchmark performance Pearson Correlation (Pre-, and Post- souping): This is the pearson correlation matrix of metrics across different categories on (L-R: ) ∼800 souped (above) and unsouped (below) checkpoints on BFCL, 40 checkpoints on MGSM and 80 checkpoints for ∞Bench. We observe that after souping, the performance across all the categories become highly linearly correlated.

categories and souping experiments. For example, in the case of checkpoints fine-tuned on Llama-70B, for 35 out of 37 souping experiments, souped candidates had a higher metric score for more than 20 categories (out of 36) with a net positive gain observed across all categories (see Figure 5).

The implications of these findings are twofold. First, the process of training large models is often ad hoc, with optimal performance across diverse capabilities typically achieved through extensive experimentation with model parameters and data proportions—an approach that is resource-intensive. Our results suggest that, within a given benchmark, cooperative gains can be achieved in a more deterministic and systematic manner by leveraging strong model baselines and formalized souping techniques. Second, the observed overall improvements in both the majority of categories and the average metric values across souping candidates provide strong evidence for the efficiency and performance benefits of adopting end-to-end souping methodologies.

###### 4.2.3 SoCE Candidate selection experiments

The SoCE framework comprises two primary components: (1) leveraging correlation patterns to identify experts for uncorrelated categories, and (2) assigning appropriate weights to these experts. Our experimental results demonstrate the effectiveness of this candidate selection strategy. Specifically, we analyze the significance of diversity in checkpoint performance and the impact of anti-correlations within the benchmarks. On the MGSM and BFCL benchmarks, SoCE yields substantial improvements, particularly in scenarios where distinct experts can be identified across anti-correlated checkpoints, as illustrated in Figures 3a and 3b. Conversely, in cases where it is challenging to discern clear experts for different categories—such as with the FLORES-36 benchmark (see Table C)—the overall performance gains over baseline methods are more modest. Similarly, when benchmarks exhibit minimal anti-correlation in checkpoint performance, the benefits of model souping are limited, resulting in only marginal improvements, as shown in Table 2b and Figure 2c.

Preliminaries: Cooperative Game Theory and the Shapley Value: Cooperative game theory addresses situations where multiple agents collaborate as a team. A (transferable-utility) cooperative game is specified by a set of players A = {a1,a2,...,an} and a characteristic function v : 2A → R, which assigns to each coalition C ⊆ A a real value v(C) representing the utility or performance achieved by that coalition. The Shapley value Shapley (1953) is a fundamental concept in cooperative game theory, providing a principled way to quantify each player’s

|Model|BN|DE|EN|ES|FR|JA|RU|SW|TH|ZH|AVG|
|---|---|---|---|---|---|---|---|---|---|---|---|
|M1|27.6%|45.6%|69.2%|52.4%|47.6%|35.6%|46.0%|27.6%|22.4%|44.8%|41.9%|
|M2|35.6%|56.0%|69.2%|60.4%|57.2%|46.4%|54.4%|37.6%|37.2%|54.8%|50.9%|
|M3|29.2%|42.4%|45.6%|43.6%|37.2%|40.0%|39.2%|30.4%|36.0%|46.4%|39.0%|
|M4|26.8%|40.0%|45.2%|43.2%|34.8%|30.8%|33.6%|30.0%|30.4%|40.0%|35.5%|
|SoSe|40.0%|56.4%|70.8%|61.2%|54.8%|48.4%|53.2%|42.4%|41.2%|54.0%|52.2%|

[Figure 9]

[Figure 10]

CorrelatedAnti-correlatedUn-correlated

- (b)

[Figure 11]

[Figure 12]

[Figure 13]

- (c)

(a)

- Figure 3 Shapley value Analysis: Figure (a) displays the linear correlation amongst categories of the MGSM benchmarks across 80 checkpoints. Table (b) shows the performance per MGSM benchmark category for a set of 4 finetuned huggingface candidate models, and Figure (c) shows the Shapley value plots for single, pairs and triplets of candidates. We clearly see that M1 and M2 are the experts for the least-correlated categories (ES-EN and ZH-EN) and they are also the strongest contributor pair. In parallel, M1 is a strong parent and M4 is a weak parent and the shapley values reflect that as well showcasing that the strength of SoCE’s candidate selection approach.

individual contribution to the team’s overall success, while satisfying key fairness properties Dubey (1975). Formally, for a player ai ∈ A in a game with characteristic function v, the marginal contribution of ai to a coalition C ⊆ A \ {ai} is given by m(ai,C) = v(C ∪ {ai}) − v(C) Let Π denote the set of all permutations of the n players, i.e., each π ∈ Π is a bijection π : A → A. For a permutation π, define pred(ai,π) as the set of players preceding ai in π. The marginal contribution of ai in permutation π is m(ai,π) = v(pred(ai,π) ∪ {ai}) − v(pred(ai,π)) The Shapley value ϕi for player ai is the average marginal contribution of ai over all possible permutations: ϕi =

1 n! π∈Π [v(pred(ai,π) ∪ {ai}) − v(pred(ai,π))] Equivalently, the Shapley value represents the expected increase in

team value when ai joins the team at a random position in a random ordering of the players.

Shapley Value Analysis in the Context of Model Souping: For the Shapley value analysis, we consider a benchmark B, a metric F, and a set V of candidate models for souping. As candidates, we analyze either (1) individual checkpoints, (2) pairs, or (3) triplets, treating each candidate as a player and each subset C ⊆ V as a coalition. This forms a cooperative game where the characteristic function is defined as v : 2V → R, mapping each subset of candidates to the performance achieved by souping only those candidates. For any coalition C ⊆ V , the team performance is given by v(C) = Performance(Souped(C);B,F), where Souped(C) denotes the model obtained by souping the candidates in C, evaluated on benchmark B using metric F. We view the souped models as participants / agents working together to build the best model, bringing different skills / strengths to the table (as shown by the categories within the benchmarks). By souping models, we allow covering these different skills well, as showcased by an improvement in the overall metric. The Shapley value indicates the relative contributions of the sub-models using souping as the combination function in the context of the skills of the set of all models.

We define our Shapley computation by the following parameters; MGSM as the benchmark, the performance metric is average accuracy, the characteristic function is souping and the set of candidates are (1) 4 open-source (OSS) models finetuned on LLama-7B (2) 6 pairs of the OSS models(3) 4 possible triplets these for model souping using a set of four candidate models, evaluating all possible combinations. Separately, we also apply the SoCE framework to the same set. As shown in Figure 6b, our analysis reveals that model contributions are not uniform; candidates and subsets selected via SoCE exhibit significantly higher Shapley values. This finding underscores the critical role of SoCE’s candidate

selection in enhancing ensemble performance. Additionally, our results indicate that disproportionately weighting a weak parent model can substantially decrease the average score (e.g., assigning weights of (0.05, 0.05, 0.9) on M4 reduces the mean to 37.0). In contrast, consistently including a strong parent model, such as M2 in combination with other candidates, reliably increases the average score. We also extend the same method of analyses to other benchmarks (see Figures 6c and 6a).

#### 5 Discussion

In the general case, the presented method can be further explored as a broader solution for adding new domain or capability to the existing open models: overcoming the overfitting to add a new unique capability to the existing models. We anticipate the future research covering:

Application to many more tasks: The presented method can we widely applied to the multi-task training objectives, including

- • multilingual applications: a task-specific checkpoint + a language specific checkpoint merged (like in Akiba et al.

(2025));

- • application to combine anti-correlated capabilities: tool calling + reasoning + coding expert checkpoints could be combined without additional training;
- • enabling specific use cases, where the training data should remain private, but the checkpoint and its unique capability can be spread across the model family by souping.

Societal Impact: While the yearly increase in computational scale leads to growing inequality in academic access to frontier models (see the detailed analysis in Kogkalidis and Chatzikyriakidis (2024)), the presented method proposes both easily accessible low cost opportunities for the broader community efforts and also promotes the iterative reuse of the existing pretrained models, which can lead to significant savings in computing resources. This framework alone can significantly expand the collaboration opportunities in the open source landscape: LLama-derivatives model family currently approaches 150k models1 with groups of the same size and architecture that can be souped.

#### 6 Conclusion

This work introduces SoCE, a category-aware, automatic model souping technique that achieves state-of-the-art results across diverse domains. We demonstrate that this approach not only improves overall benchmark performance but also enhances consistency and robustness across categories. We hope these findings inspire further research into efficient model aggregation and reuse within the community.

We believe that the presented method can propose a new perspective for the OSS community to reuse and revive derivative models of the same architecture.

#### 7 Acknowledgments

We would like to express our sincere gratitude to Andrew Budker and Ricardo Silveira Cabral for their support throughout this work. We also thank Shishir Patil for his insightful advice on the Berkeley Function Calling Leaderboard evaluations.

#### Limitations

###### 7.1 General Method Limitations

The results presented in this paper advance our understanding of model souping, but some methodological and conceptual limitations warrant more discussion.

Benchmark structure: The first limitation we want to highlight is a key assumption that a given benchmark already has some sub-categorical splits which have enough data points such that we can estimate correlations with enough model. There are several benchmarks today which do not come with pre-classified sub-categories and hence, we propose

1https://huggingface.co/models?search=llama

benchmark clustering as a future work to this approach. Furthermore, for selecting candidates for SoCEapproach for BFCL and MGSM benchmarks, we have used the leaderboard scores directly based on the assumption that in the future setup, we would have an oracle development set to select candidates without leaking any information.

Application in the model training practice: In this work, we have only tested the souping of the ’final’ posttrained and aligned checkpoints. As other works have shown, the models can be souped after pre-training Li et al. (2025), after posttraining Wortsman et al. (2022), with adapters2. We do not recommend souping of the checkpoints from different training stages, as well as souping unaligned or uncensored models with the aligned ones to avoid the inheritance of risks.

We also want to highlight that all the experiments are carried out on Llama 3 (Dubey et al., 2024) derivative models and essentially have the same pretrained checkpoint. We are currently unaware if souping requires the same pretrained checkpoint or can it work with different pretrained checkpoints as well.

Model Architecture: Lastly, this work has focused on souping largely for dense models and assumes that the same methodology can be applied to other architectures such as mixture of experts.

###### 7.2 Scaling Laws

While model souping has demonstrated promising results, it is important to consider potential limitations and diminishing returns as more models are combined. The extent to which performance continues to improve may depend on the diversity and capabilities of the individual models being souped. We have not yet systematically tested the upper bounds of this approach, and it is possible that there exists an optimal strategy for souping checkpoints that varies according to the capability differences between the models involved. Further empirical investigation is needed to better understand these scaling dynamics.

#### 8 Ethical Considerations

It is important to note that models undergoing souping procedures should be released with the disclosure and detailed description of the said procedures. Otherwise, mechanistic interpretability research on such checkpoints can be compromised: works like Sun et al. (2024); Yu et al. (2025) would show blurred results of an unknown nature.

2https://huggingface.co/docs/peft/en/developer_guides/model_merging

#### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Emre Can Acikgoz, Jeremiah Greer, Akul Datta, Ze Yang, William Zeng, Oussama Elachqar, Emmanouil Koukoumidis, Dilek Hakkani-Tür, and Gokhan Tur. Can a single model master both multi-turn conversations and tool use? coalm: A unified conversational agentic language model. arXiv preprint arXiv:2502.08820, 2025.

Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, and David Ha. Evolutionary optimization of model merging recipes. Nature Machine Intelligence, 7(2):195–204, January 2025. ISSN 2522-5839. doi: 10.1038/s42256-024-00975-8. URL http://dx.doi. org/10.1038/s42256-024-00975-8.

Haoli Bai, Lu Hou, Lifeng Shang, Xin Jiang, Irwin King, and Michael R Lyu. Towards efficient post-training quantization of

pre-trained language models. Advances in neural information processing systems, 35:1405–1418, 2022. bittensor. Bitagent-8b. https://huggingface.co/BitAgent/BitAgent-8B, 2024. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam,

Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Dongmei Zhang, and Jia Li. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. arXiv preprint arXiv:2310.20246, 2023.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

Pradeep Dubey. On the uniqueness of the shapley value. International Journal of Game Theory, 4:131–139, 1975. URL

https://api.semanticscholar.org/CorpusID:119742421. gm8.ai. Watt.ai 70b. https://huggingface.co/watt-ai/watt-tool-70B, 2024a. gm8.ai. Watt.ai 8b. https://huggingface.co/watt-ai/watt-tool-8B, 2024b. Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio

Ranzato, Francisco Guzman, and Angela Fan. The flores-101 evaluation benchmark for low-resource and multilingual machine translation, 2021. URL https://arxiv.org/abs/2106.03193.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Ajay Kumar Jaiswal, Shiwei Liu, Tianlong Chen, Ying Ding, and Zhangyang Wang. Instant soup: Cheap pruning ensembles in a single pass can draw lottery tickets from large models. In International Conference on Machine Learning, pages 14691–14701. PMLR, 2023.

Dong-Hwan Jang, Sangdoo Yun, and Dongyoon Han. Model stock: All we need is just a few fine-tuned models, 2025. URL https://arxiv.org/abs/2403.19522.

Anat Kleiman, Gintare Karolina Dziugaite, Jonathan Frankle, Sham Kakade, and Mansheej Paul. Soup to go: mitigating forgetting during continual learning with model averaging. arXiv preprint arXiv:2501.05559, 2025.

Konstantinos Kogkalidis and Stergios Chatzikyriakidis. On tables with numbers, with numbers, 2024. URL https://arxiv.org/ abs/2408.06062.

Yunshui Li, Yiyuan Ma, Shen Yan, Chaoyi Zhang, Jing Liu, Jianqiao Lu, Ziwen Xu, Mengzhao Chen, Minrui Wang, Shiyi Zhan, Jin Ma, Xunhao Lai, Deyi Liu, Yao Luo, Xingyan Bin, Hongbin Ren, Mingji Han, Wenhao Hao, Bairen Yi, LingJun Liu, Bole Ma, Xiaoying Jia, Xun Zhou, Siyuan Qiao, Liang Xiang, and Yonghui Wu. Model merging in pre-training of large language models, 2025. URL https://arxiv.org/abs/2505.12082.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, et al. Toolace: Winning the points of llm function calling. arXiv preprint arXiv:2409.00920, 2024.

MeetKai. Meetkai functionary medium model 70b. https://huggingface.co/meetkai/functionary-medium-v3.1, 2024. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal,

Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning.

Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, et al. Apigen-mt: Agentic pipeline for multi-turn data generation via simulated agent-human interplay. arXiv preprint arXiv:2504.03601, 2025.

Regent’s, ParkLondon, Ukj, and . Utans. Weight averaging for neural networksand local resampling. 1996. URL https://api. semanticscholar.org/CorpusID:475398.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.

Lloyd S Shapley. A value for n-person games. In Harold W. Kuhn and Albert W. Tucker, editors, Contributions to the Theory of Games II, pages 307–317. Princeton University Press, Princeton, 1953.

Shuaijie She, Wei Zou, Shujian Huang, Wenhao Zhu, Xiang Liu, Xiang Geng, and Jiajun Chen. Mapo: Advancing multilingual reasoning through multilingual alignment-as-preference optimization. arXiv preprint arXiv:2401.06838, 2024.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners, 2022. URL https://arxiv.org/abs/2210.03057.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on machine learning research, 2023.

Mingjie Sun, Xinlei Chen, J. Zico Kolter, and Zhuang Liu. Massive activations in large language models, 2024. URL https: //arxiv.org/abs/2402.17762.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai,

Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman

Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971,

- 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288,

- 2023b.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR, 2022.

Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao. Adamerging: Adaptive model merging for multi-task learning, 2024. URL https://arxiv.org/abs/2310.02575.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Forty-first International Conference on Machine Learning, 2024.

Mengxia Yu, De Wang, Qi Shan, Colorado J Reed, and Alvin Wan. The super weight in large language models, 2025. URL https://arxiv.org/abs/2411.07191.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. ∞Bench: Extending long context evaluation beyond 100K tokens. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15262–15277, Bangkok, Thailand, August 2024a. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.814.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, et al. infbench: Extending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15262–15277, 2024b.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Max Zimmer, Christoph Spiegel, and Sebastian Pokutta. Sparse model soups: A recipe for improved pruning via model averaging. arXiv preprint arXiv:2306.16788, 2023.

## Appendix

#### A Understanding Impact on Other Benchmarks while souping for one benchmark

We evaluated checkpoints that were the result of applying the SoCE-framework on open-source model checkpoints from the BFCL leaderboard on other benchmarks such as Hellaswag Zellers et al. (2019), IFEval Zhou et al. (2023) and BIG bench Hard Srivastava et al. (2023) and observed that SoCE did not overfit to the benchmark whose performance correlations it exploited. It also did not show any major regression on these unrelated benchmarks, achieving either comparable or more performant metric results than the candidate baselines. These results are showcased in Table A

|Models|BFCL-v3<br><br>|Hellaswag|IFEval<br><br>|BBH|
|---|---|---|---|---|
|8b-m1 8b-m2 8b-m3 Souped Model-3x Souped Model-2x<br><br>|69.50% 72.37% 67.50% 76.50%<br><br>76.17%<br><br>|78.92%<br><br>77.57%<br>78.59% 78.37% 78.61%<br>|51.68% 45.00% 50.36%<br><br>50.60%<br><br>51.92%<br>|44.80% 63.10% 36.54% 63.06% 58.92%<br><br>|

Table 3 BFCL 8b Souping: Overfitting Results: We evaluated checkpoints that were the result of applying the SoCEframework on open-source model checkpoints from the BFCL leaderboard on other benchmarks such as Hellaswag Zellers et al. (2019), IFEval Zhou et al. (2023) and BIG bench Hard Srivastava et al. (2023) and observed that SoCE did not overfit to the benchmark whose performance correlations it exploited. It also did not show any major regression on these unrelated benchmarks, achieving either comparable or more performant metric results than the candidate baselines.

#### B Deep dive on BFCL State-of-the-art Performance: Winrate Analysis of Souped Models and Individual Candidates

On the BFCL benchmark, for 70 billion parameters dense models, we analyze the win rate of SoCE over individual models in the soup: xLAM-2-70b-fc-r (Prabhakar et al., 2025), CoALM-70B (Acikgoz et al., 2025), and watt-tool70B gm8.ai (2024a).

We first examine the proportion of tasks solved by SoCE that were also solved by each individual model. SoCE successfully completes 97.2% of the tasks solved by xLAM-2-70b-fc-r (Prabhakar et al., 2025), 97.1% of those solved by CoALM-70B (Acikgoz et al., 2025), and 97.1% of those solved by watt-tool-70B gm8.ai (2024a). This indicates that SoCE retains most of the capabilities of individual models in the soup.

Next, we assess SoCE’s win rate on tasks where individual models fail. When individual models all fail on a given task, SoCE succeeds in 8.4% of cases (32 out of 380 tasks), demonstrating SoCE’s ability to solve new tasks that none of the models in the soup was able to handle.

Further, when one individual model in the soup fail on a given task, and not the others, SoCE is able to complete the task in 93.0% of cases, highlighting the robustness of the proposed souping approach.

#### C Shapley Values Figures

We computed Shapley values Shapley (1953) for a small batch of 4-5 models to understand the contributions of model candidates (single, in pairs and in triplets) and their correlation with the SoCE candidate selection mechanism. We found that SoCE’s selected candidates often contributed more significantly under the souping paradigm. Please refer to Figure [6] for the plots.

[Figure 14]

C a

- t e g o r y

C o

- u n t

Souped > Both parents

[Figure 15]

Souped > at least 1 parent

[Figure 16]

- B L E U

S

- C O R E

Min BLEU (parents)

[Figure 17]

Max BLEU (parents)

[Figure 18]

BLEU (souped)

[Figure 19]

TRIPLET (parent1, parent2, souped) INDEX

- Figure 5 Analysis of performance of 37 souped and unsouped checkpoint on Flores-36): The x-axis contains the index of one souping triplet, i.e, the two parents and the souped output on the FLORES-36 benchmark. The y-axis in the top figure is the count of the number of categories in FLORES-36 and in the bottom figure, it the BLEU metric score. The orange line maps how many souped outputs have a higher score than the at least one parent per category, the blue line maps how many souped outputs have a higher score than both the souped candidates in the top figure. In the bottom figure, the green line shows the smaller average BLEU score between the parents, the purple line shows the higher BLEU score and the red line shows the souped candidate score.

[Figure 20]

[Figure 21]

[Figure 22]

(a) ∞-Bench for a set of 5 finetuned candidate 70B models.

[Figure 23]

[Figure 24]

[Figure 25]

- (b) MGSM for a set of 4 finetuned huggingface candidate models.

[Figure 26]

[Figure 27]

[Figure 28]

- (c) FLORES-36-Bench for a set of 5 finetuned candidate 70B models.)

- Figure 6 Shapley values: We computed Shapley values Shapley (1953) for a small batch of 4-5 models to understand the contributions of model candidates (single, in pairs and in triplets) and their correlation with the SoCE candidate selection mechanism. We found that SoCE’s selected candidates often contributed more significantly under the souping paradigm.

|Model<br><br>|avg_bleu|eng_deu<br><br>|deu_eng|eng_spa<br><br>|spa_eng|eng_tha<br><br>|tha_eng|eng_por<br><br>|por_eng|
|---|---|---|---|---|---|---|---|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5 SoCE<br><br><br>|38.97 38.74 39.07 39.04 38.94 39.19|45.60 45.76 45.85 45.95 45.75 46.22<br><br>|50.18 50.03 50.55 50.09 50.38 50.39|35.49 35.77 35.78 35.35 35.15 32.06<br><br>|35.49 35.77 35.78 35.35 35.15 35.72<br><br>|35.29 35.48 35.82 35.89 35.59 36.26|36.33 35.32 36.43 36.16 36.10 36.57<br><br>|52.96 53.01 53.12 52.85 52.59 53.27<br><br>|54.79 54.41 54.75 54.61 54.42 54.83|

- (a) BLEU scores on FLORES-36 (German, Spanish, Thai, Portuguese.) 70B finetuned models Souping per language translation split. (Part 1 of 4)

|Model<br><br>|eng_vie|vie_eng|eng_ind<br><br>|ind_eng<br><br>|eng_hin|hin_eng|eng_fra|fra_eng<br><br>|eng_ita|
|---|---|---|---|---|---|---|---|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5 SoCE<br><br><br>|41.84 41.70 41.74 41.82 41.52 42.08|40.75 40.27 40.82 40.60 40.75 41.06<br><br>|44.99 44.91 44.98 45.25 44.77 45.22|48.07 47.51 47.96 48.17 48.35 48.50<br><br>|35.11 35.32 34.88 34.96 34.86 35.47|44.95 44.41 44.74 44.69 44.73 45.08<br><br>|55.55 55.49 55.68 55.80 55.52 56.15<br><br>|50.88 50.47 51.06 50.78 50.85 51.26|35.08 35.16 34.86 35.09 34.97 35.26<br><br>|

- (b) BLEU scores on FLORES-36 (Vietnamese, Indonesian, Hindi, French, Italian.) 70B finetuned models Souping per language translation split. (Part 2 of 4)

|Model|ita_eng|eng_arb<br><br>|arb_eng|eng_ben<br><br>|ben_eng|eng_zhoHs<br><br>|zhoHs_eng|eng_zhoHt<br><br>|zhoHt_eng|
|---|---|---|---|---|---|---|---|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5 SoCE<br><br><br>|38.90 39.20 38.95 38.52 38.69 39.31|34.96 35.32 35.06 35.43 35.14 36.05<br><br>|45.80 45.59 46.01 45.97 46.06 46.28<br><br>|29.68 29.94 29.94 30.13 29.98 30.31|37.55 37.10 37.88 37.92 37.45 37.94<br><br>|33.56 33.32 33.59 33.58 33.66 33.98|34.98 34.43 34.85 34.41 34.64 35.06<br><br>|17.57 17.39 18.92 21.28 18.63 20.34|31.99 31.22 31.60 31.53 31.83 31.89<br><br>|

- (c) BLEU scores on FLORES-36 (Italian, Arabic, Bengali, Chinese (Simplified/Traditional)) 70B finetuned models Souping per language translation split. (Part 3 of 4)

|Model<br><br>|eng_jpn<br><br>|jpn_eng|eng_kor|kor_eng|eng_rus<br><br>|rus_eng|eng_tur|tur_eng<br><br>|eng_tgl<br><br>|tgl_eng|
|---|---|---|---|---|---|---|---|---|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5 SoCE<br><br><br>|25.20 24.57 24.97 24.93 24.75 25.34|32.83 32.18 33.64 33.17 33.34 33.60<br><br>|24.44 24.28 24.66 23.67 23.87 24.84|34.67 33.71 34.93 34.35 34.53 34.77<br><br>|38.63 38.24 38.57 38.74 38.77 38.88<br><br>|42.32 41.96 42.22 42.13 41.88 42.45<br><br>|35.78 36.02 35.78 35.91 36.26 36.43|43.03 42.37 43.44 42.80 42.93 43.26<br><br>|33.37 33.30 32.38 32.74 33.00 33.59<br><br>|47.80 47.53 48.29 48.14 48.17 48.35|

- (d) BLEU scores on FLORES-36 (Japanese, Korean, Russian, Turkish, Tagalog.) 70B finetuned models Souping per language translation split. (Part 4)

- Table 5 Per category performance deep dive on FLORES-36 benchmark: Each language is either translated to (language_english) or from (english_language) english. These models are less diverse, i.e, it is difficult to find category experts, and display evidence of marginal gains consistently on average and per sub-benchmark using SoCE.

#### D Qualitative examples of souping results

###### D.1 Wins when candidates fail

In tables 6 and 7, we show 2 examples of tasks where our proposed approach, SoCE, applied to 70 billion parameters dense models, solves problems that none of the individual models in the soup could to solve.

For task simple_267, SoCE correctly omits the month parameter, while other models erroneously include month="upcoming". For task java_52, SoCE correctly uses "Element.class" versus "Element" for the node_type parameter.

|Task|Description|
|---|---|
|java_52|In a Java XML processing context, help me obtain a list of all child elements of type Element from a Node representing a SAML assertion SAMLAssertionNode?|
|Model<br><br>|Output|
|xLAM-2-70b-fc-r ×|["name": "SamlObjectSignerTests.getChildren", "arguments": "node": "SAMLAssertionNode", "node_type": "Element"]|
|watt-tool-70B ×<br><br>|[SamlObjectSignerTests.getChildren(node=’SAMLAssertionNode’, node_type=’Element’)]|
|CoALM-70B ×|[SamlObjectSignerTests.getChildren(node="SAMLAssertionNode", node_type="Element")]|
|SoCE ✓|[SamlObjectSignerTests.getChildren(node="SAMLAssertionNode", node_type="Element.class")]|

- Table 6 Comparison of model outputs for task java_52 in BFCL v3.

|Task|Description|
|---|---|
|simple_267<br><br>|Find the top rated modern sculpture exhibition happening in New York in the upcoming month|
|Model|Output|
|xLAM-2-70b-fc-r ×<br><br>|["name": "find_exhibition", "arguments": "location": "New York City, NY", "art_form": "sculpture", "month": "upcoming", "user_ratings": "high"]|
|watt-tool-70B ×|[find_exhibition(location="New York City, NY", art_form="sculpture", month="upcoming", user_ratings="high")]|
|CoALM-70B ×<br><br>|[find_exhibition(location="New York", art_form="modern sculpture", month="upcoming", user_ratings="high")]|
|SoCE ✓<br><br>|[find_exhibition(location=’New York City, NY’, art_form=’sculpture’, user_ratings=’high’)]|

- Table 7 Comparison of model outputs for task simple_267 in BFCL v3.

