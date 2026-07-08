# arXiv:2502.04235v2[cs.CL]19May2025

[Figure 1]

## Reformulation for Pretraining Data Augmentation

Xintong Hao1, Ruijie Zhu1,2, Ge Zhang1, Ke Shen1, Chenggang Li1 1ByteDance Seed, 2University of California, Santa Cruz

Abstract

Despite the impressive capabilities of large language models across various tasks, their continued scaling is severely hampered not only by data scarcity but also by the performance degradation associated with excessive data repetition during training. To overcome this critical bottleneck, we propose the Massive Genre-Audience (MGA) reformulation method, a lightweight and scalable data augmentation technique inspired by synthetic data methodologies. MGA systematically reformulates existing corpora into diverse, contextually-rich variations to mitigate the negative effects of repetition, and we introduce this approach along with the resulting 770 billion token MGACorpus in this work. We experimentally validate its core benefit by demonstrating superior performance against data repetition and upsampling in scaling scenarios (up to 13B parameters). Furthermore, comprehensive analysis investigates the role of prompt engineering in generation quality and reveals nuances in evaluating model capabilities using standard loss metrics. Our work shows that MGA provides a reliable pathway to substantially augment training datasets, effectively alleviating repetition bottlenecks and enabling more efficient scaling of large language models.

Date: May 20, 2025 Project Page: https://huggingface.co/datasets/ByteDance-Seed/mga-fineweb-edu

#### 1 Introduction

The remarkable success of Large Language Models (LLMs) heavily relies on the scale of model parameters and training data [1, 2]. Scaling laws demonstrate that improvements in model performance are increasingly dependent on data quantity and quality. However, the growth rate of available natural language corpora significantly lags behind the increasing demand for training data [3]. In traditional deep learning, data repetition has been a standard approach—training models for over 1,000 epochs on ImageNet is common and continues to yield improvements. Yet, in the pre-training stage of LLMs, excessive data repetition can degrade model performance and stability, creating a significant barrier to continued scaling efforts, particularly for the largest models. This raises a critical question: how can we fully utilize the potential of existing data in data-constrained situations?

Data augmentation has been widely employed to address similar challenges in traditional machine learning. However, conventional augmentation methods have proven ineffective for LLMs. One emerging approach involves leveraging LLMs themselves to synthesize high-quality training data [4, 5]. Data synthesis could theoretically generate limitless diverse training material, enabling dataset expansion without the negative consequences associated with excessive repetition.

However, prevailing data synthesis methods face significant hurdles. Many depend on large-scale models for generation, such as 12B dense models [4] or those with GPT-4-level capabilities [5], to ensure data quality.

1

Stage 1 Stage 2

(Genre, Audience) generation Document reformulated

SOURCE TEXT

Story Girl

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

cleaning

[Figure 8]

SLM-2

[Figure 9]

[Figure 10]

Dialogue Grandpa

[Figure 11]

[Figure 12]

cleaning

[Figure 13]

[Figure 14]

[Figure 15]

SLM-2

[Figure 16]

[Figure 17]

### …

### …

### …

SLM-1

[Figure 18]

[Figure 19]

[Figure 20]

Textbook Teacher

cleaning

[Figure 21]

SLM-2

[Figure 22]

[Figure 23]

Generate 5 pairs per inference pass

[Figure 24]

Reformulate one target per inference pass

- Figure 1 Overview of MGA framework. Our method expands the original corpus through a two-stage synthesis process. Each document is reformulated to 5 new documents, achieving 3.9× token number expansion while maintaining diversity through massive (genre, audience) pairs.

This effectively transforms synthetic datasets into “distillations” from larger models rather than true data augmentations. Also, using such large models to generate additional data during the actual pre-training process is clearly impractical from a computational standpoint. Furthermore, some approaches, like Phi and Cosmopedia, require sophisticated, pre-defined seed curation systems to manage data diversity [5, 6]. This dual reliance on massive models and complex seed management introduces substantial computational bottlenecks and scalability challenges, limiting their practicality for efficient pretraining corpus expansion specifically aimed at mitigating data repetition issues.

In this work, we propose MGA (Massive Genre-Audience reformulation), a more efficient approach designed to directly address the data repetition challenge. As illustrated in Figure 1, MGA utilizes a comparatively lightweight 3.3B MoE model. Crucially, it avoids complex external seed systems by adaptively generating diverse genre-audience pairs directly from raw input documents. This makes the process lightweight and scalable, offering a practical way to expand datasets while minimizing detrimental repetition. Our main contributions are:

- • We build and introduce the MGACorpus, a 770 billion token dataset based on existing high-quality text collections. We demonstrate that the MGACorpus achieves superior performance compared to the original corpus it expands upon, and also shows improved results against models trained on other synthetic datasets, underscoring the quality and effectiveness of the MGA approach.
- • We further perform a representative evaluation of data budget scaling strategies from a data augmentation perspective, revealing that MGACorpus yields consistent improvements across various model sizes (377M/1.7B/7B/13B) compared to data repetition and upsampling methods.
- • We analyze synthetic data collapse from two key perspectives, characterizing prompt engineering’s mitigating effects while revealing limitations of validation loss as a collapse detection metric, providing insights for future synthetic data optimization.

#### 2 Related Work

Data Curation While web-crawled data contains hundreds of trillions of tokens, stringent quality filters typically remove the majority of this content. Popular datasets like C4, Gopher, Dolma, RefinedWeb [7–10] use nonlearned heuristics method. And recently FineWeb-Edu [11], DCLM [12], FineFineWeb [13] focus on aggressive model-based and retrieval-based filtering. Such heavy filtering results in removal of 90% of tokens,

some researchers turn their attention to balance accuracy and data quantity [4]. However, this does not alter the fact that the total amount of high-quality data remains limited.

Repetition Training Studies on subset repetition training have revealed that model divergence tends to occur earlier as model parameters increase [14]. For scenarios training on entire datasets repeated, limiting to 4 epochs or fewer results in minimal efficiency degradation [15, 16]. Furthermore, [17] shows that some regularization techniques (e.g., dropout) and leveraging MoE architecture can help efficient LLM development on a broader scale. Overall, this topic remains understudied across different model architectures, data distributions, and repetition ratios.

Synthetic Pretrain Current synthetic data generation methods for language model pretraining can be primarily categorized into two approaches: seed based synthesis and raw text based rephrasing. The seed based method, exemplified by Phi-4 [5] and Cosmopedia [6], employs predefined seed systems and task templates to precisely control the type and structure of generated content. The rephrasing method, represented by WRAP [18] and Nemotron-CC [4], generates data by rephrasing web content into QA pairs and wiki-style texts, demonstrating significant effectiveness in processing noisy web text, though its benefits may be limited when applied to high-quality source data [19]. Additionally, Ge et al. [20] introduce an innovative text generation method based on billion personas, offering new insights into enhancing the diversity of synthetic data.

While existing approaches have made significant progress, they face key limitations: seed-based methods require complex initialization systems, limiting investigation of their scaling properties, while rephrasing-based approaches struggle to effectively augment high-quality corpus at scale. To address these limitations, our framework MGA bridges this gap by leveraging and extending the inherent diversity of existing corpus. Specifically, it adaptively generates multiple Genre and Audience seeds for each document of SmolLMCorpus [6], enabling 3.9x token expansion while maintaining diversity and quality.

#### 3 Massive Genre-Audience Reformulation

Preserving core knowledge while adapting content presentation for diverse audiences is the key motivation behind the MGA reformulation framework. As shown in Figure 1, the approach systematically expands original corpora through a two-stage synthesis process complemented by heuristic cleaning. Our implementation consists of three key components1: (1) A large language model serving as both LLM labeler and judger; (2) Task-specific Tool Models (Tool SLMs) applying W8A8 quantized [21] for efficiency; and (3) A balanced quality assessment mechanism defined as Limited Consistency. In the following sections, we first introduce the concept of genre-audience pairs that drive content diversity, then describe the reformulation process and quality evaluation framework, followed by detailed prompt engineering strategies that ensure optimal balance between information preservation and content variation.

- 3.1 Genre-Audience Pairs For pretraining corpus, there is a consensus among researchers to ensure diversity and quality. Inspired by [18];[20], we expand the simple rephrasing method from only few styles to massive genre-audience pairs.

- • Genre defines the knowledge expression framework through multiple dimensions: communication purpose (e.g., education, analysis), content structure (e.g., step-by-step tutorials, analytical reports), language style, and knowledge depth control. This framework guides how information should be reconstructed while preserving core concepts.
- • Audience profiles combine demographic factors (age, education, profession) with knowledge background and motivational characteristics. For example, a beginner-level first-aid guide would be reformulated differently for medical students versus office workers, while maintaining essential medical accuracy.

Our framework supports N genres and M audience types, theoretically enabling N×M unique reformulation patterns. To balance diversity and computational efficiency, we generate 5 genre-audience pairs per inference pass. This ensures more distinct reformulations per document than typical N-epoch repetitions (e.g., N≤4) often considered safe in LLM pretraining [15], aiming for novel augmentation while managing generation costs.

1Tool model details presented in Appendix B. Prompts and case studies in Appendix E.

- 3.2 Reformulation Once the genre-audience pairs are determined, the reformulation process follows a straightforward approach, as prompt presented in Appendix E.2. The key factor of reformulation is how to evaluate the output text, so we introduce the concept of ‘‘Limited Consistency’’ as criterion for quality controlling. This framework seeks to establish an optimal balance between textual variation and information preservation as shown in Prompt 1.

# Detailed Requirements For scoring judgment, the following standards must be followed:

- 1. The ‘scoring range’ is 1−5 points. You need to analyze and grasp each point mentioned in #Thought Process#, and give scores with distinction. Be strict, don’t be too lenient with scoring!

- 2. The ‘Reformulated Text’ is allowed to differ from the ‘Original Text’ in writing style, expression style, and focus points! This cannot be a basis for deducting points!

- 3. The ‘Reformulated Text’ is allowed to omit some information from the ‘Original Text’! Not all information from the ‘Original Text’ needs to be reflected in the ‘Reformulated Text’! The following situations will [NOT REDUCE] the score:

- 1. The ‘Reformulated Text’ can include information points not present in the ‘Original Text’

- 2. The additional content in the ‘Reformulated Text’ deviates significantly from the core information of the ‘ Original Text’

- 3. The expression style, order, and focus points of the ‘Reformulated Text’ differ from the ‘Original Text’ The following situations will [REDUCE] the score:

- 1. The information points in the ‘Reformulated Text’ differ so greatly from the ‘Original Text’ that it’s not apparent it was Reformulated from the ‘Original Text’

- 2. The ‘Reformulated Text’ lacks every information points present in the ‘Original Text’ Prompt 1 LLM judger prompt snippet.

In practice, we use the proportion of samples (score ≥ 3) as our primary metric during both labeler LLM prompt engineering and tool model development. As shown in Table 1, both our LLM and SLM achieve over 92% with only a minor performance gap (-1.05%).

Table 1 Performance comparison between SLM and LLM on reformulation quality evaluation.

Models Total 5 4 3 2 1 Rate(≥ 3) Diff Labeler LLM 15,355 4,120 7,143 3,034 661 214 93.11% Tool SLM 15,355 3,788 7,124 3,224 736 285 92.06% -1.05%

This trade-off between flexibility and fidelity is critical for maintaining reformulation quality while ensuring meaningful content adaptation. The empirical effects of different consistency levels are further explored in our ablation studies (Section 4.3.2), where we demonstrate how models perform when deviating from this balanced approach.

- 3.3 Prompt Engineering Strategies The ‘Limited Consistency’ framework is pivotal in balancing ‘variance’ (content variation) and ‘invariance’ (information preservation) during reformulation. To understand how prompt design impacts this balance, corpus quality, and downstream model performance, this section quantitatively analyzes two distinct prompt engineering strategies through controlled experiments.

Information Preservation Trade-off Since textual variance and information preservation are two conflicting yet equally critical objectives during reformulation, it is crucial to identify an optimal operating point for our prompts. We design two prompt variants: (1) a strict version that enforces high fidelity to source information, and (2) a relaxed version that allows substantial deviations while maintaining basic topical relevance, details could be found in Appendix E.2. Using these prompts, we collect training data from the same samples to train SLM variants, denoted as SLM-Strict and SLM-Relaxed respectively, with performance presented in Table 2.

Table 2 Performance comparison of different SLM variants on reformulation quality metrics.

Models Total 5 4 3 2 1 Rate(≤ 2) Rate(≥ 4) Rate(= 5)

SLM-Base 15,355 3,788 7,124 3,224 736 285 6.65% 71.06% 24.67% SLM-Strict 15,355 6,814 5,220 2,384 520 227 4.86% 78.37% 44.38% SLM-Relaxed 15,355 408 1,685 3,889 4,156 5,086 60.19% 13.63% 2.66%

Distributional Analysis of Prompt Engineering Strategies To further clarify how different prompt designs affect the resulting synthetic corpus distribution, we visualize the embeddings of documents generated by each SLM variant using t-SNE (Figure 2). As illustrated, the SLM-Base variant produces a balanced embedding distribution, effectively expanding beyond the original data while maintaining substantial overlap. In contrast, the SLM-Strict variant demonstrates a more constrained distribution, closely adhering to the original corpus and thus limiting diversity. On the other hand, the SLM-Relaxed variant exhibits a significant distributional shift, deviating extensively from the original space, which explains its inferior performance shown in Section 4.3.2.

[Figure 25]

- Figure 2 t-SNE visualization results. Base (left) maintains a distribution that overlaps with but extends beyond the original data. Strict (middle) clusters also extend original data but indicate limited diversity compared to the Base variant. Relaxed (right) shows significant distributional shift, explaining its poor performance.

These visualization results highlight the importance of carefully calibrated prompt engineering targets to achieve a desirable balance between corpus diversity and distributional coherence.

#### 4 Experiments

Having established our MGACorpus generation framework, we now evaluate its effectiveness through scaling experiments under data repetition scenarios in Section 4.2. Then we present a series of experiments in Section 4.3 to address the following key research questions:

- • RQ1: How effective is reformulation as a pretraining data augmentation strategy?
- • RQ2: What role does reformulation diversity play in high-repetition training?
- • RQ3: Why MGA reformulation benefits pretraining performance?

To address these research questions progressively, Section 4.2 first establishes the overall effectiveness of our MGA reformulation as a pretraining data augmentation strategy (RQ1). Subsequently, Section 4.3 delves into the role of reformulation diversity in high-repetition training (RQ2) and explores the underlying reasons for MGA’s benefits (RQ3).

- 4.1 Setup Datasets To ensure reproducibility, we build MGACorpus based on SmolLM-Corpus [6], which contains four subsources (fineweb-edu-dedup/cosmopedia/python-edu/open-web-math), expanding fineweb-edu-dedup source from 195B tokens to 770B tokens.

Models and Hyperparams The architecture of pretraining model follows llama3 [22]. Experiments across various sizes (134M/377M/1.7B/7B/13B) are running with Warmup-Stable-Decay lr scheduler [23] where 0.1% warmup steps, 75% stable and final 25% decay phase. Detailed model specifications are provided in Appendix C.

Evaluation We follow popular practice of LightEval [24] and LM-Harness [25], evaluate on a comprehensive suite of open benchmarks include ARC-Easy/Challenge [26], HellaSwag [27], Winogrande [28], MMLU [29], GSM8K [30], etc. For training dynamics, we report the average of 12 benchmarks and validation losses on held-out fineweb-edu-dedup data. And for comparison with other models, we evaluate MGACorpus aligned with Fineweb/SmolLM/Cosmopedia settings2. While model performance is influenced by multiple factors, we list some recently SOTA models as reference, all the models are evaluated in the same environment except Llama-3.2-1B3.

- 4.2 Main Experiments To directly evaluate MGA’s potential as a solution for data scarcity and repetition, we present a comprehensive analysis in two parts. First, we benchmark MGA’s performance against recent SOTA small LMs to establish a comparative baseline. Subsequently, we investigate its behavior under data-constrained scaling scenarios, specifically situations where the training budget exceeds the available unique high-quality data, a common limitation in practical applications.

- Table 3 Benchmark MGA with SOTA small LMs. Models of similar size are grouped. All results are obtained through LightEval [24]. Best results in each group are highlighted in bold, the second in underline, and in green for that MGA wins under fair comparison.

Model #Params. #Tokens ARC(C+E) Wino. Hella. MMLU MMLU-PRO CSQA OpenBookQA PIQA TriviaQA GSM8K Avg. SmolLM2-135M 135M 2T 44.12 51.07 42.03 31.27 11.06 33.82 35 68.23 1.91 1.52 32.00 SmolLM-135M 135M 600B 42.47 51.54 41.08 29.93 11.4 32.51 33.2 68.17 1.08 0.99 31.24 Baseline 134M 600B 41.71 52.41 40.69 30.03 11.37 34.32 35.4 67.85 0.02 1.29 31.51 MGA-Expansion 134M 600B 43.01 51.7 41.25 30.1 11.76 32.68 36.4 67.3 2.05 1.44 31.77

- Qwen2.5-0.5B 360M 18T 45.16 53.99 51.16 33.51 11.97 31.61 37.6 69.97 3.96 32.9 37.18 SmolLM2-360M 360M 4T 53.4 52.33 54.58 35.29 11.17 37.92 37.6 71.76 16.73 2.96 37.37 SmolLM-360M 360M 600B 49.99 52.96 51.67 33.84 11.42 34.81 37.6 71.87 2.27 1.97 34.84 Baseline 377M 600B 48.57 52.64 51.02 33.63 11.25 36.77 39 71 0.29 1.52 34.57 MGA-Expansion 377M 600B 49.39 52.64 51.34 34.09 11.35 37.1 38 72.31 7.28 1.74 35.52

- Qwen2.5-1.5B 1.3B 18T 58.36 58.64 66.39 40.23 13.85 34.4 39.6 75.95 20.51 60.8 46.87 SmolLM2-1.7B 1.7B 11T 60.42 59.59 68.73 41.4 19.61 43.65 42.6 77.53 36.68 29.04 47.93 Llama-3.2-1B 1.2B 9T 49.2 57.8 61.2 36.63 11.7 41.2 38.4 74.8 28.1 7.2 40.62 OLMo-1B-0724 1B 3.05T 44.71 56.04 64.38 32.3 11.8 33.09 38 75.24 13.82 2.43 37.18 SmolLM-1.7B 1.7B 1T 59.95 54.7 62.83 39.35 10.92 38 42.6 75.9 13.14 4.62 40.20 Baseline 1.7B 1T 59.63 57.38 65.19 39.4 12.11 42.59 45.6 76.88 4.95 7.81 41.15 MGA-Expansion 1.7B 1T 60.36 57.46 65.52 40.79 14.1 41.11 42.8 77.53 20.42 13.87 43.4

Performance training on MGACorpus We evaluate whether incorporating MGA data enhances model performance compared to a baseline trained solely on the original sources, using fixed training budgets and model sizes ranging from 134M to 1.7B. As shown in Table 3, MGA-Expansion shows consistent improvements across different model sizes, with larger performance gains as model size increases, +0.26/+0.95/+2.15 for 134M/377M/1.7B models respectively. Notably, MGA-Expansion achieved substantial gains in reasoningintensive tasks such as TriviaQA (+2.03/+6.99/+15.47) and GSM8K (+0.15/+0.22/+6.06), and shows strong performance on MMLU-Pro (all metrics in Table 3). We hypothesize that MGA’s data reformulation, by exposing the model to diverse phrasings of the same underlying information, fosters more robust generalization. This enhanced generalization, in turn, enhances the model’s reasoning capabilities, leading to the improvements observed on these specific benchmarks. For details on the baseline and MGA-Expansion data recipes, including context for comparisons with other models like SmolLM, please refer to Appendix C. Additional insights and explanations regarding metric changes are provided in Appendix D.

2https://github.com/huggingface/cosmopedia/blob/main/evaluation 3Our access request is rejected by repo authors, so we use scores reported by SmolLM.

###### Scaling Dynamics We further investigate MGA’s behavior under data-constrained scaling scenarios. Specifically, we train models ranging from 377M to 13B parameters with only learning rate warmup and stable phase, enabling direct comparison of performance metrics across repetition epochs.

[Figure 26]

repeat 50b high quality (hq) data 10 epochs collect more hq data (195b) to reduce repetition apply MGA reformulation to get 200b expansion

repeat 50b hq + 450b lq data 1.4 epochs upsample hq data 5 times with 450b lq data apply MGA reformulation to get 200b expansion

- Figure 3 Training dynamics of two common scenarios under data-constrained conditions: (1) expanding a 50B high-quality dataset to a 500B training budget (entire set repetition), (2) expanding a 500B mixed-quality dataset to a 700B training budget (subset repetition). For data recipe details please refer to Appendix C and benchmark details are provided in Appendix D.4.

Scaling Results As shown in Figure 3, MGA demonstrates favorable scaling properties with both data budget (D-scaling) and model parameters (N-scaling).

- • In the EntireSet experiments (expanding a 50B high-quality dataset to a 500B budget), simply increasing unique token count by collecting more high-quality data (195B via Full-Fineweb-Edu) shows marginal improvements (+0.2/+0.15/-0.16/+0.11) at 200/300/400/500 billion token steps (13B size). In contrast, MGA, through a 200B reformulation as expansion of the original 50B data, demonstrates consistent gains (+2.65/+3.14/+3.43/+3.46), highlighting effective D-scaling.
- • Similarly, in the Subset experiments (expanding a 500B mixed-quality dataset to a 700B budget), both upsampling the high-quality portion (5x) and MGA (via a 200B expansion) improve upon the baseline. However, their N-scaling properties with model parameters differ significantly: the performance advantage of upsampling remains relatively constant across model sizes (+0.89/+1.53/+1.23/+1.41), whereas MGA expansion exhibits superior N-scaling, its performance gains amplifying with increasing model scale (+1.46/+2.67/+3.59/+3.73).

These scaling experiments demonstrate that our method effectively serves as a data augmentation strategy to mitigate data repetition and aids model scaling (both N and D) in data-constrained scenarios, thus robustly supporting a key aspect of RQ1. While these results highlight MGA’s effectiveness, the specific role of its inherent diversity in these high-repetition settings (relevant to RQ2) will be explored further in Section 4.3.

Validation Losses Although MGACorpus demonstrates superior benchmark performance, we observe increasing validation losses compared to baseline models. While higher validation losses might seem concerning at first glance, it’s important to note that validation loss may not fully reflect model performance, as token-level perplexity is inherently biased by the frequency distribution of the validation set, and in-domain validation metrics may not necessarily correlate with out-of-domain generalization capabilities. This observation, combined with recent studies linking loss degradation to model collapse [31–33], calls for a more nuanced analysis, which we provide in Section 4.3.3.

##### 4.3 Discussions

- 4.3.1 How effective is reformulation? The main experiments in Section 4.2 have already established the overall effectiveness of MGA reformulation as a pretraining data augmentation strategy, particularly in data-constrained scenarios. To further solidify this aspect of RQ1 and contextualize MGA’s performance, we extend our analysis by comparing MGA-enhanced training data with other prominent open-source synthetic datasets: Cosmopedia [34] and variants from the Nemotron family [4]. This comparison aims to evaluate how MGA’s specific reformulation approach—targeting

diverse genre and audience presentations of source material—stands against alternative synthetic data generation techniques.

- Table 4 Comparative benchmark performance of 377M models trained on MGA reformulations versus other synthetic datasets for 300B tokens. For a fair comparison with Cosmopedia, MGA is sampled to 28B unique tokens, with both datasets then repeated 10.7 times during training. All benchmarks are 0-shot evaluations (obtained through LightEval), except for MMLU (5-shot).

Category Document Sources Synthetic Target ARC(C+E) Wino. Hella. MMLU CSQA OpenBookQA PIQA TriviaQA Avg. Cosmopedia Textbooks/Webs Story/Textbook/Wiki mix 42.15 50.43 45.06 29.17 30.38 33.2 68.77 0.23 35.57 MGA High quality webs Diverse Genre-Audience 45.65 51.22 42.31 31.42 32.19 37.2 68.39 3.79 37.28

Low quality webs Wrap-medium (Wiki style) 29.01 50.83 38.36 26.29 29.32 32 67.03 0 31.72

Extract knowledge 40.42 53.2 44.65 30.57 28.99 35 69.42 0.96 35.72 Knowledge list 42.08 52.17 42.7 30.71 32.51 35.4 70.08 0 36.21 Concise and clear passage 42.22 52.01 43.99 30.96 31.53 35 69.7 0.06 36.21 Wrap-medium (Wiki style) 42.95 52.17 43.72 31.06 31.53 36.2 70.13 0.82 36.63 Diverse QA pairs 46.96 52.57 49.03 31.36 38.82 38.8 70.84 9.21 40.724

Nemotron-CC

High quality webs

MGA High quality webs Diverse Genre-Audience 45.33 52.41 42.42 31.33 31.45 38 68.61 4.24 37.34

The comparative data in Table 4 highlights MGA’s strength. MGA (average 37.28) surpasses Cosmopedia (35.57). Against various Nemotron strategies, MGA (average 37.34) also outperforms most alternatives like ‘extract knowledge’ (35.72) and ‘wrap-medium (Wiki style)’ (36.63). While Nemotron’s ‘diverse QA pairs’ achieves the highest average (40.72), its format offers an advantage in the 0-shot evaluation context. Despite this, MGA’s broader reformulation approach demonstrates robust utility, outperforming five of the six Nemotron strategies and showing particular strength on benchmarks like TriviaQA, underscoring its value as a general-purpose data augmentation technique.

These comprehensive comparisons reinforce the answer to RQ1: MGA’s diverse genre-audience reformulation is a highly effective pretraining data augmentation strategy. It not only improves upon baselines using original data (as shown in Section 4.2) but also stands as a strong, and often superior, alternative to other synthetic data generation methods. The particular strength of MGA appears to lie in its ability to generate varied and contextually rich reformulations, which likely contributes to the model’s enhanced generalization and reasoning capabilities.

- 4.3.2 Does reformulation diversity help to mitigate repetition issue? To address RQ2, this section examines how different design choices in prompt engineering influence the effectiveness of the MGA framework, particularly under high-repetition conditions. By comparing SLM variants (introduced in Section 3.3) using different consistency requirements, we identify optimal strategies for balancing information preservation with content diversity.

We sample an additional 20B tokens from real data and generate three synthetic datasets: 80B tokens using SLM-Base, 80B tokens using SLM-Strict, and 40B tokens using SLM-Relaxed. Similar to experimental setup in early sections, we set a high-repetition baseline on a smaller data scale (replicating the original 20B tokens 10 times) to more clearly demonstrate the potential impact of SLM-Strict compared to SLM-Base.

Average.

Knowledge

Reasoning

Math

Validation Losses

3.0

0.150

slm-base slm-strict slm-relaxed

slm-base slm-strict slm-relaxed

|slm-b slm-s<br><br>|ase trict| | |
|---|---|---|---|
|slm-re<br><br>origin|laxed al| | |
| | | | |
| | | | |
| | | | |
| | | | |

|slm-b slm-s<br><br>|ase trict| | |
|---|---|---|---|
|slm-re<br><br>origin|laxed al| | |
| | | | |
| | | | |
| | | | |

|slm-b|ase| | |
|---|---|---|---|
|slm-s<br><br>slm-re|trict laxed| | |
|origin|al| | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.35

0.40

0.250

0.125

original

original

2.8

original_uniq

0.225

0.30

0.100

0.35

scores

losses

0.200

0.075

2.6

0.25

0.050

0.175

0.30

0.025

2.4

0.150

0.20

0.000

0.25

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200 250

# of tokens (B)

# of tokens (B)

# of tokens (B)

# of tokens (B)

# of tokens (B)

- Figure 4 Benchmark results and validation losses. The sensitivity to data repetition varies across capability domains, with knowledge dimension showing greater resilience.

4The predominantly 0-shot evaluation particularly benefits datasets like Nemotron ‘diverse QA pairs’ whose format directly aligns with many evaluation tasks.

As shown in Figure 4, our experiments reveal distinct patterns across training configurations. Both SLM-Base and SLM-Strict demonstrate performance improvements, while the SLM-Relaxed configuration leads to significant collapse. More details could be found in Appendix D.3.

Despite the apparent effectiveness of strict information preservation, can it fundamentally address the challenges of data repetition? Our examination of validation loss trajectories reveals a critical distinction: SLM-Base maintains healthy optimization characteristics throughout training, whereas SLM-Strict exhibits degraded scaling behavior at higher iteration steps, reminiscent of the limitations observed with data repetition. Therefore, this investigation into prompt engineering variants concludes that a balanced ‘Limited Consistency’ approach (SLM-Base) yields best reformulation quality and subsequent model performance answering to RQ2.

- 4.3.3 How pretraining data reformulation benefits? Having explored the impact of reformulation diversity in addressing data repetition (RQ2), we now turn to RQ3: Why does MGA reformulation benefit pretraining performance? We investigate the underlying mechanisms by analyzing learning characteristics and validating against potential issues like model collapse [31–33].

Multi-perspective Validation Analysis Our analyses across different validation sets reveal varying patterns in model behavior (Figure 5). As expected, MGA groups’ substitution of fineweb-edu data results in adverse effects on corresponding loss, with similar deterioration observed in open-web-math. Interestingly, the synthetic dataset cosmopedia demonstrates improved loss metrics. A notable contrast emerges in python-edu: while MGA exhibit negative impact at the 134M and 377M parameter, this trend reverses at 1.7B, suggesting scale-dependent effects on model behavior.

cosmopeida-v2

fineweb-edu-dedup

open-web-math

python-edu

2.6

Baseline

Baseline

Baseline

| | | | | |Bas|eline| |
|---|---|---|---|---|---|---|---|
| | | | | |134<br><br>MGA|m<br><br>-Expansio|n|
| | | | | |377 1b7<br><br>|m| |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.8

2.2

MGA-Expansion

MGA-Expansion

MGA-Expansion

3.0

2.5

134m 377m 1b7

134m 377m 1b7

134m 377m 1b7

1.7

2.4

2.0

1.6

2.8

2.3

1.8

1.5

2.2

losses

1.4

2.6

2.1

1.6

1.3

2.0

1.4

2.4

1.2

1.9

1.1

1.8

1.2

2.2

1.0

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

# of tokens (B)

# of tokens (B)

# of tokens (B)

# of tokens (B)

Figure 5 validation losses of experiments in Section 4.2.

Fine-grained Pattern Analysis To better understand whether increased validation loss truly indicates model collapse, we conduct a fine-grained analysis of loss patterns. Specifically, we compare token-level losses of 800B checkpoint between models trained on real data and synthetic data (Baseline and MGA-Expansion in Section 4.2, respectively). The document samples are from both Fineweb-Edu and MGACorpus. As illustrated in subfigures 1 and 3 of Figure 6, each point represents a sample’s average token loss, consistent with the overall loss discrepancy shown in Figure 5.

[Figure 27]

- Figure 6 Losses pattern analysis. Subfigures 1 and 3 shows comparison between models trained on different data settings, with lossreal on y-axis and losssynt on x-axis. Subfigures 2 and 4 track the position where lossisynt − lossireal (lossidiff) first becomes significantly higher than the sequence’s average difference (detailed definition in Appendix D.3).

The distribution of first anomaly positions (subfigures 2 and 4) reveals a crucial insight: when processing real data, models trained on synthetic data show performance degradation (measured by lossdiff) that predominantly manifests in later sequence positions, which intensifies as lossdiff increases. However, this positional bias disappears when evaluating on synthetic data.

The systematic pattern suggests that rather than experiencing model collapse, the synthetic-trained model may have developed a different learning strategy (cases in Appendix D.3). While it shows higher validation losses on certain real-world datasets, its strong performance in our main experiments indicates a potential trade-off:

the model may prioritize learning generalizable patterns from context over memorizing specific sequence dependencies. This shift in learning process could explain both the improved performance on benchmarks and increased losses on validation sets that potentially require more memorization-based processing.

Addressing RQ3, these findings indicate that the performance characteristics associated with MGA data likely stem from altered learning strategies, potentially prioritizing generalizability, rather than representing model collapse phenomenon.

#### 5 Conclusion

In this work, we introduced MGA, an efficient framework that leverages genre-audience reformulation to systematically expand existing corpora with diverse, synthetically generated variations. Our core finding highlights MGA’s effectiveness as a data augmentation strategy specifically targeting the repetition challenge: in data-constrained scaling experiments, MGA significantly outperformed naive data repetition and simple upsampling, enabling more effective model training beyond unique data limits. Furthermore, the quality of the MGACorpus was confirmed by consistent performance improvements when incorporated into standard training mixtures across various model sizes. While evaluating synthetically expanded data requires careful consideration, MGA’s success stems from its ability to create relevant diversity, directly counteracting the negative impacts of repeating limited datasets. Therefore, MGA offers a practical and scalable pathway to alleviate data repetition bottlenecks, facilitating continued progress in large language model development by making more effective use of available data resources.

###### Acknowledgements

We are grateful to Chao He, Zhixin Yao, Yue Chen and Runyu Shi for their help with prompt templates and case studies, and to Seed-Foundation team for providing the stable training/inference platform, which enabled us to build the synthetic pipeline and corpus within a reasonable timeframe. The icons shown in Figure 1 are designed by Freepik.

#### References

- [1] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

- [2] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. An empirical analysis of compute-optimal large language model training. Advances in Neural Information Processing Systems, 35:30016–30030, 2022.

- [3] Pablo Villalobos, Jaime Sevilla, Lennart Heim, Tamay Besiroglu, Marius Hobbhahn, and Anson Ho. Will we run out of data? an analysis of the limits of scaling datasets in machine learning. arXiv preprint arXiv:2211.04325, 1, 2022.

- [4] Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Nemotron-cc: Transforming common crawl into a refined long-horizon pretraining dataset. arXiv preprint arXiv:2412.02595, 2024.

- [5] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

- [6] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. Smollm-corpus,

2024. URL https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus.

- [7] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

- [8] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.

- [9] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data only. Advances in Neural Information Processing Systems, 36:79155–79172, 2023.

- [10] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024.

- [11] Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale,

2024. URL https://arxiv.org/abs/2406.17557.

- [12] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2024.
- [13] Ge Zhang, Xinrun Du, Zhimiao Yu, Zili Wang, Zekun Wang, Shuyue Guo, Tianyu Zheng, Kang Zhu, Jerry Liu, Shawn Yue, Binbin Liu, Zhongyuan Peng, Yifan Yao, Jack Yang, Ziming Li, Bingni Zhang, Minghao Liu, Tianyu Liu, Yang Gao, Wenhu Chen, Xiaohuan Zhou, Qian Liu, Taifeng Wang, and Wenhao Huang. Finefineweb: A comprehensive study on fine-grained domain web corpus, December 2024.
- [14] Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, et al. Scaling laws and interpretability of learning from repeated data. arXiv preprint arXiv:2205.10487, 2022.

- [15] Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36:50358–50376, 2023.

- [16] Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085, 2022.

- [17] Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, and Yang You. To repeat or not to repeat: Insights from scaling llm under token-crisis. Advances in Neural Information Processing Systems, 36, 2024.

- [18] Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, and Navdeep Jaitly. Rephrasing the web: A recipe for compute and data-efficient language modeling. arXiv preprint arXiv:2401.16380, 2024.

- [19] Michael Pieler, Marco Bellagente, Hannah Teufel, Duy Phung, Nathan Cooper, Jonathan Tow, Paulo Rocha, Reshinth Adithyan, Zaid Alyafeai, Nikhil Pinnaparaju, et al. Rephrasing natural text data with different languages and quality levels for large language model pre-training. arXiv preprint arXiv:2410.20796, 2024.

- [20] Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094, 2024.

- [21] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pages 38087–38099. PMLR, 2023.

- [22] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [23] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

- [24] Clémentine Fourrier, Nathan Habib, Thomas Wolf, and Lewis Tunstall. Lighteval: A lightweight framework for llm evaluation, 2023. URL https://github.com/huggingface/lighteval.
- [25] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/10256836.
- [26] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457,

- 2018.

[27] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics,

- 2019.

- [28] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

- [29] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

- [30] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [31] Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, and Julia Kempe. A tale of tails: Model collapse as a change of scaling laws. arXiv preprint arXiv:2402.07043, 2024.

- [32] Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, and Julia Kempe. Strong model collapse. arXiv preprint arXiv:2410.04840, 2024.

- [33] Xuekai Zhu, Daixuan Cheng, Hengli Li, Kaiyan Zhang, Ermo Hua, Xingtai Lv, Ning Ding, Zhouhan Lin, Zilong Zheng, and Bowen Zhou. How to synthesize text data without model collapse? arXiv preprint arXiv:2412.14689, 2024.

- [34] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. Smollm - blazingly fast and remarkably powerful, 2024.

## Appendix

#### A Limitations and Opportunities

While our experimental results demonstrate the effectiveness of MGA in both quality validation and scaling scenarios, several important aspects warrant further investigation. We identify three key areas:

- • The tool model (SLM) employed in this work relies on an early version with relatively moderate capabilities. While Pieler et al. [19] suggests that model size may not be the determining factor for rephrasing tasks, the influence within MGA framework remains unexplored. Understanding the relationship between SLM capacity and corpus quality is crucial for optimizing the effectiveness-efficiency trade-off.
- • Our current experiments demonstrate effectiveness up to 13B parameters and 1,000B tokens of training budget. Extending this approach to long-horizon training and larger-scale models requires additional validations, particularly for next-generation models which require hundreds of trillions of training tokens.
- • Regarding data repetition strategies, we present preliminary explorations under computational resource constraints. The underlying patterns and their sensitivity to various factors, such as repetition ratio, data distribution, and data quality, require systematic investigation. Future research should examine how these factors collectively determine optimal data strategies across different training scenarios.

Broader Impact This paper explores the use of LLMs as a data expansion method for pretraining large language models. We introduce a lightweight and scalable framework (MGA) to mitigate data repetition issues, which holds potential for positive societal impact, particularly in synthetic data generation for training language models. Nonetheless, the use of synthetic data generated by LLMs is not without risks; for instance, LLM hallucinations, even after filtering, could introduce novel errors or biases into models trained on such data, a factor that warrants careful consideration in future research and deployment.

#### B Tool Model Implementation

###### Corpus Sample

| |
|---|

[Figure 28]

[Figure 29]

- Figure 7 Implementation details. From a high-quality corpus, we sample a subset to serve as input for the LLM labeler and judger. Through iterative filtering, we train and quantize SLM tool models for each stage to improve inference efficiency, which are used to generate the reformulated corpus.

High Quality Corpus To ensure reproducibility, we conduct our reformulated corpus based on SmolLMCorpus5 [6], expanding fineweb-edu-dedup source from 195B tokens to 770B tokens. Then we setup additionally experiments on FineWeb and FineWeb-Edu [11], which constitute a solid foundation for research on data

5https://github.com/huggingface/smollm/tree/main/text/pretraining

scaling approaches. Prior to these experiments, we have validated the approach on our in-house datasets. The results demonstrate strong performance across both datasets, suggesting broad applicability of our method.

Tool Models Training Initialized from a pretrained SLM (a 3.3B MoE model), we collect 50,000 training samples through iterative filtering and training, where 15,000 of raw text to genre-audience pairs, 35,000 of raw text to reformulated output. Each model’s validation responses are scored by capable LLM judger, that ensures the SLMs achieve comparable synthesis quality to the LLM labeler as shown in Table 1. The sequence length is 8192 with maximum prompt/response length 4,096 tokens, each model is trained 3 epochs on the samples with a cosine lr scheduler.

Cleaning Stage Similar to previous synthesis work [4, 18], we involve a final cleaning stage to filter out the high frequency patterns, for example, ‘Notes: ...’, ‘Please note that ...’, ‘The above is as required ...’, ‘The following is...’, etc. And remove documents with an extremely low keyword coverage to raw documents.

Resource Analysis To generate 770B synthetic tokens, it takes 256×64 and 1024×130 NVIDIA H100 GPU hours to process two stages, or 4× more hours when using Huawei Ascend910B2. In our practice, we use Huawei Ascend910B2 synthesis most tokens of MGACorpus, which significantly reduce the cost of synthesis.

#### C Pretraining Details

Data Recipe The training token budgets are 600B/600B/1000B for size of 134M/377M/1.7B models, which are aligned with SmolLM1 series [6]. Our baseline is trained on SmolLM-Corpus dataset, in contrast to SmolLM’s recipe, we use unique token number from each source as the mixing ratio shown in Table 5. This ensures that different sources have consistent repetition epochs during training. For a fair comparison, the mixing ratios of other data sources are kept constant across experiments. We specifically adjusted the proportions of fineweb-edu-dedup and MGACorpus (which is derived from fineweb-edu-dedup) to isolate the impact of the MGA reformulation.

- Table 5 MGACorpus experiments data recipe: source weight (%) and #unique_tokens × #epochs (using 1000B budget as example).

experiments - fineweb-edu-dedup cosmopedia-v2 python-edu open-web-math MGACorpus Baseline weight 80.89 11.65 1.66 5.80 -

#unique_tokens × #epochs 195 × 4.15 28 × 4.15 4 × 4.15 14 × 4.15 MGA-Expansion weight 16.29 11.65 1.66 5.80 64.59 #unique_tokens × #epochs 195 × 0.84 28 × 4.15 4 × 4.15 14 × 4.15 770 × 0.84

The experiment design for different strategies is presented in Table 6, which involves three datasets: (1) a 50B-token random sample from fineweb-edu-dedup, (2) a corresponding filtered subset from MGACorpus, and (3) a 450B-token deduplicated corpus obtained from Fineweb [11].

Table 6 Scaling experiments data recipe, values represent #unique_tokens × #epochs.

Training fineweb-edu MGA fineweb

Repetition Experiments

Design Rationale Budget dedup corpus random

Baseline 500B 50 × 10 - Full-Fineweb-Edu 500B 195 × 2.56 - - What if we could collect more unique data. MGA Expansion 500B 50 × 2 200 × 2 - Add MGA to reduce the repetition num.

EntireSet

Baseline 700B 50 × 1.4 - 450 × 1.4 Upsample-EDU 700B 50 × 5 - 450 × 1 Upsample to get 200B more budget. MGA Expansion 700B 50 × 1 200 × 1 450 × 1 Add MGA to achieve the same target.

Subset

Training Hyperparameters We sample 100 million tokens from SmolLM-Corpus as the validation dataset. The hyperparams are presented in Table 7, tokenzier used for training and computing token counts is same as SmolLM16 with a vocab size of 49,152.

6https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer

Table 7 Hyperparams of different model size.

model batch learning hidden ffn num num shared seq tie total size size rate size inner heads layers q_head len emb params

134M 128 3e-3 1,204 4,096 8 8 1 8,192 false 134M 377M 320 1.5e-3 1,536 6,144 12 10 1 8,192 false 377M

1.7B 512 5e-4 2,560 10,240 20 16 1 8,192 false 1.68B

7B 1,024 4e-4 4,096 8,192 32 32 4 8,192 false 6.98B 13B 1,024 4e-4 4,096 12,288 32 48 4 8,192 false 12.9B

Evaluation The LightEval results provided in Section 4.2 follow SmolLM setting, that with GSM8K/MMLU 5shot and all the others 0-shot. The benchmarks presented in Figure 12 and Figure 13 follow few-shot evaluation settings, specifically ARC(8-shots), TriviaQA(5-shots), Winogrande(5-shots) and similar configurations for other tasks.

#### D Further Analysis of Experiments

- D.1 Benchmark Improvement In our experimental observations (Table 8), notable performance improvements are demonstrated in both TriviaQA and GSM8k benchmarks, warranting a detailed examination of these score variations.

Table 8 Benchmark results. A copy of SmolLM1/SmolLM1-Ours/MGA-Expansion in Table 3.

Model #Params. #Tokens ARC(C+E) Wino. Hella. MMLU MMLU-PRO CSQA OpenBookQA PIQA TriviaQA GSM8K Avg. SmolLM-1.7B 1.7B 1T 59.95 54.7 62.83 39.35 10.92 38 42.6 75.9 13.14 4.62 40.20 Baseline 1.7B 1T 59.63 57.38 65.19 39.4 12.11 42.59 45.6 76.88 4.95 7.81 41.15 MGA-Expansion 1.7B 1T 60.36 57.46 65.52 40.79 14.1 41.11 42.8 77.53 20.42 13.87 43.4

The enhanced TriviaQA performance exhibited by SmolLM1-1.7B relative to our baseline can be attributed to the larger proportion of Cosmopedia in its training configuration. Both MGACorpus and Cosmopedia employ synthetic methodologies, which contribute to improved learning efficiency. The observed gains in GSM8K performance can be traced to the target genres, including teaching schemas and problem-solving exemplars, embedded within the Reformulation component. This early exposure to structured problem-solving approaches facilitates more effective performance on analogous mathematical reasoning tasks.

- D.2 What if use MGACorpus alone?

Our primary goal with MGA is efficient dataset expansion, typically achieved by mixing the generated corpus with existing real data, aligning with current best practices for leveraging synthetic data. However, to better characterize the properties of the MGACorpus itself and understand the impact of training exclusively on reformulated content, we also investigate an experimental setting where MGACorpus completely replaces its source data (fineweb-edu-dedup).

Table 9 MGACorpus experiments data source weight (%).

experiments fineweb-edu-dedup cosmopedia-v2 python-edu open-web-math MGA-corpus Baseline 80.89 11.65 1.66 5.80 MGA-Only - 11.65 1.66 5.80 80.89 MGA-Expansion 16.29 11.65 1.66 5.80 64.59

As shown in Table 10, the absence of real data leads to performance degradation across most tasks (average -0.95), particularly in two tasks, Hellaswag(-1.23/-1.69/-2.85) and CommonsenseQA(-3.11/-4.83/-4.50). This decline can be attributed to our design choice, which focuses on diversity and overall quality rather than requiring the preservation of all information from each raw documents.

###### Table 10 Comparison between MGA-Expansion and MGA-Only

Model #Params. #Tokens ARC(C+E) Wino. Hella. MMLU MMLU-PRO CSQA OpenBookQA PIQA TriviaQA GSM8K Avg. MGA-Expansion 134M 600B 43.01 51.7 41.25 30.1 11.76 32.68 36.4 67.3 2.05 1.44 31.77 MGA-Only 134M 600B 41.98 51.38 40.02 29.87 11.5 29.57 33 68.01 2.26 1.06 30.87

↓-1.03 ↓-0.32 ↓-1.23 ↓-0.23 ↓-0.26 ↓-3.11 ↓-3.40 ↑0.71 ↑0.21 ↓-0.38 ↓-0.90

MGA-Mix 377M 600B 49.39 52.64 51.34 34.09 11.35 37.1 38 72.31 7.28 1.74 35.52 MGA-Only 377M 600B 47.95 53.35 49.65 33.31 11.38 32.27 38 70.95 6.83 1.59 34.53

↓-1.44 ↑0.71 ↓-1.69 ↓-0.78 ↑0.03 ↓-4.83 - ↓-1.36 ↓-0.45 ↓-0.15 ↓-0.99

MGA-Mix 1.7B 1T 60.36 57.46 65.52 40.79 14.1 41.11 42.8 77.53 20.42 13.87 43.40 MGA-Only 1.7B 1T 59.02 57.06 62.67 40.34 13.51 36.61 45.2 76.71 19.78 13.57 42.45

↓-1.34 ↓-0.40 ↓-2.85 ↓-0.45 ↓-0.59 ↓-4.50 ↑2.40 ↓-0.82 ↓-0.64 ↓-0.30 ↓-0.95

- D.3 Ablation Details MGA-Only Setting of PE Ablation Upon relaxing the information preservation requirements for PE objectives in the MGA-Only setting, we observe a complete collapse in knowledge-based dimensions while maintaining modest improvements in reasoning and mathematical capabilities. This divergence suggests that different cognitive capabilities have distinct requirements for the richness and nature of training data content.

Average.

Knowledge

Reasoning

Math

Validation Losses

slm-base slm-strict slm-relaxed

slm-base slm-strict slm-relaxed

|slm-b slm-s<br><br>|ase trict| | |
|---|---|---|---|
|slm-re<br><br>origin|laxed al| | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

|slm-b slm-s<br><br>|ase trict| | |
|---|---|---|---|
|slm-re<br><br>origin|laxed al| | |
| | | | |
| | | | |
| | | | |

|slm-b slm-s<br><br>|ase trict| | |
|---|---|---|---|
|slm-re<br><br>origin|laxed al| | |
| | | | |
| | | | |
| | | | |

0.35

0.15

0.250

0.40

3.2

original

original

0.225

original_uniq

3.0

0.30

0.35

0.10

scores

0.200

losses

2.8

0.25

0.30

0.175

2.6

0.05

0.150

0.25

2.4

0.20

0.125

0.00

2.2

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200 250

# of tokens (B)

# of tokens (B)

# of tokens (B)

# of tokens (B)

# of tokens (B)

Figure 8 Corresponding benchmark results described in Section 4.3.2.

Further Discussion of Section 4.3.3 For our analysis method in Figure 6, we define the token loss difference as lossidiff = lossisynt − lossireal, where i is the token index, synt/real is dataset used for model training. Note that we consistently use synthetic minus real, where a positive value indicates poorer prediction performance by the synthetic model on a given sample.

Since next token prediction is computed based on preceding context, we define the first anomaly position to identify where a model’s prediction for tokens within the window begins to significantly deteriorate. The definition is as follows:

first_anomaly_position = min{p |

p+w−1

1 w

lossidiff > |µ| + kσ},

i=p

where w = max(0.05 × seq_length,1), µ = mean(lossidiff), σ = std(lossidiff). Here, we employ the absolute value of the windowed average loss to identify significant performance degradation in either model. This

approach enables the detection of notable prediction quality drops regardless of which model (synthetic or real) experiences the deterioration.

Finally, we define the normalized position, enabling fair comparisons across various sequence lengths:

normalized_position =

first_anomaly_position

seq_length × 100% if anomaly found −1 otherwise

Below are example cases from English and Chinese documents. Figure 9 presents the token loss difference on each position. Example 2 and Example 3 show similar anomaly pattern, we can get the reason in Figure 10, that they are from the same website source contain identical boilerplate text about region selection and website localization at the end of their content.

[Figure 30]

###### Figure 9 Random examples sampling from where mean(lossidiff) > 0.5, the synthetic-trained model fail to predict the tokens in later sequence positions.

This suggests potential noise in the data preprocessing pipeline, specifically in handling website navigation elements and localization prompts that should have been removed during content extraction.

While these examples demonstrate clear patterns of model behavior differences in handling noisy web data, we acknowledge that this analysis is limited to selected cases with apparent preprocessing artifacts. A more comprehensive evaluation across diverse data sources and quality levels would be necessary to fully understand the impact of synthetic training data on model performance.

[Figure 31]

###### Figure 10 Corresponding cases sampled from Fineweb-Edu, which align with the loss patterns shown in Figure 9, with higher loss by synthetic-trained model highlighted in red .

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Figure 11 Chinese corpus samples with higher loss by synthetic-trained model in red .

##### D.4 Scaling Experiments Details

arc_challenge

arc_easy

codah_commonsense

drop

0.65

13b

Baseline

0.85

0.35

0.60

7b

Upsample fineweb-edu

0.60

1b7

MGA expansion

377m

0.30

0.55

0.80

0.55

0.25

0.50

scores

0.50

0.75

0.45

0.20

0.45

0.70

0.40

0.15

0.35

0.40

0.65

0.10

0.30

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

gsm8k

hellaswag

mmlu

openbookqa

0.60

0.25

0.70

0.450

0.65

0.20

0.425

0.55

0.60

0.400

scores

0.15

0.50

0.55

0.375

0.10

0.50

0.350

0.45

0.05

0.45

0.325

0.00

0.40

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

piqa

race

triviaqa_wiki

winogrande

0.575

0.675

0.78

0.5

0.550

0.650

0.525

0.76

0.625

0.4

0.500

scores

0.74

0.600

0.475

0.72

0.3

0.575

0.450

0.70

0.550

0.425

0.2

0.68

0.525

0.400

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

100 200 300 400 500

# of tokens (B)

# of tokens (B)

# of tokens (B)

# of tokens (B)

###### Figure 12 Detail evaluation results of EntireSet described in Table 6. MGACorpus group demonstrats advantages over other groups across most evaluation sets, consistently across models of sizes.

arc_challenge

arc_easy

codah_commonsense

drop

0.65

13b

Baseline

7b

Upsample fineweb-edu

0.35

0.80

0.60

0.60

1b7

MGA expansion

377m

0.75

0.55

0.30

0.55

0.50

0.70

scores

0.25

0.50

0.45

0.65

0.20

0.45

0.40

0.60

0.15

0.35

0.55

0.40

0.10

0.30

200 400 600

200 400 600

200 400 600

200 400 600

gsm8k

hellaswag

mmlu

openbookqa

0.60

0.75

0.425

- 0.175

0.70

0.55

- 0.150

0.400

0.65

- 0.125

0.375

0.50

0.60

scores

- 0.100

0.350

0.55

- 0.075

0.45

0.50

0.325

- 0.050

0.40

0.45

- 0.025

0.300

0.40

- 0.000

200 400 600

200 400 600

200 400 600

200 400 600

piqa

race

triviaqa_wiki

winogrande

0.6

0.80

0.70

0.55

0.5

0.78

0.65

0.76

0.50

0.4

scores

0.74

0.60

0.3

0.45

0.72

0.2

0.70

0.55

0.40

0.68

0.1

200 400 600

200 400 600

200 400 600

200 400 600

# of tokens (B)

# of tokens (B)

# of tokens (B)

# of tokens (B)

###### Figure 13 Detail evaluation results of Subset described in Table 6. As the model size increases, the performance gap between the upsampling group and MGACorpus gradually widens in ARC, DROP, GSM8K, RACE, but with some variations observed in TriviaQA and WinoGrande.

#### E Prompts and Cases

- E.1 Example outputs of SLM variants

|Raw Document| | |
|---|---|---|
|In chemistry teaching, why is it said that skillfully capturing the optimal timing of ``movement-stillness'' transitions is key to success? Because under the new curriculum philosophy, guiding students to adopt a learning method that combines ``movement and stillness'' transforms their thinking patterns and helps them learn through activities. Being able to skillfully capture the optimal timing for transitions between ``movement and stillness'' is key to success - seizing these perfect moments for transition leads to greater efficiency with less effort.| | |
|Genre-Audience Pairs|Tool Model|Reformulated Document|
|- Academic teaching paper, using rigorous academic style to deeply explore the importance, methods and practical case analysis of capturing the optimal timing of 'movement-stillness' transitions in chemistry teaching. Through references to relevant educational theories and actual teaching experiences, provide meaningful teaching insights for chemistry teachers.<br>- Middle school chemistry teachers, aged 25-45, predominantly female, with professional chemistry education background. They are patient and steady in character, have been teaching middle school chemistry for many years, and wish to improve teaching quality and help students better understand chemistry.<br>|SLM-Base|**The Importance and Practice of Capturing Optimal Timing of "Movement-Stillness" Transitions in Middle School Chemistry Teaching**<br><br>**Abstract**: This paper aims to deeply explore the importance, methods, and practical case analysis of capturing optimal timing of "movement-stillness" transitions in middle school chemistry teaching from theoretical and practical perspectives. Through references to relevant educational theories and actual teaching experiences, it provides meaningful teaching insights for middle school chemistry teachers to help improve teaching quality and promote better student understanding of chemistry knowledge.<br><br>**Keywords**: Middle school chemistry teaching; "Movement-stillness" transitions; Optimal timing; Teaching methods<br><br>**I. Introduction** With the continuous advancement of new curriculum concepts, middle school chemistry teaching is actively exploring more effective teaching methods to cultivate students' chemistry literacy and comprehensive abilities. Among these, having students adopt a learning method that combines "movement and stillness" has gradually gained attention, and skillfully capturing the optimal timing of "movement-stillness" transitions has become one of the keys to successful chemistry teaching.<br>**II. The Importance of Capturing Optimal Timing of "Movement-Stillness" Transitions**<br><br><br>(1) Alignment with Educational Theory According to constructivist learning theory, student learning is a process of actively constructing knowledge through interaction with the external environment based on existing knowledge and experience. In middle school chemistry teaching, the learning method of "movement-stillness" transitions helps students concretize abstract chemical concepts through dynamic practical operations, leading to better understanding and mastery of chemistry knowledge.|
| |SLM-Strict|In the field of chemistry teaching, why is it said that skillfully capturing the optimal timing of "movement-stillness" transitions is key to success? Under the new curriculum philosophy, students are encouraged to use a learning method that combines "movement and stillness" to transform their thinking patterns and learn through activities. Being skillful at capturing the optimal timing of "movement-stillness" transitions is truly the key point of success - once this optimal timing is grasped, it can achieve twice the results with half the effort. From a theoretical perspective, in middle school chemistry teaching, the concept of "movement-stillness" transitions aligns with students' cognitive patterns and helps stimulate their learning interest and initiative. Students can directly experience chemical phenomena through dynamic practical operations, deepening their understanding of knowledge; while in the static stages of thinking and summarizing, they can elevate perceptual knowledge to rational cognition and construct complete knowledge systems. In practice, teachers can analyze through specific cases. For example, when explaining the laboratory preparation of oxygen, teachers first guide students to observe the experimental apparatus and understand the function of each part - this is the "stillness" stage. Then, letting students personally conduct the experiment and observe phenomena such as bubble production and wood splint reignition is the "movement" stage. After students complete the experiment, organizing them to discuss problems encountered during the experiment and their solutions, summarizing and concluding, returns again to "stillness" reflection. Thus, through "movement-stillness" transitions, students not only master the knowledge and skills of oxygen preparation but also develop abilities in observation, analysis, and problem-solving. In conclusion, middle school chemistry teachers should fully recognize the importance of skillfully capturing optimal timing of "movement-stillness" transitions in teaching, and through reasonable design of teaching segments, integrate the learning method combining "movement and stillness" throughout the entire teaching process to improve teaching quality and help students better understand chemistry knowledge.|
| |SLM-Relaxed|In the field of education, for teachers of fundamental subjects, finding the key opportunities to guide students in effective learning is central to improving teaching effectiveness. Taking middle school chemistry teaching as an example, new educational concepts advocate students using flexible and diverse learning modes to transform their thinking pathways, enabling them to explore knowledge more independently in the learning process. Finding the perfect opportunity to guide students in achieving this thinking transformation during the learning process is undoubtedly key to successful teaching. Once this opportunity is successfully grasped, teaching effectiveness will be significantly improved, and students will be able to understand and master the learned knowledge more deeply.|
|- Popular science story, using a lively and interesting narrative to tell story of how middle school students realizes the importance of capturing the optimal timing of 'movement-stillness' transitions in their chemistry learning process. Through specific interesting experiments, help students understand chemistry knowledge in a relaxed and enjoyable atmosphere, sparking interest in chemistry learning.<br>- Middle school students who find chemistry difficult, aged 12-15, both male and female, with below-average academic performance. They are generally lively and active in character, lack interest in chemistry but must study it, and hope to find ways to improve their chemistry grades.<br>|SLM-Base|Once there was a middle school student named Xiao Ming, who was very lively and active, but initially had little interest in chemistry as a subject, and his grades were always below average. One day in chemistry class, the teacher taught a particularly interesting lesson about a magical chemical change. The teacher said that during this change process, it was like having a mysterious little sprite at play - sometimes it was very quiet, and sometimes it became very active. This caught Xiao Ming's attention completely. He wondered when this little sprite would be quiet, and when it would become active. Later, the teacher led everyone in doing an experiment. In the experiment, there were two small bottles - one bottle contained a transparent liquid, and the other bottle contained a magical powder. The teacher slowly poured the powder into the bottle containing the liquid. At first, nothing happened, and the liquid in the bottle remained perfectly still, as if it were sleeping. Xiao Ming and his classmates watched curiously, wondering when the little sprite would wake up. Just when everyone was getting a bit impatient, suddenly, the liquid in the bottle started bubbling - it was as if the little sprite had suddenly woken up and started playing excitedly. This change came so suddenly that all the students were stunned, their eyes wide open. The teacher smiled and told everyone that this was the "movement-stillness" transition. During this process, everyone had witnessed the sprite's change from quiet to active. The teacher also said that when studying chemistry, it's just like doing this experiment. You need to be good at capturing the optimal timing of "movementstillness" transitions. If you can grasp it at that moment, you'll be able to better understand chemistry knowledge, and learning will become twice as effective with half the effort.|
| |SLM-Strict|Hey, classmates! Do you know that in the wonderful world of chemistry learning, there's something really important - that's being good at capturing the optimal timing of "movement-stillness" transitions, this is the key to success! Just like under the new curriculum concepts, teachers guide us to use a learning method that combines "movement and stillness," which can transform our way of thinking and help us learn through activities. In my chemistry learning process, I've deeply experienced this point. Once, when we were doing an interesting experiment, the teacher told us to be good at catching the optimal timing of "movement-stillness" transitions. At first, I didn't quite understand. But later, I discovered that when certain phenomena in the experiment start to change, that's the optimal timing for "movement-stillness" transitions! As long as we catch this timing, we can achieve twice the results with half the effort in our experiments! Hey classmates, don't you think it's amazing too? Let's discover more fun together in our chemistry learning journey!|
| |SLM-Relaxed|In a vibrant middle school classroom, there was a group of lively and active students, aged between 12-15, both boys and girls, with below-average academic performance. Among them was a student named Xiao Le, who had no interest in chemistry but had to study it, and was always looking for ways to improve his grades. One day, in the school laboratory, Xiao Le watched the teacher perform a magical experiment. The teacher first put some ordinary-looking powder into a transparent bottle, then added a liquid. Immediately, it was like a lively party started in the bottle, with lots of colorful bubbles appearing - this completely captivated Xiao Le and his classmates. Xiao Le thought to himself that this was just like how he played in the playground - sometimes resting quietly, sometimes running around wildly - isn't this a combination of "movement and stillness"? Moreover, he discovered that during this process, the moment when the powder and liquid made contact was the most magical moment, like finding a hidden treasure. From then on, every time Xiao Le did chemistry experiments, he would pay special attention to this optimal timing of "movement-stillness" transitions. He found that when he caught this timing, he could understand chemistry knowledge better. His chemistry grades gradually improved, and his interest in studying chemistry grew stronger and stronger.|
|- Parenting education guide, using easyto-understand language style combined with real-life scenarios to explain the concept of capturing optimal timing of 'movement-stillness' transitions in chemistry teaching. Provide simple parent-child interaction methods to help parents guide their children in understanding and applying this learning method in daily life, while improving parent-child relationships.<br>- Parents, aged 30-50, both male and female, with diverse occupations. They care about their children's learning and growth, hope to understand what their children are learning at school to better assist with their studies, but have limited chemistry knowledge themselves.<br>|SLM-Base|Dear parents, do you know that in your child's chemistry learning process, there's an important method called the "movement-stillness combined" learning approach? It's just like in our daily life - sometimes children need to quietly think about problems, and other times they need to engage in hands-on experiments or practical activities. So why is skillfully capturing the optimal timing of "movement-stillness" transitions key to success? Think about it - at school, teachers use this learning method to transform children's thinking patterns, helping them learn through activities. For example, when students are learning about chemical elements, they might first quietly listen to the teacher explain the properties and characteristics of elements - this is the "stillness" phase. Then, the teacher will arrange experiments for children to perform hands-on operations and observe experimental phenomena - this is the "movement" phase. If parents can help their children capture the optimal timing of these "movement-stillness" transitions, their chemistry learning will become twice as effective with half the effort! So how can we help children achieve this in daily life? It's actually quite simple. For instance, after your child learns a chemistry concept, you can ask: "Sweetie, can you think of any phenomena in daily life that relate to this concept?" This is a small interaction that guides children from "stillness" to "movement." When your child starts thinking and answering, you can encourage them to do some simple experiments to verify their ideas, like conducting a small experiment using materials at home. This way, children can better understand and apply the "movement-stillness" learning method in their daily lives, and during this process, your parent-child relationship will grow closer too!|
| |SLM-Strict|As for why skillfully grasping the optimal timing of "movement-stillness" transitions is key to your child's success in learning chemistry - here's something to consider. The new curriculum approach encourages children to use a learning method that combines "movement and stillness," as this can transform their thinking patterns and help them learn through activities. Being able to capture these transition moments is crucial - when you catch these perfect moments, your child's learning becomes twice as effective with half the effort. So how can we parents help our children understand and apply this learning method in daily life? Let me share some simple parent-child interaction methods. For example, we can do a small experiment together with our children, like observing the reaction between baking soda and white vinegar. During this process, the handson operation is the "movement" part, while observing the phenomena and thinking about the reasons is the "stillness" part - see how this combines "movement and stillness"? You can guide your child to pay attention to these processes, and gradually, they'll better master this learning method. Plus, during this process, your parent-child relationship will naturally grow stronger!|
| |SLM-Relaxed|"A Parent's Guide to Supporting Children's Learning and Growth" During a child's learning and growth process, there is an important learning method called the movement-stillness transition method. It's like a magical key that can help children better understand and master knowledge, enhancing their learning effectiveness. So, how do we guide children to master this method? For parents, although their professional knowledge may be limited, they can help children understand through various scenarios in daily life. For example, during a family trip, when children encounter different scenery and activities, some children might initially run and play excitedly - this is the "movement" state. Then when they see beautiful scenery or encounter interesting things, they stop to carefully observe and think - this enters the "stillness" state. Parents should be good at ecognizing these transitions between states in their children, guiding them at appropriate times to help them understand that this process of "movement-stillness" transition is actually the learning process. Through such parent-child interactions, parents can not only better understand what their children are learning at school but also improve their parent-child relationship, accompanying their children in healthy and happy growth.|

- E.2 Prompts Although the term “rewrite” is used in some prompt templates as the editing instruction, it serves the same function as “reformulate” discussed in sections above, which aims to maintain the core meaning of the documents while only optimizing its expression.

# strict version You are a text polishing expert. You will polish text based on the given [Genre] and [Audience].

# relaxed version You are a creative expert skilled at transforming materials into creative inspiration and building independent, complete, and highly original texts.

When polishing, you must follow these 4 rules:

- 1. Read through the entire text and polish it according to the requirements of the given [Genre] and

[Audience]

- 2. The degree of polishing should not be too heavy − just aim to satisfy the requirements of [Genre] and [ Audience] as much as possible

- 3. Double−check that the polished text is suitable for the audience described in [Audience]!

- 4. Pay attention to the frequency of modal particles − the text should not contain too many modal particles

Requirements:

- 1. Read through the original text thoroughly, extract several key themes/keywords, transform to abstract or

universal concept inspiration, then generate entirely new text constructions.

- 2. Extract content from [Audience] and [Genre] sections, but don’t be constrained by them directly, just use them as creative inspiration.

- 3. Create and reformulat text around points 1/2, build new meaning from details to the whole structure.

- Figure 14 two different prompt templates, we keep the input aligned with MGA strategy, using raw text, genre, audience to fill the template.

#Identity and Capabilities# You are a content creation expert, specializing in text analysis and rewriting, capable of adapting content based on varying ‘‘genres’’ and ‘‘audiences’’ to produce ‘‘diverse’’ and ‘‘high−quality’’ texts. Your English writing is at native

editor level, and you will output your rewritten texts in English. International audiences particularly enjoy your work, which receives widespread readership and circulation, earning unanimous acclaim from the industry for your capabilities!

#Workflow# Please utilize your analytical and writing abilities to rewrite the text based on the original content and given ‘‘genre ’’ and ‘‘audience’’. Before beginning the rewrite, you will consider the following requirements:

- 1. First, read through the original text thoroughly, identify its information content and value, and consider how to prevent any loss of information points and value in the rewritten text

- 2. Focus on the original content, combine it with the given ‘‘genre’’ requirements, and rewrite the text following the descriptions, content modules, language requirements, and other stylistic elements specified in the ‘‘genre’’, to form an initial draft

- 3. Polish the initial draft according to the given ‘‘audience’’ requirements, and generate the final rewritten text in English

- 4. Refine the rewritten text to match native English speakers’ reading habits and expression patterns

#Detailed Requirements# Please ensure you follow the three workflow requirements above, then generate the final English rewritten text according to these detailed requirements. The given ‘‘audience’’ is <<<{audience}>>>. The given ‘‘genre’’ is <<<{genre}>>>.

#Raw Text# {raw_text}

Prompt 2 reformulation prompt template.

#Identity and Capabilities# You are a content creation expert, specializing in text analysis and rewriting, skilled at adapting content based on varying [genres] and [audiences] to produce ‘‘diverse’’ and ‘‘high−quality’’ texts. Your rewriting approaches consistently transform original texts into remarkable content, earning acclaim from both readers and industry professionals! #Workflow# Please utilize your imagination and creativity to generate 5 pairs of [genre] and [audience] combinations suitable for

the original text. Your analysis should follow these requirements:

- 1. First, analyze the characteristics of the source text, including writing style, information content, and value

- 2. Then, consider how to preserve the primary content and information while exploring possibilities for ‘‘broader audience engagement’’ and ‘‘alternative genres’’

#Detailed Requirements# Ensure adherence to the workflow requirements above, then generate 5 pairs of [genre] and [audience] combinations according to these specifications:

Your provided [genres] should meet the following requirements:

- 1. Clear Genre Definition: Demonstrate strong diversity; include genres you’ve encountered, read, or can envision

- 2. Detailed Genre Description: Provide 2−3 sentences describing each genre, considering but not limited to type, style, emotional tone, form, conflict, rhythm, and atmosphere. Emphasize diversity to guide knowledge adaptation for specific audiences, facilitating comprehension across different backgrounds. Note: Exclude visual formats ( picture books, comics, videos); use text−only genres.

Your provided [audiences] should meet the following requirements:

- 1. Clear Audience Definition: Demonstrate strong diversity; include both interested and uninterested parties, those who like and dislike the content, overcoming bias toward positive audiences only

- 2. Detailed Audience Description: Provide 2 sentences describing each audience, including but not limited to age, occupation, gender, personality, appearance, educational background, life stage, motivations and goals, interests, and cognitive level

#Response# {

- ‘‘audience_1’’: audience1,

- ‘‘genre_1’’: genre1,

‘‘audience_2’’: audience2,

- ‘‘genre_2’’: genre2,

‘‘audience_3’’: audience3,

- ‘‘genre_3’’: genre3,

‘‘audience_4’’: audience4,

- ‘‘genre_4’’: genre4,

‘‘audience_5’’: audience5,

- ‘‘genre_5’’: genre5

} #Input# {raw_text}

Prompt 3 genre-audience pairs prompt template.

#Identity and Capabilities# You are a Content Reviewer, skilled at analyzing texts and keenly identifying and analyzing the relationships, similarities, and differences between two texts. Your thorough analysis of each pair of texts, with attention to every detail, provides great convenience for subsequent review work!

#Thinking Process# Please fully utilize your analytical abilities, review capabilities, and deep thinking skills to analyze the ‘‘Rewritten Text’’ against the ‘‘Original Text’’ as a benchmark, ultimately providing analysis and scoring for [A]. You will follow these steps for detailed consideration:

- 1. First, you will read through the original text thoroughly, identifying the information points in the ‘‘Original Text ’’

- 2. You will also read through the rewritten text thoroughly, identifying the information points in the ‘‘Rewritten Text’’

- 3. Compare the information in both texts’ content. The ‘‘Rewritten Text’’ is allowed to have new information points, different writing styles, expression styles, order, and focus from the ‘‘Original Text’’. As long as it is created based on some information points from the ‘‘Original Text’’, it is considered good for [A]

- 4. After careful analysis and review, please clearly list the connections and differences between the two texts, and based on this, provide final analysis and scoring for [A]

#Detailed Requirements# The scoring judgment for [A] must follow these standards:

- 1. The ‘‘scoring range’’ is 1−5 points. You need to analyze and grasp each aspect mentioned in #Thinking Process #, and differentiate scores accordingly. Be strict, don’t be too lenient with scoring!

- 2. The ‘‘Rewritten Text’’ is allowed to differ from the ‘‘Original Text’’ in writing style, expression style, and focus! This cannot be a basis for deducting points!

- 3. The ‘‘Rewritten Text’’ is allowed to omit some information from the ‘‘Original Text’’! It is not required that all information from the ‘‘Original Text’’ appears in the ‘‘Rewritten Text’’! This also cannot be a basis for deducting points! If this is the only issue, please give a full score of 5 points.

In scoring [A], the following situations will ∗∗NOT reduce∗∗ the score for [A]:

- 1. The ‘‘Rewritten Text’’ can include information points not present in the ‘‘Original Text’’

- 2. The added content in the ‘‘Rewritten Text’’ significantly deviates from the core information of the ‘‘Original Text ’’

- 3. The expression style, order, and focus of the ‘‘Rewritten Text’’ differ from the ‘‘Original Text’’

In scoring [A], the following situations ∗∗WILL reduce∗∗ the score for [A]:

- 1. The information points in the ‘‘Rewritten Text’’ differ so greatly from the ‘‘Original Text’’ that it’s not recognizable as being rewritten from the ‘‘Original Text’’

- 2. The ‘‘Rewritten Text’’ contains none of the information points from the ‘‘Original Text’’

#Original Text# {raw_text}

#Rewritten Text# {rewritten_text}

#Response Format# {

‘‘A’’:{ ‘‘analysis’’: ‘‘xxx’’, provide reasons for point deductions ‘‘score’’: 1, 2, 3, 4, or 5

}, }

Prompt 4 Full LLM judger prompt.

