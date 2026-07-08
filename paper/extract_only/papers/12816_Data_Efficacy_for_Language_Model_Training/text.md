arXiv:2506.21545v1[cs.CL]26Jun2025

# Data Efficacy for Language Model Training

Yalun Dai Yangyu Huang∗ Xin Zhang Wenshan Wu Chong Li Wenhui Lu Shijie Cao Li Dong Scarlett Li

Microsoft Research

## Abstract

Data is fundamental to the training of language models (LM). Recent research has been dedicated to data efficiency, which aims to maximize performance by selecting a minimal or optimal subset of training data. Techniques such as data filtering, sampling, and selection play a crucial role in this area. To complement it, we define Data Efficacy, which focuses on maximizing performance by optimizing the organization of training data and remains relatively underexplored. This work introduces a general paradigm, DELT, for considering data efficacy in LM training, which highlights the significance of training data organization. DELT comprises three components: Data Scoring, Data Selection, and Data Ordering. Data Scoring assigns a score for each data sample based on its properties, such as quality, difficulty, and learnability. Data Selection optionally selects a subset from the original training data based on the scores. Data Ordering utilizes these scores to organize the training data in a new, optimized order, rather than the traditional random shuffling. Furthermore, we design Learnability-Quality Scoring (LQS), as a new instance of Data Scoring, which considers both the learnability and quality of each data sample from the gradient consistency perspective. We also devise Folding Ordering (FO), as a novel instance of Data Ordering, which addresses issues such as model forgetting and data distribution bias. Comprehensive experiments validate the data efficacy in LM training, which demonstrates the following: Firstly, different DELT instances enhance LM performance to varying degrees without increasing the data scale and model size. Secondly, among these instances, the combination of our proposed LQS for data scoring and FO for data ordering achieves the most significant improvement. Lastly, data efficacy can be achieved together with data efficiency by applying data selection. Therefore, we believe that data efficacy is a promising foundational area in LM training. The Code is publicly available now.

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>34.50%<br>35.00%<br><br>35.50%<br>36.00%<br><br>36.50%<br>37.00%<br><br><br>37.50%<br>38.00%<br><br><br>38.50%<br><br><br>50% 60% 70% 80% 90% 100%<br><br>AverageAccuracy<br><br>(Efficacy)<br><br>Data Selection Ratio<br><br>(Efficiency)<br><br>Conventional KenLM PDS DELT (Ours) 38.02%<br><br>36.37%<br><br>Efficacy +1.65%<br><br>36.47%<br><br>Efficiency x2.0<br><br>Average result across 8 benchmarks for different methods. High performance at|
|---|

- Figure 1: the same selection ratio indicates high efficacy, while achieving similar performance with a smaller selection ratio demonstrates high efficiency. Our method excels in both efficacy and efficiency.

∗Corresponding author

Preprint. Under review.

## 1 Introduction

The significance of language models [1, 2, 3] is immense in modern computational applications. From natural language processing tasks such as translation [4] and sentiment analysis [5] to more complex applications like automated reasoning [6] and conversational agents [7], language models have revolutionized the way machines understand, generate, and interact with human beings using natural language. To empower language models having these abilities, data is central to their training and serves as the foundation from which models learn knowledge based on linguistic patterns and structures. Consequently, meticulous data curation is essential to ensure consistently high model performance across various applications.

Recent research has therefore concentrated on data efficiency, selecting the smallest or highestquality subset of the corpus that still yields strong results [8, 9, 10]. Once that subset is chosen, however, every surviving sample is ordinarily treated the same, and the order in which samples are shown to the model is random. In this work, we define data efficacy as improving model performance by optimizing the organization of training data. This area complements data efficacy and is still in its early stage, with its potential demonstrated through curriculum learning [11, 12] that feeds examples to model from easy to hard.

In this context, we notice that the latest generation of language models [13, 3, 14] typically trains for only a few epochs, usually just one, due to the vast scale of training datasets but limited computing power. These models contrast with previous generations [15, 16] by the scaling law [17], which trained over many epochs and often led to overfitting. This aligns with the findings of QQT [18], which shows that high-quality data quickly loses its utility after being used repeatedly. In other words, it is more effective to utilize a large amount of training data with few epochs rather than rely on high-quality data with many epochs. Consequently, effectively organizing the training dataset is essential for enhancing the performance of language models trained with only a few epochs.

Expanding on this insight, we propose a general paradigm for data efficacy that achieves benefits without altering the dataset content and the model architecture, which makes it an almost cost-free approach. Specifically, this paradigm incorporates data scoring, data selection, and data ordering components. Data scoring assigns a score to each sample, which reflects factors like difficulty, quality, diversity, and learnability. Data selection involves optionally choosing a subset of the original training data based on these scores. Data ordering then organizes the training data according to these scores, either in ascending, descending, or other arrangements. Curriculum learning [11, 12] can be viewed as a specific example within our paradigm, with ascending ordering based on difficulty scoring.

To verify the proposed paradigm, we integrate some baseline methods into it and also design new methods respectively for data scoring and ordering. The key results from Figure 1 highlight that the proposed DELT significantly improves data efficacy in LM training on a set of typical benchmarks. Meanwhile, it outperforms existing methods [10, 19] in data efficiency that further boosts LM performance across all selection ratios.

The main contributions of this paper are as below:

- • We identified the potential of the underexplored area, data efficacy, in language model training and proposed a general paradigm for this area, DELT, which consists of data scoring, data selection, and data ordering.
- • We designed an innovative method for data scoring, called Learnability-Quality Scoring (LQS), which evaluates the score for each data sample based on learnability and quality from the gradient consistency perspective.
- • We devised a novel method for data ordering, named Folding Ordering (FO), which optimizes LM training and mitigates the issues of model forgetting and data distribution bias.
- • We conducted comprehensive experiments to validate the DELT paradigm on mainstream benchmarks, employing different data scoring and data ordering methods. All instances of the proposed paradigm improved performance, with our design outperforming the rest.

Through these contributions, we aim to provide a general paradigm for understanding and applying data efficacy in LM training, paving the way for more effective model development practices.

## 2 Related Work

- 2.1 Data Sources

Data source of LM training [20, 21, 3, 1, 14] can primarily be categorized into five types: internet data [22], books [23], synthetic data [24], physical sensors [25], and human perception of the real world. Internet data is the primary source for language model training due to its vast scale. Books and synthetic data offer high quality, but are limited in scale. Data from physical sensors and human perception are in other modalities or still under development. Several studies focus on extracting high-quality datasets for LM training, like C4 [26], RefinedWeb [27], RedPajama [28], and RedStone [29]. All of them utilize an identical data source, CommonCrawl [22], which captures snapshots of web pages from the entire internet at different periods and contains over 200 billion samples to date.

- 2.2 Data Efficiency

Data efficiency [19, 10, 9, 8] focuses on selecting the most relevant data points for inclusion in training dataset and optimizing the performance of the language model. This area includes well-researched strategies such as data selection [19, 10, 30, 31], sampling [9], denoising [32, 33], and deduplication [34, 35], all of which aim to select optimal data for efficient model training. The KenLM [19] trains a fast and small model for perplexity estimation and treats the perplexity as the data difficulty for LM. The PDS [10] evaluates the quality of data samples by measuring the consistency of each sample’s gradient direction with a reference direction. The DSIR [9] develops an importance weight estimator to select a subset of raw data that mirrors the distribution of the target in a specific feature space. The MATES [30] presents a data influence model that continuously adapts to the evolving data preferences of the pre-trained model, selecting the most effective data for the current stage of pre-training. The SemDeDup [34] utilizes embeddings from pre-trained models to identify and remove data pairs that are semantically similar but not exactly identical. All these methods develop strategies to decide whether a sample should be retained or discarded. However, for retained samples, language models train on them equally, without considering differences in criteria.

- 2.3 Data Efficacy

Data efficacy, distinct from data efficiency, aims to maximize the performance of language models by optimizing the organization of training data. Curriculum learning, as described by [11], involves starting with simpler examples and progressively tackling more complex ones, aiding in smoother model convergence. Within curriculum learning, [36] presents an attention score to determine the prompt difficulty, and [37] introduces a soft edit distance to measure sample difficulty. Similarly, annealing learning, as outlined by [3], seeks to improve model performance by initially training on a large, noisy dataset and concluding with a small, high-quality dataset. All these methods sort training data directly by difficulty or quality. However, since limited research on data efficacy, there is no established paradigm for effectively organizing training data.

To conclude, data is essential for training language models, and numerous large-scale data sources originate from the internet. Nevertheless, obtaining incremental public data has become challenging due to the slow growth of CommonCrawl [22] snapshots and the increasing presence of AI-generated content online. As language models scale up, effectively leveraging existing data sources becomes vital, which makes data efficacy increasingly important. Despite this, few studies focus on data efficacy in language model training. To address this gap, we propose a general paradigm for it, where curriculum learning and annealing learning are two specific instances.

## 3 Paradigm

### 3.1 Problem Formulation

Language model (LM) is represented by parameters θ ∈ RN, which is pre-trained from scratch or finetuned from weights on a dataset D = {xn}|D|n=1 with T training steps. The N is the quantity of θ and xn is the n-th sample in D. The goal of the problem is to optimize the LM performance by reorganizing training dataset D on downstream tasks, which is named Data Efficacy in this paper.

##### DELT

Efficacy

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Training Pipeline

Data Scoring

Data Ordering

[Figure 15]

[Figure 16]

[Figure 17]

or

Efficiency

Ordered

Scored data

Sorting Folding

Difficulty

[Figure 18]

[Figure 19]

[Figure 20]

Data Selection

Raw data

data Language model

Quality

[Figure 21]

Learnability

[Figure 22]

Selected data

Top-k

- Figure 2: Paradigm of Data Efficacy for LM training. The blue box represents the paradigm DELT. Both the methods for data scoring and data ordering components in DELT are flexible and can be adjusted, including those outlined in Section 4. Meanwhile, the data selection is an optional component that can further improve data efficiency. Within the proposed DELT, data efficacy and efficiency are seamlessly compatible, working together to optimize language model performance.

### 3.2 Paradigm Definition

A paradigm is proposed in Figure 2 to improve data efficacy in LM training without altering the data content D and model parameters θ. It comprises three components:

- • Data Scoring: it aims to assign a score for each training sample based on specific criteria, such as quality, difficulty, diversity, and learnability. These scores are then applied to guide data selection and data ordering in subsequent stages.
- • Data Selection: it seeks to select an optimal subset Dsub from the dataset D, ensuring that LMs trained on Dsub achieve the best possible performance. This process alters the quantity of dataset D but does not influence the organization of Dsub.
- • Data Ordering: it targets at reorganizing the order of training samples in D (or Dsub) to create D′, such that LMs trained on D′ achieve superior performance. This process focuses on the organization of dataset D (or Dsub), while it does not change the dataset scale.

Unlike the baseline method, where the language model is trained directly on the raw data D, and the data efficiency methods that use a selected subset Dsub, DELT processes the raw data D as follows:

Firstly, data scoring, defined as f, assigns a score vector γ to the raw data D, where γ lies in a |D|-dimensional simplex. Samples with large γ are considered good according to their criteria.

γ = f(D) = γ1,γ2,··· ,γ|D| ⊤ (1)

Then, data selection, denoted as fs, identifies a subset Dsub from D based on the scores γ by the selection ratio r. The number of samples K to be selected is determined by r. The function rank provides the ranking index of each element in the set γ in ascending order.

Dsub = fs(D;γ,K) = {xk | rank(γk) > |D| − K and 1 ≤ k ≤ |D|} (2) K = ⌊r · |D|⌋ (3)

Finally, data ordering, represented by fo, reorganizes the D or Dsub into a new dataset D′ with unchanged size, based on a permutation π determined by γ. It could be πsort that returns the indices of each element in γ after sorting or other functions.

D′ = fo(D;γ) = xπ(γ)

(4)

,··· ,xπ(γ)

,xπ(γ)

2

1

|D|

Compatibility of data efficacy and data efficiency in DELT. As shown in Figure 2, the DELT paradigm can build upon data scoring and data ordering by incorporating data selection to further enhance data efficiency. The entire DELT process can be defined as a transformation of the original dataset D into a reordered dataset D′:

D′ = fo(γo) ◦ fs(D;γs,K), (5) where the symbol ◦ denotes functional composition. γo and γs are the score vectors for data ordering and data selection, respectively. Since data scoring often requires substantial computation time, both data selection and data ordering in DELT apply a shared score vector for practicality, i.e., γo = γs = γ. This process ensures that the most qualified samples are selected and then optimally ordered, thereby significantly improving model performance in both data efficacy and efficiency.

## 4 Method

The DELT paradigm can adopt various specific data scoring and data ordering methods. Below, we introduce some baseline methods for each of them and propose an improved approach. These methods are not exhaustive, and other options are also viable. For data selection, top-K is applied.

- 4.1 Data Scoring

- 4.1.1 Baseline Methods

Existing methods typically focus on attributes such as quality [10], difficulty [19], noisiness [32, 33], or diversity [34, 35] to compute scores for data selection.

KenLM [19] as a small n-grams model is applied to learn the perplexity of training samples, which is considered to score data samples by their difficulty.

PDS [10] trains a compact neural network to evaluate the quality of data samples, which is represented by the consistency of each sample’s gradient direction with a reference direction.

However, these methods, designed primarily for data selection, often focus solely on how good a sample is, while overlooking the question of where a sample contributes most effectively within the context of the entire dataset.

- 4.1.2 Our Method

Learnability-Quality Scoring (LQS) is introduced to address this limitation and make the scorer more attentive to the utility of each data sample. By incorporating both learnability and quality, LQS is not only sensitive to low-quality samples but also better weights the impact of samples during model training. Our method dynamically evaluates how each sample contributes to reducing the downstream loss J(θ) by considering its behavior at different training stages.

The learnability of each data sample represents the difficulty change during model training, as illustrated in Figure 3a. For training step t from 1 to T, the learnability of a sample xn is defined as its ability to reduce the loss over time during training. The learnability is represented as:

T−1

T−1

∥∇ℓ(xn,θt)∥ ∥∇ℓ(xn,θt+1)∥

ln,t ln,t+1

, (6)

L(xn) =

=

t=1

t=1

where ∇ℓ(xn,θt) denotes the gradient of loss function for sample xn at training step t, with model parameters θt, and ln,t is its magnitude. A high learnability score indicates that the sample significantly reduces the loss in training, particularly if its gradient magnitude is initially high and decreases substantially over time. Such samples are challenging yet beneficial for training, making them more suitable for later stages of training. Conversely, noisy samples or those with unstable gradients yield a low learnability score, enabling their identification and efficient filtering during data selection.

The quality of each data sample contributes to data efficacy during model training, as depicted in Figure 3b. It is measured by the consistency of ∇ℓ(xn,θt) with a target vector λt+1 in Equation 8, which represents the average gradient of the loss function for all data at training step t + 1. The quality score is computed as:

T−1

T−1

λt+1⊤∇ℓ(xn,θt) ∥λt+1∥ · ∥∇ℓ(xn,θt)∥

Q(xn) =

, (7)

cos(αn,t) =

t=1

t=1

where αn,t is the angle between two vectors. A higher cosine similarity cos(αn,t) indicates that the gradient convergence direction on xn is more aligned with the target objective λt+1, implying a stronger contribution to reducing the loss J(θ). As defined in [10], the target vector λt is:

λt+1 + ∇J(θt) − η · ∇2L(θt,γ) · λt+1, if t < T ∇J(θt), if t = T

(8)

λt =

Finally, we combine learnability and quality into a unified function to score data samples. For a detailed explanation of the formula, please refer to the Appendix. The score vector γ is defined as:

T−1

λt+1⊤∇ℓ(xn,θt) ∥∇ℓ(xn,θt+1)∥

,1 ≤ n ≤ |D| (9)

γ = γn γn =

t=1

Larger γn values indicate samples with higher quality and significant contributions to reducing the downstream loss J(θ), especially when introduced during later training stages. In contrast, lower γn values correspond to samples that are easy and less informative, better suited for early-stage training, or potentially noisy, which can be filtered out in data selection settings. For detailed implementation of LQS, please refer to the Appendix.

𝒍𝟐,𝒕 𝟏

𝒍𝟐,𝒕

𝜶𝟑,𝒕

𝜵𝓵 𝒙𝟐,𝜽𝒕 𝟏

𝜵𝓵 𝒙𝟐,𝜽𝒕

𝒍𝟏,𝒕 𝟏

𝜶𝟏,𝒕 𝜵𝓵 𝒙𝟑,𝜽𝒕 𝜶𝟐,𝒕

𝒍𝟏,𝒕 𝒍𝟑,𝒕 𝒍𝟑,𝒕 𝟏

𝜵𝓵 𝒙𝟐,𝜽𝒕

𝜵𝓵 𝒙𝟏,𝜽𝒕 𝟏

𝜵𝓵 𝒙𝟏,𝜽𝒕

𝜵𝓵 𝒙𝟏,𝜽𝒕

𝜵𝓵 𝒙𝟑,𝜽𝒕

𝜵𝓵 𝒙𝟑,𝜽𝒕 𝟏

𝝀𝒕 𝟏

𝝀𝒕 𝟐

𝝀𝒕 𝟏

|> ><br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]|
|---|

|[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>> >|
|---|

(a) Learnability Scores:𝓛 𝒙𝟏 > 𝓛 𝒙𝟐 > 𝓛 𝒙𝟑 (b) Quality Scores:𝑸 𝒙𝟏 > 𝑸 𝒙𝟐 > 𝑸 𝒙𝟑

#### Figure 3: Illustration of LQS scoring method. The left part demonstrates the calculation of the learnability score, and the right part depicts the computation of the quality score.

- 4.2 Data Ordering

- 4.2.1 Baseline Methods

Existing methods usually utilize the shuffling method to organize training data. Additionally, some approaches apply curriculum learning [11, 36], which can be considered a type of sorting method.

Shuffling method randomly arranges the order of training data to prevent inherent imbalanced distribution. It acts as a baseline for data organization but does not account for data efficacy.

Sorting method sorts training data based on specific criteria, such as quality or difficulty. The permutation function πsort provides the ranking index of each element in the training data in ascending order. Curriculum learning is an example, mimicking the human learning process by starting with simpler samples and gradually increasing difficulty. Although sorting-based approaches can enhance training efficacy, they may encounter issues like model forgetting, data distribution bias, and even data duplication, which can hinder performance.

- 4.2.2 Our Method

Folding method is proposed to improve training data efficacy and address the negative influences brought by the sorting method. The new method, named folding learning, reorganizes the dataset by repeating the curriculum learning multiple times without duplication. The repeated times is defined as the folding layers L. As demonstrated in Figure 4, the folding method samples sorted data L times without replacement at a fixed interval L. The permutation function πfold is defined in Equation 10, while πsort is described in Equation 4. Folding learning not only inherits the benefits from curriculum learning but also mitigates issues of model forgetting, data distribution bias, and even duplication.

L−1

⟨πsort(γ)i | i ∈ {j | j ≡ ℓ (mod L), 1 ≤ j ≤ |D|}⟩ (10)

πfold(γ;L) =

ℓ=0

[Figure 29]

[Figure 30]

[Figure 31]

(a) Random Shuffle (b) Ascending Sorting (c) Folding (Ours)

#### Figure 4: Illustration of ordering methods. The right one is Folding method, an advanced multi-fold version of the Sorting method. These results are based on 500 random samples from RedPajama.

## 5 Experiment

### 5.1 Experimental Setup

Data. (1) General data. We utilize the Redpajama [28] sourced from CommonCrawl as D, which offers a relatively balanced knowledge distribution [38]. The downstream loss J(θ) for the data scoring model is computed on the LIMA [39], which is a high-quality dataset with 1,030 diverse instruction-response pairs spanning various downstream scenarios. (2) Math data. We use the OpenWebMath [40] as D. The downstream loss J(θ) is computed on the MiniF2F [41], which is a high-quality dataset consisting of 488 manually formalized mathematical problem statements, spanning multiple domains. (3) Code data. We employ The-Stack-v2 dataset [42] as D. The downstream loss J(θ) is computed on the Epicoder-380k [43], which is a synthetic dataset with 380k diverse instruction-response pairs spanning multiple code generation scenarios.

Model. We apply the Mistral [44] architecture for pre-training on general data, and the Qwen1.5 [45] for post-training in the math and code domain respectively using the official pre-trained weights.

Training. Unless otherwise specified, we pre-train all LMs for one epoch, using a batch size of 512 and a maximum input length of 1,024. For the model, we utilize 160M parameters by default. For pretraining, we randomly select a 1B-token subset from Redpajama by default, while for post-training, we sample a 1B-token subset each from OpenWebMath and The-Stack-v2. This setting allows us to evaluate the impact of different data scoring and data ordering methods on LM training using D.

Baselines. Based on the DELT pipeline, we compare the proposed methods with existing baselines for data scoring and ordering. As described in Section 4, the methods include: 1) Data scoring: KenLM [19], PDS [10]; 2) Data ordering: Shuffling (Random), Sorting (Curriculum Learning) [11].

For additional details of the experimental setup, such as evaluation, please see the Appendix.

### 5.2 Main Results

Data Efficacy with Different Model Sizes and Data Scales. Table 1 presents the evaluation results of the pre-trained LMs on the OLMo [46] evaluation benchmarks. As shown, DELT consistently outperforms the baselines on most datasets, achieving the best overall performance across models with 160M, 470M, and 1B parameters (Table 9a), as well as across data sizes of 1B, 10B, and 50B tokens (Table 9b). These results demonstrate that DELT steadily improves the data efficacy in LM training across various model sizes, data scales, and downstream tasks.

- Table 1: Efficacy results on different downstream benchmarks. The conventional method presents the average result over three random seeds in this and the following tables. Ours means applying LQS for data scoring and Folding for data ordering within the DELT paradigm.

(a) Results (%) for 1B-token data across model sizes (160M, 470M, 1B).

ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. Model size = 160M

Conventional 21.27 34.32 27.85 20.25 24.40 55.19 56.93 50.72 36.37 Ours 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08

Model size = 470M

- Conventional 21.16 34.91 28.11 21.88 23.90 56.07 58.75 50.04 36.85 Ours 22.33 35.88 28.45 23.26 26.60 57.20 60.10 52.81 38.33

Model size = 1B

Conventional 20.58 36.12 28.32 23.56 25.00 56.49 60.05 52.07 37.77 Ours 22.76 37.95 29.95 26.38 26.00 58.07 60.90 51.28 39.17

(b) Results (%) for 160M model across data sizes (10B, 50B).

ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. Data size = 10B tokens

- Conventional 22.82 38.51 30.72 30.40 25.70 57.32 64.90 51.54 40.24 Ours 24.38 39.80 31.64 32.98 27.21 58.56 66.70 51.67 41.62

Data size = 50B tokens

Conventional 24.06 41.88 32.05 33.79 26.80 58.11 69.00 51.93 42.20 Ours 24.65 41.07 33.00 36.07 29.30 59.10 68.40 52.67 43.03

Adaptability of Different Methods in DELT. Table 2 presents the results of different methods applied within the DELT paradigm, all of which significantly outperform the conventional baseline. Notably, regardless of whether data selection is applied, our proposed LQS scoring method achieves the best results. Furthermore, our proposed Folding ordering method consistently provides noticeable improvements across all baseline methods.

- Table 2: Efficacy results of different DELT implementations. The best scores for each model size are highlighted in bold, while the second-best scores are shown in italic bold. The selection methods report the highest scores across all selection ratios.

Pipeline Scoring Selection Ordering ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. Conventional - - - 21.27 34.32 27.85 20.25 24.40 55.19 56.93 50.72 36.37

KenLM - Sorting 21.93 33.96 28.09 20.69 25.20 54.79 56.20 50.59 36.43 KenLM - Folding 20.98 35.00 28.02 22.55 23.90 56.54 58.30 51.36 37.08 PDS - Sorting 22.44 34.18 27.98 21.35 25.40 55.28 55.80 49.17 36.45 PDS - Folding 21.93 34.81 28.04 22.43 26.00 56.42 59.30 50.20 37.40 LQS - Sorting 23.22 35.24 28.03 22.79 24.70 56.85 57.90 51.17 37.49 LQS - Folding 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08

DELT

KenLM ✓ Sorting 21.93 34.68 27.78 19.37 26.40 54.95 56.30 52.96 36.80 KenLM ✓ Folding 22.10 34.30 27.62 21.56 25.00 56.26 58.10 52.80 37.22 PDS ✓ Sorting 22.61 35.27 28.08 19.68 25.80 56.53 59.60 51.54 37.38 PDS ✓ Folding 21.66 36.01 28.05 24.33 24.10 55.61 61.70 52.47 37.99 LQS ✓ Sorting 22.10 35.61 28.05 22.53 23.60 55.93 59.60 51.38 37.35 LQS ✓ Folding 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08

Data Efficiency Promotion on Existing Methods. Figure 5 further evaluates the data efficiency of the DELT framework by involving the data selection setting. Compared to previous methods that only consider data selection (Shuffling), the DELT framework (Sorting and Folding) achieves superior performance across the majority of selection ratios. The results show that the DELT framework is compatible with the data selection method, and the combination further improves their data efficiency.

[Figure 32]

[Figure 33]

(a) KenLM Scoring (b) PDS Scoring

- Figure 5: The performance of KenLM [19] and PDS [10] under different data selection ratios, both with and without the DELT paradigm. Data efficiency is enhanced when integrated into DELT.

Domain Robustness in Post-training (Table 3). To validate the robustness of the DELT paradigm, we conduct post-training experiments on datasets of OpenWebMath [40] and The-Stack-v2 [42] respectively across math and code domains, both of which are sourced from the web data. As shown in Table 3, our method consistently outperforms the baselines across benchmarks in different domains, demonstrating the strong versatility of the DELT under our proposed LQS and FO methods.

Table 3: Efficacy of models trained on different domain-specific datasets. (a) Results on code domain.

(b) Results on math domain.

Method HumanEval MBPP

Method MathQA GPQA Diamond

- Qwen1.5-0.5B

Conventional 7.00 7.93 Ours 9.76 9.40

- Qwen1.5-1.8B

Conventional 9.15 12.00 Ours 16.46 13.20

- Qwen1.5-0.5B

Conventional 21.23 24.92 Ours 22.73 26.83

- Qwen1.5-1.8B

Conventional 22.72 27.17 Ours 24.75 28.94

Stability on Different Epochs (Figure 6). In addition to one-epoch training, we also evaluate the effectiveness of DELT under a multi-epoch setting. As shown in Figure 6, our method consistently

boosts the results of conventional random ordering as the number of epochs increases. While conventional random ordering exhibits fluctuations and slow improvements after the second epoch, our approach demonstrates steady progress, showcasing the effectiveness and stability of our method in maintaining superior performance over multiple epochs.

### 5.3 Ablation Study

Ordering Method (Table 4). Data ordering is a key component of the DELT framework and plays a significant role in improving data efficacy. To verify the effectiveness of different ordering methods and identify the optimal one for DELT, we conduct corresponding experiments. The results in Table 4 show that ascending sorting improves the result while descending sorting leads to a decline. Besides, the proposed Folding method shows the most improvement among all the ordering methods. This indicates the rationality and importance of data ordering within the DELT framework.

Table 4: Comparison among different ordering methods. Sortingasc refers to an ascending sorting by scores (from low to high), while sortingdes denotes a descending sorting (from high to low).

Ordering ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. Conventional - 21.27 34.32 27.85 20.25 24.40 55.19 56.93 50.72 36.37

Sortingdes 22.01 33.67 27.77 14.67 23.80 54.90 51.90 51.93 35.08↓ Sortingasc 22.44 34.18 27.98 21.35 25.40 55.28 55.80 49.17 36.45↑

PDS

###### Folding 21.93 34.81 28.04 22.43 26.00 56.42 59.30 50.20 37.39↑

Sortingdes 20.69 34.72 27.78 21.20 23.10 56.12 58.20 49.11 36.36↓ Sortingasc 22.18 35.40 28.01 23.48 23.80 55.60 56.80 51.07 37.04↑

LQS

###### Folding 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08↑

L in Folding Learning (Figure 7). We explore the influence of different folding layers L on model performance in Figure 7. As L > 1, the performance consistently surpasses that of L = 1 (curriculum learning), verifying the benefits brought by folding learning. The average performance initially increases and then gradually declines, peaking at L = 3. In the experiments conducted in this paper, L is set to a default value of 3. For more experimental details, please refer to the Appendix.

[Figure 34]

- Figure 6: Performance on different epochs. Benchmarks from OLMo [46] are applied.

[Figure 35]

Figure 7: Influence of the folding layers L. Benchmarks from OLMo [46] are applied.

## 6 Conclusion

Regarding the underexplored research on data efficacy, we propose a general paradigm, DELT, for enhancing data efficacy in language model training. Meanwhile, we explore the relationship between data efficacy and efficiency, noting that their purposes differ significantly while they are related. Our comprehensive experiments with various DELT implementations confirm the effectiveness of DELT, and our newly designed methods, respectively for data scoring and ordering, outperform the other methods. We believe that the proposed paradigm highlights the potential of data efficacy as a field beyond existing methods, such as curriculum learning. Moreover, extensive experiments reveal that data ordering and data selection are compatible, and their combined use can further boost performance. This suggests the possibility of unifying the paradigms by incorporating data scoring, data selection, and data ordering. By adopting this unified paradigm, it becomes feasible to simultaneously consider data efficacy and efficiency, paving the way for promising future directions.

## References

- [1] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [3] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [4] Julia Hirschberg and Christopher D Manning. Advances in natural language processing. Science, 349(6245):261–266, 2015.
- [5] Karthick Prasad Gunasekaran. Exploring sentiment analysis techniques in natural language processing: A comprehensive review. arXiv preprint arXiv:2305.14842, 2023.
- [6] Fei Yu, Hongbo Zhang, Prayag Tiwari, and Benyou Wang. Natural language reasoning, a survey. ACM Computing Surveys, 56(12):1–39, 2024.
- [7] Sheetal Kusal, Shruti Patil, Jyoti Choudrie, Ketan Kotecha, Sashikala Mishra, and Ajith Abraham. Aibased conversational agents: a scoping review from technologies to future directions. IEEE Access, 10:92337–92356, 2022.
- [8] Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. A survey on data selection for language models. arXiv preprint arXiv:2402.16827, 2024.
- [9] Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy S Liang. Data selection for language models via importance resampling. Advances in Neural Information Processing Systems, 36:34201–34227, 2023.
- [10] Yuxian Gu, Li Dong, Hongning Wang, Yaru Hao, Qingxiu Dong, Furu Wei, and Minlie Huang. Data selection via optimal control for language models. arXiv preprint arXiv:2410.07064, 2024.
- [11] Daniel Campos. Curriculum learning for language modeling. arXiv preprint arXiv:2108.02170, 2021.
- [12] Xin Wang, Yudong Chen, and Wenwu Zhu. A survey on curriculum learning. IEEE transactions on pattern analysis and machine intelligence, 44(9):4555–4576, 2021.
- [13] OpenAI. hello-gpt-4o. (2024).
- [14] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [15] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.
- [16] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
- [17] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [18] Sachin Goyal, Pratyush Maini, Zachary C Lipton, Aditi Raghunathan, and J Zico Kolter. Scaling laws for data filtering–data curation cannot be compute agnostic. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22702–22711, 2024.
- [19] Kenneth Heafield. KenLM: Faster and smaller language model queries. In Chris Callison-Burch, Philipp Koehn, Christof Monz, and Omar F. Zaidan, editors, Proceedings of the Sixth Workshop on Statistical Machine Translation, pages 187–197, Edinburgh, Scotland, July 2011. Association for Computational Linguistics.
- [20] Anthropic. Claude 3 haiku: our fastest model yet. (2024).

- [21] Ziheng Yang. Paml 4: phylogenetic analysis by maximum likelihood. Molecular biology and evolution, 24(8):1586–1591, 2007.
- [22] Ahad Rana. Common crawl – building an open web-scale crawl using hadoop, 2010.
- [23] Michael Hart. Project gutenberg, 2004.
- [24] Sergey I Nikolenko et al. Synthetic data for deep learning, volume 174. Springer, 2021.
- [25] Sanem Kabadayi, Adam Pridgen, and Christine Julien. Virtual sensors: Abstracting data from physical sensors. In 2006 International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM’06), pages 6–pp. IEEE, 2006.
- [26] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [27] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data only. Advances in Neural Information Processing Systems, 36:79155–79172, 2023.
- [28] Maurice Weber, Dan Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, et al. Redpajama: an open dataset for training large language models. Advances in neural information processing systems, 37:116462–116492, 2024.
- [29] Yaoyao Chang, Lei Cui, Li Dong, Shaohan Huang, Yangyu Huang, Yupan Huang, Scarlett Li, Tengchao Lv, Shuming Ma, Qinzheng Sun, et al. Redstone: Curating general, code, math, and qa data for large language models. arXiv preprint arXiv:2412.03398, 2024.
- [30] Zichun Yu, Spandan Das, and Chenyan Xiong. Mates: Model-aware data selection for efficient pretraining with data influence models. Advances in Neural Information Processing Systems, 2024.
- [31] Yalun Dai, Lingao Xiao, Ivor Tsang, and Yang He. Training-free dataset pruning for instance segmentation. In The Thirteenth International Conference on Learning Representations.
- [32] QiHao Zhao, Wei Hu, Yangyu Huang, and Fan Zhang. P-diff+: Improving learning classifier with noisy labels by noisy negative learning loss. Neural Networks, 144:1–10, 2021.
- [33] Wei Hu, QiHao Zhao, Yangyu Huang, and Fan Zhang. P-diff: Learning classifier with noisy labels based on probability difference distributions. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 1882–1889. IEEE, 2021.
- [34] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023.
- [35] Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari Morcos. D4: Improving llm pretraining via document de-duplication and diversification. Advances in Neural Information Processing Systems, 36:53983–53995, 2023.
- [36] Jisu Kim and Juhwan Lee. Strategic data ordering: Enhancing large language model performance through curriculum learning. arXiv preprint arXiv:2405.07490, 2024.
- [37] Ernie Chang, Hui-Syuan Yeh, and Vera Demberg. Does the order of training samples matter? improving neural data-to-text generation with curriculum learning. arXiv preprint arXiv:2102.03554, 2021.
- [38] Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing data mixtures speeds up language model pretraining. In Proceedings of NeurIPS, 2024.
- [39] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. In Proceedings of NeurIPS, 2024.
- [40] Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text, 2023.
- [41] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110, 2021.

- [42] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024.
- [43] Yaoxiang Wang, Haoling Li, Xin Zhang, Jie Wu, Xiao Liu, Wenxiang Hu, Zhongxin Guo, Yangyu Huang, Ying Xin, Yujiu Yang, et al. Epicoder: Encompassing diversity and complexity in code generation. arXiv preprint arXiv:2501.04694, 2025.
- [44] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [45] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [46] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. OLMo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.
- [47] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of ACL, 2019.
- [48] Hector Levesque, Ernest Davis, and Leora Morgenstern. The winograd schema challenge. In Proceedings of KR, 2012.
- [49] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of ACL, 2016.
- [50] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of EMNLP, 2018.
- [51] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.
- [52] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of AAAI, 2020.
- [53] Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text (ACL 2017), 2017.
- [54] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of NAACL-HLT, 2019.
- [55] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.
- [56] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms, 2019.
- [57] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel HerbertVoss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.
- [58] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

- [59] Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts. In Proceedings EMNLP, 2022.
- [60] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of ICLR, 2019.
- [61] Joost CF De Winter, Samuel D Gosling, and Jeff Potter. Comparing the pearson and spearman correlation coefficients across distributions and sample sizes: A tutorial using simulations and empirical data. Psychological methods, 21(3):273, 2016.

Appendix: Data Efficacy for Language Model Training

- A Additional Experimental Setup

Evaluation. For general evaluation, we assess the trained models on a range of standard natural language understanding and reasoning benchmarks, including Hellaswag (HS; 47), Winogrande (Wino; 48), LAMBADA (LAMB; 49), OpenbookQA (OBQA; 50), ARC-easy/challenge (ARCe/c; 51), PIQA [52], SciQ [53], and BoolQ [54]. For domain-specific tasks, we assess the models on mathematical reasoning and code generation benchmarks. Specifically, for math, we use GPQA Diamond [55] and MathQA [56], while for code, we use HumanEval [57] and MBPP [58].

For the code generation benchmarks, we use a 0-shot setting for HumanEval and a 3-shot setting for MBPP. Code generation performance is reported using pass rate@1, which indicates the percentage of first-attempt solutions that pass all associated unit tests (pass@1). For the other benchmarks, the tasks are framed as multiple-choice questions, where the model selects the correct answer by minimizing the normalized loss across all candidate options (acc_norm).

Compute Resources. We train the 160M model on 1B-token datasets using a single NVIDIA A100 40GB GPU. For experiments with the 160M, 470M, and 1B models on 10B and 50B-token datasets, we utilize 8 NVIDIA A100 40GB GPUs. All data scoring steps, including proxy data annotation, scorer training, and data scoring, are performed on a single NVIDIA A100 80GB GPU.

Training Configuration. To create the data scorer, we fine-tune the Fairseq-Dense-125M model [59] on the solved data weights γ. As described in algorithm 1, we apply a linear transformation to the mean-pooled representations of instances along the sequence length. The hidden state size is set to 768. The optimization of algorithm 1 is performed using the AdamW optimizer [60] for 5 epochs, with a learning rate of 1 × 10−4 and a batch size of 512. We train on 90% of the samples and reserve the remaining 10% from Dprx as a validation set. The checkpoint with the highest Spearman correlation score [61, 10] on the validation set is selected to infer data quality scores in D.

For LMs training, all models are trained with a batch size of 256 and a maximum input sequence length of 1,024 for one epoch. The AdamW optimizer [60] is paired with a cosine learning rate scheduler. The scheduler includes a warm-up phase for the first 2,000 steps, after which the learning rate decays to 10% of its peak value. The model architecture and corresponding learning rates are summarized in Table 5, following the configurations in [10].

Model Size dmodel dFFN nlayers nhead dhead learning rate

160M 768 3,072 12 12 64 6 × 10−4 470M 1,024 4,096 24 16 64 3 × 10−4 1B 1,536 6,144 24 16 96 2.5 × 10−4 1.7B 2,048 8,192 24 16 128 2 × 10−4

Table 5: Model configurations and corresponding learning rates.

- B Societal Impact

Our work introduces DELT, the first data efficacy paradigm that enhances language model performance through strategic data ordering. By unifying data scoring, selection, and ordering, DELT improves both the efficacy and efficiency in LM training, including both pre-training and post-training. This foundational advancement enables performance improvements without altering model size or dataset scale, offering substantial positive societal impacts for the development of general AI.

Meanwhile, DELT can reduce the computational resources and energy consumption required for training language models, contributing to sustainability in AI development. Furthermore, the enhanced efficiency increases accessibility, enabling researchers and practitioners with limited resources to leverage advanced language models. By fostering the development of more robust and reliable models, this work can significantly benefit various societal domains, such as education and healthcare.

## C Limitations and Future Work

The proposed paradigm has two primary limitations. 1) The current verification is specifically focused on language models, with no evaluation in other modalities like image and audio, which depend on different scoring implementations. 2) Similar to PDS data scoring, the implementation of our designed LQS method requires calculating the downstream loss J(θ) on a high quality and small scale dataset. In the future, we plan to scale up our method on larger models (e.g., tens or hundreds of billions of parameters) and larger datasets (terabyte level). Additionally, we aim to explore simpler and more effective data scoring methods and extend this paradigm to multimodal models.

## D LQS Explanation

We provide a more detailed explanation of the Learnability-Quality Scoring (LQS). As described in Section 4.1.2, the Learnability Score L(xn) (Equation 6) and the Quality Score Q(xn) (Equation 7) are proposed, both of which are directly related to the training sample xn and are used as metrics to evaluate xn.

However, the reliability of these scores is directly influenced by the capability of the model checkpoint. Models with stronger capabilities produce more reliable scores. This capability is linked to the target vector, which is calculated as an average across all samples. When the structural parameters of the model remain unchanged, a stronger model shows greater consistency in the gradient directions of all samples concerning the model parameters, resulting in a larger ∇J(θt) (Equation 8). This corresponds to a greater magnitude of the target vector. Therefore, we use the magnitude of the target vector to directly measure the model’s capability, referred to as the reliability score in Equation 11.

R(θt+1) = ∥λt+1∥ (11)

Finally, we anticipate that a stronger model will assign more weight to the scores. LQS is expressed as a combination of the sample’s learnability-quality score and the model’s capabilities.

γn = R(θt+1) · Q(xn) · L(xn) (12)

T−1

λt+1⊤∇ℓ(xn,θt) ∥λt+1∥ · ∥∇ℓ(xn,θt)∥

∥∇ℓ(xn,θt)∥ ∥∇ℓ(xn,θt+1)∥

∥λt+1∥ ·

·

=

t=1

T−1

λt+1⊤∇ℓ(xn,θt) ∥λt+1∥ · ∥∇ℓ(xn,θt+1)∥

(13)

∥λt+1∥ ·

=

t=1

T−1

λt+1⊤∇ℓ(xn,θt) ∥∇ℓ(xn,θt+1)∥

(14)

=

t=1

## E LQS Implementation

Similar to [10], our data scoring method follows these steps: 1) Proxy data sampling. A proxy dataset Dprx is first uniformly sampled from the pre-training corpus D, serving as a representative subset of the larger corpus. 2) Proxy data annotation. We apply Eq. 9 to compute data scores for each instance in Dprx, obtaining a set of data samples with scores as ground truth (see Section E.1).

- 3) Data scorer training. The data scorer, typically a small LM, is fine-tuned on the automatically annotated data samples in Dprx to predict data scores effectively. (see Section E.2) 4) Full data scoring. The trained data scorer is then applied to infer scores for the entire pre-training corpus D.

### E.1 Proxy Data Annotation

To construct the ground truth scores γ∗ for Dprx, we are inspired by [10] and adopt a bi-level optimization framework (see Algorithm 1) that quantifies the contribution of each data point in Dprx to the downstream performance. The goal is to determine an optimal score vector γ∗ = [γ1∗,γ2∗,··· ,γ|D|∗ ]⊤.

The bi-level optimization consists of two nested loops: 1) a forward loop that simulates the model training process, and 2) a reverse loop that adjusts the scores γ based on the model parameters at each training step. Specifically, in the forward loop, the model is trained for T steps using gradient descent, where the training loss is weighted by the current scores γ∗. This process updates the model parameters θt iteratively, producing a trajectory of checkpoints θt from t = 0 to t = T − 1. In the reverse loop, the target vector λ∗t is computed through the training trajectory from t = T −1 to t = 0 according to Eq. 8. where λ∗t represents the backward-propagated gradient at step t, ∇l(xn,θt) is the gradient of the loss for the n-th data point. After each update, the scores are projected onto the probability simplex to ensure they remain valid probabilities Proj[γ∗].

Algorithm 1 Proxy Data Annotation

Input: LM learning rate η. Proxy data Dprx. Downstream loss J(θ). Training steps T. Proj[·] that projects a

point in R|D| to U. Model initialization θ0. Output: Data quality scores γ∗.

γ∗ = γ1∗, γ2∗, · · · , γ|D|∗ ← |Dprx 1 |, |Dprx1 |, · · · , |Dprx1 | ; for t = 0, 1, · · · , T − 1 do ▷ Forward loop

θt+1 ← θt − η∇L(θt, γ) end for λT ← ∇J(θT) for t = T − 1, T − 2, · · · , 1 do ▷ Reverse loop

λ∗t ← λ∗t+1 + ∇J(θt) − η∇2L(θt, γ∗)λ∗t+1 ▷ Eq. (8) end for for n = 1, 2, · · · , |D| do

∗ t+1

⊤∇l(xn,θt)

γn∗ ← γn∗ + α Tt=1−1 λ

∥∇l(xn,θt+1)∥ ▷ Eq. (9)

end for γ∗ ← Proj [γ∗] return γ∗

### E.2 Data Scorer Training

After we obtain the scores γ∗ for Dprx, we then train a small LM, initialized from a pre-trained checkpoint, with a linear head to serve as the data scorer. The scorer is optimized on the proxy dataset Dprx to fit the ground-truth scores γ from Section E.1. Specifically, each instance xprxn ∈ Dprx is encoded by averaging the LM’s output hidden states along the sequence, producing a feature representation h(xprxn ,ϕ) ∈ Rd, where ϕ are the LM parameters and d is the hidden state size. This representation is passed through a linear head, with parameters w ∈ Rd and b ∈ R, to produce a predicted score. The parameters of the LM and linear head are optimized together using the Mean Squared Error (MSE) loss:

|Dprx|

1 |Dprx|

w⊤h(xprxn ,ϕ) + b − γn∗ 2 , (15)

LMSE =

n=1

The optimal parameters ϕ∗, w∗, and b∗ are obtained by minimizing this loss. Once trained, the data scorer predicts scores for xn ∈ D as γ(xn) = w∗⊤h(xn,ϕ∗) + b∗. This process enables the data scorer to generalize from Dprx to the larger pre-training corpus D effectively.

## F Samples Scored by LQS

To better illustrate our advantages, we visualized part samples. Specifically, we randomly sampled examples from the top 10% and bottom 10% based on their scores for visualization.

As shown in Table 6 and 7, high-scoring samples are high-quality, complex sentences that are not only challenging but also highly learnable, significantly aiding the model’s optimization in later stages. In contrast, as shown in Table 8, low-scoring samples are often noisy and provide little to no benefit for model training, making them ideal candidates for filtering (see Table 8 example 1 and 2). However, some low-scoring samples consist of simple words or short phrases (see Table 8 example 3 and 4), which are beneficial for the model’s learning in the early stages.

- # Example 1

<s> The precise nature of his doubt is not articulated but he aligns himself with the 17th century dissenters who put personal religious conscience before the unifying demands of the Church. This precipitates the move to the North, and determines the whole narrative arc of the novel. Before we are thrust, along with our central characters, into the smoky North, it is worth lingering a while on this short first stage of the novel. Gaskell does not make a simple thing of the South, as she might have been tempted to do as a clear point of comparison. Instead she offers us two versions of the South, two kinds of imaginings, both of which are then rejected. The drawing-room world of the Shaws, while superficially appealing, is altogether too enervating for the Margaret Hale who is gradually emerging even in these early chapters. Her decided refusal of Lennox is also a refusal of that world. The rural delights of Helstone (in the New Forest) seem initially to offer a simpler, perhaps a truer, version of the South. But that has already been put in doubt by Margaret and Henry Lennox’s rather vexed discussion of it. In London, Henry suggests playfully that it is a ‘village in a tale’, at which Margaret takes umbrage, only to offer instead that it is ‘a village in a poem’ (11). When Henry arrives bang in the middle of that poem, the scene is set for romance: ‘velvety cramoisy roses’ (25), pears plucked from the tree and arranged on a plate of beetroot leaf, and the ‘crimson and amber foliage’ (26) of the deep forest beyond. Yet instead of the completion of the romantic dream, he comes up hard against Margaret’s refusal. Indeed she herself comes up hard against it, and looks back at his proposal somewhat wistfully when she is plunged into her father’s ferment, and briefly longs for the London/Shaw world where nothing ‘called for much decision’. But if Margaret had accepted Henry – no novel. And besides, he can be kept in mind as a possible future plot line. Gaskell is astute enough to know that challenge makes for a more interesting narrative, and as it turns out, decision is something Margaret is rather good at. In the move to the South, she becomes the adult of the household, her mother declining into frailty, her father exhausted by the consequences of his own conscience. Her growth into her own strength of being is the more convincing because she often quails at what is before her. ‘But the future must be met, however stern and iron it be.’ (55) Thus, after a brief lull and taking of rest at the seaside town of Heston, Margaret and her father make the journey to Milton – the ‘North’ of the novel – where, as she says playfully, ‘“I am overpowered by the discovery of my own genius for management.”’ (57) But the obstacles are real, and the whole family must contend with their much-changed situation: They were settled in Milton, and must endure smoke and fogs for a season; indeed all other life seemed shut out from them by as thick a fog of circumstance.

... At night when Margaret realised this, she felt inclined to sit down in a stupor of despair. The heavy smoky air hung about her bedroom, which occupied the long narrow projection at the back of the house. The window, placed at the side of the oblong, looked to the blank wall of a similar projection, not above ten feet distant. It loomed through the fog like a great barrier to hope. (62) We are almost in the world of George Orwell’s The Road to Wigan Pier – a work in fact heavily influenced by nineteenth-century depictions of urban industrialised living conditions. As if to underline their changed life, a letter has come from Edith, full of the delights of married life in Corfu: ‘Edith’s life seemed like the deep vault of blue sky above her, free – utterly free from fleck or cloud’. (62) This leads Margaret to reflect in turn on how, if she had accepted Lennox’s marriage proposal, things might have been different. The omniscient narrative is here able to give much insight into Margaret’s inner thoughts, so that we see her working through these difficult ideas and eventually finding herself clearer and happier: ‘As she realised what might have been, she grew to be thankful for what was.’ (63) If there is a measure of rationalisation in Margaret’s logic here – that too is realistic.

- # Example 2

<s> For example, signals 116 and 118 may be in-phase (I) and quadrature (Q) baseband components of a signal. In the example of FIG. 1B, signals 116 and 118 undergo a zero crossing as they transition from +1 to -1. Signals 116 and 118 are multiplied by signal 120 or signal 120 phase shifted by 90 degrees. Signal 116 is multiplied by a 0 degree shifted version of signal 120. Signal 118 is multiplied by a 90 degree shifted version of signal 120. Resulting signals 122 and 124 represent time-varying complex carrier signals. Note that signals 122 and 124 have envelopes that vary according to the time-varying amplitudes of signals 116 and 118. Further, signals 122 and 124 both undergo phase reversals at the zero crossings of signals 116 and 118. Signals 122 and 124 are summed to result in signal 126. Signal 126 represents a time-varying complex signal. Signal 126 may represent an example input signal into VPA embodiments of the present invention. Additionally, signals 116 and 118 may represent example input signals into VPA embodiments of the present invention. 1.2) Example Generation of Time-Varying Complex Envelope Signals from Constant Envelope Signals The description in this section generally relates to the operation of step 508 in FIG. 50. FIG. 1C illustrates three examples for the generation of time-varying complex signals from the sum of two or more substantially constant envelope signals. A person skilled in the art will appreciate, however, based on the teachings provided herein that the concepts illustrated in the examples of FIG. 1C can be similarly extended to the case of more than two constant envelope signals. In example 1 of FIG. 1C, constant envelope signals 132 and 134 are input into phase controller 130. Phase controller 130 manipulates phase components of signals 132 and 134 to generate signals 136 and 138, respectively. Signals 136 and 138 represent substantially constant envelope signals, and are summed to generate signal 140. The phasor representation in FIG. 1C, associated with example 1 illustrates signals 136 and 138 as phasors P136 and P138, respectively. Signal 140 is illustrated as phasor P140. In example 1, P136 and P138 are symmetrically phase shifted by an angle ϕ1 relative to a reference signal assumed to be aligned with the real axis of the phasor representation. Correspondingly, time domain signals 136 and 138 are phase shifted in equal amounts but opposite directions relative to the reference signal. Accordingly, P140, which is the sum of P136 and P138, is in-phase with the reference signal. In example 2 of FIG. 1C, substantially constant envelope signals 132 and 134 are input into phase controller 130. Phase controller 130 manipulates phase components of signals 132 and 134 to generate signals 142 and 144, respectively. Signals 142 and 144 are substantially constant envelope signals, and are summed to generate signal 150. The phasor representation associated with example 2 illustrates signals 142 and 144 as phasors P142 and P144, respectively. Signal 150 is illustrated as phasor P150. In example 2, P142 and P144 are symmetrically phase shifted relative to a reference signal. Accordingly, similar to P140, P150 is also in-phase with the reference signal. P142 and P144, however, are phase shifted by an angle whereby ϕ2 ̸= ϕ1 relative to the reference signal. P150, as a result, has a different magnitude than P140 of example 1.

Table 6: Examples of Randomly Sampled High-Scoring Data Points (Top 10%).

- # Example 3

<s> Then Eddie owned up and said I took the records and my mummy said what did you do with them Eddie and he said I played cards with them that’s what I done with them where as everybody was playing cards for tuppence and thrupence and he was playing with the records and our Tommy stood and looked at him, I have never forgotten the expression on Tommy’ s face. Eddie was about 15 or 16 then. When Eddie left school he successfully applied for a job in the Belfast City Council and I remember everybody being very proud because it was difficult to get a job with the Council back then. My mummy came in one day and said I was talking to the foreman about our Eddie and he said he’s a great worker, my mummy was very proud of him. He used to land in for his lunch to my mummy with all the other binmen, she would have to feed them all. He stayed there until 1968 when he began working at boarding up buildings that had been damaged in the Troubles. As a way to earn an extra bit of money for the family he also worked nights as a barman. When Eddie got older he was always very particular about his appearance, he always wore a suit, sometimes with shirt or tee shirt, he was always very spick and span. Eddie smoked but he wasn’t a drinker. That’s not to say that he didn’t try it at the beginning but it wasn’t for him, he became a lifelong pioneer and a blood donor. He was also in the Confraternity (which was a sort of prayer group for men) at Clonard Monastery and he loved it; that was his wee place to get away to. He had strong faith. When Eddie was sixteen he met and fell in love with his future wife Marie. Marie was the love of his life and they courted for six years before getting married in 1962. Five years later their first child was born, quickly followed by three more. Eddie and Marie had three sons – Eamon, Patrick and Ciaran, and one daughter – Brenda. When they got married they went to live with Marie’s grandmother in Fort Street. But he wanted his own house for himself and Marie and the only way that was going to happen was to get the money together to buy one. My daddy said to him. “look if you’re looking extra money to buy a house go and join the TA its only 2 months a year”. So he went and joined the Territorial Army and the money he was getting he sent it home to Marie. He wasn’t in the TA for very long and had left by the time his first child was born. When Eddie came home things didn’t work out the way he wanted about the house and things got too much for him, he ended up with a bit of a breakdown. Eventually they got the money for a house in Iveagh Street it was in a bad state of repair but Eddie and Marie fixed it up and made it their home. He suffered with mental health difficulties a couple of times in the early-mid 1960s but that was well behind him by the time of his death. Eddie just lived for Marie and their kids, he took on a couple of extra jobs, working as a bar man and doing a bit of painting and decorating. He was always ready and willing to drop everything and do something for you. It was just an ordinary family life and he just loved Marie, he idolised her and she could do no wrong in his eyes. All he had time for was work, home and the confraternity. When he did have free time he liked fishing and clay pigeon shooting. He was content with what he had and he was in his own wee orbit that he owned his house, and provided for his kids he was just happy to be a husband and a daddy. On payday he would give Marie his unopened pay packet, she would then buy him his cigarettes for the week. Not too many men did that in those days. Eddie was strict in a way too with the kids, I remember Eddie coming to visit me with Ciaran, I had a rocking horse in the living room and Ciaran wanted on it and Eddie said no you’re not going over it doesn’t belong to you, and I looked at Eddie and said let the child go over and get on to the horse I said catch yourself on there is nobody even on it and he went over but he was holding him on it because he maybe would of toppled.

- # Example 4

<s> I was interested to see if I would lean closer to earlier poems or later poems since sometimes there can be a significant difference in a poets writing style compared to when they began and ended. Turns ou This.Was.My.Jam Where do I even begin? So the collection is written in reverse chronological- yeah that’s right I actually read the introduction to something. I found this particularly interesting because I feel like we often start in the beginning and naturally work our way through their work. I was interested to see if I would lean closer to earlier poems or later poems since sometimes there can be a significant difference in a poets writing style compared to when they began and ended. Turns out I pretty steadily loved it all. I think if I HAD to chose I would lean just slightly closer to the beginning of the collection, but just slightly. That might have a bit of a biases though since Annabelle Leigh is the very first poem we read and it’s always been my absolute favorite. Annabelle Leigh aside, I can only imagine what other wonderfully powerful and hauntingly beautiful pieces he could have continued to write had he lived longer. (Internally sobbing) You’ll probably notice that there’s a lot of reoccurrence with things like the moon, celestial bodies, night, and the evening star- all things I really enjoyed. Also, (and this might quite well be my favorite) Poe has some of the best rhymes. Words that rhymed but weren’t your usual rhymes, if you will. For example: departed and brokenhearted, month of June and mystic-moon, dipt in folly and melancholy, Heaven and unforgiven (you gotta twang a little for that one), itself alone and gray tombstone, heart’s content and own element. etc. And it doesn’t stop there! The the entire language being used is SO GOOD. I’d be reading a poem and then I’d hit a particular line or phrase and just have to a take a moment to say "damn" while the words were absorbed. Some examples of that are "And the silken, sad, uncertain rustling of each purple curtain" (The Raven), I stand amid the roar of a surf-tormented shore (A Dream Within a Dream), With the moon-tints of purple and pearl (Eulalie-A Song), Sound loves to revel in a summer night: Witness the murmur of the gray twilight. (Al Aaraaf Part 2) and "So like you gather in your breath, a portrait taken after death. (Tamberlane) Even the poems that I didn’t mark as favorites I still really enjoyed. My least favorite in the collect was Al Aaraaf (both parts), I’m not really sure why I just didn’t feel as wow’ed by it. Also, the play that ends the collection I wasn’t a huge fan of but I think that just speaks true to the format. Plays are different than poetry. I haven’t read any Poe stories for a long time, so I think it would be interesting to see where my enjoyment falls on the prose. But, the poetry is definitely out of the park for me. Something I do find intriguing is that growing up I also thought Poe was just a dark and haunted poet. I think he was in fact haunted, but I don’t think (the poetry at least) is as horrific as people usually indicate. In fact, I’m willing to call it beautiful. Beautifully dark, perhaps? Read it, it’s perfect. Mateo – Oct 24, 2020 I did not make this image but this is my review I did not make this image but this is my review Stephanie Grosse – Sep 23, 2018 This review has been hidden because it contains spoilers. To view it, click here. Simultaneously mysterious and familiar, like the old friend who suddenly astonishes you with his strangeness or the acquaintance whom you are convinced you must have known since childhood. I very much enjoyed the use of onomatopoeia. You will be hypnotised by the sounds (for example "ee", "em" in the summer dream beneath the tamarind tree). Poe has you forever, in "a dream within a dream" Very memorable.A must read for all poetry lovers. Simultaneously mysterious and familiar, like the old friend who suddenly astonishes you with his strangeness or the acquaintance whom you are convinced you must have known since childhood. I very much enjoyed the use of onomatopoeia.

Table 7: Examples of Randomly Sampled High-Scoring Data Points (Top 10%).

#### Examples of Randomly Sampled Low-Scoring Data Points (Bottom 10%)

- # Example 1

<s>XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Select a country but NOT a region All you need to know about the Indian Defence Forces!

- # Example 2

<s>XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Select a country but NOT a region A site focusing on Australian Modelling with galleries, articles and discussion forums.

- # Example 3

<s>USA, Liberia USA, Lithuania, Italy USA, Luxembourg USA, Luxembourg, UK USA, Malaysia USA, Malta, France, UK USA, Malta, UK USA, Mexico USA, Mexico, Australia USA, Mexico, Australia, Canada USA, Mexico, Canada USA, Mexico, Canada, Germany USA, Mexico, Germany USA, Mexico, Hong Kong USA, Mexico, Japan USA, Mexico, Spain USA, Mexico, UK USA, Mexico, United Arab Emirates USA, Monaco USA, Monaco, Morocco USA, Morocco USA, Morocco, Spain, UK USA, Morocco, Switzerland USA, Myanmar USA, Netherlands USA, Netherlands, France USA, Netherlands, Germany, France, Austria USA, Netherlands, South Africa USA, Netherlands, UK USA, Netherlands, UK, Denmark USA, New Zealand USA, New Zealand, Canada, Israel, Japan, Nigeria USA, New Zealand, Germany USA, New Zealand, Japan USA, New Zealand, South Africa, UK, Lithuania USA, New Zealand, UK USA, Nicaragua USA, Nigeria USA, Norway USA, Pakistan USA, Panama, Argentina USA, Panama, Japan, Canada USA, Panama, Mexico USA, Peru USA, Philippines USA, Philippines, Puerto Rico USA, Philippines, Taiwan, South Korea, China, Canada USA, Poland USA, Poland, Slovenia, Czech Republic, UK USA, Portugal USA, Portugal, France USA, Puerto Rico USA, Qatar USA, Romania USA, Romania, Canada USA, Romania, France, Italy, Germany USA, Romania, Germany USA, Romania, Iceland USA, Romania, UK USA, Russia USA, Russia, Hungary USA, Russia, UK USA, Saudi Arabia USA, Senegal USA, Serbia USA, Serbia, Canada USA, Singapore USA, Singapore, Taiwan USA, Slovakia USA, Slovakia, China USA, South Africa USA, South Africa, Germany USA, South Africa, India USA, South Africa, Italy USA, South Africa, Zambia, Germany USA, South Korea USA, South Korea, Australia USA, South Korea, India USA, South Korea, Japan USA, South Korea, Singapore USA, South Korea, Singapore, Russia, Malaysia, Kazakhstan, Taiwan, Hong Kong, Japan, China, India, Syria, Iran, Egypt, Pakistan USA, South Korea, Spain

- # Example 4

Phillip L. Horrell v. David Gomez, Warden, No. 20-5306 Ganaa Otgoo v. Illinois, No. 20-5109 Phillip Hartsfield v. Stepanie Dorethy, Warden, No. 19-1473 Anthony Jackson v. Supreme Court of Illinois, No. 19-8665 David Beverly v. Illinois, No. 19-8502 Lamont Dantzler v. Illinois, No. 19-8448 Joh-ner Taylor Wilson v. Illinois, No. 19-8437 Herbert Burgess v. Illinois, No. 19-8379 Joseph M. Coffman v. Illinois, No. 19-8391 Timothy J. McVay v. Illinois, No. 19-8304 Kenneth Durant v. Frank Lawrence, Warden, No. 19-7967 Seth A. Weaver v. Illinois, No. 19-7823 Lyarron T. Emers v. Illinois, No. 19-7759 Bethany Austin v. Illinois, No. 19-1029 Anthony Allen v. Illinois, No. 19-7633 Kenin L. Edwards v. Michael L. Atterberry, et al., No. 19-965 Pablo Rodriguez-Palomino v. Illinois, No. 19-7273 Christopher L. Croom v. Illinois, No. 19-7237 Tony Robinson v. Illinois, No. 19-7226 Lazaro Zapata v. Illinois, No. 19-7264 Fernando Oliveros v. Illinois, No. 19-7141 Kevin Dameron v. Illinois, No. 19-6945 Peter Gakuba v. Michelle Neese, No. 19-6543 Richard Kalinowski v. Illinois, No. 19-6368 Rafael Alvarado v. Frank Lawrence, Warden, No. 19-6347 Chad M. Cutler v. Illinois, No. 19-6150 Hezekiah Whitfield v. Deanna Brookhart, Warden, No. 19-6051 Lorenzo Davis, Jr. v. Illinois, No. 19-5831 Chadwick N. Barner v. Illinois, No. 19-5655 Charles Donelson v. Q. Tanner, et al., No. 19-5397 Robert Curry v. Illinois, No. 19-5366 Andrew Condon v. Illinois, No. 19-5349 Keith Talbert v. Illinois, No. 18-9768 Juan Rodriguez v. Illinois, No. 18-9759 Miguel Alcantar v. Illinois, No. 18-1548 Gregory Rayford v. Illinois, No. 18-9612 Irving Madden v. Michael Melvin, Warden, No. 18-9474 Denzel Pittman v. Illinois, No. 18-9451 Jesus Cotto v. Jacqueline Lashbrook, Warden, No. 18-9116 Pierre Montanez v. Ursula Walowski, No. 18-9101 Russell Frey v. Illinois, No. 18-9120 Peter Gakuba v. Illinois, No. 18-9041 Jose Cobian v. Illinois, No. 18-8963 Derrick Redmond v. Illinois, No. 18-8808 Jennifer N. Nere v. Illinois, No. 18-8625 Gerald W. Long v. Illinois, No. 18-8577 Willie White v.

Table 8: Examples of Randomly Sampled Low-Scoring Data Points (Bottom 10%).

## G More Experiments Results

Data Efficacy across Different Model Sizes and Data Scales on More Methods. (Table 9). To supplement Table 1, we provide additional experimental results comparing DELT with data selection methods across different model sizes and data scales. As shown in Table 9, with increasing model parameters and data scales, our proposed DELT demonstrates consistent improvements and outperforms various methods across most benchmarks.

- Table 9: Efficacy results on different downstream benchmarks. Ours means applying LQS for data scoring and Folding for data ordering within the DELT paradigm.

(a) Results (%) for 1B-token data across model sizes (160M, 470M, 1B).

ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. Model size = 160M

Conventional 21.27 34.32 27.85 20.25 24.40 55.19 56.93 50.72 36.37

KenLM 21.93 33.96 28.09 20.69 25.20 54.79 56.20 50.59 36.43 PDS 21.84 35.02 27.61 19.93 24.80 56.23 59.00 51.38 37.01 Ours 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08

Model size = 470M

- Conventional 21.16 34.91 28.11 21.88 23.90 56.07 58.75 50.04 36.85 KenLM 22.35 34.85 28.05 20.51 25.00 55.17 56.60 50.04 36.57

PDS 22.10 33.04 27.84 21.25 24.80 56.96 59.80 51.85 37.23 Ours 22.33 35.88 28.45 23.26 26.60 57.20 60.10 52.81 38.33

Model size = 1B Conventional 20.58 36.12 28.32 23.56 25.00 56.49 60.05 52.07 37.77

KenLM 21.67 35.86 28.76 23.46 26.80 56.58 59.00 49.88 37.75 PDS 22.10 35.56 28.20 23.56 26.40 56.37 60.50 50.67 37.92 Ours 22.76 37.95 29.95 26.38 26.00 58.07 60.90 51.28 39.17

(b) Results (%) for 160M model across data sizes (10B, 50B).

ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. Data size = 10B tokens

- Conventional 22.82 38.51 30.72 30.40 25.70 57.32 64.90 51.54 40.24 KenLM 22.78 37.92 30.54 29.98 25.60 57.29 66.00 52.80 40.36

PDS 22.70 39.35 30.73 31.85 27.20 56.04 64.90 52.88 40.71 Ours 24.38 39.80 31.64 32.98 27.21 58.56 66.70 51.67 41.62

Data size = 50B tokens Conventional 24.06 41.88 32.05 33.79 26.80 58.11 69.00 51.93 42.20

KenLM 23.74 40.14 32.10 35.13 28.41 58.15 67.52 51.71 42.11 PDS 24.57 41.37 32.44 35.36 29.20 59.25 68.10 50.83 42.64 Ours 24.65 41.07 33.00 36.07 29.30 59.10 68.40 52.67 43.03

Comparison of Data Efficiency (Table 10). To supplement Table 2, we further highlight the superiority of the DELT framework by comparing it with the data efficiency setting. Table 10 presents results for data processed through the efficiency (data selection) and efficacy (DELT) pipelines. Compared to the results from the conventional pipeline and the best-performing data efficiency pipeline (Selection ✓, Ordering -), the DELT framework, which incorporates both data selection and ordering (Selection ✓, Ordering ✓), consistently demonstrates significant improvements across various method combinations.

- Table 10: Efficiency results of different methods. The selection methods report the highest scores across all selection ratios. The best scores for each model size are highlighted in bold, while the second-best scores are shown in italic bold.

Pipeline Scoring Selection Ordering ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. Conventional - - - 21.27 34.32 27.85 20.25 24.40 55.19 56.93 50.72 36.37

KenLM[19] ✓ - 21.42 34.34 27.76 20.84 25.00 56.31 54.30 51.07 36.38 PDS[10] ✓ - 21.84 35.02 27.61 19.93 24.80 56.23 59.00 51.38 37.01 LQS (Ours) ✓ - 22.18 34.09 27.80 21.02 25.20 55.98 59.00 51.85 37.14

Efficiency

###### DELT LQS (Ours) ✓ Folding 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08

Stability on Different Epochs. (Table 11) To supplement Figure 6, we further report the detailed results of the proposed DELT across different epochs on various benchmarks. As shown in Table 11, with an increasing number of epochs, our method demonstrates stable improvements across most benchmarks, further highlighting its robustness and generalizability.

Table 11: Results on OLMo for the different epochs.

Epoch ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg.

- 1

Conventional 21.27 34.32 27.85 20.25 24.40 55.19 56.93 50.72 36.37

- DELT (Ours) 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08

2

Conventional 21.93 36.20 29.18 25.93 23.00 56.86 61.20 50.99 38.16

- DELT (Ours) 22.35 36.41 28.28 27.63 26.80 56.47 61.00 51.22 38.77

- 3

Conventional 21.35 35.78 28.76 27.14 26.20 56.51 62.80 49.51 38.51 DELT (Ours) 22.44 36.95 29.41 29.09 24.80 56.20 62.30 51.62 39.10

- 4

Conventional 21.10 35.99 28.97 27.51 27.20 55.69 61.80 49.28 38.44 DELT (Ours) 22.53 38.05 29.78 29.58 26.40 57.34 63.90 51.85 39.93

- 5

Conventional 20.59 37.55 29.31 28.05 27.00 57.11 61.20 50.54 38.92 DELT (Ours) 22.87 38.05 30.01 30.08 26.80 58.16 64.10 49.80 39.98

Details for L in Folding Learning (Table 12). To supplement Figure 7, we provide detailed results for the proposed Folding Learning method with varying values of the parameter L. As shown in Table 12, model performance reaches its peak at L = 3 and demonstrates significant advantages across most benchmarks. Notably, compared to traditional Curriculum Learning (L = 1), our proposed method (L > 1) achieves substantially better performance on all benchmarks.

Table 12: Effect of the fold layer L. L = − represents the conventional method, which is three times the random average results. When L = 1, the ordering method reduces to curriculum learning.

L ARC-c ARC-e HS LAMB OBQA PIQA SciQ Wino Avg. - 21.27 34.32 27.85 20.25 24.40 55.19 56.93 50.72 36.37

- 1 22.18 35.40 28.01 23.48 23.80 55.60 56.80 51.07 37.04

- 2 21.57 34.26 28.34 23.29 25.80 55.88 58.70 49.80 37.21

- 3 21.59 36.07 28.41 23.79 25.60 56.37 59.80 53.04 38.08

- 4 22.83 34.98 28.50 22.35 24.90 56.67 59.80 50.10 37.52

- 5 22.91 35.57 28.16 22.85 26.70 55.41 57.30 52.08 37.62

