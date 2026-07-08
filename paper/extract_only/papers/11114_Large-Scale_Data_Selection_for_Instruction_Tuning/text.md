arXiv:2503.01807v2[cs.CL]19Jun2025

# Large-Scale Data Selection for Instruction Tuning

## Hamish Ivison1,2, Muru Zhang1,3, Faeze Brahman2, Pang Wei Koh1,2, Pradeep Dasigi2 1University of Washington, 2Allen Institute for AI, 3USC

hamishiv@cs.washington.edu

## Abstract

Selecting high-quality training data from a larger pool is a crucial step when instructiontuning language models, as carefully curated datasets often produce models that outperform those trained on much larger, noisier datasets. Automated data selection approaches for instruction-tuning are typically tested by selecting small datasets (∼10k samples) from small pools (∼200k samples). However, popular deployed instruction-tuned models often train on hundreds of thousands to millions of samples, subsampled from even larger data pools. We present a systematic study of how well data selection methods scale to these settings, selecting up to 2.5M samples from pools of up to 5.8M samples and evaluating across 7 diverse tasks. We show that many recently proposed methods fall short of random selection in this setting (while using more compute), and even decline in performance when given access to larger pools of data to select over. We find that a variant of representation-based data selection (RDS+), which uses weighted mean pooling of pretrained LM hidden states, consistently outperforms more complex methods across all settings tested – all whilst being more compute-efficient. Our findings highlight that the scaling properties of proposed automated selection methods should be more closely examined.

## 1 Introduction

Instruction tuning has quickly become a crucial element of building modern language model (LM) systems, and a step for which curating high-quality data is especially important (Zhou et al., 2023; Wang et al., 2023). Finetuning on as few as 1,000 carefully chosen samples can yield models more preferred by human users than models trained on noisier datasets 10 times larger (Zhou et al., 2023).

The importance of data quality has led to a surge of papers proposing and investigating methods for selecting small (1–10k) datasets that outperform

50

###### RDS+

AveragePerformance

45

Embed (GTR)

LESS

Embed (NV)

Random (bal.)

Mid-PPL

40

Random (unbal.)

IFD

35

Top-PPL

30

1019 1020 Estimated Overall FLOPs Cost

Figure 1: Performance against estimated compute cost of varied data selection methods when selecting 10k points from data pools consisting of 200k (left points) and 5.8M (right points) data points in the single-task setup described in §4.1. We do not run LESS with 5.8M samples due to its high compute cost. Most data selection methods do not improve in performance with a larger pool, with the exception of RDS+ and Embed (GTR). We shade the Pareto frontier of efficiency and performance in red.

larger (50-200k) ones (Chen et al., 2024; Xia et al., 2024; Li et al., 2024b; Liu et al., 2024). However, the sizes of the selected datasets are often significantly smaller than those of commonly used open post-training datasets, which typically contain hundreds of thousands to millions of examples(Ivison et al., 2023b; Lambert et al., 2024; Llama Team, 2024; DeepSeek-AI, 2025). As such, it is unclear how well these data selection approaches scale as both the number of the selected data points and size of the data pool grow. Additionally, prior work suggests that common data selection approaches underperform random selection when considering the additional compute cost incurred by running the data selection techniques themselves even in smallerscale settings (Yin and Rush, 2024). It is unclear how performance and compute costs further change as we further scale beyond smaller-scale (selecting 10k data points from 200k samples) settings. In this work, we aim to answer these questions by inves-

tigating how well data selection methods work for selecting instruction-tuning data when used to select larger datasets (100s of thousands of samples) from even larger data pools (millions of samples).

We construct data pools consisting of all data considered for TÜLU 2 (Ivison et al., 2023b) and TÜLU 3 (Lambert et al., 2024), state-of-the-art instruction-tuned models with publicly released instruction tuning data. These data pools are large, diverse, unbalanced, and of mixed quality, requiring automated data selection techniques that can scale to millions of samples, filter out noise, and maintain enough diversity to ensure effective generalization. This allows us to examine how well data selection techniques scale up to selecting millions of samples from pools up to 6M data points in size. We select for and evaluate on seven diverse evaluations, covering a diverse range of skills from code generation (HumanEval) to general chat (AlpacaEval). We additionally investigate selecting a single dataset for multiple tasks, reflecting how instruction-tuned models are often used to perform multiple tasks at once.

We test nine different data selection approaches in our setting, covering a variety of different approaches to data selection: gradient-based influence methods (Xia et al., 2024), embedding-based methods (Zhang et al., 2018; Hanawa et al., 2021; Ivison et al., 2023a), and loss/perplexity-based methods (Yin and Rush, 2024; Antonello et al., 2021; Marion et al., 2023; Ankner et al., 2024; Li et al., 2024b). Surprisingly (and contrary to prior work), we find that a simple embedding method we call RDS+ — using the hidden states of a pretrained LM — works best overall across all settings tested, consistently beating all other approaches. In particular, we find that:

- 1. Many dataset selection methods struggle with increased dataset size. 4 of the 7 methods we examine drop in performance as we increase the data pool they select over, even when selecting datasets of the same size. In contrast, RDS+ not only improves as the data pool size grows, but also outperforms all other methods by at least 2 points on average. This highlights that examining the scaling properties of data selection is important for identifying strong methods (§4.1).
- 2. Similarly, when selecting for multiple tasks at once, only 3 out of 7 selection methods

beat random selection, with RDS+ performing best. We find that RDS+ even outperforms TÜLU 2, which was trained on a humancurated mixture (§4.2).

3. Controlling for the compute used during selection, RDS+ outperforms random selection by an average of two points when selecting hundreds of thousands to millions of samples, whilst using less compute. This stands in contrast to prior work (Xia et al., 2024), which does not scale up to large enough sizes to observe the point at which RDS becomes more compute-efficient. This highlights that data selection techniques may only become particularly useful for larger-scale settings than those typically used in prior work. (§4.3)

We believe our findings highlight the importance of examining data selection methods at larger scales than considered in prior work, and show that some such methods can still provide benefits at these scales.

## 2 Related Work

Data Selection For Instruction Tuning Earlier approaches to data selection for instruction tuning relied primarily on careful human curation (Zhou et al., 2023) or extensive ablation studies to determine optimal data mixtures (Wang et al., 2023). Recently, many automated data selection approaches have emerged, ranging from using GPT-based quality assessment (Chen et al., 2024; Wettig et al., 2024; Lu et al., 2023; Liu et al., 2024), embedding or perplexity features from model inference (Marion et al., 2023; Ivison et al., 2023a; Li et al., 2024b; Zhang et al., 2024c; Sachdeva et al., 2024; Li et al., 2024a; Du et al., 2023; Zhang et al., 2024b), gradient-based methods (Xia et al., 2024; Han et al., 2020; Zhang et al., 2024a; Yu et al., 2024), or shallower heuristics (Zhao et al., 2024a; Cao et al., 2024; Li et al., 2024d). These works typically focus on finding small subsets of larger datasets that can match or outperform using the entire dataset, often selecting subsets of roughly 10k samples in size (Chen et al., 2024; Xia et al., 2024; Li et al., 2024b) from datasets with 100s of thousands of samples. This does not match the known sizes of various instruction-tuning datasets used in practice, which range from 300k to millions of samples in size (Touvron et al., 2023; Llama Team, 2024; Ivison et al., 2023b; Lambert et al., 2024). In this

work, we aim specifically to study how these methods perform when selecting large datasets from large pools, and find that many proposed methods fall short of random selection in this setting.

Surveying Data Selection Methods Some prior studies have similarly studied how well automated data selection techniques work for instruction tuning: Yin and Rush (2024) examine how well various data selection approaches scale in computeconstrained scenarios. However, they use a relatively small data pool, only examine a single task setting, and assume only a single epoch of training (not common in instruction tuning). Diddee and Ippolito (2024) also examine how well data selection strategies generalize across different pools at various scales, but similarly only examine selecting relatively small datasets (up to 10k samples) and small pools (up to 200k in size). In contrast to both these works, we further scale the setting to use millions of samples, more evaluations, and more diverse datasets, and find that this changes our view of what method works best, with RDS outperforming random selection even in FLOPscontrolled scenarios.

Dai et al. (2024) examine how to balance influence scores when selecting multi-task data, and uses a selection algorithm similar to our proposed round-robin approach. However, we also examine how well multi-task selection works when selecting from significantly larger data pools, across multiple different pools and base models.

## 3 Data Selection Methods

We now describe a set of popular data selection methods commonly used and tested in prior work alongside the experimental setup we use to investigate them. We assume that one has a pool of data D and wishes to select up to n instances from the pool. Additionally, we assume that the user has some (10s to 100s) of query samples from a query set V . This query set is a small dataset that is in the same distribution as the evaluation set (e.g., we can use existing validation sets as query sets if available). We assume both V and D contain prompts and responses (i.e., both are labelled). In practice, this query split can either be task-specific (i.e., some query split of the downstream task we want to test on), or contain samples from a variety of tasks that do not overlap with downstream test tasks, but are reflective of our desired test distribution in some way. Each method either (a) takes

in a dataset D as input and produces a score for each point d ∈ D or (b) takes in a dataset D and query points v ∈ V and produces a score for each v,d ∈ V,D pair, which we then aggregate to compute a score for each d ∈ D (described further in §3.1). Given this, we explore the following methods. We pick a set of methods that represent simple but varied ways to select data: using model embeddings, using loss-based features (perplexity, IFD), using model gradients, and random selection. Our choices match common baselines used in prior work (Xia et al., 2024; Yin and Rush, 2024).

Random Selection. We explore random selection, a common strong baseline for data selection. We report both taking random samples of a given data pool, and taking ‘balanced’ random samples, where we uniformly sample from data sources. If we run out of data from a given source, we divide its remaining ‘budget’ equally among the dataset sources we have not yet exhausted.

Length. We sort examples by length (in tokens) and take the longest samples. This has been shown to be a strong baseline by Zhao et al. (2024b), and is computationally cheap, only requiring computing the length of each sample in our data pool.

Perplexity. We compute the loss of each d ∈ D on our original base model and use it as its score, following prior work (Yin and Rush, 2024; Antonello et al., 2021; Marion et al., 2023; Ankner et al., 2024). We use the same setup as Yin and Rush (2024) and examine both ‘mid-ppl’ (taking points in the middle of the score distribution) and ‘top-ppl’ (taking only the highest loss points).

IFD. We follow the procedure used in Li et al. (2024b), which involves first training a model on representative samples from the dataset, and then scoring data points using the ratio of the answer loss given the question to the loss of the answer on its own (called the IFD score). We compute the IFD score for all points d ∈ D. We use the codebase provided by the authors.1 When selecting data from the 5.8M TÜLU 2 unfiltered pool, we use the same model trained on the 200k-size pool, as the smaller pool is simply a subsampled version of the larger one.

LESS. We follow the procedure outlined in Xia et al. (2024), training LoRAs on a random subset of the data, and then selecting data by computing the

1https://github.com/tianyi-lab/Cherry_LLM

Algorithm 1 Single-task selection method Require: A dictionary of scores S such that

S[v,d] is the score given by a selection method between points v ∈ V and d ∈ D.

Require: n, the desired size of our selected dataset. L ← [] while |L| < n do

### for v ∈ V do

L ← argmaxd∈DS[v,d] S[v,d] ← −inf ▷ Set score such that

this point will not get chosen again. if |L| ≥ n then break end if

end for end while return L

gradient-based influence of each d ∈ D on each v ∈ V to obtain the selection score for the pair v,d. We use the codebase provided by Xia et al. (2024).2

Embedding. We use an embedding model to score each pair v,d ∈ V,D based on the cosine similarity of v and d when processed by the embedding model. We test using two different embedding models: NV-Embed-v2 (Lee et al., 2024), which was at the top of the MTEB leaderboard (Muennighoff et al., 2023b) at time of experimentation, and GTR-base (Ni et al., 2022), following Yin and Rush (2024).

RDS+. Finally, similar to embedding-based models, we explore using the hidden representations of the model we plan to train, as opposed to a trained embedding model (Zhang et al., 2018; Hanawa et al., 2021; Ivison et al., 2023a; Xia et al., 2024) – representation-based data similarity (RDS). We use a custom variant of RDS in which we take a position-weighted mean pool of the last hidden layer states (see §G.1 for details). We compute this for each v,d in V and D and compute the cosine similarity for each pair v,d ∈ V,D. We ablate alternate variants of RDS in App. G.1, and denote our tuned version of RDS as ‘RDS+’.

### 3.1 Selection & Aggregation

For methods that provide a score for each pair v,d ∈ V,D, we end up with |V | scores for each data point d ∈ D. As such, we need to determine

2https://github.com/princeton-nlp/LESS

how to aggregate these scores to determine the most valuable points d ∈ D. In pilot experiments, we found using a round-robin approach worked best, iteratively adding to the selected pool the highestscoring point for each v ∈ V until we reach a desired size. This is what we use to construct taskspecific datasets. We illustrate the algorithm in Alg 1.

For multi-task scenarios, we also need to aggregate on a task-level when combining scores from different tasks. We say that Vt is the query set for task t, and S[vt,d] is the score computed between dataset point d and query point vt. We then construct a dictionary of scores S′ where S[t,d] represents the score of a d for task t by setting S[t,d] = maxvt∈Vt S[vt,d]. We then run a round-robin procedure where we iterative over tasks, ‘pop’ out the highest-scoring data point, add to our dataset, and repeat until we have a dataset of the desired size (after deduplication). This is essentially running Alg. 1 again, but replacing S with S′ and V with T. We also experimented with averaging task-level scores together, but found this performed worse overall (See App. D for details).

### 3.2 Data Pool

In this work, we wish specifically to explore how well automated data selection methods work at scale. As such, we test the above methods in three different settings, across two different dataset pools: TÜLU 2 unfiltered and TÜLU 3 unfiltered. Both pools comprise of millions of samples with diverse data sources, and are constructed by examining the original datasets chosen for TÜLU 2 and 3 (state-ofthe-art post-trained models with openly available instruction data) and retrieving the same datasets but performing no downsampling. We perform exact-match deduplication of samples over the pool to ensure all samples are unique. Note that the unfiltered mix is extremely unbalanced, comprising mostly of data from FLAN and Open Orca. We compare the makeup and size of our data pools to pools in prior work (Xia et al., 2024; Li et al., 2024b) in Fig. 2. Notably, our pools are significantly larger and more diverse than prior work, and are based on real datasets used for open instructiontuned models.

### 3.3 Experimental Design

We extend our experimental design off TÜLU 2 (Ivison et al., 2023b), an open-source state-of-the-art dataset and model family (at time of release). As

1e6 5,817,792

| |4,783,043| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | |270,|679<br><br>70,0|00 52,0|00|

- 1
- 2
- 3
- 4
- 5

#Samples

Unfil. Tulu 2 (Ours) Unfil. Tulu 3 (Ours) LESS Pool IFD Pool Alpagasus Pool

- Figure 2: Size and makeup of data pools considered in this work (unfiltered Tulu 2, 3) and in past work (Xia et al., 2024; Chen et al., 2024; Li et al., 2024b). We provide the size of each pool on top of each bar. Each color represents a different dataset. See App. B for more details on data pool composition and exact per-source counts.

TÜLU 2 is finetuned starting from Llama 2 base models (Touvron et al., 2023), we primarily experiment with the Llama 2 7B model. We additionally report results using the TÜLU 3 mixture (Lambert et al., 2024), used to train a state-of-the-art open instruction-tuned model, and Llama 3.1 (Llama Team, 2024) in §4.2, and 8 varied models in §5.

Training. For finetuning, we fully finetune for two epochs with a batch size of 1, 128 gradient accumulation steps, a learning rate of 2e−5 (1e−5 for 70B size models), linear learning rate warmup for the first 3% of training steps, and linear cooldown for the rest of training. This follows the settings used in prior work (Wang et al., 2023; Ivison et al., 2023b). We report the mean across three random runs (including reselecting data) for random baselines and single-run scores for other settings.

Evaluation. For evaluation, we largely follow the same evaluation procedures as described in Wang et al. (2023); Ivison et al. (2023b). We evaluate on (and select for) MMLU (Hendrycks et al., 2020), GSM8k (Cobbe et al., 2021), BBH (Suzgun et al., 2022), TydiQA (Clark et al., 2020), HumanEvalCodex (Chen et al., 2021a), Squad (Rajpurkar et al., 2016), and AlpacaEval (Li et al., 2023). We either use the existing few-shot examples or create custom development splits from the evaluations for data selection. We also experiment with using outof-domain query sets in §4.2. We provide precise details on each evaluation and query set in Appendix A.

FLOPs Estimation. For the FLOPs estimates used throughout the paper, we follow Kaplan et al. (2020) in estimating the compute cost of a training step as roughly 6N FLOPs per token processed and an inference step as 2N per token processed, where

N is the parameter count of the model. Based on this, we estimate the FLOPs of each method using the total number of inference and training steps that take place across selection and model training. We provide further details in App. E.

## 4 Results

### 4.1 Single-Task Data Selection

We start by testing the different data selection methods over TÜLU 2 unfiltered both with a smaller pool (TÜLU 2 unfiltered downsampled to 200k samples) and the entire data pool. We select 10k samples, train models, and then evaluate for each task separately. We report the results in Table 1. We do not report LESS performance using the 5.8M pool as it requires computing gradients three times over all 5.8M samples in the pool, which is beyond our computational budget.

RDS+ performs best. RDS+ is the highestperforming method on average in both settings, despite being similar or cheaper in cost than most other baselines (Fig. 1). Additionally, models trained with RDS-selected data perform the best in each task individually (with the exception of SQuAD, where they are a close second) when selecting from the full pool.

PPL, Random, IFD, and Embed (NV) perform worse with a larger pool. As seen in Fig. 1, multiple methods we test actually perform worse when we select over the larger data pool, suggesting that they cannot effectively scale. In contrast, both RDS+ and Embed (GTR) improve with a larger pool. In later areas of the paper, we focus on RDS+ as we wish to achieve the strongest possible performance with a larger compute budget.

Method MMLU GSM8k BBH TydiQA Codex Squad AlpacaEval Average Selecting 10k datapoints for single tasks from 200k data points

- Random (unbal.) 45.8 17.2 41.5 52.7 27.0 83.5 18.4 40.9 Random (bal.) 45.4 16.6 40.7 50.7 27.9 80.6 33.2 42.2 LESS 46.7 21.5 41.8 57.5 19.6 84.7 37.9 44.2

- Embed (NV) 46.9 19.9 42.4 52.3 25.0 88.3 24.0 42.7 Embed (GTR) 47.1 20.2 42.4 52.4 27.7 88.3 16.7 42.1 Top-PPL 46.4 9.9 33.8 46.9 20.3 77.6 4.3 34.2 Mid-PPL 46.9 15.5 39.7 53.6 25.7 81.6 12.9 39.4 IFD 42.6 17.5 39.1 38.7 29.1 57.9 45.9 38.7 Length 43.2 14.2 40.2 54.3 30.4 82.3 40.0 43.5 RDS+ 47.0 20.9 42.7 52.3 31.8 88.2 42.3 46.4

Selecting 10k datapoints for single tasks from all 5.8M data points

Random (unbal.) 46.9 17.4 42.1 53.2 26.8 83.6 15.1 40.7 Random (bal.) 45.3 16.5 40.2 50.9 28.2 78.0 29.0 41.1 Top-PPL 44.6 6.1 23.9 40.4 20.9 74.4 2.5 30.4 Mid-PPL 46.1 14.9 40.7 52.7 23.0 82.3 12.5 38.9 Embed (GTR) 45.0 29.9 42.8 45.5 29.7 88.3 43.8 46.4

- Embed (NV) 47.0 23.1 41.1 51.2 29.1 88.8 10.9 41.6 IFD 41.9 13.0 37.9 40.0 26.4 46.6 44.6 35.8 Length 40.7 2.4 39.2 17.8 20.9 0.2 3.1 17.8 RDS+ 47.5 33.9 42.9 54.9 32.4 88.5 53.5 50.5

- Table 1: Single-task performance of different data selection techniques over the TÜLU 2 unfiltered set. Each cell reports the performance of a model trained with 10k samples chosen for that particular target task. We show results selecting from a downsampled form or full set of the TÜLU 2 unfiltered set. We find RDS performs best overall, even beating more computationally expensive methods like LESS.

### 4.2 Multi-task Selection

We next examine how well data selection methods work when selecting single datasets while targeting multiple tasks. We use the round-robin method described in §3.1 to balance selection across tasks for task-specific methods3, or just use the existing scores for task-agnostic methods. We select 326k samples from the TÜLU 2 unfiltered set, matching the number of samples in the TÜLU 2 SFT mixture, and so also compare to TÜLU 2. We show the results in Table 2 and find that:

RDS+ still performs best overall. RDS+ outperforms all other methods on average, including TÜLU 2 itself, showing that RDS+ can humancurated mixtures. In fact, we observe that all methods except RDS+ and Embed (GTR) still underperform random selection, suggesting that embedding-based methods overall are best for data selection.

3We found this worked best across multiple data selection methods.

RDS+ selection performs well even when the evaluations are out of distribution. We explore using WildChat (Zhao et al., 2024c) and Arena Hard (Li et al., 2024c) as out of distribution query sets for selecting data. Crucially, this means we do not assume access to any query samples from our test tasks. We find that using Arena Hard samples performs close to RDS+, showing we can select high-quality samples without assuming any data from the evaluations in our test suite. This suggests that using a high-quality selection set results in a data mixture that generalizes well to unseen tasks.

RDS+ performs strongly with other data pools. In order to test how well our findings generalize, we make use of the recently released state-of-theart TÜLU 3 data and model (Lambert et al., 2024), comparing the model trained with RDS-selected data the officially released TÜLU 3 SFT model. We then select 939k instances, equal to the size of TÜLU 3 SFT Mix, from the pool and finetune Llama 3.1 models on it using the hyperparameters from Lambert et al. (2024). We show our results in Table 3. We find that RDS+ round-robin se-

Rand. (unbal) 52.2 18.0 44.5 55.3 25.7 81.5 33.9 44.5 Rand. (bal) 51.5 21.8 45.1 50.7 32.2 87.9 43.2 47.5 Top-PPL 49.1 10.5 39.4 49.4 21.6 80.3 5.6 36.6 Mid-PPL 53.1 13.3 42.8 52.4 20.3 86.2 20.7 41.3 Embed (GTR) 49.9 32.8 44.6 54.4 30.4 88.4 35.7 48.0 Embed (NV) 50.6 28.7 44.4 56.0 30.4 89.1 17.9 45.3 IFD 45.7 21.8 41.2 39.5 27.7 17.0 57.4 35.7 Length 50.0 16.4 38.8 54.9 25.0 89.3 58.5 47.6 TÜLU 2 50.0 22.7 45.1 54.0 33.1 86.9 54.4 49.5 RDS+ 50.2 35.2 44.7 56.3 35.1 89.0 45.6 50.9

RDS+ - Wildchat 50.9 24.8 43.6 57.3 31.1 87.3 39.3 47.8 RDS+ - Arena Hard 48.1 36.2 43.9 51.8 31.8 81.3 59.4 50.4

- Table 2: Multi-task performance of dataset selection methods when selecting 326k samples from the full TÜLU 2 unfiltered pool. Each row reflects the performance of a single model trained on a single dataset chosen to perform well across tasks. For ‘WildChat’ and ‘Arena Hard’ we use samples from WildChat and Arena Hard for selection.

20.6% 43.0%

52

5.6%

100.0%

50

Avg.Performance

20.6%

12.0%

43.0%

48

5.6%

0.9%

3.1%

46

0.4%

0.9%

1.7%

44

0.2%

0.4%

RDS

42

0.2%

Balanced Random

0

8

9

101

101

102

Total FLOPs Cost (estimated)

- Figure 3: Average multi-task performance against FLOPs cost (including selection) for balanced random and RDS+. We label points with the % of the total data pool used. RDS+ outperforms random selection significantly when selecting less data, and is more FLOPs efficient at larger selection sizes. See App. E for details on FLOPs estimates.

lected data outperforms the official TÜLU 3 SFT checkpoint and random baselines, showing that RDS+ remains an effective data selection method even on a different base model and with a different selection data pool.

### 4.3 Scaling Multi-task Selection

Finally, we examine how the performance of our selection method changes as we scale the amount of data being selected. Due to the high cost of these experiments (fully finetuning on millions of samples), we only look at RDS+ (the bestperforming method overall) and random selection

52

50

Avg.Performance

48

46

44

RDS

42

Balanced Random

10k 25k 50k 100k180k326k 700k1.2M 2.5M 5.8M

# Samples Selected

Figure 4: Average multi-task performance against number of samples selected. RDS+ consistently beats balanced random at all data sizes tested, up to using the entire data pool.

(a strong, cheap baseline). We examine RDS+ as the strongest performing data selection method, and compare to random selection as a cheap but often effective baseline (that also outperforms all but two of the tested data selection approaches). We select for multiple tasks at once from the full TÜLU 2 unfiltered pool, selecting varying numbers of samples with RDS+. We plot average performance against the number of samples selected in Figure 4. We also plot performance against total FLOPs used for selection and training in Figure 3. We observe that:

RDS+ consistently beats balanced random selection across different data selection sizes (Fig. 4). This shows that targeted selection is even

Random (unbal.) 61.6 81.2 66.8 71.1 76.4 89.7 75.6 74.6 Random (bal.) 62.1 76.0 68.6 68.8 87.2 87.4 72.4 74.7 TÜLU 3 SFT 62.2 74.3 68.2 67.4 83.8 85.5 71.9 73.3

RDS+ 62.5 77.6 66.6 72.1 83.8 90.2 80.2 76.1 RDS+ - Arena Hard 57.0 78.7 59.7 49.4 75.7 66.3 84.5 67.3

- Table 3: Multi-task performance of RDS against baselines when finetuning from Llama 3.1 8B base and selecting 939k samples from the TÜLU 3 unfiltered mixture following the multitask setup in §4.2. For ‘Arena Hard’ we use samples from Arena Hard as the query set. RDS+ outperforms the official TÜLU 3 SFT model.

useful when selecting datasets up to millions of data points in size. Furthermore, RDS+ performs similarly to training on all data (50.8 vs 50.9) even when selecting only 326k samples (roughly 6% of the overall pool), and outperforms training on all data when selecting over 1M samples.

RDS+ can achieve better performance with less compute when selecting more data points (Fig. 3). When taking into account the compute used for selection and training, we find that the extra compute incurred by RDS does not pay off until we select relatively large datasets (roughly ≥ 326k), after which it becomes significantly better than random selection. This highlights that examining data selection performance at scale is important to show the potential of some methods.

## 5 Analysis

What gets selected? We examine qualitatively what samples get selected by RDS+, IFD, and PPL (see App. I for a visualization). For RDS+, We find that the sources selected vary depending on downstream evaluation, often with intuitive explanations: e.g., GSM8k requires strong chain-of-thought reasoning abilities, and so the chain-of-thought data is upweighted relative to the normal distribution. Similarly, for HumanEval-Codex the Code Alpaca dataset is upweighted.

Additionally, we observe that PPL and IFD appear to select more ShareGPT and FLAN data respectively than RDS+ or random selection, suggesting these methods have biases to particular types of data. This also explains why IFD performs relatively well in AlpacaEval (which improves when training on GPT-derived data), but drops in other aspects, while Top-PPL achieves very low AlpacaEval scores in Table 1.

Can we reduce the cost of RDS+? While we have used the same model for selection and train-

ing for RDS+, we also wished to investigate to what extent changing the model used for selection impacts data selection performance, potentially allowing RDS+ to use less or more compute for selection. Encouragingly, we find that using smaller models for selection (even from different model families) can still results in strong performance, suggesting that RDS+ can indeed be made computationally cheaper by using smaller models for selection. However, we do not observe gains from using larger models compared to Llama 2 7B, suggesting that using larger models for selection does not yield improved performance. We provide further details in App. F.

## 6 Conclusion

In this work, we have explored how well a variety of data selection techniques work when selecting datasets of up to 2.5M samples from pools comprising of close to 6M data points. By examining data selection techniques in these settings, we find that many methods proposed in the past not only underperform random selection baselines, but also perform worse with larger pools of data. Embeddingbased methods, and in particular RDS+, are the primary exception, outperforming all other data selection methods across various selection sizes. Finally, we find that using smaller models to select data can also work well, suggesting future work could further reduce the cost of RDS+ by exploring how best to use smaller proxy models. We believe that our results overall highlight the importance of testing data selection methods over large, diverse pools of data, and testing how well data selection approaches scale with respect to both data and compute. We will publicly release our data and code to aid in future work.

## Limitations

While we cover a reasonable number of base models in this work, we ultimately only use two data pools, TÜLU 2 and TÜLU 3, due to the computational cost of running experiments over millions of instances. We hope to further examine how well our findings transfer to data pools with different characteristics in future work.

Additionally, we note that any data selection method that requires model passes over data will scale in cost proportionally with the data pool used to select, which limits the use of this method in extremely large-scale settings – where even doing model forward passes over all data may be too computationally expensive. While this is less common in instruction-tuning settings (as there is less instruction-tuning data available generally), it may happen for pretraining datasets containing trillions of tokens (which is not a focus in this work).

Finally, while we do not explicitly analyze the safety and potential toxicity of our models, we hope that our findings could be used to improve data selection for selecting safety-relevant data, which often does make up some portion of popular instruction data mixes (Lambert et al., 2024). Furthermore, we hope that reducing the total FLOPs cost of training strong instruction models can help aid the overall environmental cost of training LMs (Strubell et al., 2020).

## References

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403.

Zachary Ankner, Cody Blakeney, Kartik Sreenivasan, Max Marion, Matthew L. Leavitt, and Mansheej Paul. 2024. Perplexed by perplexity: Perplexity-based data pruning with small reference models. Preprint, arXiv:2405.20541.

Richard Antonello, Nicole Beckage, Javier Turek, and Alexander Huth. 2021. Selecting informative contexts improves language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1072–1085, Online. Association for Computational Linguistics.

BIG-bench authors. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research.

Faeze Brahman, Sachin Kumar, Vidhisha Balachandran, Pradeep Dasigi, Valentina Pyatkin, Abhilasha Ravichander, Sarah Wiegreffe, Nouha Dziri, Khyathi Chandu, Jack Hessel, Yulia Tsvetkov, Noah A. Smith, Yejin Choi, and Hannaneh Hajishirzi. 2024. The Art of Saying No: Contextual Noncompliance in Language Models.

Yihan Cao, Yanbin Kang, Chi Wang, and Lichao Sun. 2024. Instruction mining: Instruction data selection for tuning large language models. Preprint, arXiv:2307.06290.

Sahil Chaudhary. 2023. Code alpaca: An instructionfollowing llama model for code generation. GitHub repository.

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, and Hongxia Jin. 2024. Alpagasus: Training a better alpaca with fewer data. In The Twelfth International Conference on Learning Representations.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021a. Evaluating large language models trained on code.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021b. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics, 8:454–470.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Qirun Dai, Dylan Zhang, Jiaqi W. Ma, and Hao Peng. 2024. Improving influence-based instruction tuning data selection for balanced learning of diverse capabilities.

Databricks. 2023. Free dolly: Introducing the world’s first truly open instruction-tuned llm. Blog post.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Harshita Diddee and Daphne Ippolito. 2024. Chasing random: Instruction selection strategies fail to generalize. Preprint, arXiv:2410.15225.

Qianlong Du, Chengqing Zong, and Jiajun Zhang. 2023. Mods: Model-oriented data selection for instruction tuning. Preprint, arXiv:2311.15653.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. 2024. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. Preprint, arXiv:2406.18495.

Xiaochuang Han, Byron C. Wallace, and Yulia Tsvetkov. 2020. Explaining black box predictions and unveiling data artifacts through influence functions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5553– 5563, Online. Association for Computational Linguistics.

Kazuaki Hanawa, Sho Yokoi, Satoshi Hara, and Kentaro Inui. 2021. Evaluation of similarity-based explanations. Preprint, arXiv:2006.04528.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring Massive Multitask Language Understanding.

Hamish Ivison, Noah A. Smith, Hannaneh Hajishirzi, and Pradeep Dasigi. 2023a. Data-efficient finetuning using cross-task nearest neighbors. In Findings of the Association for Computational Linguistics: ACL 2023, pages 9036–9061, Toronto, Canada. Association for Computational Linguistics.

Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. 2023b. Camels in a changing climate: Enhancing lm adaptation with tulu 2. Preprint, arXiv:2311.10702.

Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha Dziri. 2024. Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. Preprint, arXiv:2001.08361.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, et al. 2023. Openassistant conversations–democratizing large language model alignment. arXiv preprint arXiv:2304.07327.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. 2024. Tülu 3: Pushing frontiers in open language model post-training. Preprint, arXiv:2411.15124.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. Preprint, arXiv:2405.17428.

Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. 2024. Numinamath tir. [https://huggingface.co/ AI-MO/NuminaMath-TIR](https://github.com/ project-numina/aimo-progress-prize/blob/ main/report/numina_dataset.pdf).

Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, Ning Cheng, and Tianyi Zhou. 2024a. Superfiltering: Weak-to-strong data filtering for fast instruction-tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14255–14273, Bangkok, Thailand. Association for Computational Linguistics.

Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. 2024b. From quantity to quality: Boosting LLM performance with self-guided data selection for instruction tuning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),

pages 7602–7635, Mexico City, Mexico. Association for Computational Linguistics.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. 2024c. From crowdsourced data to highquality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval.

Yunshui Li, Binyuan Hui, Xiaobo Xia, Jiaxi Yang, Min Yang, Lei Zhang, Shuzheng Si, Ling-Hao Chen, Junhao Liu, Tongliang Liu, Fei Huang, and Yongbin Li. 2024d. One-shot learning as instruction data prospector for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4586–4601, Bangkok, Thailand. Association for Computational Linguistics.

Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". 2023. Openorca: An open dataset of gpt augmented flan reasoning traces. https://https://huggingface.

co/Open-Orca/OpenOrca.

Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. 2024. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. In The Twelfth International Conference on Learning Representations.

Llama Team. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Keming Lu, Hongyi Yuan, Zheng Yuan, Runji Lin, Junyang Lin, Chuanqi Tan, Chang Zhou, and Jingren Zhou. 2023. #instag: Instruction tagging for analyzing supervised fine-tuning of large language models. Preprint, arXiv:2308.07074.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2024. Wizardcoder: Empowering code large language models with evolinstruct. In The Twelfth International Conference on Learning Representations.

Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, and Sara Hooker. 2023. When less is more: Investigating data pruning for pretraining llms at scale. Preprint, arXiv:2309.04564.

Niklas Muennighoff. 2022. Sgpt: Gpt sentence embeddings for semantic search. Preprint, arXiv:2202.08904.

Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and

Shayne Longpre. 2023a. OctoPack: Instruction Tuning Code Large Language Models. arXiv preprint arXiv:2308.07124.

Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. 2023b. MTEB: Massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2014–2037, Dubrovnik, Croatia. Association for Computational Linguistics.

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022. Large dual encoders are generalizable retrievers. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9844–9855, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

OLMo Team. 2024. OLMo 2: The best fully open language model to date.

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277.

Qwen Team. 2024. Qwen2.5: A Party of Foundation Models.

Nazneen Rajani, Lewis Tunstall, Edward Beeching, Nathan Lambert, Alexander M. Rush, and Thomas Wolf. 2023. No robots. https://huggingface.co/ datasets/HuggingFaceH4/no_robots.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, Jianmo Ni, Lichan Hong, Ed H. Chi, James Caverlee, Julian McAuley, and Derek Zhiyuan Cheng. 2024. How to train data-efficient llms. Preprint, arXiv:2402.09668.

Shivalika Singh, Freddie Vargus, Daniel D’souza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura O’Mahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Moura, Dominik Krzemi´nski, Hakimeh Fadaei, Irem Ergun, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Chien, Sebastian Ruder, Surya Guthikonda, Emad Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee, and Sara Hooker. 2024. Aya dataset: An open-access collection for multilingual instruction tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11521–11567, Bangkok, Thailand. Association for Computational Linguistics.

Emma Strubell, Ananya Ganesh, and Andrew McCallum. 2020. Energy and policy considerations for modern deep learning research. Proceedings of the AAAI Conference on Artificial Intelligence, 34(09):13693–13696.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford_alpaca.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. 2024. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. Preprint, arXiv:2307.09288.

David Wadden, Kejian Shi, Jacob Morrison, Aakanksha Naik, Shruti Singh, Nitzan Barzilay, Kyle Lo, Tom Hope, Luca Soldaini, Shannon Zejiang Shen, Doug Downey, Hannaneh Hajishirzi, and Arman Cohan. 2024. Sciriff: A resource to enhance language model instruction-following over scientific literature. Preprint, arXiv:2406.07835.

Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. 2023. How far can camels go? exploring the state of instruction tuning on open resources. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

Alexander Wettig, Aatmik Gupta, Saumya Malik, and Danqi Chen. 2024. QuRating: Selecting high-quality data for training language models. In International Conference on Machine Learning (ICML).

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. 2024. LESS: Selecting influential data for targeted instruction tuning. In International Conference on Machine Learning (ICML).

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Junjie Oscar Yin and Alexander M. Rush. 2024. Compute-constrained data selection. Preprint, arXiv:2410.16208.

Zichun Yu, Spandan Das, and Chenyan Xiong. 2024. Mates: Model-aware data selection for efficient pretraining with data influence models. In NeurIPS.

Liangyu Zha, Junlin Zhou, Liyao Li, Rui Wang, Qingyi Huang, Saisai Yang, Jing Yuan, Changbao Su, Xiang Li, Aofeng Su, Tao Zhang, Chen Zhou, Kaizhe Shou, Miao Wang, Wufang Zhu, Guoshan Lu, Chao Ye, Yali Ye, Wentao Ye, Yiming Zhang, Xinglong Deng, Jie Xu, Haobo Wang, Gang Chen, and Junbo Zhao. 2023. Tablegpt: Towards unifying tables, nature language and commands into one gpt. Preprint, arXiv:2307.08674.

Jipeng Zhang, Yaxuan Qin, Renjie Pi, Weizhong Zhang, Rui Pan, and Tong Zhang. 2024a. Tagcos: Taskagnostic gradient clustered coreset selection for instruction tuning data. Preprint, arXiv:2407.15235.

Qi Zhang, Yiming Zhang, Haobo Wang, and Junbo Zhao. 2024b. RECOST: External knowledge guided data-efficient instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 10911–10921, Bangkok, Thailand. Association for Computational Linguistics.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR.

Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. 2024c. Autonomous data selection with language models for mathematical texts. arXiv preprint arXiv:2402.07625.

Hao Zhao, Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. 2024a. Long is more for alignment: A simple but tough-to-beat baseline for instruction fine-tuning. In Forty-first International

Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Hao Zhao, Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. 2024b. Long is more for alignment: a simple but tough-to-beat baseline for instruction fine-tuning. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. 2024c. Wildchat: 1m chatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, LILI YU, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. LIMA: Less is more for alignment. In Thirty-seventh Conference on Neural Information Processing Systems.

## A Evaluation Details

We provide more details on each evaluation setting and how we constructed the split used for data selection below. In all cases we use an instruction chat template following the TÜLU format, matching the format used during SFT training.

- 1. MMLU (Hendrycks et al., 2020): We use the official MMLU evaluation script and prompts available at https://github.com/ hendrycks/test. We evaluate using 0 fewshot examples, following the original setup of MMLU. We report average accuracy across test examples. For the query set, we use the dev examples (commonly used as few-shot samples for evaluation).
- 2. GSM8k (Cobbe et al., 2021): We evaluate models on the test set of GSM. Following Wei et al. (2022), we evaluate with chain-ofthought. We use 8 few-shot in-context examples. Because all answers in GSM are numbers, we extract the last number in the model response as the final answer. We report average accuracy across test examples. For the query set, we use the 8 few-shot examples individually (without the other shots included in the prompt).
- 3. Big Bench Hard (BBH) (BIG-bench authors, 2023; Suzgun et al., 2022): We follow the setup described in the original paper and evaluate with chain-of-thought. The officially

- provided prompts, which have 3 few-shot incontext examples are used. For the CoT setup, we extract the first word after the phrase ‘So the answer is’, or the entire response if there is no such substring present. We report average accuracy over sub-tasks (all of which use accuracy as the primary metric). For the query set, we use the 3 few-shot examples from each task individually (without the other shots included in the prompt).
- 4. TydiQA (Clark et al., 2020): We follow the setup described in the PaLM 2 technical report (Anil et al., 2023) to evaluate models’ performance in answering multilingual questions under two settings when the gold passage that contains the answer is given (i.e., gold passage setting). One in-context example is used to familiarize the model with the answering format. For the query set, we use the 9 few-shot examples from each language individually (without any shots included in the prompt).
- 5. HumanEval Codex (Chen et al., 2021b): We use the HumanEval dataset in the codex paper for evaluating models’ coding ability. The dataset contains 164 programming problems, where models are prompted to complete the Python function given its docstring. Following the original paper, we compute unbiased estimates of pass@k to measure the functional correctness of models’ outputs. We report pass@10, sampling with a temperature of 0.8. We additionally use the instructions provided by HumanEvalPack (Muennighoff et al., 2023a), as this better suites instruction-tuned models. We create a custom test split of 148 samples, and evaluate on those. We use the remaining 16 samples as the query split.
- 6. SQuAD (Rajpurkar et al., 2016): We use the validation split of SQuAD as a test set, comprising of 10,570 questions about Wikipedia articles. We include the article containing the answer in the prompt, and include 3 in-context examples (randomly selected from the train set) in order to ensure the model outputs in the desired format.4 We report text-based F1.

4In pilot experiments, we found that models without incontext samples would often provide the right answer in a verbose manner, harming the string-matching-based metrics used for SQuAD.

Dataset # Query # Test MMLU 285 14,042 GSM8k 8 1,319 BBH 81 6,511 TydiQA 9 5,077 Codex 16 148 Squad 500 10,570 AlpacaEval 50 755

- Table 4: Query and test split counts for evaluation datasets.

We use 500 random samples from the SQuAD train set as query samples.

7. AlpacaEval (Li et al., 2023): We use the package provided by Li et al. (2023), following the default setup for both AlpacaEval 1. We allow the evaluated model to generate up to 8192 tokens, without specifying special stop sequences. We create a custom split of 50 samples for the query set, and use the remaining samples for evaluation.

We provide a summary of the number of query and test samples in Table 4.

## B Data Pool Sources Breakdown

We provide a breakdown of the splits of the TÜLU 2 SFT mixture and our ‘unfiltered’ version, along with other data pools from prior work, in Table 5. We provide a similar comparison between the TÜLU 3 SFT mixture and our ‘unfiltered’ version in Table 6.

## C Compute Details

We run all experiments on a cluster containing H100s and A100s with up to 8 GPUs per node. We find that training on 10,000 examples takes roughly 30 minutes on 1 H100 GPU, while training on 2.5 million samples (our largest selected set) takes 62 hours on a node of 8 H100s. Running the selection itself varies according to method, with RDS itself taking 87 H100 GPU-hours for indexing the TÜLU 3 unfiltered set (although this can be efficiently parallelized).

## D Multi-task Selection Algorithm

We experimented with two methods for selecting multi-task datasets: ‘round-robin’ (the method explained in § 3.1) and ‘mean-max’. For ‘mean-max’,

800

AlpacaEval

GSM8k

600

Frequency

400

200

0

0.70 0.75 0.80 0.85 0.90 0.95 Cosine Sim.

Figure 5: Histogram of RDS scores for the top 10,000 samples picked for GSM8k and AlpacaEval from the TÜLU 2 unfiltered pool. We find that AlpacaEval instances have lower average similarity than GSM8k.

we compute the per-task scores for d in the same way as before, but then simply average all task scores for a datapoint together. Using the notation from § 3.1, we do: S[d] = t∈T S[t,d]

|T| , where S[d] is the score for d. We then can simply take the top-scoring points as our dataset.

We compare using round-robin and mean-max for RDS+ and Embed in Table 7. We find that round-robin consistently outperforms mean-max across methods on average, and so we use it in our core experiments.

We further investigate why this is the case by visualizing the distribution of RDS scores over the TÜLU 2 unfiltered pool in Figure 5 for GSM8k and AlpacaEval. We observe that scores for GSM8k are consistently higher than AlpacaEval, suggesting that GSM8k scores ‘dominate’ when averaging task scores together. This means that the final selected dataset is likely to be mostly comprised of samples that are useful for GSM8k, but not AlpacaEval. We find similar trends comparing AlpacaEval to other datasets, suggesting that merely averaging tasks scores leads to some targeted evaluations being under-served. This agrees with concurrent work suggesting that round-robin algorithms that carefully balance selected samples across tasks outperform more naive methods (Dai et al., 2024).

## E Further Details on FLOPs Estimates

For the FLOPs estimates used throughout the paper, we follow Kaplan et al. (2020) in estimating the compute cost of a training step as roughly 6N FLOPs per token processed, where N is the param-

##### Source TÜLU 2 TÜLU 2 Unfil. LESS IFD Alpagasus

FLAN V2 (Chung et al., 2022) 49,123 961,322 100,000 - FLAN CoT (Chung et al., 2022) 49,747 398,439 100,000 - Open Assist. (Köpf et al., 2023) 7,331 7,707 55,668 - Dolly (Databricks, 2023) 0 15,007 15,011 - GPT-4 Alpaca (Peng et al., 2023) 19,906 52,002 - - Code Alpaca (Chaudhary, 2023) 20,016 20,022 - - ShareGPT 111,912 100,054 - - LIMA (Zhou et al., 2023) 1,018 1,030 - - Wizard Evol-Instruct V2 (Xu et al., 2023) 29,810 142,802 - - Open Orca (Lian et al., 2023) 29,683 4,111,858 - - SciRIFF (Wadden et al., 2024) 7,468 7,535 - - Alpaca (Taori et al., 2023) - - - 52,002 52,002 WizardLM (70K) (Xu et al., 2023) - - - 70,000 Hardcoded 140 14 - - -

Total 326,153 5,817,792 270,679 122,002 52,002

- Table 5: Number of samples per dataset in the original TÜLU 2 dataset and our unfiltered version alongside the data pools used in LESS (Xia et al., 2024), IFD (Li et al., 2024b), and Alpagasus (Chen et al., 2024). Note that we deduplicate samples in the TÜLU 2 unfiltered mixture.

eter count of the model (roughly 7B). Kaplan et al. (2020) notes that the forward pass is roughly half the cost of the backward pass, giving us an estimate of 2N FLOPs per token when processing samples. We use a rough estimate of 2,048 tokens per sample, since during training and selection we truncate all samples to be at most this length. Note we fullyfinetune models for two epochs in all setups. Let N be the model size, P be the size of the data pool (in number of samples), and D the number of samples selected to train on. Based on this, the cost for each method is estimated as follows:

- 1. Random Selection: 2 ∗ 2048 ∗ 6ND
- 2. Perplexity: 2∗2048∗2NP +2∗2048∗6ND
- 3. IFD: 200000 ∗ 2049 ∗ 2N + 1000 ∗ 2048 ∗ 6ND + 2 ∗ 2048 ∗ 2NP + 2 ∗ 2048 ∗ 6ND (We train the initial model used to compute IFD scores on 1000 samples selected from the 200k data pool.)
- 4. LESS: 3∗2048∗6NP+2∗2048∗6ND (LESS computes gradients for three checkpoints over the entire pool.)
- 5. Embedding: 2∗2048∗2NP +2∗2048∗6ND
- 6. RDS+: 2 ∗ 2048 ∗ 2NP + 2 ∗ 2048 ∗ 6ND

Note that for embedding and RDS+, we can select using a smaller model than we train (as done

for Embed (GTR) or the experiments in App. F). In this case, we adjust the inference cost computation accordingly. We assume the cost of processing scores is negligible compared to the rest of the procedure, since in practice this runs on CPU in under an hour for methods we test. These formulations provide some intuition for why testing large selection sets (large P) is important: if P >> D, methods like RDS+ use significantly more compute than random selection. As D approaches P, the added cost of doing inference over the data pool becomes less significant.

## F Varying Model Size and Family for RDS+

Since RDS+ relies on the hidden states of the model, we examine to what extent changing the model used for selection impacts data selection performance. We construct a pool of 100k samples from TÜLU 2 unfiltered, balanced via uniform random downsampling across the different sources. We select from the pool using RDS+ with a range of models varying both in size (from 1.5B to 70B) and in family (Qwen (Qwen Team, 2024), Llama 2 (Touvron et al., 2023), Llama 3.1 (Llama Team, 2024), OLMo 2 (OLMo Team, 2024)), and show the results in Table 8. Surprisingly, we find that using selection models from different families to the ones being trained can still result in strong perfor-

### Source # Samples in TÜLU 3 # Samples in Unfil.

TÜLU 3 Hardcoded 240 24 Open Assist. (Köpf et al., 2023) 7,132 7,131 No Robots (Rajani et al., 2023) 9,500 9,500 WildChat (GPT-4 subset) (Zhao et al., 2024c) 100,000 235,028 FLAN V2 (Chung et al., 2022) 89,982 961,322 SciRIFF (Wadden et al., 2024) 10,000 35,149 TableGPT (Zha et al., 2023) 5,000 13,159 TÜLU 3 Persona MATH 149,960 149,960 TÜLU 3 Persona GSM 49,980 49,980 TÜLU 3 Persona Algebra 20,000 50,000 OpenMathInstruct 2 (Toshniwal et al., 2024) 50,000 2,570,505 NuminaMath-TIR (LI et al., 2024) 64,312 64,312 TÜLU 3 Persona Python 34,999 34,998 Evol CodeAlpaca (Luo et al., 2024) 107,276 107,276 TÜLU 3 CoCoNot (Brahman et al., 2024) 10,983 10,983 TÜLU 3 WildJailbreak (Jiang et al., 2024) 50,000 178,344 TÜLU 3 WildGuardMix (Han et al., 2024) 50,000 85,090 Aya (Singh et al., 2024) 100,000 190,320 TÜLU 3 Persona IF 29,980 29,962

Total 939,344 4,783,043

- Table 6: Number of samples per dataset in the original TÜLU 3 SFT mixture and our unfiltered version. See Lambert et al. (2024) for further details on all splits.

Sel. Meth. Aggr. Meth. Avg. Perf.

RDS+ round-robin 50.9 RDS+ mean-max 47.9

Embed (GTR) round-robin 48.0 Embed (GTR) mean-max 44.9

- Table 7: Average performance of different methods when selecting 326k samples from TÜLU 2 unfiltered using the multitask setting, using either round-robin or mean-max methods for aggregating samples across tasks. Round-robin beats mean-max performance in both cases. See App. D for details.

## G Further Details on RDS+

### G.1 Ablations

In this work, we use a custom RDS variant (‘RDS+’) that we chose based on a series of ablations testing (a) using different sets of hidden states and (b) using different parts of the input data itself. We perform the ablations in the 200k single-task setting (matching the setting used for Table 4.1).

For constructing the embeddings, we considered using just the hidden state corresponding to the final EOS token (EOS token only) (Xia et al., 2024), mean-pooling the final hidden states across the sequence (uniform mean pool), or using a weighted mean-pooling approach that takes into account the causal attention mask used with decoderonly models, inspired by past work on converting decoder-only models to generic text embedding models (Muennighoff, 2022). See below for further detail on the weighting. As shown in Table 9, using the EOS token alone or uniform mean-pooling underperforms our chosen weighted mean pooling approach. Second, we explore using just the initial user turn (prompt-only) or the only the first response (label-only) instead of the entire input.

mance, suggesting that RDS’ strong performance does not come from matching the selection and downstream model, but from good general selection of samples. While we do observe some outliers (Llama 2 7B and 13B being particularly good and bad respectively), this suggests that using larger selection models or matching the selection and training model is not crucial for RDS+. This promising finding means that we can potentially select data for much larger models with much smaller ones, significantly reducing the compute cost of RDS+.

Train Model → Sel. Model ↓

Q2.5 L2 7B L2 13B L2 70B L3.1 8B L3.1 70B O2 8B O2 13B Avg.

Qwen 2.5 1.5B 56.5 43.5 54.4 70.6 65.8 82.7 57.0 64.6 61.9 Llama 2 7B 52.0 45.1 57.0 72.4 70.4 83.1 57.5 66.7 63.0 Llama 2 13B 52.0 42.4 54.4 70.3 62.9 82.3 56.3 63.4 60.5 Llama 2 70B 53.7 45.4 54.3 72.3 68.4 83.2 57.1 64.9 62.4 Llama 3.1 8B 53.0 45.0 55.9 72.5 67.8 82.1 57.2 65.7 62.4 Llama 3.1 70B 53.2 45.1 57.3 71.7 68.0 82.1 57.6 65.3 62.5 Olmo 2 7B 53.2 43.5 54.6 71.4 66.9 82.2 57.4 64.8 61.8 Olmo 2 13B 51.0 43.2 54.9 71.0 65.8 83.0 57.1 65.1 61.4

- Table 8: Average multi-task performance of RDS round-robin when when varying the model doing the selecting and the model being trained. We find that using a different model to the one being trained does not hurt performance, with Llama 2 7B being the best selector overall.

Method Avg. Perf. RDS (ours) 46.4

- - EOS token only 45.4
- - Uniform mean-pool 45.8

- - Prompt-only 43.2
- - Label-only 45.1

- Table 9: Overall performance of RDS variations when selecting 10k samples for single target tasks (matching Tab. 1). Using only labels is surprisingly effective.

the causal mask of the decoder-only models we use for embedding: since the model is trained with a causal mask, the first input token does not see the rest of the sequence. Likewise, the second token does not see token 3 onwards, etc. As such, naive mean-pooling may bias the features towards earlier tokens in the sequence, since hidden states only accumulate features from previous tokens. The weighted mean-pooling strategy attempts to counter this by then weighting later tokens heavier.

## H Visualization of RDS+

Surprisingly, using only the label outperforms using only the prompt, although both underperform using the entire sample, despite the label often containing relatively little task information (e.g., just the letter answer for multiple-choice questions).

### G.2 RDS+ Weighted Pooling Details

For the weighted mean-pooling strategy, we follow Muennighoff (2022) in using position weighting to average hidden states across the model inputs while taking into account the causal mask. Specifically, token i has weight wi as follows:

i

wi =

L i=1 i

Where L is the total length in tokens of the given sample. Given these weights, we then simply perform weighted averaging to compute the RDS embedding:

We provide a basic visualization of RDS+ in Figure 6.

## I Visualization of Selected Samples

We visualize what samples get selected when selecting 326,000 samples using RDS+ from the TÜLU 2 unfiltered pool in Figure 7, and what samples get selected when selecting with IFD, Top-PPL, random (unbalanced), and RDS+ from the TÜLU 2 unfiltered pool in Figure 8. We provide the raw per-source counts used to construct these figures in Tables 10 and 11 respectively.

L

embedding =

wihi

i=1

Where hi is the last layer hidden state of the ith token. The rationale for this weighting is due to

[Figure 1]

###### Figure 6: Overview of RDS+ data selection. A pretrained LM encodes a data pool and a set of query points via weighted mean-pooling. We then compute cosine similarity between each point in the pool and the query set, and pick the top-K most similar points using Algorithm 1.

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

300000

code_alpaca

250000

cot

dolly

flan_v2

200000

#Samples

gpt4_alpaca

lima

oasst1

150000

open_orca

science

sharegpt wizardlm

100000

50000

0

Random MMLU GSM8k BBH TydiQA Codex Squad AlpacaEval Round-Robin

###### Figure 7: Breakdown of what data gets selected when selecting 326,000 samples using RDS from the TÜLU 2 unfiltered pool. ‘Random’ represents the samples chosen when randomly downsampling to 326,000 samples, and ‘round-robin’ refers to the samples selected by the multi-task round-robin selection.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.0

0.8

ProportionofSamples

0.6

0.4

0.2

0.0

Random IFD10k IFD326k TopPPL10k TopPPL326k RDS+326k

code_alpaca

cot

dolly

flan_v2

gpt4_alpaca

lima

oasst1

open_orca

science

sharegpt wizardlm

- Figure 8: Breakdown of what data gets selected when selecting 10,000 or 326,000 samples using RDS from the TÜLU 2 unfiltered pool using various selection methods. Sample counts normalized to add to 1. ‘Random’ represents the samples chosen when randomly downsampling to 326,000 samples. IFD has a clear bias to ShareGPT data at both sizes, while PPL has a clear bias to FLAN data.

Dataset AlpacaEval BBH Codex GSM8k MMLU SQuAD TyDiQA Round-robin Random code_alpaca 2386 38 12231 28 963 32 0 9084 1126 cot 69859 121838 74385 144253 81532 8349 7482 50481 22396 dolly 272 35 63 68 103 818 0 288 862 flan_v2 60381 21098 45531 36379 59818 64291 72724 67275 53851 gpt4_alpaca 4189 677 3609 1391 2377 562 48 3389 2864 hard_coded 0 0 0 0 0 0 0 0 1 lima 46 6 50 14 26 49 1 59 56 oasst1 222 12 199 22 48 59 416 545 439 open_orca 168766 180159 147830 139379 173252 250736 239301 170031 230329 science 0 1 528 0 173 95 0 9 440 sharegpt 8300 440 12248 869 2324 656 5879 11140 5568 wizardlm 11579 1696 29326 3597 5384 353 149 13699 8068

- Table 10: Raw dataset counts used to create Figure 7, showing the breakdown of what data gets selected when selecting 326,000 samples using RDS from the TÜLU 2 unfiltered pool.

Dataset Random IFD 10k IFD 326k Top PPL 10k Top PPL 326k RDS+ 326k flan_v2 961322 4559 53693 9305 138898 67275 cot 398439 50 29025 46 31289 50481 oasst1 7707 177 5476 27 293 545 dolly 15007 172 4382 127 1751 288 gpt4_alpaca 52002 1 19914 96 3964 3389 code_alpaca 20022 3 1362 15 854 9084 sharegpt 100054 4606 36582 109 597 11140 lima 1030 234 861 1 32 59 wizardlm 142802 11 28618 58 1699 13699 open_orca 4111858 187 146064 216 146622 170031 science 3684 0 23 0 1 9

- Table 11: Raw dataset counts used to create Figure 8, showing the breakdown of what data gets selected when selecting 10,000 or 326,000 samples using RDS from the TÜLU 2 unfiltered pool using various selection methods.

