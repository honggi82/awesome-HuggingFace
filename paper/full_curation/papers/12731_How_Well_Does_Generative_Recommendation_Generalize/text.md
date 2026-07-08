# arXiv:2603.19809v1[cs.IR]20Mar2026

## How Well Does Generative Recommendation Generalize?

Yijie Ding1, Zitian Guo2, Jiacheng Li3, Letian Peng2, Shuai Shao3, Wei Shao3, Xiaoqiang Luo3, Luke Simon3, Jingbo Shang2, Julian McAuley2, Yupeng Hou2

1Carnegie Mellon University, 2University of California San Diego, 3Meta

A widely held hypothesis for why generative recommendation (GR) models outperform conventional item ID-based models is that they generalize better. However, there is few systematic way to verify this hypothesis beyond a superficial comparison of overall performance. To address this gap, we categorize each data instance based on the specific capability required for a correct prediction: either memorization (reusing item transition patterns observed during training) or generalization (composing known patterns to predict unseen item transitions). Extensive experiments show that GR models perform better on instances that require generalization, whereas item ID-based models perform better when memorization is more important. To explain this divergence, we shift the analysis from the item level to the token level and show that what appears to be item-level generalization often reduces to token-level memorization for GR models. Finally, we show that the two paradigms are complementary. We propose a simple memorization-aware indicator that adaptively combines them on a per-instance basis, leading to improved overall recommendation performance.

Date: March 23, 2026 Correspondence: Yupeng Hou at yphou@ucsd.edu Code: https://github.com/Jamesding000/MemGen-GR

1 Introduction

Generative recommendation (GR) (Rajput et al., 2023; Zheng et al., 2024; Deng et al., 2025; He et al., 2025) has recently emerged as a promising paradigm for sequential recommendation. Compared with conventional models such as SASRec (Kang and McAuley, 2018), a key difference is that GR models tokenize each item as a sequence of sub-item tokens (e.g., semantic IDs (Tay et al., 2022; Rajput et al., 2023)) rather than a single unique item ID. However, the advantage of GR models has typically been observed in terms of overall performance, that is, GR models correctly predict more data instances than conventional methods (Rajput et al., 2023; Deng et al., 2025). This naturally raises the question of which types of data instances are better handled by generative recommendation models.

We hypothesize that each data instance requires different levels of generalization and memorization for correct prediction, leading to the performance discrepancies observed between GR and item ID-based models. To investigate this, we propose an analytical framework that categorizes each data instance by the primary model capability it requires (either memorization or generalization) based on the underlying data patterns. We then analyze model performance on each category separately. Nevertheless, conducting such analyses requires two key components: identifying the data patterns of interest in the context of sequential recommendation, and designing reasonable methods to categorize instances.

(1) Data patterns. Since the task is framed as predicting the next item from a user’s history, a natural starting point is to focus on the target items. While prior work often studies cold-start items (i.e., items that are rare or unseen during training) as out-of-distribution cases that require generalization (Singh et al., 2024; Yang et al., 2025; Ding et al., 2026), this target-centric view ignores the interaction between the history and the target. Even when a target item is popular, the transition from the given history to that item may be rare in the training data. Predicting such transitions can therefore still require generalization.

The experiments in this work were conducted using computing resources provided by UC San Diego.

Generalization (1-hop)

Memorization

The ability to recall a seen 1-hop transition during inference The ability to infer new relationships beyond memorized transitions during inference

###### Symmetry Transitivity Second-order Symmetry

- 1. The model learns the transition pattern from training data
- 2. The model predicts the target item based on learned pattern

Training:

Training:

Training:

Training:

...

...

Inference:

Inference:

Inference:

Inference:

...

A model that can memorize transitions in training data is able to solve this task

Infer the reverse relationship

Composite knowledge of more than one learned transition patterns (potentially cross users)

|... items users observed transitions model inference<br><br>|
|---|

Figure 1 Illustrated definitions for memorization vs. generalization. We define memorization and different sub-categories of generalization based on (1) the transition patterns observed in training data, and (2) the patterns required to infer.

(2) Categorization. We require a principled method to determine whether a given instance primarily relies on memorization or generalization. Prior studies, such as those based on counterfactual memorization (Zhang et al., 2023; Grosse et al., 2023; Raunak et al., 2021; Ghosh et al., 2025), are usually computationally expensive,

- as they require frequent model retraining on datasets that exclude specific data points. This makes them impractical for recommendation settings with large-scale user interaction logs (Deng et al., 2025; Zhai et al.,

- 2024). Another line of work categorizes instances by measuring representation similarities between training instances and the predictions (Ivison et al., 2025; Pezeshkpour et al., 2021; Pruthi et al., 2020). However, these methods are mainly adopted in tasks without a clear ground truth, such as language modeling. In contrast, recommendation is typically evaluated with a well-defined ground-truth target item for each instance (Kang and McAuley, 2018; He et al., 2017).

Given these considerations, we treat item transitions (from a historical item to the target) rather than single target items as the data patterns of interest (Figure 1). To categorize data instances, we examine whether the item transitions required for the correct prediction have been observed in the training data (memorization), or if they can be composed or inferred from observed patterns (generalization). Using this categorization, we explicitly partition the test data into subsets reflecting different capabilities and evaluate model performance on each, thereby distinguishing the contributions of memorization and generalization to overall performance.

To this end, we benchmark two representative models for each paradigm: TIGER (Rajput et al., 2023) as the semantic ID-based GR model and SASRec (Kang and McAuley, 2018) as the item ID-based conventional model. By evaluating performance on memorization and generalization subsets across seven real-world datasets, we find that GR models indeed excel on generalization-related subsets, while generally underperforming item ID-based models on memorization-related subsets. This observation motivates us to investigate the mechanism behind the generalization capability of GR models. We then shift our analysis from item transition patterns to sub-item token transition patterns. From this perspective, a substantial fraction of target item transitions that would be regarded as item-level generalization can instead be interpreted as token-level memorization. This effectively explains the source of the GR models’ generalization capability.

Finally, we show that these two paradigms are complementary. We introduce an adaptive ensembling method that combines a GR model with an item ID-based model. The ensemble assigns instance-specific weights based on whether each data instance primarily requires memorization or generalization, as predicted by an indicator. Experimental results show that this adaptive ensembling strategy consistently improves overall performance over both individual models and naive fixed-weight ensembles.

- 2 Defining Memorization and Generalization

In this section, we describe our proposed framework for analyzing memorization and generalization in sequential recommendation. We first outline the task definition and notation in Section 2.1. In what follows,

we present our criteria for attributing data instances as memorization-based (Section 2.2) or generalizationbased (Section 2.3), relying on the item transition patterns they contain. Subsequently, we extend the generalization criteria to encompass multi-hop generalization in Section 2.4. Finally, we discuss the remaining uncategorized instances in Section 2.5.

- 2.1 Task Definition

Sequential recommendation. A user is represented by a sequence of historical item interactions u = [i1,i2,...,it−1], where i ∈ I. The goal is to predict the next item it that the user will interact with. The recommendation models are trained on a set of user interaction sequences Dtrain. For a data instance (u,it) not present in the training set, we attribute it to memorization or generalization based on its constituent item transition patterns.

Item transition. As discussed in Section 1, we treat item transitions as the fundamental data patterns for studying memorization and generalization. Specifically, we define an item transition as a directed pair of items [is → it], where is,it ∈ u and s < t. We further define the hop count of the item transition based on the distance between is and it in the user’s history. For example, if s = t − 2, the hop count is 2. Our framework categorizes each data instance (u,it) based on the set of item transitions {[is → it],is ∈ u} it contains.

- 2.2 Memorization-Related Data Instance

We define a data instance as memorization-related if the 1-hop item transition [it−1 → it] has been observed in the training data, regardless of which user’s history it appears in. Under this condition, it’s possible for a model to correctly predict the target item only by memorizing the training data.

Def. (Memorization). A data instance (u,it) is memorization-related if:

(u,it) ∈ Dmem ⇐⇒ ∃u′ ∈ Dtrain s.t. [it−1 → it] ⊆ u′.

- 2.3 Generalization-Related Data Instance

A data instance is defined as generalization-related if: (1) it is not memorization-related; and (2) it contains

- at least one item transition that can be inferred or composed from observed transitions in the training data. We categorize generalization into multiple types based on specific inference or composition methods. Note that a single data instance may satisfy multiple generalization types.

For simplicity, we first focus on 1-hop item transitions [is → it] where s = t − 1, introducing three possible 1-hop generalization types: transitivity, symmetry, and 2nd-order symmetry.

Transitivity implies that the model can infer [it−1 → it] by bridging two observed transitions via an intermediate item x.

Def. (Transitivity). A data instance (u,it) ∈/ Dmem satisfies transitivity if:

∃x ∈ I,∃u′,u′′ ∈ Dtrain,[it−1 → x] ⊆ u′ ∧ [x → it] ⊆ u′′.

Symmetry allows the model to infer a transition [it−1 → it] if its reverse [it → it−1] has been observed. Def. (Symmetry). A data instance (u,it) ∈/ Dmem satisfies symmetry if:

∃u′ ∈ Dtrain,[it → it−1] ⊆ u′.

2nd-OrderSymmetry encompasses complex symmetric relations where it−1 and it are related via an intermediate item x in non-transitive ways.

Def. (2nd-Order Symmetry). A data instance (u,it) ∈/ Dmem satisfies 2nd-order symmetry if ∃x ∈ I,∃u′,u′′ ∈ Dtrain such that one of the following holds:

- (1) [x → it−1] ⊆ u′ ∧ [x → it] ⊆ u′′ (common cause);
- (2) [it−1 → x] ⊆ u′ ∧ [it → x] ⊆ u′′ (common effect);
- (3) [it → x] ⊆ u′ ∧ [x → it−1] ⊆ u′′ (reverse path).

- 2.4 Multi-Hop Generalization

Training:

Inference:

- 1. The model learns the k-hop skip transitions from training data

Substitutability (k-hop) Symmetry (k-hop)

...

...

Applies to other generalization patterns ...

...

...

Training:

Inference:

- 2. The model bypass unecessary intermediate items to apply learned khop patterns during inference (a form of generalization).

Figure 2 Illustration of multi-hop generalization.

Although Section 2.3 focuses on 1-hop item transitions for simplicity, our proposed criteria naturally extend to multi-hop transitions. Specifically, we define multi-hop generalization types (transitivity, symmetry, and 2nd-order symmetry) by applying the same logic to multi-hop item transitions (as defined in Section 2.1). If a data instance involves multiple item transitions with different hop counts that satisfy the generalization criteria, we use the minimum hop count for categorization.

Substitutability. Beyond the types introduced in Section 2.3, one might consider extending the definition of memorization to multi-hop item transitions. However, we argue that “multi-hop memorization” is effectively a form of generalization. It requires the model to have strong generalization capabilities to bypass unnecessary intermediate items and select the appropriate multi-hop transition for prediction. Therefore, we define substitutability as a unique generalization type involving only multi-hop item transitions.

Def. (Substitutability). A data instance (u,it) ∈/ Dmem satisfies substitutability if:

∃k ≥ 2,∃u′ ∈ Dtrain s.t. [it−k → ··· → it] ⊆ u′.

- 2.5 Uncategorized Data Instance

Given a maximum hop count (e.g., 4 in our experiments, see Section 3), any data instance that is neither memorization-related nor generalization-related is labeled as “uncategorized.” Such instances may involve items unseen during training, exhibit higher-order transition patterns, require capabilities beyond the scope of memorization and generalization, or be inherently unpredictable based on historical data alone. In our experiments, we also analyze model performance on these uncategorized instances.

### 3 Performance Breakdown: Item IDs vs. Semantic IDs

In this section, we present the empirical results for memorization and generalization capabilities for GR and item ID-based models.

- 3.1 Experiment Setup

Datasets. We conduct experiments on seven public datasets that are widely used in evaluating GR models (Rajput et al., 2023; Liu et al., 2025a; Yang et al., 2025; Wang et al., 2024): “Sports and Outdoors” (Sports) and “Beauty” (Beauty) from the Amazon Reviews 2014 collection (McAuley et al., 2015); “Industrial and Scientific” (Science), “Musical Instruments” (Music), and “Office Products” (Office) from the Amazon 2023 collection (Hou et al., 2024); Steam (Kang and McAuley, 2018) and Yelp0. The statistics of processed

0https://business.yelp.com/data/resources/open-dataset/

##### Table 1 Performance breakdown by memorization or generalization categories. We report NDCG@10 for each dataset. Bold indicates the best performing model. “Mem.” stands for “memorization”, and “UC” stands for “uncategorized”. Note that a single data instance may fall into multiple generalization categories, so the sum of the ratios of the generalization subsets can exceed that of “All”. In contrast, the proportions of memorization, generalization, and uncategorized instances sum to 100%.

Generalization

Substitutability Symmetry Transitivity 2nd-Symmetry 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4

Dataset Model Mem.

UC All

###### Ratio 5.1 84.7 5.4 4.7 4.2 1.4 3.3 3.6 3.4 15.1 26.1 18.0 11.1 25.1 32.9 15.8 7.9 10.2 SASRec .2816 .0128 .0732 .0446 .0279 .0767 .0556 .0414 .0368 .0433 .0120 .0046 .0016 .0321 .0072 .0020 .0012 .0004 TIGER .1656 .0179 .1059 .0650 .0415 .0963 .0762 .0650 .0591 .0499 .0215 .0086 .0023 .0405 .0140 .0015 .0013 .0003

Sports (2014)

###### Ratio 8.6 81.3 6.7 5.5 4.4 2.1 3.6 3.4 2.8 16.2 22.6 16.7 10.8 24.1 30.8 15.2 7.9 10.1 SASRec .3793 .0134 .0814 .0395 .0235 .0773 .0535 .0510 .0391 .0461 .0105 .0044 .0026 .0342 .0076 .0018 .0008 .0001 TIGER .2456 .0210 .1046 .0648 .0381 .1156 .1026 .0696 .0437 .0551 .0235 .0113 .0059 .0468 .0167 .0029 .0015 .0015

Beauty (2014)

###### Ratio 4.0 84.2 4.6 4.1 3.5 2.0 3.7 3.5 2.9 16.5 24.8 17.2 10.8 26.3 30.7 15.4 8.3 11.8 SASRec .2452 .0132 .0902 .0510 .0339 .1073 .0722 .0436 .0346 .0435 .0118 .0039 .0021 .0329 .0070 .0014 .0009 .0002 TIGER .2348 .0177 .1114 .0688 .0447 .1033 .1040 .0619 .0459 .0512 .0189 .0070 .0037 .0400 .0123 .0030 .0011 .0003

Science (2023)

###### Ratio 6.9 89.8 7.5 6.4 5.6 3.1 6.3 5.9 5.0 30.4 30.3 15.0 7.1 43.3 30.7 10.4 4.2 3.3 SASRec .1973 .0173 .0818 .0513 .0319 .0920 .0678 .0502 .0294 .0394 .0095 .0033 .0013 .0314 .0057 .0011 .0011 .0000 TIGER .2020 .0205 .0976 .0574 .0376 .0879 .0814 .0581 .0396 .0434 .0144 .0043 .0014 .0359 .0087 .0014 .0007 .0004

Music (2023)

###### Ratio 4.4 86.3 4.5 4.1 3.6 1.9 3.7 3.4 2.9 22.4 27.4 16.4 9.5 33.1 30.1 13.5 6.8 9.3 SASRec .2405 .0097 .0682 .0412 .0282 .0729 .0612 .0398 .0273 .0277 .0062 .0019 .0012 .0212 .0037 .0012 .0006 .0002 TIGER .2719 .0154 .1092 .0667 .0433 .0914 .1025 .0649 .0470 .0384 .0136 .0040 .0020 .0311 .0089 .0017 .0007 .0001

Office (2023)

###### Ratio 37.4 61.5 18.7 11.9 7.1 7.5 12.7 9.3 6.0 51.0 8.0 1.6 0.5 54.9 5.2 0.9 0.4 1.1 SASRec .3855 .0133 .0271 .0129 .0092 .0306 .0216 .0139 .0091 .0144 .0079 .0079 .0112 .0143 .0057 .0018 .0051 .0000 TIGER .3885 .0157 .0366 .0149 .0073 .0305 .0331 .0169 .0109 .0175 .0079 .0038 .0000 .0168 .0077 .0007 .0000 .0000

Steam

Ratio 4.2 92.8 6.4 7.8 6.9 2.7 7.1 6.5 5.8 22.6 33.5 18.5 9.7 38.7 36.8 12.0 4.4 3.0 SASRec .2487 .0233 .0858 .0714 .0435 .0724 .0855 .0484 .0300 .0436 .0215 .0133 .0139 .0439 .0120 .0016 .0010 .0007 TIGER .1402 .0213 .0898 .0620 .0445 .0659 .0861 .0479 .0389 .0436 .0213 .0112 .0055 .0356 .0150 .0034 .0021 .0000

Yelp

datasets have been reported in Table 2. We adopt the standard leave-last-out data split, using the last and second-to-last items of each sequence for testing and validation, respectively.

Models. We benchmark two models: TIGER (Rajput et al., 2023), representing the generative recommendation paradigm, and SASRec (Kang and McAuley, 2018), representing the conventional sequential recommendation paradigm. Note that, for a fair comparison, we optimize the SASRec model using cross-entropy loss and treat all items as negative samples, following Liu et al. (2025a), rather than sampling a single negative item per instance as in Rajput et al. (2023).

Table 2 Dataset statistics.

Dataset Users Items Interactions Sparsity AvgLen

Sports 35,599 18,358 296,337 99.95% 8.32 Beauty 22,364 12,102 198,502 99.93% 8.88 Science 50,986 25,849 412,947 99.97% 8.10 Music 57,440 24,588 511,836 99.96% 8.91 Office 223,309 77,552 1,800,878 99.99% 8.06 Steam 47,762 10,972 599,620 99.89% 12.55 Yelp 30,432 20,034 316,354 99.95% 10.40

Implementation details. We fine-tune the learning rate over {1e-3,3e-3} and train for a maximum of 150 epochs with early stopping. The checkpoint achieving the best validation performance is selected for testing.

Data categorization. We partition test instances into: memorization, generalization (if memorization is not satisfied), and uncategorized (if none of the above is satisfied). Note that the above three categories are mutually exclusive. However, a data instance may exhibit multiple generalization types, each associated with several possible hop distances. Following Occam’s razor, we annotate an instance with all applicable types but retain only the minimum hop distance for each type.

- 3.2 Performance Analysis

In this section, we analyze the performance comparison between SASRec and TIGER, broken down by the memorization and generalization categories defined in Section 2.

SASRec memorizes, TIGER generalizes. As illustrated in Table 1, TIGER generally underperforms SASRec on

memorization subsets (e.g., −43.6% on Yelp, −41.2% on Sports, −35.2% on Beauty, and comparable on others), while consistently outperforming SASRec on generalization subsets (e.g., +58.8% on Office, +56.7% on Beauty, and +39.8% on Sports). This trade-off suggests that SASRec relies more on memorizing observed patterns, while TIGER is more effective at composing learned item transitions for generalization. Both models achieve substantially higher performance on memorization than on generalization overall, reflecting the intrinsic difficulty of generalizing beyond observed transitions. Moreover, both models exhibit near-zero performance on the uncategorized subset while achieving reasonable performance on the others. This supports the validity of our data attribution and suggests that the uncategorized instances are indeed difficult to predict, consistent with our hypothesis in Section 2.5.

Generalization categories. Comparing performance across generalization categories, we observe that both models achieve higher performance on Substitutability and Symmetry than on Transitivity and 2nd-Symmetry. We attribute this to differences in the difficulty of the various generalization types. Substitutability and Symmetry require induction from only a single training example, whereas Transitivity and second-order Symmetry require composing knowledge from multiple examples, representing a structurally more complex form of generalization.

Generalization hops. Within each generalization category, both models perform monotonically worse as hop distance increases. This shows that nearby item transitions pose a stronger influence than distant ones. In low-hop settings, SASRec can sometimes outperform TIGER. But its performance drops faster as the hop distance grows. The decline is even sharper in more difficult categories such as Transitivity and 2nd-Symmetry. This suggests that SASRec mainly generalizes over local context, while TIGER remains more robust for longer-hop generalization.

Data ratio analysis. Finally, we examine the proportion of test instances in each category. In all datasets, memorization cases form a much smaller portion than generalization cases. This suggests that pure memorization is limited, and effective recommendation requires substantial generalization capability. Among the generalization categories, most instances require combining information from multiple training examples, whereas only a small fraction can be inferred from a single training instance (Substitutability and Symmetry). Uncategorized instances consistently account for less than 10% of the data, indicating that most test transitions can be explained by other categories.

### 4 Mechanism Analysis: A Token-Level Lens

Semantic ID-based GR models generally outperform item ID-based models on generalization-related subsets (Section 3). This raises a question: Why does GR generalize better yet memorize worse than item ID-based models? In this section, we investigate the underlying mechanisms of GR models through a token-level lens.

We first introduce the concept of prefix n-gram memorization (Section 4.1), and demonstrate that item-level generalization can often be interpreted as token-level memorization within the semantic ID space (Section 4.2). Next, we characterize models’ behavior using this new lens (Section 4.3), and find that: (1) GR generalization performance improves when the underlying token transitions are more frequently observed in the training data. (2) Different item transitions can share the same memorized prefix, which can decrease GR’s ability to memorize a specific item transition. Finally, to further validate our hypothesis, we design a controlled study to vary the token memorization ratio and measure its direct impact on the generalization-memorization trade-off (Section 4.4).

- 4.1 Prefix N-Gram Memorization

Motivation. Unlike item ID-based models, GR models represent items as sequences of discrete semantic ID tokens shared across items. This allows the model to anchor predictions on sub-item level transition patterns. However, quantifying memorization behavior at a token level is non-trivial. Directly attributing the effect of a single token on another token is difficult because token-to-token correlations are dense and highly dependent on the contexts (Grosse et al., 2023). For LLMs, people often assess memorization via n-gram correlation

between context and target text, reflecting the model’s ability to memorize the n-gram ‘knowledge’ and generate the corresponding n-gram ‘answer’ (Liu et al., 2024b; Wang et al., 2025b). Drawing inspiration from this, we propose quantifying memorization in GR by considering the prefix n-grams of the context-target item pair. Since semantic IDs encode hierarchical (coarse-to-fine) semantic information, focusing on transitions from one prefix to another prefix captures the most prominent semantic dependencies (Figure 3).

Token Prefix. Let tok(i) = [z1,z2,...,zL] denote the semantic-ID tokenization of item i, where L is the number of tokens in the semantic ID. For a prefix length n ≤ L, define the n-gram prefix operator prefn(i) ≜ [z1,...,zn].

Prefix n-gram memorization. We define token-level memorization by considering only the semantic ID prefixes of items in the transitions. A test instance is considered n-gram prefix-memorizable if the first n tokens (the n-gram prefix) of both items in the target transition [is → it] occur in the training set, even when the exact items differ.

###### Item ID-based Models

###### Semantic ID-based GR

Training:

(t2, t3, t3) 1.GR model learns

(t1, t4, t2)

prefix transitions at the token-level

- (t1, t4, t3) (t1, t5, t8)

- (t2, t3, t4) (t1, t5, t9)

Tokenization

Inference:

2. GR can leverage learned prefix transitions to solve item generation tasks

Recall learned patterns

(t2, t3, • ) (t1, t5, • )

[Figure 1]

[Figure 2]

| |
|---|

Item-level transitivity 2-gram prefix memorization

Figure 3 Illustration of how item-level generalization can be reduced to token-level memorization for GR models.

Def. (1-hop Prefix N-Gram Memorization). A data instance (u,it) satisfies 1-hop prefix n-gram memorization if:

∃u′ ∈ Dtrain, ∃s ≥ 2 s.t. [js−1 → js] ⊆ u′, prefn(it−1) = prefn(js−1), prefn(it) = prefn(js).

Analogous to the multi-hop generation framework, this definition naturally extends to k-hop transitions. Notably, when k = 1, token prefix memorization can be viewed as a relaxed form of memorization, whereas for k > 1, it is analogous to the definition of substitutability (see Section 2.4). For the following sections, we refer to prefix n-gram memorization as token memorization for brevity.

- 4.2 From Item Generalization to Token Memorization

We examine how item-level generalization can be reduced to token memorization for GR models, following the definition in Section 4.1. Unless otherwise specified, the following experiments on all datasets aggregate all token memorizations with hop ≤ 4, and utilize a 256 × 3 semantic ID quantization followed by one identifier token, consistent with Rajput et al. (2023).

ItemgeneralizationinstancesoftenreducetotokenmemorizationforGRmodels. Figure 4 illustrates the reduction of item-level generalization categories into token-level prefix-gram memorization. We observe that a non-trivial fraction of instances reduce to 1-, 2-, and 3-gram prefix memorization. For example, on average, more than

- 5% of item-level generalization transitions (symmetry, transitivity, and 2nd-symmetry) can also be explained

Symmetry Transitivity

2nd-Sym. Uncategorized

Sports

|46|.6 5.|9 20|.5 27|.1 0.|0|
|---|---|---|---|---|---|
|20|.2 4.|0 22|.3 53|.5 0.|0|
|17|.5 3.|7 21|.6 57|.3 0.|0|
|0.|0 1.|1 13|.6 85|.2 0.|1|
| | | | | | |

4 3 2 1 0

Beauty

Office

51.9 4.9 15.7 27.5 0.0

51.6 11.8 26.8 9.8 0.0

24.8 3.8 17.5 53.8 0.1

16.1 9.6 43.9 30.4 0.0

21.2 3.5 16.8 58.4 0.1

14.6 9.1 44.0 32.3 0.0

0.0 1.7 9.6 88.3 0.4

0.0 4.0 43.3 52.7 0.0

4 3 2 1 0

4 3 2 1 0

Token Memorization N-gram Length

Yelp

48.1 3.7 26.4 21.9 0.0

24.9 3.9 31.5 39.7 0.0

22.9 3.8 30.6 42.7 0.0

0.0 0.9 13.9 85.0 0.2

4 3 2 1 0

100

[Figure 3]

Percentage(%)

75

50

25

0

##### Figure 4 Token memorization ratios for each item-level generalization category. The X-axis represents the prefix length for token memorization.

Beauty

Office

1e4

1e5

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

- 1-gram

| |
|---|

- 2-gram

| |
|---|

- 3-gram

0.4

0.2

1.5

1.5

0.3

TIGER

NDCG@10

Count

SASRec

1.0

1.0

0.2

0.1

0.5

0.1

0.5

0.0

0.0

0 1 4 9 23 611

0 2 17 58 138 3948

Prefix Transition Count

Prefix Transition Count

- Figure 5 Test NDCG@10 against prefix transition counts on 2 datasets. Transition counts are grouped by quantiles.

Sports

Beauty

Office

ItemTransitionProbablity(ϕ)

ΔNDCG(TIGER-SASRec)

1.00

1.00

1.00

|-.396|-.390|-.368|-.283|
|---|---|---|---|
|-.074|-.126|-.078|-.075|
|-.085|-.037|-.078|.021|
|-.018|-.002|.058|.211|

|-.406|-.526|-.471|-.303|
|---|---|---|---|
|-.098|-.164|-.144|-.108|
|-.095|-.039|.005|.119|
|-.010|-.033|.068|.037|

|-.129|-.098|-.039|.058|
|---|---|---|---|
|.004|.056|.076|.113|
|-.015|.006|.068|.099|
|.011|.019|.028|.042|

[Figure 10]

0.4

0.11

0.16

0.06

0.2

0.04

0.07

0.02

0.0

−0.2

0.02

0.03

0.01

−0.4

0.00

0.00

0.00

0.01 0.03 0.06 0.75

0.02 0.04 0.10 1.00

0.00 0.01 0.04 1.00

Prefix Transition Probability (ψ)

Figure 6 TIGER Advantage breakdown by prefix transition probability and item transition probability. Both axes are grouped by quantiles.

as 3-hop prefix memorization. Notably, the vast majority of test instances (> 99%) across all item-level categories admit at least 1-gram prefix memorization. This demonstrates that for many test instances where the item-level transition is unseen, the training set nevertheless contains matching prefix transitions, allowing the model to leverage prefix memorization for inference.

Token memorization ratio reflects item-level difficulty. Across categories, symmetry exhibits a higher ratio of 4-gram memorization, largely due to its overlap with item-level substitutability. In contrast, transitivity and 2nd-order symmetry mostly reduces to short prefix memorization (2–3 grams), yielding weaker prefix-transition support from training and making these tasks harder. Furthermore, uncategorized instances reduce almost exclusively to 1-gram memorization, representing the weakest form of prefix-transition support. Overall, these findings show that the ratio of token memorization directly reflects the item-level task difficulty and the model performance trends observed in Section 3.

- 4.3 Explaining Performance Trade-off via Token Memorization

Having established that item-level generalization often reduces to token-level memorization, we now investigate whether this mechanism explains the performance trade-off: GR generalizes better but memorizes worse at item level. We categorize test instances through token memorization and show that token memorization enables better generalization (Section 4.3.1), and token memorization dilutes item memorization (Section 4.3.2).

- 4.3.1 Token Memorization Enables Better Generalization

To analyze how TIGER’s performance correlates with token memorization support at a finer granularity, we quantify prefix support as the count of prefix-transition occurrences in the training data:

Cn(it−k,it) = C(prefn(it−k) → prefn(it)),

where C(·) is the count function. Given an item-level generalization instance (u,it), the token memorization support Cn(u,it) = Kk=1 Cn(it−k,it) is the sum of prefix supports of all the k-hop transitions to the target item.

More token memorization support correlates with better generalization. We aggregate token memorization support for hop k ≤ 4, and plot the TIGER and SASRec models’ NDCG@10 on all item generalization instances (without item memorization support) against Cn(u,it) in Figure 5. First, we observe that supported instances consistently achieve substantially higher NDCG@10 than non-supported instances for both TIGER and SASRec. TIGER’s gain over SASRec is substantially larger on prefix-supported instances, indicating that TIGER benefits from prefix-transition support that SASRec does not explicitly model.Furthermore, TIGER’s performance gap increases with both increasing support count Cn(u,it) and prefix length n, indicating that more token memorization support correlates with better generalization performance.

- 4.3.2 Token Memorization Dilutes Item Memorization

#### Conversely, we investigate why this same mechanism might degrade performance on item memorization tasks. We hypothesize a dilution effect: while SASRec directly optimizes the specific item transition it−1 → it,

##### Table 3 Experiment configurations and token memorization ratio across different semantic ID (SID) configurations.

SID Configuration Budget Token Memorization Ratio (%)

Len (L) Size (V ) TFLOPs n = 5 n = 4 n = 3 n = 2 n = 1 n = 0

- L = 2

1024 5.00 × 1015 – – – 15.06 84.47 0.47 4096 5.00 × 1015 – – – 15.06 83.93 1.01

- L = 3

256 7.50 × 1015 – – 15.06 16.57 68.36 0.01 4096 7.50 × 1015 – – 15.06 5.40 79.54 0.01

- L = 4

64 1.00 × 1016 – 15.06 7.68 46.26 31.01 0.00 1024 1.00 × 1016 – 15.06 0.92 21.34 62.68 0.00

- L = 5

64 1.33 × 1016 15.06 1.22 5.74 46.08 31.89 0.00 256 1.33 × 1016 15.06 0.20 1.09 16.69 66.96 0.00

[Figure 11]

Figure 7 Left/Middle: Val NDCG@10 training dynamics on the Memorization and Generalization subsets. Right: Test NDCG@10 for different codebook configurations.

TIGER predicts through prefix transitions that are shared by many items. Thus, its ability to memorize specific item transitions can be diluted. To verify this, we evaluate TIGER’s relative performance (∆NDCG) against two metrics: item transition probability (ϕ), the probability of predicting target item based on all item transitions, and prefix transition probability (ψ), the probability of predicting target item’s prefix based on all prefix transitions. For a transition it−1 → it, these quantities are estimated empirically by:

C(prefn(it−1) → prefn(it)) C(prefn(it−1) → ·)

C(it−1 → it) C(it−1 → ·)

ϕ =

, ψ =

,

where C is the count function and C(x → ·) is the total number of outgoing transitions from the context x.

Token memorization dilutes item memorization. We visualize how TIGER’s NDCG gain over SASRec (i.e., ∆NDCG) breaks down by item transition probability ϕ and prefix transition probability ψ. As shown in Figure 6, across all datasets, we observe significant NDCG loss when instances have high ϕ but low ψ. In this case, TIGER spreads probability mass across many items sharing the same prefix transition, causing difficulty for the model to memorize a specific item transition pattern. Conversely, TIGER can even outperform SASRec when ψ is high, indicating that TIGER’s item memorization performance is strong only when token memorization aligns. Across the datasets, token memorization usually leads to more dilution cases, causing TIGER to underperform SASRec on memorization tasks.

- 4.4 Mechanism Validation

In this section, we test our hypothesis that increasing token memorization ratio improves item-level generalization but can reduce item-level memorization. Our goal is to vary the token memorization ratio and measure the resulting changes in performance.

Experiment Setup. To manipulate the token memorization ratio in a systematic way, we vary the codebook size. The intuition is that denser codebooks (smaller codebook sizes V ) induce more prefix sharing across items, which increases the total number of token memorization instances. We consider SID lengths L ∈ {2,3,4,5} (including identifier tokens), and for each fixed L we evaluate two codebook sizes V . In Table 3 we report the resulting token memorization ratios at each prefix length; as expected, smaller V yields a larger fraction of instances with prefix-transition support.

Controlled factors. We only compare configurations within the same SID length L, since optimizing different lengths can have different difficulty and longer SID length often leads to optimization issues (Hou et al.,

- 2025a). We also fix the recommendation model size and match the training compute budget within each L. For each L, we estimate the training compute to ensure convergence, defined as both memorization and generalization metrics no longer increasing on the validation set. We report (i) validation training dynamics of NDCG@10 and (ii) final test NDCG@10 for each configuration.

Smaller codebook sizes lead to better generalization but worse memorization. From Figure 7, we observe a consistent pattern: smaller codebook size V (denser codebooks, higher token memorization ratio) leads to

better item-level generalization, but degraded memorization performance. Averaged across settings, the smaller V improves generalization by +10.24% relative compared to the larger variant, while memorization decreases by −7.62% relative. This provides consistent evidence across all tested codebook sizes that increasing token memorization ratio improves generalization while degrading memorization.

Regularization effect in training dynamics. From Figure 7, we further observe that for large V , generalization performance peaks early and then degrades as training proceeds. In contrast, for small V , generalization remains stable or continues to improve until convergence. This suggests that denser tokenization has a stronger data-level regularization effect. It encourages TIGER to rely more on shared prefix-transition structure, rather than overfitting to noisier, item-specific transitions.

### 5 Memorization-Aware Adaptive Ensemble

So far, we have shown that GR models excel on instances requiring generalization, whereas item ID-based models perform better on memorization. Motivated by this observation, we investigate whether it is possible to combine the strengths of both paradigms. We propose a hybrid framework that dynamically weights the predictions of a GR model and an item-based model for each given input, depending on the primary capability the input may require. We first introduce a simple, training-free indicator to estimate the likelihood of memorization (Section 5.1), followed by an empirical validation of their correlation with ground-truth categories (Section 5.2). Finally, we present experimental results demonstrating that this adaptive model ensemble strategy achieves improved overall performance (Section 5.3).

- 5.1 Adaptive Ensemble Indicators

The core idea of our adaptive ensemble strategy is to estimate how likely a data instance can be solved via memorization, and use this estimate to adjust the relative weights between the item ID-based model and the GR model for each instance. A key challenge is that the memorization criterion defined in Section 2.2 cannot be directly applied at inference time, since the target item is unavailable. We therefore introduce a practical indicator that estimates the memorization likelihood of a data instance (u,it) and converts it into an ensemble weight.

Confidence-based indicator. The intuition behind this approach is that memorization-related instances tend to lie closer to the training data distribution. As a result, an item ID-based model that effectively memorizes training patterns should produce more confident predictions on such instances. We therefore use the prediction confidence of the item ID-based model as a proxy for memorization likelihood. While confidence is often defined as the distance between prediction logits and a uniform distribution, the recommendation task typically involves a very large item space where most items receive negligible probability. We thus adopt the maximum softmax probability (MSP) (Vashurin et al., 2025) as a practical confidence score, defined as:

PID(it = j | u), (1)

sConf(u) = max

j∈I

where PID(· | u) denotes the predicted probability distribution of the ID-based model given user history u. We then transform sConf(u) into an ensemble weight that controls the contribution of each model:

α(u) = sigmoid(−q(sConf(u) − τ)), (2) where q and τ are hyperparameters.

- 5.2 Validation of the Indicator

In this section, we empirically validate whether the proposed MSP indicator provides a meaningful estimate of memorization likelihood and can therefore support our adaptive ensemble strategy. Using the same datasets and models as in Section 3, we group data instances into bins according to quantiles of their indicator values. For each bin, we examine both the proportion of memorization-related instances and the performance of the two paradigms.

Sports

Beauty

Science

Office

.06

Memorization%

10

20

| |
|---|

| |
|---|

10

.10

10

.04

NDCG@10

.04

| |
|---|

.04

| |
|---|

5

10

5

| |
|---|

| |
|---|

5

.05

.02

.02

| |
|---|

| |
|---|

| |
|---|

.02

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

0

0

0

0

Q1 Q2 Q3 Q4 Q5

Q1 Q2 Q3 Q4 Q5

Q1 Q2 Q3 Q4 Q5

Q1 Q2 Q3 Q4 Q5

MSP Quantile Memorization % SASRec TIGER

| |
|---|

Figure 8 Correlation between the MSP indicator, proportion of memorization-related instances (bars), and model performance of SASRec and TIGER (lines).

- Table 4 Performance comparison of ensemble strategies. The best performance is bolded.

Metric Method Sports Beauty Scientific Instruments Office Steam Yelp

###### SASRec 0.0253 0.0436 0.0209 0.0291 0.0190 0.1525 0.0321 TIGER 0.0237 0.0383 0.0243 0.0323 0.0254 0.1551 0.0257 Fixed-weight 0.0291 0.0471 0.0260 0.0343 0.0261 0.1579 0.0351 Adaptive 0.0296 0.0476 0.0261 0.0344 0.0264 0.1579 0.0352

N@10

SASRec 0.0318 0.0566 0.0262 0.0319 0.0241 0.1738 0.0362 TIGER 0.0318 0.0542 0.0303 0.0357 0.0306 0.1773 0.0281 Fixed-weight 0.0530 0.0838 0.0483 0.0635 0.0456 0.2008 0.0630 Adaptive 0.0537 0.0841 0.0485 0.0638 0.0461 0.2008 0.0637

R@10

Correlation with memorization. As the indicator value increases, the proportion of memorization-related instances rises monotonically. This trend suggests that the indicator provides a meaningful estimate of memorization likelihood.

Performance crossover. We also observe a performance crossover between the generative model TIGER and the ID-based model SASRec as the indicator value increases. At lower indicator values, TIGER achieves better performance, whereas at higher values SASRec becomes increasingly competitive and may outperform TIGER. This crossover pattern is consistent with our hypothesis: instances with high indicator values tend to be memorization-related and are better handled by ID-based models, while instances with low values favor the stronger generalization capability of GR models.

5.3 Performance of Adaptive Ensemble

Having validated the effectiveness of the proposed MSP indicators, we proceed to evaluate the overall performance of the memorization-aware adaptive model ensemble strategy on real-world datasets.

Experimental settings. We use the same datasets and base models as in Section 3. We use Equation (2) to map the raw MSP values to an instance-specific ensemble weight α(u) ∈ [0,1]. The final prediction is obtained by combining the scores produced by the two base models, SASRec and TIGER, according to this weight. We compare our adaptive ensemble against the individual base models and a Fixed-weight Ensemble baseline, which assigns a global constant weight αstatic to SASRec for all instances. Hyperparameters are tuned on the validation set, with q ∈ {1,5,9,13}, τ ∈ {0,0.1,...,0.5}, and αstatic ∈ {0,0.1,...,1.0}.

Recommendation results. We present the performance comparison in Table 4 and make the following observations:

- • The proposed adaptive ensemble strategy generally outperforms both the individual base models and the fixed-weight ensemble baseline. These results provide empirical evidence that item ID-based models and GR models offer complementary strengths.
- • The improvement over baselines is more prominent on datasets with a stronger model cross-over effect

(see Figure 8). The greater the comparative advantage each base model holds in its respective domain, the more the memorization-aware indicator can leverage this specialization.

- 6 Related Works

Generative recommendation. Unlike conventional sequential recommendation paradigms (Rendle et al., 2010; Kang and McAuley, 2018; Sun et al., 2019; Hidasi et al., 2016; Tang and Wang, 2018; Ma et al., 2019), where each item is typically indexed by a unique ID, generative recommendation tokenizes each item as a sequence of sub-item tokens (Tay et al., 2022; Rajput et al., 2023; Zhai et al., 2024; Deng et al., 2025; He et al., 2025; Zheng et al., 2024; Hou et al., 2025c). A generative model then operates over these token sequences and autoregressively generates tokens that index the next item. Existing work on generative recommendation primarily focuses on developing improved tokenization methods, such as exploring alternative algorithms (Wang et al., 2024; Zhu et al., 2024; Hou et al., 2025a; Hua et al., 2023; Jin et al., 2024; Liu et al., 2025a; Zhang et al., 2025; Zheng et al., 2025b,a) or incorporating additional data sources (Liu et al., 2024c; Hou et al., 2025b; Zhu et al., 2025; Liu et al., 2024a; Wang et al., 2025a; Zhong et al., 2025). Despite their empirical success, there remains a lack of systematic research examining which data instances generative recommendation outperforms conventional models on. Existing work typically attributes the strong performance of GR to semantic-integrated indexing systems (Zhang et al., 2024; Hou et al., 2025b; Ju et al., 2025; Liu et al., 2025b) or finer-grained learning objectives (Wang et al., 2024; Zhu et al., 2024; Hou et al., 2025a). These studies offer design-centric explanations but lack a systematic analysis of how such designs translate into distinct prediction behaviors compared to conventional models. In this work, we conduct an in-depth comparative study of these two paradigms through the lens of memorization versus generalization, specifically aiming to answer the question: do GR models outperform conventional baselines because they generalize better?

Memorization and generalization. The study of model memorization and generalization has long been a central concern in machine learning, spanning from classical machine learning models (Han et al., 2022; Yang et al., 2023; Buchanan et al., 2025) to modern large language models (Carlini et al., 2023; Wang et al., 2025b; Jiang

- et al., 2024). A key challenge of conducting these analyses is how to characterize the influence of training data on model behavior. One line of work approaches this question through counterfactual memorization, which measures the causal effect of removing individual training instances on task performance (Zhang et al., 2023; Grosse et al., 2023; Raunak et al., 2021; Ghosh et al., 2025). However, these methods are often computationally expensive and difficult to scale (Grosse et al., 2023; Pruthi et al., 2020; Xia et al., 2024). Other approaches define generalization as robustness to local input perturbations (e.g., input clues (Djiré
- et al., 2025) or task rules (Xie et al., 2025; Barron and White, 2025; Chu et al., 2025)). In this work, we aim to investigate how generative recommendation models differ from conventional item ID-based models in their memorization and generalization capabilities. Because the recommendation task provides a concrete and well-defined ground-truth target, we categorize each data instance according to the primary capability required to make a correct prediction.

- 7 Conclusion

In this work, we presented a systematic study comparing semantic ID-based generative recommendation models with traditional item ID-based models through the lens of memorization and generalization. We proposed a framework to categorize data instances based on the item transition patterns they contain. Extensive experiments across various datasets revealed that generative recommendation models excel on generalization-related instances, while item ID-based models perform better on memorization-related instances. To explain this, we analyzed the models at the token level, finding that item-level generalization in generative models often reduces to token-level memorization. Finally, we show that the two paradigms are complementary by demonstrating that adaptively adjusting ensemble weights using a memorization-aware indicator leads to improved performance. In future work, we plan to explore advanced tokenization methods that explicitly target the memorization and generalization capabilities characterized in this study.

References

Joshua Barron and Devin White. Too big to think: Capacity, memorization, and generalization in pre-trained transformers. arXiv preprint arXiv:2506.09099, 2025.

Sam Buchanan, Druv Pai, Yi Ma, and Valentin De Bortoli. On the edge of memorization in diffusion models. arXiv preprint arXiv:2508.17689, 2025.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. In ICLR, 2023.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. In ICML, 2025.

Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965, 2025.

Yijie Ding, Jiacheng Li, Julian McAuley, and Yupeng Hou. Inductive generative recommendation via retrieval-based speculation. In AAAI, 2026.

Albérick Euraste Djiré, Abdoul Kader Kaboré, Earl T Barr, Jacques Klein, and Tegawendé F Bissyandé. Memorization or interpolation? detecting llm memorization through input perturbation analysis. arXiv preprint arXiv:2505.03019, 2025.

Bishwamittra Ghosh, Soumi Das, Qinyuan Wu, Mohammad Aflah Khan, Krishna P Gummadi, Evimaria Terzi, and Deepak Garg. Rethinking memorization measures and their implications in large language models. arXiv preprint arXiv:2507.14777, 2025.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296, 2023.

Junlin Han, Huangying Zhan, Jie Hong, Pengfei Fang, Hongdong Li, Lars Petersson, and Ian Reid. What images are more memorable to machines? arXiv preprint arXiv:2211.07625, 2022.

Ruining He, Lukasz Heldt, Lichan Hong, Raghunandan H. Keshavan, Shifan Mao, Nikhil Mehta, Zhengyang Su, Alicia Tsai, Yueqi Wang, Shao-Chuan Wang, Xinyang Yi, Lexi Baugher, Baykal Cakici, Ed H. Chi, Cristos Goodrow, Ningren Han, He Ma, Rómer Rosales, Abby Van Soest, Devansh Tandon, Su-Lin Wu, Weilong Yang, and Yilin Zheng. PLUM: adapting pre-trained language models for industrial-scale generative recommendations. arXiv preprint

arXiv:2510.07784, 2025. Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In WWW, pages 173–182, 2017. Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based recommendations with recurrent neural networks. In ICLR, 2016. Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952, 2024. Yupeng Hou, Jiacheng Li, Ashley Shin, Jinsung Jeon, Abhishek Santhanam, Wei Shao, Kaveh Hassani, Ning Yao, and Julian McAuley. Generating long semantic ids in parallel for recommendation. In KDD, 2025a.

Yupeng Hou, Jianmo Ni, Zhankui He, Noveen Sachdeva, Wang-Cheng Kang, Ed H. Chi, Julian McAuley, and Derek Zhiyuan Cheng. ActionPiece: Contextually tokenizing action sequences for generative recommendation. In ICML, 2025b.

Yupeng Hou, An Zhang, Leheng Sheng, Zhengyi Yang, Xiang Wang, Tat-Seng Chua, and Julian McAuley. Generative recommendation models: Progress and directions. In Companion Proceedings of the ACM Web Conference 2025, 2025c.

Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. How to index item ids for recommendation foundation models. In SIGIR-AP, 2023.

Hamish Ivison, Muru Zhang, Faeze Brahman, Pang Wei Koh, and Pradeep Dasigi. Large-scale data selection for instruction tuning. arXiv preprint arXiv:2503.01807, 2025.

Minhao Jiang, Ken Ziyu Liu, Ming Zhong, Rylan Schaeffer, Siru Ouyang, Jiawei Han, and Sanmi Koyejo. Investigating data contamination for pre-training language models. arXiv preprint arXiv:2401.06059, 2024.

Bowen Jin, Hansi Zeng, Guoyin Wang, Xiusi Chen, Tianxin Wei, Ruirui Li, Zhengyang Wang, Zheng Li, Yang Li, Hanqing Lu, Suhang Wang, Jiawei Han, and Xianfeng Tang. Language models as semantic indexers. In ICML, 2024.

Clark Mingxuan Ju, Liam Collins, Leonardo Neves, Bhuvesh Kumar, Louis Yufeng Wang, Tong Zhao, and Neil Shah.

Generative recommendation with semantic ids: A practitioner’s handbook. In CIKM, pages 6420–6425, 2025. Wang-Cheng Kang and Julian J. McAuley. Self-attentive sequential recommendation. In ICDM, 2018. Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, and Wayne Xin Zhao. Generative recommender with

end-to-end learnable item tokenization. In SIGIR, pages 729–739, 2025a. Han Liu, Yinwei Wei, Xuemeng Song, Weili Guan, Yuan-Fang Li, and Liqiang Nie. Mmgrec: Multimodal generative recommendation with transformer model. arXiv preprint arXiv:2404.16555, 2024a. Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, and Hannaneh Hajishirzi. Infini-gram: Scaling unbounded n-gram language models to a trillion tokens. In COLM, 2024b. Jingzhe Liu, Liam Collins, Jiliang Tang, Tong Zhao, Neil Shah, and Clark Mingxuan Ju. Understanding generative

recommendation with semantic ids from a model-scaling view. arXiv preprint arXiv:2509.25522, 2025b. Zihan Liu, Yupeng Hou, and Julian McAuley. Multi-behavior generative recommendation. In CIKM, 2024c. Chen Ma, Peng Kang, and Xue Liu. Hierarchical gating networks for sequential recommendation. In KDD, 2019. Julian J. McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. Image-based recommendations on

styles and substitutes. In SIGIR, 2015. Pouya Pezeshkpour, Sarthak Jain, Byron C. Wallace, and Sameer Singh. An empirical comparison of instance attribution methods for NLP. In NAACL, pages 967–975, 2021. Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 33:19920–19930, 2020.

Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy. Recommender systems with generative retrieval. In NeurIPS, 2023.

Vikas Raunak, Arul Menezes, and Marcin Junczys-Dowmunt. The curious case of hallucinations in neural machine translation. In NAACL, pages 1172–1183, 2021.

Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. Factorizing personalized markov chains for next-basket recommendation. In WWW, 2010.

Anima Singh, Trung Vu, Nikhil Mehta, Raghunandan Keshavan, Maheswaran Sathiamoorthy, Yilin Zheng, Lichan Hong, Lukasz Heldt, Li Wei, Devansh Tandon, Ed H. Chi, and Xinyang Yi. Better generalization with semantic ids: A case study in ranking for recommendations. In RecSys, 2024.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In CIKM, 2019.

Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In WSDM, 2018.

Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Prakash Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. Transformer memory as a differentiable search index. In NeurIPS, 2022.

Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev, Lyudmila Rvanova, Daniil Vasilev, Akim Tsvigun, Sergey Petrakov, Rui Xing, Abdelrahman Sadallah, Kirill Grishchenkov, et al. Benchmarking uncertainty quantification methods for large language models with lm-polygraph. Transactions of the Association for Computational Linguistics, 13:220–248, 2025.

Dongsheng Wang, Yuxi Huang, Shen Gao, Yifan Wang, Chengrui Huang, and Shuo Shang. Generative next poi recommendation with semantic id. In KDD, 2025a.

Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. Learnable tokenizer for llm-based generative recommendation. In CIKM, 2024.

Xinyi Wang, Antonis Antoniades, Yanai Elazar, Alfonso Amayuelas, Alon Albalak, Kexun Zhang, and William Yang Wang. Generalization v.s. memorization: Tracing language models’ capabilities back to pretraining data. In ICLR, 2025b.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: selecting influential data for targeted instruction tuning. In ICML, pages 54104–54132, 2024.

Chulin Xie, Yangsibo Huang, Chiyuan Zhang, Da Yu, Xinyun Chen, Bill Yuchen Lin, Bo Li, Badih Ghazi, and Ravi Kumar. On memorization of large language models in logical reasoning. In IJCNLP, 2025.

Liu Yang, Fabian Paischer, Kaveh Hassani, Jiacheng Li, Shuai Shao, Zhang Gabriel Li, Yun He, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Robert D. Nowak, Xiaoli Gao, and Hamid Eghbalzadeh. Unifying generative and dense retrieval for sequential recommendation. Trans. Mach. Learn. Res., 2025, 2025.

Zitong Yang, Michal Lukasik, Vaishnavh Nagarajan, Zonglin Li, Ankit Rawat, Manzil Zaheer, Aditya K Menon, and Sanjiv Kumar. Resmem: Learn what you can and memorize the rest. Advances in Neural Information Processing Systems, 36:60768–60790, 2023.

Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, and Yu Shi. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. In ICML, 2024.

Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. Counterfactual memorization in neural language models. Advances in Neural Information Processing Systems, 36:39321–39362, 2023.

Ruohan Zhang, Jiacheng Li, Julian McAuley, and Yupeng Hou. Purely semantic indexing for LLM-based generative recommendation and retrieval. arXiv preprint arXiv:2509.16446, 2025.

Taolin Zhang, Junwei Pan, Jinpeng Wang, Yaohua Zha, Tao Dai, Bin Chen, Ruisheng Luo, Xiaoxiang Deng, Yuan Wang, Ming Yue, Jie Jiang, and Shu-Tao Xia. Towards scalable semantic representation for recommendation. arXiv preprint arXiv:2410.09560, 2024.

Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, and Ji-Rong Wen. Adapting large language models by integrating collaborative semantics for recommendation. In ICDE, 2024.

Bowen Zheng, Enze Liu, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao, and Ji-Rong Wen. Pre-training generative recommender with multi-identifier item tokenization. arXiv preprint arXiv:2504.04400, 2025a.

Bowen Zheng, Hongyu Lu, Yu Chen, Wayne Xin Zhao, and Ji-Rong Wen. Universal item tokenization for transferable generative recommendation. arXiv preprint arXiv:2504.04405, 2025b.

Qiyong Zhong, Jiajie Su, Yunshan Ma, Julian McAuley, and Yupeng Hou. Pctx: Tokenizing personalized context for generative recommendation. arXiv preprint arXiv:2510.21276, 2025.

Jieming Zhu, Mengqun Jin, Qijiong Liu, Zexuan Qiu, Zhenhua Dong, and Xiu Li. Cost: Contrastive quantization based semantic tokenization for generative recommendation. In RecSys, 2024.

Jing Zhu, Mingxuan Ju, Yozen Liu, Danai Koutra, Neil Shah, and Tong Zhao. Beyond unimodal boundaries: Generative recommendation with multimodal semantics. arXiv preprint arXiv:2503.23333, 2025.

