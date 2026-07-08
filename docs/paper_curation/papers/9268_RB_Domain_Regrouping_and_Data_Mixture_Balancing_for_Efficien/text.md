# arXiv:2505.00358v1[cs.LG]1May2025

## R&B: Domain Regrouping and Data Mixture Balancing for Efficient Foundation Model Training

[Figure 1]

Albert Ge∗ Tzu-Heng Huang John Cooper Avi Trost Ziyi Chu Satya Sai Srinath Namburi GNVV Ziyang Cai Kendall Park Nicholas Roberts Frederic Sala

University of Wisconsin-Madison

{afge, thuang273, jfcooper2, astrost, zchu28, zcai75, sgnamburi}@wisc.edu {kendall, nick11roberts, fredsala}@cs.wisc.edu

May 2, 2025

Abstract

Data mixing strategies have successfully reduced the costs involved in training language models. While promising, such methods suffer from two flaws. First, they rely on predetermined data domains (e.g., data sources, task types), which may fail to capture critical semantic nuances, leaving performance on the table. Second, these methods scale with the number of domains in a computationally prohibitive way. We address these challenges via R&B, a framework that re-partitions training data based on semantic similarity (Regroup) to create finer-grained domains, and efficiently optimizes the data composition (Balance) by leveraging a Gram matrix induced by domain gradients obtained throughout training. Unlike prior works, it removes the need for additional compute to obtain evaluation information such as losses or gradients. We analyze this technique under standard regularity conditions and provide theoretical insights that justify R&B’s effectiveness compared to non-adaptive mixing approaches. Empirically, we demonstrate the effectiveness of R&B on five diverse datasets ranging from natural language to reasoning and multimodal tasks. With as little as 0.01% additional compute overhead, R&B matches or exceeds the performance of state-of-the-art data mixing strategies.

### 1 Introduction

Large language models depend on vast, diverse datasets, but the shift to general-purpose foundation models has created a fundamental imbalance: potential training data vastly exceeds available computational resources. This has driven the development of data-efficient strategies that maximize performance while minimizing compute costs. Among these approaches, data mixing is particularly promising. By optimizing the composition of training data used—rather than simply increasing its volume—we can achieve comparable or superior performance with significantly fewer computational resources.

A wide variety of data mixture optimization techniques have been proposed [1; 2; 3; 4; 5]. These adjust the relative proportions (“mixture”) of training data from different predefined domains, also known as skills. Skills are often assigned to data based on human judgments or on source metadata [6]. For example, in an instruction-tuning dataset, skills might include domain categories like open-question answering or summarization. For other datasets, skills may be defined based on where the data was scraped from (Wikipedia, StackOverflow, etc.). We find, however, that these coarse human-defined categorizations fail to capture the optimal groupings for data mixing. That is, human categorizations of skills are suboptimal when used for the development of LLM capabilities.

Consider the Dolly-15k instruction-tuning dataset, which categorizes its data into general domains such as open-question answering, information extraction, and summarization [7]. Rather than directly optimizing these coarse categories, our approach first re-partitions the data into finer-grained, semantically-grouped skills (Fig. 1, left). Optimizing the proportions across these semantically-clustered skills can significantly

∗ Corresponding author. Email: afge@wisc.edu.

Dolly-15k

2.805

EvaluationLoss

2.800

2.795

2.790

2.785

2.780

2.775

0 5 10 15 20 25 30

Number of Clusters

- Figure 1: Instead of using pre-determined domains (e.g., by task type), we find that it is often better to first repartition the data into finer-grained, semantically related domains. Optimizing the proportions of these new semantic domains can significantly improve training performance.

improve training performance over that of the general predefined domains. These improvements are even more pronounced when the number of skill partitions is optimized (Fig. 1, right).

However, the more granular semantic-based clustering approach has a critical drawback. As the number of skills increases, prior data mixing methods become computationally prohibitive. Existing approaches typically require additional evaluations—either through forward passes over evaluation datasets for each skill or by computing per-skill gradient information derived from target tasks. To overcome this limitation, we propose an efficient gradient-based approach that leverages information already computed during training, bypassing the need for these expensive evaluations.

These insights motivate our approach, Regroup & Balance (R&B), a two-stage framework for efficient data mixture optimization. First, we repartition (Regroup) training data into semantically coherent clusters based on embedding similarity. Then, we dynamically optimize domain weights (Balance) to capture individual domain contributions and cross-domain relationships, leveraging domain gradients computed throughout normal training. This produces the best of all worlds: it unlocks the performance gains in fine-grained skill clusters while dramatically reducing computational complexity.

Theoretically, we derive insights to characterize the importance of semantic-based clustering in data mixing strategies and the optimality of R&B. Empirically, we validate R&B across five diverse data settings, encompassing natural-language, instruction-following, reasoning, and multimodal tasks. R&B only requires an additional 0.01% compute overhead, which cuts computational FLOPs by more than 99% relative to existing approaches, all while maintaining or improving model performance.

We summarize our contributions as follows:

- 1. We establish that semantic-based categorizations of skills are superior to human-defined categories for foundation model data mixing algorithms (Section 3.1).
- 2. We introduce R&B, an efficient and theoretically justified two-stage framework that first repartitions data into semantically coherent clusters of skills, then dynamically reweights skill mixtures using their gradients (Sections 3, 3.1, and 3.3).
- 3. We theoretically and empirically demonstrate that R&B scales effectively with increased skill counts (Section 4).

### 2 Related Work

Our work builds upon prior works that study how to effectively curate and compose training data. We summarize three directions: data mixing, data selection, and scaling laws for data mixing.

Group-level Data Mixing. Prior work falls into two categories: static methods, which learn domain proportions before training, and dynamic methods, which adjust them as training progresses. Among the former, Fan et al. [1] uses a smaller model as a proxy to find mixture weights based on each domain’s learning contribution, then applies them for the full model. Xie et al. [2] optimizes worst-case loss with a reference and proxy model, treating the resulting weights as domain proportions. These methods, though efficient, often neglect interactions between domains. Among the latter methods, Chen et al. [3] model cross-domain interactions with a pre-built skills graph to optimize data composition. Building on it,

Chen et al. [4] introduces an online method that estimates domain interactions using training history information. However, as the domain count rises, these methods become computationally expensive.

Sample-level Data Selection. These techniques represent a more granular approach to improve dataset quality by evaluating individual samples. Several methods have been proposed, including using gradient similarity [8; 9; 10; 11], reward functions [12], object detection models [13], and embedding similarity metrics [14]. A related line of work is through deduplication, which removes redundant or nearly identical examples using clustering [15; 16; 17]. Our work integrates these perspectives, repartitioning data into finer-grained groups while simultaneously optimizing domain mixtures.

Scaling Laws. These works model and predict performance when training on diverse domains. Ge

- et al. [18] develop a bivariate scaling law to jointly model domain proportion and data volume, while Ye
- et al. [19] propose a composite exponential law that accounts for cross-domain interactions. Liu et al. [20] approach the problem empirically by training small models with varying mixtures to fit a predictive regression model, Kang et al. [21] build on this to derive compute-optimal data mixtures, and [22] points out that under a static data mix, knowledge and code skills have different compute-optima that can be aligned via data selection. R&B’s dynamic allocation approach suggests the need for new scaling models that can capture the effects of adaptive data allocation strategies.

### 3 R&B: Regroup and Balance Data Mixtures

We first provide some context and intuition. We will refer to skills and domains interchangeably throughout this paper. In data mixing, each data skill/domain is assigned a proportion weight from the probability simplex, and data is sampled according to this probability distribution. Formally, for m skills, we sample data according to the distribution p = [p1,...,pm] ∈ ∆m−1. We seek to answer two questions:

- R1. How should we define domains for data mixing? Given a dataset, we wish to group the data into m partitions suitable for data mixing strategies. We define a mapping function S : X → {1,2,...,m} which labels each data point x to its corresponding partition index. For a given dataset D, we have that Di = {x ∈ D : S(x) = i}, and mi=1 Di = D.

Intuitively, one would like to slice the data to minimize noise within each partition and reduce overlap between partitions. If perfect separation is possible, each partition would correspond to a distinct skill or capability domain, allowing for more targeted optimization of mixing weights. On the other hand, if each data sample is i.i.d. assigned to a group, then we would not expect any data mixing strategy to be better than stratified sampling.

- R2. How do we efficiently reweight domains? Once Dtrain has been partitioned into m groups, our secondary objective is determining the optimal weight proportions p for each domain. Weights may change over time, so training is split into T rounds. At the end of each round t, we can reweight the skills pt+1 = [pt1+1,...,ptm+1] ∈ ∆m−1, and resume training according to the new proportions.
- 3.1 Problem Setup

We formulate our framework as a bilevel optimization problem for minimizing test loss on a dataset. The lower-level optimization aims to find the best training proportions pt ∈ ∆m−1 for each training round t ∈ 1,...,T. The upper-level problem seeks to find the best partitioning of a dataset D into m partitions D1,...,Dm where D = mi=1 Dm. Let Deval,i = {x ∈ Deval : S(x) = i} be the set of evaluation points assigned to skill i by S. Let fθ

be the model parametrized by θt that is trained during round t, i.e. fθ

t

t+1

with proportions pt. Formally, we aim to solve the following problem min

is obtained by training fθ

t

Leval(fθ

min

), (1)

T +1

m∈Z+

p1,...,pT ∈∆m−1

where Leval(fθ

on mixture proportions pT to obtain fθ

) is the average evaluation loss for the partition Deval,i after training model fθ

T +1

T

.

T +1

Solving the full bilevel problem (1) is infeasible because for each candidate solution of the upper-level optimization problem, we must train a model for the lower-level optimization problem to obtain the loss.

Thus, we propose decomposing Equation 1 into two:

S∗,m∗ = arg min S,m

Lclustering(S;fUnif(D)), (2) min p1,...,pt∈∆m∗−1

L∗eval(fθ

). (3)

T +1

In (2), we use fUnif, which we denote as a model trained on fixed uniform proportions across skills to convergence (i.e. stratified sampling, fUnif = arg minf Leval(f)). This minimization is taken over a family of partitioning schemes S, such as the S found through k-means, on the gradients of trained model Leval(fUnif). In (3), we use the optimal choice of m∗ and partitioning scheme S∗ found in the previous stage, and solve the optimization problem at every training round t.

##### 3.2 Defining Domains

We investigate how to partition a given dataset to achieve optimal data mixing. Intuitively, data points that belong in a cluster should have a similar effect, i.e. gradients, during training. If gradients are not aligned, then swapping points with another cluster would reduce noise in both clusters. This leads to our first definition.

Definition 1. A skill-assigning function S : D → [m] is stable in the direction ∇L(θt;Dp) if for the skill i = arg maxi∈[m] ∇L(θt;Di)⊤∇L(θt;Dp) and any other j ∈ [m], exchanging a pair xi ∈ Di, xj ∈ Dj does not improve ∇L(θt;Di)⊤∇L(θt;Dp).

In other words, a clustering is said to be stable if no swapping of points improves alignment with the evaluation gradient. This definition provides a theoretical foundation for optimal data mixing, but it is still impractical to discover good groupings. The following lemma characterizes the maximum regret from swapping points between clusters.

Lemma 1. Define the regret RS(i,j) under the skill-assigning function S for class j as the difference between the gradient alignments:

∇L(θt;Dp)⊤∇L(θt,D˜i) − ∇L(θt;Dp)⊤∇L(θt;Di).

RS(i,j) = max

D˜i⊂Di∪Dj,|D˜i|=|Di|

Let i,j ∈ |m| and assume |Di| = |Dj| and ∇L(θt;Di)⊤∇L(θt;Dp) ≥ ∇L(θt;Dj)⊤∇L(θt;Dp), and let ri = maxx∈D

i |∇L(θt,x)⊤∇L(θt;Dp) − ∇L(θt;Di)⊤∇L(θt;Dp)|. Then we have

- 1

- 2

(ri + rj − (∇L(θt;Di)⊤∇L(θt;Dp) − ∇L(θt;Dj)⊤∇L(θt;Dp))) .

RS(i,j) ≤ max 0,

Proof: See Appendix B.1.2.

While RS(i,j) is still dependent on the direction of the evaluation gradient ∇L(θt;Dp), this result shows that for a clustering that assigns each class to have a small radius in every direction (an upper bound on ri) and a large separation between their means, in many directions mostly orthogonal to their difference vector, RS(i,j) is 0. This bound reveals additional structure that enables us to determine an effective clustering: clusters should have minimal radii while maintaining sufficient separation between their centroids. This is equivalent to S being stable, a property which R&B uses to achieve optimality as compared to using other clusterings (see Section 3.4 and note that ∇L(θt;Di)⊤∇L(θt;Dp) should be maximized).

In light of these theoretical findings, we seek to empirically validate our claims by clustering real data and train using fixed proportions p. Our hypothesis is that well-clustered data can result in better overall training performance. In practice, there are many choices for the skill-assigning function S and the number of skills. To keep our investigation tractable, we focus on k-means clustering, and sweep over k. We first embed our examples with ModernBERT-embed [23], a state-of-the-art embedding model that supports long-context inputs. Then, we apply k-means and train a model using a uniform proportion of k clusters. We evaluate our setup across four settings: Dolly-15k [7], Super Natural Instructions (Super-NatInst) & Super Natural Instruction Test (Super-NatInst Test) [24], and S1-Reasoning [25], and across 3 seeds. Appendix F lists the full experimental details.

The top row of Figure 2 shows that training on the resulting clusters often results in significantly better performance compared to pre-determined partitions. On 3 of the 4 datasets, there is a U-shaped

###### Regroup Domains Pre-Determined Domains # Pre-Determined Domains

- 2.800

- 2.805

EvaluationLoss

Dolly-15k

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0 50 100 150 200

Number of Clusters

2.87

2.88

2.89

2.90

2.91

2.92

2.93

EvaluationLoss

Super-NatInst Test

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

20 40 60 80 100

Number of Clusters

2.46

2.48

2.50

2.52

2.54

2.56

2.58

EvaluationLoss

Super-NatInst

20 40 60 80 100

Number of Clusters

0.745

0.746

0.747

0.748

0.749

0.750

0.751

0.752

EvaluationLoss

S1-59k

|R² = 0.047|
|---|
| |
| |
| |

0.01 0.02 0.03 0.04

Silhouette Score

2.78

2.79

2.80

EvaluationLoss

|[Figure 2]| | | | | |
|---|---|---|---|---|---|
| | | | | | |

5 10 15 20 25 30

Number of Clusters

0.10 0.15 0.20

Silhouette Score

2.88

2.90

2.92

EvaluationLoss

R² = 0.482

|[Figure 3]| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

20 40 60 80 100 120 140 160 180

Number of Clusters

|R² = 0.173|
|---|
| |
| |
| |
| |
| |

0.20 0.25 0.30 0.35

Silhouette Score

2.450

2.475

2.500

2.525

2.550

2.575

EvaluationLoss

|[Figure 4]| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

10 20 30 40 50 60 70 80 90 100

Number of Clusters

|R² = 0.895|
|---|
| |
| |
| |
| |

0.03 0.02 0.01

Silhouette Score

0.744

0.746

0.748

0.750

0.752

EvaluationLoss

|[Figure 5]| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

10 20 30 40 50 60 70 80 90 100

Number of Clusters

Figure 2: Top Row: Across various data settings, we find that there is a “sweet spot” in the number of domains used for data mixing, indicated by the green star. The optimal number of groups varies significantly with the dataset, which motivates the need for compute-efficient data mixing. Bottom Row: We find that silhouette score often correlates with model performance, suggesting that it is possible to predict data mixing performance based on clustering metrics.

pattern in the number of clusters versus evaluation loss. Thus in many cases, there is an optimal choice of k—but it varies significantly between datasets.

Are these optimal clusters compact and well-separated, as our theory suggests? We find that generally, the answer is yes. We plot the silhouette score [26] of each cluster group against the final evaluation loss of the model. The bottom row of Figure 2 shows that on 3 of the 4 datasets, there is moderate to strong correlation between the clusters’ silhouette score and model performance. These results validate our theoretical insights that clusters which are well-separated result in better data mixing performance. Furthermore, this suggests that it is possible to choose the optimal k without training a model, which would lead to further cost savings.

- 3.3 Proposed Method

The R&B algorithm (Algorithm 1) performs adaptive data selection for efficient model training on partitioned datasets. Starting with a uniform sampling distribution across clusters, it iteratively refines this distribution to focus computational resources on the most relevant partitions.

During each training round, R&B accumulates final-layer gradients from sampled batches and tracks which clusters contribute to model updates. Then it constructs a gradient similarity matrix that captures how gradients from different clusters relate to each other. This similarity information is combined with predefined evaluation proportions to produce an updated sampling distribution through a softmax operation. As training progresses, the algorithm adaptively shifts sampling probability toward clusters that contain the most valuable training examples, improving efficiency while maintaining performance across all partitions.

The key innovation of R&B lies in its use of gradient information to dynamically adjust sampling priorities, enabling models to learn effectively from heterogeneous data without requiring extensive tuning for each data partition.

- 3.4 Determining Optimal Proportions

2.795

2.790

2.785

2.780

2.775

0 5 10 15 20 25 30

Number of Clusters

Starting from the objective in (3), we greedily aim to find the best pt such that fθ

has the greatest decrease in loss possible for this iteration of gradient descent. Specifically, θt+1 is taken to be the SGD update of the current θt, where θt+1(pt) = θt − η∇L(θt;Dpt). Here, we have control over the weighting of the gradients pt in this gradient descent step. This gradient term is the weighted sum of the gradients

t+1

Algorithm 1 R&B: Online Domain Data Selection

- 1: Input: Domain datasets {Dtrain∗ ,i, Deval∗ ,i}im=1∗ , model θ ∈ Rn, training rounds T, steps per round K, evaluation weights p ∈ Rm∗
- 2: Initialize sampling distribution p0 ← Uniform(m∗)
- 3: for t = 0 to T − 1 do
- 4: Initialize gradient accumulators ∇L(θ; Di) ← 0n and sample sets Si ← ∅ for all i
- 5: for k = 0 to K − 1 do
- 6: Sample mini-batch B from Dtrain∗ ,i using domain probabilities pt
- 7: Update model: θ ← θ − η∇L(θ; B)
- 8: for each domain i present in B do
- 9: Accumulate final-layer gradients: ∇L(θ; Di) += ∇L(θ; B ∩ Dtrain∗ ,i)
- 10: Add samples to set: Si ← Si ∪ (B ∩ Dtrain∗ ,i)
- 11: end for
- 12: end for
- 13: Compute similarity matrix G with Gij = |S 1

i||Sj|∇L(θ; Di)⊤∇L(θ; Dj)

- 14: Update distribution: pt+1 ← softmax(λGp/∥Gp∥2)
- 15: end for
- 16: Return: Final model parameters θ

per skill: ∇L(θt;Dpt) = i(pt)i∇L(θt;Di). Now, assume that ∇L is well approximated by its first order Taylor expansion (for some exposition on how this method behaves under a Lipschitz loss, see Appendix

- B.1). In this case, the optimization objective for finding the best mixture weights becomes pt =

L(θt;Dp) − ∇L(θt;Dp)⊤∇L(θt;Dp′) = arg max p′∈∆m−1

arg min

p′∈∆m−1

m

(p′)i∇L(θt;Dp)⊤∇L(θt;Di).

i=1

This objective is simply a linear objective over the simplex, which will be maximized by one of the corners of the simplex; this maximal corner will correspond with the skill that aligns most with the target loss averaged over skills with distribution p.

This optimum, however, will be highly discontinuous and will only ever be able to sample from a single skill per descent step. This creates two major issues: a highly variable sampling scheme may cause highly variable and unpredictable behavior in training, and the per-gradient sampling scheme required for our method requires samples from each skill within the training batch. We address this by adding cross entropy regularization, which prevents extreme mixing proportions and aligns better with scaling law results [19]. The regularized solution remains tractable:

1 Z

λ ∥Gp∥2

pt =

softmax

Gp ,

where Z is a normalization constant, λ is a hyperparameter, and Gij = ∇L(θt;Di)⊤∇L(θt;Dj). See Appendix B.1 for derivation details.

Comparison to Multiplicative Weights and DoGE. Other works ([1], [3]) use an update rule based on multiplicative weights. The update in DoGE [1] (which is most similar to ours) is written in our notation as:

p′t = 1/Z(p′t−1 exp(ηWt/µ)) = 1/Z(p′i−1 exp(ηGtp/µ)) = 1/Z exp η ti=1 Gip/µ ,

where µ = 1/λ and Wt = Gp by the definitions of G and p. Note that Gip is aggregated up to round t. In contrast, our method aggregates Gip for K training iterations before updating p′, and then overwrites the mixture weights to be those of the previous window. This prevents the diminishing influence of later updates and gradient scale reduction that occurs in multiplicative weights approaches. An update that refreshes the proportions at every window will circumvent the strong bias towards the optimal weights remaining similar, as was also used in [3].

### 4 Experiments

We study the effectiveness of R&B empirically across a diverse range of datasets and tasks. Our experiments aim to validate the following claims:

Sup-NatInst Sup-NatInst test Dolly-15k

(m = 38) (m = 60) (m = 8) Method Loss % Overhead (↓) Loss % Overhead (↓) Loss % Overhead (↓) Stratified 2.591 0 2.877 0 2.788 0

Skill-It 2.632 595.5 2.911 6 × 107 2.786 14.46 Aioli 2.622 1336.5 2.883 7 × 106 2.779 62.5 DGA 2.591 1.723 2.893 1601 2.787 0.41

(m∗ = 30) (m∗ = 100) (m∗ = 7) R&B

2.381 0.009 2.859 0.1 2.765 0.0006

Table 1: Across three datasets, R&B significantly reduces the compute overhead for evaluation compared to existing methods, while matching or exceeding performance.

- C1. R&B can match or improve training performance on natural language tasks while significantly reducing computational overhead compared to existing methods.

C2. R&B can improve training performance beyond natural language modalities.

4.1 Data Mixing on Natural Language Tasks

Setup. We compare R&B against four existing baselines: stratified sampling, Skill-It [3], Aioli [4], and DGA [27]. We compare each method across three distinct three natural-language data settings: Dolly-15k [7], NaturalInstructions In-domain Sup-NatInst [24], and NaturalInstructions Test Sup-NatInst. For all experiments, we train 125M GPT-Neo models [28]; full experimental details are listed in Appendix F. We report the final evaluation loss and the relative compute overhead (over standard training) incurred from re-estimating proportions. The formulas for the relative compute overhead are derived in Appendix

- D.

Results. Table 1 shows R&B’s strong performance across all datasets. On Sup-NatInst, R&B achieves the best performance (loss: 2.381) with minimal computational overhead (0.009%). On Sup-NatInst test, R&B outperforms all methods (loss: 2.859) with only 0.1% overhead versus Skill-It (6 × 107%) and Aioli ((7 × 106%). On Dolly-15k, R&B performs competitively (loss: 2.765) compared to Aioli (2.779), with significantly smaller overhead (0.0006% versus 62.5%). As expected from C1, R&B consistently delivers strong results with orders of magnitude better computational efficiency than other data mixing approaches.

Ablations. Fig. 3 ablates semantic regrouping across data mixing strategies. In Sup-NatInst, regrouping improves most methods (5.3-8.1% gains) except Skill-It. On Sup-NatInst test, regrouping yields modest improvements for all methods except Aioli. Dolly-15k shows strong improvements with regrouping across all methods. While in many cases regrouping does help, it is not universally beneficial as evidenced by Skill-It’s performance degradation on Sup-NatInst and Aioli’s slight decline on Sup-NatInst test. Notably, the Balance method combined with regrouping achieves the best overall performance on both Sup-NatInst datasets, while Aioli with regrouping performs best on Dolly-15k.

Even without regrouping, our gradient-based method shows strong performance while maintaining minimal overhead. On original Sup-NatInst, Balance achieves a loss of 2.520, significantly outperforming other data-mixing methods. On Dolly-15k Original, it reaches a competitive 2.783 loss. We omit Balance results for original Sup-NatInst test since our method requires training and validation data to share the same m groups—a limitation easily addressed by re-mapping validation points to corresponding training skills.

The convergence plot in Figure 3 demonstrates R&B’s efficiency. R&B reaches convergence with only 20% of the training steps needed by other methods while achieving lower final loss values. Its smooth, monotonic descent contrasts with methods like Aioli, suggesting R&B’s gradient-based domain weighting enables more consistent optimization.

###### Sup-NatInst

Method Sup-NatInst Sup-NatInst test Dolly-15k

dga r&b aioli

3.4

EvaluationLoss

Original Regroup Original Regroup Original Regroup m 38 30 60 100 8 7

3.2

skillit static

3.0

Stratified 2.591 2.454 2.877 2.871 2.788 2.761

2.8

Skill-It 2.632 2.812 2.911 2.881 2.786 2.778 Aioli 2.622 2.488 2.883 2.947 2.779 2.760 DGA 2.591 2.453 2.893 2.871 2.787 2.761

2.6

2.4

Balance 2.520 2.381 - 2.859 2.783 2.765 0 200 400 600 800 100012001400160018002000

Number of Steps

- Figure 3: Left: Regrouping skills before applying data mixing strategies can yield substantial improvements. Underlined values indicate where regrouping beats the original grouping for that method and dataset. Highlighted values (with brown background) indicate the best overall performance for each dataset. Note that we do not apply Balance to the original categorization of Sup-NatInst test, as we assume that training data and validation data are bucketed into the same m groups. Right: Loss curve on the Sup-NatInst dataset.

##### 4.2 Beyond Language Modeling

We next explore additional tasks our approach was not specifically designed for, including reasoning and multimodal setups.

Reasoning Setup. We investigate whether optimizing data mixtures can boost model performance in reasoning tasks. We use S1-Reasoning dataset [25], which comprises reasoning traces from challenging math problems, drawn from 54 distinct sources. We use Qwen2-0.5B model [29] as an illustrative example.

Reasoning Results. Table 2 shows that regrouping improves performance compared to using original domains, with the optimal number of groups being 10. Specifically, the evaluation loss decreased from 0.7517 to 0.7449 when using our regrouping approach instead of predetermined domains. However, after applying data mixing techniques to this clustered dataset, we observe that R&B achieves comparable performance to stratified sampling with a slight improvement. This suggests that while clustering generally improves the model performance, applying data mixing techniques such as R&B does not improve model performance further. We believe it is an open question as to whether data mixing can still be applied to such reasoning traces.

S1-59K Method

Original (m = 54)

Regroup (m = 10)

Stratified 0.7517 0.7449 R&B - 0.7449

Table 2: Performance comparison of Stratified and R&B methods on the S1-59K dataset.

Multimodal Setup. We extend our setup to include multimodal tasks. We train CLIP models [30] from scratch using the small-scale DataComp dataset [31; 32]. Our dataset comprises approximately 10 million image-caption pairs sourced from the web.1 To ensure dataset quality, we select the top 30% of samples based on CLIP Score [32], retaining 3.8 million high-quality pairs.

We extract image embeddings for both the filtered training dataset and DataComp’s evaluation benchmark, which spans 38 diverse downstream tasks. We apply k-means clustering to repartition data into varying numbers of groups. We use DataComp’s training configurations and adopt R&B as our training method. And, we use stratified sampling as our baseline method.

Results. We presented CLIP models’ performance in Table 3. R&B outperforms stratified sampling when the number of domains exceeds 10. With 50 domains, R&B achieves a 3.27% relative improvement over the stratified sampling baseline. These findings highlight the effectiveness of R&B when the number of underlying domains is potentially high and validate its extensibility to modalities beyond natural language, confirming C2.

1Some URLs provided by DataComp are now broken. See here for details.

ImageNet dist. shift

Avg over 38 datasets (↑)

m Method ImageNet

VTAB Retrieval

Stratified 0.034 0.044 0.157 0.104 0.146 R&B 0.033 0.040 0.153 0.104 0.141

10

Stratified 0.036 0.044 0.153 0.106 0.145 20

R&B 0.031 0.042 0.163 0.103 0.148

Stratified 0.042 0.047 0.170 0.107 0.153

50

R&B 0.042 0.047 0.177 0.108 0.158 Stratified 0.034 0.043 0.152 0.107 0.139

100

R&B 0.041 0.047 0.151 0.104 0.145

Stratified 0.034 0.043 0.165 0.109 0.143 R&B 0.039 0.050 0.164 0.109 0.153

150

Table 3: R&B performs better than stratified sampling on image-text modalities.

### 5 Conclusion

In this paper, we introduced Regroup & Balance (R&B), a two-stage framework that breaks free from two fundamental constraints found in state-of-the-art data mixing strategies: the limitations of predetermined domains and the computational bottleneck of per-skill evaluations. Empirically, R&B matched or exceeded state-of-the-art data mixing methods while requiring two orders of magnitude less compute overhead. By reimagining both what to mix and how to mix it, R&B charts a more efficient path forward for foundation model training in an era of unlimited unstructured data and constrained computational resources.

### 6 Acknowledgements

We thank Changho Shin, Dyah Adila, Jiayu Wang, Jitian Zhao, Gabe Orlanski, June Cho, and Mayee Chen for discussions and feedback. Additionally we thank the Data Science Institute and the Center for High-Throughput Computing at UW-Madison for providing compute and research support. We are grateful for the support of the Defense Advanced Research Projects Agency (DARPA) under the Young Faculty Award, the NSF under CCF2106707 (Program Synthesis for Weak Supervision), and the Wisconsin Alumni Research Foundation (WARF).

### References

- [1] Fan, S.; Pagliardini, M.; Jaggi, M. DoGE: Domain Reweighting with Generalization Estimation. 2024; http://arxiv.org/abs/2310.15393, arXiv:2310.15393.
- [2] Xie, S. M.; Pham, H.; Dong, X.; Du, N.; Liu, H.; Lu, Y.; Liang, P.; Le, Q. V.; Ma, T.; Yu, A. W. DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. 2023; http://arxiv. org/abs/2305.10429, arXiv:2305.10429 [cs].
- [3] Chen, M. F.; Roberts, N.; Bhatia, K.; Wang, J.; Zhang, C.; Sala, F.; Re´, C. Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models. 2023; http://arxiv.org/ abs/2307.14430, arXiv:2307.14430 [cs].
- [4] Chen, M. F.; Hu, M. Y.; Lourie, N.; Cho, K.; R´e, C. Aioli: A Unified Optimization Framework for Language Model Data Mixing. 2024; http://arxiv.org/abs/2411.05735, arXiv:2411.05735.
- [5] Jiang, Y.; Zhou, A.; Feng, Z.; Malladi, S.; Kolter, J. Z. Adaptive Data Optimization: Dynamic Sample Selection with Scaling Laws. 2024; http://arxiv.org/abs/2410.11820, arXiv:2410.11820.
- [6] Wettig, A.; Lo, K.; Min, S.; Hajishirzi, H.; Chen, D.; Soldaini, L. Organize the Web: Constructing Domains Enhances Pre-Training Data Curation. 2025; http://arxiv.org/abs/2502.10341, arXiv:2502.10341 [cs].

- [7] Conover, M.; Hayes, M.; Mathur, A.; Xie, J.; Wan, J.; Shah, S.; Ghodsi, A.; Wendell, P.; Zaharia, M.; Xin, R. Free Dolly: Introducing the World’s First Truly Open Instruction-Tuned LLM. 2023; https://www.databricks.com/blog/2023/04/12/ dolly-first-open-commercially-viable-instruction-tuned-llm.
- [8] Engstrom, L.; Feldmann, A.; Madry, A. DsDm: Model-Aware Dataset Selection with Datamodels. 2024; http://arxiv.org/abs/2401.12926, arXiv:2401.12926 [cs, stat].
- [9] Xia, M.; Malladi, S.; Gururangan, S.; Arora, S.; Chen, D. LESS: Selecting Influential Data for Targeted Instruction Tuning. 2024; http://arxiv.org/abs/2402.04333, arXiv:2402.04333 [cs].
- [10] Killamsetty, K.; Durga, S.; Ramakrishnan, G.; De, A.; Iyer, R. Grad-match: Gradient matching based data subset selection for efficient deep model training. International Conference on Machine Learning. 2021; pp 5464–5474.
- [11] Huang, T.-H.; Bilkhu, M.; Sala, F.; Movellan, J. Evaluating Sample Utility for Data Selection by Mimicking Model Weights. arXiv preprint arXiv:2501.06708 2025,
- [12] Wu, M.; Vu, T.-T.; Qu, L.; Haffari, G. Mixture-of-Skills: Learning to Optimize Data Usage for FineTuning Large Language Models. 2024; http://arxiv.org/abs/2406.08811, arXiv:2406.08811 [cs].
- [13] Huang, T.-H.; Shin, C.; Tay, S. J.; Adila, D.; Sala, F. Multimodal data curation via object detection and filter ensembles. arXiv preprint arXiv:2401.12225 2024,
- [14] Xie, S. M.; Santurkar, S.; Ma, T.; Liang, P. Data Selection for Language Models via Importance Resampling. 2023; http://arxiv.org/abs/2302.03169, arXiv:2302.03169 [cs].
- [15] Abbas, A.; Tirumala, K.; Simig, D.; Ganguli, S.; Morcos, A. S. SemDeDup: Data-efficient learning at web-scale through semantic deduplication. 2023; http://arxiv.org/abs/2303.09540, arXiv:2303.09540 [cs].
- [16] Lee, K.; Ippolito, D.; Nystrom, A.; Zhang, C.; Eck, D.; Callison-Burch, C.; Carlini, N. Deduplicating Training Data Makes Language Models Better. 2022; http://arxiv.org/abs/2107.06499, arXiv:2107.06499.
- [17] Tirumala, K.; Simig, D.; Aghajanyan, A.; Morcos, A. S. D4: Improving LLM Pretraining via Document De-Duplication and Diversification. 2023; http://arxiv.org/abs/2308.12284, arXiv:2308.12284 [cs].
- [18] Ge, C.; Ma, Z.; Chen, D.; Li, Y.; Ding, B. BiMix: Bivariate Data Mixing Law for Language Model Pretraining. 2024; http://arxiv.org/abs/2405.14908, arXiv:2405.14908 [cs].
- [19] Ye, J.; Liu, P.; Sun, T.; Zhou, Y.; Zhan, J.; Qiu, X. Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance. 2024; http://arxiv.org/abs/2403.16952, arXiv:2403.16952.
- [20] Liu, Q.; Zheng, X.; Muennighoff, N.; Zeng, G.; Dou, L.; Pang, T.; Jiang, J.; Lin, M. RegMix: Data Mixture as Regression for Language Model Pre-training. 2024; http://arxiv.org/abs/2407. 01492, arXiv:2407.01492 [cs].
- [21] Kang, F.; Sun, Y.; Wen, B.; Chen, S.; Song, D.; Mahmood, R.; Jia, R. AutoScale: Automatic Prediction of Compute-optimal Data Composition for Training LLMs. 2024; http://arxiv.org/ abs/2407.20177, arXiv:2407.20177 [cs, stat].
- [22] Roberts, N.; Chatterji, N.; Narang, S.; Lewis, M.; Hupkes, D. Compute Optimal Scaling of Skills: Knowledge vs Reasoning. 2025; https://arxiv.org/abs/2503.10061, eprint: 2503.10061.

- [23] Nussbaum, Z.; Morris, J. X.; Duderstadt, B.; Mulyar, A. Nomic Embed: Training a Reproducible Long Context Text Embedder. 2024; eprint: 2402.01613.

- [24] Wang, Y. et al. Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks. 2022; http://arxiv.org/abs/2204.07705, arXiv:2204.07705 [cs].

- [25] Muennighoff, N.; Yang, Z.; Shi, W.; Li, X. L.; Fei-Fei, L.; Hajishirzi, H.; Zettlemoyer, L.; Liang, P.; Cand`es, E.; Hashimoto, T. s1: Simple test-time scaling. 2025; https://arxiv.org/abs/2501. 19393, eprint: 2501.19393.

- [26] Rousseeuw, P. J. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 1987, 20, 53–65.
- [27] Fan, S.; Grangier, D.; Ablin, P. Dynamic Gradient Alignment for Online Data Mixing. 2024; http://arxiv.org/abs/2410.02498, arXiv:2410.02498 [cs].
- [28] Black, S.; Gao, L.; Wang, P.; Leahy, C.; Biderman, S. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow. 2021; https://doi.org/10.5281/zenodo.5297715.
- [29] Yang, A. et al. Qwen2 Technical Report. arXiv preprint arXiv:2407.10671 2024,
- [30] Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; others Learning transferable visual models from natural language supervision. International conference on machine learning. 2021; pp 8748–8763.
- [31] Ilharco, G.; Wortsman, M.; Wightman, R.; Gordon, C.; Carlini, N.; Taori, R.; Dave, A.; Shankar, V.; Namkoong, H.; Miller, J.; Hajishirzi, H.; Farhadi, A.; Schmidt, L. OpenCLIP. 2021; https: //doi.org/10.5281/zenodo.5143773.
- [32] Gadre, S. Y.; Ilharco, G.; Fang, A.; Hayase, J.; Smyrnis, G.; Nguyen, T.; Marten, R.; Wortsman, M.; Ghosh, D.; Zhang, J.; others Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems 2023, 36, 27092–27112.
- [33] Kaplan, J.; McCandlish, S.; Henighan, T.; Brown, T. B.; Chess, B.; Child, R.; Gray, S.; Radford, A.; Wu, J.; Amodei, D. Scaling Laws for Neural Language Models. 2020; http://arxiv.org/abs/ 2001.08361, arXiv:2001.08361 [cs].
- [34] Hobbhahn, M. What’s the Backward-Forward FLOP Ratio for Neural Networks? 2021; https: //epoch.ai/blog/backward-forward-FLOP-ratio.
- [35] Goodfellow, I. Efficient Per-Example Gradient Computations. 2015; http://arxiv.org/abs/ 1510.01799, arXiv:1510.01799 [stat].
- [36] Wang, J. T.; Wu, T.; Song, D.; Mittal, P.; Jia, R. GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration. 2024.

The appendix is structured as follows. Appendix A introduces our notation, followed by theoretical insights and proofs in Appendix B. Algorithmic details are provided in Appendix C. We then analyze the computational cost of existing data mixing methods in Appendix D. Implementation specifics and experimental setups are detailed in Appendix E and Appendix F, respectively. Appendix G presents the results of our ablation studies. Lastly, we give interpretation for determined domains in Appendix H.

### A Notation

|Symbol|Meaning|
|---|---|
|d m|The dimensionality of each skill (toy theory) The number of skills<br><br>|
|[·] ∆m−1 D Di Dp Di G L(θ;D) ∇L(θ;Di) p p′ S<br><br>|The sequence of numbers 1,2,...,n The m − 1 dimensional simplex A mixture data distribution The distribution for skill i<br><br>The mixture of Di according to p A sample from the distribution Di The inner product between gradients of different skills ∇L(θ;Di) The expected loss of θ over Dp The skill gradient for skill i The evaluation data proportions The chosen training data proportions A skill-assigning function S : X → [m]|

### B Theoretical Results

The goal is to find the best data mixture throughout training. With the many degrees of freedom such an algorithm can take, a few assumptions are made. First, an update is restricted to being performed by SGD with the following update rule:

θt+1 = θt − η∇L(θt;Dp′).

The modification from standard SGD is the ability to change p′ which allows for a different sampling mixture.

##### B.1 Method Derivation

The design of the proportion finding algorithm described here comes from one core assumption: the gradient update works roughly linearly. Specifically, we assume, for a small enough ball around θt,

L(θ;Dp) ≈ L(θt;Dp) + ∇L(θt;Dp)⊤(θ − θt). Inputting the SGD update rule,

L(θt+1;Dp) ≈ L(θt;Dp) − η∇L(θt;Dp)⊤∇L(θt;Dp′).

Since the gradient is linear, we can treat ∇L(θt;Dp) as a weighted sum of ∇L(θt;Di), i.e. the individual skill gradients, based on the proportions pi. Define the Gram matrix G as

G(ijt) = ∇L(θt;Di)⊤∇L(θt;Dj).

Note that G(t) is dependent on the iteration, but this will often be written only as G for notational clarity. This matrix also has a clear interpretation as a neural tangent kernel (NTK) aggregated over each skill rather than over each data point as is commonly used. With this matrix and treating p and p′ as vectors, our assumption simplifies to

L(θt+1;Dp) ≈ L(θt;Dp) − ηp⊤Gp′.

Since the aim is to decrease loss as much as possible, we want to maximize ηp⊤Gp′ over p′ ∈ ∆m−1. This objective is linear over the simplex, so optima will only be found at the corners of the simplex. However, this would imply only one class is sampled from. This causes a few issues. First, the per-skill

gradient computation method requires samples from skill i to find the gradient for skill i. No information about other skills will be gathered through training. Second, the gradient as a function of time will be discontinuous. As the maximum skill of the optimization changes, the gradient will instantly change to the skill gradient for the new maximal skill. We impose a common regularity to restrict p′ to the simplex by solving for the following:

ηp⊤Gp′ − λ1

p′ = arg max p′∈∆m−1

m

m

p′i log p′i + λ2

p′i.

i=1

i=1

Equivalently, we can multiply the λ1 and λ2 terms by a constant without affecting the optimum. This constant ∥ηGp∥2 is chosen to make the parameter λ1 have a roughly equal effect between the cross entropy term of the base objective, regardless of η, G, and p. This also means that as the model trains and the gradients decrease in magnitude, the effects of the regularization term will decrease. Otherwise, proportions will tend towards uniform. Now, solve for:

p′ = arg max p′∈∆m−1

ηp⊤Gp′ − λ1∥ηGp∥

m

m

p′i log p′i + λ2∥ηGp∥

p′i.

i=1

i=1

The parameter λ1 acts as a normal hyper-parameter, but λ2 is a Lagrange multiplier enforcing p′ lay on the simplex. Taking a gradient and setting to 0,

0 = ηGp − λ1∥ηGp∥log p′ + (λ1 + λ2)∥ηGp∥1m, log p′ =

η λ1∥ηGp∥

λ1 + λ2 λ1

Gp +

1m,

η λ1∥ηGp∥

λ1 + λ2 λ1

p′ = exp

Gp exp

1 λ1∥Gp∥

λ1 + λ2 λ1

= exp

Gp exp

.

Here, 1m is the vector of all 1s of dimension m. Also, λ2 will take on a value to make p′ sum to 1. Thus, setting Z = mi=1 exp λ 1

1∥Gp∥Gp

and letting λ = 1/λ1, we have

i

λ ∥Gp∥

1 Z

p′ =

exp

Gp

λ ∥Gp∥

= softmax(

Gp).

Note that in this parametrization, a large λ indicates a small penalty from the entropy term. This solution aligns with the unconstrained solution described above from finding the maximal corner of the simplex, except now the solution is smooth.

###### B.1.1 Max can be Better than Static proportions

This method, for a small enough learning rate, will result in a reduction of a smooth loss greater than any other fixed proportions.

- Lemma 2. Let ℓ be an L-smooth loss function and fθ is learned with SGD on datasets Di with associated

sampling prior pi, and let p′′ be some other fixed distribution of datasets. If the learning rate η satisfies η ≤

maxi ∇L(θ0;Di)⊤∇L(θ0;Dp)−(∇L(θ0;Dp′′))⊤∇L(θ0;Dp)

1 L

maxi ∥∇L(θ0;Di)∥2+∥∇L(θ0;Dp′′)∥2 , then a gradient descent step with p′ = δarg max

i(Gp)i results in a greater (or equal) decrease in the loss than p′′ .

Proof. Let θ0 be some base parameter, p′ = δi for i = arg maxi(Gp)i, and let

θp′ = θ0 − η∇L(θ0;Dp′), θp′′ = θ0 − η∇L(θ0;Dp′′).

Assume that L is L-smooth. Now consider L(θp′′;Dp) − L(θp′;Dp) = (L(θp′′;Dp) − L(θ0;Dp)) − (L(θp′;Dp) − L(θ0;Dp)) ≥ −ηp⊤Gp′′ + ηp⊤Gp′ − η2Lp′⊤Gp′ − η2Lp′′⊤Gp′′

(Gp)i − p⊤Gp′′ − ηL(p′⊤Gp′ + p′′⊤Gp′′)).

= η(max

i

i−p⊤Gp′′ Gii+p′′⊤Gp′′

Now let i = arg maxi(Gp)i and let η ≤ L1 (Gp)

. This quantity is positive (and therefore well defined for η ≥ 0 since p⊤Gp′′ is the convex combination of values all less than (or equal to) (Gp)i. Thus,

L(θp′′;Dp) − L(θp′;Dp) ≥ η((Gp)i − p⊤Gp′′ − ηL(Gii + p′′⊤Gp′′)) ≥ η((Gp)i − p⊤Gp′′ −

(Gp)i − p⊤Gp′′ Gii + p′′⊤Gp′′ L(Gii + p′′⊤Gp′′))

1 L

= η((Gp)i − p⊤Gp′′ − ((Gp)i − p⊤Gp′′))

= 0.

Therefore, for a small enough learning rate, the choosing the gradient with the largest skill results in a larger decrease in the loss than using priors proportional to the evaluation data. The learning rate can also be bounded using gradient notation and taking a maximum over Gii:

;Di)⊤∇L(θθ

;Dp′′))⊤∇L(θθ

maxi ∇L(θθ

;Dp) − (∇L(θθ

;Dp) maxi ∥∇L(θθ

1 L

0

0

0

0

η ≤

.

;Di)∥2 + ∥∇L(θ0;Dp′′)∥2

0

| |
|---|

###### B.1.2 Clustering

One major phenomenon observed here is that clustering the data points works well for domain mixing, and sometimes even outperforms the provided labels for the different skills. These clusters are taken via the embeddings of some other model, which are assumed to mimic the gradients of the model being learned.

When η ≈ 0, the change in the loss can be well-approximated by its first order Taylor expansion: L(θt;Dp) − L(θt+1;Dp) ≈ η∇L(θt;Dp)⊤∇L(θt;Dp′). For the following sections, let Di = {x ∈ D : S(x) = i} for some skill-assigning function S.

- Definition 2. A skill-assigning function S : D → [m] is stable in the direction ∇L(θt;Dp) if for the skill

i = arg maxi∈[m] ∇L(θt;Di)⊤∇L(θt;Dp) and any other j ∈ [m], exchanging a pair xi ∈ Di, xj ∈ Dj does not improve ∇L(θt;Di)⊤∇L(θt;Dp).

Intuitively, a stable clustering is one which doesn’t improve ∇L(θt;Di)⊤∇L(θt;Dp) by exchanging points between classes.

- Lemma 3. Let S be some {0,1} skill-assigning function on D with ∇L(θt;D0)⊤∇L(θt;Dp) ≥ ∇L(θt;D1)⊤∇L(θt;Dp) and let x0,x1 ∈ D with S(x0) = 0 and S(x1) = 1, and let C˜ be the clustering identical to S except on x0,x1 where each is assigned to the opposite class. Then, ∇L(θt;DS,˜ p′)⊤∇L(θt;Dp)⊤ > ∇L(θt;DS,p′)⊤∇L(θt;Dp) if ∇L(θt,x1)⊤∇L(θt;Dp) > ∇L(θt,x0)⊤∇L(θt;Dp).

Proof. Let Di = {x ∈ D : S(x) = i}, D˜i = {x ∈ D : S˜(x) = i}, ni = |Di|, and p′ be the proportions that put mass only on the maximal value of A∇L(θt;Dp). If we define

A = ∇L(θt;D0) ∇L(θt;D1)

, then

∇L(θt;DS)⊤∇L(θt;Dp) = p′⊤A∇L(θt;Dp), ∇L(θt;DS˜)⊤∇L(θt;Dp) = p′⊤A˜∇L(θt;Dp)

n1 ∇L(θt,x1)⊤∇L(θt;Dp) − n1

1

∇L(θt,x0)⊤∇L(θt;Dp) 1

= p′⊤A∇L(θt;Dp) + p′⊤

.

1

n0 ∇L(θt,x0)⊤∇L(θt;Dp) − n1

∇L(θt,x1)⊤∇L(θt;Dp)

0

Therefore swapping the classes of x0 and x1 results in an improvement for S˜ over S if ∇L(θt,x1)⊤∇L(θt;Dp) > ∇L(θt,x0)⊤∇L(θt;Dp).

| |
|---|

We immediately have the following corollary:

- Lemma 4. A skill-assigning function S : D → [m] is stable in the direction ∇L(θt;Dp) if for the skill i = arg maxi∈[m] ∇L(θt;Di)⊤∇L(θt;Dp) and any other j ∈ [m], for all xi ∈ Di, xj ∈ Dj,

∇L(θt;Dp)⊤∇L(θt,xi) ≥ ∇L(θt;Dp)⊤∇L(θt,xj).

An important fact to note is that the evaluation gradient ∇L(θt;Dp) can be arbitrary, especially if the evaluation and training data come from different distributions. To reduce the benefit of swapping points between classes, a good clustering will be stable in as many directions as possible.

A simple but noisy choice is to take all points in the convex hull to be in unique clusters, and all interior points to make up another cluster. While this clustering is stable in every direction, the classes are very small and therefore likely to be noisy, especially as training progresses and the gradient landscape shifts. A better alternative is clustering points if they can be linearly separated from the others.

Assume D can be partitioned into D0 and D1 such that f(x) = sign(v⊤x + b) is a perfect classifier, so in the direction v, S which labels the data based on the partition is stable. If f has a large margin, then many other v also a linear separators, and therefore also have S stable.

This still may be too restrictive though in general settings where data points are more spread apart. Instead, it may be good to compare the regret of skill-labeling with a sub-optimal labeling.

- Definition 3. The regret RS(i,j) under the skill-assigning function S for class j is the difference between

maxD˜

i⊂Di∪Dj,|D˜i|=|Di| ∇L(θt;Dp)⊤∇L(θt,D˜i) and ∇L(θt;Dp)⊤∇L(θt;Di).

This regret is exactly difference between the first-order loss decrease using D˜i as compared to Di, where new elements in D˜i come from Dj.

- Lemma 5. Let i,j ∈ |m| and assume |Di| = |Dj|. Assume ∇L(θt;Di)⊤∇L(θt;Dp) ≥ ∇L(θt;Dj)⊤∇L(θt;Dp),

i |∇L(θt,x)⊤∇L(θt;Dp) − ∇L(θt;Di)⊤∇L(θt;Dp)| and similarly define rj. Then

and let ri = maxx∈D

- 1

- 2

(ri + rj − (∇L(θt;Di)⊤∇L(θt;Dp) − ∇L(θt;Dj)⊤∇L(θt;Dp)))}.

RS(i,j) ≤ max{0,

Proof. Let Ri = {∇L(θt,x)⊤∇L(θt;Dp)|x ∈ Di} and Rj = {∇L(θt,x)⊤∇L(θt;Dp)|x ∈ Dj}. Also, in this notation, ri = maxx∈R

i |x − Ex∼R

[x]|, and define δ = Ex∈R

[x] − Ex∈R

[x]. The problem then reduces to

i

i

j

- 1

- 2

RS(i,j) ≤ max{0,

(ri + rj − δ)}.

Also, RS(i,j) in this one dimensional case becomes the largest over m ∈ [|Ri|] of the difference between the m largest values of Rj and the m smallest values of Ri. This is maximized over all possible Ri and Rj when half of Ri is Ex∈R

[x] − ri and the other half is Ex∈R

[x] + ri, and similarly for Rj. The difference between the max Rj and the min Ri is ri + rj − δ, and only half of these data points attain these max and min values, so RS(i,j) ≤ max{0, 12(ri + rj − δ)} as desired.

i

i

| |
|---|

This extends the case where skills i and j are linearly separable in the direction ∇L(θt;Dp). It further provides insight in how to pick skills. To reduce any pairwise regret, the radii ri and rj of the clusters from their mean should be as small as possible in every direction.

###### B.1.3 OOD evaluation clusters

When performing k-means clustering, there is a choice in clustering the training points and then assigning the evaluation points, or clustering the evaluation points and then assigning the training points. The latter choice has a major problem: evaluation clusters may have no training points near them. This causes a major dilemma for the training procedure attempting to sample from a distribution that lacks any data points.

We adopt the former method of clustering based on the training points to circumvent this issue. However, a new issues arises: evaluation data may be OOD and have no representatives in the training data. The

result is the label provided to those OOD points is the same as the closest training points. These can be quite distant and therefore not a strong representation of their gradient. However, this still is the optimal choice, as all other training points are a greater distance away and therefore have a weaker similarity. This label assignment also adds more weight to the class that is most aligned with the OOD evaluation data, increasing its sample rate to learn both the ID and OOD data for that class.

### C Algorithm details

We fully outline our algorithms for solving the optimization problems specified in Equation 2 and Equation 3, respectively. And we provide our Algorithm details in Algorithm 2 and Algorithm 3.

- Algorithm 2 R&B Skill Partitioning

- 1: Input: Training data Dtrain, Evaluation data Deval, Embedding model ψ : X → Rd,
- 2: Clustering algorithm cluster : P(Rd) × K → N × (Rd → N),
- 3: Clustering metric metric : (Rd → N) × P(Rd) → R,
- 4: Range of clustering hyperparameters K ∈ K
- 5: Output: Optimal number of clusters m∗, Optimal mapping function f∗, Partitioned datasets

{Dtrain∗ ,i}m

∗

i=1, {Deval∗ ,i}m

∗

i=1

- 6: Dtrain ← {ψ(x) : x ∈ Dtrain} ▷ Collect embeddings for training data
- 7: Deval ← {ψ(x) : x ∈ Deval} ▷ Collect embeddings for eval data
- 8: for k ∈ K do
- 9: m,f = cluster(Dtrain,k)
- 10: score = metric(f,Dtrain)
- 11: m∗,f∗,score∗ = arg maxm,f,score(score,score∗)
- 12: end for
- 13: for i = 1 to m∗ do
- 14: Dtrain∗ ,i ← {x ∈ Dtrain : f∗(x) = i} ▷ Partition training data
- 15: Deval∗ ,i ← {y ∈ Deval : f∗(y) = i} ▷ Partition evaluation data
- 16: end for
- 17: Return m∗,f∗,{Dtrain∗ ,i}m

∗

i=1, {Deval∗ ,i}m

∗

i=1

- Algorithm 3 R&B Online Data Selection

- 1: Input: Partitioned datasets {Dtrain∗ ,i}m

∗

i=1, {Deval∗ ,i}m

∗

i=1, model parameters θ ∈ Rn, training rounds T, steps per round K, evaluation proportions p′ ∈ Rm

∗

- 2: Initialize sampling distribution p0 = Uniform(m∗)
- 3: for t = 0,...,T − 1 do
- 4: for i ∈ [m∗] do
- 5: ∇L(θ;Di) ← 0n ▷ Initialize gradient accumulator for domain i
- 6: Si ← ∅ ▷ Initialize set of samples from domain i
- 7: end for
- 8: for k = 0,...,K − 1 do
- 9: Sample batch D = {xj}Bj=1 where xj ∼ Dtrain∗ ,i with i ∼ pt for each j
- 10: θtK+k+1 ← θtK+k − η∇L(θtK+k;D)
- 11: for i ∈ {f∗(d) : d ∈ D} do
- 12: ∇L(θ;Di) ← ∇L(θ;Di) + ∇L(θtK+k;D ∩ Dtrain∗ ,i)
- 13: Si ← Si ∪ (D ∩ Dtrain∗ ,i)
- 14: end for
- 15: end for
- 16: Construct G ∈ Rm

∗×m∗ where Gij = |S 1

i||Sj|∇L(θ;Di)T∇L(θ;Dj)

- 17: pt+1 ← softmax(ηλGp′)
- 18: end for
- 19: Return

Relative Compute Overhead (vs. Standard Training)

Method Total Compute Cost (FLOPs)

Standard Training 6DtN 0 Skill-It [3] 6(1 + mδ)DtN + 2(T + m)DeN mδ + (T+3Dm)De

t

Aioli [4] 6DtN + 2(Tm)DeN TmD3D e

t

DGA [27] 6(1 + mδ)DtN + 6T(δDe)N mδ + Tδ DDe

t

R&B (Ours) 6DtN + Tm2N Tm6D2

t

Table 4: Computational cost comparison of data mixing methods. We report (1) total cost of training, given under the table Total Compute Cost, and relative compute overhead over standard training. Standard training requires no additional compute overhead since its proportions are fixed. In the common setting where the number of skills m is much smaller than that of evaluation tokens De and training tokens Dt, R&B enjoys superior computational efficiency.

### D Compute cost models for online data mixing

We formalize a cost model for estimating the amount of compute required for several data mixing methods. Table 4 reports cost in terms of FLOPs, or number of floating point operations required to perform each method.

Following [33], we will use the estimate for the compute cost C = Cforward + Cbackward ≈ 2ND + 4ND, where N is the number of model parameters, and D is the number of training tokens. Here, we also make use of the empirical observation that the amount of compute for a backward pass is roughly twice that of the amount for a forward pass [34].

For all methods analyzed, we make the following assumptions:

- • Each method trains on Dt tokens across m domains,
- • Each method has access to an evaluation dataset with De tokens,
- • Training is divided into T rounds with domain reweighting between rounds,
- • Each method uses some fraction of the training dataset, δ·Dt (where δ < 1), to perform their reweighting procedure.

For our analysis, it is necessary to split the forward and backward compute costs because the data mixing algorithms we study involve a domain-reweighting mechanism that requires model evaluation on a hold-out dataset. Model evaluation only requires a forward pass, whereas model training requires both a forward and backward pass. To illustrate this point, let Dtrain be the number of tokens in the training dataset, while Deval is the number of tokens in the evaluation dataset. Training a model on all available training data has a total cost of 6NDtrain, while computing model evaluation once has a cost of 2NDeval.

##### D.1 Skill-It

Skill-It [3] has two stages in its data-mixing procedure: estimating a graph A which is used as part of its domain reweighting procedure, and training itself.

For learning A, a model is trained on each of m domains for some fraction of Dtrain, then evaluated on Deval. For comparative purposes, we will assume that δDtrain training tokens are used in this process of constructing A, for δ < 1. Furthermore, we assume these tokens are divided evenly among each of m domains. Then the compute cost for learning A is 6(δDtrain)N + 2(mDeval)N. Training is split into T rounds, and after each round, the model is re-evaluated on Deval to update the domain proportions. The compute cost for training, then, is 6(Dtrain)N + 2TDevalN. This brings the total compute cost to

6(1 + δ)DtrainN + 2(T + m)DevalN.

##### D.2 Aioli

Similar to Skill-It, Aioli [4] also includes two stages for learning A and training, but incorporates both directly into the training process. At a high level, training is also split into T rounds, where each round dedicates some fraction δ < 1 to learning A. When learning A, the model is trained on each of m domains sequentially, and re-evaluating the resulting model on the evaluation dataset. Consequently, the training compute cost for learning A is simply absorbed into the overall cost for training, but the model still must be evaluated on Deval for each domain. Within a round, this process repeats for the number of sweeps k, but here we will set k = 1 to simplify the analysis. Therefore, the total compute cost for training improves to 6(Dtrain)N, but the compute cost for evaluation increases to 2(Tm)DevalN. This brings the total compute cost to

6DtrainN + 2(Tm)DevalN.

##### D.3 DGA

Dynamic Gradient Alignment [27] instead uses gradient information to reweight the domain proportions. Their method splits training into T rounds, and reweights proportions after each round. Their procedure involves sampling a batch from each domain, and then performing a forward and backward pass to obtain gradients respective of each domain. They then obtain gradients for a batch on a specific dataset Dspe (which for consistency of analysis we will simply refer to as Deval), and computes the inner product between the gradients of each domain and that of Deval. In order to equalize model performance with Skill-It and Aioli, we will assume that a batch from each training domain contains mδ Dtrain tokens, and a batch from the specific dataset contains Deval tokens. Then, computing each domain’s gradient has a cost of 6(mδ )DtrainN, and computing the specific dataset’s gradient has a cost of 6DevalN. We assume that computing the inner product between two model gradients is linear in N so there is an additional mN compute overhead. Therefore, the total compute cost is

6(1 + δ)DtrainN + 6T(Deval + m)N.

##### D.4 R&B (ours)

Similar to all above methods, we split training into T rounds, and reweight domain proportions at the end of each round. Like Dynamic Gradient Alignment, we opt to use gradient inner product information to inform our reweighting procedure. Crucially, however, we make two observations: (1) gradients per domain can be collected on the fly during normal backpropagation, and (2) our optimization problem only requires knowledge about the respective proportions of Deval, and does not use gradient or loss information about Deval. Instead, we simply compute the equivalent of matrix A which is a Gram matrix comprised of the inner products between the gradients of each respective domain. As a result, the compute cost of training our method is simply 6(Dtrain)N, and the compute cost of evaluation is just m2N. Therefore, the total compute cost is

6(Dtrain)N + m2N.

Efficiency Analysis. Under typical conditions where the number of skills is much smaller than the size of the evaluation dataset, R&B demonstrates superior computational efficiency. Its evaluation overhead scales only with m2 rather than with De, making it particularly advantageous for scenarios with large evaluation datasets but a moderate number of domains.

When comparing specifically with DGA, R&B’s advantage depends on the relationship between the number of domains and evaluation data size. R&B is more efficient when m < √De, which holds in most practical settings. Even when m approaches or exceeds √De, R&B maintains partial efficiency benefits through its 6× lower coefficient on the evaluation term, and by avoiding the additional δ fraction for computing gradients.

### E Implementation Details

We start with an explanation of gradient computations.

##### E.1 Efficient Gradient Computation

Standard training pipelines provide per-batch gradients, but we need per-example gradients in order to aggregate per-skill gradients for our method. We perform a gradient decomposition similar to the method introduced in [35] to efficiently circumvent this. A simple application of the chain-rule means we can exactly recover per-example gradients of a linear layer in a mini-batch with just one backwards pass by multiplying an example’s input with that mini-batch’s gradient.

Adopting the notational convention from [36], let s = aW be a linear layer where W ∈ Rd

1×d2 is a weight matrix, a = (a(1),...,a(B))⊤ ∈ RB×d

is the input to the mini-batch, and s = (s(1),...,s(B))⊤ ∈ RB×d

1

2

is the layer’s pre-activation output. Denote by ℓ(i) the loss on the ith example in the mini-batch. Let ℓ denote the summed loss of the mini-batch. It follows from the chain rule that the gradient of ℓ(i) with respect to W can be expressed as

∂ℓ(i) ∂W

=

∂s(i) ∂W

∂ℓ(i) ∂s(i)

=

∂ℓ(i) ∂s(i)

∂ℓ ∂s(i)

a(i) =

a(i),

(j)

where the last equality follows from the fact that the ∂ℓ

∂s(i) terms disappear when i ≠ j. Notably, the ∂s∂ℓ

(i)

term is available through standard training, and a(i) can be easily tracked. We aggregate per-example gradients into their respective skills, allowing for efficient per-skill gradient computation.

F Experimental Details

We evaluate our method, R&B, against four baseline data mixing methods: Stratified sampling, Skill-It, Aioli, and DGA (Dynamic Gradient Alignment). We conducted experiments on three datasets of varying sizes and characteristics.

##### F.1 Datasets

- • Dolly-15k: An instruction follow-up dataset consisting of 15,000 examples with eight original skill categories.
- • Sup-NatInst (Natural Instructions In-Domain): A 285k dataset created from Natural Instructions by selecting 100 tasks out of 876 available tasks containing 38 original skill categories.
- • Sup-NatInst-Test (Natural Instructions Out-of-Domain): A 3.56M dataset created from Natural Instructions with questions and answers from domains not seen Sup-NatInst, containing 60 original skill categories.

For in-distribution datasets (NI-ID and Dolly-15k), we use 90% of the total dataset for training and 5% for testing. For regrouping experiments, we generate embeddings using ModernBERT with a dimension of 786 and cluster the datasets using k-means.

##### F.2 Experimental Configurations

Table 5: Experimental Settings Across Different Datasets

###### Parameter Dolly-15k, NI-ID, NI-OOD S1-59k

Model GPT-Neo 125M Qwen2-0.5B Training batch size 16 4 Evaluation batch size 16 16 Context Length 512 8192 Learning rate 5e-5 1e-5 Optimizer AdamW AdamW

### G Extended Training Results

In this appendix, we provide additional experimental results for training on the Dolly-15k dataset for an extended period of 40,000 steps. This allows us to understand the long-term behavior of R&B compared

Table 6: Training budget allocations and experimental settings for all datasets and methods.

Dataset Domains Method Training Steps Method-Specific Settings

Stratified 2000 full eval dataset

Skill-It 2000 200 steps for graph estimation, full eval dataset Aioli 2000 rounds=2, sweeps=1, full eval dataset DGA 2000 full eval dataset R&B 2000 full eval dataset

Original (8)

Dolly-15k

Stratified 2000 full eval dataset

Skill-It 2000 200 steps for graph estimation, full eval dataset Aioli 2000 rounds=2, sweeps=1, full eval dataset DGA 2000 full eval dataset R&B 2000 full eval dataset

Regrouped (7)

Stratified 2000 full eval dataset

Skill-It 2000 200 steps for graph estimation, full eval dataset Aioli 2000 rounds=2, sweeps=1, full eval dataset DGA 2000 full eval dataset R&B 2000 full eval dataset

Original (38)

NI-ID

Stratified 2000 full eval dataset

Skill-It 2000 200 steps for graph estimation, full eval dataset Aioli 2000 rounds=2, sweeps=1, full eval dataset DGA 2000 full eval dataset R&B 2000 full eval dataset

Regrouped (30)

Stratified 2000 full eval dataset

Skill-It 2000 200 steps for graph, full eval dataset Aioli 2000 rounds=1, sweeps=1, 50k eval samples DGA 2000 full eval dataset R&B - -

Original (60)

NI-OOD

Stratified 2000 full eval dataset

Skill-It 1000 25 steps for graph, 10k eval samples Aioli 1000 rounds=1, sweeps=1, 50k eval samples DGA 2000 full eval dataset R&B 2000 full eval dataset

Regrouped (100)

∗ Note: There is no result for R&B in the NI-OOD original column because the method requires finding training skills in the evaluation dataset. In out-of-domain settings, test skills and train skills are different, causing the Gp norm in R&B to be NaN.

#### Dolly 40k steps

2.770

static

EvaluationLoss

dga r&b aioli skillit

2.765

2.760

2.755

2.750

04k8k12k16k20k24k28k32k36k40k

Training Steps

###### Method Evaluation Loss

Stratified 2.733 DGA 2.733 Aioli 2.724 Skill-It 2.728 R&B (ours) 2.723

- Figure 4: Left: Training loss curves for Dolly-15k trained for 40,000 steps with different data mixing methods using the original category partitioning. Right: Average test loss on Dolly-15k after 40,000 training steps using original category partitioning. Highlighted values (with brown background) indicate the best overall performance.

Table 7: Training budget allocations and experimental settings for S1-59K dataset.

Dataset Domains Method Training Steps Method-Specific Settings

Stratified 500 full eval dataset

Original (6)

R&B 500 num layers to track=1, lamda=3, full eval dataset Regrouped (10)

S1-59K

Stratified 500 full eval dataset R&B 500 num layers to track=1, lamda=3, full eval dataset

[Figure 6]

- Figure 5: Domain weight evolution during training. Our method dynamically adjusts the importance of each domain throughout the training process, with Domains 1 and 5 eventually receiving the highest weights while Domains 0, 2, 3, 7, 8, and 9 are downweighted over time.

to different data mixing methods. Figure 4 shows the training loss curves for different data mixing methods over the full 40,000 steps (left) alongside the final evaluation loss after 40,000 steps (right). R&B maintains consistent performance over other data mixing methods, demonstrating the stability of our approach and achieving the best performance with a loss of 2.723.

For this extended training experiment, we focus on the original category partitioning (rather than our regrouping approach) to demonstrate R&B’s effectiveness even with pre-defined categories when given sufficient training time. We observe that all data mixing methods eventually converge to similar performance levels after sufficient training, but R&B maintains a consistent advantage throughout the training process. This suggests that our gradient-based approach effectively captures the optimal training dynamics from early stages, leading to more efficient parameter updates throughout the training process.

Furthermore, we study how R&B reweights proportions over time. As illustrated in Figure 5, our method employs a dynamic approach to domain importance throughout the training process. Initially, domain weights fluctuate significantly as the model explores the contribution of each domain to overall performance. By the midpoint of training (around steps 40-60), a clear pattern emerges with Domains 1 and 5 receiving substantially higher weights (reaching approximately 0.225) compared to other domains. Notably, while these weights gradually trend toward the evaluation distribution proportions shown in Figure 6, they never completely converge to match the actual evaluation proportions. For instance, Domain 1 maintains a training weight of around 0.225 even though its evaluation proportion is higher at 53.6%, and Domain 5 stabilizes at approximately 0.200 despite its 29.3% evaluation proportion. This deliberate partial convergence suggests that optimal performance requires a strategic balance—influenced by but not identical to the evaluation distribution.

### H Clustering Interpretation

In this section, we provide some interpretation about the groups discovered via clustering.

[Figure 7]

- Figure 6: Comparison between domain proportions in training versus evaluation data (KL Divergence: 1.04). Our method strategically reweights domain distributions during training to optimize performance, notably increasing the representation of Domains 1 and 5 while reducing emphasis on Domains 2, 3, 4, 7, 8, and 9 compared to their evaluation proportions.

|Original Category<br><br>|Regrouped Cluster|
|---|---|
|brainstorming<br><br>|Cluster 0: General knowledge and open-ended questions covering a wide range of topics from science, technology, to basic concepts|
|classification<br><br>|Cluster 1: Music-related queries focusing on instrument classification, musical theory, and instrument comparisons|
|closed QA|Cluster 2: Information extraction and summarization tasks about various topics including companies, historical figures, and specific domains|
|generation|Cluster 3: Classification tasks primarily involving animals, colors, household items, and biological categorizations|
|information extraction<br><br>|Cluster 4: Sports-related queries spanning multiple disciplines including golf, F1 racing, Olympics, and team sports|
|open QA|Cluster 5: Entertainment and pop culture queries about movies, TV shows, musicians, artists, and historical personalities|
|summarization|Cluster 6: Lifestyle and creative brainstorming queries covering diverse topics from home improvements to personal recommendations|
|creative writing| |

Table 8: Mapping between original categories and regrouped clusters

Table 8 shows the difference in groups before and after clustering on the Dolly-15k dataset. The left column displays the initial eight categories used to organize the text dataset during collection: brainstorming, classification, closed QA, generation, information extraction, open QA, summarization, and creative writing. The right column shows the seven distinct clusters that emerged when applying our clustering algorithm to the entire corpus. Interestingly, rather than following the original task-based boundaries, these clusters primarily organized around content domains, and subject matter, and subject length. This suggests that semantic content features may be more salient for learning features than the original task-based categorization framework, potentially offering new insights into how language models naturally organize information.

