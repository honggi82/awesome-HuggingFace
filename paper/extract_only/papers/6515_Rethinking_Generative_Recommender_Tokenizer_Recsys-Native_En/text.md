# arXiv:2602.02338v1[cs.IR]2Feb2026

## Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs

Yu Liang*1 Zhongjin Zhang*1 Yuxuan Zhu2 Kerui Zhang2 Zhiluohan Guo2 Wenhang Zhou2 Zonqi Yang2 Kangle Wu2 Yabo Ni3 Anxiang Zeng3 Cong Fu4 Jianxin Wang1 Jiazhi Xia1

### Abstract

|Semantic Title: Snacks<br><br>Cate: Food<br><br>Desc: ...<br><br>Title: Tableware Cate: Household Desc: ...<br><br>Title: Balloons Cate: Decor Desc: ...|Collaborative Snacks<br><br>Balloons<br><br>Tableware<br><br>Frequently Co-purchased Items<br><br>|
|---|---|

Recalled Items

|𝐶3<br><br>|
|---|

[Figure 1]

Tableware Rank:15 Hit

Top100

Balloons Rank:213 Miss

|𝐶2<br><br>|
|---|

Semantic ID (SID)-based recommendation is a promising paradigm for scaling sequential recommender systems, but existing methods largely follow a semantic-centric pipeline: item embeddings are learned from foundation models and discretized using generic quantization schemes. This design is misaligned with generative recommendation objectives: semantic embeddings are weakly coupled with collaborative prediction, and generic quantization is inefficient at reducing sequential uncertainty for autoregressive modeling. To address these, we propose ReSID, a recommendation-native, principled SID framework that rethinks representation learning and quantization from the perspective of information preservation and sequential predictability, without relying on LLMs. ReSID consists of two components: (i) Field-Aware Masked AutoEncoding (FAMAE), which learns predictivesufficient item representations from structured features, and (ii) Globally Aligned Orthogonal Quantization (GAOQ), which produces compact and predictable SID sequences by jointly reducing semantic ambiguity and prefix-conditional uncertainty. Theoretical analysis and extensive experiments across ten datasets show the effectiveness of ReSID. ReSID consistently outperforms strong sequential and SID-based generative baselines by an average of over 10%, while reducing tokenization cost by up to 122×. Code is available at https://github.com/ FuCongResearchSquad/ReSID.

Snacks Rank:156 Miss

|𝐶1|
|---|

Beam Search

Generative Recsys

No Sequential Predictability Constraints

Tableware (2,5,6)

Snacks (12,24,37)

Balloons (11,12,43)

Strong Weak

|𝐶3|
|---|

|Food Decor<br><br>Household<br><br>|Balloons<br><br>|
|---|---|
|Snacks<br><br>Tableware| |

[Figure 2]

ItemEmbDistribution

Lead To

|𝐻(𝐶3|𝐶1:2)<br><br>[Figure 3]|
|---|

|𝐶2|
|---|

|𝐻(𝐶2|𝐶1)<br><br>[Figure 4]|
|---|

|𝐶1|
|---|

Vector Quantization

Collaborative Supervision Semantic Supervision

Figure 1. Illustration of a traditional semantic-centric SID-based generative recommendation pipeline. Item representations learned from foundation models are weakly aligned with collaborative prediction, and subsequent quantization does not account for sequential predictability in SID decoding, leading to high decoding uncertainty and suboptimal recommendation performance.

### 1. Introduction

Generative recommendation based on Semantic IDs (SIDs) has emerged as a promising approach for scaling recommender systems beyond conventional item-ID modeling (Zhou et al., 2025). The key idea is to encode each item as a compact sequence of discrete tokens (e.g., [21,3,54]), enabling autoregressive decoding, token-by-token, instead of predicting billions of atomic, unrelated item-IDs.

Most existing SID pipelines follow a semantic-centric design (Rajput et al., 2023; Wang et al., 2024a;b; Xiao et al., 2025). Items are first embedded using (M)LLMs, then discretized via vector quantization (e.g., RQ-VAE (Lee et al., 2022) or Hierarchical K-Means (Nister & Stewenius, 2006)), and finally used as tokens in generative recommenders. While effective for vocabulary compression, such pipelines introduce a fundamental mismatch between SID tokenization and downstream recommender modeling, manifesting in two core limitations (illustrated in Figure 1).

*Equal contribution 1Central South University, Changsha, China 2Shopee Pte. Ltd., Shanghai, China 3Nanyang Technological University, Singapore, Singapore 4Shopee Pte. Ltd., Singapore, Singapore. Correspondence to: Jiazhi Xia <xiajiazhi@csu.edu.cn>.

(1) Misalignment in representation extraction. Foundation models are primarily optimized for semantic similarity,

Preprint. February 3, 2026.

which often conflicts with collaborative signals: Items that frequently co-occur in user behaviors (e.g., snacks and balloons for parties) may be far apart in semantic or visual properties. Even with collaboration-oriented fine-tuning (Zheng

- et al., 2024; Wang et al., 2024a;b; Xiao et al., 2025), semantic and collaborative objectives inherently impose competing geometric constraints. In practice, this leads to a compromised embedding structure that is neither semantically clean nor optimally aligned with recommender objectives.

Further, continuously injecting collaborative signals into large semantic encoders incurs substantial training costs (Zhou et al., 2025; Chen et al., 2025), and there is no principled way to assess whether the learned embeddings are suitable for recommendation-oriented SID tokenization, since they are optimized indirectly via semantic objectives.

(2) Existing quantization methods weaken sequential predictability. Existing methods typically emphasize either reconstruction fidelity or hierarchical structures, but fail to jointly consider both. In hierarchical encoding pipelines (Wang et al., 2024b; Xiao et al., 2025; Si et al., 2024), child indices are often assigned locally, arbitrarily, and independently under each parent. For example, items with substantially different semantics may share the same second-level token “1” in codes (2,1,5) and (9,1,7). Since the embedding associated with token “1” is shared across all items assigned to that index, this leads to multimodality and high semantic ambiguity, thus lowering the reconstruction fidelity (see theoretical analysis in Section 3.3). Conversely, reconstruction-driven quantizers (Rajput et al., 2023; Liu

- et al., 2025; Fu et al., 2025; Chen et al., 2025) such as RQVAE or RQ-Kmeans optimize the reconstruction error but are agnostic to sequential index dependency across code levels. As a result, the two design choices produce SID sequences that either suffer from high reconstruction error (information loss) or are unfriendly to autoregressive modeling, degrading downstream generative recommendation.

Diverging from the semantic-centric paradigm, we rethink representation learning and quantization from an information-theoretic perspective. In the representation learning stage, ReSID maximizes the mutual information between a collaboratively sequential-contextualized representation and the semantic-rich target item features, ensuring that the task-relevant information is preserved for downstream recommendation. In the quantization stage, ReSID jointly minimizes reconstruction entropy of the discrete codes and prefix-conditional entropy along the SID sequence using non-parameterized methods, thereby improving predictability in SID-based generative recommendation while reducing computational costs. Our contributions are summarized as follows:

(1) Recommendation-native representation learning. We introduce Field-Aware Masked Auto-Encoding (FAMAE),

which learns item embeddings by predicting masked structured features conditioned on the user history. Guided by our information-theoretic analysis, FAMAE preserves recommendation-sufficient information for SIDs and enables intrinsic, task-aware metrics for embedding quality.

- (2) Objective-aligned SID quantization. We propose Globally Aligned Orthogonal Quantization (GAOQ), which jointly reduces reconstruction errors and prefix-conditional uncertainty in SIDs. By enforcing globally consistent indexing across hierarchical levels, GAOQ produces compact codes that improve predictability in sequential decoding.
- (3) Strong and consistent empirical results. Across ten public datasets, ReSID consistently outperforms strong sequential recommenders and SOTA SID-based generative models, achieving over 10% relative improvement while reducing tokenization costs by up to 122× on million-scale datasets. These results show that effective SID construction does not require heavy foundation models, enabling a scalable and adaptable solution for large-scale systems.

### 2. Rethinking SID Pipeline

#### 2.1. Problem Definition and Notations

We consider a sequential recommendation setting that predicts the target item iT given the user’s history H = (i1,...,iT−1), where the index t denotes the chronological order of interactions. Each item at position t is associated with a set of raw metadata Xt (e.g., text, images, or other unstructured contents) and a set of structured feature fields Ft = {ft(1),...,ft(J)} engineered from the interaction data and metadata Xt, where ft(k) denotes the k-th feature field of it. A SID-based generative recommender extends this setting by replacing it’s identifier with a finite-length SID sequence Ct = (c1,...,cL). It mainly includes three components: (i) an encoder Eθ, parameterized by θ, which maps Xt of item it to a continuous representation zt; (ii) a quantizer Q, which discretizes zt into Ct; and (iii) a generative model Gϕ, parameterized by ϕ, which predicts the target SID, Y = CT, conditioned on the user history.

From an end-to-end perspective, the ideal learning objective can be written as:

EH,X,Y L(Y,(Gϕ ◦ Q ◦ Eθ)((Xt)Tt=1−1)) , (1)

min

θ,ϕ,Q

where L(·) denotes the cross-entropy loss.

#### 2.2. The Three-Stage SID Pipeline

A fundamental challenge in SID-based generative recommendation lies in the self-referential nature of supervision. The SIDs produced by the quantizer Q are used as training targets Y for the generative model Gϕ. Consequently, the

ℒ

1 1 2

Balanced K-Means 1 2

1

Target Hidden State

Balanced K-Means

1 1

GAOQ

- (1) (2) (3)

- (1) (3)

1 2

1

2 Anchor

Transformer Backbone

2 1 3 1 3 2

Sum Pooling

Concat

1 2 3

GAOQ

[Figure 5]

[Figure 6]

Field Embedding Table

123 ...... 123

3

XL

Hungarian matching

(1) (2) (3)

(1) (2) (3) 1

(1)

(2)

(2)

(3)

Field Mask

Assigning Index

(1)

(2)

(3)

1

3

...

1

1

1

2

1

1

2 1 3 1 3 2

Item1Item2Item3CCode(2,1,3)=Code(2,2,1)(1,2,1)

User Interaction History

Masked Fields Target Item

All Items

Field-Aware Masked Auto-Encoding Globally Aligned Orthogonal Quantization

- Figure 2. Overview of ReSID. FAMAE learns recommendation-sufficient field-level item representations via masked field prediction, and GAOQ discretizes them into compact, autoregressive-decoding-friendly SIDs via global alignment.

quality and semantic consistency of the supervision signal are entirely determined by the upstream encoder–quantizer pair. If the generated SIDs are noisy, misaligned with collaborative signals, or semantically inconsistent, the generative model is forced to learn from a distorted target distribution, with no mechanism for downstream correction.

ing and quantization with the information requirements of downstream generative recommendation.

### 3. Methodology

We present ReSID, a recommendation-native SID framework that redesigns both representation learning and quantization from an information-theoretic perspective. ReSID consists of two components: (1) Field-Aware Masked AutoEncoding (FAMAE) for learning collaboration-dominant item representations, and (2) Globally Aligned Orthogonal Quantization (GAOQ) for constructing compact and autoregressive-decoding-friendly SIDs. The overall pipeline is illustrated in Figure 2.

As a result, existing methods adopt a three-stage pipeline: (i) representation learning (E-stage), (ii) discretization into SIDs (Q-stage), and (iii) autoregressive modeling on SID sequences (G-stage). Information is inevitably compressed across these stages; therefore, the effectiveness of SID-based generation critically depends on whether the intermediate representations and SID codes preserve information that is predictive of downstream generative recommendation.

#### 3.1. Field-Aware Masked Auto-Encoding

#### 2.3. Design Principles for Effective SID Tokenization

Motivation. The goal of the E-stage is to learn item representations that are recommendation-sufficient, with semantic similarity playing a secondary role. Most existing SID approaches (Zhou et al., 2025; Chen et al., 2025) ground structured, feature-engineered information into texts or multimodal inputs and then extract embeddings using foundation models. This semantic grounding dilutes task-relevant collaborative signals and makes them particularly vulnerable to information loss during subsequent quantization.

Building on the above discussions, we argue that an effective SID pipeline should follow two principled design criteria.

First, the E-stage should be collaboration-dominant: representations must be primarily shaped by user interaction signals, with semantic information serving only as auxiliary contexts. Since discretization is inherently informationlosing, ensuring that task-relevant collaborative information dominates the representation is critical to prevent it from being blurred or discarded in the Q-stage. Accordingly, representation quality should be evaluated using task-aware metrics that monitor the preservation of both recommendationrelevant information and semantic fidelity.

From a recommendation-native perspective, structured information should instead be grounded directly in its native symbols: the engineered feature fields. This design follows a standard modeling assumption widely adopted in recommender systems, namely that the prediction target is conditionally independent of raw item metadata given sufficient structured features and user contexts: Y ⊥⊥ X | (FT,H). This assumption underlies most feature-based recommendation formulations, where long-established feature engineering and interaction modeling are treated as sufficient statistics for predicting user behaviors. FAMAE explicitly builds on this premise to align representation learning with the information requirements of the downstream generative

Second, the Q-stage should preserve as much task-relevant information as possible while explicitly encouraging a sequentially predictable structure in the resulting code sequences. In particular, SID quantization should not only minimize reconstruction distortion, but also reduce intrinsic uncertainty in autoregressive decoding by promoting sequentially predictable and semantically stable codes.

Together, these principles align the representation learn-

recommendation.

FAMAE Objective. FAMAE trains a Transformer encoder using a field-aware masked prediction objective. Given a target item at position T, a random subset of its feature fields is masked, and the model is trained to predict the masked fields conditioned on the remaining fields and the user history. Formally, the training objective is:

αk· −log qθ,k(fT(k) | hT) ,

LFAMAE(θ) = EM∼π

k∈M

where M is the set of masked fields sampled from masking policy π, and hT = gθ((Fi)Ti=1−1,{fT(j)}j /∈M) is the contextualized latent representation produced by the Transformer encoder gθ at position T, and αk controls the relative importance of different feature fields. The term qθ,k(· | h) denotes the model’s predictive distribution of a field given hidden state h, and is given as:

exp K(h,e(Tk)) v∈Vk exp K(h,e(vk))

qθ,k(fT(k) | h) =

,

where e(Tk) denotes the embedding for field k after embedding lookup e(Tk) = embθ(fT(k)), K denotes scaled cosine similarity, K(·,·) =

##### √

d · cos(·,·), and Vk is the vocabulary of field k. Note that with a mild abuse of notation, we use θ as the whole parameters of the FAMAE encoder which consists of embedding layer embθ and Transformer gθ.

Unlike traditional semantic auto-encoding, this objective enforces separable, field-level supervision, encouraging the representation to retain fine-grained and task-relevant spatial structures that are robust under the discretization.

Information-Theoretic Interpretation. Although FAMAE is implemented as a supervised masked prediction loss, it admits a natural information-theoretic interpretation. Minimizing the FAMAE loss increases a variational lower bound on the mutual information between the learned representa-

tion and FT. Formally, we have the following proposition: Proposition 3.1 (Predictive Sufficiency Proxy). Consider a masking policy π and field weights αk ≥ 0. Let wk = αk PrM∼π(k ∈ M), then

J

J

wkI(hT;fT(k)) ≥

##### wkH(fT(k)) − LFAMAE(θ).

k=1

k=1

In particular, minimizing LFAMAE increases the right-hand side, which is a variational lower bound on the maskweighted mutual information between bottleneck hT and the target item’s features.

This proposition has two main implications for FAMAE.

- (1) Predictive Sufficiency. In FAMAE optimization, the learned representation hT compresses information from FT and H, serving as the sufficient statistic if it is used by the downstream models to predict any target Y , given

Y ⊥⊥ X | (FT,H). In other words, FAMAE provides a principled proxy for learning representations that preserve the task-relevant information necessary for downstream recommendation, independent of the design of the Q-stage.

- (2) Predictive Superiority. Prior sequential recommenders (e.g., SASRec (Kang & McAuley, 2018)) are trained with a single-label objective that predicts the fused representation of the next item’s features, which can be viewed as a

deterministic coarsening uT = f(FT) of the underlying structured feature vectors. The Data Processing Inequal-

ity implies I(hT;uT) ≤ I(hT;FT) = Jk=1 I(hT;fT(k) | FT(<k)). Equality holds only if uT is a sufficient statistic of FT for predicting hT, a condition that rarely holds when heterogeneous item fields are fused with non-invertible operators (e.g., pooling or MLP fusion), which discard field identity and fine-grained structures.

Moreover, mutual predictability among multiple structured fields naturally retains task-relevant semantic structures (e.g., category hierarchy and attribute constraints), which complements collaborative signals with necessary semantic structures rather than competing with them. This makes FAMAE particularly suitable as a recommendation-native pretraining objective for the SID construction.

Item Representation Extraction. Notably, the contextual representation hT is not suitable for the SID quantization, as it entangles item information with the user history. Since hT is a deterministic function of the field-level item embeddings eT and the user history H, the Data Processing Inequality implies I(eT;FT) ≥ I(hT;FT), under the dataprocessing chain (FT → eT → hT). This indicates that eT preserves at least as much information about the target item’s structured features FT as hT, while remaining independent of user-specific contexts and retaining sufficient task-relevant information for predicting the downstream target Y . eT provides a more appropriate, task-sufficient basis for the SID construction.

Implementations. The encoder takes a sequence of items as input, where the first T −1 items correspond to the user’s interaction history and the last item serves as the prediction target. For each position, embeddings of all structured feature fields (including item-ID and side information), together with a learnable positional encoding p, are aggregated via sum pooling to form the input token representation. The resulting sequence is then fed into a Transformer backbone with the bidirectional self-attention. We adopt a two-level random masking strategy for the target item. First, we uniformly sample an integer K ∼ U{1,...,J}. Second, we randomly select a subset of K fields MK to be masked.

To preserve field identity during prediction, we introduce field-specific learnable mask tokens {m1,...,mJ}, where each mj corresponds to a field f(j). For a masked field, its embedding is replaced by the corresponding mask token.

As a result, the transformer input of the target item is constructed by:

J

e(Tj), e(Tj) =

e˜T = pT +

j=1

mj, j ∈ MK, embθ(fT(j)), j ∈/ MK.

#### 3.3. Globally Aligned Orthogonal Quantization

Ideal Objective for SID Quantization. Although the downstream generator predicts cl conditioned on (C(<l),H), user history H is not accessible in Q-stage. Therefore, the quantizer should be designed to produce codes that are intrinsically autoregressive-decoding-friendly, independent of H. From an information-theoretic perspective, an effective SID quantizer should satisfy three desiderata: (i) low global reconstruction distortion under the discrete bottleneck, (ii) high semantic contribution from each individual code, and (iii) low prefix-conditional uncertainty to facilitate sequential prediction. These requirements can be formalized as:

We construct the final representation for each item by concatenating the embeddings of all its corresponding feature fields learned by FAMAE concat({e(j)}Jj=1), which serves as the input to the subsequent SID quantization stage.

#### 3.2. Task-Aware Metrics for Embedding Quality

Existing SID pipelines lack principled, task-aware metrics for evaluating the quality of item representations learned in the E-stage, independent of the downstream Q-stage and Gstage. To address this gap, we introduce two complementary proxy metrics that measure how well the FAMAE embeddings preserve different types of information required.

- Metric 1: Collaborative Modeling Capability. We measure target item prediction accuracy under full-field masking, where all structured feature fields of the target item are masked and prediction relies solely on the user history and learned embedding space. From an information-theoretic perspective, this metric evaluates whether the learned representations are sufficient to mediate the conditional dependence between the user history H and target item FT, i.e., whether the predictive information required to infer FT from H is preserved and accessible through the embeddings.
- Metric 2: Discriminative Semantics and Spatial Structure. We measure item-ID prediction accuracy under singlefield masking, where only the item-ID field is masked. This metric evaluates whether the structured feature embeddings contain sufficient discriminative semantic information—such as category hierarchy and attribute entailment—to distinguish individual items. It serves as a proxy for assessing whether the learned representation space preserves fine-grained semantic structures required for constructing informative and stable Semantic IDs.

Together, these metrics capture complementary aspects of the embedding quality: collaborative predictability and discriminative semantic structures. Empirically (Figure 3), representations that perform well on both metrics consistently yield higher-quality Semantic IDs and improved downstream generative recommendation performance.

H(z | C) + µ

H(z | cl) + λ

H(cl | C(<l)),

min

Q

l

l

s.t. H(cl) ≈ log |cl|. (2) where H(z | C) measures the overall reconstruction uncertainty from the full SID sequence, H(z | cl) forces each code to be individually informative and prefix-invariant, and H(cl | C(<l)) captures the intrinsic branching uncertainty faced by an autoregressive decoder, serving as an upper bound on the decoding entropy of the downstream recommender: H(cl | C(<l),H) ≤ H(cl | C(<l)). The entropy constraint ensures uniformly distributed marginal codes for each layer l, preventing index collapse.

We next examine how existing quantization schemes violate these principles, before introducing GAOQ.

Misalignment of RQ-style Quantization. RQ-VAE and RQ-Kmeans only optimize overall reconstruction loss under a discrete bottleneck, often with entropy regularization to encourage balanced code usage. However, their objectives are agnostic to the autoregressive nature of SID decoding. In particular, standard residual quantization assigns codes independently across levels and does not explicitly constrain prefix-conditional uncertainty H(cl | C(<l)), nor does it encourage individual codes to carry meaningful semantic information as measured by H(z | cl). As a result, such methods can achieve low overall reconstruction distortion and balanced marginal usage, yet still produce code sequences that are unstable and difficult to predict sequentially, degrading downstream autoregressive SID modeling.

Limitations of Hierarchical K-Means with Local Indexing. Hierarchical K-Means introduces a tree-structured code by path-dependent branching, which can partially reduce prefix-conditional uncertainty H(cl | C(<l)). However, existing hierarchical schemes typically assign child indices independently and locally within each parent node. As a result, the same code index may correspond to different semantic directions under different prefixes (Figure 8).

This issue can be formalized via the decomposition below: H(z | cl) = H(z | cl,C(<l)) + I(z;C(<l) | cl). (3)

Local indexing increases the prefix-dependent ambiguity term I(z;C(<l) | cl), since the same index at level l can correspond to different semantic directions under different prefixes. This means that the semantic interpretation of a code depends heavily on its prefix. Though hierarchical refinement typically reduces the term H(z | cl,C(<l)), the increase in I(z;C(<l) | cl) can offset this gain, leading to larger H(z | cl). Consequently, individual codes become less informative and less prefix-invariant, which harms both code-level semantic stability and downstream decoding.

GAOQ Design. GAOQ is designed to jointly optimize all three objectives in Eqn. (2) by combining hierarchical vector quantization with globally aligned indexing.

First, GAOQ reduces prefix-conditional uncertainty H(cl | C(<l)) through hierarchical vector quantization, where each level refines the partition of the representation space. Conditioning on deeper codes progressively shrinks the conditional variance of z, which also contributes to lowering the global reconstruction uncertainty H(z | C).

Second, GAOQ explicitly targets prefix-dependent ambiguity I(z;C(<l) | cl) by enforcing global alignment of code indices. At each level, child cluster centroids are first centered by subtracting their parent centroid, aligning cross-prefix representations to a common origin. We then introduce a set of approximately orthogonal reference directions shared across all parent nodes. Code indices are assigned by matching these centered vectors to the reference directions using Hungarian Matching based on cosine similarity. This procedure ensures that the same index within each level corresponds to a consistent semantic direction across different prefixes. This reduces both I(z;C(<l) | cl) and H(z | cl,C(<l)) and leads to lower l H(z | cl).

Together, GAOQ minimizes all three terms in Eqn. (2) while enforcing the entropy constraint on H(cl) via balanced KMeans. As a result, it produces compact, semantically stable, and autoregression-friendly SIDs that preserve task-relevant information for downstream generative recommendation. See Algorithm 1 in the appendix for more details.

### 4. Experimental Settings

We conduct extensive experiments to evaluate the effectiveness, interpretability, and efficiency of ReSID.

Specifically, we seek to answer the following Research Questions: (i) Does ReSID outperform strong sequential and SID-based baselines when side information is accessible? (ii) How do FAMAE and GAOQ individually contribute to the overall performance? (iii) Do the proposed task-aware embedding metrics reliably predict downstream SID performance? (iv) Does ReSID improve the efficiency of SID tokenization relative to existing SID pipelines?

Datasets. We evaluate ReSID on ten subsets of the Amazon2023 review dataset (Hou et al., 2024), including Musical Instruments, Video Games, Industrial & Scientific, Baby Products, Arts, Crafts & Sewing, Sports & Outdoors, Toys & Games, Health & Household, Beauty & Personal Care, and Books. We follow standard practice (Wang et al., 2024a;b; Liu et al., 2025) to preprocess the datasets. See detailed preprocessing steps and dataset statistics in Appendix C.

Fair comparison between SID- and item-ID-based recommendation. A critical but often overlooked confounder in prior SID evaluations is that SID pipelines typically exploit rich item metadata, while sequential baselines are trained with item-IDs only—making reported gains difficult to attribute to the modeling paradigm itself. Based on this, we choose our baselines as follows.

Compared Methods. We compare ReSID with three categories of baselines: (i) Sequential recommenders (item-IDonly): HGN (Ma et al., 2019), SASRec (Kang & McAuley, 2018), BERT4Rec (Sun et al., 2019), and S3-Rec (Zhou et al., 2020). (ii) Sequential recommenders with structured features: the corresponding ∗ variants (e.g., HGN∗) of the above models augmented with side-info fields. (iii) SIDbased generative recommenders: TIGER (Rajput et al.,

- 2023), LETTER (Wang et al., 2024a), EAGER (Wang et al.,
- 2024b), UNGER (Xiao et al., 2025), and ETEGRec (Liu et al., 2025), which tokenize items into SIDs and perform autoregressive generation. Detailed descriptions are provided in Appendix D. Implementation details and hyperparameter settings are provided in Appendix E.

Evaluation Settings. Following prior work (Rajput et al., 2023; Wang et al., 2024a; Liu et al., 2025), we evaluate recommendation performance using Recall@K and NDCG@K with K ∈ {5,10}. We adopt the leave-one-out protocol in which each user’s last interaction is used for testing and the second-to-last for validation.

### 5. Experimental Results

#### 5.1. Overall Results under Fair Comparison

Table 1 reports overall average relative improvements of ReSID over baselines on ten Amazon-2023 subsets (full results in Appendix F). We have the following observations.

- 5.1.1 Overall performance. ReSID consistently achieves the best performance across all metrics and datasets. To the best of our knowledge, ReSID is the first SID-based method that consistently outperforms strong item-ID–based sequential recommenders even when they are augmented with structured side information.
- 5.1.2 Impact of structured features on sequential baselines. Augmenting sequential recommenders with structured feature fields yields substantial gains: models such as

Table 1. Main results (ReSID’s average relative improvement over baselines, averaged over ten datasets for each metric). The smallest improvement is in bold, indicating the strongest baseline. See Appendix Table 4 for full results.

Model R@5 R@10 N@5 N@10

HGN 79.91% 62.16% 97.21% 80.51% SASRec 47.84% 37.52% 56.14% 47.56% BERT4Rec 52.63% 41.73% 60.08% 51.43% S3-Rec 39.56% 29.13% 47.18% 38.58%

HGN∗ 81.97% 67.86% 90.41% 79.29% SASRec∗ 13.05% 3.75% 29.24% 18.38% BERT4Rec∗ 20.63% 12.32% 27.45% 20.41% S3-Rec∗ 20.76% 12.79% 28.93% 21.57%

TIGER 22.21% 19.47% 23.66% 21.50% LETTER 16.03% 13.81% 16.17% 14.86% EAGER 37.30% 31.17% 39.02% 35.24% UNGER 30.44% 26.51% 31.82% 29.18% ETEGRec 47.07% 40.30% 50.94% 45.86%

SASRec∗ and BERT4Rec∗ significantly outperform their ID-only variants and often match or exceed prior SID-based pipelines. This indicates that many previously reported SID improvements arise from additional side information rather than the tokenization paradigm itself, highlighting the importance of matching side-info augmentations for fairness.

- 5.1.3 ReSID vs. prior SID-based methods. Compared with the best SID baseline, LETTER, ReSID achieves average improvements of 16.0%/13.8% on Recall@5/10 and 16.2%/14.9% on NDCG@5/10. In contrast, prior SID methods are on par with and often inferior to SASRec∗. These show that effective SID tokenization needs objective alignment throughout E-, Q-, and G-stages, and validate ReSID’s successful shift beyond semantic-centric paradigms.
- 5.1.4 Effect of collaborative signal injection. Methods that incorporate collaborative signals during SID tokenization (LETTER, ReSID) consistently outperform purely semantic tokenizers such as TIGER. ReSID further improves over LETTER by preserving collaborative information throughout both representation learning and quantization, rather than balancing it with semantic objectives as in LETTER, EAGER, and UNGER. These results suggest that maintaining collaborative structures is more critical for reducing autoregressive decoding uncertainty, while ReSID retains only the minimal necessary semantics through structured, feature-based representation learning with FAMAE.
- 5.1.5 “End-to-end” SID learning is suboptimal. ETEGRec, which jointly optimizes SID tokenization and downstream recommendation loss, underperforms ReSID significantly despite its tighter coupling. This supports our analysis (Sec. 2) that end-to-end SID supervision is inherently unstable: since SIDs serve as both intermediate representa-

Table 2. Ablation study (ReSID’s average relative improvement over its variants, averaged over three datasets for each metric). See Appendix Table 5 for full results.

Variants R@5 R@10 N@5 N@10

- E1 5.40% 4.60% 4.12% 3.86%
- E2 11.05% 8.45% 10.73% 9.22%
- E3 12.38% 8.76% 11.68% 9.61%

- Q1 5.64% 4.18% 4.88% 4.16%
- Q2 5.15% 1.41% 4.72% 2.49%

tions and training targets, directly backpropagating task loss through the quantization stage can distort the code space. ReSID avoids this pitfall by decoupling representation learning, quantization, and recommender stages, yielding more stable and predictive tokenization.

#### 5.2. Ablation Study

To disentangle the contributions of the E-stage (FAMAE) and the Q-stage (GAOQ), we conduct ablations on three Amazon-2023 datasets and report results in Table 2.

Variants. We consider the following controlled replacements. E-stage ablations (GAOQ fixed): (1) E1 (LLM w/ GAOQ): replacing FAMAE with LLM embeddings; (2) E2 (SASRec w/ GAOQ): replacing FAMAE with SASRec representations (collaborative-centric); (3) E3 (BERT4Rec w/ GAOQ): replacing FAMAE with BERT4Rec representations; Q-stage ablations (FAMAE fixed): (4) Q1 (FAMAE w/ RQ-VAE): replacing GAOQ with RQ-VAE; (5) Q2 (FAMAE w/ Hierarchical K-Means). We compare these variants with ReSID. The full results are shown in Table 5.

Observations. ReSID consistently outperforms all ablated variants. Compared with E1–E3, ReSID yields substantial gains, validating the predictive sufficiency analysis of FAMAE: neither purely semantic embeddings nor purely collaborative representations are sufficient for downstream SID-based recommendation tasks. ReSID’s advantage over E2 and E3 further supports the importance of preserving structured feature identity, which is lost in standard sequence encoders, supporting our predictive superiority analysis.

On the quantization side, ReSID outperforms Q1, confirming that GAOQ better reduces autoregressive decoding uncertainty than RQ-VAE, and that minimizing reconstruction error alone is insufficient for downstream tasks. ReSID outperforms Q2, demonstrating the necessity of global index alignment to reduce the reconstruction error and index ambiguity beyond the locally indexed hierarchical K-Means.

Overall, the strongest results are obtained when recommendation-native representations and globally aligned quantization are combined to preserve task-relevant

0.065

DownstreamR@10

Metric2(R@10)

0.65

0.063

0.061

0.45

0.059

0.01 0.02 0.03 0.04

Metric 1 (R@10)

0.90

0.044

DownstreamR@10

Metric2(R@10)

0.75

0.043

0.042

0.60

0.041

0.015 0.020 0.025 0.030

Metric 1 (R@10)

- Figure 3. Downstream R@10 at selected FAMAE training steps plotted against Metric 1 (R@10 of the target item when all fields are masked). The right y-axis shows the corresponding Metric 2 (R@10 of the target item-ID when only the item-ID field is masked). Left: Musical Instruments. Right: Baby Products.

- Table 3. Running time comparison of the quantization stage for different methods on various datasets (in minutes).

###### Dataset LETTER TIGER ReSID

Arts Crafts & Sewing 3356.64 224.29 27.38 Health & Household 6537.30 371.71 71.89 Beauty & Personal Care 7379.83 423.89 95.96

information while improving sequential predictability.

#### 5.3. Item Embedding Quality Metrics

As introduced in Sec. 3.2, we evaluate whether the proposed task-aware proxy metrics reflect downstream SID performance. We conduct experiments on two datasets, Musical Instruments and Baby Products, tracking Metric 1 and Metric 2 at multiple checkpoints during FAMAE training and examining their relationship with downstream results under a fixed Q/G-stage setup.

Specifically, for each dataset, we select several intermediate checkpoints, freeze the learned item embeddings, construct SIDs using the same GAOQ quantizer, and train an identical SID generator. As shown in Fig. 3, downstream R@10 consistently increases as both metrics improve across both datasets, indicating a strong positive association between embedding quality and SID performance. These results demonstrate that the proposed metrics serve as lightweight, task-aware diagnostics for E-stage FAMAE representations without requiring repeated end-to-end retraining.

#### 5.4. Efficiency of SID Tokenization

We evaluate SID tokenization efficiency by measuring the wall-clock runtime of the quantization stage under the same hardware setting, reported in Table 3. We compare ReSID with TIGER and LETTER as representative training-based SID methods: TIGER relies on purely semantic tokenization, while LETTER injects collaborative signals through RQ-VAE training and represents the strongest prior baseline.

ReSID is the most efficient tokenizer among all compared methods. LETTER is the slowest, requiring 77×–122×

more time than ReSID due to its expensive optimizationbased tokenizer training. TIGER is also slower, incurring approximately a 5× overhead relative to ReSID. These results demonstrate that ReSID improves not only recommendation effectiveness but also the computational efficiency and practicality of SID tokenization at scale.

We omit the runtime of the representation learning. FAMAE can be trained efficiently, with a cost comparable to light sequential models like SASRec. In contrast, prior SID pipelines rely on heterogeneous encoders (e.g., pretrained sequential models and large foundation models) whose costs are either amortized or unspecified, making a fair and meaningful time comparison difficult.

#### 5.5. Additional Experimental Results

Due to space constraints, we report additional experimental analyses in the appendix. Specifically: (1) Section G studies ReSID’s sensitivity to key hyperparameters; (2) Section H shows that SIDs produced by ReSID exhibit more favorable scaling behavior than semantic-centric tokenizers; (3)

- Section I visualizes item embeddings learned by FAMAE, demonstrating that FAMAE uniquely preserves both taskrelevant semantic structures and collaborative patterns; (4)
- Section J validates how FAMAE’s design aligns with downstream sequential decoding objectives; (5) Section K.1 provides direct evidence that GAOQ reduces reconstruction ambiguity through global index alignment.

### 6. Limitations and Future Work

While FAMAE provides task-aware metrics for embedding quality, principled diagnostics for GAOQ remain an open challenge. Moreover, although ReSID improves SID construction, SID-based generative models converge substantially more slowly (tens of times) than item-ID-based methods such as SASRec. We leave them for future work.

### 7. Conclusion

We introduce ReSID, a principled recommendation-native SID framework that aligns representation learning and quantization objectives with the information requirements of generative recommendation. Through FAMAE and GAOQ, ReSID produces compact and predictable SIDs efficiently without foundation models. Both theoretical analysis and extensive empirical results demonstrate the superiority of ReSID, which, to the best of our knowledge, is the first SID-based approach to outperform strong item-ID baselines augmented with side information.

### Acknowledgements

This paper is partially supported by National Natural Science Foundation of China (NO. U23A20313, 62372471) and The Science Foundation for Distinguished Young Scholars of Hunan Province (NO. 2023JJ10080). We are grateful for resources from the High Performance Computing Center of Central South University.

### Impact Statement

This paper advances machine learning for recommender systems, specifically in representation learning and discrete tokenization for recommender systems. The techniques proposed are methodological in nature and are intended to improve the performance and scalability of recommendation models. We do not foresee any significant negative societal or ethical consequences arising directly from this work.

### References

Blondel, V. D., Guillaume, J.-L., Lambiotte, R., and Lefebvre, E. Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment, 2008(10):P10008, 2008.

Chen, B., Guo, X., Wang, S., Liang, Z., Lv, Y., Ma, Y., Xiao, X., Xue, B., Zhang, X., Yang, Y., et al. Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search. arXiv preprint arXiv:2509.03236, 2025.

Fu, K., Zhang, T., Xiao, S., Wang, Z., Zhang, X., Zhang, C., Yan, Y., Zheng, J., Li, Y., Chen, Z., et al. Forge: Forming semantic identifiers for generative retrieval in industrial datasets. arXiv preprint arXiv:2509.20904, 2025.

Hidasi, B., Karatzoglou, A., Baltrunas, L., and Tikk, D. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939, 2015.

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., and McAuley, J. Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952, 2024.

Hou, Y., Li, J., Shin, A., Jeon, J., Santhanam, A., Shao, W., Hassani, K., Yao, N., and McAuley, J. Generating long semantic ids in parallel for recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 956–966, 2025.

Johnson, J., Douze, M., and J´egou, H. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7

(3):535–547, 2019.

Kang, W.-C. and McAuley, J. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM), pp. 197–206. IEEE, 2018.

Lee, D., Kim, C., Kim, S., Cho, M., and Han, W.-S. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11523–11532, 2022.

Li, J., Ren, P., Chen, Z., Ren, Z., Lian, T., and Ma, J. Neural attentive session-based recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, CIKM ’17, pp. 1419–1428, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450349185. doi: 10.1145/3132847.3132926. URL https://doi.

org/10.1145/3132847.3132926.

Li, J., Wang, M., Li, J., Fu, J., Shen, X., Shang, J., and McAuley, J. Text is all you need: Learning language representations for sequential recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1258–1267, 2023.

Liu, E., Zheng, B., Ling, C., Hu, L., Li, H., and Zhao, W. X. Generative recommender with end-to-end learnable item tokenization. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 729–739, 2025.

Ma, C., Kang, P., and Liu, X. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 825–833, 2019.

Maaten, L. v. d. and Hinton, G. Visualizing data using t-sne. Journal of machine learning research, 9(Nov): 2579–2605, 2008.

McInnes, L., Healy, J., and Melville, J. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.

Ni, J., Abrego, G. H., Constant, N., Ma, J., Hall, K., Cer, D., and Yang, Y. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. In Findings of the association for computational linguistics: ACL 2022, pp. 1864–1874, 2022.

Nister, D. and Stewenius, H. Scalable recognition with a vocabulary tree. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), volume 2, pp. 2161–2168. Ieee, 2006.

Rajput, S., Mehta, N., Singh, A., Hulikal Keshavan, R., Vu, T., Heldt, L., Hong, L., Tay, Y., Tran, V., Samost, J., et al. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems, 36: 10299–10315, 2023.

Si, Z., Sun, Z., Chen, J., Chen, G., Zang, X., Zheng, K., Song, Y., Zhang, X., Xu, J., and Gai, K. Generative retrieval with semantic tree-structured identifiers and contrastive learning. In Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pp. 154–163, 2024.

Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., and Jiang, P. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management, pp. 1441–1450, 2019.

Tan, Y. K., Xu, X., and Liu, Y. Improved recurrent neural networks for session-based recommendations. In Proceedings of the 1st workshop on deep learning for recommender systems, pp. 17–22, 2016.

Tang, J. and Wang, K. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, WSDM ’18, pp. 565–573, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450355810. doi: 10.1145/3159652.3159656. URL https://doi.

org/10.1145/3159652.3159656.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang, W., Bao, H., Lin, X., Zhang, J., Li, Y., Feng, F., Ng, S.-K., and Chua, T.-S. Learnable item tokenization for generative recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 2400–2409, 2024a.

Wang, Y., Xun, J., Hong, M., Zhu, J., Jin, T., Lin, W., Li, H., Li, L., Xia, Y., Zhao, Z., et al. Eager: Twostream generative recommender with behavior-semantic collaboration. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 3245–3254, 2024b.

Wu, S., Tang, Y., Zhu, Y., Wang, L., Xie, X., and Tan, T. Session-based recommendation with graph neural networks. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’19/IAAI’19/EAAI’19. AAAI Press, 2019. ISBN 978-1-57735-809-1. doi: 10.1609/aaai.v33i01.3301346. URL https://doi.

org/10.1609/aaai.v33i01.3301346.

Xiao, L., Wang, H., Wang, C., Ji, L., Wang, Y., Zhu, J., Dong, Z., Zhang, R., and Li, R. Unger: Generative recommendation with a unified code via semantic and collaborative integration. ACM Transactions on Information Systems, 44(2):1–31, 2025.

Yang, L., Paischer, F., Hassani, K., Li, J., Shao, S., Li, Z. G., He, Y., Feng, X., Noorshams, N., Park, S., Long, B., Nowak, R. D., Gao, X., and Eghbalzadeh, H. Unifying generative and dense retrieval for sequential recommendation, 2024. URL https://arxiv.org/abs/2411.

18814.

Yang, Y., Ji, Z., Li, Z., Li, Y., Mo, Z., Ding, Y., Chen, K., Zhang, Z., Li, J., Li, S., et al. Sparse meets dense: Unified generative recommendations with cascaded sparse-dense representations. arXiv preprint arXiv:2503.02453, 2025.

Zheng, B., Hou, Y., Lu, H., Chen, Y., Zhao, W. X., Chen, M., and Wen, J.-R. Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE), pp. 1435–1448. IEEE, 2024.

Zhou, G., Deng, J., Zhang, J., Cai, K., Ren, L., Luo, Q., Wang, Q., Hu, Q., Huang, R., Wang, S., et al. Onerec technical report. arXiv preprint arXiv:2506.13695, 2025.

Zhou, K., Wang, H., Zhao, W. X., Zhu, Y., Wang, S., Zhang, F., Wang, Z., and Wen, J.-R. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings of the 29th ACM international conference on information & knowledge management, pp. 1893–1902, 2020.

Zhu, J., Jin, M., Liu, Q., Qiu, Z., Dong, Z., and Li, X. Cost: Contrastive quantization based semantic tokenization for generative recommendation. In Proceedings of the 18th ACM Conference on Recommender Systems, pp. 969–974, 2024.

HGN0.03090.04980.01920.02530.04080.06850.02480.03370.02140.03610.01320.01790.01640.02760.01010.01370.01900.03140.01190.0159

SASRec0.03200.05280.01990.02650.05090.08200.03220.04220.02160.03490.01330.01760.02060.03380.01310.01740.02180.03530.01380.0182

BERT4Rec0.03240.05170.02060.02680.04790.07830.03030.04010.02070.03410.01310.01740.02040.03380.01290.01720.02110.03430.01340.0176

TIGER0.03850.05920.02510.03180.05690.08840.03740.04750.02970.04600.01920.02440.02560.04060.01690.02170.02850.04460.01870.0239

EAGER0.03270.05230.02100.02730.05470.08630.03570.04580.02520.04060.01630.02120.01990.03310.01260.01680.02400.03750.01550.0198

UNGER0.03620.05670.02330.02990.05470.08590.03540.04540.02320.03740.01500.01960.02210.03550.01440.01870.02530.03950.01640.0210

ETEGRec0.03720.05790.02440.03100.05580.08690.03670.04660.02510.03930.01660.02120.02290.03690.01510.01960.02490.03890.01620.0206

3S-Rec0.03360.05430.02120.02780.04950.08010.03150.04130.02100.03490.01350.01790.02130.03600.01350.01820.02280.03700.01450.0191

0.04170.06450.02730.03460.05970.09270.03960.05010.03400.02180.02730.02850.04410.01860.02360.03130.04850.02050.0261ReSID0.0512

LETTER0.04060.06230.02690.03380.05920.09140.03890.04930.03070.04820.01970.02540.02610.04160.01710.02210.02970.04640.01960.0249

0.0538∗SASRec0.03970.06390.02480.03260.05320.09140.02950.04170.03330.01850.02500.02570.04250.01570.02110.02750.04640.01570.0218

∗HGN0.02870.04690.01830.02420.03310.05410.02120.02800.01920.03210.01220.01630.01690.02780.01080.01430.01660.02710.01070.0141

∗BERT4Rec0.03770.06140.02410.03170.05370.08790.03370.04470.02960.04880.01860.02480.02410.03900.01550.02020.02750.04390.01760.0229

3∗S-Rec0.03800.06110.02420.03170.05160.08430.03260.04310.02740.04410.01690.02230.02370.03860.01520.02000.02610.04200.01680.0219

Outdoors,TG=Toys&Games,HH=Health&Household,BPC=Beauty&PersonalCare,BK=Books.Foreachdataset,wereportRecall(R@5,R@10)andNDCG(N@5,N@10).

R@5R@10N@5N@10R@5R@10N@5N@10R@5R@10N@5N@10R@5R@10N@5N@10R@5R@10N@5N@10

Table4.Fullmainresults.Datasets:MI=MusicalInstruments,VG=VideoGames,IS=Industrial&Scientific,BP=BabyProducts,ACS=Arts,Crafts&Sewing,SO=Sports&

ModelMIVGISBPACS

(A)MI/VG/IS/BP/ACS

boldThebestresultineachcolumnisshownin,andthesecond-bestisunderlined.

LETTER0.01870.02880.01240.01560.02210.03400.01470.01850.01800.02770.01210.01530.02030.03150.01330.01690.03080.04570.02090.0256

HGN0.01120.01840.00700.00930.01290.02140.00770.01050.01120.01890.00680.00930.01230.02070.00760.01030.02030.03670.01130.0166

SASRec0.01400.02310.00890.01180.01690.02700.01050.01380.01370.02270.00870.01150.01570.02540.01010.01320.02810.04590.01700.0227

BERT4Rec0.01310.02160.00820.01100.01590.02550.01010.01310.01350.02240.00860.01140.01500.02460.00950.01260.02800.04560.01730.0230

TIGER0.01800.02790.01190.01510.02140.03220.01410.01760.01800.02760.01210.01520.01920.02990.01250.01600.02650.04080.01710.0219

UNGER0.01640.02600.01090.01400.02000.03020.01320.01650.01680.02580.01120.01410.01820.02860.01190.01520.03090.04680.02030.0254

ETEGRec0.01480.02360.00970.01250.01780.02700.01160.01460.01430.02290.00920.01200.01770.02740.01150.01460.01760.02840.01100.0145

###### 0.02100.03230.01370.01740.02610.01740.02160.02080.03170.01400.01750.02300.03570.01500.01910.05300.07370.03690.0436ReSID0.0392

3S-Rec0.01530.02490.00970.01280.01860.02970.01170.01530.01500.02450.00940.01250.01710.02820.01100.01450.03180.05300.01820.0251

EAGER0.01450.02360.00950.01250.01910.02890.01270.01590.01380.02230.00890.01160.01640.02650.01070.01390.03420.04970.02340.0284

0.0401∗SASRec0.01970.03190.01220.01610.02430.01270.01780.01760.02870.01080.01440.02080.03400.01260.01690.03690.06290.02020.0285

∗HGN0.01210.01970.00780.01020.01190.01930.00750.00980.01240.02020.00780.01030.01370.02190.00880.01150.02310.03970.01380.0191

∗BERT4Rec0.01750.02860.01110.01470.02230.03560.01380.01810.01680.02740.01060.01400.01920.03120.01230.01610.03370.05640.01990.0272

3∗0.06390.02070.0290S-Rec0.01840.02970.01170.01540.02090.03380.01280.01700.01700.02740.01070.01400.01980.03220.01250.01650.0381

R@5R@10N@5N@10R@5R@10N@5N@10R@5R@10N@5N@10R@5R@10N@5N@10R@5R@10N@5N@10

ModelSOTGHHBPCBK

(B)SO/TG/HH/BPC/BK

Table 5. Full Ablation study. Datasets: MI = Musical Instruments, IS = Industrial & Scientific, BP = Baby Products. Each dataset reports Recall (R@5, R@10) and NDCG (N@5, N@10). The best result in each column is shown in bold.

Model MI IS BP R@5 R@10 N@5 N@10 R@5 R@10 N@5 N@10 R@5 R@10 N@5 N@10

- E1 0.0413 0.0629 0.0272 0.0341 0.0303 0.0472 0.0198 0.0252 0.0276 0.0428 0.0183 0.0232
- E2 0.0400 0.0621 0.0261 0.0332 0.0320 0.0496 0.0208 0.0264 0.0232 0.0373 0.0152 0.0197
- E3 0.0400 0.0624 0.0261 0.0333 0.0309 0.0488 0.0201 0.0258 0.0232 0.0373 0.0153 0.0198

- Q1 0.0396 0.0609 0.0260 0.0328 0.0320 0.0497 0.0207 0.0264 0.0271 0.0424 0.0179 0.0228
- Q2 0.0390 0.0630 0.0255 0.0332 0.0320 0.0505 0.0206 0.0266 0.0279 0.0438 0.0184 0.0235 ReSID 0.0417 0.0645 0.0273 0.0346 0.0340 0.0512 0.0218 0.0273 0.0285 0.0441 0.0186 0.0236

### A. Proof for Proposition 3.1

Proof. For each field k we denote the true conditional distribution as pk(fT(k) | hT), and the approximated distribution as qθ,k(fT(k) | hT). Given the negative log-likelihood for field k as:

Lk(θ) := E − log qθ,k(fT(k) | hT) , Using the standard cross-entropy decomposition, we have:

Lk(θ) = H(fT(k) | hT) + E DKL(pk(fT(k) | hT)||qθ,k(fT(k) | hT)) . Since the KL divergence is non-negative, we have:

−Lk(θ) ≤ −H(fT(k) | hT) According to mutual information identity, we have:

I(hT;fT(k)) = H(fT(k)) − H(fT(k) | hT) ≥ H(fT(k)) − Lk(θ).

Multiply both sides with the predefined mask-weighted coefficient wk(≥ 0) and sum over k, we have:

J

J

J

wkI(hT;fT(k)) ≥

wkH(fT(k)) −

wkLk(θ)

k=1

k=1

k=1

J

wkH(fT(k)) − LFAMAE(θ).

=

k=1

Since Jk=1 wkH(fT(k)) is independent of θ, minimizing LFAMAE(θ) monotonically increases a variational lower bound on the mask-weighted sum Jk=1 wkI(hT;fT(k)).

| |
|---|

### B. Related Work

#### B.1. Sequential Recommendation.

Sequential recommendation aims to model users’ evolving preferences from their historical interaction sequences to predict future items. Early neural approaches (Li et al., 2017; Tang & Wang, 2018; Hidasi et al., 2015; Wu et al., 2019; Tan et al., 2016), such as GRU4Rec (Hidasi et al., 2015), adopt GRU-based RNNs to encode user behavior sequences. Recently, Transformer (Vaswani et al., 2017)-based models have been introduced into sequential recommendation. SASRec (Kang & McAuley, 2018) employs a unidirectional self-attention mechanism to model item–item dependencies across the entire sequence. Inspired by masked language modeling, BERT4Rec (Sun et al., 2019) formulates sequential recommendation as a masked item prediction task and leverages bidirectional self-attention to learn contextualized item representations. Building

upon this framework, S3-Rec (Zhou et al., 2020) further enhances representation learning by incorporating multiple selfsupervised pre-training objectives. Despite architectural differences, these methods generally follow a common framework: they embed each item into a high-dimensional embedding, summarize user behavior into a sequence representation, and predict the next item by scoring candidate items—typically via dot-product (or cosine) similarity between the sequence representation and item embeddings, combined with approximate nearest neighbor (ANN) search (Johnson et al., 2019) for efficient retrieval.

#### B.2. SID-based Generative Recommendation.

Traditional sequential recommendation models assign each item a dedicated embedding and retrieve items via similarity matching (Li et al., 2023). However, this design necessitates massive embedding tables and requires an exhaustive comparison across the entire item space during retrieval, leading to substantial memory overhead and high computational costs (Rajput et al., 2023; Yang et al., 2024). In contrast, SID-based Generative Recommendation represents items as discrete identifiers and formulates recommendation as a sequence generation problem (Hou et al., 2025), directly generating the target item, avoiding explicit similarity search over the full item vocabulary.

TIGER (Rajput et al., 2023) is a pioneering work in this direction. It follows a pipeline that first derives item representations from textual information, then encodes them into discrete SIDs using RQ-VAE, and finally employs an encoder–decoder model to generate the SIDs of the target item conditioned on historical interactions. During inference, candidate SIDs are generated via beam search and mapped back to items for Top-K recommendation.

Following this pipeline, a large body of subsequent work focuses on optimizing different components of the pipeline, with particular emphasis on improving the quality and expressiveness of SIDs. For example, CoST (Zhu et al., 2024) improves the training objective of RQ-VAE to preserve important neighborhood structures among items, while LIGER (Yang et al., 2024) and COBRA (Yang et al., 2025) enable generative models to jointly model item SIDs and dense representations, providing richer item information and thereby improving prediction accuracy. Several methods further inject collaborative signals into SID learning. LETTER (Wang et al., 2024a) injects collaborative supervision directly into the RQ-VAE-based SID learning process; EAGER (Wang et al., 2024b) separately learns discrete identifiers for semantic content and user behavior; and UNGER (Xiao et al., 2025) learns item representations by jointly modeling semantic and collaborative information before deriving SIDs. In addition, ETEGRec (Liu et al., 2025) aligns the learning of item identifiers with the recommendation objective in an end-to-end way, enabling SIDs to encode information that is more directly optimized for downstream recommendation.

Although prior work such as LETTER, EAGER, and UNGER explores different ways to incorporate collaborative signals into the original learning process, these methods primarily inject collaborative supervision by adding auxiliary objectives, which are not fully consistent with the original learning objective, leading to conflicting optimization signals. As a result, the learned identifiers are still not explicitly optimized to preserve the task-relevant information required by downstream generative recommendation. ETEGRec further pushes this direction by coupling identifier learning with downstream generative recommendation in an end-to-end manner. However, the continuously evolving identifiers make the generative model’s inputs and targets non-stationary, causing strong coupling and mutual interference between SID learning and sequence generation. ReSID differs by providing an information-theoretic framework that jointly aligns representation learning and SID quantization under a unified objective, resulting in a simpler and more stable learning pipeline and consistently strong empirical performance.

### C. Dataset

Following prior work (Wang et al., 2024a;b; Liu et al., 2025), we adopt the standard 5-core setting, where users and items with fewer than five interactions are removed. The remaining interactions are chronologically ordered to construct user behavior sequences, and a leave-one-out strategy is employed for evaluation. For training, we employ a sliding window strategy with a maximum sequence length of 32. In cases where a target item in the evaluation set does not appear in the training data, it is added to the training set to ensure valid evaluation. In addition, we extract four types of structured side information for each item, including the store identifier and the first-, second-, and third-level category identifiers. Items lacking any of these features are filtered out. Table 6 reports detailed statistics of the resulting datasets.

Table 6. Statistics of the Datasets.

Dataset #Users #Items #Interactions Density Musical Instruments 57,359 23,742 490,522 0.036% Video Games 94,515 24,685 772,218 0.033% Industrial & Scientific 50,886 25,142 394,989 0.031% Baby Products 150,642 35,024 1,189,171 0.023% Arts Crafts & Sewing 196,980 87,449 1,706,484 0.010% Sports & Outdoors 409,309 151,411 3,333,753 0.005% Toys & Games 431,411 156,537 3,652,250 0.005% Health & Household 796,014 183,230 6,860,582 0.005% Beauty & Personal Care 712,259 193,383 5,785,124 0.004% Books 775,503 485,218 8,680,494 0.002%

- D. Compared Methods We compare ReSID with representative sequential recommendation models and recent generative recommendation methods. Sequential recommendation methods.

- • HGN (Ma et al., 2019) models user preferences by jointly capturing short-term and long-term interests through a hierarchical gating mechanism over interaction sequences.
- • SASRec (Kang & McAuley, 2018) employs a unidirectional Transformer with self-attention to model sequential dependencies and predicts the next item from historical interactions.
- • BERT4Rec (Sun et al., 2019) introduces bidirectional Transformer encoding with masked item prediction, leveraging both left and right context to learn sequence representations.
- • S3-Rec (Zhou et al., 2020) enhances sequential recommendation via multiple self-supervised learning objectives based on mutual information maximization, improving representation learning for next-item prediction. For S3-Rec, we follow the standard two-stage pipeline with self-supervised pretraining followed by SASRec-style fine-tuning on item-ID sequences.

#### Generative recommendation methods.

- • TIGER (Rajput et al., 2023) utilizes a pretrained text encoder and RQ-VAE quantization to learn semantic identifiers for items, and performs generative recommendation by autoregressively decoding item identifiers.
- • LETTER (Wang et al., 2024a) proposes a learnable tokenizer that extends RQ-VAE-based semantic identifiers by jointly incorporating hierarchical semantics, collaborative signals, and code assignment diversity for generative recommendation.
- • EAGER (Wang et al., 2024b) integrates semantic and collaborative information via a two-stream generative architecture with shared encoding and separate decoding for enhanced collaborative modeling.
- • UNGER (Xiao et al., 2025) integrates semantic and collaborative information into a unified item code by learning item representations via joint optimization of cross-modality alignment and next-item prediction in a sequential recommendation model.
- • ETEGRec (Liu et al., 2025) integrates item tokenization and generative recommendation into a unified end-to-end framework, jointly optimizing the tokenization and recommendation processes for improved performance.

- Table 7. ReSID’s branching factors on each dataset. Datasets: MI = Musical Instruments, VG = Video Games, IS = Industrial & Scientific, BP = Baby Products, ACS = Arts, Crafts & Sewing, SO = Sports & Outdoors, TG = Toys & Games, HH = Health & Household, BPC = Beauty & Personal Care, BK = Books.

MI VG IS BP ACS SO TG HH BPC BK (32,40,19) (32,64,13) (24,80,14) (32,64,18) (64,96,16) (128,128,11) (192,192,5) (50,512,8) (96,192,11) (256,256,8)

- Table 8. Sensitivity to branching factor at the first two SID levels on Musical Instruments. We vary (b1, b2) and report downstream Recall@10 (R@10). Each row fixes b1 and sweeps b2.

b1 \ b2 24 32 40 48 56 64 72 80

16 6.3068 6.3906 6.1548 6.2928 6.1897 6.3277 6.2928 6.2386 24 6.1985 6.3155 6.1950 6.3539 6.3609 6.2404 6.3260 6.1915 32 6.3469 6.3120 6.4465 6.3068 6.2509 6.2718 6.2945 6.2579 48 6.2806 6.2072 6.2229 6.1880 5.9941 6.0570 6.1897 6.2055

### E. Implementation Details

Our experiments follow the standard three-stage pipeline: representation learning (E-stage), SID construction via quantization (Q-stage), and SID-based generative modeling (G-stage). We summarize the hyperparameter settings below.

E-stage and sequential baselines. Unless otherwise specified, all sequential recommenders and the E-stage encoder in ReSID share the same model size and training configuration for controlled comparison. We use an embedding/hidden size of 128, 2 Transformer layers, 4 attention heads, and an FFN dimension of 512 with ReLU activation, together with dropout rate of 0.1. Feature embeddings have dimension 128 and are fused by sum-pooling when feature fields are used. We optimize with AdamW (learning rate 0.001, weight decay 1.0 × 10−5) using batch size 2048 and train for up to 500 epochs with early stopping patience 3, evaluating every epoch. When a sampled classification objective is used (e.g., for large vocabularies), we sample 128 negatives and use scaled cosine similarity. We use a cosine learning-rate scheduler.

Q-stage. For ReSID, the dataset-specific branching factors {bl} are reported in Table 7. For SID-based baselines (TIGER, LETTER, EAGER, UNGER, and ETEGRec), we follow the optimal quantization settings and other hyperparameters reported in their original papers and official implementations. For methods requiring text-based item embeddings (TIGER, LETTER, EAGER, and UNGER), we follow prior work (Rajput et al., 2023) and use the pretrained Sentence-T5-xxl (Ni et al., 2022) to obtain semantic item embeddings before discretization.

G-stage. All SID-based generative recommenders (including ReSID and prior SID baselines) use the same T5-style encoder-decoder architecture for training on SID sequences. We use 4 encoder layers and 4 decoder layers, hidden size 128, FFN dimension 512, and 4 attention heads per layer (key/value dimension 32), with dropout rate of 0.1. We use AdamW (learning rate 0.005, weight decay 1.0 × 10−5) with a cosine learning-rate scheduler, batch size 2048, and train for up to 500 epochs. During inference, we use beam search with beam size 50 at each decoding step.

### F. Full Results of Main Experiments

Table 4 reports the absolute performance of all compared methods on each Amazon-2023 subset under the experimental settings described in Section 5.1. These results complement the macro-averaged relative improvements reported in the main paper and provide detailed subset-wise comparisons for reproducibility and further analysis.

### G. Sensitivity to Branching Factors

We investigate how branching factors affect ReSID. In our hierarchical quantization, at each level l, each parent cluster is partitioned into bl balanced child clusters (Alg. 1), where bl is the branching factor (i.e., the number of children per parent) at that level. Varying {bl} controls the granularity and capacity of the discrete code space, trading off distortion under the discrete bottleneck (captured by H(z | C)) and sequential uncertainty (captured by l H(cl | C(<l))), as discussed in Section 3.3.

0.024

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | |ReSID|
| | | | |TIGER<br><br>SASRec|

0.022

0.020

NDCG@10

0.018

0.016

0.014

0.012

0.010

5.4 5.7 6.0 6.3 6.6

log10(P)

- Figure 4. Empirical scaling behavior on Baby Products. The x-axis shows log10(P), where P is the number of non-embedding model parameters, and the y-axis reports NDCG@10 on the test set. For each model size, we select the checkpoint with the best validation NDCG@10 and report its corresponding test NDCG@10. We compare ReSID with TIGER and SASRec under matched backbone parameter budgets.

Protocol. We conduct this study on Musical Instruments. Given the item scale in our experiments (20K–400K items), we use a three-level SID throughout and vary the branching factors at the first two levels (b1,b2) while keeping other settings fixed. The last level primarily serves to disambiguate items within each prefix (c1,c2); its effective branching factor depends on the population of each prefix and can be auto-computed once the prefix levels are fixed, so we do not treat b3 as a primary tuning knob. Following the GAOQ design, global alignment is applied to non-root levels where indices would otherwise be locally assigned under different parent prefixes; the root-level codes are already globally defined by clustering in the original embedding space and thus do not require global alignment. We keep the default anchor setting (gl = bl at aligned levels) and do not vary it separately.

Results. Table 8 shows a clear capacity-predictability trade-off consistent with Section 3.3. When (b1,b2) are too small, the code space is overly coarse and performance degrades due to insufficient capacity and higher quantization distortion. Increasing (b1,b2) improves performance and reaches its best range at moderate branching factors (e.g., (b1,b2) = (32,40) achieves the highest Recall@10 in our sweep), after which further increasing the branching factors yields diminishing returns and can slightly hurt performance as autoregressive decoding uncertainty increases. Overall, we observe the optimal performance when b1 × b2 is about 10 to 20 times smaller than the vocabulary size, which is also observed for other datasets.

### H. Empirical Scaling Trend

We investigate how downstream recommendation quality varies with model scale by changing the backbone parameter budget and evaluating ranking metrics. For each configuration, we periodically evaluate on the validation set, select the checkpoint that achieves the best validation NDCG@10, and report its corresponding test NDCG@10 in Fig. 4. To ensure a fair comparison across different architectures and tokenization schemes, we match models by the number of non-embedding (backbone) parameters.

As shown in Fig. 4, ReSID consistently achieves the best NDCG@10 across the explored parameter range and exhibits the most favorable scaling behavior, indicating that it leverages additional backbone capacity more effectively under the same data and training protocol. We also observe that performance does not improve monotonically at the largest scale: the final point slightly drops, which is likely due to overfitting and reduced generalization in this low-data regime. Finally, comparing the SID-based models (ReSID and TIGER) with the item-ID-based SASRec baseline under matched backbone budgets suggests that semantic IDs provide a more scalable and parameter-efficient modeling interface for generative recommendation in our setting.

### I. Representation Analysis of FAMAE

#### I.1. Semantic–Collaborative Alignment in Item Representations

To examine whether FAMAE captures both semantic and collaborative structures, we visualize item embeddings learned by FAMAE, BERT4Rec, and Sentence-T5 using t-SNE (Maaten & Hinton, 2008). Items are colored by (i) level-1 category labels to reflect semantic signals, and (ii) behavioral communities to reflect collaborative signals. Behavioral communities are identified using the Louvain algorithm (Blondel et al., 2008) on an item–item co-interaction graph, where edges connect items co-interacted by the same users.

###### Semantic Category Structure (Cate1) Behavioral Community Structure (i2i)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

FAMAE Item-ID-only Embeddings

FAMAE All-field Concat Embeddings

FAMAE Item-ID-only Embeddings

FAMAE All-field Concat Embeddings

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

BERT4Rec Item-ID-only Embeddings

BERT4Rec All-field Concat Embeddings

BERT4Rec Item-ID-only Embeddings

BERT4Rec All-field Concat Embeddings

[Figure 15]

[Figure 16]

Sentence-T5 Text Embeddings Sentence-T5 Text Embeddings

- Figure 5. T-SNE visualization of item embeddings learned by different methods. Left: Semantic Category Structure (Cate1)—items are colored by their category labels. Right: Behavioral Community Structure—items are colored by communities discovered via the Louvain algorithm on a weighted item–item co-occurrence graph constructed from user interaction histories.

As illustrated in Fig. 5, FAMAE exhibits a distinctive advantage over the compared methods. It is the only approach that produces well-structured clusters under both semantic and collaborative views, indicating that its representations simultaneously encode category-level semantics and user interaction-driven relational structure.

In comparison, embeddings learned by the pre-trained Sentence-T5-xxl model exhibit clear semantic clustering, but are inferior in reflecting behavioral communities (points with the same color are less clustered, e.g., brown and yellow communities scatter and overlap with others), as they are learned purely from textual information without collaborative supervision. Conversely, BERT4Rec embeddings effectively capture collaborative structures but show little semantic organization, resulting in poor category separability.

Moreover, FAMAE’s item-ID embeddings align closely with category-induced semantic structure, whereas BERT4Rec’s item-ID embeddings are largely unstructured. This suggests that FAMAE preserves task-relevant information from structured features, yielding representations that are predictive and sufficient for generative recommendation.

- I.2. Structured and Field-Aligned Embedding Space

- Figure 6 visualizes the joint embedding distribution of item-ID and multi-level category features using UMAP (McInnes et al., 2018).

- As shown in the figure, item embeddings learned by FAMAE exhibit a clearly structured and multi-peak distribution, forming multiple localized high-density regions rather than a single unimodal cluster. This observation suggests that FAMAE learns expressive and discriminative item representations, leading to a more structured embedding geometry. More importantly, FAMAE yields strong alignment between item-ID embeddings and category embeddings. Category

[Figure 17]

[Figure 18]

- Cate1_Id

- Cate2_Id

- Cate3_Id

- Cate1_Id

- Cate2_Id

- Cate3_Id

FAMAE Multi-Field Embedding Distribution BERT4Rec Multi-Field Embedding Distribution

- Figure 6. UMAP visualization of the joint embedding distribution of item-ID and category fields. Item-ID embeddings are rendered as a density map, while category embeddings are displayed as scatter points with sizes proportional to the number of associated items.

Table 9. Overlap ratio between target item codes and historical item codes. A historical item is counted as matched if it shares the same code with the target item at the same SID layer. Higher overlap indicates stronger task-consistent alignment in the discrete SID space.

Method Code1 Code2 FAMAE + GAOQ 0.0724 0.0260 FAMAE + Hierarchical K-Means 0.0724 0.0221 Text Embedding + Hierarchical K-Means 0.0660 0.0188

embeddings are distributed around dense item regions, and item density peaks are consistently accompanied by corresponding concentrations of category embeddings. This correspondence indicates that item identifiers and categorical fields are embedded in a shared and coherent representation space, where category information actively shapes item-level geometry rather than acting as auxiliary side information.

In contrast, embeddings learned by BERT4Rec exhibit a largely unimodal and smooth item distribution with a collapsed internal structure, while category embeddings are heavily concentrated in peripheral regions of the space.

By jointly supervising multiple fields at the final interaction position, FAMAE results in a structured, field-aligned embedding space that provides a favorable foundation for downstream SID construction.

J. Downstream Task Aligned Design of the FAMAE Objective

- J.1. From the Attention Perspective

Figure 7 compares attention patterns (averaged over validation samples) induced by different masking strategies and illustrates how FAMAE’s last-position, multi-field masking yields task-aligned temporal attention.

Under FAMAE, attention from the target position (lower-right corner) exhibits a clear recency bias, with monotonically increasing weights toward more recent interactions. This reflects the encoder’s focus on aggregating collaborative signals that are most informative for predicting the next item under a sequential recommendation objective.

In contrast, BERT-style random masking optimizes a position-agnostic reconstruction objective, resulting in diffuse and less structured attention patterns that are not explicitly aligned with next-item prediction.

Overall, these observations show that FAMAE shapes contextual aggregation in a recommendation-native manner, aligning attention structure with the information requirements of downstream generative recommendation rather than generic semantic reconstruction.

- J.2. From the Sequential Decoding Uncertainty Perspective

- Table 9 evaluates how effectively task-relevant information encoded in continuous item representations is preserved after discretization into SIDs. Specifically, we measure the overlap between SID tokens of target items and those of their historical

[Figure 19]

Attention Score: Low High

FAMAE Last-Position Masked Attention

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Head0 Head1 Head2 Head3

BERT-style Multi-Position Masked Attention

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Head0 Head1 Head2 Head3

- Figure 7. Attention score visualizations under different masking-position strategies. The top row uses last-position (target item) masking as used in FAMAE, while the bottom row uses BERT-style masking that randomly masks multiple positions.

- Table 10. Mean pairwise cosine similarity of centered embedding directions for items sharing the same code at second level. Lower values indicate greater directional diversity within the same code.

Method Musical Instruments Video Games Baby Products Beauty and Personal Care

GAOQ 0.0463 0.0567 0.0570 0.0524 Hierarchical K-Means 0.0172 0.0154 0.0086 0.0053

interactions, which serves as a proxy for prefix-consistent relational structure in the discrete space.

FAMAE-based representations consistently yield higher SID overlap ratios than text-based embeddings, indicating that FAMAE encodes interaction-aligned collaborative structures that survive the discrete bottleneck. From an informationtheoretic perspective, this suggests lower reconstruction ambiguity H(z | cl) and reduced reliance on long prefixes to disambiguate item semantics.

Furthermore, introducing global index alignment in GAOQ—particularly at deeper SID layers—further increases overlap. This demonstrates that GAOQ reduces prefix-dependent ambiguity by enforcing prefix-invariant code semantics, thereby preserving task-aligned structure learned in the E-stage. As a result, the resulting SIDs better reflect user interaction patterns and exhibit lower intrinsic uncertainty for autoregressive decoding.

- K. Understanding GAOQ Algorithm and Advantages

Concretely, at each quantization level, GAOQ first partitions a parent cluster into child clusters using balanced K-Means. It then computes residual child vectors by centering each child centroid with respect to its parent centroid. These residuals are matched to a globally shared set of anchor embeddings. The anchors are constructed to be approximately orthonormal by maximizing inter-anchor cosine separation, providing a uniform and non-overlapping reference basis. A one-to-one assignment between child clusters and anchors is obtained by maximizing cosine similarity under an injective constraint, solved via the Hungarian algorithm (Algorithm 1).

- Figure 8 provides an intuitive comparison (a toy showcase) between GAOQ and prior hierarchical quantization schemes.

Local Randomly Assigned Index

V

V

Items

B

| |
|---|

Conflict

Residual

- 1
- 2 3

|B|
|---|

| |
|---|

V

Hierarchical Clustering

B

Title: Vases

- 1
- 2

- 1
- 2

- 1
- 2

B

|V|
|---|

Cate: Decor

|V|
|---|

|T|
|---|

T

|2|3|
|---|---|
|𝐶1| |

| |3|
|---|---|
|𝐶1| |

| |3|
|---|---|
|𝐶2| |

- Bought: User1

V

B

S

Raw Emb

Trans Emb

GAOQ

Assign Centering Index

Title: Tableware Cate: Household

- Bought: User2

S

|S|
|---|

S

- 1
- 2 3

- 1
- 2 3

| |
|---|

T

| |
|---|

| |
|---|

S

T

Title: Balloons Cate: Decor Bought: User2

|B|
|---|

|𝐶2|
|---|

Hierarchical K-Means RQ-VAE

Anchor 1

V

V

V

Collaborative Signal Preserved

- 1
- 2 3

|V|
|---|

Title: Snacks Cate: Food Bought: User2

|S|
|---|

Hierarchical Clustering

| |
|---|

B

| |
|---|

B

B

- 1
- 2

- 1
- 2 3

| |
|---|

|S|
|---|

|2|3|
|---|---|
|𝐶1| |

| |
|---|

|T|
|---|

- 1
- 2 3

- 1
- 2 3

S

| |
|---|

T

S

S

|B|
|---|

|T|
|---|

T

T

T

|𝐶2|
|---|

Global Alignment

- Figure 8. Comparison of SID construction strategies. Hierarchical K-Means assigns child indices locally and arbitrarily under each parent, with no explicit correspondence across different parents, resulting in non-coherent SIDs. RQ-VAE performs residual vector quantization and assigns codes globally and independently across stages, causing individual codes to lack consistent and meaningful semantic interpretation. GAOQ introduces a global alignment mechanism on top of Hierarchical K-Means, ensuring that codes have consistent meaning across prefixes while preserving collaborative signals.

Consider four items: Vases (purchased by User1) and Balloons, Snacks, Tableware (co-purchased by User2). In the embedding space, Vases and Snacks are far apart due to distinct semantic content, while the latter three items exhibit strong collaborative proximity.

Under standard Hierarchical K-Means with local indexing, child codes are assigned independently within each parent. As a result, Vases and Snacks may share the same second-level code (e.g., “1”), while items co-purchased by User2 receive different codes. This breaks prefix-invariant semantics and causes collaborative structure captured in the embeddings to be discarded during discretization.

RQ-VAE applies residual transformations to the embedding space but assigns codes at each level independently across levels. As a result, codes across different levels are weakly correlated, and higher-level codes do not enforce consistent semantic or collaborative structure across prefixes.

In contrast, GAOQ enforces global alignment across all parent clusters. In the same example, GAOQ assigns a shared code (e.g., “3”) to all items purchased by User2 and a different code (e.g., “1”) to Vases, preserving collaborative relationships in the resulting SIDs and reducing prefix-dependent ambiguity.

#### K.1. Empirical Evidence on Reduced Indexing Ambiguity of GAOQ

In addition, Table 10 provides evidence of reduced semantic ambiguity among items sharing the same code in GAOQ. Specifically, the average inter-item cosine similarity within each second-level code produced by Hierarchical K-Means is 3–10× lower than that of GAOQ. This indicates that GAOQ yields more directionally coherent code groups, reflecting lower reconstruction ambiguity H(z | cl) and more prefix-invariant code semantics as our analysis.

### L. Computational Complexity Analysis

To provide a quantitative understanding of the computational cost of our framework, we analyze the FLOPs of each major stage in the pipeline: FAMAE for representation learning, GAOQ for quantization, and the downstream T5-based generative model. We report dominant FLOPs in Big-O form throughout this section, focusing on the leading multiply–accumulate operations and omitting constant factors and lower-order components (e.g., masking, normalization, and loss computation).

#### L.1. FLOPs of FAMAE.

Let Te denote the input sequence length of FAMAE, J the number of structured features per item, de the hidden size of the FAMAE Transformer encoder, and Le the number of encoder layers. We embed each feature into Rd

e.

Algorithm 1 GAOQ at Level l (For a Given Parent at Level l − 1)

Require: Set of item representations Z = {zi}ni=1p that belong to the current parent node; centroid µ (the current parent embedding);

branching factor bl; number of anchors gl (gl ≥ bl)

Ensure: Level-l code assignment M : Z → {1, . . . , gl} # mapping each item in Z to a globally aligned anchor index

- 1: # Partition (Balanced Clustering)
- 2: ({Zj}bjl=1, {µj}bjl=1) ← BALANCED-KMEANS({zi ∈ Z}, bl) # {Zj}bjl=1 is separated embedding sets of children clusters, {µj}bjl=1 is the centroid set of children clusters
- 3: # Residualization (Centering)
- 4: for j = 1, . . . , bl do
- 5: µ¯j ← µj − µ
- 6: end for
- 7: # Anchor Construction
- 8: A = {ak}gkl=1 ← ORTHO(gl) # Approximately orthonormal anchor generation function
- 9: # Global Alignment (Matching)
- 10: Wbl×gl ← cos(¯µj, ak) for all (j, k) pairs
- 11: W ← HUNGARIAN(W) # Injective assignment from children centers to anchors
- 12: # Code Assignment
- 13: for j = 1, . . . , bl do
- 14: for all item i ∈ Zj do
- 15: M(i) ← W(j) (∈ {1, . . . , gl}) # anchor index lookup for children cluster j
- 16: end for
- 17: end for RETURN M

Feature fusion. At each position, sum-pooling J field embeddings of dimension de costs O(Jde), and adding positional encoding costs O(de), yielding

O(TeJde).

Transformer encoder. In each layer, the dominant FLOPs come from multi-head self-attention (MHSA) and the feedforward network (FFN). For MHSA, the Q/K/V and output projections cost O(Ted2e), and the attention score computation (QK⊤) and weighted sum (AV ) cost O(Te2de). For FFN with dff = rde (constant r), the two linear layers cost O(Tededff) = O(Ted2e). Therefore, the dominant FLOPs per encoder layer are

O(Te2de + Ted2e), and the total FLOPs of an Le-layer encoder are

##### O(Le(Te2de + Ted2e)).

Overall. Combining the above, the dominant FLOPs of FAMAE satisfy

FLOPsFAMAE = O(TeJde + Le(Te2de + Ted2e)).

#### L.2. FLOPs of GAOQ.

Let N be the number of items, dq the representation dimension fed into GAOQ, and Lq the number of quantization levels.

- At level l (l is indexed from 1), GAOQ uses branching factor bl and Il iterations for balanced K-Means. For l ≥ 2, GAOQ

additionally uses gl global anchors for alignment. Let Pl = li=1 bi be the number of nodes at level l, and let np be the number of items in a node p with P

p=1 np = N.

l

Partition (Balanced Clustering). For a given parent node p, the dominant cost of balanced K-Means is the assignment step that computes distances between np points and bl centroids, costing O(npbldq) per iteration. Thus, the clustering cost per parent is O(Ilnpbldq), and aggregating over all parent nodes of the current level l yields

O(IlNbldq).

Residualization (Centering). For each parent node, we center child representations (sub-cluster centroids) µ¯j = µj − µ for j = 1,...,bl, which incurs O(bldq) FLOPs. Since this operation is performed under all Pl−1 parent nodes at level l − 1, the per-level cost is

O(Pl−1bldq).

Anchor Construction. GAOQ constructs gl approximately orthonormal anchors once per level via QR decomposition on a dq × gl random matrix, costing

O(dqgl2).

Global Alignment (Matching). For a parent node p, forming the cosine-similarity matrix W ∈ Rb

l×gl costs O(blgldq),

and solving the injective assignment via Hungarian costs O(b3l ). Aggregating over Pl−1 parents, the matching cost per level is

O Pl−1(blgldq + b3l ) .

Code Assignment. Assigning the matched anchor index to each item is an index operation and is negligible under our dominant-FLOPs accounting.

Per-level and Overall. For l ≥ 2, combining the above stages, the dominant FLOPs at level l satisfy

O IlNbldq + Pl−1bldq + dqgl2 + Pl−1(blgldq + b3l ) .

Level 1 does not use anchors or matching, and only incurs the clustering cost O(I1Nb1dq). Therefore, the dominant FLOPs of GAOQ satisfy

FLOPsGAOQ = O

Lq

Lq

IlNbldq +

l=1

l=2

Pl−1bldq + dqgl2 + Pl−1(blgldq + b3l ) .

#### L.3. FLOPs of the Downstream T5 Model.

Let Tgenc and Tgdec denote the encoder input length and decoder output length of the downstream T5 model, respectively. We use a shared hidden size dg for both encoder and decoder, and set the FFN intermediate dimension as dff,g = rgdg with a constant expansion ratio rg. Let Lencg and Ldecg be the numbers of encoder and decoder layers.

Transformer encoder. As in Section L.1, the dominant FLOPs of a Transformer encoder layer are contributed by multi-head self-attention (MHSA) and the feed-forward network (FFN). For MHSA, the Q/K/V and output projections cost O(Tgencd2g), and the attention score computation (QK⊤) and weighted sum (AV ) cost O((Tgenc)2dg). For FFN, the two linear layers cost O(Tgencdgdff,g) = O(Tgencd2g). Therefore, the total FLOPs of the Lencg -layer encoder are

O Lencg (Tgenc)2dg + Tgencd2g .

Transformer decoder. Each decoder layer contains (i) causal self-attention, (ii) encoder–decoder cross-attention, and (iii) an FFN. Causal self-attention has the same dominant FLOPs form as encoder self-attention, yielding O((Tgdec)2dg + Tgdecd2g) per layer. For cross-attention, projections cost O(Tgdecd2g + Tgencd2g), and the attention score computation and weighted sum cost O(TgdecTgencdg). The FFN costs O(Tgdecdgdff,g) = O(Tgdecd2g). Combining these terms, the total FLOPs of the Ldecg -layer decoder are

O Ldecg (Tgdec)2dg + TgdecTgencdg + (Tgdec + Tgenc)d2g .

Overall. Combining encoder and decoder, the dominant FLOPs of the downstream T5 model satisfy

FLOPsT5 = O Lencg (Tgenc)2dg + Tgencd2g + Ldecg (Tgdec)2dg + TgdecTgencdg + (Tgdec + Tgenc)d2g .

###### L.4. Overall FLOPs. The dominant FLOPs of ReSID are the sum of the costs of its three stages:

FLOPsReSID = FLOPsFAMAE + FLOPsGAOQ + FLOPsT5.

