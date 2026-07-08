# arXiv:2407.20595v5[cs.DL]25May2026

## HALvest-Contrastive: Retrieval-Like Authorship Attribution with Patch-level Late Interaction

Francis Kulumba Inria Paris Sorbonne Université francis.kulumba@inria.fr

Wissam Antoun Inria Paris Sorbonne Université

Guillaume Vimont IRIF

Florian Cafiero LRE, EPITA Ecole nationale des chartes – PSL Abstract

Laurent Romary Inria Paris

margin (Wegmann et al., 2022; Kantharuban et al., 2026). Information retrieval, a field that also contrasts documents against one another, has moved in a slightly different direction. In fact, dense retrieval has replaced triplet loss with InfoNCE (Oord et al.,

Authorship attribution asks whether two pieces of text share a writer, but topical confound makes the task deceptively easy: two authors covering the same topic may look more alike than one author covering two topics. Scholarly prose offers a natural remedy, academic writers produce multiple papers on related but distinct topics while maintaining consistent stylistic habits. We introduce HALvest, a 17-billion-token multilingual corpus of openaccess academic papers, and its English contrastive derivative HALvest-Contrastive, where same-author passages are drawn from distinct papers within a disciplinary field to minimize topical overlap. We validate our benchmark by showing that a strong lexical baseline collapses once topical shortcuts are removed. On this same benchmark, we revisit how authorship is scored. Standard systems compress each document into a single vector. We instead keep a sequence of vectors and compare them with late interaction, then propose patch-level late interaction, which groups neighboring tokens into patches before matching. Matching at the sequence level greatly improves performance over the single-vector baseline, but the optimal interaction granularity is subtle.

- 2019; Gao et al., 2021), a contrastive objective that leverages many in-batch negatives instead of one hard negative, and single-vector pooling has given way to late interaction (LI) (Khattab and Zaharia,
- 2020; Santhanam et al., 2022), where documents are represented by sets of token vectors and the score is the sum of maximum matches between said tokens (Figure 1). This paper asks what happens when the retrieval playbook is applied to AA, and where the two tasks diverge.

A primary obstacle is topical confounding: if the AA field moved toward contrastive approaches, it was, to a large extent, to be able to minimize vocabulary overlap within each triplet (Wegmann and Nguyen, 2021). The intuition is that decreasing the discriminative signal from individual tokens may encourage models to learn subtler distributional cues like stop-word frequency, sequence length, or overall tone. Building on that same line of work, we propose a solution based on the inherent nature of academic papers, where authors write on related but distinct scholarly topics while maintaining consistent stylistic patterns. We introduce HALvest, a 17-billion-token corpus harvested from 778k open-access and multilingual academic papers, and its contrastive derivation HALvest-Contrastive, an English-only benchmark with controlled topic variation. We built HALvest-Contrastive to maximize topic decoupling: an anchor and positive are drawn from different papers written by the same authorset, and the negative is mined from within the same disciplinary field, so that a model cannot entirely succeed by learning topic vocabulary.

### 1 Introduction

Authorship Attribution (AA) asks whether two pieces of text were written by the same author. The field has a long statistical tradition that treats style as a document-level property summarized by a frequency vector (Burrows, 2002; Schler et al., 2006; Treeratpituk and Giles, 2009). Neural approaches have largely inherited this view. A passage is embedded by a transformer encoder (Vaswani et al., 2017; Devlin et al., 2019), the token representations are mean-pooled into a single vector, and training minimizes a triplet loss: a three-way comparison between an anchor, a same-author positive, and a different-author negative, with a fixed

Under this design, we study how retrieval techniques transfer to authorship attribution, from the training objective to the interaction mechanism

s

s

s

MaxSim MaxSim

MaxSim MaxSim MaxSim MaxSim

(a) Single-vector Scoring Token vectors are averaged into single vectors to compare.

(b) Late Interaction Scoring Token vectors are directly compared as-is. Per querytoken maxima are summed.

(c) Patch-level Late Interaction Scoring Token vectors are locally compressed into patches to compare. Per-query-patch maxima are summed.

Figure 1: Three scoring modes for authorship attribution. The green and blue tiles represent token vectors from an anchor and a positive document respectively. (a) Single-vector pooling averages token embeddings into one vector per document and scores pairs by cosine similarity. (b) Late interaction (Khattab and Zaharia, 2020) keeps the full sequence of token vectors and scores by MaxSim. (c) Patch-level late interaction (PLI) groups token vectors into contiguous patches, pools each patch into a single vector, and applies MaxSim at the patch level. We train and evaluate all three modes under a matched contrastive objective. The granularity comparison is the main object of study in this paper.

(late interaction and its patch-level variant). The comparison is controlled: all models share the same encoder, the same data, and the same loss; only the pooling and scoring granularity differ. Our contributions are as follows.

- 1. We release HALvest, a 17-billion-token multilingual corpus harvested from open-access academic papers, and HALvest-Contrastive, an English scholarly AA benchmark with verified multi-author structure, controlled topic decoupling, and an evaluation structure by span length.
- 2. We empirically validate topic decoupling by showing that a strong lexical baseline succeeds on topic-rich but fails on topicdecoupled data, with a gap persisting across span lengths.
- 3. We present a granularity study that pits mean pooling, token-level late interaction, and fixed-policy PLI variants against one another on matched data under a matched training objective, across five span-length subsets of HALvest-Contrastive plus PAN19 zero-shot.
- 4. We observe that the empirically best fixedn patch size across our subsets and PAN19 is

√

well approximated by n⋆ ≈ 0.18 ·

S, where S is the token count. A descriptive fit that provides a practical starting point for practitioners.

The code 1, and the corpus itself 2 are released

- 1https://github.com/Madjakul/DeepStylometry
- 2https://huggingface.co/datasets/almanach/

under an open license.

### 2 Related Work

The closest precedents to our setup come from three separate literatures: stylometry, scholarly text corpora, and multi-vector retrieval.

#### 2.1 Stylometry and authorship attribution.

Stylometric AA has traditionally relied on frequency-based representations of writing style, leveraging features such as function words (Mosteller and Wallace, 1963; Burrows, 2002; Treeratpituk and Giles, 2009), part-of-speech n-grams (Argamon-Engelson et al., 1998), most frequent character n-grams (Kešelj et al., 2003), or combinations of those (Cafiero and Camps, 2019; Juola, 2015). Machine-learning treatments formalised the task in closed, verification, and open-set variants (Koppel et al., 2009; Seroussi et al., 2014). A recurring concern is topical confounding: Sapkota et al. (2015) showed that classifiers trained and tested on the same topic substantially overestimate generalization, and proposed cross-topic evaluation as a stricter protocol.

Large-scale neural AA systems followed: contrastive learning has emerged as a powerful paradigm for stylometry (Ai et al., 2022; Wegmann et al., 2022; Huertas-Tato et al., 2024). A prevailing strategy in this area has been to decorrelate an au-

halvest-contrastive

thor’s style from literal text content (Altakrori et al., 2021). This includes developing edit-invariant loss functions to capture signals beyond surface-level text (Liu et al., 2022), or creating positive pairs with high semantic overlap to force models to learn distributional cues (Rivera-Soto et al., 2021; Wegmann and Nguyen, 2021; Alshomary et al., 2025). Most recently, Kantharuban et al. (2026) studied idiolect at scale.

Our work differs in two ways. First, it targets scholarly prose, a low-entropy register with strong domain structure, rather than internet or literary text. Second, it replaces single-vector pooling and triplet loss with late interaction and InfoNCE (Oord et al., 2019), and asks whether the induced performance changes transfer (Figure 1). In fact, the InfoNCE objective has become a standard discriminative loss for representation learning: it learns embeddings by contrasting a positive pair against a pool of in-batch negatives. Chen et al. (2020) and Khosla et al. (2021) showed that large negative pools and strong augmentations suffice for self-supervised image representations. Notably, for our setting, they report that in-batch false negatives, pairs labeled as negative that in fact share a latent class, degrade performance only weakly.

#### 2.2 Scholarly text corpora.

Several corpora have been built around scientific writing. Prominent examples include domainspecific documents from ArXiv, DBLP, and PubMed (Sen et al., 2008; Do˘gan et al., 2014; Wahle et al., 2022). Other initiatives provide full-text access, such as the ACL Anthology for computational linguistics (Bird et al., 2008), or S2ORC (Lo et al., 2020), notable for its scale and multi-domain coverage.

HALvest complements these by drawing from a single open-access repository that spans multiple high-level domains, and by supplying the authorset labels as they appear in the metadata used by real retrieval systems.

#### 2.3 Late interaction and multi-vector retrieval.

ColBERT (Khattab and Zaharia, 2020) introduced late interaction: documents are represented by a set of token embeddings and scored against a query via MaxSim, the sum over query tokens of the maximum cosine similarity to any document token. ColBERTv2 (Santhanam et al., 2022) added residual compression for storage. TRIAL (Kang

et al., 2025) adds bigram-level token relation scores and per-query importance weights to MaxSim. These modifications target semantic retrieval where phrase coherence and content-word emphasis improve relevance estimation. Token pruning approaches (Zong and Piwowarski, 2025) learn to remove tokens before scoring. ColBERTer (Hofstätter et al., 2022) compresses tokens into coarser units, while ConstBERT (MacAvaney et al., 2025) pools token embeddings into a fixed-size set of learned vectors, functioning as a learned-patch reranker.

These methods are conceptually the closest precedents to our patch-level view, but they target either semantic retrieval or generative modeling rather than AA.

### 3 HALvest and HALvest-Contrastive

A benchmark for stylometry has to answer two questions: where does the text come from, and how are same-author and different-author pairs constructed so that the resulting signal is style rather than topic. This section answers both.

#### 3.1 Why scholarly prose?

Scholarly writing is an under-exploited testbed for stylometric research. In fact, academic prose is lowentropy: its vocabulary is rather constrained, its syntactic templates are strongly standardized, and its structure is recurrent across papers within a field. This compresses the range of surface variation and makes the stylistic residual easier to isolate from topical noise.

#### 3.2 Source and processing pipeline.

HAL is an open-access repository of French and international scholarly output. We extract full-text PDFs for all papers with permissive licenses, process them with GROBID (GROBID Repository) to recover structured XML, which we serialize to plain text. All in all, HALvest covers 778k documents in 56 languages across 16 disciplinary domains.

From HALvest to HALvest-Contrastive. For the contrastive dataset we apply an additional sequence-level filter that restricts to English spans; this yields HALvest-Contrastive, the English-only triplet dataset on which all models in this paper are trained. We further filter to the 13 domains with sufficient document counts for reliable hard-negative

mining (Appendix B), and remove spans with excessive symbols, abnormal layout, code fragments or HTML markup, high uppercase ratios, or anomalous sentence-length variance. Mining proceeds in three configurations at first, before scaling the best one to millions of rows.

The unrestricted configuration samples five random spans per query document. Positives are drawn from all documents sharing the exact same author-set; if no other document exists for that set, positive spans are sampled from the same document as the query. Hard negatives are drawn from documents that share no authors with the query but have the same HAL domain label. This configuration permits high topical overlap between query and positive, and serves as a topical control.

The base (restricted) configuration enforces strict topic decoupling: positives are drawn exclusively from different documents by the same author-set, never from the query document itself. The base configuration is the one we scaled and released under the name HALvest-Contrastive. The total number of triplets is described in Table 1.

#### # Sentences Train Valid Test

2 1.88M 19.1k 19.1k 4 1.4M 14.3k 14.3k 6 1.11M 11.3k 11.3k 8 892k 9.1k 9.1k

- Table 1: Number of triplets in the HALvest-Contrastive dataset.

A third inverse cloze task (ICT) configuration (Lee et al., 2019) treats a random span from a document as the anchor and the surrounding context as the positive, with a non-overlapping passage from the same document as the negative. ICT supplies a style-aware retrieval signal that is neither purely lexical nor fully topic-decoupled, and serves as an intermediate baseline in §5.1.

Triplet construction. Within each configuration, a triplet consists of three spans each containing k contiguous sentences. The span length k parametrizes task difficulty: shorter spans leave less textual evidence, so AA accuracy rises monotonically with k. The dataset supports k ∈ {2,4,6,8,10}; we report k = 4 as the primary split, with full per-k results in Appendix D. Training uses base configuration triplets throughout; evaluation reports both base and unrestricted where

relevant to the argument.

Author-set as the label. The label of a triplet is its set of co-authors, not a single “primary” writer and author. We use author-sets because they match the metadata exposed by scholarly repositories and avoid imposing an arbitrary single-author heuristic, thus being closer to real-world settings. This also covers the common case, since about 51% of our triplets are already single-author, while preserving collaborative papers without forcing a noisy reduction to one writer. Forensic linguistics supports this framing. Dauber et al. (2019) demonstrate that individual attribution within collaborative documents achieves subpar accuracy even with bestavailable methods. In the general case we cannot know which listed co-authors contributed as writers versus providing ideas or feedback.

Cross-domain evaluation via PAN19. We use PAN19 (Kestemont et al., 2019), which labels documents at the individual-author level and whose texts are drawn from fan-fiction, as a zero-shot cross-domain evaluation. Both corpora are English, but the registers are different: academic prose is heavily normalized by journal style guides, while fan-fiction exhibits broader variation in sentence length, punctuation, and paragraph structure. A model that generalizes across both the label granularity gap (author-set to individual author) and the register gap (academic to fan-fiction) is, by elimination, not a topic classifier in disguise.

### 4 Contrastive Modelling

We separate two design choices that the AA literature has historically bundled together. In this section, we make the case for InfoNCE over triplet loss as the training objective, before laying out the three families of pooling and interaction we compare in our experiments.

We adopt standard retrieval terminology throughout. A triplet is an ordered tuple of three passages (a,p,n), where a (the anchor) and p (the positive) share authorship, and n (the negative) does not. A contrastive loss pulls a and p close in embedding space while pushing a away from n. We evaluate models with the standard ranking metrics Recall@k (the fraction of anchors whose correct positive is in the top-k retrieved candidates) and nDCG@k (a graded, position-aware variant). For zero-shot cross-domain evaluation, we use the triplet accuracy (the fraction of triplets for which

s(a,p) > s(a,n)).

#### 4.1 A simple look on InfoNCE

Given an anchor a, a positive p drawn from the same author-set, and a pool of in-batch negatives N drawn from other author-sets, we train to minimize the InfoNCE loss

exp(s(a, p)/τ) exp(s(a, p)/τ) + n∈N exp(s(a, n)/τ)

L = − log

(1)

where s(·,·) is the pooling-specific similarity score (Figure 1) cosine for single-vector pooling, MaxSim for late interaction and PLI, and τ is a temperature parameter, set to 0.5 throughout. We use full-gather across GPUs, so that the negative pool at each anchor grows with the total effective batch size. We used 4 GPUs with a batch-size of 32 per device, leading to an effective number of 256 negatives per anchor. We further scale the batch size to 64, as we show in §5, this scaling is not incidental: larger negative pools yield measurable gains, consistent with the retrieval literature (Karpukhin et al., 2020; Chen et al., 2020).

InfoNCE gradient naturally concentrates on hard negatives, which aligns with suppressing topical shortcuts in authorship attribution.. Writing sn = s(a,n) for a generic negative, the partial derivative of Equation 1 with respect to sn is

∂LInfoNCE ∂sn ∝

exp(sn/τ) exp(sp/τ) + n′∈N exp(sn′/τ)

(2)

which is exactly the softmax weight of n in the contrastive denominator. The gradient thus concentrates on highly similar negatives according to the model’s current state, and this, without an explicit mining step. Early in training, the hardest negatives for a pre-trained encoder are topically the ones provided in the triplet, so InfoNCE allocates gradient to decorrelating style from topic, which is exactly the direction AA needs. A full derivation of Equation 2 is given in Appendix A.

Multi-positive evaluation under single-positive training. Our training mines one positive per anchor, but at evaluation time some anchors have multiple positives, because author-sets recur across documents. Hence, in-batch “false negatives” (other valid positives that happen to appear in the batch and are labeled negative) can be a concern when it comes to gradient stability. We treat these cases

as benign, as in practice, occasional false negatives act like mild hard-negative noise rather than a failure mode. Chen et al. (2020) report empirically that in-batch false negatives degrade performance only weakly in their self-supervised setting, and subsequent work has corroborated this. Complementarily, (Khosla et al., 2021) explicitly handles the multi-positive case and finds that aggregating multiple positives improves, rather than hurts, representation quality in supervised contrastive learning.

#### 4.2 Pooling and interaction

We compare three families of similarity functions, each determining how the representations produced by the encoder are combined into a document score.

Mean pooling. The document is represented by the mean of its token embeddings, and s(a,p) is cosine similarity between the two mean vectors. This is the standard AA baseline (Rivera-Soto et al., 2021; Huertas-Tato et al., 2024; Wegmann et al., 2022) and the default for general-purpose sentence encoders. We additionally trained a layerwise mean-pooling variant (with and without centering) as an alternative baseline (Kantharuban et al., 2026); validation-set performance only slightly outperformed mean pooling and we discarded it before test-set evaluation.

Late interaction. The document is represented by its full sequence of token embeddings, and s(a,p) = i maxj cos(hai ,hpj) is the MaxSim score of Khattab and Zaharia (2020), with punctuation and padding positions masked out. This representation is used at both train and test time; the scoring function is identical to the training objective.

Patch-level late interaction (PLI). The token sequence is first grouped into contiguous patches, each group is mean-pooled into a single patch vector, and MaxSim is computed between the two resulting patch sequences. PLI reduces the perdocument vector count by the average patch length and makes the granularity of interaction an explicit design choice; as a result, both the memory footprint and the MaxSim scoring cost drop roughly in proportion to the patch size. We study two fixedpolicy families.

N-gram PLI. Patches are fixed-length, nonoverlapping windows of n tokens. We evaluate n ∈ {2,3,4,5}. This policy is parameter-free but

ignores whitespace and word boundaries, so a single linguistic word can be split across patches.

Whole-word PLI. Patch boundaries align with word-starts. Each patch corresponds to one linguistic word, so the representation is invariant to how the word is broken into sub-word units.

What about learned patches? A natural further step is to make the patch boundaries themselves trainable: a small head emits a per-token cut probability, and the boundaries are sampled with a Gumbel-Softmax straight-through estimator (Jang et al., 2017; Maddison et al., 2017) so that the patching policy can be trained jointly with the encoder under InfoNCE. We implemented this and ran it under two single-term regularizers. Both failed in characterizable ways and the resulting model never reached the fixed-n baselines (Appendix F).

### 5 Experiments

We evaluate each trained model with two complementary views. The first is retrieval on HALvestContrastive, reporting Recall@20, Recall@100, nDCG@20, and nDCG@100, with the scorer matching the training objective: cosine for mean pooling, token-level MaxSim for late interaction, and patch-level MaxSim for PLI. The second is triplet accuracy on PAN19 zero-shot.

#### 5.1 Topic decoupling

The base configuration of HALvest-Contrastive is designed to prevent models from learning topical shortcuts. Before discussing any trained model, we verify empirically that the design works as intended. Two pieces of evidence support the claim.

We measure the Jaccard overlap between the word sets of triplet components on a sample of 10,000 triplets. Figure 2 reports the results.

A purely lexical retriever gives us a direct test. If the base configuration has successfully decoupled topic, BM25 (Robertson et al., 1994) should succeed on the unrestricted data (where the positive can share words with the query) and fail on the base data. Table 2 reports BM25 accuracy on both configurations across span lengths, alongside the ICT baseline and a RoBERTa (Liu et al., 2019) contrastive model trained on each configuration. Recall that we haven’t scaled any configuration yet and only intend to justify our design choice for HALvest-Contrastive. On four-sentence spans, BM25 achieves 79.26% accuracy on unrestricted

0.20

Jaccardsimilarity

0.15

0.101

0.10

0.076

0.05

0.00

Query Positive Query Negative

Figure 2: Mean Jaccard vocabulary overlap between the components of HALvest-Contrastive triplets. Under the base configuration, query-positive overlap is already low in absolute terms, confirming that topic is decoupled between same-author passages. Query-negative overlap is lower still, but not too dissimilar.

but only 68.36% on base, a 10.9-point drop. The gap narrows at longer spans, but persists throughout. The ICT model consistently outperforms BM25, especially on the base set. This indicates that its retrieval objective can learn more abstract representations. The RoBERTa contrastive model trained with triplets where the anchor and positive originate from different documents achieves top performance across all conditions.

We further scaled the base data afterward and made it HALvest-Contrastive.

#### 5.2 Effect of span length

Accuracy rises monotonically with the span length k for every model in Table 2but with different slopes. BM25 improves by approximately 18 percentage points on the base configuration from k = 2 to k = 8, or a relative gain of roughly 30%. A lexical model is almost entirely coupled to the availability of more keywords, so doubling and then quadrupling the span length produces nearly-linear improvement. The neural contrastive models improve much more modestly: RoBERTaBaseData gains 8.5 points (10% relative) from k = 2 to k = 8, and the improvement is concentrated at the short end. The curve flattens by k = 6.

Neural models seem to capture substantial authorial signal at short spans. Syntactic templates, preferred transition words, clause structures or function-word patterns are already present in two sentences. BM25, by contrast, extracts a signal that scales with how many content words happen to co-occur, a quantity that continues to grow with

###### k = 2 k = 4 k = 6 k = 8

Model Unrestr. Base Unrestr. Base Unrestr. Base Unrestr. Base BM25 68.15 60.24 79.26 68.36 85.03 73.72 89.09 78.21 ICT 86.61 79.72 91.60 85.11 93.77 88.51 94.41 88.07 RoBERTa-UnrestrData — 80.73 — 87.03 — 88.44 — 89.91 RoBERTa-BaseData — 82.33 — 88.07 — 90.15 — 90.83

- Table 2: Accuracy (%) on the unrestricted and base test configurations of HALvest-Contrastive across span lengths k. All models compared at 3.1k steps for fairness.

###### R@20

- 0.5

- 0.6

- 0.7

- 0.8

- 0.9 R@100

0.7

0.6

Recall

0.5

0.4

0.3

0.6

###### nDCG@20

- 0.3

- 0.4

- 0.5

- 0.6 nDCG@100

0.5

nDCG

0.4

0.3

0.2

base-2 base-4 base-6 base-8 base-10

base-2 base-4 base-6 base-8 base-10

Subset (sentences per span)

Subset (sentences per span)

LI PLI whole-word PLI n-gram 2 PLI n-gram 3 PLI n-gram 4 PLI n-gram 5

Figure 3: Retrieval metrics on HALvest-Contrastive across all five subsets (base-k = k contiguous sentences per anchor, positive, and negative span). Each panel reports one metric, each marker family one model. Bigram patches slightly over-perform at short spans, trigram patches take over from k=6 onwards. The full numerical tables underlying the figure are in Appendix D.

the span. Neural models extract a signal which marginal value saturates before the lexical signal does. All subsequent experiments use the base configuration at k = 4 as the primary reporting split, with full per-k tables in Appendix D.

#### 5.3 Granularity across span lengths

We trained ModernBERT (Warner et al., 2025) with InfoNCE for one epoch under each of the granularity introduced in §4.2: mean pooling, late interaction, whole-word PLI, and fixed-n PLI for n ∈ {2,3,4,5}. We ran each model on every subset of HALvest-Contrastive (k ∈ {2,4,6,8,10}). Figure 3 reports the four ranking metrics across the five subsets for the multi-vector configurations. The tables with every values are in Appendix D. We exclude mean pooling from the figure to keep the y-axis range readable, the comparison with mean pooling lies in a different range and is captured by the headline below.

Under a matched training objective, moving from single-vector pooling to late interaction yields

a greater qualitative gain in AA than it yields in passage retrieval. We also observe a similar gain at inference: keeping the encoder fixed and swapping mean pooling for multi-vector scoring improves performance by about 20% (Appendix D). A similar 20% lift appears when applying the same swap to E5, with an XLM-RoBERTa backbone (Conneau et al., 2020) (Appendix G, Table 14), suggesting the advantage is not specific to the ModernBERT backbone. Within the multi-vector family, no fixed policy is uniformly best across subsets. Token-level LI is competitive but not dominant, whole-word PLI sits in the middle of the cluster and the n-gram variants reorder as k grows.

#### 5.4 Alignment and uniformity diagnostics

We also report the alignment and uniformity losses (Wang and Isola, 2020) as a representationquality diagnostic alongside retrieval and triplet metrics. Alignment measures how close positive pairs are in embedding space, while uniformity measures how spread out random pairs are. Figure 4 plots the pair for every model trained in this paper.

Initial (no FT)

Uniformityloss(lowerisbetter)u

LI

0.25

PLI whole-word

0.50

PLI n-gram 2

0.75

1.00

PLI n-gram 3

1.25

1.50

- PLI n-gram 4

- PLI n-gram 5

1.75

2.00

Mean

0.1 0.2 0.3 0.4 0.5

Alignment loss (lower is better)

Figure 4: Alignment against uniformity (Wang and Isola, 2020) across pooling strategies on the base-4’s validation set.

The results align with the retrieval literature.

Mean pooling has very low uniformity (u = −2.0) but high alignment loss (α = 0.154): random pairs are well spread out, but same-author pairs are not particularly close. We hypothesize that meanpooled vectors are topically well-spread, but stylistically undiscriminating, which matches explains the performance gap. LI and whole-word PLI sit in the upper-left region. The fixed-n PLI variants trade alignment for uniformity along a clean monotonic axis as n grows, with n = 5: the representation has become dispersed at the cost of pulling positives apart.

#### 5.5 AA is not semantic retrieval

To test whether the gains come from semantic retrieval alone, we compute the cosine similarities between anchors and positives and between anchors and negatives using both E5 (Wang et al., 2024) and our fine-tuned mean-pooled model on a sample of 4,000 triplets (Kantharuban et al., 2026). If the two signals were equivalent, they would separate same-author and different-author pairs in a similar way.

Figure 5 shows that neither is the case. On the stylistic axis, our fine-tuned model separates sameauthor pairs from different-author pairs by a margin of 0.30 in mean cosine similarity. On the semantic axis, E5 separates the same populations by only 0.029. Semantic similarity under E5 is therefore almost invariant to the authorship relation between two passages, whereas the stylistic representation responds to it strongly. The pooled Pearson correlation across all 4,000 pairs is r = 0.454, which may suggest partial agreement between the two signals. This coefficient, however, is dominated by the between-group shift we just described: the within-group structure in the scatter shows stylistic similarity varying across the full range while semantic similarity stays confined in a narrow vertical band.

#### 5.6 Retrieval failure modes

To further emphasize the distinction between AA and simple retrieval, we inspect the failure modes of our strongest PLI baseline (PLI n-gram 2 on k = 4) by ranking 500 sampled queries against a 1000-document validation pool (5 seeds × 100 queries). The model places the true positive at rank 1 in 20.8% of cases and within the top 5 in 39.8%, with a long tail (16.6% of true positives at ranks 101–500 and 8.0% beyond rank 500).

| |r = 0.454 p = 0.00e+00<br><br>Same-author (pos)<br><br>Diff-author (neg)| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.0

0.8

StylisticSimilarity(Fine-tunedModel)

0.6

0.4

0.2

0.0

0.2

0.70 0.75 0.80 0.85 0.90 0.95 1.00 Semantic Similarity (Multilingual-E5)

Figure 5: Cosine similarity between HALvestContrastive anchor-candidate pairs, measured with a pretrained multilingual-E5 encoder (x-axis) and with our fine-tuned stylistic model (y-axis) using mean pooling. We report E5 performance’s on HALvest-Contrastive and PAN19 in Appendix G.

We call “distractors” highly ranked negative documents, and categorize them. Some distractors share the query’s domain at 61% across rank buckets, against 20% for random pairs. The model still has learned domain as a strong similarity cue, consistent with HALvest-Contrastive’s field-matched negative sampling. The top-K distractors share at least one author with the query in 11.4% of cases at ranks 2–5 and 11.8% at ranks 6–20, but 0.0% across 500 random pairs. The author-overlap drops by a factor of 3.5 between the top-20 distractors and the subsequent ones ranked lower. The presence of overlapping authors correlates strongly with retrieval success. The Jaccard distance with the query (unigram) is essentially flat across rank buckets (0.086, 0.087, 0.085, with random at 0.073). The Jaccard distance with the true positives however, is more informative: 0.129 at rank 1, 0.131 at ranks 2– 5, 0.126 at ranks 6–20, dropping to 0.076 at lower ranks. High-ranked distractors do not look textually similar to the query, but they do look textually similar to the true positive. This is structurally different from a topic-retrieval failure mode, where lexical query overlap would dominate.

#### 5.7 Cross-domain and patch-size scaling

Table 3 reports triplet accuracy on PAN19 for every trained model, evaluated zero-shot. The pattern

Model Triplet accuracy Mean pooling 0.520 LI (full token) 0.719 PLI whole-word 0.715 PLI n-gram 5 0.704

- PLI n-gram 3 0.724 PLI n-gram 2 0.740
- PLI n-gram 4 0.740 E5 zero-shot 0.628

- Table 3: Zero-shot triplet accuracy on PAN19. All trained models are evaluated without any further finetuning.

within fixed-policy variants is consistent with what we saw on HALvest-Contrastive’s longer subsets: token-level LI is competitive but not best, wholeword sits in the middle.

An empirical patch-size fit. The shift in optimal n across subsets is well approximated by a oneparameter square-root fit. Let S denote the token count of an anchor, which varies by subset because longer-k subsets fill more of the ModernBERT context (512 tokens). Across six subsets with span length ranging from 69.5 tokens to 512.0 tokens, the best fixed n is fitted by

n⋆ ≈ 0.18 ·

√

S. (3)

Subset S 0.18

√

S n⋆

HALvest-C base-2 69.5 1.50 1 HALvest-C base-4 133.2 2.08 2 HALvest-C base-6 201.4 2.55 3 HALvest-C base-8 266.2 2.94 3 HALvest-C base-10 327.6 3.26 3 PAN19 (zero-shot) 512.0 4.07 4

- Table 4: Empirical fit of n⋆. At PAN19’s sequence length the fit matches one of the two tied values. The limited resolution of our patch sizes means ties of this kind are expected.

The relationship is empirical, not derived. The relationship also explains the apparent disagreement between rankings in Figure 3 for a given subset: each subset samples a different point on the same curve.. We treat Equation (3) as a descriptive fit over the range we tested. It tells a practitioner who has chosen a context length what fixed patch size to start at, but does not say why √

· is the right exponent. Under that empirical fit, the best-performing patch size grows sublinearly with sequence length, implying that the number of stored patch vectors also grows sublinearly

(approximately proportional to √

S over the range we tested). Therefore patch-level interaction gives a much cheaper alternative to full token-level LI while preserving most of the gain on AA.

#### 5.8 Scaling the negative-pool size

To verify that HALvest-Contrastive exhibits the same qualitative scaling as retrieval benchmarks, we retrained the late-interaction model at perdevice batch 64, yielding 512 in-batch negatives per anchor after all-gather (versus 256 at batch 32). Retrieval metrics improved by a small but consistent margin on base-4 and the neighboring splits. R@20 increases by a fraction of a percentage point, R@100 by slightly more, and the direction is the same across nDCG metrics. This is consistent with the standard retrieval findings (§2).

### 6 Conclusion

Scholarly prose is a strong testbed for stylometry because its low-entropy register compresses topical variation and makes the stylistic cues easier to isolate. HALvest-Contrastive exploits this structure by pairing same-author passages across different papers within a disciplinary field. Our large-scale benchmark substantially weakens lexical shortcuts.

The transfer from retrieval to authorship attribution is not uniform across design choices. Moving from single-vector pooling to multi-vector interaction produces large gains, but full token-level matching is not uniformly optimal. Grouping neighboring tokens into patches preserves most of the retrieval improvement while reducing the number of stored vectors and pairwise MaxSim comparisons. Across both HALvest-Contrastive and PAN19, the best-performing fixed patch size increases gradually with sequence length and is well approximated, over the range we tested, by n⋆ ≈ 0.18 ·

√

S.

More broadly, our results suggest that authorship attribution behaves differently from semantic retrieval despite their shared contrastive structure. Semantic matching benefits from fine-grained lexical alignment, whereas stylometric similarity appears to emerge at an intermediate interaction scale.

### Limitations

Author-set rather than individual attribution. The triplet label is the set of co-authors rather than a single writer. We argue in §3 that this is

the correct target for scholarly repositories and that the forensic-linguistics literature supports it as a principled design choice rather than a concession (Dauber et al., 2019). Author-set labels represent a reliable level of supervision already available in scholarly metadata. Individual attribution within multi-author documents is a different task, and we leave it to future work.

GROBID. HALvest’s text is produced by running GROBID over PDFs. Relying on GROBID as a proxy before performing PDF parsing may seem indirect, as more accurate ways exist to extract text from PDFs—vision-language models and Optical Character Recognition (OCR), for instance. However, having a structured view of our articles enables finer control over text output. GROBID’s layout-based extraction preserves document structure, essential for our contrastive sampling strategy where we need to identify and extract contiguous sentence spans across different sections of papers. GROBID, despite being a conservative extractor, can introduce systematic biases. First, Arabic scripts are inverted in the output: word order is reversed because the extractor’s layout model assumes left-to-right reading, which makes the Arabic portion of HALvest effectively unusable. Second, CJK languages lack the whitespace cues that GROBID’s tokeniser relies on for length-based filtering, so coverage in these languages is sparser and more variable. Third, mathematical notation is often stripped or rendered inconsistently, which can bias mathematical-sciences domains away from passages where notation is central. As a rough estimation, preprocessing of HAL’s French split at the sentence level (Antoun et al., 2024) retained about 52% of the total tokens (4.7 billion tokens) after tab removal and formula handling, and this retention rate varies across languages.

Learned patching is unfinished. We attempted an end-to-end trainable patching policy but found that simple regularizers either collapse the policy to token-level scoring (when the regularizer is unbounded below) or to single-patch sequences (when constraining only the population mean of cut probabilities below the inference threshold). See Appendix F for the math and the empirical failuremode characterization. A successful learned variant would require multi-term regularization in the BLT (Pagnoni et al., 2025) style with hyperparameter tuning beyond our compute budget for this submission. The cross-attention compressor variant of

Pagnoni et al. (2025) was explored alongside it and shares the same regularizer-design weaknesses.

√

Empirical scaling fit. The 0.18·

S relationship in §5.7 is descriptive. Six points across two registers fit a one-parameter curve cleanly, but we do not provide a derivation, and we do not yet know whether the same coefficient transfers to other encoders or longer context lengths. It should be read as a practical summary for choosing a starting patch size, not as a universal law.

### Ethics

Licensing. HAL is an open-access repository. The documents in HALvest were selected from papers deposited under permissive licenses compatible with derivative redistribution.

Author identity. The author labels in HALvest are exactly those that appear in the public metadata of the source papers. No personally identifiable information beyond what is already published is introduced by the derivative corpus. Author names are tied to signed, publicly-archived academic works; the corpus does not include private correspondence or unpublished drafts.

Dual-use considerations. Stylometric authorship attribution is a dual-use technology. It enables beneficial applications (plagiarism detection, LLMgenerated-text detection, verification of anonymous peer review integrity) but can in principle be used to deanonymize pseudonymous writers, including whistleblowers and politically vulnerable authors. HALvest itself contains only signed academic work, so the corpus-specific deanonymisation risk is low; however, models trained on HALvest may be portable, and we encourage users deploying HALvest-trained models in downstream tasks to consider the risk surface of their own application domain.

### Acknowledgments

The authors are grateful to the CCSD staff—the service unit in charge of HAL—in particular Achraf Azhar. We also thank Patrice Lopez, for providing resources and support to better handle GROBID, as well as Arij Riabi, Brahim Talb and Menel Mahamdi for the productive discussions. Finally, we extend our thank to Yannis Karmim and Lydia Nishimwe for their proofreading. This work was partially realized on computing HPC and storage

resources provided by IDRIS thanks to the grant GCDA1016807 on the DALIA supercomputer.

### References

Bo Ai, Yuchen Wang, Yugin Tan, and Samson Tan. 2022. Whodunit? learning to contrast for authorship attribution. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing, Volume 1: Long Papers, pages 1142–1157, Online only. Association for Computational Linguistics.

Milad Alshomary, Nikhil Reddy Varimalla, Vishal Anand, Smaranda Muresan, and Kathleen McKeown. 2025. Layered insights: Generalizable analysis of human authorial style by leveraging all transformer layers. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 10279–10292, Suzhou, China. Association for Computational Linguistics.

Malik Altakrori, Jackie Chi Kit Cheung, and Benjamin C. M. Fung. 2021. The topic confusion task: A novel evaluation scenario for authorship attribution. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4242–4256, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Wissam Antoun, Francis Kulumba, Rian Touchent, Éric de la Clergerie, Benoît Sagot, and Djamé Seddah. 2024. CamemBERT 2.0: A smarter french language model aged to perfection. Preprint, arXiv:2411.08868.

Shlomo Argamon-Engelson, Moshe Koppel, and Galit Avneri. 1998. Style-based text categorization: What newspaper am i reading? In Proceedings of the AAAI Workshop on Learning for Text Categorization.

Steven Bird, Robert Dale, Bonnie Dorr, Bryan Gibson, Mark Joseph, Min-Yen Kan, Dongwon Lee, Brett Powley, Dragomir Radev, and Yee Fan Tan. 2008. The ACL anthology reference corpus: A reference dataset for bibliographic research in computational linguistics. In Proceedings of the Sixth International Conference on Language Resources and Evaluation, Marrakech, Morocco. European Language Resources Association.

John Burrows. 2002. Delta: A measure of stylistic difference and a guide to likely authorship. Literary and Linguistic Computing, 17(3):267–287.

Florian Cafiero and Jean-Baptiste Camps. 2019. Why molière most likely did write his plays. Science Advances, 5(11):eaax5489.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on

Machine Learning, volume 119 of ICML ’20, pages 1597–1607. JMLR.org.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Edwin Dauber, Aylin Caliskan, Richard Harang, Gregory Shearer, Michael Weisman, Frederica Nelson, and Rachel Greenstadt. 2019. Git blame who? stylistic authorship attribution of small, incomplete source code fragments. Proceedings on Privacy Enhancing Technologies, 2019(3):389–408.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1: Long and Short Papers, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Rezarta Islamaj Do˘gan, Robert Leaman, and Zhiyong Lu. 2014. NCBI disease corpus: A resource for disease name recognition and concept normalization. Journal of Biomedical Informatics, 47:1–10.

William Falcon and The PyTorch Lightning team. 2019. PyTorch Lightning.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

GROBID Repository. 2008–2024. GROBID. https: //github.com/kermitt2/grobid. Preprint, swh:1:dir:dab86b296e3c3216e2241968f0d63b68e8209d3c.

Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing neural bag of whole-words with ColBERTer: Contextualized late interactions using enhanced reduction. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management, CIKM ’22, pages 737–747, New York, NY, USA. Association for Computing Machinery.

Javier Huertas-Tato, Adrián Girón-Jiménez, Alejandro Martín, and David Camacho. 2024. Isolating authorship from content with semantic embeddings and contrastive learning. arXiv preprint. ArXiv:2411.18472 [cs].

Eric Jang, Shixiang Gu, and Ben Poole. 2017. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations.

Patrick Juola. 2015. The rowling case: a proposed standard analytic protocol for authorship questions. Digital Scholarship in the Humanities, 30(suppl_1):i100– i113.

Hyukkyu Kang, Injung Kim, and Wook-Shin Han. 2025. TRIAL: Token relations and importance aware lateinteraction for accurate text retrieval. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 16864–16877, Suzhou, China. Association for Computational Linguistics.

Anjali Kantharuban, Aarohi Srivastava, Fahim Faisal, Orevaoghene Ahia, Antonios Anastasopoulos, David Chiang, Yulia Tsvetkov, and Graham Neubig. 2026. IDIOLEX: Unified and continuous representations for idiolectal and stylistic variation. arXiv preprint. ArXiv:2604.04704 [cs].

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 6769–6781, Online. Association for Computational Linguistics.

Vlado Kešelj, Fuchun Peng, Nick Cercone, and Calvin Thomas. 2003. N-gram-based author profiles for authorship attribution. In Proceedings of the Conference of the Pacific Association for Computational Linguistics, pages 255–264.

Mike Kestemont, Efstathios Stamatatos, Enrique Manjavacas, Walter Daelemans, Martin Potthast, and Benno Stein. 2019. PAN19 authorship analysis: Cross-domain authorship attribution.

Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, pages 39–48, New York, NY, USA. Association for Computing Machinery.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2021. Supervised contrastive learning. Preprint, arXiv:2004.11362.

Moshe Koppel, Jonathan Schler, and Shlomo Argamon. 2009. Computational methods in authorship attribution. Journal of the American Society for Information Science and Technology, 60(1):9–26.

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the

57th Annual Meeting of the Association for Computational Linguistics, pages 6086–6096, Florence, Italy. Association for Computational Linguistics.

Guangyi Liu, Zichao Yang, Tianhua Tao, Xiaodan Liang, Junwei Bao, Zhen Li, Xiaodong He, Shuguang Cui, and Zhiting Hu. 2022. Don’t take it literally: An edit-invariant sequence loss for text generation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2055–2078, Seattle, United States. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. Preprint, arXiv:1907.11692.

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel Weld. 2020. S2ORC: The semantic scholar open research corpus. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4969–4983, Online. Association for Computational Linguistics.

Sean MacAvaney, Antonio Mallia, and Nicola Tonellotto. 2025. Efficient constant-space multi-vector retrieval. In Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part III, pages 237–245, Berlin, Heidelberg. Springer-Verlag.

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. 2017. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations.

Frederick Mosteller and David L Wallace. 1963. Inference in an authorship problem: A comparative study of discrimination methods applied to the authorship of the disputed federalist papers. Journal of the American Statistical Association, 58(302):275–309.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2019. Representation learning with contrastive predictive coding. arXiv preprint. ArXiv:1807.03748 [cs, stat].

Artidoro Pagnoni, Ramakanth Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason E. Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, and Srini Iyer. 2025. Byte latent transformer: Patches scale better than tokens. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, Volume 1: Long Papers, pages 9238–9258, Vienna, Austria. Association for Computational Linguistics.

Rafael A. Rivera-Soto, Olivia Elizabeth Miano, Juanita Ordonez, Barry Y. Chen, Aleem Khan, Marcus Bishop, and Nicholas Andrews. 2021. Learning universal authorship representations. In Proceedings of

the 2021 Conference on Empirical Methods in Natural Language Processing, pages 913–919, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at trec-3. In Text Retrieval Conference.

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715–3734, Seattle, United States. Association for Computational Linguistics.

Upendra Sapkota, Steven Bethard, Manuel Montes, and Thamar Solorio. 2015. Not all character n-grams are created equal: A study in authorship attribution. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 93–102, Denver, Colorado. Association for Computational Linguistics.

Jonathan Schler, Moshe Koppel, Shlomo Argamon, and James W. Pennebaker. 2006. Effects of age and gender on blogging. In AAAI Spring Symposium: Computational Approaches to Analyzing Weblogs, volume 6, pages 199–205.

Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. 2008. Collective classification in network data. AI Magazine, 29(3):93–93. Number: 3.

Yanir Seroussi, Ingrid Zukerman, and Fabian Bohnert. 2014. Authorship attribution with topic models. Computational Linguistics, 40(2):269–310.

Pucktada Treeratpituk and C. Lee Giles. 2009. Disambiguating authors in academic publications using random forests. In Proceedings of the 9th ACM/IEEECS Joint Conference on Digital Libraries, JCDL ’09, pages 39–48, New York, NY, USA. Association for Computing Machinery.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Jan Philip Wahle, Terry Ruas, Saif Mohammad, and Bela Gipp. 2022. D3: A massive dataset of scholarly metadata for analyzing the state of computer science research. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 2642–2651, Marseille, France. European Language Resources Association.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2024. Text embeddings by weakly-supervised contrastive pre-training. Preprint, arXiv:2212.03533.

Tongzhou Wang and Phillip Isola. 2020. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proceedings of the 37th International Conference on Machine Learning, ICML ’20. JMLR.org.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Griffin Thomas Adams, Jeremy Howard, and Iacopo Poli. 2025. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, Volume 1: Long Papers, pages 2526–2547, Vienna, Austria. Association for Computational Linguistics.

Anna Wegmann and Dong Nguyen. 2021. Does it capture STEL? a modular, similarity-based linguistic style evaluation framework. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7109–7130, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Anna Wegmann, Marijn Schraagen, and Dong Nguyen. 2022. Same author or just same topic? towards content-independent style representations. In Proceedings of the 7th Workshop on Representation Learning for NLP, pages 249–268, Dublin, Ireland. Association for Computational Linguistics.

Yuxuan Zong and Benjamin Piwowarski. 2025. Towards lossless token pruning in late-interaction retrieval models. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’25, pages 2407–2417, New York, NY, USA. Association for Computing Machinery.

### A InfoNCE gradient derivation

We derive Equation 2 from Equation 1. Writing Z = exp(sp/τ) + n′∈N exp(sn′/τ) for the partition function and sp = s(a,p), sn = s(a,n) for the anchor-positive and anchor-negative similarities, the InfoNCE loss for a single anchor is

exp(sp/τ) Z

sp τ

L = −log

= −

+ log Z.

Differentiating with respect to a specific negative similarity sn (treating sp and the other negatives {sn′ : n′ ̸= n} as fixed inputs),

∂L ∂sn

1 Z ·

1 τ · exp(sn/τ)

=

exp(sn/τ) exp(sp/τ) + n′∈N exp(sn′/τ)

1 τ ·

.

=

The right-hand side is exactly the softmax weight of n in the contrastive denominator, scaled by 1/τ. Negatives with high similarity to the anchor receive exponentially more gradient than negatives with low similarity. In the limit of one dominant negative, the gradient collapses onto that negative alone, reproducing the behaviour of hard-negative mining without an explicit mining procedure.

The comparison with the triplet loss follows from inspecting its form directly. With a fixed margin m, the triplet loss is Ltrip(sp,sn) = [m − sp + sn]+, whose gradient with respect to sn is +1 when sp − sn < m and 0 otherwise. The gradient does not depend on how close sn is to sp within the active region, and it vanishes entirely once the margin is satisfied. In that sense, triplet loss does not have a hard-negative-focus property in the InfoNCE sense: the loss treats all active negatives identically and ignores all inactive ones.

### B HALvest language and domain statistics

Language # Docs # Tokens English 464,679 8,158,933,235 French 199,216 9,018,529,985 Spanish 2,975 69,221,667 Italian 1,172 48,747,986 Portuguese 934 32,918,832 German 652 12,225,960 Russian 245 5,763,532 Chinese 160 2,861,585 ... ... ... Total (56 languages) 778k+ 17.4B

Table 5: Language distribution of HALvest. Full perlanguage counts (56 languages, including Basque, Catalan, Persian, and other low-resource entries) are included with the dataset release.

We selected eight features to represent HAL submissions:

- • halid: submission’s unique identifier assigned by HAL.
- • lang: the language of the document, as filled by the depositor.
- • title: title of the document.

Domain Code # Documents # Tokens Humanities and Social Sciences shs 156,566 5,614,423,171 Computer Science info 148,316 2,573,673,455 Life Sciences sdv 115,744 3,145,323,780 Engineering Sciences spi 102,751 2,254,653,825 Physics phys 65,991 1,503,190,749 Mathematics math 62,921 1,638,500,361 Chemical Science chim 40,012 899,507,319 Environmental Science sde 31,575 579,076,669 Sciences of the Universe sdu 23,557 682,356,264 Cognitive Science scco 11,772 227,487,096 Statistics stat 10,579 184,678,350 Quantitative Finance qfin 3,451 68,518,636 Nonlinear Sciences nlin 1,972 30,694,088

Table 6: Disciplinary domains retained for HALvestContrastive hard-negative mining. Three of the 16 HALvest domains are excluded from the contrastive derivation for having too few documents to support fieldmatched negative sampling; they remain in HALvest itself.

- • domain: list of fields of study3.
- • timestamp: time of access.
- • year: publication year of the document if relevant. Otherwise, it is set to year 1.
- • url: URL to access the PDF.
- • authors: list of authors. Table 5 reports per-language document and to-

ken counts for HALvest. Table 6 reports the 13 domains retained for HALvest-Contrastive triplet mining.

| | | | | |10.77<br><br>10.82<br><br>10.82<br><br>10.91<br><br>10.94<br><br>10.95<br><br>10.96<br><br>10.97<br><br>11.02<br><br>11.04<br><br>11.04<br><br>11.18<br><br>11.23| |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

sdv

chim

sde

sdu

phys

math

spi

shs

info

scco

qfin

stat

nlin

0.0 2.5 5.0 7.5 10.0 12.5

Char-trigram entropy (bits)

Figure 6: Token-level entropy across the 13 disciplinary domains of HALvest.

Figure 6 plots token-level entropy across do-

3https://hal.science/browse/domain

mains, which is approximately uniform. Academic writing remains low-entropy throughout the fields.

10000

Config

base-10

base-2 base-4 base-6 base-8

8000

Numberoftriplets

6000

4000

2000

0

1 2 3 4 5 6 7 7+ Author-set size

Figure 7: Author-set sizes in HALvest-Contrastive. Each color represents a subset of k contiguous sentences. Respectively 10, 2, 4, 6 and 8. Subsets are grouped by their number of author in ascending order. The majority of the documents, no matter the subset are single-authored.

Figure 7 plots the author sets’ distribution. The median author-set size is 1, the mean is 1.84, and the maximum set size in the corpus is 44. An author comprises three features:

- • name: string for the author name, as filled by the depositor.
- • affiliations: list of unique identifiers attributed by HAL to the institutions where the author belongs.
- • halauthorid: unique identifier assigned by HAL to each author registered on the online repository. If an author is not registered on HAL, they are considered unidentified and are assigned a halauthorid of “0”.

### C Full hyperparameters and setup

Unless stated otherwise, all trained models share a ModernBERT-base backbone (Warner et al., 2025) and are trained for a single epoch on the base configuration of HALvest-Contrastive. Experiments use four H100 GPUs with a per-device batch size of 32 and gradient accumulation of 2, so that the effective batch is 64 and each anchor is contrasted against 256 in-batch negatives after all-gather. The learning rate is 3×10−5, the InfoNCE temperature τ is 0.5, weight decay is 0.1, and we train in 16-bit mixed precision. Runs with four GB200 at a perdevice batch size of 64 (512 in-batch negatives) are reported in §5.8 as a scaling reference.

Table 7 reproduces the training configuration of every main-body model. τ is a temperature parameter, set to 0.5 throughout. This value accounts for the large magnitude of MaxSim scores on long sequences, which require a higher temperature than

Hyperparameter Value

Backbone ModernBERT-base Epochs 1 Per-device batch size 32 Gradient accumulation 2 GPUs 4×H100 (primary), 4×GB200 (scaling) Effective in-batch negatives 256 (H100), 512 (GB200) Learning rate 3 × 10−5 Weight decay 0.1 Precision 16-bit mixed InfoNCE temperature τ 0.5

- Table 7: Training and model hyperparameters shared across all main-body models. Learned-PLI hyperparameters (kept for reproducibility despite the negative result) appear in Appendix F.

typical passage-retrieval settings (τ ∈ [0.05,0.1]) to avoid saturating the softmax. All other settings are defaults of the PyTorch Lightning (Falcon and The PyTorch Lightning team, 2019) training pipeline that accompanies the release.

D Per-split results

Tables 8 to 12 report retrieval metrics for every trained model on every test split (base-2, base-4, base-6, base-8, base-10), including the learnedPLI variants whose negative-result analysis is in Appendix F. We omit layerwise pooling variants, abandoned after validation-set evaluation only marginally improving over mean pooling.

Model R@20 R@100 nDCG@20 nDCG@100 Mean pooling 0.060 0.173 0.032 0.058 Mean pooling (LI) 0.073 0.190 0.041 0.067 LI 0.280 0.457 0.202 0.243

- PLI n-gram 2 0.284 0.465 0.200 0.242
- PLI n-gram 3 0.281 0.468 0.196 0.239
- PLI n-gram 4 0.265 0.451 0.181 0.224
- PLI n-gram 5 0.266 0.457 0.180 0.224 PLI whole-word – – – – PLI learned 0.258 0.450 0.171 0.215

- Table 8: HALvest-Contrastive base-2: primary retrieval metrics. Learned-PLI rows reflect the retrained checkpoint analysed in Appendix F; the policy collapses to single-patch sequences and the resulting numbers track mean pooling.

Model R@20 R@100 nDCG@20 nDCG@100 Mean pooling 0.121 0.294 0.063 0.101 Mean pooling (LI) 0.147 0.328 0.082 0.121 LI 0.485 0.678 0.364 0.408

- PLI n-gram 2 0.497 0.700 0.365 0.411
- PLI n-gram 3 0.495 0.699 0.361 0.407
- PLI n-gram 4 0.472 0.687 0.339 0.387
- PLI n-gram 5 0.478 0.693 0.341 0.390 PLI whole-word 0.482 0.680 0.358 0.403 PLI learned 0.452 0.670 0.314 0.363

- Table 9: HALvest-Contrastive base-4: primary retrieval metrics.

Model R@20 R@100 nDCG@20 nDCG@100 Mean pooling 0.163 0.377 0.087 0.132 Mean pooling (LI) 0.193 0.420 0.107 0.156 LI 0.603 0.779 0.464 0.503

- PLI n-gram 2 0.616 0.797 0.466 0.507
- PLI n-gram 3 0.617 0.803 0.464 0.506
- PLI n-gram 4 0.593 0.788 0.442 0.485
- PLI n-gram 5 0.604 0.797 0.443 0.486 PLI whole-word 0.601 0.776 0.457 0.496 PLI learned – – – –

- Table 10: HALvest-Contrastive base-6: primary retrieval metrics.

Model R@20 R@100 nDCG@20 nDCG@100 Mean pooling 0.211 0.450 0.114 0.164 Mean pooling (LI) 0.247 0.494 0.143 0.195 LI 0.690 0.847 0.537 0.571

- PLI n-gram 2 0.706 0.862 0.545 0.580
- PLI n-gram 3 0.713 0.865 0.545 0.579
- PLI n-gram 4 0.688 0.855 0.521 0.558
- PLI n-gram 5 0.699 0.863 0.523 0.560 PLI whole-word 0.688 0.849 0.534 0.570 PLI learned 0.659 0.838 0.483 0.522

- Table 11: HALvest-Contrastive base-8: primary retrieval metrics.

- E Alignment and uniformity values Numerical values for Figure 4 are in Table 13.
- F Learned PLI: failure-mode analysis

This appendix documents our attempt at learning differentiable patch boundaries. The architecture is rather straightforward. The difficulty lies in the regularizer. We tried two natural single-term forms. The first not lower bounded and collapsed the policy onto token-level scoring. The second was bounded but failed to interact with the inference threshold and collapsed the policy onto a single patch per sequence. A successful regularizer would ressemble that of BLT (Pagnoni et al., 2025) but require hyperparameter sweep beyond our compute budget for this article.

#### F.1 Architecture and notation

Let ht ∈ Rd be the encoder’s hidden state at token position t, and let S denote the number of valid (non-padding) query positions in a sequence. A two-layer feed-forward network (a patch-boundary predictor) on top of the ModernBERT hidden states produces a boundary logit per position,

ℓt = W2 ϕ(W1ht + b1) + b2, (4)

with W1 ∈ R(d/4)×d, W2 ∈ R1×(d/4), and ϕ a GELU. The cut probability is qt = σ(ℓt). At training time, cut decisions are sampled with a GumbelSoftmax alongside a straight-through gradient esti-

Model R@20 R@100 nDCG@20 nDCG@100 Mean pooling 0.246 0.499 0.130 0.182 Mean pooling (LI) 0.287 0.547 0.163 0.217 LI 0.740 0.880 0.586 0.616

- PLI n-gram 2 0.760 0.897 0.595 0.625
- PLI n-gram 3 0.765 0.901 0.594 0.624
- PLI n-gram 4 0.742 0.892 0.571 0.604
- PLI n-gram 5 0.754 0.899 0.576 0.608 PLI whole-word 0.742 0.887 0.580 0.612 PLI learned 0.714 0.872 0.534 0.572

- Table 12: HALvest-Contrastive base-10: primary retrieval metrics.

Model α u

Pretrained, no fine-tuning 0.069 −0.171 Mean pooling 0.154 −2.000 LI (full token) 0.091 −0.331 PLI whole-word 0.098 −0.378

- PLI n-gram 2 0.201 −0.717
- PLI n-gram 3 0.320 −1.150
- PLI n-gram 4 0.438 −1.565
- PLI n-gram 5 0.519 −1.909

- Table 13: Alignment α and uniformity u per model. Lower α indicates positives are closer in embedding space. Lower u indicates that random pairs are more uniformly distributed. The learned-PLI row reflects the retrained checkpoint analysed in Appendix F.

mation. We anneal the temperature τGS from 1.0 to 0.1 over the first 20,000 optimizer steps. The first valid position is forced to be a cut, this guarantees at least one patch regardless of the sampler. At inference time, the cut decision is hard-thresholded at 0.5.

The total training loss is L = LInfoNCE + λ · Lpatch with λ = 0.1, where Lpatch is the patch regularizer whose role is to prevent degenerate solutions.

F.2 Unbounded log-sum penalty The first form was

L(1)patch = −log

S

qt . (5)

t=1

The intuition was: as t qt → 0, the regularizer diverges to +∞ and rules out the single-patch degeneracy. Beyond that, the regularizer was supposed to be roughly flat at high cut counts, letting InfoNCE push the cut count freely.

However, for t qt > 1, the function −log( t qt) is monotonically decreasing in t qt

and is not lower bounded. The regularizer, therefore, supplies a uniform downward push on every cut probability, regardless of position.

What (5) needed was a regularizer that diverges

to +∞ at both extremes of t qt. Restoring the missing one requires committing to a target rate,

which is what the second attempt did. F.3 Target-rate squared error

The second form was a target-rate-matching squared-error penalty,

L(2)patch =

S

1 S

qt − ptarget

t=1

2

, (6)

with ptarget = 1/3. This target rate corresponds to a mean patch length of 1/ptarget = 3 tokens, which lands in the middle of the empirical n-gram optima we identified (best n is 2 at base-2 and base-4, 3 at base-6 onwards). The hyperparameter is therefore not tuned but set from independent empirical evidence. Equation (6) has a minimum at q¯ := S1 t qt = ptarget, is lower-bounded at zero, is symmetric and is differentiable everywhere.

The retrained model satisfies the regularizer exactly and produces a single patch per sequence.

After retraining under L(2)patch, the diagnostic check on the resulting checkpoint reveals q¯ ≈ 1/3 for

every sequence: L(2)patch is doing exactly what was asked of it. However, cut-off the threshold is 0.5.

If every position has qt ≈ 1/3, then no position satisfies qt > 0.5, and the inference rule produces zero cuts.

The MSE regularizer only constrains the population mean of cut probabilities. It does not place any constraint on the distribution shape. Besides, InfoNCE provides no pressure to break the uniform policy. The regularizer does not align its constraint with the inference threshold. With ptarget = 1/3 and threshold 0.5, the model can satisfy the regularizer exactly while producing zero cuts at inference. Setting ptarget = 0.5 does not fix this, as qt = 0.5 everywhere produces random cuts under Gumbel sampling.

The cleanest fix follows the byte-latenttransformer pattern (Pagnoni et al., 2025), two regularizers acting in concert. Both regularizers introduce hyperparameters that need joint tuning.

### G E5 zero-shot full metrics

We report E5 zero-shot metrics under both the dense scorer (cosine similarity of mean-pooled embeddings, the scoring mode E5 was trained for) and

Metric Dense LI HALvest-Contrastive base-4

R@20 0.167 0.203 R@100 0.269 0.307 nDCG@20 0.124 0.152 nDCG@100 0.146 0.175 Triplet accuracy 0.789

PAN19 zero-shot

R@20 0.097 0.179 nDCG@20 0.029 0.064 Triplet accuracy 0.628

Table 14: E5 (Wang et al., 2024) zero-shot metrics on HALvest-Contrastive base-4 and PAN19, under dense and late-interaction scorers. Triplet accuracy is scorerindependent.

the late-interaction scorer (token-level MaxSim, included as an ablation to verify that the conclusions of §5.5 are not an artifact of the chosen scorer).

The late-interaction scorer improves retrieval metrics uniformly over the dense scorer, but the overall trend remains the same: E5 retains a high triplet accuracy (0.789 on HALvest-C, 0.628 on PAN19) while struggling on retrieval. The gap between the two metric families is the motivating observation of §5.5.

### H Sequence length

96

94

92

90

Accuracyin%

88

86

84

trained on base, tested on unrestricted

82

trained on base, tested on base

trained and tested on unrestricted

trained on unrestricted, tested on base

80

2 4 6 8

Number of sentences

Figure 8: Two fine-tuned RoBERTa models, in blue and red, respectively trained on unrestricted and base data. Plain lines track performance on unrestricted test data.

We observe that increasing sequence length from 2 to 8 sentences amplifies the lexical signal for both restricted and unrestricted triplet types. However, the effect is far more pronounced for unrestricted triplets. In contrast, the signal for restricted triplets,

while also increasing, is substantially weaker and begins plateauing after 6 sentences. While more text provides more stylistic evidence, the amount of unique author-specific evidence in topic-decoupled scenarios appears to saturate quickly as shown in Figure 8.

We also analyzed the inherent lexical signal within contrastive triplets using Jaccard similarity before scaling HALvest-Contrastive. Recall that we tested three different configurations before settling on our current version of HALvestContrastive. Queries denotes what we call anchors in the main paper. As shown in Figure 9, the lexical signal is substantially higher in the unrestricted triplets, where the query and positive can come from the same paper. For the 8-sentence split, the mean signal is 0.11 for unrestricted triplets but only 0.022 for the base configuration. This confirms that base HALvest-Contrastive successfully minimizes keyword overlap, forcing any successful model to rely primarily on signals beyond shared tokens.

### I Computational cost of late interaction and PLI

We derive the scoring cost of MaxSim for late interaction and for PLI with patch size n, then instantiate the expressions under our training configuration.

#### I.1 MaxSim cost per document pair

Let Q ∈ RSq×d and D ∈ RSd×d be the tokenembedding matrices of a query (anchor) and a document (candidate), where Sq,Sd are the respective valid token counts and d is the embedding dimension. Assuming ℓ2-normalised rows, MaxSim is computed as

MaxSim(Q,D) =

Sq

q⊤i dj . (7)

max

j∈[Sd]

i=1

The computation decomposes into three stages:

- 1. Pairwise dot products. The full similarity

matrix QD⊤ ∈ RSq×Sd requires Sq ·Sd inner products of dimension d, each costing 2d − 1 FLOPs (d multiplications and d−1 additions). The total is SqSd(2d − 1).

- 2. Row-wise maxima. Extracting maxj for each of the Sq rows requires Sq(Sd − 1) comparisons.
- 3. Summation. Summing the Sq maxima costs Sq − 1 additions.

Since d ≫ 1, the pairwise dot-product term dominates and we write the cost as

CMaxSim(Sq,Sd,d) ≈ 2Sq Sd d. (8)

For the symmetric case Sq = Sd = S common in our setup, this simplifies to 2S2d.

#### I.2 PLI: patching overhead and reduced scoring

With a fixed patch size n, each sequence of S tokens is compressed into P = ⌈S/n⌉ patch vectors by mean-pooling contiguous groups of n token embeddings.

Patching cost. For each of P patches, averaging n vectors of dimension d requires (n−1) · d additions and d scalar divisions, totalling approximately n · d FLOPs per patch and P · n · d = S · d FLOPs per sequence. For two sequences, the patching cost is 2Sd.

Reduced MaxSim cost. After patching, the similarity matrix is QpatchD⊤patch ∈ RPq×Pd, and the scoring cost becomes

CMaxSimPLI (S,d,n) ≈ 2

S n

2

d ≈

2S2d n2

. (9)

The total PLI cost per pair is therefore CMaxSimPLI +

2Sd. Since 2Sd ≪ 2S2d/n2 for all practical S and n ≤ S, the patching overhead is negligible and the scoring cost reduction factor is n2.

#### I.3 Scaling to a full training step

Under InfoNCE with full-gather, each anchor scores against every other document in the batch. With a batch size of B = 256, the positive is one of the B candidates and the remaining B − 1 serve as negatives. Every document in the batch acts as an anchor in turn, so the total number of MaxSim evaluations per training step is B2. Since each candidate’s embedding matrix is computed once by the encoder and reused across all anchors, the n2 speedup per-pair carries through unchanged: every one of the B2 pairs gets cheaper by a factor of n2, so the aggregate scoring cost drops by the same factor.

Table 15 instantiates the cost per-pair for our primary configuration (d = 768) at the maximum sequence length S = 512.

###### 2 sentences

Query-Positive Overlap

Query-Negative Overlap

Signal (QP Overlap - QN Overlap)

Noise (PN Overlap)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

| | | | | |Base dat Unrestric Base dat|a ted data m a mean: 0.|ean: 0.03 011|8| |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

Unrestricted data Base data Unrestricted data mean: 0.103 Base data mean: 0.076

Unrestricted data

Unrestricted data

Unrestricted data

12

17.5

16

Base data

Base data

12

Unrestricted data mean: 0.065

Unrestricted data mean: 0.065

14

15.0

10

Base data mean: 0.066

Base data mean: 0.066

10

12

12.5

8

8

10

Density

10.0

6

8

6

7.5

6

4

4

5.0

4

2

2

2.5

2

0

0

0

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Jaccard Similarity

0.00 0.05 0.10 0.15 0.20 0.25 Jaccard Similarity

0.2 0.0 0.2 0.4 0.6 0.8 1.0

0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 Jaccard Similarity

Jaccard Similarity

###### 4 sentences

Query-Positive Overlap

Query-Negative Overlap

Signal (QP Overlap - QN Overlap)

Noise (PN Overlap)

14

16

| | | | |Base|data| | | |
|---|---|---|---|---|---|---|---|---|
| | | | |Unre Base<br><br>|stricted data data mean:|mean: 0.13 0.088|6| |
| | | | | | | | | |
| | | | | | | | | |

20.0

Unrestricted data

Unrestricted data

Unrestricted data

Unrestricted data

20.0

Base data

Base data

Base data

14

12

17.5

Unrestricted data mean: 0.072

Unrestricted data mean: 0.065

Unrestricted data mean: 0.072

17.5

Base data mean: 0.073

Base data mean: 0.014

Base data mean: 0.073

12

15.0

10

15.0

10

12.5

12.5

8

Density

8

10.0

10.0

6

6

7.5

7.5

4

4

5.0

5.0

2

2

2.5

2.5

0

0.0

0

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Jaccard Similarity

0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 Jaccard Similarity

0.0 0.2 0.4 0.6 0.8 1.0 Jaccard Similarity

0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 Jaccard Similarity

###### 6 sentences

Query-Positive Overlap

Query-Negative Overlap

Signal (QP Overlap - QN Overlap)

Noise (PN Overlap)

14

14

| | | | |Base|data mean:|0.094| | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |

| | | |Unrestric Base dat<br><br>|ted data m a mean: 0.0|ean: 0.086 19| |
|---|---|---|---|---|---|---|
| | | | | | | |

Unrestricted data Base data Unrestricted data mean: 0.159

Unrestricted data

Unrestricted data Base data

Unrestricted data

20.0

Base data

Base data

20

Unrestricted data mean: 0.074

Unrestricted data mean: 0.074

12

12

17.5

Base data mean: 0.074

Base data mean: 0.074

10

10

15.0

15

12.5

8

8

Density

10.0

10

6

6

7.5

4

4

5.0

5

2

2

2.5

0

0

0

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Jaccard Similarity

0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 Jaccard Similarity

0.0 0.2 0.4 0.6 0.8 1.0 Jaccard Similarity

0.00 0.05 0.10 0.15 0.20 0.25 Jaccard Similarity

8 sentences

Query-Positive Overlap

Query-Negative Overlap

Signal (QP Overlap - QN Overlap)

Noise (PN Overlap)

25

| | |B|Unrestricted data ase data mean:|mean: 0.110 0.023| |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Unrestricted data

Unrestricted data

Unrestricted data Base data

Unrestricted data

14

Base data

Base data

Base data

12

Unrestricted data mean: 0.185

Unrestricted data mean: 0.075

Unrestricted data mean: 0.075

20

12

20

Base data mean: 0.099

Base data mean: 0.076

Base data mean: 0.076

10

10

15

15

8

Density

8

6

10

6

10

4

4

5

5

2

2

0

0

0

0

0.0 0.2 0.4 0.6 0.8 1.0 Jaccard Similarity

0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 Jaccard Similarity

0.0 0.2 0.4 0.6 0.8 1.0 Jaccard Similarity

0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 Jaccard Similarity

Figure 9: Signal and noise on every test split before scaling. We split 10,000 examples by word and compute Jaccard similarities. We define the signal as the difference between the query/positive and the query/negative overlaps whereas the noise is the Jaccard similarity between the positive and the negative of a triplet.

Model Patches/seq FLOPs/pair Speedup LI (token) 512 4.03 × 108 1×

- PLI n=2 256 1.01 × 108 4×
- PLI n=3 171 4.49 × 107 ∼9×
- PLI n=4 128 2.52 × 107 16×
- PLI n=5 103 1.63 × 107 ∼25×

Table 15: MaxSim scoring cost per document pair at S = 512, d = 768. The speedup column reports the ratio CMaxSimLI /CMaxSimPLI . The “∼” prefix reflects the ceiling effect in ⌈S/n⌉.

Storage. Each document requires S · d float values under LI and ⌈S/n⌉ · d under PLI, yielding a storage reduction factor of n (because storage scales with the number of vectors, not with pair-

wise interactions). At S = 512 and d = 768 in float16, a single document occupies 768KB under LI, 384KB under PLI n=2, and 154KB under PLI n=5.

Mixed sequence lengths. Training uses a mix of span lengths (Table 4), with mean token counts ranging from S¯ ≈ 70 (base-2) to S¯ ≈ 328 (base10). Since MaxSim cost is quadratic in S, longer spans dominate the compute. The n2 reduction factor applies uniformly regardless of S, so the relative advantage of PLI over LI is independent of the span-length distribution.

