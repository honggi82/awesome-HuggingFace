## zELO: ELO-inspired Training Method for Rerankers and Embedding Models

Nicholas Pipitone1 Ghita Houir Alami1 Advaith Avadhanam1 Anton Kaminskyi1 Ashley Khoo1

# arXiv:2509.12541v1[cs.AI]16Sep2025

### 1. Abstract

We introduce a novel training methodology named zELO, which optimizes retrieval performance via the analysis that ranking tasks are statically equivalent to a Thurstone model. Based on the zELO method, we use unsupervised data in order train a suite of state-of-the-art openweight reranker models: zerank-1 and zerank-1-small. These models achieve the highest retrieval scores in multiple domains, including finance, legal, code, and STEM, outperforming closed-source proprietary rerankers on both NDCG@10 and Recall. These models also demonstrate great versatility, maintaining their 0-shot performance on out-of-domain and private customer datasets. The training data included 112,000 queries and 100 documents per query, and was trained end-to-end from unannotated queries and documents in less than 10,000 H100-hours.

### 2. Summary of Contributions

#### 2.1. zELO: A novel Elo-based multi-stage training pipeline

We introduce a novel multi-stage training process inspired by Elo scoring systems. First, we generate candidate documents using a first-stage retriever (e.g. ZeroEntropy’s Search API, or Lucene + Embedding hybrid search). Then, we gather sparse pairwise preferences from an ensemble of large language models (and, for scale, a pairwise SLM distilled from the ensemble of LLMs). These pairwise preferences are converted into absolute relevance scores using the Thurstone statistical model. Finally, we fine-tune our pointwise rerankers on these query-document zELO scores.

#### 2.2. Open-source reranker models released

We release two fully open-weight rerankers trained on the zELO Method: zerank-1, initialized from Qwen3-4B (Yang et al. 2025), and zerank-1-small, initialized from Qwen3-1.7B (Yang et al. 2025). Rerankers are crossencoder models that take a query-document pair as input and output a relevance score between 0 and 1, significantly boosting the performance of first-stage search methods such as BM25, embedding search, and hybrid retrieval. Both models were trained on 112,000 query–corpus pairs, i.e. over 5 million query-zELO pairs, and their weights

are available on Huggingface. zerank-1-small in particular, is released under the permissible Apache 2.0 License. zerank-1 is open-weight and available for license by the ZeroEntropy team.

zerank-1: https://huggingface.co/ zeroentropy/zerank-1

zerank-1-small: https://huggingface.co/ zeroentropy/zerank-1-small

#### 2.3. State-of-the-art reranking accuracy across domains, languages, and retrieval methods

zerank-1 consistently outperforms commercial rerankers twice its size, with NDCG improvements of up to 5 points on benchmarks in finance, medicine, legal, code, and math. It also outperforms much larger LLMs-as-a-reranker, such as Gemini Flash 2.0, GPT-4o-mini, and GPT-5-mini/nano. It delivers strong gains regardless of the initial retrieval method, including BM25, embedding-based, and hybrid search.

#### 2.4. Unsupervised training and fine-tuning

Most importantly, at ZeroEntropy we found experimentally that ensembles of LLMs via zELO generate higher quality data than an equivalent number of human annotators on average. The zELO method has strong convergence properties. We scale ensemble inferences until the zELO score of the target document converges, which gives a strong indicator of fundamental query-document relevancy. The entire annotation method has been open sourced by ZeroEntropy as zbench (https://github.com/zeroentropyai/zbench). zELO can be used both for benchmarking internal private documents, and for generating domain-specific fine-tuning data.

Because zELO is fully automated, it can also be used for live production evaluations. While a given reranker is used in production, live query logs can be randomly sampled, annotated automatically via zELO, and can be used to easily discover and fix issues in a live production retrieval pipeline (For example: if necessary context was never ingested, or if the initial retriever couldn’t find it, etc). The annotations can also be used to fine-tune the reranker live, or if given per-customer context, can be used for personal-

ized recommendation systems.

### 3. Motivation

#### 3.1. Existing SOTA

Rerankers are the most fundamental operation in information retrieval. They take in a query document pair (q,d), and output a score s ∈ [0,1]. If it was feasible, every query into a search engine would simply have a reranker do a linear scan over the entire corpus. However, because this is computationally infeasible when the corpus is large, vectorbased approximators are used for selecting initial candidates. These include sparse lexical keyword search such as BM25 (Robertson et al. 1995), along with transformerbased embedding models. Keyword search fails if the user fails to recall the exact keyword, and dense embedding models can’t converge on true relevancy because of the limitations of N-dimensional space (Weller et al., 2025).

Given the fundamental importance of rerankers, we note that rerankers can always learn from an SFT distillation task. We have some ”teacher” reranker, traditionally human-annotated binary relevance labels, and then a ”student” reranker, often an ML model, we can train the student on the teacher annotations.

The current SOTA method for training rerankers involves InfoNCE triplet loss (q,d+,d−). Humans (the teacher) are tasked with annotated the positive, while the sampling method for negatives remains a free parameter. The most basic negative sampling strategy is in-batch random sampling (E.g. CLIP). However, the signal from random sampling is extremely weak, given that the sampled d− is usually obviously irrelevant; this method thus requires extremely large batch size (> 100k), which in turn forces the use of poor quality web-scale data. The SOTA training method is ”hard negative mining”, wherein there is an attempt to make d− as relevant as possible to maximize the signal from contrastive learning – often via an ensemble of embedding models followed by an ensemble of rerankers (Even LLM-as-a-reranker).

#### 3.2. Laffer Curve: The Fundamental Constraint on Existing SOTA Hard Negative Mining

Experimentally, we found that by making the ”hard negative miner” as intelligent as possible, the Model eventually did not learn any better. In fact, we found that it got significantly worse. Manual inspection made the issue clear: The hard negatives were on average legitimately more relevant than the human-annotated positive. This is unavoidable, as humans cannot exhaustively scan an entire corpus, and SOTA methods such as LLM-ensemble rerankers can reason on a much larger knowledge base than even expert annotators – and do so at scale.

[Figure 1]

Figure 1. The proposed Laffer curve between hard negative miner intelligence and student model performance. We did not collect sufficient data with miner intelligence as the only independent variable, so we leave such data collection to future research.

While one could human annotate (q,d−,d+) to confirm d− as a true negative vis a vis the positive, this is inherently a pairwise comparison. For a pointwise model, absolute scoring via InfoNCE requires in-batch negatives, which requires an unsupervised negative sampling strategy. Thus, the intractability of false negatives in hard negative mining remains.

We conclude that hard negative mining techniques create a Laffer curve with respect to hard negative quality: As ensemble-generated hard negatives approach and exceed the quality of human-positive annotations, the marginal benefit from the distillation process diminishes and eventually becomes negative, indicating a fundamental limitation in the hard negative mining methodology.

From here, we make the following argument: the highest possible pointwise reranker performance is not that which corresponds to the optimal point on this induced Laffer curve. ”Hard negative mining” is fundamentally flawed and its resulting accuracy is fundamentally capped by the training algorithm itself. This was the motivating reason for using pairwise annotations generated by ensembles of frontier models and their resulting ELO scores as our source of ground truth, as well as for the development of a novel training method which has no theoretical limitations on student model performance other than the accuracy of the teacher model itself.

### 4. The zELO Method

#### 4.1. Definitions:

Define Q to be the space of all queries, and D to be the space of all documents.

In this formalism, a pointwise reranker function Rpoint is a function

Rpoint : Q × D → [0,1] ⊂ R Such that, for a given query q ∈ Q and corpus C = {d1,...,dn} ⊂ D of documents, if i1,...,in is the relevance ranking of documents, we have:

Rpoint(q,di

) > Rpoint(q,di

) > ... > Rpoint(q,di

)

1

2

n

Additionally, we may define a pairwise reranker to be a function

Rpair : Q × D × D → [0,1] Such that:

pij := Rpair(q,di,dj) constitutes the probability that, w.r.t some query q ∈ Q, document di ∈ D is preferred over document dj ∈ D. Or, equivalently, such that Rpair(q,di,dj) − 12 is the strength by which di is more relevant than dj (consider Rpair(q,di,di) = 12 indicates no preference between two identical documents). Ideally, Rpair induces a total order on D for every q, but this is not strictly required.

#### 4.2. Using Rpair to generate SFT

As discussed in Section 3.1, for any given query, rerankers generally operate on a subset of the top k documents Z ⊂ D (where |Z| = k) given by some initial retrieval method (BM-25, Embeddings, Hybrid, etc.) Typical values for k are k ≤ 100. Furthermore, given that pointwise rerankers constitute the models actually used for IR tasks, we first train the pairwise Rpair with the intention of using this to create a SFT dataset upon which to train Rpoint.

Specifically, for a fixed query q with k many associated retrieved documents, inference Rpair on pairs of documents di,dj ∈ Z given subset Z ⊂ C to obtain scores pi,j. Overall, we form a dense preference matrix P that is k × k:





p11 p12 p13 ··· p1k p21 p22 p23 ··· p2k p31 p32 p33 ··· p3k

P =

 

 

... .

. . .

pk1 pk2 pk3 ··· pkk

Where the antisymmetry constraint pji = 1 − pij ensures that:

##### P + PT = 11T

Where 1 is the n-dimensional vector of ones, where diagonal elements then satisfy pii = 21.

Now define the sparse n × n matrix W as the extension of the preference matrix P to the full corpus C, where noninferred pairs are set to zero:

wij =

Pij = Rpair(q,di,dj) if (di,dj) was inferred 0 if (di,dj) was not inferred

More formally, let I ⊆ [n] × [n] denote the set of index pairs (i,j) for which pairwise comparisons were performed. Then:





w11 w12 w13 ··· w1n w21 w22 w23 ··· w2n w31 w32 w33 ··· w3n

W =

 

 

... .

. . .

wn1 wn2 wn3 ··· wnn

where each entry satisfies:

- 1. wij + wji = 1 if (i,j) ∈ I (inferred pairs maintain antisymmetry)
- 2. wij = wji = 0 if (i,j) ∈/ I (non-inferred pairs are zero)

The transformation from these pairwise preferences to pointwise scores follows the Bradley-Terry model framework. Given pairwise comparison matrix W, we seek to find latent ”abilities” (in the language of BT) or relevance scores (in that of IR) that explain the observed preferences.

4.2.1. CONNECTION TO BRADLEY-TERRY MODEL

The Bradley-Terry model assumes that for documents di and dj with latent abilities πi and πj, the probability that di is preferred over dj is:

πi πi + πj

P(di ≻ dj) =

In the Elo rating system formulation of Bradley-Terry, we parameterize πi = eElo

i where Eloi is the Elo rating of the document di, giving:

eElo

1 1 + e−(Eloi−Eloj)

i

P(di ≻ dj) =

=

eEloi + eEloj

= σ(Eloi − Eloj)

As such, we fit {Elo1,Elo2,...,Elon} such that whenever wij ̸= 0:

wij = σ(Eloi − Eloj)

And secondarily, for the purposes of normalization across queries, we constrain:

e1 + e2 + ... + en = 0 We define a negative log likelihood loss and fit Elos by gradient descent via maximum likelihood estimation. The loss is then L = −

pi pi + pj

wij log(1 + eElo

j−Eloi)

) =

wij log(

i,j

i,j

And it is a known result from Zermelo that this has a unique local minimum (given a final constraint like Elo1 + ... + Elon = 0) (Zermelo 1929). As a result, gradient descent will converge should we have a time-decaying learning rate ηt with t≥1 ηt diverging (we use ηt = t−0.125 here).

- 4.2.2. THURSTONE MODEL

The Thurstone Model posits that rankings are determined by per-document hidden scores (Which we call ELO), and that the probability that document i is better than document j is

wij =

1 + erf(Eloi − Eloj) 2

This is almost identical to Bradley-Terry, albeit assuming that a document’s intrinsic noise takes a normal distribution rather than a gumbel distribution. We observe that this adjustment makes a better fit to the observed data, and is also well-justified (Justified via the central limit theorem, with the prior that the document comparison is subject to multiple sources of noise). In our final training run, we utilize Thurstone’s erf() rather than Bradley-Terry’s σ() for this reason.

- 4.2.3. EXTENSION TO PLACKETT-LUCE FOR COMPLETE RANKINGS

While Bradley-Terry handles pairwise comparisons, the Plackett-Luce model generalizes this to complete rankings. For a ranking π of documents d1,...,dn, the Plackett-Luce probability is:

n

ππ(i)

P(π) =

n j=i ππ(j)

i=1

In our context, this allows us to model the probability of observing a complete ranking based on our Elo scores. However, since we work with sparse pairwise comparisons for computational tractability rather than complete rankings (more on this in section 5), we primarily consider the Bradley-Terry formulation.

4.2.4. SPARSE MATRIX SUBSAMPLING FOR ELOS

A dense n × n inference procedure for every query q in the dataset is prohibitively expensive given the scale of n. Therefore, we sparsely infer Rpair(q,di,dj) for selected O(n) di,dj pairs in a manner that the predicted Elos e′1,e′2,...,e′n from the Elo calculation algorithm on the sparse matrix match the actual Elos e1,...,en as well as possible.

Let G = (V,E) be our comparison graph where V = {d1,...,dn} and (di,dj) ∈ E iff Rpair(q,di,dj) is inferred. Let distG(di,dj) denote the shortest-path distance between vertices di and dj in G, and let deg(di) denote the degree of vertex di.

To ensure accurate Elo score estimation from sparse pairwise comparisons, our graph G must satisfy three key structural properties:

#### 1) Connected Graph:

- d2

- d3 d4

d5

- d2

- d3 d4

d5

d1

d1

Connected

Disconnected

- Figure 2. Left: Disconnected graph cannot establish relative Elo relationships between any of {d1, d2, d3} and any of {d4, d5}. Right: Connected graph enables global Elo ranking.

2) No Nodes With Low Degree:

d1 d2

- d3

- d4

d5

Low degree nodes

d1 d2

- d3

- d4

d5

Higher degree nodes

- Figure 3. Left: Node d1 has degree 1, making its Elo estimate unreliable. Right: All nodes have degree ≥ 3, providing more stable estimates.

#### 3) Low Diameter:

- d4

- d5 High diameter

- d4

- d5 Low diameter

d1 d2 d3

d1 d2 d3

Figure 4. Left: Documents d1 and d5 are separated by 3 edges in this pendant structure, leading to high uncertainty in their relative Elo. Right: Maximum separation is 2 edges, providing more reliable comparisons.

In particular, we employ the following heuristics to guide our sparse subsampling strategy:

- 1. Path Length Heuristic: The variance in estimated Elo differences is proportional to graph distance:

Var[e′i − e′j] ∝ distG(di,dj)

- 2. Degree Stability Heuristic: The variance of individual Elo estimates is inversely proportional to node degree:

1 deg(di)

Var[e′i] ∝

These heuristics motivate our graph construction constraints:

diam(G) = max

distG(di,dj) ≤ ρ min i

i,j

deg(di) ≥ δ

Under these heuristics, the probability of ranking error between documents di and dj increases with their graph distance and decreases with the true Elo gap |ei − ej|.

Now, with O(n) edges, the combination of these two constraints naturally lends itself to considering k-regular graphs. Consider that it is a known theoretical result that for a random k-regular graph G:

actually better to choose k2 random n-cycles and union their edge sets over the vertices; the resulting graph is k-

connected (and if no (di,dj) are duplicated across cycles, the graph is in fact k-regular as well). What is more, such a graph can be generated easily in O(n) time by taking random permutations. This graph will have N = kn2 edges. Below we illustrate a toy example with this method.

Constructing a k-regular graph via cycle splicing (example: n = 6, k = 4)

Step 1: Generate k/2 = 2 Random Cycles

d1

d1

- d5

- d6

- d2

- d3 d4

- d5

- d6

- d2

- d3 d4

- Cycle 1: 1 → 2 → 3 → 4 → 5 → 6 → 1
- Cycle 2: 1 → 5 → 2 → 6 → 3 → 4 → 1

- Figure 5. Generate two random cycles over the same vertex set.

Step 2: Overlay the Cycles d1

- d2

- d3 d4

- d5

- d6

Degree of each node (k = 4): deg(d1) = 4 deg(d2) = 4 deg(d3) = 4 deg(d4) = 4 deg(d5) = 4 deg(d6) = 4

- Figure 6. Construction of a 4-regular graph by overlaying 2 random cycles. Total edges: N = kn2 = 42·6 = 12.

5 2

diam(G) ≤ logk−1(n)+logk−1(log(n))+logk−1(

k(k−1))

With probability asymptotically 1, as per Bollob´as.1

Generating arbitrary random k-regular graphs is not efficient, however, and to ensure good connectivity, it is

1(Actually, 52 may be replaced with 2 + ϵ for any ϵ > 0) (Bollob´as 2001, Chapter 10.3)

#### Properties of the resulting graph d1

- d2

- d3

- d5

- d6

d4

Low diam: dist(d1,d3) = 2, diam(G) = 2 High connectivity: Graph is 2-connected Uniform degree: All nodes have degree 4

Figure 7. The resulting 4-regular graph has low diameter, high connectivity, and uniform degree distribution - ideal properties for sparse ELO estimation.

- 4.3. Training Rpoint from SFT

Given a dataset of query-document pairs with relevance scores {(qi,dj,yij)} where yij ∈ [0,1] represents the relevance of document dj to query qi, we train the pointwise reranker Rpoint using supervised fine-tuning.

We minimize the mean squared error loss:

LSFT =

1 |Dtrain|

(q,d,y)∈Dtrain

(Rpoint(q,d) − y)2

where Dtrain is our training dataset.

Little needs to be said here; our procedure does not deviate meaningfully from standard practice other than that our {yij} are given by the estimated {e′j} for each fixed query qi from the prior section as opposed to human-annotated binary scores.

- 5. Training the Pairwise Reranker

- 5.1. Dataset

As input to the zELO algorithm, we must collect Q,D,Z. The Q,D used by our final model consists of 112,000 publicly available queries across a wide array of domains including finance, law, medicine, code, and STEM; and > 100M publicly available web-scale documents. For generating Z, the initial retrieval method we use is cosine similarity on embeddings from Qwen3-Embedding-4B (Zhang et al., 2025), combined via RRF with a lexical sparse BM25. For lexical sparse embeddings, we use an optimized multilingual tokenizer pipeline (language detection, followed by language-specific stemming). The

top-k chosen is k = 100. Initial retrieval is done via the ZeroEntropy Search Engine, using the endpoint /queries/top-documents.

#### 5.2. Ensemble Annotation as Source of Truth

Consider a population P of annotators. We define the ensemble ranking as

1 |P| p∈P p

PRpair(q,di,dj) =

Rpair(q,di,dj)

We create a small internal dataset across numerous verticals (Medicine, Law, Finance, Code, etc) to use as the Gold Standard. For each query, the documents are selected via randomly sampling two documents out of the Top k by initial retrieval. k is randomly sampled between 10 and 100 with an inverse distribution. In order to resolve Rensemble for a particular input, we sample annotators until the standard error of the mean approaches 0.1.

As per Section 3.2 Laffer Curve, we observe the difficulties that accompany using human annotations as the source of ground truth. While sampling until convergence across all conceivable documents D for an entire querying population Puniversal against all conceivable queries Q would (definitionally) match the target ideal reranker, human annotation is in practice expensive, noisy, and observed to be simply inferior to an ensemble of SOTA large language models. As such, we preferentially inference the latter to generate pairwise annotations.

However, as discussed later in Section 5.6 RLHF, we do for each query make use of our dataset’s highest ranked human-annotated document in a second training run of our pairwise reranker. Thus, we recapture signal that pure distillation from an LLM-ensemble teacher model was unable to impart on a first pass. In this way, we make use of those human annotations with the highest signal-to-noise ratio without fully shackling ourselves to them as ground truth.

#### 5.3. Ensemble of LLMs as Synthetic Data Generators

Practically, consider P to be a set of frontier LLMs, each prompted against a given (q,di,dj) to output a chain-ofthought justification for and final judgment of document preference, and consider PRpair as before. For our purposes, we found |P| = 3 to be a good mix of economical, convergent, and accurate.

For each query q, we randomly sample a document pair {di,dj} ⊂ Z and prompt each LLM to score the relative preference on [−1,1],2 where:

2In Section 4.1 Definitions for the zELO Method we give the

- 1. −1 indicates strong preference for di
- 2. 1 indicates strong preference for dj
- 3. 0 indicates no preference

We then clamp the raw scores to {−1,0,1} and individually prompt-engineer each model to achieve roughly uniform response distribution. See the appendix for the base prompt used.

Next, we average over P to get an ensemble score, pij (a real number in [−1,1] with |P|r ∈ Z). We then map the ensemble score into the expected [0,1] range by the following update rule:

1 − pij 2

newpij =

That is, if the first document is preferred, the resulting score is near 1, while the second document being preferred results in a score near 0. This ensemble score is 2|P|1 of an integer.

In contradistinction to pij :=P Rpair(q,di,dj), the ensemble score, denote p′ij := Rpair′ (q,di,dj) the score given by the pairwise reranker being trained to approximate our ground truth. We may simply use standard Binary Cross Entropy loss to train Rpair′ (q,di,dj) on ensemble scores:

BCE(pij,p′ij)

##### L =

i,j sampled over q

Where, remember: BCE(p,q) := −(plog(q) + (1 − p)log(1 − q))

Given that inferencing PRpair is particularly expensive, we do so only once for each query, on a random (di,dj) pair taken from the top-k initially-retrieved documents. Inferencing more pairs would in theory improve the performance of the student model, but given economic constraints, our limitation proved the number of high quality queries we managed to train over – more entropy was to be had in inferencing over a new query than over an extra document pair for the same query.

We can see the convergence of this approach in Figure 8 above. This graph is from an early research run which was not used to train our final model, but which demonstrates the generalizability of the approach across model sizes and datasets. Training three differently sized Llama distillates on BioASQ using this method and evaluating it against a validation subset of that dataset (with LLM-ensemble as

codomain of Rpair as [0, 1]. Experimentally, prompting LLM’s for a [-1,1] codomain gave more uniform results. We in any case transform this back into the expected codomain later. We would also flip the order of prompted documents half the time and take the negative result to control for left/right LLM bias.

source of ground truth, not human-annotations) showcases steady performance improvement and rapid convergence. In particular, we note that even for the 8B and 70B models with strong zero-shot performance, margins of 20% and 10% were noted, resulting in large performance gains vs SOTA rerankers (93% vs 75% to 81% on this dataset). We further note that these models performed similarly on our internal standards of private data, implying generalization, whereas competing SOTA rerankers (admittedly pointwise) experienced a drop-off in performance, implying overfitting to evaluation datasets.

#### 5.4. Elo Score Calculation

Having trained the pairwise reranker Rpair′ and given a topk = 100 filtered corpus Z of k documents, we can now

infer it on some set I of pairs of documents (di,dj) : i,j ∈ I ⊂ [k] × [k] for computational efficiency. We again swap the order i,j half the time to mitigate the effects of potential biasing towards either position, and obtain scores pij, and set pji = 1 − pij. We thus form the sparse matrix P that is k × k.

This still leaves the choice of which I to infer Rpair′ on to generate P; this choice is important, as it determines the

ELO estimates on which Rpoint′ will be trained. Recall from the section on heuristics for subsampling sparse matrices for ELO scores that we desire a few crucial characteristics of our sampling strategy:

- 1. It must result in a connected graph, where every document is reachable from every other document by some chain of inferred pairwise comparisons.
- 2. No document should be pairwise-inferred a very small number of times, lest its ELO prove unstable and erroneous. In particular, we determined that all documents should be inferred the same number of times.
- 3. The maximum separation between two documents in the shortest-connecting chain of pairwise inferences (id est, the diameter of the graph) should be low, otherwise relative ELO between them may prove unreliable (since pairwise document relevancy cannot be assumed to be strongly stochastically transitive).

At ZeroEntropy, we explored various sampling methods to satisfy these constraints: simple random pairwise sampling without replacement, complete bipartite graphs Kl,k−l (to get diameter of 2) and approximately kl edges, as well as a dynamic method that selects successive (di,dj) pairs based on the smallest current ELO difference so as to maximize expected entropy. We evaluated these approaches by comparing ground-truth ELOs from densely inferred k × k matrices against ELOs generated from sparse sampling according to each method’s graph structure.

[Figure 2]

Figure 8. Frequency of the pairwise comparison models favoring documents on which the ensemble reaches consensus

[Figure 3]

Figure 9. MSE between the predicted Elos and actual Elos, versus the number of ”inferences” made to the matrix, for four sampling methods; random cycles, random pairs, the bipartite graph, and another method that calculates Elos live and favors picking i, j with low current Elo absolute difference

In the end, we discovered that a method of sampling k2 random cycles (where k is the desired valence of any doc-

ument in a k-regular graph) and unioning the edge sets proved fastest in converging and generated the most stable results. See the illustration of this process from Section

- 4.2.3 for more information. Results for these training procedure experiments are seen in Figure 9 above. The convergence properties of the ELO ranking system

are such that even with mere random sampling, the crossentropy loss of ELO estimates from subsampling vs those calculated from the full 100 × 100 matrix go to zero with only about a 1% sampling ratio:

[Figure 4]

Figure 10. Plots for a few different runs of random sampling; convergence observed around 1.2k/10k total pairs sampled

However, we were able to further improve upon this with the method of cycles, ultimately using only N = 400 inferences (0.4% of full), with k = 8 (4 random cycles) to sample the Elos. These ELO scores become our pointwise estimates on which we train Rpoint′ in the next section.

#### 5.5. Training the Pointwise Reranker

We now perform supervised fine-tuning on a Qwen-4B to obtain zerank-1 as well as a smaller Qwen-1.7B model to obtain zerank-1-small, using the same relevance scores generated from preceding steps for each. We use a standard MSE loss function to fit Rpred (our model) to Rpoint;

1 |B|

(Rpred(q,d) − Rpoint(q,d))2 .

L =

(q,d)∈B

The result is our first-training-pass pointwise reranker Rpoint′ (q,d).

#### 5.6. Improving the Pairwise Reranker (RLHF)

We improve on the pointwise model’s performance through a variant of negative mining: we add additional data to the original (ensemble-inferenced) NQ pairwise scores used to train the pairwise reranker, based on failures of the pointwise reranker.

Specifically, for each query q in our overall dataset, we let dhuman denote the document with the highest humanannotated score. Let rhuman be the rank of this document when we rank each document of q’s corpus C using Rpoint′ . For every sub-dataset constituent of our training dataset (id est, each constituent HuggingFace dataset), we manually set acceptable threshold rank values t. We determine this through factors such as how many documents are human-annotated for each query, how many human annotators were sampled for a given document-query pair, etc. If rhuman > t, we consider this a failure. Consider now the document d′ that was ranked at position rhuman − 1 by the reranker.

We will now inference the pairwise ensemble on (dhuman,d′); we observe typically that the ensemble strongly prefers the human top-scored document, though evidently we ranked it worse. We add this PRpair(q,dhuman,d′) to our pairwise reranker training dataset, while retaining the sole original example for that query, PRpair(q,di,dj), given by random sampling.

We then retrain the pairwise reranker on the expanded dataset, and repeat all previous steps to obtain Elo scores with which to train a second iteration of our pointwise reranker on. This gives us our final models, zerank-1 and zerank-small.

Specifically, we test zerank-1 and zerank-1-small against Cohere’s rerank-v3.5, Voyage AI’s rerank-2, and Salesforce’s Llama-rank-v1 on a suite of public datasets with standard NDCG@10 metric (Table 2). We also evaluate said models across smaller private customer datasets to test for generalization and overfitting on eval datasets (Table 3). Notably, neither zerank model was trained on any portion of the datasets evaluated; in-fact, as recalled, the zELO training method does not use any human annotations.

As can be seen, both zerank-1 and zerank-1-small offer large margins of improvement over existing SOTA methods, and these margins improve when tested on private datasets, indicating high generalization and wide applicability.

Furthermore, zerank-1-small maintains much of the superior performance of zerank-1 over existing models despite being less than half the size. zerank-1-small is also available under a fully open Apache 2.0 license.

Our evaluations use an initial retrieval with k = 100, and we test OpenAI-text-embedding-3-small, BM25, and hybrid search. On all three initial retrieval methods, we find that our rerankers still significantly improve the NDCG@10 compared to initial retrieval (see Figure 12).

Moreover, we achieve this performance while not compromising on speed:

|Model<br><br>|NDCG@10<br><br>|Latency (12 KB)<br><br>|Latency (150 KB)|
|---|---|---|---|
|Jina m0<br><br>|0.7279|547.14 ± 66.84 ms|2,543.8 ± 2,984.9 ms|
|Cohere 3.5<br><br>|0.7091|171.5 ± 106.8 ms|459.2 ± 87.9 ms|
|zerank-1<br><br>|0.7683|149.7 ± 53.1 ms|314.4 ± 94.6 ms|

Table 1. Zerank-1 offers improvement on SOTA NDCG@10 recall while being as fast or faster than competing models

Lastly, we compare the use of zerank-1 against simply prompting a cheaper frontier LLM, Gemini-v2.5-flash (Gemini Team 2025), to perform pairwise comparisons using the same prompt as that used to generate pairwise reranker training data. We use the pairwise comparisons to generate Elo scores using Section 5.3’s methods to obtain Gemini rankings (which we call gemini-reranker). We observe that our trained reranker still results in dramatically superior performance:

### 6. Results

We conduct evaluations across a variety of specialized domains, and zerank-1 consistently outperforms other stateof-the-art rerankers by significant margins. Our evaluation pipeline is open source and publicly available on github. Detailed results are on the subsequent page.

[Figure 5]

### 7. Conclusion

This technical report introduces zerank-1 and zerank-1small, the new state-of-the-art models for reranking and information retrieval tasks. These rerankers were trained via a novel pipeline that largely rejects pointwise human annotations, and instead focuses on mathematically modeling query-document relevance scores using Elo-inspired scoring based on more meaningful pairwise comparisons. These models exhibit great cross-domain versatility, exhibiting strong performance on fields from code to finance to medicine, and we offer the weights of both models for non-commercial uses.

[Figure 6]

Figure 11. Caption

| |Task/Benchmark| |Default(embedding) cohere-rerank-v3.5 Salesforce/Llama-rank-v1 zerank-1-small zerank-1| |
|---|---|---|---|---|

| |Code| |0.678 0.724 0.694 0.730 0.754| |
|---|---|---|---|---|
| |Conversational| |0.250 0.571 0.484 0.556 0.596| |
| |Finance| |0.839 0.824 0.828 0.861 0.894| |
| |Legal| |0.703 0.804 0.767 0.817 0.821| |
| |Medical| |0.619 0.750 0.719 0.773 0.796| |
| |STEM| |0.401 0.510 0.595 0.680 0.694| |

- Table 2. Performance comparison across different tasks and benchmarks

| |Task| |Cohere/rerank-v3.5 Salesforce/Llama-rank-v1 VoyageAI/rerank-2 zerank-1-small zerank-1| |
|---|---|---|---|---|

| |Legal| |0.718 0.766 0.746 0.799 0.854| |
|---|---|---|---|---|
| |Enterprise Search| |0.674 0.629 0.735 0.765 0.799| |
| |Conversational| |0.727 0.653 0.727 0.747 0.787| |
| |Healthcare| |0.706 0.756 0.749 0.885 0.898| |

- Table 3. Performance comparison across different tasks and benchmarks

[Figure 7]

[Figure 8]

###### Figure 12. The upper graph displays NDCG@10 of BM25 alone, compared to using zerank-1 and zerank-1-small on the top 100 results of BM25. The lower graph is the same but for hybrid search instead of BM25.

### 8. References

- 1. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., ... Zhang, Z. (2025, May 14). Qwen3 technical report (arXiv:2505.09388) [Preprint]. arXiv. https://doi.org/10. 48550/arXiv.2505.09388
- 2. Robertson, S. E., Walker, S., Jones, S., HancockBeaulieu, M. M., & Gatford, M. (1995). Okapi at TREC-3. In D. K. Harman (Ed.), Proceedings of the Third Text REtrieval Conference (TREC-3) (pp. 109–126). National Institute of Standards and Technology (NIST). https://trec.nist.gov/ pubs/trec3/papers/city.ps.gz
- 3. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 758–759). ACM.

https://plg.uwaterloo.ca/˜gvcormac/ cormacksigir09-rrf.pdf

- 4. Nogueira, R., & Cho, K. (2020). Passage re-ranking with BERT (arXiv:1901.04085v5) [Preprint]. arXiv. https://doi.org/10.48550/arXiv. 1901.04085
- 5. Together AI. (2024, June 6). Introducing Together Rerank API and Salesforce LlamaRank: Advancing reranking in retrieval. Together AI Blog. https://www.together.ai/blog/ together-rerank-api-and-salesforce-llamarank
- 6. Sean Lee, R., Huang, R., Shakir, A., & Lipp, J. (2025, March 13). Baked-in brilliance: Reranking meets RL with mxbai-rerank-v2. Mixedbread. https://www.mixedbread.com/blog/ mxbai-rerank-v2
- 7. Nogueira, R., Yang, W., Cho, K., & Lin, J. (2019, October 31). Multi-stage document ranking with BERT (arXiv:1910.14424) [Preprint]. arXiv. https:// doi.org/10.48550/arXiv.1910.14424
- 8. OpenAI. (2024, January 25). New embedding models and API updates: text-embedding-3-small & text-embedding-3-large. OpenAI. https://openai.com/index/ new-embedding-models-and-api-updates
- 9. Elo, A. E. (1978). The rating of chessplayers, past and present (Chapter 1). New York: Arco Publishing.
- 10. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3–4), 324–345. https://doi.org/10.2307/2334029
- 11. Zermelo, E. (1929). Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung. Mathematische Zeitschrift, 29(1), 436–460. https: //doi.org/10.1007/BF01180541
- 12. Bollob´as, B. (2001). Random graphs (2nd ed.). Cambridge University Press.
- 13. Gemini Team. (2025, June 17). Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Google DeepMind. https://storage.googleapis.com/ deepmind-media/gemini/gemini_v2_5_ report.pdf

### 9. Appendix

9.1. Ensemble Inference Prompts 9.1.1. SINGLE-QUERY PAIRWISE SCORES Task

You are a relevance scoring system. Given a query and two documents (A and B), your job is to decide which document is more relevant to the given query. You should think carefully, considering the pros and cons between each document. For your first few sentences, consider the pros and cons of Document A. Then, spend some time thinking about Document B. Then, at the end, compare, and make a decision as to which one is more relevant. Do NOT make a decision in the beginning of your thoughts, stay openminded until the last 1-2 sentences of your thoughts.

Scoring

The score should range from -1.0 to 1.0, where negative means document A is more relevant, and positive means Document B is more relevant. You can pick any number from -1.0 to 1.0.

Note that here, whether di or dj is document A is randomly chosen to mitigate any preference the model may have for either document position. We negate the score in that situation.

