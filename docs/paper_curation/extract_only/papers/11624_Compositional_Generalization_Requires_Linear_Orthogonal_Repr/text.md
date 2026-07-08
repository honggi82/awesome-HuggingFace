###### Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models

Arnas Uselis1 Andrea Dittadi23 Seong Joon Oh1

## arXiv:2602.24264v1[cs.CV]27Feb2026

###### Abstract

Compositional generalization, the ability to recognize familiar parts in novel contexts, is a defining property of intelligent systems. Although modern models are trained on massive datasets, they still cover only a tiny fraction of the combinatorial space of possible inputs, raising the question of what structure representations must have to support generalization to unseen combinations. We formalize three desiderata for compositional generalization under standard training (divisibility, transferability, stability) and show they impose necessary geometric constraints: representations must decompose linearly into per-concept components, and these components must be orthogonal across concepts. This provides theoretical grounding for the Linear Representation Hypothesis: the linear structure widely observed in neural representations is a necessary consequence of compositional generalization. We further derive dimension bounds linking the number of composable concepts to the embedding geometry. Empirically, we evaluate these predictions across modern vision models (CLIP, SigLIP, DINO) and find that representations exhibit partial linear factorization with low-rank, near-orthogonal perconcept factors, and that the degree of this structure correlates with compositional generalization on unseen combinations. As models continue to scale, these conditions predict the representational geometry they may converge to. Code is available at https://github.com/oshapio/ necessary-compositionality.

###### 1. Introduction

Modern vision systems are trained on a tiny, biased subset of a combinatorial space of visual concepts, like objects, attributes, relations in different contexts. Despite this, we

1Tübingen AI Center, University of Tübingen 2Helmholtz Munich 3Technical University of Munich. Correspondence to: Arnas Uselis <arnas.uselis@uni-tuebingen.de>.

Data space X

A cat on a person A person on a cat

|[Figure 1]|
|---|

|[Figure 2]|
|---|

Xtrain

X \ Xtrain

A person? A cat?

| |
|---|
| |
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |
| |

f

###### f

Figure 1. What enables compositional generalization in visionlanguage embedding models? Training data contain common configurations (left: a cat on a person) but lack rare ones (right: a person on a cat). Yet the same text-based queries, e.g. “A photo of a person”, must work on both, even when the latter was never seen during training. We investigate what properties encoder f must satisfy for such transfer to succeed.

expect them to perform well in the wild on novel recombinations of familiar concepts, an expectation tied to the view that systematic generalization, the ability to recombine learned constituents, is a hallmark of intelligence (Fodor & Pylyshyn, 1988). Yet a large body of empirical work shows that even high-performing neural models often struggle with systematicity when train/test combinations mismatch (Lake & Baroni, 2018; Keysers et al., 2020; Hupkes et al., 2022; Uselis et al., 2025). At the same time, large vision-language models such as CLIP (Radford et al., 2021) and its variants are trained on web-scale datasets (e.g., LAION-400M (Schuhmann et al., 2021a)) and achieve impressive zeroshot transfer on many tasks (Radford et al., 2021; Zhai et al., 2022). However, these systems often fail when test images contain unusual combinations of familiar concepts (Xu et al., 2022; Bao et al., 2024; Thrush et al., 2022; Abbasi et al., 2024; Yuksekgonul et al., 2023; Ma et al., 2023). Fig. 1 illustrates this tension for CLIP-like architectures: an image encoder f produces embeddings on which linear classifiers (e.g. from the text encoder) classify concepts, but the training data Xtrain cover only common compositions (such as a cat on a person) from the full data space X, while models must answer queries like “Is there a person present?” correctly even on rare compositions (such as a cartoon of a person on a cat) from X \ Xtrain. Given how rarely, if at all, such compositions appear in training, we ask: assuming that compositional generalization succeeds, what properties must the representations have to accommodate it?

We argue for non-negotiable, model-agnostic properties that

[Figure 3]

###### Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models

(Mahajan et al., 2025) (Schott et al., 2022)

(Uselis et al., 2025)

(Kempf et al., 2025)

[Figure 4]

[Figure 5]

[Figure 6]

ExampleRule⊂CTR

- Concept1

Concept2

- Concept2

Concept2

Concept2

Concept1

Concept3

Concept3

Concept 1

Concept 1

Train Test

|supp(TCij)| = 2 (i ∈ [2], j ∈ [5])

|T| = 3 |T| = 0.7|C|

T ∈ {TCG-low, TCG-high}

Figure 2. Interpreting previous works’ sampling designs T and validity rules R. Each panel shows a concept space as a grid (2D or 3D), with blue cells denoting training combinations and red cells denoting held-out test combinations.

any neural-network-based system claiming compositional generalization must satisfy. We state three desiderata: divisibility, transferability, and stability. These desiderata formalize that (i) all parts of an input should be accessible to a simple readout; (ii) readouts trained on a tiny but diverse subset should transfer to unseen combinations; and (iii) training on any valid subset should yield robust generalization. Our scope is the common setting where predictions are linear in the embedding f: CLIP-style zero-shot classifiers, linear probing, and cases where a fixed non-linear head is folded into the encoder.

Our key finding is that these desiderata necessitate a specific geometry: linear factorization with near-orthogonal concept directions. This establishes what any model must achieve to compositionally generalize under standard training, providing a concrete target for future design. Moreover, it offers theoretical grounding for the Linear Representation Hypothesis (Mikolov et al., 2013; Elhage et al., 2022) – the linear structure widely observed in neural representations is a necessary consequence of compositional generalization.

Our contributions are: (1) Defining desiderata. We define three desiderata: divisibility, transferability, stability, and formalize compositional generalization in their terms. (2) Structural necessity. Under gradient descent with the crossentropy loss, these desiderata imply linear factorization: embeddings decompose into per-concept sums with orthogonal difference directions. (3) Empirical grounding. Across CLIP and SigLIP families, we find strong evidence of factorization, near-orthogonality, low-rank per-concept geometry, and correlation with compositional generalization accuracy, suggesting that the current state-of-the-art vision models are converging to the necessary geometric properties.

###### 2. Related work

Compositional generalization. Prior work establishes sufficient conditions for compositional generalization under specific assumptions on data-generating processes or rep-

resentations (Wiedemer et al., 2023; Mahajan et al., 2025; Uselis et al., 2025). In contrast, we study necessary conditions: we ask what must be true of a model’s embeddings if it transfers from a restricted subset to the full space under our desiderata.

Geometry of learned representations. Empirical studies document linear subspaces in VLMs (Trager et al., 2023), the Linear Representation Hypothesis in LLMs (Park et al., 2023), and near-orthogonal feature encodings (Elhage et al., 2022). In parallel to our work, (Fel et al., 2025) map task-relevant concept families in DINOv2 and propose a Minkowski-sum hypothesis for token space, highlighting that strong geometric structure may arise from architecture as well as from downstream demands. In contrast, we show that under practice-driven desiderata and standard training, linearity and orthogonality are necessary, not merely observed.

Disentangled and object-centric representations. This line of work proposes desiderata and training schemes, with mixed evidence linking these to compositional generalization (Eastwood & Williams, 2018; Montero et al., 2021; Dittadi et al., 2022). We ask the complementary question: if a model does generalize compositionally, what must its embeddings satisfy? See Section B for a detailed discussion.

###### 3. Setup: A Framework for compositionality

We begin by detailing key desiderata for embedding models that contend to be compositionally generalizing. We motivate them from a practical perspective: (1) models need to support distinguishing between any combination of concepts, (2) practical data collection is limited to a subset of the concept space, so a model needs to be able to transfer from a subset of the concept space to the full concept space, and (3) in practice apriori it is not known which subset needs to be chosen, so a model should be able to transfer robustly from any subset, matching in probability distribution to retraining over any other dataset.

###### 3.1. Setup: concept spaces and data collection process

We interpret the world as a product of concepts: any input xc ∈ X (e.g., an image) has an associated tuple of concepts c ∈ C, describing its constituent parts and properties. This is a reasonable way to describe a large portion of the world. For example, current large-scale datasets (e.g., image-caption pairs) provide noisy natural-language descriptions that can be decomposed into discrete concept values. Clearly, a single concept tuple cannot capture all aspects of the world, e.g. how attributes bind to objects or how different objects relate spatially. Still, an intelligent system should at least be able to tell apart basic concepts (such as objects and their attributes), even without modeling their relations. In other words, concept spaces may not capture the full compositional structure of the world, but any model of the world must involve them in some form. Importantly, we do not assume how the concept values are distributed (e.g. being independent), only what they represent.

Definition 1 (Concept space). Suppose we have k concepts, where concept i takes ni possible values. Let Ci = [ni] for i ∈ [k]. The concept space is the Cartesian product

C = C1 × C2 × ··· × Ck. (1)

In most of the paper we use ni = n for all i, so C = [n]k and |C| = nk. We index inputs by concept tuples: for each c ∈ C we assume an associated xc ∈ X (e.g., a natural image) realizing c.

Data-related components for compositional generalization involve three notions: (1) the total variation of the data, (2) the concepts we aim to learn and expect the model to capture, and (3) the data that is actually collectible. We capture (1) by the concept space C (Definition 1). For (2), the targets that we aim to capture can be described by a label function l : C → V ⊆ C that captures which concepts and their values we want to learn.1 In this work we take the full target V = C, reflecting that foundation models attempt to align with all present concepts. For (3), we formalize collectability constraints through a validity class that specifies which training supports are valid, indicating which concept combinations may appear in training–taking into account cases that many combinations are not collectible (e.g. “a pink cat on the moon” isn’t common). We formalize this below.

Considering data collection. We are interested in models that support efficient compositional generalization from a subset of the concept space. To formalize this notion, we specify a validity class T ⊆ 2C of valid training sets, where 2C denotes the power set of C, and a validity rule R : 2C →

1More generally, the world may involve additional factors beyond the concepts represented in C. Our framework does not require C to be exhaustive: C can be viewed as the subset of concepts we model explicitly, with any remaining variation treated as nuisance.

∃h : h(f(xc)) correct ∀c ∈ C

Compositional models

∀T ∈ T , A(T) transfers

h(·) linear

Linearly compositional

Generalizing compositional

Generalizing linearly compositional h(·) linear; A(T) transfers.

Figure 3. Relationship between (generalizing) compositional models. Divisibility (orange) and transferability (blue) requirements.

{0,1} that specifies whether a given training set is valid. This setup captures the natural question of which training sets we use and for which we expect generalization.

Definition 2 (Training support, validity class, and training dataset). Let C be the concept space. A training support is any subset T ⊆ C. Validity class is a collection T ⊆ 2C whose members are called valid training sets. The class T specifies which training sets are observable. Validity class T is specified by a validity rule R : 2C → {0,1} through T = {T ⊆ C : R(T) = 1}. A training dataset for a training set T is DT = {(xc,c) : c ∈ T}.

We note that there are many validity rules used in practice. For example, if we can collect any subset of size N < |C|, then R(T) = 1 whenever |T| = N. Fig. 2 illustrates common choices: Mahajan et al. (2025) use training supports that cover every concept value; Schott et al. (2022) use random samples covering 70% of all combinations; Kempf et al. (2025) specify a small set of allowed supports; and Uselis et al. (2025) use supports whose joint marginals cover at least two values per concept. Note that these validity rules apply to concept supports rather than individual datapoints.

###### 3.2. Compositional representations and models

Given the concept space and the training supports, we now make precise how we expect models to learn. We work with encoders f that map an input to a vector representation (embedding).

Scope of models. We study embedding models: these cover modern foundation models like CLIP and SigLIP (Tschannen et al., 2025; Zhai et al., 2023), supervised-learning models, self-supervised models like DINO (Caron et al., 2021; Siméoni et al., 2025). At inference the models we study are non-contextual: the representation of an input depends only on that input (no dependence on other test examples, prompts, or the batch). Formally, the encoder is a map f : X → Z, with z = f(x) (optionally ℓ2-normalized).

Readout class (linear vs. non-linear). Usually, encoders f are associated with either a downstream or readout model h that takes z = f(x) and outputs per-concept logits h(z) ∈ Rk×n using argmax classification rule (see Definition 3).

This covers zero-shot use of text features as linear classifiers, standard linear probing, and the affine last layer in most neural classifiers. If h is non-linear in a neural network, we absorb the layers preceding the linear layer g into the encoder (f˜ = g ◦ f) and analyze the resulting affine layer. The definition below keeps the readout h general to allow future extensions beyond linear heads, but all results in this paper consider the linear case, without such restrictions, a high-capacity readout could make any injective encoder appear compositional by memorization.

Definition 3 ((Linearly) compositional model). An encoder

- f : X → Z is compositional w.r.t. C if there exists h : Z → Rk×n such that, for all c ∈ C and all i ∈ [k],

h(f(xc))i,j . (2)

ci = argmax

j∈[n]

It is linearly compositional if h can be taken affine h(z) = Wz + b. We refer to h as the readout.

###### 3.3. Compositional generalization and desiderata

Given the ingredients (concept space C, encoder f, and training-support family T ), we now define a learning rule A and state three desiderata for compositional generalization: divisibility, transferability, and stability. We emphasize that these desiderata are on the NN-based models that exhibit generalization, as defined below, not on the representations, as studied in disentangled representation learning.

Considering training. We view a learning algorithm as a map

A : DT  → hT, hT ∈ H ⊆ {h : Z → Rk×n},

from a dataset supported on T ⊆ C to a readout in a chosen hypothesis class. In practice, A is typically (stochastic) gradient descent on a cross-entropy or contrastive objective, covering contrastive vision-language encoders (e.g., CLIP, SigLIP), standard supervised classifiers, and linear probes on self-supervised vision encoders like DINO.

Desiderata for compositional generalization. Suppose we train a downstream readout hT = A(DT) on some T ∈ T . What should hT satisfy? We argue for three practicallymotivated properties.

First, every combination of concept values should be classifiable by the readout: for any c ∈ C, the corresponding region of the representation space of f is nonempty: there exists at least one z that hT assigns the concept values c. Otherwise, generalization to the full grid is impossible. We refer to this property as Divisibility.

Desideratum 1 (Divisibility). A compositional model (Definition 3) can only exist if the readout has sufficient capacity to represent every concept combination. That is, for a readout h : Z → Rk×n, every concept tuple must have a

nonempty decision region:

k

Ri,ci(h) ̸= ∅,

∀ c ∈ C :

(3)

i=1

where Rij(h) = {z ∈ Z : arg max j′∈[n]

h(z)i,j′ = j}.

Divisibility is necessary but not sufficient: it guarantees that the space is divisible, but does not imply that the readout will be correct. We therefore ask that, for every training set, the learned readout transfers to the full grid; we refer to this as Transferability.

- Desideratum 2 (Transferability). For every T ∈ T , the

trained readout hT = A(DT) correctly classifies all possible combinations of the concept space:

∀ c ∈ C, ∀ i ∈ [k] : argmax

j∈[n]

hT f(xc) i,j = ci. (4)

Note that Transferability implies Divisibility. We state Divisibility explicitly because it highlights a capacity requirement: the embedding space must be able to represent all concept combinations.

Third, consider readouts learned from different valid supports T ∈ T . Divisibility and Transferability do not say anything about the behavior of the classification decisions. Intuitively: if an input depicts a “cat”, retraining on another valid support should not flip the preference to “dog” or push the prediction toward near-indifference. We refer to this as Stability.

- Desideratum 3 (Stability). For any T,T′ ∈ T , any point x ∈ X, and any i ∈ [k], the per-concept posteriors agree across supports:

exp(hT(f(x))i,j)

p(iT)(j | f(x)) =

,

n k=1 exp(hT(f(x))i,k)

′)

p(iT)(· | f(x)) = p(T

i (· | f(x)).

(5)

We view Stability as an idealization: once the training support is sufficiently informative, retraining on any other valid support should not change the model’s per-concept predictions (and ideally its calibrated posteriors). In practice, this may only hold approximately; relaxing Stability to allow small distributional deviations across supports is a natural direction for future work. We discuss the role of Stability and what can fail without it in Section G.

Defining compositional generalization. We now tie the ingredients into a single tuple Π = (f,H,A,T ), which we use as the object that specifies the entire compositionalgeneralization setup: the encoder, the readout class, the learning rule, and the family of valid training supports. We specify compositional generalization as a process of learning readouts that generalize over all T ∈ T and satisfy Desiderata 1–3.

Definition 4 (Compositional generalization). Π = (f,H,A,T ) exhibits compositional generalization if, for

###### C2

C1 {w1j}nj=1

x11

... Sub. type

|[Figure 7]|
|---|

g

“A c1”

c11 = Cat c1n = Person

fimg(x11) . . .

f

Figure 4. Instantiating the framework with CLIP-like embedding models for analysis.

every T ∈ T with hT = A(DT), Divisibility (Desideratum 1) and Transferability (Desideratum 2) hold on the full grid, and the posteriors are Stable across valid retrainings (Desideratum 3) for all pairs T,T′ ∈ T . We say that Π exhibits linear compositional generalization when the readout hypothesis class is linear.

We illustrate the relationship between (linear) models and their compositional counterparts in Fig. 3. In practice one could consider relaxed or average-case variants; however, we here are interested in “ideal” representations that support compositional generalization under any data sample.

###### 3.4. Instantiating the framework with CLIP

We instantiate the framework in the dual-encoder, visionlanguage setting in the style of CLIP models: images and texts are embedded into a shared space and trained to align, with captions acting as noisy descriptions of concept tuples.

Encoders. Let f : X → Z be the image encoder and

- g : Y → Z the text encoder. At inference both are typically

ℓ2-normalized so that inner products are cosine similarities: ∥f(x)∥ = ∥g(y)∥ = 1.

Prompts as linear probes. Zero-shot classification uses text features as linear classifiers. For each concept i ∈ [k] and value j ∈ [n], we can choose a prompt pi,j (e.g., “a photo of a cat”) and define a probe vector wi,j := g(pi,j) ∈ Z. Stacking these gives a readout

h(z) = w⊺i,jz i,j ∈ Rk×n.

Here f is the representation model, while h is a linear readout whose weights come from the text encoder. Training in CLIP-like models can be viewed as learning a readout model where the same set of text-derived probes serves across many images; prompts often mention only parts of an image, so the system is implicitly asked to recognize objects and attributes regardless of which other concepts co-occur. We illustrate this process in Fig. 4; for a high-level schematic, see Fig. 1 in the introduction.

The question we study. Given a concept space C, what structure must z = f(xc) have so that a single set of probes {wi,j} (whether fixed by g or learned as linear probes) satisfies our desiderata (Desiderata 1–3) on the full C? In other words, what constraints does zero-shot, probe-based classi-

fication place on the geometry of image representations in order to support compositional generalization?

###### 4. Implications of compositionality on representations

We now ask what our desiderata imply for representations in common training regimes, both as necessary constraints and as sufficient conditions. Three questions guide the section:

- Q1 (§4.1) Geometry under GD with CE/BCE and stable transfer. If A is gradient descent under binary crossentropy, and Π exhibits compositional generalization (Def. 4) across a family of supports T , what structure is necessary for f (and the linear readout h)? → We show additive (linear) factorization with orthogonal concept directions under natural T .
- Q2 (§4.2) When are these geometric conditions sufficient? In the same binary GD+CE regime, do linear factorization and cross-concept orthogonality also suffice for compositional generalization? → Yes, establishing that the geometry is both necessary and sufficient.
- Q3 (§4.3) Minimal dimension of linearly compositional models. Assuming divisibility holds (Desideratum 1) and a linear (affine) readout h, what is the smallest d so that correct per-concept predictions are possible over all nk tuples? → With affine readouts, d ≥ k is necessary.

###### 4.1. Geometry of f under common training settings

We instantiate A as gradient descent on the binary crossentropy (logistic) loss. As in Section 3.4, the readout h is linear in the embedding zc = f(xc) (using either textencoder-derived probes or learned linear heads). Under these assumptions, the representation space Z must exhibit both linearity and cross-concept orthogonality. This conclusion holds under at least two validity regimes: (1) when more than half of all possible combinations are observed, |T| = 2k−1 + 1, and (2) when only a small, carefully chosen set of datapoints is observed, |T| = 1 + k.

Unstable Does not transfer

Stable Transfers

###### w1 w1 w1 w1

Figure 5. Stable and unstable examples of feature representations. The top panel shows an unstable configuration, where depending on the sample, the readout either does not transfer or unstably. Bottom panel shows a stable configuration.

Proposition 1 (Compositional generalization implies linear factorization). Let Π = (f,H,A,T ) be the tuple instantiated in Definition 4, with linear heads H and A given by GD+CE. Suppose that the training sets follow random sampling with validity rule R(T) = 1 if |T| = 2k−1 + 1. Assume Desiderata 1–3 are satisfied. Then, under the binary grid Ci = {0,1} with Z = {zc : c ∈ [2]k} ⊂ Rd, there exist {ui,0,ui,1 ∈ Rd}ki=1 such that for every c ∈ [2]k the following holds:

- 1. (Linearity) zc = ki=1 ui,c

i

.

- 2. (Cross-concept orthogonality) (ui,1 −ui,0) ⊥ (uj,1 − uj,0) for all i,j ∈ [k] with (i ̸= j).

Proof sketch. GD+CE converges to a max-margin SVM (Soudry et al., 2024). Stability implies consistent weight differences; max-margin with varying training sets makes each point a support vector, yielding prediction invariance when other concepts change. Max-margin geometry then implies flipping a concept produces an additive shift. Because of the SVM solution, weights for each concept are proportional to the segment connecting positive and negative classes, from which follows orthogonality. Minimal dataset setting can be argued similarly by carefully constructing datasets with size 1 + c to contain only pairs of datapoints differing only in one concept value.

Intuitively, linear factorization means that a combination space of nk elements can be explained using only n · k factors. The orthogonality condition says that factors of concept values belonging to different concepts (e.g., “red” and “square”) are orthogonal to each other, but no requirement is placed on the factors of concept values belonging to the same concept (e.g., “red” and “blue”). We illustrate the stable and unstable examples of feature representations in Fig. 5. Additionally, we note that linear factorization in itself is not trivial: the fact that nk datapoints can be explained using n · k factors does not have to hold for any linearly compositional model. We illustrate this with examples in Section H.4.

The datapoint requirement can be interpreted as operating in either (i) a minimal-learning regime for extrapolating to the whole grid (as in Compositional Risk Minimization framework (Mahajan et al., 2025)), where |T| = 1+k(n−1) suffices to extrapolate to the whole grid, or (ii) a large-sample regime in which random sampling yields near-complete coverage of the concept space.

Empirical evaluation on synthetic data. The necessary conditions discussed so far concern models that support compositional generalization. Neural networks trained from scratch, even on large-scale datasets, may not exhibit this structure. Prior work provides evidence that this structure can emerge under standard classification losses, without ex-

plicit pressure to generalize compositionally (Uselis et al., 2025), but those results are limited to neural networks in a two-concept setting. We therefore conduct synthetic experiments to evaluate to what extent linearity (measured by (7)) and orthogonality emerge under standard classification losses, and find positive evidence, especially as the number of concepts increases (see Figs. 38 and 39, and setup in Section I.6). Finally, in Section F.2, we analyze an idealized, but practically desired, setting where a model’s logits are exactly equal to α for the matches, and β otherwise, i.e. (h(z) ∈ {α,β}), and show that this implies strong dimensionality requirements, as well as additive representations (up to projection) that exactly preserve all probe scores (Proposition 7).

Takeaway §4.1. Under gradient descent with cross-entropy loss, compositional generalization with stable transfer requires linear factorization and orthogonality of cross-concept factors.

###### 4.2. Sufficiency of linear and cross-concept orthogonal f

In the same regime as Proposition 1, the converse holds: if embeddings are linearly factorized with cross-concept orthogonality, then GD+CE trained on sufficiently informative supports yields transferability and stability on the full concept space (Proposition 4 and section D.1). This closes the loop for that regime: the geometry is both necessary and sufficient.

Beyond the binary orthogonal case, sufficiency can be achieved, though not necessarily with SGD: if factors are recoverable, one can reconstruct the representations (Section D.2) and construct classifiers that generalize, provided the original representation space admits a linearly compositional model (i.e., satisfies Divisibility). This extends Uselis et al. (2025) to the multi-concept setting. We summarize the sufficiency results in Proposition 2.

Proposition 2 (Sufficiency summary (informal)). Under linear readouts, two sufficiency statements hold.

- 1. Binary necessity–sufficiency case. Under the binary grid with GD+CE, if embeddings decompose as zc =

i ui,c

i

with cross-concept orthogonality, then any training set with |T| = 2k−1 + 1 (or a cross dataset of size 1 + k) suffices: the learned readout recovers every concept value on the full grid (transferability) and is invariant across valid T (stability). Combined with Proposition 1, the geometry is both necessary and sufficient (Proposition 4).

- 2. General constructive case. In the multi-valued case, recoverable linear factors are sufficient to construct concept readouts in principle (Section D.2).

We believe that only the first case is of practical interest, since it assumes standard GD+CE training; together with the necessary conditions, it reinforces that linear, cross-concept

###### k = 2,n = 20 k = 3,n = 12

[Figure 8]

[Figure 9]

- Figure 6. Example geometries of linearly compositional models. Left: 2 concepts (n = 20 each) on a sphere. Each colored stripe is the argmax boundary for one concept value; their intersections yield 202 combination cells. Right: 3 concepts (n = 12 each) in

- 3D. Colored planes show argmax boundaries; their intersections carve out 123 combination cells. Each boundary is colored according to the concept it belongs to.

orthogonal structure is a plausible target that current vision(language) systems should exhibit if they are to generalize compositionally.

Takeaway §4.2. In the binary GD+CE regime, linear factorization with cross-concept orthogonality is both necessary and sufficient for compositional generalization.

- 4.3. Packing and minimum dimension

So far we have established necessary and sufficient conditions for the representation space Z to exhibit compositional generalization. However, it is not clear what exactly the capacity constraints are on the representation space to support it. Here, we ask a basic capacity question: what is the minimum embedding dimension d needed to support Divisibility (Desideratum 1), i.e. realize all possible nk combinations? The following result gives a tight lower bound. Proof and its sketch in Section E.

Proposition 3 (Minimum dimension for linear probes). For k concepts, each with n values, suppose there exist linear probes that correctly classify each concept value for all nk combinations from embeddings f(x) ∈ Rd. Then necessarily d ≥ k.

Importantly, the bound is independent of the number of values n per concept, depending only on the number of concepts k. This holds regardless of whether each factor is discrete or continuous, since the proof requires only that any two values per factor be distinguishable. We illustrate two examples of divisibility in Fig. 6: on a sphere and in Euclidean space, though our formal results establish minimal dimensionality only for Euclidean space. Additional visualizations in Fig. 21. As the number of concepts k increases while the embedding dimension d is fixed, divisibility imposes increasingly tight packing constraints: factors within each concept must lie in progressively lower-dimensional subspaces, approaching near-collinearity in the limit (an example in Fig. 16 in Appendix).

Empirical results and the dependence on representation

space and the loss function. The result in Proposition 3 is a geometric lower bound and does not depend on a specific loss or representation space. Empirically, we find that CE in Euclidean space reaches this bound most closely, BCE typically requires higher dimension (approximately 2k), and spherical geometry adds roughly one extra dimension (see the setting in Section I.6, and Fig. 40 for the empirical results). In Section F.2 we discuss an idealized case where a model’s logits are constrained to take two values, yet satisfy perfect classification, and show that this necessarily requires d ≥ 1 + k(n − 1) in Proposition 6.

Takeaway §4.3. The minimum embedding dimension scales with the number of concepts k, not the number of values n per concept (d ≥ k). When k grows relative to a fixed d, per-concept subspaces must become increasingly low-rank, approaching near-collinearity.

###### 5. Surveying necessary conditions in pretrained models

Previous section established necessary conditions for compositional generalization: representation space must be linearly factorized and the factors must be orthogonal across concepts (Proposition 1). In this section, we ask: how far are modern pretrained models from these necessary conditions?

Our theory is built on binary concept values, but some of the concepts in the datasets we consider are multi-valued. Instead of repeatedly sampling binary values and testing factorization on these, we adopt the natural multivalued extension of the necessary structure. Concretely, we test whether representations admit an approximate additive decomposition of the form

k

, (ui,a − ui,b) ⊥ (uj,a′ − uj,b′), (6)

zc ≈

ui,c

i

i=1

for all c ∈ [n]k, all i ̸= j, and all a,b,a′,b′ ∈ [n], i.e., an additive per-concept factorization with cross-concept orthogonality of difference directions. This form reduces to the binary case when n = 2.

Main qualitative result. We give intuition for both conditions in (6) using a pretrained DINOv3 model on dSprites (Fig. 7). The figure shows that the idealized linearity and orthogonality conditions are visible in the global PCA view (a), and that the same pattern is observed when some concepts are fixed and others are varied (b), consistent with approximate additive factorization and cross-concept orthogonality. Any datapoint can approximately be expressed as a sum of the factor vectors u (c), and they often take low-rank structure, as well as satisfy orthogonality (d). The remainder of this section quantifies these observations. For full qualitative results, see Section I.5.

[Figure 10]

###### Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models

(a) (b)

[Figure 11]

[Figure 12]

[Figure 13]

Square→Oval

Large→ Small

[Figure 14]

Bottom

[Figure 15]

[Figure 16]

→

Right

Top

Condition on y, and shape

→

|[Figure 17]|
|---|

Left

Heart

[Figure 18]

Blue

→

[Figure 19]

Square

→

Orange

[Figure 20]

### f( )

[Figure 21]

[Figure 22]

approximately decomposes

(c) Color Shape Size Orientation

X position Y position

#### + + + + +

Shape and Y position X position and Size X position and Y position Color and Size

(d)

- Figure 7. Summary of the linearity + orthogonality hypothesis, illustrated in DINOv3 on dSprites. (a) PCA projection of DINOv3 embeddings over all dSprites combinations. Changing shape or y-position produces near-constant direction shifts (linearity), and the two directions are nearly orthogonal. (b) After fixing shape to square and y to one value, PCA on the remaining subset shows that horizontal position x (left → right), size (large → small), and color (orange ↔ blue) each vary along near-constant directions and form a grid-like structure. (c) Embeddings exhibit approximate linear factorization: each embedding decomposes as a sum of one factor per concept,

zc ≈ i ui,ci (illustrated for the highlighted sample, with arrows pointing to the selected factor in each panel). The recovered factors are typically low-rank (≤ 3D), so these 3D plots capture most of their structure. (d) For each pair of concepts, the left mini-panel shows the

two sets of factors, illustrating near-orthogonality across concepts. The right mini-panel shows all datapoints projected onto the span of those two factors; the projected points organize into a grid aligned with the factor directions, consistent with additive decomposition. The corresponding quantitative results are reported in Sections 5.1, 5.3 and 5.4; full qualitative results in Section I.5.

Concretely, we quantitatively evaluate the necessary conditions for compositional generalization in pretrained models. We aim to answer the following questions:

- Q4 (Section 5.1) Is linear factorization present in pretrained models?
- Q5 (Section 5.2) Does the degree of linear factorization correlate with compositional generalization?
- Q6 (Section 5.3) Are per-concept difference vectors approximately orthogonal across concepts, as the theory predicts?

Q7 (Section 5.4) What geometric structure do factors u

exhibit?

Models and datasets. We evaluate a broad model set spanning contrastive vision-language and self-supervised vision encoders: OpenAI CLIP (Radford et al., 2021), OpenCLIP, MetaCLIP (Xu et al., 2023), MetaCLIP2 (Chuang et al., 2025) checkpoints from the LAION ecosystem (Schuhmann et al., 2021a), SigLIP and SigLIP 2 (Zhai et al., 2023; Tschannen et al., 2025), and DINOv1–v3 (Caron et al., 2021; Siméoni et al., 2025). This covers different architectures,

PUG-Animal

DSprites

###### MPI3D

1.00

0.9

0.98

Accuracy

0.96

0.95

Random

Random

Random

0.8

0.94

0.51

0.42

0.63

0.90

0.42

0.12

0.17

0.50 0.55 0.60 0.65 R2

0.38 0.40 0.42 0.44 R2

0.44 0.46 0.48 0.50 0.52 R2

OpenAI CLIP OpenCLIP MetaCLIP MetaCLIP2 SigLIP SigLIP2 DINOv1 DINOv2 DINOv3

- Figure 8. Linearity in embeddings correlates with compositional generalization across VLMs and self-supervised vision models. Across three datasets, we plot compositional generalization accuracy against projected R2 (our linear-factorization score) for a broad set of pretrained encoders, including vision-language models (OpenAI CLIP, OpenCLIP, SigLIP, MetaCLIP, MetaCLIP2) and pure vision backbones (DINOv1–v3). Each marker is a model variant; higher projected R2 is consistently associated with higher compositional generalization performance.

training objectives (softmax vs. sigmoid), and scales. We evaluate on three compositional datasets: PUG-Animal (Bordes et al., 2023), dSprites (Matthey et al., 2017), and MPI3D (Gondal et al., 2019), plus ImageNet-AO (Abbasi et al., 2024) (Section I.4.2). The full checkpoint roster, dataset summary, and example samples are listed in Section H.5.

Recovering the factors from representations. Assuming that a linear factorization exists in the representations of a model f as detailed in Section 4.1, we can recover the factors {ui,j}i∈[k],j∈[n] by averaging over all the datapoints that share a particular concept value (Trager et al., 2023). For analysis purposes it is sufficient to recover the centered factors. That is, given all centered embeddings {f(xc)}c∈[n]k, the factors can be recovered as ui,j = |{c∈[n]1

k:ci=j}| c∈[n]k:ci=j f(xc).

###### 5.1. Linear factorization in pre-trained models

To assess the extent of linearity in the embeddings, we measure a whitened R2 score on the probe span. We (1) project onto the probe span to discard information the embeddings may possess beyond the dataset concepts, and (2) whiten the embedding space so that R2 is not inflated by a few dominant directions. Concretely, given the recovered approximate factors {ui,j}i∈[k],j∈[n], the R2 score is computed as

f(xc) − ki=1 ui,ci 22 xc∈D ∥f(xc) − f¯∥22

R2 = 1 − xc∈D

, (7)

where D is the dataset, and f¯is the mean embedding. Note that a score of 1.0 indicates perfect linearity. We provide intuition of linear factorization and its relation to the R2 in Section H.3, additional justification of whitening in Sec-

- tion H.2.

Results. Fig. 9 shows projected R2 scores across models and datasets. Among all datasets, each model’s R2 score is consistently above the random baseline (0.4-0.6 vs. 0.12-0.42,

PUG-Animal DSprites MPI3D

1.0

| | |
|---|---|
| |0.56 0.55 0.58<br><br>0.66<br><br>0.49 0.49 0.50 0.50|
| |0.42<br><br>0.13<br><br>0.41 0.40 0.43 0.43<br><br>0.17|

0.5

2R

0.0

RandomCLIPViT-L/14OCLIPViT-L/14SigLIP-L/16SigLIP2-S/ORandomCLIPViT-L/14OCLIPViT-L/14SigLIP-L/16SigLIP2-S/ORandomCLIPViT-L/14OCLIPViT-L/14SigLIP-L/16SigLIP2-S/O

Figure 9. Linear factorization partly explains current models’ embedding spaces. Bar plots of whitened R2 on three datasets with varying concept/value counts.

respectively). This suggests that embeddings are partially captured by a sum of per-concept components, while still leaving some information unexplained. Additionally, we observe that R2 scores are similar in scale across models. The same linearity trend holds in the zero-shot setting when using text encoders as probes on both PUG-Animal and ImageNet-AO (Section I.4 and Figs. 28 and 32).

Importantly, we note that the R2 scores, while consistently above random, are far from perfect, indicating that current models only partially satisfy the linear factorization predicted by our theory.

Takeaway §5.1. Embeddings exhibit partial linear factorization (R2 typically 0.4–0.6), explaining a moderate fraction of the variance via per-concept components. The gap from perfect scores highlights a divergence from the ideal embedding structure theory predicts.

5.2. Compositional generalization and linear factorization

We ask whether the degree of linear factorization predicts compositional generalization.

Metrics and setup. For each dataset/model, we train linear probes on 10% of all concept combinations and evaluate

(b) SigLIP2-S/O

CLIP ViT-L/14

Random

(a) DINO3-B/16

Same concept Other concepts

|[Figure 23]<br><br>0.24 0.05 0.27 0.08<br><br>0.05 0.17 0.09 0.07<br><br>0.27 0.09 0.86 0.18<br><br>0.08 0.07 0.18 0.49|
|---|

|[Figure 24]<br><br>0.30 0.04 0.33 0.07<br><br>0.04 0.15 0.06 0.05<br><br>0.33 0.06 0.70 0.10<br><br>0.07 0.05 0.10 0.50|
|---|

|[Figure 25]<br><br>0.46 0.49 0.59 0.39<br><br>0.49 0.62 0.65 0.55<br><br>0.59 0.65 0.81 0.50<br><br>0.39 0.55 0.50 0.49|
|---|

|[Figure 26]<br><br>0.29 0.06 0.34 0.09<br><br>0.06 0.14 0.09 0.07<br><br>0.34 0.09 0.83 0.19<br><br>0.09 0.07 0.19 0.50|
|---|

0.59

0.6

character

| |0.52 0.53 0.53 0.52 0.53 0.54 0.52|
|---|---|
| | |
| |0.32<br><br>0.12 0.11 0.12 0.12 0.10 0.09 0.12|

Cosinesimilarity

0.4

world

size

0.2

texture

0.0

RandomCLIPViT-L/14OCLIPViT-L/14SigLIP-L/16SigLIP2-S/ODINO2-B/14DINO3-B/16MetaCLIP-H/14

character world sizetexture

character world sizetexture

character world sizetexture

character world sizetexture

Figure 10. Pre-trained models exhibit strong within-concept direction similarity and partial orthogonality across concepts. (a) Aggregated within-concept direction similarity over datasets. (b) Pairwise average cosine across concepts. Lower values indicate greater orthogonality between factor vectors. Full orthogonality results are reported in Section I.2.

[Figure 27]

[Figure 28]

on the held-out 10% unseen compositions (cf. sampling discussion in Section 4.1). This corresponds to a validity rule R(T) = 1 if |T| = 0.1nk. We compute Projected R2 on whitened PW f(x) (Section 5.1) and pair it with a compositional accuracy score on the held-out compositions. We use a randomly-initialized OpenCLIP ViT-L/14 model as a baseline by training linear probes on the embeddings. We use linear probing rather than zero-shot classification to avoid prompt-specification issues, nonetheless, the same conclusions hold in zero-shot experiments on both PUGAnimal and ImageNet-AO (Section I.4).

[Figure 29]

Ycoord.

Ycoord.

[Figure 30]

Orientation

Size

Size

Figure 11. Geometry of factors {ui,j} in OpenCLIP ViT-L/14. The factors are often low dimensional and near co-linear within a concept. Across concepts, the factors are near-orthogonal.

Compositional accuracy is computed by training one linear classifier per concept, then averaging each classifier’s accuracy on the held-out combinations. For example, dSprites has 6 concepts (shape, orientation, x position, y position, size, and color); we train 6 classifiers and report their mean accuracy on unseen combinations.

test this prediction by testing orthogonality in two ways: (1) within-concept and (2) across-concept. We defer the details to Appendix I.2.

Results. Across all datasets higher Projected R2 coincides with higher compositional accuracy (Fig. 8). The randomly initialized OpenCLIP ViT-L/14 baseline consistently occupies the low-R2/low-accuracy corner, indicating the effect is not a dimensionality or scale artifact, and follows the rationale of sanity checks in interpretability (Méloux et al., 2025). The same trend is observed in zero-shot text-probe experiments on both PUG-Animal and ImageNet-AO (Sec-

Results. Pretrained encoders exhibit consistently higher direction similarity within concepts than across concepts (Fig. 10): within-concept similarity (a) is around ≈ 0.530.55, whereas cross-concept similarity (b) is ≈ 0.09-0.12. The randomly-initialized encoder also exhibits this pattern; however, the across-concept similarity is higher (0.32 on average) compared to pre-trained models. The same withinvs-across pattern is also observed in zero-shot text-probe experiments on both PUG-Animal and ImageNet-AO (Figs. 29 and 33).

- tion I.4 and Figs. 28 and 32). A full train/test-split ablation is reported in Section I.1 and Fig. 23. This is consistent with our theory: models whose embeddings are closer to a linear factorization also generalize better to unseen concept combinations.

Takeaway §5.3: Pretrained models exhibit partial cross-concept orthogonality, substantially more so than randomly initialized encoders, suggesting that training drives factor directions toward the geometry predicted by our theory.

Takeaway §5.2: Linear factorization in pre-trained models correlates positively with compositional generalization performance.

###### 5.4. Dimensionality of factors

###### 5.3. Orthogonality of factors

Our theory (Proposition 1) predicts that per-concept difference vectors should be orthogonal across concepts under linear factorization, but not necessarily within-concept in generalizing linearly compositional models. We empirically

Our theory predicts that generalizing linear compositional models require linear factorization of embeddings into perconcept components. When many concepts must coexist in a fixed embedding dimension, each concept’s subspace should be low-rank to enable efficient packing (Section 5.1). Here, we investigate to which extent concept factors in pretrained

###### (a)

###### (b) (c)

DSprites (DINO v3)

DSprites PUG-Animal MPI3D PUG-Animal

Dim./Num.Classes

Character

Size

World

Texture

Explainedvariance

1.0

| |2/3<br><br>6/10|
|---|---|
| |1/6<br><br>4/10<br><br>2/10<br><br>1/10|

| |2/3<br><br>8/10<br><br>44/64 2/3<br><br>5/6<br><br>3/4<br><br>1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10<br><br>22/69 1/3<br><br>4/104/10|

1.0

1.0

Cumulative

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |

| |95%| | |
|---|---|---|---|
| | | | |
| | | | |

0.5

0.5

0.5

0.0

0.0

100 101

100 101

1 2 3

1 2 3

0.0

Component

Component

Component

Component

shapecolororientationsize posxposy

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

SigLIP-L/16 SigLIP2-S/O

OCLIP ViT-L/14 CLIP ViT-L/14

Figure 12. Dimensionality of factors. (a) Normalized effective rank across datasets and concepts under OpenCLIP ViT-L/14; text above each bar reports “effective dimension / number of values” for that concept. (b) Same analysis as in (a), shown for DINOv3 on dSprites; the recovered factors are typically lower-rank (variance concentrates in fewer PCs). (c) Cumulative explained variance of recovered PUG-Animal factors across model families (CLIP, OpenCLIP, SigLIP, SigLIP2); for each concept, the curves are nearly overlapping, indicating strong cross-model similarity in factor geometry.

models are low-dimensional.

Metrics and setup. We study factor geometry (with factor recovery following Section 5.1). For each concept i ∈ [k] with value set Ci (ni = |Ci|), we aggregate the per-concept factors ui,j for j ∈ Ci into a matrix Ui ∈ Rn

i×d. We then analyze (1) the dimensionality of each concept and (2) how this dimensionality compares across models. To do so, we examine the spectrum of Ui (PCA on its rows) and report the number of principal components required to explain 95% of the variance across values j.

Results. Fig. 12 shows that most ordinal factors lie in lowdimensional subspaces relative to their cardinality (e.g., dSprites size 1/6, MPI3D vertical-axis 4/10). Panel (b) shows the same tendency in a DINOv3-specific view on dSprites: factor variance typically concentrates in a small number of principal components. Panel (c) shows that this structure is highly consistent across architectures: for each PUG-Animal concept, explained-variance curves from different model families are nearly indistinguishable. This cross-model similarity is consistent with the Platonic Representation Hypothesis (Huh et al., 2024) and with recent evidence that independently trained multimodal contrastive models can often be aligned by near-orthogonal transforms (Gupta et al., 2026). Across datasets and models, ≥ 95% of variance is typically captured by one or two PCs, indicating that spectra align closely by concept. Discrete concepts show higher rank, potentially due to being composed of more atomic attributes. Overall, semantic factors are lowrank and geometrically similar across models, while discrete concepts are not strictly low-rank.

We also visualize dSprites factors (orientation, size, yposition) in Fig. 11. Each subspace is effectively < 3D (≥ 95% variance in ≤ 2 PCs). Size and y-position trace near-1D path, while orientation forms a smooth 2D curve with small curvature, matching the effective dimensions in Fig. 12.

Takeaway §5.4: Ordinal and continuous factors are typically low-dimensional (≤ 4D), while discrete factors show higher rank, potentially because they encode multiple underlying attributes. Concept factor geometry is similar across different models.

###### 6. Discussion and conclusion

Compositional generalization remains a challenge for vision embedding models, even those trained at scale. In this work we pinned down what compositional generalization demands of a representation. We formalized compositional generalization through three practically motivated desiderata: divisibility (parts must be readable), transferability (readouts trained on a limited but valid subset must generalize), and stability (the learned solution must not depend on which subset of the data space the model was trained on). Under typical training settings, our main theoretical result shows that satisfying these desiderata forces embeddings to behave like a dictionary: representations decompose additively into per-concept factors, and to prevent interference, these factors must be orthogonal across concepts. Importantly, these geometric conditions are also sufficient. However, supporting such a structure is not free: k concepts require at least k dimensions, a bound that in practice may be higher depending on the loss function.

Empirically, across a broad set of modern VLMs and compositional datasets, we find that current representations partially satisfy the predicted structure: additive per-concept components explain a meaningful fraction of variance, crossconcept factors are partially orthogonal, and, notably, the degree to which a model matches the predicted geometry correlates with compositional generalization on unseen combinations. This correlation offers a practical diagnostic for assessing compositional capability directly in representation space. Our factor-rank analysis elucidates how models “pack” many concepts into fixed-dimension embeddings: ordinal and continuous factors tend to be low-dimensional, while discrete factors occupy higher effective rank. These observations hold not only for vision-language models but

also for self-supervised DINO encoders, suggesting the geometric constraints we identify are not an artifact of contrastive language supervision.

Taken together, our results establish that the linear structure widely observed in neural representations is a necessary consequence of compositional generalization. This reframes a large body of empirical findings into a single geometric characterization of what any compositionally generalizing model must achieve. At the same time, current models only partially satisfy these conditions, which may explain their failures on compositional benchmarks. However, as the models continue to scale, these conditions predict the representational geometry they may converge to.

Limitations and future work. Our theory emphasizes worst-case stability over valid training supports; relaxing this to average-case or approximate stability is a natural direction that may better match some practical settings. Characterizing how different training supports change the implied geometry could turn these necessary conditions into a practical design guide for data collection and model building. Our theory assumes a fixed encoder and considers retraining across different training supports; in practice the encoder is trained once on a single dataset; understanding the impact of this assumption on the necessary conditions is an interesting direction for future work.

###### 7. Acknowledgments

We would like to thank Divyat Mahajan for useful discussions and insights, as well as comments on an early version of this work. We also thank Alexander Rubinstein and Darina Koishigarina for insightful discussions, and Simon Buchholz for useful comments.

###### References

Abbasi, R., Rohban, M. H., and Baghshah, M. S. Deciphering the role of representation disentanglement: Investigating compositional generalization in clip models, 2024. URL https://arxiv.org/abs/2407.05897.

Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes. In ICLR 2017 Workshop, 2017.

Assouel, R., Campbell, D., and Webb, T. Visual symbolic mechanisms: Emergent symbol processing in vision language models, 2025. URL https://arxiv.org/ abs/2506.15871.

Aurenhammer, F. Power diagrams: Properties, algorithms and applications. SIAM Journal on Computing, 16(1): 78–96, 1987. doi: 10.1137/0216006. URL https:// doi.org/10.1137/0216006.

Bangachev, K., Bresler, G., Noman, I., and Polyanskiy, Y. Global minimizers of sigmoid contrastive loss, 2025. URL https://arxiv.org/abs/2509.18552.

Bao, W., Chen, L., Huang, H., and Kong, Y. Prompting language-informed distribution for compositional zeroshot learning, 2024. URL https://arxiv.org/ abs/2305.14428.

Bengio, Y. and LeCun, Y. Scaling learning algorithms towards AI. In Large Scale Kernel Machines. MIT Press, 2007.

Bengio, Y., Courville, A., and Vincent, P. Representation learning: A review and new perspectives, 2014. URL https://arxiv.org/abs/1206.5538.

Bennett, K. P. and Bredensteiner, E. J. Duality and geometry in svm classifiers. In Proceedings of the Seventeenth International Conference on Machine Learning, ICML ’00, pp. 57–64, San Francisco, CA, USA, 2000. Morgan Kaufmann Publishers Inc. ISBN 1558607072.

Bordes, F., Shekhar, S., Ibrahim, M., Bouchacourt, D., Vincent, P., and Morcos, A. S. Pug: Photorealistic and semantically controllable synthetic data for representation learning, 2023. URL https://arxiv.org/abs/ 2308.03977.

Bradley, A. Local mechanisms of compositional generalization in conditional diffusion, 2025. URL https: //arxiv.org/abs/2509.16447.

Brady, J., Schölkopf, B., Kipf, T., Buchholz, S., and Brendel, W. Generation is required for data-efficient perception, 2025. URL https://arxiv.org/abs/2512. 08854.

Campbell, D., Rane, S., Giallanza, T., et al. Understanding the limits of vision language models through the lens of the binding problem, 2025.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers, 2021. URL https: //arxiv.org/abs/2104.14294.

Chuang, Y.-S., Li, Y., Wang, D., Yeh, C.-F., Lyu, K., Raghavendra, R., Glass, J., Huang, L., Weston, J., Zettlemoyer, L., et al. Meta clip 2: A worldwide scaling recipe. arXiv preprint arXiv:2507.22062, 2025.

Ciernik, L., Linhardt, L., Morik, M., Dippel, J., Kornblith, S., and Muttenthaler, L. Objective drives the consistency of representational similarity across datasets, 2025. URL https://arxiv.org/abs/2411.05561.

Cortes, C. and Vapnik, V. Support vector networks. Machine Learning, 20:273–297, 1995.

Courellis, H. S., Minxha, J., Cardenas, A. R., et al. Abstract representations emerge in human hippocampal neurons during inference. Nature, 632(8026):841–849, 2024. doi: 10.1038/s41586-024-07799-x.

Crammer, K. and Singer, Y. On the algorithmic implementation of multiclass kernel-based vector machines. J. Mach. Learn. Res., 2:265–292, March 2002. ISSN 1532-4435.

Desai, K., Nickel, M., Rajpurohit, T., Johnson, J., and Vedantam, R. Hyperbolic image-text representations, 2024. URL https://arxiv.org/abs/2304.09172.

Dittadi, A., Träuble, F., Locatello, F., Wüthrich, M., Agrawal, V., Winther, O., Bauer, S., and Schölkopf, B. On the transfer of disentangled representations in realistic settings, 2021. URL https://arxiv.org/abs/ 2010.14407.

Dittadi, A., Papa, S., De Vita, M., Schölkopf, B., Winther, O., and Locatello, F. Generalization and robustness implications in object-centric learning. In Proceedings of the 39th International Conference on Machine Learning (ICML), 2022.

Eastwood, C. and Williams, C. K. I. A framework for the quantitative evaluation of disentangled representations. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum? id=By-7dz-AZ.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., and Olah, C. Toy models of superposition. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/ 2022/toy_model/index.html.

Engels, J., Michaud, E. J., Liao, I., Gurnee, W., and Tegmark, M. Not all language model features are one-dimensionally linear. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.

net/forum?id=d63a4AM4hb.

Fel, T., Wang, B., Lepori, M. A., Kowal, M., Lee, A., Balestriero, R., Joseph, S., Lubana, E. S., Konkle, T., Ba, D., and Wattenberg, M. Into the rabbit hull: From taskrelevant concepts in dino to minkowski geometry, 2025. URL https://arxiv.org/abs/2510.08638.

Feng, J., Russell, S., and Steinhardt, J. Monitoring latent world states in language models with propositional probes. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/ forum?id=0yvZm2AjUr.

Fodor, J. A. and Pylyshyn, Z. W. Connectionism and cognitive architecture: A critical analysis. Cognition, 28(1–2): 3–71, 1988.

Gondal, M. W., Wuthrich, M., Miladinovic, D., Locatello, F., Breidt, M., Volchkov, V., Akpo, J., Bachem, O., Schölkopf, B., and Bauer, S. On the transfer of inductive bias from simulation to the real world: a new disentanglement dataset. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.

neurips.cc/paper/2019/file/ d97d404b6119214e4a7018391195240a-Paper. pdf.

Goodfellow, I., Bengio, Y., Courville, A., and Bengio, Y. Deep learning, volume 1. MIT Press, 2016.

Greff, K., van Steenkiste, S., and Schmidhuber, J. On the binding problem in artificial neural networks, 2020. URL https://arxiv.org/abs/2012.05208.

Gupta, S., Kansal, S., Jegelka, S., Isola, P., and Garg, V. Canonicalizing multimodal contrastive representation learning, 2026. URL https://arxiv.org/abs/ 2602.17584.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. betaVAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, 2017. URL https:// openreview.net/forum?id=Sy2fzU9gl.

Higgins, I., Amos, D., Pfau, D., Racaniere, S., Matthey, L., Rezende, D., and Lerchner, A. Towards a definition of disentangled representations, 2018. URL https: //arxiv.org/abs/1812.02230.

Hinton, G. E., Osindero, S., and Teh, Y. W. A fast learning algorithm for deep belief nets. Neural Computation, 18: 1527–1554, 2006.

Huh, M., Cheung, B., Wang, T., and Isola, P. The platonic representation hypothesis, 2024. URL https: //arxiv.org/abs/2405.07987.

Hupkes, D., Dankers, V., Mul, M., and Bruni, E. Compositionality in neural networks: A survey and taxonomy. Journal of Artificial Intelligence Research, 73:673–728, 2022.

Jarvis, D., Klein, R., Rosman, B., and Saxe, A. M. On the specialization of neural modules, 2024. URL https: //arxiv.org/abs/2409.14981.

Jiang, Y., Rajendran, G., Ravikumar, P., Aragam, B., and Veitch, V. On the origins of linear representations in large language models, 2024. URL https://arxiv.org/ abs/2403.03867.

Kapl, F., Mamaghan, A. M. K., Seitzer, M., Johansson, K. H., Marr, C., Bauer, S., and Dittadi, A. Are objectcentric representations better at compositional generalization? arXiv preprint arXiv:2602.16689, 2026.

Kempf, E., Schrodi, S., Argus, M., and Brox, T. When and how does clip enable domain and compositional generalization?, 2025. URL https://arxiv.org/abs/ 2502.09507.

Keysers, D., Sch"arli, N., Scales, N., Buisman, H., Furrer, D., Kashubin, S., Staniszewski, G., Blevins, T., Zettlemoyer, L., and Petrov, S. Measuring compositional generalization: A comprehensive method on natural language semantics. In International Conference on Learning Representations (ICLR), 2020.

Kim, H. and Mnih, A. Disentangling by factorising. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2649–2658. PMLR, 10–15 Jul 2018. URL https:// proceedings.mlr.press/v80/kim18b.html.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes, 2014. URL https://arxiv.org/ abs/1312.6114.

Koishigarina, D., Uselis, A., and Oh, S. J. Clip behaves like a bag-of-words model cross-modally but not unimodally, 2025. URL https://arxiv.org/abs/ 2502.03566.

Lake, B. M. and Baroni, M. Generalization without systematicity: On the compositional skills of sequence-tosequence recurrent networks. In Proceedings of the 35th International Conference on Machine Learning (ICML), 2018.

Lake, B. M., Ullman, T. D., Tenenbaum, J. B., and Gershman, S. J. Building machines that learn and think like people. Behavioral and brain sciences, 40:e253, 2017.

Lee, C., Chang, J., and yong Sohn, J. Analysis of using sigmoid loss for contrastive learning, 2024. URL https: //arxiv.org/abs/2402.12613.

Lee, C., Lim, S., Lee, K., and yong Sohn, J. On the similarities of embeddings in contrastive learning, 2025. URL https://arxiv.org/abs/2506.09781.

Li, Q., Xiao, H., and Shen, L. Bce vs. ce in deep feature learning, 2025. URL https://arxiv.org/abs/ 2505.05813.

Liang, Q., Qian, D., Ziyin, L., and Fiete, I. Compositional generalization via forced rendering of disentangled latents, 2025. URL https://arxiv.org/ abs/2501.18797.

Lim, H., Choi, J., Choo, J., and Schneider, S. Sparse autoencoders reveal selective remapping of visual concepts during adaptation, 2025. URL https://arxiv.org/ abs/2412.05276.

Lippl, S. and Stachenfeld, K. When does compositional structure yield compositional generalization? a kernel theory, 2025. URL https://arxiv.org/abs/2405. 16391.

Locatello, F., Bauer, S., Lucic, M., Rätsch, G., Gelly, S., Schölkopf, B., and Bachem, O. Challenging common assumptions in the unsupervised learning of disentangled representations, 2019. URL https://arxiv.org/ abs/1811.12359.

Locatello, F., Poole, B., Rätsch, G., Schölkopf, B., Bachem, O., and Tschannen, M. Weakly-supervised disentanglement without compromises, 2020. URL https: //arxiv.org/abs/2002.02886.

Ma, Z., Hong, J., Gul, M. O., Gandhi, M., Gao, I., and Krishna, R. Crepe: Can vision-language foundation models reason compositionally?, 2023. URL https: //arxiv.org/abs/2212.07796.

Mahajan, D., Pezeshki, M., Arnal, C., Mitliagkas, I., Ahuja, K., and Vincent, P. Compositional risk minimization, 2025. URL https://arxiv.org/abs/2410. 06303.

Mamaghan, A. M. K., Papa, S., Johansson, K. H., Bauer, S., and Dittadi, A. Exploring the effectiveness of objectcentric representations in visual question answering: Comparative insights with foundation models, 2024. URL https://arxiv.org/abs/2407.15589.

Matthey, L., Higgins, I., Hassabis, D., and Lerchner, A. dsprites: Disentanglement testing sprites dataset. https://github.com/deepmind/dsprites-dataset/, 2017.

Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient estimation of word representations in vector space, 2013. URL https://arxiv.org/abs/1301.3781.

Montero, M. L., Ludwig, C. J., Costa, R. P., Malhotra, G., and Bowers, J. The role of disentanglement in generalisation. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=qbH974jKUVy.

Montero, M. L., Bowers, J. S., Costa, R. P., Ludwig, C. J. H., and Malhotra, G. Lost in latent space: Disentangled

models and the challenge of combinatorial generalisation, 2024. URL https://arxiv.org/abs/2204. 02283.

Méloux, M., Dirupo, G., Portet, F., and Peyrard, M. The dead salmons of ai interpretability, 2025. URL https: //arxiv.org/abs/2512.18792.

Nielsen, B. M. G., Marconato, E., Dittadi, A., and Gresele, L. When does closeness in distribution imply representational similarity? an identifiability perspective. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Pach, M., Karthik, S., Bouniot, Q., Belongie, S., and Akata, Z. Sparse autoencoders learn monosemantic features in vision-language models, 2025. URL https://arxiv. org/abs/2504.02821.

Pal, A., van Spengler, M., di Melendugno, G. M. D., Flaborea, A., Galasso, F., and Mettes, P. Compositional entailment learning for hyperbolic vision-language models, 2024. URL https://arxiv.org/abs/2410.

06912.

Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models, 2023. URL https://arxiv.org/abs/2311. 03658.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., and Sutskever, I. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.

Rajendran, G., Buchholz, S., Aragam, B., Schölkopf, B., and Ravikumar, P. Learning interpretable concepts: Unifying causal representation learning and foundation models, 2024. URL https://arxiv.org/abs/2402.

09236.

Roeder, G., Metz, L., and Kingma, D. P. On linear identifiability of learned representations, 2020. URL https: //arxiv.org/abs/2007.00810.

Rubinstein, A. and Uselis, A. Stai-tuned experiment scheduler: Structured experiment management with google sheets, weights & biases, and hpc integration. https:

//github.com/arubique/stnd, 2025. GitHub repository. Utility code for reproducible and streamlined experiment management with Google Sheets integration, W&B logging, and HPC support (e.g., Slurm).

Schott, L., von Kügelgen, J., Träuble, F., et al. Visual representation learning does not generalize strongly within the same domain, 2022. URL https://arxiv.org/ abs/2107.08221.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Mullis, A., Katta, R., Kaczmarczyk, R., and Jitsev, J. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021a.

Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., and Komatsuzaki, A. Laion-400m: Open dataset of clipfiltered 400 million image-text pairs, 2021b. URL https://arxiv.org/abs/2111.02114.

Shu, R., Chen, Y., Kumar, A., Ermon, S., and Poole, B. Weakly supervised disentanglement with guarantees, 2020. URL https://arxiv.org/abs/1910. 09772.

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Sonthalia, A., Uselis, A., and Oh, S. J. On the rankability of visual embeddings, 2025. URL https://arxiv. org/abs/2507.03683.

Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N. The implicit bias of gradient descent on separable data, 2024. URL https://arxiv.org/abs/ 1710.10345.

Szabó, Z. G. The case for compositionality. In Werning, M., Hinzen, W., and Machery, E. (eds.), The Oxford Handbook of Compositionality. Oxford University Press, 2012.

Thasarathan, H., Forsyth, J., Fel, T., Kowal, M., and Derpanis, K. Universal sparse autoencoders: Interpretable cross-model concept alignment, 2025. URL https: //arxiv.org/abs/2502.03714.

Thrush, T., Jiang, R., Bartolo, M., Singh, A., Williams, A., Kiela, D., and Ross, C. Winoground: Probing vision and language models for visio-linguistic compositionality, 2022. URL https://arxiv.org/abs/2204.

03162.

Trager, M., Perera, P., Zancato, L., Achille, A., Bhatia, P., and Soatto, S. Linear spaces of meanings: Compositional structures in vision-language models, 2023. URL https://arxiv.org/abs/2302.14383.

Tschannen, M., Gritsenko, A., Wang, X., Naeem, M. F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Hénaff, O., Harmsen, J., Steiner, A., and Zhai, X. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features, 2025. URL https://arxiv.

org/abs/2502.14786.

Udandarao, V., Prabhu, A., Ghosh, A., et al. No ‘zero-shot’ without exponential data: Pretraining concept frequency determines multimodal model performance, 2024. URL http://arxiv.org/abs/2404.04125.

Udandarao, V., Cherti, M., Karthik, S., Jitsev, J., Albanie, S., and Bethge, M. A good crepe needs more than just sugar: Investigating biases in compositional visionlanguage benchmarks, 2025. URL https://arxiv.

org/abs/2506.08227.

Uselis, A., Dittadi, A., and Oh, S. J. Does data scaling lead to visual compositional generalization?, 2025. URL https://arxiv.org/.

Wang, H. Enhancing compositional generalization via compositional feature alignment, 2025. URL https: //arxiv.org/.

Watters, N., Matthey, L., Burgess, C. P., and Lerchner, A. Spatial broadcast decoder: A simple architecture for learning disentangled representations in vaes, 2019. URL https://arxiv.org/abs/1901.07017.

Weller, O., Boratko, M., Naim, I., and Lee, J. On the theoretical limitations of embedding-based retrieval, 2025a. URL https://arxiv.org/abs/2508.21038.

Weller, O., Boratko, M., Naim, I., and Lee, J. On the theoretical limitations of embedding-based retrieval, 2025b. URL https://arxiv.org/abs/2508.21038.

Wiedemer, T., Mayilvahanan, P., Bethge, M., and Brendel, W. Compositional generalization from first principles, 2023. URL http://arxiv.org/abs/2307. 05596.

- Xu, G., Kordjamshidi, P., and Chai, J. Prompting large pre-trained vision-language models for compositional concept learning, 2022. URL https://arxiv.org/ abs/2211.05077.
- Xu, H., Xie, S., Tan, X. E., Huang, P.-Y., Howes, R., Sharma, V., Li, S.-W., Ghosh, G., Zettlemoyer, L., and Feichtenhofer, C. Demystifying clip data. arXiv preprint arXiv:2309.16671, 2023.

Yamada, Y., Tang, Y., Zhang, Y., and Yildirim, I. When are lemons purple? the concept association bias of visionlanguage models, 2024. URL https://arxiv.org/ abs/2212.12043.

Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., and Zou, J. When and why vision-language models behave like bags-of-words, and what to do about it?, 2023. URL https://arxiv.org/abs/2210.01936.

Zaigrajew, V., Baniecki, H., and Biecek, P. Interpreting clip with hierarchical sparse autoencoders, 2025. URL https://arxiv.org/abs/2502.20578.

Zaslavsky, T. Facing up to arrangements: Face-count formulas for partitions of space by hyperplanes: Face-count formulas for partitions of space by hyperplanes, volume 154. American Mathematical Soc., 1975.

Zhai, X., Zhang, A., Kolesnikov, A., Beyer, L., Kipf, T., Kuhn, J., Minderer, M., Ilharco, G., Tran, D., and Steiner, A. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training, 2023. URL https://arxiv.org/abs/2303.15343.

Ziegler, G. M. Lectures on polytopes. Springer-Verlag, New York, 1995. URL http://www.worldcat. org/search?qt=worldcat_org_all&q= 9780387943657.

# Appendix

###### Contents

- A Notation and Symbols 18
- B Extended related work 19
- C Necessary conditions (proof of Proposition 1) 21
- D Sufficiency of linear factorization for compositional generalization (proof of Proposition 2) 26

- D.1 Binary valued-case: sufficiency of SVMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- D.2 General case: linearly factored embeddings and sufficiency of recovering the factors . . . . . . . . . . . . . . . . . . . 28

- E Packing and minimum dimension (proof of Proposition 3) 31
- F Examples of linearly compositional models and their geometries 33

- F.1 Case 1: Ideal “LRH” concept classifier . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- F.2 Case 2: Ideal “on-off” concept classifier . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

- G Discussion on stability 39
- H Additional information 40

- H.1 Testing linear factorization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- H.2 Whitening in measuring linear factorization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- H.3 Intuition of linear factorization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- H.4 Non-triviality of linear factorization of linearly compositional models . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- H.5 Models and probing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

- I Additional experimental results 46

- I.1 Linearity and compositional generalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- I.2 Orthogonality of factors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- I.3 Dimensionality of factors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- I.4 Experiments using text encoders as probes (zero-shot results) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52

- I.4.1 Experiments on PUG-Animal . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
- I.4.2 Experiments on ImageNet-AO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55

- I.5 Qualitative results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- I.6 Empirical evaluation on synthetic data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62

- I.6.1 Results: Linearity. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
- I.6.2 Results: Orthogonality. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
- I.6.3 Results: Dimensionality. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64

###### A. Notation and Symbols

This section fixes notation and collects basic identities used throughout the appendix. Table 1. Key notation used in the analysis.

Notation Description Concepts and datasets

C = C1 × ··· × Ck Concept space with |Ci| = n X = {xc | c ∈ C} Input space Z = {zc | c ∈ C} Representation space Dc Cross-dataset of size 1 + k(n − 1) (see Definition 6)

Models and learning T Training support (a subset of C) T Validity class (set of valid supports) R Validity rule R : 2C →{0,1} DT Training dataset supported on T

- f : X → Z Encoder / representation map
- g : Y → Z Text encoder (CLIP instantiation)

hT Readout learned from DT H Readout hypothesis class

- A Learning rule mapping DT  → hT Π = (f,H,A,T ) Compositional-generalization setup

Counts

Ni,j(S) Marginal count of concept i taking value j in dataset S

Interventions c(i → j) Concept index with the i-th value set to j zc(i→j) Intervened representation with concept i set to j c¯i Binary complement 1 − ci (when Ci = {0,1})

Probes and parameters wi,j Probe vector for concept i, value j W,b Linear readout parameters h(z) = Wz + b

c)

w(D

i,j Weight vector for concept i, class j b(D

c)

i,j Bias term for concept i, class j Ri,j(h) Region where h predicts value j for concept i p(iT)(· | z) Per-concept posterior under readout trained on T

Factorization objects P ∈ Rd×d Projection matrix PW Orthogonal projector onto probe span ui,c

i ∈ Rd Linear factor for concept i, value ci

###### B. Extended related work

Compositional generalization. Research on compositional generalization investigates how models can systematically combine concepts. On the objective side, approaches such as Compositional Feature Alignment (Wang, 2025) and Compositional Risk Minimization (Mahajan et al., 2025) study how model training objectives, and model architecture (Jarvis et al., 2024) affect compositional generalization. On the representational side, kernel analyses characterize when certain compositional structures in embeddings yield generalization theoretically (Lippl & Stachenfeld, 2025), and empirical work investigates the role of disentangled representations for compositional generalization (Montero et al., 2021; Dittadi et al., 2021; Liang et al., 2025). On the data side, recent work probes whether and how scaling and data coverage improve compositional behavior (Uselis et al., 2025; Schott et al., 2022; Kempf et al., 2025). Abbasi et al. (2024) investigate CLIP’s ability to recognize unlikely attribute-object combinations, finding that CLIP models still fall short on such tasks.

Other works establish formal sufficient conditions for when particular model classes can achieve compositional generalization, e.g., generative models whose data are produced by a differentiable rendering process and whose training distribution provides compositional support over latent factors (Wiedemer et al., 2023), constrained decoders paired with explicit inversion (Brady et al., 2025), conditional diffusion models with local conditional scores (Bradley, 2025), discriminative models whose inputs are drawn from an additive energy distribution (Mahajan et al., 2025), or linearly factorized representations (Uselis et al., 2025). In contrast, we do not impose specific structure on the data-generating process or on the learned representations. Instead, we ask what properties are implied if a model transfers from a restricted subset of the data space to the full space under our desiderata. Within this setting, our results can be interpreted as providing necessary conditions for compositional generalization for models that satisfy these desiderata.

Geometry of learned representations. A large literature studies the shape of learned features. In VLMs, Trager et al. (2023) report compositional linear subspaces, while in LLMs the Linear Representation Hypothesis (LRH) is examined mechanistically and statistically (Jiang et al., 2024; Park et al., 2023). Extending LRH, Engels et al. (2025) show that features can be multi-dimensional rather than rank-1, and Roeder et al. (2020) analyze identifiability constraints. Sparse-autoencoder probes provide evidence for monosemantic or selectively remapped features in VLMs (Pach et al., 2025; Zaigrajew et al., 2025; Lim et al., 2025). Beyond nominal labels, ordinal/ordered concepts motivate the rankability of embeddings (Sonthalia et al., 2025). More broadly, capacity limits for embedding-based retrieval emphasize geometric bottlenecks (Weller et al., 2025a). (Elhage et al., 2022) investigated empirically how neural networks can represent more features than there are dimensions in two-layer auto-encoder models. They found a tendency to encode features near-orthogonally with respect to neurons. Abbasi et al. (2024) find evidence of disentanglement in CLIP models. In contrast to these works, which document linear or near-orthogonal structure empirically, we show that under practice-driven desiderata and standard training, linearity and orthogonality are necessary.

Concurrent to our study of compositional generalization constraints in embedding spaces, Fel et al. (2025) conduct a large-scale, empirical concept analysis of DINOv2 using sparse autoencoders, showing strong task-dependent structure in which different downstream tasks recruit different concept families. They further propose a “Minkowski Representation Hypothesis”, in which token activations behave as a Minkowski sum of convex polytopes, emphasizing both the interpretability consequences and the non-uniqueness/identifiability limitations of such decompositions. While their focus is token-level geometry and interpretability (rather than CLS-centric compositional probing and necessary conditions), their observations offer complementary evidence that modern vision encoders exhibit rich internal geometric structure beyond a purely unstructured embedding space.

Data, objectives, and training effects on geometry. Data distribution strongly shapes zero-shot behavior; concept frequency during pretraining predicts multimodal performance (Udandarao et al., 2024). On the objective side, BCE vs. CE can induce different feature geometries (Li et al., 2025), and contrastive/InfoNCE objectives exhibit characteristic similarity patterns (Lee et al., 2025). Convergence perspectives argue that the objective drives canonical representational forms (Huh et al., 2024), and objective choice has been tied to representational similarity across datasets (Ciernik et al., 2025).

Binding, explicit structure injection, and concept identification. Work on binding asks whether models maintain factored world states (Feng et al., 2025), and CLIP has been observed to show uni-modal binding (Koishigarina et al., 2025). Surveys and empirical studies examine binding limits and emergent symbolic mechanisms (Campbell et al., 2025; Assouel et al., 2025). Other approaches inject structure directly, e.g., hyperbolic image-text embeddings and entailment learning (Pal et al., 2024; Desai et al., 2024), or pursue concept identification at the causal/foundation interface and object-centric pipelines (Rajendran et al., 2024; Mamaghan et al., 2024).

Relation to disentangled and object-centric representations. Work on disentangled representations largely focuses on specifying desiderata for internal codes (e.g., disentanglement, completeness, informativeness) and proposing metrics or training schemes to satisfy them, often with the informal motivation that such structure should help downstream generalization (Bengio et al., 2014; Eastwood & Williams, 2018; Higgins et al., 2018). Closely related, object-centric representation learning injects an inductive bias that scenes should be modeled as compositions of objects, which can be viewed as a structured form of factorization/binding that may support compositional transfer and robustness (Greff et al., 2020). Recent studies directly probe how these properties relate to out-of-distribution or compositional generalization, often with mixed or limited evidence (Watters et al., 2019; Dittadi et al., 2021; 2022; Montero et al., 2021; 2024; Kapl et al., 2026). We instead ask a complementary question: if a discriminative model does exhibit compositional generalization when learned from a subset of the data space, what must necessarily be true of its embeddings?

A large body of literature has studied the usefulness and implications of learning disentangled representations in an unsupervised way (Bengio et al., 2014; Lake et al., 2017). Most commonly, the goal is to learn a generative model, usually through a VAE (Kingma & Welling, 2014), that can compress the data in a disentangled manner, in a way that allows to reconstruct these representations. While shown to be impossible without additional assumptions (Locatello et al., 2019), under weak supervision learning is possible (Shu et al., 2020; Locatello et al., 2020). Measuring the degree of disentanglement in these models is in itself non-trivial and various metrics have been proposed, e.g. by measuring disentanglement by performing interventions on the representations (Higgins et al., 2017; Kim & Mnih, 2018).

The DCI framework (Eastwood & Williams, 2018) proposes desiderata of properties disentangled representations should satisfy, namely disentanglement, completeness, and informativeness, and proposes a metric to measure them. Some works also consider what constitutes a good disentanglement (Higgins et al., 2018) and propose a conceptual framing of meaning behind disentangled representations with respect to the data generative process in terms of group actions of transformations.

Abbasi et al. (2024) investigate the role of representation disentanglement in compositional generalization in CLIP models. Using metrics such as DCI, they find that CLIP models with more disentangled text and image representations exhibit higher compositional OOD accuracy on their attribute-object dataset (ImageNet-AO). This work is complementary to ours. Their study explores correlations between disentanglement and compositional generalization by probing CLIP embeddings with respect to the adjective and noun components present in the inputs. For instance, they estimate “attribute” and “object” subspaces by feeding isolated adjectives or nouns into the text encoder, or by generating isolated attributes/objects via a text-to-image model and embedding them with CLIP. However, this approach assumes that CLIP’s embedding space is additively decomposed with respect to individual words, an assumption that is not guaranteed to hold. Indeed, Yamada et al. (2024) show that word embeddings in language models are often highly entangled with associated concepts. In contrast, our necessary condition does not rely on word-level decomposition. We posit that models achieving perfect downstream compositional performance must possess linearly factorized representations that separate per-concept components, independent of how an encoder processes individual words. In short, our work provides principled motivation for analyses of representational decomposition, whereas Abbasi et al. (2024) offer an empirical correlation study based on CLIP’s emergent disentanglement.

Lippl & Stachenfeld (2025) investigate when a particular form of compositionally structured representations, specifically representations whose similarity depends only on how many underlying components two inputs share, supports downstream compositional generalization. Using kernel theory, they characterize exactly which tasks linear readouts on top of such representations can solve, showing that these models are fundamentally restricted to conjunction-wise additive functions. In contrast, we focus on a specific subclass of compositional tasks: identifying factors of inputs that never co-occur during training. While Lippl & Stachenfeld (2025) characterize what kinds of generalization are possible under a compositional representational structure, we ask the complementary question: given perfect downstream performance on such a task, what representational structure must the model necessarily possess under the desiderata we specify?

- C. Necessary conditions (proof of Proposition 1) In this section, we prove Proposition 1 and state the auxiliary lemmas used in the argument.

Our default training-support regime is the one used in the main result: valid supports satisfy |T| = 2k−1 + 1. In the proof below, we work with cross-datasets Dc (Definition 6), i.e., datasets centered at c that vary one concept at a time, because this makes the core geometric argument transparent. We write D for the full dataset of all nk combinations and Dc for a cross-dataset centered at c.

Although most proofs in this section are written in the binary setting for clarity, we state intermediate claims in a form that extends directly to general n where possible. Concretely, in the binary setting Ci = {0, 1}. Any learned quantity carries a superscript indicating the training set, e.g., wi(D) or w(D

c)

i . For concept i and training set S, we use the logistic score form

h(iS)(z) := (wi(S))⊺z + b(iS), p(iS)(z) = σ(h(iS)(z)), and we drop the superscript when the dataset is clear from context.

- Definition 5 (Intervention on a concept value). For any concept index i ∈ [k], target value j ∈ [n], and concept vector c ∈ [n]k, we define the intervened index and representation

c(i → j) := (c1, . . . , ci−1, j, ci+1, . . . , ck), zc(i→j) := zc with concept i set to j.

In case of multiple interventions, they compose componentwise, e.g. for distinct i, m ∈ [k], c(i → j, m → l) = (c1, . . . , ci−1, j, ci+1, . . . , cm−1, l, cm+1, . . . , ck).

- Definition 6 (Cross dataset at c). Given a concept space C = C1 × · · · × Ck, we say that a dataset Dc is a cross-dataset at c ∈ [n]k if:

- 1. It contains only samples that vary one concept at a time around the center c:

Dc = (c′1, c2, . . . , ck) : c′1 ∈ [n] ∪ · · · ∪ (c1, c2, . . . , ck−1, c′k) : c′k ∈ [n] = {c} ∪

k

i=1 a∈[n]\{ci}

{c(i → a)}.

- 2. Its size is 1 + k(n − 1),

- Definition 7 (Dataset marginal counts). For any dataset S ⊆ {(zc) : c ∈ [n]k} (e.g., S = D or S = Dc), concept i ∈ [k], and value j ∈ [n], the marginal count of value j in S is

Ni,j(S) := {c ∈ [n]k : (zc) ∈ S, ci = j} .

When S is clear, we abbreviate Ni,j := Ni,j(S). Remark 1 (Marginal counts: full vs cross-datasets). For the full dataset D, the marginal counts are balanced:

Ni,j(D) = nk−1 for all i ∈ [k], j ∈ [n]. For a cross-dataset Dc as in Definition 6, the marginal counts satisfy

Ni,j(Dc) =

1 + (k − 1)(n − 1), j = ci, 1, j ̸= ci.

Proof. In D, fixing c′i = j leaves nk−1 free coordinates. In Dc: varying concept i contributes one point for each j ̸= ci; the center contributes one more with c′i = ci; varying any other concept r ̸= i adds (n − 1) points with c′i = ci, across (k − 1) such concepts, totaling (k − 1)(n − 1).

| |
|---|

- Definition 8 (Binary complement notation). In the binary case (Ci = {0, 1}), we write c¯i := 1 − ci for the complement value of concept i. As shorthand for an intervention to the complement, we write c(¯ci) := c(i ← c¯i). To make use of Stability in the binary case, we use the identifiability of the sigmoid.

- Lemma 1 (Binary equal probabilities imply equal affine parameters). For a concept index i and two training supports S, S′, let

′)

′)

′)

h(iS)(z) := (wi(S))⊺z + b(iS), h(S

i )⊺z + b(S

i (z) := (w(S

i . Assume that for every z ∈ Rd,

′)

σ h(iS)(z) = σ h(S

i (z) , where σ(t) = 1/(1 + e−t). Then

′)

′)

wi(S) = w(S

i and b(iS) = b(S

i .

Proof. Since σ is injective,

′)

h(iS)(z) = h(S

i (z) ∀z ∈ Rd. Hence

⊺

′) i

′)

wi(S) − w(S

z + b(iS) − b(S

i = 0 ∀z ∈ Rd.

′)

Taking z = 0 gives b(iS) = b(S

i , and then

′) i

wi(S) − w(S

⊺

z = 0 ∀z ∈ Rd,

′)

implies wi(S) = w(S

i , since the equation holds for any point in Rd.

| |
|---|

(b)

(a)

- D(1,1) D(2,1)
- D(1,2) D(2,2)

|+1|
|---|

|-1|
|---|

|zc<br><br>zc(i→ y1|
|---|

|y1 ≡ zc xc(i|
|---|

c¯i)

→c¯i)

zz22

zz22

|+1|
|---|

|-1|
|---|

|zc<br><br>zc(i y2|
|---|

|zc<br><br>y2 ≡|
|---|

zc(i→c¯

→c¯i)

i)

z1 z1 z1 z1

Figure 13. Illustration of the invariance lemma (left) and the main proposition (right). (a) The invariance lemma (Lemma 3): any point zc can be made a support vector by an appropriate choice of a pair of datasets. (b) The main proposition (Proposition 1): linearity and orthogonality under GD+BCE follow because, for each concept i, the SVM support vectors correspond to counterfactual pairs that differ only in the i-th concept. Concretely, when the minority support point is zc(i→c¯i), the majority-class support vector y1 under D(1,1) reduces to zc, and analogously for y2.

- Lemma 2 (Bi-directional tight support vectors). For binary concepts Ci = {0, 1} for all i ∈ [k], consider a pair of cross-datasets Dc and Dc(i→c¯i) for some i ∈ [k] and the corresponding SVM solutions {wi, bi} and {wi′, b′i} for these datasets, respectively. Then, zc(i→c¯i) is

a tight support vector for concept i under (wi, bi) trained on Dc, and zc is a tight support vector under (wi′, b′i) trained on Dc(i→c¯i), i.e. the following hold:

(w′)⊺i zc + b′i = yi(c)

wi⊺zc(i→c¯i) + bi = yi(c(i → c¯i)),

(8)

where yi(c) ∈ {+1, −1} is the label of zc with respect to concept i.

Proof. For the second equality: by Remark 1, Ni,ci(Dc) = 1 + (k − 1)(n − 1) and Ni,c¯i(Dc) = 1. Hence both classes are non-empty in Dc; standard hard-margin SVM argument then gives at least one support vector in each class achieving equality at the margin (Cortes

& Vapnik, 1995). Repeating the same argument for Dc(i→c¯i) gives the first equation.

| |
|---|

- Lemma 3 (Invariance to irrelevant concepts). Assume each concept is binary, Ci = {0, 1} for all i ∈ [k], and Desiderata 1–3 hold, and

consider either the rule |T| = 2k−1 + 1 or the cross-dataset design (Definition 6). Then, for any i ∈ [k] and any c, c′ ∈ [2]k with ci = c′i, it holds that

P(Ci = ci | zc) = P(Ci = ci | zc′). (9)

Proof. We encode the i-label by yi(z) ∈ {+1, −1} with yi(z) = +1 iff Ci(z) = 1 and −1 otherwise, and let

hi(z) := wi⊺z + bi (10)

By Stability (Desideratum 3) and Lemma 1, the affine separator (wi, bi) is the same across valid cross-datasets. For any c ∈ [2]k, consider the cross-dataset Dc(i→c¯i). By Remark 1, in this dataset concept i has count 1 for value ci and count 1 + (k − 1)(n − 1) for value c¯i, so

the unique minority example with respect to concept i is zc. By Lemma 2, both classes have tight support vectors. Since the minority class has exactly one point, that point zc is the tight support vector for its class. Hence

###### yi(zc)hi(zc) = 1. (11)

The same argument applies to c(i → c¯i) in D, where it follows that yi(zc(i→c¯i)) hi(zc(i→c¯i)) = 1. Since this was performed for any c, it follows that

+1, if Ci(z) = 1, −1, if Ci(z) = 0,

on the whole grid {zc : c ∈ [2]k}.

hi(z) =

i(z) (and P(Ci = 0 | z) = 1 − P(Ci = 1 | z)), the conditional probability P(Ci = ci | zc) is constant over all c with fixed ci. In particular, for any c, c′ with ci = c′i,

Hence hi(z) depends only on Ci(z) and not on the other concepts. Since P(Ci = 1 | z) = σ(hi(z)) = 1+e−1h

P(Ci = ci | zc) = P(Ci = c′i | zc′).

| |
|---|

Next, we state an important property of SVMs on two separable sets and adapt it to our case, where one of the elements in the set is a singleton.

- Lemma 4 (SVM geometry for separable sets). Given a set of points Y := {yi}Ni=1 with yi ∈ Rd, and a point z ∈ Rd, let (w, b) be the optimal hard-margin SVM separator between classes Y and {z}, with the canonical scaling w⊺x + b = ±1 on support vectors. Then:

- 1. There exist coefficients {λi}Ni=1 with λi ≥ 0 and i λi = 1 such that

w⊺

i

λiyi + b = −1, (12) w⊺z + b = +1. (13)

- 2. The weight w is related to the shortest-segment between the convex hulls of the two classes as

2 ∥w∥2

w = z −

i

λiyi. (14)

Proof. This is a standard geometric characterization of hard-margin SVMs: w is parallel to the shortest segment joining the convex hulls of the two classes, and under canonical margin scaling the two supporting hyperplanes are at signed distance 1/∥w∥ from the decision

hyperplane, hence the support-point displacement equals ∥w2∥2 w (Bennett & Bredensteiner, 2000). We now establish the main result on the resulting geometry of linearly generalizable compositional models. Intuition of it, together with the invariance lemma above is presented in Fig. 13.

| |
|---|

Proposition 1 (Compositional generalization implies linear factorization). Let Π = (f, H, A, T ) be the tuple instantiated in Definition 4, with linear heads H and A given by GD+CE. Suppose that the training sets follow random sampling with validity rule R(T) = 1 if |T| = 2k−1 + 1. Assume Desiderata 1–3 are satisfied. Then, under the binary grid Ci = {0, 1} with Z = {zc : c ∈ [2]k} ⊂ Rd, there exist {ui,0, ui,1 ∈ Rd}ki=1 such that for every c ∈ [2]k the following holds:

- 1. (Linearity) zc = ki=1 ui,ci.
- 2. (Cross-concept orthogonality) (ui,1 − ui,0) ⊥ (uj,1 − uj,0) for all i, j ∈ [k] with (i ̸= j).

Proof. Although Proposition 1 is stated for the |T| = 2k−1 + 1 regime, the geometric mechanism is easiest to see on the cross-dataset construction Dc (Definition 6), which isolates one-concept flips around a center point c (in the binary case, |Dc| = 1 + k). We therefore present the proof using cross-datasets to keep the SVM geometry and the stability constraints transparent.

###### Linearity.

The idea is to show that for a pair of cross-datasets that share the datapoints in negative class, the shortest distance from a single point in the positive class to the convex set of the positive points is achieved by considering a flip in one of the concepts. We make this concrete below.

Consider any point zc and its corresponding cross-dataset Dc. For any concept i ∈ [k], let zc(i→c¯i) be the counterfactual point obtained by flipping concept i to c¯i, and consider the neighboring cross-dataset Dc(i→c¯i).

Note that for the concept i it holds that:

###### 1. Under Dc = {zc} ∪ {zc(j→c¯j) : j ∈ [k]}. For each concept i, the marginal counts are Ni,ci(Dc) = k, Ni,c¯i(Dc) = 1 (15)

(by Remark 1). Thus zc(j→c¯j) is the unique minority example for concept j (with label c¯j), and

Y1 := Dc \ {zc(i→c¯i)} (16) is the set of k majority examples (with label ci).

###### 2. Under Dc(i→c¯i) := {zc(i→c¯i)} ∪ {zc} ∪ {zc(i→c¯i,ℓ→c¯ℓ) : ℓ ∈ [k] \ {i}},

for any ℓ ̸= i the counts are unchanged: Nℓ,cℓ(Dc(i→c¯i)) = k and Nℓ,c¯ℓ(Dc(i→c¯i)) = 1, but for concept i they swap: Ni,c¯i(Dc(i→c¯i)) = k and Ni,ci(Dc(i→c¯i)) = 1. Thus zc is now the unique minority example for concept i (label ci). We denote

Y2 = Dc(i→c¯i) \ {zc} (17) as the majority examples for concept i.

Let the pair of majority support vectors for Dc and Dc(i→c¯i) be y1 and y2 respectively. By Lemma 4, we can write

λjzc(j→c¯j) and y2 = γizc(i→c¯i) +

y1 = λizc +

j∈[k]\{i}

γjzc(i→c¯i,j→c¯j) (18)

j∈[k]\{i}

for some convex combinations λj ≥ 0 with kj λj = 1 and γj ≥ 0 with kj γj = 1. Additionally, note that by Lemma 3 it holds for any point zc′ for concept j ∈ [k] that

wj⊺zc′ + bj = yj(c′), (19) where we use a shorthand yj(c) = 1 if cj = 1 and yj(c) = −1 otherwise. Then, by Lemma 4 it holds that the support vectors are aligned with the shortest segment between the convex sets (pairs of zc(i→c¯i) and y1, and zc and y2)

2 ||wi||2

2 ||wi||2

wi = y1 and zc − yi(c)

wi = y2, (20)

zc(i→c¯i) + yi(c)

where clearly yi(c(i → c¯i)) = −yi(c). From this, it follows that

###### y1 − zc(i→c¯i) = zc − y2. (21)

Now, for any j ̸= i it follows that

  + bj (22)

 λizc +

wj⊺y1 + bj = wj⊺

λlzc(l→c¯l)

l∈[k]\{i}

- k
- l

= λiwj⊺zc +

λlwj⊺zc(l→c¯l) +

λlbj (23)

l∈[k]\{i}

###### = λi(wj⊺zc + bj) +

λl(wj⊺zc(l→c¯l) + bj) (24)

l∈[k]\{i}

λlyj(c(l → c¯l)) + λjyj(c(j → c¯j)) (25)

= λiyj(c) +

l∈[k]\{i,j}

  yj(c) − λjyj(c) (26)

 

λl

= λiyj(c) +

l∈[k]\{i,j}

= (1 − λj)yj(c) − λjyj(c) = (1 − 2λj)yj(c), (27)

where we used the fact that λ are convex combinations in the second equality, and the fact that in the paired dataset k-concept values remain the same when flipping any other concept than k.

By repeating the same calculation as (22) for y2, we get:

wj⊺y2 + bj = (1 − 2γj)yj(c). (28)

By (21) it follows that (again with j ̸= i)

wj⊺(y1 − zc(i→c¯i)) = wj⊺(zc − y2)

⇒ wj⊺y1 + bj − wj⊺zc(i→c¯i) − bj = wj⊺zc + bj − wj⊺y2 − bj

(29)

⇒ (1 − 2λj)yj(c) − yj(c) = yj(c) − (1 − 2γj)yj(c)

⇒ 1 − 2λj − 1 = 1 − 1 + 2γj

⇒ λj + γj = 0.

Clearly, since λj and γj are convex combinations and thus non-negative, (29) implies that λj = γj = 0. By repeating this process for all j ̸= i, we get that λj = γj = 0 for all j ̸= i, and therefore λi = γi = 1. From this, it follows that y1 = zc and y2 = zc(i→c¯i). This means that

2 ||wi||2

2 ||wi||2

wi = zc and zc − yi(c)

wi = zc(i→c¯i), (30)

zc(i→c¯i) + yi(c)

and therefore the differences between zc − zc(i→c¯i) are independent of other concept variations. Because of that, we can write any datapoint zc as a sum of concept-specific values ui,ci(ci ∈ [2]). For instance, if we fix 0 = (0, . . . , 0) ∈ [2]k, we can express zc as, for example (up to a global linear shift per concept)

2 ||wi||2

wi,

ui,0 = z0/k, ui,1 = z0/k +

k

ui,ci,

zc =

i=1

from here, we can write any datapoint zc as

(31)

k

k

k

k

2 ||wi||2

2 ||wi||2

wi, (32)

zc =

ui,ci =

z0/k +

(z0/k +

wi) = z0 +

i=1

i∈[k]:ci=0

i∈[k]:ci=1

i∈[k]:ci=1

which establishes linearity. Orthogonality. First, note that by invariance (Lemma 3) it holds that for any concept i, changes in concept values other than i do not affect the prediction of concept i:

wi⊺zc + bi = wi⊺zc(j→c¯j) + bi (33) But by linear factorization (31) it follows that

wi⊺zc + bi = wi⊺zc(j→c¯j) + bi

⇒ wi⊺(zc − zc(j→c¯j)) = 0 ⇒ wi⊺ uj,cj − uj,c¯j = 0

2 ||wj||2

⇒ wi⊺

wj = 0

⇒ wi⊺wj = 0.

(34)

Then,

(ui,ci − ui,c¯i)⊺(uj,cj − uj,c¯j) ∝ wi⊺wj = 0. (35) More generally, orthogonality of one concept holds against the span of other concepts as well. For {αj ∈ R}j̸=i it follows that

  ∝ wi⊺

 

  = 0, (36)

 

(ui,ci − ui,c¯i)⊺

αj(uj,cj − uj,c¯j)

αjwj

j̸=i

j̸=i

and therefore orthogonality holds against the span of other concepts differences.

| |
|---|

###### D. Sufficiency of linear factorization for compositional generalization (proof of Proposition 2)

This section complements the main text’s necessity results by showing a converse: under the same linear-head setting, linearly factored representations are sufficient to obtain compositional generalization under our desiderata.

We discuss two cases. First, in the binary setting, we show that linear factorization together with cross-concept orthogonality makes stable transfer automatic for GD+CE, even from small but diverse training supports (Section D.1). Second, in the general multi-valued setting we discuss the general case when the underlying features are linear, but not necessarily orthogonal across concepts, and the learning algorithm is not necessarily GD+CE (Section D.2). There, training with GD+CE does not necessarily satisfy the desiderata, but in principle classifiers can be constructed that do, precisely because the factors can be recovered.

###### D.1. Binary valued-case: sufficiency of SVMs

In this section, we show that linear factorization together with cross-concept orthogonality makes stable transfer automatic for GD+CE, even from small but diverse training supports. Since concepts are binary-valued, we can represent the readout for each concept by a single affine separator with parameters (wi, bi), and we work with these parameters throughout (in the same way as Section C).

- Proposition 4 (Linear factorization implies compositional generalization). Consider the binary grid Ci = {0, 1} with representations Z = {zc : c ∈ [2]k} ⊂ Rd. Assume there exist {ui,0, ui,1 ∈ Rd}ki=1 such that for every c ∈ [2]k:

- 1. (Linearity) zc = ki=1 ui,ci.
- 2. (Cross-concept orthogonality) (ui,1 − ui,0) ⊥ (uj,1 − uj,0) for all i ̸= j.

Then for any training set T ⊆ [2]k satisfying either of the following conditions:

- 1. Any set with |T| = 2k−1 + 1, or
- 2. A cross dataset Dc (i.e. the center c and all one-value flips from c, Definition 6) with |T| = 1 + k.

It holds that gradient descent + cross-entropy loss trained on T satisfy the desiderata on the entire grid: they recover every concept value on all zc (transferability) and are invariant across valid T (stability).

Proof. Throughout the proof we use the standard result that the GD+CE converges to the max-margin SVM solution in the binary-class case (Soudry et al., 2024) and interpret the optimal solution as a weight vector wi and bias bi that separates the two classes.

Full dataset case. First, we establish the exact weights produced by the linear probes. For that, note that for any concept i ∈ [k], the SVM solution is proportional to the shortest segment between the convex hulls of the points in the two classes, denoted side by side as

Y− := {zc : c ∈ [2]k, ci = 0}, Y+ := {zc : c ∈ [2]k, ci = 1}. (37) with the proportionality constant ∥w2

i∥2 (Bennett & Bredensteiner, 2000). Equivalently, there exist convex combinations y− ∈ conv(Y−) and y+ ∈ conv(Y+) such that wi is parallel to the shortest segment y+ − y−, where

γc = 1. (38)

λczc, λ, γ ⪰ 0,

y− =

γczc, y+ =

λc = 1,

c∈[2]k,ci=0

c∈[2]k,ci=1

c∈[2]k,ci=1

c∈[2]k,ci=0

Importantly, these convex combinations are the support vectors of the SVM solution and correspond to the shortest distance between the two classes. To proceed, we first write the difference between any two points in the convex hulls and then lower bound the norm of the difference and arrive at the exact weight vector.

Note that for any y+ and y− their difference can be written as:

y+ − y− =

###### λczc −

c∈[2]k:ci=1

 

=

λc

c∈[2]k:ci=1

λcui,1 −

=

c∈[2]k:ci=1

= ui,1 − ui,0 +

γczc (40)

c∈[2]k:ci=0

  −

 

  (41)

uj,cj + ui,1

γc

uj,cj + ui,0

j̸=i

j̸=i

c∈[2]k:ci=0

uj,cj (42)

uj,cj −

γcui,0 +

λc

γc

j̸=i

j̸=i

c∈[2]k:ci=0

c∈[2]k:ci=1

c∈[2]k:ci=0

uj,cj, (44)

uj,cj −

γc

λc

j̸=i

j̸=i

c∈[2]k:ci=0

c∈[2]k:ci=1

where the last equality uses the fact that the sum runs over all concept combinations, and therefore sums to 1. Note that the last term in

(44) can be written as

uj,cj (45)

uj,cj −

λc

γc

j̸=i

j̸=i

c∈[2]k:ci=1

c∈[2]k:ci=0

 

  (46)

(λc − γc(i→0))

=

uj,cj

j̸=i

c∈[2]k:ci=1

(λc − γc(i→0))uj,cj (47)

=

j̸=i c∈[2]k:ci=1

  uj,1 +

 

  uj,0

  , (48)

 

 

(λc − γc(i→0))

(λc − γc(i→0))

=

j̸=i

c∈[2]k:ci=1,cj=1

c∈[2]k:ci=1,cj=0

but since c∈[2]k:c

i=1(λc − γc(i→0)) = 0, it follows that for any j ̸= i,

(λc − γc(i→0)) = 0 (49)

(λc − γc(i→0))+

c∈[2]k:ci=1,cj=1

c∈[2]k:ci=1,cj=0

(λc − γc(i→0)) (50)

⇒

(λc − γc(i→0)) = −

c∈[2]k:ci=1,cj=1

c∈[2]k:ci=1,cj=0

We thus denote ∆j := c∈[2]k:c

i=1,cj=1(λc − γc(i→0)). The full expression of (40) can be written compactly as

y+ − y− = ui,1 − ui,0 +

= ui,1 − ui,0 +

∆juj,1 − ∆juj,0 (51)

j̸=i

∆j(uj,1 − uj,0). (52)

j̸=i

Recall that by assumption, ui,1 − ui,0 ⊥ uj,1 − uj,0 for all j ̸= i, so it follows that

 

  =

(ui,1 − ui,0)⊺

∆j(ui,1 − ui,0)⊺(uj,1 − uj,0) = 0. (53)

∆j(uj,1 − uj,0)

j̸=i

j̸=i

Therefore, ui,1 −ui,0 ⊥ j̸=i ∆j(uj,1 −uj,0). This allows us to apply the Pythagorean theorem when computing the distance between y+ and y−:

2

∥y+ − y−∥2 = ∥ui,1 − ui,0∥2 +

(54)

∆j(uj,1 − uj,0)

j̸=i

≥ ∥ui,1 − ui,0∥2. (55)

Thus, any two points in their respective convex hulls are at least as far apart as the distance between ui,1 −ui,0. We make use of this result in computing the SVM solution by picking two points y+ and y− that have exactly the shortest possible distance between them, thus establishing them as the support vectors. Conveniently, any two “counterfactual” points do: for any c ∈ [2]k by picking y+ = zc(i→1) and y− = zc(i→0), we have that

∥zc(i→1) − zc(i→0)∥2 = ∥ui,1 − ui,0∥2.

Since this computation is independent of the particular choice of the concept i, it holds for any concept. As such, we can write the weight vector wi as

To show that classification works, note that

2 ||ui,1 − ui,0||2

(ui,1 − ui,0). (56)

wi =

k

zc =

j=1

uj,cj + u¯j =

k

j

uj,cj +

k

j

u¯j, (57)

where uj,cj := uj,cj − u¯j is the centered factor for concept j for the value cj, and u¯j := 12(uj,1 + uj,0) is the average of the two concept factors.

But the centered factors must sum to zero, so uj,0 = − uj,1. Therefore for any i ̸= j

(ui,1 − ui,0)⊺(uj,1 − uj,0) = 0 ⇒ ( ui,1 − ui,0)⊺( uj,1 − uj,0) = 0

⇒ u⊺i,1 uj,1 = 0. (58)

By evaluating the classification rule at any zc for any concept i we get

wi⊺zc + bi = wi⊺

= wi⊺

k

j=1

uj,cj + u¯j + bi (59)

k

uj,cj + wi⊺

j=1

k

u¯j + bi, (60)

j=1

but note that only the first term of (60) is affected by the concept i, and the following terms can be absorbed into the bias bi2. Thus, the classification is correct only if sign wi⊺ kj=1 uj,cj = 2ci − 1. By (56) it follows that

k

sign wi⊺

uj,cj = sign ((2 ui,1)⊺ ui,ci) = sign u⊺i,1 ui,ci = 2ci − 1. (61)

j=1

And therefore the classification is correct.

- (1) |T| = 2k−1 + 1 case. The main idea of the proof is that for any concept i ∈ [k] and a pair of “counterfactual” points zc, zc(i→c¯i) it holds that the distance between the two points is the shortest possible distance between any two points in the convex hulls of the two classes and is equal to ∥ui,1 − ui,0∥2. This follows the same argument as specified in the full dataset case (and in (56) in particular). All that is needed to be shown then, is the availability of such a pair of points in any training set with |T| = 2k−1 + 1.

This can be argued by contradiction. Assume that for some concept i ∈ [k] there are no two points (c, c′) with cj = c′j for all j ̸= i and ci ̸= c′i. There are 2k−1 such pairs of points in total. Thus a dataset would have to have at most 2k − 2k−1 = 2k−1 points, which contradicts the assumption that |T| = 2k−1 + 1. Therefore, such a pair of points must exist. Since this argument is independent of the particular choice of the concept i, it holds for any concept. Therefore the SVM solution will always be able to find such a pair of points that minimize the distance between the two classes (as per (54)) and will transfer as well as yield the same weight vector as the full dataset case.

- (2) Dc case with |T| = 1 + k. By construction for any concept i there exists “counterfactual” point from the center c to c(i → c¯i). By following the same argument detailed in the full dataset case as well as the previous case, it follows that the SVM solution will transfer as well as yield the same weight vector as the full dataset case. In both cases stability of the solution follows due to recovering the same weight and bias vectors.

| |
|---|

###### D.2. General case: linearly factored embeddings and sufficiency of recovering the factors

We provide a complementary analysis on the sufficient conditions for generalizing compositionally. Here, we detail the key results for recovering the factors u from representations that already possess linear factorization.

We first note the minimal dataset setting using the notion of a cross dataset, defined below.

- Lemma 5 (Uniqueness up to concept-wise shifts). Let the concept space be C = C1×· · ·×Ck. Assume two factor families {ui,j}i∈[k],j∈Ci and {vi,j}i∈[k],j∈Ci satisfy, for every c ∈ C,

k

k

zc =

ui,ci =

vi,ci.

i=1

i=1

Then there exist vectors s1, . . . , sk ∈ Rd with

k

###### si = 0

i=1

such that

vi,j = ui,j + si, ∀i ∈ [k], j ∈ Ci. Hence the factors are identifiable only up to concept-wise shifts.

2Alternatively, assume that zc is zero-centered and use the same argument.

Proof. For each concept i ∈ [k] and value j ∈ Ci, define the vector difference δi,j := vi,j − ui,j.

We first show that, within each concept i, this difference is independent of the value. Fix i ∈ [k] and pick any two values p, q ∈ Ci. Take any c ∈ C with ci = p. Using the counterfactual tuple c(i → q) and subtracting the two factorizations gives

k

k

zc − zc(i→q) =

uℓ,cℓ −

uℓ,(c(i→q))ℓ

ℓ=1

ℓ=1

k

k

vℓ,cℓ −

=

vℓ,(c(i→q))ℓ.

ℓ=1

ℓ=1

All terms except concept i cancel on both sides, so

ui,p − ui,q = vi,p − vi,q, ⇒ δi,p = δi,q. Hence, for each concept i, there is a single vector si ∈ Rd such that

δi,j = si, ∀j ∈ Ci.

Therefore vi,j = ui,j + si for all i, j. To obtain the zero-sum constraint, evaluate at any c ∈ C:

k

k

k

vi,ci − ui,ci =

si.

δi,ci =

0 =

i=1

i=1

i=1

Conversely, if s1, . . . , sk ∈ Rd satisfy ki=1 si = 0 and we set vi,j := ui,j + si, then for every c ∈ C,

k

k

k

k

ui,ci,

si =

ui,ci +

vi,ci =

i=1

i=1

i=1

i=1

so the reconstructed factors generate the same embeddings.

We illustrate this lemma graphically in Figure 14.

| |
|---|

u1

u2

- u3 v1

v2

- v3

s1

- Figure 14. Illustration of the shift ambiguity in the factorization equations. The black factors {ui} and pink factors {vi} produce the same pairwise differences; the orange arrows show concept-wise shift vectors si such that vi = ui + si.

A neat consequence is that centered factors are uniquely determined: any recovered factorization, once centered per concept, matches the true centered factors. Corollary 1 (Uniqueness of centered factors). Assume the setting of Lemma 5. For each concept i, let

1 |Ci| j∈C

µi :=

ui,j,

i

1 |Ci| j∈C

νi :=

vi,j,

i

and define the centered factors u◦i,j := ui,j − µi and vi,j◦ := vi,j − νi. Then u◦i,j = vi,j◦ for all i ∈ [k] and j ∈ Ci. Equivalently, for every c ∈ C,

k

k

vi,c◦ i, so the centered factorization is unique.

u◦i,ci =

i=1

i=1

Proof. By Lemma 5, there exist s1, . . . , sk with i si = 0 such that vi,j = ui,j + si for all i, j. Averaging over j ∈ Ci yields νi = µi + si. Thus,

vi,j◦ = vi,j − νi

= (ui,j + si) − (µi + si)

= ui,j − µi

= u◦i,j, as claimed.

| |
|---|

First, we consider the general case where concept directions are not necessarily linearly independent. Suppose the inputs zc are linearly separable for any i ∈ [k], j ∈ [n]. If we can recover all factors vi,j, we can reconstruct zc = ki=1 ui,ci and then fit linear probes for the concept values.

By Lemma 5 (and the centered-factor corollary above), recovery is unique up to concept-wise shifts, so the key requirement is rank of the one-hot design matrix induced by observed tuples.

- Proposition 5 (Maximum possible rank of the design matrix). Let A ∈ {0, 1}nk×kn be the design matrix whose kn columns are

{ai,r : i = 1, . . . , k, r = 1, . . . , n}, arranged in k blocks of size n, with all nk treatment combinations as rows and each row having exactly one 1 in each block. Then,

###### rank(A) = 1 + k(n − 1).

Proof. We first show the upper bound by spanning columns. Define 1 ∈ Rnk as the all-ones vector and, for each block i and each r = 2, . . . , n,

di,r := ai,r − ai,1. Let

B := {1} ∪ {di,r : 1 ≤ i ≤ k, 2 ≤ r ≤ n}, so |B| = 1 + k(n − 1).

For every block i, nr=1 ai,r = 1, hence

n

di,r = 1 − nai,1 ⇒ ai,1 = n1 1 −

r=2

n

di,r , ai,r = ai,1 + di,r (r ≥ 2).

r=2

Thus every original column ai,r lies in span B, so rank(A) ≤ 1 + k(n − 1). For the matching lower bound, note that A⊺ is exactly the on-off matrix from Proposition 6 with α = 1, β = 0 (rows indexed by (i, r) and columns by tuples c). The rank argument in that proof applies verbatim and gives rank(A⊺) = 1 + k(n − 1). Hence

###### rank(A) = rank(A⊺) = 1 + k(n − 1).

| |
|---|

When rank(A) = 1 + k(n − 1), the linear system Z = AU determines the centered factors uniquely (up to the shift ambiguity already characterized above). Therefore one can reconstruct the grid embeddings and, under linear separability, fit linear readouts that recover concept values on all combinations (Definition 3).

###### E. Packing and minimum dimension (proof of Proposition 3)

In this section, we elaborate on the geometric capacity question introduced in the main text: how large must the embedding dimension d be for linear probes to realize all concept combinations in C.

We use hyperplane-arrangement counting bounds to formalize this. Intuitively, each concept contributes n decision boundaries, and to classify all combinations these boundaries must carve enough regions in Rd to accommodate all nk combinations. We first recall the relevant region-count results, then apply them to prove the lower bound d ≥ k.

For d ≥ 1, we work with affine hyperplanes (the setting used by linear probes with bias):

Hw,b = { x ∈ Rd : w⊺x + b = 0 }, w ̸= 0, b ∈ R.

An arrangement H = {H1, . . . , Hm} is a family of hyperplanes. It is in general position when no d + 1 hyperplanes intersect at a common point, which maximizes the number of connected regions in Rd (Zaslavsky, 1975; Ziegler, 1995).

This allows us to count the number of connected regions in Rd that an arrangement of m hyperplanes carves out, a classical result that we state below.

Theorem 1 (Zaslavsky’s region bounds in general position (Zaslavsky, 1975)). Let H be an arrangement of m hyperplanes in Rd that is in general position. Then, the number of connected regions R(H) is given by:

- (a) Affine (with a bias) case. If the hyperplanes may carry arbitrary offsets bi (so H is not required to be central), then

R(H) = Raff(m, d) :=

d

r=0

m r

.

- (b) Central case. If every hyperplane passes through the origin,

d−1

R(H) = Rlin(m, d) := 2

r=0

m − 1 r

.

Note that for d < k, one has

d

k r

< 2k,

Raff(k, d) =

r=0

which is the key inequality we will need. We now exploit Theorem 1 to prove the lower bound on probe dimension; first for the binary case, then for general n.

3 concepts, 2 values per concept, 2D space.

Reduction to two values

|R1<br><br>R3<br><br>R4<br><br>R2<br>|
|---|

|H1<br><br>H2<br><br>H3<br><br>1 2<br><br>3<br><br>4 5<br><br>6<br><br>7|
|---|

- Figure 15. Illustration of the probe dimension lower bound. Schematic of probe hyperplane arrangements and induced regions in embedding space. (left) For k = 3 in d = 2, three affine hyperplanes partition the plane into at most 7 regions, fewer than the 23 = 8 binary concept combinations; therefore d ≥ 3. (right) For n > 2, restricting each concept to any two values induces a binary subproblem. Hence any model that realizes all nk combinations must also realize these binary restrictions, so the binary lower bound applies to the non-binary case as well.

Proposition 3 (Minimum dimension for linear probes). For k concepts, each with n values, suppose there exist linear probes that correctly classify each concept value for all nk combinations from embeddings f(x) ∈ Rd. Then necessarily d ≥ k.

Proof sketch. Reduce to the binary case by fixing two values per concept and restricting to the resulting 2k combinations. Each concept induces one affine separating hyperplane. To realize all binary labelings, the arrangement must carve at least 2k regions. By Theorem 1,

when d < k we have dr=0 kr < 2k, so 2k regions are impossible. Hence d ≥ k. Tightness follows by a d = k construction (coordinates per concept with suitable affine offsets). See Fig. 15.

Proof. First, we “reduce” the problem to the binary case. For each concept i ∈ [k], take two distinct ai, bi ∈ [n] and restrict the label space to

k

C(2) :=

{ai, bi}, |C(2)| = 2k.

i=1

Since every label combination in [n]k is classified correctly, all tuples in C(2) are classified as well. For each concept i, we can define the induced binary score on only the considered values

h(iai,bi)(z) := hi,ai(z) − hi,bi(z) = (wi,ai − wi,bi)⊺z + (bi,ai − bi,bi). (62) Let wi := wi,ai − wi,bi and βi := bi,ai − bi,bi. Each induced binary classifier defines an affine hyperplane

Hi := {z ∈ Rd | wi⊺z + βi = 0}. (63)

Because multiclass predictions are correct on C(2), each concept i is correctly separated between values ai and bi on this restricted set. Hence the k affine hyperplanes H1, . . . , Hk must separate Rd into at least 2k distinct regions.

But the number of regions formed by k affine hyperplanes in Rd is at most

d

r=0

k r

< 2k whenever d < k (by Theorem 1). (64)

Thus, we must have d ≥ k. Construction is simple. Let d = k, and assume (for ei ∈ Rd with eij = δij)

zc :=

k

i

eici, (65)

i.e. each embedding is a sum of standard basis unit vectors for each concept scaled by the value of that concept. Then we can define probe vectors as

###### wi,j := 2j ei and bi,j := −j2. (66) Then

hi,j(zc) = 2je⊺i zc − j2 = 2ci − j2 = c2i − (j − ci)2. (67) Therefore, for each i, arg maxj∈[n] hi,j(zq) = qi, so all nk combinations are correctly classified in d = k. For intuition of the constructed case, see Section F.1.

| |
|---|

###### F. Examples of linearly compositional models and their geometries

We give two toy geometries that make probe behavior explicit under strong linear-factorization assumptions. They can be understood as lying on two extremes of the space of linear representations: a case of “perfect LRH” geometry, where all the factors of concept values are co-linear (Section F.1), and a case where factors are implied to be orthogonal to any other factor, leading to a maximum possible dimensionality of linearly factorized spaces (Section F.2). These are not claims of full compositional generalization (they do not model the learning algorithm or stability), but provide intuition for probe geometry and dimensionality, especially in the cases where the classification behavior is intuitively desirable (Section F.2).

###### F.1. Case 1: Ideal “LRH” concept classifier

To gain intuition into the geometry of features and linear probes, we consider a commonly-assumed setting of the Linear Representation Hypothesis (LRH) (Elhage et al., 2022; Park et al., 2023; Rajendran et al., 2024), and consider the problem of finding probes capable of classifying the representations. Instead of a joint optimization, we assume the representations are already given and follow a strict linear-factorization structure under LRH.

Specifically, we make the following assumptions: (1) The representation for any input with concept tuple c = (c1, . . . , ck) ∈ [n]k is

k

αi,cidi di ∈ Rd. (68)

zc =

i=1

- (2) The concept direction vectors {di}ki=1 ⊂ Rd are fixed and orthogonal, i.e., d⊺i dℓ = 0 for i ̸= ℓ (hence linearly independent and d ≥ k)3.
- (3) For each concept i, its n values correspond to a known ordered set of scalar coefficients {αi,j}nj=1, with distinct values j for different i. Under these assumptions, orthogonality of the directions implies the representations form a regular grid in z inside V (each coordinate axis is one concept). We consider the problem of finding the linear probes {wi,j} that classify the representations:

exp w⊺i,c

k

zc + bi,ci

. (69)

i

− log

min

n j=1 exp w⊺i,jzc + bi,j

{wi,j,bi,j} c

i=1

We consider the nearest-neighbor classifier. Concretely, we define the approximated concept-value prototypes by averaging representations over all tuples with concept i fixed to value j:

1 nk−1

di,j :=

zc = αi,jdi +

c∈[n]k:ci=j

α¯ℓdℓ := αi,jdi + mi, (70)

ℓ̸=i

where α¯ℓ := n1 nr=1 αℓ,r and mi := ℓ̸=i α¯ℓdℓ. Using these prototypes, we set the corresponding affine probes as4

wi,j = 2 di,j, bi,j = −∥ di,j∥22. (71)

We can verify that such a classifier works. The score for the j-th probe of concept i on an input zc (where the true value for concept i is ci) is:

hi,j(zc) := wi,j⊺ zc + bi,j

= 2 d⊺i,jzc − ∥ di,j∥22

= 2(αi,jdi + mi)⊺zc − ∥αi,jdi + mi∥22

= 2αi,jd⊺i (zc − mi) − αi,j2 ∥di∥22 + 2m⊺i zc − ∥mi∥22 .

- 3We consider this for simplicity; in general one may consider the directions to be linearly independent instead
- 4We find these weights by considering the power diagram (Aurenhammer, 1987) over the points di,j, the weights are biases are then

remaps from the power diagram to the joint argmax classifier.

For every fixed concept i and a datapoint zc, maximizing over j is therefore

2αi,jd⊺i (zc − mi) − αi,j2 ∥di∥22

arg max

hi,j(zc) = arg max j∈[n]

j∈[n]

k

2αi,jd⊺i

αℓ,cℓdℓ − mi − αi,j2 ∥di∥22

= arg max

j∈[n]

ℓ=1

 2αi,jd⊺i

 αi,cidi +

  − αi,j2 ∥di∥22

 

αℓ,cℓdℓ − mi

= arg max

j∈[n]

ℓ̸=i

2αi,jαi,ci∥di∥22 − αi,j2 ∥di∥22

= arg max

j∈[n]

−(αi,j − αi,ci)2∥di∥22

= arg max

j∈[n]

(αi,j − αi,ci)2

= arg min

j∈[n]

= ci.

where we dropped the additive term 2m⊺i zc − ∥mi∥22, since it does not depend on j, and used orthogonality: d⊺i dℓ = 0 for ℓ ̸= i and d⊺i mi = 0. The last step follows since ∥di∥22 > 0 is constant and {αi,j}nj=1 are distinct.

As shown above, such a classifier is easy to construct and correctly classifies all the points in the concept space. If orthogonality did not hold, the nearest neighbor classifier is not guaranteed to be correct. We illustrate the geometry of the probes together with the decision regions in Fig. 16 in two cases, where k = 2, n = 30 and the deviation from orthogonal features varies slightly. We note that the weight vectors, indicated as arrows in the plot, are near-parallel.

[Figure 31]

[Figure 32]

- Figure 16. Probe geometry in the LRH toy setup (k = 2, n = 30). Each row corresponds to a different choice of concept directions (top: mildly non-orthogonal; bottom: closer to orthogonal). Left and middle columns show concept-wise decision regions for concept 1 (blue) and concept 2 (red), with arrows indicating probe directions. The right column overlays both concept families in the joint space; intersections of blue and red stripes form the cells associated with tuples (c1, c2).

###### F.2. Case 2: Ideal “on-off” concept classifier

Here we consider a setting that intuitively could be described as exhibiting behavior that is desirable for compositional generalization. In particular, we consider a CLIP-style linearly compositional model with the constraint that the probe scores hi,j should be constant among the matches, and constant among the mis-matches.

In CLIP-like models, this is usually viewed through cosine similarity. In the normalized view (∥z∥2 = 1 and ∥wi,j∥2 = 1), decision regions on the sphere are spherical caps rather than Euclidean half-spaces. For a cosine-similarity classifier, the decision region for class (i, j) is

Ri,j := z ∈ Sd−1 : wi,j⊺ z > wi,k⊺ z ∀k ̸= j . (72)

We refer to this setting as “on-off” concept classifier, as there are only two possible scores for each probe: α if the concept matches, and β if it does not. That is, exist constants α > β in [−1, 1] such that for all i, j and all tuples c,

wi,j⊺ zc =

α if j = ci β if j ̸= ci.

(73)

We illustrate this setting in Fig. 17.

The key results in this section are two-fold: (1) a model exhibiting this behavior requires at least 1 + k(n − 1) dimensions, meaning that under a large number of concepts and values per concept, even current systems will not be able to reliably distinguish between correct and incorrect matches between a sample and its corresponding concept values (Proposition 6); (2) assuming that the model is able to distinguish between correct and incorrect matches, one can linearly approximate the representations in a way that preserves the “on-off” pattern (Proposition 7).

- Figure 17. Illustration of the “on-off concept classifier” mechanism. (a) A vision input is processed by a neural network to produce an embedding. (b) Each concept (e.g., shape, object color, background color) is probed independently using a set of language model probes, one per possible value. (k) The probe for a given concept yields a high score α if the concept matches and a lower score β otherwise, as formalized in the logit equation at bottom.

- Proposition 6 (Minimal dimensionality from fixed dot-products). Assume k ≥ 2 and n ≥ 2. For each concept i ∈ [k] and value j ∈ [n],

let the probe weight bewi,j ∈ Rd, such that for any tuple c = (c1, . . . , ck) ∈ [n]k there exists a corresponding zc ∈ Rd, and there exist α, β ∈ R with α ̸= β such that the “on-off” pattern holds (and α ̸= −β(n − 1)):

α, j = ci, β, j ̸= ci,

w⊺i,jzc =

for all i, j, c, (74)

Then the ambient dimension d must satisfy

d ≥ 1 + k(n − 1). (75)

Moreover, this bound is tight: for any (α, β) with α ̸= β, there exist explicit probe/representation families that realize (74) in dimension d = 1 + k(n − 1).

Proof. We write the constraints in matrix form. First, we stack probe vectors as rows:

   ∈ Rkn×d, (row (i − 1)n + j = w⊺i,j). (76)

  

w⊺1,1 . w⊺k,n

P =

Then stack representations as columns:

k

nk ∈ Rd×n

. (77) The “on-off” constraints (74) become

X = zc1 · · · zc

k

Y = PX ∈ Rkn×n

, (78) where entries of Y are

α, j = ci, β, j ̸= ci.

, (79)

y(i,j),c =

and will sometimes denote rows of Y as y(i,j) instead. To understand the rank of Y , first note that it is at most kn. Additionally, note that the rows can be divided into k blocks. As we will show, each such block contains up to one dependence that is achieved by summing up the rows within a block.

For each block i ∈ [k] and any column c ∈ [n]k, we can sum the n entries in that block as

n

n

αIci=j + βIci̸=j = α + β(n − 1). (80)

y(i,j),c =

j=1

j=1

Importantly, this sum is the same regardless of c and chosen i. Therefore, each block sums up to the same vector, and is therefore redundant. This implies that the rank can be at most nk − (k − 1) = 1 + k(n − 1). Next, we show that within each block, each row is independent.

For that, we will show that for any concept “block” i ∈ [k] every row (i, j) corresponding to the value j of the i-th concept is linearly independent of all the rows in blocks i′ ̸= i. We denote the span of the other blocks’ rows as

S := span({y(i′,j′)}i′̸=i,j′∈[n]). (81)

For that, we take any two columns c, c′ ∈ [n]k that are identical up to the i-th concept and satisfy cm = c′m for m ̸= i and ci ̸= c′i, and note that all the rows outside the i-th block are identical: y(i′,j′),c = y(i′,j′),c′ for all i′ ̸= i and all j′ ∈ [n], and only differ in the i-th block rows. Because of that, any linear combination from S with some coefficients will have matching columns:

λnsn

= λnsn

c

, (82)

c′

but both c and c′ differ in exactly the i-th concept, and therefore their column values in the i-th block must be different:

y(i,ci),c ̸= y(i,ci),c′, (83)

and therefore no vector in the span can be equal to yi,ci. All that is needed to show now is that taking any linear combination of the rows in the i-th block cannot be expressed by any c ∈ S.

We do this by applying the same reasoning but instead of taking 2 rows we take n rows. Concretely, for the i-th concept we take all n rows: (i, 1), . . . , (i, n) and consider c1, . . . , cn which are all identical up to the i-th concept. Again, for any i′ ̸= i and any row index j ∈ [n], the values are constant across these columns: y(i′,j),cm = y(i′,j),cm′ for all m, m′ ∈ [n], and only the i-th block rows change across c1, . . . , cn. We can thus consider the span of the i-th block among the n rows by taking λ ∈ Rn and considering only the n columns corresponding to c1, . . . , cn, writing the linear combination as

n

k

λjy(i,j) ∈ Rn

. (84)

r :=

j=1

We now evaluate r on the n columns c1, . . . , cn. By construction, in column cm the only row in block i that takes value α is (i, m), while all other (i, j) take value β. Therefore the m-th selected entry is

rm = λmα +

λjβ = λmα + (S − λm)β, (85)

j̸=m

where S := nj=1 λj is the sum of the coefficients. Finally, for r to be within S it has to hold that for any columns m, m′ the entries are equal:

λmα + (S − λm)β = λm′α + (S − λm′)β

(86)

⇒ (λm − λm′)(α − β) = 0.

But by assumption α ̸= β, and therefore λm = λm′ for all m, m′ ∈ [n]. Clearly, if λm ̸= 0, then this results in the constant direction produced by any concept block, which is the only direction that lies in S. Therefore each concept block produces n−1 linearly independent vectors.

Combined with the redundancy within a concept block, this implies that rank(Y ) = 1 + k(n − 1). Finally, since Y = PX,

1 + k(n − 1) = rank(Y ) ≤ rank(P) ≤ d. (87) This proves (75) and shows that such a model requires at least 1 + k(n − 1) dimensions. Tightness follows from an explicit construction by placing probes and representations in, for example, orthogonal directions.

| |
|---|

Below, we provide a numerical example to illustrate the form of the logit matrix Y for the case of two concepts, three values each. Example 1 (Two concepts, three values each: c = 2, n = 3). Set (α, β) = (1, 0.2). The row indices are (i, j) ∈ {1, 2} × {1, 2, 3}, the column indices are the 32 = 9 tuples (v1, v2) ∈ {1, 2, 3}2:

Y =

11 12 13 21 22 23 31 32 33 



- (1, 1) 1 1 1 0.2 0.2 0.2 0.2 0.2 0.2
- (1, 2) 0.2 0.2 0.2 1 1 1 0.2 0.2 0.2
- (1, 3) 0.2 0.2 0.2 0.2 0.2 0.2 1 1 1

(2, 1) 1 0.2 0.2 1 0.2 0.2 1 0.2 0.2 (2, 2) 0.2 1 0.2 0.2 1 0.2 0.2 1 0.2 (2, 3) 0.2 0.2 1 0.2 0.2 1 0.2 0.2 1

 

 

(88)

Note that each block corresponding to either the first or the second concept sum up to the same vector. Within each block there are 2 linearly independent vectors. Hence, rank(Y ) = 5 = 1 + c(n − 1).

Under such a design, linear factorization holds immediately – at least up to projection onto the span of the probe vectors; or exactly if the dimensionality of the embeddings is exactly 1 + k(n − 1). To make the setup explicit, we can also impose a fixed relation between α and β. We avoid the condition of the global mean being zero, as this implies that α + (n − 1)β = 0, i.e., α = −(n − 1)β (similar to Proposition 6). Related normalization choices are discussed in (Lee et al., 2024).

- Proposition 7 (Additive factorization from the on-off pattern). Assume k ≥ 2 and n ≥ 2. For each i ∈ [k] and j ∈ [n], let wi,j ∈ Rd be

a probe vector. For each tuple c = (c1, . . . , ck) ∈ [n]k, let zc ∈ Rd be its corresponding representation. Assume there exist α > β, with α, β ∈ R, such that

α if j = ci, β if j ̸= ci, ∀i, j, c. (89)

wi,j⊺ zc =

Then there exist vectors z¯ ∈ Rd, ui,j ∈ Rd, and reconstructed representations zc ∈ Rd of the additive form

zc = z¯ +

k

ui,ci, ∀c ∈ [n]k, (90)

i=1

such that for every i, j, c,

α if j = ci, β if j ̸= ci.

wi,j⊺ zc = wi,j⊺ zc =

(91)

So zc and zc are indistinguishable to the probes. Proof. We again show that computing the factors as averages satisfies the condition. For that, we define the global mean z¯, the concept factors ai,j, and the centered factors ui,j:

z¯ :=

1 nk

1 nk−1

zc′, ai,j :=

c′∈[n]k

k

zc′, ui,j := ai,j − z¯, zc := z¯ +

ui,ci.

c′:c′i=j

i=1

We will confirm this choice works through a simple calculation. First, for any (i, j) the dot product with the average representation is

1 nk

wi,j⊺ z¯ =

α + (n − 1)β n

nk−1α + (nk − nk−1)β =

=: δ, (92)

Second, for any i′, r, we expand the dot product with the average non-centered factor ai,j:

1 nk−1

wi⊺′,rai,j =

wi⊺′,rzc′. (93)

c′:c′i=j

and decompose it into two cases:

- Case 1: i′ = i (probe and conditioning on the same concept). Here the condition c′i = j fixes whether the target concept is probed:

- • If r = j, every term in the sum equals α, so wi⊺′,rai,j = α.
- • If r ̸= j, every term in the sum equals β, so wi⊺′,rai,j = β.

- Case 2: i′ ̸= i (probe and conditioning on different concepts). Now c′i = j does not constrain c′i′, so:

- • exactly nk−2 tuples satisfy c′i′ = r and contribute α,
- • exactly (n − 1)nk−2 tuples satisfy c′i′ ̸= r and contribute β.

Therefore

nk−2α + (n − 1)nk−2β

wi⊺′,rai,j =

nk−1 = δ. Putting it all together:

 

α, i′ = i, r = j, β, i′ = i, r ̸= j, δ, i′ ̸= i.

wi⊺′,rai,j =



By linearity, subtracting the global mean term gives

 

α − δ, i′ = i, r = j, β − δ, i′ = i, r ̸= j, 0, i′ ̸= i.

wi⊺′,rui,j = wi⊺′,rai,j − wi⊺′,rz¯ =

(94)



We now evaluate the reconstructed representation:

k

wi,j⊺ zc = wi,j⊺ z¯ +

wi,j⊺ ut,ct

t=1

###### = δ + wi,j⊺ ui,ci +

wi,j⊺ ut,ct =0 by (94)

t̸=i

= δ + wi,j⊺ ai,ci − δ = wi,j⊺ ai,ci

α, j = ci, β, j ̸= ci.

=

Hence zc reproduces exactly the same on-off probe responses as zc for every concept and value, which proves (91).

| |
|---|

###### G. Discussion on stability

Here we discuss the role of Stability and what can go wrong without it. Without Stability (Desideratum 3), the only requirement on the readout is that it correctly classifies the observed grid. This places no constraint on how the decision regions are shaped: they can vary arbitrarily across training supports, as long as every grid point “lands” in the correct region. In practice, this means that decision boundaries may pass arbitrarily close to data points, that some concept-value regions may be infinitesimally thin, and that retraining on a different valid support can produce a completely different partition of the space. In short, no robustness holds. As we show below, this pathology does not disappear with scale: even as n → ∞, the decision regions can remain arbitrarily complex, which is never a desirable property. Linear separability alone therefore does not force linear factorization, nor does it force projected R2 to be close to 1 (see also Sections H.2 and H.4).

Stability prevents these pathologies by requiring that the predicted probability vector over concept values agrees at every input, regardless of which valid support was used for training. In the linear-softmax setting, this implies that probe weights are the same across supports (or, up to a shift in non-binary case), pinning down the decision boundaries and forcing the representational structure established in our main results.

Counterexamples to linear factorization even as n → ∞. Suppose we only assume linear separability on the full grid: for concept tuples q = (q1, . . . , qk) ∈ [n]k with embeddings zq ∈ Rd, there exist probes {(wi,j, bi,j)}nj=1 such that

wi,j⊺ zq + bi,j = qi, ∀i ∈ [k], ∀q ∈ [n]k. (95)

arg max

j∈[n]

This guarantees perfect classification but does not enforce Stability across supports. Does this imply additive structure? In general, no. Even in the simple case of d = 2, k = 2, and even as n → ∞, one can construct point clouds {zq1,q2} ⊂ R2 with perfect linear separability that do not admit a decomposition

zq1,q2 = u1,q1 + u2,q2. (96)

Linear separability is therefore compatible with non-factorized geometry. Fig. 18 illustrates this failure mode, which is consistent with the low-R2 counterexamples in Section H.4.

###### (a) (b) (c)

n = 8 =⇒ 64 regions n = 14 =⇒ 196 regions

Impossible factorization

- C1 = 1
- C1 = 2
- C1 = 3
- C1 = 4
- C1 = 5
- C1 = 6

| | |
|---|---|

| | |
|---|---|

x2

C1 = 7

x1 x1 x1

- Figure 18. Linear separability without linear factorization. Two families of affine decision boundaries in R2 (black for concept 1, gray for concept 2) divide the plane into regions, one per pair of concept values. (a,b): with n = 8 and n = 14 levels per concept, the arrangement yields n2 regions (64 and 196). By inserting additional nearly-parallel boundaries, existing regions can be split into arbitrarily thin pieces while maintaining perfect linear separability. (c): No linear factorization can be achieved: whichever factors we pick, the separability of some datapoints is violated.

Concretely, in Fig. 18, panels (a) and (b) show two non-parallel families of decision boundaries whose intersections produce n2 convex cells, one per pair (q1, q2). Since perfect classification only requires each grid point to fall in the correct cell, the geometry of the cells themselves is unconstrained. By adding ε-perturbed boundaries, one can subdivide regions so that some concept values occupy arbitrarily small areas as n grows, while perfect separability is preserved. Without Stability, there is nothing to prevent such degenerate configurations, and linear readout constraints alone do not pin down factorized structure. Handling such cases, e.g., by imposing minimum-area constraints on decision regions, is possible in principle but becomes technical; our framework avoids these pathologies by design through the Stability desideratum.

More broadly, any model aiming for robust compositional generalization will need to prevent such degenerate configurations, whether through exact Stability or approximate forms of it.

###### H. Additional information

In this section we expand on linear factorization, make a note on the non-triviality of linear factorization, and expand on the reasoning of using whitening in measuring linear factorization. In Section H.1 we summarize the overall procedure of measuring linear factorization. In Section H.3 we provide an intuition of linear factorization through a simple example. In Section H.4 we show that linear factorization is not a trivial property of linearly compositional models, and illustrate a few cases where the representation cannot be decomposed into a sum of per-concept components even under perfect classification.

###### H.1. Testing linear factorization

Large pre-trained models may encode information beyond the specific concepts in our dataset. To isolate the conceptual structure, we train per-concept linear probes. For each concept i ∈ [k] and value j, we learn a linear probe wi,j, form the probe matrix W ∈ Rm×d, where m is the number of values across all concepts, and project embeddings onto the joint probe span. We do this by first computing the projection matrix PW and then projecting the embeddings onto the joint probe span.

We report Projected R2 after projecting embeddings onto the probe span. To prevent trivial high scores from dominant directions, we whiten the embeddings by applying PCA and normalizing to unit covariance. We compute metrics on PW z after PCA-whitening, applying the same transform to data and reconstructions. We elaborate on this below.

###### H.2. Whitening in measuring linear factorization

We need to be cautious when assessing the degree of linearity in the representations, otherwise, we may mistake high R2 scores for linear factorization when in fact the representation is not linearly factored. For example, if certain concept values dominate the variance in the representation, the R2 may be inflated. To address this, in the main experiments in Section 5.1 we whiten the representations by applying PCA and normalizing to unit covariance. This ensures that a few dominant directions do not dominate the variance in the representation. If the representations are already linearly factored, this will not affect the R2 score.

We illustrate this through three examples in a hypothetical two-dimensional representation space with two concepts in Fig. 19. In the first case ((a)) the representation is already linearly factored: each embedding is written as a sum of two concept components without noise. This yields an R2 score of 1; whitening does not change the score.

Original space f(x)

- (a)
- (b)
- (c)

R2=1.000

Dim2

Original space f(x)

R2=0.813

Dim2

Original space f(x)

R2=0.991

| | | |
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |

Dim2

Dim 1

Whitened space

R2=1.000

Dim2

Whitened space

R2=0.509

Dim2

| | |
|---|---|
| | |
| | |

Whitened space

|R2=0.564|
|---|

Dim2

Dim 1

- Figure 19. Whitening in measuring linear factorization. The representation is not linearly factored, but the R2 score is high due to the dominance of the dominant direction.

In the second case ((b)) the representation is partly linear, but the noise ϵij, independent of the concept values, dominates the overall

variance. Since the scale of the noise is generally lower than the scale of the first concept component, the R2 score is high at 0.813. Whitening, however, removes the dominant direction, and the R2 score drops to 0.509.

Lastly, in the third case ((k)) the representation does not express any information about the second concept, yet the R2 score is still high at 0.991. Again, whitening reveals the underlying issue and changes the score to 0.564 due to the noise in the embeddings.

###### H.3. Intuition of linear factorization

We measure the extent of linearity present in the embeddings through the R2 score. Intuitively, the score quantifies how well the representation can be decomposed into a sum of per-concept components. Recall from Definition 1 that we assume a presence of k concepts, each of which can take any of the n values. A value of R2 = 1 indicates that the representation can be perfectly decomposed into a sum of per-concept components.

We illustrate a few examples to give intuition. We consider a two-dimensional representation space with two concepts (k = 2). In the first case, we consider a case of 24 values per concept (n = 24). In the second case, we consider a case of 6 values per concept (n = 6). In both cases the reported R2 are w.r.t. the whitened space.

The first case (Fig. 20, (a)) exhibits perfect linearity in the embeddings with R2 = 1. In this case, the n2 = 242 = 576 can be perfectly generated using only 2 · 24 = 48 vectors in R2. The second and third columns of the plot show the approximations of the underlying factors u0,i, u1,j, i, j ∈ [n]. As expected, using these approximate factors allows us to perfectly reconstruct the representation, shown in the fourth column.

The second case (Fig. 20, (b)) exhibits lower degree of linearity with R2 = 0.53. As such, we cannot perfectly reconstruct the representation using only the approximate factors, as shown in the last column of the plot.

- (a)
- (b)

Embeddings

Appx. Factors (0)

Appx. Factors (1)

Reconstruction

Overlay

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

Embeddings

Appx. Factors (0)

Appx. Factors (1)

Reconstruction

Overlay

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|Actual<br><br>Reconstruction|
|---|

Figure 20. Intuition of linear factorization. In (a) the representations can be perfectly reconstructed by a set of per-concept components, while in (b) they are insufficient to reconstruct the representation. Refer to the text for more details.

###### H.4. Non-triviality of linear factorization of linearly compositional models

Recall that linearly compositional models (though not necessarily generalizable ones), as defined in Definition 3, admit a set of probes that can perfectly classify all inputs in the grid C. Proposition 1 shows that linearly compositional models must exhibit linear factorization. This naturally raises the converse question: does the mere existence of a set of perfect linear classifiers imply linear factorization? We answer in the negative.

The intuition is as follows. As per Desideratum 1, linearly compositional models need to divide the representation space into all possible combinations of concept values, nk of them. Each region within the nk partitions must contain the corresponding combination of concept values. Under linear factorization, the degrees of freedom of the embeddings within each cell are low, yielding an R2 score of 1. However, even if linear factorization initially holds, the embeddings can generally be perturbed to violate the linear factorization constraint while still being contained within the correct cell.

To illustrate this point, we consider two general cases: (i) the number of concepts is equal to the dimension of the embeddings (k = d), and (ii) the number of concepts is less than the dimension of the embeddings (k < d). As detailed in Section 4.3, case (i) is tight (the dimension

cannot be further reduced), while case (ii) is not. In both cases we assume two concepts and an embedding space that admits two linear probes, one for each concept. Additionally, in both cases we illustrate separately the argmax regions where a certain concept value is predicted (Ri,j, i ∈ [2], j ∈ [n]), and the region where a certain combination of concept values is predicted (R0,j ∩ R1,k, j, k ∈ [n], as per Desideratum 1).

The first concept values’ regions in the embedding space are shown in blue, while the second concept values’ regions are shown in orange.

- Case (i): k = d. In Fig. 21, (a), (b) we show two cases that exhibit perfect linear classification. In (a) a few outliers violate the linearity of the representation, which is also reflected in the R2 = 0.53 < 1. In (b) the argmax regions are highly irregular, and the majority of the embeddings are almost intersecting the decision boundaries, resembling an extremely brittle embedding space susceptible to adversarial attacks, though the classification accuracy is still 100%.
- Case (ii): k < d. In Fig. 21, (k), (d), (e) we show three cases that exhibit perfect linear classification, but with linearity scores ranging from R2 = 0.32 to R2 = 0.83. Because of the higher degrees of freedom, the embeddings enjoy even more space to be perturbed while still exhibiting perfect linear classification.

Overall, these points illustrate that linear factorization is not a trivial property of linearly compositional models, even when perfect classification holds.

[Figure 33]

- (a)
- (b)

R2 = 0.53

+ =

R2 = 0.74

+ =

R2 = 0.32

- (c)
- (d)
- (e)

+ =

R2 = 0.55

+ =

R2 = 0.83

+ =

- Figure 21. Counterexamples of linear factorization under perfect classification. The two concepts are linearly separable, but the representation cannot be decomposed into a sum of two concept components. Each subfigure shows the embedding space overlaid with three columns of argmax regions: the first column shows R0,i, i ∈ [n] (shown in blue), the regions where the first concept values are predicted; the second column shows R1,j, j ∈ [n] (shown in orange), the regions where the second concept values are predicted; and the third column shows R0,i ∩ R1,j, i, j ∈ [n], the joint argmax regions where specific combinations of concept values are predicted. (a), (b) show embeddings for two concepts (color and shape) in R2 (k = d = 2). (d), (e) show embedding points colored by the first concept value, all for two concepts in R3 (k = 2, d = 3). See text for details.

Table 2. Datasets used in our main experiments. Here k is the number of concepts, ni is the number of values for concept i, and N is the number of samples after the preprocessing described in the Notes column.

Dataset k Values per concept (ni) N Notes (main-experiment preprocessing) PUG-Animal 4 background(64), character(68),

39,168 We fix camera-yaw=0, drop the default charac-

ter texture, and remove the Goldfish character. dSprites 6 color(10), shape(3), scale(6),

scale(3), texture(3)

180,000 We keep orientations in [0◦, 90◦) (10 bins), downsample posX/posY to 10 bins each, and add 10 colors via on-the-fly coloring.

orient(10), posX(10), posY(10)

MPI3D 7 obj-color(6), obj-shape(6), obj-size(2), cam-height(3), bg-color(3), horiz(10), vert(10)

64,800 We use the real-world complex variant (original factor sizes [6,6,2,3,3,40,40]) and downsample the horizontal/vertical axes to 10 bins each.

Table 3. Vision models used in our experiments. Family Model identifier Pretraining tag Params (M) Hidden

openaiclip

clip ViT-B/32 openai 151.3M 512 clip ViT-L/14 openai 427.6M 768 openclip

metaclip2

openclip ViT-H/14 (MetaCLIP2 Worldwide, 378px) metaclip2_worldwide 1859.4M 1024 openclip ViT-H/14 (MetaCLIP2 Worldwide, QuickGELU) metaclip2_worldwide 1858.8M 1024 openclip ViT-G/14 (bigG, MetaCLIP2 Worldwide) metaclip2_worldwide 3630.4M 1280 openclip ViT-G/14 (bigG, MetaCLIP2 Worldwide, 378px) metaclip2_worldwide 3631.2M 1280

siglip2

openclip SigLIP2-B/16 (224) webli 375.2M 768 openclip SigLIP2-B/16 (256) webli 375.2M 768 openclip SigLIP2-L/16 (256) webli 881.5M 1024 openclip SigLIP2-L/16 (384) webli 881.9M 1024 openclip SigLIP2-SO400M/16 (256) webli 1135.7M 1152

siglip

openclip SigLIP-B/16 (224) webli 203.2M 768 openclip SigLIP-B/16 (256) webli 203.2M 768 openclip SigLIP-L/16 (256) webli 652.2M 1024

openclip

openclip ViT-B/16 laion400m_e32 149.6M 512 openclip ViT-B/32 laion400m_e32 151.3M 512 openclip ViT-L/14 laion2b_s32b_b82k 427.6M 768 dino

dinov3

dino DINOv3 ViT-S/16 timm_default 21.6M 384 dino DINOv3 ViT-B/16 timm_default 85.6M 768 dino DINOv3 ViT-L/16 timm_default 303.1M 1024

dinov2

dino DINOv2 ViT-S/14 timm_default 22.1M 384 dino DINOv2 ViT-B/14 timm_default 86.6M 768 dino DINOv2 ViT-L/14 timm_default 304.4M 1024

dinov1

dino DINO ViT-S/16, 224px timm_default 21.7M 384 dino DINO ViT-B/16, 224px timm_default 85.8M 768 timm

metaclip

timm ViT-B/16 CLIP (MetaCLIP 2.5B, 224px) timm_default 86.2M 768 timm ViT-B/32 CLIP (MetaCLIP 2.5B, 224px) timm_default 87.8M 768 timm ViT-L/14 CLIP (MetaCLIP 2.5B, 224px) timm_default 304.0M 1024

- (a)
- (b)
- (c)

color=5 shape=1.0 (id=0) size=0.6 (id=1) orientation=0.0° (id=0)

color=3 shape=1.0 (id=0) size=1.0 (id=5) orientation=18.0° (id=2) posx=10 (id=10) posy=28 (id=28)

color=8 shape=1.0 (id=0) size=1.0 (id=5) orientation=0.0° (id=0) posx=24 (id=24) posy=28 (id=28)

color=1 shape=3.0 (id=2) size=0.6 (id=1) orientation=45.0° (id=5)

color=8 shape=2.0 (id=1) size=1.0 (id=5) orientation=72.0° (id=8)

- posx=10 (id=10)
- posy=21 (id=21)

- posx=17 (id=17)
- posy=0.0 (id=0)

- posx=24 (id=24)
- posy=24 (id=24)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

color=0 shape=1.0 (id=0) size=0.9 (id=4) orientation=0.0° (id=0)

color=7 shape=3.0 (id=2) size=1.0 (id=5) orientation=45.0° (id=5)

color=1 shape=1.0 (id=0) size=0.5 (id=0) orientation=54.0° (id=6)

color=7 shape=2.0 (id=1) size=0.9 (id=4) orientation=36.0° (id=4) posx=0.7741935483870968 (id=7) posy=0.0 (id=0)

color=8 shape=2.0 (id=1) size=0.7 (id=2) orientation=27.0° (id=3)

- posx=21 (id=21)
- posy=10 (id=10)

- posx=21 (id=21)
- posy=0.0 (id=0)

- posx=10 (id=10)
- posy=0.0 (id=0)

- posx=10 (id=10)
- posy=21 (id=21)

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

object-color=2 object-shape=2 object-size=0 camera-height=1 background-color=1 horizontal-axis=26 vertical-axis=39

object-color=2 object-shape=2 object-size=1 camera-height=2 background-color=1 horizontal-axis=4 vertical-axis=35

object-color=3 object-shape=0 object-size=1 camera-height=2 background-color=2 horizontal-axis=9 vertical-axis=39

object-color=1 object-shape=3 object-size=0 camera-height=1 background-color=0 horizontal-axis=39 vertical-axis=39

object-color=0 object-shape=5 object-size=0 camera-height=1 background-color=0 horizontal-axis=39 vertical-axis=9

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

object-color=4 object-shape=3 object-size=0 camera-height=0 background-color=0 horizontal-axis=17 vertical-axis=35

object-color=1 object-shape=4 object-size=1 camera-height=1 background-color=1 horizontal-axis=30 vertical-axis=39

object-color=1 object-shape=0 object-size=0 camera-height=0 background-color=0 horizontal-axis=17 vertical-axis=13

object-color=0 object-shape=0 object-size=1 camera-height=1 background-color=1 horizontal-axis=22 vertical-axis=4

object-color=0 object-shape=1 object-size=0 camera-height=1 background-color=0 horizontal-axis=9 vertical-axis=17

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

world-name=ModernGallery character-name=Anlylosaurus character-scale=0.7 character-texture=Sky camera-yaw=0

world-name=Mansion character-name=Zebra character-scale=0.7 character-texture=Sky camera-yaw=0

world-name=Forge character-name=Orca character-scale=0.7 character-texture=Grass camera-yaw=0

world-name=Bridge character-name=Salmon character-scale=1.0 character-texture=Sky camera-yaw=0

world-name=Battleground character-name=Secretarybird character-scale=1.3 character-texture=Grass camera-yaw=0

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

world-name=BoxingRing character-name=Ayu character-scale=0.7 character-texture=Sky camera-yaw=0

[Figure 59]

world-name=Circus character-name=Vulture character-scale=0.7 character-texture=Sky camera-yaw=0

[Figure 60]

world-name=RuralAustralia character-name=Swan character-scale=1.3 character-texture=Grass camera-yaw=0

[Figure 61]

world-name=Ruins character-name=Cattle character-scale=1.0 character-texture=Asphalt camera-yaw=0

[Figure 62]

world-name=BusStationExterior character-name=BlackRockFish character-scale=0.7 character-texture=Sky camera-yaw=0

[Figure 63]

###### Figure 22. Example samples from the main datasets used in our experiments. (a) dSprites, (b) MPI3D, (c) PUG-Animal.

- H.5. Models and probing

###### I. Additional experimental results

In this section we provide additional experimental results discussed in the main text. We used (Rubinstein & Uselis, 2025) for conducting the experiments.

###### I.1. Linearity and compositional generalization

We provide an extended view of the relationship between projected linearity (R2) and compositional generalization under different train/test splits. We follow the same pipeline as in Section 5.2: for each dataset/model, we train one linear probe per concept and evaluate average compositional accuracy on held-out concept combinations. The full split-ablation results are shown in Fig. 23: across held-out fractions, higher projected R2 is still associated with higher held-out accuracy, while the trend flattens in near-ceiling regimes (especially on dSprites and MPI3D).

For each split level, we set a test fraction ρtest ∈ {0.95, 0.90, 0.50, 0.30, 0.10} and train on the remaining 1 − ρtest fraction. For every model/split setting, probes are trained with Adam for 1000 epochs using cosine decay to zero, and we select the best result over learning rates {10−3, 10−2}.

- (a) 0.95
- (b) 0.90
- (c) 0.5
- (d) 0.3
- (e) 0.1

PUG-Animal

DSprites

###### MPI3D

0.975

0.85

Accuracy

0.95

0.950

0.80

Random

Random

Random

0.925

0.75

0.90

0.47

0.40

0.60

0.70

0.42

0.12

0.17

0.900

0.50 0.55 0.60 0.65 R2

0.38 0.40 0.42 0.44 R2

0.44 0.46 0.48 0.50 R2

PUG-Animal

DSprites

###### MPI3D

1.00

0.9

0.98

Accuracy

0.96

0.95

Random

Random

Random

0.8

0.94

0.51

0.42

0.63

0.90

0.42

0.12

0.17

0.50 0.55 0.60 0.65 R2

0.38 0.40 0.42 0.44 R2

0.44 0.46 0.48 0.50 0.52 R2

PUG-Animal

DSprites

###### MPI3D

1.00

1.00

0.95

0.98

Accuracy

0.90

0.98

0.96

Random

Random

Random

0.85

0.59

0.45

0.68

0.94

0.96

0.80

0.42

0.13

0.17

0.50 0.55 0.60 0.65 R2

0.38 0.40 0.42 0.44 R2

0.46 0.48 0.50 0.52 0.54 R2

PUG-Animal

DSprites

###### MPI3D

1.00

1.00

0.95

Accuracy

0.98

0.90

0.98

Random

Random

Random

0.96

0.85

0.61

0.46

0.70

0.94

0.96

0.80

0.42

0.13

0.17

0.50 0.55 0.60 0.65 R2

0.38 0.40 0.42 0.44 R2

0.475 0.500 0.525 0.550 R2

PUG-Animal

DSprites

###### MPI3D

1.00

1.00

0.95

0.98

Accuracy

0.90

0.98

Random

Random

Random

0.96

0.85

0.62

0.46

0.70

0.94

0.96

0.42

0.13

0.17

0.80

0.50 0.55 0.60 0.65 R2

0.38 0.40 0.42 0.44 R2

0.475 0.500 0.525 0.550 R2

OpenAI CLIP OpenCLIP MetaCLIP MetaCLIP2 SigLIP SigLIP2 DINOv1 DINOv2 DINOv3

- Figure 23. Projected R2 vs compositional accuracy across split regimes. Columns correspond to datasets (PUG-Animal, dSprites,

MPI3D). Rows (a)–(e) correspond to different held-out test fractions ρtest (shown next to each row). The x-axis is projected R2 and the y-axis is compositional accuracy on held-out combinations. Each marker is a model checkpoint (legend); the inset in each panel reports the randomly initialized OpenCLIP baseline for the same dataset/split. Across split regimes, higher projected R2 is consistently associated with higher held-out compositional accuracy.

###### I.2. Orthogonality of factors

Setup. For each dataset/model, we extract image embeddings zc and recover concept factors {ui,v} as in Section 5.1. For orthogonality, we compute all quantities in the original embedding space (without projection by PW ). For each concept i, we define centered and normalized factor directions:

1 |Ci| v∈C

ui,v, (97)

u¯i :=

i

di,v ∥di,v∥

di,v := ui,v − u¯i, d˜i,v :=

. (98)

Metrics. We measure orthogonality via absolute cosine between direction vectors (lower | cos | ⇒ greater orthogonality). For any concepts i ̸= j, we define

1 |Ci||Cj| a∈C

d ˜⊺i,ad˜j,b ,

Orth(i, j) :=

i b∈Cj

1 |Ci|(|Ci| − 1) a,b∈C

###### d ˜⊺i,ad˜i,b .

Orth(i, i) :=

i

a̸=b

We report Orth(i, i) as within-concept direction similarity and Orth(i, j) (for i ̸= j) as across-concept direction similarity. In this parameterization, stronger cross-concept orthogonality corresponds to lower Orth(i, j). Results. In Fig. 24, we provide the full per-model/per-dataset orthogonality matrices (including a randomly-initialized baseline) across the three datasets used in the main text.

DSprites

Random

CLIP ViT-L/14

OCLIP ViT-L/14

SigLIP-L/16

SigLIP2-S/O

###### DINO2-B/14

###### DINO3-B/16

color shape size orientation

|[Figure 64]<br><br>0.52 0.32 0.36 0.05 0.11 0.09<br><br>0.32 0.72 0.85 0.17 0.17 0.15<br><br>0.36 0.85 0.99 0.06 0.18 0.13<br><br>0.05 0.17 0.06 0.62 0.20 0.20<br><br>0.11 0.17 0.18 0.20 0.59 0.05<br><br>0.09 0.15 0.13 0.20 0.05 0.59|
|---|

|[Figure 65]<br><br>0.17 0.05 0.09 0.09 0.14 0.10<br><br>0.05 0.50 0.19 0.21 0.05 0.10<br><br>0.09 0.19 0.89 0.19 0.26 0.09<br><br>0.09 0.21 0.19 0.43 0.13 0.07<br><br>0.14 0.05 0.26 0.13 0.76 0.24<br><br>0.10 0.10 0.09 0.07 0.24 0.59<br>|
|---|

|[Figure 66]<br><br>0.16 0.06 0.08 0.08 0.07 0.07<br><br>0.06 0.50 0.05 0.22 0.08 0.11<br><br>0.08 0.05 0.93 0.10 0.08 0.06<br><br>0.08 0.22 0.10 0.37 0.11 0.12<br><br>0.07 0.08 0.08 0.11 0.54 0.15<br><br><br>0.07 0.11 0.06 0.12 0.15 0.59|
|---|

|[Figure 67]<br><br>0.20 0.05 0.04 0.06 0.06 0.04<br><br>0.05 0.48 0.17 0.28 0.09 0.07<br><br>0.04 0.17 0.94 0.12 0.10 0.04<br><br>0.06 0.28 0.12 0.45 0.09 0.05<br><br><br>0.06 0.09 0.10 0.09 0.60 0.13<br><br>0.04 0.07 0.04 0.05 0.13 0.62|
|---|

|[Figure 68]<br><br>0.27 0.06 0.08 0.08 0.09 0.10<br><br>0.06 0.49 0.17 0.25 0.07 0.10<br><br>0.08 0.17 0.85 0.17 0.25 0.05<br><br>0.08 0.25 0.17 0.44 0.10 0.10<br>0.09 0.07 0.25 0.10 0.62 0.18<br>0.10 0.10 0.05 0.10 0.18 0.60<br>|
|---|

|[Figure 69]<br><br>0.33 0.05 0.05 0.07 0.06 0.04<br><br>0.05 0.50 0.20 0.15 0.08 0.08<br><br>0.05 0.20 0.92 0.10 0.09 0.08<br><br>0.07 0.15 0.10 0.41 0.07 0.04<br><br>0.06 0.08 0.09 0.07 0.57 0.12<br><br><br>0.04 0.08 0.08 0.04 0.12 0.66|
|---|

|[Figure 70]<br><br>0.34 0.04 0.04 0.06 0.03 0.05<br><br>0.04 0.50 0.18 0.12 0.04 0.04<br><br>0.04 0.18 0.95 0.06 0.07 0.02<br><br>0.06 0.12 0.06 0.43 0.03 0.05<br><br>0.03 0.04 0.07 0.03 0.70 0.05<br><br>0.05 0.04 0.02 0.05 0.05 0.78<br>|
|---|

- posx
- posy

colorshapeorientationsizeposxposy

colorshapeorientationsizeposxposy

colorshapeorientationsizeposxposy

colorshapeorientationsizeposxposy

colorshapeorientationsizeposxposy

colorshapeorientationsizeposxposy

colorshapeorientationsizeposxposy

PUG-Animal

Random

CLIP ViT-L/14

OCLIP ViT-L/14

SigLIP-L/16

SigLIP2-S/O

###### DINO2-B/14

###### DINO3-B/16

|[Figure 71]<br><br>0.30 0.34 0.37 0.29<br><br>0.34 0.55 0.51 0.53<br><br>0.37 0.51 0.57 0.42<br><br>0.29 0.53 0.42 0.49|
|---|

|[Figure 72]<br><br>0.28 0.05 0.31 0.08<br><br>0.05 0.15 0.08 0.07<br><br>0.31 0.08 0.77 0.14<br><br>0.08 0.07 0.14 0.50|
|---|

|[Figure 73]<br><br>0.30 0.06 0.37 0.09<br><br>0.06 0.16 0.10 0.08<br><br>0.37 0.10 0.83 0.19<br><br>0.09 0.08 0.19 0.50|
|---|

|[Figure 74]<br><br>0.25 0.05 0.30 0.08<br><br>0.05 0.18 0.09 0.07<br><br>0.30 0.09 0.83 0.20<br><br>0.08 0.07 0.20 0.50|
|---|

|[Figure 75]<br><br>0.24 0.05 0.28 0.09<br><br>0.05 0.17 0.10 0.06<br><br>0.28 0.10 0.85 0.16<br><br>0.09 0.06 0.16 0.50|
|---|

|[Figure 76]<br><br>0.21 0.05 0.18 0.06<br><br>0.05 0.13 0.11 0.06<br><br>0.18 0.11 0.71 0.11<br><br>0.06 0.06 0.11 0.50<br>|
|---|

|[Figure 77]<br><br>0.30 0.04 0.33 0.07<br><br>0.04 0.15 0.06 0.05<br><br>0.33 0.06 0.70 0.10<br><br>0.07 0.05 0.10 0.50|
|---|

character world size texture

characterworldsizetexture

characterworldsizetexture

characterworldsizetexture

characterworldsizetexture

characterworldsizetexture

characterworldsizetexture

characterworldsizetexture

###### MPI3D

Random

CLIP ViT-L/14

OCLIP ViT-L/14

SigLIP-L/16

SigLIP2-S/O

###### DINO2-B/14

DINO3-B/16

object-color object-shape object-size

|[Figure 78]<br><br>0.49 0.34 0.31 0.37 0.53 0.29 0.19<br><br>0.34 0.76 0.86 0.38 0.23 0.51 0.24<br><br>0.31 0.86 1.00 0.37 0.18 0.55 0.27<br><br>0.37 0.38 0.37 0.54 0.46 0.24 0.14<br><br>0.53 0.23 0.18 0.46 0.49 0.14 0.16<br><br>0.29 0.51 0.55 0.24 0.14 0.46 0.27<br><br>0.19 0.24 0.27 0.14 0.16 0.27 0.37|
|---|

|[Figure 79]<br><br>0.24 0.10 0.06 0.07 0.08 0.07 0.07<br><br>0.10 0.42 0.22 0.04 0.05 0.08 0.06<br><br>0.06 0.22 1.00 0.15 0.06 0.19 0.08<br>0.07 0.04 0.15 0.48 0.10 0.04 0.06<br>0.08 0.05 0.06 0.10 0.53 0.08 0.06<br><br><br>0.07 0.08 0.19 0.04 0.08 0.53 0.11<br><br>0.07 0.06 0.08 0.06 0.06 0.11 0.43|
|---|

|[Figure 80]<br><br>0.27 0.05 0.07 0.03 0.08 0.04 0.09<br><br>0.05 0.45 0.13 0.04 0.03 0.07 0.06<br><br>0.07 0.13 1.00 0.18 0.04 0.21 0.05<br><br>0.03 0.04 0.18 0.48 0.09 0.13 0.07<br><br>0.08 0.03 0.04 0.09 0.74 0.05 0.07<br><br>0.04 0.07 0.21 0.13 0.05 0.54 0.14<br><br><br>0.09 0.06 0.05 0.07 0.07 0.14 0.45<br>|
|---|

|[Figure 81]<br><br>0.28 0.07 0.05 0.09 0.07 0.07 0.08<br><br>0.07 0.44 0.14 0.05 0.06 0.10 0.07<br><br>0.05 0.14 1.00 0.18 0.08 0.32 0.13<br><br>0.09 0.05 0.18 0.47 0.11 0.11 0.09<br><br>0.07 0.06 0.08 0.11 0.67 0.05 0.06<br><br>0.07 0.10 0.32 0.11 0.05 0.52 0.16<br>0.08 0.07 0.13 0.09 0.06 0.16 0.45<br>|
|---|

|[Figure 82]<br><br>0.27 0.05 0.11 0.09 0.09 0.08 0.10<br><br>0.05 0.49 0.19 0.03 0.03 0.08 0.06<br><br>0.11 0.19 1.00 0.06 0.06 0.29 0.16<br><br>0.09 0.03 0.06 0.48 0.08 0.05 0.08<br><br>0.09 0.03 0.06 0.08 0.47 0.03 0.04<br><br>0.08 0.08 0.29 0.05 0.03 0.52 0.15<br><br>0.10 0.06 0.16 0.08 0.04 0.15 0.45<br>|
|---|

|[Figure 83]<br><br>0.38 0.07 0.11 0.07 0.09 0.08 0.07<br><br>0.07 0.52 0.20 0.06 0.04 0.04 0.06<br><br>0.11 0.20 1.00 0.07 0.10 0.20 0.08<br><br>0.07 0.06 0.07 0.65 0.13 0.11 0.17<br><br>0.09 0.04 0.10 0.13 0.49 0.04 0.04<br><br>0.08 0.04 0.20 0.11 0.04 0.53 0.13<br><br><br>0.07 0.06 0.08 0.17 0.04 0.13 0.49|
|---|

|[Figure 84]<br><br>0.36 0.05 0.06 0.06 0.07 0.08 0.06<br><br>0.05 0.49 0.31 0.05 0.03 0.08 0.08<br>0.06 0.31 1.00 0.17 0.04 0.18 0.06<br><br><br>0.06 0.05 0.17 0.46 0.13 0.05 0.11<br>0.07 0.03 0.04 0.13 0.49 0.03 0.08<br>0.08 0.08 0.18 0.05 0.03 0.55 0.10<br><br><br>0.06 0.08 0.06 0.11 0.08 0.10 0.43|
|---|

camera-height background-color

horizontal-axis vertical-axis

object-colorobject-shapecamera-heightobject-sizebackground-colorhorizontal-axisvertical-axis

object-colorobject-shapecamera-heightobject-sizebackground-colorhorizontal-axisvertical-axis

object-colorobject-shapecamera-heightobject-sizebackground-colorhorizontal-axisvertical-axis

object-colorobject-shapecamera-heightobject-sizebackground-colorhorizontal-axisvertical-axis

object-colorobject-shapecamera-heightobject-sizebackground-colorhorizontal-axisvertical-axis

object-colorobject-shapecamera-heightobject-sizebackground-colorhorizontal-axisvertical-axis

object-colorobject-shapecamera-heightobject-sizebackground-colorhorizontal-axisvertical-axis

- Figure 24. Full orthogonality matrices across models and datasets. Each entry reports mean absolute cosine between centered factor directions. Diagonal blocks correspond to within-concept similarity; off-diagonal blocks correspond to across-concept similarity. Lower off-diagonal values indicate stronger cross-concept orthogonality.

We summarize the same trend in aggregate form in Fig. 25, comparing within-concept and across-concept values directly.

| | |
|---|---|
| |0.67<br><br>0.55 0.51 0.55 0.54 0.57<br><br>0.61<br><br>0.53<br><br>0.48 0.45 0.44 0.44 0.44<br><br>0.59<br><br>0.52<br><br>0.56 0.55 0.53<br><br>0.58 0.54 0.52|
| |0.21<br><br>0.13 0.10 0.09 0.12 0.08 0.06 0.09<br><br>0.42 0.39 0.41<br><br>0.41<br><br>0.12 0.15 0.13 0.12 0.10 0.11 0.15<br><br>0.33<br><br>0.09 0.08 0.10 0.09 0.09 0.09 0.09|

RandomCLIPViT-L/14OCLIPViT-L/14SigLIP-L/16SigLIP2-S/ODINO2-B/14DINO3-B/16MetaCLIP-H/14RandomCLIPViT-L/14OCLIPViT-L/14SigLIP-L/16SigLIP2-S/ODINO2-B/14DINO3-B/16MetaCLIP-H/14RandomCLIPViT-L/14OCLIPViT-L/14SigLIP-L/16SigLIP2-S/ODINO2-B/14DINO3-B/16MetaCLIP-H/14

0.0

0.5

1.0

Cosinesimilarity

DSprites PUG-Animal MPI3D

Same concept Other concepts

- Figure 25. Aggregated orthogonality summary. Bars compare within-concept direction similarity and across-concept similarity; acrossconcept values are consistently lower, matching the orthogonality pattern predicted by the theory.

###### I.3. Dimensionality of factors

This subsection provides the full model dimensionality results that complement Section 5.4. For each model and dataset, we estimate per-concept effective dimensionality as the minimum number of PCA components needed to explain 95% variance in recovered factors, and report it relative to concept cardinality. The complete per-model plots are shown in Fig. 26.

DINOv1_ViT-S_16

DINOv1_ViT-B_16

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>5/10<br><br>43/64 2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3<br><br>5/10|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>23/69 1/3<br><br>4/10|

| |2/3<br><br>6/10<br><br>47/64<br><br>2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10<br><br>25/69 1/3<br><br>4/104/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(a) DINOv1 ViT-S/16

(b) DINOv1 ViT-B/16

DINOv2_ViT-S_14

DINOv2_ViT-L_14

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>5/10<br><br>40/64 2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>30/69<br><br>1/3<br><br>4/104/10|

| |2/3<br><br>6/10<br><br>37/69<br><br>43/64 2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>1/3<br><br>4/104/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(c) DINOv2 ViT-S/14

(d) DINOv2 ViT-L/14

DINOv3_ViT-S_16

DINOv3_ViT-L_16

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>5/10<br><br>41/64 2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>20/69<br><br>1/3<br><br>4/104/10|

| |2/3<br><br>6/10<br><br>43/64 2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>25/69 1/3<br><br>4/104/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(e) DINOv3 ViT-S/16

(f) DINOv3 ViT-L/16

OpenAI_CLIP_RN50

OpenAI_CLIP_ViT-L_14

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>8/10<br><br>39/64<br><br>2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10 16/69<br><br>1/3<br><br>4/104/10|

| |2/3<br><br>8/10<br><br>43/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3<br><br>5/10|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>23/69 1/3<br><br>4/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(g) OpenAI CLIP RN50

(h) OpenAI CLIP ViT-L/14

- Figure 26. Dimensionality results computed as the number of SVD factors required to reach 95% explained variance, per dataset, across representative vision(-language) backbones.

OpenCLIP_ViT-B_16_(LAION-400M)

OpenCLIP_ViT-L_14_(LAION-2B)

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>8/10<br><br>43/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>20/69<br><br>1/3<br><br>4/104/10|

| |2/3<br><br>8/10<br><br>42/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10<br><br>22/69 1/3<br><br>4/104/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(i) OpenCLIP ViT-B/16 (LAION-400M)

(j) OpenCLIP ViT-L/14 (LAION-2B)

MetaCLIP_2.5B_ViT-B_32

MetaCLIP_2.5B_ViT-L_14

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>8/10<br><br>45/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>19/69<br><br>1/3<br><br>4/104/10|

| |2/3<br><br>8/10<br><br>42/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10<br><br>23/69 1/3<br><br>4/104/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(k) MetaCLIP 2.5B ViT-B/32

(l) MetaCLIP 2.5B ViT-L/14

MetaCLIP2_ViT-H_14_(QuickGELU)

MetaCLIP2_ViT-bigG_14

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>8/10<br><br>42/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3<br><br>5/10|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10<br><br>24/69 1/3<br><br>4/10|

| |2/3<br><br>8/10<br><br>41/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>2/102/10<br><br>26/69<br><br>1/3<br><br>4/104/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(m) MetaCLIP2 ViT-H/14 (QuickGELU)

(n) MetaCLIP2 ViT-bigG/14

DSprites PUG-Animal MPI3D

DSprites PUG-Animal MPI3D

Dim./Num.Classes

Dim./Num.Classes

1.0

1.0

| |2/3<br><br>8/10<br><br>40/64 2/3<br><br>5/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10<br><br>26/69<br><br>1/3<br><br>4/104/10|

| |2/3<br><br>6/10 40/64 2/3 4/6<br><br>3/6 1/2<br><br>2/3 2/3|
|---|---|
| |1/6<br><br>4/10<br><br>3/10<br><br>2/10<br><br>27/69<br><br>1/3<br><br>4/104/10|

0.5

0.5

0.0

0.0

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

shapecolororientationsize posxposycharacterworldsizetextureobject-colorobject-shapeobject-sizecamera-heightbackground-colorhorizontal-axisvertical-axis

(o) SigLIP-L/16

(p) SigLIP2 SO400M/16 (256) Figure 26. Dimensionality results (continued from Fig. 26).

[Figure 85]

###### Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models

##### SizevsshapeSize

Figure 27. Geometry of datapoints in OpenCLIP ViT-L/14. We show the span of the joint features of OpenCLIP ViT-L/14.

###### I.4. Experiments using text encoders as probes (zero-shot results)

In the main text (Section 5.1), we analyzed the factors of the models by training linear probes on the image embeddings using gradient descent with cross-entropy. This was done for two reasons: (1) to handle concepts that are difficult to express as text prompts (e.g., visually complex backgrounds or continuous attributes like size or orientation), and (2) to avoid potential misalignment between the text and vision modalities, where the text encoder must accommodate many visual categories, potentially leading to suboptimal performance for certain domains. Here, we ask what happens when we do not take into account these problems and instead rely on the linear probes that the text encoder already produces.

In this section, we provide analogous analyses to those in the main text, but using the text encoder as probes instead of external linear probes for two datasets: PUG-Animal and ImageNet-AO. We use these datasets for two reasons: (1) their concepts and values map naturally to text prompts, and (2) the datasets were released after the CLIP models and exhibit many unnatural concept combinations unlikely to have appeared in text captions during pre-training, and not present in the visual training data.

- I.4.1. EXPERIMENTS ON PUG-ANIMAL

Setup. Four concepts are exposed: character, background, scale, and texture. For each character we parse the character name into a set of words and use prompts of the form “A picture of a <character>”. For each background, we use prompts of the form “A picture of a <background>” (detailed in Tab. 4).

We map numeric scale values and texture labels to descriptive prompt templates for evaluating the models. Specifically, for scale, we use:

- • 0.7 → “A picture of a small object”
- • 1.0 → “A picture of a medium-sized object”
- • 1.3 → “A picture of a large object”

For textures, we use the following mappings:

- • “Sky” → “A picture of an object in sky texture”
- • “Grass” → “A picture of an object in grass texture”
- • “Asphalt” → “A picture of an object in asphalt texture”

These prompt templates are used to generate the corresponding text embeddings for each concept, matching exactly with the setup of the experiments in the main text.

Table 4. Mapping from class names to clean prompt names for PUG-Animal experiments.

###### Original Name Prompt Name

Desert a desert Tableland a tableland EuropeanStreet a European street OceanFloor the ocean floor Racetrack a racetrack Ruins ancient ruins TrainStation a train station BusStationInterior the interior of a bus station BusStationExterior the exterior of a bus station IndoorStairs indoor stairs Circus a circus BoxingRing a boxing ring Mansion a mansion ShoppingMall a shopping mall ConferenceRoom a conference room VillageOutskirt a village outskirt VillageSquare a village square Courtyard a courtyard Forge a forge Library a library Museum a museum Gallery an art gallery Opera an opera house Restaurant a restaurant RuralAustralia rural Australia AustraliaRoad a road in Australia ShadyRoad a shady road SaltFlats salt flats Castle a castle Temple a temple Snow a snowy landscape Grass a grassy field DryGrass a dry grassland Forest a forest

Concretely, for each concept value j ∈ [n], we pass the prompt template through the text encoder g to obtain a (ℓ2-normalized) probe vector wi,j = g(pi,j) ∈ Z, as detailed in Section 3.4.

0.4

Accuracy

0.3

Random

0.2

0.10

0.33

0.40 0.45 0.50 0.55 Projected R2 (whitened)

- Figure 28. Projected R2 vs accuracy on PUG-Animal across models. Higher projected R2 coincides with higher accuracy on the full dataset. The probes are extracted from the text encoder.

Linearity of factors and generalization. We show the projected R2 and average accuracy on all concept combinations on PUG-Animal across models in Fig. 28 when using the text encoder as probes. Models exhibiting higher linearity of representations generally exhibit higher accuracy on the full dataset. This coincides with the observations in the main text (Section 5.1); random baseline achieves low projected R2 and accuracy.

Orthogonality of the factors. For each of the concepts, we compute the linear factors as detailed in the main text (Section 5.1) with the text encoder as probes. We compute the within- and across-concept orthogonality as detailed in Section I.2 and illustrate the results in Fig. 29 for each of the models.

|[Figure 86]<br><br>0.45 0.48 0.55 0.40<br><br>0.48 0.62 0.63 0.55<br><br>0.55 0.63 0.72 0.49<br><br>0.40 0.55 0.49 0.49|
|---|

characterworldsizetexture

character

world

size

texture

Random

|[Figure 87]<br><br>0.27 0.07 0.25 0.13<br><br>0.07 0.20 0.13 0.09<br><br>0.25 0.13 0.79 0.16<br><br>0.13 0.09 0.16 0.49|
|---|

characterworldsizetexture

CLIP ViT-L/14

|[Figure 88]<br><br>0.30 0.07 0.31 0.13<br><br>0.07 0.22 0.13 0.09<br><br>0.31 0.13 0.85 0.18<br><br><br>0.13 0.09 0.18 0.49|
|---|

characterworldsizetexture

OCLIP ViT-L/14

|[Figure 89]<br><br>0.27 0.07 0.25 0.12<br><br>0.07 0.27 0.13 0.11<br><br>0.25 0.13 0.76 0.18<br><br>0.12 0.11 0.18 0.49|
|---|

characterworldsizetexture

SigLIP-L/16

|[Figure 90]<br><br>0.22 0.06 0.17 0.11<br><br>0.06 0.25 0.09 0.10<br><br>0.17 0.09 0.80 0.16<br><br>0.11 0.10 0.16 0.49|
|---|

characterworldsizetexture

SigLIP2-S/O

- Figure 29. Orthogonality of the factors on PUG-Animal. Heatmaps show pairwise cosine similarity between factors for the four PUGAnimal concepts (character, world, size, texture) across multiple models. The factors are more orthogonal across concepts (off-diagonal) than within concepts (diagonal). The random baseline does not generally show this pattern.

For all evaluated models, we observe the same orthogonality pattern: the factors are more orthogonal across concepts (off-diagonal) than within concepts (diagonal). The average cosine similarity for the random baseline is higher (around 0.5) both within and across concepts.

We also note the qualitative similarity between the factors to the case when probes were trained on 90% of the concept combinations (Fig. 24, second row).

Qualitative examples. We illustrate some of the highest- and lowest-scoring samples in terms of R2 for the SigLIP2 model in Fig. 30. We note that high-scoring samples generally depict clean scenes where the character and its size and texture are easier to discern compared to the lower-scoring samples.

Camel +Snow +0.7 +Grass R2=0.88

WhiteShark +Mansion +0.7 +Grass R2=0.88

Turtle +Opera +1.0 +Grass R2=0.88

Camel +IndoorStairs +0.7 +Asphalt R2=0.87

Camel +Forge +1.3 +Asphalt R2=0.87

Camel +Racetrack +0.7 +Grass R2=0.87

Goat +IndoorStairs +1.0 +Asphalt R2=0.87

Anlylosaurus +Museum +0.7 +Asphalt R2=0.87

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Bear +IndoorStairs +1.0 +Asphalt R2=0.87

PoisonDartFrog +VillageSquare +1.3 +Sky R2=0.87

Goat +IndoorStairs +0.7 +Asphalt R2=0.87

Scorpion +Library +1.0 +Asphalt R2=0.86

Pig +IndoorStairs +1.0 +Asphalt R2=0.86

Spinosaurus +Museum +0.7 +Asphalt R2=0.86

Triceraptos +Museum +0.7 +Grass R2=0.86

Scorpion +Temple +1.0 +Grass R2=0.86

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Scorpion +ConferenceRoom +1.0 +Grass R2=0.86

Beaver +Museum +1.0 +Asphalt R2=0.86

Scorpion +Racetrack +1.0 +Sky R2=0.86

Rhinoceros +IndoorStairs +0.7 +Sky R2=0.86

Pig +ConferenceRoom +1.0 +Asphalt R2=0.86

WhiteShark +Opera +0.7 +Grass R2=0.86

Camel +BusStationExterior +0.7 +Asphalt R2=0.86

Scorpion +Temple +1.3 +Sky R2=0.86

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

(a) Top-scoring samples in terms of R2.

Scorpion +TrainStation +0.7 +Asphalt R2=-0.34

Scorpion +Forge +0.7 +Sky R2=-0.28

Squirrel +BusStationInterior +1.0 +Grass R2=-0.26

Koi +BusStationInterior +0.7 +Grass R2=-0.25

Dolphin +Forge +0.7 +Asphalt R2=-0.23

Anlylosaurus +Courtyard +0.7 +Asphalt R2=-0.22

Triceraptos +BusStationInterior +0.7 +Asphalt R2=-0.22

Dolphin +BusStationInterior +1.0 +Asphalt R2=-0.22

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Anlylosaurus +BusStationInterior +0.7 +Sky R2=-0.22

Parasaurolophus +VillageSquare +0.7 +Asphalt R2=-0.20

Parasaurolophus +Castle +0.7 +Asphalt R2=-0.17

Parasaurolophus +Forge +0.7 +Grass R2=-0.16

Anlylosaurus +TrainStation +0.7 +Asphalt R2=-0.15

Koi +BusStationInterior +0.7 +Asphalt R2=-0.14

Spinosaurus +ShadyRoad +1.0 +Sky R2=-0.14

Anlylosaurus +BusStationInterior +0.7 +Asphalt R2=-0.14

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Parasaurolophus +Forge +0.7 +Asphalt R2=-0.14

Beaver +ShadyRoad +0.7 +Grass R2=-0.14

Turtle +BoxingRing +1.0 +Sky R2=-0.14

Chicken +TrainStation +1.0 +Asphalt R2=-0.13

KomodoDragon +ShoppingMall +0.7 +Asphalt R2=-0.13

Betta +BusStationInterior +1.3 +Asphalt R2=-0.13

Chicken +ShoppingMall +0.7 +Asphalt R2=-0.13

Betta +Forge +0.7 +Asphalt R2=-0.13

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

(b) Bottom-scoring samples in terms of R2.

- Figure 30. Qualitative examples of the top- and lowest-scoring samples in PUG-Animal for the SigLIP2 model. Each sample shows its character name, world name, size value (0.7 corresponds to “small”, 1.0 corresponds to “medium”, 1.3 corresponds to “large”), texture name, and its R2 score.

- I.4.2. EXPERIMENTS ON IMAGENET-AO

We additionally perform experiments on a coarse-captioned dataset ImageNet-AO Abbasi et al. (2024), where each image sample has an associated caption composed of an adjective and a noun.

The experiments here are slightly dissimilar from the main experiments in Section 5.1, for a few reasons: (1) scarcity of per-combination data, (2) inability to train linear probes, (3) noisy/ambiguous data, and (4) coarse categories. Regardless, our framework still applies.

Dataset description. The dataset contains images described by an adjective and a noun. There are around 80 unique adjectives and over 600 unique nouns. To make the analysis balanced, we work with the dataset restricted to the most common 80 nouns and adjectives. Each potential combination of adjective and noun may have between 0 and 6 images. The dataset is thus sparse, and many of the potential combinations are not observed in the dataset. This results in a total of 3243 datapoints. We illustrate the sparsity and the pairs we work with in Fig. 31.

bell pepper

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |[Figure 139]| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

acorn badger

binoculars accordion ballplayer

beaver

bison airship

banana

bagel canoe castle

hermit crab bloodhound grand piano

barn bassoon

warplane

gorilla cellular telephone

burrito pickelhaube

bow tie

gasmask black swan fire engine

geyser electric guitar

grey whale koala

harp cheetah

assault rifle orangutan megalith bathtub cheeseburger

gas pump ambulance

Noun

oxcart cannon

guillotine

collie balloon baboon

coffeepot marimba earthstar

dial telephone great white shark

cinema reflex camera ice bear eel

armadillo horse cart

mantis breakwater axolotl giant panda

pretzel jellyfish

gibbon

chow cash machine pomegranate

cowboy hat goldfish

banjo acoustic guitar

altar cassette player

bobsled

basset hippopotamus

monastery

lion photocopier

oxygen mask

cardboard

burgundy

hardwood

fluffy

furry

turquoise

red

muddy

aqua

pink

porcelain

terracotta

gray

square

archshaped

dilapidated

bronze

green

iron

gravel

cubed

brick

speckled

maroon

bamboo

paper

golden

blue

curly

glass

yellow

foamy

dusty

snowy

diamondshaped

plush

purple

spiral

denim

taupe

fabric

violet

plastic

orange

tiled

teal

wrinkled

amber

marble

mosscovered

wooden

brass

concrete

styrofoam

stainedblueplaid

spherical

redstriped

ceramic

rubber

floral

aluminum

metal

wet

granite

chocolate

dotted

striped

white

steel

transparent

stone

triangular

cobblestone

colorful

straw

cement

leather

checkered

cracked

wicker

Adjective

- 0
- 1
- 2
- 3
- 4
- 5
- 6

###### Figure 31. Adjective-noun count matrix for ImageNet-AO (Abbasi et al., 2024) of the top 80 adjectives and nouns. The adjective-noun pairs are sparse, and many of them are not observed in the dataset.

General setup. Due to limited availability of the data samples, we do not train linear probes. Because each sample is associated with a (noun, adjective) combination, we instead use the probes from the text encoder to assess the performance of the models (as detailed in the main text in Section 3.4). Concretely, we pass captions in the style of “A picture of <noun>” in the case of noun, and “A picture showing <adjective>” in the case of adjective, through the text encoder.

Because of imbalance and sparsity, we cannot rely on averaging to extract the factors as done in Section 5.1. Instead, we follow (Uselis et al., 2025) and solve a linear system of equations to recover the factors. Concretely, we construct a design matrix A ∈ {0, 1}3243×80·2 where each row corresponds to a sample, and each column corresponds to either the presence of a noun (if the column index < 80) or the presence of an adjective (if the column index ≥ 80). The matrix was of full rank 2 · 80 − 1. Then, we solve the linear system

unoun uadj

= X to recover the factors unoun ∈ R80×d and uadj ∈ R80×d, where d is the dimension of the representation space, and

A

X ∈ R3243×d is the centered image embeddings. We show the whitened R2 scores. The remaining procedure in the analysis follows Section 5.1.

Linearity of factors and generalization. We show the projected R2 vs accuracy on ImageNet-AO across models in Fig. 32. As seen in

the main text (Section 5.1), higher projected R2 coincides with higher accuracy on the full dataset. Importantly, the random baseline achieves substantially lower projected R2 (less than 0.1) compared to the other models.

0.76

0.74

0.72

Accuracy

0.70

Random

0.68

0.00

0.66

0.09

0.64

0.36 0.38 0.40 0.42 0.44 0.46 0.48 Projected R2 (whitened)

- Figure 32. Projected R2 vs accuracy on ImageNet-AO across models. Higher projected R2 coincides with higher accuracy on the full dataset. Linear probes were not trained here, and the results are computed using the text encoder.

Orthogonality of the factors. To substantiate the claims of orthogonality of factors across concepts, we extract the factors for all the models as detailed in the setup above. Concretely, for each of the attribute factor ui, i ∈ [80] and noun factor uj, j ∈ [80], within- and across-concept orthogonality as detailed in Section 5.1.

We illustrate the results in Fig. 33. For all of the evaluated models the same pattern of orthogonality is observed: the factors are more orthogonal across concepts than they are within concepts. For example, for the CLIP ViT-L/14 model, the within-concept similarity on average is 0.10 between nouns, and 0.14 between adjectives, while the average cosine similarity across concepts is 0.07. The random baseline on average yields 0.49 cosine similarity both across and within concepts.

Interestingly, all of the non-random models exhibit surprising degree of similarity in terms of the cosine similarities. For example, CLIP ViT-L/14 and OpenCLIP ViT-L/14 on average exhibit almost the same cosine similarity within and across concepts, differing only in the noun-noun cosine similarity (0.10 vs 0.11, respectively). These results support the notions of universality between models as argued by the Platonic Representation Hypothesis (Huh et al., 2024), and empirically observed in Universal Sparse Autoencoders (Thasarathan et al., 2025).

|[Figure 140]<br><br>0.49|0.49|
|---|---|
|0.49|0.49|

Nouns Adjectives

Nouns

Adjectives

Random

|[Figure 141]<br><br>0.10|0.07|
|---|---|
|0.07|0.14|

Nouns Adjectives

CLIP ViT-L/14

|[Figure 142]<br><br>0.11|0.07|
|---|---|
|0.07|0.14|

Nouns Adjectives

OCLIP ViT-L/14

|[Figure 143]<br><br>0.11|0.07|
|---|---|
|0.07|0.14|

Nouns Adjectives

SigLIP-L/16

|[Figure 144]<br><br>0.11|0.07|
|---|---|
|0.07|0.14|

Nouns Adjectives

SigLIP2-S/O

- Figure 33. Orthogonality of the factors on ImageNet-AO. We show the cosine similarity of the factors for the SigLIP2 model on ImageNet-AO; we separate the first concept (nouns) from the second concept (adjectives) and show average similarity across each 2 × 2 block. The factors are more orthogonal across concepts than they are within concepts. The random baseline does not show this pattern.

Qualitative examples. To understand the results deeper, we show the qualitative examples of the top- and lowest-scoring samples in ImageNet-AO for the SigLIP2 model in Fig. 34. The top-scoring samples show high degree of projected R2 scores (generally > 0.75), and correctly depict the adjective and noun of the sample. Even there, however, some samples are incorrectly predicted by the model, suggesting a potential lack of alignment between the image and text encoders5.

The lowest-scoring samples show low degree of projected R2 scores (generally < 0.10), and are often incorrectly predicted by the model. Few of the samples appear to be incorrectly labeled (e.g. first image depicting a orangutan as a gorilla), while some are correctly classified by the model but show a lack of factorization.

5This was less of an issue in the main experiments because the image embeddings were analyzed using linear probes.

(purple + lion) R2=0.81

(pink +

(purple + collie) R2=0.81

(green + collie) R2=0.80

(teal +

(snowy + orangutan) R2=0.80

(green + gorilla) R2=0.79

(snowy + cheetah) R2=0.79

koala) R2=0.81

bison) R2=0.80

Correct

Correct

Correct

Correct

Correct

Correct

Correct

Correct

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

(pink +

(teal + mantis)

(violet + ice bear) R2=0.78

(dusty + armadillo) R2=0.78

(violet + pomegranate) R2=0.78

(moss covered + hippopotamus) R2=0.78

(cardboard + canoe) R2=0.77

(diamond shaped + hippopotamus) R2=0.77

koala) R2=0.79

R2=0.79 Correct

Correct

Incorrect

Incorrect

Incorrect

Correct

Correct

Correct

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

(maroon + lion) R2=0.76

(muddy + black swan) R2=0.76

(chocolate + cannon) R2=0.76

(yellow + banjo) R2=0.76

(checkered + bison) R2=0.75

(turquoise + banjo) R2=0.75

(checkered +

(colorful + airship)

giant panda) R2=0.75

R2=0.75 Correct

Correct

Correct

Correct

Correct

Correct

Correct

Correct

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

(a) Top-scoring samples in terms of R2.

(stone + orangutan) R2=-0.10

(paper +

(moss covered + pomegranate) R2=-0.04

(spherical + bassoon) R2=-0.00

(straw + accordion) R2=0.00

(striped + banjo) R2=0.01

(spherical + cellular telephone) R2=0.03

(bamboo + badger) R2=0.04

cinema) R2=-0.09

Incorrect

Incorrect

Correct

Correct

Incorrect

Incorrect

Incorrect

Correct

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

(stone + chow) R2=0.04

(concrete + bobsled) R2=0.04

(snowy + burrito) R2=0.05

(wicker + pretzel) R2=0.07

(denim + castle) R2=0.09

(glass + castle) R2=0.09

(wicker + monastery) R2=0.10

(marble + bassoon) R2=0.10

Incorrect

Incorrect

Correct

Correct

Incorrect

Incorrect

Incorrect

Correct

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

(moss covered + orangutan) R2=0.10

(granite + acorn) R2=0.11

(styrofoam + monastery) R2=0.11

(metal + cinema) R2=0.11

(denim + oxcart) R2=0.11

(spiral + altar) R2=0.12

(bamboo +

(stainedblue plaid + hermit crab) R2=0.12

accordion) R2=0.12

Correct

Incorrect

Correct

Correct

Incorrect

Correct

Incorrect

Correct

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

(b) Bottom-scoring samples in terms of R2.

- Figure 34. Qualitative examples of the top- and lowest-scoring samples in ImageNet-AO for the SigLIP2 model. Each sample shows its adjective and noun, its R2 score, and whether it was correctly classified by the model. Note that both top- and lowest-scoring samples may be either correctly or incorrectly classified by the model.

###### I.5. Qualitative results

We visualize the recovered factor geometry across three models (DINOv3, OpenCLIP, MetaCLIP-G) and three datasets (dSprites, MPI3D, PUG-Animal). For dSprites (Fig. 36) and MPI3D (Fig. 37), we show both the per-concept factors and the pairwise projections (analogous to Fig. 7c,d in the main text), illustrating low-rank factor structure and near-orthogonality across concepts. For PUG-Animal (Fig. 35), we show only the per-concept factors projected onto their top 3 principal components, since most of its concepts (e.g., character, background) are discrete with many values, resulting in higher-rank factor sets where pairwise projections are less informative.

[Figure 193]

DINO V3

Character vs size Character vs texture World vs texture Size vs texture

OpenCLIP-L-14

Character vs size Character vs texture World vs texture Size vs texture

MetaCLIP2-bigG-14

Character vs size Character vs texture World vs texture Size vs texture

- Figure 35. Factor geometry on PUG-Animal across DINOv3, OpenCLIP, and MetaCLIP2-bigG. Only per-concept factors (projected onto their top 3 PCs) are shown, since most PUG-Animal concepts (character, background) are discrete with many values, yielding higher-rank factor sets where pairwise projections are less informative.

[Figure 194]

###### DINO V3

###### Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models

Color Shape Size Orientation X-position Y-position

Obj. color vs obj. shape Horizontal vs vertical axis Bg. color vs obj. size Bg. color vs obj. shape Bg. color vs camera height

Shape vs Y-position X-position and size X-position and Y-position Color and size

OpenCLIP-L-14

###### OpenCLIP-L-14

Color Shape Size Orientation X-position Y-position

Shape vs Y-position X-position and size X-position and Y-position Color and size

###### MetaCLIP2-bigG-14

Color Shape Size Orientation X-position Y-position

Shape vs Y-position X-position and size X-position and Y-position Color and size

- Figure 36. Factor geometry on dSprites across DINOv3, OpenCLIP, and MetaCLIP2-bigG. For each model: the first row shows the recovered per-concept factors (one 3D plot per concept); the second row shows the pairwise factor geometry for selected concept pairs; and the third row shows all datapoints projected onto the span of those concept pairs.

[Figure 195]

DINO V3

Object color Object shape Object size Camera height Background color Horizontal axis Vertical axis

Obj. color vs obj. shape Horizontal vs vertical axis Bg. color vs obj. size Bg. color vs obj. shape Bg. color vs camera height

OpenCLIP-L-14

Object color Object shape Object size Camera height Background color Horizontal axis Vertical axis

Obj. color vs obj. shape Horizontal vs vertical axis Bg. color vs obj. size Bg. color vs obj. shape Bg. color vs camera height

MetaCLIP2-bigG-14

Object color Object shape Object size Camera height Background color Horizontal axis Vertical axis

Obj. color vs obj. shape Horizontal vs vertical axis Bg. color vs obj. size Bg. color vs obj. shape Bg. color vs camera height

- Figure 37. Factor geometry on MPI3D across DINOv3, OpenCLIP, and MetaCLIP2-bigG. Layout follows Fig. 36: for each model, per-concept factors (first row), pairwise factor geometry (second row), and pairwise datapoint projections (third row).

###### I.6. Empirical evaluation on synthetic data

Results in Section 4 establish the geometry of compositionally generalizing models. A natural question is what geometry emerges in models trained with standard classification losses, without explicit pressure to generalize compositionally. In addition, the core theory is derived for binary concepts, whereas the experiments below also include multi-valued concepts. We study this empirically in this section.

Standard pretraining regimes (e.g., CLIP/SigLIP) do not explicitly optimize for Desideratum 2. This controlled setup therefore lets us isolate which geometric structure emerges by common objectives. The results in this subsection should be read as properties of compositional models trained with standard losses, and not as guarantees of compositional generalization.

We vary two representation geometries and two losses, independently, to cover common training settings and model families, such as CLIP and SigLIP models. We remain architecture-agnostic here, and we optimize embeddings corresponding to each concept combination directly (similarly to (Weller et al., 2025b)).

Because the number of combinations grows as nk, we use at most 100,000 combinations per setting: if nk ≤ 100,000, we use all of the combinations; otherwise, we sample 100,000 combinations. As a result, reported R2 and factor-orthogonality values in those regimes are approximate estimates and should be interpreted with caution. For each setting, we report three metrics, following Section 4: linear-factorization R2, factor orthogonality, and dimension needed to support linearly compositional models (Definition 3).

We use concept spaces Ck,n := C1 ×· · ·×Ck ⊂ [n]k, with embeddings zc ∈ Rd for each c ∈ Ck,n and scores hi,j(z) := τ wi,j⊺ z +bi,j. Each c is initialized randomly: zc ∼ N(0, I). Unless stated otherwise, optimization uses Adam for 50,000 epochs with initial learning rate 0.1 and cosine annealing (Tmax = 50,000, ηmin = 0). We fit the probes as well as the embeddings jointly.

We vary the following factors independently.

- (1) Representation geometry. We consider (1) Euclidean (zc ∈ Rd), where τ is absorbed into probe scale (so τ = 1), and (2) spherical (zc ∈ Sd−1 with ∥zc∥ = ∥wi,j∥ = 1) geometries, where CLIP/SigLIP-style normalization uses explicit temperature τ.
- (2) Loss type. We compare two per-concept losses (one object per scene). First, per-concept softmax cross-entropy (CE):

ℓCE(zc) =

k

i=1

− log

exp hi,ci(zc) n v=1 exp hi,v(zc)

. (99)

Second, per-concept one-vs-rest binary cross-entropy with sigmoid outputs (BCE):

ℓBCE(zc) =

k

i=1

1 n

 − log σ(hi,ci(zc)) −

v̸=ci

log σ(−hi,v(zc))

  . (100)

- (3) Concept space and dimension. For each geometry/loss pair, we retrain with k ∈ [10], n ∈ {2, 6, 12, 24, 48, 96}, and d ∈ {3, . . . , 32}.

- I.6.1. RESULTS: LINEARITY. We illustrate the linearity of the embeddings in the from-scratch setting in Fig. 38. Notably, the majority of the cases exhibit R2 ≥ 0.7.

96

|1.00| | |
|---|---|---|
|1.00| | |
|1.00| | |
|1.00| | |
|1.00| | |
|1.00|0.64| |
|1.00|0.99|0.97|

|0.65| | | | |
|---|---|---|---|---|
|0.73| | | | |
|0.75| | | | |
|0.84| | | | |
|0.83| | | | |
|0.99|0.97| | | |
|0.99|0.99|0.99|0.97|0.96|

|0.77| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
|0.77|0.82| | | | | | | |
|0.82|0.73| | | | | | | |
|0.91|0.73| | | | | | | |
|0.97|0.86|0.76| | | | | | |
|0.98|0.97|0.96|0.98|0.68| | | | |
|0.99|0.99|0.98|0.97|0.94|0.96|0.96|0.96|0.96|

|0.89|0.79|0.67|0.46| | | | | |
|---|---|---|---|---|---|---|---|---|
|0.90|0.83|0.75|0.52| | | | | |
|0.93|0.85|0.79|0.72|0.53| | | | |
|0.95|0.89|0.83|0.79|0.76|0.70| | | |
|0.99|0.96|0.92|0.89|0.84|0.81|0.77|0.76|0.70|
|0.98|0.98|0.97|0.96|0.96|0.96|0.96|0.97|0.97|
|0.99|0.99|0.98|0.98|0.96|0.96|0.96|0.96|0.96|

|0.91|0.86|0.79|0.72|0.62|0.48| | | |
|---|---|---|---|---|---|---|---|---|
|0.92|0.88|0.83|0.76|0.70|0.60|0.47| | |
|0.94|0.89|0.85|0.81|0.76|0.70|0.63|0.50| |
|0.96|0.93|0.88|0.84|0.82|0.78|0.74|0.74|0.68|
|0.99|0.96|0.94|0.93|0.91|0.88|0.87|0.83|0.81|
|0.99|0.98|0.97|0.96|0.96|0.96|0.96|0.96|0.95|
|0.99|0.99|0.98|0.98|0.97|0.96|0.96|0.96|0.96|

BCE,Spherical

Num.values

48

24

12

6

- 2
- 3

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

96

|1.00| | |
|---|---|---|
|1.00| | |
|1.00| | |
|1.00| | |
|1.00| | |
|1.00|0.94|0.62|
|1.00|1.00|1.00|

|0.93| | | | |
|---|---|---|---|---|
|0.93| | | | |
|0.94| | | | |
|0.80| | | | |
|0.88|0.70| | | |
|0.96|0.98|0.76| | |
|0.97|0.97|0.97|0.97|0.99|

|0.98| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
|0.92|0.75| | | | | | | |
|0.89|0.98|0.78| | | | | | |
|0.93|0.80|0.94| | | | | | |
|0.94|0.84|0.64|0.64| | | | | |
|0.92|0.91|0.95|0.99|0.77|0.69| | | |
|0.93|0.89|0.89|0.91|0.89|0.90|0.92|0.94|0.94|

|0.82|0.66|0.55| | | | | | |
|---|---|---|---|---|---|---|---|---|
|0.86|0.72|0.59|0.57| | | | | |
|0.88|0.79|0.67|0.61|0.64| | | | |
|0.88|0.86|0.79|0.71|0.63|0.62|0.71|0.74| |
|0.85|0.88|0.90|0.88|0.84|0.81|0.70|0.62|0.71|
|0.89|0.81|0.80|0.81|0.83|0.85|0.88|0.90|0.92|
|0.91|0.86|0.81|0.77|0.77|0.78|0.77|0.80|0.78|

|0.83|0.78|0.68|0.57|0.51| | | | |
|---|---|---|---|---|---|---|---|---|
|0.84|0.79|0.74|0.64|0.53|0.53| | | |
|0.86|0.82|0.79|0.73|0.65|0.55|0.54| | |
|0.83|0.85|0.83|0.81|0.77|0.72|0.63|0.56|0.58|
|0.80|0.80|0.83|0.86|0.87|0.86|0.86|0.83|0.78|
|0.86|0.78|0.74|0.74|0.76|0.78|0.79|0.80|0.82|
|0.93|0.81|0.78|0.73|0.72|0.71|0.71|0.72|0.73|

BCE,Euclidean

Num.values

48

24

12

6

- 2
- 3

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

96

|1.00| | |
|---|---|---|
|1.00| | |
|1.00|0.69| |
|1.00|0.65| |
|1.00|0.85| |
|1.00|0.84| |
|1.00|0.96|0.95|

|0.97|0.83| | | |
|---|---|---|---|---|
|0.97|0.97| | | |
|0.96|0.97|0.93| | |
|0.82|0.95|0.86|0.93| |
|0.80|0.83|0.85| | |
|0.99|0.96|0.79| | |
|0.99|0.97|0.95|0.97|0.92|

|0.93|0.95|0.94| | | | | | |
|---|---|---|---|---|---|---|---|---|
|0.95|0.93|0.92|0.95|0.88|0.93| | | |
|0.86|0.92|0.90|0.97|0.91|0.90|0.92| | |
|0.82|0.88|0.91|0.87|0.88|0.90|0.93|0.93| |
|0.66|0.78|0.84|0.85|0.83|0.82|0.91|0.88| |
|0.98|0.97|0.93|0.95|0.82|0.82|0.80|0.76| |
|0.97|0.98|0.97|0.96|0.93|0.92|0.93|0.94|0.94|

|0.83|0.85|0.88|0.90|0.92|0.95|0.91|0.94|0.95|
|---|---|---|---|---|---|---|---|---|
|0.82|0.80|0.85|0.87|0.90|0.90|0.90|0.91|0.94|
|0.80|0.77|0.81|0.84|0.87|0.87|0.88|0.90|0.92|
|0.75|0.75|0.76|0.77|0.82|0.83|0.85|0.85|0.84|
|0.97|0.95|0.79|0.73|0.78|0.80|0.81|0.84|0.84|
|0.98|0.97|0.94|0.94|0.94|0.93|0.94|0.94|0.93|
|0.99|0.98|0.97|0.95|0.94|0.94|0.92|0.93|0.92|

|0.76|0.86|0.81|0.84|0.84|0.86|0.89|0.89|0.91|
|---|---|---|---|---|---|---|---|---|
|0.76|0.78|0.81|0.83|0.83|0.83|0.85|0.87|0.88|
|0.78|0.73|0.75|0.77|0.81|0.82|0.83|0.83|0.85|
|0.73|0.66|0.71|0.73|0.74|0.78|0.81|0.82|0.82|
|0.97|0.95|0.95|0.94|0.93|0.75|0.74|0.76|0.77|
|0.99|0.97|0.95|0.94|0.93|0.93|0.93|0.93|0.93|
|1.00|0.99|0.98|0.96|0.96|0.93|0.93|0.92|0.93|

CE,Spherical

Num.values

48

24

12

6

- 2
- 3

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

96

|1.00| | |
|---|---|---|
|1.00|0.92|1.00|
|1.00|0.88|0.99|
|1.00|0.78|1.00|
|1.00|0.52|0.88|
|1.00|0.75|0.45|
|1.00|1.00|1.00|

|1.00|1.00| | | |
|---|---|---|---|---|
|0.58|1.00|0.93| |0.98|
|0.43|0.99|0.90|0.99|0.93|
|0.22|0.57|0.80|0.97|1.00|
|0.33|0.13|0.37|0.66|0.99|
|0.95|0.96|0.07|0.08|0.27|
|0.90|0.92|0.94|0.96|0.96|

|0.70|0.97|0.99|0.99| | | | | |
|---|---|---|---|---|---|---|---|---|
|0.45|0.95|0.99|1.00|0.89|0.95| | | |
|0.25|0.29|0.95|1.00|0.86|0.97|0.99|0.99|0.73|
|0.15|0.22|0.40|0.92|0.84|0.94|0.98|0.99|0.96|
|0.19|0.10|0.09|0.40|0.42|0.72|0.72|0.94|0.98|
|0.89|0.88|0.91|0.93|0.07|0.10|0.31|0.28|0.66|
|0.97|0.89|0.85|0.84|0.86|0.89|0.92|0.91|0.93|

|0.15|0.76|0.77|0.84|0.89|0.93|0.94|0.97|0.98|
|---|---|---|---|---|---|---|---|---|
|0.34|0.77|0.74|0.78|0.82|0.90|0.91|0.95|0.98|
|0.46|0.29|0.75|0.73|0.73|0.80|0.81|0.79|0.85|
|0.39|0.23|0.05|0.69|0.70|0.70|0.70|0.67|0.73|
|0.80|0.78|0.69|0.17|0.44|0.61|0.67|0.63|0.59|
|0.80|0.72|0.73|0.75|0.77|0.79|0.80|0.82|0.84|
|0.86|0.79|0.75|0.71|0.71|0.73|0.75|0.76|0.77|

|0.53|0.79|0.76|0.70|0.70|0.76|0.83|0.88|0.92|
|---|---|---|---|---|---|---|---|---|
|0.45|0.77|0.77|0.73|0.69|0.69|0.73|0.79|0.83|
|0.53|0.68|0.75|0.76|0.73|0.69|0.66|0.66|0.69|
|0.33|0.24|0.37|0.64|0.70|0.68|0.66|0.64|0.59|
|0.73|0.68|0.72|0.17|0.46|0.58|0.57|0.70|0.65|
|0.75|0.69|0.67|0.66|0.68|0.72|0.73|0.74|0.75|
|0.87|0.73|0.69|0.66|0.64|0.65|0.65|0.68|0.70|

CE,Euclidean

Num.values

48

24

12

6

- 2
- 3

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

Num. concepts

Num. concepts

Num. concepts

Num. concepts

Num. concepts

- Figure 38. Linear factorization R2 results in from-scratch setting. We show the linear-factorization R2 when representation space varies and loss type varies. In the majority of the cases, R2 ≥ 0.7.

- I.6.2. RESULTS: ORTHOGONALITY.

The theory in Section 4.1 predicts orthogonality of cross-concept factor differences, while not requiring within-concept directions to be orthogonal. We measure this exactly as in Section I.2: factors are recovered by averaging (centered) embeddings per concept value, then we compute absolute-cosine summaries for within-concept similarity (Orth(i, i)) and cross-concept orthogonality (Orth(i, j), i ̸= j). In the from-scratch experiments, we observe this pattern consistently across CE/BCE losses and Euclidean/spherical geometries: cross-concept cosine similarity is lower than within-concept cosine similarity, and it decreases as k grows. This behavior is therefore aligned with the geometric story in the main theory, though still approximate.

0.58 0.00

0.49 0.24

0.37 0.14

0.19 0.16

0.20 0.16

0.23 0.15

0.18 0.17

0.15 0.14

0.15 0.14

0.15 0.14

0.15 0.14

0.16 0.14

0.14 0.14

96

0.58 0.00

0.51 0.19

0.34 0.18

0.50 0.06

0.18 0.16

0.20 0.15

0.23 0.15

0.22 0.15

0.14 0.12

0.15 0.13

0.15 0.13

0.15 0.13

0.16 0.13

0.17 0.13

0.15 0.13

BCE,Spherical

48

Num.values

0.60 0.00

0.51 0.18

0.30 0.20

0.47 0.12

0.15 0.13

0.18 0.14

0.20 0.14

0.24 0.14

0.26 0.14

0.12 0.10

0.13 0.12

0.14 0.12

0.15 0.12

0.16 0.12

0.16 0.13

0.19 0.12

0.19 0.12

24

0.58 0.00

0.50 0.13

0.28 0.14

0.34 0.18

0.11 0.08

0.15 0.11

0.18 0.12

0.18 0.13

0.23 0.13

0.34 0.11

0.10 0.05

0.11 0.08

0.14 0.09

0.14 0.10

0.15 0.10

0.15 0.11

0.16 0.11

0.19 0.12

0.25 0.11

12

0.56 0.00

0.38 0.27

0.21 0.08

0.27 0.15

0.41 0.15

0.20 0.02

0.20 0.03

0.20 0.06

0.20 0.08

0.22 0.10

0.21 0.11

0.23 0.12

0.36 0.10

0.48 0.08

0.20 0.02

0.20 0.02

0.20 0.03

0.20 0.04

0.20 0.06

0.20 0.07

0.20 0.07

0.21 0.09

0.21 0.09

6

0.49 0.00

0.47 0.47

0.50 0.07

0.50 0.04

0.50 0.09

0.50 0.05

0.50 0.03

0.50 0.02

0.49 0.22

0.50 0.07

0.50 0.03

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.03

0.50 0.03

0.50 0.03

0.50 0.05

0.50 0.03

0.50 0.02

0.50 0.02

0.50 0.01

0.50 0.02

0.50 0.02

0.50 0.03

0.50 0.03

- 2
- 3

1.00 0.00

1.00 0.08

1.00 0.25

1.00 0.07

1.00 0.07

1.00 0.05

1.00 0.05

1.00 0.04

1.00 0.01

1.00 0.07

1.00 0.05

1.00 0.03

1.00 0.02

1.00 0.03

1.00 0.03

1.00 0.04

1.00 0.03

1.00 0.13

1.00 0.07

1.00 0.05

1.00 0.03

1.00 0.03

1.00 0.01

1.00 0.02

1.00 0.01

1.00 0.02

1.00 0.11

1.00 0.05

1.00 0.03

1.00 0.02

1.00 0.02

1.00 0.01

1.00 0.01

1.00 0.01

1.00 0.01

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

0.50 0.00

0.50 0.03

0.40 0.02

0.19 0.16

0.22 0.15

0.26 0.14

0.16 0.14

0.16 0.14

0.16 0.14

0.16 0.14

0.16 0.14

96

BCE,Euclidean

0.49 0.00

0.50 0.04

0.39 0.05

0.46 0.07

0.18 0.14

0.21 0.15

0.26 0.13

0.30 0.12

0.15 0.13

0.16 0.13

0.16 0.13

0.17 0.13

0.18 0.13

0.20 0.12

48

Num.values

0.48 0.00

0.50 0.04

0.37 0.08

0.49 0.02

0.58 0.06

0.17 0.12

0.20 0.14

0.22 0.13

0.31 0.11

0.37 0.09

0.11 0.10

0.15 0.11

0.15 0.12

0.17 0.12

0.18 0.12

0.21 0.12

0.25 0.11

24

0.46 0.00

0.53 0.10

0.33 0.09

0.46 0.08

0.56 0.02

0.13 0.08

0.14 0.11

0.19 0.11

0.24 0.12

0.31 0.11

0.39 0.09

0.46 0.07

0.51 0.05

0.09 0.07

0.12 0.07

0.14 0.09

0.15 0.10

0.16 0.10

0.18 0.10

0.19 0.11

0.24 0.11

0.32 0.09

12

0.59 0.00

0.45 0.10

0.54 0.17

0.20 0.09

0.30 0.11

0.44 0.10

0.48 0.11

0.20 0.04

0.20 0.03

0.20 0.05

0.20 0.07

0.22 0.08

0.26 0.08

0.32 0.09

0.41 0.08

0.46 0.07

0.20 0.03

0.20 0.03

0.20 0.03

0.20 0.04

0.20 0.05

0.20 0.07

0.20 0.07

0.21 0.07

0.22 0.08

6

0.49 0.00

0.49 0.51

0.51 0.38

0.50 0.09

0.50 0.06

0.51 0.20

0.50 0.02

0.50 0.06

0.50 0.04

0.50 0.03

0.49 0.11

0.50 0.16

0.50 0.06

0.50 0.03

0.50 0.04

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.03

0.50 0.02

0.50 0.02

0.50 0.03

0.50 0.02

0.50 0.02

0.50 0.01

0.50 0.01

0.50 0.01

0.50 0.01

- 2
- 3

1.00 0.00

1.00 0.14

1.00 0.08

1.00 0.02

1.00 0.09

1.00 0.03

1.00 0.05

1.00 0.06

1.00 0.00

1.00 0.11

1.00 0.03

1.00 0.05

1.00 0.04

1.00 0.03

1.00 0.04

1.00 0.03

1.00 0.06

1.00 0.21

1.00 0.05

1.00 0.04

1.00 0.05

1.00 0.02

1.00 0.02

1.00 0.03

1.00 0.02

1.00 0.02

1.00 0.19

1.00 0.05

1.00 0.05

1.00 0.04

1.00 0.02

1.00 0.03

1.00 0.02

1.00 0.02

1.00 0.01

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

0.50 0.00

0.49 0.03

0.59 0.13

0.36 0.05

0.46 0.05

0.57 0.06

0.22 0.10

0.26 0.10

0.32 0.08

0.38 0.07

0.43 0.05

0.47 0.04

0.51 0.05

0.55 0.05

0.59 0.05

0.16 0.11

0.16 0.12

0.19 0.12

0.22 0.11

0.28 0.09

0.35 0.07

0.39 0.06

0.43 0.05

0.46 0.05

96

0.49 0.00

0.49 0.02

0.63 0.03

0.36 0.05

0.45 0.06

0.54 0.07

0.61 0.05

0.69 0.10

0.77 0.09

0.21 0.10

0.25 0.10

0.30 0.09

0.36 0.08

0.42 0.06

0.46 0.06

0.50 0.05

0.54 0.05

0.58 0.04

0.15 0.10

0.17 0.11

0.18 0.11

0.21 0.11

0.25 0.10

0.30 0.09

0.36 0.08

0.41 0.06

0.44 0.06

48

CE,Spherical

Num.values

0.47 0.00

0.87 0.27

0.47 0.05

0.62 0.03

0.77 0.09

0.33 0.11

0.44 0.06

0.52 0.07

0.62 0.03

0.70 0.07

0.76 0.08

0.83 0.08

0.18 0.11

0.23 0.09

0.25 0.11

0.33 0.09

0.38 0.08

0.42 0.07

0.48 0.06

0.52 0.05

0.55 0.05

0.12 0.09

0.15 0.09

0.17 0.10

0.19 0.10

0.23 0.10

0.25 0.10

0.29 0.09

0.35 0.09

0.40 0.07

24

0.50 0.00

0.75 0.30

0.44 0.15

0.59 0.04

0.74 0.12

0.88 0.07

0.28 0.14

0.41 0.09

0.50 0.07

0.55 0.10

0.66 0.09

0.77 0.08

0.87 0.08

0.91 0.07

0.15 0.10

0.17 0.09

0.22 0.08

0.25 0.10

0.30 0.10

0.35 0.09

0.41 0.08

0.46 0.08

0.50 0.07

0.15 0.08

0.13 0.08

0.14 0.08

0.16 0.08

0.17 0.09

0.20 0.09

0.23 0.09

0.27 0.09

0.29 0.10

12

0.34 0.00

0.59 0.30

0.33 0.16

0.56 0.13

0.68 0.15

0.26 0.18

0.31 0.12

0.44 0.10

0.49 0.11

0.57 0.13

0.68 0.11

0.84 0.08

0.90 0.07

0.20 0.04

0.20 0.02

0.20 0.04

0.21 0.09

0.21 0.08

0.24 0.09

0.30 0.10

0.36 0.09

0.41 0.10

0.20 0.02

0.20 0.01

0.20 0.01

0.20 0.02

0.20 0.04

0.20 0.06

0.21 0.07

0.21 0.08

0.21 0.08

6

0.49 0.00

0.58 0.39

0.50 0.03

0.50 0.06

0.49 0.19

0.50 0.09

0.50 0.03

0.50 0.04

0.50 0.03

0.50 0.13

0.50 0.16

0.48 0.17

0.62 0.14

0.50 0.07

0.50 0.04

0.50 0.02

0.50 0.02

0.50 0.01

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.02

0.50 0.01

0.50 0.01

0.50 0.01

0.50 0.01

0.50 0.02

0.50 0.02

0.50 0.02

- 2
- 3

1.00 0.00

1.00 0.39

1.00 0.09

1.00 0.09

1.00 0.13

1.00 0.08

1.00 0.04

1.00 0.08

1.00 0.15

1.00 0.10

1.00 0.09

1.00 0.04

1.00 0.04

1.00 0.04

1.00 0.03

1.00 0.03

1.00 0.05

1.00 0.01

1.00 0.04

1.00 0.02

1.00 0.04

1.00 0.01

1.00 0.01

1.00 0.02

1.00 0.02

1.00 0.02

1.00 0.06

1.00 0.03

1.00 0.03

1.00 0.02

1.00 0.02

1.00 0.02

1.00 0.01

1.00 0.02

1.00 0.01

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

0.52 0.00

0.50 0.01

0.62 0.07

0.36 0.15

0.46 0.05

0.55 0.06

0.61 0.06

0.43 0.43

0.21 0.14

0.30 0.10

0.38 0.07

0.43 0.05

0.48 0.04

0.53 0.04

0.58 0.04

0.61 0.04

0.16 0.13

0.15 0.13

0.17 0.13

0.19 0.12

0.26 0.10

0.34 0.08

0.39 0.06

0.44 0.05

0.47 0.04

96

0.52 0.00

0.82 0.18

0.94 0.14

0.72 0.71

0.63 0.03

0.76 0.15

0.88 0.13

0.35 0.15

0.45 0.05

0.55 0.04

0.62 0.04

0.69 0.11

0.79 0.09

0.21 0.15

0.20 0.14

0.26 0.12

0.34 0.09

0.41 0.07

0.46 0.05

0.50 0.04

0.56 0.04

0.60 0.03

0.15 0.12

0.15 0.12

0.16 0.12

0.18 0.12

0.21 0.12

0.28 0.10

0.34 0.08

0.40 0.07

0.44 0.05

48

CE,Euclidean

Num.values

0.68 0.00

0.83 0.17

1.00 0.06

0.43 0.23

0.62 0.04

0.79 0.11

0.93 0.09

0.93 0.09

0.34 0.23

0.43 0.21

0.52 0.06

0.62 0.03

0.70 0.08

0.83 0.07

0.90 0.07

0.95 0.06

0.95 0.07

0.19 0.17

0.21 0.15

0.22 0.13

0.28 0.11

0.34 0.10

0.42 0.07

0.46 0.07

0.49 0.07

0.54 0.07

0.17 0.15

0.15 0.09

0.15 0.11

0.17 0.11

0.18 0.11

0.21 0.11

0.25 0.11

0.32 0.09

0.37 0.09

24

0.53 0.00

0.65 0.34

1.00 0.05

0.60 0.59

0.57 0.23

0.67 0.17

0.95 0.05

0.92 0.10

0.39 0.38

0.39 0.15

0.46 0.20

0.56 0.08

0.64 0.10

0.76 0.07

0.91 0.04

0.96 0.04

0.96 0.05

0.23 0.22

0.19 0.13

0.28 0.23

0.23 0.11

0.28 0.11

0.34 0.10

0.40 0.09

0.43 0.09

0.49 0.08

0.16 0.15

0.17 0.10

0.15 0.08

0.15 0.09

0.16 0.09

0.19 0.10

0.20 0.10

0.21 0.10

0.26 0.10

12

0.51 0.00

0.71 0.51

0.92 0.32

0.63 0.67

0.50 0.43

0.63 0.19

0.78 0.20

0.99 0.06

0.81 0.80

0.37 0.37

0.40 0.21

0.50 0.15

0.54 0.16

0.67 0.11

0.72 0.11

0.82 0.11

0.92 0.09

0.20 0.07

0.20 0.06

0.20 0.07

0.29 0.20

0.21 0.10

0.23 0.09

0.27 0.10

0.31 0.10

0.35 0.10

0.20 0.05

0.20 0.05

0.20 0.05

0.26 0.21

0.20 0.06

0.21 0.06

0.21 0.07

0.22 0.08

0.22 0.08

6

0.50 0.00

0.49 0.51

0.70 0.47

0.50 0.09

0.50 0.08

0.69 0.55

0.52 0.36

0.80 0.25

0.50 0.06

0.50 0.06

0.50 0.04

0.50 0.04

0.47 0.24

0.47 0.20

0.51 0.17

0.55 0.44

0.55 0.17

0.50 0.07

0.50 0.04

0.50 0.03

0.50 0.03

0.50 0.03

0.50 0.03

0.50 0.04

0.50 0.04

0.50 0.04

0.50 0.10

0.50 0.04

0.50 0.04

0.50 0.02

0.50 0.02

0.50 0.03

0.50 0.03

0.50 0.03

0.50 0.03

- 2
- 3

1.00 0.00

1.00 0.11

1.00 0.07

1.00 0.06

1.00 0.08

1.00 0.05

1.00 0.06

1.00 0.08

1.00 0.15

1.00 0.08

1.00 0.04

1.00 0.05

1.00 0.04

1.00 0.04

1.00 0.04

1.00 0.06

1.00 0.06

1.00 0.13

1.00 0.06

1.00 0.05

1.00 0.03

1.00 0.03

1.00 0.02

1.00 0.03

1.00 0.02

1.00 0.02

1.00 0.02

1.00 0.05

1.00 0.06

1.00 0.03

1.00 0.04

1.00 0.03

1.00 0.02

1.00 0.02

1.00 0.02

1 2 3

2 3 4 5 6

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

2 3 4 5 6 7 8 9 10

Num. concepts

Num. concepts

Num. concepts

Num. concepts

Num. concepts

- Figure 39. Within-concept and cross-concept cosine similarity of factors in the from-scratch setting. Metrics are computed as in Section I.2. For each (k, n) setting, each cell reports two values: within-concept cosine similarity (top) and cross-concept cosine similarity (bottom). As the number of concepts increases, cross-concept cosine similarity decreases, indicating stronger cross-concept orthogonality.

I.6.3. RESULTS: DIMENSIONALITY.

In Section 4.3, we proved the lower bound d ≥ k for linear readout, independent of the number of values n. That result is a capacity bound: it states what is necessary in principle, but does not guarantee that a particular training objective reaches the bound.

Here, we estimate the minimum dimension needed to reach ≥ 0.99 per-concept accuracy across concept spaces varying in k and n, under CE (CLIP-like training) and BCE (SigLIP-like) losses, and under Euclidean (dot-product) and spherical (cosine-normalized) geometries.

The trends in Fig. 40 are consistent with the theory. CE (CLIP-like) in Euclidean space is typically close to the theoretical minimum (d ≈ k), whereas BCE (SigLIP-like) generally requires larger dimension (often around 2k), perhaps intuitively: in BCE each class is trained as an independent one-vs-rest binary problem, so each positive set must be separated from all negatives by its own hyperplane; CE instead uses relative margins across classes and can thus separate more flexibly (Crammer & Singer, 2002; Bangachev et al., 2025), matching the inference rule with the optimization objective. Spherical geometry shifts required dimensionality upward by roughly one dimension relative to Euclidean settings. Overall, this supports the theoretical claim that concept count is the primary dimensional driver, while showing that objective affect the attainability of the bound.

###### (a) (b)

BCE,Spherical

BCE,Euclidean

[Figure 196]

[Figure 197]

- 3 5 10 15 19 22 26 32 48 48
- 3 6 12 17 21 25 32 32 48 48

3 6 10 18 23 27 32 48 48 48

96

96

30

- 2 4 6 8 10 12 14 15 17 18

- 2 4 8 8 11 13 14 18 22 25
- 2 5 7 10 14 18 20 27 32 48

- 2 5 10 16 19 24 28 48 48 48

48

48

Num.values

3 6 9 12 16 19 23 26 32 32

24

24

- 2 3 3 4 5 6 6 7 7 8

- 2 3 5 7 8 9 9 11 11 14
- 3 4 6 9 11 13 16 17 19 21

- 3 5 7 11 14 17 20 22 25 27

25

12

12

6

6

- 2 3 3 4 5 5 6 6 7 8
- 2 3 3 5 6 8 9 7 11 11

- 2

- 3

- 2

- 3

20

Mediandim

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

15

CE,Spherical

CE,Euclidean

[Figure 198]

[Figure 199]

- 2 3 4 5 7 7 8 9 10 11

- 2 3 4 5 6 8 9 10 10 11
- 2 3 4 6 7 8 8 10 11 12

- 2 4 5 6 7 8 9 11 11 12
- 2 4 5 5 8 8 10 10 11 12

96

96

10

48

48

Num.values

24

24

12

12

6

6

- 2 3 3 4 5 6 6 6 7 8
- 2 3 4 5 6 7 8 8 9 10

- 2 3 3 4 5 5 6 6 7 8
- 2 3 3 4 5 6 7 7 8 9

2 3 3 4 5 6 7 8 9 10

2 3 3 4 5 6 7 8 9 10

- 2 3 3 4 5 6 7 8 9 10

2 3 3 4 7 6 9 10 11 12

- 2 3 4 4 5 9 10 14 12 18

- 2

- 3

- 2

- 3

1 2 3 4 5 6 7 8 9 10 Num. concepts

1 2 3 4 5 6 7 8 9 10 Num. concepts

BCE - Spherical

BCE - Euclidean

CE - Spherical

CE - Euclidean

5

0

1 2 3 4 5 6 7 8 9 10 Num. concepts

- Figure 40. Minimum embedding dimensionality across concept spaces, losses, and representation geometries. We report the smallest dimension d needed to reach ≥ 0.99 per-concept classification accuracy for concept spaces with k concepts and n values per concept. (a) Required d as both k and n vary. (b) Median required d versus k: CE is typically near d≈k, BCE needs roughly 2k, and spherical variants require about one additional dimension compared to Euclidean.

