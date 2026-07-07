## NearID: Identity Representation Learning via Near-identity Distractors

# arXiv:2604.01973v1[cs.CV]2Apr2026

Aleksandar Cvejic1 , Rameen Abdal2⋆ , Abdelrahman Eldesokey1 , Bernard Ghanem1 , and Peter Wonka1

1 King Abdullah University of Science and Technology (KAUST), Saudi Arabia 2 Snap Research, Palo Alto, CA, USA

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Cute dragon…

Generative Models T2I

Generative Models

Cute

alien…

Inpainted Distractor

Real Data

Synthetic data

Real Data Embedding Synthetic Data Encoder Embedding Encoder

Encoder

+

Embedding Distractors

Traditional Representation Learning Representation Learning with Synthetic Data Representation Learning via NearID distractors

Fig. 1: NearID: From Context to Identity. (Left) Traditional representations entangle object identity with background context. (Middle) Synthetic Data lacks explicit control over visually similar distractors. (Right) NearID introduces matched-context distractors to remove contextual shortcuts and isolate intrinsic identity signals.

##### Project page: https://gorluxor.github.io/NearID/

Abstract. When evaluating identity-focused tasks such as personalized generation and image editing, existing vision encoders entangle object identity with background context, leading to unreliable representations and metrics. We introduce the first principled framework to address this vulnerability using Near-identity (NearID) distractors, where semantically similar but distinct instances are placed on the exact same background as a reference image, eliminating contextual shortcuts and isolating identity as the sole discriminative signal. Based on this principle, we present the NearID dataset (19K identities, 316K matched-context distractors) together with a strict margin-based evaluation protocol. Under this setting, pre-trained encoders perform poorly, achieving Sample Success Rates (SSR), a strict margin-based identity discrimination metric, as low as 30.7% and often ranking distractors above true cross-view matches. We address this by learning identity-aware representations on a frozen backbone using a two-tier contrastive objective enforcing the hierarchy: same identity > NearID distractor > random negative. This improves SSR to 99.2%, enhances part-level discrimination by 28.0%, and yields stronger alignment with human judgments on DreamBench++, a human-aligned benchmark for personalization.

Keywords: Representation Learning · Identity Representation ⋆ Served in an advisory role.

### 1 Introduction

How reliably can a vision encoder tell whether two images depict the same object? When the backgrounds differ, modern foundation models, including CLIP [49], DINOv2 [43], SigLIP2 [64], and even large vision-language models such as Qwen3-VL [3], often succeed. Yet these same models are systematically fooled by a simple adversarial test: replace the object with a NearID (different but visually similar) instance while keeping the background identical. Under this condition, the shared context dominates the embedding, causing the impostor to score higher than the true identity viewed against a different backdrop. This failure is not merely academic; it directly undermines the CLIP-I and DINO scores that the majority of personalization and editing literature relies on for automated identity preservation evaluation [21,46,51].

We provide the first principled framework to address this vulnerability through

Near-identity (NearID) (See Figure 1): semantically similar but distinct instances inpainted into the exact same background as the reference, removing all contextual shortcuts and isolating identity as the sole discriminative signal. Under the resulting NearID evaluation protocol, the frozen SigLIP2 backbone achieves only 30.74% Sample Success Rate (SSR), threshold-based similarity based on Nearidentity distractors and positives defined in Sec. 3.4, and on part-level manipulations from the Mind the Glitch (MTG) dataset [13], every standard encoder (CLIP, DINOv2, SigLIP2) scores exactly 0.0% SSR.

To close this gap we introduce NearID, comprising three tightly coupled contributions: (i) a large-scale dataset of 19k object identities with multi-view positives and over 316k NearID distractors synthesized from four generative models, providing the training signal and evaluation benchmark; (ii) a two-tier contrastive objective that enforces an explicit similarity hierarchy (same identity > NearID distractor > random batch negative), teaching the model what to be invariant to and what to discriminate; and (iii) a lightweight training recipe that keeps the foundation encoder frozen and learns only a Multi-head Attention Pooling (MAP) [24,79] projection head (∼3.6% of total parameters), preserving the general-purpose priors while reshaping the similarity geometry for identity.

Empirically, NearID resolves the identified failure at the object level on NearID, where SSR rises from 30.74% to 99.17%, and at the part level on Mindthe-Glitch (MTG), where SSR improves from 0.0% to 35.0% [13], a setting in which even the dedicated Visual Semantic Matching (VSM) metric reaches only 7.0% under the same training data and pairwise evaluation protocol [13]. Beyond SSR, we demonstrate substantially improved alignment with benchmark oracles and human judgments despite training exclusively on near-identity distractors: on MTG, Pearson correlation to the metric oracle increases from 0.180 to 0.465 under the full evaluation with view and background changes, and from 0.366 to 0.486 under the paired background context protocol (Sec. 3.4); on DreamBench++ [46], our identity-specific tuning improves overall alignment with human concept-preservation judgments even though our training data excludes style prompts and live subjects, and it generalizes to animals and humans with correlation gains of 0.105 and 0.065, respectively. These results indicate that the

|[Figure 12]|
|---|

###### Positives misaligned Frozen

|[Figure 13]| | |
|---|---|---|
| |𝒅𝟑| |

Gallery

[Figure 14]

𝒅𝒊 < 𝒅𝒋 Negatives are closer

|[Figure 15]|
|---|
|[Figure 16]|
|[Figure 17]|
|[Figure 18]|

Trained

|[Figure 19]|
|---|

[Figure 20]

Input image

Ours

SigLIP2

|[Figure 21]|
|---|

|𝒅𝟒|
|---|

|𝒅𝟏|
|---|

Image

Image

Encoder

Encoder

|[Figure 22]|
|---|

|𝒅𝟏|
|---|

MAP

MAP

|[Figure 23]| |[Figure 24]|
|---|---|---|
| |𝒅𝟑| |
| | | |

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|𝒅𝟐| |
|---|---|
| |𝒅𝟒|
| | |

Image

Manifold

Similarity Scores with Gallery

Similarity Scores with Gallery

|𝑑𝑖 - 𝑐𝑜𝑠𝑖𝑛𝑒 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒|
|---|

- Fig. 2: NearID overview. Left: In the pretrained image-embedding manifold (e.g., SigLIP2 [64]), identity-consistent positives can be misaligned: edited or re-rendered views of the same instance may lie farther from the anchor than visually confusable negatives (NearID distractors illustrated by di < dj), which degrades retrieval and scoring reliability. Right: We keep the SigLIP2 image encoder frozen and train only a lightweight MAP [24,79] projection head to reshape the similarity geometry for the target task. Compared to the frozen baseline, the trained head increases similarity for true positives (green) while pushing NearID distractors and unrelated gallery items lower (red), improving the desired ordering of similarity scores with respect to a gallery.

learned representations capture genuine identity cues rather than overfitting to synthetic artifacts.

In summary, our contributions are:

- 1. We formalize Near-identity distractors: confounders that share the exact background as the reference while changing the identity signal. We introduce NearID-bench, a matched-context evaluation protocol (SSR and PA) that quantifies identity context entanglement in modern vision encoders, and we release a large-scale NearID dataset (19k identities, 316k+ distractors, 4 generative models) to support research in this setting.
- 2. Building on the standard MAP projection head used with SigLIP2, we train a lightweight identity adapter on a frozen SigLIP2 backbone using a two-tier contrastive objective that enforces a strict similarity ordering: same identity > NearID distractor > random batch negative.
- 3. We show that this training yields embeddings that substantially improve matched-context NearID distractor rejection (NearID SSR and PA), increase sensitivity to localized identity edits in part-level evaluation (MTG), and increase metric-to-human agreement on DreamBench++, indicating that reducing identity context coupling leads to more reliable personalization evaluation.

- Table 1: Near-identity discrimination and alignment. We evaluate matchedcontext Near-identity discrimination at the object level (NearID) and part level (MTG [13]), and measure correlation with MTG oracle scores and human conceptpreservation judgments on DreamBench++ (DB++) [46], verifying alignment with both oracle-defined edit severity and human perception. Top: Existing embedding and VLM methods under the Near-identity protocol. Bottom: Training in our setting, showing improvements on NearID and gains on MTG and DB++. MTG reports metric-to-oracle alignment (M–O/M–Opair); DB++ reports metric-to-human alignment (M–H). Deltas are relative to the frozen SigLIP2 baseline. * denotes a method trained directly on MTG. SSR and PA are averaged across seven inpainting settings (three excluded from training; see Sec. 3.4).

NearID MTG DB++ SSR ↑ PA ↑ M–O ↑ M–Opair ↑ SSR ↑ PA ↑ M–H ↑

Scoring Model

Qwen3VL30B [3] 49.73 69.20 0.219 0.329 17.0 26.0 – CLIP [49] 10.31 20.92 0.239 0.484 0.0 0.0 0.493 DINOv2 [43] 20.43 34.55 0.324 0.519 0.0 0.0 0.492 VSM* [13] 32.13 46.70 0.394 0.445 7.0 24.5 0.190

SigLIP2 [64] 30.74 48.81 0.180 0.366 0.0 0.0 0.516 NearID (Ours)

###### 99.17

###### 99.71

###### 0.465

###### 35.0

###### 46.5

###### 0.545

0.486

(+68.43)

(+50.90)

(+0.285)

(+0.120)

(+35.0)

(+46.5)

(+0.029)

### 2 Related Work

Visual foundation models and identity representation: Representation learning has advanced rapidly through self-supervised [6, 8, 14, 18, 20], multimodal [2,25,37,49,76], and generative pretraining [50,51,66,67], driving progress across discriminative [31,38] and generative tasks. Modern vision-language models (VLMs) [1, 3, 10, 36, 39, 68] are now widely adopted as general-purpose encoders [54,64,71,80]. Contrastive language-image models [49,64] excel at broad category-level alignment, and the self-supervised DINO family [6,43] learns powerful dense features and patch-level structural representations. However, because these models are trained for broad semantic grouping, they often entangle instance identity with context [55], yielding high similarity for images with similar backgrounds despite different instances. Recent bias analyses confirm that VLM behavior can be strongly influenced by background patterns, and that removing the background can substantially reduce such biases [65]. Specialized identity encoders [12,30,40,44,53] achieve strong identity separation but remain confined to narrow domains such as human faces. Region-focused approaches like AlphaCLIP [59] augment CLIP with an auxiliary alpha channel to extract features from user-specified regions (e.g., masks or bounding boxes) while retaining contextual awareness, but still lack explicit identity-level supervision. In contrast, our work seeks to extract open-domain, instance-level identity representations. Rather than risking catastrophic forgetting by fine-tuning a foundation model, we leverage the robust priors of a frozen backbone and utilize a lightweight Multihead Attention Pooling (MAP) head [24,79], which we train to explicitly attend to identity-salient features while suppressing contextual background cues.

Deep metric learning and hierarchical contrastive objectives: Standard deep metric learning relies on pulling positive pairs together while pushing negatives apart. Traditional formulations utilize batch-based contrastive losses [8,41] or supervised extensions that group multiple positives SupContrast [28]. While effective for instance discrimination, these binary objectives treat all negatives equally, forcing the model to push semantically related intra-class (NearID) samples and completely unrelated random samples to the same near-zero similarity. To inject structure into the negative space, prior works have explored triplet mining [53], multi-negative objectives [56], and dynamic pair weighting [58,69]. Hierarchical metric learning further attempts to impose discrete geometric boundaries using ranked lists or tree-based proxies [70, 74]. However, these methods typically rely on online algorithmic selection of informative negatives, which is unstable during training, or on predefined categorical taxonomies. Our approach differs fundamentally by utilizing explicitly curated near-identity distractors generated via controlled inpainting. We enforce a strict three-tier structural hierarchy (positives > Near-identity negatives > batch negatives) through our dual-tier objective, providing stable, constant gradient pressure without the instability of algorithmic mining. Continuous calibration and learning-to-rank: Beyond strict categorical ranking, identity preservation often exhibits varying degrees of severity, requiring continuous calibration of similarity scores. Soft contrastive learning and contrastive regression frameworks [27, 62, 78] replace binary targets with continuous variables, adjusting the contrastive push based on label distance. Similarly, Learning-to-Rank (LTR) objectives optimize for monotonic sorting across retrieved items rather than absolute distances [4,5,73]. While posthoc calibration strategies like temperature scaling or isotonic regression [19,77] can adjust score distributions after training, they cannot fix underlying manifold collisions. In contrast to methods requiring explicit continuous regression targets, our approach achieves data-driven structural calibration. By subjecting a diverse curriculum of part-level edits (via the MTG dataset [13]) to our ranking regularizer (Lrank), the embedding magnitudes naturally correlate with physical edit severity. This smoothly interpolates between identical instances and unrelated batch negatives without relying on manual numeric margin during optimization.

Evaluation of personalized image generation: The rapid advancement of subject-driven generation and image editing [9, 17, 48, 51, 67, 75] has exposed a critical vulnerability in current evaluation protocols. The vast majority of personalization literature relies on out-of-the-box CLIP [49] cosine similarity (CLIP-I) or DINO-scores to quantify identity preservation [21]. However, as our experiments demonstrate, these generic semantic metrics are easily fooled by near-identity confounders, leading to inflated scores when a model generates a different instance on the correct background. Recent work by Kilrain et al. [29] further shows that pairwise CLIP/DINO similarities can remain near-perfect when layout and background match while identity-defining details drift, and proposes a gallery-based retrieval protocol to expose this failure mode. In personalization pipelines, this identity and context coupling manifests in both learning and inference: optimizing textual identifiers with standard diffusion losses can

Input SigLIP2 Ours

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

DINOv2 SigLIP2

NearID

VSM

NearID object-level ✗ ✗ ✗ ✓ NearID part-level ✗ ✗ ✓ ✓ Mask-free ✓ ✓ ✗ ✓ Human alignment ✓ ✓ ✗ ✓ Oracle alignment ✗ ✗ ✓ ✓

- Fig. 3: Left: Attention map comparison vs baseline on positives and negatives. Right: Summary of properties evaluated in this paper. NearID improves matched-context Near-ID rejection on NearID-bench and transfers to part-level identity evaluation on MTG, while remaining mask-free at inference.

entangle foreground and background information [7], and diffusion-transformer architectures with global attention can conflate semantic identity with spatial layout [23]. Current evaluation protocols using VLMs as judges further compound these issues, since they provide the full image as input rather than isolating the subject [32,82].

Recent efforts to bridge this gap have introduced perceptual metrics calibrated to human judgments, ranging from low-level distortion penalties [81] to mid-level holistic similarity metrics like DreamSim [16]. While DreamSim successfully aligns with human preferences regarding layout, pose, and overall composition, it evaluates holistic scene similarity rather than isolating strict object identity. Consequently, it remains susceptible to background conflation when a structurally similar distractor shares the reference context. Furthermore, recent analysis by PercepAlign [60] demonstrates that fully fine-tuning vision encoders to align with generic human perceptual judgments actively degrades their underlying high-level semantic representations.

Our proposed framework addresses these dual challenges. To prevent representation degradation, we freeze the foundation backbone and tune only a lightweight MAP head to project features into a specialized identity subspace. Concurrently, our NearID protocol provides the first rigorous, automated evaluation standard explicitly designed to penalize background conflation and reward true object-level identity preservation rather than holistic visual similarity.

### 3 NearID

#### 3.1 Overview and Problem Formulation

We formulate identity preservation as a structured metric learning problem designed to explicitly disentangle object identity from background context (see Figure 2). NearID is a unified framework comprising three tightly coupled components: (i) a hierarchical contrastive objective that enforces an explicit similarity ordering between positives, near-identity distractors, and random negatives

- (Section 3.2), (ii) a large-scale matched-context dataset with curated identities and synthesized near-identity distractors that instantiate this training setting
- (Section 3.3), and (iii) a rigorous evaluation protocol based on discriminability margins and alignment with human and oracle judgments (Section 3.4).

Together, these elements isolate identity as the sole discriminative signal and enable principled training and reliable evaluation under matched-context confounders.

Training Tuple Construction. Each training sample is constructed as a tuple Ti = {ai,{gi,p}Pp=1,{ri,k}Kk=1} where ai denotes an anchor image of object identity i, gi,p are up to P positive views of the same identity against different backgrounds, ri,k are K near-identity distractors. A NearID distractor consists of a semantically similar but distinct object instance that is inpainted into the exact same background as the anchor ai. This matched-context construction removes all contextual shortcuts and forces the model to discriminate based solely on intrinsic identity cues.

Learning Objective. The goal is to learn an embedding function fθ : I → Rd such that, for a given anchor ai, in terms of cosine similarity, cos(ai,pi) > cos(ai,nneari ) > cos(ai,nrandi ), i.e., same identity > NearID distractor > random batch negative. This explicitly enforces a structured similarity hierarchy rather than treating all negatives equally.

Parameter-Efficient Adaptation. To avoid catastrophic forgetting and preserve the robust semantic priors of large-scale vision encoders, we freeze a pre-trained backbone (SigLIP2 [64]) and optimize only a lightweight Multi-head Attention Pooling (MAP) projection head [24,79].

Given spatial patch embeddings extracted from the frozen backbone, the MAP head selectively aggregates identity-salient features and projects them into a refined metric space. This design updates only ∼3.6% of total parameters while reshaping the embedding geometry to satisfy the identity hierarchy defined above (See Figure 3 left pane). In the following subsection, we formalize the extended hierarchical contrastive objective used to enforce this structure.

#### 3.2 NearID Loss

Standard contrastive objectives treat all negatives uniformly, failing to penalize background-induced correlations that allow models to shortcut on contextual cues rather than intrinsic identity features (illustrated in Figure 1 and Figure 2). We address this with the NearID loss (LNearID), a two-component objective that enforces a structured three-level similarity hierarchy: true identity matches above NearID distractors, and NearID distractors above generic batch negatives.

Notation. Let ϕ(x) denote frozen backbone features and let f denote the trainable MAP projection head applied on top of ϕ. For an image x, we define the

ℓ2-normalized embedding zx = ∥ff((ϕϕ((xx))))∥

∈ Rd. All pairwise comparisons use

2

⊤v τ , where τ > 0 is a fixed tem-

temperature-scaled similarity logits ℓu,v = u

perature hyperparameter (τ = 0.07). Since embeddings are ℓ2-normalized, dot products equal cosine similarities.

For each anchor image ai, {gi,p}Pp=1 positive views, and {ri,k}Kk=1 NearID distractors (See Section 3.1), we define their embeddings as ai = za

##### , and ri,k = zr

##### , gi,p = zg

i

i,p

. Let G = {gj,q} denote the global set of all positive embeddings gathered across devices (DDP), with |G| = M. Define the anchor’s own positive set Gi = {gi,q}Pq=1 ⊂ G and the batch-negative pool Bneg(i) = G \ Gi. Note that the NearID distractor set for anchor i, Ri = {ri,k}Kk=1, is defined per-anchor and is not included in G. In practice, we average only over valid positives and distractors using masks (omitted for clarity).

i,k

Discrimination term (Ldisc). For each positive index p, the discrimination term applies a softmax cross-entropy over the global positive pool augmented with

the NearID distractors as L(disci) ,p = −log exp(ℓai,gi,p)

g∈G exp(ℓai,g)+ Kk=1 exp(ℓai,ri,k).

This formulation is structurally related to InfoNCE [42] and multi-class Npair objectives [56], but with two key differences. First, the NearID distractors ri,k are explicitly included in the denominator alongside global batch candidates, providing targeted identity-confusing competition that standard batch negatives cannot supply. Second, unlike Supervised Contrastive Learning [28], which places all same-class positives in the numerator, we retain the other P−1 positive views in the denominator. This per-index formulation mitigates multi-positive collapse and ensures that each view is individually discriminable under background variation. The discrimination loss averages over valid positive indices:

Ldisc =

P

1 P

Ei L(disci) ,p . (1)

p=1

Ranking regularizer (Lrank). The discrimination term alone treats NearID distractors simply as additional negatives to push away. Under strong gradient pressure from highly confusable distractors, this can collapse the local semantic neighborhood, destroying the graded similarity structure that distinguishes “nearby but different” from “unrelated”, a known concern in metric learning with highly confusable negatives [70]. To preserve this structure, we add a ranking regularizer that encourages each NearID distractor to be ranked above the generic batch-negative pool. Define the batch-negative log-sum-exp as LSEi = log g∈B(i)

i,g). which acts as a smooth maximum over the batch-negative pool.

exp(ℓa

neg

For each NearID distractor ri,k, the ranking regularizer takes the form, L(ranki,k) = log 1 + exp LSEi − ℓa

i,ri,k .

which is a softplus penalty that decreases smoothly as the NearID distractor logit exceeds the aggregate batch-negative signal, and increases when batch negatives dominate. This is equivalent to the cross-entropy form, L(ranki,k) = −log exp(ℓ exp(ℓai,ri,k)

exp(ℓai,g).

ai,ri,k) +

(i) neg

g∈B

However, the softplus view makes the listwise ranking nature explicit: LSEi acts as a smooth maximum over the batch-negative pool, and the penalty enforces that each NearID distractor logit exceeds this threshold. Aggregating:

Lrank =

K

1 K

Ei L(ranki,k) . (2)

k=1

##### NearID loss. The complete objective combines discrimination and ranking:

LNearID = Ldisc + α Lrank, (3)

where α = 0.5 controls the ranking regularizer strength. Together, the two terms enforce the intended similarity ordering ℓa

i,ri,k ≳ LSEi.

i,gi,p > ℓa

ensuring that true positives are selected over all candidates (Ldisc), while NearID distractors remain closer than generic batch negatives (Lrank), preserving the graded semantic structure of the embedding space.

#### 3.3 NearID Dataset Construction

Dataset Foundation and Curation. Constructing a robust benchmark for object-level identity requires data with high viewpoint diversity and strict identity separation. While datasets such as Mind-The-Glitch (MTG) [13] and Subjects200k [61] offer valuable baselines, they are often limited by scale (e.g., 5K samples in MTG), lack of multi-view diversity (frequently restricted to a single reference against a plain white background in Subjects200k), and a primary focus on subject-level matching with a single reference on a plain white background always, rather than instance-level object identity under viewpoint and background changes. Furthermore, MTG explicitly cites significant dataset quality issues within Subjects200k, which further motivates our decision to improve upon it.

To address these limitations, we build our synthesis pipeline on the SynCD dataset [33], which inherently provides up to three distinct, high-quality positive viewpoints per identity across 90K initial samples. To guarantee absolute identity fidelity and prevent semantic overlap during training and evaluation, we apply a rigorous filtering protocol. We first discard the “deformable” partition (approximately half of the dataset) because the publicly released version lacks the exact foreground text descriptions required for precise generative conditioning and low diversity with 16 base classes. Within the remaining “rigid” object subset, we enforce strict uniqueness constraints based on paired textual descriptions and ObjaverseID [11]. This curation process yields our proposed dataset, NearID, comprising 19,386 unique object identities and 45,215 distinct images. Notably, 12,943 identities (67%) possess two distinct views, and 6,433 identities (33%) provide three distinct views. This scale and viewpoint coverage significantly surpasses the entire MTG dataset (limited to 5K identities with only 2 views). We create mutually exclusive train/val/test splits with 18,786/100/500 identities, respectively.

Multi-Model Synthesis Pipeline. Using this curated core subset, we generate near-identity distractors employing a diverse ensemble of SoTA diffusion models. This pipeline encompasses Stable Diffusion XL (SDXL) [47], FLUX.1 [34], Qwen-Image [72], and PowerPaint [83] (utilizing the BrushNet v2.1 pipeline [26]), which ensures a wide variance in generative priors, structural adherence, and artifact distributions. By leveraging multiple distinct architectures across varying latent dimensions and conditioning mechanisms, we prevent the synthesized distractors from sharing a trivial, uniform generative fingerprint. Overall, this approach yields 7 total negative inpainting configurations as distinct generative sources, resulting in a comprehensive collection of 316,505 distinct near-identity distractor images, enabling rigorous disentanglement of (i) inpainting artifacts,

- (ii) model-specific distributional biases, and (iii) other source-dependent generative fingerprints in the generated NearID training and evaluation.

To supplement our object-level distractors with fine-grained variations, we concatenate the MTG dataset, which inherently consists of part-level inpaintings from a single source (SDXL). To ensure equal sampling probability during training, the MTG subset is repeated four times to match the scale of our synthesized data. Generation is conducted using mixed precision, where supported by the model architecture, to optimize throughput during large-scale dataset construction. We use default parameters from publicly available code for each method, with slight modifications when utilizing adapters. Exact inference hyperparameters are detailed in the supplementary material.

Data Processing Pipeline. Generation is performed at two distinct resolutions: 5122 and the respective model-native resolutions (e.g., spatial scaling to align the shortest side with 640 for SD 1.5-based variants like BrushNet, or strictly 10242 for SDXL), except for BrushNet, which operates exclusively at its native resolution. Following the synthesis of the near-identity distractors, spatial standardization is enforced by downsampling any high-resolution outputs to 5122 utilizing Lanczos resampling. This ensures consistent dimensions for downstream evaluation metrics and contrastive training while mitigating aliasing artifacts.

#### 3.4 Evaluation Protocol

Given the tuple formulation introduced in Section 3.1, we evaluate identity discrimination under a matched-context setting designed to eliminate background shortcuts. The protocol analyzes similarity margins within strict identity–background triplets. For a given identity, let pi and pj denote two positive views of the same object captured against distinct backgrounds i and j. Let ni denote a NearID distractor: a semantically similar but distinct instance inpainted into the exact background of pi.

Bidirectional Discriminability Margin. We measure whether identity similarity across backgrounds exceeds similarity to a matched-context distractor. For a pair (i,j), we define the directed margins:

δi(→ij)j = s(pi,pj) − s(pi,ni), δj(ij→)i = s(pi,pj) − s(pj,nj). (4)

###### NearID (Ours)

###### SigLIP2

###### VSM

###### DINOv2

GPT-4o

Animal Object

Animal Object

Animal Object

Animal Object

Animal Object

0.60 0.49

0.55 0.43

0.52 0.33

0.49 0.43

0.18 0.19

0.15

0.24

0.25

0.35

0.37

0.38

0.43 0.47

0.44

0.44

Human Style

Human Style

Human Style

Human Style

Human Style

- Fig. 4: Per-category human alignment (M−H) on DreamBench++ [46]. Each radar shows the Pearson correlation between a metric and human concept-preservation judgments across four DB++ categories; NearID (Ours) is repeated as a dashed reference in each subplot. Despite training exclusively on rigid objects, NearID improves over the frozen SigLIP2 baseline on Animal (+0.105) and Human (+0.065), indicating that disentangling identity from context transfers across semantic domains. The expected decrease on Style (−0.092), a category entirely absent from training, confirms that the gains reflect genuine identity learning rather than general score inflation.

A margin trial is considered successful if δ > 0, indicating that cross-background identity similarity overrides matched-context confounders.

Micro-pooled Metrics. We aggregate directed margin trials using two measures:

Sample Success Rate (SSR). A sample is counted as successful only if all valid directed margins are positive (up to six per identity): Pairwise Accuracy (PA). The proportion of successful directed margin trials across the dataset, treating each margin independently:

N

⊮ ∀ (j, d) : δd(ij) > 0 , PA = {m ∈ M : δm > 0}

1 N

SSR =

, (5)

|M|

i=1

where N is the number of identities and the set of all directed margin trials M.

Alignment with Human and Oracle Judgments. To ensure our learned embeddings align with fine-grained identity preservation, we evaluate against the MTG dataset [13] and DreamBench++ (DB++) [46]. We utilize the MTG part-level oracle score, defined as 1−(|Ωp|/|Ωo|), where Ωp and Ωo represent the part and object masks, respectively. Hence, we also compute two Pearson [45] correlations:

M – O. Correlations between the cosine similarities and the oracle scores. M – H. Correlations between the cosine similarities and human scores on DB++.

All correlations are reported as the mean Pearson correlation, r¯, computed via Fisher’s z-transformation [15] over per-sample coefficients (see supplementary material for the aggregation formula).

- Table 2: DreamBench++ concept-preservation human alignment. Pearson correlations between human (H) and metric scores: G = GPT, D = DINO, C = CLIP, S = SigLIP2, NearID (Ours). Bolded numbers indicate improvement over the SigLIP2.

Concept Preservation (Metric–Human Alignment) H–H G–H D–H C–H S–H NearID–H

Method T2I Model

Textual Inversion [17] SD v1.5 0.684 0.625 ± 0.032 0.611 ± 0.016 0.610 ± 0.001 0.660 ± 0.003 0.675 ± 0.007 DreamBooth [51] SD v1.5 0.731 0.722 ± 0.011 0.632 ± 0.010 0.623 ± 0.015 0.652 ± 0.012 0.655 ± 0.023 DreamBooth LoRA [22,51] SDXL v1.0 0.651 0.637 ± 0.004 0.446 ± 0.013 0.481 ± 0.007 0.480 ± 0.006 0.546 ± 0.006 BLIP-Diffusion [35] SD v1.5 0.630 0.466 ± 0.019 0.311 ± 0.024 0.290 ± 0.004 0.363 ± 0.016 0.382 ± 0.002 Emu2 [57] SDXL v1.0 0.763 0.777 ± 0.004 0.774 ± 0.018 0.744 ± 0.021 0.766 ± 0.021 0.768 ± 0.028 IP-Adapter-Plus ViT-H [75] SDXL v1.0 0.530 0.424 ± 0.045 0.212 ± 0.017 0.262 ± 0.002 0.253 ± 0.002 0.306 ± 0.018 IP-Adapter ViT-G [75] SDXL v1.0 0.485 0.401 ± 0.021 0.261 ± 0.004 0.283 ± 0.031 0.248 ± 0.019 0.316 ± 0.041

Fisher-z (r¯ - Sec. 4.2) 0.648 ± 0.038 0.597 ± 0.056 0.492 ± 0.081 0.493 ± 0.074 0.516 ± 0.079 0.545 ± 0.071

### 4 Experiments

#### 4.1 Implementation Details

Architecture and Optimization. All embedding models in our proposed pipeline share a frozen SigLIP2-so400m-patch14-384 backbone, unless otherwise specified. We train only the Multihead Attention Pooling (MAP) projection head, which outputs a 1152 dimensional ℓ2-normalized embedding. This head-only tuning regime updates approximately 15M parameters out of the roughly 428M total model parameters. By training merely ∼3.6% of the overall network capacity, we ensure minimal computational overhead while preserving the robust zero-shot priors of the foundation model. Models are optimized using AdamW (η = 10−4, weight decay=10−4) in mixed-precision (fp16). We employ a cosine annealing schedule with a 100-step linear warmup, utilizing a global batch size of 128 across 11 epochs (approximately 3,350 gradient steps).

Role-Aware Data Augmentation. We apply role-aware stochastic foreground masking: backgrounds are blacked out for anchors (p=0.5), positives (p=0.2), and distractors (p=0.6), the latter to prevent trivial rejection via background border artifacts. Standard augmentations (color jitter, flipping, translation/scale) are applied per slot.

Joint Training. The NearID dataset is interleaved with the MTG training split (upsampled 4×). MTG’s part-level edits are subjected to Lrank without explicit regression targets, enabling continuous calibration from the data distribution.

#### 4.2 Results

We evaluate against frozen contrastive models (CLIP [49], SigLIP2 [64]), selfsupervised transformers (DINOv2 [43]), a VLM (Qwen3-VL30B [3]), and VSM [13], trained on MTG. SSR and PA are pooled across distractor sources via supportweighted averaging.

As shown in Table 1, frozen encoders fail under matched-context evaluation: SigLIP2 achieves only 30.74% SSR. NearID resolves this, reaching 99.17% SSR

###### SigLIP2

| |
|---|

Dim2

Dim 1

NearID (Ours)

| |
|---|

Dim2

Dim 1

- S1

- S2

- S3

- S4

- S5

- S6

- S7

Positive

Near-ID Neg.

- Fig. 5: KernelPCA visualization of Near-Identity separation. We project embeddings for n=7 identities (S1–S7) into 2D using identical KernelPCA [52] settings in both panels. Circles denote positives (same identity) and crosses denote matchedcontext NearID distractors (negatives). Compared to the frozen SigLIP2 baseline, NearID increases separation by pushing the distractors away from the corresponding positive clusters, providing visual evidence of improved near-identity discrimination.

and 99.71% PA. On MTG, all standard encoders score 0.0% SSR; even VSM reaches only 7.0%. NearID achieves 35.0% SSR with substantially improved oracle alignment (M–O: 0.465 vs. 0.180). On DB++ (Table 2), NearID improves metric-to-human correlation to 0.545 vs. 0.516 for SigLIP2. As shown in Figure 4, these gains generalize across DB++ categories, including animals and humans absent from training, confirming that identity-specific training transfers to real-world evaluation. Figure 3 right pane summarizes the resulting property profile across methods.

#### 4.3 Visualization of Embeddings

- Figure 5 shows KernelPCA [52] projections of NearID validation embeddings. The frozen baseline (left) entangles distractors with positives, while NearID (right) separates them into distinct clusters, with positives grouped but individually discriminable.

#### 4.4 Training Objective Ablation

We train the same MAP head on identical data with different loss functions (Table 3); only the loss differs.

Standard contrastive training is insufficient. InfoNCE improves SSR from 30.74% to 60.97% and achieves the highest M–H (0.555), but 61% SSR means nearidentity distractors still frequently outscore true positives.

Hierarchy is necessary but collapse is a risk. Circle+Ranking achieves 99.97% NearID SSR but collapses to M–H =0.141, well below the 0.516 frozen baseline. Oracle Ranking similarly collapses (M–H =0.167). These aggressive formulations over-specialize the embedding, motivating the softer softplus formulation in Lrank.

- Table 3: Training objective ablation. All variants share a frozen SigLIP2 backbone with a trainable MAP head, trained on NearID + MTG data for 11 epochs with identical hyperparameters. More details in the supplementary material about the specifics of each loss and the extension. †Representation collapse: high discrimination but degraded general-purpose alignment (see text). We observe that our configurations pose the best balance between the different adaptations to the NearID setting.

NearID MTG DB++ SSR ↑ PA ↑ M–O ↑ M–Opair ↑ SSR ↑ PA ↑ M–H ↑

Training Loss

None (frozen) 30.74 48.81 0.180 0.366 0.0 0.0 0.516 InfoNCE (sym., 1-pos) 60.97 75.26 0.267 0.418 8.0 12.0 0.555

+ R neg 99.57 99.79 0.236 0.267 64.0 74.0 0.251

+ Oracle Ranking† 86.34 92.25 0.299† 0.444† 7.0† 11.0† 0.167†

+ R neg + Oracle Ranking 99.60 99.89 0.247 0.277 65.0 74.5 0.227 SigLIP (BCE, ext., + hard>batch rank) 61.40 77.81 0.213 0.385 6.0 9.5 0.530 Circle (ext.,+hard>batch margin rank)† 99.97 99.99 0.264† 0.303† 67.0† 76.5† 0.141† LNearID (Ours) 99.17 99.71 0.465 0.486 35.0 46.5 0.545

+ Pos. Cohesion 99.31 99.78 0.459 0.485 36.0 47.0 0.541

NearID loss balances discrimination and alignment. LNearID attains 99.17% SSR while maintaining M–H =0.545, trading only 0.010 M–H relative to InfoNCE (which fails at 61% SSR).

Adding positive cohesion yields marginal gains (99.31% SSR) with comparable MTG SSR (36.0% vs 35.0%), confirming that the base LNearID suffices for the primary setting.

### 5 Conclusion

We presented NearID, a unified framework for learning identity-discriminative representations under matched-context confounders. We showed that widely used vision encoders systematically conflate object identity with background context: under our evaluation protocol, even the best frozen encoder achieves only 30.74% SSR when near-identity distractors share the reference background.

To address this, we introduced three tightly coupled components: (i) a largescale dataset with 19K identities and 316K+ matched-context distractors from four generative models, (ii) a two-component contrastive objective that enforces a strict similarity hierarchy without hard margins or oracle supervision, and

- (iii) a discriminability-based evaluation protocol grounded in similarity margins. Training only a lightweight MAP head on a frozen backbone, NearID achieves 99.17% SSR, improves part-level discrimination on MTG, and increases alignment with human judgments on DreamBench++.

The NearID dataset and evaluation protocol can serve as a diagnostic tool for any vision encoder, providing a rigorous test of whether learned representations capture genuine identity or merely exploit contextual shortcuts.

We hope NearID establishes a stronger standard for evaluating and learning identity-aware representations in personalized generation and image editing.

### Acknowledgements

The research reported in this publication was supported by funding from King Abdullah University of Science and Technology (KAUST) - Center of Excellence for Generative AI, under award number 5940 and Snap Research.

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35, 23716– 23736 (2022)
- 3. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025)

- 4. Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., Hullender, G.: Learning to rank using gradient descent. In: Proceedings of the 22nd International Conference on Machine Learning. p. 89–96. ICML ’05, Association for Computing Machinery, New York, NY, USA (2005). https://doi.org/10. 1145/1102351.1102363, https://doi.org/10.1145/1102351.1102363
- 5. Cao, Z., Qin, T., Liu, T.Y., Tsai, M.F., Li, H.: Learning to rank: from pairwise approach to listwise approach. In: Proceedings of the 24th International Conference on Machine Learning. p. 129–136. ICML ’07, Association for Computing Machinery, New York, NY, USA (2007). https://doi.org/10.1145/1273496.1273513, https://doi.org/10.1145/1273496.1273513
- 6. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)
- 7. Chen, H., Zuo, Z., Zhao, L., Li, J., Yang, J.: Conceptcraft: One-shot personalized text-to-image generation via object-background disentanglement. IEEE Transactions on Circuits and Systems for Video Technology 36, 133–146 (2026), https: //api.semanticscholar.org/CorpusID:280589664
- 8. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International conference on machine learning. pp. 1597–1607. PmLR (2020)
- 9. Cvejic, A., Eldesokey, A., Wonka, P.: Partedit: Fine-grained image editing using pre-trained diffusion models. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–11 (2025)
- 10. Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning (2023), https://arxiv.org/abs/2305.06500
- 11. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects (2022), https://arxiv.org/abs/2212.08051

- 12. Deng, J., Guo, J., Xue, N., Zafeiriou, S.: Arcface: Additive angular margin loss for deep face recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4690–4699 (2019)
- 13. Eldesokey, A., Cvejić, A., Ghanem, B., Wonka, P.: Mind-the-glitch: Visual correspondence for detecting inconsistencies in subject-driven generation. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)
- 14. Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X., Cao, Y.: Eva: Exploring the limits of masked visual representation learning at scale. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19358–19369 (2023)
- 15. Fisher, R.A.: Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. Biometrika 10(4), 507–521 (1915)
- 16. Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., Isola, P.: Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344 (2023)
- 17. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)
- 18. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al.: Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems 33, 21271–21284 (2020)
- 19. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: International conference on machine learning. pp. 1321–1330. PMLR

(2017)

- 20. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022)
- 21. Hochberg, D.C., Anschel, O., Shoshan, A., Kviatkovsky, I., Aggarwal, M., Medioni, G.: Towards quantitative evaluation metrics for image editing approaches. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7892–7900 (2024)
- 22. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)
- 23. Hu, J., Han, T., Ma, K., Gao, J., Yang, S., He, X., Luo, J., Wei, X., Zhang, W.: Positionic: Unified position and identity consistency for image customization. arXiv preprint arXiv:2507.13861 (2025)
- 24. Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., Carreira, J.: Perceiver: General perception with iterative attention. In: International conference on machine learning. pp. 4651–4664. PMLR (2021)
- 25. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021)
- 26. Ju, X., Liu, X., Wang, X., Bian, Y., Shan, Y., Xu, Q.: Brushnet: A plug-andplay image inpainting model with decomposed dual-branch diffusion. In: European Conference on Computer Vision. pp. 150–168. Springer (2024)
- 27. Keramati, M., Meng, L., Evans, R.D.: Conr: Contrastive regularizer for deep imbalanced regression. arXiv preprint arXiv:2309.06651 (2023)

- 28. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. Advances in neural information processing systems 33, 18661–18673 (2020)
- 29. Kilrain, C., Carlyn, D., Chae, J., Beery, S., Chao, W.L., Gu, J.: Finerpersonalization rank: Fine-grained retrieval examines identity preservation for personalized generation. arXiv preprint arXiv:2512.19026 (2025)
- 30. Kim, M., Jain, A.K., Liu, X.: Adaface: Quality adaptive margin for face recognition. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 18729–18738 (2022), https://api.semanticscholar.org/CorpusID: 247940176
- 31. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything

(2023), https://arxiv.org/abs/2304.02643

- 32. Ku, M., Jiang, D., Wei, C., Yue, X., Chen, W.: Viescore: Towards explainable metrics for conditional image synthesis evaluation. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 12268–12290 (2024)
- 33. Kumari, N., Yin, X., Zhu, J.Y., Misra, I., Azadi, S.: Generating multi-image synthetic data for text-to-image customization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16524–16534 (2025)
- 34. Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024)
- 35. Li, D., Li, J., Hoi, S.: Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems 36, 30146–30166 (2023)
- 36. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 19730–19742. PMLR (23–29 Jul 2023), https: //proceedings.mlr.press/v202/li23q.html
- 37. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International conference on machine learning. pp. 12888–12900. PMLR (2022)
- 38. Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J.N., Chang, K.W., Gao, J.: Grounded language-image pretraining (2022), https://arxiv.org/abs/2112.03857
- 39. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning (2023), https:// arxiv.org/abs/2304.08485
- 40. Meng, Q., Zhao, S., Huang, Z., Zhou, F.: Magface: A universal representation for face recognition and quality assessment (2021), https://arxiv.org/abs/2103. 06627
- 41. Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018)
- 42. Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018)
- 43. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- 44. Papantoniou, F.P., Lattas, A., Moschoglou, S., Deng, J., Kainz, B., Zafeiriou, S.: Arc2face: A foundation model for id-consistent human faces. In: European Conference on Computer Vision. pp. 241–261. Springer (2024)

- 45. Pearson, K.: Vii. note on regression and inheritance in the case of two parents. proceedings of the royal society of London 58(347-352), 240–242 (1895)
- 46. Peng, Y., Cui, Y., Tang, H., Qi, Z., Dong, R., Bai, J., Han, C., Ge, Z., Zhang, X., Xia, S.T.: Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855 (2024)
- 47. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 48. Qian, G., Wang, K.C., Patashnik, O., Heravi, N., Ostashev, D., Tulyakov, S., Cohen-Or, D., Aberman, K.: Omni-id: Holistic identity representation designed for generative tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8786–8795 (2025)
- 49. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)
- 50. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 51. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22500–22510 (2023)
- 52. Schölkopf, B., Smola, A., Müller, K.R.: Nonlinear component analysis as a kernel eigenvalue problem. Neural computation 10(5), 1299–1319 (1998)
- 53. Schroff, F., Kalenichenko, D., Philbin, J.: Facenet: A unified embedding for face recognition and clustering. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 815–823 (2015)
- 54. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: Dinov3 (2025), https://arxiv.org/abs/2508.10104
- 55. Singhania, A., Malani, K., Dhawan, R., Jain, A., Tandon, G., Sharma, N., Chakraborty, S., Batra, V., Phogat, A.: Beyond the pixels: Vlm-based evaluation of identity preservation in reference-guided synthesis (2025), https://arxiv.org/ abs/2511.08087
- 56. Sohn, K.: Improved deep metric learning with multi-class n-pair loss objective. In: Proceedings of the 30th International Conference on Neural Information Processing Systems. p. 1857–1865. NIPS’16, Curran Associates Inc., Red Hook, NY, USA

(2016)

- 57. Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Wang, Y., Rao, Y., Liu, J., Huang, T., Wang, X.: Generative multimodal models are in-context learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14398–14409 (2024)
- 58. Sun, Y., Cheng, C., Zhang, Y., Zhang, C., Zheng, L., Wang, Z., Wei, Y.: Circle loss: A unified perspective of pair similarity optimization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6398–6407

(2020)

- 59. Sun, Z., Fang, Y., Wu, T., Zhang, P., Zang, Y., Kong, S., Xiong, Y., Lin, D., Wang, J.: Alpha-clip: A clip model focusing on wherever you want (2023), https: //arxiv.org/abs/2312.03818
- 60. Sundaram, S., Fu, S., Muttenthaler, L., Tamir, N., Chai, L., Kornblith, S., Darrell, T., Isola, P.: When does perceptual alignment benefit vision representations? Advances in Neural Information Processing Systems 37, 55314–55341 (2024)
- 61. Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14940–14950 (2025)
- 62. Thoma, J., Paudel, D.P., Gool, L.V.: Soft contrastive learning for visual localization. Advances in neural information processing systems 33, 11119–11130 (2020)
- 63. Tian, Y., Fan, L., Isola, P., Chang, H., Krishnan, D.: Stablerep: Synthetic images from text-to-image models make strong visual representation learners. Advances in Neural Information Processing Systems 36, 48382–48402 (2023)
- 64. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025)
- 65. Vo, A., Nguyen, K.N., Taesiri, M.R., Dang, V.T., Nguyen, A.T., Kim, D.: Vision language models are biased. arXiv preprint arXiv:2505.23941 (2025)
- 66. Wang, Q., Cvejic, A., Eldesokey, A., Wonka, P.: Editclip: Representation learning for image editing (2025), https://arxiv.org/abs/2503.20318
- 67. Wang, Q., Bai, X., Wang, H., Qin, Z., Chen, A., Li, H., Tang, X., Hu, Y.: Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024)
- 68. Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., Xu, J., Xu, B., Li, J., Dong, Y., Ding, M., Tang, J.: Cogvlm: Visual expert for pretrained language models (2024), https://arxiv.org/abs/2311.03079
- 69. Wang, X., Han, X., Huang, W., Dong, D., Scott, M.R.: Multi-similarity loss with general pair weighting for deep metric learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5022–5030 (2019)
- 70. Wang, X., Han, X., Huang, W., Dong, D., Scott, M.R.: Multi-similarity loss with general pair weighting for deep metric learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5022–5030 (2019)
- 71. Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., Xie, S.: Convnext v2: Co-designing and scaling convnets with masked autoencoders (2023), https: //arxiv.org/abs/2301.00808
- 72. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)
- 73. Xia, F., Liu, T.Y., Wang, J., Zhang, W., Li, H.: Listwise approach to learning to rank: theory and algorithm. In: Proceedings of the 25th international conference on Machine learning. pp. 1192–1199 (2008)
- 74. Yang, Z., Bastan, M., Zhu, X., Gray, D., Samaras, D.: Hierarchical proxy-based loss for deep metric learning. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 1859–1868 (2022)
- 75. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023)
- 76. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca: Contrastive captioners are image-text foundation models. Trans. Mach. Learn. Res. 2022 (2022), https://api.semanticscholar.org/CorpusID:248512473

- 77. Zadrozny, B., Elkan, C.: Transforming classifier scores into accurate multiclass probability estimates. In: Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. pp. 694–699 (2002)
- 78. Zha, K., Cao, P., Son, J., Yang, Y., Katabi, D.: Rank-n-contrast: Learning continuous representations for regression. Advances in Neural Information Processing Systems 36, 17882–17903 (2023)
- 79. Zhai, X., Kolesnikov, A., Houlsby, N., Beyer, L.: Scaling vision transformers. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12104–12113 (2022)
- 80. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training (2023), https://arxiv.org/abs/2303.15343
- 81. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)
- 82. Zhang, X., Lu, Y., Wang, W., Yan, A., Yan, J., Qin, L., Wang, H., Yan, X., Wang, W.Y., Petzold, L.R.: Gpt-4v(ision) as a generalist evaluator for vision-language tasks (2023), https://arxiv.org/abs/2311.01361
- 83. Zhuang, J., Zeng, Y., Liu, W., Yuan, C., Chen, K.: A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In: European Conference on Computer Vision. pp. 195–211. Springer (2024)

(A) SimCLR (B) StableRep (C) NearID (Ours)

Distractors

Positives

Encoder Encoder … Encoder Encoder

Encoder Encoder Encoder … Encoder

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

…

…

Aug Aug

Aug Aug

Aug Aug

Aug Aug

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

…

…

[Figure 48]

[Figure 49]

[Figure 50]

Inpaint

Inpaint

SD

SD

Real Image

“Cute dragon…”

FLUX

FLUX

“Cute alien…”

- Fig. 6: Contrastive training paradigms. (A) SimCLR [8] forms positive pairs from two heavily augmented views of a single real image, learning pixel-level invariance via pairwise InfoNCE. (B) StableRep [63] replaces hand-crafted augmentations with generative diversity: multiple synthetic images produced from the same caption (different diffusion seeds) serve as multi-positive sets, learning caption-level invariance. (C) NearID (ours) draws positives from a curated multi-view dataset of up to three synthetic images per identity, generated via the SynCD pipeline across different backgrounds. In addition to these identity-preserving positives, NearID constructs near-identity distractors, visually similar but distinct instances inpainted into the same background as the anchor, providing a structured negative signal absent from both SimCLR and StableRep.

### A Contrastive Training Paradigm Comparison

- Figure 6 situates NearID within the broader landscape of contrastive representation learning. SimCLR [8] constructs positive pairs by applying strong stochastic augmentations (random cropping, color distortion, Gaussian blur) to a single real image, encouraging pixel-level invariance via pairwise InfoNCE. StableRep [63] replaces hand-crafted augmentations with generative diversity: a text-to-image diffusion model produces multiple images from the same caption under different noise seeds, and these caption-conditioned samples form multi-positive sets optimised with a supervised contrastive [28] objective, yielding caption-level invariance.

NearID differs from both paradigms along two axes. (i) Positives are neither augmented copies of a single image nor caption-conditioned generations; they are high-quality multi-view images produced by the SynCD [33] pipeline, which leverages Objaverse [11] 3D assets as a geometric prior: depth-conditioned diffusion generation (e.g., FLUX.1-Depth) combined with cross-view feature warping ensures strict 3D-consistent multi-view positives (up to three views per identity; Section 3.3). (ii) NearID introduces near-identity distractors, visually similar yet non-matching instances inpainted into the same background as the anchor, as a structured negative signal that is absent from both SimCLR and StableRep. The NearID loss (Equation (3)) leverages these distractors to enforce a threelevel similarity ordering (identity > distractor > batch negative), preserving the graded semantic structure rather than treating all non-matching samples with

uniform repulsion. We note, however, that the naive solution of placing nearidentity distractors directly in the softmax denominator, without the ranking regulariser, achieves near-perfect discrimination (Table 10) but destroys alignment with both oracle scores and human judgments in the process.

### B Dataset Construction Details B.1 Inpainting Details

Synthesis Engines and Hyperparameters. To maximize the photorealism and contextual coherence of the generated distractors, each model is configured to its optimal operational regime:

- – PowerPaint (BrushNet Variant): We utilize the PowerPaint-v2 architecture, built upon a Stable Diffusion v1.5 backbone (Realistic Vision V6.0) and integrated with a BrushNet conditioning model [26] for disentangled mask-image guidance. Generation is driven by learned task tokens (e.g., text-guided object synthesis) to precisely govern the inpainting behavior. Inference is executed over 45 denoising steps utilizing the UniPC Multistep Scheduler, a classifier-free guidance (CFG) scale of 7.5, and a fitting degree (brushnet conditioning scale) of 1.0.
- – SDXL-Inpainting: We utilize the SDXL-1.0-inpainting-0.1 checkpoint operating at 10242 resolution. Distractor synthesis is performed over 30 denoising steps with a CFG scale of 8.0 and an inpainting strength of 0.99 to ensure maximal adherence to the unmasked context. A comprehensive negative prompt is applied to suppress text, watermarks, and low-resolution artifacts.
- – FLUX.1 Frameworks: We leverage the guidance-distilled FLUX.1-dev ecosystem to generate structurally consistent distractors. Masked region synthesis (FLUX.1-Fill) is executed for 50 steps with a high guidance scale of 30.0. For topology-preserving structural edits, we employ FLUX.1-CannyInpaint configured with 28 steps, a guidance scale of 7.0, and an inpainting strength of 0.99, guided by edge maps extracted via dual thresholds

(τlow = 50,τhigh = 200). Negative prompting is inherently omitted for all FLUX.1 variants.

- – Qwen-Image (ControlNet): We integrate the InstantX ControlNet with the Qwen-Image backbone for accelerated distractor generation. To facilitate high-throughput dataset creation, we fuse an 8-step Lightning LoRA (Qwen-Image-Lightning-8steps-V1.1) and utilize the Flow-Match Euler Discrete Scheduler, reducing the inference trajectory to 8 steps and the true CFG scale to 1.0.

### C Additional Qualitative Results

We present qualitative comparisons on both the NearID and MTG evaluation sets. For each sample we show per-image similarity scores for four metrics:

S=SigLIP2 (frozen), V =VSM, VL=Qwen3-VL30B, and N=NearID (ours). Scores are computed relative to the anchor; the anchor itself shows no score. We can see visualization of MTG in Figure 8 and NearID in Figure 7.

### D Computational Cost

Training the NearID MAP head (15M trainable parameters) for 11 epochs (∼3,350 gradient steps, batch size 128) requires approximately 6.5 hours on a single NVIDIA A100 (80GB) in mixed-precision (fp16), at roughly 7s per step including periodic checkpointing and evaluation callbacks. Because the SigLIP2 backbone is frozen and only the lightweight MAP head is updated, the compute budget is modest: a single run costs ∼6.5 A100-hours, and the full ablation suite (9 loss variants + 5 data-split runs + 5 α-sweep runs) totals ∼124 A100-hours of training. The additional β-sweep (3 cohesion variants at α=1.0) contributes a further ∼20 A100-hours. Offline evaluation of a single NearID checkpoint across all 9 distractor sources and the MTG benchmark takes approximately 10 minutes on one A100, adding negligible overhead.

VLM baseline cost. The Qwen3-VL30B VLM baseline is substantially more expensive to evaluate. Each NearID sample requires a separate autoregressive VLM inference with structured JSON output (∼260 tokens), consuming ∼72GB of VRAM on an A100 (each sample comparisons are batched with a fixed seed). Evaluating a single distractor source (500 test samples) takes approximately 3 hours, and the full NearID benchmark (9 sources × 2 masking conditions) requires ∼54 A100-hours, roughly 324× the cost of embedding-based evaluation. MTG evaluation (100 samples, 6 pairwise comparisons each) adds a further ∼2– 3 hours. This cost disparity highlights a practical advantage of learned identity embeddings: NearID achieves superior discrimination and oracle alignment at a fraction of the inference budget.

### E Additional Quantitative Results

#### E.1 Loss Decomposition

- Table 4 decomposes the training objectives compared in the ablation study (Table 3). All variants share the same frozen backbone and MAP head; only the loss function differs. The key distinction lies in how NearID distractors (R) enter each formulation: the standard InfoNCE baseline (Table 3, row 1) receives distractors from the data pipeline but ignores them entirely in the loss computation (∅), relying solely on in-batch negatives. All other objectives actively use R, either as additional candidates in the softmax denominator (D), as targets of a separate pairwise or hinge constraint (S), or exclusively for oracle-supervised ranking (R).

#### E.2 Correlation Aggregation

All per-sample Pearson correlation coefficients (rk) are aggregated using Fisher’s z-transformation [15] to ensure robustness under varying sample sizes:

r¯ = tanh

N

1 N

tanh−1(rk) . (6)

k=1

This applies to all reported correlation metrics: M–O, M–Opair, and M–H.

#### E.3 Score Distribution Analysis

In Figure 9, we showcase that NearID follows the oracle curve of scores the closest.

#### E.4 Per-Source Discrimination Breakdown

The main results (Table 1) report SSR and PA averaged over seven fill-based inpainting sources: four used during training (FLUX.1-Fill and Qwen-Image, each at 512 and native resolution) and three unseen generators (PowerPaint and SDXL-Inpaint at two resolutions). Table 5 provides a complete per-source breakdown, additionally including two Canny-guided inpainting sources (FLUX.1Canny-Inpaint at 512 and 1024) that combine edge-map conditioning with masked fill and are excluded from the main evaluation due to the additional structural prior. Frozen baselines (SigLIP2, Qwen3-VL 30B, VSM [13]) exhibit large crosssource variance (SSR 12–67%), whereas NearID maintains ≥97.8% SSR across all nine configurations with only a 0.1% gap between training and unseen fillbased averages.

#### E.5 Training Data Ablation

Table 6 isolates the contribution of each training data component under a fixed loss (LNearID, Table 3). Three findings emerge. (1) Near-identity distractors are the dominant signal: removing the NearID dataset while retaining MTG reduces SSR by 40.8% (from 99.2% to 58.3%), whereas removing MTG costs only 0.8%. This confirms that exposure to matched-context distractors during training is essential for learning background-invariant identity representations. (2) Source diversity matters: training with a single inpainting engine (FLUX.1-Fill, 1source) versus four engines drops SSR by 11.5%

- (98.4% → 86.9%). Diverse distractor styles prevent the model from overfitting to generator-specific artifacts. (3) MTG provides complementary but secondary benefit: the modest +0.8% NearID SSR gain from adding MTG suggests that part-level edits refine the embedding space without fundamentally altering its structure, consistent with MTG’s role as a fine-grained auxiliary signal. Conversely, training on MTG data alone yields the highest MTG SSR (49.0% vs 35.0% for the full pipeline), but at the cost of severely degraded object-level

NearID discrimination (58.3% SSR). To ensure a fair step-count comparison, the MTG-only setup applies 8× upsampling of the MTG split (5,000 identities ×8 = 40,000 effective samples), which matches the scale of the full joint pipeline (NearID ≈20,000 samples + MTG ×4 = 40,000 total). This reveals a critical failure mode: part-level edits alone cannot teach the model to separate different instances sharing the same background, since MTG pairs always depict the same object with localized modifications rather than a different object in matched context. The full pipeline strikes the best balance: strong NearID discrimination while retaining meaningful MTG sensitivity (35.0% SSR, 46.5% PA), confirming that near-identity distractors and part-level edits serve complementary but asymmetric roles, and that the part-level dataset alone is insufficient.

#### E.6 Positive Cohesion Variant

The “+Pos. Cohesion” variant in Table 3 extends LNearID (Equation (3)) with a term that explicitly pulls multiple positive views of the same identity toward

a shared prototype. Given P valid positive embeddings {gi,p}Pp=1 for anchor i, we compute the ℓ2-normalized prototype

P p=1 gi,p P p=1 gi,p 2

, (7)

g¯i =

and define the cohesion loss as the average cosine distance of each positive to this prototype:

P

1 P

1 − cos(gi,p, g¯i) . (8)

Lcoh =

p=1

The total objective becomes L = LNearID +β Lcoh with β = 0.1 (see Section E.9 for an ablation over β). Samples with fewer than two valid positives contribute zero cohesion loss.

The rationale is that Ldisc (Equation (1)) separates each positive individually from negatives but does not explicitly reduce variance among positives. Lcoh was hypothesised to directly tighten the positive cluster, improving intra-identity compactness and reducing cross-identity overlap in the embedding space.

In practice (Table 3), the cohesion variant yields slightly higher NearID SSR

- (99.3% vs 99.2%) and comparable MTG SSR (36.0% vs 35.0%). However, NearID

PA, M–O, and M–H remain comparable, indicating that the base LNearID already produces sufficiently compact positive clusters for the primary identity discrimination task.

#### E.7 Extended Baselines: Foreground Masking and VLM Scaling

To disentangle background dependence from identity-discrimination capacity, we evaluate all methods under two conditions: Full image (standard) and FG only (foreground extracted via oracle segmentation mask, background removed). We

additionally report Qwen3-VL at three model scales (4B, 8B, 30B) to assess whether VLM scaling can compensate for background confusion. Beyond objectlevel discrimination (SSR/PA), we also report the MTG oracle correlation (M– O) under both conditions for embedding-based methods, measuring how well each model’s similarity scores track the true magnitude of part-level edits when background context is removed.

Table 7 reveals four key findings. (1) Frozen embeddings are heavily background-dependent: removing the background improves CLIP by +41.0% SSR, DINOv2 by +43.3%, and SigLIP2 by +33.9%, confirming that these encoders rely substantially on contextual cues when the foreground instance is ambiguous. (2) VLMs benefit from masking but with diminishing returns at scale: Qwen3-VL4B gains +11.0%, 8B gains +13.5%, but 30B gains only +5.5% SSR from background removal. This suggests that larger VLMs partially learn to discount background cues through their language-grounded reasoning, but remain susceptible to matched-context confusion; even explicit anti-background prompting (Section E.11) does not fully resolve the problem. The MTG oracle alignment (M–O) under FG masking shows an inconsistent pattern across VLM scales (4B: −0.05; 8B: +0.12; 30B: −0.05), suggesting that the ability to leverage part-level similarity cues does not improve monotonically with model scale or background removal. (3) NearID is inherently backgroundinvariant for discrimination: NearID shows a slight SSR decrease (−3.8%) under foreground masking, indicating that it has learned to extract identity features from full images without background shortcuts. This is qualitatively distinct from the frozen encoders, which rely so heavily on contextual cues that removing the background yields gains of +34–43% SSR: for NearID the background is not a shortcut but a complementary signal that the model incorporates without becoming dependent on it, so removing it causes a minor performance regression from mask-boundary information loss rather than from background dependence. VSM [13] is similarly unaffected, consistent with its internal use of foreground segmentation; its M–O (0.394) is unchanged under external masking. (4) Background removal improves part-level oracle alignment: despite stable discrimination, NearID’s M–O correlation with MTG oracle scores improves substantially under FG masking (0.465 → 0.641, +0.18). This indicates that while NearID already discriminates objects correctly regardless of background, the background context still introduces noise into the magnitude of similarity differences for part-level edits. Frozen SigLIP2 shows a similar M–O gain (+0.19), while DINOv2 is unaffected (−0.02). The cohesion variant shows the highest FG M–O (0.648, vs. 0.641 for NearID), a marginal gain that mirrors its small MTG SSR improvement (36.0% vs. 35.0%) observed in the ablation study (Table 3).

#### E.8 Ranking Weight Ablation (α)

The NearID loss combines a discrimination term Ldisc with a ranking regulariser Lrank weighted by α (Equation (3)). Table 8 varies α while holding all other hyperparameters fixed. Even without the ranking term (α=0), the discrimination

loss alone achieves near-perfect NearID SSR (99.6%) because near-identity distractors already participate in the softmax denominator. Oracle alignment (M– O) generally increases with α up to 1.0, while discrimination (SSR) degrades mildly, revealing a clear trade-off between graded similarity structure and discrimination sharpness. At α=2.0 both SSR and M–O decline, indicating that an overly dominant ranking signal begins to collapse the representation. We adopt α=0.5 as the default configuration in the main paper (Table 1), as it provides the most balanced operating point: near-perfect discrimination (SSR=99.2%, within 0.4% of the α=0 ceiling) while achieving an M–O of 0.465, more than 1.5× the alignment obtained without ranking (α=0: 0.306). Among the values studied, α=1.0 achieves the highest oracle alignment (M–O=0.542) and secondbest human alignment (M–H=0.571, narrowly behind α=2.0 at 0.573 but with substantially better SSR and M–O), making it the operating point with the strongest stable alignment signal. We therefore study how positive cohesion (β) interacts with the ranking term at this value to provide guidance on hyperparameter choices in the high-alignment regime.

#### E.9 Positive Cohesion Weight Ablation (β)

The combined objective LNearID + β Lcoh (Section E.6) extends LNearID with a positive cohesion term weighted by β, designed to draw multi-view embeddings of the same identity closer via a prototype-centred objective. Table 9 evaluates β ∈ {0,0.1,0.2,0.3} at α=1.0, where the ranking term exerts its strongest stable influence. Contrary to expectation, adding positive cohesion (β > 0) consistently degrades oracle alignment (M–O), MTG discrimination (SSR), and human alignment (M–H) relative to β=0. NearID SSR remains nearly unchanged (≈98.8%), indicating that the cohesion signal does not improve identity discrimination but instead perturbs the representation away from the graded similarity structure learned by the ranking term alone. We conclude that at α=1.0 the ranking term already provides sufficient pull towards identity-consistent embeddings; adding an explicit cohesion loss introduces redundant gradient signal that interferes with fine-grained alignment. This conclusion extends to the default α=0.5 setting: in Table 3, the +Pos.Cohesion variant (R9, α=0.5, β=0.1) yields M–O=0.459 versus 0.465 for the base LNearID, confirming that the ranking term at α=0.5 already provides sufficient alignment signal and that β>0 is detrimental across both operating points. These results further support α=0.5 (without cohesion) as the recommended default.

#### E.10 InfoNCE Component Ablation: Distractors and Ranking

- Table 3 presents four InfoNCE variants that progressively add near-identity distractors (Rneg) and oracle-supervised ranking to a standard symmetric InfoNCE baseline. Table 10 isolates these four rows alongside the frozen backbone and the full NearID loss, with DB++ M–H values now available for all variants (Table 4 for mathematical definitions of each component).

The component decomposition reveals three distinct mechanisms and their failure modes.

#### (1) Distractors in the denominator drive discrimination but de-

grade alignment. Adding near-identity distractors as additional negatives in the softmax denominator (+Rneg, R1-neg) raises NearID SSR from 61.0% to 99.6% and PA from 75.3% to 99.8%, confirming that the mere presence of confusable negatives in the contrastive pool is sufficient for near-perfect discrimination. However, this comes at a substantial cost: M–O drops from 0.267 to 0.236 and M–H from 0.555 to 0.251. The model learns a binary identity decision boundary that ignores graded perceptual similarity, producing embeddings poorly aligned with both oracle scores and human judgments.

#### (2) Oracle ranking alone is insufficient without distractors in the

denominator. The +Oracle Ranking variant (R2) adds a RankNet-style pairwise ranking loss over oracle-ordered distractors without placing them in the softmax denominator. This moderately improves SSR (61.0% → 86.3%) but does not close the gap to near-perfect discrimination, and collapses the representation: M–H drops to 0.167 (below even the frozen backbone’s 0.516), indicating severe overfitting to the ranking signal. Combining both mechanisms (+Rneg +Oracle, R2-neg) recovers discrimination (99.6% SSR) but M–H remains degraded at 0.227, showing that oracle-supervised pairwise ranking on top of a one-directional InfoNCE base does not yield well-calibrated embeddings.

#### (3) The NearID formulation resolves the discrimination–alignment

trade-off. LNearID (Equation (3)) achieves 99.2% SSR while maintaining the highest M–H (0.545) among all trained variants, and an M–O of 0.465 that is more than 1.5× higher than any pure-InfoNCE variant. The critical difference is architectural: NearID uses a multi-positive symmetric softmax with distractors in the denominator (not one-directional) and replaces pairwise oracle ranking with an unsupervised softplus ranking regulariser Lrank (Equation (2)) that encourages distractors to rank above batch negatives without requiring oracle labels. This combination preserves graded similarity structure (high M–O, M–H) while achieving comparable discrimination to the brute-force denominator approach. The α-sweep (Table 8) further confirms that the ranking term controls the discrimination–alignment trade-off smoothly, with α=0.5 providing the best balance for the primary setting.

#### E.11 VLM Judge Prompt Template

Recent work has shown that multimodal LLMs struggle with multi-image comparison tasks, frequently failing to detect fine-grained differences between visually similar image pairs [32]. This limitation is particularly relevant for identity evaluation under matched-context conditions, where two images share the same background but differ only in the foreground instance. VLM behavior can be strongly influenced by background patterns [65], and when presented with full images, VLMs tend to conflate scene-level similarity with instance-level identity.

To provide the strongest possible VLM baseline, we evaluate Qwen3-VL30B [3] with a carefully designed structured prompt (Figure 10) that explicitly instructs

the model to (i) ignore background and scene cues, (ii) focus only on objectinstance evidence such as unique markings and fine geometry, and (iii) distinguish instance identity from category-level similarity. The model receives two images and returns a JSON object containing instance-level match/conflict cues, a confidence flag, and an integer score on a 0–10 rubric. All Qwen3-VL models (4B, 8B, 30B) are loaded in BF16 precision with FlashAttention-2 for memoryefficient inference. Because the structured JSON response (with evidence lists and metadata) typically spans ∼250 tokens, we set the maximum generation length to 512 tokens; shorter budgets cause truncated output and unparseable responses, silently degrading evaluation coverage. We map the VLM score to a cosine-similarity proxy via s = score/10 so that the same SSR/PA evaluation protocol applies uniformly across all methods.

Despite this optimized prompting strategy, Qwen3-VL30B achieves only 49.7%

SSR on our NearID benchmark (Table 1), compared to 99.2% for NearID. While the VLM substantially outperforms frozen embedding baselines (SigLIP2: 30.7%), the gap to NearID confirms that even state-of-the-art VLMs with explicit anti-background instructions remain susceptible to matched-context confusion, underscoring the need for specialized identity embeddings trained with near-identity distractors.

#### Scope and Future Directions

The NearID benchmark is designed for concept-preservation evaluation, where the primary criterion is whether a generated or retrieved image depicts the correct object instance, independently of scene context. This setting reflects the predominant use case in subject-driven generation [46,51], where faithfulness to the reference identity is the central objective. Text-guided editing techniques, however, impose a qualitatively different requirement: the output must simultaneously satisfy the textual editing instruction and preserve the identity of the source concept. These two objectives are often in tension, and characterising the appropriate trade-off is not addressed by the current benchmark, whose near-identity distractors hold the editing directive fixed by construction. A rigorous evaluation of this balance would require a dedicated editing benchmark comprising paired source and result images, explicit textual instructions, qualityfiltered outputs from state-of-the-art editing methods, and human judgements of both edit-intent adherence and concept fidelity. Extending the NearID evaluation framework to this setting is a natural direction for future work.

NearID Dataset

Anchor Positive 1 Positive 2 Distractor 1 Distractor 2

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

- Fig. 7: NearID qualitative results. Each row shows one identity: Anchor | Positive 1 | Positive 2 | Distractor 1 | Distractor 2. Positives depict the same instance in a different background; near-identity distractors show a different but visually similar instance inpainted into the anchor’s original background. NearID (N) consistently assigns higher scores to positives than to distractors across diverse object categories (storage crates, conch shells, basketballs, coffee cups, spacecraft). In contrast, frozen SigLIP2 (S) frequently assigns distractor scores on par with or exceeding positives; for instance, in the seashell row S rates Distractor 2 at 0.97, actually exceeding Positive 1 (0.79), while N correctly suppresses the distractor to 0.71. The VLM judge (VL) is inconsistent: near-zero scores for the basketball row (0.10) yet 1.00 for a spacecraft distractor, reflecting the VLM’s tendency toward category-level rather than instance-level reasoning. NearID successfully resolves these confounds, maintaining a clear margin between positives and near-identity distractors in all five categories shown.

MTG Dataset

Anchor Positive Distractor 1 Distractor 2

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

- Fig. 8: MTG qualitative results. Each row shows one identity: Anchor | Positive | Distractor 1 | Distractor 2. Distractors are generated by inpainting a localized part edit (e.g., the central motif of a brooch, the toe-cap design of a boot) while preserving the overall object shape and background; Oracle (O) measures the fractional area of the edited region. The red contour marks the edited part boundary; the green contour traces the full object extent. In the ornate brooch row, frozen SigLIP2 (S = 0.93) actually ranks Distractor 1 higher than the genuine Positive (0.89), a failure caused by overall appearance similarity. NearID (N) correctly ranks Positive (0.96) above both distractors (0.79, 0.84), detecting the fine-grained motif difference despite near-identical backgrounds. In the boot row, VSM (V) assigns Distractor 2 a score of 0.97, nearly identical to the Positive (0.98), whereas NearID cleanly separates them (0.95 vs. 0.66) even though the Oracle score (0.47) confirms a moderate but genuine edit. These examples illustrate that NearID captures part-level identity differences beyond what holistic embedding models detect.

- Table 4: Training objective decomposition (Table 3). Each objective combines a discrimination term Ldisc that separates positives from all negatives, and an optional hierarchy term Lrank that enforces ordering among negative types, yielding L=Ldisc+α Lrank. Notation (cf. Section 3.2): ℓu,v= cos(u, v)/τ; ℓp, ℓr, ℓb denote logits to a positive, a NearID distractor, and a batch negative, re-

spectively; LSEb= log g∈B eℓa,g (log-sum-exp over batch negatives); [·]+= max(·, 0); sp(·)= log(1+e(·)). R usage indicates how the NearID distractor set R enters each loss: D = in softmax denominator (competes directly with positives); S = separate pairwise or hinge term; R = pairwise ranking only; --- = batch-implicit negatives only (no explicit R mechanism). †Representation collapse (Table 3). Margin = the loss includes an explicit numeric margin m between similarity scores. Oracle = the ranking term is supervised by ground-truth edit severity (the ratio of edited area to object area); when ✗, ranking is unsupervised and purely structural (distractor > batch negative), requiring no per-sample annotations.

usageMarginOracle

R

Training Loss Ldisc Lrank

InfoNCE (sym.) [49] Symmetric softmax CE over

— — ✗ ✗

global pool G; single positive per anchor

+ R neg (same); distractors appended as

###### — D ✗ ✗

extra negatives in softmax over G

+ Oracle† [4] (same as InfoNCE) RankNet on

###### R ✓ ✓

oracle-ordered distractor pairs:

−log σ(ℓri−ℓrj −m)

+ R neg + Oracle (same as +Rneg) (same as +Oracle) D+R ✓ ✓ SigLIP+Rank [80] Pairwise sigmoid BCE over G;

S ✗ ✗

Log-sigmoid: −log σ(ℓrk−LSEb)

BCE(ℓr, 0) treats distractors as negatives

Circle+Rank† [58] Circle Loss with adaptive

S ✓ ✗

Hinge:

1 K k[ℓmaxb −ℓrk+m]+

weighting; R pooled with batch negatives

LNearID (Ours) [42] Softmax CE over G∪R:

###### D+R ✗ ✗

Softplus: sp(LSEb−ℓrk) (Equation (2))

multi-positive, per-index −ℓp+ log G eℓg+ k eℓrk ; (Equation (1))

(same) D+R ✗ ✗

+ Pos. Cohesion + β· P1 p(1−cos(gp,g¯)); pulls

positives toward prototype g¯

ECDF Frozen Embeddings (SigLIP2, CLIP, DINO)

| |Oracle<br><br>SigLIP2<br><br>CLIP<br><br>DINO| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

0.8

0.6

ECDF

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Score

(a) Frozen embedding baselines

ECDF VLM, VSM & NearID

| |Oracle<br><br>VLM (Qwen 30B)<br><br>VSM<br><br>NearID (Ours)<br><br>NearID + Cohesion<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

0.8

0.6

ECDF

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Score

(b) VLM, VSM & NearID (Ours)

- Fig. 9: MTG score distribution alignment with the Oracle. Empirical CDF of per-edit identity-preservation scores on the MTG benchmark [13]. The Oracle score is defined as 1−(|Ωp|/|Ωo|), where Ωp and Ωo are the part and object masks respectively, reflecting the true magnitude of each edit. (a) Frozen embeddings (SigLIP2, CLIP, DINO) cluster near 1.0 regardless of edit severity, indicating insensitivity to partlevel changes. (b) Both NearID configurations produce score distributions that track the Oracle far more closely than VLM or VSM baselines, demonstrating improved sensitivity to fine-grained identity-altering edits.

- Table 5: Per-source discrimination breakdown (full image). SSR (%) and PA (%) for each inpainting source used to generate NearID distractors. The main paper (Table 1) reports results averaged over seven fill-based sources; here we additionally include two Canny-guided sources (FLUX.1-Canny) that are excluded from the main evaluation. Training sources (†) supply distractors during NearID training; the remaining sources are unseen, testing cross-generator generalization. NearID achieves ≥ 97.8% SSR across all nine configurations, with only 0.1% difference between training and unseen fill-based averages.

SigLIP2 Qwen3-VL VSM NearID (Ours) Generator Res. n SSR PA SSR PA SSR PA SSR PA Training sources

FLUX.1-Fill† 512 500 35.8 55.7 48.3 68.3 23.7 38.7 99.4 99.8 FLUX.1-Fill† 1024 500 27.4 44.8 38.9 60.0 15.7 25.6 99.4 99.7 Qwen-Image† 512 500 33.8 51.4 48.1 67.7 27.9 42.2 99.2 99.7 Qwen-Image† 1328 500 16.2 32.1 43.3 64.7 26.7 39.3 98.2 99.5

Unseen fill-based generators (included in main results)

PowerPaint 512 500 34.0 52.7 67.1 83.2 51.2 66.8 99.2 99.8 SDXL-Inpaint 512 500 42.0 59.3 60.5 79.7 54.4 72.7 99.8 99.9 SDXL-Inpaint 1024 500 26.0 45.6 41.9 60.7 25.3 41.5 99.0 99.6

Unseen Canny-guided generators (excluded from main results)

FLUX.1-Canny 512 500 27.4 47.4 56.3 75.8 26.9 41.4 97.8 99.1 FLUX.1-Canny 1024 500 25.8 47.8 40.5 62.6 12.0 22.0 98.8 99.6

Avg. (training, 4) — 2 000 28.3 46.0 44.6 65.2 23.5 36.5 99.0 99.7 Avg. (unseen fill, 3) — 1 500 34.0 52.5 56.5 74.5 43.6 60.3 99.3 99.8 Avg. (unseen canny, 2) — 1 000 26.6 47.6 48.4 69.2 19.5 31.7 98.3 99.3 Avg. (all 7, main) — 3 500 30.7 48.8 49.7 69.2 32.1 46.7 99.2 99.7 Avg. (all 9) — 4 500 29.8 48.5 49.4 69.2 29.3 43.4 99.0 99.6

- Table 6: Training data ablation. All rows use LNearID (Table 3) with identical hyperparameters and a frozen SigLIP2 backbone with trainable MAP head. Data scale is accounted for; in the MTG setup, we use x8 to match the number of steps in our main setting. NearID src = number of inpainting sources providing near-identity distractors during training; MTG = whether the MTG part-level dataset is included. †Object-level discrimination failure: despite the highest MTG SSR/PA, NearID SSR drops to 58.3%, indicating that part-level data alone cannot teach background-invariant identity separation.

NearID MTG

Training Data NearID src MTG

SSR ↑ PA ↑ M–O ↑ M–Opair ↑ SSR ↑ PA ↑ None (frozen) — — 30.74 48.81 0.180 0.366 0.0 0.0 NearID (Ours) 4 ✓ 99.17 99.71 0.465 0.486 35.0 46.5 NearID only 4 ✗ 98.37 99.47 0.318 0.457 4.0 6.0 MTG only† 0 ✓ 58.34 75.64 0.349 0.395 49.0 62.0 NearID1 only 1 ✗ 86.86 93.77 0.326 0.456 3.0 5.5

- Table 7: Effect of foreground masking and VLM scaling on NearID discrimination and MTG oracle alignment. Full: evaluation on unmodified images. FG: evaluation on foreground-only crops (background removed via oracle mask). ∆: absolute change from masking (FG − Full). Frozen embeddings gain +34–43% SSR from background removal, confirming strong background dependence. VLMs also benefit (+5 to +14%), with diminishing gains at larger scale. NearID SSR is nearly unaffected (−3.8%), validating inherent background invariance. Notably, NearID oracle alignment (M–O) improves substantially under FG masking (+0.18), suggesting that background context adds noise to part-level correlation even when object-level discrimination is already robust. †VSM uses foreground masks internally, so external masking has no additional effect. ‡Qwen3-VL M–H on DB++ is omitted from Table 1 as the autoregressive VLM evaluation on DB++ was skipped due to computational cost (∼324× embedding-based evaluation).

Method

Full Image FG Only ∆ ∆ SSR↑ PA↑ M–O↑ SSR↑ PA↑ M–O↑ SSR M–O

CLIP ViT-L/14 10.3 20.9 0.239 51.3 71.6 0.297 +41.0 +0.06 DINOv2 ViT-L/14 20.4 34.6 0.324 63.7 77.4 0.301 +43.3 −0.02 SigLIP2-SO400M 30.7 48.8 0.179 64.7 80.5 0.365 +33.9 +0.19 VSM [13] 32.1 46.7 0.394 32.0 46.7 0.394† −0.1 ≈0

Qwen3-VL 4B [3] 13.4 32.9 0.012 24.4 44.3 −0.046 +11.0 −0.06 Qwen3-VL 8B 27.5 46.9 0.134 40.9 58.1 0.252 +13.5 +0.12 Qwen3-VL 30B 49.7 69.2 0.219 55.2 74.2 0.173 +5.5 −0.05

NearID (Ours) 99.2 99.7 0.465 95.3 98.0 0.641 −3.8 +0.18

- Table 8: Effect of the ranking weight α in the NearID loss LNearID = Ldisc + α Lrank. All variants use a frozen SigLIP2 backbone with a trainable MAP head, trained on NearID + MTG data. α=0 ablates the ranking term entirely, retaining only the discrimination loss with near-identity distractors in the softmax denominator. Oracle alignment (M–O) generally increases with α up to 1.0, while discrimination (SSR) degrades only mildly until α=1.0 and drops at α=2.0. Human-judgment alignment (M–H) rises sharply at α=0.1 and remains broadly stable (0.54–0.57) for α≥0.25, confirming that the ranking term is essential for perceptually meaningful embeddings.

NearID MTG DB++ SSR ↑ PA ↑ M–O ↑ M–Opair ↑ SSR ↑ PA ↑ M–H ↑

α

α=0 (disc. only) 99.57 99.85 0.306 0.336 63.0 73.0 0.229 α=0.1 99.57 99.84 0.362 0.422 58.0 69.0 0.521 α=0.25 99.20 99.73 0.466 0.518 49.0 62.0 0.555

- α=0.5 (default) 99.17 99.71 0.465 0.486 35.0 46.5 0.545
- α=1.0 98.80 99.61 0.542 0.583 35.0 49.5 0.571
- α=2.0 97.09 98.94 0.494 0.562 26.0 34.5 0.573

###### Table 9: Effect of positive cohesion weight β in the combined objective

LNearID + β Lcoh (Section E.6), evaluated at α=1.0. All variants use a frozen SigLIP2 backbone with a trainable MAP head, trained on NearID + MTG data. β=0 (no cohesion) achieves the best oracle alignment (M–O) and human alignment (M–H DB++); increasing β degrades all alignment metrics without improving discrimination (SSR).

NearID MTG DB++

β (cohesion)

SSR ↑ PA ↑ M–O ↑ M–Opair ↑ SSR ↑ PA ↑ M–H ↑ β=0 (no cohesion, α=1.0) 98.80 99.61 0.542 0.583 35.0 49.5 0.571

- β=0.1 98.94 99.67 0.442 0.478 30.0 40.0 0.555
- β=0.2 98.74 99.59 0.444 0.482 29.0 37.5 0.550
- β=0.3 98.94 99.68 0.430 0.468 29.0 36.5 0.553

###### Table 10: InfoNCE component ablation. Rows extracted from Table 3 with DB++ M–H now reported for all variants. “R neg” adds near-identity distractors to the softmax denominator; “Oracle Ranking” adds a pairwise RankNet term supervised

by oracle similarity. LNearID differs in three ways: symmetric multi-positive softmax, distractors in the denominator, and an unsupervised softplus ranking regulariser (see Table 4). †Representation collapse.

NearID MTG DB++

Variant

SSR ↑ PA ↑ M–O ↑ M–Opair ↑ M–H ↑ ∆M–H None (frozen) 30.74 48.81 0.180 0.366 0.516 InfoNCE (sym., 1-pos) 60.97 75.26 0.267 0.418 0.555 (ref.)

+ R neg 99.57 99.79 0.236 0.267 0.251 −0.304 + Oracle Ranking† 86.34 92.25 0.299† 0.444† 0.167† −0.388 + R neg + Oracle 99.60 99.89 0.247 0.277 0.227 −0.328

###### LNearID (Ours) 99.17 99.71 0.465 0.486 0.545 −0.010

You are a rigorous visual evaluator for INSTANCE identity (same physical object), not category.

You will see TWO images:

- - Image A
- - Image B TASK:

Decide whether A and B show the SAME physical object instance (e.g., the same specific car), ignoring background changes and typical viewpoint/lighting changes.

IMPORTANT:

- - Do NOT use background, scene, or watermark cues as evidence.
- - Do NOT decide based on category-level similarity (e.g., "both are sedans", "both are red").
- - Use only object-instance cues visible on the subject itself. RUBRIC (0–10):

10 = definitely same instance (multiple strong, unique matching cues; no meaningful conflicts) 7–9 = likely same instance (several matching distinctive cues; minor uncertainty) 4–6 = uncertain (insufficient visibility/occlusion/blur; cues are generic) 1–3 = likely different instances (distinctive conflicts: geometry, markings, parts, proportions)

0 = definitely different instances (clear contradictory unique traits) EVIDENCE PRIORITY (use these, in order):

- (1) Unique markings: scratches, dents, decals, stains, wear patterns, rust, custom mods
- (2) Fine geometry/parts: headlight shapes, grille pattern, rim design, handle shape, seams
- (3) Text/logos/serial-like details if clearly readable
- (4) Color/paint only if it includes unique patterns (not plain "red/blue") HANDLE DIFFICULT CASES:

- - If the subject is partially visible, small, blurred, or occluded: score ˜5 and say "insufficient evidence".
- - If differences can plausibly be viewpoint/lighting artifacts, do NOT over-penalize; explain uncertainty. OUTPUT:

Return JSON ONLY, exactly matching this schema: {

"match_cues": ["...","..."], "conflict_cues": ["...","..."], "background_used": false, "confidence": "high" | "medium" | "low", "score": <number from 0 to 10>

} Rules:

- - score must be a NUMBER (not a string).
- - match_cues/conflict_cues must refer to the OBJECT, not the background.
- - background_used must be false unless you truly cannot avoid it; if true, set confidence="low".
- - Do not wrap the JSON in markdown fences.
- - Do not output any text before or after the JSON object.
- - Concise wording for both match_cues and conflict_cues

- Fig. 10: VLM judge prompt template used for the Qwen3-VL 30B baseline. The prompt enforces instance-level (not category-level) identity judgment with an explicit evidence hierarchy and structured JSON output.

