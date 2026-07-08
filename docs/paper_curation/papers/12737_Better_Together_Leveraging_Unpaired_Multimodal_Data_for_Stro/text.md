## Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models

# arXiv:2510.08492v1[cs.LG]9Oct2025

###### Sharut Gupta†, Shobhita Sundaram†, Chenyu Wang†, Stefanie Jegelka† ‡, Phillip Isola† †MIT CSAIL, ‡TU Munich {sharut, shobhita, wangchy, stefje, phillipi}@mit.edu

Unpaired Multimodal Representation Learning

Downstream Task

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Image-only Image w/

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

unpaired text

[Figure 10]

ClassifierLogits

[Figure 11]

[Figure 12]

Image Modeling Objective

[Figure 13]

###### Images

[Figure 14]

[Figure 15]

[Figure 16]

Weight sharing

ViceroyMonarch

[Figure 17]

“Monarch Butterfly has orange wings with black veins”

UnpairedText

Language Modeling Objective

“Red fox with a bushy tail stalks silently at dawn”

[Figure 18]

“Viceroy resembles Monarch but has a line across hindwing”

Figure 1: Text provides complementary information beyond images, even when not paired directly; We introduce Unpaired Multimodal Learner (Uml) whereby sharing model weights across modalities (e.g., image and text) extracts synergies and enhances unimodal representations, outperforming methods that rely only on a single modality (such as images above).

###### Abstract

Traditional multimodal learners find unified representations for tasks like visual question answering, but rely heavily on paired datasets. However, an overlooked yet potentially powerful question is: can one leverage auxiliary unpaired multimodal data to directly enhance representation learning in a target modality? We introduce Uml: Unpaired Multimodal Learner, a modality-agnostic training paradigm in which a single model alternately processes inputs from different modalities while sharing parameters across them. This design exploits the assumption that different modalities are projections of a shared underlying reality, allowing the model to benefit from cross-modal structure without requiring explicit pairs. Theoretically, under linear data-generating assumptions, we show that unpaired auxiliary data can yield representations strictly more informative about the data-generating process than unimodal training. Empirically, we show that using unpaired data from auxiliary modalities—such as text, audio, or images—consistently improves downstream performance across diverse unimodal targets such as image and audio. Our project page: https://unpaired-multimodal.github.io/

### 1 Introduction

It is often taken for granted that to model a modality well, one must train on data from that modality. For instance, if one wants an accurate image classifier, one trains on images; if one wants a language model, one trains on text. Recent advances in multimodal learning, however, suggest that multiple modalities can benefit one another: in particular, using text captions paired with images yields richer representations, often surpassing their unimodal counterparts in zero-shot transfer, cross-modal retrieval, and downstream classification [Radford et al., 2021, Singh et al., 2022, Mizrahi et al., 2023, Girdhar et al., 2023a, Bachmann et al., 2022, Li et al., 2023, Bachmann et al., 2024, Jia et al., 2021]. However, these gains have been realized largely through paired data, where one has access to aligned examples (x, y) ∼ pX,Y, such as an image x and its corresponding caption y. Such paired supervision allows aligning modalities into a shared latent space, where cross-modal correlations can be captured and transferred to downstream tasks.

This reliance on paired corpora also poses a bottleneck. Collecting and curating aligned datasets

is expensive and domain-limited, whereas unpaired data—independent samples from pX and pY—are naturally abundant. For example, vast image collections and vast text corpora exist independently, but without explicit alignment. This raises a more fundamental question:

Can unpaired data from a secondary modality Y enhance representations of the target modality X, even in the absence of (x, y) correspondences?

There is growing evidence that the answer may be yes. Recent work posits the existence of a shared statistical model of reality, an empirical parallel to Plato’s concept of ideal Forms, suggesting that as deep networks scale, their embeddings across modalities converge toward a unified representation of the underlying world [Huh et al., 2024, Huang et al., 2021]. This convergence implies that when paired supervision is available, a model can leverage natural co-occurrences between modalities and thus more accurately capture shared semantics. Critically, however, achieving convergence does not necessarily require explicit pairs; as long as each modality samples from the same underlying ground-truth latent space, it may be sufficient to align their marginal distributions to uncover the common semantic structure [Timilsina et al., 2024, Sturma et al., 2023]. Intuitively, even unpaired data from another modality can provide complementary cues to better estimate the underlying reality (Figure 1).

In this work, we formalize this idea through Unpaired Multimodal Representation Learning, a framework for improving unimodal representations by leveraging unpaired data across modalities. Theoretically, under linear assumptions, we derive conditions under which incorporating unpaired samples yields strictly more informative representations than unimodal training alone. Notably, in some regimes, a single sample from Y yields a greater per-sample value than a sample from X itself when the goal is to model X. Building on these insights, we introduce Uml: Unpaired Multimodal Learner, a shared-network framework that applies the same set of parameters to inputs from different modalities. By nothing more than weight sharing, i.e., without surrogate objectives or inferred alignments, the model learns modality-agnostic features and naturally transfers information across modalities [Sutskever, 2023]. Empirically, we evaluate Uml on diverse image–text tasks in healthcare, and affective computing, as well as 10 standard visual benchmarks. Unpaired data from auxiliary modalities consistently improves unimodal representations across self-supervised and supervised regimes, in both few-shot and full-data regimes, and with diverse encoders including CLIP, DINOv2, OpenLLaMA, and others. These benefits compound when moving from two to three modalities, with audio, vision, and text each adding complementary signals. We further demonstrate effective cross-modal transfer without paired data by initializing vision models with pretrained language-model weights. Finally, we quantify the exchange rate between modalities, mapping how many words equate to one image (and vice versa) for optimal performance.

To summarize, the key contributions of our work are:

- • We introduce Uml, a modality-agnostic framework that leverages unpaired data to improve unimodal models. We test it across self-supervised and supervised encoders (CLIP, DINOv2, OpenLLaMA, and others), in both few-shot and full-data regimes, showing consistent gains over a range of image-text benchmarks and extensions to audio.
- • We theoretically characterize, under linear assumptions, the conditions where unpaired data yield strictly more informative representations than unimodal training. Remarkably, the conversion ratio between modalities can fall below one: in certain regimes, a single sample from Y contributes more to modeling X than an additional sample from X itself.
- • We quantify conversion ratios between images and text; i.e., how many words is an image worth for training vision models and show that unpaired multimodal data systematically widens inter-class margins and aligns modalities in weights. We further demonstrate that individual neurons develop strong cross-modal coupling, revealing multimodal neurons that respond coherently across vision and text, all without any paired supervision.

### 2 Paired Multimodal Representation Learning

A useful way to conceptualize learning across modalities is to posit a shared ground-truth reality, denoted Z∗, which manifests through multiple projections such as images, text, or audio recordings [Huh et al.,

- 2024, Timilsina et al., 2024, Sturma et al., 2023]. The goal of representation learning is to learn an embedding space that captures the structure of Z∗, whether from a single modality or from several jointly. Thus, unimodal representations are inherently limited by what one projection alone can reveal: a single camera view may contain occlusions; an audio recording lacks visual details; textual descriptions may lack layout information.

These limitations motivate multimodal representation learning, which extends the unimodal setting to learn from multiple modalities together. By integrating heterogeneous projections, multimodal learning leverages their interplay to recover a more complete view of the shared latent reality. For brevity, consider two modalities with observable data denoted by X (e.g., images) and Y (e.g., text) and samples x ∈ X and y ∈ Y. Mathematically, Multimodal representation learning seeks encoders fX : X → Z and fY : Y → Z mapping each modality into a shared embedding space Z. Different frameworks optimize for desirable properties of Z. Contrastive methods encourage instances of the same concept (such as an image and its text label) to be close together [Radford et al., 2021, Zhai et al., 2023, Jia et al., 2021]. Fusion approaches learn shared layers for multimodal inputs with reconstruction [Singh et al., 2022, Li et al., 2019, Lu et al., 2019, Bachmann et al., 2022] or contrastive objectives [Roy et al., 2025]. Generative methods often structure Z to enable translation between modalities[Li et al., 2022, Rombach et al., 2022]. Other methods disentangle common factors from modality-specific factors [Wang et al., 2024, Liang et al., 2023].

Common to all of these prior work, however, is a requirement for pre-existing knowledge of one-to-one correspondences between the modalities, i.e., samples (x, y) ∼ PX,Y (as shown in Figure 2(a)), which are assumed to be aligned views of the same underlying entity in Z∗. We ask whether it is possible to recover this shared structure without such correspondences—namely, when only unpaired samples from the marginals PX and PY are available, as shown in Figure 2(c)—and, if so, how this affects downstream performance individually on the primary modality X.

### 3 Unpaired Multimodal Representation Learning

Let DX = {xi}iN=x1 and DY = {yj}Nj=y1 be datasets of Nx and Ny samples respectively, drawn from marginal distributions PX and PY as shown in Figure 2(c). The joint distribution PX,Y is unknown. Critically, in

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(a) Paired Data

(b) Unpaired Data (w/ labels)

(c) Unpaired Data

[Figure 29]

[Figure 30]

[Figure 31]

“A guinea pig rests on grass”

“A guinea pig rests on grass”

,“Alooksredatpandathe camera”

Guinea Pig

Red panda

[Figure 32]

[Figure 33]

[Figure 34]

“A rabbit is near the flowers”

“A rabbit is near the flowers”

“Two dogs play happily”

,

,

,

Dog

Rabbit

[Figure 35]

[Figure 36]

“An elephant is a large mammal”

“An elephant is a large mammal”

[Figure 37]

“A guinea pig rests on grass”

,

Guinea Pig Elephant

.. ..

..

.. ..

Figure 2: (a) Paired learning uses (xi, yi) with known correspondences. We instead study Unpaired learning: (b) with labels, using (xi, ci) and (yj, cˆj), where ci and cˆj denote labels for xi and yj, but no cross-modal correspondences; and (c) without any labels or correspondences, using {xi} and {yj}.

contrast to paired multimodal learning, for any given xi ∈ DX and yj ∈ DY, there is no assumption that xi and yi are projections of the same underlying entity z ∈ Z. This is in contrast with paired multimodal learning, which assumes access to labeled one-to-one correspondences between DX and DY.

Our objective is to learn mappings fX : X → Z and fY : Y → Z from each modality to a common embedding space that captures the shared structures between the unpaired datasets. We refer to this as Unpaired Multimodal Representation Learning. In contrast to previous works [Lin et al., 2023, Lee and Yoon,

- 2025, Girdhar et al., 2022, Chada et al., 2023] (discussed in Section A), we leverage unpaired data without inferring alignment, incorporating paired data, or assuming pre-aligned embeddings. Intuitively, without

known DX-to- DY correspondences, even unpaired data can be useful if modalities capture different axes of information pertaining to the underlying Z∗. We formalize this intuition in Section 3.1, within the popularly studied framework of linear models, and introduce a modality-agnostic algorithm Uml in Section 3.2 for learning representations from unpaired data.

###### 3.1 Theoretical Perspectives

We adopt the widely used modeling choice that each modality can be expressed as a linear combination of shared and modality-specific components [Sturma et al., 2023, Timilsina et al., 2024, Huang et al., 2021]. This formulation inherently accounts for informational differences across modalities by incorporating modality-specific features [Wang et al., 2024]. Prior works [Sturma et al., 2023, Timilsina et al., 2024] have developed sufficient conditions for recovering joint distributions and the shared causal structure under this linear model. Building on this framework, we ask how joint training with unpaired modalities can still enhance representations within a single modality.

Data Generating Process. Assume that all factors of variation in reality live in a single d-dimensional space, Z∗, i.e., θ ∈ Rd, modeled using a linear data-generating pipeline. This parameter can further be decomposed as θ ≡ [θc, θx, θy]⊤ where θc ∈ Rdc, θx ∈ Rdx, θy ∈ Rdy and dc + dx + dy = d. Here, θc captures the common (shared) parameters that affect both modalities, θx denotes the parameters that only affect modality X, and θy denotes the parameters that only affect modality Y. We observe two datasets, one from each modality {Xi}iN=x1 ∈ Rm and {Yj}Nj=y1 ∈ Rn, each reflecting partial measurements of the ground

truth latent space Z∗:

Xi = Ac,i θc + Ax,i θx + ϵX,i, ϵX,i ∼ N 0, σx2Im (1) Yj = Bc,j θc + By,j θy + ϵY,j, ϵY,j ∼ N 0, σy2In . (2)

Here, Ac,i, Ax,i, Bc,j, By,j are known design blocks capturing how each sample probes the latent factors and εX,i, εY,j represent the independent measurement noise.

In our linear setting, estimating the true latent state θ—and hence the underlying reality Z∗—is

governed by the Fisher information matrix I(θ) = −E ∇2θ ℓ(θ) [Fisher, 1922, 1925], which measures how sharply the likelihood “curves” around the true θ. High curvature along a particular axis means the data tightly constrain that component, driving down estimator variance there. Because X and Y samples are independent given Z∗, their curvature contributions add pointwise, resulting in the joint Fisher information being simply the sum of the unimodal blocks. Thus, loosely speaking, any nonzero contribution from the unpaired Y-samples strictly increases curvature and thus strictly tightens the variance bound along those directions in θc. We formalize these insights in Theorem 1 and Theorem 2. Our results also generalize naturally to more than two modalities since the total contribution of all auxiliary modalities (excluding the primary modality X) can be obtained by summing their individual Fisher information matrices. Complete proofs and additional background are provided in Section C.

- Theorem 1 (Variance Reduction with Unpaired Multimodal Data). Let θˆX, θˆY be the least-squares estimators for θ using only {Xi} and only {Yj} and let θˆX,Y be the joint estimator using both unpaired datasets. Then, under the assumption that at least one Bc,j, where j ∈ {1, 2, ...Ny}, has full rank, the common-factor covariance satisfies the

strict Loewner ordering i.e. Var θ ˆX,Y

θc,θc

≺ Var θ ˆX

θc,θc

, or equivalently, the Fisher information on θc strictly increases when combining both modalities, despite not having sample-wise pairing: (IX + IY)θc,θc ≻ (IX)θc,θc.

Theorem 1 says that although there is no sample-wise pairing, simply adding an unpaired modality Y, which carries non-degenerate information about every direction in the shared subspace, can only tighten our uncertainty about the shared parameters θc. Geometrically, the uncertainty ellipsoid for θc shrinks in every direction where Y contributes any curvature, and never expands elsewhere. In practice, however, no single data sample covers every latent axis. In these settings, while global shrinkage (Theorem 1) no longer applies, we still get directional gains. We capture this in Theorem 2.

- Theorem 2 (Directional Variance Reduction with Unpaired Multimodal Data). Let all notation be as in Theorem 1, and let v ∈ Rdc \ {0}. If there exists at least one index j ∈ {1,2,...Ny} such that Bc,jv ̸= 0, then

v⊤ (IX + IY)θc,θc v > v⊤ (IX)θc,θc v. Further, the variance of the estimator in direction v is strictly reduced if v ∈/ range((IX)θc,θc).

Building on Theorem 2’s insights that any nonzero curvature in a given direction yields strict variance reduction, Corollary 1 and Corollary 2 in the Appendix (i) quantify the precise contraction factor when both modalities inform the same direction, and (ii) demonstrate how an otherwise ill-posed direction becomes well-posed once the second modality contributes.

Thus far, we have studied incorporating data from an auxiliary modality without considering sample size constraints. A natural next question is how to allocate a fixed budget of N additional samples: is it better to collect them from the same modality X, or from complementary modality Y? We formally address this in Theorem 3 below.

- Theorem 3 (Data from Auxiliary Modality Can Outperform More of the Same). Define for any m, IX(m) = ∑im=1 A⊤c,iAc,i and IY(m) = ∑mj=1 Bc⊤,jBc,j. If range IY(m) ̸⊆ range IX(m) , then there exists a nonzero v ∈ Rdc such that v⊤IY(m) v > v⊤IX(m)v.

Theorem 3 shows that if the second modality covers a “blind spot” of the first, adding additional samples from the first modality does not increase the Fisher information in that direction; however, unpaired samples from the second modality provide strictly positive information along that axis.

To empirically validate the above result, we design an illustrative Gaussian experiment (Section E.14) in which both modalities are generated as noisy linear projections of the same underlying Gaussian latent variable θc. We allocate a fixed budget of samples, splitting evenly between X and Y (details in Section E.14). Joint training with Uml significantly improves reconstruction fidelity on

3.5

ReconstructionErroronX

Unpaired (X + Y)-data

X-data only

3.0

2.5

2.0

1.5

- X (Figure 3) compared to training on X alone. Strikingly, this also reveals that the effective exchange rate between modalities is below one, meaning a single additional sample from Y is worth more than an additional sample from X, even when the test task is on X. In Section 4.4, we extend this idea by quantifying the exchange rate between images and text on real world benchmarks.

1.0

0 200 400 Training Steps

Figure 3: Adding unpaired Y samples boosts X reconstruction more than adding extra X samples.

###### 3.2 UML: Unpaired Multimodal Learner

Tokenization and Embed Unpaired Multimodal Learner (UML)

Shared Weights Information Flow

2 3 4 2’ 3’ 4’

- 1

- 2

- 3

Patch embd.

| | |
|---|---|
| | |

[Figure 38]

###### gX

gY

Image Encoder

“Butterfly” “eagle”

z

/

CLS/mean embd.

##### h

##### h h h

- 1’

- 2’

- 3’

Token embds.

| | |
|---|---|
| | |

fY fX fY

“A golden eagle”

fX

Text Encoder

z′ 1 2 3 1’ 2’ 3’

z z′

CLS/mean embd.

(a) Self-Supervised Setting (b) Supervised Setting

- Figure 4: (Left) Inputs from different modalities (e.g., images or text) are tokenized into patch or token embeddings using pretrained encoders or processed features; (Right) Uml can be trained under two settings: (a) Self-supervised, where patch/token embeddings are passed through a shared network and modality-specific decoders to perform next-token/patch prediction; (b) Supervised, where mean/CLS embeddings are fed through the shared classifier to predict labels within each modality.

The theory developed above establishes that adding an unpaired auxiliary modality strictly increases Fisher information along shared directions, thereby reducing estimator variance. We now translate these insights into a practical framework: Uml (Unpaired Multimodal Learner). The central idea is to share parameters across modalities: since both X and Y are projections of the same underlying reality Z∗, forcing them through shared weights can extract synergies by accumulating training gradients on the same parameters. This accumulation could be viewed as the practical analogue of Fisher information addition, ensuring that even without paired samples, auxiliary modalities tighten estimates of shared structure. While prior work has explored shared weights in partially paired settings [Sturma et al., 2023], for related visual streams such as video and depth [Girdhar et al., 2022], or within already-aligned spaces like CLIP [Lin et al., 2023], we study and show their pronounced impact in the most general case of unpaired modalities with unaligned encoders.

Consider unpaired samples or preprocessed features x ∼ PX and y ∼ PY, each encoded into embeddings zX = fX(x) and zY = fY(y) as shown in Figure 4. These embeddings are passed through a shared network h, producing rX = h(zX) and rY = h(zY). This shared h is the sole coupling between modalities and the locus of cross-modal transfer. We consider two regimes to train these networks: (a) self-supervised learning for fully unpaired data (Figure 2(c)) and (b) supervised learning for unpaired data with labels as shown in Figure 2(b). In the self-supervised setting, each modality has its own decoder (gX and gY) trained to reconstruct the input or predict the next token/patch depending on the form of embeddings zX and zY (refer to Figure 4(a)). The training objective (Pseudocode in the appendix: Algorithm 1) is

LUML-SSL = Ex∼PX ℓ gX(rX), x + Ey∼PY ℓ gY(rY), y ,

where ℓ is the mean squared error for continuous targets or cross-entropy for discrete tokens. In the supervised setting, when labels are available, cX for x and cY for y, a shared classifier c(·) is trained on top of rX and rY. Since both c(·) and h(·) are shared across modalities, they can be viewed as a single shared head h (refer to Figure 4(b)). The supervised loss (Pseudocode in the appendix: Algorithm 2) is

LUML-Sup = E(x,cX) ℓCE c(rX), cX + E(y,cY) ℓCE c(rY), cY .

In both scenarios, although supervision is modality-specific, the shared head h receives updates from both modalities. Consequently, gradients from h also flow into fX, effectively transferring information from fY and thus Y even without paired samples. At inference, we discard the auxiliary pathway and use rX as the representation for the target modality, training a linear probe on top for downstream tasks. Uml also naturally extends to more than two modalities by incorporating additional modality-specific encoder/decoder heads or classification layers.

### 4 Experimental Results

In this section, we empirically evaluate Uml across diverse benchmarks and configurations, leading to three main takeaways: (i) auxiliary modalities consistently boost target image representations in both self-supervised (Section 4.1) and supervised regimes (Section 4.2), with particularly strong gains in fine-grained and low-shot tasks; (ii) the benefits compound as we move from two to three modalities, with audio, vision, and text each contributing complementary signals (Section 4.2); and (iii) weight sharing across modalities generalizes beyond co-training, as pretrained language model weights transfer effectively to vision (Section 4.3). Finally, we quantify a marginal rate of substitution between modalities, asking how many text samples equate to an image (Section 4.4) and report the emergence of multimodal neurons that implicitly align modalities (such as vision and language), even under unpaired supervision (Section 4.5).

###### 4.1 UML in Self-Supervised Setting

To study Uml in the self-supervised setting, we use the real-world multimodal benchmark from MultiBench [Liang et al., 2021], which includes curated datasets such as text and images, with downstream labels covering a variety of domains, including healthcare, affective computing, and multimedia research and three standard classification benchmarks [Parkhi et al., 2012, Cimpoi et al., 2014, Soomro et al., 2012]. We follow the same setting (dataset splitting, encoder architecture, pre-extracted features) as [Liang et al., 2023, 2021]. For training (refer to Figure 4(a)), we use linear encoders (fX and fY) and decoders (gX and gY) and use an autoregressive transformer as the shared network (h). At inference, we average the transformer output embeddings and use the resulting embedding for linear probing on the downstream classification tasks. We report test accuracy using embeddings from the primary modality (image), and to ensure fair comparisons, we perform rigorous hyperparameter tuning for all methods and repeat each experiment with three random seeds. For more details, refer to Section B.3.1 and Section B.4.1. As shown in Table 1,

- Table 1: Uml (Ours) achieves higher top-1 linear probe accuracy than unimodal baselines on learned representations, averaged over three seeds, across both MultiBench and vision–text benchmarks.

Dataset MultiBench Standard vision-text

OxfordPets

Ur-funny

Mustard

UCF101

Mimic

Mosei

Mosi

DTD

Method

Unimodal 59.66 55.16 70.62 56.17 56.99 85.04 79.86 78.13 Ours 63.28 ↑ 57.10 ↑ 71.98 ↑ 58.16 ↑ 57.34 ↑ 86.32 ↑ 80.98 ↑ 78.49 ↑

Uml significantly outperforms its unimodal counterpart across all benchmarks, particularly on Mustard where unique information from the text modality expresses sarcasm, such as ironic tone of voice.

###### 4.2 UML in Supervised setting

We evaluate Uml on 9 widely-used visual classification benchmarks [Fei-Fei et al., 2004, Parkhi et al., 2012, Krause et al., 2013, Nilsback and Zisserman, 2008, Bossard et al., 2014, Xiao et al., 2010, Cimpoi et al., 2014, Soomro et al., 2012] in three settings: (1) Full fine-tuning: initializing from a pretrained vision backbone and updating all parameters on the target dataset. (2) Few-shot linear probing: freezing the vision backbone and training a linear classifier on k labeled samples per class (k = 1,2,4,8,16). (3) Full-dataset linear probing: freezing the vision backbone and training a linear classifier on the entire training dataset, discussed in Section E.1.2. In all cases, we enrich image representations with unpaired text embeddings, using ViT-S/14 DINOv2 as the vision encoder and OpenLLaMA-3B as the frozen text encoder. To construct conceptually related yet unpaired text data, we generate text templates with varying amounts of semantic information about the dataset. For further details and specific prompts, refer to Section B.3. To train Uml, we initialize the classifier with the average text embedding of each class, giving a strong prior for image–class alignment (refer to Section E.1 for ablation).

Unpaired Textual Data Improves Visual Classification. As shown in Table 2, across both full fine-tuning and few-shot linear probing, Uml consistently improves over unimodal baselines across all datasets, with the largest gains on fine-grained tasks (e.g., Stanford Cars, FGVC Aircraft) where unpaired text sharpens class boundaries, and in low-shot regimes where textual cues help disambiguate visually similar classes. Additional results on other shots, datasets, model scales, and prompt variants are reported in Section E.

We also evaluate the robustness of Uml-trained models under test-time distribution shifts. A 16-shot linear probe with DINOv2 is trained on ImageNet and tested across four distribution-shifted target datasets: ImageNet-V2, ImageNet-Sketch, ImageNet-A, and ImageNet-R. Uml consistently outperforms the unimodal baseline (Figure 5), showing that language priors yield more transferable features. Additional robustness results are provided in Section E.3.

Finally, for all these settings, we also replace the independent vision (DINOv2) and text encoders (OpenLLaMA) with those of CLIP; since CLIP embeddings are already aligned, the gains from Uml are even stronger (Section E.2.1, Section E.2.2, Section E.2.3,).

Additional Ablations. For all experiments, we keep the text encoder frozen; Unfreezing the text encoder yields slightly weaker gains but still outperforms the unimodal baseline (Section E.12). Swapping in different text encoders yields consistent gains (Section E.6), while richer, more diverse captions provide especially strong boosts in few-shot settings (Section E.7). Further, training with semantically unrelated modalities (Section E.9) does not improve over the unimodal counterpart, confirming that gains stem

- Table 2: Uml (Ours) outperforms unimodal baseline on image classification with ViT-S/14 DINOv2 and OpenLLaMA-3B in both settings: (i) Full finetuning and (ii) Few-shot linear probing (k = 1,2,4). For complete results, refer to Section E.1.

Dataset

OxfordFlowers

FGVCAircraft

StanfordCars

OxfordPets

Caltech101

Food101

SUN397

Average

UCF101

DTD

Shot Method

Full-finetuning Unimodal 79.45 66.20 66.99 72.16 83.18 80.65 90.67 99.18 95.45 81.54 Ours 86.39 ↑ 66.03 ↓ 73.44 ↑ 74.27 ↑ 84.69 ↑ 81.97 ↑ 91.72 ↑ 99.82 ↑ 97.60 ↑ 83.99 ↑

Few-shot Linear Probing

- 1

Unimodal 13.18 34.15 14.09 36.60 46.74 35.18 63.51 89.62 76.66 45.52 Ours 16.49 ↑ 41.79 ↑ 15.63 ↑ 42.04 ↑ 52.33 ↑ 42.27 ↑ 73.59 ↑ 93.64 ↑ 84.52 ↑ 51.36 ↑

- 2

Unimodal 24.68 47.88 23.09 47.75 56.81 48.54 75.32 96.02 86.90 56.33 Ours 28.65 ↑ 53.15 ↑ 24.78 ↑ 53.25 ↑ 63.86 ↑ 54.44 ↑ 81.41 ↑ 97.63 ↑ 90.55 ↑ 60.85 ↑

Unimodal 38.76 57.51 32.10 59.69 67.75 60.79 83.89 98.59 93.48 65.84 Ours 43.17 ↑ 60.89 ↑ 33.86 ↑ 62.43 ↑ 71.13 ↑ 63.88 ↑ 87.36 ↑ 99.17 ↑ 94.96 ↑ 68.53 ↑

- 4

###### Distribution Shift Results

from semantic correlations rather than spurious ones (Section E.9). Finally, unpaired multimodal data systematically widens inter-class margins and aligns modalities in weights (Section F).

- Figure 5: Our approach Uml is much more robust than its unimodal counterpart across four test time distribution shifted target test sets. All results are averaged across three random seeds.

Unpaired Image and Textual Data Improve Audio classification. We extend Uml to an audio–vision–text setting using the ImageNet-ESC benchmark [Lin et al., 2023], which links ImageNet objects and captions with ESC-50 environmental sounds. The benchmark has two versions: ImageNet-ESC-27 and ImageNetESC-19. For example, the “barking” ESC-50 class maps to ImageNet dog breeds. The benchmark has two versions: ImageNet-ESC-27 (loosely matched visual-audio pairs) and ImageNet-ESC-19 (only accurate visual-audio matches). For audio encoding, we use AudioCLIP with an ES-ResNeXT backbone [Guzhov

- et al., 2021], while images and text are encoded by DINOv2 and OpenLLaMA-3B encoders. For further details, refer to Section B.3.

Unpaired image and text data consistently improve audio classification (Figure 6), with larger gains from CLIP’s aligned encoders (Section E.13.3). Conversely, both audio and text also enhance image classification, showing that transfer works in both directions (Section E.13).

Finally, we study the full three-modality case, where we treat two modalities as auxiliaries and one as the target—for example, using both image and text as auxiliaries for audio classification. As shown in Section E.10, performance improves monotonically with each added auxiliary modality, with the

###### Audio Benchmarks

(a) ImageNet-ESC-19 (b) ImageNet-ESC-27

- Figure 6: Uml (Ours) improves audio classification using unpaired image and text samples on ImageNet-ESC-19 and ImageNet-ESC-27. All results are averaged across three random seeds.

strongest gains achieved when all three modalities (audio, vision, and text) are used together.

###### 4.3 Transfer Learning

- 55

- 56

- 57

- 58

- 59

- 60

Thus far, we have explored how sharing model weights while co-training with multiple unpaired modalities improves the learned representation. But weight sharing need not be restricted to cotraining: if modalities capture the same underlying latents, then pretrained weights from one should also serve as a useful initialization for another. We therefore study if transferring knowledge from one modality can enhance performance in another by initializing the transformer layers of a ViT [Dosovitskiy et al., 2020] with pretrained BERT [Devlin et al., 2019] weights and evaluating on ImageNet (details in Section B.4.4). Patch embeddings are learned from scratch through a linear layer, augmented by a CLS token and positional embeddings. As shown in Figure 7, initializing with BERT boosts performance for both frozen and unfrozen backbones. Our results indicate that the semantic knowledge of language models provides a strong initialization for vision compared to training from scratch.

20.0

+42.7%

TestAccuracy(%)

TestAccuracy(%)

17.5

+1.4%

15.0

12.5

10.0

Scratch from BERT

Scratch from BERT

Figure 7: Image classifier trained from BERT initialization outperforms training from scratch for (left) full fine-tuning; (right) frozen backbone

###### 4.4 Marginal Rate-of-Substitution Between Modalities

Having established that unpaired modalities boost representation learning, robustness, and transfer, we now ask a more fundamental question: what is the relative value of each modality? If both images and text provide views of the same semantic space, can we quantify their exchange rate—How many words is an image worth? Figure 8 and Figure 9 visualize image-text conversion ratios using test accuracy isolines on Oxford-Pets [Parkhi et al., 2012] dataset. These map the number of texts equivalent to an image for the same performance. Aligned CLIP encoder (1 image ≈ 228 words) is more efficient than non-aligned DINOv2 and OpenLLaMA encoders (1 image ≈ 1034 words). Indeed, in some cases, an image may quite literally be worth a thousand words. We also observe little or no additional benefit from adding more text beyond a few samples. This is likely because we do not control for increasing complexity, so adding sentences does not guarantee extra information. Further results on additional datasets and key details, refer to Section E.4.

[Figure 39]

[Figure 40]

Figure 8: 1 img ≈ 228 words (CLIP) Figure 9: 1 img ≈ 1034 words (DINOv2)

###### 4.5 Existence of Multimodal Neurons

Neuron #90 Neuron #50 Neuron #144 Neuron #46 Neuron #140 Neuron #124 Neuron #16

Figure 10: Existence of Multimodal Neurons without Paired Supervision. Bars show overall (green), correlations conditioned on label=1 (sarcastic; blue), and correlations conditioned on label=0 (nonsarcastic; pink) between visual and textual activations for a subset of neurons. Most neurons exhibit strong cross-modal correlation and specifically higher alignment for non-sarcastic samples, where visual and verbal cues are naturally congruent.

While the previous section quantified the exchange rate between modalities, our next question concerns the mechanism that enables such exchange. Models like CLIP, trained with paired image–text supervision, are known to develop multimodal neurons [Goh et al., 2021]: units that respond coherently to the same concept across both modalities. We ask whether similar cross-modal units can emerge even without such paired supervision, when the model is exposed only to unpaired but semantically related data.

Neuron #90

To test this hypothesis, we analyze neuron-level activations on the Mustard [Castro et al., 2019] dataset, a multimodal sarcasm detection benchmark containing 690 video clips paired with corresponding utterances from television shows, each labeled for the presence or absence of sarcasm. For each sam-

Cross-ModalCorrelation

0.6

0.4

0.2

ple i and neuron j, let zij(v) and zij(t) denote activations in the visual and textual pathways, respectively. We compute the Pear-

Overall

Sarcastic

Non Sarcastic

0.0

son correlation rj = corr({zij(v)}i, {zij(t)}i) to quantify cross-modal alignment. To examine label-specific effects, we further calcu-

0 5 10 15 20 25 Training epochs

late r(jy) = corr({zij(v) : yi = y}, {zij(t) : yi = y}), for each class y ∈ {sarcastic,non-sarcastic}, assigning zero to undefined cases.

Figure 11: Uml increasingly infers cross-modal correspondences with training, without paired supervision.

- As shown in Figure 10, even without paired supervision, several

neurons exhibit strong cross-modal coupling between vision and text, significantly higher than the highest correlation across neurons for an untrained network (baseline). This coupling steadily improves with training epochs, indicating that the model progressively infers more correspondences between modalities all without any paired supervision (refer to Figure 11). This correspondence, however, varies across labels. Most neurons show higher correlations for non-sarcastic samples, consistent with natural communication where facial expressions and verbal cues typically align. Sarcastic utterances, by contrast, deliberately break this alignment; words and expressions convey opposing meanings, resulting in weaker correlations between them. For detailed results across multiple neurons, refer to Section F.1.

### 5 Conclusions and Limitations

Conclusions. We propose and investigate Unpaired Multimodal Representation Learning for enhancing unimodal representations with unpaired multimodal data. Under linear assumptions, we theoretically show that unpaired data from multiple modalities strictly increases Fisher information along shared directions, resulting in a more accurate representation of the underlying world. Mechanistically, Unpaired Multimodal Learner (Uml) achieves this by accumulating gradients from different modalities on shared weights, which can be viewed as an operational analogue of the Fisher information gain. Empirically, we show performance gains across vision, text and audio benchmarks, and estimate conversion ratios between modalities. Uml provides a new perspective on how to harness the abundance of unpaired data to learn better representations. This may be especially useful in domains such as medical imaging, scientific data, and robotics, which contain rich auxiliary modalities like text, audio, or metadata that are not often paired with every instance of the primary modality.

Limitations. While our experiments study learning from unpaired multimodal data under both the selfsupervised and supervised settings, downstream evaluations are conducted primarily for classification. Investigating evaluation tasks, such as generation, offers a rich avenue for future work. Furthermore, we evaluate how multimodal data enhances image and audio classification; it remains to show if they can, in turn, offer useful information for textual tasks.

### 6 Reproducibility Statement

We provide complete proofs and relevant background for all theoretical results in Section C.3. For experiments, we detail the setup, training protocols, and algorithm implementations in Section B.4 and Section D. All datasets are publicly available, with additional details in Section B.3. The code can be found at https://github.com/Sharut/Unpaired-Multimodal-Learning/.

### 7 Acknowledgments

This research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator under Cooperative Agreement Number FA8750-19-2-1000, in part by the NSF AI Institute TILOS (NSF CCF2112665) and the Alexander von Humboldt Foundation. This work was also supported by a Packard Fellowship to P.I., and by ONR MURI grant N00014-22-1-2740. S.G. is supported by the MathWorks Engineering Fellowship. S.S. is supported by an NSF GRFP fellowship. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes, notwithstanding any copyright notation herein.

### References

- A. Almahairi, S. Rajeshwar, A. Sordoni, P. Bachman, and A. Courville. Augmented cyclegan: Learning many-to-many mappings from unpaired data. In International conference on machine learning, pages 195–204. PMLR, 2018.

R. Bachmann, D. Mizrahi, A. Atanov, and A. Zamir. Multimae: Multi-modal multi-task masked autoencoders. In European Conference on Computer Vision, pages 348–367. Springer, 2022.

- R. Bachmann, O. F. Kar, D. Mizrahi, A. Garjani, M. Gao, D. Griffiths, J. Hu, A. Dehghan, and A. Zamir. 4m-21: An any-to-any vision model for tens of tasks and modalities. Advances in Neural Information Processing Systems, 37:61872–61911, 2024.

L. Bossard, M. Guillaumin, and L. Van Gool. Food-101–mining discriminative components with random forests. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part VI 13, pages 446–461. Springer, 2014.

- S. Castro, D. Hazarika, V. P´erez-Rosas, R. Zimmermann, R. Mihalcea, and S. Poria. Towards multimodal sarcasm detection (an obviously perfect paper). arXiv preprint arXiv:1906.01815, 2019.

- R. Chada, Z. Zheng, and P. Natarajan. Momo: A shared encoder model for text, image and multi-modal representations. arXiv preprint arXiv:2304.05523, 2023.

M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3606–3613, 2014.

P. Demetci, R. Santorella, B. Sandstede, W. S. Noble, and R. Singh. Scot: Single-cell multi-omics alignment with optimal transport. Journal of Computational Biology, 29(1):3–18, 2022. doi: 10.1089/cmb.2021.0446. URL https://doi.org/10.1089/cmb.2021.0446. PMID: 35050714.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In 2004 conference on computer vision and pattern recognition workshop, pages 178–178. IEEE, 2004.

- R. A. Fisher. On the mathematical foundations of theoretical statistics. Philosophical transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character, 222(594-604):309–368, 1922.

- R. A. Fisher. Theory of statistical estimation. In Mathematical proceedings of the Cambridge philosophical society, volume 22, pages 700–725. Cambridge University Press, 1925.

P. Gao, S. Geng, R. Zhang, T. Ma, R. Fang, Y. Zhang, H. Li, and Y. Qiao. Clip-adapter: Better visionlanguage models with feature adapters. International Journal of Computer Vision, 132(2):581–595, 2024.

- X. Geng, H. Liu, L. Lee, D. Schuurmans, S. Levine, and P. Abbeel. Multimodal masked autoencoders learn transferable representations. arXiv preprint arXiv:2205.14204, 2022.

- R. Girdhar, M. Singh, N. Ravi, L. Van Der Maaten, A. Joulin, and I. Misra. Omnivore: A single model for many visual modalities. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16102–16112, 2022.

- R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15180–15190, 2023a.

- R. Girdhar, A. El-Nouby, M. Singh, K. V. Alwala, A. Joulin, and I. Misra. Omnimae: Single model masked pretraining on images and videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10406–10417, 2023b.

G. Goh, N. C. †, C. V. †, S. Carter, M. Petrov, L. Schubert, A. Radford, and C. Olah. Multimodal neurons in artificial neural networks. Distill, 2021. doi: 10.23915/distill.00030. https://distill.pub/2021/multimodalneurons.

- A. Guzhov, F. Raue, J. Hees, and A. Dengel. Esresne (x) t-fbsp: Learning robust time-frequency transformation of audio. In 2021 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2021.

M. K. Hasan, W. Rahman, A. Zadeh, J. Zhong, M. I. Tanveer, L.-P. Morency, et al. Ur-funny: A multimodal language dataset for understanding humor. arXiv preprint arXiv:1904.06618, 2019.

- Y. Huang, C. Du, Z. Xue, X. Chen, H. Zhao, and L. Huang. What makes multi-modal learning better than single (provably). Advances in Neural Information Processing Systems, 34:10944–10956, 2021.

M. Huh, B. Cheung, T. Wang, and P. Isola. The platonic representation hypothesis. arXiv preprint arXiv:2405.07987, 2024.

C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021.

- A. E. Johnson, T. J. Pollard, L. Shen, L.-w. H. Lehman, M. Feng, M. Ghassemi, B. Moody, P. Szolovits, L. Anthony Celi, and R. G. Mark. Mimic-iii, a freely accessible critical care database. Scientific data, 3(1): 1–9, 2016.

C. Jose, T. Moutakanni, D. Kang, F. Baldassarre, T. Darcet, H. Xu, D. Li, M. Szafraniec, M. Ramamonjisoa, M. Oquab, et al. Dinov2 meets text: A unified framework for image-and pixel-level vision-language alignment. arXiv preprint arXiv:2412.16334, 2024.

J. Y. Koh, R. Salakhutdinov, and D. Fried. Grounding language models to images for multimodal inputs and outputs. In International Conference on Machine Learning, pages 17283–17300. PMLR, 2023.

J. Krause, M. Stark, J. Deng, and L. Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pages 554–561, 2013.

- G. Lample, M. Ott, A. Conneau, L. Denoyer, and M. Ranzato. Phrase-based & neural unsupervised machine translation. arXiv preprint arXiv:1804.07755, 2018.

J.-J. Lee and S. W. Yoon. Can one modality model synergize training of other modality models? In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=5BXWhVbHAK.

J. Li, D. Li, C. Xiong, and S. Hoi. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.

L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019.

- Y. Li, H. Fan, R. Hu, C. Feichtenhofer, and K. He. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23390–23400, 2023.

- P. P. Liang, Y. Lyu, X. Fan, Z. Wu, Y. Cheng, J. Wu, L. Chen, P. Wu, M. A. Lee, Y. Zhu, et al. Multibench: Multiscale benchmarks for multimodal representation learning. Advances in neural information processing systems, 2021(DB1):1, 2021.
- P. P. Liang, Z. Deng, M. Q. Ma, J. Y. Zou, L.-P. Morency, and R. Salakhutdinov. Factorized contrastive learning: Going beyond multi-view redundancy. Advances in Neural Information Processing Systems, 36: 32971–32998, 2023.

- Z. Lin, S. Yu, Z. Kuang, D. Pathak, and D. Ramanan. Multimodality helps unimodality: Cross-modal few-shot learning with multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19325–19337, 2023.

- H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

M.-Y. Liu, T. Breuel, and J. Kautz. Unsupervised image-to-image translation networks. Advances in neural information processing systems, 30, 2017.

- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- J. Lu, D. Batra, D. Parikh, and S. Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019.

- S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

- M. Maniparambil, R. Akshulakov, Y. A. D. Djilali, M. El Amine Seddik, S. Narayan, K. Mangalam, and
- N. E. O’Connor. Do vision and language encoders represent the world similarly? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14334–14343, 2024.

S. Menon and C. Vondrick. Visual classification via description from large language models. arXiv preprint arXiv:2210.07183, 2022.

- J. Merullo, L. Castricato, C. Eickhoff, and E. Pavlick. Linearly mapping from image to text space. arXiv preprint arXiv:2209.15162, 2022.

D. Mizrahi, R. Bachmann, O. Kar, T. Yeo, M. Gao, A. Dehghan, and A. Zamir. 4m: Massively multimodal masked modeling. Advances in Neural Information Processing Systems, 36:58363–58408, 2023.

L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodol`a. Relative representations enable zero-shot latent space communication. arXiv preprint arXiv:2209.15430, 2022.

- M.-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pages 722–729. IEEE, 2008.

- A. Norelli, M. Fumero, V. Maiorca, L. Moschella, E. Rodola, and F. Locatello. Asif: Coupled data turns unimodal models to multimodal without training. Advances in Neural Information Processing Systems, 36: 15303–15319, 2023.

- O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pages 3498–3505. IEEE, 2012.

- S. Pratt, I. Covert, R. Liu, and A. Farhadi. What does a platypus look like? generating customized prompts for zero-shot image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15691–15701, 2023.

- A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

- R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

K. Roth, J. M. Kim, A. Koepke, O. Vinyals, C. Schmid, and Z. Akata. Waffling around for performance: Visual classification with random words and broad concepts. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15746–15757, 2023.

- S. Roy, F. Ogidi, A. Etemad, E. Dolatabadi, and A. Afkanpour. A shared encoder approach to multimodal representation learning. arXiv preprint arXiv:2503.01654, 2025.

- J. Ryu, C. Bunne, L. Pinello, A. Regev, and R. Lopez. Cross-modality matching and prediction of perturbation responses with labeled gromov-wasserstein optimal transport. arXiv preprint arXiv:2405.00838, 2024.

Y. Shi, V. De Bortoli, A. Campbell, and A. Doucet. Diffusion schr¨odinger bridge matching. Advances in Neural Information Processing Systems, 36:62183–62223, 2023.

- A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela. Flava: A foundational language and vision alignment model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15638–15650, 2022.

- K. Soomro, A. R. Zamir, and M. Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

- S. Srivastava and G. Sharma. Omnivec: Learning robust representations with cross modal sharing. 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1225–1237, 2023. URL https://api.semanticscholar.org/CorpusID:265128674.

- S. Srivastava and G. Sharma. Omnivec2 - a novel transformer based network for large scale multimodal and multitask learning. In CVPR, pages 27402–27414, 2024. URL https://doi.org/10.1109/CVPR52733. 2024.02588.

- N. Sturma, C. Squires, M. Drton, and C. Uhler. Unpaired multi-domain causal representation learning. Advances in Neural Information Processing Systems, 36:34465–34492, 2023.

- I. Sutskever. An Observation on Generalization. YouTube video, Aug 2023. URL https://www.youtube. com/watch?v=AKMuA_TVz3A. Accessed: 2025-09-24.

S. Timilsina, S. Shrestha, and X. Fu. Identifiable shared component analysis of unpaired multimodal mixtures. arXiv preprint arXiv:2409.19422, 2024.

C. Wang, S. Gupta, X. Zhang, S. Tonekaboni, S. Jegelka, T. Jaakkola, and C. Uhler. An information criterion for controlled disentanglement of multimodal data. arXiv preprint arXiv:2410.23996, 2024.

- J. Xi, J. Osea, Z. Xu, and J. S. Hartford. Propensity score alignment of unpaired multimodal data. Advances in Neural Information Processing Systems, 37:141103–141128, 2024.

J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485–3492. IEEE, 2010.

- A. Zadeh, R. Zellers, E. Pincus, and L.-P. Morency. Mosi: multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos. arXiv preprint arXiv:1606.06259, 2016.

X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, and L. Beyer. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18123–18133, 2022.

- X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- Y. Zhang, K. Gong, K. Zhang, H. Li, Y. Qiao, W. Ouyang, and X. Yue. Meta-transformer: A unified framework for multimodal learning. arXiv preprint arXiv:2307.10802, 2023.

J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.

### Appendix

- A Further Related Works 20
- B Supplementary Experimental Details and Assets Disclosure 20

- B.1 Assets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.2 Hardware and setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.3 Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- B.3.1 MultiBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.3.2 Image Classification Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.3.3 Constructing text templates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.3.4 ImageNet-ESC Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- B.4 Training Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- B.4.1 Unpaired Multimodal Representation Learning under Self-Supervision . . . . . . . 22
- B.4.2 Image Classification using Image and Unpaired Texts . . . . . . . . . . . . . . . . . 24
- B.4.3 Evaluation on ImageNet-ESC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- B.4.4 Transfer Learning from Language to Vision . . . . . . . . . . . . . . . . . . . . . . . 25

- C Proofs of Theoretical Results 25

- C.1 Background and Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- C.2 Maximum Likelihood Estimators and Fisher Contributions . . . . . . . . . . . . . . . . . . 26
- C.3 Theorems and Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- D UML: Algorithm Pseudocode 32
- E Additional Experiments 35

- E.1 Improving Image Classification using Unpaired Texts (Unaligned encoders) . . . . . . . . . 35

- E.1.1 Supervised Finetuning (across architectures) . . . . . . . . . . . . . . . . . . . . . . . 35
- E.1.2 Linear Probing (across architectures) . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- E.1.3 Few-shot Linear Probing (across architectures) . . . . . . . . . . . . . . . . . . . . . . 41

- E.2 Improving Image Classification using Unpaired Texts (Aligned encoders) . . . . . . . . . . 44

- E.2.1 Supervised Finetuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44

- E.2.2 Linear Probing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- E.2.3 Few-shot linear Probing (across architectures) . . . . . . . . . . . . . . . . . . . . . . 45

- E.3 Improving Visual Robustness Using Unpaired Texts . . . . . . . . . . . . . . . . . . . . . . . 45
- E.4 Marginal Rate-of-Substitution Between Modalities . . . . . . . . . . . . . . . . . . . . . . . . 49

- E.4.1 Unaligned Encoders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- E.4.2 Aligned Encoders (CLIP) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

- E.5 Impact of Scaling Vision Backbone . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- E.6 Impact of Varying Text Encoders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
- E.7 Learning with Coarse-Grained vs. Fine-Grained Textual Cues . . . . . . . . . . . . . . . . . 52
- E.8 Impact on Performance with Increasing Unpaired Text Prompts . . . . . . . . . . . . . . . . 53
- E.9 Effect of Unrelated Auxiliary Modalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
- E.10 Extension to More Than Two Modalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
- E.11 Effect of Ratio of Modality Batches . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
- E.12 Effect of Freezing vs. Unfreezing the Text Encoder . . . . . . . . . . . . . . . . . . . . . . . . 56
- E.13 Additional Experiments for Audio-Visual Setting . . . . . . . . . . . . . . . . . . . . . . . . 57

- E.13.1 Improving Image Classification with Unpaired Audio and Text (Unaligned encoders) 57
- E.13.2 Improving Image Classification with Unpaired Audio and Text (Aligned encoders) 57
- E.13.3 Improving Audio Classification with Unpaired Image and Text (Aligned encoders) 57

- E.14 Gaussian Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58

###### F Analysis of the Learned Predictor 59

- F.1 Multimodal Neurons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- F.2 Change in Decision Boundaries with Unpaired Data from Another Modality . . . . . . . . 60
- F.3 What do models learn from unpaired data? . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61

### A Further Related Works

Unpaired Multimodal Learning. Unpaired data has long been used for image-to-image [Zhu et al., 2017, Liu et al., 2017, Almahairi et al., 2018, Shi et al., 2023] and text-to-text translation [Lample et al., 2018] . More recently, several works have also proposed learning from unpaired data by inferring coarse- or finegrained alignments through distribution matching or optimal transport objectives [Xi et al., 2024, Demetci et al., 2022, Ryu et al., 2024]. In contrast, we leverage unpaired data for learning representations without the need for explicit or inferred alignment. [Timilsina et al., 2024, Sturma et al., 2023] theoretically analyze the problem of identifying shared latent components and causal structures in unaligned multimodal mixtures. Most closely related to our work is [Lin et al., 2023], which leverages coarse-grained text data such as class names to improve image classification on CLIP using a shared linear head. Another related line of works [Roth et al., 2023, Pratt et al., 2023, Menon and Vondrick, 2022, Gao et al., 2024] leverage prompting templates and pretrained LLMs to generate descriptive class captions, showing improved image classification performance with CLIP. Nonetheless, these methods operate on CLIP with pre-aligned representation spaces, whereas our approach also learns from unpaired data without assuming prior alignment. Several works have also proposed learning large multitask multimodal models with joint encoders and unified embedding spaces [Srivastava and Sharma, 2024, 2023, Zhang et al., 2023, Girdhar et al., 2022, Geng et al., 2022], often using joint training over separate tasks and/or masked prediction objectives. In a similar vein, [Chada et al., 2023] uses a stage-wise training strategy with both unpaired and paired data, and [Girdhar et al., 2023b] trains a single model across visual modalities. However, most of these methods rely on some amount of paired data for preliminary alignment and then leverage abundant modality-specific unpaired data for further improvement. In contrast, our approach demonstrates that a model can implicitly learn cross-modal correlations from purely unpaired data, without requiring explicit alignment as a prerequisite.

Multimodal Representation Alignment. Our method relies on the notion of shared information and structure between unaligned modalities. Closely related to this are works demonstrating that unimodal representations trained without multimodal data are nevertheless converging. Huh et al. [2024] presents evidence that better-performing language models exhibit increased alignment to self-supervised vision models. Similarly, [Maniparambil et al., 2024] shows a latent space alignment between vision and text encoders across backbones and training paradigms, and uses the CKA metric to connect unaligned encoders zero-shot. Earlier works also note alignment between models trained with different datasets and modalities [Moschella et al., 2022, Norelli et al., 2023]. Several works have also shown that a linear projection or MLP is sufficient to stitch together the latent spaces of pretrained vision and language models [Merullo et al., 2022, Liu et al., 2023, Koh et al., 2023]. Zhai et al. [2022] extends this to training a text encoder to align to a frozen pretrained image model; this method was in turn used to integrate DINOv2, a large self-supervised vision model, with a text encoder [Jose et al., 2024].

### B Supplementary Experimental Details and Assets Disclosure

###### B.1 Assets

We do not introduce new data in the course of this work. Instead, we use publicly available, widely used image datasets for the purposes of benchmarking and comparison.

###### B.2 Hardware and setup

Each experiment was conducted on 1 NVIDIA Tesla V100 GPUs, each with 32GB of accelerator RAM. The CPUs used were Intel Xeon E5-2698 v4 processors with 20 cores and 384GB of RAM. All experiments were implemented using the PyTorch deep learning framework.

###### B.3 Datasets

###### B.3.1 MultiBench

We evaluate our approach on a diverse set of multimodal fusion datasets from MultiBench [Liang et al., 2021] spanning healthcare, sentiment, and humor detection:

- • CMU-MOSEI [Moschella et al., 2022]: The largest sentence-level multimodal sentiment and emotion dataset, containing 23,000 annotated monologue videos (over 65 hours of content from more than 1,000 speakers across 250 topics). Each video is labeled with sentiment intensity in the range [−3,3], which we cast into binary positive/negative sentiment classification.
- • CMU-MOSI [Zadeh et al., 2016]: A related multimodal sentiment dataset with 2,199 YouTube video clips, reflecting real-world opinionated speech. Sentiment intensities are annotated in the range [−3,3], and we again formulate the task as binary sentiment classification.
- • UR-FUNNY [Hasan et al., 2019]: A large-scale humor detection dataset derived from over 16,000 TED talk videos. It includes 8,257 humorous punchlines identified by laughter markers, paired with 8,257 negative examples drawn from non-humorous contexts, forming a balanced binary humor classification task.
- • MUSTARD [Castro et al., 2019]: A multimodal sarcasm detection dataset with 690 video clips from TV shows. Each sample is annotated for the presence or absence of sarcasm, yielding a challenging binary classification task.
- • MIMIC-III [Johnson et al., 2016]: A large-scale clinical dataset with records of over 40,000 ICU patients. It combines time-series physiological measurements (recorded hourly over a 24-hour window) with static demographic and tabular features. We use it for binary classification of patients into a group of ICD-9 codes.

###### B.3.2 Image Classification Benchmarks

We evaluate on the following widely-used classification benchmarks: ImageNet [Deng et al., 2009], StanfordCars [Krause et al., 2013], UCF101 [Soomro et al., 2012], Caltech101 [Fei-Fei et al., 2004], Oxford Flowers [Nilsback and Zisserman, 2008], SUN397 [Xiao et al., 2010], DTD [Cimpoi et al., 2014], FGVCAircraft [Maji et al., 2013], OxfordPets [Parkhi et al., 2012], and Food101 [Bossard et al., 2014]. More details about the dataset and splits is provided in Table 3.

###### B.3.3 Constructing text templates

To construct conceptually related yet unpaired text data, we generate text templates that capture varying granularities of information about the dataset. Our first approach (Vanilla) uses the straightforward template \a photo of a {}" with a natural language label for each category, resulting in a basic text description for each class. However, this simple textual corpus lacks fine-grained information necessary

Table 3: Detailed statistics of the 10 datasets for image classification.

Dataset Classes Train Val Test Caltech101 [Fei-Fei et al., 2004] 100 4,128 1,649 2,465 OxfordPets [Parkhi et al., 2012] 37 2,944 736 3,669 StanfordCars [Krause et al., 2013] 196 6,509 1,635 8,041 Oxford Flowers [Nilsback and Zisserman, 2008] 102 4,093 1,633 2,463 Food101 [Bossard et al., 2014] 101 50,500 20,200 30,300 FGVCAircraft [Maji et al., 2013] 100 3,334 3,333 3,333 SUN397 [Xiao et al., 2010] 397 15,880 3,970 19,850 DTD [Cimpoi et al., 2014] 47 2,820 1,128 1,692 UCF101 [Soomro et al., 2012] 101 7,639 1,898 3,783 ImageNet [Deng et al., 2009] 1,000 1.28M N/A 50,000

to distinguish between visually similar subcategories or to resolve contextually ambiguous terms. To address this, for the second template, we draw from the extensive literature on improving text prompts for zero-shot classification in CLIP [Gao et al., 2024, Menon and Vondrick, 2022, Pratt et al., 2023, Roth et al., 2023]. Specifically, for the second approach (GPT-3 Descriptions), we adopt the text prompt generation strategy developed by Pratt et al. [2023], using large language models such as GPT-3 to generate diverse and contextually rich prompts for each image category. We use three generic hand-written sentences across the datasets:

Describe what a/the {} looks like: Describe a/the {} : What are the identifying characteristics of a/the {}?

The blank portion of each template is populated with the category name, along with the category type for specialized datasets (e.g., “pet” + {} for Oxford Pets or “aircraft” + {} for FGVC Aircraft). The type specification is important for disambiguating categories with multiple interpretations. Some examples of these descriptions are provided in Table 4 for the Oxford Pets dataset.

###### B.3.4 ImageNet-ESC Dataset

Experimental Setup. We extend our results beyond vision and language to an audiovisual-language dataset: the ImageNet-ESC benchmark [Lin et al., 2023]. This benchmark combines ImageNet (1000 object categories) and ESC-50 (50 environmental sound classes) by matching classes that logically correspond. For example, the dog (barking) class from ESC-50 aligns with various dog breeds from ImageNet, while the clock-alarm sound maps to both analog clock and digital clock. This alignment captures the relationship between visual objects, their sounds, and their textual descriptions. The benchmark consists of two versions: 1) ImageNet-ESC-27: A broader set including loosely matched visual-audio pairs (e.g., drinkingsipping to water bottle); 2) ImageNet-ESC-19: A more precise subset containing only accurate visual-audio matches.

###### B.4 Training Protocol

###### B.4.1 Unpaired Multimodal Representation Learning under Self-Supervision

MultiBench. For the MIMIC dataset, which contains tabular and medical time-series inputs, we train models directly on the raw modality inputs. For the four video datasets (MOSEI, MOSI, UR-FUNNY,

Table 4: Sample text descriptions per class for Oxford Pets dataset

Class Examples Wheaten Terrier A wheaten terrier is a small, shaggy dog with a soft, silky coat.

A wheaten terrier has a soft, wheat-colored coat that is low-shedding and hypoallergenic. The wheaten terrier is a medium-sized, hypoallergenic dog breed. A pet Wheaten Terrier usually has an intelligent expression and a soft, wheat-colored coat.

Great Pyrenees A great pyrenees is a large, white, shaggy-coated dog. A Great Pyrenees is a large, fluffy dog with a calm, gentle disposition. The great pyrenees was originally bred to protect livestock from predators. Great Pyrenees are known for being very large, white dogs with thick fur.

Sphynx A pet Sphynx typically has a small, wrinkled head and a hairless body. A Sphynx is a hairless cat breed known for its soft, warm skin. A Sphynx often displays large ears, pronounced cheekbones, and no fur. Sphynx are unique cats characterized by their lack of coat and wrinkled skin.

Birman A Birman is a long-haired, color-pointed cat with a “mask” of darker fur on its face. A Birman has silky, pale cream to ivory fur with deep seal- or lilac-colored points. Birman cats possess striking blue eyes and contrasting white “gloves” on their paws. They are known for being gentle, affectionate, and smooth-coated companions.

Pomeranian A Pomeranian is a small, fluffy dog with a thick double coat. Pomeranians are toy-sized, alert dogs with fox-like faces and plumed tails. A pet Pomeranian often comes in orange, black, white, or mixed coat colors. They are lively, outgoing, and known for their bold, friendly personalities.

and MUSTARD), we train models on standard pre-extracted features from text, video, and audio modalities [Liang et al., 2021]. All models are trained for 100 epochs, with hyperparameter search over learning rates {10−2,10−3,10−4}. We report the best performance across learning rates, averaged over three random seeds. To train Uml, each modality is first projected into a shared embedding space of dimension d ∈ {10,40,150,300} via a learned linear transformation. The projected inputs are processed by a shared 5-layer, 5-head autoregressive Transformer encoder, followed by a linear projection back to the original modality dimension. Training uses a next-token/patch embedding prediction objective, with both modalities sharing the Transformer backbone to encourage cross-modal synergies in the latent space.

- At inference time, we average the Transformer outputs across sequence length and use the resulting embeddings for linear probing on downstream classification tasks. All models (Unimodal baseline and Uml) are trained for 100 epochs, with hyperparameter search over learning rates {10−2,10−3,10−4}. For Uml, in addition, we perform hyperparameter search over a curriculum parameter step ∈ {0,30,50,70}. This parameter controls whether training begins on X alone for the first step epochs before switching to joint training, with step = 0 corresponding to joint training from the start. For each dataset, we select the best-performing model on the validation set and report test accuracy averaged over three random seeds.

Standard Vision-Text Benchmarks. We extract image and text embeddings using ViT-B/14 DINOv2 and OpenLLaMA-3B, respectively. As in MultiBench, patch and token embeddings are projected to a shared 256-dimensional space via modality-specific linear layers. A 4-layer, 4-head transformer serves as the shared encoder, with outputs projected back to the original embedding dimensions using modality-specific linear projections. We perform rigorous hyperparameter tuning for both the unimodal baseline and Uml, and report average test accuracy of the best model across three seeds. Full hyperparameter ranges are listed in Table 5.

###### B.4.2 Image Classification using Image and Unpaired Texts

For text, we use OpenLLaMA-3B as our default encoder and ablate against BERT-Large, RoBERTa-Large, GPT-2 Large, and the pre-aligned CLIP text encoder, keeping the text encoder frozen. For images, our main backbone is ViT-S/14 DINOv2, with ablations across other DINOv2 variants and the CLIP vision encoder. In the linear-probe setting, all encoder weights stay fixed and we train only a single linear classification head; in full fine-tuning, we jointly update the image backbone and that head, while still freezing the text encoder.

We optimize cross-entropy loss via AdamW [Loshchilov and Hutter, 2017] and perform an extensive grid search over learning rate, weight decay, cosine learning rate scheduling with linear warmup, dropout, and a learnable, modality-specific scaling on the logits. The results are reported for the best-performing model on the validation dataset. We report results for the model achieving highest validation accuracy; the full hyperparameter ranges are in Table 5.

For full fine-tuning, we jointly update the image backbone and classification head with a fixed learning rate of 5 × 10−5, batch size 64, and omit learnable modality-specific scaling, since it showed no benefit in this setting.

Table 5: Hyperparameter grid for linear probing.

Hyperparameter Values

Optimizer adamw Learning rate {0.001, 1e-4} Weight decay {0.0, 0.01, 0.001} LR scheduler cosine Batch size {8, 32} Max iterations 12,800 Warmup iterations 50 Warmup type linear Warmup min LR 1e-5 Dropout {0.0} Modality-specific learnable scaling {False, True} Early-stop patience 10

###### B.4.3 Evaluation on ImageNet-ESC

Similar to our vision-language experiments, we perform few-shot evaluation using the 5-fold splits defined in the benchmark. Each fold contains 8 samples per class, with one fold used for training and validation and the remaining four for testing. We repeat the process over 5 random splits and report the average performance. For audio encoding, we use AudioCLIP with an ES-ResNeXT backbone [Guzhov et al., 2021]. AudioCLIP is pretrained on AudioSet and generates audio embeddings in the same representation space as CLIP. Following the instructions in [Guzhov et al., 2021, Lin et al., 2023], we use train() mode in Pytorch to extract the features since eval() mode yields suboptimal embeddings. We evaluate our models on two tasks—audio classification and image classification—comparing the unimodal baseline against two multimodal variants in which the primary modality is each time augmented by one of the other modalities.

###### B.4.4 Transfer Learning from Language to Vision

To adapt a language model to image classification, we embed image patches using a linear projection and add positional encodings to capture spatial structure. We then use transformer layers initialized from pretrained BERT, and finally, a 2-layer MLP classification head. Specifically, we split each image of size 224 × 224 into patches of size 16 × 16 with 196 patch tokens. Each patch is then projected into the model’s embedding space of dimension d(e.g. d=768 for GPT-2, d = 1024 for BERT) via a learned linear layer. We then prepend a learnable “[CLS]” token, add learned positional embeddings of shape (N + 1) × d, and apply dropout with probability p = 0.1. This (N + 1) × d sequence is passed into the pretrained transformer stack (either GPT-2 or BERT), using a full bidirectional attention mask over all patch tokens and the CLS token. We extract the final hidden state corresponding to the CLS token and feed it through a two-layer MLP classification head.

During training, we evaluate two scenarios: 1) one where the pretrained backbone is frozen and only the patch embedding and linear head are trained, and 2) another where the backbone is initially frozen to align the trainable layers (patch embedding and head) with the pretrained language backbone, and then unfrozen after 2000 steps for end-to-end training. This approach allows us to test whether the semantic richness captured by language models provides a strong initialization, leading to better convergence and performance compared to training ViT from scratch.

### C Proofs of Theoretical Results

In this section, we present complete derivations and proofs of the main theoretical claims. Section C.1 gathers all definitions and background required for our arguments. Section C.2 formalizes the linear datagenerating model, derives closed-form maximum-likelihood estimators for each modality and their joint estimator, and computes the corresponding block-wise Fisher information. Finally, Section C.3 provides the detailed proofs of our variance-reduction claims, showing rigorously how unpaired multimodal estimation strictly lowers estimator variance.

###### C.1 Background and Definitions

In this section we revisit the mathematical definitions used in our theoretical analysis, including matrixorderings, characterization of symmetric matrices and Fisher information.

- Definition 1 (Positive Semidefinite Matrix). A real symmetric matrix A ∈ Rd×d is positive semidefinite if for all vectors v ∈ Rd, v⊤A v ≥ 0. Equivalently, all eigenvalues of A are nonnegative. We denote the set of all d × d symmetric, positive-semidefinite matrices as S⪰d 0.
- Definition 2 (Positive Definite Matrix). A real symmetric matrix A ∈ Rd×d is positive definite if for every nonzero v ∈ Rd, v⊤A v > 0. Equivalently, all eigenvalues of A are strictly positive. We denote the set of all d × d symmetric, positive definite matrices as S≻d 0.
- Definition 3 (Loewner Order). For two real symmetric matrices A, B ∈ Rd×d, we write A ⪯ B ⇐⇒ B − A is positive semidefinite and A ≺ B ⇐⇒ B − A is positive definite. This defines a partial order on the cone of symmetric matrices.
- Definition 4 (Fisher Information Matrix). Given a parametric family of densities p(x; θ) on data x, the Fisher information matrix at parameter θ is

I(θ) = Ex∼p(·;θ) ∇θ log p(x; θ) ∇θ log p(x; θ)⊤ .

Equivalently, for regular models, I(θ) = −E ∇2θ log p(x; θ) .

###### C.2 Maximum Likelihood Estimators and Fisher Contributions

In this section we revisit our linear data–generating model, introduce notations for the X–only, Y–only and joint likelihoods, derive the closed-form MLEs θX, θY and θX,Y, and formalize their information contributions towards estimating the ground truth parameters θ ≡ [θc, θx, θy]⊤.

Data Generating Process. Recall our linear data-generating process: Assume that all factors of variation in reality live in a single d-dimensional space Z∗ ≡ θ ∈ Rd modeled using a linear data-generating pipeline. This parameter can further be decomposed as θ ≡ [θc, θx, θy]⊤ where θc ∈ Rdc, θx ∈ Rdx, θy ∈ Rdy and dc + dx + dy = d. Here, θc captures the common (shared) parameters that affect both modalities, θx denotes the parameters that only affect modality X, and θy denotes the parameters that only affect modality Y. We observe two independent datasets, one from each modality {Xi}iN=x1 ∈ Rm and {Yj}Nj=y1 ∈ Rn, each reflecting partial measurements of the ground truth latent space Z∗:

Xi = Ac,i θc + Ax,i θx + ϵX,i, ϵX,i ∼ N 0, σx2Imi (3) Yj = Bc,j θc + By,j θy + ϵY,j, ϵY,j ∼ N 0, σy2Inj . (4)

Here, Ac,i, Ax,i, Bc,j, By,j are known design blocks capturing how each sample probes the latent factors and εX,i,εY,j represent the independent measurement noise.

In our linear setting, estimating the true latent state θ—and hence the underlying reality Z∗—is

governed by the Fisher information matrix I(θ) = −E ∇2θ ℓ(θ) , which measures how sharply the likelihood “curves” around the true θ. High curvature along a particular axis means the data tightly constrain that component, driving down estimator variance there.

Unimodal Estimators. We first estimate θ using only the X–dataset. Stacking {Xi}iN=x1 yields a design matrix A with block rows [Ac,i, Ax,i, 0]. The least-squares solution

Nx

2

#### ∑

θX = argmin

Xi − Ac,i θc − Ax,i θx

θ

i=1

omits θy entirely. Consequently, the Fisher information on θy vanishes, making it unidentifiable.

Analogously, stacking {Yj}Nj=y1 defines B with block rows [Bc,j, 0, By,j] and yields

Ny

2

#### ∑

θY = argmin

.

Yj − Bc,j θc − By,j θy

θ

j=1

This estimator doesn’t depend on θx, providing zero coverage for that component. Thus, each unimodal estimator entirely fails to recover the parameters exclusive to the omitted modality.

Multimodal Estimators. Despite the lack of one-to-one pairing, both {Xi} and {Yj} share the common parameters θc. Since the two distributions are conditionally independent, the joint likelihood factorizes as

Nx

#### ∏

p(Xi | θc, θx) ×

i=1

Ny

#### ∏

p(Yj | θc, θy).

j=1

Maximizing this yields the combined estimator

θX,Y = arg min

θc,θx,θy

Nx

∥Xi − Ac,i θc − Ax,i θx∥2 +

#### ∑

i=1

Ny

∥Yj − Bc,j θc − By,j θy∥2 .

#### ∑

j=1

Intuitively, there is no requirement to match up individual (Xi,Yj) pairs. Instead, the estimate for θc is improved by both modalities while remaining unpaired.

Fisher Information. In our linear model, each dataset contributes block-structured Fisher information. For the X–dataset:

  

   ,

A⊤c,iAc,i A⊤c,iAx,i 0 A⊤x,iAc,i A⊤x,iAx,i 0

Nx

#### ∑

IX =

i=1

0 0 0

and for the Y–dataset:

  

   .

Bc⊤,jBc,j 0 Bc⊤,jBy,j 0 0 0

Ny

#### ∑

IY =

By⊤,jBc,j 0 By⊤,jBy,j

j=1

Because X and Y samples are independent, their curvature contributions add pointwise, resulting in the joint Fisher information being simply the sum of the unimodal blocks.

  

   ,

∑i A⊤c,iAc,i + ∑j Bc⊤,jBc,j ∗ ∗ ∗ ∑i A⊤x,iAx,i 0 ∗ 0 ∑j By⊤,jBy,j

IX,Y = IX + IY =

where “∗” denotes the cross-modal blocks. In particular, we have the shared-parameter block as

Ny

Nx

#### ∑

#### ∑

A⊤c,iAc,i +

Bc⊤,jBc,j,

(IX,Y)θc,θc =

i=1

j=1

###### C.3 Theorems and Proofs

The aim of this section is to detail the proofs of the theoretical results presented in the main manuscript The key theoretical tools driving our analysis are already prepared in Section C.1 and Section C.2. Core to our theoretical analysis are a few lemmas around the Loewner-order monotonicity result for inverses that we prove below.

- Lemma 1 (Loewner Order reversal for inverses). Let M, N ∈ Sd≻0 with M ≺ N (or M ⪯ N). Then N−1 ≺ M−1 (or N−1 ⪯ M−1) .

Proof. Since N ≻ 0, N−1/2 exists and is nonsingular. Define C := N−1/2MN−1/2 ≺ I. Because a congruence with an invertible matrix preserves positive-definiteness, C ≻ 0; hence C−1 is well defined and C−1 ≻ I (the scalar map x  → x−1 is strictly decreasing on (0, ∞)). Undoing the congruence gives

M−1 = N−1/2C−1N−1/2 ≻ N−1/2IN−1/2 = N−1.

| |
|---|

- Lemma 2 (Inverse–monotonicity of the Moore–Penrose pseudoinverse). Let M, N ∈ Sd⪰0 satisfy M ≺ N and ker M = ker N =: K. Then their pseudoinverses obey N† ≺ M†.

Proof. Set S := K⊥ and let P := PS be the orthogonal projector onto S. Because M and N vanish on K, we have the decompositions M = PMP and N = PNP. Restricted to S both matrices are positive–definite:

M˜ := PMP, N˜ := PNP ∈ Sdim≻0 S, M˜ ≺ N˜ .

Apply Lemma 1 to M˜ , N˜ to obtain N˜ −1 ≺ M˜ −1 on S. The Moore–Penrose pseudoinverse equals the ordinary inverse on S and is zero on K:

M† = PM˜ −1P, N† = PN˜ −1P. Therefore N† = PN˜ −1P ≺ PM˜ −1P = M†.

| |
|---|

- Lemma 3 (Directional Loewner Order reversal). Let M, N ∈ Sd≻0 with M ⪯ N. If a non-zero vector v satisfies v⊤Mv < v⊤Nv, then

- 1. For the vector v, it holds that v⊤M−1v ≥ v⊤N−1v, with strict inequality v⊤M−1v > v⊤N−1v if and only if (N − M)M−1v ̸= 0.
- 2. There exists a non-zero vector u ∈ Rd such that u⊤M−1u > u⊤N−1u.

Proof. Denote the Loewner gap ∆ := N − M ⪰ 0. Then, the assumption v⊤Nv > v⊤Mv is equivalent to v⊤∆v > 0. Introduce the congruence–invariant normalisation C := M−1/2∆M−1/2 ⪰ 0. Now, using ∆ = M1/2CM1/2 and properties of inverse,

N = M1/2(I + C)M1/2, N−1 = M−1/2(I + C)−1M−1/2, since I + C ≻ 0 (because C ⪰ 0 and I ≻ 0). Thus,

M−1 − N−1 = M−1/2 I − (I + C)−1 M−1/2

= M−1/2C(I + C)−1M−1/2,

because (I − (I + C)−1)(I + C) = C. Finally, evaluating in the direction v, we have v⊤(M−1 − N−1)v = v⊤M−1/2(I + C)−1CM−1/2v

= u⊤(I + C)−1Cu (where u = M−1/2v)

Now, since (I + C)−1 ∈ S≻0 and C ∈ S⪰0 commute, the matrix (I + C)−1C is positive semidefinite and it has exactly the same kernel as C. Thus, if C = Q diag(λi)Q⊤ (λi ≥ 0), we have

λi 1 + λi

u⊤C(I +C)−1u = ∑

(Q⊤u)2i ≥ 0.

i

This expression is strictly positive exactly when u has a component in any eigen-subspace with λi > 0 i.e when u ̸∈ ker(C). Since M−1/2 ∈ S≻0, Cu = 0 =⇒ M−1/2∆M−1/2u = 0 =⇒ ∆M−1v = 0. Thus, this expression is strictly positive if ∆M−1v ̸= 0.

Now, from the premise v⊤∆v > 0, it follows that ∆ ̸= 0. Since M ≻ 0, M−1/2 is invertible, C is also not the zero matrix. Since C ⪰ 0, this means that C must have at least one strictly positive eigenvalue. Let λ > 0 be such an eigenvalue, and let z ̸= 0 be a corresponding eigenvector. Define, x := M1/2z ̸= 0. Thus, we have x⊤(M−1 − N−1)x = z⊤C(I + C)−1z = 1+λλ∥z∥2 > 0, showing the existence of a non-zero vector x such that x⊤M−1x > x⊤N−1x.

| |
|---|

- Theorem 1 (Restatement of Theorem 1). Let θˆX, θˆY be the least-squares estimators for θ using only {Xi} and only {Yj} and let θˆX,Y be the joint estimator using both unpaired datasets. Then, under the assumption that at least one Bc,j where j ∈ {1,2,...Ny} has full rank, the common-factor covariance satisfies the strict Loewner ordering i.e.

Var θ ˆX,Y

≺ Var θ ˆX

, or equivalently, the Fisher information on θc strictly increases when combining both modalities, despite not having sample-wise pairing: (IX + IY)θc,θc ≻ (IX)θc,θc.

θc,θc

θc,θc

Proof. For any statistic S(θ) = ∇θ log p(x; θ) and vector v,

v⊤I(θ) v = v⊤E[S(θ)S(θ)⊤] v = E (v⊤S(θ))2 ≥ 0. Thus, a Fisher Information Matrix is a positive semidefinite matrix.

In our linear–Gaussian model, the X–dataset contributes (IX)θc,θc = ∑iN=x1 A⊤c,iAc,i and the Y–dataset gives (IY)θc,θc = ∑Nj=y1 Bc⊤,jBc,j. Since at least one Bc,j has full column rank, (IY)θc,θc is positive-definite on the θc subspace. Now, if at least one Bc,j ∈ Rm×dc has full column rank dc, then for any v ∈ Rdc \ {0},

v⊤Bc⊤,jBc,j v = ∥Bc,jv∥2 > 0.

Hence, each summand in (IY)θc,θc is positive semidefinite and at least one is positive definite, so their sum ∑j Bc⊤,jBc,j is positive definite on the θc subspace. Thus,

|(IX)θc,θc ≺ (IX)θc,θc + (IY)θc,θc = (IX + IY)θc,θc|
|---|

Now, for regular exponential families (including Gaussian linear models), the covariance matrix of the maximum likelihood estimator θ near the true θ0 is (asymptotically) the inverse of the Fisher information matrix i.e. Var( θ) ≈ I(θ0)−1. Precisely, as the sample size n → ∞, we have:

###### √n(θˆ − θ0) →d N(0, I(θ0)−1),

where θ0 is the true parameter value, I(θ0) is the Fisher Information Matrix evaluated at θ0 and N(0, I(θ0)−1) denotes a multivariate normal distribution with mean 0 and covariance matrix I(θ0)−1. Thus, we compare variances via the Moore–Penrose pseudoinverse of the information matrices.

Let MX = (IX)θc,θc, MY = (IY)θc,θc and MX,Y = (IX + IY)θc,θc. Since MY ≻ 0, MX,Y = MX + MY is also positive definite (as MX,Y ⪰ MY ≻ 0). Thus, Var(θˆX,Y) = M−1

X,Y. We have established MX ≺ MX,Y. Assuming MX is positive definite (to define the matrix Var θ ˆX,Y

), we apply Lemma 1 to get M−1

X,Y ≺ M−1

θc,θc

X . Thus,

Var θ ˆX,Y

X = Var θ ˆX

###### = M−1

X,Y ≺ M−1

,

θc,θc

θc,θc

This proves the statement under the condition that MX is positive definite. Note here that, on spaces unidentifiable by X-alone i.e. v ∈ ker(MX), we have Var θ ˆX

= ∞. Since MX,Y is positive definite, it has finite variance along such v i.e. Var θ ˆX,Y

θc,θc

< ∞, thus strictly reducing the variance of the estimator. Thus, adding the unpaired Y-modality strictly reduces the variance (or, dually, increases the Fisher information) on the common factors θc.

θc,θc

| |
|---|

- Theorem 2 (Restatement of Theorem 2). Let all notation be as in Theorem 1, and define MX := (IX)θc,θc, MY := (IY)θc,θc, and MXY := MX + MY. Let v ∈ Rdc \ {0}. If there exists at least one index j ∈ {1,2,...Ny} such that Bc,jv ̸= 0, then the following hold:

- 1. The Fisher information strictly increases in direction v i.e. v⊤MXY v > v⊤MXv.
- 2. The variance of the estimator in direction v is strictly reduced i.e v⊤ Var θ ˆX,Y θc,θc

v < v⊤ Var θ ˆX

v, if v ̸∈ range(MX). For v ∈ range(MX), this strict inequality holds for v under an additional invertibility condition and is always guaranteed for some u ∈ range(MX) i.e. ∃u s.t. u⊤ Var θ ˆX,Y

θc,θc

u < u⊤ Var θ ˆX

θc,θc

u.

θc,θc

Proof. Define MX := (IX)θc,θc, MY := (IY)θc,θc, andMXY := MX + MY. By assumption, ∃j such that Bc,jv ̸= 0. Thus:

Ny

∥Bc,jv∥2 ≥ ∥Bc,jv∥2 > 0.

#### ∑

v⊤MYv =

j=1

Hence MY is positive-definite in direction v, implying MX,Y ≻ MX in this direction:

v⊤MXYv = v⊤MXv + v⊤MYv > v⊤MXv, thus proving the first part of the theorem.

- Case 1: v ∈/ Range(MX). If v ∈/ Range(MX), then v has a non-zero component in ker(MX). Let v = vS + vK, where vS ∈ Range(MX) and vK ∈ ker(MX) with vK ̸= 0. The linear combination of

parameters v⊤θc = v⊤S θc + v⊤K θc. Since vK ∈ ker(MX), the component v⊤K θc is not identifiable by the X-only model. Consequently, the asymptotic variance of an unbiased estimator for v⊤θc using only the

- X-dataset is infinite. We denote this as v⊤Var(θˆX)θc,θcv = ∞.

The strict inequality v⊤MXYv > 0, ensures that v ∈/ ker(MXY), and thus v ∈ Range(MXY). Since v ∈ Range(MXY) and v ̸= 0, MXY† v is well-defined. Furthermore, because MXY is positive semidefinite, MXY† is also positive semidefinite and shares the same kernel as MXY (since MXY is symmetric). As v ̸= 0 and v ∈/ ker(MXY), thus v ∈/ ker(MXY† ), which ensures v⊤MXY† v is a finite positive value. Thus,

v⊤Var(θˆX,Y)θc,θcv < ∞.

Comparing this to the variance from the X-only model in this case:

v⊤Var(θˆX,Y)θc,θcv < ∞ = v⊤Var(θˆX)θc,θcv, and the strict inequality holds.

- Case 2: v ∈ Range(MX). Let S := Range(MX) and let PS be the orthogonal projector onto S. Because MX = MXPS and MXY = MX + MY, the restrictions

M˜ X := PSMXPS, M˜ XY := PSMXYPS = M˜ X + PSMYPS are positive-definite on S; To see this, take any non-zero w ∈ S. Since w ∈ range(MX), PSw = w; hence w⊤M˜ Xw = w⊤MXw > 0 (PS is identity when restricted to S) Thus M˜ X ≻ 0 on S. Because PSMYPS ⪰ 0, adding it preserves positive-definiteness, so M˜ XY = M˜ X + PSMYPS ⪰ M˜ X ≻ 0 on S.

Applying Lemma 3(1) to M˜ X and M˜ XY on S gives us v⊤M˜ −1

XYv ≤ v⊤M˜ −1

X v. Strict inequality v⊤M˜ −1

XYv < v⊤M˜ −1

X v holds if and only if the condition Cv := ((M˜ XY − M˜ X)M˜ −1

X v ̸= 0) is met. Therefore, if condition Cv holds, the directional variance along this constrained space S is strictly reduced:

X v = v⊤Var(θˆX)θc,θcv1.

v⊤Var(θˆX,Y)θc,θcv = v⊤M˜ −1

XYv < v⊤M˜ −1

Further, from Lemma 3(2), there exists some non-zero vector u ∈ S such that u⊤M˜ −1

XYu < u⊤M˜ −1

X u. Thus we have,

u⊤Var(θˆX,Y)θc,θcu < u⊤Var(θˆX)θc,θcu. Thus, completing the proof.

1We note that true asymptotic variance defined as v⊤Var(θˆX,Y)θc,θcv = v⊤MXY† v, v⊤MXY† v = v⊤M˜ −1

XYv if S is an invariant subspace of MXY and MXY is block-diagonal with respect to S and S⊥ (i.e., PSMXYPS⊥ = 0, which implies PSMYPS⊥ = 0).

- Corollary 1. Assume a direction v ∈ Rdc \ {0} with a = v⊤(IX)θc,θc v > 0 and b = v⊤(IY)θc,θc v > 0 where v is the common eigenvector of (IX)θc,θc and (IY)θc,θc. Then the variance in direction v contracts by the factor

v⊤Var(θˆX,Y) v v⊤Var(θˆX) v

=

1/(a + b) 1/a

=

a a + b

< 1,

So the joint estimator achieves strictly lower error along v.

Proof. Let MX = (IX)θc,θc and MY = (IY)θc,θc. By assumption, v is a common eigenvector of MX and MY. Thus, MXv = λXv and MYv = λYv for some eigenvalues λX and λY. From the assumptions, we have λX = a/∥v∥2 > 0 and λY = b/∥v∥2 > 0. Since MX is symmetric and MXv = λXv with λX > 0, the pseudoinverse acts as MX† v = λ−X1v. Therefore, the variance in direction v for the X-only estimator is

v⊤Var(θˆX)θc,θc v = v⊤MX† v = v⊤(λ−X1v) = λ−X1∥v∥2 = a−1∥v∥4. Since v is a common eigenvector, it is also an eigenvector of MXY = MX + MY:

(MX + MY)v = MXv + MYv = λXv + λYv = (λX + λY)v.

The corresponding eigenvalue is λXY = λX + λY. Since λX > 0 and λY > 0, λXY > 0. Thus, (MX + MY)†v = (λX + λY)−1v. The variance in direction v for the joint estimator is

v⊤Var(θˆX,Y)θc,θc v = v⊤(MX + MY)†v = (λX + λY)−1∥v∥2 = (a + b)−1∥v∥4. Now, we form the ratio of these variances:

v⊤Var(θˆX,Y)θc,θc v v⊤Var(θˆX)θc,θc v

=

λX λX + λY

=

a a + b

< 1.

| |
|---|

- Corollary 2. Assume a direction v ∈ Rdc \ {0} with v⊤(IX)θc,θc v = 0 and v⊤(IY)θc,θc v > 0. Then v⊤Var(θˆX) v = ∞ and v⊤Var(θˆX,Y) v < ∞ i.e. a direction unidentifiable from X alone becomes well-posed with even unpaired data from Y.

Proof. This corollary follows directly from Case 1 of Theorem 2. The condition v⊤(IX)θc,θc v = 0 for v ̸= 0 implies v ∈ ker((IX)θc,θc), and thus v ̸∈ range((IX)θc,θc). Given the additional condition v⊤(IY)θc,θc v > 0, the conclusions of Case 1 of the theorem apply directly.

| |
|---|

- Corollary 3 (Variance Reduction for Eigenvectors of MX). Let v ∈ Rdc \ {0} be an eigenvector of MX = (IX)θc,θc with a corresponding eigenvalue λX > 0. If the Y-dataset provides information in this direction v (i.e.,

v⊤MYv > 0, where MY = (IY)θc,θc), then the variance in direction v is strictly reduced by incorporating the

- Y-dataset: v⊤Var(θˆX,Y)θc,θc v < v⊤Var(θˆX)θc,θc v.

Specifically, v⊤Var(θˆX)θc,θcv = λ−X1∥v∥2.

Proof. Let MX = (IX)θc,θc and MY = (IY)θc,θc. Since v is an eigenvector of MX with a positive eigenvalue λX > 0, it follows that v ∈ Range(MX). Let S = Range(MX). The variance using only the X-dataset in direction v is given by

v⊤Var(θˆX)θc,θcv = v⊤MX† v. Because v is an eigenvector of MX with λX > 0, MX† v = λ−X1v. Thus, v⊤Var(θˆX)θc,θcv = v⊤(λ−X1v) = λ−X1∥v∥2.

This scenario falls under Case 2 of Theorem 2, specifically its conclusion regarding v ∈ S. According to that theorem, strict variance reduction v⊤Var(θˆX,Y)θc,θcv < v⊤Var(θˆX)θc,θcv occurs if the condition Cv = ((PSMYPS)(MX|S)−1v ̸= 0) holds. Here, PS is the orthogonal projector onto S, and MX|S is the restriction of MX to S, so (MX|S)−1v = λ−X1v.

The condition Cv thus becomes (PSMYPS)(λ−X1v) ̸= 0. Since λX > 0, this is equivalent to PSMYPSv ̸= 0. We are given that v⊤MYv > 0. As v ∈ S, PSv = v. Therefore, v⊤MYv = v⊤PSMYPSv > 0. Let AS = PSMYPS restricted to S. AS is a positive semidefinite operator on S. The condition v⊤ASv > 0 for v ∈ S, v ̸= 0 implies that ASv ̸= 0 (because if ASv = 0, then v⊤ASv = 0, which contradicts v⊤ASv > 0). Thus, PSMYPSv ̸= 0, which means the condition Cv is satisfied.

Since v ∈ S and the condition Cv for strict inequality is met, by Theorem 2, it follows that v⊤Var(θˆX,Y)θc,θc v < v⊤Var(θˆX)θc,θc v.

| |
|---|

- Theorem 3 (Restatement of Theorem 3). Define for any m, IX(m) = ∑im=1 A⊤c,iAc,i and IY(m) = ∑mj=1 Bc⊤,jBc,j. If range IY(m) ̸⊆ range IX(m) , then there exists a nonzero v ∈ Rdc such that v⊤IY(m) v > v⊤IX(m)v.

Proof. Let RX := range IX(m) , RY := range IY(m) . By the assumption RY ̸⊆ RX, choose a vector w ∈ RY \ RX. Since Rdc is a finite dimensional inner product space and RX is its finite dimensional subspace, we can decompose w = w|| + v with w|| ∈ RX and v ∈ R⊥X. Because w ∈/ RX, the orthogonal component v is non-zero.

- (i) Term from IX(m). From the Fundamental Theorem of Linear Algebra, for any symmetric matrix S,

ker S = range(S)⊥; hence R⊥X = ker IX(m). Thus

v⊤IX(m)v = 0.

- (ii) Term from IY(m). Because w ∈ RY = range(IY(m)), there exists u with w = IY(m)u. Suppose, for

contradiction, that IY(m)v = 0. Then v ∈ ker IY(m) = RY⊥, so v ⊥ w. But w · v = (w∥ + v) · v = w∥· v + ∥v∥2 = ∥v∥2 > 0 because v ⊥ w∥ while v ̸= 0. This contradicts v ⊥ w; therefore IY(m)v ̸= 0 and, by positive semidefiniteness,

v⊤IY(m)v > 0.

Combining the above inequalities yields v⊤IY(m)v > v⊤IX(m)v, with v ̸= 0, which is the desired inequality.

| |
|---|

### D UML: Algorithm Pseudocode

In this section we present the full pseudocode for Uml for both the self-supervised and supervised settings as shown in Algorithm 1 and Algorithm 2 respectively.

###### Algorithm 1 Pytorch Pseudocode for Uml in the self-supervised setting

|# f_img , f_text: image encoder , text encoder<br><br># g_img , g_text: image decoder , text decoder # h: shared backbone<br><br><br>while not converged: # training loop x_img = fetch_next(image_loader) # image minibatch x_text = fetch_next(text_loader) # text minibatch (random/unaligned)<br><br>x_img = patchify_and_embed(x_img) # input image patch embeddings<br><br>x_text = tokenize_and_embed(x_text) # input text token embeddings<br><br>z_img = f_img(x_img) # image patch embeddings z_text = f_text(x_text) # text token embeddings<br><br>y_img = g_img(h(z_img)) # predict image patch embeddings<br><br><br>y_text = g_text(h(z_text)) # predict text patch embeddings<br><br><br># Next Token/Patch Embedding Prediction Loss loss_img = MSE(y_img[:,:-1,:], x_img[:,1:,:]) loss_text = MSE(y_text[:,:-1,:], x_text[:,1:,:]) loss = loss_img + lambda * loss_text # total loss<br><br>loss.backward() # back -propagate update(h, f_img , f_text , g_img , g_text) # SGD update<br><br># Define Mean Squared Error loss def MSE(pred , target):<br><br>return ((pred - target) ** 2).mean()|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28

###### Algorithm 2 Pytorch Pseudocode for Uml in the supervised setting

|# f_img: image encoder (frozen or trainable) # f_text: text encoder (frozen) # is_trainable: True if f_img is trainable else False # h: classification head<br><br>while not converged: # training loop x_img = fetch_next(image_loader) # image minibatch x_text = fetch_next(text_loader) # text minibatch (random/unaligned)<br><br>z_img = f_img(x_img) # image embeddings z_text = f_text(x_text) # text embeddings<br><br>logits_img = h(z_img) # predict image labels logits_text = h(z_text) # predict text labels<br><br>loss_img = CE(logits_img , labels_img) # image classification loss loss_text = CE(logits_text , labels_text) # text classification loss loss = loss_img + lambda * loss_text # total loss<br><br>loss.backward() # back -propagate update(h, f_img) if is_trainable else update(h) # SGD update<br><br># Define Cross -Entropy loss def CE(logits , labels):<br><br>return -sum(labels * log_softmax(logits , dim=1)) / len(labels)|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25

### E Additional Experiments

###### E.1 Improving Image Classification using Unpaired Texts (Unaligned encoders)

In this section we report image-classification results on ten benchmarks (see Section B.3), covering three settings:

- 1. Full-dataset fine-tuning: train both the vision backbone and classification head (Section E.1.1).
- 2. Full-dataset linear probe: train only the classification head (Section E.1.2).
- 3. Few-shot linear probe: train only the classification head under few-shot conditions (Section E.1.3).

In each setting, we compare Uml with baselines across all datasets and multiple DINO-initialized vision backbones. Our method has two variants: Ours (Uml), where we alternately train with both image and unpaired text data (see Algorithm 2), and Ours (init) where we initialize the classifier with the average text embedding of each class, providing a strong prior to align image and class level information.

###### E.1.1 Supervised Finetuning (across architectures)

In this section, we fine-tune both the vision backbone and the linear classifier on ten downstream tasks, comparing Uml against strong image-only baselines. We evaluate four DINO-initialized backbones:

- • ViT-B/16 in Table 6
- • ViT-B/8 in Table 7
- • DINOv2 ViT-S/14 in Table 8
- • DINOv2 ViT-B/14 in Table 9

Results for DINOv2 ViT-L/14 are omitted due to computational constraints. Across all backbones, Uml consistently improves over the image-only baseline by leveraging unpaired text embeddings. For some backbones such as DINOv2 VIT-B/16, our head-initialization variant (Ours (init)) outperforms training using unpaired multimodal data from scratch (Ours), while in others it does not.

- Table 6: Full finetuning on classification with ViT-B/16 DINO and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when fine-tuning on the target dataset. All vision encoders are initialized from DINO weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

Dataset

Method

StanfordCars

SUN397

FGVCAircraft

DTD

UCF101

Food101

OxfordPets

OxfordFlowers

Caltech101

Average

Unimodal 78.41 63.99 62.12 74.17 81.43 82.38 92.00 98.24 96.31 81.01 Ours 82.56 67.04 67.38 76.42 84.06 81.79 93.20 98.98 97.04 83.16 Ours (init) 81.95 67.12 68.29 73.84 84.31 81.12 92.60 98.73 96.84 82.76

- Table 7: Full finetuning on classification with ViT-B/8 DINO and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when fine-tuning on the target dataset. All vision encoders are initialized from DINO weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

Dataset

Method

StanfordCars

SUN397

FGVCAircraft

DTD

UCF101

Food101

OxfordPets

OxfordFlowers

Caltech101

Average

Unimodal 85.67 68.04 72.60 76.65 83.94 85.32 93.06 99.22 96.82 84.59 Ours 87.95 70.28 75.31 77.19 85.59 84.83 93.05 99.43 97.12 85.64 Ours (init) 87.44 70.03 76.09 76.24 86.49 84.71 93.81 99.27 97.16 85.69

- Table 8: Full finetuning on classification with ViT-S/14 DINOv2 and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when fine-tuning on the target dataset. All vision encoders are initialized from DINOv2 weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

###### Dataset

OxfordFlowers

FGVCAircraft

StanfordCars

OxfordPets

Caltech101

Food101

SUN397

Average

UCF101

DTD

Method

Unimodal 79.45 66.20 66.99 72.16 83.18 80.65 90.67 99.18 95.45 81.54 Ours 84.87 66.72 71.54 74.14 84.77 81.16 91.87 99.55 97.03 83.52 Ours (init) 86.39 66.03 73.44 74.27 84.69 81.97 91.72 99.82 97.60 83.99

- Table 9: Full finetuning on classification with ViT-B/14 DINOv2 and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when fine-tuning on the target dataset. All vision encoders are initialized from DINOv2 weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

###### Dataset

OxfordFlowers

FGVCAircraft

StanfordCars

OxfordPets

Caltech101

Food101

SUN397

Average

UCF101

DTD

Method

Unimodal 89.62 71.45 77.29 73.88 88.00 82.94 94.55 99.88 97.69 86.14 Ours 90.93 70.97 80.02 75.83 87.52 86.25 94.74 99.88 97.57 87.08 Ours (init) 90.73 70.92 80.23 75.87 87.60 83.43 94.47 99.80 97.93 86.77

###### E.1.2 Linear Probing (across architectures)

In this section, we train only the linear classifier, on top of the frozen vision and language backbone, on ten downstream tasks, comparing Uml against strong image-only baselines. We evaluate five DINO-initialized backbones:

- • ViT-B/16 in Table 10
- • ViT-B/8 in Table 11
- • DINOv2 ViT-S/14 in Table 12
- • DINOv2 ViT-B/14 in Table 13
- • DINOv2 ViT-L/14 in Table 14

Across all backbones, Uml consistently improves over the image-only baseline by leveraging unpaired text embeddings. For all backbones, our head-initialization variant (Ours (init)) outperforms training using unpaired multimodal data from scratch (Ours).

- Table 10: Full linear probing on classification with ViT-B/16 DINO and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when training a linear probe on the target dataset. All vision encoders are initialized from DINO weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

###### Dataset

OxfordFlowers

FGVCAircraft

StanfordCars

OxfordPets

Caltech101

Food101

SUN397

Average

UCF101

DTD

Method

Unimodal 67.10 64.63 56.02 72.42 81.27 74.96 93.07 98.32 95.01 78.08 Ours 68.71 65.14 57.42 72.95 82.06 75.30 93.18 98.46 96.19 78.82 Ours (init) 68.60 65.59 57.98 73.11 82.40 75.73 93.62 98.42 96.35 79.09

- Table 11: Full linear probing on classification with ViT-B/8 DINO and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when training a linear probe on the target dataset. All vision encoders are initialized from DINO weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

Dataset

Method

StanfordCars

SUN397

FGVCAircraft

DTD

UCF101

Food101

OxfordPets

OxfordFlowers

Caltech101

Average

Unimodal 72.01 67.19 62.02 76.18 82.95 78.57 91.99 98.78 96.23 80.66 Ours 72.93 68.17 63.49 77.13 83.16 79.87 92.59 98.50 96.47 81.37 Ours (init) 72.81 68.36 64.09 76.48 83.72 80.01 92.50 98.74 96.43 81.46

- Table 12: Full linear probing on classification with ViT-S/14 DINOv2 and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when training a linear probe on the target dataset. All vision encoders are initialized from DINOv2 weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

Dataset

Method

StanfordCars

SUN397

FGVCAircraft

DTD

UCF101

Food101

OxfordPets

OxfordFlowers

Caltech101

Average

Unimodal 77.48 70.72 66.28 78.25 82.64 84.39 94.29 99.62 97.00 83.40 Ours 78.45 71.53 67.33 78.70 83.51 84.67 94.70 99.82 97.11 83.98 Ours (init) 78.58 72.24 67.50 79.51 83.57 84.74 94.78 99.89 97.15 84.22

- Table 13: Full linear probing on classification with ViT-B/14 DINOv2 and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when training a linear probe on the target dataset. All vision encoders are initialized from DINOv2 weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

###### Dataset

OxfordFlowers

FGVCAircraft

StanfordCars

OxfordPets

Caltech101

Food101

SUN397

Average

UCF101

DTD

Method

Unimodal 85.46 75.42 72.34 79.73 87.26 88.70 95.56 99.76 97.81 86.89 Ours 85.40 75.22 75.22 80.73 87.21 89.02 95.83 99.88 97.85 87.37 Ours (init) 85.74 75.70 74.17 81.32 87.26 88.78 95.78 99.88 97.93 87.40

- Table 14: Full linear probing on classification with ViT-L/14 DINOv2 and OpenLLaMA-3B. We compare our proposed approach with the image-only baseline when training a linear probe on the target dataset. All vision encoders are initialized from DINOv2 weights, and our approach leverages unpaired text data using OpenLLaMA-3B embeddings.

###### Dataset

OxfordFlowers

FGVCAircraft

StanfordCars

OxfordPets

Caltech101

Food101

SUN397

Average

UCF101

DTD

Method

Unimodal 88.16 77.26 74.32 81.56 89.82 90.95 96.27 99.84 97.97 88.46 Ours 88.45 77.20 76.93 82.39 90.19 91.09 96.51 99.92 98.01 88.97 Ours (init) 87.99 77.75 77.20 82.51 90.17 91.29 96.32 99.92 97.93 89.01

###### E.1.3 Few-shot Linear Probing (across architectures)

In this section, we train only the linear classifier, on top of the frozen vision and language backbone, for few-shot classification on ten downstream tasks, comparing Uml against strong image-only baselines. We evaluate five DINO-initialized backbones: ViT-B/16 in Table 16, ViT-B/8 in Table 15, DINOv2 ViT-S/14 in Table 17, DINOv2 ViT-B/14 in Table 19, DINOv2 ViT-L/14 in Table 19. Across all backbones, Uml consistently improves over the image-only baseline by leveraging unpaired text embeddings. For all backbones, our head-initialization variant (Ours (init)) outperforms training using unpaired multimodal data from scratch (Ours).

- Table 15: Linear evaluation of frozen features on 11 fine-grained benchmarks for few-shot learning. We compare our proposed approach with the image-only baseline by training a linear classifier on top of frozen VIT-B/8 DINO features. Our method leverages unpaired text data using OpenLLaMA-3B

Dataset

OxfordFlowers

StanfordCars

FgvcAircraft

OxfordPets

Caltech101

Imagenet

Food101

Average

Sun397

Ucf101

Dtd

Train Shot Method

- 1 Unimodal 7.40 26.37 12.16 28.62 39.75 19.23 42.81 54.97 58.22 74.13 36.37 Ours 7.71 28.01 13.56 33.22 42.08 21.13 43.27 55.85 58.61 77.51 38.10 Ours (init) 9.24 34.23 14.49 36.27 47.55 24.81 46.75 60.09 61.59 80.23 41.52

- 2 Unimodal 14.43 37.96 20.28 39.80 53.03 30.62 54.75 68.12 77.59 81.91 47.85 Ours 15.71 40.74 21.04 43.74 55.86 33.52 54.49 69.86 77.18 84.52 49.67 Ours (init) 16.94 45.16 22.17 45.43 59.02 35.89 56.78 71.57 77.94 86.06 51.70

4 Unimodal 25.67 49.23 29.39 52.52 64.27 43.82 61.64 75.85 87.41 90.36 58.02 Ours 27.30 51.23 31.43 54.31 66.72 45.58 61.51 77.51 87.96 91.36 59.49 Ours (init) 28.54 53.68 31.31 56.13 67.47 47.40 62.84 79.10 88.29 91.98 60.67

8 Unimodal 41.04 56.86 40.03 61.15 72.39 54.47 66.10 82.30 93.95 92.28 66.06 Ours 43.76 58.14 42.56 63.12 73.13 56.30 66.36 84.27 94.25 92.71 67.46 Ours (init) 44.16 59.80 42.30 64.46 74.30 57.07 67.18 84.85 94.00 93.24 68.14

16 Unimodal 57.72 61.74 52.63 67.69 76.18 62.63 68.87 87.31 96.41 94.27 72.54 Ours 60.11 63.21 54.53 69.33 78.13 63.74 69.44 87.73 96.89 94.54 73.76 Ours (init) 60.36 64.26 54.81 70.27 78.76 64.13 70.05 88.23 96.63 94.73 74.22

- Table 16: Linear evaluation of frozen features on 11 fine-grained benchmarks for few-shot learning. We compare our proposed approach with the image-only baseline by training a linear classifier on top of frozen VIT-B/16 DINO features. Our method leverages unpaired text data using OpenLLaMA-3B

Dataset

Train Shot Method

StanfordCars

Sun397

FgvcAircraft

Dtd

Ucf101

Food101

Imagenet

OxfordPets

OxfordFlowers

Caltech101

Average

- 1 Unimodal 6.28 22.43 9.72 29.22 37.85 15.40 38.67 60.12 54.62 73.25 34.76 Ours 7.89 26.08 10.41 32.45 40.27 18.14 39.28 60.88 58.32 75.66 36.94 Ours (init) 8.96 31.34 12.12 34.22 44.32 21.46 42.68 66.39 60.37 79.74 40.16

- 2 Unimodal 12.64 35.64 14.98 38.93 51.14 26.05 50.34 70.84 75.61 83.16 45.93 Ours 14.38 38.62 17.00 40.37 54.28 29.24 50.83 72.88 77.14 85.95 48.07 Ours (init) 15.99 42.31 17.65 42.89 56.46 32.15 52.90 74.82 77.32 87.34 49.98

4 Unimodal 22.60 45.95 24.27 50.30 63.00 38.51 57.99 80.14 85.60 89.67 55.80 Ours 24.83 48.62 25.76 52.64 64.39 40.74 57.96 80.92 87.20 91.17 57.42 Ours (init) 25.83 51.01 26.35 55.06 65.86 42.69 59.32 82.23 87.83 91.99 58.82

8 Unimodal 37.68 52.94 33.67 59.18 70.62 49.48 62.97 85.26 92.83 93.17 63.78 Ours 39.31 55.31 35.56 60.48 71.88 50.46 63.08 86.25 93.23 93.47 64.90 Ours (init) 40.50 57.03 35.64 62.27 73.18 51.50 64.09 86.93 93.59 93.71 65.84

16 Unimodal 52.48 58.27 45.34 64.81 75.72 56.24 66.36 88.57 95.90 94.27 69.80 Ours 55.84 60.57 47.70 66.21 76.81 58.26 66.47 89.60 96.55 95.12 71.31 Ours (init) 55.82 61.73 48.14 67.02 77.39 58.76 67.08 90.53 96.62 94.98 71.81

- Table 17: Linear evaluation of frozen features on 11 fine-grained benchmarks for few-shot learning. We compare our proposed approach with the image-only baseline by training a linear classifier on top of frozen VIT-S/14 DINOv2 features. Our method leverages unpaired text data using OpenLLaMA-3B

Dataset

OxfordFlowers

StanfordCars

FgvcAircraft

OxfordPets

Caltech101

Imagenet

Food101

Average

Sun397

Ucf101

Dtd

Train Shot Method

- 1

Unimodal 13.18 34.15 14.09 36.60 46.74 35.18 36.48 63.51 89.62 76.66 44.62 Ours 14.95 37.25 14.88 38.93 49.18 37.91 38.35 68.92 91.42 84.04 47.58 Ours (init) 16.49 41.79 15.63 42.04 52.33 42.27 42.69 73.59 93.64 84.52 50.50

- 2

Unimodal 24.68 47.88 23.09 47.75 56.81 48.54 50.41 75.32 96.02 86.90 55.73 Ours 26.93 49.65 24.29 50.99 61.67 51.77 51.31 79.44 96.90 89.80 58.28 Ours (init) 28.65 53.15 24.78 53.25 63.86 54.44 54.21 81.41 97.63 90.55 60.19

Unimodal 38.76 57.51 32.10 59.69 67.75 60.79 58.73 83.89 98.59 93.48 65.12 Ours 41.69 58.87 33.38 61.58 69.60 62.69 59.69 86.27 98.84 94.56 66.71 Ours (init) 43.17 60.89 33.86 62.43 71.13 63.88 61.38 87.36 99.17 94.96 67.82

4

Unimodal 54.56 63.00 45.05 64.78 74.19 68.06 64.53 88.68 99.27 94.35 71.65 Ours 56.27 64.57 45.98 66.31 75.19 69.22 65.14 89.78 99.27 95.42 72.71 Ours (init) 57.91 65.82 47.40 67.81 75.99 69.71 66.40 90.29 99.54 95.84 73.67

8

Unimodal 67.96 67.35 55.89 71.36 77.92 73.24 68.14 90.73 99.63 96.43 76.22 Ours 69.42 68.50 58.54 72.24 78.69 73.80 68.70 91.87 99.72 96.63 77.80 Ours (init) 70.32 69.19 58.74 73.17 79.58 74.51 69.44 92.47 99.82 96.80 78.81

16

with DINOv2 ViT-B/14. We compare our proposed approach with the image-only baseline by training a linear classifier on top of frozen VIT-B/14 DINOv2 features. Our method leverages unpaired text data using OpenLLaMA-3B

Dataset

OxfordFlowers

StanfordCars

FgvcAircraft

OxfordPets

Caltech101

Imagenet

Food101

Average

Sun397

Ucf101

Dtd

Train Shot Method

- 1 Unimodal 22.42 43.03 15.79 38.85 58.57 48.71 52.26 76.47 97.12 83.64 53.69 Ours 23.10 45.12 16.22 42.69 61.05 51.30 52.45 78.14 98.08 87.68 55.58 Ours (init) 25.47 48.56 16.83 45.31 63.53 54.16 55.56 81.08 97.94 88.13 57.66

- 2 Unimodal 35.17 55.41 25.54 51.16 69.49 62.13 62.35 84.31 99.58 89.55 63.47 Ours 37.38 56.98 25.88 54.65 70.61 63.89 63.21 85.50 99.70 92.02 64.98 Ours (init) 38.78 59.81 26.00 55.61 71.38 66.54 65.06 86.49 99.62 92.79 66.21

4 Unimodal 51.40 63.68 34.25 61.25 76.32 71.60 68.86 89.05 99.76 94.51 71.07 Ours 54.26 64.65 35.52 62.63 76.87 72.33 69.14 90.00 99.70 95.51 72.06 Ours (init) 55.01 66.55 35.14 63.97 77.57 73.25 70.30 90.31 99.57 95.65 72.73

8 Unimodal 66.01 68.88 48.17 66.67 79.92 76.26 72.48 90.97 99.80 95.54 76.47 Ours 68.53 69.75 50.88 68.46 81.44 77.34 73.12 92.39 99.70 96.20 77.78 Ours (init) 67.91 70.66 51.26 69.56 81.85 77.95 73.75 92.50 99.68 96.51 78.16

16 Unimodal 77.31 72.17 62.38 73.76 83.80 80.74 75.15 93.34 99.81 97.40 81.59 Ours 78.92 72.80 64.51 75.16 84.62 81.00 75.46 92.92 99.59 97.38 82.24 Ours (init) 78.52 73.18 65.81 75.65 84.77 81.18 75.82 93.28 99.78 97.57 82.56

with DINOv2 ViT-L/14. We compare our proposed approach with the image-only baseline by training a linear classifier on top of frozen VIT-L/14 DINOv2 features. Our method leverages unpaired text data using OpenLLaMA-3B

Dataset

OxfordFlowers

StanfordCars

FgvcAircraft

OxfordPets

Caltech101

Imagenet

Food101

Average

Sun397

Ucf101

Dtd

Train Shot Method

- 1 Unimodal 24.89 48.36 17.69 38.77 66.46 59.27 57.50 79.83 98.13 82.96 57.39 Ours 25.88 49.63 18.08 42.93 69.18 60.12 58.37 83.51 98.42 86.23 59.24 Ours (init) 27.90 52.86 18.95 43.18 70.98 63.17 60.80 83.86 98.59 88.17 60.85

- 2 Unimodal 39.95 58.95 26.87 50.18 75.79 70.74 67.14 84.71 99.74 89.82 66.39 Ours 41.22 60.82 27.15 53.01 76.61 72.07 67.90 86.07 99.72 91.95 67.65 Ours (init) 42.93 63.36 28.14 54.96 77.72 73.87 69.20 87.13 99.81 91.71 68.88

4 Unimodal 56.49 66.37 38.59 59.08 80.84 77.39 72.41 89.90 99.73 94.44 73.52 Ours 58.19 67.36 39.57 61.78 81.36 78.19 72.82 90.99 99.76 95.27 74.53 Ours (init) 58.60 68.84 39.19 62.77 81.50 78.99 73.63 90.74 99.88 96.02 75.02

8 Unimodal 70.00 70.71 51.57 66.47 83.84 81.69 76.02 93.53 99.89 95.55 78.93 Ours 71.63 71.59 55.13 67.91 84.47 82.12 76.43 93.62 99.88 96.36 79.91 Ours (init) 72.02 72.51 55.49 69.03 84.57 82.52 76.78 93.80 99.89 96.73 80.33

16 Unimodal 80.84 73.83 64.13 73.96 87.43 84.58 77.78 94.69 99.91 97.36 83.45 Ours 81.85 74.39 69.45 74.70 87.35 84.58 78.35 94.59 99.89 97.61 84.28 Ours (init) 82.76 74.80 69.42 74.88 87.65 84.96 78.58 94.42 99.81 97.62 84.49

###### E.2 Improving Image Classification using Unpaired Texts (Aligned encoders)

###### E.2.1 Supervised Finetuning

In this section, we fine-tune both the vision backbone and the linear classifier on nine downstream tasks, comparing Uml against strong image-only baselines. We evaluate two different backbones: ResNet-50 and VIT-B/16.

- As shown in Table 20, across all backbones, Uml consistently improves over the image-only baseline

by leveraging unpaired text embeddings. Further, our head-initialization variant (Ours (init)) outperforms training using unpaired multimodal data from scratch (Ours).

E.2.2 Linear Probing

In this section, we train only the linear classifier, on top of the frozen vision and language backbone from CLIP, on ten downstream tasks, comparing Uml against strong image-only baselines.

- As shown in Table 21, Uml consistently improves over the image-only baseline by leveraging unpaired

text embeddings. Further, our head-initialization variant (Ours (init)) outperforms training using unpaired multimodal data from scratch (Ours).

- Table 20: Supervised finetuning on 9 fine-grained classification benchmarks with CLIP. We compare our proposed approach with the image-only baseline when fine-tuning on the target dataset. All vision encoders are initialized from CLIP ResNet50 weights, and our approach leverages unpaired text data using the corresponding CLIP text encoder.

Dataset

Method

StanfordCars

Sun397

FgvcAircraft

Dtd

Ucf101

Food101

OxfordPets

OxfordFlowers

Caltech101

Average

Unimodal 36.12 25.93 37.70 51.06 52.49 69.24 63.17 88.42 83.61 56.42 Ours 37.00 24.05 41.34 55.67 60.48 69.77 74.49 92.57 84.79 60.02 Ours (init) 72.75 62.33 66.58 56.50 67.54 76.95 86.97 94.80 87.95 74.71

- Table 21: Full linear probing on classification with CLIP ResNet-50 Image Encoder and Text encoder. We compare our proposed approach with the image-only baseline when training a linear probe on the target dataset. All vision encoders are initialized from ResNet-50 weights, and our approach leverages unpaired text data using the corresponding CLIP text embeddings.

###### Dataset

OxfordFlowers

FGVCAircraft

StanfordCars

OxfordPets

Caltech101

Food101

SUN397

Average

UCF101

DTD

Method

Unimodal 76.36 70.97 41.88 72.81 81.23 81.60 88.39 97.89 92.78 78.21 Ours 77.23 71.18 42.66 71.81 81.81 81.51 87.84 97.65 93.01 78.30 Ours (init) 79.14 73.83 42.81 73.76 82.13 82.44 90.90 97.69 94.19 79.65

- E.2.3 Few-shot linear Probing (across architectures)

In this section, we train only the linear classifier, on top of the frozen vision and language backbone from CLIP, for few-shot classification on ten downstream tasks, comparing Uml against strong image-only baselines. We evaluate two different backbones: ResNet-50 and VIT-B/16.

As shown in Table 22 and Table 23, across both backbones, Uml consistently improves over the image-only baseline by leveraging unpaired text embeddings. Further, our head-initialization variant (Ours (init)) outperforms training using unpaired multimodal data from scratch (Ours).

###### E.3 Improving Visual Robustness Using Unpaired Texts

In this section, we evaluate the robustness of models trained with Uml to test-time distribution shifts. We train a k-shot linear probe (where k ∈ {1,2,4,8}) with DINOv2 on ImageNet and evaluate across four distribution-shifted target datasets: ImageNet-V2, ImageNet-Sketch, ImageNet-A, and ImageNet-R. Our

- Table 22: Linear evaluation of frozen features on 10 fine-grained benchmarks for few-shot learning. We compare our proposed approach with the image-only baseline by training a linear classifier on top of frozen CLIP ResNet50 features. Our method leverages unpaired text data using the corresponding CLIP text encoder

Dataset

OxfordFlowers

StanfordCars

FgvcAircraft

OxfordPets

Caltech101

Imagenet

Food101

Average

Sun397

Ucf101

Dtd

Train Shot Method

- 1 Unimodal 23.24 29.14 12.38 30.24 37.55 27.26 34.61 21.36 59.07 66.52 34.14 Ours 36.32 45.40 16.84 40.92 53.19 49.76 53.03 36.48 68.56 76.80 47.73 Ours (init) 57.88 64.59 22.23 50.85 65.99 76.73 86.59 60.92 81.08 83.79 65.06

- 2 Unimodal 38.37 43.83 18.63 40.33 53.25 44.60 47.75 32.62 75.03 78.90 47.33 Ours 46.64 53.53 20.81 48.35 62.01 56.67 60.64 42.21 77.97 84.58 55.34 Ours (init) 61.86 65.90 24.19 55.30 70.39 77.07 87.40 61.40 86.20 85.94 67.57

4 Unimodal 51.34 54.38 23.08 52.07 64.06 57.29 61.32 41.72 86.16 85.41 57.68 Ours 55.21 59.48 24.77 56.78 67.65 62.68 67.31 47.04 86.46 87.23 61.46 Ours (init) 65.80 68.11 27.49 60.13 73.62 77.79 86.54 62.37 91.60 87.57 70.10

8 Unimodal 61.74 61.47 30.22 60.15 70.16 64.63 68.94 49.48 92.20 89.14 64.81 Ours 62.75 63.70 30.69 61.84 70.74 67.73 73.62 52.14 92.31 89.89 66.54 Ours (init) 69.78 69.61 31.62 64.13 77.24 78.58 89.07 63.34 94.21 91.58 72.92

16 Unimodal 70.94 65.53 35.91 64.30 75.13 70.67 78.49 55.07 95.21 91.26 70.25 Ours 71.58 67.08 36.23 65.62 76.09 71.63 79.52 56.92 95.44 91.94 71.20 Ours (init) 74.56 71.33 37.13 68.09 78.66 79.06 89.71 64.31 96.17 93.31 75.23

Distribution Shift Results

method consistently improves robustness over the unimodal baseline (Figure 12, Figure 13, Figure 14 and Figure 15) across different training shots, indicating that language priors help capture more transferable features.

- Shot-1
- Shot-2

- Figure 12: Robustness under test-time distribution shifts. Our approach (trained on 1-shot) is much more robust than its unimodal counterpart across four distribution-shuffled target test sets.

- Table 23: Linear evaluation of frozen features on 10 fine-grained benchmarks for few-shot learning. We compare our proposed approach with the image-only baseline by training a linear classifier on top of frozen CLIP VIT-B/16 features. Our method leverages unpaired text data using the corresponding CLIP text encoder

Dataset

OxfordFlowers

StanfordCars

FgvcAircraft

OxfordPets

Caltech101

Imagenet

Food101

Average

Sun397

Ucf101

Dtd

Train Shot Method

- 1 Unimodal 31.53 33.51 17.76 31.72 43.64 39.40 37.43 27.65 67.95 71.68 40.23 Ours 48.28 53.44 22.06 47.04 63.40 63.92 60.95 47.35 77.82 83.14 56.74 Ours (init) 67.76 70.13 32.26 55.16 75.02 84.25 90.91 69.50 87.58 88.87 72.14

- 2 Unimodal 48.45 48.70 23.38 42.04 60.08 58.30 53.56 41.68 82.01 83.20 54.14 Ours 57.89 59.95 27.19 52.27 69.60 71.18 66.78 54.24 87.43 90.20 63.67 Ours (init) 70.75 71.52 33.99 60.17 78.37 85.39 90.67 70.19 92.18 90.09 74.33

Distribution Shift Results

- Shot-1
- Shot-2

4 Unimodal 61.64 60.66 31.01 54.37 70.49 71.91 69.35 52.15 90.99 91.08 65.36 Ours 66.24 65.56 32.98 59.95 74.16 76.19 75.92 58.50 91.32 93.23 69.40 Ours (init) 74.58 73.54 37.38 64.30 81.10 86.05 91.64 70.89 94.80 93.70 76.80

8 Unimodal 71.76 66.67 38.47 61.96 77.11 78.16 78.25 59.90 95.20 92.98 72.05 Ours 72.77 69.50 39.09 64.89 79.01 80.07 80.85 62.63 94.98 94.36 73.82 Ours (init) 78.43 75.07 41.77 68.50 83.41 86.87 92.55 71.97 96.94 95.27 79.08

16 Unimodal 78.76 71.49 44.74 68.79 80.43 82.08 85.16 63.87 96.97 94.54 76.68 Ours 79.40 72.19 45.06 69.41 81.97 82.12 85.92 64.93 96.49 95.28 77.28 Ours (init) 82.38 76.51 47.14 72.13 84.66 86.60 92.68 72.79 97.70 96.08 80.87

- Figure 13: Robustness under test-time distribution shifts. Our approach (trained on 2-shots) is much more robust than its unimodal counterpart across four distribution-shuffled target test sets.

Distribution Shift Results Shot-4

- Figure 14: Robustness under test-time distribution shifts. Our approach (trained on 4-shots) is much more robust than its unimodal counterpart across four distribution-shuffled target test sets.

Distribution Shift Results Shot-4

Shot-8

- Figure 15: Robustness under test-time distribution shifts. Our approach (trained on 8-shots) is much more robust than its unimodal counterpart across four distribution-shuffled target test sets.

Shot-8

| | |
|---|---|
| | |
| | |

###### E.4 Marginal Rate-of-Substitution Between Modalities

How many words is an image worth? In this section, we extend our results to evaluate image-text conversion ratios using test accuracy isolines on the remaining eight datasets. We measure these global equivalence ratios by fitting a plane to the accuracy values given the number of image and text shots. Figures 16 to 23 demonstrate the conversion ratios for DINOv2 VIT-S/14 as the vision backbone and OpenLLaMa-3B as the text backbone (unaligned encoders). Analogously, Figures 24 to 31 show the same ratios for CLIP ResNet-50 as the vision and text encoders (aligned encoders). As expected, with the fully aligned CLIP backbone, each image equates to far fewer text prompts than under the unaligned DINO setting, showing the higher efficiency of aligned embeddings.

###### E.4.1 Unaligned Encoders

[Figure 41]

[Figure 42]

Figure 16: SUN397. 1 img ≈ 1568 words Figure 17: Caltech101. 1 img ≈ 1248 words

[Figure 43]

[Figure 44]

Figure 18: Stanford Cars. 1 img ≈ 1799 words Figure 19: DTD. 1 img ≈ 2309 words

Figure 20: FGVC Aircraft. 1 img ≈ 3220 words Figure 21: Oxford Flowers. 1 img ≈ 1895 words

[Figure 47]

[Figure 48]

Figure 22: Food101. 1 img ≈ 2608 words Figure 23: UCF101. 1 img ≈ 2617 words

###### E.4.2 Aligned Encoders (CLIP)

[Figure 49]

[Figure 50]

Figure 24: SUN397. 1 img ≈ 221 words Figure 25: Caltech101. 1 img ≈ 256 words

Figure 26: Stanford Cars. 1 img ≈ 649 words Figure 27: DTD. 1 img ≈ 228 words

[Figure 53]

[Figure 54]

Figure 28: FGVC Aircraft. 1 img ≈ 691 words Figure 29: Oxford Flowers. 1 img ≈ 851 words

[Figure 55]

[Figure 56]

Figure 30: Food101. 1 img ≈ 202 words Figure 31: UCF101. 1 img ≈ 393 words

###### E.5 Impact of Scaling Vision Backbone

In this section, we study how our method’s performance scales with the size and architecture of the vision backbone. In addition to ViT-S/14 DINOv2, we extend our analysis to a range of ViT-based architectures, including ViT-B/14 and ViT-L/14 DINOv2 and ViT-B/16 and ViT-B/8 DINO models. To ensure a fair comparison, we follow the same training protocol as in previous experiments. Our method consistently

outperforms the unimodal baselines in every setting. In few-shot linear probing across ViT-B/8, ViT-B/16, DINOv2-ViTs and ViT-L/14 backbones (Tables 15 to 19), we see clear gains. The same holds for full-dataset end-to-end fine-tuning of both encoder and head (Tables 6 to 9), and even when only the linear classifier is trained on the full splits (Tables 10 to 14).

###### E.6 Impact of Varying Text Encoders

In this section, we study how our method’s performance varies with different language models used for generating text embeddings. Through this experiment, we aim to understand how differences in embedding quality and model capacity affect the integration of textual information in our multimodal setup. Specifically, we cover LLMs with diverse architectures and scales, including BERT-Large, RoBERTaLarge and GPT-2 Large. As shown in Figure 32, adding unpaired text embeddings shows a significant boost in 1-shot accuracy and still decent gains at 16 shots on SUN397 dataset. Overall, OpenLLaMA-3B outperforms all other language models.

70

65

60

TestAccuracy(%)

55

50

Unimodal

45

+BERT +GPT2 +RoBERTa

40

35

+OpenLLaMA-3B

1 2 4 8 16 Training shots

- Figure 32: Few-shot classification accuracy on SUN397 using Uml with unpaired, frozen embeddings from various pretrained language models.

###### E.7 Learning with Coarse-Grained vs. Fine-Grained Textual Cues

Understanding the type of information extracted from textual cues is crucial to assessing the effectiveness of our multimodal approach. A key question is whether the model merely utilizes class names or goes beyond to capture richer, more descriptive features. To investigate this, we compare the performance of our method using two types of text templates: a vanilla template that consists solely of the class name (e.g., ”a photo of a [class]”) and descriptive templates generated from GPT-3, as detailed in Section Section B. As shown in Figure 33 and Figure 34, both multimodal approaches consistently outperform the unimodal baseline, with descriptions from GPT-3 offering a more substantial performance gain. This shows that leveraging richer, contextually diverse text cues can significantly enhance model performance, even in low-shot learning scenarios.

70

65

60

TestAccuracy(%)

55

50

45

40

Unimodal

+ Vanilla Descriptions

35

+ GPT3-Descriptions

1 2 4 8 16 Training shots

70

65

60

TestAccuracy(%)

55

50

45

40

Unimodal

+ Vanilla Descriptions

35

+ GPT3-Descriptions

1 2 4 8 16 Training shots

- Figure 33: Few-shot SUN397 accuracy with Uml using two levels of textual granularity: (a) vanilla class descriptions and (b) GPT-3–generated fine-grained descriptions.

Figure 34: Few-shot SUN397 accuracy with Uml (init) using two levels of textual granularity: (a) vanilla class descriptions and (b) GPT-3–generated fine-grained descriptions.

###### E.8 Impact on Performance with Increasing Unpaired Text Prompts

Here, we investigate how classification accuracy evolves as we augment each image with an increasing number of unpaired text prompts . Figure 35 shows these accuracy curves as we vary the number of unpaired text prompts per image shot across five image-shot budgets. In every regime, our multimodal initialization (“Ours (init)”) outperforms training the head from scratch, with most of the gain coming from the first few prompts and gains tapering off thereafter. Note that we do not enforce diversity or novelty in the unpaired text prompts—simply adding more sentences does not guarantee additional information.

Image shots: 1

Image shots: 2

Image shots: 4

- 58

- 59

- 60

- 61

42

TestAccuracy(%)

TestAccuracy(%)

TestAccuracy(%)

52

40

38

50

36

Ours

Ours

Ours

Ours (init)

Ours (init)

Ours (init)

48

34

0 1 2 5 7 10 12 15 Text shots

0 1 2 5 7 10 12 15 Text shots

0 1 2 5 7 10 12 15 Text shots

Image shots: 8

Image shots: 16

- 63

- 64

- 65

- 66

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | |O O|urs urs|(in|it)| |
| | | | | | | | | |
| | | | | | | | | |

69.0

TestAccuracy(%)

TestAccuracy(%)

68.5

68.0

Ours

67.5

Ours (init)

0 1 2 5 7 10 12 15 Text shots

0 1 2 5 7 10 12 15 Text shots

- Figure 35: Classification accuracy as a function of the number of text prompts per image shot for the SUN397 Dataset.

###### E.9 Effect of Unrelated Auxiliary Modalities

While our main experiments investigate the benefits of incorporating semantically related auxiliary modalities, an important question is what happens when the auxiliary data is unrelated or even adversarial. In our unpaired setting, this requires reasoning not about individual mismatched pairs, but about the relationship between entire data distributions. To meaningfully study the adversarial case, one would need a principled definition of “negative correlation” between modality distributions, as well as metrics for quantifying it. We leave such a rigorous framework to future work.

Instead, here, we evaluate a simpler but informative scenario: the independent case, where the auxiliary modality is semantically unrelated to the target modality. We use our few-shot learning setup to train Uml with image data from SUN397 and text data from Stanford-Cars, two semantically unrelated datasets.

- Table 24 reports results for few-shot image classification on the SUN397 dataset. When the auxiliary text is unrelated to the image domain, performance does not improve over the unimodal baseline. In contrast, semantically related text provides consistent gains across all shot counts.

- Table 24: Training Uml with unrelated auxiliary text from the Stanford Cars dataset does not yield performance gains on image classification on the SUN397 dataset. However, semantically correlated text (from SUN397) consistently improves accuracy across few-shot settings.

Method 1-shot 2-shot 4-shot 8-shot 16-shot

Unimodal (Image) 34.15 47.88 57.51 63.00 67.35 Uml (Image + Unrelated Text) 35.27 47.12 57.50 62.45 67.25 Uml (Image + Related Text) 41.79 53.15 60.89 65.82 69.19

###### E.10 Extension to More Than Two Modalities

Our framework naturally extends beyond two modalities. We validate this by extending our image and audio classification experiments on ImageNet-ESC to use all three modalities: image, audio, and text. Training alternates batches from each modality while applying modality-specific classification losses, consistent with our two-modality setup.

Tables 25 and 26 summarize results on audio and image classification. In both cases, incorporating additional unpaired modalities consistently improves performance over unimodal or pairwise settings. These findings demonstrate that the performance benefits of Uml extend robustly to more than two modalities.

From a theoretical perspective, our results also generalize directly. Since modality-specific observations are conditionally independent given the ground-truth latent Z∗, their joint contribution to the Fisher information reduces to the sum of the unimodal blocks. Consequently, the total contribution of all auxiliary modalities (excluding the primary modality X) can be obtained by summing their individual Fisher information matrices.

- Table 25: Training Uml with all three modalities from ImageNet-ESC outperforms unimodal or pairwise training on audio classification. Numbers in parentheses denote relative improvements over the unimodal baseline.

Dataset Method 1-shot 2-shot 4-shot

ESC-19 Audio-Only 28.78 39.85 52.22 Audio + Image 34.59 44.13 50.00 Audio + Text 35.47 52.19 52.90 Audio + Image + Text 44.46 (+54.4%) 51.48 (+29.2%) 56.57 (+8.3%)

ESC-27 Audio-Only 25.65 35.99 44.79 Audio + Image 37.15 42.86 51.15 Audio + Text 41.97 47.02 53.85 Audio + Image + Text 44.68 (+74.2%) 48.03 (+33.4%) 54.16 (+20.9%)

- Table 26: Training Uml with all three modalities from ImageNet-ESC outperforms unimodal or pairwise training on image classification. Numbers in parentheses denote relative improvements over the unimodal baseline.

Dataset Method 1-shot 2-shot 4-shot

ESC-19 Image-Only 60.28 74.10 78.70 Image + Audio 64.63 76.17 85.02 Image + Text 88.84 89.71 92.07 Image + Audio + Text 90.55 (+50.2%) 91.08 (+22.9%) 91.72 (+16.5%)

ESC-27 Image-Only 55.33 65.60 77.85 Image + Audio 59.75 70.93 78.14 Image + Text 86.39 88.91 90.14 Image + Audio + Text 88.22 (+59.5%) 88.96 (+35.6%) 91.78 (+17.9%)

###### E.11 Effect of Ratio of Modality Batches

In our main experiments, Uml was trained with a simple 1:1 alternation of batches across modalities. To study the effect of this schedule, we ablate the ratio of text to image batches, denoted by r, on SUN397

using ViT-S/14 DINOv2 (vision) and OpenLLaMA-3B (text). We evaluate both in the (a) linear probe setting and the (b) full finetuning setting.

Across both settings and for both Uml and Uml (init), we observe that the choice of r has little impact on performance. The gains primarily arise from the presence of auxiliary information rather than the exact frequency of its appearance during training.

- Table 27: Few-shot linear probing with different ratios of text-to-image batches on SUN397 using ViT-S/14 DINOv2 and OpenLLaMA-3B. Performance remains stable across ratios r, indicating robustness of Uml to batch scheduling.

Method Shot r = 0.25 r = 0.5 r = 1.0 r = 2.0 r = 4.0

Uml 2-shot 49.03 49.23 49.65 49.56 49.97 4-shot 59.05 59.06 58.87 58.92 58.70 8-shot 64.63 65.24 64.57 64.98 65.29

- 16-shot 68.55 68.79 68.50 68.86 68.36

Uml (init) 1-shot 42.60 42.13 41.79 41.95 42.04 2-shot 52.83 53.01 53.15 53.04 52.81 4-shot 61.10 61.09 60.89 60.63 60.99 8-shot 65.29 65.32 65.82 65.15 65.21

- 16-shot 69.19 68.91 69.19 68.51 68.71

- Table 28: Full finetuning of the vision encoder and linear head with different text-to-image batch ratios r on SUN397. Results show that performance is largely insensitive to r.

Method r = 0.25 r = 0.5 r = 1.0 r = 2.0 r = 4.0

Uml 66.44 66.80 66.72 67.60 66.44 Uml (init) 66.58 65.41 66.03 65.86 64.25

E.12 Effect of Freezing vs. Unfreezing the Text Encoder

In all our main experiments, we freeze the text encoder. This design choice allows us to isolate the role of the auxiliary modality, ensuring that improvements in the primary modality (e.g., vision) arise from cross-modal transfer rather than joint training of both encoders.

In principle, however, one can also unfreeze the text encoder and update it during training. To study this, we ablate freezing versus unfreezing on Stanford Cars and SUN397 using ViT-S/14 DINOv2 (vision) and OpenLLaMA-3B (text). As shown in Table 29, unfreezing the text encoder improves performance on Stanford Cars, but slightly reduces performance on SUN397—likely due to the larger number of trainable parameters and increased optimization complexity.

- Table 29: Effect of freezing versus unfreezing the text encoder when training with Uml. Freezing stabilizes training and often yields slightly stronger gains.

Method Stanford Cars SUN397

Unimodal (Image Only) 79.45 66.20 Uml (Unfrozen Text Encoder) 84.23 65.80 Uml (Frozen Text Encoder) 84.87 66.72

###### E.13 Additional Experiments for Audio-Visual Setting

In this section, we extend our unpaired multimodal framework to the tri-modal ImageNet–ESC benchmark, examining how unpaired audio and text signals can enhance image classification under both aligned (Section E.13.2) and unaligned encoders(Section E.13.1). We then reverse the setting, showing that unpaired visual and textual context likewise improves audio classification (Section E.13.3).

Image Benchmarks (Appendix)

###### E.13.1 Improving Image Classification with Unpaired Audio and Text (Unaligned encoders)

Image (DINO)

(a) ImageNet-ESC-19 (b) ImageNet-ESC-27

Figure 36: Uml improves image classification using unpaired audio and text samples on both ImageNetESC-19 and ImageNet-ESC-27 benchmarks when trained on top of DINOv2 VIT-S/14 and OpenLLaMa-3B.

Image Benchmarks (Appendix)

###### E.13.2 Improving Image Classification with Unpaired Audio and Text (Aligned encoders)

Image (RN50 CLIP)

(a) ImageNet-ESC-19 (b) ImageNet-ESC-27

Figure 37: Uml improves image classification using unpaired audio and text samples on both ImageNetESC-19 and ImageNet-ESC-27 benchmarks when trained on top of CLIP ResNet-50 image and text encoders

###### E.13.3 Improving Audio Classification with Unpaired Image and Text (Aligned encoders)

###### Audio Benchmarks (Appendix)

###### Audio (RN50 CLIP)

(a) ImageNet-ESC-19 (b) ImageNet-ESC-27

- Figure 38: Uml improves audio classification using unpaired image and text samples on both ImageNetESC-19 and ImageNet-ESC-27 benchmarks when trained on top of CLIP ResNet-50 image and text encoders

###### E.14 Gaussian Experiments

Here, we shift our attention to a more nuanced and intriguing question: can incorporating unpaired multimodal data actually improve the reconstruction quality of a single modality? At first glance, this seems unlikely—why would adding data from a different modality make X reconstruction better than training with X? Moreover, we push this question further: can incorporating data from a different modality, while keeping the total dataset size fixed, still improve the reconstruction of X compared to using the same number of samples X dataset alone? This setup isolates the importance of multimodal information from mere data scaling, and surprisingly, our experiments show that this improvement is indeed possible.

To investigate this, we design a synthetic experiment inspired by our theoretical framework in Section 3.1. We generate data from two partially overlapping modalities, X and Y, derived from a shared latent space θc, while also containing unique components (θx and θy). The observations follow the same linear structure as in our theory:

Xi = Ac,iθc + Ax,iθx + ϵX,i Yj = Bc,jθc + By,jθy + ϵY,j

The training data are generated from Gaussian latents with dimensions dim(θc) = 10, dim(θx) = 5, and dim(θy) = 5. For X, only the first 10% of the shared components in θc are retained at full strength, while the rest are downscaled by 0.05; Y observes all shared components at full strength. This asymmetry makes Y informative about structure that is only weakly present in X. The validation set is constructed from the same projections but without attenuation, so both modalities fully observe θc. Observations are 50-dimensional with Gaussian noise ϵX, ϵY ∼ N (0,0.09I). In the unimodal setting, training on X alone uses 10,000 samples from X. When training Uml on unpaired X and Y, we instead use 5,000 samples from each modality to keep the total sample budget fixed, ensuring a fair comparison.

Our architecture is a shared autoencoder. Each modality X ∈ R50, Y ∈ R50 is projected into a common space of dimension 128 through modality-specific linear layers. A shared encoder (two linear layers with ReLU) maps into a latent space of dimension 10, followed by a shared decoder (two linear layers) that expands back to dimension 128. Finally, modality-specific heads reconstruct the original inputs. The shared pathway enforces cross-modal alignment, while the separate adapters preserve modality fidelity.

As shown in Figure 39, the surprising outcome is that training on both modalities, even when they are unpaired, consistently improves the reconstruction of X compared to training solely on X. More strikingly, this improvement holds even when the total number of training samples is fixed, with half the data coming from X and half from Y; showing that the model is not just benefiting from increased data quantity but from the diversity and complementary information provided by the second modality.

3.5

ReconstructionErroronX

Unpaired (X + Y)-data

X-data only

3.0

2.5

2.0

1.5

1.0

0 200 400 Training Steps

- Figure 39: Training on N/2 samples from X and N/2 unpaired samples from Y improves test reconstruction on X, more than training on N samples from X.

F Analysis of the Learned Predictor

- F.1 Multimodal Neurons

- Figure 40 tracks how cross-modal correlations between vision and text embeddings evolve during training for the top-performing neurons in our sarcasm detection model. Each subplot shows a single neuron’s correlation values across 30 training epochs, separately measured across all samples (green), correlations conditioned on label=1 (sarcastic; blue), and correlations conditioned on label=0 (non-sarcastic; pink). The neurons are ranked by their final overall correlation magnitude, revealing both positive correlation neurons (#50, #144, #46, #140, #124, #16) and negative correlation neurons (#136, #89, #114) that encode the same semantic pattern with opposite signs.

The most striking observation across neurons is that sarcasm consistently appears as cross-modal discord. Non-sarcastic samples exhibit strong correlations (around 0.6–0.8) as training progresses as opposed to sarcastic samples, which exhibit weaker correlations. Across all neurons, the same trend holds: non-sarcastic correlations are highest, overall correlations fall in the middle, and sarcastic correlations remain lowest. This pattern shows that the model may have learnt to recognize sarcasm not by what is said or shown alone, but by detecting when the two disagree or agree.

Neuron #50

Neuron #144

Neuron #46

Cross-ModalCorrelation

Cross-ModalCorrelation

Cross-ModalCorrelation

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

Overall

Overall

Overall

Sarcastic

Sarcastic

Sarcastic

Non Sarcastic

Non Sarcastic

Non Sarcastic

0.0

0.0

0.2

0 5 10 15 20 25 Training epochs

0 5 10 15 20 25 Training epochs

0 5 10 15 20 25 Training epochs

Neuron #140

Neuron #124

Neuron #16

Cross-ModalCorrelation

Cross-ModalCorrelation

Cross-ModalCorrelation

0.6

0.6

0.5

0.4

0.4

0.0

0.2

0.2

Overall

Overall

Overall

0.0

Sarcastic

Sarcastic

Sarcastic

0.0

Non Sarcastic

Non Sarcastic

Non Sarcastic

0.5

0.2

0 5 10 15 20 25 Training epochs

0 5 10 15 20 25 Training epochs

0 5 10 15 20 25 Training epochs

Neuron #136

Neuron #89

Neuron #114

0.0

Cross-ModalCorrelation

Cross-ModalCorrelation

Cross-ModalCorrelation

0.0

0.2

0.2

0.2

0.4

0.4

0.4

Overall

Overall

Overall

0.6

0.6

0.6

Sarcastic

Sarcastic

Sarcastic

Non Sarcastic

Non Sarcastic

Non Sarcastic

0.8

0.8

0 5 10 15 20 25 Training epochs

0 5 10 15 20 25 Training epochs

0 5 10 15 20 25 Training epochs

- Figure 40: Evolution of cross-modal correlation during unpaired co-training. Each curve shows the correlation between visual and textual activations over training epochs for sarcastic (label=1) and nonsarcastic (label=0) samples. Most neurons develop positive correlations for non-sarcastic content, while others exhibit negative correlations, suggesting functional specialization for multimodal alignment and the detection of incongruence.

###### F.2 Change in Decision Boundaries with Unpaired Data from Another Modality

Our decision boundary visualizations are constructed by projecting the high-dimensional embedding space of a given classifier to a 2D plane. Axis 1 is computed as the normalized difference between the classifier weights of the two selected classes, representing the primary decision direction. Axis 2 is chosen to be orthogonal to Axis 1, constructed from the difference between the class mean embeddings after removing the component parallel to Axis 1. This orthogonalization ensures that the two axes capture complementary aspects: Axis 1 reflects the primary model decision boundary, while Axis 2 captures the variation orthogonal to that decision. The final 2D projection matrix combines these two vectors as columns, and embedding vectors are then mapped to this plane using a simple dot product. Figure 41 , Figure 42 and Figure 43 show the change in decision boundary when adding unpaired textual information for 2-shot classification on top of frozen CLIP ResNet-50 features for Oxford Pets, DTD and Oxford Flowers datasets.

| |
|---|

| |
|---|

Train Image Test Image Russian Blue Abyssinian Decision Boundary

[Figure 57]

|[Figure 58]|
|---|

|[Figure 59]|
|---|

A pet Russian Blue typically has a blue-gray coat, green or blue eyes, and a muscular body.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

A pet Abyssinian looks like a small, brown and black spotted cat with large ears.

[Figure 65]

[Figure 66]

(a) Unimodal (b) Ours

- Figure 41: Impact of unpaired text on decision boundaries (CLIP ResNet50). (Left) Visual features alone learn ambiguous class boundaries between Russian Blue and Abyssinian cats. (Right) Adding unpaired text sharpens the boundary, leveraging semantic cues to better distinguish similar categories.

(a) Unimodal (b) Ours

| |
|---|

| |
|---|

Train Image Test Image Knitted Cobwebbed Decision Boundary

[Figure 67]

[Figure 68]

Additional Plots: DTD

[Figure 69]

[Figure 70]

|[Figure 71]|
|---|

[Figure 72]

[Figure 73]

|[Figure 74]|
|---|

[Figure 75]

[Figure 76]

Cobwebbed texture looks like a spiderweb.

The knitted texture looks like a series of interconnected loops.

- Figure 42: Impact of unpaired text on decision boundaries (CLIP ResNet50). (Left) Visual features alone learn ambiguous class boundaries between knitted and cobwebbed. (Right) Adding unpaired text sharpens the boundary, leveraging semantic cues to better distinguish similar categories

###### F.3 What do models learn from unpaired data?

To understand what the model is truly learning and how its weights evolve, we develop and analyze three key metrics: functional margin, silhouette score, and class-prototype vectors. These metrics inform on how well the model distinguishes between classes and how text information influences the structure of feature-space

Functional margin. This quantifies how confidently a model separates a given sample from the decision boundary. For a sample i belonging to class y, we calculate the margin relative to the next highest competing class. Specifically, we identify the second-highest logit among the incorrect classes, denoted as class j∗, and compute the functional margin as

wyTxi − wTj∗xi ∥wy − wj∗∥2

(5)

γi =

where wyTxi represents the logit for the true class, while wTj∗xi represents the highest logit among the competing classes. Larger margins indicate more confident and robust classification, while smaller margins

###### Additional Plots: Flowers

| |
|---|

| |
|---|

Train Image Test Image Ball Moss Passion Flower Decision Boundary

[Figure 77]

[Figure 78]

The flower passion flower looks like a purple and white flower.

|[Figure 79]|
|---|

|[Figure 80]|
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Ball moss is a small greenishbrown plant

(a) Unimodal (b) Ours

- Figure 43: Impact of unpaired text on decision boundaries (CLIP ResNet50). (Left) Visual features alone learn ambiguous class boundaries between ball moss and passion flower. (Right) Adding unpaired text sharpens the boundary, leveraging semantic cues to better distinguish similar categories

imply that the sample lies closer to a misclassification boundary. As shown in Figure 44, both Ours and Ours (init) exhibit substantially larger classification margins than the unimodal baseline, demonstrating that augmenting primary-modality training with unpaired multimodal data improves confidence in predictions over the primary modality.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

|m p|atrix, roduct|repres s betw|ent een|
|---|---|---|---|
|w orm|ell tex ation.|t featu Figur|res e 47|
|ali|gns clo|sely w|ith|

- Figure 44: Functional margin of the linear head trained on SUN397 dataset for few-shot classification significantly increases when training with both Uml and Uml with linear head initialization.

Silhouette Score and DB-Index. The Silhouette Score indicates how well-separated the clusters are, while the DB-Index measures intra-class compactness versus inter-class separation. Higher silhouette and lower DB-Index values mean better-defined clusters, indicating that text helps tighten intra-class spread and widen inter-class gaps. As shown in Figure 45 and Figure 46, both Ours and Ours (init) exhibit reduced intra-class distances and increased inter-class separations, further confirming improved class separability.

Class-Prototype Vectors. These vectors are the rows of the final linear layer’s weight ing the class centroids in the shared embedding space. We compute a heatmap of inner class prototypes and average text embeddings of the corresponding class to assess how align with class centers. This helps reveal how the model organizes multimodal inf shows a pronounced diagonal structure, indicating that each class’s text embedding the learned weights of the model.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

###### Figure 45: Silhouette Score of the linear head trained on SUN397 dataset for few-shot classification significantly increases when training with both Uml and Uml with linear head initialization.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

###### Figure 46: DB-Index of the linear head trained on SUN397 dataset for few-shot classification significantly improves when training with both Uml and Uml with linear head initialization.

[Figure 87]

###### Figure 47: Inner products between each linear-head weight vector and its class’s mean text embedding, demonstrating that text features align well with class prototypes.

