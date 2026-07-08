arXiv:2507.03683v1[cs.CV]4Jul2025

# On the rankability of visual embeddings

Ankit Sonthalia1∗ Arnas Uselis1 Seong Joon Oh1 1Tübingen AI Center, Universität Tübingen, Germany

## Abstract

We study whether visual embedding models capture continuous, ordinal attributes along linear directions, which we term rank axes. We define a model as rankable for an attribute if projecting embeddings onto such an axis preserves the attribute’s order. Across 7 popular encoders and 9 datasets with attributes like age, crowd count, head pose, aesthetics, and recency, we find that many embeddings are inherently rankable. Surprisingly, a small number of samples, or even just two extreme examples, often suffice to recover meaningful rank axes, without full-scale supervision. These findings open up new use cases for image ranking in vector databases and motivate further study into the structure and learning of rankable embeddings. Our code is available at https://github.com/aktsonthalia/rankablevision-embeddings.

## 1 Introduction

Visual embeddings are widely used for image retrieval. This relies on embeddings forming a metric space, where similar images are placed nearby. Modern visual encoders generally satisfy this property, and many systems depend on it in the form of vector databases.

Ranking is another core operation in databases. It allows users to navigate large collections by sorting, paginating, and filtering results. For instance, e-commerce platforms like Amazon benefit from ranking product images by visual quality or certain product-specific attributes (eg. ranking shoes by how formal they look).

In this work, we ask: are visual embeddings also rankable? Retrieval relies on local similarity around a query. Ranking requires a global ordering along an attribute. Prior work has largely addressed the former. The global rankability of embeddings remains underexplored.

High

Rankaxis

Local similarity

Query

Global ordering

We define rankability as follows: given an embedding function f and a continuous attribute A (e.g. “age”), we say f is rankable with respect to A if there exists a rank axis vA such that the projection vA⊤f(x) preserves the correct order of the target attribute A(x) over a dataset. For instance, if A denotes “age”, this projection would sort face images from youngest to oldest.

Low

Retrieval Ranking

We examine two questions: (1) Are visual embeddings rankable? (2) How easily can we recover the rank axis for a given attribute?

- To address (1), we evaluate 7 modern visual encoders, from ResNet to CLIP, across 9 datasets with 7 attributes: age, crowd count, 3 head pose angles (pitch, roll, yaw), image aesthetics, and recency. We find that many embedding spaces are indeed rankable (Section 3).

∗Corresponding author: ankit.sonthalia@uni-tuebingen.de

Preprint.

- To address (2), we estimate the rank axis vA with minimal supervision. The structure of the embedding space makes full-dataset regression unnecessary. In many cases, a handful of annotated samples and, in some cases, a pair of samples xl (low) and xh (high) already recover non-trivial ranking performance. For the latter case, we define the rank axis as vA = (f(xh)−f(xl))/∥f(xh)−f(xl)∥2. This opens up the possibility for fast ordering of new images by arbitrary attributes. For example, a photo app lets users sort selfies by age appearance. It uses CLIP embeddings and two reference

images: one of a child and one of an elderly person. The app computes vage without training. Users scroll from youngest-looking to oldest-looking faces in their album (Section 4).

Contributions:

- 1. We define and motivate rankability as a property of visual embeddings, distinct from retrieval.
- 2. We study rankability across modern encoders and real-valued attributes; results show that current embeddings are rankable.
- 3. We show that rank axes can sometimes be recovered using only two or a handful of labelled samples.

## 2 Related work

Embeddings for retrieval. Visual encoders are commonly used to index images in vector databases, enabling nearest neighbour search for retrieval tasks [38, 45, 31, 49]. This setup, known as deep metric learning [5, 6, 38], predates vision-language models like CLIP [45]. CLIP and related models shifted the focus to cross-modal similarity modelling, where vision and language share a joint embedding space used for classification [45], retrieval [63, 2], and captioning [36, 26]. While the majority of work in visual encoders is devoted to the understanding of the local similarity structure, we study how visual embeddings support global ranking instead of just local retrieval.

Improving embedding geometry and structure. Prior work has explored ways to improve the geometry of the embedding space. Order embeddings and hyperbolic representations have been used to model hierarchies [55, 21, 8, 44]. Training disentangled representation [60] is considered critical for compositionality, where attributes are assigned to certain linear subspaces [51, 3]. Others have defined concepts like uniformity and separability of the representations [59]. In this work, we focus on the analysis of a wide range of visual encoders, rather than introducing recipes for improvements.

Analysing embedding geometry and structure. A large body of work has examined the geometry and structure of CLIP’s learned embedding space. CLIP and its derivatives have been studied extensively [4, 46, 30, 65]. Several works have reported modality gaps between vision and language embeddings [12]. Some studies point to the absence of certain structures and capabilities in CLIP representations: attribute-object bindings [29, 68, 25], or the association of attributes to corresponding instances. Others argue that much information is already present in CLIP representations, including parts-of-speech and linguistic structure [39], attribute-object bindings [24], and compositional attributes [53]. The platonic representation hypothesis further suggests that models converge to similar internal structures [19]. In this work, we analyse the embedding geometry and structure for modern visual embeddings from the novel perspective of rankability.

Linearly probing an embedding. Linear probing is a fast and widely used method to test for the presence of concepts in visual embeddings [23, 18, 54]. It measures the accuracy of a linear classifier trained on intermediate-layer features, effectively testing whether a hyperplane can separate embeddings containing a concept from those that do not. This technique has been used to study the geometry of CLIP’s embeddings [28] and to probe for specific information such as attribute-object bindings and compositionality [24, 53]. While effective for binary separability, linear probes are limited in capturing non-binary, ordinal, or relational structures [1]. Prior work has not directly analysed how continuous attributes are laid out in the embedding space. Our work extends this line of research by moving beyond concept detection to characterise the internal structure of embeddings along ordinal axes, introducing rankability as a new property not captured by prior probing methods.

Ordinal information in embeddings. Early works on relative and ordinal attributes explored how continuous visual attributes could be inferred and ranked [43, 34, 71, 14, 64]. However, these efforts were limited to smaller models and datasets. Recent studies have begun to examine ordinal signals in large pre-trained models. These include aligning CLIP with ordinal supervision [58], and applying CLIP or general VLMs to specific tasks such as aesthetics [62, 56], object counting [41, 20], crowd

counting [33], and difference detection [48, 22]. Several works have adapted CLIP for ranking through prompt tuning [32], adapter-based methods [67] or regression-based fine-tuning [10, 57]. Despite these efforts, most focus on task-specific performance or modifying the embedding space, rather than understanding its internal ordinal structure in the embeddings themselves. Our work fills this gap by systematically analysing the rankable structure of existing visual encoders and revealing the presence of ordinal directions in their embedding spaces.

## 3 Are vision embeddings rankable?

We set out to answer the question. For this, we first formally define rankability and strategies to measure it (Section 3.1). We introduce the models and data used for our experiments in Section 3.2. We present results in Section 3.3.

Notation. Our experiments use RGB image datasets with real-valued attribute labels. We denote an image dataset as X ⊂ R3×H×W and define an ordinal attribute A as a function A : X → Y ⊂ R where Y is the range of possible labels and A(x) is the ground-truth label for a given image x. An image encoder is a function f : X → Rd where d is the dimensionality of the embedding space. We occasionally use the term “representation” to refer to an image encoder.

### 3.1 Rankability

We aim to characterize the linear structure of the ordinal information present in visual embedding spaces. Our definition of rankability then naturally emerges as follows.

Definition 1 (Rankability). A representation f : X → Rd is rankable for an ordinal attribute A over an image dataset X if there exists a rank axis vA ∈ Sd−1, a d − 1 dimensional unit sphere, such that for any x1,x2 ∈ X with A(x1) ≥ A(x2), it follows that vA⊤f(x1) ≥ vA⊤f(x2).

The above definition requires a rank axis vA to exactly preserve the ordering provided by the attribute A over the dataset X. In practice, we measure rankability using the generalisation performance of

the rank axis vA learned on a training split Xtrain and tested on a disjoint split Xtest. In this section, we obtain vA via linear regression on the labelled samples: {(f(xi),ai)}i, where ai ∈ R is the ground-truth continuous attribute label for each xi. In Section 4, we consider approaches that do not require access to the attribute labels ai.

Spearman’s rank correlation coefficient (SRCC), denoted as ρ serves as our primary metric quantifying the monotonicity of the relationship between the true ordinality of the attribute A and the predicted ranking along vA. SRCC is widely used in related contexts [66].

We provide three reference points for the obtained rankability to contextualise the SRCC values:

- (1) No-train (“lower bound”). Even for untrained visual encoders, the embeddings do not result in null rank correlation (ρ=0). In order to correctly capture the no-information baseline, we consider the performances of randomly initialized version of each encoder considered [52]. The optimal rank axis vA in this space serves as a lower bound for the rankability of the pretrained encoder.
- (2) Nonlinear (upper bound for embedding). To estimate the total ordinal information in the given embedding, we use a two-layer multilayer perceptron (MLP), known to be a universal approximator [16]. Comparing against a non-linear regressor lets us estimate the proportion of ordinal information in an embedding that can be extracted linearly with a rank axis.
- (3) Finetuning (upper bound for encoder architecture). To measure a broader upper bound indicating the capacity of the encoder architecture and the learnability of the attribute, we finetune the encoder. This conceptually envelops the nonlinear regression upper bound.

### 3.2 Experimental details

We provide further details on the list of attributes, datasets, encoder architectures, and the model selection protocol.

- 3.2.1 Attributes and datasets In total, we use 9 datasets, covering 7 attributes. We provide a detailed breakdown in Table 1.

- Table 1: Datasets and attributes. Summary of datasets used for evaluating the rankability of visual representations.

Attribute Dataset #Train-val #Test Label Range Split Age

UTKFace [70] 13,146 3,287 21–60 From [27] Adience [11] ∼14k ∼4k 8 age groups Official 5-fold

UCF-QNRF [7] 1,201 334 49–12,865 Official

Crowd count

- ShanghaiTech-A [69] 300 182 33–3,139 Official
- ShanghaiTech-B [69] 400 316 9–578 Official

Pitch BIWI Kinect [13] 10,493 4,531 ±60◦ 6 test seqs. Yaw BIWI Kinect 10,493 4,531 ±75◦ 6 test seqs. Roll BIWI Kinect 10,493 4,531 ±45◦ 6 test seqs.

AVA [37] 230,686 4,692 Ratings (1–10) From [66] KonIQ-10k [17] 8,058 2,015 Ratings (1–100) Official

Aesthetics

Image modernness Historical Color Images [42] 1,060 265 5 decades From [66]

### 3.2.2 Architectures

We test representative image-only and CLIP-based encoders. Among image-only encoders, we use the ResNet50 [15], ViT-B/32@224px [9], and ConvNeXtv2-L [61] architectures. Likewise, among CLIP encoders, we test the ResNet50, ViT-B/32, and ConvNeXt-L@320px variants. We also test DINOv2 [40] (ViT-B/14 variant). See Table A7 for more information.

### 3.2.3 Model selection

Hyperparameters for reporting final performances on the test set are selected based on the best validation SRCC, using either official validation splits or holding them out from the corresponding training splits.

Linear and nonlinear regression. We test 30 random hyperparameter configurations per datasetmodel pair. The initial learning rate is sampled from a log-uniform distribution over [10−6, 10−1] and decayed over a cosine schedule to zero, while the weight decay is sampled from a log-uniform distribution over [10−7,10−4]. Data augmentation (horizontal flipping) is also toggled on or off randomly for a given run. We use 100 epochs and a batch size of 128 throughout.

Finetuning. For image-only ResNet50 and ConvNeXt encoders, we conduct a grid search over two encoder learning rates (10−4,10−3) and two weight-decay rates (0,10−5). For ViT-B/32, DINOv2 and CLIP models, we conduct a larger grid search over three encoder learning rates (10−7,10−6,10−5) and four weight-decay values (0,10−5,10−4,10−3). The downstream model always uses a learning rate of 0.1. We use 20 epochs for all runs. Batch size is fixed at 128, except for ConvNeXt-v2, DINOv2 and CLIP-ConvNeXt-v2, where batch size is reduced to 16 because of memory constraints. We use horizontal-flip augmentation in all finetuning runs to mitigate overfitting.

### 3.2.4 Compute resources

All experiments were conducted on a single NVIDIA A100 GPU with 40GB of memory. All primary experiments using frozen embeddings completed within 1–2 minutes, as we cache the embeddings. Finetuning experiments took between 1 and 24 hours each, depending on the dataset. In total, our experiments amounted to approximately 150 compute-days.

### 3.3 Results

Setup. The SRCC for the linear regressor measures the rankability of the underlying vision encoder, while the baselines (no-encoder, nonlinear regression and finetuning) provide reference points. We report our main results in Table 2 and Table 3. In Table 2, we average metrics across all architectures. Then in Table 3, we zoom into individual architectures while retaining only the rankability metric and averaging across all datasets that contain the same attribute. We also report qualitative results in Figure 1. Our observations vary across invididual attributes and architectures, but some patterns emerge.

- Table 2: Vision embeddings are generally rankable. Spearman rank correlation ρ between the true ranking of data samples and the predicted ranks. Higher is better. Results are averaged across all 7 architectures. No-train: Find vA on the embeddings of a randomly-initialised encoder. Rankability: How well does vA encode ordinal information? Nonlinear: Maximal non-linear ordinal information contained in embeddings. Finetuned: How learnable is the target attribute? For more information, see Section 3.1.

Attribute (Dataset) No-train

lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Age (UTKFace) 0.199 0.766 0.776 0.799 Age (Adience) 0.266 0.861 0.878 0.910

Crowd (UCF-QNRF) 0.220 0.843 0.854 0.886

- Crowd Count (ST-A) 0.120 0.734 0.749 0.689

- Crowd Count (ST-B) 0.135 0.869 0.887 0.840

Pitch (Kinect) 0.405 0.803 0.811 0.975 Yaw (Kinect) 0.078 0.434 0.597 0.967 Roll (Kinect) 0.151 0.218 0.326 0.859

Aesthetics (AVA) 0.156 0.653 0.692 0.693 Aesthetics (KonIQ-10k) 0.435 0.761 0.793 0.901 Recency (HCI) 0.324 0.680 0.688 0.722

- Table 3: Rankability across datasets and models. Spearman’s rank correlation ρ between the true ranking of data samples and the predicted ranks. Higher is better. See Appendix Table A7 for model details.

Model Attribute (Dataset) RN50 ViTB32 CNX

DINOB14

CLIPRN50

CLIPViTB32

CLIPCNX

Age (UTKFace) 0.633 0.739 0.772 0.770 0.820 0.810 0.820 Age (Adience) 0.723 0.828 0.871 0.853 0.898 0.924 0.928

Crowd (UCF-QNRF) 0.864 0.837 0.810 0.788 0.870 0.870 0.860

- Crowd (ST-A) 0.799 0.700 0.695 0.653 0.760 0.750 0.780

- Crowd (ST-B) 0.879 0.878 0.867 0.821 0.890 0.860 0.890

Pitch (Kinect) 0.663 0.673 0.909 0.716 0.860 0.920 0.880 Yaw (Kinect) 0.624 0.305 0.384 0.804 0.120 0.360 0.440 Roll (Kinect) 0.352 0.196 0.298 0.512 0.090 0.020 0.060

Aesthetics (AVA) 0.589 0.609 0.644 0.566 0.700 0.710 0.750 Aesthetics (KonIQ-10k) 0.739 0.713 0.744 0.681 0.800 0.790 0.860

Recency (HCI) 0.600 0.592 0.631 0.571 0.780 0.770 0.820

Average rankability of vision embeddings is non-trivially high. We observe in Table 2 that with respect to age on the Adience dataset, the average rankability of 0.861 is much higher than the no-train lower bound of 0.266, while the nonlinear and finetuned upper bounds (0.878 and 0.910) are only slightly higher in comparison. In fact, for all attributes except for yaw and roll, the average rankability is closer to the two upper bounds than to the lower bound.

CLIP embeddings are more rankable than non-CLIP embeddings. We observe in Table 3 that on age, aesthetics, recency and pitch, the best CLIP encoder (eg. CLIP-ConvNeXt with an SRCC of 0.928 on Adience) wins out against the best non-CLIP encoder (eg. vanilla ConvNeXt with an SRCC of 0.871 on Adience). On crowd count, the best CLIP encoders are largely tied with the best non-CLIP encoders (eg. CLIP-RN50 at 0.870 vs vanilla RN50 at 0.864 for UCF-QNRF). On yaw and roll, DINO massively outperforms CLIP encoders (eg. 0.804 vs 0.440 for yaw). In conclusion, apart

from isolated but interesting exceptions, CLIP encoders generally outperform or match non-CLIP encoders.

Some attributes are better ranked than others. For example, the average SRCC over the two age datasets is ∼ 0.8 (see Table 2). Similar average SRCCs are observed for crowd count and pitch angle. Image aesthetics and recency are less well-ranked with average SRCCs between 0.65 and 0.75. Even within the same dataset (BIWI Kinect), yaw and roll angles with SRCCs of 0.434 and 0.218 are quite poorly ranked in comparison to pitch (0.803). We hypothesize that attribute-wise rankabilities are directly proportional to attribute-wise variety present in the training data.

(young) 0th

(old) 100th

- 25th
- 26

50th

75th

[Figure 1]

23

54 vage

27

28

(not crowded) 0th

(crowded) 100th

25th

50th

75th

[Figure 2]

129

233

1408 vcrowd_count

751

944

(looking down) 0th

(looking up) 100th

25th

50th

75th

[Figure 3]

-55

-10

74 vpitch

14

36

(bad) 0th

(good) 100th

25th

50th

75th

[Figure 4]

25

48

76 vaesthetics

58

78

- Figure 1: Visualisation of rank axes. We show rth percentile samples along the rank axes found using linear regression over CLIP-ViT-B/32 embeddings from each respective dataset.

Caveats. The current results are empirical, and our claims are based on the set of attributes considered in our study. Despite following choices established in the literature, we may sometimes not uncover the most optimal finetuned upper bounds. A broader study involving theoretical support for rankability, using more ordinal attributes and possibly stronger upper bounds would be a promising direction for future work.

Takeaway from §3. In general, visual embeddings are highly rankable compared to both the lower bound and upper bound baselines, although there exist variations across attributes (eg. most encoders struggle to rank based on yaw and roll angles) and encoders (eg. CLIP embeddings are, in general, more rankable than non-CLIP embeddings).

## 4 How to (efficiently) find the rank axis?

In Section 3, we showed that visual embeddings are generally rankable. However, the rank axes vA, determining the direction along which the continuous attribute A is sorted accordingly (Section 3.1), were learned using a lot of training data with continuous-attribute annotations that are typically expensive to collect. We examine whether rank axes can also be discovered in more sample- and label-efficient manners.

We organise the section by first tackling an easier setting with abundant data and then more challenging settings with less data available. We begin with the few-shot setting with continuous attribute labels (Section 4.1). Then, we examine the possibility to compute the rank axes with a few “extreme” points that require no cumbersome continuous attribute annotation (Section 4.2). For further practicality, we examine whether the obtained rank axes are resilient to domain shifts (Section 4.3). Finally, we discuss potential strategies to compute the rank axis in a zero-shot manner with text encoders in vision-language models (Section 4.4).

### 4.1 Few-shot learning with continuous attribute labels

In this section, we investigate the learnability of the rank axis vA for an attribute of interest A, when fewer samples with continuous attribute annotations are available. We report the results in Figure 2. We choose the datasets UTKFace, Adience and AVA because of their relatively large sizes compared to the other datasets used in our experiments.

Only a fraction of training data is sufficient. For age on the Adience dataset, only ∼ 1k training samples out of over 11k are sufficient for covering as much as 95% of the gap between the SRCCs of the no-train baseline and full-dataset linear regression. Similarly, for image aesthetics on AVA (CLIP-ViT), only ∼ 16k (CLIP-ViT) or even ∼ 8k (CLIP-ConvNeXt) data points out of ∼ 230k are sufficient. It is evident that learning the rank axis often requires only a fraction of the original training dataset.

### 4.2 Few-shot learning with extreme pairs

We consider the practical scenario where a user wishes to sort their vector database using an arbitrary attribute A. They do not have access to continuous attribute annotations (as was assumed in Section 4.1), but can readily obtain a few samples at the “extremes” of the desired rank axis. We test the effectiveness of this approach.

Setup. Given training and test splits Xtrain and Xtest, respectively, we sample sets of images Sl and Su from the lower and upper extremes of Xtrain, respectively. This simulates the scenario where a user may obtain such “extreme” images using a readily available source like a Web search engine. Next, we calculate the “lower extreme cluster” xl = mean(Sl) and “upper extreme cluster” xu = mean(Su). Finally, the vector vA = x

u−xl

||xu−xl|| gives the rank axis (akin to a steering vector [47, 50]).

Finetuned

1.0

| | | | | | |
|---|---|---|---|---|---|
|re|me|Few| | | |
| | |Uninfo|rmativ|e| |
| |95|% of fu|ll perf.| | |
| | | | | | |
| | | | | | |

Ext

SRCC

0.5

0.0

21 24 27 210 213

#samples

(a) Adience (CLIP-ViT)

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.5

SRCC

0.0

21 24 27 210 213

#samples

(d) Adience (CLIP-ConvNeXt)

1.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.5

SRCC

0.0

21 24 27 210 213

#samples

(b) UTKFace (CLIP-ViT)

1.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.5

SRCC

0.0

21 24 27 210 213

#samples

(e) UTKFace (CLIP-ConvNeXt)

1.0

0.5

SRCC

0.0

21 24 27 210 213

#samples

(c) AVA (CLIP-ViT)

1.0

0.5

SRCC

0.0

21 24 27 210 213

#samples

(f) AVA (CLIP-ConvNeXt)

- Figure 2: Few-shot learning with continuous labels vs extreme samples without continuous labels: extreme samples win out in the small-train-set regime. “Extreme” refers to training using samples from the extreme ends of the ranking axis, while "few" refers to few-shot learning on labeled samples. Full-dataset linear regression performance is given by the star-shaped marker.

Observations. We report the results in Figure 2, comparing with the few-shot scenario where GT continuous attribute labels are available. We observe that when the size of the training dataset is extremely small (upto ∼ 1k), the extreme-pairs method performs better than or at par with few-shot training with continuous attribute labels. This effect can be seen most prominently in the Adience dataset where a rank axis obtained from just two extreme samples achieves an SRCC of ∼ 0.75 on average, while the few-shot case results in an average SRCC close to 0. As the training dataset continues to grow in size, few-shot training with labeled data catches up and finally surpasses the extreme-pairs method. It is nevertheless striking that extreme pairs perform so well at the lower ends of the few-shot setting.

- Table 4: Rank axis transferability. Spearman rank correlation coefficients when a rank axis is trained on dataset i (rows) and evaluated on dataset j (columns).

Evaluated on

Age

(UTKFace)

Age

(Adience)

Crowd

(UCF-QNRF)

Crowd

(ST-A)

Crowd

(ST-B)

Aesthetics

(AVA)

Aesthetics

(KonIQ10k)

Age (UTKFace) +0.81 +0.55 −0.12 −0.11 −0.15 −0.07 +0.05 Age (Adience) +0.68 +0.91 −0.13 −0.10 −0.21 +0.01 −0.08 Crowd (UCF-QNRF) +0.16 −0.39 +0.87 +0.82 +0.66 +0.01 +0.07

Trainedon

- Crowd (ST-A) +0.05 −0.14 +0.73 +0.75 +0.72 +0.13 +0.07

- Crowd (ST-B) +0.31 +0.17 +0.41 +0.39 +0.86 +0.11 +0.05 Aesthetics (AVA) −0.12 −0.11 +0.22 +0.00 +0.49 +0.70 +0.45 Aesthetics (KonIQ10k) +0.06 −0.08 +0.03 +0.18 +0.05 +0.28 +0.79

Takeaway. If one has an extremely small number of samples (1k or less), obtaining extreme samples from both ends of the rank axis (with no additional labeling) may consititute a better expenditure of resources than obtaining continuous attribute labels.

### 4.3 Robustness of rank axis

We now consider the case where there is no access to samples from the target distribution. Assuming that a rank axes was previously learned using some source distribution, how transferable is it to other distributions? We investigate this using three attributes: age, crowd count, and image aesthetics.

Setup. In our main experiments, we use two age datasets (UTKFace, Adience), three crowd count datasets (UCF-QNRF, ShanghaiTech-A, and ShanghaiTech-B), and two aesthetics datasets (AVA, KonIQ-10k). Within each set of datasets containing the same attribute, we then test how well a rank axis learned from one dataset transfers to another, and vice versa. We employ SRCC on the target dataset as our transferability metric in Table 4. Further, we also report cosine similarities between each pair of rank axes in Table 5. As reference points, we also report inter-attribute observations (eg. transfrability from an age dataset to an aesthetics dataset).

Transfer is non-trivial and asymmetric. For example, the rank axis learned from Adience has an SRCC of 0.680 on UTKFace, which is significantly higher than the SRCCs of the same rank axis on other datasets. Given that the two datasets have different labeling systems (especially with Adience labels being much more coarse than those in UTKFace), this is not immediately intuitive, or trivial, and indicates the presence of some (albeit imperfect) “age” axis in the embedding space. At the same time, transfer in the opposite direction is significantly worse, ie. a rank axis trained on UTKFace achieves an SRCC of only 0.55 on Adience. This observation most likely stems from the quality differences between the two datasets.

Rank directions are non-trivially correlated. For example, the cosine similarity between age axes trained on UTKFace and Adience is 0.360. While this similarity is much lower than 1.0, it is still significant given the high dimensionality of the embedding space. This observation again indicates the presence of a “universal” age axis which was (albeit not perfectly) captured by regressors trained on both datasets. Notably, some unexpected correlations also emerge, eg. age (UTKFace) and aesthetics (KonIQ-10k) rank axes have a cosine similarity of 0.220. This suggests the presence of unintended correlations in the training / test data.

### 4.4 Zero-shot setting

For VLMs, language is a potentially data-free approach to finding a rank axis. In principle, a text prompt could correspond to a rank axis in the embedding space. The SRCC of our linear regressors trained in Section 3 then sets an upper bound to the SRCC of any rank axis recovered via prompting. In this section, we investigate the gap between this linear regression upper bound and zero-shot prompt search.

- Table 5: Cosine similarity of rank axes. We compute geometric alignment of rank axes trained on dataset i (rows) and dataset j (columns).

##### Dataset j

Age

(UTKFace)

Age

(Adience)

Crowd

(UCF-QNRF)

Crowd

(ST-A)

Crowd

(ST-B)

Aesthetics

(AVA)

Aesthetics

(KonIQ10k)

Age (UTKFace) +1.00 +0.36 +0.14 +0.08 +0.06 −0.04 +0.22 Age (Adience) +0.36 +1.00 +0.02 +0.04 +0.03 +0.03 +0.05 Crowd (UCF-QNRF) +0.14 +0.02 +1.00 +0.54 +0.26 +0.00 +0.21 Crowd (ST-A) +0.08 +0.04 +0.54 +1.00 +0.31 +0.01 +0.14 Crowd (ST-B) +0.06 +0.03 +0.26 +0.31 +1.00 +0.08 +0.07 Aesthetics (AVA) −0.04 +0.03 +0.00 +0.01 +0.08 +1.00 +0.29 Aesthetics (KonIQ10k) +0.22 +0.05 +0.21 +0.14 +0.07 +0.29 +1.00

Dataseti

Setup. We consider two zero-shot settings. In the single-prompt setting, the direction is defined by the embedding of a text prompt. In the text-difference setting, the direction is defined by the difference between the embeddings of two text prompts, each describing one extreme of the given attribute. We perform a prompt search over 500 GPT-generated prompts for the single-prompt setting, and 100 GPT-generated prompt pairs for the text-difference setting. We report our results in Table 6.

Table 6: Zero-shot results.

Observations and takeaway. Overall, zero-shot methods are suboptimal. For instance, even the best-performing zero-shot approach, our difference-based method on Adience, achieves ρ = 0.782 vs. the corresponding linear model at ρ = 0.917. The gap is more pronounced for some attributes (eg. , ρ = 0.449 vs. ρ = 0.790 for image recency). While more optimal prompts may have been overlooked during the search, this also reflects a realistic setting where one exhausts all intuitive prompt choices. Currently, it is evident that language-based prompting lags considerably behind linear regression using image data, although the text-difference method improves upon vanilla prompting.

Zero-shot One prompt

Zero-shot Difference

Linear upper bound

Attribute (Dataset)

Age (UTKFace) 0.577 0.600 0.817 Age (Adience) 0.670 0.782 0.917 Crowd Count (UCF-QNRF) 0.523 0.315 0.867

- Crowd Count (ST-A) 0.487 0.242 0.763

- Crowd Count (ST-B) 0.590 0.535 0.880 Pitch (Kinect) 0.520 0.617 0.887 Yaw (Kinect) 0.117 0.060 0.307 Roll (Kinect) -0.070 -0.010 0.057 Aesthetics (AVA) 0.367 0.410 0.720 Aesthetics (KonIQ-10k) 0.103 0.547 0.817 Recency (HCI) 0.190 0.449 0.790

Takeaway from §4. One may often efficiently discover the (almost) optimal rank axis using only a fraction of the total labeled data. Learned rank axes are quite transferable across different datasets, suggesting the presence of “universal” rank axes.

## 5 Conclusion and future work

In this work, we investigate visual encoders for the presence of ordinal information. Extensive experiments reveal not only that such information is present, as indicated by the high Spearman rank correlation of nonlinear regressors, but also that most of the available ordinality is linearly encoded, as indicated by the small gap between the performances of linear and MLP regressors. The embedding space indeed possesses a “rankable” structure. This is unexpected and practically useful.

These findings provoke further questions. Most notably, the linearity of ordinal information suggests that one could potentially also characterise embeddings as interpretable collections of latent ordinal subspaces. This would involve discovering a far bigger set of ordinal attributes: we leave this exciting direction to future work.

Acknowledgements. This work was supported by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039A. The authors thank the International

Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Ankit Sonthalia and Arnas Uselis. The authors are also grateful to Wei-Hsiang Yu, the first author of [66], for helpful insights.

## References

- [1] Guillaume Alain and Yoshua Bengio. “Understanding intermediate layers using linear classifier probes”. In: arXiv preprint arXiv:1610.01644 (2016).
- [2] Alberto Baldrati et al. “Conditioned and composed image retrieval combining and partially fine-tuning CLIP-based features”. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2022), pp. 4955–4964. URL: https://api. semanticscholar.org/CorpusId:251034454.
- [3] Davide Berasi et al. “Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models”. In: ArXiv abs/2503.17142 (2025). URL: https://api. semanticscholar.org/CorpusId:277244112.
- [4] Ting Chen et al. “A Simple Framework for Contrastive Learning of Visual Representations”. In: Proceedings of the 37th International Conference on Machine Learning. International Conference on Machine Learning. PMLR, Nov. 21, 2020, pp. 1597–1607. URL: https: //proceedings.mlr.press/v119/chen20j.html.
- [5] Wei Chen et al. “Deep Learning for Instance Retrieval: A Survey”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (2021), pp. 7270–7292. URL: https://api. semanticscholar.org/CorpusId:245837930.
- [6] Wei Chen et al. “Deep learning for instance retrieval: A survey”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 45.6 (2022), pp. 7270–7292.
- [7] “Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds”. In: Haroon Idrees et al. Lecture Notes in Computer Science. Cham: Springer International Publishing, 2018, pp. 544–559. ISBN: 978-3-030-01215-1 978-3-030-01216-8. DOI: 10.1007/ 978-3-030-01216-8_33. URL: https://link.springer.com/10.1007/978-3-03001216-8_33.
- [8] Karan Desai et al. “Hyperbolic Image-Text Representations”. In: ArXiv abs/2304.09172 (2023). URL: https://arxiv.org/pdf/2304.09172.pdf.
- [9] Alexey Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929. June 2021. DOI: 10.48550/arXiv.2010.11929. URL: http: //arxiv.org/abs/2010.11929.
- [10] Yao Du et al. Teach CLIP to Develop a Number Sense for Ordinal Regression. Aug. 7, 2024. DOI: 10.48550/arXiv.2408.03574. arXiv: 2408.03574 [cs]. URL: http://arxiv. org/abs/2408.03574. Pre-published.
- [11] Eran Eidinger, Roee Enbar, and Tal Hassner. “Age and Gender Estimation of Unfiltered Faces”. In: IEEE Transactions on Information Forensics and Security 9.12 (Dec. 2014), pp. 2170–2179. ISSN: 1556-6021. DOI: 10.1109/TIFS.2014.2359646. URL: https: //ieeexplore.ieee.org/document/6906255.
- [12] Abrar Fahim, Alex Murphy, and Alona Fyshe. “Its Not a Modality Gap: Characterizing and Addressing the Contrastive Gap”. In: ArXiv abs/2405.18570 (2024). URL: https://api. semanticscholar.org/CorpusId:270095104.
- [13] Gabriele Fanelli et al. “Real Time Head Pose Estimation from Consumer Depth Cameras”. In: Pattern Recognition. Ed. by Rudolf Mester and Michael Felsberg. Red. by David Hutchison et al. Vol. 6835. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011, pp. 101–110. ISBN: 978-3-642-23122-3 978-3-642-23123-0. DOI: 10.1007/978-3-642-23123-0_11. URL: http://link.springer.com/10.1007/978-3-642-23123-0_11.
- [14] Hu Han et al. “Heterogeneous face attribute estimation: A deep multi-task learning approach”. In: IEEE transactions on pattern analysis and machine intelligence 40.11 (2017), pp. 2597– 2609.
- [15] Kaiming He et al. “Deep Residual Learning for Image Recognition”. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Las Vegas, NV, USA: IEEE, June 2016, pp. 770–778. ISBN: 978-1-4673-8851-1. DOI: 10.1109/CVPR.2016.90. URL: http://ieeexplore. ieee.org/document/7780459/.

- [16] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. “Multilayer feedforward networks are universal approximators”. In: Neural networks 2.5 (1989), pp. 359–366.
- [17] Vlad Hosu et al. “KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment”. In: IEEE Transactions on Image Processing 29 (2020), pp. 4041–

4056. ISSN: 1057-7149, 1941-0042. DOI: 10.1109/TIP.2020.2967829. arXiv: 1910.06180 [cs]. URL: http://arxiv.org/abs/1910.06180.

- [18] Yunshi Huang et al. “Lp++: A surprisingly strong linear probe for few-shot clip”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024, pp. 23773–23782.
- [19] Minyoung Huh et al. “The platonic representation hypothesis”. In: arXiv preprint arXiv:2405.07987 (2024).
- [20] Ruixiang Jiang, Lingbo Liu, and Changwen Chen. “CLIP-Count: Towards Text-Guided ZeroShot Object Counting”. In: Proceedings of the 31st ACM International Conference on Multimedia. Oct. 26, 2023, pp. 4535–4545. DOI: 10.1145/3581783.3611789. arXiv: 2305.07304 [cs]. URL: http://arxiv.org/abs/2305.07304.
- [21] Valentin Khrulkov et al. “Hyperbolic Image Embeddings”. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019), pp. 6417–6427. URL: http: //ieeexplore.ieee.org/stamp/stamp.jsp?tp=%5C&arnumber=9156432.
- [22] Jihyung Kil et al. CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs. July 23, 2024. DOI: 10.48550/arXiv.2407.16837. arXiv: 2407.16837 [cs]. URL: http://arxiv.org/abs/2407.16837. Pre-published.
- [23] Been Kim et al. “Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav)”. In: International conference on machine learning. PMLR. 2018, pp. 2668–2677.
- [24] Darina Koishigarina, Arnas Uselis, and Seong Joon Oh. CLIP Behaves like a Bag-of-Words Model Cross-modally but Not Uni-modally. Feb. 8, 2025. DOI: 10.48550/arXiv.2502.

03566. arXiv: 2502.03566 [cs]. URL: http://arxiv.org/abs/2502.03566. Prepublished.

- [25] Darina Koishigarina, Arnas Uselis, and Seong Joon Oh. “CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally”. In: arXiv preprint arXiv:2502.03566 (2025).
- [26] Chia-Wen Kuo and Z. Kira. “Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning”. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022), pp. 17948–17958. URL: https://api. semanticscholar.org/CorpusId:248572429.
- [27] Maksim Kuprashevich and Irina Tolstykh. MiVOLO: Multi-input Transformer for Age and Gender Estimation. Sept. 22, 2023. DOI: 10.48550/arXiv.2307.04616. arXiv: 2307. 04616 [cs]. URL: http://arxiv.org/abs/2307.04616. Pre-published.
- [28] Meir Yossef Levi and Guy Gilboa. The Double-Ellipsoid Geometry of CLIP. Nov. 21, 2024. DOI: 10.48550/arXiv.2411.14517. arXiv: 2411.14517 [cs]. URL: http://arxiv. org/abs/2411.14517. Pre-published.
- [29] Martha Lewis et al. “Does clip bind concepts? probing compositionality in large image models”. In: arXiv preprint arXiv:2212.10537 (2022).
- [30] Junnan Li et al. “Align before fuse: Vision and language representation learning with momentum distillation”. In: Advances in Neural Information Processing Systems. Vol. 34. 2021, pp. 9694–9705.
- [31] Junnan Li et al. “BLIP: Bootstrapping Language-Image Pre-training for Unified VisionLanguage Understanding and Generation”. In: Proceedings of the 39th International Conference on Machine Learning. International Conference on Machine Learning. PMLR, June 28, 2022, pp. 12888–12900. URL: https://proceedings.mlr.press/v162/li22n.html.
- [32] Wanhua Li et al. OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression. Oct. 1, 2022. DOI: 10.48550/arXiv.2206.02338. arXiv: 2206.02338 [cs]. URL: http://arxiv.org/abs/2206.02338. Pre-published.
- [33] Dingkang Liang et al. CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model. Apr. 9, 2023. DOI: 10.48550/arXiv.2304.04231. arXiv: 2304.04231 [cs]. URL: http://arxiv.org/abs/2304.04231. Pre-published.

- [34] Lucy Liang and Kristen Grauman. “Beyond comparing image pairs: Setwise active learning for relative attributes”. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2014, pp. 208–215.
- [35] Yiming Ma, Victor Sanchez, and Tanaya Guha. CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification. Version 3. Mar. 25, 2025. DOI: 10.48550/arXiv. 2403.09281. arXiv: 2403.09281 [cs]. URL: http://arxiv.org/abs/2403.09281. Pre-published.
- [36] Ron Mokady, Amir Hertz, and Amit H Bermano. “Clipcap: Clip prefix for image captioning”. In: arXiv preprint arXiv:2111.09734 (2021).
- [37] N. Murray, L. Marchesotti, and F. Perronnin. “AVA: A Large-Scale Database for Aesthetic Visual Analysis”. In: 2012 IEEE Conference on Computer Vision and Pattern Recognition. 2012 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Providence, RI: IEEE, June 2012. DOI: 10.1109/cvpr.2012.6247954. URL: http://ieeexplore.ieee. org/document/6247954/.
- [38] Kevin Musgrave, Serge Belongie, and Ser-Nam Lim. “A metric learning reality check”. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXV 16. Springer. 2020, pp. 681–699.
- [39] James Oldfield et al. “Parts of Speech-Grounded Subspaces in Vision-Language Models”. In: ArXiv abs/2305.14053 (2023). URL: https://api.semanticscholar.org/CorpusId: 258841517.
- [40] Maxime Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. en. arXiv:2304.07193 [cs]. Feb. 2024. URL: http://arxiv.org/abs/2304.07193.
- [41] Roni Paiss et al. Teaching CLIP to Count to Ten. Feb. 23, 2023. DOI: 10.48550/arXiv. 2302.12066. arXiv: 2302.12066 [cs]. URL: http://arxiv.org/abs/2302.12066. Pre-published.
- [42] Frank Palermo, James Hays, and Alexei A. Efros. “Dating Historical Color Images”. In: Computer Vision – ECCV 2012. Ed. by Andrew Fitzgibbon et al. Red. by David Hutchison et al. Vol. 7577. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012, pp. 499–512. ISBN: 978-3-642-33782-6 978-3-642-33783-3. DOI: 10.1007/978-3-642-33783-3_36. URL: http://link.springer.com/10.1007/978-3-642-33783-3_36.
- [43] Devi Parikh and Kristen Grauman. “Relative Attributes”. In: 2011 International Conference on Computer Vision. 2011 IEEE International Conference on Computer Vision (ICCV). Barcelona, Spain: IEEE, Nov. 2011, pp. 503–510. ISBN: 978-1-4577-1102-2 978-1-4577-1101-5 9781-4577-1100-8. DOI: 10.1109/ICCV.2011.6126281. URL: http://ieeexplore.ieee. org/document/6126281/.
- [44] Zhi Qiao et al. “HYDEN: Hyperbolic Density Representations for Medical Images and Reports”. In: International Conference on Computational Linguistics. 2024. URL: https://api. semanticscholar.org/CorpusId:271903357.
- [45] Alec Radford et al. “Learning Transferable Visual Models From Natural Language Supervision”. In: International Conference on Machine Learning. 2021. URL: https://arxiv.org/ pdf/2103.00020.pdf.
- [46] Alec Radford et al. “Learning Transferable Visual Models From Natural Language Supervision”. In: Proceedings of the 38th International Conference on Machine Learning. International Conference on Machine Learning. PMLR, July 1, 2021, pp. 8748–8763. URL: https://proceedings.mlr.press/v139/radford21a.html.
- [47] Nina Rimsky et al. “Steering Llama 2 via Contrastive Activation Addition”. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). ACL 2024. Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 15504–15522. DOI: 10.18653/v1/2024.acl-long.828. URL: https://aclanthology.org/2024.acllong.828/.
- [48] Dylan Sam et al. Finetuning CLIP to Reason about Pairwise Differences. Sept. 15, 2024. DOI: 10.48550/arXiv.2409.09721. arXiv: 2409.09721 [cs]. URL: http://arxiv.org/ abs/2409.09721. Pre-published.

- [49] Konstantin Schall et al. “Improving Image Encoders for General-Purpose Nearest Neighbor Search and Classification”. In: Proceedings of the 2023 ACM International Conference on Multimedia Retrieval. ICMR ’23. Thessaloniki, Greece: Association for Computing Machinery, 2023, pp. 57–66. ISBN: 9798400701788. DOI: 10.1145/3591106.3592266. URL: https: //doi.org/10.1145/3591106.3592266.
- [50] Curt Tigges et al. Linear Representations of Sentiment in Large Language Models. Oct. 23,

2023. DOI: 10.48550/arXiv.2310.15154. arXiv: 2310.15154 [cs]. URL: http:// arxiv.org/abs/2310.15154. Pre-published.

- [51] Matthew Trager et al. “Linear Spaces of Meanings: Compositional Structures in VisionLanguage Models”. In: 2023 IEEE/CVF International Conference on Computer Vision (ICCV)

(2023), pp. 15349–15358. URL: https : / / api . semanticscholar . org / CorpusId : 257766294.

- [52] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. “Deep image prior”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018, pp. 9446–9454.
- [53] Arnas Uselis, Andrea Dittadi, and Seong Joon Oh. “BEYOND DECODABILITY: LINEAR FEATURE SPACES ENABLE VISUAL COMPOSITIONAL GENERALIZATION”. In: Workshop on Spurious Correlation and Shortcut Learning: Foundations and Solutions. Mar. 6, 2025. URL: https://openreview.net/forum?id=IaTj8xNn7F&referrer= %5BAuthor % 20Console % 5D(%2Fgroup % 3Fid % 3DICLR . cc % 2F2025 % 2FWorkshop % 2FSCSL%2FAuthors%23your-submissions).
- [54] Arnas Uselis and Seong Joon Oh. “Intermediate Layer Classifiers for OOD Generalization”. In: International Conference on Learning Representations (ICLR). 2025.
- [55] Ivan Vendrov et al. “Order-Embeddings of Images and Language”. In: CoRR abs/1511.06361

(2015). URL: https://arxiv.org/pdf/1511.06361.pdf.

- [56] Jianyi Wang, Kelvin C.K. Chan, and Chen Change Loy. “Exploring CLIP for Assessing the Look and Feel of Images”. In: Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence. Vol. 37. AAAI’23/IAAI’23/EAAI’23. AAAI Press, Feb. 7, 2023, pp. 2555–2563. ISBN: 9781-57735-880-0. DOI: 10.1609/aaai.v37i2.25353. URL: https://doi.org/10.1609/ aaai.v37i2.25353.
- [57] Rui Wang et al. “Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification”. In: (2023).
- [58] Rui Wang et al. “Learning-to-rank meets language: Boosting language-driven ordering alignment for ordinal classification”. In: Advances in Neural Information Processing Systems 36

(2023).

- [59] Tongzhou Wang and Phillip Isola. “Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere”. In: ArXiv abs/2005.10242 (2020). URL: https://arxiv.org/pdf/2005.10242.pdf.
- [60] Xin Wang et al. “Disentangled representation learning”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).
- [61] Sanghyun Woo et al. ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. Jan. 2, 2023. DOI: 10.48550/arXiv.2301.00808. arXiv: 2301.00808 [cs]. URL: http://arxiv.org/abs/2301.00808. Pre-published.
- [62] Liwu Xu et al. “Clip brings better features to visual aesthetics learners”. In: arXiv preprint arXiv:2307.15640 (2023).
- [63] Lewei Yao et al. “Filip: Fine-grained interactive language-image pre-training”. In: arXiv preprint arXiv:2111.07783 (2021).
- [64] Aron Yu and Kristen Grauman. “Just noticeable differences in visual attributes”. In: Proceedings of the IEEE International Conference on Computer Vision. 2015, pp. 2416–2424.
- [65] Jiahui Yu et al. “CoCa: Contrastive Captioners are Image-Text Foundation Models”. In: Transactions on Machine Learning Research (2022). URL: https://openreview.net/ forum?id=M_Vb2v063oH.
- [66] Wei-Hsiang Yu et al. “RANKING-AWARE ADAPTER FOR TEXT-DRIVEN IMAGE ORDERING WITH CLIP”. In: ICLR (2025).

- [67] Wei-Hsiang Yu et al. Ranking-Aware Adapter for Text-Driven Image Ordering with CLIP. Dec. 9, 2024. DOI: 10.48550/arXiv.2412.06760. arXiv: 2412.06760 [cs]. URL: http://arxiv.org/abs/2412.06760. Pre-published.
- [68] Mert Yuksekgonul et al. “When and why vision-language models behave like bags-of-words, and what to do about it?” In: arXiv preprint arXiv:2210.01936 (2022).
- [69] Yingying Zhang et al. “Single-Image Crowd Counting via Multi-Column Convolutional Neural Network”. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Las Vegas, NV, USA: IEEE, June 2016, pp. 589–597. ISBN: 978-1-4673-8851-1. DOI: 10.1109/CVPR.2016.

70. URL: http://ieeexplore.ieee.org/document/7780439/.

- [70] Zhifei Zhang, Yang Song, and Hairong Qi. Age Progression/Regression by Conditional Adversarial Autoencoder. Mar. 28, 2017. DOI: 10.48550/arXiv.1702.08423. arXiv: 1702.08423 [cs]. URL: http://arxiv.org/abs/1702.08423. Pre-published.
- [71] Daniel Zoran et al. “Learning Ordinal Relationships for Mid-Level Vision”. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV). Dec. 2015.

- A Further Experimental Details Here, we list further details of the architectures considered in our study.

Table A7: Summary of vision encoders used in our study. “Acronym” refers to how each model is denoted in our main results tables.

Acronym Architecture #Dims Year #Params Type Input Size

RN50 ResNet50 2048 2015 25.6M ConvNet 224×224 ViTB32 ViT-B/32 768 2020 88.2M Transformer 224×224 CNX ConvNeXtV2 1536 2023 198M ConvNet 224×224 DINO-B14 DINOv2 (ViT-B/14) 768 2023 86.6M Transformer 518×518 CLIP-RN50 CLIP ResNet50 1024 2021 38.3M ConvNet 224×224 CLIP-ViTB32 CLIP ViT-B/32 512 2021 87.8M Transformer 224×224 CLIP-CNX CLIP ConvNeXtV2 768 2023 199.8M ConvNet 320×320

- B Further details on datasets and dataset-specific results

In the main paper, we present results for each dataset aggregated over all models (Table 2 in the main paper) and rankabilities (Spearman ρ) for each model-dataset pair (Table 3 in the main paper). While the main results convey the primary evidence towards our claim (vision embeddings are rankable), we also report more detailed results on each dataset in this section. Please also refer to Table 1 in the main paper for a condensed overview of all datasets considered.

### B.1 Age

UTKFace, introduced in [70], is a dataset of face images with age labels ranging from 0 to 116. Following [66, 27], we use a smaller subset with ages ranging between 21 and 60. The dataset was downloaded from the official website (https://susanqq.github.io/UTKFace/). We report results in Table B8.

Table B8: UTKFace (Age). Spearman’s rank correlation ρ across evaluation strategies. No-train: linear probe on untrained encoder. Rankability: linear probe on pretrained encoder. Nonlinear: MLP on frozen encoder. Finetuned: encoder + head trained end-to-end. Higher is better.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.212 0.633 0.636 0.762 ViT-B/32 0.283 0.739 0.749 0.737 ConvNeXtV2-L 0.283 0.772 0.776 0.815 DINOv2 ViT-B/14 0.054 0.770 0.770 0.812 OpenAI CLIP ResNet-50 0.130 0.820 0.830 0.790 OpenAI CLIP ViT-B/32 0.250 0.810 0.830 0.830 OpenCLIP ConvNeXt-L (D, 320px) 0.180 0.820 0.840 0.850

Mean 0.199 0.766 0.776 0.799

Adience, introduced in [11], is another age dataset. Unlike UTKFace, it contains coarse labels (8 age groups instead of exact ages). We use the “aligned” version of the images and five-fold cross-validation as in [66]. Results can be found in Table B9.

Table B9: Adience (Age). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.328 0.723 0.759 0.894 ViT-B/32 0.310 0.828 0.860 0.892 ConvNeXtV2-L 0.522 0.871 0.885 0.910 DINOv2 ViT-B/14 0.120 0.853 0.877 0.914 OpenAI CLIP ResNet-50 0.070 0.898 0.914 0.894 OpenAI CLIP ViT-B/32 0.292 0.924 0.922 0.928 OpenCLIP ConvNeXt-L (D, 320px) 0.220 0.928 0.932 0.938

Mean 0.266 0.861 0.878 0.910

### B.2 Crowd count

UCF-QNRF, introduced in [7], is a large crowd counting dataset containing images from diverse parts of the world. We use the official download link at https://www.crcv.ucf.edu/data/ucf-qnrf/ and the official train-test splits. Results can be found in Table B10.

- Table B10: UCF-QNRF (Crowd Count). Spearman’s rank correlation ρ across evaluation strategies.

Model

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

ResNet-50 0.466 0.864 0.870 0.826 ViT-B/32 0.288 0.837 0.840 0.794 ConvNeXtV2-L −0.054 0.810 0.816 0.938 DINOv2 ViT-B/14 0.219 0.788 0.842 0.961 OpenAI CLIP ResNet-50 0.240 0.870 0.880 0.820 OpenAI CLIP ViT-B/32 0.280 0.870 0.870 0.900 OpenCLIP ConvNeXt-L (D, 320px) 0.100 0.860 0.860 0.960

Mean 0.220 0.843 0.854 0.886

ShanghaiTech, introduced in [69], is another crowd counting dataset consisting of two parts: A and

- B. While part A was crawled from the Internet and features larger crowds in general, part B was taken from metropolitan areas of Shanghai and features much smaller crowds. We use the DropBox link available at https://github.com/desenzhou/ShanghaiTechDataset and official train-test splits for both parts. Results can be found in Table B11 and Table B12.

- Table B11: ShanghaiTech-A (Crowd Count). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.359 0.799 0.802 0.529 ViT-B/32 0.148 0.700 0.623 0.558 ConvNeXtV2-L 0.061 0.695 0.753 0.834 DINOv2 ViT-B/14 −0.017 0.653 0.722 0.786 OpenAI CLIP ResNet-50 0.200 0.760 0.770 0.510 OpenAI CLIP ViT-B/32 0.010 0.750 0.770 0.700 OpenCLIP ConvNeXt-L (D, 320px) 0.080 0.780 0.800 0.910

Mean 0.120 0.734 0.749 0.689

- Table B12: ShanghaiTech-B (Crowd Count). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.280 0.879 0.906 0.672 ViT-B/32 0.225 0.878 0.889 0.710 ConvNeXtV2-L 0.070 0.867 0.876 0.955 DINOv2 ViT-B/14 0.020 0.821 0.869 0.972 OpenAI CLIP ResNet-50 0.270 0.890 0.900 0.690 OpenAI CLIP ViT-B/32 0.040 0.860 0.860 0.900 OpenCLIP ConvNeXt-L (D, 320px) 0.040 0.890 0.910 0.980

Mean 0.135 0.869 0.887 0.840

### B.3 Headpose (Euler angles)

The BIWI Kinect dataset, introduced in [13], is a collection of 24 different videos wherein the subject of the video sits about a meter away from a Kinect (https://en.wikipedia.org/wiki/Kinect) sensor and rotates their head to span the entire range of possible head-pose angles pitch (rotation about the x-axis), yaw (rotation about the y-axis) and roll (rotation about the z-axis). As there exists no official split that we know of, we randomly hold out 6 sequences for testing. Results can be found in Table B13 (pitch), Table B14 (yaw) and Table B15 (roll).

Table B13: Kinect (Pitch). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.359 0.663 0.615 0.973 ViT-B/32 0.401 0.673 0.505 0.951 ConvNeXtV2-L 0.548 0.909 0.882 0.984 DINOv2 ViT-B/14 0.231 0.716 0.986 0.979 OpenAI CLIP ResNet-50 0.450 0.860 0.870 0.970 OpenAI CLIP ViT-B/32 0.400 0.920 0.940 0.980 OpenCLIP ConvNeXt-L (D, 320px) 0.450 0.880 0.880 0.990

Mean 0.405 0.803 0.811 0.975

Table B14: Kinect (Yaw). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.160 0.624 0.726 0.990 ViT-B/32 0.209 0.305 0.305 0.838 ConvNeXtV2-L 0.113 0.384 0.716 0.989 DINOv2 ViT-B/14 −0.046 0.804 0.871 0.994 OpenAI CLIP ResNet-50 −0.060 0.120 0.530 0.980 OpenAI CLIP ViT-B/32 0.160 0.360 0.330 0.990 OpenCLIP ConvNeXt-L (D, 320px) 0.010 0.440 0.700 0.990

Mean 0.078 0.434 0.597 0.967

Table B15: Kinect (Roll). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.202 0.352 0.430 0.930 ViT-B/32 0.098 0.196 0.375 0.477 ConvNeXtV2-L 0.550 0.298 0.368 0.963 DINOv2 ViT-B/14 0.256 0.512 0.551 0.912 OpenAI CLIP ResNet-50 0.140 0.090 0.300 0.920 OpenAI CLIP ViT-B/32 −0.060 0.020 0.170 0.850 OpenCLIP ConvNeXt-L (D, 320px) −0.130 0.060 0.090 0.960

Mean 0.151 0.218 0.326 0.859

### B.4 Aesthetics (Mean Opinion Score)

The Aesthetics Visual Analysis (AVA) dataset, introduced in [37], is a large-scale dataset including aesthetic preference scores provided by human annotators. Each image is labeled by multiple annotators, each assigning a score in the range 1-10. The mean opinion score (MOS) of the image is then computed as a weighted average over the ratings where the weight of a rating is provided by its frequency. We use the split provided by [66] in their official repository (https://github.com/uynaes/RankingAwareCLIP/tree/main/examples). Results are reported in Table B16.

- Table B16: AVA (Image Aesthetics). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.237 0.589 0.628 0.672 ViT-B/32 0.157 0.609 0.666 0.672 ConvNeXtV2-L 0.158 0.644 0.685 0.728 DINOv2 ViT-B/14 0.057 0.566 0.648 0.590 OpenAI CLIP ResNet-50 0.150 0.700 0.710 0.700 OpenAI CLIP ViT-B/32 0.200 0.710 0.730 0.700 OpenCLIP ConvNeXt-L (D, 320px) 0.130 0.750 0.780 0.790

Mean 0.156 0.653 0.692 0.693

KonIQ-10k, introduced in [17], is another aesthetics or image quality assessment (IQA) dataset that aims to model naturally occurring image distortions with mean opinion scores ranging roughly between 1 and 100. We use the official train-test splits. Results can be found in Table B17.

- Table B17: KonIQ-10k (Image Aesthetics). Spearman’s rank correlation ρ across evaluation strategies.

Model

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

ResNet-50 0.563 0.739 0.739 0.874 ViT-B/32 0.488 0.713 0.753 0.813 ConvNeXtV2-L 0.487 0.744 0.765 0.930 DINOv2 ViT-B/14 0.324 0.681 0.753 0.948 OpenAI CLIP ResNet-50 0.400 0.800 0.840 0.900 OpenAI CLIP ViT-B/32 0.460 0.790 0.830 0.890 OpenCLIP ConvNeXt-L (D, 320px) 0.320 0.860 0.870 0.950

Mean 0.435 0.761 0.793 0.901

B.5 Image recency

Historical Color Images (HCI), introduced in [42], was designed for the task of classifying an image by the decade during which it was taken. Therein emerges a natural ordering over the decades, defining the ordinal attribute of image “modernness” or “recency”. We use the split provided by [66] in their repository (https://github.com/uynaes/RankingAwareCLIP/tree/main/examples) and report the results in Table B18.

- Table B18: HCI (Historical Color Images). Spearman’s rank correlation ρ across evaluation strategies.

No-train lower bound

Rankability main

Nonlinear upper bound

Finetuned upper bound

Model

ResNet-50 0.351 0.600 0.592 0.614 ViT-B/32 0.377 0.592 0.618 0.529 ConvNeXtV2-L 0.362 0.631 0.663 0.771 DINOv2 ViT-B/14 0.131 0.571 0.601 0.748 OpenAI CLIP ResNet-50 0.320 0.780 0.770 0.760 OpenAI CLIP ViT-B/32 0.430 0.770 0.780 0.760 OpenCLIP ConvNeXt-L (D, 320px) 0.300 0.820 0.790 0.870

Mean 0.324 0.680 0.688 0.722

## C Comparison with SOTA

Prior research has presented results from dedicated or general efforts to solve the datasets considered in our study. Our main aim is to understand the rankability emerging out of the structure in visual embedding spaces, and we contextualize our numbers using reference metrics (lower bound provided by the no-encoder baseline and upper bound provided by nonlinear regression and finetuned encoders). However, in Table C19, we also provide comparisons with state-of-the-art results to further contextualize our results. Some comparisons suggest that simple linear regression over pretrained embeddings often performs comparably with or even surpasses dedicated efforts. Although architectural and training dataset differences mean that this comparison is not always fair, we emphasize the contrast in implementational simplicty between dedicated efforts and simple linear regression over pretrained embeddings that are often readily available and easy to use.

- 20

Table C19: Comparing linear / nonlinear regression against recent state-of-the-art methods. “Linear” and “Nonlinear” use regression over CLIP-ConvNeXt-L embeddings. Dashes indicate metrics unreported in prior work. We take the numbers for age, aesthetics and recency from [66], and crowd count from [35]. Under “Downstream model”, we report the components used on top of pretrained visual embeddings (CLIP or non-CLIP models); sometimes, we also report if the encoder itself was retrained. Under “Downstream data”, we report the additional data used for training the method. Finally, under “Other ingredients”, we also report miscellaneous extra components used by the corresponding method. All method interpretations are to the best of our knowledge.

Attribute Dataset Method Spearman ρ MAE Downstream model Downstream data Other ingredients

Age UTKFace Yu et al. [66] – 3.83 Cross-attn encoder, two ranking heads, learnable text prompt tokens

Images with age labels Text encoder

MiVOLO [27] – 4.23 Regression heads Body images, face patches, age labels Feature enhancer module for fused joint representations

Linear – 4.25 Linear regressor Images with age labels None Nonlinear – 4.10 2-layer MLP regressor Images with age labels None

Age Adience Yu et al. [66] – 0.36 (0.03) Cross-attn encoder, two ranking heads, learnable text prompt tokens

Images with age labels Text encoder

OrdinalCLIP [32] – 0.47 (0.06) (Retrain image encoder for the task) Images with age labels Text encoder; learn “continuous” rank prototype (text) embeddings for each rank

Linear regressor – 0.48 (0.02) Linear Images with age labels None Nonlinear – 0.45 (0.02) 2-layer MLP regressor Images with age labels None

Crowd Count UCF-QNRF CLIP-EBC [35] – 80.3 Blockwise classification module Images with count labels Text encoder CrowdCLIP [33] – 283.3 (Retrain image encoder) Crowd images Text encoder, three-stage progressive filtering during inference

Linear – 246.4 Linear Images with count labels None Nonlinear – 248.0 2-layer MLP Images with count labels None

- Crowd Count ST-A CLIP-EBC [35] – 52.5 Blockwise classification module Images with count labels Text encoder CrowdCLIP [33] – 146.1 (Retrain image encoder) Crowd images Text encoder, three-stage progressive

filtering during inference

Linear – 167.1 Linear Images with count labels None Nonlinear – 151.7 2-layer MLP Images with count labels None

- Crowd Count ST-B CLIP-EBC [35] – 6.6 Blockwise classification module Images with count labels Text encoder CrowdCLIP [33] – 69.3 (Retrain image encoder) Crowd images Text encoder, three-stage progressive

filtering during inference Continued on next page

- 21

Table C19 – Continued from previous page

###### Attribute Dataset Method Spearman ρ MAE Downstream model Downstream data Other ingredients

Linear – 34.7 Linear Images with count labels None Nonlinear – 29.7 2-layer MLP Images with count labels None

Aesthetics AVA* Yu et al. [66] 0.747 – Cross-attn encoder, two ranking heads, learnable text prompt tokens

Images with MOS labels (1–10) Text encoder

CLIP-IQA [56] 0.415 – Softmax over two similarity scores None Text encoder, prompt engineering, remove position embedding

Linear 0.749 – Linear Images with MOS labels(1–10) None Nonlinear 0.775 – 2-layer MLP Images with MOS labels (1–10) None

Aesthetics KonIQ-10k Yu et al. [66] 0.911 – Cross-attn encoder, two ranking heads, learnable text prompt tokens

Images with MOS labels (1–100) Text encoder

CLIP-IQA [56] 0.727 – Softmax over two similarity scores Images with MOS labels (1–100) Text encoder, prompt engineering, remove position embedding

Linear 0.860 – Linear Images with MOS labelss (1–100) None Nonlinear 0.870 – 2-layer MLP Images with MOS labels (1–100) None

Recency HCI* Yu et al. [66] – 0.32 (0.03) Cross-attn encoder, two ranking heads, learnable text prompt tokens

Images with decade labels Text encoder

OrdinalCLIP [32] – 0.67 (0.03) (Retrain image encoder for the task) Images with decade labels Text encoder; learn “continuous” rank prototype (text) embeddings for each rank

Linear – 0.64 Linear Images with decade labels None Nonlinear – 0.60 2-layer MLP Images with decade labels None

