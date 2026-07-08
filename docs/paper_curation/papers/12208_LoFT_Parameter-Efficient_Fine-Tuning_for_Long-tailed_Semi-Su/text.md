## LoFT: Long-Tailed Semi-Supervised Learning via Parameter-Efficient Fine-Tuning in Open-World Scenarios

Zhiyuan Huang*1 Jiahao Chen*1 Bing Su1

# arXiv:2509.09926v5[cs.LG]8Apr2026

### Abstract

Long-tailed semi-supervised learning (LTSSL) presents a formidable challenge where models must overcome the scarcity of tail samples while mitigating the noise from unreliable pseudolabels. Most prior LTSSL methods are designed to train models from scratch, which often leads to issues such as overconfidence and low-quality pseudo-labels. To address this problem, we first theoretically prove that utilizing a foundation model significantly reduces the hypothesis complexity, which tightens the generalization bound and in turn minimizes the Balanced Posterior Error (BPE). Furthermore, we demonstrate that the feature compactness of foundation models strictly compresses the acceptance region for outliers, providing a geometric guarantee for robustness. Motivated by these theoretical insights, we extend LTSSL into the foundation model fine-tuning paradigm and propose a novel framework: LoFT (Long-tailed semisupervised learning via parameter-efficient FineTuning). Furthermore, we explore a more practical setting by investigating semi-supervised learning under open-world conditions, where the unlabeled data may include out-of-distribution (OOD) samples. To handle this problem, we propose LoFT-OW (LoFT under Open-World scenarios) to improve the discriminative ability. Experimental results on multiple benchmarks demonstrate that our method achieves superior performance. Code is available: https://github.com/ games-liker/LoFT

### 1. Introduction

Real-world data often follows a long-tailed or imbalanced distribution, where a small number of head classes dominate

1Renmin University of China. Correspondence to: Zhiyuan Huang <huangzhiyuan@ruc.edu.cn>, Jiahao Chen <nicelemon666@gmail.com>. Preprint. April 9, 2026.

[Figure 1]

Figure 1. Differences among supervised learning, semi-supervised learning, and semi-supervised learning in open-world scenarios.

the majority of samples, while the remaining tail classes are represented by only a limited number of instances (Cui et al., 2019). This imbalance poses significant challenges for model training, particularly in achieving satisfactory performance on tail classes. To address this issue, LTSSL has emerged as an effective solution by incorporating a large amount of unlabeled data into the imbalanced labeled dataset (Wei et al., 2021; Wei & Gan, 2023). The basic idea of LTSSL is to generate pseudo-labels for unlabeled data and select high-confidence samples to guide model training (Ouali et al., 2020). While the current methods have achieved notable success and demonstrated promising results, they still face dilemmas hindering further improvement.

Previous LTSSL approaches typically rely on training Convolutional Neural Networks (CNNs) from scratch (Wei et al., 2021), which presents several challenges. First, CNNs are known to be overconfident (Guo et al., 2017), often assigning high-confidence scores to incorrect predictions. Although methods like FixMatch (Sohn et al., 2020) employ a “weak-to-strong” pipeline, using weakly augmented samples to determine labels and strong augmented samples to determine the logits, this overconfidence issue persists, especially for tail classes, as shown in Fig. 2. Second, in the early training stages, the model produces unreliable predictions, resulting in low-quality pseudo-labels. As a result, current LTSSL approaches often require more training iterations and carefully designed strategies to dynamically manage the use of unlabeled data (Wei & Gan, 2023). Both of the

Many Medium Few Overall

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Many Medium Few Overall

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Training from scratch

Training from scratch

Many Medium Few Overall

Many Medium Few Overall

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

PEFT

PEFT

(a) ImageNet-LT

(b) Places365-LT

Figure 2. The reliability diagrams on (a) ImageNet-LT and (b) Places365-LT based on training from scratch and PEFT, respectively. The horizontal axis represents confidence, and the vertical axis represents accuracy. The Expected Calibration Error (ECE) is computed to quantify the calibration quality, with lower values indicating better confidence-accuracy alignment.

dilemmas limit the application of LTSSL.

Beyond the standard LTSSL setting (Yang & Xu, 2020), a critical gap exists between theoretical assumptions and reality: most methods assume distributional homogeneity, where the labeled and unlabeled sets share the same class space. To bridge this gap, we explore a more pragmatic and challenging setting: Open-World LTSSL. In realworld scenarios, such as wildlife classification, the unlabeled stream inevitably contains Out-of-Distribution (OOD) samples (e.g., rare or unknown species) not present in the labeled set. Directly applying existing LTSSL methods in this context carries the risk of forcing OOD samples into in-distribution (ID) classes, thereby corrupting the feature space. Moreover, models trained from scratch typically lack the semantic capacity to effectively identify or reject these OOD samples (Hendrycks & Gimpel, 2016).

Motivated by theoretical insights and empirical observations, we introduce a streamlined framework called LoFT(LongTailed Semi-Supervised Learning via Parameter-Efficient Fine-Tuning) that leverages the high generalization capabilities of Foundation Models(FMs) to achieve inherently well-calibrated and high-quality pseudo-labels. Moreover, we propose LoFT-OW (LoFT under Open-World scenarios). By leveraging the compact feature clusters of the fine-tuned foundation model, LoFT-OW incorporates a built-in OOD detection mechanism to natively filter out irrelevant samples. This effectively mitigates negative transfer and preserves the purity of representation learning.

Our contributions are summarized as follows:

• We made theoretical contributions, and motivated by theoretical analysis and sufficient empirical experiments, we address the LTSSL problem and propose LoFT that is inherently well-calibrated and can be effectively utilized to impove the quality of pseudolabels.

- • We extend LTSSL to a more realistic Open-World Scenario, named LoFT-OW, where unlabeled data may contain OOD samples. LoFT incorporates a built-in OOD detection mechanism, filtering out irrelevant samples and improving model robustness and representation learning in diverse real-world data conditions.
- • We conduct experiments on traditional LTSSL benchmarks, including CIFAR-LT and ImageNet127, and observe that LoFT achieves competitive performance. Furthermore, LoFT achieves superior performance in the more challenging open-world scenarios, outperforming previous methods even when using only 10% of the unlabeled data compared with previous works, highlighting its strong discriminative capability.

### 2. Related Work

Long-tailed semi-supervised learning LTSSL (Peng et al., 2023; Hou & Jia, 2025; Wei et al., 2021) aims to improve the performance of models trained on long-tailed labeled data by leveraging additional unlabeled data. The basic idea is to generate pseudolabels for the unlabeled samples and incorporate them into the training process. CReST (Wei et al., 2021) observes that models trained under imbalanced distributions can still generate high-precision pseudolabels for tail classes. Based on this insight, it proposes the class-rebalancing self-training framework to improve performance. In (Wei & Gan, 2023), the authors relax the assumption of consistent class distributions between labeled and unlabeled data and introduce ACR, a method that dynamically refines pseudo-labels by estimating the true class distribution of unlabeled data under a unified formulation. ADELLO (Sanchez Aimar et al., 2024) presents FlexDA, a dynamic logit adjustment and distillation-based framework that enhances calibration and achieves strong performance in LTSSL settings. Recently, FMs (Radford

et al., 2021), pre-trained on large-scale datasets, have demonstrated strong generalization capabilities across a variety of downstream tasks, including those with long-tailed distributions (Shi et al., 2024; Tian et al., 2022; Dong et al., 2022). However, how to effectively leverage FMs to benefit LTSSL remains an open and underexplored research direction. In this paper, we aim to address this challenge and propose LoFT, a novel framework designed to integrate the strengths of FMs into the LTSSL paradigm.

Long-tailed Confidence calibration Confidence calibration aims to align the predicted confidence scores with the true accuracy, which is important for safety measurement, OOD detection (Liu et al., 2024). Prior studies have shown that modern CNNs tend to be overconfident (Tomani et al., 2021; Guo et al., 2017), particularly under longtailed distributions (Zhong et al., 2021). MiSLAS (Zhong et al., 2021) addresses this issue by introducing a two-stage training pipeline that incorporates three key techniques: mixup (Zhang et al., 2017) pre-training, label-aware smoothing, and batch normalization (Ioffe & Szegedy, 2015) shifting. These techniques collectively enhance the model’s calibration capability. UniMix (Xu et al., 2021) extends the mixup strategy to imbalanced scenarios by adopting an advanced mixing factor and a sampling strategy that favors minority classes, thereby improving calibration performance under long-tailed distributions. Recently, adapting FMs to imbalanced learning has attracted increasing attention. However, the issue of confidence calibration in this setting remains largely underexplored. As previously discussed, a well-calibrated model is crucial for generating high quality pseudo-labels, which are essential for effective semi-supervised learning. In this work, we investigate confidence calibration within the context of LTSSL to further enhance performance under long-tailed distributions.

Table 1. The results on OOD tasks on different datasets. PEFT† and PEFT‡ denote the fine-tuned model from CLIP and OpenCLIP, respectively.

OOD Dataset Method AUC↑ AP-in↑ AP-out↑ FPR↓

OE 76.01 85.28 57.47 87.45 OCL 75.92 82.99 66.48 70.01

Texture

PEFT† 87.86 92.79 80.15 49.45 PEFT‡ 91.32 94.66 86.22 38.26

OE 81.82 73.25 89.10 80.98 OCL 78.64 69.21 86.26 86.38

SVHN

PEFT† 86.62 73.87 94.26 47.29 PEFT‡ 90.68 81.80 95.98 41.00

OE 62.60 66.16 57.77 93.53 OCL 60.29 63.21 55.71 94.22

CIFAR-10

PEFT† 83.97 84.42 82.61 61.98 PEFT‡ 86.39 86.95 85.38 57.38

OE 68.22 79.36 51.82 88.54 OCL 69.56 79.97 54.47 85.91

TinyImageNet

PEFT† 81.34 88.30 70.20 70.03 PEFT‡ 83.35 89.85 72.98 66.02

OE 76.81 85.33 60.94 83.79 OCL 79.14 86.56 66.58 75.07

LSUN

PEFT† 78.16 86.32 65.86 75.45 PEFT‡ 81.29 88.45 70.49 69.50

OE 75.68 60.99 86.51 83.55 OCL 77.81 62.80 88.39 79.97

Place365

PEFT† 84.65 71.67 93.00 58.36 PEFT‡ 86.04 74.25 93.65 55.43

OE 73.52 75.06 67.27 86.30 OCL 73.56 74.12 69.65 81.93

Average

PEFT† 83.77 82.90 81.01 60.43 PEFT‡ 86.51 85.99 84.12 54.60

### 3. Theoretical Motivation and Observations

Current LTSSL methods predominantly rely on training deep models from scratch. While effective to a degree, this paradigm suffers from a “vicious cycle”: the scarcity of tail samples leads to high generalization error, which induces severe overconfidence and high Balanced Posterior Error (BPE), ultimately resulting in unreliable pseudo-labels that mislead the self-training process.

In this section, we provide a unified theoretical framework to explain how fine-tuning FMs via PEFT fundamentally breaks this cycle. We structure our analysis into three logical steps:

1. We first introduce a generalization bound based on hypothesis complexity (Lemma 3.1), proving that limiting the search space via PEFT can mathematically compensate for the scarcity of tail samples.

- 2. We then bridge this bound to the BPE (Theorem 3.2), demonstrating that better generalization directly translates to lower worst-case risk and more reliable pseudolabels.
- 3. Finally, we extend the analysis to Open-World scenarios (Proposition 3.3), showing that the geometric compactness of pre-trained features inherently facilitates the rejection of OOD noise.

These theoretical insights are further validated by empirical observations, which collectively inspire the design of LoFT and LoFT-OW. The detailed proofs are in the Appendix.

#### 3.1. Theoretical Analysis

Here, we formally compare the learning dynamics of a model trained from scratch, denoted as hscr, versus a model

###### Construct pseudo-label

Hard pseudo-label

[Figure 18]

[Figure 19]

Weakly Augmented

Frozen Finetune Stop Gradient

Confidence threshold

[Figure 20]

###### OOD Detection

Visual Encoder Classifier

[Figure 21]

Image Features

Soft pseudo-label

[Figure 22]

OOD threshold

Visual Encoder

OOD threshold

[Figure 23]

Strongly Augmented

H(p, q)

[Figure 24]

prediction logits

Classnames

Text Encoder

Visual Encoder Classifier

[Figure 25]

[Figure 26]

Text Features

Figure 3. Illustration of the proposed LoFT-OW. H(p, q) denotes the cross-entropy.

initialized from pre-training and fine-tuned via PEFT, denoted as hpeft. Let DS (with size NS) and DU denote the distributions of the labeled source set and the unlabeled target set, respectively. Our analysis proceeds by examining how the Hypothesis Complexity determines the upper bound of generalization error, and how this bound dictates the reliability of the model under long-tailed and open-world settings.

Generalization Error and Hypothesis Complexity Standard generalization bounds depend on both the sample size and the complexity of the hypothesis class. For long-tailed

data, the sample size NS(y) for a tail class y is negligible, causing the error bound for scratch models (which have

high complexity) to become vacuously loose. The following lemma illustrates that PEFT tightens this bound by constraining the hypothesis space, effectively compensating for the lack of data with prior knowledge.

Lemma 3.1 (Generalization Bound via Complexity Reduction). Let R(H) denote the Rademacher complexity of the hypothesis class. The generalization error RS(h) is bounded with probability at least 1 − δ.

For Training from Scratch, the model searches a vast hypothesis space Hscr (e.g., all possible CNN weights). The bound for a specific class y is dominated by the ratio of high complexity to scarce samples:

 R(Hscr)

 . (1)

RS(hscr|y) ≤ RˆS(hscr|y) + O

NS(y)

For tail classes where NS(y) → 0, the uncertainty term explodes.

In contrast, PEFT constrains the search to a significantly smaller subspace Hpeft (e.g., only head or adapter parameters), conditioned on robust pre-trained features. This dras-

tically reduces the complexity term R(Hpeft) ≪ R(Hscr).

 ,

 R(Hpeft)

RS(hpeft|y) ≤ RˆS(hpeft|y)+ϵtrans+O

NS(y)

(2) where ϵtrans is the transfer error (assumed small for FMs). Even with small NS(y), the significantly reduced numerator ensures a tight bound, effectively compensating for sample scarcity.

Bridging Generalization to BPE Having established that PEFT yields a tighter generalization bound, we now connect this result to the core challenge of LTSSL: the BPE (Wei et al., 2024). The following theorem explains that by minimizing the worst-case generalization error across classes, PEFT specifically prevents the “collapse” of performance on tail classes.

Theorem 3.2 (Superiority of PEFT on BPE). Assuming the transfer error ϵtrans is negligible compared to the complexity gap, the PEFT guarantees a lower upper bound on the BPE compared to training from scratch.

Proof Sketch. The BPE is explicitly determined by the worstcase class-conditional risk: BPE(h) = maxy RS(h|y). By Lemma 3.1, for tail classes, the risk upper bound for hscr is loose due to the high R(Hscr). In contrast, hpeft enforces a tight bound for all classes y by minimizing the hypothesis complexity. Consequently, the maximum risk over all classes is strictly reduced:

RS(hpeft|y) ≪ max

RS(hscr|y) =⇒ BPE(hpeft) ≪ BPE(hscr).

max

y

y

(3)

This lower BPE ensures more reliable pseudo-labels for tail classes, establishing a virtuous cycle for self-training.

Theoretical Insight for Open-World Scenarios Finally, beyond the closed-set error, we analyze the capability to distinguish OOD samples. We justify the robustness of

PEFT via the geometry of the feature space, showing that a compact feature distribution is mathematically equivalent to a stronger rejection capability against open-world noise.

Proposition 3.3 (OOD Robustness via Feature Concentration). Let the feature space be the unit hypersphere Sd−1. For a normalized linear classifier (equivalent to a nearestcentroid classifier), the acceptance region for class k is a spherical cap defined by an angular threshold θk.

Models trained from scratch typically suffer from feature scattering, resulting in loose decision boundaries (large θkscr). The probability of a random OOD sample xout falling into this region is proportional to the volume of the spherical cap. By the concentration of measure on the sphere (Vershynin, 2018):

d 2

P(f(xout) ∈ Cap(θk)) ≤ exp −

cos2 θk . (4)

FMs trained with contrastive objectives (like CLIP) exhibit Feature Compactness (Wang & Isola, 2020), enforcing tight intra-class clustering (small θkloft). Since the volume decays exponentially with cos2 θ, a small reduction in angular spread leads to a massive reduction in OOD acceptance probability:

P(f(xout) ∈ Cap(θkpeft)) ≪ P(f(xout) ∈ Cap(θkscr)).

(5) This geometric property allows FMs to effectively filter OOD noise using confidence thresholding.

#### 3.2. Empirical Observations

To substantiate the theoretical claims, we provide empirical evidence focusing on two key aspects: confidence calibration (corroborating the reduced BPE in Theorem 3.2) and open-world robustness (verifying the geometric compactness in Proposition 3.3).

We attribute this improvement to the extensive pretraining of FMs on large-scale data, which reduces model uncertainty and enhances calibration. Additionally, PEFT modifies only a small subset of parameters, thereby preserving the generalization capabilities of the FMs while effectively adapting to the target task.

Open-World Robustness. As shown in Tab. 1, we fine-tune the FMs on CIFAR-100-LT and evaluate its performance on a variety of OOD datasets, including SVHN (Goodfellow et al., 2013), CIFAR-10 (Krizhevsky et al., 2009), Tiny ImageNet (Le & Yang, 2015), LSUN (Yu et al., 2015), and Places365 (Zhou et al., 2017). We adopt the Maximum Softmax Probability (MSP) (Hendrycks & Gimpel, 2016) as the OOD detection strategy and compare our approach against baseline methods, including OE (Hendrycks et al., 2018) and OCL (Miao et al., 2024). Across multiple evaluation metrics, the model fine-tuned on OpenCLIP achieves the best overall performance, with an average score of 86.51 across the six datasets. These results validate Proposition 3.3: the compact feature space of FMs create tighter spherical acceptance regions, natively filtering out OOD noise.

### 4. Method

Building on the analysis in Sec. 3, we propose LoFT and its open-world extension, LoFT-OW. The theoretical guarantee of reduced BPE motivates LoFT to leverage the superior calibration of FMs to employ a confidence-aware self-training strategy, leveraging improved calibration to assign reliable hard and soft pseudo-labels. Meanwhile, the geometric compactness of pre-trained features inspires LoFT-OW, which utilizes a dual-stage filtering mechanism to effectively reject OOD samples. The details are formulated below.

Confidence Calibration. Calibration serves as a proxy for validating the BPE reduction. According to Theorem 3.2, pre-training should suppress the conditional risk on tail classes by reducing hypothesis complexity. As shown in Fig. 2, we visualize the confidence–accuracy diagram on ImageNet-LT and Places365-LT. Following previous works (Liu et al., 2019), we divide the classes into three groups, “Many”, “Medium”, and “Few”, based on the number of training samples per class. We observe that models trained from scratch tend to exhibit significant overconfidence on the unseen test set, particularly for the tail classes. Specifically, the scratch-trained model yields an ECE of 0.1372 across the entire dataset. Moreover, the tail classes suffer from more pronounced overconfidence compared to head classes. In contrast, models fine-tuned using PEFT demonstrate substantially improved calibration, with tail classes no longer exhibiting such severe overconfidence.

#### 4.1. LoFT

In modern LTSSL, models are typically optimized by jointly minimizing a supervised classification loss on labeled data, used to learn initial discriminative representations, and a regularization loss on unlabeled data, which further refines the learned features and enhances generalization.

For the supervised classification loss, we adopt the Logit Adjustment (Menon et al., 2020) as the criterion on the labeled long-tailed dataset. The optimization objective is:

1 | DS | x∼D

H yb, f W(x) + τ log PS(Y ) ,

Ls =

S

(6) where W(·) denotes a weak augmentation operation (e.g., random crop or horizontal flip), τ is a scaling hyperparameter, and PS(Y ) represents the empirical class prior estimated

from the labeled dataset.

For the regularization loss on unlabeled samples, we follow the basic principle from prior work (Sohn et al., 2020), where a weakly augmented view is used to generate pseudolabels, and a strongly augmented view is used to obtain logits for optimization. To better handle uncertain predictions, we partition unlabeled samples into high-confidence and low-confidence subsets based on their MSP, and apply different optimization strategies accordingly. Specifically, we define a binary mask Mx to indicate whether an unlabeled sample is considered high-confidence, computed as:

1, MSP(x) > cu 0, MSP(x) ≤ cu

(7)

Mx =

The optimization objective for unlabeled samples is:

1 | DU | x∼D

Lu =

λ1Mx · H y, ˆ f(A(x))

U

+ λ2(1 − Mx) · H f(W(x)), f(A(x)) ,

(8) where yˆ = arg maxf(W(x)) denotes the hard pseudolabel derived from the weakly augmented view, and A(·) denotes a strong augmentation. λ1 and λ2 are hyperparameters.

In Eq. 8, for high-confidence samples (Mx = 1), we apply hard pseudo-labels by assigning the most probable class using the model’s prediction. For low-confidence samples (Mx = 0), we apply soft pseudo-labels by leveraging the full predicted probability distribution, which provides smoother supervision and better captures prediction uncertainty. We analyze that, as shown in Fig. 2, under our fine-tuning framework, the model’s confidence score is strongly correlated with prediction accuracy. Since highconfidence samples are generally more reliable, we apply hard supervision to them, while soft supervision is used for low-confidence samples to mitigate overfitting and enhance generalization. Furthermore, as discussed previously, our fine-tuned model exhibits better calibration for tail classes compared to models trained from scratch. Consequently, we do not distinguish between head and tail classes when determining the confidence mask in Eq. 7, e.g., setting different thresholds for head or tail classes, which also reduces the number of required hyper-parameters. Finally, the overall training objective is:

L = Ls + Lu (9)

#### 4.2. LoFT-OW (LoFT under Open-World scenarios)

Traditional LTSSL methods typically assume that all unlabeled data originates from the same distribution as the labeled data—a condition that rarely holds in real-world scenarios. In practice, unlabeled data are often collected

from broad, unconstrained sources such as the web or dynamic field environments, where it is highly likely that a substantial portion of samples lie outside the distribution of the predefined labeled classes. These OOD samples, if not properly handled, can degrade model performance by introducing misleading supervision. To address this challenge, we propose an extension of our framework to open-world settings, termed LoFT-OW (LoFT under Open-World scenarios). LoFT-OW is designed to effectively detect and filter out OOD samples during training, thereby mitigating their adverse effects and enhancing performance in long-tailed, semi-supervised learning.

As shown in Fig. 3, we adopt a two-stage filtering strategy to identify OOD samples. In the first stage, we employ a zero-shot filtering mechanism, where the foundation model assigns confidence scores to each unlabeled sample. Only those with confidence exceeding a high-confidence threshold tHC are retained, resulting in a cleaner and more reliable pseudo-labeled subset, denoted as DU. This filtered dataset is typically smaller in size and can be leveraged for subsequent fine-tuning. Beyond this initial stage, we further exploit the strong OOD detection capability of the fine-tuned model, which has been verified previously. We define the filtering function as follows:

Mxood =

1, MSP(x) > cood 0, MSP(x) ≤ cood

, (10)

where cood is the hyper-parameter to control the filtering strength. Then the optimization object for the unlabeled set under open-world scenarios is:

1 | DU | x∼ D

λ1MxoodMx · H y, ˆ f(A(x))

Lu =

U

+ λ2Mxood(1 − Mx) · H f(W(x)), f(A(x)) ,

(11)

### 5. Experiments

#### 5.1. Experimental Setup

To validate the efficacy of our method under long-tailed distributions and in open-world semi-supervised learning scenarios, we conduct experiments on two long-tailed benchmarks: CIFAR-100-LT (Cui et al., 2019), ImageNet127 (Wei et al., 2021). For ImageNet-127, we only use 10% of the unlabeled data compared with ACR. For CIFAR100-LT, let Nk denote the number of labeled samples for class k, with N1 ≥ N2 ≥ ··· ≥ NK. The imbalance ratio of the labeled dataset is defined as γl = N

NC . Similarly, let Mc denote the number of unlabeled samples for class c, and the imbalance ratio of the unlabeled dataset is defined as γu = max

1

c Mc

minc Mc , without assuming any specific class

- Table 2. The accuracy results on CIFAR-100-LT with different hyper-parameters of γu and γl. PEFT refers to the fine-tuning method of LoFT using only supervised data that demonstrates the capabilities of the foundation model when using only labeled data. The comparison proves the performance improvement achieved by utilizing unlabeled data within the LoFT and LoFT-OW framework.

Method

γ = γl = γu = 10 γ = γl = γu = 20 γu = 1 (uniform) γu = 1/10 (reversed) N1 = 50 N1 = 150 N1 = 50 N1 = 150 N1 = 50 N1 = 150 N1 = 50 N1 = 150

M1 = 400 M1 = 300 M1 = 400 M1 = 300 M1 = 400 M1 = 300 MC = 400 MC = 300

FixMatch 45.2 56.5 40.0 50.7 45.5 58.1 44.2 57.3 +ACR 55.7 65.6 48.0 58.9 66.0 73.4 57.0 67.6 +ACR+BEM 55.8 66.3 48.6 59.8 - - - -

+TCBC - 59.4 - 53.9 - 63.2 - 59.9 +CPE 50.3 59.8 43.8 55.6 - - - 60.8 +CCL 53.5 63.5 46.8 57.5 59.8 67.9 54.4 64.7

CLIP

PEFT 75.5 79.7 74.0 78.4 75.5 79.7 75.5 79.7 LoFT 78.8 81.1 75.3 79.3 78.0 81.0 77.3 80.6

LoFT-OW 76.5 79.9 73.6 78.6 76.6 80.0 76.4 80.0

OpenCLIP

PEFT 78.0 81.7 75.3 81.1 78.0 81.7 78.0 81.7 LoFT 81.8 83.2 78.4 81.2 80.3 83.6 79.8 82.3

LoFT-OW 79.3 81.6 75.4 80.8 78.6 82.1 79.7 82.0

- Table 3. The results on ImageNet-127. PEFT refers to the finetuning method of LoFT using only supervised data.

ods, ACR (Wei & Gan, 2023), ACR+BEM (Zheng et al., 2024), TCBC (Li et al., 2024), CPE (Ma et al., 2024), and CCL (Zhou et al., 2024). To ensure a comprehensive evaluation, we validate our method on two foundation model backbones: CLIP (Radford et al., 2021) and OpenCLIP(Cherti et al., 2023), assessing its robustness and generalizability. All experiments are performed on a single NVIDIA A40 GPU. More hyper-parameter settings of our method are in the Appendix.

Method training iterations Accuracy

FixMatch 250000 42.3 +BEM 250000 58.2 +ACR 250000 63.6 +ACR+BEM 250000 63.9 +CCL 250000 67.8

PEFT 10000 71.7 LoFT 10000 73.3

CLIP

- LoFT-OW 10000 73.1

OpenCLIP

PEFT 10000 72.5 LoFT 10000 73.9

- LoFT-OW 10000 74.2

#### 5.2. Results on LoFT

CIFAR-100-LT As shown in Tab. 2, LoFT consistently outperforms PEFT across all settings on CIFAR-100-LT, using both CLIP and OpenCLIP backbones. With OpenCLIP, LoFT achieves the best results in all cases (up to 83.2%), demonstrating its effectiveness. In terms of imbalance levels, LoFT performs well under all γ values. Performance slightly decreases as γ increases (e.g., from γ = 10 to γ = 20), indicating increased difficulty with more severe imbalance, but LoFT still maintains a clear margin over PEFT. Moreover, LoFT remains robust under uniform and reversed unlabeled distributions (γu = 1 and 1/10), further validating its ability to handle various class distributions.

distribution. We consider three representative settings:

- • Consistent: M1 ≥ M2 ≥ ··· ≥ MC with γu = γl.
- • Uniform: M1 = M2 = ··· = MC, i.e., γu = 1.
- • Reversed: M1 ≤ M2 ≤ ··· ≤ MC, i.e., γu = 1/γl.

To simulate the open-world setting, we introduce the COCO (Lin et al., 2014) dataset as OOD source. COCO contains a diverse set of object categories that are semantically disjoint from those in the target classification task, making it a suitable candidate for evaluating OOD robustness. We mix the COCO dataset with the current unlabeled set to form a more realistic and challenging unlabeled pool, which better reflects the distributional uncertainty encountered in open-world scenarios. We set tHC = 0.95 for all datasets.

ImageNet-127 As shown in Tab. 3, our method outperforms other methods on a large-scale long-tailed dataset, demonstrating the strong generalization ability of LoFT. Compared to PEFT, LoFT consistently achieves higher accuracy with both CLIP and OpenCLIP backbones, reaching 73.3% and 73.9%, respectively. These improvements over strong baselines and prior methods (e.g., FixMatch+CCL at 67.8%) highlight LoFT’s effectiveness beyond small-scale

We compare LoFT and LoFT-OW with FixMatch (Sohn et al., 2020), as well as equiped with different meth-

datasets, confirming its robustness and scalability in realworld, large-scale LTSSL scenarios. Moreover, we visualize the unlabeled samples and their prediction scores, as shown in Fig. 4. For samples containing meaningful content within the label space, LoFT-OW generates reliable pseudo-labels. In contrast, for uninformative OOD samples, LoFT-OW assigns low confidence scores, facilitating their detection.

#### 5.3. Results on LoFT-OW

As shown in Tab. 2 and Tab. 3, LoFT-OW achieves strong performance on both CIFAR-100-LT and ImageNet-127, with fewer training iterations and less data. While its accuracy on CIFAR-100-LT is slightly lower than LoFT due to the inclusion of OOD unlabeled data, which introduces distributional shifts and may hinder representation learning, LoFT-OW remains competitive across all imbalance settings. Notably, on the larger and more complex ImageNet127 dataset, LoFT-OW outperforms all baselines, including LoFT, demonstrating its superior scalability. This highlights the effectiveness of LoFT-OW in leveraging OOD data when generalizing to more diverse and large-scale benchmarks.

#### 5.4. Sensitivity Analysis

We perform two experiments on the CIFAR-100-LT benchmark (N = 50, M = 400, imbalance ratio = 10), using CLIP

- as our foundation model.

Effect of the hyper-parameter cu The hyper-parameter cu controls the balance between hard and soft pseudo-label assignments. With a large value of cu, more unlabeled samples are assigned hard pseudo-labels, encouraging confident and deterministic supervision but potentially introducing noise if the predictions are incorrect. In contrast, a smaller value of cu leads to a greater proportion of soft pseudo-labels, which provides more nuanced guidance by preserving model uncertainty, thereby reducing the risk of reinforcing incorrect predictions. As shown in Fig. ??, the test accuracy rises from 74.0% at cu = 0.2 to a maximum of 78.8 % at cu = 0.6, then declines to 75.3% at cu = 0.95. This behavior indicates that a moderate confidence cutoff best balances the benefit of incorporating pseudo-labels with the risk of introducing erroneous predictions.

Effect of the hyper-parameter cood The hyper-parameter cood controls the sensitivity of OOD detection among unlabeled samples. A larger value of cood enforces stricter filtering, ensuring higher quality among the retained samples but resulting in fewer valid pseudo-labeled instances. Conversely, a smaller cood allows more samples to pass the filter, increasing quantity but potentially compromising quality due to the inclusion of OOD data. Fig. ?? shows that accuracy improves from 75.6% at cood = 0.1 to 76.5%

- at cood = 0.6 before falling to 75.2% at cood = 0.7. These

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Bird 0.9797 Athlete 0.9980 Mammal 0.9997 OOD 0.1297

- Figure 4. Visualizations of unlabeled samples and their predicted confidence scores on ImageNet-127. Samples with a green background are assigned reliable pseudo-labels with high confidence, while the sample with a red background is identified as an OOD instance.

[Figure 31]

[Figure 32]

- Figure 5. Ablation studies on hyper-parameters cu (top) and cood (bottom). The horizontal axes represent the values of the respective hyper-parameters, and the vertical axes represent the accuracy.

results suggest that a moderate OOD cutoff effectively excludes OOD samples without discarding too much valuable unlabeled data. Combined with previous experiments, both cu and cood present the optimal result at the value of 0.6. In the standard LTSSL scenario, cu = 0.6 corresponds to a confidence level high enough to regard predictions as reliable pseudo-labels. In the open-world setting, cood = 0.6 similarly acts as a boundary above which samples are very likely to be in-distribution, thus improving data filtering.

- 6. Conclusion

In this work, we propose LoFT to address the limitations of training-from-scratch paradigms in LTSSL. Theoretically, we show that fine-tuning FMs reduces the BPE and enforces feature compactness, which strictly compresses the acceptance region for OOD samples. Guided by these insights, we introduce LoFT-OW, utilizing a dual-stage filtering mechanism for open-world scenarios. Extensive experiments demonstrate that our framework achieves state-of-the-art performance and superior robustness, establishing a new standard for robust imbalanced learning.

### Impact Statements

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Bartlett, P. L. and Mendelson, S. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of machine learning research, 3(Nov):463–482, 2002.

Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818–2829, 2023.

Cui, Y., Jia, M., Lin, T.-Y., Song, Y., and Belongie, S. Classbalanced loss based on effective number of samples. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9268–9277, 2019.

Dong, B., Zhou, P., Yan, S., and Zuo, W. Lpt: Long-tailed prompt tuning for image classification. arXiv preprint arXiv:2210.01033, 2022.

Goodfellow, I. J., Bulatov, Y., Ibarz, J., Arnoud, S., and Shet, V. Multi-digit number recognition from street view imagery using deep convolutional neural networks. arXiv preprint arXiv:1312.6082, 2013.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. In International conference on machine learning, pp. 1321–1330. PMLR, 2017.

Hendrycks, D. and Gimpel, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136, 2016.

Hendrycks, D., Mazeika, M., and Dietterich, T. Deep anomaly detection with outlier exposure. arXiv preprint arXiv:1812.04606, 2018.

Hou, Y. and Jia, Y. A square peg in a square hole: Metaexpert for long-tailed semi-supervised learning. arXiv preprint arXiv:2505.16341, 2025.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pp. 448– 456. pmlr, 2015.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

Le, Y. and Yang, X. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.

Li, L., Tao, B., Han, L., Zhan, D.-c., and Ye, H.-j. Twice class bias correction for imbalanced semi-supervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 13563–13571, 2024.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Liu, K., Fu, Z., Jin, S., Chen, C., Chen, Z., Jiang, R., Zhou, F., Chen, Y., and Ye, J. Rethinking out-of-distribution detection on imbalanced data distribution. Advances in Neural Information Processing Systems, 37:109152–109176, 2024.

Liu, Z., Miao, Z., Zhan, X., Wang, J., Gong, B., and Yu, S. X. Large-scale long-tailed recognition in an open world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2537–2546, 2019.

Ma, C., Elezi, I., Deng, J., Dong, W., and Xu, C. Three heads are better than one: Complementary experts for longtailed semi-supervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 14229–14237, 2024.

Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., and Kumar, S. Long-tail learning via logit adjustment. arXiv preprint arXiv:2007.07314, 2020.

Miao, W., Pang, G., Bai, X., Li, T., and Zheng, J. Outof-distribution detection in long-tailed recognition with calibrated outlier class learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 4216–4224, 2024.

Ouali, Y., Hudelot, C., and Tami, M. An overview of deep semi-supervised learning. arXiv preprint arXiv:2006.05278, 2020.

Peng, H., Pian, W., Sun, M., and Li, P. Dynamic reweighting for long-tailed semi-supervised learning. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 6464–6474, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Sanchez Aimar, E., Helgesen, N., Xu, Y., Kuhlmann, M., and Felsberg, M. Flexible distribution alignment: Towards long-tailed semi-supervised learning with proper

calibration. In European Conference on Computer Vision, pp. 307–327. Springer, 2024.

Shi, J.-X., Wei, T., Zhou, Z., Shao, J.-J., Han, X.-Y., and Li, Y.-F. Long-tail learning with foundation model: Heavy fine-tuning hurts. In Forty-first International Conference on Machine Learning, 2024.

Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C. A., Cubuk, E. D., Kurakin, A., and Li, C.-L. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems, 33:596–608, 2020.

Tian, C., Wang, W., Zhu, X., Dai, J., and Qiao, Y. Vl-ltr: Learning class-wise visual-linguistic representation for long-tailed visual recognition. In European Conference on Computer Vision, pp. 73–91. Springer, 2022.

Tomani, C., Gruber, S., Erdem, M. E., Cremers, D., and Buettner, F. Post-hoc uncertainty calibration for domain drift scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10124–10132, June 2021.

Vershynin, R. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

Wang, T. and Isola, P. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning, pp. 9929–9939. PMLR, 2020.

Wei, C., Sohn, K., Mellina, C., Yuille, A., and Yang, F. Crest: A class-rebalancing self-training framework for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10857–10866, 2021.

Wei, T. and Gan, K. Towards realistic long-tailed semisupervised learning: Consistency is all you need. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3469–3478, June 2023.

Wei, T., Mao, Z., Zhou, Z.-H., Wan, Y., and Zhang, M.L. Learning label shift correction for test-agnostic long-tailed recognition, 2024. URL https:// openreview.net/forum?id=u1yvEwYfK9.

Xu, Z., Chai, Z., and Yuan, C. Towards calibrated model for long-tailed visual recognition from prior perspective. Advances in Neural Information Processing Systems, 34: 7139–7152, 2021.

Yang, Y. and Xu, Z. Rethinking the value of labels for improving class-imbalanced learning. Advances in neural information processing systems, 33:19290–19301, 2020.

Yu, F., Seff, A., Zhang, Y., Song, S., Funkhouser, T., and Xiao, J. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.

Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.

Zheng, H., Zhou, L., Li, H., Su, J., Wei, X., and Xu, X. Bem: Balanced and entropy-based mix for long-tailed semisupervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22893–22903, 2024.

Zhong, Z., Cui, J., Liu, S., and Jia, J. Improving calibration for long-tailed recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16489–16498, 2021.

Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

Zhou, Z.-H., Fang, S., Zhou, Z.-J., Wei, T., Wan, Y., and Zhang, M.-L. Continuous contrastive learning for longtailed semi-supervised recognition. Advances in Neural Information Processing Systems, 37:51411–51435, 2024.

### A. Detailed Theoretical Proofs

In this section, we provide the detailed mathematical formulations, assumptions, and proofs for the theoretical claims presented in Section 3 of the main paper.

- A.1. Generalization Analysis (Lemma 3.1) We analyze the generalization error through the lens of Rademacher Complexity.

- A.1.1. PRELIMINARIES AND DEFINITIONS

Let X be the input space and Y = {1,...,K} be the label space. A hypothesis h : X → RK belongs to a hypothesis class H. We consider the standard supervised learning setting with a loss function ℓ : Y × RK → [0,1] (e.g., bounded cross-entropy or 0-1 loss).

Definition A.1 (Rademacher Complexity). Let S = {x1,...,xN} be a sample of size N drawn i.i.d. from distribution D. The empirical Rademacher complexity of a hypothesis class H is defined as:

RˆS(H) = Eσ sup h∈H

1 N

N

i=1

σiℓ(h(xi),yi) , (12)

where σi are independent Rademacher variables taking values {−1,+1} with equal probability. Theorem A.2 (Generalization Bound via Rademacher Complexity (Bartlett & Mendelson, 2002)). For any δ > 0, with probability at least 1 − δ over the draw of sample S of size N, for all h ∈ H:

R(h) ≤ RˆS(h) + 2RˆS(H) + 3

ln(2/δ) 2N

, (13)

where R(h) = E[ℓ(h(x),y)] is the expected risk and RˆS(h) is the empirical risk.

- A.1.2. PROOF OF LEMMA 3.1 Assumptions on Hypothesis Spaces.

- • Let Hscr represent the hypothesis space of training a deep neural network from scratch. This involves optimizing all parameters W ∈ RD, where D is very large.
- • Let Hpeft represent the hypothesis space of Fine-tuning a Foundation Model (FM) via PEFT (e.g., LoRA or Adapter). Here, the backbone weights θpre are frozen, and only a small set of parameters ϕ ∈ Rd (d ≪ D) are optimized.
- • Consequently, we assume Hpeft ⊂ Hscr (conceptually), or strictly that the effective capacity satisfies RˆS(Hpeft) ≪ RˆS(Hscr).

Class-Conditional Bound. In Long-Tailed Learning, we analyze the risk for a specific class y. Let Sy be the subset of samples belonging to class y, with size Ny = |Sy|. Applying Theorem A.2 conditionally on class y:

#### 1. Case: Training from Scratch (hscr ∈ Hscr)

ln(2/δ) 2Ny

(hscr|y) + 2RˆS

R(hscr|y) ≤ RˆSy

. (14)

(Hscr) Complexity Term

+ 3

y

Sample Size Term

For tail classes, Ny → 0. Since Hscr is a deep network with massive capacity, RˆS

(Hscr) is large (typically scaling with the spectral norms of weight matrices and depth). The ratio R(H

y

√ scr) Ny dominates the bound, making it vacuously loose.

##### 2. Case: PEFT (hpeft ∈ Hpeft) Since hpeft is constrained to a neighborhood of the pre-trained weights, we decompose its risk. Note that the optimal risk achievable in Hpeft might be slightly higher than Hscr due to the restricted search space.

We define the Transfer Error (approximation gap) as ϵtrans ≈ minh∈H

peft R(h) − minh′∈Hscr R(h′). However, for the specific hypothesis hpeft obtained via training, the bound is:

(hpeft|y) + 2RˆS

R(hpeft|y) ≤ RˆSy

(Hpeft) + 3

y

ln(2/δ) 2Ny

. (15)

Crucially, since PEFT only optimizes a few parameters (or low-rank matrices), the complexity term is drastically reduced: RˆS

(Hpeft) ≪ RˆS

(Hscr). Even if Ny is small, the low numerator keeps the bound tight.

y

y

| |
|---|

#### A.2. BPE Analysis (Proposition 3.2)

Definition A.3 (Balanced Posterior Error (BPE)). Following (Wei et al., 2024), BPE measures the worst-case error across classes. For theoretical analysis, we use the surrogate of worst-case class-conditional risk:

BPE(h) ≈ max

R(h|y). (16)

y∈Y

Proof. Let Ytail be the set of tail classes and Yhead be the set of head classes.

- 1. For hscr: In head classes, Ny is large, so the bound is tight and risk is low. In tail classes (y ∈ Ytail), Ny is small and R(Hscr) is large. The generalization gap explodes, leading to high expected risk (overfitting). Thus, maxy R(hscr|y) is determined by the tail classes.
- 2. For hpeft: The complexity R(Hpeft) is small for all classes. Provided the Foundation Model features are robust

(meaning RˆSy

(hpeft) can be minimized during training), the upper bound remains low even for y ∈ Ytail.

Comparing the worst-case scenarios:

BPE(hscr) = max

y

BPE(hpeft) = max

y

R ˆ(hscr|y) + O

R(Hscr) Ny ≈ Large (dominated by tail),

R ˆ(hpeft|y) + O

R(Hpeft) Ny ≈ Small.

(17)

Thus, BPE(hpeft) < BPE(hscr).

| |
|---|

#### A.3. OOD Robustness Analysis (Proposition 3.3)

We model the feature space as a unit hypersphere Sd−1, which is a common assumption for normalized embeddings (e.g., Cosine Classifier, CLIP features).

- A.3.1. GEOMETRIC SETUP

- • Let µk ∈ Sd−1 be the prototype (centroid) for class k.
- • A sample x is classified as class k if cos(x,µk) ≥ tk, where tk = cosθk is the decision threshold.
- • The acceptance region is a Spherical Cap: Cap(µk,θk) = {x ∈ Sd−1 | ⟨x,µk⟩ ≥ cosθk}.

- A.3.2. CONCENTRATION OF MEASURE

We assume OOD samples xout are uniformly distributed on the sphere (maximum entropy assumption for unknown noise). The probability of an OOD sample being falsely accepted by class k is the ratio of the cap area to the sphere area.

ImageNet-LT ImageNet-r ImageNet-sketch ImageNet-v2

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

CLIP

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

OpenCLIP

Figure 6. Zero-shot confidence–accuracy curves across multiple datasets.

Lemma A.4 (Concentration on the Sphere (Vershynin, 2018)). For any vector µ on the unit sphere Sd−1 and any ϵ ∈ (0,1):

dϵ2 2

. (18)

P(|⟨xout,µ⟩| ≥ ϵ) ≤ 2exp −

Considering only the positive direction (the cap), for threshold tk = cosθk:

dcos2 θk 2

P(xout ∈ Cap(µk,θk)) ≤ exp −

. (19)

- A.3.3. COMPARISON: SCRATCH VS. FM+PEFT

- 1. Scratch Model (hscr): Models trained from scratch on long-tailed data suffer from loose decision boundaries for tail classes due to lack of negative sampling constraints. This implies a large angular acceptance θkscr (small cosθkscr).

Perrscr ∝ exp −

d 2

(cosθkscr)2 . (20)

- 2. FM + PEFT (hpeft): Foundation Models pre-trained with contrastive loss (e.g., InfoNCE) explicitly optimize for alignment (concentration) and uniformity. This results in highly compact intra-class distributions. Fine-tuning with PEFT preserves this geometry. Thus, the angular spread is small: θkpeft ≪ θkscr. This implies cosθkpeft → 1.

Result: Since the probability decays exponentially with the square of the cosine threshold, a smaller angle (larger cosine) leads to a massive reduction in error probability.

P(xout ∈ Cap(θkpeft)) P(xout ∈ Cap(θkscr)) ≈ exp −

d 2

[cos2 θkpeft − cos2 θkscr] ≪ 1. (21) This proves that the geometric compactness of PEFT features inherently rejects OOD noise.

| |
|---|

- B. Code Detailed code is in the supplementary materials.
- C. Experimental Details

All experiments are conducted by fine-tuning both CLIP and OpenCLIP as backbone models. We integrate the AdaptFormer modules into every Transformer block. Stochastic Gradient Descent (SGD) is employed as the optimizer, with an initial learning rate of 0.01 scheduled via cosine annealing.

#### C.1. CIFAR100-LT

For the relatively small CIFAR100-LT dataset, we set the total number of optimization steps to 1,024. The learning rate is updated with a cosine annealing schedule every 32 steps.

- • SSL setting: λ1 = 3.0, λ2 = 0.0, and confidence threshold cu = 0.6.
- • SSL-OW setting: λ1 = 2.0, λ2 = 1.0, cu = 0.95, and out-of-distribution threshold cood = 0.6.

#### C.2. ImageNet-127

Given the large scale of ImageNet-127 and the strong representational power of FMs, we sample only 1% of the training images for our fine-tuning. This still outperforms scratch-trained baselines that use 10% of the data. We set the total number of optimization steps to 10,000, updating the learning rate via cosine annealing every 100 steps.

- • SSL setting: λ1 = 3.0, λ2 = 1.0, and cu = 0.6.
- • SSL-OW setting: λ1 = 2.0, λ2 = 1.0, cu = 0.95, and cood = 0.6.

### D. Additional Confidence Calibration Results

To demonstrate the efficiency of our method in the SSL-OW setting, we first perform OOD filtering on the unlabeled dataset, leveraging the foundation model’s reliable confidence estimates. To validate the accuracy of our zero-shot confidence scores, we further visualize confidence–accuracy curves across several datasets.

Based on the Expected Calibration Error (ECE), we observe that—even in the zero-shot scenario—the foundation model’s confidence estimates remain highly accurate, providing strong justification for using these scores to guide OOD filtering.

### E. Computational Complexity

Our proposed method, LoFT, demonstrates significantly superior efficiency compared to traditional semi-supervised learning frameworks in terms of training costs and resource utilization.

Training Efficiency and Convergence. As summarized in Table 3, LoFT exhibits rapid convergence properties. Specifically, our model reaches optimal performance within only 10,000 iterations (requiring approximately 4 hours). in stark contrast, standard methods such as FixMatch combined with ACR require up to 250,000 iterations (approximately 8 hours) to achieve comparable results. Consequently, LoFT delivers a 2× speedup in total training time while reducing the number of required update steps by a factor of 25.

Parameter Efficiency and FLOPs. We further analyze the computational overhead in terms of model parameters and Floating Point Operations (FLOPs). Utilizing the CLIP-ViT-B/16 as the foundation model, LoFT contains a total of 149.80M parameters; however, it introduces a highly efficient fine-tuning strategy that updates only 0.18M trainable parameters. This is significantly more efficient than the WideResNet-28-8 baseline, which requires updating all 23.40M parameters. Although the large-scale foundation model incurs higher computational costs per forward pass (16.89G FLOPs vs. 3.37G FLOPs for WideResNet), the drastic reduction in total training iterations results in a significantly lower aggregate computational cost. This makes LoFT not only faster but also more computationally economical for practical deployment.

