# No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces

Daniel Marczak12 Simone Magistri3 Sebastian Cygert45 Bartłomiej Twardowski678 Andrew D. Bagdanov3 Joost van de Weijer78

arXiv:2502.04959v3[cs.LG]11Jun2025

## Abstract

Model merging integrates the weights of multiple task-specific models into a single multi-task model. Despite recent interest in the problem, a significant performance gap between the combined and single-task models remains. In this paper, we investigate the key characteristics of task matrices – weight update matrices applied to a pre-trained model – that enable effective merging. We show that alignment between singular components of task-specific and merged matrices strongly correlates with performance improvement over the pre-trained model. Based on this, we propose an isotropic merging framework that flattens the singular value spectrum of task matrices, enhances alignment, and reduces the performance gap. Additionally, we incorporate both common and task-specific subspaces to further improve alignment and performance. Our proposed approach achieves state-of-the-art performance on vision and language tasks across various sets of tasks and model scales. This work advances the understanding of model merging dynamics, offering an effective methodology to merge models without requiring additional training.

## 1. Introduction

Pre-trained models are the foundation of modern machine learning systems (Carion et al., 2020; Radford et al.,

1Warsaw University of Technology, Poland 2IDEAS NCBR, Warsaw, Poland 3Department of Information Engineering, Media Integration and Communication Center (MICC), University of Florence, Italy 4NASK - PIB, National Research Institute, Warsaw, Poland 5Gda´nsk University of Technology, Poland 6IDEAS Research Center, Warsaw, Poland 7Computer Vision Center, Barcelona, Spain 8Department of Computer Science, Universitat Autonoma de Barcelona, Spain. Correspondence to: Daniel Marczak <daniel.marczak.dokt@pw.edu.pl>.

Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

[Figure 1]

Figure 1. Spectrum of singular values for a single layer weight update matrix obtained by merging using Task Arithmetic (top) compared to our approaches: Iso-C (middle) and Iso-CTS (bottom). Task Arithmetic sums the task-specific matrices, which result in a spectrum with a few dominant components. Iso-C instead replaces this spectrum with a uniform one, which results in significant performance improvement. Iso-CTS enhances the common subspace with task-specific subspaces and yields state-of-the-art model merging performance.

2021; Caron et al., 2021; Zhai et al., 2023). In practice, they are typically fine-tuned for specialization on specific tasks (Wortsman et al., 2022b; Ilharco et al., 2022). Recently, a growing body of research has focused on model merging (Li et al., 2023), which combines multiple task-specific experts into a single multi-task model. Many methods have been proposed to improve the effectiveness of model merging by reducing sign conflicts (Yadav et al., 2023), by aligning gradients (Daheim et al., 2024), or through magnitudebased selection (Marczak et al., 2024). However, a significant performance gap between the combined and single-task models remains.

A key insight from Ilharco et al. (2023) is that task vectors, defined as the offset between the flattened fine-tuned weights and the pre-trained checkpoint, from different tasks are typically close to orthogonal. This orthogonality has been seen as a fundamental property enabling effective merging with reduced interference and has inspired works that en-

force the orthogonality by modifying the fine-tuning procedure (Po et al., 2024). Most recently, Stoica et al. (2025) and Gargiulo et al. (2025) have shown that accounting for the structure of the weight update matrix, dubbed task matrix, is a more effective strategy for improving the performance of model merging. In this paper, we investigate precisely what the characteristics of task matrices are that favor effective model merging. Different from previous works, we propose to analyze the alignment between task-specific and merged subspaces.

Specifically, to capture the similarity between task matrices, we propose to investigate the Subspace Alignment Ratio. Through the lens of Singular Value Decomposition, our metric quantifies the similarity between subspaces spanned by the top singular vectors of task matrices. When applied to compare matrices of the merged model to the task-specific ones, this metric strongly correlates with the performance of the merged model on a given task. This allows us to identify the directions amplified by multiple tasks as well as the underrepresented directions that lead to poor performance on corresponding tasks.

Our goal is to design a model merging technique that balances directions in the weight space across different tasks. We achieve this by flattening the singular values spectrum of the merged matrix, making it more uniform. Enforcing a uniform (isotropic) spectrum significantly improves the alignment and performance of the merged model. This simple yet effective adjustment, which requires no changes to the fine-tuning procedure, leads to substantial gains in merging performance (see method Iso-C in Figure 1).

However, tasks with dominant directions of smaller intensity compared to the majority of tasks and whose directions are orthogonal to the common directions may still remain underrepresented, especially when the number of tasks increases. To address this, we enhance isotropic model merging by introducing task-specific subspaces that retain unique task features while preserving shared knowledge. Our approach begins with the top singular values of the common subspace and iteratively replaces the least significant singular vectors with task-specific directions. This strategy allows us to increase the scalability of our merging approach to more tasks (see method Iso-CTS in Figure 1).

The main contributions of this paper are:

- • We show that the alignment between the subspace spanned by the principal directions of the task-specific matrices and that of the merged matrix positively correlates with the performance of the merged model.
- • We demonstrate that applying an isotropic scaling to singular directions of merged task matrices improves the alignment between merged and task-specific matrices. This results in a simple yet highly effective tech-

- nique for model merging that we call Iso-C, which outperforms most baselines.
- • We further enhance our approach by incorporating taskspecific directions into the merged matrix resulting in Iso-CTS, a merging method that achieves state-ofthe-art results, in particular for a large number of tasks.
- • Our methods demonstrate versatility, achieving stateof-the-art on vision and language merging benchmarks for both fully and LoRA fine-tuned models1.

## 2. Related Work

Model merging. Pre-trained models serve as a foundation for expert models specialized in specific downstream tasks (Radford et al., 2021). Recently, model merging has emerged as a promising technique to combine multiple expert models into a single multi-task model. One of the pioneering works in the field, Task Arithmetic (TA) (Ilharco

- et al., 2023), proposed to compute a task vector as a difference between the expert and the pre-trained model and to then aggregate task vectors via scaled addition to create an expert in multiple tasks. The significant performance gap between individual experts and the combined model sparked an abundance of works with the aim of reducing interference when merging models. TIES (Yadav et al., 2023) proposed a novel way to reduce sign conflicts between the parameters of expert models, Model Breadcrumbs (Davari & Belilovsky, 2024) removed outliers from the task vectors, and Consensus Merging (Wang et al., 2024b) removed catastrophic and selfish weights. These methods focused on per-parameter techniques to mitigate the interference, treating each parameter independently.

The aforementioned static merging methods output a single set of multi-task weights which can be used as a drop-in replacement for the pre-trained model. However, a number of recent methods, dubbed dynamic merging, alter the inference procedure to improve the results. Twin-Merging (Lu

- et al., 2024) composes task-specific components at test-time and alters the inference algorithm requiring two forward passes. EMR-Merging (Huang et al., 2024) uses additional per-task parameter masks and rescalers to perform inference. In this paper, we consider static merging exclusively.

Singular Value Decomposition of model weights. While SVD of weight matrices has been primarily used for model compression (Denton et al., 2014; Kim et al., 2016), recently its effectiveness was also identified for fine-tuning of large models. LoRA (Hu et al., 2021) uses SVD to identify the similarities of weight updates between low-rank and fullrank fine-tuning. MiLORA (Wang et al., 2024a) identifies

1The code is available at https://github.com/ danielm1405/iso-merging.

that the bottom singular components correspond to noisy or long-tail information, while the top singular vectors contain important knowledge. Therefore, they propose a fine-tuning approach that updates only the minor singular components of the weight matrix while keeping the top singular components frozen. SVFT (Lingam et al., 2024) computes outer products of its singular vectors and, during fine-tuning updates, only sparse coefficients of these combinations.

SVD for model merging. The structure imposed by SVD was used for model merging in KnOTS (Stoica et al., 2025), which proposes to concatenate the task-specific low-rank adaptation matrices (LoRA) and average the right-singular vectors before SVD reconstruction to obtain the merged weights. The most similar work to us is the parallel work Task Singular Vectors (TSV) (Gargiulo et al., 2025), which measures task interference based on the interaction of singular vectors from different tasks and uses it to increase merging effectiveness. We share the motivation to improve model merging through SVD decomposition. However, while they focus on the orthogonalization of task-specific subspaces to reduce interference, we show that making singular values uniform in a common subspace is a surprisingly powerful method. Further, we show how to combine shared and task-specific subspaces for improved performance.

## 3. Background and Motivation

In this Section, we first describe the general framework of model merging and provide the notation used throughout the rest of the paper. We then motivate our approach via an analysis of the correlation between task similarity and performance improvement of the merged model.

### 3.1. Model Merging

Model merging integrates multiple deep neural network models, each individually trained (i.e. fine-tuned) on distinct tasks starting from the same pre-trained model, into a single merged model. Let θ0 denote the weights of the pre-trained network, and θt denote the fine-tuned weights for task t, with t = 1,...,T, where T is the total number of tasks. We will use the notation θt(l) to identify the weights of layer l for task t and L to denote the total number of layers in a network. The objective of model merging is to find a merging function f, such that the model:

#### θM(ℓ) = f(θ0(ℓ),{θt(ℓ)}Tt=1), ∀ℓ = 1,...,L (1)

is able to perform all tasks on which the individual models θt are trained.

Building upon Task Arithmetic (TA), we define the layerwise task matrix ∆(tℓ) as the difference between the weights

Cosine similarities of task vectors

1.0

Cars

MNIST

MNIST

1.00 0.19 0.08 0.04 0.04 0.04 0.03 0.03

DTD

RESISC45

EuroSAT

SUN397

0.8

GTSRB

SVHN

SVHN

0.19 1.00 0.08 0.04 0.03 0.04 0.02 0.02

NormalizedAccuracyImprovement

GTSRB

0.08 0.08 1.00 0.04 0.04 0.04 0.03 0.03

0.6

EuroSAT

0.04 0.04 0.04 1.00 0.07 0.05 0.03 0.03

RESISC45

- 0.04 0.03 0.04 0.07 1.00 0.05 0.03 0.04
- 0.04 0.04 0.04 0.05 0.05 1.00 0.04 0.04

0.4

DTD

0.2

Cars

- 0.03 0.02 0.03 0.03 0.03 0.04 1.00 0.03
- 0.03 0.02 0.03 0.03 0.04 0.04 0.03 1.00

SUN397

0.0

MNIST SVHNGTSRBEuroSATRESISC45 DTD CarsSUN397

0.36 0.38 0.40 0.42 0.44 Cosine Similarity between ∆i and ∆TA

(a) Cosine similarity between pairs of task vectors.

(b) NAI vs cosine similarity between task and merged vectors.

Figure 2. (a) Tasks vectors are typically close to orthogonal to each other. (b) Models with very different normalized accuracy improvements (NAI) exhibit very close cosine similarities, and the correlation between cosine similarity and NAI is low.

of the model θt and the pre-trained model θ0 for layer ℓ:

∆(tℓ) = θt(ℓ) − θ0(ℓ). (2)

In the rest of the paper, the ℓ superscript is omitted when not relevant to the discussion, and all definitions refer to an arbitrary layer. The authors of Task Arithmetic propose to solve the problem of model merging by defining a merging function that sums all task matrices to the pre-trained model weights:

θTA(ℓ) = θ0(ℓ) + α∆(TAℓ), (3) where α is a scaling factor determined on a held-out validation dataset and ∆(TAℓ) = Tt=1 ∆(tℓ). The advantage of this merging strategy is that it allows for the reuse and transfer of knowledge from many fine-tuned models to the pre-trained model without requiring additional training or access to the original training data (Ilharco et al., 2023).

3.2. Cosine Similarity and Performance Improvement are Uncorrelated

Starting from the definition of Task Arithmetic (TA) in Eq. (3), we aim to explore the possible reasons for the improvement achieved by TA merging over the pre-trained (or zero-shot) model across multiple tasks. To empirically quantify performance gain, we propose the Normalized Accuracy Improvement (NAI) metric, defined as:

Acc(θM) − Acc(θ0) Acc(θt) − Acc(θ0)

NAI(θM,θt;θ0) =

, (4)

which quantifies the improvement of the merged model θM relative to that achieved by the task-specific model θt, both measured with respect to the zero-shot baseline θ0.2

2NAI differs from Normalized Accuracy (Ortiz-Jim´enez et al.,

2023) which does not account for zero-shot performance.

Ilharco et al. (2023) hypothesize that minimal inter-task interference – captured by near-zero cosine similarity between the vectorized representation of the task matrices, i.e., ⟨vec(∆i),vec(∆j)⟩ ≈ 0 for i ̸= j (see Figure 2a) – explains the effectiveness of Task Arithmetic. To investigate this further, we examine whether the cosine similarity between each task vector and the merged Task Arithmetic vector, ⟨vec(∆TA),vec(∆t)⟩, serves as an indicator of performance improvement, as quantified by NAI(θTA,θt;θ0). However, we observe no clear correlation (see Figure 2b), suggesting that cosine similarity alone does not fully explain the observed performance gains. This indicates that the improvement achieved by Task Arithmetic likely originates from other factors, which we unveil below through spectral analysis of the Task Arithmetic and task-specific matrices.

### 3.3. Performance Correlates with Subspace Alignment

We argue that the improvement in Task Arithmetic performance derives from the relationship between the top singular vectors of ∆TA and those of each ∆t. Specifically, we hypothesize that the subspace of ∆TA approximates the union of the subspaces of each ∆t, and that the overlap of this overall subspace with each task matrix correlates with the performance improvement of the merged model.

In order to empirically quantify the overlap between subspaces, we propose the Subspace Alignment Ratio (SAR) metric. We define SAR between a task matrix ∆t and a generic merged task matrix ∆M as:

SAR(∆t,∆M;kM) = ||Πk

M,M∆t||F ||∆t||F

, (5)

where Πk

M,M is the projection matrix onto the subspace spanned by the top kM left-singular vectors of ∆M. The columns of Uk

M,MUk⊤

M,M = Uk

M,M are obtained from the SVD decomposition of ∆M, and the number of singular vectors used (kM) is determined from the merged task matrix ∆M by minimizing the approximation error with ϵ = 0.05:

kM = min k : ∥∆M − Πk,M∆M∥F ≤ ϵ∥∆M∥F

r i=k+1 σi2

r i=1 σi2 ≤ ϵ2 , (6)

= min k :

where Σ = diag(σ1,...,σr) contains the singular values of ∆M, and the equivalence follows from the definition of the Frobenius norm (see Appendix A.1).

SAR quantifies the alignment between the subspaces of two task matrices as a function of the number of dominant singular vectors of the merged matrix. To provide a single score measuring the overlap between two models, we denote with SARavg the Average Subspace Alignment Ratio across all layers.

In Figure 3a (left, represented by stars), we plot the Normalized Accuracy Improvement achieved by TA on each task, given by NAI(θTA,θt;θ0), against the Average Subspace Alignment Ratio of each task matrix ∆t with the merged task matrix ∆TA, i.e. SARavg(∆t,∆TA;kTA). First, we note that the alignment between task and merged matrices are notably high (ranging from 0.75 to 0.87), but vary significantly across datasets. This suggests that task vectors are well represented in the subspace identified by the task-arithmetic matrix but with different degrees of alignment and consistency depending on dataset characteristics. Furthermore, we highlight a strong correlation (Pearson correlation coefficient ρTA = 0.94) between the performance improvement on individual tasks achieved by θTA and the degree of alignment of ∆t with ∆TA.

Analogous to the pairwise cosine similarity analysis between task vectors performed by Ilharco et al. (2023), in Figure 3b we measure the SAR between pairs of task matrices, SARavg(∆i,∆j;kTA), using the kTA dominant components of the merged Task Arithmetic model. Some groups of tasks exhibit higher alignment which is due to their semantic similarity, e.g. MNIST, SVHN, and GTSRB are digit recognition datasets, while EuroSAT and RESISC45 are satellite image datasets. On the other hand, datasets such as Cars, DTD or SUN397 are less aligned to other tasks. Most importantly, tasks belonging to highly aligned groups are also highly aligned with the TA model and achieve the highest accuracy improvements (see Figure 3a). The tasks that are not aligned are underrepresented in the dominant subspace of ∆TA, and the performance on them is low.

Based on the observed correlation between performance and alignment ratio, we hypothesize that a merging method that aims to achieve high alignment will also achieve strong performance. Therefore, in the next Section, we propose an approach called Isotropic Merging that improves alignment and, most importantly, the performance of the merged models.

## 4. Isotropic Merging in Common and Task-specific Subspaces

In this Section, we propose a novel model merging method we call Isotropic Merging in Common and Task-Specific Subspaces (Iso-CTS). First, we introduce Isotropic Merging in Common Subspace (Iso-C), which is able to enhance the normalized accuracy improvement and the alignment of each task matrix using common directions identified by Task Arithmetic. Then, we show how to further enhance the performance of merged models by introducing taskspecific directions to improve merging performance on sets of many diverse tasks.

1.0

0.8

0.6

NAI

0.4

TA

Iso-C

Cars

MNIST

0.2

DTD

RESISC45

EuroSAT

SUN397

GTSRB

SVHN

0.0

0.75 0.80 0.85 0.90 0.95 SARavg

(a) Normalized Accuracy Improvement (NAI) vs. Average Subspace Alignment Ratio (SARavg).

0.99 0.75 0.68 0.58 0.57 0.58 0.58 0.56

MNIST

0.73 0.99 0.68 0.59 0.57 0.58 0.57 0.56

SVHN

0.64 0.66 0.99 0.59 0.58 0.59 0.58 0.58

GTSRB

0.57 0.58 0.59 0.99 0.67 0.62 0.57 0.62

EuroSAT

0.56 0.57 0.58 0.64 0.97 0.61 0.58 0.63

RESISC45

0.56 0.57 0.58 0.60 0.61 0.97 0.58 0.60

DTD

0.55 0.55 0.56 0.56 0.57 0.57 0.96 0.58

Cars

0.54 0.55 0.56 0.58 0.60 0.59 0.58 0.95

SUN397

MNISTSVHNGTSRBEuroSATRESISC45 DTD CarsSUN397

(b) Average Subspace Alignment Ratios (SARavg) between pairs of task matrices.

- Figure 3. (a) NAI strongly correlates with SARavg (Pearson correlation coefficient ρTA = 0.94). (b) Note the groups of highly aligned tasks such as {MNIST, SVHN, GTSRB} and {EuroSAT, RESISC45}. By comparing (b) and (a), the mutually aligned datasets exhibit higher alignment with the merged model and consequently achieve good performance. On the other hand, tasks with low mutual alignment, such as DTD, Cars, and SUN397, are less aligned with the merged model and achieve poor performance.

### 4.1. Isotropic Merging in Common Subspace

In Section 3.3, we demonstrated the high alignment of each task matrix with the matrix obtained by Task Arithmetic. This alignment indicates that the span of dominant singular vectors of the merged matrix effectively covers the subspace of each task and provides a good approximation of the common subspace. However, significant variability in the average alignment ratio across the dataset leads to a lower accuracy improvement for less aligned tasks compared to the tasks belonging to groups with high alignment. This variability stems from the skewness of the task arithmetic spectrum (Figure 1 and 12), which is concentrated in the first few singular values (which we call top or dominant), favoring the tasks from the highly aligned groups. Our proposed methodology, which we call Isotropic Merging in Common Subspace (Iso-C), aims to equalize the spectrum of the task arithmetic matrix in order to enhance the average subspace alignment ratio and ensure a more balanced representation across tasks in the merged model.

Consider the sum of task matrices ∆TA = t ∆t, where ∆t ∈ Rm×n. Via Singular Value Decomposition (SVD) on ∆TA we obtain ∆TA = UΣV ⊤, where U ∈ Rm×r and V ∈ Rn×r represent, respectively, the left and right singular vectors of ∆TA, and Σ ∈ Rr×r is the diagonal matrix containing the singular values. We denote the vector of singular values by σ = diag(Σ) ∈ Rr.

Algorithm 1 Iso-C: Isotropic Merging in Common Subspace

Require: Task matrices ∆1,...,∆T with ∆t ∈ Rm×n

- 1: Sum task matrices: ∆TA = Tt=1 ∆t
- 2: Compute the SVD of ∆TA: ∆TA = UΣV ⊤, with U ∈ Rm×r,Σ ∈ Rr×r,V ∈ Rn×r,σ = diag(Σ)∈ Rr
- 3: Calculate isotropic factor: σ = 1r ri=1 σi (Eq.7)

- 4: Reconstruct the matrix: ∆Iso-C = σUV ⊤ (Eq.8)

- 5: return ∆Iso-C

To reduce the skewness towards the dominant singular vectors of ∆TA, we propose scaling all directions of the transformation applied by the right-singular vectors V to a fixed value rather than using their corresponding singular values. This ensures that the final transformation is isotropic, with the scaling factor set to the average singular value:

1 r

σ =

r

σi, (7)

i=1

and merged matrix is computed using the reconstruction:

#### ∆Iso-C = σUV ⊤. (8)

We apply this operation to all network layers, and the final merged model is defined as:

θIso-C(ℓ) = θ0(ℓ) + α∆(Iso-Cℓ) , ∀ℓ = 1,...,L (9)

Algorithm 2 Iso-CTS: Isotropic Merging in Common and Task-Specific Subspaces (green – shared with Iso-C)

negligible information. At the same time, tasks with dominant directions of smaller intensity compared to the majority of tasks and whose directions are orthogonal to the common directions remain underrepresented. This limitation becomes more pronounced as the number of tasks increases and the tasks become more diverse (see Appendix A.4 for an extended discussion).

Require: Task matrices ∆1,...,∆T with ∆t ∈ Rm×n

- 1: Sum task matrices ∆TA = Tt=1 ∆t
- 2: Compute the SVD of ∆TA: ∆TA = UΣV ⊤, with U ∈ Rm×r,Σ ∈ Rr×r,V ∈ Rn×r,σ = diag(Σ)∈ Rr
- 3: Retain top-k singular vectors and values from common

subspace: U1:k = [u1|...|uk] V 1:k = [v1|...|vk] σcm = diag(Σ)1:k

- 4: Accumulate task-specific directions via projection:
- 5: for t = 1 to T do
- 6: ∆t = ∆t − U1:k(U1:k)⊤∆t (Eq.10)

- 7: Compute SVD: ∆t = UtΣtV ⊤t

- 8: Retain first s = r−Tk components of Ut and V t: U1:t s = [ut,1|...|ut,s] V 1:t s = [vt,1|...|vt,s] σtts = diag(Σt)1:s

- 9: end for
- 10: Combine common and task-specific spaces:

- U∗ = [U1:k|U1:1 s|...|U1:Ts] ∈ Rm×r

- V∗ = [V 1:k|V 1:1 s|...|V 1:Ts] ∈ Rn×r

- 11: Orthogonalize U∗ and V∗ via whitening (Eq.11)
- 12: Calculate isotropic factor σ:

σ =

1 r

k

i=1

σicm+

T

t=1

s

i=1

σt,its (Eq.13)

- 13: Reconstruct the matrix ∆Iso-CTS = σU∗V∗⊤ (Eq.12)

- 14: return ∆Iso-CTS

To address this limitation, we propose enhancing the range of directions used by Iso-C to ensure that the task-specific directions, which are orthogonal to those of the common subspace, are incorporated into the singular basis of the final merged matrix. We call this methodology as Isotropic Merging in Common and Task-Specific Subspaces (Iso-CTS).

Our approach starts with the top singular values of the common subspace and iteratively replaces the singular vectors associated with the lowest singular values with task-specific directions. The final goal is to find two orthonormal matrices U∗ ∈ Rm×r and V∗ ∈ Rn×r whose columns contain both common and task-specific directions. Afterward, the final matrix is reconstructed, and isotropic merging is applied. In the following, we provide a detailed explanation of our proposed algorithm.

Retaining components from the common subspace. We retain the top-k singular vectors associated with the subspace identified by ∆TA:

U1:k = [u1|...|uk] V 1:k = [v1|...|vk],

where U1:k, V 1:k are the top-k left- and right-singular vectors from the SVD of ∆TA. We analyze the impact of selecting k in Section 5.4.

where α is chosen on a held-out validation set.

Accumulating task-specific directions. We project each task-specific matrix ∆t onto the subspace orthogonal to the common subspace, i.e. the space spanned by top leftsingular directions of the common subspace U1:k:

Applying isotropic merging results in an enhancement of the normalized accuracy improvement and subspace alignment ratio (SAR) compared to Task Arithmetic (see Figure 3a). The increase in SAR is due to a higher number of dominant components kIso-c in ∆Iso-c (see Equation (6)), derived from the singular vectors of ∆TA, which are aligned with the subspaces of individual tasks (see Appendix A.2 for details). In Appendix A.3, we show that increased SAR is associated with reduced inter-task interference, measured by changes in internal activations induced by merging. In Algorithm 1, we present the Iso-C algorithm for a single layer.

∆t = ∆t − U1:k(U1:k)T∆t. (10)

We then compute the SVD of ∆t = Ut ΣtV t and retain the top s = r−Tk directions for each task t:

U1:t s=[ut,1|...|ut,s] V 1:t s=[vt,1|...|vt,s],∀t = 1,...,T. The orthogonal projection Eq. (10) guarantees that both the left- and right-singular vectors of ∆t, representing taskspecific directions, are orthogonal to the subspace spanned by the common directions (given by U1:k).

- 4.2. Isotropic Merging in Common and Task-Specific Subspaces

The effectiveness of Iso-C depends on how well the common subspace – identified by the dominant singular vectors of ∆TA – approximates the subspaces of the individual tasks. The approximation error arises from how these tasks interact when summed. The top singular directions of ∆TA capture only the dominant common variations, while singular vectors associated with near-zero singular values provide

Combining common and task-specific matrices. After identifying the k principal vectors for the common subspace and s = r−Tk principal vectors for each task, we now combine the common and task-specific directions by concatenating them: U∗ = [U1:k|U1:1 s|...|U1:Ts] ∈ Rm×r and V∗ = [V 1:k|V 1:1 s|...|V 1:Ts] ∈ Rn×r.

- Table 1. Iso-CTS achieves state-of-the-art performance for all backbones on all evaluated scenarios. We present average absolute accuracy and average normalized accuracy (in subscript) in %. The best method in bold and the second-best underlined.

ViT-B/32 ViT-B/16 ViT-L/14 8 tasks 14 tasks 20 tasks 8 tasks 14 tasks 20 tasks 8 tasks 14 tasks 20 tasks Zero-shot 48.3 57.2 56.1 55.3 61.3 59.7 64.7 68.2 65.2

Method

Fine-tuned 92.8 90.9 91.3 94.6 92.8 93.2 95.8 94.3 94.7

Weight Averaging 66.3(72.1) 64.3(71.1) 61.0(67.5) 72.2(76.6) 69.5(74.8) 65.3(70.4) 79.6(83.2) 76.7(81.1) 71.6(75.6) Task Arithmetic 70.8(76.5) 65.3(72.1) 60.5(66.8) 75.4(79.6) 70.5(75.9) 65.8(70.8) 84.9(88.7) 79.4(84.0) 74.0(78.1)

TIES 75.1(81.0) 68.0(74.8) 63.4(69.9) 79.7(84.3) 73.2(78.7) 68.2(73.3) 86.9(90.7) 79.5(84.1) 75.7(79.8) Consensus TA 75.0(80.8) 70.4(77.4) 65.4(72.0) 79.4(83.9) 74.4(79.9) 69.8(74.9) 86.3(90.1) 82.2(86.9) 79.0(83.2)

TSV-M 85.9(92.3) 80.1(87.9) 77.1(84.3) 89.0(93.9) 84.6(91.0) 80.6(86.5) 93.0(97.0) 89.2(94.4) 87.7(92.5) Iso-C (Ours) 86.3(92.9) 80.3(88.1) 75.5(82.5) 90.6(95.6) 84.8(91.1) 79.6(85.4) 94.2(98.3) 89.3(94.5) 87.6(92.2)

Iso-CTS (Ours) 86.2(92.8) 81.7(89.7) 78.1(85.5) 91.1(96.1) 86.4(92.8) 82.4(88.4) 94.7(98.8) 91.0(96.3) 90.1(94.9)

Orthogonalization. There is no guarantee that the left- and right-singular task-specific vectors are orthogonal to each other, as we are only projecting each task matrix onto the common subspace. To reconstruct the final merged matrix, we must orthogonalize U∗ and V∗. Following Gargiulo et al. (2025), we compute the SVD of U∗ = PU

and V∗ = PV

#### Q⊤U

#### ΣU

∗

∗

∗

, and whiten (Sch¨onemann, 1966):

#### Q⊤V

#### ΣV

∗

∗

∗

#### Q⊤U

#### U∗ = PU

∗

∗

#### Q⊤V

. (11)

#### V∗ = PV

∗

∗

Isotropic scaling and reconstruction. Finally, we reconstruct the final merged matrix and apply isotropic merging:

∆Iso-CTS = σU∗V∗⊤, (12)

where σ is obtained by averaging the singular values associated with the vectors selected for both common and task-specific subspaces. Specifically, defining σcm = diag(Σ)1:k ∈ Rk, the vector of singular values associated with the common subspace identified by U1:k and V1:k, and σtts = diag(Σt)1:s ∈ Rs, with s = r−Tk, the vector of singular values associated with each task-specific subspace U1:t s and V 1:t s, we define the scaling factor as:

1 r

σ =

k

T

σicm +

t=1

i=1

s

σt,its . (13)

i=1

Finally, similar to ISO-C, the merged model is defined as:

θIso-CTS(ℓ) = θ0(ℓ) + α∆(Iso-CTSℓ) , ∀ℓ = 1,...,L (14) where α is chosen on a held-out validation set.

## 5. Experimental Results

### 5.1. Fully fine-tuned vision models

We evaluate our approaches over sets of 8, 14, and 20 datasets, following Wang et al. (2024b). We provide the details of the datasets in Appendix C.1. We consider three

variants of CLIP (Radford et al., 2021) with ViT-B/32, ViTB/16 and ViT-L/14 as visual encoders (Dosovitskiy et al., 2021). We use the checkpoints fine-tuned on the tasks above, provided in Wang et al. (2024b) (see Appendix D.1 for results using TA checkpoints). If not stated otherwise, we present the results using the ViT-B/16 visual encoder.

We compare our approaches with the following model merging methods: weight averaging (Wortsman et al., 2022a), Task Arithmetic (Ilharco et al., 2023), TIES-Merging (Yadav et al., 2023), Consensus TA (Wang et al., 2024b) and TSV-M (Gargiulo et al., 2025). We include the results of the zero-shot model and fine-tuned models serving as lower- and upper-bound, respectively. We compare the results based on absolute and normalized accuracy following standard practice (Wang et al., 2024b; Gargiulo et al., 2025).

Table 1 presents our main results for multi-task model merging. Iso-CTS achieves state-of-the-art results in all of the settings. Iso-C achieves very similar results to Iso-CTS in the 8 task scenario. However, Iso-CTS significantly outperforms Iso-C when merging 14 and 20 models, with improvements of up to 2.8% in absolute accuracy. This suggests that it is possible to faithfully represent a small number of tasks in the common subspace. However, when the number of tasks increases, it becomes crucial to retain important directions from the task-specific subspaces in order to maximize model merging effectiveness.

### 5.2. LoRA-adapted vision models

To evaluate our approaches in low-rank adaptation scenario, we follow the evaluation protocol of KnOTS (Stoica et al., 2025), a recent state-of-the-art method for merging LoRA fine-tuned models. We use codebase and checkpoints provided by KnOTS: ViT-B/32 and ViT-L/14 fine-tuned with rank 16 LoRA (Hu et al., 2021) on 8 vision tasks. To adapt our methodologies to low-rank regime, we simply operate on reconstructed task matrices, i.e. ∆Wt = BtAt, where At,Bt are LoRA matrices for task t. We compare Iso-C and Iso-CTS with TIES and DARE-TIES (Yu et al., 2024)

0.1

β=0.0 (TA)

β=0.5 β=0.7

SingularValue

- β=0.9

- β=1.0 (Iso-C)

0.0

0 100 200 300 400 500 600 700 Singular Value Index

(a) Spectra of singular values for different values of interpolation coefficient (β).

0.90

SARavg

0.85

Cars

MNIST

0.80

DTD

RESISC45

EuroSAT

SUN397

GTSRB

SVHN

0.75

0.0 0.2 0.4 0.6 0.8 1.0 β

(b) Average Subspace Alignment Ratio (SARavg) vs. interpolation coefficient (β).

1.0

0.8

0.6

NAI

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0 β

(c) Normalized Accuracy Improvement (NAI) vs. interpolation coefficient (β).

- Figure 4. (a) Interpolating from ∆TA (β = 0) towards ∆Iso-C (β = 1) makes the spectrum of singular values of ∆M more uniform and increases the number of preserved components kM (Eq. (6)) denoted by dashed lines. (b) This results in an increased alignment between each task-specific model and merged model measured by SARavg. (c) As alignment increases, the performance also improves as predicted based on the strong correlation between these two properties investigated in Section 3.3.

- Table 2. Normalized per-task average accuracy. We merge 8 models fine-tuned with LoRA following (Stoica et al., 2025).

Method ViT-B/32 ViT-L/14

TA 63.7 74.4 TIES 63.7 75.2 DARE-TIES 63.7 74.7 KnOTS-TIES 68.0 78.2 KnOTS-DARE-TIES 63.9 75.6

Iso-C (Ours) 73.6 83.7 Iso-CTS (Ours) 73.7 85.3

– combined with KnOTS or not – and TA.

- We present the results in Table 2. Our methods, which are general purpose merging techniques, significantly outperform KnOTS, which are specifically designed for the LoRA merging. This highlights the versatility of Iso methods.

5.3. Language models

We present NLP results following the experimental setup from MaTS (Tam et al., 2023). We use T5-Large-LMAdapt (Lester et al., 2021) base model (a variant of T5Large (Raffel et al., 2020)) fine-tuned on subsets of 8 and 7 NLP tasks from T0 mixture (Sanh et al., 2022). We compare our approaches with weight averaging, TA, TIES, Fisher Merging (Matena & Raffel, 2021), RegMean (Jin et al., 2023), and MaTS (Tam et al., 2023).

- We present the results in Table 3. Both Iso-C and Iso-CTS significantly outperform the competing approaches, which highlights the versatility of our proposed methods. We observe that Iso-CTS achieves very similar results to Iso-C suggesting that the common space captures all the directions necessary to reliably represent these 7 and 8 NLP tasks.

Table 3. NLP results using T5-Large-LM-Adapt fine-tuned on tasks from T0 mixture. We present average absolute accuracy.

Method 8 tasks 7 tasks

(Zhou et al., 2022) (Yadav et al., 2023)

Fine-tuned 80.7 85.9 Weight Averaging 56.4 60.5 Task Arithmetic 63.8 69.2 TIES 62.8 71.9 Fisher Merging 57.7 61.0 RegMean 69.1 74.3 MaTS 72.5 81.5 Iso-C (Ours) 75.6 83.3 Iso-CTS (Ours) 75.2 82.8

computational complexity analysis of our approaches.

From Task Arithmetic to Isotropic Merging. We analyze what happens when interpolating between the singular values obtained by Task Arithmetic (TA) and those obtained by Iso-C, i.e. the model with the following spectra:

### 5.4. Analysis and Ablations

All the experiments in this Section are conducted on fully fine-tuned ViT-B/16 models. In Appendix B we provide the

Σβ = (1 − β)ΣTA + βΣIso-C, (15)

where β is an interpolation coefficient. Firstly, Figure 4a presents the change in singular values spectrum as we interpolate towards ∆Iso-C (β → 1). The skewed spectrum achieved by Task Arithmetic becomes isotropic, i.e. the scaling factor is equal along all of the singular directions. In Figure 4b we observe a steady increase in alignment between task-specific and merged models as measured by SARavg (Eq. (5)), and Figure 4c shows that as alignment increases (with β → 1), the performance of the merged model improves across all tasks. These results are consistent with our findings from Section 3.3 that show a strong correlation between alignment and the performance of the final model.

The impact of singular directions on performance. We analyze which singular directions contribute to the improvement of individual tasks. We truncate the flattened spectrum of Iso-C, keeping the k directions associated with the leftmost singular values, i.e. σi = σ for i ≤ k and σi = 0

1.0

0.8

0.6

NAI

0.4

Cars

MNIST

DTD

RESISC45

0.2

EuroSAT

SUN397

GTSRB

SVHN

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Fraction of retained components k/r from Iso-C

(a) Normalized Accuracy Improvement (NAI) of a model created by retaining k components of Iso-C (associated with top-k singular vectors from ∆TA).

0.9

SARavg

0.8

0.7

8 14 20 Number of tasks

(b) Average Subspace Alignment Ratios (SARavg) between merged and taskspecific models for varying sets of tasks.

1.0

0.8

Accuracy

0.6

TA Iso-C Iso-CTS

| |
|---|

| |
|---|

0.4

8 14 20 Number of tasks

(c) Distribution of accuracies of the merged models for varying sets of tasks.

- Figure 5. (a) The directions associated with the least significant singular values of ∆TA have a minor contribution to the performance of Iso-C model. (b) Task-specific directions introduced in Iso-CTS improve the Average Subspace Alignment Ratio (SARavg) between task-specific models and the merged model compared to Iso-C which uses only a common subspace. (c) Higher alignment translates to higher accuracy of Iso-CTS with respect to Iso-C.

0.0 0.2 0.4 0.6 0.8 1.0 Relative size of the common subspace kr

0.80

0.81

0.82

Accuracy

Iso-CTS

Iso-C

- Figure 6. Iso-CTS is robust to the selected size of the common subspace as any value leads to improvement over Iso-C. These results are for the 20-task scenario.

racy and the fraction of subspace assigned for the common subspace (kr) when merging 20 tasks. When kr = 1 Iso-CTS is equivalent to Iso-C and suffers a 2.8% drop in accuracy from the maximum. The optimal fraction of common subspace kr = 0.8, and we use this as a default value for Iso-CTS across all settings. Moreover, note that Iso-CTS is quite robust to the selection of this hyperparameter – any kr ∈ (0.0,1.0) offers a performance improvement over Iso-C while the performance for kr ∈ [0.5,0.9] varies by less than 0.5% from the optimal one.

## 6. Conclusion

for i > k. Note that the leftmost k directions are the ones associated with the highest singular values of ∆TA. We plot the task-wise Normalized Accuracy Improvement (NAI, Eq. (4)) for varying k in Figure 5a. We observe that the first few directions are responsible for rapid improvement on several tasks. Notably, these tasks belong to the aligned groups identified in Section 3.3 such as {MNIST, SVHN, GTSRB} and {EuroSAT, RESISC45}. Moreover, the directions associated with the least significant singular values of ∆TA have a negligible contribution to the performance. This supports our intuition for replacing less significant common directions with task-specific components in Iso-CTS (see Section 4.2). Figure 5b shows that Iso-CTS achieves higher Average Subspace Alignment Ratio (SARavg, Eq. (5)) than Iso-C. Most importantly, Figure 5c shows that thanks to the addition of task-specific directions, Iso-CTS achieves better performance across tasks.

In this work, we introduced an isotropic model merging framework that enhances alignment between task-specific and merged model subspaces to significantly improve the multi-task performance of the final merged model. We proposed Iso-C, which leverages Singular Value Decomposition to equalize singular values and create a more balanced representation across tasks, and Iso-CTS, which further incorporates task-specific directions to retain unique task features while preserving shared knowledge. Iso-CTS achieves state-of-the-art results across multiple model scales and task sets, demonstrating that subspace alignment is a critical factor in effective model merging. These findings provide new insights into model merging and pave the way for the future development of more effective techniques to combine the knowledge of multiple models.

Limitations. The common subspace is determined by Task Arithmetic, which can be suboptimal, and better methods could be developed. Although the proposed methods achieve state-of-the-art results in the LoRA merging scenario, they could be adapted to leverage the low-rank structure of task matrices to further improve the performance and efficiency.

Size of the common subspace for Iso-CTS. While Iso-C operates only in the common subspace, Iso-CTS enhances it with task-specific subspaces. Therefore, we must select the size of the common subspace k (and consequently the size of each task-specific subspace given by r−k

T ). Figure 6 plots the relationship between accu-

## Acknowledgements

Daniel Marczak is supported by National Centre of Science (NCN, Poland) Grant No. 2021/43/O/ST6/02482. This work was supported by Horizon Europe Programme under GA no. 101120237, project “ELIAS: European Lighthouse of AI for Sustainability”. Simone Magistri acknowledges travel support from ELIAS (GA no 101120237). We acknowledge the Spanish project PID2022-143257NB-I00, financed by MCIN/AEI/10.13039/501100011033 and FEDER, and Funded by the European Union ELLIOT project. Bartłomiej Twardowski acknowledges the grant RYC2021-032765-I and National Centre of Science (NCN, Poland) Grant No. 2023/51/D/ST6/02846. Andrew D. Bagdanov acknowledges funding support from the Italian national project “Collaborative Explainable neuro-symbolic AI for Decision Support Assistant”, CAI4DSA, CUP B13C23005640006.

## Impact Statement

This paper aims to advance the field of Machine Learning, specifically the subfield focused on merging models finetuned on different tasks to create a more effective multitask model. With the growing popularity of deep learning, increasingly powerful open-source models are becoming widely available and are being adopted in both research and industry. Advances in model merging could enhance the flexibility of utilizing these models by providing an efficient way to combine their specialized capabilities. Beyond this, our paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

Vedaldi, A. Describing textures in the wild. In CVPR, 2014.

Clanuwat, T., Bober-Irizar, M., Kitamoto, A., Lamb, A., Yamamoto, K., and Ha, D. Deep Learning for Classical Japanese Literature. arXiv preprint arXiv: 1607.06450, 2018.

Coates, A., Ng, A., and Lee, H. An Analysis of SingleLayer Networks in Unsupervised Feature Learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings, 2011.

Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. EMNIST: Extending MNIST to handwritten letters. In IJCNN, 2017.

Daheim, N., M¨ollenhoff, T., Ponti, E. M., Gurevych, I., and Khan, M. E. Model merging by uncertainty-based gradient matching. In ICLR, 2024.

Davari, M.-J. and Belilovsky, E. Model breadcrumbs: Scaling multi-task model merging with sparse masks. ECCV, 2024.

Denton, E. L., Zaremba, W., Bruna, J., LeCun, Y., and Fergus, R. Exploiting linear structure within convolutional networks for efficient evaluation. In NeurIPS, 2014.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

## References

Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv: 1607.06450, 2016.

Bossard, L., Guillaumin, M., and Van Gool, L. Food101 – Mining Discriminative Components with Random Forests. In ECCV, 2014.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. End-to-end object detection with transformers. ECCV, 2020.

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. ICCV, 2021.

Cheng, G., Han, J., and Lu, X. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 2017.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and

Du, G., Lee, J., Li, J., Jiang, R., Guo, Y., Yu, S., Liu, H., Goh, S. K., Tang, H.-K., He, D., and Zhang, M. Parameter competition balancing for model merging. In NeurIPS, 2024.

Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., Scardapane, S., Silvestri, F., and Rodol`a, E. Task singular vectors: Reducing task interference in model merging. In CVPR, 2025.

Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D.-H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J., Milakov, M., Park, J., Ionescu, R., Popescu, M., Grozea, C., Bergstra, J., Xie, J., Romaszko, L., Xu, B., Chuang, Z., and Bengio, Y. Challenges in Representation Learning: A Report on Three Machine Learning Contests. Neural Networks, 2013.

Helber, P., Bischke, B., Dengel, A., and Borth, D. Eurosat: A novel dataset and deep learning benchmark for land

use and land cover classification. Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

Hu, J. E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., and Chen, W. Lora: Low-rank adaptation of large language models. ICLR, 2021.

Huang, C., Ye, P., Chen, T., He, T., Yue, X., and Ouyang, W. Emr-merging: Tuning-free high-performance model merging. NeurIPS, 2024.

Ilharco, G., Wortsman, M., Gadre, S. Y., Song, S., Hajishirzi, H., Kornblith, S., Farhadi, A., and Schmidt, L. Patching open-vocabulary models by interpolating weights. In NeurIPS, 2022.

Ilharco, G., Ribeiro, M. T., Wortsman, M., Schmidt, L., Hajishirzi, H., and Farhadi, A. Editing models with task arithmetic. In ICLR, 2023.

Jin, X., Ren, X., Preotiuc-Pietro, D., and Cheng, P. Dataless knowledge fusion by merging weights of language models. In ICLR, 2023.

Kim, Y., Park, E., Yoo, S., Choi, T., Yang, L., and Shin, D. Compression of deep convolutional neural networks for fast and low power mobile applications. In ICLR, 2016.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3D Object representations for fine-grained categorization. In ICCV Workshops, 2013.

Krizhevsky, A. and Hinton, G. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009. URL https://www.cs.toronto.edu/ ˜kriz/learning-features-2009-TR.pdf.

Lecun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 1998.

Lee, C., Choi, J., Lee, C., Kim, D., and Hong, S. Adarank: Adaptive rank pruning for enhanced model merging. arXiv preprint arXiv: 2503.22178, 2025.

Lester, B., Al-Rfou, R., and Constant, N. The power of scale for parameter-efficient prompt tuning. EMNLP, 2021.

Li, W., Peng, Y., Zhang, M., Ding, L., Hu, H., and Shen, L. Deep model fusion: A survey. arXiv preprint arXiv: 2309.15698, 2023.

Lingam, V., Tejaswi, A., Vavre, A., Shetty, A., Gudur, G. K., Ghosh, J., Dimakis, A., Choi, E., Bojchevski, A., and Sanghavi, S. SVFT: parameter-efficient fine-tuning with singular vectors. CoRR, abs/2405.19597, 2024.

Lu, Z., Fan, C., Wei, W., Qu, X., Chen, D., and Cheng, Y. Twin-merging: Dynamic integration of modular expertise in model merging. NeurIPS, 2024.

Marczak, D., Twardowski, B., Trzcinski, T., and Cygert, S. MagMax: Leveraging Model Merging for Seamless Continual Learning. In ECCV, 2024.

Matena, M. and Raffel, C. Merging models with fisherweighted averaging. In NeurIPS, 2021.

Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. Reading digits in natural images with unsupervised feature learning. In NeurIPS Workshops, 2011.

Nilsback, M.-E. and Zisserman, A. Automated Flower Classification over a Large Number of Classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, 2008.

Ortiz-Jim´enez, G., Favero, A., and Frossard, P. Task arithmetic in the tangent space: Improved editing of pretrained models. In NeurIPS, 2023.

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. Cats and dogs. In CVPR, 2012.

Po, R., Yang, G., Aberman, K., and Wetzstein, G. Orthogonal adaptation for modular customization of diffusion models. In CVPR, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, 2021.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 2020.

Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Scao, T. L., Raja, A., et al. Multitask prompted training enables zero-shot task generalization. ICLR, 2022.

Sch¨onemann, P. H. A generalized solution of the orthogonal procrustes problem. Psychometrika, 1966.

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, 2013.

Stallkamp, J., Schlipsing, M., Salmen, J., and Igel, C. The german traffic sign recognition benchmark: a multi-class classification competition. In IJCNN, 2011.

Stoica, G., Ramesh, P., Ecsedi, B., Choshen, L., and Hoffman, J. Model merging with SVD to tie the Knots. In ICLR, 2025.

Tam, D., Bansal, M., and Raffel, C. Merging by matching models in task subspaces. TMLR, 2023.

Vasudevan, V. and Ramakrishna, M. A hierarchical singular value decomposition algorithm for low rank matrices. arXiv preprint arXiv: 1710.02812, 2017.

Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., and Welling, M. Rotation Equivariant CNNs for Digital Pathology. In MICCAI, 2018.

Wang, H., Xiao, Z., Li, Y., Wang, S., Chen, G., and Chen, Y. Milora: Harnessing minor singular components for parameter-efficient LLM finetuning. CoRR, abs/2406.09044, 2024a.

Wang, K., Dimitriadis, N., Ortiz-Jim´enez, G., Fleuret, F., and Frossard, P. Localizing task information for improved model merging and compression. In ICML, 2024b.

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In ICML, 2022a.

Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., Lopes, R. G., Hajishirzi, H., Farhadi, A., Namkoong, H., and Schmidt, L. Robust fine-tuning of zero-shot models. In CVPR, 2022b.

Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv: 1708.07747, 2017.

Xiao, J., Ehinger, K. A., Hays, J., Torralba, A., and Oliva, A. Sun database: Exploring a large collection of scene categories. IJCV, 2016.

Yadav, P., Tam, D., Choshen, L., Raffel, C., and Bansal, M. TIES-merging: Resolving interference when merging models. In NeurIPS, 2023.

Yang, E., Shen, L., Wang, Z., Guo, G., Chen, X., Wang, X., and Tao, D. Representation surgery for multi-task model merging. ICML, 2024.

Yu, L., Yu, B., Yu, H., Huang, F., and Li, Y. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In ICML, 2024.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. ICCV, 2023.

Zhou, J., Lin, Z., Zheng, Y., Li, J., and Yang, Z. Not all tasks are born equal: Understanding zero-shot generalization. In ICLR, 2022.

## A. Theoretical properties of Iso-C

In this Appendix, we discuss the theoretical properties of Iso-C by explicitly showing the connection between spectral skewness and the increased subspace dimensionality kM in the merged model achieved by Iso-C, which leads to a higher Subspace Alignment Ratio (SAR). Moreover, we explain why the increased SAR reduces inter-task interference. Finally, we highlight the limitations of Iso-C that lead to the development of Iso-CTS.

### A.1. Spectral skewness and the definition of kM

In this Section, we show that the number of dominant components kM of the merged model ∆M (see Equation (6)) is directly influenced by the skewness of its singular value spectrum. Using the singular value decomposition (SVD), let ∆M = UΣV T, where Σ = diag(σ1,...,σr). By the definition of Frobenius norm:

∥∆M∥2F =

r

r

σi2, ∥∆M − Πk,M∆M∥2F =

σi2.

i=1

i=k+1

Hence, the relative approximation error becomes:

∥∆M − Πk,M∆M∥2F ∥∆M∥2F

=

Accordingly, kM can be defined in terms of singular values:

r i=k+1 σi2

.

r i=1 σi2

r i=k+1 σi2

r i=1 σi2 ≤ ϵ2 .

kM = min k :

This formulation explicitly shows how the skewness of the spectrum {σi} controls kM. When ∆M has a skewed spectrum (e.g. σ12 ≫ ri=2 σi2), a small kM is sufficient to meet the error bound. This explains why Task Arithmetic ∆TA (β = 0 in Figure 4a) – which has a skewed spectrum – yields a smaller kTA than Iso-C, whose flatter spectrum leads to a larger kIso-C. Therefore, expressing kM directly in terms of singular values highlights the link between the spectral skewness and subspace dimensionality.

### A.2. Iso-C increases Subspace Alignment Ratio (SAR)

In this Section, we formally show how Iso-C increases Subspace Alignment Ratio (SAR) by expanding the effective subspace dimensionality of the merged model – from kTA in Task Arithmetic to kIso-C in Iso-C.

The rank kM defines the effective rank of the subspace identified by the merged model and it is directly determined directly by its spectrum (as discussed Appendix A.1). Let kTA be the effective rank of ∆TA, and define

#### T = {u1,..,uk

TA}

as the orthonormal basis formed by those kTA singular vectors. Flattening the spectrum of ∆TA (Figure 4a), yields ∆Iso-C with effective rank kIso-C > kTA (as discussed in Appendix A.1). This flattening modifies only the singular values of TA, leaving the singular vectors unchanged. Therefore, the original subspace T is contained within the larger subspace spanned by the top singular vectors of ∆Iso-C, defined as:

#### I = {u1,..,uk

Iso-C}.

,..,uk

TA

Thus, by construction, we have T ⊂ I. For simplicity, let ΠT = Πk

Iso-C,Iso-C denote the projection operators onto the subspaces spanned by T and I, respectively. Since T ⊂ I, for any matrix ∆t it holds that:

TA,TA and ΠI = Πk

SAR(∆t,∆TA;kTA) = ∥ΠT∆t∥F

∥ΠI∆t∥F ∥∆t∥F

= SAR(∆t,∆Iso-C;kIso-C), (16)

≤

∥∆t∥F

This inequality holds because by definition:

TA+1 j⟨ui,∆(tj)⟩2 ∥∆t∥2F

kTA i=1 j⟨ui,∆(tj)⟩2

kTA i=1 j⟨ui,∆(tj)⟩2 + ki=Iso-Ck

= ∥ΠI∆t∥2F ∥∆t∥2F

∥ΠT∆t∥2F ∥∆t∥2F

≤

=

,

∥∆t∥2F

where ∆(tj) denotes the j-th column of ∆t. The equality in Equation (16) holds only if the additional vectors added to the basis T – that is {uk

Iso-C} – are orthogonal to each ∆(tj) or, equivalently, if they do not intersect the column space of ∆t (i.e. its left singular vectors).

TA+1,...,uk

Hence, in general a lower kM yields smaller or equal SAR than a larger kM. However, our empirical findings show that enriching the basis T with singular vectors corresponding to smaller singular values in original task arithmetic spectrum (i.e. {uk

Iso-C}) consistently increases the alignment ratio (Figure 4b), implying that these vectors are relevant for representing each task matrix ∆t and not orthogonal to its left singular vectors. This analysis formally supports the claim that increasing the effective rank kM of the merged matrix – achieved by spectrum flattening in Iso-C – leads to a higher Subspace Alignment Ratio.

TA+1,...,uk

### A.3. Iso-C mitigates inter-task interference

Iso-C increases the Subspace Alignment Ratio (SAR), which quantifies how well the principal directions of a task matrix align with the principal directions of the merged model. In this Section, we demonstrate how a higher SAR contributes to mitigate inter-task interference by analyzing the relationship between subspace alignment and changes in internal activations following merging. Specifically, we define the interference as the degradation in a task’s internal representation due to merging —- that is, the deviation between the activations of the merged model and those of the corresponding single-task fine-tuned model. Intuitively, we can minimize the task interference by ensuring that the internal representations of task j remain stable after merging.

Let θ0 be the pre-trained weights for a layer l. Define the task matrix ∆j = θj − θ0 and the merged task matrix ∆M for the layer l. Then, for an input x(jl), we desire that the post-merging activation h(jl) = (θ0 + α∆M)x(jl), with α chosen on a validation set, be close to the task-specific activation hˆ(jl) = (θ0 + ∆j)x(jl). Hence, we can quantify the interference as:

||hˆ(jl) − h(jl)|| = ||(∆j − α∆M)x(jl)|| ≤ ||∆j − α∆M|| · ||x(jl)||. (17)

To show that the interference is lower when the Subspace Alignment Ratio (SAR) between ∆j and ∆M is higher, we decompose ∆j into components aligned with and orthogonal to ∆M:

∆j = ∆||j + ∆⊥j where ∆||j = Πk

M,M∆j, ∆⊥j = (I − Πk

M,M)∆j, (18) and Πk

M,M is the projection matrix onto the subspace spanned by the top kM left-singular vectors of ∆M (see Equation (6) for the definition of kM). The Subspace Alignment Ratio is then:

= ||∆||j ||F ||∆||j + ∆⊥ j ||F

SAR(∆j,∆M;kM) = ||Πk

M,M∆j||F ||∆j||F

. (19)

Similarly, decomposing ∆M into ∆||M and ∆⊥M and substituting Equation (18) in Equation (17), the interference becomes:

||∆j − α∆M|| = ||∆||j − α∆||M + ∆⊥j − α∆⊥M|| ≈ ||∆||j − α∆||M + ∆⊥j ||, (20) since kM minimizes the approximation error of ∆M, leading to ||∆⊥M|| ≈ 0.

If the SAR defined in Equation (19) is close to 1, then ||∆⊥j || is small, so the interference in Equation (20) mainly depends on ||∆||j − α∆||M||. Conversely, if SAR is near zero, the large orthogonal component ∆⊥j increases the overall interference, regardless of the choice of α. Even with an optimal α chosen via validation, interference cannot be reduced below the norm of the orthogonal component.

Iso-C increases the SAR of ∆t with the merged model — bringing it close to 1, as shown in the paper — by flattening the singular values. Thus, the optimal α can adjust the merged model such that interference is minimized. In contrast, Task Arithmetic (TA), with SAR varying across tasks, exhibits interference that cannot be reduced below the norm of the orthogonal component. We experimentally evaluate that the interference is lower for Iso-C than TA in Appendix D.2.

### A.4. Limitations of Iso-C that motivate Iso-CTS

This Section details the limitations of Iso-C that motivate the development of Iso-CTS. Specifically, Iso-C relies on the singular vectors obtained through Task Arithmetic to perform model merging. As a result, it tends to underrepresent tasks whose dominant directions have lower intensity compared to the majority, particularly when those directions are orthogonal to the shared (common) directions. This limitation becomes increasingly pronounced as the number and diversity of tasks increase (see Section 4.2).

To make this limitation explicit, we formalize the computation – via SVD – of the first left singular vector in Task Arithmetic, used by Iso-C, as the variance maximization problem:

T

T

||∆⊤TAu||2 = u⊤

∆t∆⊤t u + u⊤(

∆t∆⊤s )u

u1 = arg max

||u||=1

t=1

t,s=1,t̸=s

If a particular task ∆j has dominant directions with significantly lower intensity compared to the other tasks (i.e. lower Frobenius Norm), then its individual contributions ∆j∆⊤j to the total variance becomes smaller. Similarly, cross terms involving ∆j will also be comparatively small. Therefore, task j explicitly contributes less to the maximized variance captured by the first principal singular direction.

Moreover, if the directions of ∆j are orthogonal or nearly orthogonal to u1, (i.e. u⊤1 ∆j = 0), task j contributes minimally or not at all along this principal direction. Similar considerations apply to subsequent singular vectors u2,...uk, defining the common subspace. Finally, as the number of tasks T increases and tasks become more diverse, it becomes increasingly likely that tasks with distinct but smaller-magnitude directions will be underrepresented or absent in the dominant singular directions identified by the task arithmetic decomposition.

The goal of Iso-CTS is to address this limitation by incorporating orthogonal directions that are overlooked by the Task Arithmetic spectrum. This strategy yields the greatest improvements in settings with a large number of diverse tasks, as shown in our experimental results.

## B. Computational complexity analysis

In this Section, we analyze the computational complexity of Iso-C and Iso-CTS and compare it with that of our main competitor, TSV-M (Gargiulo et al., 2025).

Let ∆t ∈ Rn×n, and let T and L be the number of tasks and network layers, respectively. For simplicity, assume that each layer consists of a single squared n × n matrix.

In our analysis, we focus on the number of SVD performed by each algorithm, as this is by far the most costly component of each algorithm. The complexity of a single SVD on ∆t ∈ Rn×n is equal to O(n3) (Vasudevan & Ramakrishna, 2017). Below, we detail the total computational complexity for each merging method:

- • Iso-C performs a single SVD on ∆TA per layer, with total complexity: O(Iso-C) = O(Ln3)
- • Iso-CTS performs:

- – One SVD on ∆TA per layer (lines 2-3, Algorithm 2) with complexity O(Ln3)
- – One SVD on each ∆t, for all T tasks and each of the L layers (line 5, Algorithm 2), with complexity O(TLn3)
- – Two SVDs on two matrices U∗,V∗ ∈ Rn×n per layer (line 11, Algorithm 2), with complexity of O(2Ln3).

Therefore, the total complexity equals:

O(Iso-CTS) = O(Ln3 + TLn3 + 2Ln3) = O((T + 3)Ln3) = O(TLn3)

- • TSV-M (Gargiulo et al., 2025) performs:

- – T SVDs per layer on each task matrix (line 1, Alg. 1 from Gargiulo et al. (2025)): O(TLn3)
- – Two additional SVDs per layer (lines 10-11, Alg.1 from Gargiulo et al. (2025)): O(2Ln3)

Yielding the total complexity:

O(TSV) = O(TLn3 + 2Ln3) = O((T + 2)Ln3) = O(TLn3)

While Iso-CTS and TSV-M share the same asymptotic complexity, Iso-CTS incurs slightly more overhead due to the SVD on ∆TA (lines 2-3, Algorithm 2). Both methods can be further optimized by computing Truncated SVDs for Iso-CTS and TSV-M, since only a few components are retained. This reduces the complexity for both approaches. Iso-C is the most computationally efficient algorithm – its complexity is constant with respect to number of task T.

- C. Experimental details In this Appendix, we provide the dataset and implementation details used to carry out the experiments presented in the paper.

- C.1. Datasets

The 8-dataset benchmark consists of: Cars (Krause et al., 2013), DTD (Cimpoi et al., 2014), EuroSAT (Helber et al., 2019), GTSRB (Stallkamp et al., 2011), MNIST (Lecun et al., 1998), RESISC45 (Cheng et al., 2017), SUN397 (Xiao et al., 2016), and SVHN (Netzer et al., 2011).

The 14-dataset benchmark builds on the preceding one, incorporating six additional datasets: CIFAR100 (Krizhevsky & Hinton, 2009), STL10 (Coates et al., 2011), Flowers102 (Nilsback & Zisserman, 2008), OxfordIIITPet (Parkhi et al., 2012), PCAM (Veeling et al., 2018), and FER2013 (Goodfellow et al., 2013).

Finally, the 20-dataset benchmark includes the preceding 14 plus the following six: EMNIST (Cohen et al., 2017), CIFAR10 (Krizhevsky & Hinton, 2009), Food101 (Bossard et al., 2014), FashionMNIST (Xiao et al., 2017), RenderedSST2 (Socher et al., 2013), and KMNIST (Clanuwat et al., 2018).

- C.2. Implementation details

Our method relies on SVD, which is defined for two-dimensional matrices ∆ ∈ Rm×n. However, some weights of the neural networks are represented by vectors δ ∈ Rn, e.g. bias vectors and parameters of layer normalization (Ba et al., 2016). Therefore, following Gargiulo et al. (2025), we apply simple averaging to combine these parameters.

- D. Additional experiments

In this Appendix, we present additional experiments that complement the main paper, including comparisons with new vision baselines using Task Arithmetic model checkpoints (Ilharco et al., 2023) for evaluation. We empirically assess the reduced interference of Iso-C compared to Task Arithmetic and analyze the impact of the scaling factor α on our approaches. Finally, we present an ablation study showing what happens when spectrum flattening is applied to each task model individually.

### D.1. Additional vision baselines

In this Section we provide results with additional methods: Fisher Merging (Matena & Raffel, 2021), RegMean (Jin et al., 2023), PCB (Du et al., 2024), MaTS (Tam et al., 2023) and CART (Lee et al., 2025). These methods were originally evaluated on checkpoints from Task Arithmetic (Ilharco et al., 2023) provided for 8 tasks on ViT-B/32 and ViT-L/14. We follow this experimental protocol with Iso-C and Iso-CTS and present the results in Table 4. Iso-CTS sill achieves state-of-the-art performance followed by Iso-C. Note that the results differ from Table 1 where we used checkpoints from Consensus Merging (Wang et al., 2024b).

Table 4. Additional baselines for merging ViT-B/32 and ViT-L/14 on 8 tasks. We report absolute accuracy.

###### Method ViT-B/32 ViT-L/14

Zero-shot 48.3 64.7 Fine-tuned 90.5 94.2

Task Arithmetic 70.5 84.6 Fisher Merging 68.3 83.7 RegMean 71.8 82.2 PCB 76.3 87.5 MaTS 82.6 90.2 CART 83.0 90.8

Iso-C (Ours) 84.1 92.5 Iso-CTS (Ours) 84.3 93.0

[Figure 2]

- Figure 7. Mean L1 distance between the final embeddings of task-specific models and the merged one for Iso-C and TA. We used ViT-B/16 model.

### D.2. Interference quantification

In this Section, we experimentally show that merging interference (defined in Appendix A.3) is lower when merging is performed with Iso-C than with TA. Following Yang et al. (2024), we measure the interference as L1 distance between the final embeddings of task-specific models and merged one. In Figure 7 we present the results for merging 8 tasks on ViT-B/16. We observe that the interference is lower for Iso-C than for TA highlighting the effectiveness of Iso-C in reducing interference when merging models.

### D.3. Selection of scaling coefficient α

In Figure 10, we present the relationship between the validation accuracy and scaling factor α. We observe that TA is very sensitive to the selection of α, which potentially may require a more fine-grained search. On the other hand, both Iso-C and Iso-CTS are more robust to α selection, resembling the task-specific models. For reproducibility, In Table 5, we provide the optimal α value chosen on the held-out validation set for each model and number of tasks.

### D.4. Importance of isotropic scaling in Iso-CTS

In this Section we ablate the need for isotropic scaling in Iso-CTS. We present the comparison of the performance of Iso-CTS with and without isotropic scaling (Equation 13) in Figure 8. We observe that isotropic scaling is indeed a

Table 5. Optimal α value chosen on a held-out validation set for different model types and numbers of tasks for Iso-C and Iso-CTS.

Method Model 8 tasks 14 tasks 20 tasks

ViT/32-B 1.30 1.00 0.90 ViT/16-B 1.40 1.00 0.80 ViT/14-L 1.50 1.30 1.00

Iso-C

ViT/32-B 1.50 1.20 1.10 ViT/16-B 1.60 1.20 1.10 ViT/14-L 1.90 1.50 1.20

Iso-CTS

Performance of Iso-CTS with and without isotropic scaling

0.80

Accuracy

0.75

0.70

Iso-CTS (varying k)

Iso-CTS (optimal k)

Iso-CTS (varying k) w/o line 12

Iso-C (optimal k)

0.0 0.2 0.4 0.6 0.8 1.0 Relative size of the common subspace kr

- Figure 8. Performance of Iso-CTS with and without isotropic scaling (Eq.13). Isotropic scaling is a crucial component of Iso-CTS. Results for merging 20 tasks with ViT-B/16.

crucial component of Iso-CTS as long as common subspace exists. When only task-specific subspaces are in use (kr = 0), isotropic scaling does not make a significant difference. However, the design in Algorithm 2 also plays an important role,

especially when the number of merged models increases, leading to up to 2.8% improvement over Iso-C on 20 tasks (see Table 1).

### D.5. Applying Iso to individual task matrices

Flattening the skewed spectrum of singular values significantly improves the performance of the merged model, as demonstrated in Section 5.4. One may wonder if this operation might also be an effective strategy for improving single-task models. Figure 11 presents the performance of task-specific models in their original form along with their modified versions with singular value spectra of their task matrices flattened (which is equivalent to performing Iso-C for a single model). We observe a 3.3% drop in average performance across tasks. Therefore, the reason for the success of Iso-C lies in its ability to mitigate the negative effects of summing task matrices, not in inadvertently improving the original individual task matrices.

## E. Additional visualizations

In this Appendix, we provide additional visualizations that could not be included in the main paper due to space constraints. These include the spectra of task matrices, the Subspace Alignment Ratio per layer, and the correlation between Normalized Accuracy Improvement and Subspace Alignment Ratio when using the larger ViT-L/14 model.

### E.1. Visualization of task matrix spectra

When visualizing spectra of singular values of task matrices (Figure 1 and Figure 4a), we selected an output projection matrix WO from layer ℓ = 4 of ViT/B-16 as an illustrative example. In Figure 12, we present spectra across a variety of layers of ViT/B-16 for the task matrices of task-specific models, TA, Iso-C and Iso-CTS.

NAI vs. SARavg for ViT-L/14

1.0

0.9

0.8

0.7

NAI

0.6

TA

Iso-C

0.5

Cars

MNIST

DTD

RESISC45

0.4

EuroSAT

SUN397

GTSRB

SVHN

0.3

0.75 0.80 0.85 0.90 0.95 SARavg

Figure 9. Normalized Accuracy Improvement (NAI) vs. Average Subspace Alignment Ratio (SARavg) for ViT-L/14.

### E.2. Visualization of per layer Subspace Alignment Ratio

In Figure 3a and Figure 4b in the main paper, we presented SARavg – Subspace Alignment Ratio averaged across all the task matrices. In this Section, we present SAR at different depths of the model. Specifically, we calculate SAR between fine-tuned and merged weight matrices and an average of all the matrices for a given layer of the ViT-B/16 model. We present the results in Figure 13. We observe that the alignment is higher for Iso-C across all layers of the vision transformer. One may expect early layers to be more aligned but we find that for both approaches the alignment is similar across the layers.

### E.3. Visualization of Normalized Accuracy Improvement versus Subspace Alignment Ratio for ViT-L/14

In Figure 9 we replicate the experiment from Figure 3a (conducted on ViT-B/16) on ViT-L/14. The observations from the main paper hold – Normalized Accuracy Improvement strongly correlates with average Subspace Alignment Ratio, and increasing SARavg via merging with Iso-C leads to better performance.

###### Cars

###### DTD

###### EuroSAT

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

90

100

100

ValidationAccuracy(%)

80

80

80

60

70

60

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

###### GTSRB

###### MNIST

###### RESISC45

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

|100| | | | |
|---|---|---|---|---|
|70<br><br>80<br><br>90| | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100

100

ValidationAccuracy(%)

80

80

Individual task vectors

TA

60

Iso-C

60

Iso-CTS

40

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

###### SUN397

###### SVHN

###### Average

100

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

ValidationAccuracy(%)

80

90

80

75

80

70

70

60

60

65

0.0 0.5 1.0 1.5 2.0 α

0.0 0.5 1.0 1.5 2.0 α

0.0 0.5 1.0 1.5 2.0 α

- Figure 10. TA is sensitive to the selection of α, while both Iso-C and Iso-CTS are more robust to α selection, resembling the task-specific models. The α is chosen based on the best average performance on the validation set across tasks. The bottom right subplot denotes the optimal α for each method (Eq. (3), Eq. (9) and Eq. (14)). The model is ViT-B/16.

Original individual task matrices ∆t Iso applied to individual task matrices ∆t

Cars

85

ValidationAccuracy(%)

80

75

70

65

0.0 0.5 1.0 1.5 2.0 2.5 3.0

###### EuroSAT

100

ValidationAccuracy(%)

90

80

70

60

0.0 0.5 1.0 1.5 2.0 2.5 3.0

###### MNIST

100

ValidationAccuracy(%)

90

80

70

60

50

0.0 0.5 1.0 1.5 2.0 2.5 3.0

###### SUN397

80

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

ValidationAccuracy(%)

75

70

65

0.0 0.5 1.0 1.5 2.0 2.5 3.0 α

###### DTD

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

100

80

60

0.0 0.5 1.0 1.5 2.0 2.5 3.0

###### GTSRB

100

80

60

40

0.0 0.5 1.0 1.5 2.0 2.5 3.0

###### RESISC45

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

90

80

70

0.0 0.5 1.0 1.5 2.0 2.5 3.0

###### SVHN

90

80

70

60

50

0.0 0.5 1.0 1.5 2.0 2.5 3.0 α

- Figure 11. Validation Accuracy while scaling task matrices with α coefficient (Eq. (3) applied for a single task). We observe a performance gap between the accuracy of original and modified models for the optimal values of α (denoted by square).

Cars

MNIST

TA

DTD

RESISC45

Iso-C

EuroSAT

SUN397

Iso-CTS

GTSRB

SVHN

###### positional embedding

resblocks.0.attn.out proj

###### resblocks.0.mlp.c fc

0.15

0.4

0.15

SingularValue

0.3

0.10

0.10

0.2

0.05

0.05

0.1

0.00

0.0

0.00

0 50 100 150 200

0 200 400 600 800

0 200 400 600 800

###### resblocks.3.attn.in proj weight

###### resblocks.3.attn.out proj

###### resblocks.3.mlp.c fc

0.25

0.10

0.25

0.20

0.08

0.20

SingularValue

0.15

0.06

0.15

0.10

0.04

0.10

0.05

0.02

0.05

0.00

0.00

0.00

0 200 400 600 800

0 200 400 600 800

0 200 400 600 800

###### resblocks.3.mlp.c proj

###### resblocks.7.attn.out proj

###### resblocks.7.mlp.c fc

0.10

0.25

0.3

0.08

0.20

SingularValue

0.06

0.2

0.15

0.04

0.10

0.1

0.02

0.05

0.00

0.0

0.00

0 200 400 600 800

0 200 400 600 800

0 200 400 600 800

###### resblocks.11.attn.in proj weight

###### resblocks.11.mlp.c proj

###### proj

0.4

0.4

1.00

0.3

SingularValue

0.3

0.75

0.2

0.2

0.50

0.1

0.1

0.25

0.00

0.0

0.0

0 200 400 600 800 Singular Value Index

0 200 400 600 800 Singular Value Index

0 200 400 Singular Value Index

Figure 12. Visualization of singular value spectra of different task matrices for different types of layers in ViT/B-16.

Subspace Alignment Ratios by Layer for All Datasets

Cars

###### DTD

1.0

0.8

0.8

0.6

0.6

SAR

SAR

0.4

0.4

Alignment with Iso-C

0.2

0.2

Alignment with TA

0.0

0.0

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

###### EuroSAT

1.0

0.8

0.6

SAR

0.4

0.2

0.0

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

###### GTSRB

1.0

0.8

0.6

SAR

0.4

0.2

0.0

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

###### MNIST

1.0

0.8

0.6

SAR

0.4

0.2

0.0

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

###### RESISC45

0.8

0.6

SAR

0.4

0.2

0.0

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

###### SVHN

1.0

0.8

0.6

SAR

0.4

0.2

0.0

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

SUN397

0.8

0.6

SAR

0.4

0.2

0.0

Layer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11

Figure 13. Per layer Subspace Alignment Ratio between fine-tuned and merged weight matrices for ViT-B/16.

