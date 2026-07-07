[Figure 1]

## Alchemist: Unlocking Efficiency in Text-to-Image Model Training via Meta-Gradient Data Selection

Kaixin Ding1†, Yang Zhou2, Xi Chen1, Miao Yang3, Jiarong Ou3, Rui Chen3, Xin Tao3 , Hengshuang Zhao1 1The University of Hong Kong, 2South China University of Technology, 3Kling Team, Kuaishou Technology

|[Figure 2]|
|---|

# arXiv:2512.16905v1[cs.CV]18Dec2025

#### 50% outperforms Full !

Alchemist Select Random Select Full

[Figure 3]

1.5

[Figure 4]

AverageScore%（）

1.4

1.3

Full Data

1.2

[Figure 5]

1.1

|[Figure 6]<br><br>[Figure 7]|Redundancy<br><br>Outliers|
|---|---|

- 0.9
- 1

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

3M 6M 9M 12M 15M Full

Held-out Data

#Images

Alchemist Selected Data

Figure 1. Alchemist extracts informative data to improve data efficiency in Text-to-Image model training. We employ a rater model to score each sample and select the most informative subset for training. Alchemist remains effective under arbitrary retention percentiles. Training on the Alchemist-selected 50% subset can even outperform training on the full dataset with the same number of training epochs. Different rating are indicated by color-coded boxes. Full refers to the 30M LAION [33] dataset. The average score is computed as CLIPScore divided by FID. Code is available on Project Page.

### Abstract

consists of two key stages: data rating and data pruning. We train a lightweight rater to estimate each sample’s influence based on gradient information, enhanced with multi-granularity perception. We then use Shift-Gsampling strategy to select informative subsets for efficient model training. Alchemist is the first automatic, scalable, metagradient-based data selection framework for Text-to-Image model training. Experiments on both synthetic and webcrawled datasets demonstrate that Alchemist consistently improves visual quality and downstream model performance. Training on an Alchemist-selected 50% of the data can outperform training on the full dataset.

Recent advances in Text-to-Image (T2I) generative models, such as Imagen, Stable Diffusion, and FLUX, have led to remarkable improvements in visual quality. However, their performance is fundamentally limited by the quality of training data. Web-crawled and synthetic image datasets often contain low-quality or redundant samples which lead to degraded visual fidelity, unstable training, and inefficient computation. Hence, effective data selection is crucial for improving data efficiency. Existing approaches rely on costly manual curation or heuristic scoring based on single-dimensional features in Text-to-Image data filtering. Although meta-learning based method has been explored in LLM, there is no adaptation for image modalities. To this end, we propose Alchemist, a meta-gradient-based framework to select a suitable subset from large-scale textimage data pairs. Our approach automatically learns to assess the influence of each sample by iteratively optimizing the model from a data-centric perspective. Alchemist

### 1. Introduction

The remarkable progress of recent Text-to-Image (T2I) generative models, such as Imagen [32], Stable Diffusion [9], and FLUX [16], has marked a major milestone in artificial intelligence, enabling the synthesis of highly realistic and semantically aligned images from textual descriptions. These models have demonstrated transformative potential across diverse domains, including art, design, and

† Work done at Kling Team, Kuaishou Technology.

Corresponding Author

entertainment. However, the performance of such models is fundamentally bounded by the quality of their training data. While large-scale web-crawled [33] and synthetic image datasets [10] provide abundant visual information, they inevitably contain low-quality samples—such as blurry images, plain-background advertisements, or other redundant content. Training on such noisy or less-informative data can lead to degraded visual fidelity, training instability, and inefficient resource consumption, significantly hindering the scalability of T2I model development [1, 37]. Therefore, effectively organizing and curating the training data is crucial for improving the performance of text-to-image models and thus enhancing data efficiency [2, 12, 43].

Existing data selection approaches remain insufficient. Manual data curation is prohibitively expensive and lacks scalability. Human-defined heuristic criteria [26, 29, 39] often rely on classifiers that evaluate data from one single dimension. Although meta-learning-based methods [4, 47] have been explored in Large Language Model training, recent data selection approaches for Text-to-Image model training still rely predominantly on surface-form features [17], e.g., CLIP-based [30] linear estimators that predict aesthetic or clarity scores. Current methods fail to integrate complementary evaluation metrics. They also do not filter data from the perspective of maximizing downstream model performance, instead relying on human intuition.

To address these challenges, we propose Alchemist, which is specially designed for large-scale text-image pairs data selection via meta-gradient. Unlike heuristic scoring strategies, meta-learning simply requires users to specify what they want at the end of training. The model then autonomously learns how to value data samples through iterative meta-optimization, providing a scalable, datadriven approach to filter the training data stream. Our framework consists of two key stages: data rating and data pruning. We first train a lightweight rater network to reweight every image item with predicted classification score based on gradient information extracted from the T2I proxy model to estimate each sample’s influence. This design distinguishes images on their training dynamics (i.e. loss) with different ratings. We also add multi-granularity perception to enable the rater to learn not only sample’s own value but its contextual contribution within the group. Furthermore, we propose the Shift-GSample data pruning strategy which focuses on retaining the informative middleto-late region—samples that are detailed yet learnable. This strategy selects a subset of the rated dataset from a large corpus, thereby accelerating text-to-image model training and improving downstream task performance. We apply Alchemist to both synthetic image datasets and webcrawled datasets and also uses selected dataset to train on various model architectures. Alchemist consistently

achieves higher performance under the same number of training epochs (see Fig. 1). Moreover, we find that the selected subsets also align well with human intuitions, effectively filtering out images with plain backgrounds or highly chaotic content.

Our main contributions are summarized as follows:

- • A general paradigm for text-to-image data selection. We propose the first automatic, scalable, meta-gradientbased data selection framework for text-to-image model training. Alchemist often selects a small subset of the data (50%) that outperforms training on the full dataset.
- • Multi-granularity perception. We design a rater that considers both individual sample features and their collective context, providing a more comprehensive and reliable assessment of data quality.
- • Effective large-scale data selection. We validate our Alchemist selection method on four model families and three datasets of different domains. Our 15M selected subset enables the model to reach the same performance as random sampling but with 5× faster training time.

### 2. Related Work

Coreset selection. Coreset selection [27, 45] aims to identify a small, representative subset of large-scale data that yields performance comparable to training on the full dataset. Many approaches rely on static heuristics. These include pruning data via predefined rules based on intrinsic properties (e.g., resolution, watermarks) [6, 12, 13, 40, 46], or using pretrained models to score proxy attributes like aesthetic quality or text-image alignment [31, 38, 44]. While effective for removing outliers, these strategies are limited because the proxy metrics are static and not directly optimized for the final downstream model’s performance. To address this gap, recent work has explored more dynamic, model-aware selection strategies. A promising direction leverages signals from the training process itself, such as model gradients, to identify high-impact data for pretraining [20, 41]. This gradient-driven approach, shown to be effective in improving in-context learning, offers a more direct link between a sample’s value and the learning objective. Building on this principle, our work adapts a meta-learning framework to the unique demands of T2I generation, learning a data-scoring function that is explicitly optimized for the generalization performance of the final generative model.

Bilevel optimization for data selection. Model-based coreset selection methods [4, 41, 47] that compute the exact gradient of validation performance are often formulated as a bilevel optimization problem. While recent works [4, 8, 24] adopt second-order derivatives to find precise solutions, these methods are computationally expensive and thus infeasible for large-scale modalities such as text-to-image (T2I) models. To overcome this limitation, Alchemist is

instead closely related to penalty relaxation methods [7, 15, 19, 21, 34, 36, 42], which efficiently approximate bilevel optimization solutions. Our work is most similar to Shen et al. [35], but we introduce a continuous scoring mechanism that evaluates each sample based on the alignment between its individual gradient and the validation gradient. This design facilitates highly efficient adaptation to new downstream tasks.

### 3. Method

Data efficiency lies in selecting the smallest or highest quality subset of the corpus that still yields strong results [2, 12, 43]. Data selection aims to drop redundancy and outlier samples.

Formally, our objective is to find a text-image pairs subset of full dataset i.e. S ⊂ D such that the model θ trained on S achieves better or comparable evaluation performance M than that trained on D with the same training epoch:

M θ(S) ≳ M θ(D) . (1) θ can be T2I models with different architectures. We have proposed Alchemist, as illustrated in Fig. 2, which consists of two principal stages: data rating and data pruning. Data rating based on gradient information extracted from a T2I proxy model to estimate each data’s influence and data prunning chooses the suitable subset from the rated dataset to achieve data efficiency.

##### 3.1. Data Rating via Meta-Learning

Task formulation as bilevel optimization. The influence of each training sample can be quantified by its impact on a model’s performance on a held-out validation set [4, 28]. Let θ ∈ Rd

θ denote the parameters of a proxy Text-toImage (T2I) model, where dθ is the number of learnable parameters.. Instead of selecting a discrete subset of data, we generalize this concept by learning a continuous weight Wx

i ∈ [0,1] for each training instance xi ∈ Dtrain. The objective is to determine an optimal weighting scheme that minimizes the loss on a validation set Dval.

To achieve this, we introduce a rating network (rater) parameterized by µ ∈ Rd

µ, where dµ is the number of learnable parameters of the rater. Given a training sample xi, the network outputs a continuous classification score Wx

(µ). The goal is to learn the optimal rater parameters µ∗ that minimize the validation loss of the proxy T2I model. This naturally leads to a bilevel optimization problem formulated as:

i

[L(θ∗(µ);x)]. (2) The inner optimization seeks the optimal proxy model

µ∗ = arg min

Ex∼D

val

µ

parameters:

θ∗(µ) = arg min

(µ)L(θ;xi), (3) where θ∗(µ) is implicitly a function of rater parameters µ.

Wx

i

θ

xi∈Dtrain

Directly solving the bilevel problem is computationally prohibitive, as it requires retraining the proxy T2I model to convergence for every update of the rater parameters.

Meta-gradient optimization algorithm. To circumvent the challenge of solving the bilevel optimization problem, we employ an efficient gradient-based meta-learning strategy. This approach avoids fully solving the inner optimization by unrolling the training process for a few steps to approximate the meta-gradient for the rater’s parameters µ.

The optimization process, illustrated in Fig. 2 (a), proceeds as follows. First, we warm up the T2I proxy model to establish a stable initialization. During this phase, a reference proxy model with parameters θˆ is updated using only the training data:

θˆk+1 = θˆk − βk∇θL(θˆk;Dtrain), (4) where βk is the learning rate at step k. After the warmup period, the primary model parameters, θ, are initialized with θˆ and are subsequently updated using gradients from both the training and validation sets:

θk+1 = θk − βk (gval(θk) + gtrain(θk,µk)), (5) where gval(θk) = ∇θL(θk;Dval) and gtrain(θk,µk) =

(µk)∇θL(θk;xi). In parallel, the reference parameters θˆk continue to be updated using only the reweighted training samples. Concurrently, the rater’s parameters µ are updated. Based on the theoretical framework of [34], we employ an approximate stochastic gradient descent rule for the rater:

xi∈Dtrain Wx

i

µk+1 = µk − αk L(θk;xi) − L(θˆk;xi) ∇µWx

(µk),

i

(6) where αk is the meta learning rate and xi is a sample from Dtrain. This update rule encourages the rater to assign higher weights to training samples that cause a larger reduction in the validation-informed loss relative to the training-only loss.

are normalized within each batch. The rater outputs a raw score Wˆx

To ensure training stability, the sample weights Wx

i

for each sample, which is then passed through a softmax function:

i

exp(Wˆx

) j exp(Wˆx

, (7)

i

Wx

=

i

)

j

where the summation is over all samples xj in the current batch.

In current SOTA T2I models, the parameter space is often significantly larger than the number of training samples (dθ ≫ |Dtrain|), so it is reasonable to assume the existence of model parameters θ∗ that can achieve near-zero loss on

[Figure 12]

[Figure 13]

High Low

|[Figure 14]|
|---|

| | |
|---|---|
| | |

###### Data Rating Data Pruning Training

Text-to-Image Model

Raw Data Rated Data

Selected Data

#Images

Random

[Figure 15]

🔥

[Figure 16]

6M

[Figure 17]

[Figure 18]

###### Rater 𝜇

[Figure 19]

Top-K

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Train Data

[Figure 24]

Block

[Figure 25]

[Figure 26]

3M

|Instance MLP| |
|---|---|
| | |

Gsample

Shift-Gsample

Softmax

Transformer

|Group MLP| |
|---|---|
| | |

∇

0.6M

Position

s0 s10 s20 s30 s40 s50 s60 s70 s80

###### Proxy Model 𝜃

[Figure 27]

45 21

[Figure 28]

Rating 𝑾(𝒙)

[Figure 29]

Weighted Loss

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

🔥

[Figure 35]

[Figure 36]

Total Loss

Valid Data

Transformer

Loss

∇

18

( a ) ( b )

FID

- Figure 2. Overall pipeline of Alchemist. In the initial data rating stage (a), the rater predicts a classification score for each image based on gradient extracted from a T2I proxy model. The rater and the proxy model are jointly optimized through weighted loss and total loss. In the data pruning stage (b), we introduce the Shift-Gsample strategy to efficiently retain informative samples while filtering out redundant data and outliers. The resulting Alchemist-selected dataset enables highly efficient training of downstream text-to-image models.

the entire training set, i.e., L(θ∗;xi) ≈ 0 for all xi ∈ Dtrain.

output by the rater network parameterized by µ.

The use of minibatches inevitably introduces stochastic bias, as some batches may contain samples of generally higher or lower quality than others. To mitigate this instability, we introduce a lightweight Group MLP module within the rater network, which captures the global characteristics of each minibatch. Given the pooled instance features, we first derive a batch-level representation by concatenating the mean and variance of all features. This representation is passed through a two-layer MLP with sigmoid activation to produce a scalar batch weight Wbatch. Each instance feature is further processed by an Instance MLP and normalized via softmax to obtain instance weights Winst,i. The final weight for each instance is computed as the product Wi = Winst,i · Wbatch. This multi-level weighting enables the rater to capture both instance-level distinctiveness and batchlevel coherence.

This assumption implies that the inner optimization problem (Equation 3), if solved to convergence, would yield a near-zero objective value regardless of the weights:

(µ)L(θ;xi)

Wx

min

i

θ

xi∈Dtrain

(8)

(µ)L(θ∗;xi) ≈ 0.

=

Wx

i

xi∈Dtrain

In our meta-learning framework, the reference model θˆk is trained exclusively on the training data and thus could serve as an approximation of these optimal parameters θ∗. Consequently, we can posit that L(θˆk;xi) ≈ 0 for a training sample xi. Substituting this approximation into the rater update rule (Equation 6) simplifies the gradient estimation significantly:

(µk). (9) This simplified rule provides a more direct and efficient update mechanism for the rater in over-parameterized settings. Multi-granularity Perception. Computing gradients over the entire training dataset Dtrain is computationally infeasible in practice. Therefore, we approximate the bilevel optimization using random minibatches. At each iteration t, a minibatch Bt = {x1,...,xN} ⊂ Dtrain of size N is sampled from the training set, and the gradient of the weighted loss is computed as:

µk+1 = µk − αkL(θk;xi)∇µWx

i

##### 3.2. Data Pruning Strategy

With the rated data sorted from high to low, existing data selection methods typically adopt a Top-K pruning scheme. However, for text-image data, we observe that simply keeping top-ranked samples does not always lead to better performance. As rapid convergence may lead the model to overfit rather than achieve actual performance gains.

To analyze this, we randomly sample 10k data pairs across score ranges and track their training loss Li(t) and gradient norm gi(t) = ∥∇θLi(t)∥2 during proxy model training. As shown in Fig. 3, top-ranked samples maintain consistently low loss but exhibit minimal gradient change,

N

1 N

(µ)L(θ;xi) , (10) where Wx

gt = ∇θ

Wx

i

i=1

(µ) denotes the learned weight for instance xi

i

###### Epoch 1 Epoch 3 Epoch 5 Epoch 7

- 2
- 3
- 4
- 5
- 6

8.5

8.0

###### GradientNorm

7.5

###### Loss

7.0

6.5

6.0

| | |
|---|---|
| | |

1

0.02 0.08 0.02 0.08 0.02 0.08 0.02

0.08

###### Rating Rating Rating Rating

- Figure 3. Loss and gradient norm across different rating score ranges. For each training sample, we record its instantaneous loss and gradient norm at each training step during STAR-0.3B training. We track the evolution of loss and gradient norm over epochs.

[Figure 37]

[Figure 38]

[Figure 39]

Plain Informative Chaotic

- Figure 4. Representative examples of data distribution across score regions. The head region mainly contains plain samples, the middle-to-late region contains informative samples, and the tail region contains chaotic samples. Alchemist-selected data aligns with human intuition, filtering out most plain and chaotic samples.

performance improvement. The trend of the Block method indicates that the most informative region lies in the midto-late range (around s40–s60), yet restricting sampling to this narrow segment reduces diversity—something shiftGsample naturally balances.

We further provide insights into data distributions across different regions, as illustrated in Fig. 4. Samples in the head region are typically easy and uninformative, while those in the tail are noisy or overly complex. Our pruning strategy focuses on retaining the informative middle-to-late region—samples that are detailed yet learnable—striking a balance between learnability and diversity. As a result, the selected subset aligns well with human intuition, effectively removing overly plain or chaotic images and leading to a more robust and generalizable training set for downstream T2I models.

indicating limited learning contribution. In contrast, samples within the mid-to-late score range show more active gradient dynamics and thus contribute more effectively to learning. At the tail of the distribution, samples’ gradients hardly descend. Motivated by these findings, we argue that the most performance-contributing data lie in the mid-tolate portion of the distribution, while the highest-scoring region should be pruned out.

### 4. Experiments

##### 4.1. Experimental Setup

Based on these observations, we propose the pruningbased shift-Gaussian sampling (Shift-Gsample) strategy that better balances data informativeness and diversity. Specifically, after discarding the top n% of the sorted dataset, we perform Gaussian sampling with a shifted mean over the remaining samples:

Training datasets. We use the following text–image datasets for data selection and model evaluation: (1) LAION [33]-30M, a large-scale web-crawled corpus with broad coverage but contains a considerable proportion of low-quality samples. Unless otherwise specified, LAION serves as the primary source for our experiments; (2) Fluxreason [10]-6M, a high-quality synthetic dataset generated by the FLUX model, specifically designed to capture reasoning-oriented image–text pairs; and (3) HPDv3 [23], a mixed real–synthetic dataset containing 1.17M annotated pairwise samples. It integrates both high-quality real photographs and images produced by state-of-the-art generative models, while retaining examples from older models and lower-quality real data to ensure diversity. We construct HPDv3-2M by pairing each text prompt with both humanpreferred and non-preferred images, thereby introducing a mixture of samples with different tastes.

i − µ)2 2σ2

(wx

i ∈ [n%,100%], (11) where wx

p(i) ∝ exp −

, wx

denotes the normalized weight (percentile) of sample xi in the sorted dataset, µ represents the target mean percentile for sampling, and σ controls the sampling spread. We validate these insights experimentally in Fig. 2 (b). Five pruning methods are compared, each selecting an equal number of samples (20% of the total number, i.e., 6M samples in our setting). Random denotes uniform sampling independent of data ranking, while Top-K selects the highest-scoring samples. Block refers to a contiguous segment from the sorted list (e.g., s0 starts at percentile 0% ). Gsample is the Gaussian Sample with the mean at the mid. Our proposed shift-Gsample achieves the best

i

Evaluation metrics. We evaluate the trained T2I models on MJHQ-30K [18] and report quantitative metrics for both

0-10% 30-40%

20% 50%

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

60-70%

80% 90% 100%

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Figure 5. Image distribution of Alchemist-selected LAION data subsets. Samples are sorted from high to low scores. The early portion mainly contains images with white or plain backgrounds, the middle portion is more informative and content-rich, and the tail portion gradually becomes noisier, containing unclear content, multiple objects, or visually dense regions.

image fidelity and text–image alignment through FID and CLIP-Score. Additionally, we use GenEval [11] to assess generation performance in complex multi-object reasoning scenarios. Due to the relatively low intrinsic quality of the original full datasets, the overall absolute metrics are lower than those of the base models. Further analysis is provided in Supplementary Material.

Models for data selection and training. We evaluate Alchemist on four base models: STAR-40M, STAR-0.3B, STAR-0.9B [22], as well as FLUX-mini-3B [3]. All STAR models are trained from scratch, whereas FLUXmini is fine-tuned with LoRA. The rater model uses the same backbone as STAR-40M for image feature extraction. The T2I proxy model uses STAR-0.3B. STAR is a scalewise text-to-image generation model designed for multiresolution text-conditioned synthesis. Flux-mini is distilled from Flux-dev [16] and pruned by reducing network depth. We use the corresponding optimal resolution.

Implementation details. All experiments are implemented in PyTorch on NVIDIA A800 GPUs. The rater model is trained with a batch size of 64 across 2 GPUs, while the downstream text-to-image models are trained with the maximum feasible batch size across 8 GPUs. For fairness, we keep the batch size consistent across all comparison methods with the same model. For evaluating different data-selection strategies, we use STAR-0.3B. STAR models

are trained from scratch for three epochs, and FLUXmini is fine-tuned for one epoch unless otherwise stated. Additional implementation details are provided in the supplementary material.

##### 4.2. Main Results

We present comparisons with other approaches in Tab. 1, main results of Alchemist across different models in Tab. 2 and adaptation to different datasets in Tab. 3. Key findings are summarized below.

50% outperforms full. We compare our Alchemist with several methods, including using the full dataset (Full), random sampling (Random), and heuristic selection metrics based on four common image quality metrics—Aesthetic, Clarity, Frequency, and Edge Density. Detailed pruning strategies for each heuristic metric are provided in Supplementary Material. We report the best-performing results for each to ensure fairness. As shown in Tab. 1, the 50% subset selected by our method achieves comparable performance of training on the full dataset while requiring only half the data. Moreover, using merely 20% of the data selected by Alchemist yields performance comparable to training with 50% randomly sampled data, demonstrating substantially improved data efficiency.

Effectiveness across different models Tab. 2 presents comparison results against baseline random selection on

- Table 1. Comparison of different image data selection methods evaluated on the MJHQ-30K and GenEval benchmarks. All models are trained from scratch for three epochs. Additional results with more training epochs are provided in Fig. 6. Ours-small refers to a subset selected by Alchemist with a smaller number of images. Due to the relatively lower quality of the original full datasets, the overall evaluation metrics tend to be lower.

Method #Params #Images

Training Time (hours)

MJHQ-30K[18] GenEval[11] FID↓ CLIP-Score↑ Score↑

Full 0.3B 30M 65.34 17.48 0.2336 0.2752 Random 0.3B 15M 34.60 19.70 0.2220 0.2632 Aesthetic[17] 0.3B 15M 34.60 17.36 0.2299 0.2604 Clarity[17] 0.3B 15M 34.60 17.85 0.2261 0.2251 Frequency[5] 0.3B 15M 34.60 18.77 0.2276 0.2519 Edge-density[25] 0.3B 15M 34.60 20.13 0.2240 0.2429 Ours-small 0.3B 6M 13.08 18.22 0.2277 0.2367 Ours 0.3B 15M 34.60 16.20 0.2325 0.2645

- Table 2. The values in green superscript indicate the performance improvement compared to the random sampling in different model scales, model types and data scales. From-scratch training on STAR and LoRA finetuning on FLUX-mini.

Table 3. Comparison of multiple data ratention percentage on different domain datasets, HPDv3-2M and Flux-reason-6M.

HPDv3-2M [23] Flux-reason-6M [10] FID ↓ CLIP-Score ↑ FID ↓ CLIP-Score ↑

Data percentage

###### #Params #Images FID ↓ CLIP-Score ↑

Random (20%) 35.55 0.1977 23.66 0.2135 Ours (20%) 32.27 0.2039 22.78 0.2142 Random (50%) 20.21 0.2201 19.35 0.2260 Ours (50%) 18.15 0.2265 18.59 0.2265

- (a) Experiments on STAR with different model scale

40M 6M 26.74−1.92 0.2059+0.0050 0.3B 6M 18.22−2.47 0.2277+0.0058 0.9B 6M 16.71−1.43 0.2356+0.0027

- (b) Experiments on STAR with different data scale

0.3B 3M 21.54−0.61 0.2193+0.0024 0.3B 6M 18.22−2.47 0.2277+0.0058 0.3B 15M 16.20−3.50 0.2325+0.0105

- (c) Experiments on FLUX-mini 3B 6M 19.98−0.43 0.2366+0.0007

across model scales. Importantly, the training cost of the small rater model is negligible compared with training larger generative models, enabling data selection to be amortized across multiple downstream tasks.

Adaptability across different data domains. Alchemist also generalizes effectively across diverse data sources. We conduct experiments on HPDv3 [23] and Flux-reason [10], comprising 2M and 6M image-text pairs respectively. HPDv3 contains both real photographs and synthetic images, with human-preferred and non-preferred image-text pairs, while Flux-reason consists of entirely synthetic, reasoning-oriented data. As shown in Tab. 3, our method outperforms the baseline at both 20% and 50% data retention percentage. Combined with results on webcrawled dataset LAION from previous experiments, Alchemist demonstrates strong adaptability to heterogeneous data domains.

the MJHQ-30K evaluation benchmark. Using the rater to evaluate the dataset once, the selected data can be transferred to train any target model, whether training STAR models from scratch or fine-tuning FLUX models with LoRA adapters. Alchemist is also effective across different retention ratios, as shown in Fig. 1. In all cases, Alchemist consistently outperforms random selection, demonstrating that our data selection approach identifies informative samples and improves data efficiency for text-to-image models.

Efficiency in text-to-image model training. As shown in Fig. 6, STAR-0.3B achieves 2.33× and 5× faster training time using Alchemist-selected data while reaching performance comparable to random-selected at 20% and 50% data retention percentage, respectively.

Scalability to larger and different model families. We demonstrate that using a small proxy model can effectively boost performance for both larger models and different model families. As shown in Tab. 2, we apply Alchemist on LAION dataset and then train STAR-T2I-0.3B, START2I-0.9B (i.e., a larger model), and finetune FLUX-mini3B (i.e., a different model family) for evaluation. Across all settings, Alchemist consistently outperforms random selection, indicating that the selected data transfers well

##### 4.3. Ablation Study

Multi-granularity perception. Due to memory constraints, we perform gradient updates within minibatches

### 5. Analysis

Table 4. Ablation study at 6M text-image pairs. Method #Images FID ↓ CLIP-Score ↑

What does the Alchemist-sorted dataset look like? We visualize the Alchemist-sorted datasets in Fig. 5, showing subsets corresponding to the 0%-100% percentiles of the ranked data. We observe that samples in the top-ranked subset (0-20%) are typically clean, simple, and often with plain backgrounds. While these samples lead to lower training loss, they lack semantic richness. In contrast, the middle-and-late ranked subsets (e.g., 40-80%) contain higher-quality, informative samples with clear subjects and well-defined actions. Finally, the lowest-ranked subset (e.g., 90-100%) predominantly consists of noisy, chaotic, cluttered scenes with multiple fine-grained or inconsistent subjects.

TopK 6M 48.20 0.1945 Gsample 6M 19.22 0.2248 Ours (Shift-Gsample) 6M 18.37 0.2272

+Group-MLP 6M 18.22 0.2277

Ours 6M Random 6M

26

0.235

24

0.225

CLIP↑

22

2.33× faster 2.33× faster

FID↓

20

0.215

18

16

0.205

4.36 13.08 21.8 30.52 39.24

4.36 13.08 21.8 30.52 39.24

Training Time

Training Time

What are the differences in image modality? Previous studies in LLMs have already explored meta-learningbased data selection [41]. But they only limited to one stage and simply rated data with lower loss and achieve the top part, assuming that lower loss leads to improved data efficiency. While clean text often indicates high information with no redundant tokens, it works in LLM. However this assumption does not directly hold for images. In the image domain, a single image exhibits substantial redundancy (up to 70% [14]), e.g. large regions with plain backgrounds. Consequently, highly rated images with lower loss tend to be visually clean but contain limited information, contributing little to model learning. Although such samples may accelerate convergence, we observe that the resulting models are often overfitted and produce lowerquality outputs. Focusing solely on minimizing the inner T2I proxy model loss can therefore introduce bias. To this end, we propose Alchemist. Our approach uses ratings to estimate each sample’s influence on convergence speed and applies pruning to balance convergence speed and final performance. We also selectively retain primarily informative samples while including a small portion of easy or noisy examples to maintain data diversity.

Ours 15M Random 15M

22

0.235

5× faster

20

0.225

###### CLIP↑

FID↓

18

5× faster

0.215

16

0.205

11.53 34.6 57.65 80.71 103.77

11.53 34.6 57.65 80.71 103.77

Training Time

Training Time

- Figure 6. Performance of models trained on 6M and 15M data vs training time. Evaluations are conducted on MJHQ-30K benchmark.

rather than on the entire dataset. Differences between batches can introduce bias. To mitigate this, we incorporate a group MLP in the rater to capture global information across the dataset. As shown in Tab. 4, considering this group perspective improves performance compared to only evaluating each data sample individually.

Pruning strategies. We evaluate three different pruning strategies in Tab. 4. The Shift-Gsample strategy achieves the best performance, indicating that samples from the middle-to-late portions of the rated dataset yield the largest gains. These portions correspond to the samples exhibiting the fastest gradient changes, as shown in Fig.3. The ShiftGsample pruning strategy, guided by the data rating, can be adapted across different domains to filter out redundant or less informative samples.

Why autoregressive proxy T2I model? While our T2I proxy model adopts an autoregressive (AR) structure, the particular modeling paradigm is not crucial, as long as it provides sufficient discrimination between data samples. We adopt an AR proxy primarily for its single-step generation process, which enables efficient gradient propagation and stable meta-optimization. This choice avoids the problems faced in diffusion models, where some intermediate timesteps results can correlate with good visual quality but still suffer from high noise and large loss, thus misleading the optimization signal.

Training trends. As shown in Fig. 6, we track training time and performance for models trained on 6M and 15M data, respectively. Alchemist consistently accelerates improvements throughout training. In contrast, random sampling exhibits early-stage fluctuations and slower gains. Alchemist demonstrates steady and stable improvement, highlighting the effectiveness and robustness of our data selection approach. As models approach convergence (approximately 9-10 epochs), performance naturally fluctuates before stabilizing.

### 6. Conclusion

We present Alchemist, the first automatic, scalable, and gradient-based data selection method specifically designed

for real-world Text-to-Image model training. Extensive experiments demonstrate that Alchemist consistently improves both training efficiency and model performance. Our method establishes a new paradigm for meta-learning based data curation in Text-to-Image development and opens up opportunities for larger text–image datasets in the future.

### References

- [1] Amro Abbas, Kushal Tirumala, D´aniel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023. 2
- [2] Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. A survey on data selection for language models. arXiv preprint arXiv:2402.16827, 2024. 2, 3
- [3] Tencent ARC. Fluxkits: A repo that facilitates the usage of flux series models, 2024. Accessed: 2025-11-09. 6
- [4] Dan A Calian, Gregory Farquhar, Iurii Kemaev, Luisa M Zintgraf, Matteo Hessel, Jeremy Shar, Junhyuk Oh, Andr´as Gy¨orgy, Tom Schaul, Jeffrey Dean, et al. Datarater: Metalearned dataset curation. arXiv preprint arXiv:2505.17895,

2025. 2, 3

- [5] Kenneth R Castleman. Digital image processing. Prentice Hall Professional Technical Reference, 1979. 7
- [6] Mayee Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, and Christopher R´e. Skill-it! a datadriven skills framework for understanding and training language models. Advances in Neural Information Processing Systems, 36:36000–36040, 2023. 2
- [7] Frank H Clarke. Optimization and nonsmooth analysis. SIAM, 1990. 3
- [8] Yalun Dai, Yangyu Huang, Xin Zhang, Wenshan Wu, Chong Li, Wenhui Lu, Shijie Cao, Li Dong, and Scarlett Li. Data efficacy for language model training. arXiv preprint arXiv:2506.21545, 2025. 2
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1

- [10] Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A millionscale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025. 2, 5, 7
- [11] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 6, 7
- [12] Yuxian Gu, Li Dong, Hongning Wang, Yaru Hao, Qingxiu Dong, Furu Wei, and Minlie Huang. Data selection

- via optimal control for language models. arXiv preprint arXiv:2410.07064, 2024. 2, 3
- [13] Suchin Gururangan, Ana Marasovi´c, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. Don’t stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964, 2020. 2
- [14] Daniel Kersten. Predictability and redundancy of natural images. Journal of the Optical Society of America A, 4(12): 2395–2400, 1987. 8
- [15] Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert Nowak. On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation. arXiv preprint arXiv:2309.01753, 2023. 3
- [16] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 1, 6
- [17] LAION-AI. aesthetic-predictor: A linear estimator on top of clip to predict the aesthetic quality of pictures. GitHub repository, 2022. 2, 7
- [18] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation, 2024. 5, 7
- [19] Bo Liu, Mao Ye, Stephen Wright, Peter Stone, and Qiang Liu. Bome! bilevel optimization made easy: A simple firstorder approach. Advances in neural information processing systems, 35:17248–17262, 2022. 3
- [20] Zikang Liu, Kun Zhou, Wayne Xin Zhao, Dawei Gao, Yaliang Li, and Ji-Rong Wen. Less is more: High-value data selection for visual instruction tuning. arXiv preprint arXiv:2403.09559, 2024. 2
- [21] Songtao Lu. Slm: A smoothed first-order lagrangian method for structured constrained nonconvex optimization. Advances in Neural Information Processing Systems, 36: 80414–80454, 2023. 3
- [22] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Biye Li, Huaian Chen, and Yi Jin. Star: Scale-wise text-conditioned autoregressive image generation. arXiv preprint arXiv:2406.10797, 2024. 6
- [23] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025. 5, 7
- [24] Mahdi Nikdan, Vincent Cohen-Addad, Dan Alistarh, and Vahab Mirrokni. Efficient data selection at scale via influence distillation. arXiv preprint arXiv:2505.19051, 2025. 2
- [25] Nobuyuki Otsu et al. A threshold selection method from gray-level histograms. Automatica, 11(285-296):23–27,

1975. 7

- [26] Guilherme Penedo, Hynek Kydl´ıˇcek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024. 2
- [27] Jeff M Phillips. Coresets and sketches. arXiv preprint arXiv:1601.00617, 2016. 2
- [28] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing

- gradient descent. Advances in Neural Information Processing Systems, 33:19920–19930, 2020. 3
- [29] Jiantao Qiu, Haijun Lv, Zhenjiang Jin, Rui Wang, Wenchang Ning, Jia Yu, ChaoBin Zhang, Zhenxiang Li, Pei Chu, Yuan Qu, et al. Wanjuan-cc: A safe and high-quality open-sourced english webtext dataset. arXiv preprint arXiv:2402.19282,

2024. 2

- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [32] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1
- [33] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 1, 2, 5
- [34] Han Shen and Tianyi Chen. On penalty-based bilevel gradient descent method. In International conference on machine learning, pages 30992–31015. PMLR, 2023. 3
- [35] Han Shen, Pin-Yu Chen, Payel Das, and Tianyi Chen. Seal: Safety-enhanced aligned llm fine-tuning via bilevel data selection. arXiv preprint arXiv:2410.07471, 2024. 3
- [36] Han Shen, Zhuoran Yang, and Tianyi Chen. Principled penalty-based methods for bilevel reinforcement learning and rlhf. Journal of Machine Learning Research, 26(114): 1–49, 2025. 3
- [37] Eric Slyman, Stefan Lee, Scott Cohen, and Kushal Kafle. Fairdedup: Detecting and mitigating vision-language fairness disparities in semantic dataset deduplication. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13905–13916, 2024. 2
- [38] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025. 2
- [39] Alexander Wettig, Aatmik Gupta, Saumya Malik, and Danqi Chen. Qurating: Selecting high-quality data for training language models. arXiv preprint arXiv:2402.09739, 2024. 2
- [40] Mengzhou Xia, Antonios Anastasopoulos, Ruochen Xu, Yiming Yang, and Graham Neubig. Predicting perfor-

- mance for natural language processing tasks. arXiv preprint arXiv:2005.00870, 2020. 2
- [41] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning. arXiv preprint arXiv:2402.04333, 2024. 2, 8
- [42] Quan Xiao, Songtao Lu, and Tianyi Chen. A generalized alternating method for bilevel learning under the polyak-{\L} ojasiewicz condition. arXiv preprint arXiv:2306.02422,

2023. 3

- [43] Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy S Liang. Data selection for language models via importance resampling. Advances in Neural Information Processing Systems, 36:34201–34227, 2023. 2, 3
- [44] Jia Yu, Lichao Zhang, Zijie Chen, Fayu Pan, MiaoMiao Wen, Yuming Yan, Fangsheng Weng, Shuai Zhang, Lili Pan, and Zhenzhong Lan. Quality and quantity: Unveiling a million high-quality images for text-to-image synthesis in fashion design. arXiv preprint arXiv:2311.12067, 2023. 2
- [45] Haizhong Zheng, Rui Liu, Fan Lai, and Atul Prakash. Coverage-centric coreset selection for high pruning rates. arXiv preprint arXiv:2210.15809, 2022. 2
- [46] Haizhong Zheng, Elisa Tsai, Yifu Lu, Jiachen Sun, Brian R Bartoldson, Bhavya Kailkhura, and Atul Prakash. Elfs: Label-free coreset selection with proxy training dynamics. arXiv preprint arXiv:2406.04273, 2024. 2
- [47] Xinlin Zhuang, Jiahui Peng, Ren Ma, Yinfan Wang, Tianyi Bai, Xingjian Wei, Qiu Jiantao, Chi Zhang, Ying Qian, and Conghui He. Meta-rater: A multi-dimensional data selection method for pre-training language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10856–10896, 2025. 2

