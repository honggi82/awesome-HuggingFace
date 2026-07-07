# arXiv:2408.16357v3[cs.CV]6Oct2025

## Law of Vision Representation in MLLMs

Shijia Yang1 Bohan Zhai4 Quanzeng You4

Jianbo Yuan4 Hongxia Yang3 Chenfeng Xu2∗

1STANFORD UNIVERSITY 2UC BERKELEY 3THE HONG KONG POLYTECHNIC UNIVERSITY 4INDEPENDENT RESEARCHER

#### Abstract

We introduce the “Law of Vision Representation” in multimodal large language models (MLLMs), revealing a strong correlation among crossmodal alignment, vision representation correspondence, and overall model performance. We quantify these factors using the cross-modal Alignment and Correspondence scores. Extensive experiments across fifteen distinct vision representation settings and evaluations on eight benchmarks show that A and C scores correlate with performance following a quadratic relationship. By leveraging this relationship, we can identify and train the optimal vision representation for an MLLM, achieving a 99.7% reduction in computational cost without the need for repeated finetuning of the language model. The code is available at https://github.com/bronyayang/ Law of Vision Representation in MLLMs.

#### 1 Introduction

Current multimodal large language models (MLLMs) (Chen et al., 2024a; Liu et al., 2024e;d) have achieved remarkable advancements by integrating pretrained vision encoders with powerful language models (Touvron et al., 2023; Zheng et al., 2023). Among the core components of a general MLLM, vision representation plays a critical role. Many researchers have utilized CLIP (Radford et al., 2021) or SigLIP (Zhai et al., 2023b; Tschannen et al., 2025) as the primary image feature encoder, but their limitations are becoming increasingly noticeable (Tong et al., 2024b; Geng et al., 2023; Yao et al., 2021). As a result, alternative vision representations (Tang et al., 2025) and the combination of multiple vision encoders are being actively explored (Tong et al., 2024a; Lin et al., 2023).

Despite this growing attention, selection of vision representation has largely been empirical. Researchers typically test a set of vision representations on a specific MLLM and choose the one that yields the highest performance on benchmark tasks. This approach, however, is constrained by the number of representations tested and does not address the underlying factors that drive performance differences. As a result, the optimal vision representation for a specific MLLM is often determined by empirical performance rather than a deep understanding of the factors that contribute to success. The question of what fundamentally makes a feature representation achieve the highest performance remains largely unanswered.

To address this gap in understanding what makes a vision representation optimal for MLLMs, we propose the Law of Vision Representation in MLLMs. It aims to explain the key factors of vision representation that impact MLLM benchmark performance. Our findings reveal that cross-modal Alignment (A) and Correspondence (C) of the vision representation are strongly correlated with model performance.; specifically, higher A and C lead to improved performance. To quantify this relationship, we define A and C scores that measures cross-modal alignment and correspondence in vision representation. The A and C

∗Corresponding author: xuchenfeng@berkeley.edu

[Figure 1]

[Figure 2]

Law of Vision Representation in MLLMs models and predicts MLLM performance with different vision representations through cross-model Alignment and Correspondence

[Figure 3]

[Figure 4]

|[Figure 5]|
|---|

[Figure 6]

|cat.|
|---|

This is a Law of VR works with different:

[Figure 7]

encoder size

input resolution

feature combination

encoder type

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Figure 1: Visualization of the Law of Vision Representation in MLLMs.

scores as well as model performance exhibit a quadratic relationship, with a coefficient of determination of 94.06%.

Furthermore, the Law of Vision Representation guides the selection of an optimal vision representation for MLLMs. Originally, this process was extremely costly because even subtle changes in vision encoding—such as switching encoder types, altering image resolution, or testing feature combinations—require finetuning the language model (Lin et al., 2024). For example, using a top data-efficient MLLM pipeline with a 7B language model requires 3,840 NVIDIA A100 GPU hours to test the 10 encoders, amounting to a cost of approximately $20,0001. Testing additional encoders leads to a linear increase in cost. Moreover, the recent trend of feature combination, which often results in better performance, necessitates combinatorial testing of vision encoders. Testing all possible combinations of 10 encoders results in 1023 combinations, exponentially increasing the cost and energy consumption. This process consumes approximately 100,000 kilowatt-hours2, enough to drive an electric vehicle around the Earth 13 times.

Thus, we are the first to propose a policy, AC policy, that selects the optimal vision representation using AC scores within the desired search space. Unlike traditional methods that rely on benchmarking performance, the AC policy enables the expansion of the search space—allowing for an increased number of vision representations to be considered—without incurring additional costs. We demonstrate that this approach enhances both accuracy and efficiency compared to randomly searching for the optimal representation. The policy successfully identifies the optimal configuration among the top three choices in 96.6% of cases, with only three language model finetuning across a 15-setting search space.

#### 2 Related Works

###### 2.1 Vision Representations for MLLMs

Recent studies have explored various vision representations in MLLMs (Beyer et al., 2024; Ge et al., 2024; Liu et al., 2024e; Wang et al., 2024b; Sun et al., 2023; Luo et al., 2024). Interestingly, some findings indicate that relying solely on encoders outside of the CLIP family (Cherti

- 1https://replicate.com/pricing
- 2https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-

datasheet.pdf

- et al., 2023; Zhai et al., 2023b; Li et al., 2022), such as DINOv2 (Oquab et al., 2023) and Stable Diffusion (Rombach et al., 2021), often leads to lower performance (Karamcheti et al., 2024; Tong et al., 2024a). However, combining features from these encoders with CLIP features, such as concatenating image embeddings in the channel dimension, significantly enhances performance beyond using CLIP alone (Tong et al., 2024a;b; Liu et al., 2024c; Kar et al., 2024). Researchers intuitively suggest that these additional encoders provide superior detail-oriented capabilities, but no studies have thoroughly analyzed the underlying causes of the performance change (Wei et al., 2023; Lu et al., 2024a). This suggests that the attributes of an optimal vision representation remain not fully understood.

- 2.2 Cross-modal Alignment Cross-modal alignment refers to the alignment between image and text feature spaces (Duan

- et al., 2022). This concept emerged with the introduction of text-image contrastive learning (Radford et al., 2021; Jia et al., 2021). Although current MLLMs utilize contrastively pretrained image encoders, the challenge of achieving effective alignment persists (Ye et al., 2024; Zhai et al., 2023a; Woo et al., 2024). Despite efforts to critique the limitations of CLIP family representations and explore alternative vision representations, many approaches continue to rely on contrastively pretrained encoders or adding contrastive loss without fully eliminating them (Zhang et al., 2024b; Lu et al., 2024a; Tong et al., 2024a;b; Liu et al., 2024b). In our work, we point out that alignment in vision representation is essential for improved model performance and is crucial for data efficiency. Without pre-aligned vision representations, extensive data pretraining is required to achieve cross-modal alignment within the language model (Ge et al., 2024; Chen et al., 2024b; Li et al., 2024c).

- 2.3 Visual Correspondence

Visual correspondence is a fundamental component in computer vision, where accurate correspondences can lead to significant performance improvements in tasks, such as image detection (Xu et al., 2024; Nguyen & Meunier, 2019), visual creation (Tang et al., 2023; Zhang et al., 2024c), and MLLMs (Liu et al., 2024a), etc. Correspondences are typically categorized into semantic- and geometric-correspondences. Semantic correspondences (Zhang et al., 2024c; Min et al., 2019) involve matching points that represent the same semantic concept not necessarily representing the same instance. Geometric correspondences (Sarlin et al., 2020; Lindenberger et al., 2023), on the other hand, require matching the exact same point across images, which is often crucial for low-level vision tasks, such as pose estimation (Sarlin et al., 2020; Lindenberger et al., 2023; Zhang & Vela, 2015), and SLAM tasks, etc.

Several studies have pointed out that the CLIP family’s vision representation ”lacks visual details” (Lu et al., 2024a; Tong et al., 2024b; Ye et al., 2024). We explain this observation through the concept of correspondence. Current MLLMs convert images into embeddings, with each embedding representing a patch of the image. Image features with high correspondence increase the similarity within internal image patches on similar semantics, thereby enabling the retrieval of more detailed information.

#### 3 Law of Vision Representation in MLLMs

We introduce the Law of Vision Representation in Multimodal Large Language Models (MLLMs). It states that the performance of a MLLM, denoted as Z, can be estimated by two factors: cross-modal alignment (A) and correspondence (C) of the vision representation, assuming vision representation is the sole independent variable while other components (e.g., language model and alignment module) remain fixed. This relationship can be expressed as:

Z ∝ f(A,C) (1) where f is a quadratic function of A and C.

###### 3.1 Assumptions

Following NVLM (Dai et al., 2024), we categorize MLLMs into the following types: (1) Decoder-only MLLMs (Tong et al., 2024a; Liu et al., 2024e; Li et al., 2024a; Liu et al., 2024f; Dai et al., 2024; Lu et al., 2024b; Zhang et al., 2024a; Wang et al., 2024a): These MLLMs consist of vision encoder(s) and an alignment module, such as a multilayer perceptron (MLP), which maps the vision representation into vision tokens. These tokens are designed to have a similar distribution as language tokens and are directly input into a language model in the same manner as language tokens. (2) Cross-attention-based MLLMs (Dai et al., 2024; Bai et al., 2023; Alayrac et al., 2022; Laurenc¸on et al., 2024; Chen et al., 2024c): These MLLMs include vision encoder(s) and an additional module, often serving as a downsampling component, such as a perceiver resampler. The vision tokens generated are integrated into the language model through cross-attention mechanisms.

- • The Law of Vision Representation specifically focuses on decoder-only MLLM architecture due to their widespread adoption and their simplicity, which facilitates controlling variables in training recipes and enables clear mathematical modeling.
- • We further assume vision representation is the only independent variable, while the alignment module and LLM architecture remain fixed. In the case of a unfrozen vision encoder, we cannot guarantee that the vision encoder does not take the function of the alignment module. This causes the architecture and role of the alignment module to change alongside the encoder, making the experiment uncontrolled and the models no longer comparable.

###### 3.2 Theoretical Justification

In this section, we theoretically analyze how an increase in A and C leads to improved model performance. When a vision representation demonstrates high cross-modal alignment and accurate correspondence, the MLLM exhibits the following desired properties:

- • When training a MLLM, if the vision representation is closely pre-aligned with the language distribution, the pretrained language model requires less computational effort to bridge the gap between different modalities during finetuning. In Section A.1, we provide theoretical justification that finetuning on well-aligned multimodal data is about equivalent to finetuning on text-only data, eliminating additional effort beyond language finetuning. This efficiency can lead to improved performance, especially in scenarios where the available training data for finetuning is limited.
- • If the vision representation ensures accurate correspondence, the attention within the image embeddings is precise. Consequently, the MLLM develops a refined focus on visual content, capturing even details that cannot be derived solely from text-to-image attention, leading to a more detailed interpretation of the image. We provide theoretical justification in Section A.2.

###### 3.3 Empirical Justification

In this section, we empirically show that A and C scores are strongly correlated to model performance. To quantify the correlation between A and C as well as model performance, we first propose methods to measure cross-modal alignment and correspondence within the vision representation:

- • To quantify cross-modal alignment, we define a metric A SCORE, that measures how well the vision representation is mapped into the language model’s space. With both the vision encoder and the LLM frozen, if visual features aligns with the LLM’s language space effectively, the LLM’s prediction error will be minimized. In other words, a well-aligned vision embedding leads to a higher likelihood for the correct caption tokens.

Formally, for an input image I and its associated caption y = (y1, y2, . . . , yT), where T is the sequence length, let f(I) denote the projected visual representation (i.e., the

output of the vision encoder + projector). The conditional probability of generating token yt is given by:

P yt | f(I), y<t ,

, with y<t representing the tokens preceding yt. The alignment score is then defined as the average log likelihood over all tokens:

T

1 T

### ∑

A SCORE(I, y) =

log P yt | f(I), y<t

t=1

A higher A SCORE indicates that the visual features are more effectively aligned with the language model’s representation, as reflected by the increased loglikelihood of the correct caption.

- • To quantify correspondence, we measure how accurately key points in one image can be matched to their semantically corresponding locations in another image. Given a pair of image with annotated, semantically matching key points, we first extract features from each image pair. Let Fs and Ft denote the feature maps of the source and target images, respectively. Using the feature vectors at the labeled key point positions in Fs, we predict the corresponding key points in Ft by selecting the location with the maximum simi-

larity, yielding a set of predicted key points {p1pred, . . . , pmpred} for m key points. The ground-truth key points for the image pair are denoted by {p1GT, . . . , pmGT}. The correspondence score is then defined as the Percentage of Correct Keypoints (PCK), computed as follows:

m

1 m

### ∑

C SCORE =

<T (2)

pipred−piGT

i=0

2

where T is a threshold proportional to the bounding box size of the object in the image, and (·) is the indicator function that returns 1 when the condition is satisfied and 0 otherwise.

A higher C SCORE indicates more accurate key point correspondence, reflecting better vision feature matching.

To capture the overall performance, we integrate the A SCORE and the C SCORE into a single metric called the AC SCORE. We do this by fitting a second-degree polynomial to benchmark performance, which allows us to model potential nonlinear interactions between A and C. Formally, the AC Score is defined as:

2

### ∑

AC SCORE =

α=0

2−α

### ∑

wαβAαCβ (3)

β=0

where wαβ are trainable parameters that are optimized to best fit the benchmark performance.

This formulation allows the AC Score to capture both the individual contributions of alignment and correspondence, as well as their interaction effects.

[Figure 13]

Results. We fit a simple regression model using 15 vision representations across 4 vision-based MLLM benchmarks. As shown in Figure 2, the average coefficient of determination (R2) obtained is 94.06% when using the AC score of the vision representations. For comparison, we also fit models using 15 random scores, the A score alone, and the C score alone, all with quadratic functions. The random scores and single-factor models show lower correlations with performance. This result highlights the strong correlation between the AC score and MLLM performance, validating the Law of Vision Representation. Refer to Section 5.4 for details.

Figure 2: R2 values for regression models fitted on various scores.

#### 4 AC Policy

1 Feature extraction from any vision encoder 2 Score features with AC

Image Features

CLIP+DINOv2

Cross-modal Alignment = ? Correspondence = ?

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

DINOv2

CLIP@224

CLIP@336

SD

F with AC score

SigLIP

[Figure 21]

3 Fit + predict performance with AC

Train only optimal model

Performance

Performance

Vision Model Connector LLM

Predict

Fit

[Figure 22]

AC

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Figure 3: Overall framework of AC policy.

Problem Formulation. The MLLM architecture assumed in this framework consists of a frozen vision encoder, followed by a trainable connector (alignment module) and the pretrained language model. To determine the optimal out of k vision representations for the MLLM, we originally needs finetune LLM k times, making the scaling of k difficult. Therefore, we propose AC policy, as illustrated in Figure 3, to efficiently estimate the optimal vision representation from a search space consisting of k vision representations. We finetune only k′ LLMs to obtain downstream performance, allowing k to scale without significant cost, where k′ ≪ k. The value of k′ should be determined based on the computational budget allocated for vision representation selection.

Policy Fitting. Let X ∈ Rk×6 be the matrix containing AC scores of vision representation in the search space. We subsample k′ data points from X, denoted as Xs ∈ Rk′×6, to serve as the input to the regression model:

###### y = Xsw + ϵ (4)

Here, w ∈ R6 is the vector of model parameters, ϵ ∈ Rk′ is the vector of error terms, and y ∈ Rk′ represents the downstream performance on a desired benchmark.

[Figure 27]

Vision Representation Resolution Single vision encoder: feed-forward models

Average Predicted Optimal VR 87.72%

ACPolicyImprovement

OpenAI CLIP ViT-L/14 224 OpenAI CLIP ViT-L/14 (Radford et al., 2021) 336 OpenCLIP ViT-L/14 (Cherti et al., 2023) 224 DINOv2 ViT-L/14 (Oquab et al., 2023) 224 SigLIP ViT-B/16 (Zhai et al., 2023b) 224 SigLIP ViT-L/16 (Zhai et al., 2023b) 256 SigLIP2 ViT-L/16 (Tschannen et al., 2025) 256

###### +61.02%

Single vision encoder: diffusion models

Randomly Testing VR 26.7%

- SD 1.5 (Rombach et al., 2022) 768
- SD 2.1 (Rombach et al., 2022) 768 SD Image Variations 768 SD XL (Podell et al., 2023) 512 DiT (Peebles & Xie, 2023) 512 SD 3 (Esser et al., 2024) 512

Multiple vision encoders: feature combination

CLIP+DINOv2 ViT-L/14 224 CLIP+DINOv2 ViT-L/14 336

- Figure 4: Given a limited budget of 4 finetunings, AC policy achieves 87.72% Recall@3 in predicting the optimal vision representation.

Table 1: Vision representations explored.

Sampling Strategy. The selection of k′ can impacts the function fit and, consequently, the accuracy of predictions. To avoid sampling points that are too close in terms of their A and C scores, we employ a sampling strategy based on the coordinates.

The normalized A and C score pairs of k vision representation can be plotted on a 2D graph as coordinates (A,C), To ensure diverse sampling, we divide the graph into regions. For each iteration j in which the total sampled points do not yet fulfill k′, we divide the graph into 4j equal regions. We then remove empty regions and those that contain previously sampled points. The next data point is randomly selected from a remaining region.

Results. In Figure 4, we demonstrate that the AC policy consistently predicts the optimal vision representation using minimal resources within a finite search space of 15 settings. Our aim is to finetune only a small subset of this space while ensuring that the optimal vision representation is among the top-3 predictions (Recall@3). With a computational budget equivalent to 4 full finetuning runs, a random subset selection achieves only 26.7% Recall@3. In contrast, the AC policy achieves 87.72% Recall@3 (averaged over 6 benchmarks) while still requiring just 4 full training runs. For further details, see Section 5.5.

#### 5 Empirical Result Details

###### 5.1 Experiment Settings

For our MLLM pipeline, we deploy LLaMA-based LLM, specifically Vicuna-7B 1.5 (Zheng

- et al., 2023) as well as Qwen-14B 2.5 Instruct (Qwen et al., 2025), and utilize a widely used 2layer GeLU-MLP connector as the projector. For vision representation, we explore a variety of encoder types and sizes, input resolutions, training paradigms, and feature combinations,

- as detailed in Table 1.

Our training process consists of two stages. In Stage 1, we perform alignment using the LLaVA 1.5 dataset with 558K samples (Liu et al., 2024e), training only the projector. In Stage 2, we train both the connector and the language model on an expanded LLaVA 1.5 dataset containing 665K samples.

The MLLM benchmarks used in this paper include four vision-based benchmarks (MMBench (Liu et al., 2023), MME (Fu et al., 2023), OKVQA (Marino et al., 2019), SEED-Bench (Li

- et al., 2024b)) and four QCR-based benchmarks (MMMU (Yue et al., 2024), TextVQA (Singh et al., 2019), VizWiz (Gurari et al., 2018), ScienceQA (Lu et al., 2022)).

###### 5.2 AC Score

To compute the cross-modal alignment score, we perform Stage 1 training on all vision representations to obtain the projected vision representations. This stage requires significantly less computation than Stage 2, involving only 0.298% of the trainable parameters. The image–caption pairs are taken from the LLaVA-558K dataset, and the alignment score is computed by averaging the results across 100 randomly sampled images.

For the correspondence score, we follow common practices using the SPair-71k dataset (Min et al., 2019). Note that all benchmarks share the same A and C scores, while the quadratic parameters in the function fitting adapt to capture variations across tasks.

###### 5.3 Feature Extraction

Both MLLM training and score computation involve image feature extraction. Below, we introduce the approach for obtaining two types of vision representations.

Vision Representation from Feed-forward Models. Given an image I ∈ RH×W×3 we process it either in its raw form for U-Net models or in a patchified form for transformer models. For transformers, we extract the last hidden state F ∈ Rl×c where l is the sequence length and c is the hidden dimension. In the case of the U-Net model, we take the intermediate activation F ∈ RHˆ×Wˆ ×c after the first upsampling block. Note that the features from these two types of models are interchangeable between sequence and grid formats through reshaping and flattening. For consistency, the following sections assume that all features have been pre-converted into the same format.

Vision Representation from Diffusion Models. Diffusion model is primarily used for generating images via multi-step denoising, yet a recent trend is to use diffusion model as the vision representation model (Xu et al., 2024; 2023; Zhang et al., 2024c; Tong et al., 2024a). Specifically, for diffusion models, given an image I ∈ RH×W×3, we first add noise to the VAE-encoded representation of I:

xt = √at · VAE(I) + ( 1 − at) · ϵ (5)

where ϵ ∼ N (0, I) and at is determined by the noise schedule. Note that we utilize the little-noise strategy by setting the t = 1. In that case, the diffusion model only denoises the noise-latents once and we treat the one-step denoising latents as the vision representation features.

###### 5.4 Additional Results on the Law of Vision Representation

- In Section 3, we demonstrate the strong correlation between the AC score and MLLM performance by analyzing the coefficient of determination (R2) obtained from fitting a quadratic regression model. In this section, we further ablate the experiments by adding baselines, fitting model performance with random scores, A scores, and C scores separately. Additionally, we explored the relationship between the A score and C score by fitting a linear regression model. We provide more insights about the AC relationship in appendix A.3. Besides, we avoid higher-degree transformations to prevent overfitting, which could obscure the true relationship between A and C scores.

As shown in Table 2, the results indicate that using the AC score consistently outperforms all other set-

Fitting Data R2 (Vision) R2 (OCR) No transformation on fitting data Random 4.03% 1.75% A Score 80.53% 58.00% C Score 39.02% 14.57% AC Score 80.55% 62.06% Polynomial transformation on fitting data

Random 36.90% 31.26% A Score 91.48% 79.67% C Score 56.56% 30.11% AC Score 94.06% 83.85%

Table 2: Averaged R2 results of AC and other baselines fitting on MLLM benchmarks.

tings in terms of R2 values. While this observation holds regardless of the degree of fitted function, using a second-degree polynomial on A and C scores yields the highest correlation with model performance. This suggests an inherent trade-off between A and C scores: vision representations with high cross-modal alignment often exhibit lower correspondence, and vice versa.

Interestingly, we observe a lower correlation between OCR-based benchmark performance and C scores, which leads to a reduced correlation between the AC score and OCR-based benchmark performance. In Section 6, we discuss how the use of the SPair-71k correspondence dataset across all benchmarks fails to adequately capture correspondence in images containing text.

Here, we additionally provide MLLM with Qwen 2.5 14B Instruct as LLM backbone fitting results, as shown in Table 3. Specifically, we use the exact same settings (data, vision representations, hyperparameters), changing only the LLM backbone to Qwen 2.5 14B. We again evaluate the resulting MLLMs on 8 benchmarks, recompute the A and C scores, reporting the fitting R2. The averaged fitting R2 are above 90% for vision based benchmarks and above 80% for OCR based benchmarks, comparable to Vicuna 1.5 7B, showing that our conclusion applies to different types and sizes of LLM backbones.

LLM Backbone R2 (Vision) R2 (OCR)

Vicuna 1.5 7B 94.06% 83.85% Qwen 2.5 14B 91.50% 82.27%

Table 3: Averaged law fitting R2 comparison for different types and sizes of LLM backbones.

###### 5.5 Additional Results on the AC Policy

[Figure 28]

[Figure 29]

[Figure 30]

SEED-Bench MMBench OKVQA

[Figure 31]

AC Predicted ScienceQA

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

TextVQA VizWiz

PredictionRecall@3

[Figure 40]

[Figure 41]

Random Predicted

OptimalVR

[Figure 42]

Number of Full Training

- Figure 5: Number of full training (LLM finetuning) cycles required to include the optimal vision representation within the top-3 predictions (Recall@3).

- In Section 4, we demonstrate that fitting the AC score consistently predicts the optimal vision representation with minimal resources, given a finite search space—in this case, 15 settings. In this section, we provide detailed visualization for Figure 4.

When performing ablation experiments on vision encoders, it’s common to randomly select a subset to train on. However, as shown in Figure 5, with 1000 runs of simulated ablation experiments, we found that to include the optimal vision representation 85.6% of the time,

- at least 13 out of the 15 settings need to be trained. This suggests that running a small subset of vision representations is unreliable, especially as the search space expands, making it increasingly unlikely to identify the true optimal representation by training only a subset.

In contrast, the AC policy requires only 4 full training runs on average to reach 87.72% Recall@3. For the most successful prediction benchmark, OKVQA, the policy successfully identifies the optimal configuration among the top three choices in 91.7% of cases, with only four language model finetuning runs across a 15-setting search space. This result shows that

AC policy significantly reduces the effort and cost of exploring vision representations for MLLMs.

Obtaining the AC policy results in significant computational cost reduction. After fitting the AC function, testing a new vision encoder only requires MLLM Stage 1 training (i.e. training a small projector), while traditional full MLLM post-training requires training both the projector and LLM. Concretely, we quantify computational cost using the number of trainable parameters, which is commonly correlated with both memory and compute. A decoder-only MLLM’s Stage 1 trains a small two-layer MLP, while Stage 2 involves finetuning a large language model with 7B parameters. This leads to a parameter ratio of about 0.003, yielding a 99.7% reduction.

#### 6 Limitation

Correspondence of Nature Image Correspondence of Image with Text

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

CLIP DINOv2 CLIP DINOv2

[Figure 52]

Figure 6: Visualization of correspondence on natural images and images containing text for CLIP and DINOv2.

We find that OCR-based benchmarks correlate less with the AC score than vision-based ones, making MME and MMMU outliers. For example, SigLIP2-Large@256 outperforms CLIP-Large@336 in both alignment (-1.81 vs. -1.97) and correspondence (16.75 vs. 15.66) but underperforms on MME due to OCR-heavy categories.

This discrepancy arises because vision representations exhibit different correspondence accuracy on different domain of images, as shown in Figure 6. The SPair-71k dataset (for the C score) focuses on natural images (e.g., cats, trains), whereas CLIP excels at text-based tasks not captured in SPair-71k. Likewise, our alignment score, calculated using LLaVA 558K (a subset of LAION, CC, SBU), lacks sufficient OCR, numeric, and symbolic data. Consequently, both the A and C scores underrepresent performance on OCR-related tasks.

To our knowledge, an OCR-specific correspondence dataset does not currently exist, and systematically designed OCR short caption datasets are scarce. We intend to pursue further investigation in this direction and encourage other researchers to do the same, as advancements in this area would be valuable for the broader MLLM community—particularly for tasks requiring the understanding of tables and charts, a fundamental capability.

#### 7 Conclusion

In this work, we present the Law of Vision Representation in MLLMs, uncovering a strong and quantifiable relationship between cross-modal alignment, feature correspondence, and overall model performance. Our proposed A and C scores capture this relationship with high fidelity, enabling a principled way to reason about vision representation quality. Through extensive evaluation, we demonstrate that model performance exhibits a clear quadratic correlation with AC scores, providing not only an interpretable diagnostic but also a predictive signal for selecting optimal visual features.

By leveraging this insight, we introduce the AC policy, a cost-efficient strategy to identify high-performing vision representations without repeatedly finetuning the language model. This yields up to 99.7% reduction in compute cost, transforming what was previously an expensive and combinatorially large search problem into a tractable one. Our findings offer a new perspective on vision-language alignment and open the door to more scalable MLLM development.

#### References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, et al. Evlm: An efficient vision-language model for visual understanding. arXiv preprint arXiv:2407.14177, 2024a.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024b.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185–24198, 2024c.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818–2829, 2023.

Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontierclass multimodal llms. arXiv preprint arXiv:2409.11402, 2024.

Jiali Duan, Liqun Chen, Son Tran, Jinyu Yang, Yi Xu, Belinda Zeng, and Trishul Chilimbi. Multi-modal alignment using representation codebook. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15651–15660, 2022.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller,¨ Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Chunjiang Ge, Sijie Cheng, Ziming Wang, Jiale Yuan, Yuan Gao, Jun Song, Shiji Song, Gao Huang, and Bo Zheng. Convllava: Hierarchical backbones as visual encoder for large multimodal models. arXiv preprint arXiv:2405.15738, 2024.

Shijie Geng, Jianbo Yuan, Yu Tian, Yuxiao Chen, and Yongfeng Zhang. Hiclip: Contrastive language-image pretraining with hierarchy-aware attention. arXiv preprint arXiv:2303.02995, 2023.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3608–3617, 2018.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pp. 4904–4916. PMLR, 2021.

O˘guzhan Fatih Kar, Alessio Tonioni, Petra Poklukar, Achin Kulshrestha, Amir Zamir, and Federico Tombari. Brave: Broadening the visual encoding of vision-language models. arXiv preprint arXiv:2404.07204, 2024.

Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. arXiv preprint arXiv:2402.07865, 2024.

Hyunjik Kim, George Papamakarios, and Andriy Mnih. The lipschitz constant of selfattention. In International Conference on Machine Learning, pp. 5562–5571. PMLR, 2021.

Hugo Lauren¸con, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2024.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13299–13308, 2024b.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.

Yifan Li, Yikai Wang, Yanwei Fu, Dongyu Ru, Zheng Zhang, and Tong He. Unified lexical representation for interpretable visual-language alignment. arXiv preprint arXiv:2407.17827, 2024c.

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26689–26699, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.

Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.

Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17627–17638, 2023.

Benlin Liu, Yuhao Dong, Yiqin Wang, Yongming Rao, Yansong Tang, Wei-Chiu Ma, and Ranjay Krishna. Coarse correspondence elicit 3d spacetime understanding in multimodal language model. arXiv preprint arXiv:2408.00754, 2024a.

Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for fine-grained multi-page document understanding. arXiv preprint arXiv:2405.14295, 2024b.

Fanfan Liu, Feng Yan, Liming Zheng, Chengjian Feng, Yiyang Huang, and Lin Ma. Robouniview: Visual-language model with unified view representation for robotic manipulaiton. arXiv preprint arXiv:2406.18977, 2024c.

Haogeng Liu, Quanzeng You, Xiaotian Han, Yiqi Wang, Bohan Zhai, Yongfei Liu, Yunzhe Tao, Huaibo Huang, Ran He, and Hongxia Yang. Infimm-hd: A leap forward in highresolution multimodal understanding. arXiv preprint arXiv:2403.01487, 2024d.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024e.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024f.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024a.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525, 2024b.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.

Run Luo, Yunshui Li, Longze Chen, Wanwei He, Ting-En Lin, Ziqiang Liu, Lei Zhang, Zikai Song, Xiaobo Xia, Tongliang Liu, et al. Deem: Diffusion models serve as the eyes of large language models for image perception. arXiv preprint arXiv:2405.15232, 2024.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pp. 3195–3204, 2019.

Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. Spair-71k: A large-scale benchmark for semantic correspondence. arxiv prepreint. arXiv preprint arXiv:1908.10543, 6:12–14, 2019.

Trong-Nguyen Nguyen and Jean Meunier. Anomaly detection in video sequence with appearance-motion correspondence. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1273–1283, 2019.

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller,¨ Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for highresolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men,

Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.

Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4938–4947, 2020.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In The Twelfth International Conference on Learning Representations, 2023.

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=ypOiXjdfnU.

Zineng Tang, Long Lian, Seun Eisape, XuDong Wang, Roei Herzig, Adam Yala, Alane Suhr, Trevor Darrell, and David M. Chan. Tulip: Towards unified language-image pretraining,

2025. URL https://arxiv.org/abs/2503.15485.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024a.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024b.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Wenxuan Wang, Quan Sun, Fan Zhang, Yepeng Tang, Jing Liu, and Xinlong Wang. Diffusion feedback helps clip see better. arXiv preprint arXiv:2407.20171, 2024b.

Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. arXiv preprint arXiv:2312.06109, 2023.

Sangmin Woo, Donguk Kim, Jaehyuk Jang, Yubin Choi, and Changick Kim. Don’t miss the forest for the trees: Attentional vision calibration for large vision language models. arXiv preprint arXiv:2405.17820, 2024.

Chenfeng Xu, Huan Ling, Sanja Fidler, and Or Litany. 3difftection: 3d object detection with geometry-aware diffusion features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10617–10627, 2024.

Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2955–2966, 2023.

Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783, 2021.

Hanrong Ye, De-An Huang, Yao Lu, Zhiding Yu, Wei Ping, Andrew Tao, Jan Kautz, Song Han, Dan Xu, Pavlo Molchanov, et al. X-vila: Cross-modality alignment for large language model. arXiv preprint arXiv:2405.19335, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Bohan Zhai, Shijia Yang, Chenfeng Xu, Sheng Shen, Kurt Keutzer, and Manling Li. Halleswitch: Controlling object hallucination in large vision language models. arXiv e-prints, pp. arXiv–2310, 2023a.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11975–11986, 2023b.

Guangcong Zhang and Patricio A Vela. Good features to track for visual slam. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1373–1382, 2015.

Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024a.

Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, et al. Ferret-v2: An improved baseline for referring and grounding with large language models. arXiv preprint arXiv:2404.07973, 2024b.

Junyi Zhang, Charles Herrmann, Junhwa Hur, Eric Chen, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. Telling left from right: Identifying geometry-aware semantic correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024c.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.

#### A Appendix

###### A.1 Theoretical Justification of Vision Representation with High Cross-modal Alignment

In Section 3.2, we state that when training an MLLM, if the vision representation is closely pre-aligned with the language distribution, then the pretrained language model requires less computational effort to bridge the gap between different modalities during finetuning. In this section, we show that using well-aligned vision representation, finetuning on multimodal data is about equivalent to finetuning on text-only data, eliminating additional effort beyond language finetuning.

Assume the vision embedding distribution Dimage and text embedding distribution Dtext are well-aligned in the MLLM. For a shared concept c, the image embedding after the alignment

module and its corresponding text embedding, Ecimage ∼ Dimage and Ectext ∼ Dtext, are close in distance, meaning:

∥Ecimage − Ectext∥ ≤ ϵ (6) where ϵ is a small constant. Given this condition, we can show that the output of the MLLM with multimodal embeddings [Ecimage, E1, E2, . . . , En] is close to the output with text-only embeddings [Ectext, E1, E2, . . . , En].

Since our language model f is well-trained and pre-normed, the input space to each transformer layer is bounded and compact, meaning that the values of the input are bounded by a small constant. This implies that the continuously differentiable function f is Lipschitz (Kim et al., 2021). This property ensures that small changes in the input of the language model of the MLLM result in small, controlled changes in the output:

f [Ecimage, E1, E2, . . . , En] − f [Ectext, E1, E2, . . . , En]

≤ L [Ecimage, E1, E2, . . . , En] − [Ectext, E1, E2, . . . , En] ≤ Lϵ. (7)

where L is the Lipschitz constant. This closeness in output distance implies that even with multimodal data, the pretrained language model mimics the training dynamics closely resemble language-only finetuning.

###### A.2 Theoretical Justification of Vision Representation with Accurate Correspondence

In Section 3.2, we state that if the vision representation ensures accurate correspondence, the attention within the image embedding is precise. In this section, we show that vision representation with accurate correspondence can help vision information retrieval in the attention mechanism. Therefore, more visual details are considered even if not attended by the text token.

Consider an input [E0image, E1image, E2, . . . , En] to the transformer, where the image embeddings E0image and E1image are derived from different patch of a high correspondence vision representation. By definition, the dot product E0image · E1image is large if the two corresponding original image patches share related information.

Suppose a text token E2 attends to E0image. We show that it is also able to retrieve E1image and vice versa. This can be demonstrated as follows:

(E2WQ) · (E0imageWK) √dk

score(E2, E0image) =

(8)

If score(E2, E0image) is high, and (E0imageWK)⊤(E1imageWK) is also large (assuming WK does not distort the vectors drastically), then by transitivity, score(E2, E1image) is also likely to be high. This transitivity ensures that attention is effectively spread across related visual information, enhancing the model’s ability to interpret visual content in greater detail.

###### A.3 Analysis of Quadratic Relationship between A and C

We first provide an intuitive explanation of why we choose to capture a quadratic relationship between cross-modal alignment and correspondence. Then, we analysis the coefficent of the fitted function to reveal a more in-depth relationship between these two factors.

Here’s our insight:

- • Good semantic correspondence means the vision part of MLLM can really tell what things are in a picture, even if they look a bit different. Like knowing all dogs are dogs, no matter their breed. But this alone isn’t enough for the MLLM to ”talk” about them. Those visual details need to be understood by the LLM.
- • Good language alignment means the visual information is perfectly understood by the language model. But if the initial visual information isn’t very good at telling different things apart (poor semantic correspondence), then aligning it perfectly won’t help much when the MLLM sees something new or slightly different.

So, for MLLMs to work best, both semantic correspondence and language alignment need to work together. More importantly, they interplay with each other more than just adding up:

If the visual features are already well-aligned with language (like in models such as CLIP), it actually helps the MLLM understand the semantic details better. That’s because if you can map ”dog” to the word ”dog” consistently, it reinforces the concept of ”dog” in the visual world.

But why is this interplay represented in a quadratic form (like a hill)? Think of it in this way: If language alignment is very weak, even if you make semantic correspondence a little better, the MLLM’s performance won’t jump much. It’s like trying to have a good conversation when you don’t speak the same language very well. Take a model like DINOv2. It’s great at seeing things, but it might not be as good at linking those visual details to language as, say, CLIP. In this case, making the visual part even better won’t help MLLM performance much because the language alignment is the bottleneck.

As language alignment improves, the MLLM can then really benefit from better semantic correspondence. It’s like they’re working together to reach that ”sweet spot” where the performance really shines. Both getting better at the same time, or one unlocking the potential of the other, leads to this curved, quadratic rise in performance.

In short, quadratic models include A2 and C2 and interaction term αAC which includes exactly the relationship we want to capture. In our paper, we test linear, second-degree polynomial relationships. Our result shows that the second-degree relationship fitted the best among others while still maintaining simplicity. More complex functional forms would obscure the clear individual contribution of A and C, as well as interaction between alignment and correspondence.

Next, We discuss the interation alignment and correspondence from two aspects: the ideal relationship between A and C suggested by the law, and the actual relationship we observe from existing encoders. To provide the ideal relationship, we extract the cross-term coefficient (αAC) of the fitted functions as shown in Table 4.

We observe two kind of interactions:

##### Benchmark αAC

Synergy (αAC ≫ 0): MMBench, MME, TextVQA, VizWiz, ScienceQA, and SeedBench all have αAC > 0. For these six tasks, performance improves more than additively when both A and C increase.

MMBench +1.1301 MME +1.5819 MMMU -0.4872 OKVQA -0.0364 TextVQA +4.9804 VizWiz +2.9964 ScienceQA +4.4644 SeedBench +0.5772

Mild trade-off (αAC < 0): MMMU and OKVQA show αAC < 0, indicating a slight penalty if A and C are both pushed upward (i.e., diminishing returns).

These results demonstrate that on most benchmarks, alignment and correspondence actually reinforce each other (positive interaction). Therefore, A and C strengthen each other, so the ideal scenario would be increasing both to achieve the best performance.

Table 4: Interaction term αAC of fitted function for all eight benchmarks.

However, from our observations of A and C of vision encoders, the factors often have a trade-off: Intuitively, A measures how well image features align to a shared text embedding, whereas

C captures intra-image consistency (e.g., visual cluster compactness). Forcing very high alignment often collapses visual cluster structure (hurting correspondence). In practice, looking at the A and C scores in Appendix Section A.5, common visual encoders often have high A but low C (CLIP and SigLIP) or low A but high C (DINOv2 and Diffusion features).

- A.4 All Settings Benchmark Performance

In this section, we present the performance results of all 15 vision representation settings, as summarized in Table 5. The benchmarks we evaluated include:

- • MMBench (Liu et al., 2023): A set of multiple-choice questions designed to assess 20 different ability dimensions related to perception and reasoning.
- • MME (Fu et al., 2023): A dataset focused on yes/no questions, covering areas such as existence, counting, position, and color, primarily based on natural images.
- • MMMU (Yue et al., 2024): Multiple-choice questions targeting college-level subject knowledge and deliberate reasoning, primarily testing the language model’s abilities.
- • OKVQA (Marino et al., 2019): Open-ended questions based on the MSCOCO (Lin et al., 2014) dataset, spanning 10 different knowledge categories.
- • TextVQA (Singh et al., 2019): Open-ended questions designed to evaluate the model’s OCR capabilities.
- • VizWiz (Gurari et al., 2018): Open-ended questions sourced from people who are blind, aimed at testing the model’s OCR capabilities.
- • ScienceQA (Lu et al., 2022): A multiple-choice science question dataset, with 86% of the images being non-natural, covering topics in natural science, social science, and language science.
- • SEED-Bench (Li et al., 2024b): A benchmark consisting of multiple-choice questions designed to assess both spatial and temporal understanding.

- A.5 All Settings AC Scores We provide the AC scores of all 15 vision representation settings, as summarized in Table 6.
- A.6 More Visualization of Correspondence

We provide additional visualizations of correspondence for four different vision representations: CLIP, SigLIP, DINOv2, and Stable Diffusion 1.5. Figures 7 and 8 display pairs

CLIP@336 CLIP@224 OpenCLIP SigLIP Base SigLIP Large

- MMBench 64.26 64.18 63.40 61.86 65.46 MME 1502.70 1449.64 1460.28 1425.00 1455.18

- MMMU 35.0 36.2 37.2 35.8 36.6 OKVQA 53.20 56.13 56.36 54.01 57.02 TextVQA 46.04 42.67 40.13 36.00 42.44 VizWiz 54.27 51.69 52.11 53.17 51.49 ScienceQA 69.26 68.82 67.87 66.88 70.20 SEED-Bench 66.09 65.13 64.71 64.40 66.28

C+D@224 C+D@336 SigLIP2 Large DINOv2 DiT

MMBench 65.72 65.12 67.35 58.50 33.68 MME 1436.42 1475.19 1486.66 1295.47 902.00

- MMMU 36.9 34.6 34.2 34.6 32.7 OKVQA 55.94 56.92 57.57 54.78 33.75 TextVQA 40.04 46.17 47.2 14.27 10.82 VizWiz 54.04 53.44 56.55 49.67 49.92 ScienceQA 69.11 67.63 69.41 65.15 63.46 SEED-Bench 65.39 66.38 68.22 61.39 40.66

SDXL SD3 SD2.1 SD1.5 SDim

MMBench 43.73 32.82 28.87 42.53 52.84 MME 1212.69 843.43 905.27 1163.90 1205.33 MMMU 32.8 32.4 32.8 33.9 33.7 OKVQA 41.78 34.95 34.41 39.14 46.04 TextVQA 11.81 10.77 10.46 11.64 13.77 VizWiz 47.14 47.12 46.59 50.14 47.33 ScienceQA 65.25 62.27 62.67 63.31 66.34 SEED-Bench 53.78 38.94 38.82 50.00 50.33

- Table 5: Benchmark performance of all 15 settings. C+D means feature combination of CLIP and DINOv2. The table provides data points for function fitting and is not intended for comparison.

CLIP@336 CLIP@224 OpenCLIP SigLIP-B SigLIP-L Correspondence

- PCK@0.10 15.66 14.30 16.22 12.89 13.66 Cross-modal Alignment Log likelihood -1.97 -1.98 -1.93 -1.92 -1.83

C+D@224 C+D@336 SigLIP2-L DINOv2 DiT

Correspondence PCK@0.10 23.62 26.08 16.75 24.51 1.91 Cross-modal Alignment

- Log likelihood -1.96 -1.95 -1.81 -2.32 -3.76 SDXL SD3 SD2.1 SD1.5 SDim

Correspondence PCK@0.10 16.52 3.09 6.99 22.02 20.90 Cross-modal Alignment

- Log likelihood -2.69 -4.13 -2.81 -2.53 -2.37

- Table 6: AC scores of all 15 settings. C+D means feature combination of CLIP and DINOv2. The table provides data points for function fitting and is not intended for comparison.

of source-target images for each of the four vision representations. In each pair, the left image is the source, and the right image is the target. The red dot on both images indicates the predicted key points using the vision representation. Ideally, these key points should correspond to the same semantic meaning. For example, a red dot on the ”left cat ear” in the source image should correspond to the ”left cat ear” in the target image. The green areas highlight regions of relatively high similarity with the source points.

- In Figure 7, DINOv2 demonstrates superior correspondence for natural images compared to the other vision representations. It accurately matches small parts of the cat between the left and right images, whereas CLIP struggles to correctly identify and align features such as left, right, front, and back.
- In Figure 8, , the CLIP family shows precise correspondence for text within images. For instance, when the source image points text like “LLaVA” or “VQAv2”, CLIP accurately matches all instances of the text in the target image. In contrast, other vision representations known for ”accurate correspondence” in computer vision, such as DINOv2 and Stable Diffusion, fail to provide the same level of accuracy when dealing with images containing text. This emphasizes a key distinction in selecting vision representations for computer vision tasks versus multimodal large language models (MLLMs).

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

CLIP SigLIP DINOv2 SD1.5

Figure 7: Correspondence of natural images for different vision representations.

###### A.7 Pseudo Code

Computing the A score is a simple loss calculation using the frozen MLLM; therefore, we provide pseudocode for the other algorithms, including the computation of the C score at Algorithm 1, region-based sampling at Algorithm 2, and the AC policy at Algorithm 3.

###### A.8 Limitation of AC Policy

- Figure 9 shows two benchmarks—MME and MMMU—where the AC policy fails to predict the optimal vision representation. For details on the reason of this behavior, please refer to Section 6.

###### A.9 AC Policy Recall@1

- Algorithm 1: COMPUTE C SCORE

Input: Set of paired images with key points S from SPair-71k; Vision encoder E; Threshold T.

Output: C score for vision encoder E. // Initialize correspondence lists

G ← [] ; // Ground truth keypoint correspondences P ← [] ; // Predicted keypoint correspondences

foreach (I1, K1, I2, K2) ∈ S do

// Extract feature representations

- F1 ← E(I1);
- F2 ← E(I2); // Compute similarity matrix Ssim ← F1 · F2T; // Transform keypoints from I1 to I2 Kˆ2 ← calculate keypoint transformation(Ssim, K1); // Store ground truth and predicted keypoints G.append(K2); P.append(Kˆ2);

// Compute correctness score Eerror ← Euclidean distance(P, G); Ccorrect ← sum(Eerror < T); Cscore ← total keypointsCcorrect in K2;

- Algorithm 2: REGION-BASED SAMPLING

Input: k A and C score pairs from models ACs; past sampled models; current sampling level (1 to k′, increments when regions are exhausted as each region is sampled only once)

Output: Sampled model to train next regions ← {}; for AC ∈ ACs do

region key ← determine region(A,C, level) ; // Identify the region based on A

and C coordinates

regions[region key].append((model, A,C)); Remove models in past sampled from regions; remaining regions ← keys of regions; chosen region ← randomly select from remaining regions; model ← randomly select from regions[chosen region];

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

CLIP SigLIP DINOv2 SD1.5

Figure 8: Correspondence of images with text for different vision representations.

###### Algorithm 3: AC POLICY

Input: k vision encoders with pretrained projectors V; computation budget k′ Output: A ranking of k MLLMs based on performance ACs ← [(Compute A Score(v),Compute C Score(v)) | v ∈ V]; past sampled ← []; train ACs ← []; train performance ← []; for i ← 1 to k′ do

model ← Region based Sampling(ACs, past sampled); performance ← Fully train model; train ACs.append(AC of model); train performance.append(performance); past sampled.append(model);

poly ← PolynomialFeatures(degree = 2); transformed train ACs ← poly.fit transform(train ACs); regression ← LinearRegression(); regression.fit(transformed train ACs, train performance); ranking ← Rank V by regression predictions on ACs;

[Figure 93]

[Figure 94]

MME MMMU

In Section 5.5, we used Recall@3 as a metric for AC policy effectiveness to capture potential performance fluctuations during MLLM training and evaluation. Our AC policy’s core aim is to narrow the search space to a small set of promising vision encoders, rather than pinpointing a single best encoder. Recall@3 directly quantifies whether the true top-performers appear within our top-3 recommendations, which:

Figure 9: Limitation of AC policy.

[Figure 95]

[Figure 96]

[Figure 97]

SEED-Bench MMBench OKVQA

[Figure 98]

AC Predicted ScienceQA

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

TextVQA VizWiz

PredictionRecall@3

[Figure 107]

[Figure 108]

Random Predicted

OptimalVR

[Figure 109]

Number of Full Training

- Figure 10: Number of full training (LLM finetuning) cycles required to include the optimal vision representation within the top-1 predictions (Recall@1).

- • Reflects practical workflows—researchers typically inspect multiple (“top 3–5”) candidates, not just one.
- • Smooths score noise—unlike Top-1 accuracy, Recall@3 tolerates minor ranking fluctuations while still ensuring coverage of strong encoders.
- • Focuses on coverage—rather than overall ranking correlation (e.g. Pearson’s r) or mean reciprocal rank (which over-emphasizes exact positions), Recall@3 measures actionable retrieval quality: “Is the best encoder in my shortlist?” In other words, we put coverage in higher priority, “Is the best encoder in my shortlist” is more important than is the top-3 order correct.

While we believe this makes Recall@3 the most appropriate metric for AC’s intended usecase, we also report Recall@1 for predicting the optimal vision representation. As shown in Figure 10, AC policy consistently outperforms random testing even under this stricter metric.

