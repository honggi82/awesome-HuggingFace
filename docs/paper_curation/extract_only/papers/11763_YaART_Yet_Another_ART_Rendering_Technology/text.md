# arXiv:2404.05666v1[cs.CV]8Apr2024

## YaART: Yet Another ART Rendering Technology

Sergey Kastryulin1,2, Artem Konev1∗, Alexander Shishenya1∗, Eugene Lyapustin1∗, Artem Khurshudov1∗, Alexander Tselousov1, Nikita Vinokurov1, Denis Kuznedelev1,2, Alexander Markovich1, Grigoriy Livshits1, Alexey Kirillov1,3, Anastasiia Tabisheva1, Liubov Chubarova1,4, Marina Kaminskaia1, Alexander Ustyuzhanin1, Artemii Shvetsov1, Daniil Shlenskii1,2, Valerii Startsev1, Dmitrii Kornilov1, Mikhail Romanov1, Artem Babenko1†, Sergei Ovcharenko1†, and Valentin Khrulkov1†

1 Yandex 2 Skolkovo Institute of Science and Technology

- 3 Moscow State University
- 4 Higher School of Economics

Abstract. In the rapidly progressing field of generative models, the development of efficient and high-fidelity text-to-image diffusion systems represents a significant frontier. This study introduces YaART, a novel production-grade text-to-image cascaded diffusion model aligned to human preferences using Reinforcement Learning from Human Feedback (RLHF). During the development of YaART, we especially focus on the choices of the model and training dataset sizes, the aspects that were not systematically investigated for text-to-image cascaded diffusion models before. In particular, we comprehensively analyze how these choices affect both the efficiency of the training process and the quality of the generated images, which are highly important in practice. Furthermore, we demonstrate that models trained on smaller datasets of higher-quality images can successfully compete with those trained on larger datasets, establishing a more efficient scenario of diffusion models training. From the quality perspective, YaART is consistently preferred by users over many existing state-of-the-art models.

Keywords: Diffusion models · Scaling · Efficiency

### 1 Introduction

Large-scale diffusion models have recently achieved unprecedented success in text-conditional image generation, which aims to produce high-quality images closely aligned with user-specified textual prompts. The state-of-the-art models [1, 3, 7, 11, 30, 31, 37] can generate complex, photorealistic images, which paves the way to their ubiquitous usage in applications, including web design, graphics editors, e-commerce and others.

* Equal contribution † Equal senior authors

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### Fig. 1: RL-aligned YaART generates visually pleasing and highly consistent images.

Outstanding image generation ability typically comes at a price of scale, i.e., state-of-the-art systems require large models, large training datasets, and vast amounts of GPU resources. At the moment, it is not completely clear what the trade-offs are between these three aspects of scale and the generation quality. Moreover, it is unclear if one should prefer large datasets of random web images over smaller datasets containing only well-filtered, high-quality ones.

In this paper, we introduce YaART — a cascaded text-to-image diffusion model enhanced with human preference tuning. YaART is specifically designed to optimize data and computational resource usage. We thoroughly explore the impact of model and dataset sizes on generation performance and training costs. Furthermore, we investigate the quality-vs-quantity trade-off for the training dataset construction and reveal that training on smaller datasets of high-fidelity samples provides performance on par with models trained on larger datasets. Additionally, we show that the model size can be successfully traded for training time, i.e., smaller models can perform on par with the larger ones if their training time budget is long enough. Regarding quality, YaART is consistently preferred in human evaluations over well-known benchmarks, such as SDXL v1.0, MidJourney v5, Kandinsky v3, and OpenJourney.

To sum up, our main contributions are:

- – We introduce YaART, an RLHF-tuned cascaded diffusion model that outperforms the established models regarding side-by-side human evaluation of image realism and textual alignment.
- – We systematically study the influence of interactions of model and dataset sizes on the training efficiency and the final model quality. To the best of our knowledge, this is the first work to extensively investigate the scalability of diffusion models in convolution-based cascaded schemes using a reliable human evaluation protocol.
- – We investigate the practically important trade-off between data quality and quantity for pre-training diffusion models. We reveal that competitive results can be obtained by training on a small amount of high-quality data.
- – To the best of our knowledge, we are the first to thoroughly describe RLHFtuning of the production-grade diffusion model. We show that RLHF tuning is an important ingredient of advanced performance; in particular, it substantially improves image aesthetics and reduces visual defects.

### 2 Related work

Despite the recent progress on large-scale diffusion models, achieving high-fidelity synthesis of high-resolution images remains a formidable challenge. Two primary approaches address this issue. The first, employed in models like DALLE-2 [35] and Imagen [39], termed ‘cascaded’, involves a sequential pipeline of multiple diffusion models that progressively generate higher-resolution images. The second approach, known as the Latent Diffusion Model [37], leverages score matching on the latent space of images generated by a pre-trained VAE. This method has

yielded impressive results and has been adopted in various subsequent works such as Stable Diffusion XL [31], DALL-E 3 [3], PixArt-α [7], EMU [11].

Traditionally, the denoiser network in diffusion models is implemented as a variant of the convolutional U-Net model [38] with additional attention blocks. However, inspired by the success of the transformer architecture [43] in language modeling, recent research [2,7,29,48] has replaced the convolutional backbone with a transformer backbone, enabling straightforward parameter expansion.

In our paper, we adhere primarily to the original approach proposed in DALLE 2 and Imagen. Specifically, we employ a cascaded scheme of three diffusion models and a convolutional backbone for denoisers.

Alignment of Text-to-Image Models on Human Preferences. Finetuning diffusion models on a clean filtered dataset has been shown to drastically improve the quality of generated images; however, improving quality even beyond the level of the available datasets requires other training methods. Approaches like [26] and [10] suggest generating a dataset with the current diffusion model, ranking them with scoring models, rejecting samples with low scores, and fine-tuning diffusion model on this dataset with MSE loss or reward-weighted MSE loss. In [15], [4], and [27], authors adapt REINFORCE and PPO for tuning diffusion models. Papers [45] and [44] exploit DPO for improving quality of generated images, which allows to avoid training explicit reward functions. Some methods like [10] rely on differentiable scoring functions while tuning diffusion models.

Diffusion models require multiple steps of inference for generating a single image; thus, methods like [10] that utilize differentiable rewards and backpropagate through these rewards and all the iterations of image generation face a couple of significant problems: vanishing gradients and large GPU memory requirements for each sample. RL-based approaches like [15], [4], and [27] require agents to generate trajectories online during training, which induces additional computational cost. Besides, these methods use on-policy algorithms, which do not allow them to use images from any sources but the current model itself. DPO-based approaches like [45] perform on par with D3PO but have the advantage that they can exploit samples from any other sources.

Scaling of Diffusion Models. The remarkable progress of text-to-image generative models is largely driven by improvements in scale, where bigger models are trained on larger datasets for longer schedules. However, the amount of research aimed to understand the scalability of text-to-image models is still limited. Saharia et al. [39] investigate the influence of U-Net model size on image quality and conclude that increasing the model size has little effect, whereas scaling the text encoder has a greater impact. Caballero et al. [6] build a power scaling law from experiments on ImageNet 64x64 [12], and LSUN Bedrooms 256x256 [46] datasets for single-step diffusion models, confirming the potential benefit of size and compute scaling. Diffusion Transformer (DiT) [29] study the effectiveness of usage of transformer models in the Latent Diffusion [37] framework and finds

that larger DiT models are more computationally efficient and that scaling the DiT improves generation quality in all stages of training.

All mentioned studies mainly rely on FID [18] for image quality estimation, which has been shown to correlate poorly with human judgments [5,40]. Moreover, the questions of data quality and quality remain unexplored. In this work, we study the scalability of the convolutional diffusion model in the cascade framework based on large-scale study with human annotators. We investigate the influence of model and dataset sizes on end quality, specifically considering the data quality-quantity trade-off.

### 3 Approach

Here, we describe our approach to train large-scale diffusion models.

High-resolution image synthesis. We initially opted for the cascaded pixelbased diffusion approach because of several considerations. Firstly, by leveraging a pre-trained VAE to extract latent codes for LDM, the quality of the generated images is inherently constrained by the VAE’s performance, which often has limited capacity. Although the VAE could be fine-tuned later, such adjustments might not substantially enhance its overall capability due to inherent capacity limitations and difficulties with training VAE-based models.

In contrast, cascaded diffusion models allow for important practical advantages. In cascaded models, the “decoder” function is a stack of two super-resolution diffusion models that operate independently of the primary pixel-space generative model. This autonomy allows for the natural interchangeability of the superresolution blocks. Furthermore, the diffusion decoder’s increased capacity and enhanced generative capability significantly contribute to improving the fidelity of the resultant images.

Large high-quality datasets for model pre-training. One of the most difficult challenges in training a large-scale text-to-image model is curating a high-quality dataset of text-image pairs. We have continually refined the process of assembling such a dataset, ultimately converging on a multi-stage procedure. This approach encompasses separate pre-filtration stages of image and text filtration and the final stage of selecting image-text pairs according to their overall fidelity with a focus on relevance.

Boosting relevance and quality via supervised fine-tuning. Pre-trained text-to-image models often encounter difficulties in image generation. Exact prompt following, in combination with high visual attractiveness, is often a challenge. Our strategy involves supervised fine-tuning using a manually selected subset of data with exceptional aesthetic quality and detailed descriptions. This process notably enhances prompt following and relevance, focusing on the latter.

Aesthetic refinement through Reinforcement Learning. In the final stage, we further elevate image aesthetics and quality through Reinforcement Learning (RL). We directly use the signal provided by human annotators to improve the aesthetic properties of the produced images.

To sum up, our approach has three main stages. First, we pre-train a diffusion model on a large-scale dataset, followed by the fine-tuning stage. This model is further polished via RL-tuning, which incorporates human feedback. Next, we discuss each component in detail.

#### 3.1 Cascaded Diffusion

As discussed above, we base our approach on the framework of cascaded diffusion models. Specifically, we follow Imagen [39] to design the cascaded diffusion model. The initial diffusion model in the cascade, which produces 64 × 64 images (GEN64), and the first super-resolution model, which performs up-sampling 64 → 256 (SR256), follow the architecture of U-Net in [14,35,37,39]. The GEN64 is conditioned on text via the cross-attention mechanism as in [37]. The SR256 model is purely convolutional with a single self-attention in the middle block. Additionally, both models are conditioned on text via modulation of residual blocks by linear projections of the mean-pooled text embeddings (as described in [35]). For the final super-resolution stage 256 → 1024 (SR1024), we utilize the text-unconditional Efficient U-Net backbone with the same configuration as the third stage in [39]. Similarly to Imagen, we utilize large pre-trained text encoders for conditioning. Specifically, we use a CLIP [34]-like proprietary model of size ∼ 1.3B which follows BERT [13]-xlarge architecture. It was pre-trained on a mixture of public and proprietary datasets of text-image pairs in a setup similar to OpenAI CLIP models and demonstrated competitive performance on public benchmarks. Our preliminary smaller-scale experiments verified that this text encoder provides better quality of generated images according to human raters than the other choices such as CLIP/OpenCLIP [8]/FlanT5-XXL [9] encoders. In all our experiments, we use the standard linear noise scheduler [14] with continuous time; we follow the implementation in [23]. The timestep t is provided to the model via its respective LogSNR(t), computed by the scheduler.

#### 3.2 Data Selection Strategy

Our proprietary pool of images is comprised of 110B image-text pairs. For each pair we have a pre-calculated set of scores produced by multiple learned predictors. These predictors are fully connected classifiers, trained on top of image and text features, extracted by our proprietary visual and textual foundation models. For example, we have various classifiers for image aesthetics and quality, watermark detectors, NSFW detectors, a binary classifier to check the presence of text within an image, and more.

Selecting the best images. We formulate several stages of filtering for the initial pool of images, starting from defining the Image Score as a linear combination of the image-based predictors with weights tuned on the SAC [32] dataset. By visual inspection in the preliminary experiments, we confirm that the visual appeal of the images is correlated with the Image Score predictions. We decided to keep only the top 1⁄3 of the images because most of the images below this threshold had low visual attractiveness. Additionally, we remove NSFW images utilizing our proprietary classifiers and take ones with a size in the range of [512, 10240] pixels and an aspect ratio in the range of [0.5, 2]. These values were chosen to filter out images with unusual sizes that may increase computational and engineering burden on our image processing pipeline or have undesirable content. Then, we filter out images that do not pass thresholds for the image quality classifiers, specifically those that contain high levels of noise, blurriness, watermarks, or a checkered background and were subjected to a high degree of compression. Finally, we manually select thresholds for aesthetics classifiers, trained on AVA [28] and TAD66k [16] datasets. We also explicitly control background monotonicity by splitting the dataset into two parts and allowing only 10% of the samples during training to have a monotonic background. The motivation is to emphasize images rich with details during training while retaining the ability to generate mono backgrounds if required.

Our SR1024 model does not have a condition on the texts. Hence, we obtain the final version of the 180M Main SR dataset by further tightening the filtering conditions at this stage, especially those related to image quality.

Text filtration. First, we focus on English texts only by recognizing text language with our proprietary language detector. Next, we manually annotate a random subset of 4.8K texts so that each raw input text receives its cleaned-up version or an empty mark if deemed unsuitable for training. Finally, we compile the training dataset from the refined versions of the texts, filtering out empty text entries. We then fine-tune our proprietary 180M Language Model on that dataset and use its prediction as a factor of text quality.

Combining All Together. Previously, we discussed the image filtering process and computation of various image and text factors. At this point, we have a pool of 2.3B of relatively poor-quality data, which we aim to filter out further. For that, we manually label 66K image-text pairs for their visual attractiveness, textual descriptiveness, and relevance on a scale from 1 to 3. Unlike the more frequently used Likert scale, the simplified scale balances the descriptiveness of the score with the noise in the votes of the assessors. After that, we train a CatBoost [33] model on a set of 56 factors, among which there are 6 variations of CLIP scores, 38 text-only, and 12 image-only factors, to obtain Sample Fidelity Classifier (SFC).

All image-text pairs from the previous filtration stage are sorted according to the SFC model prediction, and the top pairs are selected as the final training

Model Training Stage Image Resolution #Samples Training Steps Batch Size Learning Rate

GEN64 Pre-training 64x64 330M Main Gen 1100K 4800 1 ×10−4 SR256 Pre-training 256x256 330M Main Gen 1500K 960 5 × 10−5 SR1024 Pre-training 1024x1024 (crops) 180M Main SR 1000K 512 5 × 10−5

GEN64 Fine-tuning 64x64 50K 40K 240 1 × 10−5 SR256 Fine-tuning 256x256 50K 40K 96 1 × 10−5 SR1024 Fine-tuning 1024x1024 180M Main SR 200K 122 2 × 10−6

GEN64 RL-alignment 64x64 300K 50K 192 1 × 10−4

- Table 1: A summary of training stages of the YaART model cascade. Note that Main Gen and Main SR are large-scale datasets designed with an intent to cover a broad range of concepts while the fine-tune datasets consist of carefully selected image-text pairs, collected with a desire to boost specific quality aspects such as text-image relevance and image aesthetics.

pool. We select a threshold so the final dataset has 300M images with a nonmonotonic background. To obtain the final 330M Main Gen dataset for pretraining, we also add a random sample of 30M pairs containing images with monotonic backgrounds.

#### 3.3 Model Training

Pre-training. The model is initially pre-trained on a dataset collected as described above. The GEN64 model with 2.3B parameters is trained on 160 A10080GB GPUs with the total batch size being 4800 for 1.1·106 training iterations. The model is trained with automated mixed-precision using ZeRO [36]. We use Adam optimizer with learning rate being 10−4 and β1 = 0.9,β2 = 0.98; we observed that the smaller values of β2 enhance training stability in half precision. The SR256 model with 700M parameters is trained similarly on 80 A100-80GB GPUs with total batch size being 960 and reduced learning rate 5 × 10−5; we train it for 1.5 × 106 iterations. The SR1024 EfficientUNet [39], with 700M parameters, is trained on 256 × 256 crops under a similar setup, albeit with a smaller batch size of 512. We summarize the main information about training in Table 1. Detailed model configurations are provided in the supplementary material.

Fine-tuning. To fine-tune the GEN64 and SR256 models, we manually collected a dataset of 50K samples consisting of high-quality images and relevant descriptive captions. At this stage, we value relevance more than image attractiveness, which we align later via RL. We also found that fine-tuning the SR1024 model on full-resolution images from the 180M Main SR dataset leads to a noticeable improvement in the fine-grained details and sharpness of the images.

RL Alignment. For RL alignment, we exploit the PPO [41] approach with ϵ = 0.5 for loss computation and follow DDPO with importance sampling [4]. We also use a value function that is trained with MSE loss. However, we do not use KL-divergence and classifier-free guidance because, in our experiments,

Quality Aspects YaART 2.3B RL Beauty Defects Alignment Overall vs MidJourney v5 0.58 ± 0.03 0.49 ± 0.01 0.52 ± 0.01 0.51 ± 0.01 vs SDXL 0.75 ± 0.03 0.68 ± 0.02 0.53 ± 0.02 0.77 ± 0.01 vs Kandinsky v3 0.69 ± 0.07 0.69 ± 0.03 0.46 ± 0.03 0.72 ± 0.04 vs OpenJourney 0.83 ± 0.09 0.75 ± 0.04 0.71 ± 0.01 0.89 ± 0.01

- Table 2: We compare YaART with state-of-the-art based on three main evaluation criteria and overall human preferences. Bold denotes statistically significant difference in overall models’ quality.

they worsen convergence speed, reward growth, and slow down the generation of trajectories.

For text-to-image diffusion models, the full probability density pθ (xt−1|xt,c) is a product of multiple Gaussian probability density functions for each pixel and channel. It causes rapid deviation from the value of pθ

(xt−1|xt,c), which leads to a significant loss of precision. To cope with this problem, we split each sample xt into patches and compute the objective for each patch independently, followed by averaging over all the patches.

old

We use three reward models: Relevance-focused OpenCLIP ViT-G/14 [22] and predictors of image Consistency (lack of defects) and Aesthetics, trained on our human preference data in a Siamese [25] manner with hinge loss as a fully-connected head over our in-house image foundation model. We independently compute loss for each reward, followed by averaging them with weights. We aim to ensure the rise of Aesthetics and Consistency rewards without significantly declining the Relevance reward. We experimentally found that weighting, resulting in the same rewards scale, works well in our setup.

To reduce the latency for saving and reading model updates, we train the condition encoder of our model and LoRA [21] for the remaining layers. We also tune the frequency of the updates so they occur often enough, as it is crucial for the on-policy algorithms. We use 100 steps of p-sample [19] so the agent produces 100 × batch size samples for the loss optimization.

#### 3.4 YaART

Following the pre-training, fine-tuning, and RL alignment pipeline, we obtain the final YaART model. While the combination of pre-training and fine-tuning is relatively common in the modern diffusion training pipelines, we emphasize the importance of the RL alignment stage, which pushes generation quality even further. In particular, Figure 2 shows that throughout this stage, the model learns to generate more aesthetic and consistent (less defective) images while preserving image-text relevance tuned on the fine-tuning stage.

We compare YaART with MidJourney v5, Stable Diffusion XL with the refiner module, Kandinsky v3, and OpenJourney. Table 2 demonstrates that

RL Reward Dynamics

SbS vs Baseline Dynamics

- 0.5

- 1

70

Reward Model Relevance Aesthetics Consistency

WeightedReward

60

%Wins

50

0

40

0 10K 30K 50K

0 10K 30K 50K

Training Steps

Training Steps

- Fig. 2: An evolution of rewards (left) leads to an increase of human preference rate (right) throughout the RL alignment stage.

YaART performs on par with MidJourney v5 and is generally preferred over other counterparts.

### 4 Experiments

In this section, we first describe the model evaluation approach used in our experiments. After that, we report the results of a systematic analysis that provides an in-depth understanding of training dynamics for different choices of data and model sizes, as well as the trade-off between data quality and quantity. Finally, we discuss the relationship between the quality of pre-trained models and their fine-tunes.

#### 4.1 Evaluation Setting

Prompts. For evaluation, we use two prompt sets: our practice-driven YaBasket300 (Figure 3) and DrawBench [39] – a de-facto standard in the field of text-toimage models. YaBasket complements public benchmarks with several important practical use cases that have not been previously covered. Specifically, we enrich small subsets of PartiPrompts [47] and Winoground [42], placed into the Common Sense category, with popular requests from users of image generation models and products-driven prompts. For completeness, we perform comparisons with established reference models in 3.4 using both prompt sets, while all remaining sections use only the YaBasket set. We publish the prompts from YaBasket for their subsequent usage in the community†.

Metric. Our primary evaluation metric is the Side-by-side (SbS) comparisons between two images generated from the same prompt. To compute SbS, we use a proprietary crowdsource platform with non-expert assessors. Before participating in the labeling process, each candidate assessor must pass an exam and achieve at least an 80% accuracy rate among a pre-defined set of 20 assignments. After that, we ask assessors to select one of the two images shown side-by-side based on three evaluation criteria placed in order of their importance:

† Prompts and additional information are available on the YaART project page.

###### Food

User Inputs

Games

15%

18%

###### Clothes

Interior 10%

16%

6%

35% Products

10%

16%

Merch

Jewelry

47%

13%

13%

Common Sense

Furniture

Logo

(b)

(a)

- Fig. 3: The content of YaBasket. The three major prompt categories (left) include Products, almost equally split into eight sub-categories (right).

- 1. Defectiveness represents the number of visual artifacts and inconsistencies on an image. Any distortions of objects, limbs, faces, and muzzles belong to this aspect.
- 2. Image-text Relevance reflects prompt following, including the correctness of counting, sizes, and positions of objects, their relation, and properties explicitly stated in the provided textual description.
- 3. Aesthetic Quality is responsible for image style and attractiveness. It includes color choice and balance, the beauty of the background and environment, and following the basic rules of photo composition, such as centering and the rule of thirds.

If the images are indistinguishable based on all three criteria, they are said to have equal quality, and neither of the images gets a vote. Each pair of images is labeled by three assessors, after which an image with more votes gets a point. We conclude that one set of images is better than another by contesting a null hypothesis of their equality by conducting a two-sided Binomial test with a p-value of 0.05. We do not report the FID score [18] because of its poor correlation with human judgments, especially for higher-quality generations. This phenomenon was already discussed in recent papers (e.g., [24, 31]), and our analysis in the supplementary material confirms that.

- 4.2 Scaling Experiments

In this section, we aim to answer the following questions: i) How much compute is required to train a text-to-image model to reach its quality limits? ii) Is it possible to obtain a higher-quality model by trading its size (by reducing the number of parameters) for compute time during pre-training? iii) How do model and dataset sizes affect the model training dynamic?

First, we define a set of experimental GEN64 models of different sizes: [233M, 523M, 929M, 1.45B], scaled only by width in terms of the number of channels in the convolutional layers. All other details, such as the number of cross-attention layers, their position, etc., remain as discussed in section 3.1.

vs YaART 2.3B, Full Dataset

50 vs SD v1.4, Full Dataset

50

%Wins

25

25

| |
|---|

| |
|---|

| |
|---|

Test Model

233M 523M

929M

1450M

0 100K 200K 300K

2.5K 5K 7.5K 10K

0 100K 200K 300K

2.5K 5K 7.5K 10K

vs YaART 2.3B, 1/4 Dataset

50 vs SD v1.4, 1/4 Dataset

50

|| |
|---|
| | |
|---|---|---|
|| |
|---|
<br><br>| | |

%Wins

| |
|---|

| |
|---|

| |
|---|

25

25

| |
|---|

| |
|---|

Test Model

| |
|---|

| |
|---|

| |
|---|

233M 523M

929M

1450M

0 100K 200K 300K

2.5K 5K 7.5K 10K

0 100K 200K 300K

2.5K 5K 7.5K 10K

vs YaART 2.3B, 523M Model

50 vs SD v1.4, 523M Model

50

| || |
|---|
<br><br>| |
|---|---|---|
| || |
|---|
<br><br>| |

%Wins

| |
|---|

| |
|---|

| |
|---|

| |
|---|

25

25

Sample Fraction

1

1/8

1/4

1/16

0 100K 200K 300K

2.5K 5K 7.5K 10K

0 100K 200K 300K

2.5K 5K 7.5K 10K

Training Steps

GPU hours

Training Steps

GPU hours

- Fig. 4: Scaling up the convolutional GEN64 model improves training, leading to higher quality models. Larger models train faster regarding training steps and GPU hours, leading to better results across different dataset sizes (top and middle rows). Dataset size weakly influences the model’s end quality (bottom row).

Learning Rate and Batch Size. Following the best practices from the area of Large Language Models [20], we first aim to estimate the optimal hyperparameters (learning rate and batch size) for training models of various sizes. For that, we perform more than 100 short pre-trainings of the experimental GEN64 models with batch sizes ∈ [48,1152] and learning rate ∈ [1 × 10−8,5 × 10−4]. All runs are performed on the full 330M Main Gen dataset discussed in sec. 3.2. By analyzing training results in terms of FID and CLIP score [17] we conclude that learning rate of 1 × 10−4 consistently leads to healthy training dynamics for larger batch sizes across all model sizes.

Model Size, Dataset Size and Compute. Previously, DiT [29] showed that scaling transformer denoiser in the LDM framework [37] improves training in terms of FID. However, the question of the scalability of convolutional models in the cascade diffusion paradigm remains open. Moreover, it is unclear how dataset size is related to the model size and whether there is an optimal combination between the two.

To investigate that, we train four experimental GEN64 models defined above on five dataset scales, resulting in 20 pre-training runs. The experimental datasets are obtained by uniform sub-sampling of the 330M Main Gen dataset with the following sampling fractions: [1,1/2,1/4,1/8,1/16]. All models are trained with the same learning rate of 1 × 10−4 and batch size of 4800, the same as for the 2.3B

###### vs SD v1.4

vs YaART 2.3B

60

60

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

Wins(%)

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

50

50

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

40

40

| | |
|---|---|
| | |
| | |
| | |

50K 250K 450K 650K 850K 1050K 1250K 1450K

50K 250K 450K 650K 850K 1050K 1250K 1450K

Training Steps

Training Steps

- Fig. 5: Dynamics of side-by-side comparisons of the half-size model with Stable Diffusion v1.4 [37] (left) and the fully sized model pre-train (right). Each point shows the mean and standard deviation between three independent human evaluation experiments. Note the rapid quality growth through the first few hundred iterations, after which performance reaches a plateau. Given enough compute, the test model is capable of surpassing the Stable Diffusion quality, while the performance of the fully sized YaART model remains unsurpassed even with more compute.

model pre-training. We leave the remaining super-resolution models from the YaART cascade unchanged.

During training, each model is evaluated every 50K training steps with six SbS comparisons: three against the YaART 2.3B pre-trained model and three against SD v1.4. We perform all evaluations on YaBasket according to the protocol described in 4.1. Using different random seeds for image generation and random pools of human assessors allows us to regularize the evaluation procedure, avoiding potential bias towards one particular type of images.

Our experiments (Figure 4) show that the quality of convolutional models grows along with the increased number of parameters. Also, scaling up the models makes training more efficient in terms of number of training steps and GPU hours required to achieve quality comparable to our two strong baselines: Stable Diffusion v1.4 [37] and YaART 2.3B pre-training. Remarkably, this effect persists when models are trained on datasets of different sizes. The dataset size alone neither dramatically influences training dynamics nor leads to substantial changes in resulting model quality. In these experiments, the models trained on the smallest train set (1/16 of the 330M Main Gen dataset, around 20M imagetext pairs) perform about the same as models trained on the entire 330M Main Gen dataset. This observation suggests that datasets of hundreds of millions and, sometimes, billions of image-text pairs may be excessive, and the dataset size alone is a poor predictor of the resulting model quality.

An Exchange of Model Size for Compute. The common wisdom in deep Learning is that scaling up a model size and computational resources typically leads to better models. But can we trade one for another? To answer this question, we continue pre-training the 1.45B model from the previous section on the 330M Main Gen dataset. Our goal is to train it for longer than the 2.3B baseline, both in terms of the number of training steps and GPU hours, to see whether it allows the test model to surpass the baseline in terms of generation quality.

In Figure 5, showing the training dynamics, we see that the visual quality relatively quickly saturates, reliably reaching the performance of SD v1.4 and

Filtering Model

Sample Type

60

60

SFC Image Aethetics

SFC Top 1% SFC Top 10% 100%

%Wins

%Wins

50

50

40

40

100K 200K 300K

1 10 25 50 75

% of Dataset

Train Steps

- Fig. 6: Training on a small fraction of high-quality data still leads to visually appealing generations. Notably, using image aesthetics as a sole criterion for data selection is constantly inferior compared to the SFC model, taking into account multiple sample quality aspects.

YaART 2.3B pre-train around 350K and 500K iterations, respectively. After that, the quality fluctuates for several hundred thousand iterations and even stagnates in the later stages of training. We conclude that scaling up the model size is an unavoidable prerequisite for further growth of generation quality. We also see that model performance grows non-monotonically, suggesting that generation quality should be periodically evaluated throughout training.

#### 4.3 Data Quality vs Quantity

Most practical settings assume an inherent trade-off between data quality and data quantity; therefore, the vast majority of application-oriented systems [1,7, 11,31] perform some variation of data selection and filtering during the dataset collection. However, the influence of the balance between data quality and quantity on the text-to-image model training dynamics remains unexplored. In this section, we aim to investigate how a gradual decrease in dataset size in favor of selecting higher-quality samples affects the dynamics of model pre-training. We base our analysis on a comparison of the two approaches for sample ranking used for the initial data filtering: the SFC model based on 56 image-text factors and the Image Aesthetics model.

Starting from the 330M Main Gen dataset, we progressively drop a fraction of the worst samples according to one of the aforementioned models. We then train 2.3B models for 300K iterations with default hyperparameters. We use SbS comparisons to compare the resulting models with the baseline trained on the full dataset for the same 300K iterations.

Figure 6 shows that the sub-sampling dataset with the Image Aesthetics classifier only monotonically reduces the quality of the final model, matching with the intuition that larger datasets might lead to better-performing models. It also suggests that removing image-text pairs only by means of image aesthetic attractiveness is suboptimal.

At the same time, sub-sampling data according to the SFC model is more nuanced. Our evaluations suggest that the best results are obtained at 10% dataset size, 33M samples. Further reduction of the dataset size employing more strict filtering leads to a drop in generation quality. There is also a limit on the

###### 2.3B Model

###### Scaled Models

32

Model

%Wins,Fine-tuneSbS

%Wins,Fine-tuneSbS

54

1.4B

929M 523M 233M

28

52

24

50

48

20

54 56 58 60

15 20 25 30 35

% Wins, Pre-train SbS

% Wins, Pre-train SbS

- Fig. 7: Pre-train quality strongly correlates with fine-tune quality both for well-trained models performing similarly with YaART 2.3B pre-train (left) and smaller models from the Scaling Experiments (right).

minimum size of the dataset, as we show that scaling down to 1% significantly hurts the performance.

#### 4.4 Effect of the model performance on fine-tuning quality.

In previous experiments, we aimed to understand the influence of various factors on training dynamics and the quality of the model after the pre-training stage. However, from the practical perspective, one typically aims to maximize the final quality, i.e., after fine-tuning the pre-trained model. However, the relation between quality after these two stages of training is unclear. To investigate that, we fine-tune various quality models from the Scaling Experiments 4.2 following the procedure described in Section 3.3.

Figure 7 summarizes the results of SbS comparisons between pre-trained models and their fine-tuned versions. We observe a strong correlation between performances in these two stages of training despite variations in model size, dataset size, and dataset quality.

### 5 Conclusion

In this paper, we presented YaART — a production-grade diffusion model for text-to-image generation. We show that the choices of model and dataset sizes, as well as the quality of the training images, are important and interconnected degrees of freedom that should be accurately specified for the optimal exploitation of the available GPU power. Also, we describe the procedure of tuning via Reinforcement Learning for YaART, thus confirming that RL is beneficial for production-level diffusion models.

### References

- 1. Arkhipkin, V., Filatov, A., Vasilev, V., Maltseva, A., Azizov, S., Pavlov, I., Agafonova, J., Kuznetsov, A., Dimitrov, D.: Kandinsky 3.0 technical report. arXiv preprint arXiv:2312.03511 (2023) 1, 14
- 2. Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., Zhu, J.: All are worth words: A vit backbone for diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22669–22679 (2023) 4
- 3. Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al.: Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf 2(3), 8 (2023) 1, 4
- 4. Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023) 4, 8
- 5. Borji, A.: Pros and cons of gan evaluation measures. Computer vision and image understanding 179, 41–65 (2019) 5
- 6. Caballero, E., Gupta, K., Rish, I., Krueger, D.: Broken neural scaling laws. arXiv preprint arXiv:2210.14891 (2022) 4
- 7. Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., et al.: Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426 (2023) 1, 4, 14
- 8. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2818–2829 (2023) 6
- 9. Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al.: Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022) 6
- 10. Clark, K., Vicol, P., Swersky, K., Fleet, D.J.: Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400 (2023) 4
- 11. Dai, X., Hou, J., Ma, C.Y., Tsai, S., Wang, J., Wang, R., Zhang, P., Vandenhende, S., Wang, X., Dubey, A., et al.: Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807 (2023) 1, 4, 14
- 12. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009) 4
- 13. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805

(2018) 6

- 14. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances in neural information processing systems 34, 8780–8794 (2021) 6
- 15. Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: Reinforcement learning for fine-tuning textto-image diffusion models. Advances in Neural Information Processing Systems 36

(2024) 4

- 16. He, S., Zhang, Y., Xie, R., Jiang, D., Ming, A.: Rethinking image aesthetics assessment: Models, datasets and benchmarks. In: Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22. pp. 942–948

(2022) 7

- 17. Hessel, J., Holtzman, A., Forbes, M., Bras, R.L., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021) 12

- 18. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017) 5, 11
- 19. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 6840–6851. Curran Associates, Inc. (2020), https://proceedings.neurips.cc/paper_files/paper/2020/file/ 4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf 9
- 20. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D.d.L., Hendricks, L.A., Welbl, J., Clark, A., et al.: Training computeoptimal large language models. arXiv preprint arXiv:2203.15556 (2022) 12
- 21. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021) 9
- 22. Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., Schmidt, L.: Openclip (Jul 2021). https://doi.org/10.5281/zenodo.5143773, https://doi. org/10.5281/zenodo.5143773, if you use this software, please cite it as below. 9
- 23. Kingma, D., Salimans, T., Poole, B., Ho, J.: Variational diffusion models. Advances in neural information processing systems 34, 21696–21707 (2021) 6
- 24. Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems 36 (2024) 11
- 25. Koch, G., Zemel, R., Salakhutdinov, R., et al.: Siamese neural networks for one-shot image recognition. In: ICML deep learning workshop. vol. 2. Lille (2015) 9
- 26. Lee, K., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Gu, S.S.: Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192 (2023) 4
- 27. Lee, S.H., Li, Y., Ke, J., Yoo, I., Zhang, H., Yu, J., Wang, Q., Deng, F., Entis, G., He, J., Li, G., Kim, S., Essa, I., Yang, F.: Parrot: Pareto-optimal multi-reward reinforcement learning framework for text-to-image generation (2024) 4
- 28. Murray, N., Marchesotti, L., Perronnin, F.: Ava: A large-scale database for aesthetic visual analysis. In: 2012 IEEE conference on computer vision and pattern recognition. pp. 2408–2415. IEEE (2012) 7
- 29. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4195–4205

(2023) 4, 12

- 30. Pernias, P., Rampas, D., Richter, M.L., Pal, C., Aubreville, M.: Würstchen: An efficient architecture for large-scale text-to-image diffusion models. In: The Twelfth International Conference on Learning Representations (2023) 1
- 31. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023) 1, 4, 11, 14
- 32. Pressman, J.D., Crowson, K., Contributors, S.C.: Simulacra aesthetic captions. Tech. Rep. Version 1.0, Stability AI (2022), url https://github.com/JDP/simulacra-aesthetic-captions 7
- 33. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A.: Catboost: unbiased boosting with categorical features. Advances in neural information processing systems 31 (2018) 7

- 34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021) 6
- 35. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022) 3, 6
- 36. Rasley, J., Rajbhandari, S., Ruwase, O., He, Y.: Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. pp. 3505–3506 (2020) 8
- 37. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 1, 3, 4, 6, 12, 13
- 38. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. pp. 234–241. Springer (2015) 4
- 39. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022) 3, 4, 6, 8, 10
- 40. Sajjadi, M.S., Bachem, O., Lucic, M., Bousquet, O., Gelly, S.: Assessing generative models via precision and recall. Advances in neural information processing systems 31 (2018) 5
- 41. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017) 8
- 42. Thrush, T., Jiang, R., Bartolo, M., Singh, A., Williams, A., Kiela, D., Ross, C.: Winoground: Probing vision and language models for visio-linguistic compositionality. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5238–5248 (2022) 10
- 43. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017) 4
- 44. Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Diffusion model alignment using direct preference optimization (2023) 4
- 45. Yang, K., Tao, J., Lyu, J., Ge, C., Chen, J., Li, Q., Shen, W., Zhu, X., Li, X.: Using human feedback to fine-tune diffusion models without any reward model. arXiv preprint arXiv:2311.13231 (2023) 4
- 46. Yu, F., Zhang, Y., Song, S., Seff, A., Xiao, J.: Lsun: Construction of a largescale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365 (2015) 4
- 47. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., et al.: Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 2(3), 5 (2022) 10
- 48. Zheng, H., Nie, W., Vahdat, A., Anandkumar, A.: Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305 (2023) 4

### Supplementary Material

Here we show additional results on the correlation between pre-train and finetune performance for quality vs quantity experiments (Figure 8). We make additional comparisons with state-of-the-art on DrawBench (Table 3) and provide a detailed description of hyperparameters (Table 4). Finally, we discuss YaART’s limitations and provide additional examples of generated images, showcasing its high visual consistency, aesthetic attractiveness, and strict prompt-following abilities.

Effect of the model performance on fine-tuning for Quality vs Quantity trade-off

We previously discussed that pre-train quality strongly correlates with fine-tune quality in terms of our three-aspect side-by-side (SbS) comparisons regardless of the model size given a full 330M Main Gen dataset for training. This experiment aims to understand whether the dataset size and quality introduce any difference. For that, we fine-tune checkpoints from the Quality vs Quality experiment, i.e., 2.3B GEN64 models trained on samples of the best image-text pairs according to Image Aesthetics and Sample Fidelity Classifier (SFC) models.

Figure 8 shows that even models pre-trained on small sample fractions (1%, 3%) are easily exposed to fine-tuning. However, even though the best pre-trained model was obtained by training on the Top 10% of data according to the SFT model, the model trained on the entire dataset still achieves the best fine-tuning quality. This suggests that having a large pre-train set of moderate-quality data is a strong prerequisite for achieving high-end quality.

### Limitations

##### Contrary to the mainstream approach to training text-to-image diffusion models as LDMs, we have chosen the cascaded diffusion variant. Although this ap-

###### 2.3B Model, Top by Aesthetics

###### 2.3B Model, Top by SFT

| |Fraction<br><br>1% 3%<br><br>10% 25%<br><br>50% 75%<br><br>100%| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

74

70

%Wins,Fine-tuneSbS

%Wins,Fine-tuneSbS

66

65

58

60

50

Fraction

1% 3%

10% 25%

50% 75%

100%

55

42

15 25 35 45 55

40 45 50 55 60

% Wins, Pre-train SbS

% Wins, Pre-train SbS

###### Fig. 8: Pre-train quality strongly correlates with fine-tune quality when trained on a fraction of higher-quality data. This effect is consistent across two data quality estimation model types: Image Aesthetics (left) and SFT (right).

Quality Aspects YaART 2.3B RL Beauty Defects Alignment Overall

vs MidJourney v5 0.58 ± 0.01 0.54 ± 0.01 0.52 ± 0.02 0.55 ± 0.03 vs SDXL 0.78 ± 0.04 0.76 ± 0.03 0.54 ± 0.02 0.82 ± 0.02 vs Kandinsky v3 0.61 ± 0.03 0.70 ± 0.02 0.48 ± 0.02 0.73 ± 0.02 vs OpenJourney 0.86 ± 0.06 0.80 ± 0.01 0.82 ± 0.05 0.94 ± 0.01

Table 3: We compare YaART with state-of-the-art based on three main evaluation criteria and overall human preferences on DrawBench. Bold denotes a statistically significant overall model quality difference.

64 64 → 256 256 → 1024

Noise Schedule linear linear linear Sampling Steps 32 32 32 Crop Fraction 1 1 1⁄4 Model Size 2.3B 700M 700M Channels 448 320 128 Depth 4 4 5 Channels Multiplier [1, 2, 3, 4] [1, 2, 3, 4] [1, 2, 2, 4, 8] Heads Channels 64 64 Attention Resolution [2, 4, 8] + middle middle Text Encoder Context 128 128 uncond Text Encoder Width 1536 1536 uncond Dropout 0 0 0 Weight Decay 0 0 0 Batch Size 4800 960 512 Iterations 1.1 × 106 1.5 × 106 1.5 × 106 Learning rate 1 × 10−4 5 × 10−5 5 × 10−5

- Adam β1 0.9 0.9 0.9
- Adam β2 0.99 0.99 0.99 EMA Decay 0.9999 0.9999 0.9999

Table 4: Hyperparameters for the models in the YaART cascade.

proach demands more computational resources at runtime, its interactive nature presents significant benefits: we can provide a 256x256 image that users can further filter and refine. Moreover, we apply the super-resolution stage only to a selected fraction of these images. Despite all the recent progress in this area, modern diffusion models still require substantial human supervision. This typically involves iterative prompt modification, sampling parameter tuning (or even adjusting the random seed), and a post-filtering process for the generated images.

We have also applied automatic filters to eliminate images containing text from our dataset. This decision stems from our belief that the quality of text generation currently falls short of practical standards. Interestingly, despite these limitations, our model has successfully learned to generate reasonable characters and words from the few text-containing images in our training dataset.

[Figure 10]

[Figure 11]

[Figure 12]

Fat fat ﬂuﬀy caterpillar cat in the shape of a caterpillar crawling on a leaf on a tree, art bionic, surrealism, hyperdetalization, hyperrealism 8k

aerial view of a green tree in a green ﬁeld next to the brown freshly turned soil

a dog towering over a spider

[Figure 13]

[Figure 14]

[Figure 15]

cutlet with mushrooms and celery puree. in the menu of the gourmet restaurant. dimmed lights. dark background. dark blue plate with a small rim and pale yellow border.

A crazy cock with SpongeBob eyes ﬂies on farting gas into space leaving a plume of smoke behind him

A lion in a uniform and a cap is talking on a walkie-talkie and driving fast in a police car with ﬂashing lights on

[Figure 16]

[Figure 17]

[Figure 18]

Anime girl loli, height 140 bear ears,cute,bear in her hands,pink eyes,pink hair short,loli, aesthetically pleasing, beautiful, close-up, high resolution, high detail,against a background of many sakura trees

Top-down view. A small realistic fox cub in a hat in a cup of marshmallows. lilac,candles,4k, beautiful, realistic, professional photo, high detail, high resolution

Two seahorses playfully swirling around a sunken treasure chest.

[Figure 19]

[Figure 20]

[Figure 21]

the snail has a transparent round shell-aquarium, neon lighting, jungle waterfall background, hyperdetalization, ultra-realistic, professional photo, 55 mm lens, iso 100, f/6, 180s, 32k, macro photography

watercolor, contour drawing, short black hair, in proﬁle, ﬁtness model, young, black split bikini swimsuit, push-up bra, thong panties, hips, legs, half-turn posing, beach

apple tree ﬂowers, aesthetically pleasing, beautiful, realistic,high detail professional photo, 1024k, high resolution.

[Figure 22]

[Figure 23]

[Figure 24]

Three seashells lined up along the shore of a calm beach.

portrait of a well-dressed raccoon, oil painting in the style of Rembrandt

A crowd of colorful paper boats racing each other downstream.

[Figure 25]

[Figure 26]

[Figure 27]

Early morning. In this picture, full of movement and drama, a drakkar is ﬁghting a ferocious storm. Giant waves, swirling clouds and sparkling lightning create a feeling of real sea adventure and violent elements

shy image of a beautiful girl in a knitted hat with a bouquet of ﬂowers, sweet smile, close-up, comic sketch, high detail

glass dandelions

[Figure 28]

[Figure 29]

[Figure 30]

A photographer capturing the stunning architecture of St. Basil’s Cathedral in Moscow.

The sharp, tangy taste of the pickles added a zingy punch to the sandwich

Sketch, delicate painting, oil painting, beautiful scenery, large ﬂowers lush white and pink lace peony branches, pixel graphics, lots of detail, sensuality, realism, high quality, decoration, hyper-ditalization, professionally, ﬁligree, hyper-realism, transparency, delicate pastel tones, backlight, contrast, Fantastic, wonderful, unreal, translucent, bright, clear lines, Light green and Light green

[Figure 31]

[Figure 32]

[Figure 33]

a pink cat eats a chocolate bar

knitted bag made of knitted threads on a white background

Illustration in a medieval bestiary, ink drawing, on old parchment, engineering diagram, text explanations in Gothic font, close-up, tyrannosaurus rex

[Figure 34]

[Figure 35]

[Figure 36]

ﬂuﬀy anthropomorphic white ball with big blue eyes, plush toy, high quality, photorealism, detail, 8k

a realistic red-haired little squirrel on a ﬁr branch in a knitted hat with a ﬂuﬀy fur round pompom, a fashionable scarf around his neck, high detail

A vintage-style photograph of a classic car on Route 66, with a desert backdrop, taken with the eﬀect of ﬁlm photography and graininess for a nostalgic feel.

[Figure 37]

[Figure 38]

[Figure 39]

an overhead view of a pickup truck with boxes in its ﬂatbed

sketch,A black crow in a ﬂuﬀy squirrel hat with earﬂaps,full face

A pair of curtains colored like a lavender ﬁeld.

[Figure 40]

[Figure 41]

[Figure 42]

terrible clown, gloomy, rain , high resolution and quality, play of colors, high detail

a smiling banana wearing a bandana

Japanese Demon

[Figure 43]

[Figure 44]

[Figure 45]

photorealism, solid iron armor, a completely thin body and an iron all-metal molten head in the form of a face, which has thin plate arms and very long thin ﬁve scalpel ﬁngers

epic battle of two Easter eggs in the style of blockbusters

A dog in the colors of the French ﬂag

