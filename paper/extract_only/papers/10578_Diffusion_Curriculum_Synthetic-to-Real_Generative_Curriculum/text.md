## Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion

# arXiv:2410.13674v4[cs.CV]26Sep2025

Yijun Liang*, Shweta Bhardwaj*, Tianyi Zhou University of Maryland, College Park

{yliang17,shweta12}@umd.edu, tianyi.david.zhou@gmail.com Project: https://github.com/tianyi-lab/DisCL

#### Abstract

Low-quality or scarce data has posed significant challenges for training deep neural networks in practice. While classical data augmentation cannot produce very different new data, diffusion models open up a new door to build selfevolving AI by generating high-quality and diverse synthetic data through text-guided prompts. However, text-only guidance cannot control synthetic images’ proximity to the original images, resulting in out-of-distribution data detrimental to model performance. To overcome the limitation, we study image guidance to achieve a spectrum of interpolations between synthetic and real images. With stronger image guidance, the generated images are similar to the training data, but are hard to learn. With weaker image guidance, the synthetic images will be easier to learn but suffer from a larger distribution gap to the original data. The generated full spectrum of data enables us to build a novel “Diffusion CurricuLum (DisCL)”. DisCL adjusts the image guidance level of image synthesis for each training stage: It identifies and focuses on hard samples for the model and assesses the most effective guidance level of synthetic images to improve hard data learning. We apply DisCL to two challenging tasks: long-tail (LT) classification and learning from lowquality data. It focuses on lower-guidance images of high quality to learn prototypical features as a warm-up for learning higher-guidance images that might be weak on diversity or quality. DisCL achieves a gain of 2.7% and 2.1% in OOD and ID macro-accuracy when applied to iWildCam dataset. On ImageNet-LT, DisCL improves the base model’s tail-class accuracy from 4.4% to 23.64% and leads to a 4.02% improvement in all-class accuracy.

#### 1. Introduction

While existing machine learning approaches can train representation or discriminative models with promising general-

*These authors contributed equally to this work.

ization performance, their success highly relies on the quality and quantity of the training data. However, in enormous practical scenarios, the data are collected from real environments so neither the quality nor the quantity can always be guaranteed. For example, it is difficult to control the light conditions, weather, motion blur, or the position of objects in the scenes captured by trail/animal cameras, traffic cameras, motion cameras, or robot cameras. Likewise, it is also difficult to keep different classes in the collected data balanced so the model may perform much poorer on tail classes with scarce data. On the other hand, the low-quality/quantity of data also makes the model more prone to the gap between the test and training distributions, thereby posing an out-ofdistribution challenge. In many cases, such “hard” training data hinders effective learning, introduces biases or outliers, and may even impact the learning of other data.

Data augmentation and synthesis have been studied to address the challenges of hard real data. By applying pre-defined transformations [1] to data in scarce classes or modifying their backgrounds [3, 11], data augmentation helps learn representations robust to these task-irrelevant variations. While the augmented data may lack sufficient diversity or non-trivial difference to the original data, the recent text-to-image generative models such as GAN or Stable Diffusion enable more sophisticated data synthesis [9] of diverse higher-quality samples, while the text prompts retain the task-related features. For instance, recent works [15, 23, 39] have focused on training specialized diffusion models specifically for sampling from underrepresented or hard classes to increase diversity. However, these existing methods are still challenged when scaling to real-world data, such as in-the-wild or long-tail learning scenarios, where the data distribution is highly imbalanced, diverse, and unpredictable. Although text-to-image synthesis improves the data quality and quantity, the synthetic data are solely controlled by text prompts but lack sufficient visual similarity to the original image, which leads to a distribution gap to the original data and hurts the generalization performance. To maximize the merits of synthetic data for learning hard

data in real applications and address the syn-to-real gap, we harness the image guidance in diffusion models to generate a full spectrum of interpolations between synthetic data (i.e., generated only from text prompts) and real data (i.e., original images that may suffer from low-quality or sparse quantity). The synthetic data at each level of interpolation are generated under the weighted guidance of both the text prompt (e.g., the class name) and the real images. While stronger image guidance preserves visual similarities to the original image, for low-quality or low-quantity data, weaker image guidance could lead to high-quality, diverse, and potentially easier (e.g., with prototypical features) data. Hence, the syn-to-real interpolations create a novel space of synthetic data to design a generative curriculum that can adjust the quality, diversity, and/or difficulty of data for different training stages, by selecting the guidance level according to a pre-defined schedule or training dynamics.

In this paper, we develop novel generative curriculum learning approaches for two types of challenging applications with “hard” real images: long-tail classification and learning from low-quality images. In long-tail classification, learning the tail classes’ features is challenging due to their data deficiency and the lack of diversity compared to “head classes”. To address this challenge, we propose a curriculum that first learns synthetic images with lower image guidance for tail classes since they enhance the diversity and quantity of the original data. The curriculum then gradually increases the guidance level and learns synthetic images closer to the original images, thereby progressively bridging the syn-to-real gap. In learning from low-quality data, the primary challenge is to capture the critical features of the target classes, which is hard due to intricate background, occlusion, or motion blur in the original images. In contrast, images generated with lower image guidance usually contain prototypical features easier to learn. That being said, an overly high or low guidance level may enlarge the domain gap between the training data and the target (indistribution or out-of-distribution) data. To avoid negative transfer caused by the domain gap and to maximize the merits of synthetic data, we develop an adaptive curriculum that selects the guidance level of synthetic data leading to the greatest progress of each training stage.

We examine two DisCL curricula on benchmark datasets, WILD-iWildCam [4] and ImageNet-LT [25], for learning from low-quality images, and long-tail classification respectively. Our DisCL curricula improve OOD and ID accuracy by 2.7% and 2.1% respectively on iWildCam. On ImageNet-LT, DisCL improves the minority classes’ accuracy by 19.24% and leads to a 4.02% improvement in the overall accuracy. Our main contributions can be summarized as follows:

• Harness image guidance in diffusion models to systematically create a spectrum of synthetic-to-real data for each

sample, enabling the design of effective training curricula to address hard data learning.

- • Propose the “Diffusion CurricuLum (DisCL)” paradigm, which selects synthetic data of different guidance levels to meet the needs of each training stage. We propose two novel DisCL curricula to address two key applications: long-tail classification and learning from low-quality data.
- • Examine the two DisCL curricula on challenging datasets and demonstrate their effectiveness in significantly boosting the performance of existing image classifiers, particularly on hard data.

#### 2. Related Work

Diffusion models for Synthetic Data Recently, a diverse array of generative diffusion models have been proposed, including GLIDE [14], Imagen [31], Stable Diffusion [30], Dall-E [28], and Muse [7]. These models can generate realistic, high-resolution images when conditioned on text prompts, and therefore, are used off-the-shelf to augment the datasets for enhancing data diversity. For instance, He et al. [16] demonstrates that synthetic data created with GLIDE can significantly improve both zero-shot and few-shot performance on image classification. Recent works like Bansal and Grover [2], Sariyildiz et al. [32], Dunlap et al. [9] and Hemmat et al. [17] have shown that real data combined with synthetic data generated by Stable Diffusion models, boosts the robustness of standard ImageNet classifiers. Other works such as Shao et al. [34] train a diffusion model on original data to generate large-scale synthetic samples across the distribution, improving alignment with real data but limiting diversity. Hemmat et al. [17] further improves diversity by employing an off-the-shelf diffusion model for single-stage synthetic generation at a large scale. In contrast, in this work, we focus on learning hard data, and adopt a progressive approach in generating only-useful synthetic data at a significantly smaller scale. We also leverage an off-the-shelf diffusion model, but unlike prior works, we harness different image guidance levels to generate training samples at each stage of training. This method allows for smooth transition across spectrum of interpolations from syn-to-real data, and adapts the model to diverse, in-the-wild scenarios, while maintaining data diversity and alignment.

Curriculum Learning (CL) Curriculum Learning (CL) was first proposed by Bengio et al. [5], introducing a training method analogous to the step-by-step progressive learning of humans. Subsequent works have further explored this idea; for example, Jiang et al. [22], Zhou et al. [45] adjusted the progression pace based on the difficulty of samples, and Jiang et al. [21], Zhou and Bilmes [44] further take the data diversity into account. Previous works [13, 43, 47] have tried CL on more challenging domains like noisy web images and visual QA; this highlights its potential in tackling challenging scenarios. Few works have explored the combination of

data augmentation and curriculum learning [20], but mainly for the text data [26, 42]. Some initial efforts have been made by Ahn et al. [1] to combine CL with engineered image augmentations for tail classes in long-tail learning. In contrast, our work aims to design a generative curriculum on a syn-to-real spectrum of data produced by diffusion models, with broader applications in learning from long-tail or low-quality data.

#### 3. Methodology

We propose diffusion curriculum (DisCL) to “close the distribution gap between original data and the target data distribution”. DisCL comprises two phases: (Phase 1) Syntheticto-Real Data Generation that generates a syn-to-real spectrum of interpolated data for hard samples, and (Phase 2) Generative Curriculum learning based on the synthetic data from Phase 1. The two phases are illustrated in Fig. 1.

##### 3.1. Synthetic-to-Real Data Generation

Hard Sample Identification We first identify the difficult samples where the model struggles to extract helpful features for target classification. The difficulty estimation can be taskspecific. For instance, in long-tail classification with scarce data, the difficulty of each sample depends on whether it belongs to tail classes. For tasks with low-quality data, we can utilize the loss or confidence on the ground-truth class to measure the difficulty. These samples are marked as “hard samples” within the training set (see Fig. 1), to highlight their role in the model’s learning process.

Synthetic Data Generation with Image Guidance Classifier-free guidance in diffusion models was introduced by Ho and Salimans [19], to integrate conditional information into the image denoising process of diffusion without requiring a classifier. It has been adopted by several Textto-Image generation models such as Stable Diffusion (SD) [30]. Given the latent representation of original image as zreal, the denoising (backward diffusion) process can start from any step t with initial zt defined as:

zt = α˜tzreal + 1 − α˜tϵ, ϵ ∼ N(0,I). (1) The remaining denoising steps iteratively apply the following process of noise estimation ϵˆt at each step t to get a less noisy generation of zt−1, until t = 0, resulting in a synthetic image z0.

ϵˆt = (1 + w)ϵθ(zt,t|c) − wϵθ(zt,t), zt−1 =

1 √αt

βt √1 − α˜t

ϵˆt + βtϵ′, t ← t − 1

zt −

(2)

In Eq. 1-2, α˜t,αt, and βt together define the variance schedule of the diffusion process. ϵ,ϵ′ ∼ N(0,I) are two independently-sampled Gaussian noises, ϵθ(·,·) refers to the noise estimation model, and w ∈ R controls the strength of the textual prompt c as a condition to ϵθ(·,·).

Since α˜t monotonically decreases with t, the choice of the initial t in Eq. 1 controls the impact of the original zreal in the denoising process, and more visual information of zreal tends to be preserved in z0 if initializing from a small t. To achieve a full spectrum of interpolations between the real image zreal and synthetic images depicted by c, following prior work in Meng et al. [27], we modify the initial step t in Eq. 1 to t(λ) ≜ ⌊(1 − λ)T⌋ where λ ∈ [0,1) defines the image-guidance level, i.e.,

zt(λ) = α˜t(λ)zreal + 1 − α˜t(λ)ϵ, t(λ) ≜ ⌊(1 − λ)T⌋.

(3) Hence, a larger guidance level λ leads to higher fidelity of generated image z0 to original zreal, while a smaller λ results in a more prototypical image z0′ depicted by textual prompt c. λ = 0 results in a generated image based on text only. 1

Synthetic-to-Real Spectrum of Generated Images We use state-of-the-art Stable Diffusion Model 2 to generate synthetic images for the hard samples identified in Phase 1 of Fig. 1. By adjusting the image guidance scale λ ∈ [0,1) in Eq. 3, the denoising process in Eq. 2 can produce a full spectrum of smooth transitions between text-only guided synthetic images and real images. We next study the effect of varying the image guidance scales λ on the generated synthetic images. As shown in Fig. 2, changing λ leads to varying difficulty and diversity of synthetic images. With a smaller λ, diffusion model mainly relies on the text information provided in the prompt c, generating synthetic images that differ markedly from the original and focus more on the distinct prototypical features of the class in c. As λ increases, the synthetic images increasingly inclines towards the original image, exhibiting less diversity (across random seeds) and more resemblance to the original ones. When the original images are of low-quality, a large λ makes it challenging for the classifier to learn discriminating features from synthetic images. Therefore, the broad spectrum of synthetic data offers diverse properties, e.g., diversity, hardness, proximity to the real ones, providing a novel design space for curriculum learning.

Filter out Synthetic Data with Low-Fidelity As shown in Fig. 2, some synthetic images may suffer from poor quality and low fidelity to the text prompt c, e.g. the class object is missing or obscured, which can hinder the downstream tasks. To mitigate this, we filter-out such images by applying CLIP-based filtering used in [9, 18, 33], which measures CLIP cosine similarity between synthetic images and text prompt. We discard images that fall below a predefined CLIPScore threshold before the training begins.

- 1λ = 0 corresponds to images generated using only text prompt guidance, while λ = 1 corresponds to replication of the original image without any text guidance.
- 2We use Stable Diffusion XL model for generation

Full Spectrum of Syn-to-Real Data

|Phase 1: Syn-to-Real Data Generation|
|---|

Hard Samples

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Diffusion Model

[Figure 5]

| | |
|---|---|
| | |

[Figure 6]

###### Training Data

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Pretrained Model

Non-Hard Samples 𝛌=1 Image Guidance 𝛌 𝛌=0

Text Prompt with Class Info

Training Process

Curriculum Strategies

Selected Guidance

Guidance Selection

Non-Adaptive Curriculum

Data with guidance 𝛌i

Data with guidance 𝛌j

Data with guidance 𝛌k

Adaptive Curriculum

Non-Hard Samples

Non-Hard Samples

Non-Hard Samples

[Figure 11]

Trained Model

|Phase 2: Generative Curriculum Learning|
|---|

Finetuned model with selected data

𝛌1→ 𝛌2 →.. →𝛌k

Training Stage

e

e+1

…

- Figure 1. Overview of Diffusion Curriculum (DisCL). DisCL consists of two phases: (Phase 1) Syn-to-Real Data Generation and (Phase

2) Generative Curriculum learning. In Phase 1, we use a pretrained model to identify the “hard” samples in the original images and use them as guidance to generate a full spectrum of synthetic to real images by varying image guidance strength λ. In Phase 2, a curriculum strategy (Non-Adaptive or Adaptive) selects training data from the full spectrum, by determining image guidance level λi for each training stage e. The Adaptive strategy chooses λi to maximize expected progress, while the Non-Adaptive strategy follows a predefined schedule. Synthetic data of the selected guidance level is then combined with non-hard real samples to train the task model.

[Figure 12]

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

RandomSeeds

A photo of <Cheetah> in the wild

[Figure 13]

[Figure 14]

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

RandomSeeds&Prompts

The Acadian Flycatcher flits gracefully through the trees, its distinctive call echoing through the forest.

eep in the forest, an Acadian Flycatcher engages in a courtship dance, its movements fluid and graceful.

[Figure 15]

- Figure 2. Synthetic images generated with different image guidance levels λ and random seeds. × marks images with low-fidelity to the text prompt, which are filtered out by CLIPScore (ref. the end of §3.1). Zoom-in for better view.
- 3.2. Generative Curriculum with Synthetic Data With the full spectrum of syn-to-real generated data, we achieve a smooth transition from images with prototypical features and high diversity to task-specific features that

closely resemble real images. This enables us to design a curriculum that selects data based on their diversity and feature types for different training stages. With a curriculum of rich synthetic data, we can improve the model’s performance

in challenging and diverse cases that would be difficult to address using only real data. Additionally, this approach allows us to control the distribution gap to the original data. We apply our generative curriculum to two challenging applications: long tail learning and learning from low-quality.

For long tail classification, the scarcity of data in minority classes makes it difficult for models to extract useful features for these classes, leading to poor generalization on balanced test set. For tail classes, we first generate a full spectrum of synthetic data using techniques in §3.1, following the standard split of tail classes in the studied dataset. A diverse set of textual prompts is used to achieve this goal3. The generated spectrum offers varying degrees of data diversity, which if used at once, can introduce a syn-to-real distribution gap. To mitigate this gap, we first expose the model to diverse synthetic images of tail classes, and then progressively shift to a task-specific distribution that resembles the original images. This yields a non-adaptive “Diverse-to-Specific” curriculum that starts with synthetic data with a lower guidance scale (λ → 0) and gradually moves toward data of a higher guidance scale (λ → 1). The algorithm for our non-adaptive curriculum strategy is provided here in Algorithm 1.

Learning from low-quality images can be challenging due to inevitable quality issues, such as obscurity in images from traffic, motion, or wildlife observation cameras. We investigate wildlife observation as an example application of DisCL to enable effective learning under such scenarios. For low-quality camera trap images, we aim to generate simpler, more prototypical images of animals to warm up training and generalize to harder cases. We identify hard samples using a pretrained classifier—lower ground-truth class probabilities indicate higher difficulty. Varying the image guidance scale, we synthesize a full spectrum of data for these samples using class information in the text prompts4, which steers the diffusion model toward relevant animal and habitat features. Unlike long-tail settings, hard samples in low-quality domains often lack both prototypical and generalizable features, as shown in prior work on camera trap images [37]. Consequently, synthetic data generated with low guidance often appears prototypical but out-of-distribution. A nonadaptive curriculum that introduces such synthetic data early risks distribution shift or overemphasis on outlier features. DoCL [46] addressed this by selecting real data adaptively to optimize learning progress. Inspired by this, we propose an adaptive curriculum detailed in Algorithm 2, which selects the image guidance level λ at each epoch based on progress – defined by improvement in ground-truth class confidence on validation subsets corresponding to each λ. The guidance level with the highest progress is chosen for the next epoch. This ensures the model learns from the most informative

- 3Text prompts are provided in Appendix A.2.2
- 4Text prompts are provided in Appendix A.2.3.

data at each stage, enabling a smooth transition from simple to complex patterns and maximizing improvement on the real data distribution. Details are in Algorithm 2.

#### 4. Experiments

##### 4.1. Long-Tail Classification

Setup To validate the efficacy of DisCL method on long-tail classification, we conduct main experiments with ImageNet-LT (IN-LT) dataset [25]. This dataset includes 1000 classes, with class cardinality ranging from 5 to 1,280. To assess the robustness of DisCL more comprehensively, we conduct experiments on two additional datasets: a synthetically imbalanced dataset, CIFAR100-LT [6], and a realworld benchmark, iNaturalist2018 [40]. CIFAR100-LT is provided with imbalanced classes by synthetically sampling the training data with multiple imbalance ratios {100,50}. iNaturalist2018 dataset represents a naturally occurring longtailed distribution with class cardinality ranging from 2 to 1000. We evaluate overall accuracy and the accuracy across three class categories: many (most frequent), medium, and few (least frequent, tail) classes on the standard balanced test sets of three datasets. For synthetic data generation, we use DDIM [36] as our noise scheduler. For training, we follow Ahn et al. [1] and Han et al. [15], using ResNet-10 as visual backbone. We average results over 3 runs and report training details and hyper-parameters in Appendix A.4.1 and A.5.

Baselines We compare the effect of DisCL with comparable baseline of CUDA [1] and LDMLR [15], mainly using Cross-Entropy (CE) loss function. To further illustrate the robustness of DisCL, we try Balanced Softmax (BS) loss [29], known for its competitive performance on long-tail learning.

- • CUDA: Engineered data augmentation + curriculum learning on IN-LT.
- • LDMLR: A three-stage training using diffusion model to improve LT.
- • BS loss: Balanced softmax to address class-distribution shift between training and test sets.

We also conduct ablation study to analyze the effect of DisCL under different hyperparameter settings. We note that, real data for hard samples (λ ∼1) is included by default; however, this doesn’t apply to the Fixed Guidance and Textonly Guidance ablation:

- • Text-only Guidance: Using data at image guidance scale λ = 0 without curriculum strategy.
- • Fixed Guidance 5: uses data generated from a single guid-

ance scale λi ∈ [0,1). We report results for the guidance with the highest few-class accuracy.

5Text-only Guidance (λ=0) reaches the best performance amongst all guidance scales. Hence, the result of Fixed Guidance here are same as Text-only Guidance, reported in Table 1. We also report the performance of all other scales in Fixed Guidance experiment in the Fig. 14.

ImageNet-LT

Method Curriculum Many Medium Few Overall

CE N/A 57.70 26.60 4.40 35.80 CE + LDMLR N/A 57.20 29.20 7.30 37.20 CE + CUDA N/A 57.49 28.16 6.58 36.30 BS† N/A 51.14 37.02 19.29 39.80 BS + CUDA† N/A 51.16 37.35 19.28 40.03

Baselines

CE + Text-only Guidance N/A 56.63 30.69 17.90 39.10 CE + All-Level Guidance N/A 56.77 30.88 19.17 39.40 CE + DisCL Adaptive 56.21 30.43 16.78 38.65 CE + DisCL Specific to Diverse 56.71 30.67 18.36 39.18 CE + DisCL [Lower CLIPScore Threshold] Diverse to Specific 57.66 30.61 23.69 39.67 CE + DisCL [Higher CLIPScore Threshold] Diverse to Specific 56.92 30.64 22.88 39.68

Ablations

CE + DisCL Diverse to Specific 56.78 30.73 23.64 39.82 BS + DisCL Diverse to Specific 52.68 37.68 21.36 41.33

Ours

- Table 1. Accuracy (%) of long-tail classification on ImageNet-LT with base model ResNet-10. The best accuracies among baseline and DisCL are highlighted in bold. † marks our reproduced results using the original paper-provided code. BS refers to Balanced Softmax Loss[29]. Baselines (LDMLR, CUDA) are defined in §4.1.

• DisCL: employs multiple levels of guidance scales alongside a range of curriculum strategies. These strategies and the guidance intervals used for training, are defined below:

- – Diverse to Specific: Non-adaptive strategy with guidance changing from smallest (diverse augmentation) to largest (task-specific augmentation).
- – Specific to Diverse: Non-adaptive strategy with guidance changing from largest to smallest.
- – Adaptive: Curriculum strategy6 to adaptively select guidance during training.

Results We present the results of our method alongside the baselines for the ImageNet-LT dataset in Table 1. With CE loss, DisCL significantly improves accuracy in all 4 class-categories. Notably, “Few” class accuracy increases by 17.06%, from 6.58% to 23.64%, demonstrating DisCL’s effectiveness in addressing the data scarcity challenge, especially for tail classes. DisCL also works effectively with BS loss, resulting in additional gains of 1.52% in Many, 2.08% in Few, and 1.3% overall accuracy, further underscoring our method’s impact even with a class-balancing loss function. Results on CIFAR100-LT (Table 2) and iNaturalist2018 (Table 3) confirm the robustness of DisCL across several diverse datasets, achieving better accuracy in tail classes along with improved overall generalization.

CIFAR-100-LT Imbalance Ratio=100 Imbalance Ratio=50

Method Curriculum Many Medium Few Overall Many Medium Few Overall

CE N/A 52.86 25.34 5.49 29.02 49.60 25.41 5.33 31.72 CE + CUDA N/A 54.55 26.07 5.43 29.85 52.29 26.17 5.53 33.13 CE + DisCL Diverse to Specific 53.14 25.52 10.65 30.91 53.4 31.69 21.47 36.22

BS N/A 47.87 30.07 14.41 31.61 46.01 30.76 18.55 34.82 BS + CUDA N/A 48.01 32.79 15.55 33.02 46.08 32.51 22.11 36.21 BS + DisCL Diverse to Specific 49.02 29.02 19.07 33.08 49.51 32.60 25.58 36.77

- Table 2. Accuracy (%) of long-tail classification on CIFAR-100-LT with base model ResNet-10. The best accuracy for overall and classes of {many, medium, few} samples is highlighted in bold.

6Curriculum strategy proposed in §3.2

iNaturalist2018 Method Curriculum Many Medium Few Overall

CE N/A 55.02 43.40 37.33 42.20 CE + CUDA N/A 55.94 44.21 39.13 43.18 CE + DisCL Diverse to Specific 54.71 44.37 48.92 47.25

BS N/A 46.12 49.31 50.27 49.46 BS + CUDA N/A 48.77 49.94 50.87 50.23 BS + DisCL Diverse to Specific 45.44 48.18 53.63 50.30

Table 3. Accuracy (%) of long-tail classification on iNaturalist2018 with base model ResNet-10. The best accuracy for overall and classes of {many, medium, few} samples is highlighted in bold.

##### 4.2. Learning from Low-quality Data

Setup We also conduct DisCL experiments with iWildCam dataset [4] to evaluate its efficacy in classifying low-quality data. The task is to classify 182 different animal species from images captured by camera traps. We evaluate model performance on standard out-of-domain (OOD) and in-domain (ID) test sets in terms of macro F1 score. We choose the CLIP ViT model as our base model and finetune CLIP ViT-B/16 and CLIP ViT-L/14 7 models with DisCL. The reported accuracy is averaged over 3 random seeds. More training details and hyperparameters are provided in Appendix A.4.2 and Appendix A.5.

Baselines We compare the effect of our method with three benchmark algorithms, LP-FT [24], FLYP [12], and ALIA [9]. To further analyze the gain of our model, we try Weighted Ensembling (WE) method [41], which can further improve model performance by integrating prior knowledge from pretrained model:

- • LP-FT: A two-step process involving linear probing and full fine-tuning of model to avoid distortion of pretrained features, to improve OOD generalization.
- • FLYP: Finetuning with the pretraining (contrastive) loss.
- • ALIA: Diffusion-based data-augmentation on fine-grained classification tasks.
- • WE: Linearly merging the weights of pretrained and finetuned model.

We conduct ablation study to analyze the effect of DisCL with different hyper-parameters introduced in §4.1, and the newly introduced ablation hyper-parameters:

• DisCL: employs multiple levels of guidance scale and a range of curriculum strategies:

- – Easy to Hard: Non-adaptive strategy with guidance changing from smallest (easiest & most prototypical features) to largest (hardest & task-specific features).
- – Random: Randomly select guidance at each train stage.

Results We present the results of our method and comparable baselines for the iWildCam dataset in Table 4. Compared to the nearest competing baseline, DisCL significantly enhances the OOD F1 performance by 2.6%. Additionally, DisCL boosts the ID F1 performance by 2.1%. Among

7We use hyperparameters provided in Goyal et al. [12] with a batchsize of 128 to train the model.

iWildCam

Method Curriculum OOD ID

CLIP (zero-shot) 11.0 (-) 8.7 (-)

Baselines

LP-FT N/A 34.7 (0.4) 49.7 (0.5) LP-FT + WE N/A 35.7 (0.4) 50.2 (0.5) FLYP† N/A 35.5 (1.1) 52.2 (0.6) FLYP + WE† N/A 36.4 (1.2) 52.0 (1.0) FLYP + ALIA N/A 36.9 (0.3) 52.6 (0.4)

FLYP + Text-only Guidance N/A 34.2 (0.4) 51.4 (0.3) FLYP + Fixed Guidance N/A 36.0 (0.3) 50.8 (0.6) FLYP + All-Level Guidance N/A 36.5 (0.6) 53.4 (0.5) FLYP + DisCL Easy-to-Hard 35.2 (0.9) 51.4 (0.5) FLYP + DisCL Random 35.9 (0.1) 52.1 (0.2) FLYP + DisCL [Lower CLIPScore Threshold] Adaptive 37.1 (0.8) 50.9 (0.9) FLYP + DisCL [Higher CLIPScore Threshold] Adaptive 38.1 (1.3) 52.8 (0.8)

Ablations

FLYP + DisCL Adaptive 38.2 (0.5) 54.3 (1.4) FLYP + DisCL + WE Adaptive 38.7 (0.4) 54.6 (0.7)

Ours

Table 4. In-distribution (ID) and out-of-distribution (OOD) macro F1 score of low-quality image learning on iWildCam with CLIP ViT-B/16 model. The best performance is highlighted in bold. † marks our reproduced results using the original paper provided code. Baselines are defined in §4.2.

OOD Accuracy

###### ID Accuracy

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

60

F1ScoreoniWildCam

50

40

CLIP (Zero Shot) FLYP FLYP + DisCL

| |
|---|

30

| |
|---|

20

10

Without WE With WE

Without WE With WE

- Figure 3. iWildCam accuracy of CLIP ViT-L/16 model trained with and without weight ensembling (WE). The best model performance is highlighted with a red horizontal line.

all evaluated methods, DisCL achieves the highest scores in both OOD and ID generalization, underscoring its effectiveness for low-quality classification. Moreover, our model could still deliver performance improvements on larger model when using ViT-L/14, as shown in Fig 3; DisCL achieves gains of 2.8% in OOD F1 and 3.7% in ID F1. These findings reinforce the versatility and robustness of the DisCL framework across different model scales and complexities. We further study the performance of model after employing WE method. DisCL still benefits from this method and maintains superior performance compared to other methodologies, despite integrating prototypical features from synthetic data that might overlap with the pretrained model’s knowledge.

#### 5. Ablation Study and Analysis

##### 5.1. Effect of Syn-to-Real Interpolation Data

We examine the effectiveness of using a spectrum of data generated with our DisCL method, by comparing All-Level Guidance and Text-only Guidance rows in both the task tables (INLT and iWildCam). For IN-LT results in Table 1, All-Level Guidance brings ∼1.27% gain in few-class accuracy, alongwith significant gains across other class-categories. Likewise, All-Level Guidance shows a superior ID and OOD perfor-

mance as compared to Text-only Guidance for the iWildCam as well, see Table 4. These findings corroborate that utilizing a spectrum of data with multiple guidance levels helps mitigate the negative effects of the distribution gap.

- 5.2. Effect of Curriculum Learning Strategy Long Tail Classification We compare the impact of our Diverse to Specific curriculum strategy tailored for IN-LT task against other strategies, notably All-Level Guidance which employ no curriculum and uses all synthetic data. The Diverse to Specific demonstrate a higher few-class accuracy with a margin of 4.47%, see Fig. 4a. We then compare it with a reverse strategy Specific-to-Diverse, and found the latter one to be worse. The reverse strategy can overfit model to real distribution early on, increasing the gap between real and synthetic data; hence, later-stage training on the data with larger distribution gap can decrease models’ few-class accuracy. For IN-LT, we also try Adaptive strategy (mainly developed for learning from low-quality data), in which strategy’s progression is based on a validation set, comprising few tail images sampled from each guidance scale and few original images. But, validation set is scarce interms of tail samples, which renders it ineffective for identification of truly useful guidance. Hence, this strategy ranks as the least effective for LT task.

Learning from Low Quality Data For iWildCam task, we study the effect of our designed Adaptive strategy, catering to the challenge of learning from low quality data. As shown in Fig. 4b, for this task, Adaptive surpasses the All-Level Guidance with a clear margin, underscoring the benefit of using progressive curriculum over using all synthetic data. Further comparisons with the Non-Adaptive curricula including Easy-to-Hard and Random, show an impactful increase in OOD F1, while using our Adaptive.

These findings highlight how the structured data selection used in Diverse-to-Specific, is more effective in directing model’s focus on scarce data (classes), however, when dealing with real-world low-quality data, an Adaptive strategy is more successful in adjusting to models’ needs by adaptively selecting the suited data.

- 5.3. Effect of CLIPScore Threshold Long Tail Classification Our analysis of CLIPScore distribution on IN-LT generated data leads us to infer that the best CLIPScore threshold for filtering is 0.3 (detailed explained in the Appendix A.2.2). We then assess different CLIPScore thresholds with the Diverse to Specific curriculum strategy, by experimenting with different values: lower (0.28), and higher (0.32), shown in Fig. 4c. However, we find that changing the CLIPScore threshold does not significantly affect the performance. As shown in Fig. 6b, CLIPScore of synthetic data remains concentrated, as Stable Diffusion model performs well on generating high-quality images for ImageNet classes. Changes in the CLIPScore threshold will

Few Accuracy (%)

OOD F1 Score (%)

All-Level

All-Level

Adaptive Specific to Diverse Diverse to Specific

Easy to Hard

Random

Adaptive

10 15 20 25

35 40

(a) Strategies: IN-LT

(b) Strategies: iWildCam

Few Accuracy (%)

OOD F1 Score (%)

Lower

Lower

DisCL

DisCL

Higher

Higher

10 15 20 25

32 34 36 38 40

(c) Thresholds: IN-LT

(d) Thresholds: iWildCam

- Figure 4. Ablation study of Curriculum Strategies (a,b) and CLIPScore Thresholds (c,d) & on ImageNet-LT and iWildCam respectively. Error bar reports the standard deviation of each experiment.

not significantly affect the quality of synthetic images and corresponding effects in downstream classification tasks.

Learning from Low Quality Data In the iWildCam task, we identify the optimal threshold as 0.25. To further validate this choice, we experiment with nearby thresholds (0.23 and 0.27) with the chosen Adaptive Curriculum strategy suited for low-quality image classification. As depicted in Fig. 4d, the 0.25 threshold markedly improves OOD performance compared to other CLIPScore thresholds. Unlike the ImageNet dataset, the iWildCam images are characterized by significant difficulty and poor quality, leading to high variance in CLIPScores of synthetic data (as shown in Fig. 7b). In this scenario, adjusting the CLIPScore threshold can impact model performance. When a higher threshold is used, the selected synthetic images include more prototypical visual features but they are less similar to the original images. Hence, they improve OOD performance but lead to a drop of ID F1 score.

The ablation study results on two classification tasks demonstrate that the selection of the CLIPScore threshold should be carefully aligned with the generation quality inherent to the task-at-hand.

- 5.4. Scaling Synthetic Data for Long Tail Learning We empirically analyze the effect of scaling synthetic tail data on IN-LT performance using ResNet-10 model in Fig.5. DisCL consistently improves the few class accuracy upto 3-4X scale ; however, beyond this point, the gains diminish with a slight degradation in both many and medium classes. As a result, we chose to use 3X scale of synthetic tail data for all DisCL training experiments. Notably, many-class accuracy shows the lowest degradation across scaling, confirming the findings of long-tail learning that hard-sample

###### Increasing scale of Synthetic Tail Data

no-syn data 1X tail data

- 2X tail data

- 3X tail data

- 4X tail data

- 5X tail data

###### Many Catgory

###### Med Category

Few Category

- 50

- 51

- 52

- 53

- 54

- 55

- 56

- 57

- 58

- 59

- 60

- 25

- 26

- 27

- 28

- 29

- 30

26

24

22

20

IN-LTAccuracy(%)

IN-LTAccuracy(%)

IN-LTAccuracy(%)

18

16

14

12

10

8

6

4

2

0

Figure 5. Ablation study on scale of Synthetic data (generated for tail class samples) used in DisCL training with ResNet-10 on IN-LT task. Here X refers to the original number of tail classes’ samples.

synthetic data can improve tail class generalization without disrupting many-class representations.

#### 6. Conclusion

In this paper, we introduce DisCL, a novel paradigm designed to enhance model performance when dealing with low-quality or scarce data. DisCL effectively bridges the distribution gap between original and target data using a spectrum of synthetic data, particularly for challenging samples. Our method utilizes image guidance in diffusion models to generate a comprehensive range of interpolated data from synthetic to real. Additionally, we design specific curricula to maximize the benefits of synthetic data for learning hard samples and closing the gap between synthetic and real data. The efficacy of DisCL is demonstrated through its significant and robust performance improvements in longtail classification and learning from low-quality data, across various base model settings. Our analyses reveal that the interpolation of synthetic-to-real data, the selection of guidance intervals, and the proposed curriculum strategy are all essential components contributing to these gains.

Despite the promising results, the performance of DisCL is influenced by certain limitations. The quality of the generated data spectrum is dependent on the capabilities of the diffusion model and the visual-text alignment ability of filtering models. These dependencies constrain the overall performance of DisCL. Additionally, the current approach to generate text prompts for long-tail classification relies solely on category names derived from large language models (LLMs). To better align with the real data distribution and to reduce the gap between synthetic and real data, future works could focus on generating text prompts from image captions. Lastly, discrepancies in the position and size of class objects between real and synthetic images can widen the distribution gap. Addressing this issue may involve detecting objects and performing crop operations on real images or using detailed prompts to control these properties in synthetic data. These areas present opportunities for further research and improvement.

#### References

- [1] Sumyeong Ahn, Jongwoo Ko, and Se-Young Yun. Cuda: Curriculum of data augmentation for long-tailed recognition. arXiv preprint arXiv:2302.05499, 2023. 1, 3, 5, 16
- [2] Hritik Bansal and Aditya Grover. Leaving reality to imagination: Robust classification via generated datasets. arXiv preprint arXiv:2302.02503, 2023. 2
- [3] Sara Beery, Yang Liu, Dan Morris, Jim Piavis, Ashish Kapoor, Neel Joshi, Markus Meister, and Pietro Perona. Synthetic examples improve generalization for rare classes. In Proceedings of the ieee/cvf winter conference on applications of computer vision, pages 863–873, 2020. 1
- [4] Sara Beery, Arushi Agarwal, Elijah Cole, and Vighnesh Birodkar. The iwildcam 2021 competition dataset. arXiv preprint arXiv:2105.03494, 2021. 2, 6
- [5] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009. 2
- [6] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with label-distribution-aware margin loss. Advances in neural information processing systems, 32, 2019. 5, 14
- [7] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 2
- [8] Kevin Clark and Priyank Jaini. Text-to-image diffusion models are zero shot classifiers. Advances in Neural Information Processing Systems, 36, 2024. 12
- [9] Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph E Gonzalez, and Trevor Darrell. Diversify your vision datasets with automatic diffusion-based augmentation. Advances in Neural Information Processing Systems, 36, 2024. 1, 2, 3, 6
- [10] Yunxiang Fu, Chaoqi Chen, Yu Qiao, and Yizhou Yu. Dreamda: Generative data augmentation with diffusion models. arXiv preprint arXiv:2403.12803, 2024. 12
- [11] Irena Gao, Shiori Sagawa, Pang Wei Koh, Tatsunori Hashimoto, and Percy Liang. Out-of-distribution robustness via targeted augmentations. In NeurIPS 2022 Workshop on Distribution Shifts: Connecting Methods and Applications, 2022. 1
- [12] Sachin Goyal, Ananya Kumar, Sankalp Garg, Zico Kolter, and Aditi Raghunathan. Finetune like you pretrain: Improved finetuning of zero-shot vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19338– 19347, 2023. 6

- [13] Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan Zhuang, Dengke Dong, Matthew R Scott, and Dinglong Huang. Curriculumnet: Weakly supervised learning from large-scale web images. In Proceedings of the European conference on computer vision (ECCV), pages 135–150, 2018. 2
- [14] Thomas A Halgren, Robert B Murphy, Richard A Friesner, Hege S Beard, Leah L Frye, W Thomas Pollard, and Jay L Banks. Glide: a new approach for rapid, accurate docking and scoring. 2. enrichment factors in database screening. Journal of medicinal chemistry, 47

(7):1750–1759, 2004. 2

- [15] Pengxiao Han, Changkun Ye, Jieming Zhou, Jing Zhang, Jie Hong, and Xuesong Li. Latent-based diffusion model for long-tailed recognition. arXiv preprint arXiv:2404.04517, 2024. 1, 5
- [16] Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, and Xiaojuan Qi. Is synthetic data from generative models ready for image recognition? arXiv preprint arXiv:2210.07574, 2022. 2
- [17] Reyhane Askari Hemmat, Mohammad Pezeshki, Florian Bordes, Michal Drozdzal, and Adriana RomeroSoriano. Feedback-guided data synthesis for imbalanced classification. arXiv preprint arXiv:2310.00158,

2023. 2

- [18] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. 3, 12
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3
- [20] Chengkai Hou, Jieyu Zhang, and Tianyi Zhou. When to learn what: Model-adaptive data augmentation curriculum. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 1717– 1728, 2023. 3
- [21] Lu Jiang, Deyu Meng, Shoou-I Yu, Zhenzhong Lan, Shiguang Shan, and Alexander Hauptmann. Self-paced learning with diversity. Advances in neural information processing systems, 27, 2014. 2
- [22] Lu Jiang, Deyu Meng, Qian Zhao, Shiguang Shan, and Alexander Hauptmann. Self-paced curriculum learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2015. 2
- [23] Jae Myung Kim, Jessica Bader, Stephan Alaniz, Cordelia Schmid, and Zeynep Akata. Datadream: Fewshot guided dataset generation. In Computer Vision – ECCV 2024, pages 252–268. Springer, 2024. 1
- [24] Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-

- distribution. arXiv preprint arXiv:2202.10054, 2022. 6
- [25] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Large-scale long-tailed recognition in an open world. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2537–2546, 2019. 2, 5
- [26] Hongyuan Lu and Wai Lam. PCC: Paraphrasing with bottom-k sampling and cyclic learning for curriculum data augmentation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 68–82, Dubrovnik, Croatia, 2023. Association for Computational Linguistics. 3
- [27] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073,

2021. 3

- [28] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. 2
- [29] Jiawei Ren, Cunjun Yu, shunan sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, and hongsheng Li. Balanced metasoftmax for long-tailed visual recognition. In Advances in Neural Information Processing Systems, pages 4175– 4186, 2020. 5, 6, 21
- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3
- [31] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479– 36494, 2022. 2
- [32] Mert Bulent Sariyildiz, Karteek Alahari, Diane Larlus, and Yannis Kalantidis. Fake it till you make it: Learning (s) from a synthetic imagenet clone. 2022. 2
- [33] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clipfiltered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 3, 12
- [34] Jie Shao, Ke Zhu, Hanxiao Zhang, and Jianxin Wu. Diffult: Diffusion for long-tail recognition without external knowledge. In Advances in Neural Information

- Processing Systems, pages 123007–123031. Curran Associates, Inc., 2024. 2
- [35] Jiang-Xin Shi, Tong Wei, Yuke Xiang, and Yu-Feng Li. How re-sampling helps for long-tail learning? In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 16
- [36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 5
- [37] Michael A. Tabak, Mohammad S. Norouzzadeh, David W. Wolfson, Steven J. Sweeney, Kurt C. Vercauteren, Nathan P. Snow, Joseph M. Halseth, Paul A. Di Salvo, Jesse Lewis, Michael D. White, Ben Teton, James C. Beasley, Peter E. Schlichting, Raoul K. Boughton, Bethany Wight, Eric S. Newkirk, Jacob S. Ivan, Eric A. Odell, Ryan K. Brook, Paul M. Lukacs, Anna K. Moeller, Elizabeth G. Mandeville, Jeff Clune, and Ryan S. Miller. Machine learning to classify animal species in camera trap images: Applications in ecology. Methods in Ecology and Evolution, 10(4):585–590, 2019. Publisher Copyright: © 2018 The Authors. Methods in Ecology and Evolution © 2018 British Ecological Society. 5
- [38] Brandon Trabucco, Kyle Doherty, Max Gurinas, and Ruslan Salakhutdinov. Effective data augmentation with diffusion models. arXiv preprint arXiv:2302.07944, 2023. 12
- [39] Soobin Um, Suhyeon Lee, and Jong Chul Ye. Don’t play favorites: Minority guidance for diffusion models. In ICLR, 2024. 1
- [40] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8769–8778, 2018. 5, 14
- [41] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust finetuning of zero-shot models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7959–7971, 2022. 6
- [42] Seonghyeon Ye, Jiseon Kim, and Alice Oh. Efficient contrastive learning via novel data augmentation and curriculum learning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1832–1838, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. 3
- [43] Zhenghang Yuan, Lichao Mou, Qi Wang, and Xiao Xiang Zhu. From easy to hard: Learning language-guided curriculum for visual question answering on remote

- sensing data. IEEE transactions on geoscience and remote sensing, 60:1–11, 2022. 2
- [44] Tianyi Zhou and Jeff Bilmes. Minimax curriculum learning: Machine teaching with desirable difficulties and scheduled diversity. In International Conference on Learning Representations, 2018. 2
- [45] Tianyi Zhou, Shengjie Wang, and Jeffrey Bilmes. Curriculum learning by dynamic instance hardness. In Advances in Neural Information Processing Systems, pages 8602–8613. Curran Associates, Inc., 2020. 2
- [46] Tianyi Zhou, Shengjie Wang, and Jeff Bilmes. Curriculum learning by optimizing learning dynamics. In International Conference on Artificial Intelligence and Statistics, pages 433–441. PMLR, 2021. 5, 19
- [47] Tianyi Zhou, Shengjie Wang, and Jeff A. Bilmes. Robust curriculum learning: from clean label detection to noisy label self-correction. In International Conference on Learning Representations (ICLR), 2021. 2

## Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion

### Supplementary Material

#### A. Appendix

- A.1. Motivation for DisCL’s Data Selection When curating data for a training curriculum, real data often aligns with the test distribution better but suffers from deficiency, noise, low quality, or imbalance; Synthetic data can potentially fix these problems but suffers from a large distribution gap to the test. Our synthetic-to-real curriculum is designed to combine the complementary strengths of both data types and overcome their weaknesses. Unlike previous methods using synthetic data with no real-image guidance or a fixed guidance level, DisCL dynamically adjusts the real-image guidance level per training stage to generate a spectrum of synthetic-to-real samples that accelerate learning progress and meanwhile progressively bridging the distribution gap. Unlike pre-defined easy-to-hard curricula on real data, DisCL’s data selection is adaptive to the training dynamics, considers diversity and distribution gap, and is optimized for achieving the greatest progress per stage.
- A.2. Synthetic Data Generation with Image Guidance

In this section, we visualize more generated images in (Phase 1) of our method with various levels of image guidance, for two different classification tasks.

- A.2.1. Generation Settings and Statistics We provide the statistics for the synthetic data generation within our paradigm on ImageNet-LT, CIFAR100-LT, iNaturalist2018, and iWildCam, as shown in Table 5.
- A.2.2. ImageNet-LT Synthetic Generation Selection of Text prompts To improve model performance on the minority classes, high-quality and diverse synthetic samples are required. To achieve so, we follow the approach in Fu et al. [10], and utilize publicly available GPT-3.5-turbo to generate diverse prompts for these 1000 IN-LT classes. We use the following prompt to query GPT-3.5-turbo for generating descriptions for class X:

“Please provide 10 language descriptions for random scenes that contain only the class X from the ImageNet-LT dataset. Each description should be different and contain a minimum of 15 words. These descriptions will serve as a guide for Stable Diffusion in generating images.”

The sample-prompts generated by GPT-3.5-turbo are listed in Table 6.

Selection of Images Guidance Levels We first analyze the cosine similarity between synthetic images and real images, as well as between synthetic images and text prompts. The similarity score between synthetic images and real images

can be used to quantify the diversity introduced in the synthetic images. As depicted in Fig. 6a, the similarity between synthetic images and real images decrease as the guidance level reduces, demonstrating the trend of increased diversity in the data spectrum. However, the changes in the scores are relatively small across varying guidance levels. Combined with the visual cases for this dataset (examples shown in Fig. 8), we observe that for images generated with high guidance levels (λ ≥ 0.7), only minor details are modified by the diffusion model, resulting in high similarity scores above 0.85. However, we aim to provide more diverse synthetic data to increase the model’s generalization on the classbalanced test set. Including these highly similar images may hinder the diversity and cause the model to overfit to specific visual features, thereby negatively impacting its generalization ability. Therefore, we select {0.0,0.1,0.3,0.5} as the interval of image guidance levels used in the training process for this dataset.

Selection of CLIPScore Threshold We leverage the widely used CLIPScore [18] to filter out poor-quality images in the synthetic data. In this method, the CLIP cosine similarity between synthetic images’ embeddings and text embeddings is computed to measure the alignment between images and the corresponding classes provided in text prompts. For the synthetic data generation for ImageNet-LT, we use a unified template that emphasizes the class information in text prompts. Following Trabucco et al. [38], we use "a photo of

<class name>" to prompt the CLIP model and compute the cosine similarity. We also consider the value of the filtering threshold for synthetic data. Following previous work [33], we set the threshold to 0.3 based on the distribution of similarity scores and a review of generation quality, as shown in Fig. 6b. We observe that a threshold of 0.3 effectively filters out synthetic images with poor quality or mismatched classes.

###### A.2.3. iWildCam Synthetic Generation

Selection of Text prompts Following previous work [8, 38], we first define prompts for each class using the template "a photo of <class>". However, the classnames in iWildCam comprises of scientific names, which are usually unseen/unknown concepts to the diffusion text encoder. For example, "canis lupus" is the class name for "wolf" animal. To address this, we replace the scientific names with their common names and add a postfix "in the wild" in the prompt to drive the generation of wild images. The final text prompt we use is "a photo of <common name of class> in the wild".

###### Images’ Details ImageNet-LT CIFAR100-LT iNaturalist2018 iWildCam

Irb=100 Irb=50

No. of Hard Samples 1643 324 268 44956 8260 Number of Image Guidance Scales λ 4 4 4 4 3 Number of Random Seed Per Image 8 8 8 4 8 Number of Generated Images 51917 2592 2144 179824 197756 Number of Generated Images After Filtering 24141 809 668 75234 90093 Acceptance Rate 46.50% 31.21% 31.16% 41.84% 45.56%

Table 5. Statistics about Generated Synthetic Data. Irb refers to the imbalance ratio used to sample CIFAR100-LT dataset.

1.0

0.9

ImageSimialrityScore

0.8

0.7

0.6

0.5

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Guidance Level

(a) Similarity b/w synthetic images & its original real image.

0.40

0.35

CosineSimialrityScore

0.30

0.25

0.20

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Guidance Level

(b) Similarity b/w synthetic images & defined text prompt.

Figure 6. CLIP Cosine similarity score for ImageNet-LT Synthesis.

Selection of Images Guidance Levels Based on the generated data with multiple image guidance scales, we search for effective image guidance scales for this task using CLIP cosine similarity scores between synthetic image embeddings and real image embeddings. As shown in Fig. 7a, as the difference between real images and synthetic images increases, the cosine similarity between image embeddings decreases from λ = 1 to λ = 0.3. However, when the image guidance continues to decrease to λ = 0, the cosine similarity score increases slightly. With low image guidance scales, the diffusion model tends to generate images that heavily rely on text information, maintaining only global information (such as the color of the image background) in the synthetic data for some images. This creates a distribution gap between these synthetic data and real data that is too large for the model to accurately compare the differences between the two images using embedding representation. Additionally, based on the analysis of the quality of synthetic images and to leverage the difficulty of the features and the distribution gap between synthetic and real data, we set the image guidance scales to {0.5,0.7,0.9} for this task.

Selection of CLIPScore Threshold To filter out lowquality images, we assess the CLIP cosine similarity scores between synthetic image embeddings and corresponding text embeddings for each class. We use the same prompt template

as in the generation process ("a photo of <common name for animal> in the wild") to compute CLIPScore for synthetic images. The distribution of CLIPScores is shown in Fig. 7b, which reveals a distinct gap around 0.25. Combined with a review of the quality of synthetic data, we set the threshold to 0.25. Synthetic data with a CLIPScore lower than 0.25 are considered poor-quality samples.

A.2.4. Visualization Visual Cases We provide additional visual examples of synthetic data generated with multiple guidance levels and text prompts for the ImageNet-LT and iWildCam datasets. The results are visualized in Fig. 8 and Fig. 9. These examples demonstrate that the model can generate synthetic data with various postures, backgrounds, and actions as the image guidance level decreases. Particularly for ImageNet-LT generation results, diverse prompts introduce more varied features into low-guidance data. These diverse features enable the model to achieve better generalization on the target distribution.

Failure Cases During generation, despite designing text prompts and applying CLIPScore to filter to remove lowquality data, some failure cases still occur in the synthetic dataset. In this section, we discuss these failure cases encountered during the generation process. As shown in Fig. 10 and Fig. 11, the first failure case is caused due to the in-

Class Name Prompts Grand Piano A grand piano sits elegantly in a sunlit room, its glossy finish reflecting the warm glow.

In a cozy living room, the grand piano adds a touch of luxury and sophistication to the space.

The grand piano sits silently in a dimly lit room, waiting patiently for a skillful pianist to bring it to life.

In a grand ballroom, the grand piano provides a majestic backdrop for a glamorous event. A vintage grand piano exudes timeless elegance in a quaint parlor, filled with antique charm.

Pufferfish A colorful pufferfish swimming gracefully in a crystal-clear ocean, surrounded by vibrant

coral reefs.

A group of playful pufferfish blowing bubbles and chasing each other in a sunlit underwater cave.

A shoal of pufferfish moving in unison, creating a mesmerizing dance of synchronized swimming in the deep sea.

A fierce pufferfish defending its territory from intruders, puffing up its body and displaying its sharp spikes as a warning.

A baby pufferfish following its larger parent closely, learning the ropes of survival in the vast ocean ecosystem.

Table 6. Generated text prompts for ImageNet-LT classes

1.0

0.9

ImageSimialrityScore

0.8

0.7

0.6

0.5

0.4

0.3

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Guidance Level

(a) Synthetic image & original real images.

0.35

CosineSimialrityScore

0.30

0.25

0.20

0.15

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Guidance Level

(b) Synthetic image & defined text prompt.

Figure 7. CLIP Cosine similarity score for iWildCam Synthesis.

ability to recognize objects in the original images. If these objects are clearly obscured or hard-to-identify (e.g. second case in Fig. 11 and first case in Fig. 10), diffusion models cannot accurately identify the object or modify details for generating diverse and useful data. For these seed images, only synthetic data generated with a low-guidance scale can achieve a CLIPScore higher than the threshold. However, this approach compromises the smooth transition of data from synthetic to real distribution. Even though the diffusion model can generate images with a smooth transition for most-of-the-cases, our quality-check on synthetic data can constrain the feature extraction and alignment ability of the CLIP model. For example, in second case of Fig. 10, CLIPScore filters out the slightly modified but perceptually

useful images, containing prototypical class features.

##### A.3. Application of DisCL to Other Datasets and Model Scale

To further assess the robustness of DisCL, we extend our experiments to two additional widely used imbalanced datasets: CIFAR-100-LT [6] and iNaturalist2018 [40]. For iNaturalist2018, we generate synthetic data following the same approach and settings used for the long-tail classification task on ImageNet-LT. In the case of CIFAR-100-LT dataset, due to the lower image resolution, we adjust the image guidance scale to {0.5,0.7,0.9} so as to ensure high-quality synthetic data generation. Visual examples of the generated data are shown in Fig. 12 and 13. For CIFAR-100-LT, we evaluate

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 16]

[Figure 17]

RandomSeeds&Prompts

A majestic American Staffordshire Terrier standing proudly in a lush green field, with the sun setting in the background.

A loyal American Staffordshire Terrier standing guard at the front door, alert and ready to protect its family from any intruders.

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 18]

[Figure 19]

RandomSeeds&Prompts

A vibrant American robin perched on a tree branch, its red breast glowing in the sunlight.

A group of American robins bathing in a shallow puddle, splashing water everywhere in their joy.

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

RandomSeeds&Prompts

[Figure 24]

A soft bath towel hanging on a metal rack in a modern bathroom. A fluffy bath towel neatly folded on a wooden shelf next to a bathtub.

[Figure 25]

[Figure 26]

Figure 8. Synthetic generation with various image guidance and random seeds based on ImageNet-LT.

[Figure 27]

[Figure 28]

RandomSeeds

A photo of <Ocelot> in the wild

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 29]

RandomSeeds

A photo of <African bush elephant> in the wild

[Figure 30]

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 31]

[Figure 32]

RandomSeeds

A photo of <grey-cowled wood rail> in the wild

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Figure 9. Synthetic generation with various image guidance and random seeds based on iWildCam.

the performance of DisCL under different imbalance ratios (50 and 100). Additionally, we expand our model evaluation to a larger scale, ResNet-34 (widely adopted for ImageNet) with the same experimental settings of DisCL as before. As evident from Table 2 and Table 3, our results demonstrate that DisCL achieves a notable improvements in overall top-1 accuracy (e.g., +1–3.3% over baselines) and few class performance (e.g., +3-8% for tail classes) across both datasets. We also notice that combining a class-reweighting loss (BS) with DisCL causes an oversaturation in tail-class signals, causing the model to neglect many classes during the training. This suggests that reweighting and mixing synthetic with real data address different aspects of class imbalance; aligning with prior works, [1] and [35]. Notably, the top-1 accuracy gains persist when scaling the model to ResNet-34, as demon-

strated for CIFAR-100-LT in Table 7 and ImageNet-LT in Table 8. This underscores the flexibility of our proposed DisCL method across different datasets and model scales.

##### A.4. Training with Curriculum Learning

###### A.4.1. Long-Tail Learning with Non-Adaptive Strategy

For long-tail classification, we propose a non-adaptive curriculum learning strategy that starts with the lowest guidance and progressively increases to the highest guidance within the defined interval Λ. We employ a linear scheduler to adjust the guidance levels during training, allowing the model to train with data from various guidance levels for equal durations. Furthermore, the test set of ImageNet-LT is indistribution to its training data; unlike the training data, it is a class-balanced set. To mitigate the potential negative effects of the distribution gap between synthetic and real data, all

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

RandomSeeds&Prompts

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

A plush bath towel thrown over a wooden stool by a glass shower in a minimalist bathroom.

A luxurious bath towel hanging on a chrome hook in a sleek hotel bathroom.

[Figure 46]

[Figure 47]

[Figure 48]

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

RandomSeeds&Prompts

[Figure 54]

[Figure 55]

[Figure 56]

An acorn squash sitting on a wooden table with its vibrant green color standing out.

A whole acorn squash cut in half, revealing its bright orange flesh and seeds inside.

[Figure 57]

[Figure 58]

[Figure 59]

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 60]

[Figure 61]

RandomSeeds&Prompts

[Figure 62]

[Figure 63]

A pair of European polecats engage in a fierce yet playful wrestling match on a sandy beach.

[Figure 64]

The European polecat elegantly climbs a tree, showing off its agility and grace.

###### Figure 10. Failure cases for ImageNet-LT synthetic generation

[Figure 65]

[Figure 66]

[Figure 67]

RandomSeeds

[Figure 68]

[Figure 69]

[Figure 70]

A photo of <vulturine guineafowl> in the wild

[Figure 71]

[Figure 72]

[Figure 73]

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

RandomSeeds

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

A photo of <Impala> in the wild

[Figure 85]

[Figure 86]

[Figure 87]

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 88]

[Figure 89]

RandomSeeds

[Figure 90]

[Figure 91]

[Figure 92]

A photo of <Cattle Cow Bull> in the wild

[Figure 93]

Figure 11. Failure cases for iWildCam synthetic generation

CIFAR-100-LT (Imbalance Ratio=100) CIFAR-100-LT (Imbalance Ratio=50) Method Curriculum Many Medium Few Overall Many Medium Few Overall

CE N/A 51.71 23.51 5.05 27.7 52.14 29.97 10.7 32.04 CE + DisCL Diverse to Specific 49.83 23.26 7.9 28.4 51.83 29.12 12.64 32.18

BS N/A 46.23 28.0 13.13 29.79 46.48 33.48 22.1 34.6 BS + DisCL Diverse to Specific 44.9 27.4 16.8 30.3 45.51 32.08 23.99 34.5

- Table 7. Accuracy (%) of ResNet-34 on CIFAR-100-LT classification task with imbalance ratios of 100 and 50, highlighting the best accuracy in bold for overall and class categories (many, medium, and few).

the hard tail samples from original data are involved into training at all times. Furthermore, with DisCL, number of samples for tail classes increases along with the introduction of synthetic data at each stage, however the ratio of tail-tonontail samples is still very skewed. To preserve a constant imbalance-ratio throughout all training stages and experiments, we undersample the non-tail samples at "each stage"

so that ratio of tail-samples to non-tail samples matches the proportion of tail classes to non-tail classes present in the original data (13.6%).

All experiments are conducted based on this proportion setting. Complete strategy details are covered in Algorithm 1.

ImageNet-LT Method Curriculum Many Medium Few Overall

CE N/A 63.01 35.90 10.10 42.98 CE + CUDA N/A 62.78 36.91 11.92 43.34 CE + DisCL Diverse to Specific 63.54 36.93 13.64 44.26

BS N/A 62.78 36.91 11.92 43.34 BS + CUDA N/A 57.16 44.5 30.49 47.33 BS + DisCL Diverse to Specific 58.82 45.21 32.53 48.42

- Table 8. Accuracy (%) of ResNet-34 on ImageNet-LT classification task, highlighting the best accuracy in bold for overall and class categories (many, medium, and few).

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0.5

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0.5

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

RandomSeeds&Prompts

RandomSeeds&Prompts

[Figure 98]

[Figure 99]

[Figure 100]

Figure 12. Synthetic generation with various image guidance and random seeds based on CIFAR100. Sample Prompt: (1) A bright sunflower standing tall in a field, basking in the warm sunlight of a summer day. (2) A majestic whale breaches the surface of the deep blue ocean, sending a spray of water into the air.

- A.4.2. Learning from Low-Quality Data with “Adaptive

learning face is introduced by DoCL [46] as shown in Eq 4.

Curriculum” Strategy

∂f(x)

An approximation method to assess the effectiveness of samples in helping model achieve greatest progress on and fastest

Ex∈D,x∼D⟨y − f(x),

∂t |S⟩ ≈

(4)

∂f(x(j)) ∂t |D⟩

1 |D| j∈V

⟨y(j) − f(x(j)),

Algorithm 1: Training with non-adaptive curriculum strategy

where D is the training distribution and x ∈ D is a set of finite samples randomly sampled from the original distribution D. V denotes the subset of samples from S. Here, y and f(x) denotes the target-class and sample prediction.

Input: Image guidance levels Λ = {λi | λi ∈ [0, 1]}, Non-hard samples Dnh = {(x(j), y(j), λ(j) = 1)}Nj=1, Spectrum of syn-to-real data

S = {(x′(j), y(j), λ(j)) | λ(j) ∈ Λ}Mj=1, Original hard samples Dh = {(x(j), y(j), λ(j) = 1) | (x(j), y(j), λ(j)) ∈ S},

⟨y−f(x), ∂f∂t(x)|V⟩ represents the project of residual y−f(x) on the model dynamics ∂f∂t(x)|V. This equation indicates that when trained with subset V, the expected progress E of samples in the original training dataset can be approximated by the progress of samples on subset V achieved via training on the set D.

Total training epochs E, curriculum cutoff ECL, Predefined linear guidance schedule

G = {λ1, λ2, . . . , λe, . . . , λECL} Output: Trained model fθ Initialize: Pretrained model fθ

For learning from low-quality data, we adopt DoCL and implement an adaptive curriculum strategy to select the synthetic data with best guidance level for each training stage. We showcase the implementation in Algorithm 2, wherein we preserve i for indexing the guidance level in Λ and j for indexing the sample in a given dataset. Before the training process, we randomly select samples from the spectrum for each guidance level in Λ and mark it as guidance validation set V for progress evaluation. This set has zero overlap with the training data Dall. At each training stage, we randomly sample a set D (termed as random-real set) from the training

- 1 for e ≤ ECL do

- 2 λe = G(e)
- 3 Extract Sλe = {(x′(j), y(j), λ(j)) | λ(j) = λe}
- 4 Gather new training set De = Sλe ∪ Dnh ∪ Dh
- 5 Finetune model fθ with De
- 6 end
- 7 for ECL < e ≤ E do

- 8 Gather new training set De = Dnh ∪ Dh
- 9 Finetune model fθ with De
- 10 end

𝛌=0.9 Image Guidance 𝛌: High → Low 𝛌=0

[Figure 101]

[Figure 102]

RandomSeeds&Prompts

A Cascade Golden-mantled Ground Squirrel foraging for food amongst the rocks and tall grasses in its natural habitat.

The agile Cascade Golden-mantled Ground Squirrel climbing a tree branch to reach the tasty fruits hanging above.

Figure 13. Synthetic generation with various image guidance and random seeds based on iNaturalist 2018.

dataset Dall. Before selecting the guidance level, we train the model on dataset D and evaluate the progress (in terms of classifier’s prediction score) achieved on samples of each subset Vi corresponing to a given guidance λi. We then select the λi with the highest progress to gather synthetic data and combine it with other non-hard samples from the original training data for the current training stage. This technique encourages the model to adaptively select the most informative guidance for the current training stage. At the end of the curriculum-training, to alleviate the negative effect of the distribution gap between synthetic data and real data for this task, we keep finetuning the model with real data for a short period. The steps of algorithm are detailed in Algorithm 2.

Algorithm 2: Training with adaptive curriculum strategy Input: Image guidance levels Λ = {λi | λi ∈ [0, 1]}, Non-hard samples Dnh = {(x(j), y(j), λ(j) = 1)}Nj=1, Syn-to-real spectrum data

S = {(x′(j), y(j), λ(j)) | λ(j) ∈ Λ}Mj=1, Combined training data Dall = Dnh ∪ {(x′(j), y(j), λ(j)) | λ(j) = 1}, Guidance validation set V = {(x′(j), y(j), λ(j)) | λ(j) ∈ Λ}mj=1, Total training epochs E, curriculum cutoff epoch ECL, size of real-random set |D| Output: Trained model fθ Initialize: Pretrained model fθ /* Note: V has no overlap with Dall */

- 1 for e ≤ ECL do

- 2 Compute true-class probability pbef of model fθ on V
- 3 Sample a random-real set D from Dall /* contains only real data */
- 4 Train model fθ with D
- 5 Compute true-class probability paft of model fθ on V
- 6 λe ← arg maxλi∈Λ (paft(λi) − pbef(λi))
- 7 Extract Sλe = {(x′(j), y(j), λ(j)) | λ(j) = λe}
- 8 Form training set De = Sλe ∪ Dnh
- 9 Train model fθ with De
- 10 end
- 11 for ECL < e ≤ E do

- 12 Train model fθ with Dall
- 13 end

- A.5. Hyperparameters for Synthetic Generation and Model Training

The values of all hyperparameters used for synthetic data generation with diffusion model and curriculum learning strategy are listed in Table 9.

For ImageNet-LT, we implement baselines based on the codebase and the pretrained model from LDMLR. We also re-implement CUDA baseline from this codebase, containing some missing models. We use the same hyper-parameter settings as listed in the CUDA paper. For FLYP, we implement baseline models with FLYP codebase and leverage the available pretrained model from Open CLIP.

- A.6. Computational Requirements for Synthetic Generation

For computational requirements of offline generation, 1 RTX A5000 GPU is used to generate synthetic images. For time efficiency, It took 10 seconds to generate a full spectrum (6 image guidance levels) of synthetic images for each real image with resolution=480 × 270.

- A.7. Further Discussion on Experiment Results In this section, we analyze the results of each guidance level under Fixed Guidance experiment to observe the effect of

different image guidance levels on the classifier’s performance. During the training process, synthetic data generated from only a specific guidance level combined with original real data is presented to the model. The ablation numbers are shown in Fig. 14.

For the iWildCam dataset, data generated with text-only guidance (λ = 0) has the largest distribution gap between

###### Hyperparameter Name Value

Text Guidance Scale w 10 Noise Scheduler DDIM Stable Diffusion Denoising Steps 1000 Stable Diffusion Checkpoint stabilityai/stable-diffusion-xl-refiner-1.0 CLIP Filter Model openai/clip-vit-base-patch32 Filtering Threshold for iWildCam 0.25 Filtering Threshold for ImageNet-LT 0.30 GPU Used Nvidia rtx5000 with 24GB

SyntheticGeneration

Level of Image Guidances λ {0,0.1,0.3,0.5,1.0} CLIP Filtering Threshold 0.3 Batch Size for ResNet-10 128 Learning Rate 1e-3 Optimizer Adam Scheduler Cosine Training Epoch 65 Training Epoch for Curriculum Learning 60 GPU Used Nvidia rtx5000 with 24GB

ImageNet-LT

Level of Image Guidances λ {0.5,0.7,0.9,1.0} CLIP Filtering Threshold 0.25 Size of Dataset D 30000 Size of Guidance Validate Dataset S 2000 Batch Size for CLIP ViT-B/16 256 Batch Size for CLIP ViT-L/16 200 Learning Rate 1e-5 Optimizer AdamW Scheduler Cosine with Warmup Warmup Step 500 Training Epoch 20 Training Epoch for Curriculum Learning 15 GPU Used 2 Nvidia A100 with 80GB

iWildCam

Table 9. Hyperparameters and their values

synthetic and real data, and it also showcases lowest Outof-Distribution (OOD) performance. As the guidance scale increases, this distribution gap diminishes, and the OOD F1 score consistently improves. This outcome aligns with the visually observed reduction in distribution differences between generated and real images.

Conversely, the trend seen with ImageNet-LT diverges from above. In long-tail classification, we aim to increase data diversity while keeping the distribution gap small. As detailed in Appendix A.2.2, on one hand, generating synthetic data that closely resemble real data further reduces the diversity, and generating synthetic data far from real distribution can offer diversity but hurt OOD performance. In case of ImageNet-LT, we observe that more diverse synthetic data tends to significantly improve the classifiers’ generalization.

Inspired by these observations, we tailor our guidance scales intervals according to the task-at-hand.

##### A.8. Improvement on Worst-k classes: Balanced Softmax (BS) v/s DisCL with BS [29]

While DisCL’s average gain over Balanced Softmax Baseline(BS) is +2.07%, it improves BS’s worst-k class accuracy by 4.5%–7.6%, verifying our targeted advantage on the most difficult classes—precisely where strong baselines struggle. It demonstrates that DisCL complements existing methods, improving performance where it matters most, even compared with strong baselines.

k 10 50 100 150 200

AccBS+DisCL − AccBS 7.6% 6.0% 5.7% 5.2% 4.5%

Table 10. Improvement in Accuracy on Worst-k classes in INLT.

###### A.9. Societal Impact Our proposed method is beneficial for diverse fields, where inadequate quantity and low quality of data is common, e.g. medical domain. The synthetic data generation, as followed by DisCL approach can reduce the need for extensive data

40

CE

Baseline

DisCL CUDA

DisCL

20

38

FewAccuracy(%)

OODF1Score(%)

15

36

10

34

5

32

30

0

0.0 0.1 0.3 0.5 0.7 0.9 All-Level DisCL Guidance Level

0.0 0.1 0.3 0.5 0.7 0.9 All-Level DisCL

Guidance Level

Figure 14. Effect of Image Guidance (mixing syn+real). All-level experiments use the synthesis samples from all guidance scales selected for each task. 0.5 refers to only using synthetic data with guidance level λ = 0.5 for fine-tuning. Left: results on iWildCam. Right: results on ImageNet-LT

collection, therefore mitigating the ethical concerns related to data-privacy. Overall, our method DisCL can democratize the access of effectively training ML models in the low-resource environments. However, by leveraging the pretrained generative models, the potential biases of models can perpetuate into the synthetic data and eventually affect the sensitive real-world applications consuming this data, such as medical diagnosis, law enforcement etc.

