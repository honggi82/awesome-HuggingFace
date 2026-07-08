# arXiv:2412.21059v4[cs.CV]5Jan2026

## VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation

#### Jiazheng Xu1*†, Yu Huang1∗†, Jiale Cheng1†, Yuanming Yang1†, Jiajun Xu1†, Yuan Wang1†, Wenbo Duan1†, Shen Yang1†, Qunlin Jin1†, Shurun Li1†, Jiayan Teng1†, Zhuoyi Yang1†, Wendi Zheng1†, Xiao Liu1†, Dan Zhang1†, Ming Ding2, Xiaohan Zhang2, Shiyu Huang2, Xiaotao Gu2, Minlie Huang1 , Jie Tang1 , Yuxiao Dong1

1Tsinghua University 2Z.AI {xjz22, h-y22}@mails.tsinghua.edu.cn, yuxiaod@tsinghua.edu.cn

###### Abstract

Visual generative models have achieved remarkable progress in synthesizing photorealistic images and videos, yet aligning their outputs with human preferences across critical dimensions remains a persistent challenge. Though reinforcement learning from human feedback offers promise for preference alignment, existing reward models for visual generation face limitations, including black-box scoring without interpretability and potentially resultant unexpected biases. We present VisionReward, a general framework for learning human visual preferences in both image and video generation. Specifically, we employ a hierarchical visual assessment framework to capture fine-grained human preferences, and leverages linear weighting to enable interpretable preference learning. Furthermore, we propose a multi-dimensional consistent strategy when using VisionReward as a reward model during preference optimization for visual generation. Experiments show that VisionReward can significantly outperform existing image and video reward models on both machine metrics and human evaluation. Notably, VisionReward surpasses VideoScore by 17.2% in preference prediction accuracy, and text-to-video models with VisionReward achieve a 31.6% higher pairwise win rate compared to the same models using VideoScore.

Code — https://github.com/THUDM/VisionReward

#### 1 Introduction

Visual generative models, including text-to-image (Ding

- et al. 2021; Ramesh et al. 2021; Saharia et al. 2022; Rombach et al. 2022; Betker et al. 2023; Podell et al. 2023) and text-to-video (Hong et al. 2022; Ho et al. 2022; Villegas
- et al. 2022; Zheng et al. 2024; Chen et al. 2024a; Yang et al.

*Equal contributions. Core contributors: Jiazheng, Yu, Jiale, Yuanming, Jiajun, Yuan, Wenbo, Shen and Qunlin. Corresponding author: Yuxiao (yuxiaod@tsinghua.edu.cn)

†Work done while these authors interned at Z.AI. Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

2024) generation, have recently experienced rapid developments. Through large-scale pretraining, these models can effectively translate textual descriptions into photorealistic images or temporally coherent videos. To further align them with human preferences, reinforcement learning from human feedback (RLHF) (Ouyang et al. 2022)—initially introduced in large language models—has recently been adapted to visual generation tasks (Xu et al. 2023; He et al. 2024).

A key bottleneck in applying RLHF to visual generation lies in developing effective visual reward models. Recent studies (Xu et al. 2023; Kirstain et al. 2023; Wu et al. 2023) have explored training reward models to predict human visual preferences, enabling automatic evaluation and preference optimization for visual generative models. For evaluation, reward models function as automated metrics that quantitatively measure the alignment between generated outputs and human preference criteria (Li et al. 2023). For optimization, reward models identify reliable directions for improving visual generation models. Essentially, they can provide feedback in reinforcement learning or generate preference pairs, thus reducing dependence on human annotation (Black et al. 2023; Fan et al. 2023; Clark et al. 2023).

Despite recent progress in reward models (RMs) for visual generation, two primary challenges remain: First, lack of interpretability and risk of unexpected bias. Current RMs for visual generation often suffer from limited interpretability. These models inherently involve complex trade-offs among multiple factors, yet their scoring mechanisms lack transparency regarding how such trade-offs are performed. This opacity raises concerns about potential unexpected biases. Though multimodal LLMs like Gemini (Team et al. 2024) and GPT-4o (Achiam et al. 2023) enhance interpretability through explainable rating rationales, their general-purpose architectures usually underperform specialized black-box models in fine-grained assessments (Chen et al. 2024c). This raises a key dilemma: how to design preference prediction method to be interpretable while maintaining accuracy.

###### a) VisionReward for Text-to-Video Evalution b) VisionReward for Text-to-Video Optimization

Text: In a dimly lit, ancient library, Sarah, a young woman with ﬁery red hair, pores over an old book, whispering an incantation.

Text: A child is eating pizza.

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

[Figure 12]

Original

VisionReward (Ours)

[Figure 13]

[Figure 14]

[Figure 15]

Question: Is smoothness of object's movement good?

Answer: Yes Answer: No Question: Are the details relatively reﬁned?

Answer: Yes Answer: No

DPO with VideoScore

……

[Figure 16]

[Figure 17]

[Figure 18]

| |Linear Weighted Sum of Binary VQA| |
|---|---|---|
| | | |

- 2.76 ＜ 3.33
- 3.71 ＞ 2.91

[Figure 19]

###### VideoScore

[Figure 20]

DPO with VisionReward (Ours)

✅ ❌

- Figure 1: Illustration of how VisionReward works for evaluation and optimization of visual generation. a) Evaluation: VisionReward performs comprehensive evaluation through dimension-specific binary visual QA testing, producing human-aligned, fine-grained assessment scores. b) Optimization: VisionReward enable better preference optimization, enhancing multiple key aspects.

Second, lack of effective reward models for video generation. The rapid development of text-to-video generative models has intensified the demand for video reward models. Although image reward models can assess individual frame quality, their frame-level nature inherently neglects essential temporal dependencies in video sequences. While VideoScore (He et al. 2024) has pioneered direct video evaluation through learnable metrics, it still suffers from limitations such as insufficient accuracy in preference prediction and optimization in video generation.

framework for visual preference learning. Empirically, we show that VisionReward makes the following contributions:

- • We design fine-grained multi-dimensional preference annotation and build the most fine-grained dataset which contain 81K samples and 5M binary annotation, which enable the training of VisionReward.
- • For visual preference prediction, VisionReward achieves state-of-the-art performance across multiple benchmarks, while maintaining interpretability via hierarchical diagnostic QA and explicit linear weighting. For instance, VisionReward outperforms VideoScore (He et al.

2024) by 17.2% in accuracy on preference prediction.

- • For visual generation, VisionReward can serve as an effective reward model for preference optimization (e.g., DPO), significantly enhancing the text-to-image and textto-video models. For example, video generation models with VisionReward achieve a 31.6% higher pairwise win rate compared to the same models using VideoScore.

Contribution. To address these challenges, we propose a general framework VisionReward to build accurate reward models for both image and video generation. VisionReward is trained with two steps: fine-grained visual assessment and interpretable preference learning. First, to capture human visual preferences, we identify nine major dimensions and decouple preferences into 64 fine-grained questions. Second, to ensure interpretable preference learning, we propose to use the classical linear weighting mechanism on the question outcomes. It enables intuitive visualization of each question’s impact.

#### 2 Related Work

Reinforcement Learning from Human Feedback (RLHF) (Stiennon et al. 2020; Nakano et al. 2021; Ouyang et al. 2022) refers to optimizing models with reinforcement learning based on human feedback, which is also explored in image and video generation.

To apply it as a reward model for visual generation models, we propose a multi-dimensional consistent strategy during preference optimization. The goal of this strategy is to mitigate unintended and unquantifiable biases. Specifically, a pair of visual samples is used for preference optimization (e.g., DPO (Wallace et al. 2024)) only if one sample is consistently preferred over the other across all dimensions.

Preference Learning for Visual Generation. There are many works learning from human preferences, which collect human annotation for text-to-image (Xu et al. 2023; Kirstain

To summarize, we present VisionReward as a general

Dataset Image Video #Samples #Dimensions #Fine-Grained #Annotation ImageReward (Xu et al. 2023) ✓ 9K - - 0.1M

Pick-a-Pic (Kirstain et al. 2023) ✓ 38K - - 0.6M HPDv2 (Wu et al. 2023) ✓ 430K - - 0.8M RichHF-18K (Liang et al. 2024) ✓ 18K 4 - 0.1M MPS (Zhang et al. 2024b) ✓ 608K 4 - 2.4M

VideoScore (He et al. 2024) ✓ 38K 5 - 0.2M VisionReward (Ours) ✓ ✓ 81K 18–20 61–64 5.0M

Table 1: Comparison of dataset of VisionReward and other datasets.

Main Object Object Pairing

Clarity Light Aes Detail

Richness

Light Acc

Symmetry

Clarity

Color

Letters

Color Brightness

Quality Fidelity

Composition

Background

Movement Reality

Composition

Composition

Alignment

Color Aes

Alignment

Safety

Alignment

Quality

###### Image

###### Video

Alignment

Safety

3 million QA pairs

###### 2 million QA pairs

Shape Begining

Object Dynamic

Light Distinction

Dynamic

Preservation

Emotion

Safety

Physics

Shape Throughout

Camera Dynamic

Safety

Light Aes

Stability

Fidelity

Camera Motion

Physics

Detail Refinement

Hands

Camera Stability

Smoothness

Detail Reality Body

Face

Movement Stability

Focus

- Figure 2: Illustration of fine-grained multi-dimensional design. (Left) For image: 5 dimensions, 18 sub-dimensions, and 61 binary questions. (Right) For video: 9 dimensions, 20 sub-dimensions, and 64 binary questions.

et al. 2023; Wu et al. 2023) and text-to-video (He et al. 2024). Note that existing approaches (Zhang et al. 2024b; Liang et al. 2024) have attempted to augment human annotations or expand dimensions of human preferences in visual generation. Different from them, VisionReward defines finegrained multi-dimensional human preferences with the goal of disentangling distinct factors to decouple human preferences, to build more accurate and interpretable RM.

RLHF for Visual Generation. For visual generation tasks, several works have explored RLHF, optimizing from the gradient (Xu et al. 2023; Wu et al. 2024) or using a policybased RL approach (Black et al. 2023; Fan et al. 2023; Clark et al. 2023). All these methods require a reward model (RM) to provide feedback for online learning. DiffusionDPO (Wallace et al. 2024) has proposed to optimize the diffusion model directly using human-labeled preference data. However, most RLHF methods face the issue of biasedoptimization. By employing a multi-dimensional method, VisionReward achieves robust RLHF.

Preliminary for Diffusion-DPO. Given a data distribution q(x0), Diffusion models (Sohl-Dickstein et al. 2015; Ho, Jain, and Abbeel 2020; Song et al. 2020) contains

forward process and reverse process. Forward process q(x1:T|x0) gradually add noise to the data x0 and reverse process pθ(x0:T) learns transitions to recover data. Training diffusion model can be performed by evidence lower bound (Kingma et al. 2021; Song et al. 2021):

0,ϵ∼N(0,I),t ∥ϵ − ϵθ (xt,t)∥22 , (1) with t ∼ U(0,T) and xt ∼ q (xt | x0).

LDM = Ex

Diffusion-DPO (Wallace et al. 2024) introduces direct preference optimization based on preference pairs. We denote the “win” and “lose” samples as xw0 ,xl0, and the objective is as follows:

L(θ) = −Et∼U(0,T),xw

t ∼q(xwt |xw0 ),xlt∼q(xlt|xl0)

log σ (−βTω (λt)( ∥ϵw − ϵθ (xwt ,t)∥22 − ∥ϵw − ϵref (xwt ,t)∥22

− ϵl − ϵθ xlt,t 22 − ϵl − ϵref xlt,t 22 . (2)

DPO is ordinarily based on overall preference, which may be biased. VisionReward enables Multi-Dimensional Preference Optimization to enhance it.

!

❄

!

1. Fine-Grained Visual Assessment

2. Interpretable Preference Learning

Multi-Dimensional Preference Optimization

| | |
|---|---|
| | |
| | |

###### Binary Question List

Multimodal LLM

[Figure 21]

Hierarchical Diagnostic Dimensions

Dim1: Stability

###### Linear Weight

[Figure 22]

- Q1: Is the movement smooth?
- Q2: Is the image quality stable?
- Q3: Is the camera stable? …

[Figure 23]

[Figure 24]

Stability

Composition

Fidelity

Smooth, Camera

Richness, Pairing

Detail, Human

###### … Dim9

Dim1 Dim2

Dim3

Safety

Quality

Alignment

###### Total Score

Violence, Porn

Color, Lighting

Text-Video/Img.

Multimodal LLM (Freeze)

[Figure 25]

Physics

Preservation

Dynamic

Dim1: Video1 > Video2 Dim2: Video1 > Video2 Dim3: Video1 > Video2 … Total: Video1 > Video2

Dim1: Video1 >> Video2 Dim2: Video1 < Video2 Dim3: Video1 < Video2 … Total: Video1 > Video2

Physical law

Shape preserv.

Object Motion

Predicted Answers

###### Predicted Answers

###### Binary Question & Answer List

- A1: Yes.
- A2: No.
- A3: Yes.

- A1: No.
- A2: Yes.
- A3: Yes. …

Dim1: Stability

###### Optimization (DPO) Optimization (DPO)

- Q1: Is the movement smooth?
- Q2: Is the image quality stable?
- Q3: Is the camera stable? …

- A1: Yes.
- A2: No.
- A3: Yes.

…

Dimensions

Dimensions

|0|1|1|1|0|0|…|1|
|---|---|---|---|---|---|---|---|

|1|0|1|1|0|1|…|0|
|---|---|---|---|---|---|---|---|

[Figure 26]

[Figure 27]

###### Linear Weight (Regression)

[Figure 28]

Preference Preference

Multimodal LLM (Fine-tune)

Our Strategy: MultiDimensional Consistent

Previous Strategy: Only Total Score

Score 1 Score 2

Human Preference

[Figure 29]

- Figure 3: Overall framework of VisionReward. 1) Fine-grained Visual Assessment: fine-tune multimodal LLM to perform binary visual question-answering through hierarchical dimensions. 2) Interpretable Preference Learning: utilize visual QA outputs to predict preferences through linear weighted summation. 3) Multi-Dimensional Preference Optimization: optimization strategy across multiple dimensions.

#### 3 VisionReward

##### 3.1 VisionReward Annotation

Fine-Grained Design. Human preferences are often a result of the interplay of multiple factors (Palmer, Schloss, and Sammartino 2013; Ibarra et al. 2017), necessitating a balance among various considerations. To deconstruct human preferences systematically, we develop a fine-grained multidimensional framework, as shown in Table 1 and Fig. 2. For each sub-dimension, we set options that vary gradually in degree, and decompose these options into a series of binary questions (Cf. Tables 15 to 18 in Appendix).

Dataset Preparation. For images, we sample images from multiple popular datasets, including ImageRewardDB (Xu et al. 2023), HPDv2 (Wu et al. 2023), and Pick-aPic (Kirstain et al. 2023), and obtain 48k images after filtering. For videos, we sample prompts from VidProM (Wang and Yang 2024). To ensure diversity of prompts, we use Rouge-L (Lin 2004) for initial filtering, follow UniFL (Zhang et al. 2024a) to perform a semanticbased filtering, and use ChatGPT (Achiam et al. 2023) for data cleaning, finally get 10k prompts. Then we use CogVideoX (Yang et al. 2024), VideoCrafter2 (Chen et al. 2024a) and OpenSora (Zheng et al. 2024) to generate 30k videos, sample from Panda-70M (Chen et al. 2024b) to get 3k real videos, leading to 33k videos for annotation. More details are provided in Appendix Section 6.1.

Annotation Management. To avoid bias of annotators, our annotation management includes professional management and standard document. Cooperating with a specialized company, we strictly conduct annotation training for annotators, select qualified annotators, and perform quality inspection of annotation results. Our annotation document

gives clear definitions and provides more than 10 examples for each judgment, to align the standard among annotators. Due to these efforts, the consistency of annotators in the binary results reaches 89.29% (images) and 89.33% (videos).

Annotation Analysis. Through specialized annotation, we obtain an image dataset containing 48k images and 3 million question-answer pairs, while a video dataset with 33k videos and 2 million pairs. More statistical analysis of the annotation results is in Appendix Section 6.2.

##### 3.2 VisionReward Training

The complete training process of VisionReward and its application methodology during preference optimization are illustrated in Fig. 3.

Fine-grained Visual Assessment. Specifically, we use CogVLM2 (Hong et al. 2024b) as the base model for image understanding, and CogVLM2-Video (Hong et al. 2024b) as the base model for video understanding. In terms of data, we have obtained millions of annotated binary questionanswering pairs. Initially, we performed a balanced sampling on each binary question by addressing the imbalance between positive and negative examples, ensuring a roughly equal number of positive and negative instances associated with each binary question. Then we use balanced instruction tuning dataset consisting of binary questions to finetune base VLM.

Interpretable Preference Learning. After trained on finegrained dataset, VisionReward can be adopt to give a series of binary response answers (“yes” or “no”) {Ai}Ni=1, where N represents the number of binary questions. We define reward of every binary question as {xi}Ni=1:

xi = 1[Ai = “yes”]. (3)

❄

❄

We construct a feature vector X = (x1,...,xN), and use a set of linear weights W = (w1,...,wN) to obtain the final reward R:

N

wi1[Ai = “yes”]. (4) In order to learn linear weights W, we collect human

R =

i=1

preferences for pairs of {(Xi,Xj)}. Specifically, we compute the feature difference for each pair, given by ∆X =

Xi − Xj, and the corresponding label is assigned as y = 1 or y = 0 depending on the human preference. We then perform logistic regression y = ∆XWT to learn linear weights W:

L(W) = − E y log σ(∆XWT)

+(1 − y)log 1 − σ(∆XWT) . (5)

By calculating dimension-specific scores through intradimensional weighting, VisionReward facilitates multidimensional preference prediction. We note dimensions as {dimk}Kk=1 where dimk contains questions belonging to the dimension. Then we define reward for certain dimension as:

R(dimk) =

wi1[Ai = “yes”]. (6)

i∈dimk

##### 3.3 Multi-Dimensional Preference Optimization

To empirically validate the model’s capacity, we leverage Direct Preference Optimization (DPO) (Wallace et al. 2024) for Diffusion Models in our experiments, where VisionReward generates multi-dimensional preference pairs to guide the optimization process while maintaining interdimensional balance.

Challenges. We replicate the Diffusion-DPO training procedure using SDXL (Podell et al. 2023) on the Pick-aPic (Kirstain et al. 2023) dataset, employing VisionReward for comprehensive data analysis and model evaluation. As demonstrated in Fig. 4, both the preference data and optimized model exhibit biases across several fine-grained dimensions. These findings not only underscore VisionReward’s capability for fine-grained analysis but also emphasize the necessity for optimization approaches that account for multi-dimensional representation.

MPO: Insight and Solution. Compared to ordinary DPO method which select pairs using overall preference, we propose MPO-enhanced DPO that take account fine-grained multi-dimensional preference. For reward of two samples Ri and Rj, we define Ri as dominating Rj if Ri(dimk) ≥ Rj(dimk) holds for every dimension dimk. The key differences between the MPO strategy and standard DPO are:

- • Ordinary DPO: During DPO optimization, we directly select the pair based on the total reward R.
- • MPO-enhanced DPO: MPO strategy introduces an additional constraint: we only select pairs that Ri dominates Rj, then proceed with standard DPO.

We analyze the effects of MPO in Section 4.3. The MPO strategy can also be applied to other algorithms, which we leave for future exploration.

alignment +8.22

rich +6.69

clear +5.99

realistic +5.17

detail +4.37

emotion +3.99

main-subject +3.70

light-distinct +3.38

coordinated +2.98

light-aes +2.38

background +0.82

color-bright +0.46

- -2.65 body
- -1.69 face
- -1.62 hands
- -0.38 color-aes

-6.24 symmetry

-9.87 safe

10 0 10 Performance(%)

(a) Data analysis.

clear +21.05

coordinated +11.22

color-bright +10.96

main-subject +8.88

light-aes +7.64

light-distinct +7.26

hands +6.75

detail +4.20

realistic +3.89

alignment +2.61

body +1.63

emotion +1.23

- -6.97 safe
- -3.56 rich
- -2.84 color-aes
- -2.36 background
- -2.33 face
- -1.44 symmetry

20 0 20 Performance(%)

(b) DPO analysis.

Figure 4: (a) We sample 10,000 human preference pairs from Pick-a-Pic dataset and analyze score deviations across 18 sub-dimensions (represented by the average yes-proportion of checklist questions within each sub-dimension). (b) We show score deviations for images generated by SDXL after Diffusion-DPO, using the same 10,000 prompts.

#### 4 Experiments

##### 4.1 VisionReward for Text-to-Vision Evaluation

Dataset & Training Setting. After balanced sampling, we obtain 40,743 images and corresponding 97,680 judgment questions for training, leaving 6,910 images for subsequent validation and test. For videos, we obtain 28,605 videos and corresponding 89,473 judgment questions for training, with 3,080 videos reserved.

To fine-tune CogVLM2 (Hong et al. 2024b), we set a batch size of 64, a learning rate of 1e-6, and train for 1,500 steps. For CogVLM-Video, we set a batch size of 64, a learning rate of 4e-6, and train for 1,500 steps.

To learn linear weights for preference prediction, we sample human preference pairs and perform logistic regression. For images, We sample 44k pairs (24k from HPDv2 (Wu et al. 2023) and 20k from ImageRewardDB (Xu et al.

- 2023)); and for videos, we sample prompts from VidProM (Wang and Yang 2024) and generate videos (using CogVideoX (Yang et al. 2024), VideoCrafter2 (Chen et al.
- 2024a) and OpenSora (Zheng et al. 2024)), getting 1,795 annotated video pairs with preference.

To establish a comprehensive evaluation benchmark for both image and video generation, we construct MonetBench, which contains separate test sets for images and videos, each consisting of 1,000 prompts. More details are introduced in Appendix Section 12.

Main Results: Preference Accuracy. Preference accuracy means the probability that a reward model has the same judgment as humans about which image is better. We use MonetBench to construct our test set for human preference,

Image Video HPDv2

Method

MonetBench GenAI-Bench MonetBench tau∗ diff∗∗ tau diff tau diff task-specific discriminative models

ImageReward (Xu et al. 2023) 74.0 48.8 56.5 48.4 72.1 55.8 58.4 PickScore (Kirstain et al. 2023) 79.8 49.8 57.6 52.4 75.4 57.7 61.6 HPSv2 (Wu et al. 2023) 83.3 48.4 55.6 49.3 73.0 59.3 62.5 MPS (Zhang et al. 2024b) 83.5 44.2 50.7 46.9 67.6 55.8 58.9 generative models

GPT-4o (Achiam et al. 2023) 77.5 38.9 52.7 41.8 54.3 45.7 48.3 Gemini (Team et al. 2024) 60.7 27.4 55.1 46.9 61.7 52.2 56.8 VQAScore (Lin et al. 2025) 69.7 49.4 56.5 45.2 68.0 56.1 59.5 VideoScore (He et al. 2024) 76.8 45.8 52.5 47.8 71.4 49.1 54.9 VisionReward (Ours) 81.7 51.8 59.5 51.8 74.4 64.0 72.1

Table 2: Preference accuracy on multiple dataset. Bold denotes the best score within the generative models, while underline signifies the best score among all categories. Tau∗ means taking account of ties (Deutsch, Foster, and Freitag 2023), and diff∗∗ means dropping ties in labels (we drop ties both in labels and responses for GPT-4o and Gemini in diff∗∗ because too many ties are given by them).

[Figure 30]

75

PreferenceAcc.(%)

SDXL w/o DPO Pick-a-pic HPSv2 CogVideoX w/o DPO VideoScore

70

VisionReward for SDXL

65

Before Weight Masking

After Weight Masking

VideoScore Baseline

60

VisionReward for CogVideoX

4 8 16 32 64

Number of Binary Questions

Figure 6: Human evaluation results of DPO using different reward models or preference datasets. We require five annotators to comprehensively evaluate two samples and select the better one. VisionReward achieve the best performance.

Figure 5: The accuracy of VisionReward on GenAI-Bench improves as the number of binary questions increases. After masking weights from full regression, VisionReward maintains high performance.

VisionReward can effectively address this issue after finegrained visual learning.

using SDXL (Podell et al. 2023) to generate images and CogVideoX (Yang et al. 2024) / VideoCrafter2 (Chen et al. 2024a) / OpenSora (Zheng et al. 2024) to generate videos, resulting in 500 pairs for image and 1,000 pairs for video. We employ annotators to assess the generated images using a preference rating scale from 1 to 5 (with 3 indicating no preference). The average preference score is used as the final preference label. We also take HPDv2 (Wu et al. 2023) and GenAI-Bench (Jiang et al. 2024) as test set.

Ablation Study: Scalability of Question Scale. Fig. 5 shows that accuracy of preference prediction exhibits significant improvement as the question scale increases, and masking minimal weights maintains performance (Details in Appendix Section 7). Scalability validates the effects of decomposing preferences via fine-grained questions. We do more analysis of VisionReward in Appendix Section 7 and analysis of fine-grained results in Section 10.

Table 2 shows that VisionReward obtains state-of-thestate results in multiple datasets. Notably, in video evaluation, image reward models demonstrate competitive performance when the video duration is within 2 seconds (GenAIBench). However, when the video duration reaches 6 seconds (MonetBench), only VisionReward is capable of accurately predicting human preference, being twice (22.1% over random) as high as the best (12.5% over random) among other methods. This indicates that dynamic information in longer videos poses a challenge for RMs, while

##### 4.2 VisionReward for Preference Optimization

To evaluate the efficacy of VisionReward in preference optimization for visual generative systems, we conduct a series of comparative experiments against current state-of-theart reward models and established preference datasets. We use SDXL as text-to-image base model and CogVideoX as text-to-video base model. The empirical results presented in Fig. 6 demonstrate VisionReward’s superior performance,

achieving statistically significant improvements in human preference metrics over competing approaches.

This section focuses on preference optimization for textto-video using different reward models. Details of text-toimage are provided in Appendix Section 8.2.

Dataset & Training Settings. For our backbone model, we select CogVideoX-2B. The training prompts are sampled from VidProM (Wang and Yang 2024) (details in Appendix Section 8.1). To adapt these prompts for video generation, we have optimized them following guidelines from CogVideoX (Yang et al. 2024), which results in roughly 22,000 samples. We generate 4 videos for each prompt, and use VisionReward to score these videos and apply the MPO strategy to select approximately 9,400 effective preference pairs. In all our experiments, we maintain a batch size of 32, a learning rate of 5e-6, and employ 100 warmup steps followed by linear decay. We set the DPO parameter β to 500. The MPO training process spans around 500 steps, equivalent to about 2 epochs. During training, we save a checkpoint every 40 steps and use a validation set split from the training set to pick the checkpoint with the highest reward.

Evaluation Settings. To comprehensively assess the MPO models, we have conducted both automatic and human evaluations. The automatic evaluation is conducted across various benchmarks, including VBench (Huang et al. 2024) and our Video-MonetBench. For VBench, we focus on commonly reported key metrics, including Human Action, Scene, Multiple Objects, and Appearance Style. In all these experiments, we utilize prompt optimization recommended in CogVideoX. Our baseline comparisons include the original CogVideoX-2B and DPO with VideoScore.

Human

Multiple

Appear.

Methods

Scene

Action

Objects

Style

Original 98.20 55.60 68.43 24.20 VideoScore 97.60 56.25 68.66 23.96 VisionReward 98.40 57.57 71.54 24.02

Table 3: Evaluation results on VBench.

Experimental Results. The main results are shown in Table 3. When compared to the original CogVideoX-2B, optimization with VisionReward significantly enhances model performance across these benchmarks. In contrast, optimization with VideoScore tends to degrade performance. The empirical evidence substantiates VisionReward’s advanced capacity for multi-dimensional optimization. (Case study in Appendix Section 8.3.)

##### 4.3 Ablation Study of MPO

To comprehensively illustrate how MPO addresses the factor optimization bias inherent in DPO, We conduct experiments based on CogVideoX-5B. We set the threshold of the total score to 0.8 for DPO and 0.6 for MPO, ensuring that the number of pairs obtained through all three strategies is 5k. We use a batch size of 64, a learning rate of 2e6, the DPO parameter β of 500, and the training steps of

300. VisionReward is employed to evaluate scores across various dimensions, with the detailed results presented in Table 4. Through the implementation of the MPO strategy, CogVideoX is optimized in such a manner that it avoids the degradation of certain factors (e.g., alignment), thereby achieving improved trade-offs, such as maintaining good preservation while avoiding excessively slow dynamic changes. The empirical evidence further substantiates VisionReward’s capacity for algorithm-agnostic preference alignment, as evidenced by comparative testing with other approaches like MaPO (Hong et al. 2024a).

Method Align. Quality Dynamic Physics Preserv. Overall

Original 1.733 0.660 0.053 0.344 0.653 4.303 DPO 1.697 0.680 0.034 0.356 0.741 4.515 DPO w/ MPO 1.766 0.688 0.042 0.356 0.721 4.573 MaPO 1.736 0.660 0.052 0.345 0.645 4.295 MaPO w/ MPO 1.737 0.656 0.055 0.349 0.649 4.321

- Table 4: Ablation Study of MPO strategy. Scores are given by VisionReward on MonetBench.

For efficiency discussion, the experimental results in Table 5 reveal the comparative effectiveness of the MPO and DPO methods. We analyze the pairs selected in perspective of “Ri dominating Rj” mentioned in Section 3.3. These results suggest that MPO outperforms the DPO approach in terms of both efficiency and effectiveness. These findings highlight promising directions for future research in developing novel optimization algorithms and adapting VisionReward for multi-dimensional optimization.

Method #Dom. #Not-Dom. Reward Original - - 4.303 DPO w/o MPO 3814 1456 4.515 (+0.212) DPO w/ MPO 5028 0 4.573 (+0.270) ∆ (vs w/o MPO) +31.8% -100% +27.4%

- Table 5: Comparison of MPO and DPO on MonetBench. “#Dom.” means the number of pairs that match the rule of “Ri dominating Rj”, while “#Not-Dom.” pairs not match.

#### 5 Conclusion

We introduce VisionReward, a reward model for visual generation, which is fine-grained and multi-dimensional. By enabling Vision-Language Model (VLM) to perform binary assessments and applying linear summation with weighting coefficients derived from preference learning, VisionReward achieves highly accurate and interpretable. For visual generative optimization, VisionReward surpasses other reward models and enable multi-dimensional strategy.

#### Acknowledgments

This research was supported by Natural Science Foundation of China (NSFC) No. 62276148, NSFC No. 62495063. The authors would like to thank Z.AI for sponsoring the computation resources used in this work.

#### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Betker, J.; Goh, G.; Jing, L.; Brooks, T.; Wang, J.; Li, L.; Ouyang, L.; Zhuang, J.; Lee, J.; et al. 2023. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3): 8.

Black, K.; Janner, M.; Du, Y.; Kostrikov, I.; and Levine, S. 2023. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301.

Chen, H.; Zhang, Y.; Cun, X.; Xia, M.; Wang, X.; Weng, C.; and Shan, Y. 2024a. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7310–7320.

Chen, T.-S.; Siarohin, A.; Menapace, W.; Deyneka, E.; Chao, H.-w.; Jeon, B. E.; Fang, Y.; Lee, H.-Y.; Ren, J.; Yang, M.-H.; et al. 2024b. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13320–13331.

Chen, Z.; Du, Y.; Wen, Z.; Zhou, Y.; Cui, C.; Weng, Z.; Tu, H.; Wang, C.; Tong, Z.; Huang, Q.; et al. 2024c. MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation? arXiv preprint arXiv:2407.04842.

Clark, K.; Vicol, P.; Swersky, K.; and Fleet, D. J. 2023. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400.

Deutsch, D.; Foster, G.; and Freitag, M. 2023. Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 12914–12929.

Ding, M.; Yang, Z.; Hong, W.; Zheng, W.; Zhou, C.; Yin, D.; Lin, J.; Zou, X.; Shao, Z.; Yang, H.; et al. 2021. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34: 19822–19835.

Fan, Y.; Watkins, O.; Du, Y.; Liu, H.; Ryu, M.; Boutilier, C.; Abbeel, P.; Ghavamzadeh, M.; Lee, K.; and Lee, K. 2023. DPOK: reinforcement learning for fine-tuning textto-image diffusion models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, 79858–79885.

GLM, T. 2024. ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools. arXiv:2406.12793.

He, X.; Jiang, D.; Zhang, G.; Ku, M.; Soni, A.; Siu, S.; Chen, H.; Chandra, A.; Jiang, Z.; Arulraj, A.; et al. 2024. VideoScore: Building Automatic Metrics to Simulate Finegrained Human Feedback for Video Generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2105–2123.

Ho, J.; Chan, W.; Saharia, C.; Whang, J.; Gao, R.; Gritsenko, A.; et al. 2022. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.

Hong, J.; Paul, S.; Lee, N.; Rasul, K.; Thorne, J.; and Jeong, J. 2024a. Margin-aware preference optimization for aligning diffusion models without reference. arXiv preprint arXiv:2406.06424.

Hong, W.; Ding, M.; Zheng, W.; Liu, X.; and Tang, J. 2022. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868.

Hong, W.; Wang, W.; Ding, M.; Yu, W.; Lv, Q.; Wang, Y.; Cheng, Y.; Huang, S.; Ji, J.; Xue, Z.; et al. 2024b. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500.

Huang, Z.; He, Y.; Yu, J.; Zhang, F.; Si, C.; Jiang, Y.; Zhang,

- Y.; Wu, T.; Jin, Q.; Chanpaisit, N.; et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21807–21818.

Ibarra, F. F.; Kardan, O.; Hunter, M. R.; Kotabe, H. P.; Meyer, F. A.; and Berman, M. G. 2017. Image feature types and their predictions of aesthetic preference and naturalness. Frontiers in Psychology, 8: 632.

Jiang, D.; Ku, M.; Li, T.; Ni, Y.; Sun, S.; Fan, R.; and Chen, W. 2024. GenAI Arena: An Open Evaluation Platform for Generative Models. arXiv preprint arXiv:2406.04485.

Kingma, D.; Salimans, T.; Poole, B.; and Ho, J. 2021. Variational diffusion models. Advances in neural information processing systems, 34: 21696–21707.

Kirstain, Y.; Polyak, A.; Singer, U.; Matiana, S.; Penna, J.; and Levy, O. 2023. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36: 36652–36663.

Li, C.; Zhang, Z.; Wu, H.; Sun, W.; Min, X.; Liu, X.; Zhai, G.; and Lin, W. 2023. Agiqa-3k: An open database for aigenerated image quality assessment. IEEE Transactions on Circuits and Systems for Video Technology, 34(8): 6833– 6846.

Li, F.; Zhang, R.; Zhang, H.; Zhang, Y.; Li, B.; Li, W.; Ma,

- Z.; and Li, C. 2024. Llava-next-interleave: Tackling multiimage, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895.

Liang, Y.; He, J.; Li, G.; Li, P.; Klimovskiy, A.; Carolan, N.; Sun, J.; Pont-Tuset, J.; Young, S.; Yang, F.; et al. 2024. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19401–19411.

Lin, C.-Y. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, 74–81. Barcelona, Spain: Association for Computational Linguistics.

Lin, Z.; Pathak, D.; Li, B.; Li, J.; Xia, X.; Neubig, G.; Zhang, P.; and Ramanan, D. 2025. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, 366–384. Springer.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning.

Nakano, R.; Hilton, J.; Balaji, S.; Wu, J.; Ouyang, L.; et al. 2021. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.

Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744.

Palmer, S. E.; Schloss, K. B.; and Sammartino, J. 2013. Visual aesthetics and human preference. Annual review of psychology, 64(1): 77–107.

Podell, D.; English, Z.; Lacey, K.; Blattmann, A.; Dockhorn, T.; M¨uller, J.; Penna, J.; and Rombach, R. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PMLR.

Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021. Zero-shot text-toimage generation. In International conference on machine learning, 8821–8831. Pmlr.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10674– 10685. IEEE Computer Society.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, K.; et al. 2022. Photorealistic textto-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479–36494.

Schuhmann, C.; Beaumont, R.; Vencu, R.; Gordon, C. W.; Wightman, R.; et al. 2022. LAION-5B: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Sohl-Dickstein, J.; Weiss, E.; Maheswaranathan, N.; and Ganguli, S. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, 2256–2265. PMLR.

Song, Y.; Durkan, C.; Murray, I.; and Ermon, S. 2021. Maximum likelihood training of score-based diffusion models. Advances in neural information processing systems, 34: 1415–1428.

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2020. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.

Stiennon, N.; Ouyang, L.; Wu, J.; Ziegler, D.; Lowe, R.; Voss, C.; Radford, A.; Amodei, D.; and Christiano, P. F. 2020. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33: 3008–3021.

Team, G.; Georgiev, P.; Lei, V. I.; Burnell, R.; Bai, L.; et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint

- arXiv:2403.05530.

Villegas, R.; Babaeizadeh, M.; Kindermans, P.-J.; Moraldo, H.; Zhang, H.; Saffar, M. T.; Castro, S.; Kunze, J.; and Erhan, D. 2022. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations.

Wallace, B.; Dang, M.; Rafailov, R.; Zhou, L.; Lou, A.; Purushwalkam, S.; Ermon, S.; Xiong, C.; Joty, S.; and Naik, N. 2024. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8228–8238.

Wang, W.; and Yang, Y. 2024. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. arXiv preprint arXiv:2403.06098.

Wu, H.; Mao, J.; Zhang, Y.; Jiang, Y.; Li, L.; Sun, W.; and Ma, W.-Y. 2019. Unified Visual-Semantic Embeddings: Bridging Vision and Language With Structured Meaning Representations. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6602–6611.

Wu, X.; Hao, Y.; Sun, K.; Chen, Y.; Zhu, F.; Zhao, R.; and Li, H. 2023. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341.

Wu, X.; Hao, Y.; Zhang, M.; Sun, K.; Huang, Z.; Song, G.; Liu, Y.; and Li, H. 2024. Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models. arXiv preprint arXiv:2405.00760.

Xu, J.; Liu, X.; Wu, Y.; Tong, Y.; Li, Q.; Ding, M.; Tang, J.; and Dong, Y. 2023. ImageReward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, 15903–15935.

Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang, Y.; Hong, W.; Zhang, X.; Feng, G.; et al. 2024. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072.

Zhang, J.; Wu, J.; Ren, Y.; Xia, X.; Kuang, H.; Xie, P.; Li, J.; Xiao, X.; Zheng, M.; Fu, L.; and Li, G. 2024a. UniFL: Improve Stable Diffusion via Unified Feedback Learning.

- arXiv:2404.05595.

Zhang, S.; Wang, B.; Wu, J.; Li, Y.; Gao, T.; Zhang, D.; and Wang, Z. 2024b. Learning multi-dimensional human preference for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8018–8027.

Zheng, Z.; Peng, X.; Yang, T.; Shen, C.; Li, S.; Liu, H.; Zhou, Y.; Li, T.; and You, Y. 2024. Open-Sora: Democratizing Efficient Video Production for All.

#### Appendix 6 More Details of Annotation

SYSTEM Assume you are a model responsible for refining and polishing English expressions. You will receive an English prompt that may contain abbreviations or non-standard expressions. Your task is to standardize the expressions, and your output must be in pure English without any non-English characters. If the prompt is fragmented or difficult to understand, discard it by outputting “F”. Your output must strictly follow the format: each sentence should be on a single line, either as the rewritten prompt or a standalone “F”.

##### 6.1 Details of Annotation Design

To ensure the diversity of our annotation, we collect our annotated data from various sources, as presented in Table 6. The prompt and example for data cleaning is shown in Table 8. Table 7 illustrates our preference dimensions and the checklist count. We identify 5 dimensions for text-to-image generation and expand to 9 dimensions for text-to-video.

Type Source #Samples #Checklist

ImageRewardDB (Xu et al. 2023) 16K 1M Pick-a-Pic (Kirstain et al. 2023) 16K 1M HPDv2 (Wu et al. 2023) 16K 1M

Image

USER Here is the prompt you have received:

CogVideoX (Yang et al. 2024) 10K 0.6M Open-Sora (Zheng et al. 2024) 10K 0.6M

[[PROMPT]]

Video

VideoCrafter2 (Chen et al. 2024a) 10K 0.6M Panda-70M (Chen et al. 2024b) 3K 0.2M

INPUT Soft rays of light through the many different types of trees inside a forest, sunrise, misty, photorealistic, ground level, -neg &quot;no large bodies of water&quot; -ar 16:9 4K, -ar 16:9

Table 6: Statistics of source data and annotation.

#Sub-dimension #Checklist Image Video Image Video

OUTPUT The soft rays of light filter through the myriad types of trees within the forest at sunrise, creating a misty, photorealistic scene from ground level. Exclude any large bodies of water. The aspect ratio should be 16:9 in 4K resolution. Aspect ratio: 16:9.

Dimension

Alignment 1 1 1 4

Composition 5 1 13 2 Quality 5 4 14 14 Fidelity 5 3 25 9

Safety&Emotion 2 1 8 4

Table 8: Prompt template and example for prompt cleaning.

Stability - 5 - 12 Dynamic - 2 - 8

Physics - 1 - 4 Preservation - 2 - 7

reduced. To assist the model in learning the features of each sub-dimension, we can impose a quantitative limit on the predominant options.

Total 18 20 61 64

• Certain sub-dimensions, such as the presence of hands, require a mask when predicting human preferences. This means that the sub-dimension should only be evaluated when the image indicates the presence of hands. We also annotate the sub-dimensions that require a mask and record the relevant counts.

Table 7: Taxonomy of annotation for VisionReward.

##### 6.2 Statistics of Annotation Result

Fig. 7 is the statistical results of the labeled data for images and videos. When compiling the statistics, higher labels indicate better performance for the image or video subdimension, while a label of 0 indicates neutrality. For video data, the original labels only had positive values and the neutral value was inconsistent. Hence, a neutral value was determined for each sub-dimension, and the original labels were adjusted by subtracting this neutral value to make 0 represent neutrality. In sub-dimensions such as Background, Face, and Hand, there might be cases where these elements are not present in the image or video. In such instances, ”Not Contain” is treated as a separate category for statistical purposes.

#### 7 More Details and Results of VisionReward

VisionReward comprises two steps: visual judgment and linear regression.

Visual Judgment Process. As the options for each subdimension are progressive, for any given option corresponding to a judgment question in the checklist, samples with an option greater than or equal to the given one are considered positive examples for the judgment question, whereas samples with an option less than the given one are considered negative. To balance the number of positive and negative examples for each binary question, we screen out any excess positive and negative examples for each question, ensuring that the number of positive and negative examples used for training is balanced. For alignment, we use different

There are two main characteristics to note.

• For most sub-dimensions, the distribution of options roughly follows a normal distribution, with the majority being ordinary, and the quantities of instances with extreme characteristics, either very good or very bad, are

Annotation

Annotation

+2 +1 0 -1 -2 -3 Not Contain

+2 +1 0 -1 -2 -3 -4 Not Contain

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

Alignment

Adjective

Composition

Collocation

Color

Place

Lighting Accurate

Richness

Lighting Aes

Background

Clear

Detail Refinement

Sharpness

Movement Reality

Color

Letters

Sub-Dimension

Sub-Dimension

Color Aes

Safety

Shadow Degree

Movement smoothness

Shadow Aes

Image quality stability

Emotion

Focus

Camera movement

Detail Fineness

Camera stablity

Detail Facticity

Shape at beginning

Body Correctness

Shape throughout

Face

Object Motion dynamic

Hand

Camera motion dynamic

Safe

Physics law

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Proportion

Proportion

(a) Text-to-image

(b) Text-to-video

Figure 7: Annotation statistics of different sub-dimensions.

methods on images and videos. For images, we use VQAScore (Lin et al. 2025) as an alignment judgment. For videos, we train five levels of judgment for VisionReward.

Main Results: Accuracy on Judgment. To evaluate the effectiveness of judgment learning, we construct a visual quality QA set to assess the visual assessment capabilities of VisionReward compared to other VLMs. We collect 1,364 test cases for images involving 14 types of questions across 4 dimensions, and additionally 1,308 cases covering 8 types of questions across 4 dimensions related to dynamic content in videos. To ensure the generality of these questions, we combine adjacent degrees in the checklists under each subdimension, enhancing distinctiveness and minimizing incidental subjectivity. Table 10 demonstrates VisionReward’s superiority in judging visual quality over existing generalized multimodal LLMs, which remains a challenge for generalized multimodal LLMs.

Composition Quality Fidelity Safety 97.9 98.2 98.3 99.1 Stability Dynamic Physics Preservation 97.4 99.9 88.2 99.8

- Table 9: Consistency of VisionReward in each dimension.

Main Results: Consistency on Judgment. As questions corresponding to each sub-dimension assess varying degrees of a particular factor, it’s important to measure con-

sistency of VisionReward across multiple questions of the same sub-dimension. Consistency measures the likelihood that the model provides consistent responses across a series of judgments concerning this factor. Table 9 shows that VisionReward has high consistency (more than 97 %) in most (7 of 8) dimensions.

Algorithm 1: Iterative Regression with Weight Masking

Input: Dataset of human preferences D = {(Xi, Xj, y)}, where each pair (Xi, Xj) represents feature vectors of binary responses and y ∈ {0, 1} is the human preference label

- 1: Initialization: Initialize linear weights w = [w1, . . . , wn]
- 2: Initialize convergence criterion diff ← ∞
- 3: while diff > ϵ do
- 4: wold ← w
- 5: for each (Xi, Xj, y) in D do
- 6: ∆X ← Xi − Xj
- 7: yˆ ← σ(∆XTw)
- 8: ∇wLoss = (ˆy − y)∆X
- 9: w ← w − α∇wLoss
- 10: end for
- 11: Mask negative weights w ← w ⊙ (w > 0)
- 12: diff ← ||w − wold||
- 13: end while Output: Trained weights w

Weight Masking. In the linear regression step, we learn the correlation between human preferences and the results of visual judgment. In our design, if the result of the visual judg-

Image Video

Method

Composition Quality Fidelity Safety Stability Dynamic Physics Preservation

LLaVa∗ 59.9 65.7 59.8 64.4 52.5 53.8 50.6 47.5 CogVLM2 (Hong et al. 2024b) 65.8 67.1 53.1 74.7 49.3 57.1 51.2 47.8 GPT-4o (Achiam et al. 2023) 73.1 62.7 61.9 70.1 57.9 69.1 62.4 58.8 Gemini (Team et al. 2024) 69.4 59.9 59.7 74.9 58.1 71.1 58.1 59.6 VisionReward (Ours) 78.8 81.1 80.9 83.9 64.8 75.4 68.1 72.0

- Table 10: Accuracy of VisionReward and other vision-language models (VLMs) on vision quality questions constructed from our annotation. ∗We test LLaVA-v1.5-7B (Liu et al. 2023) for image and LLava-Next-Video-34B (Li et al. 2024) for video.

ment is “yes”, human preference improves. We examine the correlation between human preference and each judgment result, and the numerical results indicate a positive correlation. However, in linear regression, we observe that some coefficients corresponding to the judgment results were negative. This is because there are correlations among the judgment results themselves. To enhance the robustness of the regression outcomes, we employ an iterative masking algorithm during the regression.

Ablation Study: Impact of Training Set Size. We conduct experiments with varying sizes of the training set to investigate its influence on regression performance. As demonstrated in Table 11, the accuracy improves monotonically as the training set size increases up to 4,000 samples, beyond which the accuracy stabilizes.

Size 200 500 1k 2k 4k 8k Acc. 77.6 80.3 80.6 80.9 81.3 81.3

- Table 11: Accuracy on HPDv2-test for different sizes of train set.

ages per prompt using SDXL (this procedure theoretically produces 1.76M text-image pairs). Employing the MPO algorithm, we obtain 760k dominant pairs with 63,069 unique prompts. For comparison, we use HPSv2 with a threshold of 0.0015, getting 770k pairs with 63,107 unique prompts from the same source. We also compare with human annotated pairs, sampling 780k human preference pairs with 57,674 unique prompts from Pick-a-Pic v2 dataset.

We maintain consistent training parameters and dataset sizes across all experiments to ensure fair comparison. For all three experiments, we used an effective batch size of 256 (with GAS set to 4 and train batch size set to 1), set β to 5000, and a learning rate of 5e-9 (before scaling). We employ a constant warmup strategy with 100 steps and the training is conducted over 3,000 steps (approximately 1 epoch).

Evaluation Settings. We conducted both automatic and human evaluation on DrawBench (Saharia et al. 2022). Automatic evaluation includes multiple metrics such as human preference RMs, CLIP (Radford et al. 2021) and LAIONAesthetic (Schuhmann et al. 2022).

#### 8 More Details and Results of MPO

- 8.1 Details of Prompt for MPO We employ the prompt filtering approach proposed by UniFL (Zhang et al. 2024a) to curate our dataset. This strategy comprises two pivotal steps:

- • Semantic-Based Filtering: Utilizing an existing scene graph parser (Wu et al. 2019), we evaluate the semantic richness of prompts by analyzing the number of subjective and objective relationships. Prompts with fewer than one meaningful relationship are filtered out to reduce noise data.
- • Cosine Similarity-Based Selection: Following the semantic filtering process, we apply a cosine similaritybased iterative selection mechanism. By maintaining a maximum similarity threshold of 0.8 between any two prompts, we ensure dataset diversity and effectively eliminate redundant entries.

- 8.2 VisionReward for Text-to-Image Optimization

Dataset & Training Settings. We strategically sample 63,165 prompts from existing datasets and generate 8 im-

Experimental Results. Main results are demonstrated in Table 12 and Table 13. VisionReward gets leading results across multiple machine metrics and achieves significant improvements across all four dimensions of VisionReward.

Methods CLIP Aes HPSv2 PickScore Original 0.273 5.463 0.282 22.25 Pick-a-Pic 0.279 5.511 0.286 22.45 HPSv2 0.277 5.599 0.292 22.58 VisionReward 0.279 5.612 0.289 22.61

Table 12: Evaluation results of multiple metrics on DrawBench.

Methods Composition Quality Fidelity Safety Original 0.755 0.550 0.009 -0.008 Pick-a-Pic 0.765 0.588 0.009 -0.009 HPSv2 0.874 0.630 0.010 -0.004 VisionReward 0.894 0.670 0.017 -0.001

Table 13: Evaluation results on MonetBench.

##### 8.3 More Results of MPO

Case Study. Fig. 8 shows MPO cases for text-to-image, while Fig. 9 and Fig. 10 show MPO cases for text-to-video. MPO fine-tuned model surpasses the original model in multiple aspects and also outperforms other scoring methods.

Training Curve. Fig. 11 shows the variation of the dimensional scores during the MPO process with respect to the number of training samples. The results demonstrate that the MPO method enables the model to avoid trade-offs during training, thereby achieving simultaneous improvements across various sub-dimensions. In contrast, the DPO (Wallace et al. 2024) method fails to achieve this level of concurrent enhancement.

Ablation Study: Different Strategy for MPO. In using the MPO strategy, we define the way in which Ri dominates Rj (given two images xi and xj) to select pairs. The definition of “dominate” includes at least three methods for different reward objective: score of each dimension, score of each sub-dimension, and score of each binary question. To investigate the impact of different definitions of “dominate”, we conduct experiments based on CogVideoX-5B. Specifically, we employ three different strategies, setting the threshold of the total score to 0.6, 0.5, and 0.4 respectively, ensuring that the number of pairs obtained through all three strategies is 5k. We use a batch size of 64, a learning rate of 2e-6, the DPO parameter β of 500, and the training steps of 300. After training, we compare the evaluation results of VisionReward on MonetBench. Table 14 shows that using the score for each dimension as objective yields the best results.

Methods Baseline Dimension Sub-dimension Question Reward 4.303 4.573 4.539 4.514

- Table 14: Score of VisionReward after different strategies of MPO. Dimension: “dominate” based on score of each dimension. Sub-dimension: “dominate” based on score of each sub-dimension. Question: “dominate” based on score of each binary question.

9 Details of Fine-Grained Questions

- Table 15 and Table 16 show the annotation taxonomy of

images, while Table 17 and Table 18 show for videos.

#### 10 More Results of Fine-Grained Design

Weight and Accuracy of Checklist. We curate separate test sets of images and videos from outside the training set to evaluate the accuracy of judgment questions. The test set comprises 1,209 images and 1,000 videos, respectively. We report the accuracy of judgment questions (Cf. Table 19 and Table 20 for text-to-image, Table 21 and Table 22 for textto-video). As a reference, we specifically record the linear weights obtained from linear regression on human preference data as well as the Spearman rank correlation coefficient between human preference and the results of each judgment question.

Correlation of Sub Dimensions. To mine the correlation between the sub-dimensions after preference decoupling, we show the correlation coefficients between the subdimensions in a heat map (Cf. Fig. 12).

#### 11 Limitations

This section outlines several limitations of the VisionReward framework that emerged during its development.

Supported Video Frame Rate and Length. While training datasets of VisionReward contain videos up to 6 seconds in duration at 4 frames per second (fps), this may be insufficient for evaluating the outputs of next-generation video generation models, which are capable of producing longer and more complex content. Therefore, extending the model’s capacity to handle longer video sequences is a critical direction for future work.

Leverage of Foundation Model. The model’s performance is inherently tied to the capabilities of its base VisionLanguage Model (VLM). Our current approach utilizes a question-answering (QA) mechanism to tap into the VLM’s foundational knowledge. However, with the rise of sophisticated reasoning models, a more effective avenue for enhancing our reward model would be to integrate explicit reasoning abilities. This represents a key area for future investigation to move beyond foundational understanding towards more complex evaluation.

Text Prompt: A woman that is standing near an open oven.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Text Prompt: A group of people in suits standing in a kitchen.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Text Prompt: A whale is reading a book about avoiding Japanese spears in an underwater library.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Text Prompt: A Lamborghini Countach in the Arizona desert, depicted in an oil painting.

[Figure 43]

Original DPO with Pick-a-Pic DPO with HPSv2 DPO with VisionReward

Figure 8: Qualitative result of MPO in text-to-image.

Text Prompt: The massive aircraft carrier battles towering waves, its metallic hull gleaming amid the ocean's furious roar.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Original

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

DPO with VideoScore

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

DPO with VisionReward (Ours)

Text Prompt: A sixteen-year-old boy, lying on his bed in a retro, black-and-white, medium closeup, appears serene and contemplative, with tousled hair and soft shadows.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Original

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

DPO with VideoScore

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Text Prompt: Amidst rolling hills and olive groves, an elegant young man in ﬁne linen approaches with a serene demeanor and a look of determination and humility.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Original

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

DPO with VideoScore

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

DPO with VisionReward (Ours)

Text Prompt: A sleek, futuristic spaceship, gleaming with a metallic sheen, has landed softly amidst the historic architecture of Berlin.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Original

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

DPO with VideoScore

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Overall

Composition

MPO

MPO

Hpsv2-DPO

Hpsv2-DPO

- 75

- 76

- 77

- 78

- 79

- 80

- 81

- 82

- 83

- 84

- 85

Pickapicv2-DPO

Pickapicv2-DPO

0 100 200 300 400 500 600 700

0 100 200 300 400 500 600 700

Samples (k)

Samples (k)

(a) Overall Score

(b) Composition Score

Fidelity

Alignment

76.0

52.6

MPO

MPO

Hpsv2-DPO

Hpsv2-DPO

75.5

52.4

Pickapicv2-DPO

Pickapicv2-DPO

75.0

52.2

74.5

52.0

74.0

51.8

73.5

51.6

73.0

51.4

72.5

51.2

0 100 200 300 400 500 600 700

0 100 200 300 400 500 600 700

Samples (k)

Samples (k)

(c) Fidelity Score

(d) Alignment Score

Quality

Safety&Emotion

- 83

- 84

- 85

- 86

- 87

- 88

- 89

- 90

MPO

MPO

- 44

- 45

- 46

- 47

- 48

- 49

Hpsv2-DPO

Hpsv2-DPO

Pickapicv2-DPO

Pickapicv2-DPO

0 100 200 300 400 500 600 700

0 100 200 300 400 500 600 700

Samples (k)

Samples (k)

(e) Quality Score

(f) Safety & Emotion Score

Figure 11: Variation of dimensional scores during the MPO process with respect to the number of training samples.

[Figure 104]

###### Figure 12: Correlation heatmap of text-to-video sub dimensions.

Dimension Sub-dimension Option Checklist

symmetrical Is the image symmetrical?

Composition Symmetry

ordinary Does the image avoid asymmetry?

asymmetrical

coordinated Are the objects well-coordinated?

Composition Object pairing

ordinary Does the image avoid poorly coordinated objects?

uncoordinated

prominent Is the main subject prominent?

Composition Main object

ordinary Does the image avoid an unclear main subject?

prominent

very rich Is the image very rich?

rich Is the image rich?

Composition Richness

ordinary Is the image not monotonous?

monotonous Is the image not empty?

empty

beautiful Is the background beautiful?

somewhat beautiful Is the background somewhat beautiful?

Composition Background

ordinary Is there a background?

no background

very clear Is the image very clear?

clear Is the image clear?

Quality Clarity

ordinary Does the image avoid being blurry?

blurry Does the image avoid being completely blurry?

completely blurry

bright Are the colors bright?

Quality Color Brightness

ordinary Are the colors not dark?

dark

beautiful colors Are the colors beautiful?

Quality Color Aesthetic

ordinary colors Are the colors not ugly?

ugly colors

very distinct Is the lighting and shadow very distinct?

distinct Is the lighting and shadow distinct?

Quality Lighting Distinction

ordinary Is there lighting and shadow?

no lighting

very beautiful Are the lighting and shadows very beautiful?

beautiful Are the lighting and shadows beautiful?

Quality Lighting Aesthetic

ordinary Is there lighting and shadow?

no lighting

Table 15: Annotation taxonomy and checklist details for text-to-image evaluation. (part 1)

Dimension Sub-dimension Option Checklist

realistic Are the image details realistic?

neutral Do the image details avoid being unrealistic?

Fidelity Detail reality

unrealistic Do the image details avoid being very unrealistic? very unrealistic Do the image details avoid being greatly unrealistic?

greatly unrealistic

very refined Are the image details very exquisite?

refined Are the image details exquisite?

ordinary Do the image details avoid being coarse?

Fidelity Detail refinement

rough Do the image details avoid being very coarse?

very rough Does the image avoid being hard to recognize?

indistinguishable Does the image avoid being fragmented?

fragmented

no errors Is the human body in the image completely correct?

neutral Does the human body in the image avoid errors?

some errors Does the human body in the image avoid obvious errors?

Fidelity Body

obvious errors Does the human body in the image avoid serious errors?

serious errors Is there a human body in the image?

no human figure

very beautiful Is the human face very beautiful?

beautiful Is the human face beautiful?

normal Does the human face avoid errors?

Fidelity Face

some errors Does the human face avoid serious errors?

serious errors Is there a human face in the image?

no human face

perfect Are the human hands perfect?

mostly correct Are the human hands essentially correct?

minor errors Do the human hands avoid obvious errors?

Fidelity Hands

obvious errors Do the human hands avoid serious errors?

serious errors Are there human hands in the image?

no human hands

very positive Can the image evoke a very positive emotional response?

positive Can the image evoke a positive emotional response?

Safety & Emotion Emotion

ordinary Does the image avoid evoking a negative emotional response?

negative Does the image avoid evoking a very negative emotional response?

very negative

safe Is the image completely safe?

neutral Is the image harmless?

Safety & Emotion Safety

potentially harmful Does the image avoid obvious harmfulness?

harmful Does the image avoid serious harmfulness?

very harmful

Table 16: Annotation taxonomy and checklist details for text-to-image evaluation. (part 2)

Dimension Sub-dimension Option Checklist

meet 100% Does the video meet all the requirements stated in the text ”[[prompt]]”?

meet 80%-100% Does the video meet most of the requirements stated in the text ”[[prompt]]”? meet 60%-80% Does the video meet some of the requirements stated in the text ”[[prompt]]”?

Alignment Alignment

meet 40%-60% Does the video not completely fail to meet the requirements stated in the text ”[[prompt]]”?

meet 0-40%

good Is the composition aesthetically pleasing?

Composition Composition

normal Does the composition have no obvious flaws?

bad

very beautiful Are the colors exceptionally beautiful?

beautiful Are the colors beautiful?

Quality Color

normal Are the colors not unattractive?

unattractive Are the colors not significantly unattractive?

very unattractive

good Is the lighting perfectly accurate?

normal Does the lighting have no obvious errors?

Quality Lighting Accurate

bad Is there any lighting present?

no lighting

very good Is the lighting exceptionally beautiful?

good Is the lighting beautiful?

Quality Lighting Aes

normal Is the lighting not unattractive?

bad

very clear Is it very clear?

clear Is it clear?

Quality Clear

normal Is it not blurry?

blurry Is it not completely blurry?

completely blurry

very refined Are the details very refined?

refined Are the details refined?

Fidelity Detail Refinement

normal Are the details not rough?

rough Are the details not significantly rough?

very rough

good Is the object’s movement completely realistic?

Fidelity Movement Reality

normal Does the object’s movement have no obvious realism issues?

bad

good Are all the letters correct?

normal Do the letters have no obvious errors?

Fidelity Letters

bad Are there any letters present?

no letter

100% safe Is the video content safe?

80%-100% safe Is the video content definitely free of harmful material?

Safety Safety

60%-80% safe Does the video content contain no harmful material?

40%-60% safe Does the video content contain no extremely harmful material?

0-40% safe

Table 17: Annotation taxonomy and checklist details for text-to-video evaluation. (part 1)

Dimension Sub-dimension Option Checklist

good Is the smoothness of the object’s movement good?

Stability Movement smoothness

normal Does the smoothness of the object’s movement have no obvious issues?

bad

very stable Is the image quality very stable?

stable Is the image quality stable?

Stability Image quality stability

normal Is the image quality not unstable?

unstable Is the image quality free of noticeable instability?

very unstable

good Is the focus aesthetically pleasing?

Stability Focus

normal Does the focus have no obvious flaws?

bad

good Is the camera movement aesthetically pleasing?

Stability Camera movement

normal Does the camera movement have no obvious flaws?

bad

stable Is the camera stable?

Stability Camera stability

normal Is the camera not unstable?

unstable

completely accurate Is the shape of the object at the beginning of the video completely accurate?

no errors Does the shape of the object at the beginning have no obvious errors?

Preservation Shape at beginning

not chaotic Is the shape of the object at the beginning not chaotic?

flawed

perfectly maintained Is the shape of the object perfectly maintained throughout the video?

no issues Does the shape of the object have no obvious issues throughout the video?

Preservation Shape throughout

normal Does the shape of the object generally have no major issues throughout the video?

not chaotic Is the shape of the object not chaotic throughout the video?

flawed

highly dynamic Is the object’s motion highly dynamic?

dynamic Is the object’s motion dynamic?

Dynamic Object Motion dynamic

normal Is the object’s motion not minimal?

not static Is the object’s motion not static?

static

highly dynamic Is the camera motion highly dynamic?

dynamic Is the camera motion dynamic?

Dynamic Camera motion dynamic

not minimal Is the camera motion not minimal?

not static Is the camera motion not static?

static

full compliance Does it fully comply with the laws of physics?

partial compliance Does it partially comply with the laws of physics?

Physics Physics law

no obvious violations Does it have no obvious violations of the laws of physics?

physical world Is the video content part of the physical world?

non-compliance

Table 18: Annotation taxonomy and checklist details for text-to-video evaluation. (part 2)

ID Checklist Acc ρ Weight

- 1 Is there a human body in the image? 93.13 0.090 mask

- 2 Is there a human face in the image? 96.20 0.110 mask

- 3 Are there human hands in the image? 93.30 0.022 mask

- 4 Is the image symmetrical? 79.98 0.104 0.069

- 5 Does the image avoid asymmetry? 71.30 0.236 0.102

- 6 Are the objects well-coordinated? 58.31 0.138 0.000

- 7 Does the image avoid poorly coordinated objects? 68.24 0.204 0.000

- 8 Is the main subject prominent? 86.27 0.210 0.131

- 9 Does the image avoid an unclear main subject? 77.75 0.258 0.070

- 10 Is the image very rich? 80.40 0.084 0.056

- 11 Is the image rich? 65.84 0.138 0.044

- 12 Is the image not monotonous? 77.01 0.271 0.211

- 13 Is the image not empty? 99.67 0.205 0.583

- 14 Is the background beautiful? 72.70 -0.019 0.000

- 15 Is the background somewhat beautiful? 67.26 0.021 0.000

- 16 Is there a background? 84.86 0.079 mask

- 17 Is the image very clear? 63.85 0.111 0.051

- 18 Is the image clear? 62.03 0.170 0.068

- 19 Does the image avoid being blurry? 88.92 0.284 0.065

- 20 Does the image avoid being completely blurry? 97.11 0.282 0.032

- 21 Are the colors bright? 63.69 0.098 0.076

- 22 Are the colors not dark? 82.88 0.141 0.077

- 23 Are the colors beautiful? 65.84 0.115 0.000

- 24 Are the colors not ugly? 74.77 0.232 0.042

- 25 Is the lighting and shadow very distinct? 75.45 -0.043 0.000

- 26 Is the lighting and shadow distinct? 58.37 0.035 0.000

- 27 Is there lighting and shadow? 75.93 0.108 mask

- 28 Are the lighting and shadows very beautiful? 80.47 -0.055 0.000

- 29 Are the lighting and shadows beautiful? 71.99 -0.026 0.000

- 30 Can the image evoke a very positive emotional response? 82.63 0.068 0.051

- 31 Can the image evoke a positive emotional response? 63.94 0.117 0.000

- 32 Does the image avoid evoking a negative emotional response? 76.01 0.179 0.000

- 33 Does the image avoid evoking a very negative emotional response? 91.56 0.117 0.000

- 34 Are the image details very exquisite? 74.03 0.078 0.010

- 35 Are the image details exquisite? 71.79 0.091 0.000

- 36 Do the image details avoid being coarse? 68.73 0.215 0.000

- 37 Do the image details avoid being very coarse? 84.62 0.247 0.000

- 38 Does the image avoid being hard to recognize? 87.34 0.267 0.017

- 39 Does the image avoid being fragmented? 85.36 0.288 0.115

- 40 Are the image details realistic? 63.85 0.099 0.000

- Table 19: Accuracy, spearman correlation, and linear weights of VisionReward in text-to-image. (Part 1)

ID Checklist Acc ρ Weight

- 41 Do the image details avoid being unrealistic? 63.94 0.140 0.000

- 42 Do the image details avoid being very unrealistic? 74.19 0.156 0.000

- 43 Do the image details avoid being greatly unrealistic? 83.62 0.177 0.000

- 44 Is the human body in the image completely correct? 61.31 0.063 0.082

- 45 Does the human body in the image avoid errors? 59.02 0.129 0.000

- 46 Does the human body in the image avoid obvious errors? 82.57 0.135 0.055

- 47 Does the human body in the image avoid serious errors? 90.83 0.121 0.030

- 48 Is the human face very beautiful? 65.50 -0.046 0.000

- 49 Is the human face beautiful? 56.88 -0.006 0.000

- 50 Does the human face avoid errors? 57.61 0.113 0.031

- 51 Does the human face avoid serious errors? 91.56 0.132 0.077

- 52 Are the human hands perfect? 90.18 -0.015 0.072

- 53 Are the human hands essentially correct? 25.84 0.059 0.000

- 54 Do the human hands avoid obvious errors? 37.98 0.066 0.000

- 55 Do the human hands avoid serious errors? 77.26 0.048 0.000

- 56 Is the image completely safe? 78.74 0.118 0.000

- 57 Is the image harmless? 86.44 0.106 0.000

- 58 Does the image avoid obvious harmfulness? 92.39 0.109 0.012

- 59 Does the image avoid serious harmfulness? 92.80 0.092 0.015

- 60 Does the image show ”[[prompt]]”? - 0.297 2.354

- Table 20: Accuracy, spearman correlation, and linear weights of VisionReward in text-to-image. (Part 2)

ID Checklist Acc ρ Weight

- 1 Does the video meet all the requirements stated in the text ”[[prompt]]”? 69.5 0.315 0.954

- 2 Does the video meet most of the requirements stated in the text ”[[prompt]]”? 72.9 0.303 0.252

- 3 Does the video meet some of the requirements stated in the text ”[[prompt]]”? 72.9 0.281 0.000

- 4 Does the video not completely fail to meet the requirements stated in the text ”[[prompt]]”? 78.7 0.320 1.142

- 5 Is the composition aesthetically pleasing? 50.8 0.263 0.035

- 6 Does the composition have no obvious flaws? 90.4 0.239 0.025

- 7 Is the focus aesthetically pleasing? 49.8 0.232 0.000

- 8 Does the focus have no obvious flaws? 91.6 0.246 0.000

- 9 Is the camera movement aesthetically pleasing? 76.2 0.012 0.000

- 10 Does the camera movement have no obvious flaws? 97.3 0.142 0.126

- 11 Are the colors exceptionally beautiful? 46.5 0.214 0.000

- 12 Are the colors beautiful? 50.1 0.217 0.000

- 13 Are the colors not unattractive? 82.2 0.225 0.000

- 14 Are the colors not significantly unattractive? 88.6 0.202 0.032

- 15 Is the lighting perfectly accurate? 51.9 0.346 0.163

- 16 Does the lighting have no obvious errors? 86.2 0.259 0.217

- 17 Is there any lighting present? 87.8 0.215 0.020

- 18 Is the lighting exceptionally beautiful? 65.1 0.212 0.136

- 19 Is the lighting beautiful? 55.8 0.240 0.096

- 20 Is the lighting not unattractive? 83.5 0.280 0.155 Table 21: Accuracy, spearman correlation, and linear weights of VisionReward in text-to-video. (Part 1)

ID Checklist Acc ρ Weight

- 21 Is the shape of the object at the beginning of the video completely accurate? 63.0 0.292 0.129

- 22 Does the shape of the object at the beginning have no obvious errors? 76.3 0.274 0.099

- 23 Is the shape of the object at the beginning not chaotic? 91.3 0.256 0.188

- 24 Is the shape of the object perfectly maintained throughout the video? 54.2 0.300 0.184

- 25 Does the shape of the object have no obvious issues throughout the video? 68.8 0.267 0.000

- 26 Does the shape of the object generally have no major issues throughout the video? 84.5 0.259 0.000

- 27 Is the shape of the object not chaotic throughout the video? 93.5 0.240 0.264

- 28 Is the object’s motion highly dynamic? 78.0 -0.079 0.000

- 29 Is the object’s motion dynamic? 69.0 -0.024 0.000

- 30 Is the object’s motion not minimal? 71.2 -0.009 0.000

- 31 Is the object’s motion not static? 66.5 -0.014 0.000

- 32 Is the camera motion highly dynamic? 86.9 -0.054 0.112

- 33 Is the camera motion dynamic? 80.6 -0.062 0.000

- 34 Is the camera motion not minimal? 72.1 -0.061 0.052

- 35 Is the camera motion not static? 58.1 -0.059 0.000

- 36 Is the smoothness of the object’s movement very good? 59.8 0.263 0.026

- 37 Does the smoothness of the object’s movement have no obvious issues? 61.6 0.139 0.000

- 38 Is the object’s movement completely realistic? 66.8 0.338 0.439

- 39 Does the object’s movement have no obvious realism issues? 69.2 0.235 0.000

- 40 Is it very clear? 52.1 0.261 0.000

- 41 Is it clear? 51.0 0.290 0.000

- 42 Is it not blurry? 81.8 0.271 0.000

- 43 Is it not completely blurry? 93.1 0.226 0.000

- 44 Is the image quality very stable? 43.1 0.313 0.269

- 45 Is the image quality stable? 61.2 0.294 0.000

- 46 Is the image quality not unstable? 79.0 0.277 0.000

- 47 Is the image quality free of noticeable instability? 87.6 0.247 0.000

- 48 Is the camera very stable? 54.2 0.197 0.000

- 49 Is the camera not unstable? 83.5 0.267 0.000

- 50 Are the details very refined? 73.0 0.324 0.429

- 51 Are the details relatively refined? 62.3 0.331 0.000

- 52 Are the details not rough? 74.2 0.302 0.008

- 53 Are the details not significantly rough? 89.2 0.271 0.128

- 54 Are all the letters correct? 87.3 0.114 0.058

- 55 Do the letters have no obvious errors? 86.8 0.115 0.000

- 56 Are there any letters present? 89.7 0.104 0.145

- 57 Does it fully comply with the laws of physics? 36.6 0.254 0.000

- 58 Does it partially comply with the laws of physics? 66.7 0.248 0.000

- 59 Does it have no obvious violations of the laws of physics? 77.4 0.231 0.000

- 60 Is the video content part of the physical world? 86.6 0.231 0.394

- 61 Is the video content safe? 92.8 0.000 0.000

- 62 Is the video content definitely free of harmful material? 94.3 0.000 0.000

- 63 Does the video content contain no harmful material? 97.7 0.000 0.000

- 64 Does the video content contain no extremely harmful material? 100.0 0.000 0.000 Table 22: Accuracy, spearman correlation, and linear weights of VisionReward in text-to-video. (Part 2)

#### 12 More Details of MonetBench

Type Image Video

People, Objects, Animals, Story, Human Activity,

Architecture, Landscape, Artificial Scene, Others,

Content

Vehicles, Plants, Food, Natural Animal Activity,

Others, Scenes Physical Phenomena

Unreal, Style, History, Material, Angle and Lens,

Fine-grained Detail, Emotional Expression,

Color, Famous Character, Color/Tone, Surreal,

Normal, Famous Places, World Knowledge,

Challenge

Writing, Complex Combo, Special Effects, Text, Positional, Counting, Spatial Relationship,

Camera Movement,

Logical Consistency,

Style, Temporal Speed

Table 23: Content and Challenge of MonetBench.

Image-MonetBench Construction. We first establish our dataset foundation by collecting 4,038 seed prompts through strategic sampling from established datasets (1,000 from ImageRewardDB (Xu et al. 2023), 1,000 from HPDv2 (Xu et al. 2023), and 2,038 from Pick-a-Pic (Kirstain et al. 2023)). Through systematic analysis of these prompts, we identify nine fundamental visual elements as content categories and twelve distinct aspects of generation complexity as challenge categories, maintaining the categorical distributions observed in the source datasets.

Video-MonetBench Construction. For video prompt evaluation, we initially sample 20,000 prompts from the VproM (Wang and Yang 2024) dataset, which are filtered to 13,342 valid entries after removing duplicates and invalid content. Our video classification system comprises seven content categories reflecting different video scenarios, and thirteen challenge categories capturing various technical and creative aspects of video generation.

Prompt Generation and Filtering. To ensure benchmark quality and diversity, we employ ChatGLM (GLM 2024) to generate 1,000 new prompts for each benchmark following the established category distributions. Each generated prompt undergoes a three-stage filtering process: (1) RougeL (Lin 2004) similarity checking for textual diversity, (2) semantic filtering with a cosine similarity threshold of 0.9, and (3) proportional sampling to maintain the intended category distributions.

Table 23 shows content and challenge categories for MonetBench. The resulting benchmark achieves balanced coverage across all categories while maintaining high standards of prompt diversity and quality. This carefully crafted multi-dimensional design enables comprehensive evaluation of visual reward models across both fundamental content types and various generation challenges.

For experimental efficiency, we provide a condensed ver-

sion by randomly sampling 500 prompts from each benchmark while preserving the categorical distribution.

Details of Classification Proportions. After completing the design of the comprehensive two-dimensional classification framework, we utilized ChatGLM to categorize each prompt in the dataset across content and challenge dimensions. We then calculated the proportions of different classification labels for content and challenges. The content and challenge categories and their respective examples are summarized in Tables 24 and 25. Based on these proportions, we used ChatGLM to construct Benchmark prompts (all prompts were generated by ChatGLM, not directly sampled from the dataset). During the construction process, we specified the investigation direction and randomly sampled four ”seed prompts” from the categorized prompts to generate new, higher-quality prompts with ChatGLM. This synthesis approach produced two benchmark datasets, containing 1,000 and 1,007 meticulously crafted prompts, respectively, preserving the statistical characteristics of the original data.

The final datasets provide balanced and comprehensive coverage of content and challenge categories. Table 26 lists the specific content and challenge categories with detailed descriptions and example prompts, providing a clear understanding of the dataset’s composition. The structured methodology ensures the datasets’ diversity and alignment with real-world visual generation requirements, enabling nuanced benchmarking of visual models.

Image Video Type Ratio Count Type Ratio Count People 8 286 Story 5 265 Objects 4 143 Human Activity 4 212 Animals 4 143 Artificial Scene 3 159 Architecture 4 143 Natural Scenes 3 159 Others 2 72 Animal Activity 2 106 Landscape 2 72 Physical Phenomena 1 53 Vehicles 2 71 Other 1 53 Plants 1 35 Food 1 35

Table 24: Content Categories for Image and Video

Image Video Type Ratio Count Type Ratio Count Unreal 8 187 Style 13 465 Style & Format 8 187 Material/Texture 8 292 Fine-grained Detail 8 186 Emotional Expr. 7 249 Color 4 93 Color/Tone 7 261 Famous Character 4 93 World Knowledge 5 192 History & Culture 4 93 Special Effects 5 183 Normal 2 46 World Knowledge 4 192 Writing 1 23 Spatial Relat. 4 136 Complex Combo 1 23 Camera Move. 4 153 Famous Places 1 23 Surreal 3 108 Positional 1 23 Logical Consist. 2 116 Counting 1 23 Temporal Speed 1 66

Text 1 46

Table 25: Challenge Categories for Image and Video

Categorie Description Example Prompt Content

Human Activity Descriptions about daily human activities, sports, performing arts, and professional skills.

A family enjoying a picnic in a park, children playing soccer.

Animal Activity Descriptions about wild animals, domestic pets, and interactions between animals.

A group of dolphins jumping out of the water.

Natural Scenes Descriptions about weather changes, geological events, and astronomical phenomena.

A thunderstorm with lightning striking the ground.

Artificial Scenes Descriptions about cityscapes, interiors of buildings, vehicles, and industrial production.

A bustling city street with traffic and pedestrians.

A glass shattering in slow motion.

Physical Phenomena Descriptions about physical occurrences like candle burning, ice melting, glass breaking, and explosions.

Alice, a young girl, falls down a rabbit hole into a wonderland full of fantastical creatures and adventures.

Story Descriptions about coherent narratives based on a story or fantasy rather than a single scene or activity.

Other Descriptions about various contents that do not fit into the other specified categories.

Various clips of miscellaneous activities not fitting into other categories.

###### Challenge

Style Descriptions about artistic styles such as realistic, cyberpunk, and animated.

A futuristic city with neon lights and flying cars, portrayed in a cyberpunk style.

Color/Tone Descriptions about color schemes like warm tones, cool tones, monochrome, and high saturation.

A serene landscape in warm, golden tones during sunset.

Camera Movement Descriptions about different camera movements, including fixed, panning, zooming, tracking, and aerial shots.

A drone shot capturing a bird’s eye view of a mountain range.

Special Effects Descriptions about special effects such as particle effects, lighting effects, and transitions.

Fireworks exploding with sparkling particle effects.

Material/Texture Descriptions about materials and textures like metal, wood, glass, and fabric.

Close-up shot of rain droplets on a glass window.

Surreal Descriptions about dreamlike, fantastical, or non-realistic elements.

A dreamlike scene with floating islands in the sky.

Temporal Speed Descriptions about different speeds, including slow motion, normal speed, fast motion, and time reversal.

Slow-motion capture of a hummingbird in flight.

Spatial Relationships Descriptions about the spatial arrangement of objects, their sizes, occlusions, and perspectives.

A house of cards being built, showing each layer’s spatial arrangement.

A documentary about the pyramids of Egypt.

World Knowledge Descriptions about physical laws, famous landmarks, historical events, and renowned personalities.

A mystery story where clues are pieced together logically.

Logical Consistency Descriptions about ensuring logical relationships among events, timelines, and spatial layouts.

Emotional Expression Descriptions about expressions of emotions such as joy, sorrow, fear, and surprise.

A close-up of a person expressing joy after receiving good news.

Text Descriptions about incorporating textual elements dynamically within the footage.

An animated title sequence with dynamic text effects.

Table 26: Video classification standards with example prompts.

