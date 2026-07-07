# arXiv:2412.04814v3[cs.CV]5Mar2025

## LiFT: Leveraging Human Feedback for Text-to-Video Model Alignment

Yibin Wang 1,2 Zhiyu Tan 1,2 Junyan Wang 3 Xiaomeng Yang 2 Cheng Jin 1† Hao Li 1,2†

###### Step 2. Reward Function Learning

1 Fudan University 2 Shanghai Academy of Artificial Intelligence for Science 3 Australian Institute for Machine Learning, The University of Adelaide codegoat24.github.io/LiFT

##### Abstract

A waiter in a black suit. He nods politely...

Caption

[Figure 1]

Caption &Prompt

T2V Model

T2V Model

Recent advances in text-to-video (T2V) generative models have shown impressive capabilities. However, these models are still inadequate in aligning synthesized videos with human preferences (e.g., accurately reflecting text descriptions), which is particularly difficult to address, as human preferences are subjective and challenging to formalize as objective functions. Existing studies train video quality assessment models that rely on human-annotated ratings for video evaluation but overlook the reasoning behind evaluations, limiting their ability to capture nuanced human criteria. Moreover, aligning T2V model using video-based human feedback remains unexplored. Therefore, this paper proposes LIFT, the first method designed to leverage human feedback for T2V model alignment. Specifically, we first construct a Human Rating Annotation dataset, LIFTHRA, consisting of approximately 10k human annotations, each including a score and its corresponding reason. Based on this, we train a reward model LIFT-CRITIC to learn reward function effectively, which serves as a proxy for human judgment, measuring the alignment between given videos and human expectations. Lastly, we leverage the learned reward function to align the T2V model by maximizing the reward-weighted likelihood. As a case study, we fine-tune CogVideoX-2B using our pipeline, surpassing CogVideoX5B across all 16 metrics, demonstrating the effectiveness of human feedback in enhancing video quality.

LiFT-Critic

[Figure 2]

Generated Videos

[Figure 3]

LiFT-Critic

- 2. Semantic Consistency: Bad. The waiter is not nodding.

1. Motion Smoothness:

Normal. There is no obvious action.

- 3. Video Fidelity: Good. No obvious problem.

Reward-weigthed Learning

3. T2V Model Alignment

2. Reward Function Learning

1. Human Feedback Collection

Figure 1. An illustration of the proposed LIFT. First, we construct a comprehensive human feedback dataset. Then, a reward model is trained to learn the reward function. Finally, the T2V model is fine-tuned by the reward model to align its output with human expectations.

primarily arise from the inherent subjectivity of human preferences, which are difficult to formalize as objective functions [17]. Consequently, effectively incorporating human preferences into T2V models remains a major challenge.

In language modeling and the text-to-image domain, learning from human feedback to align model outputs [1, 18, 23, 26] has demonstrated efficiency. Building on this success, recent efforts in T2V generation [21, 38] have incorporated feedback from image reward models [28, 33] into the training process. However, image reward models fail to capture crucial temporal aspects of video, such as motion continuity and smooth transitions between frames, thereby struggling to align T2V models with human expectations. To this end, methods like T2VQA [17] and VideoScore [13] train video quality assessment models based on human-annotated video ratings to better evaluate video’s visual quality, temporal consistency, and so on by assigning a score to each dimension. However, these studies face two main challenges: (1) Lack of interpretability: by focusing solely on assessment results (only learning ratings), these models overlook the underlying reasoning, hindering their ability to capture nuanced human criteria. (2) Limited human feedback guidance: Utilizing video-

##### 1. Introduction

Recent advancements in Text-to-Video (T2V) generation models [11, 12, 30–32, 36, 42] have achieved remarkable results. These models enable users to generate high-quality videos, offering a flexible, controllable approach to video creation. Despite this progress, these models still face challenges such as artifacts, misalignment with semantic requirements, and unnatural motion [13, 21]. These issues

based human feedback for aligning T2V models remains an unexplored area, restricting their ability to meet complex and diverse human expectations.

We argue that the key to aligning T2V models with human expectations lies in constructing an effective pipeline that integrates human feedback into the training process. This pipeline should consist of two key components: a highquality dataset where each annotation not only provides ratings for synthesized videos but also explains the reason, and a reward model that learns a human feedback-based reward function. The dataset is essential for capturing nuanced human evaluation criteria beyond simple numerical scores, while the reward model ensures accurate interpretation of both explicit ratings and underlying human judgments rather than relying on superficial correlations. With this foundation, the reward model can then be leveraged to fine-tune T2V models, providing a reliable optimization signal that guides them toward generating videos that better align with complex and diverse human preferences.

Therefore, this paper takes the first step to propose a novel fine-tuning method, LIFT, leveraging human feedback for T2V model alignment through three key stages, as illustrated in Fig. 1: (1) Human Feedback Collection: generate video-text pairs through diverse prompts, followed by detailed human annotations to build a comprehensive feedback dataset; (2) Reward Function Learning: train a reward model capturing human preferences based on this dataset to predict human feedback scores; and (3) T2V model alignment: fine-tune the T2V model using the learned reward function to optimize its output in alignment with human expectations. Specifically, we first categorize human preferences into three key dimensions: semantic consistency, motion smoothness, and video fidelity. For these dimensions, we propose LIFT-HRA dataset that collects 10K human feedback annotations, each including both ratings and the corresponding reasoning. We then train a reward model, LIFT-CRITIC, based on the large multimedia model (LMM) [24] to learn human feedback-based reward function. Unlike existing assessment models [13, 17], our model not only learns to predict ratings but also captures the reason behind them, thereby improving interpretability and providing a deeper understanding of the evaluation process. Finally, we apply this learned reward function to evaluate the quality of T2V model outputs and update the model using reward-weighted likelihood maximization.

In summary, this work introduces LIFT, the first finetuning pipeline for aligning T2V models with human feedback. To achieve this, we construct LIFT-HRA, a comprehensive dataset where each annotation provides both ratings and detailed reason, enabling models to capture nuanced human evaluation criteria. Additionally, we develop LIFTCRITIC, a reward model designed to accurately learn a human feedback-based reward function, effectively bridging

the gap between subjective human preferences and model optimization. Leveraging these components, we apply our pipeline to fine-tune CogVideoX-2B [36], demonstrating that the resulting model surpasses the larger CogVideoX5B across all 16 metrics of the VBench [15] benchmark, significantly improving alignment with human preferences.

##### 2. Related Work

Diffusion-based T2V models. Recent advancements in T2V focus on leveraging large-scale datasets to train more robust models [6, 10, 11, 30–32]. A common approach involves adapting pre-trained text-to-image (T2I) models by introducing temporal layers and fine-tuning them on video datasets or employing a joint image-video training strategy [3, 30, 32]. Despite their effectiveness, such pipelines are limited in improving spatiotemporal resolution. To address this, recent works [5, 12, 36, 42] offer diverse solutions. For example, VEnhancer [12] unifies temporal and spatial super-resolution with refinement in a single model. However, these models still remain inadequate in aligning synthesized videos with human preferences, such as accurately reflecting text descriptions. We thus train a reward model to capture human preferences and then use it to fine-tune T2V models. Related studies will be introduced in the following. Vision-and-language reward models. In T2I generation, several studies aim to incorporate human preferences into model evaluation and optimization [16, 19, 33, 34, 40]. For instance, [16, 34] collects user-generated prompts and preference annotations via a web platform. For T2V generation, recent works [21, 38] integrate feedback from image reward models [28, 33] into the training process. However, imagebased human feedback overlooks crucial temporal aspects of video which are vital for generating coherent and natural videos. To bridge this gap, methods such as T2VQA [17] and VideoScore [13] train video quality assessment models directly on human-annotated video ratings. However, these models focus solely on assessment results, overlooking the interpretability of the evaluation process. Thus, we propose a novel reward model that evaluates synthesized videos by providing both ratings and corresponding reasons.

Aligning generative model use reward learning. Leveraging reward models, various approaches have been developed to align visual generative models with human preferences. These methods include reinforcement learningbased techniques [2, 4, 9, 41] and reward fine-tuning approaches [8, 18, 20, 27, 37]. By integrating feedback mechanisms, these methods aim to enhance model outputs to better reflect human preferences. To date, video-based human feedback has not been fully leveraged to align T2V models, limiting their ability to meet the complex and diverse expectations of users. Therefore, we explore a novel fine-tuning pipeline to align T2V models with human preferences.

###### Step 1. Human Feedback Collection

###### Step 3. T2V Model Alignment

###### Step 2. Reward Function Learning LiFT-Critic

Caption

Categories: Human, Animal, Place, Action

random selection

T2V

Caption (Cap.)

A waiter in a black suit. He nods politely, conveying attentive service.

###### LLM

Question: (Q)

Generated Video

Captions:

Please identify the semantic consistency issues in this video, focusing on the alignment between the text and the visual content.

- 1. A waiter in a black suit. He nods politely,

conveying attentive service.

- 2. A curious gerbil with bright eyes is crawling

Answer: (A)

Bad. The waiter is not nodding.

LiFT-Critic

Frame 1 Frame 2 Frame n

across a high-tech spacecraft.

Image Encoder ...

[Figure 4]

[Figure 5]

[Figure 6]

MS: Good

SC: Bad

VF: Normal

T2V Human Annotation

[Figure 7]

Prompt Cap.&Q&A

F1 F2 ... Fn

[Figure 8]

[Figure 9]

[Figure 10]

score mapping

[Figure 11]

[Figure 12]

[Figure 13]

Visual-Language Model

Reward-Weighted Learning

Generated Videos

LiFT-HRA

- Figure 2. The overview of our proposed pipeline. This illustration depicts three key steps of our fine-tuning pipeline: (1) Human Feedback Collection: we generate video-text pairs using prompts expanded from random category words with an LLM, then annotate them to create LIFT-HRA. (2) Reward Function Learning: a visual-language model LIFT-CRITIC, is trained to predict human preference scores across three dimensions, learning the reward function from the dataset. (3) T2V Model Alignment: LIFT-CRITIC evaluates the T2Vgenerated videos, assigns scores, and maps them into a reward weight to fine-tune the T2V model, aligning it with human preferences.

##### 3. LIFT

evaluation scores. While effective for training models to assess high-level metrics, these datasets lack critical insights into the evaluation process, i.e., the reasoning behind the assigned scores. For reward function learning, we argue that incorporating such reasoning is crucial for accurately aligning with nuanced human preferences. To address this gap, we introduce LIFT-HRA, a new dataset tailored for reward model training. This dataset combines evaluation scores with their corresponding reason, enabling a more holistic and interpretable alignment with human expections, which will be elaborated in the following.

###### 3.1. Overview

This work aims to integrate human feedback into the training of text-to-video (T2V) models, a challenging task due to the inherent subjectivity of human preferences, which are difficult to formalize as objective functions. Previous research, such as [33] and [28], has explored the use of imagebased reward models for aligning T2V models. However, these reward models are difficult to capture the crucial temporal dynamics of video, thereby limiting their ability to satisfy the complex expectations of video generation. To this end, this work proposes LIFT, the first fine-tuning pipeline designed to align T2V models with human feedback. As illustrated in Fig. 2, our pipeline consists of three sequential stages: (1) Human Feedback Collection (Sec. 3.2): We first collect human feedback and construct a comprehensive dataset LIFT-HRA, where each annotation provides both numerical ratings and detailed reasoning. (2) Reward Function Learning (Sec. 3.3): We then train a reward model that learns a human feedback-based reward function based on this dataset to capture nuanced evaluation criteria. (3) T2V Model Alignment (Sec. 3.4): Finally, the learned reward function is leveraged to fine-tune the T2V model, enabling it to better align with human expectations.

Video-Text Dataset. Our reward model is designed to evaluate synthesized videos based on human preferences. Therefore, constructing a comprehensive video-text dataset is essential. Specifically, we start by generating a set of diverse prompts. This involves creating selection lists for different categories such as humans, animals, places, simple actions, and complex actions. For each prompt, as shown in Fig. 2 (left), we randomly choose 1–2 subjects from the human (e.g., waiter) and animal (e.g., gerbil) categories, a scene (e.g., spacecraft) from the places list, and an action (e.g., nod) from either the simple actions or complex actions categories. These selected elements are combined into a phrase and refined into a detailed textual description using LLM [35]. Finally, multiple videos are generated for each prompt using T2V models, forming the video-text dataset.

###### 3.2. LIFT-HRA: Human Feedback Collection

Annotation. To collect comprehensive human evaluations, we categorize human preferences into three key dimensions: semantic consistency, motion smoothness, and video fidelity. For each video-text pair in the dataset, we enlist

Existing human feedback-based text-to-video datasets [13, 17] primarily focus on outcome evaluation, such as overall video quality or video-text alignment, by collecting single

Video Caption:

Overall, the diversity and comprehensiveness of LIFTHRA provide a robust foundation for accurately capturing human preferences across diverse contexts.

[Figure 14]

A rugged farmer, clad in a plaid shirt, denim jeans, and a widebrimmed hat, rides a powerful chestnut horse across a vast, sunlit field.

Human Annotations:

- 1. Semantic Consistency: Good. (perfectly aligned.) Normal. (minor inconsistencies but the overall meaning is clear.) Bad. (largely inconsistent, with significant discrepancies.)
- 2. Motion Smoothness: Good. (Motions are fluid and natural, with no abrupt transitions.) Normal. (Slight motion irregularities or mild jittering.) Bad. (Motions are choppy or erratic, with obvious frame drops.)
- 3. Video Fidelity: Good. (high visual quality, with clear details, minimal artifacts.) Normal. (Some visual artifacts, blurring, or slight color inaccuracies.) Bad. (low visual quality, with severe blurring, significant artifacts.)

###### 3.3. LIFT-CRITIC: Reward Function Learning

[Figure 15]

Reason: No obvious problem.

Based on the constructed high-quality dataset, we fine-tune a pre-trained Visual-Language Model (VLM) [24] to develop our reward model, LIFT-CRITIC, that learns a reward function based on human feedback. We transform scores into natural language labels (e.g., “Good”, “Normal”, “Bad”) and predict these labels through text generation, aiming to not only predict scores but also provide explanations. Current studies [13, 23] typically add a regression head to VLM for direct score prediction. While straightforward, it fails to capture the complexity of human judgment, which involves nuanced, subjective criteria. In contrast, our approach improves the interpretability of the reward model by capturing the underlying reasoning behind human evaluations. Additionally, the text generation method aligns naturally with the pre-trained VLM’s strengths in multimodal understanding and natural language generation, allowing the model to leverage its capabilities in producing semantically rich, contextually meaningful outputs.

Reason: The horse in the video is running with slight motion irregularities.

[Figure 16]

Reason: In the beginning of the video, the horse is missing a leg.

The farmer's hand appears unnaturally twisted.

- Figure 3. An illustration of our annotation UI. Annotators evaluate each video by assigning scores to each dimension and providing the reason behind their assessments.

annotators to evaluate the generated videos across these dimensions by assigning a score in each dimension and providing a detailed reason. The annotation UI is illustrated in Fig. 3. After all data has been annotated, we perform a three-stage correction process to ensure reliable data quality: 1) Coarse Filtering: Annotators manually review each labeled sample, removing those with obvious annotation errors or misaligned feedback that does not adhere to the evaluation criteria, ensuring data quality. 2) Iterative Refinement: The dataset is then split into two halves. One half trains an initial reward model, which is then used to annotate the other half. If the model’s output aligns with human annotations, they are retained; otherwise, human annotators decide which to keep. The corrected data is used to retrain the model, refining the other half. 3) Final Integration: Using the fully corrected dataset, we train a final reward model. This model is then used to re-annotate the data that was removed during the first stage, and the newly annotated data is incorporated back into the dataset. More details of LIFT-HRA construction are provided in Appendix A.

Fig. 2 (Step 2) illustrates our supervised fine-tuning process. The training begins by passing the video frames through an image encoder to extract image tokens. These image tokens, along with the textual caption, question, and answer, are then input into the pre-trained Visual-Language Model (VLM). Formally, the training sample consists of a triplet: multimodal input M, which includes both the video and its textual description; question Q; and answer A. The optimization objective is based on the autoregressive loss function commonly used in training large language models, but it is applied exclusively to the answer. The loss function can be expressed as:

N

Statistical Analysis. Finally, we collect 10K high-quality human feedback video quality question-answer pairs, forming the LIFT-HRA dataset. Its statistical analysis is visualized in Fig. 4. Specifically, (a) the dataset covers a diverse range of categories, (b) each category contains multiple video types with varying numbers of samples, and (c) human feedback is distributed across all categories, with the majority of videos rated as “Bad”, followed by “Good” and “Normal” in descending order. This indicates that, on average, the videos evaluated tend to have more room for improvement in terms of their alignment with human preferences. Additionally, the variation in feedback distribution across categories highlights different challenges and complexity in evaluating videos from diverse domains, such as actions, animals, and places. This variability is crucial for training robust reward models that can generalize across different video types and better align with human expectations.

log p(Ai | Q,M,A<i;ϕ), (1)

L(ϕ) = −

i=1

where N is the length of the ground-truth answer. To enhance the model’s performance, this work enriches each answer option with detailed reasoning, training the model to predict multi-level scores and generate justifications for its judgments. This added context helps the VLM better understand the meaning of each option, thereby improving its alignment with human preferences.

After training, the reward model can predict a reward score for a given synthesized video and its corresponding caption. Specifically, LIFT-CRITIC evaluates the video across three dimensions, including semantic consistency, motion smoothness, and video fidelity. Since these scores are multi-level qualitative evaluations (e.g., Good, Normal, Bad), a reward score mapping function s, is employed to

Ablation Results on

w/

Ours

Semantic Consistency

Background Consistency

Temporal Flickering

The Number of Category Types Distribution of Video Counts Human Feedback Distribution

[Figure 19]

Easy Act. Complex Act.

[Figure 20]

[Figure 21]

Complex Act.

Place

Easy Act.

Place

Animal

Human

Human

Animal

Human Animal Place Easy Act. Complex Act.

Percentage (%)

(a) (b) (c)

- Figure 4. The visualized statistic results of our proposed LIFT-HRA. It illustrates the distribution of category types, the video count across these categories, and the corresponding human feedback distribution for each category.

translate these qualitative assessments into numerical values. Formally, the reward score for a specific dimension d ∈ D is computed as: s(rϕ,d(x,z)), where rϕ represents the reward model with parameters ϕ. Finally, the overall reward score for a synthesized video is obtained by averaging the scores across all dimensions:

Dsyn, where we set the score function s(·) as {“Good”: 0.9, “Normal”: 0.2, “Bad”: 0.05}. The second term is designed to mitigate the limitations of training exclusively on the synthesized dataset since synthesized videos often suffer from low temporal consistency, which may hinder the model’s ability to maintain subject alignment across frames. By incorporating a real video-text dataset Dreal, the second term acts as a regularizer, grounding the model in realistic frame-to-frame dynamics and ensuring that it learns to generate videos with higher semantic and temporal fidelity. The hyperparameter λ balances the loss between synthetic datasets Dsyn and the real dataset Dreal, where we set λ = 1 here. This balance between synthesized and real data helps the model to generalize better, achieving improved performance in overall video quality.

1 |D| d∈D

rϕ(x,z) =

s(rϕ,d(x,z)), (2)

where x is the synthesized video, and z is the associated caption. The obtained reward score can effectively substitute human evaluations, providing feedback for the alignment of T2V models with human expectations.

###### 3.4. T2V Model Alignment

RWL assigns weights to all samples, allowing the model to leverage the entire dataset. This smooth weighting ensures higher-reward samples contribute more significantly while still utilizing lower-reward samples, enhancing data efficiency and preventing overfitting. The potential problem is the training dataset can become quite large, which may result in increased computational and time costs. Therefore, we also explore another effective reward-learning function. Rejection Sampling (RS). This method can be viewed as a special case of reward-weighted learning, where the score function serves as a hard filter. Specifically, the reward weight is defined such that rϕ(x,z) = 1 for samples evaluated as Good in all dimensions D, and rϕ(x,z) = 0 for all other. Under this definition, the loss function for RS simplifies to:

While the text-to-image domain has showcased the potential of human feedback-based alignment [18], extending this paradigm to video generation poses unique challenges due to the increased dimensionality and the intricate temporal dynamics inherent in video data. Therefore, this work explores novel methods that employ our video reward model for T2V model alignment:

Reward-Weighted Learning (RWL). The core idea of RWL is to use the reward model rϕ to guide the learning of the T2V model p with parameters θ by adjusting the likelihood of generated outputs. Specifically, the T2V model is trained to maximize the likelihood of generating videos that align with higher human-provided rewards, which reflect the more accurate human preferences. Inspired by [18], this process can be formulated as follows:

[−log pθ(x|z)]

LRS(θ) = E(x,z)∼Dsyn

(4)

filtered

L(θ) = E(x,z)∼Dsyn [−rϕ(x,z)log pθ(x|z)]

+ λ · E(x,z)∼Dreal [−log pθ(x|z)].

(3)

+ λ · E(x,z)∼Dreal [−log pθ(x|z)],

Here, Dfilteredsyn = {(x,z) ∈ Dsyn |rϕ,d(x,z) = Good,d ∈ D}, where Dsyn denotes the original synthesized dataset.

where rϕ(x,z) computed by Eq. 2 denotes the reward score for synthesized video x with caption z.

This design effectively reduces the synthesized dataset Dsyn to only include high-quality samples, achieving a similar effect as applying a binary mask.

The first term encourages the model pθ to output aligned with the reward signal rϕ by assigning higher probabilities to high-reward samples from the synthesized dataset

Generally, in comparing RS and RWL, we identify key

Table 1. Quantitative results on video assessment metrics. The first seven metrics correspond to the Quality type, while the remaining correspond to the Semantic type. “RM” denotes the “Reward Model”.

RM Size

Subject Consistency

Background Consistency

Aesthetic Quality

Imaging Quality

Temporal Flickering

Motion Smoothness

Dynamic Degree

Human Action

Models

CogVideoX-2B 94.58 95.45 61.94 63.04 96.94 97.86 60.22 97.34 CogVideoX-5B 94.84 95.64 63.13 63.14 97.19 97.81 62.78 98.40

CogVideoX-2B-LiFT (Ours)

13B 96.15 96.48 63.24 64.01 98.10 98.17 61.89 97.90 40B 96.82 96.79 63.72 64.19 98.20 98.33 62.85 98.44

RM Size

Spatial Relationship

Temporal Style

Overall Consistency

Object Class

Multiple Objects

Appearance Style

Color

Scene

Models

CogVideoX-2B 80.86 61.78 53.30 24.19 27.34 85.75 69.11 24.67 CogVideoX-5B 82.50 55.94 56.57 25.12 27.84 88.99 70.98 24.70

CogVideoX-2B-LiFT (Ours)

13B 84.40 64.66 56.76 24.83 27.23 91.27 77.45 24.86 40B 85.15 66.04 57.63 25.23 27.93 91.77 79.34 25.92

trade-offs: RS is lightweight and efficient but risks overfitting due to reduced training data, while RWL, though computationally more intensive, leverages all reward samples to enhance data efficiency and prevent overfitting. Our work explores the effectiveness of both methods in experiments.

##### 4. Experiments

###### 4.1. Experimental Setup

Models and Settings. For our reward model LIFT-CRITIC, we leverage VILA-1.5 13B/40B [24], which has been pretrained on extensive video understanding datasets, demonstrating robust capabilities in video comprehension tasks. To adapt the model for our specific evaluation scenario, we employ Low-Rank Adaptation (LoRA) [14] to fine-tune all linear layers. For our baseline T2V generative model, we adopt CogVideoX-2B [36], a foundational T2V generation model. We fine-tune all its transformer blocks to enhance its performance, enabling the model to better align with human preferences and improve video quality. More implementation details are provided in Appendix B.1.

Datasets. We fine-tune LIFT-CRITIC using our proposed LiFT-HRA dataset, which comprises approximately 10K video quality question-answer pairs annotated with human feedback. Of these, 9K samples are used for training, while the remaining samples serve as the test set. Fig. 4 visualizes the statistic result of this dataset. To diversify the dataset and expose the model to real video distributions, we incorporate 1K high-quality real video samples from VIDGEN [29]. For T2V model alignment, we generate 40K videos from prompts using CogVideoX-2B as the synthesized dataset and select approximately 20K video-text pairs from OpenVid [25] as the real dataset.

###### 4.2. Results

Quantitative Results. Tab. 1 demonstrates the significant performance improvements achieved by our finetuning pipeline. Specifically, integrating reward learning

Table 2. Evaluation results of our reward model. We assess LIFT-CRITIC across the first three dimensions in the test set and compute the average accuracy based on these evaluations.

Avg. Accuracy GPT-4o-mini 60.42 64.33 68.26 64.34

Semantic Consistency

Motion Smoothness

Video Fidelity

Size

GPT-4o 70.59 67.29 71.99 69.96

LiFT-Critic 13B 80.27 77.48 82.72 80.15 w/o reason 40B 84.55 88.01 88.62 87.06

13B 85.61 88.92 88.51 87.68 40B 90.23 94.89 95.53 93.55

LiFT-Critic

using LIFT-CRITIC at various scales leads to consistent enhancements across nearly all evaluation metrics. For example, the model fine-tuned with LIFT-CRITIC-40B shows marked improvements in “Subject Consistency” and “Motion Smoothness”, reflecting better alignment with human preferences for coherent video generation and fluid motion. This enhancement also indicates that LIFT-CRITIC40B better understands and captures the continuity and smoothness required in high-quality video. When compared to the larger baseline, CogVideoX-5B, the LIFT-CRITIC40B-enhanced model outperforms in critical areas such as “Imaging Quality” and “Multiple Objects”. This improvement highlights the model’s superior ability to generate visually detailed scenes with multiple objects, which is essential for creating richer and more complex video content. We also apply our pipeline to T2V-turbo [21], and the results are shown in Appendix B.4.

Qualitative Results. Fig. 5 displays the visual comparison results. It is clear that our fine-tuned model achieves relatively better performance in terms of semantic consistency, motion smoothness, and video fidelity. Specifically, in the first case, CogVideoX-2B fails to perform the required camera transition as described in the caption, while our method successfully captures this motion. In the second case, both CogVideoX-2B and CogVideoX-5B generate a farmer with a very blurred face, whereas our model produces a clearer

A farmer harvests ripe apples in an orchard during golden hour. The camera captures the lush trees laden with fruit, the farmer’s gentle movements, and the sunlight filtering through the branches.

A student sits in a quiet library, surrounded by towering shelves of books. The camera captures their focused expression as they take notes, then pans to reveal sunlight streaming through a large arched window.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

CogVideoX-2B

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

CogVideoX-5B

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

CogVideoX-2B-LiFT (Ours)

- Figure 5. Qualitative Comparison. We compare the performance of our CogVideoX-2B-LiFT (fine-tuned using reward-weighted learning) against CogVideoX-2B and CogVideoX-5B.

CogVideoX-2b CogVideoX-5b Tie Tie

[Figure 46]

[Figure 47]

[Figure 48]

Semantic Consistency

Percentage(%)

Motion Smoothness Video Fidelity

CogVideoX-2b LiFT

36%

31%

29%

26%

23%

CogVideoX-5b

CogVideoX-2b LiFT

30%

28%

CogVideoX-5b

CogVideoX-2b LiFT

25%

CogVideoX-2b Tie

[Figure 49]

[Figure 50]

[Figure 51]

Semantic Consistency

Percentage(%)

Motion Smoothness Video Fidelity

CogVideoX-2b LiFT

Tie

41%

25%

10%

46%

28%

15%

CogVideoX-2b

CogVideoX-2b LiFT

Tie

49%

29%

14%

CogVideoX-2b LiFT

Tie

[Figure 52]

25%

CogVideoX-2b versus CogVideoX-2b-LiFT CogVideoX-5b versus CogVideoX-2b-LiFT

- Figure 6. Human evaluation results. We report the percentage of queries receiving positive votes for each method and highlight the percentage of queries achieving a majority consensus (at least three out of six positive votes) in the black box.

and more detailed facial representation. These examples highlight the effectiveness of our approach in aligning the generated video with both the textual description and human expectations. More qualitative comparison results are provided in Appendix B.5.

The quantitative comparison and qualitative results of our LIFT-CRITIC are provided in Appendix. B.2-3.

###### 4.3. Human Evaluation

We conduct a human evaluation to compare CogVideoX2B, CogVideoX-5B, and our CogVideoX-2B-LiFT. Using a set of 120 text prompts, we generate videos for each method to ensure consistency. To assess performance, we perform pairwise comparisons between CogVideoX-2B-LiFT and both CogVideoX-2B and CogVideoX-5B. Human raters evaluate each pair, selecting the better-performing method or indicating a tie if both produce similar results. The results, summarized in Fig. 6, show the percentage of queries receiving positive votes for each method. We also highlight

the proportion of cases achieving a majority consensus (at least three out of six positive votes) in the black box. These results demonstrate the advantages of CogVideoX-2B-LiFT over its counterparts. See Appendix. B.6 for more details.

###### 4.4. Ablation Studies

Choices of reward learning. As introduced in Sec. 3.4, we also explore another effective reward learning function, rejection sampling (RS). In this process, only synthesized videos that receive a “Good” score across all three evaluation dimensions are retained. Compared to rewardweighted learning (RWL), this approach effectively reduces the amount of fine-tuning data while ensuring the selected samples meet high-quality standards for model alignment. Fig. 8 presents a comprehensive overview of performance across multiple evaluation dimensions. The results show that RS significantly outperforms the baseline CogVideoX2B, while maintaining competitive performance with the larger CogVideoX-5B. Although slightly less effective than

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

CogVideoX-2B

Oursw/rejectionsampling

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

CogVideoX-2B

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

A professor works in his cozy office as snow falls outside the window. Clad in a yellow sweater, he sits at a desk cluttered with books and manuscripts. The camera moves in for a close-up, slowly advancing towards him.

- Figure 7. Qualitative comparison. We compare the performance of CogVideo-2B, its variations fine-tuned using reward-weighted learning (Ours) and rejection sampling.

[Figure 68]

Object Class

Subject Consistency

Multiple Objects

Imaging Quality

Appearance Style

Overall Consistency

Aesthetic Quality

Motion Smoothness

Background Consistency

Temporal Style

Temporal Flickering

Scene

Color

Spatial Relationship

Human Action

CogVideoX-2B w/ Reward-Weighted Learning w/ Rejection Sampling CogVideoX-5B

- Figure 8. Visualized evaluation results in multiple evaluation dimensions. The middle two methods in the label region represent the CogVideoX-2B model fine-tuned using different reward learning strategies.

LIFT-CRITIC-40B enhanced model achieves consistent improvements across all metrics. These enhancements indicate that a larger reward model better captures nuanced human preferences, leading to improved alignment with subjective quality assessments.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Effects of learning reason. Tab. 2 underscores the advantages of incorporating reason-based annotations into reward function learning. By comparing LIFT-CRITIC with and without reasoning at the same model size, we observe significant performance improvements. Notably, for the 40B model, the inclusion of reason boosts the average accuracy from 87.06% to 93.55%. These results emphasize that providing reasoning behind human feedback allows the reward model to better capture subtle evaluation criteria, leading to improved alignment with human preferences.

CogVideoX-5B

The effect of the real video-text dataset in reward learning is analyzed in Appendix. B.7.

RWL, RS proves to be a more efficient and lightweight method for aligning T2V models with human preferences. Additionally, Fig. 7 showcases the visual improvements and enhanced alignment with human expectations achieved through these approaches.

Effects of reward model size. We explore the performance comparison of LIFT-CRITIC at various scales. Tab. 2 demonstrates that increasing the model size significantly improves performance across all evaluation metrics, resulting in a notable increase in overall accuracy. Besides, the results in Tab. 1 illustrate the impact of reward model size on T2V model performance. Compared to the 13B, the

##### 5. Conclusion

This work proposes the first fine-tuning pipeline LIFT for aligning T2V models with human preferences. First, we curate a human-annotated dataset, LIFT-HRA, which includes both ratings and reason for video evaluation. We then train a reward model LIFT-CRITIC to learn the reward function from this dataset. Lastly, we employ it to align the T2V model. As a case study, we apply this pipeline to CogVideoX-2B, showing that the fine-tuned model outperforms the CogVideoX-5B across all 16 metrics, showcasing the effectiveness of our approach in aligning T2V models with human expectations.

##### References

- [1] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022. 1
- [2] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, pages 22563–22575, 2023. 2
- [4] Chaofeng Chen, Annan Wang, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Enhancing diffusion models with text-encoder reinforcement learning. In ECCV, pages 182–198, 2025. 2
- [5] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [6] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Delving deep into diffusion transformers for image and video generation. arXiv preprint arXiv:2312.04557, 2023. 2
- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, pages 24185–24198, 2024. 12
- [8] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 2
- [9] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. NeurIPS, 36, 2024. 2
- [10] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [11] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. In ECCV, pages 393–411, 2025. 1, 2
- [12] Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667, 2024. 1, 2
- [13] Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang,

- Aaran Arulraj, et al. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv preprint arXiv:2406.15252, 2024. 1, 2, 3, 4, 12
- [14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 6, 12
- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. CVPR, pages 21807–21818, 2023. 2, 12, 13
- [16] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. NeurIPS, 36:36652–36663, 2023. 2
- [17] Tengchuan Kou, Xiaohong Liu, Zicheng Zhang, Chunyi Li, Haoning Wu, Xiongkuo Min, Guangtao Zhai, and Ning Liu. Subjective-aligned dataset and metric for text-to-video quality assessment. arXiv preprint arXiv:2403.11956, 2024. 1, 2, 3
- [18] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 1, 2, 5
- [19] Chunyi Li, Zicheng Zhang, Haoning Wu, Wei Sun, Xiongkuo Min, Xiaohong Liu, Guangtao Zhai, and Weisi Lin. Agiqa-3k: An open database for ai-generated image quality assessment. IEEE Transactions on Circuits and Systems for Video Technology, 2023. 2
- [20] Jiachen Li, Weixi Feng, Wenhu Chen, and William Yang Wang. Reward guided latent consistency distillation. arXiv preprint arXiv:2403.11027, 2024. 2
- [21] Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. T2vturbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. arXiv preprint arXiv:2405.18750, 2024. 1, 2, 6
- [22] Jiachen Li, Qian Long, Jian Zheng, Xiaofeng Gao, Robinson Piramuthu, Wenhu Chen, and William Yang Wang. T2vturbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design. arXiv preprint arXiv:2410.05677, 2024. 13
- [23] Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, et al. Rich human feedback for text-to-image generation. In CVPR, pages 19401–19411,

2024. 1, 4

- [24] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, pages 26689–26699, 2024. 2, 4, 6
- [25] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai.

- Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 6
- [26] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 35:27730–27744, 2022. 1, 14
- [27] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737,

2024. 2

- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 1, 2, 3
- [29] Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, and Hao Li. Vidgen-1m: A large-scale dataset for text-to-video generation. arXiv preprint arXiv:2408.02629, 2024. 6
- [30] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1, 2
- [31] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. NeurIPS, 36, 2024.
- [32] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 1, 2
- [33] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 1, 2, 3

- [34] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. NeurIPS, 36, 2024. 2
- [35] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint

- arXiv:2407.10671, 2024. 3, 11, 12

[36] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint

- arXiv:2408.06072, 2024. 1, 2, 6, 12

- [37] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. CVPR, pages 6463–6474,

2023. 2

- [38] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: instructing video diffusion models with human feedback. In CVPR, pages 6463– 6474, 2024. 1, 2
- [39] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pages 11975–11986, 2023. 12
- [40] Sixian Zhang, Bohan Wang, Junqiang Wu, Yan Li, Tingting Gao, Di Zhang, and Zhongyuan Wang. Learning multidimensional human preference for text-to-image generation. In CVPR, pages 8018–8027, 2024. 2
- [41] Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk. Large-scale reinforcement learning for diffusion models. arXiv preprint arXiv:2401.12244, 2024. 2
- [42] Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv:2410.15458, 2024. 1, 2

Training Loss of LiFT-Critic

##### A. LIFT-HRA

[Figure 73]

###### A.1. Video-Text Dataset Generation

To create a comprehensive video-text dataset, we begin by generating a diverse set of prompts. This process starts with creating selection lists for different categories, including humans, animals, places, and actions. A full list of our categories, along with examples, is provided in Tab. 6. For each prompt, we randomly choose 1-2 subjects from the human and animal categories, a scene from the places category, and an action from either the simple or complex actions categories. These selected elements are then combined into a phrase, which is further refined into a detailed textual description using a large language model (LLM) [35]. Once the prompts are generated, multiple videos are generated for each prompt using T2V models, resulting in a videotext dataset that captures a wide variety of subjects, scenes, and actions.

Loss

Epochs

Figure 9. Visualization of training loss. This figure depicts the progression of training loss for LIFT-CRITIC 13B/40B over the course of epochs.

and realistic.

- • Normal: Some visual artifacts, such as blurring or slight color inaccuracies, but they do not significantly detract from the video’s quality.
- • Bad: Low visual quality, with severe blurring, significant color distortion, or noticeable artifacts that impact the viewing experience.

###### A.2. Annotator Management

We enlist 10 expert annotators, all of whom are undergraduate or graduate students. Some of them will become coauthors of the paper, while others are compensated with a fair salary. All the annotators are proficient in English, ensuring they can fully understand the text prompts.

To ensure the annotators clearly understand the definitions, we provide a set of several example cases for each rating category in the guideline. Some prompts may contain unfamiliar terms or names, which could lead to confusion. In such cases, we instruct annotators to quickly search for any unknown concepts online and skip samples that contain confusing or obscure terminology.

###### A.3. Annotation Guideline

We categorize human preferences in videos into three key dimensions: semantic consistency, motion smoothness, and video fidelity. The definitions of these dimensions are detailed in Tab. 3. For each video-text pair, annotators are asked to evaluate the synthesized videos across these dimensions by assigning a score and providing the corresponding reason. We provide the detailed scoring criteria for each dimension:

###### A.4. Annotation Interface

We design a web-based user interface to streamline the data collection process, as shown in Fig. 3 in the main paper. The main interface features the video on the left side, with a panel on the right displaying the corresponding text prompt at the top. For each dimension of evaluation, annotators are first asked to select a score from the available options and then provide a detailed reason in the text box at the bottom.

###### Semantic Consistency

- • Good: Perfectly aligned. The video content matches the given caption with no ambiguity or confusion.
- • Normal: Minor inconsistencies, but the overall meaning is still clear and understandable.
- • Bad: Largely inconsistent, with significant discrepancies between the video and the caption, making it hard to interpret the intended meaning.

###### A.5. Data Correction

After all data has been annotated, we undertake a comprehensive three-stage data correction process to ensure the dataset is of high quality and can be reliably used for training our models. Each stage is designed to address specific issues that may arise during the annotation process.

###### Motion Smoothness

- • Good: Motions are fluid and natural, with no abrupt transitions or noticeable stuttering.
- • Normal: Slight motion irregularities or mild jittering that does not distract from the overall visual flow.
- • Bad: Motions are choppy or erratic, with obvious frame drops or unnatural movements that break the immersion.

1.Coarse Filtering: In the first step, we perform a broad initial correction by removing data that contains obvious annotation errors or responses that are irrelevant to the evaluation criteria. This includes any annotations where the feedback from the annotators does not align with the predefined evaluation dimensions or scoring guidelines. Additionally, we remove samples where the video or text prompt is un-

###### Video Fidelity

• Good: High visual quality, with clear details, sharp images, and minimal artifacts. The video appears natural

Table 3. Evaluation Dimensions and Definitions. This table provides the detailed definitions of each evaluation dimension used in our annotation process.

Evaluation Dimension Definition Semantic Consistency The consistency and coherence of semantic elements across frames, ensuring the de-

picted objects, characters, and actions remain contextually appropriate and logically related throughout the video.

Motion Smoothness The naturalness and fluidity of motion transitions between consecutive frames, reflecting the absence of jittering, abrupt movements, or frame discontinuities in the video.

Video Fidelity The overall visual accuracy and realism of the video, including the sharpness, texture details, and absence of visual artifacts such as blurriness, distortions, or unnatural effects.

clear or too ambiguous.

- 2.Iterative Refinement: Once the coarse filtering step is

completed, we split the dataset into two halves. The first half is used to train an initial version of the reward model. This model is then applied to the second half of the dataset to generate annotations for those samples. We compare the model-generated annotations with the original human annotations to assess their alignment. If the model’s annotations closely match the human-provided ratings and rationales, they are retained. However, in cases of significant disagreement, the conflicting samples are flagged for further review by human annotators. These annotators manually decide which annotations are correct and should be retained, based on the evaluation criteria. After correcting and validating the second half of the data, it is used to retrain the reward model. The retrained model is then re-applied to the first half of the dataset to improve the quality of its annotations.

- 3.Final Integration: In the final step of the correction

process, we use the fully corrected dataset to train a final version of the reward model. This model is then applied to re-annotate any data that was removed in the first stage during the coarse filtering step. We review the newly annotated data and assess whether the model’s predictions are consistent with human judgment. The annotations that pass this review are re-integrated into the dataset.

Through these rigorous steps, we ensure that our data correction process effectively identifies and rectifies any inconsistencies, ambiguities, or errors in the dataset.

##### B. More Experiments B.1. Implementation Details

We train LIFT-CRITIC-13B with batch size of 16 for 20 epochs, using LoRA [14] rank of 128 and alpha value of 256. For LIFT-CRITIC-40B, we used batch size of 8 for 20 epochs, with LoRA rank of 64 and alpha value of 128. For the vision towers, we employ siglip-so400m-patch14-384 [39] for LIFT-CRITIC-13B and InternViT-6B-448px-V1-2 [7] for LIFT-CRITIC-40B. Both vision towers, along with

Table 4. Evaluation results on VideoScore benchmark. We assess baselines across visual quality (VQ), temporal consistency (TC), dynamic degree (DD), text-to-video alignment (TA), and factual consistency (FC) and compute the average accuracy.

VideoScore Benchmark VQ TC DD TA FC

Method

CogVideoX-2B 2.86 2.78 2.65 2.91 2.71 CogVideoX-5B 3.01 2.94 2.73 2.96 2.87

CogVideoX-2B-LiFT 3.18 2.99 2.96 3.04 2.90

their MLP projectors, are frozen during training, and only the linear layers of the LLM are fine-tuned. Optimization is performed using the AdamW optimizer with a base learning rate of 1e−5. A cosine learning rate scheduler is employed, incorporating a warmup ratio of 3e−2. The training process is conducted on 8 NVIDIA H100 GPUs. The visualization of training loss is shown in Fig. 9.

we employ Vbench [15], a comprehensive benchmark suite to assess the performance of T2V generation, which decomposes “video generation quality” into specific, hierarchical, and disentangled dimensions. Each dimension is evaluated using tailored prompts and specialized evaluation methods. The evaluation prompts are optimized using Qwen2.5-72B-Instruct [35] since the CogVideoX [36] model is trained with long prompts.

###### B.2. Evaluation on VideoScore Benchmark

We compare the performance of CogVideoX-2B, its finetuned version CogVideoX-2B-LiFT, and CogVideoX-5B using VideoScore [13] as the assessment model. The results in Tab. 4 further demonstrate the effectiveness of our LIFT.

###### B.3. Qualitative Results of LIFT-CRITIC

To demonstrate the effectiveness of our LIFT-CRITIC in providing nuanced evaluations that align with human preferences, we provide several qualitative results in Fig. 12 and Fig. 13.

Table 5. Quantitative results on video assessment metrics. The first seven metrics correspond to the Quality type, while the remaining correspond to the Semantic type. “RM” denotes the “Reward Model”.

RM Size

Subject Consistency

Background Consistency

Aesthetic Quality

Imaging Quality

Temporal Flickering

Motion Smoothness

Dynamic Degree

Human Action

Models

T2V-Turbo 95.6 97.01 63.2 73.36 97.19 97.16 56.67 95.00 T2V-Turbo-LiFT 40B 96.5 97.41 63.88 73.91 97.53 98.01 68.44 96.00

Temporal Style

Overall Consistency

Object Class

Multiple Objects

Appearance Style

Spatial Relationship

RM Size

Scene

Color

Models

T2V-Turbo 90.09 36.36 55.09 25.51 28.17 95.63 48.86 24.40 T2V-Turbo-LiFT 40B 92.63 44.81 58.71 26.03 30.22 96.12 56.81 26.17

[Figure 74]

Ablation Results on

###### B.4. Reward-weighted Learning on T2V-Turbo

[Figure 75]

To demonstrate the robustness of our proposed pipeline, we apply it to T2V-Turbo [22]. The results are presented in Tab. 5.

w/

Ours

###### B.5. More Qualitative Comparison

We provide more qualitative comparison in Fig. 14 and Fig. 15.

###### B.6. Human Evaluation

We conduct a human evaluation to compare the performance of CogVideoX-2B, CogVideoX-5B, and the finetuned version CogVideoX-2B-LiFT. To facilitate this process, we design a web-based user interface as illustrated in Fig. 16.

Semantic Consistency

Background Consistency

Temporal Flickering

Figure 10. Visualized ablation results on Dreal. We compare the performance of our CogVideoX-2B-LiFT model with and without the inclusion of Dreal during T2V model alignment.

We also utilize our reward model, LIFT-CRITIC, to compare the videos generated by the three methods. The evaluation results are presented in Fig. 11, which demonstrate that the rankings produced by our reward model are highly consistent with the results of human evaluations across all three dimensions. This alignment further validates that our reward model effectively captures human preferences, providing reliable and interpretable feedback for T2V model alignment.

all three metrics, underscoring the critical role of real data in enhancing temporal quality. These results demonstrate that incorporating Dreal effectively grounds the model in realistic frame-to-frame dynamics, enabling it to produce videos with superior semantic and temporal fidelity. This aligns closely with our design objectives, reaffirming the importance of blending synthesized and real-world data to achieve robust performance.

###### B.7. Ablation Study of Real Video-Text Dataset

During our T2V model alignment, we incorporate a real video-text dataset, Dreal, to address the limitations of relying solely on synthesized datasets: synthesized videos often exhibit low temporal consistency, which hinders the model’s ability to maintain coherent subject alignment across frames and generate smooth motion dynamics.

The Number of Category Types Distribution of Video Counts Human Feedback Distribution

##### C. Societal Impacts

[Figure 76]

Easy Act. Complex Act.

Our work aims to improve the alignment of text-to-video generation models with human preferences, enhancing their usability in applications such as educational content creation, video summarization, and creative media production.

[Figure 77]

[Figure 78]

Complex Act.

Place

Easy Act.

To evaluate its effectiveness, we conduct an ablation study, with results presented in Fig. 10. Specifically, we compare the performance of our full CogVideoX-2BLiFT model to an ablated version (excluding Dreal) using VBench’s [15] temporal quality-related metrics: subject consistency, background consistency, and temporal flickering. The ablated model exhibits significant declines across

However, our method also introduces potential societal risks. First, as the reward model depends on human annotations, biases present in the data may propagate into the model, potentially leading to unintended or unfair outcomes in specific applications. Therefore, the enhanced alignment with human preferences may inadvertently amplify harm-

Place

Animal

Human

Human

Animal

Human Animal Place Easy Act. Complex Act.

13

Percentage (%)

Semantic Consistency

Motion Smoothness Video Fidelity

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

CogVideoX-2b

CogVideoX-5b

###### CogVideoX-2b LiFT

Percentage (%) Percentage (%)

Percentage (%)

- Figure 11. Visualized results of comparison evaluated by LIFT-CRITIC. We leverage LIFT-CRITIC to evaluate and compare the performance of CogVideoX-2B, CogVideoX-5B, and our proposed CogVideoX-2B-LiFT.

##### E. Ethical Statement

ful content if the underlying feedback data contains inappropriate or unethical biases. We have taken steps to mitigate these risks by employing diverse and carefully curated datasets and emphasizing transparency in the annotation and training processes.

In this work, we affirm our commitment to ethical research practices and responsible innovation. To the best of our knowledge, this study does not involve any data, methodologies, or applications that raise ethical concerns. All experiments and analyses were conducted in compliance with established ethical guidelines, ensuring the integrity and transparency of our research process.

In summary, while our work primarily targets technical advancements, its broader societal implications require continuous scrutiny to ensure ethical and equitable deployment. We encourage the community to further evaluate and refine such systems to maximize their positive impact while minimizing potential harm.

##### D. Limitations and Future Works

Our work has several limitations and opens avenues for future exploration:

Expanding the diversity and scope of the dataset. Currently, our curated dataset only focuses on three primary objective dimensions, i.e., video fidelity, motion smoothness, and semantic consistency. Additionally, it is limited to a specific set of text categories (i.e., human, animal, place, action) and relies on a relatively simple feedback score (good, normal, bad). This constrained scope reduces the diversity of human feedback data. Therefore, expanding the dataset to include more evaluation dimensions, subjective text categories (e.g., artistic style), and richer feedback formats, such as ranking or multi-level annotations, represents a valuable direction for future research.

Exploring alternative optimization strategies. Our work employs reward-weighted likelihood maximization to align the T2V model. However, integrating reinforcement learning from human feedback (RLHF), as demonstrated in language models [26], may offer further advantages. RLHF allows for online sample generation during training and incorporates KL-regularization, potentially mitigating overfitting to specific reward functions and fostering a more balanced improvement across evaluation metrics.

### Video

Caption:

[Figure 83]

A diligent student, dressed in a casual green sweater and jeans, kneels beside a row of vibrant potted plants in a sunny garden. She carefully waters each plant with a small, red watering can, her focused expression and gentle movements highlighting her care and attention. The sunlight filters through the leaves, casting a warm, natural glow on the scene, emphasizing the tranquility and connection with nature.

[Figure 84]

LiFT-Critic-40B:

- 1. Semantic Consistency: Good. Basically no problem.
- 2. Video Fidelity:
- 3. Motion Smoothness: Normal. The character's movements are slightly unsmooth.

[Figure 85]

Bad. The character's face is slightly distorted, the hands are twisted.

### Video

Caption:

[Figure 86]

A diligent cashier, wearing a green apron over a white shirt and black pants, types swiftly on a modern cash register in a well-lit convenience store. Her focused expression and quick, precise movements highlight her efficiency. The store's clean, organized shelves and a few customers in the background add to the realistic setting. The scene captures the everyday hustle and bustle of a busy retail environment.

[Figure 87]

LiFT-Critic-40B:

- 1. Semantic Consistency: Normal. The video does not show black pants and cashier’s

expression.

- 2. Video Fidelity: Bad. The cashier's hand in the video is severely deformed.

- 3. Motion Smoothness: Good. The character in the video walks basically continuous.

[Figure 88]

###### 3. Motion Smoothness:

Bad. The character's movements is not continuous, there are sudden changes.

### Video

- Figure 12. Qualitative results of LIFT-CRITIC. We present several case studies illustrating how our LIFT-CRITIC evaluates synthesized videos.

Caption:

[Figure 89]

A woman in a casual outfit, consisting of a light blue sweater and dark jeans, walks confidently through a modern apartment complex. The setting features sleek, multi-story buildings with clean lines and large windows, reflecting the afternoon sunlight. She moves gracefully along a paved pathway, surrounded by well-manicured landscaping and occasional benches. Her steps are purposeful, and her expression is calm, capturing a sense of daily routine in an urban residential environment.

15

[Figure 90]

LiFT-Critic-40B:

- 1. Semantic Consistency: Good. Basically correct.
- 2. Video Fidelity: Good. No obvious problems.

#### Video

Caption:

[Figure 92]

A creative artist, dressed in a vibrant red vest and black beret, stands next to a playful monkey wearing a tiny blue hat. They are both positioned in front of a colorful easel with a half-finished painting, set against a backdrop of a bustling, artistic studio filled with paintbrushes, palettes, and canvases. The monkey holds a small brush, mimicking the artist's movements, adding a whimsical touch to the scene.

[Figure 93]

LiFT-Critic-40B:

1. Semantic Consistency: Good. Basically correct.

[Figure 94]

- 2. Video Fidelity: Bad. The character's face is distorted, and the hands are twisted.

- 3. Motion Smoothness: Good. The character in the video walks basically continuous.

###### 3. Motion Smoothness:

Normal. The character's movements in the video are not smooth, and the monkey's movements are also not smooth.

#### Video

Caption:

[Figure 95]

A woman in a casual outfit, consisting of a light blue sweater and dark jeans, walks confidently through a modern apartment complex. The setting features sleek, multi-story buildings with clean lines and large windows, reflecting the afternoon sunlight. She moves gracefully along a paved pathway, surrounded by well-manicured landscaping and occasional benches. Her steps are purposeful, and her expression is calm, capturing a sense of daily routine in an urban residential environment.

[Figure 96]

LiFT-Critic-40B:

- 1. Semantic Consistency: Good. Basically correct.
- 2. Video Fidelity: Good. No obvious problems.

[Figure 97]

- Figure 13. Qualitative results of LIFT-CRITIC. We present several case studies illustrating how our LIFT-CRITIC evaluates synthesized videos.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

CogVideoX-2B

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

CogVideoX-2B-LiFT

###### (Ours)

A musician sits on a wooden porch, strumming his acoustic guitar under a starlit sky. The moon casts a soft, silvery glow, illuminating his focused expression and the gentle movements of his hands. The serene night is filled with the melodic sounds of his music, blending harmoniously with the rustling leaves and distant cricket chirps.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

CogVideoX-2B

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

CogVideoX-2B-LiFT

###### (Ours)

A woman with long, flowing hair stands on a sandy beach, pulling a colorful kite string. The kite, vibrant and large, soars high above her against a clear blue sky. Her casual attire, consisting of a white tank top and denim shorts, complements the relaxed, sunny atmosphere. She looks upwards, her face lit with a sense of joy and freedom.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

CogVideoX-2B

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

CogVideoX-2B-LiFT

###### (Ours)

A person sits at a wooden desk in a quiet room, writing in a leather-bound journal. A desk lamp casts a warm glow, illuminating the open pages. The camera focuses on the person's hand as they write, showing a steaming cup of tea and a stack of books nearby.

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

CogVideoX-2B

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

CogVideoX-2B-LiFT

(Ours)

In a classroom, the teacher stands in front of a large chalkboard, explaining a complex concept with vivid gestures while students take notes at their desks.

- Figure 14. More qualitative comparison. We compare the performance of CogVideo-2B and its LiFT fine-tuned version CogVideo-2BLiFT.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

CogVideoX-2B

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

CogVideoX-2B-LiFT

###### (Ours)

A director, dressed in a chic black blazer, white shirt, and dark trousers, walks purposefully across a bustling film set. The background features crew members operating cameras and adjusting lights, while the director's confident strides and focused expression convey a sense of authority and determination. The scene captures the dynamic energy of a professional film production in action.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

CogVideoX-2B

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

CogVideoX-2B-LiFT

###### (Ours)

An artist, dressed in a futuristic spacesuit with a sleek, silver helmet and glowing blue accents, crawls through the narrow, dimly lit corridors of a space bank. The walls are lined with advanced security panels and blinking lights, creating a tense atmosphere. The artist's movements are cautious and deliberate, hands and knees pressing against the cold metal floor. In the background, the hum of the ship's systems and the occasional red warning light add to the sense of urgency and danger.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

CogVideoX-2B

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

CogVideoX-2B-LiFT

###### (Ours)

A middle-aged man, dressed in a plaid shirt, khaki pants, and gardening gloves, tends to a vibrant flower bed in a sunny garden. He kneels on a green mat, carefully planting colorful flowers, his tools neatly arranged beside him. The lush greenery and blooming plants around him create a peaceful and serene environment.

- Figure 15. More qualitative comparison. We compare the performance of CogVideo-2B and its LiFT fine-tuned version CogVideo-2BLiFT.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

###### Figure 16. An illustration of Human evaluation UI. We design a web-based user interface to streamline the human evaluation process.

###### Table 6. Categories and examples. This table displays a subset of our category list used to construct prompts.

|Category|Examples|
|---|---|
|Human<br><br>|man, woman, children, people, student, teacher, worker, doctor, nurse, engineer, lawyer, police officer, firefighter, chef, waiter, cashier, janitor, farmer, artist...|
|Land Anim.|horse, antelope, chicken, dog, cat, ch, badger, bat, bear, beaver, bison, bobcat, buffalo, camel, capybara, cheetah, chimpanzee, cobra, coyote, deer, dingo, elephant, mammoth, ferret, fox, gazelle, gerbil, giraffe, gorilla, hamster, hare, hedgehog, hippopotamus, hyena, ibex, jackal, jaguar, kangaroo, koala, leopard, lion, lizard, mammoth, meerkat, mongoose, monkey, moose, mouse, otter, panda, panther...|
|Other Anim.<br><br>|Fish, Squid, Octopus, Shark, Dolphin, Whale, Turtle, Seahorse, Starfish, Crab, Lobster, Snail, Clam, Oyster, Jellyfish, Anemone, Ray, Eel, Cuttlefish, Manta Ray, Pufferfish, Angelfish, Butterfly Fish, Parrotfish, Grouper, Mackerel, Sardine, Herring, Cod, Salmon, Trout, Bass, Halibut, Flounder, Tuna, Marlin, Swordfish, Nudibranch, Sea Cucumber, Sea Urchin, Sand Dollar, Zebrafish, Pipefish, Mole Crab, Ghost Shrimp, Hermit Crab, Coral, Bird, Eagle...|
|Place|mountains, forests, parks, city streets, Tokyo streets, Shanghai streets, beaches, rivers, lakes, deserts, oceans, valleys, canyons, waterfalls, cliffs, glaciers, volcanoes, geysers, hot springs, caves, swamps, wetlands, tundra, grasslands, savannas, rainforests, deciduous forests, boreal forests, taiga, orchards, vineyards, farmlands, meadows, heaths, woodlands, thickets, scrublands, chaparral, tropical dry forests, temperate rainforests, temperate deciduous forests, temperate grasslands, temperate shrublands, subtropical deserts, subtropical dry forests, subtropical moist forests, subtropical shrublands, subtropical wet forests, subtropical woodlands, urban parks, botanical gardens, zoological parks, amusement parks, historical sites, museums, galleries, stadiums, arenas, theaters, cinemas, restaurants, shops, markets, supermarkets, malls, hotels, resorts, airports, train stations, bus stops, highways, bridges...|
|Imagine Place<br><br>|car in a pot, airplane in a bowl, train in the cloud, elephant on a spoon, guitar in a teacup, house on a leaf, shoe in a tree, phone in a puddle, piano in a bubble, dinosaur in a bathtub, moon in a saucer, book in a galaxy, clock in a flower, computer in a handbag, lamp in a well, camera in a fridge, bicycle in a cake, hat in a pool, sun in a shoebox, camera in a birdcage, laptop in a fishbowl, glasses in a garden, bed in a suitcase, mirror in a pond, television in a tent, comb in a forest, microwave in a hat, towel in a jar, oven in a backpack, table in a bubble, chair in a cloud, toothbrush in a lake, soap in a park, perfume in a cave, toilet in a spaceship, key in a painting, window in a book, door in a cake, stove in a suitcase, sink in a meadow...|
|Easy Act.|running, walking, jumping, crawling, wiping, caressing, swimming, dancing, singing, drawing, writing, reading, cooking, eating, drinking, climbing, hiking, fishing, sleeping, sitting, standing, kicking, throwing, catching, pushing, pulling, lifting, digging, painting, typing, listening, talking, laughing, crying, smiling...|
|Difficult Act.|blinking, breathing, yawning, stretching, sneezing, coughing, scratching, cleaning the car, washing dishes, stir frying, doing laundry, vacuuming the house, mopping the floor, dusting furniture, organizing shelves, ironing clothes, polishing silverware, gardening, watering plants, cooking dinner, baking a cake, making coffee, loading the dishwasher, unloading the dishwasher, taking out the trash, sweeping the porch, raking leaves, shoveling snow, painting a room, fixing a leaky faucet, changing a light bulb, replacing batteries, checking smoke detectors, cleaning windows, scrubbing the bathroom, rearranging furniture, hanging pictures, repairing a fence, washing the dog, feeding pets, walking the dog, cleaning the fish tank, organizing the garage, cleaning the gutters, repairing a bike...|

