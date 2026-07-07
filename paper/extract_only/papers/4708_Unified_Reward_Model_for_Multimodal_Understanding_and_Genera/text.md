## Unified Reward Model for Multimodal Understanding and Generation

### arXiv:2503.05236v2[cs.CV]25Feb2026

Yibin Wang1,2 Yuhang Zang3† Hao Li1,2 Cheng Jin1,2† Jiaqi Wang2,3† 1 Fudan University 2 Shanghai Innovation Institute 3 Shanghai AI Lab Project Page: codegoat24.github.io/UnifiedReward

###### 1. Unified Reward Model Training

Abstract—Recent advances in human preference alignment have significantly improved multimodal generation and understanding. A key approach is to train reward models that provide supervision signals for preference optimization. However, existing reward models are often task-specific, limiting their adaptability across diverse visual applications. We also argue that a reward model that jointly learning to assess multiple vision tasks may foster a synergistic effect, where improved image understanding enhances image generation assessment, and refined image evaluation benefits video assessment through better frame analysis. To this end, this paper proposes UNIFIEDREWARD, the first unified reward model for multimodal understanding and generation assessment. It supports both pairwise ranking and pointwise scoring, providing effective reward signals for vision model preference alignment. Specifically, (1) we first train UNIFIEDREWARD on our constructed large-scale human preference dataset, which covers both image and video generation/understanding tasks. (2) Then, we leverage it to automatically construct high-quality pairwise preference data from vision models by progressively filtering their outputs through our two-stage strategy, i.e., pair ranking and point sifting. (3) Finally, we use these data to align vision models with human preferences via Direct Preference Optimization (DPO). Experimental results show that jointly learning to assess diverse visual tasks yields substantial mutual benefits. We further apply our pipeline to both vision understanding and generation, achieving consistent improvements across each domain.

[Figure 1]

Reward Score

Point Data Unified Reward

Unified Preference Data

Unified Preference Learning

Preference Rank

Pair Data

2. Preference Data Construction

Preference Data

[Figure 2]

[Figure 3]

Data

Unified Reward

VLM/ Diffusion

Generation 1. Pair Rank 2. Point Sift

Chosen: Img/Vid/Resp. Rejected: Img/Vid/Resp.

3. Generation/Understanding Model Alignment

[Figure 4]

Preference Data

Direct Preference Optimization

VLM/ Diffusion

Img/Vid, Caption Img/Vid, Question, Resp

Fig. 1. Overview of UnifiedReward for Multimodal Understanding and Generation Alignment. The pipeline includes three steps: (1) Unified Reward Model Training, (2) Preference Data Construction, and (3) Generation/Understanding Model Alignment.

Img1/Vid1, Img2/Vid2, Caption

Img/Vid, Question, Resp1, Resp2

in the lack of a comprehensive human preference dataset that spans a wide range of visual tasks. (2) We intuitively argue that visual tasks are inherently interconnected, and jointly learning multiple visual tasks may create a mutually reinforcing effect. Specifically, enhanced image understanding may improve the evaluation of image generation by providing a more accurate assessment of content quality and contextual relevance. Similarly, improvements in image evaluation may benefit video evaluation, as high-quality image assessments lead to more accurate evaluations of video frames, contributing to overall better quality video assessment. This cross-task synergy facilitates a more robust evaluation of outputs across both image and video modalities in tasks involving understanding and generation. It inspires the development of a unified multimodal reward model that yields more precise reward signals for preference optimization.

Index Terms—Reward model, preference alignment, multimodal understanding, multimodal generation.

I. INTRODUCTION

# R

ECENT advancements in human preference alignment have substantially propelled the progress of multimodal

generation and understanding tasks. A straightforward technique is to directly collect human feedback to construct preference datasets for model optimization [1]–[3]. Despite its effectiveness, collecting large-scale human feedback is timeconsuming and resource-intensive. To this end, an alternative popular approach involves learning reward models [4]–[9] from a limited amount of preference data and using the learned reward function to generate preference data based on the output of vision models. This synthetic preference data can then be leveraged for vision model preference alignment, significantly reducing the need for extensive human annotations.

To this end, we propose UNIFIEDREWARD, the first unified reward model for assessing multimodal understanding and generation, capable of both pairwise ranking and pointwise scoring, which can be utilized for preference alignment on diverse vision tasks.

Despite their progress, we posit two concerns: (1) current reward models are often tailored to specific tasks, as shown in Tab. I, limiting their adaptability across diverse visual understanding and generative tasks. The key challenge lies

As illustrated in Fig. 1, our fine-tuning pipeline includes three key stages: (1) First, we construct a large-scale human preference dataset that spans both image and video generation/understanding tasks and develop UNIFIEDREWARD based on this dataset. (2) Next, we employ UNIFIEDREWARD to

†Corresponding authors.

TABLE I COMPARISON OF OUR REWARD MODEL WITH RECENT APPROACHES. UNIFIEDREWARD IS CAPABLE OF ASSESSING BOTH IMAGE AND VIDEO UNDERSTANDING AND GENERATION. “PAIR” AND “POINT” REFER TO “PAIR RANKING” AND “POINT SCORING”.

Image Generation

Image Understand

Video Generation

Video Understand

Reward Model Method

PickScore’23 [10] Point ✓ HPS’23 [11] Point ✓ ImageReward’23 [11] Point ✓ LLaVA-Critic’24 [5] Pair/Point ✓ VideoScore’24 [4] Point ✓ LiFT’24 [4] Point ✓ VisionReward’24 [12] Point ✓ ✓ VideoReward’25 [7] Point ✓

UnifiedReward Pair/Point ✓ ✓ ✓ ✓

automatically construct high-quality preference pair data by selecting the outputs of specific baselines, such as Vision Language Models (VLM) and diffusion models, through multistage filtering, i.e., pair ranking and point sifting. (3) Finally, we use these preference pairs to align the outputs of these models with human preferences via direct preference optimization. Our experiments show that learning multiple visual tasks together yields significant reciprocal benefits, enhancing performance in each individual domain. By implementing our pipeline across both vision understanding and generation baselines, we observe notable improvements in each domain.

Contributions: (1) We construct a large-scale human preference dataset that spans diverse vision tasks and develop UNIFIEDREWARD, the first unified reward model for assessing multimodal understanding and generation. (2) We propose a general pipeline for both vision understanding and generation model preference alignment, which remains an underexplored area in current research. Extensive experiments demonstrate its effectiveness in improving the performance of vision models in each domain. (3) Our experiments reveal that learning to assess image and video tasks jointly leads to a synergistic improvement in performance across different visual domains.

Through this work, we aim to expand the scope of reward models, making them more adaptable, generalizable, and effective across various visual applications.

II. RELATED WORK

Reward Models are crucial in aligning vision understanding and generation models with human preferences. Traditional methods [13]–[15] for evaluating vision quality and semantic consistency rely on metrics such as FID [16] and CLIP scores [17]. Despite their effectiveness, they are limited in their ability to capture human preferences. Therefore, recent studies [18]– [20] utilize human preference data to fine-tune CLIP, enabling them to better predict and align with human evaluations. With the advent of VLMs [21], [22], their robust ability to align visual and textual data makes them promising candidates for reward modeling. These models can be adapted into two main categories based on their capabilities: understanding assessment models [5], [6], which are designed exclusively for evaluating visual understanding tasks, and generation assessment models

- [4], [7], [12], [23], which focus on assessing visual synthesis quality.

However, these reward models are typically designed for specific tasks, as illustrated in Tab. I, restricting their ability to adapt to diverse visual understanding and generative tasks. In this work, we propose the first unified reward model for both image and video understanding and generation assessment, which is more adaptable, generalizable, and effective across various visual applications.

Preference Learning for VLM/Diffusion is widely utilized to enhance their image and video understanding/generation performance. In video understanding, prior works have explored reinforcement learning with human feedback to refine reward models for factuality assessment [24], while [3], [9], [25] have used reinforcement learning on AI-generated feedback to enhance video LMMs. For image understanding, researchers investigate Direct Preference Optimization (DPO) as an alternative approach to preference modeling. [25], [26] apply DPO to refine rewards distilled from GPT-4V across different model outputs, while [27] constructs preference datasets by generating positive and negative sample pairs using ChatGPT, informed by detailed image descriptions. Similar methods have been applied to image generation [1], [8], [12] and video generation [2], [4], [7], [12], [28]–[31], using reward models or human preference data to align pre-trained diffusion models.

However, these methods rely on task-specific reward models, and no unified reward model has been developed for preference learning across both image and video generation and understanding tasks. This limits the generalizability and efficiency of reward-based alignment. Our work investigates the effectiveness of joint learning to assess multiple visual tasks, demonstrating that cross-task synergy enhances the evaluation capabilities across each domain.

III. METHOD A. Overview

This work aims to develop a unified reward model for vision model preference alignment. Existing studies typically develop specialized reward models for specific tasks as shown in Tab. I, which restricts their adaptability across diverse visual applications. Furthermore, we intuitively argue that jointly learning multiple visual tasks can create a mutually reinforcing effect, yet this remains an underexplored area. To this end, this work proposes UNIFIEDREWARD, the first unified reward model for multimodal understanding and generation assessment, enabling both pair ranking and point scoring. It is then leveraged for aligning Vision-Language Models (VLMs) and Diffusion model alignment, enabling more robust and adaptable preference learning across diverse visual tasks.

Our pipeline is illustrated in Fig. 3. Specifically, we first construct a large-scale, unified preference dataset (Sec. III-B1) and train our UNIFIEDREWARD model on this dataset (Sec. III-B2). Then, we curate preference datasets for VLMs and diffusion models by applying pair ranking and point sifting on their outputs (Sec. III-C). These curated datasets are subsequently used for Direct Preference Optimization (DPO) (Sec. III-D), effectively aligning models with human preferences.

[Figure 5]

Eval M

TABLE II TRAINING DATASETS FOR IMAGE AND VIDEO GENERATION/UNDERSTANDING ASSESSMENT. “*” INDICATES THE DATASET IS PREPROCESSED IN OUR WORK.

Share GPT Video

[Figure 6]

Pairwise

Pairwise

use

Video Understand

Pointwise

HPD

OIP

VideoDPO

LLava-Critic

Video Generation

Preference Data Total: 236K

LLava-Critic

Image Understand

Task Method Dataset Size

ShareGPT

VideoFeedback

Video

Image Generation

|Generation Pair<br><br>EvalMuse* HPD* OIP<br><br>Point EvalMuse*<br><br>|3K 25.6K 7.4K 32.7K|
|---|---|
|Understanding<br><br>Pair LLaVA-Critic Point LLaVA-Critic|25K 25K<br><br>|

EvalMuse

A

R

H

Pointwise

-T

FiL

Proportion (%)

- Fig. 2. Visualization of Statistical Results. This figure presents the distribution of our constructed unified preference dataset, along with the pairwise and pointwise distributions for each task.

Image

- B. Unified Reward Model Training

|Generation<br><br>Pair VideoDPO Point<br><br>LiFT-HRA VideoFeedback<br><br>|10K 20K 36.6K|
|---|---|
|Understanding<br><br>Pair ShareGPTVideo Point ShareGPTVideo*<br><br>|17K 34K|

1) Unified Preference Dataset Construction: A comprehensive human preference dataset that spans multiple visionrelated tasks is essential for training a unified reward model. However, existing human feedback datasets, such as [2], [4],

Video

- [5], are typically designed for specific tasks, limiting their generalizability. Currently, there is no human preference dataset that comprehensively covers both visual understanding and generation tasks, highlighting the need for a more versatile dataset. To bridge this gap, we integrate existing datasets and preprocess them to construct the first large-scale unified human preference dataset, which consists of approximately 236K data samples covering both image and video understanding and generation tasks. The detailed statistics and visualized distributions of the dataset are presented in Fig. 2 and Tab. II, respectively. We will elaborate on the data construction process for each task in the following. Image Generation. EvalMuse [32] consists of 4K prompts, each with multiple images generated by different models. Each image is evaluated by at least three annotators, who provide an overall score (1-5) and element-wise labels indicating whether specific elements are present. For pointwise score learning, we compute the final score as the average of all ratings. An element is considered generated if at least two annotators agree; otherwise, it is marked as not generated. We integrate the overall score and element-wise labels as assessment answers for reward model learning. For pairwise ranking, we select the images with the highest and lowest average score from the same prompt as a ranking pair. Human Preference Dataset (HPD) [33] contains 700K human preference votes. For each prompt, two images generated by different models are provided, each with its respective vote count. In our work, we directly use the vote counts to construct pairwise ranking data, ranking the image with more votes as the preferred one. Open-Image-Preferences (OIP) 1 contains 7.4K text-toimage preference pairs, which are directly used in this work. Image Understanding. LLava-Critic-113K [5] consists of 40K pointwise score and 73K pairwise ranking data samples for image understanding assessment learning. From this dataset, we select 25K samples for each of pairwise ranking and pointwise scoring learning. Video Generation. VideoDPO [2] includes 10K synthesized video pairs for text-to-video model DPO. We directly use this dataset for our pairwise ranking learning in video generation. LiFT-HRA [4] and VideoFeedback [23]

provide extensive human feedback for pointwise scoring of synthesized videos, which we directly incorporate into our work. Video Understanding. ShareGPTVideo-DPO [34] contains 17K video understanding DPO data, where each response in a pair is assigned an evaluation score. We directly use the pair data for pairwise ranking learning, while the individual response scores are extracted for pointwise scoring learning.

For pairwise ranking datasets, we standardize the answer format as “image/video/response X is better than image/video/response Y”, where “X” and “Y” represent the assigned indices. If the dataset includes evaluation justifications [4], [5], we retain them to allow the model to learn from human reasoning. For pointwise scoring, we do not enforce a unified response format or score range, allowing the model to learn from diverse rating styles and scoring systems across different datasets. To ensure alignment between evaluation criteria and responses, we adjust instruction prompts accordingly.

As shown in Fig. 2, the training data for video generation pairwise ranking assessment is relatively limited compared to other tasks, but we believe that the synergistic effect of multitask learning can alleviate this deficiency. Overall, our dataset provides a diverse and comprehensive collection of human preferences, covering both pairwise ranking and pointwise scoring across image and video understanding/generation tasks. This enables effective reward model training, ensuring robust performance across multimodal understanding and generation applications.

2) Unified Preference Learning: Based on the comprehensive datasets, we fine-tune a pre-trained VLM [35] with strong vision understanding capabilities to develop UNIFIEDREWARD, jointly training it across diverse vision tasks. Instead of learning evaluation from scratch, we integrate assessment ability as an additional discriminative skill, leveraging the model’s existing visual comprehension to enhance its evaluation performance across various tasks.

Fig. 3 (top) illustrates our training process. Specifically, for multimodal generation evaluation, our model takes vision tokens, instruction input, and a caption as input. In contrast, for multimodal understanding, the caption is replaced by a question

1https://huggingface.co/datasets/data-is-better-together/open-imagepreferences-v1-binarized

1. Unified Reward Model Training

Mutimodal Generation Mutimodal Understanding

###### 1. Pointwise Score:

###### 1. Pointwise Score:

You are given a text caption and a generated image/video based on that caption. Your task is to evaluate this...

You are provided with a image/video and a question for this video. Please review the corresponding response...

###### 2. Pairwise Rank:

###### 2. Pairwise Rank:

You are given a text caption and two generated images/videos based on that caption. Your task is to evaluate and compare these...

You are provided with a image/video and a question for this video. Please review the two responses...

Image/Video Tokens (point/pair)

Image/Video Tokens

Response (point/pair) ...

Instruction ...

Caption ...

Instruction ...

Question ...

...

...

Unified Reward

Unified Reward

[Figure 7]

[Figure 8]

Answer ...

...

Answer

#### 2. Preference Data Construction

#### 3. Generation/Understanding Model Alignment

(c) Point Sifting

(a) Data Generation

(b) Pair Ranking

Final Chosen

Pairs

Output 1

[Figure 9]

[Figure 10]

[Figure 11]

w/ Max point score

Chosen List

[Figure 12]

VLM/ Diffusion

Unified Reward

Direct Preference Optimization

VLM/ Diffusion

Unified Reward

Preference Data

...

...

w/ Min point score

Rejected List

Final Rejected

Output N

1

N/2

- Fig. 3. Method Overview. (1) Unified Reward Model Training: train a unified reward model for both multimodal generation and understanding assessment using pointwise scoring and pairwise ranking strategy. (2) Preference Data Construction: use the trained reward model to construct high-quality preference data through three steps: (a) data generation from vision models, (b) pairwise ranking to divide the chosen and rejected outputs, and (c) pointwise filtering to refine the chosen and rejected samples. (3) Generation/Understanding Model Alignment: the constructed preference data is then used to align vision models with human preference via Direct Preference Optimization (DPO).

and the corresponding response(s), aligning the input format with the respective task requirements. The model is trained to predict the pointwise score or pairwise ranking based on the criteria specified in the instruction prompt. If the training data includes justifications, the model is also trained to generate detailed explanations to support its evaluations. During training, the optimization objective is standard cross-entropy loss, but it is computed only on the model’s predicted answer.

Specifically, our pipeline includes three sequential steps:

- (1) Data Generation. Given an image/video-question pair (or generation prompt), a VLM (or diffusion model) generates

multiple candidate outputs {O1,O2,...,ON}. These outputs serve as the initial pool for followed preference data filtering.

- (2) Pair Ranking. Given N outputs, we group them into N/2 pairs and use our model to perform pairwise ranking for each pair. Then, we classify these ranked pairs into a

[Figure 13]

After training our UNIFIEDREWARD, we leverage it for preference alignment in multimodal understanding and generation models. This process consists of two sequential steps, i.e., Preference Data Construction and Generation/Understanding Model Alignment. The following sections provide a detailed explanation of each step.

chosen list C = {O1c,O2c,...,ON/c 2} and a rejected list R = {O1r,O2r,...,ON/r 2}. (3) Point Sifting. Finally, we apply our model to assign pointwise scores to all outputs in both the chosen list and the rejected list. The final preference data pair is determined as:

- C. Preference Data Construction

The quality of preference alignment data directly determines the effectiveness of model alignment. Existing methods [4], [5], [7] are often limited to a single evaluation strategy, either assigning pairwise rankings or pointwise scores to model outputs for preference data construction. In contrast, this work leverages both pairwise ranking and pointwise scoring capabilities of UNIFIEDREWARD, enabling a higher quality preference data construction pipeline, as illustrated in Fig. 3 (bottom left).

(Oc∗ = arg max O∈C

S(O), Or∗ = arg min O∈R

S(O)),

where S(O) represents the pointwise score assigned by our model, Oc∗ is the most preferred output and Or∗ is the least preferred output.

By combining pairwise ranking and pointwise scoring, the final preference data could provide a high-quality and reliable preference signal, effectively capturing both relative comparisons and absolute quality assessments.

- D. Generation/Understanding Model Alignment

After constructing the preference data, we leverage it for multimodal generation and understanding model alignment using DPO, which enables models to align their outputs with human preferences without explicit reward modeling, optimizing directly based on ranked preference pairs.

DPO for Multimodal Generation. For multimodal generation tasks, diffusion [36] is widely used due to their strong capability in generating high-quality and diverse outputs across image and video synthesis. Therefore, we apply DPO on diffusion models to align their outputs with human preferences.

Given the constructed preference pair dataset DGen = {(xw0 ,xl0)i}Mi=1, where xw0 and xli represents the preferred generated sample and the less preferred sample respectively, M denotes the number of samples, we optimize the diffusion model by comparing the noise prediction differences between the fine-tuned model and pre-trained reference model [1]:

L(θ) = −E(xw

0 ,xl0)∼DGen, t∼U(0,T), xwt ∼q(xwt |xw0 ), xlt∼q(xlt|xl0) log σ − βgTω(λt) ∥ϵw − ϵθ(xwt ,t)∥22 − ∥ϵw − ϵref(xwt ,t)∥22

− ∥ϵl − ϵθ(xlt,t)∥22 − ∥ϵl − ϵref(xlt,t)∥22 ,

where xwt and xlt are the noisy latents derived from xw0 and xl0 at timestep t, respectively. ϵθ(x∗t,t) and ϵref(x∗t,t) denote the predicted noise from the fine-tuned and pre-trained reference diffusion models, respectively. βg is a temperature hyperparameter controlling optimization strength, σ is the logistic function, λt represents the signal-to-noise ratio, and Tω(λt) is a weighting function, which is treated as a constant equal to βg in this work.

This loss encourages the fine-tuned diffusion model to reduce the denoising error for preferred samples while increasing it for less preferred ones, thereby improving the generation quality.

DPO for Multimodal Understanding. Similar to generation alignment, we apply DPO to adjust the model’s response preference for multimodal understanding models, i.e., VLMs. Given an input x (e.g., an image/video-question pair) with a preferred response yw and a less preferred response yl from preference pair dataset DUnd, the optimization is formulated as:

πθ(yw|x) πref(yw|x)

L(θ) = −E(x,y

w,yl)∼DUnd βu log σ log

πθ(yl|x) πref(yl|x)

−log

,

where πθ(y∗|x) and πref(y∗|x) are the response probability under the fine-tuned model and reference model, respectively. βu is a hyperparameter that controls optimization sensitivity.

This loss encourages the VLM to increase the likelihood of generating preferred responses while decreasing it for less preferred ones, thereby improving the model’s alignment with human preferences and enhancing reasoning quality.

TABLE III IMAGE UNDERSTANDING ASSESSMENT. WE EVALUATE VARIOUS ASPECTS ON VLREWARDBENCH.

Overall Accuracy

Macro Accuracy

Models General Hallu. Reason.

Gemini-1.5-Pro 50.8 72.5 64.2 67.2 62.5 GPT-4o 49.1 67.6 70.5 65.8 62.4 LLaVA-Critic 47.4 38.5 53.8 46.9 46.6

OV-7B 32.2 20.1 57.1 29.6 36.5 w/ Img. Und. 47.6 38.3 54.5 47.4 46.8 w/ Img. Und.+Gen. 49.8 52.6 58.1 50.4 53.5 w/ Img.+Vid. Und 52.4 55.6 57.2 52.7 55.1

UnifiedReward 60.6 78.4 60.5 66.1 66.5

IV. EXPERIMENTS A. Implementation Details

Reward Model. We adopt the pre-trained LLaVA-OneVision 7B (OV-7B) [35] as the base architecture for UNIFIEDREWARD. Training is conducted on 8 H100 GPUs with a batch size of 2, gradient accumulation steps of 16, a learning rate of 2.5×10−6, and a warm-up ratio of 0.3. Inference remains efficient: ∼1s for direct answers and ∼3s with brief rationales, with no additional overhead compared to the base model. We additionally train UNIFIEDREWARD on Qwen2.5-VL [37] to verify the robustness of UNIFIEDREWARD across different baselines.

Multimodal Understanding DPO. Based on UNIFIEDREWARD, we apply DPO to LLaVA-OneVision 7B [35] and LLaVA-Video [38] to enhance their performance in image and video understanding, respectively. We use a batch size of 1, gradient accumulation steps of 16, a learning rate of 5 × 10−7, and set βu = 0.1.

Multimodal Generation DPO. For image and video generation DPO, we use SDXL-Turbo [39] and T2V-Turbo [29], respectively. The parameter βg is set to 5000, with batch sizes of 32 for SDXL-Turbo and 16 for T2V-Turbo. We construct 10K preference data for video generation DPO and 14k for other DPO tasks. The number of candidate outputs N is set to 10. All models are trained for 3 epochs.

Evaluations. Multimodal Understanding: We evaluate the image and video understanding assessment of UNIFIEDREWARD on VLRewardBench [40] and ShareGPTVideo [34] (1K samples for testing), respectively. Multimodal Generation: GenAI-Bench [41] includes both image and video generation reward benchmarks, which are utilized. Besides, we also employ VideoGen-RewardBench [7] as the video generation assessment benchmark. DPO: For image understanding, LLaVABench [42], WildVision [43], LLaVABenchWilder [44], LiveBench [45], MMHal [46], MMBench [47], MME [48], MathVista [49], DocVQA [50], and TextVQA [51] are employed. We use LMMs-Eval [52] toolkit to evaluate. For video understanding, we employ “gpt-3.5-turbo-1106” for MSRVTT [53], MSVD [54], TGIF [55] evaluation, while using the VLMEvalKit [56] toolkit for evaluate LongVideoBench [57], MLVU [58] and VideoMME [59] evaluation. For image generation evaluation, we generate images conditioned on captions from the Partiprompt [60] and HPSv2 [11] benchmarks (1632 and 3200 captions respectively) and utilize the

A sorcerer with mechanical arms casting spells in deep space, manipulating stars with his gestures.

A glowing dragon soaring through floating islands, leaving behind a trail of shimmering stardust.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

T2V-Turbo

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

w/ VideoDPO

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

w/ UnifiedReward (Ours)

- Fig. 4. Qualitative DPO Comparison on T2V-Turbo. We compare the qualitative performance of the original T2V-Turbo, DPO trained with VideoDPO, and DPO trained with UnifiedReward.

[Figure 32]

[Figure 33]

[Figure 34]

beautiful witch, hyper detailed, flowing psychadelic background intricate and detailed, octane render.

[Figure 35]

[Figure 36]

[Figure 37]

hyperrealism, epic photography, closeup, 35mm film, photography, of young girl, in city.

[Figure 38]

[Figure 39]

[Figure 40]

- A person sits in a comfortable armchair, reading a book.

SDXL-Turbo w/ Pick-a-Pic

w/ UnifiedReward (Ours)

[Figure 41]

[Figure 42]

[Figure 43]

photograph of a beautiful female model wearing high tech military clothing in the desert, detailed.

SDXL-Turbo w/ Pick-a-Pic

w/ UnifiedReward (Ours)

Fig. 5. Qualitative DPO Comparison on SDXL-Turbo. We compare the qualitative performance of the original SDXL-Turbo, DPO trained on Pick-aPic dataset, and DPO trained with UnifiedReward.

image reward models, i.e., PickScore [10], HPSv2 [11] and ImageReward [18] for quality assessment. VBench [15] is used for video generation assessment.

- B. Reward Model Comparison Results

TABLE IV IMAGE AND VIDEO GENERATION COMPARISON. “TAU” INDICATES THAT ACCURACY IS CALCULATED WITH TIES, AND “DIFF” EXCLUDES TIED PAIRS WHEN CALCULATING ACCURACY.

A person sits in a comfortable armchair, reading a book.

###### Image Generation

###### Video Generation

Method

Method

GenAI-Bench GenAI-Bench VideoGen-Reward tau diff tau diff tau diff

PickScore 53.2 67.2 VideoScore 46.2 70.6 42.1 49.9 HPSv2 51.6 68.4 LiFT 41.2 60.1 40.6 58.3 ImageReward 47.8 65.0 VisionReward 52.1 73.1 57.4 68.2 VisionReward 46.8 66.4 VideoReward 50.2 73.3 60.1 73.9

OV-7B 39.7 53.2 OV-7B 40.8 51.4 40.4 50.2 w/ Img. Gen. 39.4 64.0 w/ Vid. Gen. 48.2 69.4 44.3 62.4 w/ Img. Gen.+Und. 47.7 65.9 w/ Vid. Gen.+Und. 49.1 71.6 45.1 64.9 w/ Img.+Vid. Gen. 50.5 67.6 w/ Img.+Vid. Gen. 52.0 73.6 53.6 70.7

UnifiedReward 54.8 70.9 UnifiedReward 60.7 77.2 66.6 79.3

Image Understanding. We compare our method with LLaVA-Critic [5], as well as two closed-source models [61], [62]. The experimental results, shown in Tab. III, indicate that our method outperforms baselines in most metrics, e.g., macro accuracy, which demonstrates the superiority of our method in image understanding assessment. For Video Understanding, we explore the effectiveness of multi-task learning in video understanding assessment on ShareGPTVideo, which will be analyzed in Sec. IV-D. In Image Generation assessment, we compare our method with both traditional and state-of-the-art approaches [10]–[12], [18]. The results are presented in Tab. IV. Notably, the latest work, VisionReward, supports reward modeling for both image and video generation. However, it trains separate models for each task using their respective datasets, whereas our approach jointly learns multiple tasks within a unified framework, leading to relatively better performance. For Video Generation, we compare our method with

the latest approaches [2], [4], [12], [23]. As shown in Fig. 2, our training data for video generation assessment is relatively limited. However, as demonstrated in Tab. IV, our method excels across all metrics when compared to all baselines, highlighting that multitask learning not only mitigates the issue of insufficient training data but also enhances the learning effectiveness for video generation assessment.

C. DPO Comparison Results

Image Understanding. We compare our method with LLaVACritic by employing the same image-question pair source [63] to construct preference data for OV-7B, ensuring a fair comparison. The results, presented in Tab. VI, demonstrate that DPO using our method consistently outperforms the baseline across all benchmarks. For instance, our method achieves a 3.4% improvement on LLaVABench, highlighting its superior effectiveness. Video Understanding. We extract prompts from ShareGPTVideo-DPO [34] to construct preference data for LLaVA-Video-7B [38], sharing the same video-question pair source as LLaVA-Houd-DPO [3]. To evaluate the effectiveness, we compare our UNIFIEDREWARD-based DPO with HoudDPO and the latest TPO [9]. The results, presented in Tab.

TABLE V VIDEO UNDERSTANDING DPO COMPARISON. ALL METHODS ARE TRAINED WITH THE SAME SETTINGS.

MSRVTT MSVD TGIF LongVideoBench MLVU Video-MME Acc. Score Acc. Score Acc. Score Acc. M-Avg. Short Medium Long Avg.

Method

LLaVA-Video-7B’24 52.8 3.24 69.7 3.90 51.9 3.37 58.1 70.9 76.1 61.6 52.3 63.3 w/ Houd-DPO’24 56.8 3.34 72.8 3.97 54.9 3.45 58.0 71.8 76.3 61.3 51.2 63.0 w/ TPO’25 55.0 3.25 72.6 3.93 53.7 3.40 58.2 72.6 76.9 62.1 52.1 63.7

###### w/ UnifiedReward 65.0 3.45 78.3 4.01 59.7 3.51 58.4 72.3 76.2 61.3 52.5 63.5

TABLE VI IMAGE UNDERSTANDING DPO COMPARISON. WE COMPARE OUR METHOD WITH LLAVA-CRITIC FOR DPO BASED ON LLAVA-ONEVISION-7B.

LLaVABen. WildVision LLaVABenWilder LiveBen. MMHal MMBen MME MathVista DocVQA TextVQA

OV-7B 90.3 54.9 67.8 77.1 3.19 80.9 1994.1 62.6 87.2 80.1 w/ LLaVA-Critic 100.3 67.3 71.6 84.5 3.91 80.5 1998.9 63.2 86.98 79.2

###### w/ UnifiedReward 101.4 67.8 75.0 85.4 4.01 81.2 2008.5 62.9 87.4 79.5

TABLE VII GENERATION DPO COMPARISON. (A) IMAGE GENERATION DPO EVALUATED BY IMAGE REWARD METRICS. (B) VIDEO GENERATION DPO EVALUATED ON VBENCH.

##### (A) Image Generation DPO (SDXL-Turbo) Method PickScore HPSv2 ImageReward

Baseline 43.24 29.37 0.82 w/ Pick-a-Pic 54.32 30.03 0.93 w/ UnifiedReward 63.32 32.44 1.05

##### (B) Video Generation DPO (T2V-Turbo) Method Total Quality Semantics

Baseline 80.95 82.71 73.93 w/ VideoDPO 81.80 83.80 73.81 w/ UnifiedReward 82.10 84.11 74.06

V, demonstrate the superiority of our approach. Notably, our method significantly outperforms the baselines on MSRVTT, MSVD, and TGIF, demonstrating its effectiveness in video understanding. For the other three multi-choice question datasets, although our DPO data do not include this question type, this does not lead to any negative impact. Our performance still remains comparable to the baselines, indicating the robustness and generalization ability of our approach. For Image Generation, we extract prompts from Pick-a-Pic [10], to construct preference data. As shown in Tab. VII (A), training on the constructed data using our UNIFIEDREWARD achieves better performance compared to directly training on the original dataset. This demonstrates the effectiveness of our approach in refining preference data for improved model alignment. The qualitative comparison results are shown in Fig. 5.

For Video Generation, we compare our method with VideoDPO [2], using the same prompt source for preference data construction. The results in Tab. VII (B) demonstrate our superiority in enhancing both generation quality and semantic consistency, highlighting the effectiveness of our approach. The qualitative comparison results are shown in Fig. 4.

TABLE VIII VIDEO UNDERSTANDING ASSESSMENT. WE EVALUATE THE PERFORMANCE OF OUR MODEL USING DIFFERENT TRAINING DATA CONFIGURATIONS.

w/ Vid. Und.

w/ Vid.&Img. Und.

w/ Vid Und.&Gen.

UnifiedReward Acc. 48.2 74.2 76.6 78.6 84.0

OV-7B

D. Discussion

- 1) Multi-task Assessment Learning: This work intuitively argues that visual tasks are inherently interconnected, and jointly learning multiple visual tasks may create a mutually reinforcing effect. Therefore, we explore the effectiveness of multi-task learning on the reward model. Specifically, for each task, we employ different training data configurations to train the model, investigating the impact of jointly learning across different modalities (image and video) and tasks (understanding and generation). For example, for the image understanding task, we design three training configurations to investigate the impact of multi-task learning: (1) training solely on image understanding assessment, (2) jointly learning image understanding and image generation assessment, and (3) jointly learning image understanding and video understanding assessment. The results are presented in Tab. III. Notably, our findings indicate that multi-task learning significantly enhances the model’s overall performance compared to training on a single task. For instance, jointly training on both image and video understanding tasks improves overall accuracy and macro accuracy by 5.3% and 8.3%, respectively, compared to training solely on image understanding. Results for other tasks are presented in Tabs. VIII and IV, which consistently demonstrate its effectiveness. These results highlight the benefits of leveraging shared knowledge across different visual tasks, leading to a more robust and generalizable reward model.
- 2) Cross-Task Synergy Beyond Scaling Data: To verify that the gains of UNIFIEDREWARD do not simply stem from a larger training set, we introduce a budget-matched control in

TABLE IX CROSS-TASK SYNERGY ANALYSIS. (A) BUDGET-MATCHED CONTROL. (B) TRANSFER MATRIX BY SINGLE-DOMAIN TRAINING AND CROSS-DOMAIN EVALUATION.

VLReward Bench

ShareGPT Video

GenAI Image

GenAI Video

Method

- (A) Budget-matched control Baseline 29.6 48.2 53.2 50.2 Single-task (native) 47.4 74.2 64.0 62.4 Single-task (step-matched) 49.0 75.5 65.0 63.1 UnifiedReward 66.1 84.0 70.9 79.3

- (B) Transfer matrix (single-domain training) Baseline 29.6 48.2 53.2 50.2 Image Understanding-only 47.4 61.5 61.8 52.5 Image Generation-only 41.0 55.8 64.0 55.2 Video Understanding-only 40.2 74.2 57.5 57.8 Video Generation-only 36.0 62.7 60.2 62.4 UnifiedReward 66.1 84.0 70.9 79.3

TABLE X PERFORMANCE COMPARISON ON DIFFERENT BACKBONES. WE COMPARE THE PERFORMANCE OF UNIFIEDREWARD TRAINED ON LLAVA-ONEVISION AND QWEN2.5-VL.

GenAI-Bench VLRewardBench UnifiedReward Image Video General Hallu. Reason.

Overall Accuracy

Macro Accuracy

LLaVA-OV-7b 70.9 77.2 60.6 78.4 60.5 66.1 66.5 Qwen2.5VL-3b 68.9 78.5 82.1 60.8 65.7 72.8 69.5 Qwen2.5VL-7b 76.0 82.5 84.2 68.4 73.6 77.7 75.4 Qwen2.5VL-32b 79.0 85.9 87.8 74.8 75.5 81.5 79.3

Tab. IX (A). Specifically, Single-task (native) denotes singledomain reward models trained on their original task-specific data, while Single-task (step-matched) denotes the same models further oversampled to match the total number of update steps used by UNIFIEDREWARD. Despite this budget matching, our model still achieves the best performance across all evaluation axes. This shows that its advantage is not merely due to having more training data or longer optimization, but arises from positive cross-task synergy between understanding and generation. We further analyze directional transfer by training the reward model on a single domain and evaluating across all domains (Tab. IX (B)). The results show clear crosstask promotion: understanding-centric training also improves generation evaluation, indicating that transfer is not limited to the source domain. At the same time, single-domain training generalizes best within its own modality, while still providing consistent positive gains to other modalities. In contrast, UNIFIEDREWARD remains consistently strong across all targets, suggesting that unified multi-task learning strengthens positive transfer while reducing modality-specific overfitting.

3) Robustness on Different Baselines: To further demonstrate robustness across base models, we additionally train UNIFIEDREWARD on Qwen2.5-VL [37]. As shown in Tab. X, the same improvement trend holds on this stronger backbone. We also observe that larger backbones deliver better overall results while preserving the advantage of UNIFIEDREWARD, suggesting that our approach is compatible with stronger priors and scales reliably with model capacity.

TABLE XI PERFORMANCE UNDER IMBALANCED TRAINING-DATA COMPOSITIONS. THE FIRST COLUMN DENOTES THE SAMPLE AMOUNT OF Video Generation : Video Understanding : Image Generation : Image Understanding; “K” INDICATES THOUSANDS OF SAMPLES.

Vid.Gen:Und: Img.Gen:Und

GenAI-Video ShareGPTVideo GenAI-Image VLRewardBench

10:10:10:10K 71.5 73.0 62.1 58.1 10:20:20:20K 71.0 74.3 64.9 60.2 10:30:30:30K 69.8 75.9 66.0 61.5 20:30:30:30K 72.0 76.2 66.4 61.1 30:30:30:30K 72.9 77.1 66.9 61.8

- 4) Impact of Imbalanced Training Data: We investigate the impact of cross-task data imbalance in Tab. XI. Starting from a balanced allocation, increasing the amount of nonvideo-generation data while keeping the video-generation data fixed improves performance on the over-represented tasks, but consistently degrades video-generation performance. This suggests that underrepresented tasks are more susceptible to being overwhelmed during joint optimization. Once the videogeneration data is rebalanced, its performance recovers, while the other tasks remain largely stable. These results highlight the importance of balanced sampling for maintaining mutual gains across tasks.
- 5) Preference Signal Source Comparison: We compare different preference signal sources for constructing SDXL [39] DPO pairs. To ensure a fair comparison, we keep the same DPO backbone and training pipeline, and only replace the signal source. As shown in Tab. XII (A), UNIFIEDREWARD consistently outperforms both GPT-4o and the image-generation-only reward model (under matched training budgets) across evaluation dimensions, indicating that our reward signal provides stronger supervision than closed-source and single-domain alternatives. The qualitative cases in Fig. 7 show the same trend, with better prompt alignment, cleaner compositions, and fewer visible artifacts.

###### 6) Data Construction Strategy Ablation: We further con-

duct an ablation on our proposed two-stage preference data construction strategy. Tab. XII (B) shows that the full two-stage pipeline consistently outperforms random selection, point-scoreonly, and pair-rank-only variants. This comparison indicates that the two stages play complementary roles: pair ranking provides reliable relative ordering between candidates, while point scoring filters out low-quality responses that may still survive pairwise comparison. Combining them yields cleaner and more stable preference pairs, which translates into stronger downstream alignment performance.

###### 7) Applied to Image Generation GRPO: We further test

whether UNIFIEDREWARD can generalize beyond DPO-style optimization pipelines by applying it to group relative policy optimization (GRPO) [64] on FLUX.1-dev [65]. As shown in Tab. XIII, reward optimization with UNIFIEDREWARD consistently improves all reported evaluation views over the vanilla FLUX baseline. Compared with alternative reward signals, our method remains strongest overall, indicating that the learned reward provides stable and transferable guidance

FLUX.1-dev w/ HPSv2 w/ PickScore w/ UnifiedReward

FLUX.1-dev w/ HPSv2 w/ PickScore w/ UnifiedReward

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

In the ink painting style, a naive giant panda is sitting on the majestic Great Wall, leisurely chewing bamboo.

The art deco music festival poster has a fox made of polished brass in the center of the picture, with smooth lines and elegant posture.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

The magic book on the left shot out light as it was opened and hit a much larger crystal ball on the right. The words "The ancient prophecy awakens the sleeping giant" appeared on the ball.

Design an environmentally themed poster with a picture of vibrant forests and streams, with the art word 'Protect Green Home'on the top of the poster.

- Fig. 6. Qualitative GRPO Comparison on FLUX. We compare the visual quality of FLUX.1-dev and GRPO variants optimized with different reward models.

[Figure 60]

[Figure 61]

SDXL

w/ UnifiedReward (Two-Stage)

[Figure 62]

w/ UnifiedReward (Pair Ranking)

w/GPT-4o

[Figure 63]

a portrait of young girl.

[Figure 64]

w/ UnifiedReward (Point Sifting)

- Fig. 7. Qualitative DPO Comparison on SDXL. We compare the performance of SDXL, DPO with GPT-4o, UnifiedReward, and UnifiedReward without the pair ranking and point sifting stage.

###### TABLE XIII GRPO RESULTS ON FLUX WITH DIFFERENT REWARD MODELS.

Method CLIP ImageReward Aesthetic FLUX.1-dev 34.40 1.27 6.13 w/ HPSv2 33.35 1.34 6.20 w/ PickScore 33.61 1.32 6.25 w/ UnifiedReward 34.43 1.38 6.31

that our learned reward can serve as an effective optimization signal in various settings.

V. ETHICAL STATEMENT

In this work, we affirm our commitment to ethical research practices and responsible innovation. To the best of our knowledge, this study does not involve any data, methodologies, or applications that raise ethical concerns. All experiments and analyses were conducted in compliance with established ethical guidelines, ensuring the integrity and transparency of our research process.

TABLE XII PREFERENCE CONSTRUCTION ANALYSIS. (A) PREFERENCE-SIGNAL SOURCE COMPARISON. (B) DATA-CONSTRUCTION STRATEGY ABLATION. ALL DPO RUNS USE SDXL AND THE SAME OPTIMIZATION BUDGET.

Method PickScore HPSv2 ImageReward

- (A) Preference-signal source comparison Baseline (SDXL) 57.82 32.61 0.84 w/ GPT-4o signal 59.12 32.98 0.92 w/ Image-generation-only reward model 66.12 33.46 1.04 w/ UnifiedReward (ours) 68.28 34.46 1.09

- (B) Data-construction strategy ablation Baseline (SDXL) 57.82 32.61 0.84 w/ Random selection 58.32 31.49 0.75 w/ Point score only 60.89 32.56 0.94 w/ Pair rank only 62.94 33.14 1.01 w/ Two-stage (ours) 68.28 34.46 1.09

under policy optimization. The qualitative comparison in Fig. 6 further supports this trend: relative to the FLUX baseline and other reward variants, samples optimized with UNIFIEDREWARD show better prompt faithfulness, cleaner compositions, and more coherent fine-grained details. This consistency between quantitative and qualitative results suggests

VI. CONCLUSION

This paper proposes UNIFIEDREWARD, the first unified reward model for multimodal understanding and generation assessment, capable of both pair ranking and point scoring, which can be utilized for vision model preference alignment. Specifically, we first fine-tune a pre-trained VLM on our constructed large-scale, comprehensive dataset that spans a wide range of visual tasks to develop UNIFIEDREWARD. This model is then employed to automatically construct high-quality preference pair data from the outputs of vision models through a two-stage filtering process, involving pair ranking and point sifting. These data are subsequently used for model preference alignment via direct preference optimization. Experimental results demonstrate that joint learning across diverse visual tasks yields significant mutual benefits. By applying our pipeline to both image and video understanding and generation tasks, we achieve substantial improvements in each domain.

REFERENCES

- [1] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik, “Diffusion model alignment using direct preference optimization,” in CVPR, 2024, pp. 8228–8238.
- [2] R. Liu, H. Wu, Z. Ziqiang, C. Wei, Y. He, R. Pi, and Q. Chen, “Videodpo: Omni-preference alignment for video diffusion generation,” arXiv preprint arXiv:2412.14167, 2024.
- [3] R. Zhang, L. Gui, Z. Sun, Y. Feng, K. Xu, Y. Zhang, D. Fu, C. Li, A. Hauptmann, Y. Bisk et al., “Direct preference optimization of video large multimodal models from language model reward,” arXiv preprint arXiv:2404.01258, 2024.
- [4] Y. Wang, Z. Tan, J. Wang, X. Yang, C. Jin, and H. Li, “Lift: Leveraging human feedback for text-to-video model alignment,” arXiv preprint arXiv:2412.04814, 2024.
- [5] T. Xiong, X. Wang, D. Guo, Q. Ye, H. Fan, Q. Gu, H. Huang, and C. Li, “Llava-critic: Learning to evaluate multimodal models,” arXiv preprint arXiv:2410.02712, 2024.
- [6] Y. Zang, X. Dong, P. Zhang, Y. Cao, Z. Liu, S. Ding, S. Wu, Y. Ma, H. Duan, W. Zhang et al., “Internlm-xcomposer2.5-reward: A simple yet effective multi-modal reward model,” arXiv preprint arXiv:2501.12368, 2025.
- [7] J. Liu, G. Liu, J. Liang, Z. Yuan, X. Liu, M. Zheng, X. Wu, Q. Wang, W. Qin, M. Xia et al., “Improving video generation with human feedback,” arXiv preprint arXiv:2501.13918, 2025.
- [8] K. Lee, H. Liu, M. Ryu, O. Watkins, Y. Du, C. Boutilier, P. Abbeel, M. Ghavamzadeh, and S. S. Gu, “Aligning text-to-image models using human feedback,” arXiv preprint arXiv:2302.12192, 2023.
- [9] R. Li, X. Wang, Y. Zhang, Z. Wang, and S. Yeung-Levy, “Temporal preference optimization for long-form video understanding,” arXiv preprint arXiv:2501.13919, 2025.
- [10] Y. Kirstain, A. Polyak, U. Singer, S. Matiana, J. Penna, and O. Levy, “Pick-a-pic: An open dataset of user preferences for text-to-image generation,” NeurIPS, vol. 36, pp. 36652–36663, 2023.
- [11] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li, “Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,” arXiv preprint arXiv:2306.09341, 2023.
- [12] J. Xu, Y. Huang, J. Cheng, Y. Yang, J. Xu, Y. Wang, W. Duan, S. Yang, Q. Jin, S. Li et al., “Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation,” arXiv preprint arXiv:2412.21059, 2024.
- [13] K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu, “T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation,” NeurIPS, vol. 36, pp. 78723–78747, 2023.
- [14] Y. Liu, X. Cun, X. Liu, X. Wang, Y. Zhang, H. Chen, Y. Liu, T. Zeng, R. Chan, and Y. Shan, “Evalcrafter: Benchmarking and evaluating large video generation models,” in CVPR, 2024, pp. 22139–22149.
- [15] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in CVPR, 2024, pp. 21807–21818.
- [16] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” NeurIPS, vol. 30, 2017.
- [17] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021, pp. 8748–8763.
- [18] J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong, “Imagereward: Learning and evaluating human preferences for text-toimage generation,” NeurIPS, vol. 36, pp. 15903–15935, 2023.
- [19] S. Zhang, B. Wang, J. Wu, Y. Li, T. Gao, D. Zhang, and Z. Wang, “Learning multi-dimensional human preference for text-to-image generation,” in CVPR, 2024, pp. 8018–8027.
- [20] Y. Liang, J. He, G. Li, P. Li, A. Klimovskiy, N. Carolan, J. Sun, J. PontTuset, S. Young, F. Yang et al., “Rich human feedback for text-to-image generation,” in CVPR, 2024, pp. 19401–19411.
- [21] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [22] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.
- [23] X. He, D. Jiang, G. Zhang, M. Ku, A. Soni, S. Siu, H. Chen, A. Chandra, Z. Jiang, A. Arulraj et al., “Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation,” arXiv preprint arXiv:2406.15252, 2024.

- [24] Z. Sun, S. Shen, S. Cao, H. Liu, C. Li, Y. Shen, C. Gan, L.-Y. Gui, Y.-X. Wang, Y. Yang et al., “Aligning large multimodal models with factually augmented rlhf,” arXiv preprint arXiv:2309.14525, 2023.
- [25] D. Ahn, Y. Choi, Y. Yu, D. Kang, and J. Choi, “Tuning large multimodal models for videos using reinforcement learning from ai feedback,” arXiv preprint arXiv:2402.03746, 2024.
- [26] A. Gunjal, J. Yin, and E. Bas, “Detecting and preventing hallucinations in large vision language models,” in AAAI, vol. 38, 2024, pp. 18 135–18 143.
- [27] Z. Zhao, B. Wang, L. Ouyang, X. Dong, J. Wang, and C. He, “Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization,” arXiv preprint arXiv:2311.16839, 2023.
- [28] H. Furuta, H. Zen, D. Schuurmans, A. Faust, Y. Matsuo, P. Liang, and S. Yang, “Improving dynamic object interactions in text-to-video generation with ai feedback,” arXiv preprint arXiv:2412.02617, 2024.
- [29] J. Li, Q. Long, J. Zheng, X. Gao, R. Piramuthu, W. Chen, and W. Y. Wang, “T2v-turbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design,” arXiv preprint arXiv:2410.05677, 2024.
- [30] H. Yuan, Z. Chen, K. Ji, and Q. Gu, “Self-play fine-tuning of diffusion models for text-to-image generation,” arXiv preprint arXiv:2402.10210, 2024.
- [31] J. Zhang, J. Wu, W. Chen, Y. Ji, X. Xiao, W. Huang, and K. Han, “Onlinevpo: Align video diffusion model with online video-centric preference optimization,” arXiv preprint arXiv:2412.15159, 2024.
- [32] S. Han, H. Fan, J. Fu, L. Li, T. Li, J. Cui, Y. Wang, Y. Tai, J. Sun, C. Guo et al., “Evalmuse-40k: A reliable and fine-grained benchmark with comprehensive human annotations for text-to-image generation model evaluation,” arXiv preprint arXiv:2412.18150, 2024.
- [33] D. Christodoulou and M. Kuhlmann-Jørgensen, “Finding the subjective truth: Collecting 2 million votes for comprehensive gen-ai model evaluation,” 2024. [Online]. Available: https://arxiv.org/abs/2409.11904
- [34] R. Zhang, L. Gui, Z. Sun, Y. Feng, K. Xu, Y. Zhang, D. Fu, C. Li, A. Hauptmann, Y. Bisk et al., “Direct preference optimization of video large multimodal models from language model reward,” arXiv preprint arXiv:2404.01258, 2024.
- [35] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, Y. Li, Z. Liu, and C. Li, “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.
- [36] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, vol. 33, pp. 6840–6851, 2020.
- [37] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.
- [38] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li, “Video instruction tuning with synthetic data,” arXiv preprint arXiv:2410.02713, 2024.
- [39] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952,

- 2023.

[40] L. Li, Y. Wei, Z. Xie, X. Yang, Y. Song, P. Wang, C. An, T. Liu, S. Li, B. Y. Lin et al., “Vlrewardbench: A challenging benchmark for visionlanguage generative reward models,” arXiv preprint arXiv:2411.17451,

- 2024.

- [41] D. Jiang, M. Ku, T. Li, Y. Ni, S. Sun, R. Fan, and W. Chen, “Genai arena: An open evaluation platform for generative models,” arXiv preprint arXiv:2406.04485, 2024.
- [42] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” NeurIPS, 2023.
- [43] Y. Lu, D. Jiang, W. Chen, W. Y. Wang, Y. Choi, and B. Y. Lin, “Wildvision: Evaluating vision-language models in the wild with human preferences,” arXiv preprint arXiv:2406.11069, 2024.
- [44] B. Li, K. Zhang, H. Zhang, D. Guo, R. Zhang, F. Li, Y. Zhang, Z. Liu, and C. Li, “Llava-next: Stronger llms supercharge multimodal capabilities in the wild,” May 2024. [Online]. Available: https: //llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/
- [45] C. White, S. Dooley, M. Roberts, A. Pal, B. Feuer, S. Jain, R. ShwartzZiv, N. Jain, K. Saifullah, S. Naidu et al., “Livebench: A challenging, contamination-free llm benchmark,” arXiv preprint arXiv:2406.19314, 2024.
- [46] Z. Sun, S. Shen, S. Cao, H. Liu, C. Li, Y. Shen, C. Gan, L.-Y. Gui, Y.-X. Wang, Y. Yang et al., “Aligning large multimodal models with factually augmented rlhf,” arXiv preprint arXiv:2309.14525, 2023.
- [47] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in ECCV. Springer, 2024, pp. 216–233.

- [48] Y. S. Y. Q. M. Zhang, X. L. J. Y. X. Zheng, K. L. X. S. Y. Wu, R. J. C. Fu, and P. Chen, “Mme: A comprehensive evaluation benchmark for multimodal large language models,” arXiv preprint arXiv:2306.13394, 2021.
- [49] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” arXiv preprint arXiv:2310.02255, 2023.
- [50] M. Mathew, R. Tito, D. Karatzas, R. Manmatha, and C. Jawahar, “Document visual question answering challenge 2020,” arXiv preprint arXiv:2008.08899, 2020.
- [51] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in CVPR, 2019, pp. 8317–8326.
- [52] B. Li, P. Zhang, K. Zhang, F. Pu et al., “Lmms-eval: Accelerating the development of large multimoal models,” March 2024. [Online]. Available: https://github.com/EvolvingLMMs-Lab/lmms-eval
- [53] J. Xu, T. Mei, T. Yao, and Y. Rui, “Msr-vtt: A large video description dataset for bridging video and language,” in CVPR, 2016, pp. 5288–5296.
- [54] W. F. Hendria, “Msvd-indonesian: A benchmark for multimodal videotext tasks in indonesian,” arXiv preprint arXiv:2306.11341, 2023.
- [55] Y. Li, Y. Song, L. Cao, J. Tetreault, L. Goldberg, A. Jaimes, and J. Luo, “Tgif: A new dataset and benchmark on animated gif description,” in CVPR, 2016, pp. 4641–4650.
- [56] H. Duan, J. Yang, Y. Qiao, X. Fang, L. Chen, Y. Liu, X. Dong, Y. Zang, P. Zhang, J. Wang et al., “Vlmevalkit: An open-source toolkit for evaluating large multi-modality models,” in ICME, 2024, pp. 11198– 11201.
- [57] H. Wu, D. Li, B. Chen, and J. Li, “Longvideobench: A benchmark for long-context interleaved video-language understanding,” NeurIPS, vol. 37, pp. 28828–28857, 2025.
- [58] J. Zhou, Y. Shu, B. Zhao, B. Wu, S. Xiao, X. Yang, Y. Xiong, B. Zhang, T. Huang, and Z. Liu, “Mlvu: A comprehensive benchmark for multi-task long video understanding,” arXiv preprint arXiv:2406.04264, 2024.
- [59] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” arXiv preprint arXiv:2405.21075, 2024.
- [60] J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan et al., “Scaling autoregressive models for content-rich text-to-image generation,” arXiv preprint arXiv:2206.10789, vol. 2, no. 3, p. 5, 2022.
- [61] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.
- [62] R. Islam and O. M. Moushi, “Gpt-4o: The cutting-edge advancement in multimodal llm,” Authorea Preprints, 2024.
- [63] Z. Sun, S. Shen, S. Cao, H. Liu, C. Li, Y. Shen, C. Gan, L.-Y. Gui, Y.-X. Wang, Y. Yang, K. Keutzer, and T. Darrell, “Aligning large multimodal models with factually augmented rlhf,” arXiv preprint arXiv:2309.14525, 2023.
- [64] DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025. [Online]. Available: https: //arxiv.org/abs/2501.12948
- [65] Black Forest Labs, “Flux,” 2024. [Online]. Available: https://github.com/ black-forest-labs/flux

APPENDIX A MORE IMPLEMENTATION DETAILS A. Reward Model Baselines

knowledge, this is the first unified reward model for multimodal understanding and generation assessment.

w/ UnifiedReward (Ours)

SDXL-Turbo w/ Pick-a-Pic

PickScore [10] is an image generation assessment model trained over Pick-a-Pic by combining a CLIP-style model with a variant of InstructGPT’s reward model objective. This work employs its checkpoint “yuvalkirstain/PickScore v1” as one of the image generation reward model baselines.

[Figure 65]

[Figure 66]

[Figure 67]

HPSv2 [11] is an image generation scoring model based on CLIP, fine-tuned on the HPD v2 [33] dataset. It is capable of predicting human preferences for generated images. We utilize its official code and checkpoint for evaluation.

Darth Vader playing with raccoon in Mars during sunset.

[Figure 68]

[Figure 69]

[Figure 70]

ImageReward [18] is a text-to-image human preference reward model designed to effectively encode human preferences. It is trained based on a systematic annotation pipeline that includes both rating and ranking, collecting 137k expert comparisons. We utilize its official code and checkpoint for evaluation.

A horse riding an astronaut.

LLaVA-Critic [5] is designed to assess image understanding performance based on the LLM, enabling pair ranking and point scoring. It is trained on a high-quality critic instructionfollowing dataset that incorporates diverse evaluation criteria and scenarios. In this work, we employ the “lmms-lab/llavacritic-7b” model as our baseline for image understanding assessment.

[Figure 71]

[Figure 72]

[Figure 73]

a colossal metallic robot with an angular design stands atop a modern glass building nestled on a mountain peak. the robot dons a pair of gigantic headphones with neon lights, while a vivid sunset paints the sky and surrounding mountains with hues of orange and purple.

VideoScore [23] is a video quality assessment model, trained on the VideoFeedback dataset, which contains human-provided multi-aspect scores for 37.6K synthesized videos generated by 11 existing video generative models. We utilize its official code and checkpoint for video quality assessment evaluation. LiFT [4] is the first fine-tuning method that leverages human feedback for T2V model alignment. It constructs a Human Rating Annotation dataset, LiFT-HRA, consisting of approximately 20k human annotations, each including a score and its corresponding reason. Based on this dataset, a reward model, LiFT-Critic, is trained to learn a human feedback-based reward function. In this work, we utilize the released code and checkpoint of LiFT-Critic for video generation quality assessment.

Fig. 8. More Qualitative DPO Comparison on SDXL-Turbo. We compare the qualitative performance of the original SDXL-Turbo, DPO trained on Pick-a-Pic dataset, and DPO trained with UnifiedReward.

B. Evaluation Benchmarks

Multimodal Understanding: We evaluate the image and video understanding assessment of UNIFIEDREWARD on VLRewardBench [40] and ShareGPTVideo [34] (1K samples for testing), respectively. Multimodal Generation: GenAIBench [41] includes both image and video generation reward benchmarks, which are utilized. Besides, we also employ VideoGen-RewardBench [7] for video generation assessment benchmark.

VisionReward [12] is a fine-grained, multi-dimensional reward model designed to capture human preferences in images and videos. It constructs separate human preference datasets for images and videos, and trains corresponding reward models for each. In our work, we utilize its image and video reward models for evaluating image and video generation assessment, respectively.

1) Multimodal Understanding: VLRewardBench [40] is a comprehensive benchmark for assessing image understanding, covering general multimodal queries, visual hallucination detection, and complex reasoning tasks. It consists of 1,250 high-quality examples meticulously designed to evaluate model limitations and challenge their capabilities. During evaluation, we randomly shuffle the order of responses to ensure more robust and reliable assessment results.

VideoReward [7] is a multi-dimensional video reward model trained on a newly proposed 182k-sized human-labeled video generation preference dataset, sourced from 12 video generation models. We utilize its official code and checkpoint for evaluation.

ShareGPTVideo [34] is an open-source, large-scale training dataset comprising 900k captions that cover a diverse range of video content, including temporal dynamics, world knowledge, object attributes, and spatial relationships. It also includes 17k preference data specifically curated for DPO training. In this work, we utilize 16k preference data for reward model training and 1k for video understanding evaluation.

Our UnifiedReward is based on LLaVA-OneVision-7B (OV7B) [35] and trained on our constructed large-scale, comprehensive human feedback dataset, which spans a wide range of visual tasks. Through joint multi-task learning and evaluation, our experimental results demonstrate that this approach fosters a mutually reinforcing effect across tasks. To the best of our

A young alchemist crafting a potion, causing the air around her to ripple with arcane energy.

A student sitting alone in a library, light flipping through a book late at night.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

T2V-Turbo

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

w/ VideoDPO

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

w/ UnifiedReward (Ours)

A chef carefully plating a Michelin-star dish in a fine dining restaurant.

A busker playing the violin on a busy shopping street.

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

T2V-Turbo

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

w/ VideoDPO

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

w/ UnifiedReward (Ours)

- Fig. 9. More Qualitative DPO Comparison on T2V-Turbo. We compare the qualitative performance of the original T2V-Turbo, DPO trained with VideoDPO, and DPO trained with UnifiedReward.

2) Multimodal Generation: GenAI-Bench [41] is a reward benchmark for multimodal generative models, designed to assess the ability of MLLMs to evaluate AI-generated content by comparing their judgments with human preferences. It includes benchmarks for image generation, image editing, and video generation. In this work, we utilize the image and video generation parts for generation reward evaluation.

VideoGen-RewardBench [7] builds upon VideoGen-Eval to establish a fair benchmark for assessing the performance of reward models on modern T2V models. It comprises 26.5k manually constructed video pairs, with annotators evaluating each pair based on Visual Quality, Motion Quality, Text Alignment, and Overall Quality. In this work, we utilize the Overall Quality metric for baseline reward comparison.

We will release all evaluation codes to facilitate community reproduction.

C. DPO Baselines

LLaVA-Critic [5] leverages image-question pairs from LLaVARLHF [63] to construct preference data for OV-7B DPO which is trained for 3 epochs. In this work, for a fair comparison, we also use the image-question pairs from LLaVA-RLHF to construct preference data while keeping all other settings the same.

LLaVA-Houd-DPO [3] utilizes the 17k preference data from the ShareGPTVideo [34] dataset for DPO training. In this work, to ensure a fair comparison, we apply the same dataset for DPO training on LLaVA-Video [38] following its method as the baseline. For our approach, we randomly sample 14k data points from the 17k dataset to construct the DPO training data and then perform DPO on LLaVA-Video. All training parameters and settings are kept identical to maintain fairness in evaluation.

LLaVA-TPO [9] adopts a self-training approach that enables models to distinguish between well-grounded and less accurate

GPT Evaluation Prompt

system, must be carefully designed to ensure fairness and robustness. There is always a risk that biases in the training data could influence model predictions. However, we have taken measures to curate a diverse dataset and will continue refining our approach to mitigate such concerns. Overall, we believe our work contributes positively to the AI field by providing a more effective and scalable way to align vision models with human preferences. We encourage future research and collaborations to further enhance the fairness, adaptability, and real-world applicability of reward-based AI evaluation.

System:

You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:

-----##INSTRUCTIONS:

- - Focus on the meaningful match between the predicted answer and the correct answer.
- - Consider synonyms or paraphrases as valid matches.
- - Evaluate the correctness of the prediction compared to the answer.

###### User:

Please evaluate the following video-based question-answer pair: Question: {question} Correct Answer: {answer} Predicted Answer: {pred} Provide your evaluation only as a yes/no and score where the score is an integer value between

0 and 5, with 5 indicating the highest meaningful match. Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.

- Fig. 10. GPT Evaluation Prompt. We use “gpt-3.5-turbo-1106” for video understanding evaluation on MSRVTT, MSVD, and TGIF benchmarks.

temporal responses by leveraging curated preference datasets at two granularities: localized temporal grounding and comprehensive temporal grounding. In this work, since its training dataset has not been open-sourced, we utilize its released checkpoint for comparison.

VideoDPO [2] is the first video generation DPO method built upon its comprehensive preference scoring system, OmniScore, which evaluates both the visual quality and semantic alignment of generated videos. In this work, we use its released preference dataset for T2V-Turbo [29] DPO as a baseline. For our method, we extract video-caption pairs from its dataset to construct our own preference data for DPO, ensuring a fair evaluation.

Pick-a-Pic [10] is a large, open dataset of text-to-image prompts paired with real user preferences over generated images. After excluding approximately 12% of tied pairs, the dataset contains around 851k preference pairs with 58.9k unique prompts. In this work, we directly use this dataset for SDXL-Turbo [39] DPO as a baseline. For our method, we randomly sample 14k captions from this dataset to construct preference data for DPO, ensuring a fair evaluation.

APPENDIX B MORE QUALITATIVE COMPARISON

More qualitative results are shown in Figs. 8 and 9.

APPENDIX C SOCIETAL IMPACTS

Our unified reward model for multimodal understanding and generation assessment has the potential to significantly enhance AI applications across various domains. By aligning AI-generated content more closely with human preferences, our work can improve the quality and reliability of vision models, benefiting industries such as digital media, entertainment, education, and accessibility. For example, one of the key advantages of our approach is its ability to provide a more consistent and interpretable evaluation of generative models. This can lead to better AI-assisted creativity, enabling artists, designers, and content creators to generate higher-quality visuals with greater control. While our work brings many benefits, we recognize that reward models, like any AI-driven

