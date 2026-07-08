# arXiv:2406.15252v3[cs.CV]14Oct2024

## VIDEOSCORE: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation

1,2Xuan He∗† 1Dongfu Jiang∗† 1,3Ge Zhang 1Max Ku 1Achint Soni 1Sherman Siu 1Haonan Chen 1Abhranil Chandra 1Ziyan Jiang 1Aaran Arulraj 4Kai Wang 1Quy Duc Do 1Yuansheng Ni 2Bohan Lyu 1Yaswanth Narsupalli 1Rongqi Fan 1Zhiheng Lyu 5Bill Yuchen Lin 1Wenhu Chen †

1University of Waterloo, 2Tsinghua University, 3Stardust.AI, 4University of Toronto, 5AI2

∗ Equal Contribution † {x36he, dongfu.jiang, wenhuchen}@uwaterloo.ca

https://tiger-ai-lab.github.io/VideoScore/

Results

|VideoScore|
|---|

|VideoFeedback|
|---|

80

Spearman CorrelationPairwise Accuracy

Evaluation Aspects

Regression

Generation

60

VidProM VQ TC DD TVA FC

|Visual Quality: 3<br><br>Temporal Consistency: 2 Dynamic Degree: 2 Text-to-Video Alignment: 4 Factual Consistency: 4|
|---|

CLIP-sim

40

[Figure 1]

CLIP-Score

Visual Quality

|3.87|
|---|

|1.96|
|---|

|2.89|
|---|

|2.03|
|---|

|3.69|
|---|

20

T2V Models

Temporal Consistency Dynamic Degree Text-to-Video Alignment

Gemini-1.5-Pro

0

GPT-4o/V

[Figure 2]

regression head

-20

###### VideoScore

lmhead

Videos

Factual Consistency

VideoFeedbak-test EvalCrafter

100

Language Model

[Figure 3]

80

Rater Training IAA

CLIP-sim

|⋯|
|---|

60

Prompt F1 F2 F3

F4

CLIP-Score

[Figure 4]

40

Gemini-1.5-Pro

Review Annotation Augment VideoFeedback

[Figure 5]

[Figure 6]

20

GPT-4o/V

Image Encoder

| |[Figure 7]<br><br>⋯|
|---|---|
| | |

[Figure 8]

VideoScore

0

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

GenAI-Bench VBench

Frame 1 Frame 2 Frame 3

Figure 1: Construction process of VIDEOFEEDBACK dataset and illustration of VIDEOSCORE.

### Abstract

The recent years have witnessed great advances in video generation. However, the development of automatic video metrics is lagging significantly behind. None of the existing metrics are able to provide reliable scores over generated videos. The main barrier is the lack of largescale human-annotated datasets. In this paper, we release VIDEOFEEDBACK, the first largescale dataset containing human-provided multiaspect score over 37.6K synthesized videos from 11 existing video generative models. We train VIDEOSCORE (initialized from Mantis) based on VIDEOFEEDBACK to enable automatic video quality assessment. Experiments show that the Spearman correlation between VIDEOSCORE and humans can reach 77.1 on VIDEOFEEDBACK-test, beating the prior best metrics by about 50 points. Further results on other held-out EvalCrafter, GenAI-Bench, and VBench show that VIDEOSCORE has consistently much higher correlation with human judges than other metrics. Due to these results, we believe VIDEOSCORE can serve as a great proxy for human raters to (1) rate different video models to track progress (2) simulate fine-grained human feedback in Reinforcement Learning with Human Feedback (RLHF) to improve current video generation models.

### 1 Introduction

Powerful text-to-video (T2V) generative models have been exponentially emerging these days. In 2023 and 2024, we have witnessed an array of T2V models like Sora (OpenAI, 2024b), Runway Gen-2 (Esser et al., 2023), Lumiere (Bar-Tal et al., 2024), Pika1, Luma-AI2, Kling3, Emu-video (Girdhar et al., 2023), StableVideoDiffusion (Blattmann et al., 2023a). These models have shown their potential to generate longer-duration, higher-quality, and more natural videos. Despite the great progress video generative models have made, they still suffer from artifacts like unnaturalness, inconsistency and hallucination, which calls for reliable fine-grained metrics for evaluation and a robust reward model for reinforcement learning (RLHF).

The recent literature has adopted a wide range of metrics to evaluate videos. However, these metrics suffer from the following issues: (1) some of them are computed over distributions and cannot be adopted to evaluate a single model output. Examples include FVD (Unterthiner et al., 2019) and

- 1https://pika.art/home
- 2https://lumalabs.ai/dream-machine
- 3https://kling.kuaishou.com/

IS (Salimans et al., 2016). (2) most metrics can only be used to evaluate visual quality or text alignment, while failing on other aspects like motion smoothness, factual consistency, etc. Examples of such metrics include CLIP (Radford et al., 2021b), DINO (Caron et al., 2021), BRISQUE (Mittal et al., 2012a). (2) some metrics focus only on a single mean opinion score (MOS), failing to provide finegrained subscores across different multiple aspects. Examples include T2VQA (Kou et al., 2024b), FastVQA (Wu et al., 2022), and DOVER (Wu et al.,

- 2023a). (3) Several works (Ku et al., 2023; Bansal et al., 2024) propose to prompt multi-modal largelanguage-models (MLLM) like GPT-4o (Achiam et al., 2023) or Gemini-1.5 (Reid et al., 2024) to produce multi-aspect quality assessment for given videos. However, our experiments show that they also have low correlation with humans. These feature-based metrics or MLLM prompting methods either fail to provide reliable evaluation or cannot simulate the human feedback from real world well, which have lagged behind and restricted us from training better video generative models.

Since obtaining large-scale human feedback is highly costly, we can try to approximate humanprovided scores with model-based metrics. To this end, our work can be divided into two parts: (1) curating VIDEOFEEDBACK, the first largescale dataset containing human-provided scores for 37.6K videos, (2) training VIDEOSCORE on VIDEOFEEDBACK, which is an automatic video metric to simulate human feedback.

In preparation of VIDEOFEEDBACK, we solicit prompts from VidProM (Wang and Yang, 2024) and use 11 popular text-to-video models, including Pika, Lavie (Wang et al., 2023c), SVD (Blattmann

- et al., 2023a), etc, to generate videos of various quality based on these prompts. We define five key aspects for evaluation as shown in Table 2, and get our videos from VIDEOFEEDBACK annotated for each aspect from 1 (bad) to 4 (perfect).

To build the video evaluator, we select MantisIdefics2-8B (Jiang et al., 2024a) as our main backbone model due to its superior ability to handle multi-image and video content, accommodating up to 128 video frames and supporting native resolution. After fine-tuning Mantis on VIDEOFEEDBACK-train, we get our video evaluator, VIDEOSCORE. Experiments show that we achieve a Spearman correlation of 77.1 on VIDEOFEEDBACK-test and 59.5 on EvalCrafter (Liu et al.,

- 2023b) for the text-to-video alignment aspect, sur-

passing the best baseline by 54.1 and 4.4 respectively. The pairwise comparison accuracy gets 78.5 on GenAI-Bench (Jiang et al., 2024b) video preference part, and 72.1 in average on 5 aspects of VBench (Huang et al., 2023), surpassing the previous best baseline by 11.4 and 9.6 respectively. Additional ablation studies with different backbone models confirmed that the Mantis-based metric provides a gain of 12.1 compared to using the Idefics2based metric. Due to the significant improvement, we believe that VIDEOSCORE can serve as the reliable metrics for future video generative models.

### 2 Related Work

#### 2.1 Text-to-Video Generative Models

Recent progress in diffusion models (Ho et al., 2020; Rombach et al., 2022) has significantly pushed forward the development of Text-to-Video (T2V) generation. Given a text prompt, the T2V generative model can synthesize new video sequences that didn’t previously exist (Wang et al., 2023c; OpenAI, 2024b; Chen et al., 2023a, 2024a; Henschel et al., 2024; Bar-Tal et al., 2024). Early diffusion-based video models generally build upon Text-to-Image (T2I) models, adding a temporal module to extend itself into the video domain (Wang et al., 2023c; Chen et al., 2023c). Recent T2V generation models are directly trained on videos from scratch. Among these, models based on Latent Diffusion Models (LDMs) have gained particular attention for their effectiveness and efficiency (Zhou et al., 2022; An et al., 2023; Blattmann et al., 2023b). While the other works used the pixel-based Diffusion Transformers (DiT) also achieve quality results (Gupta et al., 2023; Chen et al., 2023b; OpenAI, 2024b).

#### 2.2 Video Quality Assessment

As the current progress of Text-to-Video generative models leaves it uncertain how close we are to reaching the objective, researchers have worked on evaluation methods to benchmark the generative models. Common methods involve the use of FVD (Unterthiner et al., 2018) and CLIP (Radford et al., 2021a) to evaluate the quality of frames and the text-frames alignment respectively. However, other aspects like subject consistency, temporal consistency, factual consistency cannot be captured by these metrics. Recent works like VBench (Huang et al., 2023) proposes to use different DINO (Caron et al., 2021), optical flow (Horn

and Schunck, 1981) to reflect these aspects. However, the correlation with human judgment is relatively low. For example, most models have subject/background consistency scores over 97% in VBench, which is a massive overestimation of the current T2V models’ true capability. Another work EvalCrafter (Liu et al., 2023b) instead resorts to human raters to perform comprehensive evaluation.

A recent work VideoPhy (Bansal et al., 2024) follows VIEScore (Ku et al., 2023) prompt large multi-modal models like Gemini (Reid et al., 2024) and GPT-4o (Achiam et al., 2023) to provide quality assessment. However, our later study shows that these multimodal language models also achieve very low agreement with human raters. A concurrent work T2VQA (Kou et al., 2024a) also proposes to train a quality assessment model on humanannotated video ratings. However, there are a few distinctions. Firstly, our dataset contains ratings for multiple aspects. Secondly, our dataset is 4x larger than the T2VQA dataset. Thirdly, our metric is built on pre-trained video-language foundation models to maximize its performance.

#### 2.3 RLHF in image/video generation

In recent years, reinforcement learning from human feedback (RLHF) has emerged as a significant approach to enhancing the performance of image/video generative models. Numerous studies have focused on training reward models with large datasets of image-text pairs, such as HPSv2 (Wu

- et al., 2023b), ImageReward (Xu et al., 2023), RichHF-18K (Liang et al., 2023), or video-text pairs like T2V-Score (Wu et al., 2024). Moreover, some research efforts have combined existing reward models to simulate human feedback, such as T2V-Turbo (Li et al., 2024c), while some recent works (Wang et al. (2024b), Wang et al. (2024a)) proposed multi-objective reward model with regression head to provide human preferences. Utilizing these reward models or feedback simulators, diverse methods have been proposed to align the output of visual generative models with human preferences, including RL-based methods (Fan et al.

(2024), Zhang et al. (2024)) and reward fine-tuning methods (Clark et al. (2023), Li et al. (2024b), Yuan et al. (2023)). Additionally, some works adopt data distillation to fine-tune diffusion models on highquality data, while others like Diffusion-DPO (Wallace et al., 2023), extend Direct Preference Optimization (DPO) to train diffusion models based on preference data. Our VIDEOSCORE aims to ap-

proximate human feedback, which is expected to be beneficial in enhancing video generative models with RLHF methods like PPO or DPO.

### 3 VIDEOFEEDBACK

This section introduces the construction process of our dataset, VIDEOFEEDBACK. We start by explaining how we gathered and filtered diverse text prompts for video generation, followed by the video-generation processes using 11 selected textto-video models. Next, we outline the annotation pipeline that guides raters to score videos across multiple aspects defined in Table 2. We also include supplementary data to enhance robustness. Finally, we summarize the dataset statistics in Table 1, with 760 examples as the test set.

#### 3.1 Data preparation

Prompt Sources We utilize VidProM (Wang and Yang, 2024), a dataset containing extensive textto-video pairs from different models. VidProM’s video-generation prompts are diverse and semantically rich, derived from real-world user inputs. To create a manageable subset from the 1.04 million unique prompts, we apply two filters: a length filter and an NSFW filter. The length filter eliminates prompts with fewer than 5 words or more than 100 words. The NSFW filter removes prompts with a high probability of containing inappropriate content. After filtering, we perform random down-sampling to obtain a set of 44.5K prompts, 31.6K of them are used in video generation and some videos may have the same text prompt.

Video Generation We select 11 text-to-video (T2V) generative models (shown in Table 1) with various capabilities so that the quality of the generated video ranges from high to low in a balanced way. Some videos are pregenerated in the VidProM dataset, including Pika, Text2Video-Zero (Khachatryan et al., 2023), VideoCrafter2 (Chen et al., 2024a), and ModelScope (Wang et al., 2023a), whereas the others are generated by ourselves or collected from the Internet (i.e. SoRA). To eliminate differences between models in subsequent annotation stage, we normalize the videos into a unified format. First, we standardized the frame rate to 8 fps to address discrepancies in temporal consistency between high and low fps videos. Specifically, for high frame rate model Pika and AnimateDiffusion (Guo et al., 2023) we use frame down sampling, while for

Base Model or Video Type Video Source Total Size Resolution Duration FPS Score Human Annotated Videos

Pika VidProM 4.6k (768, 480) 3.0s 8 [1-4] Text2Video-Zero (Khachatryan et al., 2023) VidProM 4.6k (512,512) 2.0s 8 [1-4] VideoCrafter2 (Chen et al., 2024a) VidProM 4.9k (512, 320) 2.0s 8 [1-4] ModelScope (Wang et al., 2023a) VidProM 4.5k (256, 256) 2.0s 8 [1-4] LaVie-base (Wang et al., 2023c) Generated 3.2k (512, 320) 2.0s 8 [1-4] AnimateDiff (Guo et al., 2023) Generated 1.4k (512, 512) 2.0s 8 [1-4] LVDM (He et al., 2022) Generated 3.1k (256, 256) 2.0s 8 [1-4] Hotshot-XL (Mullan et al., 2023) Generated 3.2k (512, 512) 1.0s 8 [1-4] ZeroScope-576w (Sterling, 2024) Generated 2.2k (256, 256) 2.0s 8 [1-4] Fast-SVD (Blattmann et al., 2023a) Generated 1.0k (1024, 576) 3.0s 8 [1-4] SoRA-Clip (OpenAI, 2024b) Collected 0.9k various 2.0/3.0s 8 [1-4]

Augmented Videos

DiDeMo (Hendricks et al., 2017) Real 2.0k various 2.0/3.0s 8 4 Panda70M (Chen et al., 2024b) Real 2.0k various 2.0/3.0s 8 4

Table 1: Statistics of our curated VIDEOFEEDBACK for training video-generation evaluator. It consists of 33.6K human-scored videos across multiple aspects, with 4k real-world videos collected from DiDeMo (Hendricks et al., 2017) and Panda70M (Chen et al., 2024b) as the supplementary data. Ultimately, we get 37.6K high-quality rated videos as the final VIDEOFEEDBACK.

low frame rate model like Text2Video-Zero, we employed frame interpolation (Huang et al., 2022) on it. Details are shown in Appendix E. Additionally, we cropped Pika videos to remove the watermark, making them indistinguishable from other models. Ultimately, we obtained 33.6K videos from 11 T2V models, along with their generation prompts.

#### 3.2 Annotation Pipeline

Evaluation Dimensions As discussed in section 1, fine-grained and multi-aspect rating of videos is crucial for enhancing both the reliability and explainability of the video evaluator. Inspired by VBench (Huang et al., 2023) and EvalCrafter (Liu et al., 2023b), and FETV (Liu et al.,

- 2023c), we propose five key dimensions for textto-video evaluation, detailed in Table 2. These dimensions encompass both low-level vision aspects, such as Visual Quality, which evaluates basic visual impressions, and higher-level aspects, like Text-to-Video Alignment and Factual Consistency, which require a deep understanding of world knowledge, is a capability previous metrics do not have. Besides definition, a checklist for error points for each dimension is also provided to assist the rater in contributing more accurate and consistent rating. Detailed are provided in Table 9.

Annotation We hired 20 expert raters, with each rater performing rating for 1K-2K videos. Our raters are mostly college graduate students. For each aspect, there are three available ratings, 1

(Bad), 2 (Average), and 3 (Good), the score 4 (Perfect) is post-annotated, as described in the subsection 3.3. To ensure the consistency and quality of the annotations, we conducted a system training for each rater. Initially, we conducted a pilot training session with examples of multi-aspect ratings for various videos. Following this, multiple rounds of small-scale annotation were conducted to compute the inter-annotator agreement (IAA) across five aspects, as shown in Table 3. The results indicate a high score-matching ratio for all aspects, along with Fleiss’ κ (Fleiss and Cohen, 1973) and Krippendorff’s α (Krippendorff, 2011) metrics, with values around 0.4 or 0.5, suggesting sufficient agreement to proceed with large-scale annotation. The annotation process takes roughly 4 weeks to finish.

Review We conduct random checks on different raters during the annotating phase to ensure the alignment. Once we find the exceeded unqualified ratio in certain rater, we promptly communicate with the respective rater and review the annotations for that segment of the video. This helps calibrate the annotation provided by that rater during the relevant period. For example, we found several raters are too lenient and tend to give high scores to unqualified videos. We then step in to make sure they are aligned with our understanding of evaluation dimensions. With periodical random inspection on annotating, we completed the largescale annotation of 33.6K videos and moved to the

Aspect Definition Visual Quality (VQ) the quality of the video in terms of clearness, resolution, brightness, and color Temporal Consistency (TC) the consistency of objects or humans in video Dynamic Degree (DD) the degree of dynamic changes Text-to-Video Alignment (TVA) the alignment between the text prompt and the video content Factual Consistency (FC) the consistency of the video content with common-sense and factual knowledge

Table 2: The five evaluation aspects of VIDEOFEEDBACK and their definitions.

IAA metric VQ TC DD TVA FC

- Trial 1 (#=30)

Match Ratio 0.733 0.706 0.722 0.678 0.633 Kappa 0.369 0.414 0.413 0.490 0.265 Alpha 0.481 0.453 0.498 0.540 0.365

- Trial 2 (#=100)

Match Ratio 0.787 0.699 0.913 0.570 0.727 Kappa 0.088 0.562 0.565 0.125 -0.089 Alpha 0.078 0.579 0.620 0.205 -0.106

Table 3: Inter-Annotator Agreement (IAA) analysis results considering Matching Ratio, Fleiss’ κ, and Krippendorff’s α on the two trial annotations.

[Figure 15]

Figure 2: The rating distribution on all the videos.

data augmentation stage.

#### 3.3 Dataset Augmentation

To enhance the robustness of our dataset, we incorporated post-augmentation into the dataset. Firstly, expert raters will review the excellent videos (all aspects are scored 3) again to select perfect ones and raise their scoring to 4 (Perfect) in certain aspects, particularly among the SoRA and FastSVD (Blattmann et al., 2023a) videos.

Additionally, we gather 4k real-world videos from the DiDeMo (Hendricks et al., 2017) and Panda70M (Chen et al., 2024b) with each video accompanied by a text description. We select and cut clips from the ones less than 5 seconds to ensure a strong match between video and its text. We apply similar normalization in subsection 3.1 and also use SSIM and MSE between interval sampled frames to filter out the possible static videos, ensur-

ing the quality in Dynamic Degree. Finally the 4K real videos are scored 4 (perfect) in all aspects.

We plot the rating distributions across each dimension in Figure 2. which is balanced except for Dynamic Degree. We inspected in detail via case study and turned out this distribution is expected. Eventually, we get the final 37.6K examples as the training split of VIDEOFEEDBACK, and reserve 760 validation examples as test set.

### 4 Experiments

In this section, we describe our experiment setup, including baseline methods for video evaluation, and evaluation benchmarks for video evaluation. We also discuss the training details of VIDEOSCORE, and analyze our experiment results.

#### 4.1 Baselines

To compare with our evaluator model, we selected two categories of video quality metrics. The first category relies on statistical or neural features for evaluation. These metrics typically assess a single video dimension such as temporal consistency, and then yield a numerical value. The second category employs advanced MLLMs to evaluate videos across multiple dimensions. Extensive literature demonstrates that MLLMs not only excel in generating content on user instructions but also outperform traditional metrics in evaluating AI-generated content (AIGC). All baselines are listed in Table 4.

Feature-Based Metrics We list all the experimented metrics as follows:

- 1. Visual Quality. We use two no-reference image quality metrics PIQE (Venkatanath et al.,

2015) and BRISQUE (Mittal et al., 2012b). We apply them on all frames of video and take the average score across frames.

- 2. Temporal Consistency. In this dimension, CLIP-sim (Radford et al., 2021b) and DINOsim (Caron et al., 2021) are computed as cosine similarities of between adjacent frames features, following VBench (Huang et al.,

Method Visual Quality Temporal Dynamic Degree Text Alignment Factual Avgerage Random -3.1 0.5 0.4 1.1 2.9 0.4

###### Feature-basd automatic metrics

PIQE -17.7 -14.5 1.2 -3.4 -16.0 -10.1 BRISQUE -32.4 -26.4 -4.9 -8.6 -29.1 -20.3 CLIP-sim 21.7 29.1 -34.4 2.0 26.1 8.9 DINO-sim 19.4 29.6 -37.9 2.2 24.0 7.5 SSIM-sim 33.0 30.6 -31.3 4.7 30.2 13.4 MSE-dyn -20.3 -24.7 38.0 3.3 -23.9 -5.5 SSIM-dyn -31.4 -29.1 31.5 -5.3 -30.0 -12.9 CLIP-Score -10.9 -10.0 -14.7 -0.3 -0.3 -7.2 X-CLIP-Score -3.2 -2.7 -7.3 5.9 -2.0 -1.9

###### MLLM Propmting

- LLaVA-1.5-7B 9.4 8.0 -2.2 11.4 15.8 8.5

- LLaVA-1.6-7B -8.0 -4.1 -5.7 1.4 0.8 -3.1 Idefics2 4.2 4.5 8.9 10.3 4.6 6.5 Gemini-1.5-Flash 24.1 5.0 20.9 21.3 32.9 20.8 Gemini-1.5-Pro 35.2 -17.2 18.2 26.7 21.6 16.9 GPT-4o 13.6 17.6 28.2 25.7 30.2 23.0

###### Ours

VIDEOSCORE (gen) 86.2 80.3 77.6 59.4 82.1 77.1 VIDEOSCORE (reg) 84.7 81.5 68.4 59.5 84.6 75.7 ∆ over Best Baseline +51.0 +50.9 +39.6 +32.8 +51.7 +54.1

- Table 4: Correlation (Spearman’s ρ) between model answer and human reference on VIDEOFEEDBACK-test.

2023). Additionally, we calculate SSIM between adjacent frames, denoted as SSIM-sim.

- 3. Dynamic Degree. We uniformly sample four frames from the target video and calculate the average MSE (Mean Square Error) and SSIM (Wang et al., 2004) between adjacent frames in the sample as final score.
- 4. Text-to-Video Alignment. We include CLIPScore (Radford et al., 2021b) and X-CLIPScore (Ma et al., 2022) as metrics in this dimension. CLIP-Score calculates cosine similarity between the feature of each frame and the text prompt and then averages across all frames, while X-CLIP-Score utilizes the feature of video instead of frames.
- 5. Factual Consistency. It is challenging to find a feature-based metric to determine whether the visual content aligns with common sense. Therefore, we rely on the second category of metrics for this dimension.

We discretized the continuous outputs of these metrics to align with our labeling scores [1, 2, 3, 4]. For instance, for CLIP-sim, values are converted to: ’4’ if raw output in [0.97, 1], ’3’ if in [0.9, 0.97), ’2’ if in [0.8, 0.9) and ’1’ otherwise. See Table 12 for more details.

MLLM Prompting Based Metrics To understand how existing MLLMs perform on the multi-

aspect video evaluation task, we designed a prompting template in Table 10 to let them output scores ranging from 1 (Bad) to 4 (Perfect) for each aspect. However, some models, including Idefics2 (Laurençon et al., 2024), Fuyu (Adept AI, 2023), Kosmos-2 (Peng et al., 2023), and CogVLM (Wang et al., 2023b) and OpenFlamingo (Awadalla et al.,

- 2023), fail to give reasonable outputs. We thus exclude them from the tables. MLLMs that follow the output format like LLaVA-1.5 (Liu et al., 2023a), LLaVA-1.6 (Liu et al., 2024), Idefics1 (Laurençon et al., 2023), Google’s Gemini 1.5 (Reid et al.,
- 2024), and OpenAI’s GPT-4o (OpenAI, 2024a).

#### 4.2 Evaluation Benchmarks

We have included the following benchmarks to evaluate the ability of VIDEOSCORE and the abovementioned baselines on evaluating model generated videos to see their correlation with human raters.

VIDEOFEEDBACK-test As mentioned in section 3, we split 760 video entries from VIDEOFEEDBACK dataset, which contains 680 annotated videos and 80 augmented videos. We take label prediction accuracy and Spearman’s ρ in each dimension as evaluation indicators. For a specific aspect in the this set (e.g. Visual Quality), we use the predicted score from the same aspect to measure the performance for baselines and our models.

GenAI-Bench GenAI-Bench (Jiang et al., 2024b) is a benchmark designed to evaluate MLLM’s ability on preference comparison for tasks including text-to-video generation and others. The preference data is taken from GenAI-Arena from user voting. We select the video preference data in our experiments. This involves the MLLM judging which of the two provided videos is generally better, measured by pairwise accuracy. We use the averaged scores of the five aspects for MLLM prompting baselines and our models to give the preference. We compute the correlation between model-assigned preference vs. human preference as our indicator.

VBench VBench (Huang et al., 2023) is a comprehensive multi-aspect benchmark suite for video generative models, where they use a bunch of existing auto-metrics in each aspect. VBench have released a set of human preference annotations on all the aspects, comprising videos by 4 models, including ModelScope (Wang et al., 2023a), CogVideo (Hong et al., 2022), VideoCrafter1 (Chen et al., 2023a), and LaVie (Wang et al., 2023c). We select the subset from 5 aspects of VBench, like technical quality, subject consistency, and so on, to compute the preference comparison accuracy. For each aspect, we subsample 100 unique prompts in the testing. We use the averaged scores of the five aspects for MLLM prompting baselines and our models to predict the preference.

EvalCrafter EvalCrafter (Liu et al., 2023b) is a text-to-video benchmark across four dimensions: Video Quality, Temporal Consistency, Text-toVideo Alignment, and Motion Quality. We focused on the first three ones and gathered 2,541 videos by five models: Pika, Gen2, Floor33 (Floor33, 2024), ModelScope, and ZeroScope (Sterling, 2024). In EvalCrafter, human annotators rated each video on a scale of 1-5, with each scored by three raters. We calculated the average score across raters and normalized it to [0, 1]. After inference on benchmark videos, we excluded "Dynamic Degree" and "Factual Consistency" to match EvalCrafter’s dimensions. Finally, we used Spearman’s ρ in each dimension as an indicator.

#### 4.3 Training Details

For VIDEOSCORE, We use two scoring methods: generative scoring and regression scoring. Generative scoring involves training the model to output

fixed text forms, from which aspect scores are extracted using regular expressions. These scores are integers corresponding to human annotation scores. In contrast, regression scoring replaces the language model head with a linear layer that outputs 5 logits representing scores for each aspect. Regression scoring is trained using MSE loss.

We select Mantis-Idefics2-8B (Jiang et al., 2024a) as the base model, which can accommodate 128 video frames at most. The learning rate is set to 1e-5. Each model is trained for 1 epoch on 8 A100 (80G) GPUs, finishing in 6 hours.

#### 4.4 Evaluation Results

We report the Spearman correlation results on the VIDEOFEEDBACK-test and EvalCrafter in Table 4 and Table 6, respectively. For the preference comparison on videos, we report the pairwise accuracy on the GenAI-Bench and VBench in Table 5.

VIDEOSCORE achieves the SoTA performance On the VIDEOFEEDBACK-test, VIDEOSCORE gets an average of 54.1 improvements on all the five aspects compared to the baseline GPT4o. What’s more, on the EvalCrafter benchmark, VIDEOSCORE (reg) has 4.4 improvements on textto-video alignment. For pairwise preference comparison, VIDEOSCORE also gets 78.5 accuracy on GenAI-Bench, surpassing the second-best Gemini1.5-Flash by 11.4 points. on the Vbench, our model archives the highest pairwise accuracy on 4 out of 5 aspects from VBench, with an average of 16.1 improvements.

Feature-based Automatic Metrics are limited While some feature-based automatic metrics are good at a single aspect, they might fail to evaluate well on others. For example, on the VIDEOFEEDBACK-test, the correlation scores of SSIM-dyn and MSE-dyn achieve 31.5 and 38.0 for the dynamic degree aspect, but they both get a negative correlation for others. Besides, PIQE, BRISQUE, CLIPScore, and X-CLIP-Score get nearly all negative correlations for all 5 aspects. This proves the image quality assessment metrics cannot be easily adapted to the video quality assessment task.

#### 4.5 Best-of-K Sampling with VIDEOSCORE

While best-of-K sampling has proven effective in boosting the performance of LLMs (Li et al., 2023), it’s still unknown whether it also works for video generation tasks. To investigate this problem and also better show the effectiveness of VIDEOSCORE,

Benchmark → GenAI-Bench VBench Model ↓ Sub-Aspect →

Overall Consistency Random 37.7 44.5 42.0 37.3 40.5 44.8

Video Preference

Technical Quality

Subject Consistency

Dyanmic Degree

Motion Smoothness

###### Feature-based Automatic Metrics

PIQE 34.5 60.8 44.3 71.0 45.3 53.8 BRISQUE 38.5 56.7 41.2 75.5 41.2 54.2 CLIP-sim 34.1 47.8 46.0 34.8 44.7 44.2 DINO-sim 31.4 49.5 51.2 24.7 55.5 41.7 SSIM-sim 28.4 30.7 46.2 24.5 54.2 27.2 MSE-dyn 34.2 32.8 31.7 81.7 31.2 39.2 SSIM-dyn 38.5 37.5 36.3 84.2 34.7 44.5 CLIP-Score 45.0 57.8 46.3 71.3 47.0 52.2 X-CLIP-Score 41.4 44.0 38.0 51.0 28.7 39.0

###### MLLM Prompting

- LLaVA-1.5-7B 49.9 42.7 42.3 63.8 41.3 8.8

- LLaVA-1.6-7B 44.5 38.7 26.8 56.5 28.5 43.2 Idefics1 34.6 20.7 22.7 54.0 27.3 33.7 Gemini-1.5-Flash 67.1 52.3 49.2 64.5 45.5 49.9 Gemini-1.5-Pro 60.9 56.7 43.3 65.2 43.0 56.3 GPT-4o 52.0 59.3 49.3 46.8 42.0 60.8

###### Ours

VIDEOSCORE (gen) 59.0 64.2 57.7 55.5 54.3 61.5 VIDEOSCORE (reg) 78.5 78.2 71.5 68.0 74.0 69.0 ∆ over Best Baseline +11.4 +17.4 +20.3 -16.2 +18.5 +8.2

- Table 5: Pairwise preference accuracy on GenAI-Bench (Jiang et al., 2024b) and VBench (Huang et al., 2023). For MLLM prompting and our method, we averaged the five aspect scores defined in Table 2 as the score for each video in the comparison, where the higher one deemed the winner.

Method Visual Temporal Text Align

Random -2.0 1.4 -0.9 EvalCrafter 55.4 56.7 32.3

Feature-based Automatic Metrics

PIQE 0.5 -3.3 -0.9 BRISQUE 6.4 -1.3 6.7 CLIP-sim 36.0 53.5 19.2 DINO-sim 30.6 50.3 15.3 SSIM-im 32.4 36.9 11.4 MSE-dyn -15.4 -27.5 -8.1 SSIM-dyn -32.6 -33.9 -12.6 CLIP-Score 18.7 11.5 35.0 X-CLIP-Score 12.2 3.1 24.5

MLLM Prompting

- LLaVA-1.5-7B 13.4 15.6 2.6

- LLaVA-1.6-7B 12.2 8.5 18.9 Idefics1 1.5 -1.5 0.8 Gemini-1.5-Flash 34.9 -27.8 44.8 Gemini-1.5-Pro 37.8 -24.1 55.1 GPT-4o 32.9 12.5 40.7

Ours

VIDEOSCORE (gen) 20.8 51.3 10.7 VIDEOSCORE (reg) 42.4 51.3 59.5 ∆ over Best Baseline -13.1 -5.4 4.4

- Table 6: Spearman’s Correlation (ρ) of VIDEOSCORE on EvalCrafter (Liu et al., 2023b).

we conduct a comparison of different T2V models with and without employing best-of-k sampling with VIDEOSCORE on EvalCrafter.

We set k = 5 and generated videos using 700 prompts from EvalCrafter. For each prompt, the video with the highest average VIDEOSCORE across five dimensions was selected. We then evaluated both the "best videos" and randomly chosen ones using EvalCrafter’s metrics, averaging the results over 700 videos to obtain the model’s final score. As shown in Table 8, compared to the random sample, most scores on the EvalCrafter benchmark have increased after the best-of-5 process.

#### 4.6 Ablation Study

We conducted an ablation study on the base model selection and scoring types by training different variants on VIDEOFEEDBACK. Results of the ablation study are shown in Table 7.

Base model ablation To investigate the effects of changing the base model, we choose VideoLLaVA7B and Idefics2-8B as base models from recent popular VLMs (Lin et al. (2023), Laurençon et al. (2024), Zhang et al. (2023), Li et al. (2024a)). Since VIDEOFEEDBACK-test, EvalCrafter, and

Base Model Scoring Type VIDEOFEEDBACK ∗ EvalCrafter∗ GenAI-Bench VBench∗ Average

VideoLLaVA-7B Generation 71.9 9.8 42.6 46.5 42.7 Idefics2-8B Generation 73.9 11.3 50.7 53.9 47.5 Mantis-Idefics2-8B Generation 77.1 27.6 59.0 58.7 55.6 Idefics2-8B Regression 73.9 17.4 74.5 64.4 57.5 Mantis-Idefics2-8B Regression 75.7 51.1 78.5 73.0 69.6

- Table 7: Ablation study on the base model and scoring function for VIDEOSCORE. "∗" means that we take the average of Spearman correlation or pairwise accuracy across the multiple aspects of the benchmark. The highest numbers are bold for each benchmark, and the second are underlined.

T2V Model

|Dimensions of EvalCrafter| | | | |
|---|---|---|---|---|
|Average random best<br><br>|Visual random best|Temporal<br><br>random best<br><br>|Motion random best|T2V Align random best<br><br>|

AnimateDiff 62.16 61.87 60.79 60.89 60.69 60.94 54.35 54.83 72.79 70.83 HotShot-XL 53.69 61.22 57.56 60.39 58.85 60.94 51.52 54.05 46.83 69.52 LaVie-base 57.88 59.90 57.79 58.52 54.21 57.70 52.51 53.53 66.99 69.86 VideoCrafter2 58.31 58.98 58.44 59.70 59.18 60.79 54.39 54.65 61.23 60.77 VideoCrafter1 54.30 57.28 52.40 53.68 56.48 59.81 54.01 54.88 54.32 60.76 ModelScope 52.45 54.48 44.80 45.01 56.70 60.34 53.20 54.42 55.09 58.17 ZeroScope-576w 51.07 54.09 43.36 43.82 55.98 58.74 54.55 54.68 50.39 59.12 LVDM 45.80 46.04 44.45 44.64 40.44 43.09 53.68 53.26 44.61 43.16

- Table 8: Performace of T2V models on EvalCrafter with and without best-of-5 sampling using VIDEOSCORE. Most EvalCrafter scores have increased compared to the random sample, proving the effectiveness of VIDEOSCORE

VBench both have multiple aspects in the benchmarks, we take the average score across these aspects and report the general performance in Table 7. The results show that the Video-LLaVA-based version gets the worst performance on the four benchmarks, even if it is specifically designed for video understanding. The Idefics2-8B-based version has marginal improvements compared to the VideoLLaVA. After changing to Mantis-Idefics28B, the scores on the four benchmarks keep improving from 47.5 to 55.6 on average. When the scoring type is regression, the mantis-based version is still better than the Idefics2-based version by 12.1 points. Therefore, we select the Mantisbased version as the final choice.

Regression scoring or generative scoring? The primary difference between regression scoring and generative scoring is that regression scoring can give more fine-grained scores instead of just the four labels. Results on EvalCrafter, GenAI-Bench, and VBench all indicate that using regression scoring can consistently improve the Spearman correlation or the pairwise comparison accuracy. For example, on GenAI-Bench, VIDEOSCORE (reg) achieves 78.5 accuracy, which is higher than the 59.0 of the VIDEOSCORE (gen). The results are similar for the other benchmarks. We thus conclude

that regression scoring with more fine-grained scores is a better choice.

### 5 Conclusion

In this paper, we introduce VIDEOSCORE, which is trained on our meticulously curated dataset VIDEOFEEDBACK for video evaluation and can serve as good simulator for human feedback on generated videos. We hired 20 expert raters to annotate the 37.6K videos generated from 11 popular textto-video generative models across 5 key aspects, Visual Quality, Temporal Consistency, Dynamic Degree, Text-to-Video Alignment and Factual Consistency. Our IAA match ratio gets more than 60%. We test the performance of VIDEOSCORE using Spearman correlation on VIDEOFEEDBACK-test and EvalCrafter, and using pairwise comparison accuracy on GenAI-Bench and VBench. The results show that VIDEOSCORE consistently gets the best performance, surpassing the powerful baseline GPT-4o and Gemini 1.5 Flash/Pro by a large margin. Our work highlights the importance of using MLLM for video evaluation and demonstrates the future prospects of simulating human scores or feedback in generative tasks, due to its rich world knowledge and the high-quality rating dataset with fine-grained and multiple dimensions.

### Acknowledgement

We express our gratitude to StarDust for providing video raters and to DataCurve for supplying the GPU compute resources. Additionally, we express our thanks to all the raters who offered valuable feedback and suggestions, which were instrumental in completing this work.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Adept AI. 2023. Fuyu-8B: A Multimodal Architecture for AI Agents. https://www.adept.ai/blog/ fuyu-8b.

Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. 2023. Latentshift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477.

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. 2023. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. 2024. Videophy: Evaluating physical commonsense for video generation.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. 2024. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023a. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: Highresolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 2021. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. 2023a. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. 2024a. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047.

Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. 2023b. Gentron: Delving deep into diffusion transformers for image and video generation. arXiv preprint arXiv:2312.04557.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, MingHsuan Yang, and Sergey Tulyakov. 2024b. Panda70m: Captioning 70m videos with multiple crossmodality teachers. arXiv preprint arXiv:2402.19479.

Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2023c. Seine: Short-to-long video diffusion model for generative transition and prediction. arXiv preprint arXiv:2310.20700.

Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. 2023. Directly fine-tuning diffusion models on differentiable rewards. ArXiv, abs/2309.17400.

Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. 2023. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. 2024. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36.

Joseph L Fleiss and Jacob Cohen. 1973. The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability. Educational and psychological measurement, 33(3):613–619.

Floor33. 2024. Floor33 pictures: Ai video generator.

Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. 2023. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725.

Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. 2023. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662.

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent video diffusion models for high-fidelity long video generation.

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. 2017. Localizing moments in video with natural language. Preprint, arXiv:1708.01641.

Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2024. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840– 6851.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. 2022. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868.

Berthold KP Horn and Brian G Schunck. 1981. Determining optical flow. Artificial intelligence, 17(13):185–203.

Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. 2022. Real-time intermediate flow estimation for video frame interpolation. Preprint, arXiv:2011.06294.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2023. Vbench: Comprehensive benchmark suite for video generative models. ArXiv, abs/2311.17982.

Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. 2024a. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483.

Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. 2024b. Genai arena: An open evaluation platform for generative models. arXiv preprint arXiv:2406.04485.

Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439.

Tengchuan Kou, Xiaohong Liu, Zicheng Zhang, Chunyi Li, Haoning Wu, Xiongkuo Min, Guangtao Zhai, and Ning Liu. 2024a. Subjective-aligned dataset and metric for text-to-video quality assessment. ArXiv, abs/2403.11956.

Tengchuan Kou, Xiaohong Liu, Zicheng Zhang, Chunyi Li, Haoning Wu, Xiongkuo Min, Guangtao Zhai, and Ning Liu. 2024b. Subjective-aligned dateset and metric for text-to-video quality assessment. arXiv preprint arXiv:2403.11956.

Klaus Krippendorff. 2011. Computing krippendorff’s alpha-reliability.

Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. 2023. Viescore: Towards explainable metrics for conditional image synthesis evaluation. Preprint, arXiv:2312.14867.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. 2023. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Preprint, arXiv:2306.16527.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. 2024. What matters when building vision-language models? ArXiv, abs/2405.02246.

Bo Li, Hao Zhang, Kaichen Zhang, Dong Guo, Yuanhan Zhang, Renrui Zhang, Feng Li, Ziwei Liu, and Chunyuan Li. 2024a. Llava-next: What else influences visual instruction tuning beyond data?

Jiachen Li, Weixi Feng, Wenhu Chen, and William Yang Wang. 2024b. Reward guided latent consistency distillation. ArXiv, abs/2403.11027.

Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. 2024c. T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval.

Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, Junjie Ke, Krishnamurthy Dvijotham, Katie Collins, Yiwen Luo, Yang Li, Kai Kohlhoff, Deepak Ramachandran, and Vidhya Navalpakkam. 2023. Rich human feedback for text-to-image generation. ArXiv, abs/2312.10240.

Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-llava: Learning united visual representation by alignment before projection. ArXiv, abs/2311.10122.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. Llavanext: Improved reasoning, ocr, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. ArXiv, abs/2304.08485.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. 2023b. Evalcrafter: Benchmarking and evaluating large video generation models.

Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. 2023c. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. Preprint, arXiv:2311.01813.

Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. 2022. X-clip: End-toend multi-grained contrastive learning for video-text retrieval. In Proceedings of the 30th ACM International Conference on Multimedia, pages 638–647.

Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. 2012a. No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing, 21(12):4695–4708.

Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. 2012b. No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing, 21(12):4695–4708.

John Mullan, Duncan Crawbuck, and Aakash Sastry.

2023. Hotshot-XL.

- OpenAI. 2024a. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/. Accessed: 2024-06-15.
- OpenAI. 2024b. Video generation models as world simulators.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. ArXiv, abs/2306.14824.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,

Gretchen Krueger, and Ilya Sutskever. 2021a. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021b. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. 2016. Improved techniques for training gans. Advances in neural information processing systems, 29.

Spencer Sterling. 2024. Zeroscope v2.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. 2019. FVD: A new metric for video generation.

N. Venkatanath, D. Praneeth, Maruthi Chandrasekhar Bh, Sumohana Channappayya, and Swarup Medasani. 2015. Blind image quality evaluation using perception based features. 2015 21st National Conference on Communications, NCC 2015.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq R. Joty, and Nikhil Naik. 2023. Diffusion model alignment using direct preference optimization. ArXiv, abs/2311.12908.

Haoxiang Wang, Yong Lin, Wei Xiong, Rui Yang, Shizhe Diao, Shuang Qiu, Han Zhao, and Tong Zhang. 2024a. Arithmetic control of llms for diverse user preferences: Directional preference alignment with multi-objective rewards. ArXiv, abs/2402.18571.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. 2024b. Interpretable preferences via multi-objective reward modeling and mixture-ofexperts. ArXiv, abs/2406.12845.

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023a. Modelscope text-to-video technical report. ArXiv, abs/2308.06571.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2023b. Cogvlm: Visual expert for pretrained language models. ArXiv, abs/2311.03079.

Wenhao Wang and Yi Yang. 2024. Vidprom: A millionscale real prompt-gallery dataset for text-to-video diffusion models. ArXiv, abs/2403.06098.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. 2023c. Lavie: Highquality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612.

Haoning Wu, Chaofeng Chen, Jingwen Hou, Liang Liao, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. 2022. Fast-vqa: Efficient end-to-end video quality assessment with fragment sampling. In European Conference on Computer Vision, pages 538–554.

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. 2023a. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154.

Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang, Yixiao Ge, Xiaodong Cun, David Junhao Zhang, Jia-Wei Liu, Yuchao Gu, Rui Zhao, Weisi Lin, Wynne Hsu, Ying Shan, and Mike Zheng Shou. 2024. Towards a better metric for text-to-video generation. ArXiv, abs/2401.07781.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023b. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. Imagereward: Learning and evaluating human preferences for text-to-image generation. ArXiv, abs/2304.05977.

Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. 2023. Instructvideo: Instructing video diffusion models with human feedback. ArXiv, abs/2312.12490.

Hang Zhang, Xin Li, and Lidong Bing. 2023. Videollama: An instruction-tuned audio-visual language model for video understanding. In Conference on Empirical Methods in Natural Language Processing.

Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk. 2024. Large-scale reinforcement learning for diffusion models. ArXiv, abs/2401.12244.

Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. 2022. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018.

### A Ethical Statement

This work fully complies with the ACL Ethics Policy. We declare that there are no ethical issues in this paper, to the best of our knowledge.

### B Risks and Limitation

Although we have designed systematic pipelines to recruit expert raters and annotate the video evaluation scores, we still find out that some annotations contain errors and may harm the overall quality of the dataset. Our IAA score computation is only based on a small number of trial examples and, thus might not represent the actual IAA of the whole annotations. Besides, while VIDEOSCORE is proven to be able to effectively give reasonable scores on our defined five aspects, it can still sometimes output wrong scores that do not match our expectations. We admit this drawback and list that as one of our future works.

### C Dataset Licence

We have used VidProM (Wang and Yang, 2024) to collect the prompts used for video generation, whose usage LICENSE is CC BY-NC 4.0 license. For other evaluation datasets, We did not find license for EvalCrafter (Liu et al., 2023b) human annotations. GenAI-Bench (Jiang et al., 2024b) is under MIT licence, and VBench (Huang et al., 2023) is under Apache 2.0 license. We are thus able to utilize these datasets in our experiments.

We also release our curated dataset, VIDEOFEEDBACK, under MIT license to contribute to the video evaluation dataset.

### D Annotator Management

During the annotation, we have recruited 20 expert raters, where 14 of them are undergraduate or graduate students, who will become one of the authors of our paper, and the rest of them are assured to be paid with decent salary.

### E Video Format Normalizing Details

To mitigate difference of videos format from different generative models, we normalize the frame rate of all the generated videos to 8 fps (frames per second). Specifically, for high frame rate model Pika and AnimateDiffusion (Guo et al., 2023), we use uniform down-sampling to normalize Pika from 24 fps to 8fps, and AnimateDiffusion from 23 fps to 8 fps. For low frame

rate model Text2Video-Zero (Khachatryan et al., 2023), we use video frame interpolation model RIFE (Huang et al., 2022) to interpolate frames, adding the frame rate from 4 fps to 8 fps. For real-world videos from DiDeMo (Hendricks et al., 2017) and Panda70M (Chen et al., 2024b) in post augmentation of VIDEOFEEDBACK, we use the same down-sampling as Pika and AnimateDiffusion to reduce their frame rate from 30 fps to 8 fps.

Additionally, since video from Pika are always attached a watermark "PIKA-LABS", we cropped all the Pika videos from the resolution of (1088, 640) to (768, 480), making Pika video indistinguishable from videos from other models.

### F Annotation Details

Additional annotation details are put in this section for the reference.

Firstly we show the user interface of our annotating website in Figure 3 and Figure 4. In both welcome page and working page, we list the definition and a checklist of error points in five evaluation dimensions, as shown in Table 9. Additionally we also provide many Good/Average/Poor videos as examples in each dimension for raters to quickly understand each dimension and align well with our understanding.

### G Prompting Template

In process of training Mantis (Jiang et al., 2024a) for generation scoring and the testing with "MLLM Prompting" baselines, we use the same prompt template provided in Table 10.

For training Mantis with regression scoring, we make modification to the above template accordingly, instructing model to output a float number ranges from 1.0 to 4.0, as shown in Table 11.

### H Feature-based Baselines Discretization

As described in subsection 4.1, we employ several statistical or neural feature-based metrics as baselines for comparison with our model. The continuous float-format outputs of these metrics are discretized into labels [1, 2, 3, 4], aligning with our annotation data format. The discretization rules are presented in Table 12. Metrics with a ↑ symbol indicate that higher values are better, while those with a ↓ symbol indicate that lower values are better.

[Figure 16]

##### Figure 3: Welcome Page of our video annotating website, with definition, checklist for error points and diverse video examples.

[Figure 17]

Figure 4: Working page of our video annotating website

### I Case study of VIDEOFEEDBACK

We showcase the annotations examples in Figure 5. The first example depicts a clear video of a woman with her hair moving, thus scoring 3 in all 5 aspects. The second example shows a distorted video, thus scoring 1 across all the aspects except the dynamic degree. We further analyzed the correlations between the designed aspects in Figure 6. We found that visual quality achieves a high correlation of 0.6 with temporal consistency, while dynamic degree has a very low correlation with all other aspects.

|Video Annotation|
|---|

Text: The wind gently blows, and her hair moves. Nothing else moves.

[Figure 18]

[Figure 19]

[Figure 20]

Visual Quality : 3 Temporal Consistency : 3 Dynamic Degree : 3 Text-to-Video Alignment : 3

Factual Consistency : 3

Text: A transformation of a happy, energetic cartoon character gradually transitioning to a state of burnout and exhaustion.

Visual Quality : 1 Temporal Consistency : 1 Dynamic Degree : 3 Text-to-Video Alignment : 1 Factual Consistency : 1

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Figure 5: Example of annotations. Each video has a text description and is rated for the 5 aspects.

[Figure 26]

Figure 6: Correlation study on the evaluation aspects.

shown in Table 13.

### J Leaderboard

We generated 200 videos using various T2V models with prompts sampled from our prompt set, then we rank these T2V models based on the average score of five dimensions in VIDEOSCORE, as

Evaluation Aspect

Detailed Description for Annotation

Expected Case:

- (1) The video looks clear and normal on its appearance.
- (2) The features like Brightness, Contrast, Color, etc, are appropriate and stable. Error point:

Visual Quality

(a) local obvious unclear or blurry, (b) too low resolution, (c) some speckles or black patches, (d) appearance of video is skewed and distorted, (e) unstable optical property, such as brightness, contrast, saturation, exposure etc, (f) flickering color of main objects and background

Note:Some videos have watermark, we can ignore that.

Expected Case:

- (1) The main objects, main characters and overall appearance are consistent across the video.
- (2) The appearance of video as well as the movements of humans and objects are smooth and natural. Error points:

Temporal Consistency

(a) The person or object suddenly disappears or appears, (b) The type or class of objects has obvious changes, (c) There is an obvious switch in the screen shot,(d) the appearance of video or movements in it is laggy and un-smooth, (e) local deformation or dislocation of human or objects due to the motion.

(for large scale deformation, the video should also be rated as bad in "1. visual quality"). Note: For a video almost static or with small dynamic degree, as long as it does not have error points, then it should be scored as good.

Expected Case:

- (1) The video is obviously not static, the people or objects or the video screen is dynamic.
- (2) The video can be easily distinguished from a static image. Note: You are supposed to focus on only dynamic degree, regardless of the visual quality and video content

Dynamic Degree

Expected Case: The characters, objects, motions, events etc. that are mentioned in text input prompts all exist reasonably. Error points:

(a) The people and objects in prompt do not appear in video, (b) The actions and events in prompt do not appear in video, (c) The number, size, shape, color, state, movement and other attributes of the objects in the video do not match the prompt, (d) Text mentioned in prompt is not displayed correctly in the video, such as "a placard saying ’No Smoking’" but "No Smoking" is not spelled correctly in the video, (e) The video format (such as width, height, screen ratio, duration) does not match the format in prompt.

Text-toVideo Alignment

Expected Case:

(1) Overall appreance and motion are consistent with our common-sense, physical principles, moral standards, etc. Error points:

- (a) static ones: Content in video goes against common sense in life, such as lighting a torch in the water, standing in the rain but not getting wet, etc.
- (b) static ones: The size, color, shape and other basic properties of objects violate scientific principles
- (c) dynamic ones: The overall movement of people or objects violates common-sense and laws of physics, such as spontaneous upward movement against gravity, abnormal water flow, etc.
- (d) dynamic ones: Partial movements of people or objects violate common-sense and laws of physics, such as the movement of hands or legs is anti-joint, etc. Notes: Relation with ’5. text-to-video alignment’:

Factual Consistency

Some text prompts express fictional and unrealistic content, for example, "a dog plays the guitar in the sky" or "an astronaut rides a horse in space". In this case, regardless of the veracity of the text prompt, you should only consider whether the other content in the video makes sense.

Table 9: Expected cases and error cases for each aspect that annotators can see during the annotation.

Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please watch the following frames of a given video and see the text prompt for generating the video, then give scores from 5 different dimensions:

- (1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
- (2) temporal consistency, the consistency of objects or humans in video
- (3) dynamic degree, the degree of dynamic changes
- (4) text-to-video alignment, the alignment between the text prompt and the video content
- (5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a number from [1,2,3,4], in which ’1’ means ’Bad’, ’2’ means ’Average’, ’3’ means ’Good’, ’4’ means ’Real’ or ’Perfect’ (the video is like a real video) Here is an output example: visual quality: 4 temporal consistency: 4 dynamic degree: 3 text-to-video alignment: 1 factual consistency: 2

For this video, the text prompt is "{text_prompt}", all the frames of video are as follows:

Table 10: Prompting template in generation format used for VIDEOSCORE training and the MLLM prompting baselines

Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please watch the following frames of a given video and see the text prompt for generating the video, then give scores from 5 different dimensions:

- (1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
- (2) temporal consistency, the consistency of objects or humans in video
- (3) dynamic degree, the degree of dynamic changes
- (4) text-to-video alignment, the alignment between the text prompt and the video content
- (5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a float number from 1.0 to 4.0, higher the number is, better the video performs in that dimension, the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video) Here is an output example: visual quality: 2.24 temporal consistency: 3.89 dynamic degree: 3.17 text-to-video alignment: 1.86 factual consistency: 2.16

For this video, the text prompt is "{text_prompt}", all the frames of video are as follows:

Table 11: Prompting template used for the MLLM prompting baseline and VIDEOSCORE training

Dimension Metric 1 (Bad) 2 (Avg) 3 (Good) 4 (Perfect)

PIQE↓ [50, ∞) [30,50) [15,30) [0,15) BRISQUE↓ [50,∞) [30,50) [10,30) [0,10)

Visual Quality

CLIP-sim↑ [0,0.80) [0.80,0.90) [0.90,0.97) [0.97,1] DINO-sim↑ [0,0.75) [0.75,0.85) [0.85,0.95) [0.95,1] SSIM-sim↑ [0,0.6) [0.6,0.75) [0.75,0.9) [0.9,1]

Temporal Consistency

MSE-dyn↑ [0,100) [100,1000) [1000,3000) [3000,∞)

Dynamic Degree

SSIM-dyn↓ [0.9,1] [0.7,0.9) [0.5,0.7) [0,0.5)

CLIP-Score↑ [0.2,0.27) [0.27,0.31) [0.31,0.35) [0.35,0.4]

Text-to-Video Alignment

X-CLIP-Score↑ [0,0.15) [0.15,0.23) [0.23,0.30) [0.30,1]

Table 12: Discretization rules for featured-based baselines.

T2V Model Avg VQ TC DD TVA FC

Kling 81.83 84.63 82.49 85.10 74.29 82.65 OpenSora-v1.2 69.85 70.36 67.17 75.91 68.17 67.62 Morph Studio 67.56 66.79 70.98 66.71 68.55 64.77 Morph Studio 67.56 66.79 70.98 66.71 68.55 64.77 VideoCrafter-2 66.23 66.72 68.81 67.04 65.77 62.79 Gen-2 65.74 65.14 67.37 67.97 65.39 62.84 PikaLab 65.72 67.21 67.98 65.17 63.79 64.43 Video-LaVIT 64.04 67.57 59.68 70.83 66.7 55.40 LaVie-base 62.86 63.11 59.33 70.25 62.40 59.20 MagicTime 62.61 65.05 63.2 67.33 61.99 55.50 HotShot-XL 62.31 58.55 59.63 70.90 63.93 58.54 Latte 62.02 63.34 59.23 68.38 62.56 56.60

- OpenSora-v1.0 61.94 62.49 58.75 69.65 64.30 54.51

- OpenSora-v1.1 61.29 61.50 56.82 69.82 64.72 53.61 VideoCrafter-1-512 60.92 60.94 60.12 67.38 60.44 55.73 AnimateDiff 57.33 64.83 41.89 73.46 65.76 40.69 ModelScope 52.45 47.41 49.76 69.14 54 41.94 LVDM 47.20 33.15 44.01 72.75 58.85 27.25 ZeroScope_576w 44.69 31.35 39.65 73.76 49.40 29.27

Table 13: Leaderboard of VIDEOSCORE for existing text-to-video models on 200 curated examples.

