# arXiv:2412.00927v1[cs.CV]1Dec2024

## VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by VIdeo SpatioTemporal Augmentation

Weiming Ren1,2,3, Huan Yang3,∗, Jie Min1, Cong Wei1,2, Wenhu Chen1,2,* 1University of Waterloo, 2Vector Institute, 301.AI

{w2ren,wenhuchen}@uwaterloo.ca, hyang@fastmail.com

https://tiger-ai-lab.github.io/VISTA/

#### Abstract

Current large multimodal models (LMMs) face significant challenges in processing and comprehending long-duration or high-resolution videos, which is mainly due to the lack of high-quality datasets. To address this issue from a data-centric perspective, we propose VISTA, a simple yet effective VIdeo SpatioTemporal Augmentation framework that synthesizes long-duration and high-resolution video instruction-following pairs from existing video-caption datasets. VISTA spatially and temporally combines videos to create new synthetic videos with extended durations and enhanced resolutions, and subsequently produces questionanswer pairs pertaining to these newly synthesized videos. Based on this paradigm, we develop seven video augmentation methods and curate VISTA-400K, a video instructionfollowing dataset aimed at enhancing long-duration and high-resolution video understanding. Finetuning various video LMMs on our data resulted in an average improvement of 3.3% across four challenging benchmarks for long-video understanding. Furthermore, we introduce the first comprehensive high-resolution video understanding benchmark HRVideoBench, on which our finetuned models achieve a 6.5% performance gain. These results highlight the effectiveness of our framework.

#### 1. Introduction

Recent advancements in large language models (LLMs) and large multimodal models (LMMs) have brought transformative changes to video understanding tasks. Traditionally, video understanding relied on training task-specific models using domain-specific datasets (e.g. action recognition on Kinetics [3], video retrieval on MSR-VTT [54], and video captioning on YouCook2 [67]). In contrast, it is now possible to process video inputs and address diverse tasks using a single video LMM [6, 26, 32] through instruction follow-

*Corresponding authors.

###### VISTA

|Long/High-Res Videos|
|---|

Low-quality Videos

Video Augmentation

###### Paired

|Simple Captions| |
|---|---|
| | |

|Instruction Data|
|---|

Instruction Synthesis

60

| |Baseline Models VISTA-finetuned Models<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

AverageAccuracy

40

20

0

Video-MME MLVU LVBench LongVideo Bench

HRVideo Bench

Short Video Avg.

Figure 1. VISTA is a simple but effective framework that generates high-quality video instruction data from existing video-caption pairs. Our VISTA-400K dataset enhances model performances on various long and high-resolution video benchmarks.

ing. However, most current (open-sourced) video LMMs are optimized for understanding and reasoning over short, low-resolution videos. Processing long-sequence video input, such as long or high-resolution videos, remains a significant challenge for video LMMs.

Most efforts to enhance long-sequence video understanding have centered around designing better model architectures. Conversely, the creation of higher-quality (long/high-resolution) video instruction-following datasets remains highly under-explored. The primary challenges stem from the scarcity of publicly available (licensefriendly) high-quality video-instruction data. Consequently, open-source video instruction-following datasets often face limitations such as low resolution or short duration. For example, VideoChat2 [23] collects a video instructionfollowing dataset by combining videos from multiple domain-specific datasets and rewriting the instructions using ChatGPT [34]. However, the video sources used in this dataset primarily contain short videos. ShareGPT4Video [4] curates a video-instruction dataset containing 40K videos with detailed video captions. Nevertheless, the source videos are collected at a sparse sampling rate of 0.15

fps, and the video contents often lack motion and can appear nearly static. FineVideo [8] is a long video dataset featuring diverse video contents and high-quality metadata such as detailed captions and narrative progressions. While it offers satisfactory video durations, the dataset is limited by its resolution, as it predominantly comprises 360p videos.

Though some concurrent studies have improved LMMs’ long-sequence video understanding capabilities, they often release only the model weights, hiding their video training data. For instance, Kangaroo [28]’s training data contains 700K long video data, but specifics about the curation of the videos and the construction of the instructions are not provided. Similarly, Qwen2-VL [46], MiniCPM-V-2.6 [57] and Aria [19] all claim that their instruction-following dataset contains long video training data, yet without providing the sources and statistics of the video data. This lack of transparency hinders a clear understanding of what types of video instruction data truly benefit long-sequence video understanding tasks and impedes further advancements in improving existing models.

In this study, we propose VISTA, a simple yet effective video augmentation pipeline to synthesize long and highresolution video instruction data from existing video datasets. VISTA leverages insights from image and video classification data augmentation techniques such as CutMix [59], MixUp [61] and VideoMix [60], which demonstrate that training on synthetic data created by overlaying or mixing multiple images or videos results in more robust classifiers. Similarly, our method spatially and temporally combines videos to create (artificial) augmented video samples with longer durations and higher resolutions, followed by synthesizing instruction data based on these new videos. Our data synthesis pipeline utilizes existing public videocaption datasets, making it fully open-sourced and scalable. This allows us to construct VISTA-400K, a high-quality video instruction-following dataset aimed at improving the long and high-resolution video understanding capabilities of video LMMs. By finetuning various video LMMs on our dataset, we observe an average of 3.3% improvement across multiple long video understanding benchmarks. Additionally, we compile a new benchmark HRVideoBench that focuses on high-resolution videos. Results on HRVideoBench (+6.5% after finetuning) demonstrate that our method produces models well-suited for high-resolution video understanding. Our contributions are summarized below:

- 1. We present VISTA-400K, a high-quality synthetic video instruction-following dataset. VISTA-400K includes challenging QA pairs designed to enhance video LMMs’ ability to understand long and high-res video inputs.
- 2. We collect HRVideoBench, a comprehensive benchmark dedicated to evaluating video LMMs’ capability in understanding fine object details and subtle, localized actions within high-resolution videos.

3. Our VISTA-finetuned models achieve an average gain of 3.3% on four challenging long-video benchmarks and 6.5% on HRVideoBench compared to the vanilla models. Our ablation study indicates that disabling our proposed video augmentations significantly reduces performance.

#### 2. VISTA-400K Dataset

Our goal is to create high-quality video instructionfollowing data from existing video-caption datasets. Specifically, we aim to (1) generate extended or higher-resolution videos by spatially or temporally combining existing videos, and (2) produce high-quality question-answer pairs by leveraging existing video captions. Formally, given a set of candidate videos V = {V1,V2,...,VN} with captions C = {C1,C2,...,CN}, we seek to create an augmented video V ∗ and synthesize a QA pair (q,a):

V ∗ = Φ(V), (q,a) = Θ(C). (1)

For the video augmentation operator Φ, we draw inspiration from data augmentation techniques used in image and video classification, such as CutMix [59] and VideoMix [60], to perform video mixing and combination. We use a highly capable language model Gemini-1.5-Pro [44] as Θ to generate synthetic QA pairs. As shown in Figure 2, our dataset contains a total of seven subsets, each featuring different video augmentation methods. In this section, we detail the dataset construction process for each subset.

##### 2.1. Long Video Captioning & Event QA

Our first approach to synthesizing long video instruction data is to generate longer videos by temporally concatenating multiple short clips, as illustrated by Figure 2-A. We observe that public video-text datasets like InternVid [50] and Panda-70M [5] often contain short clips that are drawn from longer videos, which can be combined to form extended sequences. To harvest long video data, we combine multiple short clips from the same source video, ensuring that the interval between them does not exceed five seconds. This preserves natural scenes and content transitions in the extended videos. Given sampled videos containing multiple short clips and their accompanying captions, we generate two types of instruction-following data using Gemini:

- 1. Long Video Captioning: given the captions for each short clip, we prompt Gemini to generate a longer caption describing the entire video. This task is designed to enhance the summarization ability of video LMMs for input videos with longer durations.
- 2. Event Relationship QA: we let Gemini generate freeform or multiple choice questions related to the order of the events based on the short clip captions. Figure 2B shows an example of such instruction data. These QA pairs focus on improving video LMM’s ability to recognize action and event sequences.

|A. Long Video Captioning|
|---|

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Question: Describe the following video in detail, including the actions and scenes. Answer: A team of race car drivers and crew members prepare for a race. One man, sitting at a table with a microphone, speaks to the camera. Soon, a car is driving down the race track at a high speed.

|B. Event Relationship QA|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Question: Where does the man go after catching the fish? Answer: He goes to his white pickup truck parked nearby.

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

|C. Temporal NIAH|
|---|

Question: There exists a brief interval in the video that diverges into different content compared to the rest of the video. What does this short interval feauture? A. A different car on a racetrack. B. The game's menu screen. C. A woman is sitting in front of a blue background and talking to the camera. D. A technical error message. Answer: C

|D. Two Needle NIAH|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Question: This video includes a short clip divided into two parts, which are randomly inserted into a longer video. Your objective is to locate both parts and respond to a question. Briefly explain what the short clip is about. Answer: A person is using a spatula to stir scrambled eggs on a plate.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

|E. Spatial NIAH|
|---|

Question: In one small area in the video, there is a different scene. What is happening in that small area?

- Answer: A person is cleaning a pair of shoes.

|G. HR Video Grid QA|
|---|

| |
|---|

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Question: Detect a specific area of the video during a set timeframe that shows varying content, and respond to a question regarding that section.What is printed on the man's T-shirt? A: Apple, B: Google, C: Microsoft, D: Amazon

- Answer: B. Google.

|F. Spatiotemporal NIAH|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Question: Take a close look at the presented visuals and deliver a precise answer to the corresponding question. What content is displayed in the 3rd row, 6th column of the video? Answer: A man is cutting grass with an electric mower.

Figure 2. Our proposed video augmentation and instruction-following data synthesis schemes for VISTA-400K. Given input videos, We perform spatiotemporal video combinations to produce augmented video samples with longer duration and higher resolution.

##### 2.2. Video Needle-in-a-haystack (NIAH) QA

To effectively understand and reason over long or highresolution videos, video LMMs must learn to accurately retrieve relevant information from this long sequence of video tokens, i.e. finding “needles in a haystack”. NIAH experiments have been widely adopted to evaluate LLMs [15] and have also been adapted to LMM evaluations [45, 48]. In this approach, we propose to spatially or temporally combine videos to form various NIAH QA data:

1. Temporal NIAH: as shown in Figure 2-C, this approach

randomly inserts a short clip within a longer video, creating a “temporal needle” in the sequence. Based on this combined video, our instruction data contain questions that ask video LMMs to describe the content of the needle video. This task challenges video LMMs to locate the relevant needle tokens within a long video and accurately summarize their content.

2. Two Needle NIAH: we consider a variant of Temporal NIAH, where a short video clip is split into two parts and inserted at different timestamps within a longer clip.

Table 1. Statistics of our synthetic video instruction-following dataset. “(N)” and “(H)” corresponds to the “needle” (short or low-res videos) and the “haystack” (long or high-res videos) in NIAH subsets.

Subset Instruction Type Video Source #Videos Avg. Duration Avg. Resolution

Long Video Captioning Video Captioning Panda-70M [5] 58,617 33.2s 1277×720 Event Relationship QA Freeform QA/MCQ Panda-70M [5] 56,854 33.4s 1278×720 Temporal NIAH Freeform QA/MCQ Panda-70M [5] (N), MiraData [14] (H) 59,751 67.6s 640×358 Two Needle NIAH Freeform QA Panda-70M [5] (N), FineVideo [8] (H) 52,349 112.4s 591×382 Spatial NIAH Freeform QA/MCQ InternVid [50] (N), OpenVid-1M [33] (H) 59,978 9.9s 1726×971 Spatiotemporal NIAH Freeform QA/MCQ OpenVid-1M [33] (N), FineVideo [8] (H) 56,494 89.9s 591×383 HR Video Grid QA Freeform QA/MCQ InternVid [50] 59,901 3s 1920×1080

VISTA-400K - - 403,944 48.6s 1160×666

Video LMMs must summarize the short clip’s content by locating both “needles”, challenging them to retrieve relevant information from multiple temporal locations within the video sequence.

- 3. Spatial NIAH: recognizing local and small objects in high-resolution videos is essential for video LMMs, yet obtaining suitable training data for high-resolution videos with detailed captions or QA pairs is challenging. We propose to generate spatial NIAH data to simulate such training data. As depicted in Figure 2-E, we overlay a small and low-resolution video onto a high-resolution video at a random position, and then ask a question related to this small video. This approach focuses our QA data on a specific ”needle” region, forcing video LMMs to extract local information from high-resolution videos.
- 4. Spatiotemporal NIAH: finally, we generate synthetic training data by integrating spatial and temporal NIAH. As shown in Figure 2-F, we embed a low-resolution, short-duration “needle” video within a longer, highresolution video at a random spatial position and timestamp. This combined setup encourages video LMMs to understand video contents across both spatial and temporal dimensions, fostering a more comprehensive understanding of complex video inputs.

For all four NIAH variants, we obtain freeform QA pairs by prompting Gemini using the captions of the “needle” videos. We further convert half of these QA pairs into multiple-choice questions by prompting Gemini to generate incorrect distractors derived from the captions of the “haystack” videos. This method ensures the distractor options are contextually related to the haystack content. Consequently, if the model fails to identify the needle video correctly, it is more likely to choose these distractor options, increasing the challenge and rigour of the NIAH tasks.

##### 2.3. High-Resolution Video Grid QA

In this task, we explore the idea of generating highresolution video instruction data by combining multiple low-resolution videos. From a large collection of low-

resolution video clips, we randomly sample 64 videos and arrange them in a 8×8 grid (c.f. Figure 2-G). Each video is resized to 240 × 135, resulting in a combined video resolution of 1920 × 1080. We then randomly select a cell at row i and column j in the video grid and synthesize a question about the content in this specific cell. This design enhances video LMM’s ability to understand high-resolution videos by requiring it to locate the correct cell based on indices and accurately interpret the content within that small area. For MCQ data, we randomly select other cells within the grid and use their captions to create distractor options.

##### 2.4. Summary of Video Instruction-Following Data

Table 1 provides a summary of our video instructionfollowing dataset. Our dataset comprises ∼400K entries, with each long video over 30 seconds and each highresolution video at least 960p. A key advantage of our data synthesis pipeline is that our QA synthesis process (c.f. Equation 1) requires only text processing via the Gemini API, without needing Gemini’s multimodal functionality. This makes our approach significantly more cost-efficient compared to existing methods [4, 65]. We leverage five datasets: Panda-70M [5], MiraData [14], FineVideo [8], InternVid [50] and OpenVid-1M [33] to generate synthetic video instruction data. However, our method also applies to any raw video source by first using an off-the-shelf video captioning model to generate simple captions, followed by our pipeline to produce high-quality video instruction data.

#### 3. Evaluation: HRVideoBench

We observe that existing video understanding benchmarks are inadequate for accurately assessing the ability of video LMMs to understand high-resolution videos, especially the details inside the videos. Prior benchmarks mainly consist of low-resolution videos. More recent benchmarks focus on evaluating the long video understanding capability of video LMMs, which contain questions that typically pertain to a short segment in the long video. As a result, a model’s high-resolution video understanding performance

Table 2. Comparisons between baseline models and VISTA-finetuned models on long/short video understanding benchmarks. The best results among open-source models are bolded. ∆ denotes the performance differences before and after finetuning on VISTA-400K.

Long Video Understanding Short Video Understanding Video-MME w/o subtitles MLVU LVBench LongVideoBench MVBench NExT-QA

Models Size

avg short medium long m-avg test val test mc

Proprietary Models GPT-4V [1] - 59.9 70.5 55.8 53.5 49.2 - 59.1 43.5 GPT-4o [35] - 71.9 80.0 70.3 65.3 64.6 34.7 66.7 - 76.0 Gemini-1.5-Pro [44] - 75.0 81.7 74.3 67.4 - 33.1 64.0 - -

Open-source Models VideoChat2 [23] 7B 39.5 48.3 37.0 33.2 47.9 - 39.3 51.9 78.6 LLaMA-VID [25] 7B - - - - 33.2 23.9 - 41.3 ST-LLM [29] 7B 37.9 45.7 36.8 31.3 - - - 54.9 ShareGPT4Video [4] 7B 39.9 48.3 36.3 35.0 46.4 - 39.7 51.2 LongVILA [55] 7B 50.5 61.8 50.4 46.2 - - - - LongLLaVA [49] 7B 52.9 61.9 51.4 45.4 - - - 54.6 Video-XL [41] 7B 55.5 64.0 53.2 49.2 64.9 - 49.5 55.3 77.2 VideoLLaVA [26] 7B 39.9 45.3 38.0 36.2 47.3 29.3 39.9 43.8 61.8 VISTA-VideoLLaVA 7B 43.7 48.2 43.9 38.9 49.5 33.8 42.3 47.2 63.0 ∆ - VideoLLaVA +3.8 +2.9 +5.9 +2.7 +2.2 +4.5 +2.4 +3.4 +1.2 Mantis-Idefics2 [13] 8B 45.4 55.9 43.0 37.2 49.4 35.0 45.8 51.4 75.8 VISTA-Mantis 8B 48.2 58.4 46.7 39.6 55.5 36.4 49.1 52.5 75.2 ∆ - Mantis-Idefics2 +2.8 +2.5 +3.7 +2.4 +6.1 +1.4 +3.3 +1.1 -0.6 LongVA [63] 7B 52.4 61.4 50.9 45.0 56.3 35.9 51.8 49.2 68.3 VISTA-LongVA 7B 55.5 66.0 53.1 47.4 62.1 39.0 53.1 51.1 69.3 ∆ - LongVA +3.1 +4.6 +2.2 +2.4 +5.8 +3.1 +1.3 +1.9 +1.0

can be undermined if it struggles to sample or retrieve the relevant frames from a lengthy video sequence.

To address this gap, we introduce HRVideoBench, a comprehensive benchmark with 200 multiple-choice questions designed to assess video LMMs for high-resolution video understanding. HRVideoBench focuses on the perception and understanding of small regions and subtle actions in the video. Our test videos are at least 1080p and contain 10 different video types collected with real-world applications in mind. For example, key applications of high-resolution video understanding include autonomous driving and video surveillance. We correspondingly collect POV driving videos and CCTV footage for the benchmark. Our benchmark consists of 10 types of questions, all of which are manually annotated and can be broadly categorized into object and action-related tasks. Further details can be found in the Appendix. The question types are:

- • Object-related Tasks: Object Counting, OCR problem, Object Recognition, Entity Recognition, Object Property Recognition, Object Status Change Recognition.
- • Action-related Tasks: Action Recognition, Moving Direction Identification, Interaction Detection, Temporal Sequence Recognition.

#### 4. Experimental Results

##### 4.1. Evaluation Setup

To validate the effectiveness of VISTA-400K, we finetune a diverse set of LMMs on our dataset. Specifically, we choose VideoLLaVA [26], Mantis-Idefics2 [13] and LongVA [63] as the base models because these models disclose details about their training dataset. Further details of these models can be found in the Appendix. We directly finetune LongVA and Mantis-Idefics2 on our dataset as they are mainly pretrained on image data. For VideoLLaVA, we additionally include 300K short video from VideoChat2-IT [23] to preserve the model’s short video understanding capability. We use 8 frames for VideoLLaVA, 24 frames for Mantis-Idefics2, and 64 frames for LongVA for finetuning and evaluation. The input resolution is fixed at 224×224 for VideoLLaVA and 336×336 for LongVA, while MantisIdefics2 supports dynamic resolutions up to 980×980. The finetuned model is denoted as VISTA-[base model].

Our evaluation assesses video LMMs’ capabilities in long and high-resolution video understanding. We test the finetuned models on four comprehensive long video understanding benchmarks: Video-MME [10],

Table 3. Quantitative results on HRVideoBench and open-ended video QA benchmarks. “acc.” represents accuracy.

High-Res Video Understanding Open-Ended Video QA

HRVideoBench MSVD-QA MSRVTT-QA TGIF-QA ActivityNet-QA Models

avg object action acc. score acc. score acc. score acc. score

VideoLLaVA [26] 32.5 36.0 27.9 60.3 3.7 42.1 3.0 63.5 3.8 48.6 3.3 VISTA-VideoLLaVA 47.5 50.0 44.2 71.5 4.0 58.5 3.5 78.0 4.3 49.1 3.4 ∆ - VideoLLaVA +15.0 +14.0 +16.3 +11.2 +0.3 +16.4 +0.5 +14.5 +0.5 +0.5 +0.1

Mantis-Idefics2 [13] 48.5 50.9 45.4 57.4 3.5 34.9 2.7 65.7 3.8 46.5 3.1 VISTA-Mantis 51.0 53.5 47.7 65.2 3.8 46.4 3.1 71.4 4.0 48.8 3.3 ∆ - Mantis +2.5 +2.6 +2.3 +7.8 +0.3 +11.5 +0.4 +5.7 +0.2 +2.3 +0.2

LongVA [63] 48.0 52.6 41.9 56.3 3.5 37.7 2.8 55.4 3.4 48.0 3.2 VISTA-LongVA 50.0 56.1 41.9 61.0 3.7 42.5 3.0 67.5 3.9 51.8 3.4 ∆ - LongVA +2.0 +3.5 +0.0 +4.7 +0.2 +4.8 +0.2 +12.1 +0.5 +3.8 +0.2

MLVU [66], LVBench [47] and LongVideoBench [51]. For high-resolution video understanding, we utilize our HRVideoBench. Additionally, we report results on prior short video understanding benchmarks including MVBench [23] and NExT-QA [52], as well as open-ended video QA benchmarks such as MSVD-QA [53], MSRVTT-QA [53], TGIF-QA [12] and ActivityNet-QA [58].

##### 4.2. Quantitative Results

Long Video Understanding Quantitative results for long video understanding are shown in Table 2. All three models demonstrate a consistent performance boost across the evaluation benchmarks, with average improvements of 3.3% for Video-MME, 4.7% for MLVU, 3.0% for LVBench, and 2.3% for LongVideoBench. For Video-MME, our method yields the largest average improvement for medium-length questions, which correspond to videos ranging from 4 to 15 minutes. Given that most of the augmented videos in our dataset are under 3 minutes, this result suggests that our approach generalizes effectively to longer videos. One possible explanation for this observation is that longer videos often contain more scene cuts and transitions, which is aligned with our synthetic data where multiple clips with different scenes are concatenated. Our VISTA-LongVA achieves state-of-the-art performance on Video-MME, LVBench and LongVideoBench among open-source models, demonstrating the effectiveness of our approach.

High-Resolution Video Understanding We present the HRVideoBench evaluation results for VISTA-finetuned models and the baseline models in Table 3. Among the baseline models, we observe that Mantis-Idefics2, with its higher input video resolution, outperforms LongVA on both object and action-related tasks, despite being considered weaker for long video understanding. This suggests that our HRVideoBench requires detailed comprehension of high-resolution videos, and simply adding more input

frames during testing does not necessarily lead to enhanced model performance. Overall, we find that finetuning on our dataset enhances the high-resolution video understanding capabilities for all three models. Additionally, we observe that recognizing subtle actions remains more challenging than identifying objects and their properties, suggesting that there remains significant potential for improvement in highresolution video understanding for video LMMs.

Short Video Understanding We further validate our proposed method on several prior short video understanding benchmarks. According to Table 2, all three VISTAfinetuned models show improvement on both MVBench and NExT-QA benchmarks except Mantis. This result is likely due to NExT-QA’s training split being included as part of the in-domain training data for Mantis-Idefics2. Consequently, finetuning on our out-of-domain data causes a slight decrease in performance. While our results are slightly worse than Video-XL [41], we note that their pretraining and instruction-tuning datasets are larger (>2.7M training samples). Here, we focus more on the relative improvement of our method over the baseline models.

Since the benchmarks mentioned above consist solely of multiple-choice questions, we also include several openended QA benchmarks to verify the text generation capabilities of the finetuned models. Following Video-ChatGPT [32], we use GPT-3.5-Turbo [34] to evaluate the accuracy of the responses and rate their quality on a scale of 1 to 5. As shown in Table 3, finetuning on our dataset improves both accuracy and generation quality across all four benchmarks for the VISTA-finetuned models. This demonstrates that our dataset can enhance the generation quality of LMMs.

##### 4.3. Ablation Study

To verify that each subset in VISTA-400K contributes to the performance gain of long or high-resolution video understanding, we conduct an ablation study and train sev-

Table 4. Ablation study results for VISTA-Mantis. Each “w/o [Subset]” denotes a Mantis-Idefics2 model finetuned on a modified VISTA-400K by replacing the corresponding subset with the same amount of training examples from VideoChat2-IT [23].

Video-MME HRVideoBench w/o sub. avg avg VISTA-Mantis 48.2 51.0 w/o Long Video Captioning 47.9 48.0 w/o Event Relationship QA 47.7 49.5 w/o Temporal NIAH 47.5 48.0 w/o Two Needle NIAH 48.1 50.5 w/o Spatial NIAH 47.2 47.5 w/o Spatiotemporal NIAH 47.7 50.0 w/o HR Video Grid QA 47.8 48.0 w/o Video Augmentation 45.7 44.5

Models

eral Mantis-Idefics2 models by excluding each subset from the training data. In addition, we add the same amount of video training data from VideoChat2-IT [23] to each of the ablation experiments to ensure the total number of training examples does not change. The results on VideoMME and HRVideoBench are shown in Table 4. We observe that excluding the long video subsets decreases the model performances on Video-MME. Similarly, excluding high-resolution video subsets leads to a drop in scores on HRVideoBench. These observations indicate that all seven video augmentation methods are effective at their corresponding tasks. We also report results for a Mantis-Idefics2 model without our proposed video augmentations by using the original videos directly for training before combining them with others (e.g., using the needle video in NIAH tasks to train the model directly). According to the last row in Table 4, this yields very low scores on Video-MME and HRVideoBench, highlighting the necessity of our augmentations to enhance video instruction data quality.

Furthermore, previous studies [20, 63] have shown that training on high-resolution image data yields strong long video understanding models. Our ablation study supports this conclusion and demonstrates that the proposed highresolution video augmentation methods enhance the performances of long video understanding tasks. In addition, our results indicate that the reverse is also true: our long video augmentation methods also improve the performance of high-resolution video understanding. We believe this transferability between long and high-resolution video understanding capabilities is because both types of inputs convert to a long sequence of video tokens in video LMM’s embedding space. As a result, identifying short moments from a long video and recognizing small regions in a highresolution video both correspond to the task of finding key information needles in a haystack of video tokens.

##### 4.4. Case Study

To qualitatively evaluate our method, we compare the generated responses of VISTA-finetuned models with the baseline models, as illustrated in Figure 3. The “helicopter” example involves recognizing event orders within a long video sequence. The baseline models LongVA and VideoLLaVA exhibit hallucination issues, while our VISTALongVA describes the action accurately. Due to the sparse sampling rate of VideoLLaVA (8 frames per video), VISTAVideoLLaVA misses frames related to the action immediately following “the helicopter takes off”. Nevertheless, it successfully describes the later event, “the helicopter crashes onto a rooftop”, which the vanilla VideoLLaVA fails to recognize. The “table tennis” example challenges video LMMs to identify a subtle and unusual action within a high-resolution video. As demonstrated in the second part of Figure 3, Mantis-Idefics2 fails to recognize this action, instead generating a response based on common knowledge about table tennis. Conversely, VISTA-Mantis correctly identifies the unusual action that the player hits the table tennis ball with his leg.

#### 5. Related Work

LMMs for Video Understanding Recent research has advanced the ability of large multimodal models (LMMs) to process video inputs and generate natural language responses. Earlier works on video LMMs such as VideoChatGPT [32] and Video-LLaVA [26] employ video encoders that are restricted to a limited number of low-resolution frames (e.g. 224 × 224, eight frames). To enable the processing of longer and higher-resolution videos, several studies [9, 22, 23, 25, 28, 38, 42, 62] have developed token compression and pooling methods to manage the sequence length of video tokens. Video-LLaMA [62], VideoChat [22] and VideoChat2 [23] utilize Q-Former [21] in the visual encoder to compress video tokens into a few learnable queries. Video-CCAM [9] further applies causal crossattention masks within the video Q-Former to better capture the temporal order of the video frames. In addition to using Q-Former, VideoLLaMA 2 [6] designs a 3D convolution-based STC Connector for efficient token compression. LongVA [63] alternatively increases the maximum number of tokens the model can receive by training a long-context LLM with a 128K context length, allowing the model to handle longer video sequences.

Video Language Datasets Previous efforts on building video-text datasets have typically focused on a specialized domain, such as video retrieval [2, 17, 54], video question answering [52, 53, 58] and action recognition [3, 11, 43]. VideoChat2 [23] creates a large-scale video instruction tuning dataset by combining multiple specialized datasets and rewriting instructions using ChatGPT [34]. However, the

|[Figure 45]|
|---|

|[Figure 46]|
|---|

[Figure 47]

[Figure 48]

[Figure 49]

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

[Figure 53]

[Figure 54]

Question: What happens after the helicopter takes off in this video?

LongVA: In the video, after the helicopter takes off, it appears to be flying over a cityscape. ...

VideoLLaVA: ...The man in the video is seen holding a sword and fighting with another man. ...

VISTA-LongVA: A man is seen jumping off a building onto a helicopter.

VISTA-VideoLLaVA: The helicopter crashes onto a rooftop, creating a large cloud of smoke and debris.

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

[Figure 58]

[Figure 59]

Question: What is unusual about the way of the player in the red jersey hitting the table tennis ball?

Mantis-Idefics2: The player in the red jersey is hitting the table tennis ball with a spin, which is unusual because...

VISTA-Mantis: The player in the red jersey is hitting the table tennis ball with his leg.

- Figure 3. Qualitative comparisons between the baseline models and our VISTA-finetuned models. Red text indicates hallucinations or incorrect responses, while green text highlights the correct responses that correspond accurately to the video content.

quality of the videos is suboptimal. More recently, LLaVAHound [64] and ShareGPT4Video [4] generate high-quality video captioning and open-ended QA data using GPT-4V [1]. Nonetheless, the video sources in these datasets often have limited motion degree and variety.

A concurrent study to ours is the LLaVA-Video-178K dataset [65], which contains 178K high-quality video data and 1.3M instruction-following samples annotated by GPT4o [35]. However, this dataset relies on a costly annotation process that involves feeding in video frames to GPT-4o at one fps and generating three levels of video descriptions. In contrast, our work emphasizes the synthesis of high-quality video data from existing short or low-resolution videos, offering a more scalable and cost-effective solution for generating synthetic video language data.

Video Understanding Benchmarks The advancement of video LMMs has also driven the creation of various video understanding benchmarks. MVBench [23] introduces a comprehensive video LMM benchmark by regenerating question-answer pairs based on various existing benchmarks. TempCompass [30] and VideoVista [24] evaluate the temporal reasoning and understanding capabilities of video LMMs. Recently, new benchmarks have emerged to assess video LMMs’ ability to comprehend extremely long

videos, including Video-MME [10], MLVU [66], LVBench [47] and LongVideoBench [51]. Despite these advancements, we note that no existing benchmark specifically evaluates video LMMs’ ability to understand high-resolution videos, which is a crucial task in video understanding that has significant applications in fields such as autonomous driving and sports analysis. In this study, we aim to address this gap by collecting a comprehensive benchmark for high-resolution video understanding targeting small objects and localized actions within videos.

#### 6. Conclusion

We presented VISTA, a video augmentation framework to enhance long-duration and high-resolution video understanding by generating high-quality synthetic video instruction-following data from existing video-caption datasets. VISTA performs spatiotemporal video combination over input videos to create new video samples and utilizes video captions for instruction synthesis. Through extensive experiments on various long video understanding benchmarks and our proposed HRVideoBench, we demonstrated that VISTA-400K consistently improves the performance of three different video LMMs. For future work, we aim to design and explore additional video augmentation methods to further strengthen our approach’s robustness.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 5, 8

- [2] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017. 7
- [3] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 1, 7
- [4] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 1, 4, 5, 8
- [5] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 2, 4
- [6] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 1, 7
- [7] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. 1
- [8] Miquel Farr´e, Andi Marafioti, Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. Finevideo. https: //huggingface.co/datasets/HuggingFaceFV/ finevideo, 2024. 2, 4
- [9] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. arXiv preprint arXiv:2408.14023, 2024. 7
- [10] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 5, 8, 1
- [11] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 7
- [12] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in

- visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2758–2766, 2017. 6, 2
- [13] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024. 5, 6, 1
- [14] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. arXiv preprint arXiv:2407.06358,

2024. 4

- [15] Garrett Kamradt. Llmtest needleinahaystack. https : / / github . com / gkamradt / LLMTest _ NeedleInAHaystack, 2024. Accessed: 2024-10-24. 3

- [16] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 1
- [17] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 7
- [18] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 1
- [19] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-ofexperts model. arXiv preprint arXiv:2410.05993, 2024. 2
- [20] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 7
- [21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 7

- [22] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 7
- [23] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 1, 5, 6, 7, 8, 2
- [24] Yunxin Li, Xinyu Chen, Baotian Hu, Longyue Wang, Haoyuan Shi, and Min Zhang. Videovista: A versatile benchmark for video understanding and reasoning. arXiv preprint arXiv:2406.11303, 2024. 8
- [25] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2025. 5, 7
- [26] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual represen-

- tation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1, 5, 6, 7
- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1
- [28] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542,

2024. 2, 7

- [29] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li St-llm. Large language models are effective temporal learners. arXiv preprint arXiv:2404.00308, 2024. 5
- [30] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 8
- [31] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Da Li, Pengcheng Lu, Tao Wang, Linmei Hu, Minghui Qiu, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207,

2023. 1

- [32] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 1, 6, 7, 2
- [33] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 4
- [34] OpenAI. Chatgpt. https://openai.com/index/ chatgpt/, 2023. 1, 6, 7
- [35] OpenAI. Gpt-4o. https://openai.com/index/ hello-gpt-4o/, 2024. 5, 8
- [36] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. 1
- [37] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020. 1
- [38] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 7
- [39] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 1

- [40] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 1
- [41] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. arXiv preprint arXiv:2409.14485, 2024. 5, 6
- [42] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 7
- [43] K Soomro. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402,

2012. 7

- [44] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 5
- [45] Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja Ganu, and Hao Wang. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. arXiv preprint arXiv:2406.11230, 2024. 3
- [46] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2
- [47] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024. 6, 8, 2
- [48] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, et al. Needle in a multimodal haystack. arXiv preprint arXiv:2406.07230, 2024. 3
- [49] Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang, and Benyou Wang. Longllava: Scaling multi-modal llms to 1000 images efficiently via hybrid architecture. arXiv preprint arXiv:2409.02889, 2024. 5
- [50] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 2, 4
- [51] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024. 6, 8, 2
- [52] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining

- temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 6, 7, 2
- [53] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017. 6, 7, 2
- [54] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 1, 7
- [55] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 5
- [56] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [57] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 2
- [58] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9127–9134, 2019. 6, 7, 2
- [59] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6023–6032, 2019. 2
- [60] Sangdoo Yun, Seong Joon Oh, Byeongho Heo, Dongyoon Han, and Jinhyung Kim. Videomix: Rethinking data augmentation for video classification. arXiv preprint arXiv:2012.03457, 2020. 2
- [61] Hongyi Zhang. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 2
- [62] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 7
- [63] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 5, 6, 7, 1
- [64] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024. 8
- [65] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 4, 8

- [66] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264,

2024. 6, 8, 1

- [67] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. 1

## VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by VIdeo SpatioTemporal Augmentation

### Supplementary Material

#### 7. HRVideoBench Details

As detailed in Section 3, our HRVideoBench consists of 200 questions covering 10 question types and 10 video types. That is, we collect two questions for each combination of video type and question type. The 10 video types are:

- • POV driving videos
- • Egocentric sports videos
- • Sportscast videos (broadcasting of sports events)
- • Public event recordings
- • Surveillance camera/CCTV footage
- • Wildlife stock videos
- • Aerial videos/Drone videos
- • Factory and industrial stock videos
- • Public transport videos
- • Product review videos For each question in the benchmark, we ensure the video

duration falls between 3 to 10 seconds. This relatively short duration is chosen to maximize the likelihood of the frames relevant to the question getting sampled by the models. The final dataset has an average video duration of 5.4 seconds and an average resolution of 3048×1699. Example questions and answers from our HRVideoBench are shown in Figure 4.

#### 8. Model Training and Evaluation Details

In this section, we provide additional details for training and benchmarking our selected baseline models.

##### 8.1. Baseline Models

VideoLLaVA [26] is a video LMM jointly pretrained on image and video data. It uses the pretrained Vicuna v1.5 model as its LLM backbone and LanguageBind as its image and video encoder. The model is pretrained on 558K imagetext pairs from LAION-CC-SBU [36, 39, 40] and 702K video-text pairs from Valley [31]. During the instructiontuning stage, it incorporates 665K image-text pairs from LLaVA-1.5 [27] and 100K video-text pairs from VideoChatGPT [32].

Mantis-Idefics2 [13] is an LMM specialized in processing inputs with multiple interleaved images. It is initialized from Idefics2 [18] and continually pretrained on MantisInstruct, a dataset comprising 721K interleaved image-text instruction-tuning examples. This dataset focuses on enhancing multi-image understanding across four dimensions: co-reference, comparison, reasoning, and temporal understanding. Mantis-Idefics2 achieves state-of-the-art perfor-

mance on various multi-image benchmarks and excels on short video understanding benchmarks, such as MVBench [23].

LongVA [63] is a long-context LMM designed for understanding long video content. It first performs continual pretraining using a Qwen2 [56] model to support up to 224K context length. Following this, it uses this modified Qwen2 model as the backbone for visual instruction tuning. LongVA is instruction-tuned on pure image data, using the same training data as LLaVA-1.6 [27]. It introduces the UniRes strategy, which divides an image into multiple grids and encodes each grid independently using the vision encoder. During inference, these grids are replaced by different frames from the input video, enabling effective processing of long video sequences.

##### 8.2. Additional Implementation Details

For all three models, we conduct full-finetuning for one epoch using 8 Nvidia H800 GPUs. The total training time for ∼400K data is around one day. We use the Adam [16] optimizer with a batch size of 128 during training. The learning rate is set to 5e-6 for VideoLLaVA and 1e-7 for LongVA and Mantis-Idefics2, with a cosine learning rate scheduler and a warm-up ratio of 0.03 applied to all models. We employ Flash-Attention 2 [7] and DeepSpeed ZeRO-3 [37] to accelerate training.

##### 8.3. Evaluation Benchmarks

Video-MME [10] is a comprehensive benchmark designed to evaluate the video analysis capabilities of LMMs. It includes 900 videos and 2700 questions across six visual domains. The questions are categorized based on video durations into short, medium, and long video questions, with median durations of 26s, 164.7s, and 890.7s, respectively. The median duration values for short, medium and long video questions are 26s, 164.7s, and 890.7s, respectively. Video-MME supports two evaluation formats: (1) the “w/ subtitle” format, which includes both the video subtitles and questions as text inputs, and (2) the “w/o subtitle” format, which uses only the raw video and questions as inputs. In the main paper, we focus on the “w/o subtitle” format to emphasize improving the long video understanding capabilities of video LMMs through video augmentation, rather than relying on additional subtitle information. For completeness, we provide results for the “w/ subtitle” format in Section 9.

MLVU [66] is a long video understanding benchmark en-

[Figure 60]

[Figure 61]

[Figure 62]

Question: What is the color of the car in the left rearview mirror? A. red, B. green, C. yellow, D. orange. Answer: A. red.

[Figure 63]

[Figure 64]

[Figure 65]

Question: A person is in the bottom right of the video using a green umbrella. In which direction is this person going? A. bottom left, B. top right, C. bottom right, D. top left.

Answer: D. top left.

Figure 4. Example questions from our HRVideoBench. Zoom in for better visualizations.

compassing diverse tasks and video genres. It features two types of questions: multiple-choice questions and freeform generation questions. The benchmark evaluates LMMs across three dimensions: holistic video understanding, requiring global information from the entire video; singledetail video understanding, focused on short and salient moments within the video; and multi-detail video understanding, involving connections across multiple short clips in the video. In this paper, we report the accuracy scores for the multiple-choice questions from the development set of MLVU. In the paper, we report the accuracy scores for the multiple-choice questions from the dev set of MLVU.

LVBench [47] evaluates the comprehension capabilities of video LMMs for extremely long videos. It consists of 1549 QA pairs, with an average video duration of 4101 seconds. The benchmark assesses video LMMs across six core aspects: temporal grounding, video summarization, video reasoning, entity recognition, event understanding, and key information retrieval. We use the full test set for evaluation.

LongVideoBench [51] is a question-answering benchmark featuring interleaved long video-text input. The dataset contains 3763 videos and 6678 human-annotated multiple-choice questions spanning 17 fine-grained categories. LongVideoBench supports two evaluation formats: the standard input format, where video tokens are processed first followed by question descriptions, and an interleaved video-text format, where subtitles are inserted between video frames. Although Mantis-Idefics2 supports in-

terleaved image-text input, as our VISTA-400K does not include training examples in such format, we still evaluate Mantis-Idefics2 and the finetuned VISTA-Mantis using the standard format. We report the results of the validation split.

MVBench and NExT-QA [23, 52] are short video understanding benchmarks, focusing on videos under one minute in duration. MVBench includes 4,000 multiple-choice questions derived from 3,641 video clips, with an average video duration of 16 seconds. NExT-QA comprises 8,564 questions (both multiple-choice and open-ended) sourced from 1,000 videos, averaging 40 seconds in length. In our experiments, we evaluate the models on the full MVBench dataset and the MCQ split of NExT-QA.

MSVD-QA, MSRVTT-QA, TGIF-QA and ActivityNetQA [12, 53, 58] are open-ended QA benchmarks designed to evaluate the response generation capabilities of video LMMs. These benchmarks consist of short videos and assess the ability of video LMMs to produce simple, coherent answers. For all four benchmarks, we follow VideoChatGPT [32] and use GPT-3.5-Turbo to evaluate the accuracy and quality of the responses. Specifically, GPT is prompted with the ground truth answer and the model’s response to determine if the answer is correct (yes/no) and to assign a quality score between 1 and 5. Following VideoChatGPT, we evaluate the models on the validation sets of MSVD-QA, MSRVTT-QA and ActivityNet-QA, and use the FrameQA split from TGIF-QA’s test set for evaluation. Since GPT-3.5-Turbo’s API version has changed and the

- Table 5. Comparison between the baseline VideoLLaVA model, VideoLLaVA finetuned on VISTA-400K and VideoLLaVA finetuned on VISTA-400K + 300K VideoChat2-IT data (VISTA-VideoLLaVA in the main paper) on long video understanding benchmarks. “SFT” indicates supervised finetuning.

Long Video Understanding

Video-MME w/o subtitles MLVU LVBench LongVideoBench Models

avg short medium long m-avg test val

VideoLLaVA 39.9 45.3 38.0 36.2 45.0 29.3 39.1 VideoLLaVA (SFT on VISTA-400K) 43.6 47.3 43.8 39.8 48.7 32.6 41.0 ∆ - VideoLLaVA +3.7 +2.0 +5.8 +3.6 +3.7 +3.3 +1.9 VideoLLaVA (SFT on VISTA-400K + 300K VideoChat2-IT) 43.7 48.2 43.9 38.9 49.5 33.8 42.3 ∆ - VideoLLaVA (SFT on VISTA-400K) +0.1 +0.9 +0.1 -0.9 +0.8 +1.2 +1.3

- Table 6. Comparison between the baseline VideoLLaVA model, VideoLLaVA finetuned on VISTA-400K and VideoLLaVA finetuned on VISTA-400K + 300K VideoChat2-IT data (VISTA-VideoLLaVA in the main paper) on HRVideoBench. “SFT” indicates supervised finetuning.

High-Resolution Video Understanding

HRVideoBench Models

avg object action

VideoLLaVA 32.5 36.0 27.9 VideoLLaVA (SFT on VISTA-400K) 44.0 42.1 46.5 ∆ - VideoLLaVA +11.5 +6.1 +18.6

VideoLLaVA (SFT on VISTA-400K + 300K VideoChat2-IT) 47.5 50 44.2 ∆ - VideoLLaVA (SFT on VISTA-400K) +3.5 +7.9 -2.3

older API versions are no longer accessible, we are unable to reproduce the results for some baseline models. In the paper, we report all scores based on our evaluation script.

#### 9. Additional Experimental Results

##### 9.1. Training Data Ablations for VideoLLaVA

As mentioned in Section 4.1, unlike Mantis-Idefics2 and LongVA, we fine-tune VideoLLaVA using a combination of our VISTA-400K and 300K short video samples from VideoChat2-IT to preserve its short video understanding capabilities. In this section, we examine how this additional training data impacts the model’s performance on long and high-resolution video understanding tasks after finetuning. To assess this, we finetune another VideoLLaVA model exclusively on our VISTA-400K and compare the results against the combined training approach in Table 5 and Table 6.

As shown in Table 5, finetuning VideoLLaVA exclusively on our VISTA-400K results in consistent improvements across all long video understanding benchmarks. On the other hand, incorporating an additional 300K short video samples does not yield further significant gains in long video understanding. Notably, the Video-MME results

indicate that adding short video data slightly detracts from the model’s performance on long videos, underscoring the importance of our dataset for enhancing long video understanding capabilities.

For high-resolution video understanding, according to Table 6, finetuning VideoLLaVA on our data leads to a significant improvement (+11.5%) on HRVideoBench. While adding additional short video data further enhances model performance, the improvement is less substantial. These findings suggest that our dataset remains the primary driver of performance gains in high-resolution video understanding. Moreover, incorporating VideoChat2-IT training data leads to a decline in performance on action-related questions, highlighting the superior effectiveness of our dataset for tasks requiring temporal understanding.

##### 9.2. Video-MME w/ Subtitles Results

We show the results for Video-MME w/ subtitles in Table 7. In this evaluation setting, the video’s subtitles are provided as part of the question input to the model. The results indicate that both baseline models and our VISTAfinetuned models can be further enhanced by providing extra subtitle information. Similar to Video-MME w/o subtitles results, our VISTA-finetuned models consistently

- Table 7. Comparison between VISTA-finetuned models and baseline models on Video-MME w/ subtitle benchmark.

Video-MME w/ subtitles Models

avg short medium long

VideoLLaVA 41.6 46.1 40.7 38.1 VISTA-VideoLLaVA 45.1 50.2 45.7 39.3 ∆ - VideoLLaVA +3.5 +4.1 +5.0 +1.2

Mantis-Idefics2 49.0 60.4 46.1 40.3 VISTA-Mantis 50.9 61.8 48.6 42.3 ∆ - Mantis-Idefics2 +1.9 +1.4 +2.5 +2.0

LongVA 54.3 61.6 53.6 47.6 VISTA-LongVA 59.3 70.0 57.6 50.3 ∆ - LongVA +5.0 +8.4 +4.0 +2.7

achieve better performances compared to the baseline models. This shows that our synthetic data provides consistent and model-agnostic enhancements to the long video understanding capability of video LMMs.

#### 10. Limitations

Our method exhibits a few limitations. First, since we generate instruction data based on video captions, and most public video-caption datasets contain simple captions for video clips, our synthesized data often contain short responses, leading to a shorter response from the finetuned models. This issue could be addressed by recaptioning the raw video data using high-capacity video captioning models. Second, while our synthesized augmented video data have been shown to enhance long and high-resolution video understanding, the current video augmentation paradigm does not fully align with real-world video distributions. Addressing this limitation would require more advanced video combination and blending techniques, such as leveraging segmentation maps to isolate specific regions from one video and seamlessly integrating them into another to create more natural and realistic augmented video samples.

#### 11. Instruction Synthesis Prompt Templates

In this section, we list the Gemini prompts we used to synthesize instruction data below.

Freeform QA Generation Prompt User: Given a short paragraph of caption describing a video clip, can you try to extract relevant information from the caption and come up with a question-answer pair that could possibly reflect the facts of some local and fine-grained scenes in the video? The caption of the video is as follows: <Video Caption>

Please try not to come up with questions that you cannot answer. Please also note that the caption will not be presented in the actual training data. Return only the question and the answer. Format your output as: ###Question### <your question> ###Answer### <your answer>

Assistant: <Synthesized Freeform QA pairs>

MCQ Generation Prompt User:

Given the following Question-Answer pair, turn this short answer question into a multiple-choice question by synthesizing three additional incorrect options. Assume the correct option is <Random Option between A to D>.

Question: <Question> Answer: <Freeform Answer>

Your output should be in the format of a python list: [

- “A. <answer1>”,
- “B. <answer2>”,
- “C. <answer3>”,
- “D. <answer4>” ]

Assistant: <Synthesized MCQ pairs>

Event Relationship QA Generation Prompt User: Given multiple short captions, each representing a short chunk of video in a longer video, generate a question-answer pair related to the order of the events in the video. Note that because the short captions are from the same video, you can combine entities with slightly different descriptions in different captions, as they most likely represent the same thing. Format the output using the following format: ###Question### <Your question> ###Answer### <Your answer> For example, given captions like:

- Caption 1: A squirrel is sitting on a tree branch in a forest, surrounded by pine trees and blue sky.
- Caption 2: A cartoon squirrel is holding an egg in a tree.
- Caption 3: A cartoon squirrel is standing next to an egg. Your output can be: ###Question### What happens after the squirrel sits on a tree branch? ###Answer### The squirrel holds an egg.

Try to be creative with your question and answer.

The short captions (in chronological order) are listed below:

- Caption 1. <Caption 1>
- Caption 2. <Caption 2>

... Caption N. <Caption N>

Assistant: <Synthesized Event Relationship QA pairs>

Long Video Caption Generation Prompt User:

Given multiple short captions, each representing a short chunk of video in a longer video, create a detailed caption by combining the short captions such that the detailed caption describes the whole video. Note that because the short captions are from the same video, you can combine entities with slightly different descriptions in different captions, as they most likely represent the same thing. Return only the caption.

The short captions (in chronological order) are listed below:

- Caption 1. <Caption 1>
- Caption 2. <Caption 2>

... Caption N. <Caption N>

Assistant: <Synthesized Long Video Caption>

