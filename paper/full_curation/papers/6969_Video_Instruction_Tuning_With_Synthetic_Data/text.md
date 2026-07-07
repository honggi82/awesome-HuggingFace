# arXiv:2410.02713v3[cs.CV]1Aug2025

## LLaVA-Video: Video Instruction Tuning With Synthetic Data

Yuanhan Zhang yuanhan002@e.ntu.edu.sg S-Lab, Nanyang Technological University

Jinming Wu wjm_18@bupt.edu.cn BUPT

Wei Li liwei.speech@bytedance.com ByteDance

Bo Li libo0013@e.ntu.edu.sg S-Lab, Nanyang Technological University

Zejun Ma mazejun@bytedance.com ByteDance

Ziwei Liu ziwei.liu@ntu.edu.sg S-Lab, Nanyang Technological University

Chunyuan Li lichunyuan24@gmail.com ByteDance

Reviewed on OpenReview: https: // openreview. net/ forum? id= EElFGvt39K

### Abstract

The development of video large multimodal models (LMMs) has been hindered by the difficulty of curating large amounts of high-quality raw data from the web. To address this, we consider an alternative approach, creating a high-quality synthetic dataset specifically for video instruction-following, namely LLaVA-Video-178K. This dataset includes key tasks such as detailed captioning, open-ended question-answering (QA), and multiple-choice QA. By training on this proposed dataset, in combination with existing visual instruction tuning data, we introduce LLaVA-Video, a new video LMM. Our experiments demonstrate that LLaVA-Video achieves strong performance across various video benchmarks, highlighting the effectiveness of our dataset. We plan to release the dataset, its generation pipeline, and the model checkpoints.

### 1 Introduction

We are in an era where large-scale computing and data is crucial for multimodal learning (Li et al., 2024d). A significant recent advancement was introduced by visual instruction tuning (Liu et al., 2024a), which laid the foundation for building a general-purpose visual assistant. Notably, it proposed a data generation pipeline to create high-quality image-language instruction-following data. This pipeline has inspired subsequent researches (Li et al., 2024c;b;a; Lin et al., 2024) aimed at generating diverse image-language instruction data across various visual domains, accelerating the development of visual instruction tuning techniques.

Compared to the construction of image-language instruction-following data, obtaining high-quality videolanguage instruction-following data is challenging (Zhang et al., 2023; Li et al., 2024e). First, sourcing high-quality videos is difficult. We need to find videos with significant temporal changes that provide more knowledge than what image-language data can offer. However, we have found that most videos in current video-language instruction-following datasets (Chen et al., 2024a; Zhang et al., 2024d) are relatively

static. Additionally, these videos are mostly trimmed based on scene changes, resulting in simplified plots. Such simplified video-language instruction-tuning data is inadequate for models to understand videos with complex narratives. Furthermore, current video-language instruction-following datasets often use a very sparse sampling rate for frame annotation. For instance, ShareGPT4Video (Chen et al., 2024a) has an average sampling rate of 0.15, sometimes sampling only 2 frames from a 30-second video. This sparse sampling rate is effective in describing overall scenes but fails to capture detailed movements or changes in the video, resulting in hallucination when detailed descriptions of the video are required.

To overcome these shortcomings, we introduce a comprehensive video instruction-tuning dataset named LLaVA-Video-178K, consisting of 178,510 videos ranging from 0 to 3 minutes. This dataset is enriched with detailed annotations, open-ended questions, and multiple-choice questions, developed through a combination of GPT-4o (OpenAI, 2024) and human efforts. It features four favorable properties: (i) Extensive Video Source: We conduct a comprehensive survey on the video sources of exsiting video understanding datasets, and conclude 10 major video data sources, from which we start our video data collection by building a video pool. Although there are over 40 video-language datasets, their video data are mainly sourced from 10 datasets (Zhou & Corso, 2017; Xue et al., 2022; Goyal et al., 2017; Caba Heilbron et al., 2015; Kay et al., 2017; Sigurdsson et al., 2016; Wang et al., 2023; Shang et al., 2019; Grauman et al., 2022; Zhu et al., 2023a), covering a wide range of video domains, such as activities, cooking, TV shows, and egocentric views. (ii) Dynamic Untrimmed Video Selection: From these sources, we use several filtering logic to select the most dynamic videos from the video data pool. Notably, we select original, untrimmed videos to ensure plot completeness. (iii) Recurrent Detailed Caption Generation Pipeline with Dense Frame Sampling: We propose a detailed video caption pipeline that operates recurrently, enabling us to generate detailed captions for videos of any length. This pipeline has three levels, each level of description represents a different time-range: from 10 seconds to the entire video length. It is recurrent as the historical description from any level serves as the context for generating new descriptions at any level. Additionally, we adopted a dense sampling strategy of one frame per second to ensure the sampled frames are rich enough to represent the videos. (iv) Diverse Tasks: Based on the detailed video descriptions, we can generate question-answer pairs. To ensure our questions cover a wide range of scenarios, by referring to the video question-answering dataset, we define 16 question types. We prompt GPT-4o to generate question-answer pairs by referring to these question types, covering open-ended and multi-choice questions.

Based upon the LLaVA-Video-178K dataset, we developed LLaVA-Video. Contrary to previous studies suggesting that training with single frames is sufficient for video-language understanding (Lei et al., 2022), our findings reveal a significant impact of frame count on LLaVA-Video’s performance, attributable to the detailed features of LLaVA-Video-178K. Observing this, we explored maximizing frame sampling within the constraints of limited GPU memory. We introduce LLaVA-Video SlowFast, a video representation technique that optimally distributes visual tokens across different frames. This approach allows for incorporating up to three times more frames than traditional methods, which allocate an equal number of visual tokens to each frame.

Our contributions are as follows:

- • Video-language Instruction-Following Data: We present a high-quality dataset LLaVA-Video-178K tailored for video instruction-following. It consists of 178K video with 1.3M instruction samples, including detailed captions, free-form and multiple-choice question answering.
- • Video Large Multimodal Models: We develop LLaVA-Video, a series of advanced large video-language models that expand the capabilities of open models in understanding video content.
- • Open-Source: In an effort to support the development of general-purpose visual assistants, we release our multimodal instruction data, codebase, model checkpoints, and a visual chat demo to the public.

### 2 Related Work

In this work, our goal is to create a high-quality video-language dataset that goes beyond simple video captions. We aim to improve the ability to follow instructions, which includes detailed video descriptions, open-ended video question-answering, and multiple-choice video question-answering data. We discuss related datasets in

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|Source|Filtering Logic|
|---|---|
|HD-VILA-100M InternVid ActivityNet VIDAL Charades Ego4D VidOR YouCook2 Kinetics-700 Sth-sthv2<br><br>|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]|

|[Figure 33]|
|---|

Panda-70M

NextQA

Ego4D

HD-VILA-100M

DiDeMo

Kinetics-700

VidOR

VidSTR

VAST27M VidLN Vatex

EgoQA

Shot2Story

[Figure 34]

ShareGPT4Video

|[Figure 35]|
|---|

###### VIDAL

VTimeLLM

TextVR

VIdeoXum

ActivityNetQA

ActivityNet

LLaVA-Hound

VideoChat

VideoChatGPT

ShareGPT4O

###### Sth-sthv2

LLaVA-Hound

TextVR

ActivityNetCaption

InternVid

|[Figure 36]|
|---|

YouCook2

|[Figure 37]|
|---|

Sth-sthv2

STAR

|[Figure 38]|
|---|

Charades

|[Figure 39]|
|---|

Charades-STA

AGQA

- Figure 1: Video sources in the proposed LLaVA-Video-178K. (Left) The relationship between 10 video sources we have utilized and other existing video-language datasets. (Right) Filtering logic for video sources. The detail of filtering logic: ① Sorted by Views, ② Number of scenes greater than 2, ③ Video duration between 5 seconds and 180 seconds, ④ Ratio of scenes to video duration less than or equal to 0.5, ⑤ Resolution greater than 480p, ⑥ 50 samples for each category.

Table 1. Previous video-language datasets (Miech et al., 2019) include manually annotated data for various tasks, such as video captions (Chen & Dolan, 2011; Xu et al., 2016; Rohrbach et al., 2015; Anne Hendricks et al., 2017a; Caba Heilbron et al., 2015; Zhou & Corso, 2017), and video question-answering (Yu et al., 2019; Zadeh et al., 2019; Xiao et al., 2021). However, manual annotation is expensive and limits the size of such datasets. To address the shortage of data, studies like (Miech et al., 2019; Lee et al., 2021; Zellers et al., 2021; Xue et al., 2022) suggest automatically annotating data using subtitles created by ASR. While this method greatly expands the dataset size to 100 million samples, the subtitles often fail to accurately describe the main video content. Additionally, other studies (Xu et al., 2017; Grunde-McLaughlin et al., 2021; Wu et al., 2024a) use language models (Xu et al., 2017) or question templates (Grunde-McLaughlin et al., 2021; Wu et al., 2024a) to generate question-answer pairs. Although this approach can generate a large number of questions and answers, it often produces poor-quality questions that do not reflect real-world user inquiries. More recent research (Chen et al., 2024b) has prompted video-language models such as BLIP-2 (Li et al.,

- 2023), VideoChat (Li et al., 2024e), Video-LLaMA (Zhang et al., 2023), and MiniGPT-4 (Zhu et al., 2023b) to generate video captions. However, these models are limited in their ability to provide detailed descriptions.

The most related works to ours are the recent AI-generated synthetic video instruction tuning data, Islam et al. (2024) introduced Video ReCap, which recursively annotates video captions. Unlike Video ReCap, each clip-wise (level-1) description in our pipeline is generated with historical context. This ensures that connections from previous events in the video timeline are linked to the current event. LLaVA-Hound (Zhang et al., 2024d) and ShareGPT4Video (Chen et al., 2024a), where they have used GPT-4 (OpenAI, 2023) to generate video captions and open-ended video question-answering. Although the quality of the captions and question-answer pairs has significantly improved, the video sources they use are too static to produce high-quality data for instruction-following scenarios. They also only use very sparse frames for prompting GPT-4V, which results in annotations that fail to capture nuanced actions and continuous plots in the videos. Additionally, Shot2Story (Han et al., 2023) and Vript (Han et al., 2023) also employ GPT-4V (OpenAI, 2023) for video captioning. Their outputs, however, include audio details, which are outside the scope of this study.

### 3 Video Instruction-Following Data Synthesis

A high-quality dataset for video instruction-tuning is crucial for developing effective video-language models. We identify a key factor in building such datasets: ensuring richness and diversity in both video content and

its language annotations. We perform comprehensive survey on the existing video benchmarks, covering across various public video captioning and question-answering datasets, then identify ten unique video sources that contribute to over 40 video-language benchmarks. From each source, we select videos that exhibit significant temporal dynamics. To maintain diversity in the annotations, we establish a pipeline capable of generating detailed captions for videos of any length. Additionally, we define 16 types of questions that guide GPT-4o in creating question-answer pairs to assess the perceptual and reasoning skills of the video-language models.

#### 3.1 Video source

One important starting point in building a high-quality video instruction-following dataset is to find a sufficiently diverse pool of video data. From this pool, we can select the qualified videos. In our study of public video-language datasets—including video captioning, video question answering, video summarization, and moment-wise captioning—we noticed that although different datasets focus on various video understanding tasks (e.g., , AGQA (Grunde-McLaughlin et al., 2021) for spatial-temporal relations and STAR (Wu et al., 2024a) for situational reasoning), most are sourced from ten main video sources. For instance, both AGQA and STAR use data from Charades (Sigurdsson et al., 2016). Specifically, these ten sources are HD-VILA-100M (Xue et al., 2022), InternVid-10M (Wang et al., 2023), VidOR (Shang et al., 2019), VIDAL (YouTube Shorts)(Zhu et al., 2023a), YouCook2(Zhou & Corso, 2017), Charades (Sigurdsson et al., 2016), ActivityNet (Caba Heilbron et al., 2015), Kinetics-700 (Kay et al., 2017), Something-Something v2 (Goyal et al., 2017), and Ego4d (Grauman et al., 2022). These sources offer a wide range of video data from different websites, viewpoints, and domains. The relationship between these ten selected video datasets and others is shown in Fig. 1. The videos from this ten datsets build the video pool for the further video selection. Notably, we use untrimmed videos from each source except for YouCook2 and Kinetics-700. We believe that cutting videos into clips can break the plot continuity, which is essential for understanding the videos.

Based on the video pool, we aim to select dynamic videos. In Figure 1, we outline our criteria for selecting high-quality data. Our main method for identifying dynamic content involves using PySceneDetect, which calculates the number of scenes in a video We found that the number of scenes is a good indicator of video dynamism. Additionally, we have designed a specific approach ④ to exclude videos that mainly contain “slides."

#### 3.2 Video Detail Description

Automated Generation For selected videos, we use GPT-4o (OpenAI, 2024) to systematically describe their content. We start by sampling video frames at one frame per second (fps). However, due to the input size constraints of GPT-4o, we cannot use all sampled frames. Instead, we describe the videos sequentially, as shown in Fig 2. We create descriptions at three distinct levels, detailed below.

- • Level-1 Description: Every 10 seconds, we provide a level-1 description that outlines the events in that segment. This description considers: frames from the current clip and historical context, which includes all recent level-1 descriptions not yet summarized into a level-2 description and the latest level-2 description.
- • Level-2 Description: Every 30 seconds, we creat a level-2 summary of the entire video plot up to that point. This is based on the last three level-1 descriptions, covering the most recent 30 seconds; and the latest level-2 description.
- • Level-3 Description: At the video’s end, we generate a level-3 description to encapsulate the entire video. The inputs for this description are the recent level-1 descriptions not yet summarized, covering the last moments of the plot after the recent summary; and the latest level-2 description.

#### 3.3 Video Question Answering

Question Type definition In addition to detailed video descriptions, our dataset includes a variety of question-answer pairs designed for complex interactions. This setup improves the video understanding model’s ability to handle real-life queries. We refer to public video question-answering benchmarks (Xiao et al., 2021;

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

v

Video

Time

0s 5s 10s 15s 20s 25s 30s 35s 40s 45s 50s 55s 60s

Description

- Level-1
- Level-2
- Level-3

… t-2 t T

Video clip

[Figure 53]

[Figure 54]

[Figure 55]

LMM

LLM

LLM

Time interval

t-1 t

t-1

T-1 T

t

(a) Level-1 Description (b) Level-2 Description (c) Level-3 Description

- Figure 2: The video detail description creation pipeline. A three-level creation pipeline is considered, with each level developed via a recurrent approach. Note that t is the index of time internal at its own level, and T is the last time internal index. (a) To generate the caption for time internal t at level-1, we condition on the current frames in this internal, the caption for time internal t − 1, and the most recent description summary at level-2 if applicable. (b) To generate caption for time internal t at level-2, we condition on the previous caption at level-2, and captions from three most recent time internals at level-1. (c) To generate the overall caption at the last time internal T at level-3, we condition on the the most recent caption at level-2 and the current caption from level-1.

|Q: How do the audiences react after the child hits the pinata correctly?<br><br>Temporal|Spatial Q:What is behind the 8th man?|Q:Why do the little boy in red go towards woman in green at first?<br><br>Causal|Q:Which is faster, the white Speed car or the bicycle?|
|---|---|---|---|
|Q: Did the child wear shoes<br><br>Binary while running on the beach?|Q: How many times did the man put his right hand into his pocket?<br><br>Count|Q: How does the interaction between the monkey and the cat indicate?<br><br>Plot|Q:What colors are the railings of the staircase?<br><br>Description Object|
|Q:What actions did the person in the red hoodie carry out, and in what order?<br><br>Time Order|Q: Does the person in the video undergo a real physical transformation?<br><br>Fine-grain Action|Q:What is the reaction of the audience when the keynote speaker delivers his speech?<br><br>Object Existence|Description Human<br><br>Q:What does the person on the right's facial expression suggest?|
|Q: How do the ice cream change?<br><br>Attribute Change|Q: Is the camera following the joggers as they move?<br><br>Camera Direction|Q:Which direction did the man walk towards before exiting the scene relative to the camera?<br><br>Object Direction|Q:Where did the rescue operation in the video take place?<br><br>Description Scene|

- Figure 3: Question types for video question answering in data creation. For each type, we provide its name and an example question.

Yu et al., 2019; khattak et al., 2024; Liu et al., 2024b) to organize these questions into 16 specific categories, as shown in Fig. 3.

Automated Generation Given a detailed video description, we use GPT-4o to generate at most one question-answer pair for each type of question. The prompts include: (1) The task definition for the current question type. (2) In-context examples for this type, which include three video descriptions and their three question-answer pairs of this specific type. (3) The detailed video description for the current video. We instruct GPT-4o to return None if it cannot generate question-answer pairs for a specific question type.

Filtering. To filter out the generated question-answer pairs, we apply the following strategy: (1) remove duplicates using the sentence-transformer (Reimers & Gurevych, 2020), (2) discard answers that begin with

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

v

5s 10s 15s 20s 25s 30s 35s 40s 45s 50s 55s 60s Annotation type 1: detailed description

0s

###### Annotation type 2: open-ended question

The video begins with a black screen displaying the text ‘Normal People Vs Ultra’ in pink and white letters, accompanied by two smiling face emojis. The scene transitions to a modern building with a staircase. Three individuals, dressed in black suits and white sneakers, stand in a line on the stairs. The text ‘Normal’ appears in a red box at the top left corner. The individuals start walking up the stairs in a synchronized manner… <omited>

Question: How many steps does “normal people” climb? Answer: "Normal people" climb 7 steps in the video.

###### Annotation type 3: multi-choice question

Question: How many steps does “normal people” climb? A. 5 B. 6 C. 7 D.8 Answer: C.7

Figure 4: One example to illustrate the video instruction-following data.

[Figure 69]

[Figure 70]

[Figure 71]

###### Caption Open-endedQA Multi-choiceQA

VIDAL

VIDAL

VIDAL

VidOR YouCook2

VidOR

VidOR

YouCook2 Charades ActivityNet

YouCook2 Charades ActivityNet

Charades ActivityNet

HD-VILA-100M

HD-VILA-100M

HD-VILA-100M

-InternVid 10M

InternVid10M

-InternVid 10M

Ego4D

700

Kinetics-

Ego4D

Sthsth2

Ego4D

###### Dataset #Caption #Open-Ended #Multi-Choice Dataset #Caption #Open-Ended #Multi-Choice

VidOR 4,018 19,875 4,773 Sthsth2 8,700 0 0 YouCook2 7,411 32,143 5,776 Ego4D 1,065 5,912 520 Charades 9,803 48,187 13,401 InternVid-10M 45,000 245,840 48,246 ActivityNet 7,953 44,100 12,771 HD-VILA-100M 48,260 263,652 51,743 Kinetics-700 34,998 0 0 VIDAL 55,000 300,472 58,968

- Figure 5: Distribution of data across different datasets and question types (Caption, Open-ended, and Multi-Choice).

phrases like “does not specify,” “does not mention,” “does not specifically,” “does not depict,” or “does not show.”

#### 3.4 Dataset Statistics

Overview. We carefully select from our collected data sources to form a balanced and comprehensive collection, resulting in a total of 178K videos and 1.3M instruction-following samples. This includes 178K captions, 960K open-ended QAs, and 196K multiple-choice QAs.

Dataset Comparison We provide a comparison of high-quality instruction following video-language datasets, with a focus on synthetic data created with strong AI models, as shown in Table 1. (i) A broad collection of dynamic videos. In terms of video sources, although LLaVA-Hound (Zhang et al., 2024d) contains the largest number of videos, 44% of its video data are sourced from WebVid (Bain et al., 2021), where most

25K

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

10K

1.7K

1.5K

20K

#WordCount

#videoCount

8K

#videoCount

1.3K

15K

6K

- 0.8K
- 1.0K

10K

4K

0.5K 0.3K

5K

2K

180

Video Duration (s) #Word Count

Video Duration (s)

- Figure 6: (Left) Visualization of the video duration. (Middle) Visualization of the number of words in the video caption. (Right) Visualization of caption length versus video duration.

Beauty&StyleComedy

[Figure 72]

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

15633

4500

2628

513

261

0 10000 20000

Comedy

Sports

Beauty & style

Travel

Compupter Games

Distribution of 5 Uniformly Sampled VideoCategories

- Figure 7: (Left) Display of YouTube Shorts across four video categories. (Right) Distribution of 5 uniformly chosen video categories.

videos are static. ShareGPT4Video (Chen et al., 2024a) includes 30% of its videos from Pexels, Pixabay, and Mixkit, which are aesthetically good but also mostly static. Additionally, the majority of its videos come from Panda-70M, which are short clips from longer videos—suggesting simpler plots. In contrast, we carefully select video sources that offer dynamic, untrimmed videos with complex plots, which are crucial for developing a powerful video understanding model.1 (ii) High frames per second. Regarding frame sampling in language annotations, the proposed datasest considers 1 FPS, while other datasets consider much lower FPS. LLaVA-Hound uniformly samples 10 frames from videos of any length. The average FPS is 0.008, which may miss some fine details. ShareGPT4Video picks key frames using CLIP (Radford et al., 2021) based on frame uniqueness. This method might also miss subtle changes in the video because CLIP embeddings do not capture fine-grained dynamics well. Our method samples FPS=1 without using key frame selection algorithms, ensuring the detailed temproal information can be expressed in annotations and high coverage. (iii) Diverse tasks. The proposed dataset considers three common task types, including caption, free-form and closed-form QA, while existing datasets only consider a subset. Meanwhile, the quality and numbers of samples in our dataset is higher.

### 4 Experiments

We conducted evaluations for the LLaVA-Video models across all benchmarks using LMMs-Eval (Zhang et al., 2024a) to ensure standardization and reproducibility. To fairly compare with other leading video LMMs, we primarily used results from original papers. When results were not available, we integrated the models into LMMs-Eval and assessed them under consistent settings. Following LLaVA-OneVision (Li et al., 2024c), we employed SigLIP (Zhai et al., 2023) as our vision encoder, and Qwen2 (Yang et al., 2024) as the LLM. The

1Example videos: WebVid,Pixabay,Pexels,Mixkit.

- Table 1: Comparison of LLaVA-Video-178K and other video-language datasets. Average FPS represents the average number of frames per second that are used to prompt GPT-4o/GPT-4V for annotation.

★ VIDAL, WebVid, ActivityNet. ■ Panda-70M, Pexels, Pixabay, Mixkit, BDD100K, Ego4d. ✪ HD-VILA100M, Kinetics-700M, Ego4D, VidOR, InternVid, YouCook2, ActivityNet, Sth-sthv2, VIDAL, Charades.

Video

Total Video Average

#OE #MC

Text

#Video

#Caption

Source Length FPS QA QA

LLaVA-Hound GPT-4V ★ 900K 3Khr 0.008 900K 900K 0 ShareGPT4Video GPT-4V ■ 40K 0.2Khr 0.15 40K 0 0 LLaVA-Video-178K GPT-4o ✪ 178K 2Khr 1 178K 960K 196K

LLaVA-Video model builds on the single-image (SI) stage checkpoint from the LLaVA-OneVision model (Li et al., 2024c), which was trained using only image data.

Video Representations Following the classic SlowFast idea in video representations (Feichtenhofer et al., 2019; Xu et al., 2024b; Huang et al., 2024), we develop LLaVA-Video SlowFast to optimize the balance between the number of frames and the count of visual tokens, within the budget of the limited context window in LLM and GPU memory for video representation. Please refer to Appendix 7 for detailed information. Specifically, we represent each video as a sequence with maximum T frames. Each frame is represented in M tokens. we categorize the frames into two groups, based on the a strike rate s, where the every s frames are uniformly selected to form the slow frame group, and the rest of the frames are consdiered as the fast frame group. Note that a special case s = 1 leads to only one group, reducing the SlowFast representation to the original simple representation. For each group, we apply different pooling rate using Pytorch function pooling avg_pool2d(). p × p pooling and 2p × 2p pooling for slow and fast frames, respectively. To summarize, we paramterize the video representation configuration as V = (T,M,s,p). The total number of tokens is #tokens = ⌊T/s⌋ × M/p2 + (T − ⌊T/s⌋) × M/4p2

Evaluation Benchmarks. For full evaluation, we consdier 11 video benchmarks. conducted tests across various video captioning , video open-ended question-answering and video multiple-choice question-answering benchmarks, including ActivityNet-QA (Yu et al., 2019), which features human-annotated action-related QA pairs from the ActivityNet dataset. We also utilized LongVideoBench (Wu et al., 2024b), EgoSchema (Mangalam et al., 2024), and MLVU (Zhou et al., 2024) for long video understanding, PerceptionTest (Pătrăucean

- et al., 2023) for assessing fine-grained perception skills, and VideoMME (Fu et al., 2024) and NExT-QA (Xiao et al., 2021) for diverse video domains and durations. Additional tests included VideoDetailCaption (LMMsLab, 2024), Dream-1K (Wang et al., 2024), Video-ChatGPT (Maaz et al., 2024) for detailed video descriptions, TemporalBench Cai et al. (2024) for fine-grained temporal understanding.

For ablation studies in . 4.2 and Sec. 4.3, we conduct evaluation across 4 datasets. NExT-QA (Xiao et al., 2021) and PerceptionTest (Pătrăucean et al., 2023), which use training data from the LLaVA-Video-178K, are treated as in-domain datasets. Conversely, VideoMME (Fu et al., 2024) and EgoSchema (Mangalam

- et al., 2024) are consider as zero-shot datasets.

#### 4.1 Overall Results

We fine-tune LLaVA-OneVision (SI) on the joint dataset of video and image data. Specifically, we added video data from the LLaVA-Video-178K dataset and four public datasets: ActivityNet-QA (Yu et al., 2019), NExT-QA (Xiao et al., 2021), PerceptionTest (Pătrăucean et al., 2023), and LLaVA-Hound-255K (Zhang et al., 2024d), focusing on videos shorter than three minutes. These datasets were selected to improve our model’s performance, contributing to a total of 1.6 million video-language samples, which include 193,510 video descriptions, 1,241,412 open-ended questions, and 215,625 multiple-choice questions. Remarkably, 92.2% of the video descriptions, 77.4% of the open-ended questions, and 90.9% of the multiple-choice questions were newly annotated. Additionally, we used 1.1 million image-language pairs from the LLaVA-OneVision model (Li et al., 2024c). We consider the same video representation configurations for the training and inference

- Table 2: LLaVA-Video performance on video benchmarks. We report the score out of 5 for VideoDC, VideoChatGPT while other results are reported in accuracy. All results are reported as 0-shot accuracy.

*indicates that the training set has been observed in our data mixture.

Caption Open-Ended Q&A Multi-Choice Q&A

LongVideoBench

TemporalBench

VideoChatGPT

PerceptionTest

VideoMME

ActNet-QA

EgoSchema

Dream-1K

NExT-QA

MVBench

VideoDC

MLVU

Model

test test test test test m-avg test mc val val m-acc wo/w-subs Proprietary models

GPT-4o (OpenAI, 2024) - 39.2 - - - 64.6 - - - 66.7 35.3 71.9/77.2 Gemini-1.5-Pro (Team et al., 2023) - 36.2 57.5 - 72.2 - - - - 64.0 25.6 75.0/81.3

Open-source models VILA-40B (Lin et al., 2024) 3.37 33.2 58.0 3.36 58.0 - - 67.9 54.0 - - 60.1/61.1 PLLaVA-34B (Xu et al., 2024a) - 28.2 60.9 3.48 - - 58.1 - - 53.2 - LongVA-7B (Zhang et al., 2024c) 3.14 - 50.0 3.20 - 56.3 - 68.3 - - - 52.6/54.3 IXC-2.5-7B (Zhang et al., 2024b) - - 52.8 3.46 - 37.3 69.1 71.0 34.4 - 16.7 55.8/58.8 LLaVA-OV-7B (Li et al., 2024c) 3.75 31.7 56.6 3.51 60.1 64.7 56.7 79.4* 57.1 56.5 18.7 58.2/61.5 VideoLLaMA2-72B (Cheng et al., 2024) - 27.1 55.2 3.16 63.9 61.2 62.0 - - - - 61.4/63.1 LLaVA-OV-72B (Li et al., 2024c) 3.60 33.2 62.3 3.62 62.0 68.0 59.4 80.2* 66.9 61.3 26.6 66.2/69.5

LLaVA-Video-7B 3.66 32.5 56.5* 3.52 57.3 70.8 58.6 83.2* 67.9* 58.2 22.9 63.3/69.7 LLaVA-Video-72B 3.73 34.0 63.4* 3.62 65.6 74.4 64.1 85.4* 74.3* 61.9 33.7 70.5/76.9

stages. On 128 NVIDIA H100 GPUs, the video representations for LLaVA-Video-7B and LLaVA-Video-72B are V = (64,679,1,2) and V = (64,679,3,2), respectively.

In Table 2, we compare the performance of different models on various video benchmarks. The 72B model performs as well as the commercial, closed-source model Gemini-1.5-Flash (Team et al., 2023), highlighting the effectiveness of open-source efforts in achieving comparable results. The LLaVA-Video-7B model outperforms the previous top model, LLaVA-OV-7B, in seven out of ten datasets. Analysis of individual datasets shows some noteworthy trends. For instance, on benchmarks like MLVU, LongVideoBench, and VideoMME, which primarily use video data from YouTube, this improvement may be due to the inclusion of extensive YouTube data in LLaVA-Video-178K, as illustrated in Fig. 5. Additionally, the improvement on ActivityNet-QA is small; this could be because many questions in ActivityNet-QA, such as “What’s the color of the ball?” can be answered by viewing a single frame. The visibility of the ball from the beginning to the end of the video means understanding the video sequence is unnecessary, so LLaVA-Video-178K offers little advantage in this context. We find that LLaVA-Video-7B is notably weaker in the specialized task of EgoSchema, an ego-centric dataset. This weakness may be due to a significant reduction in the proportion of ego-centric data in the training dataset of LLaVA-Video. However, this impact is less pronounced in larger models, as demonstrated by the LLaVA-Video-72B model’s superior performance over LLaVA-OV-72B in EgoSchema.

#### 4.2 Dataset Ablation

Note that the training set for LLaVA-Video includes six datasets: LLaVA-Video-178K, LLaVA-Hound (Zhang et al., 2024d), NExT-QA (Xiao et al., 2021), ActivityNet-QA (Yu et al., 2019), PerceptionTest (Pătrăucean

- et al., 2023), and image data from LLaVA-OneVision (Li et al., 2024c). In this section, we conduct ablation studies to assess the impact of each dataset. We separately fine-tune the LLaVA-OneVision (SI) model for each experimental setting, progressively adding datasets to the baseline.

The results are presented in Table 3. Initially, we used a basic model trained solely on the LLaVA-Hound dataset as our baseline. Compared to this baseline, adding the LLaVA-Video-178K dataset significantly

- Table 3: Ablation study on the LLaVA-Video model with various configurations of training data. Three Q&A datasets indicate: NExT-QA, ActivityNet-QA and PerceptionTest.

in-domain out-of-domain

VideoMME

EgoSchema

PercepTest

NExT-QA

Method

mc val test wo

LLaVA-Hound 64.4 51.4 51.0 54.1 +LLaVA-Video-178K 80.1 57.1 56.5 63.2 +Three Q&A datasets 80.1 69.0 55.6 61.9 +LLaVA-OV (images) 83.2 67.9 57.3 63.4

Table 4: Comparison of LLaVA-Video-178K and other video instruction-following datasets.

in-domain out-of-domain

VideoMME

EgoSchema

PercepTest

NExT-QA

#Cap #OE #MC

mc val test wo

LLaVA-Hound 900K 900k 0 39.8 53.1 25.8 55.2 LLaVA-V-178K 178K 900k 0 73.2 55.9 49.8 59.6

ShareGPT4Video 40K 40K 19K 69.6 55.2 58.9 51.0 LLaVA-V-178K 40K 40K 19K 75.8 55.4 55.8 53.5

improved performance, enhancing scores in both in-domain and out-of-domain tasks. Specifically, we observed a 31.9-point increase in NExT-QA scores and a 9.1-point rise in VideoMME scores. Furthermore, including the PerceptionTest dataset enhanced its associated task. Additionally, integrating high-quality image data provided modest benefits on EgoSchema.

#### 4.3 Dataset Comparison

We conduct two ablation studies to analyze our dataset and training strategy. In Table 4, we compared three datasets where the language annotations are from GPT-4V/GPT-4o. For each experiment, we fine-tune the LLaVA-OneVision (SI) model separately on each specific dataset setting.

Two group of experiments are considered to assess the data quality of LLaVA-Video-178K compare to LLaVAHound and ShareGPT4Video. In the first group, to compare LLaVA-Video-178K with LLaVA-Hound, we randomly selected 900K open-ended questions to match the number in LLaVA-Hound. We included all captions and did not sample the multiple-choice questions. In the second group, comparing LLaVA-Video-178K to ShareGPT4Video, we randomly sampled 40K video captions to align with those in ShareGPT4Video. Since ShareGPT4Video lacks open-ended and multiple-choice questions, we supplemented with annotations from NExT-QA, PerceptionTest, and ActivityNet-QA. In the first group of Table 4, we compare LLaVA-Video-178K with LLaVA-Hound. Although LLaVA-Hound has more captions than LLaVA-Video-178K, our results are still better. The quality of LLaVA-Hound is limited due to two main issues: (1) Static video: Its primary video source is WebVid (Bain et al., 2021), which tends to have relatively static content. (2) Sparse sampling: its sampling rate of 10 frames per video leads to annotations that do not fully capture the complete plot of the video. This underscores that the quality of video instruction-following data is more important than its quantity. Additionally, the second experiment group in Table 4 shows that the model trained with LLaVA-Video-178K outperforms that of ShareGPT4Video, highlighting the superiority of our data’s quality.

### 5 Conclusion

This study introduces the LLaVA-Video-178K dataset, a high-quality synthetic dataset for video-language instruction-following. It is favored for its dense frame sampling rate in longer, untrimmed videos, covering diverse tasks such as captioning, open-ended and multi-choice QA. By training on the joint dataset of LLaVA-Video-178K with existing visual instruction tuning data, we developed a new model family, LLaVA-Video, which considers video representation to effectively use GPU resources. This allows us to include more frames in the training process. The experimental results have demonstrated the effectiveness of the proposed synthetic dataset, and LLaVA-Video models have achieved excellent performance on a wide range of video benchmarks.

### 6 Limitations

The videos in LLaVA-Video-178K are sourced from various platforms. This diversity introduces potential biases inherent in these sources. Furthermore, there is a concern regarding the potential skew in the question-answer pairs, possibly influenced by the annotators’ perspectives.

### Acknowledgement

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

### References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning, 2022. URL https://arxiv.org/abs/2204.14198. 17

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer

- vision, pp. 5803–5812, 2017a. 3

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer

- vision, pp. 5803–5812, 2017b. 21

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021. 6, 10, 19

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pp. 961–970, 2015. 2, 3, 4, 21

Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. Temporalbench: Towards fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024. 8

David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190–200, 2011. 3, 19, 21

Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024a. 1, 2, 3, 7, 21

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. arXiv preprint arXiv:2402.19479, 2024b. 3, 21

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms, 2024. URL https://arxiv.org/abs/2406.07476. 9

Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6202–6211, 2019. 8, 17

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 8

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pp. 5842–5850, 2017. 2, 4

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18995–19012, 2022. 2, 4

Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11287–11297, 2021. 3, 4

Mingfei Han, Linjie Yang, Xiaojun Chang, and Heng Wang. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos. arXiv preprint arXiv:2311.17043, 2023. 3

De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. arXiv preprint arXiv:2403.19046, 2024. 8, 17

Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and Gedas Bertasius. Video recap: Recursive captioning of hour-long videos. arXiv preprint arXiv:2402.13250, 2024. 3

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017. 2, 4

Muhammad Uzair khattak, Muhammad Ferjad Naeem, Jameel Hassan, Naseer Muzzamal, Federcio Tombari, Fahad Shahbaz Khan, and Salman Khan. How good is my video lmm? complex video reasoning and robustness evaluation suite for video-lmms. arXiv:2405.03690, 2024. 5

Sangho Lee, Jiwan Chung, Youngjae Yu, Gunhee Kim, Thomas Breuel, Gal Chechik, and Yale Song. Acav100m: Automatic curation of large-scale datasets for audio-visual video representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10274–10284, 2021. 3, 21

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering. arXiv preprint arXiv:1809.01696, 2018. 21

Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7331–7341, 2021. 19

Jie Lei, Tamara L Berg, and Mohit Bansal. Revealing single frame bias for video-and-language learning. arXiv preprint arXiv:2206.03428, 2022. 2, 19

Bo Li, Hao Zhang, Kaichen Zhang, Dong Guo, Yuanhan Zhang, Renrui Zhang, Feng Li, Ziwei Liu, and Chunyuan Li. Llava-next: What else influences visual instruction tuning beyond data?, May 2024a. URL https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/. 1

Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, May 2024b. URL https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/. 1

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024c. 1, 7, 8, 9

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 2024d. 1

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. URL https://arxiv.org/abs/2301.12597. 3, 17

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2024e. URL https://arxiv.org/abs/2305.06355. 1, 3

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26689–26699, 2024. 1, 9

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024a. 1, 17

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024b. 5

LMMs-Lab. Video detail caption, 2024. URL https://huggingface.co/datasets/lmms-lab/ VideoDetailCaption. 8

Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval. arXiv preprint arXiv:2104.08860, 2021. 19

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 8

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36,

2024. 8

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. In ICCV, 2019. 3, 21

OpenAI. Gpt-4v. https://openai.com/index/gpt-4v-system-card/, 2023. 3 OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024. 2, 4, 9

Viorica Pătrăucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=HYEGXFnPoq. 8, 9

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), pp. 8748–8763. PMLR, 2021. 7, 17

Nils Reimers and Iryna Gurevych. Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2020. URL https://arxiv.org/abs/2004.09813. 5

Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3202–3212, 2015. 3, 21

Xindi Shang, Donglin Di, Junbin Xiao, Yu Cao, Xun Yang, and Tat-Seng Chua. Annotating objects and relations in user-generated videos. In Proceedings of the 2019 on International Conference on Multimedia Retrieval, pp. 279–287. ACM, 2019. 2, 4

Gunnar A Sigurdsson, Gül Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pp. 510–526. Springer, 2016. 2, 4

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 9

Jiawei Wang, Liping Yuan, and Yuchen Zhang. Tarsier: Recipes for training and evaluating large video description models, 2024. URL https://arxiv.org/abs/2407.00634. 8

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In The Twelfth International Conference on Learning Representations, 2023. 2, 4

Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. arXiv preprint arXiv:2405.09711, 2024a. 3, 4

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024b. URL https://arxiv.org/abs/2407.15754. 8

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9777–9786, 2021. 3, 4, 8, 9, 21

Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In ACM Multimedia, 2017. 3, 21

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296,

2016. 3, 21

Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024a. 9, 17, 18

Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024b. 8

Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models, 2024c. URL https://arxiv.org/abs/2407.15841. 17

Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3, 4, 21

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 7, 17

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, pp. 9127–9134, 2019. 3, 5, 8, 9, 21

Amir Zadeh, Michael Chan, Paul Pu Liang, Edmund Tong, and Louis-Philippe Morency. Social-iq: A question answering benchmark for artificial social intelligence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8807–8817, 2019. 3, 21

Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. Advances in neural information processing systems, 34:23634–23651, 2021. 3, 21

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11975–11986,

2023. 7, 17

Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. URL https://arxiv.org/abs/2306.02858. 1, 3

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024a. 7

Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024b. 9

Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024c. 9

Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, and Yiming Yang. Direct preference optimization of video large multimodal models from language model reward, 2024d. 1, 3, 6, 8, 9, 21

Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024e. URL https: //llava-vl.github.io/blog/2024-04-30-llava-next-video/. 17, 18

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 8

Luowei Zhou and Jason J. Corso. Youcookii dataset. 2017. URL https://api.semanticscholar.org/ CorpusID:19774151. 2, 3, 4, 21

Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, Wang HongFa, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Cai Wan Zhang, Zhifeng Li, Wei Liu, and Li Yuan. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment, 2023a. 2, 4

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023b. 3

### A Video Representations

#### A.1 Efficient Video Representations in LMMs

Current designs of large multimodal models (LMM) typically connect a vision encoder (Radford et al., 2021; Zhai et al., 2023) to a large language model (Yang et al., 2024) through a lightweight projector (Liu et al., 2024a) or a resampler (Li et al., 2023; Alayrac et al., 2022). These components transform a set of visual representations into “visual tokens” aligned with text embeddings. In contrast to image-based LMMs, which generate only a small number of visual tokens easily managed by a standard GPU, video LMMs face challenges due to a large number of visual tokens derived from multiple video frames. The LLaVA-NeXT-Video (Zhang et al., 2024e) and PLLaVA (Xu et al., 2024a) models address this by simly considering average pooling to reduce the number of tokens representing each frame.

Following the idea of SlowFast in the traditional video understanding (Feichtenhofer et al., 2019), adaptive reductions in visual tokens are demonstrated by recent video LMMs, LITA (Huang et al., 2024) and SlowFastLLaVA (Xu et al., 2024c). Initially, these methods represent all sampled frames with a minimal number of visual tokens (fast frame)— typically just one—by using a large pooling stride. They then switch to a smaller pooling stride for certain frames to retain more visual tokens (slow frame). Finally, they combine the visual tokens of fast frames with those of slow frames. However, this approach can lead to some frames being represented twice. In contrast, our method uses a larger pooling stride for sampled frames to maintain fewer visual tokens (fast frame) or a smaller stride for others to keep more (slow frame). We then arrange slow and fast frames in an interleaving pattern.

𝑠 = 2

- 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

- 1 1 1 1 2 3 4 4 4 4 5 6 7 7 7 7

𝑝 = 2

Pooling

M tokens

T Frames

𝐹 are assigned fewer visual tokens 𝐹 are assigned more visual tokens

Figure 8: Video representations. A different number of tokens are utilized to represent frames.

#### A.2 LLaVA-Video SlowFast

We represent each video as a sequence with maximum T frames. Each frame is represented in M tokens. FPS-based video representation can be considered in the future. Specifically, each frame is encoded via an image encoder and a two-layer MLP for projection. These visual tokens are concatenated with word tokens and processed by a large language model (LLM). Managing tokens for every frame can be computationally demanding. For instance, employing the SigLIP (Zhai et al., 2023) encoder for a video with T = 100 results in 67,600 tokens, assuming M = 729 tokens per frame, which often exceeds GPU memory limits. This issue is exacerbated when using large-parameter LLMs; with the Qwen2-72B model, we could only process 8 frames before maxing out the memory on 128 NVIDIA H100 GPUs. Such a limited number of frames can introduce inconsistencies in language annotations, reducing model efficacy. One strategy to incorporate more frames is by applying p × p spatial average pooling to reduce M to M/p2, thus lowering the token count per frame as

suggested by recent studies (Xu et al., 2024a; Zhang et al., 2024e). However, the number of visual tokens is crucial for preserving the informational content of each frame, which is vital for video comprehension.

In our LLaVA-Video SlowFast, we categorize the frames into two groups, based on the a strike rate s, where the every s frames are uniformly selected to form the slow frame group, and the rest of the frames are consdiered as the fast frame group. Note that a special case s = 1 leads to only one group, reducing the SlowFast representation to the original simple representation. For each group, we apply different pooling rate using Pytorch function pooling avg_pool2d(). p × p pooling and 2p × 2p pooling for slow and fast frames, respectively. To summarize, we paramterize the video representation configuration as V = (T,M,s,p). The total number of tokens is #tokens = ⌊T/s⌋ × M/p2 + (T − ⌊T/s⌋) × M/4p2

### B Data

- B.1 Video Detail Description

As discussed in Section 3.2, we show that generating level-1 description should consider historical context. Figure 9 illustrates the impact of excluding historical context on the quality of video descriptions. Specifically, including historical context helps accurately identify characters across different times as the same individual.

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

0-10s 10-20s

The video begins with a person standing on a balcony, holding a large red water balloon in one hand and a knife in the other,….The camera then shifts to show a woman below the person on the balcony, near a pool, wearing a yellow top, pink leggings, … The person on the balcony moves closer to the edge, positioning the water balloon over the woman below. …

The video shows with a person standing on a balcony, holding a large red water balloon in one hand and a knife in the other,….Then a woman in a yellow shirt and pink pants walking on a sidewalk near a pool. Finally, one person stand on the high platform cuts the balloon, causing water to splash out and soak the woman.

Generating Video Captions with Historical Context

Generating Video Captions without Historical Context

Figure 9: Generating video captions with or without historical context.

- B.2 Video Question Answering

In Table 5, we list the names and descriptions of different question types and their corresponding proportions in the LLaVA-Video-178K dataset. The prompt used to generate video question-answer pairs from GPT-4O is shown in Table. 6. In Fig. 4, we show an example of a video along with its detailed description, an open-ended question, and a multiple-choice question.

- B.3 Dataset Comparison

We provide a more comprehensive comparison of LLaVA-Video-178K with other video-language datasets for the video caption task and video question answer task. Specifically, we organize the table into four groups, each characterized by its method of text annotation. As shown in Table 7, unlike other datasets, LLaVA-Video-178K uniquely includes all three types of annotations: captions, open-ended questions, and multiple-choice questions.

### C Beyond Singularity: Extensive Sampling Matters

We perform experiments to explore how video representations affect the model’s performance. All experiments were carried out in a video-only setting, using video data with durations from 0 to 30 seconds as our training data. We focused on evaluating how the number of frames and the number of visual tokens per frame impact

- Table 5: Question types for video question answering in data creation. For each type, we provide its name, description, and the proportion it represents in the LLaVA-Video-178K.

Question type Description Proportion Temporal

Designed to assess reasoning about temporal relationships between actions/events. Questions involve previous, present, or next actions.

7.2%

Tests ability to perceive spatial relationships between observed instances in a video scene.

Spatial

7.2%

Focuses on explaining actions/events, determining intentions of actions or causes for subsequent events.

Causal

7.2%

Assesses ability to describe the major scene of the video, like where it takes place and the overall environment.

Description-Scene

7.2%

Involves describing actions or attributes of people, such as their activities and appearances.

Description-Human

6.7%

Assesses ability to describe attributes of objects, like their appearance and function.

Description-Object

7.0%

Tests ability to count instances of objects, people, actions, and to distinguish between old and new elements in a scene.

Count

7.1%

Binary Involves yes or no questions related to the video content. 7.2% Fine Grained Action Understanding

Creates questions challenging comprehension of subtle actions. 6.5%

Plot Understanding Challenges ability to interpret the plot in the video. 7.1% Non-Existent Actions with Existent Scene Depictions

Assesses reasoning with introduced non-exist ent activities without changing physical details.

6.6%

Time Order Understanding Challenges recognition of temporal sequence of activities in videos. 6.9% Object Direction Emphasizes perception of object movement direction. 3.8% Camera Direction Focuses on the direction of camera movement. 4.1%

Delves into discerning variations in speed, including absolute and relative speeds.

Speed

3.6%

Centers on how attributes of objects or the entire video change over time, like size, shape, color, and more.

Attribute Change

4.5%

model performance. Regarding the frame count, it is noteworthy that observing the effects of a high number of frames—such as over 100—does not necessarily require long videos. Our results indicate that the dynamic properties of the data render even 100 frames insufficient to fully capture the condent of a 30-second video, which typically runs at 15 FPS.

In Table 8, the first group shows an increase in the number of frames from 32 to 110. We set 110 frames as the upper limit to avoid overloading the GPU. With more frames, we see significant improvements in all datasets. While it’s generally expected that using more frames boosts performance, previous studies (Luo et al., 2021; Lei et al., 2021; 2022) have noted that performance tends to plateau when training with more than 16 frames. We propose that the saturation observed in earlier studies arises due to the selection of training datasets such as MSVD (Chen & Dolan, 2011) and WebVid (Bain et al., 2021), where the video content is highly static, allowing a small number of frames to represent the entire video effectively. In contrast, the dynamic nature of the videos and the detailed nature of the annotations in LLaVA-Video-178K allow for continuous benefits from extensive sampling

The second group in Table 8 demonstrates the effects of varying the number of inference frames while keeping the number of training frames constant. A modest increase in the inference frames slightly enhances performance; however, excessively increasing the number of inference frames can degrade it.

tasks = “ # Temporal: this task is designed to assess the capability of reasoning ...<omitted> ## caption-1: The video features a child sitting in a baby chair at a dining table, creating...<omitted> ## question-1: What was the child doing as he sat on the baby chair? ## answer-1: The child was reading a book.

... ## caption-3: ...<omitted> ## question-3: ...<omitted> ## answer-3: ...<omitted> # Spatial: this task involves creating questions that test a person’s ability...<omitted>

...<omitted> ” system_message = “ ### Task: Given a detailed description that summarizes the content of a video, generate question-answer pairs based on the description to help humans better understand the video. The question-answer pairs should be faithful to the content of the video description and developed from different dimensions to promote comprehensive understanding of the video. Here are some question dimensions and their explanations and exampled question-answer pairs for reference:

{task_definitions} #### Guidelines For Question-Answer Pairs Generation:

- - Read the video description provided carefully, paying attention to the content, such as the scene where the video takes place, the main characters and their behaviors, and the development of the events.
- - Generate appropriate question-answer pairs based on the description. The question-answer pairs should cover as many question dimensions and not deviate from the content of the video description.
- - Generate 1 question-answer pair for each dimension. ### Output Format:

- 1. Your output should be formed in a JSON file.
- 2. Only provide the Python dictionary string. Your response should look like: ["Dimension": <dimension-1>, "Question": <question-1>, "Answer": <answer-1>, "Dimension": <dimension-2>, "Question": <question-2>, "Answer": <answer-2>...] ” user_message = “ Please generate question-answer pairs for the following video description: Description: {caption} ”

for cur_video in videos: sys_msg = system_messages.format(task_definitions=tasks) usr_msg = user_messages.format(caption=cur_video) response = GPT4O(sys_msg,usr_msg)

- Table 6: We explain the process of creating prompts for GPT-4O to gather question-answer pairs from each video description. tasks includes the definition of all question types along with examples of question-answer pairs. We instruct GPT-4O to generate questions that cover as many question types as possible.

In Table 8’s third group, we illustrates the trade-off between the number of frames and the number of tokens per frame. Configurations with fewer tokens per frame but more frames yield superior results, even with a lower total count of visual tokens (18,590 versus 21,632). This finding emphasizes that increasing the number of frames, rather than the tokens per frame or the total number of tokens, enhances performance. However, a balance is necessary; as the number of frames increases to 440 and the tokens per frame decreases to 64, performance drops. This observation led us to use LLaVA-Video SlowFast for video representation.

- Table 7: Comparison of LLaVA-Video-178K and other video-language datasets. Average FPS represents the average number of frames per second that are used to prompt GPT-4o/GPT-4V for annotation.

Text #Video

Total Video Average

#Caption

#OE #MC

Length FPS QA QA

HowTo100M (Miech et al., 2019) ASR 136M 134.5Khr - 136M 0 0 ACAV (Lee et al., 2021) ASR 100M 277.7Khr - 100M 0 0 YT-Temporal-180M (Zellers et al., 2021) ASR 180M - - 180M 0 0 HD-VILA-100M (Xue et al., 2022) ASR 103M 371.5Khr - 103M 0 0

MSVD (Chen & Dolan, 2011) Manual 1970 5.3h - 1K 0 0 LSMDC (Rohrbach et al., 2015) Manual 118K 158h - 118K 0 0 MSR-VTT (Xu et al., 2016) Manual 10K 40h - 10K 0 0 DiDeMo (Anne Hendricks et al., 2017b) Manual 27K 87h - 27K 0 0 ActivityNet (Caba Heilbron et al., 2015) Manual 100K 849h - 100K 0 0 YouCook2 (Zhou & Corso, 2017) Manual 14K 176h - 14K 0 0 TVQA (Lei et al., 2018) Manual 21K 3.39Khr - 0 0 152K ActivityNet-QA (Yu et al., 2019) Manual 5.8K 290h - 0 58K 0 Social-IQ (Zadeh et al., 2019) Manual 1.2K 20h - 0 0 7.5k NExT-QA (Xiao et al., 2021) Manual 5.4K 66h - 0 52K 47K

MSVD-QA (Xu et al., 2017) Open-source Model 1.9K 5.3h - 41K 50K 0 MSRVTT-QA (Xu et al., 2017) Open-source Model 10K 40h - 0 243K 0 Panda-70M (Chen et al., 2024b) Open-source Model 70.8M 166.8Khr - 70.8M 0 0

LLaVA-Hound (Zhang et al., 2024d) GPT-4V 900K 3Khr 0.008 900K 900K 0 ShareGPT4Video (Chen et al., 2024a) GPT-4V 40K 0.2Khr 0.15 40K 0 0 LLaVA-Video-178K GPT-4o 178K 2Khr 1 178K 960K 196K

- Table 8: Visual Representation Configurations and Performance Correlation. Ttrain and Ttest are the number of frames in the training and inference stage, respectively. M/p2: number of visual tokens per frame.

in-domain out-of-domain Ttrain Ttest M/p2

###### NExT-QA PerceptionTest EgoSchema VideoMME mc val test wo

Training with more frames 32 32 169 80.4 68.2 56.3 59.1 64 64 169 81.4 (+1.0) 68.3 (+0.1) 58.4 (+2.1) 59.6 (+0.5)

- 110 110 169 82.0 (+1.6) 68.3 (+0.1) 59.1 (+2.8) 60.4 (+1.3)

Inference with more frames 32 32 169 80.4 68.2 56.3 59.1 32 64 169 80.7 (+0.3) 68.9 (+0.7) 56.3 (+0.0) 59.9 (+0.8) 32 110 169 80.5 (+0.1) 67.2 (-1.0) 55.2 (-1.1) 58.8 (-0.3)

Using more frames with fewer visual tokens per frame 32 32 729 79.4 69.5 58.3 59.1

- 110 110 169 82.0 (+2.6) 68.3 (-1.2) 59.1 (+0.8) 60.4 (+1.3) 440 440 64 81.6 (+2.2) 67.2 (-2.3) 59.4 (+1.1) 60.2 (+1.1)

### D Capabilities

Beyong achieve good benchmark performance, Our observations of LLaVA-Video reveal various capabilities in video understanding. Specifically, it show a great abilities in the understanding video using real-world knowledge,including, but not limited to:

- Table 9: Comparison of different video representations. The video representation V is consistent in training and inference for all methods, except that SlowFast-LLaVA considers simple representation V in training and its specified V in inference.

in-domain out-of-domain Method V = (T, M, s, p)

#Visual NExT-QA PerceptionTest EgoSchema VideoMME Tokens mc val test wo

Simple representation (32, 729, 1, 2) 5,408 80.4 68.2 56.3 59.1 LLaVA-Video SlowFast (64, 729, 3, 2) 5,396 81.1 67.7 57.1 59.8 LITA (42, 729, 2, 2) 5,313 80.8 68.3 54.3 59.1 SlowFast-LLaVA (42, 729, 2, 2) 5,313 79.4 68.2 56.2 58.9

- • Optical Illusion: As shown in Table 11, LLaVA-Video recognizes that the green dragon in the video is not a real 3D object. It appears three-dimensional due to an optical illusion that affects human perception.
- • Special Domain: As indicated in Table 11, LLaVA-Video understands the content within special domains in the video, such as sketches and fights in video games.
- • Unusual Action: As detailed in Table 12, LLaVA-Video identifies atypical actions in the video, such as "physical therapy" for pets, beyond ordinary activities.
- • Physical Laws: As shown in Table 13, LLaVA-Video comprehends basic physical laws demonstrated in the video, like zero gravity in space stations, which allows objects to float without falling. Real-world Knowledge: Illusion

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

User Please describe this video.

LLaVA-Video The video features a person interacting with a large, cartoonish green dragon cutout in a well-lit, spacious room adorned with framed pictures on the walls. The person, dressed in a blue tank top and light blue jeans, is seen hugging and adjusting the dragon cutout, which has a wide, toothy grin and large, expressive eyes. The text overlay reads ’Reply to seloutria’s comment: Once you see it you can’t un see it,’ suggesting that the cutout is designed to be visually striking and memorable. The person moves around the cutout, showcasing different angles and perspectives.

User Is this green dragon a real 3D structure? LLaVA-Video No, the green dragon is a 2D cutout that creates an optical illusion.

- Table 10: LLaVA-Video learns to understand the optical illusion in the video.

##### Real-world Knowledge: Special Domain

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

User Please describe this video.

LLaVA-Video The video begins with a blank, lined sheet of paper. A hand holding a black pen starts drawing on the paper, creating the outline of a flower with five petals. The artist adds details to the petals, including lines and shading to give them a more realistic appearance. The stem and leaves are also drawn, completing the flower illustration. The artist adds final touches to the flower drawing, enhancing the details and shading to give it a more realistic appearance. The artist then switches to a yellow pen and begins coloring the center of the flower, adding depth and contrast to the drawing. The video concludes with the artist adding final touches to the coloring, enhancing the overall appearance of the flower illustration.

User What could this flower be? LLaVA-Video This flower could be a magnolia or a similar species.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

User Please describe this video.

LLaVA-Video The video showcases an intense battle scene set in a snowy, mountainous landscape. A character dressed in dark, heavy armor with glowing orange accents wields a sword and faces off against a large, menacing creature with glowing red eyes and antlers. The text ’EVERY BATTLE SCREAMS GOTY POTENTIAL!’ is prominently displayed at the top of each frame, emphasizing the high stakes and epic nature of the confrontation. The battle is fierce, with the character and the creature exchanging powerful attacks. The creature emits blue energy and lightning, while the character retaliates with fiery and icy attacks. The snowy ground is littered with debris and frost, adding to the harsh and brutal environment. The background features a large body of water and a distant lighthouse, enhancing the sense of isolation and danger.

- Table 11: LLaVA-Video learns to understand the the video in special domain.

##### Real-world Knowledge: Unusual Action

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

User Please describe this video.

LLaVA-Video The video begins with a person holding a tabby cat, with the text ’DOGS WITH BACK PAIN’ and a crying emoji displayed at the bottom of the screen. The person, who has tattoos on their arms and is wearing a light-colored shirt, is in a room with large windows and a white wall. The scene transitions to the person petting a large, brown and white dog, who is sitting on the floor. The person then stands up and gestures with their hands, while the dog looks up at them. The next scene shows the person kneeling and petting a large, fluffy dog with a black and white coat, who is sitting on a table. The person continues to pet the dog, who appears calm and relaxed.

User What did this man do to his pets? LLaVA-Video This man provided physical therapy to his pets.

- Table 12: LLaVA-Video learns to understand the unusual action in the video.

Real-world Knowledge: Physical Laws

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

User Please describe this video.

LLaVA-Video The video features a person in a space station environment, wearing a dark gray shirt and a wristwatch. They are holding two containers filled with a thick, yellowish substance. The person tilts the containers to demonstrate the fluid’s behavior, showing how it stretches and moves between the containers. The background includes various equipment and panels typical of a space station, with labels such as ’GALLEY’ and ’FUSION’. The person continues to manipulate the containers, emphasizing the fluid’s properties. The video concludes with the person holding one container and twisting the lid to close it, further demonstrating the fluid’s properties.

User Why doesn’t this yellowish substance fall? LLaVA-Video This yellowish substance doesn’t fall because it is in zero gravity.

- Table 13: LLaVA-Video learns to understand the physical laws in the video.

