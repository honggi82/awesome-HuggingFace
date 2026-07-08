# arXiv:2307.06942v2[cs.CV]4Jan2024

## InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation

Yi Wang∗1, Yinan He∗1, Yizhuo Li∗4,1, Kunchang Li6,1, Jiashuo Yu1, Xin Ma3,1, Xinhao Li2,1 Guo Chen3,1, Xinyuan Chen1, Yaohui Wang1, Conghui He1, Ping Luo4,1, Ziwei Liu5,1 Yali Wang†6,1, Limin Wang†2,1, Yu Qiao†1 1OpenGVLab, Shanghai AI Laboratory 2Nanjing University 3Monash University 4The University of Hong Kong 5Nanyang Technological University 6Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid

### Abstract

This paper introduces InternVid, a large-scale video-centric multimodal dataset that enables learning powerful and transferable video-text representations for multimodal understanding and generation. The InternVid dataset contains over 7 million videos lasting nearly 760K hours, yielding 234M video clips accompanied by detailed descriptions of total 4.1B words. Our core contribution is to develop a scalable approach to autonomously build a high-quality video-text dataset with large language models (LLM), thereby showcasing its efficacy in learning videolanguage representation at scale. Specifically, we utilize a multi-scale approach to generate video-related descriptions. Furthermore, we introduce ViCLIP, a video-text representation learning model based on ViT-L. Learned on InternVid via contrastive learning, this model demonstrates leading zero-shot action recognition and competitive video retrieval performance. Beyond basic video understanding tasks like recognition and retrieval, our dataset and model have broad applications. They are particularly beneficial for generating interleaved video-text data for learning a video-centric dialogue system, advancing video-to-text and text-to-video generation research. These proposed resources provide a tool for researchers and practitioners interested in multimodal video understanding and generation.

### 1 Introduction

Learning transferable video-text representations is both challenging and essential for video understanding in various real-world applications such as autonomous driving, intelligent surveillance, human-computer interaction, and visual searching. While multimodal contrastive learning using web-scale data has been successful in image-text representation, it remains underexplored in the video-language domain.

A key reason for this limited exploration is the lack of a high quality video-language dataset for pretraining at scale. Current research relies on datasets like HowTo100M [2], HD-VILA [3], and YTTemporal [4, 5], whose texts are generated using automatic speech recognition (ASR). Despite their large scale, these datasets often have low semantic correlations between the videos and corresponding textual descriptions [2–5]. Empirical studies demonstrate that improving this correlation (e.g. aligning videos with subtitles to improve their matching) significantly benefits downstream tasks such as video retrieval and video question answering [6]. Recent works have utilized WebVid10M [6], a dataset

* Equal contribution. † Corresponding authors.

|[Figure 1]<br><br>|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]<br><br>|
|---|

|[Figure 5]<br><br>|
|---|

|[Figure 6]<br><br>|
|---|

[Figure 7]

[Figure 8]

a woman in a red wagon with a child in it in a

women dressed in traditional costumes standing in a room.

store as she walks through the store.

|drive a car out.<br><br>[Figure 9]|
|---|

|[музыка] [аплодисменты]<br><br>([Music] [Applause])<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>|
|---|

|[Figure 12]<br><br>|
|---|

|[Figure 13]<br><br>|
|---|

|[Figure 14]<br><br>|
|---|

|[Figure 15]<br><br>|
|---|

|[Figure 16]<br><br>|
|---|

[Figure 17]

[Figure 18]

a man and a woman brushing their teeth in the bathroom.

older women sitting at a table eating a bowl of food with chopsticks.

|分かってるわよ待ってティッシュはどこシャワーを買って閃いた<br><br>まずはこんな風に居るの (I understand, just wait. Where are the tissues? I had an idea to buy a shower, but for now I’ll stay like this.)<br><br>[Figure 19]|
|---|

|老祖婆就是这点好，不挑食。啥子都吃得。老祖婆坐车哪里都能去。<br><br>(The old granny is good in this aspect, she is not picky and can eat anything. She can go anywhere by car.)<br><br>[Figure 20]|
|---|

- Figure 1: Examples (we give three frames of each video clip), the corresponding generated captions, and ASR transcripts in InternVid. In the captions, we highlight nouns in blue and verbs in green. Non-English transcripts are translated to English using LLM [1].

with higher-quality alt-texts, to address the low video-text correlation issue. However, its limited scale and dynamics hinder its use in current data and model scaling studies. Specifically, only 10M video-text pairs are provided, and the depicted scenes contain relatively few actions or activities.

We propose a large-scale video-centric dataset InternVid to address the challenge of scaling up videolanguage modeling while maintaining high video-text correspondence. Visual examples are given in Figure 1. Note the ASR transcripts barely depict visual elements in videos while the generated captions do. The dataset contains highly-correlated video-text pairs and includes over 7 million videos, totaling 760,000 hours and resulting in 234 million video clips, with various subsets for different needs. These videos cover 16 scenarios and around 6,000 motion descriptions. To improve video-text matching, we generate captions using a multiscale approach. In the coarse scale, we caption the middle frame of each video and use the description as the video caption. In the fine scale, we produce frame-by-frame captions and summarize them with a language model.

Leveraging InternVid, we scale a video-language transformer (ViT-L) in contrastive learning from a data perspective, and its experiments prove InternVid enables learning scalable video-text models. We introduce video masking to the model to accelerate the whole learning without compromising its effectiveness. The video and text encoders are initialized from the CLIP pretrained model with the same scale. With InternVid, we learn a video-text model for several epochs, achieving impressive zero-shot performance. Compared with previous Video CLIP variants, our proposed ViCLIP shows notable performance improvement, especially in zero-shot settings.

In addition to large-scale video-language contrastive pretraining, we discover its effectiveness in producing interleaved video-text data for learning a video-centric dialogue system like Flamingo [7, 8], and advancing video generation. Since the text-annotated clips are extracted from videos, we naturally collect clips and their corresponding text based on the sampling locations. This results in approximately 7 million interleaved data pieces, suitable for instruction tuning as multi-turn video-centric dialogue. For video generation, we filter the core set and obtain 18 million video clips. Alongside WebVid-10M, InternVid can significantly improve a stable-diffusion based video generation model to new heights.

In summary, our contributions are threefold.

- • We introduce a new web-scale video-language dataset InternVid. This dataset, aimed at advancing video-related multimodal understanding and generation at scale, is created using a multi-scale video captioning approach powered by LLM, ensuring high-quality video-text data with minimal human intervention. InternVid has 7 million videos, corresponding to 234 million clips each with the generated captions. Spanning 16 scenes and about 6 thousand actions, the dataset includes computational features (video-text correlation and visual aesthetics) across the entirely of the dataset and gives way to diverse subsets to cater to varying training needs.
- • We learn a new video-language model, ViCLIP, which is trained on InternVid using ViT-L. It incorporates both constrastive learning and mask modeling techniques, allowing for efficient

- learning of transferrable video-language representation. This model achieves state-of-the-art zeroshot action recognition in Kinetics, scoring 75.7, 73.5, and 66.4 on K400, K600, and K700 with the average top1 and top5 accuracies, respectively. It also gets competitive performance on video retrieval, setting a new baseline for video-text understanding.
- • InternVid fosters the development of multimodal dialogue systems and text-to-video generation. The proposed ViCLIP learned on InternVid could serve as a vision backbone of video-centric dialogue systems[9–11], conducting tasks as action recognition, temporal understanding, reasoning, and creativity within an open-ended environment. Furthermore, we provide a subset, InternVidAesthetics, created using specific video-text relation and visual aesthetic filtering. This subset aids in generating high-resolution watermark-free videos. Utilizing InternVid-Aesthetics, both visual and quantitative outcomes of a simple text-to-video baseline can be noticeably enhanced (FVD: 705.3 -> 616.5).

### 2 Related Work

Multimodal Datasets. Vision-text data pairs are necessary to enable crossmodal learning. To learn vison-language representation effectively, these datasets should be large at scale and high at visiontext correlations. To this end, researches usually leverage existing web images with alt-text [12–18] and videos with ASR transcriptions [2, 4, 5, 3, 6, 19, 20] for scalable learning. With LAION-5B’s introduction [19], researchers now have access to hundreds or millions or billions of image-text pairs, opening up new avenues for research on large-scale image-language pretraining.

For video-centric multimodal datasets, HowTo100M [2] collected instructional YouTube videos and exploited the corresponding ASR subtitles for learning joint representations. Zellers et al. [4, 5] and Xue et al. [3] proposed YT-Temporal and HD-VILA for Audio-Visual-Language joint learning and high-resolution video crossmodal learning, respectively. On the other hand, Bain et al. [6] found video-text alignment matters more than their quantities, so they produced WebVid [6] where 10M videos with the corresponding alt-texts. This is frequently employed in recent video-language pretraining approaches [21]. Similarly, based on CC3M, Nagrani et al. proposed VideoCC3M [22] by transferring captions from image-text datasets to video ones. In this work, we target to present a large-scale video-language dataset with high-quality descriptions.

Video Understanding. Pretraining large-scale video-text models and fine-tuning them for downstream tasks has become the norm in the video-language field [23–25, 21, 26, 25, 15, 27–33, 4, 5, 34– 36]. Early techniques [30, 31] used pretrained visual and language encoders to obtain offline video and text features, but recent methods [24, 23, 15, 27, 37, 38] highlight the advantages of end-toend training. Common practices include two or three pretraining tasks, such as masked language modeling [39], video-text matching [40], video-text contrastive learning [25, 32], masked video modeling [37, 38, 32], and video-text masked modeling [41].

In the multimodal video context, VIOLET [41] combined masked language and video modeling, while All-in-one [40] proposes a unified pretraining approach with a shared backbone, and LAVENDER [39] unified tasks through masked language modeling. Despite their success in multimodal benchmarks, these methods’ reliance on limited video-text data hampers performance in video-only tasks like action recognition. Conversely, InternVideo [32] and UMT [21] combined masked modeling with crossmodal contrastive learning, leading to competitve performance in both video-only and videolanguage tasks. MERLOT Reserve [5] exploited 20 million video-text-audio pairs for training joint video representations using contrastive matching, setting new standards in video recognition and visual commonsense reasoning. VALOR [42] also employed different modality encoders for video, audio, and text processing, and introduces video-to-text and audio-to-text pretasks to improve visionaudio-language learning. To address modality entanglement in crossmodal learning, mPLUG-2 [43] introduced a shared module across image, video, and text to encourage modality collaboration while reserving modality-specific modules for their differences. Similar to [32, 26], VLAB [44] adapted a CLIP-pretrained ViT to model spatiotemporal variations and blends it with CLIP ViT with cross attention for handling both images and videos.

### 3 InternVid: A Video-Centric Multimodal Dataset

A high-quality video-text dataset at scale is a premise to conduct large-scale video-language learning and associated tasks. We identify three crucial factors in constructing this dataset: substantial

Dataset Caption Domain #Videos #Clips LenClip LenCap Dur(h) Res

MSR-VTT [45] Manual open 7.2K 10K 15.0 9.3 40 240P DideMo [46] Manual Flickr 10.5K 27K 6.9 8.0 87 LSMDC [47] Manual movie 200 118K 4.8 7.0 158 1080P

YouCook2 [48] Manual cooking 2K 14K 19.6 8.8 176 How2 [49] Manual instruct 13.2K 80K 90.0 20.0 2K ANet Caption [50] Manual action 20K 100K 36.0 13.5 849 VideoCC3M [22] Transfer open 6.3M 10.3M 10 - 17.5K WebVid10M [6] Alt-text open 10.7M 10.7M 18.0 12.0 52K 360P WTS70M [51] Metadata action 70M 70M 10 - 194K -

HowTo100M [2] ASR instruct 1.2M 136M 3.6 4.0 134.5K 240P HD-VILA-100M [3] ASR open 3.3M 103M 13.4 32.5 371.5K 720P

YT-Temporal-180M [4] ASR open 6M 180M - - - InternVid (ours) Generated open 7.1M 234M 11.7 17.6 760.3K 720P*

- Table 1: Statistics of InternVid and its comparison with existing video-language datasets. *In InternVid, most videos (around 85%) are in 720P and the remaining are in from 360P to 512P.

temporal dynamics, rich and diverse semantics, and strong video-text correlations. To ensure high temporal dynamics, we gather videos retrieved using action/activity-based query words. For rich and varied semantics, we not only crawl trending videos across various categories but also deliberately increase the proportion of data consciously collected from various countries and languages. To strengthen video-text correlations, we employ image captioning and language models to generate video descriptions from frame-specific annotations. Next, we elaborate the dataset construction process and discuss its statistics and characteristics.

#### 3.1 Data Curation

We collect videos from YouTube considering the diversity and richness of its data, and its support for academic usage. Totally we obtain 7 million public YouTube videos with an average duration of 6.4 minutes, covering 16 topics. We ensure the uniqueness of our dataset by creating a database of YouTube video IDs and excluding any videos already present in publicly available datasets (released prior to April 2023). The data curation strategies are two-fold. On one hand, We select popular channels and the corresponding hot or high-rated videos from the categories e.g. news, gaming, etc., resulting in 2 million videos. On the other hand, we create a list of verbs related to actions/activities. With it, we also obtain 5.1 million videos by choosing the top retrieved ones.

Defining Actions in Kinetics & Motives for Queries. We define around 6.1K action phrases from American Time Use Survey (ATUS), public video datasets, and text corpus. Then they are refined both manually and automatically. We employ actions from ATUS from 2017 to 2022 [52], merging them and removing the duplicates. For the referenced public video data, we leverage Kinetics [53], SomethingSomething series [54, 55], UCF101 [56], and so on. This provides us with 1103 action labels. Moreover, we access several visual grounding corpus [57–59]. A language model [1] is employed to extract actions and their corresponding targets (if exist) to form phrases from the corpus, leading to 5001 actions with manual checking. Totally, we collect 6104 action queries for searching videos on YouTube.

Collection Strategies. To ensure the quality of our dataset, we established specific crawling rules. We only collected videos that were between 10 seconds and 30 minutes in duration and had resolutions ranging from 360P to 720P. Videos with resolutions below 360P were excluded, and those above 720P were either downloaded in their 720P version or resized to 720P. In this process, we prioritize the highest available resolution. To provide a comprehensive mutimodal dataset, we gather videos along with their audio, subtitles, titles, and summaries. Captions for the videos were generated automatically using a video captioning pipeline described in Section 3.2.

In formation, the collected multimodal data contain videos V, their audios A, metadata (title Wtitle, video descriptions Wcontent, query words Wquery, tags Wtag, etc), subtitles (user generated contents or auto-generated ones), and more. Each video V could be treated as a sequence of clips {Ci}i=1,2,..., and we can segment their corresponding audio as {Ai}i=1,2,... and ASR subtitles as {Wiasr}i=1,2,.... For the metadata, we suppose clips share the same meta when they are sampled from the same video.

Trimming. We segment videos (lasting an average of 5 minutes) into clips (for around 10 seconds) using scene variance. For starters, videos are cut into shorter ones based on their scene changes.

|a table topped<br><br>with lots of food|
|---|

|a person is cooking some food in a frying pan.|
|---|

[Figure 21]

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

Sampled

Frames

|Image Captioning Model (BLIP2, Tag2Text)|
|---|

|a person cooking some food in a frying pan.|
|---|

|the food is prepared and ready to be eaten|
|---|

|a person cooking some food on a stove top|
|---|

|a table topped with lots of food|
|---|

|a person putting food in a basket|
|---|

Frame

Captions

|Language Model| |
|---|---|
| | |

|A person cooks various dishes on a stove top and frying pan, prepares the food, sets a table with lots of food, and puts some of it in a basket.|
|---|

Video

Captions

- Figure 2: The proposed multiscale video caption pipeline. The captions in coarse and fine scales are marked in green and dark green, respectively.

We directly employ the corresponding filter in PySceneDetect 1 with a threshold as 27. During this procedure, we also filter out clips in still or extreme dynamics (e.g. a browse of a photo gallery). After the filtering, we get total 234M video clips whose durations range from 2s to more than 30s.

#### 3.2 Multiscale Video Captioning

To generate video captions that are scalable, rich, and diverse, we employ a multiscale method with two distinct captioning strategies, as depicted in Figure 2. On the finer scale, we simplify the video captioning process by concentrating on the common objects, actions, and scene descriptions within the video clip. We deliberately overlook intricate details such as subtle facial expressions & movements, and other nuanced elements. On the coarser scale, we adopt the single-frame bias assumption from [60] and exclusively caption the central frame of the video. Given our focus on brief clips (around 10 seconds) filtered via scene segmentation, most videos predominantly display consistent objects without substantial appearance alterations. This circumvents the identity-preserving issue when dealing with videos from image perspectives. Technically, we employ the lightweight image captioning model Tag2Text [61] for the finer scale, which describes videos at low fps in a frame-by-frame manner. These individual image captions are then synthesized into a comprehensive video description using a pretrained language model [62, 63]. At the coarser scale, we use BLIP2 [64] to caption the middle frame of the clip.

#### 3.3 Statistics and Features

We present the key statistics of InternVid with other popular video-language datasets in Table 1. More detailed ones are given below.

Diversity & Richness. We collected videos from 16 popular categories with varying percentages, as illustrated in Figure 3. Unlike prior studies [2–4], we ensured diversity by selecting videos from countries with different languages instead of relying on a dominant language environment. The countries we sampled from include the UK, USA, Australia, Japan, Korea, China, Russia, and France, among others. In terms of duration, every video lasts 351.9s on average. Almost half (49%) of the videos are five minutes or less, while a quarter (26%) fall between five and ten minutes. Only 8% of the videos are over 20 minutes long. Among the curated videos, 85% were high-resolution (720P), while the remaining 15% had lower resolutions ranging from 360P to 720P. Although the lower-resolution videos may not perform as well as the high-resolution ones in content generation tasks, they can still be useful in video-language representation learning, provided that they have appropriate captions.

InternVid exhibits diverse clip durations and caption lengths in the segmented clip level. The aesthetic scores and clip-caption similarities are distributed uniformly, as shown in Figure 4. The majority of clips are 0-10 seconds in length, accounting for 85% of all clips (Figure 4: left). Approximately half of the clips have captions with 10-20 words, while one-third of the clip captions have fewer than 10 words. About 11% of clips have long captions with more than 20 words.

1https://github.com/Breakthrough/PySceneDetect

Video Countries Video Durations Video Categories

People & Blogs

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

4% jp th

Education

8%

[Figure 37]

15%

[Figure 38]

14%

<5min

[Figure 39]

16%

[Figure 40]

[Figure 41]

News & Politics Howto & Style

[Figure 42]

29%

[Figure 43]

5%

17%

5%

[Figure 44]

5min~10min

[Figure 45]

kor

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

5%

fra en

[Figure 50]

49%

Entertainment

[Figure 51]

[Figure 52]

10min~15min

5%

[Figure 53]

[Figure 54]

16%

18%

[Figure 55]

[Figure 56]

Gaming Science & Technology

[Figure 57]

[Figure 58]

15min~20min

[Figure 59]

9%

[Figure 60]

zh

[Figure 61]

[Figure 62]

[Figure 63]

14% 10%

[Figure 64]

[Figure 65]

26%

8% 18%

ru Other 4 languages

9%

[Figure 66]

>20min

Sports

[Figure 67]

[Figure 68]

[Figure 69]

Others 11 Categories

[Figure 70]

- Figure 3: Video statistics in InternVid. It encompasses a diverse set of categories, gathered from multiple countries and averaging a duration of five minutes.

|[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>36%<br><br>26%<br><br>23%<br><br>8%<br><br>7%<br><br>Clip Durations|[Figure 76]<br><br>0.0s~2.0s 2.0s~4.0s 4.0s~10.0s 10.0s~20.0s<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>>20s|[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>34%<br><br>55%<br><br>4%<br><br>7%<br><br>[Figure 85]<br><br>0~10words<br><br>[Figure 86]<br><br>10~20words<br><br>[Figure 87]<br><br>20~30words<br><br>[Figure 88]<br><br>>30words<br><br>Clip Caption Length|
|---|---|---|

|0<br><br>0.15<br><br>0.3<br><br>0.45<br><br>0<br><br>0.3<br><br>0.6<br><br>0.9<br><br>UMT-SIM & Aesthetic<br><br>UMT-SIM Aesthetic Score<br><br>|
|---|

- Figure 4: Clip statistics in InternVid. InternVid contains a diverse distribution of clip durations and caption lengths. It also offers aesthetic scores and multimodal similarity scores for each clip.

We measured the aesthetic scores of all clips using an open-source model [19]. We uniformly sampled four frames of each clip, calculated their aesthetic scores, and took the maximum score as the video aesthetic score. For clip-caption similarity computation, we used a video-language model called UMT [21]. We computed the cosine similarity between video embeddings and text embeddings, again using a uniform sampling of four frames for each clip. Most clips score around 4-6 in terms of aesthetics, accounting for approximately 75% of the data. For UMT-SIM, over 80% of the clips scored between 0.3-0.4, with the remaining clips scoring around 0.2-0.3 or 0.4-0.5. Based on these computed aesthetics and UMT-SIM scores, we can generate different versions of InternVid to meet various requirements.

Actionness. In terms of actionness, the InternVid dataset contains about ten times more verbs than the WebVid10M dataset. To evaluate this, we used the NLTK toolkit to analyze the number of verbs in captions, focusing on extracting and tagging all unique verbs. We found a total of 109,485 verbs in the WebVid10M caption dataset, while the InternVid dataset contained 212,155 unique instances of verbs. While these counts may not be entirely accurate due to our simple counting method, we believe they provide a rough indication of the actionness of the two datasets.

#### 3.4 Interleaved Video-Text Data Generation

Utilizing the created video captions, we can develop an integrated video-text dataset for in-context video learning, allowing video-based sequence models to perform new tasks without additional training. Previous research, such as Flamingo [7, 8], Kosmos-1 [65], and Multimodal C4 [66], confirms that pretraining on the interleaved image-text sequences results in significant multimodal in-context abilities. To the best of our knowledge, a large-scale interleaved video-text dataset has not yet been established. Our work represents the initial step in creating and making it publicly available.

We create InternVid-ICL, containing 7.1M interleaved video-text data pairs. We propose three distinct methods for organizing clips and their captions:

- • Arrange clips and their descriptions sequentially based on their temporal order within the same video, as illustrated in Figure 5 (a).
- • Enhance diversity in interleaved video-text items by assigning ASR text to a used clip in addition to its caption, as demonstrated in Figure 5 (b).

[Figure 89]

Figure 5: Interleaved video-text data generation in InternVid with three formats.

[Figure 90]

Figure 6: Framework of ViCLIP.

- • Extend method 1 by concatenating two interleaved multimodal items, creating a video-centric dialogue simulating user queries involving multiple videos (Figure 5 (c)).

[..., "the inside of a home has a rug and a light on.", "♪ We could leave the Christmas lights up til January ♪", ..., "woman with blond hair playing guitar", "♪ Have I known you 20 seconds or 20 years?

|[Figure 91]<br><br>|
|---|

|[Figure 92]<br><br>|
|---|

|[Figure 93]<br><br>|
|---|

♪",

, "close-up of a bathroom sink with soap bubbles and other items", "a bathroom is seen with a sink and two lights", "a woman swiming inside of a fishbowl with a ladder and

|[Figure 94]<br><br>|
|---|

|[Figure 95]<br><br>|
|---|

|[Figure 96]<br><br>|
|---|

a man", "♪ Can I go wher you go? ♪", , "devils roll the dice, angels roll their eyes","♪ And, take me out, and take me home ♪" ,..., "the man is standing in a room with pink carpet","♪ You’re my, my ♪", "a woman in yellow is dancing with a man in a red room", "♪ My, My lover ♪",

|[Figure 97]|
|---|

|[Figure 98]<br><br>|
|---|

|[Figure 99]<br><br>|
|---|

, "a woman is sitting on a chair, playing a guitar and a woman holding a balloon", "♪ ♪ ♪", "two men smiling while holding wine glasses and drinking beer", "♪ We could let our friends crash in the living room ♪" ...]

- Table 2: Interleaved video-text data format (b) in InternVid. The caption and ASR transcript of each clip is shown in black and gray, respectively. We can achieve interleaved video-text data format (a) by abandoning ASR transcripts. To obtain data format (c), we concatenate multiple videos with interleaved video-text data (a). One visual example of these arrangements is provided in Table 9.

### 4 ViCLIP: Learning Video-Text Representation at Scale

Built upon CLIP [67], we make a simple video-text pretraining baseline ViCLIP. It consists of a video encoder (ViT) [68] and a text encoder, as given in Figure 6. Both modules are initialized

K400 K600 K700 top-1 (↑) AVG (↑) top-1 (↑) AVG (↑) top-1 (↑) AVG (↑)

Method Training Data

CLIP CLIP400M 58.42 70.14 55.11 67.16 46.12 58.38 CLIP DataComp-1B 56.14 67.67 54.15 65.83 45.36 57.01 EVA-CLIP-L Merged-2B - 65.00 - 64.90 - 59.10 EVA-CLIP-E LAION-2B - 69.80 - 69.30 - 63.40 ViCLIP +WebVid10M 59.88 71.03 58.66 69.84 50.23 61.86 ViCLIP +InternVid-10M 56.68 68.17 54.67 66.28 46.53 58.73 ViCLIP +InternVid-50M 57.18 68.93 55.36 67.07 47.00 59.36 ViCLIP +InternVid-200M 59.80 71.09 57.80 69.34 49.30 61.25 ViCLIP +InternVid-10M-DIV 63.00 74.15 60.68 72.07 52.50 64.59 ViCLIP +InternVid-10M-FLT 64.80 75.70 62.20 73.53 54.30 66.38

Table 3: Zero-shot action recognition results on Kinetics 400/600/700.

from the corresponding CLIP components. We update the native attention in the video encoder to spatiotemporal attention while maintaining other design elements. For efficient learning, we apply masking to videos in pre-training. The optimization target is the contrastive loss between input video and text embeddings.

Video & Text Encoders with Masking Learning. Our video encoder uses a standard ViT with spatiotemporal attention. We apply random patch masking following MAE-based methods [37, 38] to the input videos. It significantly alleviates the computational burden. The used text encoder is also a transformer followed by [67, 19].

Unmasked Video-Text Pretraining. We feed all visual tokens into the video transformer instead of just the masked ones towards the end of the pretraining process. This helps bridge the gap between pretraining and downstream applications where the full video is used as input. We perform unmasked training for 0.5 epochs with a learning rate of 4e-6.

Training Objectives. Our framework optimizes video-text alignment. It minimizes InfoNCE loss [69] using global video and text features, as

N

LC = LVC →T+LTC→V = −

i=1

N

exp(sim(fiV, fiT)/τ) N j=1 exp(sim(fiV, fjT)/τ) −

log

i=1

exp(sim(fiT, fiV)/τ) N j=1 exp(sim(fiT, fjV)/τ)

log

, (1)

where fV and fT denote the learned video and text embeddings, respectively. sim(·) computes the cosine similarity between two features. τ is the learnable temperature.

Implementation. ViCLIP is learned with 64 NVIDIA A100 GPUs for 3 days with 50M video-text pairs. We introduce DeepSpeed and FlashAttention [70] for training and inference acceleration.

We learn ViCLIP on five subsets of InternVid and evaluated its performance on popular video-related benchmarks using full-finetuned and zero-shot settings. We sample subsets InternVid-10M, InternVid50M, and InternVid-200M randomly. For InternVid-10M-DIV, we prioritize to sample clips from different videos first, then we sample clips with varying probabilities according to the video length where they are extracted. The longer their source video is, the lower chance they are sampled. For InternVid-10M-FLT, we employ the sampling strategy of InternVid-10M-DIV and select clips with UMT-SIM scores ranking among the top 30% to ensure high quality.

#### 4.1 Transferable Video Representation Performance

Action Recognition. In addition to OpenAI’s CLIP-L (CLIP400M [67]) and LAION (DataComp1B [71]), we also include EVA-CLIP-L/14 and EVA-CLIP-E/14 [72] for comparison. More experimental settings are given in App. E.1.

Zero-Shot. Table 3 shows that when trained on InternVid-10M-FLT, ViCLIP outperforms all other methods, including EVA-CLIP-E. This result validates InternVid’s effectiveness in learning videotext embeddings. Note that ViCLIP with InternVid-10M-FLT sets new records on zero-shot action recognition in Kinetics 400/600/700, demonstrating a significant performance boost compared to ViCLIP with WebVid10M or other models. Moreover, ViCLIP trained on InternVid-10M-FLT exceeds its performance on InternVid-200M. Normally, we would expect the model trained on InternVid-200M to perform better than those on -10M-DIV or -FLT, given that the latter two subsets

K400 SthSthV2 top-1 (↑) top-5 (↑) top-1 (↑) top-5 (↑)

Method Training Data

CLIP CLIP400M 86.7 97.2 70.1 92.5 CLIP DataComp-1B 85.6 96.8 68.9 91.8 ViCLIP +WebVid10M 85.0 96.8 68.7 91.9 ViCLIP +InternVid-10M-FLT 86.8 97.5 71.2 93.2 ViCLIP +InternVid-10M-FLT+K710 88.0 97.8 71.8 93.6 ViCLIP +InternVid-200M 87.9 97.9 73.6 94.9 ViCLIP +InternVid-200M+K710 88.7 98.2 74.2 95.0

Table 4: Fine-tuned action recognition results on Kinetics 400 and SomethingSomethingV2.

MSR-VTT LSMDC DiDeMo MSVD ANet T2V V2T T2V V2T T2V V2T T2V V2T T2V V2T

Method Data

CLIP CLIP400M 29.0 25.8 13.9 15.2 11.5 19.1 37.9 60.0 8.3 12.2 CLIP DataComp-1B 30.4 24.2 13.9 11.9 12.7 18.7 40.5 57.2 9.1 13.2 CLIP4Clip [73] +HowTo100M 32.0 - 15.1 - - - 38.5 - - ViCLIP +WebVid10M 35.6 33.1 16.5 13.4 14.5 23.3 45.3 69.0 12.4 19.0 ViCLIP +InternVid-10M 36.4 37.1 17.1 15.0 16.4 25.9 45.2 69.8 13.5 23.4 ViCLIP +InternVid-50M 39.7 40.7 18.0 16.7 16.7 26.4 46.5 72.2 13.6 23.2 ViCLIP +InternVid-200M 39.3 39.5 18.3 16.6 17.1 25.5 47.3 70.0 13.7 21.6 ViCLIP +InternVid-10M-DIV 41.5 41.6 18.5 17.4 17.7 26.2 48.6 71.9 14.8 23.4 ViCLIP +InternVid-10M-FLT 42.4 41.3 20.1 16.9 18.4 27.9 49.1 75.1 15.1 24.0

- Table 5: Results of zero-shot video retrieval on MSR-VTT, LSMDC, DiDeMo, MSVD, and ANet.

derive from the former. Unless this discrepancy results from improper learning, we conjecture that false negative samples could severely impede video-text contrastive learning if we don’t purposefully reduce the number of clips taken from the same video. Specifically, we hypothesize that clips from the same video share similar representations and captions. Contrastive learning, however, assumes these clips to be different. This situation also undermines the significance of using a large batch size in current training since it increases the probability of encountering more false negatives. We believe this assumption is applicable to other video tasks as well and plan to explore this further in the future. Fine-tuned. In Table 4, note when comparing ViCLIP trained on InternVid with image CLIP models or ViCLIP trained with WebVid, there is a clear increase in accuracy. Unlike the zero-shot results, when ViCLIP is pretrained with a larger number (200M) of video-text data pairs, it achieves higher accuracy in fine-tuned recognition tasks (87.9% in K400 and 73.6% in SthSthV2) compared to when pretrained (86.8% in K400 and 71.2% in SthSthV2) with fewer data (10M). This suggests that InternVid provides greater benefits for fine-tuned action-related tasks. The decrease in performance of ViCLIP with WebVid highlights the importance of addressing the distribution gap between WebVid and the action videos used for evaluation, emphasizing the need to collect videos with evident temporal dynamics. Video-Text Retrieval. We evaluate the video retrieval performance of baselines and ViCLIP using different pretraining datasets on five popular benchmarks [52, 45, 74, 46, 75], as shown in Table 5 and 6. We uniformly sample eight frames from the input videos. For the CLIP models from OpenAI [67] and LAION [19], we utilize their officially released ViT-L models and extract video embeddings by averaging the computed frame-wise image embeddings. Our ViCLIP directly predicts video embeddings. For evaluating retrieval performance, we report R@1 scores for both text-to-video (t2v) and video-to-text (v2t) tasks in 5 and 6.

Both Table 5 and 6 demonstrate that video-language pretraining is crucial for enhancing fine-tuned and zero-shot retrieval performance. This point is substantiated by the comparison between CLIP and ViCLIP using InternVid-50M. Table 5 exhibits a boost of nearly 4-10 points across different benchmarks in the zero-shot setting. Meanwhile, Table 6 shows an increase of approximately 10 points across all R@1 scores in the fine-tuned setting.

Zero-Shot. Table 5 reveals InternVid-10M outperforms WebVid when employing the same method, ViCLIP, with an average increase of 6.3% in R@1 across nearly all benchmarks. This improvement can be further amplified by diversifying the training clips used, as InternVid-10M-DIV and -FLT surpass WebVid on ViCLIP with gains in R@1 of 14.0% and 17.1%, respectively. These results underline, once again, the effectiveness of the correspondence between our generated video captions

MSR-VTT LSMDC DiDeMo MSVD ANet T2V V2T T2V V2T T2V V2T T2V V2T T2V V2T

Method Data

CLIP CLIP400M 38.2 38.7 22.5 22.6 32.2 33.9 67.3 69.9 26.1 26.9 CLIP DataComp-1B 37.2 37.5 18.7 18.5 33.5 34.2 66.3 70.2 24.5 25.8 CLIP4Clip [73] +HowTo100M 45.6 45.9 24.3 23.8 43.0 43.6 45.2 48.4 40.3 41.6 ViCLIP +WebVid10M 50.8 49.3 27.3 28.4 48.1 48.5 76.7 81.2 44.5 43.2

ViCLIP +InternVid-10M 51.8 49.7 28.5 29.4 49.5 50.6 77.2 80.0 49.7 48.4 ViCLIP +InternVid-50M 52.8 52.2 30.9 30.9 49.4 48.7 78.1 80.0 49.7 49.0 ViCLIP +InternVid-200M 53.7 53.4 29.3 31.3 51.1 50.8 79.9 78.4 52.8 51.1 ViCLIP +InternVid-10M-DIV 55.0 53.3 32.0 30.0 51.7 52.1 75.8 77.8 50.4 48.9 ViCLIP +InternVid-10M-FLT 52.5 51.8 33.0 32.5 49.4 50.2 77.2 79.0 49.8 48.1

- Table 6: Results of fine-tuned video retrieval on MSR-VTT, LSMDC, DiDeMo, MSVD, and ANet.

[Figure 100]

[Figure 101]

Figure 7: Zero-shot action recognition (top-1 accuracy) on Kinetics-400 / -600 / -700.

Figure 8: Video retrieval average performance (text-to-video R@1) across five datasets.

and their corresponding videos. Comparing CLIP4Clip using HowTo100M with ViCLIP using WebVid10M or InternVid-10M shows that the correlation between video and text influences performance more significantly than their quantity. Moreover, the zero-shot performance demonstrates that the video-text representation learned using InternVid is transferable. This claim is supported by its superior performance across multiple video retrieval benchmarks.

Fine-Tuned. Table 6 exhibits a noticeable improvement when transitioning from InternVid-10M to WebVid10M while using ViCLIP for both t2v and v2t retrieval across almost all datasets. On average, there is a 3.7% increase in t2v R@1 across all benchmarks, with particularly significant rise observed in ActivityNet (an increase of over 11.9%). However, ViCLIP using WebVid10M yields better v2t R@1 scores than when using InternVid-10M (81.2 vs. 80.0). We believe this does not alter the overall trend that InternVid-10M generally provides more advantage to ViCLIP than WebVid10M does.

The benefits of used video data become even more apparent when comparing InternVid-10M-DIV or InternVid-10M-FLT with WebVid10M. Their overall increases are 5.8% and 5.1%, respectively. Despite these improvements, issues related to data diversity persist.

Data Scaling and Issues. Figure 7 and 8 illustrate how ViCLIP’s performance changes in zeroshot and fine-tuning settings when varying the scale of InternVid. In both scenarios, increasing the data scale results in significant increases in performance. As shown in Figure 7, ViCLIP’s discriminative ability linearly increases with the increasing volume of training videos used (10M → 200M). Meanwhile, Figure 8 shows that the retrieval performance increase becomes marginal when scaling the training data beyond 50M. It’s vital to note our model is trained using only contrastive loss without employing popular designs such as matching head and its corresponding loss. Consequently, this retrieval result doesn’t allow for any definitive conclusions about whether there exists a turning point after which scaling up the training videos becomes less beneficial currently. More explorations are necessary in these retrieval experiments. However, these findings generally suggest that enhancing the scale of pretraining data can improve the transferability of the learned representation.

UCF-101 MSR-VTT IS (↑) FID (↓) FVD (↓) CLIPSIM (↑)

Method Training Data

VideoCrafter2 WebVid10M 18.26 66.95 910.87 0.2875 VideoFusion 3 WebVid10M 17.49 75.77 639.90 0.2795

t2v baseline WebVid10M 13.97 98.25 705.25 0.2657 t2v baseline WebVid10M+InternVid18M 21.04+7.07 60.25−38.00 616.51−88.74 0.2951+0.0294

Table 7: Zero-shot text-to-video generation performance.

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

VideoCrafter （WebVid10M)

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

VideoFusion （WebVid10M)

a bald man in a black t-shirt is playing a guitar

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

t2v baseline （WebVid10M)

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

t2v baseline (+InternVid)

- Figure 9: Comparison of samples from t2v baseline to others. The used prompt is: a bald man in a black t-shirt is playing a guitar.

#### 4.2 Text-to-Video Generation

Our InternVid dataset improves existing text-to-video generation models by providing video-text pairs with high correspondence. To establish a video generation baseline, we extend spatiotemporal modeling on the latent space of an open-source text-to-image diffusion model [76]. We train the video generation approach with two settings: one using WebVid10M [6], and the other using InternVidAesthetics-18M in addition to WebVid10M [6]. InternVid-Aesthetics-18M is a subset of InternVid consisting of clips with an aesthetic score of at least 4. Quantitative (Table 7) and qualitative (Figure 18) evaluations demonstrate the effectiveness of InternVid in video generation tasks. To evaluate our models quantitatively, we perform zero-shot text-to-video experiments and randomly sample 2,020 videos from the UCF-101 dataset and 2,990 videos from the MSRVTT dataset. Following the protocols in [77], we report CLIPSIM, IS, FID, and FVD metrics.

In Table 7, we observe that our t2v baseline trained on WebVid10M performs poorly in terms of IS, FID, and CLIPSIM when compared to other approaches. However, with the addition of InternVid-Aesthetics-18M, our t2v baseline demonstrates significant improvements in these metrics and outperforms other methods by a considerable margin. In Figure 18, we observe that the text-tovideo (t2v) baseline using both WebVid10M and InternVid-Aesthetics-18M significantly outperforms other methods in terms of visual quality and temporal coherence. It is worth noting that the t2v baseline using InternVid does not contain watermarks, which is a data bias in WebVid10M. These results demonstrate the potential of InternVid for high-quality video generation.

#### 4.3 Video-Centric Dialogue System

Inspired by recent vision-centric dialogue systems [10, 78, 79], we integrate our pretrained ViCLIP (with InternVid) into VideoChat [10] to show how our data and model can empower multimodal dialogue methods with effective video modeling capability. In implementation, we inherit nearly all

|Where is this video taken from?<br><br>[Figure 122]<br><br>[Figure 123]<br><br>The video is taken from an aerial view of a skyscraper-filled city, with the Shanghai skyline being the prominent feature in the background. The specific location is Shanghai, China, with several skyscrapers and a river visible in the scene. Additionally, a few specific buildings and roads are also seen, such as the Oriental Pearl Tower and the Huangpu River.<br><br>|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|[Figure 128]|
|---|---|---|---|---|
<br><br>How many buildings in 00:12? There are several buildings visible in the 00:12 frame, including<br><br>[Figure 129]<br><br>[Figure 130]<br><br>several towers and skyscrapers, as well as a river and a city skyline.<br><br>The exact number of buildings is difficult to count, as they are interspersed with other objects and structures in the scene, but there are several visible. The buildings include the Oriental Pearl Tower, the World Financial Center, and the IFC Mall, among others.|
|---|

- Figure 10: Video Spatial Understanding Task. This figure demonstrates the spatial understanding and temporal localization capabilities of VideoChat-ViCLIP trained with our dataset.

|Describe the activity in the video.<br><br>[Figure 131]<br><br>[Figure 132]<br><br>The man is playing the guitar and moving around. The video shows<br><br>him holding the guitar and playing it.<br><br>|[Figure 133]|[Figure 134]|[Figure 135]|[Figure 136]|[Figure 137]|
|---|---|---|---|---|
|
|---|

- Figure 11: Video Action Recognition Task. The video comes from Kinetics-400, with the label "playing guitar".

designs of VideoChat-Embed, just replacing its visual encoder with our ViCLIP (trained on InternVid). We evaluate VideoChat-ViCLIP in spatial understanding (Figure 10), action recognition (Figure 11), temporal understanding (Figure 12), video reasoning (Figure 13), and video creative (Figure 14) tasks. Our qualitative evaluations demonstrate its decent video-to-text capabilities, suggesting promising potential for improving video captioning further.

|Evaluation Aspect<br><br>|Correctness Detail Contextual Temporal<br><br>Consistency of Information Orientation Understanding Understanding<br><br>|Avg|
|---|---|---|
|VideoChat (Eva-g) LLaMA Adapter Video LLaMA Video-ChatGPT VideoChat-ViCLIP|2.23 2.5 2.53 1.94 2.24 2.03 2.32 2.3 1.98 2.15<br><br>1.96 2.18 2.16 1.82 1.79<br><br>2.4 2.52 2.62 1.98 2.37<br><br><br>2.86 2.52 3.08 2.36 2.4<br><br>|2.29 2.16 1.98 2.38 2.64|

Table 8: Performance benchmarking of text generation models.

In terms of quantitative comparison, as shown in Table 4.3, VideoChat-ViCLIP significantly outperforms the vanilla VideoChat (using Eva-g as the vision encoder) and other systems across all evaluation aspects of the quantitative video conversation evaluation framework in [78]. Specifically, the model shows remarkable improvements in the correctness of information (from 2.23 to 2.86),

|Step-by-step describe the video.<br><br>[Figure 138]<br><br>[Figure 139]<br><br>First, a woman in a green shirt stands in a kitchen with a transparent bowl on a table, holding a knife. Next, she uses the<br><br>knife to cut a green apple on the table in the kitchen. Then, she<br><br>continues to cut the apple, still using the knife. Afterward, she cuts another apple on the table, holding the knife in her hand. Finally, she cuts the apple and puts it in a glass blender on the table. Throughout the video, other apples are visible on the table, and the woman uses a knife to cut them as well. The background<br><br>consists of a wooden table, countertops, and a kitchen. The video<br><br>is in slow motion and has a blurry, green-tinted appearance.<br><br>|[Figure 140]|[Figure 141]|[Figure 142]|[Figure 143]|[Figure 144]|
|---|---|---|---|---|
<br><br>What will she do next?<br><br>[Figure 145]<br><br>Based on the context, it is reasonable to assume that she may continue to cut apples and prepare them for juicing.<br><br>[Figure 146]|
|---|

- Figure 12: Temporal Understanding Task. VideoChat-ViCLIP can easily handle temporal understanding tasks and make predictions based on these temporal sequences due to its training on rich temporal information on InternVid.

|What is strange about this video?<br><br>[Figure 147]<br><br>[Figure 148]<br><br>It shows a man and a lion interacting in a forest. The man is holding a lion by the ears, and the lion seems to be okay with being petted by the man. They seem to be enjoying each other’s company. The video shows a rare and unusual interaction between a man and a lion, which is very rare and surprising to see in real life.<br><br>|[Figure 149]<br><br>|[Figure 150]<br><br>|[Figure 151]<br><br>|[Figure 152]<br><br>|[Figure 153]<br><br>|
|---|---|---|---|---|
|
|---|

- Figure 13: Video Reasoning Task. Our dialogue system has the ability to understand common sense and make causal inferences based on the contents within the video.

contextual understanding (from 2.53 to 3.08), and temporal understanding (from 1.94 to 2.36). The average score also increases from 2.29 to 2.64, showing an overall performance gain.

### 5 Conclusion

Our dataset, InternVid, is designed for multimodal research (both understanding and generation) focused on videos. It consists of over 230 million video clips sourced from 7 million high-resolution (720P) YouTube videos. We use existing models with a multiscale approach to generate clip-level descriptions. Our studies confirm the efficacy of captions, and the large volume of video-text data enables crossmodal learning and text-to-video generation at scale. By training with our data, we develop a video-text representation baseline ViCLIP using ViT-L and analyze briefly how the data scale affects learned crossmodal embeddings. In addition to perception tasks, we show that InternVid improves text-to-video generation performance when using a subset of clips based on their aesthetic

|Write a short poem for this video.<br><br>[Figure 154]<br><br>[Figure 155]<br><br>In blue flowers she walks, wind in hair. White straw hat, peace in her stare. She breathes, admires the sky. Tall she stands, with eyes up high.<br><br>|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|[Figure 160]<br><br>|
|---|---|---|---|---|
|
|---|

- Figure 14: Video Creative tasks. This figure shows the performance of VideoChat-ViCLIP on creative and generative tasks within our dataset, like generating poetry based on video content.

scores. With its data, annotations, metadata, and computed scores, we believe InternVid can fuel a variety of studies and applications.

[Figure 161]

- Figure 15: The word cloud (Top-200) of the generated captions in the InternVid dataset reveals that the captions predominantly highlight the rich actions of the objects.

### A Data Availability Statement

We are committed to maintaining transparency and compliance in our data collection and sharing methods. In accordance with these principles, please note the following:

Publicly Available Data: The data utilized in our studies is publicly available. We do not use any exclusive or private data sources.

Data Sharing Policy: Our data sharing policy builds upon the precedent set by prior works like Kinetics, HD-VILA, and others. Instead of providing the original raw data, we only supply the YouTube video IDs necessary for downloading the respective content.

Usage Rights: The data released by us is intended exclusively for research purposes. Any potential commercial usage is not sanctioned under this agreement.

Compliance with YouTube Policies: Our data collection and release practices are strictly in accord with YouTube’s data privacy policies. We ensure that no user data or privacy rights are violated during the process.

Data Licence: We employ the protocol of CC BY 4.0.

### B Limitations & Societal Impact

All video data used in our research are downloaded from YouTube using Safe for Work (SFW) queries and channels. To ensure appropriate content, we employ a simple NSFW filter: a binary classifier designed to recognize and exclude non-ethical videos. For privacy considerations and in respect of data sharing practices, we share only the YouTube ID of the videos, similar to previous academic works. This approach aligns with YouTube’s data protocols and ensures no violation of privacy or

[Figure 162]

[Figure 163]

English. Chinese.

[Figure 164]

[Figure 165]

Korean. German.

- Figure 16: The word clouds of the ASR transcripts of four different languages (English, Chinese, Korean, and German). We collect videos from various countries or regions with 11 different languages. Here we list four of them to show how these transcripts are distributed in words.

data usage rules. Despite these precautions, our work has some limitations, primarily related to data diversity and representativeness. Although YouTube is an extensive source encompassing a wide range of video categories, certain specific types of footage may be excluded or scarcely collected, including: public area surveillance, sports competitions, movies, documentaries, etc. The exclusion of such categories is often due to copyright restrictions or other limits imposed by the platform. Therefore, while our dataset provides a broad view of everyday video content, its coverage does not extend to every possible category or type of video. These limitations should be taken into account when considering the generalizability of our results across all types of video data.

### C More Statistics in InternVid

Actionness. InternVid contains way more verbs than the WebVid10M. We used NLTK toolkit to analyze the number of verbs in captions, focusing on tagging all unique verbs. We found a total of 109,485 verbs in the WebVid10M, while InternVid contained 212,155 ones. While the counts may not be that accurate due to our simple counting, we believe they provide a rough indication of the actionness of the two datasets.

Video Caption and Transcript Distribution. To analyze the word distribution of our generated captions and multilingual (ASR) transcripts, we compute their distributions. The resulting word distribution of the captions is presented in Figure 15, which includes objects (tv, car, door, plant, etc.), attributes (green, young, large, long, etc.), locations (middle, behind, south, next, etc.), scenes (room, stage, kitchen, office, etc.), actions/events (walking, eating, cutting, holding, etc.), and more.

We also include four word distributions of different languages in Figure 16, reflecting trends in different countries and offering potential data customization along with the provided metadata.

### D InternVid-ICL: Interleaved Video-Text for In-Context Video Learning

Visual Examples. As given in the paper, we provide examples video+text interleaved entries for in-cntext learning as Flamingo. Table 9 gives an example about format (a): arrange clips and their descriptions sequentially based on their temporal order within the same video. Note the videos are randomly dropped with a probability (0.3) for constructing richer text context compared with the original video-text pair combinations in sequential.

[..., "the inside of a home has a rug and a light on.", "♪ We could leave the Christmas lights up til January ♪", ..., "woman with blond hair playing guitar", "♪ Have I known you 20 seconds or 20 years?

|[Figure 166]<br><br>|
|---|

|[Figure 167]<br><br>|
|---|

|[Figure 168]<br><br>|
|---|

♪",

, "close-up of a bathroom sink with soap bubbles and other items", "a bathroom is seen with a sink and two lights", "a woman swiming inside of a fishbowl with a ladder and

|[Figure 169]<br><br>|
|---|

|[Figure 170]<br><br>|
|---|

|[Figure 171]<br><br>|
|---|

a man", "♪ Can I go wher you go? ♪", , "devils roll the dice, angels roll their eyes","♪ And, take me out, and take me home ♪" ,..., "the man is standing in a room with pink carpet","♪ You’re my, my ♪", "a woman in yellow is dancing with a man in a red room", "♪ My, My lover ♪",

|[Figure 172]|
|---|

|[Figure 173]<br><br>|
|---|

|[Figure 174]<br><br>|
|---|

, "a woman is sitting on a chair, playing a guitar and a woman holding a balloon", "♪ ♪ ♪", "two men smiling while holding wine glasses and drinking beer", "♪ We could let our friends crash in the living room ♪" ...]

Table 9: Interleaved video-text data format (b) in InternVid. The caption and ASR transcript of each clip is shown in black and gray, respectively. We can achieve interleaved video-text data format (a) by abandoning ASR transcripts. To obtain data format (c), we concatenate multiple videos with interleaved video-text data (a).

### E Implementation Details

#### E.1 ViCLIP

Action Recognition. In the zero-shot action recognition, we sample 8 frames in each video. Following the settings in CLIP and EVA-CLIP, we report the mean of top-1 and top-5 accuracy for Kinetics-400 / -600 / -700. In Section 4.1, we show ViCLIP learnt on WebVid or InternVid is an effective zero-shot action recognition model.

In the full fine-tuned setting, we conduct two experiments with two receipts. In Table 4, for the experiments where the training data excluded K710, we followed the common practice of finetuning the pretrained ViCLIP with the training data from the evaluation dataset. On the other hand, for the experiments where the training data included K710, we adopted a training trick inspired by [26]. We first finetuned the pretrained ViCLIP with K710 [26], and then proceeded with the common supervised finetuning setting. By incorporating the supervised finetuning with K710, ViCLIP demonstrated better performance in the fine-tuned tasks compared to experiments that did not include K710.

Video Retrieval. In the full-finetuning setting, we tune the pretrained ViCLIP with not only videotext contrastive loss but also video-text matching loss on the training data of the evaluated benchmarks. During both training and testing, we sample 12 frames. Detailed hyper-parameters are given in Table 10. In the zero-shot setting, we sample only 8 frames for evaluations.

#### E.2 Video Generation Baseline

We used the spatiotemporal modeling approach from [80] and built our text-to-video generation baseline on the work of [76]. Our approach consists of a U-Net with a transformer that models its latents, using interleaved spatiotemporal attention (ST-Attn), cross-attention for visual-text, a feed-forward network (FFN), and temporal attention (T-Attn), as illustrated in Figure 17. To adapt the 2D convolutional layers in [76] to 3D, we extended 3 × 3 kernels into 1 × 3 × 3 ones. We also extended the original spatial attentions to spatiotemporal ones. We initialized our baseline using all text-to-image diffusion model parameters, while the newly added temporal attention layers used default parameters.

|config<br><br>|MSRVTT DiDeMo ANet LSMDC MSVD|
|---|---|
|optimizer optimizer momentum weight decay learning rate schedule learning rate batch size warmup epochs total epochs input frame max text length drop path flip augmentation augmentation|AdamW β1,β2=0.9,0.999 0.02 cosine decay<br><br>2e-5 4e-5 2e-5 2e-5 4e-5<br><br>256 1<br><br>7 8 5 10 20<br><br>12<br><br>32 96 64 64 150 0.3 0.2 0.3 0.3 0.2<br><br>yes MultiScaleCrop [0.5, 1]<br><br>|

Table 10: Video-text retrieval fine-tuning settings.

Reconstruction Loss

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Diffusion

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

… …

noise

Conv Block

“A cat walking on the piano. ”

Transformer

Block

[Figure 197]

Training Frozen

|STAttn<br><br>Cross<br><br>-Attn FFN T-<br><br>Attn<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]|
|---|

[Figure 202]

Figure 17: Framework of our text-to-video generation baseline.

For the ST-Attn implementation, we used frame embeddings from the U-Net encoder instead of video embeddings as in [80]. We concatenated the embeddings of the previous and current frame for values and keys in attention, while using the current frame embedding alone as queries. The rest of the implementation remained the same as the original.

Text-to-Video Evaluation. To evaluate our text-to-video model, we conducted zero-shot experiments on the UCF-101 and MSRVTT datasets, following the method from [77]. For UCF-101, we used the class names as text prompts and generated 20 samples per class (total of 2,020 videos). For MSRVTT, we randomly selected one caption per video from the official test set (total of 2,990 videos). To ensure a fair comparison, we used the official implementation of VideoCrafter and VideoFusion [81] to generate the same number of videos with the same text prompts. During video sampling and evaluation, we generated 16 frames per video.

We assess the overall quality of the synthesized results on UCF-101 using framewise-FID, FVD, and Inception Score (IS), and evaluate the text-video semantic similarity on MSRVTT using clip similarity (CLIPSIM). For framewise-FID and IS, we use the pretrained Inceptionv3 network weights as our image encoder. For FVD, we use the pretrained InceptionI3d model and followed the TATS method [82]. To compute CLIPSIM, we calculate the clip text-image similarity for each frame with respect to the given text prompts and computed the average score. We use the ViT-B-32 clip model as the backbone, consistent with previous work [77].

Retrieval Action Recognition Zero-Shot Fine-Tuned Zero-Shot MSR-VTT MSR-VTT K400 K600 K700

Captioning Method

T2V V2T T2V V2T top-1 AVG top-1 AVG top-1 AVG VideoChat 33.9 32.3 46.6 47.1 54.68 67.74 51.70 64.91 43.67 56.51

Ours 38.6 38.5 49.0 49.2 58.52 71.11 55.37 68.27 47.09 59.98 Table 11: Video retrieval and action recognition results of ViCLIP-B trained on InternVid-FLT-10M with the captions generated by VideoChat and our captioning approach.

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

VideoCrafter （WebVid10M)

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

VideoFusion （WebVid10M)

a bald man in a black t-shirt is playing a guitar

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

t2v baseline （WebVid10M)

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

t2v baseline (+InternVid)

Figure 18: Comparison of samples from t2v baseline to others. We provide zero-shot text-to-video generation results of different methods trained on both WebVid10M and the additional InternVidAes-18M. The used prompt is: a bald man in a black t-shirt is playing a guitar.

### F More Results

#### F.1 Effectiveness of Our Multiscale Captioning Approach

To further validate the effectiveness of our proposed captioning method, we establish a video caption baseline using the video multimodal model VideoChat [10] for comparison. We input the video clip into the model with the prompt "Please describe the content in the given video." and apply it to InternVid-10M-FLT, resulting in 10 million new captions generated by VideoChat. Subsequently, we train two versions of ViCLIP-Base using InternVid-10M-FLT, each version trained with one of the two types of captions.

Table 11 demonstrates that ViCLIP-B trained using our captions outperforms the version trained using captions from VideoChat in both video retrieval (MSR-VTT) and action recognition (K400/600/700). These results are particularly noteworthy considering that the only difference in training lies in the captions generated by the two different approaches. Therefore, these findings further confirm the superior performance of our proposed captioning method compared to the baseline VideoChat.

#### F.2 Text-to-Video Generation

In Figure 18, we observe that the t2v baseline using both WebVid10M and InternVid-Aes-18M significantly outperforms others in visual quality and temporal coherence. Note that the t2v baseline using InternVid does not contain watermarks, which is a data bias in WebVid10M.

### References

- [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877–1901, 2020.
- [2] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In ICCV, pages 2630–2640, 2019.
- [3] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In CVPR, pages 5036–5045, 2022.
- [4] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. NeurIPS, 34:23634–23651, 2021.
- [5] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. In CVPR, pages 16375–16387, 2022.
- [6] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, pages 1728–1738, 2021.
- [7] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. ArXiv, abs/2204.14198, 2022.
- [8] Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev, et al. Openflamingo, 2023.
- [9] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [10] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [11] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Internchat: Solving vision-centric tasks by interacting with chatbots beyond language. arXiv preprint arXiv:2305.05662, 2023.
- [12] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.
- [13] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, pages 2556–2565, 2018.
- [14] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pages 3558–3568, 2021.
- [15] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In CVPR, pages 17980–17989, 2022.
- [16] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431, 2021.

- [17] Conghui He, Zhenjiang Jin, Chao Xu, Jiantao Qiu, Bin Wang, Wei Li, Hang Yan, JiaQi Wang, and Dahua Lin. Wanjuan: A comprehensive multimodal dataset for advancing english and chinese large models. arXiv preprint arXiv:2308.10755, 2023.
- [18] Conghui He, Wei Li, Zhenjiang Jin, Wang Wang, Chao Xu, and Dahua Lin. Opendatalab: Empowering general artificial intelligence with open datasets. https://opendatalab.com, 2022.
- [19] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.
- [20] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipediabased image text dataset for multimodal multilingual machine learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2443–2449, 2021.
- [21] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. arXiv preprint arXiv:2303.16058, 2023.
- [22] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In ECCV, pages 407–426. Springer, 2022.
- [23] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In CVPR, 2020.
- [24] Tianhao Li and Limin Wang. Learning spatiotemporal features via video and text pair discrimination. CoRR, abs/2001.05691, 2020.
- [25] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021.
- [26] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, and Yu Qiao. Uniformerv2: Spatiotemporal learning by arming image vits with video uniformer. arXiv preprint arXiv:2211.09552, 2022.
- [27] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-andlanguage transformers. In CVPR, 2022.
- [28] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks? arXiv preprint arXiv:2107.06383, 2021.
- [29] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783, 2021.
- [30] Chen Sun, Austin Myers, Carl Vondrick, Kevin P. Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. ICCV, 2019.
- [31] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. CVPR, 2020.
- [32] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.
- [33] Guo Chen, Sen Xing, Zhe Chen, Yi Wang, Kunchang Li, Yizhuo Li, Yi Liu, Jiahao Wang, Yin-Dong Zheng, Bingkun Huang, et al. Internvideo-ego4d: A pack of champion solutions to ego4d challenges. arXiv preprint arXiv:2211.09529, 2022.
- [34] Ziyun Zeng, Yuying Ge, Xihui Liu, Bin Chen, Ping Luo, Shu-Tao Xia, and Yixiao Ge. Learning transferable spatiotemporal representations from natural script knowledge. In CVPR, pages 23079–23089, 2023.
- [35] Ziyun Zeng, Yixiao Ge, Zhan Tong, Xihui Liu, Shu-Tao Xia, and Ying Shan. Tvtsv2: Learning out-of-thebox spatiotemporal visual representations at scale. arXiv preprint arXiv:2305.14173, 2023.

- [36] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, et al. Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292, 2023.
- [37] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS, 2022.
- [38] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In CVPR, 2023.
- [39] Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, and Lijuan Wang. Lavender: Unifying video-language understanding as masked language modeling. arXiv preprint arXiv:2206.07160, 2022.
- [40] Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. All in one: Exploring unified video-language pre-training. arXiv preprint arXiv:2203.07303, 2022.
- [41] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. Violet: Endto-end video-language transformers with masked visual-token modeling. arXiv preprint arXiv:2111.12681, 2021.
- [42] Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audio-language omni-perception pretraining model and dataset. arXiv preprint arXiv:2304.08345, 2023.
- [43] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. mplug-2: A modularized multi-modal foundation model across text, image and video. arXiv preprint arXiv:2302.00402, 2023.
- [44] Xingjian He, Sihan Chen, Fan Ma, Zhicheng Huang, Xiaojie Jin, Zikang Liu, Dongmei Fu, Yi Yang, Jing Liu, and Jiashi Feng. Vlab: Enhancing video language pre-training by feature adapting and blending. arXiv preprint arXiv:2305.13167, 2023.
- [45] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, pages 5288–5296, 2016.
- [46] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, pages 5803–5812, 2017.
- [47] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. IJCV, 123:94–120, 2017.
- [48] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In AAAI, 2018.
- [49] Ramon Sanabria, Ozan Caglayan, Shruti Palaskar, Desmond Elliott, Loïc Barrault, Lucia Specia, and Florian Metze. How2: a large-scale dataset for multimodal language understanding. arXiv preprint arXiv:1811.00347, 2018.
- [50] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In ICCV, pages 706–715, 2017.
- [51] Jonathan C Stroud, Zhichao Lu, Chen Sun, Jia Deng, Rahul Sukthankar, Cordelia Schmid, and David A Ross. Learning video representations from textual web supervision. arXiv preprint arXiv:2007.14937, 2020.
- [52] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A largescale video benchmark for human activity understanding. In 2015 IEEE conference on computer vision and pattern recognition (CVPR), pages 961–970. IEEE, 2015.
- [53] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.
- [54] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In ICCV, pages 5842–5850, 2017.

- [55] Farzaneh Mahdisoltani, Guillaume Berger, Waseem Gharbieh, David Fleet, and Roland Memisevic. On the effectiveness of task granularity for transfer learning. arXiv preprint arXiv:1804.09235, 2018.
- [56] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.
- [57] Sijie Song, Xudong Lin, Jiaying Liu, Zongming Guo, and Shih-Fu Chang. Co-grounding networks with semantic attention for referring expression comprehension in videos. In CVPR, June 2021.
- [58] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Tubedetr: Spatio-temporal video grounding with transformers. In CVPR, pages 16442–16453, 2022.
- [59] Zhenyang Li, Ran Tao, Efstratios Gavves, Cees G. M. Snoek, and Arnold W. M. Smeulders. Tracking by natural language specification. CVPR, 2017.
- [60] Jie Lei, Tamara L Berg, and Mohit Bansal. Revealing single frame bias for video-and-language learning. arXiv preprint arXiv:2206.03428, 2022.
- [61] Xinyu Huang, Youcai Zhang, Jinyu Ma, Weiwei Tian, Rui Feng, Yuejie Zhang, Yaqian Li, Yandong Guo, and Lei Zhang. Tag2text: Guiding vision-language model via image tagging. arXiv preprint arXiv:2303.05657, 2023.
- [62] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 21(1):5485–5551, 2020.
- [63] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.
- [64] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [65] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023.
- [66] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. arXiv preprint arXiv:2304.06939, 2023.
- [67] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021.
- [68] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- [69] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
- [70] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. NeurIPS, 35:16344–16359, 2022.
- [71] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023.
- [72] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.
- [73] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 2022.
- [74] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3202–3212, 2015.

- [75] David L Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1, pages 190–200. Association for Computational Linguistics, 2011.
- [76] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.
- [77] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, pages 22563–22575, 2023.
- [78] Salman Khan Muhammad Maaz, Hanoona Rasheed and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. ArXiv 2306.05424, 2023.
- [79] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023.
- [80] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022.
- [81] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In CVPR, pages 10209–10218, 2023.
- [82] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and time-sensitive transformer. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XVII, pages 102–118. Springer, 2022.

