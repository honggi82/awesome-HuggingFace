## 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining

[Figure 1]

# arXiv:2501.00958v4[cs.CV]13May2025

Wenqi Zhang1* Hang Zhang2 Xin Li2,† Jiashuo Sun2 Yongliang Shen1 Weiming Lu1,† Deli Zhao2 Yueting Zhuang1 Lidong Bing2

1College of Computer Science and Technology, Zhejiang University 2DAMO Academy, Alibaba Group

zhangwenqi@zju.edu.cn Project: https://multimodal-interleaved-textbook.github.io/

#### Abstract

Compared to image-text pair data, interleaved corpora enable Vision-Language Models (VLMs) to understand the world more naturally like humans. However, such existing datasets are crawled from webpage, facing challenges like low knowledge density, loose image-text relations, and poor logical coherence between images. On the other hand, the internet hosts vast instructional videos (e.g., online geometry courses) that are widely used by humans to learn foundational subjects, yet these valuable resources remain underexplored in VLM training. In this paper, we introduce a high-quality multimodal textbook corpus with richer foundational knowledge for VLM pretraining. It collects over 2.5 years of instructional videos, totaling 22,000 class hours. We first use an LLM-proposed taxonomy to systematically gather instructional videos. Then we progressively extract and refine visual (keyframes), audio (ASR), and textual knowledge (OCR) from the videos, and organize as an image-text interleaved corpus based on temporal order. Compared to its counterparts, our video-centric textbook offers more coherent context, richer knowledge, and better image-text alignment. Experiments demonstrate its superb pretraining performance, particularly in knowledge- and reasoning-intensive tasks like ScienceQA and MathVista. Moreover, VLMs pre-trained on our textbook exhibit outstanding interleaved context awareness, leveraging visual and textual cues in their few-shot context for task solving 1.

*This work was conducted when Wenqi Zhang was interning at Alibaba

DAMO Academy. †Corresponding author. 1Our code are available at https://github.com/DAMO-NLP-

SG/multimodal_textbook

#### 1. Introduction

Vision-Language Models (VLMs) have demonstrated impressive development recently, delivering exceptional performance across a variety of visual tasks, including image captioning, dialogue, and visual question answering [3, 5, 10, 16, 20, 24, 31–33, 44, 45, 49, 58, 59]. These advancements can be primarily attributed to the swift improvements of large language models (LLMs) and the community’s ongoing creation of diverse, high-quality multimodal training corpora [6, 8, 9, 18, 19, 42], collectively driving VLMs forward. A multimodal corpus typically consists of numerous image-text pairs to align images with textual descriptions. Pretraining on such paired datasets allows LLMs to be efficiently adapted into VLMs, with the ability to perceive and interpret visual information.

Beyond image-text pair data, previous researchers have also introduced image-text interleaved corpus as a more natural and flexible multimodal corpus [4, 22, 25, 36, 60]. These corpora, consisting of sequences of text paragraphs interspersed with images, are typically crawled from webpage and document, such as Common Crawl. Pretraining on a combination of interleaved corpus and image-pair datasets enables VLMs to handle interwoven multi-modal inputs, while also unlocking advanced capabilities such as in-context learning [28] and multi-image comparison [19].

Despite their benefits to multi-modal pre-training, existing interleaved datasets still suffer from the following issues (shown in Fig. 1): (1) Loose text-image relation: The associations between images and text in a webpage are often loose and may even include irrelevant images, e.g., logos or advertisements. (2) Lack of logical coherence in image sequences: most webpages contain relatively few images, and more importantly, the logical relations between images are often vague, making it difficult to learn complex visual

###### Previous Interleaved Datasets

###### Our Multimodal Textbook

[Figure 2]

Massive Instructional Videos

[Figure 3]

[Figure 4]

[Figure 5]

###### High-quality Corpus

Image-text relation is loose

Keyframe ASR & OCR

- • More closer Image-text relation
- • Rich visual and textual knowledge
- • More coherent image sequences

If ‘eclectic’ to you is when Green Day change their guitar tone or McDonald’s puts two burgers in one bun, then steer clear of this album. If however you take your pepperoni pizza with extra cream and can stomach the idea of an album with something other than one song reworked ten times, then you should buy Stimmung now.” (so says some English writer type, and he ought to know…)

Multi-Level

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

22000 Class Hours Extraction & Filtering 2.5 Years Duration

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Textbook-Level interleaved Dataset

###### 《Textbook: Mathematics》 《 Textbook: Physics 》 《Textbook: Earth Science》

[Figure 16]

- Video 1:58

- Video 2:13

Tutorial Text Extract From Video: So, the velocity is simply the distance divided by the time. How far did you go, and how long did it take? If you divide those two quantities, you get what’s called velocity …….

Tutorial Text Extract From Video: The next term in Geometry is complementary angles. So, what are Complementary Angles? Complementary Angles are two angles whose measures add up to 90…..

[Figure 17]

[Figure 18]

Lacking connection between images

Video 9:21

[Figure 19]

Video 16:19

[Figure 20]

Tutorial Text Extract From Video: The Appalachian Mountains in eastern North America contain limestones that are composed of the shells of marine animals.

[Figure 21]

On the left-hand side, we’ll have velocity multiplied by time, and on the right-hand side, we’ll be left with just the distance. I’m traveling at 45 miles per hour, and I’m going down the road for two hours, how far will I have gone? It’s clear 90 miles

The Firearm Licensing and Registration Act would establish licensing requirements to posses a firearm and ammunition, including a psychological evaluation and insurance policy. Individuals hospitalized with a mental illness would be denied a license. File photo\n\nOCEANSIDE ……

Let’s consider a right triangle, and we will label it as triangle ABC. The symbol for this triangle is as follows: triangle ABC

###### ...

[Figure 22]

Video 9:42

[Figure 23]

- Video 17:53

- Video 18:18

[Figure 24]

[Figure 25]

These animals lived in a shallow ocean more than 400 million years ago. Around 300 million years ago…..

###### 《Textbook: Engineering 》

angle A measures 40 degrees and angle C measures 50 degrees. In this case, we can say that angle A and angle C are complementary, because the sum of their measures equals 90 degrees

Now, if I throw the ball at an angle, or we could simply say it’s 25 degrees from the ground. What happens? You all know that it will rise to a maximum height, then come back down and eventually hit the ground.

[Figure 26]

[Figure 27]

###### Video 0:11

0:27

###### Low Knowledge Density

Dedicated to mince, peel and cut with delicacy, the slicing knives are precision tools that you have to choose with care

[Figure 28]

Video 9:54

[Figure 29]

The high zirconium oxide content of the ceramic blade of these TB knives makes it a premium tool.

[Figure 30]

[Figure 31]

Tutorial Text Extract From Video: I’m using a suspension bridge as an example. Let’s first discuss how it works.

The parts of a suspension bridge include towers, anchors, main cables, hangers..

[Figure 32]

0:40

[Figure 33]

With optimum durability and everlasting sharp edge that hardly ever need sharpening, the ceramic blade of these slicing knives signed TarreriasBonjean is as efficient as resistant.

[Figure 34]

1:13

So, if I throw the ball at 39 meters per second at a certain angle, .., I want to know how high the ball goes, and how far it travels horizontally. I’m also likely interested in how long the ball stays in the air

So, the fundamental concept behind Complementary Angles is that the measure of angle A plus the measure of angle C is equal to 90 degrees.

- Figure 1. Previous interleaved datasets, e.g., MMC4 and OBELICS, suffer from limitations like weak text-image relations, low knowledge density, and incoherent image sequences. Our multimodal textbook, sourced from massive tutorial videos, employs coarse-to-fine knowledge extraction and multi-level filtering to create a high-quality, textbook-level dataset. It interleaves video keyframes with tutorial texts (extracted from ASR and OCR), enabling VLMs to acquire rich knowledge through tightly coupled text-image and more coherent logic.

reasoning. (3) Low knowledge density: crawled webpages inevitably include content such as news, entertainment, and advertisement recommendations, with little involvement of fundamental knowledge. These issues may severely affect the learning effectiveness of interleaved corpora. Therefore, exploring how to extract high-quality, textbook-level interleaved datasets from vast internet data is quite necessary.

class hours, covering six fundamental subjects, including mathematics, physics, and others. The whole corpus is presented in an image-text interleaved format, where the text and images are more closely aligned, and the logical relations between images are also more coherent.

To create our textbook, we develop an LLM-powered pipeline to systematically collect a vast array of instructional videos from the internet. To achieve automation, we prompt LLMs to construct a knowledge taxonomy covering six subjects and 3900 knowledge points. Then based on this, we gather relevant instructional videos. After that, we design a multi-level, coarse-to-fine knowledge extraction and data filtering pipeline for these collected videos. From a visual perspective, we extract keyframes and recognition text, symbols, and formulas (OCR). From an auditory perspective, we perform automatic speech recognition (ASR) on the instructor’s verbal explanations and refine their quality. Finally, the keyframes and tutorial text are organized into an interleaved format, sequenced chronologically.

On the other hand, the internet contains a vast array of instructional videos [15, 37, 40, 56], e.g., online mathematics courses on YouTube, where people often turn to acquire both foundational knowledge and specialized skills. Most videos contain frame-by-frame demonstrations along with detailed verbal explanations by the instructor, making them an ideal source of training data. However, these valuable resources have received limited attention for VLM training. Besides, Microsoft’s Phi-series models [1, 2, 14, 17, 26] have also demonstrated that high-quality textbook-level datasets are critical for LLM training.

In this paper, we introduce a multimodal Textbook: a high-quality pre-training corpus that encompasses a wealth of foundational knowledge. Our textbook is constructed from 2.5 years of instructional videos, amounting to 22,000

Our textbook is an openly accessible pre-training dataset with high-quality 6.5 million images interleaving with 0.75 billion texts. It drawn from 75,000 extensive instructional

videos, totoaling over 22000 class hours, covering multiple core subjects such as mathematics, physics, chemistry. As demonstrated in Fig. 1, our textbook (the first example) presents three keyframes interleaved with four tutorial texts to dynamically illustrate the geometric concept of complementary angles. These more coherent interleaved context and better-aligned image-text sequences enable VLMs to better grasp foundational knowledge during the pretraining.

Experiments show that VLMs pre-trained on our textbook achieve noticeable improvement on knowledge- and reasoning-intensive benchmarks, like MathVista, and ScienceQA. Besides, we also observe some intriguing findings: our textbook can significantly enhance the interleaved context awareness of VLMs, i.e., pretrained on our textbook, VLMs can more effectively attend to their few-shot context, leveraging visual or textual cues for question-solving. In contrast, the VLMs training on other datasets often overlooked their interleaved context.

#### 2. Related Works

##### 2.1. Vision Language Models

With the development of large language models (LLMs) [38, 46, 51], VLMs have evolved from these task-specific, closed-set models [23, 39] to more flexible systems capable of handling open-world scenarios. Large VLMs adopt a general paradigm of mapping pretrained visual encoder outputs to the embedding space of LLMs, enabling cross-modal understanding [24, 32]. By leveraging large-scale caption datasets [41, 47] and meticulously crafted instruction-following data [12, 32], these models exhibit remarkable capabilities. Building on this foundation, researchers have further boosted VLM performance by diversifying instruction data [51, 54], refining data quality [13, 28], and increasing image resolution [10, 53]. These improvements have led to breakthroughs across OCR, VQA, and visual grounding tasks, with VLMs now achieving impressive results on benchmarks that demand precise, context-aware understanding [10, 28, 31, 55].

##### 2.2. Multi-modal Pretraining Data

Recent developments in Vision-Language Models have typically involved a two-stage process: pretraining followed by a high-quality instruction-following phase [7, 10, 11, 29, 30, 49, 53, 57]. Most VLMs utilize paired image-caption datasets [41, 42, 47] for pretraining which facilitate a quick alignment between image and text spaces [10, 29, 53]. However, image-caption datasets lack the naturalness and authenticity found in more comprehensive text corpora used for LLMs, as they are often limited in diversity and complexity [28]. This limitation reduces VLMs’ capacity for in-context learning and chain-of-thought (CoT) reasoning. Recognizing this gap, some researchers have introduced

webpage-centric interleaved datasets, like MMC4 [60] and OBELICS [22], sourced from webpages and documents [4, 5]. These interleaved datasets can enhance in-context learning capabilities in VLMs [28, 48]. However, these datasets still face issues such as low image-text relevance, poor sequence logic, and sparse knowledge density. Our work proposes a multimodal “textbook” corpus curated from instructional videos, intending to enhance multimodal pretraining and expand the model’s ability to handle interleaved visual and textual inputs.

#### 3. Curation of Multimodal Textbook

Our goal is to construct a textbook-level interleaved corpus that delivers high-quality, specialized knowledge for pretraining VLMs in a more natural and efficient manner. To achieve this, we choose online instructional videos as the primary data source. Compared to common videos, such as entertainment, sports, or TV-show, instructional videos exhibit greater textual-visual consistency and sequential frame coherence, making them ideal for creating a “multimodal textbook”. While these videos are generally reliable, they still contain significant noise and redundancy, such as unrelated segments (e.g., advertisements), mismatches between visual content and text (e.g., almost “static” scene predominantly featuring a single lecturer), or redundant scenes. To address this, we employ a multi-level pipeline (video-level, clip-level, and keyframe-level) with a coarse-to-fine strategy. The curation process is outlined in Fig. 2.

##### 3.1. Collecting Instructional Videos

LLM-proposed Knowledge Taxonomy. In this work, we propose a knowledge taxonomy with four hierarchical layers for the desired instructional videos, namely Subject → Course → Sub-course → Knowledge Point. To guarantee a broad coverage of instructional videos, we instruct an LLM to span the proposed knowledge taxonomy so that multiple educational stages (from primary school to middle school) and diverse subjects (mathematics, physics, etc.) will be involved. Eventually, as shown in Sec. 7.4, we obtain a knowledge taxonomy comprising 6 subjects (mathematics, physics, chemistry, earth science, engineering, and computer science), 55 courses (Algebra, Solid Geometry,..), and 3915 knowledge points. For example in the mathematics: Mathematics → Elementary Mathematics → Rational and Irrational Numbers → the definition of Irrational Numbers.

Taxonomy-based Video Collection and Filtering. Each knowledge point in the taxonomy is then used as a keyword to retrieve relevant instructional videos via YouTube’s search API2. We retain the top 50 videos for each knowledge point. Then, we perform deduplication based on video IDs and filter the low-quality videos using their

2https://www.youtube.com/

Collecting Instructional Videos

…..

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Metadata-Level

###### Search Video Metadata

Knowledge Points

LLM • 159,565 instructional videos

Download

- • 6 Subjects: Mathematics, Physics ..
- • 55 Courses: Algebra, Solid Geometry ..
- • 3915 Knowledge Points: How to calculate..

Vid : xxx Ttitle : xxx Desc : xxx Comment:

- • LLM-based
- • Deduplicate
- • Content Filtering

[Figure 39]

[Figure 40]

[Figure 41]

Knowledge Taxonomy

Video-to-Textbook Pipeline

###### Video-Level Clip-Level Keyframe-Level

Multimodal Textbook

[Figure 42]

[Figure 43]

[Figure 44]

Clip-to-Keyframe, OCR

Video-to-ASR Long video-to-Short Clip

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Extracting Extracting

Extracting

- • Extract Audio
- • Transcribe into text
- • ASR Refining

- • Merge incomplete ASR Segments
- • Split long video based on ASR’s timestamps

- • Detecting inter-frame changes using the SSIM
- • Extracting text, symbols, and formulas using OCR.

Filtering

- • Rule-based filtering
- • Scoring ASR by LLM
- • Discard noninstructional videos using ASR score.

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

ASR Video Clip

- • 75,000 instructional videos
- • 2.5 years video duration (22697 hours)
- • 4M video clips and 6.5M keyframes
- • 259M ASR tokens and 500M OCR tokens
- • 610K image-text interleaved samples

[Figure 70]

###### Filtering

###### Filtering

- • Caption each video clip
- • Calculate the similarity between clip’s caption and ASR
- • Discard clip unrelated to ASR, e.g., only showing speaker.

- • Discard keyframes with object occlusion.
- • Remove OCR that are identical to previous frames or lack useful info.

- Figure 2. An illustration of constructing a multimodal textbook from instructional videos. We first instruct LLMs to construct a knowledge taxonomy, then retrieve and filter videos at metadata level, collecting 159K instructional videos. Then a video-to-textbook pipeline is designed for multi-level knowledge extraction. ① We filter out non-instructional videos using ASR transcripts, retaining 75K high-quality videos. ② We use ASR’s timestamp to segment long videos into short clips, discarding those with misaligned visuals and ASR. ③ We detect keyframes from each clip and extract text and symbols by OCR. Our pipeline produces 6.5M keyframes, 259M ASR, and 500M OCR tokens and organizes them into an image-text interleaved textbook.

metadata: we prompt LLMs to review each video’s metadata—including the title, description, and comments—to exclude irrelevant, pornographic, or illegal content. Lastly, we collect a total of 159,565 videos from YouTube.

- 3.2. Video-to-Textbook Pipeline

the concrete numbers). Therefore, we further introduce Qwen2-72B-Instruct [51] to rewrite the raw ASR transcriptions, with the purpose of improving their fluency and coherence while not changing the original semantics

Video-Level Filtering: Low-quality Videos based on ASR. We first filter the videos using a set of predefined rules, including non-English videos, videos shorter than 10 seconds, and silent videos with very few ASR text tokens. Next, we assess the remaining videos by instructing an LLM to review their ASR transcriptions and filter out the non-instructional videos in terms of the following criteria:

For an instructional video, both the visual content (e.g., slide or animation) and the auditory content (e.g., instructor’s narration) contain valuable knowledge. Therefore, we design a multi-level extraction pipeline to gather instructional keyframes and text from raw videos, interleaving them into a textbook.

- • Relevance: The ASR represents the tutorial content of the video. We assess the alignment between the ASR and the targeted knowledge point, filtering out irrelevant videos, e.g., advertisements or entertainment videos.
- • Knowledge Density: We evaluate the knowledge involved in ASR, as many videos contain meaningless filler phrases like ”um,” ”the next up is this,” or ”then we get this.” Such videos fail to provide valuable textual knowledge and are therefore discarded.
- • Transcription Quality: We examine the transcription quality by whisper, excluding repetitive or erroneous ASR text. This step occurs before ASR rewriting.

Video-Level Extraction: Video-to-ASR. We employ FFmpeg3 to extract the audio from each video (videoto-audio) and then transcribe it into text (audio-to-text, ASR) using whisper-large-v34. These transcriptions contain substantial knowledge and reasoning details, such as the instructor’s explanations of on-screen content and step-by-step derivations of specific mathematical concepts. However, due to the nature of tutorial speech where the instructors prefer to use colloquial expressions to explain a concept, the perplexities (PPLs) of the raw ASR transcriptions are usually much higher than those of the texts from standard corpora (see Tab. 6 for

After LLM evaluation across these three dimensions, the retained 75,000 videos are generally of high quality, as verified by their ASR transcriptions.

- 3https://www.ffmpeg.org/
- 4https://huggingface.co/openai/whisper-large-v3

#Image #Text Token In-sample Image SIML ↑ Source

Dataset

Min. Max. Avg. Min. Max. Avg. L=4 L=5 L=6 L=7 L=8 Avg. Common Crawl Image-text Paired Dataset

COYO-700M 1 1 1 1 811 16 - - - - - - Common Crawl LAION-5B 1 1 1 6 683 27 - - - - - - Common Crawl Image-text Interleaved Dataset

MMC4 0 117 5.7 4 16715 417 0.363 0.348 0.310 0.298 0.276 0.319 Common Crawl MMC4-core-ff 0 15 4.1 15 16715 329 0.431 0.406 0.404 0.403 0.396 0.407 Common Crawl OBELICS 1 30 2.5 12 10717 816 0.366 0.351 0.339 0.337 0.336 0.345 Common Crawl OmniCorpus∗ 1 16 3.9 14 6893 574 0.358 0.329 0.310 0.305 0.301 0.321 Multi-sources Ours 2 45 10.7 11 34174 1297 0.687 0.697 0.698 0.688 0.662 0.686 Video Website

- Table 1. We compare our multimodal textbook with image-text paired datasets and webpage-centric interleaved datasets in terms of image and text distributions. In-sample Image SIML measures the semantic and structural correlation between multiple images within an interleaved sample. OmniCorpus∗: Due to the extensive size of the dataset, we perform statistical analysis on a randomly sampled subset.

Clip-Level Extraction: Long Video-to-Short Clips. To achieve temporal alignment between text and frames, we use the timestamps of each ASR transcription to segment the long video into multiple video clips. However, it is essential to consider that the original ASR transcriptions are often fragmented. First, we merge multiple incomplete ASR segments into a single, semantically coherent paragraph. Then, we use their timestamps to segment the video clips accordingly. Each clip lasts 10 to 20 seconds, accompanying an ASR text segment: ⟨clip1,asr1⟩,⟨clip2,asr2⟩,...,⟨clipn,asrn⟩

Clip-Level Filtering: Video Clips without Visual Knowledge. Previous filtering of long videos is based on ASR text. Next, we also assess each video clip from a visual perspective to determine if it contains sufficient visual knowledge. In most videos, it is inevitable to contain uninformative scenes, such as transitions, shots focused solely on the speaker, or cluttered backgrounds, which are not suitable for a multimodal textbook. A good scene should contain slides, blackboards, or demonstrative animations that introduce a knowledge concept or illustrate specific objects, rather than just the speaker alone. To this end, we employ a VideoLlama2 [12] to generate a detailed caption for each video clip. We then calculate the text similarity between the clip’s caption and ASR transcription using the text embeddings model (gte-Qwen2-7B-instruct [27]), filtering out uninformative video clips.

Notably, even if an uninformative video clip is discarded, its ASR transcription may still contain valuable information. Thus, we retain these transcriptions in our textbook: ⟨clip1,asr1⟩,asr2,asr3,⟨clip4,asr4⟩,...,⟨clipn,asrn⟩

Keyframe-Level Extraction: Clip-to-Keyframes by Comparing Changes between Consecutive Two Frames. Then we need to extract keyframes from each video clip, removing similar or even duplicate shots. A frame is identified as a keyframe if it exhibits significant visual change compared to the previous one. Therefore, we compute the similarity between consecutive frames and filter out those with minimal scene changes.

Considering efficiency and accuracy, we employ the Structural Similarity Index algorithm (SSIM) [50] to compare the similarity between consecutive frames iteratively. Starting from the first frame, we calculate the similarity with the subsequent frame. If the similarity is quite high, we skip to the next until a frame with significant change is found. We then use this frame as a new reference point and continue to seek subsequent frames with notable differences. The detailed process is provided in Algorithm 1. The keyframe-ASR sequence is as follows: ⟨framek

1 ,framek

1 ,asr1⟩,asr2,asr3,⟨framek

###### 4 ,asr4⟩,...

1

2

1

Keyframe-Level Extraction: Keyframe-to-OCR. Last but not least, most instructional videos often use bulletpointed text, formulas, and mathematical symbols to illustrate knowledge points, physical concepts, and calculation processes. These texts, symbols, and mathematical formulas encapsulate substantial knowledge. Therefore, we extract these texts from keyframes as the ASR’s supplement. Specifically, we employ two advanced VLMs (InternVL240B [11]) to perform optical character recognition (OCR) on each keyframe, extracting on-screen text, mathematical symbols, formulas, and other elements.

Keyframe-Level Filtering: Uninformative Keyframe and Redundant OCR. Despite filtering visual content at multiple levels, some keyframes may still contain low informational scenes, e.g., occlusion. Therefore, we also utilize InternVL2 to score each keyframe after conducting OCR. Additionally, we do not retain all OCR texts, as the OCR from consecutive keyframes is likely to be highly similar or even identical. Therefore, we filter out OCR results that are similar to previous ones.

Lastly, as shown in Fig. 2, through our multi-level extracting and filtering, we curate high-quality video keyframes, OCR text, and ASR transcriptions. These elements represent the useful visual content in videos and the instructor’s in-depth explanation of knowledge points. To create the pretraining dataset, we interleave the selected keyframes of a long video with refined ASR and OCR text in chronological order, creating our multimodal textbook:

{framek

1 ,framek

1 ,ocr1,asr1,asr2,asr3,framek

4 ,ocr4,asr4,..}

1

2

1

#### 4. Analysis of Multimodal Textbook

##### 4.1. General statistics

We utilize GPT-4o to synthesize our knowledge taxonomy with 3915 knowledge points across 6 subjects, which enabled us to automatically collect 159K English instructional videos based on this taxonomy. Following our video-totextbook pipeline, we filter 53% low-quality or repetitive videos and retain 75K videos (22,697 class hours) with an average duration of 18 minutes. Then we extract 6.5M keyframes and 0.75B text (ASR+OCR) tokens from these videos. To enhance training efficiency, we concatenate multiple ⟨framek

i ,..,framek

i ,ocri,asri⟩ fragment into a single sample, producing a total of 610K interleaved samples. Each sample contains an average of 10.7 keyframes and 1,297 text tokens. The detailed statistics for each subject are shown in Appendix (Tab. 7). Besides, we randomly select 100 videos and corresponding samples for manual evaluation, with detailed results presented in Sec. 7.2.

n

1

##### 4.2. Comparison with Existing Datasets

Image and Text Distribution. To better demonstrate the advantages of our video-centric dataset, we compare our multimodal textbook with existing datasets (image-text paired datasets and webpage-centric datasets), focusing on the distribution of images and tokens across these datasets. As shown in Tab. 1, we observe that our dataset exceeds previous datasets in terms of the average number of images and text tokens. For instance, our dataset contains an average of 10.7 images per sample, compared to only 5.7 in MMC4 and 4.1 in OBELICS.

Images within a Sample are More Closely Related. A notable feature of our video-centric design is the inherent association between multiple images within a sample, providing a dynamic illustration of mathematical concepts or physical phenomena. To validate this, we design an insample image similarity metric (InSI-SIM). It measures the similarity between all images within a sample, i.e., calculating the average of all pairwise similarity of a sample. For similarity, we consider both semantic (CLIP score) and structural similarity (SSIM score) respectively. The detailed formula is presented in Sec. 7.5.

As shown in Tab. 1, we report InSI-SIM for 8 imagesubset (i.e., the subset containing 4 images) to 8 imagesubset (L: 4 to 8). For all subsets, our multimodal textbook achieves a significantly higher InSI-SIM score than other datasets, nearly more than double. For example, our textbook scores 0.686 on average, while OBELICS reaches only 0.345. Besides, we also observed that, as the number of images per sample increases, the InSI-SIM of our dataset remains stable at around 0.68, whereas other datasets expe-

rience a noticeable decline (about ↓ 10%). This further validates that our video-centric dataset provides more coherent and contextually related images.

#### 5. Experiments

##### 5.1. Experimental Settings

Baselines. We first employ LLaVA-1.5-7B [31] as base models to study the pretraining performance on our dataset and reference datasets (MMC4, OBELICS). For LLaVA1.5-7B, we apply continual pretraining on its pre-trained model (aligned using 558K paired data). To investigate our dataset more comprehensively, we also pre-train Idefics28B model [21] on our dataset, which is an advanced VLM that already supports multi-image and interleaved format input. For the Idefics2-8B, we design two pretraining settings: 1. Training from scratch using the architecture of Idefics2-8B (i.e., Idefics2-8B with randomly initialized projector) and 2. Continual pretraining from the Idefics2-8Bbase which is already pre-trained on OBELICS. For a fair comparison, we sample an equivalent number of samples (610K) from MMC4 and OBELICS and apply the same training parameters across all datasets.

Evaluation Methods. Following OpenFlamingo [5] and OmniCorpus [25], we evaluate the performance of the pre-trained models on two VQA benchmarks (TextVQA [43], OKVQA [35]), three visual reasoning benchmarks (MathVista, MathVision, MathVision), and ScienceQA-IMG [34], covering general, OCR, mathematics, and science domains. We compute model accuracy in few-shot settings using either randomly sampled or retrieved examples as previous works [21, 25, 52].

##### 5.2. Main Results

As shown in Tabs. 2 and 3, after being pretrained on our Textbook-6.5M, both LLaVA-1.5 and Idefics-8B exhibit significant improvements across seven benchmarks, achieving average gains of +3.2%, +8.3%, +4.0%, and +4.6% in the 0-shot to 4-shot settings, respectively. Notably, even for cutting-edge VLMs like Idefics2, our multimodal textbook brings an additional improvement of +1.4%, underscoring rich knowledge content and its high data quality.

Our Textbook Brings Improvement on Knowledgeoriented and Reasoning Benchmarks. In Tab. 2, we observe that our textbook dataset delivers notably greater improvements on knowledge-oriented and reasoning-related benchmarks compared to counterpart datasets. For instance, on ScienceQA, our dataset achieves over a 20% improvement in both zero-shot and few-shot settings compared to MMC4. Similarly, on math-related benchmarks such as MathVista, which require both mathematical knowledge and visual reasoning capabilities, our dataset demonstrates an average improvement of +5.3% and +6.4% compared to

#Shot 0 1 2 4 0 1 2 4 0 1 2 4 0 1 2 4

|Dataset<br><br>|ScienceQAIMG OKVQA TextVQA TextVQAocr| | | |
|---|---|---|---|---|
|MMC4 MMC4-Core-ff OBELICS Textbook-6.5M<br><br>|- 1.6 3.9 11.6<br>- 2.1 10.1 10.2<br>- 2.8 3.0 16.4<br><br><br>26.3 29.4 25.1 37.3<br><br>|8.6 23.6 21.5 28.7 11.8 21.2 25.3 30.4 13.0 31.7 35.7 37.5 10.2 31.2 36.8 39.9<br><br>|12.1 16.2 16.8 20.9<br><br>13.6 18.7 18.8 22.1 9.2 26.5 30.2 32.2<br><br><br>11.8 26.7 32.1 33.5<br><br>|14.5 23.9 29.9 34.7 16.1 26.6 28.7 33.1 11 30.7 36.3 41 14.1 33.1 36.4 42.8<br><br>|
|Dataset|MathVista MathVision MathVerse Avg.| | | |
|MMC4 MMC4-Core-ff OBELICS Textbook-6.5M<br><br>|20.4 30 27.9 26 22.5 33.0 29.2 27.8<br><br>21.6 28.5 31.1 27.6 24.3 43.4 33.2 29.2<br><br><br>|12.2 21.3 15.5 16.1<br>13.7 23.4 16.3 17.7<br><br><br>13.4 20.1 16.8 14.9<br><br>14.5 25.6 18.2 18.1<br><br><br>|8.6 19.4 21.2 15.9 8.6 19.9 21.8 15.2<br><br>6.9 19.4 20.7 14<br><br>7.7 28.5 19.8 14.6<br><br><br>|10.9 19.4 19.5 21.9 12.3 20.7 21.4 22.3 10.7 22.8 24.8 26.2 15.5 31.1 28.8 30.8<br><br>|

- Table 2. We continued pre-training the base model of LLaVA-1.5-7B using different interleaved datasets. The results are evaluated on 4 common VQA and 3 math-related benchmarks under few-shot settings.

Continual Pre-training from Idefics2-8B-base Pre-training Idefics2-8B from scratch Dataset OKVQA TextVQA MathVista MathVison MathVerse OKVQA TextVQA MathVista MathVison MathVerse

MMC4-cf 54.1 57.7 27.8 14.0 17.3 9.4 25.1 24 13.3 18.3 OBELICS 54.6 57.5 27.6 14.3 17.5 10.5 25.7 24.2 13.6 17.7 Textbook-6.5M 55.1 58.2 29.7 16.2 19.4 10.1 26.8 26.1 14.4 19.8

- Table 3. Except for LLaVA, we also pre-train advanced VLMs with multi-image ability (Idefics): continual pretraining from Idefics-8Bbase or pre-training from scratch. The evaluations are extended to an 8-shot using randomly selected examples as previous works [21].

OBELICS. This improvement highlights the high quality of our textbook, which distills extensive knowledge from instructional videos into an interleaved textbook.

Coherent Video Frame Interleaving with ASR Enhance the In-context learning capabilities. We observe an interesting phenomenon: even on general-domain benchmarks such as OKVQA and TextVQA, our textbook dataset yields modest improvements in few-shot settings. Specifically, as shown in Tab. 2, in the zero-shot scenario, our textbook lags behind OBELICS by 2.8%; however, in the 1-shot setting, performance becomes comparable. Notably, in the 2-shot and 4-shot settings, our dataset surpasses OBELICS with improvements of +1.1% and +2.4%, respectively. A similar trend can also be observed on the TextVQA. This can be attributed to our video-centric interleaved design, which provides more coherent context and enhances the incontext learning capabilities of VLMs.

##### 5.3. Analysis

Whether VLMs Can Truly Attend to their Interleaved Context? To better investigate why our textbook enhances few-shot performance, we design a “Cheat Test”: We replace one of the few-shot examples with the test sample itself and then observe whether the VLMs can notice this “cheat shortcut”. A VLM with strong in-context ability would recognize that its context already contains an identical question and answer, thereby answering the question effortlessly. Therefore, we design a 1-shot and 2-shot “cheat test”. For the 1-shot “cheat test”, the prompt contains only one example ({It, qt, at}) that is identical to the test sample ({It, qt}). In 2-shot “cheat test”, it includes two examples in the prompt: one identical example ({It, qt, at}) and one random example ({It, qt, at}). This setup allows us to ob-

Dataset OKVQA TextVQA Mathvista Mathvision Mathverse

- 1-shot Cheat: Example:{It,qt,at} + Test-case: It,qt MMC4-cf 69.0 41.0 72.6 69.3 55.7 OBELICS 71.5 43.8 67.7 66.5 62.8 Ours 79.2 51.9 94.1 98.4 76.8

- 2-shot Cheat: Example:{It,qt,at}, {Ie,qe,ae}+Test-case: It,qt MMC4-Cf 53.5 39.2 55.7 51.9 40.8 OBELICS 71.3 42.8 56.7 39.9 39.5 Ours 84.3 49.4 77.1 70.7 63.1

Table 4. We design “Cheat Test” to observe whether VLMs can attend to their interleaved context. We replace a few-shot example with the test sample itself and observe whether VLM notice this identical <image,question,answer> within their prompt. It, qt, at denote the test case, Ie, qe, ae denote a random selected example.

serve whether the VLMs can allocate sufficient attention to their image-text interleaved context and identify relevant information for question answering.

As shown in Tab. 4, in both 1-shot and 2-shot scenarios, our dataset significantly outperforms MMC4 and OBELICS by nearly 20%, particularly on MathVista and MathVision, where we nearly reach 100% in the 1-shot setting, while MMC4 achieves only 72.6% and 69.3%, respectively. Furthermore, from the 1-shot cheat to the 2-shot, the difficulty of cheating increasesas as the context lengthens. As a result, we observe significant performance drops for OBELICS and MMC4 from 1-shot to 2-shot cheating scenarios. However, our textbook dataset only exhibits a smaller drop on most benchmarks and even shows an improvement in OKVQA from 79.2 (1-shot) to 84.3 (2-shot). These results show that VLMs pre-trained with our multimodal textbook can more effectively allocate attention to their interleaved context and capture useful information from longer contexts.

###### The Influence of Disrupting the Image’s Order. As

[Figure 71]

Figure 3. We randomly select 20%, 50%, and 100% samples from datasets and shuffle the image order within each sample. These datasets with shuffled images are also used for pretraining. The Accuracy denotes the average of seven benchmarks.

Pretraining Continual Pretraining SFT OKVQA MathVista

✓ − ✓ 61.1 23.2

✓ MMC4-Core-ff ✓ 61.5 ↑0.4 24.8 ↑1.6 ✓ OBELICS ✓ 61.8 ↑0.7 25.6 ↑2.4 ✓ Textbook-6.5M ✓ 62.2 ↑1.1 28.7 ↑5.5

Table 5. We also evaluated the zero-shot result after instruction fine-tuning using the 665K data from LLaVA-1.5.

previously noted, compared to webpage-centric datasets, the video-centric design offers a more coherent image sequence along with a frame-by-frame text explanatory, presented in an interleaved image-text format. To verify this, we shuffle the image order of interleaved datasets and then also use it for pre-training. For each dataset, we randomly select 20%, 50%, and 100% of the samples and then shuffle the order of images within each sample.

As shown in Fig. 3, whether shuffled at 20%, 50%, or even 100%, the shuffled MMC4 appears largely unaffected. OBELICS exhibits a moderate decline. In contrast, our multimodal textbook shows a significant performance drop, which becomes increasingly severe as the shuffling ratio increases. These observations confirm our motivation that there is no strong sequential dependency between images in these webpage-centric datasets. However, these coherent images and tightly aligned image-text are beneficial, enabling VLMs to effectively learn complex knowledge and the underlying reasoning logic.

The Performance after Instruction Turning. Except for analyzing the pre-training performance, we also report the SFT performance after instruction tuning on LLaVA665K corpus. All training parameters remain the same for OBELICS, MMC4 and our textbook. As shown in Tab. 5, on Mathvista, our textbook elevates the performance of the original LLaVA-1.5 from 23.1 to 28.7, achieving an improvement twice (+5.5%) that of OBELICS (+2.4%) and three times that of MMC4-Core-ff (+1.6%). The results of other benchmarks are similar. These results demonstrate that the knowledge learned during pretraining on our multi-

Dataset Perplexity ↓ 1-shot Acc.

MMC4-Core-ff 12.56 20.7 OBELICS 11.27 22.8 Ours (ASR Refine, OCR, SSIM) 13.92 31.1

- - w/o ASR Refine 16.86 26.2 (↓4.9)
- - w/o OCR 12.7 28.8 ((↓2.3) Keyframe Extraction algorithms #Keyframe 1-shot Acc.

- - SSIM→ Pixel-level extractor 6.5M→ 18M 22.1 (↓9)
- - SSIM→ CLIP-based extractor 6.5M→1.7M 24.6 (↓6.5)

Table 6. We perform an ablation study on video-to-textbook pipeline, including the impact of ASR refinement, the necessity of incorporating OCR, and the algorithms for extracting keyframes.

modal textbook can transfer to instruction fine-tuning stage, leading to positive outcomes for downstream tasks.

##### 5.4. Ablation of Video-to-Textbook’s Design

In Sec. 3.2, we detail the process of our video-to-textbook pipeline, including multi-level extraction and filtering. In this section, we delve into the impact of these designs.

Raw ASR Text Impairs the Language Ability. In our pipeline, we instruct an LLM to refine the transcribed ASR text. As demonstrated in Tab. 6 (w/o ASR refine), using raw ASR text results in an average performance drop of 4.9% across 7 benchmarks. We calculated the perplexity (PPL) of the raw ASR text and found it significantly higher than other corpora (16.8 Vs. 11.2). This is primarily due to the colloquial characteristics of the video-transcribed ASR, which is often relatively brief, incomplete, and contains a high frequency of meaningless conjunctions. Training directly on such text may impair the model’s language abilities. In contrast, refined ASR has a lower PPL (13.9) and more closely aligns with standard training corpora.

Integrating OCR Provides Additional Benefits. We also analyzed the impact of integrating OCR into our pipeline. The results indicate that OCR provides additional improvements (+2.3%), particularly in benchmarks such as TextVQA and MathVista. Similar to humans taking notes during lectures, OCR extracts textual knowledge points, formulas, and mathematical symbols from the videos, thereby enhancing the model’s domain-specific expertise. However, we also observed that low-quality OCR can introduce noise and even significantly degrade performance. Therefore, selecting reliable external tools to extract high-quality OCR is crucial.

How to Extract Keyframe? We detect keyframes from video clips using frame-to-frame differences, exploring pixel-level methods (e.g., OpenCV absdiff), structural algorithms (SSIM), and semantic models (CLIP-ViT-L), with results detailed in Tab. 6. We observed that in these instructional videos, which primarily feature abstract diagrams or geometric images, the pixel-level method often extracts an excessive number of keyframes (18M), resulting in a 9% drop in training performance. Conversely, the semanticlevel model may struggle to distinguish between these ge-

ometric images on a semantic level, frequently treating them as similar and consequently missing many critical keyframes (only 1.7M). Therefore, we ultimately adopted SSIM for keyframe extraction, which yielded noticeably better training performance than the other two methods.

#### 6. Conclusion

We introduce a multimodal textbook to pre-train VLMs, enabling them to acquire specialized knowledge in a natural and contextual manner. By aggregating online educational videos (e.g., mathematics and physics courses) and transforming them into a frame-ASR interleaved dataset, this textbook provides a coherent and interconnected learning context, complementing traditional image-text alignment methods. Using our pipeline, we curated over 2.5 years of instructional videos (22,000 class hours) into a high-quality dataset with 6.5 million keyframes and 0.75 billion text tokens. Experiments demonstrate its effectiveness, especially in enhancing VLMs’ in-context learning capabilities.

#### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 2
- [2] Marah Abdin, Jyoti Aneja, Harkirat Behl, S´ebastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024. 2
- [3] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 1, 3

- [5] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 1, 3, 6
- [6] Anas Awadalla, Le Xue, Oscar Lo, Manli Shu, Hannah Lee, Etash Kumar Guha, Matt Jordan, Sheng Shen, Mohamed Awadalla, Silvio Savarese, et al. Mint-1t: Scaling opensource multimodal data by 10x: A multimodal dataset with one trillion tokens. arXiv preprint arXiv:2406.11271, 2024. 1
- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren

- Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 3
- [8] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 1
- [9] Wei Chen, Lin Li, Yongqi Yang, Bin Wen, Fan Yang, Tingting Gao, Yu Wu, and Long Chen. Comm: A coherent interleaved image-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2406.10462, 2024. 1
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 1, 3
- [11] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 3, 5
- [12] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 3, 5
- [13] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, et al. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data. arXiv preprint arXiv:2410.18558, 2024. 3
- [14] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio C´esar Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023. 2
- [15] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025. 2
- [16] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36:72096–72109, 2023. 1
- [17] Mojan Javaheripi, S´ebastien Bubeck, Marah Abdin, Jyoti Aneja, Sebastien Bubeck, Caio C´esar Teodoro Mendes, Weizhu Chen, Allie Del Giorno, Ronen Eldan, Sivakanth Gopi, et al. Phi-2: The surprising power of small language models. Microsoft Research Blog, 1(3):3, 2023. 2
- [18] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

2021. 1

- [19] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024. 1
- [20] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions. arXiv preprint arXiv:2408.12637, 2024. 1
- [21] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 6, 7
- [22] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelics: An open webscale filtered dataset of interleaved image-text documents,

2023. 1, 3, 2

- [23] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In Proceedings of the AAAI conference on artificial intelligence, pages 11336– 11344, 2020. 3
- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 1, 3

- [25] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, Guanzhou Chen, Yinan He, Zhangwei Gao, Erfei Cui, et al. Omnicorpus: An unified multimodal corpus of 10 billion-level images interleaved with text. arXiv preprint arXiv:2406.08418, 2024. 1, 6, 2
- [26] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023. 2
- [27] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023. 5
- [28] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023. 1, 3
- [29] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 3
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 3
- [31] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 3, 6
- [32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 3
- [33] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li,

- Hao Yang, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525, 2024. 1
- [34] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521,

2022. 6

- [35] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 6
- [36] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024. 1
- [37] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019. 2
- [38] Ethan Perez, Douwe Kiela, and Kyunghyun Cho. True fewshot learning with language models. Advances in neural information processing systems, 34:11054–11070, 2021. 3
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [40] Ramon Sanabria, Ozan Caglayan, Shruti Palaskar, Desmond Elliott, Lo¨ıc Barrault, Lucia Specia, and Florian Metze. How2: a large-scale dataset for multimodal language understanding. arXiv preprint arXiv:1811.00347, 2018. 2
- [41] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION400M: open dataset of clip-filtered 400 million image-text pairs. CoRR, abs/2111.02114, 2021. 3
- [42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 1, 3
- [43] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 6
- [44] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun

- Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024. 1
- [45] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023. 3
- [47] Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, and Oncel Tuzel. Mobileclip: Fast image-text models through multi-modal reinforced training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15963– 15974, 2024. 3
- [48] Junjie Wang, Yin Zhang, Yatai Ji, Yuxiang Zhang, Chunyang Jiang, Yubo Wang, Kang Zhu, Zekun Wang, Tiezhen Wang, Wenhao Huang, et al. Pin: A knowledge-intensive dataset for paired and interleaved multimodal documents. arXiv preprint arXiv:2406.13923, 2024. 3
- [49] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 3
- [50] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5
- [51] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 3, 4
- [52] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. In Proceedings of the AAAI conference on artificial intelligence, pages 3081–3089, 2022. 6
- [53] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 3
- [54] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840, 2024. 3
- [55] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming

- Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 3
- [56] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16375–16387, 2022. 2
- [57] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 3
- [58] Wenqi Zhang, Mengna Wang, Gangao Liu, Xu Huixin, Yiwei Jiang, Yongliang Shen, Guiyang Hou, Zhe Zheng, Hang Zhang, Xin Li, et al. Embodied-reasoner: Synergizing visual search, reasoning, and action for embodied interactive tasks. arXiv preprint arXiv:2503.21696, 2025. 1
- [59] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1
- [60] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. Advances in Neural Information Processing Systems, 36, 2024. 1, 3, 2

## 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining

[Figure 72]

### Supplementary Material

#### 7. Detail of Video-to-Textbook Pipeline

##### 7.1. Implementation Details

When synthesizing the Knowledge Taxonomy, we utilize GPT-4o to construct the taxonomy. When filtering video at the metadata level, GPT-4o is also employed to review the metadata of the searched videos. During the Video-to-ASR phase, Whisper-large-v3 is used to convert audio into text. Then Qwen2-72B-Instruct is applied to refine the raw ASR transcriptions. In the video-level filtering stage, DeepSeek-V2 and Llama3-70B-Instruct are used to score each ASR transcription, enabling the filtration of low-quality videos. A video is filtered out if both LLMs determine its ASR does not meet the required standards. After splitting long videos into short clips, we first use VideoLlama2-7B to generate a detailed caption for each video clip. Subsequently, we compute the similarity between the clip’s caption and the ASR using GTE-Qwen2-7B-Instruct. Finally, InternVL2 is employed to extract and filter OCR from the keyframe.

##### 7.2. Human Evaluation

We randomly sample 100 examples from the multimodal textbook and conduct a manual quality evaluation, focusing on three key aspects: (1) image quality, (2) the connections between different images in a sample and (3) the relevance between texts and images. After manual inspection, we observe that, aside from chemistry, this batch of samples covers five domains: mathematics (31), physics (16), computer science (16), engineering (25), and earth sciences (12). It contains a total of 1,421 images, including 378 slide-style images, 214 lecture-style images, 414 demonstration animations, and 415 natural scenes. Image analysis reveals that only 7% (72 images) are highly similar, while the remaining images are related to each other but also exhibit clear distinctions. Text-image relevance analysis shows that the attached text (ASR) correctly explains the visual concepts or computational processes presented in the images, with no ambiguity or redundancy.

##### 7.3. Constructing Pretraining Sample

After collecting 6.5M keyframes, and 750M refined ASR, and OCR tokens, we can employ various strategies to construct image-text interleaved samples for pre-training. ① Similar to a webpage-centric dataset, where each webpage is treated as a separate sample, we treat each video as an individual sample. This simple strategy maintains the

Algorithm 1 SSIM-Based Key Frame Extraction Algorithm

Require: Frame sequence {F1, F2, . . . , FN}, similarity thresh-

old T

Ensure: Key frame sequence {K1, K2, . . . }

- 1: K ← {F1} ▷ Initialize key frame sequence with the first frame F1
- 2: reference frame ← F1 ▷ Set the reference frame to F1

- 3: for i = 2 to N do
- 4: SSIM ← CalculateSSIM(reference frame, Fi) ▷ Calculate SSIM between reference frame and frame Fi

- 5: if SSIM < T then
- 6: K ← K ∪ {Fi} ▷ If SSIM is below threshold, add frame Fi as a key frame
- 7: reference frame ← Fi ▷ Update the reference frame to Fi

- 8: end if
- 9: end for
- 10: return K ▷ Return the sequence of key frames

semantic integrity of a video. However, it also leads to overly long contexts for most samples, as each video contains an average of 86 keyframes, far exceeding the maximum context length supported by most VLMs. ② As an alternative, we segment a single long video into multiple samples. It can flexibly segment videos based on the maximum context length supported by VLMs. ③ Besides, we directly concatenate multiple video clips i.e., ⟨framek

i ,..,framek

i ,ocri,asri⟩, to the maximum context length. This strategy breaks video boundaries, effectively utilizing computational resources. However, mixing multiple video clips within a single sample may adversely affect training performance. Therefore, we insert a specific token: End of Video at the end of each video to mitigate this.

1

n

##### 7.4. Knowledge Taxonomy

As stated in the main text, to include richer knowledge in our textbook, we propose a hierarchical knowledge taxonomy comprising four hierarchical layers, namely Subject → Course → Sub-course → Knowledge Point. We instruct an LLM to span the knowledge taxonomy across multiple educational stages (from primary school to middle school) and diverse subjects (mathematics, physics, etc.). Lastly, we obtain a knowledge taxonomy comprising 6 subjects (mathematics, physics, chemistry, earth science, engineering, and computer science), 55 courses (Algebra, Solid Geometry,..), and 3915 knowledge points. As illustrated in Fig. 4, we plot six subjects along with their corresponding courses. Due to

space constraints, we visualized the top 9 courses and their proportion. The number of knowledge points included in each course is approximately the same.

- 7.5. Detail of InSI-SIM

As mentioned in Sec. 4.2, we design an in-sample image similarity metric (InSI-SIM). It measures the similarity between all images within a sample. Formally, for a subset D containing M samples, each comprising L images, the in-sample image similarity is computed as follows:

InSI-SIML=

1 M

M

k=1

- 1 L

- 2

L−1

i=1

L

j=i+1

CLIP(Imgk,i,Imgk,j) (1)

+ SSIM(Imgk,i,Imgk,j) /2

where CLIP(Imgk,i,Imgk,j) and SSIM(Imgk,i,Imgk,j) represent the semantic and structural similarity scores between images i and j in sample k, respectively.

- 8. Details of Experiments

- 8.1. Detail of Evaluation

We evaluate the pre-trained VLMs on two VQA benchmarks (TextVQA, OKVQA), a knowledge-centric benchmark (ScienceQA), and three math-related benchmarks (MathVista, MathVerse, MathVision) under few-shot settings. Following the previous works [25], we use the RICES-based few-shot prompting strategy which retrieves the k most similar samples from the training set based on the testing image feature. It should be noted that since MathVista, MathVerse, and MathVision only contain testing sets, we can not retrieve samples from their respective training sets. Consequently, for MathVista and MathVerse, we retrieve k examples from MathVision, while for MathVision, we retrieve examples from MathVista. When evaluating, we adopt the same prompt as Llava-1.5:

System Prompt: A chat between a human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human’s questions

- USER: <image>\n{example1 query}\nAnswer the question using a single word or phrase.

- ASSISTANT: {example1 answer}</s>

USER: <image>\n{example2 query}\nAnswer the

question using a single word or phrase.

- ASSISTANT: {example2 answer}</s>

.... USER: <image>\n{testing query}\nAnswer the question using a single word or phrase. ASSISTANT:

##### 8.2. Examples of Multimodal Textbook

We provide several detailed examples in Figs. 5 to 10. Specifically, Fig. 5 offers a detailed explanation of the Earth’s water cycle, presented through slides, photographs, and schematic diagrams. Figures 6 and 7 provide rich visualizations, including diagrams and texts, to elucidate the concepts of velocity and acceleration in physics. Figure 8 demonstrates the step-by-step, frame-by-frame problemsolving process for a mathematical geometry problem, detailing each critical step with accompanying text and visuals. Figure 9 presents a detailed depiction of chemical concepts such as atoms, molecules, and compounds through a combination of text and illustrations. Figure 10 introduces the depth-first search algorithm using an animation.

Except for refined ASR texts, we also provide the OCR texts in our textbook, which can be helpful for math-related scenario. For example, in Fig. 7, we utilize OCR to recognize formulas and symbols displayed on the screen, which facilitates better comprehension of physical concepts.

#### 9. Limitations

Although we already designed multiple levels of filtering, our textbook may still contain some redundant keyframes, low-quality texts, and so on. We will continue to improve the quality and knowledge density of our textbook. Besides, similar to prior multimodal models, our textbook primarily focuses on multimodal understanding and text generation for interleaved contexts. During training, the loss is not computed for image tokens. However, our textbook can also be used for omni-modal models including both understanding and generation tasks. We leave this for future work.

#### 10. Ethical discussion

During the collection and release of our multimodal textbook dataset, We are very concerned about ethical considerations. In addition to following the established corpora (e.g., MMC4 [60], OBELICS [22] and Omnicorpus [25]), we make additional efforts to uphold high ethical standards, such as employing LLMs to filter out inappropriate videos, including those with biases, pornographic content, or personal privacy information, such as identification documents and bank account details. We are open to further refining our strategy while maintaining open-source resources based on community feedback.

#### 11. License and Author Statement

We release the dataset under a CC-BY license and Terms of Use that require disclosure of when the dataset is used for the purpose of training models. This license is not intended to replace the licenses of the source content, and any use of content included in the dataset must comply with the

Subject #Video Duration (h) #Topic #Video Clip #Keyframe #ASR Token #OCR Token #Sample Mathematics 21.7k 4,423 725 809k 1.67M 72.5M 145M 123k Physics 11k 3,511 530 822k 0.95M 36.7M 73.4M 119k Chemistry 4.5k 2,643 410 234k 0.49M 15M 30M 32k Earth Science 12k 3,670 520 640k 1.03M 40M 80M 88k Engineering 13k 4,096 810 713k 1.15M 43.3M 86.6M 98k Computer Science 12.8k 4,354 820 782k 1.21M 42.8M 85.5M 150k All 75k 22,697 3,915 4M 6.58M 258M 500M 610k

Table 7. The statistics of our multimodal textbook. Topic denotes the knowledge points covered by each category of videos, which are sourced from our knowledge taxonomy.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Figure 4. Top: We plot six subjects along with their corresponding sub-courses. Due to space constraints, we selectively visualized only the courses with the highest proportions. Bottom: We count the knowledge points distribution belongs to each subject and its course

ASR The hydrologic cycle works like this: wherever there is water on the surface, evaporation can occur. This water vapor eventually cools, leading to condensation and precipitation. Once the water reaches the surface, we call it surface water. This includes water running over the surface in lakes, swamps, and rivers. However, when water hits the ground, it can also infiltrate into the soil and underlying ground.

[Figure 80]

[Figure 81]

ASR To illustrate infiltration, imagine a beaker. Is the beaker full? It might appear full of air, but if you add marbles, there’s still space between them. If you then fill it with sand, you’ll see even smaller spaces remain. Finally, if you pour water into the beaker, it begins to truly fill up. This is what infiltration looks like—water flowing down into the soil, eventually becoming groundwater. The point at which the soil or rock becomes fully saturated is called the water table.

[Figure 82]

ASR Most of the Earth’s surface is covered in water, but unfortunately, the majority is seawater, which contains dissolved salt. Drinking seawater is lethal, and it’s unsuitable for crops. Ocean water circulates through currents, which are influenced by

atmospheric patterns, Coriolis effects, and differences in salinity. For example, areas with high evaporation rates have higher salt concentrations, while regions with melting glacial ice have lower salt concentrations. These variations in salinity, combined with temperature differences, drive thermohaline circulation, connecting the entire ocean system into one cohesive flow.

[Figure 83]

ASR When it comes to freshwater, it can be categorized as either surface water or groundwater. Surface water is above the ground, like streams and lakes. Groundwater, on the other hand, lies beneath the surface. You can see evidence of the groundwater table near streams—if you dig a hole beside the stream, it will fill with water due to the surrounding groundwater.

[Figure 84]

ASR This brings us to aquifers, which are underground storage areas for groundwater. An unconfined aquifer allows water to move freely between the surface and the aquifer. If you dig deeper, you might encounter a confined aquifer, which is trapped between impermeable layers of rock, restricting its movement. This distinction is crucial for understanding how water is stored and accessed beneath the ground.

Figure 5. A case presented in our textbook illustrates the water cycle within the domain of earth science.

original licenses and applicable rights of its data subjects.

The purpose of this statement is to clarify the responsibilities and liabilities associated with the use of this dataset. While we have made every effort to ensure the accuracy and legality of the data contained within this dataset, we cannot guarantee its absolute completeness or correctness.

Therefore, if any rights, legal or otherwise, are violated through this dataset, including but not limited to copyright infringement, privacy violations, or misuse of sensitive information, we, the authors, assume no liability for such violations.

By utilizing this dataset, you agree that any conse-

ASR: Assume the object is initially at rest at point A. It then moves to the right, reaching a velocity of 10 meters per second in 5 seconds. What would be the acceleration at point B?

[Figure 85]

[Figure 86]

[Figure 87]

ASR: To find the answer, we can refer to the formula. The initial velocity, denoted as u, is 0 meters per second since the object is stationary. The final velocity, v is 10 meters per second, and the time taken, t is 5 seconds.

[Figure 88]

ASR: Therefore, the acceleration, A can be calculated by subtracting the initial velocity from the final velocity and dividing the result by the time taken. This gives 10 meters per second minus 0 meters per second, divided by 5 seconds. Performing this calculation, we have 10 divided by 5, which equals 2.

[Figure 89]

[Figure 90]

ASR: Now, let’s determine the units of acceleration. Since acceleration is defined as the change in velocity divided by the time taken, its units will be meters per second per second. This can be simplified by expressing it as meters per second squared, which is more concise. Thus, in this case, the acceleration is 2 meters per second squared.

OCR 0 - 10 m/s 00:05 (Time taken) t = 5 sec

[Figure 91]

[Figure 92]

- (A) (Initial velocity) u = 0 m/s a = (v - u) / t a = (10 m/s - 0 m/s) / 5 s

= 10 m/s / 5 s

= 2 m/s^2

- (B) (Final velocity) v = 10 m/s m/s/s = m / s x s = m / s^2

[Figure 93]

Figure 6. A case presented in our textbook introducing the principles of mechanics within the domain of physics.

quences, legal or otherwise, arising from using this dataset will be the user’s sole responsibility. You acknowledge that you will exercise due diligence and adhere to all applicable laws, regulations, and ethical guidelines when using the

dataset.

By accessing, downloading, or using this dataset, you signify your acceptance of this statement and your commitment to abide by the terms and conditions of the CC-BY

ASR To illustrate the concept of inertia, let’s consider two objects. The first object has a mass of 10 kilograms, while the second object has a mass of 100 kilograms. In both cases, we apply a force of 50 newtons.

[Figure 94]

ASR Now, which object do you think has more inertia—the one with less mass or the one with more mass? Intuitively, you know it’s the object with more mass. It’s easier to move a lighter object, but more difficult to move a heavier object because the heavier object has greater inertia. This demonstrates that inertia is directly proportional to mass.

[Figure 95]

ASR Now, according to Newton’s second law, the net force acting on an object is equal to the product of its mass and acceleration. From this, we can see that acceleration is the net force divided by the mass. For the first object, with a force of 50 newtons and a mass of 10 kilograms, the acceleration is 50÷10, which equals 5 meters per second squared. For the second object, with the same force of 50 newtons but a mass of 100 kilograms, the acceleration is 50÷100, which equals 0.5 meters per second squared.

[Figure 96]

ASR As you can see, the lighter object has a much larger acceleration, while the heavier object has a much smaller acceleration. This indicates that the lighter object has less inertia, as it was easier to accelerate with a small force. In contrast, the heavier object has more inertia, since applying the same force resulted in a much smaller acceleration. Therefore, as the mass of an object increases, its inertia also increases. This demonstrates that inertia is directly proportional to mass.

[Figure 97]

ASR That’s the basic concept of inertia as it applies to translational motion. But what about rotational motion? How does inertia come into play there? Let’s compare two objects to explore this: the first is a thin hoop, and the second is a solid disk.

[Figure 98]

ASR For the thin hoop, the mass is concentrated along the edge of the circle, whereas for the solid disk, the mass is distributed throughout the circle. Now, let’s assume that both objects have the same mass—10 kilograms—and the same radius of 2 meters. With these conditions, which one do you think has more rotational inertia? Is it the thin hoop or the solid disk? What would be your answer?

[Figure 99]

Figure 7. A case presented in our textbook introducing the concepts of velocity and acceleration within the context of physics.

license.

If you disagree with the terms of this statement or the CC-BY license, you are not authorized to use this dataset.

The dataset will be hosted and maintained on the Hug-

ging Face Hub.

ASR Let’s find the area of the blue rectangle. The rectangle is twice as wide as it is tall and intersects the semicircle at specific points. If you’d like to solve this on your own, pause here, because I’ll explain the solution step by step.

[Figure 100]

ASR First, let’s connect two points on the rectangle and focus on the triangle that forms. This triangle is a right triangle because the shape is a rectangle, and it’s also isosceles since two of its sides are congruent. This tells us it’s a 45-45-90 triangle. Since one of its angles is 45 degrees, we can deduce another angle by noting that these two angles form a linear pair, meaning they add up to 180 degrees. That makes the other angle 135 degrees.

[Figure 101]

[Figure 102]

[Figure 103]

ASR Next, let’s extend the circle and focus on the 135-degree angle. This angle is an inscribed angle, and inscribed angles have a special property—they are exactly half the measure of the arc they subtend. So, the red arc cut by this angle measures double 135 degrees, which is 270 degrees. The remaining arc of the circle, which we’ll call the blue arc, must then be 90 degrees.

[Figure 104]

ASR Now, let’s draw two radii to mark this 90-degree blue arc. The angle formed at the center of the circle is called a central angle, and central angles are equal to the arcs they subtend. So, this angle is a 90-degree angle.

[Figure 105]

ASR Since both sides of this angle are radii, their lengths are equal to the radius of the circle, which is 5. Connecting the points forms another isosceles right triangle. Using the properties of a 45-45-90 triangle, we can determine that the hypotenuse of this triangle is 5\sqrt{2}. This value represents the diagonal of the rectangle. Returning to the rectangle, we know the shorter side is x,

and the longer side, being twice as long, is 2x. Using the right triangle formed by these sides and the diagonal, we can determine the value of x. Solving this, we find that the shorter side of the rectangle is \sqrt{10}, and the longer side is 2\sqrt{10}.

[Figure 106]

[Figure 107]

[Figure 108]

ASR Now, we can calculate the area of the rectangle. The area is equal to the base times the height. With the base as 2102\sqrt{10}210 and the height as \sqrt{10}, the area becomes 2×10, which is 20. Adding a label, the area of the blue rectangle is 20 square units.

- Figure 8. A case presented in our textbook demonstrates how to solve a question about planar geometry in the domain of mathematics.

ASR What is the difference between an atom and a molecule? Let’s consider three substances: helium, hydrogen gas, and water (H₂O). Which of these are composed of atoms, and which are composed of molecules?

[Figure 109]

[Figure 110]

ASR Helium is composed of atoms. Each particle of helium consists of a single helium atom. In contrast, hydrogen gas is composed of molecules. Each particle of hydrogen gas contains two hydrogen atoms bonded together.

[Figure 111]

ASR A molecule, by definition, is a particle made up of two or more atoms. Water is also composed of molecules, specifically one oxygen atom and two hydrogen atoms. At the center of a water molecule is the larger oxygen atom, with the two smaller hydrogen atoms attached. Thus, a molecule can consist of either the same type of atom or different types of atoms bonded together.

[Figure 112]

ASR Helium is classified as a pure element because it is made up of only one type of atom. Similarly, hydrogen gas (H₂) is a pure element, as it is composed solely of hydrogen atoms. Water, however, is not a pure element; it is a compound, as it contains two different types of atoms: hydrogen and oxygen. While H₂ and H₂O are both considered molecules, only H₂ is a pure element, and H₂O is classified as a compound due to the presence of multiple atom types.

[Figure 113]

ASR Now, let’s work through more examples. For each substance, determine whether it is composed of atoms or molecules and whether it is a pure element or a compound.

[Figure 114]

ASR Let’s start with O₂. Is it composed of atoms or molecules? A particle of O₂ consists of two oxygen atoms bonded together, so it’s a molecule, as it contains multiple atoms. Since it only contains one type of atom—oxygen—it is classified as a pure element. Now, what about CO₂? Would you classify a particle of CO₂ as an atom or a molecule? Since CO₂ is made up of multiple atoms, it is considered a molecule.

[Figure 115]

ASR However, it doesn’t consist of just one type of atom. It contains both carbon and oxygen atoms, meaning it is not a pure element. Instead, it is a compound. A compound is defined as a substance composed of different types of atoms bonded together. In other words, a compound is not a pure element. Finally, let’s look at neon. Neon is composed of individual atoms. A particle of neon consists of a single neon atom, unlike oxygen molecules, which are made up of two oxygen atoms bonded together. Neon, therefore, is not a molecule but is instead classified as an atom and a pure element, as it contains only one type of atom.

- Figure 9. A case presented in our textbook illustrates the concepts of molecules, atoms, and compounds in the domain of chemistry.

ASR Depth-first search (DFS) works by selecting the next node to explore until it cannot proceed further, at which point it backtracks and continues its exploration from a previous point. Let’s start a depth-first search on node 0 and see how it unfolds. We begin at node 0 and arbitrarily choose a node to move to.

[Figure 116]

ASR From node 0, we go to node 9. At node 9, there’s only one option, so we move to node 8.

[Figure 117]

ASR At node 8, we arbitrarily select an edge and proceed to node 7. At node 7, there are multiple edges to choose from, so we decide to go to node 10. From node 10, we move to node 11, and then back to node 7. However, we do not revisit already visited nodes or nodes that are currently being explored, so at this point, we need to backtrack. To indicate backtracking, we’ll label the edges and nodes as gray. We backtrack all the way to node 7. Since we haven’t finished exploring all edges from node 7, we pick another edge and move to node 3.

[Figure 118]

ASR At node 3, we proceed to node 2. Node 2 is a dead end, so we backtrack and explore another edge from node 3, moving to node 4. Node 4 is also a dead end, so we backtrack again to node 3. Finally, we take the last edge from node 3 and move to node 5. From node 5, we go to node 6, and from node 6, we reach node 7 again. Since node 7 is currently being visited, we backtrack all the way to node 8. At node 8, there’s still one unvisited edge that leads to node 1. From node 1, we move back to node 0. However, since node 0 is currently being explored, we backtrack all the way to node 0. At this point, we’ve completed the depth-first search traversal of the graph.

[Figure 119]

[Figure 120]

[Figure 121]

ASR This was just one possible depth-first search traversal, as the path can vary depending on the choices made during exploration. Now, let’s look at some pseudocode for depth-first search to gain a deeper understanding of its implementation. The first step we’ll need is to...

[Figure 122]

OCR "# Global or class scope variables\nn = number of nodes in the graph\ng = adjacency list representing graph\nvisited = [false, …, false] # size n\n\nfunction dfs(at):\n if visited[at]: return\n visited[at] = true\n\n neighbours = graph[at]\n for next in neighbours:\n dfs(next)\n\n# Start DFS at node zero\nstart_node = 0\ndfs(start_node)"

Figure 10. A case presented in our textbook introduces a depth-first search algorithm.

