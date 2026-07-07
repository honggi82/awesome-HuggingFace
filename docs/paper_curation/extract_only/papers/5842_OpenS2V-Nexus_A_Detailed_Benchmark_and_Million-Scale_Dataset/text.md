arXiv:2505.20292v4[cs.CV]3Jun2025

# OPENS2V-NEXUS: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation

### Shenghai Yuan1,3,*, Xianyi He1,3,*, Yufan Deng1, Yang Ye1,3, Jinfa Huang2, Bin Lin1,3, Jiebo Luo2, Li Yuan1,

†

∗ Equal Contributors, † Corresponding Authors

1 Peking University, Shenzhen Graduate School, 2 University of Rochester, 3 Rabbitpre AI {yuanshenghai@stu, yuanli-ece@}.pku.edu.cn

## Abstract

Subject-to-Video (S2V) generation aims to create videos that faithfully incorporate reference content, providing enhanced flexibility in the production of videos. To establish the infrastructure for S2V generation, we propose OPENS2V-NEXUS, consisting of (i) OpenS2V-Eval, a fine-grained benchmark, and (ii) OpenS2V-5M, a million-scale dataset. In contrast to existing S2V benchmarks inherited from VBench [36] that focus on global and coarse-grained assessment of generated videos, OpenS2V-Eval focuses on the model’s ability to generate subject-consistent videos with natural subject appearance and identity fidelity. For these purposes, OpenS2V-Eval introduces 180 prompts from seven major categories of S2V, which incorporate both real and synthetic test data. Furthermore, to accurately align human preferences with S2V benchmarks, we propose three automatic metrics, NexusScore, NaturalScore, and GmeScore, to separately quantify subject consistency, naturalness, and text relevance in generated videos. Building on this, we conduct a comprehensive evaluation of 18 representative S2V models, highlighting their strengths and weaknesses across different content. Moreover, we create the first open-source large-scale S2V generation dataset OpenS2V-5M, which consists of five million high-quality 720P subject-text-video triples. Specifically, we ensure subject-information diversity in our dataset by (1) segmenting subjects and building pairing information via cross-video associations and (2) prompting GPT-Image on raw frames to synthesize multi-view representations. Through OPENS2V-NEXUS, we deliver a robust infrastructure to accelerate future S2V generation research. 1

## 1 Introduction

With the advancement of video foundational models [50, 89, 59, 124, 41, 70, 86, 105], Subject-toVideo (S2V) generation has attracted increasing attention, enabling the generation of videos centered on reference subjects. Previous tuning-based methods [69, 30, 65, 24] require fine-tuning for each sample during inference, which is time-consuming. Recently, several open-source S2V models [123, 97, 21], including ConsisID [113], Phantom [55], and VACE [40], as well as closed-source models [44, 5, 43, 87, 18], have demonstrated the ability to perform tuning-free S2V generation.

Although these methods demonstrate promising results, there remains a shortage of benchmarks for objectively evaluating the strengths and limitations of S2V models. As shown in Table 1, existing video generation benchmarks predominantly focus on text-to-video tasks, with prominent examples including VBench [37] and ChronoMagic-Bench [115]. While ConsisID-Bench [113] is applicable

1The source data and code are publicly available on https://pku-yuangroup.github.io/OpenS2V-Nexus.

Preprint.

- Table 1: Comparison of the Characteristics of our OpenS2V-Eval with existing Benchmarks. Most of them focus on T2V and neglect the evaluation of subject naturalness. _ means suboptimal.

Benchmark # Type Visual Quality Text Relevance Motion Subject Consistency Subject Naturalness

Make-a-Video-Eval [81] Text-to-Video ✓ ✓ ✗ ✗ ✗ FETV [58] Text-to-Video ✓ ✓ ✓ ✗ ✗ T2VScore [100] Text-to-Video ✓ ✓ ✓ ✗ ✗ EvalCrafter [57] Text-to-Video ✓ ✓ ✓ ✗ ✗

- VBench [36] Text-to-Video ✓ ✓ ✓ ✗ ✗ VBench++ [37] Text-to-Video ✓ ✓ ✓ ✗ ✗ ChronoMagic-Bench [115] Text-to-Video ✓ ✓ ✓ ✗ ✗ ConsisID-Bench [113] Subject-to-Video ✓ ✓ ✓ ✓ ✗ Alchemist-Bench [13] Subject-to-Video ✓ ✓ ✓ ✓ ✗ A2 Bench [21] Subject-to-Video ✓ ✓ ✓ ✓ ✗ VACE-Bench [40] Subject-to-Video ✓ ✓ ✓ ✓ ✗ OpenS2V-Eval Subject-to-Video ✓ ✓ ✓ ✓ ✓

to S2V, it is restricted to assessing facial consistency. Alchemist-Bench [13], VACE-Benchmark [40], and A2 Bench [21] support the evaluation of open-domain S2V; however, their evaluation are primarily global and coarse-grained. For example, they neglect to assess the naturalness of subjects. Furthermore, the latter two benchmarks [40, 21] inherit their subject consistency metrics from

- VBench [37], which calculates similarity directly between uncropped video frames and reference images—an approach that unavoidably introduces background noise and reduces accuracy.

Subject-to-Video (S2V) models currently face three major challenges: (1) Poor generalization: These models often perform poorly when encountering subject categories not seen during training [40, 113]. For instance, a model trained exclusively on Western subjects typically performs worse when generating Asian subjects; (2) Copy-paste issue: The model tends to directly transfer the pose, lighting, and contours from the reference image to the video, resulting in unnatural outcomes [21]; (3) Inadequate human fidelity: Current models often struggle to preserve human identity as effectively as they do non-human entities [55]. An effective benchmark should be able to identify these issues. However, even when the generated subject appears unnatural or when the fidelity is low, existing benchmarks [40, 21, 121] still yield high scores, hindering progress in the field.

To address this challenge, we introduce OpenS2V-Eval, the first comprehensive subject-to-video benchmark in the field. Specifically, we define seven categories: ① single-face-to-video, ② singlebody-to-video, ③ single-entity-to-video, ④ multi-face-to-video, ⑤ multi-body-to-video, ⑥ multientity-to-video, and ⑦ human-entity-to-video, as in Figure 1. For each category, we design 30 test samples with rich visual content, which assess the model’s generalization ability across different subjects. To address the limited robustness of existing automatic metrics, we first develop NexusScore, which combines an image-prompt detection model [15] and a multimodal retrieval model [119] to accurately evaluate subject consistency. Next, we introduce NaturalScore, a GPT-based metric designed to bridge the gap in evaluating subject naturalness. Finally, we propose GmeScore, based on MLLM [119], which provides a more precise assessment of text relevance compared to conventional CLIPScore [73]. Using OpenS2V-Eval, we conduct both qualitative and quantitative evaluations of nearly all open-source and closed-source S2V models, offering valuable insights for model selection.

Furthermore, when the community attempts to extend foundational models to downstream tasks, existing datasets are limited in their support for complex tasks [8, 31, 69, 82, 32, 83, 61], as shown in Table

- 2. To address this limitation, we propose OpenS2V-5M, the first million-scale dataset specifically designed for subject-to-video, which is also applicable to text-to-video [105, 78]. Unlike previous methods [113, 40, 21, 13, 55] that rely solely on regular subject-text-video triples—where subject images are segmented from training frames, potentially causing the model to learn shortcuts rather than intrinsic knowledge—we enrich it with Nexus Data, through (1) building pairing information via cross-video associations and (2) prompting GPT-Image-1 [1] on raw frames to synthesize multi-view representations, to address the three core challenges mentioned above at the data level. The contributions of this work are as follows:

- i) New S2V Benchmark. We introduce OpenS2V-Eval for comprehensive evaluation of S2V models and propose three new automatic metrics aligned with human perception.
- ii) New Insights for S2V Model Selection. Our evaluations using OpenS2V-Eval provide crucial insights into the strengths and weaknesses of various subject-to-video generation models.

||[Figure 1]|
|---|
<br><br>|[Figure 2]|
|---|
<br><br>Single-Entity-to-Video<br><br>|[Figure 3]|
|---|
<br><br>|[Figure 4]|
|---|
<br><br>|[Figure 5]|
|---|
<br><br>|[Figure 6]|
|---|
<br><br>|[Figure 7]|
|---|
<br><br>|[Figure 8]|
|---|
<br><br>Single-Body-to-Video Multi-Entity-to-Video<br><br>|[Figure 9]|
|---|
<br><br>|[Figure 10]|
|---|
<br><br>|[Figure 11]|
|---|
<br><br>|[Figure 12]|
|---|
<br><br>Multi-Body-to-Video<br><br>|[Figure 13]|
|---|
<br><br>|[Figure 14]|
|---|
<br><br>|[Figure 15]|
|---|
<br><br>|[Figure 16]|
|---|
<br><br>|Single-Thing-to-Video<br><br>[Figure 17]|
|---|
<br><br>ideo<br><br>Single-Face-to-Video<br><br>|[Figure 18]|
|---|
<br><br>|[Figure 19]|
|---|
<br><br>Human-Entity-to-Video<br><br>Multi-Face-to-Video<br><br>|[Figure 20]|
|---|
<br><br>|[Figure 21]|
|---|
<br><br>|[Figure 22]|
|---|
<br><br>|[Figure 23]|
|---|
<br><br>|[Figure 24]|
|---|
<br><br>|[Figure 25]|
|---|
<br><br>|[Figure 26]|
|---|
<br><br>|[Figure 27]|
|---|
<br><br>|[Figure 28]|
|---|
<br><br>|
|---|

- Figure 1: Example of Seven Categories from OpenS2V-Eval. These categories fully encompass the subject-to-video tasks, allowing comprehensive evaluation. Videos are generated by Kling [43].

- Table 2: Comparison of the Statistics of OpenS2V-5M with existing Video Generation Datasets. Most of them are inadequate for extending foundational models to subject-to-video generation task.

Dataset # Type Resolution Video Clips Average Length (s) Video Duration (h)

MSRVTT [106] Text-to-Video 240P 10K 14.4 40 WebVid-10M [4] Text-to-Video 360P 10M 18.7 52K InternVid [95] Text-to-Video 720p 234M 11.7 760K HD-VG-130M [94] Text-to-Video 720p 130M 4.9 178K Panda-70M [12] Text-to-Video 720P 70M 8.6 167K OpenVid-1M [67] Text-to-Video 512P 1M 7.2 2K Koala-36M [91] Text-to-Video 720P 36M 17.2 172K ChronoMagic-Pro [115] Text-to-Video 720p 460K 234.8 30K OpenHumanVid [45] Text-to-Video 720P 52.3M 4.9 70K

OpenS2V-5M Subject-to-Video 720P 5.4M 6.6 10K

- iii) Large-Scale S2V Dataset. We create OpenS2V-5M, a dataset with 5.1M high-quality regular data and 0.35M Nexus Data, the latter is expected to address the three core challenges of subject-to-video.

## 2 Related Work

Automatic Metrics for Subject-to-Video Generation. Existing video generation benchmarks typically focus on text-to-video tasks [42, 101, 108, 96, 19, 28]. Notable examples include MSR-VTT [106] and Make-a-Video-Eval [81], which are pioneering benchmarks for video generation evaluation. Later, VBench [36, 37, 121] and EvalCrafter [57] consider multiple evaluation dimensions, providing a more comprehensive benchmark by considering additional mode-specific factors. ConsisID-Bench [113] represents an early work for S2V, but is limited to human domain. Although recent benchmarks, such as A2 Bench [21] and VACE-Benchmark [40], are applicable to open-domain S2V tasks, they rely on VBench [36] metrics to calculate subject consistency without being specifically tailored for S2V. Therefore, we develop the first comprehensive subject-to-video benchmark, which includes 180 balanced test pairs. Furthermore, we introduce NexusScore, NaturalScore, GmeScore to accurately measure subject consistency, naturalness, and text relevance, thereby addressing this gap in the field.

Datasets for Subject-to-Video Generation. Large-scale, high-quality video datasets [4, 95, 94, 67, 93] are essential to emerging DiT-based generation model [118, 79, 54, 7, 20, 54, 60, 111, 52, 122]. For instance, newly released Panda-70M [12], Koala-36M [91], and ChronoMagic-Pro [115] feature millions of high-resolution video-text pairs, which have substantially contributed to the progress of the field. However, when the community seeks to extend the foundational model to downstream tasks, existing open-source datasets are inadequate for subject-to-video [18, 55]. Moreover, we identify a significant issue, whether the model is closed-source [44, 5, 43] or open-source: they all suffer the

[Figure 29]

[Figure 30]

##### GPT 4o

[Figure 31]

[Figure 32]

real Images

[Figure 33]

[Figure 34]

[Figure 35]

- (e1)

Single Face

- (e2)

Single Body

- (e3)

Single Entity

- (e4)

Multi Face

- (e5)

Multi Body

- (e6)

Multi Entity

- (e7)

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Subject Consistency (NexusScore)

Visual Quality (AestheticScore)

ConsisID

[Figure 42]

###### Video Caption

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Aggregate

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Subject Naturalness (NaturalScore)

Motion Amplitude (MotionScore)

A2 Bench

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Pixabay Pexels Mixkit

[Figure 57]

[Figure 58]

[Figure 59]

Human Entity

DreamBench

Synthetic Images

Text Relevance (GmeScore)

Face Consistency (FaceSim-Cur)

High-quality Videos

Test Sample Construction Evaluation

- Figure 2: The Pipeline of Constructing OpenS2V-Eval. (Left) Our benchmark includes not only real subject images but also synthetic images constructed through GPT-Image-1 [1], allowing for a more comprehensive evaluation. (Right) The metrics are tailored for subject-to-video generation, evaluating not only S2V characteristics (e.g., consistency) but also basic video elements (e.g., motion).

three core issues of subject-to-video mentioned above. To address this gap, we introduce the first million-scale subject-to-video dataset, named OpenS2V-5M. In addition to extracting subject images from segmented training frames, we further propose constructing subject images through building pairing information and synthesis using GPT-Image-1 [1], thereby empowering the community.

- 3 OpenS2V-Eval

### 3.1 Prompt Construction

To comprehensively evaluate the capabilities of subject-to-video models [18, 55, 22], the designed text prompts must encompass a wide range of categories, and the corresponding reference images must meet high-quality standards. Consequently, to construct a benchmark for subject-to-video that incorporates diverse visual concepts, we divide this task into seven categories: ① single-face-to-video, ② single-body-to-video, ③ single-entity-to-video, ④ multi-face-to-video, ⑤ multi-body-to-video, ⑥ multi-entity-to-video, and ⑦ human-entity-to-video. Based on this, we collect 50 and 24 subjecttext pairs from ConsisID [113] and A2 Bench [21], respectively, for constructing ①, ②, and ⑥. Additionally, we gather 30 reference images from DreamBench [71] and utilized GPT-4o [1] to generate captions for building ③. Subsequently, we source high-quality videos from copyright-free websites, employe GPT-Image-1 [1] to extract subject images from the videos, and use GPT-4o to caption the videos, thereby obtaining the remaining subject-text pairs. Collection for each sample is performed manually to ensure benchmark quality. Unlike prior benchmark [13, 40] that relied solely on real images, the inclusion of synthetic samples enhances the diversity and precision of evaluation.

### 3.2 Benchmark Statistics

We collect 180 high-quality subject-text pairs, consisting of 80 real and 100 synthetic samples. Except for ④ and ⑤, which each contain 15 samples, all other categories include 30 samples. The data statistics are shown in Figure 3. As illustrated in (c) and (d), the seven major categories of the S2V task encompass a broad range of testing scenarios, including various objects, backgrounds and actions. Additionally, terms associated with humans, such as “woman” and “man,” make up a significant proportion, allowing for a comprehensive evaluation of existing methods’ ability to preserve human identity—an especially challenging aspect of the S2V task. Furthermore, since some methods prefer long captions [40] while others prefer short ones [55], we ensure that the text prompts vary in length, as shown in (b). We also assess the aesthetic scores of the collected reference images, with the results showing that most score above 5, indicating high quality. Moreover, we retain some lower-quality images to preserve the diversity of evaluation. Due to the limitations of existing S2V models [43, 18, 44], we restrict the number of subject images for each sample to no more than three.

[Figure 60]

0.5

[Figure 61]

6%

16%

0.4

19%

Percentage

0.3

0.2

22%

0.1

36%

0

Score Interval

<50 50-100 100-150 150-200 >200

<4.5 4.5-5.0 5.0-5.5 5.5-6.0 >6.0

(a) Distribution of Aesthetic Scores (b) Prompt Word Range (c) Prompt Word Cloud (d) Subject Categories

- Figure 3: Statistics in OpenS2V-Eval. The benchmark covers diverse categories and prompt words, with subject images displaying high aesthetics, thus enabling a thorough evaluation.

### 3.3 New Automatic Metrics

As previously mentioned, existing S2V benchmarks are usually adapted from T2V rather than being specifically tailored. For subject-to-video, it is crucial to evaluate not only global aspects such as visual quality and motion but also subject consistency and naturalness in the synthesized output.

NexusScore To calculate subject consistency, prior studies [40, 55, 21, 36, 37] directly compute the similarity between uncropped video frames and reference images in the DINO [116] or CLIP [73] space. However, this method introduces background noise, and the feature space has been proven to be unreasonable [100, 58, 115]; please refer to the Appendix B.1 for more details. To address this issue, we introduce the NexusScore SNexus, which utilizes the image-prompt detection model Mdetect [15] and the multimodal retrieval model Mretrieve [119]. Specifically, both the reference images {Ri}Ii=1 and video frames {It}Tt=1 are firstly fed into the Mdetect, which identifies the relevant target in each frame and generates the corresponding bounding box Bi,t that encloses the target:

Bi,t = Mdetect(Ri,It), (1)

To improve the accuracy of the bounding box, for each subject, we crop the region Bi,t to get the cropped reference image Ci,t. Then, we compute the similarity between the cropped reference image Ci,t and the corresponding target entity name Ei,t in the unified text-image feature space. This similarity is denoted as s, and it is computed using the multimodal retrieval model Mretrieve:

si,t = Mretrieve(Ci,t,Ei,t), (2)

If bbox Bi,t confidence ci,t and si,t exceeds a predefined threshold α and β, we proceed to the next stage. Finally, the similarity between Ci,t and Ri is evaluated in the image feature space, yielding:

T′

I

1 I × T′

Mretrieve (Ci,t,Ri), where ci,t > α and si,t > β (3)

SNexus =

t=0

i=0

where T′ means the total number of frames in which an object is detected. Appendix D.4 for details. NaturalScore Unlike existing subject-to-video benchmarks [113, 21, 40, 55] that focus exclusively on subject consistency, we additionally evaluate whether the generated subject appears natural, i.e., whether it conforms to physical laws. This is due to the prevalent “copy-paste” issue in current S2V methods, where the model blindly copies the reference image onto the generated scene, resulting in high consistency scores even when the output fails to align with typical human perception.

To address this issue, a straightforward solution is to employ the AIGC anomaly detection model [107, 46, 66]. However, we found that the accuracy of open-source models is suboptimal. An alternative approach is to utilize open-source multimodal large language models [3, 51, 85] for video scoring. However, these models exhibit poor instruction-following performance and are prone to significant hallucinations. For a more details, please refer to Appendix B.2. As a result, we use GPT-

#### 4o [1] to simulate human evaluators, which provides superior accuracy and flexibility. Specifically, we subtly design a five-point evaluation criterion based on common sense and physical laws, denoted as C = {c1,c2,c3,c4,c5}, where each ci represents a score corresponding to a specific evaluation level. For each video, we uniformly sample T frames, denoted as {It}Tt=1. These frames are then input into GPT-4o MGPT, which assigns a score st and provides reasoning based on the five-point scale. The final score SNatural is computed as the average of the scores assigned to all T frames:

T

1 T

MGPT(It) (4)

SNatural =

t=1

|[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>A1 A1 A2 A2<br><br>[Figure 74]<br><br>Video Clips Cluster<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>Subject Alignment<br><br>B2<br><br>|[Figure 85]|
|---|
<br><br>|[Figure 86]|
|---|
<br><br>[Figure 87]<br><br>[Figure 88]<br><br>GPT 4o<br><br>Synthetic Subject 1 Synthetic Subject 2<br><br>Cross-Frame Pairs GPT-Frame Pairs<br><br>Cluster<br><br>Group Deletion<br><br>Boundary<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>High-quality Videos<br><br>[Figure 92]<br><br>Data Pipeline<br><br>Data Curation<br><br>(source from Open-Sora Plan)<br><br>Data Preprocessing<br><br>(aesthetic, motion, watermark)<br><br>Subject-Driven<br><br>Annotation<br><br>(caption, bbox, mask)<br><br>Regular Data<br><br>Nexus Data<br><br>|-Frame Pairs Cross-Frame Pairs|-Frame Pairs GPT-Frame Pairs|
|---|---|
<br><br>human<br><br>phone<br><br>next|
|---|

- Figure 4: The Pipeline of Constructing OpenS2V-5M. First, we filter low-quality videos based on scores such as aesthetics and motion, then utilize GroundingDino [56] and SAM2.1 [76] to extract subject images and get Regular Data. Subsequently, we create Nexus Data through cross-video association and GPT-Image-1 [1] to address the three core issues encountered by S2V models.

GmeScore Existing methods commonly calculate text relevance using CLIP [73] or BLIP [117]. However, several studies [58, 115, 100] have identified inherent flaws in these models’ feature spaces, resulting in inaccurate scores. Additionally, their text encoders are limited to 77 tokens, which makes them unsuitable for the long text prompts preferred by current DiT-based video generation models [59, 79, 109, 89]. In light of this, we opt to utilize GME [119], a model fine-tuned on Qwen2-VL [90], which naturally accommodates text prompts of varying lengths and yields more reliable scores.

## 4 OpenS2V-5M

### 4.1 Data Construction

Subject-Driven Processing. As noted previously, existing large-scale video generation datasets typically consist only of text and video [115, 12, 91, 45], limiting their applicability for developing complex subject-to-video tasks. To overcome this limitation, we develop the first large-scale subjectto-video dataset, with raw videos sourced from Open-Sora Plan [50]. Given that the metadata includes video captions, we initially select videos featuring human, as these tend to contain a larger number of subjects. Next, we filter out low-quality video based on aesthetic [16], motion [6], and technical scores [99], resulting in 5,437,544 video clips. Building on this, and following the ConsisID data pipeline [113], we utilize Grounding DINO [56] and SAM2.1 [76] to extract subjects from each video, yielding regular data suitable for subject-to-video tasks. Finally, to ensure data quality, we assign aesthetic score and GmeScore to the reference images using the aesthetic [16] and multimodal retrieval models [119], enabling users to adjust thresholds to balance data quantity and quality.

Generalized Nexus Construction. Existing S2V methods primarily rely on regular data, where the extracted subject often shares the same view as the one in the training frames and may be incomplete, leading to the three core challenges discussed in Section 1. This limitation arises due to the extraction of the reference image directly from the ground truth video, leading the model to take shortcuts by copying the reference image onto the generated video instead of learning the underlying knowledge, reducing generalization. To overcome this, we introduce Nexus Data, including GPT-Frame Pairs and Cross-Frame Pairs. Comparison between regular data and Nexus Data is shown in Figure 5.

For GPT-Frame Pairs: let I0 represent the first frame of a given video, and let K = {k1,k2,...,kn} be a set of keywords associated with the subject of the video. We input I0 and K into GPT-Image-1 [1] MGPT, which then generates a complete image Igen of the corresponding subject, forming the pair ⟨I0,Igen⟩, which we refer to as GPT-Frame Pairs. Due to the powerful generative capabilities of GPT-Image-1, it can reconstruct incomplete subjects and generate consistent content from multiple perspectives, ensuring alignment with our data requirements. This relationship can be formalized as:

Igen = MGPT(I0,K) (5) For Cross-Frame Pairs: since clips are split from long videos, where there exists an inherent temporal and semantic correlation between clips [123]. To capture this, we aggregate clips from the same long video, denoted as C = {C1,C2,...,Cm}, where each Ci corresponds to a different segment of the

Regular Data: incomplete, same view, low resolution

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Nexus Data: complete, novel view, high resolution

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

(a) Input Video Frame (b) Output Subject Images

### Figure 5: Comparison between Regular Data and Nexus Data. The latter is of higher quality.

video. The similarity between subjects across these clips is computed using a multimodal retrieval model [119] Mretrieval, which computes the similarity score S(Cij,Ckl) for any pair of clips Cij and Ckl, where i ̸= k represents different segments of the video, and j and l represent different subjects:

S(Cij,Ckl) = sim(Mretrieval(Cij),Mretrieval(Ckl)) (6) where sim(·,·) means computing the similarity. This process enable the formation of Cross-Frame Pairs ⟨Cij,Ckl⟩. Finally, we assign aesthetic score [16] and GmeScore to each sample.

### 4.2 Dataset Statistics

OpenS2V-5M is the first open-source million-scale subject-to-video dataset. It includes 5.1M regular data, commonly used in existing methods [40, 21, 55], as well as 0.35M Nexus Data, generated through GPT-Image-1 [1] and cross-video associations. This dataset is anticipated to address the three core challenges faced by S2V models. Detailed statistics can be found in the Appendix C.2.

5 Experiments

### 5.1 Evaluation Setups

Evaluation Baseline. We evaluate almost all S2V models, including four closed-source and fourteen open ones, including models that support all type of subject (e.g., Vidu [5], Pika [44], Kling [43], VACE [40], Phantom [55], SkyReels-A2 [21], and HunyuanCustom [33]), as well as models that only support human identity (e.g., Hailuo [87], ConsisID [113], Concat-ID [123], FantasyID [120], EchoVideo [97], VideoMaker [103], and ID-Animator [29]).

Application Scope. OpenS2V-Eval presents an automated scoring method for evaluating subject consistency, subject naturalness, and text relevance. By incorporating existing metrics for visual quality, motion amplitude, and face similarity (e.g., Aesthetic Score [16], Motion Score [6], and FaceSim-Cur [113]), it facilitates a comprehensive evaluation of the S2V model across six dimensions. Furthermore, human evaluation can be utilized to provide a more precise assessment.

Implementation Details. Closed-source S2V models can only perform manually through their interfaces, and the inference speed of open-source models is relatively slow (e.g., VACE-14B [40] requires over 50 mins to get a 81 × 720 × 1280 video on a single Nvidia A100). Therefore, for each baseline, we only generate a video for each test sample in OpenS2V-Eval. We then evaluate all generated videos using the six aforementioned automated metrics. All inference settings follow the official implementation, with the seed fixed at 42. Further details are provided in the Appendix D.

### 5.2 Comprehensive Analysis

Quantitative Evaluation. We first present a comprehensive qualitative evaluation of different methods, with results displayed in Table 3, 4, and 5. All models are capable of generating videos with high visual quality and text relevance. For open-domain S2V, closed-source models generally outperform their open-source counterparts. Among these, Pika [44] achieves the highest GmeScore, indicating that the generated videos are better aligned with the provided instructions. Kling [43], on

### Table 3: Quantitative Comparison among Different Methods for the Open-Domain Subject-toVideo task. Total score is the normalized weighted sum of other scores. “↑” higher is better.

Method Venue Total Score↑ Aesthetics↑ Motion↑ FaceSim↑ GmeScore↑ NexusScore↑ NaturalScore↑

Vidu2.0 [5] Closed-Source 47.59% 41.47% 13.52% 35.11% 67.57% 43.55% 71.44% Pika2.1 [44] Closed-Source 48.88% 46.87% 24.70% 30.80% 69.21% 45.41% 69.79%

Kling1.6 [43] Closed-Source 54.46% 44.60% 41.60% 40.10% 66.20% 45.92% 79.06%

VACE-P1.3B [40] Open-Source 43.95% 47.27% 12.03% 16.58% 71.38% 40.04% 70.56% VACE-1.3B [40] Open-Source 45.53% 48.24% 18.83% 20.58% 71.26% 37.95% 71.78% VACE-14B [40] Open-Source 52.87% 47.21% 15.02% 55.09% 67.27% 44.20% 72.78%

Phantom-1.3B [55] Open-Source 50.71% 46.67% 14.29% 48.55% 69.43% 42.44% 70.26% Phantom-14B [55] Open-Source 52.32% 46.39% 33.42% 51.48% 70.65% 37.43% 68.66%

SkyReels-A2-P14B [21] Open-Source 49.61% 39.40% 25.60% 45.95% 64.54% 43.77% 67.22%

### Table 4: Quantitative Comparison among Different Methods for the Human-Domain Subjectto-Video task. Total score is the normalized weighted sum of other scores. “↑” higher is better.

###### Method Venue Domain Total Score↑ Aesthetics↑ Motion↑ FaceSim↑ GmeScore↑ NaturalScore↑

Vidu2.0 [5] Closed-Source Open-Domain 51.11% 47.33% 14.80% 38.50% 70.42% 71.99% Pika2.1 [44] Closed-Source Open-Domain 52.56% 52.39% 28.94% 29.41% 75.03% 72.53%

Kling1.6 [43] Closed-Source Open-Domain 59.13% 50.94% 50.55% 41.02% 67.79% 78.28%

VACE-P1.3B [40] Open-Source Open-Domain 46.28% 51.45% 8.78% 19.98% 73.27% 70.89% VACE-1.3B [40] Open-Source Open-Domain 49.02% 53.18% 16.87% 22.29% 73.61% 73.00% VACE-14B [40] Open-Source Open-Domain 58.57% 52.78% 11.76% 64.65% 69.53% 74.33%

Phantom-1.3B [55] Open-Source Open-Domain 53.64% 50.80% 14.14% 46.30% 72.17% 71.67% Phantom-14B [55] Open-Source Open-Domain 58.69% 49.14% 41.24% 55.02% 72.55% 68.33%

SkyReels-A2-P14B [21] Open-Source Open-Domain 54.27% 39.88% 31.98% 55.02% 63.63% 67.33% HunyuanCustom [33] Open-Source Open-Domain 55.85% 49.67% 15.13% 62.25% 69.78% 67.00%

Hailuo [87] Closed-Source Human-Domain 60.20% 52.75% 31.83% 57.79% 71.42% 74.52%

ConsisID [113] Open-Source Human-Domain 52.97% 41.76% 38.12% 43.14% 72.03% 64.67% Concat-ID-CogVideoX [123] Open-Source Human-Domain 53.32% 44.13% 31.76% 43.83% 73.67% 66.44% Concat-ID-Wan-AdaLN [123] Open-Source Human-Domain 53.18% 43.13% 17.19% 50.05% 71.90% 69.44%

FantasyID [120] Open-Source Human-Domain 49.80% 45.60% 23.48% 32.42% 72.68% 68.11% EchoVideo [97] Open-Source Human-Domain 54.52% 39.93% 35.76% 48.57% 68.40% 69.22%

VideoMaker [103] Open-Source Human-Domain 52.31% 31.76% 50.09% 76.45% 45.28% 47.08% ID-Animator [29] Open-Source Human-Domain 43.37% 42.03% 33.54% 31.56% 52.91% 54.03%

Ours † - Human-Domain 51.67% 41.30% 20.83% 47.64% 72.12% 65.42% Ours ‡ - Human-Domain 52.97% (+1.30%) 41.86% (+0.56%) 22.77% (+1.94%) 49.51% (+1.87%) 72.35% (+0.23%) 66.80% (+1.38%)

the other hand, produces videos with higher fidelity and realism, securing the highest NexusScore and NaturalScore. While SkyReels-A2 [21] holds the high NexusScore among open-source models, its relatively low NaturalScore suggests the presence of a copy-paste issue. VACE-1.3B and VACE14B [40] achieves superior generation quality across the board compared to the VACE-P1.3B [40] by scaling both the parameter size and the dataset. In the human-domain S2V task, proprietary models outperform open-domain models in terms of preserving human identity, particularly Hailuo [87], which achieves the highest Total Score of 60.20%. Furthermore, NaturalScore reveals that open-source models such as ConsisID [113] and Concat-ID [123], despite having relatively strong FaceSim, suffer from significant copy-paste issues. In contrast, EchoVideo [97] achieves the highest score among the open-source human-domain models. Since HunyuanCustom [33] only released the single-subject version as open source, we additionally provide results for the single-domain scenario, as presented in Table 5. Notably, although HunyuanCustom [33] achieves high subject fidelity, its generated styles tend to exhibit artificial characteristics, resulting in less realistic outputs.

Qualitative Evaluation. Next, we randomly select three test data for qualitative analysis, as shown in Figures 6, 7, and 8. Overall, closed-source models exhibit a clear advantage in terms of overall capability (e.g., Kling [43]). Open-source models, represented by Phantom [55] and VACE [40], are closing this gap; however, both models share the following three common issues: (1) Poor generalization: Fidelity is low for certain subjects. For instance, in case 2 of Figure 6, Kling [43] generates an incorrect playground background, while VACE [40], Phantom [55], and SkyReels-A2 [21] produce low-fidelity humans and birds; (2) Copy-paste issues: In Figure 7, SkyReels-A2 [21] and VACE [40] incorrectly replicate the expression, lighting, or pose from the reference image into the generated video, resulting in unnatural output; (3) Inadequate human fidelity: In case 2 of Figure 6, only Kling [43] maintains human identity in the first half of the video, while the other models lose significant facial details throughout the video. Figure 7 shows that all models fail to accurately render the profile of the individual. Additionally, we observe that (1) As the number of reference images increases, fidelity gradually decreases; (2) the initial frames may blurry or directly copied; (3) fidelity gradually declines over time. For more details, please refer to the Appendix B.4.

Human Preference. Then, we validate the effectiveness of metrics through manual cross-validation. Sixty generated videos corresponding to the prompts are randomly selected, and 173 participants are invited to vote, yielding evaluation results. To improve user satisfaction, we employ a binary classification questionnaire format. Figure 9(a) illustrates the correlation between the automatic metrics

### Table 5: Quantitative Comparison among Different Methods for the Single-Domain Subject-toVideo task. Total score is the normalized weighted sum of other scores. “↑” higher is better.

###### Method Venue Total Score↑ Aesthetics↑ Motion↑ FaceSim↑ GmeScore↑ NexusScore↑ NaturalScore↑

Vidu2.0 [5] Closed-Source 48.67% 34.78% 24.40% 36.20% 65.56% 45.20% 72.60% Pika2.1 [44] Closed-Source 48.93% 38.64% 31.90% 32.94% 62.19% 47.34% 70.60%

Kling1.6 [43] Closed-Source 53.12% 35.63% 36.40% 39.26% 61.99% 48.24% 81.40%

VACE-P1.3B [40] Open-Source 44.28% 42.58% 18.00% 18.02% 65.93% 36.26% 76.00% VACE-1.3B [40] Open-Source 47.33% 41.81% 33.78% 22.38% 65.35% 38.52% 76.00% VACE-14B [40] Open-Source 58.00% 41.30% 35.54% 64.65% 58.55% 51.33% 77.33%

Phantom-1.3B [55] Open-Source 49.95% 42.98% 19.30% 44.03% 65.61% 37.78% 76.00% Phantom-14B [55] Open-Source 53.17% 47.46% 41.55% 51.82% 70.07% 35.35% 69.35%

SkyReels-A2-P14B [21] Open-Source 51.64% 33.83% 21.60% 54.42% 61.93% 48.63% 70.60% HunyuanCustom [33] Open-Source 51.64% 34.08% 26.83% 55.93% 54.31% 50.75% 68.66%

||[Figure 115]|
|---|
<br><br>|[Figure 116]|
|---|
<br><br>Kling1.6 Phantom<br><br>1.3B Pika2.1<br><br>SkyReels-A2<br><br>P14B<br><br>VACE<br><br>P1.3B Vidu2.0<br><br>a man sitting on<br><br>the grass in the<br><br>park, a bird flying around him.<br><br>|[Figure 117]|
|---|
<br><br>|[Figure 118]|
|---|
|[Figure 119]|
|[Figure 120]|
<br><br>|[Figure 121]|
|---|
<br><br>|[Figure 122]|
|---|
|[Figure 123]|
|[Figure 124]|
<br><br>|[Figure 125]|
|---|
|[Figure 126]|
|[Figure 127]|
<br><br>|[Figure 128]|
|---|
<br><br>|[Figure 129]|
|---|
<br><br>|[Figure 130]|
|---|
<br><br>|[Figure 131]|
|---|
<br><br>|[Figure 132]|
|---|
<br><br>|[Figure 133]|
|---|
<br><br>|[Figure 134]|
|---|
<br><br>|[Figure 135]|
|---|
<br><br>|[Figure 136]|
|---|
<br><br>|[Figure 137]|
|---|
<br><br>|[Figure 138]|
|---|
<br><br>|[Figure 139]|
|---|
<br><br>|[Figure 140]|
|---|
<br><br>|[Figure 141]|
|---|
<br><br>VACE<br><br>14B VACE<br><br>1.3B<br><br>|[Figure 142]|
|---|
<br><br>|[Figure 143]|
|---|
|[Figure 144]|
|[Figure 145]|
<br><br>|[Figure 146]|
|---|
|[Figure 147]|
|[Figure 148]|
<br><br>|[Figure 149]|
|---|
|[Figure 150]|
|[Figure 151]|
<br><br>|[Figure 152]|
|---|
|[Figure 153]|
|[Figure 154]|
<br><br>|[Figure 155]|
|---|
<br><br>|[Figure 156]|
|---|
<br><br>|[Figure 157]|
|---|
<br><br>|[Figure 158]|
|---|
<br><br>|[Figure 159]|
|---|
<br><br>|[Figure 160]|
|---|
<br><br>|[Figure 161]|
|---|
<br><br>|[Figure 162]|
|---|
<br><br>|[Figure 163]|
|---|
<br><br>|[Figure 164]|
|---|
<br><br>|[Figure 165]|
|---|
<br><br>Kling1.6 Phantom<br><br>1.3B Pika2.1<br><br>Phantom<br><br>14B VACE<br><br>P1.3B Vidu2.0<br><br>The video begins with a close-up of a teapot sitting on a cozy<br><br>kitchen table, steam<br><br>rising gently from its spout. The camera zooms in to capture the swirling steam, which moves gracefully in the warm light. A hand reaches into the frame, lifting the teapot’s lid slightly, causing the steam to intensify for a ...<br><br>|[Figure 166]|
|---|
<br><br>VACE<br><br>14B VACE<br><br>1.3B<br><br>|[Figure 167]|
|---|
<br><br>|[Figure 168]|
|---|
<br><br>|[Figure 169]|
|---|
<br><br>|[Figure 170]|
|---|
<br><br>|[Figure 171]|
|---|
<br><br>|[Figure 172]|
|---|
<br><br>|[Figure 173]|
|---|
<br><br>|[Figure 174]|
|---|
<br><br>|[Figure 175]|
|---|
<br><br>|[Figure 176]|
|---|
<br><br>|[Figure 177]|
|---|
<br><br>|[Figure 178]|
|---|
<br><br>|[Figure 179]|
|---|
<br><br>|[Figure 180]|
|---|
<br><br>|[Figure 181]|
|---|
<br><br>|[Figure 182]|
|---|
<br><br>SkyReels-A2<br><br>P14B<br><br>Phantom<br><br>14B<br><br>|[Figure 183]|
|---|
<br><br>|[Figure 184]|
|---|
<br><br>|[Figure 185]|
|---|
<br><br>|[Figure 186]|
|---|
<br><br>|[Figure 187]|
|---|
<br><br>|[Figure 188]|
|---|
<br><br>|[Figure 189]|
|---|
<br><br>|[Figure 190]|
|---|
|
|---|

P1.3B Vidu2.0

P1.3B Vidu2.0

Kling1.6Pika2.1 VACE

Kling1.6Pika2.1 VACE

14B VACE

14B VACE

1.3B

1.3B

VACE

VACE

- Figure 6: Qualitative Comparison among Different Methods for the Open-Domain Subject-toVideo task. Existing methods handle non-human entities better than human identities, and perform better with single subject compared to multiple subjects.

||[Figure 191]|
|---|
<br><br>The video features a man sitting in the driver's seat of a car. he is wearing glasses and a dark-colored dress, and his hair is neatly styled. The interior of the car appears to be modern, with a light-colored dashboard and a steering ...<br><br>Phantom<br><br>1.3B Phantom<br><br>14B Kling1.6<br><br>|[Figure 192]|[Figure 193]|[Figure 194]|[Figure 195]|
|---|---|---|---|
|[Figure 196]|[Figure 197]|[Figure 198]|[Figure 199]|
|[Figure 200]|[Figure 201]|[Figure 202]|[Figure 203]|
|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|
|[Figure 208]|[Figure 209]|[Figure 210]|[Figure 211]|
|[Figure 212]|[Figure 213]|[Figure 214]|[Figure 215]|
|[Figure 216]|[Figure 217]|[Figure 218]|[Figure 219]|
|[Figure 220]|[Figure 221]|[Figure 222]|[Figure 223]|
|[Figure 224]|[Figure 225]|[Figure 226]|[Figure 227]|
<br><br>VACE<br><br>1.3B Pika2.1<br><br>VACE<br><br>P1.3B Vidu2.0<br><br>VACE<br><br>14B<br><br>Concat-ID<br><br>CogVideoX EchoVideoFantasyIDConsisIDHailuo<br><br>|[Figure 228]| |[Figure 229]|[Figure 230]|[Figure 231]| |
|---|---|---|---|---|---|
|[Figure 232]| |[Figure 233]|[Figure 234]|[Figure 235]| |
|[Figure 236]| |[Figure 237]|[Figure 238]|[Figure 239]| |
|[Figure 240]| |[Figure 241]|[Figure 242]|[Figure 243]| |
|[Figure 244]| |[Figure 245]|[Figure 246]|[Figure 247]| |
|[Figure 248]| |[Figure 249]|[Figure 250]|[Figure 251]| |
|[Figure 252]| |[Figure 253]|[Figure 254]|[Figure 255]| |
|[Figure 256]||[Figure 257]|
|---|
|[Figure 258]|
<br><br>|[Figure 259]|
|---|
|[Figure 260]|
<br><br>|[Figure 261]|
|---|
|[Figure 262]|
<br><br>|[Figure 263]|
|---|
|[Figure 264]|
<br><br>|[Figure 265]|
|---|
|[Figure 266]|
| | | |[Figure 267]|
|[Figure 268]| | | | |[Figure 269]|
<br><br>Hunyuan<br><br>Custom<br><br>VideoMakerID-Animator<br><br>SkyReels-A2<br><br>P14B<br><br>Concat-ID<br><br>CogVideoX|
|---|

- Figure 7: Qualitative Comparison among Different Methods for the Human-Domain Subject-toVideo task. They are unable to generate consistent side profiles and suffer from copy-paste issues.

and human perception. It is evident that the three proposed metrics—Nexus Score, NaturalScore, and GmeScore—align with human perception and accurately reflect the subject consistency, subject

||[Figure 270]|
|---|
<br><br>Kling1.6 Phantom<br><br>1.3B Phantom<br><br>14B VACE<br><br>P1.3B<br><br>The video begins with a close-up of a t-shirt<br><br>laid out on a sunny<br><br>rock, the fabric gently catching the light of the sun. The camera zooms in to capture the texture of the shirt as it rests on the rock, with ...<br><br>VACE<br><br>14B VACE<br><br>1.3B Vidu2.0<br><br>|[Figure 271]|
|---|
<br><br>|[Figure 272]|
|---|
<br><br>|[Figure 273]|
|---|
<br><br>|[Figure 274]|
|---|
<br><br>|[Figure 275]|
|---|
<br><br>|[Figure 276]|
|---|
<br><br>|[Figure 277]|
|---|
<br><br>|[Figure 278]|
|---|
<br><br>Pika2.1<br><br>|[Figure 279]|
|---|
<br><br>|[Figure 280]|
|---|
<br><br>|[Figure 281]|
|---|
<br><br>|[Figure 282]|
|---|
<br><br>|[Figure 283]|
|---|
<br><br>|[Figure 284]|
|---|
<br><br>|[Figure 285]|
|---|
<br><br>|[Figure 286]|
|---|
<br><br>|[Figure 287]|
|---|
<br><br>|[Figure 288]|
|---|
<br><br>|[Figure 289]|
|---|
<br><br>|[Figure 290]|
|---|
<br><br>|[Figure 291]|
|---|
<br><br>|[Figure 292]|
|---|
<br><br>|[Figure 293]|
|---|
<br><br>|[Figure 294]|
|---|
<br><br>Hunyuan<br><br>Custom<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>Vidu2.0Pika2.1<br><br>The video features a man standing at an easel, focused intently as his brush dances across the canvas. His expression is one of deep concentration, with a hint of satisfaction as each brushstroke adds color and form ...<br><br>VACE<br><br>14B VACE<br><br>1.3B Kling1.6<br><br>|[Figure 303]|
|---|
<br><br>|[Figure 304]|
|---|
<br><br>|[Figure 305]|
|---|
<br><br>|[Figure 306]|
|---|
<br><br>|[Figure 307]|
|---|
<br><br>|[Figure 308]|
|---|
<br><br>|[Figure 309]|
|---|
<br><br>|[Figure 310]|
|---|
<br><br>|[Figure 311]|
|---|
<br><br>|[Figure 312]|
|---|
<br><br>|[Figure 313]|
|---|
<br><br>|[Figure 314]|
|---|
<br><br>|[Figure 315]|
|---|
<br><br>|[Figure 316]|
|---|
<br><br>|[Figure 317]|
|---|
<br><br>|[Figure 318]|
|---|
<br><br>|[Figure 319]|
|---|
<br><br>VACE<br><br>P1.3B Phantom<br><br>|[Figure 320]|
|---|
<br><br>|[Figure 321]|
|---|
<br><br>|[Figure 322]|
|---|
<br><br>|[Figure 323]|
|---|
<br><br>1.3B<br><br>|[Figure 324]|
|---|
<br><br>|[Figure 325]|
|---|
<br><br>|[Figure 326]|
|---|
<br><br>|[Figure 327]|
|---|
<br><br>SkyReels-A2<br><br>P14B Hunyuan<br><br>Custom<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>[Figure 331]<br><br>[Figure 332]<br><br>[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]<br><br>[Figure 343]<br><br>SkyReels-A2<br><br>P14B<br><br>Phantom<br><br>14B<br><br>|[Figure 344]|
|---|
<br><br>|[Figure 345]|
|---|
<br><br>|[Figure 346]|
|---|
<br><br>|[Figure 347]|
|---|
<br><br>|[Figure 348]|
|---|
<br><br>|[Figure 349]|
|---|
<br><br>|[Figure 350]|
|---|
<br><br>|[Figure 351]|
|---|
|
|---|

1.3B Vidu2.0Pika2.1

Vidu2.0Pika2.1 VACE

Kling1.6 Phantom

1.3B Kling1.6

14B VACE

VACE

P1.3B VACE

P1.3B Phantom

14B VACE

14B VACE

1.3B Phantom

1.3B

Phantom

14B

SkyReels-A2

SkyReels-A2

P14B

P14B Hunyuan

Hunyuan

Custom

Custom

### Figure 8: Qualitative Comparison among Different Methods for the Single-Domain Subject-toVideo task. Existing models perform better on single-subject than multi-subject tasks.

|Ours†<br><br>|[Figure 352]<br><br>Copy-Paste: Incorrect replication of black-and-white style|
|---|
<br><br>|[Figure 353]|
|---|
<br><br>|[Figure 354]|
|---|
<br><br>|[Figure 355]|
|---|
<br><br>0%<br><br>10%<br><br>20%<br><br>30%<br><br>40%<br><br>50%<br><br>60%<br><br>70%<br><br>80%<br><br>90%<br><br>NexusScore NaturalScore GmeScore FaceSim AestheticScore MotionScore<br><br>HumanPreference<br><br>| |
|---|
<br><br>[Figure 356]<br><br>(a) Verification of Automatic Metric (b) Verification of the Proposed Dataset<br><br>The video features a man with a rugged beard, wearing a leather jacket, riding a<br><br>vintage motorcycle along a desert highway. His expression is focused, eyes<br><br>narrowed slightly against the wind, as the setting sun casts a warm glow ...<br><br>Ours‡<br><br>|[Figure 357]|
|---|
<br><br>|[Figure 358]|
|---|
<br><br>|[Figure 359]|
|---|
<br><br>|[Figure 360]|
|---|
|
|---|

### Figure 9: (a) Alignment between Automatic Metrics and Human Perception. The proposed metrics are comparable to other metrics [17, 6, 16] in terms of human preference. (2) Validation of ConsisID-Nexu-5M with † and without ‡ Nexus Data. Training are based on ConsisID [113].

naturalness, and text relevance. Moreover, the proposed metrics are comparable to other metrics [17, 6, 16] in terms of human preference. Further details can be found in the Appendix D.6.

Validation of OpenS2V-5M. Finally, to evaluate the effectiveness and robustness of OpenS2V-5M, we fine-tune a model initialized with Wan2.1 1.3B weights [89] using the ConsisID method [113], employing only MSE loss and omitting mask loss. Given computational constraints, we randomly use 300k samples from OpenS2V-5M, focusing solely on single human identity during training. The results, presented in Figure 9(b) and Table 7, demonstrate that our dataset successfully converts a text-to-video model into a subject-to-video model, thus validating the proposed dataset and its data collection pipeline, especially the Nexus Data plays a crucial role. Since the model is not fully trained, it has not yet achieved optimal performance and is intended for verification purposes only.

## 6 Conclusion

In this paper, we present OpenS2V-Eval, the first benchmark specifically designed for evaluating subject-to-video (S2V) generation. This benchmark addresses the limitations of existing benchmarks, which are primarily derived from text-to-video models and overlook crucial aspects such as subject consistency and subject naturalness. Additionally, we present three new automated metrics aligned with humans—NexusScore, NaturalScore, and GmeScore. Furthermore, we introduce OpenS2V-5M, the first open-source million-scale S2V dataset, which not only includes regular subject-text-video triples but also incorporates Nexus Data constructed using GPT-Image-1 and cross-video associations, thus promoting further research within the community and resolving the three core issues of S2V.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Nouar AlDahoul and Yasir Zaki. Detecting ai-generated images using vision transformers: A robust approach for safeguarding visual media integrity. Available at SSRN, 2024.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021.
- [5] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.
- [6] Gary Bradski, Adrian Kaehler, et al. Opencv. Dr. Dobb’s journal of software tools, 3(2), 2000.
- [7] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. arXiv preprint arXiv:2412.18597, 2024.
- [8] Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Xiao Yang, and Mohammad Soleymani. Magicdance: Realistic human dance video generation with motions & facial expressions transfer. arXiv preprint arXiv:2311.12052, 2023.
- [9] Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. arXiv preprint arXiv:2311.12052, 2023.
- [10] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023.
- [11] Liuhan Chen, Zongjian Li, Bin Lin, Bin Zhu, Qian Wang, Shenghai Yuan, Xing Zhou, Xinhua Cheng, and Li Yuan. Od-vae: An omni-dimensional video compressor for improving latent video diffusion model. arXiv preprint arXiv:2409.01199, 2024.
- [12] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. arXiv preprint arXiv:2402.19479, 2024.
- [13] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multisubject open-set personalization in video generation. arXiv preprint arXiv:2501.06187, 2025.
- [14] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774, 2024.
- [15] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yoloworld: Real-time open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16901–16911, 2024.
- [16] christophschuhmann. improved-aesthetic-predictor. improved-aesthetic-predictor Lab, 2024.

- [17] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, pages 4690–4699, 2019.
- [18] Yufan Deng, Xun Guo, Yizhi Wang, Jacob Zhiyuan Fang, Angtian Wang, Shenghai Yuan, Yiding Yang, Bo Liu, Haibin Huang, and Chongyang Ma. Cinema: Coherent multi-subject video generation via mllm-based guidance. arXiv preprint arXiv:2503.10391, 2025.
- [19] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983, 2025.
- [20] Weichen Fan, Chenyang Si, Junhao Song, Zhenyu Yang, Yinan He, Long Zhuo, Ziqi Huang, Ziyue Dong, Jingwen He, Dongwei Pan, et al. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025.
- [21] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen, Yang Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025.
- [22] Zhengcong Fei, Debang Li, Di Qiu, Changqian Yu, and Mingyuan Fan. Ingredients: Blending custom photos with video diffusion transformers. arXiv preprint arXiv:2501.01790, 2025.
- [23] Chaoran Feng, Wangbo Yu, Xinhua Cheng, Zhenyu Tang, Junwu Zhang, Li Yuan, and Yonghong Tian. Ae-nerf: Augmenting event-based neural radiance fields for non-ideal conditions and larger scene. arXiv preprint arXiv:2501.02807, 2025.
- [24] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [25] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [26] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024.
- [27] Junjie He, Yifeng Geng, and Liefeng Bo. Uniportrait: A unified framework for identitypreserving single-and multi-human image personalization. arXiv preprint arXiv:2408.05939, 2024.
- [28] Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, et al. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv preprint arXiv:2406.15252, 2024.
- [29] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, Man Zhou, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.
- [30] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [31] Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024.
- [32] Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, and Liefeng Bo. Animate anyone 2: High-fidelity character image animation with environment affordance. arXiv preprint arXiv:2502.06145, 2025.
- [33] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512, 2025.

- [34] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In CVPR, pages 5901–5910, 2020.
- [35] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025.
- [36] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023.
- [37] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024.
- [38] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. arXiv preprint arXiv:2503.16418, 2025.
- [39] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6689–6700, 2024.
- [40] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.
- [41] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [42] Tengchuan Kou, Xiaohong Liu, Zicheng Zhang, Chunyi Li, Haoning Wu, Xiongkuo Min, Guangtao Zhai, and Ning Liu. Subjective-aligned dateset and metric for text-to-video quality assessment. arXiv preprint arXiv:2403.11956, 2024.
- [43] Kwai. Keling. Kwai, 2024.
- [44] Pika Lab. Pika-2.0 lab discord server. Pika Lab, 2024.
- [45] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. arXiv preprint arXiv:2412.00115, 2024.
- [46] Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Fuli Feng. Improving synthetic image detection towards generalization: An image transformation perspective. arXiv preprint arXiv:2408.06741, 2024.
- [47] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR, pages 8640–8650, 2024.
- [48] Zongjian Li, Bin Lin, Yang Ye, Liuhan Chen, Xinhua Cheng, Shenghai Yuan, and Li Yuan. Wf-vae: Enhancing video vae by wavelet-driven energy flow for latent video diffusion model. arXiv preprint arXiv:2411.17459, 2024.
- [49] Feng Liang, Haoyu Ma, Zecheng He, Tingbo Hou, Ji Hou, Kunpeng Li, Xiaoliang Dai, Felix Juefei-Xu, Samaneh Azadi, Animesh Sinha, et al. Movie weaver: Tuning-free multi-concept video personalization with anchored prompts. arXiv preprint arXiv:2502.07802, 2025.
- [50] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [51] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. EMNLP, 2024.

- [52] Zongyu Lin, Wei Liu, Chen Chen, Jiasen Lu, Wenze Hu, Tsu-Jui Fu, Jesse Allardice, Zhengfeng Lai, Liangchen Song, Bowen Zhang, et al. Stiv: Scalable text and image conditioned video generation. arXiv preprint arXiv:2412.07730, 2024.
- [53] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [54] Dongyang Liu, Shicheng Li, Yutong Liu, Zhen Li, Kai Wang, Xinyue Li, Qi Qin, Yufei Liu, Yi Xin, Zhongyu Li, et al. Lumina-video: Efficient and flexible video generation with multi-scale next-dit. arXiv preprint arXiv:2502.06782, 2025.
- [55] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025.
- [56] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.
- [57] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22139–22149, 2024.
- [58] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. Advances in Neural Information Processing Systems, 36, 2024.
- [59] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.
- [60] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [61] Xuran Ma, Yexin Liu, Yaofu Liu, Xianfeng Wu, Mingzhe Zheng, Zihao Wang, Ser-Nam Lim, and Harry Yang. Model reveals what to cache: Profiling-based feature reuse for video diffusion models. arXiv preprint arXiv:2504.03140, 2025.
- [62] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4117–4125, 2024.
- [63] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, Heung-Yeung Shum, Wei Liu, et al. Follow-your-click: Open-domain regional image animation via short prompts. arXiv preprint arXiv:2403.08268, 2024.
- [64] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. arXiv preprint arXiv:2406.01900, 2024.
- [65] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic-me: Identity-specific video customized diffusion. arXiv preprint arXiv:2402.09368, 2024.
- [66] Abdellahi El Moustapha. Multi-task image classifier. https://huggingface.co/Abdu07/ multitask-model, 2025.
- [67] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.

- [68] Yasir Zaki Nouar AlDahoul. Nyuad ai generated images detector.
- [69] Yatian Pang, Bin Zhu, Bin Lin, Mingzhe Zheng, Francis EH Tay, Ser-Nam Lim, Harry Yang, and Li Yuan. Dreamdance: Animating human images by enriching 3d geometry cues from 2d poses. arXiv preprint arXiv:2412.00397, 2024.
- [70] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in 200 k. arXiv preprint arXiv:2503.09642, 2025.
- [71] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855, 2024.
- [72] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [73] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [74] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. arXiv preprint arXiv:2204.06125, 2022.
- [75] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pages 8821–8831, 2021.
- [76] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [77] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023.
- [78] Sand-AI. Magi-1: Autoregressive video generation at scale, 2025.
- [79] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025.
- [80] Yujun Shi, Chuhui Xue, Jun Hao Liew, Jiachun Pan, Hanshu Yan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8839–8849, 2024.
- [81] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [82] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.
- [83] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image animation with enhanced motion representation. arXiv preprint arXiv:2410.10306, 2024.

- [84] Zhenyu Tang, Junwu Zhang, Xinhua Cheng, Wangbo Yu, Chaoran Feng, Yatian Pang, Bin Lin, and Li Yuan. Cycle3d: High-quality and consistent image-to-3d generation via generationreconstruction cycle. arXiv preprint arXiv:2407.19548, 2024.
- [85] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.
- [86] Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024.
- [87] Hailuo Team. Hailuo. Hailuo Lab, 2024.
- [88] PaddleOCR Team. Paddleocr. PaddleOCR Lab, 2024.
- [89] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [90] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [91] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. arXiv preprint arXiv:2410.08260, 2024.
- [92] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.
- [93] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-tovideo diffusion models. arXiv preprint arXiv:2403.06098, 2024.
- [94] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023.
- [95] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.
- [96] Yiping Wang, Xuehai He, Kuan Wang, Luyao Ma, Jianwei Yang, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Is your world simulator a good story presenter? a consecutive events-based benchmark for future long video generation. arXiv preprint arXiv:2412.16211, 2024.
- [97] Jiangchuan Wei, Shiyue Yan, Wenfeng Lin, Boyuan Liu, Renjie Chen, and Mingyu Guo. Echovideo: Identity-preserving human video generation by multimodal feature fusion. arXiv preprint arXiv:2501.13452, 2025.
- [98] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In CVPR, pages 6537–6549, 2024.
- [99] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154, 2023.
- [100] Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang, Yixiao Ge, Xiaodong Cun, David Junhao Zhang, Jia-Wei Liu, Yuchao Gu, Rui Zhao, et al. Towards a better metric for text-to-video generation. arXiv preprint arXiv:2401.07781, 2024.

- [101] Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang, Yixiao Ge, Xiaodong Cun, David Junhao Zhang, Jia-Wei Liu, Yuchao Gu, Rui Zhao, et al. Towards a better metric for text-to-video generation. arXiv preprint arXiv:2401.07781, 2024.
- [102] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. arXiv preprint arXiv:2406.17758, 2024.
- [103] Tao Wu, Yong Zhang, Xiaodong Cun, Zhongang Qi, Junfu Pu, Huanzhang Dou, Guangcong Zheng, Ying Shan, and Xi Li. Videomaker: Zero-shot customized video generation with the inherent force of video diffusion models. arXiv preprint arXiv:2412.19645, 2024.
- [104] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.
- [105] Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv preprint arXiv:2405.18991, 2024.
- [106] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. MSR-VTT: A large video description dataset for bridging video and language. CVPR, pages 5288–5296, 2016.
- [107] Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Weidi Xie. A sanity check for ai-generated image detection. arXiv preprint arXiv:2406.19435, 2024.
- [108] Yuhang Yang, Ke Fan, Shangkun Sun, Hongxiang Li, Ailing Zeng, FeiLin Han, Wei Zhai, Wei Liu, Yang Cao, and Zheng-Jun Zha. Videogen-eval: Agent-based system for video generation evaluation. arXiv preprint arXiv:2503.23452, 2025.
- [109] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [110] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [111] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast causal video generators. arXiv preprint arXiv:2412.07772, 2024.
- [112] Wangbo Yu, Chaoran Feng, Jiye Tang, Xu Jia, Li Yuan, and Yonghong Tian. Evagaussians: Event stream assisted gaussian splatting from blurry images. arXiv preprint arXiv:2405.20224, 2024.
- [113] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identity-preserving text-to-video generation by frequency decomposition. arXiv preprint arXiv:2411.17440, 2024.
- [114] Shenghai Yuan, Jinfa Huang, Yujun Shi, Yongqi Xu, Ruijie Zhu, Bin Lin, Xinhua Cheng, Li Yuan, and Jiebo Luo. Magictime: Time-lapse video generation models as metamorphic simulators. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [115] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024.
- [116] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and HeungYeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022.
- [117] Lei Zhang, Fangxun Shu, Sucheng Ren, Bingchen Zhao, Hao Jiang, and Cihang Xie. Compress & align: Curating image-text data with human knowledge. arXiv preprint arXiv:2312.06726, 2023.

- [118] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.
- [119] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855, 2024.
- [120] Yunpeng Zhang, Qiang Wang, Fan Jiang, Yaqi Fan, Mu Xu, and Yonggang Qi. Fantasyid: Face knowledge enhanced id-preserving video generation. arXiv preprint arXiv:2502.13995, 2025.
- [121] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.
- [122] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, et al. Videogen-of-thought: A collaborative framework for multi-shot video generation. arXiv preprint arXiv:2412.02259, 2024.
- [123] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-id: Towards universal identity-preserving video synthesis. arXiv preprint arXiv:2503.14151, 2025.
- [124] Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv:2410.15458, 2024.
- [125] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. Celebv-hq: A large-scale video facial attributes dataset. In European conference on computer vision, pages 650–667. Springer, 2022.

## Paper Appendix for OpenS2V-Nexus: A Ultra-Scale Dataset and Benchmark for Subject-Consistent Video Generation

- A Related Works: Subject-Consistency Video Generation Models 1
- B More Details of OpenS2V-Eval 2 B.1 Comparison with Existing Metircs for Subject Consistency and Text Relevance . . 2 B.2 Comparison with Existing Metrics for Subject Naturalness . . . . . . . . . . . . . 3 B.3 Visual Reference of Different Metrics . . . . . . . . . . . . . . . . . . . . . . . . 3 B.4 More Qualitative Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 B.5 Guideline for Model Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- C More Details of OpenS2V-5M 5

- C.1 Additional Details of Subject-Driven Processing . . . . . . . . . . . . . . . . . . . 5
- C.2 Additional Details of Dataset Statistics . . . . . . . . . . . . . . . . . . . . . . . . 5
- C.3 Further Verification on OpenS2V-5M . . . . . . . . . . . . . . . . . . . . . . . . . 5
- C.4 Samples of Collected Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- D More Details of Experiment 6

- D.1 Details of Resource . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- D.2 Details of Evaluation Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- D.3 Additional Details of Evaluation Settings . . . . . . . . . . . . . . . . . . . . . . . 9
- D.4 Additional Details of Implementations . . . . . . . . . . . . . . . . . . . . . . . . 9
- D.5 Additional Details of Metircs Normalization . . . . . . . . . . . . . . . . . . . . . 10
- D.6 Additional Details of Human Evaluation . . . . . . . . . . . . . . . . . . . . . . . 10
- D.7 Additional Details of Input Prompts . . . . . . . . . . . . . . . . . . . . . . . . . 11

- E Additional Statement 11

- E.1 Limitations and Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- E.2 Declaration of LLM Usage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- E.3 Potential Harms Caused by the Research Process . . . . . . . . . . . . . . . . . . 12
- E.4 Societal Impact and Potential Harmful Consequences . . . . . . . . . . . . . . . . 12
- E.5 Impact Mitigation Measures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

## A Related Works: Subject-Consistency Video Generation Models

Diffusion models are widely acknowledged for their remarkable generative capabilities [75, 74, 72, 63, 64, 62, 84, 112, 23], which have significantly advanced the development of subject-consistency generation models [38, 27, 26, 10]. Initially, researchers utilized tuning-based methods to generate consistent image content, such as DreamBooth [77], Lora [30], and Textual Inversion [24]. These methods integrate specific reference content into the training process through fine-tuning existing parameters, adding extra parameters, or modifying text embeddings. Later models, including MagicMe [65], MotionBooth [102], and DreamVideo [98], extended these approaches to video generation. However, since these methods require training on each new reference content before inference, their practical application is limited. To mitigate the high computational cost, tuning-free methods were

|(a) Quantitative Comparison with existing Metrics (b) Qualitative Comparison with existing Metrics<br><br>0%<br><br>10%<br><br>20%<br><br>30%<br><br>40%<br><br>50%<br><br>60%<br><br>70%<br><br>80%<br><br>DINO-I vs NexusScore CLIP-I vs NexusScore CLIP-T vs GmeScore<br><br>UserPreference<br><br>[Figure 361]<br><br>[Figure 362]<br><br>Other Metrics Ours<br><br>|[Figure 363]<br><br>[Figure 364]<br><br>[Figure 365]<br><br>[Figure 366]| |
|---|---|
|[Figure 367]<br><br>[Figure 368]| |
| | |
<br><br>|[Figure 369]<br><br>[Figure 370]|[Figure 371]<br><br>[Figure 372]|
|---|---|
|[Figure 373]<br><br>[Figure 374]| |
| | |
<br><br>|[Figure 375]<br><br>[Figure 376]|
|---|
<br><br>|[Figure 377]<br><br>[Figure 378]|
|---|
<br><br>|[Figure 379]<br><br>[Figure 380]|[Figure 381]<br><br>[Figure 382]|
|---|---|
| | |
<br><br>|[Figure 383]<br><br>[Figure 384]|[Figure 385]<br><br>[Figure 386]|
|---|---|
| | |
<br><br>DINO-I: , NexusScore:<br><br>DINO-I: , NexusScore:<br><br>CLIP-I: , NexusScore:<br><br>CLIP-I: , NexusScore:<br><br>CLIP-T: , GmeScore:<br><br>CLIP-T: , GmeScore:<br><br>DINO-I: 82.84%, NexusScore: 50.92%<br><br>DINO-I: 79.79%, NexusScore: 60.99%<br><br>CLIP-I: 84.33%, NexusScore: 20.68%<br><br>CLIP-I: 82.75%, NexusScore: 52.69%<br><br>CLIP-T: 38.18%, GmeScore: 63.82%<br><br>CLIP-T: 29.87%, GmeScore: 60.49%|
|---|

UserPreference

- Figure 10: Comparison with Existing Metircs for Subject Consistency and Text Relevance. The proposed automatic metricsalign more closely with human preferences compared to the commonly used DINO-I [116], CLIP-I [73], and CLIP-T [73] in existing S2V methods [40, 55, 37, 21].

||[Figure 387]|
|---|
<br><br>|[Figure 388]|
|---|
<br><br>|[Figure 389]|
|---|
<br><br>|[Figure 390]|
|---|
<br><br>|[Figure 391]|
|---|
<br><br>|[Figure 392]|
|---|
<br><br>|[Figure 393]|
|---|
<br><br>|[Figure 394]|
|---|
<br><br>|[Figure 395]|
|---|
<br><br>NexusScore: 60.00% NYUAD-ComNets: 99.83%<br><br>DualSIght: 43.36%<br><br>SAFE: 94.66% Qwen2.5-VL-7B: 80.00%<br><br>NexusScore: 60.00%<br><br>NYUAD-ComNets: 99.79%<br><br>DualSIght: 54.05% SAFE: 75.67% Qwen2.5-VL-7B: 80.00%<br><br>NexusScore: 60.00% NYUAD-ComNets: 99.87% DualSIght: 81.25%<br><br>SAFE: 93.51%<br><br>Qwen2.5-VL-7B: 80.00%<br><br>|
|---|

- Figure 11: Comparison with Existing Methods for Subject Naturalness. Existing AIGC anomaly detection models and multimodal models are both prone to misidentifying generated content as real.

introduced. A notable example is IP-Adapter [110], which leverages large datasets to train additional adapters for open-domain subject-consistency generation. However, due to its lower fidelity to human identity, InstantID [92] and PhotoMaker [47] developed human-domain subject-consistency generation models based on this approach. Similar to these image consistency techniques, IDAnimator [29] and ConsisID [113] achieved tuning-free Subject-to-Video (S2V) generation on UNet and DiT, respectively. Nevertheless, these approaches [123, 97, 22, 120] are confined to the human domain, limiting their broader applicability. Recent works, such as Phantom [55], VACE [40], and SkyReels-A2 [21], have demonstrated the ability to generate consistent multi-subject videos in the open domain [49, 13, 35], gradually narrowing the gap with commercial S2V models [43, 44, 87, 5]. However, a unified and comprehensive benchmark to assess the strengths and weaknesses of these models remains absent, and the lack of publicly released training data impedes further progress in this field. Therefore, we introduce OpenS2V-Eval and OpenS2V-5M, aimed at bridging this gap.

## B More Details of OpenS2V-Eval

### B.1 Comparison with Existing Metircs for Subject Consistency and Text Relevance

As previously noted, Alchemist-Bench [13], VACE-Benchmark [40], and A2 Bench [21] enable the evaluation of open-domain S2V. However, these evaluations are typically derived from VBench [37] and are predominantly limited to global, coarse-grained assessments. Specifically, they often rely on CLIP [73] or DINO [116] to calculate the similarity between text and images, both of which have been shown to exhibit poor robustness [100, 115, 58]. To substantiate these claims, we employ an

||[Figure 396]|
|---|
<br><br>|[Figure 397]|
|---|
<br><br>|[Figure 398]|
|---|
<br><br>|[Figure 399]|
|---|
<br><br>|[Figure 400]|
|---|
<br><br>|[Figure 401]|
|---|
<br><br>|[Figure 402]|
|---|
<br><br>|[Figure 403]|
|---|
<br><br>|[Figure 404]|
|---|
<br><br>|[Figure 405]|
|---|
<br><br>|[Figure 406]|
|---|
<br><br>|[Figure 407]|
|---|
<br><br>|[Figure 408]|
|---|
<br><br>|[Figure 409]|
|---|
<br><br>|[Figure 410]|
|---|
<br><br>|[Figure 411]|
|---|
<br><br>|[Figure 412]|
|---|
<br><br>|[Figure 413]|
|---|
<br><br>NaturalScore<br><br>40%≈ NaturalScore<br><br>60%≈ NaturalScore<br><br>80%≈<br><br>Aesthetic<br><br>15.07%≈ Aesthetic<br><br>47.16%≈ Aesthetic<br><br>73.23%≈<br><br>|[Figure 414]<br><br>a rooster and two hens standing on a grassy area in front of a wooden fence ...|
|---|
<br><br>|[Figure 415]|
|---|
<br><br>|[Figure 416]|
|---|
<br><br>|[Figure 417]<br><br>A woman stands on the bridge, wearing a black one-piece swimsuit with the Nike Air logo ...|
|---|
<br><br>|[Figure 418]|
|---|
<br><br>|[Figure 419]|
|---|
<br><br>|[Figure 420]|
|---|
<br><br>|[Figure 421]|
|---|
<br><br>|[Figure 422]|
|---|
<br><br>|[Figure 423]|
|---|
<br><br>|[Figure 424]|
|---|
<br><br>|[Figure 425]|
|---|
<br><br>|[Figure 426]<br><br>a man is watching a bird in the park.|
|---|
<br><br>|[Figure 427]|
|---|
<br><br>|[Figure 428]|
|---|
<br><br>GmeScore<br><br>42.72%≈ GmeScore<br><br>68.55%≈ GmeScore<br><br>76.56%≈ Motion<br><br>2.4%≈ Motion<br><br>55.19%≈ Motion<br><br>100%≈<br><br>|[Figure 429]<br><br>[Figure 430]| |
|---|---|
| | |
<br><br>|[Figure 431]|[Figure 432]|
|---|---|
| | |
<br><br>|[Figure 433]|
|---|
<br><br>|[Figure 434]|
|---|
<br><br>|[Figure 435]<br><br>[Figure 436]| |
|---|---|
| | |
<br><br>|[Figure 437]|
|---|
<br><br>|[Figure 438]|
|---|
<br><br>|[Figure 439]<br><br>[Figure 440]| |
|---|---|
| | |
<br><br>|[Figure 441]|
|---|
<br><br>|[Figure 442]|
|---|
<br><br>|[Figure 443]|
|---|
<br><br>|[Figure 444]|
|---|
<br><br>NexusScore<br><br>10.98%≈ NexusScore<br><br>71.76%≈ NexusScore<br><br>100%≈ FaceSim<br><br>13.70%≈ FaceSim<br><br>56.63%≈ FaceSim<br><br>88.53%≈<br><br>|[Figure 445]<br><br>[Figure 446]| |
|---|---|
| | |
<br><br>|[Figure 447]|
|---|
<br><br>|[Figure 448]|
|---|
<br><br>|[Figure 449]|
|---|
<br><br>|[Figure 450]|
|---|
<br><br>|[Figure 451]|
|---|
<br><br>|[Figure 452]<br><br>[Figure 453]| |
|---|---|
| | |
<br><br>|[Figure 454]|
|---|
<br><br>|[Figure 455]|
|---|
|
|---|

NaturalScore

NexusScore

GmeScore

10.98%≈ NexusScore

42.72%≈ GmeScore

40%≈ NaturalScore

71.76%≈ NexusScore

68.55%≈ GmeScore

60%≈ NaturalScore

76.56%≈ Motion

100%≈ FaceSim

80%≈

13.70%≈ FaceSim

15.07%≈ Aesthetic

Aesthetic

2.4%≈ Motion

56.63%≈ FaceSim

47.16%≈ Aesthetic

55.19%≈ Motion

88.53%≈

73.23%≈

100%≈

- Figure 12: Visual Reference for Varying Scores of Different Metircs. It is evident that the proposed NexusScore, NaturalScore, and GmeScore are highly correlated with human perception.

evaluation akin to human evaluation to gather user preferences for DINO-I, CLIP-I, and CLIP-T. Additionally, six samples are randomly selected for qualitative analysis, as illustrated in Figure 10. The results demonstrate that the proposed NexusScore and GmeScore offer greater accuracy in assessing subject consistency and text relevance compared to others. All higher scores are better.

### B.2 Comparison with Existing Metrics for Subject Naturalness

To evaluate whether a generated video is natural—meaning whether it complies with the laws of physics and common sense—a simple solution is to apply AIGC anomaly detection models [107, 46, 66, 2, 68], using the probability of the real label as the score. Alternatively, open-source multimodal large language models [3, 90, 51, 85] can be used for video scoring. However, we found that the former lacks accuracy, while the latter suffers from poor instruction-following performance and is prone to significant hallucinations. None of these methods perform as effectively as the NexusScore we propose, which is based on GPT-4o [1], as shown in Figure 11.

### B.3 Visual Reference of Different Metrics

We also provide visual samples of NexusScore, NaturalScore, GmeScore, FaceSim-Cur [113], AestheticScore [16], and MotionScore [6] with different scoring scales, as shown in Figure 12. It can be observed that all the metrics are consistent with human perception, especially the three proposed automatic metrics targeting subject consistency, subject naturalness, and text relevance.

### B.4 More Qualitative Analysis

We present further qualitative analysis, as illustrated in Figures 13, 22, 21, and 23. Both open-source and closed-source models encounter the following challenges:

Poor Generalization Although open-domain S2V models claim to support input from images of any category, they do not consistently produce satisfactory results. As illustrated in case 5 of Figure 21, while Kling [43] largely preserves the mole’s body shape, it loses the original fur color. Other models [44, 55, 21] entirely lose the reference subject information. Furthermore, as the number of reference images increases, the model’s ability to retain information progressively diminishes. This issue is particularly pronounced in open-source models [21, 40], as shown in cases 1–6 of Figure 21.

Copy-Paste Issue Existing models often inaccurately replicate the lighting, pose, expression, and other attributes from reference images directly onto generated videos, instead of generating content by learning the intrinsic features of the reference subjects. Although this may result in higher fidelity content, it generally fails to align with human perception and appears unnatural. As illustrated in Figure 13(c), the model directly places a face onto a person leaning against a pillar, creating an unnatural and visually awkward effect. This problem is particularly evident in generating human.

Inadequate Human Fidelity As demonstrated in Figures 21, 23, and 24, current models often face difficulties in preserving human identity as effectively as they preserve non-human entities.

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

###### (a)

[Figure 462]

First Frame Blurry

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

###### (b)

First Frame Copy

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

###### (c)

Copy-Paste

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

###### (d)

Consistency Fade

- Figure 13: Example of Common Issues faced by current Subject-to-Video Generation Models. These videos are generated by Kling [43] and SkyReels-A2 [21] for demonstration purposes only.

[Figure 481]

[Figure 482]

[Figure 483]

(a) Open-Domain Evaluation (b) Human-Domain Evaluation (c) Single-Domain Evaluation

### Figure 14: Visualization of all the Quantitative Results in OpenS2V-Eval.

While part of this issue can be attributed to human perception being more sensitive to facial changes, the primary cause lies in the models’ insufficient capabilities. This is also one of the reasons why human-domain models exist, such as ConsisID [113], EchoVideo [97] and Hailuo [87].

First Frame Blurry or Copy In addition to the three core issues outlined above, we also observe a noteworthy phenomenon in which the model directly replicates the reference image into the generated video, as illustrated in Figure 13(b), generated by Kling [43]. Furthermore, it is possible that the first few frames of the generated video appear blurry, gradually becoming clear as shown in Figure 13(a), generated by SkyReels-A2 [21]. Similar phenomena are also observed in the Phantom [55], ConsisID [113], and Concat-ID [123] models, likely due to the use of VAE [11, 48] as the control signal.

Consistency Fade As shown in Figure 13(d), although the model effectively preserves both global and local information of the subject in the first half of the video, the diamond embedded in the ring gradually disappears as the sequence progresses. This issue may stem from the underlying video generation model [89, 41, 109], but it remains a noteworthy concern.

### B.5 Guideline for Model Selection

We visualize all the results of OpenS2V-Eval, as shown in Figure 14. As the number of S2V models increases, the community faces challenges in selecting the most appropriate model, as each one tends to highlight its best results. To address this challenge, we offer model selection guidelines based on the evaluation outcomes of OpenS2V-Eval: (1) For content creators (e.g., advertisements, product displays), the closed-source Kling [43] is the clear leader, providing a more flexible and user-friendly experience. However, due to its high inference cost, more cost-effective alternatives such as Pika

[44] and Vidu [5] may be preferred. While these alternatives do not surpass Kling [43], they still outperform open-source models. (2) For community developers, it is recommended to base S2V model development on Phantom [55] or VACE [40], as it generates videos with relatively high quality and subject fidelity. Fine-tuning these methods can reduce development costs. (3) Although Hailuo has a narrower scope of application, it outperforms open-domain models like Kling in preserving human identity, making it more suitable for generating human-centric videos, such as those involving models and voice-over content. (4) For developing human-centric S2V models, open-source methods like HunyuanCustom [33], and ConsisID [123] offer high-quality pretrained weights, which may could also be extended to open-domain subject-to-video generation.

## C More Details of OpenS2V-5M

### C.1 Additional Details of Subject-Driven Processing

Human-Centric Filtering. Our data comes from 14,818,489 raw videos crawled from Internet through the Open-Sora Plan [50], consisting of no transition, clean clips with detailed raw captions. We design 100 human-related verbs and nouns as search terms, which lead to the identification of 12,654,783 human-related videos based on the raw captions. Finally, we apply the Aesthetic Predictor [16], the OpenCV [6], the DOVER [99], and the OCR model [88] to obtain aesthetic scores, motion scores, technical scores, and watermark-free video areas, respectively, and filter out low-quality data, ultimately yielding 5,437,544 high-quality clips.

Subject-Driven Annotation. Unlike text-to-video, subject-to-video data requires captions that emphasize the subject. To achieve this, we first use Qwen2.5-VL-7B [90] to describe the appearance and changes of the subject while preserving essential elements of the video, such as environmental context and camera movements, to get the subject-centric video caption. Next, to obtain highquality reference images, we use DeepSeekV3 [53] to extract keywords related to the environment and objects from the caption. We then input the first frame of the video and these keywords into GroundingDino [56], an open-vocabulary object detection algorithm, to extract reference images for each video. Finally, the bounding boxes obtained from the previous step are fed into SAM2.1 [76], which generates a mask for each subject. This mask can be used to extract reference images without background pixels. To ensure data quality, we further assign Aesthetic Score [16] and text GmeScore to the reference images, allowing users to adjust thresholds to balance data quantity and quality.

### C.2 Additional Details of Dataset Statistics

OpenS2V-5M is the first high-quality, large-scale S2V dataset. In contrast to standard datasets [45, 9, 12], it includes Nexus Data specifically designed to address three critical challenges faced by S2V methods. As depicted in Figure 15, the word cloud illustrates the dataset’s rich visual content. Regarding video duration, the majority (91%) of videos are between 0 and 10 seconds, while the remaining videos exceed 10 seconds. In terms of resolution, 65% are 720P, with the rest being high-resolution videos. The captions primarily consist of detailed descriptions, with a wide range of word usage. These settings are tailored to the emerging DiT-based models [59, 41, 109, 89], which favor long prompts and are constrained by input limitations, such as 81 frames and 480P resolution. Furthermore, low-quality videos were excluded during preprocessing based on motion, technical, and aesthetic scores, ensuring that most videos are of high quality. Due to resource constraints, we select the top 10K samples with the highest average scores from the 5M dataset to construct gpt-frame pairs. For cross-frame pairs, we identify 0.35M clustering centers from the regular data, each containing an average of 10.13 samples, meaning we could theoretically create far more than 0.35M × 10.13 pairs.

### C.3 Further Verification on OpenS2V-5M

Due to limited space in the main text, we provide additional qualitative analysis of Ours‡ here, with results shown in Figure 16. It can be observed that Ours‡ is capable of generating high-quality videos, thereby validating the effectiveness of the proposed OpenS2V-5M.

0.6

6%

9%

0.5

20%

14%

14%

0.4

35%

Percentage

0.3

0.2

65%

0.1

33% 24%

80%

0

Score Interval

<0.03 0.03-0.06 >0.06

720P 1080P

<5.0 5.0-5.5 5.5-6.0 >6.0

<0.5 0.5-1.0 1.0-1.5 1.5-2.0 >2.0

(a) Video Resolution (b) Motion Score (c) Technical Score (d) Aesthetic Score

[Figure 484]

###### 4%

9%

19%

12%

26%

51%

28%

51%

<150 150-200 200-250 >250

<5 5-10 10-15 >15

(e) Video Length (f) Prompt Word Range (g) Prompt Word Cloud

Figure 15: Statistics in OpenS2V-5M. The dataset includes a diverse range of categories, clip durations and caption lengths, with most of videos being in high quality (e.g., resolution, aesthetic).

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

Figure 16: More Showcases Generated by Ours‡.

- C.4 Samples of Collected Data

Figure 17 presents diverse samples from the OpenS2V-5M dataset, which consists of subject-textvideo triples across multiple categories, offering rich visual information. The subjects include both regular data obtained through segmentation and Nexus Data generated via cross-video association and GPT-Image-1, encompassing humans, objects, backgrounds, and more. These samples highlight the dataset’s diversity and depth, and are expected to address the three primary challenges faced by subject-to-video generation models, thereby advancing the field and contributingto the community.

## D More Details of Experiment

### D.1 Details of Resource

We employ Nvidia A100 (x40) for all the experiments. All implementations are conducted on the basis of the official code using the PyTorch framework or official interface.

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

Figure 17: Samples from the OpenS2V-5M dataset. The dataset consists of subject-text-video triples, which exhibit more physical knowledge than existing large-scale T2V dataset [12, 91].

### D.2 Details of Evaluation Models

As most S2V models [113, 123, 18, 55, 21, 40] do not support dynamic resolution or variable duration, standardization of these parameters is infeasible. Therefore, we adopt the commonly used official settings [58, 36, 81, 101] to maintain fairness across comparisons.

Vidu Model Details. Vidu [5] has released three versions of closed-source models: 1.0, 1.5, and 2.0. Among these, versions 1.5 and 2.0 support multi-reference image input, enabling opendomain subject-to-video generation. However, as the technical report has not been published, specific implementation details remain undisclosed. Implementation Setups. We employ the official Vidu 2.0 charactertovideo feature with default parameter settings. Using the turbo mode, we generate a 4-second video (65-frames) with a spatial resolution of 704 × 396, automatic motion amplitude, and a frame rate of 16 fps.

Pika Model Details. Pika [44] has developed five iterations of closed-source model, designated as versions 1.0, 1.5, 2.0, 2.1, and 2.2. Notably, versions 2.0, 2.1 and 2.2 incorporate multi-reference image input capability, enabling open-domain subject-to-video generation. However, due to the absence of an official technical report, the underlying implementation details remain undisclosed. Implementation Setups. We employ the official Pika 2.1 pikaadditions feature with default parameter settings. The generated video maintains a resolution of 1920 × 1080 pixels and a frame rate of 24 fps, with a total duration of 5 seconds (121-frames).

Kling Model Details. Kling [43] has released five versions of closed-source model: 1.0, 1.6, and 2.0, among which version 1.6 supports the input of multiple reference images for open-domain subject-to-video generation. However, as no technical report has been released for this version, we are unable to obtain further details. Implementation Setups. We employ the official Kling 1.6 multi|-id feature with default parameter settings. Using the standard mode, we generate a 5-second video (153-frames) with a spatial resolution of 1280 × 720, and a frame rate of 30 fps.

Hailuo Model Details. Hailuo [87] has released six versions of closed-source model: I2V-01Director, I2V-01-live, I2V-01, T2V-01-Director, T2V-01, and S2V-01. Among them, S2V-01 supports the input of multiple reference images to achieve human-domain subject-to-video generation. However, since no technical report has been released for this model, we are unable to obtain further details. Implementation Setups. We use the S2V function of the official Hailuo-S2V-01, available at Hailuo-S2V-01, and keep the default settings. We generate a 5-second video (141-frames) with a spatial resolution of 1280 × 720 and a frame rate of 25fps.

VACE Model Details. VACE [40] is a video generation model based on DiT that integrates various inputs in four data modalities—text, image, video, and mask—and unifies multiple video generation and editing tasks within a single model, including open-domain subject-to-video generation. It releases four model weights: VACE-Wan2.1-1.3B-Preview, VACE-LTX-Video-0.9, Wan2.1-VACE1.3B, and Wan2.1-VACE-14B. The training data consists of over a million text-to-video samples, which it collects and processes internally. Implementation Setups. We use the officially released

VACE code and models, maintaining the original settings. For VACE-Wan2.1-1.3B-Preview and VACE-Wan2.1-1.3B, we generate 5-second (81-frame) videos at a spatial resolution of 832×480 and a frame rate of 16 fps. For VACE-Wan2.1-14B, we generate 5-second (81-frame) videos at a spatial resolution of 1280 × 720 and a frame rate of 16 fps.

Phantom Model Details. Phantom [55] is a video generation model based on DiT that extracts reference image information using both CLIP and VAE, and employs a windowed attention mechanism to reduce computational overhead, enabling open-domain subject-to-video generation. It includes three model weights: Phantom-Seaweed, Phantom-Wan-1.3B, and Phantom-Wan-14, but only Phantom-Wan-1.3B&14B are publicly released. The training data come from panda70M [12], subject200k [14], OmniGen [104], and internal datasets, totaling over 10 million samples. Implementation Setups. We use the officially released Phantom-Wan code and model, maintaining the original settings. We generate 5-second (81-frame) videos at a resolution of 832 × 480 and a 16 fps.

SkyReels-A2 Model Details. SkyReels-A2 [21] is a model fine-tuned based on Wan2.1 [89], employing an approach similar to Phantom. It utilizes a dual-stream architecture to enhance the model’s response to reference images and textual prompts, enabling open-domain subject-to-video generation. There are four variants in total: A2-Wan2.1-14B-Preview, A2-Wan2.1-14B, A2-Wan2.114B-Pro, and A2-Wan2.1-14B-Infinity, but only A2-Wan2.1-14B-Preview has been open-sourced. The training data comes from 2 million high-quality subject-text-video triples collected internally. Implementation Setups. We use the officially released SkyReels-A2-Wan2.1-14B-Preview code and model, maintaining the original settings. Videos are generated with a spatial resolution of 832 × 480 and a frame rate of 16 fps, resulting in a duration of 5 seconds (81 frames).

HunyuanCustom Model Details. HunyuanCustom [33] is a model fine-tuned based on HunyuanVideo [33], which achieves open-domain subject-to-video generation by injecting ID information into both the MLLM and the video-driven injection module. In theory, it supports the input of multiple reference images, but currently only the weights supporting Single-Subject have been opensourced. The training data is processed from internally collected and open-source datasets, but the size of the dataset has not been disclosed. Implementation Setups. We use the officially released HunyuanCustom-Single-Subject code, maintaining the original settings. Videos are generated with a spatial resolution of 1280 × 720 and a 25 fps, resulting in a duration of 5 seconds (129 frames).

ConsisID Model Details. ConsisID [113] is a model fine-tuned based on CogVideoX [109], which achieves human-domain subject-to-video generation by decomposing ID information into high- and low-frequency signals and injecting them into DiT via cross-attention. It only supports the input of a single face image. The training data is processed from internally collected data, with a dataset size of approximately 0.1 million. Implementation Setups. We use the officially released ConsisID code and model, maintaining the original settings. Videos are generated with a spatial resolution of 720 × 480 and a frame rate of 8 fps, resulting in a duration of 6 seconds (49 frames).

Concat-ID Model Details. Concat-ID [123] is a model fine-tuned based on CogVideoX [113] and Wan2.1 [89]. It concatenates image features with video latents along the token dimension, thereby avoiding the issue of blurry initial frames. It only supports input of a single face image. The training data is processed from internally collected data, with a dataset size of approximately 1.3 million. Implementation Setups. We use the officially released Concat-ID code and model, maintaining the original settings. For CogVideoX version, videos are generated with a spatial resolution of 720 × 480 and a frame rate of 8 fps, resulting in a duration of 6 seconds (49 frames). For Wan-AdaLN version, videos are generated with a spatial resolution of 832 × 480 and a frame rate of 16 fps, resulting in a duration of 5 seconds (81 frames).

FantasyID Model Details. FantasyID [120] is a model fine-tuned from CogVideoX [109] that facilitates identity-consistent generation by constructing multi-view facial datasets, incorporating

- 3D geometric priors, and utilizing a layer-aware control signal injection mechanism. The model currently supports only single face image input. Its training data are drawn from ConsisID [113], CelebV-HQ [125], and Open-vid [67], comprising approximately 50,000 samples. Implementation Setups. We employ the officially released Fantasy-ID code and model while retaining the original settings. Videos are generated at a spatial resolution of 720 × 480 and a frame rate of 8 fps, yielding a duration of 6 seconds (49 frames).

EchoVideo Model Details. EchoVideo [97] is a model fine-tuned from CogVideoX [109] that employs the multimodal feature fusion module IITF to achieve identity-preserving video generation

through the integration of textual, visual, and facial identity information. The model supports only a single face image as input. The training data are sourced from internal collections and comprise approximately 3.3 million samples. Implementation Setups. We employ the officially released EchoVideo code and model while retaining the original settings. Videos are generated at a spatial resolution of 848 × 480 and a frame rate of 16 fps, yielding a duration of 3 seconds (49 frames).

VideoMaker Model Details. VideoMaker [103] is a UNet-based model fine-tuned from AnimateDiff [25]. It directly inputs reference images into the video diffusion model and utilizes its intrinsic feature extraction process to achieve subject-to-video generation (e.g., only supports 10 categories of subjects). The training data are sourced from CelebV-Text [125] and VideoBooth [39], comprising approximately 0.1M samples. Implementation Setups. We employ the officially released VideoMaker code and model while retaining the original settings. Videos are generated at a spatial resolution of 512 × 512 and a frame rate of 8 fps, yielding a duration of 2 seconds (16 frames).

ID-Animator Model Details. ID-Animator [29] is a UNet-based model fine-tuned from AnimateDiff [25] that employs FaceAdapter and cross-attention to inject facial information. The model supports only a single face image as input. The training data are sourced from CelebV-Text [125] and comprise approximately 15K samples. Implementation Setups. We employ the officially released ID-Animator code and model while retaining the original settings. Videos are generated at a spatial resolution of 512 × 512 and a frame rate of 8 fps, yielding a duration of 2 seconds (16 frames).

### D.3 Additional Details of Evaluation Settings

Because some models support only a single subject, while others support multiple subjects, we categorize the evaluation tasks into the following three groups:

Open-Domain Subject-to-Video including ① single-face-to-video, ② single-body-to-video, ③ single-entity-to-video, ④ multi-face-to-video, ⑤ multi-body-to-video, ⑥ multi-entity-to-video, and ⑦ human-entity-to-video.

Human-Domain Subject-to-Video including ① single-face-to-video and ② single-body-to-video. In this context, only the face image is input, without the body image.

Single-Domain Subject-to-Video including ① single-face-to-video, ② single-body-to-video, and

③ single-entity-to-video.

### D.4 Additional Details of Implementations

With the exception of MotionScore, which requires the use of all frames, the other metrics (e.g., NexusScore, NaturalScore, GmeScore, FaceSim, AestheticScore) are calculated by uniformly sampling 32 frames to ensure fairness and minimize overhead. Additionally, due to the differing optimal inference settings for each model, it is not feasible to standardize the resolution of generated videos. (1) For MotionScore, we use OpenCV [6] to compute this using the OpticalFlowFarneback. (2) For FaceSim, following the approach outlined in ConsisID [123], we first apply insightface [17] to detect the face regions in the video frames and the reference image. We then calculate the similarity between these regions in the curricularface [34] feature space. Finally, we average the sum of all valid scores to obtain the FaceSim for the video. (3) For AestheticScore, following the method presented in the improved-aesthetic-predictor [16], we directly input the video frames into the model to obtain scores, then compute the average of all valid scores to obtain the AestheticScore for the video. (4) For NexusScore, since we have filtered out low-quality Bi,t using ci,t and si,t, high-quality scores may be obtained when only one frame of the video is of high quality while the remaining frames are of lower quality. Therefore, after summing and averaging all valid scores, we divide by T′ to mitigate this issue. Here, T′ refers to the total number of frames in which an object is detected. In addition, this metric is not used to calculate face similarity to improve robustness, which is why we retain FaceSim. (5) For NaturalScore, we use gpt-4o-2024-11-20 [1] as the base model. For each video, we resize the longer side to 512 pixels and run the model three times, taking the average of these results as the score for the video. (6) For GmeScore, since it is based on Qwen2-VL [90], which natively supports dynamic resolution and variable duration, no special processing is necessary.

|(a) Distribution of NexusScore (b) Distribution of MotionScore (c) Distribution of AestheticScore<br><br>[Figure 545]<br><br>[Figure 546]<br><br>[Figure 547]|
|---|

Figure 18: Distribution of NexusScore, AestheticsScore and MotionScore.

- D.5 Additional Details of Metircs Normalization

OpenS2V-Eval evaluates six key dimensions: subject consistency, subject naturalness, text relevance, face similarity, visual quality, and motion amplitude. Due to differing units of measurement across these metrics, direct comparisons and comprehensive analysis are infeasible without normalization. To resolve this, we normalize each metric by defining its theoretical or empirical bounds:

- • FaceSim-Cur and GmeScore are bounded by construction, with ranges fixed at [0,1].
- • NaturalScore employs a 5-point Likert scale, spanning [1,5].
- • For unbounded metrics (NexusScore, AestheticScore, and MotionScore), we derive ranges of [0,0.05], [0,1], and [4,7], respectively, from their empirical distributions (Figure 18). Out-of-range values are truncated.

To aggregate these normalized metrics into a unified performance score, we compute a weighted sum: Total_Score =

i∈M

wi · Si, where M = {Nexus,Natural,Gme,FaceSim,Aesthetic,Motion},

(7)

with weights wi assigned as ι = 0.20 (NexusScore), κ = 0.24 (NaturalScore), λ = 0.12 (GmeScore), µ = 0.20 (FaceSim-Cur), ν = 0.12 (AestheticScore), and ξ = 0.12 (MotionScore). For humamdomain S2V task, κ = 0.30, λ = 0.15, µ = 0.25, ν = 0.15 and ξ = 0.15.

- D.6 Additional Details of Human Evaluation

Pre-processing The questionnaire for human evaluation of generated content is developed based on prior studies [113, 115, 75, 81, 80], as shown in Figure 19. The evaluation focuses on six key aspects: subject consistency, subject naturalness, text relevance, face similarity, visual quality, and motion amplitude. For each criterion, a pairwise comparison method is employed, allowing participants to choose between two video options, thereby improving user pleasure and increasing the number of effective questionnaire samples. To ensure category balance, 30 test samples are randomly selected from OpenS2V-Eval, with each sample paired with two videos generated by different models, yielding a total of 60 videos. These videos are annotated with six evaluation scores: NexusScore, NaturalScore, GmeScore, FaceSim-Cur [113], AestheticScore [16], and MotionScore [6]. Taking subject consistency as an example, a sample is labeled as a positive instance for NexusScore if a participant prefers video A over video B and A’s NexusScore exceeds that of B; otherwise, it is labeled as a negative instance. The final human preference ratio for each metric is computed as the proportion of positive instances among all test samples. Participants include undergraduate, master’s, and doctoral students, as well as members of the general public with no direct affiliation to the research domain. They are drawn from a diverse international pool, including individuals from China, and the United States. This heterogeneous composition ensures both the reliability and generalizability of the evaluation results.

Post-processing Folloing [113, 115, 114], to ensure data quality given the use of a five-point evaluation scale, we exclude outlier responses through the following procedures: ① We limit each submission to a single response per IP address and require users to log in prior to voting, thereby ensuring that each participant can submit only one response. ② We assess data validity by considering questionnaire completion time. As it requires 5 to 10 minutes to complete the survey, we exclude responses submitted in less than 5 minutes. ③ We randomize the playback order of videos for each

||[Figure 548]|
|---|
<br><br>|[Figure 549]|
|---|
|
|---|

### Figure 19: Visualization of the Questionnaire for User Study.

||Given an image caption, please retrieve the entity words that indicate background, subject, and visually separable objects.<br><br>[Definition of background] The background spaces that appear in most of the image area. [Definition of subject] Human or animal subjects that appear in the image. [Definition of object] Entities that are visually separable, tangible, and physically present in part of the image.<br><br>Attention! All entity words need to strictly follow the rules below:<br><br>1) The entity word is a singular or plural noun without any quantifier or descriptive phrase.<br>2) The entity word must be an exact subset of the caption, including its characters, words, and symbols. (e.g, 'red top' better than 'top', 'martial arts uniforms' better than 'uniforms')<br>3) Exclude any part of the body (e.g., 'hands', 'legs', 'feet', 'head').<br>4) Exclude abstract or non-physical concepts (e.g., 'facial expressions', 'gestures', 'stance').<br>5) Exclude actions or descriptions (e.g., 'adjusting', 'imitating'). Do not modify or interpret any part of the caption.<br><br><br>Here is an example, follow this JSON format to output the results: Caption: A woman in a mask and coat, with long brown hair, shows a small green-capped bottle to the camera. Output: {'background': [''], 'subject': ['woman'], 'object': ['mask', 'coat', 'long brown hair', 'green-capped bottle']}<br><br>Here is the input: Caption: {{{}}} Output:|
|---|
<br><br>|Your task is to determine how realistic the given video clip appears, based on 16 extracted frames. Consider the following aspects in your evaluation:<br><br>- **Common sense consistency**: Are the objects, people, and interactions logically coherent in the context of the video?<br>- **Physical plausibility**: Do lighting, shadows, motion, and reflections obey the laws of physics? Are the objects in motion consistent with real-world physics?<br>- **Naturalness**: Does the visual quality (textures, details, proportions, etc.) resemble what we would expect in real life? Is there any unnatural visual distortion?<br>- **AI generation artifacts**: Are there signs of unnatural blurring, morphing, glitches, distortions, or inconsistencies across frames?<br><br>**If the video contains humans**, pay special attention to:<br><br>- Are the facial features realistic and anatomically correct (e.g., eyes, mouth, and nose proportions)?<br>- Do the body parts appear proportionate and natural in motion (e.g., arm and leg movements, hand gestures)?<br><br>If **no humans** are present in the video, you can focus on evaluating the realism of other visual aspects like object consistency, motion fluidity, and environmental plausibility without needing to specifically assess human-related elements.<br><br>Output a score from 1 to 5 based on the criteria below, followed by an explanation of the reasoning behind your score:<br><br>- **1 — Definitely AI-Generated**: Clear and frequent artifacts (e.g., blurry faces or objects, unnatural movements, inconsistent lighting), distorted shapes, implausible physics (e.g., impossible movements, lighting issues), and severe inconsistencies. Violates common sense or real-world logic. Faces and bodies may be unrealistic or distorted if humans are present.<br>- **2 — Likely AI-Generated**: Noticeable AI generation cues such as inconsistent anatomy, fluctuating object textures, or mild physical implausibility (e.g., unnatural hand positions or eye movements). Faces and bodies may appear unnatural or inconsistent if humans are present. Still clearly synthetic upon inspection.<br>- **3 — Uncertain / Borderline**: Mixed indicators — the video may appear mostly natural but contains subtle flaws or small anomalies that raise suspicion. Faces and bodies might show mild inconsistencies (e.g., slight distortion in facial features or body parts) if humans are present. Hard to determine definitively.<br>- **4 — Likely Real**: Mostly natural and physically plausible, with only minor and rare irregularities that might be explainable (e.g., slight compression, mild lighting inconsistencies). Faces and body parts are mostly natural, with only minor imperfections, if humans are present.<br>- **5 — Definitely Real**: Fully consistent with real-world physics, common sense, and appearance. No visible artifacts or signs of AI generation. Faces and body parts appear fully realistic, without any visible distortions or unnatural movements, if humans are present. Please only return the score (1-5), no additional explanation.<br>|
|---|
<br><br>(a) Prompt for Extracting Tags (b) Prompt for Getting NaturalScore|
|---|

### Figure 20: Visualization of Different Input Text Prompts.

participant to mitigate cognitive bias. ④ We implement a sliding verification upon submission to ensure that all questionnaires are completed manually, thereby preventing automated (bot) responses. ⑤ We exclude any questionnaires for which more than 50% of evaluations are extreme values, defined as responses where the sum of the highest (5) and lowest (1) ratings exceeds 50%.

### D.7 Additional Details of Input Prompts

Regarding how to obtain tags through Deepseek [53] and how to annotate videos with NaturalScore using GPT-4o [1], we visualize the input text prompt, as shown in Figure 20.

## E Additional Statement

### E.1 Limitations and Future Work

Although NexusScore and NaturalScore are introduced to evaluate subject consistency and naturalness, these metrics show only approximately 75% correlation with human preferences. Future work aims to better align automated metrics with human judgments. The videos in OpenS2V-5M come from multiple video platforms, and we can only make publicly available those that comply with the CC BY 4.0 license or are copyright-free, totaling approximately 4 million videos.

### E.2 Declaration of LLM Usage

We utilized Large Language Models (LLMs), such as ChatGPT, to support the preparation of this paper. Specifically, LLMs were employed for language-related tasks, including grammar correction, spelling checks, and word choice refinement, to improve the manuscript’s clarity and fluency. Additionally, LLMs assisted with data processing and filtering (e.g., our NaturalScore is GPT-based), as well as

generating draft figures to assist the authors in creating refined visualizations. All scientific content, analyses, and conclusions were independently conceived, validated, and interpreted by the authors.

### E.3 Potential Harms Caused by the Research Process

The subject images of OpenS2V-Eval are derived from three open-source datasets—ConsisID [113], A2-Bench [21], and DreamBench [71]—that adhere to the Apache license, as well as from three video platforms—Pexels, MixKit, and PixaBay—that operate under the Creative Commons Zero (CC0) license. The video data in OpenS2V-5M originates from the Open-Sora Plan [50], with some content licensed under Creative Commons Attribution 4.0 (CC BY 4.0) and others under the Royalty-Free (RF) license. The licensing information for these data is explicitly stated on their respective platforms. The CC0 license designates content as public domain, permitting unrestricted use without additional permissions or authorizations. For CC BY 4.0-licensed videos from the Open-Sora Plan [50], video IDs are included in the metadata to mitigate potential contractual disputes. For RF-licensed videos, we are working to resolve intellectual property issues. In total, approximately 4 million data will be made available as open source. The collected data is organized into seven categories, with contributions from global sources. This diversity ensures that OpenS2V-Eval and OpenS2V-5M are fully representative. The ConsisID model [113] fine-tuned on our dataset demonstrated no significant content bias. Furthermore, video content has been filtered to exclude NSFW material based on subtitle detection. Due to the presence of videos containing identifiable individuals, access to OpenS2V-Nexus is restricted to academic use only, with contact information provided on the https://pku-yuangroup.github.io/OpenS2V-Nexus to ensure the security of personal identity data.

Data collection was made possible through the dedicated efforts of numerous contributors, including the authors of this paper and those involved in the manual evaluation. We consider individual hourly wages or compensation as personal information, and for privacy reasons, these details cannot be disclosed. Nonetheless, we can confirm that all participants have received appropriate compensation in accordance with the legal requirements of their respective countries or regions. The privacy of all participants is safeguarded, ensuring that no additional risks are posed to them.

### E.4 Societal Impact and Potential Harmful Consequences

The objective of OpenS2V-Eval is to identify the limitations of existing subject-to-video generation models and to develop the OpenS2V-5M dataset to further advance research in this area. While subject-to-video generation models hold significant potential for enhancing creativity, their broader societal impacts must be carefully considered during development:

First, environmental resource consumption. Training subject-to-video generation models requires extensive GPU computing power, with a single large-scale training session potentially consuming tens of thousands of kilowatt-hours of electricity, resulting in carbon emissions comparable to the annual emissions of several dozen cars. This high energy consumption not only exacerbates global climate change but also consolidates computational resources within a few dominant tech companies, exacerbating inequality in the research community. To address this, efforts should focus on exploring techniques for model lightweighting, optimizing distributed training efficiency, and promoting the development of green data centers powered by renewable energy to reduce the carbon footprint.

Second, the risk of linguistic homogeneity and cultural bias. The text prompt in OpenS2V-Nexus are currently limited to English, which may introduce bias in the model’s interpretation of multilingual contexts, such as Chinese. For instance, when generating videos involving non-Western cultural symbols (e.g., Hanfu, Kung fu), the lack of relevant training data could lead to semantic distortions or cultural misinterpretations. Solutions include creating a multilingual annotation system and establishing an open-source collaborative framework to encourage researchers globally to contribute localized data, helping bridge language barriers.

Finally, the ethical concerns associated with deepfake misuse. Subject-consistency video generation technologies may be exploited for malicious purposes, such as creating political misinformation, forging celebrity images, or fabricating criminal evidence. The level of realism achievable with these technologies surpasses that of traditional Photoshop techniques. Such misuse poses a threat to public opinion security and judicial integrity. Effective countermeasures should combine technological governance and regulatory oversight: developing generative models embedded with imperceptible watermarks, establishing blockchain-based content traceability protocols, and advocating for legislation

requiring mandatory labeling of generated content. Additionally, public media literacy campaigns should be implemented to enhance society’s resilience to false information.

### E.5 Impact Mitigation Measures

We are fully responsible for the authorization, distribution, and maintenance of OpenS2V-Eval and OpenS2V-5M. Our datasets and benchmarks are released under the CC-BY-4.0 license, while the code is released under the Apache license. We explicitly state on our homepage that all data is intended for academic research purposes to prevent misuse or improper use. We also provide metadata for each video, allowing video creators to contact us promptly and remove invalid videos. All metadata is hosted on GitHub and HuggingFace, with the following links: https://github.com/PKUYuanGroup/OpenS2V-Nexus and https://huggingface.co/collections/BestWishYsh.

||[Figure 550]|
|---|
<br><br>VACE<br><br>14B<br><br>Kling1.6 Phantom<br><br>1.3B Pika2.1<br><br>Phantom<br><br>14B VACE<br><br>P1.3B Vidu2.0<br><br>VACE<br><br>14B VACE<br><br>1.3B<br><br>The video opens with a serene outdoor scene at 00:00, featuring two individuals standing in a field of tall grass. The setting is a rural area with a clear sky and a few scattered trees in the background. The person on the left, wearing a white shirt, is holding a smartphone and appears to be taking a selfie ...<br><br>The video showcases a serene indoor setting with a wooden table in the foreground, adorned with two white cushions, a white teapot, two white cups, and a vase with yellow flowers. The scene is set against a backdrop of a ...<br><br>|[Figure 551]|
|---|
<br><br>|[Figure 552]|
|---|
<br><br>|[Figure 553]|
|---|
<br><br>|[Figure 554]|
|---|
<br><br>|[Figure 555]|
|---|
<br><br>|[Figure 556]|
|---|
<br><br>|[Figure 557]|
|---|
<br><br>|[Figure 558]|
|---|
<br><br>|[Figure 559]|
|---|
<br><br>|[Figure 560]|
|---|
<br><br>|[Figure 561]|
|---|
<br><br>|[Figure 562]|
|---|
<br><br>Vidu2.0<br><br>|[Figure 563]|
|---|
<br><br>|[Figure 564]|
|---|
<br><br>|[Figure 565]|
|---|
<br><br>|[Figure 566]|
|---|
<br><br>|[Figure 567]|
|---|
<br><br>|[Figure 568]|
|---|
<br><br>|[Figure 569]|
|---|
<br><br>|[Figure 570]|
|---|
<br><br>Pika2.1<br><br>|[Figure 571]|
|---|
<br><br>|[Figure 572]|
|---|
<br><br>|[Figure 573]|
|---|
<br><br>|[Figure 574]|
|---|
<br><br>|[Figure 575]|
|---|
<br><br>|[Figure 576]|
|---|
<br><br>|[Figure 577]|
|---|
<br><br>|[Figure 578]|
|---|
<br><br>Kling1.6 VACE<br><br>|[Figure 579]|
|---|
<br><br>|[Figure 580]|
|---|
<br><br>|[Figure 581]|
|---|
<br><br>|[Figure 582]|
|---|
<br><br>|[Figure 583]|
|---|
<br><br>|[Figure 584]|
|---|
<br><br>|[Figure 585]|
|---|
<br><br>|[Figure 586]|
|---|
<br><br>P1.3B Phantom<br><br>|[Figure 587]|
|---|
<br><br>|[Figure 588]|
|---|
<br><br>|[Figure 589]|
|---|
<br><br>|[Figure 590]|
|---|
<br><br>|[Figure 591]|
|---|
<br><br>|[Figure 592]|
|---|
<br><br>|[Figure 593]|
|---|
<br><br>|[Figure 594]|
|---|
<br><br>1.3B<br><br>|[Figure 595]|
|---|
<br><br>|[Figure 596]|
|---|
<br><br>|[Figure 597]|
|---|
<br><br>|[Figure 598]|
|---|
<br><br>|[Figure 599]|
|---|
<br><br>|[Figure 600]|
|---|
<br><br>|[Figure 601]|
|---|
<br><br>|[Figure 602]|
|---|
<br><br>SkyReels-A2<br><br>14B<br><br>|[Figure 603]|
|---|
<br><br>VACE<br><br>14B<br><br>Kling1.6<br><br>Phantom<br><br>1.3B Pika2.1<br><br>Phantom<br><br>14B<br><br>VACE<br><br>P1.3B Vidu2.0<br><br>VACE<br><br>14B<br><br>VACE<br><br>1.3B<br><br>The video captures a serene outdoor scene featuring two squirrels interacting with a wooden crate. Initially, one squirrel is seen standing on the edge of the crate, while the other is inside, possibly foraging or exploring. The setting is natural ...<br><br>a man playing with his dog on the beach<br><br>|[Figure 604]|
|---|
<br><br>|[Figure 605]|
|---|
<br><br>| |
|---|
<br><br>[Figure 606]<br><br>|[Figure 607]|
|---|
<br><br>|[Figure 608]|
|---|
<br><br>|[Figure 609]|
|---|
<br><br>|[Figure 610]|
|---|
<br><br>|[Figure 611]|
|---|
<br><br>|[Figure 612]|
|---|
<br><br>|[Figure 613]|
|---|
<br><br>|[Figure 614]|
|---|
<br><br>|[Figure 615]|
|---|
<br><br>Vidu2.0<br><br>|[Figure 616]|
|---|
<br><br>|[Figure 617]|
|---|
<br><br>|[Figure 618]|
|---|
<br><br>|[Figure 619]|
|---|
<br><br>|[Figure 620]|
|---|
<br><br>|[Figure 621]|
|---|
<br><br>|[Figure 622]|
|---|
<br><br>|[Figure 623]|
|---|
<br><br>Pika2.1<br><br>|[Figure 624]|
|---|
<br><br>|[Figure 625]|
|---|
<br><br>|[Figure 626]|
|---|
<br><br>|[Figure 627]|
|---|
<br><br>|[Figure 628]|
|---|
<br><br>|[Figure 629]|
|---|
<br><br>|[Figure 630]|
|---|
<br><br>|[Figure 631]|
|---|
<br><br>Kling1.6 VACE<br><br>|[Figure 632]|
|---|
<br><br>|[Figure 633]|
|---|
<br><br>|[Figure 634]|
|---|
<br><br>|[Figure 635]|
|---|
<br><br>|[Figure 636]|
|---|
<br><br>|[Figure 637]|
|---|
<br><br>|[Figure 638]|
|---|
<br><br>|[Figure 639]|
|---|
<br><br>P1.3B Phantom<br><br>|[Figure 640]|
|---|
<br><br>|[Figure 641]|
|---|
<br><br>|[Figure 642]|
|---|
<br><br>|[Figure 643]|
|---|
<br><br>|[Figure 644]|
|---|
<br><br>|[Figure 645]|
|---|
<br><br>|[Figure 646]|
|---|
<br><br>|[Figure 647]|
|---|
<br><br>1.3B<br><br>|[Figure 648]|
|---|
<br><br>|[Figure 649]|
|---|
<br><br>|[Figure 650]|
|---|
<br><br>|[Figure 651]|
|---|
<br><br>|[Figure 652]|
|---|
<br><br>|[Figure 653]|
|---|
<br><br>|[Figure 654]|
|---|
<br><br>|[Figure 655]|
|---|
<br><br>SkyReels-A2<br><br>P14B<br><br>|[Figure 656]|
|---|
<br><br>|[Figure 657]| |[Figure 658]<br><br>[Figure 659]| |
|---|---|---|---|
<br><br>|[Figure 660]|
|---|
<br><br>|[Figure 661]|
|---|
<br><br>|[Figure 662]|
|---|
<br><br>|[Figure 663]|
|---|
<br><br>|[Figure 664]|
|---|
<br><br>|[Figure 665]|
|---|
<br><br>|[Figure 666]|
|---|
<br><br>|[Figure 667]|
|---|
<br><br>|[Figure 668]|
|---|
<br><br>|[Figure 669]|
|---|
<br><br>|[Figure 670]|
|---|
<br><br>|[Figure 671]|
|---|
<br><br>|[Figure 672]|
|---|
<br><br>|[Figure 673]|
|---|
<br><br>|[Figure 674]|
|---|
<br><br>|[Figure 675]|
|---|
<br><br>VACE<br><br>1.3B VACE<br><br>|[Figure 676]|
|---|
<br><br>|[Figure 677]| |[Figure 678]|
|---|---|---|
<br><br>|[Figure 679]|
|---|
<br><br>1.3B<br><br>|[Figure 680]|
|---|
<br><br>|[Figure 681]|
|---|
<br><br>|[Figure 682]|
|---|
<br><br>|[Figure 683]|
|---|
<br><br>|[Figure 684]|
|---|
<br><br>|[Figure 685]|
|---|
<br><br>|[Figure 686]|
|---|
<br><br>|[Figure 687]|
|---|
<br><br>SkyReels-A2<br><br>P14B<br><br>Phantom<br><br>14B<br><br>SkyReels-A2<br><br>14B<br><br>Phantom<br><br>14B<br><br>|[Figure 688]|
|---|
<br><br>|[Figure 689]|
|---|
<br><br>|[Figure 690]|
|---|
<br><br>|[Figure 691]|
|---|
<br><br>|[Figure 692]|
|---|
<br><br>|[Figure 693]|
|---|
<br><br>|[Figure 694]|
|---|
<br><br>|[Figure 695]|
|---|
<br><br>|[Figure 696]|
|---|
<br><br>|[Figure 697]|
|---|
<br><br>|[Figure 698]|
|---|
<br><br>|[Figure 699]|
|---|
<br><br>|[Figure 700]|
|---|
<br><br>|[Figure 701]|
|---|
<br><br>|[Figure 702]|
|---|
<br><br>|[Figure 703]|
|---|
|
|---|

P1.3B Vidu2.0

Kling1.6Vidu2.0Pika2.1 VACE

1.3B Pika2.1

Kling1.6 Phantom

14B VACE

P1.3B Phantom

14B VACE

Vidu2.0 VACE

1.3B

1.3B

VACE

VACE

14B

1.3B

Phantom

Phantom

14B

SkyReels-A2

SkyReels-A2

14B

14B

Vidu2.0

### Figure 21: More Showcases in OpenS2V-Eval for Open-Domain Subject-to-Video Generation.

||[Figure 704]|
|---|
<br><br>Kling1.6 Phantom<br><br>1.3B<br><br>SkyReels-A2<br><br>P14B<br><br>VACE<br><br>P1.3B<br><br>The video begins with a close-up of a vintage television resting on a classic wooden stand in a cozy living room. The camera zooms in to<br><br>highlight the<br><br>television\u2019s retro design\u2014its rounded edges, dials, and the soft glow of the screen. A hand reaches into the frame, gently turning one of the dials ...<br><br>VACE<br><br>14B VACE<br><br>1.3B Vidu2.0Pika2.1<br><br>Hunyuan<br><br>Custom<br><br>|[Figure 705]|
|---|
<br><br>|[Figure 706]|
|---|
<br><br>|[Figure 707]|
|---|
<br><br>|[Figure 708]|
|---|
<br><br>|[Figure 709]|
|---|
<br><br>|[Figure 710]|
|---|
<br><br>|[Figure 711]|
|---|
<br><br>|[Figure 712]|
|---|
<br><br>|[Figure 713]|
|---|
<br><br>|[Figure 714]|
|---|
<br><br>|[Figure 715]|
|---|
<br><br>|[Figure 716]|
|---|
<br><br>|[Figure 717]|
|---|
<br><br>|[Figure 718]|
|---|
<br><br>|[Figure 719]|
|---|
<br><br>|[Figure 720]|
|---|
<br><br>|[Figure 721]|
|---|
<br><br>|[Figure 722]|
|---|
<br><br>|[Figure 723]|
|---|
<br><br>|[Figure 724]|
|---|
<br><br>|[Figure 725]|
|---|
<br><br>|[Figure 726]|
|---|
<br><br>|[Figure 727]|
|---|
<br><br>|[Figure 728]|
|---|
<br><br>|[Figure 729]|
|---|
<br><br>|[Figure 730]|
|---|
<br><br>|[Figure 731]|
|---|
<br><br>|[Figure 732]|
|---|
<br><br>|[Figure 733]|
|---|
<br><br>|[Figure 734]|
|---|
<br><br>|[Figure 735]|
|---|
<br><br>|[Figure 736]|
|---|
<br><br>|[Figure 737]|
|---|
<br><br>|[Figure 738]|
|---|
<br><br>|[Figure 739]|
|---|
<br><br>|[Figure 740]|
|---|
<br><br>Vidu2.0Pika2.1<br><br>In the video, a girl is sitting on a grassy hill, overlooking a vast field with wildflowers in full bloom. She is wearing a casual tshirt and denim shorts, with a wide-brimmed hat. The wind is gently blowing through the tall grasses, and the sky is clear and blue. She gazes out at the horizon ...<br><br>VACE<br><br>14B VACE<br><br>1.3B Kling1.6<br><br>|[Figure 741]|
|---|
<br><br>VACE<br><br>P1.3B Phantom<br><br>1.3B<br><br>SkyReels-A2<br><br>P14B Hunyuan<br><br>Custom<br><br>|[Figure 742]|
|---|
<br><br>|[Figure 743]|
|---|
<br><br>|[Figure 744]|
|---|
<br><br>|[Figure 745]|
|---|
<br><br>|[Figure 746]|
|---|
<br><br>|[Figure 747]|
|---|
<br><br>|[Figure 748]|
|---|
<br><br>|[Figure 749]|
|---|
<br><br>|[Figure 750]|
|---|
<br><br>|[Figure 751]|
|---|
<br><br>|[Figure 752]|
|---|
<br><br>|[Figure 753]|
|---|
<br><br>|[Figure 754]|
|---|
<br><br>|[Figure 755]|
|---|
<br><br>|[Figure 756]|
|---|
<br><br>|[Figure 757]|
|---|
<br><br>|[Figure 758]|
|---|
<br><br>|[Figure 759]|
|---|
<br><br>|[Figure 760]|
|---|
<br><br>|[Figure 761]|
|---|
<br><br>|[Figure 762]|
|---|
<br><br>|[Figure 763]|
|---|
<br><br>|[Figure 764]|
|---|
<br><br>|[Figure 765]|
|---|
<br><br>|[Figure 766]|
|---|
<br><br>|[Figure 767]|
|---|
<br><br>|[Figure 768]|
|---|
<br><br>|[Figure 769]|
|---|
<br><br>|[Figure 770]|
|---|
<br><br>|[Figure 771]|
|---|
<br><br>|[Figure 772]|
|---|
<br><br>|[Figure 773]|
|---|
<br><br>|[Figure 774]|
|---|
<br><br>|[Figure 775]|
|---|
<br><br>|[Figure 776]|
|---|
<br><br>|[Figure 777]|
|---|
<br><br>Phantom<br><br>14B<br><br>Phantom<br><br>14B<br><br>|[Figure 778]|
|---|
<br><br>|[Figure 779]|
|---|
<br><br>|[Figure 780]|
|---|
<br><br>|[Figure 781]|
|---|
<br><br>|[Figure 782]|
|---|
<br><br>|[Figure 783]|
|---|
<br><br>|[Figure 784]|
|---|
<br><br>|[Figure 785]|
|---|
|
|---|

Huny

Huny

Cust

Cust

### Figure 22: More Showcases in OpenS2V-Eval for Single-Domain Subject-to-Video Generation.

||[Figure 786]|
|---|
<br><br>Kling1.6 Phantom<br><br>1.3B<br><br>Pika2.1<br><br>SkyReels-A2<br><br>P14B<br><br>VACE<br><br>P1.3B<br><br>Vidu2.0<br><br>Concat-ID<br><br>CogVideoX<br><br>EchoVideoFantasyIDConsisIDHailuo<br><br>|[Figure 787]|
|---|
<br><br>|[Figure 788]|
|---|
<br><br>|[Figure 789]|
|---|
<br><br>|[Figure 790]|
|---|
<br><br>The video features a young man walking through a park during sunset. he is wearing a sleeveless top with a geometric pattern and denim shorts. The man has long, dark hair that falls over his shoulders ...<br><br>|[Figure 791]|
|---|
<br><br>|[Figure 792]|
|---|
<br><br>|[Figure 793]|
|---|
<br><br>|[Figure 794]|
|---|
<br><br>|[Figure 795]|
|---|
<br><br>|[Figure 796]|
|---|
<br><br>|[Figure 797]|
|---|
<br><br>|[Figure 798]|
|---|
<br><br>|[Figure 799]| | | |[Figure 800]| | | | | |[Figure 801]| | | | | |[Figure 802]| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 803]| |[Figure 804]| | | |[Figure 805]| |[Figure 806]| | | |[Figure 807]| |[Figure 808]| | | |[Figure 809]|
<br><br>|[Figure 810]|
|---|
<br><br>|[Figure 811]|
|---|
<br><br>|[Figure 812]|
|---|
<br><br>|[Figure 813]|
|---|
<br><br>|[Figure 814]|
|---|
<br><br>|[Figure 815]|
|---|
<br><br>|[Figure 816]|
|---|
<br><br>|[Figure 817]|
|---|
<br><br>|[Figure 818]|
|---|
<br><br>|[Figure 819]|
|---|
<br><br>|[Figure 820]|
|---|
<br><br>|[Figure 821]|
|---|
<br><br>|[Figure 822]|
|---|
<br><br>|[Figure 823]|
|---|
<br><br>|[Figure 824]|
|---|
<br><br>|[Figure 825]|
|---|
<br><br>|[Figure 826]|
|---|
<br><br>|[Figure 827]|
|---|
<br><br>|[Figure 828]|
|---|
<br><br>|[Figure 829]|
|---|
<br><br>|[Figure 830]|
|---|
<br><br>|[Figure 831]|
|---|
<br><br>|[Figure 832]|
|---|
<br><br>|[Figure 833]|
|---|
<br><br>|[Figure 834]|
|---|
<br><br>|[Figure 835]|
|---|
<br><br>|[Figure 836]|
|---|
<br><br>|[Figure 837]|
|---|
<br><br>Hunyuan<br><br>Custom<br><br>VACE<br><br>14B<br><br>|[Figure 838]|
|---|
<br><br>Kling1.6 Phantom<br><br>1.3B Pika2.1<br><br>Phantom<br><br>14B VACE<br><br>P1.3B Vidu2.0<br><br>The video features a man walking down a city street at night, engrossed in his smartphone. He is dressed in a formal suit and tie, suggesting he might be a professional or businessman. The street is illuminated ...<br><br>ConsisID<br><br>Concat-ID<br><br>Wan-AdaLN<br><br>Hunyuan<br><br>Custom<br><br>Concat-ID<br><br>CogVideoX<br><br>SkyReels-A2<br><br>P14B<br><br>|[Figure 839]|
|---|
<br><br>|[Figure 840]|
|---|
<br><br>|[Figure 841]|
|---|
<br><br>|[Figure 842]|
|---|
<br><br>|[Figure 843]|
|---|
<br><br>|[Figure 844]|
|---|
<br><br>|[Figure 845]|
|---|
<br><br>|[Figure 846]| | | |[Figure 847]| | | | | |[Figure 848]| | | | | |[Figure 849]| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 850]| |[Figure 851]| | | |[Figure 852]| |[Figure 853]| | | |[Figure 854]| |[Figure 855]| | | |[Figure 856]|
<br><br>|[Figure 857]|
|---|
<br><br>|[Figure 858]|
|---|
<br><br>|[Figure 859]|
|---|
<br><br>|[Figure 860]|
|---|
<br><br>|[Figure 861]|
|---|
<br><br>|[Figure 862]|
|---|
<br><br>|[Figure 863]|
|---|
<br><br>|[Figure 864]|
|---|
<br><br>|[Figure 865]|
|---|
<br><br>|[Figure 866]|
|---|
<br><br>|[Figure 867]|
|---|
<br><br>|[Figure 868]|
|---|
<br><br>|[Figure 869]|
|---|
<br><br>|[Figure 870]|
|---|
<br><br>|[Figure 871]|
|---|
<br><br>|[Figure 872]|
|---|
<br><br>|[Figure 873]|
|---|
<br><br>|[Figure 874]|
|---|
<br><br>|[Figure 875]|
|---|
<br><br>|[Figure 876]|
|---|
<br><br>|[Figure 877]|
|---|
<br><br>|[Figure 878]|
|---|
<br><br>|[Figure 879]|
|---|
<br><br>|[Figure 880]|
|---|
<br><br>|[Figure 881]|
|---|
<br><br>|[Figure 882]|
|---|
<br><br>|[Figure 883]|
|---|
<br><br>|[Figure 884]|
|---|
<br><br>|[Figure 885]|
|---|
<br><br>|[Figure 886]|
|---|
<br><br>|[Figure 887]|
|---|
<br><br>|[Figure 888]|
|---|
<br><br>|[Figure 889]|
|---|
<br><br>Hailuo VACE<br><br>1.3B VACE<br><br>14B<br><br>|[Figure 890]|
|---|
<br><br>|[Figure 891]|
|---|
<br><br>|[Figure 892]|
|---|
<br><br>|[Figure 893]|
|---|
<br><br>|[Figure 894]|
|---|
<br><br>|[Figure 895]|
|---|
<br><br>|[Figure 896]|
|---|
<br><br>|[Figure 897]|
|---|
<br><br>|[Figure 898]|
|---|
<br><br>|[Figure 899]|
|---|
<br><br>|[Figure 900]|
|---|
<br><br>|[Figure 901]|
|---|
<br><br>|[Figure 902]|
|---|
<br><br>|[Figure 903]|
|---|
<br><br>|[Figure 904]|
|---|
<br><br>|[Figure 905]|
|---|
<br><br>VACE<br><br>1.3B<br><br>|[Figure 906]|
|---|
<br><br>|[Figure 907]|
|---|
<br><br>|[Figure 908]|
|---|
<br><br>|[Figure 909]|
|---|
<br><br>|[Figure 910]|
|---|
<br><br>|[Figure 911]|
|---|
<br><br>|[Figure 912]|
|---|
<br><br>|[Figure 913]|
|---|
<br><br>VideoMaker<br><br>VideoMakerID-Animator<br><br>|[Figure 914]|
|---|
<br><br>|[Figure 915]|
|---|
<br><br>|[Figure 916]|
|---|
<br><br>|[Figure 917]|
|---|
<br><br>|[Figure 918]|
|---|
<br><br>|[Figure 919]|
|---|
<br><br>|[Figure 920]|
|---|
<br><br>|[Figure 921]|
|---|
<br><br>|[Figure 922]|
|---|
<br><br>|[Figure 923]|
|---|
<br><br>|[Figure 924]|
|---|
<br><br>|[Figure 925]|
|---|
<br><br>|[Figure 926]|
|---|
<br><br>|[Figure 927]|
|---|
<br><br>EchoVideoFantasyIDID-Animator<br><br>Concat-ID<br><br>Wan-AdaLN<br><br>Phantom<br><br>14B<br><br>|[Figure 928]|
|---|
<br><br>|[Figure 929]|
|---|
<br><br>|[Figure 930]|
|---|
<br><br>|[Figure 931]|
|---|
<br><br>|[Figure 932]|
|---|
<br><br>|[Figure 933]|
|---|
<br><br>|[Figure 934]|
|---|
<br><br>|[Figure 935]|
|---|
<br><br>|[Figure 936]|
|---|
<br><br>|[Figure 937]|
|---|
<br><br>|[Figure 938]|
|---|
<br><br>|[Figure 939]|
|---|
<br><br>|[Figure 940]|
|---|
<br><br>|[Figure 941]|
|---|
<br><br>|[Figure 942]|
|---|
<br><br>|[Figure 943]|
|---|
|
|---|

P1.3B Vidu2.0

Vidu2.0

Kling1.6Pika2.1 VACE

Kling1.6Pika2.1 VACE

P1.3B

E

E

B

B

### Figure 23: More Showcases in OpenS2V-Eval for Human-Domain Subject-to-Video Generation.

||[Figure 944]|
|---|
<br><br>The video features a woman in exquisite hybrid armor adorned with iridescent gemstones, standing amidst gently falling cherry blossoms. Her piercing yet serene gaze hints at quiet determination, as a breeze catches ...<br><br>Hunyuan<br><br>Custom VACE<br><br>1.3B<br><br>VACE<br><br>14B<br><br>|[Figure 945]|
|---|
<br><br>Kling1.6 Phantom<br><br>1.3B Pika2.1<br><br>VACE<br><br>P1.3B Vidu2.0<br><br>The video features a woman with blonde hair standing on a beach near the water's edge. She is wearing a black swimsuit and appears to be enjoying her time by the sea. The sky above is clear with some clouds ...<br><br>Concat-ID<br><br>CogVideoX<br><br>EchoVideoFantasyIDConsisIDHailuo Hunyuan<br><br>Custom VACE<br><br>1.3B<br><br>|[Figure 946]|
|---|
|[Figure 947]|
|[Figure 948]|
|[Figure 949]|
|[Figure 950]|
|[Figure 951]|
|[Figure 952]|
<br><br>|[Figure 953]|
|---|
|[Figure 954]|
|[Figure 955]|
|[Figure 956]|
|[Figure 957]|
|[Figure 958]|
|[Figure 959]|
<br><br>|[Figure 960]|
|---|
|[Figure 961]|
|[Figure 962]|
|[Figure 963]|
|[Figure 964]|
|[Figure 965]|
|[Figure 966]|
<br><br>|[Figure 967]|
|---|
|[Figure 968]|
|[Figure 969]|
|[Figure 970]|
|[Figure 971]|
|[Figure 972]|
|[Figure 973]|
<br><br>|[Figure 974]|
|---|
|[Figure 975]|
|[Figure 976]|
|[Figure 977]|
|[Figure 978]|
|[Figure 979]|
|[Figure 980]|
<br><br>|[Figure 981]|
|---|
<br><br>|[Figure 982]|
|---|
|[Figure 983]|
|[Figure 984]|
|[Figure 985]|
|[Figure 986]|
|[Figure 987]|
|[Figure 988]|
<br><br>|[Figure 989]|
|---|
|[Figure 990]|
|[Figure 991]|
|[Figure 992]|
|[Figure 993]|
|[Figure 994]|
|[Figure 995]|
<br><br>Vidu2.0<br><br>|[Figure 996]|
|---|
<br><br>Pika2.1<br><br>|[Figure 997]|
|---|
<br><br>Kling1.6 VACE<br><br>|[Figure 998]|
|---|
<br><br>P1.3B<br><br>|[Figure 999]|
|---|
|[Figure 1000]|
<br><br>Phantom<br><br>1.3B<br><br>SkyReels-A2<br><br>P14B<br><br>|[Figure 1001]|
|---|
<br><br>|[Figure 1002]|
|---|
<br><br>|[Figure 1003]|
|---|
<br><br>|[Figure 1004]|
|---|
<br><br>|[Figure 1005]|
|---|
<br><br>|[Figure 1006]|
|---|
<br><br>|[Figure 1007]|
|---|
<br><br>|[Figure 1008]|
|---|
<br><br>SkyReels-A2<br><br>P14B<br><br>|[Figure 1009]|
|---|
<br><br>|[Figure 1010]|
|---|
<br><br>|[Figure 1011]|
|---|
<br><br>|[Figure 1012]|
|---|
<br><br>|[Figure 1013]|
|---|
<br><br>|[Figure 1014]|
|---|
<br><br>|[Figure 1015]|
|---|
<br><br>|[Figure 1016]|
|---|
<br><br>Hailuo<br><br>|[Figure 1017]|
|---|
<br><br>|[Figure 1018]|
|---|
<br><br>|[Figure 1019]|
|---|
<br><br>|[Figure 1020]|
|---|
<br><br>|[Figure 1021]|
|---|
<br><br>|[Figure 1022]|
|---|
<br><br>|[Figure 1023]|
|---|
<br><br>|[Figure 1024]|
|---|
<br><br>ConsisID<br><br>|[Figure 1025]|
|---|
<br><br>|[Figure 1026]|
|---|
<br><br>|[Figure 1027]|
|---|
<br><br>|[Figure 1028]|
|---|
<br><br>|[Figure 1029]|[Figure 1030]|[Figure 1031]|[Figure 1032]|
|---|---|---|---|
<br><br>Concat-ID<br><br>CogVideoX<br><br>|[Figure 1033]|
|---|
<br><br>|[Figure 1034]|
|---|
<br><br>|[Figure 1035]|
|---|
<br><br>|[Figure 1036]|
|---|
<br><br>|[Figure 1037]|[Figure 1038]|[Figure 1039]|[Figure 1040]|
|---|---|---|---|
<br><br>FantasyID<br><br>|[Figure 1041]|
|---|
<br><br>|[Figure 1042]|
|---|
<br><br>|[Figure 1043]|
|---|
<br><br>|[Figure 1044]|
|---|
<br><br>|[Figure 1045]|[Figure 1046]|[Figure 1047]|[Figure 1048]|
|---|---|---|---|
<br><br>EchoVideo<br><br>|[Figure 1049]|
|---|
<br><br>|[Figure 1050]|
|---|
<br><br>|[Figure 1051]|
|---|
<br><br>|[Figure 1052]|
|---|
<br><br>|[Figure 1053]|
|---|
<br><br>|[Figure 1054]|
|---|
<br><br>|[Figure 1055]|
|---|
<br><br>|[Figure 1056]|
|---|
<br><br>VACE<br><br>14B<br><br>|[Figure 1057]|
|---|
<br><br>VideoMakerID-Animator<br><br>VideoMakerID-Animator<br><br>|[Figure 1058]|
|---|
<br><br>|[Figure 1059]|
|---|
<br><br>|[Figure 1060]|
|---|
<br><br>|[Figure 1061]|
|---|
<br><br>|[Figure 1062]|
|---|
<br><br>|[Figure 1063]|
|---|
<br><br>|[Figure 1064]|
|---|
<br><br>|[Figure 1065]|
|---|
<br><br>|[Figure 1066]|
|---|
<br><br>|[Figure 1067]|
|---|
<br><br>|[Figure 1068]|
|---|
<br><br>|[Figure 1069]|
|---|
<br><br>|[Figure 1070]|
|---|
<br><br>|[Figure 1071]|
|---|
<br><br>|[Figure 1072]|
|---|
<br><br>|[Figure 1073]|
|---|
<br><br>|[Figure 1074]|
|---|
<br><br>|[Figure 1075]|
|---|
<br><br>|[Figure 1076]|
|---|
<br><br>|[Figure 1077]|
|---|
<br><br>|[Figure 1078]|
|---|
<br><br>|[Figure 1079]|
|---|
<br><br>|[Figure 1080]|
|---|
<br><br>|[Figure 1081]|
|---|
<br><br>|[Figure 1082]|
|---|
<br><br>|[Figure 1083]|
|---|
<br><br>|[Figure 1084]|
|---|
<br><br>|[Figure 1085]|
|---|
<br><br>Concat-ID<br><br>Wan-AdaLN<br><br>Phantom<br><br>14B<br><br>Concat-ID<br><br>Wan-AdaLN<br><br>Phantom<br><br>14B<br><br>|[Figure 1086]|
|---|
<br><br>|[Figure 1087]|
|---|
<br><br>|[Figure 1088]|
|---|
<br><br>|[Figure 1089]|
|---|
<br><br>|[Figure 1090]|
|---|
<br><br>|[Figure 1091]|
|---|
<br><br>|[Figure 1092]|
|---|
<br><br>|[Figure 1093]|
|---|
<br><br>|[Figure 1094]|
|---|
<br><br>|[Figure 1095]|
|---|
<br><br>|[Figure 1096]|
|---|
<br><br>|[Figure 1097]|
|---|
<br><br>|[Figure 1098]|
|---|
<br><br>|[Figure 1099]|
|---|
<br><br>|[Figure 1100]|
|---|
<br><br>|[Figure 1101]|
|---|
|
|---|

P1.3B Vidu2.0

Kling1.6Vidu2.0Pika2.1 VACE

Kling1.6Pika2.1 VACE

P1.3B

E

E

B

B

### Figure 24: More Showcases in OpenS2V-Eval for Human-Domain Subject-to-Video Generation.

