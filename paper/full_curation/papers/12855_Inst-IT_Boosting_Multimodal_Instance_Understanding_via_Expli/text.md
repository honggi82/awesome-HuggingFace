# arXiv:2412.03565v3[cs.CV]16Mar2026

[Figure 1]

## Inst-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning

### Wujian Peng1,2*, Lingchen Meng1*, Yitong Chen1,2, Yiweng Xie1, Yang Liu1, Tao Gui1,2, Hang Xu3, Xipeng Qiu1,2, Zuxuan Wu1,2,† Yu-Gang Jiang1

1Institute of Trustworthy Embodied AI, Fudan University, 2Shanghai Innovation Institute, 3Huawei Noah’s Ark Lab

### Abstract

Large Multimodal Models (LMMs) have made significant breakthroughs with the advancement of instruction tuning. However, while existing models can understand images and videos at a holistic level, they still struggle with instance-level understanding that requires a more fine-grained comprehension and alignment. Instance-level understanding is crucial for LMMs, as it focuses on the specific elements that we are most interested in. Excitingly, existing works find that the state-of-the-art LMMs exhibit strong instance understanding capabilities when provided with explicit visual cues. Motivated by this, we proposed Inst-IT, a solution to enhance LMMs in Instance understanding via explicit visual prompt Instruction Tuning for instance guidance. Inst-IT consists of a benchmark to diagnose multimodal instance-level understanding, a large-scale instruction-tuning dataset, and a continuous instruction-tuning training paradigm to effectively enhance spatial-temporal instance understanding capabilities of existing LMMs. Experimental results show that, enhanced by InstIT, our models not only achieve outstanding performance on Inst-IT Bench and other instance understanding benchmarks, but also demonstrate significant improvements across various generic image and video understanding benchmarks. This highlights that our method not only boosts instance-level understanding but also strengthens the overall capabilities of generic image and video comprehension.

Correspondence: zxwu@fudan.edu.cn Website: https://inst-it.github.io

### 1 Introduction

Recently, Large Multimodal Models (LMMs) have seen remarkable advancements. A key breakthrough is visual instruction tuning [17, 43], enabling models to follow any type of user instructions. This paves the way to building general-purpose multimodal assistants capable of handling a wide range of realworld tasks [32]. Inspired by this initial work, numerous follow-up studies have emerged in both imagelanguage [9, 13, 44, 60, 106] and video-language [23, 51, 82, 84, 98] modeling. However, although they can understand images or videos at a holistic level, they still struggle to comprehend instance-specific content that the users are most interested in, as illustrated in figure 1 (a).

Instance-level understanding involves comprehending the attributes of individual instances within an image

∗Equal Contribution. †Corresponding authors.

Frame 1 Frame 2 Frame 3 Frame 4

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

3 2

1

…

[Figure 12]

[Figure 13]

Xt 3

[Figure 14]

What is the difference about the man and the woman between 1 and 3 ?

Xt 2

Xt 1

[Figure 15]

Xt

[Figure 16]

[Figure 17]

The difference is that the woman‘s shirt is white, matching the man's shirt.

[Figure 18]

[Figure 19]

[Figure 20]

…

[Figure 21]

The man and the woman are walking in the same direction in , but in , the woman is walking in the opposite direction.

[Figure 22]

1 3

[Figure 23]

⇥N

[Figure 24]

[Figure 25]

1

In , the and are walking together, in , the woman is standing near the car, and the man is no longer visible.

man woman

Ytf

[Figure 26]

3

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

[Figure 37]

[Figure 38]

Y vid

- (a)

(b)

⇥M

LLaVA-Next-Video ShareGPT4Video Our Model

[Figure 39]

Y qa

- Figure 1 (a) LMMs struggle with instance understanding, failing to capture the nuanced details of instances specified in user queries. (b) Our instance-centric data annotation pipeline, providing multi-level annotations for individual instances in images and videos.

or video, as well as the relationships and interactions between them. This requires models to exhibit nuanced comprehension and fine-grained alignment. Instance understanding has been a long-standing pursuit of the community with extensive efforts devoted to object detection [22, 58, 71, 73], instance segmentation [29, 59, 74], and object tracking [19, 76]. This capability is essential for real-world applications, where users pay more attention to the instances that they are interested in. In the era of LMMs, although there have been some attempts in exploring multimodal instance understanding [5, 24, 101, 102, 105], they are primarily limited in the image domain, leaving the videos under-explored. Compared to images, understanding instances in videos is considerably more challenging, as it requires not only capturing their spatial information but also temporal dynamics. Driven by this, we aim to advance the multimodal instance understanding in both images and videos. To this end, we focus on three aspects: instruction-tuning dataset, evaluation benchmark, and training recipe.

Existing multimodal benchmarks and datasets primarily provide coarse-grained knowledge for images and videos, lacking fine-grained annotations for individual instances. To address this, we introduce an automated pipeline to generate detailed instance-specific annotations. As illustrated in figure 1 (b), we leverage GPT-4o [61] to produce multi-level annotations, including instance-level descriptions, image-level captions, temporal dynamics, video-level summaries, and open-ended question-answer pairs. To fully unleash the capability of GPT-4o for more accurate annotations, we systematically design task prompts and employ set-of-marks visual prompts [88] to highlight instances in the visual inputs. Powered by this pipeline, we construct Inst-IT Dataset, an instance-grounded multimodal dataset comprising 51k images and 21k videos, 207k image-level captions, 135k temporal dynamics descriptions, 21k video-level captions, and 335k open-ended question-answer pairs. Furthermore, we carefully design the Inst-IT Bench to diagnose the instance-level understanding capabilities of LMMs, and perform rigorous manual verification and refinement to ensure its data quality.

Building on Inst-IT Dataset, we propose a continuous instruction tuning recipe that effectively integrates our instance understanding datasets with general instruction-tuning data. We augment images and videos with visual prompts, and convert the fine-grained annotations from Inst-IT Dataset into instruction tuning format, emphasizing the model’s spatiotemporal understanding of individual instances. Experimental results show that our enhanced models achieve strong instance understanding performance not only on Inst-IT Bench, but also demonstrate consistent improvements on other instance understanding benchmarks e.g. RefCOCOg [53]

and ViP-Bench [5]. We also investigate the models’ general comprehension capabilities on widely used generic benchmarks. The results reveal significant improvements over the baseline, achieving 4.4% and 13.5% gains on AI2D [28] and ChartQA [54] image benchmarks, as well as 7.8% and 11.8% improvements on Egoschema [52] and NExT-QA [85] video benchmarks, respectively. This highlights the effectiveness of Inst-IT in boosting instance understanding while strengthening general comprehension in both images and videos. Our contributions are three-fold:

- 1. We construct the Inst-IT Dataset, the first instance-grounded instruction-tuning dataset that includes both images and videos, featuring explicit instance-level visual prompts and fine-grained annotations grounded on individual instances.
- 2. We introduce the Inst-IT Bench, a human-verified benchmark specifically designed to evaluate the instance-level understanding capabilities of LMMs on both images and videos.
- 3. We propose a continuous instruction tuning recipe, which leverages our instance-level dataset alongside general data, effectively enhancing models in instance understanding while consistently improving general comprehension in both images and videos.

### 2 Related Work

Large multimodal models. Recently, significant progress has been witnessed in LMMs [91]. BLIP-2 [35] and Flamingo [2] leverage visual re-samplers to integrate image features as language inputs by extracting a fixed number of visual tokens. LLaVA [43] and its follow-ups [11, 30, 42, 44, 57, 60, 99] achieve remarkable success by connecting vision and language through a simple projection module. Additionally, researchers are extending LMMs’ capabilities to temporal understanding by incorporating multi-frame inputs [41, 82, 104] or explicit temporal modules [25, 37] However, existing LMMs struggle with instance-level understanding and often fail to accurately follow instructions to ground specific instances. We emphasize the importance of instance understanding and enhance it through instruction fine-tuning with explicit visual prompts.

Multimodal datasets and benchmarks. With the rapid progress in LMMs, numerous instruction-tuning datasets have been developed. LLaVA-Instruct [43] leverages object categories, bounding boxes, and image-level captions to generate diverse visual instruction tuning data. Follow-up studies use more powerful models to generate synthetic data [7, 9, 81] and improve the annotation pipeline [10, 36, 104]. Simultaneously, various benchmarks are proposed to evaluate LMMs across different aspects [21, 34, 40], such as comprehensive understanding [31], OCR [54–56, 78], temporal understanding [6, 20, 46, 47, 52, 85], and instruction-following [66]. However, they focus more on image or video-level understanding and lack fine-grained emphasis on specific instances. We emphasize the importance of instance understanding in both images and videos, and propose the Inst-IT Bench to evaluate the instance understanding of LMMs and create the Inst-IT Dataset, providing detailed instance-level annotations to enhance instance understanding.

Multimodal instance understanding. Understanding individual instances is a central focus in computer vision community, with key tasks like object detection [12, 71, 73], instance segmentation [29, 74], and object tracking [19, 50, 90]. In the era of LMMs, instance understanding gains increasing attention. SPEC [62], ARO[96], and Winoground [77] reveal that CLIP [67] struggle to understand instances. To address this, KOSMOS-2 [63], Ferret [92], GLaMM [69] and Shikra [8] encode instance information in textual form. In parallel, SoM-LLaVA [86], RegionGPT [24], GPT4ROI [101], MG-LLaVA [105], OMG-LLaVA [102], and ViP-LLaVA [5], explores the use of visual prompting to guide models in focusing on specific instances. SoM-LLaVA [86] and Elysium [79] are closely related to ours. SoM-LLaVA [86] asks models to list the instances in images, finding this effective in enhancing model comprehension. However, it is limited to the image domain. Elysium [79] focuses on object understanding in videos but employs relatively simplistic instance annotations. In contrast, we focus on both images and videos and provide multi-level fine-grained annotations for instances, aiming to advance multimodal models in understanding the spatiotemporal dynamics of individual instances.

### 3 Inst-IT

To address the scarcity of instance-grounded data, we propose an automated pipeline to generate detailed annotations for both images and videos, with a particular emphasis on instances of interest (Sec. 3.1). Based on this, we build a large-scale instance-grounded multimodal dataset (Sec. 3.2), and carefully design an instance-centric evaluation benchmark (Sec. 3.3). Furthermore, we propose a continuous instruction-tuning recipe (Sec. 3.4) to enhance LMMs in instance understanding.

#### 3.1 Instance-centric annotation pipeline

Overview. We propose an automated pipeline to generate annotations grounded on individual instances. As in figure 1 (b), we annotate each frame sequentially, aggregate frame-level annotations into a comprehensive video-level description, and generate open-ended question-answer pairs.

Visual prompting. Directly processing the raw visual inputs suffers from hallucinations and distraction. To mitigate this issue, we augment the images and videos with visual prompts to highlight the instances. Specifically, we use set-of-marks (SoMs) visual prompt [88], which overlays a numerical ID on each instance. We find this method highly effective in guiding GPT-4o to provide annotations focused on individual instances. For more details, please refer to Sec. A.1.

Frame-level annotation. We annotate video frames sequentially. At timestamp t, we provide GPT-4o with the current frame Xt, the previous frame Xt-1, and a tailored task prompt Pf. We then obtain a frame-level annotation Ytf= ytins,ytimg,ytdif encompassing three aspects, where ytins represents the captions for individual instances, ytimg is a caption for the entire image, and ytdif describes the temporal differences from the previous frame:

Ytf = GPT Pf,Xt,Xt−1 . (1)

Video-level summary. After obtaining annotations for each frame, we aggregate them into a caption for the entire video Y vid, capturing detailed spatiotemporal information of individual instances:

Y vid = GPT Pvid,Y1f,Y2f,··· ,YNf , (2) where Pvid is the task prompt designed for video-level summary and N is the total number of frames. Open-ended question-answer pairs. We also prompt GPT-4o with the task prompt Pqa to create M open-ended QA pairs Y qa={qi,ai}Mi=1 focusing on instance understanding:

Y qa = GPT Pqa,Y1f,Y2f,··· ,YNf . (3)

Following these steps, each video is enriched with multi-granularity annotations that incorporate instancespecific information. As illustrated in figure 2, these annotations include the following aspects:

- • N frame-level annotations, each contains detailed descriptions of individual instances, the entire image, and the temporal dynamics between adjacent frames.
- • A comprehensive description covering the entire video.
- • M open-ended question-answer pairs that focused on individual instances or their relationships.

Additional information about the design of each task prompt is provided in Sec. A.2.

[Figure 40]

[Figure 41]

###### Frame 1 Frame 2 Frame 3

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

- 2

- 3 4

1

2

###### Sample

- 3

- 4

3

4

Time

[Figure 49]

1 1

At , , a person in a dark outfit, is visible partially under a pavilion. , dressed in a light blue patterned top and dark pants, is bending forward towards , a small child in blue denim overalls. extends , a bright yellow ball, towards , who is standing nearby, seemingly reaching for the ball. At , has left the frame, and has shifted to an upright position…, has taken possession of .… The camera angle has also adjusted, showcasing the gravel path and the surrounding grassy area. At , is central in the frame, is facing away from the camera, slightly bent forward, with.…

2

Instance-level Captions:

3 2 4 3

###### Image-level Captions:

- 2

- 3

- 4

: a person wearing a light blue patterned top and dark pants. : a small child with short hair, wearing a blue denim overall. : a bright yellow ball being held by .

- 2

- 3 4

- 2 1 2

- 3 4

is standing upright, visible from the waist down, wearing dark pants and standing on a concrete and grassy surface. , a small child in denim overalls, holding , a yellow object, in its right hand. The scene appears to be in a partially outdoor…. The angle is elevated, showing the scene from a standing viewpoint, focusing on interaction between , , and ….

3

3 3

Temporal Change Captions:

- (b) Entire video description
- (c) Question-answer pairs

- 1

- 2

- 3 2

- 4 3 3

is no longer visible. has moved from a bending to an

2 3 2 4 3

Q: What object did hold towards in the first frame? A: held a bright yellow ball towards .

upright position and....

1 1 2

Q: What change occurred to between frames and ? A: is no longer visible possibly caused by camera movement.

has changed direction and appears to have moved further away from ....

1

1 2

Q: What happens to from frame to frame ? A: moved from bending to an upright, and is less visible.

2 2

is in the grasp of , suggesting has maintained its possession….

2 3 4

Q: …?

(a) Frame-level annotations at frame

2

###### Figure 2 Visualization of data structure in Inst-IT Dataset. For each video, we provide (a) N frame-level annotations,

- (b) a video-level description, and (c) M open-ended question-answer pairs. A complete example data can be found in Sec. C.3.

#### 3.2 Inst-IT Dataset

Instruction tuning plays a crucial role in multimodal training; however, the lack of instance-level datasets hinders the advancement of instance understanding. Using the data annotation pipeline described in Sec. 3.1, we create a large-scale instruction-tuning dataset, the Inst-IT Dataset. To the best of our knowledge, this is the first instruction fine-tuning dataset that provides multi-level fine-grained annotations centric on individual instances in both images and videos.

Data sources. We utilize five video instance segmentation datasets (BRUST [3, 18], UVO [83], OVIS [65], LVVIS [80] and YoutubeVIS-2021 [89]) and two object tracking datasets (BenSMOT [39], VidOR [75]) as our video sources, as they provide annotations of instance locations, which is useful in SoM visual prompting [88]. For the image source, we select the SA-1B [29] dataset due to its diversity and abundance of instance objects. In total, we collect 51k images and 21k videos. More details can be found in Sec. C.1.

Statistics. On average, each video includes one video-level annotation, 7.3 frame-level annotations, and 15.6 open-ended QA pairs. Images are regarded as single-frame videos without temporal changes. In total, Inst-IT Dataset includes 21k videos and 51k images, alongside 21k video-level captions, 207k frame-level captions, 836k instance-level captions, 135k temporal descriptions, and 335k open-ended QA pairs. More statistical analyses are provided in Sec. C.2.

Data quality. We employ three strategies to ensure the data quality: (1) High-quality visual prompts, we use manually annotated labels in segmentation and tracking tasks as SoMs to reduce noise. (2) Specialized prompt design, we introduce multi-level prompt engineering at the instance, image, two-frame, and video levels to mitigate long-term inconsistencies. (3) Diversity filtering, we filter out samples with few instances to enhance diversity and domain coverage. We randomly select 500 data samples and invite 3 volunteers to independently rate each sample with a score ranging from 1 to 5 (higher is better). The mean±std of scores and average time spent per sample are in Tab. 10. The average score is 4.49±0.05, indicating the satisfactory quality of our data. We use the maximum score difference (maxdiff) among volunteers to assess rating consistency.

- Table 1 Comparison of Inst-IT with existing datasets and instance understanding benchmarks. Left: Instruction tuning datasets. Right: Instance understanding benchmarks. Img and Vid indicate whether the data contains images or videos, respectively. Inst denotes the availability of instance-level annotations. OE and MC indicate open-ended and multiple-choice QA.

Img Vid Inst

Img Vid Task

ShareGPT4Video [10] ✓ LLaVA-Video [104] ✓ ViP-LLaVA-Data [5] ✓ ✓ Inst-IT Dataset ✓ ✓ ✓

RefCOCO [27] ✓ caption RefCOCOg [53] ✓ caption ViP-Bench [5] ✓ OE Inst-IT Bench ✓ ✓ OE&MC

49.8% of samples have maxdiff=0, and 78.6% maxdiff ≤ 1, showing high agreements on the ratings of different volunteers.

Comparison with existing instruction tuning datasets. Tab. 1 (left) compares Inst-IT Dataset with other datasets. Prior video datasets, e.g. ShareGPT4Video [10] and LLaVA-Video [104], focus on holistic understanding without instance-level annotations. While VIP-LLaVA [5] offers instance annotations for images, it does not include any video data. In contrast, Inst-IT Dataset encompasses both images and videos with multi-level, fine-grained annotations grounded on individual instances.

#### 3.3 Inst-IT Bench

Existing benchmarks primarily focus on global understanding, failing to provide more in-depth insights into the instance-level comprehension. We present the Inst-IT Bench, specifically designed to diagnose multimodal instance-level understanding in both images and videos.

Construction process. To prevent data leakage, we use videos from the test split, ensuring no overlap with Inst-IT Dataset. We apply the pipeline in Sec. 3.1 to generate 20 open-ended QA pairs for each image and video. Then, we manually review these QA pairs to ensure their accuracy, diversity, and difficulty. Overly simple questions are removed to ensure the remaining ones are instances-centric. We also refine the questions and answers, making necessary rephrasing to ensure correctness. After this rigorous checking process, each sample retains an average of 3.7 carefully polished QA pairs. In addition, we generate three hard negative for each question to construct a multiple-choice QA data with four options. More details are provided in Sec. B.1.

Statistics. Inst-IT Bench comprises 1,000 QA pairs for 338 images and 1,000 QA pairs for 206 videos. Each QA pair supports two evaluation formats, i.e. open-ended and multiple-choice.

Metrics. For open-ended QAs, we leverage GPT-4o to evaluate the response from a model based on its similarity to the ground-truth answer. For multiple-choice QAs, we calculate the average accuracy across all samples. More details about the metric calculations can be found in Sec. B.2.

Comparison with existing instance understanding benchmarks. Tab. 1 (right) highlights the main differences between Inst-IT Bench and existing instance understanding benchmarks such as RefCOCO [27], RefCOCOg [53] and ViP-Bench [5]: (1) its inclusion of evaluation data for both images and videos, pioneered the evaluation in video LMMs; and (2) it supports both open-ended and multiple-choice formats, enabling comprehensive evaluation.

#### 3.4 Instruction tuning with explicit visual prompt

Architecture. We adopt the widely-used LLaVA-NeXT [45] architecture to evaluate the effectiveness of our Inst-IT. We train our model under an image-video joint training pipeline, where we mix our Inst-IT Dataset with the open-source LLaVA-NeXT-DATA [48]. For single-image samples, we follow the original AnyRes paradigm [45] to split and encode sub-images according to the aspect ratio. For video and multi-image data,

- Table 2 Results on Inst-IT Bench. We conduct evaluations on Inst-IT Bench, including state-of-the-art open-source image models, video models, and cutting-edge proprietary models. #IT indicates the number of training samples used during the instruction-tuning stage. N/A indicates that the number is unknown. OE and MC represent open-ended and multiple-choice evaluations, respectively.

Image Video OE Q&A MC Q&A OE Q&A MC Q&A

Model LLM #IT

Random Guess - N/A - 25.0 - 25.0 GPT-4o [61] - N/A 74.1 84.8 65.5 81.0 Gemini-1.5-flash [72] - N/A 65.3 79.5 57.9 75.8

Open-source image models

LLaVA-1.5 [44] Vicuna-7B 665K 41.6 32.1 - ViP-LLaVA [5] Vicuna-7B ∼1.2M 42.1 29.2 - SoM-LLaVA [86] Vicuna-7B 695K 45.1 40.0 - LLaVA-NeXT [45] Vicuna-7B 765K 46.0 42.4 - -

Open-source video models

LLaVA-NeXT-Video [103] Vicuna-7B 860K 46.5 39.5 25.8 24.8 ShareGPT4Video [10] Llama3-8B ∼1.0M 43.2 48.7 27.8 16.1 LLaVA-OV (SI) [30] Qwen2-7B ∼7.2M 60.3 61.8 31.4 36.4 LLaVA-OV [30] Qwen2-7B ∼8.8M 48.0 71.7 33.2 45.6 LLaVA-Video [104] Qwen2-7B ∼7.4M 45.1 67.0 34.1 53.2 InternVL2 [13] InternLM2.5-7B N/A 58.6 66.5 39.8 45.5 Qwen2-VL-Instruct [82] Qwen2-72B N/A 55.5 74.7 45.5 74.6

Our models

LLaVA-NeXT-Inst-IT Vicuna-7B 920K 68.6 63.0 49.3 42.1 LLaVA-NeXT-Inst-IT Qwen2-7B 920K 67.9 75.3 45.7 53.3

we batch the samples together, encode them, and flatten them into a sequence. Additionally, we apply 2 × 2 spatial pooling to reduce the number of visual tokens in the video inputs. More details are in Sec. 4.1.

Converting Inst-IT Dataset into instruction tuning format. Inst-IT Dataset provides annotations at multiple levels of granularity. For the instance- and image-level captions in figure 2(a), we use a single frame as input and structure the task as a two-turn dialogue: the model is first prompted to describe all individual instances, followed by a holistic description of the entire scene. To capture temporal dynamics, we use temporal captions from figure 2(a), asking the model to describe the differences between two consecutive frames. The video-level description in figure 2(b) is treated as a captioning task, where the model is instructed to generate a summary based on all video frames. For the open-ended QA pairs in figure 2(c), we organize them into a multi-turn conversation, with the model answering one question per turn. In total, we construct 243k instruction tuning samples in the form of single-turn and multi-turn dialogues. All images and video frames are augmented with SoM visual prompts to explicitly provide instance-level guidance.

### 4 Experiments

#### 4.1 Implementation details

We use LLaVA-NeXT [45] as our baseline due to its widespread adoption. In the default configuration, Vicuna-1.5-7B [16] serves as the language model with CLIP-ViT-336 [68] as the vision encoder. We utilize the AdamW [49] with a cosine learning rate schedule for optimization. During the vision-language alignment stage, we use the LCS-558K dataset [44], and for the supervised fine-tuning stage, we leverage the open-source LLaVA-NeXT-DATA [48]. For single images, we split the original image into up to 4 sub-images based on its aspect ratio following the AnyRes [45] approach, and then concatenate the global image with these sub-images. For multiple images and video inputs, we skip the AnyRes procedure and encode every single

Table 3 Main results on image benchmarks.

AI2D[28] MMMU[95] POPE[38] GQA[26] ChartQA[54]

Method LLM Vision Encoder

(test) (val) (test F1) (val) (test)

LLaVA-1.5 [44] Vicuna-7B CLIP-ViT-Large 54.8 35.3 85.9 62.0 18.2 DeepStack-L [60] Vicuna-7B CLIP-ViT-Large - 35.7 86.7 63.1 21.0 DeepStack-L-HD [60] Vicuna-7B CLIP-ViT-Large - 35.6 86.5 65.2 56.3 VILA [42] Vicuna-7B CLIP-ViT-Large - - 85.5 62.3 LLaVA-OV (SI) [30] Qwen2-7B SigLIP-SO400M 81.6 47.3 - - 78.8 LLaVA-OV [30] Qwen2-7B SigLIP-SO400M 81.4 48.8 - - 80.0 Qwen2-VL-Instruct [82] Qwen2-7B DFN-CLIP-H 83.0 54.1 - - 83.0 LLaVA-NeXT [45] (baseline) Vicuna-7B CLIP-ViT-Large 66.6 35.1 86.4 64.2 54.8 LLaVA-NeXT-Inst-IT (ours) Vicuna-7B CLIP-ViT-Large 71.0 ↑4.4 37.4 ↑2.3 87.2 ↑0.8 65.9 ↑1.7 68.3 ↑13.5 LLaVA-NeXT-Inst-IT (ours) Qwen2-7B SigLIP-SO400 78.7 ↑12.1 42.7 ↑7.6 87.6 ↑0.2 65.5 ↑1.3 72.8 ↑18.0

Table 4 Main results on video benchmarks. We report the average of MCQA, Y/N and CM in TempCompass for determinism results. ∗ indicates results reproduced by us.

ANetQA[94] EgoSchema[52] NExTQA[85] VideoMME[20] TempCompass[47]

Method LLM Vision Encoder

(oe) (subset) (mc) (w/o subs) (3 avg)

DeepStack-L [60] Vicuna-7B CLIP-ViT-Large 49.3 38.4 61.0 - Video-ChatGPT [51] Vicuna-7B CLIP-ViT-Large 35.2 47.3 - - VideoLLaMA2 [14] Vicuna-7B CLIP-ViT-Large 50.2 - 51.7 - LLaVA-Next-Video [103] Vicuna-7B CLIP-ViT-Large 53.5 43.9 - 46.5 InternVL2 [13] InternLM-7B InternViT-300M - - - 54.0 LLaVA-OV [30] Qwen2-7B SigLIP-SO400M 56.6 60.1 79.4 58.2 69.4 LLaVA-Video [104] Qwen2-7B SigLIP-SO400M 56.5 57.3 83.2 63.3 Qwen2-VL-Instruct [82] Qwen2-7B DFN-CLIP-H - 66.7 - 63.3 72.9 LLaVA-NeXT [45] (baseline) Vicuna-7B CLIP-ViT-Large 53.8 50.0∗ 58.4∗ 36.2∗ 56.8∗ LLaVA-NeXT-Inst-IT (ours) Vicuna-7B CLIP-ViT-Large 53.7 ↓0.1 57.8 ↑7.8 70.2 ↑11.8 44.3 ↑8.1 59.8 ↑3.0 LLaVA-NeXT-Inst-IT (ours) Qwen2-7B SigLIP-SO400 55.2 ↑1.4 50.4 ↑0.4 73.0 ↑14.6 54.0 ↑17.8 63.9 ↑7.1

image. Additionally, we apply 2 × 2 spatial pooling to reduce the number of visual tokens for video inputs. We limit the maximum number of frames to 32 and the context length of LLMs to 6K due to GPU memory constraints. To enhance instance-level understanding with our Inst-IT Dataset, we combine Inst-IT Dataset with LLaVA-Next-DATA in an additional continuous supervised fine-tuning stage. In this stage, we freeze the first 12 layers of the vision encoder to mitigate potential distribution shifts caused by visually prompted images. Furthermore, we use Qwen2-7B [87] with SigLIP-SO400M-384 [97] for improved performance in our main experiment, and Qwen2-1.5B with CLIP-ViT-336 for efficiency in our ablation study. We use 8×H100 for all experiments. The image-video joint training stage takes approximately 20 hours when using Vicuna-7B as the language model and 24 hours using Qwen2-7B with SigLIP-SO400M-384.

#### 4.2 Main experiments

Results on Inst-IT Bench. We conduct extensive evaluations on Inst-IT Bench. The results in Tab. 2 show that with instruction tuning using Inst-IT Dataset, our models achieve a significant improvement of nearly 20% on average score, validating the effectiveness of Inst-IT. Moreover, although ViP-LLaVA [5] utilizes visual prompts for instruction tuning, it shows minor improvement over its baseline, i.e. LLaVA-1.5 [44], possibly due to overfitting to its training data. In contrast, our model demonstrates consistent improvements on other instance understanding benchmarks, such as ViP-Bench [5] and RefCOCOg [53] (Sec. 4.3), as well as on general-purpose evaluation sets like AI2D and Egoschema (will be discussed in the following sections). This suggests that the model trained with Inst-IT generalizes well to other tasks. Qwen2VL-72B does not show substantial improvements over its smaller 7B model, indicating that simply scaling up the model size

Table 5 Results on ViP-Bench. We perform evaluation with our Inst-IT models without fine-tuning.

Synthesized visual prompts Visual prompts from human Rec OCR Know Math Rel Lang All Rec OCR Know Math Rel Lang All

Model

GPT-4V-turbo-detail:high [1] 58.1 69.8 59.5 71.0 61.4 51.9 60.7 56.9 69.7 63.7 80.6 61.1 45.6 59.9 GPT-4V-turbo-detail:low [1] 53.2 50.3 55.6 67.7 57.5 57.5 52.8 51.7 50.3 59.3 60.3 55.0 43.8 51.4 InstructBLIP-7B [17] 36.9 16.3 34.2 22.3 26.8 7.5 31.7 38.9 17 35.4 9.7 29.3 17.5 33.3 Shikra-7B [8] 40.2 10.0 28.0 3.5 18.9 20.6 33.7 – – – – – – – GPT4ROI-7B [101] 35.6 16.7 29.7 9.7 32.5 13.8 35.1 – – – – – – – Kosmos-2 [64] 29.5 14.2 18.5 9.7 7.5 21.9 26.9 – – – – – – – LLaVA-1.5-7B [44] 50.8 12.4 49.2 6.5 51.8 23.8 41.6 49.1 13.0 42.9 9.7 50.0 27.5 40.2 Qwen-VL-Chat [4] 43.0 30.4 40.2 9.7 25.7 28.7 39.2 48.7 22.1 41.2 6.5 48.2 25.0 41.7 ViP-LLaVA-7B [5] 54.8 18.8 52.9 9.7 53.9 42.5 45.5 55.3 17.6 45.9 8.1 44.6 33.1 46.8 LLaVA-NeXT-Inst-IT-Vicuna-7B 51.3 23.7 54.2 12.9 64.3 46.2 45.1 55.0 21.3 52.5 16.1 57.5 40.6 48.2 LLaVA-NeXT-Inst-IT-Qwen2-7B 58.9 24.5 48.5 12.9 48.2 46.3 50.5 57.7 22.5 53.2 19.4 53.6 45.0 49.0

cannot address the challenges in instance understanding. Similarly, by comparing the amount of instruction tuning data used by each model, we observe that large-scale coarse-grained annotations do not lead to essential improvements either. This highlights the importance of instance-specific annotated data.

Results on generic benchmarks. To evaluate general understanding capabilities, we assess our models on several widely used image and video benchmarks using the LMMs-Eval [100]. To ensure a fair comparison with other models, we primarily report results from their original papers or reproduced results in previous studies. On generic image benchmarks, as shown in Tab. 3, Inst-IT consistently outperforms our direct baseline model, i.e. LLaVA-NeXT. The improvement in AI2D, a benchmark that requires grounding and referring understanding capability, is particularly clear. This suggests that Inst-IT effectively boosts the model in fine-grained understanding. Furthermore, when utilizing a more advanced language model and vision encoder, our method achieves performance comparable to large-scale SFT LMMs, such as LLaVA-OV and Qwen2-VL-Instruct, while requiring significantly less computational and data cost. For video understanding benchmarks in Tab. 4, Inst-IT significantly outperforms both LLaVA-NeXT and LLaVA-NeXTVideo. These consistent improvements demonstrate that enhancing instance-level understanding through explicit visual prompted instruction tuning is an effective strategy for improving generic spatiotemporal understanding capabilities.

#### 4.3 Evaluation on other instance-understanding benchmarks

To assess whether our model has learned generalizable instance understanding capability, we conducted evaluations on out-of-domain instance understanding benchmarks in zero-shot manner.

ViP-Bench [5] is a region-level understanding benchmark that closely aligns with the objectives of Inst-IT. As shown in Tab. 5, our model exhibits strong generalization performance. In particular, our Inst-IT with Vicuna-7B achieves performance comparable to ViP-LLaVA when using rectangular bounding boxes as visual prompts and even surpasses ViP-LLaVA when employing human-style visual prompts. Notably, our model performs as a generalist under zero-shot evaluation, whereas ViP-LLaVA benefits from in-domain tuning, since it is fine-tuned on the dataset of ViP-Bench.

RefCOCOg [53] is a referring expression comprehension benchmark, with fewer labeling errors than its counterpart RefCOCO [27]. We evaluate our LLaVA-NeXT-Inst-IT-Vicuna-7B model on this benchmark and observe a clear improvement of 10.8% over the baseline LLaVA-NeXT-Vicuna-7B (63.0% vs. 52.2%). This further confirms that our approach effectively enhances the model in instance understanding, rather than simply overfitting to our Inst-IT data format.

- Table 6 Ablation on data training recipe. L.N. denotes LLaVA-NeXT-Data, while Inst-IT img and Inst-IT vid refer to the image and video subsets of Inst-IT. Inst-IT-I and Inst-IT-V indicate the multi-choice splits of the image and video part of our Inst-IT Bench, respectively.

CL Tune Enc Data Combination

AI2D POPE GQA Inst-IT-I Next-QA VideoMME Inst-IT-V

(test) (test F1) (val) (mc) (mc) (w/o subt) (mc) All L.N. 61.1 86.9 61.4 45.3 56.6 45.7 31.3 All L.N. & Inst-IT vid 60.7 86.1 61.2 60.7 59.7 47.1 43.0

✓ All L.N. & Inst-IT vid 62.3 86.7 62.9 61.8 62.4 46.7 44.4 ✓ None L.N. & Inst-IT vid 63.1 86.9 62.5 60.2 63.2 47.2 44.3 ✓ Last 12 L.N. & Inst-IT vid 63.2 87.0 62.5 60.1 63.3 47.2 44.0 ✓ None L.N. & Inst-ITimgvid 63.0 87.0 62.7 58.6 59.8 46.7 41.6 ✓ Last 12 L.N. & Inst-ITimgvid 63.0 87.2 62.7 59.6 64.3 46.6 43.7

- Table 7 Ablation on detailed data combination. The dataset combination in line #3 corresponds to the video part of Inst-IT Dataset, while line #4 represents the complete Inst-IT Dataset by incorporating the image part into line #3.

AI2D MMMU POPE GQA Inst-IT-I Next-QA VideoMME Inst-IT-V

# Data Combination

(test) (val) (F1) (val) (mc) (mc) (w/o subt) (mc)

- 0 LLaVA-NeXT 61.1 35.9 86.9 61.4 45.3 56.6 45.7 31.3

- 1 inst-cap & img-cap 63.0 35.1 86.1 62.7 58.9 62.4 46.0 33.8

- 2 temporal diff 63.0 35.6 87.1 62.7 59.6 64.2 45.6 36.9

- 3 video-description & qa 63.2 34.9 87.0 62.5 60.1 63.3 47.2 44.0

- 4 Inst-IT Dataset img 63.0 36.1 87.2 62.7 59.6 64.3 46.6 43.7

#### 4.4 Ablation study

We use Qwen2-1.5B [87] as the language model and CLIP-ViT-L-336 [67] as the vision encoder for ablation experiments. We first conduct ablation on the training recipe to investigate how to effectively integrate Inst-IT Dataset with existing academic SFT datasets [48] for a balanced improvement. Next, we perform a detailed analysis of the impact of each component in our Inst-IT Dataset.

Effectiveness of our continuous instruction-tuning paradigm. As shown in Tab. 6, directly mixing the video split of Inst-IT Dataset with LLaVA-Next-DATA leads to significant improvements on video benchmarks. However, the performance on generic image understanding slightly declines. We believe this is due to two main reasons: (1) the increased ratio of video data may suppress image understanding; (2) visually prompted images may introduce a distribution shift from natural images. To address these issues, we propose a continuous SFT paradigm based on single-image models and freeze the first 12 layers of the vision encoder to preserve realistic low-level features. Our model achieves balanced performance across both image and video benchmarks with this training approach.

Detailed dataset combination. As illustrated in figure 2, Inst-IT Dataset contains fine-grained annotations at multi-level. To investigate the effectiveness of each component in Inst-IT Dataset, we conduct an extensive ablation by progressively adding data components. As shown in Tab. 7, the instance-level and image-level frame captions are essential for improving instance understanding in images. Meanwhile, temporal differences, along with video-level descriptions and QA, significantly enhance video instance understanding. Finally, incorporating the image component of Inst-IT Dataset enables our model to achieve the most balanced performance across generic image and video understanding benchmarks, as well as our Inst-IT Bench.

### 5 Conclusion

Instance understanding that detects, segments, and reasons nuanced relationships among objects has long been the goal of computer vision research, yet limited effort has been made to equip LMMs with such

capabilities. We introduced Inst-IT Bench, a carefully curated benchmark for evaluating multimodal instance understanding abilities. Extensive evaluations for a wide range of models demonstrate the limitations of current models for understanding at the instance level. To mitigate this issue, we collected Inst-IT Dataset, the first instruction-tuning dataset with explicit instance-level visual prompts and annotations. Based on InstIT Dataset, we proposed Inst-IT, a continuous finetuning framework that excels in instance understanding and general comprehension.

Acknowledgement This work was supported in part by the National Natural Science Foundation of China (Grant 62472098) and the Science and Technology Commission of Shanghai Municipality (No. 24511103100).

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.

- [3] Ali Athar, Jonathon Luiten, Paul Voigtlaender, Tarasha Khurana, Achal Dave, Bastian Leibe, and Deva Ramanan. Burst: A benchmark for unifying object recognition, segmentation and tracking in video. In WACV, 2023.

- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shĳie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 1(2):3, 2023.

- [5] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In CVPR, 2024.

- [6] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024.

- [7] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model, 2024.
- [8] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.

- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV, 2024.

- [10] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. In NeurIPS, 2024.

- [11] Yitong Chen, Lingchen Meng, Wujian Peng, Zuxuan Wu, and Yu-Gang Jiang. Comp: Continual multimodal pre-training for vision foundation models. ArXiv, 2025.

- [12] Yitong Chen, Wenhao Yao, Lingchen Meng, Sihong Wu, Zuxuan Wu, and Yu-Gang Jiang. Comprehensive multi-modal prototypes are simple and effective classifiers for vast-vocabulary object detection. In AAAI, 2025.

- [13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weĳie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.

- [14] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.

- [15] Cheng-Han Chiang and Hung-yi Lee. Can large language models be an alternative to human evaluations? In ACL, 2023.

- [16] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/2023-03-30-vicuna/.
- [17] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2023.

- [18] Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. Tao: A large-scale benchmark for tracking any object. In ECCV, 2020.

- [19] Patrick Dendorfer, Aljosa Osep, Anton Milan, Konrad Schindler, Daniel Cremers, Ian D. Reid, Stefan Roth, and Laura Leal-Taixé. Motchallenge: A benchmark for single-camera multiple target tracking. ĲCV, 129, 2020.

- [20] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

- [21] Chaoyou Fu, Yi-Fan Zhang, Shukang Yin, Bo Li, Xinyu Fang, Sirui Zhao, Haodong Duan, Xing Sun, Ziwei Liu, Liang Wang, et al. Mme-survey: A comprehensive survey on evaluation of multimodal llms. arXiv preprint arXiv:2411.15296, 2024.

- [22] Yuqian Fu, Yu Wang, Yixuan Pan, Lian Huai, Xingyu Qiu, Zeyu Shangguan, Tong Liu, Yanwei Fu, Luc Van Gool, and Xingqun Jiang. Cross-domain few-shot object detection via enhanced open-set object detector. In ECCV, 2024.

- [23] Yuqian Fu, Runze Wang, Yanwei Fu, Danda Pani Paudel, Xuanjing Huang, and Luc Van Gool. Objectrelator: Enabling cross-view object relation understanding in ego-centric and exo-centric videos. In ICCV, 2025.

- [24] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In CVPR, 2024.

- [25] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In CVPR, 2024.

- [26] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.

- [27] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014.

- [28] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.

- [29] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023.

- [30] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

- [31] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

- [32] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lĳuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. FTCGV, 2024.

- [33] Feng Li, Hao Zhang, Peize Sun, Xueyan Zou, Shilong Liu, Jianwei Yang, Chunyuan Li, Lei Zhang, and Jianfeng Gao. Semantic-sam: Segment and recognize anything at any granularity. In ECCV, 2024.

- [34] Jian Li, Weiheng Lu, Hao Fei, Meng Luo, Ming Dai, Min Xia, Yizhang Jin, Zhenye Gan, Ding Qi, Chaoyou Fu, et al. A survey on benchmarks of multimodal large language models. arXiv preprint arXiv:2408.08632, 2024.

- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.

- [36] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Ling-Yu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. In NeurIPS, 2024.

- [37] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In ECCV, 2024.

- [38] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.

- [39] Yunhao Li, Qin Li, Hao Wang, Xue Ma, Jiali Yao, Shaohua Dong, Heng Fan, and Libo Zhang. Beyond mot: Semantic multi-object tracking. In ECCV, 2024.

- [40] Zongxia Li, Xiyang Wu, Hongyang Du, Huy Nghiem, and Guangyao Shi. Benchmark evaluations, applications, and challenges of large vision language models: A survey. arXiv preprint arXiv:2501.02189, 2025.

- [41] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2024.

- [42] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, 2024.

- [43] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.

- [44] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.

- [45] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. URL https://llava-vl.github.io/blog/2024-01-30-llava-next/.
- [46] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. Et bench: Towards open-ended event-level video-language understanding. In NeurIPS, 2024.

- [47] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024.

- [48] lmms lab. Llava-next-data, 2024. URL https://llava-vl.github.io/blog/2024-01-30-llava-next/.
- [49] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

- [50] Wenhan Luo, Junliang Xing, Anton Milan, Xiaoqin Zhang, Wei Liu, and Tae-Kyun Kim. Multiple object tracking: A literature review. AI, 2021.

- [51] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024.

- [52] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In NeurIPS, 2023.

- [53] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L. Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, 2016.

- [54] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

- [55] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021.

- [56] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, 2022.

- [57] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Anton Belyi, et al. Mm1: methods, analysis and insights from multimodal llm pre-training. In ECCV, 2025.

- [58] Lingchen Meng, Xiyang Dai, Yinpeng Chen, Pengchuan Zhang, Dongdong Chen, Mengchen Liu, Jianfeng Wang, Zuxuan Wu, Lu Yuan, and Yu-Gang Jiang. Detection hub: Unifying object detection datasets via query adaptation on language embedding. In CVPR, 2023.

- [59] Lingchen Meng, Shiyi Lan, Hengduo Li, José M. Álvarez, Zuxuan Wu, and Yu-Gang Jiang. Segic: Unleashing the emergent correspondence for in-context segmentation. In ECCV, 2024.

- [60] Lingchen Meng, Jianwei Yang, Rui Tian, Xiyang Dai, Zuxuan Wu, Jianfeng Gao, and Yu-Gang Jiang. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms. In NeurIPS, 2024.

- [61] OpenAI. GPT-4o system card, 2024.

- [62] Wujian Peng, Sicheng Xie, Zuyao You, Shiyi Lan, and Zuxuan Wu. Synthesize diagnose and optimize: Towards fine-grained vision-language understanding. In CVPR, 2024.

- [63] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos2: Grounding multimodal large language models to the world. ArXiv, abs/2306.14824, 2023. URL https: //api.semanticscholar.org/CorpusID:259262263.

- [64] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.

- [65] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. ĲCV, 2022.

- [66] Yusu Qian, Hanrong Ye, Jean-Philippe Fauconnier, Peter Grasch, Yinfei Yang, and Zhe Gan. Mia-bench: Towards better instruction following evaluation of multimodal llms. arXiv preprint arXiv:2407.01509, 2024.

- [67] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

- [68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

- [69] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In CVPR, 2024.

- [70] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

- [71] J Redmon. You only look once: Unified, real-time object detection. In CVPR, 2016.

- [72] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Rohan Anil, Sebastian Borgeaud, Andrew M. Dai, Katie Millican, Ethan Dyer, Mia Glaese, Thibault Sottiaux, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, James Molloy, Jilin Chen, Michael Isard, Paul Barham, Tom Hennigan, Ross McIlroy, Melvin Johnson, Johan Schalkwyk, Eli Collins, Eliza Rutherford, Erica Moreira, Kareem W. Ayoub, Megha Goel, Clemens Meyer, Gregory Thornton, Zhen Yang, Henryk Michalewski, Zaheer Abbas, and etc. Nathan Schucher. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv preprint, 2024.

- [73] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. TPAMI, 2015.

- [74] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. ArXiv, 2015.

- [75] Xindi Shang, Donglin Di, Junbin Xiao, Yu Cao, Xun Yang, and Tat-Seng Chua. Annotating objects and relations in user-generated videos. In ICMR, 2019.

- [76] Yuedong Tan, Zongwei Wu, Yuqian Fu, Zhuyun Zhou, Guolei Sun, Eduard Zamfi, Chao Ma, Danda Pani Paudel, Luc Van Gool, and Radu Timofte. Xtrack: Multimodal training boosts rgb-x video object trackers. ArXiv, 2024.

- [77] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In CVPR, 2022.

- [78] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. Pattern Recognition, 2023.

- [79] Han Wang, Yongjie Ye, Yanjie Wang, Yuxiang Nie, and Can Huang. Elysium: Exploring object-level perception in videos via mllm. In ECCV, 2024.

- [80] Haochen Wang, Cilin Yan, Keyan Chen, Xiaolong Jiang, Xu Tang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. Ov-vis: Open-vocabulary video instance segmentation. ĲCV, 2024.

- [81] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv preprint arXiv:2311.07574, 2023.

- [82] Peng Wang, Shuai Bai, Sinan Tan, Shĳie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

- [83] Weiyao Wang, Matt Feiszli, Heng Wang, and Du Tran. Unidentified video objects: A benchmark for dense, open-world segmentation. In ICCV, 2021.

- [84] Zuxuan Wu, Zejia Weng, Wujian Peng, Xitong Yang, Ang Li, Larry S Davis, and Yu-Gang Jiang. Building an open-vocabulary video clip model with better architectures, optimization and data. TPAMI, 2024.

- [85] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In CVPR, 2021.

- [86] An Yan, Zhengyuan Yang, Junda Wu, Wanrong Zhu, Jianwei Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Julian McAuley, Jianfeng Gao, et al. List items one by one: A new data source and learning paradigm for multimodal llms. In COLM, 2024.

- [87] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Ke-Yang Chen, Kexin Yang, Mei Li, Min Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shĳie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yunyang Wan, Yunfei Chu, Zeyu Cui, Zhenru Zhang, and Zhi-Wei Fan. Qwen2 technical report. ArXiv, abs/2407.10671, 2024.

- [88] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023.

- [89] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In ICCV, 2019.

- [90] Alper Yilmaz, Omar Javed, and Mubarak Shah. Object tracking: A survey. CSUR, 2006.

- [91] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 2024.

- [92] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. In ICLR, 2024.

- [93] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lĳuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024.

- [94] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, 2019.

- [95] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.

- [96] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Y. Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In ICLR, 2023.

- [97] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.

- [98] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In EMNLP, 2023.

- [99] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024.

- [100] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024.

- [101] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023.

- [102] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Change Loy Chen, and Shuicheng Yan. Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding. In NeurIPS, 2024.

- [103] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, 2024. URL https://llava-vl.github.io/blog/ 2024-04-30-llava-next-video/.
- [104] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.

- [105] Xiangyu Zhao, Xiangtai Li, Haodong Duan, Haian Huang, Yining Li, Kai Chen, and Hua Yang. Mg-llava: Towards multi-granularity visual instruction tuning. ArXiv, 2024.

- [106] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. ArXiv, 2023.

### Appendix

- • In Sec. A, we outline additional implementation details of the GPT-4o-assisted data annotation pipeline.
- • In Sec. B, we present further information about the instance understanding benchmark, Inst-IT Bench.
- • In Sec. C, we share more details about the instruction fine-tuning dataset, Inst-IT Dataset.
- • In Sec. D, we provide more discussions on failure cases and real-world applications.

### A Data Annotation Pipeline

- A.1 Set-of-Marks Visual Prompting

Performing instance-level annotations is challenging, and we adopt the SoM visual prompting technique [88] to address this. Specifically, as illustrated in figure 3, we overlay a numeric ID at the center of each instance and maintain the same ID for a given instance across all frames. This simple augmentation can explicitly guide GPT-4o to focus more effectively on the instances of interest, enabling finer-grained and more accurate annotations. Furthermore, segmentation masks are necessary to calculate the center coordinates of each instance. Details on how these masks are obtained are provided in Sec. C.1.

[Figure 50]

[Figure 51]

- Figure 3 Set-of-Marks visual prompting on the original videos. Each instance is assigned a unique numeric ID, which remains consistent across all frames.

- A.2 Prompting GPT-4o

Task prompt templates. Prompt engineering is crucial for enabling GPT-4o to accomplish specific tasks. In this section, we present the task prompts that we designed to prompt GPT-4o for data annotation:

- • The task prompt Pf for frame-level annotation, figure 5.
- • The task prompt Pvid for video-level annotation, figure 6.
- • The task prompt Pqa for open-ended question-answer pairs generating, figure 7.

GPT-4o API version. During the annotation process, we use the GPT-4o-2024-08-06 API and leverage its structured output functionality to facilitate output parsing, enabling the model to respond in a predefined JSON format.

### B More Details about Inst-IT Bench

# Task Description : You are an expert evaluator tasked with scoring the accuracy of responses to open-ended questions. You will be provided with a set of questions, each with a corresponding ground-truth answer, as well as responses from a tester. Your job is to assess the accuracy of each response and provide a score between 0 and 1.

###### # Guidelines:

- - Score Range: Your score for each test item must be between 0 and

1. A higher score means more correctness. Choose from the following: 0 (completely incorrect), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (completely correct)

- - For each test item, consider the question, the ground-truth answer, and the tester’s response together to determine correctness.
- - Objects in questions and answers may be referenced using the format [ID] (e.g., [1], [2]). Ensure that any objects referenced in the tester’s response match correctly with the ground-truth answer.

- - Time points may be indicated with <timestamp> (e.g., <1>), and time intervals with <start_timestamp>-<end_timestamp> (e.g., <3>-<5>). Verify that the tester’s response includes accurate time expressions.

# Input Format: The input is a set of test items to be scored, where each item includes:

- - id: the unique identifier for the test item;
- - question;
- - ground-truth answer for the question;
- - response from the tester. Now, let's begin the evaluation, here are the input test items: <samples to be scored> …

- Figure 4 GPT-4o-based open-ended question answering correctness assessment. The underlined parts in the figure are included only when evaluating the video split, while the italicized parts will be replaced by the actual sample for scoring.

###### Frame-level Annotation Task Prompt

# Task Description: You are an expert in video analysis, skilled at detecting dynamic changes between consecutive video frames. In this task, you are given two consecutive frames. Each image contains objects with unique numeric IDs (referred to as "marked objects"). Your task is to:

- - Provide object-level descriptions for each marked objects in the current frame.
- - Provide a dense and detailed image-level description for the entire current frame.
- - Identify any dynamic changes or differences between the current frame and the previous frame. # Guidelines for Object-level Descriptions:
- - Describe each marked object’s appearance in the current frame, focus on attributes like color, shape, textual, size.
- - If you are confident, specify the category to which the object belongs in the description, i.e., answer what the object is. # Guidelines for Image-level Descriptions:
- - Mention all the marked objects in the current frame, focusing on the behaviors, movements, states, positions, and other dynamic information.
- - Describe the interactions between the objects, as well as the background, environment, perspective, and angle of the shot. # Guidelines for Describe the Temporal Changes:
- - Highlight changes in each marked object, such as movements, actions, status, position, as well as object interactions or relationships.
- - Note any changes in the background , environment, camera angles and scene transitions.
- - Reasonably infer the causes of the changes, trends, and possible impacts. # Constraints:
- - Accuracy is critical: If a marked object is too small or obscured, and you cannot confidently identify it, skip it without attempting to describe it.
- - Frame of reference: Describe movement direction, or object position from the camera's point of view.
- - Specify what the interaction is, do not simply saying "[1] is interacting with [2]", you should say "[1] is catching [2]" .
- - Object Referring Format: When refer to a single object, use the format: [1]; when listing multiple objects, use the format: "[1] [2] [3]".

# Structured Output: Your output should have three sections:

- - Object-level Descriptions: For each marked objects in the current frame, provide a comprehensive description of its appearance.
- - Image-level Description: Provide a dense and comprehensive description of the entire image, capturing as many details as possible.
- - Temporal Changes: Outline any changes and differences compared to the previous frame, highlighting important transitions or events.

# Input: You will receive two consecutive images: <the previous frame image> <the current frame image>

Figure 5 Frame-level annotation task prompt, the italicized part are placeholders for the actual inputs.

###### Video-level Annotation Task Prompt

# Task Description: You are an expert in summarizing video content. Given a sequence of frame-by-frame text descriptions of a video. Your task is to aggregate these descriptions into an accurate, cohesive summary of the entire video.

###### # Guidelines and Rules:

- - Base your description solely on the input to ensure accuracy; avoid inferring any unmentioned content.
- - Please note that the description of a single frame may contain some inaccuracies. You need to use the overall context to further correct these errors, ensuring accuracy and consistency.
- - Use chronological order: organize your summary according to the timestamps of the frames, follow these conventions: for specific moments, write <timestamp>, e.g., at <3>; for time intervals: write <start_timestamp>-<end_timestamp>, e.g., during <5>-<7>
- - Referencing objects by ID: in your response, use the same [ID] format provided in the input to reference objects: for one object: [ID] (e.g., [8] a white dog); for multiple objects: [ID1] [ID2] ... (e.g., [3] [4] [5]).

# Output Requirements: Your output should be a dense, detailed, and accurate description of the entire video, summarizing main objects, key events, and various spatial and event-related details.

# Input Format: Each frame’s description includes four parts:

- 1. Timestamp: marks the chronological position of the frame in the video.
- 2. Instance-level description: lists the primary objects in the frame using the format “[object ID]: object description”
- 3. Frame-level description: offers a comprehensive view of the frame’s content, covering main objects, object relationships, and the background or environment details.
- 4. Temporal change description: highlights key changes or movements since the previous frame, capturing dynamic information essential for understanding the video’s progression.

# Input Frame-level Annotations:

- Timestamp: <1>; Instance-level description: … ; Frame-level description: … ; Temporal changes: None, as this is the first frame.
- Timestamp: <2>; Instance-level description: … ; Frame-level description: … ; Temporal changes: ... …

Figure 6 Video-level annotation task prompt, the italicized part are placeholders for the actual inputs.

Open-Ended QA Generation Task Prompt

# Task Description : You are an expert in video content analysis. In this task, you will receive textual descriptions of individual video frames. Your task is to generate high-quality and contextually coherent questions and accurate answers based on the content of the video.

###### # Guidelines:

- - Avoid speculative questions; ensure all questions can be answered from the frame descriptions.
- - Diversify the types of questions (who, what, where, when, how, why) to cover different aspects of the video.
- - The number of question-answer pairs should between 10 to 20, this depends on how much valuable information contained in the video.
- - Be creative and flexible in forming questions and answers, and avoid redundant or overly simple questions.
- - Use the frame timestamps to express time in the video: for a specific moment, use <timestamp> , e.g., at <3>; for a time interval, use <start_timestamp>-<end_timestamp>, e.g., during <5>-<7>. Don't forget to enclose the timestamps in <>.
- - In the input, ID is used to refer to a specific object; you can use the same format in your output to refer to specific objects: for a single object, write [ID] (e.g., "[8]"); for multiple objects, use "[ID1] [ID2] ...", such as "[3] [4] [5]".

# Output: The output is a list of 10 to 20 high-quality, context-aware question-answer pairs about the video's content.

# Input Format: The input consists of frame-by-frame descriptions, where each frame includes:

- 1. Timestamp: marks the chronological position of the frame in the video.
- 2. Frame-level description: offers a comprehensive view of the frame’s content, covering main objects, object relationships, and the background or environment details.
- 3. Temporal change description: highlights key changes or movements since the previous frame, capturing dynamic information essential for understanding the video’s progression.

# Input Frame-level Annotations:

- Timestamp: <1>; Frame-level description: … ; Temporal changes: None, as this is the first frame.
- Timestamp: <2>; Frame-level description: … ; Temporal changes: ... …

###### Figure 7 Open-ended question-answer pairs generation task prompt, the italicized part are placeholders for the actual inputs.

timestamp <7>

timestamp <4> timestamp <5>

timestamp <10>

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

7

7

3 3 7

3

- 1

2 2 2

- 2

3

1

4 4

1

4

… … … …

1

4

8

5

5

Open-Ended QA

Open-Ended QA

Q: What action does [2] take at the scene captured at <10>? A: At <10>, [2] bends over, reaching toward the ground, indicating they are searching for or picking up something.

Q: Which individuals are no longer visible in the later frames of the video?

A: [8] disappears from the frame at <5>, and [5] disappears from the frame at <7>.

Multiple-Choice QA

Multiple-Choice QA

Q: Which individuals are no longer visible in the later frames of the video?

Q: What action does [2] take at the scene captured at <10>?

- A. The individuals who were no longer visible in the later frames of the video were the man and the woman who were standing next to the horse.
- B. The individuals no longer visible in the later frames of the video are those labeled as [3] and [5].
- C.
- D. [3]

- A. At the scene captured at <10>, [2] is petting the horse.
- B. [2] is seen holding a white cloth at the scene captured at <10>.
- C. Petting the horse
- D.

At <10>, [2] bends over, reaching toward the ground, indicating they are searching for or picking up something.

[8] disappears from the frame at <5>, and [5] disappears from the frame at <7>.

###### Figure 8 A data example from Inst-IT Bench. Each test sample includes both open-ended QA and multiple-choice QA, focusing on specific instances or the relationships and interactions between instances.

Table 8 Data sources. We use seven video datasets and one image dataset as our data sources. We show their annotation formats, the splits we used, and the number of samples from each dataset.

Dataset Name Ann. Type Split Sample Num.

Video Instance Segmentation BRUST [3] mask training 500 UVO [83] mask training 5,135 OVIS [65] mask training 599 LVVIS [80] mask training 3,057 YoutubeVIS [89] mask training 2,897

Video Object Tracking

BenSMOT [39] box training 2,261 VidOR [75] box training 6,969

Image SA-1B [65] none 1-10 51,101

#### B.1 Negative Options Generation

We use the ground-truth from open-ended QA as the positive option and additionally craft three negative options, forming a multiple-choice question with four options. To create hard negatives, we first have the model answer the open-ended questions and use GPT-4o to score the correctness of the responses. If the score is lower than 0.4, we consider it a difficult negative answer and include it as one of the negative options. Finally, we randomly shuffle the four options to ensure that the correct one appears in each position with equal probability.

#### B.2 LLM-based Evaluator for Open-Ended QA

Recent studies [15, 93] suggest that LLMs can serve as effective evaluators. Building on this, we use GPT-4o to assess the accuracy of open-ended question answering. Specifically, GPT-4o assigns a score between 0 and 1 based on three key factors: the question, the ground-truth answer, and the model prediction. Given that Inst-IT Bench prioritizes instance-level understanding, we pay special attention to the accuracy of instance ID references. Furthermore, for the video split of Inst-IT Bench, we emphasize the correctness of timestamps to ensure temporal correctness. The task prompt for GPT-4o is illustrated in figure 4.

#### B.3 Data Example

To provide a clearer understanding of Inst-IT Bench, we present a data example in figure 8. Each question includes both open-ended and multiple-choice formats, focusing on specific instances or exploring the relationships and interactions between multiple instances. This design highlights the significant distinction from other benchmarks, emphasizing fine-grained understanding at the instance level.

### C More Details about Inst-IT Dataset

#### C.1 Data Collection and Processing

Collection. We select five instance segmentation datasets and two multi-object tracking datasets as sources of video data. To prevent data leakage, we only used the training splits of these datasets, leaving their test and validation splits untouched. Additionally, we use the SA-1B [29] dataset as a source of image data and only utilize the first ten officially provided data splits. For each split, we only use the first 50% of its images. In total, we collect 21,418 videos and 51,101 images. Tab. 8 provides detailed statistics on our data sources.

Processing. When constructing SoM [88] visual prompts, we need to obtain the mask annotations for each instance to determine the location of the numeric IDs. For the video instance segmentation datasets [3, 65,

Video Split Image Split All Sample

0.30

| |
|---|

| |
|---|

0.25

[Figure 56]

0.20

Density

0.15

0.10

0.05

0.00

1 3 5 7 9 11 13 15

Number of Instances per Sample

- Figure 9 The distribution of the number of instances per sample in Inst-IT Dataset. We separately present the distribution for the video split, image split, and the entire dataset.

Figure 10 Analysis of object categories in InstIT Dataset, which shows a diverse range of types spanning multiple domains.

80, 83, 89], the instance masks are already provided and can be used directly. For multi-object tracking datasets [39, 75], we prompt SAM [29] with their bounding box annotations to generate instance masks. For images in the SA-1B dataset [29], we employ Semantic-SAM [33] to segment the instances and obtain their masks.

#### C.2 Statistics Analysis.

Number of instances. The key characteristic of Inst-IT Dataset is its specific focus on individual instances in images and videos, which provides a more fine-grained description of the visual inputs. We visualize the distribution of the number of instances in each sample in figure 9. For the video split, each sample has an average of 3.7 instances, with a total of 79,709 instances. For the image split, each sample contains an average of 6.9 instances, totaling 351,495 instances. Across the entire dataset, each sample includes an average of

- 5.9 instances, adding up to 431,204 instances in total. We measure the scene complicity by the number of instances in each sample. Specifically, 31% of the samples contain ≤ 3 instances (simple), 39% have between 3 to 8 instances (medium), and the remaining 30% contain ≥ 8 instances (hard).

Dataset diversity. We visualize the object categories in Inst-IT Dataset in figure 10, highlighting its diverse range. The objects include humans, animals, plants, vehicles, landmarks, etc. , covering domains like daily life, egocentric perspectives, sports, transportation, etc. . The rich diversity of Inst-IT Dataset ensures its applicability to real-world scenarios and enhances its transferability to different domains.

Text captions. Inst-IT Dataset contains multi-level textual descriptions of visual content, covering instances, frames, temporal changes, and video-level annotations. We conduct statistical analysis on these text annotations, including the number of each type of text, and their average length. As shown in Tab. 9, the average length of Inst-IT Dataset is 49.1 words per caption, with video-level averaging 323.2 words, highlighting its richness of details. We also present the results of lexical analysis in Tab. 9. The instance-level captions contain a rich variety of nouns and adjectives, indicating that they primarily describe the objects’ categories and attributes. The captions of temporal changes include a high volume of verbs and adverbs, suggesting that they capture dynamic information.

Human evaluation of data quality We invited three volunteers to rate each sample on a scale from 1 to 5, with higher scores indicating better quality. Tab. 10 presents the scores of different types of annotations, along with the average time spent by each volunteer to evaluate each sample. The average score across all types is 4.49±0.05, indicating that the data in Inst-IT Dataset is of satisfactory quality.

Table 9 Statistical and lexical analysis of Inst-IT Dataset. We present the results for each annotation level as well as the entire dataset.

Caption Type #Caption #Char./Cap. #Word/Cap. #Sen./Cap. Nouns Adj. Adv. Verb. Prep. Instance-level 836,524 102.1 24.3 1.5 26.5% 13.3% 2.3% 12.3% 10.7% Frame-level 207,662 458.0 106.5 5.7 25.2% 10.5% 2.6% 14.9% 11.5% Temporal-change 135,143 306.6 67.7 3.7 21.2% 10.0% 6.0% 16.4% 10.8% Video-level 21,372 1441.8 342.2 14.3 24.8% 10.6% 3.6% 13.2% 11.8% All 1,200,701 210.5 49.1 2.7 25.0% 11.4% 3.1% 14.0% 11.1%

Table 10 Human evaluation on the quality of Inst-IT Dataset.

Instance Caption Image Caption Temporal Caption Video Caption QA Pairs

Score (↑) 4.66±0.12 4.68±0.02 4.48±0.05 4.34±0.18 4.31±0.11 Time (s) 7.3 12.4 11.9 31.0 10.6

#### C.3 Data example.

In this section, we provide a complete video data sample from Inst-IT Dataset to offer a clearer understanding of its content and format. In all annotations, we use the format [ID] to refer to instances and <timestamp> to refer to timestamps. We present the frame-level annotations in Tab. 11. We can see that each frame-level annotation Y f consists of three parts: instance-level descriptions yins, image-level descriptions yimg, and temporal differences ydif. Additionally, each video is accompanied by a series of open-ended questionanswer pairs Y qa, most of which center on specific instances or their relationships, as illustrated in Tab. 12. Furthermore, we generate a dense video-level caption Y vid summarizing the entire video in chronological order, as shown in Tab. 13.

### D More discussions.

- D.1 Failure cases.

We manually inspect the dataset and model to identify the failure cases. We find that occasional failures occur in scenarios where instances are severely occluded, the image is blurry, or instances are excessively small or crowded. These challenges are common among LMMs, and future research can further investigate them.

- D.2 Real-world applications.

In real-world applications, users can interactively prompt models like SAM2 [70] to automatically track instances of interest and generate SoMs. Additionally, our model also supports inputs without SoMs, allowing users to specify particular instances using textual descriptions. In the first scenario, our Inst-IT introduces only a marginal overhead for generating SoMs, while in the second case, it incurs no extra cost compared to the base model.

### E Limitations and broader impacts.

Limitations. Our current experiments are conducted on 7B and 1.5B models due to the computation cost. Moreover, our current data pipeline is automated but constrained by the overhead of GPT-4o. We can further scale the model size and scale the dataset using a model-in-the-loop approach and improve the model through multi-round instruction tuning with self-synthesized data. We leave this direction for future work.

Broader impacts. This paper proposes an enhancement of instance-level understanding capabilities in large multimodal models, enabling them to better assist users by answering questions about the content of interest. However, similar to existing large multimodal models, this approach also faces potential risks, such as issues

Table 11 Inst-IT Dataset frame-level annotations. For the ease of visualization, we only demonstrate the first three frames. Please zoom in to view the instance ID labels.

Frame Instance-level captions Image-level captions Temporal differences

[Figure 57]

- timestamp<1>

1: Wearing a light gray suit with a white shirt, standing indoors. 2: Wearing a sleeveless white lace dress, holding an object in the hand. 3: Wearing a dark floralpatterned dress with long wavy hair.

[1] [2] [3] are standing closely together in an indoor setting. [1] is on the left side wearing a formal, light gray suit with a white shirt. [2], in the middle, is wearing a sleeveless white lace dress, holding something in their hand. [3] is on the right side in a dark floral-patterned dress with long, wavy hair. They appear to be in a room with wooden paneling and some framed art on the wall. null

[Figure 58]

- timestamp<2>

1: A person wearing a gray suit with a white shirt, short hair. 2: A person in a white, sleeveless dress with long dark hair. 3: A person wearing a dark floral dress with long dark hair. 5: A person wearing red, partially visible in the background. 6: A small black cellphone held in a hand.

The scene appears to be in an office setting with a wooden table at the foreground. [1] is standing to the left, facing [2], and appears to be holding [2]’s finger or hand. [2] stands slightly to the right, returning focus with [1]. [3] is to the right of [2], slightly in the background, smiling and looking forward. A bouquet of white flowers lies on the table near [2]. [5] is partially visible in the background on the right, seated and wearing red. [6] is a cellphone held by [5]. Background shows a wooden wall and a reflection in a window.

- [1] has moved closer to [2] and is now in contact with
- [2]’s hand. [2] has turned slightly towards [1] compared to the previous frame.
- [3] remains in a similar position, but the expression suggests more engagement with the scene. [5] and [6] have appeared in the frame; [5] is visible in the background holding [6]. The table with a bouquet of flowers is now visible, indicating a shift in camera angle slightly to include more of the right side of the room.

[Figure 59]

- timestamp<3>

The scene shows [1] [2] [3] near a wooden conference table in a professional setting, possibly an office. [1] wears a grey suit and is standing to the left, engaged with [2] who is wearing a white dress and holding flowers. [3], who is in a patterned dress, stands closely behind [2]. The newly appeared [4] is seated to the far left, partially visible at the edge of the frame. [5] is seated on the right side, holding an object above their head, possibly obscuring their face. The room has wooden walls and a framed picture hanging on the wall.

1: Wearing a grey suit, standing beside [2] and slightly turned towards them. 2: Wearing a white, sleeveless dress with floral textures. Holding a bouquet of white flowers. 3: Wearing a dark patterned dress, standing slightly behind [2]. 4: Partially visible, wearing dark clothing, located at the edge of the left side of the frame. 5: Seated, wearing a red outfit. Holding a white object above their head, possibly obscuring their face.

Object [5] has lifted an object above their head, possibly a piece of paper. Object [4] has appeared in the scene, seated on the left side of the frame, which was not visible earlier. The positions of objects [1], [2], and [3] remain unchanged, as does the background and setting of the room. Overall, no significant movement is noticed in terms of camera angle or position for objects [1] [2] [3].

Table 12 Inst-IT Dataset Open-ended question-answer pairs.

##### Question Answer

Whatchangeoccurswith[1]’sexpressionbetween <10> and the previous frame?

[1] changes from smiling to a neutral expression.

What activity are [1] and [2] involved in at <11>? [1] and [2] are engaged in a kiss. What is the overall mood during <11> as suggested by [3]’s actions?

A celebratory or joyous event.

What interaction occurs between [1] and [2] at

- <5>?

[1] holds [2]’s hand, suggesting an intimate gesture or exchange, likely a ring.

Who joins [1] and [2] in the frame at <7>? [4] appears in the frame, joining [1] and [2]. What changes in the group’s composition between <7> and <8>?

[3] reappears, and [4] is no longer visible.

What element is seen throughout the frames <1> to <12>?

The scene is in an indoor setting with wooden paneling and framed art.

What type of event is likely taking place based on the atmosphere in <4> and <6>?

A formal event, possibly a wedding or official gathering.

What new elements are introduced in the scene at <2>?

[5] holds a cellphone in the background, partially visible.

What is the mood and lighting like at <6>? The mood is formal and celebratory, with bright

lighting enhancing this atmosphere. What new background element appears at <7>? There is a map or blueprint on the wall. What is notable about [5]’s actions at <3>? [5] is lifting an object above their head, possibly a

piece of paper. What is the setting like in <3>? The group is gathered near a wooden conference table in a formal setting. How are [1] and [2] interacting at <8>? They are engaged in conversation or communication, indicated by body language and focus. What does [1]’s expression suggest at <12>? [1] speaks or smiles, suggesting engagement with [2] or others.

What shift occurs in the focus of the camera between <5> and <6>?

The camera focuses more on individuals standing together, reducing focus on the foreground objects.

What are [3] and [4] doing at <9>? They clapping their hands in celebration. What decorative element is visible at <2>? A bouquet of flowers lies on the table near [2]. How has the posture of [1] and [2] changed by

- <6>?

[1] and [2] face slightly outward, suggesting a pose for a photograph or audience.

What overall physical change occurs between [1] and [2] from <10> to <11>?

There’s a noticeable increase in their physical interaction, enhancing emotional engagement.

Table 13 Inst-IT Dataset video-level caption.

##### Video-level caption

The video appears to document a formal or celebratory event indoors, possibly a ceremony such as a wedding or official gathering, occurring in a room with wooden paneling and art or framed pictures on the wall. At the beginning, during <1>, [1] is wearing a light gray suit and stands with [2] in a sleeveless white lace dress, and [3] in a dark floral-patterned dress. The three are close together, suggesting an intimate or focused setting. The progression between <2> and <3> involves subtle changes in posture and interaction. [1] moves closer to [2], appearing to hold hands or engage in an exchange, possibly involving a ring, as indicated by a bouquet of flowers. [3] remains supportive and smiling, while [5], in red, momentarily holds an object above their head, before disappearing from view by <4>.In frames <5> to <7>, [1] and [2] maintain a close interaction, suggestive of a significant moment such as an exchange of vows or rings. They are closely observed by [3], who stands smiling nearby, while [1] and [2] occasionally adjust their positions, facing each other initially and then turning outward, which may signal transitioning from an intimate moment to posing for a photo. By <7>, [4] joins, dressed in darker attire, emphasizing the formal setting as [3] is no longer visible. Through <8> and <9>, the group dynamics change slightly with the absence of [4] and [3] entering the scene again. [1] and [2] appear to engage in a warm interaction as [3] supports them, clapping, alongside the visible hands of [4] indicating applause, marking a cheerful tone. Finally, during <10> to <12>, the focus shifts as [1] and [2] first engage in a kiss, underscoring an intimate conclusion to their ceremony. They later stand apart slightly at the center, with [1] smiling or speaking, and [2] leaning towards [1] suggestively content. Throughout, the consistent joyous mood is accentuated by [3]’s ongoing clapping and expression of joy, emphasizing shared celebration and approval from the audience captured.

related to fairness and bias. Future work can address this issue through approaches such as data filtering and validation.

