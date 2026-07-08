## MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation

# arXiv:2511.22989v2[cs.CV]26Mar2026

Yuta Oshima1,∗ Daiki Miyake1,∗ Kohsei Matsutani1 Yusuke Iwasawa1 Masahiro Suzuki1 Yutaka Matsuo1 Hiroki Furuta2,†

1The University of Tokyo 2Google DeepMind

{yuta.oshima, daiki.miyake}@weblab.t.u-tokyo.ac.jp

``On the left, the woman from Image 1 is sitting on the sofa from Image 4. In the center, the character from Image 2 and the banana from Image 7 are placed on the table from Image 3. On the right, the girl from Image 5 is writing on a whiteboard. The text on the whiteboard is converted into the

language used in Image 6, and the pattern of the girl's skirt matches the one in Image 8.’’

|[Figure 1]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Nano Banana<br><br>[Figure 2]|
|---|

|[Figure 3]<br><br>Reference 2|
|---|

|[Figure 4]<br><br>| |
|---|
<br><br>GPT<br><br>[Figure 5]|
|---|

|[Figure 6]<br><br>Reference 3|
|---|

|[Figure 7]<br><br>Reference 1|
|---|

|[Figure 8]<br><br>Reference 4|
|---|

Generate!

[Figure 9]

|[Figure 10]<br><br>Reference 6|
|---|

|[Figure 11]<br><br>Reference 7|
|---|

|[Figure 12]<br><br>Reference 8|
|---|

|[Figure 13]<br><br>Reference 5|
|---|

``The woman in Image 1 is on the left, the giraffe in Image 2 is in the center, and the anime

``The man from image 1, the woman from image 2, and the woman on roller skates from image 3

``The man from image 1 and the woman from image 2 are standing on a snowy landscape, and the red bird from image 3 is perched on a branch in front of them. The background of the

character in Image 3 is on the right. The cat from Image 4 is in the foreground, in front of the

are in front of a bridge with autumn trees and buildings in the background. The style of the image is the same as in image 4.’’

[Figure 14]

woman.’’

image is the same as in image 4.’’

|[Figure 15]<br><br>Reference 1|
|---|

|[Figure 16]<br><br>Reference 2|
|---|

|[Figure 17]<br><br>Reference 1|
|---|

|[Figure 18]<br><br>Reference 2|
|---|

|[Figure 19]<br><br>| |
|---|
<br><br>[Figure 20]<br><br>GPT|
|---|

|[Figure 21]<br><br>Reference 1|
|---|

|[Figure 22]<br><br>Reference 2|
|---|

|[Figure 23]<br><br>| |
|---|
<br><br>[Figure 24]<br><br>Nano Banana|
|---|

|[Figure 25]<br><br>GPT|
|---|

|[Figure 26]<br><br>Reference 3|
|---|

|[Figure 27]<br><br>Reference 4|
|---|

|[Figure 28]<br><br>Reference 3|
|---|

|[Figure 29]<br><br>Reference 4|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

Reference 3 Reference 4

``The mythical creature from Image 1 is standing majestically in the center of the scene. The chef from Image 2 is positioned to the left, looking towards the creature with a friendly smile. To the right of the creature, the two mice from Image 3 are cautiously approaching. The pocket watch from Image 4 is resting on a surface in the foreground, slightly to the right, appearing as if it has

``The woman in image 1 should be combined with the bird in image 3, and the woman's dress

``The man from image 3, wearing the colorful sash from image 2, is seated at the ornate

just been placed there.’’

should have the color and texture of the bird's feathers. The text in image 2 should be

golden table from image 1, with the pearl from image 4 resting on the table's surface.’’

[Figure 32]

rewritten by the font and style of the text in image 4.’’

|[Figure 33]<br><br>| |
|---|
<br><br>[Figure 34]<br><br>Nano Banana|
|---|

|[Figure 35]<br><br>Reference 1|
|---|

|[Figure 36]<br><br>Reference 1|
|---|

|[Figure 37]<br><br>Reference 2|
|---|

Reference 2

|[Figure 38]<br><br>Nano Banana|
|---|

|[Figure 39]<br><br>Reference 1|
|---|

|[Figure 40]<br><br>Reference 2|
|---|

|[Figure 41]<br><br>| |
|---|
<br><br>[Figure 42]<br><br>GPT|
|---|

|[Figure 43]|
|---|

|[Figure 44]<br><br>Reference 3|
|---|

|[Figure 45]<br><br>Reference 4|
|---|

|[Figure 46]<br><br>Reference 3|
|---|

|[Figure 47]<br><br>Reference 4|
|---|

|[Figure 48]<br><br>Reference 3|
|---|

|[Figure 49]<br><br>Reference 4|
|---|

Figure 1. The overview of MultiBanana. MultiBanana broadly covers problems specific to multi-reference settings, including varying the number of references (top row), domain and scale mismatches among references (two on the left in the middle row), multilingual text rendering (center in the bottom row), and the presence of rare concepts (right in the bottom row).

#### Abstract

Recent text-to-image generation models have acquired the ability of multi-reference generation and editing; that is, to inherit the appearance of subjects from multiple reference images and re-render them in new contexts. However,

*Equal contribution †Work done as an advisory role only.

existing benchmark datasets often focus on generation using a single or a few reference images, which prevents us from measuring progress in model performance or identifying weaknesses when following instructions with a larger number of references. In addition, their task definitions are still vague, limited to axes such as “what to edit” or “how many references are given”, and therefore fail to capture the challenges inherent in combining heterogeneous refer-

ences. To address this gap, we introduce MultiBanana, which is designed to assess the edge of model capabilities by widely covering problems specific to multi-reference settings: (1) varying the number of references (up to 8), (2) domain mismatch among references (e.g., photo vs. anime), (3) scale mismatch between reference and target scenes, (4) references containing rare concepts (e.g., a red banana), and (5) multilingual textual references for rendering. Our analysis among a variety of text-to-image models reveals their respective performances, typical failure modes, and areas for improvement. MultiBanana is released as an open benchmark to push the boundaries and establish a standardized basis for fair comparison in multi-reference image generation. Our data and code are available at https: //github.com/matsuolab/multibanana.

#### 1. Introduction

Recent advances in image generation, driven by large-scale multimodal LLM backbones, have enabled unprecedented levels of instruction-following [7, 13, 22, 56, 79–81, 83– 85]. Among these developments, multi-reference image generation has emerged as a notable capability. When a user provides multiple reference images, the model can inherit the subject’s appearance and re-render it in a new context while faithfully preserving identity. This shift moves image generation beyond single-reference conditioning toward greater controllability, driving increasing demand in industrial domains such as content production [66, 78, 90], advertising [28, 33, 51], and fashion design [11, 16, 98], where reference-guided generation is essential.

Despite this progress, we lack a benchmark to evaluate how well models follow instructions that involve multiple references under varying conditions. As shown in Table 1, existing benchmarks [3, 30, 48, 66, 69, 77, 80, 83, 91– 93] suffer from two fundamental limitations. First, existing benchmarks limit the number of references to a narrow range (typically 1–4) and do not examine how well models perform when following instructions with a larger number of references. Second, their task definitions are still vague, limited to axes such as “what to edit” or “how many references are given”, and therefore fail to capture the challenges inherent in combining heterogeneous references.

To address these limitations, we introduce a new evaluation framework that is grounded in the core challenges of multi-reference generation. Our benchmark incorporates multiple orthogonal axes specific to the multi-reference regime, including broader coverage of the number of references (up to 8), domain gaps between references (e.g., photo–anime mixtures), mismatches in object scale between reference and target scenes, rare concepts in the references (e.g., a red banana), and multilingual textual rendering references (English, Chinese, and Japanese).

Using this benchmark, we analyze model capabilities, identify characteristic failure modes, and highlight remaining gaps. For example, our analysis reveals that as instruction difficulty increases (through more references, crossdomain mixtures, or rare concepts), closed-source models (e.g., Nano Banana) tend to incorporate all required subjects but suffer from compositional distortion due to overfitting to reference details, while open-source models (e.g., Qwen-Image) generate visually cleaner outputs but often omit some reference subjects altogether. We release MultiBanana as an open-source benchmark to establish a standard foundation for fair comparison and to accelerate progress in multi-reference image generation.

#### 2. Related Works

##### 2.1. Controllable Text-to-Image Generation

Diffusion models have achieved state-of-the-art performance in high-fidelity image synthesis [26, 70]. Large pretrained diffusion models with textual conditioning, such as Stable Diffusion [15, 60, 65] and FLUX [41, 42], can generate novel images while preserving the identity depicted in the reference images. Building on these developments, approaches based on fine-tuning (e.g., DreamBooth [66]) and adapter-based methods (e.g., IP-Adapter [90]) have been explored to improve the controllability [4, 18, 34, 44, 50, 72]. Recent research on multimodal LLM-based image generation has gained momentum, aiming to handle multiple image generation tasks within a single model. For example, GPT-Image-1 [56] and Nano Banana [22] can jointly process text and image inputs, enabling integrated image generation and editing within a unified framework. Similarly, open-source models such as Qwen-Image [79] and FLUX.1-Kontext-dev [42] demonstrate high-quality, flexible image generation and editing. Beyond these, numerous open-source unified generative models continue to emerge [80, 83–85]. These advancements demonstrate that diffusion models are becoming flexible and adaptable across diverse conditional generation settings.

##### 2.2. Benchmarks for Reference-Based Generation

Among reference-driven image generation tasks, image editing is the most widely recognized: it involves generating an edited image from a single reference image and a textual instruction describing how the edit should be performed. Representative benchmarks such as MagicBrush [93] and EMU-Edit [69] have extended task-specific evaluations but still rely heavily on similarity-based metrics. SmartEdit [30] supports editing of complex scenes, but does not sufficiently cover more general settings. Similarly, I2EBench [48] employs GPT-4o to provide human-aligned evaluations across a variety of editing tasks, yet it uses different metrics for each task and therefore fails to capture the

Table 1. Comparison among major benchmarks for reference-based image generation. Existing benchmarks do not provide systematic evaluation across diverse multi-reference conditions, support only a limited number of references, and fail to adequately account for important factors such as differences and compatibility among reference images. Our benchmark expands the upper limit on the number of references and introduces difficult reference combinations, including domain mismatches, scale mismatches, rare concepts, and multilingual prompts, thereby explicitly addressing challenges unique to multi-reference image generation.

###### Benchmark #Size #Tasks #References Difficult Reference Combination Metrics

EditBench [77] 240 1 1 CLIP [62] EditVal [3] 648 13 1 CLIP, VLM, manual EmuEdit [69] 3055 7 1 L1, CLIP, DINO [8] MagicBrush [93] 1053 9 1 L1, L2, CLIP, DINO AnyEdit [92] 1250 25 1 L1, CLIP, DINO I2EBench [48] 2240 16 1 GPT [54] ImgEdit-Bench [91] 811 14 1 GPT (3 dim.), Fake Det. [87] DreamBooth [66] 75 1 1 CLIP. DINO OmniContext [80] 400 8 3 GPT (3 dim.) DreamOmni2 [83] 319 29 4 Gemini [19], Doubao [5] MultiBanana (Ours) 3769 36 8 GPT, Gemini (5 dim.)

shared characteristics of editing as a whole. ImgEdit [91] has enabled evaluation of multi-turn editing tasks, in which image generation and editing are alternated interactively. One of the earliest benchmarks for reference-based image generation beyond instruction-based editing is the DreamBooth [66] dataset. However, such benchmarks are still limited to generation tasks conditioned on a single reference image and thus cannot handle more complex or compositional settings. Recently, advances demonstrated by GPTImage-1 [56] and Nano Banana [22] have drawn increasing attention to multi-reference image generation, where multiple reference images are jointly used for generation. Benchmarks such as OmniContext [80] and DreamOmni2 [83] have been proposed to evaluate this task. Nevertheless, these benchmarks encompass only a small number of images and task types, thereby limiting the scope of the evaluation. Moreover, although designed for multi-reference generation, they do not account for the heterogeneous nature of reference images. This challenge is analogous to instruction following in LLMs [23, 24, 35, 40, 97], where performance degrades as the number of constraints grows; multi-reference image generation similarly requires models to satisfy all given references without neglecting any.

#### 3. MultiBanana

##### 3.1. Task Types

In reference-driven image generation, tasks can be categorized based on the number and relational structure of reference images. In this work, we follow the overall design philosophy of prior benchmarks such as DreamOmni2 [83], but extend the number of references up to 8 to enable a more fine-grained analysis of scalability. We also go beyond task-

category-based evaluation by incorporating axes that capture the intrinsic properties of the reference set itself, such as domain gaps between references, scale mismatches, rare concepts, and multilingual text rendering.

Single Reference. This is the simplest setting, corresponding to the standard image editing task. Given a single reference image and a textual instruction, the model generates an edited image that maintains semantic consistency and preserves the subject’s visual identity. This configuration follows traditional instruction-driven editing benchmarks such as MagicBrush [93] and EMU-Edit [69], which represent the foundation of reference-driven generation.

Two References – 11 Task Types. The two-reference setting provides the model with two reference images (typically a main subject and an auxiliary one) and a textual instruction. We define 11 editing task types, following prior work, while balancing feasibility and challenge: subject addition, subject replacement, background change, color modification, material modification, pose modification, hairstyle modification, makeup modification, tone transformation (lighting, season, weather, etc.), style transfer (oil painting, sketch, anime, etc.), and text correction (in-image text editing or replacement). These tasks span both local attribute manipulations and global transformations, evaluating a model’s ability to maintain visual coherence, subject identity, and contextual adaptability.

Multi References – Compositional Generation and Inter-Reference Relationships. The multi-reference setting uses 3–8 reference images, requiring the model to perform more compositional generation. We define four base task structures, each evaluated with {3,4,5,6,7,8} reference images, resulting in 24 tasks:

• X Objects: Compose X objects from the reference set.

Person

- Task 1

- Task 2

- Task 3

Real Data

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

``Change the color of the skirt in Image 1 to the color of

the jacket in Image 2.’’

[Figure 55]

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

Object

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

``The lifeguard from Image 1 stands behind the man from Image 2. The style of the image is the same as in Image 3.’’

[Figure 74]

| |
|---|

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Synthetic Data

[Figure 82]

Background

[Figure 83]

[Figure 84]

|[Figure 85]<br><br>[Figure 86]|
|---|

[Figure 87]

[Figure 88]

[Figure 89]

``The cat in Image 1, …, and the man from Image 6 are gathered under the nebulous sky with planetary rings.’’

[Figure 90]

[Figure 91]

[Figure 92]

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

[Figure 105]

Removed (too small)

| |∙∙∙| |
|---|---|---|
| | | |

| |∙∙∙| |
|---|---|---|
| | | |

∙∙∙

∙∙∙

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Image Collection Image Filtering

Category Classification

Task Construction

- Figure 2. Construction pipeline for our benchmark, consisting of four stages: (1) collecting high-quality real and synthetic images, (2) filtering out inappropriate or low-quality samples, (3) performing hierarchical category classification, and (4) generating and validating editing instructions by Gemini and humans.

- • X–1 Objects + Background: Place X −1 objects within the background of another reference.
- • X–1 Objects + Local: Apply a local change (specified by a reference) to one of the X − 1 objects.
- • X–1 Objects + Global: Arrange X −1 objects and modify the overall tone according to another reference.

This design enables us to assess whether a model can integrate multiple reference sources into a consistent scene.

Diversity Factors in Multi-Reference Generation. A central contribution of our benchmark is the explicit incorporation of inter-reference diversity factors, which have been largely overlooked in previous benchmarks such as DreamOmni2 [83]. We highlight the following dimensions:

- • Cross-domain diversity: mixing references from different domains such as real photos, anime, CG, or sketches.
- • Scale and viewpoint differences: variation in object size or camera perspective between references and outputs.
- • Rare concept references: inclusion of uncommon concepts (e.g., “a red banana”).
- • Multilingual references: assessing multilingual instruction understanding during rendering.

These factors go beyond prior benchmarks that focus on simple reference composition; our benchmark evaluates whether models can interpret relationships among heterogeneous references and integrate them into coherent outputs.

##### 3.2. How to Construct MultiBanana

The construction pipeline is illustrated in Figure 2.

Image Collection. The reference images used in our benchmark consist of both real and synthetic data. All real images were extracted from the LAION-5B dataset [68]. To ensure high quality, we selected only images with an Aesthetic score [43] higher than 6.25 and a resolution greater than 512 pixels. However, the collected real images exhibited a distributional bias (Figure 3; left). Specifically, land-

scape scenes appeared frequently, whereas humans and objects were underrepresented. While landscape images can serve as useful background references, the scarcity of human and object references limits the diversity of editing tasks. To address this imbalance, we generated additional synthetic images under diverse conditions using Nano Banana [22] and GPT-Image-1 [56], and used them as supplementary reference images (Figure 3; right). This approach not only mitigated the categorical bias in the reference set but also expanded its coverage and diversity by incorporating both real and synthetic examples.

Image Filtering. Among the collected images, some contained small subjects that were difficult to use as references, or were inappropriate for reference use, such as images containing harmful content, tables, or corrupted data. To filter out such images and retain only those suitable for a reference-based image editing benchmark, we applied the following procedure. Following [91], we extracted images that contained large and clearly visible objects suitable for editing tasks. Object detection was performed using YOLOv12 [73] and SAM2 [64], while CLIP [62] was used to verify semantic consistency. Additionally, harmful or corrupted images were removed through a combination of automatic filtering with Gemini and manual inspection, resulting in the final curated set of real images.

Category Classification. To construct multi-reference image generation tasks, we categorized each reference image. For example, in a portrait transformation task that modifies a person’s hairstyle or makeup, both reference images must contain a person. Similarly, in replacement tasks involving humans or objects, both reference images must include either a person or an object, and for background replacement tasks, images categorized as background must be used. To satisfy such task-dependent conditions, we performed hierarchical image classification on a combined

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Real only Real + Synthetic

800

NumberofImages

Male Female Hairstyle Face Expression Makeup Pose Useful

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

600

400

Animal Animal Animal Animal Animal Animal

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

200

Industrial Design Industrial Design Clothing Design Clothing Design Object Object

0

AnimalClothing DesignIndustrialDesign

Object MaleFemaleFace Expression HairstyleMakeupPose Useful

Tone Light StyleBackground

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Object Person Tone Light Style Background

Light Light Tone Tone Background Background

- Figure 3. (Left) Comparison between the statistics of real data only and those after adding synthetic data. The original dataset was biased toward background images, with few person- and object-related samples. To correct this imbalance, we generated additional synthetic images using Nano Banana and GPT-Image-1, focusing on clear subjects such as people, animals, and objects. This results in a more balanced and comprehensive distribution of data. (Right) Examples of synthesized images in each category.

set of real and synthetic images. We first obtained automated multi-level labels using Gemini, then conducted human verification to ensure correct categorization. At the top level, we defined six major categories: person, object, background, light, tone, and style. Each of these top-level categories was further divided into more specific subcategories. For instance, under the person category, we included male, female, hairstyle, makeup, pose useful, and face expression; and under the object category, we defined animal, clothing design, industrial design, and object. This hierarchical labeling enabled systematic task construction.

Task Construction. The editing instructions were generated using Gemini. For each task, we manually selected suitable reference categories, randomly sampled images, and presented them to Gemini to generate corresponding editing instructions. Gemini then evaluated whether following each instruction would lead to visual breakdowns. For instance, in a replacement task, whether the replaced object would appear unnaturally suspended in midair. Among the generated candidates, we retained only the reference–instruction pairs that did not exhibit visual breakdowns and then asked Gemini and human annotators to further filter out tasks that were nonsensical or too trivial.

Difficult Reference Categorization. For the crossdomain category, we selected samples whose reference images spanned at least two domains (e.g., color photos, anime, paintings). The different-scale category included samples with both close-up and non–close-up views. The rare-concept category contained samples featuring animals with unusual colors or patterns, humans or animals with atypical makeup or clothing, or fictional creatures. The multilingual category included samples with text in two different languages, including English, Chinese, and Japanese.

##### 3.3. Statistics in MultiBanana

Image Category Statistics. This section presents a statistical overview of both the real and synthetic source data used to construct our benchmark. The dataset is organized into six major categories and thirteen subcategories based on visual and semantic attributes.

In the real dataset, the largest major category is background, comprising 798 images, one-fourth of the total. These mainly consist of images without distinct subjects. The next-largest categories are person (741 images) and style (579 images), and together these three categories account for 70% of the dataset. In contrast, light (195 images), object (413 images), and tone (118 images) are small in number. Although these categories are useful as references for changing global conditions such as illumination and overall atmosphere, their sample sizes remain limited. Within the object category, there are 111 images of animals, 40 of clothing and textile design, 71 of industrial design, and 191 of miscellaneous objects, indicating that data focusing on specific materials or design patterns is scarce. Most images in the background, style, and tone categories lack clear subjects. Furthermore, images containing textual elements, such as logos or signs, are nearly absent. The person category is subdivided into finer attributes: face expression (10), female (181), male (139), hairstyle (135), makeup (59), and pose usefulness (217). While pose-related data account for the largest portion (around 30%), reference data capturing subtle characteristics such as facial expression or makeup remain limited, resulting in a skewed distribution.

To address this imbalance, synthetic data were generated, focusing primarily on images containing explicit subjects (e.g., persons, animals, and objects) and on those including text. The generation process was designed to intentionally expand the coverage of underrepresented subcategories and improve the overall balance among attributes.

[Figure 135]

500

8

3 4

7

6

400

5

5

X 1 Objects + Background

- 3
- 4

6

300

###### Count

X Objects

200

- 7
- 8

- 6
- 7
- 8

X 1 Objects + Local

100

X 1 Objects + Global

3

5

0

4

3 4 5 6 7 8

4

5

Number of References

3

6

7 8

X 1 Objects

X 1 Objects

X 1 Objects

X Objects

+ Background

| |
|---|

| |
|---|

| |
|---|

+ Global

+ Local

- Figure 4. (Left) Breakdown of the multi-reference tasks. The editing sets were selected to ensure that the number of sets per task is balanced across reference counts. (Middle) For every X–references task, the dataset contains at least 390 editing sets. Furthermore, each colored task category includes at least 70 sets, which is larger than in prior work [83]. (Right) Word cloud generated from all prompts. It primarily consists of terms that describe a wide range of object categories and words indicating spatial directions.

As a result, the distribution of major categories changed: background (802), light (729), object (2041), person (2437), style (592), and tone (649). At the subcategory level, significant increases were observed: animal (111→915), clothing design (40→396), industrial design (71→385), and face expression (10→146), indicating a notable enrichment in person- and object-related data.

Consequently, the dataset became more balanced between background-centric and subject-centric images, with broader coverage of diverse visual attributes. This improvement enhances the benchmark’s comprehensiveness and enables more versatile evaluation and instruction-based generation tasks under a wide range of conditions.

Task Statistics. After filtering the editing instructions using both Gemini and manual checks, we obtained 3,769 high-quality reference images and instruction sets. Among them, 264 sets correspond to single-reference tasks, 907 to two-reference tasks, and 2,598 to multi-reference tasks. Figure 4 provides a further breakdown of the multireference tasks, showing that they are evenly distributed with respect to the number of reference images. Additionally, for each task type, X Object, X–1 Object + Background, X–1 Object + Local, and X–1 Object + Global, we ensured that at least 70 sets were included. This is larger than the number of sets in multi-reference tasks of OmniContext [80] and DreamOmni2 [83], enabling more reliable benchmarking. Furthermore, Table 2 shows that our newly introduced evaluation axes – cross-domain, scale and viewpoint differences, rare concept, and multilingual – provide sufficient samples. See Appendix C for details.

##### 3.4. Evaluation Setting

For evaluating the quality of edited images, human evaluation is among the most valuable methods, yet gathering human feedback at scale is prohibitively costly. A practical approach to reducing time and cost is to leverage AI feedback from VLMs (e.g., Gemini 2.5 [19] and GPT-5 [57]), which has been shown to align reasonably with human judgments

Table 2. Statistics and ratios of difficult reference combinations relative to the total 3,769 tasks. Our benchmark provides sufficient samples for assessing a model’s capacity to interpret relationships among heterogeneous references and integrate them into outputs.

Cross Different Rare MultiDomain Scale & View Concept lingual

Count 1063 1357 743 99 Ratio (%) 28.2% 36.0% 19.7% 2.6%

of image quality [17, 38, 53, 58, 82]. Therefore, we used Gemini 2.5 and GPT-5 to rate all generated images.

In our evaluation protocol, all generated images are assessed against five criteria to evaluate the performance of multi-reference image generation: Text-Instruction Alignment, Reference Consistency, Background-Subject Match, Physical Realism, and Visual Quality. Text-Instruction Alignment and Reference Consistency are fundamental criteria for multi-reference image generation and core components of many prior benchmarks [80, 91]. In contrast, Background-Subject Match, Physical Realism, and Visual Quality refine what previous work typically treated as a single “overall quality” dimension [80, 91], offering a more fine-grained assessment of holistic visual fidelity. Each of the five criteria is scored on a 10-point scale (10 is the best). We then compute a total weighted score, assigning weights of {3, 3, 1, 1, 1} to {Instruction Alignment, Reference Consistency, Background-Subject Match, Physical Realism, Visual Quality}, respectively.

#### 4. Experiments

We adopt several closed models (i.e., Nano Banana [22], GPT-Image-1 [56]) and open models (i.e., Qwen-ImageEdit-2509 [79], DreamOmni2 [83], OmniGen2 [80]) for MultiBanana evaluation. For all models except OmniGen2, we generate outputs for every combination of reference images and text prompts included in the MultiBanana benchmark. Since OmniGen2 does not support more than 6 ref-

Table 3. Average performance per model across different task categories (10-point scale; the higher the better). The average scores from Gemini 2.5 and GPT-5 are reported. Both open-source and closed-source models exhibit notably lower performance on background replacement tasks, regardless of the number of reference images.

###### Model Single Two X Objects X-1 + Local X-1 + Global X-1 + Background

Nano Banana 7.817 4.891 4.453 4.118 4.698 3.575 + Agent (IPR) 7.606 5.030 4.433 4.030 4.496 3.767

GPT-Image-1 7.804 6.585 5.086 5.147 5.757 5.019 + Agent (IPR) 7.820 6.808 5.258 5.419 5.775 5.284

Qwen-Image-Edit-2509 7.499 3.699 2.256 2.031 2.351 2.033 DreamOmni2 6.520 4.069 2.804 3.037 2.867 2.594 OmniGen2 5.919 3.442 3.256 3.598 3.369 3.022

###### Instruction Alignment

###### Reference Consistency

###### Background Subject Match

###### Physical Realism

###### Visual Quality

Total

- 2

- 3

- 4

- 5

- 6

- 7

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- 2
- 3
- 4
- 5
- 6
- 7
- 8

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 4

- 5

- 6

- 7

- 1
- 2
- 3
- 4
- 5

- 3

- 4

- 5

- 6

- 7

Score

3 4 5 6 7 8

3 4 5 6 7 8

3 4 5 6 7 8

3 4 5 6 7 8

3 4 5 6 7 8

3 4 5 6 7 8

# References

# References

# References

# References

# References

# References

GPT-Image-1 Nano Banana Qwen-Image-Edit-2509 DreamOmni2 OmniGen2

- Figure 5. Changes in scores for each evaluation criterion when varying the number of reference images. Both open-source and closedsource models exhibit a general trend of decreasing scores across all metrics as the number of references increases.

erence images, we evaluate it only under the 5-reference setting. As described in Section 3.4, Gemini 2.5 [19] and GPT-5 [57] are used for evaluation. In the main paper, we report the average scores from Gemini 2.5 and GPT-5. See Appendix A for the results of Qwen3-VL as judge, and Appendix F for the detailed results and discussion.

##### 4.1. Per-Task Evaluation

We compute the MultiBanana scores for each task type defined in Section 3.1, and present the results in Table 3. For the two-reference tasks, we report the average score across all 11 task variants. For the multi-reference tasks, we compute the average score for each task type (i.e., X Objects, X–1 Objects + Local, X–1 Objects + Global, X–1 Objects + Background) across X = {3,4,5,6,7,8}.

Both open-source and closed-source models exhibit lower performance on background-replacement tasks, regardless of the number of reference images. The results also reveal clear performance gaps between closed-source and open-source models, even in relatively simple editing scenarios, such as single- and two-reference tasks.

##### 4.2. Effect of the Number of References

We investigate how the MultiBanana score changes when varying the number of reference images X = {3,4,5,6,7,8} for each multi-reference generation task (Figure 5). Both open-source and closed-source models exhibit a general trend of decreasing Total Score with increasing number of references. However, the nature of this

degradation differs between the two model families.

Closed-source models demonstrate strong adherence to reference attributes and prompt instructions. As a result, even when the number of references grows, the decline in Instruction Alignment and Reference Consistency remains relatively mild. Nevertheless, because these models attempt to strictly satisfy all reference constraints, increasing the number of subjects often degrades overall visual quality, including overcrowded scenes and compositional inconsistencies (see Figure 1 and Figure 6; left).

In contrast, open-source models exhibit weaker adherence to prompts and references; even under the 3-reference setting, both Instruction Alignment and Reference Consistency remain low, and these scores deteriorate further as the number of references increases. Qualitative inspection also reveals that open models frequently ignore multiple subjects under high-reference conditions. However, their overall quality (background consistency, physical realism, visual quality) remains relatively stable, and they tend not to suffer from the severe compositional collapse observed in closed models (see Figure 6; right).

In summary, closed-source models succeed in incorporating all required subjects but often generate globally inconsistent or distorted scenes due to overfitting to finegrained reference details. Open-source models, by contrast, tend to omit multiple reference subjects in complex scenarios but generate visually clean images with fewer structural failures. These observations highlight a clear trade-off between strict reference fidelity and holistic image coherence.

The lighthouse, house, and boat from Image 1 are on the left, the dog from Image 2 is in the

``The man in Image 1 is in the foreground, the rabbit in Image 2 is in the middle ground, the woman in Image 3 is in the background, and the vulture in Image 4 is in the sky. They are all in a surreal landscape with ruins and the aurora

``The woman from image 1 and the blue bird from image 2 are placed on the hood of the vintage car from image 4, which is set in the Tuscan courtyard from image 3. The style of

center, and the tent with the person from Image 3 is on the right. The woman from Image 4 is standing in front of the tent on the right. They are all gathered around a campfire on a

borealis.’’

the image is the same as in image 4 .’’

[Figure 136]

[Figure 137]

beach.

[Figure 138]

|[Figure 139]|[Figure 140]<br><br>|
|---|---|
|[Figure 141]<br><br>Reference 3|[Figure 142]<br><br>Reference 4|

|[Figure 143]<br><br>Qwen-Image-Edit-2509|
|---|

Reference 1 Reference 2

|[Figure 144]|[Figure 145]<br><br>|
|---|---|
|[Figure 146]<br><br>Reference 3|[Figure 147]<br><br>Reference 4|

|[Figure 148]<br><br>Nano Banana<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|[Figure 149]<br><br>|GPT|
|---|
<br><br>|
|---|

|[Figure 150]|[Figure 151]<br><br>|
|---|---|
|[Figure 152]<br><br>Reference 3|[Figure 153]<br><br>Reference 4|

Reference 2 Reference 1 Reference 2

Reference 1

Reference Ignoring

Domain mismatch Scale mismatch

The artist cat from image 1 should be placed in front of the stone fountain from image 2, with the cat's easel and canvas oriented towards the fountain. The cat should wear a gray

The chameleon in Image 1 is on the left, the punk dinosaur in Image 2 is in the center, and

``The couch from image 1, the woman from image 2, the woman from image 3, the woman from image 4, and the man

the white bird in Image 3 is on the right. They are arranged in a surreal, stylized scene with the owl in Image 4 standing on top of the vending machine in Image 1.

wool coat similar to the one worn by the man in image 3. The text from image 4 should be

from image 5 are arranged in front of a mountain. The background of the image is the same as in image 6.’’

superimposed, aligning its language with the text on the stone fountain in image 2.

[Figure 154]

[Figure 155]

[Figure 156]

|[Figure 157]<br><br>Nano Banana<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|[Figure 158]|[Figure 159]<br><br>|
|---|---|
|[Figure 160]<br><br>Reference 3|[Figure 161]<br><br>Reference 4|

|[Figure 162]|[Figure 163]<br><br>|
|---|---|
|[Figure 164]<br><br>Reference 3|[Figure 165]<br><br>Reference 4|

|[Figure 166]<br><br>DreamOmni2|
|---|

|[Figure 167]|[Figure 168]<br><br>|[Figure 169]<br><br>|
|---|---|---|
|[Figure 170]<br><br>Reference 4|[Figure 171]<br><br>Reference 5|[Figure 172]<br><br>Reference 6|

|[Figure 173]<br><br>GPT|
|---|

Reference 1 Reference 2 Reference 3

Reference 1 Reference 2 Reference 1 Reference 2

Wrong text

Duplicating birds Reference Ignoring

- Figure 6. (Left) Failure cases of closed-source models under challenging conditions such as cross-domain references, scale mismatches, rare concepts, and multilingual text. (Right) Failure cases of open-source models under many-reference conditions. They tend to ignore some reference subjects entirely.

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Score

Cross domain

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Different scale and view

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Rare concept

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Multilingual

Include

Others

Nano Banana GPT-Image-1 Qwen-Image-Edit-2509 DreamOmni2 OmniGen2

- Figure 7. Results for difficult reference combinations. For crossdomain and different scale and view tasks, every model shows lower scores than tasks without such conditions.

Table 4. Correlation between human and VLM judges.

###### Judge Pearson r Spearman r Cohen’s κ

GPT-5 0.6936 0.7134 0.6135 Gemini 2.5 0.5726 0.6113 0.3908 Qwen3-VL 0.5457 0.5642 0.5235

##### 4.5. Agentic Framework

To address the performance limitations on our challenging benchmark, we further construct three agentic framework baselines: Iterative Prompt Refinement (IPR), ContextAware Feedback Generation (CAFG), and Selective Reference Adaptation (SRA). We then conduct experiments on our MultiBanana benchmark. We show the results of IPR in Table 3, where the generator creates images, and the planner rewrites the prompt over multiple steps based on these images. While GPT’s performance improved with the agent framework, Nano Banana showed no improvement and, in some cases, performance degraded due to information loss from the initial instruction. For details on the problem setting, ablations, and results, see Appendix F.7.

##### 4.3. Analysis of Difficult Reference Combinations

To analyze the challenging reference combinations described in Section 3.1, we compare MultiBanana scores between tasks containing these difficult combinations and those that do not, as shown in Figure 7. For cross-domain, different scale and view, rare-concept, and multilingual tasks, most models achieve lower scores than on tasks without such conditions. These results indicate that the difficult reference concepts focused on by our benchmark remain challenging, even for state-of-the-art closed models. This demonstrates that our benchmark effectively uncovers the current limitations of multi-reference image generation.

#### 5. Conclusion

##### 4.4. Reliability of VLM Judge

We constructed a comprehensive benchmark for multireference image generation, focusing on challenges specific to this setting, such as difficult reference combinations. Using it, we evaluated state-of-the-art open and closed models and found two notable failure modes as the number of references increased: models either ignored editing instructions or collapsed when trying to follow them. Across different types of reference difficulty, tasks involving cross-domain inputs, large-scale or viewpoint gaps, rare concepts, and multilingual inputs are consistently challenging.

To assess the reliability of VLM-based evaluation, we measured the correlation between VLM and human ratings using three human evaluators on 32 samples from our benchmark (Table 4). Both GPT-5 and Gemini 2.5 show strong correlation with human judgments. This validates the use of VLM judges as a time- and cost-efficient proxy for human evaluation, as described in Section 3.4. We also confirmed that Qwen3-VL-8B-Instruct [2] is a reliable open-source alternative, thereby broadening accessibility for researchers without API access. See Appendix E for further discussion.

#### Acknowledgements

We thank Minsu Kim and Heiga Zen for their support with this work and for their review of the initial version of the paper. We also appreciate the funding support from Google Japan. MS was supported by JSPS KAKENHI Grant Number JP23H04974.

#### References

- [1] Rameen Abdal, Or Patashnik, Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, Sergey Tulyakov, Daniel Cohen-Or, and Kfir Aberman. Dynamic concepts personalization from single videos, 2025. 10
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. 8, 1, 6
- [3] Samyadeep Basu, Mehrdad Saberi, Shweta Bhardwaj, Atoosa Malemir Chegini, Daniela Massiceti, Maziar Sanjabi, Shell Xu Hu, and Soheil Feizi. Editval: Benchmarking diffusion based text-guided image editing methods, 2023. 2, 3
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 2
- [5] ByteDance. Doubao. https://www.doubao.com/,

2025. 3, 5

- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22560–22570,

2023. 10

- [7] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, Tiankai Hang, Duojun Huang, Jie Jiang, Zhengkai Jiang, Weijie Kong, Changlin Li, Donghao Li, Junzhe Li, Xin Li, Yang Li, Zhenxi Li, Zhimin Li, Jiaxin Lin, Linus, Lu-Hao Liu, Shu Liu, Songtao Liu, Yu Liu, Yuhong Liu, Yanxin Long, Fanbin Lu, Qinglin Lu, Yuyan Peng, Yuanbo Peng, Xiang-Yu Shen, Yi-Ping Shi, Jiale Tao, Yang-Dan Tao, Qianhui Tian, Pengfei Wan, Chunyu Wang, Kai Wang, Lei Wang, Linqing Wang, Lucas Wang, Qixun Wang, Weiyang Wang, Hao Wen, Bing Wu, Jianbing Wu, Yue Wu, Senhao Xie, Fangzhou Yang, Miles Yang, Xiaofeng Yang, Xuan

- Yang, Zhantao Yang, Jingmiao Yu, Zhengang Yuan, Chao Zhang, Jianwei Zhang, Pei pei Zhang, Shiyuan Zhang, Tao Zhang, Weigang Zhang, Yepeng Zhang, Yingfang Zhang, Zihao Zhang, Zijian Zhang, Penghao Zhao, Zhiyuan Zhao, Xuefei Zhe, Jian-Xiang Zhu, and Zhao Zhong. Hunyuanimage 3.0 technical report. ArXiv, abs/2509.23951, 2025. 2
- [8] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. arXiv preprint arXiv:2104.14294, 2021. 3
- [9] Ruoxi Chen, Dongping Chen, Siyuan Wu, Sinan Wang, Shiyun Lang, Peter Sushko, Gaoyang Jiang, Yao Wan, and Ranjay Krishna. Multiref: Controllable image generation with multiple visual references. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13325–13331, 2025. 10
- [10] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 10
- [11] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. arXiv preprint arXiv:2403.05139, 2024. 2, 10
- [12] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models, 2024. 10
- [13] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners, 2025. 2
- [14] Antoine Dussolle, Andrea Carde˜na D´ıaz, Shota Sato, and Peter Devine. M-IFEval: Multilingual instruction-following evaluation. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 6176–6191, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. 10
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 2, 10
- [16] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and ZhengJun Zha. Vivid: Video virtual try-on using diffusion models,

2024. 2, 10

- [17] Hiroki Furuta, Heiga Zen, Dale Schuurmans, Aleksandra Faust, Yutaka Matsuo, Percy Liang, and Sherry Yang. Improving dynamic object interactions in text-to-video generation with ai feedback. arXiv preprint arXiv:2412.02617,

2024. 6

- [18] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. 2
- [19] Gemini Team. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 3, 6, 7, 1, 5
- [20] Sara Ghazanfari, Wei-An Lin, Haitong Tian, and Ersin Yumer. Spotedit: Evaluating visually-guided image editing methods. arXiv preprint arXiv:2508.18159, 2025. 10
- [21] Google DeepMind. Veo 2, 2024. 10
- [22] Google DeepMind. Nano banana: Gemini 2.5 flash image model. https://developers.googleblog.com/ en/introducing-gemini-2-5-flash-image/,

2025. Accessed: 2025-10-31. 2, 3, 4, 6, 5

- [23] Keno Harada, Yudai Yamazaki, Masachika Taniguchi, Edison Marrese-Taylor, Takeshi Kojima, Yusuke Iwasawa, and Yutaka Matsuo. When instructions multiply: Measuring and estimating LLM capabilities of multiple instructions following. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 16506–16526, Suzhou, China,

2025. Association for Computational Linguistics. 3, 10

- [24] Yun He, Di Jin, Chaoqi Wang, Chloe Bi, Karishma Mandyam, Hejia Zhang, Chen Zhu, Ning Li, Tengyu Xu, Hongjiang Lv, et al. Multi-if: Benchmarking llms on multiturn and multilingual instructions following. arXiv preprint arXiv:2410.15553, 2024. 3, 10
- [25] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023. 10
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 6840–6851, 2020. 2
- [27] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022. 10
- [28] Daichi Horita, Naoto Inoue, Kotaro Kikuchi, Kota Yamaguchi, and Kiyoharu Aizawa. Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation. In CVPR, 2024. 2, 10
- [29] Chi-Pin Huang, Yen-Siang Wu, Hung-Kai Chung, Kai-Po Chang, Fu-En Yang, and Yu-Chiang Frank Wang. Videomage: Multi-subject and motion customization of text-tovideo diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17603– 17612, 2025. 10
- [30] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, and Ying Shan. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8362–8371, 2024. 2
- [31] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and

manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12469–12478, 2024. 10

[32] Google Imagen-Team, Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Lluis Castrejon, Kelvin Chan, Yichang Chen, Sander Dieleman, Yuqing Du, Zach Eaton-Rosen, Hongliang Fei, Nando de Freitas, Yilin Gao, Evgeny Gladchenko, Sergio G´omez Colmenarejo, Mandy Guo, Alex Haig, Will Hawkins, Hexiang Hu, Huilian Huang, Tobenna Peter Igwe, Christos Kaplanis, Siavash Khodadadeh, Yelin Kim, Ksenia Konyushkova, Karol Langner, Eric Lau, Rory Lawton, Shixin Luo, Soˇna Mokr´a, Henna Nandwani, Yasumasa Onoe, A¨aron van den Oord, Zarana Parekh, Jordi Pont-Tuset, Hang Qi, Rui Qian, Deepak Ramachandran, Poorva Rane, Abdullah Rashwan, Ali Razavi, Robert Riachi, Hansa Srinivasan, Srivatsan Srinivasan, Robin Strudel, Benigno Uria, Oliver Wang, Su Wang, Austin Waters, Chris Wolff, Auriel Wright, Zhisheng Xiao, Hao Xiong, Keyang Xu, Marc van Zee, Junlin Zhang, Katie Zhang, Wenlei Zhou, Konrad Zolna, Ola Aboubakar, Canfer Akbulut, Oscar Akerlund, Isabela Albuquerque, Nina Anderson, Marco Andreetto, Lora Aroyo, Ben Bariach, David Barker, Sherry Ben, Dana Berman, Courtney Biles, Irina Blok, Pankil Botadra, Jenny Brennan, Karla Brown, John Buckley, Rudy Bunel, Elie Bursztein, Christina Butterfield, Ben Caine, Viral Carpenter, Norman Casagrande, MingWei Chang, Solomon Chang, Shamik Chaudhuri, Tony Chen, John Choi, Dmitry Churbanau, Nathan Clement, Matan Cohen, Forrester Cole, Mikhail Dektiarev, Vincent Du, Praneet Dutta, Tom Eccles, Ndidi Elue, Ashley Feden, Shlomi Fruchter, Frankie Garcia, Roopal Garg, Weina Ge, Ahmed Ghazy, Bryant Gipson, Andrew Goodman, Dawid G´orny, Sven Gowal, Khyatti Gupta, Yoni Halpern, Yena Han, Susan Hao, Jamie Hayes, Jonathan Heek, Amir Hertz, Ed Hirst, Emiel Hoogeboom, Tingbo Hou, Heidi Howard, Mohamed Ibrahim, Dirichi Ike-Njoku, Joana Iljazi, Vlad Ionescu, William Isaac, Reena Jana, Gemma Jennings, Donovon Jenson, Xuhui Jia, Kerry Jones, Xiaoen Ju, Ivana Kajic, Christos Kaplanis, Burcu Karagol Ayan, Jacob Kelly, Suraj Kothawade, Christina Kouridi, Ira Ktena, Jolanda Kumakaw, Dana Kurniawan, Dmitry Lagun, Lily Lavitas, Jason Lee, Tao Li, Marco Liang, Maggie Li-Calis, Yuchi Liu, Javier Lopez Alberca, Matthieu Kim Lorrain, Peggy Lu, Kristian Lum, Yukun Ma, Chase Malik, John Mellor, Thomas Mensink, Inbar Mosseri, Tom Murray, Aida Nematzadeh, Paul Nicholas, Signe Nørly, Jo˜ao Gabriel Oliveira, Guillermo Ortiz-Jimenez, Michela Paganini, Tom Le Paine, Roni Paiss, Alicia Parrish, Anne Peckham, Vikas Peswani, Igor Petrovski, Tobias Pfaff, Alex Pirozhenko, Ryan Poplin, Utsav Prabhu, Yuan Qi, Matthew Rahtz, Cyrus Rashtchian, Charvi Rastogi, Amit Raul, Ali Razavi, Sylvestre-Alvise Rebuffi, Susanna Ricco, Felix Riedel, Dirk Robinson, Pankaj Rohatgi, Bill Rosgen, Sarah Rumbley, Moonkyung Ryu, Anthony Salgado, Tim Salimans, Sahil Singla, Florian Schroff, Candice Schumann, Tanmay Shah, Eleni Shaw, Gregory Shaw, Brendan Shillingford, Kaushik Shivakumar, Dennis Shtatnov, Zach Singer, Evgeny Sluzhaev, Valerii Sokolov, Thibault Sottiaux, Flo-

rian Stimberg, Brad Stone, David Stutz, Yu-Chuan Su, Eric Tabellion, Shuai Tang, David Tao, Kurt Thomas, Gregory Thornton, Andeep Toor, Cristian Udrescu, Aayush Upadhyay, Cristina Vasconcelos, Alex Vasiloff, Andrey Voynov, Amanda Walker, Luyu Wang, Miaosen Wang, Simon Wang, Stanley Wang, Qifei Wang, Yuxiao Wang, Agoston´ Weisz, Olivia Wiles, Chenxia Wu, Xingyu Federico Xu, Andrew Xue, Jianbo Yang, Luo Yu, Mete Yurtoglu, Ali Zand, Han Zhang, Jiageng Zhang, Catherine Zhao, Adilet Zhaxybay, Miao Zhou, Shengqi Zhu, Zhenkai Zhu, Dawn Bloxwich, Mahyar Bordbar, Luis C. Cobo, Eli Collins, Shengyang Dai, Tulsee Doshi, Anca Dragan, Douglas Eck, Demis Hassabis, Sissie Hsiao, Tom Hume, Koray Kavukcuoglu, Helen King, Jack Krawczyk, Yeqing Li, Kathy Meier-Hellstern, Andras Orban, Yury Pinsky, Amar Subramanya, Oriol Vinyals, Ting Yu, and Yori Zwols. Imagen 3, 2024. 10

- [33] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. LayoutDM: Discrete Diffusion Model for Controllable Layout Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10167–10176, 2023. 2, 10
- [34] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. InfiniteYou: Flexible photo recrafting while preserving your identity. In ICCV, 2025. 2
- [35] Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. FollowBench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, Bangkok, Thailand, 2024. Association for Computational Linguistics. 3, 10
- [36] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8176–8185, 2024. 10
- [37] Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, Masayoshi Tomizuka, and Kurt Keutzer. Streamdiffusion: A pipeline-level solution for realtime interactive generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12371–12380, 2025. 10
- [38] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation, 2023. 6
- [39] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversionfree text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19721–19730, 2025. 10
- [40] Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs get lost in multi-turn conversation. In The Fourteenth International Conference on Learning Representations, 2026. 3, 10
- [41] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 10

- [42] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 2, 5, 10

- [43] LAION-AI. aesthetic-predictor, 2022. 4
- [44] DONGXU LI, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-toimage generation and editing. In Advances in Neural Information Processing Systems, pages 30146–30166. Curran Associates, Inc., 2023. 2
- [45] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. arXiv:2301.07093, 2023. 10
- [46] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12: 157–173, 2024. 10
- [47] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval. arXiv preprint arXiv:2207.07285, 2022. 2
- [48] Yiwei Ma, Jiayi Ji, Ke Ye, Weihuang Lin, Yonghan Zheng, Qiang Zhou, Xiaoshuai Sun, Rongrong Ji, et al. I2ebench: A comprehensive benchmark for instruction-based image editing. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2, 3
- [49] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2063–2072, 2025. 10
- [50] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023. 2
- [51] Ryugo Morita, Stanislav Frolov, Brian Bernhard Moser, Takahiro Shirakawa, Ko Watanabe, Andreas Dengel, and Jinjia Zhou. Tkg-dm: Training-free chroma key content generation diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13031–13040, 2025. 2, 10
- [52] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 10
- [53] Sanghyeon Na, Yonggyu Kim, and Hyunjoon Lee. Boost your own human image generation model via direct preference optimization with ai feedback. arXiv preprint arXiv:2405.20216, 2024. 6

- [54] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3, 5
- [55] OpenAI. Sora, 2024. 10
- [56] OpenAI. Gpt-4o image generation. https : / / openai.com/index/introducing-4o-imagegeneration/, 2025. Accessed: 2025-10-31. 2, 3, 4, 6, 5
- [57] OpenAI. Gpt-5. https://openai.com/ja-JP/ index/introducing-gpt-5/, 2025. Accessed: 202511-14. 6, 7, 1
- [58] Y. Oshima, M. Suzuki, Y. Matsuo, and H. Furuta. Inferencetime text-to-video alignment with diffusion latent beam search, 2025. arXiv preprint arXiv:2501.19252. 6
- [59] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 conference proceedings, pages 1–11, 2023. 10
- [60] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 2, 10
- [61] Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, and TatSeng Chua. Layoutllm-t2i: Eliciting layout guidance from llm for text-to-image generation, 2023. 10
- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021. 3, 4
- [63] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 10
- [64] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In The Thirteenth International Conference on Learning Representations, 2025. 4, 2
- [65] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. arXiv preprint arXiv:2112.10752, 2022. 2, 10
- [66] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arxiv:2208.12242, 2022. 2, 3, 10
- [67] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image

- diffusion models with deep language understanding, 2022. 10
- [68] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 4, 2
- [69] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8871–8879, 2024. 2, 3
- [70] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, pages 2256–2265, 2015. 2
- [71] Peter Sushko, Ayana Bharadwaj, Zhi Yang Lim, Vasily Ilin, Ben Caffee, Dongping Chen, Mohammadreza Salehi, Cheng-Yu Hsieh, and Ranjay Krishna. Realedit: Reddit edits as a large-scale empirical dataset for image transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13403– 13413, 2025. 10
- [72] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14940–14950, 2025. 2
- [73] Yunjie Tian, Qixiang Ye, and David Doermann. Yolov12: Attention-centric real-time object detectors. arXiv preprint arXiv:2502.12524, 2025. 4, 2
- [74] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1921–1930, 2023. 10
- [75] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22532–22541, 2023. 10
- [76] Meiyun Wang, Takeshi Kojima, Yusuke Iwasawa, and Yutaka Matsuo. Lost in the distance: Large language models struggle to capture long-distance relational knowledge. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 4536–4544, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. 10
- [77] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi PontTuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J. Fleet, Radu Soricut, Jason Baldridge, Mohammad Norouzi, Peter Anderson, and William Chan. Imagen editor and editbench: Advancing and evaluating textguided image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18359–18369, 2023. 2, 3

- [78] X Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 2, 10
- [79] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 2, 6, 5

- [80] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 2, 3, 6, 5
- [81] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025. 2
- [82] Xun Wu, Shaohan Huang, Guolong Wang, Jing Xiong, and Furu Wei. Boosting text-to-video generative model with MLLMs feedback. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 6
- [83] Bin Xia, Bohao Peng, Yuechen Zhang, Junjia Huang, Jiyang Liu, Jingyao Li, Haoru Tan, Sitong Wu, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamomni2: Multimodal instruction-based editing and generation, 2025. 2, 3, 4, 6, 5
- [84] Bin Xia, Yuechen Zhang, Jingyao Li, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamomni: Unified image generation and editing, 2025.
- [85] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 2
- [86] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv preprint arXiv:2403.01779, 2024. 10
- [87] Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, and Jian Zhang. Fakeshield: Explainable image forgery detection and localization via multi-modal large language models. In International Conference on Learning Representations, 2025. 3
- [88] Dingkun Yan, Xinrui Wang, Zhuoru Li, Suguru Saito, Yusuke Iwasawa, Yutaka Matsuo, and Jiaxian Guo. Image referenced sketch colorization based on animation creation workflow. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23391–23400, 2025. 10
- [89] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael

- Zeng, and Lijuan Wang. Reco: Region-controlled text-toimage generation. In CVPR, 2023. 10
- [90] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arxiv:2308.06721,

2023. 2, 10

- [91] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 2, 3, 4, 6, 5, 9
- [92] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26125–26135, 2025. 3, 5
- [93] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. In Advances in Neural Information Processing Systems, 2023. 2, 3, 5
- [94] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 10
- [95] Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, 2023. 10
- [96] Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6818– 6828, 2024. 10
- [97] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023. 3, 10
- [98] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4606– 4615, 2023. 2, 10

## MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation

### Supplementary Material

#### A. Results of Qwen3-VL Judge

Table 5 and Table 6 present per-task total scores evaluated using Qwen3-VL as the sole judge, calculated in the same way as described in Section 3.4. The results are broadly consistent with those obtained using closedmodel judges. The overall ranking of image generation models is preserved. We also observe the same key trends: performance degrades as the number of reference images increases, and background replacement tasks tend to yield lower scores across both open-source and closedsource models. These observations reinforce the validity of Qwen3-VL as a reliable, reproducible alternative. We expect this fixed, open-weight judge to serve as a dependable baseline for future evaluations on our benchmark.

In Section 4, we report the average scores from Gemini 2.5 and GPT-5 as judges. As demonstrated in Section 4.4, Qwen3-VL-8B-Instruct [2] achieves strong correlation with human judgments comparable to GPT-5 [57] and Gemini 2.5 [19], confirming its reliability as a judge model (see

- Table 4). Although we used stable API versions for the closed-source judges, to ensure full reproducibility, we additionally employ Qwen3-VL-8B-Instruct, an open-weight, fixed-version model, as the judge. This also broadens access to our benchmark for researchers who do not have access to closed-source VLMs via APIs.
- Table 5. Per-task total scores for each single- and two-reference task, evaluated by Qwen3-VL. “Back.” denotes the Background task. Both open-source and closed-source models exhibit lower performance on background replacement tasks. Note that for Nano Banana and GPT-Image-1, we use the versions available as of January 2026.

Model Single Add Back. Color Hair Makeup Material Pose Replace Style Text Tone DreamOmni2 8.47 6.51 4.59 8.60 7.40 7.89 6.76 6.25 6.86 5.10 5.48 5.14 OmniGen2 9.04 5.34 6.04 5.07 6.70 6.68 6.12 5.91 6.34 3.24 4.29 4.00 Qwen-Image-Edit-2509 9.16 7.01 4.34 6.46 8.00 6.68 5.86 5.87 7.57 3.41 4.95 6.08 Nano Banana 9.45 8.13 7.97 9.67 9.16 9.25 8.95 8.76 8.30 9.03 8.81 8.52 GPT-image-1 9.51 8.09 8.08 9.70 9.04 9.24 9.17 8.80 8.30 9.10 8.52 8.78

- Table 6. Per-task total scores for each multi-reference task, evaluated by Qwen3-VL. The local, global, back, and object columns under X correspond to X–1 Objects + Local, X–1 Objects + Global, X–1 Objects + Background, and X Object, respectively. Both open-source and closed-source models exhibit a general trend of decreasing scores across all tasks as the number of references increases. They also tend to perform worse on background replacement tasks, especially as the number of reference images increases. Note that for Nano Banana and GPT-Image-1, we use the versions available as of January 2026.

Model 3-references 4-references 5-references

local global back object local global back object local global back object

DreamOmni2 4.39 4.27 4.33 3.68 3.80 2.95 2.77 2.90 3.23 2.12 2.86 2.26 OmniGen2 4.71 4.86 4.39 4.13 4.37 3.86 3.43 2.95 3.99 2.75 2.14 2.26 Qwen-Image-Edit-2509 5.22 5.81 5.81 5.63 3.24 4.51 4.42 3.94 1.05 1.48 1.28 1.10 Nano Banana 7.63 7.21 6.96 6.91 7.96 6.77 6.47 6.78 7.96 6.54 6.03 5.87 GPT-image-1 7.83 7.26 6.99 6.72 7.73 6.74 5.94 6.83 7.92 6.54 6.16 5.83

###### Model 6-references 7-references 8-references

local global back object local global back object local global back object

DreamOmni2 3.34 2.26 1.72 1.93 2.61 2.45 2.10 1.81 2.44 1.65 1.83 1.59 OmniGen2 - - - - - - - - - - - Qwen-Image-Edit-2509 1.00 1.49 1.00 1.00 1.00 1.46 1.00 1.13 1.00 1.00 1.00 1.00 Nano Banana 6.84 6.08 5.34 5.64 6.37 5.78 5.13 5.44 6.46 6.24 5.31 5.28 GPT-image-1 7.11 5.86 5.51 5.92 6.81 5.78 5.37 5.74 7.07 5.86 5.14 5.55

#### B. Details on Benchmark Construction

##### B.1. Image Collection

We constructed the reference image collection for our benchmark using both real images sourced from LAION5B [68] and synthetic images generated by Nano Banana [22] and GPT-Image-1 [56]. By combining real and synthetic data, we mitigated the categorical biases inherent in the real-image distribution while expanding the coverage and diversity of the reference set. To generate synthetic images, we designed distinct axes of variation for four major categories: humans, animals, objects, and text-containing images.

Human Category. We systematically combined attributes such as makeup, hairstyle, hair color, pose, strong facial expressions, lighting, and overall tonal style to create a wide range of appearance conditions. Examples include ethnic or fashion-model-style makeup, vivid or unconventional hair colors, model-like or manga-like poses, strong emotional expressions (e.g., anger, joy, surprise), dramatic lighting (e.g., window light through blinds or neon illumination), and global color palettes (e.g., sepia or bluish). These conditions were intentionally selected to supplement the variations underrepresented in real-world data.

Animal Category. We first specified major species groups (dogs, cats, birds, reptiles, wild animals, and fantasy animals). We then added attributes to enhance the diversity of animal imagery. These included human-like behaviors (e.g., a bear reading a book, a fox playing in a rock band), rare colors (e.g., a blue horse or a pink panda), distinctive lighting conditions, and varied global tones such as sepia. This incorporated rare or extreme examples that seldom appear in real datasets.

Object Category. We expanded diversity along two axes: intrinsic object attributes and photographic conditions. The former included distinctive textures (e.g., denim, leather, knit), characteristic prints (e.g., floral, striped, checkered), and variations in material or design (e.g., matte, glossy, transparent). The latter included variations in lighting and global tonal adjustments (e.g., sepia, cool color tones), thereby ensuring a broad range of appearance conditions for objects.

Images Containing Text. Images containing text, which were largely absent from real data, were generated exclusively using GPT-Image-1, as Nano Banana exhibited limited ability to render non-English text. We defined three levels of text length (a single word, a short phrase of up to five words, and a longer sentence). We combined each with two scene types: text rendered directly within a landscape (e.g., ocean, forest) and text appearing naturally in the environment (e.g., on signs in real scenes). These six prompt patterns were generated in English and subsequently translated into Japanese and Chinese to enrich the multilingual

text in images.

Scene Diversity. Finally, all synthetic prompts were designed to incorporate scene diversity, including European cityscapes, university campuses, tropical beaches, event halls, and other varied environments. The generated subjects were well-framed, resulting in images suitable for use as high-quality reference material.

##### B.2. Image Filtering

First, to remove images unsuitable for editing, we filtered them based on the size of the foreground objects, as in [91]. YOLOv12 [73] was applied to all images to detect objects and then performed semantic segmentation using SAM [64], conditioned on the detected bounding boxes. Images were discarded if they met any of the following criteria: the bounding box area covered less than 2% of the entire image, the segmented region within the bounding box covered less than 30%, or the CLIP similarity [47] between the YOLO-predicted class name and the bounding box region was below 20. These cases were judged to contain objects that were not clearly visible. However, images where no objects were detected were retained, assuming they are useful for background or style-level editing tasks. After this filtering, 48% of the images were retained.

We also filtered out images that were inappropriate or unsuitable as references, such as unsafe content, charts, and screenshots of system messages, using both automated screening based on Gemini and human review. For inappropriate content, we specifically targeted categories including hate, harassment, violence, self-harm, sexual content, nudity, shocking content, illegal activity, and other distressing material, ultimately excluding approximately 3% of the collected images. We also manually removed nearly identical synthetic data. An example is shown in Figure 8.

##### B.3. Category Classification

To construct multi-reference image editing tasks, each reference image had to be accurately categorized. For example, in portrait transformation tasks that modify a person’s hairstyle or makeup, both reference images must contain a person. Similarly, in replacement tasks involving humans or objects, the corresponding entities must be present in the reference images, and in background replacement tasks, the background scenes must be identified in the reference images. To satisfy these task-dependent requirements, we performed hierarchical image classification on a large combined set of real and synthetic images.

For synthetic data, attributes can be specified directly in the prompt during generation, eliminating the need for additional classification. In contrast, real images exhibit substantial variability, requiring precise categorization to determine whether they can serve as valid reference images. To address this, we designed a structured annotation protocol

[Figure 174]

Figure 8. Examples of a duplicated synthetic image.

using Gemini that consists of three classification processes for the human, object, and background components of an image.

Human Image Classification Human classification begins with identifying how many people appear in the image. If there is precisely one person, the model additionally classifies attributes such as gender, makeup, hairstyle distinctiveness, pose clarity, and facial expression characteristics. The output follows a fixed template, enabling consistent extraction of appearance features relevant to multi-reference generation.

Object Image Classification Object image classification follows a similar structure. The model first determines whether the image contains no object, a single object, or multiple objects. When exactly one object is detected, it is further categorized according to its visual and material properties, distinguishing among animal-like forms, patterned or decorative textiles, engineered or manufactured materials, and objects lacking distinctive design features.

Background Image Classification Background classification is conducted from three viewpoints: realism, tonal uniformity, and lighting strength. Realism distinguishes photographic scenes from stylized or synthetic ones; tone indicates whether the overall color distribution is uniform or varied; and lighting strength identifies whether a clear directional key light is present.

##### B.4. Task Construction

The editing instructions were generated using Gemini. First, we manually selected suitable reference categories for each task, for instance, the person category for the hairstyle modification task. Next, we randomly sampled images from each category and presented them to Gemini to produce corresponding editing instructions. For example, in editing tasks that require object transformations, we first prompted the model to determine which reference image’s object should be modified and which reference image’s attributes (e.g., color) should be used as the target. Afterward, several instruction examples were provided to Gemini, and we asked Gemini to generate new instructions following the same syntax. Afterward, editing instructions that could cause visual breakdowns were removed after being scored with Gemini. Additionally, Gemini assessed the difficulty of editing instructions, and those judged easy, such as cases without domain differences, were removed. Finally, the editing instructions were manually verified not to cause image breakdowns or result in overly simple edits, and assigned the remaining samples to the appropriate difficulty categories (cross-domain, scale and viewpoint differences, rare concept, and multilingual text rendering).

1000

Tone 6.0%

Add 6.8% Background 6.4%

800

Text 6.1%

Style 7.3%

Color 6.1%

600

Count

Hair 10.3%

400

Replace 21.8%

200

Makeup 13.5%

Material 9.1%

Pose 6.6%

0

1 2 3 4 5 6 7 8

Number of References

Figure 9. (Left) Number of tasks by reference count. The two-reference tasks contain more samples than the other reference tasks because they include multiple task types. (Right) Breakdown of the two-reference tasks. It contains eleven tasks.

###### Remove the lemon slice inside the glass, and turn the slice on the rim into an orange slice.

###### Change the clothes of the person in the lower right corner to green

###### Turn the nearly spoiled banana into an unripe banana.

###### Darken the screen directly above the stage

[Figure 175]

|[Figure 176]<br><br>Reference|
|---|

|[Figure 177]<br><br>Nano Banana|
|---|

|[Figure 178]<br><br>Reference|
|---|

|[Figure 179]<br><br>Nano Banana|
|---|

|[Figure 180]<br><br>Reference|
|---|

|[Figure 181]<br><br>Nano Banana|
|---|

|[Figure 182]<br><br>Reference|
|---|

|[Figure 183]<br><br>Nano Banana<br><br>[Figure 184]|
|---|

[Figure 185]

[Figure 186]

- Figure 10. Example of “Hard” category of ImgEdit. These tasks are almost fully solvable by advanced models such as Nano Banana.

The banana from Image 1 and the telephone from Image 2 are placed side by side. The background is from Image 3.

|[Figure 187]<br><br>Reference 1|
|---|

|[Figure 188]<br><br>Reference 2|
|---|

|[Figure 189]<br><br>Reference 3|
|---|

|[Figure 190]<br><br>Nano Banana|
|---|

The woman from image 1 and the man from image 2 are standing in front of a mountain. The dog from image 3 is standing between them. The style of the image is the same as in image 4.

|[Figure 191]<br><br>Reference 1|
|---|

|[Figure 192]<br><br>Reference 2|
|---|

|[Figure 193]<br><br>Reference 3|
|---|

|[Figure 194]<br><br>Reference 4|
|---|

|[Figure 195]<br><br>Nano Banana|
|---|

[Figure 196]

[Figure 197]

- Figure 11. Example of DreamOmni2 benchmark. These tasks are almost fully solvable by advanced models such as Nano Banana.

#### C. Further Statistics for MultiBanana

The left of Figure 9 shows the number of tasks by each reference count. The two-reference tasks contain more samples than the other tasks because they include multiple task types. The right of Figure 9 shows the breakdown of the two-reference tasks. It contains eleven tasks considered in the prior work [83]: subject addition, subject replacement, background change, color modification, material modification, pose modification, hairstyle modification, makeup modification, tone transformation, style transfer, and text correction. Each task is guaranteed to include at least 6% of the editing samples, corresponding to roughly 60 samples, which exceeds the number of samples in Xia et al. [83].

#### D. Comparison with Prior Benchmarks

Table 1 shows that existing benchmarks do not provide systematic evaluation across diverse multi-reference conditions, support only a limited number of references, and fail to adequately account for heterogeneity among reference images. To illustrate that existing benchmarks are almost solved by state-of-the-art models such as Nano Banana [22] and GPT-Image-1 [56], we evaluate the image generation capabilities of these models using ImgEdit [91], the latest benchmark for image editing, and DreamOmni2 [83], the state-of-the-art benchmark for multi-reference image generation. The quantitative results show that recent closedsource, state-of-the-art models achieve consistently high scores, indicating that these benchmarks are nearing saturation and may soon be unable to meaningfully distin-

- Table 7. Comparison results of different models on ImgEdit-Bench [91]. “Overall” is computed by averaging the scores across all task types. GPT-4.1 [54] is used for evaluation. The results show that recent closed-source, state-of-the-art models achieve substantially higher scores, suggesting that the benchmark may be approaching its ceiling in distinguishing high-end models. Evaluation results excluding Nano Banana are taken from the official GitHub repository of Ye et al. [91].

Model Add Adjust Extract Replace Remove Background Style Hybrid Overall

MagicBrush [93] 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.83 AnyEdit [92] 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.45 OmniGen2 [80] 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 3.44 Kontext-dev [42] 3.83 3.65 2.27 4.45 3.17 3.98 4.55 3.35 3.71

Nano Banana [22] 3.63 4.66 3.73 4.69 4.71 4.60 4.47 4.10 4.37 GPT-Image-1 [56] 4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.20

- Table 8. Quantitative comparison of multimodal instruction-based editing and generation in DreamOmni2 Benchmark [83]. Both tasks are evaluated using Gemini [19] and Doubao [5]. This result shows that recent closed-source, state-of-the-art models achieve substantially higher scores, suggesting that the benchmark may be approaching its ceiling in distinguishing among high-end models. We refer to evaluation results from Xia et al. [83].

Editing Task Generation Task

Method

Concrete ↑ Abstract ↓ Concrete ↑ Abstract ↓ Gemini Doubao Gemini Doubao Gemini Doubao Gemini Doubao

Omnigen2 [80] 0.2195 0.2927 0.0427 0.0793 0.2083 0.2500 0.1000 0.0778 Kontext [42] 0.0488 0.1220 0.0183 0.0122 0.2500 0.3750 0.0556 0.1222 Qwen-Image-Edit-2509 [79] 0.2683 0.2927 0.0488 0.1159 0.1250 0.2917 0.1111 0.1556 DreamOmni2 [83] 0.5854 0.6585 0.5854 0.6280 0.5833 0.6667 0.5778 0.6333

Nano Banana [22] 0.6829 0.7073 0.6463 0.5488 0.5000 0.5417 0.5556 0.5488 GPT-Image-1 [56] 0.6829 0.7805 0.7195 0.7439 0.6250 0.6250 0.6889 0.6333

guish among high-performing models (Table 7 and Table 8). The evaluation results are taken from the official GitHub repository of ImgEdit1 and the results reported in the DreamOmni2 paper. Qualitative results suggest that, even in the additional “Hard” category of ImgEdit, most tasks are already solvable by advanced models such as Nano Banana (Figure 10). A similar tendency is observed for DreamOmni2 benchmark as well (Figure 11).

Taken together, these findings indicate that current benchmarks for image editing and multi-reference image generation are approaching their limits in evaluating cutting-edge models. There is a clear need for more challenging benchmarks that better capture the capabilities and failure modes of the latest generation of image models.

#### E. Reliability and Cost of VLM Judges

As described in Section 3.4, we employ VLM judges as a time- and cost-efficient proxy for human evaluation. In Section 4.4, we demonstrate that the VLM judge correlates

1https://github.com/PKU-YuanGroup/ImgEdit?tab= readme-ov-file

strongly with human ratings, supporting our usage of VLM. Here, we provide a more detailed analysis of their reliability and the associated cost considerations.

##### E.1. Cross-VLM Correlation

To demonstrate the consistency of our evaluation metrics, we examined correlations among different VLMs. We use the evaluated results for Nano Banana and GPT-Image-1 and examine the correlation between GPT’s and Gemini’s evaluation scores. To assess the reliability of the correlation coefficient, we partitioned the evaluation results into 10 subsets and computed the mean correlation across them. Table 9 shows that the evaluated scores exhibit positive correlations, which demonstrates that our evaluation criteria are well-defined and consistent.

##### E.2. Self-Consistency Analysis

We measure the standard error of the VLM judge scores across three different random seeds in Table 10. On the 1–10 rating scale, the observed standard errors are below 0.1, and we do not observe an increasing trend in standard error as the number of references increases, indicating

- Table 9. Correlation between GPT’s and Gemini’s evaluated scores. ± shows 95% confidence interval.

Evaluation criteria Correlation coefficients

Instruction Alignment 0.645 ±0.021 Reference Consistency 0.549 ±0.026

Background Subject Match 0.601 ±0.020 Physical Realism 0.620 ±0.022 Visual Quality 0.577 ±0.018

Overall 0.650 ±0.014

- Table 10. Standard error of total scores evaluated by the average of GPT and Gemini, for GPT-Image-1 and Nano Banana.

Model 3-ref 4-ref 5-ref 6-ref 7-ref 8-ref GPT-Image-1 .0234 .0138 .0139 .0153 .0343 .0244 Nano Banana .0260 .0423 .0033 .0089 .0062 .0254

that the VLM judge provides self-consistent evaluations for multi-reference image generation.

##### E.3. Sensitivity Analysis

To further ensure the reliability of our evaluation metrics, we examined whether they can detect performance gains from fine-tuning on external image-editing datasets, thereby demonstrating that they align well with those used in other image-editing benchmarks. We compare QwenImage-Edit with Qwen-Image-Edit-2509, a fine-tuned variant designed to improve reference consistency and support multi-reference inputs. As shown in Table 11, our evaluation metrics successfully capture the performance gains of Qwen-Image-Edit-2509 over Qwen-Image-Edit. We conducted this comparison using single-reference tasks because Qwen-Image-Edit supports only a single input image.

##### E.4. Cost of VLM Judge

Regarding evaluation costs, a single model evaluation costs approximately $44 with GPT-5 and $29 with Gemini 2.5 in January 2026. As an API-free alternative, we confirmed that Qwen3-VL [2] can be reliable enough (see Table 4).

#### F. Detailed Results F.1. Per-Task Evaluation

Figure 15, Figure 16, and Figure 17 show the scores of each model across five evaluation metrics for each task, evaluated by GPT, Gemini, and their average, respectively. Taking the weighted average across five metrics, Figure 18 shows the total score of each model for single and two-reference tasks, evaluated by GPT, Gemini, and their average, respectively. For multi-reference tasks, Table 12, Table 13, and Table 14

Table 11. Comparison of Qwen-Image-Edit and Qwen-ImageEdit-2509 on single-reference task.

###### Qwen-Image-Edit Qwen-Image-Edit-2509 single 6.332 7.499

show the total score, evaluated by GPT, Gemini, and their average, respectively.

Overall, we observed a substantial gap in Instruction Alignment and Reference Consistency between the closedsource models and the other open-source models. In particular, GPT-Image-1 achieves significantly superior performance in Instruction Alignment, especially in style and text modification tasks in the two-reference task. Meanwhile, in Reference Consistency, Nano Banana achieves the highest scores in the single-reference task. Nano Banana also performs on par with GPT-Image-1 in Reference Consistency across background modification, subject replacement, subject addition, and makeup tasks, indicating its strong ability to leverage reference images, particularly when the number of references is small. On the other hand, Nano Banana performs worse than the other models in background modification tasks in Background Subject Match, Physical Realism, and Visual Quality. This result suggests that Nano Banana struggles with background modification.

##### F.2. The Effect of the Number of References

According to the per-metric scores in Figure 17 and the total scores in Table 14, all models exhibit decreasing performance as the number of reference images increases. For Instruction Alignment and Reference Consistency, GPTImage-1 and Nano Banana achieve relatively high scores, whereas the open-source model, DreamOmni2, tends to approach the minimum score of 1 as the number of references increases. In contrast, for quality-related metrics, Background Subject Match, Physical Realism, and Visual Quality, the DreamOmni2 maintains high scores even when more reference images are provided. As discussed in Section 4, this is because closed-source models prioritize adherence to the references and editing instructions over visual quality, while open-source models show a tendency to preserve visual quality but often ignore the references.

In the finer-grained task-level evaluation, we found that, even as the number of reference images increased, the X–1 Object + Global task did not show severe decreases in Background Subject Match, Physical Realism, or Visual Quality scores. The task requires modifying the image to a specified style; therefore, even if the reference images come from different domains, which may degrade the visual quality, the final output often converges to a unified style.

In the results of Qwen-Image-Edit-2509, the model may be trained to handle up to three reference images; therefore,

- Table 12. Detailed per-task total scores for the multi-reference tasks, evaluated by GPT. The local, global, back, and object columns under X correspond to X–1 Objects + Local, X–1 Objects + Global, X–1 Objects + Background, and X Object, respectively.

Model 3-references 4-references 5-references

local global back object local global back object local global back object

DreamOmni2 3.94 4.19 3.92 3.91 3.34 3.24 2.87 3.20 3.30 3.06 2.73 3.22 OmniGen2 4.42 4.59 4.47 4.47 4.41 3.37 3.10 3.27 3.78 3.28 2.87 3.31 Qwen-Image-Edit-2509 4.42 4.65 4.75 4.77 3.08 3.35 2.78 3.54 1.93 2.40 1.98 2.10 Nano Banana 5.44 5.81 5.10 5.64 4.57 5.36 4.27 5.25 5.12 5.93 3.96 4.93 GPT-image-1 6.58 7.40 6.98 6.40 6.18 6.98 6.73 6.56 6.18 6.66 6.22 6.28

Model 6-references 7-references 8-references

local global back object local global back object local global back object

DreamOmni2 3.37 3.03 2.82 2.96 3.24 3.02 2.71 2.86 3.26 2.79 2.57 2.82 OmniGen2 - - - - - - - - - - - Qwen-Image-Edit-2509 1.85 2.18 1.68 1.79 1.51 1.89 1.57 1.88 1.68 2.29 1.62 1.83 Nano Banana 4.43 4.88 3.89 4.90 4.40 4.78 3.57 4.81 4.31 4.82 3.21 4.45 GPT-image-1 6.16 6.08 5.68 5.90 5.38 6.09 5.59 5.94 4.96 5.82 5.03 5.15

- Table 13. Detailed per-task total scores for the multi-reference tasks, evaluated by Gemini. The local, global, back, and object columns under X correspond to X-1 Objects + Local, X–1 Objects + Global, X–1 Objects + Background, and X Object, respectively.

###### Model 3-references 4-references 5-references

local global back object local global back object local global back object

DreamOmni2 3.00 2.49 2.45 2.74 2.84 2.73 2.38 2.44 2.63 2.50 2.36 2.66 OmniGen2 3.03 2.89 3.00 3.25 3.10 3.06 2.27 2.73 2.84 3.02 2.41 2.50 Qwen-Image-Edit-2509 3.27 3.16 3.29 3.97 2.29 3.00 2.15 2.79 1.14 1.69 1.21 1.20 Nano Banana 4.34 4.69 4.61 4.89 3.42 4.34 3.79 4.14 4.25 4.50 3.16 3.85 GPT-image-1 5.48 6.28 5.37 5.16 4.83 5.26 4.75 5.04 4.89 5.20 4.25 4.10

###### Model 6-references 7-references 8-references

local global back object local global back object local global back object

DreamOmni2 2.47 2.54 2.14 2.36 2.65 2.50 2.15 2.30 2.42 2.33 2.04 2.19 OmniGen2 - - - - - - - - - - - Qwen-Image-Edit-2509 1.00 1.26 1.00 1.03 1.01 1.06 1.33 1.17 1.18 1.31 1.01 1.00 Nano Banana 3.25 3.91 2.55 3.79 3.05 3.73 2.26 3.52 2.85 3.64 2.53 3.25 GPT-image-1 3.66 4.66 3.46 4.02 4.18 4.58 3.24 3.52 3.28 4.07 2.91 2.96

when provided with four or more references, the input becomes out-of-distribution. This causes the model to produce perceptually invalid outputs, and then the scores approach the minimum value of 1 (see Figure 29).

##### F.3. Difficult Reference Combination

Figure 19 shows the scores of each model in difficult reference combination tasks across evaluation metrics.

Cross-domain diversity In cross-domain cases, samples with reference images from different domains (darker color) received lower Reference Consistency scores because all models tend to distort individual reference attributes to achieve a consistent style.

Scale and viewpoint differences In the different scale and viewpoint cases, the Reference Consistency scores declined consistently across all models. This suggests that when models attempt to reproduce objects at different scales or from different viewpoints, they may fail to preserve fine details or adjust the pose to achieve a more natural appearance.

Rare concept references In the rare-concept case, the model may struggle to handle uncommon subjects relative to more familiar ones. Because the reference images often depict these subjects at a large scale, the model tends to reproduce this scale without appropriate adjustment. This suggests that, unlike common concepts, the model may find it more difficult to flexibly control attributes such as size

- Table 14. Detailed per-task total scores for each multi-reference task, evaluated by the average of GPT and Gemini. The local, global, back, and object columns under X correspond to X–1 Objects + Local, X–1 Objects + Global, X–1 Objects + Background, and X Object, respectively.

###### Model 3-references 4-references 5-references

local global back object local global back object local global back object

DreamOmni2 3.47 3.34 3.18 3.33 3.09 2.98 2.62 2.82 2.97 2.78 2.54 2.94 OmniGen2 3.73 3.74 3.74 3.86 3.76 3.22 2.69 3.00 3.31 3.15 2.64 2.91 Qwen-Image-Edit-2509 3.85 3.90 4.02 4.37 2.69 3.17 2.47 3.17 1.54 2.04 1.60 1.65 Nano Banana 4.89 5.25 4.85 5.27 4.00 4.85 4.03 4.70 4.69 5.22 3.56 4.39 GPT-image-1 6.03 6.84 6.18 5.78 5.50 6.12 5.74 5.80 5.54 5.93 5.24 5.19

###### Model 6-references 7-references 8-references

local global back object local global back object local global back object

DreamOmni2 2.92 2.78 2.48 2.66 2.94 2.76 2.43 2.58 2.84 2.56 2.31 2.50 OmniGen2 - - - - - - - - - - - Qwen-Image-Edit-2509 1.42 1.72 1.34 1.41 1.26 1.47 1.45 1.52 1.43 1.80 1.32 1.42 Nano Banana 3.84 4.39 3.22 4.35 3.72 4.25 2.92 4.16 3.58 4.23 2.87 3.85 GPT-image-1 4.91 5.37 4.57 4.96 4.78 5.33 4.42 4.73 4.12 4.94 3.97 4.05

when dealing with rare concepts. As a result, rare concept samples tend to receive lower Physical Realism scores.

Multilingual references In the multilingual text rendering case, we observed a consistent decline across all metrics except Reference Consistency. This may occur because models have limited text-rendering capability in languages other than English, leading to failures in cross-language conversion. As a result, instruction alignment and visual quality decrease. However, GPT-Image-1 exhibits a smaller decline than other models, owing to its comparatively stronger textrendering ability in non-English languages.

##### F.4. Risks of Model Bias in Synthetic References

Synthetic references generated by closed models (i.e., Nano Banana, GPT-Image-1) may introduce in-distribution advantages when those same models are also evaluated on the benchmark. To verify this, we divided the benchmark into three subsets based on the source of the reference images: those from GPT-Image-1, those from Nano Banana, and those from both. The mean scores across all subsets remain within their respective 99% confidence intervals, indicating no statistically significant bias (Figure 12).

##### F.5. Potential Conflict in Cross-Domain Task

One might expect that preserving subject details (reference consistency) and maintaining a coherent scene (background-subject match) are inherently at odds in crossdomain tasks—e.g., placing a photorealistic person into an anime background. However, our evaluation shows this is not always the case. As shown in Figure 13 (left), a generated image can satisfy both criteria, achieving a GPT-5 score of 7 for reference consistency and 9 for background-

| |
|---|

| |
|---|

Figure 12. Analysis of potential in-distribution bias. Error bars indicate the 99% confidence intervals. For image generation models, we use the versions available as of January 2026.

Table 15. Comparison of the mean scores when prioritizing either reference consistency or background–subject match for the subset generated by Nano Banana, latest version as of January 2026.

Reference Consistency

Background– Subject Match

Prompt

Original 3.58 3.67 Consistency priority 4.17 3.25 Matching priority 3.40 4.03

subject match. To further verify this, we created crossdomain editing tasks with explicit instructions that prioritize one criterion over the other for 20 randomly selected prompts. Table 15 shows that Nano Banana can improve the prioritized criterion while maintaining the other criterion.

##### F.6. Alignment with Real-World Use Cases

As shown in Figure 13 (right), MultiBanana is directly relevant to real-world use cases, such as advertising. Our

``Combine the dog from image 2 with the office setting and city background. Change the

dog's attire to a business suit and place it in a chair at a desk with a laptop, matching the pose of the dog in image 2. Rewrite the text from image 1 in the style and font of the

`` The woman from image 1 and the woman from image 2 are standing in a snowy

landscape in front of a cabin. The background of the image is the same as in image 3 .’’

text from image 3, and place it below the dog in the combined image.’’

|Example|[Figure 198]|
|---|---|
| | |

|[Figure 199]<br><br>GPT| |
|---|---|
| | |

|Reference 1|[Figure 200]|
|---|---|
| | |

|Reference 2|[Figure 201]|
|---|---|
| | |

|[Figure 202]<br><br>Reference 3| |
|---|---|
| | |

|[Figure 203]<br><br>Reference 3| |
|---|---|
| | |

|[Figure 204]<br><br>Reference 2| |
|---|---|
| | |

|[Figure 205]<br><br>Reference 1| |
|---|---|
| | |

- Figure 13. Left: A sample generated by GPT-Image-1 on a cross-domain task. Right: Task example of our benchmark directly relevant to real-world use cases, such as advertising.

benchmark development using synthetic data also aligns with the community standard. For example, ImgEdit [91] similarly employs GPT-4o for instruction generation, which also aims to develop “practical and powerful tools for realworld applications.”

##### F.7. Agentic Inference

We introduced three agentic frameworks: Iterative Prompt Refinement (IPR), Context-Aware Feedback Generation (CAFG), and Selective Reference Adaptation (SRA). Here, we refer to Gen as the Generator, which produces an image, and Plan as the Planner, which updates the instruction prompt (and selects reference images) for the next step. Let Pt denote the instruction prompt at step t, {Ri}i∈[I] the reference images, and Gt the generated image with G0 = ∅. We conducted experiments with both GPT and Gemini. For GPT, we use GPT-Image-1 as the Generator and GPT-5 as the Planner. For Gemini, we use Nano Banana as the Generator and Gemini 2.5 Flash as the Planner.

###### F.7.1. Iterative Prompt Refinement (IPR)

The IPR framework is formulated as follows. The Planner refines the prompt based on the generation result from the previous step.

Gt+1 = Gen(Pt,{Ri}i∈[I],∅), Pt+1 = Plan(Pt,{Ri}i∈[I],Gt+1).

###### F.7.2. Context-Aware Feedback Generation (CAFG)

The CAFG framework is formulated as follows. The Generator generates the image based on the generation result (context) from the previous step.

Gt+1 = Gen(Pt,{Ri}i∈[I],Gt), Pt+1 = Plan(Pt,{Ri}i∈[I],Gt+1).

###### F.7.3. Selective Reference Adaptation (SRA)

The SRA framework is formulated as follows. The Planner selects only the reference images that should be improved based on the generation result from the previous step, and the Generator receives these as context. SRA is expected

to reduce agents’ task complexity by adaptively decreasing the number of references.

Gt+1 = Gen(Pt,{Ri}i∈Ut,Gt), Pt+1,Ut+1 = Plan(Pt,{Ri}i∈[I],Gt+1).

where Ut is the index set of reference images that are insufficiently reflected in the generated image Gt.

###### F.7.4. Experiments

We set the maximum number of steps t to 3 and evaluate three agentic frameworks—IPR, CAFG, and SRA—using Gemini, and the IPR framework using GPT. Figure 14 illustrates an example of step-wise improvement in the IPR framework. Figure 20, Figure 21, Figure 22, and Figure 23 present the evaluation results of multi-reference generations at each step, averaged across judgments from Gemini 2.5 Flash and GPT-5, broken down by category. Figure 24 presents the results for single and two-reference generations. Nano Banana (Gemini) shows modest improvements in Physical Realism and Visual Quality across refinement steps, while Instruction Alignment and Reference Consistency either remain unchanged or deteriorate. In contrast, GPT demonstrates consistent improvements across all categories. This suggests that Gemini’s planner progressively loses information from the original prompt as refinement steps proceed.

#### G. Implementations

Code: https : / / github . com / matsuolab / multibanana. Dataset: https://huggingface.co/datasets/ kohsei/MultiBanana-Benchmark.

API endpoints: For VLM evaluation, we used stable API versions: gpt-5-2025-08-07 and gemini-2.5-flash. For image generation models, we use the latest available API versions as of November 2025, unless otherwise noted.

[Figure 206]

- Figure 14. Example of changes in the Iterative Prompt Refinement (IPR) framework. Comparison and evaluation results between Nano Banana and GPT on the task of generating images from 8 object references according to the instruction prompt.

#### H. Extended Related Works

##### H.1. Controllable Text-to-Image Generation

Stable Diffusion [15, 60, 65], FLUX [41, 42], DALLE [63], and Imagen [32, 67], have demonstrated strong textto-image generation capabilities, establishing a scalable foundation for the task. To enhance controllability, models such as ControlNet [94] and T2I-Adapter [52] introduced external conditioning modules, enabling image-conditioned generation. Additionally, training-free approaches (e.g., pix2pix-zero [59] and prompt-to-prompt [25]) have also been widely proposed [6, 31, 39, 49, 74, 75]. These advancements collectively demonstrate that diffusion models are becoming increasingly flexible and adaptable across diverse conditional generation settings.

##### H.2. Benchmarks for Reference-Based Generation

Reference-based image generation is closely related to industrial applications such as content production [37, 66, 78, 88, 90, 96], advertising [28, 33, 51], and fashion design [11, 12, 16, 16, 36, 86, 98]. Among reference-driven image generation tasks, the most widely recognized one is image editing. RealEdit [71] focuses on single-image editing and provides an empirical analysis of real-world imageediting use cases. SpotEdit [20] proposes a benchmark for visually guided image editing. These works focus on benchmarking single-reference image edition/generation, while we focus on multi-reference image generation. In recent years, increasing attention has been directed toward multireference image generation, in which multiple reference images are jointly used to generate a single image. MultiRefBench [9] studies controllable image generation with multiple visual references, but our benchmark outperforms it by providing 3,769 samples (compared to 1,990), supporting up to 8 references (compared to 6), and using raw images

(compared to bounding boxes or masks), making it more practical.

##### H.3. Instruction-Following in LLMs

The task of generating an image that faithfully reflects multiple references is analogous to instruction following in LLMs [14, 46, 76, 95, 97], in which models must simultaneously satisfy multiple constraints. In the text domain, recent studies have revealed that LLMs struggle to follow all given instructions as the number of constraints grows, often ignoring or inadequately addressing a subset of them [23, 24, 35, 40]. Multi-reference image generation poses a parallel challenge: as the number of reference images increases, models must reconcile potentially conflicting visual cues while remaining faithful to every reference.

#### I. Limitations and Discussions

Our benchmark focuses on static image generation and does not address video generation or editing [21, 27, 55], where temporal consistency across frames introduces additional challenges for reference-based conditioning [1, 10, 29]. Moreover, our prompts describe object placement in natural language (e.g., ”on the left,” ”in the foreground”), which is inherently ambiguous. Combining reference images with structured layout inputs [45, 61, 89, 94] could enable finergrained control over subject composition.

#### J. Prompts

##### J.1. Prompts of Agentic Framework

###### Prompt of the Generator in the IPR framework

{prompt}{reference image files}

###### Prompt of the Planner in the IPR framework

You are a prompt-refiner agent. Previous prompt: {previous prompt} You are given multiple reference images and one generated image. The generated image is the last one, and the reference images come before it.

Based on the reference images and the generated image (the last image), propose a refined prompt that improves how reference images are used and adjusts the composition/style.

Return only the new prompt text. {reference image files}{previously generated image file}

###### Prompt of the Generator in the CAFG framework

{prompt}

The last image provided is the previously generated image from the last step and all images before it are reference images.

Please refine the previous generation using the reference images and enhance composition/style. {reference image files}{previously generated image file}

###### Prompt of the Planner in the CAFG framework

You are a prompt-refiner agent. Previous prompt: {previous prompt} You are given multiple reference images and one generated image. The generated image is the last one, and the reference images come before it.

Based on the reference images and the generated image (the last image), propose a refined prompt that improves how reference images are used and adjusts the composition/style.

Return only the new prompt text. {reference image files}{previously generated image file}

###### Prompt of the Generator in the SRA framework

{prompt}

The last image provided is the previously generated image from the last step and all images before it are reference images.

Please refine the previous generation using the reference images and enhance composition/style. {reference image files}{previously generated image file}

###### Prompt of the Planner in the SRA framework

You are a prompt-refiner and reference-selector agent. Previous prompt: {previous prompt} You are given multiple reference images (in order) and one generated image (the last one). Your task:

- 1. Identify which reference images are insufficiently reflected in the generated image.
- 2. Propose a refined prompt that improves the generation.
- 3. Return your response in JSON format ONLY. Do not include any other text.

JSON format: {

”indices”: [0, 2, 3], ”prompt”: ”Your refined prompt text here”

} Where ’indices’ is an array of 0-based indices (from 0 to {len(reference image files)-1}) of reference images that are insufficiently reflected in the generated image. ’prompt’ is the refined instruction prompt for the next generation step. Example response: {

”indices”: [0, 2, 3], ”prompt”: ”Focus more on the lighting from the first im-

age and the composition from images 3 and 4. Emphasize the color palette and texture details.”

} {reference image files}{previously generated image file}

##### J.2. Prompts of VLM as Judge

Multi-Reference Image Genearation Evaluation

You are a strict data rater specializing in grading multi-reference driven image generation. You will be given reference images, a task instruction, and the generation results. Reference Images: {reference image files} Editing Instruction: {instruction} Final Output: {generated image files} Your task is to evaluate the effectiveness of replacement editing from five independent perspectives, each on a 10point scale. Note that the average score should be considered 4 points.

- 1. Text-Instruction Alignment Evaluate whether the generated image accurately follows the given text instruction. Check whether the specified objects appear in the correct positions, whether the instructed subjects are depicted properly, and whether no unintended elements are introduced. For example, if the instruction says “change the language,” but the actual written content itself is altered incorrectly, or if unnecessary objects are added, the score should be reduced. If the instruction requires including a reference subject but the generated image fails to include that referenced content, the score must be 1. Even if the instruction is followed correctly, the score must not exceed 6 points if the generated image still exhibits any composited or unnatural appearance.
- 2. Reference Consistency Evaluate how consistent the generated image is with the provided reference images. Compare the output to each reference and assess how faithfully the structure and attributes are reproduced. Fine details, such as hair ornaments, patterns on clothing, and other small features, must match the references; otherwise the score must not exceed 4 points. If even a single object fails to follow the details of the reference images, the score must not exceed 6 points.
- 3. Background-Subject Match Evaluate whether the subject blends naturally with the background. Check whether the subject appears to be floating, unnaturally pasted on, or visually inconsistent with its surroundings. Images that look like multiple pictures simply pasted together should receive a score of 1. If there is even the slightest inconsistency in style, tone, lighting, or overall visual impression compared to the reference images, the score must also not exceed 4 points.
- 4. Physical Realism Evaluate whether the generated image maintains physical plausibility. Penalize cases where the image violates basic physical laws—for example, a person floating in mid-air, standing on water, or having the lower body missing despite no obstruction. If there is even a slight impression that the image looks composited or artificially pasted together, the score must not exceed 4 points. Likewise, if it is unclear whether the subject is actually making proper contact with the ground, the score must also not exceed 6 points.
- 5. Visual Quality Evaluate the overall perceptual quality of the image. Assess whether the image is visually appealing and aesthetically coherent. If the composition appears unnatural or the image does not look aesthetically pleasing to a human observer, the score must not exceed 4 points.

Each of the five scores must be evaluated independently. Do not force any score to be tied to or capped by another score. First, explain the reasoning, then present the final assessment. Start the reasoning with Reasoning: . After explaining the reasoning, present the final assessment in the format: Instruction Alignment: ⟨A number from 1 to 10⟩. Reference Consistency: ⟨A number from 1 to 10⟩. Background-Subject Match: ⟨A number from 1 to 10⟩. Physical Realism: ⟨A number from 1 to 10⟩. Visual Quality: ⟨A number from 1 to 10⟩.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

###### Figure 15. Scores of each model across evaluation metrics for each task by GPT. The horizontal axis denotes the scores for the five evaluation criteria, and the vertical axis denotes the number of reference images.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

###### Figure 16. Scores of each model across evaluation metrics for each task by Gemini. The horizontal axis denotes the scores for the five evaluation criteria, and the vertical axis denotes the number of reference images.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

###### Figure 17. Average scores of each model across evaluation metrics for each task by the average of GPT and Gemini. The horizontal axis denotes the scores for the five evaluation criteria, and the vertical axis denotes the number of reference images.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

###### Figure 18. Total scores of each model for single and two-reference tasks by GPT, Gemini, and their average.

Cross domain

###### Different scale and view

###### Rare concept

###### Multilingual

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Include Others

Instruction

Alignment

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Consistency Background

Reference

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

SubjectMatch

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

PhysicalRealism

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

VisualQuality

Nano Banana GPT-Image-1 Qwen-Image-Edit-2509 DreamOmni2

Figure 19. Total scores for each difficult reference combination across models. Darker colors represent the average score for tasks that include the corresponding combination, whereas lighter colors indicate the average score for tasks that do not include it.

###### IPR Framework with Gemini

Instruction Alignment Steps

###### Reference Consistency

###### Background Subject Match

###### Physical Realism

###### Visual Quality

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 3

- 4

- 5

4.5

3.5

- 3
- 4
- 5
- 6

- Step 1

- Step 2

- Step 3

3.5

4.0

3.0

back

3.0

3.5

2.5

2.5

3.0

2.0

5.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

7.0

5.0

6.5

4.5

6.0

4.5

###### global

4.0

6.5

6.0

4.0

5.5

3.5

3.5

5.5

3.0

6.0

5.0

3.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

6.5

4.5

5.5

4.0

5.5

4.0

###### local

6.0

3.5

5.0

5.0

3.5

3.0

5.5

4.5

3.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

- 4
- 5
- 6

- 2 3 4 5 6 7 8 Reference
- 3
- 4
- 5

4.5

4.5

5.5

object

4.0

4.0

5.0

3.5

3.5

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

Reference

Reference

Reference

Reference

Figure 20. Detailed results of multi-reference image generation using the IPR framework with Gemini.

###### CAFG Framework with Gemini

Instruction Alignment Steps

###### Reference Consistency

###### Background Subject Match

###### Physical Realism

###### Visual Quality

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

5.0

3.00

- 3
- 4
- 5
- 6

2.75

- Step 1

- Step 2

- Step 3

3.5

4.5

2.75

back

2.50

4.0

2.50

3.0

3.5

2.25

2.25

3.0

2.00

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

7.0

6.0

5.0

4.5

6.5

4.5

###### global

6.5

4.0

5.5

6.0

4.0

3.5

5.0

6.0

3.5

5.5

3.0

5.5

6.5

5.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

4.0

4.5

6.0

5.0

5.0

###### local

3.5

4.0

5.5

4.5

4.5

3.0

3.5

5.0

2.5

4.0

3.0

4.0

5.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

4.4

- 4
- 5
- 6

4.2

5.5

5.0

4.2

4.0

###### object

4.0

3.8

4.5

5.0

3.8

3.6

4.0

3.6

3.4

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

Reference

Reference

Reference

Reference

Reference

Figure 21. Detailed results of multi-reference image generation using the CAFG framework with Gemini.

###### SRA Framework with Gemini

Instruction Alignment Steps

###### Reference Consistency

###### Background Subject Match

###### Physical Realism

###### Visual Quality

- 3

- 4

- 5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

3.00

- 4
- 5
- 6

2.6

3.4

Step 1 Step 2 Step 3

2.75

3.2

2.4

back

2.50

3.0

2.2

2.25

2.8

5.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

7.0

5.0

6.5

6.0

5.0

4.5

6.5

###### global

4.5

6.0

4.0

5.5

6.0

4.0

3.5

5.5

3.5

5.0

5.5

3.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

4.5

6.0

5.0

4.5

5.0

4.0

###### local

4.0

5.5

4.5

4.5

3.5

3.5

5.0

4.0

4.0

3.0

3.0

5.5

4.2

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

5.25

- 4
- 5
- 6

4.00

5.0

4.0

5.00

###### object

3.75

4.5

3.8

4.75

3.50

4.0

3.6

4.50

3.25

3.5

3.4

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

Reference

Reference

Reference

Reference

Reference

Figure 22. Detailed results of multi-reference image generation using the SRA framework with Gemini.

###### IPR Framework with GPT

Instruction Alignment Steps

###### Reference Consistency

###### Background Subject Match

###### Physical Realism

###### Visual Quality

5.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 5
- 6
- 7

- Step 1

- Step 2

- Step 3

- 4

- 5

- 6

- 4
- 5
- 6

- 4
- 5
- 6

5.0

back

4.5

4.0

3.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

7.5

8.0

5.0

- 4
- 5
- 6
- 7

8.00

7.0

7.75

###### global

7.5

4.5

7.50

6.5

4.0

7.0

7.25

6.0

3.5

6.5

7.5

7.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

8.0

- 3
- 4
- 5
- 6

5.0

7.0

7.0

7.5

6.5

###### local

6.5

4.5

7.0

6.0

6.0

4.0

5.5

6.5

5.5

6.0

6.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

7.0

- 4
- 5
- 6

5.5

5.5

5.5

6.5

###### object

5.0

5.0

6.0

5.0

4.5

4.5

5.5

4.0

4.5

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

Reference

Reference

Reference

Reference

Reference

###### Figure 23. Detailed results of multi-reference image generation using the IPR framework with GPT.

Instruction Alignment

###### Reference Consistency

###### Background Subject Match

###### Physical Realism

###### Visual Quality

Steps

hair

hair makeup

hair

hair

hair makeup

- Step 1

- Step 2

- Step 3

color

color

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

background

background

background

background

background

material

material

material

material

material

10

10

GeminiIPR

7.5

7.5

5

5

2.5

2.5

0

0

pose

add

pose

add

pose

add

pose

add

pose

add

rep

e

replace

tone

rep

e

rep

e

replace

tone

single

single

text

text

style

style

style

style

style

hair

hair makeup

hair

hair makeup

hair

color

color

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

background

background

background

background

background

material

material

material

material

material

10

10

7.5

7.5

5

5

###### GptIPR

2.5

2.5

0

0

pose

add

pose

add

pose

add

pose

add

pose

add

rep

e

replace

tone

rep

e

replace

tone

rep

e

single

single

text

text

style

style

style

style

style

hair

hair

hair makeup

hair

hair

color

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

background

background

background

background

background

material

material

material

material

material

###### GeminiCAFG

10

7.5

5

2.5

0

pose

add

pose

add

pose

add

pose

add

pose

add

rep

e

rep

e

replace

tone

rep

e

rep

e

single

text

style

style

style

style

style

hair

hair

hair

hair makeup

hair

color

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

|makeup<br><br>|color<br><br>0<br><br>2.5<br><br>5<br><br>7.5<br><br>10<br><br>|
|---|---|
|replace<br><br>single<br><br>|text<br><br>tone<br><br>|

background

background

background

background

background

material

material

material

material

material

10

###### GeminiSRA

7.5

5

2.5

0

pose

add

pose

add

pose

add

pose

add

pose

add

rep

e

rep

e

rep

e

replace

tone

rep

e

single

text

style

style

style

style

style

###### Figure 24. Detailed results of single and two-reference image generation using the agentic frameworks.

|“The man in Image 1 is on the far left, the blue rabbit in Image 2 is in the center, the rocks and Zen garden<br><br>elements from Image 3 are in the middle-right, and the dragon blacksmith in Image 4 is on the far right. They are arranged in a surreal landscape.”|
|---|

|Reference 1|[Figure 207]|
|---|---|
| | |

|Reference 2|[Figure 208]|
|---|---|
| | |

|[Figure 209]<br><br>GPT| |
|---|---|
| | |

|[Figure 210]<br><br>Nano Banana| |
|---|---|
| | |

|[Figure 211]<br><br>Reference 3| |
|---|---|
| | |

|Reference 4|[Figure 212]|
|---|---|
| | |

|DreamOmni2|[Figure 213]|
|---|---|
| | |

|[Figure 214]<br><br>OmniGen2| |
|---|---|
| | |

|[Figure 215]<br><br>Qwen-Image-Edit-2509| |
|---|---|
| | |

|“The man in Image 1 is standing in the center, the woman in Image 2 is seated to the left, the person in Image 3 is to the right, and the cat in Image 4 is in the foreground. They are gathered in a mystical forest clearing.”|
|---|

|Reference 1|[Figure 216]|
|---|---|
| | |

|Reference 2|[Figure 217]|
|---|---|
| | |

|GPT|[Figure 218]|
|---|---|
| | |

|[Figure 219]<br><br>Nano Banana| |
|---|---|
| | |

|Reference 3|[Figure 220]|
|---|---|
| | |

|[Figure 221]<br><br>Reference 4| |
|---|---|
| | |

|[Figure 222]<br><br>DreamOmni2| |
|---|---|
| | |

|[Figure 223]<br><br>OmniGen2| |
|---|---|
| | |

|Qwen-Image-Edit-2509|[Figure 224]|
|---|---|
| | |

|“The woman in Image 1 is on the left, the giraffe in Image 2 is in the center, and the anime character in Image 3 is on the right. The cat from Image 4 is in the foreground, in front of the woman.”|
|---|

|Reference 1|[Figure 225]|
|---|---|
| | |

|[Figure 226]<br><br>Reference 2| |
|---|---|
| | |

|GPT|[Figure 227]|
|---|---|
| | |

|Nano Banana|[Figure 228]|
|---|---|
| | |

|[Figure 229]<br><br>Reference 3| |
|---|---|
| | |

|Reference 4|[Figure 230]|
|---|---|
| | |

|[Figure 231]<br><br>DreamOmni2| |
|---|---|
| | |

|OmniGen2|[Figure 232]|
|---|---|
| | |

|Qwen-Image-Edit-2509|[Figure 233]|
|---|---|
| | |

|“The woman in Image 1 is on the left, the woman in Image 2 is in the center, the cat in Image 3 is on the right, and the man in Image 4 is in the background.”|
|---|

|[Figure 234]<br><br>Reference 1| |
|---|---|
| | |

|[Figure 235]<br><br>Reference 2| |
|---|---|
| | |

|[Figure 236]<br><br>GPT| |
|---|---|
| | |

|Nano Banana|[Figure 237]|
|---|---|
| | |

|[Figure 238]<br><br>Reference 3| |
|---|---|
| | |

|[Figure 239]<br><br>Reference 4| |
|---|---|
| | |

|[Figure 240]<br><br>DreamOmni2| |
|---|---|
| | |

|OmniGen2|[Figure 241]|
|---|---|
| | |

|[Figure 242]<br><br>Qwen-Image-Edit-2509| |
|---|---|
| | |

Figure 25. Qualitative example for 4-Object Tasks.

|Reference 1|[Figure 243]|
|---|---|
| | |

|[Figure 244]<br><br>GPT| |
|---|---|
| | |

|Nano Banana|[Figure 245]|
|---|---|
| | |

[Figure 246]

Reference 2

|[Figure 247]<br><br>Reference 3| |
|---|---|
| | |

|DreamOmni2|[Figure 248]|
|---|---|
| | |

|[Figure 249]<br><br>OmniGen2| |
|---|---|
| | |

|[Figure 250]<br><br>Qwen-Image-Edit-2509| |
|---|---|
| | |

[Figure 251]

Reference 4

|“The man from image 1 and the woman from image 3 are in the garden of the house from image 4.”|
|---|

|Reference 1|[Figure 252]|
|---|---|
| | |

|GPT|[Figure 253]|
|---|---|
| | |

|Nano Banana|[Figure 254]|
|---|---|
| | |

[Figure 255]

###### Reference 2

|Reference 3|[Figure 256]|
|---|---|
| | |

|[Figure 257]<br><br>DreamOmni2| |
|---|---|
| | |

|[Figure 258]<br><br>OmniGen2| |
|---|---|
| | |

|[Figure 259]<br><br>Reference 4| |
|---|---|
| | |

|[Figure 260]<br><br>Qwen-Image-Edit-2509| |
|---|---|
| | |

|“The man from image 1, the woman from image 2, and the lifeguard from image 3 are standing in front of a waterfall. The background of the image is the same as in image 4.”|
|---|

|[Figure 261]<br><br>Nano Banana| |
|---|---|
| | |

| |Reference 1|[Figure 262]|
|---|---|---|
| | | |

|GPT|[Figure 263]|
|---|---|
| | |

[Figure 264]

Reference 2

|[Figure 265]<br><br>Reference 3| |
|---|---|
| | |

|[Figure 266]<br><br>DreamOmni2| |
|---|---|
| | |

|OmniGen2|[Figure 267]|
|---|---|
| | |

|Qwen-Image-Edit-2509|[Figure 268]|
|---|---|
| | |

[Figure 269]

Reference 4

|“The girl from image 1, the person from image 2, and the bird from image 3 are arranged in front of a snowy landscape with a cabin and the aurora borealis, which is the background of image 4.”|
|---|

|[Figure 270]<br><br>Reference 2| |
|---|---|
| | |

|[Figure 271]<br><br>GPT| |
|---|---|
| | |

|Nano Banana|[Figure 272]|
|---|---|
| | |

|Reference 1|[Figure 273]|
|---|---|
| | |

|Re|[Figure 274]<br><br>ference 3| |
|---|---|---|
| | | |

|[Figure 275]<br><br>DreamOmni2| |
|---|---|
| | |

|OmniGen2|[Figure 276]|
|---|---|
| | |

|Reference 4|
|---|

|[Figure 277]|
|---|

|Qwen-Image-Edit-2509|[Figure 278]|
|---|---|
| | |

Figure 26. Qualitative example for 3-Object + Background Tasks.

|[Figure 279]<br><br>Reference 1| |
|---|---|
| | |

|Reference 2|[Figure 280]|
|---|---|
| | |

|[Figure 281]<br><br>GPT| |
|---|---|
| | |

|Nano Banana|[Figure 282]|
|---|---|
| | |

|Reference 3|[Figure 283]|
|---|---|
| | |

|[Figure 284]<br><br>Reference 4| |
|---|---|
| | |

|DreamOmni2|[Figure 285]|
|---|---|
| | |

|OmniGen2|[Figure 286]|
|---|---|
| | |

|Qwen-Image-Edit-2509|[Figure 287]|
|---|---|
| | |

|“The leopard gecko from image 2 is sitting on a wooden barrel from image 4, which is filled with grapes. The<br><br>leopard gecko wears the blue eyeliner and black trench coat from the man in image 1, and has the silhouette and golden hour lighting of the giraffe from image 3.”|
|---|

|[Figure 288]<br><br>Reference 1| |
|---|---|
| | |

|Reference 2|[Figure 289]|
|---|---|
| | |

|GPT|[Figure 290]|
|---|---|
| | |

|[Figure 291]<br><br>Nano Banana| |
|---|---|
| | |

|[Figure 292]<br><br>Reference 3| |
|---|---|
| | |

|Reference 4|[Figure 293]|
|---|---|
| | |

|DreamOmni2|[Figure 294]|
|---|---|
| | |

|OmniGen2|[Figure 295]|
|---|---|
| | |

|Qwen-Image-Edit-2509|[Figure 296]|
|---|---|
| | |

|“The coffee grinder from image 1 is held by the man in image 4, while the bird from image 2 is perched on the<br><br>man's shoulder, and the hairstyles from image 3 are reflected on the coffee grinder.”|
|---|

|[Figure 297]<br><br>Reference 1| |
|---|---|
| | |

|Ref|[Figure 298]<br><br>erence 2| |
|---|---|---|
| | | |

|GPT|[Figure 299]|
|---|---|
| | |

[Figure 300]

Nano Banana

|[Figure 301]<br><br>Reference 3| |
|---|---|
| | |

|Ref|[Figure 302]<br><br>erence 4| |
|---|---|---|
| | | |

|DreamOmni2|[Figure 303]|
|---|---|
| | |

|[Figure 304]<br><br>OmniGen2| |
|---|---|
| | |

Qwen-Image-Edit-2509

[Figure 305]

|“Combine the truck from image 1 and the doll from image 3 into a single image, with the truck's color changed to<br><br>match the doll's dress pattern and the doll's pose adjusted to be sitting in the truck. Rewrite the text on the sign in<br><br>image 2 using the font from the text in image 4..”|
|---|

|Reference 1|[Figure 306]|
|---|---|
| | |

|[Figure 307]<br><br>Reference 2| |
|---|---|
| | |

|[Figure 308]<br><br>GPT| |
|---|---|
| | |

|Nano Banana|[Figure 309]|
|---|---|
| | |

|[Figure 310]<br><br>Reference 3| |
|---|---|
| | |

|[Figure 311]<br><br>Reference 4| |
|---|---|
| | |

|[Figure 312]<br><br>DreamOmni2| |
|---|---|
| | |

|OmniGen2|[Figure 313]|
|---|---|
| | |

|[Figure 314]<br><br>Qwen-Image-Edit-2509| |
|---|---|
| | |

- Figure 27. Qualitative example for 3-Object + Local Tasks.

|“The woman from image 2 should be placed on the bench in image 1, and the woman from image 3 should be positioned in the foreground of image 1. The entire composition should be rendered in the style of the painting in image 4.”|
|---|

|[Figure 315]<br><br>Reference 1| |
|---|---|
| | |

|Reference 2|[Figure 316]|
|---|---|
| | |

|GPT|[Figure 317]|
|---|---|
| | |

|[Figure 318]<br><br>Nano Banana| |
|---|---|
| | |

|Reference 3|[Figure 319]|
|---|---|
| | |

|[Figure 320]<br><br>DreamOmni2| |
|---|---|
| | |

|OmniGen2|[Figure 321]|
|---|---|
| | |

|Reference 4|[Figure 322]|
|---|---|
| | |

|Qwen-Image-Edit-2509|[Figure 323]|
|---|---|
| | |

|“The ornate chair from the first image, the woman from the second image, and the man from the third image are combined into a single image. The woman is seated on the chair, and the man is standing behind her. The style of the image is the same as in the fourth image.”|
|---|

|[Figure 324]<br><br>Reference 1| |
|---|---|
| | |

|[Figure 325]<br><br>Reference 2| |
|---|---|
| | |

|GPT|[Figure 326]|
|---|---|
| | |

|Nano Banana|[Figure 327]|
|---|---|
| | |

|DreamOmni2|[Figure 328]|
|---|---|
| | |

|OmniGen2|[Figure 329]|
|---|---|
| | |

[Figure 330]

|Qwen-Image-Edit-2509|[Figure 331]|
|---|---|
| | |

[Figure 332]

Reference 4

Reference 3

|“The man from image 1 and the fairy from image 2 are gathered with the woman and people from image 3, with the style of the image being the same as in image 4.”|
|---|

|[Figure 333]<br><br>Reference 1| |
|---|---|
| | |

|[Figure 334]<br><br>GPT| |
|---|---|
| | |

[Figure 335]

|[Figure 336]<br><br>Nano Banana| |
|---|---|
| | |

Reference 2

|Reference 3|[Figure 337]|
|---|---|
| | |

|DreamOmni2|[Figure 338]|
|---|---|
| | |

|OmniGen2|[Figure 339]|
|---|---|
| | |

|Reference 4|[Figure 340]|
|---|---|
| | |

[Figure 341]

Qwen-Image-Edit-2509

|“The man from image 1, the geisha from image 2, and the couple from image 3 are standing in a bustling street market, with a large red heart sculpture placed prominently in the foreground. The style of the image is the same as in image 4.”|
|---|

|Reference 1|[Figure 342]|
|---|---|
| | |

|[Figure 343]<br><br>Reference 2| |
|---|---|
| | |

|[Figure 344]<br><br>GPT| |
|---|---|
| | |

|Nano Banana|[Figure 345]|
|---|---|
| | |

|[Figure 346]<br><br>Reference 3| |
|---|---|
| | |

|[Figure 347]<br><br>DreamOmni2| |
|---|---|
| | |

|OmniGen2|[Figure 348]|
|---|---|
| | |

|Reference 4|[Figure 349]|
|---|---|
| | |

|[Figure 350]<br><br>Qwen-Image-Edit-2509| |
|---|---|
| | |

- Figure 28. Qualitative example for 3-Object + Global Tasks.

|“The man with his hand on his forehead in Image 1 is on the left, the flamingos in Image 2 are in the background, the woman with neon<br><br>hair in Image 3 is in the center, the cat in Image 4 is on the right, the person with black lipstick in Image 5 is in the foreground, the raven<br><br>in Image 6 is on a gargoyle, the woman with cybernetic markings in Image 7 is in the middle, and the iguana in an astronaut suit in<br><br>Image 8 is on the right. They are gathered in a surreal cityscape.”|
|---|

|[Figure 351]<br><br>Reference 1| |
|---|---|
| | |

|Reference 2|[Figure 352]|
|---|---|
| | |

|GPT|[Figure 353]|
|---|---|
| | |

|Nano Banana|[Figure 354]|
|---|---|
| | |

|Reference 3|[Figure 355]|
|---|---|
| | |

|[Figure 356]<br><br>Reference 4| |
|---|---|
| | |

|Reference 5|[Figure 357]|
|---|---|
| | |

|[Figure 358]<br><br>Reference 6| |
|---|---|
| | |

|DreamOmni2|[Figure 359]|
|---|---|
| | |

|Qwen-Image-Edit-2509|[Figure 360]|
|---|---|
| | |

|Reference 7|[Figure 361]|
|---|---|
| | |

|Reference 8|[Figure 362]|
|---|---|
| | |

|“The woman from image 1, the woman from image 2, the woman from image 3, the man from image 4, the suit from image 5, the<br><br>astronaut from image 6, and the man from image 7 are arranged in a scene with the background of image 8.”|
|---|

|Reference 2|[Figure 363]|
|---|---|
| | |

|GPT|[Figure 364]|
|---|---|
| | |

|Reference 1|
|---|

|Nano Banana|
|---|

|Reference 3|[Figure 365]|
|---|---|
| | |

|[Figure 366]<br><br>Reference 4| |
|---|---|
| | |

|[Figure 367]|
|---|

|[Figure 368]|
|---|

|Reference 6|[Figure 369]|
|---|---|
| | |

|Reference 5|
|---|

|[Figure 370]<br><br>DreamOmni2| |
|---|---|
| | |

|[Figure 371]<br><br>Reference 7| |
|---|---|
| | |

|Qwen-Image-Edit-2509| |
|---|---|
|[Figure 372]| |

|Reference 8| |
|---|---|
|[Figure 373]| |

|[Figure 374]|
|---|

|“The woman in image 1 should have the pose and expression of the man in image 5, the dog in image 2 should have<br><br>the breed and coloring of the dog in image 6, the tiger chef in image 7 should be angry like the man in image 3, and the<br><br>bell in image 8 should be placed on the wet ground of the city from image 4.”|
|---|

|Reference 1|[Figure 375]|
|---|---|
| | |

|[Figure 376]<br><br>Reference 2| |
|---|---|
| | |

|GPT|[Figure 377]|
|---|---|
| | |

|[Figure 378]<br><br>Nano Banana| |
|---|---|
| | |

|[Figure 379]<br><br>Reference 3| |
|---|---|
| | |

|[Figure 380]<br><br>Reference 4| |
|---|---|
| | |

|[Figure 381]<br><br>Reference 5| |
|---|---|
| | |

|[Figure 382]<br><br>Reference 6| |
|---|---|
| | |

|DreamOmni2|[Figure 383]|
|---|---|
| | |

|[Figure 384]<br><br>Qwen-Image-Edit-2509| |
|---|---|
| | |

|Reference 7|[Figure 385]|
|---|---|
| | |

|Reference 8|[Figure 386]|
|---|---|
| | |

|“The man from image 1, the woman from image 2, the anime girl from image 3, the dog from image 4, the chaise lounge<br><br>from image 5, the woman from image 6, and the man from image 7 are arranged in an impressionistic street scene with<br><br>cars and buildings, in the style of image 8.”|
|---|

|Reference 2|[Figure 387]|
|---|---|
| | |

|[Figure 388]<br><br>GPT| |
|---|---|
| | |

|Re|[Figure 389]<br><br>ference 4| |
|---|---|---|
| | | |

|Reference 1| |
|---|---|
|[Figure 390]| |

|Nano Banana|
|---|

|Reference 3|
|---|

|[Figure 391]|
|---|

|[Figure 392]|
|---|

|Reference 5|[Figure 393]|
|---|---|
| | |

|[Figure 394]<br><br>Reference 6| |
|---|---|
| | |

|[Figure 395]<br><br>DreamOmni2| |
|---|---|
| | |

|Reference 7|
|---|

|Qwen-Image-Edit-2509|
|---|

|Reference 8|
|---|

|[Figure 396]|
|---|

|[Figure 397]|
|---|

|[Figure 398]|
|---|

- Figure 29. Qualitative example for Tasks with 8 references.

