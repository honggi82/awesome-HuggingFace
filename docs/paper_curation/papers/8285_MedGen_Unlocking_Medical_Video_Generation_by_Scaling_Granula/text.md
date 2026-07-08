# arXiv:2507.05675v1[cs.CV]8Jul2025

## MedGen: Unlocking Medical Video Generation by Scaling Granularly-annotated Medical Videos

### Rongsheng Wang1*Junying Chen1*, Ke Ji1, Zhenyang Cai1, Shunian Chen1, Yunjin Yang1, Benyou Wang1†

1The Chinese University of Hong Kong, Shenzhen wangbenyou@cuhk.edu.cn

https://github.com/FreedomIntelligence/MedGen

##### Abstract

Recent advances in video generation have shown remarkable progress in open-domain settings, yet medical video generation remains largely underexplored. Medical videos are critical for applications such as clinical training, education, and simulation, requiring not only high visual fidelity but also strict medical accuracy. However, current models often produce unrealistic or erroneous content when applied to medical prompts, largely due to the lack of large-scale, high-quality datasets tailored to the medical domain. To address this gap, we introduce MedVideoCap-55K, the first large-scale, diverse, and caption-rich dataset for medical video generation. It comprises over 55,000 curated clips spanning real-world medical scenarios, providing a strong foundation for training generalist medical video generation models. Built upon this dataset, we develop MedGen, which achieves leading performance among open-source models and rivals commercial systems across multiple benchmarks in both visual quality and medical accuracy. We hope our dataset and model can serve as a valuable resource and help catalyze further research in medical video generation.

[Figure 1]

Disclaimer: This paper contains clinical content that may be disturbing to some readers.

### 1 Introduction

Recent advances in video generation have led to impressive breakthroughs, with models now capable of producing high-quality, cinematic visuals that align closely with user prompts (Blattmann et al. 2023). In particular, latent video diffusion models (LVDMs), such as Sora (OpenAI 2025) and Veo (Sharma et al. 2024), have achieved state-of-theart performance by operating efficiently in latent space and delivering diverse, coherent video outputs from textual descriptions.

Despite this progress, medical video generation remains a largely underexplored yet crucial domain. Medical videos are indispensable in numerous real-world applications, including clinical training, surgical simulation, and patient education (Li et al. 2024b). Unlike everyday video content, medical videos demand precise rendering of anatomical structures, accurate surgical steps, and realistic physiological dynamics. These requirements place significantly higher

*Equal Contribution. †Corresponding author.

demands on visual fidelity, semantic correctness, and temporal coherence.

[Figure 2]

[Figure 3]

Sora Sora

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Pika Pika

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Hailuo Hailuo

[Figure 12]

[Figure 13]

（b）distortion errors

（a）medical common sense errors

（b）distortion errors

Figure 1: Failure cases of Sora, Pika, and Hailuo on medical video generation. (a): Medical common sense errors. (b): Distortion errors.

However, current video generation models are trained almost exclusively on general-purpose datasets that focus on natural scenes and everyday activities (Blattmann et al. 2023). As a result, when applied to medical prompts, they often generate outputs with critical errors—such as anatomical inconsistencies, tool misuse, and implausible clinical scenarios. As shown in Figure 1, even leading models like Sora (OpenAI 2025), Pika (Team 2025a), and Hailuo (MiniMax 2025) fail to maintain basic medical realism, revealing a clear mismatch between training data and medical domain requirements. A key bottleneck lies in the lack of largescale, high-quality datasets tailored for medical video generation (Sun et al. 2024). Existing medical datasets are limited in size, narrow in scope (e.g., only endoscopic or surgical videos), and mostly provide categorical labels instead of detailed descriptions—making them unsuitable for training or evaluating text-to-video generation models that rely on rich, semantic input.

To address this gap, we introduce MedVideoCap-55K,

- Table 1: Comparison of existing medical video datasets. MedVideoCap-55K offers the largest scale, highest diversity, and is the first to include detailed text captions tailored for medical text-to-video generation. “#” and “*” denote “number” and “average.”

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

: clinical practice, : medical science, : medical teaching, : medical imaging, : medical animation.

Dataset Caption #Vid. Len.* Words* Resolution Category HyperKvasir (Borgli et al. 2020) Category Labels 120 4s None 512 × 320 SurgicalActions160 (Schoeffmann et al. 2018) Category Labels 160 2s None 320 × 240 Pitvis (Das et al. 2024) Category Labels 25 62m None 1280 × 720 CatRelDet (Ghamsarian et al. 2021) Category Labels 2,200 3s None 224 × 224 MedvidCL (Gupta, Attal, and Demner-Fushman 2023) Category Labels 6,117 4s None 512 × 320 Cholec80 (Twinanda et al. 2016) Category Labels 80 50m None 1920 × 1080 Colonoscopic (Mesejo et al. 2016) Category Labels 76 20s None 768 × 576 MESAD-Real (Bawa et al. 2021) Category Labels 23,366 5s None 720 × 756 SurgToolLoc (Zia et al. 2023) Category Labels 24,695 30s None 1280 × 720 Bora (Sun et al. 2024) Text Captions 4,897 5s 110 256 × 256 MedVideoCap-55K (ours) Text Captions 55,803 8s 174 720 × 480

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

the first large-scale medical video dataset specifically designed for text-to-video generation. It contains over 55K carefully filtered video clips spanning a wide spectrum of real-world medical scenarios, including clinical practice, imaging, education, animation, and science popularization. Each video is paired with high-quality, descriptive captions generated by multimodal large language models (MLLMs). The dataset is curated using a rigorous filtering pipeline that ensures high visual quality, medical relevance, and training suitability.

Building on MedVideoCap-55K, we develop MedGen, a specialized medical video generation model. Through comprehensive experiments, we demonstrate that MedGen significantly outperforms other open-source models in both visual quality and medical accuracy. In addition, we propose evaluation metrics tailored to medical video generation—addressing gaps in current benchmarks—and showcase downstream applications such as medical data augmentation and simulation.

Our key contributions are summarized as follows:

- 1. We introduce MedVideoCap-55K, the first large-scale medical video dataset designed for text-to-video generation, containing 55K diverse, high-quality medical clips with detailed captions.
- 2. We develop MedGen, a medical video generation model trained on MedVideoCap-55K. MedGen excels in medical video generation, demonstrating strong performance in both visual quality and medical accuracy.
- 3. We propose a suite of evaluation protocols tailored for medical video generation, including both automatic metrics and expert-based assessments, addressing the limitations of existing general-purpose benchmarks.

### 2 Motivations

Existing Models Fall Short in Medical Video Generation Despite recent advances in general video generation, existing commercial models such as Sora (OpenAI 2025), Pika (Team 2025a), and Hailuo (MiniMax 2025) struggle in the medical domain. As shown in Figure 1, these models frequently generate content with severe medical hallucinations—such as incorrect anatomical structures, misuse of

instruments, or implausible clinical scenes. These failures are not isolated cases but a systemic issue rooted in a lack of medical-specific knowledge and training data. Current models are optimized for general visual patterns, not the precise semantics, constraints, and domain logic required in medical content.

Challenges: Limited Medical Video Datasets A core reason for these failures is the absence of high-quality, largescale medical video datasets. As summarized in Table 1, existing datasets are limited in scope—often focused on narrow tasks such as surgical action recognition or endoscopy classification—and rarely include descriptive captions. This makes them insufficient for training or evaluating text-tovideo generation models that require semantic understanding and rich supervision.

To bridge this gap, we introduce MedVideoCap-55K, the first large-scale, caption-rich dataset specifically designed for medical video generation. Sourced from over 25 million public videos, MedVideoCap-55K contains more than 55K curated medical video clips covering diverse domains such as clinical practice, education, imaging, and animation. Each video is paired with high-quality textual descriptions, enabling training and evaluation in natural language-driven generation tasks. We believe MedVideoCap-55K provides the foundation needed to advance faithful, diverse, and clinically meaningful medical video generation.

### 3 Scaling Granularly-annotated Medical Videos

In this section, we aim to scale up a high-quality medical video dataset, resulting in MedVideoCap-55K. As illustrated in Figure 2, each sample consists of a medical video clip paired with a granularly-annotated textual caption. Next, we will respectively cover dataset construction, enhanced data filtering, and dataset statistics.

#### 3.1 Data Construction

##### Collecting Videos from YouTube

We construct MedVideoCap-55K using publicly available videos from YouTube. To ensure scalability, we began with a large-scale collection of 25 million YouTube videos along

Video

Brief Caption Detailed Caption

"The video shows a surgical procedure in a sterile operating room, focusing on a patient's torso covered in blue cloth, with marked skin and a yellow antiseptic. Two medical professionals in full surgical gear engage in the procedure, one holding a suction device while the other uses instruments. The lighting enhances visibility on the surgical site, capturing the team's precise and coordinated efforts, including the use of a retractor. The video emphasizes the meticulous care taken during the operation."

"The short video depicts a surgical procedure taking place in a sterile operating room environment. The scene is focused on a patient's torso, which is draped in surgical blue cloth, exposing only the area of interest. The patient's skin is marked with tattoos, and the area is prepped with antiseptic solution, evident from the yellowish tint. Two medical professionals, dressed in full surgical attire including gloves, gowns, and masks, are actively engaged in the procedure. One of them is holding a small container, possibly a suction device or a container for collecting fluids, while the other is manipulating surgical instruments. The lighting is concentrated on the surgical site, ensuring visibility and precision. The video captures the meticulous and coordinated efforts of the surgical team as they perform the procedure. The professionals are seen handling various tools, including what appears to be a retractor or similar device to keep the incision open. The presence of a bright light source highlights the area being operated on, ensuring that the surgeons have a clear view of the surgical site. Overall, the video provides a close-up view of the surgical process, emphasizing the precision and care taken by the medical team during the operation."

[Figure 35]

Video Name Resolution

Video Bit Rate Aesthetic Score Dover Score

00003.mp4 720*480 642758 4.88

0.705

Duration

FPS Camera Motion Type Video Categories

8.99s 8 static shot clinical practice videos

Figure 2: Sample from MedVideoCap-55K. Each data point consists of a medical video clip, a brief caption, and a detailed caption.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Medical

[Figure 40]

[Figure 41]

24 Million Content filtering Youtube Videos

Channel recollection

140K Medical Videos

37K Medical Videos

Per-frame Expansion Filter Out Non-medical Frames

[Figure 42]

Remove Temporal Inconsistencies

Clipping

[Figure 43]

Video Clip Video Title and Description Clip Transcript

[Figure 44]

[Figure 45]

Generate 111K Medical Video Clips

[Figure 46]

###### Captions

MLLMs

Figure 3: Overview of the data construction process.

with their associated titles and descriptions. We applied a two-stage filtering pipeline to identify medically relevant content. First, we used a curated medical keyword dictionary to flag potentially relevant videos. Second, we applied a text classifier trained on snowflake-arctic-embed-m (Merrick et al. 2024) to further verify medical relevance based on video metadata. This yields 37K medical videos. To expand coverage, we collect additional videos from the channels that hosted these medical videos, ultimately assembling a total of 140K medical videos with a total duration of 10,269 hours.

Video Segmentation To extract coherent segments rich in medical content, we applied the following video segmentation methods:

- 1. Medical Frame Classification: Leveraging the visual encoder from CLIP (Radford et al. 2021) and humanannotated data, we trained a frame-level classifier C. Videos were sampled at 1 frame per second (FPS), and each frame xi (the i-th frame) was labeled as either medical or non-medical: C(xi) ∈ {1,0}.
- 2. Temporal Consistency: To ensure visual coherence, we

calculated similarity scores S(xi,xi−1) between adjacent frames using CLIP embeddings. Only consecutive frames with similarity above a threshold τ were retained.

A valid clip [xi,xj] was preserved if it satisfied the fol-

lowing conditions:

- j
- k=i

I[j−i ≥ 6]·

j

I[C(xk) = 1]·

k=i+1

I[S(xk,xk−1) > τ] = 1

Finally, we kept only clips with a resolution above 480p, resulting in 111K medical video clips. The overall data construction pipeline is illustrated in Figure 3.

Caption Generation We leverage a multimodal LLM (GPT-4o) to generate detailed captions. Due to context limits, we uniformly sampled 8 frames per clip as visual input. To minimize hallucinations, we supply the video title and description, along with the transcript of the clip. To support models with input length constraints, we further prompt the MLLM to generate a brief caption.

###### Prompt MLLM to Generate Video Caption

The provided images are sampled from a medical video clip (8 evenly spaced frames). This clip is taken from a video with the following metadata:

Video Title: {Video Title} Video Description: {Video Description} The transcript of this medical clip is as follows: {Clip Transcript}

Using the visual content of the clip, along with the title, description, and transcript, please generate a clear, detailed, and accurate description of what is shown in this clip. Focus on visible scenes, actions, and medical elements, including any procedures, instruments, or clinical interactions shown.

#### 3.2 Data Filtering and Quality Refinement

To further improve data quality, we apply a second-stage filtration after initial segmentation. While earlier steps remove obviously irrelevant or low-resolution videos, many subtle yet impactful quality issues remain—such as persistent black borders, heavy subtitles, visual clutter, or technical artifacts. These imperfections are difficult to detect through simple heuristics, but they can significantly degrade the learning of video generation models. To address this, we

|Step1 Rema<br><br>97,7|ining Data<br><br>66<br><br>Step2 Rema<br><br>87,3|ining Data<br><br>44<br><br>Step3 Rema<br><br>71,0|ining Data<br><br>42<br><br>Step4 Rema<br><br>69,0|ining Data<br><br>44|
|---|---|---|---|---|

Final Dataset

55,803

Video Slicing

110,969

Combined Aesthetic Filtered

13,241

Low-level Quality Filtered

1,998

Aesthetic Filtered

16,302

OCR Filtered

10,422

Black Boraders Filtered

13,203

- Figure 4: Overview of our data filtering pipeline. Each stage applies specific filters and shows the volume of data removed and retained.

Medical Popular Science Videos

9.41%

Medical Animation Videos

6.15%

Medical Teaching Videos

55.93%

Medical Imaging Videos

2.39%

Clinical Practice Videos

26.12%

(a) Category Distribution

(b) Detailed Captions

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

(c) Duration

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

(d) Aesthetic Quality

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

(e) Dover

- Figure 5: Data distribution in our MedVideoCap-55K. (a): category distribution of medical videos. (b): word count distribution in detailed captions. (c): duration of video clips. (d): aesthetic score distribution. (e): dover score distribution.

Made at SankeyMATIC.com

design a fine-grained filtering pipeline that systematically removes such noisy content. The full filtration process and the number of videos removed at each stage are visualized in Figure 4. However, each stage processes a large amount of data, making full human validation impractical. Instead, we randomly sample 200 videos at each stage and evaluate them across key quality dimensions. A stage is considered successful if at least 95% of the sampled videos are rated as clean or only slightly noisy.

Black Border Removal We use OpenCV1 to detect black borders, which frequently occur in screen-recorded or lecture-style videos. These borders introduce large static regions that harm the spatial modeling capacity of generative models. Specifically, we apply Canny edge detection followed by Hough line transforms to localize consistent black edges along the video frame boundaries. Videos with persistent borders on any side are excluded.

Subtitle Filtering via OCR To detect overlaid subtitles that may obscure key visual information, we apply EasyOCR2 on 5 uniformly sampled frames from each video. If

- 1https://github.com/opencv/opencv-python
- 2https://github.com/JaidedAI/EasyOCR

the total number of detected words exceeds 20, the video is discarded. This threshold effectively filters out clips with persistent subtitles or dialogue captions, which tend to dominate the visual field and distract from medically relevant content such as instruments or anatomy.

Aesthetic Quality Filtering We assess aesthetic quality using the LAION aesthetic predictor 3. By averaging scores over 5 sampled frames, we filter out videos that are overly blurry, low-resolution, poorly lit, or cluttered with logos and watermarks. A conservative threshold of 3.0 is used to avoid excluding medically important but visually plain content.

Technical Quality Filtering Low-level artifacts such as compression glitches, frame jitter, and bitrate instability can also degrade model training. We assess these factors using Dover (Wu et al. 2023), which provides a technical quality score independent of content semantics. Videos with a Dover score greater than 0 are excluded to ensure temporal and encoding consistency.

Joint Filtering To further eliminate severely degraded content, we apply a stricter combined filter: videos with both

3https://github.com/christophschuhmann/improved-aestheticpredictor

- Table 2: Benchmarking video generation models on Med-VBench. Within each segment, bold highlights the best scores. The warping error score was derived by aggregating the scores from all three evaluators.

Model Total ↑ Imaging

Warping Error ↓ Open-Source Video Generation Models

Subject Consistency ↑

Background Consistency ↑

Motion Smoothness ↑

Quality ↑

CogVideoX-2B 60.05 61.22 94.03 95.01 98.04 88.00 CogVideoX-5B 60.70 56.10 93.92 96.04 98.13 80.00 Wan2.1-T2V-1.3B 59.59 53.81 90.48 92.75 98.03 77.50 OpenSora (v1.2) 59.44 64.88 95.40 96.46 99.40 99.50 OpenSoraPlan (v1.3) 61.39 67.60 97.08 97.38 99.33 93.03 VideoCrafter-2 61.43 70.52 98.13 98.42 98.53 97.00 ModelScope-1.7B 58.27 63.99 92.99 96.24 96.43 100.00 Latte-1 60.60 69.87 96.41 96.75 98.09 97.50 Vchitect-2 56.50 63.02 88.07 93.50 93.89 99.50 Pyramid-Flow 59.10 67.41 91.54 94.75 99.38 98.50 Allegro 60.98 72.22 92.96 95.16 99.01 93.50 Mochi-1-preview 59.84 56.38 92.51 94.60 99.08 83.50 LTX-Video 59.12 61.85 97.44 95.82 99.60 100.00 HunyuanVideo 67.89 65.85 91.47 95.82 99.20 45.00 Wan2.1-T2V-14B 65.13 56.27 91.86 94.13 98.51 50.00 MedGen (Ours) 70.93 72.50 94.41 97.39 99.76 38.50

###### Proprietary Video Generation Models

Pika (v2.0) 70.29 69.62 97.59 97.26 99.57 42.29 Hailuo (video-01) 69.45 69.84 94.55 95.24 99.35 42.27 Kling (v1.6) 72.32 74.42 95.64 93.19 97.39 26.70 Sora 71.92 71.96 93.40 95.36 99.21 28.40

Dover score >0.3 and LAION score <4.0 are removed. This approach balances aesthetic and technical quality, helping preserve clinically valuable yet visually simple videos.

#### 3.3 Data Statistics

High-Quality Clips As shown in Figure 5, most video clips in MedVideoCap-55K range from 6 to 10 seconds, striking a balance between semantic richness and training efficiency. This duration is sufficient to capture complete medical actions or concepts, while remaining suitable for current video generation model limits. Each clip is accompanied by both a brief and a detailed caption, supporting a wide range of supervision granularity. In addition, aesthetic and technical quality scores are concentrated in the medium-to-high range, indicating that the dataset maintains a high standard of visual clarity, coherence, and temporal smoothness.

Diversity of Medical Domains MedVideoCap-55K spans a broad set of medical domains, including clinical procedures, medical education, scientific communication, animations, and imaging, as shown in Figure 5. This diversity supports training models that generalize across various real-world medical contexts. More visual examples and data analysis of MedVideoCap-55K are provided in the Appendix.

### 4 Experiments

In this section, we develop a medically-adapted video generation model, MedGen, which is further trained from a general video generation model using MedVideoCap-55K.

#### 4.1 Medical Video Generation Model: MedGen

Experimental Setup MedGen is built on top of the opensource latent diffusion video model HunyuanVideo4. We train MedGen on 8 NVIDIA A800 GPUs using 50,000 steps with a batch size of 32, LoRA rank set to 32, and a learning rate of 5e-5. Each input sample consists of a medical video clip and a corresponding detailed caption from MedVideoCap-55K. Training focuses on improving the model’s ability to generate semantically aligned and medically accurate videos from text prompts. Further training details are provided in Appendix.

Baselines We compare MedGen with 15 open-source and 4 commercial state-of-the-art video generation models. The open-source baselines include popular latent diffusion and autoregressive models such as CogVideoX (Yang et al.

- 2024), Mochi-1-preview (Team 2024), and Wan2.1 (Team
- 2025b), while the commercial models include Sora (OpenAI 2025), Pika (Team 2025a), and Hailuo (MiniMax 2025).

Evaluation Metrics We evaluate all the models with the following metrics. (1) Med-VBench. VBench (Huang et al. 2024) is a widely used benchmark for evaluating general video generation models, breaking down video quality into specific, hierarchical, and independent dimensions with tailored metrics. Building on this framework, we introduce medical video prompts and rename the benchmark MedVBench to reflect its domain-specific focus. As aesthetic quality is less relevant for medical videos, we omit this dimension from our evaluation. (2) Human Evaluation. Besides the automatic metrics, we also use human evaluation,

4https://huggingface.co/tencent/HunyuanVideo

Ours Win Both Loss Ours Loss

| |64<br><br>83<br><br>94<br><br>1|111<br><br>06<br><br>37<br><br>32|33<br><br>21<br><br>27<br><br>9|9<br><br>85<br><br>73<br><br>68<br><br>67|
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |76<br><br>83<br><br>92<br><br>10|114<br><br>1|65<br><br>72<br><br>67<br><br>57<br><br>63|59<br><br>45<br><br>41<br><br>29<br><br>36|
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |44<br><br>88<br><br>77<br><br>1<br><br>84|07<br><br>55|46<br><br>59<br><br>51<br><br>48<br><br>10|1<br><br>66<br><br>64<br><br>42<br><br>68|
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

HunyuanVideo

HunyuanVideo

HunyuanVideo

Wan2.1-T2V-14B

Wan2.1-T2V-14B

Wan2.1-T2V-14B

Pika

Pika

Pika

Hailuo

Hailuo

Hailuo

Sora

Sora

Sora

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

(a) Text Alignment

(b) Medical Accuracy

(c) Visual Quality

- Figure 6: Results of the doctor evaluators assessing the medical videos generated by MedGen and other models across three dimensions. (a) Text Alignment focuses on the consistency between the generated video and the prompt. (b) Medical Accuracy focuses on the generated video adheres to medical common sense. (c) Visual Quality focuses on the overall quality of the generated video.

- Table 3: Comparison of data augmentation performance between MedGen and HunyuanVideo in downstream supervised Tasks.

MedVidCL HyperKvasir SurgVisDom ACC. Pre. Rec. F1. ACC. Pre. Rec. F1. ACC. Pre. Rec. F1.

Dori 61.73 61.29 61.12 61.01 44.44 11.11 25.00 15.38 14.29 5.56 16.67 8.33 Dori + Dhunyuan 63.58+1.85 63.40+2.11 63.25+2.13 63.27+2.26 44.44+0.00 21.25+10.14 31.25+6.25 25.00+9.62 28.57+14.28 9.52+3.96 33.33+16.66 14.81+6.48

Dori + Dmedgen 64.81+3.08 65.61+4.32 65.33+4.21 64.75+3.74 55.56+11.12 26.79+15.68 37.50+12.50 30.68+15.30 42.86+28.57 14.29+8.73 33.33+16.66 20.00+11.67

we invited three doctors to score these models. Specifically, for medical videos generated by different models based on the same prompt, the experts assessed them across three dimensions: Text Alignment, Medical Accuracy, and Visual Quality. More details on evaluation design and implementation are provided in Appendix.

#### 4.2 Main Results

Automatic Evaluation Results Table 2 compares MedGen against a range of open-source and commercial video generation models using the Med-VBench benchmark. MedGen achieves the best performance among all open-source models. Notably, MedGen excels in Factual Consistency, Text-to-Video Alignment, and Imaging Quality, while maintaining one of the lowest warping error scores across all models. These results highlight MedGen’s strong ability to generate medically accurate and visually coherent content. Commercial systems like Sora, Pika, Kling, and Hailuo outperform most open-source models, indicating a general quality gap likely due to access to larger and more diverse training data. However, MedGen narrows this gap significantly, especially on medically relevant metrics. Interestingly, model size does not always correlate with performance in the medical domain. For example, Wan2.1-T2V1.3B achieves similar results to CogVideoX-5B, suggesting that domain adaptation and training data quality are more critical than parameter count. In summary, MedGen offers a strong open-source alternative for medical video generation, combining domain precision with competitive visual quality.

Human Evaluation Results Figure 6 shows the results of human evaluation. Compared with most video generation

models, MedGen performs excellently across three dimensions: Text Alignment, Medical Accuracy, and Visual Quality. Its outstanding performance in Medical Accuracy further validates the effectiveness of the MedVideoCap-55K. In addition, the Cohen’s Kappa coefficient among the three independent experts was above 0.75, indicating strong interrater agreement and confirming the reliability of the scoring process. More details on Cohen’s Kappa coefficient can be found in Appendix.

#### 4.3 Transferability of MedVideoCap-55K on Mochi

Table 4: Performance comparison of original vs. MedVideoCap-55K-trained Mochi-1-preview on VBench. Metrics (abbreviated) follow Table 2. ”Original” is the base model; ”Trained” is fine-tuned on MedVideoCap-55K. † indicates statistically significant improvements based on a two-tailed t-test (significance level p = 0.05).

Total ↑ WE ↓ IQ ↑ SC ↑ BC ↑ MS ↑ Original 59.84 83.50 56.38 92.51 94.60 99.08 Trained 62.77+2.93 72.00† 63.42† 94.27† 93.11 97.82

Experimental Setup. To verify the transferability of our constructed MedVideoCap-55K, we switched the base model from HunyuanVideo to the Mochi-1-preview model (Team 2024). We conducted LoRA training for Mochi-1-preview 5 using MedVideoCap-55K. The training

5https://huggingface.co/genmo/mochi-1-preview

[Figure 47]

Figure 7: Examples of MedGen-generated videos across diverse real-world medical scenarios.

setup remained consistent with the configurations used in MedGen training.

Results Table 4 shows a performance comparison between the original Mochi-1-preview model and the furthertrained Mochi-1-preview model using MedVideoCap55K. Experiments demonstrates that training on our MedVideoCap-55K can significantly enhance the medical video generation capabilities in metrics like warping wrror, image quality and subject consistency; this confirms the quality of our dataset MedVideoCap-55K. The slight decline in background consistency and motion smoothness may stem from the limited diversity of the training dataset. We hypothesize that incorporating a balanced proportion of non-medical video data could mitigate this issue while preserving performance in these metrics. Exploring this approach to enhance dataset variety and its impact on model robustness remains an avenue for future work.

### 5 Applications of MedGen

#### 5.1 Application I: Data Augmentation

We evaluate MedGen’s effectiveness as a data augmenter for downstream medical video analysis. As shown in Table 3, we use MedGen-generated videos to augment training data across three classification benchmarks: MedVidCL (Gupta, Attal, and Demner-Fushman 2023), HyperKvasir (Borgli et al. 2020), and SurgVisDom (Schoeffmann et al. 2018). Compared to using only the original annotated data (Dori) and augmentation with HunyuanVideo, incorporating MedGen-generated data leads to consistently higher gains across all metrics and datasets. Notably, MedGen improves F1 scores by up to +15.3 on HyperKvasir and +11.7 on SurgVisDom, outperforming HunyuanVideo by a clear margin. These results demonstrate MedGen’s strong potential as a high-quality, domain-relevant data source for enhancing medical video understanding tasks.

#### 5.2 Application II: Science Popularization and User Simulation

MedGen shows strong potential across a wide range of medical video generation scenarios, including surgical training,

patient education, medical animation, and remote consultation. As illustrated in Figure 7, MedGen can generate diverse and high-quality videos spanning science popularization, surgical simulation, medical imaging, educational content, and patient simulations. This versatility enables MedGen to serve as a powerful tool for Science Popularization and User Simulation, particularly in scenarios where real video data is scarce, privacy-sensitive, or costly to obtain. Its ability to produce visually coherent and medically relevant content makes it well-suited for augmenting datasets, prototyping simulations, or enhancing medical communication.

### 6 Related Work

Text-to-Video Generation. Video generation has made significant progress over the past two years. The introduction of Sora (OpenAI 2025) has sparked significant research interest in text-to-video generation. Commercial models such as Kling (Kuaishou 2025), and Hailuo (MiniMax 2025) demonstrate strong performance. Meanwhile, opensource models such as LTX-Video (HaCohen et al. 2024), CogVideoX (Yang et al. 2024) and Mochi-1 (Team 2024) enable researchers to experiment with, customize, and enhance existing frameworks. Despite their power, general video generation models face limitations in medical applications due to the scarcity of medical video data, which hinders the professionalism and practicality of generated content.

Generation Models in Medical Domain. Video generation has shown great potential in the medical field, with applications including medical concept explanation, disease simulation, and biomedical data augmentation (Li et al. 2024b). In the field of endoscopy and surgery, Endora (Li et al. 2024a) is a generative model designed as an endoscopy simulator, capable of replicating diverse endoscopic scenarios for educational purposes. MedSora (Wang et al. 2024) combines spatio-temporal Mamba modules, optical flow alignment, and frequency-compensated video VAEs to generate high-quality medical videos, improving performance on downstream classification tasks. Additionally, Surgen (Cho et al. 2024) and SurgSora (Chen et al. 2024b) can generate highly realistic surgical operation videos based

on text instructions, opening up new possibilities for surgical simulation. Unlike previous works, our goal is to develop medical video generation models that can meet a wide range of medical and healthcare needs.

Text-to-Video Datasets. Data quality and quantity are closely related to model performance. For general domains, many text-video datasets, such as OpenVid (Nan et al. 2024), Panda (Wang et al. 2020), and WebVid (Bain et al. 2022), have been proposed to advance multimodal understanding and generation. However, despite their extensive and diverse collections of text-video pairs, these datasets are not directly applicable to the medical domain. Additionally, there are several datasets related to medical videos, such as MedVidCL (Gupta, Attal, and Demner-Fushman 2023), Colonoscopy (Mesejo et al. 2016), Kvasir-Capsule (Borgli et al. 2020), and CholecTriplet (Nwoye et al. 2022), which are often utilized for performing supervised classification tasks.

### 7 Conclusion

Current models lack clinical accuracy in medical video generation, largely due to the scarcity of high-quality annotated datasets. To this end, we introduce MedVideoCap55K, a large-scale, diverse, and caption-rich medical video dataset designed to address the lack of high-quality data for medical video generation. With over 55,000 curated clips spanning a broad range of real-world medical scenarios, MedVideoCap-55K provides the foundation for training generalist medical video generation models. Built on it, our model MedGen achieves state-of-the-art performance among open-source models and rivals commercial systems across multiple benchmarks. In addition, we demonstrate its utility for data augmentation in downstream medical video analysis tasks, showing significant performance gains. Our findings highlight the importance of domain-specific datasets and tailored evaluation in advancing medical video generation. We hope MedVideoCap-55K and MedGen will provide a valuable reference for the field of medical video generation, while also driving further advancements and innovation in related research.

Ethics Statement All data are publicly available, compliant with YouTube’s terms, and exclude personal/sensitive content. Captions were auto-generated (MLLMs) and manually verified to remove inappropriate/identifiable material. The dataset is intended solely and strictly for research purposes and should not be used for non-research settings, especially in clinical practice.

### Acknowledgment

This work was supported by the Shenzhen Science and Technology Program (JCYJ20220818103001002), Shenzhen Doctoral Startup Funding (RCBS20221008093330065), Tianyuan Fund for Mathematics of National Natural Science Foundation of China (NSFC) (12326608), Shenzhen Science and Technology Program (Shenzhen Key Laboratory Grant No. ZDSYS20230626091302006), and Shenzhen Stability

Science Program 2023, Shenzhen Key Lab of Multi-Modal Cognitive Computing.

### References

Bain, M.; Nagrani, A.; Varol, G.; and Zisserman, A. 2022. Frozen in Time: A Joint Video and Image Encoder for Endto-End Retrieval. arXiv:2104.00650.

Bawa, V. S.; Singh, G.; KapingA, F.; Skarga-Bandurova, I.; Oleari, E.; Leporini, A.; Landolfo, C.; Zhao, P.; Xiang, X.; Luo, G.; et al. 2021. The saras endoscopic surgeon action detection (esad) dataset: Challenges and methods. arXiv preprint arXiv:2104.03178.

Blattmann, A.; Dockhorn, T.; Kulal, S.; Mendelevitch, D.; Kilian, M.; Lorenz, D.; Levi, Y.; English, Z.; Voleti, V.; Letts, A.; et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127.

Borgli, H.; Thambawita, V.; Smedsrud, P. H.; Hicks, S.; Jha, D.; Eskeland, S. L.; Randel, K. R.; Pogorelov, K.; Lux, M.; Nguyen, D. T. D.; et al. 2020. HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy. Scientific data, 7(1): 283.

Chen, H.; Zhang, Y.; Cun, X.; Xia, M.; Wang, X.; Weng, C.; and Shan, Y. 2024a. VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models. arXiv:2401.09047.

Chen, T.; Yang, S.; Wang, J.; Bai, L.; Ren, H.; and Zhou, L. 2024b. Surgsora: Decoupled rgbd-flow diffusion model for controllable surgical video generation. arXiv preprint arXiv:2412.14018.

Cho, J.; Schmidgall, S.; Zakka, C.; Mathur, M.; Kaur, D.; Shad, R.; and Hiesinger, W. 2024. SurGen: TextGuided Diffusion Model for Surgical Video Generation.

- arXiv:2408.14028.

Das, A.; Khan, D. Z.; Psychogyios, D.; Zhang, Y.; Hanrahan, J. G.; Vasconcelos, F.; Pang, Y.; Chen, Z.; Wu, J.; Zou, X.; et al. 2024. Pitvis-2023 challenge: Workflow recognition in videos of endoscopic pituitary surgery. arXiv preprint

- arXiv:2409.01184.

Fan, W.; Si, C.; Song, J.; Yang, Z.; He, Y.; Zhuo, L.; Huang, Z.; Dong, Z.; He, J.; Pan, D.; et al. 2025. Vchitect-2.0: Parallel Transformer for Scaling Up Video Diffusion Models. arXiv preprint arXiv:2501.08453.

Ghamsarian, N.; Taschwer, M.; Putzgruber-Adamitsch, D.; Sarny, S.; and Schoeffmann, K. 2021. Relevance detection in cataract surgery videos by spatio-temporal action localization. In 2020 25th International conference on pattern recognition (ICPR), 10720–10727. IEEE.

Gupta, D.; Attal, K.; and Demner-Fushman, D. 2023. A dataset for medical instructional video classification and question answering. Scientific Data, 10(1): 158.

HaCohen, Y.; Chiprut, N.; Brazowski, B.; Shalem, D.; Moshe, D.; Richardson, E.; Levin, E.; Shiran, G.; Zabari, N.; Gordon, O.; et al. 2024. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103.

He, X.; Jiang, D.; Zhang, G.; Ku, M.; Soni, A.; Siu, S.; Chen, H.; Chandra, A.; Jiang, Z.; Arulraj, A.; et al. 2024. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv

- preprint arXiv:2406.15252. Huang, Z.; He, Y.; Yu, J.; Zhang, F.; Si, C.; Jiang, Y.; Zhang,

- Y.; Wu, T.; Jin, Q.; Chanpaisit, N.; et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21807–21818.

Jin, Y.; Sun, Z.; Li, N.; Xu, K.; Xu, K.; Jiang, H.; Zhuang, N.; Huang, Q.; Song, Y.; Mu, Y.; and Lin, Z. 2024. Pyramidal Flow Matching for Efficient Video Generative Modeling. arXiv:2410.05954.

Kong, W.; Tian, Q.; Zhang, Z.; Min, R.; Dai, Z.; Zhou, J.; Xiong, J.; Li, X.; Wu, B.; Zhang, J.; et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603.

Kuaishou. 2025. Kling. https://klingai.kuaishou.com/. Li, C.; Liu, H.; Liu, Y.; Feng, B. Y.; Li, W.; Liu, X.; Chen,

- Z.; Shao, J.; and Yuan, Y. 2024a. Endora: Video generation models as endoscopy simulators. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 230–240. Springer.

Li, L.; Qiu, J.; Saha, A.; Li, L.; Li, P.; He, M.; Guo, Z.; and Yuan, W. 2024b. Artificial Intelligence for Biomedical Video Generation. arXiv:2411.07619.

Lin, B.; Ge, Y.; Cheng, X.; Li, Z.; Zhu, B.; Wang, S.; He, X.; Ye, Y.; Yuan, S.; Chen, L.; et al. 2024. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131.

Ma, X.; Wang, Y.; Jia, G.; Chen, X.; Liu, Z.; Li, Y.-F.; Chen, C.; and Qiao, Y. 2024. Latte: Latent Diffusion Transformer for Video Generation. arXiv preprint arXiv:2401.03048.

Merrick, L.; Xu, D.; Nuti, G.; and Campos, D. 2024. Arcticembed: Scalable, efficient, and accurate text embedding models. arXiv preprint arXiv:2405.05374.

Mesejo, P.; Pizarro, D.; Abergel, A.; Rouquette, O.; Beorchia, S.; Poincloux, L.; and Bartoli, A. 2016. Computeraided classification of gastrointestinal lesions in regular colonoscopy. IEEE transactions on medical imaging, 35(9): 2051–2063.

MiniMax. 2025. Hailuo. https://hailuoai.com/video.

Nan, K.; Xie, R.; Zhou, P.; Fan, T.; Yang, Z.; Chen, Z.; Li, X.; Yang, J.; and Tai, Y. 2024. Openvid-1m: A largescale high-quality dataset for text-to-video generation. arXiv

- preprint arXiv:2407.02371.

Nwoye, C. I.; Yu, T.; Gonzalez, C.; Seeliger, B.; Mascagni, P.; Mutter, D.; Marescaux, J.; and Padoy, N. 2022. Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos. Medical Image Analysis, 78: 102433.

OpenAI. 2025. Video generation models as world simulators. https://openai.com/sora/.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.;

et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Schoeffmann, K.; Husslein, H.; Kletz, S.; Petscharnig, S.; Muenzer, B.; and Beecks, C. 2018. Video retrieval in laparoscopic video recordings with dynamic content descriptors. Multimedia Tools and Applications, 77: 16813–16832. Sharma, A.; Yu, A.; Razavi, A.; Toor, A.; Pierson, A.; Gupta, A.; Waters, A.; van den Oord, A.; Tanis, D.; Erhan, D.; Lau, E.; Shaw, E.; Barth-Maron, G.; Shaw, G.; Zhang, H.; Nandwani, H.; Moraldo, H.; Kim, H.; Blok, I.; Bauer, J.; Donahue, J.; Chung, J.; Mathewson, K.; David, K.; Espeholt, L.; van Zee, M.; McGill, M.; Narasimhan, M.; Wang, M.; Bi´nkowski, M.; Babaeizadeh, M.; Saffar, M. T.; de Freitas, N.; Pezzotti, N.; Kindermans, P.-J.; Rane, P.; Hornung,

- R.; Riachi, R.; Villegas, R.; Qian, R.; Dieleman, S.; Zhang,
- S.; Cabi, S.; Luo, S.; Fruchter, S.; Nørly, S.; Srinivasan, S.; Pfaff, T.; Hume, T.; Verma, V.; Hua, W.; Zhu, W.; Yan, X.; Wang, X.; Kim, Y.; Du, Y.; and Chen, Y. 2024. Veo. https://deepmind.google/models/veo/. Sun, W.; You, X.; Zheng, R.; Yuan, Z.; Li, X.; He, L.; Li, Q.; and Sun, L. 2024. Bora: Biomedical generalist video generation model. arXiv preprint arXiv:2407.08944. Team, G. 2024. Mochi 1. https://github.com/genmoai/ models. Team, P. 2025a. Pika. https://pika.art/. Team, W. 2025b. Wan: Open and Advanced Large-Scale Video Generative Models. https://github.com/Wan-Video/ Wan2.1. Twinanda, A. P.; Shehata, S.; Mutter, D.; Marescaux, J.; De Mathelin, M.; and Padoy, N. 2016. Endonet: a deep architecture for recognition tasks on laparoscopic videos. IEEE transactions on medical imaging, 36(1): 86–97. Wang, J.; Yuan, H.; Chen, D.; Zhang, Y.; Wang, X.; and Zhang, S. 2023. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571. Wang, X.; Zhang, X.; Zhu, Y.; Guo, Y.; Yuan, X.; Xiang, L.; Wang, Z.; Ding, G.; Brady, D. J.; Dai, Q.; and Fang, L. 2020. PANDA: A Gigapixel-level Human-centric Video Dataset. arXiv:2003.04852. Wang, Z.; Zhang, L.; Wang, L.; Zhu, M.; and Zhang, Z.

2024. Optical flow representation alignment mamba diffusion model for medical video generation. arXiv preprint

- arXiv:2411.01647. Wu, H.; Zhang, E.; Liao, L.; Chen, C.; Hou, J. H.; Wang, A.; Sun, W. S.; Yan, Q.; and Lin, W. 2023. Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives. In International Conference on Computer Vision (ICCV). Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang, Y.; Hong, W.; Zhang, X.; Feng, G.; et al. 2024. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072. Zheng, Z.; Peng, X.; Yang, T.; Shen, C.; Li, S.; Liu, H.; Zhou, Y.; Li, T.; and You, Y. 2024. Open-sora: Democratizing efficient video production for all. arXiv preprint
- arXiv:2412.20404.

Zhou, Y.; Wang, Q.; Cai, Y.; and Yang, H. 2024. Allegro: Open the Black Box of Commercial-Level Video Generation Model. arXiv preprint arXiv:2410.15458.

Zia, A.; Bhattacharyya, K.; Liu, X.; Berniker, M.; Wang, Z.; Nespolo, R.; Kondo, S.; Kasai, S.; Hirasawa, K.; Liu, B.; et al. 2023. Surgical tool classification and localization: results and methods from the miccai 2022 surgtoolloc challenge. arXiv preprint arXiv:2305.07152.

### A Experiments

- A.1 Training Details on MedGen Table 5 shows the detailed training parameter settings used to train MedGen.

Argument Value

--seed 1024

--train batch size 32

--dataloader num workers 4 --gradient accumulation steps 8

--learning rate 5e-5 --mixed precision bf16 --checkpointing steps 100 --validation steps 1000

--validation sampling steps 50

--checkpoints total limit 100 --allow tf32 True --ema start step 0

--cfg 0.0

--ema decay 0.999

--num frames 93

--validation guidance scale ”1.0”

--shift 7

--use lora True

--lora rank 32 --lora alpha 32

Table 5: Training configuration.

| | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |

Figure 8: Comparison of training efficiency. (a): the change of loss during training. (b): training time for different training strategies.

We compared the differences in training efficiency between using LoRA and full-parameter training. Figure 8 (a) shows the variation of the loss during training with the same number of steps, comparing the use of LoRA and full-parameter fine-tuning. Figure 8 (b) demonstrates that, under the same number of training steps, LoRA required only half the GPU runtime to update its internal parameters.

- A.2 Baseline Models All tests for commercial video generation models were conducted using the latest versions available before February 10, 2025. Pika used version 2.0, Kling used version 1.6, while Hailuo adopted the video-01 version.

- A.3 Evaluation Details

To prompt all models to generate medical videos for evaluation, we designed 200 different prompts, covering a variety of medical video categories and medical scenarios. For fair comparison, we conducted inference only once and maintained the default settings for all selected models, avoiding any cherry-picking of results. Additionally, we found that although many videos generated by the models received high scores, they suffered from severe distortion, which could significantly affect the accurate delivery of information. To address this issue, we invited three human evaluators to specifically assess the degree of distortion in the videos, and the warping error score was derived by aggregating the scores from all three evaluators.

We predefined evaluation criteria for different levels of image distortion, as illustrated in Figure 9, and detailed the corresponding assessment guidelines in Table 7. Three human evaluators were invited to independently assess each video and assign a distortion level accordingly. If at least two of the evaluators rated a video as “No Distortion” or “Minor Distortion,” the medical video was considered acceptable. Conversely, if two of the evaluators rated it as “Moderate Distortion” or “Severe Distortion,” the video was deemed to exhibit an unacceptable level of distortion.

To evaluate the similarity in scoring among the three evaluators, we employed Cohen’s Kappa coefficient to quantify the inter-annotator agreement. This statistical method is widely used to assess the degree of consistency in classification tasks, especially when multiple evaluators make judgments under the same conditions. The results showed that the agreement score between Evaluator 1 and Evaluator 2 was 0.82, between Evaluator 1 and Evaluator 3 was 0.79, and between Evaluator 2 and Evaluator 3 was 0.77. Overall, these relatively high Kappa values indicate a strong level of consistency and scoring reliability among the three evaluators in the video distortion assessment task.

###### Model Model Link

Open-Source Video Generation Model CogVideoX-2B (Yang et al. 2024) https://huggingface.co/THUDM/CogVideoX-2b CogVideoX-5B (Yang et al. 2024) https://huggingface.co/THUDM/CogVideoX-5b Wan2.1-T2V-1.3B (Team 2025b) https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B OpenSora V1.2 (Zheng et al. 2024) https://huggingface.co/hpcai-tech/Open-Sora-v2 OpenSoraPlan V1.3 (Lin et al. 2024) https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0 VideoCrafter-2 (Chen et al. 2024a) https://huggingface.co/VideoCrafter/VideoCrafter2 ModelScope-1.7B (Wang et al. 2023) https://huggingface.co/ali-vilab/text-to-video-ms-1.7b Latte-1 (Ma et al. 2024) https://huggingface.co/maxin-cn/Latte-1 Vchitect-2 (Fan et al. 2025) https://huggingface.co/Vchitect/Vchitect-2.0-2B Pyramid-Flow (Jin et al. 2024) https://huggingface.co/rain1011/pyramid-flow-sd3 Allegro (Zhou et al. 2024) https://huggingface.co/rhymes-ai/Allegro Mochi-1-preview (Team 2024) https://huggingface.co/genmo/mochi-1-preview LTX-Video (HaCohen et al. 2024) https://huggingface.co/Lightricks/LTX-Video HunyuanVideo (Kong et al. 2024) https://huggingface.co/tencent/HunyuanVideo Wan2.1-T2V-14B (Team 2025b) https://huggingface.co/Wan-AI/Wan2.1-T2V-14B

Commercial Video Generation Model Sora (OpenAI 2025) https://sora.com/ Pika V2.0 (Team 2025a) https://pika.art/ Hailuo (MiniMax 2025) https://hailuoai.video/ Kling V1.6 https://klingai.com/

Table 6: Baseline Video Generation Models and Their Official Links. Table 7: Deformation Evaluation Criteria.

Label Evaluation Criteria No Distortion No deformation was observed, and the image is fully intact. Minor Distortion The distortion is not significant or is only a slight local deformation. Overall, the original form of

the characters, objects, or scenes can still be recognized, and viewers will not experience noticeable discomfort or disturbance.

Moderate Distortion The shape of the characters or objects may deviate significantly, and certain details in the image may become blurred or distorted. The distortion may affect the recognizability of some scenes, but it will not disappear entirely.

Severe Distortion The shape of the characters or scenes undergoes extreme distortion, with effects such as unrecognizable twisting, compression, stretching, or even complete distortion to the point of being unidentifiable.

##### One of 200 Prompts used for Evaluation

The short video depicts a surgical procedure taking place in an operating room. The scene is filled with medical professionals, all dressed in surgical attire, including scrubs, masks, and caps. They are gathered around a patient who is lying on an operating table, covered with sterile drapes. The focus is on the surgical team as they perform the operation, with various medical instruments and equipment visible in the background. The lighting is concentrated on the surgical area, providing a clear view of the procedure being conducted. The team appears to be working collaboratively, with some members actively engaged in the surgery while others assist or monitor the process. The overall atmosphere is one of precision and care, characteristic of a professional medical setting.

To evaluate MedGen’s performance in practical applications, we invited three doctors to participate in the assessment. The specific method was as follows: we randomly selected a model to compare with MedGen and presented the videos generated by both models—based on the same text prompts—in an anonymous manner to the doctors. The doctors judged the videos from three perspectives: Text Alignment, Medical Accuracy, and Visual Quality. For each dimension, the doctors were asked to choose the better-performing model. If they believed neither performed satisfactorily, they could select ”Both Loss”.

#### A.4 VideoScore Evaluation Result

In addition to the experiments on Med-VBench, we conducted supplementary evaluations on the VideoScore to comprehensively assess the generalization and robustness of our model, MedGen. VideoScore (He et al. 2024) is an end-to-end video gener-

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

No Distortion

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Minor Distortion

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Moderate Distortion

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Severe Distortion

Figure 9: Example video clips of distortion at different levels, with the distorted areas highlighted in red boxes.

ation evaluation benchmark whose results exhibit a high correlation with human judgment. We used version 1.1 of VideoScore. These additional experiments further validate MedGen’s competitiveness in mainstream video generation tasks, highlighting its strong modeling capabilities. The results are shown in Figure 8.

Table 8: Benchmarking video generation models on VideoScore. Within each segment, bold highlights the best scores. The warping error score was derived by aggregating the scores from all three evaluators.

Model Total ↑ Visual Quality

Warping Error ↓ Open-Source Video Generation Models

Temporal Consistency ↑

Text-to-Video Alignment ↑

Factual Consistency ↑

↑

CogVideoX-2B 2.21 3.35 3.21 3.13 3.09 88.00 CogVideoX-5B 2.22 3.30 3.13 3.07 3.03 80.00 Wan2.1-T2V-1.3B 2.12 3.11 2.89 2.99 2.86 77.50 OpenSora (v1.2) 2.03 3.21 3.04 2.97 2.93 99.50 OpenSoraPlan (v1.3) 2.21 3.39 3.29 3.13 3.18 93.03 VideoCrafter-2 1.88 2.91 2.78 2.80 2.64 97.00 ModelScope-1.7B 1.82 2.86 2.72 2.77 2.58 100.00 Latte-1 1.95 3.08 2.87 2.91 2.76 97.50 Vchitect-2 1.88 2.94 2.80 2.82 2.67 99.50 Pyramid-Flow 1.84 2.87 2.74 2.76 2.60 98.50 Allegro 2.09 3.28 3.12 2.82 3.06 93.50 Mochi-1-preview 2.02 3.03 2.84 2.84 2.73 83.50 LTX-Video 1.90 3.01 2.83 2.83 2.71 100.00 HunyuanVideo 2.35 3.16 3.01 2.82 2.91 45.00 Wan2.1-T2V-14B 2.25 2.99 2.77 2.93 2.68 50.00 MedGen (Ours) 2.57 3.35 3.30 3.09 3.22 38.50

Proprietary Video Generation Models

Pika (v2.0) 2.19 2.80 2.85 2.53 2.63 42.29 Hailuo (video-01) 2.27 2.99 2.94 2.63 2.78 42.27 Kling (v1.6) 2.80 3.55 3.56 3.42 3.86 26.70 Sora 2.96 3.97 3.85 3.24 3.82 28.40

#### A.5 Inter-Rater Reliability Among Expert Evaluators

We used Cohen’s Kappa coefficient to measure agreement among the three medical experts. Table 9 indicate strong agreement among the experts, confirming the reliability of the scoring process.

|Comparison Pair|Cohen’s Kappa|
|---|---|
|Expert1 vs. Expert2<br>Expert1 vs. Expert3 Expert2 vs. Expert3<br>|0.83 0.81 0.78<br><br>|

Table 9: Comparison of Cohen’s Kappa values between experts.

|Model|Clinical Practice<br><br>|Animation<br><br>|Popular Science|Imaging|
|---|---|---|---|---|
|HunyuanVideo<br><br>+ trained on 10k teaching videos<br><br>|62.53 64.30 (+1.77)<br><br>|65.19 62.11 (-3.08)|72.33 75.01 (+2.68)<br><br>|55.17 58.15 (+2.98)|

Table 10: Med-VBench results of HunyuanVideo on unseen video types.

#### A.6 The generalizability of MedGen

To evaluate MedGen’s generalization ability on unseen video types, we trained the HunyuanVideo model using a subset of 10k medical teaching videos extracted from MedVideoCap-55K. Subsequently, we evaluated its performance on various unseen medical video categories using the VBench. The results are summarized in the Table 10.

The results indicate that even when trained on unrelated datasets, the model can still demonstrate strong generalization performance on unseen data.

