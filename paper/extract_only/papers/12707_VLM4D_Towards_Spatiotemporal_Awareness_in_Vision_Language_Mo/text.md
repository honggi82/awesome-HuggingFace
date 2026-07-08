# arXiv:2508.02095v2[cs.CV]6Aug2025

## VLM4D: Towards Spatiotemporal Awareness in Vision Language Models

Shijie Zhou1* Alexander Vilesov1* Xuehai He2,3* Ziyu Wan2 Shuwang Zhang1 Aditya Nagachandra1 Di Chang4 Dongdong Chen2 Xin Eric Wang3 Achuta Kadambi1

1UCLA 2Microsoft 3UCSC 4USC

https://vlm4d.github.io/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

  o   h         s    s     v     n wh          on  s  h        ov n  

w       a  n   ha   n  D   h     n 

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

<     <          h   D   a      a     

[Figure 10]

[Figure 11]

Ob          h   a      n   h   h   a      n   h 

H  an

Figure 1. Spatiotemporal (4D) Awareness. Humans intuitively reason in 4D (3D space + time), effortlessly reconstructing the dynamic spatial trajectory of moving objects from any perspective. In contrast, current Vision Language Models (VLMs) typically rely on aggregating 2D visual features across time, leading to incorrect predictions when motion understanding and interpretation requires deeper spatiotemporal reasoning. In this example, humans correctly perceive the car moving to the right, while the VLM (GPT-4o) inaccurately predicts leftward movement, suggesting VLMs struggle to perform spatiotemporal reasoning.

#### Abstract

Vision language models (VLMs) have shown remarkable capabilities in integrating linguistic and visual reasoning but remain fundamentally limited in understanding dynamic spatiotemporal interactions. Humans effortlessly track and reason about object movements, rotations, and perspective shifts-abilities essential for robust dynamic real-world understanding yet notably lacking in current VLMs. In this paper, we introduce VLM4D, the first benchmark specifically designed to evaluate the spatiotemporal reasoning capa-

*Equal contribution.

bilities of VLMs. Our benchmark comprises diverse realworld and synthetic videos accompanied by carefully curated question-answer pairs emphasizing translational and rotational motions, perspective awareness, and motion continuity. Through comprehensive evaluations of state-of-theart open and closed-source VLMs, we identify significant performance gaps compared to human baselines, highlighting fundamental deficiencies in existing models. Extensive analysis reveals that VLMs struggle particularly with integrating multiple visual cues and maintaining temporal coherence. We further explore promising directions, such as leveraging 4D feature field reconstruction and targeted

spatiotemporal supervised fine-tuning, demonstrating their effectiveness in enhancing spatiotemporal comprehension. Our work aims to encourage deeper exploration into improving VLMs’ spatial and temporal grounding, paving the way towards more capable and reliable visual intelligence for dynamic environments.

#### 1. Introduction

Humans possess an innate ability to perceive, track, and interpret motion as well as spatial and temporal changes, enabling rich interpretations of complex dynamic events from both egocentric and allocentric perspectives [10, 21, 59]. When observing an object move, we can inherently process any changes such as lateral shifts, rotational directions, and periodic or repeated actions unfolding along a specific trajectory [10]. These sophisticated perceptual abilities are a product of our spatiotemporal cognition [28], and form an essential foundation that allows us to comprehend and reason about physical phenomena, object interactions, and causal relationships within our environment [36, 42].

Vision language models (VLMs), which also have the potential to perceive motion and spatiotemporal changes in videos, represent a prominent class of methods aimed at emulating or surpassing human capabilities in integrated visual and linguistic reasoning [23, 41]. While previous work on VLMs has primarily focused on static visual understanding—through large-scale training on paired language and image data [68]—or explored video understanding tasks such as captioning [57] and scene understanding [13], we find that their strong performance in these tasks does not naturally translate into robust spatiotemporal reasoning. This limitation is particularly striking given that state-of-the-art VLMs are typically trained on datasets comprising hundreds of billions of tokens [53]. In contrast, human infants naturally develop robust spatiotemporal cognition within the first few months of life [71].

Another key challenge that limits VLM performance on spatiotemporal tasks is the need to implicitly or explicitly reconstruct a four-dimensional (4D) representation—3D space + time—of dynamic scenes, and subsequently reason over this reconstruction [80]. As illustrated in Fig. 1, the car is advancing forwards and turning to the left in its own frame of reference. However, from the camera’s perspective, its motion appears as a combination of heading to the right and receding into the distance despite the car being in the center of the frame due to camera rotation. Human observers can seamlessly disentangle these complex dynamics, accurately interpreting trajectories by synthesizing diverse visual cues including camera rotation compensation, stationary scene landmarks, prior knowledge of 3D and 4D environmental structures, and perspective projections [10, 28, 42, 59]. The inability of current VLMs to in-

37.5% Exo-Centric (DAVIS, YTS) 37.5% Ego-Centric (Ego4D) 25% Synthetic (Cosmos)

######     D

55% Translational

19% Rotational

17% Counting

9% False Positives

Figure 2. Distribution of Dataset Sources and Annotations. Overview of the dataset composition, illustrating the proportions of real third-person (exo-centric) videos (DAVIS [66], YouTubeVOS [88]), real first-person (ego-centric) videos (Ego4D [32]), and synthetic videos (Cosmos [4]). The real video data is further categorized by annotation types, including translational, rotational, action, counting, and false positive queries (targeting nonexistent events to assess critical reasoning).

tegrate spatial, temporal, and semantic cues in a human-like manner stems from their fundamentally different encoding paradigms for visual and language information. This discrepancy highlights a significant gap between human and machine spatiotemporal understanding, suggesting that future VLMs may benefit from insights in cognitive science and neuroscience to develop more advanced mechanisms for perceiving, integrating, reconstructing, and reasoning over dynamic scenes.

In order to effectively characterize and challenge the existing spatiotemporal reasoning abilities of VLMs, we introduce VLM4D, a rigorous benchmark specifically designed to probe the spatiotemporal grounding capabilities of current vision language models. Through this contribution, we aim to catalyze research that addresses the critical gap in spatiotemporal understanding and reasoning within VLMs and provide a foundational analysis highlighting key deficiencies in existing models.

We summarize our main contributions as follows:

- 1. We propose the first benchmark VLM4D explicitly designed to rigorously evaluate the spatiotemporal (4D) reasoning capabilities of Vision Language Models (VLMs).
- 2. We introduce a novel, meticulously curated dataset consisting of diverse real-world and synthetic video sequences paired with carefully crafted spatiotemporal question-answer (QA) annotations.
- 3. We analyze the critical limitations of contemporary VLMs in spatiotemporal reasoning and conduct experiments exploring potential solutions, highlighting fundamental challenges and outlining clear directions for impactful future research.

Human-in-the-loop Quality Review

QA Pairs VLM4D-Bench Generation

Dataset Collection

###### Dataset VLM Score

Real Videos Synthetic Generation

Human Annotated QA

Ambiguous Videos Ambiguous QA Misannotations Temporal Alignment

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Real: Synth:

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

LLM Gen. MC

Total:

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

- Figure 3. Dataset Generation and Annotation Pipeline. Our dataset was constructed by collecting real videos and generating synthetic data, followed by human-in-the-loop quality reviews to address ambiguous videos and annotations. After temporal alignment and quality assurance, human-annotated questions and ground-truth answers were created, complemented by multiple-choice (MC) answers generated by large language models (LLMs). The final dataset includes real and synthetic video data with comprehensive VLM scoring metrics.

#### 2. Related Work

Spatiotemporal Understanding in Vision Language Models Early methods for video understanding before the advent of large vision language models (VLMs) leveraged trajectory-based representations and separate modeling of spatial and motion cues for tasks like action and motion recognition [11, 40, 70, 77]. Recently, VLMs have evolved rapidly by fully leveraging the significant achievements of Large Language Models (LLMs) [7, 9, 22, 67, 74, 83] and large-scale visual instruction tuning datasets [20, 52, 103]. While VLMs [3, 31, 35, 44, 52, 73, 79, 103] exhibit transformative potential for applications such as embodied AI [24, 38, 72], robotics [64, 76], scene generation [51, 62, 101], and world modeling [54, 96], most existing methods remain constrained to static images, focusing narrowly on spatial understanding [12, 17, 27, 69] while overlooking the dynamic temporal dimension inherent in real-world interactions. To bridge this gap, emerging research [18, 46, 58, 94, 95] has begun exploring video modality integration, aiming to equip VLMs with spatialtemporal awareness critical for tasks like video comprehension, where both contextual details and motion dynamics are essential. For example, VideoLLM-MoD [86] proposes to address the efficiency issue when processing long-term video by mixture-of-depths. [91] introduces VideoRefer to enhance the finer-level (like object-level) spatial-temporal video understanding of VLMs. Grounded-VideoLLM [78] also targets for fine-grained video understanding through incorporating an additional temporal stream. In this work, we aim to rigorously evaluate the 4D spatial-temporal reasoning capabilities of state-of-the-art VLMs, probing how and to what extent these models internalize spatial intelligence and temporal dependencies.

VLM Benchmarks Following the development trends of VLMs, benchmarking VLMs shares the similar trajectory by first evaluating vision QA on static images [33, 43, 55, 92], to align with models’ early focus on 2D understanding. As VLMs evolved to tackle dynamic scenarios, benchmarks expanded to evaluate general-purpose video comprehension

tasks that probe temporal coherence and event understanding [29, 37, 48, 50, 61]. Notably, MMVU [98] further proposes a knowledge-intensive benchmark to assess the expert-level reasoning ability of current video-based large models. However, while these works assess perception and semantic understanding, they largely overlook the explicit evaluation of spatial-temporal awareness, a core capability for real-world applications requiring 4D (3D space + time) reasoning. Recent efforts like [90] pioneer benchmarks for 3D visual-spatial intelligence but restrict evaluation to static 3D scene, neglecting the interplay of object motion and temporal dynamics intrinsic to videos. In this work, we introduce VLM4D, the first benchmark designed to holistically evaluate the 4D intelligence in VLMs, unifying spatial understanding, temporal continuity, and motion reasoning. By curating tasks that demand precise analysis of dynamic interactions (e.g., direction prediction, perspective anticipation, and motion reasoning), VLM4D exposes critical gaps in current models’ ability to internalize spatiotemporal relationships. Our work not only advances the granularity of VLM evaluation but also shares insights and potential solutions to improve the model performance.

#### 3. The VLM4D Benchmark

We introduce VLM4D, the first benchmark specifically designed to test the spatiotemporal reasoning ability of VLMs. VLM4D consists of 1000 videos paired with over 1800 question-answer pairs, each carefully designed to assess both spatial and temporal understanding jointly. The majority of these videos are sourced from datasets with rich spatiotemporal characteristics, thus ensuring a diverse range of motion-related scenarios. We also augment the dataset with synthetic videos generated by a world-foundation model, Cosmos [4], that has been modified using techniques introduced in [34] to obtain more accurate correspondence between motion-oriented prompts and the resulting generated video. Fig. 2 illustrates the composition of our dataset.

##### 3.1. Benchmark Construction

Unlike prior work that often relies heavily on LLMs and VLMs to generate first iterations of benchmarks and

Is the camel in the foreground turning to the left or right from its own perspective? A. Not moving B. Left C. Moving straight, not turning D. Right

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

In which direction of rotation does the person pour the batter into the frying pan? A. Counter-clockwise B. Left to right C. Right to left D. Clockwise

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

What direction is the robotic dog moving towards? A. Right B. no robotic dog there C. Not moving

D. Left

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- Figure 4. Qualitative Examples of Dataset Annotations. (Top) A third-person (exo-centric) video with translational annotations (“camel turning left from its perspective”). (Middle) A first-person (ego-centric) video with a rotational question (“clockwise rotation of ladle”). (Bottom) A synthetic scene with motion recognition “robotic dog moving left”).

datasets [15] followed by human quality control, we found that existing VLMs and automated methods showed significant limitations in terms of reliability and quality. This shortcoming necessitated direct human annotations that were then followed by augmentation by LLMs to ensure a high-quality benchmark. An overview of the benchmark curation pipeline is shown in Fig. 3.

Real Video Data Collection Real-world videos were sourced from datasets with rich spatiotemporal characteristics that ensured diverse motion and perspective variations. For egocentric data, we relied mainly on the Ego4D dataset [32], while most exocentric data points were collected from the DAVIS [66] and YouTube-VOS [88] datasets. To minimize confounders and to focus attention of VLM abilities to only spatiotemporal reasoning, we preprocessed the videos by temporally segmenting and centering them around the most relevant action, thus resulting in videos with an average duration of 3-8 seconds. This en-

sures that the key event described in the question is clear and reduces ambiguities or confounders that would reduce VLM accuracy.

Synthetic Video Data Generation The rapid advancement of video generation techniques has led to their widespread application in diverse domains like robotics and immersive entertainment [4, 8, 49, 75, 85], making 4D reasoning on synthetic content essential for VLMs and underscoring the need to include such videos in our benchmark. For synthetic video generation, we use Cosmos [4] as our video generation backbone. To ensure that the generated videos align with the intended object moving directions, we incorporate input bounding boxes as additional spatial guidance. Specifically, we follow the approach introduced in [34] modifying the diffusion forward steps to enforce object localization constraints at each timestep, ensuring consistency between the generated object direction and the user-specified trajectory. The average duration of generated

synthetic videos is 5 seconds. To maintain high-quality outputs, we perform a manual verification step after generation, filtering out low-quality videos and retaining only those that accurately match the specified directions. Once a video is generated, we use an LLM (GPT-4o) to create two types of evaluation questions: Directional questions, derived from the textual prompt used to generate the video; and False Positive (FP) questions (constituting 10%, matching the ratio in the real dataset), which query non-existent objects in the scene. Both question types follow the format: “What direction is the ⟨Object Name⟩ moving?”, where the model must select one of four possible answers: “left”, “right”, “not moving”, or “no ⟨Object Name⟩ there”. A final manual review is conducted to filter out or revise ambiguous questions, ensuring the quality of the questions and ground-truth answers.

QA Generation and Quality Control Question-answer pairs are primarily constructed through human annotations. The question answer pairs are then supplemented with alternative answers by an LLM (GPT-4o) for multiple choice (MC) questions. To ensure high-quality annotations, we applied a three-round, multi-person cross-checked verification process, during which ambiguous videos were filtered out and vague, misleading, or incorrect QA pairs were refined to improve spatial and temporal alignment between language and visual content. Fig. 4 showcases some qualitative examples of annotations for different types of videos.

Assessing Human Performance To establish a human performance baseline on our benchmark, we conducted an evaluation in which participants independently answered 100 randomly sampled questions from the dataset. The accuracy of human responses was then aggregated to approximate the performance of human spatiotemporal reasoning on the dataset.

##### 3.2. Categorizing Spatiotemporal Performance

To systematically evaluate spatiotemporal reasoning capabilities, we first categorize videos into two primary groups: egocentric (first-person) videos and exocentric (third-person) videos. Egocentric videos are sourced from the Ego4D [32] dataset, where scenes are captured from a head-mounted camera, thus offering dynamic video data that is inherently coupled with the individual’s actions. Exocentric videos encompass a diverse range of recorded scenes, from sports footage to everyday scenes. Beyond this categorization, we also evaluate spatiotemporal performance across four dimensions: translational movement (TM), rotational movement (RM), spatiotemporal counting (STM), and false positives (FP), with their proportions shown in Fig. 2. Translational movement assesses a model’s ability to track linear motion within scenes, while rotational

[Figure 42]

Figure 5. Comparison of accuracy across types of spatiotemporal questions. Model accuracy is shown only for the six topperforming VLMs.

movement assesses the understanding of changes in orientation and perspective shifts over time. Spatiotemporal counting extends these core motion-based tasks by requiring a more complex reasoning strategy to determine the number of objects performing a translation or rotational movement. Lastly, the false positives category evaluates the model’s critical thinking [25] in determining whether an object or event actually occurred within the spatiotemporal context. By structuring the benchmark along these axes, we aim for a comprehensive framework for assessing spatiotemporal reasoning (Fig. 5).

#### 4. Evaluation of VLM4D Benchmark

##### 4.1. Evaluation Setup

Benchmark Models We evaluate 23 most recently released VLMs thus covering a wide range of model sizes, architectures, and training methodologies. For closedsource VLMs, we evaluate GPT-4o [63], Gemini 2.5 Pro [30], Claude Sonnet 4 [6], and Grok-2-Vision [87]. For open-source models, we include Llama 4 [60], DeepSeek-VL [56], Qwen2.5-VL [89], Qwen2-VL [79], InternVL2.5 [16], Aria [45], InternVideo 2.5 [82], InternVideo2 [81], Phi-4-multimodal [2], Phi-3.5-vision [1], Pixtral [5], VideoLLama3 [94], Llava-One-Vision [44], LlavaNeXT-Video [97]. When available, we evaluate different sizes for each model, resulting in models ranging from 2 to 72 billion parameters.

Evaluation Settings The evaluations were performed in a zero-shot setting with video or a set of sampled frames of

Organization Model Release Real Synthetic Overall Ego-centric Exo-centric Average Directional FP Average

User Study Human Performance 99.6 99.7 99.7 95.8 100 96.2 98.8 Random Random Selection 24.4 23.2 23.6 25.5 24.7 25.4 24.1

###### Latest Proprietary VLMs

OpenAI GPT-4o 2024-11 55.5 62.2 60.0 49.5 53.3 49.9 57.5 Google Gemini-2.5-Pro 2025-6 64.6 62.9 63.5 54.8 80.0 57.3 62.0 Anthropic Claude-Sonnet-4 2025-5 52.6 52.1 52.2 44.0 86.7 48.3 51.3 xAI Grok-2-Vision 2024-12 48.8 49.7 49.4 49.3 66.7 51.0 49.8

Open-source Image VLMs Meta Llama-4-Maverick-17B 2025-4 52.6 54.3 53.8 53.3 51.1 53.0 53.6

Llama-4-Scout-17B 2025-4 48.6 56.2 53.7 53.3 75.6 55.5 54.1 Microsoft Phi-4-Multimodal 2025-3 41.0 35.4 37.2 37.5 11.1 34.8 36.6

Phi-3.5-Vision 2024-7 33.4 38.8 37.1 23.3 37.8 24.7 34.0 DeepSeek DeepSeek-VL2 2024-12 33.6 32.9 33.1 31.8 46.7 33.3 33.2 Shanghai AI Lab InternVL2.5-38B 2024-11 46.6 50.1 48.9 43.3 57.8 44.7 47.9

InternVL2.5-8B 2024-11 39.0 44.0 42.4 40.8 42.2 40.9 42.0 Mistral AI Pixtral-12B 2024-9 32.3 25.8 27.9 24.3 22.2 24.0 27.0 Rhymes Aria 2024-11 47.2 44.0 45.1 38.5 71.1 41.8 44.3

###### Open-source Video VLMs

Alibaba Qwen2.5-VL-7B 2025-1 42.3 43.7 43.3 43.5 64.4 45.6 43.8 Qwen2.5-VL-72B 2025-1 54.3 52.5 53.1 49.5 80.0 52.6 53.0 Qwen2-VL-7B 2024-8 36.1 34.7 35.2 40.5 35.6 40.0 36.3 Qwen2-VL-72B 2024-9 48.1 43.0 44.6 40.8 73.3 44.0 44.5

DAMO VideoLLama3-2B 2025-1 53.2 42.5 46.0 34.3 55.6 36.4 43.7

VideoLLama3-7B 2025-1 49.4 45.1 46.5 42.8 53.3 43.8 45.9 Shanghai AI Lab InternVideo2.5-8B 2025-1 57.2 50.5 52.7 44.3 46.7 44.5 50.7

InternVideo2-8B 2024-8 35.6 39.3 38.1 43.0 0.0 38.7 38.2 LLaVA LLaVA-One-Vision-7B 2024-9 36.8 35.6 36.0 37.8 35.6 37.5 36.3 LLaVA-NeXT-Video-34B 2024-6 29.6 31.6 30.9 24.5 55.6 27.6 30.1

Table 1. Evaluation on VLM4D Benchmark across various proprietary and open-source VLMs. Top three performers in each category are highlighted from dark (highest) to light (third highest). Human and random selection baselines are included for reference.

video, followed by the prompt forming the input. For each model, we evaluate on two different inference settings. In the first setting, the model prompted to directly output (DO) the answer immediately without any reasoning, and in the second evaluation setting, the model is directed to create intermediate reasoning steps, chain-of-thought (CoT) [84], before inferring the final answer.

Metrics Following prior work [90] and given the nature of our target task, we adopt multiple-choice questions (MCQs) for evaluation, using accuracy as the primary metric. For the two inference settings described earlier, we employ LLMas-Judge [98] to assess the outputs of VLMs. We opt for this method instead of string or template matching, as VLMs—especially under chain of thought (CoT) prompt-

ing—often generate all possible answer choices during reasoning, with varying frequencies and slight formatting differences. In some cases, the final answer may even contradict the reasoning. To better evaluate whether the model truly understands the video, we prompt two advanced LLMs (GPT-o3 and o4-mini) to grade based on the full CoT reasoning, not just the final answer. We then perform a crosscheck between their judgments and manually resolve any disagreements. The evaluation results are reported in Tab. 1.

##### 4.2. Benchmark Results

Our comprehensive evaluation on the VLM4D benchmark, detailed in Tab. 1, systematically assesses the spatiotemporal reasoning capabilities of modern Vision-Language Models (VLMs). The results highlight a clear hierar-

chy, with proprietary models demonstrating superior performance over their open-source counterparts. Google’s Gemini-2.5-Pro emerges as the top-performing model with an overall accuracy of 62.0%, followed by OpenAI’s GPT4o at 57.5%. Within the open-source domain, image-based models like Meta’s Llama-4-Scout-17B (54.1%) and videosupported models like Alibaba’s Qwen2.5-VL-72B (53.0%) demonstrate highly competitive results, even outperforming some proprietary counterparts. Despite these achievements, a significant performance gap persists when compared to human accuracy (98.8%), underscoring that sophisticated 4D awareness remains a formidable challenge for AI. Notably, performance varies significantly across different categories, such as real versus synthetic data and egocentric versus exo-centric perspectives, indicating that current models lack generalized spatiotemporal understanding.

#### 5. Analysis: Why VLMs Don’t Work Well?

##### 5.1. Limited Spatiotemporal Cognition

Despite significant advances in VLMs, their abilities to understand and reason about motion, spatial relationships, and temporal coherence remains fundamentally underdeveloped [14, 104]. Chain of thought (CoT) [84] is widely employed as a method to improve accuracy through step-by-step reasoning. We showcase a comparison between CoT and DO in Fig. 6. Overall, there is no indication of a large advantage of CoT over all evaluated models. Upon deeper exploration of the CoT reasoning of some models, we observe that the reasoning process was primarily flawed in the following ways: irrelevant information and arriving at conclusions that are inconsistent with the reasoning process. Larger models exhibited strategies that would be similar to how a human processes spatiotemporal information, but the resulting execution falls short of human performance. This demonstrates a disconnect between its visual and linguistic knowledge. We provide examples of this behavior in the supplementary material.

##### 5.2. Deficiencies in Spatiotemporal Labeling

Another avenue of exploration we undertook is to understand the richness of spatiotemporal labels in popular supervised fine-tuning (SFT) VLM datasets. Typically, video captioning occurs at the ‘scene’ level, lacking fine-grained temporal, spatial, and object-level details. We performed an extensive analysis, encompassing over 2 million samples [15, 19, 46, 47, 97]. We performed this analysis through string-matching of spatiotemporal descriptors related to directionality, translational motion, rotation, and perspective shifts and provide the overall results in Fig. 7. We then performed a manual finegrained evaluation of the ShareGPT4Video dataset [15] which we found had the highest density of spatiotemporal dataset. We found that

[Figure 43]

- Figure 6. Comparison of CoT and DO Accuracy Across Models. Accuracy comparison between Chain-of-Thought (CoT) and Direct Output (DO) prompting across VLMs.

[Figure 44]

- Figure 7. Heatmap of Occurances of Spatial-Temporal Terms in popular video SFT datasets.

from a sample of 100 labels that were detected as spatiotemporal, less than 10% of them were judged as accurate upon human evaluation. This result underscores the inadequacy of current dense captioning approaches, which frequently generate spatiotemporal descriptors without capturing precise motion dynamics. We provide more detailed analysis and explanations in the supplementary material.

#### 6. Probing Future Solutions

To probe promising future solutions for enhancing spatiotemporal video understanding, we propose two approaches that address some of the shortcomings of current

state-of-the-art VLMs: fine-tuning a VLM on data-rich in spatiotemporal actions and the other leveraging 4D reconstruction and feature fields jointly with a VLM. SFT refines the model’s abilities by training on datasets that contain temporally and spatially rich actions and interactions. By integrating structured visual representations and targeted fine-tuning, these approaches enhance video-language models’ ability to interpret motion. The second method lifts the feature space of VLMs into a temporally coherent 4D feature field, providing structured scene representations that improve motion and spatial reasoning in the stage of decoding and inference.

Spatial-Temporal SFT We evaluate on a subset split of the real dataset by randomly splitting the real-world dataset into a training and testing split (80% / 20%) and we try settings using synthetic/real/both for training. We conducted the experiments using Qwen 2VL (7B) and Qwen 2.5VL (7B) through LLama-Factory [99], and compared the performance before and after supervised fine-tuning in Tab. 2. The results demonstrated an improvement in accuracy in spatiotemporal reasoning, suggesting that performance gains can be obtained through targeted training. However, the addition of synthetic data does not necessarily increase performance over using real data alone, suggesting the importance of synthetic data quality.

4D Feature Fields Reconstruction Recent advances in 3D/4D feature fields reconstruction methods [26, 39, 93, 100, 102] have significantly enhanced the vision foundation model’s performance in 3D/4D space by integrating structured latent scene representations into the model’s inference stage. Inspired by the promising results of feature lifting, we explore enhancing the InternVideo2-8B model [81] with spatiotemporal awareness by adopting the strategy proposed in Feature4X [102], which constructs the VLM’s 2D feature space along the time dimension into a 4D feature field. To assess this approach, we evaluate performance on a subset of the VLM4D benchmark, specifically leveraging all 50 videos from the DAVIS 2016 dataset [65]. Our experimental evaluation compares inference performance across three distinct input modalities: original 2D videos; rendered novel global-view videos (which provide broader contextual information in 2D format); and reconstructed global feature fields (which implicitly incorporate 4D scene-level information during reasoning). Table 3 reveals that reconstructed feature fields achieve the highest accuracy across both reasoning types. This success stems from two key advantages: the inherent structure of 4D representations and the ability of feature field inference to avoid the rendering artifacts of RGB reconstruction (global view video). However, the current approach requires per-scene optimization as a post-processing step, limiting its generalizability and

###### Model MC

| | |
|---|---|
|Original Model Qwen 2VL (7B) Qwen 2.5VL (7B)|38.3 43.4<br><br>|
|Finetuned Model<br><br>Qwen 2VL (7B) (R)<br>Qwen 2VL (7B) (S) Qwen 2VL (7B) (R+S)<br><br><br>Qwen 2.5VL (7B) (R)<br>Qwen 2.5VL (7B) (S) Qwen 2.5VL (7B) (R+S)<br>|54.5 42.5 55.5 55.6 48.6 56.3|

- Table 2. SFT on Spatial-Temporal Datasets. MC refers to the performance of multiple-choice, while R, S, and R+S denote real, synthetic, and both real+synthetic usage of data for fine-tuning.

Input Modality Accuracy

| | |
|---|---|
|Chain of Thought Response Original 2D Video Global View Video Global Feature Field|36.0 32.7 37.4|
|Direct Output Response Original 2D Video Global View Video Global Feature Field|24.3 23.8 29.0|

- Table 3. InternVideo2 Accuracy with 4D Feature Field Reconstruction. Comparison of InternVideo2’s performance given different input modalities from the same subset.

making it computationally intensive.

#### 7. Conclusion

Through the construction of the VLM4D benchmark, we evaluate the spatiotemporal reasoning capabilities of various vision language models (both open-source and proprietary). While more recently released models demonstrate improved performance over their counterparts, they remain significantly behind human proficiency. Overall, our work questions whether VLMs possess spatiotemporal reasoning abilities that are imperative to have for more sophisticated visual agents in fields ranging from robotics to interactive AI systems that require a deep understanding of dynamic visual environments. We hope to inspire future work to explore novel approaches for integrating spatiotemporal grounding, thereby enhancing their spatiotemporal reasoning capabilities and facilitating robust deployment.

#### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219,

2024. 5

- [2] Marah Abdin, Jyoti Aneja, Harkirat Behl, S´ebastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024. 5
- [3] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, and Xiren Zhou. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras, 2025. 3
- [4] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2, 3, 4
- [5] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024. 5
- [6] Anthropic. System card: Claude opus 4 & claude sonnet 4. Technical report, 2025. 5
- [7] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3
- [8] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 4

- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.

- Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3
- [10] Neil Burgess. Spatial memory: how egocentric and allocentric combine. Trends in Cognitive Sciences, 10(12):551– 557, 2006. 2
- [11] Pradyumna Chari, Yunhao Ba, Shijie Zhou, Chinmay Talegaonkar, Shreeram Athreya, and Achuta Kadambi. On learning mechanical laws of motion from video using neural networks. IEEE Access, 11:30129–30145, 2023. 3
- [12] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465,

2024. 3

- [13] Jun Chen, Han Guo, Kai Yi, Boyang Li, and Mohamed Elhoseiny. Visualgpt: Data-efficient adaptation of pretrained language models for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18030–18040, 2022. 2
- [14] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 7
- [15] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 4, 7
- [16] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 5
- [17] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024. 3
- [18] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 3
- [19] Erfei Cui, Yinan He, Zheng Ma, Zhe Chen, Hao Tian, Weiyun Wang, Kunchang Li, Yi Wang, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Yali Wang, Limin Wang, Yu Qiao, and Jifeng Dai. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o, 2024. 7
- [20] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 3

- [21] Julian De Freitas, Nicholas E. Myers, and Anna C. Nobre. Tracking the changing feature of a moving object. Journal of Vision, 16(3):22, 2016. 2
- [22] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019. 3
- [23] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 2
- [24] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. 2023. 3
- [25] Chenrui Fan, Ming Li, Lichao Sun, and Tianyi Zhou. Missing premise exacerbates overthinking: Are reasoning models losing critical thinking skill? arXiv preprint arXiv:2504.06514, 2025. 5
- [26] Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, et al. Large spatial model: End-to-end unposed images to semantic 3d. Advances in neural information processing systems, 37:40212–40229,

2024. 8

- [27] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025. 3
- [28] Jennifer J. Freyd and Ronald A. Finke. Representational momentum. Journal of Experimental Psychology: Learning, Memory, and Cognition, 10(1):126–132, 1984. 2
- [29] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 3
- [30] Google Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Technical report, 2025. 5
- [31] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023. 3
- [32] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d:

- Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 2, 4, 5
- [33] Xuehai He, Weixi Feng, Kaizhi Zheng, Yujie Lu, Wanrong Zhu, Jiachen Li, Yue Fan, Jianfeng Wang, Linjie Li, Zhengyuan Yang, et al. Mmworld: Towards multidiscipline multi-faceted world model evaluation in videos. arXiv preprint arXiv:2406.08407, 2024. 3
- [34] Xuehai He, Shuohang Wang, Jianwei Yang, Xiaoxia Wu, Yiping Wang, Kuan Wang, Zheng Zhan, Olatunji Ruwase, Yelong Shen, and Xin Eric Wang. Mojito: Motion trajectory and intensity control for video generation. arXiv preprint arXiv: 2412.08948, 2024. 3, 4
- [35] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 3
- [36] Gunnar Johansson. Visual perception of biological motion and a model for its analysis. Perception & Psychophysics, 14(2):201–211, 1973. 2
- [37] Muhammad Uzair Khattak, Muhammad Ferjad Naeem, Jameel Hassan, Muzammal Naseer, Federico Tombari, Fahad Shahbaz Khan, and Salman Khan. How good is my video lmm? complex video reasoning and robustness evaluation suite for video-lmms. arXiv preprint arXiv:2405.03690, 2024. 3
- [38] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 3
- [39] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via feature field distillation. Advances in neural information processing systems, 35:23311–23330, 2022. 8
- [40] Ivan Laptev. On space-time interest points. International Journal of Computer Vision, 64(2-3):107–123, 2005. 3
- [41] Yann LeCun, Bernhard Boser, John S. Denker, Donnie Henderson, Richard E. Howard, Wayne Hubbard, and Lawrence D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541– 551, 1989. 2
- [42] Alan M. Leslie. Spatiotemporal continuity and the perception of causality in infants. Perception, 13(3):287–305,

1984. 2

- [43] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 3
- [44] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 3, 5
- [45] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen,

- and Junnan Li. Aria: An open multimodal native mixtureof-experts model. arXiv preprint arXiv:2410.05993, 2024. 5
- [46] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 3, 7
- [47] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark, 2023. 7
- [48] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 3
- [49] Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, Zhengzhong Tu, and Zhiwen Fan. 4k4DGen: Panoramic 4d generation at 4k resolution. In The Thirteenth International Conference on Learning Representations, 2025. 4
- [50] Xinhao Li, Zhenpeng Huang, Jing Wang, Kunchang Li, and Limin Wang. Videoeval: Comprehensive benchmark suite for low-cost evaluation of video foundation model. arXiv preprint arXiv:2407.06491, 2024. 3
- [51] Lu Ling, Chen-Hsuan Lin, Tsung-Yi Lin, Yifan Ding, Yu Zeng, Yichen Sheng, Yunhao Ge, Ming-Yu Liu, Aniket Bera, and Zhaoshuo Li. Scenethesis: A language and vision agentic framework for 3d scene generation. arXiv preprint arXiv:2505.02836, 2025. 3
- [52] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 3
- [53] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint, 2024. 2
- [54] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268,

2024. 3

- [55] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multimodal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 3
- [56] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525,

2024. 5

- [57] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Neural Information Processing Systems, 2019. 2
- [58] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed

- video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 3
- [59] D. Marr and S. Ullman. Directional selectivity and its use in early visual processing. Proceedings of the Royal Society of London. Series B, Biological Sciences, 211(1183):151– 180, 1981. 2
- [60] Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. Technical report, 2025. 5
- [61] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. arXiv preprint arXiv:2311.16103, 2023. 3
- [62] Ba¸sak Melis Ocal,¨ Maxim Tatarchenko, Sezer Karao˘glu, and Theo Gevers. Sceneteller: Language-to-3d scene generation. In European Conference on Computer Vision, pages 362–378. Springer, 2024. 3
- [63] OpenAI. Hello gpt-4o. Technical report, 2024. 5
- [64] Shivansh Patel, Xinchen Yin, Wenlong Huang, Shubham Garg, Hooshang Nayyeri, Li Fei-Fei, Svetlana Lazebnik, and Yunzhu Li. A real-to-sim-to-real approach to robotic manipulation with vlm-generated iterative keypoint rewards. arXiv preprint arXiv:2502.08643, 2025. 3
- [65] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732, 2016. 8
- [66] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 2, 4
- [67] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 3
- [68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763, 2021. 2
- [69] Kanchana Ranasinghe, Satya Narayan Shukla, Omid Poursaeed, Michael S Ryoo, and Tsung-Yu Lin. Learning to localize objects improves spatial reasoning in visual-llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12977–12987, 2024. 3
- [70] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. In Advances in Neural Information Processing Systems (NIPS), pages 568–576, 2014. 3
- [71] Elizabeth S. Spelke and Katherine D. Kinzler. Core knowledge. Developmental Science, 10(1):89–96, 2007. 2
- [72] Alessandro Suglia, Claudio Greco, Katie Baker, Jose L Part, Ioannis Papaioannou, Arash Eshghi, Ioannis Konstas,

- and Oliver Lemon. Alanavlm: A multimodal embodied ai foundation model for egocentric video understanding. arXiv preprint arXiv:2406.13807, 2024. 3
- [73] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 3
- [74] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [75] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 4
- [76] Beichen Wang, Juexiao Zhang, Shuwen Dong, Irving Fang, and Chen Feng. Vlm see, robot do: Human demo video to robot action plan via vision language model. arXiv preprint arXiv:2410.08792, 2024. 3
- [77] Heng Wang, Alexander Kl¨aser, Cordelia Schmid, and Cheng-Lin Liu. Action recognition by dense trajectories. In 2011 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3169–3176, 2011. 3
- [78] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024. 3
- [79] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 3, 5
- [80] Xingrui Wang, Wufei Ma, Angtian Wang, Shuo Chen, Adam Kortylewski, and Alan Yuille. Compositional 4d dynamic scenes understanding with physics priors for video question answering. arXiv preprint arXiv:2406.00622,

2024. 2

- [81] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024. 5, 8
- [82] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 5
- [83] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021. 3
- [84] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.

- Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 6, 7
- [85] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26057–26068,

2025. 4

- [86] Shiwei Wu, Joya Chen, Kevin Qinghong Lin, Qimeng Wang, Yan Gao, Qianli Xu, Tong Xu, Yao Hu, Enhong Chen, and Mike Zheng Shou. Videollm-mod: Efficient video-language streaming with mixture-of-depths vision computation. Advances in Neural Information Processing Systems, 37:109922–109947, 2024. 3
- [87] xAI. Grok-2 beta release. Technical report, 2024. 5
- [88] Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, and Thomas Huang. Youtube-vos: Sequence-to-sequence video object segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 585– 601, 2018. 2, 4
- [89] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 5
- [90] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024. 3, 6
- [91] Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, et al. Videorefer suite: Advancing spatial-temporal object understanding with video llm. arXiv preprint arXiv:2501.00599, 2024. 3
- [92] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024. 3
- [93] Yuanwen Yue, Anurag Das, Francis Engelmann, Siyu Tang, and Jan Eric Lenssen. Improving 2d feature representations by 3d-aware fine-tuning. In European Conference on Computer Vision, pages 57–74. Springer, 2024. 8
- [94] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 3, 5
- [95] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 3
- [96] Hongxin Zhang, Zeyuan Wang, Qiushi Lyu, Zheyuan Zhang, Sunli Chen, Tianmin Shu, Yilun Du, and Chuang Gan. Combo: compositional world models for embodied

multi-agent cooperation. arXiv preprint arXiv:2404.10775,

2024. 3

- [97] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model,

2024. 5, 7

- [98] Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, et al. Mmvu: Measuring expertlevel multi-discipline video understanding. arXiv preprint arXiv:2501.12380, 2025. 3, 6
- [99] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. 8
- [100] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676– 21685, 2024. 8
- [101] Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, and Achuta Kadambi. Dreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting. In European Conference on Computer Vision, pages 324–342. Springer, 2024. 3
- [102] Shijie Zhou, Hui Ren, Yijia Weng, Shuwang Zhang, Zhen Wang, Dejia Xu, Zhiwen Fan, Suya You, Zhangyang Wang, Leonidas Guibas, et al. Feature4x: Bridging any monocular video to 4d agentic ai with versatile gaussian feature fields. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14179–14190, 2025. 8
- [103] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3
- [104] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, et al. Apollo: An exploration of video understanding in large multimodal models. arXiv preprint arXiv:2412.10360, 2024. 7

## VLM4D: Towards Spatiotemporal Awareness in Vision Language Models Supplementary Material

###### Appendix Outline

- • Section A: Statistics of VLM4D benchmark
- • Section B: Evaluation setup
- • Section C: Analysis of Existing Video Instruction Tuning Datasets
- • Section D: Examples of VLMs’ responses with Chain-ofThought and Direct Output prompting
- • Section E: More details of 4D reconstruction using Feature4X

#### A. VLM4D Benchmark Statistics

|Dataset Statistics<br><br>|Video|QA pair|
|---|---|---|
|Real Samples Synthetic Samples Total Samples|600 400 1,000|1,371 445 1,816|

Table A. VLM4D Dataset Breakdown

Tab. A presents the breakdown of our VLM4D benchmark dataset. Additionally, Fig. A visualizes the detailed performance of VLMs across different question categories. For models that support only image input, we convert videos into multi-frame image sequences, using the maximum number of frames allowed within the model’s context window. For models that support video input, we follow their default frame rate settings, typically 1 fps.

#### B. Evaluation setup

|Task<br><br>|GPU Configuration|
|---|---|
|Model Evaluation 4D Feature Field Reconstruction|8 × A100 1 × A100|

Table B. Evaluation Type and GPU Requirements

benchmarking the spatiotemporal validity of the highly used video instruction tuning datasets.

We show our target strings in Tab. E. In our comprehensive analysis of the ShareGPT-4o dataset (Fig. E), we observed that over 40% of the captions incorporate at least one target category. As depicted in Fig. G, our target string search highlights a significant emphasis on the directional descriptors “left” and “right.” Furthermore, the analysis of negative samples, illustrated in Fig. F, indicates that these directional terms are seldom employed in conjunction with rotational or translational actions. This observation is further substantiated by the minimal overlaps between directional descriptors and action-related terms, as shown in Fig. G. Collectively, this shows a notable gap in the dataset’s ability to capture complex spatiotemporal relationships, particularly those involving dynamic textures and nuanced motion patterns. The statistics are summarized in Tab. D.

###### Dataset Count

VideoChat2-IT 423,497 ShareGPT4Video 40,178 LLaVA-178K 1,627,017 ShareGPT-4o 2,111

###### Total 2,092,803

- Table C. No. of Video Instruction Tuning Samples (QA pair)

|Caption Category|Count<br><br>|Percentage (%)|
|---|---|---|
|With Target String Without Target String|1,164,006 928,897|55.6 44.4|

- Table D. Distribution of Captions Containing Target Strings

#### C. Video Instruction Tuning Dataset Analysis

We begin by analyzing four individual collections of datasets, which together contribute a substantial body of data for our experiments. The datasets used in this study are:

- • ShareGPT-4o
- • VideoChat2-IT
- • ShareGPT4Video
- • LLava-178k As shown in Tab. C, these datasets contain over 2 mil-

lion samples in total, a robust foundation for evaluating and

#### D. Examples of VLMs’ responses

Please refer to Fig. I - M for detailed responses from all evaluated VLMs under both chain-of-thought (CoT) and direct output (DO) prompting, based on the given example video and question.

#### E. Details of 4D Reconstruction

We utilize the Feature4X framework (Fig. H) for 4D reconstruction experiments conducted on our dataset. Given an input monocular RGB video, Feature4X reconstructs the

[Figure 45]

Figure A. Performance comparison of various VLMs across annotated question categories including counting, rotational motion, translational motion, false positives, and action recognition

|Category<br><br>|Terms|
|---|---|
|Directional Descriptors<br><br>|left, right, up, down, north, south, east, west, ahead, behind, towards the front, away from the, to the left, toward, to the right, in front of, behind you, side to side, straight ahead, high ground, low ground, left and right, front and back, top and bottom, northern, southern, eastern, western, northeast, northwest, southeast, southwest, around|
|Translational Actions<br><br>|moving, running, walking, sprinting, gliding, sliding, crawling, trotting, jogging, skipping, bounding, rushing, hurrying, traveling, shifting, advancing, progressing, traversing, racing, zooming, going fast, going|
|Rotational Descriptors<br><br>|rotate, revolve, spin, gyrate, twirl, whirl, twist, turn, pivot, flip, roll, spiral, swing, shake, oscillate, swing around, rolls, roll|
|Rotational Actions<br><br>|clockwise, anticlockwise, turn right, turn left, spin, rotate, revolve, twist, pivot, gyrate, whirl, rotating, spinning, turning 360 degrees, turning 180 degrees, rotation, twisting, turning around, circular motion, turning 90 degrees,|
|Perspective Descriptors|camera’s perspective, frame’s perspective, viewpoint, point of view, line of sight, from above, from below, from the side, from the front, from behind, top view, bottom view, rear view, side view, front view, bird’s eye view, aerial view, close-up view, distant view|

Table E. Video Instruction Tuning Datasets Categories & Target strings Analysis

dynamic 3D scene by employing dynamic 3D Gaussians, specifically Dynamic 3D Gaussian Splatting, which represent dynamic foreground elements that deform over time. These dynamic Gaussians are guided by a 4D Motion Scaffold, a sparse graph of trajectory nodes, enabling the interpolation of dense motion trajectories and features for each Gaussian efficiently. A separate set of static 3D Gaussians represents static background elements.

Feature4X introduces a unified latent feature embedding, distilled from various foundational 2D models, which facilitates multiple downstream tasks such as segmentation, scene editing, and visual question answering (VQA).

Specifically, Feature4X extracts video segment features from the InternVideo2-Chat model, a foundation model fine-tuned for video question answering.

This unified feature field is directly used by the InternVideo decoder for inference, bypassing the video encoding step entirely. This approach significantly improves inference efficiency and retains comprehensive structural information from the 4D scene representation, which surpasses the context available from the original 2D videos alone. Consequently, this method enhances downstream tasks by providing richer spatiotemporal context and semantic consistency.

[Figure 46]

###### Figure B. Heatmap of Occurrences of Spatial-Temporal Terms in LLava-178k

(Note: LLava-178k actually comprises over 1.6 million samples, as we combine many of the available dataset splits within this collection.)

[Figure 47]

###### Figure C. Heatmap of Occurrences of Spatial-Temporal Terms in ShareGPT4Video

[Figure 48]

###### Figure D. Heatmap of Occurrences of Spatial-Temporal Terms in VideoChat-IT

[Figure 49]

- Figure E. Heatmap of Occurrences of Spatial-Temporal Terms in ShareGPT-4o

[Figure 50]

Caption:

### ShareGPT-4o

A: the video captures a person moving left, near the kitchen table. the individual, wearing an apron, interacts with various

System Prompt:

items on the table, including a white mug, a cup with blue circular patterns, and a metallic container. throughout the video, the person uses their hands to manipulate the objects,

Generate video descriptions that include details of visual effects, character actions, and movement of people/objects within frames. describe this video and its style to generate a description. pay attention to all objects in the video. do not describe each frame individually. instead of describing the imaginary content, only describing the content one can determine confidently. do not describe the contents by itemizing them in list form.

moving them around the table. the table's surface and the items on it remain in focus while the background, which appears to be a kitchen setting, includes additional items like a water container that is turned and groceries. the video focuses on the hands and objects with no other significant movement, maintaining a steady and close-up viewpoint throughout the sequence.

Directional Descriptors Translational Actions

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Rotational Descriptors

Rotational Actions Perspective Descriptors

- Figure F. Example of a clip with multiple target categories. Spatiotemporal grounding remains a challenge, as the direction descriptor is incorrect, and the translational actions provide limited insight, primarily indicating that objects/subjects are not static.

[Figure 55]

###### Figure G. Heatmap to visualize the prevalence of co-occurring target strings across the ShareGPT-4o dataset, informing our evaluation of spatiotemporal grounding in video instruction data.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

2D Video

[Figure 61]

Feature4X

[Figure 62]

Global View Video Lifted Internvideo2 Global Features

###### Figure H. General Input-Output Architecture of Feature4X

Is the black and green toy truck going clockwise or counter-clockwise around the course? A. moving straight and not turning B. counter-clockwise C. clockwise D. not moving

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

|VLM Responses| |
|---|---|
|GPT-4o|Chain of Thought Response: To determine the direction the black and green toy truck is moving around the course, we need to analyze the sequence of images:\n\n1. **Initial Position:** In the first few images, the truck is on the right side of the course.\n\n2. **Movement Observation:** As the sequence progresses, the truck moves towards the lower part of the images, passing obstacles on its left side.\n\n3. **Further Movement:** The truck continues its path to the bottom left of the images, indicating it is following a path that curves to the left.\n\n4. **Final Position:** By the final images, the truck is seen moving off the course, having taken a wide left turn.\n\nBased on these observations, the truck is moving in a pattern that makes left turns, indicating it is following a counter-clockwise direction around the course.\n\nTherefore, the final answer is: B<br><br>Direct Output Response: B: counter-clockwise|
|Gemini-2.5-Pro|Chain of Thought Response: * **Step 1: Observe the truck's position at the start of its run.** At 00:02, we see a wide shot of the course. The black and green toy truck is positioned in the right-hand lane, pointing away from the camera, preparing to move \"up\" the course.\n\n* **Step 2: Observe the truck's movement in the top section of the course.** At 00:00, the truck is shown jumping over a ramp in the top section of the course. It is moving from the right side of the screen towards the left side. This indicates that after traveling up the right lane, it is crossing over to the left lane at the top of the course.\n\n* **Step 3: Observe the truck's position at the bottom of the course.** At 00:01, the truck is seen at the bottom of the course. It has just come down the left lane and is making a left turn to re-enter the right lane and begin another lap.\n\n*<br><br>**Step 4: Synthesize the path.** The truck travels up the right-hand lane, turns left at the top, travels down the left-hand lane, and turns left again at the bottom. This complete path forms a loop.\n\n* **Step 5: Determine the direction of the loop.** A path that involves moving up on the right and down on the left, with left turns connecting the straightaways, is a counter-clockwise direction.\n\nTherefore, the final answer is: B<br><br>Direct Output Response: B|
|Claude-Sonnet-4|Chain of Thought Response: I need to track the movement of the black and green toy truck throughout the sequence of images to determine if it's moving clockwise, counter-clockwise, straight, or not moving at all.\n\nLet me observe the black and green truck's position across the frames:\n\n1. In the early frames, I can see the black and green truck is positioned in the lower portion of the course\n2. As the sequence progresses, the truck moves toward the right side of the course\n3. Then it continues moving toward the upper portion of the course\n4. Later in the sequence, it moves toward the left side\n5. Finally, it appears to be moving back toward the lower portion, completing what looks like a circular pattern\n\nWhen viewed from above, if I imagine the course as a clock face:\n- The truck starts at roughly the 6 o'clock position (bottom)\n- Moves to about 3 o'clock position (right)\n- Then to about 12 o'clock position (top)\n- Then toward 9 o'clock position (left)\n- And back toward 6 o'clock position (bottom)\n\nThis movement pattern from bottom → right → top<br><br>→ left → bottom represents a clockwise rotation around the course.\n\nTherefore, the final answer is: C<br><br>Direct Output Response: C|
|DeepSeek-VL2|Chain of Thought Response: In the image provided, we can observe a black and green toy truck on a marked course. Its position relative to multiple white cones, ramps, and white boundary lines indicates movement direction. By analyzing spatial orientation and the location of the truck relative to obstacles, it<br><br>appears that the black and green toy truck is making a leftward, counter-clockwise curve among the hurdles. Therefore, the final answer is: B.<br><br>Direct Output Response: B|

###### Figure I. Complete CoT and DO Responses of VLMs Models 1-4

|Grok-2-Vision|Chain of Thought Response: To determine the direction the black and green toy truck is moving, let's analyze the sequence of images step by step:\n\n1. **First Image**: The black and green truck is positioned on the right side of the course, near the starting line.\n2. **Second Image**: The black and green truck has moved slightly forward.\n3. **Third Image**: The black and green truck is moving further forward, and the blue truck is still ahead.\n4. **Fourth Image**: The black and green truck is now clearly moving forward, passing the blue truck.\n5. **Fifth Image**: The black and green truck continues moving forward.\n6.<br><br>**Sixth Image**: The black and green truck is further along the course.\n7. **Seventh Image**: The black and green truck is now on the left side of the course, heading towards the top.\n8. **Eighth Image**: The black and green truck is at the top-left corner, starting to turn down.\n9. **Ninth Image**: The black and green truck is moving down on the left side of the course.\n10. **Tenth Image**: The black and green truck<br><br>is further down on the left side.\n11. **Eleventh Image**: The black and green truck is now at the bottomleft corner, starting to turn right.\n12. **Twelfth Image**: The black and green truck is moving right along the bottom of the course.\n13. **Thirteenth Image**: The black and green truck is moving further<br><br>right.\n14. **Fourteenth Image**: The black and green truck is now at the bottom-right corner, starting to turn up.\n15. **Fifteenth Image**: The black and green truck is moving up on the right side.\n16.<br><br>**Sixteenth Image**: The black and green truck is further up on the right side.\n17. **Seventeenth Image**: The black and green truck is nearing the top-right corner.\n\nFrom this sequence, we can see that the black and green truck is moving in a pattern that takes it from the right side to the left side, then down, then to the right, and then up again. This movement pattern indicates a clockwise direction.\n\nTherefore, the final answer is: **C**<br><br>Direct Output Response: C|
|---|---|
|Llama-4Maverick-17B128E-Instruct|Chain of Thought Response: To determine the direction in which the black and green toy truck is moving around the course, we need to analyze the given images.\n\n1. **Observation of Images**: The images provided show a series of snapshots of toy trucks being operated in an outdoor setting, presumably a parking<br><br>lot that has been converted into a racing or obstacle course. The images are taken from different angles and at different times, capturing the movement of the trucks.\n\n2. **Identifying the Black and Green Truck**: We need to identify the black and green toy truck across the different images to track its movement.\n\n3.<br><br>**Tracking Movement**: By examining the sequence of images, we can track the position and movement of the black and green truck relative to other objects and the course layout.\n\n4. **Analyzing Direction**: \n In Image 1, the truck is seen near the center.\n - In Image 3, it has moved further away from the center towards the top-left.\n - In Image 4, it continues to move away from the center, now towards the top.\n - In Image 5, it is seen at the bottom-right, indicating it has moved in a certain direction around the course.\n\n5.<br><br>**Determining Clockwise or Counter-Clockwise Movement**: To ascertain if the truck is moving clockwise or counter-clockwise, we need to observe its position relative to fixed points or other trucks over the sequence of images.\n\n6. **Analyzing the Sequence**: \n - The images suggest that the trucks are moving around the course in a manner that can be inferred from their positions in successive images.\n - The specific path taken by the black and green truck indicates it is moving in a particular direction around the course.\n\n7. **Conclusion**: Based on the progression of the truck's position across the images (from center to top-left, then top, and eventually bottom-right), and considering the layout of the course, we can<br><br>infer the direction of its movement.\n\nGiven the observed movement from the images, the black and green toy truck appears to be moving in a counter-clockwise direction around the course as it progresses from one part of the course to another.\n\nTherefore, the final answer is: $\\boxed{B}$<br><br>Direct Output Response: B|
|Llama-4-Scout17B-16E-Instruct|Chain of Thought Response: To determine the direction of the black and green toy truck, let's analyze its position in the images provided.\n\n1. **Initial Position and Movement**: The truck starts on the right side of the course in several images, indicating its initial position.\n\n2. **Observing Movement**: In images 2, 7, and 8, the truck's position relative to other objects and its progression can be seen. \n\n3. **Determining Direction**: \n - In image 2, it is airborne, heading towards the left side of the frame.\n - In image 7, it appears on the right side of the frame, closer to the starting point or a specific landmark, suggesting it has moved.\n - In image 8, it is on the left side of the frame but seems to have moved further.\n\n4. **Course Layout and Direction**: The course has various obstacles and is laid out in a way that vehicles move from one end to the other, navigating through or over obstacles.\n\n5. **Conclusion**: Given that the truck moves from the right side towards the left side of the images and considering the course layout, we can infer its direction.\n\nTherefore, the final answer is: B<br><br>Direct Output Response: C|

###### Figure J. Models 5-7

|Phi-4-Multimodalinstruct|Chain of Thought Response: To determine the direction in which the black and green toy truck is moving around the course, we need to observe its path in the video. The black and green toy truck is seen moving around the course, and it appears to be moving in a counter-clockwise direction. This is evident from the way it navigates the ramps and obstacles, moving from one side of the course to the other in a reverse direction compared to the blue and red toy truck. Therefore, the black and green toy truck is moving counterclockwise around the course. Therefore, the final answer is: B.<br><br>Direct Output Response: B|
|---|---|
|Phi-3.5-Visioninstruct|Chain of Thought Response: Looking at the video frames, we see a black and green toy truck positioned at the start of what appears to be an agility course. Initially, the truck is placed on the left side, facing towards the bottom right corner of the screen. As time progresses, the truck moves along the course, which includes turns and jumps. Throughout the course, the truck's orientation changes, indicating it is making turns. After completing one lap, the truck is positioned at the far right side, facing the left corner. Based on the direction the truck is moving from the initial to final position, we can deduce that the barriers and obstacles are arranged in a manner that requires the truck to move counter-clockwise. Thus, the final answer is: B. counter-clockwise. Therefore, the final answer is: B.<br><br>Direct Output Response: C|
|InternVL2.5-38B|Chain of Thought Response: To determine the direction in which the black and green toy truck is moving, let's closely examine the images:\n\n- In Image 2, the black and green toy truck is positioned slightly to the left of the center of the frame.\n- In Image 3, the truck has moved to the right side of the image.\n\nThis indicates that the truck moved from left to right. Given that it starts from the left side and moves to the right side, it is going clockwise around the course.\n\nTherefore, the final answer is: C<br><br>Direct Output Response: B|
|InternVL2.5-8B|Chain of Thought Response: To determine the direction the black and green toy truck is moving around the course, let's closely observe the positioning and trajectory of the toys in the images:\n\n1. **Image 1**: Both the blue and black and green toy trucks are on the course.\n2. **Image 2**: The black and green truck is on the lefthand side of the ramps, heading towards the obstacles.\n3. **Image 3**: The black and green toy truck is now positioned more centrally and appears to be near the middle of the course, likely moving forward.\n4. **Image 4**: The black and green toy truck has moved off the course, closer to the perimeter.\n\nBased on the sequence of images observed:\n- The truck transitions from the far left to the center, which suggests movement towards the rear of the course.\n- Given the arrangement and the constraint of the track, an advancing movement towards the back translates logically to a counter-clockwise progression since the truck is moving away from the immediate path towards outer obstacles.\n\nTherefore, the black and green toy truck is moving counter-clockwise around the course.\n\nTherefore, the final answer is: B<br><br>Direct Output Response: A|
|Pixtral-12b|Chain of Thought Response: To determine the direction in which the toy truck is moving, we need to consider the standard convention for clockwise and counter-clockwise directions.\n\n1. **Understanding Clockwise and Counter-Clockwise:**\n - Clockwise direction means moving in a circular path in the same direction as the hands of a clock, starting from the top-right and moving to the right, then down, and so on.\n<br><br>- Counter-clockwise direction means moving in a circular path opposite the direction of the clock's hands, starting from the top-left and moving to the left, then up, and so on.\n\n2. **Analyzing the Options:**\n -<br><br>**Option A: moving straight and not turning** – This implies that the truck is not following a circular path, and thus it does not have a directional component of being clockwise or counter-clockwise.\n - **Option B: counter-clockwise** – This suggests the truck is moving in the counter-clockwise direction.\n - **Option C: clockwise** – This suggests the truck is moving in the clockwise direction.\n - **Option D: not moving** – This implies that the truck is stationary and not moving at all.\n\n3. **Matching the Given Information:**\n<br><br>- The information provided does not specify a circular path or any turning, indicating no clear direction of motion in a circular path.\n - The truck is explicitly described as \"moving straight and not turning,\" which aligns with Option A.\n\nTherefore, the final answer is: A Direct Output Response: B<br>|

###### Figure K. Models 8-12

|Aria-Chat|Chain of Thought Response: From the provided images, we can observe the movement of the black and green toy truck around the course. In the first image, the truck is moving away from the ramps towards the center of the course. In the second image, it is driving towards the next set of obstacles. By the third image, the truck is turning towards the left, indicating a counter-clockwise direction. This pattern continues, and by examining the sequence of its movement around the course, it consistently follows a counter-clockwise path.\n\nTherefore, the final answer is: B: counter-clockwise.<br><br>Direct Output Response: B: counter-clockwise|
|---|---|
|Qwen2.5-VL-7BInstruct|Chain of Thought Response: To determine the direction of the black and green toy truck, let's analyze the image step by step:\n\n1. **Identify the truck's position**: The black and green truck is on the track, and we need to observe its orientation and the path it appears to be taking.\n\n2. **Analyze the track layout**: The<br><br>track is rectilinear with designated paths for the trucks. There are other trucks visible on the track, but we focus on the black and green one.\n\n3. **Determine the truck's movement**:\n - The truck appears to be moving forward, not along the straight sections but rather following the curved path.\n - The curve in the<br><br>track suggests a direction that can be described as moving from one side of the track to the other, indicating a turn.\n\n4. **Direction of movement**:\n - If we consider the overall direction the truck is taking, it is<br><br>moving from the top-left corner of the image towards the bottom-right corner.\n - This implies that the truck is moving in a counter-clockwise direction around the course.\n\nGiven the analysis, the black and green toy truck is moving counter-clockwise around the course.\n\nTherefore, the final answer is: B<br><br>Direct Output Response: B|
|Qwen2.5-VL-72BInstruct-AWQ|Chain of Thought Response: To determine the direction of the black and green toy truck, let's analyze its movement in the image:\n\n1. **Identify the truck's position and direction**: The black and green truck is seen near the bottom right of the image. It appears to be moving towards the bottom of the frame.\n\n2.<br><br>**Follow the truck's path**: If we trace the likely path of the truck, it seems to be heading down and then will likely turn left towards the middle of the frame.\n\n3. **Determine the overall direction**: Given the truck's trajectory, it appears to be moving in a direction that is consistent with a counter-clockwise movement around the course. It is not moving straight, clockwise, or staying still.\n\nBased on this analysis, the truck is moving counter-clockwise around the course. Therefore, the final answer is: B<br><br>Direct Output Response: B|
|Qwen2-VL-7BInstruct|Chain of Thought Response: To determine whether the black and green toy truck is going clockwise or counter-clockwise around the course, let's analyze the positions of the trucks:\n\n1. The current positions of the trucks are as follows:\n - The black and green truck is on the right side of the course.\n - The orange truck is on the left side of the course.\n\n2. The trucks are currently separate from each other, moving in different directions.\n\n3. Since the image is not sequential, we cannot determine the movement between frames. However, we can infer the direction based on the current position.\n\nGiven the options:\n- A: moving straight and not turning\n- B: counter-clockwise\n- C: clockwise\n- D: not moving\n\nSince the black and green toy truck is on the right side of the course and there is no visual indication that it is turning towards the left, it is likely moving counter-clockwise around the course.\n\nTherefore, the final answer is: $B<br><br>Direct Output Response: B|
|Qwen2-VL-72BInstruct-AWQ|Chain of Thought Response: In the given image, the black and green toy truck is moving clockwise around the course. It appears to be in the process of crossing one of the ramps on the course.\n\nTherefore, the final answer is: $C<br><br>Direct Output Response: B|
|VideoLLaMA3-2B|Chain of Thought Response: Answer: C<br><br>Direct Output Response: C|
|VideoLLaMA3-7B|Chain of Thought Response: The black and green toy truck is moving in a counter-clockwise direction around the course. It starts at the bottom of the frame, moves to the right, then turns left and continues moving in a counter-clockwise direction. Therefore, the final answer is: B.<br><br>Direct Output Response: B|

###### Figure L. Models 13-19

|InternVideo2.5Chat-8B|Chain of Thought Response: Answer: (C) clockwise<br><br>Direct Output Response: B: counter-clockwise|
|---|---|
|InternVideo2Chat-8B|Chain of Thought Response: The toy truck is moving straight and not turning, so it is not going clockwise or counter-clockwise. Therefore, the final answer is: D.<br><br>Direct Output Response: B: counter-clockwise|
|LLaVA-onevisionqwen2-7b-ov-chathf|Chain of Thought Response: Initially, the black and green toy truck with large black tires navigates around the course in a clockwise direction, moving back and forth between two blue ramps. In the final scene, it starts moving towards the edge of the parking lot, partially on the grass next to the concrete surface, continuing its movement in a clockwise direction before exiting the frame. Therefore, the final answer is: C<br><br>Direct Output Response: C|
|LLaVA-NeXTVideo-34B-hf|Chain of Thought Response: I'm unable to determine the direction of motion for the toy truck in this image. The image does not provide enough information to conclude whether the truck is moving clockwise or counter-clockwise around the course. Without more information or seeing the movement of the truck in action, it is impossible to determine its direction of motion accurately.<br><br>Direct Output Response: B|

###### Figure M. Models 20-23

