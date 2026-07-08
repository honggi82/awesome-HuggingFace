## Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation

#### Kaining Ying Henghui Ding Guangquan Jie Yu-Gang Jiang Fudan University, China https://henghuiding.com/OmniAVS/

# arXiv:2507.22886v2[cs.CV]31Jul2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

- (a)
- (b)
- (c)

- (d)

[Figure 6]

[Figure 7]

or

Which object mimics another? Output: It is [SEG]. The girl is imitating the barking of the dog on the right.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

or

The one speak the same language as <sound: Korean>. Output: It is [SEG]. They both speak Korean.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

or

The one similar to <image: > serves as lead. Output: It is [SEG]. The guitar on the right serves as lead.

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

or

The instrument is played by someone of the same gender in <image: > and makes a similar sound in <sound: 🎶Sound of Pipa.>.

Output: It is [SEG]. The pipa held by the man on the right.

[Figure 39]

Figure 1. Examples of the proposed benchmark Omnimodal Referring Audio-Visual Segmentation (OmniAVS) to show its nature and flexibility. OmniAVS introduces 3 key features: 1) It supports diverse multimodal referring expressions that flexibly combine text , speech ,

sound , and image for referring audio-visual segmentation; 2) It emphasizes understanding the content of audio rather than merely hearing them; 3) It incorporates complex reasoning and world knowledge in expressions, prompting models to provide explanations for their segmentation decisions. These characteristics make OmniAVS practical for real-world use and well-suited for developing omnimodal models with fine-grained perception.

### Abstract

Referring audio-visual segmentation (RAVS) has recently seen significant advancements, yet challenges remain in integrating multimodal information and deeply understand-

Henghui Ding (henghui.ding@gmail.com) is the corresponding author with the Institute of Big Data, College of Computer Science and Artificial Intelligence, Fudan University, Shanghai, China.

ing and reasoning about audiovisual content. To extend the boundaries of RAVS and facilitate future research in this field, we propose Omnimodal Referring AudioVisual Segmentation (OmniAVS), a new dataset containing 2,104 videos and 61,095 multimodal referring expressions. OmniAVS stands out with three key innovations: (1) 8 types of multimodal expressions that flexibly combine text, speech, sound, and visual cues; (2) an emphasis

[Figure 40]

[Figure 41]

[Figure 42]

Talking Mute Coughing

on understanding audio content beyond just detecting their presence; and (3) the inclusion of complex reasoning and world knowledge in expressions. Furthermore, we introduce Omnimodal Instructed Segmentation Assistant (OISA), to address the challenges of multimodal reasoning and finegrained understanding of audiovisual content in OmniAVS. OISA uses MLLM to comprehend complex cues and perform reasoning-based segmentation. Extensive experiments show that OISA outperforms existing methods on OmniAVS and achieves competitive results on other related tasks.

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

[Figure 53]

[Figure 54]

Input: Who is making the loudest sound? Output: Mask.

Input: Who is most likely to be sick? Output: Mask. The person coughed, probably sick.

(a) Ref-AVS

(b) OmniAVS (Ours)

Figure 2. Comparison of (a) Ref-AVS [68] and (b) our proposed OmniAVS. In Ref-AVS, expressions mainly focus on surface-level sound properties, while OmniAVS demands deeper understanding and reasoning about sound content, enabling complex reasoning like identifying potential illness from coughing sounds.

### 1. Introduction

Referring Audio-Visual Segmentation (RAVS) [19, 68] is an emerging field focused on segmenting target objects indicated by referring expressions (e.g., text) within audiovisual scenes using audio-visual cues. The combination of referring and audio distinguishes RAVS from Referring Video Object Segmentation (RVOS) [17, 26, 31, 40, 42, 57], which uses text descriptions to refer to objects in silent videos, and Audio-Visual Segmentation (AVS) [24, 86], which segments any sound-producing object without referring expressions. RAVS enables practical applications in video conferencing and robotics, etc.

inputs can greatly enhance flexible referring and humanmachine interactions. However, existing datasets [17, 68] are limited in scope, supporting only single or isolated modalities in their referring expressions. To address this issue, we include diverse expressions in OmniAVS. Beyond the most common text-only expressions, OmniAVS supports text with sound, text with image, and text with both sound and image. Additionally, we allow text to be replaced by speech*, resulting in 8 different kinds of expression types in OmniAVS, as shown in Figure 1.

Omnimodal Instructed Segmentation Assisant. Given the unique and complex requirements of OmniAVS, which necessitate both flexible interfaces and sophisticated multimodal perception and reasoning capabilities, no existing models can be directly applied to this emerging field. To address this issue, we introduce a baseline model named Omnimodal Instructed Segmentation Assistant, based on Multimodal Large Language Model (MLLM) [13]. This model seamlessly integrates text, speech, sound, and image inputs to perform referring object segmentation while providing explanations for prediction. We identify audiovisual synchronization as a critical factor and implement Audio-Visual Interleaving to achieve temporal alignment without introducing additional parameters. Furthermore, unlike concurrent works [3] that use a single token to segment all frames, we adopt a more lightweight query propagation approach to achieve efficient and accurate segmentation. Experimental results demonstrate that the proposed method OISA possesses strong multimodal reasoning capabilities and supports multiple expression patterns, outperforming other methods on OmniAVS dataset. To further validate the effectiveness of our method OISA, we conduct evaluations on other datasets, including referring video/image segmentation and Ref-AVS, achieving competitive results compared to existing methods across all tasks.

Understanding Rather than Just Hearing. However, the association between expressions and audio cues in the previous RAVS dataset Ref-AVS [68] remains superficial, primarily limited to basic acoustic properties such as sound occurrence, intensity, and temporal sequence. For example, expressions like “Who is making the loudest sound?” only address the surface-level characteristics without delving into the content of the sounds. Recent works [3, 33, 53, 70, 84] in image and video segmentation have evolved from simple semantic perception to advanced reasoning understanding. However, such progress is notably absent in audio-visual segmentation. This gap naturally leads us to explore the potential for complex reasoning in audiovisual scenes. Besides, drawing insights from human auditory perception studies [6], understanding sound content is essential for sophisticated audio-visual scene reasoning. To achieve these goals, we present OmniAVS dataset to advance reasoning-based segmentation in audio-visual scenes. As shown in Figure 2(b), our dataset contains expressions that demand reasoning about audio content, such as “Who is most likely to be sick?”. These expressions establish complex cognitive chains that go beyond basic acoustic features, e.g., sound→coughing→sickness. Furthermore, we enhance the dataset’s interpretability by providing detailed explanations for reasoning-based expressions.

This work makes four main contributions: i) we explore omnimodal referring audio-visual segmentation, which is more aligned with real-world scenarios and provides a solid foundation for developing future omnimodal AI systems; ii)

Towards Omnimodal Referring. The emergence of flagship model ChatGPT-4o [51] sets a new standard for modern AI systems, emphasizing the need to process arbitrary combinations of omnimodal inputs, e.g., audio, vision, and text. In the context of RAVS, enabling omnimodal

*To clarify, speech is human-produced spoken sound.

we introduce a large-scale referring audio-visual segmentation dataset OmniAVS that enables advanced multimodal reasoning in audio-visual scenes, with a particular emphasis on deep understanding and reasoning from audio content; iii) we enhance interaction flexibility by incorporating multiple referring modalities in the proposed OmniAVS dataset, broadening the scope of referring AVS; iv) we develop OISA, a multimodal large language model designed to uniformly handle omnimodal expressions and reasoning in RAVS, with extensive experiments across different tasks.

### 2. Related Work

Audio-Visual Scene Perception. Audio-visual scene perception [35–37, 47, 60, 66–68, 72] aims to simulate human multimodal sensory capabilities by integrating auditory and visual information. The rapid development of deep learning has enabled joint audio-visual learning, achieving significant progress in tasks such as event detection [62], robotic manipulation [64], object localization [30], action recognition [8], and object segmentation [68, 86]. The integration of multiple modalities enables models to achieve superior performance compared to single-modality approaches, particularly in handling incomplete or ambiguous information. However, datasets that comprehensively incorporate audio, visual, and language components remain scarce. Among the few available datasets, Music-AVQA [35], AVQA [72], and Ref-AVS [68] stand out as notable examples. MusicAVQA [35] and AVQA [72] primarily address questionanswering tasks. Ref-AVS [68] focuses on referring audiovisual segmentation. Nevertheless, Ref-AVS’s utilization of audio information is limited, as it employs relatively simple expressions without complex reasoning, thus constraining its effectiveness in real-world scenarios. To overcome these limitations, we introduce OmniAVS, which is specifically designed to maximize the utilization of audio information and seamlessly integrate it with textual descriptions.

Reasoning-based Perception. With the rapid development of MLLMs [2, 13, 16, 43–45, 58, 61, 63, 76, 81], the research community has shifted from perception based on simple semantic [17, 71] to perception based on world knowledge or complex reasoning [3, 25, 33, 53, 70, 82]. For example, GPT4RoI [82] prompts MLLM with bounding boxes to solve region-level complex tasks that require reasoning. DetGPT [53] combines a pretrained MLLM with an open-vocabulary object detector to enable reasoningbased object detection from natural language instructions. LISA [33] uses a MLLM [43] to output a [SEG] token, which is then processed by a mask head [32] to generate the masks. Leveraging the world knowledge and reasoning capabilities inherent in LLMs, LISA demonstrates superior performance in complex segmentation scenarios that require complex reasoning. Building upon this foundation, VideoLISA [3] and VISA [70] extend LISA’s capabilities to

reasoning video object segmentation by incorporating comprehensive temporal modeling. Drawing inspiration from these works, we propose OISA, which extends reasoning capabilities to audio-visual contexts. Unlike previous works that primarily focus on vision-language reasoning, OISA leverages MLLM to perform reasoning about audio content in videos. While AnyRef [25] pioneers the use of audio and images as referring inputs for segmentation, it is constrained to static images and lacks the ability to reason about audio in dynamic video contexts. Moreover, some works [21, 39, 51] extend LLMs to multiple modalities, e.g., vison, audio, and text, but they cannot be used for segmentation tasks.

### 3. Benchmark: OmniAVS

##### 3.1. Audio-Visual Video Collection

The video sources consist of three main parts: i) Realworld web videos under the Creative Commons License, where we use techniques from [10, 68, 86] to align audio and visual snippets with intended semantics; ii) The opensource dataset TVQA [34], featuring extensive dialogue from TV shows; iii) Self-recorded videos, with consent from all participants. We select videos from above that meet the following two standards:

- • We encourage to select videos that contain audio content with informative or reasoning-based meanings and corresponding identifiable objects in the scene.
- • We strive to select videos with complex scenes and multiple objects, enabling more diverse referring expressions, e.g., referring to multiple targets, and makes the dataset more representative of real-world complex scenarios.

After reviewing about 10,871 potential candidates, we carefully selected the most satisfactory and appropriate videos, which meet the above standards. Finally, we chose 2,104 videos to create the OmniAVS dataset, which is diverse and representative of a wide range of real-world audio-visual scenarios. After collecting the videos, our research team annotates potential objects with expressions and bounding boxes at their first appearance. This information is then used by the annotation team to achieve video mask annotation on the referred objects. Next, we will introduce the corresponding annotation process.

##### 3.2. Omnimodal Expression and Mask Annotation

The expression annotation process integrates 4 different modalities: text, speech, sound, and image. We categorize referring expressions into 8 forms: I) Text; II) Speech; III) Text with Sound; IV) Speech with Sound; V) Text with Image; VI) Speech with Image; VII) Text with Sound and Image; and VIII) Speech with Sound and Image. Each expression includes either text or speech, as they provide essential instructions for the model to understand the task and other modalities. All speech expressions are generated

Table 1. Statistical comparison between the newly proposed OmniAVS and other datasets of related referring video segmentation tasks. Avail., Expl., and Expr. are abbreviations for Availability of reasoning, Explanations, and Expressions, respectively.

|Dataset Venue<br><br>|Content Video Audio<br><br>|Referring Text Audio Image<br><br>|Reasoning Avail. Expl.<br><br>|Statistics Video Frame Object Mask Expr. Expl.<br><br>|
|---|---|---|---|---|
|Refer-YouTubeVOS [57] [ECCV’20] MeViS [17] [ICCV’23] ReVOS [70] [ECCV’24] Ref-AVS [68] [ECCV’24]<br><br>|✓ ✗ ✓ ✗ ✓ ✗ ✓ ✓<br><br>|✓ ✗ ✗ ✓ ✗ ✗ ✓ ✗ ✗ ✓ ✗ ✗<br><br>|✗ ✗<br><br>✗ ✗<br><br>✓ ✗<br><br>✗ ✗<br><br><br>|3,978 116k 7,451 131k 15,009 ✗ 2,006 44k 8,175 443k 28,570 ✗ 1,042 116k 5,535 469k 35,074 ✗<br><br>4,002 40k 6,888 78k 20,261 ✗<br><br><br>|
|OmniAVS (ours) [ICCV’25]<br><br>|✓ ✓<br><br>|✓ ✓ ✓<br><br>|✓ ✓<br><br>|2,104 103k 4,277 206k 61,095 34,841<br><br>|

by converting the corresponding text expressions into audio. Following previous works [21, 52], we use a combination of several TTS models [23, 50] and human readings for this conversion. In light of this, we will focus on introducing the annotation process for text-related expressions. Figure 3 shows the distribution of different expression types. The inner areas (e.g., text) are composed of their corresponding outer areas (e.g., 4 components of text).

I) Text. This is the main component of OmniAVS. The annotation process follows these rules: 1) Expressions should relate to the sound in the video, not just visual cues. 2) Expressions should emphasize the sound’s content, not just the act of making it. For example, instead of “The dog barking”, use “The dog warning” as it requires understanding the sound. 3) Encourage reasoning in expressions, with necessary explanations. 4) Expressions can refer to any number of objects, from none to many (more than one). III) Text with Sound. This form combines text and sound to provide a more comprehensive expression. The annotation process follows the same rules as in the Text modality, with the following additional requirements when adding audio: 1) Use natural sounds, instrumental music, or environmental sounds, e.g., dog barking, bird chirping. 2) Text should complement the sound, adding context or instructions for identifying the object(s). An example is shown in the 2nd row of Figure 1.

V)) Text with Image. This type combines text and image. The annotation follows the rules of Text modality with additional considerations for images: 1) Include relevant images that depict the target object(s) or provide context for the audio-visual content. 2) Ensure the text complements the image, offering instructions for identifying the object(s). An example is shown in the 3rd row of Figure 1.

VII) Text with Sound and Image. This type combines text, sound, and image in the expressions. The text should guide identification using both sound and image. An example is shown in the 4th row of Figure 1.

Mask Annotation. After collecting expressions, the annotation team proceeds to segment the objects referred to by each expression in every video. Using the bounding box of each object’s first appearance (not necessarily in the first frame) annotated in the previous step as a reference,

Table 2. More dataset statistics.

|Statistic|Number<br><br>|
|---|---|
|Words per Expr. FPS (mean/min/max) Expression Types Frames (min/max) Obj. per Expr. (mean) Obj. per Expr. (max) Duration Videos (train/test) Expr. (train/test)<br><br>|9.3 4.8/3/15 8 11/251 1.12 7 6.2 h 1,864/240 54,304/6,791<br><br>|

Figure 3. Distribution of expression types. SI: sound+image.

Image 1,731

SI 1,553

Sound 1,738

Speech 25,472

[Figure 55]

###### OmniAVS 61,095

Text Speech

[Figure 56]

Text25,474

Image 1,811

Sound 1,755

SI 1,561

the annotation team is required to carefully confirm the corresponding object in the video, then track and annotate the object’s mask in all frames †. We designed an interactive annotation tool to automatically load the video and corresponding object bounding boxes. Annotators use the tool to load and preview the corresponding video, then sequentially track and segment target objects in the video. The tool integrates state-of-the-art interactive video object segmentation model SAM2 [56] to assist with mask annotation, enhancing efficiency.

##### 3.3. Dataset Statistics and Analysis

Table 1 shows a statistical comparison between OmniAVS and related datasets. Refer-YouTubeVOS [57], MeViS [17], and ReVOS [70] focus on silent videos without audio and provide only text expressions. While ReVOS [70] further supports reasoning expression, it lacks explanations. RefAVS Bench [68] includes audio-visual videos but supports only single-modality text expressions without reasoning or explanations. Compared to previous datasets, the proposed OmniAVS dataset offers multimodal content (audio-visual videos) and diverse expression modalities (text, speech, sound, image), supports reasoning with explanations, and allows for referring to arbitrary number of target objects. We compare more datasets in supplementary.

Main Differences with Ref-AVS. As Ref-AVS [68] is the most related dataset to our proposed OmniAVS, we summarize the differences here to facilitate reference:

• Omnimodal Expressions: While Ref-AVS uses only textual referring expressions, OmniAVS supports expres-

†We follow MeViS [17] to perform full-temporal segmentation, even when objects only partially match expressions in specific time segments.

[Figure 57]

sions that can combine text, speech, sound, image, providing a richer multimodal interactive interface.

Pixel Decoder

Sync Audio-Visual Content

|[Figure 58]| | |[Figure 59]<br><br>[Figure 60]| |
|---|---|---|---|---|

Mask Decoder

- • Enhanced Audio-Visual Integration: Unlike Ref-AVS, which primarily focuses on “the sounding object”, OmniAVS offers a more comprehensive audio-visual analysis. Our expressions delve deeper into the content of sounds and their contextual impact on the scene.
- • Reasoning and Explanation: Our dataset includes numerous expressions requiring reasoning, such as “Who is most likely to be sick?” (see Figure 2), which necessitates inferring from audio-visual cues like sneezing. We also provide explanations for some reasoning expressions.
- • Number of Referred Target Objects: Unlike Ref-AVS that only refers to 0 or 1 object per expression, OmniAVS allows each referring expression to indicate an arbitrary number of target objects, i.e., from 0 to many as [41].
- • Annotation FPS: While Ref-AVS uses fixed video settings (10s, 1 FPS), OmniAVS includes videos of varying durations and annotation frame rates (3-15 FPS) to capture temporal dynamics. Table 2 provides more statistics.

ViT Adapter

[Figure 61]

Audio-Visual Interleaving

It is [SEG]. The left side guitar is pipa-like and makes sound.

Video Content

[Figure 62]

Audio Content

Response

Multimodal LLM

[Figure 63]

Omnimodal Expression

|[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]|
|---|

The one has a shape similar to <image> and makes a sound like <sound>?

Text or Speech

[Figure 67]

[Figure 68]

<image> <sound>

Output Mask

Figure 4. Overview of the proposed method OISA. Vision/audio encoders and text embedding are omitted for clarity.

integrated into the corresponding text tokens to produce the final Omnimodal Expression Tokens. All types of tokens are subsequently fed into the MLLM. The MLLM generates text responses and produces a [SEG] token representing the target object, which is then used by mask decoder for segmentation prediction. Our mask decoder uses query propagation to segment each frame, as described in Section 4.3. The query is refined online during the segmentation process, alleviating dynamic motion. Following previous work [33], we employed cross entropy loss for text generation, DICE loss [49], and binary cross entropy loss for segmentation.

### 4. Method: OISA

##### 4.1. Overview Architecture

Because of the unique nature of this task and the proposed OmniAVS dataset, there is no off-the-shelf model available. We propose an Omnimodal Instructed Segmentation Assistant (OISA) as a baseline method to support omnimodal referring and reasoning audio-visual segmentation. As shown in Figure 4, OISA consists of two main components: a MLLM for multimodal understanding and generation, and a mask head for segmentation and tracking. The MLLM includes audio encoder, vision encoder and LLM, while the mask head consists of ViT-Adapter [12], pixel decoder and mask decoder [14]. It’s worth noting that the mask head is flexible and can be replaced with models like SAM [32].

##### 4.2. Audio-Visual Interleaving

For videos of arbitrary length, we uniformly sample N frames from the beginning to the end. Each frame is individually processed by the vision encoder to obtain Lv vision tokens, denoted as V = {v1,v2,...,vN}, where vi ∈ RL

v×d represents the Lv vision tokens of the i-th frame with dimension d. For the audio content in the video, we use an audio encoder to process it and obtain audio tokens A ∈ RL

A×d, where LA denotes the total number of audio tokens. To achieve audio-visual alignment, we segment the audio into clips corresponding to the frame rate, obtaining {a1,a2,...,aN}, where ai ∈ RL

Given an audio-visual video, the multimodal encoder processes video and audio content to obtain corresponding vision and audio tokens. Different from VideoLLaMA [80] that concatenates these two modalities sequentially, we adopt an Audio-Visual Interleaving strategy, as described in Section 4.2, where we divide audio tokens into clips and interleave them with vision tokens. The interleaved sequence forms our Audio-Visual Content Tokens. This approach effectively synchronizes audio and frames without introducing additional parameters, which is particularly useful for scenarios requiring audio-visual alignment. See examples at Figure 2, where without proper audiovisual alignment it would be difficult to determine which object sneezed when multiple objects are making sounds simultaneously. For omnimodal expression inputs, we employ the audio encoder to process speech and sound, and the vision encoder to handle images. The resulting tokens are then

a×d and La = LA/N. Next, we interleave audio token clips and vision tokens to form the audio-visual interleaving token sequence {v1,a1,v2,a2,...,vN,aN}. In comparison, VideoLLaMA [15, 80] directly concatenates audio and visual tokens sequentially, {v1,...,vN,a1,...,aN}. However, this approach makes it difficult to align audio frames, especially in segmentation tasks where the video’s FPS is not fixed, thereby exacerbating the challenge of audiovisual synchronization. Besides, video-SALMONN [58] fuses audio and visual tokens by concatenating them along the feature dimension and then applying a weight matrix for mapping. This way introduces additional training parameters and causes the fused audio-visual features becoming

Frame 1

Frame 2

Frame 3

Frame 1

Frame 2

Frame 3

extra audio MLP to project audio features into LLM feature space. For ViT-Adapter [12], we remove all Injectors to reduce memory consumption and ensure vision tokens remain unaffected. For pixel decoder and mask decoder, we follow the default design of Mask2Former [14] but remove the self-attention module in each transformer block since there is only one query input to the mask decoder. We denote our model with the above settings as OISA-1B.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Mask Decoder

Mask Decoder

Mask Decoder

Mask Decoder

Mask Decoder

Mask Decoder

share share

update update

(a) One-Token-Seg-All (b) Query Propagation

Dataset Split. We split OmniAVS into training and testing sets, containing 1,864 videos with 54,304 expressions and 240 videos with 6,791 expressions, respectively.

Figure 5. Comparision of different mask decoder forms.

Evaluation Metrics. Following previous work [17, 18, 20, 68, 77], we adopt J &F as the segmentation evaluation metric, where J measures region similarity (IoU) and F evaluates contour accuracy. Following [41], for no-target expressions that referring to nothing in the video, J &F is set to 1 if the prediction is empty; otherwise, it is 0. We employ METEOR [4] as evaluation metric for text explanation generation.

misaligned with the original pretrained MLLM model.

##### 4.3. Query Propagation

As shown in Figure 4, once the Audio-Visual Content Tokens and Omnimodal Expression Tokens are obtained, the MLLM generates the corresponding responses. We employ the [SEG] token, following LISA [33], to represent the object embedding, which is subsequently passed to the mask decoder. While MLLMs excel at multimodal understanding and reasoning, they are not good at segmentation tasks. Previous works [3, 33] attach an additional vision encoder for plain feature extraction and a mask head for segmentation after MLLM, resulting in a redundant and suboptimal design. To address these issues, when extracting vision tokens from video, we simultaneously extract corresponding multi-scale features through ViT Adapter [12], which are then enhanced via pixel decoder [14]. These enhanced features, along with the [SEG] generated by the MLLM, are input into the mask decoder. When using mask decoder for segmentation, VideoLISA [3] employs the same [SEG] for the independent segmentation of each frame, aka One-Token-Seg-ALL (OTSA), as shown in Figure 5 (a). However, this approach has limitations. Previous studies [21, 27, 74, 75, 83] have shown that a single query often fails to adequately represent a target object, especially when rapid motion is present within the video. A single query carries a positional prior, making it difficult to encapsulate dynamic motion processes, such as an object moving from right to left, as shown in Figure 5 (a). This limitation results in ID switches, consistently tracking the target on the right part of the video. To address this, we employ a query propagation mechanism, as shown in Figure 5 (b), which allows the query to be updated online on a frame-byframe basis [27, 79]. This approach enables the query to smoothly capture the temporal motion of the object while also effectively modeling contextual temporal information.

##### 5.1. Training Details

Audio-Text Alignment. Since InternVL2 does not support audio input natively, we introduce an audio encoder MLP to align audio features with the LLM space. This stage mainly aims to align audio features with LLM space. To achieve this goal, we use Automatic Speech Recognition datasets [9] and Audio Caption datasets [59] for audio-text alignment. During this process, we only train the audio encoder MLP while keeping all other parameters frozen.

Omnimodal Instruct Segmentation Tuning. Following previous works [3, 33, 70, 84], we use multiple datasets for training, including 1) semantic segmentation datasets: ADE20K[85], COCO-Stuff[7], PASCAL-Part [11], PACOLVIS [55]; 2) referring segmentation datasets: RefCOCO, RefCOCO+ [78], RefCOCOg [48], ReasonSeg [33]; 4) referring video segmentation datasets: Refer-YouTube-VOS [57], Refer-DAVIS-17 [31], MeViS [17], ReVOS [70]; 5) audio-visual segmentation datasets: Ref-AVS Bench [68], OmniAVS. We use LoRA [29] to fine-tune LLM, train all parameters of the mask head, and freeze all remaining parameters. For video samples, during training, we uniformly sample 10 frames from the entire video and follow [3] to use 4 frames as dense frames and the remaining frames as sparse frames. During inference, we sample 32 frames with 4 frames as dense frames.

##### 5.2. Ablation Study

Ablation Study on Audio-Visual Fusion. In Table 3 we evaluate different audio-visual fusion strategies. For attention baseline, we adopt a simple cross-attention block where image tokens serve as key and audio tokens serve as query/value. For weighted sum, we first pad the audio tokens to match the length of vision tokens, then perform

### 5. Experiment

Implementation Details. We adopt InternVL2-1B [13] as our MLLM, equipped with InternViT-300M-448px [13]

- as vision encoder and Qwen2-0.5B-Instruct [1] as LLM. Whisper-large-v3 [54] is used as our audio encoder, with an

- Table 3. Ablation study on audio-visual fusion. AVI means Audio-Visual Interleaving. TVQA Split represents the performance evaluated on samples collected from TVQA [34]. These samples demand precise audio-visual alignment and feature a substantial amount of dialogue. See an example in Figure 6 (a).

Fusion Type Sketch TVQA Split Overall

Attention - 37.4 35.8 Concat 36.9 35.3 Weighted Sum 34.8 31.1 AVI 40.8 39.2 AVI + Concat 42.0 40.5

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- Table 4. Ablation study of mask head. OTSA: One-Token-SegAll [3]; QP: query propagation (Ours); M2F: Mask2Former. FPS is tested on one A6000 GPU (batch size=1).

ID

Query Type Head Type OmniAVS Metrics OTSA QP M2F SAM J &F J F FPS

- I ✓ ✗ ✗ ✓ 38.1 36.4 39.8 4.3
- II ✓ ✗ ✓ ✗ 35.2 33.8 36.6 15.7
- III ✗ ✓ ✗ ✓ 41.2 39.2 43.2 4.1
- IV ✗ ✓ ✓ ✗ 40.5 38.7 42.3 12.3

weighted sum on them. Attention-based and concatenation [15] methods achieve similar performance around 35-37% J &F. Our proposed Audio-Visual Interleaving (AVI) strategy significantly improves the overall performance to 39.2% and get 40.2% in TVQA split. Previous works [13, 44] add a thumbnail after tiling to capture global multimodal features to capture global features more effectively. Similarly, we append original audio tokens A at the end of the audio-visual interleaving sequence. This supplements complete and untruncated audio tokens based on our AVI audio-visual alignment, enabling the MLLM to handle more comprehensive audio tokens and thereby improving the model performance to 40.5% J &F.

Ablation Study on Mask Head. In Table 4 we study different mask head designs. While SAM mask head achieves competitive performance compared to M2F, it suffers from slower inference speed due to its huge ViT backbone. When using M2F as the mask head (Setting II and IV), we achieve at least 3× speedup with a slight accuracy trade-off. Regarding query types, our QP approach (Setting III and IV) significantly outperforms OTSA in segmentation quality, improving J &F from 38.1% to 41.2% when using SAM mask head, and from 35.2% to 40.5% when using M2F mask head. Considering the balance between performance and efficiency, we choose Setting IV (QP with M2F head) for our subsequent experiments.

- 5.3. OmniAVS Benchmark Results

In Table 5, we report the OmniAVS benchmark results. For traditional referring segmentation methods [17, 68, 71], we process each modality independently and concatenate their features to create a unified query representation.

Table 5. Testing on OmniAVS. We use J &F as the default metric. All is the average result across 8 splits. MET.: METEOR.

Method All I II III IV V VI VII VIII MET. LMPM [17] 25.8 31.2 28.7 20.0 22.7 21.3 20.9 30.0 31.4 EEMC [68] 29.6 34.4 32.6 19.6 26.0 28.0 24.7 35.6 36.0 MUTR [71] 32.3 35.4 33.3 28.4 29.8 26.5 22.8 41.6 40.5 LISA-7B [33] 33.6 33.3 31.2 29.2 32.7 28.6 27.3 43.4 43.1 11.6 LISA-13B [33] 36.1 36.4 32.1 30.4 35.7 31.6 30.2 46.7 45.7 16.5 OISA-1B (ours) 41.1 40.1 38.5 34.9 38.5 35.9 35.2 52.6 53.0 21.7

Regarding LISA [33], we enhance its audio processing capability by applying the same audio-text alignment method as our OISA, while maintaining its frame-by-frame inference approach. Our OISA-1B achieves state-of-the-art performance on the OmniAVS benchmark, significantly surpassing previous methods. Specifically, it attains an average score of 41.1% across all modality combinations, outperforming LISA-13B by 5.0%. These improvements are consistent across all splits, with the most notable gain of 7.3% on split VIII (speech+sound+image), showcasing our model’s superior ability in multimodal input processing and reasoning. In terms of explanation generation, our model achieves a METEOR score of 21.7%, significantly surpassing LISA-13B’s score of 16.5%. This indicates that our model excels not only in segmentation tasks but also in understanding and explaining its decisions. Analysis across different modality combinations reveals that splits VII (text+speech+image) and VIII (text+sound+image) achieve the highest scores of 52.6% and 53.0%, respectively. This suggests that incorporating more modalities generally enhances performance, as the model effectively leverages complementary information from various sources. Comparison with the previous dataset Ref-AVS Bench shows that our OISA-1B achieved a score of 58.0% (shown in Table 6), whereas it only attained 41.1% on our OmniAVS. This significant difference underscores the increased difficulty and complexity of our dataset.

Qualitative Results and Failure Cases. In Figure 6, we present several success and failure cases of the proposed baseline method OISA. Case (a) demonstrates that OISA effectively aligns audio with visual content, identifies key dialogues in the fourth frame, and subsequently performs reasoning and accurately locates the target object, providing a clear explanation. Besides, Case (a) shows that OISA achieves robust tracking in scenarios involving viewpoint changes, occlusions, and reappearances. Case (b) illustrates the model’s ability to handle multimodal expressions by understanding both the fire in the image and the siren sound to locate the fire truck. However, as shown in Case (c), OISA struggles in scenarios with complex sounds, indicating that understanding intricate sounds still requires further exploration, such as decoupling sound events [28, 46] before perception and reasoning.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- Table 7. cIoU Results on refCOCO [48, 78] and ReasonSeg [33].

Method

refCOCO refCOCO+ refCOCOg ReasonSeg

val testA testB val testA testB valU testU val test

|LAVT [73] ReLA [41] LISA-7B [33] LISA-13B [33] VISA-7B [70] VideoLISA-3.8B [3]<br><br>|72.7 75.8 68.8<br>73.8 76.5 70.2<br>74.1 76.5 71.1<br><br><br>- - -<br><br>72.4 75.5 68.1<br>73.8 76.6 68.8<br>|62.1 68.4 55.1 66.0 71.0 57.7<br><br>62.4 67.4 56.5<br><br>- - 59.8 64.8 53.1<br><br>63.4 68.8 56.2<br>|61.2 62.1<br><br>65.0 66.0<br><br>66.4 68.5<br><br><br>- 65.5 66.4 68.3 68.8<br><br>|- -<br>- -<br><br><br>54.0 48.4 62.9 51.1 57.8 -<br><br>67.1 54.4|
|---|---|---|---|---|
|OISA-1B (ours)|74.2 78.3 72.0<br><br>|68.8 71.7 62.0<br><br>|70.6 72.3<br><br>|61.2 50.8<br><br>|

- Table 8. Results on referring/reasoning video segmentation [17, 31, 57, 70]. J &F is the default metric, R is the robustness score.

[Figure 86]

Sir? Just yanking you chain!

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

- (a)

Left man: Nice job!

- (b)
- (c) Figure 6. Success and failure cases of OISA-1B.

Who is joking? Output: [SEG]. The boss is teasing his colleague.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Fire truck and ambulance sirens

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

The one makes <sound: Siren> and moves to ? Output: [SEG]. The first fire truck.

[Figure 108]

[Figure 109]

Blended sounds of Bassoon, Trombone, Clarinet, Violin, Cello and Piano

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

The one produces the most crisp and resonant sound? Output: No object match this.

[Figure 116]

[Figure 117]

Method MeViS R-YTVOS R-DAVIS17 ReVOS ReVOS(R) ReferFormer [69] 31.0 62.9 61.1 14.9 4.9 MTTR [5] 30.0 55.3 - 25.5 5.6 LMPM [17] 37.2 - - 26.4 3.2 LISA-7B [33] - 50.2 58.4 40.9 9.3 TrackGPT-7B [87] 40.1 56.4 63.2 43.6 11.6 VideoLISA-3.8B [3] 42.3 61.7 67.7 - VISA-7B [70] 44.5 63.0 70.4 47.1 15.3 OISA-1B (ours) 43.2 62.1 65.2 47.3 19.3

Table 6. Performance on testing set of Ref-AVS [68].

Seen Unseen Mix (S+U) Null J F J F J F S (↓)

Method

AVSBench [86] 23.2 51.1 32.4 54.7 27.8 52.9 20.8 AVGSegFormer [22] 33.5 47.0 36.1 50.1 34.8 48.6 17.1 GAVS [65] 28.9 49.8 29.8 49.7 29.4 49.8 19.0 ReferFormer [69] 31.3 50.1 30.4 48.8 30.9 49.5 17.6 R2VOS [38] 25.0 41.0 27.9 49.8 26.5 45.4 18.3 EEMC [68] 34.2 51.3 49.5 64.8 41.9 58.1 0.7 OISA-1B (ours) 51.7 58.7 58.3 65.1 54.5 61.4 9.8

### 6. Conclusion and Discussion

We introduce OmniAVS, a large-scale dataset supporting advanced multimodal referring and reasoning in audiovisual scenes with 8 types of expressions across text, speech, sound, and image. Compared to previous referring datasets focused on salient video or sound occurrence, OmniAVS enables deeper understanding and reasoning of audiovisual content. To address the challenges posed by OmniAVS, we propose a simple yet effective baseline OISA, a segmentation model based on multimodal large language models. Our OISA leverages audio-visual interleaving for temporal alignment and query propagation for efficient segmentation, enabling effective handling of multimodal inputs without additional parameters. Extensive experiments on 10 datasets demonstrate the effectiveness of our OISA.

##### 5.4. Results on Related Tasks

Results on Ref-AVS. As shown in Table 6, our OISA1B achieves SOTA performance on Ref-AVS [68], greatly surpassing previous methods. Specifically, OISA-1B improves the J by 17.5% and 8.8% on Seen and Unseen splits respectively compared to EEMC [68], demonstrating good generalization ability. While OISA-1B gets suboptimal score on Null split, which may inherit from hallucination in MLLM, EEMC achieves 0.7% S. We argue that videos in Null split contain empty or fixed audio with unrelated expressions, so EEMC is likely overfiting to identify audio patterns without understanding. Therefore, our model’s performance (9.8%) is reasonable, and OISA-1B’s advantages on other metrics also demonstrate its robustness.

Future Directions. Although our baseline OISA has shown promising results on OmniAVS, there is still much room for improvement: i) Exploring more efficient audiovisual fusion methods, such as learning more robust audiovisual representations. ii) Disentangling sounds when multiple objects make sounds simultaneously, which is crucial for subsequent reasoning. iii) How to effectively combine multiple modalities in expressions by using joint representations or cross-modal fusion. iv) Exploring more advanced segmentation models to handle complex scenarios, e.g., occlusion, disappearance and reappearance. v) Pre-training OISA on larger audio-visual datasets to further improve its generalization to unseen tasks. vi) Developing more interactive and conversational capabilities to enable multiround interactions with users, allowing for clarification and refinement of referring expressions.

###### Results on Referring Image Segmentation. As shown

- in Table 7, our OISA-1B outperforms previous referring image segmentation methods on refCOCO series [48, 78], achieving state-of-the-art performance. The performance on ReasonSeg [33] falls short of LISA [33] and VideoLISA [3], likely due to the limited reasoning abilities of our smaller 0.5B LLM, which are crucial for this dataset.

Results on Referring Video Segmentation. As shown

- in Table 8, the proposed OISA-1B achieves competitive results on MeViS [17] (43.2%), R-YTVOS [57] (62.1%),

and R-DAVIS17 [31] (65.2%). On ReVOS [70], it achieves new state-of-the-art results with 47.3% J &F and 19.3% robustness score, surpassing VISA [70] by 0.2% and 4.0% respectively, demonstrating strong perception and reasoning ability in complex video scenes.

Acknowledgement. This project was supported by the National Natural Science Foundation of China (NSFC) under Grant No. 62472104.

### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen Technical Report. arXiv, 2023. 6
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. arXiv preprint arXiv:2308.12966,

2023. 3

- [3] Zechen Bai, Tong He, Haiyang Mei, Pichao WANG, Ziteng Gao, Joya Chen, liulei, Zheng Zhang, and Mike Zheng Shou. One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos. In Adv. Neural Inform. Process. Syst., 2024. 2, 3, 6, 7, 8
- [4] Satanjeev Banerjee and Alon Lavie. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. In Assoc. Comput. Linguist. Worksh., 2005. 6
- [5] Adam Botach, Evgenii Zheltonozhskii, and Chaim Baskin. End-to-End Referring Video Object Segmentation with Multimodal Transformers. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 8
- [6] Albert S Bregman. Auditory Scene Analysis: The Perceptual Organization of Sound. MIT press, 1994. 2
- [7] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. COCOStuff: Thing and Stuff Classes in Context. In IEEE Conf. Comput. Vis. Pattern Recog., 2018. 6
- [8] Jacob Chalk, Jaesung Huh, Evangelos Kazakos, Andrew Zisserman, and Dima Damen. TIM: A Time Interval Machine for Audio-Visual Action Recognition. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 3
- [9] Guoguo Chen, Shuzhou Chai, et al. GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio. In Proc. Interspeech 2021, 2021. 6
- [10] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. VGGSound: A Large-scale Audio-Visual Dataset. In IEEE Int. Conf. Acoust. Speech Signal Process., 2020. 3
- [11] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect What You Can: Detecting and Representing Objects using Holistic Models and Body Parts. In IEEE Conf. Comput. Vis. Pattern Recog.,

2014. 6

- [12] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision Transformer Adapter for Dense Predictions. In Int. Conf. Learn. Represent., 2023. 5, 6
- [13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with OpenSource Suites. arXiv preprint arXiv:2404.16821, 2024. 2, 3, 6, 7
- [14] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention Mask Transformer for Universal Image Segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 5, 6

- [15] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. VideoLLaMA 2: Advancing SpatialTemporal Modeling and Audio Understanding in VideoLLMs. arXiv preprint arXiv:2406.07476, 2024. 5, 7
- [16] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-Audio Technical Report. arXiv preprint arXiv:2407.10759, 2024. 3
- [17] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions. In Int. Conf. Comput. Vis., 2023. 2, 3, 4, 6, 7, 8
- [18] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. MOSE: A New Dataset for Video Object Segmentation in Complex Scenes. In Int. Conf. Comput. Vis., 2023. 6
- [19] Henghui Ding, Song Tang, Shuting He, Chang Liu, Zuxuan Wu, and Yu-Gang Jiang. Multimodal referring segmentation: A survey. arXiv, 2025. 2
- [20] Henghui Ding, Kaining Ying, Chang Liu, Shuting He, YuGang Jiang, Philip HS Torr, and Song Bai. MOSEv2: A more challenging dataset for video object segmentation in complex scenes. arXiv, 2025. 6
- [21] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. VITA: Towards Open-Source Interactive Omni Multimodal LLM. arXiv, 2024. 3, 4, 6
- [22] Shengyi Gao, Zhe Chen, Guo Chen, Wenhai Wang, and Tong Lu. AVSegFormer: Audio-Visual Segmentation with Transformer. In AAAI, 2024. 8
- [23] GPT-SoVITS. https://github.com/RVC-Boss/ GPT-SoVITS, 2024. 4
- [24] Ruohao Guo, Liao Qu, Dantong Niu, Yanyu Qi, Wenzhen Yue, Ji Shi, Bowei Xing, and Xianghua Ying. OpenVocabulary Audio-Visual Semantic Segmentation. In ACM Int. Conf. Multimedia, 2024. 2
- [25] Junwen He, Yifan Wang, Lijun Wang, Huchuan Lu, JunYan He, Jin-Peng Lan, Bin Luo, and Xuansong Xie. Multimodal Instruction Tuned LLMs with Fine-grained Visual Perception. In IEEE Conf. Comput. Vis. Pattern Recog.,

2024. 3

- [26] Shuting He and Henghui Ding. Decoupling Static and Hierarchical Motion Perception for Referring Video Segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [27] Miran Heo, Sukjun Hwang, Jeongseok Hyun, Hanjung Kim, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. A Generalized Framework for Video Instance Segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 6
- [28] John R Hershey, Zhuo Chen, Jonathan Le Roux, and Shinji Watanabe. Deep clustering: Discriminative embeddings for segmentation and separation. In IEEE Int. Conf. Acoust. Speech Signal Process., 2016. 7
- [29] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In Int. Conf. Learn. Represent., 2022. 6

- [30] Chao Huang, Yapeng Tian, Anurag Kumar, and Chenliang Xu. Egocentric Audio-Visual Object Localization. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 3
- [31] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video Object Segmentation with Language Referring Expressions. In ACCV, 2019. 2, 6, 8
- [32] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment Anything. In Int. Conf. Comput. Vis., 2023. 3, 5
- [33] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: Reasoning Segmentation via Large Language Model. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2, 3, 5, 6, 7, 8
- [34] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. TVQA: Localized, Compositional Video Question Answering. In Proc. of the Conf. on Empirical Methods in Nat. Lang. Process., 2018. 3, 7
- [35] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, JiRong Wen, and Di Hu. Learning to Answer Questions in Dynamic Audio-Visual Scenarios. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 3
- [36] Guangyao Li, Henghui Du, and Di Hu. Boosting Audio Visual Question Answering via Key Semantic-Aware Cues. In ACM Int. Conf. Multimedia, pages 5997–6005, 2024.
- [37] Jia Li, Wenjie Zhao, Ziru Huang, Yunhui Guo, and Yapeng Tian. Do Audio-Visual Segmentation Models Truly Segment Sounding Objects? arXiv, 2025. 3
- [38] Xiang Li, Jinglu Wang, Xiaohao Xu, Xiao Li, Bhiksha Raj, and Yan Lu. Robust Referring Video Object Segmentation with Cyclic Structural Consensus. In Int. Conf. Comput. Vis.,

2023. 8

- [39] Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, et al. Baichuan-Omni Technical Report. arXiv preprint arXiv:2410.08565, 2024. 3
- [40] Linfeng Yuan and Miaojing Shi and Zijie Yue and Qijun Chen. Losh: Long-short text joint prediction network for referring video object segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [41] Chang Liu, Henghui Ding, and Xudong Jiang. GRES: Generalized Referring Expression Segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 5, 6, 8
- [42] Chang Liu, Xudong Jiang, and Henghui Ding. Primitivenet: decomposing the global constraints for referring segmentation. Visual Intelligence, 2024. 2
- [43] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In Adv. Neural Inform. Process. Syst., 2023. 3
- [44] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 7
- [45] Shuo Liu, Kaining Ying, Hao Zhang, Yue Yang, Yuqi Lin, Tianle Zhang, Chuanhao Li, Yu Qiao, Ping Luo, Wenqi Shao, et al. ConvBench: A Multi-Turn Conversation Evaluation Benchmark with Hierarchical Ablation Capability for Large Vision-Language Models. In Adv. Neural Inform. Process. Syst. D&B, 2024. 3

- [46] Yi Luo and Nima Mesgarani. Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation. IEEE/ACM Trans. Audio Speech Lang. Process., 27

(8), 2019. 7

- [47] Juncheng Ma, Peiwen Sun, Yaoting Wang, and Di Hu. Stepping Stones: A Progressive Training Strategy for AudioVisual Semantic Segmentation. In Eur. Conf. Comput. Vis.,

2024. 3

- [48] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and Comprehension of Unambiguous Object Descriptions. In IEEE Conf. Comput. Vis. Pattern Recog., 2016. 6, 8
- [49] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In IEEE Int. Conf. 3D Vis.,

2016. 5

- [50] OpenAI. https://platform.openai.com/docs/ guides/text-to-speech, 2023. 4
- [51] OpenAI. https://openai.com/index/hellogpt-4o, 2024. 2, 3
- [52] Wenwen Pan, Haonan Shi, Zhou Zhao, Jieming Zhu, Xiuqiang He, Zhigeng Pan, Lianli Gao, Jun Yu, Fei Wu, and Qi Tian. Wnet: Audio-Guided Video Object Segmentation via Wavelet-Based Cross-Modal Denoising Networks. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 4
- [53] Renjie Pi, Jiahui Gao, Shizhe Diao, Rui Pan, Hanze Dong, Jipeng Zhang, Lewei Yao, Jianhua Han, Hang Xu, Lingpeng Kong, et al. DetGPT: Detect What You Need via Reasoning. In Proc. of the Conf. on Empirical Methods in Nat. Lang. Process., 2023. 2, 3
- [54] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust Speech Recognition via Large-Scale Weak Supervision. In Int. Conf. Mach. Learn., 2023. 6
- [55] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. PACO: Parts and Attributes of Common Objects. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 6
- [56] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. SAM 2: Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714, 2024. 4
- [57] Seonguk Seo, Joon-Young Lee, and Bohyung Han. URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark. In Eur. Conf. Comput. Vis., 2020.

- 2, 4, 6, 8

[58] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-SALMONN: Speech-Enhanced Audio-Visual Large Language Models. In Int. Conf. Mach. Learn., 2024.

- 3, 5

- [59] Luoyi Sun, Xuenan Xu, Mengyue Wu, and Weidi Xie. AutoACD: A Large-scale Dataset for Audio-Language Representation Learning. In ACM Int. Conf. Multimedia, 2024. 6

- [60] Peiwen Sun, Honggang Zhang, and Di Hu. Unveiling and Mitigating Bias in Audio Visual Segmentation. In ACM Int. Conf. Multimedia, 2024. 3
- [61] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. SALMONN: Towards Generic Hearing Abilities for Large Language Models. In Int. Conf. Learn. Represent., 2023. 3
- [62] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu. Audio-Visual Event Localization in Unconstrained Videos. In Eur. Conf. Comput. Vis., 2018. 3
- [63] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191, 2024. 3
- [64] Yefei Wang, Kaili Wang, Yi Wang, Di Guo, Huaping Liu, and Fuchun Sun. Audio-Visual Grounding Referring Expression for Robotic Manipulation. In IEEE Int. Conf. Robot. Autom., 2022. 3
- [65] Yaoting Wang, Weisong Liu, Guangyao Li, Jian Ding, Di Hu, and Xi Li. Prompting Segmentation with Sound Is Generalizable Audio-Visual Source Localizer. In AAAI,

2024. 8

- [66] Yaoting Wang, Weisong Liu, Guangyao Li, Jian Ding, Di Hu, and Xi Li. Prompting Segmentation with Sound Is Generalizable Audio-Visual Source Localizer. In AAAI,

2024. 3

- [67] Yaoting Wang, Peiwen Sun, Yuanchao Li, Honggang Zhang, and Di Hu. Can Textual Semantics Mitigate Sounding Object Segmentation Preference? In Eur. Conf. Comput. Vis., 2024.
- [68] Yaoting Wang, Peiwen Sun, Dongzhan Zhou, Guangyao Li, Honggang Zhang, and Di Hu. Ref-AVS: Refer and Segment Objects in Audio-Visual Scenes. In Eur. Conf. Comput. Vis.,

2024. 2, 3, 4, 6, 7, 8

- [69] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as Queries for Referring Video Object Segmentation. In IEEE Conf. Comput. Vis. Pattern Recog.,

2022. 8

- [70] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. VISA: Reasoning Video Object Segmentation via Large Language Models. In Eur. Conf. Comput. Vis., 2024. 2, 3, 4, 6, 8
- [71] Shilin Yan, Renrui Zhang, Ziyu Guo, Wenchao Chen, Wei Zhang, Hongyang Li, Yu Qiao, Hao Dong, Zhongjiang He, and Peng Gao. Referred by Multi-Modality: A Unified Temporal Transformer for Video Object Segmentation. In AAAI, 2024. 3, 7
- [72] Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. AVQA: A Dataset for Audio-Visual Question Answering on Videos. In ACM Int. Conf. Multimedia, 2022. 3
- [73] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. LAVT: Language-Aware Vision Transformer for Referring Image Segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 8
- [74] Kaining Ying, Zhenhua Wang, Cong Bai, and Pengfei Zhou. Isda: Position-aware instance segmentation with deformable

attention. In IEEE Int. Conf. Acoust. Speech Signal Process.,

2022. 6

- [75] Kaining Ying, Qing Zhong, Weian Mao, Zhenhua Wang, Hao Chen, Lin Yuanbo Wu, Yifan Liu, Chengxiang Fan, Yunzhi Zhuge, and Chunhua Shen. CTVIS: Consistent Training for Online Video Instance Segmentation. In Int. Conf. Comput. Vis., 2023. 6
- [76] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI. In Int. Conf. Mach. Learn., 2024. 3
- [77] Kaining Ying, Hengrui Hu, and Henghui Ding. MOVE: Motion-guided few-shot video object segmentation. In Int. Conf. Comput. Vis., 2025. 6
- [78] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling Context in Referring Expressions. In Eur. Conf. Comput. Vis., 2016. 6, 8
- [79] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. MOTR: End-to-End Multiple-Object Tracking with Transformer. In Eur. Conf. Comput. Vis., 2022. 6
- [80] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. In Proc. of the Conf. on Empirical Methods in Nat. Lang. Process., 2023. 5
- [81] Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Yu Qiao, Hongsheng Li, and Peng Gao. LLaMA-Adapter: Efficient Fine-tuning of Large Language Models with Zeroinitialized Attention. In Int. Conf. Learn. Represent., 2024. 3
- [82] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest. arXiv preprint arXiv:2307.03601, 2023. 3
- [83] Tao Zhang, Xingye Tian, Yu Wu, Shunping Ji, Xuebo Wang, Yuan Zhang, and Pengfei Wan. DVIS: Decoupled Video Instance Segmentation Framework. In Int. Conf. Comput. Vis., 2023. 6
- [84] Rongkun Zheng, Lu Qi, Xi Chen, Yi Wang, Kun Wang, Yu Qiao, and Hengshuang Zhao. ViLLa: Video Reasoning Segmentation with Large Language Model. arXiv preprint arXiv:2407.14500, 2024. 2, 6
- [85] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene Parsing through ADE20K Dataset. In IEEE Conf. Comput. Vis. Pattern Recog., 2017. 6
- [86] Jinxing Zhou, Jianyuan Wang, Jiayi Zhang, Weixuan Sun, Jing Zhang, Stan Birchfield, Dan Guo, Lingpeng Kong, Meng Wang, and Yiran Zhong. Audio-Visual Segmentation. In Eur. Conf. Comput. Vis., 2022. 2, 3, 8
- [87] Jiawen Zhu, Zhi-Qi Cheng, Jun-Yan He, Chenyang Li, Bin Luo, Huchuan Lu, Yifeng Geng, and Xuansong Xie. Tracking with Human-Intent Reasoning. arXiv, 2023. 8

