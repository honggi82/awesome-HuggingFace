# arXiv:2412.09283v1[cs.CV]12Dec2024

### InstanceCap: Improving Text-to-Video Generation via Instance-aware Structured Caption

###### Tiehan Fan1* Kepan Nan1* Rui Xie1 Penghao Zhou2 Zhenheng Yang2 Chaoyou Fu1 Xiang Li3 Jian Yang1 Ying Tai1

1 Nanjing University 2 ByteDance 3 Nankai University https://github.com/NJU-PCALab/InstanceCap

###### Abstract

Text-to-video generation has evolved rapidly in recent years, delivering remarkable results. Training typically relies on video-caption paired data, which plays a crucial role in enhancing generation performance. However, current video captions often suffer from insufficient details, hallucinations and imprecise motion depiction, affecting the fidelity and consistency of generated videos. In this work, we propose a novel instance-aware structured caption framework, termed InstanceCap, to achieve instance-level and fine-grained video caption for the first time. Based on this scheme, we design an auxiliary models cluster to convert original video into instances to enhance instance fidelity. Video instances are further used to refine dense prompts into structured phrases, achieving concise yet precise descriptions. Furthermore, a 22K InstanceVid dataset is curated for training, and an enhancement pipeline that tailored to InstanceCap structure is proposed for inference. Experimental results demonstrate that our proposed InstanceCap significantly outperform previous models, ensuring high fidelity between captions and videos while reducing hallucinations.

###### 1. Introduction

Recently, text-to-video (T2V) generation with advanced diffusion transformers (DiT) [2, 8, 10–12, 15, 18, 21, 23, 25, 32, 36] have attracted significant attention for the ability to generate realistic, long-duration videos based on text prompts. Video-caption paired data is typically used in training and plays a crucial role in enhancing generation performance. Current video recaption methods often incorporate multimodal large language models to produce detailed captions, which however usually suffer from hallucinations, leading to inconsistencies between captions and

* Equal contributions. Ying Tai is the corresponding author.

video content. Consequently, creating consistent videocaption pairs with accurate details and precise motion depiction for T2V generation remains a significant challenge.

As shown in Figure 1, current video recaption methods can be broadly categorized into three types: 1) Short captions, such as Panda-70M [4], lack sufficient coverage of video content, leading to low fidelity. 2) Dense captions, like ShareGPT4Video [3], enrich textual content but suffer from hallucination issues, often generating meaningless or inaccurate video content. 3) Coarse-level structured captions, exemplified by MiraData [9], improve video quality but provide coarse-level details. Moreover, the redundancy introduced by MLLM across structures diminishes its overall effectiveness. To this end, achieving accurate captions remains two crucial challenges: 1) High fidelity between caption and video: Retain as much of the original video’s objects, textures, and motion information as possible. 2) Accurate content in caption: Enable MLLM model to generate precise content, minimizing hallucinations and repetition.

To address the challenges, we propose a novel instanceaware structured caption framework, termed InstanceCap, to achieve instance-level and fine-grained video caption for the first time. Our structure is specifically designed to incorporate instances, background, and camera movement. For each instance, we specify class, appearance, actions, motion, and position. To enhance the fidelity and accuracy of video captions, we focus on two key aspects: 1) From Global Video to Local Instances: For each instance, we propose an auxiliary models cluster (AMC) to isolate it from the original video and obtain the corresponding position and category information. This operation minimizes interference from unrelated regions while retaining as much of the original video’s information as possible. 2) From Dense prompt to Structured Phrases: We leverage multimodal large language models (MLLMs) in an improved Chain-of-Thought (CoT) process to obtain concise yet accurate descriptions of textures, camera movement, actions and motion for each instance. This reduces the probability of hallucinations and irrelevant content produced by the lan-

|[PANDA-70M](CVPR 2024) A woman wearing glasses anda blue tie is sitting on a red couch.<br><br>[ShareGPT4Video](NeurIPS 2024) The video features an individual standing in front of a red background,engaging in what appears to be a presentation or discussion. Throughout the video, the person is seen making various gestures with their hands, indicating a dynamic and expressive delivery. Initially, the person's hands are raised, palms facing upwards, suggesting an explanatory or emphatic gesture. As the video progresses, the individual's gestures evolve; the hands are lowered and then raised again, with the fingers pointing upwards, indicating a shift in the emphasis or point of discussion. Subsequently, the hands are brought closer together, with the fingers interlocked, suggesting a moment of emphasis or a transition in the presentation. Finally, the hands are raised again, with the fingers spread apart, indicating a continuation of the expressive gestures. Throughout the video, the individual's attire remains consistent, featuring a dark-colored top with a<br><br>lighter-colored collar, and the background remains a solid red,<br><br>providing a stark contrast that keeps the focus on the speaker. The camera angle and framing stay constant, with no discernible movement, maintaining a steady focus on the individual and their gestures. The lighting is even, ensuring that the subject is clearly visible against the red backdrop. There are no changes in the environment, the individual's attire, or the camera's perspective, indicating a continuous and uninterrupted presentation or discussion.<br><br>[MiraData](NeurIPS 2024) In this video, a person with blond hair is engaging in an animated conversation while seated. They are wearing glasses and a striped shirt and express themselves through a series of dynamic hand gestures and head movements. The individual appears to be deeply focused on the dialogue, occasionally looking down or away, possibly to emphasize their points or formulate thoughts. The person’s gestures escalate throughout the video, suggesting an increasing intensity or passion about the topic. The main subject, identifiable by their blond hair and glasses, transitions from initially looking down to making various expressive hand gestures and head movements. Their body language involves considerable arm movement, indicating engagement and enthusiasm in the conversation. The sequence begins with the individual in a somewhat reserved posture, escalating to more expansive and assertive gestures, before returning to a composed state. The background of the video is a studio setting with prominent red and white color blocks, suggesting a professional and possibly formal environment.Part of a sign or text that reads "RVI" is visible,<br><br>indicating a possible connection to a broadcast or presentation context. The<br><br>exact location isn’t specified, but the controlled indoor setting implies a purposeful and staged production. The view shot remains consistent, focused on the upper body of the main subject, ensuringtheir gesturesand facial expressions are clearly visible. The camera maintains a mid-shot with minimal movement, emphasizing the individual's actions and expressions. There are no noticeable changes in the shooting angle, keeping the subject centered throughout the frames to convey their animated discourse. The video employs a straightforward, television broadcast style, emphasizing clear and focused framing to capture the subject’s expressive communication in a professional studio environment.<br><br>Wrong Ambiguity Detailed<br><br>[Figure 1]<br><br>A<br><br>B<br><br>C<br><br>[Figure 2]<br><br>[Figure 3]<br><br>|[Figure 4]<br><br>[Figure 5]<br><br>A<br><br>B<br><br>C<br><br>Ground Truth|
|---|
<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>InstanceCap(Ours) ShareGPT4Video MiraData<br><br>-<br><br>}<br><br>-<br><br>} }<br><br>[InstanceCap](Ours) {<br><br>"Global Description": "A woman with blonde hair, wearing a dark blue shirt with white stripes, is speaking and gesturing with her hands.",<br><br>"Structured Description": { "Main Objects": {<br><br>"No.0": { "Class": "person", "Appearance": "The person in the video is a woman with blonde hair, wearing glasses, her face is to the right of the frame. She is dressed in a dark blue blouse with white stripes. The woman appears to be middle-aged and is of Caucasian race. She is wearing a necklace and has a ring on her finger . Her facial expressions change throughout the video, indicating that she is engaged in a conversation or presentation.",<br><br>"Actions and Motion": "Her gestures are expressive, with her hands moving fluidly to emphasize her points and she is holding a pen . The way she speaks suggests she is passionate about the topic at hand, her body language open and inviting. ",<br><br>"Position": "person stays at bottom-middle." }<br><br>}, "Background Detail": "The environment in the video is a stage or presentation area with a grand and elegant atmosphere. The backdrop consists of a red curtain, which adds a touch of<br><br>drama and sophistication to the setting. Text square "ROVO" can be seen in the lower right<br><br>corner. The lighting is bright and evenly distributed, illuminating the stage and creating a clear and focused environment for the speaker. The overall ambiance is professional and formal, indicative of a well-organized and prestigious event.",<br><br>"Camera Movement": "The camera is static, maintaining a short distance shot throughout the video. The shooting angle remains consistent, capturing the speaker from the waist up."<br><br>} }<br><br>A B<br><br>C<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>Panda<br><br>Figure 1. Top: Comparison of the reconstruction-via-recaption results between InstanceCap and state-of-the-art captioning methods for annotating the ground truth video. InstanceCap produces results that more closely resemble the original video, showing greater detail fidelity (highlighted by the red circle). Bottom: The corresponding captions generated by InstanceCap and others. Red denotes incorrect captions, blue represents ambiguous captions, and green indicates detailed and accurate descriptions of video. Specific visual hints are marked as A, B, and C for clarity. All videos are generated using the same video generation product, Hailuo AI1, which has robust prompt-<br><br>|
|---|

​ ​ ​ ​

​ ​ ​ ​

​

​

​ ​ ​ ​

​ ​ ​ ​

​

​

following capabilities, clearly highlighting the effectiveness of InstanceCap.

guage model compared to traditional caption methods that directly describe video content in a complex, dense caption.

To validate the effectiveness of InstanceCap in T2V generation, we constructed a high-definition video dataset comprising 22K samples to create a training dataset with our instance-aware structured captions, named InstanceVid. At the inference stage, we also implemented a prompt enhancement pipeline tailored to our structured captioning method, enabling the generation of concise captions that better align with user needs. Our InstanceCap integrates seamlessly with existing diffusion models. Experimental results demonstrate that after finetuned with our InstanceVid, the T2V model exhibits better ability with prompt following on details and motion actions. In summary, our main contributions are as follows:

- • We propose InstanceCap, the first instance-aware structured caption method for text-to-video generation. We also constructed a 22K InstanceVid dataset during training, and developed an enhancement pipeline that tailored to InstanceCap structure during inference.
- • We design the AMC paradigm to convert global video

into instances, enhancing instance fidelity. Additionally, we propose an improved CoT pipeline with MLLMs to refine dense prompts into structured phrases, achieving concise yet precise descriptions.

• Extensive experiments on video reconstruction demonstrate that our InstanceCap significantly enhances the fidelity between captions and videos. T2V models finetuned on our InstanceVid further achieve more precise generation on instance details and motion actions.

###### 2. Related works

Video recaptioning. Advancements in text-to-video generation demand high-quality video-text datasets to build robust foundational video models for visual-language alignment. Current video recaption methods fall into two main categories: manual annotation [1, 24, 37] and end-to-end recaption using multimodal large language models (MLLM). Although manual annotation provides higher accuracy, scaling datasets to meet the needs of high-quality video generation models remains a substantial challenge. Recent

1https://hailuoai.com/video

advances in MLLM have demonstrated impressive capabilities in video understanding and description generation. Panda [4] and InternVid [26] are with short captions, offering computational efficiency but frequently omitting crucial content and exhibit low video fidelity. OpenVid1M [16], Vript [28] and ShareGPT4Video [3] are with dense captions, which provide richer content but face challenges: Hallucinations due to complexity, inclusion of redundant information, and text encoder truncation caused by excessively long text. MiraData [9] achieves coarselevel structured captions that attempts to mediate these issues but struggle with fine detail and redundancy across structures. Different from the existing video recaption methods, InstanceCap is the first instance-aware structured caption approach for text-to-video generation, ensuring high fidelity between caption and video while reducing hallucinations and repetition.

Text-to-video generation. Despite numerous highquality video generation models [2, 8, 10, 11, 18, 30] perform well with simple directives, they often falter with complex prompts requiring precise instance-level details or intricate camera movements. Analyzing current video-text datasets suggests these limitations may stem from suboptimal training data quality. Traditional recaption methods have not sufficiently captured instance-specific detail granularity or provided comprehensive descriptions of camera movements. To enhance instance-level details and motion consistency, we construct a 22K InstanceVid dataset for training/finetuning, and develope an enhancement pipeline InstanceEnhancer that tailored to the proposed instance-aware structure during inference.

###### 3. Method

In Section 3.1, we first present the InstanceCap pipeline, as shown in Figure 2. Based on this pipeline, we recaption the carefully curated dataset InstanceVid in Section 3.2, enhancing T2V models’ instance generation. Additionally, in Section 3.3, we introduce InstanceEnhancer to convert short prompts into our proposed instance-aware structured caption format during inference.

###### 3.1. InstanceCap

Video preprocessing with auxiliary model cluster. For continuous video processing, we implemented uniform sampling using decord2, following the methodology established in LLaVA-Video [34]. This approach enables us to extract essential temporal metadata, including duration, frame count, and timestamps, allowing MLLMs to better interpret temporal sequences in recaptioning tasks. Additionally, to enhance MLLMs’ capabilities through structured guidance, we incorporate several auxiliary models to

2https://github.com/dmlc/decord

achieve accurate object detection, video instance segmentation and camera motion prediction, providing precise prior information to the MLLMs.

Global description, background detail and camera movement. When describing video content, a highquality global description should capture primary elements, environmental context, camera movements, angles, and tonal qualities. MLLMs excel in generating high-level video summaries using Chain-of-Thought methodology. By employing carefully designed prompts with CoT, we can guide MLLMs to produce detailed background descriptions while minimizing references to foreground elements.

However, MLLMs’ limitations in processing discrete frames rather than continuous video segments make it challenging to distinguish camera motion from instance action. To address this, we achieve camera annotations from OpenSora [35] for basic movements (e.g., zoom, rotation) and rely on MLLMs to capture subtle motion attributes (e.g., intensity, speed). The integration of camera movement indicators with MLLM capabilities provides comprehensive annotations, as illustrated in Figure 10 (a).

Structured Description on Instances. In this subsection, we introduce the details to achieve our instanceaware structured description. To address MLLMs’ limitations in instance annotation and the suboptimal results of directly adapting weak visual prompts from images to videos [22, 31], we make full use of the auxiliary model cluster, including initial object detection [38], video instance segmentation with SAM2 [20], and blur non-instance regions to achieve blur background, resulting in better recaptioning outcomes compared to alternative visual prompt methods in video, as shown in Figure 11. To this end, we decompose the global videos into local instances.

Next, we describe how to achieve detailed and accurate description of each instance (Figure 3). To maintain instance-level precision, we deliberately constrain the information accessible to MLLMs during instance annotation. Crucially, the global video remains invisible due to our designed blurred backgrounds, preventing MLLMs from confusing information across multiple instances. This approach allows InstanceCap to focus on local instances identified through auxiliary model cluster. Furthermore, to avoid the potential limitation of MLLMs seeing only isolated instances, which could lead to overlooking inter-instance interactions and subsequent misinterpretations, we incorporate the global description mentioned in previous subsection. Specifically, we inject the global description into the instance-annotation MLLMs. This strategic mitigates potential biases in instance action descriptions while maintaining instance-specific accuracy.

To enhance the capability of InstanceCap in capturing instance-level details, we introduce novel insights into the improved CoT process. Our analysis of current MLLM-

###### Auxiliary Model Cluster

###### Prior Information

###### Results

{ "Global Description": "A man in a striped shirt and shorts skillfully surfs on a wave.",

[Figure 12]

"Structured Description": { "Main Instance": {

[Figure 13]

[Figure 14]

"No.0": { "Class": "person", "Appearance": "The person in the video is wearing a horizontally striped shirt

Camera Motion Detector

with alternating shades of white and dark colors, paired with navy blue shorts. The individual appears to be male, with short hair, and is not wearing any visible accessories such as necklaces or watches.",

Object Detection

Position Class

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

"Actions and Motion": "The individual appears to be engaged in an action that

[Figure 19]

[Figure 20]

person

involves balancing and movement, consistent with surfing. Their body is angled in a way

that suggests they are riding a wave, with one arm extended for balance and the other bent at the elbow. The person's legs are bent, and their feet are positioned wide apart, indicating a stance that is both stable and ready to adjust to the motion of the water.",

[Figure 21]

[Figure 22]

-left to bottom-middle." },

"Position": "person moves from bottom-left to bottom-middle."

SAM2

}, "No.1": {

surfboard

From Global videos to Local instances

"Class": "surfboard", "Appearance": "The surfboard in the video is white with a black stripe running

Positive/Negative Lexicon

down the middle. The text on the surfboard reads \"SOFTOPS\" in green letters.",

"Actions and Motion": "The surfboard is in motion, gliding smoothly over the water. The board's nose occasionally pierces through the wave, and its tail follows the curve of the wave, maintaining balance. The surfboard's white and black stripes are clearly visible, and it appears to be a longboard style, suitable for the waves present in the

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

video.",

"Position": "surfboard moves from bottom-left to bottom-middle."

-left to bottom-middle."

Chain-of-Thought- Prompt

} },

}

}, "Background Detail": "The environment in the video is a vibrant and dynamic ocean

scene. The water is a clear, turquoise blue-green color, with white, frothy waves crashing and splashing around. The sunlight reflects off the surface of the water, creating a sparkling effect. The waves are in constant motion, with some cresting and others breaking, indicating a strong current or wind. The overall atmosphere is energetic and refreshing, with the sound of the waves implied by their appearance.",

-

MLLMs

Appearance Actions and Motion

From Dense prompts to Structured phrases

[Figure 35]

[Figure 36]

"Camera Movement": "The camera movement in the video is a zoom-in, which occurs gradually and smoothly. The shooting angle appears to be at medium distance, capturing the surfer and the wave from a side perspective. The camera does not move rapidly or sharply; instead, it maintains a steady focus on the surfer as they ride the wave."

-

Human-Designed Class Hints

Background And

LLaVA LLaVA-Video Qwen-VL GPT-4o

Camera

...

}

}

Figure 2. Overview of InstanceCap pipeline. Details of “from dense prompts to structured phrases” design are shown in Figure 3.

From Dense prompts to Structured phrases

Blur Background Video

Appearance Actions and

[Figure 37]

[Figure 38]

[Figure 39]

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

[Figure 53]

[Figure 54]

[Figure 55]

MLLMs

Motion

Appearance

###### Actions and motion

Background And Camera

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

MLLMs

Camera Movement

Background Detail

Global Description

Global Video

Human-Designed Class Hints

Positive/Negative Lexicon

Chain-of-Thought Prompt

Camera Movement Hints

[Figure 63]

- Figure 3. Details on “from dense prompts to structured phrases” design. We propose an improved CoT pipeline with carefully designed information interactions (red arrow), which facilitates MLLMs to accurately capture instances with precise descriptions on attributes.

based video recaption methods shows that simple prompts, like “Please provide a detailed description of this video” or Chain-of-Thought prompts “Let’s think step by step... First, please note... Finally, summarize the video content...” fail to capture precise instance details. Additional experiments reveal that MLLMs can effectively annotate these details when given fine-grained prompts, such as “ Please note if the characters have any accessories” or “Please observe whether there are spots on the bananas”. To enhance the details of instances, we develope Human-designed Class Hints, crafting specific prompts for about 80 detectable categories using our auxiliary models cluster. Specifically, we present the “person” class prompt here: “Please focus primarily on the person’s facial expressions, attire, age, gender, and race in the video and give a detailed description. Please mention if there are any necklaces, watches, hat or other decoration; otherwise, there’s no need to bring them up.” Besides, we also developed a curated Posi-

tive/Negative Lexicon to guide MLLMs in generating more aesthetically refined captions. More details can be found in our supplementary material.

###### 3.2. InstanceVid

Data collection. InstanceVid is curated via refining a subset from the high-aesthetic, high-consistency videos from OpenVid-1M [16]. To showcase our method’s highfidelity labeling of instance details and motion, we selected video samples that included at least one instance exhibiting high motion intensity during dataset filtering.

Statistical analysis of InstanceVid. Figure 4 illustrates the statistical characteristics of InstanceVidacross two main dimensions: video scenes, and temporal durations. Our data collection emphasizes videos with distinct instances while ensuring a balanced representation of outdoor scenes to prevent biases from an overemphasis on

Duration（s） Scene

[Figure 64]

[Figure 65]

Main Instance Identification

Short Prompt Long Prompt

Structured Prompt

Stage A Stage B(I)&(II)

- Figure 5. High-level overview of InstanceEnhancer, illustrating the data flow and the partitioning of stages. For a detailed implementation, refer to the supplemental materials, which provide an in-depth description of the enhancer pipeline design and the interdependencies between the stages.

| |
|---|

| |
|---|

[Figure 66]

[Figure 67]

[Figure 68]

| |
|---|

| |
|---|

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

| |
|---|

| |
|---|

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Ours - 131.125

glasses grey sweater correct relative position

Ground Truth

MiraData - 145.875

- Figure 6. Comparison on reconstruction-via-recaption between InstanceCap and MiraData. Corresponding 3DVAE scores are also indicated. Similar semantics shared between InstanceCap and GT are indicated by red circles and lines.

- Figure 4. InstanceVid provides structured captions for videos in open-domain scenarios, featuring diverse instance, expansive scenes, precise and instance-aware captions, and videogeneration-friendly durations.

instance-focused content. We achieve detailed descriptions capturing human movements, physical appearances, and documentation of common objects and animals. Besides, InstanceVid focuses on short-duration videos (2-10 seconds) for two main reasons. First, OpenVid-1M segments longer sequences to eliminate excessive scene transitions. Second, most of the current open-source T2V models are optimized for video generation within this duration range.

###### 3.3. InstanceEnhancer

When the caption distribution of training data differs from that of inference text, it may result in poor instructionfollowing performance or even problematic outputs. This issue is evident in T2V generation, particularly when long captions are used for training but short captions for inference, leading to subpar results. Since users typically prefer short captions, it is essential to enhance short caption effectively to better align with our proposed instance-aware structured caption during training.

###### 4. Experiments

###### 4.1. Experimental setup

Video reconstruction with recaptions. To comprehensively evaluate InstanceCap, we conducted a series of experiments, benchmarking against state-of-the-art methods including Panda-70M [4], ShareGPT4Video [3], and MiraData [9]. To this end, we carefully selected 100 video clips from OpenVid-1M [16] and Animal Kingdom [17]. For each video, we generated one caption using various caption models, which were then input into the advanced T2V model CogvideoX-5b [30] for video generation. We calculated the differences between the generated videos and the ground truth videos to evaluate each caption model’s performance, where smaller visual differences indicate more accurate captions and higher fidelity.

As shown in Figure 5, we introduce a tuning-free approach called InstanceEnhancer that achieves this by strictly limiting the generated formats to match the caption corresponding to the training input we used. Our method differs from existing tuning-free caption enhancement approaches, such as those presented in RPG [29]. Instead of directly enhancing short captions, which we found can introduce inconsistencies between multiple instances’ actions and their environmental context in video generation, we employ a two-stage enhancement strategy. In Stage A, short prompts are expanded into detailed long prompts. Stage B(I)&(II) uses both expanded and original captions to segment and enhance specific instances, preserving contextual coherence while ensuring precise instance identification. Due to the space limitation, more details of our enhance pipeline can be found in the supplementary material.

We introduced several metrics to evaluate the video reconstruction performance: 1) 3DVAEscore: Using 3DVAE from CogVideoX [30] as the backbone, we extract hiddenspace representations from both the original videos and their recaption-reconstructed counterparts. These representations quantify the perceptual distance between them. 2) CLIPSenbySen: To handle CLIP’s 77-token processing limit, we segment long captions into individual sentences

|[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>| |
|---|
| |
<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>| |
|---|
<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>Ours<br><br>| |
|---|
<br><br>| |
|---|
| |
<br><br>Multiple Action: A fish swims and a turtle moves on the seabed.<br><br>OpenSora<br><br>Ours<br><br>| |
|---|
<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>OpenSora<br><br>| |
|---|
<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>Single Action: A swift lion sprints across the plain.<br><br>Figure 7. Visual comparison of InstanceCap and Opensora on Single and Multiple Action Score. In terms of the dynamic degree of video generation, we show better consistency and enhanced multi-instance dynamic generation effect.<br><br>and compute CLIP [19] similarity between each sentence and every original video frame. The final score is obtained by first averaging the similarity scores of each sentence across all frames, then averaging these sentence-level scores for a comprehensive result. 3) Human Evaluation: We con-<br><br>information, leading to higher fidelity. For instance, our InstanceCap can retain information such as “glasses”, “grey sweater”, and “relative position of two people”, whereas MiraData [9] almost completely loses these important details. A similar conclusion can be drawn from|
|---|

ducted a user study with a panel of evaluators to assess caption quality across two aspects: Instance Detail (ID) and Hallucination Scores (HS).

Figure 1. These results underscore the significant improvements achieved by our InstanceCap, resulting in highquality reconstruction characterized by rich detail and high fidelity between our captions and the original videos.

T2V generation. To thoroughly evaluate the T2V generation performance of our InstanceCap, we utilize the InstanceVid dataset to finetune the state-of-the-art DiT-based T2V generation model Open-Sora [35]. In our evaluation, we compare with Open-Sora, CogVideoX5b [30], Pyramid-Flow [8], and Open-Sora-Plan [11]. To enable fine-grained, instance-level assessment, we construct a highly challenging evaluation benchmark called Inseval, inspired by recent advancements in T2I and T2V evaluation [5, 6, 13, 14, 27]. Specifically, we curate a diverse evaluation dataset of over 200 carefully crafted instancelevel prompt-answer pairs, covering both single-object and multi-object scenarios systematically across five key dimensions: Action, Color, Shape, Texture, and Detail. Motivated by the previous evaluation benchmarks [6], we implement a CoT reasoning framework for generating structured QA responses to ensure objective and consistent evaluation, allowing us to derive instance-level evaluation scores that align closely with human perception and preferences. This approach provides a more nuanced and reliable assessment of instance-level generation quality.

Quantitative evaluation. Table 1 presents quality comparisons between our InstanceCap and other caption methods across two metrics. Based on the results, we make the following observations: 1) Our method delivers comparable or superior quality to the four baselines, demonstrating its ability to enhance fidelity between videos and captions. This strong alignment with human perceptual judgments and preferences is evident in Figures 1 and 6. 2) Our method consistently excels across all metrics for captions under 200 words, an acceptable length for most T2V models, illustrating its generalizability. Figure 8 compares MiraData’s captions with our instance-aware structured ones. We randomly selected videos from an opendomain dataset, and a panel of evaluators assessed caption quality using standardized criteria. Results indicate that our captions offer significantly richer and more accurate descriptions while reducing hallucination artifacts compared to MiraData’s output.

###### 4.3. Text-to-video generation

Qualitative evaluation. Figures 7 and 9 provide visual comparisons of the T2V generation results. It can be observed that the infusion of our InstanceVid dataset into Open-Sora [35] serves to further enhance its video synthesis capabilities across four fundamental dimensions (Action, Color, Shape, and Texture). These four different aspects correspond to information in our instance-aware structured captions such as “Actions and Motion”, “Appearance”, etc. For instance, our model accurately generates the “sprints” action of the lion in Figure 7, as opposed to Open-

###### 4.2. Comparison with SOTA caption models

Qualitative evaluation. CogVideoX-5b [30] is a latent text-to-video generation model known for its capability to generate realistic, long-duration videos based on text prompts. With the integration of our InstanceCap into CogVideoX-5b, as substantiated by Figure 6, the model exhibits a notable enhancement in the video reconstruction capacity. This demonstrates that our instance-aware structured captions retain more of the original video’s

Captioning Methods 3DVAEscore↓ CLIPSenbySen↑ Avg. Length Panda-70M 140.25 0.1956 13 words ShareGPT4Video 141.00 0.2132 191 words LLaVA-Video-72B 139.88 0.2060 102 words MiraData(GPT-4o) 137.50 0.2156 263 words InstanceCap(Ours) 134.25 0.2133 157 words

- Table 1. Quantitative comparisons on reconstruction-via-recaption results. The best results are marked in bold, and the second-best are underscored. As a reference, CogVideoX-5b accepts 226 text tokens, with any excess being truncated.

3.21

4.60

Instance Detail

3.35

4.12

Hallucination Scores

Figure 8. User study on instance detail and hallucination scores. Our instance-aware structured caption shows clear advantages compared to the coarse-structured MiraData [9].

T2V Model Single↑ Multiple↑ Average↑ Action Color Shape Texture Detail Action Color Texture

CogVideoX-5B [30] 64% 60% 44% 60% 20% 8% 48% 40% 43.00% Pyramid-Flow-2B [8] 44% 68% 32% 32% 7% 4% 24% 16% 28.38%

Open-Sora Plan v1.3-2.7B [11] 64% 44% 36% 32% 27% 20% 32% 12% 33.38% Open-Sora v1.2-1.1B [35] 40% 56% 36% 40% 13% 12% 16% 16% 28.63% + InstanceCap(Ours) 56% 60% 40% 48% 27% 16% 32% 24% 37.88% + Panda-captioner [4] 40% 48% 28% 40% 20% 8% 20% 12% 27.00% + ShareGPT4Video [3] 40% 44% 32% 24% 13% 16% 8% 20% 24.63% + LLaVA [16] 52% 52% 28% 28% 20% 12% 28% 16% 29.50%

- Table 2. Quantitative comparison between InstanceCap and SOTA video captioning models, all based on the popular T2V model OpenSora. Additionally, we also compare three powerful T2V models, including CogVideoX-5B, Pyramid-Flow, and Open-Sora Plan. The best results of video captioning methods and Open-Sora are marked in bold, and the second-best are underscored.

bag” instance described in caption, where Open-Sora [35] completely loses this instance. These results indicate that our InstanceVid can provide accurate and instance-level guiding information for video generation models.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

###### Ours OpenSora

Quantitative evaluation. We conduct the quantitative evaluation for InstanceCap using the proposed Inseval metrics in Table 2. We can draw the following conclusions: 1) Fine-tuning with InstanceVid consistently improves all metrics over the base model Open-Sora, demonstrating the effectiveness of InstanceCap. In particular, our Detail score ranks first, justifying the capacity of InstanceCap to capture complex instance detail in video. 2) Compared to other video captioning models finetuned based on OpenSora, InstanceCap shows clear advantages in video generation tasks. 3) Compared to larger models like CogVideoX or Pyramid-Flow, our approach achieves a higher average metric than Pyramid-Flow, and performs comparably to CogVideoX in several specific metrics like ‘SingleColor/Shape/Detail’ and ‘Multiple-Action’, but with much fewer parameters.

Multiple Color: A sapphire blue keychain hangs from a light brown bag.

| | | |
|---|---|---|

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

###### Ours OpenSora

| | |
|---|---|

Single Shape: A hexagonal helmet rests on the seat of the bench, near the window.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Ours OpenSora

| | | |
|---|---|---|

Single Texture: A smooth paper bag rests on the counter near the coffee machine.

- Figure 9. Visual comparison of InstanceCap and Open-Sora on instance-level attributes. InstanceCap excels in precise instance detail fidelity and instruction-following capabilities, even with complex multi-instance and multi-attribute scenarios.

###### 4.4. Ablation Study

Effects of human design class hints and camera movement hints. We discuss the impact of incorporating human-designed class hints and camera movement hints on annotation outcomes and provide relevant caption visualizations in Figure 10. These annotations aid MLLMs in focusing more precisely on key elements, resulting in richer and

Sora [35]. In Figure 9, benefiting from our instance-aware caption, our model generates the accurate “light brown

| |
|---|

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

- （a）
- （b）

zoom in

| |
|---|

w/o hints w/ hints(ours)

The camera movement in the video is smooth and steady, providing a clear and stable view of the scene.

The camera movement in the video is a steady and gradual zoom-in, focusing on a specific area of the cityscape.

| |
|---|

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

| |
|---|

w/o hints w/ hints(ours)

An older man, his hair white, is wearing a blue denim shirt and jeans. The person in the video is a middle-aged male with white hair. He is wearing a denim shirt and jeans, and he has a watch on his left wrist.

- Figure 10. (a) Ablation study on the effect of camera movement hints on the accuracy of MLLM labeling. (b) Impact of human-designed class hints on the details of instance labeling.

GroundTruth BlurBackground

(Ours)

Red-screen

Background

GroundTruth WeakVisual

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

| |
|---|

| |
|---|

[Figure 138]

[Figure 139]

[Figure 140]

Prompt BlurBackground

[Figure 141]

[Figure 142]

[Figure 143]

| |
|---|

| |
|---|

[Figure 144]

[Figure 145]

[Figure 146]

(Ours)

[Figure 147]

[Figure 148]

[Figure 149]

| |
|---|

[Figure 150]

[Figure 151]

[Figure 152]

(a) (b)

[Figure 153]

[Figure 154]

[Figure 155]

| |
|---|

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

- Figure 11. (a) Comparison against the weak visual prompt for reconstruction-via-caption visualization on multi-instance targets. (b) Comparison against color screen backgrounds (red), which may negatively affect MLLM labeling performance. more accurate annotations.

preserves visual consistency with natural scenes, enabling MLLMs to generate accurate and contextually relevant annotations with minimal prompting guidance.

Ablations on different video visual prompts. Comparative results for various video visual prompt methods used in caption generation are shown in Figure 11. As illustrated in (a), weak visual prompts derived from static image techniques, such as red circles, bounding boxes, or selective grayscale manipulation of non-target areas [22, 31], limit MLLMs in distinguishing and describe specific targets in multi-instance scenes, leading to attribute blending and vague annotations across instances. In contrast, our method excels in instance-specific feature extraction, accurately differentiating figures like the coach and players. Figure 11 (b) illustrates strong visual prompts that involve complete occlusion of non-target regions to eliminate MLLM interactions with irrelevant instances. Conventional methods use primary color screens, but this often misguides MLLMs, causing them to incorporate incorrect context in captions. Our designed blur background masking approach, however,

###### 5. Conclusions and Limitations

In this paper, we introduce InstanceCap, the first instanceaware structured caption method for text-to-video generation. We design an Auxiliary Models Cluster (AMC) to convert global video into instances, enhancing instance fidelity. we also propose an improved CoT pipeline with MLLMs to refine dense prompts into structured phrases, achieving concise yet precise instance descriptions compared to the previous video caption models. Additionally, based on InstanceCap, we curated InstanceVid dataset for training and InstanceEnhancer during inference, significantly enhancing T2V models’ generation capabilities on instance details and actions.

Limitations. Since the precision of InstanceCap partly depends on object detection methods, requiring fine-tuning of the detection model for domain-specific instances, and its benefits decrease in instance-free scenes. Furthermore, the scale of InstanceVid limits its use as a large-scale pre-training dataset. Moving forward, we plan to apply InstanceCap to a larger video dataset and train more powerful T2V models to amplify its impact.

###### References

- [1] David L. Chen and William B. Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL-2011), Portland, OR, 2011. 2
- [2] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, pages 7310–7320, 2024. 1, 3
- [3] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 1, 3, 5, 7, 2
- [4] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. arXiv preprint arXiv:2402.19479,

2024. 1, 3, 5, 7

- [5] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv:2307.06350, 2023. 6
- [6] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 6, 2
- [7] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023. 4
- [8] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 1, 3, 6, 7

- [9] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. arXiv preprint arXiv:2407.06358,

2024. 1, 3, 5, 6, 7

- [10] Kuaishou. Kling. https://kling.kuaishou.com,

2024. 1, 3

- [11] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 3, 6, 7
- [12] Jiachen Li, Qian Long, Jian Zheng, Xiaofeng Gao, Robinson Piramuthu, Wenhu Chen, and William Yang Wang. T2vturbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design. arXiv preprint arXiv:2410.05677, 2024. 1
- [13] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation, 2024. 6
- [14] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023. 6, 2
- [15] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 1
- [16] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 3,

- 4, 5, 7

[17] Xun Long Ng, Kian Eng Ong, Qichen Zheng, Yun Ni, Si Yong Yeo, and Jun Liu. Animal kingdom: A large and diverse dataset for animal behavior understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19023–19034, 2022.

- 5

- [18] Pika. Pika 1.0. https://pika.art, 2023. 1, 3
- [19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 6, 2
- [20] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3
- [21] Runway. Gen-2. https://research.runwayml. com/gen2, 2023. 1
- [22] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11987–11997, 2023. 3, 8
- [23] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. ModelScope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1
- [24] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, highquality multilingual dataset for video-and-language research,

2020. 2

- [25] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. LaVie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 1
- [26] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In The Twelfth International Conference on Learning Representations, 2023. 3
- [27] Yinwei Wu, Xianpan Zhou, Bing Ma, Xuefeng Su, Kai Ma, and Xinchao Wang. Ifadapter: Instance feature control for grounded text-to-image generation, 2024. 6
- [28] Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words, 2024. 3
- [29] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024. 5
- [30] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3, 5, 6, 7, 2
- [31] Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, TatSeng Chua, and Maosong Sun. Cpt: Colorful prompt tuning for pre-trained vision-language models. AI Open, 5:30–38,

2024. 3, 8

- [32] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. IJCV, 2024. 1
- [33] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. 2
- [34] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. 3
- [35] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 3, 6, 7
- [36] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-Sora: Democratizing efficient video production for all. https://github.com/hpcaitech/OpenSora, 2024. 1
- [37] Luowei Zhou, Nathan Louis, and Jason J. Corso. Weaklysupervised video object grounding from text by loss weighting and object interaction, 2018. 2
- [38] Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with collaborative hybrid assignments training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6748–6758, 2023. 3

### InstanceCap: Improving Text-to-Video Generation via Instance-aware Structured Caption

|Supplementary Material<br><br>present comprehenfollowing sections:<br><br>for constructing accompanied by their<br><br>compilation of humantheir diverse applica-<br><br>Chain-of-Thought<br><br>Figure 3, with particimprovements. framework of In-<br><br>exemplary prompts implementations.<br><br>discussion of the princi-<br><br>video reconstruction, and empirical valiused by Inseval in<br><br>stages.<br><br>[Figure 174]<br><br>Select the appropriate ones of following words in your description: kaleidoscopic, delicate, grand, gentle, soothing, cool, mature, solitary, worn, chaotic, dramatic, cozy, shimmering, desolate, serene, weathered, whispering, loose-fitting, vibrant,<br><br>tranquil, dimly-lit, purplish, introspective, artfully, sleek, energetic, overcast,<br><br>brilliant, slender, graceful, picturesque, whimsical, contented, gentle, warm, tender, pastel-colored, elegant.<br><br>[Figure 175]<br><br>Do not use any of the following negative words when describing: dull, rough, harsh, chaotic, cluttered, bleak, uninspired, garish, stiff, unrefined, artificial, heavy, disorderly, grim, rusty, faded, cramped, jarring, obtrusive, awkward, ordinary, harsh, gloomy, cold, rigid, overcrowded, mismatched, messy, uneven, tacky, lifeless,<br><br>unbalanced, heavy-handed, overbearing, dissonant, grating, oversaturated,<br><br>unpleasant, rigid, blur.<br><br>Positive Lexicon<br><br>Negative Lexicon<br><br>Figure S1. The detail of Positive/Negative Lexicon<br><br>• Bicycle: “Please describe the bicycle in terms of color, type, size, condition, and any distinctive marks or decorations. Include details such as the presence of baskets, reflectors, or any branding.”|
|---|

In this supplementary material, we sive details and analyses across the fo

- • Section 1 elucidates our methodology positive/negative lexical databases, details.
- • Section 2 provides an extensive co designed class hints, demonstrating tions.
- • Section 3 delineates the improved prompting strategies employed in F ular emphasis on their methodological
- • Section 4 explicates the architectural stanceEnhancer, supplemented with utilized in our Large Language Model
- • Section 5 elaborates a detailed dis ples behind our metric design for including mathematical formulations dations.
- • Section 6 demonstrates the prompts both the inference and evaluation st
- • Section 7 presents a evaluation of our methodology across both commercial and open-source models, including experimental results and analytical findings.

- • Car: “Please describe the car by its color, make, model, condition, license plate (if visible), and any distinguishing features such as stickers, dents, or modifications.”
- • Airplane: “Please describe the airplane by its type (commercial, private, etc.), airline brand, color scheme, size, and any visible markings such as logos or tail numbers.”
- • Bus: “Please describe the bus by its color, type (public, school, etc.), condition, any branding or advertising on its surface, and the route number or destination if visible.”
- • Train: “Please describe the train by its type (freight, passenger, high-speed, etc.), color, length, condition, and any visible logos or car numbers.”
- • Truck: “Please describe the truck by its type (pickup, semi, etc.), color, make, model, any visible logos or branding, and details such as cargo or modifications.”
- • Boat: “Please describe the boat by its type (sailboat, motorboat, yacht, etc.), size, color, condition, and any identifying features like registration numbers or flags.”
- • Traffic Light: “Please mention the current state of the traffic light (red, yellow, green), its location, and any additional details like the presence of pedestrian signals.”
- • Fire Hydrant: “Please describe the fire hydrant by its color, condition, and any notable features such as signs, markings, or proximity to other objects.”
- • Stop Sign: “Please describe the stop sign’s condition, location, and any visible obstructions or markings on it.”
- • Parking Meter: “Please describe the parking meter by its condition, type (modern, traditional), and any visible

###### 1. Positive/Negative Lexicon

To enhance the aesthetic quality of generated videos, we carefully collected prompts from various open-source model galleries, extracting adjectives to build a Positive Lexicon. Conversely, we manually constructed a Negative Lexicon, which was further enriched using the powerful LLM, GPT-4o. Both lexicons were refined through meticulous manual screening. The detailed contents of the Positive/Negative Lexicons are shown in Figure S1.

###### 2. Human-designed Class Hints

For the Human-designed Class Hints, we carefully crafted additional prompts for over eighty categories, each specifically tailored to its specific characteristics. Below, we present twenty of these categories. The full JSONformatted hints for all classes, ready for direct use, will be provided in the code we plan to release later.

- • Person: “Please focus primarily on the person’s facial expressions, attire, age, gender, and race in the video and give description in detail. Please mention if there are any necklaces, watches, hat or other decoration; otherwise, there’s no need to bring them up.”

information like pricing or operational status.”

- • Bench: “Please describe the bench by its material, color, condition, and any distinctive features such as inscriptions, decorations, or nearby objects.”
- • Bird: “Please describe the bird by its species (if identifiable), color, size, behavior, and any unique markings or features.”
- • Cat: “Please describe the cat by its color, breed (if identifiable), size, behavior, and any distinguishing features such as collars or patterns.”
- • Dog: “Please describe the dog by its breed (if identifiable), color, size, behavior, and any accessories such as collars or leashes.”
- • Horse: “Please describe the horse by its color, breed (if identifiable), size, behavior, and any accessories such as saddles or reins.”
- • Sheep: “Please describe the sheep by its color, size, behavior, and any distinguishing features such as markings or tags.”
- • Cow: “Please describe the cow by its color, breed (if identifiable), size, behavior, and any distinguishing features such as tags or markings.”
- • Elephant: “Please describe the elephant by its size, tusk length, condition, and any unique features such as markings or behavior.”

###### 3. Prompt Design of Figure 3

System prompt. Referring to ShareGPT4Video [3], we divided the System prompt into three parts. Through extensive tests on challenging samples, including multi-instance, complex scenes, and high-intensity motion, we finalized the system prompt shown in Figure S5. Additionally, temporal metadata extracted using the code provided in Figure S6.

Prompts of global description, background detail and camera movement. The global description is derived from a single prompt: “Please describe this video in one sentence, no more than 20 words.”. To illustrate the acquisition of camera motion and background details, we provide an example of implementing camera hints with movement cues in Figure S7. A similar approach is used for extracting background details included in our code released later.

Prompts of structured caption. In the structured caption section, we we use Actions and Motion as examples, with the CoT prompt shown in Figure S8. The acquisition of Appearance and the injection of Human-designed class hints follow a similar approach.

###### 4. Design of InstanceEnhancer

In InstanceEnhancer, prompt alignment during inference is achieved through a two-stage process (Figure S2). To pro-

vide more precise instructions to LLMs, we meticulously designed multiple examples as part of the CoT, which are fed into the LLMs. An example of this is shown in Figure S9.

###### 5. Evaluation metrics for video reconstruction

3DVAE score (3DVAEscore). The LIPIPS score [33] which is widely used to evaluate image reconstruction quality, measures perceptual distance between ground truth (GT) and reconstructed images. We extent this concept for video data by using 3DVAE [30] to extract latentspace video representations from both GT videos and their caption-reconstructed versions. 3DVAEscore computes the distance between latent representations across spatial and temporal dimensions:

wl ⊙ zlGT,hwt − zlrec,hwt 22

d(zGT,zrec) =

l t h,w

(1)

where zlGT,hwt and zlrec,hwt represent the latent representations at layer l, spatial location (h,w), and temporal frame t, with wl as the layer-specific weight matrix. We set (h,w,t) = (224,224,8) for evaluation.

To ensure consistency, we use the same video generation model across all captioning methods. Following LIPIPS methodology, we validate the 3DVAE score by comparing GT videos against various distorted versions. As shown in Tab. S1, the results demonstrate that our score effectively captures perceptual similarities between GT and reconstructed videos.

Distortion type 3DVAE score↓ Setting

Blurring 7.71 GaussianBlur(kernel=(5, 5), sigma=0) Compression artifacts 11.19 JPEG compression (quality 5-30) Corruptions 39.80 Random pixel masking (binary mask) Random noise 49.70 Gaussian noise (mean=0, stddev=25) Brightness distortion 63.25 Scaling (factor 0.5-1.5) Spatial shifts 78.94 Random affine shifts (±10 pixels)

T2V models Avg. 134 ∼ 145 Broken video 149.50 -

Table S1. 3DVAE scores for various distortions and video models, showcasing its effectiveness in capturing perceptual similarities and reconstruction accuracy. The setting column provides details of the experimental setup for each distortion type.

CLIP score sentence by sentence (CLIPSenbySen). While CLIP [19] is widely used for text-video similarity computation [6, 14], its 77-token limit restricts processing of long texts. To overcome this, we propose CLIP score sentence by sentence (SenbySen), which segments texts into individual sentences and computes CLIP similarity between each sentence and video frame.

Long Prompt: The astronaut adjusted the visor of her helmet, feeling both the familiar weightlessness of space and the surreal jolt of hooves beneath her. The horse, a sleek chestnut mare named Comet, moved gracefully across the lunar surface……

[System Prompt] You are a good prompt enhancement assistant, you will follow the original text on the basis of following the instructions of the user to supplement and enhance the details of the original prompt, both to ensure rationality and to ensure the high quality of the text. Please use the "subject" + "attribute" structure more often. For example, "An old man, his hair is white."

LLMs

The astronaut is riding a horse.

Claude Qwen

GPT-4o

[Figure 176]

[Figure 177]

[Figure 178]

+[Example A] Write a reasonably detailed silent film for this text: The astronaut is riding a horse.

Gemini LLaMA

Positive/Negative Lexicon

A. Short Text Expansion

B(I). Instance Segmentation

+[Example B] “Identify the subjects contained in the following passage, using BREAK to separate them in one line……”

“Please ensure that only the main instance are output, no more than MAX, and ensure that all instance in The astronaut is riding a horse. appear……”

Identify Main Instance: the astronaut BREAK horse BREAK

Results

{

B(II). Structured Caption Generation

"Global Description": "The astronaut is riding a horse.", "Structural Description": {

Appearance： Please describe appearance of the astronaut/horse. If the original text lacks specific descriptions, provide reasonable and vivid descriptions based on the context {Long Prompt} …… Action and Motion： Please describe action of the astronaut/horse. If the original text lacks specific descriptions, provide reasonable and vivid descriptions based on the context {Long Prompt} ……

"Main Instance": {

- "No.0": { "Class": "an astronaut", "Appearance": "An astronaut, clad in a sleek and pristine spacesuit with a shimmering...", "Actions and Motion": "The astronaut, gliding on a horse through the tranquil desert night...", "Position": "An astronaut from bottom-left to center and out of frame in the end."

},

- "No.1": { "Class": "a horse", "Appearance": "The horse, a magnificent creature with a glossy black coat and graceful...", "Actions and Motion": "The horse moves with fluid, deliberate strides, its hooves striking the

},

###### Position：

Select a position change for the astronaut/horse that fits the context. You need to choose from the following combinations： Background Detail： Follow the original text {Long Prompt} and provide a background description of the scene…… Camera Movement： Based on the text provided {Long Prompt}, write a suitable and varied description of camera movements and angles in one sentence……

weathered...",

- },

"Position": "A horse from center-left to center-right and out of frame in the end." },

"Background Detail": "The scene is an expansive, desolate lunar landscape...",

"Camera Movement": "The camera weaves around the astronaut and horse with an artful rhythm..."

} }

} }

###### [Example A]

|[Example]<br><br>1. Two wolves were hunting a rabbit in the snow.<br>2. Two men were fishing and a woman was standing by.<br>3. A completely still tree. [Other examples]<br>| |
|---|---|
| | |

|[Example B]<br><br>1. a wolf BREAK a wolf BREAK a rabbit<br>2. a man BREAK a man BREAK a woman<br>3. a tree BREAK a boy BREAK a man BREAK a woman [Other examples]<br>|
|---|

- 1. In the vast whiteness of the snowy forest, two wolves stalk quietly…
- 2. On the bank of a quiet river, two men sit, fishing rods in hand, lost in their shared pursuit…
- 3. The film opens with a solitary tree standing tall in the middle of a vast field, perfectly motionless… [Other examples]

Figure S2. Detailed overview of the InstanceEnhancer pipeline. Example No.1 as shown in Figure S9.

[Figure 179]

Single

Action {"sentence": "A tall man stands at the left.", "instance": {"class": "man", "action": "standing"}}

###### Color

{"sentence": "A neon pink pen lies on the desk.", "instance": {"class": "pen", "color": "neon pink"}} Shape {"sentence": "A round chair is pushed against the far left corner of the desk.", "instance": {"class": "chair", "shape": "round"}} Texture {"sentence": "A rough towel hangs on the rack beside the bathroom door.", "instance": {"class": "towel", "texture": "rough"}} Detail

{"sentence": "A fluffy gray cat with bright green eyes sits on the windowsill, gazing outside at the fluttering leaves in the breeze. Its tiny pink nose twitches as it watches a bird land on a nearby

branch, and a distinct purple stripe runs down its belly, adding to its unique charm. A small silver bell dangles from its ear, jingling softly with each movement.", "instance": {"detail": "a distinct purple stripe runs down its belly", "class": "cat"}}

###### Multiple

###### Action

{"sentence": "A horse running as a butterfly flutters by.", "instance": {"0":{"class": "horse", "action": "running"}, "1":{"class": "butterfly", "action": "fluttering"}}}

Color {"sentence": "A sapphire blue keychain hangs from a light brown bag.", "instance": {"0":{"class": "keychain", "color": "sapphire blue"}, "1":{"class": "bag", "color": "light brown"}}} Shape {"sentence": "A square speaker lies on a round shelf.", "instance": {"0":{"class": "speaker", "shape": "square"}, "1":{"class": "shelf", "shape": "round"}}} Texture {"sentence": "A satin pillow sits on a faux leather sofa.", "instance": {"0":{"class": "pillow", "texture": "satin"}, "1":{"class": "sofa", "texture": "faux leather"}}}

###### Detail

{"sentence": "A fluffy gray cat with a purple stripe down its belly sits on the windowsill. Beside the fireplace, a golden retriever with a heart-shaped birthmark on its front paw rests comfortably.", "instance": {"0": {"detail": "a distinct purple stripe on its belly", "class": "cat"}, "1": {"detail": "a heart-shaped birthmark on its front paw", "class": "retriever"}}}

Figure S3. Inference examples of Inseval.

Let S = {s1,s2,...,sn} be the sentences from input text and V = {v1,v2,...,vt} be the video frames. For a sentence si and frame vj, we denote their CLIP similarity as CLIP(si,vj). The comprehensive score is computed as:

OverallScore =

n

1 n

i=1

 1

  (2)

t

CLIP(si,vj)

t

j=1

|Multiple Shape: A square speaker lies on a round shelf.<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|---|
<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|---|
<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|---|
<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|---|
<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>Multiple Detail: A green iguana with a spiked crest along its back basks on a rock. Nearby, an otter with a tiny shell necklace floats on its back.<br><br>CogVideoX<br><br>Hailuo<br><br>CogVideoX<br><br>Hailuo<br><br>Figure S4. Visualization comparing open-source models and commercial models on prompts with poorer performance.<br><br>This approach not only addresses the token limitation but also enhances assessment quality by naturally assigning lower weights to non-specific textual descriptions.<br><br>Human evaluation. Automated machine-based scoring systems, while offering enhanced objectivity and efficiency, often fail to align with human preferences or fully grasp the nuances of context and meaning in a given task. To ensure a comprehensive and balanced evaluation, we adopted<br><br>2) Intrinsic Hallucination: Evaluate whether the text hallucinates descriptions of things present in the video. 3) Extrinsic Hallucination: Evaluate whether the text introduces content that is not present in the video. For convenience, the latter two have been combined into a single metric called the Hallucination Scores (HS) [7]. The specific guidelines and scoring criteria for each metric refers to Table S2.<br><br>6. Inseval|
|---|

Inference prompts of Inseval. In implementing Inseval, we designed multiple prompts to test each dimension, as illustrated in Figure S3. To further evaluate the model’s generative capabilities and instruction-following accuracy,

a human-based assessment framework. This evaluation is carried out across several key dimensions, including: 1) Instance Detail (ID): Evaluate whether the text provides accurate descriptions of the details of the examples in the video.

###### Instance Detail Hallucination Scores

- 1 Descriptions are extremely vague, imprecise, or largely inaccurate. Almost no specific details from the video are captured correctly.

1 Severe hallucination - Describes many nonexistent details, significantly misrepresents what is shown, or introduces extensive irrelevant content with many unrelated topics or external information.

- 2 Descriptions have major inaccuracies or omit many important details. Only a few basic elements are described correctly.

2 Frequent hallucination - Multiple instances of fabricated or misrepresented details and significant extra content introducing information beyond the video scope.

- 3 Descriptions are moderately accurate but lack precision in some areas. Core details are present but some secondary details are missing or incorrect.

3 Occasional hallucination - A few minor instances of fabricated details, misrepresentations, or the addition of extra content not covered in the video.

- 4 Descriptions are largely accurate and detailed. Most key elements and nuances from the video are captured correctly, with only minor omissions or imprecisions.

4 Minimal hallucination - One or two very minor discrepancies or limited introduction of external information.

- 5 Descriptions are highly precise and comprehensive. All important details from the video are captured accurately, including subtle elements and specific examples.

5 No hallucination - All described details accurately reflect what is shown in the video, with no external content added.

Table S2. This table outlines scoring criteria for Instance Detail and Hallucination Scores, integrating intrinsic and extrinsic hallucinations into a unified framework for evaluation.

we deliberately included some “counter-intuitive” shapes in the prompt design.

Evaluation prompts of Inseval. For the evaluation, we used a general CoT Q-A pair format (with a slightly different design for the ‘Detail’ dimension, shown in Figure S10 to assess whether the MLLMs successfully matched the generated videos to the corresponding dimensions, as outlined in the specific code. In single-object scenarios, the success rate is calculated as the percentage of correctly matched prompts. In multi-object scenarios, the generation is deemed successful only if all targets meet the requirements. For reproducibility, fixed random seeds are used during generation and evaluation.

In Table 2, the ‘Shape’ and ‘Detail’ dimensions under Multiple category are omitted due to consistently very poor performance across all tested models. Even CogVideoX5B, the overall best performer, struggles with multi-object tasks in these dimensions, as shown in Figure S4. Two primary error types are observed in Multiple Shape tasks: attribute confusion (Top case) and failure to follow multiple target instructions (Bottom case), where targets are either missing or rendered incorrectly. Commercial models demonstrate relatively better performance, which we further analyze in Section 7.

###### 7. Analysis on Commercial Products vs. Opensource Models

Prompt processing analysis. Commercial T2V products excel at processing complex input prompts, effectively handling long-form text in structured formats while preserving semantic coherence. They can seamlessly interpret detailed scene descriptions, character interactions, and sequential events within a single prompt, producing coherent visual narratives, have shown surprising results in many situations.

Open-source T2V models, however, are unable to directly process long-text structured prompts, requiring an additional alignment step (Figure S11). This preprocessing can lead to potential information loss and inconsistencies in the final output, restricting the ability to capture nuanced details from the original prompt.

Information retention capabilities. Different models exhibit notable differences in information retention (Figure S4). Commercial products (e.g., Hailuo AI) excel in maintaining fidelity between text and visual content, effectively preserving detailed instructions and translating multiple attributes into video sequences. This strength is particularly apparent when our caption contains complex scenes that demand temporal consistency and fine-grained details.

Open-source models face challenges in consistently representing instance information (Figure S4), exhibiting variability in detail preservation and limited capability with complex attribute combinations. These shortcomings are particularly evident when processing prompts with multiple interrelated instances or maintaining consistent visual characteristics across temporal sequences.

|[Figure 210]<br><br>System Prompt<br><br># Character You are an excellent video frame analyst. Utilizing your incredible attention to detail, you provide clear, sequential descriptions for video frames. You are good at identifying and describing the properties of each target in the video frame, the actions and movement.<br><br># Skills<br><br>## Skill 1: Describing Objects Appearances<br><br>- Describe the appearances of instance.<br>- Determine which parts are colored parts, as the goal of the main description.<br>- Focus mainly on the color part, the black and white part only as an auxiliary role.<br>- Highly sensitive to person, describe they in detail, such as the style and color of hat, the style and color of clothes, age, gender, body type, expression, etc.<br><br><br>## Skill 2: Describing Objects' Actions and Behaviors<br><br>- Elaborate the action of instance.<br>- Notice and describe changes in the actions or behaviors.<br>- Determine which Objects are main instance and give more detailed description.<br><br><br>## Skill 3: Use Fine Words to Describe.<br><br><br>- Select the appropriate ones of following words in your description: kaleidoscopic, delicate, grand, gentle, soothing, cool, mature, solitary, worn, chaotic, dramatic, cozy, shimmering, desolate, serene, weathered, whispering, loose-fitting, vibrant, tranquil, dimly-lit, purplish, introspective, artfully, sleek, energetic, overcast, brilliant, slender, graceful, picturesque, whimsical, contented, gentle, warm, tender, pastel-colored, elegant.<br>- State facts objectively without using any rhetorical devices such as metaphors or personification.<br>- Do not use any of the following negative words when describing: dull, rough, harsh, chaotic, cluttered, bleak, uninspired, garish, stiff, unrefined, artificial, heavy, disorderly, grim, rusty, faded, cramped, jarring, obtrusive, awkward, ordinary, harsh, gloomy, cold, rigid, overcrowded, mismatched, messy, uneven, tacky, lifeless, unbalanced, heavy-handed, overbearing, dissonant, grating, oversaturated, unpleasant, rigid, blur. # Constraints<br>- State facts objectively without using any rhetorical devices such as metaphors or personification.<br>- Exclude sounds-related aspects, given the unavailability of audio signals.<br>- Descriptions should be fluent and precise, avoiding analyzing and waxing lyrical.<br>- Descriptions need to be concise, describing only the information that can be determined, without analysis or speculation.<br>- Do not mention the frame number and timestamp of the current frame.<br>- The main object will occupy most of the content of the picture, and there may be more than one main object, and there may be no main object in the landscape type of video.<br>- Only the main object needs to be described in detail, and the other objects only need to be described briefly.<br>- Strictly follow the format of the structured output, containing all of its elements.<br><br><br>Figure S5. System prompt of InstanceCap.<br><br>7|
|---|

[Figure 211]

def load_video(video_path, max_frames_num,fps=1,force_sample=False): if max_frames_num == 0:

return np.zeros((1, 336, 336, 3)) vr = VideoReader(video_path, ctx=cpu(0),num_threads=1) total_frame_num = len(vr)

[Figure 212]

video_time = total_frame_num / vr.get_avg_fps()

Example Prompt of Stage A&B(I)

fps = round(vr.get_avg_fps()/fps) frame_idx = [i for i in range(0, len(vr), fps)] frame_time = [i/fps for i in frame_idx] if len(frame_idx) > max_frames_num or force_sample:

Stage A. Short Text Expansion # USER Write a reasonably silent film for this text: Two wolves were hunting a rabbit in the snow. (Short prompt)

sample_fps = max_frames_num uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int) frame_idx = uniform_sampled_frames.tolist() frame_time = [i/vr.get_avg_fps() for i in frame_idx]

###### # ASSISTANT（human-provided）

In the vast whiteness of the snowy forest, two wolves stalk quietly. One is large and heavyset, its fur a mix of dark grays and black, blending into the shadows of the trees. The other is leaner and quicker, with a pale, almost silver coat that glimmers in the winter light. Their eyes are sharp and focused, and each step is measured—one wolf pacing powerfully, the other darting with agile precision. Ahead of them, a small white rabbit crouches in the snow, nearly invisible but for

frame_time = ",".join([f"{i:.2f}s" for i in frame_time]) spare_frames = vr.get_batch(frame_idx).asnumpy() return spare_frames,frame_time,video_time

the black tips of its ears and nose. Suddenly, the rabbit bolts, kicking up a spray of powder. The dark wolf charges forward

def build_video_prompt(image_processor, video, frame_time, video_time): video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16() video = [video] time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "\ f"These frames are located at {frame_time}.Please answer the following questions related to this video." pre_instruction = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n"

with brute force, while the silver one sprints with smooth, loping strides. The rabbit’s movements are frantic and unpredictable, weaving between tree roots and leaping over snowdrifts. The wolves flank it from both sides, their breaths clouding the air, closing in like shadows. At the last moment, the rabbit dives into a narrow burrow, disappearing into safety. The dark wolf sniffs the ground, frustrated, while the silver wolf paces briefly before the two melt back into the forest, leaving only their crisscrossing prints behind in the snow. (Long prompt)

###### Stage B(I). Instance Segmentation

return video, pre_instruction

# USER Identify the subjects contained in the following passage, using BREAK to separate them in one line:

Long prompt Please distinguish between instances and scenes, and return only instances, such as: "a man BREAK two cat BREAK a cup..."Please ensure that only the main instance are output, no more than X, and ensure that all objects in Short prompt appear. Note that the instance you extract must be an entity that can be touched. When there are multiple targets, you need to give them separately.

Figure S6. Code of getting video temporal metadata.

###### # ASSISTANT（human-provided） a wolf BREAK a wolf BREAK a rabbit

- Figure S9. Designed example for LLMs.

[Figure 213]

Evaluation Prompt(Single)

- STEP 1 # USER Describe this video in one sentence, care about the {dimension_type}. # ASSISTANT

- Answer of STEP 1

STEP 2 # USER Tell me if '{instance_class}' is in the video? Note the synonyms of '{instance_class}'. Your answer can only be YES or NO.Do not output any answer that is not YES or NO. # ASSISTANT

- Answer of STEP 2(if answer is“No”, return False)

STEP 3

# USER Based on your previous answer, tell me what is the '{dimension_type}' of '{instance_class}' in the video? Be careful to ignore camera movement. # ASSISTANT

- Answer of STEP 3

STEP 4 of Others # USER Do you think the '{dimension_type}' of '{instance_class}' in the video Approximatly close to to '{instance_specific}'? Your answer can only be YES or NO. Do not output any answer that is not YES or NO. # ASSISTANT

- Answer of STEP 4(Final answer)

STEP 4 of Detail # USER Is the '{instance_specific}' of '{instance_class}' partly reflected in the video? Your answer can only be YES or NO. Do not output any answer that is not YES or NO. # ASSISTANT Answer of STEP 4(Final answer)

- Figure S10. Evaluation prompts of Inseval.

[Figure 214]

Camera Movement Prompt

# User Let's think step by step... Try to separate the camera movement from the video.

if camera_motion == "Undetermined": The motion of the video camera is very complex, can you infer the possible motion of the camera and the shooting Angle (long distance/medium distance/overhead Angle/POV, etc.) from the changes in the video? elif camera_motion == "static": Is the camera static or moving? Can you deduce the possible motion of the camera and the shooting Angle (long distance/medium distance/overhead Angle/POV, etc.) in the video?" else: The motion of the camera in this video is {camera_motion}. According to my tips, can you deduce the possible motion of the camera and the shooting Angle (long distance/medium distance/overhead Angle/POV, etc.) in the video?

Summarize the camera movement and shooting angle, use degree adverbs appropriately(Sharply, rapidly, slowly, etc), try to give description in detail.

- Figure S7. Prompt of camera movement.

[Figure 215]

Actions and Motion

- STEP 1 # USER Let's think step by step… Can you tell what color the background of this video is? Unimportant parts have been intentionally obscured with a mosaic; please focus only on the clearly visible portions. If I remind you that there's a {target_object}, can you see it in the video? This is supplementary information to the full video to aid your description: {global_description} Can you find the {target_object} corresponding to this paragraph in the video? # ASSISTANT

- Answer of STEP 1

STEP 2 # USER Continue ignore mosaic background, if {target_object} has any movement, answer what you think

{target_object} is doing in video.

Extract the information related to {target_object} that you are currently describing.

Do not answer any information unrelated to {target_object}'s state of motion or you're not sure. Observe carefully, use appropriate adjectives, and give description in detail. Do not answer in bullet points. Do not mention any objects that are not related to {target_object}. Do not speculate about the environment in which the {target_object} is located. Do not mention anything about a \"blurred background.\" Use a continuous paragraph. # ASSISTANT

- Answer of STEP 2(Final answer)

- Figure S8. Prompt of actions and motion.

|[Figure 216]<br><br>Aligning Prompt<br><br>Step 1 # User Let's think step by step... Read the following JSON and summarize it to continuous text paragraph, ensuring that all main ideas and crucial details are preserved:<br><br>{InstanceCap}<br><br>What you need to pay attention to is the "Global Description" and "Structural Description" sections. The "Global Description" provides an overall summary of the video, while the "Structural Description" contains detailed information about various aspects of the video, such as main characters, background details, and camera movements. Please focus on the details in the "Structural Description" and combine them with the "Global Description" to create a summary. Select the appropriate ones of following words in your description: kaleidoscopic, delicate, grand, gentle, soothing, cool, mature, solitary, worn, chaotic, dramatic, cozy, shimmering, desolate, serene, weathered, whispering, loose-fitting, vibrant, tranquil, dimly-lit, purplish, introspective, artfully, sleek, energetic, overcast, brilliant, slender, graceful, picturesque, whimsical, contented, gentle, warm, tender, pastel-colored, elegant. Do not use any of the following negative words when describing: dull, rough, harsh, chaotic, cluttered, bleak, uninspired, garish, stiff, unrefined, artificial, heavy, disorderly, grim, rusty, faded, cramped, jarring, obtrusive, awkward, ordinary, harsh, gloomy, cold, rigid, overcrowded, mismatched, messy, uneven, tacky, lifeless, unbalanced, heavy-handed, overbearing, dissonant, grating, oversaturated, unpleasant, rigid, blur.<br><br>Step 2 # User<br><br><br>1. Please use the "subject" + "attribute" + "position" structure more often. For example, "An old man, his hair is white."<br>2. Please tell me the content of the video directly, don't use "The video shows..." Or other similar forms, you should begin with a direct description of the content, for example:"An old man..."<br>3. If there are multiple objects, such as people, introduce them with phrases like "A man... and a woman..., and a man...","A car..., and a car...".<br>4. When you need to describe The background in detail, use "The scene is..." As an opening sentence.<br>5. Summarize it to approximately 180 words<br><br><br>Figure S11. Aligning prompt used during alignment with the open source model.<br><br>9|
|---|

