[Figure 1]_images/imageFile1.png>)

### MM-VID : Advancing Video Understanding with GPT-4V(ision)

Kevin Lin∗, Faisal Ahmed∗, Linjie Li∗, Chung-Ching Lin∗, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, Ce Liu, Lijuan Wang∗♠ Microsoft Azure AI

∗ Core Contribution ♠ Project Lead

https://multimodal-vid.github.io/

# arXiv:2310.19773v1[cs.CV]30Oct2023

[Figure 2]_images/imageFile2.png>)

[Figure 3]_images/imageFile3.png>)

[Figure 4]_images/imageFile4.png>)

[Figure 5]_images/imageFile5.png>)

[Figure 6]_images/imageFile6.png>)

[Figure 7]_images/imageFile7.png>)

[Figure 8]_images/imageFile8.png>)

[Figure 9]_images/imageFile9.png>)

[Figure 10]_images/imageFile10.png>)

[Figure 11]_images/imageFile11.png>)

[Figure 12]_images/imageFile12.png>)

[Figure 13]_images/imageFile13.png>)

[Figure 14]_images/imageFile14.png>)

[Figure 15]_images/imageFile15.png>)

[Figure 18]_images/imageFile18.png>)

…… …

[Figure 20]_images/imageFile20.png>)

[Figure 23]_images/imageFile23.png>)

[Figure 24]_images/imageFile24.png>)

…

[Figure 25]_images/imageFile25.png>)

[Figure 26]_images/imageFile26.png>)

[Figure 27]_images/imageFile27.png>)

[Figure 28]_images/imageFile28.png>)

♪ ♪ ♪

[Figure 29]_images/imageFile29.png>)

[Figure 30]_images/imageFile30.png>)

[Figure 31]_images/imageFile31.png>)

[Figure 32]_images/imageFile32.png>)

[Figure 33]_images/imageFile33.png>)

[Figure 34]_images/imageFile34.png>)

[Figure 35]_images/imageFile35.png>)

[Figure 36]_images/imageFile36.png>)

[Figure 37]_images/imageFile37.png>)

Faraon Thut Nefer Nimona Ballister Boldheart

[Figure 38]_images/imageFile38.png>)

[Figure 39]_images/imageFile39.png>)

[Figure 40]_images/imageFile40.png>)

[Figure 41]_images/imageFile41.png>)

[Figure 42]_images/imageFile42.png>)

[Figure 43]_images/imageFile43.png>)

[Figure 44]_images/imageFile44.png>)

[Figure 45]_images/imageFile45.png>)

[Figure 46]_images/imageFile46.png>)

The commentator mentions that there is a different sound when the ball leaves Will Smith's bat, implying that he might have become stronger.

[Figure 47]_images/imageFile47.png>)

[Figure 48]_images/imageFile48.png>)

[Figure 49]_images/imageFile49.png>)

[Figure 50]_images/imageFile50.png>)

[Figure 51]_images/imageFile51.png>)

[Figure 52]_images/imageFile52.png>)

baal bir 

[Figure 53]_images/imageFile53.png>)

[Figure 54]_images/imageFile54.png>)

[Figure 55]_images/imageFile55.png>)

[Figure 56]_images/imageFile56.png>)

[Figure 57]_images/imageFile57.png>)

Live-streaming QA

[Figure 58]_images/imageFile58.png>)

[Figure 59]_images/imageFile59.png>)

[Figure 60]_images/imageFile60.png>)

[Figure 61]_images/imageFile61.png>)

[Figure 62]_images/imageFile62.png>)

[Figure 63]_images/imageFile63.png>)

[Figure 64]_images/imageFile64.png>)

[Figure 65]_images/imageFile65.png>)

……

Figure 1. MM-VID allocates specialized vision, audio, speech experts with GPT-4V(ision) to address challenging video understanding tasks. For example, the system could associate information from multiple uploaded episodes and reason the storyline of the queried characters (“Multi-Video Episodic Analysis”). We highlight key information here and postpone full MM-VID responses to Figures 4-32. Demo videos are available at this link.

[Figure 66]_images/imageFile66.png>)

[Figure 67]_images/imageFile67.png>)

[Figure 68]_images/imageFile68.png>)

[Figure 69]_images/imageFile69.png>)

1

#### Abstract

We present MM-VID, an integrated system that harnesses the capabilities of GPT-4V1, combined with specialized tools in vision, audio, and speech, to facilitate advanced video understanding. MM-VID is designed to address the challenges posed by long-form videos and intricate tasks such as reasoning within hour-long content and grasping storylines spanning multiple episodes. MM-VID uses a video-to-script generation with GPT-4V to transcribe multimodal elements into a long textual script. The generated script details character movements, actions, expressions, and dialogues, paving the way for large language models (LLMs) to achieve video understanding. This enables advanced capabilities, including audio description, character identification, and multimodal high-level comprehension. Experimental results demonstrate the effectiveness of MM-VID in handling distinct video genres with various video lengths. Additionally, we showcase its potential when applied to interactive environments, such as video games and graphic user interfaces.

#### 1. Introduction

People around the world create numerous videos on a daily basis [14, 27, 48, 55], including user-generated live streams, video-game live streams, short clips, movies, sports broadcasts, advertising, and more. Videos serve as a versatile medium for conveying information and content through various modalities [12, 59,60, 74,77,81, 82], such as text, visuals, and audio. Developing methods that can learn from diverse modalities will enable us to design cognitive machines with enhanced capabilities for analyzing uncurated real-world videos, extending beyond the confines of hand-curated datasets. However, this rich representation introduces many challenges for the study of video understanding, particularly when dealing with extended-duration videos [62,70].

Understanding long videos, especially those spanning over an hour, is a complex task that demands advanced methods capable of analyzing sequences of images and audio across multiple episodes. This challenge is compounded by the need to extract information from various sources, such as distinguishing speakers [17,49,61], identifying characters [29,46,50], and maintaining narrative coherence [26, 57]. Additionally, answering questions based on video evidence [31] requires a deep comprehension of the content, context, and subtitles. When it comes to live streaming and gaming videos [1, 11, 55], there are challenges in processing dynamic environments in real-time, re-

1In this work, we explore GPT-4V(ision) with the vision capability and refers to the model as “GPT-4V,” following the OpenAI reports [51, 52]. We refer to the text-only version of the model as “GPT-4” [51].

quiring semantic understanding, and the ability of long-term strategy planning [11,54,67,75,83].

Recently, substantial advances have been made with large pre-trained video models [9, 13, 20, 21, 42, 69] and video-language models [10, 22, 23, 30, 36–40, 66], which have demonstrated their reasoning capabilities for video content. However, these models are usually trained on short clips (e.g., 10-second videos in Kinetics [15] and VATEX [68]) or pre-defined action classes (e.g., 174 classes in Something-Something v1 [24]). Consequently, these models may fall short in providing a detailed comprehension of intricate videos in real world [62, 70]. To achieve a more comprehensive understanding of the videos we encounter in daily life, we need methods capable of addressing complex challenges. It involves not only identifying who are in the scene and what they do, but also pinpointing when and how they act, while recognizing subtle nuances and visual cues across different scenes. The aim of this work is to address these challenges and explore methods that can be applied directly to real-world video understanding. Our approach involves breaking down extended video content into coherent narratives and subsequently employing these generated stories for video analysis.

Recent advances in Large Multimodal Models (LMMs) [7,8,19,47,51,52,78], such as GPT-4V(ision) [52], have demonstrated significant breakthroughs in processing both input images and text for multimodal understanding. This has sparked interest in applying LMMs to the video domain. In this work, we present MM-VID, a system that integrates specialized tools with GPT-4V for video understanding. Given an input video, MM-VID performs multimodal pre-processing, including scene detection and automatic speech recognition (ASR), to collect important information in the video. The input video is then split into multiple clips according to the scene detection algorithm. Then, we employ GPT-4V, which takes the clip-level video frames as input and generates a detailed description for each video clip. Finally, GPT-4 is adopted to generate a coherent script for the full video, conditioning on the clip-level video descriptions, ASR, and video metadata if available. As shown in Figure 1, the generated script allows MM-VID to perform a diverse set of video tasks.

Experimental results demonstrate the effectiveness of MM-VID in different challenging scenarios. MM-VID is able to comprehend hour-long videos through multiple modalities, and localize specific events with correct timestamps. MM-VID also demonstrates intriguing results in an interactive environment, such as predicting the possible next steps when playing a video game [4] or interacting with a graphical user interface (GUI) [78].

[Figure 71]_images/imageFile71.png>)

[Figure 72]_images/imageFile72.png>)

[Figure 73]_images/imageFile73.png>)

| | |
|---|---|
| | |
| | |

[Figure 74]_images/imageFile74.png>)

[Figure 75]_images/imageFile75.png>)

- Clip 1

- Clip 2

[Figure 76]_images/imageFile76.png>)

[Figure 77]_images/imageFile77.png>)

|Visual Prompt|
|---|

Multimodal Pre-Processing

Clip-Level Video Description Generation

[Figure 78]_images/imageFile78.png>)

[Figure 79]_images/imageFile79.png>)

[Figure 80]_images/imageFile80.png>)

[Figure 81]_images/imageFile81.png>)

LLM with Script

[Figure 82]_images/imageFile82.png>)

[Figure 83]_images/imageFile83.png>)

[Figure 84]_images/imageFile84.png>)

|Text Prompt|
|---|

[Figure 85]_images/imageFile85.png>)

(Scene Detection, ASR)

(GPT-4V)

[Figure 86]_images/imageFile86.png>)

[Figure 87]_images/imageFile87.png>)

[Figure 88]_images/imageFile88.png>)

Clip N

Video Tasks: Audio Description, Grounded QA, Summarization, Speaker Identification, Character Identification, etc.

[Figure 89]_images/imageFile89.png>)

External Knowledge

Script Generation using LLM (GPT-4)

Script Input Video

|Text Prompt| |
|---|---|
| | |

(Metadata, Title, Abstract, Character Snapshot, etc.)

[Figure 90]_images/imageFile90.png>)

- Figure 2. Overview of MM-VID. Our system takes a video file as input, and outputs a long textual script describing the video contents. MM-VID consists of four modules: (i) Multimodal Pre-Processing, (ii) External Knowledge Collection, (iii) Clip-Level Video Description Generation, and (iv) Script Generation.

[Figure 91]_images/imageFile91.png>)

###### Task List:

Chat with Reference (Script)

[Figure 92]_images/imageFile92.png>)

[Figure 93]_images/imageFile93.png>)

Please generate audio description for the input video

Audio Description, Grounded QA, Summarization, Speaker Identification, Character Identification, Multimodal Reasoning, etc.

#### 2. Related Work

researchers [6, 33, 65, 72] explore the LangChain system paradigm [16], which aims to integrate expert tools with existing LLMs to create new functionalities. For example, VLog [6] uses BLIP2 [34] and GRIT [71] as dense image captioners, Whisper [56] as ASR translator, and ChatGPT as a reasoner. By transcribing a given video to textual descriptions (e.g., document), it enables ChatGPT for video question-answering tasks. Inspired by the efficacy of these tool-using approaches [16, 65, 79], we explore integration with GPT-4V for video understanding.

Conventional Video Understanding Methods. Early work in computer vision centered on building video foundation models [9,13,20,21,42,69]. These models, with different neural network architecture designs and training methods, have achieved great breakthrough at analyzing short video clips [14,15,28,63], typically lasting less than 30 seconds. However, these models are typically pre-trained with vision modality only, and thus may require specific adjustment or fine-tuning for multimodal downstream tasks.

[Figure 94]_images/imageFile94.png>)

#### 3. Preliminary Study with GPT-4V(ision)

Video-Language Models. Recent studies [10, 22, 23, 30, 36–40, 66] have made remarkable improvements in multimodal representation learning for video-and-language understanding. These advancements have been particularly evident in popular downstream tasks such as video question answering [31], text-video retrieval [32,73] and video captioning [68]. Building on this momentum, researchers typically embark on a pretrain-finetune paradigm: initially pre-training a video-language foundation model on largescale video-text pairs, followed by a fine-tuning process on specific downstream datasets. However, these methods are usually trained on short video clips, often restricted to durations of around 10 seconds, posing potential challenges in comprehending longer video sequences.

Recent studies [3,51,52,78] show that GPT-4V can accept a range of inputs, such as textual descriptions, questions, or even visual cues like images or short video clips. GPT-4V’s inherent ability to comprehend visual inputs and generate contextually relevant text opens the door for a wide range of applications. By introducing a sequence of frames as input, GPT-4V can grasp temporal relationships and interactions, aiding in the identification and interpretation of dynamic visual content.

#### 4. MM-VID

Figure 2 shows the overview of our system pipeline. MM-VID takes the video file as input, and outputs a script describing the video contents. The generated script enables LLMs to achieve various video understanding capabilities. MM-VID consists of four modules: (i) Multimodal PreProcessing, (ii) External Knowledge Collection, (iii) ClipLevel Video Description Generation, and (iv) Script Generation. We describe each module in detail below.

Visual Instruction Tuning. Inspired by the breakthrough of Large Language Models (LLMs) [18,19,51,64,85], recent studies [35,43,44,62,84] suggest using a frozen LLM combined with an image encoder and a few learnable modules for video understanding tasks. Specifically, researchers propose the visual instruction tuning [35, 41, 44], which aims to fine-tune the learnable modules and thus enable LLMs to generate textual descriptions for the video content. While promising performance is presented, these models may fall short when it comes to handling videos with extended duration. Our work aims to fill this gap, exploring methods that can be directly applied to the understanding of long videos in real-world situations.

Multimodal Pre-Processing. Starting with an input video file, our process begins by using the established ASR tool to extract transcriptions from the video. Following this, we divide the video into several short video clips. This process involves uniform sampling of video frames, with each clip consisting of 10 frames. To enhance the overall quality of frame sampling, we use established scene detection

Prompting LLMs for Video Understanding. Recently,

Feedback

Action

(Natural language or scalar values)

(Multimodal GPT-4)

Streaming Inputs

[Figure 101]_images/imageFile101.png>)

[Figure 102]_images/imageFile102.png>)

[Figure 103]_images/imageFile103.png>)

[Figure 104]_images/imageFile104.png>)

resenting the ongoing visual information unfolding in the environment. These states are then processed by GPT-4V to make informed decisions and generate responses.

[Figure 105]_images/imageFile105.png>)

State

(e.g., Video Frames, Scalar Signals)

By continually analyzing the streaming video frames, MM-VID plays a crucial role in transforming raw visual data into meaningful insights, making it valuable for applications such as video game play, the embodied agent, and GUI navigation.

Environment

Agent

(Visual World, Game, GUI, etc.)

(GPT-4V and Memory)

[Figure 106]_images/imageFile106.png>)

#### 6. Experiments

Action

(e.g., Gaming Console Control)

##### 6.1. Experiment Setup

- Figure 3. MM-VID for streaming inputs. MM-VID can serve as an agent in an interactive environment, continually receiving and processing the streaming video frames.

We implement MM-VID based on MM-REACT [79] codebase. We use the Automatic Speech Recognition (ASR) tool publicly available via the Azure Cognitive Services APIs [2], and utilize PySceneDetect [5] for scene detection.

tools like PySceneDetect [5] to help identify crucial scene boundaries.

##### 6.2. MM-VID Capabilities

Audio Description Generation,

External Knowledge Collection. We incorporate external knowledge into our input prompts to GPT-4V. This involves gathering available information, such as metadata, title, abstract, and face photos of characters within the video. In our experiments, the metadata, title, and abstract are gathered from YouTube.

Figures 4-9 provide illustrative examples of MM-VID’s complete execution flow. When a user uploads a video file, MM-VID initiates the process by first assessing the estimated video length. Subsequently, it performs multimodal pre-processing by invoking expert tools, including scene detection and ASR. Additionally, MM-VID collects external knowledge, encompassing video metadata such as title and abstract.

Clip-Level Video Description Generation. During our multimodal pre-processing, the input video is segmented into multiple clips. For each clip, which typically consists of 10 frames, we employ GPT-4V to generate video descriptions. By feeding the video frames along with the associated text prompt into the model, GPT-4V utilizes the input to generate detailed descriptions that capture the visual elements, actions, and events depicted in those frames.

Following this preliminary stage, MM-VID proceeds to generate clip-level video descriptions for each segment of the video. Finally, it invokes GPT-4, integrating these cliplevel descriptions into a coherent script. Once the script is generated, it empowers LLMs to provide a summarized understanding of the video content. That equips the system to address users’ questions with grounded answers. We discuss MM-VID’s distinct capabilities as below.

In addition, we explore the use of visual prompting, where the character’s face photos are presented alongside the character’s name in the input to GPT-4V. Our empirical results suggest that visual prompting is helpful to enhance the quality of video descriptions, particularly for more accurate character identification. These findings align with the insights from [78].

Grounded Question-Answer (QA). The generation of a comprehensive script empowers our system with the capability of grounded QA. As shown in Figure 8, let us consider a scenario where a user poses the question, “Show me the most exciting moment in this video.” In response, MMVID displays a highlight, specifically featuring a home run, and provides the corresponding timestamp. When a user asks “Who are the best pitchers in this video?” MM-VID addresses the question by referring to relevant evidence in the generated script. This grounding capability owes its success to the extensive and detailed script generation process, which documents essential timestamps and significant events within the video, enabling accurate and contextually grounded responses to user inquiries.

Script Generation using LLM. After generating the descriptions for each video clip, we use GPT-4 to integrate these clip-level descriptions into a coherent script. This script serves as a comprehensive description of the entire video, and is used by GPT-4 for a diverse set of video understanding tasks.

#### 5. MM-VID for Streaming Inputs

Figure 3 shows the diagram of MM-VID when applied to the context of streaming inputs. Our system operates as an agent within a dynamic environment where streaming video frames serve as the primary input. In this context, the agent continually receives streaming video frames as states, rep-

Multimodal Reasoning. MM-VID considers multimodal inputs, including video frames, speech transcriptions, and external knowledge if available. In Figure 8, when a user inquires, “How did you know the sound is different?”

[Figure 108]_images/imageFile108.png>)

###### Clip-Level Video Description Generation Using GPT-4V

[Figure 109]_images/imageFile109.png>)

[Figure 110]_images/imageFile110.png>)

Unknown: Or we could.

0:03:11 - 0:03:26: The Dodgers are in a roll. Max Muncie adds another point to the Dodgers lead by sending the ball deep into the right field.

[Figure 111]_images/imageFile111.png>)

- 0:01:59 Nimona: Just talk.
- 0:02:02 Unknown: The the the point. 0:02:03 Nimona: Is whatever your dark heart desires. Boss, yo

Output: Timestamp: 00:00:00 - 00:00:02 Scene: Opening graphics introducing the game between the Colorado Rockies and the Los Angeles Dodgers.

- 0:03:27 - 0:03:42: The Rockies' are unfortunate to lose a run despite a successful hit due to swift coordination between the Dodgers' players.

- 0:03:43 - 0:03:58: Jason Hayward from Dodgers adds to Rockies' dismay by smacking a lead-off shot in the third inning of the game.

- 0:03:59 - 0:04:14: Julio Urias, from the Dodgers, efficiently strikes out the Venezuelans' Luis Montero and Ryan McMahon in quick succession.

- 0:04:15 - 0:04:30: A moment of respite for the Rockies as Urias throws a pitch, allowing the batter to hit a ground ball into infield. However, it is quickly fielded, ending their chances. The innings end with the Dodgers in the lead by four runs.

0:04:31 - 0:04:46: The scene captures dodgers' Max Muncy blasting a hit into the field, which bumps the Dodgers' lead to four.

- 0:04:47 - 0:05:02: Despite a promising start for the Rockies at the top of the third, their momentum slows down as they lose a run due to a double play induced by Urias.

- 0:05:03 - 0:05:18: Hayward makes a fantastic lead-off shot for the Dodgers' in the third inning, continuing their advantageous streak.

0:05:19 - 0:05:34: Back at the pitching mound, Urias expertly strikes out the Rockies' Luis Montero and leaves the Rockies scoreless for 6 games.

- 0:05:35 - 0:05:50: Urias's superior pitching continues as he strikes out Ryan McMahon and skillfully fields a sharp grounder ending the fourth inning, serving a huge blow to the Rockies.

- 0:05:51 - 0:06:06: Scene cuts to Dodgers' Mookie Betts, who despite an injury, hammers a hit into the field, landing him a double.
- 0:06:07 - 0:06:22: Despite leading the series, Dodgers' catcher Will Smith lifts a high ball into center field which brings Mookie Betts running home, netting the Dodgers another point.

- 0:06:23 - 0:06:38: The Rockies' show some fight as Chris Brian leads the ninth innings with a double bringing the Rockies trailing the Dodgers by five.

[Figure 112]_images/imageFile112.png>)

Timestamp: 00:00:20 - 00:00:24 Scene: Pitcher on the mound at Dodger Stadium. : : : Timestamp 00:09:01-00:09:07 Scene Description: In this scene, we see a baseball game between the Los Angeles Dodgers and the Colorado Rockies. The Dodgers are leading with a score of 5-2. A player from the Dodgers hits the ball, and it is fielded by a player from the Rockies.

Loading video… Video length is 9 minutes 19 seconds.

[Figure 113]_images/imageFile113.png>)

Invoke tools: Scene Detection, ASR, Metadata Collection

Scene Detection: 00:00:00, 00:00:02, 00:00:20, 00:00:24,

..., 00:08:42, 00:08:53, 00:09:01, 00:09:07

ASR: 0:05 gets ready for his second start of the 0:08 Season began this home stand got the win : : : : 9:04 that's the ball game Dodgers sweep the 9:07 two-game series and they go four and two 9:09 on the season opening home stand

[Figure 114]_images/imageFile114.png>)

###### Script Generation Using GPT-4:

Input Prompt: You are an expert at understanding audio descriptions of different scenes in a video. Can you leverage the information provided, including title, abstract, audio descriptions and generate full audio description of each scene with non overlapping time ranges. Keep as many scenes possible covering all time ranges. You may find character names in the title or abstract. Use character names wherever possible in the audio descriptions. Keep the audio description for each time range within one short sentence.

Metadata Collection: Title: Rockies vs. Dodgers Game Highlights Abstract: Rockies vs. Dodgers Game Highlights (4/4/23) | MLB Highlights

Clip-Level Video Description Generation Using GPT-4V

Output: 0:00:00 - 0:00:14: Video begins with a montage of the Colorado Rockies and the Los Angeles Dodgers players warming up for their game at Dodger Stadium over the backdrop of the Los Angeles skyline.

[Figure 115]_images/imageFile115.png>)

Input Prompt:

[Figure 116]_images/imageFile116.png>)

[Figure 117]_images/imageFile117.png>)

[Figure 118]_images/imageFile118.png>)

0:00:15 - 0:00:30: Dodge.rs pitcher, Julio Urias, makes his second start of the season and deals a powerful pitch towards the Rocky's batter. As the batter swings and makes contact, the ball heads down the line under the glove of the Dodgers and is marked as a double for the Rockies.

00:00:00 00:00:02 00:09:07

You are an expert in understanding scene transitions based on visual features in a video. For the given sequence of images per timestamp, identify different scenes in the video. Generate audio description for each scene with time ranges.

0:00:31 - 0:00:46: Urias successfully strikes out

Figure 4. An example of MM-VID’s execution flow. Given a baseball video, MM-VID provides an estimated video length, and then invokes scene detection and ASR tools, and collects external knowledge. Then, we generate clip-level video descriptions by using GPT-4V. GPT-4V takes the video frames and the text prompt as input, and outputs the video descriptions. Finally, given the clip-level descriptions, video metadata and ASR, we use GPT-4 to generate a coherent script for the input video. Figures 7-9 show continued outputs. The original video is available at https://www.youtube.com/watch?v=-pNe0p4H8ec

MM-VID explains that this information was derived from the commentator’s remarks during the game. The examples illustrate MM-VID’s multimodal reasoning capabilities, where it integrates both visual and auditory cues to provide contextually accurate responses to user queries.

Hour-Long Video Comprehension. Figures 10-13 demonstrate MM-VID’s capabilities in processing lengthy videos. In this example, MM-VID effectively analyzes a documentary video spanning approximately 50 minutes in duration. For simplicity, the intermediate outputs are omitted in the figures, and only the final generated script is presented. We observe that MM-VID is able to generate a long script with the corresponding timestamps to represent the documentary video. By leveraging this generated script as contextual information, MM-VID is equipped to perform a range of tasks, including summarizing the lengthy video, addressing specific queries raised within the video, and indexing pivotal moments.

Multi-Video Episodic Analysis. MM-VID’s proficiency in handling extensive video content can be expanded to encompass multiple lengthy videos, as illustrated in Figures 14-16. In these examples, we upload multiple episodes to MM-VID, showcasing its ability to perform a variety of complex tasks. MM-VID exhibits the capability to summarize the video series, engage in cross-episode reasoning, provide detailed descriptions of character journeys across multiple episodes, and facilitate grounded QA interactions. Character Identification. We found that incorporating visual prompts enhances the quality of script generation, particularly with regards to character identification. In Figure 17, we illustrate this by providing MM-VID with additional inputs consisting of characters’ face photos and their corresponding names. MM-VID effectively utilizes these visual prompts to identify the characters depicted in the video, based on the provided face photos. As a result, the script generation process is notably improved, ensuring more accurate and contextually relevant descriptions of characters and their interactions within the video content.

Speaker Identification. Our exploration has revealed another valuable application of visual prompting in enhancing the quality of Automatic Speech Recognition (ASR). In Figures 18-19, we highlight a scenario where conventional ASR struggles to accurately recognize the number of speakers and their identities in the video. Visual prompting plays a pivotal role in enhancing ASR performance by providing contextual cues to identify individuals and attribute speech to specific speakers. This improvement ensures more precise transcriptions, enabling a more accurate representation of the dialogue and interactions within the video content.

Audio Description Generation. Audio descriptions [26, 57] play a crucial role in making videos accessible to individuals who are blind, have low vision, or face difficulties in visually understanding the content. These descrip-

tions provide contextual narration of meaningful visual elements, clarify speakers, and convey the essence of visual information within a video. In our experiments, we also explore MM-VID’s performance in audio description generation. We experiment with videos where there is limited or no speech content. In Figure 20, we showcase an example featuring a short film of Mr. Bean taking an exam, which primarily lacks speech. Without ASR inputs, MM-VID processes the video and generates a detailed script. This shows MM-VID’s versatility in handling various types of video content and its potential in creating inclusive and accessible multimedia content.

Self-Refinement. While the generated script offers a comprehensive understanding of video content, our experiments have unveiled occasional inaccuracies, especially in cases involving blurry or low-resolution video frames, as demonstrated in Figure 21. In this example, MM-VID mistakenly identifies a bird as a rock due to the challenges posed by the video’s visual quality. To address such inconsistencies and elevate the overall accuracy of the generated script, we employ a self-refinement approach [45, 58, 80]. This involves revising the script based on both the initially generated script and a concurrently generated video summary. Through this process, MM-VID is able to rectify errors and inaccuracies, resulting in a more refined output.

Fast-Changing Short Videos. In Figure 22, we present an example of our experimentation with fast-changing shortform videos, such as those found on platforms like TikTok. Short videos often feature non-standard frame sizes and significantly shorter durations compared to conventional videos. Remarkably, MM-VID excels at accurately describing the cooking recipes depicted in these short videos, despite the distinct characteristics of such content.

These examples demonstrate the versatility of MM-VID in processing a diverse array of video content. Whether dealing with lengthy documentaries, episodic series, or short-form clips, MM-VID adapts seamlessly to the unique attributes of each video type, consistently delivering meaningful and contextually relevant descriptions.

##### 6.3. Applications to Interactive Environments

In the following section, we evaluate MM-VID when applying to the context of streaming inputs. MM-VID serves as an agent in an interactive environment, continually receiving streaming video frames as the inputs.

Embodied Agent. Figure 23 illustrates an example where MM-VID is applied to an egocentric video captured by a head-mounted camera. This video, collected from Ego4D dataset [25], provides a brief glimpse into the wearer’s daily life within their home environment. Remarkably, MM-VID showcases its capability in understanding such video content and assists the user in a few practical tasks. Specifically, MM-VID helps the user locate items like the pink

Table 1. Questionnaire for the group with visual impairments. Participants listen to a video and subsequently assign scores (ranging from 0 to 10) for distinct auditory criteria.

Effectiveness of Delivery: If the original audio and the embedded AD are effectively presented?

Informative: Is it easy to follow the storyline? Does the AD provide context and background information when necessary?

Audio Quality: Is the overall audio production quality good? Overall Satisfaction: Are you satisfied with the overall AD experience?

jacket and the laptop within the home. Additionally, it generates a list of the user’s activities within a specified time range, offering insights into the wearer’s daily routine.

Playing Video Games. Figures 24-27 demonstrate the results of applying MM-VID to a Mario video game [4]. In these experiments, our agent consistently receives three video frames as states and calculates the next possible control action. Remarkably, our agent displays an understanding of the specific video game dynamics and generates reasonable action controls to play the game effectively. These examples highlight MM-VID’s ability to comprehend and navigate in an interactive gaming environment. Interested readers may find the full gameplay demonstration on our project website.

GUI Navigation. Figures 28-32 provide the demonstration of MM-VID’s performance in the GUI navigation scenario. In this context, the agent continually receives iPhone screenshots and previous user actions as states. The agent effectively predicts the possible next steps in the user’s journey, which may include clicking on the correct shopping apps, initiating searches for items of interest, and ultimately placing an order. These results demonstrate MM-VID’s remarkable ability to interact with graphical user interfaces, facilitating seamless and intelligent navigation through digital interfaces.

##### 6.4. User Study

We explore the potential of MM-VID for people who are blind or have low vision. Audio description (AD) [26, 57] provides an auditory narration integrated into the video’s soundtrack, offering important visual details that may not be discernible from the main video soundtrack. Such descriptions play a pivotal role in conveying essential visual content to those with visual impairments.

To assess the efficacy of MM-VID in generating audio descriptions (AD), we conduct a user study. We invited 9 participants for the evaluation. 4 participants were either blind or had low vision, while the remaining 5 had normal

Table 2. Questionnaire for the group with normal vision. Participants view a video and subsequently assign scores (ranging from 0 to 10) for various auditory and visual criteria.

Clarity: Are the visual elements clearly and accuratetly described? Conciseness: Does the AD convey essential visual information without overloading the user?

Timing and Synchronization: Are the original audio and the embedded AD effectively presented? Does the AD properly synchronize with visual contents?

Informative: Is it easy to follow the storyline? Does the AD provide context and background information when necessary?

Audio Quality: Is the overall audio production quality good? Overall Satisfaction: Are you satisfied with the overall AD experience?

vision. All the participants have normal hearing. For the purposes of the experiments, we segregated participants into two distinct groups: (i) Group with visual impairments, and (ii) Group with normal vision.

###### 6.4.1 Evaluation Procedure

Our experiments utilize a curated set of videos, which are mainly suggested by the American Council of the Blind2. We also collected accessibility videos from YouTube3. For every video used in our evaluation, participants are exposed to two versions: the first containing human-crafted AD and the second powered by MM-VID-generated AD. Both renditions are narrated using text-to-speech (TTS) technology.

We have designed two questionnaires for the two groups, referenced in Table 1 and Table 2, respectively. Participants with visual impairments are instructed to base their evaluation exclusively on auditory cues. In contrast, those with normal vision are instructed to consider both visual and auditory elements.

The assessment adopts the standardized Likert scale for ratings. For each posed question, participants are guided to assign a score ranging from 0 to 10, with higher values indicating more favorable feedback. Furthermore, participants are urged to share feedback and remarks concerning their overall experience.

###### 6.4.2 Results on the Group with Visual Impairments

We utilized 3 different videos for our evaluation, with durations of 1 minute, 1 minute 42 seconds, and 2 minutes

2The Audio Description Project: https://adp.acb.org/ 3Apple Accessibility: https://www.youtube.com/watch?v=

SL7YSqlEd8k

42 seconds, respectively. Each of the 4 participants with visual impairment was well versed with screen reader and other common accessibility tools. After listening to the audio descriptions for each video, they were asked to respond to the 4 questions outlined in Table 1.

###### Hypotheses and Results

- H1: The MM-VID-generated audio description and original video dialogues are effectively presented to the participants. Results: Using the Likert scale (0=Not Effective to 10=Most Effective) the participants rated the effectiveness of the delivery of human-crafted AD and MM-VIDgenerated AD. On average, participants gave 7.14 ± 1.39 for MM-VID-generated AD and 8.33 ± 0.90 for humancrafted AD, which shows a MM-VID-generated AD very close to human-crafted one in terms of effective delivery (Figure 5).
- H2: Participants are able to follow the main story line of the video based on MM-VID-generated audio description only. Results: Using the Likert scale (0=Not Informative to 10=Highly Informative) the participants rated the informativeness of human-crafted AD and MM-VID-generated AD. On average, participants gave 7.14 ± 1.16 for MMVID-generated AD and 9.29 ± 0.91 for human-crafted AD, which shows little difference in informativeness between MM-VID-generated AD and human-crafted one (Figure 5).
- H3: MM-VID-generated AD and human-crafted AD are close in terms of voice and audio quality. Results: Using the Likert scale (0=Low Quality to 10=High Quality) the participants rated the voice and audio quality on average as 8.91 ± 1.23 for MM-VID-generated AD and 9.07 ± 0.65 for human-crafted AD. This minimal difference between the scores indicates the close-to-human voice and audio quality of MM-VID-generated AD (Figure

- 5).

###### Discussion:

The results show that the participants’ overall satisfaction of MM-VID-generated ADs was on average around 2 points less than human-crafted ones in the Likert scale (0=Not Satisfied to 10=Highly satisfied) (Figure 5). Some of the difficulties indicated by participants while listening to MM-VID-generated ADs were 1) occasional overlaps between AD audio and original video dialogues 2) wrong descriptions due to hallucinations of GPT-4V(ision). Regardless of the difference in overall satisfaction, all the participants agreed that MM-VID-generated AD can pro-

Results on Group with Visual Impairments

10

8

6

Rating

4

Human-written AD

2

MMVid-generated AD

0

Effect.Delivery Informative AudioQuality Satisfaction

Figure 5. Results on the group with visual impairments. MMVID-generated AD is close to human-generated ones in terms of audio quality and effectiveness of delivery. However, MMVID’s AD yields lower satisfaction levels compared to the humangenerated ones. This was primarily attributed to occasional overlaps between the audio descriptions and the video dialogues.

vide a cost-effective and scalable solution. Thus, millions of videos that cannot afford to be professionally audio described, can be auto-processed by a tool like MM-VID to make them accessible to the visual-impaired community.

###### 6.4.3 Results on the Group with Normal Vision

For sighted individuals, we used the same set of videos as we used for individuals with visual impairments. All of our 5 participants answered to 6 questions listed in Table 2 after watching videos embedded with MM-VID-generated AD as subtitles and audio track.

###### Hypotheses and Results

- H1: The MM-VID-generated AD is accurate and conveys essential information without overloading the listener. Results: The sighted individuals rated the clarify and accuracy of MM-VID-generated AD as 7.83 ± 1.24 and human-curated AD as 8.9 ± 0.74 on average, using the Likert scale (0=Not Accurate to 10=Most Accurate). In terms of conciseness, the participants on average gave 8.73±0.49 for the MM-VID-generated AD and 9.16±0.54 for human-curated AD based on the Likert scale (0=Not concise to 10=Most concise). These results indicate MM-VID-generated ADs are close to human-curated ones in terms of accuracy and conciseness (Figure 6).
- H2: The MM-VID-generated ADs are in sync with visual content and do not overlap with other dialogues ensuring listeners can follow the story line.

Results: Participants gave on average 8.90 ± 0.90 and

- 7.97 ± 1.54 to human-crafted AD and MM-VID-generated AD respectively using the Likert scale (0=Not Informative to 10=Highly Informative). Human-crafted AD and MM-VID-generated AD received 8.59 ± 0.95 and
- 8.53 ± 0.58 respectively on the aspect of timing and synchronization using the Likert scale (0=Not Effective to 10=Most Effective). These indicates while listening to MM-VID-generated ADs participants were able to follow main story line and found the audios are in sync with video content very close to that of human-crafted ADs (Figure 6).

H3: The voice and audio quality of MM-VID-generated ADs are close to human-crafted ADs.

Results: The results are very similar to results on group with visual impairments. Sighted participants rated the voice and audio quality on average as 8.30±0.89 for MMVID-generated AD and as 8.93 ± 0.32 for human-crafted AD. Therefore the voice and audio experience did not degrade much while listening to MM-VID-generated ADs compare to human-crafted ADs (Figure 6).

###### Discussion:

The evaluations on sighted individuals helped to cross verify the hypotheses of individuals with visual impairments, that are based on audio cues only. Although the overall satisfaction points for sighted participants with MM-VID-generated ADs was on average <1 points lower than human-generated ADs (Figure 6), the overall satisfaction points for participants who were blind was worse. This is expected because sighted individuals had access to both audio and video modalities but individuals with visual impairments did not. We also believe the reason for lower overall satisfaction, may have been the lack of practice listening to auto generated ADs. Some of the users also mentioned they have preference between pitches of voice and number of concurrent audio channels. These may add to the reason of lower overall satisfaction.

###### 6.4.4 Participant Feedback

We present a collection of interview quotes from our participants who were visually impaired, in which they share their personal experiences and insights about the audio descriptions (AD) generated by MM-VID. The participants expressed a unanimous desire to continue utilizing this AD generation service in the future, highlighting its exceptional quality (“Nearly perfect”), intricate details (“favorite was the details”), extensive applicability (“allowed me to follow anything visual”), and the profound impact it has on them (“I did not depend on someone else”). Below, we provide additional quotes for further insight.

P1: “I understand what is going on very quickly

Results on Group with Normal Vision

10

8

6

Rating

4

Human-written AD

2

MMVid-generated AD

0

ClarityConcisenessInformativeTimingandSync.AudioQualitySatisfaction

Figure 6. Results on the group with normal vision. MM-VIDgenerated AD was on average <1 points lower than humangenerated ADs. The participants were able to follow the main story line and the audios are in sync with the video content.

and I did not depend on someone else.”

- P2: “If it’s AI-generated, there are so many places it’s not available, and we need it there.”

- P2: “First time listening to auto-generated AD. As a user, if I am offered this AD, I would take it.”
- P3: “Nearly perfect. Most favorite was the details.”

- P3: “More information helped me follow the storyline.”

- P3: “It allowed me to follow anything visual. It felt natural the way AD describes how the actor interacts with the environment.”

- P3: “I love animal kingdom, and I watch Wild Earth safari virtual tour. I would love to have audio descriptions of Wild Earth videos and daily safaris.”
- P4: “I would like to have auto-generated audio description for live conferences in Microsoft Teams.”

- P4: “It worked best as the original audio had not much value.”

Despite the positive feedback, not all responses were favorable:

P4: “I am skeptical when it becomes subjective. Sometimes I feel they make up stories which is not good.”

P4: “After listening to the human-generated AD, I figured I misunderstood parts of the original story.”

P1: “It keeps referring to the same person using their names instead of pronouns.”

P4: “I don’t deal well with overlapped or two parallel audios.”

Interestingly, even those participants who provided critical feedback still rated the MM-VID-generated AD closely to human-generated AD, during the questionnaire sessions. This indicates that, similar to human-curated AD, adapting to MM-VID-generated ADs might necessitate some practice and acclimatization over time.

#### 7. Conclusion

We have presented MM-VID, a system that synergizes with GPT-4V for advancing video understanding. MMVID employs GPT-4V to transcribe video content into long and detailed scripts, thereby enriching LLMs with advanced video understanding capabilities. Experimental results demonstrate the effectiveness of MM-VID in addressing challenging tasks, including comprehension of hourlong videos, analysis across multiple episodes, identification of characters and speakers, and interaction with video games and graphical user interfaces.

Beyond the development of the MM-VID system, we conducted an extensive user study, drawing feedback from a varied group of participants. The outcomes of this study indicated that the audio descriptions generated by MM-VID closely mirror the quality of those crafted by humans. In our future work, we plan to explore SoM [76] and object tracking techniques to enhance various tasks and functionalities.

##### Acknowledgment

We are deeply grateful to OpenAI for providing access to their exceptional tool [3, 51–53]. We are profoundly thankful to Misha Bilenko for his invaluable guidance and support. We also extend heartfelt thanks to our Microsoft colleagues for their insights, with special acknowledgment to Cenyu Zhang, Saqib Shaikh, Ailsa Leen, Jeremy Curry, Crystal Jones, Roberto Perez, Ryan Shugart, Anne Taylor for their constructive feedback.

#### References

- [1] Dota 2. https://openai.com/research/dota-2,

2017. 2

- [2] Azure cognitive services apis. https://azure. microsoft . com / products / cognitive services, 2023. 4
- [3] Chatgpt can now see, hear, and speak. https://openai. com/blog/chatgpt-can-now-see-hear-andspeak, 2023. 3, 10
- [4] Pygame library. https://www.pygame.org/, 2023. 2, 7, 31, 32, 33, 34
- [5] Pyscenedetect: Video scene cut detection and analysis tool. https://www.scenedetect.com/, 2023. 4

- [6] Vlog: Video as a long document. https://github. com/showlab/VLog, 2023. 3
- [7] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022. 2
- [8] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 2
- [9] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia Schmid. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6836–6846,

2021. 2, 3

- [10] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738,

2021. 2, 3

- [11] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems, 35:24639–24654, 2022. 2
- [12] Tadas Baltruˇsaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: A survey and taxonomy. IEEE transactions on pattern analysis and machine intelligence, 41(2):423–443, 2018. 2
- [13] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, volume 2, page 4, 2021. 2, 3
- [14] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019. 2, 3
- [15] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 2, 3
- [16] Harrison Chase. Langchain. https://langchain. readthedocs.io/, 2023. 3
- [17] Sanyuan Chen, Yu Wu, Zhuo Chen, Jian Wu, Jinyu Li, Takuya Yoshioka, Chengyi Wang, Shujie Liu, and Ming Zhou. Continuous speech separation with conformer. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5749–5753. IEEE, 2021. 2
- [18] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. 3
- [19] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. 2, 3

- [20] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6824–6835, October 2021. 2, 3
- [21] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6202–6211, 2019. 2, 3
- [22] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. Violet: End-to-end video-language transformers with masked visual-token modeling. arXiv preprint arXiv:2111.12681, 2021. 2, 3
- [23] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. An empirical study of end-to-end video-language transformers with masked visual modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22898– 22909, 2023. 2, 3
- [24] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 2
- [25] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022. 6, 30
- [26] Tengda Han, Max Bain, Arsha Nagrani, G¨ul Varol, Weidi Xie, and Andrew Zisserman. Autoad: Movie description in context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18930– 18940, 2023. 2, 6, 7
- [27] Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and Li Fei-Fei. Large-scale video classification with convolutional neural networks. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 1725–1732, 2014. 2
- [28] Hildegard Kuehne, Hueihan Jhuang, Est´ıbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In 2011 International conference on computer vision, pages 2556–2563. IEEE, 2011. 3
- [29] Anna Kukleva, Makarand Tapaswi, and Ivan Laptev. Learning interactions and relationships between movie characters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9849–9858, 2020. 2
- [30] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7331–7341, June

2021. 2, 3

- [31] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering. In EMNLP, 2018. 2, 3
- [32] Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. Tvr: A large-scale dataset for video-subtitle moment retrieval. In ECCV, 2020. 3
- [33] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. Multimodal foundation models: From specialists to general-purpose assistants. arXiv preprint arXiv:2309.10020, 2023. 3
- [34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 3
- [35] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 3
- [36] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. Hero: Hierarchical encoder for video+ language omni-representation pre-training. arXiv preprint arXiv:2005.00200, 2020. 2, 3
- [37] Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, and Lijuan Wang. Lavender: Unifying videolanguage understanding as masked language modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23119–23129, 2023. 2, 3
- [38] Linjie Li, Jie Lei, Zhe Gan, Licheng Yu, Yen-Chun Chen, Rohit Pillai, Yu Cheng, Luowei Zhou, Xin Eric Wang, William Yang Wang, et al. Value: A multi-task benchmark for video-and-language understanding evaluation. In 35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks, 2021. 2, 3
- [39] Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. Swinbert: End-to-end transformers with sparse attention for video captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17949– 17958, 2022. 2, 3
- [40] Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Z XU, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. Advances in Neural Information Processing Systems, 35:7575–7586, 2022. 2, 3
- [41] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 3
- [42] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022. 2, 3
- [43] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023. 3
- [44] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video

- understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 3
- [45] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023. 6
- [46] EVERINGHAM Mark. Hello! my name is... buffy’automatic naming of characters in tv video. In Proceedings of British Machine Vision Conference, 2006, 2006. 2
- [47] Microsoft. Bingchat. https://www.microsoft.com/ en-us/edge/features/bing-chat, 2023. 2
- [48] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. In ICCV,

2019. 2

- [49] Arsha Nagrani, Joon Son Chung, and Andrew Zisserman. Voxceleb: a large-scale speaker identification dataset. arXiv preprint arXiv:1706.08612, 2017. 2
- [50] Arsha Nagrani and Andrew Zisserman. From benedict cumberbatch to sherlock holmes: Character identification in tv series without a script. arXiv preprint arXiv:1801.10442,

2018. 2

- [51] OpenAI. Gpt-4 technical report. 2023. 2, 3, 10
- [52] OpenAI. Gpt-4v(ision) system card. 2023. 2, 3, 10
- [53] OpenAI. Gpt-4v(ision) technical work and authors. https: //cdn.openai.com/contributions/gpt-4v. pdf, 2023. 10
- [54] Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023. 2
- [55] Karine Pires and Gwendal Simon. Youtube live and twitch: a tour of user-generated live streaming systems. In Proceedings of the 6th ACM multimedia systems conference, pages 225–230, 2015. 2
- [56] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pages 28492–

28518. PMLR, 2023. 3

- [57] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3202–3212, 2015. 2, 6, 7
- [58] Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. 6
- [59] Mustafa Shukor, Corentin Dancette, Alexandre Rame, and Matthieu Cord. Unified model for image, video, audio and language tasks. arXiv preprint arXiv:2307.16184, 2023. 2
- [60] Cees GM Snoek and Marcel Worring. Multimodal video indexing: A review of the state-of-the-art. Multimedia tools and applications, 25:5–35, 2005. 2

- [61] David Snyder, Daniel Garcia-Romero, Gregory Sell, Alan McCree, Daniel Povey, and Sanjeev Khudanpur. Speaker recognition for multi-speaker conversations using x-vectors. In ICASSP 2019-2019 IEEE International conference on acoustics, speech and signal processing (ICASSP), pages 5796–5800. IEEE, 2019. 2
- [62] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449, 2023. 2, 3
- [63] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 3
- [64] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [65] Junke Wang, Dongdong Chen, Chong Luo, Xiyang Dai, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. arXiv preprint arXiv:2304.14407, 2023. 3
- [66] Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Kevin Qinghong Lin, Satoshi Tsutsui, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, et al. All in one: Exploring unified video-language pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6598–6608, 2023. 2, 3
- [67] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents,

2023. 2

- [68] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, highquality multilingual dataset for video-and-language research. In The IEEE International Conference on Computer Vision (ICCV), October 2019. 2, 3
- [69] Yunbo Wang, Mingsheng Long, Jianmin Wang, and Philip S Yu. Spatiotemporal pyramid network for video action recognition. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 1529–1538, 2017. 2, 3
- [70] Chao-Yuan Wu and Philipp Krahenbuhl. Towards long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1884–1894, 2021. 2
- [71] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. arXiv preprint arXiv:2212.00280, 2022. 3
- [72] Yujia Xie, Luowei Zhou, Xiyang Dai, Lu Yuan, Nguyen Bach, Ce Liu, and Michael Zeng. Visual clues: Bridging vision and language foundations for image paragraph captioning. Advances in Neural Information Processing Systems, 35:17287–17300, 2022. 3

- [73] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016. 3
- [74] Peng Xu, Xiatian Zhu, and David A Clifton. Multimodal learning with transformers: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 2
- [75] Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. Exploring large language models for communication games: An empirical study on werewolf. arXiv preprint arXiv:2309.04658, 2023. 2
- [76] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 10
- [77] Ziyi Yang, Yuwei Fang, Chenguang Zhu, Reid Pryzant, Dongdong Chen, Yu Shi, Yichong Xu, Yao Qian, Mei Gao, Yi-Ling Chen, et al. i-code: An integrative and composable multimodal learning framework. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 10880–10890, 2023. 2
- [78] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v(ision). arXiv preprint arXiv:2309.17421, 2023. 2, 3, 4
- [79] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023. 3, 4
- [80] Zhengyuan Yang, Jianfeng Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. Idea2img: Iterative self-refinement with gpt-4v (ision) for automatic image design and generation. arXiv preprint arXiv:2310.08541, 2023. 6
- [81] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 2
- [82] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. Advances in Neural Information Processing Systems, 34:23634–23651,

2021. 2

- [83] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. arXiv preprint arXiv:2307.02485,

2023. 2

- [84] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 3
- [85] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models, 2022. URL https://arxiv. org/abs/2205.01068. 3

[Figure 119]_images/imageFile119.png>)

[Figure 120]_images/imageFile120.png>)

takes a turn at bat. After a quick scuffle with the ball, he blasts it into left field, scoring yet another home run.

Script Generation Using GPT-4 (continued):

Unknown: Or we could.

- 0:01:59 Nimona: Just talk.
- 0:02:02 Unknown: The the the point. 0:02:03 Nimona: Is whatever your dark heart desires. Boss, yo

Chris Bryant from the Rockies, employing a series of breaking stuff and tempting fastballs at the top of the strike zone.

0:03:11 - 0:03:26: The Dodgers are in a roll. Max Muncie adds another point to the Dodgers lead by sending the ball deep into the right field.

- 0:00:47 - 0:01:02: Next up, KRON from the Rockies pops the ball to the infield. Dodgers' catcher, Will Smith, steps out and catches the ball despite a one-out double, stopping the Rockies from making a score.
- 0:01:03 - 0:01:18: The video pans to Dodgers' lead-off hitter, Mookie Betts, who manages to hit a lead-off single to the centre field, setting a positive start for the Dodgers.

- 0:01:19 - 0:01:34: Will Smith steps up and drives a fly ball to left which goes deep, resulting in him scoring a home run for the third game in a row.

- 0:01:35 - 0:01:50: Scene next highlights Max Muncy's performance who contributes to the Dodgers scoring by cracking a ball to the left-center field which is then picked up by the Rockies' fielder.

- 0:01:51 - 0:02:06: The scene changes to a duo of spectacular plays by the Dodgers. First, a double play managed by shortstop Tovar, swiftly eliminating JD Martinez to curb the first inning. Then in the second inning, we see Urias successfully striking out Rockies' Ryan McMahon.
- 0:02:07 - 0:02:22: Urias faces a setback as McMahon bounces back with a high-quality deep shot to the center field, netting himself a double.

- 0:02:23 - 0:02:38: Despite the Rockies improving, Urias ends the inning by striking out the Rockies' third baseman, Ellen Trejo, leaving McMahon stranded at second base.

- 0:02:39 - 0:02:54: The tables turn as the dodgers regain their footing. Mookie Betts smacks the ball into right field and makes his way safely to second base.

- 0:02:55 - 0:03:10: Dodgers player Will Smith

- 0:03:27 - 0:03:42: The Rockies' are unfortunate to lose a run despite a successful hit due to swift coordination between the Dodgers' players.

- 0:03:43 - 0:03:58: Jason Hayward from Dodgers adds to Rockies' dismay by smacking a lead-off shot in the third inning of the game.

- 0:03:59 - 0:04:14: Julio Urias, from the Dodgers, efficiently strikes out the Venezuelans' Luis Montero and Ryan McMahon in quick succession.

- 0:04:15 - 0:04:30: A moment of respite for the

- Rockies as Urias throws a pitch, allowing the batter to hit a ground ball into infield. However, it is quickly fielded, ending their chances. The innings end with the Dodgers in the lead by four runs.

0:04:31 - 0:04:46: The scene captures dodgers' Max Muncy blasting a hit into the field, which bumps the Dodgers' lead to four.

0:04:47 - 0:05:02: Despite a promising start for the

- Rockies at the top of the third, their momentum slows down as they lose a run due to a double play induced by Urias.

0:05:03 - 0:05:18: Hayward makes a fantastic lead-off shot for the Dodgers' in the third inning, continuing their advantageous streak.

0:05:19 - 0:05:34: Back at the pitching mound, Urias expertly strikes out the Rockies' Luis Montero and leaves the Rockies scoreless for 6 games.

- 0:05:35 - 0:05:50: Urias's superior pitching continues as he strikes out Ryan McMahon and skillfully fields a sharp grounder ending the fourth inning, serving a huge blow to the Rockies.

Following is the generated script for this video.

- 0:05:51 - 0:06:06: Scene cuts to Dodgers' Mookie Betts, who despite an injury, hammers a hit into the field, landing him a double.
- 0:06:07 - 0:06:22: Despite leading the series, Show me the most exciting moment in this video

Figure 7. An example of MM-VID’s execution flow (continued). We present the full script generated by MM-VID. The original video is available at https://www.youtube.com/watch?v=-pNe0p4H8ec

14

The most exciting moment in this video is when Will Smith hits a home run in the first inning at timestamp 0:01:19 - 0:01:34 in this video

Compared to the last season, did Will Smith become stronger?

Based on the video, it is suggested that Will Smith has become stronger compared to the last season, as there is a different sound when the ball leaves his bat.

Will Smith hits a home run at 0:01:19 - 0:01:34

[Figure 127]_images/imageFile127.png>)

[Figure 128]_images/imageFile128.png>)

(continued)

###### Script Generation Using GPT-4 (continued):

Unknown: Or we could.

- 0:01:59 Nimona: Just talk.
- 0:02:02 Unknown: The the the point. 0:02:03 Nimona: Is whatever your dark heart desires. Boss, yo

[Figure 129]_images/imageFile129.png>)

Dodgers' catcher Will Smith lifts a high ball into center field which brings Mookie Betts running home, netting the Dodgers another point.

0:06:23 - 0:06:38: The Rockies' show some fight as Chris Brian leads the ninth innings with a double bringing the Rockies trailing the Dodgers by five.

Will Smith hits a home run in the first inning at timestamp 0:01:19 - 0:01:34

0:06:39 - 0:06:54: A series of unfortunate events for the Rockies as batters Montero and Diaz get directly hit by pitched balls consecutively. Trejo from the Rockies finally manages to hit a fly ball into left field, allowing Montero to make it to the home plate, closing the gap between the two teams to three.

[Figure 130]_images/imageFile130.png>)

Compared to the last season, did Will Smith become stronger?

[Figure 131]_images/imageFile131.png>)

Will Smith hits a home run at 0:01:19 - 0:01:34

Based on the video, it is suggested that Will Smith has become stronger compared to the last season, as there is a different sound when the ball leaves his bat.

- 0:06:55 - 0:07:10: The game comes to a close as Charlie Blackman grounds to the first base, ending the game in favor of Dodgers, after a thrilling and eventful match.
- 0:07:11 - 0:09:09: The scene concludes with the teams leaving the field.

[Figure 132]_images/imageFile132.png>)

How did you know the sound is different?

[Figure 133]_images/imageFile133.png>)

The commentator mentions that there is a different sound when the ball leaves Will Smith's bat, implying that he might have become stronger.

[Figure 134]_images/imageFile134.png>)

Can you summarize the game?

The game was a thrilling and eventful match between the Colorado Rockies and the Los Angeles Dodgers. The Dodgers dominated the game with superior pitching and powerful hits, while the Rockies showed some fight towards the end but ultimately fell short.

[Figure 135]_images/imageFile135.png>)

Evidence in ASR: 1:27 the track at the wall Smith goes deep 1:29 again third game in a row 1:34 from Big Time trajectory on that one

1:37 Will Smith must have gotten stronger in 1:39 the off season there's a different sound 1:41 now when the ball leaves his Bat so the 1:44 Dodgers on the board quickly two-run 1:46 home run in the first inning for Will 1:48 Smith first time in his career he's gone 1:50 deep three games in a row one two to

[Figure 136]_images/imageFile136.png>)

Show me the most exciting moment in this video

1:52 Muncie is crack the left center field

[Figure 137]_images/imageFile137.png>)

The most exciting moment in this video is when Will Smith hits a home run in the first inning at timestamp 0:01:19 - 0:01:34 in this video

[Figure 138]_images/imageFile138.png>)

Who won the game, and what are their final scores?

Figure 8. An example of MM-VID’s execution flow (continued). Given the generated script as reference, MM-VID achieves various video understanding capabilities, including grounded QA and multimodal reasoning. The original video is available at https://www. youtube.com/watch?v=-pNe0p4H8ec

15

The Los Angeles Dodgers won the game with a final score of 5-2.

Final scores were displayed at 0:07:11 - 0:09:09, and mentioned by the commentator at 0:08:53

Who are the best pitchers in this video?

Julio Urias was the best pitcher in this video, striking out multiple batters and inducing a double play.

Evidence in the Generated Script:

- 0:02:07 - 0:02:22: Urias faces a setback as McMahon bounces back with a high-quality deep shot to the center field, netting himself a double.

- 0:03:59 - 0:04:14: Julio Urias, from the Dodgers, efficiently strikes out the Venezuelans' Luis Montero and Ryan McMahon in quick succession.

- 0:04:47 - 0:05:02: Despite a promising start for the Rockies at the top of the third, their momentum slows down as they lose a run due to a double play induced by Urias.

- 0:05:35 - 0:05:50: Urias's superior pitching continues as he strikes out Ryan McMahon and skillfully fields a sharp grounder ending the fourth inning, serving a huge blow to the Rockies.

When did Julio Urias have his first strikeout?

Julio Urias had his first strikeout in the video at 0:00:31, striking out Chris Bryant from the Rockies.

Julio Urias had his first strikeout at 0:00:31

When did Julio Urias induce a double play?

Julio Urias induced a double play at around 0:02:07 in the video.

Julio Urias induced a double play around 0:02:07

[Figure 150]_images/imageFile150.png>)

The Los Angeles Dodgers won the game with a final score of 5-2.

[Figure 151]_images/imageFile151.png>)

(continued)

Unknown: Or we could.

- 0:01:59 Nimona: Just talk.
- 0:02:02 Unknown: The the the point. 0:02:03 Nimona: Is whatever your dark heart desires. Boss, yo

[Figure 152]_images/imageFile152.png>)

[Figure 153]_images/imageFile153.png>)

Final scores were displayed at 0:07:11 - 0:09:09, and mentioned by the commentator at 0:08:53

Julio Urias had his first strikeout at 0:00:31

[Figure 154]_images/imageFile154.png>)

Who are the best pitchers in this video?

[Figure 155]_images/imageFile155.png>)

When did Julio Urias induce a double play?

[Figure 156]_images/imageFile156.png>)

Julio Urias was the best pitcher in this video, striking out multiple batters and inducing a double play.

[Figure 157]_images/imageFile157.png>)

Julio Urias induced a double play at around 0:02:07 in the video.

Evidence in the Generated Script:

[Figure 158]_images/imageFile158.png>)

- 0:02:07 - 0:02:22: Urias faces a setback as McMahon bounces back with a high-quality deep shot to the center field, netting himself a double.

- 0:03:59 - 0:04:14: Julio Urias, from the Dodgers, efficiently strikes out the Venezuelans' Luis Montero and Ryan McMahon in quick succession.

- 0:04:47 - 0:05:02: Despite a promising start for the Rockies at the top of the third, their momentum slows down as they lose a run due to a double play induced by Urias.

- 0:05:35 - 0:05:50: Urias's superior pitching continues as he strikes out Ryan McMahon and skillfully fields a sharp grounder ending the fourth inning, serving a huge blow to the Rockies.

Julio Urias induced a double play around 0:02:07

[Figure 159]_images/imageFile159.png>)

When did Julio Urias have his first strikeout?

[Figure 160]_images/imageFile160.png>)

Julio Urias had his first strikeout in the video at 0:00:31, striking out Chris Bryant from the Rockies.

- Figure 9. An example of MM-VID’s execution flow (continued). Given the generated script as reference, MM-VID achieves various video understanding capabilities, including grounded QA and multimodal reasoning. The original video is available at https://www. youtube.com/watch?v=-pNe0p4H8ec

16

The Los Angeles Dodgers won the game with a final score of 5-2.

Final scores were displayed at 0:07:11 - 0:09:09, and mentioned by the commentator at 0:08:53

Who are the best pitchers in this video?

Julio Urias was the best pitcher in this video, striking out multiple batters and inducing a double play.

Evidence in the Generated Script:

- 0:02:07 - 0:02:22: Urias faces a setback as McMahon bounces back with a high-quality deep shot to the center field, netting himself a double.

- 0:03:59 - 0:04:14: Julio Urias, from the Dodgers, efficiently strikes out the Venezuelans' Luis Montero and Ryan McMahon in quick succession.

- 0:04:47 - 0:05:02: Despite a promising start for the Rockies at the top of the third, their momentum slows down as they lose a run due to a double play induced by Urias.

- 0:05:35 - 0:05:50: Urias's superior pitching continues as he strikes out Ryan McMahon and skillfully fields a sharp grounder ending the fourth inning, serving a huge blow to the Rockies.

When did Julio Urias have his first strikeout?

[Figure 167]_images/imageFile167.png>)

[Figure 168]_images/imageFile168.png>)

[Figure 169]_images/imageFile169.png>)

[Figure 170]_images/imageFile170.png>)

[Figure 171]_images/imageFile171.png>)

###### Figure 10. Case studies of MM-VID ’s capabilities and application scenarios: hour-long video comprehension. Figures 11-13 show continued outputs. The original video is available at https://www.youtube.com/watch?v=um2Q9aUecy0

- Scene 46: 00:09:03 Close-up of a bug on a leaf.
- Scene 47: 00:09:43 Landscape view of mountains with clouds hovering overhead.
- Scene 48: 00:10:32 Close-up of a bird as it takes flight.
- Scene 49: 00:10:50 Close-up of a gorilla's face hidden behind the leaves.
- Scene 50: 00:11:23 A gorilla is seen searching the dense forest floor.
- Scene 51: 00:12:15 Close-up of a black bird with large eyes.
- Scene 52: 00:13:03 A black bird is seen walking in a jungle path.
- Scene 53: 00:13:29 Bird of prey perched on tree branches in the jungle.
- Scene 54: 00:15:05 Close-up of a bird's eye in the forest.
- Scene 55: 00:16:38 A shot of a gorilla hiding behind a tree in a dense jungle.
- Scene 56: 00:17:50 Two elephants engage in a playful fight near a water body.
- Scene 57: 00:19:20 The leopard is seen slowly moving through the foliage.
- Scene 58: 00:21:51 Bird seen entering a hole in a tree.
- Scene 59: 00:22:22 A red ant crawls on a fern.
- Scene 60: 00:23:32 A peculiar mushroom blooms in the darkness near the forest floor.
- Scene 61: 00:25:07 A deer seen foraging in the foliage.

- Scene 31: 00:06:46 An elephant and a gorilla roam the jungle.
- Scene 32: 00:06:57 The gorilla relocates deeper into the jungle.
- Scene 33: 00:07:05 The gorilla is seen wading in a small body of water.
- Scene 34: 00:07:13 Two gorillas spotted playing in the grass.
- Scene 35: 00:07:27 Admiring the grass, the gorilla takes a deserved break.
- Scene 36: 00:07:32 Aerial view of gorillas and elephants in the marsh.
- Scene 37: 00:07:37 Close-up of the elephant in the water.
- Scene 38: 00:07:43 Two elephants are seen fighting and splashing in the water.
- Scene 39: 00:08:00 A lone elephant is caught grazing in the landscape.
- Scene 40: 00:08:08 Scene transitions to a group of gorillas in their natural habitat.
- Scene 41: 00:08:16 A herd of elephants is spotted in shallow water.
- Scene 42: 00:08:22 Aerial view of a vast landscape with a mountain peak and flying birds.
- Scene 43: 00:08:28 Aerial view of a wide, green landscape with scattered water bodies and flying birds.
- Scene 44: 00:08:34 Aerial view of a mountainous range with a valley and a river.
- Scene 45: 00:08:41 Close-up of gorilla hanging from a tree branch.

[Figure 172]_images/imageFile172.png>)

[Figure 173]_images/imageFile173.png>)

- Figure 11. Case studies of MM-VID ’s capabilities and application scenarios: hour-long video comprehension; We present the full script generated by MM-VID. The original video is available at https://www.youtube.com/watch?v=um2Q9aUecy0

[Figure 174]_images/imageFile174.png>)

[Figure 175]_images/imageFile175.png>)

[Figure 176]_images/imageFile176.png>)

[Figure 177]_images/imageFile177.png>)

[Figure 178]_images/imageFile178.png>)

[Figure 179]_images/imageFile179.png>)

[Figure 180]_images/imageFile180.png>)

[Figure 181]_images/imageFile181.png>)

[Figure 182]_images/imageFile182.png>)

Can you summarize the video?

[Figure 183]_images/imageFile183.png>)

- Figure 12. Case studies of MM-VID ’s capabilities and application scenarios: hour-long video comprehension; MM-VID is able to summarize the video and address specific questions with regard to the video contents. Blue highlights the correct answers. The original video is available at https://www.youtube.com/watch?v=um2Q9aUecy0

What is Cordyceps fungus?

The video is a documentary that explores the diverse and rich ecosystems of jungles and rainforests around the world. It showcases various animal species, their behaviors, and their roles in maintaining the balance of these ecosystems. The documentary also highlights the threats faced by these habitats, such as deforestation and poaching, and emphasizes the importance of preserving them for the health of our planet.

[Figure 184]_images/imageFile184.png>)

[Figure 185]_images/imageFile185.png>)

[Figure 186]_images/imageFile186.png>)

[Figure 187]_images/imageFile187.png>)

Cordyceps fungus is a parasitic fungus that infects and takes control of its host, usually an insect. The fungus manipulates the host's behavior, eventually leading to the host's death. The fungus then erupts from the host's body and releases spores to infect more victims. Cordyceps fungus plays a role in maintaining the balance of ecosystems by preventing any one species from dominating.

19

What are the balance ecosystems shown in this video?

The balance ecosystems shown in this video include the Congo rainforest, New Guinea's jungles, Borneo's ancient jungles, the Philippines' fragmented forests, and the Amazon basin. These ecosystems are maintained through various checks and balances, such as the Cordyceps fungus controlling insect populations and the relationships between different species, like the pitcher plants and the animals they rely on for nutrients.

When can I find Cordyceps fungus in this video?

Cordyceps fungus can be found in the video around the timestamp 00:37:06.

When can I find pitcher plants in this video?

Pitcher plants can be found in the video around the timestamp 00:18:18.

[Figure 199]_images/imageFile199.png>)

[Figure 200]_images/imageFile200.png>)

[Figure 201]_images/imageFile201.png>)

[Figure 202]_images/imageFile202.png>)

[Figure 203]_images/imageFile203.png>)

[Figure 204]_images/imageFile204.png>)

[Figure 205]_images/imageFile205.png>)

[Figure 206]_images/imageFile206.png>)

[Figure 207]_images/imageFile207.png>)

[Figure 208]_images/imageFile208.png>)

[Figure 209]_images/imageFile209.png>)

- Figure 13. Case studies of MM-VID ’s capabilities and application scenarios: hour-long video comprehension; MM-VID is able to localize the specific moments or topics in the lengthy documentary video. Blue highlights the correct answers. The original video is available at https://www.youtube.com/watch?v=um2Q9aUecy0

[Figure 210]_images/imageFile210.png>)

[Figure 211]_images/imageFile211.png>)

Pitcher plants can be found in the video around the timestamp 00:18:18.

20

Unknown: Or we could.

- 0:01:59 Nimona: Just talk.
- 0:02:02 Unknown: The the the point. 0:02:03 Nimona: Is whatever your dark heart desires. Boss, yo

[Figure 212]_images/imageFile212.png>)

[Figure 213]_images/imageFile213.png>)

(continued) the deep web for information. The episode features various tense conversations and political strategizing, with characters navigating complex relationships and making crucial decisions.

[Figure 214]_images/imageFile214.png>)

[Figure 215]_images/imageFile215.png>)

[Figure 216]_images/imageFile216.png>)

Loading video… Video length is 46 minutes 54 seconds.

[Figure 217]_images/imageFile217.png>)

What are the main challenges Frank facing in the two videos?

[Figure 218]_images/imageFile218.png>)

[Figure 219]_images/imageFile219.png>)

[Figure 220]_images/imageFile220.png>)

Main challenges Frank faces in the two episodes:

[Figure 221]_images/imageFile221.png>)

- - Erasing all leads connecting him and Doug to their past actions, particularly their involvement in Peter Russo's death.
- - Managing the political fallout from Zoe's death and preventing any further investigation into his involvement.
- - Navigating the complex relationships and political strategies within the government, particularly in relation to the Whip race and the US-China talks.
- - Balancing his personal life with his new role as Vice President, including dealing with the increased security measures and lack of privacy.

[Figure 222]_images/imageFile222.png>)

Loading video… Video length is 48 minutes 30 seconds.

[Figure 223]_images/imageFile223.png>)

Can you summarize the two episodes?

[Figure 224]_images/imageFile224.png>)

- Episode 1 Summary: In the first episode of House of Cards Season 2, Frank Underwood prepares for his promotion to Vice President. Doug Stamper works on erasing any leads that could connect him and Frank to their past actions. Claire faces a legal battle against Gillian, while Zoe considers working with Frank again. Throughout the episode, various characters engage in tense conversations and political maneuvering, with Frank ultimately being sworn in as Vice President in a private ceremony.
- Episode 2 Summary: In the second episode, Frank is announced as Vice President and faces obstacles in his new role. The US engages in talks with China, which fall apart over cyber warfare issues. Jackie Sharp plans to run for the new whip position, facing challenges along the way. Meanwhile, Lucas investigates Zoe's death and delves into

[Figure 225]_images/imageFile225.png>)

When did Frank win the promotion? include the timestamps

[Figure 226]_images/imageFile226.png>)

###### Frank wins the promotion to Vice President in Season 2 Episode 1, with the swearing-in ceremony taking place between 0:02:24 and 0:02:36 in Episode 2.

[Figure 227]_images/imageFile227.png>)

[Figure 228]_images/imageFile228.png>)

Describe Frank's journey in the two videos.

- Figure 14. Case studies of MM-VID ’s capabilities and application scenarios: multi-video episodic analysis; Given multiple video files as inputs, MM-VID can summarize the video series, perform cross-episodic reasoning, describe a specific character’s journey across multiple videos. Figure 15 shows continued outputs. Blue highlights the correct answers. The two videos are from House of Cards, Season 2: episode 1 and episode 2.

Frank's journey in these two episodes involves him preparing for his new role as Vice President while trying to maintain control over various situations. He deals with the consequences of his past actions, such as managing Doug Stamper's attempts to erase their tracks and handling the fallout from Zoe's death. Frank also navigates the political landscape, working with Jackie to secure her position as Whip and engaging in tense negotiations with China. Throughout these challenges, Frank continues to demonstrate his cunning and ruthlessness in achieving his goals.

[Figure 229]_images/imageFile229.png>)

21

[Figure 231]_images/imageFile231.png>)

Describe Frank's journey in the two videos.

[Figure 232]_images/imageFile232.png>)

Frank's journey in these two episodes involves him preparing for his new role as Vice President while trying to maintain control over various situations. He deals with the consequences of his past actions, such as managing Doug Stamper's attempts to erase their tracks and handling the fallout from Zoe's death. Frank also navigates the political landscape, working with Jackie to secure her position as Whip and engaging in tense negotiations with China. Throughout these challenges, Frank continues to demonstrate his cunning and ruthlessness in achieving his goals.

- Figure 15. Case studies of MM-VID ’s capabilities and application scenarios: multi-video episodic analysis; Given multiple video files as inputs, MM-VID can summarize the video series, perform cross-episodic reasoning, describe a specific character’s journey across multiple videos. Blue highlights the correct answers. The two videos are from House of Cards, Season 2: episode 1 and episode 2.

Unknown: Or we could.

- 0:01:59 Nimona: Just talk.
- 0:02:02 Unknown: The the the point. 0:02:03 Nimona: Is whatever your dark heart desires. Boss, yo

Who win the championship?

Ash wins the championship.

[Figure 235]_images/imageFile235.png>)

[Figure 236]_images/imageFile236.png>)

[Figure 237]_images/imageFile237.png>)

Did Pikachu use 10000000 Volts move?

[Figure 238]_images/imageFile238.png>)

[Figure 239]_images/imageFile239.png>)

###### Yes, Pikachu used the 10,000,000 Volts move (Timestamp: 00:20:09 - 00:20:14 in Episode 131).

[Figure 240]_images/imageFile240.png>)

[Figure 241]_images/imageFile241.png>)

Loading video… Video length is 23 minutes 36 seconds.

[Figure 242]_images/imageFile242.png>)

[Figure 243]_images/imageFile243.png>)

[Figure 244]_images/imageFile244.png>)

[Figure 245]_images/imageFile245.png>)

Did Pikachu use Gigantamax?

[Figure 246]_images/imageFile246.png>)

[Figure 247]_images/imageFile247.png>)

Loading video… Video length is 21 minutes 57 seconds.

###### Yes, Pikachu used Gigantamax. (Timestamp: 00:03:45 - 00:03:49 in Episode 132)

[Figure 248]_images/imageFile248.png>)

[Figure 249]_images/imageFile249.png>)

Can you summarize the two episodes?

[Figure 250]_images/imageFile250.png>)

- In Episode 131, Ash and Leon face off in the finals of the Pokémon World Coronation Series Masters Eight Tournament. The intense battle sees various Pokémon, including Rillaboom, Dragonite, and Cinderace, fighting fiercely. The combined power of Pikachu's Z-Move and Charizard's Gigantamax attack causes Eternatus to appear unexpectedly.
- In Episode 132, the final round of the tournament continues with the appearance of Eternatus, which recharges Leon and Ash's Dynamax bands. Ash and Pikachu face Leon's Gigantamax Cinderace and Charizard in a thrilling battle. With the support of their friends and Pokémon, Ash and Pikachu emerge victorious, making Ash the new World Champion.

[Figure 251]_images/imageFile251.png>)

Show me the keyframe when Pikachu defeats Charizard

[Figure 252]_images/imageFile252.png>)

The keyframe when Pikachu defeats Charizard is at timestamp 00:16:02-00:16:03 in Episode 132.

[Figure 253]_images/imageFile253.png>)

Figure 16. Case studies of MM-VID ’s capabilities and application scenarios: multi-video episodic analysis; Given multiple video files as inputs, MM-VID can summarize the video series, and localize specific moments across multiple videos. Blue highlights the correct answers. The two videos are from Pok´emon Journeys: episode 131 and episode 132.

[Figure 254]_images/imageFile254.png>)

[Figure 255]_images/imageFile255.png>)

(continued) 0:00:06 - 0:00:08: Nefer looks down the cliff's edge, revealing her concerned reaction.

[Figure 257]_images/imageFile257.png>)

[Figure 258]_images/imageFile258.png>)

Unknown: Or we could.

- 0:01:59 Nimona: Just talk.
- 0:02:02 Unknown: The the the point. 0:02:03 Nimona: Is whatever your dark heart desires. Boss, yo

- 0:00:12 - 0:00:14: A shot of a rock with a golden ankh necklace sets an ominous tone.

- 0:00:15 - 0:00:16: Transition to a worried Nefer under the open sky.

- 0:00:17 - 0:00:18: Faraon overlooks Thut lying on the rocky ground.
- 0:00:18 - 0:00:24: Various interactions between Thut, Nefer, and Faraon in the desert transpire.

- 0:00:24 - 0:00:30: Nefer is hugged by Faraon as Thut watches closely.

- 0:00:30 - 0:00:36: Thut shares a moment with Nefer before seating himself beside a frog as Faraon joins Nefer.

- 0:00:36 - 0:00:42: Nefer and Faraon approach a slightly confused Thut.

- 0:00:42 - 0:00:48: Thut and Nefer stand alone exchanging a few words.

- 0:00:48 - 0:00:54: Faraon joins Thut and Nefer, looking at a ring Thut holds.
- 0:01:00 - 0:01:06: Thut presents a ring between his and Nefer's fingers.

- 0:01:06 - 0:01:12: Thut listens as Nefer talks excitingly about the ring.

- 0:01:12 - 0:01:18: Faraon oversees a close moment between Thut and Nefer.

0:01:18 - 0:01:24: Nefer and Thut stand in front of an audience as a small girl with a snake appears. 0:01:24 - 0:01:30: Thut and Nefer perform in front of a crowd. 0:01:42 - 0:01:48: Thut and Nefer share a kiss as the crowd cheers on, transitioning to the girl with her crocodile. 0:01:54 - 0:02:00: Thut and Nefer perform on stage with others in a palace courtyard. 0:02:12 - 0:02:18: Multiple performances ensue in the courtyard, ending with Thut and Nefer sharing a gaze. 0:02:18 - 0:02:24: Thut and Nefer continue their romantic interaction under the watchful eyes of their audience. 0:02:24 - 0:02:30: The dancing in the courtyard continues. 0:02:30 - 0:02:36: Thut, Nefer, and Faraon perform on stage as their performers enjoy their display.

- 0:02:36 - 0:02:43: As the sun sets, Thut and Nefer conclude their musical performance on a balcony.

[Figure 259]_images/imageFile259.png>)

Loading video… Video length is 2 minutes 43 seconds.

[Figure 260]_images/imageFile260.png>)

[Figure 261]_images/imageFile261.png>)

Invoke tools: Scene Detection, Metadata Collection

Script Generation: 0:00:00-0:00:03: A shocked girl in Egyptian attire stands in the desert. 0:00:04-0:00:09: The girl looks down a cliff's edge, then up towards the sky. 0:00:09-0:00:11: An Egyptian headdress-wearing girl in a blue background. 0:00:12-0:00:14: The girl stands on a rock with a golden snake. 0:00:15-0:00:18: Another Egyptian character climbs up to reach the girl. 0:00:18-0:00:20: An arguing Egyptian man and woman in the desert.

[Figure 263]_images/imageFile263.png>)

Scene Detection:

0:00:00, 0:00:01, 0:00:02, 0:00:03, 0:00:04, 0:00:05, 0:00:06, 0:00:07, 0:00:08, 0:00:09, ..., 0:02:39, 0:02:40, 0:02:41, 0:02:42

Metadata Collection: Title: Mummies Abstract: Thut is a young charioteer. Nefer is the daughter of Faraon. In this video, Thut saves himself from falling off a cliff. Thut and Nefer meets Faraon. Thut and Nefer gets married in front of the goddess statue and all people. Thut and Nefer celebrates by singing and dancing.

- 0:00:20-0:00:23: The man and woman part ways, looking sad.

- 0:00:23-0:00:24: A new Egyptian man appears with a young boy.
- 0:00:24-0:00:26: The woman from Scene 1 hugs the new man.

- 0:00:26-0:00:27: The man from Scene 1 looks sad, alone. : :
- 0:01:47-0:01:48: Happy, loving couple's faces.

0:01:57-0:02:01: Egyptian musical characters on decorated stage. 0:02:01-0:02:03: Female musician smiles at male musician.

- 0:02:03-0:02:05: Female talks to male musician in front of palace.

Visual Prompting:

[Figure 264]_images/imageFile264.png>)

[Figure 265]_images/imageFile265.png>)

[Figure 266]_images/imageFile266.png>)

[Figure 267]_images/imageFile267.png>)

- 0:02:05-0:02:06: Woman leans to kiss man on palace balcony.
- 0:02:06-0:02:07: Dancing couple with smiling palace guard. 0:02:15-0:02:17: Characters perform in ancient Egyptian musical celebration. 0:02:17-0:02:20: Pharaoh is amused by queen's antics. 0:02:24-0:02:26: Egyptian characters on a colorful decorated stage. 0:02:33-0:02:37: Characters play instruments as others dance. 0:02:38-0:02:40: Cleopatra sings on a stage with a microphone. 0:02:42-0:02:43: Egyptian musical couple on palace balcony.

[Figure 268]_images/imageFile268.png>)

Faraon Thut Nefer

Script Generation:

[Figure 269]_images/imageFile269.png>)

0:00:00 - 0:00:06: Nefer, an Egyptian princess, is seen gazing at the desert landscape and examining a rocky cliff.

- Figure 17. Case studies of MM-VID ’s capabilities and application scenarios: character identification; By providing the face photos of the characters and their names, MM-VID can generate the script with the identified characters. Blue highlights the correct character identification. The original video is available at https://www.tokyvideo.com/video/mummies-online-free-film, from 1:19:47 to 1:22:30.

[Figure 270]_images/imageFile270.png>)

###### Regenerated ASR with Identified Speakers:

[Figure 271]_images/imageFile271.png>)

[Figure 272]_images/imageFile272.png>)

- 0:00:01 Ballister Boldheart: Good trade flesh and we run and we hide.

- 0:00:33 Unknown: That was weird.
- 0:00:34 Nimona: Hey boss.

- 0:00:39 Nimona: I love the secret layer.

- 0:00:42 Nimona: The garbage and the smell of sadness really pull the whole thing together.

- 0:00:45 Ballister Boldheart: Well, wait, wait, wait. Who? Who are you?

- 0:00:47 Nimona: The names. Pneumonia. Whoa. Yeah. Sick arm.
- 0:00:48 Ballister Boldheart: And how did you?

- 0:00:51 Unknown: Did it bleed a lot? Did they let you keep the old one?

- 0:00:54 Ballister Boldheart: No, let's go. That's wrong with you.

- 0:00:58 Nimona: Can I have it? What? Put that?
- 0:00:59 Ballister Boldheart: Down that is.

- 0:01:00 Ballister Boldheart: Not for little girls, little.

- 0:01:02 Nimona: Girls. OK, how old do you think I?

- 0:01:05 Ballister Boldheart: AM 10.

- 0:01:09 Unknown: Alright, home.

- 0:01:09 Ballister Boldheart: We are more or less than 10.

- 0:01:10 Nimona: Not a lot of kids in your life, huh?

[Figure 273]_images/imageFile273.png>)

[Figure 274]_images/imageFile274.png>)

Loading video… Video length is 2 minutes 57 seconds.

[Figure 275]_images/imageFile275.png>)

###### Invoke tools: ASR, Metadata Collection

ASR: 0:00:01 Guest_0: Good trade flesh and we run and we hide.

- 0:00:33 Guest_2: That was weird.
- 0:00:34 Guest_3: Hey boss. 0:00:39 Guest_3: I love the secret layer. … 0:02:53 Guest_10: From you. No. Fine then from. 0:02:55 Guest_10: Who? The guy who cut off your arms.

Metadata Collection: Title: Nimona Abstract: Ballister boldheart is a knight. Nimona is a mischievous teen who is a shapeshifting creature. The video shows how Ballister met Nimona first time.Nimona thought boldheart was the villain but disappointed to know he is not.

- 0:01:12 Ballister Boldheart: You know what?
- 0:01:13 Ballister Boldheart: No. And then like if you stay that?

- 0:01:14 Ballister Boldheart: Way you have to go job.
- 0:01:15 Nimona: But I'm here about this.
- 0:01:16 Unknown: What job? 0:01:18 Nimona: Oh, it's all here. 0:01:18 Nimona: It's all here in my application. 0:01:22 Nimona: This is just a bunch of drawings. 0:01:26 Unknown: Very disturbing drawings. 0:01:28 Ballister Boldheart: Oh, look, it's me on a rhinoceros skewering. 0:01:32 Ballister Boldheart: Guards like a human kebab.

Visual Prompting:

[Figure 276]_images/imageFile276.png>)

[Figure 277]_images/imageFile277.png>)

[Figure 278]_images/imageFile278.png>)

- 0:01:34 Nimona: Yeah. Do you like?
- 0:01:35 Nimona: It I thought a visual aid. 0:01:37 Unknown: Really embarrass me. Puff up here. So about the job. What job? 0:01:45 Nimona: To be your.

Nimona Ballister Boldheart

- Figure 18. Case studies of MM-VID ’s capabilities and application scenarios: speaker identification; By leveraging visual prompting, MM-VID can enhance ASR predictions with the speakers’ identity. Blue and Red highlight the correct and incorrect predictions, respectively. Figure 19 shows continued outputs. The original video is available at https://www.netflix.com/title/81444554, from 9:52 to 12:52.

n, which is located in the third row and third column of the screen. I would then click on the icon to open the app and search

(continued):

[Figure 279]_images/imageFile279.png>)

- 0:01:45 Nimona: Sidekick, you know.
- 0:01:46 Nimona: To help you do whatever it takes to.
- 0:01:47 Nimona: Get revenge on.
- 0:01:48 Nimona: The cold, cruel world that rejected you. Shall we pillage village? Lay low until they don't remember you, and then we rise like a phoenix. From the Ashes, overthrow the government.

- 0:01:58 Unknown: Or we could.
- 0:01:59 Nimona: Just talk.

- 0:02:02 Unknown: The the the point.
- 0:02:03 Nimona: Is whatever your dark heart desires. Boss, your sidekick has arrived.

- 0:02:08 Ballister Boldheart: I don't need a.
- 0:02:09 Ballister Boldheart: Side wait, I am not a. 0:02:09 Nimona: Every villain needs a sidekick. 0:02:12 Unknown: Sweetener. Murdo. That's how you want to kill first. Oh.

- 0:02:14 Nimona: Are these your next targets?

- 0:02:15 Nimona: Who do?
- 0:02:16 Ballister Boldheart: I'm not killing anybody.

- 0:02:18 Nimona: This one, this guy looks extremely punchable.
- 0:02:19 Nimona: Right. He is actually extremely.

- 0:02:21 Ballister Boldheart: You're right, he is actually.
- 0:02:22 Ballister Boldheart: Punchable. Wait, that is not a murder wall. It's a.

- 0:02:26 Ballister Boldheart: The innocence wall, yeah.
- 0:02:27 Nimona: Citizens will. 0:02:30 Nimona: We are you saying that you're not a villain? Yes.

- 0:02:36 Nimona: Yes you are.
- 0:02:37 Ballister Boldheart: No, no, no. I'm not a villain, no.

- 0:02:42 Ballister Boldheart: But you.
- 0:02:43 Ballister Boldheart: Disappointed that I'm not a murderer. 0:02:46 Nimona: I mean. 0:02:46 Nimona: I mean a little. Why?

- 0:02:48 Ballister Boldheart: You know what we.
- 0:02:49 Ballister Boldheart: Don't know.
- 0:02:50 Ballister Boldheart: The real villain is still out there and I do need help not. 0:02:53 Nimona: From you. No. Fine then from. 0:02:55 Nimona: Who? The guy who cut off your arms.

- Figure 19. Case studies of MM-VID ’s capabilities and application scenarios: speaker identification; By leveraging visual prompting, MM-VID can enhance ASR predictions with the speakers’ identity. Blue and Red highlight the correct and incorrect predictions, respectively. The original video is available at https://www.netflix.com/title/81444554, from 9:52 to 12:52.

[Figure 280]_images/imageFile280.png>)

[Figure 281]_images/imageFile281.png>)

(continued) 0:01:03 - 0:01:09 : Mr. Bean tries to regain focus in this scene, tapping his fingers as he nervously attempts to recall information for the exam.

[Figure 282]_images/imageFile282.png>)

[Figure 283]_images/imageFile283.png>)

- 0:01:11 - 0:01:12 : Two men, including Mr. Bean, are discussing a portion of the exam in a library, pointing at a book while brainstorming answers.
- 0:01:12 - 0:01:21 : Mr. Bean and the man next to him become more desperate, whispering to each other and trying to cheat as their frustration grows. The mood is tense, mirroring their anxiety.

[Figure 284]_images/imageFile284.png>)

###### Loading video… Video length is 1 minutes 44 seconds.

###### Script Generation:

0:00:00 - 0:00:08 : Mr. Bean enters a classroom and takes a seat next to another man. They both anxiously look at their exam papers as they begin their test.

0:00:09 - 0:00:16 : Mr. Bean glances over at the other man's paper and tries to sneakily copy his answers. Both men continue to work on their tests, unaware of their surroundings.

[Figure 285]_images/imageFile285.png>)

Why Mr. Bean cheat in the exam?

0:00:17 - 0:00:23 : Close-up shot of Mr. Bean's face showing his frustration as he struggles to figure out the answers to the exam.

[Figure 286]_images/imageFile286.png>)

Mr. Bean cheats in the exam because he is unprepared, lacks confidence in his ability to answer the questions, and feels desperate to pass the test. His comedic nature and inability to cope with the pressure of the exam lead him to attempt cheating as a last resort.

0:00:24 - 0:00:27 : Mr. Bean continues to take the exam while holding a pink flamingo pen, struggling to concentrate.

0:00:27 - 0:00:35 : Mr. Bean becomes more desperate, sticking his nose into the exam paper as he continues to copy answers from the man sitting next to him.

[Figure 287]_images/imageFile287.png>)

Why do people find the video funny?

0:00:36 - 0:00:45 : Mr. Bean takes a break, rubbing his eyes as the scene cuts back and forth between him and his colleague.

[Figure 288]_images/imageFile288.png>)

People might find the video funny because of Mr. Bean's exaggerated expressions and actions, such as sticking his nose into the exam paper and using a pink flamingo pen,

0:00:45 - 0:00:54 : The students continue to work on their task as the scene shows the peaceful and quiet environment of the exam room.

which add humor to the otherwise tense and stressful situation of taking an exam.

- 0:00:54 - 0:01:00 : Mr. Bean is deep in thought as he taps his pen on the desk, looking more worried and nervous about his exam performance.
- 0:01:01 - 0:01:02 : Close-up of Mr. Bean struggling with a problem and looking upward with a frustrated expression.

- Figure 20. Case studies of MM-VID ’s capabilities and application scenarios: audio description generation; Without ASR inputs, MMVID processes the video and generates a detailed script. Blue and Red highlight the correct and incorrect predictions, respectively. The original video is available at https://www.youtube.com/watch?v=29ZmXkICT1g, from 2:50 to 4:34.

Show me a moment when Mr. Bean cheat in the video

A moment when Mr. Bean cheats in the video is at 0:00:09 - 0:00:16, where he glances over at the other man's paper and tries to sneakily copy his answers.

[Figure 292]_images/imageFile292.png>)

[Figure 293]_images/imageFile293.png>)

[Figure 294]_images/imageFile294.png>)

[Figure 295]_images/imageFile295.png>)

……

[Figure 296]_images/imageFile296.png>)

###### …

[Figure 297]_images/imageFile297.png>)

[Figure 298]_images/imageFile298.png>)

## bapl biotects the

## biotects the

rohammad examining a small

rohammad examining a small

- Figure 21. Case studies of MM-VID ’s capabilities and application scenarios: self-refinement; In this example, MM-VID mistakenly identifies a bird as a rock due to low-resolution video frames. The self-refinement helps rectify the errors, resulting in a more refined output. Blue and Red highlight the correct and incorrect predictions, respectively. The original video is available at https://www. youtube.com/watch?v=o4UKvgBkO2Y, from 1:06 to 4:17.

[Figure 299]_images/imageFile299.png>)

[Figure 300]_images/imageFile300.png>)

[Figure 301]_images/imageFile301.png>)

[Figure 302]_images/imageFile302.png>)

[Figure 303]_images/imageFile303.png>)

[Figure 304]_images/imageFile304.png>)

…

[Figure 305]_images/imageFile305.png>)

[Figure 306]_images/imageFile306.png>)

[Figure 307]_images/imageFile307.png>)

[Figure 308]_images/imageFile308.png>)

…

[Figure 309]_images/imageFile309.png>)

[Figure 310]_images/imageFile310.png>)

[Figure 311]_images/imageFile311.png>)

[Figure 312]_images/imageFile312.png>)

[Figure 313]_images/imageFile313.png>)

…

[Figure 314]_images/imageFile314.png>)

- Figure 22. Case studies of MM-VID ’s capabilities and application scenarios: fast-changing short videos. The original videos are available at https://www.instagram.com/mealtimewithmummy/reels/

When do you see it in the video?

You can see it in the video at 0:01:03-0:01:05.

[Figure 320]_images/imageFile320.png>)

[Figure 321]_images/imageFile321.png>)

[Figure 322]_images/imageFile322.png>)

[Figure 324]_images/imageFile324.png>)

[Figure 325]_images/imageFile325.png>)

###### Loading video… Video length is 1 minutes 48 seconds.

[Figure 327]_images/imageFile327.png>)

###### …

Where can I find my pink jacket?

[Figure 329]_images/imageFile329.png>)

Show me the trajectory of my activities from 0:01:00 to 0:01:25 in this video.

[Figure 331]_images/imageFile331.png>)

Your pink jacket can be found on the chair covered with clothes.

[Figure 333]_images/imageFile333.png>)

From 0:01:00 to 0:01:25, the trajectory of your activities in the video is as follows:

[Figure 335]_images/imageFile335.png>)

- - Watering a plant next to the couch (0:01:10-0:01:11)
- - Managing plants in the living room (0:01:12-0:01:19)
- - Walking towards and opening a door (0:01:24-0:01:25)

When do you see it in the video?

[Figure 338]_images/imageFile338.png>)

[Figure 340]_images/imageFile340.png>)

You can see it in the video at the time range 0:00:05-0:00:06.

[Figure 342]_images/imageFile342.png>)

.

[Figure 344]_images/imageFile344.png>)

[Figure 346]_images/imageFile346.png>)

[Figure 347]_images/imageFile347.png>)

Where can I find my laptop?

[Figure 350]_images/imageFile350.png>)

[Figure 351]_images/imageFile351.png>)

Your laptop can be found near the sofa.

[Figure 353]_images/imageFile353.png>)

[Figure 354]_images/imageFile354.png>)

Figure 23. Case studies of MM-VID ’s capabilities and application scenarios: embodied agent. Blue highlights the correct prediction. The original video is collected from Ego4D dataset [25].

[Figure 355]_images/imageFile355.png>)

[Figure 357]_images/imageFile357.png>)

[Figure 360]_images/imageFile360.png>)

[Figure 361]_images/imageFile361.png>)

[Figure 363]_images/imageFile363.png>)

[Figure 365]_images/imageFile365.png>)

[Figure 367]_images/imageFile367.png>)

[Figure 369]_images/imageFile369.png>)

[Figure 371]_images/imageFile371.png>)

[Figure 373]_images/imageFile373.png>)

[Figure 375]_images/imageFile375.png>)

[Figure 376]_images/imageFile376.png>)

[Figure 378]_images/imageFile378.png>)

[Figure 380]_images/imageFile380.png>)

[Figure 383]_images/imageFile383.png>)

[Figure 388]_images/imageFile388.png>)

[Figure 390]_images/imageFile390.png>)

[Figure 391]_images/imageFile391.png>)

[Figure 392]_images/imageFile392.png>)

[Figure 395]_images/imageFile395.png>)

[Figure 398]_images/imageFile398.png>)

[Figure 400]_images/imageFile400.png>)

[Figure 401]_images/imageFile401.png>)

[Figure 402]_images/imageFile402.png>)

[Figure 403]_images/imageFile403.png>)

[Figure 405]_images/imageFile405.png>)

[Figure 406]_images/imageFile406.png>)

[Figure 409]_images/imageFile409.png>)

[Figure 410]_images/imageFile410.png>)

You are expert in playing Super Mario Bros. What is the next controls to be pressed after the last frame. Output JSON.

[Figure 412]_images/imageFile412.png>)

[Figure 413]_images/imageFile413.png>)

[Figure 414]_images/imageFile414.png>)

8.33

Available controls are: Jump, ArrowLeft, ArrowRight, Wait

Output: {

"controls": "ArrowLeft+Jump", "reason": "Mario should"

[Figure 416]_images/imageFile416.png>)

[Figure 417]_images/imageFile417.png>)

You are expert in playing Super Mario Bros. What is the next controls to be pressed after the last frame. Output JSON.

}

For this sequence of frames of super mario, what is the next controls.

Available controls are: Jump, ArrowLeft, ArrowRight, Wait

Output:

Output: {

{

"controls": "ArrowLeft+Jump", "reason": "Mario should"

"control": "ArrowRight+Jump", "reason": "Mario should jump over the

[Figure 420]_images/imageFile420.png>)

}

You are expert in playing Super Mario Bros. What is the next controls to be pressed after the last frame. Output JSON.

goomba and move right to avoid the pit"

For this sequence of frames of super mario, what is the next controls.

}

Available controls are: Jump, ArrowLeft, ArrowRight, Wait

Output:

Output: {

{

"control": "ArrowRight+Jump", "reason": "Mario should jump over the

[Figure 423]_images/imageFile423.png>)

"controls": "ArrowLeft+Jump", "reason": "Mario should"

goomba and move right to continue the level"

You are expert in playing Super Mario Bros. What is the next controls to be pressed after the last frame. Output JSON.

}

}

For this sequence of frames of super mario, what is the next controls.

Available controls are: Jump, ArrowLeft, ArrowRight, Wait

Output:

Output: {

k flamingo pe

{

[Figure 426]_images/imageFile426.png>)

"control": "Jump", "reason": "Mario should jump to

"controls": "ArrowLeft+Jump", "reason": "Mario should"

avoid the goomba"

}

} (Agent presses control “Jump”)

Figure 26. Case studies of MM-VID ’s capabilities and application scenarios: playing video game. Blue highlights the correct prediction. Figure 27 shows continued outputs. The video is generated by Pygame library [4].

[Figure 430]_images/imageFile430.png>)

[Figure 431]_images/imageFile431.png>)

[Figure 432]_images/imageFile432.png>)

[Figure 433]_images/imageFile433.png>)

[Figure 435]_images/imageFile435.png>)

[Figure 436]_images/imageFile436.png>)

[Figure 438]_images/imageFile438.png>)

[Figure 440]_images/imageFile440.png>)

[Figure 443]_images/imageFile443.png>)

[Figure 445]_images/imageFile445.png>)

###### Figure 27. Case studies of MM-VID ’s capabilities and application scenarios: playing video game. Blue highlights the correct prediction. The video is generated by Pygame library [4].

[Figure 447]_images/imageFile447.png>)

[Figure 448]_images/imageFile448.png>)

[Figure 449]_images/imageFile449.png>)

[Figure 450]_images/imageFile450.png>)

[Figure 451]_images/imageFile451.png>)

[Figure 452]_images/imageFile452.png>)

[Figure 453]_images/imageFile453.png>)

Imagine that you are a robot operating a phone. Like how humans operate the phone, you can move your finger to an icon, click an icon with your finger, scroll down a screen, or type some texts with the keyboard. You are asked to shop for an ergonomic keyboard, your budget is between $50 and $100. Previous action: "My next move would be to move my finger to the Amazon icon, which is located in the third row and third column of the screen. I would then click on the icon to open the app and search for an ergonomic keyboard within my budget of $50. " You have now opened the Amazon app. Below is what you see on the phone screen. What will you do next? If the action involves moving your finger to an icon, describe the location of that icon as detailed as possible.

Imagine that you are a robot operating a phone. Like how humans operate the phone, you can move your finger to an icon, click an icon with your finger, scroll down a screen, or type some texts with the keyboard. You are asked to shop for an ergonomic keyboard, your budget is between $50 and $100. Previous action: "My immediate next move would be to move my finger to the "Filters" button, which is located at the top right corner of the screen, next to the "Windows" button. I would then click on the "Filters" button to open the filter options and narrow down my search results to only show ergonomic keyboards within my budget range of $50 to $100. " You have now completed the previous action as described above. Below is what you see on the phone screen. Please predict your immediate next move. If the action involves moving your finger to an icon, describe the location of that icon as detailed as possible.

[Figure 454]_images/imageFile454.png>)

[Figure 455]_images/imageFile455.png>)

[Figure 456]_images/imageFile456.png>)

[Figure 457]_images/imageFile457.png>)

[Figure 458]_images/imageFile458.png>)

My next move would be to move my finger to the search bar, which is located

###### at the top of the screen. I would then click on the search bar and type in "ergonomic keyboard" using the keyboard. After

My immediate next move would be to move my finger to the "Price & Deals" filter option, which is located under the "Colors"

[Figure 459]_images/imageFile459.png>)

typing in the search term, I would click on

###### the "Search" button, which is located on the bottom right corner of the keyboard, to initiate the search.

filter option and above the "Sort by" filter option. I would then click on the "Price & Deals" filter option to open the price range

My next move would be to move my finger to the "Show 1,987 results" button, which is located at the bottom of the screen, below the "Clear Filters" button and above the navigation bar. I would then click on the "Show 1,987 results" button to view the search results for ergonomic keyboards within my budget range.

(continued) filter and set my minimum price to $50 and my maximum price to $100. This would further narrow down my search results to only show ergonomic keyboards within my budget range.

[Figure 460]_images/imageFile460.png>)

[Figure 461]_images/imageFile461.png>)

Imagine that you are a robot operating a phone. Like how humans operate the phone, you can move your finger to an icon, click an icon with your finger, scroll down a screen, or type some texts with the keyboard. You are asked to shop for an ergonomic keyboard, your budget is between $50 and $100. Previous action: " My immediate next move would be to move my finger to the "Price & Deals" filter option, which is located under the "Colors" filter option and above the "Sort by" filter option. I would then click on the "Price & Deals" filter option to open the price range filter and set my minimum price to $50 and my maximum price to $100. This would further narrow down my search results to only show ergonomic keyboards within my budget range. " You have now completed the previous action as described above. Below is what you see on the phone screen. What will you do next? If the action involves moving your finger to an icon, describe the location of that icon as detailed as possible.

Imagine that you are a robot operating a phone. Like how humans operate the phone, you can move your finger to an icon, click an icon with your finger, scroll down a screen, or type some texts with the keyboard. You are asked to shop for an ergonomic keyboard, your budget is between $50 and $100. Previous action: " My next move would be to move my finger to the "Show 1,987 results" button, which is located at the bottom of the screen, below the "Clear Filters" button and above the navigation bar. I would then click on the "Show 1,987 results" button to view the search results for ergonomic keyboards within my budget range. " You have now completed the previous action as described above. Below is what you see on the phone screen. What will you do next? If the action involves moving your finger to an icon, describe the location of that icon as detailed as possible.

[Figure 462]_images/imageFile462.png>)

[Figure 463]_images/imageFile463.png>)

[Figure 464]_images/imageFile464.png>)

[Figure 465]_images/imageFile465.png>)

[Figure 467]_images/imageFile467.png>)

[Figure 468]_images/imageFile468.png>)

[Figure 469]_images/imageFile469.png>)

[Figure 470]_images/imageFile470.png>)

[Figure 471]_images/imageFile471.png>)

[Figure 473]_images/imageFile473.png>)

###### Figure 31. Case studies of MM-VID ’s capabilities and application scenarios: GUI navigation. Blue and Red highlight the correct and incorrect predictions, respectively. Figure 32 shows continued outputs.

[Figure 474]_images/imageFile474.png>)

[Figure 475]_images/imageFile475.png>)

[Figure 476]_images/imageFile476.png>)

[Figure 477]_images/imageFile477.png>)

[Figure 478]_images/imageFile478.png>)

[Figure 479]_images/imageFile479.png>)

###### Figure 32. Case studies of MM-VID ’s capabilities and application scenarios: GUI navigation. Blue and Red highlight the correct and incorrect predictions, respectively.

