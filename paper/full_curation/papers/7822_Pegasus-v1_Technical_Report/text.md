# arXiv:2404.14687v1[cs.MM]23Apr2024

## Pegasus-1 Technical Report

##### Abstract

This technical report introduces Pegasus-1, a multimodal language model specialized in video content understanding and interaction through natural language.1 Pegasus-1 is designed to address the unique challenges posed by video data, such as interpreting spatiotemporal information, to offer nuanced video content comprehension across various lengths. This technical report overviews Pegasus-1’s architecture, training strategies, and its performance in benchmarks on video conversation, zero-shot video question answering, and video summarization. We also explore qualitative characteristics of Pegasus-1 , demonstrating its capabilities as well as its limitations, in order to provide readers a balanced view of its current state and its future direction.2

##### 1 Introduction

In the evolving landscape of large language models (LLMs) [1, 18, 2, 4, 3], cultivating the video understanding capabilities of LLMs [8, 21, 22, 7, 12, 10, 9, 15, 16] has emerged as a frontier of innovation and practical utility. This technical report introduces and analyzes Pegasus-1, a state-ofthe-art multimodal model that offers versatile capabilities in interpreting, generating, and interacting with video content through natural language.

The major goal of developing Pegasus-1 is to overcome inherent challenges in video data that contain multiple modalities within a single format. Critical to this understanding is to interpret the temporal sequence of visual data, capturing the essence of movement and change over time while providing a spatially detailed analysis within each frame of the video. Concurrently, audio information should be incorporated to enhance the interpretation of visual elements and ensure a nuanced understanding of video content. Moreover, handling a wide range of video lengths is essential to cover various video types from short clips to extended footage.

In this technical report, we discuss our approach to dealing with these challenges to enable Pegasus-1 to understand video content comprehensively. The discussion will include a brief description of its model architecture, training data, and training strategies, providing a broad perspective on the factors contributing to Pegasus-1’s advanced video understanding capabilities.

Pegasus-1 achieves new state-of-the-art results in video conversation benchmark [7], zero-shot video question answering [19, 20], and video summarization [5, 17]. By outperforming both open-source and proprietary models, Pegasus-1 demonstrates its generalization capabilities in understanding complex video data.

This report also presents a wide range of qualitative results to offer insights into Pegasus-1’s capabilities and to explore potential emerging use cases. Our objective is to unveil and preview the range of functionalities Pegasus-1 can potentially deliver, acknowledging that while these capabilities

- 1Please cite this paper as (Twelve Labs, 2024). Please see Authorship Section at the end of this report for the

full list of contributors.

- 2Pegasus-1 blog post and API are available at https://www.twelvelabs.io/blog/

upgrading-pegasus-1

[Figure 1]

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

[Figure 12]

[Figure 13]

[Figure 14]

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

Pegasus-1 - 17B

###### Vdeo-language alignment model

Video-language aligned embedding

###### Video input

Text output

###### Large language model

###### Video encoder model

Video embeddings

Map video embedding to language embedding

+

“In this video, we witnes a series of intense events unfold on the streets. It starts with a police oficer driving.”

Video Frame

###### Generate text

Generate rich embedding

Video embedding

Language embedding

Video ASR

User prompt

Marengo 2.6

- Figure 1: Architectural Overview of Pegasus-1: Pegasus-1 is comprised of three main components: 1) the Video Encoder model for generating multi-modal embeddings from visual and audio inputs, the Video-Language Alignment Model for synchronizing video and text representations, and the Large Language Model for generating contextually relevant textual output.

showcase new possibilities, they may not yet be fully reliable or consistent. Through this exploration, we aim to highlight Pegasus-1’s potential to catalyze new applications and advancements in the field.

Despite its capabilities, Pegasus-1 has inherent limitations that users should be aware of, as we aim for transparency while continuing to refine and enhance its functionalities. These limitations are briefly discussed within this technical report. Our objective is to provide users with a comprehensive understanding of Pegasus-1’s current strengths, weaknesses and areas for growth, ensuring informed usage and setting accurate expectations for its performance and applicability.

##### 2 Model Architecture and Training

###### 2.1 Model Architecture

To efficiently comprehend video data, it is crucial to seamlessly integrate both auditory and visual information while effectively managing longer video lengths. In our architectural strategy, we have developed an architecture capable of jointly processing audio and visual data, and it is distinguished by its efficient computation tailored for longer videos. As depicted in Fig. 1, Pegasus-1 is structured into a comprehensive tripartite framework designed to encode, align, and decode:

- • Video Encoder Model: Powered by our Marengo 2.6 [6], the video encoder model processes the video input to produce rich embeddings from both video frames and audio speech recognition (ASR) data. These embeddings are dense representations that capture the visual and auditory essence of the video content.
- • Video-language Alignment Model: The model maps video embeddings to corresponding language embeddings, establishing a shared space where video and text representations are coherently aligned. Such an alignment is pivotal, enabling the model to discern and correlate the visual content of the video with its descriptive language effectively. Additionally, to enhance the processing of longer video lengths, we employ token reduction techniques aimed at minimizing the input token sizes within the large language model, thereby optimizing computational efficiency and maintaining context integrity over longer durations.
- • Large Language Model (Decoder Model): The large language model decoder takes the aligned embeddings and user prompts to generate coherent and contextually relevant text output. This output can range from descriptive summaries to answers to specific questions about the video content. This large language model decoder leverages efficient attention mechanisms for efficiently handling the long context of video data.

###### 2.2 Training

The training of Pegasus-1, a state-of-the-art multimodal foundation model, focuses on the harmonious understanding of various modalities in video data, including visual and audio modes. To achieve this, we capitalized on the unique strengths of each modality: employing spatial understanding from image data and deriving insights from audio data for robust audio understanding. Notably, despite the video having a richer information source compared to standalone images or audio, there is a marked scarcity of high-quality video data. We address this challenge by amassing an extensive collection of proprietary text annotations for training. Specifically, we annotate 10M+ diverse videos with highly detailed descriptions. These descriptions capture most of the events that appear in each video, incorporating detailed real-world knowledge about activities and objects, including specific names of objects and technical terms for various sports.3

Our training process for Pegasus-1 consists of two phases: pretraining phase and instruction tuning phase. During the pretraining phase, we initialize both the video encoder and large language model with pre-trained weights, subsequently training with our expansive multi-modal datasets, which include the rich detailed video data mentioned previously. In the second phase, we conduct supervised instruction finetuning, utilizing proprietary multimodal instruction datasets to refine the model’s responsiveness to user instructions. However, a significant challenge in such multi-stage training is the risk of catastrophic forgetting [13], where the pretrained model loses previously acquired knowledge upon assimilating new information.

To mitigate this, our training strategy incorporates two meticulously planned stages, aiming to harmonize the integration of new data while preserving existing knowledge. The cornerstone of this strategy is the selective unfreezing of model parameters combined with precise adjustments to learning rates during the training progression. This approach ensures that Pegasus-1 not only acquires new capabilities efficiently but also retains and refines its previously learned skills, establishing a robust foundation for advanced video-language understanding.

##### 3 Results on Benchmarks

We evaluate Pegasus-1’s performance across well-established video large language model benchmarks, including video conversation [12], zero-shot video question answering [20, 19], and video summarization [5]. This section presents the performance of Pegasus-1 in comparison to both proprietary and open-source models.

- 3.1 Comparison Models We compare our model against the following baselines:

- • Gemini models (proprietary) [16, 15]: Gemini is Google’s proprietary multimodal model, also known for its state-of-the-art performance in video-language tasks. To ensure the accuracy and relevance of our comparative analysis, we excerpt the performance of Gemini 1.0 and Gemini 1.5 from Google’s technical report for the Video Question Answering benchmark. For other benchmarks that are not included in their reports, such as video conversation and summarization, we run the February 17 2024 version of the publicly available Gemini API.4 Note that Gemini 1.5 Pro API is not available as of March 30th, 2024 and is only accessible through its playground-like preview access.
- • Open-source models: We compare Pegasus-1 with VideoChat [7], Video-ChatGPT [12], Video LLAMA [21], BT-adapter [10], LLaMA-VID [9], and VideoChat2 [8]. We use the benchmark results reported in their papers if they are available.

- 3.2 Results

- 3Note that the number of tokens per video depends on its length and spatiotemporal resolution. For instance, if one assumes 1 frame per second, 256 patches per frame, and 3-minute (180s) per video for simplicity, each video would be equivalent to around 46k tokens and 10M videos would be equivalent to 0.46T tokens.
- 4Gemini API documentation: https://github.com/GoogleCloudPlatform/generative-ai/blob/ main/gemini/getting-started/intro_gemini_pro_vision_python.ipynb

###### Model Correctness Detail Context Temporal Consistency

VideoChat [7] 2.23 2.5 2.53 1.94 2.24 Video-ChatGPT [12] 2.4 2.52 2.62 1.98 2.37 Video LLAMA [21] 1.96 2.18 2.16 1.82 1.79 BT-adapter [10] 2.68 2.69 3.27 2.34 2.46 LLaMA-VID [9] 3.07 3.05 3.6 2.58 2.63 VideoChat2 [8] 3.02 2.88 3.51 2.66 2.81

Gemini Pro [16] 2.98 2.99 3.44 2.32 2.32 Pegasus-1 3.79 3.76 4.29 3.34 4.03

- Table 1: In the comparison conducted using the video conversation benchmark [12], the open-ended generation outputs are evaluated by a language model (LLM) through a comparison with the ground truth answers. This evaluation specifically focuses on five distinct aspects. Notably, Pegasus-1 outperforms both open-source models and a leading proprietary model. These results were compiled in February 2024. At that time, the Gemini 1.5 API was not yet generally available, hence its exclusion from the comparative table.

|Random mPLUG-Owl V-LLaVA VideoChat2<br><br>|Action Fine Coarse| |Direction Object Camera<br><br>| |Speed Absolute Relative| |Event Order Order|Attribute Change Color Size Combined Other<br><br>| | | |Avg.|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |39.7|40.1|39.8|39.0|40.8|42.0|41.5|40.4|39.9|38.9|39.4|40.5 44.4 49.6 50.8<br><br>|
| |48.8 66.1 55.0 83.1 60.4 77.2| |38.7 36.8 41.9 42.1 45.0 40.8<br><br>| |42.2 38.4 44.7 41.6 52.6 41.5<br><br>| |42.0 46.2 44.8<br><br>|41.7 44.7 41.9 39.9 45.7 45.7 43.5 49.8 50.8 47.9 49.9 44.3| | | | |
|Pegasus-1|79.8 92.6<br><br>| |46.1 44.4| |48.1 44.2<br><br>| |56.3<br><br>|51.7 49.6 46.7 45.7<br><br>| | | |57.1|

- Table 2: TempCompass [11] assesses the temporal understanding capabilities of video language models across five dimensions. Pegasus-1 clearly outperforms competing open-source models.

Video conversation benchmark. In Table 1, we present the evaluation results of Pegasus-1 on a video-based conversation benchmark [12]. The results indicate that Pegasus-1 performs robustly in the context of video conversations, achieving notable performance in various evaluative dimensions. Specifically, Pegasus-1 attains a score of 3.79 in Correctness and 4.29 in Context, which demonstrates its effective processing and understanding of video conversation content and its relevant contexts. These scores are indicative of Pegasus-1’s performance in critical areas such as Correctness, Detail, Contextual Awareness, Temporal Comprehension, and Consistency, emphasizing its capability to interpret and engage with video-based dialogue.

Zero-shot video question answering. Additionally, we detail Pegasus-1’s performance on two popular zero-shot video question-answering benchmarks—ActivityNet-QA [20] and NExT-QA [19]—in Table 3. In video question-answering tasks, Pegasus-1 showcases significant enhancement in zero-shot capabilities when evaluated on the ActivityNet-QA and NExT-QA datasets compared to open-source models and Gemini series. The generated responses are provided to GPT-3.5 Turbo, which is utilized to determine if the predictions align with the ground truth answers..

Video summarization. To evaluate the performance of video summarization, we follow the video conversation benchmarks [12] and evaluate the generated summary in three areas: Correctness of Information, Detailed Orientation, and Contextual Understanding. GPT-4 measures the score for each metric by setting the reference summary as ground truth. Here, we use the ActivityNet detailed caption dataset proposed in [12], and Table 4 shows these results. As shown above, Pegasus-1 outperforms the baseline models in all metrics by a significant margin.

Temporal understanding. As a video language model, Pegasus-1 is expected to have the capability to grasp temporal information. This capability is put to the test with TempCompass [11], a dataset designed specifically for evaluating temporal understanding. TempCompass focuses on assessing five aspects of temporal information: action, direction, speed, event order, and attribute change. To evaluate temporal understanding, the benchmark includes synthetic videos created from a single

###### Model ActivityNet-QA NExT-QA

Test Split (%) Test Split (%) Video-ChatGPT [12] 35.2 VideoChat2 [8] 49.1 61.7 Gemini 1.0 Pro [16] 49.8 28.0 Gemini 1.0 Ultra [16] 52.2 29.9 Gemini 1.5 Pro [15] 56.7 Pegasus-1 59.9 71.1

- Table 3: Zero-shot video QA results on ActivityNet-QA [20] and NExT-QA dataset [19]. Note that Gemini 1.5 Pro result is not available for NExT-QA as it is not reported in its technical report.

Model Correctness Detail Context Average

Video-ChatGPT [12] 1.19 1.33 1.42 1.31 VideoChat2 [8] 1.78 1.52 1.98 1.76

Gemini 1.0 Pro [16] 1.65 1.69 1.94 1.76 Pegasus-1 2.30 2.58 2.75 2.54

- Table 4: Utilizing the video conversation benchmark [12], which assesses generation output across five dimensions, our evaluation focuses on three specific aspects relevant to the video summarization task. Within this framework, Pegasus-1 demonstrates superior performance compared to other baseline models. ActivityNet with detailed caption proposed in [12] was utilized as a source dataset.

original video. These manipulations involve reversing the footage, playing it forward at varying speeds, or slowing it down, thus creating conflicting scenarios where the visual content remains constant while the temporal dynamics are altered. According to the findings presented in Table 2, Pegasus-1 outperforms other open-source benchmarks, notably surpassing VideoChat2 [8], which is also designed with a focus on temporal information. These results are detailed in TempCompass [11]. Given that TempCompass is formatted as a multiple-choice QA dataset requiring parsable responses, an intermediate output from Pegasus-1 is employed. Although this intermediate output conveys the same information as the final response, it is more readily adjusted to align with the dataset’s parsing requirements.

##### 4 Capabilities of Pegasus-1

Here, we present qualitative results to offer insight into Pegasus-1’s capabilities to explore potential emerging use cases. Also, to show the performance gap between Pegasus-1 and other state-of-the-art models, we provide a comparative analysis with qualitative results from Gemini 1.0 Pro (as of March 15th, 2024, using its publicly available API) and Gemini 1.5 Pro (as of March 30th, 2024 using its playground interface). In our analysis, we exclusively include Gemini 1.0 Pro and Gemini 1.5 Pro as our comparison baseline. Our selection criteria are twofold: the baseline must support native video input, and it must be widely acknowledged for its robust performance across an extensive array of benchmarks.

###### 4.1 Comparison Protocol for Qualitative Results

The presented prompts are carefully crafted, ensuring clarity and coherence to accurately assess the model’s performance. While crafting the prompts, we are meticulous in avoiding excessive obscurity or unnecessary complexity, aiming to ensure that the prompts remain straightforward and facilitate an unambiguous evaluation of the model’s capabilities.

Additionally, we focus on comparing the abilities in visually accurate responses, a capability not achievable through language model runoff. Language model runoff refers to the tendency of language

models to rely on their strong linguistic priors for generating contextually relevant information. For example, once a concept, such as "beach" is recognized by a Large Language Model (LLM), it can utilize its inherent knowledge to produce responses related to that concept, like "sunny weather" or "people swimming," without direct evidence from pixel information [14]. This process, while generating coherent responses, may not accurately reflect the visual content.

###### 4.2 Characteristics of Pegasus-1

Verbose output Pegasus-1 is designed to follow instructions closely, providing extensive details about a video’s audio and visual elements. As a result, the outputs from Pegasus-1 can be verbose, offering an abundance of information in response to queries. However, these features are deliberate for two key reasons.

First, Pegasus-1 is not equipped with chat interface capabilities; therefore, it is designed to provide comprehensive responses in a single interaction. This approach aligns well with video understanding tasks, given that videos are rich in information. Second, our research indicates that users tend to prefer detailed explanations over shorter ones.

Despite these benefits, reinforcing Pegasus-1 to generate elongated and detailed responses may lead to hallucinations. To mitigate this, we plan to implement a feature allowing users to adjust the level of detail in Pegasus-1’s responses in the future.

Categories of capabilities Our observations of Pegasus-1 reveal various capabilities in video understanding. These capabilities encompass a wide range of functionalities, demonstrating the model’s proficiency across different dimensions of video analysis, including, but not limited to:

- • Real-world knowledge: Pegasus-1 integrates extensive real-world knowledge, enabling it to contextualize and interpret video content with remarkable accuracy and depth.
- • Video-based reasoning: The model exhibits sophisticated reasoning abilities, drawing inferences from the video data to construct coherent understandings and insights.
- • 3D spatial understanding: Pegasus-1’s capability to comprehend 3D spatial relationships within video frames allows it to interpret complex scenes and object interactions, enhancing its analysis of video content that requires an understanding of depth and space.
- • Temporal reasoning: Pegasus-1 maintains an awareness of the chronological sequence of events in a video, a critical feature for understanding narratives.
- • Visual referring prompts: Building upon traditional text-based prompting methods, Pegasus-1 adopts visual referring prompts, enabling modifications directly within the pixel space. This advanced approach allows users to use arrows, boxes, or other visual markers to pinpoint specific regions, directing the model’s focus to these emphasized areas.

###### 4.3 Real World Knowledge

In this section, we assess the real-world knowledge capabilities of Pegasus-1, a video language model proficient in recognizing and identifying real-world entities within video content. We evaluated Pegasus-1 by challenging it to name specific real-world objects shown across different video scenarios.

The qualitative results, illustrated in Fig. 2 to 5, substantiate Pegasus-1’s capacity for such identification. In Fig. 2, Pegasus-1 demonstrates its precision by correctly identifying the Tuileries Garden from merely its visual cues, distinguishing specific landscape features like gravel pathways and manicured hedges, which are indicative of its advanced visual recognition capabilities. Conversely, Fig. 3 depicts Pegasus-1’s ability to discern the city of Kyoto from a compilation of distinct sceneries, showcasing its adeptness at integrating multiple visual inputs to form a coherent identification. In this instance, Pegasus-1 not only recognizes individual elements but also understands their collective representation of Kyoto, evidencing a sophisticated level of contextual comprehension.

Further reinforcing its object recognition proficiency, Pegasus-1 accurately identifies a Bugatti Chiron in Fig. 4, contrasting with a Gemini Pro 1.0’s less precise identification. Moreover, as seen in Fig. 5, Pegasus-1 accurately discerns the video game title "The Legend of Zelda: Breath of the Wild," attesting to its robust capability in accurately categorizing even within specific entertainment domains.

These instances collectively affirm Pegasus-1’s adeptness at extracting and interpreting nuanced visual details, affirming its advanced real-world knowledge capabilities and its potential application in varied contexts where accurate visual recognition is paramount.

###### 4.4 Video-based Reasoning

We investigate the visual reasoning capabilities of Pegasus-1, an endeavor that necessitates a synergy of visual interpretation and logical reasoning. To adeptly navigate visual reasoning tasks, the model must harness these competencies in tandem to formulate precise responses to inquiries. Such tasks encompass a broad spectrum of challenges, and to demonstrate the model’s proficiency, we showcase three distinct scenarios, each highlighting a different aspect of Pegasus-1’s visual reasoning ability.

The first scenario, depicted in Fig. 6 and 7, illustrates Pegasus-1’s ability to comprehend ongoing events within videos and anticipate future developments based on the present context. In Fig. 6, Pegasus-1 accurately predicts the immediate reactions required in a post-tidal surge scenario, contrasting with Gemini 1.0 Pro’s less precise anticipation of events. Similarly, Figure 7 demonstrates Pegasus-1’s adeptness at inferring the likely cause and subsequent unfolding of events encountered during a night drive, indicative of its nuanced understanding of context and sequence.

The second scenario, represented by Fig. 8, demands advanced reasoning, such as estimating property values in New York. Here, Pegasus-1 showcases its capacity for intricate reasoning by synthesizing visual cues into coherent, logical conclusions, outperforming Gemini 1.0 Pro’s more superficial analysis as highlighted in the respective figures.

The third scenario focuses on commonsense reasoning, where Pegasus-1 discerns the unusualness of a scenario involving a cat “driving” a car, as shown in Fig. 9. This case emphasizes Pegasus-1’s ability to differentiate between the ordinary and the extraordinary, leveraging visual details to arrive at plausible interpretations that align with real-world logic.

These examples collectively attest to Pegasus-1’s advanced capability in interpreting and reasoning from visual content.

###### 4.5 3D spatial Understanding

In assessing the 3D spatial understanding capabilities of Pegasus-1, we focus on its ability to interpret and navigate complex spatial relationships as depicted in video content. The task involves extracting and synthesizing spatial information from visual cues within a video to provide accurate navigational directions within a three-dimensional space.

Illustrated in Fig. 10, the video presents an exploration of various spaces within a residential setting, challenging the model to interpret and articulate the spatial layout effectively. The specific task, “Suppose I am in front of the porch. Tell me the way to get to the fridge”, requires Pegasus to deduce a path through the house’s structure, demonstrating an understanding that transcends mere object recognition to encompass spatial orientation and logical pathway deduction.

Pegasus-1’s response to this task is indicative of its advanced spatial reasoning. It successfully deciphers the sequence of spaces and their interconnections presented in the video: from the porch, through the entrance, up the stairs, and finally to the kitchen where the fridge is located. This performance highlights Pegasus-1’s ability to process, integrate, and articulate complex 3D spatial data, affirming its competency in navigating and explaining real-world spatial scenarios based on video content.

###### 4.6 Temporal Reasoning

In the realm of video-language modeling, grasping the sequential nature of audio-visual content is imperative. A model must discern and maintain the chronological order of events within a video to understand and interpret the content accurately. Pegasus-1 exemplifies this capability, as illustrated in Fig. 11 and 12, by effectively tracking and preserving the temporal sequence of events and demonstrating an acute awareness of their order.

In Fig. 11, the video details the meticulous process of preparing biryani, a dish requiring sequential steps. Pegasus-1 articulates each stage methodically: beginning with the preparation of chicken,

followed by the spices, then layering with onions and rice, and culminating in the combination of all ingredients. This delineation underscores Pegasus-1’s adeptness at temporal reasoning within a culinary context. Conversely, Gemini 1.0 Pro exhibits a discrepancy by misinterpreting the absence of narration, an element often expected in instructional cooking videos, demonstrating a gap in its temporal analysis.

Figure 12 presents a video montage of iconic London landmarks. Here, Pegasus-1 distinguishes itself by accurately capturing and relating the sequence of the presented landmarks, illustrating its proficiency in processing and understanding the temporal flow between distinct scenes. On the other hand, Gemini 1.0 Pro’s response includes inaccuracies and unwarranted assumptions, such as the misidentification of geographical features and the incorrect attribution of narration, highlighting its challenges with temporal and contextual accuracy in video content interpretation.

These instances affirm Pegasus-1’s temporal reasoning capabilities, showcasing its ability to interpret and convey the flow of events in video narratives accurately, a critical skill for comprehensive video understanding.

###### 4.7 Visual Referring Prompts

Pegasus-1 demonstrates a notable capacity to direct its attention to specific segments within a video, as indicated by visual markers. This proficiency is pivotal when the model is required to interpret and respond to cues that emphasize particular areas or actions within the video frame. Figure 14, 15, and 13 exemplify this ability, where discerning and focusing on highlighted details are essential for accurate analysis.

In Fig. 13, a soccer match clip singles out a player with a red circle, drawing attention to their role in the game. Pegasus-1 correctly focuses on this highlighted player, mentioning that the player is participating in offensive play to move the ball forward to the opposing teams goal. Conversely, Gemini 1.0 Pro inaccurately identifies the player’s team position, demonstrating a lapse in contextual understanding when analyzing the specified visual cue.

Figure 14 and 15 delve into a technical examination of Gerrit Cole’s pitching mechanics, with certain movements accentuated by white circles and green arrows. Pegasus-1 aptly acknowledges the importance of these marked regions, showing a clear comprehension of the content delineated by the white circles, in stark contrast to Gemini 1.0 Pro’s inability to interpret these annotations correctly.

These scenarios underline Pegasus-1’s adeptness in responding to visual referring prompts, showcasing its advanced capability to integrate and analyze specific visual information within broader video content.

###### 4.8 Other Specific Use Cases

In this section, we highlight some domain-specific use cases, such as medical video analysis and anomaly detection in dashcam videos that could help autonomous driving. Figure 16 and 17 feature two instances of anomaly detection captured by a dashcam. The former presents a more urgent and significant scenario (a car catching on fire), while the latter depicts a less critical situation (failing to signal when turning left). In both instances, Pegasus-1 accurately identifies the events unfolding in the videos. Figure 18 shows that Pegasus-1 effectively comprehends the complexities of a surgery video, which contains highly specialized and rare content.

- - Key Frames of Video
- - Pegasus-1

|Based on the information provided in the video, the exact location where the video was recorded is the Jardin des Tuileries, a public garden located between the Louvre Museum and the Place de la Concorde in Paris, France. This lovely garden, with its well-maintained lawns, manicured hedges, ornamental trees, and various statues, is a serene and beautiful space that invites visitors to take a leisurely stroll. The classical architecture of the nearby buildings, including the prominent facade of a grand building seen in the background, adds to the historical value of the area. The garden is filled with sunlight, casting shadows and highlighting the textures of the plants and the sculpted surfaces of the statues. Overall, the video offers a comprehensive tour of this iconic Parisian garden, allowing viewers to experience its tranquility and beauty.|
|---|

- - Gemini 1.0 Pro

|The video was recorded in the Tuileries Garden in Paris, France. The Tuileries Garden is a public park located in the 1st arrondissement of Paris, on the right bank of the Seine River. It is one of the largest and most popular parks in Paris, and is known for its beautiful gardens, sculptures, and fountains. The video was taken from the perspective of a person walking through the park. The video starts at the entrance to the park, and then continues through the park, past the sculptures and fountains, and finally to the exit. The video is a good way to get a sense of the size and beauty of the Tuileries Garden.|
|---|

- - Question

|Specify the exact location where the video was recorded.|
|---|

(source: https://www.youtube.com/watch?v=3LpuGYsOD9A)

Title: Beautiful Garden in front of Louvre Museum - Paris France

[Figure 56]

Real-World Knowledge: Place Detection

- - Gemini 1.5 Pro

|The video was recorded in the Tuileries Garden in Paris, France.|
|---|

- Figure 2: The video, captured in the Tuileries Garden, lacks any verbal narration, allowing Pegasus-1 to deduce the location purely through visual cues. Impressively, it identifies key visual elements such as manicured hedges, ornamental trees, and the distinctive facade of a grand building—critical factors in pinpointing the exact setting. This capability showcases Pegasus-1’s proficiency in analyzing and interpreting essential visual information to arrive at accurate conclusions.

- - Key Frames of Video
- - Pegasus-1

Based on the provided video content, the video appears to have been filmed at the Arashiyama Bamboo Grove in Kyoto, Japan. This location is famous for its natural beauty and cultural significance. The grove's tall bamboo stalks create a unique visual experience, and its proximity to historical sites like the Tenryu-ji Temple adds to its cultural value.

The video opens with a serene view of the bamboo forest, showcasing the sunlight filtering through the tall bamboo stalks. It then transitions to a traditional Japanese temple or building, characterized by its golden roof and intricate architectural details. The temple is surrounded by a well-maintained garden with a pond, and the changing colors of the leaves suggest that the location might be autumn.

The video focuses on natural elements, traditional architecture, and the serene atmosphere, which suggests that the location is a place of cultural and spiritual significance, likely visited for its historical and aesthetic values. The footage includes scenes of people walking through the bamboo forest and along a pathway, possibly tourists or visitors enjoying the scenery. One individual is even seen using a camera gimbal, indicating that the location is popular for photography and videography.

The exact coordinates of the shooting location are 35.0299° N, 135.6576° E. I believe this site was chosen for the video due to its renowned natural beauty and cultural significance, making it an ideal choice to capture the essence of Japan.

- - Gemini 1.0 Pro

|The video was shot at Kiyomizu-dera, a Buddhist temple in Kyoto, Japan. The temple is located on the eastern side of the cit y, in the Higashiyama ward. The temple is famous for its wooden stage, which is supported by 139 pillars and overlooks the city of Kyoto. The temple is also known for its beautiful gardens, which are home to a variety of plants and flowers. I chose t o shoot the video at Kiyomizu-dera because it is a beautiful and historic location. The temple is a UNESCO World Heritage S ite, and it is one of the most popular tourist destinations in Kyoto. I wanted to capture the beauty of the temple and its surrou ndings, and I believe that the video does this successfully.|
|---|

- - Question

|Provide the geographical coordinates of where the video was shot and explain the rationale b ehind choosing this site.|
|---|

Title: Beautiful Kyoto Scenery Compilation

[Figure 57]

[Figure 58]

(source: https://www.youtube.com/watch?v=aNC3UOYOejI)

Real-World Knowledge: Place Detection

- - Gemini 1.5 Pro

(source: https://www.youtube.com/watch?v=3LpuGYsOD9A)

|The video was shot in Kyoto, Japan. The specific coordinates are 35.0116° N, 135.7681° E. Kyoto is a city with a rich history and culture, and it is home to many beautiful temples and shrines. The video showcases some of the most iconic landmarks in Kyoto, including the Arashiyama Bamboo Grove, Kinkaku-ji Temple, and Kiyomizu-dera Temple. The rationale behind choosing this site is that it is a popular tourist destination and offers a glimpse into the beauty and culture of Japan.|
|---|

- Figure 3: The input video showcases various landscapes across Kyoto. Pegasus’s response meticulously orders scenes to reflect their chronological appearance in the video, detailing Kyoto’s landmarks in sync with their sequence. It intelligently infers the filming season as autumn, deduced from the presence of autumn foliage. Adhering precisely to the inquiry, Pegasus concludes by accurately providing Kyoto’s coordinates, demonstrating its adeptness in both recognizing visual patterns and extracting relevant contextual information from the imagery presented.

- - Key Frames of Video
- - Pegasus-1

|Based on the visual information presented in the video, the feature car is a Bugatti Chiron. The Bugatti Chiron is a high-performance luxury sports car, renowned for its speed and agility. In the video, the car is black in color with distinctive yellow accents on the grille and side sills. The opening scene of the video shows the car accelerating down the runway, leaving a trail of tire smoke in its wake. The Bugatti Chiron then performs a series of high-speed turns, showcasing its remarkable handling capabilities. Throughout the video, the powerful exhaust note of the car adds to the overall sensory experience. The airstrip where the demonstration takes place is surrounded by a forested area, providing a serene and picturesque backdrop to the high-octane action. At the end of the video, the Bugatti Chiron's logo appears, signifying the completion of the demonstration.|
|---|

- - Gemini 1.0 Pro

|Bugatti Bolide|
|---|

- - Question

|What car is this?|
|---|

Real-World Knowledge: Automobile model detection

Title: Bugatti Chiron

[Figure 59]

[Figure 60]

(source: https://www.youtube.com/watch?v=NMThdHhrLoM)

- - Gemini 1.5 Pro

(source: https://www.youtube.com/watch?v=3LpuGYsOD9A)

|This is a Bugatti Chiron Super Sport 300+|
|---|

- Figure 4: The video showcases a Bugatti Chiron performing high-skill maneuvers, including producing smoke from its tires. Pegasus accurately identifies the exact car model, showcasing its broad knowledge of the real world. This highlights Pegasus’s ability to recognize and interpret specific details from visual inputs.

- - Key Frames of Video
- - Pegasus-1

Based on my comprehensive analysis of the video, the game being played in the footage is identified a s "The Legend of Zelda: Breath of the Wild." In the early stages of the gameplay, there is an outdoor c ombat scene taking place on a wooden platform. The character in the red cape and armed with a swor

- d is engaging in a battle against a character wielding a large hammer. It appears that the character wit h the sword possesses a special ability, indicated by an electric blue light emitted during the attack. Th
- e character using the hammer, seemingly a standard enemy, takes damage and exhibits a health bar a bove their head.

The environment presents a fantasy setting, hinting at a high-fantasy game with themes rooted in medi eval or ancient times. Despite the quick sequence of the video, I was able to gather this information thr ough my in-depth understanding of the visual aspects presented, in addition to using my reasoning and inference abilities to supplement any ambiguities or potential errors. Therefore, the name of the game mentioned in the question is "The Legend of Zelda: Breath of the Wil d." This information is accurate and supported by the visual and audio details contained in the video, of fering a polished and detailed response.

- - Gemini 1.0 Pro

|The Legend of Zelda: Breath of the Wild|
|---|

- - Question

|What is the name of this game?|
|---|

Real-World Knowledge: Electronic game title detection

Title: Short Gameplay Clip of The Legend of Zelda: Breath of the Wild.

[Figure 61]

[Figure 62]

(source: https://www.youtube.com/shorts/AqMp-2Zd7dA)

- - Gemini 1.5 Pro

(source: https://www.youtube.com/watch?v=3LpuGYsOD9A)

|The Legend of Zelda: Breath of the Wild|
|---|

- Figure 5: The video features a brief gameplay moment from "Legend of Zelda: Breath of the Wild." Unlike the Gemini models, Pegasus provides a detailed visual analysis, such as noting the enemy’s health bar, pinpointing the scene’s location on a wooden platform, and mentioning the electric blue light during an attack. Additionally, Pegasus accurately identifies the game’s title, showcasing its ability to interpret and convey comprehensive visual details along with correct contextual understanding.

Title: Tsunami captured from a close distance

[Figure 63]

[Figure 64]

(source: https://www.youtube.com/shorts/SqaCd_5u7r0)

###### - Question

|What will be happening after this video?|
|---|

###### - Pegasus-1

Based on the information provided in the video, it appears that the individuals in the coastal area are currently experiencing a powerful natural event, such as a large wave or tidal surge. After the video ends, it is likely that they will continue their work or activities in the area, taking steps to secure their boats and equipment against the harsh weather conditions. They may also be preparing to leave the area or take shelter once the immediate danger has passed. The aftermath of such a storm would involve assessing any damage to the boats, equipment, and the surrounding environment, as well as recovering from the impact of the storm on the local community and ecosystem.

The visuals in the video show a group of people on the rocky shore, with boats and debris scattered around. The wave begins to approach, and as it hits, it causes a significant amount of water to splash onto the shore and the people nearby. The force of the wave is strong enough to move objects and people around, indicating the intensity of the event. The people on the shore appear to be caught off guard, and some are seen running or moving away from the incoming water. The video ends with the wave fully crashing onto the shore, and the final frames are obscured by the water and spray. The dialogue in the video is primarily brief greetings and calls to come closer to the camera. It does not provide any specific information related to the natural event or the actions of the individuals in the aftermath.

Overall, the video suggests that the individuals in the coastal area are currently experiencing a significant and potentially damaging storm. After the storm passes, they will need to assess the damage and take steps to secure their boats, equipment, and community against further harm.

###### - Gemini 1.0 Pro

|The video shows a large wave coming towards the shore. There are several boats on the shore, and it looks like the wave is going to destroy them. The people in the video are running away from the wave. The wave hits the shore and destroys the b oats. The people who were running away are safe, but they are probably very scared. The video is a reminder of how powerful nature can be. It is also a reminder of the importance of being prepared for natural disasters.|
|---|

###### - Gemini 1.5 Pro

|The video shows a tsunami hitting a village. The tsunami is likely to cause significant damage to the village, including flooding, destruction of buildings, and loss of life. The people in the village will need to evacuate to higher ground and seek medical attention if necessary. The government and relief organizations will need to provide assistance to the village in the aftermath of the tsunami.|
|---|

- Figure 6: The video captures a tidal surge, setting a complex scenario for Pegasus-1 to analyze. Tasked with predicting the aftermath, Pegasus-1 demonstrates both an understanding of the current situation and the ability to foresee post-disaster actions, such as securing boats and assessing damage. This highlights Pegasus-1’s comprehensive reasoning capabilities and awareness of real-world phenomena and their consequences.

Title: Suspicious individual recorded by a dashcam while driving at night on a highway

[Figure 65]

[Figure 66]

(source: https://www.youtube.com/watch?v=ClPZINVp0y8)

###### - Question

|What would the man do after the video ends?|
|---|

###### - Pegasus-1

Based on the information provided in the video clip, it appears that the man in the video is driving on a highway at n ight. The footage shows the dashboard of his vehicle, displaying the speed, which is 81 mph. The man initially drive s through a dark and empty stretch of road, but as he proceeds further, he enters a construction zone. Here, the ro ad is marked by orange traffic cones and is illuminated by streetlights. The man slows down his vehicle as per the r educed speed limit indicated by a sign.

Although the video does not depict the man performing any specific actions after the construction zone or towards t he end of the clip, it can be inferred that he would continue driving on the highway following the road and any traffic ahead of him. As there are no indications of him exiting the vehicle or stopping, it can be assumed that the man int ends to complete his journey on the highway.

Therefore, to answer the question, "What would the man do after the video ends?", it can be said that based on the visual information provided, the man would continue driving on the highway, focusing on the road and adjusting to any traffic conditions ahead.

###### - Gemini 1.0 Pro

|The man will run away.|
|---|

###### - Gemini 1.5 Pro

|It is impossible to know what the man would do after the video ends. The video shows a man walking on the side of a highway at night. He is wearing dark clothing and is difficult to see. The video ends before the man reaches his destination. It is possible that the man was simply walking to his destination and would continue to do so after the video ended. However, it is also possible that the man was planning to do something else, such as commit a crime. Without more information, it is impossible to say for sure what the man would do after the video ended.|
|---|

- Figure 7: The video depicts an individual driving on a highway at night, encountering traffic cones along the way. Pegasus-1 accurately deduces that the scene likely represents a construction site, a reasonable explanation for the cones’ presence on the highway during nighttime, despite the absence of explicit context. Pegasus-1 predicts the driver will continue to follow the road, demonstrating its ability to process and interpret not only the visual but also the temporal dynamics of the scenario. This is evident from Pegasus-1’s responses highlighted in green, showcasing its nuanced understanding. Conversely, Gemini 1.0 Pro’s interpretation significantly deviates from the video’s content, offering an unrelated response that fails to acknowledge the situational elements depicted.

- - Key Frames of Video
- - Pegasus-1

Based on the video, the property in question is a two-story brick house located in New York. While the video does not provide an exact price of the property, some clues can be gleaned from the visual and dialogue elements prese nted.

The house boasts a number of desirable features, such as a garage, modern kitchen appliances, hardwood floorin g, and a well-maintained backyard. Additionally, there are signs of a child or nursery in the house, suggesting that i t may be a family home. These factors, along with the location in New York, suggest that the property is likely to be on the pricier side.

A rough estimate of the property value based on the video might be in the range of $500,000 to $800,000. Howeve r, it is important to note that this is only a rough estimate, and the actual price could vary significantly depending on a multitude of factors, such as the specific location within New York, the square footage of the house, and the cond ition of the property.

In summary, while the video does not provide an exact price of the property, it does offer some insight into the desi rable features and location of the house, which can be used to make an educated guess at its potential value.

- - Gemini 1.0 Pro

|This house is located in Queens, New York. The house is a Cape Cod-style home with 4 bedrooms and 2 bathroo ms. It has a living room, dining room, kitchen, and finished basement. The house is in good condition and has been recently renovated. The backyard is small but has a patio and a garden. The house is located in a quiet and safe n eighborhood, close to schools, shops, and public transportation. I would estimate that the house is worth around $7 50,000.|
|---|

- - Question

|The house is located in New York. Can you guess the price of the property based on the video? Tell me in U SD. It does not have to be precise, use your knowledge to guess the value of the property|
|---|

Video-Based Reasoning: Reasoning in Real Estate Value Estimation

Title: Photo compilation of real estate properties.

[Figure 67]

[Figure 68]

(source: https://www.youtube.com/watch?v=pYT_F9aCFN0)

- - Gemini 1.5 Pro

|Based on the video, the house appears to be a small, single-family home in a residential neighborhood. It has a brick exterior, a front porch, and a backyard. The interior of the house appears to be well-maintained and updated, with hardwood floors, a modern kitchen, and updated bathrooms. Given the location and the condition of the house, I would estimate that the property is worth between $500,000 and $700,000.|
|---|

- Figure 8: The video features a compilation of real estate photos. Both Pegasus-1 and the Gemini families offer logical evaluations, suggesting a price range that encompasses the actual value of $695,000. Pegasus-1 notably excels by discerning specific visual details, such as signs of a child’s presence in the nursery and the quality of hardwood flooring, showcasing its adeptness at detailed visual analysis.

- - Key Frames of Video
- - Pegasus-1

In the presented video, an intriguing and unconventional scene unfolds, as a cat is depicted sitting in the driver's seat of a Tesla car. The cat, donning a floral shirt, engages with the car's steering w heel and controls, creating a humorous and endearing illusion that it is driving the vehicle. This sit uation is atypical because, in reality, cats are incapable of operating vehicles or comprehending th eir functions. Thus, the scene is intended to bring delight and amusement to the audience.

The video begins with the cat situated in the driver's seat, busily interacting with the car's interior. The vehicle is in motion, evident from the rapidly changing scenery out of the windows and the dy namic movement of the steering wheel. The video then captures various angles offering close-up views of the cat, as well as brief glimpses of the dashboard and the road before them. Despite the cat's small size and lack of ability to operate the vehicle in reality, its playful demeanor in the drive r's seat creates a captivating and unexpected spectacle for the viewer.

- - Gemini 1.0 Pro

|The video shows a cat sitting in the driver's seat of a Tesla Model 3, wearing a hawaiian shirt, and appearing to be driving the car.|
|---|

- - Question

|Describe the unusual or atypical aspect featured in this video.|
|---|

Video-Based Reasoning: Spotting Oddities

Title: Funny video of cat "Driving"

[Figure 69]

(source: https://www.reddit.com/r/AnimalMemes/comments/1bdt92e/guess_the_upcoming_moment/)

- - Gemini 1.5 Pro

|The video shows a cat wearing a shirt and sitting behind the wheel of a Tesla car, seemingly driving it. Cats are not capable of driving cars, so this is an unusual and atypical sight.|
|---|

- Figure 9: The video depicts a cat humorously positioned as if "driving" a Tesla, clad in a floral shirt and interacting with the steering wheel. Pegasus-1 effectively notes the comedic illusion of the cat driving and observes the car’s motion through the rapidly changing scenery, highlighting its capability to capture and interpret key visual elements within the content.

- - Key Frames of Video
- - Pegasus-1

To get to the fridge from the porch of the modern residential property, you would first need to enter the house. Once inside, make your way through the living area. Depending on the layout of the specific property being showcased in the video, you may find the kitchen adjacent to the living room. The fridge will typically be located in the kitchen amongst other appliances and cabinetry, making it easily accessible.

In the first clip, the property tour starts by introducing us to the exterior of the townhouse community and then proceeds to s howcase various rooms and features inside the property. The kitchen is revealed to have a modern refrigerator, stove, and dishwasher, and it is usually an easy transition from the living area to the kitchen.

In the second clip, the video continues the tour by showcasing additional interior features, such as a master suite with a larg e bedroom and a well-lit bathroom. Afterward, the video progresses through the living area with a ceiling fan and fireplace b efore transitioning to the kitchen, where the refrigerator is located.

In the third clip, the video continues to provide a comprehensive walkthrough of the apartment, starting on the balcony and proceeding to the spacious living area combined with the kitchen. The refrigerator can usually be found in the kitchen area within arm's reach, making it convenient to access.

Therefore, from the information presented in the video and your understanding of the situation, the best approach to get to t he fridge from the porch would be to enter the house and make your way through the living area, ultimately reaching the kitc hen area where the fridge would be located.

- - Gemini 1.0 Pro

|Go up the stairs, turn left, and go through the door on the left. The fridge is on the left.|
|---|

- - Question

|Suppose I am in front of the porch. Tell me the way to get to the fridge|
|---|

3D Spatial Understanding: Embodied Agent

Title: Introduction to real estate in Atlanta

[Figure 70]

[Figure 71]

(source: https://www.youtube.com/watch?v=04eXkFmES9U)

- - Gemini 1.5 Pro

|1. Go up the stairs to the main level.<br>2. Turn right and walk through the dining room.<br>3. The kitchen will be on your left.<br>4. The fridge is located on the left side of the kitchen.<br>|
|---|

- Figure 10: Pegasus-1 adeptly reconstructs spatial information from the video, guiding seamlessly from the porch into the living room while accurately noting the adjacency of the kitchen. However, within the Gemini families, there’s a slight misunderstanding of the house’s 3D layout. Often, they misdirect after ascending the stairs or misplace the location of the fridge, necessitating correction.

- - Key Frames of Video
- - Pegasus-1

Based on the visual and audio information provided in the video, a detailed description of the video content would be as follows: The video starts by showcasing a cooking tutorial for a savory chicken and rice dish. The instructor begins by placing raw chicken pi eces into a large glass bowl, located on a marble countertop. The chicken is then ground into a coarse paste using a blender. Follow ing this, various whole spices like cumin seeds, dried red chilies, and black peppercorns are crushed in a mortar and pestle, which a re then added to the ground chicken in the bowl. Other ingredients such as yogurt and vegetable oil are also added to the mixture, w hich is then stirred to form a marinade. The next step involves the preparation of vegetables. A red onion is finely sliced, and green herbs, possibly cilantro, are chopped. T hese vegetables are then mixed into the marinated chicken mixture. The cooking process initiates as the marinated chicken is place d in a pan with oil, where it is sautéed. As the chicken cooks, sliced onions are added to the pan and stirred regularly. The dish is th en garnished with saffron strands, which is known for its distinctive flavor and color. Rice is rinsed and cooked separately in a pot, with the rice grains being steamed to perfection. Once the rice is cooked, it is combin ed with the cooked chicken and vegetables, forming a one-pot meal. The dish is then served heated on a plate, with a side of raita, a yogurt-based sauce, to complement the savory flavors of the chicken and rice. The audio information in the video primarily consists of non-English sounds and no clear dialogue. The occasional sounds of choppi ng, grinding, and sizzling indicate the various steps in the cooking process. The overall video showcases a comprehensive cooking tutorial for a traditional savory chicken and rice dish that is commonly found i n various cuisines around the world. The use of whole spices, herbs, and vegetables enhances the flavors of the dish, and the visual presentation makes for an engaging and informative learning experience for viewers.

- - Gemini 1.0 Pro

|The video begins with a bowl of chicken pieces. The chicken is then marinated in a mixture of spices and yogurt. Once the chicken is marinated, it is cooked in a pot with onions, tomatoes, and spices. The rice is then added to the pot and cooked until it is fluffy and te nder. Finally, the biryani is garnished with cilantro and served with raita. The video is well- paced and easy to follow. The narrator sp eaks clearly and concisely, and the in structions are easy to understand. The video is also visually appealing, with bright and colorful ingredients and a clean, modern kitchen. Overall, this is a great video for anyone who wants to learn how to make chicken biryani. It is easy to follow, visually appealing, and the narrator speaks clearly and concisely. I would highly recommend this video to anyone w ho is interested in learning how to make this delicious dish.|
|---|

- - Question

|Describe this video in detail.|
|---|

Temporal Reasoning: Cooking Video

Title: How to cook chicken biriyani.

[Figure 72]

(source: https://www.youtube.com/watch?v=G3fz4H-HrQE)

[Figure 73]

- - Gemini 1.5 Pro

|The video shows how to make chicken biryani. It starts with a person cutting chicken into pieces and placing them in a bowl. Then, they add spices to a pan and heat them until they are fragrant. They then add the spices to a blender and blend them into a powder. The person then adds the spice powder, yogurt, and turmeric to the chicken and mixes everything together. They then chop onions and fry them until they are golden brown. They then add the chicken to a pot of boiling water and cook it until it is done. They then add the rice to the pot and cook it until it is done. They then garnish the biryani with fried onions, cilantro, and mint. Finally, they serve the biryani with raita.|
|---|

- Figure 11: The video showcases the step-by-step process of making biryani. Pegasus explains the sequence of actions in detail: starting with cooking the chicken, followed by preparing the spices, then adding onions, cooking the rice, and finally combining all the components to create the final dish, biryani. Also, an intriguing error in Gemini 1.0 Pro’s response is the false detection of narration—a feature commonly expected in such videos—when, in fact, there is no narration present.

### Temporal Reasoning: London Scenery Compilation

- - Key Frames of Video
- - Pegasus-1

###### Title: Compilation of landmarks of London.

[Figure 74]

[Figure 75]

(source: https://www.youtube.com/watch?v=sNJZkxXjlVg)

- - Gemini 1.0 Pro

|The video is a one- minute travel guide to London, England. It features various iconic landmarks of the city, such as the Tower of Lo ndon, Tower Bridge, the Houses of Parliam ent, and Big Ben. The video also provides a glimpse of some of the city's lesser- known areas, such as the Lea River and the London Docklands. The video is narrated by a British man with a posh accent.|
|---|

- - Question

|Describe this video in detail.|
|---|

The video in question presents an engaging and enlightening exploration of London's diverse landscapes and iconic landmarks. Beg inning with a bird's-eye view of residential areas, the scene sets the stage for a captivating journey through the heart of London. As t he camera swiftly pans through the city, we are introduced to a fast-paced sequence of its most recognizable sites, beginning with t he historic Tower of London and the Houses of Parliament, both bathed in natural light, bringing their architectural grandeur to life.

The bustling streets burst with activity as London's signature red double-decker buses and black taxis make their way across the me ticulously planned cityscape. The juxtaposition of the ancient and the modern continues as we take in the contemporaneous skyline, graced with sleek, modern skyscrapers standing proudly alongside the city's traditional and historic structures.

Venturing further, the video captures the essence of London's contrasting urban and natural aspects. We observe scenes of London ers unwinding in a tranquil parks, surrounded by an array of colors and sounds that characterize the city's vivacious, cultural fabric.

Moving on, we witness the iconic Big Ben clock tower, ticking away time as it has for centuries, before being transported to the sere ne Thames riverbank, where the city's famous bridges frame a picturesque reflection of London's diverse skyline.

Employing a time-lapse sequence, the video concludes with a breathtaking transition from day to night, where the London Eye and o ther landmarks are illuminated against the evening sky. A sense of awe and admiration pervades as the London skyline undergoes an enchanting transformation.

Overall, this visually stimulating video offers a holistic perspective of London – a city steeped in history and tradition, yet contempora ry and bustling with life.

###### - Gemini 1.5 Pro

|The video is a montage of various scenes from London, England. It begins with an aerial view of the city, showing the skyline and the River Thames. The video then cuts to a shot of a park, with people walking and relaxing on the grass. The next scene shows a busy street, with red double-decker buses and black cabs driving by. The video then cuts to a shot of Big Ben and the Houses of Parliament, followed by a shot of the London Eye. The video ends with a shot of people walking down a street, with the sun setting in the background.|
|---|

- Figure 12: The video comprises a compilation of short shots capturing landmarks of London. Pegasus meticulously describes each shot, adhering to the chronological order in which they appear. Conversely, Gemini families occasionally omit certain shots, resulting in a relatively low recall rate. One notable error in Gemini 1.0 Pro is its erroneous suggestion of the presence of "narration by a British man with a posh accent," despite the absence of any narration in the video. This incorrect detection by Gemini 1.0 Pro is particularly noteworthy since such videos commonly feature narration, yet in this instance, no narration is present. 19

- - Key Frames of Video
- - Pegasus-1

The video depicts a soccer match where a player identified by a circle is prominently featured. Early in the clip, this player is observed playing an active role in the game, contributing to offensive maneuvers in an attempt to move the ball closer to the opposing team's goal and create potential scoring chances. Their involvement is evident through their actions, which consist of dribbling, passing the ball to teamm ates, and making runs into the opponent's half of the field.

The broader context of the video reveals a competitive atmosphere, as players clad in white and darker

- -colored uniforms engage in a lively contest, with some players in continuous motion. The match takes place on a synthetic turf field, marked with white lines denoting boundaries and yardage, and is enclos ed by metal bleachers. The goalposts, situated at either end of the field, indicate the objective of the ga me—namely, to score goals by getting the ball past the opposing team's goalkeeper. The scoreboard i n the backdrop provides a sense of the game's progression, although specific scores or remaining time are not clearly visible from the provided frames.

Overall, this player's participation in the soccer match, along with the game's dynamic context, highlight s the competitive nature and excitement of the sport.

- - Gemini 1.0 Pro

|The player marked as the circle is playing forward for the team in white. He is involved in the buildup of play and takes a shot on goal.|
|---|

- - Question

|Please describe the action of player marked as circle in this video.|
|---|

Visual Referring Prompts: Soccer Player Pointing

Title: Dynamic soccer match moment.

(source: https://www.youtube.com/watch?v=CDXAb-wJlis)

[Figure 76]

- - Gemini 1.5 Pro

|The player marked as a circle in this video is playing as a forward. He is trying to receive a pass from his teammate. He is making a run towards the goal and is calling for the ball. He is also trying to stay onside.|
|---|

- Figure 13: The video features a brief segment of a soccer match, with a player highlighted in red engaged in an offensive play. Pegasus-1 adeptly focuses on this player within the red circle, accurately noting their involvement in offensive maneuvers aimed at advancing the ball closer to the opposing team’s goal.

[Figure 77]

https://www.youtube.com/shorts/t9P_d5EFlAE

###### - Question

|Describe pointed regions by white circles.|
|---|

###### - Pegasus-1

In the initial part of the video, which is shown in the first clip, we encounter a baseball pitching scene featuring a pitcher named Garrett Cole, identified by his uniform bearing the name 'CO LLE' and the number '54'. The atmosphere of the scene indicates a professional baseball ga me with a filled stadium of spectators. throughout this clip, an analyst shares his insights on t he pitcher's delivery and how it represents efficient linear energy.

The white circles depicted in the video serve to highlight the region where the pitcher's foot is expected to land, acting as a visual guide to illustrate the correct foot placement during the pit ching process. Based on the visual information presented, it's clear that these circles serve a crucial role in the execution of a proper pitching motion.

###### - Gemini 1.0 Pro

|bad|
|---|

###### - Gemini 1.5 Pro

|The white circles point to the pitcher's knee and foot. The knee is circled in the first few frames, and the foot is circled in the last few frames. The green arrows point to the direction of the pitch.|
|---|

- Figure 14: The video centers on an analysis of Gerrit Cole’s pitching mechanics, utilizing white circles and green arrows to visually underscore pivotal aspects. When asked about the areas designated by the white circles, Pegasus-1 showcases an understanding of their role as visual markers pertaining to pitching form. In contrast, Gemini 1.0 Pro entirely disregards the instructions. However, Gemini 1.5 Pro shifts its focus to the white circle following the query, demonstrating robustness compared to its previous version.

[Figure 78]

(source: https://www.youtube.com/shorts/t9P_d5EFlAE)

###### - Question

|Describe what is the meaning of green arrows in this video.|
|---|

###### - Pegasus-1

|In the video, green arrows serve as an indicative tool to demonstrate the direction of the pitch er's throwing motion and the trajectory of the baseball. These arrows help visualize the linear energy and the pitcher's consistency in his delivery without any misdirection or disconnects, c ontributing to an efficient pitching stance and successful delivery. The clear progression of the pitcher's movements, as shown in the video, ensures a straight and aligned throwing action.|
|---|

###### - Gemini 1.0 Pro

|The green arrows in the video are tracking the movement of the, baseball as it is thrown by the pitcher. The arrows show the path of the ball as it leaves the pitcher's hand travels throug h the air, and reaches the catcher's mitt. The arrows also show the spin of the ball as it is thro wn, which can affect the way the ball moves through the air and how it behaves when it is hit by the batter.|
|---|

###### - Gemini 1.5 Pro

|The green arrows in this video show the direction of the pitcher's momentum as he throws the ball. The arrows start at the pitcher's foot and point towards the batter. This shows that the pitcher is using his legs to generate power and drive the ball towards the batter.|
|---|

- Figure 15: The green arrow signifies the intended direction of the ball’s delivery by the pitcher. Both Pegasus-1 and the Gemini families comprehend the green arrow as a visual indicator of the pitching direction.

- - Key Frames of Video
- - Pegasus-1

Based on the video contents, I can confirm that there is an instance that appears unusual or dangerous, as queried. Towards the end of the clip, around the 23rd second, smoke or dust c an be seen rising from the side of the road. Although the cause of this event is not clear, it ma y indicate a vehicle incident or a fire.

The video's base description reveals it to be a recording from a vehicle's dashboard camera, showing a journey on a highway. The frames present a clear sky, the road ahead, and variou s vehicles, with a sign for 'Yerba Buena Rd' and 'Exit 43B' appearing towards the end. Mean while, the dialogue during this period is promoting Progressive Insurance and its affordability, encouraging finding policies to fit personal budgets. Overall, the situation appears to be an or dinary highway journey until the unexpected occurrence of the rising smoke or dust.

- - Gemini 1.0 Pro

|The video shows a car driving on a highway. The car is going at a high speed and weaving in and out of traffic. The driver is clearly not paying attention to the road and is at risk of causing an accident.|
|---|

- - Question

|Do you see any unusual or dangerous moment?|
|---|

Domain Specific Use Case: Dashcam analysis

Title: Dashcam footage of driving on the highway and spotting a car on fire

[Figure 79]

(source: https://www.youtube.com/watch?v=2DVM2v68CTA)

- - Gemini 1.5 Pro

|Yes, there is a car on fire on the side of the road. This is a dangerous situation, and it is important to stay away from the car and call for help.|
|---|

- Figure 16: This dashcam video captures a car on fire. Initially, the vehicle seems small and distant, but as the video continues, it comes closer. Pegasus-1 accurately detects the hazardous condition, highlighting the smoke rising from the roadside. Additionally, Pegasus-1 notes the dashcam-equipped car’s approach toward an exit, accurately reflecting the events in the video. On the other hand, Gemini 1.0 Pro fails to identify the danger presented in the dashcam video. However Gemini 1.5 Pro detects the car on fire, demonstrating superior performance compared to Gemini 1.0 Pro.

- - Key Frames of Video
- - Pegasus-1

Based on the provided video content, I have carefully considered each moment in the video that could require law enforcement attention, taking into account both the visual information and the context provided by the base descrip tion and extracted dialogue.

Early in the video, the traffic rules are being followed strictly. However, towards the end of the video, there are insta nces that may warrant law enforcement attention. One such instance is when a cyclist is seen riding on the road wit hout a helmet. This behavior is dangerous and against the law in many jurisdictions. Another instance is when a ve hicle appears to be making a turn without signaling. Failure to signal could potentially cause an accident and put ot her road users at risk.

It is important to note that I have gained a comprehensive understanding of the entire video, rather than limiting my self to specific clips or segments. I have integrated the information from the visual information, base description, an d extracted dialogue to provide a detailed and accurate response.

As a helpful assistant, I would recommend that these instances be reported to the appropriate law enforcement ag ency for further investigation and potential action.

- - Gemini 1.0 Pro

|The video contains a number of moments that appear dangerous or atypical and might require law enforcement att ention. These include: - A vehicle failing to stop at a stop sign. - A vehicle speeding and weaving through traffic. - A vehicle failing to yield to a pedestrian in a crosswalk. - A vehicle driving on the wrong side of the road. - A vehicle m aking an illegal U-turn. - A vehicle failing to signal a turn. - A vehicle driving with a broken headlight. - A vehicle drivi ng with a cracked windshield. - A vehicle driving with a missing license plate.|
|---|

- - Question

|Highlight any moment in the video that appears dangerous or atypical and might require law e nforcement attention.|
|---|

Title: Beautiful scenery captured by a dashcam during sunset.

[Figure 80]

[Figure 81]

(source: https://www.youtube.com/watch?app=desktop&v=EDK-t0KOlaQ )

Domain Specific Use Case: Dashcam analysis

- - Gemini 1.5 Pro

|At 00:27, a cyclist is riding in the middle of the road, which is dangerous and atypical. This might require law enforcement attention.|
|---|

- Figure 17: The dashcam video captures casual scenes of driving around a town. When queried about any potentially dangerous moments requiring law enforcement attention, Pegasus-1 identifies two incidents present in the video. One involves a man riding a bike on the road without a proper helmet, and the other pertains to a vehicle turning left without signaling. Gemini 1.0 Pro falls short of providing the correct answer, continuously printing responses that is not present in the video. (For convenience, it was truncated). Conversely, Gemini 1.5 Pro offers a relatively convincing response, noting that the cyclist is riding in the middle of the road.

- - Key Frames of Video
- - Pegasus-1

The video you have provided is an educational or informational piece focused on the surgical management of pituitary adenomas, which are benign tumors of the pituitary gland. The video begins by introducing the topic as 'The Surgical Management of Pituitary Adenomas' and presenting the objectives of the presentation, which include defining the anatomy of the pituitary gland, discussing the different types of pituitary adenomas, and outlining the surgical approaches to treating these tumors. The video then introduces three experts in the field of neurosurgery, Dr. Ismail Han, Dr. Daniel Yoshor, and Dr. Ali Serdar Goksun, who are likely to be the ones performing the surgery or discussing the procedure. Subsequent frames provide a comprehensive overview of the anatomy of the pituitary gland, using both illustrations and endoscopic footage to highlight the different structures within the gland, such as the anterior lobe, posterior lobe, infundibulum, and the hypothalamus. The vascular supply to the pituitary gland is also discussed, specifically mentioning the inferior hypophyseal artery and the superior hypophyseal artery. The video then delves into the surgical approaches to treating pituitary adenomas. The focus is on providing a comprehensive overview of the surgical management of pituitary adenomas, from the anatomy of the pituitary gland to the specific surgical techniques used to treat these tumors. The video is likely intended for medical professionals, such as neurosurgeons and endocrinologists, who are interested in learning more about the surgical management of pituitary adenomas. In summary, the video is an educationally focused piece that provides a comprehensive overview of the surgical management of pituitary adenomas. It covers the anatomy of the pituitary gland, the different surgical approaches to treating these tumors, and the specific surgical techniques used to perform the procedures. The video is likely intended for medical professionals who want to learn more about the subject.

- - Gemini 1.0 Pro

|The video describes a transphenoidal approach for pituitary macroadenoma. It starts with a brief introduction to the patient's history, which includes symptoms of chemical and symptomatic panhypopituitarism, headache, and blurry vision. The patient was found to have a 1.9 x 1.9 x 1.8 cm pituitary macroadenoma with distension and upward displacement of the optic chiasm. The patient underwent a transphenoidal approach to resection, pituitary hormone replacement, and management of acute postop Diabetes Insipidus with fluid restriction and DDAVP. The video then shows the surgical procedure in detail, with the surgeon first making an incision in the nasal septum and then using a variety of instruments to remove the pituitary macroadenoma. The video is narrated by the surgeon, who provides detailed commentary on the procedure. The video is for educational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Viewers are advised to consult with a qualified medical professional for any questions or concerns they may have regarding their health.|
|---|

- - Question

|Please describe the video in detail.|
|---|

Title: Video Presentation of Surgical Techniques

[Figure 82]

(source: https://www.youtube.com/watch?v=x3e39nctVaE )

Domain Specific Use Case: Medical Video Analysis

- - Gemini 1.5 Pro

[Figure 83]

|The video is a surgical demonstration of a transsphenoidal approach for pituitary macroadenoma. The video begins with an introduction to the Mount Sinai Skull Base Surgery team. The clinical history of the patient is then presented, followed by preoperative MRI scans showing the pituitary lesion. The video then shows the surgical procedure, which begins with a right-sided posterior ethmoidectomy and septoplasty to enhance access and facilitate instrument placement. The sphenoid sinus is then entered and the pituitary lesion is identified. The lesion is then carefully removed using a combination of suction and microdissection. The video concludes with the closure of the surgical site.|
|---|

- Figure 18: This video is part of a medical seminar demonstrating surgical management. While endoscopic videos are relatively rare, Pegasus-1 captures essential information. It recognizes that it is an educational video for the neurosurgery field, implying the potential usage of Pegasus-1 for domain-specific cases.

##### 5 Limitations

Despite the initial success of Pegasus in various domains, there are limitations that have been identified and are targeted for improvement in future developments.

Maximum Video Duration The current iteration of Pegasus is optimized for videos up to 15 minutes in length. A notable limitation is its diminishing efficiency in capturing the relationships between timestamps that are significantly apart as the video length increases. Enhancing the model’s capability to process longer videos and maintain accuracy in connecting distant frames stands as a principal objective in our road map.

Hallucinations Pegasus, like other vision-language models, is prone to hallucinations, which can manifest as inaccurately identifying nonexistent objects, actions, or the sequential order of events. These inaccuracies underscore the need for further refinement in the model’s interpretive accuracy.

Safety and Biases Safety and biases represent a critical area of concern. Pegasus can unintentionally perpetuate biases present in its training data, leading to the reinforcement of stereotypes or incorrect perceptions. Additionally, the model may validate or produce content that is inappropriate or harmful, posing risks of spreading misinformation. The reliance on visual cues for interpretation further complicates this issue, as ambiguities in these cues can result in interpretative errors. Moreover, the current capabilities of Pegasus do not fully grasp the complexity of social contexts and dynamics, potentially leading to outputs that are contextually inappropriate. Efforts to mitigate these challenges will focus on developing strategies for creating more equitable and ethically aware models and improving content moderation mechanisms.

Chat Capability The absence of chat functionality in the current version of Pegasus is also noted as a limitation. Plans to incorporate chat capabilities in future versions are underway, aiming to enhance the model’s interactivity and user engagement.

Overall, addressing these limitations requires a dedicated approach to research and development, with the aim of advancing the model’s performance, ethical standards, and user experience.

##### Authorship

This is a joint team effort across multiple functional groups including model and data, engineering, product, and business development. (“core” indicates Core Contributor; first-name alphabetical order)

Model Raehyuk Jung (core), Hyojun Go (core), Jaehyuk Yi (core), Jiho Jang (core), Aiden Lee, Cooper Han, Jae Lee, Jeff Kim, Jin-Young Kim, Junwan Kim, Kyle Park, Lucas Lee, Mars Ha, Minjoon Seo,

Data Daniel Kim (core), Jay Suh (core) Deployment Abraham Jo, Ed Park, Hassan Kianinejad, SJ Kim, Tony Moon, Wade Jeong Product Andrei Popescu, Esther Kim, EK Yoon, Genie Heo, Henry Choi, Jenna Kang, Kevin Han, Noah Seo, Sunny Nguyen, Ryan Won, Yeonhoo Park Business & Operations Anthony Giuliani, Dave Chung, Hans Yoon, James Le, Jenny Ahn, June Lee, Maninder Saini, Meredith Sanders, Soyoung Lee, Sue Kim, Travis Couture

##### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [2] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023. 1
- [3] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1– 113, 2023. 1
- [4] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 1
- [5] Twelve Labs. Introducing Pegasus-1, 2023. 1, 3
- [6] Twelve Labs. Introducing marengo-2.6-medium, 2024. 2
- [7] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355,

2023. 1, 3, 4

- [8] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. arXiv preprint arXiv:2311.17005, 2023. 1, 3, 4, 5
- [9] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043, 2023. 1, 3, 4
- [10] Ruyang Liu, Chen Li, Yixiao Ge, Ying Shan, Thomas H Li, and Ge Li. One for all: Video conversation is feasible without video instruction tuning. arXiv preprint arXiv:2309.15785,

2023. 1, 3, 4

- [11] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 4, 5
- [12] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 1, 3, 4, 5
- [13] Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, volume 24, pages 109–165. Elsevier, 1989. 3
- [14] Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick. Linearly mapping from image to text space. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 6
- [15] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 3, 5
- [16] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 3, 4, 5
- [17] Twelve Labs Team. Pegasus-1 open beta: Setting new standards in video-language modeling,

2024. 1

- [18] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1

- [19] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 1, 3, 4, 5
- [20] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9127–9134, 2019. 1, 3, 4, 5
- [21] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 3, 4
- [22] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023. 1

