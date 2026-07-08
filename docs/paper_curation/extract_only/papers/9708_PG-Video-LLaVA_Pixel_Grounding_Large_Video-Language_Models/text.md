## PG-Video-LLaVA: Pixel Grounding Large Video-Language Models

Shehan Munasinghe1∗, Rusiru Thushara1∗, Muhammad Maaz1, Hanoona Abdul Rasheed1, Salman Khan1,2, Mubarak Shah4, Fahad Khan1,3 1Mohamed bin Zayed University of AI, 2Australian National University 3Linköping University, 4University of Central Florida

Project: https://github.com/mbzuai-oryx/Video-LLaVA

# arXiv:2311.13435v2[cs.CV]13Dec2023

### Abstract

|What does the child in yellow hold outdoors?<br><br>[Figure 1]|
|---|

Extending image-based Large Multimodal Models (LMMs) to videos is challenging due to the inherent complexity of video data. The recent approaches extending image-based LMMs to videos either lack the grounding capabilities (e.g., VideoChat, Video-ChatGPT, Video-LLaMA) or do not utilize the audio-signals for better video understanding (e.g., Video-ChatGPT). Addressing these gaps, we propose PG-Video-LLaVA, the first LMM with pixellevel grounding capability, integrating audio cues by transcribing them into text to enrich video-context understanding. Our framework uses an off-the-shelf tracker and a novel grounding module, enabling it to spatially localize objects in videos following user instructions. We evaluate PG-Video-LLaVA using video-based generative and question-answering benchmarks and introduce new benchmarks specifically designed to measure prompt-based object grounding performance in videos. Further, we propose the use of Vicuna over GPT-3.5, as utilized in VideoChatGPT, for video-based conversation benchmarking, ensuring reproducibility of results which is a concern with the proprietary nature of GPT-3.5. Our framework builds on SoTA image-based LLaVA model and extends its advantages to the video domain, delivering promising gains on video-based conversation and grounding tasks.

|The child in yellow holds a tennis racket.<br><br>[Figure 2]|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|Who walks to the door and opens it?<br><br>[Figure 7]|
|---|

|A man walks to the door and opens it.<br><br>[Figure 8]|
|---|

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Figure 1. Video spatial grounding on example videos from VidSTG [48] (above) and HC-STVG [34] (below) datasets. PGVideo-LLaVA can generate textual responses with referred objects grounded in the video content (tennis racket and man are localized in the top and bottom examples, respectively).

approaches towards video-LMMs such as VideoChat [15], Video-LLaMA [45], and Video-ChatGPT [22] have demonstrated capabilities in video comprehension and dialogue, they lack the crucial feature of visual grounding. Visual grounding in videos aims to associate the LMM responses to specific objects within the video input. Addressing this gap, we introduce PG-Video-LLaVA, the first video-LMM capable of localizing objects appearing in LMM responses. This task leads to enhanced intractability and demonstrates deep understanding of video content.

### 1. Introduction

Recent efforts on Large Multimodal Models (LMMs), spearheaded by GPT-4V [25], allow detailed conversations about images but generally do not scale well to videos. The magnitude of video data scales far beyond other modalities due to its massive volume on social and internet media. Furthermore, extending LMMs to videos is challenging due to their complex dynamics with long temporal context that needs to be understood accurately. Although recent

In PG-Video-LLaVA, we address the unique challenges posed by video data. The model is designed to track objects within shorter video clips that maintain consistent camera views, enabling accurate visual grounding across scenes and motions. This tracking links spatio-temporal segments directly to conversational elements, enhancing the model’s

*Equal Contribution

contextual understanding. A salient feature of PG-VideoLLaVA is its modular design, allowing for easy integration with existing grounding modules and the flexibility to adapt to future enhancements in visual grounding technology. Moreover, PG-Video-LLaVA enriches its capabilities by incorporating audio context. It achieves this by leveraging video audio in a form understandable to LLM, which is particularly useful in situations where the auditory information is essential to the conversation. This inclusion broadens the model’s understanding, making it more versatile in interpreting video content.

Furthermore, this work introduces an improved framework for benchmarking video-based conversational models, pivoting from previous approaches [22] that predominantly used the proprietary GPT-3.5-Turbo model for evaluation. Given that GPT-3.5-Turbo is subject to changes at any time and lacks transparency due to its closed-source nature, it presents challenges in terms of reliability and reproducibility. To address this, we propose the use of Vicuna, an open-source LLM for benchmarking. This shift not only enhances reproducibility but also improves transparency in the evaluation process. We evaluate PG-Video-LLaVA using our improved benchmarks and show notable improvements over existing video conversational models like VideoChatGPT [22] and Video-LLaMA [45] in ungrounded dialogues, achieving state-of-the-art (SoTA) performance.

The key contributions of this work are:

- • We propose PG-Video-LLaVA, the first video-based LMM with pixel-level grounding capabilities, featuring a modular design for enhanced flexibility.
- • By incorporating audio context, PG-Video-LLaVA significantly enhances its understanding of video content, making it more comprehensive and aptly suited for scenarios where the audio signal is crucial for video understanding (e.g., dialogues and conversations, news videos, etc.).
- • We introduce improved quantitative benchmarks for video-based conversational models. Our benchmarks utilize open-source Vicuna LLM to ensure better reproducibility and transparency. We also propose benchmarks to evaluate the grounding capabilities of video-based conversational models.

### 2. Related Works

Recent advancements in Large Multimodal Models (LMMs) [8, 18, 50] and Large Language Models (LLMs) [7, 26, 36] have significantly transformed the artificial intelligence landscape, particularly in natural language processing and multimodal tasks. These breakthroughs have enhanced machine learning models’ ability to understand and generate human-like text, while also enabling more effective integration of various data types like images, sounds and videos with textual information. This progress represents a major leap in creating AI systems that can accurately

interpret and interact with a diverse range of content.

Large Language Models (LLMs): The natural language processing (NLP) field has undergone a revolution with the advent of LLMs such as GPT [4], LLaMA [36], OPT [46], and MOSS [27], particularly noted for their zero-shot learning abilities and adaptability. The development of models like InstructGPT [28] and ChatGPT [26] has further propelled advancements in conversational AI and complex query handling, chiefly through instruction tuning. Within the LLaMA framework, the emergence of opensource models such as Alpaca [35] and Vicuna [7] exemplifies how instruction tuning can significantly boost model performance. This shift towards open-source initiatives in language modeling, highlighted by models like Alpaca and Vicuna, indicates a growing trend towards more accessible and collaborative approaches in the field. In this work, we build on the open-source Vicuna LLM and extend it with multimodal capabilities. We also propose an open-source benchmark for video conversation and reasoning tasks using Vicuna LLM that is reproducible for fair evaluations.

Large Multimodal Models (LMMs): The field of AI has witnessed significant advancements with the development of vision-language models like CLIP [30], renowned for their impressive zero-shot capabilities using extensive image-text pairs during training. These models have proven effective in a variety of applications, from image detection and segmentation [3, 17] to more complex tasks such as 3D modeling and video analysis [23, 31, 33, 37]. The introduction of BLIP-2 marked a pivotal transition, pioneering the integration of image features encoded by a visual encoder with text embeddings, setting the stage for the evolution into Large Multimodal Models (LMMs). This advancement influenced subsequent models like LLaVA [19], InstructBLIP [8], and MiniGPT-4 [50], which further refined image-text feature alignment and instruction tuning. VideoChat [15], Video-ChatGPT [22] and VideoLLaMA [45] represents an extension of these LMMs, moving from image-based to video-based applications, while models such as Otter [14], mPLUG-Owl [42], LLaMaAdapter [9], and InternGPT [21] continue to push the boundaries of multimodal interaction. Despite these significant strides, challenges in achieving robust visual grounding in LMMs highlight key areas for ongoing research and development in this dynamic field. Further, effective integration of audio signals within LMMs for comprehensive video understanding is an open research question that this work aims to address.

Visual-Language Grounding: Grounded Large Language Models (LLMs) have made notable progress in enhancing visual and language comprehension. A diverse array of models including Kosmos-2 [29], Ferret [43], AllSeeing Model [38], LISA [13], BuboGPT [49], Shikra [5], and GLaMM [32] have employed various methodologies

to master complex grounding tasks. These models demonstrate proficiency in tasks like referring expression comprehension and image segmentation, showcasing the advanced image understanding capabilities of LLMs. Methodologically, Kosmos-2, Shikra, and All-Seeing focus predominantly on creating language-based context for visual grounding. In contrast, BuboGPT merges visual elements with language, and LISA leverages vision-language embeddings for producing segmentation masks. Furthermore, GLaMM is adept at generating natural language responses linked with object segmentation masks, facilitating detailed visual-textual interactions. However, challenges remain, such as LISA’s constrained performance in multi-object scenarios and the limitations of BuboGPT and GLaMM to image-based applications, not extending to video processing. To this end, we introduce PG-Video-LLaVA, a video conversational model with pixel-level grounding capability. Further, PG-Video-LLaVA incorporates audio transcripts alongside visual and textual data, aiming to provide a more detailed understanding of video content.

### 3. PG-Video-LLaVA

#### 3.1. Overview

In this paper, we introduce PG-Video-LLaVA, a novel Large Multimodal Model (LMM) designed to align video and audio representations with a Large Language Model (LLM). This integration equips PG-Video-LLaVA with the capability to proficiently manage both video and audio data in conversational contexts. Additionally, our method integrates a specialized plug-and-play module for effective video grounding (see Figure 2).

In constructing PG-Video-LLaVA, our approach integrates sophisticated mechanisms for aligning video and audio signals with language processing capabilities, thereby facilitating a comprehensive multimodal analysis. Central to our model is an advanced CLIP-based video encoder, which has been specifically adapted to process both spatial and temporal dimensions of video data. This adaptation enables a deeper understanding of video content, setting PGVideo-LLaVA apart from conventional image-centric models.

For training, PG-Video-LLaVA utilizes the VideoInstruct100K [22] dataset comprising 100K video instructions derived from ActivityNet-200 [11]. This diverse dataset ensures that the model is well-equipped to handle a broad spectrum of video contexts with high accuracy. In addition to visual processing, PG-Video-LLaVA incorporates stateof-the-art audio analysis by leveraging advanced audio transcription techniques, similar to those employed in WhisperX [2] and Whisper-AT[10]. This integration allows the model to process and understand audio inputs effectively, enhancing its overall multimodal interpretation capabilities.

While PG-Video-LLaVA’s foundation is based on the LLaVA-1.5 [18] framework, it is extended for videos to incorporate spatio-temporal representations, audio understanding and visual grounding capabilities. Its unique combination of enhanced video encoding, extensive training dataset, integrated audio processing and grounding capability marks it as a forward step in the field of LMMs.

#### 3.2. Architecture

Our architecture utilizes the CLIP ViT-L/14@336 as the visual encoder, which, unlike its original image-focused design, has been adapted for video processing in PG-VideoLLaVA. This adaptation is crucial for the model to capture spatio-temporal representations in videos effectively. In our model, video samples are represented as Vi ∈ RT×H×W×C, where T denotes the frame count. The encoder processes each of the T frames independently, treating them as a series of images. This leads to the generation of frame-level embeddings xi ∈ RT×h×w×D, where h = H/p and w = W/p, with p being the patch size (14 for ViT-L/14) and N = h × w indicating the total token count.

To construct a comprehensive video-level representation, we apply average pooling across the temporal dimension of these frame-level embeddings, resulting in a temporal representation ti ∈ RN×D. This temporal pooling technique effectively amalgamates information across multiple frames. Similarly, for spatial information, we achieve spatial representation zi ∈ RT×D through average pooling along the spatial dimension. The final video-level features vi are a combination of these temporal and spatial features, as shown in the equation:

vi = [ti zi] ∈ R(T+N)×D (1)

In our architectural design, the spatio-temporal feature extraction is inspired by Video-ChatGPT [22], with an additional enhancement of employing a higher resolution of 336×336 pixels to encode frame-level features.

Within the architecture of PG-Video-LLaVA, we have implemented a learnable Multi-Layer Perceptron (MLP), designated as g, to serve as our cross-modal connector. This MLP is intricately designed to project video-level features into the embedding space of the language decoder. This is inspired from LLaVA-1.5 [18], aiming to optimize the model’s multi-modal capabilities beyond what could be achieved with a simple linear projection. The process yields language embedding tokens Qv, calculated as follows:

Qv = g(vi) ∈ R(T+N)×K (2)

Text queries, denoted as Qt ∈ RL×K where L is the length of the query, are tokenized to be dimensionally compatible with these video embeddings. The combination of Qv and Qt is then fed into the language decoder, facilitating

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Entitiy Matching Module

Class Agnostic Object Tracker

Response

Response

The person in the video is a man who is sitting on the ground and petting a lion.

Large Language Model

[Figure 18]

Proposal Refinement

User Query Audio Transcript

System Command Spatio-temporal Video Representation

Grounding Module

System Command

[Figure 19]

MLP

Audio Tagging

lion | man | mountain

You are PG-VideoLLaVA, a large vision language model trained with video instruction data.

Temporal Features Spatial Features

Phoneme Model

Tagging Module

….

Whisper

Spatial Pooling Temporal Pooling

….

….

| | |
|---|---|

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

User Query

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

What is the person in the video doing?

[Figure 36]

Visual Encoder (CLIP)

Scene Detection

Voice Activity Detection

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

- Figure 2. Architecture of PG-Video-LLaVA: PG-Video-LLaVA integrates a CLIP-based visual encoder with a multimodal language model for video understanding. The CLIP visual encoder extracts spatio-temporal features from videos by averaging frame-level features across temporal and spatial dimensions. These features are then projected into the LLM’s input space using a learnable Multi-Layer Perceptron (MLP). The system features a grounding module for spatially locating textual descriptions within video frames, a class-agnostic object tracker, and an entity-matching module. Audio processing incorporates voice activity detection, phoneme modeling, and Whisperbased audio transcription, resulting in a multimodal pipeline that facilitates robust video-question answering. The architecture is trained on a hybrid dataset of video instructions, enabling the handling of diverse conversational contexts with high accuracy.

the seamless integration of video and textual data within the model (see Figure 2).

integrate Whisper-AT[10]. This advanced version of the Whisper model specializes in audio tagging. It annotates the audio stream with labels from an extensive set of 527 audio event classes, allowing for precise temporal resolution.

##### 3.2.1 Audio Modality Integration

In PG-Video-LLaVA, we have integrated an audio processing pipeline that significantly enhances the video-question answering capabilities by incorporating audio cues from the input, drawing inspiration from the architecture of WhisperX[2]. The process begins with the deployment of a Voice Activity Detection (VAD) model. This model is crucial for pinpointing speech-containing temporal segments within the audio track. Following the VAD’s identification of speech segments, these segments undergo processing—cutting, merging, and padding—to align with the input specifications of the Whisper model [24]. Simultaneously, a phoneme segmentation model operates in parallel, producing phone-level segmentations essential for the subsequent alignment of raw transcriptions with the audio.

The audio transcripts are then subjected to a multistage filtering process. Initially, a VAD-based filter is applied, followed by a phoneme-based forced alignment using the Whisper model, ensuring temporally accurate text transcriptions. Utilizing Whisper’s language identification feature, we eliminate non-English speech segments at this stage. For each identified sentence segment, we apply Whisper-AT [10] for audio tagging, focusing on the top three predicted audio classes. Segments that do not predominantly feature ‘speech’, or where ‘music’ probabilities significantly exceed ‘speech’, are excluded from further processing.

Finally, the integration of the audio transcript with the video component is executed through a carefully designed prompt template. This template is pivotal in guiding the system to understand user instructions, assimilate the video frames, and incorporate the transcriptions generated by the automatic speech recognition model. This structured ap-

The VAD model serves a dual purpose: it not only identifies speech segments but also aids in filtering out nonspeech audio components. To enhance the compatibility of transcriptions generated by Whisper with our model, we

proach ensures that PG-Video-LLaVA efficiently leverages all available modalities—visual and auditory—thereby enabling users to achieve task completion and query resolution based on a comprehensive analysis of both visual and auditory content (refer to Figure 2 for details).

- 3.2.2 Grounding Module

In PG-Video-LLaVA, our spatial grounding approach starts with processing video-question pairs to generate textual descriptions. These descriptions are then used for grounding within the video frames. Key noun phrases are extracted from the generated text using Vicuna, targeting the most critical content aspects. Simultaneously, an image tagging model, RAM [47], tags visual elements in each frame, creating a detailed map of the video content.

The video is segmented into smaller parts using PySceneDetect [1], based on changes in scene composition. This segmentation facilitates a more focused grounding process. In each segment, our grounding ensemble, composed of GroundingDINO [20], DEVA [6], and SAM [12], utilizes the image tags to create segmentation masks and tracking IDs for the identified visual elements.

The visual cues from these segmentation masks are then matched with the textual noun phrases using CLIP [30]. This matching process links text to the corresponding visual elements in the video, enhancing our understanding of the content.

In quantitative analysis, from the descriptive textual response to an interrogative text, a referring expression or a phrase is extracted using Vicuna. This phrase is input into our grounding module, which then generates segmentation masks and tracking IDs. We measure the spatial grounding accuracy of our model by calculating the Intersection over Union (IoU) between these segmentation masks and ground truth bounding boxes.

This systematic approach enables PG-Video-LLaVA to effectively ground textual descriptions within video content, thereby improving the performance and interpretability of video-question answering systems.

- 4. Experiments

#### 4.1. Implementation Details

We build our stronger baseline on top of LLaVA-1.5 which utilizes CLIP ViT-L/14@336 as the image encoder and Vicuna 1.5 as the LLM. We only tune the MLP projection layers during training, while keeping the rest of the architecture frozen. We finetune the model for 3 epochs using a learning rate of 2e−5 and an overall batch size of 32. The training of our 7B and 13B models took around 6 hours and 15 hours respectively on 4 A100 80GB GPUs.

For audio transcript extraction, base Whisper model is

used. Our grounding module is based on GroundingDINOT variant and CLIP ViT-B/32. For the image-tagging model we use RAM Swin-Large variant (with input size 384). DEVA Tracker is applied under online-setting in our experiments.

Vicuna-13b-v1.5 model is used in performing videobased conversational benchmarking, zero-shot question answering evaluation, and extracting the key noun or referring expression from the model output in the quantitative evaluation of the spatial grounding task. Further, Vicuna-13b-v1.5 was used to implement the entity matching as in [49].

#### 4.2. Stronger Baseline

This section provides an overview of the quantitative evaluations conducted to determine the effects of the strengthened baseline on PG-Video-LLaVA. To evaluate the impact of the enhanced baseline on PG-Video-LLaVA, we apply the benchmarking framework from Video-ChatGPT[22]. This framework measures performance on several axes critical for video-based conversational agents, including correctness of information, detail orientation, contextual understanding, temporal understanding, and consistency.

In order to facilitate a reliable and reproducible evaluation, we have updated our assessment pipeline by replacing GPT-3.5-Turbo with Vicuna-13b-v1.5. This adjustment addresses the limitations in reproducibility inherent to the closed-source nature of GPT-3.5-Turbo. Subsequently, we have re-assessed both PG-Video-LLaVA and other recent models to ensure a fair and consistent comparison. The results shown in Table 1 demonstrate that PG-Video-LLaVA outperforms the foundational Video-ChatGPT model and exhibits superior performance when compared to other recent contributions in the domain.

Following the quantitative assessment, the qualitative results in Figure 3 indicate the enhanced baseline’s impact on PG-Video-LLaVA’s performance. The PG-Video-LLaVA (13B) model exhibits improved accuracy in the information presented, a deeper level of descriptive detail, and a stronger alignment with the context and temporal progression of the videos. This advancement is particularly noticeable in the precise depiction of the child’s engagement with their surroundings and the giraffe’s behaviour, indicating a refined interpretation of both the activities and their settings. These qualitative insights are consistent with the quantitative results, highlighting the augmented baseline’s role in advancing PG-Video-LLaVA’s capacity in video understanding.

#### 4.3. Spatial Grounding in Videos

To quantitatively assess PG-Video-LLaVA’s spatial grounding capability, we conducted quantitative evaluations of PGVideo-LLaVA’s spatial grounding capabilities using two benchmarks that are derived from the test set of the VidSTG [48] and HC-STVG [34] datasets. Due to the novelty

Evaluation Metrics Correctness Detail Orientation Contextual Understanding Temporal Understanding Consistency

Model

LLaMA Adapter [9] 2.34 2.43 2.65 2.20 3.02 Video Chat [15] 2.49 2.82 2.92 2.27 3.11 Video-LLaMA [45] 2.29 2.61 2.68 2.17 2.87 Video-ChatGPT [22] 2.51 2.53 2.85 2.32 3.10 PG-Video-LLaVA (7B) 2.73 2.89 3.13 2.44 3.40 PG-Video-LLaVA (13B) 2.86 2.95 3.23 2.53 3.49

- Table 1. Performance benchmarking of video-based conversational models. Comparative performance evaluation of PG-Video-LLaVA against various models using the benchmarking framework from Video-ChatGPT [22]. The metrics include correctness, detail orientation, contextual understanding, temporal understanding, and consistency. The updated assessment pipeline incorporates Vicuna-13b-v1.5 [7] for enhanced reproducibility, replacing GPT-3.5-Turbo. Results indicate that PG-Video-LLaVA achieves favourable performance across all metrics, particularly in contextual and temporal understanding, as compared to foundational models and recent advancements in the field.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Describe this Video.

The video shows a small child sitting on the floor in front of a fireplace, playing with a toy. The child is holding a toy and appears to be playing with it while sitting on the floor.

Video-ChatGPT

The video shows a young child playing with a toy in a living room. The child is sitting on the floor and playing with a toy car, which is placed on a rug. The room is furnished with a couch, a chair, and a TV. The child appears to be enjoying their playtime, and the scene captures a moment of innocent childhood fun.

PG-Video-LLaVA (7B)

PG-Video-LLaVA (13B)

The video shows a young child sitting on the floor in a living room, playing with a toy. The room is furnished with a couch, a chair, and a television. The child is holding a toy in his hand while playing with it. The room appears to be

well-lit, with a fireplace in the background. The child seems to be enjoying himself, as he is laughing and smiling while playing with the toy. Overall, the video captures a joyful moment of a child's life, as he engages in playful activities in a comfortable and cozy living room setting.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Describe this Video.

Video-ChatGPT

The video shows a giraffe performing a trick by standing on a diving board and then jumping into the water.

The video shows a giraffe standing on a diving board and jumping into a pool. The giraffe is seen standing on the diving board, looking down, and then jumping into the pool. The scene takes place outdoors, with a clear blue sky visible in the background. The giraffe appears to be enjoying the water and the experience of jumping into the pool.

PG-Video-LLaVA (7B)

The video features a giraffe standing on a diving board and jumping into a pool. The giraffe is seen swimming in the pool and then jumping out of it. The video is set in a pool area with a blue sky in the background. The giraffe's movements are graceful and fluid, and it appears to be enjoying itself. The video captures the unique and unexpected sight of a giraffe engaging in a human activity, showcasing its incredible agility and ability to adapt to new situations. Overall, the video is a delightful and entertaining spectacle that captures the beauty and grace of the giraffe in an unconventional setting.

PG-Video-LLaVA (13B)

- Figure 3. Qualitative results comparison of Video-ChatGPT vs PG-Video-LLaVA (Ours) Qualitative analysis of video descriptions generated by Video-ChatGPT, PG-Video-LLaVA (7B), and PG-Video-LLaVA (13B) models. The evolution in model performance is evident, with enhancements in the accuracy of information, richness of descriptive detail, and alignment with the video’s context and sequence of events as we move from the baseline Video-ChatGPT to the more advanced PG-Video-LLaVA (13B) model.

of integrating spatial grounding within video-conversational models, we highlight the modular nature of our grounding

pipeline, which can be incorporated with other state-of-theart video conversation models. For the VidSTG dataset,

[Figure 59]

What do you see in this video? In the video, we see a giraffe standing on a diving board, and then it jumps into the water. The giraffe is the main focus of the video, and we see it in various positions throughout the video.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

What do you see in this video? In the video, I see a large statue of a woman holding a torch in her right hand. The statue is located on top of a hill, and it is surrounded by a large body of water. The statue is lit up at night, and it is visible from a distance. The video also shows a boat passing by the statue.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

What do you see in this video? In the video, a young girl is performing a gymnastics routine on a balance beam. She starts by jumping onto the beam and then proceeds to perform various flips and turns. She finishes her routine by jumping off the beam and landing on a mat.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

- Figure 4. Qualitative Results for Video Grounding: Visual representation of the grounding capability of advanced video-conversational capabilities of PG-Video-LLaVA. The highlighted regions in each video frame indicate the model’s ability to identify and spatially locate key subjects mentioned in the textual description, such as the giraffe, the statue, and the gymnast on a balance beam.

alongside alternative methods using the same benchmarks, demonstrating our model’s enhanced ability to accurately answer questions, thereby leading to improved spatial grounding performance.

Model VidSTG [48] HC-STVG [34]

Grounding DINO [20] 25.3 19.5 Video-LLaMA [45] 28.6 26.1 Video-ChatGPT [22] 32.8 20.8 PG-Video-LLaVA (7B) 34.2 28.3 PG-Video-LLaVA (13B) 35.1 27.3

The qualitative results shown in Figure 4 emphasize the model’s refined spatial grounding precision. The accurate overlay of masks on the subjects within the videos confirms the model’s adeptness at correlating textual descriptors with visual elements, a critical aspect of contextual comprehension. This refined ability is crucial for applications that integrate visual data with language, improving the model’s utility in environments that demand rich, interactive visual and linguistic processing.

- Table 2. Performance of PG-Video-LLaVA and other models on spatial grounding task: Evaluated using the VidSTG and HCSTVG benchmarks, the results demonstrate PG-Video-LLaVA’s favorable spatial grounding capabilities, as evidenced by its ability to generate accurate descriptive responses and effectively locate referring expressions within video frames. The table shows the model’s progress, particularly in the 13B version, showcasing its performance among other SoTA video-conversational models.

#### 4.4. Zero-Shot Visual Question Answering

we selectively processed interrogative prompts to assess the grounding accuracy. The model generates descriptive textual responses to these prompts, from which Vicuna-13bv1.5 extracts relevant referring expressions. These expressions are then spatially grounded in the video frames using our grounding pipeline. For the HC-STVG dataset, interrogative prompts are first mined from the text captions using Vicuna and then used similarly to VidSTG prompts.

For PG-Video-LLaVA, zero-shot question-answering (QA) capabilities were evaluated quantitatively using several established open-ended QA datasets: MSRVTT-QA[40], MSVD-QA [39], TGIF-QA [16], and ActivityNet-QA [44]. These datasets are benchmarks for assessing a model’s ability to generate accurate answers without any datasetspecific fine-tuning. We adopted a zero-shot evaluation methodology, utilizing Vicuna-13b-v1.5 to evaluate the model’s understanding and predictive accuracy, with scores

The results shown in Table 2 position PG-Video-LLaVA

Model MSVD-QA [39] MSRVTT-QA [40] TGIF-QA [16] Activity Net-QA [44] Accuracy Score Accuracy Score Accuracy Score Accuracy Score

FrozenBiLM [41] 32.2 – 16.8 – 41.0 – 24.7 – LLaMA Adapter [9] 53.7 3.3 45.6 3.2 54.3 3.3 37.3 3.2 Video LLaMA [45] 48.6 3.2 32.8 2.8 51.4 3.4 27.1 2.9 Video-ChatGPT [22] 62.6 3.6 50.0 3.3 66.5 3.7 40.8 3.3 PG-Video-LLaVA 64.1 3.7 51.6 3.3 66.8 3.8 39.9 3.3

- Table 3. Zeroshot video-based question-answering: Comparison of PG-Video-LLaVA with other video generative models. The latest available models are used for all the approaches and the benchmarks are calculated using open-source Vicuna LLM. PG-Video-LLaVA performs better than the previously proposed video-based conversational methods.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

What is the name of the product in this video?

Without Audio The product in the video is a moisturizing lotion.

Audio Transcript: Everyone loves the way skin authority sunscreens feel when they go on. … Sunscreen moisturizer SPF 30 is a broad spectrum sunscreen plus a daily moisturizer in one.

[Figure 90]

With Audio The product in the video is Sunscreen Moisturizer SPF 30.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Can you suggest a title for this video clip based on what you see and hear?

Audio Transcript: … just how amazing Chat GPT turn out to be and then the improvement in GPT for. So GPT 4 has not surprised me at all in terms of reception there. Chat GPT surprised us a little bit, but I still …. … You know, they're like, oh, it's better than 3.5, but I thought it was going to be better than 3.5. …

Based on the video, I would suggest "The Future of Technology and Communication" as a title for the video clip.

Without Audio

[Figure 98]

Based on the video, the title could be "The Future of AI: A Discussion on GPT-4 and Chat GPT."

With Audio

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

What is Ross complaining about?

Ross is complaining about the taste of the beer he is drinking. He is seen making a face and spitting out the beer, while the other people in the scene watch him.

Without Audio

Audio Transcript: What's wrong, buddy? Someone at work ate my sandwich. Well, what did the police say? My thanksgiving leftover sandwich. I can't believe someone just ate it. Ross it’s just a sandwich.

[Figure 106]

Ross is complaining about someone eating his sandwich at work. He is upset and frustrated about the situation.

With Audio

- Figure 5. Qualitative Results for Including Audio Modality: The figure illustrates the integrated audio processing pipeline that augments video-question answering with audio cues. It provides side-by-side comparisons showing how audio cues offer additional context, leading to a more accurate interpretation of the video content, as seen in the examples above.

assigned on a scale from 1 to 5. Results are presented in Table 3.

In comparison to Video-ChatGPT, PG-Video-LLaVA demonstrates superior performance, surpassing not only the predecessor but also other notable models in the field, such as FrozenBiLM[41] and Video Chat[15]. The results from our evaluations indicate that PG-Video-LLaVA has significantly enhanced its ability to comprehend video content and generate contextually relevant answers, thus establishing a new state-of-the-art in zero-shot VideoQA.

As shown in Figure 4, our method is able to visually ground the key objects in the given video. Improvement in the model’s capability to describe the content in the video is demonstrated in Figure 3. Further, it can be observed that adding the audio modality helps make correct outputs, whereas the model without audio modality fails to capture those details from visual content alone (Figure 5).

### 5. Conclusion

In this work, we introduced PG-Video-LLaVA, a novel video-based conversational model equipped with pixellevel grounding capabilities. PG-Video-LLaVA enhances image-based conversational models by extracting spatiotemporal features essential for comprehensive video understanding. It incorporates filtered audio transcripts to enrich the interpretation of visual scenes where audio cues are pivotal. Additionally, we developed a novel grounding module capable of tracking and generating pixel-level grounding of objects within videos. To promote reproducibility, we propose quantitative benchmarks for videobased conversational models, utilizing the open-sourced Vicuna LLM instead of GPT-3.5, as employed by previous approaches. These benchmarks are specifically designed to evaluate grounding capabilities. In summary, this work represents the first effort to integrate grounding capabilities into video-based LMMs.

### References

- [1] Pyscenedetect. https://github.com/Breakthrough/ PySceneDetect, 2023. 5
- [2] Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. Whisperx: Time-accurate speech transcription of longform audio. arXiv preprint arXiv:2303.00747, 2023. 3, 4, 1
- [3] Hanoona Bangalath, Muhammad Maaz, Muhammad Uzair Khattak, Salman H Khan, and Fahad Shahbaz Khan. Bridging the gap between object and image-level representations for open-vocabulary detection. In NeurIPS, 2022. 2
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, , Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 2
- [5] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang,

- Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 2
- [6] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In ICCV, 2023. 5
- [7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023. 2, 6
- [8] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023. 2
- [9] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Qiao. Llamaadapter v2: Parameter-efficient visual instruction model. arXiv:2304.15010, 2023. 2, 6, 8
- [10] Yuan Gong, Sameer Khurana, Leonid Karlinsky, and James Glass. Whisper-at: Noise-robust automatic speech recognizers are also strong audio event taggers. In Proc. Interspeech 2023, 2023. 3, 4, 1
- [11] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 961–970, 2015. 3, 7
- [12] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 5
- [13] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692,

2023. 2

- [14] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023. 2
- [15] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv:2305.06355, 2023. 1, 2, 6, 9
- [16] Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. TGIF: A New Dataset and Benchmark on Animated GIF Description. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 7, 8
- [17] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In Proceedings of the IEEE/CVF Con-

- ference on Computer Vision and Pattern Recognition, pages 7061–7070, 2023. 2
- [18] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. ArXiv, abs/2304.08485, 2023. 2, 3
- [19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 2

- [20] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv:2303.05499, 2023. 5, 7
- [21] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Internchat: Solving vision-centric tasks by interacting with chatbots beyond language. arXiv preprint

- arXiv:2305.05662, 2023. 2

[22] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models.

- arXiv:2306.05424, 2023. 1, 2, 3, 5, 6, 7, 8

- [23] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. In European Conference on Computer Vision, pages 1–18. Springer, 2022. 2
- [24] OpenAI. Whisper. https://openai.com/research/ whisper, 2022. 4, 1
- [25] OpenAI. Gpt-4v(ision) system card. https://openai.com/ research/gpt-4v-system-card, 2023. 1
- [26] OpenAI. Chatgpt: Large language model for human-style conversation. https://chat.openai.com, 2023. 2
- [27] OpenLMLab. MOSS: Codebase for MOSS Project. https: //github.com/OpenLMLab/MOSS, 2023. 2
- [28] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022. 2
- [29] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 2
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 5
- [31] Hanoona Rasheed, Muhammad Uzair Khattak, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Fine-tuned clip models are efficient video learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6545–6554, 2023. 2
- [32] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Erix Xing, Ming-Hsuan Yang, and Fahad S Khan.

- Glamm: Pixel grounding large multimodal model. arXiv preprint arXiv:2311.03356, 2023. 2
- [33] David Rozenberszki, Or Litany, and Angela Dai. Languagegrounded indoor 3d semantic segmentation in the wild. In European Conference on Computer Vision, pages 125–141. Springer, 2022. 2
- [34] Zongheng Tang, Yue Liao, Si Liu, Guanbin Li, Xiaojie Jin, Hongxu Jiang, Qian Yu, and Dong Xu. Human-centric spatio-temporal video grounding with visual transformers. IEEE Transactions on Circuits and Systems for Video Technology, 2021. 1, 5, 7
- [35] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023. 2
- [36] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [37] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip: A new paradigm for video action recognition. arXiv preprint arXiv:2109.08472, 2021. 2
- [38] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. arXiv preprint arXiv:2308.01907, 2023. 2
- [39] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In ACM Multimedia, 2017. 7, 8
- [40] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language.

2016. 7, 8

- [41] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. In NeurIPS, 2022. 8, 9
- [42] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 2
- [43] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023. 2
- [44] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, pages 9127–9134, 2019. 7, 8
- [45] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv:2306.02858, 2023. 1, 2, 6, 7, 8
- [46] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab,

Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068,

- 2022. 2

- [47] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514,

2023. 5

- [48] Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng Liu, and Lianli Gao. Where does it exist: Spatio-temporal video grounding for multi-form sentences. In CVPR, 2020. 1, 5, 7, 6
- [49] Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, and Bingyi Kang. Bubogpt: Enabling visual grounding in multi-modal llms. arXiv preprint arXiv:2307.08581,

2023. 2, 5, 3

- [50] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2

## PG-Video-LLaVA: Pixel Grounding Large Video-Language Models Supplementary Material

### A. Audio Modality Integration

with a variety of tasks using natural language.

Here, we outline the implementation details of audio modality integration in PG-Video-LLaVA.

USER: <Instruction> <Video-Tokens> The noisy audio transcript of this video is: <Audio-Transcript>

#### A.1. Audio Transcript Filtering

To generate audio transcripts, we first experimented with using the state-of-the-art Whisper [24] directly. However, the obtained transcripts were too noisy, contained hallucinations, and unwanted text such as lyrics from songs. Passing these raw audio transcripts directly to the LLM without any filtering can negatively affect the overall model’s performance. Therefore, a preprocessing method is required to filter out noisy text and keep only the parts of the audio that carry meaningful information.

ASSISTANT:

### B. Visual Grounding: Quantitative Evaluation

#### B.1. Overview

The following steps combining WhisperX[2] and Whisper-AT[10] are used to refine the original Whisper transcripts to be usable as inputs to the video LMM.

We introduce novel benchmarks for quantitatively evaluating conversation-based video spatial grounding, based on two existing spatio-temporal video grounding datasets, VidSTG[48] and HC-STVG[34].

- 1. We first apply VAD-based preliminary filtering to the audio, and then use the Whisper model with Phonemebased forced alignment to get temporally aligned text transcriptions.
- 2. As Whisper is able to identify the language spoken, all non-English speech can be ignored at this point since PG-Video-LLaVA generates responses in English.
- 3. For each sentence segment obtained, slice the original audio at the corresponding timestamps and pass to Whisper-AT to produce audio-tagging output.
- 4. For each sentence segment, consider the top 3 audio classes predicted.

In conversation-based spatial grounding, the objective is to localize interrogative sentences with unknown objects in the given video (e.g. “What is caught by the squatting boy on the floor?” ). Unlike grounding for declarative sentences where the explicit characteristics of objects (e.g. the class “toy” and visual appearance “yellow”) are present within the sentence itself, grounding for interrogative sentences is challenging due to the fact that it can only depend on relationships between the unknown object and other objects (e.g. the action relation “caught by the squatting boy” and spatial relation “on the floor”) (Figure 6). A benchmark based on this task can be regarded as a measure of the sufficient relationship construction and cross-modal relation reasoning ability of the video-language model.

- (a) If “speech” is not among the top 3 predictions, the segment is ignored.
- (b) If P[music] > P[speech] and P[music] − P[speech] > threshold, the segment is ignored (the threshold is set empirically to 1.1).

Interrogative Sentence: What is caught by the squatting boy on the floor? Declarative Sentence: A little boy with a Christmas hat is catching a yellow

Figure 7 shows the effectiveness of our audio transcript preprocessing method in filtering out hallucinations, music, and garbage characters from the raw audio transcript.

toy.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

#### A.2. Integrating Audio Transcript into the LLM

The following prompt template is used when combining the spatiotemporal video features and audio transcript with the user instruction text.

Figure 6. Interrogative vs declarative sentences

To evaluate our model for conversation-based video spatial grounding, we pass interrogative prompts to the model. It then generates descriptive textual responses to these prompts, from which Vicuna-13b-v1.5 extracts relevant referring expressions. These expressions are then passed into the GroundingDINO-based spatial grounding and tracking module. For the obtained object tracks, bounding box IoU

SYSTEM: You are PG-Video-LLaVA, a large vision-language assistant. You are able to understand the video content that the user provides, and assist the user

is calculated by comparing them with the ground truth annotations.

From the two spatiotemporal grounding datasets, to form a spatial-only grounding benchmark, we crop the video in the temporal axis to contain only the segment where the target object is present, and the mean spatial IoU is reported as the metric for comparison.

It should be noted that we evaluate our model in these benchmarks only in the zero-shot setting, without any training on these datasets.

##### 1. Benchmark based on the VidSTG Dataset: Vid-

STG dataset consists of videos paired with multiform sentences (both interrogative and declarative). To form a benchmark to quantitatively evaluate the performance of conversation-based video spatial grounding, we leverage the 5693 video and interrogative sentence pairs in its test set.

##### 2. Benchmark based on HC-STVG Dataset: Un-

like in VidSTG dataset, in HC-STVG dataset contains only declarative form sentences for all of its videos. Therefore interrogative sentences are first generated from the declarative text captions in 3025 samples of the test set using Vicuna-13b-v1.5 model. Then the evaluation is performed in a similar manner to VidSTG.

#### B.2. Generating Interrogative Statements

The original text annotations in the HC-STVG dataset are in the declarative statement format. In order to make our video prompt-based grounding evaluation pipeline, we extract interrogative statements (questions) from these text annotations using Vicuna-13b-v1.5 using the following prompt template. SYSTEM:

You are an intelligent chatbot designed for generating question-answer pairs from sentences.

USER: Your task is to generate a question and answer from the given sentence. The question should start with ’Who’. The question should refer to the subject of the given sentence. The answer should include the subject of the given sentence. Please generate the response in the form of a Python dictionary string with keys ’Q’ for question and ’A’ for answer. Each corresponding value should

be the question and answer text respectively. For example, your response should look like this: {’Q’: ’Your question here...’, ’A’: ’Your answer here...’}. Please note that the generated question and answer should only include information from the given sentence. Please process the following sentence: The man in the suit goes to the man in white and looks at him.

ASSISTANT: {’Q’: ’Who goes to the man in white?’, ’A’:’The man in the suit’}

USER: Please process the following sentence: <DECLARATIVE_STATEMENT>

ASSISTANT:

#### B.3. Extracting Referring Expression Using Vicuna

In the quantitative evaluation, we use the following prompt template with Vicuna-13b-v1.5 to extract the referring expression from the output of the video-based LMM, which is used as the input prompt to the off-the-shelf-grounding module.

SYSTEM:

You are an intelligent chatbot designed for identifying the most relevant subject/object phrases in video-based question-sentence pairs.

USER: Your task is to compare the question with the sentence, and extract the subject or object phrase of the sentence that most accurately answers the given question. The selected phrase should be short and should contain only one noun. The selected phrase can include adjectives that explain the attributes of the subject/object.

Transcript Obtained from Whisper

Transcript Obtained from Whisper Transcript Obtained from Whisper

|Yeah You know that? I didn't think that I had a debt to pay Till the king had take what I left away It was all my fault, you beat it to destiny But I remember you saying that yesterday There was a time when my heart wasn't on the show …YOUR BLESS BUT YOU are a forever Oh to be the best WOO Kennedy I'm not saying it right here YOUR BLESS I don't know what I thought I might say Seems like we never were talking right away Every other minute I'm fuffing my place|
|---|

|1.5-1.1 2.5-1.1 2.5-1.1 2.5-1.1 2.5-1.1 1.5-1.1|
|---|

|Hi, I'm Stephanie Ragnodden and I'm going to show you step 3 of a 4 step process to groom your horse to shine naturally. This time we're going to use a finishing brush with a cocoa fiber horse here mix because it just gives us a little bit more of an edge. …. We're struggling with the oil so we'll put the coat and always cleaning our brush on the curry and we're going to do this from the head all the way to the tail. Thanks for watching.|
|---|

Transcript Obtained from Our Filtering Pipeline Transcript Obtained from Our Filtering Pipeline Transcript Obtained from Our Filtering Pipeline

|Yeah You know that? I didn't think that I had a debt to pay Till the king had take what I left away It was all my fault, you beat it to destiny But I remember you saying that yesterday There was a time when my heart wasn't on the show …YOUR BLESS BUT YOU are a forever Oh to be the best WOO Kennedy I'm not saying it right here YOUR BLESS I don't know what I thought I might say Seems like we never were talking right away Every other minute I'm fuffing my place|
|---|

|1.5-1.1 2.5-1.1 2.5-1.1 2.5-1.1 2.5-1.1 1.5-1.1|
|---|

|Hi, I'm Stephanie Ragnodden and I'm going to show you step 3 of a 4 step process to groom your horse to shine naturally. This time we're going to use a finishing brush with a cocoa fiber horse here mix because it just gives us a little bit more of an edge. …. We're struggling with the oil so we'll put the coat and always cleaning our brush on the curry and we're going to do this from the head all the way to the tail. Thanks for watching.|
|---|

- Figure 7. Filtering the audio transcript: to remove hallucinations (left), music (center), and garbage (right) characters from the raw audio transcript.

Answer: The old man is pointing to the window.

The selected phrase should not exceed 4 words. The selected phrase should not include articles (’a’, ’the’, ’and’). Please generate the response in the form of a Python dictionary string with keys ’OBJECT’, where its value is the extracted phrase in Python string format. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary. For example, your response should look like this: {’OBJECT’: ’green toy’}. Please process the following video-based question-answer pair: Question: who is in front of the guitar at the show? Answer: A woman in a black dress is in front of the guitar on stage.

ASSISTANT:

{’OBJECT’: ’old man’} USER:

Question: who is inside the blue car? Answer: The driver of the blue car.

ASSISTANT:

{’OBJECT’: ’driver’} USER:

Please process the following video-based question-answer pair: Question: <INPUT_TO_VIDEO_LMM> Answer: <OUTPUT_OF_VIDEO_LMM>

ASSISTANT:

#### B.4. Entity Matching with Vicuna

ASSISTANT:

{’OBJECT’: ’woman in black dress’}

As shown in Figure 2, our method employs an LLMpowered entity matching module similar to [49] to match the key phrases in the video-LMM’s output with the object tracks obtained from the grounding and tracking module. We use the same prompt template as [49].

USER:

Question: who points to the window?

### C. Qualitative Results for Visual Grounding

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Describe this video.

The video is a Christmas tree decoration tutorial featuring a woman in a red chair. The woman is seen decorating a Christmas tree with ornaments and lights. The video also features a Christmas tree with lights and ornaments in a living room. The woman is seen decorating the tree while a cat is sitting on the floor. The video is set to the song "I see trees of green red roses too I see them bloom from in you and I think to myself what a wonderful world I see skies of blue"

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Describe this video.

The video is about a man named Sharin who talks about his classic Mercedes-Benz 190. He introduces us to Cherine Raudis, a woman who owns the car. The video is shot in a backyard and the man is sitting at a table with a cup of coffee. The video also shows the car being driven on the road and parked in front of a house.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Briefly describe the video.

The video shows a man in a red jacket and red hat brushing snow off his car in a snowy driveway. He uses a red brush to remove the snow from the top of the car, the sides, and the back windshield. He also uses a scraper to remove ice from the car. The man is careful to make sure that he cleans off all the snow and ice, including the brake lights, to ensure that his car is safe and ready to go.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Describe what you see in this video.

'The video shows Chef Ann Nolan cooking chocolate chip cookies in a Gen Air V2 convection oven. She explains the benefits of using true convection and demonstrates how evenly the cookies cook. The cookies are taken out of the oven and look delicious.'

[Figure 138]

- Figure 8. Qualitative results for video grounding obtained using image tags as the queries to the off-the-shelf grounding module and an entity matching module to match the image tags with the corresponding phrase in the LMM output. (e.g., in the 4th row, the phrase ’Chef Ann Nolan’ is matched with the image tag ’woman’.)

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Who nods his head and turns?

[Figure 145]

Who pulls off the earrings on their right side?

[Figure 146]

A man in a white suit nods his head and turns.

[Figure 147]

A woman wearing a white dress and earrings on her right side is seen removing the earrings.

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Who says a few words to the woman in plaid clothes?

[Figure 154]

Who puts things in the bowls on both sides as he walks?

[Figure 155]

A man in a black suit says a few words to the woman in plaid clothes.

[Figure 156]

A man in a black hat and a purple shirt puts things in the bowls on both sides as he walks.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Who turns around and walks next to the woman with yellow hair?

[Figure 163]

Who pulls the woman to sit down?

[Figure 164]

[Figure 165]

A man in a white lab coat turns around and walks next to a woman with yellow hair.

A man in a suit pulls the woman to sit down.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Who picks up the cup, drinks, puts down the cup and turns to look at the man next to her?

[Figure 171]

[Figure 172]

Who raises his hand and touches his head?

[Figure 173]

[Figure 174]

A woman in a green dress picks up a cup, drinks, puts down the cup, and turns to look at a man next to her.

A man in a red shirt raises his hand and touches his head.

- Figure 9. Qualitative results for video grounding on example videos from the HC-STVG[34] dataset. These results are obtained by using Vicuna with the prompt template in B.3 to extract the referring expression from the LMM output which is then passed to the off-the-shelf grounding module.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Who does the woman in the yellow dress hug?

Who walks to the wooden frame?

[Figure 182]

[Figure 183]

The woman in the yellow dress hugs the man in the blue suit.

A woman walks to the wooden frame in the video.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Who pushes the woman to the table?

Who walks to the door and then turns?

[Figure 191]

[Figure 192]

A man walks to the door and then turns around to face the camera.

A man in a white lab coat pushes the woman to the table.

- Figure 10. Qualitative results for visual grounding on the HC-STVG dataset (failure cases): errors in our model’s output (e.g., bottomleft: our model identifies the woman as a man), incorrect localizations in the off-the-shelf grounding module (e.g., top-left), and incorrect tracking (e.g., top-right, bottom-right) result in these failure cases.

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

What does the adult in white clothes hug?

What does the adult hold outside the room?

[Figure 201]

The adult holds a bottle of champagne outside the room.

[Figure 202]

The adult in white clothes is hugging a small dog.

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

What is beneath the adult on the boat?

Who is in front of the guitar at the show?

[Figure 210]

The boat is on the water, and the man is sitting on the back of the boat.

[Figure 211]

The woman in the black dress is in front of the guitar.

Figure 11. Qualitative results for video grounding on example videos from the VidSTG[48] dataset

### D. Quantitative Evaluations of Video-based Conversation Performance

We leverage the video-based conversation performance benchmarks introduced in Video-ChatGPT[22], while changing the evaluation LLM from GPT-3.5-Turbo to Vicuna-13b-v1.5 model. The prompt templates used with Vicuna are as same as with [22].

Video-based Generative Performance Benchmarking: In this benchmark we continue to use the same test set of 500 samples curated from the ActivityNet-200[11] videos as in [22].

Zero-Shot Question-Answer Evaluation: Following Video-ChatGPT, we perform zero-shot evaluation on four standard open-ended question-answer datasets: MSRVTT[40], MSVD[39], TGIF[16], and ActivityNetQA[44]. No specific training is performed on these datasets, and the evaluation is performed in a zero-shot manner.

