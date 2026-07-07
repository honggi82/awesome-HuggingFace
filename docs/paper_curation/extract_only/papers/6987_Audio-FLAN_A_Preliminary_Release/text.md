# arXiv:2502.16584v2[cs.SD]7Jun2026

## Audio-FLAN: An Instruction-Following Dataset for Unified Audio Understanding and Generation of Speech, Music, and Sound

Liumeng Xuea,b∗, Ziya Zhoua,b∗, Jiahao Pana,b Zixuan Lic, Shuai Fand, Yinghao Mae,b, Sitong Chenga Dongchao Yangf, Haohan Guof, Yujia Xiaof, Xinsheng Wanga Zixuan Shena, Chuanbo Zhua, Xinshen Zhanga, Tianchi Liug Ruibin Yuana,b, Zeyue Tiana,b, Haohe Liub,h Xingjian Dui, Emmanouil Benetosb,e, Ge Zhangb Xu Tanj, Yike Guoa, Wei Xuea

a The Hong Kong University of Science and Technology, b M-A-P c Inner Mongolia University, d Beihang University e Queen Mary University of London, f The Chinese University of Hong Kong g National University of Singapore, h University of Surrey, i University of Rochester, j Independent Researcher

### Abstract

Recent advancements in audio tokenization have significantly enhanced the integration of audio capabilities into large language models (LLMs). However, audio understanding and generation are often treated as distinct tasks, hindering the development of truly unified audio-language models. While instruction tuning has demonstrated remarkable success in improving generalization and zero-shot learning across text and vision, its application to audio remains largely unexplored. A major obstacle is the lack of comprehensive datasets that unify audio understanding and generation. To address this, we introduce Audio-FLAN, a large-scale instruction-tuning dataset covering 80 diverse tasks across speech, music, and sound domains, with over 100 million instances. Audio-FLAN lays the foundation for unified audio-language models that can seamlessly handle both understanding (e.g., transcription, comprehension) and generation (e.g., speech, music, sound) tasks across a wide range of audio domains in a zero-shot manner. The Audio-FLAN dataset is available on HuggingFace 1 and GitHub 2.

- 1https://huggingface.co/HKUSTAudio
- 2https://github.com/lmxue/Audio-FLAN

### 1 Introduction

Recent advances in large language models and multimodal models have highlighted the effectiveness of instruction tuning for broad generalization [Ouyang et al., 2022, Touvron et al., 2023, Achiam et al., 2023]. Instruction-tuned models can generalize to unseen tasks far better than task-specific counterparts. In the text domain, models like FLAN (Finetuned Language Net) [Wei et al., 2021] demonstrate remarkable zero-shot and few-shot capabilities when fine-tuned on diverse instructions. For example, FLAN (137B parameters) was fine-tuned on 60 NLP tasks and outperformed even larger models, like the 175B GPT-3 [Brown et al., 2020], on many unseen tasks. Similarly, LIMA [Zhou et al., 2024], which used only 1,000 curated examples, achieved results preferred over much larger models, showing that minimal high-quality instruction data can significantly improve a model’s ability to follow complex queries. In the vision domain, unified models like Chameleon [Team, 2024] and Janus-Pro 7B [Wu et al., 2024] have demonstrated strong performance by handling both understanding and generation tasks in a single system, outperforming specialized models in image captioning, visual question answering, and image generation. In contrast, the audio domain 3 still lags behind, with audio understanding and generation often treated as separate tasks.

This gap between modalities highlights a critical limitation: audio-language models still lack the unified modeling and generalization capabilities that are now common in NLP and computer vision. Despite the wide variety of audio tasks (such as speech transcription, speaker identification, emotion recognition, sound event recognition, music understanding, and text-to-speech generation), there is no "audio GPT" or "audio foundation model" that can seamlessly switch between understanding and generating audio across speech, music, and audio domains. For example, models like Musilingo [Deng et al., 2023] focus on music understanding, while LTU (Listen, Think, Understand) [Gong et al., 2023b] and Audio-Flamingo [Kong et al., 2024] focus on the audio domain. The SALMONN [Tang et al., 2023] and Qwen-Audio series [Chu et al., 2023] are designed for understanding speech, sound, and music, but lack generation capabilities. On the other hand, UniAudio [Yang et al., 2023] supports audio generation, but it is limited to 11 tasks spanning speech, sound, music, and singing, each with specific task identifiers.

Currently, no audio model exhibits the broad zero-shot generalization seen in text and vision models. Recent benchmarks highlight these limitations. Dynamic-SUPERB [yu Huang et al., 2024], a comprehensive benchmark with 33 speech tasks for speech models, shows that unlike text models, speech models remain confined to narrow tasks. It finds that systems perform well on seen tasks but struggle with unseen tasks, revealing poor zero-shot generalization. Dynamic-SUPERB Phase-2 [Huang et al., 2024], which has expanded to include 180 understanding tasks, reports that while recent models perform well on specific tasks, they struggle with generalization, underscoring the need for more research on developing universal models. Similarly, the MMAU benchmark [Sakshi et al., 2024], which covers speech, environmental sounds, and music, shows that even top models like Gemini-Pro v1.5 [Team et al., 2024] and Qwen2-Audio [Chu et al., 2024] only achieve about 52% accuracy. This stark contrast with text models underscores the underexplored potential of audio-language models for general auditory intelligence. Additionally, the lack of comprehensive evaluation frameworks further hinders progress. AIR-Bench [Yang et al., 2024], the first generative audio-

3In this paper, ‘audio’ refers to two distinct meanings: (a) in a narrower sense, ‘audio’ refers to ‘sound’, which is related to but different from speech and music, often used in the context of ‘speech, music, and audio’; (b) in a broader sense, ‘audio’ encompasses speech, music, and sound, used in the context of ‘text, vision, and audio’.

language comprehension benchmark, reveals significant limitations in current models’ ability to follow instructions across tasks. In summary, audio-language research is still in an early stage, similar to the pre-GPT-3/FLAN era of NLP: while there are task-specific models, there is no unified model with broad, zero-shot capabilities.

A key challenge in the audio domain is the lack of large-scale, diverse instruction-tuning datasets tailored to audio-language tasks. While NLP has benefited from extensive multi-task instruction datasets like Super-NaturalInstructions [Wang et al., 2022a] with 1,616 tasks and vision-language models use resources like LLaVA [Liu et al., 2024] and InstructBLIP [Dai et al., 2023], the audio field lacks comparable datasets in scale or diversity. Some efforts, like GAMA [Ghosh et al., 2024] synthesize an instruction dataset, called CompA-R, for audio reasoning, but they focus mainly on narrow tasks like question-answering and captioning. Other works have used GPT-4 or LLMs to generate instruction data from existing speech corpora, e.g., LTU [Gong et al., 2023b] and LTU-AS [Gong et al., 2023a], but these are fragmented, limited in scope, and often biased by the prompts used. No existing dataset spans the breadth of audio content, including speech, music, and sound, with instructions. In short, the audio domain lacks a “FLAN” equivalent—a consolidated, high-quality instruction dataset to unify myriad audio tasks. This absence of data is a key reason we do not yet have audio models with the generalization of GPT-4 or Chameleon. Even as benchmarks like the Dynamic-SUPERB series and AIR-Bench call for instruction-following audio models, researchers struggle to train such models without a large, diverse training corpus tailored to audio-language understanding and generation.

In this work, we introduce Audio-FLAN, a preliminary attempt to bridge this data gap and enable truly unified audio-language modeling. Audio-FLAN (Preliminary Release) is a large-scale, diverse instruction-tuning dataset for both understanding and generation tasks across speech, music, and audio, constructed by collecting and standardizing nearly all publicly available academic audio datasets into a common instructionbased format. By normalizing the format of these heterogeneous datasets, we provide each audio sample with one or more accompanying instructions (or question/prompt) and the expected output (transcription, description, answer for understanding tasks, or an audio clip for generative tasks). Crucially, Audio-FLAN is designed to support both pre-training and supervised fine-tuning (SFT) of models for unified audio-language tasks. We envision that models trained on Audio-FLAN dataset will be capable of both audio understanding (e.g., transcribing and comprehending audio, answering questions about it) and audio generation (e.g., following instructions to produce speech, music and sounds) within one unified framework. In other words, Audio-FLAN lays the groundwork for an audio equivalent of multimodal foundation models—an audio-language model that can listen, understand, speak, sing and compose in a general way.

To our knowledge, Audio-FLAN is the first comprehensive compilation that combines diverse audio datasets into a single, instruction-driven corpus of considerable scale. It includes approximately 80 tasks and over 100 million instances, significantly surpassing prior efforts in both quantity and diversity. We aim for Audio-FLAN to achieve for audio what FLAN and other instruction-tuned models have accomplished for text—enabling models to generalize across a wide range of audio tasks in a zero-shot manner and follow open-ended instructions related to audio content. The preliminary release of Audio-FLAN is only the beginning: we invite the research community to build on this resource, contribute new tasks (similar to Dynamic-SUPERB Phase-2), and explore unified models for speech, music, and audio. By unifying both

audio understanding and generation, Audio-FLAN paves the way toward foundational models that can hear and generate audio as flexibly and broadly as language models process text.

### 2 Audio-FLAN Dataset Construction

- Figure 1 illustrates the pipeline for constructing the Audio-FLAN dataset. We first collect the publicly released datasets and use their original labels, or manually processed labels, to determine the tasks that can be performed based on the task definitions. Next, instructions are generated and structured using task templates, which guide the format and content of instruction, input, and output. To increase the diversity of the instruction set, we apply a self-instruct-like method [Wang et al., 2023], where the instructions are varied through tools like LLaMA and GPT, which allow for the creation of multiple variations for each task and instance. These varied instructions are then validated to ensure they meet the required standards before being integrated into the dataset.

GPT-assisted Initiation

Task Taxonomy

[Figure 1]

[Figure 2]

[Figure 3]

Varied Instance Record (VIR)

Seed Instruction Pool

Unified Task Template(UTT)

[Figure 4]

Generate the music based on the text and music excerpt as the intro.

- • {"instruction": “Write a complete music based on the description and audio prompt.", "input": "text: bass, jazz, cozy. audio prompt: <|SOA|>Prompt<|EOA|>", "output": "<|SOA|>Music<|EOA|>"}

- • {"instruction": “Turn the text into the matching speech audio please.", "input": "text: The story has a happy ending. We are all pleased to see that.", "output": "<|SOA|>Speech<|EOA|>"}

Please continue the music

given the text description. Please convert the text into the corresponding speech audio.

[Figure 5]

Raw and Augmented Datasets

Synthesize the speech of the given text.

[Figure 6]

Template-Instantiated Record (TIR)

{“instruction”: “Please continue the music given the text description and music.”, “input”: “text: bass, jazz, cozy; music:

[Figure 7]

[Figure 8]

<|SOA|>Prompt<|EOA|>", "output": "<|SOA|>Music<|EOA|>"}

Manual & Automatic Validation

{"instruction": "Please convert the text into the corresponding speech audio.", "input": "text: The story has a happy ending. We are all pleased to see that.", "output": "<|SOA|>Speech<|EOA|>"}

Llama-assisted Variation

a. Manual Processing & Augmentation c. Cross Validation

b. Instruction Variation

Figure 1: Overview pipeline of Audio-FLAN dataset construction.

#### 2.1 Task Category

We classify tasks into Major Tasks and Minor Tasks following a hierarchical structure based on the scope and specificity of the tasks within the broader domains of speech, music, and audio, as shown in Table 1.

- • Major Tasks represent broad categories that encompass a variety of related activities within each domain. For example, in the speech domain, major tasks include Speech Recognition, Speech Generation, and Speech Enhancement, which cover the general areas of recognizing spoken words, generating speech, and improving speech quality, respectively.
- • Minor Tasks are specific subcategories under each major task, providing more focused and detailed areas of work. For example, under Speech Recognition, the minor tasks include Automatic Speech Recognition, Dialect Automatic Speech Recognition, and Phonetic Recognition, each representing a specialized area within the overarching task of recognizing speech. Similarly, under Speech

Generation, tasks like Text to Speech, Voice Conversion, and Speech to Speech Translation address more specific aspects of generating speech.

Table 1: Task category in Audio-FLAN dataset.

Domain Major Task Minor Task Speech

Automatic Speech Recognition Dialect Automatic Speech Recognition Phonetic Recognition

Speech Recognition

Intent Classification Speech to Text Translation

Spoken Language Understanding

Gender Recognition Age Recognition Emotion Recognition Accent Recognition Spoken Paragraph Recognition Language Identification Dialect Identification

Paralinguistic Attribute Recognition

Speaker Verification Speaker Diarization Speaker Extraction Speaker Identification

Speaker Recognition

Speech Caption Speech Caption

Deepfake Detection Vocoder Type Classification Device Recognitin

Speech Detection

Denoising Dereverberation Declipping Speech Bandwidth Extension Signal-to-noise Ratio Estimation

Speech Enhancement

Text to Speech Zero-shot Text to Speech Emotional Text to Speech Zero-shot Emotional Text to Speech Descriptive Speech Synthesis Spontaneous Text to speech Voice Conversion Emotion Conversion Speech to Speech Translation

Speech Generation

Total 8 34

Music

Key Detection Scale Recognition Music Tagging Genre Classification Emotion Classification Pitch Classification Instrument Classification Vocal Technique Classification Instrumental Technique Classification Artist Identification

Global MIR

Beat Tracking Chord Estimation Progression Extraction

Sequential MIR

Beat-level Instruments Recognition Beat-level Pitch Estimation

Single Music Reasoning

Tempo Comparison Instrument Comparison Key Comparison Instrumental Technique Comparison Emotion Comparison

Multiple Music Reasoning

Music Caption Music Caption Music Separation

Melody Extraction Text-guided Source Separation

Text-to-music Generation Text-guided Music Continuation Lyrics-to-song Generation Singing Voice Synthesis Singing Voice Conversion

Music Generation

- Total 7 28 Audio

Sound Event Sequence Recognition Sound Event Recognition Sound Event Detection Acoustic Scene Classification

Audio Event Recognition

Audio Caption Audio Caption Audio Advanced Understanding Sound Event Understanding

Deepfake Audio Detection Voice Activity Detection

Audio Detection

Speech, Silence, Music and Noise Classification

Audio Classification

Speech Nonspeech Detection Audio Enhancement

Audio Inpainting Audio Super-resolution

Text-guided Audio Source Separation Label-querying Sound Extraction Audio-querying Sound Extraction

Audio Separation

Text-guided Audio Generation Time-grounded Text-to-audio Generation Audio Continuation

Audio Generation

- Total 8 18 Total 23 80

This hierarchical approach provides a clear structure that allows for easy navigation of the tasks. By categorizing tasks into major and minor tasks, it is easier to understand the broad objectives as well as the specific challenges and techniques involved in each sub-area. Besides, this classification system allows researchers and practitioners to target specific areas of interest. Furthermore, the system is flexible, accommodating new tasks as the fields evolve. New minor tasks can be added under existing major tasks, or new major tasks can be created as technology advances, ensuring that the classification system can adapt to future developments.

The Audio-FLAN dataset introduces time-sequential tasks that have been underexplored in previous research, particularly in the textual domain, as time sequences are a distinctive feature of the audio domain. These include tasks like Melody Extraction, and Pitch Estimation (with timestamps) in the music domain, as well as Sound Event Sequence Recognition and Sound Event Detection (with timestamps) in the audio domain. These tasks require processing entire audio sequences or segments, highlighting the importance of time-based analysis. In the speech domain, tasks like Spoken Paragraph Recognition further emphasize the role of time sequences, as the model must compare recordings and analyze linguistic content aligned over time.

Additionally, text-based LLMs are often praised for their reasoning capabilities in tackling complex tasks that involve interdependent results. In the music domain, we introduce reasoning tasks where models must first localize a time segment based on instructions and then perform estimations to generate precise answers. For example, Beat-level Pitch Estimation and Beat-level Instrument Recognition (under Single Music Reasoning) require models to interpret musical elements at specific time points, while Tempo/Key/Instrument/Emotion Comparison (under Multiple Music Reasoning) involves comparing musical features over time. These tasks push the limits of model generalization across complex, time-based data, positioning Audio-FLAN as a unique resource for developing unified models capable of processing time-sensitive audio across speech, music, and audio.

In conclusion, the hierarchical classification system effectively organizes each domain into high-level tasks (Major Tasks) and more specific subtasks (Minor Tasks), providing a clear structure. With 23 major tasks

and 80 minor tasks, the dataset covers a wide range of understanding and generation tasks across speech, music, and audio, underscoring the depth of research and application in these fields. Notably, the AudioFLAN dataset is the first instruction-tuning dataset to incorporate tasks from speech, music, and audio, addressing both generation and understanding tasks. This contribution fosters the development of unified audio-language models with generalization capabilities similar to those in the NLP and computer vision domains.

#### 2.2 Dataset Processing

Our goal is to develop a large and diverse instruction dataset by aggregating tasks from various domains and applications. Building such an extensive instruction dataset from scratch would be highly resource-intensive and time-consuming. To mitigate this challenge, we leverage existing audio datasets from the research community, transforming them into an instructional format. This approach capitalizes on the wealth of labeled data that is already available or manually processed, allowing us to repurpose datasets for broader applications. Specifically, we aggregate over 52 datasets that are either publicly accessible or can be obtained upon request. The datasets associated with each task are listed in Table 3.

In the speech, music and audio domains, many tasks depend heavily on pre-labeled data, such as genre labels, speech annotations, or musical characteristics. For instance, tasks like Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) rely on paired text and speech data, while Emotion Recognition and Gender Recognition tasks in speech utilize emotion and gender labels, respectively. In the Music domain, tasks like Genre Classification and Emotion Classification require labeled music data with genre or emotion tags, and Pitch Classification and Instrument Classification rely on instrument-specific annotations. However, there are several tasks for which suitable labeled datasets are not readily available or require additional processing. For example, tasks such as Audio Inpainting or Music Generation often lack directly available labels or training data that match the specific needs of these tasks. In these cases, manual processing is required to create the necessary data.

In the speech domain, for Speech Enhancement tasks, data simulation techniques generate task-specific datasets from clean speech corpora. For Denoising, noisy-clean pairs are created by adding noise to clean speech samples. Dereverberation involves generating reverberant-clean pairs by convolving clean speech with real or simulated room impulse responses. In the Declipping task, clean speech is randomly clipped for model input. For Speech Bandwidth Extension, high-sample-rate speech is downsampled to teach the model how to recover high-quality speech from lower-quality input. In Speaker Recognition, Speaker Extraction creates datasets by mixing clean speech from multiple speakers and providing reference speech for the target speaker.

Similarly, the Music Generation tasks in the music domain, such as for the Text-guided Music Continuation or Lyrics-to-song Generation, manually processed data might be needed to create the text-to-music pairs. This could involve taking existing music pieces and pairing them with relevant textual descriptions, or generating new musical content based on textual input using music generation models. In cases where music data is not paired with lyrics, data augmentation techniques might be used, where new synthetic music tracks are generated by modifying or extending the existing ones to suit the task.

In the audio domain, the Audio Generation tasks such as Audio Inpainting, the data processing involves selecting clean audio samples, cutting them to create gaps, and preparing the dataset for further use in reconstructing the missing segments. In Audio Super-resolution, the process includes downsampling highquality audio to a lower resolution and then using the downsampled version to recreate the original highresolution audio. These processing steps facilitate the generation of suitable datasets for these tasks.

These cases highlight the flexibility and adaptability of existing datasets in the speech, music, and audio domains, where manual dataset processing and augmentation are crucial for handling tasks with limited labeled data or where the required labels do not exist. By applying these dataset processing techniques, we can ensure that tasks with scarce resources are still effectively addressed, broadening the applicability of existing datasets to more diverse machine learning applications. Furthermore, the Audio-FLAN dataset is continuously being expanded and processed to cover additional tasks. We also invite all interested researchers and practitioners to contribute to the ongoing development of the Audio-FLAN instruction tuning dataset, enhancing its scope and utility for the community.

#### 2.3 Task Instruction Template

The instruction data we aim to generate consists of a collection of instructions {Ii}, each describing a specific task i in natural language. For each task i, there are ni ≥ 1 input-output pairs {(Xt,i,Yt,i)}n

t=1. Once the tasks to be covered by the dataset are determined, we process the data into three core components: instruction, input, and output, all formatted in JSONL (JSON Lines) format. The instruction serves as a concise description of the task, guiding the model on the expected input and the type of output to generate. For tasks that involve understanding, the output is text, while for tasks focused on generation, the output is typically audio. The input can be audio, text, or a combination of both, depending on the task. Formally, given this structured data, a model M is expected to generate the appropriate output based on the task instruction and the corresponding input: M(Ii,Xt,i) = Yt,i, for i ∈ {1,...,ni}.

i

In the speech domain, the task of Speech-to-Text Translation involves both text and audio as input (e.g., an audio recording of speech and the corresponding transcription in target language), and the output is text, which is the translated text in a different language. In the music domain, the task of Text-guided Music Generation uses a combination of text and audio as input (e.g., a description of the type of music and a short melody clip), and the output is audio, which is a generated music track that matches the input description and melody. In the audio domain, tasks like Audio Super-resolution can take a combination of low-resolution audio and textual description of the expected quality improvements as input, and the output is high-resolution audio that enhances the quality of the input signal.

To generate the task instructions {Ii}, we initially employ template-based instructions. These instructions are human-written, task-specific descriptions that explicitly define the task. For example, the instruction for the Speech-to-Text Translation task could be "Please translate the speech into the text in Chinese.". For Text-guided Music Generation, the instruction might be "Please continue the audio music prompt based on the given text description." The instruction for the Audio Super-resolution task can be "Please increase the resolution of the given audio signal to 32K Hz". Here are the three task instruction templates:

Speech-to-Text Translation

{ "instruction": "Please translate the speech into the text in English.", "input": "<|SOA|>Audio_ID<|EOA|>", "output": "Nevertheless, there are many distinctive ways of drinking coffee around the world that are worth experiencing." }

Text-guided Music Continuation

{ "instruction": Please continue the audio music prompt based on the given text description", "input": "This is a Carnatic music piece set in the atana raga. It follows the 5/8 meter and is composed in the khandaChapu taala. The lead instrument featured in this performance is vocal, accompanied by Mridangam. The kalai of this composition is 1. \n audio prompt: <|SOA|>Audio_ID<|EOA|>", "output": "audio: <|SOA|Audio_ID<|EOA|>" }

Sound Super-resolution

{ "instruction": "Please increase the resolution of the given audio signal to 32k Hz.", "input": "audio: <|SOA|>Audio_ID<|EOA|>." "output": "<|SOA|>Audio_ID<|EOA|>", }

We include <SOA> to mark the start of audio, and <EOA> to signify the end of audio. When the input contains multiple values, they are separated by \n. Note that the JSONL format files contain not only the instruction, input, and output, but also other relevant fields such as uuid, split, task_type, and domain. The complete JSON file content can be found in Appendix A.3. These task-specific templates serve as foundational structures, which can later be refined and expanded upon to better suit a wide range of tasks across different domains. This method ensures that the instructions are both clear and aligned with the model’s input-output expectations.

#### 2.4 Instruction Variation

While fixed, template-based instructions provide consistency in task execution, they inherently constrain flexibility and creativity. This rigidity can hinder the model’s ability to adapt to diverse and nuanced task descriptions. To mitigate these limitations and enhance the diversity and creativity of the instructions, we introduce an approach that expands template-based instructions into a broader set of variations using advanced language models, like LLaMA [Touvron et al., 2023]. By leveraging the generative power of these models, we can produce multiple distinct variations for each task instruction template, thereby augmenting the model’s capacity to handle a wide array of task descriptions.

The process of instruction variation follows a three-step pipeline, inspired by the self-instruct approach [Wang et al., 2023], designed to systematically enhance instruction diversity. These steps include: (1) initializing the variation seed pool, (2) generating new diverse instructions, and (3) validating the generated instructions.

In the first step, we begin by generating five new instruction examples for each task using GPT-4o, which serves as the initial "seed" pool. These initial variations form the basis for subsequent instruction generation. In the second step, we utilize the Llama-3.1-70B-Instruct model to generate instruction variation, drawing

from the seed pool. Llama-3.1-70B-Instruct allows for the generation of diverse and contextually varied instructions, along with modifying or adding prefixes within the input and output fields based on the specific characteristics of the task. This process allows for further customization of task instructions that are both rich in variation and contextually appropriate.

The final step involves rigorous validation of the generated instructions to ensure their integrity and quality. Specifically, we verify that the audio ID remains consistent with the original task instance and confirm that the JSONL format adheres to the required structure. Any variations that exhibit formatting errors, such as incorrect JSONL syntax or mismatched audio IDs, are identified and excluded from the pool. Any instructions deemed invalid are flagged for regeneration, and if no suitable variation can be generated by the model, manual intervention is employed to address the issue. This ensures that both the quantity and quality of the variations are maintained. Valid instructions are then reintegrated into the task pool for use in generating further variations.

This iterative process promotes a dynamic and evolving pool of task instructions, effectively maximizing their diversity. As a result, the model becomes more adept at handling a wide range of task descriptions, ultimately improving its overall performance and generalization ability across diverse use cases. The prompt used to produce various instructions by GPT-4 and LLaMA is provided in Appendix A.4. Specific examples of the instruction template and generated instruction variations are shown in Appendix A.5.

### 3 Audio-FLAN Dataset

- Figure 2 illustrates the structure of the Audio-FLAN dataset, which spans a diverse range of tasks and instances. It is organized into 23 major tasks and 80 minor tasks from 52 released datasets4, totaling 108.5M instances. These tasks are divided into two primary categories: understanding and generation.

- • Understanding: This category consists of 16 major tasks and 51 minor tasks with 51 open-sourced datasets, amounting to 62.44M instances. The understanding tasks are further divided into three domains:

- – Speech: 6 major tasks and 20 minor tasks, with 24 datasets and 57.42M instances.
- – Music: 5 major tasks and 21 minor tasks, with 19 datasets and 1.46M instances.
- – Audio: 5 major tasks and 10 minor tasks, with 8 datasets and 3.56M instances.

- • Generation: This category includes 7 major tasks and 29 minor tasks with 31 publicly available datasets, with a total of 46.06M instances. The generation tasks are categorized as follows:

- – Speech: 2 major tasks, 14 minor tasks, with 12 datasets and 43M instances.
- – Music: 2 major tasks, 7 minor tasks, with 13 datasets and 0.71M instances.
- – Audio: 3 major tasks, 8 minor tasks, with 6 datasets and 2.35M instances.

Overall, the Audio-FLAN dataset provides a comprehensive and balanced set of tasks across the speech, music, and audio domains, supporting both understanding and generation tasks in the audio field. The

4Each dataset may correspond to one or more tasks. The 52 datasets in Audio-FLAN represent the unique datasets after deduplication. The total number of data points for different tasks can exceed 52.

Audio-FLAN dataset fills a critical gap in the audio research community, offering the first large-scale, instruction-driven corpus for unified audio-language models.

[Figure 9]

[Figure 10]

[Figure 11]

Speech/Music/Audio Generation

Audio-Flan Dataset Overview

Convert the given text into speech:

Extract the melody from the provided

Could you suggest the most suitable

诶，说不定，说不定我们直接以后用那种机器 语言交流哟。

audio.

audio for the text provided? Bark.

[Figure 12]

[Figure 13]

[Figure 14]

23 Major Tasks

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

80 Minor Tasks

Chinese Speech

:

: Melody STEM 1

:

Melody STEM 2

108M Instances 52 Datasets

Could you generate an audio clip that describes Bird, specifically between 1.000-11.000 seconds?

Could you please change the source music to match the target voice.

Would you please translate the

provided speech into Hungarian?

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Hungarian Speech

: English

[Figure 29]

[Figure 30]

[Figure 31]

--Generation--

Speech

: Source Music

Target Voice

:

7 Major Tasks 29 Minor Tasks 46M Instances 31 Datasets

Transform the provided text into a

Continue the given music based on the

I need help completing the audio file. There's a gap after the third second. Can you assist with that?

happy emotional speech while maintaining the given speaker's timbre: 我老家在北京，哇塞！太精彩了。

text: Jingju aria, featuring the vocal and rhythm pattern of xipi, yuanban and sanban.

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

Chinese Speech Speaker

Completed clip

:

:

:

Diverse instructions Flexible inputs and outputs

[Figure 42]

[Figure 43]

Speech/Music/Audio Understanding

Complex sequential

[Figure 44]

Can you summarize this audio recording into a written format?

Check if these two speech samples are from the same chapter?

Which of the two has a faster tempo?

reasoning

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

--Understanding--

: People are making sounds directed by a director.

: Yes. Speech 1 Speech 2

: The second. Music 1 Music 2

16 Major Tasks

51 Minor Tasks

Please transcribe the content of the

Can you detect the instruments in the

Can you identify the start and end seconds of the Frog sound in the given audio?

62M Instances 51 Datasets

provided audio into written format?

first 10 beats of the provided music?

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

: It is linked to the row over proposed changes at Scottish Ballet.

:

- beat1: Drums, Cello, Double Bass;
- beat2: Trombone, Drums, Cello ……

: The Frog sound duration is from 10.000 to 20.000

Figure 2: Overview of Audio-FLAN dataset.

#### 3.1 Statistics of Task

The Audio-FLAN dataset, spanning across the speech, music, and audio domains, is summarized in Table 2. The dataset consists of 23 major tasks and 80 minor tasks across these domains, totaling 108.5M instances. These tasks cover a wide range of applications and modalities, integrating both understanding and generation tasks across various domains. The dataset’s diversity is further enhanced by the variety of input-output formats, including audio, text, and multimodal combinations such as audio and text, allowing it to represent complex and realistic scenarios.

Speech Domain: The Speech domain encompasses 8 major tasks, including Speech Recognition, Speech Generation, and Paralinguistic Attribute Recognition, addressing both understanding and generation tasks. The Speech domain includes 34 minor tasks, with a total of 100.42M instances, showcasing a comprehensive and diverse task representation. Notably, tasks such as Speech Enhancement and Speech Generation focus on generation tasks, while tasks like Speech Recognition and Speaker Recognition are geared toward understanding tasks. The large number of instances in this domain provides a rich dataset for training models, enhancing their ability to generalize across a wide range of speech-related tasks. This abundance of data enables models to learn robust representations, improving their performance and versatility when tackling unseen tasks in the Speech domain.

###### Domain Major Task # Minor Task # Instances Input/Output U/G

Speech Recognition 3 12.05M audio/text U Spoken Language Understanding 2 26.25M audio/text U Paralinguistic Attribute Recognition 7 16.47M audio/text U Speaker Recognition 4 0.73M audio/text U Speech Caption 1 0.35M audio/text U

Speech

Speech Detection 3 1.57M audio/text U Speech Enhancement 5 1.48M audio/audio G

Speech Generation 9 41.52M (audio, text)/audio G Total 8 34 100.42M - -

Global MIR 10 0.34M audio/text U Sequential MIR 3 0.43M audio/text U

Single Music Reasoning 2 95.86K audio/text U Multiple Music Reasoning 5 0.57M audio/text U

Music

Music Caption 1 28.21K audio/text U Music Separation 2 40.26K audio/audio G Music Generation 5 0.67M (audio, text)/audio G

- Total 7 28 2.17M - -

Audio

Audio Event Recognition 4 1.30M audio/text U Audio Caption 1 0.82M audio/text U Audio Advanced Understanding 1 10K audio/text U

Audio Detection 2 1.08M audio/text U Audio Classification 2 0.38M audio/text U Audio Enhancement 2 0.15M audio/audio G

Audio Separation 3 0.89M audio/audio G Audio Generation 3 1.31M (audio, text)/audio G

- Total 8 18 5.91M - Total 23 80 108.5M - -

Table 2: Detailed information of tasks and instances in Audio-FLAN. "U/G" indicates whether the task is for understanding (U) or generation (G). If the output is audio, it is classified as generation; otherwise, it is understanding.

Music Domain: The Music domain features 7 major tasks, covering various music-related applications such as Global MIR (Music Information Retrieval), Music Generation, and Text-guided Music Generation. Both understanding tasks (e.g., genre classification, emotion recognition) and generative tasks (e.g., music composition from text descriptions) are included. With 28 minor tasks and over 2.17 million instances, the Music domain excels in multi-modal tasks, such as Text-guided Music Generation, where input combinations of text descriptions and audio prompts are used. The inclusion of music generation tasks involving multimodal inputs enhances the flexibility and capability of the unified model to generate and comprehend music in diverse ways. The variety in input-output combinations fosters a more comprehensive understanding of

music, making the model highly adaptable and capable of handling both music-related understanding and generation tasks seamlessly.

Audio Domain: The Audio domain includes 8 major tasks, such as Audio Event Recognition, Audio Generation, and Audio Separation, along with 18 minor tasks and 5.91 million instances. The tasks span a broad range of applications, from sound classification to audio enhancement and separation. Notably, the Audio domain includes tasks such as Audio Generation and Audio Super-resolution, which play a key role in advancing the field of audio processing. The diversity of tasks in this domain enhances the model’s ability to understand and generate a wide variety of audio content, further enriching the overall capabilities of the unified audio-language model.

The Audio-FLAN dataset makes a significant contribution to the development of unified models that can both understand and generate audio across multiple domains, including speech, music, and audio. By integrating a diverse set of tasks, the dataset ensures that the models can handle a broad spectrum of real-world audio applications. The varying number of instances across different tasks in the dataset provides a rich foundation for training models. Tasks with larger datasets, such as those in the speech domain, provide ample data for the model to develop a robust understanding of common patterns and features. This helps models generalize well across various tasks, improving their performance and robustness in real-world applications. The variety in instance sizes ensures that the model can remain adaptable and flexible, capable of learning from both high- and low-representation tasks, which is crucial for tasks that are less represented.

While the dataset is highly diverse, it is worth noting that the data distribution across domains is not perfectly balanced. The speech domain, with its larger number of instances, naturally provides more data for training compared to the music and audio domains. We are committed to continuously updating and expanding the Audio-FLAN dataset to include more tasks, domains, and instances. We also encourage the community to contribute by adding new tasks and improving the dataset. By working together, we can build a more comprehensive resource that further advances the development of unified audio-language models and benefits the broader research community.

#### 3.2 Distribution of Audio Attributes

Each subdomain in the audio field encompasses a wide range of attributes. Specifically, the speech domain captures semantic content, speaker identity, and critical paralinguistic features such as emotion, language, accent, age, and more. The music domain contains a variety of musical attributes, including different instruments, timbres, techniques, and structures. Meanwhile, the audio domain covers diverse sounds, including events, animals, scenes, and even speech or music. To explore the different audio attributes in the Audio-FLAN dataset, we analyze the instance distribution of tasks related to these attributes across the speech, music, and audio domains, as shown in Figure 3.

Speech Domain: As shown in Figure 3 (a), in the speech domain, the most prominent features are content (35.5%) and language (32.1%). content-related tasks, like Automatic Speech Recognition (ASR), focus on transcribing spoken language into text, while language-related tasks, such as Language Identification and Speech to Text Translation, handle the translation and identification of speech across languages.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Figure 3: Distribution of audio attributes in (a) speech domain, (b) music domain, and (c) audio domain.

Additional tasks in the speech domain cover features like gender (8.8%), which identifies the speaker’s gender, and age (5.7%), dialect (5.5%), and distortion (4.2%) tasks, such as Denoising and Dereverberation, which improve speech quality. Smaller, yet significant contributions come from tasks related to emotion, accent, and device (1.1%), contributing to a more nuanced understanding of speech signals.

Music Domain: As shown in Figure 3 (b), the music domain’s most prominent features are instrumental (17.6%) and timbre (12.9%). instrumental tasks, like Instrument Classification and Beat-level Instrument Recognition, focus on identifying and analyzing different musical instruments. timbre is related to the tonal quality of sound, and tasks like Singing Voice Conversion capture the unique characteristics of sound sources.

The domain also includes ethnomusicology (12.3%), which helps the model understand diverse cultural music, and tasks like Text-to-Music Generation and Text-guided Music Continuation. vocals (19.4%) and melody (5.3%) tasks like Vocal Technique Classification and Melody Extraction focus on analyzing vocal and melodic elements in music. Additional tasks cover pitch (5.1%), key (4.9%), and chord (2.2%), focusing on musical structure and harmony.

Audio Domain: As shown in Figure 3 (c), the audio domain is dominated by scene (33.4%), which represents environmental sounds, aiding in contextualizing audio. Tasks like Acoustic Scene Classification categorize different environments based on their audio characteristics. event (22.2%) and speech (20.3%) features involve tasks like Sound Event Recognition and Speech Detection, which identify specific events and speech elements in general soundscapes.

Additionally, the others category (24.1%) includes music (28.3%), object (26.1%), and human (25.3%) features, covering tasks like Audio Event Detection, Audio Source Separation, and Speech and Non-speech Detection, providing a comprehensive approach to general audio processing and recognition.

It is important to note that each instance may contain multiple features. As a result, the statistics presented reflect the frequency of feature occurrences rather than the absolute count of instances associated with each feature. This distribution highlights the rich diversity of attributes within both the speech, music and audio domains, encompassing foundational tasks such as speech recognition and speaker identification, as

well as more specialized areas like noise reduction, environmental sound recognition, and music analysis. The broad range of features and tasks in these domains supports the development of unified models that can be generalized across various audio-language tasks. This diversity enables models to adapt to a wide variety of contexts, enhancing their zero-shot generalization capabilities across different types of audio with diverse attributes.

### 4 Conclusion and Discussion

The Audio-FLAN dataset represents a groundbreaking contribution to the audio domain by enabling instruction-tuning for both understanding and generation tasks across the speech, music, and audio domains. This pioneering dataset consists of 23 major tasks and 80 minor tasks, with 16 major tasks dedicated to understanding and 7 major tasks focused on generation, totaling 108.5 million instances. By covering a wide array of tasks from speech recognition and emotion detection to music generation and audio event recognition, the Audio-FLAN dataset provides a comprehensive foundation for developing unified models that can handle both understanding and generation across multiple audio domains. This dataset is designed to support instruction-tuning, empowering models to follow complex audio instructions with minimal task-specific data. It paves the way for zero-shot generalization, enabling models to perform well on unseen tasks within and across domains, much like the advancements seen in text and vision models.

The Audio-FLAN dataset, while a major step towards unifying understanding and generation tasks across the speech, music, and audio domains, exhibits an imbalance in instance distribution. Understanding tasks, particularly in the speech domain, dominate the dataset, benefiting from well-established datasets and easier labeling. In contrast, generation tasks, such as text-to-audio or music generation, are more complex and less represented. This imbalance results in a greater number of instances in the speech domain, while the music and audio domains have fewer. This skew may lead to models being biased toward understanding tasks, potentially impacting their generalization to generation tasks or underrepresented domains.

Future work should focus on balancing the distribution of tasks across domains, ensuring a more even representation between understanding and generation tasks, especially in the music and audio domains. Additionally, expanding the dataset to include more tasks and incorporating additional datasets will strengthen the audio domain’s instruction-tuning capabilities, enhancing the development of unified models that can handle both understanding and generation tasks with improved zero-shot performance. Furthermore, integrating conversational data will be crucial for equipping models with the ability to engage in dynamic, realtime dialogue, broadening the dataset’s applicability to intelligent virtual agents and multimodal interaction systems.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Tosiron Adegbija. jazznet: A dataset of fundamental piano patterns for music audio machine learning research. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Adaeze Adigwe, Noé Tits, Kevin El Haddad, Sarah Ostadabbas, and Thierry Dutoit. The emotional voices database: Towards controlling the emotion dimension in voice generation systems. arXiv preprint arXiv:1806.09514, 2018.

Andrea Agostinelli, Timo I Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023.

AI-Hobbyist. Genshin datasets. https://github.com/AI-Hobbyist/Genshin_Datasets,

- 2024a. Accessed: 2025-03-19.

AI-Hobbyist. Starrail datasets. https://github.com/AI-Hobbyist/StarRail_Datasets,

- 2024b. Accessed: 2025-03-19.

Akshay Anantapadmanabhan, Ashwin Bellur, and Hema A Murthy. Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization. In 2013 IEEE international conference on acoustics, speech and signal processing, pages 181–185. IEEE, 2013.

Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus. arXiv preprint arXiv:1912.06670, 2019.

Evelina Bakhturina, Vitaly Lavrukhin, Boris Ginsburg, and Yang Zhang. Hi-Fi Multi-Speaker English TTS

Dataset. arXiv preprint arXiv:2104.01497, 2021. Ltd Beijing DataTang Technology Co. aidatatang 200zh: A free chinese mandarin speech corpus, n.d. Rachel M Bittner, Justin Salamon, Mike Tierney, Matthias Mauch, Chris Cannam, and Juan Pablo Bello.

Medleydb: A multitrack dataset for annotation-intensive mir research. In ISMIR, volume 14, pages 155–160, 2014.

Rachel M Bittner, Katherine Pasalo, Juan José Bosch, Gabriel Meseguer-Brocal, and David Rubinstein. vocadito: A dataset of solo vocals with f_0, note, and lyric annotations. arXiv preprint arXiv:2110.05580, 2021.

Dawn AA Black, Ma Li, and Mi Tian. Automatic identification of emotional cues in chinese opera singing. ICMPC, Seoul, South Korea, 2014.

Dmitry Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter, and Xavier Serra. The mtg-jamendo dataset for automatic music tagging. In Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019), Long Beach, CA, United States, 2019. URL http: //hdl.handle.net/10230/42015.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In 2017 20th conference of the oriental chapter of the international coordinating committee on speech databases and speech I/O systems and assessment (O-COCOSDA), pages 1–5. IEEE, 2017.

Rafael Caro Repetto. The musical dimension of chinese traditional theatre: An analysis from computer aided musicology. PhD thesis, Universitat Pompeu Fabra, 2018.

Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 721–725. IEEE, 2020.

Soonbeom Choi, Wonil Kim, Saebyul Park, Sangeon Yong, and Juhan Nam. Children’s song dataset for singing voice research. In International Society for Music Information Retrieval Conference (ISMIR), volume 4, 2020.

Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805. IEEE, 2023.

Joris Cosentino, Manuel Pariente, Samuele Cornell, Antoine Deleforge, and Emmanuel Vincent. Librimix: An open-source dataset for generalizable speech separation. arXiv preprint arXiv:2005.11262, 2020.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.

Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840, 2016.

Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. FMA: A dataset for music analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR), 2017. URL https://arxiv.org/abs/1612.01840.

Zihao Deng, Yinghao Ma, Yudong Liu, Rongchen Guo, Ge Zhang, Wenhu Chen, Wenhao Huang, and Emmanouil Benetos. Musilingo: Bridging music and text with pre-trained language models for music captioning and query response. arXiv preprint arXiv:2309.08730, 2023.

Jiayu Du, Xingyu Na, Xuechen Liu, and Hui Bu. Aishell-2: Transforming mandarin asr research into industrial scale. arXiv preprint arXiv:1808.10583, 2018.

Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Mohammad Norouzi, Douglas Eck, and Karen Simonyan. Neural audio synthesis of musical notes with wavenet autoencoders. In International Conference on Machine Learning, pages 1068–1077. PMLR, 2017.

Frederic Font, Gerard Roma, and Xavier Serra. Freesound technical demo. In Proceedings of the 21st ACM international conference on Multimedia, pages 411–412, 2013.

Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In Proc. IEEE ICASSP 2017, New Orleans, LA, 2017.

Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Kiran Reddy Evuru, Utkarsh Tyagi, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha. Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities. arXiv preprint arXiv:2406.11768, 2024.

Yuan Gong, Alexander H Liu, Hongyin Luo, Leonid Karlinsky, and James Glass. Joint audio and speech understanding. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE, 2023a.

Yuan Gong, Hongyin Luo, Alexander H Liu, Leonid Karlinsky, and James Glass. Listen, think, and understand. arXiv preprint arXiv:2305.10790, 2023b.

Swapnil Gupta, Ajay Srinivasamurthy, Manoj Kumar, Hema A Murthy, and Xavier Serra. Discovery of syllabic percussion patterns in tabla solo recordings. In ISMIR, pages 385–391, 2015.

Toni Heittola, Annamaria Mesaros, and Tuomas Virtanen. Acoustic scene classification in dcase 2020 challenge: generalization across devices and low complexity solutions. arXiv preprint arXiv:2005.14623, 2020.

Chien-yu Huang, Wei-Chih Chen, Shu-wen Yang, Andy T Liu, Chen-An Li, Yu-Xiang Lin, Wei-Cheng Tseng, Anuj Diwan, Yi-Jen Shih, Jiatong Shi, et al. Dynamic-superb phase-2: A collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks. arXiv preprint arXiv:2411.05361, 2024.

Rongjie Huang, Feiyang Chen, Yi Ren, Jinglin Liu, Chenye Cui, and Zhou Zhao. Multi-singer: Fast multisinger singing voice vocoder with a large-scale corpus. In Proceedings of the 29th ACM International Conference on Multimedia, pages 3945–3954, 2021.

Keith Ito and Linda Johnson. The lj speech dataset. https://keithito.com/ LJ-Speech-Dataset/, 2017.

Chang-Bin Jeon, Hyeongi Moon, Keunwoo Choi, Ben Sangbae Chon, and Kyogu Lee. Medleyvox: An evaluation dataset for multiple singing voices separation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Ye Jia, Michelle Tadmor Ramanovich, Quan Wang, and Heiga Zen. Cvss corpus and massively multilingual speech-to-speech translation. arXiv preprint arXiv:2201.03713, 2022.

Bongjun Kim, Madhav Ghei, Bryan Pardo, and Zhiyao Duan. Vocal imitation set: a dataset of vocally imitated sound events using the audioset ontology. In DCASE, pages 148–152, 2018.

Peter Knees, Ángel Faraldo Pérez, Herrera Boyer, Richard Vogl, Sebastian Böck, Florian Hörschläger, Mickael Le Goff, et al. Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections. In Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR); 2015 Oct 26-30; Málaga, Spain.[Málaga]: International Society for Music Information Retrieval, 2015. p. 364-70. International Society for Music Information Retrieval (ISMIR), 2015.

Gopala Krishna Koduri, Vignesh Ishwar, Joan Serrà, and Xavier Serra. Intonation analysis of r¯agas in carnatic music. Journal of New Music Research, 43(1):72–93, 2014.

Yuma Koizumi, Heiga Zen, Shigeki Karita, Yifan Ding, Kohei Yatabe, Nobuyuki Morioka, Michiel Adriaan Unico Bacchiani, Yu Zhang, Wei Han, and Ankur Bapna. Libritts-r: Restoration of a large-scale multi-speaker tts corpus. 2023.

Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities, 2024. URL https: //arxiv.org/abs/2402.01831.

Jom Kuriakose, J Chaitanya Kumar, Padi Sarala, Hema A Murthy, and Umayalpuram K Sivaraman. Akshara transcription of mrudangam strokes in carnatic music. In 2015 Twenty First National Conference on Communications (NCC), pages 1–6. IEEE, 2015.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.

Xuechen Liu, Xin Wang, Md Sahidullah, Jose Patino, Héctor Delgado, Tomi Kinnunen, Massimiliano Todisco, Junichi Yamagishi, Nicholas Evans, Andreas Nautsch, et al. Asvspoof 2021: Towards spoofed and deepfake speech detection in the wild. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:2507–2522, 2023.

Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio. Speech model pre-training for end-to-end spoken language understanding. arXiv preprint arXiv:1904.03670, 2019.

Ugo Marchand, Quentin Fresnel, and Geoffroy Peeters. Gtzan-rhythm: Extending the gtzan test-set with beat, downbeat and swing annotations. 2015.

Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D. Plumbley, Yuexian Zou, and Wenwu Wang. WavCaps: A ChatGPT-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. IEEE/ACM Transactions on Audio, Speech, and Language Processing, pages 1–15, 2024.

Fabian Ostermann, Igor Vatolkin, and Martin Ebeling. Aam: a dataset of artificial audio multitracks for diverse music information retrieval tasks. EURASIP Journal on Audio, Speech, and Music Processing, 2023(1):13, 2023.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/ abs/2203.02155.

Igor Pereira, Felipe Araújo, Filip Korzeniowski, and Richard Vogl. Moisesdb: A dataset for source separation beyond 4-stems. arXiv preprint arXiv:2307.15913, 2023.

Karol J. Piczak. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd Annual ACM Conference on Multimedia, pages 1015–1018. ACM Press. ISBN 978-1-4503-3459-4. doi: 10.1145/ 2733373.2806390. URL http://dl.acm.org/citation.cfm?doid=2733373.2806390.

Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A large-scale multilingual dataset for speech research. ArXiv, abs/2012.03411, 2020.

Niccolò Pretto, Barı¸s Bozkurt, Rafael Caro Repetto, Xavier Serra, et al. Nawba recognition for arabandalusian music using templates from music scores. In Proceedings of 15th Sound and Music Computing Conference (SMC’18), pages 405–410, 2018.

Yao Qian, Ximo Bianv, Yu Shi, Naoyuki Kanda, Leo Shen, Zhen Xiao, and Michael Zeng. Speech-language pre-training for end-to-end spoken language understanding. In 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7458–7462. IEEE, 2021.

Antonio Ramires, Frederic Font, Dmitry Bogdanov, Jordan B. L. Smith, Yi-Hsuan Yang, Joann Ching, Bo-Yu Chen, Yueh-Kao Wu, Hsu Wei-Han, and Xavier Serra. The freesound loop dataset and annotation tool. In Proc. of the 21st International Society for Music Information Retrieval (ISMIR), 2020.

CK Reddy, E Beyrami, H Dubey, V Gopal, R Cheng, R Cutler, S Matusevych, R Aichner, A Aazami, S Braun, et al. The interspeech 2020 deep noise suppression challenge: Datasets, subjective speech quality and testing framework. arxiv 2020. arXiv preprint arXiv:2001.08662.

Manuel Sam Ribeiro. Parallel audiobook corpus. [dataset]. University of Edinburgh. School of Informatics. https://doi.org/10.7488/ds/2468, 2018. URL https://datashare.is.ed.ac.uk/ handle/10283/3217.

S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv preprint arXiv:2410.19168, 2024.

Yao Shi, Hui Bu, Xin Xu, Shaoji Zhang, and Ming Li. Aishell-3: A multi-speaker mandarin tts corpus and the baselines. arXiv preprint arXiv:2010.11567, 2020.

Ajay Srinivasamurthy and Xavier Serra. A supervised approach to hierarchical metrical cycle tracking from audio music recordings. In 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5217–5221. IEEE, 2014.

Ajay Srinivasamurthy, Andre Holzapfel, Ali Taylan Cemgil, and Xavier Serra. Particle filters for efficient meter tracking with dynamic bayesian networks. In ISMIR-International Society for Music Information Retrieval Conference, 2015.

Ajay Srinivasamurthy, Andre Holzapfel, Ali Taylan Cemgil, and Xavier Serra. A generalized bayesian model for tracking long metrical cycles in acoustic music signals. In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 76–80. IEEE, 2016.

Ajay Srinivasamurthy, Sankalp Gulati, Rafael Caro Repetto, and Xavier Serra. Saraga: open datasets for research on indian art music. Empirical Musicology Review, 16(1):85–98, 2021.

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Salmonn: Towards generic hearing abilities for large language models. arXiv preprint arXiv:2310.13289, 2023.

Zhiyuan Tang, Dong Wang, Yanguang Xu, Jianwei Sun, Xiaoning Lei, Shuaijiang Zhao, Cheng Wen, Xingjun Tan, Chuandong Xie, Shuran Zhou, et al. Kespeech: An open source speech dataset of mandarin and its eight subdialects. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Christophe Veaux, Junichi Yamagishi, Kirsten MacDonald, et al. Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit. University of Edinburgh. The Centre for Speech Technology Research (CSTR), 6:15, 2017.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705, 2022a.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2023.

Yu Wang, Xinsheng Wang, Pengcheng Zhu, Jie Wu, Hanzhao Li, Heyang Xue, Yongmao Zhang, Lei Xie, and Mengxiao Bi. Opencpop: A high-quality open source chinese popular song corpus for singing voice synthesis. arXiv preprint arXiv:2201.07429, 2022b.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

Julia Wilkins, Prem Seetharaman, Alison Wahl, and Bryan Pardo. Vocalset: A singing voice dataset. In ISMIR, pages 468–474, 2018.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.

Kangxiang Xia, Dake Guo, Jixun Yao, Liumeng Xue, Hanzhao Li, Shuai Wang, Zhao Guo, Lei Xie, Qingqing Zhang, Lei Luo, et al. The iscslp 2024 conversational voice clone (covoc) challenge: Tasks, results and findings. In 2024 IEEE 14th International Symposium on Chinese Spoken Language Processing (ISCSLP), pages 506–510. IEEE, 2024.

Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Xixin Wu, et al. Uniaudio: An audio foundation model toward universal audio generation. arXiv preprint arXiv:2310.00704, 2023.

Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, et al. Air-bench: Benchmarking large audio-language models via generative comprehension. arXiv preprint arXiv:2402.07729, 2024.

Jiangyan Yi, Jianhua Tao, Ruibo Fu, Xinrui Yan, Chenglong Wang, Tao Wang, Chu Yuan Zhang, Xiaohui Zhang, Yan Zhao, Yong Ren, et al. Add 2023: the second audio deepfake detection challenge. arXiv preprint arXiv:2305.13774, 2023.

Fan Yu, Shiliang Zhang, Yihui Fu, Lei Xie, Siqi Zheng, Zhihao Du, Weilong Huang, Pengcheng Guo, Zhijie Yan, Bin Ma, Xin Xu, and Hui Bu. M2MeT: The ICASSP 2022 multi-channel multi-party meeting transcription challenge. In Proc. ICASSP. IEEE, 2022.

Chien yu Huang, Ke-Han Lu, Shih-Heng Wang, Chi-Yuan Hsiao, Chun-Yi Kuan, Haibin Wu, Siddhant Arora, Kai-Wei Chang, Jiatong Shi, Yifan Peng, Roshan Sharma, Shinji Watanabe, Bhiksha Ramakrishnan, Shady Shehata, and Hung yi Lee. Dynamic-superb: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech, 2024. URL https://arxiv.org/abs/2309.09510.

Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. Libritts: A corpus derived from librispeech for text-to-speech. arXiv preprint arXiv:1904.02882, 2019.

Lichao Zhang, Ruiqi Li, Shoutong Wang, Liqun Deng, Jinglin Liu, Yi Ren, Jinzheng He, Rongjie Huang, Jieming Zhu, Xiao Chen, et al. M4singer: A multi-style, multi-singer and musical score provided mandarin singing corpus. Advances in Neural Information Processing Systems, 35:6914–6926, 2022a.

Yu Zhang, Ziya Zhou, Xiaobing Li, Feng Yu, and Maosong Sun. Ccom-huqin: An annotated multimodal chinese fiddle performance dataset. arXiv preprint arXiv:2209.06496, 2022b.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36, 2024.

Kun Zhou, Berrak Sisman, Rui Liu, and Haizhou Li. Emotional voice conversion: Theory, databases and esd. Speech Communication, 137:1–18, 2022.

### A Appendix

- A.1 Task Definition

Speech Domain

Here, we provide a detailed list of each minor task definition for the speech, music, and audio domains, respectively.

#### Speech Recognition (3 minor tasks)

- 1. Automatic Speech Recognition: transcribing speech into text.

- 2. Dialect Automatic Speech Recognition: Automatic Speech Recognition adapted for dialectal variations.

- 3. Phonetic Recognition: identifying and classifying the smallest units of sound in spoken language, known as phonemes. Spoken Language Understanding (2 minor tasks)

- 1. Intent Classification: determining the purpose behind a user’s spoken input.

- 2. Speech to Text Translation: translating spoken language into written text in a different language. Paralinguistic Attribute Recognition (7 minor tasks)

- 1. Gender Recognition: classifying the biological gender of a speaker based on acoustic features of their voice. This task leverages acoustic features of speech, such as pitch, formant frequencies, and speech patterns, which tend to differ between male and female speakers due to physiological differences in the vocal tract and larynx.

- 2. Age Prediction: estimating the age of a speaker based on the acoustic properties of their voice. This task utilizes various speech features, such as pitch, speaking rate, formant frequencies, and spectral characteristics, which can provide cues about the speaker’s age.

- 3. Emotion Recognition: identifying and classifying the emotional state of a speaker based on their vocal expressions.

- 4. Accent Recognition: identifying the regional or cultural accent of a speaker based on their speech characteristics.

- 5. Spoken Paragraph Recognition: determining whether two audio recordings contain the same spoken paragraph by analyzing the linguistic content.

- 6. Language Identification: determining the language spoken from a given audio sample.

- 7. Dialect Identification: determining the specific dialect or regional variation of a language spoken in a given audio sample. Speaker Recognition (4 minor tasks)

- 1. Speaker Verification: verifying a speaker’s identity by comparing their voice to a pre-recorded voiceprint (voice model) of the claimed identity. This process is used to authenticate or verify a speaker’s identity, ensuring that the person speaking is who they claim to be. It includes text-independent and text-dependent speaker verification.

- 2. Speaker Diarization: identifying "who spoke when" in an audio recording containing multiple speakers. This task segments an audio stream into homogeneous regions according to the speaker identity, effectively attributing each segment of speech to its corresponding speaker.

- 3. Speaker Extraction: extracting the speech of a target speaker from a mixture of sounds that may include multiple speakers and background noise.

- 4. Speaker Identification: identifying a speaker from a set of known speakers based on their voice. Speech Caption (1 minor task)

1. Speech Caption: generating synchronized text captions from spoken language. Speech Detection (3 minor tasks)

- 1. Deepfake Detection: detecting whether an audio clip has been artificially manipulated or synthesized using AI techniques, such as voice cloning or deepfake speech generation.

- 2. Vocoder Type Classification: identifying and categorizing the type of vocoder used in a given speech signal.

- 3. Vocoder Type Classification: identifying the device used to record a given speech segment based on its acoustic features. Speech Enhancement (5 minor tasks)

- 1. Denoising: removing unwanted noise from an audio signal to enhance the clarity and quality of the speech. This task involves distinguishing between the speech signal and the background noise, which can include sounds like traffic, machinery, conversations, or other environmental noises.

- 2. Dereverberation: reducing or eliminating the effects of reverberation from an audio signal. Reverberation occurs when sound waves reflect off surfaces such as walls, ceilings, and floors, causing the original speech signal to be combined with multiple delayed copies of itself.

- 3. Declipping: restoring audio signals that have been distorted due to clipping. Clipping occurs when the amplitude of an audio signal exceeds the maximum limit that a recording or playback system can handle, causing the peaks of the waveform to be "clipped" off.

- 4. Speech Bandwidth Extension: enhancing narrowband speech quality by extending its frequency range. Narrowband speech often lacks the higher frequencies that contribute to the naturalness and clarity of speech.

- 5. Signal-to-noise Ratio Estimation: quantifying the ratio of the power of a signal to the power of background noise. This task provides a quantitative measure of the quality of a signal. Speech Generation (9 minor tasks)

- 1. Text to Speech: converting written text into spoken words. It involves synthesizing speech that is natural and understandable, enabling computers to "read" text aloud.

- 2. Zero-shot Text to Speech/Voice Cloning: generating synthetic speech for voices or styles it has never encountered during training.

- 3. Emotional Text to Speech: synthesizing speech with emotional nuances. The goal is to produce speech that not only conveys the content of the text but also expresses specific emotions, making the synthetic voice more engaging and human-like.

- 4. Zero-shot Emotional Text to Speech: generating emotional speech that adapts to an unseen speaker’s voice while rendering specified emotions.

- 5. Descriptive Speech Synthesis: generating synthetic speech that not only replicates the spoken content but also conveys descriptive information about the context of the speech, such as emotions, tone, or other paralinguistic features.

- 6. Spontaneous Text to Speech: generating synthetic speech that mimics the characteristics of spontaneous unscripted human speech. Spontaneous TTS aims to replicate the naturalness, variability, and informal aspects of everyday conversational speech. This includes features such as hesitations, fillers (e.g., "um," "uh"), varying speech rates, and natural prosody changes.

- 7. Voice Conversion: converting one speaker’s voice to resemble another’s while preserving linguistic content and prosody.

- 8. Emotion Conversion: transforming the emotional tone of a spoken utterance from one emotion to another while preserving the linguistic content.

- 9. Speech to Speech Translation: converting spoken language in one language directly into spoken language in another language.

Music Domain

Global MIR (10 minor tasks):

- 1. Key Detection: recognizing the key signature of the given music.

- 2. Scale Recognition: recognizing the scale of the given music.

- 3. Music Tagging: assigning descriptive tags to audio files, such as genre, style, tempo, key, artist, and emotion.

- 4. Genre Classification: categorizing the music into certain genres.

- 5. Emotion Classification: recognizing emotion categories from the music.

- 6. Pitch Classification: classifying the pitch of the given audio.

- 7. Instrument Classification: identifying all existing instruments from the music.

- 8. Vocal Technique Classification: detecting the playing techniques used in the vocal music.

- 9. Instrumental Technique Classification: detecting the playing techniques used in the instrumental music.

- 10 Artist Identification: identifying the relevant artists of a piece of music, given a set of artists as the options. Sequential MIR (3 minor tasks)

- 1. Beat Tracking: detecting and aligning beats of a music excerpt.

- 2. Chord Estimation: estimating the chords sequence at each time step of a music excerpt.

- 3. Progression Extraction: extracting the chord progression represented by chord number sequence. Single Music Reasoning (2 minor tasks)

- 1. Beat-level Instruments Recognition: recognizing the instruments from a certain beat or a certain segment.

- 2. Beat-level Pitch Estimation: estimating the pitch of a certain beat or segment. Multiple Music Reasoning (5 minor tasks)

- 1. Tempo Comparison: comparing the tempo characteristics between two music excerpts.

- 2. Instruments Comparison: comparing instruments of two music excerpts.

- 3. Key Comparison: comparing keys of two music excerpts.

- 4. Instrumental Technique Comparison: comparing playing techniques of two music excerpts.

- 5. Emotion Comparison: comparing emotions of two excerpts. Music Caption (1 minor task)

1. Music Caption: generating textual descriptions for a piece of music. Music Separation (2 minor tasks)

- 1. Melody Extraction: extracting the melody at each time step from a music excerpt.

- 2. Text-guided Source Separation: separate certain tracks from a piece of mixed music with the text instruction. Music Generation (5 minor tasks)

- 1. Text-to-Music Generation: generating the music given the text caption.

- 2. Text-guided Music Continuation: extending a given initial audio segment based on a textual description of musical characteristics while ensuring continuity and coherence.

- 3. Lyrics-to-song Generation: composing a song with the vocal track and instrumental track based on the given lyrics.

- 4. Singing Voice Synthesis: synthesizing the voice given the pitches and lyrics sequence.

- 5. Singing Voice Conversion: transforming the vocals (including the lyrics and melody) of singer A(source vocals) to sound like Singer B (target singer).

Audio Domain Audio Event Recognition (4 minor tasks)

- 1. Sound Event Sequence Recognition: identifying and sequencing various sounds in an audio stream.

- 2. Sound Event Recognition: detecting and identifying a particular sound in audio data.

- 3. Sound Event Detection: determining when a specific sound occurs within an audio clip.

- 4. Acoustic Scene Classification: classifying audio by its acoustic environment (e.g., park, street). Audio Caption (1 minor task)

1. Audio Caption: generating text descriptions that summarize or explain the content of an audio clip. Audio Advanced Understanding (1 minor task)

1. Sound Event Understanding: extracting meaningful information from multiple audio signals (e.g. What is happening in the given audio).

#### Audio Detection (2 minor tasks)

- 1. Deepfake Audio Detection: identifying synthetic or manipulated audio content.

- 2. Voice Activity Detection: identifying segments where human speech is present in the given audio. Audio Classification (2 minor tasks)

- 1. Speech, Silence, Music and Noise Classification: distinguishing between music, speech, and various types of noise.

- 2. Speech and Non-speech Detection: identifying segments which contain speech or non-speech of the given audio. Audio Enhancement (2 minor tasks)

- 1. Audio Inpainting: filling in missing parts of an audio signal.

- 2. Audio Super-resolution: improving the perceptual quality of an audio signal by increasing its resolution. Audio Separation (3 minor tasks)

- 1. Text-guided Audio Source Separation: isolating target sounds from audio using text prompts.

- 2. Label-querying Sound Extraction: extracting sounds belonging to a predefined category from an audio mixture, given a textual label

- 3. Audio-querying Sound Extraction: separating target sounds using an audio reference. Audio Generation (3 minor tasks)

- 1. Text-guided Audio Generation: creating audio based on a textual description.

- 2. Time-grounded Text-to-audio Generation: generating time-aligned audio from text prompts.

- 3. Audio Continuation: generating content that smoothly extends a given audio clip.

- A.2 Datasets for Each Task Here, we present the datasets associated with each minor task.

Table 3: Minor task and its corresponding datasets

Domain Minor Task Dataset Speech Automatic Speech Recognition Aishell1 [Bu et al., 2017], Aishell2 [Du

- et al., 2018], Aishell3 [Shi et al.,

- 2020], ESD [Zhou et al., 2022], EmoV_DB [Adigwe et al., 2018], FLEURS [Conneau et al., 2023], Fluent Speech Commands [Lugosch et al., 2019], HQ-Conversations [Xia et al., 2024], HiFi TTS [Bakhturina et al.,
- 2021], LJSpeech [Ito and Johnson, 2017], MLS [Pratap et al., 2020], The Parallel Audiobook Corpus [Ribeiro, 2018], VCTK [Veaux et al., 2017], aidatatang [Beijing DataTang Technology Co., n.d.], common voice [Ardila

- et al., 2019], LibriTTS-R [Koizumi et al., 2023]

Dialect Automatic Speech Recognition KeSpeech [Tang et al., 2021] Phonetic Recognition Aishell3 [Shi et al., 2020], LibriTTS-

R [Koizumi et al., 2023] Intent Classification Fluent Speech Commands [Qian et al., 2021]

Gender Recognition Aishell1 [Bu et al., 2017] [Bu et al., 2017], Aishell2 [Du et al., 2018], Aishell3 [Shi et al., 2020], Fluent Speech Commands [Lugosch et al., 2019], HQ-Conversations [Xia et al., 2024], KeSpeech [Tang et al., 2021], The Parallel Audiobook Corpus [Ribeiro, 2018], LibriTTSR [Koizumi et al., 2023]

Age Recognition HQ-Conversations [Xia et al., 2024],

KeSpeech [Tang et al., 2021] Emotion Recognition ESD [Zhou et al., 2022] Accent Recognition HQ-Conversations [Xia et al., 2024] Spoken Paragraph Recognition LibriTTS-R [Koizumi et al., 2023]

Language Identification Aishell1 [Bu et al., 2017] [Bu et al.,

- 2017], Aishell2 [Du et al., 2018], Aishell3 [Shi et al., 2020], ESD [Zhou

- et al., 2022], EmoV_DB [Adigwe et al.,

2018], FLEURS [Conneau et al., 2023], HQ-Conversations [Xia et al., 2024], HiFi TTS [Bakhturina et al., 2021], LJSpeech [Ito and Johnson, 2017], MLS [Pratap et al., 2020], The Parallel Audiobook Corpus [Ribeiro, 2018], aidatatang [Beijing DataTang Technology Co., n.d.], common voice [Ardila et al., 2019], LibriTTS-R [Koizumi

- et al., 2023]

Dialect Identification KeSpeech [Tang et al., 2021] Speaker Verification Aishell1 [Bu et al., 2017] [Bu et al., 2017], Aishell2 [Du et al., 2018], Aishell3 [Shi et al., 2020], ESD [Zhou et al., 2022], EmoV_DB [Adigwe et al., 2018], Fluent Speech Commands [Lugosch et al., 2019], HQConversations [Xia et al., 2024], HiFi TTS [Bakhturina et al., 2021], KeSpeech [Tang et al., 2021], The Parallel Audiobook Corpus [Ribeiro, 2018], LibriTTS-R [Koizumi et al., 2023]

Speaker Diarization AliMeeting [Yu et al., 2022] Speaker Extraction LibriMix [Cosentino et al., 2020] Speaker Identification KeSpeech [Tang et al., 2021] Speech Caption LibriTTS-R [Koizumi et al., 2023] Deepfake Detection ASVSpoof2021 [Liu et al., 2023] Vocoder Type Classification ASVSpoof2021 [Liu et al., 2023] Device Recognition HQ-Conversations [Xia et al., 2024] Denoising DNS [Reddy et al.] Dereverberation DNS [Reddy et al.] Declipping DNS [Reddy et al.] Speech Bandwidth Extension DNS [Reddy et al.] Signal-to-noise Ratio Estimation LibriTTS-R [Koizumi et al., 2023]

Speech to Text Translation CVSS [Jia et al., 2022], FLEURS [Conneau et al., 2023] Text to Speech Aishell1 [Bu et al., 2017] [Bu et al.,

- 2017], Aishell2 [Du et al., 2018], Aishell3 [Shi et al., 2020], ESD [Zhou

- et al., 2022], EmoV_DB [Adigwe et al.,

2018], FLEURS [Conneau et al., 2023], Fluent Speech Commands [Lugosch et al., 2019], HQ-Conversations [Xia et al., 2024], HiFi TTS [Bakhturina et al., 2021], KeSpeech [Tang et al., 2021], LJSpeech [Ito and Johnson,

- 2017], MLS [Pratap et al., 2020], The Parallel Audiobook Corpus [Ribeiro,
- 2018], VCTK [Veaux et al., 2017], aidatatang [Beijing DataTang Technology Co., n.d.], common voice [Ardila et al., 2019], LibriTTS-R [Koizumi

- et al., 2023], Genshin [AI-Hobbyist,
- 2024a], StarRail [AI-Hobbyist, 2024b]

Zero-shot Text to Speech Fluent Speech Commands [Lugosch

- et al., 2019], LibriTTS-R [Koizumi

- et al., 2023],HQ-Conversations [Xia
- et al., 2024],Fluent Speech Commands [Lugosch et al., 2019], Aishell2 [Du et al., 2018], Aishell3 [Shi

- et al., 2020], KeSpeech [Tang et al.,
- 2021]

Emotional Text to Speech ESD [Zhou et al., 2022],

EmoV_DB [Adigwe et al., 2018] Zero-shot Emotional Text to Speech ESD [Zhou et al., 2022] Descriptive Speech Synthesis LibriTTS-R [Koizumi et al., 2023] Voice Conversion ESD [Zhou et al., 2022] Emotion Conversion ESD [Zhou et al., 2022] Speech to Speech Translation FLEURS [Conneau et al., 2023]

Music Key Detection AAM [Ostermann et al., 2023], FreeSound Loop Dataset [Ramires et al., 2020]

Music Tagging MTG [Bogdanov et al., 2019]

Genre Classification CSD [Choi et al., 2020], MTG [Bogdanov et al., 2019],FreeSound Loop Dataset [Ramires et al., 2020]

Emotion Classification MTG [Bogdanov et al., 2019] Pitch Classification NSynth [Engel et al., 2017] Instrument Classification AAM [Ostermann et al., 2023],

MTG [Bogdanov et al., 2019], NSynth [Engel et al., 2017]

Vocal Technique Classification VocalSet [Wilkins et al., 2018] Instrumental Technique Classification CCOM-HuQin [Zhang et al., 2022b] Artist Identification FMA [Defferrard et al., 2016] Beat Tracking AAM [Ostermann et al., 2023] Melody Extraction MedleyDB [Bittner et al., 2014] Chord Estimation AAM [Ostermann et al., 2023] Beat-level Instrument Recognition AAM [Ostermann et al., 2023] Progression Extraction JazzNet [Adegbija, 2023] Scale Recognition JazzNet [Adegbija, 2023] Beat-level Pitch Estimation AAM [Ostermann et al., 2023],

CSD [Choi et al., 2020], Vocadito [Bittner et al., 2021]

Tempo Comparison GTZAN Rhythm [Marchand et al., 2015], FreeSound Loop Dataset [Ramires et al., 2020]

Instrument Comparison NSynth [Engel et al., 2017] Key Comparison GiantSteps Key [Knees et al., 2015] Emotion Comparison MTG [Bogdanov et al., 2019] Instrumental Technique Comparison CCOM-HuQin [Zhang et al., 2022b] Music Caption Musiccaps [Agostinelli et al., 2023],

FreeSound Loop Dataset [Ramires et al., 2020]

Text-to-music Generation FreeSound Loop Dataset [Ramires et al., 2020], Musiccaps [Agostinelli et al., 2023], Compmusic [Srinivasamurthy et al., 2021, Anantapadmanabhan et al., 2013, Black et al.,

- 2014, Caro Repetto, 2018, Gupta et al.,
- 2015, Koduri et al., 2014, Kuriakose et al., 2015, Pretto et al., 2018, Srinivasamurthy and Serra, 2014, Srinivasamurthy et al., 2015, 2016]

Text-guided Music Continuation Compmusic [Srinivasamurthy et al., 2021, Anantapadmanabhan et al., 2013, Black et al., 2014, Caro Repetto, 2018, Gupta et al., 2015, Koduri et al., 2014, Kuriakose et al., 2015, Pretto et al., 2018, Srinivasamurthy and Serra, 2014, Srinivasamurthy et al., 2015, 2016]

Lyrics2song Generation CSD [Choi et al., 2020], Vocadito [Bittner et al., 2021], Opencpop [Wang et al., 2022b], Opensinger [Huang et al., 2021]

Singing Voice Synthesis CSD [Choi et al., 2020], Vocadito [Bittner et al., 2021], Opencpop [Wang et al., 2022b], Opensinger [Huang et al., 2021]

Singing Voice Conversion Opensinger [Huang et al., 2021], m4singer [Zhang et al., 2022a] Text-guided Source Separation MedleyVox [Jeon et al., 2023], Moises [Pereira et al., 2023]

Audio Sound Event Sequence Recognition Audioset [Gemmeke et al., 2017] Acoustic Scene Classification TAU Urban Acoustic Scenes [Heittola et al., 2020] Audio Caption Audioset [Gemmeke et al., 2017], Freesound [Font et al., 2013] Text-guided Audio Generation Audioset [Gemmeke et al., 2017], Freesound [Font et al., 2013]

Time-grounded Text-to-audio Generation

Audioset [Gemmeke et al., 2017]

Audio Continuation Wavcaps [Mei et al., 2024]

Audio Inpainting Audioset [Gemmeke et al., 2017] Audio Super-resolution Audioset [Gemmeke et al., 2017] Sound Event Understanding Vocal Imitation [Kim et al., 2018] Text-guided Audio Source Separation Wavcaps [Mei et al., 2024] Label-querying Sound Extraction VGG [Chen et al., 2020] Audio-querying Sound Extraction VGG [Chen et al., 2020] Deepfake Audio Detection ADD2023 [Yi et al., 2023] Voice Activity Detection DNS for VAD [Reddy et al.] Speech, Silence, Music and Noise Classification

Audioset [Gemmeke et al., 2017]

Speech Nonspeech Detection Wavcaps [Mei et al., 2024]

Table 4: Detailed information of datasets.

#### Domain Dataset Audio Length (#hours)

common voice [Ardila et al., 2019] 19,673 aidatatang [Beijing DataTang Technology Co., n.d.]

200

libritts-R [Koizumi et al., 2023] 585 libritts [Zen et al., 2019] 586 HQ-Conversations [Xia et al., 2024] 100 EmoV_DB [Adigwe et al., 2018] 9.49 VCTK [Veaux et al., 2017] 44 MLS [Pratap et al., 2020] 45,042 FLEURS [Conneau et al., 2023] 17 Fluent speech commands [Lugosch et al., 2019]

19

Speech

LibriMix [Cosentino et al., 2020] 500 Aishell1 [Bu et al., 2017] 155 Aishell2 [Du et al., 2018] 1,036 Aishell3 [Shi et al., 2020] 65 LJSpeech [Ito and Johnson, 2017] 23.9 The Parallel Audiobook Corpus [Ribeiro, 2018]

121

HiFi TTS [Bakhturina et al., 2021] 291.6 KeSpeech [Tang et al., 2021] 1,428 ESD [Zhou et al., 2022] 29 CVSS [Jia et al., 2022] 3,809

#### Domain Dataset Audio Length (#hours)

ASVSpoof2021 [Liu et al., 2023] 1270.5

Opencpop [Wang et al., 2022b] 5.2 m4singer [Zhang et al., 2022a] 29.77 FreeSound Loop Dataset [Ramires et al., 2020]

34.7

Opensinger [Huang et al., 2021] 50 MedleyVox [Jeon et al., 2023] 1.1 Vocadito [Bittner et al., 2021] 0.23 MoisesDB [Pereira et al., 2023] 14.4 CSD [Choi et al., 2020] 4.86 Musiccaps [Agostinelli et al., 2023] 15.28 GTZAN rhythm [Marchand et al., 2015]

Music

8.3

GiantSteps key [Knees et al., 2015] 20.07 CCOM-HuQin [Zhang et al., 2022b] 4.3 NSynth [Engel et al., 2017] 340 MedleyDB [Bittner et al., 2014] 7.45 Free Music Archive [Defferrard et al., 2017]

8,232

AAM [Ostermann et al., 2023] 125 MTG [Bogdanov et al., 2019] 3,777

Audioset [Gemmeke et al., 2017] 5208 VGGSound [Chen et al., 2020] 550 Wavcaps [Mei et al., 2024] 3,793.3 freesound [Font et al., 2013] 6446.05 TAU Urban Acoustic Scenes [Heittola et al., 2020]

68.18

Audio

Vocal Imitation [Kim et al., 2018] 24 ESC [Piczak] 2.78 DNS for VAD [Reddy et al.] 562.72 ADD2023 [Yi et al., 2023] 220

Total 50 104,549.18

- A.3 Instruction Template

Here we provide the complete task instruction template in JSONL format with all fields.

Speech-to-Text Translation

{"instruction": "Please translate the speech into the text in

English.", "input": "<|SOA|>Speech_Audio<|EOA|>", "output": "Nevertheless, there are many distinctive ways of

drinking coffee around the world that are worth experiencing

.", "uuid": "UUID", "split": ["train"], "task_type": {

"major": ["Spoken Language Understanding"], "minor": ["Speech-to-text Translation"], "U/G": ["understanding"], "unseen": false

}, "domain": "speech", "source": ["unknown"]

"other": null}

Text-guided Music Continuation

{"instruction": "Please continue the audio music prompt based on the given text description",

"input": "This is a Carnatic music piece set in the atana raga. It follows the 5/8 meter and is composed in the khandaChapu taala

. The lead instrument featured in this performance is vocal, accompanied by Mridangam. The kalai of this composition is 1.\n

audio prompt: <|SOA|>Music_Audio<|EOA|>", "output": "audio: <|SOA|>Musi_Audio<|EOA|>", "uuid": "UUID", "split": ["test"], "task_type": {

"major": ["Music Generation"], "minor": ["Text-guided Music Continuation"], "U/G": ["generation"], "unseen": false },

"domain": "music", "source": ["unknown"], "other": null}

Sound Super-resolution

{"instruction": "Please increase the resolution of the given audio

signal to 32k Hz.", "input": "audio: <|SOA|>Sound_Audio<|EOA|>.", "output": "<|SOA|>Sound_Audio<|EOA|>", "uuid": "UUID", "split": ["train"], "task_type": {

"major": ["Sound Generation"], "minor": ["Sound Super-resolution"], "U/G": ["generation"], "unseen": false },

"domain": "audio", "source": ["youtube"], "other": null}

The definitions of each field are described as follows: Instruction: this field provides the instructions for the task, outlining the specific operation to be performed. Input: this field contains the input data for the task, which represents the raw information to be processed. Output: this field represents the expected result or outcome after processing the input data. Uuid: this field assigns a unique identifier to each task instance, enabling the system to track and manage individual tasks. Split: this field specifies the dataset partition for the task, such as "train", "test", or "dev", which correspond to the training, testing, and development datasets, respectively. Task_type: this field outlines the nature of the task:

- - Major: indicates the primary category of the task.
- - Minor: specifies the secondary or more specific task.
- - U/G: distinguishes whether the task focuses on generation or understanding.
- - Unseen: a boolean value that indicates whether the task involves data that has not been encountered before. Domain: this field defines the domain in which the task is situated, such as "speech", "music", or "audio".

Source: this field identifies the origin of the audio, such as "audiobook", "youtube", or "studio", signifying where the audio signal is sourced from.

Other: this field can store any additional metadata relevant to the task, if applicable.

#### A.4 Instruction Variation Prompt

As mentioned in Section 2.4, all instances are automatically varied by entering a standard prompt in the existing LLMs, which is presented as follows.

You are tasked with paraphrasing the values of the following fields: "instruction", "input", and "output". Your goal is to generate varied and creative rewrites for each of these fields. Please adhere to the following guidelines:

#### 1. Paraphrase Instructions:

- • Paraphrase the "instruction" field in diverse ways by changing the sentence structure, style, and tone. Use a variety of sentence types, including:

- – Direct commands (e.g., "Turn this into speech.")
- – Polite requests (e.g., "Could you please convert this to speech?")
- – Questions (e.g., "Can you turn this into audio?")
- – Suggestions (e.g., "It would be great if you could convert this.")
- – Exclamations or emphatic forms (e.g., "I really need this to be in audio form.")

- • Feel free to add polite elements, such as "please," "kindly," or "if you would be so kind," as long as they remain natural.

#### 2. Paraphrase Inputs:

- • Change the labels for fields like "text:"‘, "text_description:"‘, "audio:"‘, "speaker_audio:"‘, "audio_sample1"‘, "audio_sample2"‘ etc., according to "instruction", while retaining their original meaning. Examples include:

- – "text:" to "spoken text," "speech input," "text excerpt," etc.
- – "text_description:" to "voice style," "descriptive text," "tone characteristics," etc.
- – "audio:" to "source audio," "reference speech," "given recording," etc.
- – "speaker_audio:" to "speaker prompt," "reference voice," "voice sample," etc.

- • Ensure that the content following "text:" remains semantically identical to the original. The content following each label should remain unchanged, with only the labels varying.

#### 3. Maintain Consistency in Outputs:

- • Depending on the tone of the instruction, introduce additional phrases such as:

- – "The gender is ", "Gender: ".
- – "The language is ", "Language in the given speech is ".
- – "The speakers in the given two speechs are ", "The anwser is ".
- – "Transcription is: ", "The text of the given speech is: ".
- – "IPA Phonemes is: ", "phonemes of the given speech is: ".
- – "Descriptive text of the given speech is: ", "The speaking style is: ", "Speech caption is: ".

- • Ensure the "output" field contains the substring |SOA|>audio<|EOA| and the content that follows it, preserving both the structure and meaning.
- • You may optionally introduce phrases before |SOA|>audio<|EOA| (e.g., "Generated speech is:", "Audio output:", "The resulting audio is:"). Avoid altering or introducing inconsistencies in the audio filename (e.g., |SOA|>13_LibriTTS-R_260_123288_000009_000000<|EOA|).

#### 4. Ensure JSON Validity:

- • All strings must be enclosed in double quotes.
- • Key-value pairs must be separated by commas.
- • The JSON structure for "instruction", "input", and "output" fields must be valid and consistent.

##### 5. The number of objects in the output should match the number of inputs. Ensure the format is valid JSON and all JSON objects are properly enclosed. Each entry should be separated by a newline and all JSON should be enclosed in an array.

Please provide the output in valid JSON format (an array of JSON objects), and ensure proper formatting.

#### A.5 Instruction Variation

After the instances (shown in Appendix A.3) are diversified by the prompt in Appendix A.4, we obtain instances with various instructions, which are shown below.

Speech-to-Text Translation Task Template

{ "instruction": "Please translate the speech into the text in

English.", "input": "<|SOA|>Speech_Audio<|EOA|>", "output": "Nevertheless, there are many distinctive ways of

drinking coffee around the world that are worth experiencing." }

- Speech-to-Text Translation Insutrction Variation Example 1

{ "instruction": "Would you be able to convert the spoken words into

English text?", "input": "<|SOA|>Speech_Audio<|EOA|>", "output": "Nevertheless, there are many distinctive ways of

drinking coffee around the world that are worth experiencing.", }

- Speech-to-Text Translation Insutrction Variation Example 2

{ "instruction": "How about translating the speech into Mandarin

English text?", "input": "<|SOA|>Speech_Audio<|EOA|>", "output": "Nevertheless, there are many distinctive ways of

drinking coffee around the world that are worth experiencing.", }

- Speech-to-Text Translation Insutrction Variation Example 3

{ "instruction": Please provide the English translation of the audio

speech.", "input": "<|SOA|>Speech_Audio<|EOA|>", "output": "Nevertheless, there are many distinctive ways of

drinking coffee around the world that are worth experiencing.", }

- Speech-to-Text Translation Insutrction Variation Example 4

{ "instruction": Could you kindly translate the given speech into

written English?", "input": "<|SOA|>Speech_Audio<|EOA|>", "output": "Nevertheless, there are many distinctive ways of

drinking coffee around the world that are worth experiencing.", }

- Speech-to-Text Translation Insutrction Variation Example 5

{ "instruction": Please provide the English translation of the audio

speech.", "input": "<|SOA|>Speech_Audio<|EOA|>", "output": "Nevertheless, there are many distinctive ways of

drinking coffee around the world that are worth experiencing.", }

