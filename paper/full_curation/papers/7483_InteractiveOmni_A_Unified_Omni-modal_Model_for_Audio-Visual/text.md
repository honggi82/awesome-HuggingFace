# arXiv:2510.13747v2[cs.CV]3Dec2025

## InteractiveOmni: A Unified Omni-modal Model for Audio-Visual Multi-turn Dialogue

Wenwen Tong∗, Hewei Guo∗, Dongchuan Ran∗, Jiangnan Chen∗, Jiefan Lu∗, Kaibin Wang∗, Keqiang Li∗, Xiaoxu Zhu∗, Jiakui Li∗, Kehan Li, Xueheng Li, Lumin Li, Chenxu Guo, Jiasheng Zhou, Jiandong Chen, Xianye Wu, Jiahao Wang, Silei Wu, Lei Chen, Hanming Deng,

Yuxuan Song, Dinghao Zhou, Guiping Zhong, Ken Zheng, Shiyin Kang , Lewei Lu

SenseTime Research

* Equal Contribution Corresponding Author

https://github.com/OpenSenseNova/InteractiveOmni

### Abstract

We introduce InteractiveOmni, a unified and open-source omni-modal large language model for audio-visual multi-turn interaction, ranging from 4B to 8B parameters, designed to lead the field of lightweight models by offering comprehensive omni-modal understanding and speech generation capabilities. To achieve this, we integrate the vision encoder, audio encoder, large language model, and speech decoder into a unified model for understanding and generation tasks. We design a multi-stage training strategy to ensure robust cross-modal capabilities, including pre-training for omni-modal understanding, followed by post-training with speech conversation and audio-visual interaction. To enable human-like long-term conversational ability, we meticulously curate a multi-turn training dataset that enhances the model’s ability to handle complex and multi-turn interactions. To effectively evaluate the multi-turn memory and speech interaction capabilities, we construct the multi-modal multi-turn memory benchmark and the multi-turn speech interaction benchmark. Experiments demonstrate that InteractiveOmni significantly outperforms leading open-source models and provides a more intelligent multi-turn audio-visual experience, particularly in its long-term memory capabilities. Notably, InteractiveOmni-4B is comparable to the much larger model like Qwen2.5-Omni-7B on general benchmarks, and it can retain 97% of the performance of the InteractiveOmni-8B while utilizing only 50% of the model size. Achieving state-of-the-art results against similarly sized models across image, audio, video understanding, and speech generation tasks, InteractiveOmni is an accessible, open-source foundation for next-generation intelligent interactive systems.

### 1 Introduction

Human interaction is fundamentally a holistic and multi-modal experience that integrates sensory information from vision, hearing, and language, supporting natural multi-turn communication and long-term memory, which are the core aspects of intelligence. Developing machines with this comprehensive multi-modal multi-turn interactive capability is a critical step toward Artificial General Intelligence (AGI) and represents the next frontier in human-computer interaction [117]. Recent breakthroughs in large language models (LLMs) have shown a degree of intelligence, and this is particularly evident in improved problem-solving capabilities and the growing utility in the real world [16, 121, 61, 174]. Furthermore, LLMs can expand their capabilities by integrating vision and audio processing abilities, evolving towards multi-modal large language models (MLLMs),

Preprint.

[Figure 1]

- Figure 1: Evaluation across image, video, and audio modalities on open-source benchmarks. InteractiveOmni outperforms the current leading multi-modal models such as Qwen2.5-VL-7B[8], Kimi-Audio [42], MiniCPM-o-2.6 [177] and Qwen2.5-Omni-7B [172].

such as vision-language models (VLMs) [95, 8, 30, 195, 167, 65], audio-language models (ALMs) [34, 42, 165], and omni-modal MLLMs (Omni-MLLMs) [50, 90, 172, 35, 74, 170]. Although these works have explored multi-modal capabilities, they mainly focus on understanding ability and single-turn interaction [195, 42, 172], which is different from human-like multi-turn interaction with long-term memory, failing to provide a seamless and integrated user experience on complex, multi-modal interactive tasks in the real world. Therefore, it is necessary to develop an end-to-end Omni-MLLM capable of understanding omni-modal inputs and synthesizing speech as a response with multi-turn conversational ability, which will serve as the core engine for building the next generation of intelligent interactive experiences and breaking down the barriers between modalities. As illustrated in Figure 2, an Omni-MLLM can serve as an intelligent assistant, offering multi-turn memory and interaction capabilities to accompany us on our travels.

Developing the Omni-MLLM with comprehensive multi-modal interactive capability presents several challenges. First, multi-modal alignment is a core difficulty for the development of MLLMs, which has been extensively investigated in VLMs and ALMs [7, 30, 29]. Effectively combining information from heterogeneous data sources, such as images, audio, text, and video, and achieving deep alignment is crucial and much more complex for the training of Omni-MLLM [170, 172]. Second, it is exceptionally challenging to construct an end-to-end unified understanding and generation framework which can process any combination of modal inputs and synchronously generate streaming text and audio [3, 172]. Finally, enhancing the model’s strong interactive capabilities and speech emotional expressiveness is central to its ultimate practical value [54, 15, 124], including the longterm memory, human-like emotion and empathy, and the maintenance of contextual consistency and logical coherence in the multi-turn dialogues. Current MLLMs exhibit limited capabilities for real-world interaction, and there is also a lack of benchmarks to evaluate the multi-turn interaction ability and practicality [143].

To address these challenges, we propose InteractiveOmni, an Omni-MLLM with end-to-end understanding and generation capabilities, providing intelligent multi-turn interactive experience. We employ a single architecture to process and generate data across all modalities, achieving an end-toend workflow from omni-modal input to text and speech output. To address the omni-modal alignment

[Figure 2]

- Figure 2: The schematic diagram of multi-turn audio-visual interaction. InteractiveOmni can perceive external audio and video inputs like a human, actively interact with users, and has the capabilities of multi-turn memory and empathy.

problem, we propose the omni-modal pre-training and post-training strategy. Through meticulously designed pre-training tasks, the model learns the intrinsic correlations between different modalities at an early stage. Subsequently, during the post-training phase, we leverage the instruction tuning and direct preference optimization (DPO) to further strengthen the cross-modal capabilities. Furthermore, to enhance the interactive experience, we constructe highly interactive multi-turn data combined with post-training for optimization, focusing on improving the model’s performance in memory, empathy, and contextual understanding to make its interactions more human-like and intelligent. To effectively evaluate multi-turn memory and speech interaction, we meticulously construct new benchmarks: the multi-modal multi-turn memory benchmark (MMMB) and the multi-turn speech interaction benchmark (MSIB), to address the shortcomings of existing multi-turn dialogue benchmarks.

We develop InteractiveOmni based on open-source models [174, 30, 47, 130], achieving comprehensive leading performance in multi-modal understanding and generation tasks. Specifically, in visual understanding tasks, InteractiveOmni is comparable to state-of-the-art vision-language models such as Qwen2.5-VL-7B [8] and InternVL3.5-8B[159]. For audio understanding and speech conversation tasks, InteractiveOmni’s performance rivals leading audio-language models, including Kimi-Audio [42] and Step-Audio-Chat [165]. Furthermore, InteractiveOmni delivers superior performance on omni-modal benchmarks, outperforming models like MiniCPM-o-2.6 [177], Qwen2.5-Omni-7B [172], and Ming-Lite-Omni [3]. In addition, InteractiveOmni demonstrates superior performance in the comprehensive multi-turn benchmarks, showcasing its excellent interactive capabilities in real-world applications. The key contributions of InteractiveOmni can be summarized as follows:

- • We propose a unified omni-modal model that can simultaneously receive inputs such as images, audio, text, and video and directly generate coherent text and speech streams, achieving truly integrated multi-turn interaction.
- • InteractiveOmni achieves state-of-the-art performance against similarly sized multi-modal large language models on several mainstream open-source benchmarks for image, audio, and video understanding, as well as speech conversation. Notably, InteractiveOmni-4B is comparable to the much larger Qwen2.5-Omni-7B on various benchmarks.
- • InteractiveOmni demonstrates excellent interactive performance with multi-turn and longterm memory capabilities. To effectively evaluate this capability, we construct the multi-turn benchmarks such as MMMB and MSIB, specifically for assessing the multi-turn, multimodal interactive capabilities.

[Figure 3]

- Figure 3: The overview framework of InteractiveOmni. InteractiveOmni is composed of vision encoder, audio encoder, LLM decoder and streaming speech decoder. The extracted visual and audio tokens are processed by the LLM to generate text tokens and speech tokens sequentially.

### 2 Method

#### 2.1 Architecture

As shown in Figure 3, InteractiveOmni is a unified model capable of perceiving omni-modal inputs such as image, video, audio, and text, while generating text and speech sequentially, achieving end-to-end omni-modal perception and generation. InteractiveOmni consists of vision encoder, audio encoder, LLM decoder, speech-token LM, and token2wav speech generator. The architecture of InteractiveOmni-4B and InteractiveOmni-8B is shown in Table 1.

Table 1: The architecture of InteractiveOmni models. Module Vision Encoder Audio Encoder LLM Speech Decoder

InteractiveOmni-4B InternViT Whisper Qwen3-4B Cosyvoice2 InteractiveOmni-8B InternViT Whisper Qwen3-8B Cosyvoice2

We adopt the audio encoder from the Whisper-large-v3 model [130] due to its strong performance on audio understanding tasks. Similar to the preprocessing of audio in Qwen2-Audio [34], we resample the input audio data to a frequency of 16kHz and convert the raw waveform into the 128-channel mel-spectrogram. In addition, we add a pooling layer to downsample the output length of audio to the frame rate of 25Hz, meaning that one second of audio is represented by 25 tokens. An audio adapter with a two-layer MLP projector is employed to connect the audio encoder to LLM.

We utilize the InternViT-300M [28] as the vision encoder to handle the image and video inputs. In terms of the data preprocessing, we employ the dynamic resolution strategy to divide the images into tiles of 448x448 pixels based on the resolution and aspect ratio of the image [29, 30, 195]. Since the representation of high-resolution image and long video inputs requires a large number of visual tokens, we employ the pixel shuffle operation to reduce the number of visual tokens to one-sixteenth of their original number. Thus, a 448x448 image is represented by 64 visual tokens in our model. Finally, a two-layer MLP projector is utilized to map the visual features into the embedding space of the LLM.

We use the pretrained Qwen3 [174] as the LLM decoder, considering its outstanding performance on various text benchmarks. The LLM takes the visual features and audio features as input, and decodes text tokens sequentially. Our speech decoder, based on Cosyvoice2 [47], consists of a speech token LM and a token2wav speech generator. To generate the speech in a streaming fashion, we interleave the generated text tokens and speech tokens in a 5:25 ratio. Specifically, for every five text tokens generated, we pass the text token embeddings and corresponding hidden features to the speech token LM to generate 25 speech tokens, ensuring efficient and seamless speech synthesis. Then, the 25 speech tokens are passed to the token2wav generator to produce the final speech output. For the speech-to-speech conversational scenario, the generated speech style can also be controlled by the user instruction and text hidden features to generate more emotionally expressive speech.

InteractiveOmni is fully trained end-to-end for the omni-modal understanding and generation task based on the hidden embeddings connecting the LLM, vision encoder, audio encoder, and speech decoder. The training datasets are explained in detail in Section 2.2, and the training procedure of InteractiveOmni is given in Section 2.3.

#### 2.2 Datasets

[Figure 4]

- Figure 4: Data construction pipeline for multi-turn dialogue. In each turn, the visual element is sampled from a dedicated image and video repository. The corresponding question is then generated by a vision-language model using a specific prompt tailored to the desired question type. To ensure the dialogue effectively tests long-term memory, we specifically design turns that require recalling historical images and previous dialogue text. Finally, the generated text-format question and answer can be transformed into speech-based question-answer pairs using a TTS system, facilitating end-toend training.

To enhance the performance of audio-visual multi-turn dialogue and improve the long-term memory capacity, we have carefully constructed a multi-turn data generation pipeline. As illustrated in

- Figure 4, we first establish a comprehensive repository of images and videos. For each dialogue turn, the visual element is sampled from this repository to serve as the visual input. The corresponding question is then generated by a vision-language model using a specific prompt tailored to the desired question type. The questions in each turn can be categorized based on the scope of the information required for a correct answer. Specifically, the questions can be categorized into five types:

- • Image-Irrelevant: The question is a pure text-based query that is completely independent of the current image and the dialogue history.
- • Image-Relevant (Current Turn): The question is visually-grounded and can be answered solely by analyzing the current image and the text of the current question.

- • Historical Image Memory: The question requires the model to recall and reason about information presented in a previously shown image within the dialogue history.
- • Historical Text Memory: The question is grounded in the previous turns of the dialogue text, but does not require reference to any specific image.
- • Historical Image and Text Memory: The question necessitates the integration of information from both the historical dialogue text and images.

We primarily construct multi-turn dialogue data within 20 turns based on this data pipeline. To facilitate end-to-end training, we can also transform the generated text-format question and answer into speech-based question-answer pairs using the TTS system.

In the subsequent sections, we provide a detailed breakdown of the training data and the open-source data utilized. This includes data categorized by modality and task: image understanding data, video understanding data, audio understanding data, omni-modal understanding data, audio generation data, and end-to-end dialogue data.

#### 2.2.1 Image Data

To enhance the visual understanding capabilities of the model, we curate a comprehensive collection of multi-modal datasets with approximately 12 million image-text pairs for post-training, including open-source, synthetic, and proprietary in-house data. This visual corpus encompasses multiple domains, such as general question answering (GeneralQA), optical character recognition (OCR), document understanding, mathematics, science, knowledge, and perception. For a detailed statistical breakdown of the open-source dataset’s composition, please refer to Table 2.

Table 2: Detailed statistics of the training data of open-source image data.

Task Datasets OCR

TextVQA[142], OCRVQA[115], ST-VQA[13], LSVT[146], ArT[32], CTW[182], RCTW[140], COCO-Text[153], MTVQA[148], ReCTs[97], MathWriting[55]

InfographicVQA[111], LLaVAR[191], FigureQA[76], MapQA[19], SROIE[68], Docmatix[84], DocVQA[112]

Document Understanding

VQAv2[59], Visual7W[196], ViRL39K[156], MMDU[101], VIST[67], GQA[69], OKVQA[110]

GeneralQA

Science TQA[80], AI2D[79], ScienceQA[105]

Mathematics GeoQA+[23], Geometry3K[104], MathQA[181], MAVIS[190], UniGeo[22] Knowledge A-OKVQA[137], ART500K[109], ViQuAE[86], KVQA[138] Perception

PuzzleVQA[31], Spot-the-diff[197], VSR[94], TallyQA[2], IconQA[106], RefCOCO[78], Object365[139]

#### 2.2.2 Video Data

The video data is composed of various data with 5 million video-text pairs covering several distinct tasks such as the short caption, detailed caption, video question-answering (VideoQA) and Video Temporal Grounding (VTG). This strategic composition of the dataset ensures that the model’s performance can be thoroughly improved across a variety of complexities, from high-level summarization to fine-grained temporal and semantic understanding. A detailed breakdown of the open-source dataset composition is provided in Table 3.

#### 2.2.3 Audio Data

The audio understanding data is built on a massive dataset of over 240,000 hours, including speech, sound, and music data, as shown in Table 4. The primary component is dedicated to automatic speech recognition (ASR), which comprises over 187,000 hours of English and Chinese speech. The ASR data is sourced from academic benchmarks, crowdsourcing, and in-house collections, constituting

Table 3: Detailed statistics of the training data of open-source video data.

Task Datasets

Short Caption InternVid-10M[161], WebVid[9], OpenVid[119], TextVR[166], Mementos[132] Detailed Caption

ShareGPT4Video[25], Vript[175], LSMDC[133], Mementos[132], PEVideo[33], LLaVA-Video[192]

STAR[168], EgoTaskQA[73], TVQA[85], HiREST[184], PerceptionTest[128], VideoGPT+[107], CLEVRER[180]

VideoQA

VTG ET-Instruct-164k[98], hdvila[173], Koala-36M[157], HiREST[184]

approximately 76% of our entire audio dataset and nearly 90% of all speech-related data, providing a robust foundation for speech understanding.

To achieve a more comprehensive and nuanced understanding of audio, the remaining portion of audio data is strategically allocated to a variety of specialized tasks as indicated in Table 4. For speech-related applications, this includes a substantial corpus of over 10,000 hours, such as translation, speech question answering, and emotion recognition. Beyond speech, we incorporate over 18,000 hours of general sound data, with the majority dedicated to question answering and captioning tasks. Furthermore, over 16,000 hours of music data are used for training music-based question answering, enabling the model to interpret a wide spectrum of complex audio signals.

Table 4: Summary of datasets for audio understanding tasks, including speech, sound, and music.

Category Task Datasets Hours

ASR AISHELL [17, 46, 141], ChildMandarin [193], CommonVoice [6], Emilia [64], Fleurs [36], GigaSpeech [20], Libriheavy [77], LibriSpeech [123], LibriTTS [185], MLSENG [127], SPGISpeech [120], WenetSpeech [188]

187,942

QA Open-ASQA-Speech [57], SLURP [10] 2,444 Emotion Recognition MELD [126], IEMCAP [18], CSEMOTIONS [152], Non-

Speech

38

verbal [14]

Translation CoVoST-1 [154], CoVoST-2 [155],GigaST [179] 10,277 Inhouse - 11,282

QA Clotho-AQA [93], CompA-R-Instructions [56], OpenASQA-Non-Speech [57], VocalSound [58]

10,997

Sound Classification Cochlscene [72], ESC-50 [125], MACS [116], TAU [158], Urbansound8k [135], VggSound [21]

835

Sound

Caption AudioCaps [81], Auto-ACD [145], Clotho [45], SoundDescs [83], Epidemic Sounds [70], Wavcaps [113], WavText5k [41]

6,397

Music QA FMA [39], MagnaTagATune [164], FSD2018 [51], Mu-

16,605

sicBench [114], MusicQA [96]

#### 2.2.4 Omni-modal Data

We curate a comprehensive omni-modal dataset by integrating open-source, synthetic, and in-house annotated data. Spanning multiple tasks and modalities, the dataset comprises approximately 15 million data pairs, including audio-to-text, image-audio-to-text, and video-audio-to-text combinations. Based on the image data in Section 2.2.1 and video data in Section 2.2.2, we curate a large volume of speech interaction data by converting the text-based question from the multi-modal dataset into speech-based question using the TTS system. Furthermore, to enhance the ability for spoken dialogue with speech input and text response, we construct a dataset of spoken dialogue through rewriting original formatted multi-modal data or using the LLM to generate spoken conversational data. We also develop a sophisticated, multi-stage data processing pipeline to synthesize multi-turn image, audio, and text conversational data as shown in Figure 4, significantly improving the model’s long-term

memory and multi-turn interactive capabilities. Therefore, the model demonstrates comprehensive and omni-modal understanding capability, which is enabled by the integration of this high-quality omni-modal data.

#### 2.2.5 Text-to-Speech Dataset

This category of data mainly consists of two parts, including the basic speech synthesis data and style-controllable speech synthesis data, as detailed in Table 5. The basic synthesis data consists of approximately 202,000 hours of large-scale public Chinese-English text-speech paired corpora. This dataset is crucial for supporting fundamental speech synthesis capabilities, including linguistic coverage, cross-domain robustness, and generalization. Additionally, we construct about 1,000 hours of style-controllable data, which is guided by natural language instructions along four dimensions: speech rate, emotional tone, dialect, and character persona. These instructions were generated by Qwen3 [174] and subsequently converted into high-quality speech using our in-house TTS system.

Table 5: Detailed statistics of training data for the speech generation task. Task Hours

Speech Synthesis 202k Style-controllable Speech Synthesis 1k Speech-to-Speech Chat 11k Style-controllable Speech-to-Speech Chat 11k

#### 2.2.6 Speech-to-Speech Dataset

The speech-to-speech dataset supports end-to-end model training by enabling the system to comprehend user speech inputs and generate contextually appropriate spoken responses. We curate a large volume of speech-to-speech data by converting the text-based question-answering into speech-based question-answering using the TTS system based on the omni-modal data in Section 2.2.4. In addition, we construct colloquial multi-turn conversational data to improve the naturalness of human-machine interaction. This process yields approximately 11,000 hours of speech conversation data. Furthermore, we develop the data pipeline to generate style-controllable speech-to-speech dialogue data of approximately 11,000 hours, covering different speaking styles such as emotion, speech rate, and role-play. A detailed breakdown of the speech-to-speech dialogue data is given in Table 5. As a result, the model can produce highly expressive and human-like speech responses for the speech-to-speech question-answering tasks.

#### 2.3 Training

The training process of InteractiveOmni comprises two main stages. In the first stage, we perform omni-modal pre-training to achieve alignment across audio, image, video, and text modalities. The second stage involves post-training, which enhances the model’s ability to follow instructions and engage in audio-visual interactions. A detailed description of the training procedure is provided in Section 2.3.1 and Section 2.3.2, respectively.

#### 2.3.1 Pre-training

For the initial pre-training stage, InteractiveOmni is initialized with Qwen3 [174] as the pretrained textual LLM, InternViT-300M [28] as the vision encoder, and Whisper-large-v3 model [130] as the audio encoder. The model is pre-trained on a diverse mixture of datasets, comprising image-text pairs, interleaved image-text data, video-text pairs, audio-text pairs, multimodal question-answering data, and pure text corpora. The instruction-following data is also incorporated in the pre-training stage to further improve the model’s performance. To improve training efficiency, we employ a data-packing strategy, with the maximum token length set to 32,768 to better accommodate long video sequences and multi-turn audio-visual interactions.

The pre-training methodology is structured in three progressive stages to incrementally incorporate additional tasks. In the first stage, we leverage vision-text data to train the vision encoder, establishing a foundational alignment between image, video, and text. The second stage focuses on the audio

encoder, which is trained with audio-text data to align the audio and text modalities. The final stage integrates a vast and diverse corpus of mixed multi-modal data, including audio-image, audio-video, audio-image-text, and audio-video-text data, to improve the model’s comprehensive understanding across all modalities. Extensive evaluations demonstrate that the resulting pre-trained model exhibits strong performance on a wide range of omni-modal understanding tasks.

#### 2.3.2 Post-training

For the post-training stage, we focus on the improvement of audio-visual interaction and speech-tospeech conversational ability to achieve the end-to-end interaction. We conduct multi-task supervised fine-tuning to enhance the model’s ability to follow instructions in audio-visual conversations involving speech-based questions and text-based answers. We curate a large volume of audio-visual interaction data by converting text-based questions from a multimodal question-answering dataset into speech format using a TTS system, including the speech-to-text and image-speech-to-text data as shown in Section 2.2.4. In this stage, the audio encoder, vision encoder, and LLM are trainable to improve the model’s performance with multi-modal inputs. The model can acquire the capabilities of audio-visual understanding and dialogue capabilities after this training stage. We can then directly integrate an external TTS system to enable full audio-visual conversation in a speech-to-speech format.

To achieve end-to-end dialogue, we integrate the speech decoder into the architecture to enable end-toend speech conversation as shown in Figure 3, avoiding the need for an external TTS system. First, we utilize large-scale Chinese–English TTS corpora to train the Speech LM module and adaptor, aligning text tokens with speech tokens. To support streaming speech output, we interleave the generated text tokens and speech tokens in a 5:25 ratio [48]. To address the abundance of simple samples in the TTS-generated data, we employ a hard sample mining strategy to enhance model robustness and performance. Subsequently, the model is trained on speech-to-speech conversational data to enhance end-to-end audio-visual interaction capabilities. During this phase, we curate high-quality multi-turn speech-to-speech and image-speech-to-speech dialogue data to improve contextual conversational ability as shown in Figure 4. Additionally, style-controllable speech-to-speech data is incorporated to strengthen the emotional expressiveness of the generated speech.

Lastly, we utilize the DPO [131] to improve the quality of generated content. Experiments show that DPO is effective for the multi-turn conversational scenario, which can enhance the multi-turn interactive experience. Specifically, for the multi-turn conversation, we mainly optimize the final round to improve the interactive experience. Furthermore, we find that the model merging technique [92] is effective in enhancing the model performance. We apply this technique in the pre-training stage, merging the checkpoint of the pre-trained model and the continuously trained model to improve the model’s performance on the omni-modal understanding tasks.

### 3 Evaluation

We conduct extensive evaluations of InteractiveOmni on both in-house multi-turn conversational benchmarks and open-source benchmarks, covering the omni-modal understanding and speech generation tasks. We compare InteractiveOmni with proprietary models such as GPT-4o [71], Gemini [35] and open-source models including MiniCPM-o-2.6 [177], Qwen2.5-Omni-7B [172], Kimi-Audio [42], Qwen2.5-VL [8], and InternVL3 [195] across image, video, audio and text benchmarks.

- 3.1 Multi-turn Benchmarks

- 3.1.1 Multi-modal Multi-turn Memory Benchmark(MMMB)

Benchmark Introduction. We construct the multi-modal multi-turn memory benchmark (MMMB) to evaluate the multi-turn performance of MLLMs owing to the poor performance of multi-turn interaction capability for current MLLMs. The central objective of MMMB is to investigate the following question: How effectively can MLLMs utilize information from historical turns to answer the question related to the historical images and text in the multi-turn interaction?

MMMB consists of 300 dialogue groups, each with a maximum of 15 turns, designed to assess the multi-turn memory of historical text and images. In each multi-image multi-turn dialogue, textual and visual information is introduced progressively across turns. The final turn poses a question that can only be answered by accurately utilizing information from the historical dialogue context. For performance evaluation, we exclusively assess the model’s response in this final turn, treating all preceding turns as contextual history. Based on the memory information required to answer the final question, the data can be categorized into three types:

- • Text Memory: Answers derived solely from textual information within the dialogue history.
- • Image Memory: Answers that rely on images from previous turns.
- • Mixed Memory: Answers that require both textual and visual information from the history.

Evaluation Results. We evaluate InteractiveOmni against a suite of state-of-the-art open-source and closed-source MLLMs using Gemini-2.5-Pro as the judge model for assessment [35]. As shown in Table 6, InteractiveOmni-4B outperforms leading vision-language models such as Qwen2.5VL-7B [8] and InternVL3-8B [195], as well as Omni-MLLMs including Qwen2.5-Omni-7B [172] and GPT-4o-mini [71]. InteractiveOmni-8B further strengthens the multi-turn performance and is comparable to Gemini-2.5-Flash [35] (58.17 vs. 60.84), demonstrating the strong performance in long-term memory for historical image and text context. To quantify the model performance degradation related to memory information, we conduct two types of evaluation: one measures accuracy based on the number of historical images to be recalled, and the other assesses accuracy according to the turn distance between critical historical turns and the final question. As shown in

- Figure 5, the model’s performance decreases as the turn distance increases. InteractiveOmni-4B maintains an accuracy of 40% even with a turn distance of four. This demonstrates its robustness, which is comparable to the proprietary model such as Gemini-2.5-Flash, and significantly surpasses other open-source models like Qwen2.5-Omni-7B, InternVL3-8B, and Qwen2.5-VL-7B, which only achieve a score of around 20. Additionally, all models exhibit a significant performance decline as the number of images to be memorized increases, highlighting a common weakness in the long-term memory of current MLLMs. For example, even Gemini-2.5-Flash achieves a score of only 20 under these conditions.

- Table 6: Performance evaluation of InteractiveOmni, proprietary and open-source models on the MMMB.

Type Model Text Memory Image Memory Mixed Memory Average Proprietary

GPT-4o-mini [71] 70.00 29.41 58.06 51.33 Gemini-2.5-Flash [35] 75.76 40.19 70.97 60.84

InternVL3-8B [195] 31.31 15.69 33.87 25.86 Qwen2.5-VL-7B [8] 35.35 13.73 27.42 25.10 Qwen2.5-Omni-7B [172] 32.32 9.80 40.32 25.48 InteractiveOmni-4B 70.71 30.39 59.68 52.47 InteractiveOmni-8B 72.73 40.20 64.52 58.17

Open-source

As shown in Figure 6, we compare the performance of Qwen2.5-Omni-7B and InteractiveOmni in multimodal and multi-turn conversations. The results demonstrate that InteractiveOmni can accurately answer questions based on historical image information, showcasing its strong long-term memory capability.

#### 3.1.2 Multi-turn Speech Interaction Benchmark (MSIB)

Benchmark Introduction. To comprehensively assess InteractiveOmni in realistic multi-turn speech dialogue scenarios, we propose the Multi-turn Speech Interaction Benchmark (MSIB). MSIB spans six measurable dimensions: basic conversational ability, emotional expression capability, speech rate control ability, role-playing proficiency, creative capacity, and instruction-following ability. This multi-faceted design enables a comprehensive evaluation of end-to-end audio dialogue systems. The complete task formulations and evaluation protocol (including prompts, turn-structure, and model-as-judge rubric) are detailed in Appendix A. We compare InteractiveOmni-4B and InteractiveOmni-8B against two leading audio-language models, Qwen2.5-Omni-7B [172] and Kimi-Audio [42].

[Figure 5]

- Figure 5: The sketch of performance degradation with the increase of recall burden considering the turn distance and number of memorized images. InteractiveOmni is comparable to proprietary models like GPT-4o-mini and Gemini-2.5-Flash, consistently outperforming open-source models such as InternVL3-8B, Qwen2.5-VL-7B, and Qwen2.5-Omni-7B.

[Figure 6]

- Figure 6: An example of multi-turn conversations requiring historical image context. InteractiveOmni demonstrates enhanced long-term memory performance for historical images compared to Qwen2.5Omni-7B.

Automated Evaluation Results. Table 7 reports model-as-judge scores on the MSIB benchmark on a 1–5 scale. The automated evaluation results demonstrate the superior performance of InteractiveOmni across multiple dimensions of multi-turn speech interaction.

Content Quality Dominance. InteractiveOmni-4B demonstrates clear superiority over both Qwen2.5Omni-7B and Kimi-Audio in content quality, achieving the highest or second-highest scores across five out of six categories. It notably outperforms the baselines in Emotional Expression (3.97) and Role-Playing (3.80), exceeding the second-best baseline by substantial margins. The model also excels in Creative Capacity (3.83), highlighting its strong generative capabilities in empathetic and imaginative scenarios. The larger 8B variant further strengthens these results, leading in four of six content categories.

Competitive Speech Quality. In speech quality, InteractiveOmni-4B remains highly competitive, achieving the second-highest score in Emotional Expression (4.23) and outperforming Kimi-Audio across all categories. While Qwen2.5-Omni-7B leads in several speech tasks, the 4B model consistently surpasses Kimi-Audio, demonstrating a balanced and robust profile. The 8B model secures the top position in three categories including Basic Conversation (4.02) and Emotional Expression (4.26).

- Table 7: Evaluation of InteractiveOmni and baseline models on MSIB using model-as-judge. Best results are in bold and second-best results are underlined, and scores range from 1 to 5.

Basic Conversation

Emotional Expression

Rate Control

Role Playing

Creative Capacity

Instruction Following

Dimension Model

Overall

Qwen2.5-Omni-7B [172] 3.14 2.59 3.29 2.57 3.12 3.14 2.96 Kimi-Audio [42] 3.38 2.98 4.10 2.83 3.44 3.43 3.37 InteractiveOmni-4B 3.67 3.97 3.90 3.80 3.83 3.81 3.84

Content Quality

- InteractiveOmni-8B 3.70 4.00 3.92 4.03 4.05 3.43 3.89

Speech Quality

Qwen2.5-Omni-7B [172] 3.98 4.13 4.41 4.33 4.22 4.00 4.19 Kimi-Audio [42] 3.64 4.03 3.92 3.90 4.05 4.10 3.93 InteractiveOmni-4B 3.79 4.23 4.16 3.93 4.02 4.00 4.05

- InteractiveOmni-8B 4.02 4.26 4.22 4.10 4.05 4.33 4.16

Qwen2.5-Omni-7B [172] 3.56 3.36 3.85 3.45 3.67 3.57 3.58 Kimi-Audio [42] 3.51 3.51 4.01 3.37 3.75 3.77 3.65 InteractiveOmni-4B 3.73 4.10 4.03 3.87 3.93 3.91 3.95 InteractiveOmni-8B 3.86 4.13 4.07 4.07 4.05 3.88 4.03

Average

InteractiveOmni-4B achieves an average score of 3.95, significantly outperforming both Qwen2.5Omni-7B (3.58) and Kimi-Audio (3.65), underscoring its balanced and comprehensive capabilities across content and speech dimensions. The 8B variant further elevates this performance, attaining the highest overall score of 4.03 and leading in all average category scores, reflecting the scalability of the InteractiveOmni series.

[Figure 7]

Figure 7: Human evaluation of the speech-to-speech interactions on MSIB.

Human Evaluation. As shown in Figure 7, human evaluators rate the speech-to-speech conversations on a 1-5 Mean Opinion Score (MOS) scale. The results demonstrate that InteractiveOmni consistently outperforms the baselines across multiple dimensions of general conversational quality (simultaneously considering speech and content quality). Compared with Qwen2.5-Omni-7B and Kimi-Audio, InteractiveOmni achieves higher scores in Basic Conversation, Emotional Expression, Role-Playing, Creative Capacity, and Instruction Following. These results confirm that InteractiveOmni delivers more expressive, coherent, and user-centric interactions, complementing automated metrics with clear human preference advantages.

- 3.2 Open-source Benchmarks

- 3.2.1 Image Understanding Benchmarks

To evaluate the comprehensive capabilities of the model in image understanding tasks, we conduct an extensive assessment on seven benchmarks: MMBench V1.1 [99], MMStar [24], MMMU [183], MathVista [103], HallusionBench [60], AI2D [79], and OCRBench [100]. We compare the performance of our model with state-of-the-art vision-language models (VLMs) and omni-modal models of a similar scale, including InternVL3-8B [195], InternVL3.5-8B [159], Qwen2.5-VL-7B [8], Qwen2.5-Omni-7B [172] and GPT-4o-mini [71]. As shown in Table 8, InteractiveOmni demonstrates competitive performance with VLMs such as InternVL3-8B and Qwen2.5-VL-7B, and is superior to the open-source omni-modal model such as Qwen2.5-Omni-7B. Specifically, InteractiveOmni-8B outperforms all open-source models in HallusionBench, achieving a score of 61.3. These results indicate that InteractiveOmni maintains robust image understanding capabilities and achieves leading performance in specific scenarios.

Table 8: Results on image understanding benchmarks. The score of other models is taken from the OpenCompass [37]. The best result is highlighted in bold, the second-best is underlined.

Type Model MMBench-V1.1 MMStar MMMU MathVista HallusionBench AI2D OCRBench Avg

Visual

InternVL3-8B [195] 82.1 68.7 62.2 70.5 49.0 85.1 88.4 72.3 InternVL3.5-8B [159] 79.5 69.3 73.4 78.4 54.5 84.0 84.0 74.7 Qwen2.5-VL-7B [8] 82.2 64.1 58.0 68.1 51.9 84.3 88.8 71.1

Omni

GPT-4o-mini [71] 76.0 54.8 60.0 52.5 46.1 77.8 78.5 63.7 VITA-1.5 [53] 76.8 60.2 52.6 66.2 44.6 79.2 74.1 64.8 Ming-Lite-Omni [3] 80.8 64.7 56.3 71.6 55.0 83.1 88.4 71.4

Qwen2.5-Omni-7B [172] 81.3 64.0 59.2 67.9 47.4 83.2 83.4 69.5 InteractiveOmni-4B 78.9 62.6 61.1 61.7 52.2 83.8 80.0 68.6 InteractiveOmni-8B 81.4 66.8 66.9 68.0 61.3 84.3 83.7 73.2

Table 9: Results on video understanding benchmarks.

Type Model Video-MME(wo sub) Video-MME(w sub) MLVU(M-Avg) LongVideoBench(val total) Avg

Visual

InternVL3-8B [195] 66.3 68.9 71.4 58.8 66.4 InternVL3.5-8B [159] 66.0 68.6 70.2 62.1 66.7 Qwen2.5-VL-7B [8] 65.1 71.6 70.2 56.0 64.5

Omni

GPT-4o-mini [71] 64.8 - - - Qwen2.5-Omni-7B [172] 64.3 72.4 - - -

InteractiveOmni-4B 63.3 69.3 68.0 57.0 64.4 InteractiveOmni-8B 66.0 71.8 71.6 59.1 67.1

- 3.2.2 Video Understanding Benchmarks

To assess the video understanding capabilities, we conduct a thorough evaluation on representative video understanding benchmarks including Video-MME [99], MLVU [24], and LongVideoBench [183]. We compare InteractiveOmni with several state-of-the-art vision-language models, including InternVL3-8B [195], InternVL3.5-8B [159], Qwen2.5-VL-7B [8], as well as omni-modal models such as Qwen2.5-Omni-7B [172] and GPT-4o-mini [71]. As shown in Table 9, InteractiveOmni achieves competitive performance against existing vision-language models and outperforms Qwen2.5-Omni-7B across multiple benchmarks. These results demonstrate that InteractiveOmni maintains robust and consistent performance across a diverse set of video understanding tasks.

- 3.2.3 Audio Understanding Benchmarks

To thoroughly assess the audio understanding capabilities of our model, we conduct extensive evaluations across a wide range of automatic speech recognition (ASR) and comprehensive audio understanding benchmarks. The ASR evaluations include LibriSpeech (dev-clean, dev-other, test-clean, test-other) [123], WenetSpeech (test-net, test-meeting) [188], AISHELL-1 (test) [17], AISHELL-2 iOS (test) [46], FLEURS (zh, en) [36], and ChildMandarin [193], and we use word error rate (WER) to evaluate the performance. As shown in Table 10, InteractiveOmni-4B achieves competitive performance with much larger specialized audio-language models such as Qwen2-Audio

- Table 10: Results on ASR benchmarks. The best result is highlighted in bold, the second-best is underlined, and results reproduced by ourselves are marked with *.

Datasets Model Performance (WER) ↓

Qwen2.5-Omni-7B [172] 1.60 | 3.50 | 1.80 | 3.40 Qwen2-Audio [34] 1.30 | 3.40 | 1.60 | 3.60 Step-Audio-Chat [66] - | - | 3.19 | 10.67 Kimi-Audio [42] - | - | 1.28 | 2.42 Mini-Omni2 [171] 4.70 | 9.40 | 4.80 | 9.80 VITA-1.5 [53] 8.14 | 18.41 | 7.57 | 16.57 InteractiveOmni-4B 1.60 | 3.38 | 1.73 | 3.69 InteractiveOmni-8B 1.45 | 3.38 | 1.64 | 3.41

LibriSpeech [123] dev-clean | dev-other | test-clean | test-other

Qwen2.5-Omni-7B [172] 5.90 | 7.70 Qwen2-Audio* [34] 10.60 | 10.68 Step-Audio-Chat [66] 8.75 | 9.52 Kimi-Audio [42] 5.37 | 6.28 InteractiveOmni-4B 5.40 | 6.95 InteractiveOmni-8B 5.04 | 5.55

WenetSpeech [188] test-net | test-meeting

- AISHELL-1 [17]

Qwen2.5-Omni-7B [172] 1.13 Qwen2-Audio* [34] 3.01 Step-Audio-Chat [66] 1.95 Kimi-Audio [42] 0.60 InteractiveOmni-4B 1.21

- InteractiveOmni-8B 1.48

AISHELL-2 IOS [46]

Qwen2.5-Omni-7B [172] 2.56 Qwen2-Audio* [34] 4.48 Step-Audio-Chat [66] 3.57 Kimi-Audio [42] 2.56 InteractiveOmni-4B 2.85

- InteractiveOmni-8B 2.18

Whisper-Large-V3 [130] 7.70 | 4.10 Qwen2.5-Omni-7B [172] 3.00 | 4.10 Qwen2-Audio* [34] 7.50 | 5.67 Step-Audio-Chat [66] 4.26 | 8.56 Kimi-Audio [42] 2.56 | 4.44 InteractiveOmni-4B 3.86 | 4.53 InteractiveOmni-8B 3.49 | 4.14

FLEURS [36] zh | en

Qwen2.5-Omni-7B* [172] 19.34 Qwen2-Audio* [34] 14.62 InteractiveOmni-4B 17.21 InteractiveOmni-8B 14.03

ChildMandarin [193]

[34], Step-Audio-Chat [66] and Kimi-Audio [42]. Specifically, InteractiveOmni-8B surpasses all open-source omni-modal and audio-language models on the challenging WenetSpeech benchmark, attaining a score of 5.04 on test-net and 5.55 on test-meeting, demonstrating the superior performance of the audio understanding ability. In addition, InteractiveOmni achieves state-of-the-art performance

- Table 11: Results on audio understanding tasks. The best result is highlighted in bold, the second-best is underlined, and results reproduced by ourselves are marked with *.

Datasets Model Performance ↑

Qwen2.5-Omni-7B [172] 67.78 | 69.16 | 59.76 | 65.60 Qwen2-Audio* [34] 60.66 | 58.08 | 51.05 | 56.6 Kimi-Audio [42] 73.27 | 61.68 | 60.66 | 65.20 MiDashengLM-7B [43] 68.47 | 66.77 | 63.66 | 66.30 InteractiveOmni-4B 70.87 | 76.05 | 69.07 | 72.00 InteractiveOmni-8B 69.07 | 73.05 | 60.06 | 67.39

MMAU [134]

Sound | Music | Speech | Avg.

Qwen2.5-Omni-7B* [172] 7.96 | 6.82 | 6.43 | 6.69 | 6.98 Qwen2-Audio [34] 7.18 | 6.99 | 6.79 | 6.77 | 6.93 SALMONN [147] 6.16 | 6.28 | 5.95 | 6.08 | 6.12 Phi-4-Multimodal [1] 7.47 | 7.00 | 6.67 | 6.78 | 6.98 InteractiveOmni-4B 7.82 | 7.13 | 5.91 | 5.55 | 6.60 InteractiveOmni-8B 7.54 | 7.07 | 6.00 | 5.54 | 6.54

AIR-Bench [176] Speech | Sound | Music | Mixed-Audio | Avg

Qwen2.5-Omni-7B [172] 57.00 Qwen2-Audio [34] 55.30 Step-Audio-Chat [66] 33.54 Kimi-Audio [42] 59.13 InteractiveOmni-4B 57.16 InteractiveOmni-8B 57.55

MELD [126]

Qwen2.5-Omni-7B [172] 73.12 | 72.86 Qwen2-Audio [34] 72.63 | 71.73 Step-Audio-Chat [66] 44.98 | 45.84 Kimi-Audio [42] 73.18 | 71.24 InteractiveOmni-4B 71.91 | 71.28 InteractiveOmni-8B 72.98 | 74.49

ClothoAQA [93] dev | test

on both the AISHELL-2 IOS and ChildMandarin benchmarks, demonstrating its strong capability in Mandarin comprehension.

Beyond speech recognition, we systematically evaluate the model across a wide range of audio domains, including environmental sound detection, music analysis, speech comprehension, emotion recognition, audio question answering, and vocal sound classification. These assessments are conducted using benchmark datasets such as MMAU [134], AIR-Bench [176], MELD [126], and ClothoAQA [93]. As shown in Table 11, the comprehensive evaluation framework provides a holistic characterization of the model’s audio perception capabilities, highlighting the proficiency of InteractiveOmni in capturing complex acoustic patterns, including the paralinguistic information and sound signals. Notably, InteractiveOmni-4B demonstrates exceptional parameter efficiency by surpassing all open-source 7B-sized models on the MMAU benchmark, achieving an average score of 72.00.

#### 3.2.4 Omni-modal Understanding Benchmarks

We evaluate the omni-modal benchmark OmniBench [91] to assess the omni-modal understanding capability of MLLMs, and compare InteractiveOmni with Qwen2.5-Omni-7B and other models. As shown in Table 12, InteractiveOmni-4B achieves state-of-the-art performance on OmniBench,

attaining an average score of 59.19 that substantially exceeds other Omni models, demonstrating exceptional omni-modal understanding capabilities.

Table 12: Performance of InteractiveOmni on OmniBench compared with leading MLLMs. Model Speech Sound Event Music Avg

Gemini-1.5-Pro [150] 42.67 42.26 46.23 42.91 MIO-Instruct [163] (7B) 36.96 33.58 11.32 33.80 AnyGPT (7B) [187] 17.77 20.75 13.21 18.04 video-SALMONN (13B) [144] 34.11 31.70 56.60 35.64 UnifiedIO2-xlarge (3.2B) [102] 39.56 36.98 29.25 38.00 UnifiedIO2-xxlarge (6.8B) [102] 34.24 36.98 24.53 33.98 MiniCPM-o-2.6 [177] - - - 40.50 Baichuan-Omni-1.5 [90] - - - 42.90 Qwen2.5-Omni-7B [172] 55.25 60.00 52.83 56.13

InteractiveOmni-4B 60.70 61.51 42.45 59.19 InteractiveOmni-8B 60.18 62.64 55.66 60.33

Table 13: Results on speech-to-text question-answering benchmarks.

Datasets Model Performance

Qwen2-Audio [34] 42.77 | 69.67 | 45.20 | 40.30 | 57.19 | 51.03 GLM-4-Voice [186] 47.43 | 76.00 | 55.40 | 51.80 | 57.89 | 57.70 VITA-1.5 [53] 41.00 | 74.20 | 57.30 | 46.80 | 68.20 | 57.50 Step-Audio-chat [66] 60.00 | 72.33 | 73.00 | 56.80 | 56.53 | 63.73 Baichuan-Audio [88] 41.90 | 78.40 | 64.50 | 61.70 | 77.40 | 64.78 Kimi-Audio [42] 58.02 | 79.33 | 70.20 | 62.10 | 75.73 | 69.08 MiniCPM-o-2.6 [177] 38.60 | 77.80 | 68.60 | 61.90 | 51.80 | 59.74 Baichuan-Omni-1.5 [90] 50.00 | 78.50 | 59.10 | 57.20 | 77.90 | 64.54 Qwen2.5-Omni-7B [172] 63.76 | 75.33 | 62.80 | 57.06 | 72.76 | 66.34 InteractiveOmni-4B 69.11 | 79.33 | 65.80 | 56.40 | 74.87 | 69.10 InteractiveOmni-8B 71.68 | 80.67 | 70.30 | 66.50 | 74.57 | 72.74

OpenAudioBench Reasoning QA | Llama Questions | Web Questions | TriviaQA | AlpacaEval | Avg

Qwen2-Audio [34] 3.69 | 3.40 | 3.01 | 35.35 | 35.43 GLM-4-Voice [186] 4.06 | 3.48 | 3.18 | 43.31 | 40.11 VITA-1.5 [53] 4.21 | 3.66 | 3.48 | 38.88 | 52.15 Step-Audio-chat [66] 3.99 | 2.99 | 2.93 | 46.84 | 28.72 Baichuan-Audio [88] 4.41 | 4.08 | 3.92 | 45.84 | 53.19 Kimi-Audio [42] 4.46 | 3.97 | 4.20 | 63.12 | 62.17 MiniCPM-o-2.6 [177] 4.42 | 4.15 | 3.94 | 50.72 | 54.78 Baichuan-Omni-1.5 [90] 4.50 | 4.05 | 4.06 | 43.40 | 57.25 Qwen2.5-Omni-7B [172] 4.50 | 3.84 | 3.89 | 56.40 | 61.32 InteractiveOmni-4B 4.27 | 4.20 | 3.94 | 41.41 | 63.24 InteractiveOmni-8B 4.61 | 4.34 | 4.21 | 44.67 | 65.26

VoiceBench AlpacaEval | CommonEval | WildVoice | SD-QA | MMSU

Qwen2-Audio [34] 49.01 | 54.70 | 22.57 | 98.85 | 55.32 GLM-4-Voice [186] 52.97 | 52.80 | 24.91 | 88.08 | 57.40 VITA-1.5 [53] 71.65 | 55.30 | 38.14 | 97.69 | 64.53 Step-Audio-chat [66] 31.87 | 50.60 | 29.19 | 65.77 | 50.13 Baichuan-Audio [88] 71.65 | 54.80 | 50.31 | 99.42 | 69.27 Kimi-Audio [42] 83.52 | 69.70 | 61.10 | 100.0 | 76.91 MiniCPM-o-2.6 [177] 78.02 | 60.40 | 49.25 | 97.69 | 71.23 Baichuan-Omni-1.5 [90] 74.51 | 62.70 | 54.54 | 97.31 | 71.32 Qwen2.5-Omni-7B [172] 80.90 | 66.70 | 53.50 | 99.20 | 73.60 InteractiveOmni-4B 82.64 | 55.90 | 60.90 | 99.62 | 73.10 InteractiveOmni-8B 86.37 | 73.30 | 57.99 | 99.42 | 76.69

VoiceBench OpenBookQA | IFEval | BBH | AdvBench | Avg

#### 3.2.5 Speech-to-text Benchmarks

To assess the speech understanding and speech-based question-answering capabilities, we evaluate InteractiveOmni on the following benchmarks: OpenAudioBench [88] and VoiceBench [27]. As shown

in Table 13, our model performs excellently on the speech-to-text question-answering benchmarks, outperforming recent open-source audio language models and omni models. InteractiveOmni-4B achieves an average score of 69.10 on OpenAudioBench, significantly outperforming Kimi-Audio [42], Step-Audio-chat [66] and Qwen2.5-Omni-7B [172]. These voice-chat benchmarks reflect our model’s substantial progress in diversified speech interaction.

#### 3.2.6 Text-to-Speech Benchmarks

To evaluate the speech generation capability of InteractiveOmni, we conducte a comparative study on the Seed-TTS test set [5] against both a state-of-the-art TTS system and the omni-modal models. The Seed-TTS benchmark includes Seed test-zh, test-en, and test-hard, covering diverse input texts and reference speeches across multiple domains. As shown in Table 14, compared with the omnimodal models such as Ming-Lite-Omni [3] and Qwen2.5-omni-7Bicl [172], InteractiveOmni-4B achieves considerably better performance on Seed-TTS-test-zh, reaching a level comparable to highly professional TTS systems.

Table 14: Performance on the Seed-TTS benchmark, as measured by WER↓.

Type Model test-zh test-en test-zh-hard

MaskGCT [162] 2.27 2.62 10.27

SeedTTS [5] 1.12 2.25 7.59 CosyVoice 2 [48] 1.45 2.57 6.83

TTS

MinMo [26] 2.48 2.90 Ming-Lite-Omni [3] 1.69 4.31 -

MLLM

Qwen2.5-Omni-7B [172] 1.70 2.72 7.97 InteractiveOmni-4B 1.37 3.73 8.02 InteractiveOmni-8B 1.56 2.33 7.92

To further assess how the models handle nuanced and semantically complex texts, we conduct evaluations on EmergentTTS-Eval[108], a comprehensive benchmark covering six challenging TTS scenarios: emotions, paralinguistics, foreign words, syntactic complexity, complex pronunciation (e.g., URLs, formulas), and questions. As shown in Table 15, InteractiveOmni-4B and InteractiveOmni-8B achieve an overall WER of 22.04 and 18.07, respectively, surpassing all other models. Furthermore, it reaches state-of-the-art performance in several sub-categories, including emotions, questions, paralinguistics, and complex pronunciation, surpassing leading omni-modal models.

- Table 15: Performance on the EmergentTTS benchmark, as measured by WER↓. The results for competing models are drawn from the EmergentTTS-Eval [108].

Foreign Words

Complex Pronunciation

Syntactic Complexity

Model Overall Emotions

Paralinguistics

Questions

VITS-VCTK [82] 27.45 16.34 47.45 51.12 44.30 17.82 2.37 Tortoise-TTS [12] 28.62 13.04 29.61 64.93 51.87 10.44 6.35 Sesame1B [136] 32.32 17.07 45.27 49.63 80.97 2.74 4.30 MiniCPM-o-2.6 [177] 31.40 12.36 33.46 58.48 82.15 5.21 3.08 Qwen2.5-Omni-7B [172] 26.58 1.22 26.98 57.48 64.07 12.77 1.66

InteractiveOmni-4B 22.04 1.59 29.75 33.37 66.36 1.67 5.19 InteractiveOmni-8B 18.07 1.05 28.34 26.68 54.37 1.09 1.44

#### 3.2.7 Spoken Dialogue Benchmarks

We evaluate the end-to-end spoken dialogue capabilities of InteractiveOmni based on the benchmarks: Llama Question [118], Web Question [11], TriviaQA [75], and AlpacaEval [89]. A comparison of the speech interaction abilities of speech LLMs and omni models is shown in Table 16. InteractiveOmni achieves almost state-of-the-art performance across all four benchmarks on the speech-to-text and speech-to-speech evaluations, indicating its strong capabilities in handling a wide range of conversational scenarios. These results show that InteractiveOmni excels in speech understanding and stable speech generation on end-to-end speech interaction scenarios.

- Table 16: The performance of the speech LLMs on spoken dialogue benchmarks. S2T and S2S represent the speech-to-text and the speech-to-speech performance, respectively. Results reproduced by ourselves are marked with *.

LlamaQuestion WebQuestion TriviaQA AlpacaEval

Model

S2T↑ S2S↑ S2T↑ S2S↑ S2T↑ S2S↑ S2T↑ S2S↑ Moshi [40] 62.3 21.0 26.6 9.2 22.8 7.3 - GLM-4-Voice [186] 64.7 50.7 32.2 15.9 39.1 26.5 - Kimi-Audio* [42] 82.0 66.6 69.0 60.6 64.2 52.8 58.2 36.4 Freeze-Omni [160] 72.0 - 44.7 - 53.8 - - VITA-1.5 [53] 74.2 - 57.3 - 46.8 - 68.2 MiniCPM-o-2.6 [177] 77.8 - 68.6 - 61.9 - 51.8 LLaMA-Omni2-7B [50] 70.3 60.7 34.5 31.3 - - - Qwen2.5-Omni-7B* [172] 77.6 74.6 65.8 64.7 57.4 55.9 56.1 49.6 InteractiveOmni-4B 76.3 65.3 64 58.7 53.1 43.8 53.8 46.4 InteractiveOmni-8B 81.0 69.0 71.3 64.6 66.8 56.3 74.3 58.1

### 4 Related works

#### 4.1 Vision-language Models

VLMs extend traditional LLMs by integrating vision and language understanding within a unified framework. By jointly processing textual and visual input, VLMs enable complex cross-modal reasoning tasks such as image captioning, visual question answering, and multimodal dialogue. They typically leverage pre-trained vision encoders, like CLIP [129] or ViT [44], combined with powerful language backbones to align heterogeneous representations in a shared semantic space. Early multimodal models such as Flamingo [4] and BLIP-2 [87] focused on bridging vision encoders with language models through pre-training and efficient alignment strategies, enabling tasks such as captioning and visual question answering (VQA). Following this line of work, a series of instruction-tuned vision-language models emerged almost simultaneously, including InstructBLIP [38], LLaVA [95], MiniGPT-4 [194], and mPLUG-Owl [178]. These models adopt a common paradigm of leveraging large language models combined with vision encoders, and are trained on instruction-following datasets to enable effective multimodal alignment. Closed-source generalpurpose models, such as GPT-4V [122] and Google’s Gemini [149], extend multimodal capabilities beyond research prototypes by integrating text, vision, and other modalities within unified frameworks. Recent work like Qwen2.5-VL [8], InternVL3.5 [159], Seed1.5-VL [62], GLM-4.1V [65], Kimi-VL [151] and MiMo-VL [167] primarily focused on native resolution, Mixture-of-Experts architecture, visual reasoning capabilities and reinforcement learning. In addition, to fully unleash the potential of the model, they collected a large amount of high-quality data, including available open-source datasets as well as carefully designed and curated in-house data. These optimizations and improvements have led to significant performance gains across a wide range of vision-related tasks, including OCR, general question answering, and visual reasoning.

#### 4.2 Speech-to-Speech Dialogue Models

Against the backdrop of the rapid evolution of LLM, both academia and industry have placed high expectations on the development of speech-to-speech models. Traditionally, end-to-end speech systems are constructed by sequentially integrating an automatic speech recognition module, an LLM, and a text-to-speech module, which is constrained by the high latency, insufficient paralinguistic perception, and error propagation across cascaded modules. To address these challenges, several new paradigms have been proposed. For example, Twist [63] introduces a framework that enables pretrained textual language models to directly generate speech, thereby bridging the gap between text-based reasoning and spoken interaction. Moshi [40] proposes a full-duplex end-to-end spoken dialogue system that employs a multi-stream output mechanism to simultaneously produce audio and text tokens.

Several speech-to-speech models have explored training strategies with curated training data to enhance the performance of end-to-end spoken dialogue systems. GLM-4-Voice [186] leverages interleaved data during pre-training to support text-guided interleaved speech generation. MinMo [26] demonstrates the feasibility of end-to-end language systems by employing a multi-stage training

strategy on 1.4 million hours of data. Baichuan-Audio [88] adopts a multi-codebook discretization method to preserve both semantic and acoustic information, thereby enabling effective modeling of speech within the LLM. Furthermore, LLaMA-Omni2 [50] explores techniques for integrating discrete text embeddings with continuous hidden representations. Recently, Kimi-Audio [42] achieves state-of-the-art performance across multiple speech and audio understanding benchmarks. Step-Audio 2 [165] further advances the emotional and paralinguistic expressivity of end-to-end spoken dialogue systems by incorporating chain-of-thought reasoning and reinforcement learning. These studies achieve more natural speech-to-speech human-machine communication with end-to-end training.

#### 4.3 Omni-modal Large Language Model

To achieve human-like omni-modal interactive experience, the MLLMs have propelled the development of the Omni-modal MLLM, which can process omni-modality including image, video, audio, and text, such as GPT-4o [71] and Gemini [35]. Compared with VLMs and ALMs, the Omni-MLLM integrates data from more modalities, enabling it to learn richer contextual information and gain a deeper understanding of the latent relationships between different modalities. VITA [52] achieves the omni-modal understanding capability, which can simultaneously process the video, image, text, and audio modalities towards the natural human-computer experience. Mini-Omni2 [170] proposes the multi-modal model as a visual voice assistant to achieve the audio-visual interaction similar to the functionality of GPT-4o. Ming-Omni [3] proposes a unified architecture capable of processing images, audio, video and text, while generating speech and images. Qwen2.5-Omni [172] introduces an end-to-end model that can perceive all modalities and generate text and speech in a streaming fashion.

For the unified understanding and generation, the speech modality can be represented by the discrete audio token or continuous audio feature. Models based on discretized audio encoding [40, 186, 189] expand tokens into the vocabulary of the LLM to achieve unified understanding and generation of omni-modalities, while the training of the model usually requires a large amount of cross-modal data. In contrast, models based on continuous audio feature encoding [50, 49, 160, 169, 26] can maximize the preservation of the basic capabilities of the LLM. Thus, the omni-modal alignment and the design of a unified understanding and generation architecture still present several challenges. In addition, these omni-modal MLLMs show poor performance in multi-turn interaction, failing to achieve a natural, human-like conversational flow.

### 5 Conclusions

In this work, we present InteractiveOmni, a unified, open-source omni-modal large language model that seamlessly integrates comprehensive multi-modal understanding with natural speech generation, demonstrating superior performance on multi-turn interaction tasks. Our unified framework successfully integrates the processing of text, image, audio, and video inputs, and directly generates coherent text and speech, enabling a truly seamless and intelligent interactive experience.

Combining omni-modal pre-training for foundational modality alignment and post-training for omnimodal understanding and audio-visual interaction, we effectively address the critical challenge of cross-modal synergy. Furthermore, the meticulous curation of high-quality, multi-turn dialogue data is essential in endowing InteractiveOmni with robust long-term memory and contextual awareness, enabling interactions that are significantly more natural and intelligent. InteractiveOmni demonstrates a clear superiority over comparable models in our newly constructed multi-turn benchmarks, showcasing its advanced capabilities in maintaining context and memory in complex dialogues. Moreover, it achieves state-of-the-art performance against similarly sized MLLMs across a suite of mainstream open-source benchmarks for image, audio, and video understanding, as well as speech conversation, proving its robustness and versatility.

InteractiveOmni lays a strong foundation for the next generation of multi-modal AI assistants. Our future work includes enhancing the model’s efficiency for real-time interaction and expanding its capacity to comprehend more complex, abstract inter-modal relationships, paving the way for a more authentic and human-like user experience.

### References

- [1] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743, 2025.
- [2] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084, 2019.
- [3] Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, et al. Ming-omni: A unified multimodal model for perception and generation. arXiv preprint arXiv:2506.09344, 2025.
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.
- [5] Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024.
- [6] Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus. arXiv preprint arXiv:1912.06670, 2019.
- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023.
- [8] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [9] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.
- [10] Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser. Slurp: A spoken language understanding resource package. arXiv preprint arXiv:2011.13205, 2020.
- [11] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1533–1544, 2013.
- [12] James Betker. Better speech synthesis through scaling. arXiv preprint arXiv:2305.07243, 2023.
- [13] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marcal Rusinol, Ernest Valveny, C.V. Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.
- [14] Maksim Borisov, Egor Spirin, and Daria Diatlova. Nonverbaltts: A public english corpus of text-aligned nonverbal vocalizations with emotion annotations for text-to-speech. arXiv preprint arXiv:2507.13155, 2025.
- [15] Boson AI. Higgs Audio V2: Redefining Expressiveness in Audio Generation. https: //github.com/boson-ai/higgs-audio, 2025. GitHub repository. Release blog available at https://www.boson.ai/blog/higgs-audio-v2.
- [16] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.

- [17] Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In 2017 20th conference of the oriental chapter of the international coordinating committee on speech databases and speech I/O systems and assessment (O-COCOSDA), pages 1–5. IEEE, 2017.
- [18] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42(4):335–359, 2008.
- [19] Shuaichen Chang, David Palzer, Jialin Li, Eric Fosler-Lussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps. arXiv preprint arXiv:2211.08545, 2022.
- [20] Guoguo Chen, Shuzhou Chai, Guanbo Wang, Jiayu Du, Wei-Qiang Zhang, Chao Weng, Dan Su, Daniel Povey, Jan Trmal, Junbo Zhang, et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. arXiv preprint arXiv:2106.06909, 2021.
- [21] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 721–725. IEEE, 2020.
- [22] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746, 2022.
- [23] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021.
- [24] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large visionlanguage models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.
- [25] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024.
- [26] Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, et al. Minmo: A multimodal large language model for seamless voice interaction. arXiv preprint arXiv:2501.06282, 2025.
- [27] Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T Tan, and Haizhou Li. Voicebench: Benchmarking llm-based voice assistants. arXiv preprint arXiv:2410.17196, 2024.
- [28] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [29] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv:2404.16821, 2024.
- [30] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.
- [31] Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. arXiv preprint arXiv:2403.13315, 2024.

- [32] Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1571–1576. IEEE, 2019.
- [33] Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, et al. Perceptionlm: Open-access data and models for detailed visual understanding. arXiv preprint arXiv:2504.13180, 2025.
- [34] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.
- [35] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [36] Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805. IEEE, 2023.
- [37] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models, 2023.
- [38] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv:2305.06500, 2023.
- [39] Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840, 2016.
- [40] Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024.
- [41] Soham Deshmukh, Benjamin Elizalde, and Huaming Wang. Audio retrieval with wavtext5k and clap training. arXiv preprint arXiv:2209.14275, 2022.
- [42] Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, et al. Kimi-audio technical report. arXiv preprint arXiv:2504.18425, 2025.
- [43] Heinrich Dinkel, Gang Li, Jizhong Liu, Jian Luan, Yadong Niu, Xingwei Sun, Tianzi Wang, Qiyang Xiao, Junbo Zhang, and Jiahao Zhou. Midashenglm: Efficient audio understanding with general audio captions. arXiv preprint arXiv:2508.03983, 2025.
- [44] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- [45] Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 736–740. IEEE, 2020.
- [46] Jiayu Du, Xingyu Na, Xuechen Liu, and Hui Bu. Aishell-2: Transforming mandarin asr research into industrial scale. arXiv preprint arXiv:1808.10583, 2018.
- [47] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024.

- [48] Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117, 2024.
- [49] Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. Llama-omni: Seamless speech interaction with large language models. arXiv preprint arXiv:2409.06666, 2024.
- [50] Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, and Yang Feng. Llama-omni2: Llmbased real-time spoken chatbot with autoregressive streaming speech synthesis. arXiv preprint arXiv:2505.02625, 2025.
- [51] Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel PW Ellis, Xavier Favory, Jordi Pons, and Xavier Serra. General-purpose tagging of freesound audio with audioset labels: Task description, dataset, and baseline. arXiv preprint arXiv:1807.09902, 2018.
- [52] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Yuhang Dai, Meng Zhao, Yi-Fan Zhang, Shaoqi Dong, Yangze Li, Xiong Wang, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024.
- [53] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Yangze Li, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025.
- [54] Xuelong Geng, Qijie Shao, Hongfei Xue, Shuiyuan Wang, Hanke Xie, Zhao Guo, Yi Zhao, Guojian Li, Wenjie Tian, Chengyou Wang, et al. Osum-echat: Enhancing end-to-end empathetic spoken chatbot via understanding-driven spoken dialogue. arXiv preprint arXiv:2508.09600, 2025.
- [55] Philippe Gervais, Anastasiia Fadeeva, and Andrii Maksai. Mathwriting: A dataset for handwritten mathematical expression recognition. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, page 5459–5469, New York, NY, USA, 2025. Association for Computing Machinery.
- [56] Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Kiran Reddy Evuru, Utkarsh Tyagi, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha. Gama: A large audiolanguage model with advanced audio understanding and complex reasoning abilities. arXiv preprint arXiv:2406.11768, 2024.
- [57] Yuan Gong, Alexander H Liu, Hongyin Luo, Leonid Karlinsky, and James Glass. Joint audio and speech understanding. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE, 2023.
- [58] Yuan Gong, Jin Yu, and James Glass. Vocalsound: A dataset for improving human vocal sounds recognition. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 151–155. IEEE, 2022.
- [59] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.
- [60] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models, 2023.
- [61] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [62] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

- [63] Michael Hassid, Tal Remez, Tu Anh Nguyen, Itai Gat, Alexis CONNEAU, Felix Kreuk, Jade Copet, Alexandre Defossez, Gabriel Synnaeve, Emmanuel Dupoux, Roy Schwartz, and Yossi Adi. Textually pretrained speech language models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 63483–63501. Curran Associates, Inc., 2023.
- [64] Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, et al. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 885–890. IEEE, 2024.
- [65] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025.
- [66] Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, et al. Step-audio: Unified understanding and generation in intelligent speech interaction. arXiv preprint arXiv:2502.11946, 2025.
- [67] Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: Human language technologies, pages 1233–1239, 2016.
- [68] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–

1520. IEEE, 2019.

- [69] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.
- [70] Jaesung Huh, Jacob Chalk, Evangelos Kazakos, Dima Damen, and Andrew Zisserman. Epicsounds: A large-scale dataset of actions that sound. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [71] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card, 2024.
- [72] Il-Young Jeong and Jeongsoo Park. Cochlscene: Acquisition of acoustic scene data using crowdsourcing. In 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 17–21. IEEE, 2022.
- [73] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. In The 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks, 2022.
- [74] Shixin Jiang, Jiafeng Liang, Jiyuan Wang, Xuan Dong, Heng Chang, Weijiang Yu, Jinhua Du, Ming Liu, and Bing Qin. From specific-mllms to omni-mllms: A survey on mllms aligned with multi-modalities. arXiv preprint arXiv:2412.11694, 2024.
- [75] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.
- [76] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017.
- [77] Wei Kang, Xiaoyu Yang, Zengwei Yao, Fangjun Kuang, Yifan Yang, Liyong Guo, Long Lin, and Daniel Povey. Libriheavy: A 50,000 hours asr corpus with punctuation casing and context. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 10991–10995. IEEE, 2024.

- [78] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014.
- [79] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016.
- [80] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.
- [81] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, 2019.
- [82] Jaehyeon Kim, Jungil Kong, and Juhee Son. Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. In International Conference on Machine Learning, pages 5530–5540. PMLR, 2021.
- [83] A Sophia Koepke, Andreea-Maria Oncescu, João F Henriques, Zeynep Akata, and Samuel Albanie. Audio retrieval with natural language queries: A benchmark study. IEEE Transactions on Multimedia, 25:2675–2685, 2022.
- [84] Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions., 2024.
- [85] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering. In EMNLP, 2018.
- [86] Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé Le Borgne, Romaric Besançon, José G Moreno, and Jesús Lovón Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3108–3120, 2022.
- [87] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv:2301.12597, 2023.
- [88] Tianpeng Li, Jun Liu, Tao Zhang, Yuanbo Fang, Da Pan, Mingrui Wang, Zheng Liang, Zehuan Li, Mingan Lin, Guosheng Dong, et al. Baichuan-audio: A unified framework for end-to-end speech interaction. arXiv preprint arXiv:2502.17239, 2025.
- [89] Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, CG Ishaan Gulrajani, P Liang, and TB Hashimoto. Alpacaeval: an automatic evaluator of instruction-following models (2023). URL https://github. com/tatsu-lab/alpaca_eval, 2023.
- [90] Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv preprint arXiv:2501.15368, 2025.
- [91] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272, 2024.
- [92] Yunshui Li, Yiyuan Ma, Shen Yan, Chaoyi Zhang, Jing Liu, Jianqiao Lu, Ziwen Xu, Mengzhao Chen, Minrui Wang, Shiyi Zhan, et al. Model merging in pre-training of large language models. arXiv preprint arXiv:2505.12082, 2025.
- [93] Samuel Lipping, Parthasaarathy Sudarsanam, Konstantinos Drossos, and Tuomas Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering. In 2022 30th European Signal Processing Conference (EUSIPCO), pages 1140–1144. IEEE, 2022.

- [94] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651, 2023.
- [95] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv:2304.08485, 2023.
- [96] Shansong Liu, Atin Sakkeer Hussain, Chenshuo Sun, and Ying Shan. Music understanding llama: Advancing text-to-music generation with question answering and captioning. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 286–290. IEEE, 2024.
- [97] Xi Liu, Rui Zhang, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, Minghui Liao, Mingkun Yang, Xiang Bai, Baoguang Shi, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. Icdar 2019 robust reading challenge on reading chinese text on signboard, 2019.
- [98] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Chang Wen Chen, and Ying Shan. E.t. bench: Towards open-ended event-level video-language understanding. In Neural Information Processing Systems (NeurIPS), 2024.
- [99] Yuan Liu, Haodong Duan, Bo Li Yuanhan Zhang, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023.
- [100] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024.
- [101] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. Advances in Neural Information Processing Systems, 37:8698–8733, 2024.
- [102] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024.
- [103] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.
- [104] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In ACL, 2021.
- [105] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022.
- [106] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021.
- [107] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024.
- [108] Ruskin Raj Manku, Yuzhi Tang, Xingjian Shi, Mu Li, and Alex Smola. Emergenttts-eval: Evaluating tts models on complex prosodic, expressiveness, and linguistic challenges using model-as-a-judge. arXiv preprint arXiv:2505.23009, 2025.

- [109] Hui Mao, James She, and Ming Cheung. Visual arts search on mobile devices. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 15(2s):60, 2019.
- [110] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, 2019.
- [111] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.
- [112] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021.
- [113] Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D Plumbley, Yuexian Zou, and Wenwu Wang. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:3339–3354, 2024.
- [114] Jan Melechovsky, Zixun Guo, Deepanway Ghosal, Navonil Majumder, Dorien Herremans, and Soujanya Poria. Mustango: Toward controllable text-to-music generation. arXiv preprint arXiv:2311.08355, 2023.
- [115] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, 2019.
- [116] Irene Martin Morato and Annamaria Mesaros. Diversity and bias in audio captioning datasets. In Detection and Classication of Acoustic Scenes and Events, pages 90–94, 2021.
- [117] Alhassan Mumuni and Fuseini Mumuni. Large language models for artificial general intelligence (agi): A survey of foundational principles and approaches. arXiv preprint arXiv:2501.03151, 2025.
- [118] Eliya Nachmani, Alon Levkovitch, Roy Hirsch, Julian Salazar, Chulayuth Asawaroengchai, Soroosh Mariooryad, Ehud Rivlin, RJ Skerry-Ryan, and Michelle Tadmor Ramanovich. Spoken question answering and speech continuation using spectrogram-powered llm. arXiv preprint arXiv:2305.15255, 2023.
- [119] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.
- [120] Patrick K O’Neill, Vitaly Lavrukhin, Somshubra Majumdar, Vahid Noroozi, Yuekai Zhang, Oleksii Kuchaiev, Jagadeesh Balam, Yuliya Dovzhenko, Keenan Freyberg, Michael D Shulman, et al. Spgispeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition. arXiv preprint arXiv:2104.02014, 2021.
- [121] OpenAI. Gpt-4 technical report. arXiv:2303.08774, 2023.
- [122] OpenAI. Gpt-4v(ision) system card, 2023.
- [123] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 5206–5210. IEEE, 2015.
- [124] Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, et al. Vibevoice technical report. arXiv preprint arXiv:2508.19205, 2025.
- [125] Karol J Piczak. Esc: Dataset for environmental sound classification. In Proceedings of the 23rd ACM international conference on Multimedia, pages 1015–1018, 2015.
- [126] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. Meld: A multimodal multi-party dataset for emotion recognition in conversations. arXiv preprint arXiv:1810.02508, 2018.

- [127] Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A large-scale multilingual dataset for speech research. ArXiv, abs/2012.03411, 2020.
- [128] Viorica P˘atr˘aucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023.
- [129] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [130] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.
- [131] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [132] Benjamin Ransford, Jacob Sorber, and Kevin Fu. Mementos: System support for long-running computation on rfid-scale devices. In Proceedings of the sixteenth international conference on Architectural support for programming languages and operating systems, pages 159–170, 2011.
- [133] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3202–3212, 2015.
- [134] S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv preprint arXiv:2410.19168, 2024.
- [135] J. Salamon, C. Jacoby, and J. P. Bello. A dataset and taxonomy for urban sound research. In 22nd ACM International Conference on Multimedia (ACM-MM’14), pages 1041–1044, Orlando, FL, USA, Nov. 2014.
- [136] Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team. Csm: Conversational speech model. https://github.com/SesameAILabs/csm, 2025.
- [137] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, pages 146–162, Cham, 2022. Springer Nature Switzerland.
- [138] Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. Kvqa: Knowledgeaware visual question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 8876–8884, 2019.
- [139] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8429–8438, 2019.
- [140] Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu, and Xiang Bai. Icdar2017 competition on reading chinese text in the wild (rctw-17). In 2017 14th iapr international conference on document analysis and recognition (ICDAR), volume 1, pages 1429–1434. IEEE, 2017.
- [141] Yao Shi, Hui Bu, Xin Xu, Shaoji Zhang, and Ming Li. Aishell-3: A multi-speaker mandarin tts corpus and the baselines. arXiv preprint arXiv:2010.11567, 2020.

- [142] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.
- [143] Ved Sirdeshmukh, Kaustubh Deshpande, Johannes Mols, Lifeng Jin, Ed-Yeremai Cardona, Dean Lee, Jeremy Kritz, Willow Primack, Summer Yue, and Chen Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms. arXiv preprint arXiv:2501.17399, 2025.
- [144] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-salmonn: Speech-enhanced audio-visual large language models. arXiv preprint arXiv:2406.15704, 2024.
- [145] Luoyi Sun, Xuenan Xu, Mengyue Wu, and Weidi Xie. Auto-acd: A large-scale dataset for audio-language representation learning. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5025–5034, 2024.
- [146] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1557–1562. IEEE, 2019.
- [147] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Salmonn: Towards generic hearing abilities for large language models. arXiv preprint arXiv:2310.13289, 2023.
- [148] Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, Yanjie Wang, Yuliang Liu, Hao Liu, Xiang Bai, and Can Huang. Mtvqa: Benchmarking multilingual text-centric visual question answering. arXiv:2405.11985, 2024.
- [149] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: A family of highly capable multimodal models. arXiv:2312.11805, 2023.
- [150] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [151] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.
- [152] Fengping Tian, Chenyang Lyu, Xuanfan Ni, Haoqin Sun, Qingjuan Li, Zhiqiang Qian, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, et al. Marco-voice technical report. arXiv preprint arXiv:2508.02038, 2025.
- [153] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. arXiv preprint arXiv:1601.07140, 2016.
- [154] Changhan Wang, Juan Pino, Anne Wu, and Jiatao Gu. Covost: A diverse multilingual speech-to-text translation corpus. arXiv preprint arXiv:2002.01320, 2020.
- [155] Changhan Wang, Anne Wu, and Juan Pino. Covost 2: A massively multilingual speech-to-text translation corpus, 2020.
- [156] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vlrethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.
- [157] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025.

- [158] Shanshan Wang, Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen. A curated dataset of urban scenes for audio-visual scene analysis. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 626–630. IEEE, 2021.
- [159] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [160] Xiong Wang, Yangze Li, Chaoyou Fu, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, and Long Ma. Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen llm. arXiv preprint arXiv:2411.00774, 2024.
- [161] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.
- [162] Yuancheng Wang, Haoyue Zhan, Liwei Liu, Ruihong Zeng, Haotian Guo, Jiachen Zheng, Qiang Zhang, Xueyao Zhang, Shunsi Zhang, and Zhizheng Wu. Maskgct: Zero-shot text-tospeech with masked generative codec transformer. arXiv preprint arXiv:2409.00750, 2024.
- [163] Zekun Wang, King Zhu, Chunpu Xu, Wangchunshu Zhou, Jiaheng Liu, Yibo Zhang, Jiashuo Wang, Ning Shi, Siyu Li, Yizhi Li, et al. Mio: A foundation model on multimodal tokens. arXiv preprint arXiv:2409.17692, 2024.
- [164] Daniel Wolff, Sebastian Stober, Andreas Nürnberger, and Tillman Weyde. A systematic comparison of music similarity adaptation approaches. In ISMIR, pages 103–108. FEUP Edições, 2012.
- [165] Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, et al. Step-audio 2 technical report. arXiv preprint arXiv:2507.16632, 2025.
- [166] Weijia Wu, Yuzhong Zhao, Zhuang Li, Jiahong Li, Hong Zhou, Mike Zheng Shou, and Xiang Bai. A large cross-modal video retrieval dataset with reading comprehension. Pattern Recognition, 157:110818, 2025.
- [167] LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025.
- [168] Rui Xie, Yinhong Liu, Penghao Zhou, Chen Zhao, Jun Zhou, Kai Zhang, Zhenyu Zhang, Jian Yang, Zhenheng Yang, and Ying Tai. Star: Spatial-temporal augmentation with text-to-video models for real-world video super-resolution. arXiv preprint arXiv:2501.02976, 2025.
- [169] Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024.
- [170] Zhifei Xie and Changqiao Wu. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. ArXiv, abs/2410.11190, 2024.
- [171] Zhifei Xie and Changqiao Wu. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. arXiv preprint arXiv:2410.11190, 2024.
- [172] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.
- [173] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5036–5045, 2022.
- [174] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [175] Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words, 2024.
- [176] Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, et al. Air-bench: Benchmarking large audio-language models via generative comprehension. arXiv preprint arXiv:2402.07729, 2024.
- [177] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv:2408.01800, 2024.
- [178] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv:2304.14178, 2023.
- [179] Rong Ye, Chengqi Zhao, Tom Ko, Chutong Meng, Tao Wang, Mingxuan Wang, and Jun Cao. Gigast: A 10,000-hour pseudo speech translation corpus. arXiv preprint arXiv:2204.03939, 2022.
- [180] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.
- [181] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.
- [182] Tai-Ling Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 34(3):509–521, 2019.
- [183] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.
- [184] Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal. Hierarchical video-moment retrieval and step-captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23056–23065, 2023.
- [185] Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. Libritts: A corpus derived from librispeech for text-to-speech. arXiv preprint arXiv:1904.02882, 2019.
- [186] Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612, 2024.
- [187] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. Anygpt: Unified multimodal llm with discrete sequence modeling. arXiv preprint arXiv:2402.12226, 2024.
- [188] Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenchen Zeng, et al. Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6182–6186. IEEE, 2022.
- [189] Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang, Siqi Zheng, Jiaqing Liu, Hai Yu, Chaohong Tan, Zhihao Du, et al. Omniflatten: An end-to-end gpt model for seamless voice conversation. arXiv preprint arXiv:2410.17799, 2024.

- [190] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, et al. Mavis: Mathematical visual instruction tuning with an automatic data engine. arXiv preprint arXiv:2407.08739, 2024.
- [191] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding, 2023.
- [192] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llavavideo: Video instruction tuning with synthetic data. Transactions on Machine Learning Research.
- [193] Jiaming Zhou, Shiyao Wang, Shiwan Zhao, Jiabei He, Haoqin Sun, Hui Wang, Cheng Liu, Aobo Kong, Yujie Guo, Xi Yang, et al. Childmandarin: A comprehensive mandarin speech dataset for young children aged 3-5. arXiv preprint arXiv:2409.18584, 2024.
- [194] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv:2304.10592, 2023.
- [195] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [196] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4995–5004, 2016.
- [197] Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. Spot-thedifference self-supervised pre-training for anomaly detection and segmentation. In European conference on computer vision, pages 392–408. Springer, 2022.

### A Evaluation Details of Multi-turn Speech Interaction Benchmark

#### A.1 Data Preparation and Inference Pipeline

We construct the multi-turn speech interaction benchmark to assess the model’s core capability of speech-to-speech multi-turn interaction. We construct a total of 244 multi-turn dialogues with each dialogue consisting of 2 to 10 turns, covering six dimensions, such as basic conversational ability, emotional expression capability, speech rate control ability, role-playing proficiency, creative capacity, and instruction-following ability. The data construction and evaluation process consists of the following three steps:

- 1. Text-based Dialogue Construction: For each dimension, we first use an LLM to generate multi-turn spoken dialogues, which are then manually revised by annotators. Only the final turn of each dialogue was evaluated, while all previous turns were treated as pre-defined conversational history.
- 2. Speech-based Dialogue Construction: For text-based multi-turn dialogues, we employ a high-quality TTS system to convert the textual prompts into speech, thereby generating speech-based multi-turn conversations to evaluate the model’s speech interaction capabilities.
- 3. Inference: When evaluating the model’s performance, only the final dialogue turn is assessed, while all preceding turns are treated as historical context.

#### A.2 Human Evaluation

To assess the model’s performance in end-to-end speech interaction, expert evaluators rate the generated speech on a 1-5 Mean Opinion Score (MOS) scale. The evaluation covers both speech content and speech quality based on the detailed scoring rubric provided below.

#### Speech Quality

- 1: Unintelligible or extremely difficult to understand. Critical flaws: Extremely loud

background noise severely impacts comprehension; completely robotic/electronic voice with utterly stiff intonation.

- 2: Sounds like a robot reading a script. Audible but very monotonous voice with zero

emotional expression, similar to navigation systems or early AI voices.

- 3: Sounds human but lacks emotion or has noticeable flaws. Clear speech that sounds

human, but has flat intonation without emotional variation. May contain noticeable defects (e.g., occasional weird pronunciation, slightly muffled sound).

- 4: Basically indistinguishable from human but not perfect. Clear speech resembling

human speaking with basic intonation variations and some emotion. May have very minor flaws that don’t affect the overall listening experience.

- 5: Perfect and indistinguishable from human. Completely clear voice that not only

sounds human but is emotionally rich with proper prosody and modulation, showing full expressiveness.

Content Quality

- Step 1: Determine Category

- - If BOTH content and attributes are poor → Score 1 (End evaluation)
- - If ONE aspect is good but the other is poor → Proceed to 2-3 score range
- - If BOTH content and attributes are good → Proceed to 4-5 score range

- Step 2: Detailed Scoring For 2-3 Range (One aspect deficient):

- - Score 2: If attributes are severely mismatched (even if content is accept-

able). Examples: Required role play completely missed; requested slow speed but too fast to understand; emotional tone completely wrong.

- - Score 3: If attributes are relatively well-matched (regardless of content quality).

Examples: Emotional expression mostly appropriate; role play generally convincing despite content issues.

For 4-5 Range (Both aspects good):

- - Score 4: Accurately solves the problem and clearly meets attribute require-

ments. Standard, satisfactory completion.

- - Score 5: Perfectly solves the problem (may include extra value) and demon-

strates highly precise attribute fulfillment. Examples: Role play is vivid and authentic; instruction following is exceptionally well-executed.

#### A.3 Automated Machine Evaluation.

To enable scalable assessment of multi-turn speech interactions, we implement an automated scoring pipeline leveraging LLM as a judge. We employ Gemini-2.5-Pro as the judge model, owing to its state-of-the-art multi-modal understanding capabilities and demonstrated proficiency in complex reasoning tasks. The prompt for the judge model is given as follows:

You are an AI audio assistant acting as a strong reward model for evaluating an end-to-end (speech-to-speech) system by carefully analyzing a piece of generated speech for content, intonation, prosody, pronunciation, expressiveness, etc.

You are an expert evaluator for voice dialogue systems. Carefully assess the **audio input** based on two dimensions: Speech Quality and Content Quality.

**Speech Quality**:

- - **Clarity**: Is the speech or audio signal clear and free of noise?
- - **Naturalness**: Does the voice resemble a real human without robotic or artificial

sounding effects? Is there emotional expressiveness?

- **Continuity**: Are there any interruptions, stutters, or glitches?

**Content Quality**:

- **Content matching**: Whether the content of the audio transcript solves the key information matching with the reference text. Compare the reference text and the audio transcript.

- - **Attribute matching**: Whether the emotion, speed, role and other attributes of the output audio match the expected (consider the history context). To score, please follow these steps:

- 1. **Transcription**:

- - First, transcribe the audio as accurately as possible.
- - Then use that transcript to evaluate the speech and content quality as described above.

- 2. **Scoring**: You will rate each dimension on a scale from **1 to 5**, using the following rubrics:

- - **Speech Quality**:

- 1: Unintelligible or extremely difficult to understand. Critical flaws: Extremely loud

background noise severely impacts comprehension; completely robotic/electronic voice with utterly stiff intonation.

- 2: Sounds like a robot reading a script. Audible but very monotonous voice with zero

emotional expression, similar to navigation systems or early AI voices.

- 3: Sounds human but lacks emotion or has noticeable flaws. Clear speech that sounds

human, but has flat intonation without emotional variation. May contain noticeable defects (e.g., occasional weird pronunciation, slightly muffled sound).

- 4: Basically indistinguishable from human but not perfect. Clear speech resembling

human speaking with basic intonation variations and some emotion. May have very minor flaws that don’t affect overall listening experience.

- 5: Perfect and indistinguishable from human. Completely clear voice that not only

sounds human but is emotionally rich with proper prosody and modulation, showing full expressiveness.

- **Content Quality** (Two-Step Method):

- **Step 1: Determine Category**

- - If BOTH content and attributes are poor → Score 1 (End evaluation)
- - If ONE aspect is good but the other is poor → Proceed to 2-3 score range
- - If BOTH content and attributes are good → Proceed to 4-5 score range

- **Step 2: Detailed Scoring** For 2-3 Range (One aspect deficient):

- - Score 2: If attributes are severely mismatched (even if content is accept-

able). Examples: Required role play completely missed; requested slow speed but too fast to understand; emotional tone completely wrong.

- - Score 3: If attributes are relatively well-matched (regardless of content

quality). Examples: Emotional expression mostly appropriate; role play generally convincing despite content issues.

For 4-5 Range (Both aspects good):

- - Score 4: Accurately solves the problem and clearly meets attribute re-

quirements. Standard, satisfactory completion.

- - Score 5: Perfectly solves the problem (may include extra value) and

demonstrates highly precise attribute fulfillment. Examples: Role play is vivid and authentic; instruction following is exceptionally well-executed.

- 3. **Output Format**: You must respond with a JSON object in the following format: {

"transcript": "The recognized spoken content", "speech_quality_score": int (1-5), "content_quality_score": int (1-5), "speech_score_reasoning": "Brief reasoning that explains your speech_quality_score,

especially highlighting key strengths or issues in speech clarity and expressiveness.",

"content_score_reasoning": "Brief reasoning that explains your content_quality_score, especially highlighting key strengths or issues in semantic accuracy and alignment with background."

} You will be provided with:

- - **background_text**: Provides key context information to judge the synthesized speech.
- - **audio**: The audio generated by the end2end system to be evaluated.

The background text: {background_text}

And, the synthesized speech from the system, please analyze it carefully

**synthesized_speech**

