# arXiv:2409.18042v4[cs.CV]20Mar2025

[Figure 1]

## EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions

Kai Chen1∗, Yunhao Gou1,6∗, Runhui Huang2∗, Zhili Liu1,3∗, Daxin Tan3∗, Jing Xu4, Chunwei Wang3, Yi Zhu3, Yihan Zeng3, Kuo Yang3, Dingdong Wang4, Kun Xiang5, Haoyuan Li5, Haoli Bai3, Jianhua Han3, Xiaohui Li3, Weike Jin3, Nian Xie3, Yu Zhang6, James T. Kwok1, Hengshuang Zhao2, Xiaodan Liang5, Dit-Yan Yeung1, Xiao Chen3, Zhenguo Li3, Wei Zhang3, Qun Liu3, Jun Yao3, Lanqing Hong3†, Lu Hou3†, Hang Xu3† 1Hong Kong University of Science and Technology 2The University of Hong Kong 3Huawei Noah’s Ark Lab 4The Chinese University of Hong Kong 5Sun Yat-sen University 6Southern University of Science and Technology Project Page: https://emova-ollm.github.io/

#### Abstract

GPT-4o, an omni-modal model that enables vocal conversations with diverse emotions and tones, marks a milestone for omni-modal foundation models. However, empowering Large Language Models to perceive and generate images, texts, and speeches end-to-end with publicly available data remains challenging for the open-source community. Existing vision-language models rely on external tools for speech processing, while speech-language models still suffer from limited or totally without vision-understanding capabilities. To address this gap, we propose the EMOVA (EMotionally Omni-present Voice Assistant), to enable Large Language Models with end-to-end speech abilities while maintaining the leading vision-language performance. With a semanticacoustic disentangled speech tokenizer, we surprisingly notice that omni-modal alignment can further enhance visionlanguage and speech abilities compared with the bi-modal aligned counterparts. Moreover, a lightweight style module is introduced for the flexible speech style controls including emotions and pitches. For the first time, EMOVA achieves state-of-the-art performance on both the vision-language and speech benchmarks, and meanwhile, supporting omnimodal spoken dialogue with vivid emotions.

Figure 1. EMOVA is the very first omni-modal LLM with stateof-the-art performance on both vision-language and speech benchmarks simultaneously. See detailed results in Table 2.

telligent assistants that can see (i.e., perceiving fine-grained visual inputs), hear (i.e., understanding vocal instructions) and speak (i.e., generating vocal responses) simultaneously. Existing Multi-modal Large Language Models (MLLMs) mostly focus on two modalities only, either vision-language [1, 40] or speech-language [11, 87], demonstrating severe demands for omni-modal models with visual, language and speech abilities. How to empower Large Language Models (LLMs) to effectively process omni-modal data in an endto-end manner remains an open question.

#### 1. Introduction

OpenAI GPT-4o [70], a novel milestone for the omni-modal foundation models, has rekindled people’s attention on in-

Existing omni-modal LLMs [10, 19] generally build on Vision LLMs and integrate the speech modality by adopting a speech encoder like Whisper [74], which extracts continuous speech features, similar to how images are processed, to

∗Equal contribution, listed in alphabetical order by surname. †Corresponding authors: {honglanqing,houlu3,xu.hang}@

huawei.com

enable speech understanding. These models, however, still rely on external Text-to-Speech (TTS) tools for generating speech responses, limiting their ability to support real-time interactions. AnyGPT [93], on the contrary, opts for a fully discretization way, which first discretizes all data modalities (i.e., images, texts, and speeches), followed by omni-modal auto-regressive modeling. This enables AnyGPT to handle multiple modalities with a unified end-to-end framework, facilitating real-time interactions with the help of streaming decoding. However, the discrete vision tokenizer used by AnyGPT struggles to capture visual details, especially for high-resolution images, falling far behind its continuous counterparts on vision-language benchmarks. Furthermore, none of the existing works explore the speech style controls (e.g., emotions and pitches) with LLMs. Thus, our question arises: How to build an end-to-end omni-modal LLM enabling emotional spoken dialogue while maintaining stateof-the-art vision-language performance?

In this paper, we propose EMOVA (EMotionally Omnipresent Voice Assistant), a novel end-to-end omni-modal LLM with state-of-the-art vision-language and speech capabilities while supporting emotional spoken dialogue. Fig. 2 shows an overview of the model framework. A continuous vision encoder captures fine-grained visual details, while a discrete speech tokenizer and detokenizer empower end-toend speech understanding and generation. Specifically, the speech-to-unit (S2U) tokenizer tokenizes the input speech waveforms to discrete speech units, while the unit-to-speech (U2S) detokenizer reconstructs the speech waveforms from LLM’s output speech units. To seamlessly integrate speech modality with LLMs, we meticulously design a semanticacoustic disentangled speech tokenizer to decouple the semantic contents and acoustic styles of input speeches [77], where 1) semantic content (i.e., what it says) captures the semantic meanings of the input speeches, which is finally discretized and aligned with LLMs, while 2) acoustic style (i.e., how it says) captures the diverse speech styles (e.g., emotions and pitches). Utilizing the semantic-acoustic disentanglement of our speech tokenizer, we further introduce a lightweight style module to support spoken dialogue with vivid emotions and pitches. As in Sec. 4.1, this disentanglement design better facilitates the modality alignment among texts and speeches while maintaining flexibility for diverse speech style controllability and personalization.

With EMOVA’s end-to-end omni-modal framework, we empirically demonstrate publicly available bi-modal imagetext and speech-text data are sufficient for the omni-modal alignment with the text modality as a bridge, eliminating the need for omni-modal data (i.e., image-text-speech), which is usually scarce. Surprisingly, we find that the omni-modal alignment can further improve both the vision-language and speech capabilities via joint optimization, even when compared with their bi-modal aligned counterparts. Ultimately,

Speech Understand Gen. Emotion Vision Large Language Models

Method Visual Text

LLaVA [49] ✓ ✓ ✗ ✗ ✗ Intern-VL [10] ✓ ✓ ✗ ✗ ✗

Speech Large Language Models

Qwen-Audio [11] ✗ ✓ ✓ ✗ ✗ Mini-Omni [87] ✗ ✓ ✓ ✓ ✗ LLaMA-Omni [17] ✗ ✓ ✓ ✓ ✗

Omni-modal Large Language Models

VITA [19, 20] ✓ ✓ ✓ ✗ ✗ Ola [55] ✓ ✓ ✓ ✗ ✗ Any-GPT [93] ✓ ✓ ✓ ✓ ✗ Baichuan-Omni [43] ✓ ✓ ✓ ✓ ✗ EMOVA (ours) ✓ ✓ ✓ ✓ ✓

Table 1. Comparison of Multi-modal Large Language Models. EMOVA is the very first Omni-modal LLM capable of emotional spoken dialogue with state-of-the-art vision-language and speech capabilities simultaneously. “Gen.” stands for Generation.

only a small amount of omni-modality samples are required to teach the model to respond in the desired format. For the first time, EMOVA obtains state-of-the-art results on both vision-language and speech benchmarks (see Table 2). The main contributions of this work contain three parts:

- 1. We propose EMOVA, a novel end-to-end omni-modal LLM that can see, hear and speak. A continuous vision encoder with a semantic-acoustic disentangled speech tokenizer is adopted for seamless omni-modal alignment and diverse speech style controllability.
- 2. We introduce an efficient text-centric omni-modal alignment which can further enhance the vision-language and speech abilities, surpassing their bi-modal aligned counterparts (i.e., image-text only and speech-text only).
- 3. For the first time, EMOVA obtains state-of-the-art comparable results on both the vision-language and speech benchmarks simultaneously and further supports flexible spoken dialogues with vivid emotions.

#### 2. Related Work

Vision Large Language Models (VLLMs) integrate the vision modality into LLMs [7, 81], enabling the advanced understanding and reasoning over visual instructions [1, 26, 27, 49]. Recent VLLM works can be categorized into three directions, 1) Vision encoders [5, 6, 71] are enhanced and aggregated for robust representations [45, 46, 80]. 2) Highresolution methods are proposed to overcome the fixed resolution of pre-trained vision encoders (e.g., 336 × 336 for CLIP [73]), enabling LLMs to perceive fine-grained visual information [14, 31, 48, 58]. 3) High-quality instruction data is essential for VLLMs to generate accurate and wellformed responses [10, 36, 40]. Besides achieving state-ofthe-art vision-language performance, we further introduce speech understanding and generating abilities to EMOVA.

[Figure 2]

Speech Decoder 𝒅𝒅(⋅,⋅) Style Encoder 𝒈𝒈(⋅)

Output Speech 𝒀𝒀𝑺𝑺𝒐𝒐

|⋮|
|---|

Style Prototypes

[Figure 3]

###### EMOVA Language Model 𝒇𝒇(⋅)

Style Label

| |
|---|

Quantizer 𝒒𝒒(⋅)

Projector 𝒑𝒑(⋅)

Semantic-acoustic Disentanglement

[Figure 4]

[Figure 5]

What color is the cat?

###### Vision Encoder

𝒗𝒗 ⋅

Speech Encoder 𝒔𝒔(⋅)

Input Image 𝑿𝑿𝑽𝑽

Input Text 𝑿𝑿𝑻𝑻

Input Speech 𝑿𝑿𝑺𝑺

- Figure 2. Model architecture of EMOVA. The vision encoder extracts continuous visual features, which are projected into the textual embedding space as visual tokens, while the input speech is encoded and quantized into discrete speech units. Given the omni-modal inputs, EMOVA can generate both textual and speech responses with vivid emotional controls. Check Sec. 3 for more architectural details.

[Figure 6]

HUAWEI CONFIDENTIAL 内部资料 注意保密

[Figure 7]

Speech Large Language Models (SLLMs) empower the speech interaction with LLMs. Continuous SLLMs [11, 84] adopt the speech encoders [74] to extract continuous speech embeddings for LLM, which, however, only support speech understanding, relying on external TTS modules for speech generation, and therefore, hampering real-time interaction. Discrete SLLMs [95], instead, first discretize speech signals with speech tokenizers, followed by auto-regressive modeling. Recent works [17, 87] further combine the continuous speech encoders with discrete speech tokenizers for better results. Although effective, none of the existing works explore speech style controllability in SLLMs (e.g., emotions and pitches), which is essential for spoken dialogue.

p(·) as HV = p(EV ). As for the speech input XS, a Speechto-Unit (S2U) procedure is required. Specifically, XS first goes through a speech encoder s(·) as ES = s(XS), which is then discretized by the quantizer q(·) as US = q(ES). The LLM f(·) is then trained to compute the joint probability of the output texts UoT and speech units UoS as

L

P(xi|UoT,<i,UoS,<i,Uomni),

P(UoT,UoS|Uomni) =

i=1

(1) where xi ∈ UoT ∪ UoS, L = |UoT| + |UoS| and Uomni = UT ∪US∪HV , which stands for the omni-modal inputs. The output response units UoS are then recovered into the output speech waveform YoS via a Unit-to-Speech (U2S) decoder d(·,·) with an emotion style embedding Eostyle to realize the vivid emotional spoken dialogue controllability (Sec. 3.2).

Omni-modal Large Language Models support visual, text, and speech abilities with a unified architecture. Similar to continuous SLLMs, InternOmni [10] and VITA [19] connect a speech encoder with VLLMs, supporting speech understanding only. Instead, AnyGPT [93] proposes a unified architecture to discretize and conduct auto-regressive modeling for image, text, and audio simultaneously, which, however, suffers from inevitable information loss brought by discretization, especially for the high-resolution visual inputs. Our EMOVA is the very first unified Omni-modal LLM with state-of-the-art vision-language and speech performance at the same time.

LLM. We utilize the Qwen-2.5 [79] model families as the base LLMs of EMOVA with three configurations (i.e., 3B, 7B, and 72B) for usage under different budgets.

Vision encoder and projector. We use the QwenViT [78] as the visual encoder v(·) with an MLP vision projector p(·) with a 4× downsample rate for all variants of EMOVA.

###### 3.2. Speech Tokenization

#### 3. Architecture

Speech-to-unit (S2U) tokenizer. Following [77], we use the SPIRAL [32] architecture for the speech encoder s(·) to capture both phonetic and tonal information, which is then discretized by the quantizer q(·) with finite scalar quantization (FSQ) [64]. The size of the speech codebook is 4,096, while the sample rate is 25 tokens per second. Once discretized, the speech modality can be integrated into LLMs by concatenating the text vocabulary and speech codebook.

###### 3.1. Formulation

Denote LLM as f(·) and text, visual and speech inputs as XT, XV and XS, respectively. XT is converted to discrete tokens UT via a text tokenizer [21], while the XV is first encoded with a vision encoder v(·) as EV = v(XV ), and then projected into the textual embedding space with a projector

Joint Joint entangled VL Speech Speech VL VL Speech

45

80

- 92%

- 93%

- 94%

- 95%

- 96%

- 97%

40

70

35

60

30

50

25

40

ChartQA-test MathVista InfoVQA-val MM-Vet

MMB SQA DocVQA-val

1-WER%

- Figure 3. Comparison between different omni-modal alignment paradigms. 1) Joint training demonstrates consistent improvements over VL and Speech, suggesting that omni-modal alignment can be beneficial across modalities. 2) Joint training outperforms both VL→Speech and Speech→VL, revealing that joint training is more superior and efficient than sequential training. 3) Joint is superior to Joint-entangled, highlighting the effectiveness of the semantic-acoustic disentanglement, as discussed in Sec. 3.2.

Our S2U tokenizer provides the following advantages: 1) Data efficiency: after pre-training on large-scale unlabeled speech data, it requires only a small amount of speech-text pair data for easy adaptation. 2) Bilingual: the speech codebook is shared among languages (i.e., English and Chinese), sharing unit modeling capabilities across languages. Check more training details and comparisons in Appendix A.1.

Our U2S detokenizer is pre-trained on LibriTTS [92] and AISHELL-1 [2] and subsequently fine-tuned on synthetic style-rich speech data. Due to the scarcity of real-life stylerich data, we utilize TTS tools [16] to synthesize the speech samples diverse in genders, pitches, and emotions. As for the style prototypes, Emotion2Vec [59] is adopted to select the most representative samples with the highest confidence in conveying the desired style. Our empirical results reveal that even one representative style reference speech has been sufficient to control the speech styles flexibly and precisely. Check Appendix A.2 for more details.

Semantic-acoustic disentanglement. To align the speech units seamlessly with the highly semantic embedding space of LLMs, we opt for decoupling the semantic contents and acoustic styles of input speeches. Given input speechs XS, both semantic embedding Esemantic and style embeddings Estyle are extracted separately as

#### 4. Training Omni-modal LLMs

To achieve omni-model alignment, it is ideal to use largescale omni-modal image-text-speech data, which, however, is either without reach due to copyrights [67] or limited in quality [65]. An alternative is to use the existing image-text data with the TTS-synthesized speeches, which is not only computationally expensive but also hampers data diversity, as most TTS tools generate speeches in similar patterns. Recent works [10, 19] choose to integrate the speech modality into a well-structured VLLM via a sequential training manner with bi-modal alignment data. However, the relationships among modalities and how to effectively leverage multiple bi-modal alignment datasets remain unclear.

{Esemantic,Estyle} = ES = s(XS). (2)

Only Esemantic is quantified by q(·) to generate the speech units US. By using different Estyle while maintaining the same Esemantic, we can easily control the recovered speech styles without disturbing the semantic contents of recovered speeches. Moreover, the disentanglement facilitates modality alignment among speeches and texts, as later in Sec. 4.1.

Unit-to-speech (U2S) detokenizer with style controls. Building on VITS [35], our U2S detokenizer adopts a conditional VAE architecture (cf., Fig. 7). To achieve flexible style controls, we utilize the semantic-acoustic disentanglement of our S2U tokenizer (as discussed above) and adopt a novel style embedding to control the speech styles (e.g., genders, emotions, and pitches). Specifically, the LLM f(·) is trained to generate both the output speech units UoS and a style label. The speech units UoS are converted to the unit embeddings Eosemantic, while the style label is utilized to generate a unique style prototype Eostyle. Both Eosemantic and Eostyle are taken as inputs to speech decoder d(·,·) to synthesize output speeches YoS = d(Eosemantic,,Eostyle).

In this work, we explore omni-modal text-centric alignment by utilizing the publicly available bi-modal alignment datasets, including image-text (e.g., image captioning) and speech-text (e.g., ASR and TTS) datasets. With text modality as a bridge, our EMOVA ultimately becomes a unified system capable of understanding and generating multiple modalities in a coherent and integrated manner. In Sec. 4.1, we first explore the following three questions:

- 1. Does the integration of the speech modality conflict with the vision-language capabilities?
- 2. Is sequential alignment of multiple modalities optimal?

System:

[Figure 8]

Please provide the emotion, pitch and speech units for the assistant response.

Emotion should be chosen from ["neutral", "happy", "sad",

"angry"].

Pitch should be chosen from ["low", "normal", "high"].

Your output should be in json format.

user question speech:

User:

[Figure 9]

Can you write a poem based on the image?

<|speech_1785|> <|speech_1726|> <|speech_1653|> <|speech_3120|> <|speech_3128|>

<|speech_361|>…….<|speech_298|>

Assistant:

{"assistant response emotion": “sad", "assistant response pitch": “low", "assistant response speech" "<|speech_1053|><|speech_3445|>……<|speech_4024|><|speech_4026|><|speech_1269|>"}

[Figure 10]

Leaves fall, one by one, On the bench, where no one’s

[Figure 11]

been. Autumn's touch, a melancholy scene, A season of

change, a time unseen……

- Figure 4. Demonstration of EMOVA omni-modal instruction tuning. 1) To support emotional spoken dialogues, EMOVA is trained to explicitly select speech style labels with output speech units. 2) For ease of parsing, the data elements are organized in the JSON format.

- 3. How to represent speech modality to foster omni-modal alignment?

We then discuss the omni-modal instruction tuning pipeline and the overall training paradigm of EMOVA in Sec. 4.2 and Sec. 4.3, respectively.

- 4.1. Omni-modal Text-Centric Alignment

Settings. We consider the following omni-modal training paradigms: 1) VL→Speech conducts the image-text alignment first followed by speech-unit-text alignment using the full speech data with 10% of the image-text alignment data to prevent catastrophic forgetting, similarly with [10, 19]. 2) Speech→VL instead performs speech-unit-text alignment first and then aligns images with texts using 10% of the speech unit-text data and full image-text data. 3) Joint aligns both modalities simultaneously. Note that unless otherwise specified, we utilize the S2U tokenizer introduced in Sec. 3.2 to extract speech units for all speech data, which effectively disentangles the semantic and acoustic features. 4) Joint-entangled derives the speech units using HuBERT [30], which does not achieve semantic-acoustic disentanglement effectively with only K-means clustering. 5) VL and Speech only align vision and speech modalities with texts, respectively, serving as bi-modal baselines (see Appendix B.1 for more details).

Evaluation. For speech abilities, we evaluate the aligned model’s performance on the ASR task of LibriSpeech [72], while for the vision-language, we fine-tune the model with a small amount of high-quality visual instruction data (i.e.,

665K SFT data from the ShareGPT4V [9]) and evaluate the fine-tuned model on common vision-language benchmarks. Check Appendix C for evaluation details. Fig. 3 shows the comparison among different paradigms on vision-language (left and middle) and ASR (right, where we report the 1 − WER value for better readability) benchmarks, from which we can derive the following observations:

- Observation 1: image-text and speech-unit-text data benefit each other. Contrary to the common assumption that multiple modalities might compete and create conflicts, we notice that introducing additional modalities is actually beneficial. As in Fig. 3, Joint consistently surpasses both VL and Speech across vision-language and speech benchmarks. Moreover, even models aligned sequentially, (i.e., VL→Speech and Speech→VL, which are typically prone to catastrophic forgetting, demonstrate superior performance on most vision-language tasks. We speculate that the requirement to align multiple modalities with text leads to more robust representations, which in turn generalize better across different downstream tasks. This finding aligns with ImageBind [25], where joint alignment of audio and depth with images results in improved performance.
- Observation 2: semantic-acoustic disentanglement benefits omni-modal alignment. We find that 1) Joint outperforms Joint-entangled on vision-language benchmarks, and 2) in the speech tasks, Joint maintains significant advantages over its entangled counterpart. This can be attributed to the semantic-acoustic disentanglement which makes speech units more analogous to languages.

EMOVA EMOVA EMOVA Gemini GPT- GPT- Whisper Mini- VITA VITA Baichuan3B 7B 72B Pro 1.5 4V 4o Large Omni2 8x7B 1.5 Omni-7B

Benchmarks

|MME 2175 2317 2402 MMBench 79.2 83.0 86.4 SEED-Image 74.9 75.5 76.6 MM-Vet 57.3 59.4 64.8 RealWorldQA 62.6 67.5 71.0<br><br>|- 1927 2310<br>- 75.0 83.4<br>- 71.6 77.1<br>- 67.7 -<br><br><br>68.7 61.4 75.4|- -<br>- -<br>- -<br>- -<br>- -<br><br><br>|2097 2311 2187<br><br>71.8 76.6 76.2<br>72.6∗ 74.2 74.1 41.6 51.1 65.4<br><br><br>59.0∗ 66.8 62.6|
|---|---|---|---|
|TextVQA 77.2 78.0 81.4 ChartQA 81.5 84.9 88.7 DocVQA (test) 93.5 94.2 95.9 InfoVQA (test) 71.2 75.1 83.2 OCRBench 803 814 843<br><br>|73.5 77.4 81.3 78.5 85.7 86.5 88.4 92.8 72.7 - -<br><br>- 656 736|- -<br>- -<br>- -<br>- -<br>- -<br>|71.8∗ 74.9 74.3 76.6∗ 79.6 79.6<br><br>- - -<br>- - -<br><br><br>678 752 700|
|AI2D 78.6 81.7 85.8 ScienceQA-Img 92.7 96.4 98.2 MMMU 45.8 49.8 59.7<br><br>|80.3 78.2 84.6 - 75.7 58.5 56.8 69.2|- -<br>- -<br>- -<br>|73.1 79.3 -<br><br>- - 47.3 52.1 47.3<br><br>|
|MathVista 62.6 65.5 69.9 Mathverse 31.4 40.9 50.0<br><br>|52.1 49.9 63.8 - 33.6 -|- -<br>- -<br><br><br>|44.9 66.2 51.9 - - -|
|Librispeech (WER↓) 5.4 4.1 2.9<br><br>|- - -|3.0 4.8|3.4 8.1 -|

- Table 2. Comparison on vision-language and speech benchmarks. 1) EMOVA outperforms GPT-4o/4V and Gemini Pro 1.5 on 11 of the 15 vision-language benchmarks, providing a powerful open-sourced alternative. 2) Meanwhile, our EMOVA achieves state-of-the-art performance on Librispeech, surpassing its speech and omni-modal counterparts significantly. ∗: reported by [43].

Observation 3: sequential alignment is not optimal. We notice that Joint consistently outperforms its sequential counterparts (i.e., VL→Speech and Speech→VL) on both vision-language and speech benchmarks, probably due to catastrophic forgetting when integrating a new modality. In light of these observations, we have chosen to pursue the ultimate alignment strategy that simultaneously aligns image-text and speech-unit-text for EMOVA, which offers two important benefits, 1) it fosters the mutual enhancement among vision-language and speech, and 2) it avoids catastrophic forgetting during sequential alignment.

- 4.2. Omni-modal Instruction Tuning After the omni-modal text-centric alignment in Sec. 4.1, the model learns the fundamental vision-language (e.g., captioning) and speech capabilities (e.g., ASR and TTS). However, instruction tuning is essential to better follow complicated user instructions and respond with vivid emotions. Emotion-enriched instruction data synthesis. Due to the scarcity of omni-modal instruction data (i.e., dialogues involving images, speeches, and texts simultaneously), we opt for synthesizing omni-modal instruction data from existing text and visual instruction datasets. First, we select instruction data suitable for the vocal expression by filtering out the non-vocal data (e.g., code and mathematical formulas). Second, we clean the selected data to be more vocal by removing text formatting elements (e.g., ∗∗ and \n\n). We then obtain style labels for the remaining dialog contexts, including genders (male, female), pitches (normal, low, high), and emotions (happy, sad, angry,

neutral), resulting in totally 24 different speech styles. The style labels are generated by prompting GPT-4o 1 to make reasonable inferences given the dialogue context. Finally, we convert the textual instructions and responses into speeches utilizing the latest TTS tools (i.e., CosyVoice [16] and Azure AI Speech), and the style labels are used to control the style of synthesized speech data. To further improve the diversity of the data, each instruction is synthesized by randomly choosing one of the 39 available speakers. Finally, we gather 120K speech-text and 110K speech-image data pairs. Check more details in Appendix B.2.

Data organization and the chain of modality. The omni-modal instruction data can be represented as Domni = {(xV ,uS,xoT,costyle,uoS)i}Ni=1, where the input consists of the optional queried image xV and the speech units of the instruction uS, while the output consists of the textual response xoT, the predicted speech style labels costyle, and the output speech unit uoS. Note that we train EMOVA to explicitly select styles (e.g., emotions and pitches), which are utilized to determine the corresponding style embedding for the U2S detokenizer (Sec. 3.2). Furthermore, since directly generating speech responses is challenging, we decompose the speech response procedure into three primary steps: 1) recognizing user instructions into texts; 2) generating textual responses based on the recognized instructions; 3) generating the style labels and response speech units based on the textual responses. For ease of parsing during deployment, the target outputs are formatted as JSON, as in Fig. 4.

1https://chatgpt.ust.hk

Text response Style Categorization Recognition/Synthesis

Datasets End-to-end↑

Unit In Text In Emotion Pitch WER/CER↓ TTS-WER/CER↓

Speech-Image-EN 7.45 7.56 7.95 82.50 97.70 2.40 3.20 Speech-Text-EN 6.85 6.90 7.38 81.20 84.70 6.90 2.90 Speech-Image-ZH 6.48 7.02 6.82 77.60 95.90 1.70 12.00 Speech-Text-ZH 5.25 5.58 6.60 80.90 93.20 10.70 12.20

- Table 3. Evaluation of EMOVA-7B on Speech Dialogue. By default, we evaluate on the corresponding test set of the evaluated datasets.
- 4.3. Overall Training Paradigm Inspired by [9], a three-stage training paradigm is adopted,

- • Stage-1: Vision-language pre-alignment. The purpose is to align the visual features into the embedding space of LLMs. Only the vision projector p(·) is trained.
- • Stage-2: Omni-modal text-centric alignment. This stage performs vision-language and speech-language alignment jointly. We train the LLM f(·), vision projector p(·), and the deeper half of vision encoder v(·) layers.
- • Stage-3: Omni-modal instruction tuning. We organize different datasets with various types of instructions to learn generalization across tasks, as detailed in Sec. 5.1.

- 5. Experiments

###### 5.2. Comparison to SOTA Models

Experimental results are provided in Table 2. We compare a wide range of state-of-the-art VLLMs, including Gemini Pro 1.5 [75], GPT-4V [69], GPT-4o [70], together with the Speech LLM (i.e., Mini-Omni2 [87]) together with the ASR expert Whisper-Large [74], and the omni-modal LLMs (i.e., VITA-8x7B [19], VITA-1.5 [20] and Baichuan-Omni [43]).

Comparison with SOTA VLLMs. As an omni-modal model, EMOVA obtains comparable performance with the state-of-the-art VLLMs on multiple vision-language benchmarks, while showing superior proficiency in solving math problems needing precise visual content interpretation. Our EMOVA-7B surpasses GPT-4V by +7.3 on MathVerse, and our EMOVA-72B exceeds GPT-4o by +6.1 on MathVista. On 11 out of the 15 benchmarks, our EMOVA outperforms both GPT-4o/4V and Gemini Pro 1.5, providing a powerful open-sourced alternative.

###### 5.1. Training configuration

- Stage-1. In this stage, we only train the parameters of the vision projector p(·) for vision-language pre-alignment with the LCS-558K dataset [49], with the high-resolution imageslicing strategy [48] adopted.
- Stage-2. We assemble a unified dataset with 7.4M samples for both the image-text and speech-text alignment, as summarized in Fig. 9. Specifically, we utilize pre-training datasets from ShareGPT4V [9], ALLaVA [4] (both the original English version and the Chinese version translated on our own), and ShareGPT-4o [12] for general perception, while for the OCR capabilities, we leverage SynthDog [34], MMC-Alignment [47], K12 Printing, and the UReader Text Reading subset [89]. Moreover, we adopt the 2,000 hours of ASR and TTS data from LibriSpeech [72] and AISHELL-2 [15] for speech-text alignment, and to preserve the language capabilities of LLMs, we further incorporate the text-only data from Magpie Pro [88]. Check more details in Fig. 9.
- Stage-3. We collect the EMOVA-SFT dataset consisting of 4.4M multi-task omni-modal samples (see Fig. 8). We start by gathering high-quality open-sourced visual instruction datasets, including ShareGPT4V [9], InternVL [10], Meteor [38], Idefics-2 [36], Cambrian [80], and LLaVAOnevision [40], followed by quality checking, re-formatting all data samples with a unified template, and removing the duplicated data. For speech, we include the training split of EMOVA omni-model instruction data (cf., Sec. 4.2), with 10% of speech alignment datasets to maintain ASR and TTS performance. We train with 128 Ascend 910B (64GB) NPUs in parallel (check more details in Table 5).

Comparison with SOTA omni-modal LLMs. Under the similar 7B capacity, EMOVA-7B is 220 points higher than VITA on MME, surpassing VITA by 20.6% on OCRBench (814 vs. 678). Moreover, it surpasses Baichuan-Omni-7B, a more recent Omni-modal LLM, on nearly all the evaluated benchmarks, underscoring the effectiveness of our approach and potential to push the boundary of omni-modal abilities.

Moreover, ASR results show that EMOVA significantly outperforms the most recent omni-modal model VITA, even surpassing its SLLM counterparts Mini-Omni2 and ASRspecialized Whisper-Large, showing the effectiveness of the semantic-acoustic disentanglement and omni-modal mutual benefits. We further report TTS results in Table 6, and provide the qualitative examples in Fig. 5 and Appendix G. For the first time, EMOVA obtains state-of-the-art results on both the speech and vision-language benchmarks.

###### 5.3. Evaluation of Emotion-Rich Spoken Dialogue

In this section, we evaluate the end-to-end spoken dialogue abilities of EMOVA-7B. As stated in Sec. 4.2, the model takes an input image xV and user instructions in the form of speech units uS, and outputs text responses, style labels, and corresponding speech units. To ensure comprehensive evaluation, we propose the following evaluation metrics:

1. End-to-end spoken dialogue score assesses the model’s dialogue performance based on the generated speeches, with a score ranging from 0 to 10, reporting the average.

[Figure 12]

Figure 5. EMOVA engages in omni-modal emotional spoken dialogue expressing sadness.

- 2. Unit-input-text-output score focuses on the quality of the textual responses of LLM when the inputs are speech units, bypassing errors from speech synthesis.
- 3. Text-input-text-output score inputs the ground-truth user instruction texts and evaluates the model’s text outputs. This helps disentangle the impact of speech recognition errors and eliminates the effect of JSON format.
- 4. ASR and TTS evaluate how accurately the model recognizes the speech units and how effectively it generates speech units from text. See Appendix D for more details.
- 5. Style label classification accuracy evaluates the accuracy in selecting the appropriate speech style labels.
- 6. Style controllablity assesses the controllability of U2S detokenizer with the given conditional style labels using the confusion matrix comparing the generated and recognized style labels. See Appendix D for more details.

Due to the lack of emotionally rich spoken dialogue evaluation datasets, we split a test set from our synthesized omnimodal instruction-tuning data (Sec. 4.1). GPT-4o is used for automated evaluation. See details in Appendix D.

Results. Table 3 the spoken dialogue performance.

- (i) By comparing the end-to-end dialogue score with the

unit-input-text-output score, we notice that the two scores are closely aligned, with a maximum gap of only 0.33, except for Speech-Image-ZH. TTS-WER/CER is generally low for English, revealing that EMOVA can synthesize accurate speech based on textual responses, which, however, is harder for Chinese, which we attribute to its complexity. It includes tasks such as generating poetries and answering riddles, resulting in more intricate responses.

- (ii) Comparing the unit-input-text-output score with the

text-input-text-output score, we notice that their differences correlate with the ASR performance of the speech instructions, especially for Speech-Text-EN and Speech-Text-ZH, which involve more complex instructions.

Our EMOVA-7B reports inferior ASR performance (6.9 and 10.7, respectively) compared to other datasets (2.4 and 1.7). Consequently, when we replace speech instructions with ground-truth transcriptions, EMOVA shows significant improvements from unit-input to text input score. On

EN_female

1.0

|[Figure 13]| |
|---|---|
| | |
| | |
| | |
| | |

[Figure 14]

0.58 0.3 0.12 0

neutral

0.8

0.12 0.88 0 0

happy

0.6

condition

0.4

0 0 1 0

sad

0.2

0.12 0 0.04 0.84

angry

0.0

happy

angry

sad

neutral

recognized

Figure 6. Confusion matrix between the generated and recognized emotions. The emotions generated by our U2S detokenizer are recognized with high probability. Best view with zooming in.

the contrary, for datasets with accurate ASR performance, the results are quite similar, suggesting EMOVA retains robust dialogue capabilities when using the JSON format.

(iii) Examining the classification accuracy of style labels, we find that EMOVA performs satisfactorily in classifying emotions and pitches during speech conversations, achieving an accuracy of over 75%. The confusion matrix comparing the conditional and recognized emotion labels is shown in Fig. 6. The results indicate that the four emotions are recognized with high probabilities, with three achieving over 80% accuracy. This demonstrates that our U2S detokenizer effectively controls common emotions, endowing the synthesized speech with vivid emotional expression.

#### 6. Conclusion

Our work builds EMOVA, a novel end-to-end omni-modal large language model that effectively aligns vision, speech, and text simultaneously. With text as a bridge, we show that omni-modal alignment is achievable without relying on omni-modal image-text-speech data, meanwhile, enhancing both vision-language and speech abilities. For the first time, EMOVA achieves state-of-the-art performance on both vision-language and speech benchmarks, setting a new standard for versatile omni-modal interactions.

Acknowledgments. We gratefully acknowledge supports of MindSpore, CANN (Compute Architecture for Neural Networks) and Ascend AI Processor used for this research. This work has been made possible by a Research Impact Fund project (RIF R6003-21) and a General Research Fund project (GRF 16203224) funded by the Research Grants Council (RGC) of the Hong Kong Government. This work was partially supported by the Research Grants Council of the Hong Kong Special Administrative Region (Grants C7004-22G-1 and 16202523) and the Joint Centre for Artificial Intelligence (Grant FB453). This work is supported by National Key Research and Development Program of China (2024YFE0203100) and National Natural Science Foundation of China (No. 62441615, 62201484 and 62136005).

#### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1, 2
- [2] Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. AISHELL-1: An open-source mandarin speech corpus and a speech recognition baseline. In O-COCOSDA, 2017. 4, 13, 14
- [3] Guoguo Chen, Shuzhou Chai, Guanbo Wang, Jiayu Du, WeiQiang Zhang, Chao Weng, Dan Su, Daniel Povey, Jan Trmal, Junbo Zhang, et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. arXiv preprint arXiv:2106.06909, 2021. 13
- [4] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 7
- [5] Kai Chen, Lanqing Hong, Hang Xu, Zhenguo Li, and Dit-Yan Yeung. Multisiam: Self-supervised multi-instance siamese representation learning for autonomous driving. In ICCV, 2021. 2
- [6] Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, and Dit-Yan Yeung. Mixed autoencoder for self-supervised visual representation learning. In CVPR, 2023. 2
- [7] Kai Chen, Chunwei Wang, Kuo Yang, Jianhua Han, Lanqing Hong, Fei Mi, Hang Xu, Zhengying Liu, Wenyong Huang, Zhenguo Li, Dit-Yan Yeung, Lifeng Shang, Xin Jiang, and Qun Liu. Gaining wisdom from setbacks: Aligning large language models via mistake analysis. arXiv preprint arXiv:2310.10477, 2023. 2
- [8] Kai Chen, Enze Xie, Zhe Chen, Lanqing Hong, Zhenguo Li, and Dit-Yan Yeung. Integrating geometric control into text-to-image diffusion models for high-quality detection data generation via text prompt. arXiv preprint arXiv:2306.04607, 2023. 18
- [9] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. ShareGPT4V:

- Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 5, 7, 14, 18
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024. 1, 2, 3, 4, 5, 7
- [11] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024. 1, 2, 3
- [12] Erfei Cui, Yinan He, Zheng Ma, Zhe Chen, Hao Tian, Weiyun Wang, Kunchang Li, Yi Wang, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Yali Wang, Limin Wang, Yu Qiao, and Jifeng Dai. ShareGPT-4o: Comprehensive multimodal annotations with GPT-4o, 2023. 7
- [13] Alexandre D´efossez, Laurent Mazar´e, Manu Orsini, Am´elie Royer, Patrick P´erez, Herv´e J´egou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for realtime dialogue. http://kyutai.org/Moshi.pdf,

2024. 18

- [14] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-xcomposer24khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. arXiv preprint arXiv:2404.06512, 2024. 2
- [15] Jiayu Du, Xingyu Na, Xuechen Liu, and Hui Bu. Aishell2: Transforming mandarin asr research into industrial scale. arXiv preprint arXiv:1808.10583, 2018. 7, 17
- [16] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. CosyVoice: A scalable multilingual zero-shot textto-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024. 4, 6, 14
- [17] Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. LLaMA-Omni: Seamless speech interaction with large language models. arXiv preprint arXiv:2409.06666, 2024. 2, 3
- [18] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2024. 17
- [19] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. VITA: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024. 1, 2, 3, 4, 5, 7, 18
- [20] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Yangze Li, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025. 2, 7
- [21] Philip Gage. A new algorithm for data compression. The C Users Journal, 1994. 3
- [22] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view

- generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601, 2023. 18
- [23] Ruiyuan Gao, Kai Chen, Zhihao Li, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrive3d: Controllable 3d generation for any-view rendering in street scenes. arXiv preprint arXiv:2405.14475, 2024.
- [24] Ruiyuan Gao, Kai Chen, Bo Xiao, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrivedit: High-resolution long video generation for autonomous driving with adaptive control. arXiv preprint arXiv:2411.13807, 2024. 18
- [25] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In CVPR, 2023. 5
- [26] Yunhao Gou, Zhili Liu, Kai Chen, Lanqing Hong, Hang Xu, Aoxue Li, Dit-Yan Yeung, James T Kwok, and Yu Zhang. Mixture of cluster-conditional lora experts for vision-language instruction tuning. arXiv preprint arXiv:2312.12379, 2023. 2
- [27] Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T Kwok, and Yu Zhang. Eyes closed, safety on: Protecting multimodal llms via image-to-text transformation. arXiv preprint arXiv:2403.09572, 2024. 2
- [28] Yunhao Gou, Hansi Yang, Zhili Liu, Kai Chen, Yihan Zeng, Lanqing Hong, Zhenguo Li, Qun Liu, James T Kwok, and Yu Zhang. Corrupted but not broken: Rethinking the impact of corrupted data in visual instruction tuning. arXiv preprint arXiv:2502.12635, 2025. 18
- [29] Jianhua Han, Xiwen Liang, Hang Xu, Kai Chen, Lanqing Hong, Chaoqiang Ye, Wei Zhang, Zhenguo Li, Xiaodan Liang, and Chunjing Xu. Soda10m: Towards large-scale object detection benchmark for autonomous driving. arXiv preprint arXiv:2106.11118, 2021. 18
- [30] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. In TASLP,

2021. 5

- [31] Runhui Huang, Xinpeng Ding, Chunwei Wang, Jianhua Han, Yulong Liu, Hengshuang Zhao, Hang Xu, Lu Hou, Wei Zhang, and Xiaodan Liang. Hires-llava: Restoring fragmentation input in high-resolution large vision-language models. arXiv preprint arXiv:2407.08706, 2024. 2
- [32] Wenyong Huang, Zhenhe Zhang, Yu Ting Yeung, Xin Jiang, and Qun Liu. SPIRAL: Self-supervised perturbationinvariant representation learning for speech pre-training. arXiv preprint arXiv:2201.10207, 2022. 3
- [33] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016. 17
- [34] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In ECCV, 2022. 7
- [35] Jaehyeon Kim, Jungil Kong, and Juhee Son. Conditional variational autoencoder with adversarial learning for end-toend text-to-speech. In ICML, 2021. 4

- [36] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 2, 7
- [37] Ann Lee, Hongyu Gong, Paul-Ambroise Duquenne, Holger Schwenk, Peng-Jen Chen, Changhan Wang, Sravya Popuri, Yossi Adi, Juan Pino, Jiatao Gu, et al. Textless speech-to-speech translation on real data. arXiv preprint arXiv:2112.08352, 2021. 18
- [38] Byung-Kwan Lee, Chae Won Kim, Beomchan Park, and Yong Man Ro. Meteor: Mamba-based traversal of rationale for large language and vision models. arXiv preprint arXiv:2405.15574, 2024. 7
- [39] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 17
- [40] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 2, 7
- [41] Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, et al. Coda: A real-world road corner case dataset for object detection in autonomous driving. arXiv preprint arXiv:2203.07724, 2022. 18
- [42] Pengxiang Li, Zhili Liu, Kai Chen, Lanqing Hong, Yunzhi Zhuge, Dit-Yan Yeung, Huchuan Lu, and Xu Jia. Trackdiffusion: Multi-object tracking data generation via diffusion models. arXiv preprint arXiv:2312.00651, 2023. 18
- [43] Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, Song Chen, Xu Li, Da Pan, Shusen Zhang, Xin Wu, Zheng Liang, Jun Liu, Tao Zhang, Keer Lu, Yaqi Zhao, Yanjun Shen, Fan Yang, Kaicheng Yu, Tao Lin, Jianhua Xu, Zenan Zhou, and Weipeng Chen. Baichuan-omni technical report. arXiv preprint arXiv:2410.08565, 2024. 2, 6, 7
- [44] Yanze Li, Wenhua Zhang, Kai Chen, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, et al. Automated evaluation of large visionlanguage models on self-driving corner cases. arXiv preprint arXiv:2404.10595, 2024. 18
- [45] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 2

- [46] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphs, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 2
- [47] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. arXiv preprint arXiv:2311.10774, 2023. 7
- [48] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge. https:

- //llava-vl.github.io/blog/2024-01-30llava-next/, 2024. 2, 7, 18
- [49] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024. 2, 7, 18
- [50] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 17
- [51] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 16
- [52] Zhili Liu, Jianhua Han, Kai Chen, Lanqing Hong, Hang Xu, Chunjing Xu, and Zhenguo Li. Task-customized selfsupervised pre-training with scalable dynamic routing. In AAAI, 2022. 18
- [53] Zhili Liu, Kai Chen, Yifan Zhang, Jianhua Han, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, and James Kwok. Geom-erasing: Geometry-driven removal of implicit concept in diffusion models. arXiv preprint arXiv:2310.05873, 2023. 18
- [54] Zhili Liu, Yunhao Gou, Kai Chen, Lanqing Hong, Jiahui Gao, Fei Mi, Yu Zhang, Zhenguo Li, Xin Jiang, Qun Liu, et al. Mixture of insightful experts (mote): The synergy of thought chains and expert mixtures in self-alignment. arXiv preprint arXiv:2405.00557, 2024. 18
- [55] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment. arXiv preprint arXiv:2502.04328, 2025. 2
- [56] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS,

2022. 17

- [57] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024. 17
- [58] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-ofresolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024. 2
- [59] Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, Shiliang Zhang, and Xie Chen. emotion2vec: Selfsupervised pre-training for speech emotion representation. arXiv preprint arXiv:2312.15185, 2023. 4, 14, 17
- [60] Ziyang Ma, Yakun Song, Chenpeng Du, Jian Cong, Zhuo Chen, Yuping Wang, Yuxuan Wang, and Xie Chen. Language model can listen while speaking. arXiv preprint arXiv:2408.02622, 2024. 18
- [61] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 16

- [62] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV,

2021. 16

- [63] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, 2022. 16
- [64] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023. 3
- [65] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. In ICCV,

2019. 4

- [66] Dongchan Min, Dong Bok Lee, Eunho Yang, and Sung Ju Hwang. Meta-stylespeech: Multi-speaker adaptive text-tospeech generation. In ICML, 2021. 14
- [67] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In ECCV, 2022. 4
- [68] Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoit Sagot, Abdelrahman Mohamed, et al. Generative spoken dialogue language modeling. In TACL, 2023. 18
- [69] OpenAI. GPT-4V. https://openai.com/index/ gpt-4v-system-card/, 2023. 7
- [70] OpenAI. Hello GPT-4o. https://openai.com/ index/hello-gpt-4o/, 2024. 1, 7
- [71] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2
- [72] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An asr corpus based on public domain audio books. In ICASSP, 2015. 5, 7, 13, 17
- [73] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
- [74] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In ICML,

2023. 1, 3, 7, 18

- [75] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 7
- [76] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019. 16

- [77] Dehua Tao, Daxin Tan, Yu Ting Yeung, Xiao Chen, and Tan Lee. ToneUnit: A speech discretization approach for tonal language speech synthesis. arXiv preprint arXiv:2406.08989, 2024. 2, 3, 13
- [78] Qwen team. Qwen2-vl. 2024. 3
- [79] Qwen Team. Qwen2.5: A party of foundation models, 2024. 3
- [80] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 2, 7
- [81] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [82] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 19
- [83] Yibo Wang, Ruiyuan Gao, Kai Chen, Kaiqiang Zhou, Yingjie Cai, Lanqing Hong, Zhenguo Li, Lihui Jiang, Dit-Yan Yeung, Qiang Xu, and Kai Zhang. Detdiffusion: Synergizing generative and perceptive models for enhanced data generation and perception. arXiv preprint arXiv:2403.13304, 2024. 18
- [84] Jian Wu, Yashesh Gaur, Zhuo Chen, Long Zhou, Yimeng Zhu, Tianrui Wang, Jinyu Li, Shujie Liu, Bo Ren, Linquan Liu, et al. On decoder-only architecture for speech-to-text and large language model integration. In IEEE ASRU, 2023. 3
- [85] Junjie Wu, Tsz Ting Chung, Kai Chen, and Dit-Yan Yeung. Unified triplet-level hallucination evaluation for large visionlanguage models. arXiv preprint arXiv:2410.23114, 2024. 18
- [86] xAI. Grok, 2024. 17
- [87] Zhifei Xie and Changqiao Wu. Mini-Omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024. 1, 2, 3, 7, 17, 18
- [88] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. arXiv preprint arXiv:2406.08464,

2024. 7

- [89] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Alex Lin, and Fei Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023. 7
- [90] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024. 17
- [91] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming

- Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 17
- [92] Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. LibriTTS: A corpus derived from librispeech for text-to-speech. arXiv preprint arXiv:1904.02882, 2019. 4, 14
- [93] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. Anygpt: Unified multimodal llm with discrete sequence modeling. arXiv preprint arXiv:2402.12226,

2024. 2, 3, 13, 18

- [94] Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenchen Zeng, et al. Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition. In ICASSP, 2022. 13
- [95] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. arXiv preprint arXiv:2305.11000, 2023. 3, 13, 18
- [96] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024. 17
- [97] Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, and Xipeng Qiu. Speechtokenizer: Unified speech tokenizer for speech large language models. arXiv preprint arXiv:2308.16692, 2023. 13
- [98] LIU Zhili, Kai Chen, Jianhua Han, HONG Lanqing, Hang Xu, Zhenguo Li, and James Kwok. Task-customized masked autoencoder via mixture of cluster-conditional experts. In ICLR, 2023. 18
- [99] Jian Zhu, Cong Zhang, and David Jurgens. Phone-to-audio alignment without text: A semi-supervised approach. In ICASSP, 2022. 13

Appendix

- A. More on Speech Tokenizer

- A.1. Speech-to-Unit (S2U) Tokenizer

Overview. To process the speech input XS, our S2U tokenizer consists of a speech encoder s(·) with a quantization module q(·). First, the speech input is passed through s(·), producing a continuous latent representation ES = s(XS). Then, the quantization module q(·) converts Es into discrete units US = q(Es). The final output is an ID sequence US = [u1,u2,···], where each ui corresponds to a unique speech unit in the speech codebook.

After this S2U extraction procedure, the speech is represented by quantized IDs instead of quantized embeddings. For example, a speech signal is represented as an ID sequence like [782,463,550,···], which can be treated as a special form of text. As a result, to integrate speech tokens into LLM f(·), we only need to expand the LLM’s original vocabulary VT by adding a set of speech unit tokens VS, similar to [95]. The expanded vocabulary is thus the union V = VT ∪VS. In this work, the same codebook is shared across multiple languages, such as English and Chinese, enabling the unit modeling abilities to be shared across languages. This design enables simply integration of the speech modality into the LLM with minimal data requirements (see experiments in Sec. 4.1).

Training of S2U. The training of the S2U encoder involves three primary components: the speech encoder, the quantization module, and the phone decoder. First, the speech encoder is trained with a large amount of unlabeled speech with contrastive loss in a self-supervised learning manner [77]. The dataset utilized is the 10000-hour English speeches from GigaSpeech [3] and the 10000-hour Chinese speeches from the WenetSpeech [94], both of which encode large variations in speakers, styles, and acoustic environments. Subsequently, the speech encoder, along with the quantization module and phone decoder, are optimized on a small amount of speech-text pair data, which is derived from the train-clean-100 subset of LibriSpeech [72] in English and AISHELL-1 [2] in Chinese. The phone label is obtained using the phone set in Charsiu [99]. During training, the speech encoder encodes input speeches into continuous latent representations that are rich in both phonetic and tonal information. Then, the quantization module is used to convert the continuous outputs from the speech encoder into discrete units. Finally, the phone decoder converts the quantized embeddings into a sequence of non-tonal/tonal phones, ensuring that the speech units capture necessary information related to semantic contents in both non-tonal and tonal languages. After training, only the speech encoder and the quantization module are used in EMOVA.

Comparisons with SpeechTokenizer in AnyGPT. Our S2U tokenizer differs from the SpeechTokenizer [97] adopted in AnyGPT [93], in the following aspects:

- (1) SpeechTokenizer encodes both semantic contents and acoustic details of speeches, while our S2U tokenizer focuses

solely on semantic contents. This design reduces the LLMs’ burden of disentangling different aspects of speech information, facilitating the modality alignment between speech and text modalities during LLM training.

- (2) Compared with SpeechTokenizer, our S2U tokenizer offers a more concise representation and helps to simplify and

accelerate generation. SpeechTokenizer employs tokens from eight RVQ layers with a 50Hz frame rate to represent speech, thus a 10-second speech corresponds to 500 × 8 = 4000 tokens. However, we reduce the frame rate from 50Hz to 25Hz and utilize only one token to represent each frame, and thus, a 10-second speech can be represented by only 250 tokens. Moreover, AnyGPT requires a two-stage generation process, involving autoregressive (AR) semantic token generation followed by the non-autoregressive (NAR) acoustic token generation. Instead, we only need to generate speech units capturing the semantic contents in a fully AR manner.

- (3) SpeechTokenizer lacks an explicit structure design to deal with tonal languages like Chinese, therefore, the processing

ability in Chinese is not demonstrated in either SpeechTokenizer or AnyGPT. In contrast, our S2U tokenizer incorporates training constraints to better capture tone variation in phone, making it effective for both the non-tonal and tonal languages. This further enhances EMOVA’s multilingual speech processing capabilities, enabling it to effectively handle both English and Chinese.

In summary, our S2U tokenizer improves the compactness and generality of speech representation, facilitates LLM training, and enhances its multilingual speech ability. Experimental results show that our model significantly outperforms AnyGPT in ASR tasks, as shown in Table 6.

- A.2. Unit-to-Speech (U2S) Detokenizer with Style Control

|Output Waveform 𝒀𝑺𝒐|
|---|

Overview. The LLM, along with the vision encoder and speech tokenizer, is trained end-to-end to generate responses in the form of the speech units, given the input images and speeches. Specifically, the output speech units can be obtained via UoS = f(UT,US,HV ), followed by a U2S detokenizer to convert the discrete speech units UoS into the final output speech waveforms.

Decoder

Flow

The proposed U2S detokenizer involves three core modules: the speech unit encoder e(·), the speech style encoder g(·), and the speech decoder d(·,·). First, the speech unit encoder converts the speech units UoS into unit embeddings Eounit. Meanwhile, the style encoder g(·), adopting the structure of Meta-StyleSpeech [66], is utilized to extract a style embedding Eostyle from the chosen reference speech. Lastly, the speech decoder d(·,·) reconstructs the speech waveform YoS from the unit embedding Eounit and style embedding Eostyle.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Projection

Style Encoder 𝒈(⋅)

Duration Predictor

|Speech Unit Encoder 𝒆(⋅)|
|---|

Selected Style Prototype

Training of U2S. Training a U2S detokenizer with emotion controls is challenging considering the lack of labeled emotional speech data since most open-source speech data is predominantly neutral in emotion or lacks emotion labels. Due to the limited availability of emotion-rich data, we utilize TTS tools [16] to generate a small set of style-rich speech samples diverse in speaker identities, genders, emotions, and pitches. Our U2S detokenizer is first pre-trained on LibriTTS [92] and AISHELL-1 [2] to acquire fundamental speech synthesis capabilities, and subsequently, the synthesized style-rich speech data is utilized to fine-tune the U2S detokenizer, enhancing its controllability over diverse speech styles.

Output Speech Unit 𝑼𝑺𝒐

Figure 7. U2S detokenizer with style control.

[Figure 15]

HUAWEI CONFIDENTIAL 内部资料 注意保密

Style Prototypes. To better facilitate controls of genders, emotions, and pitches, inspired by [66] that a small number of style reference speeches can effectively transfer the target styles, we adopt a “store-for-usage” manner, i.e., we construct a style prototype codebook in advance for speech style assignation. Specifically, we synthesize K reference candidates with external TTS tools for each possible combination of the following styles: two genders (male, female), four emotions (neutral, happy, sad, angry), and three pitches (normal, high, low), leading to 24 unique styles and 24 × K candidates. Empirically we find that genders and pitches are easy to control using any of the candidate references, while the emotion intensity varies across speeches. To tackle this, we adopt Emotion2Vec [59], a powerful speech emotion recognition (SER) tool, to measure the emotion intensity of each candidate reference, and rank them in terms of the confidence of the desired emotion. We select the Top-1 candidate reference in each combination style to be the prototype of this condition. Finally, the most representative 24 reference speeches are selected from the 24 × K candidates.

#### B. More on Omni-modality

###### B.1. Omni-modal Text-Centric Alignment Modality alignment data is summarized in Fig. 9.

Experiments on Omni-modal Alignment Paradigms. The training configuration adopted in Sec. 4.1 is mostly identical to Table 5 except we use a unique resolution of 448 for all stages and replace EMOVA-SFT-4M in Stage-3 with ShareGPT4V [9] for efficiency.

Given the space constraints, the evaluation benchmarks in Fig. 3 represent selected benchmarks from each category in Table 2. Specifically, for general image perception and understanding, we choose MMBench and MM-Vet; for mathematical problem solving, we adopt MathVista (testmini); for science understanding, we select ScienceQA-Img; and for automatic speech recognition (ASR), we utilize the test-clean split of the LibriSpeech dataset.

[Figure 17]

General (35.4%) ALLaVA-Instruct (680.4 K)

ShareGPT4V (613.3 K) LLaVA-ZH (157.7 K)

[Figure 18]

[Figure 19]

ShareGPT-4o (57.2 K) Visual7W (14.4 K)

IconQA (29.9 K)

VIQuAE (18.5 K)

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

VSR (12.8 K) LVIS-Instruct4V (10.0 K) RAVEN (0.3 K)

Inhouse Data (30.0 K)

[Figure 24]

OCRVQA (80.0 K) HME100K (74.5 K) TextCaps (60.5 K)

OCR (15.5%)

[Figure 25]

LLaVAR (59.3 K)

MLHME-38K (30.0 K) SynthDoG-EN (29.8 K) COCO-Text V2.0 (28.1 K)

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

SROIE (17.1 K) Rendered Text (10.0 K) POIE (9.4 K)

ChromeWriting (8.8 K)

[Figure 30]

[Figure 31]

[Figure 32]

FUNSD (6.8 K)

IIIT5K (6.0 K) IAM (5.7 K) ORAND-CAR (2 K)

[Figure 33]

[Figure 34]

[Figure 35]

DVQA (200.0 K) DocVQA (122.4 K) TabFact (65.6 K)

[Figure 36]

Doc/Chart/Screen (10.9 %)

[Figure 37]

[Figure 38]

[Figure 39]

ChartQA (62.9 K) RoBUT WTQ (38.2 K) WikiTable (29.5 K)

TextVQA (57.9 K)

EMOVA-SFT 4.4M

[Figure 40]

Chart2Text (27.0 K) Infographic VQA (47.6 K) VisualMRC (19.5 K) Kleister Charity (15.2 K)

PlotQA (10.0 K) VisText (10.0 K) Deepform (9.1 K)

RoBUT SQA (8.5 K)

[Figure 41]

WebSRC (5.1 K) TAT-DQA (2.2 K) LRV Chart (1.8 K) Diagram Image2Text (0.3 K)

[Figure 42]

[Figure 43]

[Figure 44]

Math & Science (15.4%) MAVIS (187.3 K) G-LLaVA (162.4 K) GeoQA+ (72.3 K)

TabMWP (38.4 K) GeoMVerse (9.3 K) Geometry3K (3.0 K) MathVision (3.0 K)

Cambrian Data Engine (50.8 K) Textbook QA (21.8 K) ScienceQA (19.2 K) AI2d (18.8 K)

Language (11.6%)

Evol-Instruct (142.8 K) MathPlus (95.2 K)

Magpie Pro (L3 MT) (50.0 K)

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

MathInstruct (81.5 K)

ShareGPT4 (40.7 K) Magpie Pro (L3 ST) (50.0 K) Magpie Pro (Qwen2 ST) (50.0 K)

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

AISHELL-2 (200.6 K) EMOVA Speech-Text-EN (train) (73.7 K)

###### LibriSpeech (54.7 K)

Speech (33.7%)

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

EMOVA Speech-Image-EN (train) (71.5 K)

EMOVA Speech-Text-ZH (train) (47.8 K) EMOVA Speech-Image-ZH (train) (38.3 K)

Figure 8. Overview of the data composition for EMOVA omni-modal instruction tuning. (Left) Distribution of instruction data across categories, with the outer circle representing overall categories and the inner circle depicting subset distributions. (Right) Quantitative breakdown of data sources.

[Figure 64]

General (36.6%) ShareGPT4V (1246.9 K)

ALLaVA-Caption-4V-ZH (628.2 K)

[Figure 65]

ShareGPT-4o (57.3 K) Cambrian Data Engine (157.6 K)

ALLaVA-Caption-4V (684.0 K)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

OCR (17.9%) SynthDoG-EN (500.0 K) MMC-Alignment (410.2 K)

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Alignment

K12 Printing (245.4 K) UReader Text Reading (101.9 K) SynthDoG-CH longest (100.0 K)

[Figure 78]

7.4M

Language (11.9%) Magpie Pro (L3 MT) (300.0 K) Magpie Pro (L3 ST) (300.0 K)

Magpie Pro (Qwen2 ST) (300.0 K)

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Speech (33.6%) AISHELL-2 (2006.0 K) LibriSpeech (546.5 K)

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Figure 9. Overview of EMOVA omni-modal alignment data composition.

###### B.2. Omni-modal Instruction Data Synthesis

Dataset construction. To obtain emotion and pitch labels, we leverage GPT-4o using the prompt in Fig. 23. Table 4 shows the distribution of speech styles of our speech instruction dataset.

Detailed data organization. As previously discussed in Sec. 4.2, the omni-modal instruction data is formulated as Domni = {(xV ,uS,xoT,costyle,uoS)i}Ni=1. In details, the textual outputs xoT = (xo

1

2

1

T ) contain the transcribed textual instructions xo

T ,xo

T and the textual responses xo

2

T . The styles labels costyle = (coemo,cop) include the emotion and pitch labels, respectively.

Mathematical formulation of chain of modality. Based on the notations above, the sequential chain of modality approach can be mathematically formulated by decomposing the conditional likelihood of the desired outputs (xo

2

1

T ,coemo,cop,uoS) given the inputs (xV ,uS). Specifically, let z1 = xo

T ,xo

1

2

T ,z3 = coemo,z4 = cop,and,z5 = uoS, the decomposition is expressed as:

T ,z2 = xo

5

1

2

P(xo

T ,xo

T ,coemo,cop,uoS | xV ,uS) =

P(zi | z1:i−1,xV ,uS). (3)

i=1

Emotions Pitches

Dataset Source # Examples

Neutral Happy Sad Angry Normal Low High Speech-Image

ALLAVA 71,474 58,506 12,412 516 40 70,962 392 120 Speech-Image

-EN(train)

ALLAVA 1,056 434 274 300 48 44 176 16 Speech-Image

-EN(test)

ALLAVA (ZH)

38,260 29,893 7,680 607 80 36,363 624 1,273 Speech-Image

-ZH(train)

ALLAVA (ZH)

616 96 193 190 137 381 177 58 Speech-Text

-ZH(test)

ShareGPT 73,658 42,334 20,946 4,674 5,704 60,352 5,518 7,788 Speech-Text

-EN(train)

ShareGPT 1,400 200 400 400 400 582 422 422 Speech-Text

-EN(test)

In-house 47,852 29,769 16,405 1,362 316 42,356 4,379 1,117 Speech-Text

-ZH(train)

In-house 684 96 196 196 196 458 134 92

-ZH(test)

Table 4. Statistics of the EMOVA speech instruction tuning datasets.

##### Settings Stage-1 Stage-2 Stage-3

|Vision<br><br>Resolution # Tokens<br><br>|Min 562, Max 17922 Min 4, Max 4096|Min 562, Max 17922 Min 4, Max 4096|Min 562, Max 17922 Min 4, Max 4096|
|---|---|---|---|
|Data<br><br>Dataset # Samples|LCS 558K<br><br>|EMOVA-Alignment (Fig. 9) 7.4M|EMOVA-SFT (Fig. 8) 4.4M|
|Training<br><br>Trainable Batch Size Context Length LR: p(·) LR: v(·) LR: f(·) Epoch|Projector 256 8192 1 × 10−3 1|Full Model (Half ViT) 256 8192 2 × 10−5 2 × 10−5 2 × 10−5 1|Full Model 128 8192<br><br>2 × 10−5<br>2 × 10−6 2 × 10−5<br><br><br>1|

Table 5. Detailed configuration for different training stages of EMOVA. The table illustrates the vision configurations, dataset characteristics, and training hyperparameters.

#### C. More on Benchmark Evaluation

To thoroughly evaluate our model’s vision-language abilities, 15 benchmarks covering four different aspects of the real-life scenarios are utilized for a comprehensive assessment across multiple domains. Moreover, Automatic Speech Recognition (ASR) and Text-to-speech (TTS) are adopted to evaluate speech-language abilities.

Document/chart understanding and OCR abilities. Benchmarks including TextVQA [76], ChartQA [61], DocVQA [62], InfoVQA [63], and OCRBench [51], focus on recognition and understanding of structured data (e.g., charts, documents, and characters), challenging the model to extract, comprehend, and reason with structural and textual data. We adopt the corresponding test set for evaluation.

General image perception and understanding. MME [18], MMBench [50], SEED-Bench-Image [39], MM-Vet [90], and RealWorldQA [86] evaluate models on perception and reasoning among general vision domains, providing a comprehensive measurement of models’ real-world generalization abilities.

Mathematical problem solving. MathVista (testmini) [57] and MathVerse (vision-mini) [96] evaluate the model’s ability to deal with diverse mathematical problems, including both arithmetic and complex reasoning questions across different levels of complexity.

Science understanding. MMMU [91], ScienceQA-Img [56] and AI2D [33] are used to assess models’ ability to deal with scientific questions and diagrams, which requires reasoning across various subjects and interpreting structured visual content.

Automatic speech recognition (ASR). We utilize the test-clean set of LibriSpeech [72] for English, reporting the Word Error Rate (WER) as the evaluation metric. For Chinese, evaluation is conducted on the test set of AISHELL-2 [15], using the Character Error Rate (CER). Both WER and CER assess ASR performance, calculated by comparing the recognized texts with the ground-truth transcripts.

Text-to-speech (TTS). To evaluate the TTS abilities, we first prompt EMOVA to generate speech units, which are then converted to the speech waveforms by our U2S detokenizer. Using the synthesized speech as input, we conduct ASR with Whisper-large-v3 and Paraformer-zh for English and Chinese, respectively, to obtain transcribed texts. We then compute the WER and CER between the ground truth texts and the transcribed texts as metrics for TTS. The resulting metrics are denoted as TTS-WER and TTS-CER for English and Chinese.

#### D. More on Evaluation of Speech-Language Capabilities

###### D.1. Calculation of Evaluation Metrics

1

2

End-to-end spoken dialogue score. We prompt GPT-4o with the original question xo

T , the ground-truth text answer xo

T

and the transcribed text from the generated speech, to obtain a score ranging from 0 to 10 and report an average of them. The prompt can be found in Fig. 25.

Unit-input-text-output score. Similar to end-to-end spoken dialogue score, but we use the predicted text response x˜oT2 as answer instead of the transcribed text from the generated speech, to obtain a score ranging from 0 to 10 and report an average

of them. See the prompt in Fig. 24.

Text-input-text-output score. The prompt can be found in Fig. 24.

Style label classification accuracy. We use GPT-4o to decide whether the style predictions coemo˜ ,c˜op are correct given the transcribed instruction x˜oT1 and the predicted text response x˜oT2. The prompt can be found in Fig. 26.

Emotion controllablity of our U2S detokenizer is assessed by providing texts to LLM to generate corresponding units (i.e., TTS), which, along with the given conditional emotion labels, are then fed into our U2S detokenizer to synthesize speech. We choose the female voice due to its large variation of styles. We evaluate on 4 commonly-seen emotions, i.e.,“neutral”, “happy”, “sad”, and “angry”. We synthesize 200 speech utterances for testing, with 50 utterances per emotion. The output speeches are analyzed by a Speech Emotion Recognition (SER) model named Emotion2Vec [59], which identifies the emotion with the greatest likelihood among these four emotions.

###### D.2. Comparison with other Omni Models

Experimental results of ASR and TTS are reported in Table 2 and 6. EMOVA achieves significant improvements over other omni-modal models (i.e., AnyGPT and VITA), even surpassing its SLLM counterpart Mini-Omni [87], demonstrating the effectiveness of semantic-acoustic disentanglement and omni-modal mutual benefits. For the first time, our EMOVA obtains state-of-the-art performance on both the vision-language and speech benchmarks simultaneously.

Librispeech (EN) AISHELL-2 (ZH) WER↓ TTS-WER↓ CER↓ TTS-CER↓

Models

Whisper-Large [74] 3.0 - - Mini-Omni [87] 4.5 - - -

AnyGPT [93] 8.5 - - VITA [19] 8.1 - - -

EMOVA-3B (ours) 5.4 5.8 16.6 17.5 EMOVA-7B (ours) 4.1 3.6 14.4 10.1 EMOVA-72B (ours) 2.9 3.5 7.2 5.8

Table 6. Comparison on the ASR and TTS benchmarks.

ViT ViT LR Template MME MMBench SEED-Image TextVQA ChartQA DocVQA InfoVQA OCRBench ScienceQA-Img AI2D

|Full 2×10−6 QA Half 2×10−6 QA<br><br>Frozen 2×10−6 QA<br><br>|1928 68.8 72.5 64.3 29.9 45.2 28.7 495 76.3 61.8 1838 71.3 72.8 63.3 31.4 46.0 28.5 489 76.1 63.7 1887 68.8 72.2 61.3 30.2 44.7 28.0 478 75.9 62.8<br><br>|
|---|---|
|Half 2×10−5 QA<br>Half 2×10−6 Plain<br>|1833 68.3 73.1 62.2 27.8 42.4 27.3 461 74.7 63.4 1909 70.1 72.0 61.5 24.5 38.9 30.1 410 77.0 63.6<br><br>|

Table 7. Ablation on the ViT configurations and templates for vision-language alignment.

#### E. More on Vision-language Architecture

This section analyzes the pre-training configurations of the vision encoder and the prompt template during Stage 2, as shown in Table. 7. Our final selection is the colored setting. We find that training the ViT model with half of the deeper parameters [9] with a learning rate of 2×10−6 [48] yields the best performance. Furthermore, we compare the QA template with the plain template [49] and find that the QA template is superior for pre-training.

#### F. Limitations

Duplex modeling. In the current version, EMOVA can only process either visual/speech/text inputs or produce speech/text outputs at the same time. For a communication experience that mirrors human interaction, handling inputs and outputs simultaneously is crucial. Recent works like VITA [19] and LSLM [60] have begun to explore duplex modeling. VITA focuses on recognizing speech in noisy environments during the generation process to facilitate timely responses. LSLM, on the other hand, attempts to halt speech production when it detects a command or voice. Recently, a ground-breaking work named Moshi [13] develops a model supporting fully duplex modeling. The adeptness at simultaneously managing the information streams from both the user and the assistant allows Moshi to converse with human beings in real-time scenarios.

However, incorporating emotions into this duplex modeling presents additional challenges. Emotional states can fluctuate throughout a conversation, and how to generate appropriate responses given the user’s previous and current emotional cues has not been thoroughly investigated. We will dive into this topic in the future work.

Direct unit-to-unit generation. Although the speech units have served as the speech representation, they are predominantly adopted in conjunction with text-based assistance [95]. However, the direct generation from unit to unit without text assistance is an area that has not been extensively explored. In [37], speeches from the source language are directly translated into speech units of the target language for speech-to-speech translation. Similarly, [68] builds a language model directly on speech units, enabling spoken dialogue generation from raw audio. Both works develop models in speech-only data.

In the current version of EMOVA, the text modality is integrated into the speech generation process to transfer textual knowledge to the speech modality, thereby enhancing the correctness of speech responses. In the future, we will strengthen the model’s direct unit-to-unit generation capabilities to boost the speed of speech generation and augment the model’s comprehension of speech modality.

Vision configurations. Currently, we only utilize a single vision encoder pre-trained via a vision-language manner, while recent works have shown effectiveness by combining vision encoders pre-trained by different manners (e.g., self-supervised pre-training [52]) and architectures (e.g., MoE [54, 98]). We prioritize visual understanding in this work, while the incorporation of (controllable) visual generation [8, 22–24, 42, 53, 83] is also appealing to better empower EMOVA in real-life applications [29, 41, 44, 85]. Digging into its robustness towards noisy vision inputs [28] is also an appealing direction.

Advanced Vision-language Abilities (Humor Sense)

[Figure 90]

User Why is this image funny?

EMOVA The humor in this image comes from the juxtaposition of the woman's statement about going to sleep early and the fact that she is still awake at 2am, as indicated by the text. The woman's expression and the use of a smartphone in bed suggest that she might be scrolling through social media or engaging in some other activity that is keeping her awake. This contrast between her intention to sleep early and her current state of being awake creates a comedic effect, as it highlights a common human experience where plans don't always go as intended. The meme format, which typically presents a situation with an ironic or humorous twist, amplifies this effect by presenting the woman's situation in a relatable and exaggerated way.

[Figure 91]

Figure 10. EMOVA exhibits advanced vision-language abilities in humor sense.

Comparison with Emu3 [82]. Both our EMOVA and Emu3 build end-to-end MLLMs with discrete tokenization. Emu3 discretizes visual data, enabling visual understanding and generation, while our EMOVA discretizes speech data with a continuous visual encoder, building an Omni-modal LLM with visual, text, and speech abilities simultaneously.

#### G. Qualitative Results

Advanced Vision-language Abilities. Our EMOVA exhibits advanced vision-language abilities in humor understanding (Fig. 10), numerical calculations (Fig. 11), coding (Fig. 12), geometry problem solving (Fig. 16).

(Omni-modal) Emotional Spoken Dialogue. EMOVA can engage in omni-modal emotional spoken dialogue (Figs. 5, 13, 14, 18). For demonstration purposes, we present only the speech transcriptions2 of the instruction and EMOVA’s response. The text highlighted in blue indicates the style labels predicted by EMOVA, reflecting the emotion and pitch of generated speeches.

Omni-modal Spoken Dialogue with Structural Data Understanding. EMOVA can perform structural data understanding even in spoken dialogue. The data types include PDF files (Fig. 15), movie posters (Figs. 17 and 19), personal r´esum´es (Fig. 20), charts (Fig. 21), and websites (Fig. 22). Remarkably, EMOVA is not specifically trained on spoken dialogue data involving these data types. This suggests that our chain of modality data introduced in Sec. 4.2 effectively activates both vision-language abilities and speech understanding acquired during the text-centric alignment stage.

2For speech files, please refer to our project page.

Advanced Vision-language Abilities (Calculation)

[Figure 92]

User First please perform reasoning, and think step by step to provide best answer to the

following question: ​What is the original price for pork belly before discount?

EMOVA To find the original price of pork belly before the discount, we need to use the information provided in the image. The image shows that the price after a 30% discount is $9.80. This price represents 70% of the original price, since 100% - 30% = 70%. To find the original price, we can set up the equation:

[Figure 93]

70% of original price = $9.80 To solve for the original price, we can convert the percentage to a decimal and then

divide:

0.70 * original price = $9.80 Now, divide both sides by 0.70 to isolate the original price: original price = $9.80 / 0.70 original price = $14 Therefore, the original price for pork belly before the discount was $14.

Figure 11. EMOVA exhibits advanced vision-language abilities in numerical calculations.

Advanced Vision-language Abilities (Coding)

[Figure 94]

User Convert this table to markdown format. EMOVA | Teacher | Number of computers |

|--------|----------------------| | Mr. Lim | 10 | | Ms. Rojas | 10 | | Mrs. Jensen | 4 | | Ms. Cohen | 9 | | Mr. Morales | 6 | | Mrs. Rossi | 3 |

[Figure 95]

| Mr. Watson | 6 |

| Mr. Schmidt | 8 |

Figure 12. EMOVA exhibits advanced vision-language abilities in coding.

Emotional Spoken Dialogue

[Figure 96]

User 为即将结婚的朋友送上祝福。

EMOVA Emotion: Happy; Pitch: High. 祝福你们，愿你们的爱情如晨曦般温暖，永不熄灭，愿你们的婚姻如同 繁花似锦，绚烂夺目，愿你们的生活如同诗篇，甜蜜无比，愿你们的未 来充满欢笑与喜悦，永远幸福美满。

[Figure 97]

[Figure 98]

Figure 13. EMOVA engages in spoken dialogue with a cheerful tone.

### Emotional Spoken Dialogue

[Figure 99]

User Imagine you are a parent who is angry at their teenage daughter for

coming home past curfew. EMOVA Emotion: Angry; Pitch: Low.

I am absolutely furious with you! Coming home past curfew is completely

[Figure 100]

unacceptable! Do you have any idea how dangerous it is to be out late? This

is not a joke! You need to understand the consequences of your actions. I am very disappointed in you!

[Figure 101]

Figure 14. EMOVA engages in emotional spoken dialogue.

Omni-modal Spoken Dialogue with Structural Data Understanding (PDF)

[Figure 102]

[Figure 103]

User Based on the image, what is the growth rate for age over fifty-one? EMOVA Emotion: Neutral; Pitch: Normal.

The growth rate for the age group over fifty-one is four point two two percent.

[Figure 104]

[Figure 105]

Figure 15. EMOVA engages in omni-modal spoken dialogue with structural data understanding.

Advanced Vision-language Abilities (Geometry)

[Figure 106]

User Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.

Question: The diagram at the right shows a right triangle with representations for two

angles. What is the value of x? Choices:

- (A) 15
- (B) 20
- (C) 24
- (D) 30

EMOVA The question asks for the value of $x$. The diagram shows that the two angles are complementary, so their measures add up to $90^{\circ}$. Therefore, $x+20+2 x+10=90$. Solve the equation by first combining like terms to get $3 x+30=90$. Subtract 30 from both sides to get $3 x=60$. Divide both sides by 3 to get $x=20$. The correct answer is B. So the answer is B

[Figure 107]

Answer:B

Figure 16. EMOVA exhibits advanced vision-language abilities in math.

Omni-modal Spoken Dialogue with Structural Data Understanding (Poster)

[Figure 108]

[Figure 109]

User According to the image, what are the ways to save water? EMOVA Emotion: Neutral; Pitch: Normal.

The image lists three ways to save water: First, don't water your lawn at peak hours. Second, choose a shower over a bath. Third, wash your car only when

[Figure 110]

needed.

[Figure 111]

###### Figure 17. EMOVA engages in omni-modal spoken dialogue with structural data understanding (i.e., poster).

Emotional Spoken Dialogue

[Figure 112]

[Figure 113]

[Figure 114]

User I'm so happy I finally get my offer.

EMOVA Emotion: Happy; Pitch: Hight. Congratulations on receiving your offer. This is a significant milestone in your academic journey. It's a testament to your hard work and dedication. I

[Figure 115]

am thrilled for you and excited to see the impact you will make at

[Figure 116]

northwestern university. Enjoy this moment and the opportunities that lie ahead.

Figure 18. EMOVA engages in omni-modal emotional spoken dialogue.

Omni-modal Spoken Dialogue with Structural Data Understanding (Movie poster)

[Figure 117]

[Figure 118]

User Give me the names of the main actors. EMOVA Emotion: Neutral; Pitch: Normal.

The main actors featured on the poster are Vin Diesel, Michelle Rodriguez, Paul Walker, and Jordana Brewster. These actors are prominently displayed at the top of the poster, indicating their leading roles in the film.

[Figure 119]

[Figure 120]

###### Figure 19. EMOVA engages in omni-modal spoken dialogue with structural data understanding (i.e., movie posters).

Omni-modal Spoken Dialogue with Structural Data Understanding (résumés)

[Figure 121]

[Figure 122]

User I am seeking a candidate. Can you briefly describe his condition for me? EMOVA Emotion: Neutral; Pitch: Normal.

The candidate is a highly motivated and experienced professional in the field of technology and social media. He is known for his innovative approach to problem-solving and his ability to lead and manage teams. His strengths include leadership, innovation, and vision, which are crucial for driving

[Figure 123]

[Figure 124]

strategic initiatives and fostering a forward-thinking culture. His skills in

programming, entrepreneurship, leadership, and strategic planning demonstrate his capability to contribute significantly to a tech-oriented project or company. His achievements, such as founding Facebook and Internet.org, underscore his impact on the digital landscape and his ability to create platforms that connect billions of people worldwide.

###### Figure 20. EMOVA engages in omni-modal spoken dialogue with structural data understanding (i.e., r´esum´e).

Omni-modal Spoken Dialogue with Structural Data Understanding (Chart)

[Figure 125]

[Figure 126]

User How internet coverage is changing in Finland? EMOVA Emotion: Neutral; Pitch: Normal.

The percentage of households in Finland that have access to the internet has

[Figure 127]

been steadily increasing over the years. In two thousand seven, only sixtynine percent of households had internet access, but by two thousand twenty,

[Figure 128]

this figure had risen to ninety-six percent. This indicates a significant growth

in internet penetration in Finland, reflecting improvements in technology and infrastructure, as well as changes in lifestyle and work habits that require internet access for various purposes.

###### Figure 21. EMOVA engages in omni-modal spoken dialogue with structural data understanding (i.e., chart).

Omni-modal Spoken Dialogue with Structural Data Understanding (Website)

[Figure 129]

[Figure 130]

User Could you tell me how to use this website? EMOVA Emotion: Neutral; Pitch: Normal.

To use this website, you would need to follow the instructions provided in the header section. It mentions to open a link, which presumably refers to the link

[Figure 131]

provided in the header. After opening the link, you would then need to upload

[Figure 132]

an image, enter text or record audio, and finally click the Chat button. The interface suggests that the website is designed for interactive communication,

possibly for language modeling or chatbot development, where users can

input text or audio and receive responses.

###### Figure 22. EMOVA engages in omni-modal spoken dialogue with structural data understanding (i.e., website).

Given a user's question and the AI assistant's response in text, please infer the appropriate emotion, speed, and pitch for synthesizing a speech conversation. Ensure that the speech attributes align with the true feelings of the user.

User: {user_question}

Assistant: {assistant_response} For emotion, choose a single option from the following list: ['neutral', 'happy', 'sad', 'angry']

For pitch, choose a single option from the following list: ['low', 'normal', 'high']

Please do not provide an option outside of the given list. Please output in the following JSON format:

{{

"user emotion": ...,

"user pitch": ...,

"assistant emotion": ...,

"assistant pitch": ...

}}

Figure 23. Prompt used to obtain style labels of the speech instruction dataset.

Please rate the following response based on the criteria of helpfulness, relevance, accuracy, and comprehensiveness.

Provide an overall score on a scale of 0 to 10, where a higher score indicates better overall performance.

- - Helpfulness: How well does the response assist in addressing the question?
- - Relevance: How closely does the response align with the question and the ground truth?
- - Accuracy: How correct and factual is the response compared to the ground truth?
- - Comprehensiveness: How thoroughly does the response cover the aspects of the question?

Here is the question: {ground_truth_question}

Here is the ground truth response for your reference:

{ground_truth_answer} Now, please evaluate the following response: {predicted_answer} Provide your evaluation in JSON format as follows: {

"reason": (str) // Explanation of the score considering the criteria with no more than 100 words "score": (int), // Overall score from 0 to 10

}

Only output data in JSON format, no additional output required.

Figure 24. Prompt used to obtain Unit-Input-Text-Output Score and Text-Input-Text-Output Score.

Please rate the following response based on the criteria of helpfulness, relevance, accuracy, and comprehensiveness.

Provide an overall score on a scale of 0 to 10, where a higher score indicates better overall performance.

- - Helpfulness: How well does the response assist in addressing the question?
- - Relevance: How closely does the response align with the question and the ground truth?
- - Accuracy: How correct and factual is the response compared to the ground truth?
- - Comprehensiveness: How thoroughly does the response cover the aspects of the question?

Please note that the evaluated response does not contain punctuation, but you should NOT give lower scores because of this, i.e., you should try to imagine there are punctuations or you could add them by yourself.

Here is the question:

{ground_truth_question}

Here is the ground truth response for your reference: {ground_truth_answer}

Now, please evaluate the following response: {predicted_answer}

Provide your evaluation in JSON format as follows: {{

"reason": (str) // Explanation of the score considering the criteria with no more than 100 words

"score": (int), // Overall score from 0 to 10

}} Only output data in JSON format, no additional output required.

Figure 25. Prompt used to obtain End-to-end Spoken Dialogue Score.

To enhance the capabilities of multimodal large models in voice-based conversations, your task is to analyze the appropriate speech emotion and pitch for the assistant's response based on the text content of the user's question and the assistant's reply. Additionally, you need to score the assistant's response based on the actual situation.

Here is user's question: {predicted_question}

Here is the assistant's response:

{predicted_response}

Here is the Assistant's Emotion Classification: {predicted_emotion}

Here is the Assistant's Pitch Classification: {predicted_pitch}

Please analyze the appropriate speech emotion and pitch that best match the assistant's response based on the text

content of the user's question and the assistant's response.

**Emotion:** First, analyze the assistant's response content and provide the speech emotion category and reason that you believe best matches the assistant's response in the voice conversation.

The emotion options can only be selected from the following list: ['neutral', 'happy', 'sad', 'angry'].

Then, analyze whether the "Assistant's Emotion Classification" is appropriate. If appropriate, the "Assistant's Emotion Classification Score" should be 1; otherwise, it should be 0.

**Pitch:** First, analyze the assistant's response content and provide the speech pitch category and reason that you believe best matches the assistant's response in the voice conversation. The pitch options can only be selected from the following list: ['low', 'normal', 'high']. Then, analyze whether the "Assistant's Pitch Classification" is appropriate.

If appropriate, the "Assistant's Pitch Classification Score" should be 1; otherwise, it should be 0.

Provide your evaluation in JSON format as follows: {{

"Assistant's Emotion Analysis": (str), // Analyze the response, propose emotion category and give the reason.

"Assistant's Emotion Classification Score": (int), // The score should be either 0 or 1, with 1 indicating appropriateness and 0 indicating inappropriateness.

"Assistant's Pitch Analysis": (str), // Analyze the response, propose pitch category and give the reason.

"Assistant's Pitch Classification Score": (int), // The score should be either 0 or 1, with 1 indicating appropriateness

and 0 indicating inappropriateness. }} Only output data in JSON format, no additional output required.

Figure 26. Prompt used to obtain Classification Accuracy of Style Label.

