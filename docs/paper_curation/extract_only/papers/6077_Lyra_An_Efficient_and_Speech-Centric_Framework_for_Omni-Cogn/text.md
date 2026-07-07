## Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition

Zhisheng Zhong1∗ Chengyao Wang1∗ Yuqi Liu1∗ Senqiao Yang1 Longxiang Tang1 Yuechen Zhang1 Jingyao Li1 Tianyuan Qu1 Yanwei Li1 Yukang Chen1 Shaozuo Yu1 Sitong Wu1 Eric Lo1 Shu Liu2 Jiaya Jia 2,3

∗ Equal contribution Corresponding author Code: https://github.com/dvlab-research/Lyra CUHK1 SmartMore2 HKUST3

Image | Video Understanding

Handle Hours Audio & Video

[Figure 1]

EgoSchema

TextVQA 82.6

|33<br><br>30<br><br>25<br><br>16<br><br>|27|25| |
|---|---|---|
| | | |
<br><br>17<br><br>OOM 13 14 15 16<br><br>TPS<br><br>N<br><br>Full<br><br>|| | | |
|---|---|---|
| | | |
|
|---|---|
| |19 21 24<br><br>| | |
|---|---|
| |49|
<br><br>30 41<br><br>60<br><br>OOM<br><br>13 14 15 16<br><br>Memory<br><br>N<br><br>Lyra|
| | |

###### 63.2

# arXiv:2412.09501v1[cs.CV]12Dec2024

MME 2335

MVBench 67.2

Efficiency

See

MM-Vet 63.5

VideoMME 62.8

Input token num. : 2N

More Accurate Omni-Understanding Speech & Sound Understanding

Hear Write & Speak

Image-Speech

|69.1 → 81.0|
|---|
|79.9 → 89.4|
| |
|56.0 → 68.5|
|[TextVQAS, DocVQAS, ChartQA|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Text-Speech

|4.0% → 1.8%<br><br>LibriSpeech Word Error Rate|
|---|

[Figure 7]

[Figure 8]

|10 min video<br><br>[Figure 9]|
|---|

VLM Lyra

S] Previous Omni SOTA Lyra

: Rocket launch failed, crashing back to the ground …

[Figure 10]

+ : Blastoff! Nails ‘chopsticks’ booster catch …

Figure 1. Overview of Lyra. Lyra shows superiority compared with leading models in the following aspects: 1. Stronger performance. Lyra achieves state-of-the-art results across a variety of modalities understanding and reasoning tasks. 2. More versatile. Lyra can directly handle images, videos and audio tasks even lasting several hours. 3. More efficient. Lyra is trained with less data and increases the speed, reduces memory usage, making it suitable for latency-sensitive and long-context multi-modality applications.

#### Abstract

#### 1. Introduction

As Multi-modal Large Language Models (MLLMs) evolve, expanding beyond single-domain capabilities is essential to meet the demands for more versatile and efficient AI. However, previous omni-models have insufficiently explored speech, neglecting its integration with multi-modality. We introduce Lyra, an efficient MLLM that enhances multimodal abilities, including advanced long speech comprehension, sound understanding, cross-modality efficiency, and seamless speech interaction. To achieve efficiency and speech-centric capabilities, Lyra employs three strategies: (1) leveraging existing open-source large models and a proposed multi-modality LoRA to reduce training costs and data requirements; (2) using a latent multi-modality regularizer and extractor to strengthen the relationship between speech and other modalities, thereby enhancing model performance; and (3) constructing a high-quality, extensive dataset that includes 1.5M multi-modal (language, vision, audio) data samples and 12K long speech samples, enabling Lyra to handle complex long speech inputs and achieve more robust omni-cognition. Compared to other omni-methods, Lyra achieves state-of-the-art performance on various vision-language, vision-speech, and speech-language benchmarks, while also using fewer computational resources and less training data.

With the rapid evolution in Large Language Models (LLMs) [20, 25, 42, 55, 57], empowering the impressive capabilities for multi-modality inputs is becoming an essential part of current Multimodal Large Language Models (MLLMs). However, most current MLLMs are limited to just two modalities: either vision-language [2, 11, 24, 28, 29, 31, 34, 76] or speech-language [10, 13, 65]. OpenAI’s recent release of GPT-4o [43], an advanced omni-modal model, has reignited interest in intelligent assistants capable of fine-grained visual perception, understanding spoken instructions, and generating vocal responses simultaneously. It highlights a strong demand for MLLMs that integrate more functions and modalities, such as visual, language, speech, sound, and even other new abilities [6, 16, 63, 71].

Based on our study, most existing omni-models [6, 13, 16, 71] primarily focus on the relationship between speech and text, without exploring connections between speech and other modalities, such as vision. Consequently, speechrelated evaluation metrics are typically limited to text. In this paper (Sec. 4.3), we observe that strong performance in the speech-text modality does not necessarily imply good performance in the speech-vision modality. Thus, we suggest that omni-model evaluation should be speech-centric, expanding its involvement with additional modalities.

To further enhance the speech capabilities of MLLMs, we inevitably encounter the following challenges: First, larger datasets (e.g., the extensive data required to train models like LLaMA3 [12] and Qwen2-VL [60]) are needed for both previous modalities and speech. Second, there is a clear trend toward increasing context length across modalities. More long-context benchmarks for specific modalities are being proposed, including long-document [5, 8] and long-video tasks [15, 32, 62, 66, 73]. Last, building a sufficiently powerful model may demand significant financial and computational resources, which raises environmental concerns related to high carbon emissions.

Combining the above three points, we propose Lyra, an efficient and speech-centric framework for omni-cognition: Leveraging existing open-source large models. We efficiently start with powerful LLMs and VLMs, like LLaMA3 [12] and Qwen2-VL [60], which already demonstrate strong multi-modal capabilities. Through our multimodality LoRA module, we can effectively preserve certain strong capabilities of open-source large models in specific modalities with minimal training data, while simultaneously developing their abilities in the speech modality.

Enhancing information interaction between modalities, especially within the speech modality. 1) Considering the implicit correspondence between speech and text, we propose latent cross-modality regularizer. 2) Based on instructions, we identify potential redundancy in context token information across multiple modalities. We further propose latent multi-modality extractor to mine informative tokens, which brings significant advantages in training speed, inference speed and GPU memory efficiency.

High-Quality Datasets for Omni-Cognition. Centered on speech, we have constructed two types of high-quality datasets: To enhance the model’s speech capabilities, we collect and generate a multi-modal dataset of 1.5M textimage-speech samples from diverse public sources, ensuring a rich and varied data foundation; To handle longer speech inputs and demands, we are the first to construct a long speech dataset comprising 12K samples. Through training, our model achieves robust omni-cognitive abilities and can handle long speech inputs lasting several hours.

With these three improvements, Lyra offers the following advantages (Fig. 1). More versatile: As shown in Table 1, Lyra now supports both sound and speech understanding and generation, while also handling more complex long speech cases. More efficient: Lyra achieves faster training and inference speed across speech, image, and video tasks. Compared to previous models, Lyra has a smaller model size and is trained with less data. Stronger: Lyra demonstrates enhanced omni-comprehension capabilities over previous MLLMs, achieving state-of-the-art performance in vision-language and vision-speech and speechlanguage tasks simultaneously.

Vision Audio Image Video SU SG LS Sound

Function Method

LLaVA-OV ✓ ✓ ✗ ✗ ✗ ✗ Intern-VL ✓ ✓ ✗ ✗ ✗ ✗ Mini-Gemini ✓ ✓ ✗ ✗ ✗ ✗

Vision

Qwen-Audio ✗ ✗ ✓ ✗ ✗ ✓ Mini-Omni ✗ ✗ ✓ ✓ ✗ ✗ LLaMA-Omni ✗ ✗ ✓ ✓ ✗ ✗

Audio

Intern-Omni ✓ ✗ ✓ ✗ ✗ ✗ VITA ✓ ✓ ✓ ✗ ✗ ✗ Any-GPT ✓ ✓ ✓ ✓ ✗ ✗ EMOVA ✓ ✗ ✓ ✓ ✗ ✗

Omni

Lyra ✓ ✓ ✓ ✓ ✓ ✓

Table 1. Function comparison of related work. SU, SG, and LS represents speech understanding, speech generation, and long speech support, respectively.

#### 2. Related Work

Multi-modal Large Language Models. Recent advancements in Large Language Models (LLM) and Multimodal Large Language Models (MLLMs) have pushed the boundaries of human-computer interaction, expanding their capabilities from text-based tasks to complex multi-modality scenarios. Large Language Models, like GPTs [42], LLaMA [12, 57] and Qwen [4, 67], have demonstrated strong capabilities in textual understanding and generation. Building on these foundations, Vision Language Models [28, 31–36, 60, 61, 68] extend LLMs with visual perception capabilities, leveraging advanced encoders [47] and high-resolution techniques to interpret visual inputs. Speech Language Models (SLMs) [49], including SpeechGPT [72] and LLaMA-Omni [13], have introduced real-time speech understanding and generation, with advanced models enabling control over speech styles. Moving further, MLLMs [63] such as AnyGPT [71], VITA [16] and EMOVA [6], integrate vision, text, and audio within a unified architecture, enabling robust interaction across diverse modalities. The abilities and modalities of previous leading MLLMs are listed in Table 1. In contrast, Lyra tackles complex scenarios, enabling seamless, dynamic multimodal interactions for rich, real-time AI experiences.

Token Reduction for MLLMs. Token reduction techniques aim to improve the efficiency of LLMs and VLMs by minimizing redundant tokens during inference and training. In LLMs, methods like StreamingLLM [64] and FastGen [17] optimize memory usage by selectively retaining essential tokens, while techniques like H2O [75], ScissorHands [37] and Quest [53] use attention-based scoring to prioritize valuable tokens. In VLMs, approaches such as FastV [7] reduce visual tokens to tackle the high computational cost of image processing. Lyra extends token reduction to more modalities, such as video and speech, where token lengths tend to increase in long-context scenarios. By evaluating the relationship between context and instruction tokens, we progressively discard redundant tokens to enhance efficiency without compromising performance.

Speech & Sound

“The”, “answer”, “of” … …

|Audio rank 𝒓𝐀|
|---|

|Top 𝐾 mined audio tokens|
|---|

[Figure 11]

| |[Figure 12]| | |
|---|---|---|---|

[Figure 13]

[Figure 14]

Audio Encoder Audio Projector

discarded

[Figure 15]

[Figure 16]

𝑨𝐀 𝑩𝐀

[Figure 17]

Speech

|[Figure 18]<br><br>|Softmax(𝒒A ⋅ 𝒌TT)|
|---|
|
|---|

Generator

|Powerful Pretrained LLM|
|---|

User: Did anyone celebrate a birthday?

Text Tokenizer

Top 𝐾 mined vision tokens

[Figure 19]

User: What is unusual about this image?

|Vision rank 𝒓𝐕|
|---|

[Figure 20]

Text Decoder

[Figure 21]

Vision Encoder Vision Projector

𝑨𝐕 𝑩𝐕

Softmax(𝒒V ⋅ 𝒌TT)

Lyra: The answer of … …

Image & Video

i-th layer of LLM

j-th block of LLM × 𝒏

× 𝒎

Latent Cross-Modality Regularizer Multi-Modality LoRA Latent Multi-Modality Extractor Streaming Generation

- Figure 2. The framework of Lyra. Lyra supports multi-modal inputs. When the data contains a speech modality, we use the latent cross-modality regularizer to assist. Data from each modality is processed through encoders and projectors before being sent into the LLM. Within the LLM, multi-modality LoRA and latent multi-modality extraction modules operate synergistically, facilitating the simultaneous generation of both speech and text outputs.

#### 3. Lyra

As shown in Fig. 2, the overall architecture of Lyra is composed of four main components: latent cross-modality regularizer, multi-modality LoRA, latent multi-modality extractor, and streaming generation. Lyra is designed as a unified framework, with each component being easily and efficiently extendable to support additional modalities and functionalities. In this paper, Lyra primarily focuses on the three modalities of audio (speech, sound), vision, and language. Therefore, in the following sections of this section, We will provide a detailed introduction to the mechanisms of the following modules: latent cross-modality regularizer, multi-modality LoRA, and latent multi-modality extractor. Due to space limitations, streaming speech-text generation will be detailed in the appendix. Since speech contexts tend to be lengthy, the integration of long speech capabilities will be discussed at the end of this section. To ensure clarity in the following discussion, let’s define some key notations: the X[i] be the token of modality-i. For example, X[text] represents the text token, X[image] represents the image token, X[video] represents the video token, X[speech], X[sound] represents the speech and sound token, respectively.

###### 3.1. Latent Cross-Modality Regularizer

For MLLMs, it is crucial to achieve effective alignment between tokens from each modality and LLM. As the view from the speech modality, there is a high degree of informational overlap with the text modality. Specifically, considering only semantic information, speech can be converted into its corresponding transcribed text. However, our experiments have shown that using speech with naive alignment

training as the instruction (S+I, S for speech instruction, I for image context) generally yields less effective results compared to using transcribed text (T+I, T for text instruction, I for image context):

TextVQA (S+I) TextVQA (T+I) MM-Vet (S+I) MM-Vet (T+I) 76.7(-2.8) 79.5 53.1(-8.0) 63.1

To address this, we aim to make the tokens from the speech modality as similar as possible to the corresponding transcribed text tokens before feeding them into LLM, thereby minimizing the loss of relevant information. Another challenge arises from the variable length of speech: a sentence can be spoken quickly or slowly while retaining the same meaning in the text modality, leading to length discrepancies. In general, the tokens produced by a speech encoder (like Whisper) tend to be much longer than the corresponding text tokens (speech-to-text, STT), i.e., X[speech] ∈ Rd×L , X[STT] ∈ Rd×S, L > S, d is the token dimension. We define the latent distance between the l-th speech token and the s-th SST token as:

dist(l,s)=−log softmax(X[speech],lX⊤[STT],s/τ) , (1)

Where τ is the temperature. To get the minimum distance between two different length tokens, we follow the Dynamic Time Warping (DTW) algorithm:

Dl,s = dist(l,s) + min{Dl,s−1,Dl−1,s,Dl−1,s−1}. (2) The illustration is shown in Fig. 3. We define the latent cross-modality regularization loss as LLCMR = L+1SDL,S. Finally, the total loss of the system becomes the combination of two losses: Ltotal = LCE + λLLCMR, where LCE is the cross-entropy loss on LLM output, and λ is a loss weight hyper-parameter.

- 0

- 1

- 2

- 3

- 4

- 5

- 0

- 1

- 2

- 3

- 4

- 5

Audio Duration Distribution

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Game 9% Sports 8% Film 6% Comedy 6% Music 11% Edu. 7%

- 1

- 2

- 3

- 4

- 5

- 6

- 1

- 2

- 3

- 4

- 5

- 6

Travel & Event 9% Entertainment 9% People & Blogs 8% Pets & Animals 7% News & Politics 11%

STTtokenID(totalStokens)

STTtokenID(totalStokens) DistanceValue

[Figure 27]

9%

9% 8%

CumulativeDistanceValue

8%

Long Speech: From 8 min to 2 hours

[Figure 28]

9%

1000

[Figure 29]

6% 6%

[Figure 30]

750

4K hours 40M words

8%

[Figure 31]

[Figure 32]

500

7%

[Figure 33]

[Figure 34]

11%

250

[Figure 35]

11% 7%

8%

[Figure 36]

Science & Tech. 8% 0

Audio Data Distribution

1000 2000 3000 4000 5000 6000

|Summarization|
|---|

[Figure 37]

[Figure 38]

[Figure 39]

1 2 3 4 5 6 7 8 9 Speech token ID (total L tokens)

1 2 3 4 5 6 7 8 9 Speech token ID (total L tokens)

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

|Question-Answer|
|---|

…… Long Speech SFT Data

[Figure 44]

[Figure 45]

Transcripts

Tokenize

- Figure 3. Illustration of the DTW algorithm in our alignment. Our goal is to make the speech tokens as similar as possible to the corresponding translated tokens.

SFT Data Construction

Clipping

AudioProjector

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

AudioEncoder

[Figure 51]

[Figure 52]

[Figure 53]

flatten

LLM

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

###### 3.2. Multi-Modality LoRA Pipeline

Long Speech Training

The current open-source VLM (such as Qwen2-VL) is already quite powerful. With limited data quantity and quality, jointly training vision-speech-language modalities may reduce the model’s original capabilities. Therefore, we adopt an efficient multi-modality LoRA [23] pipeline. Revisiting the notation introduced at the beginning of this section, we represent X[i] as the token of modality-i. The modality-i can be text, image, video, speech token, and sound. Since our model involves joint training across multiple modalities, here we define X[M] can be any combination of the above different modality tokens. The output of multimodality LoRA can be written as:

Figure 4. Long speech capability integration pipeline. (Middle) Our pipeline for generating instruction-following data for long speech. (Top) The proportion of question and speech categories in our long speech SFT dataset. (Bottom) Our long speech SFT pipeline. Long speech segments will be clipped and flattened.

relevance to the text query, discarding redundant multimodality tokens. To achieve this, we introduce a latent multi-modality information extraction strategy.

Concretely, instead of applying this strategy to every layer, we implement a block-based manner. Suppose the LLM consists of mn layers; we divide them into blocks of m layers each, resulting in n blocks. At the final layer of each block, we apply our following information extraction strategy, which evaluates the similarity between the attention scores of tokens from each modality and the question text tokens. We represent this with the following equation:

H = B[M]A[M] + W X[M], (3)

where W is the original weight of LLM, A[M] and B[M] is low-rank adapter of combination-M. During training, our Multi-Modality LoRA is integrated into each layer of the LLM. Because each modality is trained using LoRA, the process is highly efficient, achieving strong performance with minimal data while preserving much of the original model’s visual capabilities.

Q[text]K⊤[\text]

, (4)

√

topk softmax

d

where Q[text] denotes the query corresponding to text modality tokens, and K⊤[\text] represents the key corresponding to tokens from other modalities. For clarity, let’s assume that the length of multi-modality tokens K⊤[\text] is L. After passing through each block, we retain only ρL multi-modality tokens. From a block-wise perspective, the token length decays exponentially, significantly reducing computational and memory costs. A similar mechanism exists in the brain’s neural processing of complex information [50]. Notably, text tokens can be extended to instruction tokens for other modalities, such as speech. This extractor enables us to handle long speech more efficiently.

###### 3.3. Latent Multi-Modality Extractor

As MLLMs expand their functionality and accommodate longer contexts, efficiently using tokens within a limited context window becomes essential to address the longcontext problem. We now consider the relationship between non-text modalities and the text modality. In response to a given question, many tokens from non-text modalities may be largely irrelevant to the question itself. For example, as shown in Fig.2, only a subset of image tokens is relevant to the instructed question. Similarly, for the video and speech modality, only a portion of tokens from video and speech directly corresponds to the question instruction.

###### 3.4. Long Speech Capability Integration

We observe that in LLM training, the long-context effect brought by high-resolution images, lengthy videos, and long audio (in the following subsection) often includes tokens with limited relevance, which not only increases the computational load for training and inference but also consumes unnecessary memory. To address this, we propose dynamically selecting multi-modality tokens based on their

There is a growing trend toward increasing the length of single-modality content processed by models, such as long text and long video inputs in MLLMs. However, existing MLLMs are limited in handling long speech due to the constraints of speech encoders. Specifically, models like Intern-Omni [44], VITA [16], and LLaMA-Omni [13] use Whisper-like encoders, which restrict audio input to around

30 seconds. VITA and Mini-Omni, which employ more complex encoders, can process at most one minute of audio input. This limitation largely stems from the lack of suitable long speech SFT datasets and appropriate preprocessing methods. To address this issue, we developed the first SFT dataset for long speech understanding, aimed at enhancing model capabilities in handling extended audio content. Our dataset comprises about 12K long-form audio recordings, with durations ranging from several minutes to two hours. These recordings were collected from diverse YouTube sources, including informational videos, interviews, and speeches, covering a wide range of topics—from humanities and current events to technology and society. With related transcripts, we utilized LLM to generate question-and-answer pairs derived from the captions and instructions. These questions cover summarization and other types of inquiries that encourage a comprehensive understanding of long speech content. The overall question distribution and details are illustrated in Fig. 4.

Once the dataset was ready, we tackled the challenge with the speech encoder. Inspired by high-resolution image segmentation methods like LLaVA-NeXT [36], we adopted a similar strategy to better handle the speech encoder for long audio processing (illustrated in Fig. 4). However, unlike previous speech cases, a new challenge emerged: for a naive Whisper-v3 encoder, a 30-second audio clip is encoded into 1,500 tokens. Under typical short speech scenarios, an LLM can handle 1,500 tokens comfortably. When we consider long speech cases, such as a two-hour audio clip, this would result in an astonishing 360,000 tokens, which is beyond our processing capacity. Thus, it is essential to consider compression techniques on speech tokens. The experimental results are presented as follows:

#(Token) 100 150 300 500 1500

TextVQAS 75.9% 76.8% 77.8% 78.0% 76.8% MM-VetS 55.3 54.4% 56.3% 58.8% 58.9%

Experimental results indicate that having a higher number of speech tokens provides certain benefits. However, beyond a certain threshold, the performance improvement becomes quite limited. Taking into account both computational costs and model performance, we ultimately decided to use the 300 compressed tokens version for extending the model to handle long speech cases.

#### 4. Experiments

In this section, we conduct a speech-centric evaluation, assessing its integration with image, video, and text modalities. we first outline our experimental framework, commencing with the experimental setup. Subsequently, we compare Lyra with leading methods on various benchmarks and qualitative results. Detailed component wise analysis (based on Lyra-Base) is given at the end of this section. More experiment details and results refer to our Appendix.

###### 4.1. Experimental Setup

Implementation Details. In this study, we instantiate Lyra with the following designs and settings:

- 1. Strong vision encoders and LLMs: Building on the previously applied vision model Qwen2-VL’s ViTs and LLMs [60], they can now process images of any resolution, dynamically converting them into a variable number of visual tokens. We have also designed three versions: For Lyra-Mini, we use Qwen2-VL 2B. For Lyra-Base, we apply Qwen2-VL 7B. For Lyra-Pro, we choose Qwen2-VL 72B.
- 2. Efficient audio encoder: We adopted Whisper-largev3 [48] (Lyra-Base and Lyra-Pro) and its light-weight version, Whisper-large-v3-turbo (Lyra-Mini), which have been trained on a large amount of audio data and has strong capabilities in speech recognition and translation.
- 3. Four stage training for omni-cognition (refer to our appendix for specific details): In the first stage, we conduct text-to-speech pretraining to train the speech encoder. In the second stage, we perform joint training on text, image, and speech modalities to train the LLM along with the corresponding projectors. In the third stage, we train the LLM to extend the model’s capability in handling long speech. In the fourth stage, we train our speech generator, enabling the model to simultaneously output text and corresponding audio in a streaming manner.

Datasets and Evaluations. For model optimization, we construct high-quality data for omni-understanding and speech generation.

- 1. High-quality multi-modal dataset: Based on the MiniGemini SFT [31] dataset, we carefully collected and extended a high-quality multi-modal dataset that covers common scenes and document images and speeches. It contains about 1.5M open-source image-speech, text-image, and text-speech instruction samples. To enhance the generalization of speech modality, we utilize ChatTTS [1] with varying configurations to generate different audios.
- 2. Long speech SFT dataset: As mentioned in Sec. 3.4, we constructed a delicate long speech SFT dataset for long speech capability integration with 12K samples. The dataset involves a distribution of longer audio durations and covers a wide range of domains.
- 3. Evaluation: Unlike the previous omni-model [6, 16], which only tested text-to-speech capabilities, we employed a more omni comprehensive evaluation that covers interactions across image, video, text, and speech modalities.
- 4.2. Main Results

Quantitative Results. In the quantitative analysis experiments, we primarily compare our model with current leading VLMs, such as Mini-Gemini [31], LlavaOV [28], Intern-VL2 [9], and SLM, like Mini-Omni [65], SALMONN [52], Qwen2-Audio [10], and Omni models in-

Omni Comparison Text-Image Text-Video Image-Speech Text-Speech Method Params. TextVQA MME MM-Vet VideoMME MVBench Egoschema TextVQAS DocVQAS ChartQAS LibriSpeech↓

Mini-Gemini 8B 71.9 1989 53.5 - - - - - - LLaVA-OV 7B 65.4 1998 57.5 58.2 56.7 60.1 - - - Intern-VL2 8B 77.4 2211 60.0 54.0 66.4 - - - - -

Mini-Omni 7B - - - - - - - - - 4.5 SALMONN 13B - - - - - - - - - 2.1 Qwen2-Audio 8B - - - - - - - - - 1.6

Intern-Omni 8B 80.6 2210 60.0 - - - 69.1 79.9 56.0 VITA 66B - 2097 41.6 59.2 - - - - - 8.1 EMOVA 14B 82.0 2205 55.8 - - - - - - 4.0

Lyra-Mini 3B 78.3 1884 51.2 55.0 62.5 54.1 73.4 74.8 40.7 2.1 Lyra-Base 9B 82.6 2335 63.5 62.8 67.2 63.2 80.0 85.5 61.0 2.0 Lyra-Pro 74B 83.5 2485 71.4 69.9 72.3 75.8 81.0 89.4 68.5 1.8

Table 2. Omni-comparison on vision-language-speech benchmarks. BenchS indicates that it uses speech instruction as the input.

cluding Intern-Omni [44], AnyGPT [71], VITA [16], and EMOVA [6]. The input modalities we compare are also the most widely used, including text-image, text-video, imagespeech, and text-speech. Detailed results are presented in

- Table 2. In calculating the total parameters of the model, we considered all modality-specific encoders, projectors, and related components. Our model includes three versions: a mini version (3B), a based version (9B), and a pro version (74B). Benefiting from multi-modality LoRA and Qwen2VL, our model maintains relatively high performance in text-image and text-video tasks. For the speech modality, as we mentioned in Introduction part, previous models have evaluated the speech modality rather crudely, without extensively testing metrics for interactions between the speech modality and other modalities. Our model comprehensively outperforms existing omni models in both imagespeech (with an improvement of approximately 9%) and text-speech (with an improvement of approximately 2%) tasks. Additionally, our model is more lightweight, requiring fewer training samples.

Qualitative Results. To ascertain the omni comprehension prowess of Lyra in real world settings, we apply it to a variety of understanding and reasoning tasks in the bottom left part of Fig. 1 and our Appendix. By contrast, Lyra can well solve more complex multi-modality cases.

###### 4.3. Component-Wise Analysis

Latent Cross-Modality Regularizer. We first delve into the proposed latent cross-modality regularizer and report results in Table 3. It is clear that the model achieves significant gains for both speech-image inputs and text-image inputs, with the regularizer integrated as an assistance between speech modality and text modality. In the training of the image-speech-text tri-modal model, introducing the LLCMR significantly enhances the performance of both image-speech and image-text alignments, reducing the gap between them. We also observe that, with only LCE, image-

text performance lags behind image-speech by 8% on the MM-Vet benchmark. However, the performance of speechtext remains relatively unchanged whether using the CE loss or joint loss. Therefore, previous omni models [6, 16] that assessed the speech modality just based on the LibriSpeech [45] WER metric for speech-text alignment are rather arbitrary. We need to evaluate the performance of the speech modality alongside other modalities to accurately measure the effectiveness of omni-models. This also demonstrates the effectiveness of our LLCMR.

Latent Multi-Modality Extractor. For the latent multimodality extractor (LMME) module, we focus primarily on its efficiency and effectiveness in multi-modal tasks. First, we analyze its efficiency, with specific results summarized in Tables 4a and 4b. In Table 4a, we vary the token length, ranging from 211 to 217 (under a long-context case). We denote LMME(n, ρ) as splitting the LLM into n blocks, with each block retaining the top ρ proportion of the most important tokens. We compare three models: the baseline, LMME(4, 0.8), and LMME(4, 0.7). The key metrics examined include Prefill Time, tokens-per-second (TPS), and memory usage on the A100 GPUs. Under the baseline model, multimodal content exceeding 215 tokens results in out-of-memory (OOM) errors. In contrast, our models LMME(4, 0.8) and LMME(4, 0.7) still have room for 217 tokens, consuming over 50% less memory. Additionally, the Prefill Time is significantly shorter than the baseline model (by 100%), and the token generation speed is also notably faster (by 50%).

In Table 4b, we primarily examine the improvement in training speed. We evaluate it using our proposed Lyra SFT and long-speech SFT dataset, which contains 1.5M samples and 12K samples, respectively. From the table, our LMME can reduce training time by more than 50% compared to the original. Since the context in the long-speech dataset is generally longer than it in the 1.5M dataset, the acceleration effect becomes even more pronounced.

Effectiveness TexVQA MM-Vet LibriSpeech Type S+I T+I S+I T+I S+T Baseline - 82.3 - 62.8 -

LCE 76.7 79.5 53.1 61.1 1.9 LCE + λLLCMR 77.8 80.1 58.1 62.6 2.0

- Table 3. Latent cross-modality regularizer. With our regularizer, the performance of both the speech-image and text-image modalities improves, and the gap narrows.

Metric # (Tokens) 211 212 213 214 215 216 217

Baseline 0.19 0.33 0.65 1.47 2.99 OOM OOM LMME(4, 0.8) 0.17 0.24 0.44 0.76 1.60 4.24 10.2

Prefill(s)↓

- LMME(4, 0.7) 0.16 0.21 0.37 0.59 1.23 3.05 7.75

TPS↑

Baseline 32.6 30.8 27.3 25.3 16.6 OOM OOM

- LMME(4, 0.8) 32.7 31.5 31.8 28.6 22.7 14.1 8.37

- LMME(4, 0.7) 33.8 33.3 32.5 30.1 25.3 16.6 10.1

Memory↓

Baseline 20G 23G 30G 41G 60G OOM OOM

- LMME(4, 0.8) 17G 18G 19G 21G 24G 33G 49G LMME(4, 0.7) 17G 18G 19G 21G 24G 33G 49G

(a) Prefill time, tokens per second (TPS), GPU memory comparison.

Data Type Baseline LMME(4, 0.9) LMME(4, 0.8) LMME(4, 0.7)

Lyra-MM-1.5M 66h 58h (-13%) 47h (-29%) 41h (-38%) Lyra-LongSpeech-12K 9.6h 7.0h (-27%) 5.7h (-40%) 4.5h (-54%)

(b) Training time on multi-modality datasets comparison.

Table 4. Efficiency of latent multi-modality extractor.

To verify the effectiveness of our extractor module, we examine the retention of multi-modal tokens. We primarily assess three types of tokens: image tokens, video tokens, and speech tokens. The specific visualizations are shown in Fig. 5. As seen in the figure, our model ultimately retains only about 10%-25% of the tokens across all three modalities. Moreover, the retained token positions are highly relevant to the user-provided instructions, effectively helping to remove information unrelated to the instructions and thereby accelerating training and inference. We also have included the performance experiments related to LMME in the appendix section.

Long-Speech Capability Integration. After performing SFT on our Lyra long speech 12K data mentioned Sec. 3.4, we design the following experiments to validate the model’s capabilities in processing long speech and latent multimodality extraction, given the current lack of a long-speech benchmark. The first experiment is the long speech “Needle in a Haystack” evaluation. We selected five audio files, each more than 3 hours in length, and inserted open-ended audio questions and answers at various points throughout the files. The results are shown on the left side of Fig. 6. According to the figure, we observe that, without enhancing long-speech processing capabilities, the model can handle up to approximately eight minutes of audio. beyond that length, it fails to generate a proper output (Fig. 6a). However, with SFT on our Lyra long speech 12K data, the model

Method Overall Short Medium Long Baseline (7B) 62.8 73.8 62.3 52.3 Baseline + subtitle 64.4 76.2 63.4 53.4 LSCI (7B, solve 33%) 78.6 89.8 77.7 74.8 Baseline + LSCI 66.2 75.7 64.0 58.9 GPT-4o [43] + subtitle 77.1 82.8 76.6 72.1

- Table 5. Effectiveness of long speech capability integration. Lyra integrated with long speech ability, using only audio input, can handle one-third of VideoMME cases, and its accuracies on long, medium, short metrics are better than the current best VLM.

Modality Benchmark Baseline + SFT + MLoRA

Image

TextVQA [51] 82.3 81.3 82.6 MME [14] 2332 2275 2335 MMMU [70] 49.2 48.7 50.8

Video

VideoMME [15] 62.8 61.0 62.8 MVBench [30] 66.7 66.8 67.2 EgoSchema [39] 62.4 63.5 63.2

TextVQAS [51] - 77.8 80.0 Speech DocVQAS [56] - 84.0 84.6

MM-VetS [69] - 54.0 60.0

- Table 6. Effectiveness of multi-modality LoRA (MLoRA). For powerful pretrained models, adding new modality can impair the abilities of other modalities. MLoRA can effectively address it.

can handle audio lengths of up to 4,500 seconds. With audio exceeding 4,500 seconds, the model’s memory usage surpasses the limit (Fig. 6b). By leveraging the latent multimodality extractor module, we achieve the ability to process even longer audio, extending up to and beyond two hours (Fig. 6c). Additionally, In Fig. 6d, we visualize the tokenlevel attention retention and variations for the “needle” with the information extractor module, under the same question instructions. Notably, we can see that as the needle is placed in different locations, the information extractor module dynamically adjusts the attention distribution and retention for positions accordingly.

The second experiment is based on VideoMME. This benchmark includes videos ranging from 30 seconds to one hour. We first extract the audio from these videos and feed only the audio data into our long speech model to obtain predictions and perform the VideoMME evaluation. Along with generating predictions, we also require our model to output whether it can answer the question based on the audio alone. Specific results are shown in Table 5. From the table, it is evident that long audio can resolve about onethird of the test samples, with model accuracy exceeding 78%, significantly outperforming the 7B model. We integrate the long-speech output into our Lyra model, which ultimately performs better than using subtitles alone.

Multi-Modality LoRA (MLoRA) Pipeline. The effectiveness results of MLoRA are presented in Table 6. Compared to multi-modal SFT, MLoRA maintains better original vision performance while enhancing the capability in

[Figure 62]

[Figure 63]

[Figure 64]

###### Input: 8-frame Video (for better visualization) User: What temperature and time are needed to bake the bacon?

4.8% tokens 6.9% tokens 16.8% tokens 7.8% tokens

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

10.5% tokens 14.4% tokens 27.5% tokens 5.0% tokens

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

User: How many casualties did Hurricane Helen cause?

Input: 20-min Audio User: Did anyone in the above content Input: 20-min Audio celebrate a birthday? And how old?

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Org. Wave (100%) Keep Wave (25%) Norm. Attn. Score

| | | |
|---|---|---|

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

"Hurricane Helen rising more than 150 lives now lost search teams now …"

"death toll from Hurricane Helen Rising more than 150 lives now lost hundreds are still … "

###### “In tonight former President Jimmy Carter turning 100, watching the flyover for his birthday … "

"Jimmy Carter at his white cap what he witnessed on this his 100th birthday…"

- Figure 5. Visualization of latent multi-modality extractor in various modalities. The upper part is the video modality, and the lower part is the audio modality. Through latent multi-modality information extraction, semantic tokens related to the instruction are retained, reducing the computational cost of the MLLM. The visualization of the image modality and different blocks can be found in the appendix.

150 300 450 600 750 90010501200135015001650

Time (second)

100%

90%

80%

70%

60%

50%

40%

30%

20%

10%

0%

DepthofNeedle

Baseline

9001800270036004500540063007200810090009900

Time (second)

100

90

80

70

60

50

40

30

20

10

0

Naive SFT On Lyra-LongSpeech-12K (96%)

9001800270036004500540063007200810090009900

Time (second)

100

90

80

70

60

50

40

30

20

10

0

Lyra Training (98%)

[Figure 98]

0.0

0.2

0.4

0.6

0.8

1.0

Score

OOM!

| | |
|---|---|
|Needle Arround #3600|Needle Arround #14400|

0 5000 10000 15000 20000 25000 30000 35000 Position ID

0.00

0.02

0.04

0.06

0.08

AttentionValuesSum

| | |eedle Arround #36000|
|---|---|---|
| |Needle Arround #21600 N| |
| | | |

0 5000 10000 15000 20000 25000 30000 35000 Position ID

0.0

0.1

0.2

AttentionValuesSum

(a) (b) (c) (d)

- Figure 6. Comparison of needle in long speech haystack (average with five samples). (a) The baseline model can not retrieve right needles after 450 seconds. (b) Model finetuned on our long speech datasets can not retrieve right needles after 4,500 seconds and achieves 96% accuracy in 4,500 seconds. (c) Our latent extractor, trained on our long speech datasets, can retrieve longer audio (9,900 seconds), and presents 98% accuracy in 4,500 seconds. (d) As the position of the needle changes, the attention in our model also shifts accordingly.

new modalities like speech. Additionally, our framework is more efficient, achieving better results with less data (50%).

Intern-Omni VITA EMOVA Lyra 27M samples 5M samples 4M samples 2.7M samples

#### 5. Conclusion

In conclusion, Lyra represents a significant step forward in MLLMs, efficiently integrating complex speech, vision, and language modalities with reduced computational requirements (less data, faster speed). We focus on speech to en-

hance its interaction with other modalities within MLLMs. By leveraging the proposed modules, and high-quality, comprehensive SFT datasets, Lyra achieves state-of-theart performance across vision-speech, speech-language, and vision-language benchmarks, which is a more comprehensive evaluation for omni-models to previous research. Our experiments also reveal that speech plays a critical role in multimodal understanding, yet current MLLMs do not effectively leverage this information. We hope our work encourages future researchers to further explore and harness the potential of speech/long speech within MLLMs.

## Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition

### Supplementary Material

We strongly recommend that readers watch the video in our supplementary materials, which include more audio and video examples to get a better understanding and experience. In the following supplementary material, we provide more details about the training configurations and the construction and information of our dataset in Sec. A. In Sec. B, we present additional module settings along with some experimental results and analyses. In Sec. C, we showcase the qualitative results of Lyra.

#### A. Training Configuration and Data

###### A.1. Detailed Training Configuration

- Stage-1: Speech Alignment. In this stage, we only train the parameters of the speech projector for speech-language prealignment with the LibriSpeech [45] and Common Voice Corpus [59] datasets, with about 1.0M data samples.
- Stage-2: Joint Text-Image-Speech Training. Based on the Mini-Gemini [31] SFT data, we assemble and construct a unified dataset with 1.5M samples for the image-text-speech joint training. We use the ChatTTS [1] model to convert high-quality SFT data from text instructions into speech instructions. The multi-modal dataset, i.e., Lyra-MultiModal-1.5M, includes not only single-turn instructions but also multi-turn instructions.
- Stage-3: Long Speech SFT. To enable the model to integrate the long speech capability, we construct the first long-speech SFT dataset, called Lyra-LongSpeech-12K. Details can be found in Sec. 3.4 of the main paper. To ensure more robust performance, the dataset covers a wide range of topics, including humanities, social sciences, technology, education, and more. At this stage, we train both the speech module and the whole LLM module.
- Stage-4: Streaming Text-Speech Generation. During the speech generation stage, we only train the speech generator. To better align the speech generator with the text decoder, we exclusively use text-speech modality QA pairs in our dataset. We filtered and selected a portion of suitable data from the datasets in our Stage-1, Stage-2, and Stage-3 for speech generation, resulting in a dataset of approximately 227K samples. Detailed training settings are further explicated in Table 7.

Settings Stage-1 Stage-2 Stage-3 Stage-4

Speech

Audio Length < 30s < 30s < 2500s, 30s clips < 30s # Tokens 300 300 Max 25, 000 300

Dataset LibriSpeech + CommonVoice Lyra-MultiModal-1.5M Lyra-LongSpeech-12K Filter from Stage-1, 2, 3 # Samples 1.2M 1.5M 12K 227K

Data

Trainable Projector Projector + LLM Projector + LLM Speech Generator Batch Size 256 128 16 32 Learning rate 1 × 10−3 2 × 10−4 2 × 10−4 2 × 10−4 Epoch 1 1 3 1

Training

Table 7. Detailed training settings of Lyra.

###### A.2. Data Collection and Curation

To ensure the data quality and training efficiency, we consider the following aspects while generating speech data for three modalities of joint training.

Generate multi-modal interleave data. To ensure models’ ability to process interleaved multi-modal data, we randomly select one round from multi-round conversations and convert its text into speech, while keeping the remaining rounds in text format. This guarantees that our SFT data preserves its multi-modal interleaved structure.

Oral Expression. Certain types of text are not well-suited for direct conversion using TTS technology. In these cases, we ensure the content is rewritten in a more conversational, oral form. For example, we rephrase “A:” as “Option A is” to enhance clarity and naturalness.

Method LLM Vision Data Time TextVQA MME MM-Vet MMB-EN SEED MMMU Avg. Rate Baseline Vicuna-7B CLIP+Conv Lyra-MM-1.5M 65h 68.4 1865 41.3 65.8 68.1 36.8 100.0%

+ Extractor Vicuna-7B CLIP+Conv Lyra-MM-1.5M 35h(-46%) 69.9 1899 44.9 66.7 67.5 35.3 101.5%(+1.5%) Baseline Qwen2-7B SigLIP LLaVA-665K 18h 69.7 1974 39.4 76.7 74.2 40.8 100.0%

+ Extractor Qwen2-7B SigLIP LLaVA-665K 14h(-22%) 69.1 2005 38.6 76.9 73.5 40.6 99.6% (-0.4%) Baseline Qwen2-7B SigLIP Lyra-MM-1.5M 51h 71.9 2030 51.0 78.1 74.5 40.2 100.0%

+ Extractor Qwen2-7B SigLIP Lyra-MM-1.5M 35h(-31%) 71.8 2007 50.6 77.7 73.7 42.1 100.1%(+0.1%)

Table 8. Latent multi-modality extractor training performance. The training time is reduced by an average of one-third, while the average performance does not degrade and even improves by 0.4%.

###### Lyra Data Examples

Training conversations: human: <image>\nWhat are the two people holding?\nAnswer the question using a single word or phrase. GPT: Umbrella. human: What is the person with the Red Hat doing? GPT: Taking pictures. human: <speech> GPT: Blanket. Evaluation cases: human: <image>\nReference OCR token: DAKOTA, DIGITAL, Single-Use, Camera, Pire, digitat\n<speech>

Figure 7. Lyra training and evaluation data examples.

Speaker Diversity. To maintain diversity in our generated speech, we randomly select speakers with varying timbres and pitches for each instance. Since ChatTTS [1] obtains different speaker characteristics through various Gaussian sampling, it exhibits great diversity and robustness. During our generation process, we switch to a new set of ChatTTS random samples every 128 instructions.

Be Aware of the OCR Text. In real-world applications, a MLLM retrieves text by calling the OCR interface, such as TextVQA. Many OCR tokens, such as ‘G0’ and ‘EF’, lack clear meaning and are not suitable for verbal expression as speech input. Following this practice, we do not convert OCR text into speech. Here, we list some training prompts and evaluation examples of our data in Table 7.

#### B. More Component-Wise Details & Analysis

###### B.1. Latent Multi-Modality Extractor

Qwen2-VL is exceptionally powerful, with the quantity and quality of its training data far surpassing those of public datasets and open-source models. As a result, most approaches to continual learning based on Qwen2-VL tend to result in performance degradation. Therefore, to evaluate the performance of our extractor module, we opt to train a new model from scratch. The results are shown in Table 8. Under the same training settings, models using latent multi-modality extractor achieve faster training speeds, with a maximum acceleration of nearly 50%. Additionally, they maintain or even improve average performance by up to 1% across multiple benchmarks. This series of experiments demonstrates the effectiveness of our extractor. Visualization of the latent multi-modality extractor in image modality is shown in Fig. 10. From it, the tokens retained in different blocks are all related to the user’s instruction. Additionally, for different questions, the token regions in the image most relevant to the question are preserved. This result is consistent with the video and speech modalities discussed in our main paper.

###### B.2. Long Speech Capability Integration

In this part, we primarily introduce prompts related to the long speech capability. The detailed prompts are shown in Table 8. The first is the GPT-4o-based prompt used to generate Q&A during the long speech data collection process. The second is the inference prompt we used to apply the long-speech Lyra model on the VideoMME benchmark. For detailed results and analysis, refer to Sec. 3.4 and the long-speech capability integration part in Sec. 4.3.

###### Long Speech Question-Answer Generation Prompt Example

Task: You will be provided with a transcript from an audio or video recording. Your task is to generate question-answer pairs based on the content of the transcript. Guidelines for Question-Answer Pair Generation:

- - The first question should be about summarizing the content of this recording.
- - Carefully read the transcript provided and base all questions and answers strictly on the content within.
- - Ensure that each question is directly related to specific details in the transcript, such as events, facts, or points made by the speaker.
- - Provide clear, concise, and specific questions, along with accurate answers derived from the transcript.
- - Do not introduce any new information that isn’t in the transcript. If the speaker does not introduce themselves, refer to them as “Speaker” or “Narrator”.
- - Avoid generic or overly broad questions; aim for a range of question types (e.g., factual, inferential, explanation-based).
- - Generate five question-answer pairs. Output Format:
- - Your output should be structured as a JSON object.
- - Each question-answer pair should be formatted as:

‘‘‘json {

[

- {"Question": <question-1>, "Answer": <answer-1>},
- {"Question": <question-2>, "Answer": <answer-2>},

... ]

} ‘‘‘

Long Speech VideoMME Evaluation Prompt Example

Based on the context, determine if it provides enough information to answer the question: <question> with the provided choices <option-A>, <option-B>, <option-C>, <option-D>. Do not introduce any information not found in the context.

- - If the context is sufficient to answer the question, respond “yes” and answer with the option’s letter from the given choices directly.
- - If the context does not contain enough information to answer the question, respond “no”.

Figure 8. Long speech related prompt examples.

###### B.3. Sound Capability Integration

For the sound modality, due to the lack of many pretrained models, we primarily follow ImageBind[18] as the sound encoder. ImageBind processes sound, text, and image modalities using a training approach similar to CLIP [47], ultimately encoding them into just one single token. This approach is not particularly generalizable. During the sound SFT process, our model based on LLaMA3 [12] is trained on the AudioCaps [26] dataset, which contains a total of 46K training samples. The quantitative performance of our model on the test set is shown in Table 9.

Regarding this dataset, as the authors of AudioCaps [26] have noted, “Even to humans, recognizing the true identity of a sound can be ambiguous.” Moreover, LLM-based multimodal models tend to produce more detailed descriptions, while metrics like SPICE [3] and CIDEr [58] are outdated and fail to effectively reflect the most suitable results. Even under such circumstances, our Lyra, trained on just 46K samples for the sound modality, outperforms previous sound models. Some qualitative results are shown in Fig. 9.

###### B.4. Streaming Text-Speech Generation

For the speech-text streaming generation component, we primarily refer to LLaMA-Omni [13] to enable the MLLM to output speech audio.

Speech Discretization. To handle speech responses, we discretize the audio into discrete units with the following steps: 1). Continuous representations are extracted using the HuBERT model [22]. 2). These representations are clustered into discrete indices via the K-means algorithm. 3). Consecutive repeated indices are merged to form a sequence of discrete units, which can be converted back to waveforms using a vocoder [46].

|[Figure 99]|
|---|

|[Figure 100]|
|---|

User:

User:

10s sound (only the sound input)

10s sound (only the sound input)

Lyra:

Lyra:

Water gurgling and splashing.

A small engine is running and a man is talking in the background.

###### Figure 9. Sound capability qualitative results.

[Figure 101]

[Figure 102]

| |
|---|

###### Input: Single Image

User: I want to go to Sunset Blvd, what should I do? User: I want to go to Ventura, what should I do?

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Keep tokens of Block #4

Keep tokens of Block #6 Keep tokens of Block #4 Keep tokens of Block #6

To go to Sunset Blvd, you should follow the right lane and take the exit towards Sunset Blvd. To go to Ventura, you should follow the left lane and continue straight

Figure 10. Visualization of latent multi-modality extractor in the image modality.

Speech Decoder for Streaming Generation. A streaming speech decoder is introduced after the LLM to enable simultaneous generation of text and speech: To ensure the overall structure remains consistent with the LLM, the decoder is built using two transformer layers similar to Qwen2-VL [60]. Similar to LLaMA-Omni, it processes the hidden states from the LLM and generates discrete speech units in a non-autoregressive manner [38, 74]. For upsampling, the text hidden states from the LLM are upsampled to match the speech sequence’s length. These upsampled representations are processed by the speech decoder to produce output features for the discrete speech units.

Alignment and CTC Training. Following LLaMA-Omni, Connectionist Temporal Classification (CTC) [21] is used to align the decoder’s output with the discrete speech units. During training, the model learns to match the output features to the target speech units by minimizing the CTC loss. During inference, the most likely sequence is selected, converted into discrete units, and passed through the vocoder to generate audio.

:.

###### B.5. TTS Methods Ablation Study

In this subsection, we briefly compare the impact of different TTS (text-to-speech) methods on the generalization and robustness of speech instruction (across different domains). We primarily used two TTS methods: ChatTTS [1] and Edge-TTS [41]. ChatTTS employs Gaussian sampling to simulate different speakers (As shown in Listing 1), while Edge-TTS randomly selects from a fixed set of 41 speakers. ChatTTS is likely to be more diverse. We trained models using instruction data generated by these TTS methods and evaluated TextVQA speech instructions generated by different TTS methods. Detailed results can be found in Table 9a. Models trained with speech generated by ChatTTS demonstrated better generalization due to its diversity.

Similar results were observed when compared with speech instructions generated by Intern-Omni [44]. Because we cannot access their training speech instruction data; they only provided the evaluation speech instruction data of DocVQA and ChartQA. Specific results are provided in Table 9b and 9c. While models perform better when trained and evaluated on instructions generated by the same system, the experiments overall demonstrate that instructions generated by ChatTTS are more robust compared to the other two methods.

#### C. Qualitative Results C.1. Examples of Images and Videos

In Fig. 11, we present additional interactions with Lyra, showcasing the model’s adeptness in knowledge-based perception and reasoning for both images and videos. In various complex scenarios, such as recognition of complex PC backgrounds, understanding of game interfaces, and analyzing football match videos with significant differences between frames, Lyra demonstrates superior understanding and reasoning cognitive outcomes.

AT [40] BART [19] PairMix [27] CoDi [54] Lyra-Base

16.8 17.7 18.1 17.1 19.5

Table 9. Sound SPICE performance comparison.

Eval/Train ChatTTS Edge-TTS

Eval/Train ChatTTS

Eval/Train ChatTTS

ChatTTS 80.0 79.5 Edge-TTS 79.7 78.3

ChatTTS 84.6 Intern-O 82.3

ChatTTS 60.4 Intern-O 58.3

(a) TextVQAS

(b) DocVQAS

(c) ChartQAS

###### Table 10. Different TTS training and evaluation.

- 1
- 2
- 3
- 4
- 5
- 6
- 7

###### Listing 1. Sample Random Function in ChatTTS (Pytorch)

|def sample_random(self) -> torch.Tensor:<br><br>spk = ( torch.randn(self.dim, device=self.std.device, dtype=self.std.dtype)<br><br>.mul_(self.std)<br><br>.add_(self.mean)<br><br>) return spk<br><br>|
|---|

###### C.2. Examples of Long Speeches

In the main paper experimental section, Fig. 6a shows that existing Speech Language Models (SLMs) fail when processing audio longer than 450 seconds (about seven minutes): the output becomes nonsensical with extensive repetition. In this part, we demonstrate Lyra’s ability to handle long audio inputs. In Fig. 12, 13, 14, and 15 we demonstrate Lyra’s capability to process long-form speech (best view the following part together with the video in the supplementary materials). Lyra effectively extracts the information that users need from extended speech contents. It excels at capturing both the details and the overall structure of long speeches. In news scenarios (Fig. 12, with frequent topic switches), it accurately identifies the focused information and responds exceptionally well.

For more complex tasks, as shown in Fig. 13, such as scenarios with visual ambiguity, our model leverages long-form speech and keyframes from videos to provide more accurate results compared to a powerful VLM like Qwen2-VL that rely solely on visual information.

- In Fig. 14, our model demonstrates its ability to process daily lectures, offering significant advantages for educational-

related applications. Lyra can handle speech content durations exceeding two hours, which enables intelligent models to tackle more complex multi-modal tasks.

- In Fig. 15, For tasks with longer temporal sequences and higher complexity, Lyra can also understand them and provide

subjectively reasonable answers to the questions.

[Figure 108]

[Figure 109]

| |
|---|

How many APP icons are in this image?

###### Input: Single Image

List their names and the corresponding functions in two columns.

[Figure 110]

There are 10 APP icons visible

There are 12 APP icons

11 APP icons

Name Function Recycle Bin A place to store files …. Zoom A tool for magnifying the screen Windows The operating system … Google Chrome A web browser for browsing … Google A search engine for finding … Adobe Acrobat A program for viewing … YouTube A video-sharing platform … Slack An app designed for teams … Discord A voice and text chat app … Tabs A browser extension… ARC media player A media player for playing …

Name

Name Function

Recycle Bin

Recycle Bin A container for temporary….

Zoom WinSCP Coolcut Tabby Terminal Google Chrome VLC media player

Zoom A video conferencing … WinSCP A secure file transfer …

CoolTerm A terminal emulator … Tabby Terminal A terminal emulator … Google Chrome A web browser for accessing … VLC media player A multimedia player … YouTube Music A music streaming Slack A collaboration platform … Adobe Acrobat A PDF document viewer …

YouTube Music

Slack Adobe Acrobat Discord

Google Drive A cloud storage service …

Lyra-Base-9B

Mini-Gemini-HD-34B

GT

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

###### Input: 2 min Video

User: Please describe this video.

The video showcases a series of scenes from a video game, including a woman in a bikini walking on the beach, a man in a car crashing into a store, and a woman being chased by a group of men on motorcycles. The video also features a logo for the game “Grand Theft Auto VI” and a warning that the game may contain content inappropriate for children. The scenes are set in a city with palm trees and a beach, and the game appears to be a crime-themed action game.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Input: 15 sec Video

User: Provide a detail commentary for this football match clip.

The soccer match is between England and Iran, and the game is currently at the 34th minute. The teams are playing on a field with a large crowd watching. The players are actively engaged in the game, with one player from England scoring a goal. The goalkeeper from Iran is unable to catch the ball, and the crowd cheers for the goal. The match is being broadcasted by FIFA.

Figure 11. Image-text and video-text qualitative results of Lyra.

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Input: 20min 1s Audio

User: Give me a summary about today’s news.

The recording covers multiple significant events and news stories.

- • It begins with Iran launching over 200 ballistic missiles at Israel in response to the killing of Hezbollah leader in Beirut, causing air raid sirens and explosions across Israel, with Israeli defenses and US military intercepting many missiles.
- • The situation raises fears of a wider regional war, affecting the US presidential race with the vice-presidential debate looming.
- • Meanwhile, in the US, tens of thousands of dockworkers are on strike, threatening the national supply chain, while search efforts for victims of Hurricane Helene continue with over 150 deaths reported.
- • There are also developments in legal cases against attorney Shaun Diddie Combs and a manhunt for an escapee.
- • The recording also covers the passing of baseball great Pete Rose and actor John Amos, along with updates on President Jimmy Carter’s 100th birthday.

[Figure 126]

[Figure 127]

[Figure 128]

Input: 20min 1s Audio

User: Give me more information about the Hurricane Helene.

Hurricane Helene was a significant hurricane that struck the northeastern United States, particularly affecting areas such as New England and the Mid-Atlantic states. The hurricane made landfall on the coast of Maine and brought heavy rainfall, strong winds, and storm surges that caused extensive damage. The death toll from the hurricane rose to more than 150 lives, and search and rescue operations were ongoing as teams worked through the night to locate missing individuals. The disaster also caused significant destruction, with rescuers searching from structure to structure, including finding a couple who had been swept away by a wall of water. The search efforts were described as growing more desperate by the hour, with a focus on locating one man’s fiancée who was still missing.

Figure 12. Lyra long speech capability qualitative results for handling daily news.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

###### Input: 10min 20s Audio + Video

User: Did any accidents happen during the video?

Yes, there was an accident during the process. The rocket fell back to the ground after reaching a certain altitude. (Qwen2-VL)

No accidents occurred during the entire process. The mission went smoothly, with the booster successfully returning to the launch site for a catch, and the ship achieving nominal orbit insertion. (Lyra)

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

User: Briefly describe the whole process.

Input: 10min 20s Audio + Video

2. The hot staging process; This includes: 4. The ship’s orbit insertion.

1. The countdown and launch phase;

3. The booster’s return to the launch site;

[Figure 146]

[Figure 147]

[Figure 148]

Input: 10min 20s Audio + Video

User: Why is the day an important day in the history?

The day is significant in engineering history because it marks the successful retrieval of the Super Heavy booster back at the launch tower, a first-ever attempt which was accomplished with precision and skill. This event is described as a day for the engineering history books, highlighting its rarity and the level of achievement. The successful catch of the booster demonstrates advancements in technology and engineering capabilities, providing a milestone that will be remembered in the engineering field.

Figure 13. More long speech examples results. Lyra achieves more accurate omni-cognition compared to naive VLMs like Qwen2-VL.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

User: Give me more other activation functions not mentioned in today’s lecture, and give me the mathematical expression.

Input: 18min 39s Audio

According to the video lecture, neural networks have shifted from using sigmoid functions, which compress inputs to

values between 0 and 1, to the simpler ReLU function, which outputs either 0 or the input value itself, offering more efficient neuron activation.

Other activation functions that have been used or proposed include Leaky ReLU, which allows a small, non-zero gradient when the neuron is not activated, and ELU (Exponential Linear Unit), which uses an exponential function to provide negative values for inputs below zero. These functions help introduce non-linearity into the network, enabling it to learn more complex patterns.

|Leaky ReLU: 𝐦𝐚𝐱 𝟎,𝒙 + 𝜶 𝐦𝐢𝐧(𝟎,𝒙)<br><br>|
|---|

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Input: 2h 27min 55s Audio User: What is the topic of today’s discussion?

- • Today’s discussion is about the evolution and formation of stars within the universe.
- • Initially, the speaker discusses the shift in understanding from stars being the primary focus of astronomy to recognizing the significant roles played by dark energy and dark matter.
- • The discussion covers the origins of stars, the processes involved in their formation, and the various environments in which they are born.
- • The speaker explains how cold molecular clouds and the interstellar medium play crucial roles in condensing into stars.
- • The talk also delves into the lifecycle of stars, including their eventual transformation into planetary nebulae and the role of magnetic fields in their evolution.
- • Furthermore, the speaker highlights the challenges and advancements in observing and studying these phenomena, including the use of radio and millimeter telescopes.
- • The discussion concludes with insights into the formation of star clusters and the dynamics of stars moving through the galaxy.

Figure 14. More examples of Lyra with hour-long lectures (more than two hours).

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Input: 10min 25s Audio

User: What steps can be taken to make the smart speaker?

The speaker can be made by:

- 1. Choosing a wooden cabinet and purchasing scrap wood for less than a dollar. After returning to the wood mill, better quality wood is bought for around two dollars more. Carpentry tools are purchased at a bargain price, including a handsaw, hole-making saw, wood glue, and sandpapers of different grits.
- 2. The wood is marked, cut, and sanded, then drilled for speakers. The pieces are joined with wood glue and gaps filled with a mixture of wood dust and glue. The LED notification ring is made from a silicon mold and epoxy resin.
- 3. The electronics are built using a Raspberry Pi Zero W, a PCB, and various components, including a power bank and audio amplification. The circuit is soldered, and the microphone, speakers, battery, and Raspberry Pi are attached. Alexa is installed by creating an account on the Amazon Developer page, and the Google Assistant is enabled through the Google Developer page.
- 4. The speaker is assembled inside a prepared cabinet, and the system is tested with Alexa and Google Home functionalities.

[Figure 168]

[Figure 169]

[Figure 170]

Input: 10min 25s Audio

User: What do you think is the most error-prone step?

The most error-prone step in the process described in the transcript is likely the soldering of the components onto the PCB. This step is described as challenging, especially when dealing with SMD (Surface Mount Device) components. Soldering is a delicate process that requires precision and attention to detail to ensure that each component is correctly

placed and connected to avoid shorts or loose connections. The speaker notes that they will consider creating a dedicated

video on SMD soldering, indicating that it’s an area that might need extra care and practice to avoid mistakes.

[Figure 171]

[Figure 172]

[Figure 173]

Input: 10min 25s Audio

User: What are the advantages of the speaker mentioned?

The speaker has several advantages:

- • It is a single device that combines the functionalities of both Amazon Alexa and Google Home, allowing users to control multiple smart home devices and functions.
- • It has built-in battery power, which means it can be used anywhere without needing a power supply, unlike the traditional Amazon Echo or Google Home devices.
- • Additionally, it has audio in and out functionality, which was achieved through a custom circuit board designed to work with the Raspberry Pi Zero W.
- • The speaker is also aesthetically pleasing, with a wooden cabinet that was crafted by the speaker himself, including an LED notification ring.
- • building a smart speaker costs a low budget of thirty dollars, including sourcing materials like wood and tools.

Figure 15. More results from long speech examples: Lyra can subjectively answer questions about complex steps.

#### References

- [1] 2noise. ChatTTS. https://github.com/2noise/ ChatTTS, 2024. 5, 9, 10, 12
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS,

2022. 1

- [3] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. SPICE: Semantic propositional image caption evaluation. In ECCV, pages 382–398. Springer, 2016. 11
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023. 2
- [5] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2023. 2
- [6] Kai Chen, Yunhao Gou, Runhui Huang, Zhili Liu, Daxin Tan, Jing Xu, Chunwei Wang, Yi Zhu, Yihan Zeng, Kuo Yang, et al. EMOVA: Empowering language models to see, hear and speak with vivid emotions. arXiv preprint arXiv:2409.18042, 2024. 1, 2, 5, 6
- [7] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In ECCV, pages 19–35. Springer, 2025. 2
- [8] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. LongLoRA: Efficient fine-tuning of long-context large language models. In ICLR,

2024. 2

- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to GPT-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 5
- [10] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024. 1, 5
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards generalpurpose vision-language models with instruction tuning. In NeurIPS, 2023. 1
- [12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The LLaMA 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 2, 11

- [13] Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. LLaMA-Omni: Seam-

- less speech interaction with large language models. arXiv preprint arXiv:2409.06666, 2024. 1, 2, 4, 11
- [14] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv:2306.13394, 2023. 7
- [15] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 2, 7
- [16] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. VITA: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024. 1, 2, 4, 5, 6
- [17] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive KV cache compression for LLMs. arXiv preprint arXiv:2310.01801, 2023. 2
- [18] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. ImageBind: One embedding space to bind them all. In CVPR, pages 15180–15190, 2023. 11
- [19] F´elix Gontier, Romain Serizel, and Christophe Cerisara. Automated audio captioning by fine-tuning bart with audioset tags. In DCASE 2021-6th Workshop on Detection and Classification of Acoustic Scenes and Events, 2021. 13
- [20] Google. Gemma: Introducing new state-of-the-art open models. https://blog.google/technology/ developers/gemma-open-models/, 2024. 1
- [21] Alex Graves, Santiago Fern´andez, Faustino Gomez, and J¨urgen Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In ICML, pages 369–376, 2006. 12
- [22] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing, 29: 3451–3460, 2021. 11
- [23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. ICLR, 2021. 4
- [24] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 1
- [25] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv:2401.04088, 2024. 1
- [26] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. AudioCaps: Generating captions for audios

- in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, 2019. 11
- [27] Eungbeom Kim, Jinhee Kim, Yoori Oh, Kyungsu Kim, Minju Park, Jaeheon Sim, Jinwoo Lee, and Kyogu Lee. Exploring train and test-time augmentations for audio-language learning. arXiv preprint arXiv:2210.17143, 2022. 13
- [28] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 2, 5
- [29] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 1
- [30] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. MVBench: A comprehensive multi-modal video understanding benchmark. In CVPR, pages 22195–22206, 2024. 7
- [31] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-Gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 1, 2, 5, 9

- [32] Yanwei Li, Chengyao Wang, and Jiaya Jia. LLaMA-VID: An image is worth 2 tokens in large language models. In ECCV, pages 323–340. Springer, 2025. 2
- [33] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: On pre-training for visual language models. In CVPR, pages 26689–26699, 2024.
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeruIPS, 2023. 1
- [35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26296–26306, 2024.
- [36] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, ocr, and world knowledge, 2024. 2, 5
- [37] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of importance hypothesis for LLM KV cache compression at test time. NeruIPS, 36, 2024. 2
- [38] Zhengrui Ma, Qingkai Fang, Shaolei Zhang, Shoutao Guo, Yang Feng, and Min Zhang. A non-autoregressive generation framework for end-to-end simultaneous speech-to-any translation. arXiv preprint arXiv:2406.06937, 2024. 12
- [39] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. NeurIPS, 36:46212– 46244, 2023. 7
- [40] Xinhao Mei, Xubo Liu, Qiushi Huang, Mark D Plumbley, and Wenwu Wang. Audio captioning transformer. arXiv preprint arXiv:2107.09817, 2021. 13
- [41] Microsoft. Edge-TTS. https://github.com/rany2/

edge-tts, 2024. 12

- [42] OpenAI. ChatGPT. https://openai.com/blog/ chatgpt/, 2023. 1, 2
- [43] OpenAI. GPT-4o. https://openai.com/index/ hello-gpt-4o/, 2024. 1, 7
- [44] OpenGVLab. InternOmni: Extending internvl with audio modality. https://internvl.github.io/blog/ 2024-07-27-InternOmni/, 2024. 4, 6, 12
- [45] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. LibriSpeech: an asr corpus based on public domain audio books. In ICASSP, pages 5206–5210. IEEE,

2015. 6, 9

- [46] Adam Polyak, Yossi Adi, Jade Copet, Eugene Kharitonov, Kushal Lakhotia, Wei-Ning Hsu, Abdelrahman Mohamed, and Emmanuel Dupoux. Speech Resynthesis from Discrete Disentangled Self-Supervised Representations. In Proc. Interspeech 2021, 2021. 11
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 11
- [48] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In ICML, pages 28492–28518. PMLR, 2023. 5
- [49] Paul K Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zal´an Borsos, F´elix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, et al. AudioPaLM: A large language model that can speak and listen. arXiv preprint arXiv:2306.12925,

2023. 2

- [50] John T Serences and Steven Yantis. Selective visual attention and perceptual coherence. Trends in cognitive sciences, 10

(1):38–45, 2006. 4

- [51] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR,

2019. 7

- [52] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, MA Zejun, and Chao Zhang. SALMONN: Towards generic hearing abilities for large language models. In ICLR, 2024. 5
- [53] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-aware sparsity for efficient long-context llm inference. arXiv preprint arXiv:2406.10774, 2024. 2
- [54] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. NeurIPS, 36, 2024. 13
- [55] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 1
- [56] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Document collection visual question answering. In ICDAR 2021,

2021. 7

- [57] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation language models. arXiv:2302.13971, 2023. 1, 2
- [58] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In CVPR, pages 4566–4575, 2015. 11
- [59] Common Voice. Common Voice. https : / / commonvoice.mozilla.org/en/datasets, 2024. 9
- [60] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 5, 12
- [61] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. CogVLM: Visual expert for pretrained language models. arXiv:2311.03079, 2023. 2
- [62] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024. 2
- [63] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and TatSeng Chua. Next-GPT: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023. 1, 2
- [64] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023. 2
- [65] Zhifei Xie and Changqiao Wu. Mini-Omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024. 1, 5
- [66] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. LongVILA: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 2
- [67] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 2
- [68] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 2
- [69] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. MM-Vet: Evaluating large multimodal models for integrated capabilities. arXiv:2308.02490, 2023. 7
- [70] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal un-

- derstanding and reasoning benchmark for expert agi. In CVPR, 2024. 7
- [71] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. AnyGPT: Unified multimodal llm with discrete sequence modeling. arXiv preprint arXiv:2402.12226,

2024. 1, 2, 6

- [72] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. SpeechGPT: Empowering large language models with intrinsic cross-modal conversational abilities. arXiv preprint arXiv:2305.11000, 2023. 2
- [73] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 2
- [74] Shaolei Zhang, Qingkai Fang, Shoutao Guo, Zhengrui Ma, Min Zhang, and Yang Feng. Streamspeech: Simultaneous speech-to-speech translation with multi-task learning. arXiv preprint arXiv:2406.03049, 2024. 12
- [75] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher R´e, Clark Barrett, et al. H2O: Heavy-hitter oracle for efficient generative inference of large language models. NeruIPS, 36:34661–34710, 2023. 2
- [76] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing visionlanguage understanding with advanced large language models. arXiv:2304.10592, 2023. 1

