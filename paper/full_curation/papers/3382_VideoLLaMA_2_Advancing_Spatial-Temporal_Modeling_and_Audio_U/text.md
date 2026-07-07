# arXiv:2406.07476v3[cs.CV]30Oct2024

VideoLLaMA 2

[Figure 1]

Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs

Zesen Cheng∗, Sicong Leng∗, Hang Zhang∗, Yifei Xin∗, Xin Li∗, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, Lidong Bing

DAMO Academy, Alibaba Group https://github.com/DAMO-NLP-SG/VideoLLaMA2

## Abstract

In this paper, we present VideoLLaMA 2, a set of Video Large Language Models (Video-LLMs) designed to enhance spatial-temporal modeling and audio understanding in video and audio-oriented tasks. Building upon its predecessor, VideoLLaMA 2 incorporates a tailor-made Spatial-Temporal Convolution (STC) connector, which effectively captures the intricate spatial and temporal dynamics of video data. Additionally, we integrate an Audio Branch into the model through joint training, thereby enriching the multimodal understanding capabilities of the model by seamlessly incorporating audio cues. Comprehensive evaluations on multiple-choice video question answering (MC-VQA), open-ended video question answering (OE-VQA), and video captioning (VC) tasks demonstrate that VideoLLaMA 2 consistently achieves competitive results among open-source models and even gets close to some proprietary models on several benchmarks. Furthermore, VideoLLaMA 2 exhibits reasonable improvements in audio-only and audiovideo question-answering (AQA & OE-AVQA) benchmarks over existing models. All models are public to facilitate further research.

## 1 Introduction

In recent years, the field of artificial intelligence (AI) has achieved significant advancements (OpenAI, 2023a; Google, 2023; Anthropic, 2024), profoundly transforming industries and societal functions across the board. Models capable of image recognition (Bai et al., 2023; Dong et al., 2024; Chen et al., 2023d; Liu et al., 2024c; Chen et al., 2023b; Lin et al., 2023c; Young et al., 2024) and photorealistic image generation (Esser et al., 2024; Saharia et al., 2022; Ramesh et al., 2021) have approached near-human capabilities, catalyzing major breakthroughs in sectors such as medical imaging (Li et al., 2024b; Tu et al., 2024) and autonomous driving (Xu et al., 2023b; Jin et al., 2023a). Despite these successes, the domain of video understanding and generation (Kondratyuk et al., 2023; Menapace et al., 2024; Kondratyuk et al., 2023; Brooks et al., 2024) remains relatively nascent. Unlike static images, videos incorporate temporal dynamics and synchronous audio streams, significantly enriching the information content. This integration of continuous audio-visual data complicates extracting and interpreting meaningful patterns, amplifying data complexity and introducing unique computational challenges.

∗ ZC, SL, HZ, YX, and XL contributed equally to this project.

Preprint. Work in progress

While Image Large Language Models (Image-LLMs) (Alayrac et al., 2022; Li et al., 2023a; Zhu et al., 2023b; Liu et al., 2024c; Ye et al., 2023; Bai et al., 2023; Chen et al., 2023d; Dong et al., 2024) processing static images have matured with impressive capabilities, Video Large Language Models (Video-LLMs) (Li et al., 2023b; Zhang et al., 2023; Maaz et al., 2023a; Lin et al., 2023a; Wang et al., 2024c; Liu et al., 2024a) lag notably behind due to inherent complexities. The primary challenge in video understanding lies in the temporal dynamics—recognizing visual patterns while understanding changes over time and correlating these with synchronous audio inputs. These temporal dynamics complicate the accurate prediction of future states and the understanding of complex scenarios, such as interactions among multiple entities or subtle environmental changes. Moreover, the integration of audio with visual data, essential for comprehensive understanding, remains underdeveloped in current models, limiting their effectiveness.

Current Video-LLMs are constrained by several limitations that affect their performance and utility. Firstly, these models often struggle with effectively processing temporal dynamics due to their limited capabilities in fusing features across different frames. This results in a failure to fully capitalize on the available temporal information, hindering their ability to predict future events accurately based on past and present data. Secondly, the integration of audio streams is frequently overlooked, despite audio being a rich source of contextual cues that are vital for a complete scene understanding. This neglect leads to a significant gap in models’ ability to perform comprehensive multimodal analyses. These limitations illustrate the need for more advanced Video-LLMs that can handle the complexities of multimodal video data without compromising processing efficiency or contextual integrity.

This technical report unveils the VideoLLaMA 2, a set of generalist Video-LLMs designed to enhance video-language understanding by integrating and interpreting the complex interplay of visual and auditory signals. Built on the technical foundations established by earlier models (Radford et al., 2021; Radosavovic et al., 2020; Chen et al., 2023c; Cha et al., 2024; Jiang et al., 2023), VideoLLaMA 2 delivers a system capable of not only comprehending but also articulating the rich narratives inherent in video content. The model’s robust performance is anchored by its ability to effectively process temporal dynamics, achieved through the implementation of our specially designed Spatial-Temporal Connector module. This allows VideoLLaMA 2 to excel in various video-language tasks, from video captioning to complex question answering, demonstrating a profound comprehension of video content. Further contributing to its effectiveness is the joint-trained Audio Branch, which enhances VideoLLaMA 2’s capacity for advanced audio-visual integration. This feature ensures that audio data, often underutilized in video language models, significantly bolsters the interpretative depth, capturing subtle cues lost in visual-only analyses.

These technological enhancements make VideoLLaMA 2 a pivotal development in videolanguage analytics, setting a new standard for the capabilities of intelligent video understanding systems. The subsequent sections of this report will delve into the technical architecture of the model, explore the innovative methodologies employed, and present a detailed evaluation of its performance, illustrating its superiority over existing models.

## 2 Method

### 2.1 Architecture

As depicted in Figure 1, VideoLLaMA 2 adheres to the design principle established in its previous version (i.e., VideoLLaMA (Zhang et al., 2023)), which integrates a dual-branch framework comprised of a Vision-Language Branch and an Audio-Language Branch. Both branches operate independently, connecting pre-trained visual and audio encoders to an instruction-finetuned large language model in a modular fashion. This modality-specific independence of the visual and audio branches, with cross-modal interactions occurring solely within the highly capable language model, not only allows streamlined training by preserving the integrity of individual modal inputs, but also facilitates future expansions and adaptations.

[Figure 2]

[Figure 3]

[Figure 4]

VideoLLaMA 2: The video features a kitten and a baby chick playing together. They are seen cuddling, playing, and even taking a nap together. The video has a very cute and heartwarming feel to it, as the two animals seem to have formed a close bond.

[Figure 5]

[Figure 6]

Pre-trained Large Language Model

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Prompt: What animals are in the video, what are Projection W

Projection W

they doing, and how does the video feel?

[Figure 13]

[Figure 14]

|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]|
|---|

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

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

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

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Flatten Flatten

[Figure 90]

|[Figure 91]<br><br>[Figure 92]<br><br>Spatial Convolution|
|---|

Visual Encoder

Audio Encoder

[Figure 93]

[Figure 94]

[Figure 95]

|[Figure 96]<br><br>[Figure 97]<br><br>Spatial-Temporal Downsampling|
|---|

|[Figure 98]<br><br>[Figure 99]|
|---|

|[Figure 100]<br><br>[Figure 101]<br><br>Spatial Convolution|
|---|

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Audio

Video Frames Encoding STC connector

- Figure 1: The overall pipeline of VideoLLaMA 2. For the vision-language branch, the video frames are encoded into features frame by frame, processed through our STC connector, and then these features are fed into a large language model to generate responses based on text prompts. For the audio-language branch, audio signals are first transformed into log mel spectrograms, which are then encoded to extract auditory features. These features are then processed through a multilayer perceptron (MLP) block to align the audio modalities with the large language model.

Vision-Language Branch Although video modality is our main focus, we choose imagelevel CLIP (ViT-L/14) (Radford et al., 2021) as our vision backbone. The main reason is that image encoders are compatible with arbitrary frame sampling strategies and enable a more flexible frame-to-video feature aggregation scheme, as observed in Xue et al. (2023). During training and inference, we adopt a consistent frame sampling approach that extracts a fixed number of frames from each video. Each frame undergoes padding and resizing to a standardized 336x336 dimension. The preprocessed frames are then fed into the image encoder. Instead of the Q-former in VideoLLaMA 1 (Zhang et al., 2023), we propose a Spatial-Temporal Convolution Connector (STC Connector) in VideoLLaMA 2 for spatialtemporal representation learning. The STC Connector could preserve spatial and temporal local details more effectively than the Q-former while not producing a large number of video tokens. A detailed exploration of the working mechanism and advantages of the STC Connector is provided in Section 2.1.

Audio-Language Branch With an established foundation in the visual domain, our exploration extends into the auditory realm to enhance the multimodal capabilities of VideoLLaMA. Initially, audio signals undergo a preprocessing step where they are transformed into fbank spectrograms with 128 frequency bins. To effectively harness these preprocessed audio signals, we integrate BEATs (Chen et al., 2023c), a cutting-edge audio encoder, known for its exceptional ability to capture detailed audio features and temporal dynamics. These features are then processed through a MLP block with two linear layers to align with the dimension of LLMs, therefore providing a more cohesive understanding of the video content when combined with the visual and acoustic modalities. By incorporating BEATs into VideoLLaMA, we address the challenge of synchronizing audio-visual data points. The encoder’s ability to capture temporal dynamics aligns with the STC Connector employed in the visual branch, ensuring a seamless integration of audio-visual features.

Large Language Model Backbone Like its predecessor, VideoLLaMA 2 adopts the instruction-following large language models (LLMs) as its language decoder. We do not extensively search the optimal LLMs for VideoLLaMA 2 but use Mistral-Instruct (Jiang et al.,

|[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]|
|---|

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Pooling/ Convolution

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |[Figure 127]<br><br>| | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

[Figure 128]

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

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

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

Visual Encoder

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Flatten

[Figure 170]

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| |[Figure 171]<br><br>| | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

[Figure 172]

[Figure 173]

Frozen

[Figure 174]

Video Frames Spatial Interaction Spatial-Temporal Aggregation

Learnable

Spatial Interaction

- Figure 2: The pipeline of STC connector. The video frames are first encoded into features frame by frame, and then processed via our proposed STC connector (two spatial interaction modules and one spatial-temporal aggregation module). We adopt RegStage to implement "Spatial interaction" and 3D convolution to implement "Spatial-Temporal Aggregation".

MV-Bench Egoschema ActivityNet-QA

RegStage Downsample # Tokens

Video QA (Acc.) Long Video QA (Acc.) Open-ended Video QA (Acc.) Avg.

(Li et al., 2023c) (Mangalam et al., 2024) (Yu et al., 2019)

- 2D Pool (1, 2, 2) 1152 46.9 36.5 49.8 44.4

- 2D Conv (1, 2, 2) 1152 43.9 32.6 48.6 41.7

- 3D Pool (8, 1, 1) 576 43.7 38.7 47.5 43.3

- 3D Pool (2, 2, 2) 576 44.5 41.1 48.1 44.6 3D Conv (2, 2, 2) 576 44.8 39.0 45.6 43.1

- ✔ 2D Pool (1, 2, 2) 1152 46.7 35.3 46.9 43.0

- ✔ 2D Conv (1, 2, 2) 1152 45.3 33.5 48.0 42.3

- ✔ 3D Pool (2, 2, 2) 576 44.4 39.1 46.0 43.2

- ✔ 3D Conv (2, 2, 2) 576 45.5 42.2 47.6 45.1

Table 1: Empirical Study of STC Connector design choices: Spatial Interaction (RegStage), Spatial-Temporal Aggregation (Downsample). To accelerate such empirical exploration, we train all of the models on the dataset provided by Video-LLaVA (Lin et al., 2023a). The green line represents our chosen STC connector design. Furthermore, we default to sample 8 frames in this study.

2023), Mixtral-Instruct (Jiang et al., 2024) and Qwen2-Instruct (Qwen, 2024) across all of the experiments. We leave the exploration of other popular LLMs, such as Gemma-IT (Gemma et al., 2024) and LLaMA3-Instruct (Dubey et al., 2024), for future work. We also include the results of VideoLLaMA 2 with Qwen2-Instruct (Qwen, 2024) as language decoder in Section 5.

Spatial-Temporal Convolution Connector The overall architecture and the workflow of our STC connector are depicted in Figure 2. We primarily follow three principles when designing the video-language connector: 1) Maintaining the spatial-temporal order in the output visual tokens; 2) Reducing the number of spatial-temporal tokens; 3) Alleviating information loss during spatial-temporal downsampling. ❶ The first principle guides us to avoid using resampler architecture (Li et al., 2023a; Alayrac et al., 2022) because the resampling operation does not guarantee the preservation of spatial-temporal order. This may cause suboptimal convergence because autoregressive models (i.e., LLM backbone) are highly dependent on consistent token order between training and inference. Therefore, we only consider the operations of convolution or pooling when building our connector. ❷ According to the second principle, we insert 3D downsample operator to compress spatialtemporal tokens. ❸ To complement the information loss caused by the spatial-temporal downsampling, we insert RegStage (a strong convolution block (Radosavovic et al., 2020)) before and after spatial-temporal downsampling. To empirically investigate the effectiveness

Algorithm 1: PyTorch pseudo-code for STC connector

- 1 import torch.nn as nn

- 2 from timm.models.regnet import RegStage

- 3

- 4 class STCConnector(nn.Module):

- 5 def __init__(self, config, depth, mlp_depth):

- 6 # Temporal and spatial downsampling factor

- 7 td, sd = config.td, config.sd

- 8 # Input and output hidden dimension

- 9 in_size, out_size = config.in_size, config.out_size

- 10 # The first RegStage block

- 11 self.s1 = RegStage(

- 12 depth=depth, in_chs=in_size, out_chs=out_size)

- 13 # Conv3D downsampler

- 14 self.downsampler = nn.Conv3d(

- 15 in_channels=out_size,

- 16 out_channels=out_size,

- 17 kernel_size=(td, sd, sd))

- 18 # The second RegStage block

- 19 self.s2 = RegStage(

- 20 depth=depth, in_chs=out_size, out_chs=out_size)

- 21 self.proj = build_mlp(mlp_depth, out_size, out_size)

- 22

- 23 def forward(self, x):

- 24 x = self.s1(x)

- 25 x = self.downsampler(x)

- 26 x = self.s2(x)

- 27 x = self.proj(x)

- 28 return x

of the designs above, we establish a quick but reasonable architectural search using the training data from Video-LLaVA (Lin et al., 2023a). We select three representative video understanding benchmarks: Egoschema (Mangalam et al., 2024), MV-Bench (Li et al., 2023c) and ActivityNet-QA (Yu et al., 2019) as the testbed2 and the comparison results are listed in Table 1. As can be seen, 3D convolution together with the ReStage block (i.e., the green line), which forms our STC connector, works the best in terms of the average performance. Another interesting finding is that almost all of the 3D downsampling designs perform better than the 2D ones on Egoschema, suggesting that the early fusion of frame-level features is beneficial for long video understanding.

## 3 Training

In this section, we detail the training process for the Video-Language and Audio-language branches, followed by the joint training that integrates both modalities.

- 3.1 Video-Language Training

- 3.1.1 Pre-training

For the pre-training stage, we utilize a large-scale, weakly labeled, web-crawled dataset of image-text and video-text pairs, sourced from several publicly accessible databases. The video-text sources include Panda-70M (Chen et al., 2024c), VIDAL-10M (Zhu et al., 2023a), WebVid-10M (Bain et al., 2021), and InternVid-10M (Wang et al., 2023b), while the imagetext sources include CC-3M (Changpinyo et al., 2021) and DCI (Urbanek et al., 2023).

2See Section 5.1 for benchmark details and evaluation protocols.

##### Modality Dataset Original Used Ratio%

Panda-70M (Chen et al., 2024c) 70M 2.8M 4%

WebVid-10M (Bain et al., 2021) 10M 4M 40% VIDAL-10M (Zhu et al., 2023a) 10M 2.8M 28%

Video-Text

InternVid-10M (Wang et al., 2023b) 10M 650K 6.5%

CC-3M (Changpinyo et al., 2021) 3M 595K 19.8% Image-Text

DCI (Urbanek et al., 2023) 7.8K 7.8K 100% Vision-Language Total 103M 12.2M 11.8%

- Table 2: Video-Language Pre-training Data Statistics. We directly adopt the filtered version of CC-3M from Liu et al. (2023b).

|Modality<br><br>|Task|# Samples Dataset|
|---|---|---|
|Video-Text|Captioning Classification<br><br>VQA<br><br>Instruction|23K VideoChat, In-house data 79K Kinetics-710, SthSthv2<br><br>161K<br><br>NExTQA, CLEVRER, EgoQA, Tgif, WebVidQA, RealworldQA, Hm3d<br><br>225K<br><br>Valley, VideoChatGPT, VideoChat, VTimeLLM, VideoChat2|

Captioning 82K ShareGPT4V Image-Text VQA 198K LLaVA

Instruction 466K LLaVA Pure Text Instruction 120K Magpie, ALLaVA

Table 3: Video-Language Multi-task Fine-tuning Data Statistics.

During this stage, the vision encoder and the large language model are frozen, and only the connector is optimized. The input video frames are evenly sampled and resized to 336 × 336 pixels. The training objective is to minimize the cross-entropy loss of the text tokens.

3.1.2 Multi-task Fine-tuning

In the second stage of multi-task fine-tuning, we incorporate high-quality, fine-grained multi-modal annotations using both video-text and image-text data. As summarized in

- Table 3 , VideoLLaMA 2 is fine-tuned on four tasks simultaneously. For video captioning, we use training samples from VideoChat (Li et al., 2023b) and in-house collected data. For video classification and VQA tasks, we utilize a mixture of publicly available datasets, including Kinetics-710 (Kay et al., 2017), SthSthv2 (Goyal et al., 2017), NExtQA (Xiao et al., 2021), CLEVRER (Yi et al., 2019), EgoQA (Fan, 2019), Tgif (Li et al., 2016), WebVidQA (Yang et al., 2021), RealworldQA (X.AI, 2024), and Hm3d (Ramakrishnan et al., 2021). To enhance instruction-following capabilities, we collect and integrate data from multiple frontier VideoLLMs, such as Valley (Luo et al., 2023), VideoChatGPT (Maaz et al., 2023b), VideoChat (Li et al., 2023b), VTimeLLM (Huang et al., 2023), and VideoChat2 (Li et al., 2023c) to improve generalization. Additionally, we mix image captioning data from ShareGPT4V (Chen et al., 2023b) and image VQA and instruction-following data from LLaVA (Liu et al., 2023b,a) to maintain the capabilities in understanding static visual concepts. To further improve the instruction-following abilities, we incorporate pure text data from Magpie (Xu et al., 2024b) and ALLaVA (Xu et al., 2023a; Chen et al., 2024a).

In this stage, the visual encoder is frozen, and we optimize the language model and spatialtemporal connector. The training objective remains consistent with the pre-training stage.

Multi-stage # Samples Data Sources Pre-training 400K WavCaps

ClothoAQA, WavCaps, AudioCaps, Clotho, MusicCaps, VGGSound, UrbanSound8K, ESC50, TUT2017, TUT2016, VocalSound

Instruction Tuning 698K

AVQA, AVQA-music, AVSD, VGGSound, AVinstruct, MusicCaps, AudioCaps, VocalSound, WavCaps, UrbanSound8K, Clotho, ClothoAQA, TUT2017, TUT2016, Evol-Instruct, SthSthv2, LLaVA, Kinetics-710, VideoChat, NExTQA, Valley, EgoQA, CLEVRER, Tgif, ShareGPT4V, In-house data

Audio-Video Joint Training 836K

Total 1.9M -

Table 4: Datasets used in multi-stage audio-language training.

### 3.2 Audio-Language Training

Audio-language training of our VideoLLaMA 2 starts with the language decoder being initialized as the language model of the fine-tuned video model in Sec. 3.1.2. Similar to video-language training, audio-language training undergoes audio-language pre-training and multi-task fine-tuning.

### 3.2.1 Pre-training

In the initial phase, we focus on the foundational aspect of audio understanding by leveraging WavCaps (Mei et al., 2023), a comprehensive dataset comprising approximately 400,000 audio samples. Each sample is meticulously annotated for audio captioning tasks, aimed at training models to generate descriptive text based on audio inputs. This dataset serves as a crucial base for our models to learn intricate patterns in audio data, thereby preparing them for more complex audio-language tasks.

In this stage, the audio encoder and the large language model (LLM) are frozen, concentrating optimization exclusively on the audio projector. The primary training objective is to minimize the next token prediction loss over the textual responses, enhancing the model’s capability to understand and map audio data to textual representations. This approach ensures that the audio processing components effectively leverage the optimized language model to achieve more precise audio-text alignment.

### 3.2.2 Multi-task Fine-tuning

The second stage of audio-language training aims to enhance the versatility and applicability of our model through multi-task learning, involving a variety of datasets tailored to different audio processing tasks. ClothoAQA (Lipping et al., 2022b), with about 1,500 entries, is utilized for refining our model’s capabilities in question answering based on audio cues, with each sample enriched by six associated questions sourced through crowdsourcing. The instruction tuning phase also capitalizes on the continued use of WavCaps (Mei et al., 2023), alongside AudioCaps (Kim et al., 2019) and Clotho (Drossos et al., 2020), which together contribute about 453,000 audio samples for audio-text training. MusicCaps (Agostinelli et al., 2023), comprising roughly 5,400 entries, extends our model’s exposure to the musical domain, specializing in music-based captioning. For the purpose of sound event classification, we engage with VGGSound (Chen et al., 2020), which offers over 190,000 labeled audio clips of diverse acoustic events. UrbanSound8k (Salamon et al., 2014) and ESC50 (Piczak, 2015) provide additional layers of complexity with 8000+ and 2,000 urban and miscellaneous environmental sounds, respectively. TUT2017 (Mesaros et al., 2017), TUT2016 (Mesaros

et al., 2016b), and VocalSound (Gong et al., 2022) enrich our dataset collection with more than 4,000, 1100, and 15,000 samples respectively, focusing on general sound events and human vocal sounds classification, thus broadening our model’s acoustic perceptual skills.

At this stage, the audio encoder and audio projector are optimized, with the LLM remaining frozen, maintaining a consistent training objective to minimize text label cross-entropy loss as established in the pre-training stage.

### 3.3 Audio-Video Joint Trainng

The third phase shifts our focus towards the integration of audio and visual modalities, aiming to harness and understand the interactions between these two crucial aspects of multimodal perception. This phase incorporates audio-visual datasets including AVQA (Yang et al., 2022) and AVQA-music (Li et al., 2022b), featuring around 57,000 and 35,000 entries, respectively. These datasets are specifically designed for audio-visual question answering, challenging our model to not only perceive but also interpret cross-modal content effectively. The AVSD (Alamri et al., 2019) dataset, consisting of approximately 80,000 pairs, is pivotal for developing and refining audio-visual dialogue systems. Furthermore, VGGSound (Chen et al., 2020) reappears with a substantial subset dedicated to audio-visual classification tasks, reinforcing the dataset’s utility and versatility in our multimodal training framework. Additionally, we incorporate the AVInstruct (Ye et al., 2024a) dataset, an audiovisual instruction dataset emphasizing co-learning on dynamic audio-visual pairs to address diverse AVQA tasks.

In addition to these audio-visual datasets, we also leverage several video-only datasets that enrich our system’s capacity to process and analyze diverse forms of input. For video-text pairs, our resources include SthSthv2 (Goyal et al., 2017) with 20K entries, EgoQA (Fan, 2019) featuring 7.8K, Tgif (Li et al., 2016) contributing 24.7K, Kinetics-710 (Kay et al., 2017) incorporating 19.5K, and VideoChat2 (Li et al., 2023c) with 9.5K. Other video-text datasets such as CLEVRER (2K) (Yi et al., 2019) and Valley (35.5K) (Luo et al., 2023) enhance our model’s robustness in understanding video content in conjunction with textual cues. Additionally, NExTQA (Xiao et al., 2021), though smaller with 1K pairs, serves as a valuable supplement. To further enhance the image-text interaction, we incorporate image-text datasets such as ShareGPT4V (Chen et al., 2023b) with a substantial 82K entries and LLaVA (Liu et al., 2023b,a) using 62K pairs, providing the necessary frameworks for visual comprehension alongside textual analysis.

Alongside these, we incorporate a broad range of audio-text pair datasets to bolster our model’s multimodal capabilities. Specifically, we include AudioCaps (Kim et al., 2019) with 49.3K entries, VocalSound (Gong et al., 2022) offering 15K, and UrbanSound8k (Salamon et al., 2014) with 8K, which supply extensive audio-text data for urban and vocal soundscapes. Additionally, we utilize Clotho (Drossos et al., 2020) with 19.3K entries, Wavcaps (Mei et al., 2023) with 100K, MusicCaps (Agostinelli et al., 2023) providing 5.4K, and niche datasets such as TUT2017 (Mesaros et al., 2017) and TUT2016 (Mesaros et al., 2016b) with 4.6K and 1.1K entries respectively, which focus on various environmental sounds. Furthermore, ClothoAQA (Lipping et al., 2022b) with 1.5K pairs adds to our resources for query-answering capabilities within audio-text contexts. These datasets collectively enable our model to engage with a rich variety of sound types and contexts, making it adept at handling complex audio-related tasks.

Finally, our text dataset, evol-instruct (Xu et al., 2023a), consisting of 23.2K examples, rounds out our multimodal approach, facilitating a comprehensive training environment that ensures our models can effectively follow instructions and interpret a wide range of multimodal inputs.

In this stage, audio tracks are extracted from videos and cut to align the video clips. These audio clips are then truncated or padded to the same duration as the audio-language traning stage. For videos that lack an audio track, we fill the waveform with zeros to ensure uniformity across all data samples. During this training phase, we sample data for each batch in a 2:1:1 ratio of audio-visual data to visual data to audio data, ensuring that all audio-visual data is covered in one epoch. The video encoder remains frozen while we

optimize the audio/video projector and the audio encoder, alongside the unfrozen LLM. The training objectives align with the instruction tuning stage, ensuring a coherent and effective progression in our multimodal training approach. By effectively leveraging the synchronized audio-visual data, VideoLLaMA 2 achieves a deeper understanding of multimodal content, thereby enhancing its performance across a spectrum of multimedia analysis tasks.

## 4 Implementation

Our VideoLLaMA2 is built upon LLaVA 1.5 library (Liu et al., 2023a)3. Across all VideoLLaMA 2 variants, we use clip-large-3364 as the primary visual encoder, though siglip-so400m-3845 is preferred in later variants due to its superior performance. For audio encoding, we utilize Fine-tuned BEATs_iter3+(AS2M)(cpt2)6. The language decoders are initialized with either Mistral-7B-Instruct7, Mixtral-8x7B-Instruct8, or, in some model variants, Qwen2-7B-Instruct9 and Qwen2-72B-Instruct10. The audio branch training utilizes the video LLM initialized using the weights of the VideoLLaMA 2 (7B) model, which is trained on the two-stage video branch with 16 frames. We do not do any hyperparameter tuning during both pre-training and fine-tuning. Instead, we empirically set the global batch size and the learning rate as 1,024 and 1e-3 for pre-training and 2,048 and 2e-5 for fine-tuning. For the video-only training stage, VideoLLaMA 2 is pre-trained for just one epoch, followed by a fine-tuning process lasting up to three epochs. In the audio-only training, we also pre-train VideoLLaMA 2 for a single epoch, but limit fine-tuning to two epochs. For the audio-visual joint training stage, the pre-trained model undergoes fine-tuning for up to two epochs.

## 5 Model Evaluation

In this section, we present a comprehensive evaluation of VideoLLaMA 2, comparing it with other frontier models on various video and audio understanding benchmarks. The evaluation includes both quantitative metrics and qualitative analyses, highlighting the strengths and advancements of our model in handling complex multimodal data.

- 5.1 Video Understanding

- 5.1.1 Evaluation Benchmarks

We conduct extensive evaluations on Multi-choice Video Question Answering (MC-VQA), Open-Ended Video Question Answering (OE-VQA), and Video Captioning (VC) tasks to systematically assess the video understanding capabilities of VideoLLaMA 2.

MC-VQA For the MC-VQA task, we select EgoSchema (Mangalam et al., 2024), PerceptionTest (Patraucean et al., 2024), MV-Bench (Li et al., 2023c), and VideoMME (Fu et al., 2024). We report the top-1 accuracies for all benchmarks. For VideoMME, due to some unknown issues when getting the subtitles, we only report the results under the setting of “w/o subs”.

OE-VQA For open-ended question answering, we conduct comparative studies using the MSVD-QA (Xu et al., 2016), ActivityNet-QA (Yu et al., 2019), and Video-ChatGPT Maaz et al. (2023b) benchmarks. Following the protocols of Maaz et al. (2023b), we employ a

- 3https://github.com/haotian-liu/LLaVA
- 4https://huggingface.co/openai/clip-vit-large-patch14-336
- 5https://huggingface.co/google/siglip-so400m-patch14-384
- 6https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea
- 7https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- 8https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
- 9https://huggingface.co/Qwen/Qwen2-7B-Instruct
- 10https://huggingface.co/Qwen/Qwen2-72B-Instruct

|Model # Frames<br><br>|MC-VQA<br><br>|VC|
|---|---|---|
| |EgoSchema Perception-Test MVBench VideoMME (Acc.) (Acc.) (Acc.) (Acc.)<br><br>|MSVC (Score) correctness detailedness|

Proprietary Models Gemini 1.0 Pro (Google, 2023) - 55.7♥ 51.1♥ - - - Gemini 1.0 Ultra (Google, 2023) - 61.5♥ 54.7♥ - - - Gemini 1.5 Flash (Google, 2024) - - - - 70.3/75.0♢ 3.46♠ 3.24♠ Gemini 1.5 Pro (Google, 2024) - 63.2♥ - - 75.0/81.3♢ 3.67♠ 3.52♠ GPT4-V (OpenAI, 2023b) - 55.6♥ - 43.7♢ 59.9/63.3♢ 2.70♠ 2.76♠ GPT4-O (OpenAI, 2024) - 72.2♥ - - 71.9/77.2♢ - Reka-Flash (Reka, 2024) - - 56.4♥ - - - Reka-Core (Reka, 2024) - - 59.3♥ - - 2.61♠ 2.73♠

Open-source Models LLaMA-VID (7B) 1 fps 38.5♠ 44.6♠ 41.9♠ 25.9/-♠ 1.84♠ 2.11♠ Video-LLaVA (7B) 8 38.4♠ 44.3♠ 41.0♠ 39.9/41.6♢ 1.85♠ 2.05♠ VideoChat2 (7B) 16 42.2♠ 47.3♠ 51.1♥ 33.7/-♢ 2.01♠ 2.10♠ LLaVA-NeXT-Video (7B) 32 43.9♠ 48.8♠ 46.5♠ - 2.40♠ 2.52♠ LLaVA-NeXT-Video (32B) 32 60.9♥ - - 60.2/63.0♥ - PLLaVA (34B) 16 - - 58.1♥ - - VILA 1.5 (34B) - 58.0♠ - - 62.3/64.1♢ - LLaVA-OneVision (72B) 32 62.0♥ - 59.5♥ 66.3/69.6♥ - VideoLLaMA2 (7B) 8 50.5 49.6 53.4 45.1/46.6 2.57 2.61 VideoLLaMA2 (7B) 16 51.7 51.4 54.6 47.9/50.3 2.53 2.59 VideoLLaMA2 (8x7B) 8 53.3 52.2 53.9 47.9/49.7 2.53 2.56 VideoLLaMA2 (72B) 16 63.9 57.5 62.0 61.4/63.1 2.71 2.67 VideoLLaMA2.1 (7B) 16 53.1 54.9 57.3 54.9/56.4 2.87 2.81

- Table 5: Main Results on Multiple-Choice Video QA (MC-VQA) and Video Captioning (VC). We do not follow the latest version of Gemini 1.5 (Google, 2024) to invoke explicit CoT on EgoSchema, which could give large performance gains. ♥: officially reported results. ♢: results retrieved from the leaderboard. ♠: results reproduced by ourselves.

GPT-assisted evaluation to assess the quality of the generated answers. Specifically, GPT-3.5 provides a binary "Yes-or-No" decision on the correctness of answers, and we report the percentage of "Yes" responses as Accuracy.

VC For the video captioning task, we perform experiments on the newly introduced MultiSource Video Caption (MSVC) benchmark. MSVC is introduced to address limitations in existing video caption benchmarks, MSVC samples 500 videos with human-annotated captions from MSVD (Chen & Dolan, 2011), MSRVTT (Xu et al., 2016), and VATEX (Wang

- et al., 2019), ensuring diverse scenarios and domains. Traditional evaluation metrics rely on exact match statistics between generated and ground truth captions, which are limited in capturing the richness of video content. Thus, we use a ChatGPT-assisted evaluation similar to Maaz et al. (2023b). Both generated and human-annotated captions11 are evaluated by GPT-3.5-turbo (0613) for Correctness of Information and Detailed Orientation 12.

### 5.1.2 Baselines

To comprehensively evaluate the performance of VideoLLaMA 2, we compare it against a diverse set of baselines, including both proprietary and open-source frontier models. The included models are listed below:

- 11Each video in the MSVC benchmark is annotated with multiple human-written captions, covering different aspects of the video. This comprehensive annotation ensures a robust and thorough evaluation of Video-LLMs.
- 12We also include evaluation results on several other popular video benchmarks (Zhou et al., 2024; Li et al., 2024c; Fang et al., 2024; Wang et al., 2024a) in Appendix C.

Proprietary Models The proprietary models selected for comparison are state-of-the-art multi-modal systems developed by leading companies. These models include Gemini 1.0 Series (Google, 2023), Gemini 1.5 Series (Google, 2024), GPT4-V (OpenAI, 2023b), GPT4O (OpenAI, 2024), Reka Series (Reka, 2024), and Pegasus-1 (Jung et al., 2024). These models represent the cutting-edge proprietary multi-modal understanding technologies and serve

- as benchmarks to gauge the performance of VideoLLaMA 2.

Open-Source Models We also include several prominent open-source models to provide a broader context for our evaluations. Specifically, we compare our VideoLLaMA 2 with VistaLLaMA (Ma et al., 2023), ChatUniVi (Jin et al., 2023b), LLaMA-VID (Li et al., 2023d), Video-LLaVA (Lin et al., 2023a), VideoChat2 (Li et al., 2023c), LLaVA-NeXT-Video Series13 (Liu et al., 2024b), VILA 1.5 (Lin et al., 2024), PLLaVA (Xu et al., 2024a), and LLaVA-OneVision (Li et al., 2024a). These open-source models are crucial for evaluating the performance of VideoLLaMA 2 within the context of accessible and reproducible research.

### 5.1.3 Evaluation Protocol

All experiments, including the reproduction of baseline models, are conducted in a zero-shot manner to objectively assess generalization capabilities. For model decoding strategies, we use greedy search for all benchmarks except MSVC, where we apply sampling with a temperature of 0.2 to enhance the diversity and detailedness of the generated captions. We follow the original setup of baseline models regarding the number of frames used for each input video, ensuring fair and consistent comparisons across all models.

### 5.1.4 Main Results

Results on MC-VQA and VC The overall performance on multiple-choice video question answering (MC-VQA) and video captioning (VC) tasks are summarized in Table 5. VideoLLaMA 2 demonstrates strong performance compared to open-source models and shows competitive results against proprietary models in certain benchmarks.

For MC-VQA, VideoLLaMA 2 exhibits substantial improvements over open-source models. On the EgoSchema benchmark, VideoLLaMA 2-7B achieves an accuracy of 51.7%, outperforming the previous SOTA LLaVA-NeXT-Video (43.9%) by a large margin. Similarly, on the Perception-Test and MV-Bench datasets, VideoLLaMA 2-7B attains accuracies of 51.4% and 53.9%, respectively, surpassing other open-source models. Notably, VideoLLaMA 2 also outperforms the proprietary model GPT4-V (43.7%) on the MV-Bench dataset. Additionally, VideoLLaMA 2 shows competitive performance on the VideoMME benchmark with an accuracy of 48.4%, highlighting its robust capabilities in video understanding tasks. Furthermore, scaling up the LLM backbone from Mistral (7B) to Mixtral (8×7B) further enhances model performance in MC-VQA. This upscaling results in notable improvements across multiple benchmarks, with VideoLLaMA 2-8×7B achieving the highest accuracies on Egoschema, Perception-Test, and Video-MME.

In the VC task, VideoLLaMA 2 performs well on the MSVC benchmark, scoring 2.57 in correctness and 2.61 in detailedness. While these scores are slightly lower than GPT4-V’s 2.70 and 2.76, they are higher than all other open-source models, demonstrating the model’s strong capabilities in interpreting dynamic video content.

Results on OE-VQA The performance on Open-Ended Video Question Answering (OEVQA) tasks is summarized in Table 6. VideoLLaMA 2 demonstrates strong performance compared to both proprietary and open-source models across several benchmarks. For the MSVD dataset, VideoLLaMA 2-7B gets an accuracy of 71.7% with a score of 3.9, outperforming other open-source models, e.g., LLAVA-NeXT-Video (67.8%/3.5) and VideoChat2 (70.0%/3.9).

However, on the ActivityNet dataset, VideoLLaMA 2-7B attains an accuracy of 49.9% with a score of 3.3, which is slightly lower than LLAVA-NeXT-Video (53.5%/3.2). Similarly,

13For a fair comparison, we do not include the preference-optimized LLaVA-NeXT-Video as baseline. However, we still provide the full comparison results between VideoLLaMA 2 and LLaVA-NeXT-VideoDPO in Appendix B for the reference of readers.

|Model # Frames<br><br>|MSVD ActivityNet|Video-ChatGPT (Score)|
|---|---|---|
| |(Acc./Score) (Acc./Score)|Correctness Detail Context Temporal Consistency|

Proprietary Models Gemini 1.0 Pro - - 49.8/-♥ - - - - Gemini 1.0 Ultra - - 52.2/-♥ - - - - Gemini 1.5 Pro - - 56.7/-♥ - - - - GPT4-V - - 59.5/-♥ 4.09 3.88 4.37 3.94 4.02 GPT4-O - - 61.9/-♥ - - - - Pegasus-1 - - 59.9/-♥ 3.79♥ 3.76♥ 4.29♥ 3.34♥ 4.03♥

Open-Source Models VideoLLaMA (7B) 8 51.6/2.5 12.4/1.1 1.96 2.18 2.16 1.82 1.79 Video-ChatGPT (7B) 8 64.9/3.3 35.2/2.7 2.50 2.57 2.69 2.16 2.20 VideoChat (7B) 8 56.3/2.8 26.5/2.2 2.23 2.50 2.53 1.94 2.24 Chat-UniVi (7B) 8 65.0/3.6♥ 46.1/3.3♥ 2.89 2.91 3.46 2.89 2.81 LLaMA-VID (7B) 1 fps 69.7/3.7♥ 47.4/3.3♥ 2.96 3.00 3.53 2.46 2.51 Video-LLaVA (7B) 8 70.7/3.9♥ 45.3/3.3♥ 2.87 2.94 3.44 2.45 2.51 VideoChat2 (7B) 16 70.0/3.9♥ 49.1/3.3♥ 3.02 2.88 3.51 2.66 2.81 LLaVA-NeXT-Video (7B) 32 67.8/3.5♠ 53.5/3.2♥ 3.39♥ 3.29♥ 3.92♥ 2.60♥ 3.12♥ LLaVA-NeXT-Video (32B) 32 - 54.3/-♥ - - - - PLLaVA (34B) 16 - 60.9/-♥ 3.60♥ 3.20♥ 3.90♥ 2.67♥ 3.25♥ LLaVA-OneVision (72B) 32 - 62.3/-♥ - - - - VideoLLaMA2 (7B) 8 71.7/3.9 49.9/3.3 3.09 3.09 3.68 2.63 3.25 VideoLLaMA2 (7B) 16 70.9/3.8 50.2/3.3 3.16 3.08 3.69 2.56 3.14 VideoLLaMA2 (8x7B) 8 70.5/3.8 50.3/3.4 3.08 3.11 3.64 2.67 3.26 VideoLLaMA2 (72B) 8 71.0/3.8 55.2/3.4 3.23 3.11 3.71 2.62 3.12 VideoLLaMA2.1 (7B) 16 70.6/3.8 53.0/3.4 3.30 3.18 3.78 2.66 3.20

- Table 6: Main Results on Open-Ended Video QA (OE-VQA) benchmarks. ♥: officially reported results. ♠: results reproduced by ourselves. The numbers without marks are retrieved from Maaz et al. (2023b); Luo et al. (2023); Liu et al. (2024b).

on the Video-ChatGPT benchmark, VideoLLaMA 2-7B scores 3.09 in correctness, 3.09 in detail, 3.68 in context, 2.63 in temporal understanding, and 3.25 in consistency. While VideoLLaMA 2 achieves high scores, it is outperformed by LLAVA-NeXT-Video in several metrics, particularly in correctness (3.39), detail (3.29), and context (3.92). These two benchmarks, both based on the ActivityNet (Yu et al., 2019) dataset, typically include videos depicting a single human activity. Given that LLAVA-NeXT-Video (Liu et al., 2024b) is trained primarily on static image data with a comparably small amount of dynamic video data, it shows advantages in benchmarks where static visual information is crucial for answering questions. This suggests that training on massive static image data can be beneficial for video tasks that are heavily reliant on static visual information, a hypothesis we will explore further in future research.

### 5.2 Audio Understanding

To evaluate the audio understanding capabilities of VideoLLaMA 2, we conduct comprehensive evaluations on several established audio understanding benchmarks. These evaluations aim to measure the model’s proficiency in interpreting and integrating audio information in conjunction with video data. We use ChatGPT 3.5 to assess the prediction of models and the prompt is provided in AppendixA. The ChatGPT is required to provide the “yes” or “no” binary response, followed by a integer score to quantify the degree of match.

### 5.2.1 Evaluation Benchmarks

We conduct extensive experiments on Audio-only Question Answering (AQA) task, followed by Open-Ended Audio-Video Question Answering tasks to assess the audio comprehension abilities of VideoLLaMA 2.

|Method # Hours<br><br>|Clotho-AQA TUT2017 VocalSound|
|---|---|
|Qwen-Audio (7B) 137k Qwen2-Audio (7B) 520k|57.90 64.90 92.89 - - 93.92<br><br>|

VideoLLaMA2-AV (7B) 5k 70.11 78.40 93.19 VideoLLaMA2.1-AV (7B) 5k 71.02 77.28 92.40

- Table 7: Comparison with existing LLM-based methods on open-ended AQA (Clotho-AQA) and multiple-choice AQA (TUT2017 and VocalSound) benchmarks.

AQA For the AQA task, we select open-ended Clotho-AQA (Lipping et al., 2022a) and multiple-choice TUT2017 (Mesaros et al., 2016a) and VocalSound (Gong et al., 2022) datasets as our benchmark and report accuracy as the evaluation metric.

OE-AVQA For open-ended audio-video question answering, we adopt VGGSound (Chen

- et al., 2020) (15341 samples), AVSD (Alamri et al., 2019) (18630 samples) and MusicAVQA (Li et al., 2022a) (9129 samples) as benchmarks. For VGGSound and AVSD, we employ a GPT-assisted evaluation as the same as the protocols in OE-VQA.

### 5.2.2 Baselines

To evaluate the performance of VideoLLaMA 2 within the realms of audio understanding and audio-visual integration, we compare it against an array of established and cutting-edge models in these fields.

Audio Understanding To assess the audio understanding capabilities of VideoLLaMA 2-7B, we utilize Qwen-Audio (Chu et al., 2023) and Qwen2-Audio (Chu et al., 2024) as the benchmarks. Qwen-Audio and Qwen2-Audio are renowned for its robust performance across various audio understanding tasks, providing a strong comparison point for our model.

Audio-Visual Understanding Expanding the evaluation to audio-visual integration, VideoLLaMA 2 is benchmarked against advanced models that specialize in handling multimodal data. This includes comparisons with PandaGPT (Su et al., 2023), MacawLLM (Lyu et al.,

- 2023), Video-LLaMA (Zhang et al., 2023), and X-InstructBLIP (Panagopoulou et al., 2023), which are adept at understanding complex multimodal scenes. Further, we examine newer integrative models like AV-LLM (Shu et al., 2023), OneLLM (Han et al., 2024), CREMA (Yu et al., 2024), and AVicuna (Tang et al., 2024), which utilize high-quality audio-visual training datasets to enhance their multimodal understanding capabilities.

### 5.2.3 Main Results

Results on AQA The experimental findings presented in Table 7 delineate the compelling advantages of VideoLLaMA 2-AV (7B) and VideoLLaMA 2-AV (7B) on audio-only question answering (AQA).

Using the widely recognized Qwen-Audio and Qwen2-Audio as benchmarks, both VideoLLaMA 2-A (7B) and VideoLLaMA 2-AV (7B) demonstrate exceptional performance across multiple datasets, showcasing remarkable efficiency and advanced learning capabilities despite being trained on significantly fewer hours of data. Specifically, VideoLLaMA 2-A (7B), with only 4k hours of audio data, outperforms Qwen-Audio (137k hours) across most benchmarks. It achieves 68.90% on Clotho-AQA, a substantial improvement over QwenAudio’s 57.90%, and records 75.19% on TUT2017, surpassing Qwen-Audio’s 64.90% by 10.29%. This highlights its robust understanding and interpretative capabilities in complex audio question-answering tasks. Meanwhile, in the VocalSound benchmark, where high accuracy is essential, VideoLLaMA 2-A (7B) achieves 92.73%, closely matching Qwen2-Audio (7B)’s 93.92% despite being trained on far fewer hours (4k vs. 520k).

|Method # Pairs<br><br>|MUSIC-QA AVSD VGGSound|
|---|---|
|PandaGPT (13B) 128M Macaw-LLM (7B) 0.3M VideoLLaMA (7B) 2.8M X-InstructBLIP (13B) 32M AV-LLM (13B) 1.6M OneLLM (7B) 1007M AVicuna (7B) 1.1M CREMA (4B) -|33.7 26.1 32.7 31.8 34.3 36.1 36.6 36.7 40.8<br><br>44.5 - -<br><br>45.2 52.6 47.6<br><br><br>47.6 - 49.6 53.1 -<br><br>52.6(75.6) - -|

VideoLLaMA2-AV (7B) 1.9M 79.2 57.2 70.9 VideoLLaMA2.1-AV (7B) 2.0M 80.9 57.2 71.4

- Table 8: Comparison with existing LLM-based methods on Open-Ended Audio-Video QA (OE-AVQA) benchmarks. # Pairs: the adopted instruction-response pairs. Note: All baseline models for MUSIC-QA and AVSD are zero-shot. For the VGGSound dataset, the first three models are zero-shot, while the remaining are in-domain. The number in the bracket represents the enhanced result obtained through specialized model fine-tuning.

Furthermore, VideoLLaMA 2-AV (7B), leveraging about 5k hours of multimodal (audiovisual) data, builds on the strengths of VideoLLaMA 2-A and achieves even higher performance. It records the highest scores on Clotho-AQA (70.11%) and TUT2017 (78.40%). On VocalSound, VideoLLaMA 2-AV (7B) reaches 93.19%, further narrowing the gap with Qwen2-Audio (93.92%). These results emphasize the effectiveness of the VideoLLaMA framework, showcasing both models’ ability to achieve superior or comparable results with much fewer training hours. This highlights their efficiency and demonstrates the value of integrating audio-visual modalities for enhanced learning.

Results on OE-AVQA The performance results summarized in Table 8 highlight VideoLLaMA 2-AV (7B)’s significant advancements compared to open-source models. This analysis underscores VideoLLaMA 2-AV (7B)’s capabilities across various AVQA benchmarks, emphasizing its superior performance.

In the MUSIC-QA benchmark, VideoLLaMA 2-AV (7B) demonstrates substantial competency with a score of 79.2%. This score indicates a robust ability to analyze complex musical and audio cues, which is central to the MUSIC-QA benchmark. The model’s performance, outreaching CREAM with specialized model fine-tuning at 75.6%, underscores its advanced capabilities in handling intricate audio-visual interactions. In the AVSD benchmark, which focuses on audio-visual scene description involving dialogue understanding, VideoLLaMA 2-AV (7B) also demonstrates strong capabilities by scoring 57.2%, illustrating its adeptness

- at integrating visual context with audio inputs to generate coherent scene descriptions. The VGGSound benchmark, which necessitates a deep understanding of both visual scenes and their auditory elements, sees VideoLLaMA 2-AV (7B) leading with a score of 70.9%. This performance, much higher than AV-LLM’s 47.6%, showcases VideoLLaMA 2-AV (7B)’s exceptional ability to interpret and synthesize information across both modalities effectively. It indicates a profound understanding of dynamic interactions and the contextual synthesis necessary in AVSSD (audio-visual sound source detection) tasks.

## 6 Cases

As illustrated in Figure 3, we show some cases to demonstrate VideoLLaMA 2’s multi-modal instruction-following capability in video-centric conversations:

❶ Global Scene Understanding: As shown in Figure. 3a, It can be observed that VideoLLaMA 2 clearly describes the details of objects in the scene, such as the game console, the

atmosphere of the scene, and the boy’s dancing movements, which demonstrates the strong scene understanding ability of VideoLLaMA2.

❷ Spatial-temporal Orientation Awareness: In Figure. 3b, VideoLLaMA 2, by observing the entire video, correctly judged the car’s turning direction, proving that VideoLLaMA2 is able to perceive spatial-temporal orientation information of a video.

❸ Commonsense Reasoning: In Figure. 3c, VideoLLaMA 2 first understands the events occurring in the video (i.e., reading and drinking), then observes the environmental details in the video (e.g., natural light), and finally infers that it is likely morning based on common sense. This demonstrates its strong capability for common sense reasoning.

❹ Spatial-Temporal Fine-grained Recognition: Figure. 3d exhibits VideoLLaMA 2’s finegrained understanding ability of objects that appear at specific moments and specific locations in the video. Without being disturbed by the early information of the video, VideoLLaMA 2 accurately pinpoints the moment and location where the red sticky note appeared and recognizes the text on it.

## 7 Related Works

Existing Video-LLMs typically consist of a pre-trained visual encoder (e.g., CLIP (Radford

- et al., 2021) or DINO (Caron et al., 2021)) to encode video frames into low-dimensional visual features, a vision-language adapter to aggregate these features and transform them into representations understandable by large language models (LLMs), and an instruction-tuned language decoder (e.g., LLaMA-2-Chat (Touvron et al., 2023) or Mistral-Instruct (Jiang et al., 2023)) to generate text responses based on instructions and user-uploaded videos. In this project, we primarily explore the better designs of vision-language connectors for Video-LLMs and leave the visual encoder and language decoder untouched.

The vision-language adapters of Image-LLMs, such as Cross-Attention (Alayrac et al., 2022; Bai et al., 2023), Q-Former (Li et al., 2023a; Zhu et al., 2023b; Ye et al., 2024b), Linear Projection (Liu et al., 2023b; Chen et al., 2023a; Wang et al., 2023a) and Dynamic Visual Tokenizer (Jin et al., 2023c), have been widely adopted in popular Video-LLMs (Zhang et al.,

- 2023; Maaz et al., 2023b; Lin et al., 2023a; Li et al., 2023c; Jin et al., 2024; Ataallah et al., 2024; Liu et al., 2024e). Despite their reasonable results, these designs are still insufficient for Video-LLMs because they completely ignore temporal aggregation and leave everything of temporal modeling to language decoder, which is neither computationally efficient (i.e., LLMs need to process much more tokens) nor effective on video understanding tasks, as observed in our preliminary experiments (See Table 1).

Considering efficiency and effectiveness, we introduce the Spatial-Temporal Convolution Connector (STC Connector) into VideoLLaMA 2 to better capture the spatial-temporal features while maintaining a reasonable number of visual tokens. Specifically, our STConnector employs 3D convolution for spatial-temporal aggregation. To preserve local visual patterns during spatial compression, we follow Cha et al. (2024) to incorporate a RegStage (Radosavovic et al., 2020) block before and after the 3D convolution, which has been shown to enhance spatial understanding.

Furthermore, expanding on the core capabilities of Video-LLMs, recent advancements have focused on integrating audio streams, a rich source of contextual cues that are vital for a complete video scene understanding. Models such as PandaGPT (Su et al., 2023) and VideoLLaMA (Zhang et al., 2023) utilize pre-trained systems like ImageBind (Girdhar et al., 2023) for a universal modality alignment, while MacawLLM (Lyu et al., 2023) optimizes separate branches for visual and audio data using diverse instructional sets. X-InstructBLIP (XBLIP) (Panagopoulou et al., 2023) integrates multiple modalities within a single framework using frozen Large Language Models (LLMs), with modality-specific Q-Formers serving as adapters to bridge various encoders. OneLLM (Han et al., 2024) introduces a universal encoder and projection module designed to align multiple modalities with linguistic data, thereby improving the coherence of multimodal integration. CREMA (Yu et al., 2024) takes an efficient and modular approach, utilizing parameter-efficient adapters for each modality to enhance flexibility and ease of incorporating new modalities into existing frameworks. Other

models like AV-LLM (Shu et al., 2023) and AVicuna (Tang et al., 2024) leverage integrated audio-visual training data to further refine their understanding of complex multimodal content. These developments represent significant strides toward creating more versatile and capable LLMs that can navigate and interpret intricate temporal dynamics with synchronous audio streams effectively.

[Figure 175]

Question: What's going on in this video? What about the scene makes you feel that way?

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

###### VideoLLaMA 2: In the video, a boy is dancing in front of a group of arcade machines. He seems to be

enjoying himself as he moves around and dances. The scene is set in an arcade, which is a place where people go to play video games and other electronic games. The boy's movements and the setting give off a joyful and playful vibe, which makes me feel that the video captures a moment of carefree fun.

##### (a) Global Scene Description Case.

[Figure 193]

Question: Which way does the yellow cab turn?

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

VideoLLaMA 2: The yellow cab turns left at the intersection.

##### (b) Spatial-temporal Orientation Awareness Case.

[Figure 211]

Question: Do you think it's morning or night in this video? Why?

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

VideoLLaMA 2: It's hard to say for certain, but based on the available information, it's possible that it's morning. The person in the video is having breakfast and reading a book, which is a common morning routine for many people. Additionally, the lighting in the video appears to be natural light, which could also suggest that it's daytime. However, without more context or information, it's difficult to say for certain.

##### (c) Commonsense Reasoning Case.

[Figure 229]

Question: Please tell me what the text on the red sticker is.

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

VideoLLaMA 2: The text on the red sticker is “don't waste your time”.

(d) Spatial-temporal Fine-grained Recognition Case.

Figure 3: Qualitative Video Understanding Cases from different perspectives of VideoLLaMA 2.

## 8 Conclusion

We present the VideoLLaMA 2 series, a set of generalist Video Large Language Models designed to advance multimodal research in the arena of video-language modeling. By incorporating a meticulously designed Spatial-Temporal Convolution (STC) connector and a jointly trained Audio Branch, VideoLLaMA 2 consistently improves multimodal comprehension across various video and audio-oriented tasks. It outperforms the open-source models of similar size across multiple benchmarks and, in several aspects, achieves performance levels comparable to proprietary models, demonstrating impressive capabilities in modeling temporal dynamics with synchronous audio streams. Furthermore, as a foundational model, VideoLLaMA 2 can be further developed to benefit various more specialized but challenging problems, like long video understanding (Ren et al., 2023; Song et al., 2023; Wang et al.,

- 2024d), video agent (Lin et al., 2023b; Fan et al., 2024; Wang et al., 2024b; He et al., 2024), autonomous driving (Xu et al., 2023b; Shao et al., 2024), motion understanding (Wu et al.,

- 2024; Chen et al., 2024b), and robotic manipulation (Liu et al., 2024d).

## References

Andrea Agostinelli, Timo I Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023.

Huda Alamri, Vincent Cartillier, Abhishek Das, Jue Wang, Anoop Cherian, Irfan Essa, Dhruv Batra, Tim K Marks, Chiori Hori, Peter Anderson, et al. Audio visual scene-aware dialog. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7558–7567, 2019.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.

Anthropic. The claude 3 model family: Opus, sonnet, haiku. 2024.

Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. arXiv preprint arXiv:2404.03413, 2024.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.

Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Localityenhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3558–3568, 2021.

David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190–200, 2011.

Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model, 2024a.

Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 721–725. IEEE, 2020.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023a.

Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023b.

Ling-Hao Chen, Shunlin Lu, Ailing Zeng, Hao Zhang, Benyou Wang, Ruimao Zhang, and Lei Zhang. Motionllm: Understanding human behaviors from human motions and videos. arXiv preprint arXiv:2405.20340, 2024b.

Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Wanxiang Che, Xiangzhan Yu, and Furu Wei. BEATs: Audio pre-training with acoustic tokenizers. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 5178–5193. PMLR, 23–29 Jul 2023c. URL https://proceedings.mlr.press/v202/chen23ag.html.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. arXiv preprint arXiv:2402.19479, 2024c.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023d.

Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.

Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 736–740. IEEE, 2020.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

Chenyou Fan. Egovqa-an egocentric video question answering benchmark dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pp. 0–0, 2019.

Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. arXiv preprint arXiv:2403.11481, 2024.

Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024.

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

Thomas Gemma, Teamand Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15180–15190, 2023.

Yuan Gong, Jin Yu, and James Glass. Vocalsound: A dataset for improving human vocal sounds recognition. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 151–155. IEEE, 2022.

Gemini Team Google. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Gemini Team Google. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz MuellerFreitag, et al. The" something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pp. 5842–5850, 2017.

Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26584–26595, 2024.

Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. arXiv preprint arXiv:2404.05726, 2024.

Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. arXiv preprint arXiv:2311.18445, 2(3):9, 2023.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Bu Jin, Xinyu Liu, Yupeng Zheng, Pengfei Li, Hao Zhao, Tong Zhang, Yuhang Zheng, Guyue Zhou, and Jingjing Liu. Adapt: Action-aware driving caption transformer. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 7554–7561. IEEE, 2023a.

Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046, 2023b.

Yang Jin, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, et al. Unified language-vision pretraining with dynamic discrete visual tokenization. arXiv preprint arXiv:2309.04669, 2023c.

Yang Jin, Zhicheng Sun, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, et al. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. arXiv preprint arXiv:2402.03161, 2024.

Raehyuk Jung, Hyojun Go, Jaehyuk Yi, Jiho Jang, Daniel Kim, Jay Suh, Aiden Lee, Cooper

- Han, Jae Lee, Jeff Kim, et al. Pegasus-v1 technical report. arXiv preprint arXiv:2404.14687, 2024.

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 119–132, 2019.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-andvision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36, 2024b.

Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19108–19118, June 2022a.

Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19108–19118, 2022b.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023a.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023b.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. arXiv preprint arXiv:2311.17005, 2023c.

Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043, 2023d.

Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. Tgif: A new dataset and benchmark on animated gif description. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4641–4650, 2016.

Yunxin Li, Xinyu Chen, Baotian Hu, Longyue Wang, Haoyuan Shi, and Min Zhang. Videovista: A versatile benchmark for video understanding and reasoning, 2024c.

Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023a.

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26689–26699, 2024.

Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023b.

Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023c.

Samuel Lipping, Parthasaarathy Sudarsanam, Konstantinos Drossos, and Tuomas Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering. In 2022 30th European Signal Processing Conference (EUSIPCO), pp. 1140–1144, 2022a. doi: 10.23919/EUSIPCO55093.2022.9909680.

Samuel Lipping, Parthasaarathy Sudarsanam, Konstantinos Drossos, and Tuomas Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering. In 2022 30th European Signal Processing Conference (EUSIPCO), pp. 1140–1144. IEEE, 2022b.

- Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023b.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b. URL https://llava-vl.github.io/blog/2024-01-30-llava-next/.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024c.

Jinyi Liu, Yifu Yuan, Jianye Hao, Fei Ni, Lingzhi Fu, Yibin Chen, and Yan Zheng. Enhancing robotic manipulation with ai feedback from multimodal large language models. arXiv preprint arXiv:2402.14245, 2024d.

Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. St-llm: Large language models are effective temporal learners. arXiv preprint arXiv:2404.00308, 2024e.

Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023.

Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu, Zefeng Du, Shuming Shi, and Zhaopeng Tu. Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration. arXiv preprint arXiv:2306.09093, 2023.

Fan Ma, Xiaojie Jin, Heng Wang, Yuchen Xian, Jiashi Feng, and Yi Yang. Vista-llama: Reliable video narrator via equal distance to visual tokens. arXiv preprint arXiv:2312.08870, 2023.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Videochatgpt: Towards detailed video understanding via large vision and language models.

- arXiv preprint arXiv:2306.05424, 2023a.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Videochatgpt: Towards detailed video understanding via large vision and language models.

- arXiv preprint arXiv:2306.05424, 2023b.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36, 2024.

Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D Plumbley, Yuexian Zou, and Wenwu Wang. Wavcaps: A chatgpt-assisted weaklylabelled audio captioning dataset for audio-language multimodal research. arXiv preprint arXiv:2303.17395, 2023.

Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. arXiv preprint arXiv:2402.14797, 2024.

Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen. Tut database for acoustic scene classification and sound event detection. In 2016 24th European Signal Processing Conference (EUSIPCO), pp. 1128–1132, 2016a. doi: 10.1109/EUSIPCO.2016.7760424.

Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen. TUT database for acoustic scene classification and sound event detection. In 24th European Signal Processing Conference 2016 (EUSIPCO 2016), Budapest, Hungary, 2016b.

Annamaria Mesaros, Toni Heittola, Aleksandr Diment, Benjamin Elizalde, Ankit Shah, Emmanuel Vincent, Bhiksha Raj, and Tuomas Virtanen. Dcase 2017 challenge setup: Tasks, datasets and baseline system. In DCASE 2017-Workshop on Detection and Classification of Acoustic Scenes and Events, 2017.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023a. OpenAI. Gpt-4v(ision) system card, 2023b. URL https://openai.com/research/

gpt-4v-system-card. OpenAI. Gpt-4o system card, 2024. URL https://openai.com/index/ hello-gpt-4o/.

Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, and Juan Carlos Niebles. X-instructblip: A framework for aligning x-modal instruction-aware representations to llms and emergent cross-modal reasoning. arXiv preprint arXiv:2311.18799, 2023.

Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36, 2024.

Karol J Piczak. Esc: Dataset for environmental sound classification. In Proceedings of the 23rd

ACM international conference on Multimedia, pp. 1015–1018, 2015. Team Qwen. Qwen2 technical report. 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763, 2021.

Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10428–10436, 2020.

Santhosh K Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alex Clegg, John Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, et al. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238, 2021.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pp. 8821–8831. Pmlr, 2021.

Reka team Reka. Reka core, flash, and edge: A series of powerful multimodal language models. arXiv preprint arXiv:2404.12387, 2024.

Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. arXiv preprint arXiv:2312.02051, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. A dataset and taxonomy for urban sound research. In Proceedings of the 22nd ACM International Conference on Multimedia, pp. 1041–1044. Association for Computing Machinery, 2014. ISBN 9781450330633.

Yihua Shao, Hongyi Cai, Wenxin Long, Weiyi Lang, Zhe Wang, Haoran Wu, Yan Wang, Yang Yang, and Zhen Lei. Accidentblip2: Accident detection with multi-view motionblip2. arXiv preprint arXiv:2404.12149, 2024.

Fangxun Shu, Lei Zhang, Hao Jiang, and Cihang Xie. Audio-visual llm for video understanding. arXiv preprint arXiv:2312.06720, 2023.

Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449, 2023.

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.

Yunlong Tang, Daiki Shimada, Jing Bi, and Chenliang Xu. Avicuna: Audio-visual llm with interleaver and context-boundary alignment for temporal referential dialogue. arXiv preprint arXiv:2403.16276, 2024.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. NEJM AI, 1(3):AIoa2300138, 2024.

Jack Urbanek, Florian Bordes, Pietro Astolfi, Mary Williamson, Vasu Sharma, and Adriana Romero-Soriano. A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions. arXiv preprint arXiv:2312.08578, 2023.

Jiawei Wang, Liping Yuan, and Yuchen Zhang. Tarsier: Recipes for training and evaluating large video description models, 2024a. URL https://arxiv.org/abs/2407.00634.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023a.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. arXiv preprint arXiv:2403.10517, 2024b.

Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4581–4591, 2019.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023b.

Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024c.

Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. arXiv preprint arXiv:2405.19209, 2024d.

Qi Wu, Yubo Zhao, Yifan Wang, Yu-Wing Tai, and Chi-Keung Tang. Motionllm: Multimodal motion-language learning with large language models. arXiv preprint arXiv:2405.17013, 2024.

X.AI. Grok-1.5 vision, 2024. URL https://x.ai/blog/grok-1.5v. Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-

answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9777–9786, 2021.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023a.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296, 2016.

Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024a.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. ArXiv, abs/2406.08464, 2024b. URL https://api. semanticscholar.org/CorpusID:270391432.

Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kenneth KY Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. arXiv preprint arXiv:2310.01412, 2023b.

Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. Clip-vip: Adapting pre-trained image-text model to video-language alignment. In The Eleventh International Conference on Learning Representations, 2023.

Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1686–1697, 2021.

Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. Avqa: A dataset for audio-visual question answering on videos. In Proceedings of the 30th ACM international conference on multimedia, pp. 3480–3491, 2022.

Qilang Ye, Zitong Yu, Rui Shao, Xinyu Xie, Philip Torr, and Xiaochun Cao. Cat: Enhancing multimodal large language model to answer questions in dynamic audio-visual scenarios, 2024a.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yi Zhou, Junyan Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qiang Qi, Ji Chao Zhang, and Feiyan Huang. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13040–13051, 2024b.

Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024.

Shoubin Yu, Jaehong Yoon, and Mohit Bansal. Crema: Multimodal compositional video reasoning via efficient modular adaptation and fusion. arXiv preprint arXiv:2402.05889, 2024.

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 9127–9134, 2019.

Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024.

Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023a.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023b.

## A The Prompt for GPT-aided AVQA Task Evaluation

System: You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here’s how you can accomplish the task:----##INSTRUCTIONS:

- - Focus on the meaningful match between the predicted answer and the correct answer.
- - Consider synonyms or paraphrases as valid matches.
- - Evaluate the correctness of the prediction compared to the answer. User: Please evaluate the following video-based question-answer pair: Question: "{QUESTION}" Correct Answer: "{ANSWER}" Predicted Answer: "{PREDICTION}" Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. Please generate the response in the form of a Python dictionary string with keys ’pred’ and ’score’, where value of ’pred’ is a string of ’yes’ or ’no’ and value of ’score’ is in INTEGER, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this: {’pred’: ’yes’, ’score’: 4}.

Table 9: The prompt for GPT-aided AVQA task evaluation.

## B Comparision Results with LLaVA-NeXT-Video Series

MC-VQA VC EgoSchema Perception-Test MVBench VideoMME MSVC (Score)

Model # Frames

(Acc.) (Acc.) (Acc.) (Acc.) correctness detailedness LLaVA-NeXT-Video (7B) 32 43.9♠ 48.8♠ 46.5♠ 33.7♠ 2.40♠ 2.52♠ LLaVA-NeXT-Video-DPO (7B) 32 44.6♠ 49.3♠ 46.0♠ 35.6♠ 2.87♠ 2.86♠ VideoLLaMA 2 (7B) 8 50.5 49.6 53.4 44.0 2.57 2.61 VideoLLaMA 2 (7B) 16 51.7 51.4 54.6 46.6 2.53 2.59 VideoLLaMA 2 (8x7B) 8 53.3 52.2 53.9 48.4 2.53 2.56

Table 10: Comparison results on MV-VQA and VC tasks. ♠: reproduced results.

MSVD ActivityNet Video-ChatGPT (Score)

Model # Frames

(Acc./Score) (Acc./Score) Correctness Detail Context Temporal Consistency LLaVA-NeXT-Video (7B) 32 67.8/3.5♠ 53.5/3.2♥ 3.39♥ 3.29♥ 3.92♥ 2.60♥ 3.12♥ LLaVA-NeXT-Video-DPO (7B) 32 71.0/3.7♠ 60.2/3.5♥ 3.64♥ 3.45♥ 4.17♥ 2.95♥ 4.08♥ VideoLLaMA 2 (7B) 8 71.7/3.9 49.9/3.3 3.09 3.09 3.68 2.63 3.25 VideoLLaMA 2 (7B) 16 70.9/3.8 50.2/3.3 3.16 3.08 3.69 2.56 3.14 VideoLLaMA 2 (8x7B) 8 70.5/3.8 50.3/3.4 3.08 3.11 3.64 2.67 3.26

- Table 11: Comparison results on MSVD, ActivityNet, and Video-ChatGPT. ♥: officially reported results. ♠: reproduced results.

## C Results on More Video Benchmarks

|Model # Frames<br><br>|MLVU VideoVista MMBench-Video DREAM-1k|
|---|---|
| |(M-AVG) (Acc.) (Score) (F1)<br><br>|
|VideoLLaMA 2 (7B) 16 VideoLLaMA 2 (72B) 16<br><br>|32.7 60.47 1.08 26.2 45.6 - - 27.1|

- Table 12: Results of VideoLLaMA 2 models on MLVU (Zhou et al., 2024), VideoVista (Li et al., 2024c), MMBench-Video (Fang et al., 2024), and DREAM-1k (Wang et al., 2024a) benchmarks.

