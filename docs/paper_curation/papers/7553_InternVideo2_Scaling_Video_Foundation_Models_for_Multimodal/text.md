# arXiv:2403.15377v4[cs.CV]14Aug2024

### INTERNVIDEO2: SCALING FOUNDATION MODELS FOR MULTIMODAL VIDEO UNDERSTANDING

Yi Wang∗1, Kunchang Li∗3,1, Xinhao Li∗2,1, Jiashuo Yu∗1, Yinan He∗1, Chenting Wang∗2,1 Guo Chen2,1, Baoqi Pei1, Ziang Yan1, Rongkun Zheng1, Jilan Xu1, Zun Wang1 Yansong Shi1, Tianxiang Jiang1, Songze Li1, Hongjie Zhang1, Yifei Huang1 Yu Qiao†1, Yali Wang†3,1, Limin Wang†2,1 1OpenGVLab, Shanghai AI Laboratory, Shanghai, China 2Nanjing University, Nanjing, China 3Shenzhen Institutes of Advanced Technology, CAS, Shenzhen, China

https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2

Support for Long Videos

Transferable Video(-Text) Representation

| |
|---|

| |
|---|

Inaccessible in eval

Visual inputs

[Figure 1]

3 min

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

92.1

63.4

51.2

… … … … … …

88.8

55.2

56.5

47.8

Q: identify the most significant one related to food preparation A: C stirring the food in the pot.

55.8

41.2

Temporal Commonsense Reasoning

82.4

|[Figure 7]<br><br>[Figure 8]<br><br>|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

49.2

… … … …

34.6

43.5

71.5 55.9 63.2

Q: Infer the shape drawn by the robotic arm on the surface of the latte according to its movements? A: It seems like the robotic arm is drawing a heart on the latte.

Figure 1: InternVideo2 yields strong transferable visual and visual-linguistic representations across a total of 70 video understanding tasks, ranging from action recognition, video-text understanding, to video-centric dialogue. It also exhibits capability of long-form video understanding and procedure-aware reasoning.

##### ABSTRACT

We introduce InternVideo2, a new family of video foundation models (ViFM) that achieve the stateof-the-art results in video recognition, video-text tasks, and video-centric dialogue. Our core design is a progressive training approach that unifies the masked video modeling, crossmodal contrastive learning, and next token prediction, scaling up the video encoder size to 6B parameters. At the data level, we prioritize spatiotemporal consistency by semantically segmenting videos and generating video-audio-speech captions. This improves the alignment between video and text. Through extensive experiments, we validate our designs and demonstrate superior performance on over 60 video and audio tasks. Notably, our model outperforms others on various video-related dialogue and long video understanding benchmarks, highlighting its ability to reason and comprehend longer contexts.

##### 1 Introduction

Learning transferrable spatiotemporal representations is a critical research area in computer vision, holding diverse applications across domains such as video search [Gabeur et al., 2020], game control [Bruce et al., 2024], robotic learning [Driess et al., 2023], self-driving [Zablocki et al., 2022], and scientific studies [Team et al., 2023]. Recently, the advancement of Large Language Models (LLMs) [Brown et al., 2020, OpenAI, 2023a, Touvron et al., 2023a,b] and their multimodal variations (MLLMs) [OpenAI, 2023b, Gong et al., 2023, Liu et al., 2023, Team et al., 2023] have had a profound impact on vision research and other disciplines. Embedding videos effectively into these large models and

*Equal contribution. †Corresponding authors.

harnessing their capabilities to enhance video understanding performance has emerged as pivotal tasks [Li et al., 2023c, Maaz et al., 2023].

Previous research has identified several effective learning schemes for video representations, including reconstructing videos with masked inputs [He et al., 2022, Tong et al., 2022, Wang et al., 2023b, Feichtenhofer et al., 2022], aligning videos with languages [Li and Wang, 2020, Xu et al., 2021, Yan et al., 2022, Li et al., 2023e], and predicting the next token using videos [Alayrac et al., 2022, Sun et al., 2023c, Li et al., 2023d]. These approaches have turned out to be complementary and can be unified through a progressive training scheme. Notably, methods such as InternVideo [Wang

- et al., 2022], UMT [Li et al., 2023e], and VideoPrism [Zhao et al., 2024] have utilized a two-stage training approach involving masked reconstruction and multimodal contrastive learning, leading to improved performance in downstream tasks. Following this line, we aim to further extend this progressive training scheme by incorporating video-based next token prediction and scaling the entire training process, including models and data, to build a new family of video foundation models.

The proposed video foundation model, coined as InternVideo2, is built through a progressive training scheme. The learning involves three stages: (1) capturing spatiotemporal structure via unmasked reconstruction, (2) aligning with semantics from other modalities, and (3) enhancing its open-ended dialogue power through next token prediction. In the initial stage, the model learns to reconstruct the unmasked video tokens, allowing the video encoder to develop basic spatiotemporal perception capability. To estimate the existing tokens, vision encoders (InternViT [Chen et al., 2023c] and VideoMAE-g [Wang et al., 2023b]) trained differently are employed as proxies. In the next stage of crossmodal learning, the architecture is expanded to include audio and text encoders. This not only improves the alignment between videos and text but also endows InternVideo2 the ability to handle video-audio tasks. By incorporating these additional modalities, the model’s understanding of videos is enriched and aligned with their semantics. Finally, in the next-token prediction stage, a video-centric dialogue system and the corresponding instruction-finetuning dataset are built to further tune the InternVideo2. By connecting InternVideo2 to LLMs, the video encoder is further updated through next-token prediction training, enhancing its ability for open-ended tasks such as VQA and video description.

For the training of InternVideo2, we emphasize the spatiotemporal consistency and labeling quality in the data. We build a large-scale multimodal video-centric dataset consisting of 402M data entries, which includes 2M videos, 50M videotext pairs (from WebVid [Bain et al., 2021] and InternVid [Wang et al., 2023d]), 50M video-audio-speech-text pairs (InternVid2), and 300M image-text pairs. Specifically, for InternVid2, we segment videos into clips semantically and focus on recalibrating the clip descriptions using three modalities: audio, video, and speech. We first generate captions for these three modalities separately. Then individual captions are fused together to create a more comprehensive description, which will improve the model’s ability to comprehend and interpret the video accurately.

We evaluate InternVideo2 across a wide range of video-related tasks. These tasks span from basic spatiotemporal perception, such as action recognition, to high-level reasoning tasks, such as long video or procedure-aware questionanswering (QA), as given in Fig. 1. The results (in Sec. 5) demonstrate that InternVideo2 achieves the state-of-the-art performance on multiple tasks and is able to analyze and reason over sequences of actions. This top performance signifies its capability to effectively capture and understand video content. These empirical findings validate that InternVideo2 could serves as a general video encoder for future exploration in video understanding. In summary, our contributions to video understanding are as follows.

- • This paper introduces InternVideo2, a competitive family of video foundation models that leverages masked reconstruction, crossmodal contrastive learning, and next token prediction to make model perceptive, semantic, and capable of reasoning in video understanding.
- • InternVideo2 achieves the state-of-the-art performance for more than 60 video / audio tasks. Our model demonstrates superior performance in video-related dialogue and long video understanding, highlighting its potential in modeling high-level world knowledge.
- • We provide an enhanced dataset to train InternVideo2. This includes the validation and incorporation of audio data during training, as well as the improved captioning method. These improvements result in significant enhancements in model performance and generalization ability.

##### 2 Related Work

Video Foundation Models. Studies on learning video foundation models become increasingly crucial considering its wide applications [Li and Wang, 2020, Xu et al., 2021, Li et al., 2023e, Wang et al., 2022, Zhao et al., 2024, Wang et al., 2023c, Yan et al., 2022, Feichtenhofer et al., 2022, Wang et al., 2023a, Tong et al., 2022, Wang et al., 2023b]. Typical methods in building video foundation models (ViFM) include video-text contrastive learning [Li and Wang, 2020, Xu et al., 2021, Wang et al., 2022], masked video modeling [Tong et al., 2022, Wang et al., 2023b, 2022, Fu

###### Supervise Forward Parametrize

Video Encoder Video Encoder

Video Encoder

[Figure 13]

Align

Audio Encoder

Expert Encoder

LLM Qformer

| |
|---|

[Figure 14]

Text Encoder

Expert Encoder

|Answers|
|---|

|Instructions|
|---|

Stage 1 Stage 2 Stage 3

Figure 2: Framework of InternVideo2. It consists of three consecutive training phases: unmasked video token reconstruction, multimodal contrastive learning, and next token prediction. In stage 1, the video encoder is trained from scratch, while in stages 2 and 3, it is initialized from the version used in the previous stage.

et al., 2021], and next token prediction [Alayrac et al., 2022, Sun et al., 2023c,a]. Specifically, All-in-one [Wang et al., 2023a] utilized a single backbone with unified multiple pretraining objectives. On the other hand, UMT [Li et al., 2023e] combined masked modeling with video-text contrastive learning, demonstrating strong performance in both action recognition and video-language tasks. Another approach is mPLUG-2 [Xu et al., 2023], which introduced a new design for modulating different modalities. It shared a common module across modalities to enhance relations while incorporating modality-specific modules for discrimination. In addition to video-text pretraining, researchers have also explored the use of audio information in videos to improve performance. MERLOT Reserve [Zellers et al.,

- 2022] learned video representations using a large-scale dataset of video-speech-transcript pairs. VALOR [Chen et al.,
- 2023b] employed independent encoders for video, audio, and text and trains a joint visual-audio-text representation. VAST [Chen et al., 2024b] constructed an audio-visual-speech dataset and develops a multimodal backbone that excels in video-audio-related tasks. VideoPrism [Zhao et al., 2024] combined video-text contrastive learning and video token reconstruction on a combination of public and proprietary videos, achieving leading results across various video tasks.

Multimodal Large Language Models. With advances in large language models (LLMs) [Devlin et al., 2018, Raffel et al., 2020, Brown et al., 2020], their multimodal versions (MLLMs) is becoming popular as it can handle open-world tasks. Seminal works like Flamingo [Alayrac et al., 2022] showed outstanding zero/few-shots performances over a range of multimodal tasks [Goyal et al., 2017b, Plummer et al., 2015, Xu et al., 2016, Marino et al., 2019]. Public MLLMs [Zhu et al., 2023b, Liu et al., 2023, Gong et al., 2023] such as LLaVA [Liu et al., 2023] and InstructBLIP [Dai

- et al., 2023] proposed to use visual instruction-tuning data to improve the visual dialogue ability. Some video-centric MLLMs have been proposed, such as VideoChat [Li et al., 2023c], VideoChatGPT [Maaz et al., 2023] and Valley [Luo et al., 2023], by using instruction data to connect video encoders to LLMs for open-world video understanding.

##### 3 Method

We learn InternVideo2 in three stages, illustrated in Fig. 2. The stages include spatiotemporal token reconstruction, video-audio-speech-language contrastive learning, and connecting to a large language model (LLM) for joint training.

Video Encoder. The video encoder used in InternVideo2 follows the Vision Transformer (ViT) [Dosovitskiy et al., 2020] and includes additional projection layers for distillation. Inspired by previous works [Chen et al., 2023c, Yu et al., 2022], we introduce attention pooling to the ViT. For input videos, we sparsely sample 8 frames [Wang et al., 2016] and perform a 14×14 (h × w) spatial downsampling. These spatiotemporal tokens are then concatenated with a class token and combined with 3D position embeddings. The details of ViT-6B architecture are given in Supp.

###### 3.1 Stage1: Reconstructing Unmasked Video Tokens

We exploit two expert models to guide the video encoder to conduct token-level reconstruction for unmasked areas. Specifically, we adopt InternVL-6B [Chen et al., 2023c] and VideoMAEv2-g [Wang et al., 2023b] to transfer unmasked knowledge via simple projection layers. When training, we input the full videos into different teachers and mask out 80% of the tokens frame by frame, under the semantic guidance of the multimodal model InternVL and motion-aware model VideoMAEv2. We only align the unmasked tokens, by minimizing their mean squared error (MSE) between student and teachers. The learning objective is to reconstruct the remaining tokens as:

1 Z

L =

(α1|fV (Vp) − h(Vp)|2 + α2|fV (Vp) − g(Vp)|2), (1)

where fV , h, and g are our video encoder, InternViT-6B [Chen et al., 2023c], and ViT-g of VideoMAEv2, respectively. p stands for the token index and f(Vp) is the corresponding token extracted by InternVideo2 for input video V. Z is the normalization factor. α1 and α2 balance the influence between the employed models.

In our implementation, we randomly initialize the video encoder and then align its outputs from different layers (transformed by learnable multilayer perceptrons) to those of expert models. Specifically, we align: 1) the last 6 layers of InternVL, 2) the last 4 layers of VideoMAEv2, and 3) the final output token of InternVL. These alignments are made to the corresponding outputs of the video encoder using the l2 norm. The different loss terms are simply summed for optimization. After pretraining, we drop those projection layers and only use the basic encoder. Compared with only using the multimodal model in UMT and VideoPrism, our strategy makes the vision encoder multimodal-friendly as well as enhances its temporal sensitivity for action modeling.

###### 3.2 Stage 2: Aligning Video to Audio-Speech-Text

We exploit the correspondence between video and audio, speech, and text to encourage InternVideo2 to learn more semantics. In practice, InternVideo2 has a huge video encoder, and its employed audio and text encoders are relatively lightweight. The used audio encoder is a 12-layer transformer initialized with BEATs [Chen et al., 2023a] (90M). It takes as input 64-dimensional log Mel filterbank spectrograms, generated using a 25ms Hamming window, from 10second-long clips (padded with zeros). For the text and speech encoders, we initialize the text encoder and multimodal decoder using BERT-Large [Devlin et al., 2018]. Specifically, we utilize the initial 19 layers of BERT-Large as the text encoder, with the subsequent 5 layers equipped with cross-attention layers serving as the multimodal decoder.

For pretraining objectives, we establish alignment across different modalities via text, including video, audio, image, and speech. We employ crossmodal contrastive and matching losses with masked language reconstruction loss as:

L = LCON + LMAC + LMLM, (2)

The employed LMAC and LMLM are standard loss from [Cheng et al., 2022]. Specifically, the crossmodal contrastive learning is given as:

T

M i ,fiTM′ )/τ)

M′

i ,fiM)/τ)

+ Ni=1 log exp(sim(f

N i=1 log exp(sim(f

,

LCON = M,T

LCON(M,TM′) = − M,TM′

N j=1 exp(sim(fiM,fjTM′ )/τ)

N j=1 exp(sim(fiTM′ ,fjM)/τ)

M′

(3)

where fV and fT denote the learned video and text embeddings, respectively. M and TM′ indicates the modality of input signals and the text descriptions describing, respectively. sim(·) computes the cosine similarity between two features. τ is the learnable temperature.

For the matching part, it is given as:

LMAC = −y log fp(V,T) − (1 − y)log(1 − fp(V,T)), (4)

where fp(V,T) computes the matching likelihood between V and T. y denotes whether the given video and text are paired (y = 1) or not (y = 0).

The employed masked language modeling loss is:

LMLM = −log fpT(Tj|T<j), (5)

where fpT(Tj|T<j) computes the likelihood of the jth text token based on the previous ones. Here T refers to video captions.

To improve the training efficiency, we employ the masked learning strategy, aligning unmasked video tokens to tokens from other modalities first, then using full video tokens reconstruction shortly. Specifically, it consists of two steps as follows:

Aligning Masked Visual-Language-Audio. We freeze the audio encoder and focus on aligning visual, audio, and text features. For pre-training, we use a comprehensive set of image, video, and audio-video data. The combinations of modalities used are represented as {M,TM′} ∈ {{I,TI},{V,TV },{V,TVAS},{VA,TVAS}} where each pair denotes the concatenated features from the respective modalities.

Unmasked Visual-Audio-Language Post-Pretraining. We freeze the vision encoder to jointly align audio, visual, and text features. Post-pretraining is conducted using a smaller subset of image and video data (25M samples), along with the full set of audio (0.5M samples) and audio-video data (50M samples). Since the parameters of the largest ViT-6B model are frozen, we do not use masking strategies in this phase to ensure consistency with the inference process and to minimize any performance degradation in downstream tasks. The modality combinations used here are {M,TM′} ∈ {{I,TI},{V,TV },{A,TA},{V,TVAS},{VA,TVA}}}.

Table 1: Summary of datasets used in InternVideo2 pretraining process.

Pretraining Stage Dataset Domain # of clips Annotation Stage 1 KMash Web Video 2M -

|Stage 2 (img-txt)<br><br>|LAION, etc Web Image 300M Alt-text / Generated Caps|
|---|---|
|Stage 2 (vid-txt)<br><br>视频多模|WebVid2M Web Video 250k Alt-text WebVid10M Web Video 9.7M Alt-text InternVid Youtube Video 40M Generated Caption<br><br>InternVid2态数据集Youtube-VideoInternVid50M GeneratedCaption|

[Figure 15]

## 成果1：

Stage 3 LLaVA, etc Web Image/Video 2.1M Conversation, QA

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

|[Figure 20]<br><br>Video Cap: A green tractor with cables attached to it|
|---|

Video Captioner

[Figure 21]

|[Figure 22]<br><br>Audio Cap: A man is speaking and engine operates in the background|
|---|

Video

Audio Captioner

|[Figure 23]|
|---|

[Figure 24]

|[Figure 25]<br><br>ASR: I'll show you what they do|
|---|

Speech Captioner

Audio

|Video-Audio-Speech Caption: As the man talks about the tractor's capabilities, it is attached by cables and engine operates in the background.|
|---|

[Figure 26]

Crossmodal CapFusion

|Video-Audio Caption: The tractor operator is chatting while maneuvering the engine with cables attached to it.|
|---|

- Figure 3: The framework of our video multimodal annotation system, called VidCap, consists of four main components: video, audio, and speech captioners, along with a LLM for integrating captions from these modalities.

14

###### 3.3 Stage3: Predicting Next Token with Video-Centric Inputs

To further enrich the semantics embedded in InternVideo2 and improve its support for video-centric dialogue, we tune it by connecting it to a LLM with QFormer design [Li et al., 2022a,b]. We employ the progressive learning scheme in [Li et al., 2023d] by using InternVideo2 as the video encoder and train a video blip for communicating with open-sourced LLM [Zheng et al., 2023, Jiang et al., 2023]. Additionally, we implement a high-definition post-training stage to improve the model’s fine-grained and long spatiotemporal capabilities. During this stage, the input video is divided into up to six sub-videos with a resolution of 224x224 pixels each, along with one global resized sub-video of the same resolution. We then train the model for two additional epochs: the first epoch uses 8-frame video inputs, while the second epoch uses 16-frame inputs. During the additional training process, we update the video encoder and BLIP Qformer, while the LLM is updated using LoRA [Hu et al., 2021].

- 4 Multimodal Video Data

We list our training data in Tab. 1. Among the datasets used, KMash and InternVid2 are newly built, whereas the rest are publicly available.

###### 4.1 Video-only Data for Masked Autoencoders

We curate a new video set without labels named K-Mash from action recognition datasets [Carreira and Zisserman, 2017, Goyal et al., 2017a, Monfort et al., 2020, Heilbron et al., 2015, Zhao et al., 2019], as detailed in Supp. It encompasses a wide range of video types, including first- and third-person perspectives, with both short and long duration, and featuring various settings. Further, we give K-Mash2M with additionally sourced and selected 844K videos from YouTube for diversity.

- 4.2 Videos with Audio-Speech Modalities

We build a multimodal video dataset, coined as InternVid2, with video-audio-speech information and their descriptions for strengthening video perception via other modalities. It consists of 100M videos along with their VAS captions. In InternVid2, we collect videos from several sources (detailed in the supp), segment them into clips, and automatically annotate them based on their unimodal or crossmodal inputs. We highlight the importance of temporal segmentation in clip generation and our video mutimodal annotation system. We find refining them leads to notable downstream improvements.

Temporal Consistency Matters. We employ a temporal boundary detection model AutoShot [Zhu et al., 2023c] to segment videos into clips instead of SceneDet filter from FFmpeg. It predicts boundaries based on temporal semantic variations rather than pixel differences, and is able to generate semantically complete cuts without mixing extra frames with inconsistent context.

Video Multimodal Annotation. We design a video multimodal annotation system VidCap to give proper unimodal and crossmodal descriptions for textualizing videos from different perceptions. It automatically captions visual, audio, and speech of InternVid2, then it corrects them and fuses them for cross-modal captions via LLM. The system frame is given in Fig. 3. VidCap has independent video, audio, speech captioner, and a LLM for caption refinement and fusion. For video, speech, and caption postprocessing, we employ existing methods as the video captioning pipeline in [Wang

- et al., 2023d], WhisperV2-large model [Radford et al., 2023], and Vicuna-1.5 [Zheng et al., 2023]. For audio, we craft a audio captioner upon VideoChat [Li et al., 2023c], as we find no open-sourced one. It extracts audio features from inputs by Beats [Chen et al., 2023a]. We learn it by only tuning its Qformer using a combination of the large-scale audio-text corpus WavCaps [Mei et al., 2023] dataset. Details are given in the supplementary material.

4.3 Instruction-Tuning Data for Video Dialogue

We employ a updated training version of MVBench [Li et al., 2023d]. Originally, it comprises 1.9M samples (both images and videos) from 34 distinct sources. We decrease the amount of caption data from WebVid and CoCo to 80k and 100k separately and add new data from the S-MiT to boost the diversity rather than quantity of the instruction dataset. In the additional HD training stage, we incorporate the videos with corresponding GPT-4 annotations from [Chen et al., 2024a] into the training set. We further expand the training set by including datasets from PerceptionTestQA [Patraucean

- et al., 2024], TVQA [Lei et al., 2018], NTU-RGB-D [Liu et al., 2020], and EgotaskQA [Grauman et al., 2022], along with grounding datasets based on DiDeMo [Anne Hendricks et al., 2017] and COCO [Lin et al., 2014]. This training data encompasses key features of image and video understanding across crucial tasks, including 1) conversation, 2) caption, 3) visual question answer, 4) reasoning, and 5) classification.

##### 5 Experiments

In our evaluation of InternVideo2, we assess the models from three learning stages. The evaluation covers a wide range of tasks, including video recognition, video retrieval, question-answering, and more. It includes various scenarios such as zero-shot learning, finetuning, and linear probing.

For InternVideo2 trained in stage 1, 2, and 3, we denote them with InternVideo2s1, InternVideo2s2, and InternVideo2s3, respectively. We also learn a CLIP-style InternVideo2 indicated by InternVideo2clip. It is post-pretrained from InternVideo2s2 by only preserving video and text encoders and contrastive loss.

Each training stage of InternVideo2-6B uses different configurations and resources. In the first stage, we employ 256 NVIDIA A100 GPUs and train the model for 18 days. The second stage also utilizes 256 A100 GPUs and spans a training period of 14 days. Finally, in the third stage, we use 64 A100 GPUs and train the model for 3 days. We introduce DeepSpeed and FlashAttention [Dao et al., 2022] for training and inference. More implementation details and experiment results are given in the supp.

- 5.1 Video Classification

- 5.1.1 Action Recognition

We test InternVideo2 on Kinetics (i.e., K400, 600, and 700 [Carreira and Zisserman, 2017, Carreira et al., 2018, 2019]), Moments in Time V1 (MiT) [Monfort et al., 2020], Something-Something V2 (SSv2) [Goyal et al., 2017a], UCF [Soomro et al., 2012], HMDB [Kuehne et al., 2011], Charades [Gao et al., 2017], ActivityNet [Heilbron et al., 2015] (ANet) and HACS [Zhao et al., 2019]. We evaluate in four settings: (a) end-to-end finetuning the whole backbone;

- Table 2: End-to-end finetuning action recognition results (top-1 accuracy) on Kinetics, SomethingSomething, and Moments in Time. † denotes that the result is achieved with different resolutions or frame rates.

Method Training Data Setting K400 K600 K700 Sth-Sthv2 MiT ANet HACS

CoVeR [Zhang et al., 2021] IV-3B 16 × 448 87.1 87.9 79.8 70.8 46.1 - Hiera-H [Ryali et al., 2023] V-0.25M 16 × 224 87.8 88.8 81.1 76.5 - - CoCa-g [Yu et al., 2022] I-3B 16 × 576 88.9 89.4 82.7 - 49.0 - MTV-H [Wang et al., 2023b] IV-370M 32 × 280 89.9 90.3 83.4 - - - VideoMAEv2-g [Wang et al., 2023b] V-1.35M 64 × 266 90.0 89.9 - 77.0† - - V-JEPA-H [Bardes et al., 2024] V-2M 16 × 224 - - - 77.0 - - MVD-H [Wang et al., 2023c] IV-1.25M 16 × 224 - - - 77.3 - - UniFormerV2-L [Li et al., 2022c] IV-401M 64 × 336 90.0 90.1 82.7 73.0† 47.8† 94.7 95.4

InternVideo [Wang et al., 2022] V-12M ensemble 91.1 91.3 84.0 77.2 - - InternVideo2s1-1B IV-1.1M 8 × 224 91.3 91.4 85.0 77.1 50.8 - InternVideo2s1-1B IV-1.1M 16 × 224 91.6 91.6 85.4 77.1 50.9 - InternVideo2s1-6B IV-2M 8 × 224 91.9 91.7 85.7 77.5 51.0 - InternVideo2s1-6B IV-2M 16 × 224 92.1 91.9 85.9 77.4 51.2 95.9 97.0

- Table 3: Attentive probing recognition results (top-1 accuracy) on Kinetics-400/600/700, Moments in Time and Something-Something V2.

Method Training Data Setting K400 K600 K700 MiT SSV2

UMT-L [Li et al., 2023e] IV-25M - 82.8 - - 40.3 54.5 VideoMAEv2-g [Wang et al., 2023b] V-1.35M - 82.1 - - 35.0 56.1 V-JEPA-H [Bardes et al., 2024] V-2M 16 × 384 81.9 - - - 72.2 DINOv2-g [Oquab et al., 2023] I-142M 16 × 224 83.4 - - - 50.0 VideoPrism-g [Zhao et al., 2024] V-619M 16 × 288 87.2 - - 45.5 68.5 ViT-e [Dehghani et al., 2023] I-4B 128 × 224 86.5 - - 43.6 ViT-22B [Dehghani et al., 2023] I-4B 128 × 224 88.0 - - 44.9 CoCa-g [Yu et al., 2022] I-3B 16 × 576 88.0 88.5 81.1 47.4 -

InternVideo2s2-1B IV-25.5M 16 × 224 87.9 88.0 79.5 46.3 67.3 InternVideo2s2-6B IV-400M 16 × 224 88.8 89.1 81.0 47.8 67.7

(b) attentive probing is similar to linear pooling, but extra trains the attention pooling layer [Yu et al., 2022]. (c) linear probing which freezes the backbone and only trains the task head; and (d) zero-shot.

End-to-end Finetuning. Tab. 2 shows InternVideo2-6B obtains new state-of-the-art (SOTA) results on Kinetics (92.1%/91.9%/85.9% on K400/600/700, respectively), SthSthv2, MiT, ANet, and HACS with only 16 frames, while the previous SOTAs require larger resolution (224 vs. 576) or model ensemble. As for MiT in Tab. 2, InternVideo2-6B exceeds the previous SOTA, CoCa-g, by a significant margin of 2.2% (51.2% vs. 49.0%). Regarding the temporalrelated actions in Tab. 2, our InternVideo2-6B also surpasses MVD [Wang et al., 2023c] on SSv2 (77.5% vs. 77.3%). Moreover, our InternVideo2-6B showcases top performance on untrimmed video analysis, as indicated in Tab. 2, with 95.9% on ActivityNet and 97.0% on HACS. These results affirm our model’s superior capability for robustly identifying complex actions across varied scenes. Note “I" and “V" denotes images and videos, respectively. “IV-3B" means the total number of the used images and videos is 3B, while “I-3B" means using 3B images.

Attentive Probing. As in Tab. 3, InternVideo2-6B not only outperforms ViT-22B [Dehghani et al., 2023] and CoCa-g [Yu et al., 2022] in scene-focused datasets but also surpasses or matches the performance of the latest video foundation model [Bardes et al., 2024, Zhao et al., 2024], on datasets emphasizing temporal dynamics (SthSthV2). This underscores our model’s exceptional ability to understand and interpret both spatial and temporal information effectively.

Linear Probing. In Tab. 4, InternVideo2-1B significantly outperforms the previous SOTA, DINOv2-g [Oquab et al., 2023], by notable margins: +3.2% on K400, +8.0% on SthSthV2, and +4.8% on UCF-101. As we scale the model, an upward trend in results is observed, underscoring the benefits of model enhancement. Notably, the integration of multimodal pretraining (stage 2) yields further rise in results. We suppose stage 2 enhances feature discrimination.

Zero-shot. Table 5 and 6 show InternVideo2 gets 72.7% / 71.7% / 64.2% on K400/600/700, respectively, outperforming others but VideoPrism on K400 (76.4%). On UCF [Soomro et al., 2012], HMDB [Kuehne et al., 2011], MiT [Monfort et al., 2020], SSv2-MC, and Charades, InternVideo2 gives an cutting edges over others. The clear gap between VideoPrism and InternVideo2 on K400 may signify the importance of pretraining corpus in VideoPrism (311M

- Table 4: Linear probing action recognition results (top-1 accuracy) on Kinetics-400, Something-Something V2, UCF-101 and HMDB-51.

Method Setting K400 SSV2 UCF-101 HMDB-51

VideoMAEv2-H [Wang et al., 2023b] 12 × 224 25.8 - 56.4 34.1 TVTSv2-H [Zeng et al., 2023] 12 × 224 73.1 - 91.8 65.7 OpenCLIP-G [Cherti et al., 2023] 8 × 224 78.3 35.8 90.7 DINOv2-g [Oquab et al., 2023] 8 × 224 78.4 38.3 91.2 InternVideo2s1-1B 16 × 224 81.6 46.3 96.0 71.6

- InternVideo2s1-6B 16 × 224 82.0 47.8 96.3 71.8
- InternVideo2s2-6B 16 × 224 84.2 56.7 97.3 80.7

Table 5: Zero-shot action recognition on UCF, HMDB, MiTv1, SSv2-MC, and Charades.

Method #F Training Data UCF HMDB MiT SSv2-MC Charades

CLIP [Radford et al., 2021] 12 I-400M 68.9 43.2 - 29.6 TVTSv2 [Zeng et al., 2023] 12 V-8.5M 78.0 52.1 - 48.4 VideoCoCa-g [Yan et al., 2022] 16 V-145M - - - - 25.8 VideoPrism-g [Zhao et al., 2024] 16 V-619M - - - - 32.4

InternVideo2clip-1B 8 IV-25.5M 88.8 53.9 31.6 61.5 32.9 InternVideo2clip-6B 8 IV-400M 89.5 56.7 32.9 63.5 34.6

Table 6: Zero-shot action recognition results on Kinetics.

K400 K600 K700 top-1 AVG top-1 AVG top-1 AVG

Method #F

CLIP [Radford et al., 2021] 8 58.4 70.1 55.1 67.2 46.1 58.4 EVA-CLIP-L [Sun et al., 2023b] 1 - 65.0 - 64.9 - 59.1 EVA-CLIP-E [Sun et al., 2023b] 1 - 69.8 - 69.3 - 63.4 ViCLIP-L [Wang et al., 2023d] 8 64.8 75.7 62.2 73.5 54.3 66.4 VideoCoCa-g [Yan et al., 2022] 16 72.0 81.3 - - - InternVL-6B [Chen et al., 2023c] 8 69.1 79.4 68.9 78.8 60.6 71.5 EVA-CLIP-18B [Sun et al., 2024] 16 - 79.4 - 79.4 - 72.2 VideoPrism-g [Zhao et al., 2024] 16 76.4 85.4 - - - -

InternVideo2clip-1B 8 73.1 82.4 72.8 81.8 64.9 75.2 InternVideo2clip-6B 8 72.7 82.2 71.7 81.2 64.2 75.2

videos with text and 36.1M of them are manually labeled) for K400 in zero-shot. Note that on the datasets of Kinetics, UCF101 and HMDB51, which have a distribution closer to the pre-training dataset used in stage1, the performance of Internvideo2-6B is slightly inferior to that of Internvideo2-1B. We suppose this is caused by Internvideo2-6B uses a more abundant pretraining dataset in stage2, leading to the forgetting of pretraining data in stage1.

###### 5.1.2 Temporal Action Localization

We evaluate models on four temporal action localization (TAL) datasets: THUMOS14 [Idrees et al., 2017], ActivityNet [Krishna et al., 2017], HACS Segment [Zhao et al., 2019] and FineAction [Liu et al., 2022] in a feature-based manner with finetuning. We employ output of the 7-th layer from InternVideo2 for inputs as the corresponding features without fusing anything else. ActionFormer [Anne Hendricks et al., 2017] is used as the detection head. We report mean Average Precision(mAP) under multiple tIoU as in [Lin et al., 2019, Chen et al., 2022a, Anne Hendricks et al., 2017, Yang et al., 2023]. In Table 7, InternVideo2-6B gets the highest mAP among all comparisons in all datasets, while InternVideo2-1B nearly surpass other methods except in THUMOS14. We find InternVideo2-6B almost consistently improves mAP with a notable margin from InternVideo2-1B except in FineAction. We suppose scaling model capacity without data refinement cannot nontrivially improve fine-grained discrimination abilities of models. Scaling detailed annotations in training may address this issue.

###### 5.1.3 Video Instance Segmentation

We evaluate on the Video Instance Segmentation (VIS) dataset Youtube-VIS 2019 [Yang et al., 2019]. Built upon Mask2Former [Cheng et al., 2021], we employ the video encoder of InternVideo2 as backbone with ViT-adapter [Chen

Table 7: Finetuned temporal action localization results on THUMOS14 [Idrees et al., 2017], ActivityNet [Krishna et al., 2017], HACS Segment [Zhao et al., 2019] and FineAction [Liu et al., 2022]. We report average mAP. “Flow” denotes the ensembling I3D flow feature. * denotes the result is achieved with Flow.

Backbone THUMOS14 HACS ActivityNet FineAction I3D [Carreira and Zisserman, 2017] + Flow 66.8 - 35.6 R(2+1)D [Tran et al., 2018] 55.6 - 36.6 InternVideo 71.6∗ 41.3 39.0 17.6 VideoMAEv2-g [Wang et al., 2023b] 69.5 - - 18.2 InternVideo2s1-1B 69.8 42.4 40.4 27.2 InternVideo2s1-6B 72.0 43.3 41.2 27.7

Table 8: Video instance segmentation performance (mAP) on YouTube-VIS19 [Yang et al., 2019].

Method Backbone #Params YouTubeVIS19

Mask2Former Swin-L (image) [Liu et al., 2021] 219M 60.3 Mask2Former InternViT (image) 6B 63.4

Mask2Former InternVideo2s1 6B 64.2

- Table 9: Results of zero-shot video retrieval in both text-to-video (T2V) and video-to-text (V2T) on MSR-VTT, LSMDC, DiDeMo, MSVD, ActivityNet (ANet), and VATEX.

Method

MSR-VTT LSMDC DiDeMo MSVD ANet VATEX T2V V2T T2V V2T T2V V2T T2V V2T T2V V2T T2V V2T

CLIP [Radford et al., 2021] 30.4 24.2 13.9 11.9 12.7 18.7 40.5 57.2 9.1 13.2 - CLIP4Clip [Luo et al., 2022] 32.0 - 15.1 - - - 38.5 - - - - ViCLIP [Wang et al., 2023d] 42.4 41.3 20.1 16.9 18.4 27.9 49.1 75.1 15.1 24.0 - InternVideo-L [Wang et al., 2022] 40.7 39.6 17.6 13.2 31.5 33.5 43.4 67.6 30.7 31.4 49.5 69.5 UMT-L [Li et al., 2023e] 40.7 37.1 24.9 21.9 48.6 49.9 49.0 74.5 41.9 39.4 - VideoCoCa-g [Yan et al., 2022] 34.4 64.7 - - - - - - 34.5 33.0 53.2 73.6 VideoPrism-g [Zhao et al., 2024] 39.7 71.0 - - - - - - 52.7 50.3 62.5 77.1

InternVideo2s2-1B 51.9 50.9 32.0 27.3 57.0 54.3 58.1 83.3 60.4 54.8 70.4 85.4 InternVideo2s2-6B 55.9 53.7 33.8 30.1 57.9 57.1 59.3 83.1 63.2 56.5 71.5 85.3

- Table 10: Results of finetuning video retrieval in both text-to-video (T2V) and video-to-text (V2T) on MSR-VTT, LSMDC, DiDeMo, MSVD, ActivityNet (ANet), and VATEX.

MSR-VTT LSMDC DiDeMo MSVD ANet VATEX T2V V2T T2V V2T T2V V2T T2V V2T T2V V2T T2V V2T

Method

CLIP [Radford et al., 2021] 38.2 38.7 22.5 22.6 32.2 33.9 - - 26.1 26.9 - CLIP4Clip [Luo et al., 2022] 45.6 45.9 24.3 23.8 43.0 43.6 45.2 48.4 40.3 41.6 - ViCLIP [Wang et al., 2023d] 52.5 51.8 33.0 32.5 49.4 50.2 - - 49.8 48.1 - UMT-L [Li et al., 2023e] 58.8 58.6 43.0 41.4 70.4 65.7 58.2 82.4 66.8 64.4 72.0 86.0 InternVideo2s2-6B 62.8 60.2 46.4 46.7 74.2 71.9 61.4 85.2 74.1 69.7 75.5 89.3

- et al., 2022b] for features. We also try InternViT [Chen et al., 2023c] for comparisons. In Table 8, InternVideo2 gets the highest mAP among all. This validates its effectiveness in relatively fine-grained spatiotemporal perception.

###### 5.2 Video-Audio-Language Tasks

We evaluate InternVideo2 on video retrieval, captioning, and multi-choice question-answersing (QA). The former two tasks are conducted by matching video representation and the candidate text ones using the text encoder in stage 2. The latter is tested by the VideoLLM learned in stage 3. We also test audio tasks.

###### 5.2.1 Video Retrieval

We evaluate the video retrieval on six popular benchmarks: MSR-VTT [Xu et al., 2016], LSMDC [Rohrbach et al., 2015], DiDeMo [Anne Hendricks et al., 2017], MSVD [Chen and Dolan, 2011], ActivityNet (ANet) [Heilbron et al.,

- Table 11: Finetuned temporal grounding on QVHighlight [Lei et al., 2021] and Charade-STA [Gao et al., 2017]. (a) QVHighlight

Feature R1@0.5R1@0.7 mAP mAP HiT@1

CLIP [Radford et al., 2021] 64.97 48.65 42.96 39.83 64.19 CLIP+SlowFast [Feichtenhofer et al., 2019] 65.43 48.38 42.86 40.33 66.21 InternVideo2s2-1B 70.00 54.45 47.02 42.36 69.74 InternVideo2s2-6B 71.42 56.45 49.24 42.90 72.00

(b) Charade-STA

Feature R1@0.3R1@0.5R1@0.7mIoU CLIP [Radford et al., 2021] 65.62 52.77 30.16 45.85 CLIP+SlowFast [Feichtenhofer et al., 2019] 70.43 58.44 36.34 50.13 InternVideo2s2-1B 78.41 68.36 45.03 57.12 InternVideo2s2-6B 79.70 70.03 48.95 58.79

- Table 12: Audio retrieval results on AudioCaps [Kim et al., 2019], Clothov1, and Clothov2 [Drossos et al., 2020]. We report text-to-audio R@1 accuracy in zero-shot and finetuning settings.

Method

Zero-shot Finetuning AudioCaps ClothoV1 ClothoV2 AudioCaps ClothoV1 ClothoV2

VIP-ANT [Zhao et al., 2021] 27.7 - - VAST [Chen et al., 2024b] - - - 52.0 25.1 26.9 LanguageBind [Zhu et al., 2023a] - 12.1 12.1 - - InternVideo2s2-6B 37.1 17.4 17.4 55.2 25.3 27.2

- Table 13: Results of AudioQA on ClothoAQA [Lipping et al., 2022] and Audio-MusicAVQA (AMAVQA) [Behera

- et al., 2023], and audio classification on the ESC-50 [Piczak, 2015]. Both in the finetuning setting. (a) ClothoAQA and AMAVQA

###### (b) ESC-50

Backbone ClothQA AMAVQA

Method Top-1 Acc

AquaNet [Lipping et al., 2022] 14.78 65.59 MWAFM [Li et al., 2023b] 22.24 67.54 InternVideo2s2 30.14 80.51

AST [Gong et al., 2021] 95.60 BEATs [Chen et al., 2023a] 98.10 InternVideo2s2 98.60

2015], and VATEX [Wang et al., 2019], as shown in Tab. 9 and 10. In evaluation, eight frames from the input videos are uniformly sampled. We report R@1 scores for both text-to-video (t2v) and video-to-text (v2t) tasks in Tab. 9 and 10. R@5 and R@10 are given in Supp.

Tab. 9 and 10 demonstrate that InternVideo2 outperforms other state-of-the-arts with a notable margin in both t2v and v2t of all used datasets no matter in zero-shot or finetuned settings, except for the v2t of MSR-VTT, where VideoPrism gives the best result. This shows the video-language semantic alignment of transferrity of InternVideo2.

###### 5.2.2 Video Temporal Grounding

We evaluate InternVideo2 on two video temporal grounding (VTG) datasets: QVhighlight [Lei et al., 2021], and Charade-STA [Gao et al., 2017]. The eval setting and used features are the same as in TAL. We use CG-DETR [Moon et al., 2023] as the grounding head. We report R1@0.3, R1@0.5, R1@0.7, and mAP for moment retrieval as in [Lei et al., 2021, Moon et al., 2023, Lin et al., 2023]. Highlight Detection is evaluated in terms of “Very Good” mAP and HiT@1. In Table 11, InternVideo2-1B and InternVideo2-6B bring gradual performance improvements compared to CLIP [Radford et al., 2021] and CLIP [Radford et al., 2021]+Slowfast [Feichtenhofer et al., 2019]. This suggests that a larger spatiotemporal model is more beneficial to short-term video semantic alignment capabilities.

###### 5.2.3 Audio-related Tasks

We evaluate InternVideo2’s audio and text encoders on audio tasks, including audio-text retrieval on AudioCaps [Kim et al., 2019], Clothov1, and Clothov2 [Drossos et al., 2020]; audioQA on ClothoAQA [Lipping et al., 2022] and Audio-MusicAVQA [Behera et al., 2023]; and audio classification on the ESC-50 [Piczak, 2015]. As shown in Tab. 12, 13a, and 13b, our model achieves state-of-the-art performance on all downstream tasks. Considering the limited size of the used audio and text encoders, these audio-related results show crossmodal contrastive learning’s benefits are mutual to the used modalities. Audio and the corresponding text models also gain from this learning.

- Table 14: Results of Chat-centric Evaluation on Multiple Choice Video-QA on MVBench [Li et al., 2023d], Egoschema [Mangalam et al., 2023] and Perception Test [Patraucean et al., 2024].

Model ViEncoder LLM MVBench Egoschema Perception Test

GPT-4V [OpenAI, 2023b] - GPT-4 43.5 - Gemini 1.0 Pro [Team et al., 2023] - - 37.7 55.7 51.1 Gemini 1.0 Ultra [Team et al., 2023] - - - 61.5 54.7 Gemini 1.5 Pro [Team et al., 2023] - - - 72.2 -

|LLaVA-Next-Video [Liu et al., 2024] VideoLLaMA2 [Cheng et al., 2024] VideoLLaMA2 [Cheng et al., 2024]<br><br>|CLIP-L CLIP-L-336 CLIP-L-336<br><br>|Vicuna-7B Mistral-7B Mistral-8*7B|46.5 54.6 53.9<br><br>|43.9 51.7 53.3|48.8 51.4 52.2<br><br>|
|---|---|---|---|---|---|
|VideoChat2 VideoChat2 VideoChat2-HD VideoChat2-HD-F16<br><br>|UMT-L<br><br>InternVideo2s3-1B InternVideo2s3-1B InternVideo2s3-1B<br><br>|Vicuna-7B Mistral-7B Mistral-7B Mistral-7B|51.1 60.3 65.4 67.2<br><br>|55.8 60.2 60.0<br><br>|53.0 60.1 63.4<br><br>|

- Table 15: Average top-1 accuracy of action recognition (K400, SSv2, and MiT) and video retrieval (MSR-VTT, LSMDC, DiDeMo, MSVD, ANet, and VATEX in t2v) using zero-shot and finetuning settings. * denotes results are from InternVideo2s1.

Model

Zero-shot Finetuning

Action Recognition Video Retrieval Action Recognition

InternVideo2s2-1B 55.5 55.0 73.2* InternVideo2s2-6B 56.9(+1.4) 56.9(+1.9) 73.6(+0.4)*

- Table 16: Ablation of Stage1, conducted with finetuned action recognition on Kinetics, MiT, and SthSthv2. All models are tested with 8×224×224 input.

Model Teacher Data K400 K600 K700 MiT SSv2 Avg ViT-L CLIP-L K710 90.3 90.4 83.2 48.0 74.7 77.3 ViT-L CLIP-L K-Mash1.1M 90.5 90.4 83.4 48.1 74.7 77.4 ViT-1B InternVL-6B K710 90.9 91.0 84.7 49.8 75.9 78.5 ViT-1B InternVL-6B K-Mash1.1M 91.4 91.5 85.1 50.5 76.5 79.0 ViT-1B InternVL-6B+VideoMAE-g K-Mash1.1M 91.3 91.4 85.0 50.8 77.1 79.1 ViT-1B InternVL-6B K-Mash2M 91.3 91.5 85.1 50.6 76.6 79.0 ViT-6B InternVL-6B+VideoMAE-g K-Mash2M 91.9 91.7 85.7 51.0 77.5 79.6

###### 5.3 Video-centric Dialogue and its Applications

- Tab. 14 shows results on MVBench [Li et al., 2023d], Egoschema [Mangalam et al., 2023], and Perception Test [Patraucean et al., 2024], by equipping VideoChat2 [Li et al., 2023d] with our InternVideo2 encoder, compared with other MLLMs. Note VideoChat2-HD-F16 with InternVideo2 outperforms other systems with a clear margin in Perception Test and MVBench (F16 denotes the model is trained and tested with 16-frame inputs), except in Egoschema compared with proprietary commercial models (GPT4 & Gemini). Our method demonstrates superior short-term fine-grained video understanding compared to both open-source and commercial methods. Egoschema requires longer context harnessing, and we plan to further explore leveraging LLM for long multimodal sequence modeling. Considering these benchmarks not only involve perception but also reasoning, these results suggest InternVideo2 does embed knowledge to partially model the world. It also validates the importance of learning a transferrable video representation for current video-related MLLM. We also give several qualitative evaluations with popular GPT-4V and GeminiPro, including dialogues on action sequence (Fig. 4), confused action (Fig. 5), temporal order understanding (Fig. 6), temporal event counting (Fig. 7, unexpected action reasoning (Fig. 8), and vision-language navigation (Fig. 9).

5.4 Ablation Studies

- 5.4.1 Scaling Video Encoder

- Tab. 15 gives InternVideo2’s average performance on action recognition and video retrieval. It shows that scaling video encoder from 1B to 6B still leads to notable improvements in generalization of action recognition and video-retrieval by

|What happened before the person took the blanket? Before the person took the blanket, they were sitting on the couch, possibly watching<br><br>TV or engaging in some other activity. They might have been relaxing or enjoying their<br><br>time on the couch. The presence of the remote control suggests that they were likely controlling a television or other electronic device. The person then decided to take the blanket, possibly to use it for warmth or comfort while sitting on the couch.<br><br>It appears that the person has been prompted to get a blanket either due to feeling cold while watching something on the television or as a reaction to what they're watching, seeking comfort or cozying up.<br><br>Before the person took the blanket, he sat in the middle of the sofa watching TV.<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>Gemini Pro<br><br>[Figure 36]<br><br>GPT-4V<br><br>InternVideo2Chat<br><br>❌<br><br>✔<br><br>✔<br><br>|
|---|

- Figure 4: Temporal action recognition tasks. In questions about an action before it happens, Gemini Pro and InternVideo2-Chat both describe accurately the action, while GPT-4V hallucinates.

|What is the action performed by the person in the video?<br><br>The person in the video is peeling a banana.<br><br>It appears that the person in the video is performing a series of actions to create the illusion of peeling a banana but then revealing it as unpeeled.<br><br>The person in the video is dropping a banana.<br><br>[Figure 37]<br><br>[Figure 38]<br><br>Gemini Pro<br><br>[Figure 39]<br><br>GPT-4V<br><br>InternVideo2Chat<br><br>❌<br><br>✔<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]|
|---|

- Figure 5: Confused action recognition. The person in the video is performing a misleading action while holding a banana. Gemini Pro gives a wrong answer. GPT-4V identifies the misleading action but doesn’t give a correct answer. InternVideo2-Chat gives a correct answer.

1.4% and 1.9% (in zero-shot), respectively. Meanwhile, the increase in finetuned action recognition result is relatively marginal (0.4%) with the growth of model scale.

###### 5.4.2 Training Data and used Teachers in Stage 1

In Tab. 16, we examine the impact of distillation teachers and used dataset sizes on model performance. (a) Data Scale: Note that pretraining data scale for MAE should grow with the increasing model scale, otherwise the downstream performance would be saturated, such as K710 (0.66M videos) for ViT-L, K-Mash1.1M for ViT-1B, and K-Mash2M for ViT-6B. (b) Teacher: Tab. 16 reveals that the synergy between a multimodal teacher (e.g., CLIP [Radford et al., 2021]) and a motion-aware teacher (e.g., MAE [Tong et al., 2022]) markedly boosts performance, especially on SthSthV2. It highlights the importance of strategic teacher model selection in the distillation process.

|What letters did the person show in order?<br><br>The person showed the letters "I", "N", "K", "J", "M", and "P" in order.'<br><br>A. K. Y. L. I. N. I.<br><br>The person displayed the letters [j k m a n] in sequence.<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Gemini Pro<br><br>[Figure 48]<br><br>GPT-4V<br><br>InternVideo2Chat<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>|
|---|

- Figure 6: Video Object Temporal Recognition. The person in this video takes out different letters in the order of time. Gemini Pro recognizes 4 letters, but the order is totally reversed; GPT-4V recognizes only 3 letters, and the result is mixed with wrong answers; InternVideo2-Chat has the fewest errors among them and the order is correct.

|How many times did the person launch objects on the table?<br><br>The person launched objects on the table 6 times.(I'm not programmed to assist with<br><br>that.) Based on the sequence provided, it appears that the person launches objects on the table three times. In the provided frames, you can see the person's hand motion, which<br><br>suggests the launching action of the objects. The three sequences appear to show the<br><br>progression of three separate launching actions. The person launched objects on the table 3 times.<br><br>[Figure 55]<br><br>[Figure 56]<br><br>Gemini Pro<br><br>[Figure 57]<br><br>GPT-4V<br><br>InternVideo2Chat<br><br>❌<br><br>✔<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>✔|
|---|

- Figure 7: Event counting task. Both GPT-4V and InternVideo2-Chat are able to correctly capture the times of actions and not be confused by redundant frames and other actions.

###### 5.4.3 Training Arch, Method, and Data in Stage 2

We ablate the necessity of introducing an audio encoder in Stage 2. We employ ViT-B and Bert-B for video and text encoders, respectively. The used text are simple video captions. The baseline is conducting video-text contrastive and matching as well as masked language generation loss for training with only video and text encoders. Other settings including adding audio or speech encoder or them both, and how to update newly added encoders i.e., whether train them with only text encoder or both video and text encoders. Tab. 17 shows that only introducing audio encoder and learn it along with both video and text encoders can best improve video-text retrieval performance. The speech encoder harms such effectiveness more or less.

Further, we verify the impact of video temporal segmentation and the used captions as text inputs in Stage 2 in Tab. 18. We still use ViT-B and Bert-B for video-text training. Tab. 18 finds the fused text from video-audio-speech works best for retrieval tasks compard with others, rising zero-shot t2v R1 of MSR-VTT from 24.7 to 27.1. Moreover, using AutoShot instead of SceneDet notably improves t2v retrieval (increasing by nearly 7 points). It validates the effectiveness of the introduced video-text dataset and its annotation systems.

|What transition does the man make in the magic trick showcased in the video?<br><br>The man makes a transition from a 2D drawing of an elephant to a 3D elephant<br><br>sculpture.<br><br>There is a transitional frame where the paper closely approaches or covers the camera lens. This likely serves as the moment of the trick where the switch or transformation is supposed to occur.<br><br>In the video, a man makes a transition by simply changing the drawings of an elephant into a plastic elephant doll.<br><br>[Figure 64]<br><br>[Figure 65]<br><br>Gemini Pro<br><br>[Figure 66]<br><br>GPT-4V<br><br>InternVideo2Chat<br><br>❌<br><br>✔<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>✔|
|---|

- Figure 8: Unexpected Action Recognition tasks. The model needs to recognize the magical parts of the video. Both Gemini Pro and InternVideo2-Chat can capture part of the transition in the video and infer the shooting technique of the video. GPT-4V recognizes the transition but fails to successfully explain the process of the transition.

|This is a navigation video of an agent following instruction: "Go up the stairs. Take a left at the top of the stairs. Go into the bedroom on the left. Stop in the doorway."<br><br>What is the next action it should take?<br><br>The agent should stop in the doorway.<br><br>After reaching the top of the stairs, the agent needs to take a left turn, as depicted in the frames, where the camera orientation shifts to show the left turn.<br><br>The next action is to go into the bedroom on the left.<br><br>[Figure 73]<br><br>[Figure 74]<br><br>Gemini Pro<br><br>[Figure 75]<br><br>GPT-4V<br><br>InternVideo2-<br><br>Chat<br><br>❌<br><br>✔<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]|
|---|

- Figure 9: Visual Language Navigation Tasks. GPT-4V and InternVideo2-Chat are able to understand the instruction and make decisions about next steps based on the content of the video, while Gemini Pro is subject to hallucination.

###### 5.4.4 Training and Evaluation in Stage 3

As shown in Tab. 19, incorporating questions (i.e., the ‘q’ in ‘QA’) into QFormer during Stage 3 training, which was found useful in [Dai et al., 2023], actually harms the out-of-domain performance of the scaled-up VideoLLM. The NextQA training data is already included in the training corpus, and this is the only benchmark where the questioninjected version performs better. Therefore, we believe that adding questions to QFormer during the instruction tuning stage of the scaled VideoChat model leads to some degree of overfitting.

##### 6 Conclusion and Discussion

We have introduced a new family of video foundation models called InternVideo2, which achieves the state-of-the-art performance across various video and audio tasks. In InternVideo2, we combine masked video modeling, video-audio-

- Table 17: Ablation of Stage2. All models are tested with 8×224×224 input.

Method MSR-VTT

Baseline (video & text encoders w/ video-text learning) 24.7 Baseline + audio encoder w/ audio-text learning 24.0 Baseline + speech encoder w/ video-speech-text learning 24.9 Baseline + audio encoder w/ video-audio-text learning 27.8 Baseline + audio & speech encoders + video-audio-speech-text learning 25.7

- Table 18: Zero-shot t2v retrieval on MSR-VTT with different training captions.

SceneDet AutoShot Video Cap Audio Cap Speech Cap MSR-VTT

24.7 26.6 27.1 34.8

Table 19: Ablation on using qformer instruction in Stage3 training of Chat-Centric Model.

Model MVBench NextQA Egoschema-full Egoschema-subset VideoChat2 w qformer inst 59.9 79 52.9 65.8 VideoChat2 w/o qformer inst 60.4 (+0.5) 78.6 (-0.4) 55.8 (+2.9) 66.4 (+0.6)

text contrastive learning, and next token prediction into a unified framework. Additionally, we create a new video-text dataset that incorporates video-audio-speech fused captions as descriptions. The dataset contains temporally segmented clips with high semantic coherence. These designs in InternVideo2 contribute to enhancing video understanding in both perception and reasoning tasks. Notably, InternVideo2 excels in video-related dialogue and long video understanding, demonstrating its effectiveness in capturing high-level semantics.

Limitations and Discussions. Despite its achievements, InternVideo2 does not introduce specific novel architectural design. Instead, it leverages the existing learning techniques for scaling video foundation models while focusing on improving data processing to enhance its spatiotemporal perception, semantic alignment, and basic knowledge embedding. Similarly to previous studies [Li et al., 2023e, Ye et al., 2023], InternVideo2 still grapples with limitations stemming from fixed input resolutions, sampling rates, and highly compressed tokens, which restrict its ability to express rich video information and capture fine-grained details.

The progressive learning scheme adopted by InternVideo2 strikes a balance between model capabilities and training compute. While jointly learning the three optimization objectives simultaneously is computationally feasible, scalability becomes an issue when confronted with limited resources.

Although InternVideo2 has demonstrated leading performance in some video understanding and reasoning benchmarks, it cannot guarantee an implicit world model that ensures consistency in visual reasoning. The inherent constraints imposed by fixed input representations, coupled with the complexity of visual reasoning tasks, present challenges in achieving a comprehensive and consistent understanding of the visual world.

Potential Biases. We investigate the potential biases here. We focus on age, gender, and race distributions, as these are commonly recognized areas where bias can occur. We count keywords related to these categories in the used captions. Note that these synthetic captions may not fully reflect the truth of the corresponding videos, thereby creating a gap between our analysis and the actual reality. Here are the results of our analysis:

- • Age: The majority were about adults (86.99%), followed by children (12.87%) and barely any mentions of senior citizens (0.04%).
- • Gender: 62.04% pertained to men and 37.96% pertained to women.
- • Race: 56.19% are Asians, 23.04% are Black people, 14.55% are White people, 3.78% are Middle Eastern people, and 2.43% are Latin American people.

##### 7 Broader Impact

It is important to acknowledge that, similar to other foundational models, InternVideo2 has the potential to embed biases present in its training data and the associated models used during training, such as neural teachers [Tong et al., 2022, Chen et al., 2023c] and language models (LLMs) [Jiang et al., 2023, Zheng et al., 2023]. These biases may emerge due to a variety of factors, including the personal ideas, preferences, values, and perspectives of the data creators and the training corpus utilized.

The presence of biases in AI models can have societal implications and reinforce existing inequalities or prejudices. Biases within InternVideo2 could manifest in the form of unfair or discriminatory outputs, potentially perpetuating social biases or stereotypes present in the training data. Consequently, it is crucial to carefully consider the potential impact of deploying InternVideo2 in real-world applications and take proactive measures to mitigate biases and ensure fairness.

##### References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. ArXiv, abs/2204.14198, 2022.

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. V-JEPA: Latent video prediction for visual representation learning, 2024. URL https://openreview. net/forum?id=WFYbBOEOtv.

Swarup Ranjan Behera, Krishna Mohan Injeti, Jaya Sai Kiran Patibandla, Praveen Kumar Pokala, and Balakrishna Reddy Pailla. Aquallm: Audio question answering data generation using large language models. arXiv preprint arXiv:2312.17343, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. arXiv preprint arXiv:2402.15391, 2024.

Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.

João Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-600. ArXiv, abs/1808.01340, 2018.

João Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. ArXiv, abs/1907.06987, 2019.

David L Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1, pages 190–200. Association for Computational Linguistics, 2011.

Guo Chen, Yin-Dong Zheng, Limin Wang, and Tong Lu. Dcan: improving temporal action detection via dual context aggregation. In AAAI, volume 36, pages 248–257, 2022a.

Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024a.

Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Wanxiang Che, Xiangzhan Yu, and Furu Wei. BEATs: Audio pre-training with acoustic tokenizers. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 5178–5193. PMLR, 23–29 Jul 2023a.

Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audiolanguage omni-perception pretraining model and dataset. arXiv preprint arXiv:2304.08345, 2023b.

Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audiosubtitle-text omni-modality foundation model and dataset. NeurIPS, 36, 2024b.

Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. arXiv preprint arXiv:2205.08534, 2022b.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023c.

Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In NeurIPS, 2021.

Feng Cheng, Xizi Wang, Jie Lei, David J. Crandall, Mohit Bansal, and Gedas Bertasius. Vindlu: A recipe for effective video-and-language pretraining. ArXiv, abs/2212.05051, 2022.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR, pages 2818–2829, 2023.

Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, PaulAmbroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye, Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-jussà, Onur Celebi, Maha Elbayad, Cynthia Gao, Francisco Guzmán, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, and Skyler Wang. Seamlessm4t: Massively multilingual & multimodal machine translation, 2023.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2023.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. NeurIPS, 35:16344–16359, 2022.

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim M. Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin F. Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Collier, Alexey A. Gritsenko, Vighnesh Birodkar, Cristina Nader Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Paveti’c, Dustin Tran, Thomas Kipf, Mario Luvci’c, Xiaohua Zhai, Daniel Keysers, Jeremiah Harmsen, and Neil Houlsby. Scaling vision transformers to 22 billion parameters. In ICML, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. ArXiv, abs/1810.04805, 2018.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. ArXiv, abs/2010.11929, 2020.

Danny Driess, F. Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Ho Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Peter R. Florence. Palm-e: An embodied multimodal language model. In ICML, 2023.

Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In ICASSP, pages 736–740. IEEE, 2020.

Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, 2019.

Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, and Kaiming He. Masked autoencoders as spatiotemporal learners. NeurIPS, 2022.

Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. Violet: End-to-end video-language transformers with masked visual-token modeling. arXiv preprint arXiv:2111.12681, 2021.

Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval. In ECCV, pages 214–229. Springer, 2020.

J. Gao, Chen Sun, Zhenheng Yang, and Ramakant Nevatia. Tall: Temporal activity localization via language query. In ICCV, 2017.

Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qianmengke Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. ArXiv, abs/2305.04790, 2023.

Yuan Gong, Yu-An Chung, and James Glass. Ast: Audio spectrogram transformer. arXiv preprint arXiv:2104.01778, 2021.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fründ, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The “something something” video database for learning and evaluating visual common sense. In ICCV, 2017a.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017b.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh K. Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Z. Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Christian Fuegen, Abrham Gebreselasie, Cristina González, James M. Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jáchym Kolár, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran K. Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Yunyi Zhu, Pablo Arbeláez, David J. Crandall, Dima Damen, Giovanni Maria Farinella, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard A. Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, and Jitendra Malik. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, 2022.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022.

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR, 2015.

J. Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2021.

Haroon Idrees, Amir R Zamir, Yu-Gang Jiang, Alex Gorban, Ivan Laptev, Rahul Sukthankar, and Mubarak Shah. The thumos challenge on action recognition for videos “in the wild”. CVIU, 2017.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016.

Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 119–132, 2019.

Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In ICCV, 2017.

Hildegard Kuehne, Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In ICCV, pages 2556–2563. IEEE, 2011.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L. Berg. Tvqa: Localized, compositional video question answering. In EMNLP, 2018.

Jie Lei, Tamara L. Berg, and Mohit Bansal. Qvhighlights: Detecting moments and highlights in videos via natural language queries, 2021.

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023a.

Guangyao Li, Yixin Xu, and Di Hu. Multi-scale attention for audio question answering. arXiv preprint arXiv:2305.17993, 2023b.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2022a.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022b.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, and Yu Qiao. Uniformerv2: Spatiotemporal learning by arming image vits with video uniformer. arXiv preprint arXiv:2211.09552, 2022c.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023c.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al.

Mvbench: A comprehensive multi-modal video understanding benchmark. arXiv preprint arXiv:2311.17005, 2023d. Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards

training-efficient video foundation models. arXiv preprint arXiv:2303.16058, 2023e. Tianhao Li and Limin Wang. Learning spatiotemporal features via video and text pair discrimination. CoRR, abs/2001.05691, 2020. URL https://arxiv.org/abs/2001.05691. Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and

Mike Zheng Shou. Univtg: Towards unified video-language temporal grounding. In ICCV, pages 2794–2804, 2023. Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, and Shilei Wen. Bmn: Boundary-matching network for temporal action

proposal generation, 2019. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. Samuel Lipping, Parthasaarathy Sudarsanam, Konstantinos Drossos, and Tuomas Virtanen. Clotho-aqa: A crowdsourced

dataset for audio question answering. In EUSIPCO, pages 1140–1144. IEEE, 2022. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved

reasoning, ocr, and world knowledge, 2024. Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot. Ntu rgb+d 120: A large-scale benchmark for 3d human activity understanding. TPAMI, 2020. Yi Liu, Limin Wang, Yali Wang, Xiao Ma, and Yu Qiao. Fineaction: A fine-grained video dataset for temporal action localization. TIP, 31:6937–6950, 2022. Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 2022. Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Ming-Hui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. ArXiv, abs/2306.07207, 2023. Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. ArXiv, abs/2306.05424, 2023.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, NeurIPS, pages 46212–46244, 2023.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, 2019.

Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D. Plumbley, Yuexian Zou, and Wenwu Wang. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research, 2023.

Mathew Monfort, Bolei Zhou, Sarah Adel Bargal, Alex Andonian, Tom Yan, Kandan Ramakrishnan, Lisa M. Brown, Quanfu Fan, Dan Gutfreund, Carl Vondrick, and Aude Oliva. Moments in time dataset: One million videos for event understanding. TPAMI, 2020.

WonJun Moon, Sangeek Hyun, SuBeen Lee, and Jae-Pil Heo. Correlation-guided query-dependency calibration in

video representation learning for temporal grounding. arXiv preprint arXiv:2311.08835, 2023. OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023a. OpenAI. Gpt-4v(ision) system card. https://api.semanticscholar.org/CorpusID:263218031, 2023b. Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. In NeurIPS, 2024.

Karol J Piczak. Esc: Dataset for environmental sound classification. In Proceedings of the 23rd ACM international conference on Multimedia, pages 1015–1018, 2015.

Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV, 2015.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In ICML, pages 28492–28518. PMLR, 2023.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020.

Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In CVPR, pages 3202–3212, 2015.

Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, Jitendra Malik, Yanghao Li, and Christoph Feichtenhofer. Hiera: A hierarchical vision transformer without the bells-and-whistles, 2023.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. arXiv preprint arXiv:2312.13286, 2023a.

Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023b.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023c.

Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, and Xinlong Wang. Eva-clip-18b: Scaling clip to 18 billion parameters. arXiv preprint arXiv:2402.04252, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. NeurIPS, 35:10078–10093, 2022.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288, 2023b.

Du Tran, Hong xiu Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In CVPR, 2018.

Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Kevin Qinghong Lin, Satoshi Tsutsui, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, et al. All in one: Exploring unified video-language pre-training. In CVPR, pages 6598–6608, 2023a.

Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In ECCV, 2016.

Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In CVPR, 2023b.

Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, and Yu-Gang Jiang. Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning. In CVPR, pages 6312–6322, 2023c.

Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In ICCV, 2019.

Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023d.

Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. mplug-2: A modularized multi-modal foundation model across text, image and video. arXiv preprint arXiv:2302.00402, 2023.

Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, pages 5288–5296, 2016.

Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Video-text modeling

with zero-shot transfer from contrastive captioners. ArXiv, abs/2212.04979, 2022. Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In ICCV, 2019. Min Yang, Guo Chen, Yin-Dong Zheng, Tong Lu, and Limin Wang. Basictad: an astounding rgb-only baseline for

temporal action detection. CVIU, page 103692, 2023.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality, 2023.

Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022.

Éloi Zablocki, Hédi Ben-Younes, Patrick Pérez, and Matthieu Cord. Explainability of deep vision-based autonomous driving systems: Review and challenges. IJCV, 130(10):2425–2452, 2022.

Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. In CVPR, pages 16375–16387, 2022.

Ziyun Zeng, Yixiao Ge, Zhan Tong, Xihui Liu, Shu-Tao Xia, and Ying Shan. Tvtsv2: Learning out-of-the-box

spatiotemporal visual representations at scale. arXiv preprint arXiv:2305.14173, 2023. Biao Zhang and Rico Sennrich. Root mean square layer normalization. NeurIPS, 32, 2019. Bowen Zhang, Jiahui Yu, Christopher Fifty, Wei Han, Andrew M. Dai, Ruoming Pang, and Fei Sha. Co-training

transformer with videos and images improves action recognition. ArXiv, abs/2112.07175, 2021.

Hongjie Zhang, Yi Liu, Lu Dong, Yifei Huang, Zhen-Hua Ling, Yali Wang, Limin Wang, and Yu Qiao. Movqa: A benchmark of versatile question-answering for long-form movie understanding. arXiv preprint arXiv:2312.04817, 2023.

Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and segments dataset for recognition and temporal localization. In ICCV, 2019.

Long Zhao, Nitesh B. Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, Jennifer J. Sun, Luke Friedman, Rui Qian, Tobias Weyand, Yue Zhao, Rachel Hornung, Florian Schroff, Ming-Hsuan Yang, David A. Ross, Huisheng Wang, Hartwig Adam, Mikhail Sirotenko, Ting Liu, and Boqing Gong. Videoprism: A foundational visual encoder for video understanding, 2024.

Yanpeng Zhao, Jack Hessel, Youngjae Yu, Ximing Lu, Rowan Zellers, and Yejin Choi. Connecting the dots between audio and text without parallel data through visual knowledge transfer. arXiv preprint arXiv:2112.08995, 2021.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023a.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. ArXiv, abs/2304.10592, 2023b.

Wentao Zhu, Yufang Huang, Xiufeng Xie, Wenxian Liu, Jincan Deng, Debing Zhang, Zhangyang Wang, and Ji Liu. Autoshot: A short video dataset and state-of-the-art shot boundary detection. In CVPR, pages 2237–2246, 2023c.

##### A Model

Table 20: Architecture of vision encoder (6B).

|Stage|ViT-6B<br><br>|Output Size|
|---|---|---|
|Video|sparse sampling|3×8×224×224|
|Patch Embedding|1×14×14, 3200 stride 1×14×14<br><br>|3200×8×256<br><br>|
|Position Embedding|learnable, 3D sine-cosine initialization 3200×2048<br><br>|3200×2048|
|Mask|semantic mask w/ mask ratio = ρ<br><br>|3200×2048·(1-ρ)|
|Encoder<br><br>|MHSA(3200) MLP(12800) ×48 + AttnPool(768)<br><br>|3200×2048·(1-ρ) 768×1|
|Projection<br><br>|LN(3200) MLP(3200) ×KCLIP,<br><br>LN(3200) MLP(1408) ×KMAE,<br><br>LN(3200) MLP(768) ×1<br><br>|K×3200×2048·(1-ρ) K×1408×2048·(1-ρ) K×768×1|

Video Encoder. In Tab. 20, we take ViT-6B as an example and omit the class token for a simple presentation. “MHSA”, “MLP”, “AttnPool”, and “LN” refer to spatiotemporal multi-head self-attention, multi-layer perception,

attention pooling [Yu et al., 2022] and root mean square layer normalization [Zhang and Sennrich, 2019]. KCLIP and KMAE means the layer number for unmasked token alignment with multimodal and motion-aware teachers. We mark the channel number, frame number, spatial size, and token number by different colors. The projection layers are dropped after stage 1 training.

##### B Video-centric Multimodal Data

We prepare our training data according to the learning objectives of the three stages in our learning scheme. Specifically, it consists of video-only pretraining set for masked video token reconstruction, Video-Audio-Speech-Text one for multimodal alignment, and video instruction dataset for human-computer interaction alignment. They are detailed in the following.

###### B.1 Video-only Data

To create the curated collection of videos, named K-Mash, we source videos from renowned action recognition datasets such as Kinetics-400 (K400) [Carreira and Zisserman, 2017], Something-Something (Sth) [Goyal et al., 2017a], Moments in Time (MIT) [Monfort et al., 2020], ActivityNet [Heilbron et al., 2015], and HACS [Zhao et al., 2019]. These datasets provide a wide range of video types, including both first-person and third-person perspectives, short and long durations, and featuring a rich variety of characters and settings.

For the enhanced version of the dataset, called K-Mash2M, we push a step further and meticulously selected an additional 844,000 videos from YouTube to further enhance the diversity of the dataset. It’s important to note that all videos in this dataset are utilized for training without any labels, allowing the model to learn from the unlabeled data in an unsupervised manner. This approach helps to broaden the model’s understanding of different visual concepts and improves its performance on various video-related tasks.

###### B.2 Videos with Audio-Video-Speech Modalities

In addtion to publicly available video-text datasets (e.g. InternVid [Wang et al., 2023d] and WebVid [Bain et al., 2021]), we introduce a new video dataset that incorporates both audio-visual-speech information and their corresponding textual descriptions. This dataset is included in the training process of InternVideo2. To create this multimodal dataset, named InternVid2, we leverage several video sources and provide detailed annotations. InternVid2 includes videos with synchronized audio, visual, and speech information, along with their corresponding textual descriptions. This multimodal dataset enables the training of InternVideo2 to better understand and capture the connections between different modalities, enhancing its performance in various video-related tasks that require audio, visual, and speech understanding.

Collection. In the InternVid2 dataset, approximately half of the videos are sourced from YouTube, while the remaining videos are gathered from anonymous sources. This is to improve the diversity of dataset, as relying solely on YouTube may result in limited depth of the dataset. Furthermore, to study the impact of video culture backgrounds on the learned models, a small portion of the dataset consists of Chinese data. These videos were collected with proper permissions for academic usage, ensuring compliance with legal and ethical considerations.

Table 21: Statistics of Stage1 data. All the videos are used without any labels.

|Dataset<br><br>|K710|SthSthV2|HACS<br><br>|ANet<br><br>|MiT<br><br>|Self-collected|
|---|---|---|---|---|---|---|
|K-Mash1.1M K-Mash2M|658K 658K<br><br>|169K 169K<br><br>|106K 106K<br><br>|15K 15K|152K 207K<br><br>|0 844K<br><br>|

By incorporating videos from various sources and including a subset of Chinese data, InternVid2 provides a more diverse and representative dataset for training InternVideo2. This approach allows the model to learn from a wide range of video content, encompassing different cultural backgrounds and further enhancing its ability to understand and process videos from various sources.

Trimming. In our approach, instead of relying on the widely-used SceneDet filter of FFMPEG, we employ a temporal boundary detection model called AutoShot [Zhu et al., 2023c] to segment videos into clips. AutoShot is capable of predicting clip boundaries based on temporal semantic variations, as opposed to pixel differences. This leads to the generation of semantically complete cuts without mixing extra frames that may contain inconsistent context. By using AutoShot, we aim to reduce captioning errors by producing fewer clips with obvious transitions, resulting in a more coherent input for video captioning models. In the inference of AutoShot, we use a threshold of 0.5 to determine the shot boundaries for AutoShot’s estimations.

For the video dataset, we first preserve clips longer than 2 seconds. For video clips longer than 30 seconds, as the segments within the clip are from the same shot, we randomly choose a 30-second segment. During this process, we also abandon clips with still or extreme dynamics, such as browsing a photo gallery.

Table 22: Fusion prompt.The above prompt is to generate 2 multi-modal captions, the following prompt is to generate

- 3 multi-modal captions.

You are a text analysis expert. About one video, here is 1 vision caption: vid_cap, 1 audio caption:aud_cap. You need to understand and encode them into 1 sentence. Do not simply concatenate them together. The weights of video/audio are equaled. Considering dropping audio caption if it is incomprehensible. The output must be a complete and natural sentence. The sentence is: ...

You are a text analysis expert. About one video, here is 1 vision caption: vid_cap, 1 audio caption:aud_cap, and one speech subtitle: asr_cap,. You need to understand and encode them into 1 compelete sentence. The weights of video/audio/speech are equaled. Considering dropping audio caption or speech subtitle if it is incomprehensible. The output must be a complete and natural sentence, do not simply concatenate them together. The complete sentence is:

Annotation. We automatically caption visual, audio, and speech of InternVid2. Then we correct them and fuse them for cross-modal captions for training using LLM. We list several annotation examples of our method in Fig. 10.

- • Vision Captioner. We employ the video captioning pipeline in InternVid to annotate our data. Rather than using VideoLLM [Li et al., 2023c, Maaz et al., 2023] to describe videos, we choose this validated method due to its better downstream results.
- • Audio Captioner. We craft a audio captioner upon VideoChat [Li et al., 2023c], as we find no reliable ones. It extracts audio features from inputs by Beats [Chen et al., 2023a]. We learn it by only tuning its Qformer (the interface between audio encoder and LLM) using a combination of the large-scale audio-text corpus WavCaps [Mei et al., 2023] dataset.
- • Speech Captioner. We utilize the audio transcription model Whisper [Radford et al., 2023] to obtain speech from videos.Concretely, we use the WhisperV2-large model since its concurrent state-of-the-art performance. During the data collection process, a portion of the data is directly adopted from YT-Temporal-180M, which already has well-aligned timestamps and adjusted speech recognition content. The remaining data is first passed through a pre-trained language identification model Fasttext-lid [Joulin et al., 2016] to determine the language category, and then transcribed the non-English text into English using the pretrained Seamless M4T [Communication et al., 2023] model. For the text with language identification confidence less than 0.95, we use Vicuna-1.5 as a translation alternative.
- • Caption Triming & Fusion with LLM. After obtaining the captions of audio, video, and speech from the given video, we utilize an LLM (Vicuna-1.5 [Zheng et al., 2023]) to integrate the uni-modal captions into the multimodal ones. To fulfill the request for multiple contrastive objectives, we combine the audio caption with the video caption

[Figure 84]

###### InternVideo2: Scaling Foundation Models for Multimodal Video Understanding

#### ECCV Supp. Case1

|[Figure 85]|
|---|

|Video Cap: A woman is talking to a reporter in a store.|
|---|

|Video-Audio-Speech Cap:<br><br>In the store, music plays in the background as a woman<br><br>excitedly and nervously tells a reporter about Carolyn, who is going to be their second competitor and has lost 69 pounds and 31% of her body weight.|
|---|

|Audio Cap: A woman is talking and<br><br>music plays in the background.|
|---|

[Figure 86]

|ASR: She is gonna be our second competitor. Take a lap. She has lost 69 pounds and thirty-one percent of her body weight. Hey I'm excited but nervous but excited Carolyn.|
|---|

|Video-Audio Cap: In a store, a woman talks to a reporter while music are heard in the background.|
|---|

[Figure 87]

|[Figure 88]|
|---|

|Video Cap: A man in a red shirt is standing in front of a lion.|
|---|

|Video-Audio-Speech Cap:<br><br>The man in the red shirt is attempting to barricade himself away from a wild lion in a room with most of<br><br>the line already there while a man is talking and the<br><br>radio plays music.|
|---|

|Audio Cap: A man is talking while a radio plays music.|
|---|

|ASR: Scene consisted of Denver attempting to barricade himself away<br><br>from a wild lion spreading to most of<br><br>the line is already in the room with him unfortunately.|
|---|

|Video-Audio Cap: The man in the red shirt stands in front of the lion and talking while the radio plays music.|
|---|

15

|[Figure 89]|
|---|

|Video Cap: A group of people are standing<br><br>in a field under a starry night sky.|
|---|

|Video-Audio-Speech Cap:<br><br>A group of people are standing in a field under a starry night sky and a woman is talking in the background that a group of people observe the moon through a telescope, amazed by its craters.|
|---|

|Audio Cap: A woman is talking.|
|---|

[Figure 90]

|ASR: When people get a chance look<br><br>through a big telescope for the first time<br><br>they're just floored at what you can see I was just showing the moon to someone and they're like oh my god you really can't see the craters it's really interesting.|
|---|

16

|Video-Audio Cap: A group of people are standing in a field under a starry night sky and a woman is talking in the background.|
|---|

|[Figure 91]|
|---|

|Video Cap: A small black fish is<br><br>swimming in a pond|
|---|

|Video-Audio-Speech Cap: While a bird is chirping in the background, a small black<br><br>fish is swimming in a pond and a man is saying 'Come<br><br>on, dude'.|
|---|

17

|Audio Cap: A bird is chirping while a<br><br>man is talking.|
|---|

|Video-Audio Cap: As a small black fish swims in a pond, a bird chirps and a man talks in the background.|
|---|

|ASR: Come on, dude|
|---|

Figure 10: Annotation examples using our captioning approach.

as the audio-visual caption, as well as integrate the audio, video, and speech captions as the audio-visual-subtitle captions. In this way, we acquire 5 types of captions (3 uni-modal captions (A, V, S) and 2 multi-modal captions (AV, AVS)) for each video automatically. Specifically, we have carefully designed prompt templates (Fig. 22) and employed vLLM [Kwon et al., 2023] for inference acceleration, effectively get the visual caption, audio caption, subtitle, audio-visual caption and audio-visual-speech caption while maintaining a natural human-like subtitle style.

18

Filtering & Sampling. After obtaining the captions, we calculate the CLIP similarity between the video segments and captions. We select the top 60 million data as the video segment data for InternVid2. For LAION-2B, we only select samples with CLIP similarity in the top 158 million for training.

##### C Experiments

- C.1 Ablations C.1.1 How InternVideo2 Works in Feature-based Tasks.

We study which part of InternVideo2s1’s predictions are suitable for feature-based tasks, i.e. temporal action localition. We adhere the same train and test protocols as in the main paper.

In Tab. C.1.1, the most effective feature tends to be located within the last few layers of the video encoder. This observation aligns with the similarity between feature-based temporal tasks and linear probing classification, which is reasonable. We undertake comprehensive experiments to investigate the impact of features from various layers, as detailed in Table 23. The results reveal the best features appear between the last 5-th layer and 7-th layer.

Table 23: Effect of feature extracted from the last 7 layers.

Layer Index THUMOS-14 ActivityNet HACS Segment FineAction

1B@mAP 6B@mAP 1B@mAP 6B@mAP 1B@mAP 6B@mAP 1B@mAP 6B@mAP

- -1 67.9 70.3 39.0 40.7 39.5 42.1 25.4 25.3

- -2 68.4 71.0 39.3 40.5 41.0 42.7 26.2 26.4

- -3 69.0 71.3 39.7 40.5 41.2 42.7 27.0 26.6

- -4 69.3 71.4 39.6 41.1 41.3 43.1 27.1 27.1

- -5 69.9 71.8 39.7 40.9 41.4 43.1 27.2 27.7

- -6 69.6 72.0 39.6 41.2 40.7 43.3 27.0 27.7

- -7 69.5 71.9 40.0 41.1 40.6 42.8 26.9 27.7

- Table 24: Video retrieval results on MSR-VTT, DiDeMo, LSMDC, ActivityNet, VATEX, and MSVD. We report R@1, R@5, and R@10. #F denotes input frame number in eval.

(a) MSR-VTT

Method #F

Text-to-Video Video-to-Text R@1 R@5 R@10 R@1 R@5 R@10

InternVideo2s2-1B 4 51.9 74.6 81.7 49.6 73.6 81.2 InternVideo2s2-1B 8 51.9 75.3 82.5 50.9 73.4 81.8 InternVideo2s2-6B 4 54.5 77.5 83.7 52.3 75.3 83.5 InternVideo2s2-6B 8 55.9 78.3 85.1 53.7 77.5 84.1

(b) LSMDC

Method #F

Text-to-Video Video-to-Text R@1 R@5 R@10 R@1 R@5 R@10

InternVideo2s2-1B 4 31.5 51.3 59.5 27.1 44.8 51.8 InternVideo2s2-1B 8 32.0 52.4 59.4 27.3 44.2 51.6 InternVideo2s2-6B 4 34.8 54.0 61.6 30.1 48.0 55.0 InternVideo2s2-6B 8 33.8 55.9 62.2 30.1 47.7 54.8

(c) VATEX

Method #F

Text-to-Video Video-to-Text R@1 R@5 R@10 R@1 R@5 R@10

InternVideo2s2-1B 4 70.7 93.7 96.9 85.9 97.6 99.2 InternVideo2s2-1B 8 70.4 93.4 96.9 85.4 97.6 99.1 InternVideo2s2-6B 4 71.1 93.8 97.0 85.2 97.7 99.4 InternVideo2s2-6B 8 71.5 94.0 97.1 85.3 97.9 99.3

(d) DiDeMo

Method #F

Text-to-Video Video-to-Text R@1 R@5 R@10 R@1 R@5 R@10 InternVideo2s2-1B 4 56.7 78.7 83.9 54.4 74.4 80.6

- InternVideo2s2-1B 8 57.0 80.0 85.1 54.3 77.2 83.5 InternVideo2s2-6B 4 56.2 77.6 83.6 53.2 76.8 82.7 InternVideo2s2-6B 8 57.9 80.0 84.6 57.1 79.9 85.0

(e) ActivityNet

Method #F

Text-to-Video Video-to-Text R@1 R@5 R@10 R@1 R@5 R@10

InternVideo2s2-1B 4 56.9 81.7 89.8 53.6 80.0 88.5 InternVideo2s2-1B 8 60.4 83.9 90.8 54.8 81.5 89.5 InternVideo2s2-6B 4 59.4 83.2 90.3 53.7 80.5 88.9 InternVideo2s2-6B 8 63.2 85.6 92.5 56.5 82.8 90.3

(f) MSVD

Method #F

Text-to-Video Video-to-Text R@1 R@5 R@10 R@1 R@5 R@10 InternVideo2s2-1B 4 58.9 83.0 88.7 83.6 94.8 97.0

- InternVideo2s2-1B 8 58.1 83.0 88.4 83.3 94.3 96.9 InternVideo2s2-6B 4 59.8 84.2 89.7 82.5 94.6 97.2 InternVideo2s2-6B 8 59.3 84.4 89.6 83.1 94.2 97.0

C.2 Video Retrieval

We detail R@1, R@5, and R@10 of zero-shot video retrieval from InternVideo2 in Tab. 24a -24f for reference.

- Table 25: The top-1 accuracy of zero-shot video QA (multi-choice) on MSR-VTT and LSMDC. Finetuned results are marked in gray.

Method MSR-VTT LSMDC

VIOLET [Fu et al., 2021] 91.9 82.8 InternVideo [Wang et al., 2022] 93.4 77.3 InternVideo2s2-6B 94.4 76.9

Table 26: Performance of multimodal LLMs on different question types and scenes of MoVQA.

Method Backbone Synopsis Temporal Spatial Causal Hypothetical Knowledge Single-Scene Multi-Scene Full-Scene Overall Mplug-Owl [Ye et al., 2023] CLIP ViT-L 25.1 19.9 25.3 21.9 23.5 27.5 25.2 23.5 22.1 24.8

Otter [Li et al., 2023a] CLIP ViT-L 22.6 20.7 19.6 26.1 24.2 21.8 23.1 22.1 21.3 22.6 VideoChatGPT [Maaz et al., 2023] CLIP ViT-L 23.8 20.2 22.1 22.1 21.4 24.1 23.4 22.7 22.3 22.9

VideoChat [Li et al., 2023c] Eva-g 33.6 24.3 34.5 36.6 35.5 32.0 35.3 32.9 33.3 34.7 VideoChat2 [Li et al., 2023d] InternVideo2s3 42.6 27.8 39.9 44.3 442.5 41.2 40.9 39.3 38.6 40.1

###### C.3 Multi-Choice Video Question Answering

We evaluate zero-shot multi-choice (MC) video QA using InternVideo2 in stage 2 on MSR-VTT and VATEX. Tab. 25 shows InternVideo2 consistently improves MC accuracy compared with previous SOTAs except on LSMDC, where it gets a comparable result with InternVideo.

###### C.4 Movie Understanding

We evaluate InternVideo2 on the MoVQA for movie understanding. It is a long-form movie question-answering dataset[Zhang et al., 2023]. MoVQA assesses the diverse cognitive capabilities of multimodal systems by considering both video length and clue length, relying on multi-level temporal lengths (single-scene, multi-scene and full-scene). There are six types of QAs, including information synopsis, temporal perception, spatial perception, causal reasoning, hypothetical reasoning and external knowledge. We evaluate InternVideo2-6B in the form of open-ended QAs, and detailed results are shown on Table 26.

