# arXiv:2412.01820v3[cs.CV]24Mar2025

## Towards Universal Soccer Video Understanding

Jiayuan Rao1,2∗, Haoning Wu1,2∗, Hao Jiang3, Ya Zhang1, Yanfeng Wang1†, Weidi Xie1† 1School of Artificial Intelligence, Shanghai Jiao Tong University, China 2CMIC, Shanghai Jiao Tong University, China 3Alibaba Group, China https://jyrao.github.io/UniSoccer/

[Figure 1]

[Figure 2]

[Figure 3]

Videos in SoccerReplay-1988

### Abstract

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

As a globally celebrated sport, soccer has attracted widespread interest from fans all over the world. This paper aims to develop a comprehensive multi-modal framework for soccer video understanding. Specifically, we make the following contributions in this paper: (i) we introduce SoccerReplay-1988, the largest multi-modal soccer dataset to date, featuring videos and detailed annotations from 1,988 complete matches, with an automated annotation pipeline; (ii) we present an advanced soccer-specific visual encoder, MatchVision, which leverages spatiotemporal information across soccer videos and excels in various downstream tasks; (iii) we conduct extensive experiments and ablation studies on event classification, commentary generation, and multi-view foul recognition. MatchVision demonstrates state-of-the-art performance on all of them, substantially outperforming existing models, which highlights the superiority of our proposed data and model. We believe that this work will offer a standard paradigm for sports understanding research.

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

Classification

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

“Shot Saved”

[Figure 35]

[Figure 36]

Commentary

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

MatchVision

“[PLAYER] ([TEAM]) shakes off the attention of a defender and from mid-range lets fly towards the middle of the goal, but [PLAYER] is well positioned to make a comfortable save.”

[Figure 41]

Figure 1. Overview. We present SoccerReplay-1988, the largest soccer dataset to date, and a powerful soccer-specific visual encoder, MatchVision, capable of excelling in various tasks such as event classification and commentary generation.

fan engagement through interactive and personalized content [35, 37, 41]. These capabilities promote a deeper understanding of soccer, simplify content creation, and foster a more engaging experience for fans and professionals.

“Football is one of the world’s best means of communication. It is impartial, apolitical, and universal.”

—— Franz Beckenbauer (1945 - 2024)

Existing research in soccer video analysis primarily revolves around the SoccerNet series datasets [7, 10, 15], which comprise 500 full-match videos for benchmarking various tasks, such as action spotting [10, 15] and commentary generation [35, 37, 41]. Despite this extensive coverage, the focus has predominantly been on designing specialized models for task-specific applications, leading to fragmented and incompatible solutions. Such fragmentation underscores the need for a unified framework capable of integrating diverse demands, enabling more holistic and scalable advancements in soccer video understanding.

### 1. Introduction

Soccer, celebrated worldwide for its significant commercial value, has recently seen great research interest in integrating artificial intelligence (AI) for soccer video understanding. This is primarily motivated by the sport’s complexity and the growing demand for enhanced analytics and improved viewing experiences. AI systems facilitate tactical analysis [51], allowing coaches to devise better strategies by uncovering patterns not apparent to the naked eye. In addition, it also supports automated content generation and enriches

In this paper, we introduce SoccerReplay-1988, the largest and most comprehensive multi-modal soccer video dataset to date, featuring 1,988 complete match videos with

*: These authors contribute equally to this work. †: Corresponding author.

rich annotations, such as event labels and textual commentaries. This dataset offers a solid foundation for developing advanced soccer understanding models and establishes a challenging new benchmark for the field. Additionally, we have harmonized existing datasets to be compatible with ours, further expanding the available data resources.

Leveraging this dataset, we develop MatchVision, an advanced soccer-specific visual encoder tailored for diverse soccer understanding tasks. It employs the cutting-edge visual-language foundation model as the backbone, e.g., SigLIP [60]. We further extend framewise visual features into spatiotemporal representations with temporal attentions [3], by training on diverse visual-language tasks on SoccerReplay-1988, as depicted in Figure 1. As a result, MatchVision exhibits strong adaptability across various tasks, such as event classification and commentary generation, serving as a universal and unified framework for comprehensive soccer video understanding.

To summarize, we make the following contributions in this paper: (i) we construct SoccerReplay-1988, the largest and most diverse soccer video dataset to date, featuring videos of 1,988 soccer matches with rich annotations, supported by an automated curation pipeline. This provides a solid foundation for developing robust and comprehensive soccer understanding models; (ii) we present a powerful soccer-specific visual encoder, termed MatchVision, which effectively leverages spatiotemporal information in soccer videos, and can adapt to various tasks such as event classification and commentary generation, serving as a unified framework for soccer understanding; (iii) we establish more comprehensive and challenging benchmarks based on our dataset, enabling more professional evaluation of soccer understanding models; (iv) extensive experiments and ablation studies demonstrate the superiority of our data and model across various downstream tasks, achieving state-ofthe-art performance on both existing benchmarks and our newly established ones. We believe this work offers a viable paradigm for future sports video understanding.

### 2. Related Works

Sports Understanding [46] is an evolving field that encompasses multiple research topics and integrates diverse data modalities, covering various tasks such as action spotting [10, 15, 16], commentary generation [35, 38, 41, 53, 54, 59], athlete analysis [42, 57], tactical planning [51], sports health [40], and intelligent refereeing [22, 23]. Furthermore, with the rapid development of multimodal large language models (MLLMs), recent efforts [27, 52, 55, 56] have attempted to build more generalized frameworks to uniformly handle a variety of sports understanding tasks.

Visual-Language Models [1, 28, 29, 39, 60] have exhibited remarkable performance across extensive applications like classification, segmentation, image-text retrieval, and image

captioning. Recent efforts have ventured into more challenging video understanding [30, 31, 43, 61, 62] tasks, such as temporal alignment [17, 32], dense captioning [5, 58, 64], and audio description [18–20]. However, these efforts typically focus on general scenarios, limiting their adaptability to specific professional fields. Thus, this paper aims to bridge this gap by advancing visual-language models tailored for comprehensive soccer understanding.

Soccer Game Analysis [9] has primarily focused on tasks such as action spotting [15], replay grounding [10, 63], commentary generation [35, 37, 41], player tracking [8], state reconstruction [44], camera calibration [6, 8, 10] and foul recognition [22, 23], as facilitated by the SoccerNet [7, 10, 14, 15] series datasets, with 500 full-match videos from 2015 to 2017. Unlike existing methods that target designing specific models for distinct tasks, this paper aims to design a unified multi-modal framework that leverages spatiotemporal information within videos, serving as a specialized visual encoder for soccer video understanding.

### 3. SoccerReplay-1988 Dataset

To establish a solid foundation for soccer understanding, we construct SoccerReplay-1988, the largest soccer dataset to date. Here, we first outline our data collection details and an overview of the dataset in Sec. 3.1; followed by elaborating on our automated data curation pipeline in Sec. 3.2; lastly, in Sec. 3.3, we present the data statistics and discussion.

#### 3.1. Dataset Collection

To construct the SoccerReplay-1988 dataset, we have collected untrimmed, full-match videos from the Internet, encompassing a total of 1,988 matches from six European major soccer leagues and championships1, spanning the 201415 to 2023-24 seasons. For each match, we acquire textual commentaries with second-level timestamps from a sports text live website2, with part of them annotated with specific event types such as corner and goal. Additionally, we also incorporate extensive metadata, including detailed background information about the games, players, coaches, referees, and teams, providing a solid foundation for future soccer understanding research.

We partition the SoccerReplay-1988 dataset into train, validation, and test sets, containing 1,488, 250, and 250 full-match videos with diverse and comprehensive annotations, respectively. These sets provide rich training data for downstream tasks, such as event classification and commentary generation, while establishing comprehensive and challenging benchmarks for soccer understanding, as further discussed in subsequent sections.

1Premier (England), Laliga (Spain), Bundesliga (Germany), Serie-a

(Italy), League-1 (France) and UEFA Champions League. 2flashscore.com

[Figure 42]

[Figure 43]

Existing Datasets # Game Duration(h) # Event # Anno. # Com.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

SoccerNet-v1 [15] 500 764 7 6.7k SoccerNet-v2 [10] 500 764 17 110k MatchTime [41] 471 716 14 14k 37k GOAL [37] 20 25.5 - - 8.9k

[Figure 52]

###### 87:09 87:17

[Figure 53]

Crop from Kick-off moment

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

Our Curated Datasets # Game Duration(h) # Event # Anno. # Com. SoccerNet-pro 500 764 24 102k 37k

Dense Commentaries

###### 87:09 “Goal”

###### Temporal Alignment Event Summarization Anonymization Mask

Timestamp: 87:17 Commentary text:

SoccerReplay-1988 1,988 3,323 24 150k 150k Integrated 2,488 4,087 24 252k 187k

“Goal! Lei Wu (Espanyol) neatly controls a decent pass from Matias Vargas. He looks up and smashes an unstoppable drive into the bottom left corner from the edge of the box. 2:2.”

|“Goal! [PLAYER] ([TEAM]) neatly controls a decent pass from [PLAYER]. He looks up and smashes an unstoppable drive into the bottom left corner from the edge of the box. 2:2.”|
|---|

Table 1. Statistics of Soccer Datasets. Our SoccerReplay-1988 significantly surpasses existing datasets in both scale and diversity. Here, # Anno. and # Com. refer to the number of event annotations and textual commentaries, respectively.

- Figure 2. Automated Data Curation Pipeline. The collected soccer video data are automatically processed for temporal alignment, event summarization, and anonymization by our curation pipeline.

datasets available for soccer understanding tasks.

#### 3.3. Statistics & Discussion

#### 3.2. Automated Data Curation

Given the potential noise in raw data, such as irrelevant video content, inaccurate timestamps, and incomplete event annotations, we design an automated data curation pipeline, comprising (i) temporal alignment, (ii) event summarization, and (iii) anonymization, as illustrated in Figure 2.

Dataset Statistics. As shown in Table 1, our dataset encompasses 3,323 hours of footage from 1,988 soccer matches, with an average duration of 100.3 minutes per match. The videos range in resolution from 360p to 720p and frame rates between 25 and 30 FPS.

For textual annotations, this dataset features approximately 150K commentaries, averaging 76 per match, precisely temporal-aligned by the robust alignment model from MatchTime [41]. These commentaries cover 4,467 unique words, significantly surpassing the 2,873 words in existing datasets [35, 41], greatly enriching textual diversity. Automated event summarization based on these commentaries has yielded about 150K event annotations. Notably, a random sampling of 2% of the data yields 98% manual verification accuracy, ensuring high-quality automated labeling.

Temporal Alignment. Here, we divide match videos into two halves, each starting at kick-off, and adopt the temporal alignment model from MatchTime [41], to synchronize textual commentary timestamps with those of video frames. Event Summarization. For samples without event annotations, we leverage LLaMA-3-70B [12] to summarize the events based on textual commentaries. Concretely, we have expanded the event categories from 17 in SoccerNet [10] to 24 types, for finer-grained soccer understanding, for example, categorizing penalties into scored and missed, and integrating modern soccer regulations like VAR. The resulting 24 event labels include: ‘corner’, ‘goal’, ‘injury’, ‘own goal’, ‘penalty’, ‘penalty missed’, ‘red card’, ‘second yellow card’, ‘substitution’, ‘start of game (half)’, ‘end of game (half)’, ‘yellow card’, ‘throw in’, ‘free kick’, ‘saved by goal-keeper’, ‘shot off target’, ‘clearance’, ‘lead to corner’, ‘off-side’, ‘var’, ‘foul (no card)’, ‘statistics and summary’, ‘ball possession’, and ‘ball out of play’. More details on the used prompts are provided in the Appendix.

SoccerReplay-test Benchmark. To facilitate a more comprehensive evaluation of soccer understanding models, we integrate 250 matches from SoccerReplay-1988 with 50 matches from the curated SoccerNet-pro, establishing SoccerReplay-test, a more challenging benchmark for event classification and commentary generation. This benchmark features nearly four times larger than existing datasets and comprises finer-grained event labels, richer textual commentaries, and up-to-date soccer regulations.

Anonymization. Similar to [35], we extract all person and team entity names from the metadata of SoccerReplay1988, and replace them in textual commentaries with placeholders, such as “[PLAYER]”, “[TEAM]”, “[COACH]”, and “[REFEREE]”, ensuring consistency across tasks.

Discussion. To summarize, SoccerReplay-1988 exhibits advancements in three aspects: (i) it is the largest soccer video dataset to date, with nearly four times more videos than existing datasets; (ii) it features more professional and diverse annotations, more suitable for fine-grained and comprehensive soccer understanding tasks; (iii) It employs an automated curation pipeline for annotations and is thus scalable to provide a solid data foundation for future research. All data from SoccerReplay-1988, including videos

Moreover, our data curation pipeline can seamlessly extend to existing datasets, converting the SoccerNet series [10, 35] into our unified data format, termed SoccerNetpro. This expansion further enlarges the standardized

{ℱ }

…

|[Figure 64]|
|---|

Concat

|[Figure 65]|
|---|

|[Figure 66]|
|---|

… Prefix tokens [BOS][A][yellow][card][for] [make][this][decision]

Perceiver&MLP

…

|[Figure 67]|
|---|

Aggregation Layer

Φ

…

…

ℱ𝒱

ℱ

…

|…|
|---|

|…|
|---|

|…|
|---|

LLM Decoder

Feature Aggregator

SpatiotemporalAttention

Commentary:

×K

“A yellow card for a tackle by [PLAYER] ([TEAM]). [REFEREE] doesn‘t hesitate to make this decision.”

𝜙

Spatial Attention

Linear

Ψ

|Logits…|
|---|

ℱ𝒱

…

𝜙

Temporal Attention

…

Pooling

Linear

ℒ ℒ

…

𝐳

CLS

Severity

… … …

Ψ

|𝐄 …|
|---|

“Standing Tackling” “Yellow Card”

| | | | |
|---|---|---|---|
| | | | |
| | |[Figure 68]| |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
|[Figure 69]| | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
|[Figure 70]| | | |
| | | | |

Φ (𝐂)…

Concat

Concat

Concat

…

……

Logits

Text Encoder

One-hot

ArgMax

Ball out of Play

2.3%

𝐞 ,  𝐞 ,  𝐞 , 

CLS

CLS

CLS

21.5% 11.7%

2nd Yellow Card

Φ

Token Embedding

Offside

Commentary: Type: “Yellow Card”

Yellow Card Lead to Corner

55.4%

0.2%

[Figure 71]

[Figure 72]

[Figure 73]

“[REFEREE] shows a yellow card to [PLAYER] ([TEAM]), who isn't surprised as the tackle was really hard and late.”

[Figure 74]

[Figure 75]

[Figure 76]

Free Kick

8.9%

[Figure 77]

[Figure 78]

……

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

𝒱

…

Ψ

Type: “Yellow card”

(a) Structure of MatchVision (b) MatchVision Pretraining (c) Downstream Tasks

- Figure 3. Overview of MatchVision. (a) The model architecture and its spatiotemporal feature extraction process; (b) Details of visual encoder pretraining, including supervised classification and video-language contrastive learning; (c) Implementation details of specific heads for various downstream tasks, including commentary generation, foul recognition, and event classification. and annotations, are open-source for non-commercial use.
- 4. Method

Aggregation Layer, as depicted in Figure 3.

Token Embedding. In accordance with the convention in Vision Transformer [11], each frame (Ii) from the video segment (V = {I1,I2,··· ,IT}) is divided into M nonoverlapping patches of size P×P that span the entire frame.

In this paper, we aim to develop a soccer-specific visual encoder, MatchVision, tailored for diverse soccer video analysis tasks. We start by outlining our problem formulation in Sec. 4.1. Next, in Sec. 4.2, we detail the architecture of MatchVision. The training procedures are thoroughly discussed in Sec. 4.3. Finally, we describe the configurations for downstream tasks in Sec. 4.4, demonstrating the practical applications and effectiveness of our model.

These patches are flattened into vectors (xpi ), where p and i denote the spatial and temporal positions, respectively. Each vector is transformed via an embedding layer (ΦEmb) into a token vector of size R1×D, and then added with a spatial position embedding (eposs ∈ RM×D). Subsequently, we concatenate a [cls] token along with each frame. Finally, a temporal positional embedding (epost ∈ RT×D) is added across features of all frames, as formulated below:

#### 4.1. Problem Formulation

In this work, we tackle the challenge of analyzing soccer video segments, denoted as V ∈ RT×3×H×W. Our goal is to utilize the visual encoder (ΦMatchVision) to extract spatiotemporal features from these segments, which are then processed by multiple task-specific heads, formulated as:

yi = [xclsi , ΦEmb([x1i,··· ,xMi ]) + eposs ] z = [y1,··· ,yT] + epost

Here, [·,·] denotes concatenation, and yi ∈ R(M+1)×D represents the frame-wise features. The embedded features (z) will then serve as input for spatiotemporal attention blocks. Spatiotemporal Attention Block. Similar to TimeSformer [3], we utilize interleaved temporal and spatial attention to integrate spatiotemporal information in soccer videos. Concretely, each spatiotemporal attention block comprises a temporal self-attention layer and a spatial selfattention layer, i.e., ϕt(·) and ϕs(·), respectively.

E,C,F = Ψ(ΦMatchVision(V))

Here, Ψ = {Ψcls,ΨCmt,ΨFoul} represents the task-specific heads, with E, C, and F denoting the output event types, textual commentaries, and foul types, respectively. This unified framework effectively learns relevant spatiotemporal features, and enables seamless integration across various downstream tasks for comprehensive soccer understanding.

Given a video feature (z ∈ RT×(M+1)×D), we alternate temporal and spatial attention: temporal attention facilitates interactions among tokens at the same spatial positions across distinct frames, while spatial attention enables interactions among tokens within the same frame. Residual con-

#### 4.2. Architecture

MatchVision comprises three key components: (i) Token Embedding, (ii) Spatiotemporal Attention Block, and (iii)

nections are employed in each layer. After passing through a total of K spatiotemporal attention blocks, the resulting feature (F) captures both intra-frame and inter-frame relationships, i.e., F = [ϕs(ϕt(z))]K ∈ RT×(M+1)×D.

Aggregation Layer. To obtain video-level features, we employ an aggregation layer on the frame-wise spatiotemporal features. Specifically, for the i-th frame, we utilize spatial self-attention to aggregate information into its [cls] token, denoted as Fˆicls = ΦAgg(Fi). Concatenating the [cls] tokens of all frames yields the final video feature (FV), that effectively encapsulates spatiotemporal characteristics of soccer video segments, thus enabling it to be applicable for various downstream soccer understanding tasks. This process can be formulated as:

FV = ΦMatchVision(V) = [Fˆ1cls,··· ,FˆTcls] ∈ RT×D

#### 4.3. MatchVision Pretraining

In this part, we aim to pretrain the visual encoder with triplet samples ({V,E,C}), comprising videos, event labels, and textual commentaries. Concretely, we investigate two distinct pretraining strategies: supervised classification and video-language contrastive learning.

Supervised Classification. One way to pretrain the visual encoder is supervised learning on event classification. To be specific, the extracted visual features (FV) are aggregated by a temporal self-attention layer into a learnable [cls] token, denoted as FVcls, similar to the spatial-wise aggregation mentioned above. This token is then fed into a linear classifier, and trained with a cross-entropy loss for event classification. The objective is denoted as Lsup.

Video-Language Contrastive Learning. As an alternative, we can also pretrain our visual encoder with video-text contrastive learning. Specifically, we adopt simple average pooling on the video feature to get the aggregated visual feature (FVAvg), and encode the textual commentary (C) with a text encoder (ΦText). We train the model with sigmoid loss (Lsigmoid), as used in SigLIP [60]. Note that, some video clips may have highly similar commentaries, for example, ‘start of the game’, we treat the commentaries with high similarity in the same batch as positive samples when calculating loss functions. This can be expressed as follows:

Lcontra = Lsigmoid(FVAvg,ΦText(C))

#### 4.4. Downstream Tasks

After the pretraining mentioned above, MatchVision can now serve as a versatile visual encoder, to map the soccer video segments into visual features (FV), for training task-specific heads Ψ = {Ψcls,ΨCmt,ΨFoul} across different downstream tasks, including: (i) event classification, (ii) commentary generation, and (iii) foul recognition.

Event Classification. Similar to supervised classification above, we concatenate a learnable [cls] token to aggregate frame-wise visual features via temporal self-attention. This token is then fed into a linear classifier for event classification. The event classification head (Ψcls) is trained with a cross-entropy loss while freezing the visual encoder.

Commentary Generation. We follow the paradigm in MatchTime [41] to generate anonymized textual commentary for soccer video clips. Concretely, the commentary generation head (ΨCmt) employs a Perceiver [25] aggregator to consolidate visual features, which are then projected by a trainable MLP, serving as prefix embeddings for a large language model (LLM). Subsequently, an off-theshelf LLM decodes these embeddings into textual commentary. We adopt the negative log-likelihood loss, commonly used for auto-regressive next-token prediction.

Foul Recognition. As outlined in [22], the foul recognition task takes multi-view videos from the same scene as inputs, with each sample annotated with a foul class (8 types) and severity (4 levels). We encode these multi-view videos with MatchVision, and aggregate the extracted features into a single feature vector, via either max or average pooling, following the common practice. Subsequently, the foul recognition head (ΨFoul) employs a shared MLP and two task-specific linear classifiers, to predict foul type and severity, respectively. Similar to event classification, we use the combination of cross-entropy losses on the foul type and severity classification to jointly train ΨFoul.

Discussion. Pretraining MatchVision on large-scale soccer data equips it with substantial domain-specific knowledge, enabling it to serve as a universal visual encoder adaptable to various downstream soccer understanding tasks.

### 5. Experiments

This section begins with implementation details in Sec. 5.1; followed by quantitative evaluations across downstream tasks in Sec. 5.2; then, we conduct ablation studies on our SoccerReplay-test benchmark to analyze the effectiveness of the proposed dataset and model in Sec. 5.3; finally, we provide qualitative results for comparison in Sec. 5.4.

#### 5.1. Implementation Details

In our experiments, video segments are sampled at 1FPS around annotated timestamps, capturing a 30-second window for each sample. Frames are resized to 224×224 pixels as inputs. We initialize the embedding layer, spatial attention layers, aggregation layer, and text encoder of MatchVision with pretrained weights from SigLIP Base-16 [60] and adopt LLaMA-3 (8B) [12] as the off-the-shelf LLM decoder for commentary generation. All experiments are conducted on 4× Nvidia H800 GPUs with the AdamW [34] optimizer. Next, we elaborate on the training and evaluation details

Dataset Classification (%) Commentary SN MT SR Acc.@1 Acc.@3 Acc.@5 B@1 B@4 M R-L C

Visual Encoder

Off-the-shelf Models

I3D [4] ✗ ✗ ✗ 45.4 82.5 93.2 26.77 5.57 24.17 23.12 18.73 C3D [47] ✗ ✗ ✗ 47.8 85.1 95.0 28.13 6.64 24.52 24.23 27.88 ResNet [21] ✗ ✗ ✗ 47.2 84.6 94.4 27.34 6.57 24.72 24.43 27.29 CLIP [39] ✗ ✗ ✗ 48.5 85.5 95.2 26.25 6.51 24.27 24.75 28.17 InternVideo [50] ✗ ✗ ✗ 49.9 87.0 95.9 27.12 6.54 25.02 24.82 29.90 SigLIP [60] ✗ ✗ ✗ 50.2 86.7 95.6 27.85 6.98 25.16 25.03 31.38

Pretrain with Supervised Classification

Baidu [63] ✓ ✗ ✗ 56.4 91.9 97.3 31.20 8.88 26.56 26.61 38.93 SigLIP ✓ ✗ ✗ 55.9 89.6 94.9 28.51 7.39 25.96 25.94 35.71 SigLIP ✓ ✓ ✓ 57.9 91.7 97.5 30.95 8.56 25.79 26.17 38.24 MatchVision ✓ ✗ ✗ 82.5 96.6 98.8 29.45 7.92 26.01 26.21 36.15 MatchVision ✓ ✓ ✓ 84.0 97.3 99.2 31.05 9.06 26.94 27.93 42.20

Pretrain with Visual-Language Contrastive Learning

SigLIP ✗ ✓ ✗ 55.4 88.8 97.0 28.72 7.72 25.91 26.17 32.27 SigLIP ✗ ✓ ✓ 66.8 93.7 98.6 30.35 8.12 26.05 26.38 39.41 MatchVision ✗ ✓ ✗ 58.9 89.0 97.1 30.33 7.97 25.48 26.33 33.87 MatchVision ✗ ✓ ✓ 67.9 93.9 98.6 31.94 9.12 26.24 27.56 40.76

Pretrain with Hybrid Supervised-Contrastive Training

SigLIP ✓ ✓ ✗ 71.2 94.5 98.7 28.63 7.82 25.74 25.35 34.09 SigLIP ✓ ✓ ✓ 67.1 93.2 98.1 30.71 8.78 26.26 26.74 41.82 MatchVision ✓ ✓ ✗ 76.4 96.0 99.0 30.65 8.33 25.28 26.31 37.23 MatchVision ✓ ✓ ✓ 80.1 97.1 99.1 33.58 9.14 26.82 28.21 44.18

- Table 2. Quantitative Results on Event Classification and Commentary Generation. Here, SN, MT, and SR represent finetuning with curated SoccerNet-v2 [10], MatchTime [41], and SoccerReplay-1988, respectively. B, M, R-L, and C refer to BLEU, METEOR, ROUGEL, and CIDEr metrics, respectively. Within each unit, we denote the best performance in RED and the second-best performance in BLUE.

about visual encoder pretraining and downstream tasks.

Visual Encoder Pretraining. For both pretraining strategies, we use a batch size of 40 for 15 epochs. The learning rate for all randomly initialized modules, including the temporal attention layers, aggregator layer, and linear classifier, is set to 1×10−4. Meanwhile, the learning rate for modules initialized with pretrained parameters (including the text encoder) is set to 5 × 10−5. In contrastive training, we adopt a multi-positive strategy where each textual commentary, based on its event label, considers closely related categories (e.g. “start of game” and “offside”) as positive samples.

Downstream Tasks. In all downstream tasks, unless otherwise specified, we use the frozen visual encoder for feature extraction and only train the task-specific heads with a learning rate of 1 × 10−4 for 30 epochs. The batch sizes for event classification, commentary generation, and foul recognition are set to 40, 32, and 8, respectively. We adopt specific evaluation metrics for these three tasks: (i) For event classification, we use the top-1/3/5 classification accuracy; (ii) For commentary generation, we employ several commonly-used language evaluation metrics, including BLEU [36], METEOR [2], ROUGE-L [33], and

CIDEr [49]; (iii) For foul recognition, we follow the common practice, and report top-1/2 and top-1 accuracy for the foul type and severity classification, respectively.

Benchmarks & Baselines. To ensure fair and reliable comparisons with existing work, we evaluate event classification (24 types) on 100 matches from curated SoccerNetv2 [10] test set; commentary generation on 49 matches from SN-Caption-test-align benchmark manually aligned in [41]; and foul recognition on MVFoul [22]. We consider various baselines: for the first two tasks, this includes off-the-shelf general visual encoders such as ResNet [21], C3D [47], I3D [4], CLIP [39], SigLIP [60], and InternVideo [50], along with Baidu [63] and SigLIP finetuned with soccerspecific data. For foul recognition, we follow previous work [22, 23] and adopt ResNet [21], R(2+1)D [48], and MViT [13] jointly finetuned with classifiers, as baselines.

#### 5.2. Quantitative Evaluation

As depicted in Table 2, we draw two observations on event classification and commentary generation: (i) visual encoders trained on soccer data substantially outperform offthe-shelf general encoders (ResNet, C3D, I3D, CLIP, and

Visual Encoder Foul Class Severity Backbone Train Agg. Acc.@1 Acc.@2 Acc.@1

Mean 0.31 0.56 0.34

ResNet [21] ✓

Max 0.32 0.60 0.32 R(2+1)D [48] ✓

Mean 0.31 0.55 0.34

Max 0.32 0.56 0.39 MViT [13] ✓

Mean 0.40 0.65 0.38

Max 0.47 0.69 0.43 MatchVision ✗

Mean 0.44 0.53 0.58 Max 0.35 0.70 0.46

- Table 3. Quantitative Results on Multi-view Foul Recognition. Our frozen MatchVision encoder can achieve comparable performance with other jointly finetuned visual encoders.

InternVideo), underscoring the necessity of building specialized models for soccer understanding; (ii) almost all visual encoders, across all training settings, benefit from SoccerReplay-1988, emphasizing the value of constructing large-scale, high-quality data for soccer understanding. Next, we will delve into each task to discuss the results.

Event Classification. With identical training strategies and data, MatchVision considerably outperforms other methods in classification accuracy, demonstrating the superiority of its architecture, which effectively leverages spatiotemporal features within soccer videos. For example, MatchVision achieves a Top-1 accuracy of 82.5%, significantly surpassing SigLIP’s 55.9% under the same training conditions. Moreover, models trained via supervised classification excel others, primarily because the pre-training task shares the same objectives as the downstream event classification task. Commentary Generation. Visual encoders trained with visual-language contrastive learning exhibit better commentary generation performance than those trained with supervised classification, as this strategy better captures correlations between visual and textual features. Additionally, while MatchVision trained solely on SoccerNet slightly underperforms Baidu [63], incorporating SoccerReplay-1988 enables it to outperform on most metrics. This demonstrates that MatchVision can take advantage of large-scale datasets. Finally, a hybrid training approach, starting with supervised classification followed by visual-language contrastive learning, enables MatchVision to achieve optimal performance. This indicates that learning coarse-grained tasks such as classification provides a foundation for finegrained tasks like commentary generation, and fully leveraging data unlocks the potential of soccer understanding.

Foul Recognition. As demonstrated in Table 3, MatchVision achieves performance comparable to jointly finetuned state-of-the-art methods in foul recognition, even with a frozen visual encoder. This highlights that MatchVision effectively learns substantial knowledge from large-scale soc-

Pretrain Classification(%) Sup. Contra. SR Acc.@1 Acc.@3 Acc.@5

✓ ✗ ✗ 62.67 83.00 89.81 ✓ ✗ ✓ 68.03 86.90 92.38

- ✗ ✓ ✗ 46.97 75.53 85.85
- ✗ ✓ ✓ 57.41 83.13 91.00

✓ ✓ ✗ 56.86 80.30 88.09 ✓ ✓ ✓ 63.59 85.21 91.63

Table 4. Ablations on Event Classification. We explore the impact of various training settings of our MatchVision encoder on the SoccerReplay-test benchmark. Here, Sup., Contra., and SR refer to supervised classification, visual-language contrastive learning, and the SoccerReplay-1988 dataset, respectively.

cer data and adapts seamlessly to downstream tasks. Comparisons with additional baselines from the SoccerNet foul recognition challenges [9] are provided in the Appendix.

#### 5.3. Ablation Studies

We conduct ablation experiments on event classification and commentary generation using our SoccerReplay-test benchmark. These experiments validate the effectiveness of our proposed dataset and model, while establishing a baseline for future evaluations on this benchmark.

Event Classification. We evaluate event classification on 300 matches from our SoccerReplay-test benchmark using the MatchVision visual encoder pretrained with various strategies. Features are extracted by MatchVision and processed with a learnable aggregation layer and a linear classifier. The default training set is our curated SoccerNet-pro. As shown in Table 4, integrating SoccerReplay-1988 for training results in significant performance improvements across all pretraining strategies, yielding the significance of our dataset. Additionally, supervised classification outperforms visual-language contrastive learning and hybrid pretraining. This is due to its closer alignment with downstream event classification task, and the scale of event annotations is far larger than that of textual commentaries, further confirming the substantial benefits of data scaling for boosting soccer understanding.

Commentary Generation. With the pretrained MatchVision encoder, we train the commentary generation head on the MatchTime [41] and SoccerReplay-1988 datasets using various training strategies. By default, only the Perceiver [25] aggregation layer and projection layer within the head are trained. For joint training with the LLM decoder, considering computational costs, we incorporate LoRA [24] layers while freezing the original LLM layers. As shown in Table 5, incorporating SoccerReplay-1988 significantly improves performance on all metrics, confirming substantial advantages of our proposed dataset. This performance

[Figure 87]

[Figure 88]

###### Soccer Videos Events Commentaries

[Figure 89]

- a

- b

- c

- d

- e

[Figure 90]

[Figure 91]

[Figure 92]

Goal! [PLAYER] ([TEAM]) puts the ball into the top left corner past the outstretched arm of [PLAYER]. [PLAYER] ([TEAM]) is going to take the penalty! Goal! [PLAYER] ([TEAM]) wins the battle of wills and sends an unstoppable penalty past [PLAYER] into the top left corner.

GT:

GT: Penalty w/o SR:

w/o SR: w/ SR:

Penalty ( ) Penalty ( )

[Figure 93]

w/ SR:

[Figure 94]

GT:

What a save from [PLAYER]! [PLAYER] ([TEAM]) jumps high to meet a cross and glances a promising header from close range, but the goalkeeper pulls off a brilliant stop to keep it out.

[Figure 95]

[Figure 96]

[Figure 97]

GT: Lead to Corner Lead to Corner ( ) Saved by Goal-keeper ( )

[PLAYER] ([TEAM]) whips in the corner, but one of the defending players gets a head on it and intercepts.

w/o SR: w/ SR:

w/o SR: w/ SR:

[Figure 98]

[Figure 99]

[PLAYER] ([TEAM]) takes the resulting free kick from outside the box. He sends a nice cross into the box, but his intention is well anticipated by the goalkeeper who comes out to collect the ball.

GT:

[PLAYER] ([TEAM]) connects with a pass but sees his shot from the edge of the box blocked.

[Figure 100]

[Figure 101]

[Figure 102]

GT: Clearance Shot off Target ( ) Clearance ( )

w/o SR: w/ SR:

[PLAYER] ([TEAM]) makes a yard for himself on the edge of the box after picking up a pass, but produces a poor effort that sails high over the bar.

w/o SR: w/ SR:

[Figure 103]

[PLAYER] ([TEAM]) fires a shot at goal from a very promising position outside the penalty area, but it is blocked by a self-sacrificing defender.

[Figure 104]

GT:

[Figure 105]

[Figure 106]

[Figure 107]

Play continues despite the shouts by some [TEAM] players for a penalty. Wait! Referee [REFEREE] interrupts the game right now and he will review that decision using VAR! Let's see what happens. [PLAYER] ([TEAM]) receives a yellow card from the referee for a foul that he committed a little earlier. Referee [REFEREE] makes the VAR signal and is going to check whether it’s a penalty for [TEAM]. Let’s see what happens!

GT: VAR Yellow Card ( ) VAR ( )

w/o SR: w/ SR:

w/o SR: w/ SR:

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

GT:

[TEAM] take a short corner kick.

GT: Corner

w/o SR: w/ SR:

[PLAYER] ([TEAM]) takes the corner but only sends it into a huddle of the defenders and one of them makes a good clearance.

Corner ( ) Corner ( )

[Figure 113]

w/o SR: w/ SR:

[Figure 114]

[PLAYER] ([TEAM]) works the corner short instead of sending the ball into the penalty area.

Figure 4. Qualitative Results for Event Classification and Commentary Generation. Here, “w/o SR” and “w/ SR” indicate models trained without and with the SoccerReplay-1988 dataset, respectively. Incorporating SoccerReplay-1988 improves event classification accuracy. Moreover, this enriched training data enables the model to demonstrate several advantages in commentary generation: (a) more detailed descriptions, (b) greater linguistic variety, (c) higher event depiction accuracy, (d) better adherence to updated rules, and (e) improved specificity in scenario response.

remain contextually relevant. For commentary generation, hybrid training on SoccerReplay-1988 enables MatchVision to produce richer, more detailed textual commentary, reflecting a deeper understanding of soccer dynamics. More qualitative results are available in the Appendix.

Trainable Commentary Metrics V L B@1 B@4 M R-L C

Trained on MatchTime

- ✗ ✗ 21.65 3.27 21.02 17.79 12.90

- ✓ ✗ 27.62 7.02 24.03 23.51 30.77

✗ ✓ 27.04 6.41 24.15 23.88 31.91 ✓ ✓ 27.49 6.96 24.50 23.33 30.81

Trained on MatchTime & SoccerReplay-1988 ✗ ✗ 24.17 4.09 20.51 20.70 15.70

- ✓ ✗ 28.98 8.39 24.45 25.35 45.85

### 6. Conclusion

In this paper, we establish a unified, scalable multi-modal framework for soccer understanding. Specifically, we introduce SoccerReplay-1988, the largest and most comprehensive soccer video dataset to date, annotated by an automated curation pipeline. This provides a solid foundation for developing soccer understanding models and serves as a more challenging benchmark. Built upon this, we develop MatchVision, an advanced soccer-specific visual encoder, which effectively leverages spatiotemporal information within soccer videos and can be applied to various tasks such as event classification and commentary generation. Extensive experiments demonstrate the superiority of our model, with MatchVision achieving state-of-the-art performance on both existing benchmarks and our newly established one. We believe this work will set a viable, universal paradigm for future research in sports understanding.

✗ ✓ 27.54 7.76 24.50 24.70 42.79 ✓ ✓ 29.21 8.22 25.25 25.54 43.18

Table 5. Ablations on Commentary Generation. We investigate the impact of different training strategies and datasets on MatchVision using the SoccerReplay-test benchmark. ‘V’ and ‘L’ denote the visual encoder and the LLM decoder, respectively.

gap also reflects the challenges of our established benchmark, which features diverse vocabulary, richer semantics, and updated soccer rules. Additionally, jointly finetuning the visual encoder and the LLM decoder provides a feasible approach for further improvements.

#### 5.4. Qualitative Comparisons

As depicted in Figure 4, we present qualitative results of MatchVision on the SoccerReplay-test benchmark, comparing models pretrained with and without SoccerReplay1988. For event classification, incorporating our data improves accuracy, and even in misclassified cases, the results

### Acknowledgments

This work is funded by National Key R&D Program of China (No.2022ZD0161400).

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, pages 23716– 23736, 2022. 2
- [2] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, 2005. 6
- [3] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Proceedings of the International Conference on Machine Learning, page 4, 2021. 2, 4
- [4] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 6
- [5] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2
- [6] Anthony Cioppa, Adrien Deliege, Floriane Magera, Silvio Giancola, Olivier Barnich, Bernard Ghanem, and Marc Van Droogenbroeck. Camera calibration and player localization in soccernet-v2 and investigation of their representations for action spotting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 4537–4546, 2021. 2
- [7] Anthony Cioppa, Adrien Deliege, Silvio Giancola, Bernard Ghanem, and Marc Van Droogenbroeck. Scaling up soccernet with multi-view spatial localization and re-identification. Scientific data, 9(1):355, 2022. 1, 2
- [8] Anthony Cioppa, Silvio Giancola, Adrien Deliege, Le Kang, Xin Zhou, Zhiyu Cheng, Bernard Ghanem, and Marc Van Droogenbroeck. Soccernet-tracking: Multiple object tracking dataset and benchmark in soccer videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 3491–3502, 2022. 2
- [9] Anthony Cioppa, Silvio Giancola, Vladimir Somers, Victor Joos, Floriane Magera, Jan Held, Seyed Abolfazl Ghasemzadeh, Xin Zhou, Karolina Seweryn, Mateusz Kowalczyk, Zuzanna Mr´oz, Szymon Łukasik, Michał Hało´n, Hassan Mkhallati, Adrien Deli`ege, Carlos Hinojosa, Karen Sanchez, Amir M. Mansourian, Pierre Miralles, Olivier Barnich, Christophe De Vleeschouwer, Alexandre Alahi, Bernard Ghanem, Marc Van Droogenbroeck, Adam Gorski, Albert Clap´es, Andrei Boiarov, Anton Afanasiev, Artur Xarles, Atom Scott, ByoungKwon Lim, Calvin Yeung, Cristian Gonzalez, Dominic R¨ufenacht, Enzo Pacilio, Fabian Deuser, Faisal Sami Altawijri, Francisco Cach´on, HanKyul Kim, Haobo Wang, Hyeonmin Choe, Hyunwoo J Kim, Il-Min

- Kim, Jae-Mo Kang, Jamshid Tursunboev, Jian Yang, Jihwan Hong, Jimin Lee, Jing Zhang, Junseok Lee, Kexin Zhang, Konrad Habel, Licheng Jiao, Linyi Li, Marc Guti´errezP´erez, Marcelo Ortega, Menglong Li, Milosz Lopatto, Nikita Kasatkin, Nikolay Nemtsev, Norbert Oswald, Oleg Udin, Pavel Kononov, Pei Geng, Saad Ghazai Alotaibi, Sehyung Kim, Sergei Ulasen, Sergio Escalera, Shanshan Zhang, Shuyuan Yang, Sunghwan Moon, Thomas B. Moeslund, Vasyl Shandyba, Vladimir Golovkin, Wei Dai, WonTaek Chung, Xinyu Liu, Yongqiang Zhu, Youngseo Kim, Yuan Li, Yuting Yang, Yuxuan Xiao, Zehua Cheng, and Zhihao Li. Soccernet 2024 challenges results. arXiv preprint arXiv:2409.10587, 2024. 2, 7, 18
- [10] Adrien Deliege, Anthony Cioppa, Silvio Giancola, Meisam J Seikavandi, Jacob V Dueholm, Kamal Nasrollahi, Bernard Ghanem, Thomas B Moeslund, and Marc Van Droogenbroeck. Soccernet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 4508–4519, 2021. 1, 2, 3, 6, 14, 15, 16, 17
- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings of the International Conference on Learning Representations, 2021. 4
- [12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 3, 5, 15

- [13] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6824–6835, 2021. 6, 7
- [14] Sushant Gautam, Mehdi Houshmand Sarkhoosh, Jan Held, Cise Midoglu, Anthony Cioppa, Silvio Giancola, Vajira Thambawita, Michael A Riegler, P˚al Halvorsen, and Mubarak Shah. Soccernet-echoes: A soccer game audio commentary dataset. arXiv preprint arXiv:2405.07354,

2024. 2

- [15] Silvio Giancola, Mohieddine Amine, Tarek Dghaily, and Bernard Ghanem. Soccernet: A scalable dataset for action spotting in soccer videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 1711–1721, 2018. 1, 2, 3, 14
- [16] Xiaofan Gu, Xinwei Xue, and Feng Wang. Fine-grained action recognition on a novel basketball dataset. In International Conference on Acoustics, Speech, and Signal Processing, pages 2563–2567, 2020. 2
- [17] Tengda Han, Weidi Xie, and Andrew Zisserman. Temporal alignment networks for long-term video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2906–2916, 2022. 2

- [18] Tengda Han, Max Bain, Arsha Nagrani, G¨ul Varol, Weidi Xie, and Andrew Zisserman. Autoad: Movie description in context. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 2
- [19] Tengda Han, Max Bain, Arsha Nagrani, Gul Varol, Weidi Xie, and Andrew Zisserman. Autoad ii: The sequel-who, when, and what in movie audio description. In Proceedings of the International Conference on Computer Vision, 2023.
- [20] Tengda Han, Max Bain, Arsha Nagrani, G¨ul Varol, Weidi Xie, and Andrew Zisserman. Autoad iii: The prequel - back to the pixels. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2
- [21] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016. 6, 7
- [22] Jan Held, Anthony Cioppa, Silvio Giancola, Abdullah Hamdi, Bernard Ghanem, and Marc Van Droogenbroeck. Vars: Video assistant referee system for automated soccer decision making from multiple views. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 5085–5096, 2023. 2, 5, 6
- [23] Jan Held, Hani Itani, Anthony Cioppa, Silvio Giancola, Bernard Ghanem, and Marc Van Droogenbroeck. X-vars: Introducing explainability in football refereeing with multimodal large language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 3267–3279, 2024. 2, 6
- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In Proceedings of the International Conference on Learning Representations, 2022. 7, 17
- [25] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In Proceedings of the International Conference on Machine Learning, pages 4651– 4664, 2021. 5, 7, 17
- [26] Yudong Jiang, Kaixu Cui, Leilei Chen, Canjin Wang, and Changliang Xu. Soccerdb: A large-scale database for comprehensive video understanding. In Proceedings of the 3rd International Workshop on Multimedia Content Analysis in Sports, pages 1–8, 2020. 14
- [27] Haopeng Li, Andong Deng, Qiuhong Ke, Jun Liu, Hossein Rahmani, Yulan Guo, Bernt Schiele, and Chen Chen. Sports-qa: A large-scale video question answering benchmark for complex and professional sports. arXiv preprint arXiv:2401.01505, 2024. 2
- [28] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the International Conference on Machine Learning, pages 12888–12900, 2022. 2
- [29] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learn-

ing, pages 19730–19742, 2023. 2

- [30] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 2
- [31] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In Proceedings of the European Conference on Computer Vision,

2024. 2

- [32] Zeqian Li, Qirui Chen, Tengda Han, Ya Zhang, Yanfeng Wang, and Weidi Xie. Multi-sentence grounding for longterm instructional video. In Proceedings of the European Conference on Computer Vision, 2024. 2
- [33] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, 2004. 6
- [34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of the International Conference on Learning Representations, 2019. 5
- [35] Hassan Mkhallati, Anthony Cioppa, Silvio Giancola, Bernard Ghanem, and Marc Van Droogenbroeck. Soccernetcaption: Dense video captioning for soccer broadcasts commentaries. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 5074–5085, 2023. 1, 2, 3, 16, 17
- [36] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Association for Computational Linguistics, pages 311–318, 2002. 6
- [37] Ji Qi, Jifan Yu, Teng Tu, Kunyu Gao, Yifan Xu, Xinyu Guan, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li, et al. Goal: A challenging knowledge-grounded video captioning benchmark for real-time soccer commentary generation. In Proceedings of the ACM International Conference on Information and Knowledge Management, pages 5391–5395, 2023. 1, 2, 3, 14
- [38] Mengshi Qi, Yunhong Wang, Annan Li, and Jiebo Luo. Sports video captioning via attentive motion representation and group relationship modeling. IEEE Transactions on Circuits and Systems for Video Technology, 2019. 2
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, 2021. 2, 6
- [40] Prem N Ramkumar, Bryan C Luu, Heather S Haeberle, Jaret M Karnuta, Benedict U Nwachukwu, and Riley J Williams. Sports medicine and artificial intelligence: a primer. The American Journal of Sports Medicine, 50(4): 1166–1174, 2022. 2
- [41] Jiayuan Rao, Haoning Wu, Chang Liu, Yanfeng Wang, and Weidi Xie. Matchtime: Towards automatic soccer game commentary generation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing,

2024. 1, 2, 3, 5, 6, 7, 14, 15, 16, 17

- [42] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2616–2625, 2020. 2
- [43] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. arXiv preprint arXiv:2409.14485,

2024. 2

- [44] Vladimir Somers, Victor Joos, Anthony Cioppa, Silvio Giancola, Seyed Abolfazl Ghasemzadeh, Floriane Magera, Baptiste Standaert, Amir M Mansourian, Xin Zhou, Shohreh Kasaei, et al. Soccernet game state reconstruction: End-toend athlete tracking and identification on a minimap. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 3293–3305, 2024. 2
- [45] Alessandro Suglia, Jos´e Lopes, Emanuele Bastianelli, Andrea Vanzo, Shubham Agarwal, Malvina Nikandrou, Lu Yu, Ioannis Konstas, and Verena Rieser. Going for goal: A resource for grounded football commentaries. arXiv preprint arXiv:2211.04534, 2022. 14
- [46] Graham Thomas, Rikke Gade, Thomas B Moeslund, Peter Carr, and Adrian Hilton. Computer vision for sports: Current applications and research topics. Computer Vision and Image Understanding, 159:3–18, 2017. 2
- [47] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the International Conference on Computer Vision, pages 4489–4497,

2015. 6

- [48] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6450–6459, 2018. 6, 7
- [49] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4566–4575, 2015. 6, 17
- [50] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 6
- [51] Zhe Wang, Petar Veliˇckovi´c, Daniel Hennes, Nenad Tomaˇsev, Laurel Prince, Michael Kaisers, Yoram Bachrach, Romuald Elie, Li Kevin Wenliang, Federico Piccinini, et al. Tacticai: an ai assistant for football tactics. Nature Communications, 15(1):1–13, 2024. 1, 2
- [52] Dekun Wu, He Zhao, Xingce Bao, and Richard P Wildes. Sports video analysis on large-scale data. In Proceedings of the European Conference on Computer Vision, 2022. 2
- [53] Zeyu Xi, Ge Shi, Xuefen Li, Junchi Yan, Zun Li, Lifang Wu, Zilin Liu, and Liang Wang. A simple yet effective knowledge guided method for entity-aware video captioning on a basketball benchmark. Neurocomputing, 2025. 2

- [54] Zeyu Xi, Ge Shi, Haoying Sun, Bowen Zhang, Shuyi Li, and Lifang Wu. Eika: Explicit & implicit knowledge-augmented network for entity-aware sports video captioning. Expert Systems with Applications, 2025. 2
- [55] Haotian Xia, Zhengbang Yang, Yuqing Wang, Rhys Tracy, Yun Zhao, Dongdong Huang, Zezhi Chen, Yan Zhu, Yuanfang Wang, and Weining Shen. Sportqa: A benchmark for sports understanding in large language models. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics, 2024. 2
- [56] Haotian Xia, Zhengbang Yang, Junbo Zou, Rhys Tracy, Yuqing Wang, Chi Lu, Christopher Lai, Yanjun He, Xun Shao, Zhuoqing Xie, et al. Sportu: A comprehensive sports understanding benchmark for multimodal large language models. In Proceedings of the International Conference on Learning Representations, 2025. 2
- [57] Jinglin Xu, Yongming Rao, Xumin Yu, Guangyi Chen, Jie Zhou, and Jiwen Lu. Finediving: A fine-grained dataset for procedure-aware action quality assessment. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2949–2958, 2022. 2
- [58] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 10714–10726, 2023. 2
- [59] Huanyu Yu, Shuo Cheng, Bingbing Ni, Minsi Wang, Jian Zhang, and Xiaokang Yang. Fine-grained video captioning for sports narrative. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. 2, 14
- [60] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the International Conference on Computer Vision, pages 11975–11986, 2023. 2, 5, 6, 17
- [61] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2023. 2
- [62] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264,

2024. 2

- [63] Xin Zhou, Le Kang, Zhiyu Cheng, Bo He, and Jingyu Xin. Feature combination meets attention: Baidu soccer embeddings and transformer based temporal detection. arXiv preprint arXiv:2106.14447, 2021. 2, 6, 7
- [64] Xingyi Zhou, Anurag Arnab, Shyamal Buch, Shen Yan, Austin Myers, Xuehan Xiong, Arsha Nagrani, and Cordelia Schmid. Streaming dense video captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2

## Towards Universal Soccer Video Understanding Appendix

### Contents

- 1. Introduction 1
- 2. Related Works 2
- 3. SoccerReplay-1988 Dataset 2

- 3.1. Dataset Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- 3.2. Automated Data Curation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3.3. Statistics & Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

4. Method 4

- 4.1. Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 6. Conclusion 8

- 4.2. Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 4.3. MatchVision Pretraining . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 4.4. Downstream Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

5. Experiments 5

- 5.1. Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 5.2. Quantitative Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 5.3. Ablation Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5.4. Qualitative Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

###### A. SoccerReplay-1988 Dataset Details 13

- A.1. Dataset Format . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- A.2. Additional Dataset Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- A.3. Event Summarization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

###### B. SoccerNet-pro Dataset Details 16

- B.1. SoccerNet-v2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- B.2. MatchTime . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- B.3. Data Split Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

###### C. Implementation Details 17

- C.1. Data Preprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.2. Validation Strategy during Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.3. Hyperparameter Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

###### D. Experiments 17

- D.1. Training Curves . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- D.2. More Quantitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- D.3. More Qualitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

###### E. Limitations & Future Work 18

### A. SoccerReplay-1988 Dataset Details

In this section, we provide additional details of our SoccerReplay-1988 dataset. Specifically, Sec. A.1 elaborates on the structure and format of the dataset; Sec. A.2 presents statistical information and analyses of the dataset; and Sec. A.3 describes the methodology to automatically generate event labels within the dataset.

#### A.1. Dataset Format

The SoccerReplay-1988 dataset consists of match videos, descriptions of events, and related game information of 1988 soccer matches. Each match includes two mkv video files (1-half and 2-half), covering the match from the initial kick-off to the final whistle. Additionally, a json file is accompanied by encapsulating detailed information, including event descriptions and comprehensive match backgrounds, structured as follows:

Match Information provides background details of the match, including competing teams, final results, and match contexts, such as start time, team formations, and venue details, as illustrated below:

{

}

"timestamp": "2022-08-07 21:00:00", # Match start time "score": "1 - 2", # Final score "home_team": "Manchester Utd", # Home team name "away_team": "Brighton", # Away team name "home_formation": "4 - 3 - 3", # Home team formation "away_formation": "3 - 4 - 2 - 1", # Away team formation "venue": "Old Trafford (Manchester)", # Venue and city "capacity": "75 635", # Stadium capacity "attendance": "73 711", # Number of attendees

Referee Information includes details about the primary referee officiating the match, which is formatted as follows:

{

}

"country": "Eng", # Referee’s nationality "name": "Paul Tierney" # Referee’s name

Player Information contains details about various types of individuals involved in the match, including starting players, substitutes, absent players, and coaches. All these types are stored in a unified list, with the following format:

{

"players_name": "Caicedo M.", # Player’s abbreviated name "players_number": "25", # Jersey number "Full Name": "Moises Caicedo", # Player’s full name "players_rating": 7.6, # Post-match rating "Country": "Ecuador", # Player’s nationality "Role": "Midfielder", # On-field role "Age and Birthdate": "22, (02.11.2001)", # Age and birth date "Market Value": "C89.4m" # Player’s market value

}

Event Descriptions is a list that records all key events during the match, including their types and detailed commentary. A typical example of an event entry is shown below:

{

"half": 1, # Match half (1 or 2) "time_stamp": "00:16", # Timestamp within the half "comments_type": "shot off target", # Event type "comments_text": "A mistake by Leandro Trossard (Brighton)...", # Commentary text "comments_text_anonymized": "A mistake by [PLAYER]([TEAM])..." # Commentary after anonymization

}

#### A.2. Additional Dataset Statistics League # Match

Season # Match

- 2017-2018 172
- 2018-2019 325
- 2019-2020 300
- 2020-2021 323
- 2021-2022 330
- 2022-2023 416
- 2023-2024 122

Italy Serie-a 367 England Premier League 552 z UEFA Champions League 469 France Ligue 1 123 Spain LaLiga 235 Germany Bundesliga 242

Table 6. League-wise Match Statistics.

###### Table 7. Season-wise Match Statistics.

To provide a comprehensive analysis of SoccerReplay-1988, we present statistics and visualizations in the following tables and figures. Concretely, Table 6 and 7 illustrate the distribution of the 1988 matches across different leagues and seasons. Figure 5 (a) compares SoccerReplay-1988 with other soccer datasets [10, 15, 26, 37, 41, 45, 59], highlighting its unique scale. Figure 5 (b) depicts the distribution of event labels for the 24 newly defined categories. Finally, Figures 5 (c), (d), (e), and (f) present detailed analyses of commentary data, including frequency distributions, timestamps, and word counts.

[Figure 115]

[Figure 116]

Ball out of play

free kick

injury end of half game show added time

##### SoccerReplay-1988

corner

startofhalfgame

Videoduration(min)

offside

###### SoccerNet-v1

ballpossession

MatchTime

clearance

Savedby goal-keeper

SoccerNet-v2

###### SoccerDB

leadtocorner

substitution

yellow card

no cardfoul with

Shot offtarget

###### Yu et al

GOAL (Suglia et al)

GOAL (Qi et al)

# Annotations

(a) Comparisons of Different Soccer Datasets.

(b) Distribution of Event Labels of 24 Classes in SoccerReplay-1988.

[Figure 117]

[Figure 118]

(c) Top-20 Frequency Distribution from Commentaries. (d) Word Cloud of Commentaries.

Corner

[Figure 119]

[Figure 120]

(e) Distribution of Temporal Occurrences of Commentaries. (f) Commentary Word Count Distribution.

###### Figure 5. Comprehensive Visualizations of SoccerReplay-1988 Dataset.

#### A.3. Event Summarization

Our dataset expands the original 17 event categories in SoccerNet [10] to 24 types. For MatchTime [41] and SoccerReplay1988, LLaMA-3 (70B) [12] is employed to analyze commentaries and generate corresponding event labels. This is guided by a carefully refined prompt, iteratively improved through manual checks and enriched with comprehensive definitions and examples. Notably, text commentaries unrelated to visual content (e.g., ‘possession ratio is 55:45’) are categorized as ‘statistics and summary’, and excluded from model training and testing. Finally, a random sample of 2% of the data yields 98% manual verification accuracy, confirming the high quality of automated labeling. The entire prompt is presented below:

<|begin of text|> <|start header id|>system<|end header id|>

You are an expert in soccer, you have a very important task to summarize a soccer commentary into certain types of events. The accuracy of your classification is the most emergency thing. I will give you a commentary sentence. You need to select one type of event that can best describe this event from the following 24 types: ’corner’, ’goal’, ’injury’, ’own goal’, ’penalty’, ’penalty missed’, ’red card’, ’second yellow card’, ’substitution’, ’start of game(half)’, ’end of game(half)’, ’yellow card’, ’throw in’, ’free kick’, ’saved by goal-keeper’, ’shot off target’, ’clearance’, ’lead to corner’, ’off-side’, ’var’, ’foul (no card)’, ’statistics and summary’, ’ball possession’, ’ball out of play’.

Here are some rules you have to obey when summarizing types, you should consider it strictly following these steps:

- 1. Firstly, you need to find if there is any evidence of foul in commentary, if yes, it can only be ’foul (no card)’, ’yellow card’, ’red card’ or ’second yellow card’ according to the situation, even though it introduces the result ’free kick’ or ’penalty’. For example: ’Per Mertesacker (Arsenal) commits a rough foul. Michael Dean stops the game and makes a call. That’s a free kick to Manchester Utd.’ can ONLY be ’foul (no card)’ since there is a foul in commentary, even though the result is ’free kick’.
- 2. Secondly, only if the word ’corner’ is in the commentary, you need to select it from ’lead to corner’. ’lead to corner’ means the process of how the ’corner’ occurs, which is before the ’corner’ kick. For type ’lead to corner’, there will always be words like ’award a corner’, ’will have a corner’, ’point at corner flag’ and so on. For example: ’Victor Wanyama (Southampton) goes on a solo run, but he fails to create a chance as an opposition player blocks him. The referee signals a corner kick to Southampton.’ is ’lead to corner’.
- 3. Thirdly is the most easy-confused part, you need to be cautious: only if the word ’free-kick’/’free kick’ is in the commentary will it be a ’free kick’. According to the first rule, if there is foul in the sentence, it cannot be ’free kick’. ’free kick’ can only be selected when ’free-kick’/’free kick’ occurs in commentary and is describing the process of a ’free kick’ attack. For example: ’Olivier Giroud (Arsenal) gets on the ball and beats an opponent, but his run is stopped by the referee Michael Dean who sees an offensive foul. It’s a free kick to Burnley, but they probably won’t attempt a direct shot on goal from here.’ is ’foul (no card)’; ’Ander Herrera (Manchester United) makes a slide tackle, but referee Michael Dean blows for a foul. Free kick. Arsenal will probably just try to cross the ball in from here.’ is ’foul (no card)’; ’Marcos Rojo (Manchester United) connects with the free kick and produces a header goalwards which is well blocked. The goalkeeper doesn’t have to worry about that one.’ is ’free kick’.
- 4. Similarly, ’penalty’ and ’penalty missed’ only describe things that happen during a ’penalty’ kick. If it is introducing the reason that leads to a ’penalty’, you should return the type describing the reason, like ’foul (no card)’, ’yellow card’, and so on.
- 5. The type ’statistics and summary’ includes all the commentaries that are not introducing visually evidential events, but those statistics or overviews of the game. These sentences won’t concentrate on certain events, but on the overall game.
- 6. ’ball possession’ represents those commentaries that describe any of the teams controlling the ’ball possession’.
- 7. You need to be sensitive about the type ’shot off target’; if there is an event of a shot happening in the commentary, it is a shot. If it’s not a ’goal’, didn’t make a score, and was not saved by the goalkeeper, it would probably be a ’shot off target’. Normally there will be keywords like ’wide of the right post’, ’over the crossbar’, ’crashes against the crossbar’ and so on. You have to judge it sensitively about the situation after the shot.
- 8. An important type after a shot: ’saved by goal-keeper’ describes that the shot is saved by the goalkeeper; there would be words like ’blocked’, ’saved’, and so on. Especially when ’goal-keeper’/’goal keeper’ occurs in the commentary sentence!!! it will probably be ’saved by goal-keeper’. You need to find it carefully!!!
- 9. If a player lofts or swings a pass to a penalty area/dangerous area, they might be ’shot off target’, ’clearance’, ’saved by goal-keeper’, and so on. It should NOT be identified as ’corner’ or ’free kick’ if there is no obvious evidence in commentary! For example: ’Tomas Rosicky (Arsenal) fails to find any of his teammates inside the box as his pass is blocked.’ should be ’clearance’ rather than ’corner’ or ’free kick’.
- 10. ’clearance’ means those good performances in defense; they stop the offense of opponents. If such a successful defense happens in the commentary, it can only be ’clearance’. In these commentaries, there are always some words like ’opponent’s defense’, ’intercepts the ball’, ’clear the ball’, and so on.
- 11. ’offside’ is an obvious event; there are always the words ’flag’, ’linesman’, ’too fast to defense’ in the commentary since ’offside’ is the player running forward the defense line, and the linesman will raise the flag.
- 12. ’ball out of play’ means any player kicks the ball out of boundary lines. These commentary sentences will mostly end up with throw-ins or goal kicks.
- 13. ’throw-in’ means exactly the process of ’throw-in’ balls.
- 14. Most ’goals’ are normal ’goals’. If you see a scoring event, you can only identify the score as ’own goal’ when there is obvious evidence.

<|eot id|> <|start header id|>user<|end header id|>

With the classification rules, you should tell me the type of a commentary from above candidate options: ’corner’, ’goal’, ’injury’, ’own goal’, ’penalty’, ’penalty missed’, ’red card’, ’second yellow card’, ’substitution’, ’start of game(half)’, ’end of game(half)’, ’yellow card’, ’throw in’, ’free kick’, ’saved by goal-keeper’, ’shot off target’, ’clearance’, ’lead to corner’, ’off-side’, ’var’, ’foul (no card)’, ’statistics and summary’, ’ball possession’, ’ball out of play’. The commentary sentence you need to define type is:

[COMMENTARY TEXT HERE (before anonymization)]

You need to carefully consider the rules in order and make your final decision. Now, you must return me the name of its type from candidate options (in lower case, only return the name of type, answer it right away after my prompt without any other words).

<|eot id|>

### B. SoccerNet-pro Dataset Details

As discussed in the main text, alongside the SoccerReplay-1988 dataset, we also incorporate two existing datasets, SoccerNetv2 [10] and MatchTime [41] to enrich the training data. These datasets undergo the following preprocessing strategies and are then unified into the SoccerNet-pro dataset, ensuring format consistency with SoccerReplay-1988.

#### B.1. SoccerNet-v2

The SoccerNet-v2 [10] dataset comprises over 110k event labels across 500 matches, categorized into 17 distinct classes. Based on soccer rules and specific domain knowledge, these labels are systematically reclassified into 24 categories with our proposed automated data curation pipeline, as detailed in Table 8.

Original Label Processed Label Reference Penalty

Penalty Scored penalties are categorized as “Penalty.” Penalty Missed Missed penalties are categorized as “Penalty Missed.”

Kick-off Start of Game (Half) Matches the start of a half after goals. Shots off target Shot Off Target No change. Throw-in Throw In No change. Ball out of play Ball Out of Play No change. Foul Foul (No Card) Refers to fouls without cards for only. Yellow card Yellow Card No change. Yellow→red card Second Yellow Card No change. Red card Red Card No change. Direct free-kick

Free Kick Both direct and indirect free kicks are grouped.

Indirect free-kick Substitution Substitution No change. Goal Goal No change. Clearance Clearance No change. Offside Off-Side No change. Corner Corner No change.

Table 8. Processing Strategy for SoccerNet-pro. The Reference column details the specific processing applied to the original labels.

#### B.2. MatchTime

The MatchTime dataset [41], curated from SoccerNet-Caption [35], contains a substantial amount of commentary, with only a small portion accompanied by event labels. To bridge this gap, we apply the prompt-based approach described in Sec. A.3 to summarize commentaries into event labels, assigning each commentary a corresponding label.

#### B.3. Data Split Strategy

As described in the manuscript, SoccerReplay-1988 is divided into 1,488 matches for training, 250 for validation, and 250 for testing. For the processed SoccerNet-pro dataset (including SoccerNet-v2 and MatchTime), we adhere to the original partitioning strategies and match distributions of its source datasets as detailed in Table 9.

###### Dataset Train Valid Test Total

SoccerNet-v2 [10] 300 100 100 500 MatchTime [41] 373 49 49 471 SoccerReplay-1988 1488 250 250 1988

Table 9. Dataset Splits for Training, Validation, and Testing.

### C. Implementation Details

In this section, we provide additional implementation details about MatchVision. Sec. C.1 presents more information on data preprocessing strategies; Sec. C.2 elaborates on the evaluation strategies used during model training; and Sec. C.3 discusses several hyperparameter choices inspired by prior works.

- C.1. Data Preprocessing

Our automated data curation pipeline filters out video clips with missing annotations, incorrect cropping, or invalid timestamps. In all experiments, video frames are resized to 224 × 224 and preprocessed using the image preprocessor of SigLIP [60], which normalizes frames to a mean of 0.5 and a standard deviation of 0.5 before serving as inputs. For overlapping video content between SoccerNet-v2 [10] and MatchTime [41], we prioritize using event labels from SoccerNet-v2.

- C.2. Validation Strategy during Training

We select the best-performing checkpoints on the validation set with the evaluation strategies detailed below:

During pretraining: (i) For supervised classification, we adopt top-1/3/5 event classification accuracy on the validation set to select the best model; (ii) For visual-language contrastive learning, video-to-text retrieval is performed, with top-1/3/5 accuracy of event classification (comparing retrieved texts’ event labels to ground truth) as the validation metric.

During downstream tasks training: (i) In event classification and foul recognition, classification accuracy on the validation set is used as the evaluation metric; (ii) For commentary generation, the CIDEr [49] score of the model’s predictions on the validation set is employed to select the best checkpoint.

- C.3. Hyperparameter Selection

Here, we provide further explanations about the hyperparameters in our model, inspired by prior works, as detailed below:

Temporal Window Size. We adopt a 30-second temporal window to extract video clips. This is inspired by MatchTime [41], which demonstrates that a 30-second window is sufficient to capture adequate visual information for optimal performance, outperforming the 45-second window used in SoccerNet-Caption [35].

LoRA Rank. For finetuning the commentary generation head, we use LoRA [24] with a rank of 16, following [41]. Query Length of Perceiver. For the Perceiver [25] module in the commentary generation head, we utilize a query length of 32 for temporal information aggregation, consistent with the optimal configuration reported in [41].

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

SigLIP

SigLIP

MatchVision

MatchVision

CrossEntropyLoss

SigLIPLoss

# Iterations # Iterations

(a) Supervised Training (b) Video-Language Contrastive Learning

Figure 6. Training Loss Curves of Visual Encoders Pretraining.

### D. Experiments

In this section, we provide additional details to offer deeper insights into our model and its performance. Specifically, Sec. D.1 presents training curves to clearly illustrate the training process; Sec. D.2 and Sec. D.3 showcase more quantitative and qualitative results, respectively, demonstrating the model’s capability to effectively understand soccer dynamics.

#### D.1. Training Curves

We present the loss curves for visual encoder pretraining in Figure 6. Our MatchVision demonstrates significantly better convergence compared to SigLIP [60] backbone, indicating that it effectively leverages spatiotemporal attention to utilize

temporal information, learning representations better suited for highly dynamic soccer videos.

- D.2. More Quantitative Results

Here, we compare present comparisons with more advanced methods in the SoccerNet Foul Recognition challenge [9], where MatchVision remains competitive even with a frozen visual encoder.

Top-1 Accuracy Ours Baseline zyz PD PS GSN Redsox xiao he shang

Foul 0.44 0.36 0.58 0.44 0.60 0.46 Severity 0.58 0.54 0.58 0.50 0.05 0.47 Table 10. More Quantitative Results on Multi-view Foul Recognition.

- D.3. More Qualitative Results More qualitative visualizations of commentary generation across various events on the field are depicted in Figure 7, 8, and 9.

### E. Limitations & Future Work

Although MatchVision explores establishing a soccer-specific visual encoder, it is not without its limitations: (i) Currently, MatchVision is adapted to event classification, commentary generation, and foul recognition tasks. In the future, we plan to further extend it to more challenging tasks such as player tracking and dense video captioning, aiming to develop a more comprehensive foundation model for soccer analysis. (ii) Given computational and annotation constraints, SoccerReplay-1988 primarily focuses on European league soccer data. We aim to leverage our scalable automated annotation pipeline to further expand the dataset, encompassing a more comprehensive range of soccer data. (iii) Following prior works, our commentary generation remains anonymized. This is left for future work, where we aim to fully leverage contextual information available in our SoccerReplay-1988 dataset to enable more vivid, accurate, and context-aware commentary generation.

###### Corner

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

GT: Ours:

[PLAYER] ([TEAM]) puts a cross into the box from the corner but there is no panic from the opposition and they easily clear. [TEAM] failed to take advantage of the corner as the opposition's defence was alert and averted the threat.

###### Clearance

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

GT: Ours:

The ball is cleared after [PLAYER] ([TEAM]) attempted to dribble past an opposing player. The ball is out of play. A goal-scoring opportunity from a corner for [TEAM]. [PLAYER] ([TEAM]) delivers a lovely cross into the penalty area. Nevertheless, an opposition defender is alert and averts the danger with a brilliant clearance. [TEAM] have been awarded a corner kick. The referee and one of his assistants both point at the corner flag.

###### Substitution

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

GT: Ours:

[PLAYER] prepares a substitution. [PLAYER] is replaced by [PLAYER] ([TEAM]). Here is a change. [PLAYER] is going off and [PLAYER] gives the last tactical orders to [PLAYER] ([TEAM]).

Foul with no card

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

GT: Ours:

[PLAYER] ([TEAM]) does well to dispossess the attacker with a slide tackle, but the referee blows his whistle for a foul. [TEAM] get a free kick. [PLAYER] ([TEAM]) makes a slide tackle, but referee [REFEREE] blows for a foul.

###### Shot off target

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

GT: Ours:

[PLAYER] ([TEAM]) latches on to an accurate pass on the edge of the box and immediately unleashes a shot which is wide of the mark. [PLAYER] ([TEAM]) receives a precise pass on the edge of the box and shoots. His poorly placed shot flies well wide of the left post.

Yellow card

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[PLAYER] ([TEAM]) is lucky to receive a yellow card from the referee because he could easily have been given a red card for his foul. [PLAYER] ([TEAM]) is shown a yellow card by the referee for making a challenge on his opponent, but he looks angry with the decision.

GT: Ours:

Lead to corner

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[PLAYER] ([TEAM]) passes the ball from the edge of the box in order to find his teammate, who is in a good scoring position, but the defender blocks the pass and spanks the ball to safety. The ball goes out for a corner. [TEAM] can continue in their attacking effort. [PLAYER] ([TEAM]) goes on a solo run, but he fails to create a chance as an opposition player blocks him. The ball goes out of play. [TEAM] are awarded a corner kick.

GT: Ours:

Goal

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Goal! [PLAYER] ([TEAM]) takes a short pass from [PLAYER] in his stride in the box, jinks inside his man and unleashes an unstoppable shot into the bottom right corner. 0:1. Goal! [PLAYER] displays great vision and sends a pass to [PLAYER] ([TEAM]), who shows brilliant composure inside the box to bury the ball from close range in the back of the net. The score is 0:1.

GT: Ours:

Ball possession

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

GT: Ours:

[TEAM] are seeing far more of the ball now. The [TEAM] players are exchanging some short passes to try and open up the opposition’s defence and hit them swiftly on the break.

###### Off side

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[PLAYER] ([TEAM]) is adjudged offside. [PLAYER] ([TEAM]) fails to beat the offside trap and the linesman puts his flag up.

GT: Ours:

###### Start of half game

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

GT: Ours:

Referee [REFEREE] blows his whistle to start the second half. The half-time break is over and the second half is about to start.

Injury

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[PLAYER] ([TEAM]) hasn't suffered any serious injury which would see him leave the pitch. He's back in the game now. [PLAYER] ([TEAM]) is having a really tough time right now. We are about to find out how serious his injury is.

GT: Ours:

End of half game

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

GT: Ours:

The referee blows for the end of today's match. That's it for today, [REFEREE] has blown his whistle and the game is over.

###### Show added time

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

3 additional min. will be played. The fourth official shows 3 min. of added time.

GT: Ours:

###### Free kick

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

GT: Ours:

[PLAYER] ([TEAM]) sends in a lofted cross from a long-range free kick, but it goes out of play. [PLAYER] ([TEAM]) commits a foul and [REFEREE] immediately signals a free kick.

###### Ball out of Play

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

GT: Ours:

[PLAYER] ([TEAM]) does his best to latch onto a cross into the box, but he can't get to the ball. The ball goes out of play and [TEAM] will have a goal kick. [PLAYER] ([TEAM]) attempts to find one of his teammates, but puts far too much on the pass and the chance is gone. The ball goes out of play and [TEAM] will have a goal kick.

###### VAR

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

GT: Ours:

Referee [REFEREE] makes the VAR signal and is going to check whether it’s a penalty for [TEAM]. Let’s see what happens! Wait! The referee makes the VAR signal and he's going to review that incident in the box from earlier. This could be a penalty for [TEAM]! Let's see what he decides.

Red Card

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

###### GT:

[PLAYER] ([TEAM]) receives a red card after his awful challenge. He completely lost his temper and referee [REFEREE] sends him off the pitch. [TEAM] win a free kick. It's a promising situation for a direct shot. [PLAYER] ([TEAM]) has to be very careful for the rest of the match after receiving a yellow card from the referee for a bad tackle on an opponent.

Ours:

Throw in

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

GT: Ours:

The ball is behind the sideline. [PLAYER] ([TEAM]) takes a throw-in. The [TEAM] players are exchanging some short passes to try and open up the opposition’s defence and hit them swiftly on the break.

Second yellow card

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

GT: Ours:

[PLAYER] ([TEAM]) sees a yellow card in the hand of [REFEREE]. And as it's a second one this game, a red follows and [PLAYER] ([TEAM]) is sent from the pitch. [PLAYER] ([TEAM]) makes a reckless foul in order to win the ball from his opponent. [REFEREE] has a clear sight of it and blows his whistle.

