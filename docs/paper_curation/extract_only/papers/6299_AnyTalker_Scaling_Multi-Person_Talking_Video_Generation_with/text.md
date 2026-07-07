# arXiv:2511.23475v1[cs.CV]28Nov2025

## AnyTalker: Scaling Multi-Person Talking Video Generation with Interactivity Refinement

Zhizhou Zhong1,2 Yicheng Ji2,3 Zhe Kong1 Yiying Liu2∗ Jiarui Wang2 Jiasun Feng2 Lupeng Liu2,4 Xiangyi Wang2,4 Yanjia Li2 Yuqing She2,4 Ying Qin4 Huan Li3 Shuiyang Mao2 Wei Liu2 Wenhan Luo1† 1Hong Kong University of Science and Technology 2Video Rebirth 3Zhejiang University 4Beijing Jiaotong University

Homepage: https://hkust-c4g.github.io/AnyTalker-homepage Code: https://github.com/HKUST-C4G/AnyTalker

|[Figure 1]<br><br>Animal<br><br>[Figure 2]|[Figure 3]<br><br>Gestures| |[Figure 4]<br><br>| |
|---|---|---|---|---|
| |[Figure 5]<br><br>Lively| |[Figure 6]| |
|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]| | | | |
| |[Figure 10]<br><br>Interactive<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]|[Figure 15]| |[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]|

Figure 1. We propose AnyTalker, a powerful audio-driven framework for interactive multi-person video generation. It can generate natural videos that are rich in gestures, lively emotions, and interactivity, and can freely generalize to arbitrary IDs or even non-human cases.

#### Abstract

Recently, multi-person video generation has started to gain prominence. While a few preliminary works have explored audio-driven multi-person talking video generation, they often face challenges due to the high costs of diverse multi-

∗ Project leader. † Corresponding author.

person data collection and the difficulty of driving multiple identities with coherent interactivity. To address these challenges, we propose AnyTalker, a multi-person generation framework that features an extensible multi-stream processing architecture. Specifically, we extend Diffusion Transformer’s attention block with a novel identity-aware attention mechanism that iteratively processes identity–audio pairs, allowing arbitrary scaling of drivable identities. Besides, training multi-person generative models demands

massive multi-person data. Our proposed training pipeline depends solely on single-person videos to learn multiperson speaking patterns and refines interactivity with only a few real multi-person clips. Furthermore, we contribute a targeted metric and dataset designed to evaluate the naturalness and interactivity of the generated multi-person videos. Extensive experiments demonstrate that AnyTalker achieves remarkable lip synchronization, visual quality, and natural interactivity, striking a favorable balance between data costs and identity scalability.

#### 1. Introduction

In the era of digital media, video creation has emerged as a crucial component of media platforms. Podcasts, livestreaming sales, and entertainment programs often feature rich multi-person interactions, resulting in a growing demand for multi-person video generation. Despite the advent of large-scale video generation models [4, 28, 59, 70], which have furnished robust backbone architectures for audio-driven talking video generation methods [7, 11, 14, 15, 24, 26, 31, 35, 69], enabling the creation of realistic lip movements for single subjects, these models still struggle to accommodate the complexities of multi-person interactions.

This limitation has prompted multi-person driving approaches that scale to arbitrary numbers of persons, handle multi-stream signals, and enable differentiated control over each subject. Despite recent advances [23, 31, 63] that propose solutions for handling multi-stream audio signals, these methods typically require hundreds to thousands of hours of meticulously curated multi-person data, leading to prohibitive collection costs and limited reproducibility. Specifically, the challenge of gathering training data for multi-person scenarios is heightened by their complex dynamics, including turn-taking, role-switching, and nonverbal cues like eye gaze, which complicate data annotation. Most existing audio-visual datasets [9, 11, 33, 41, 65, 80] focus on single-person monologues or isolated facial animations, thereby limiting their applicability for training the audio-driven multi-person generation model. In addition, current multi-person driving approaches frequently struggle to model genuine interactivity, leading to unnatural results.

To address the aforementioned challenges, we introduce an innovative driving framework named AnyTalker. Built on the pre-trained video diffusion model [59], AnyTalker achieves impressive multi-person video driving with remarkably low data costs and a distinct emphasis on interactivity, as illustrated in Fig. 1. The central idea is to leverage low-cost single-person data for scalable multi-person video driving and refine interactivity with just a small amount of multi-person data (as little as 12 hours). Specifically, AnyTalker supports extending the number of drivable identities (IDs) to arbitrary numbers, with guaranteed interac-

tivity among all IDs. To accommodate multi-stream control signals, we design an extensible audio-to-face in-context attention mechanism supporting any number of IDs and audio inputs. The training process is divided into two stages based on the types of data used, with the number of speaker IDs in individual data samples evolving from one to many. In the first stage, we randomly concatenate single-person talking videos along the horizontal dimension to simulate multiperson talking scenarios, ensuring that the model acquires a baseline capacity for multi-person speaking patterns. In the second stage, we fine-tune the model using a small amount of multi-person data to enhance its interactive capabilities.

Commonly used single-person talking head benchmarks [65, 80, 81] lack multi-person interactions, rendering them unsuitable for assessing multi-person generation methods. While InterActHuman [63] offered a related benchmark, it focuses on single speakers, limiting its utility for interaction analysis. To address this limitation, we introduce a meticulously annotated benchmark featuring videos of two individuals engaging in both speech and eye contact, with fine-grained labels that mark the speaking and listening intervals, thus facilitating the assessment of interactions during listening states. Additionally, we firstly introduce a novel metric to evaluate interactivity by measuring the activity of eye keypoints during listening periods. The proposed benchmark and metric will fill the gap in the assessment of interactivity in multi-person generation methods, benefiting future research in this area.

The main contributions are summarized as follows: (1) We present an extensible multi-stream processing architecture for multi-person generation that can scale drivable identities arbitrarily. (2) A novel two-stage training pipeline is introduced for the model to learn multi-speaker speaking patterns from single-person data and achieve seamless inter-identity interactions via multi-person data refinement. (3) We propose a new metric that quantitatively evaluates multi-person interactivity for the first time, accompanied by a tailored benchmark dataset for thorough assessment. (4) Comprehensive experiments demonstrate that AnyTalker achieves state-of-the-art performance and strikes a favorable balance among identity scalability, interactivity, lip synchronization, and data cost.

#### 2. Related Work

##### 2.1. Audio-driven Talking Video Generation

Recent key advancements [19, 20, 44, 50, 75] in text-toimage fields have significantly catalyzed progress in downstream applications. Early efforts like EMO [55] extend pre-trained diffusion text-to-image models [50] to endto-end audio-driven virtual-human video generation [6, 8, 25, 34, 43, 60, 66, 73]. They typically integrate modules for temporal attention [17], identity control via Ref-

(a) Architecture (b) Data Construction

|[Figure 19]<br><br>[Figure 20]<br><br>50% Concatenated Multi-Person Data 50% Real Single-Person Data<br><br>[Figure 21]<br><br>&<br><br>Stage 1 (~1000h)|
|---|

[Figure 22]

ConcatenatedVideo

[ , ]

[Figure 23]

[Figure 24]

Mask List

…

[Figure 25]

[Figure 26]

cropfaces

Audio1 Audio2

&resize

|[Figure 27]<br><br>[Figure 28]<br><br>100% Real Multi-Person Data<br><br>Stage 2 (～12h)|
|---|

Wav2Vec

First Frame

[Figure 29]

| | | | |
|---|---|---|---|
| | | | |

3D VAE

CLIP

Text

T5

Audio-Face Tokens List

flatten patchify

Face token1 Face token2

[ , ]

| |
|---|

(c) Audio-Face Cross Attention

Text Tokens

Ref Tokens

Video Tokens

Audio token1 Audio token2

|,|
|---|

For ( , Mask) in zip(Audio-Face_Tokens_List, Mask_List):

k = k_project( ) v = v_project( ) q = q_project(

Text Cross Attention

Reference Cross Attention

Audio-Face Cross Attention

Self Attention

)

Video Feature X

output_list.append(CrossAttention(q, k, v) * Mask) Attention_output = output_list.sum()

AnyTalker Attention Block x N

- Figure 2. (a) The architecture of AnyTalker, which incorporates a novel multi-stream audio processing layer, Audio-Face Cross Attention, enables the handling of multiple facial and audio inputs. (b) The training of AnyTalker is divided into two stages: the first stage uses concatenated multi-person data derived from single-person data mixed with single-person data to learn accurate lip movements; the second stage employs authentic multi-person data to enhance the interactivity in generated videos. (c) The detailed implementation of Audio-Face Cross Attention, a recursively callable structure that applies masking to the output using face masks.

targeted control. MultiTalk [31] proposes Label Rotary Position Embedding [52] to address audio–person binding. All three approaches above rely on expensive data, ranging from hundreds to thousands of hours of multi-person talking data. Some models [7, 39] trained on single-person data generalize to multi-person driving through specialized controllers: HunyuanVideo-Avatar [7] leverages a FaceAware Audio Adapter to activate attention across different characters selectively, and Playmate2 [39] uses token-level masking within a classifier-free guidance framework to realize similar binding. Despite enabling multi-person outputs, these approaches might yield fragmented interactions across distinct characters, with limited interactivity between individuals.

erenceNet [22, 30, 74, 82], and audio-conditioning using pre-trained audio processing models [1, 47, 51], with conditional signals injected via cascaded attention layers [58]. Video foundation models [4, 28, 59, 70, 76] enable end-toend, audio-driven virtual-human models [11, 15, 24, 26, 35, 40, 56, 69]. These works largely employ audio-injection practices from earlier methods [25, 55] and report breakthroughs in long-video generation [11, 69], full-body synthesis [35], driving non-human cases [14, 15], synchronized lip movements [24], and coherent hand movements [40]. Consequently, they surpass earlier GAN-based generation methods [16, 67, 78, 79] and image diffusion-based techniques [44, 50] in terms of both quality and controllability. However, most methods still target single-person scenarios; in multi-person scenarios, they tend to synchronize identical motions or lip movements across all speakers, with restricted multi-person interaction.

In this work, AnyTalker explores the potential of learning multi-person speaking patterns from single-person data and designs an extensible multi-stream audio processing attention architecture, achieving a favorable balance among interactivity, identity scalability, and lip synchronization in the generated videos with low training data cost.

##### 2.2. Multi-person Video Generation

Multi-person video generation has advanced rapidly through specialized architectures, including portrait video generation [42, 62], dancing video generation [68], and talking video generation [23, 31, 63]. Within the audiodriven talking-head axis, Bind-your-Avatar [23] introduces a fine-grained Embedding Router that binds “who” with “what they speak”. InterActHuman [63] trains a mask predictor to identify which body regions to activate, enabling

#### 3. Method

The overall framework of the proposed AnyTalker is illustrated in Fig. 2. AnyTalker inherits certain architectural components from the Wan I2V model [59]. To handle multi-stream audio and identity input, we introduce a

trated in Eq. (4) and Fig. 2 (c), it enables flexible processing of diverse audio and identity inputs, with summed outputs from each iteration yielding the final attention output.

###### (a) Customized Audio-Face Attention Mask

4F + 1

Audio Token Modeling. We employ Wav2Vec2 [1] to encode audio features. The first latent frame attends to all audio tokens, whereas each subsequent latent frame focuses only on a local temporal window corresponding to four audio tokens. This structured alignment between video and audio streams is achieved by applying a Temporal Attention Mask Mtemporal, as explicitly shown in Fig. 3 (a). Furthermore, to enable comprehensive information integration, each audio token faudio utilized in the AFCA computation is concatenated with a face token fface encoded by ECLIP. This concatenation allows all video query tokens Qvideo to attend to different pairs of audio and face information effectively, as computed below:

- 0
- 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1

F+1

0

window size

: Video Token : Audio Token

: Face Token

| |
|---|

###### (b) Mask Token for Attention Output

”1” ”0” ”0” ”0” ”0” ”0”

| | |Reshape|
|---|---|---|
| | |Patchify &|

flatten x

Face Mask

/

: 1/0 Mask Token

| |
|---|

| |
|---|

| |
|---|

: Attention Output Token

- Figure 3. (a) Mapping of video tokens to audio tokens, facilitated by a custom attention mask. Every 4 audio tokens are bound to 1 video token, except for the first. (b) Mask token used for output masking in Audio-Face Cross Attention.

Kaf = Concat(faudio,fface) · WK, Vaf = Concat(faudio,fface) · WV ,

(2)

Attnout = MHCA(Qvideo,Kaf,Vaf,Mtemporal).

specialized multi-stream processing structure, termed as the Audio-Face Cross Attention (AFCA), which will be further described in Sec. 3.2. Our training pipeline is divided into two stages, which are summarized in Sec. 3.3.

Here, MHCA denotes Multi-Head Cross Attention, while WK and WV represent the key matrix and value matrix, respectively. The attention output Attnout will later be refined by the face mask token, as described in Eq. (3).

##### 3.1. Preliminaries

Face Token Modeling. The facial image is obtained by online cropping the first frame of the selected video clip using InsightFace [13] during training, while the facial mask Mface is precomputed offline to cover the maximum extent of the face mask across the entire video, i.e., the global face bounding box. This mask ensures that facial movements will never exceed this region, preventing the mask from incorrectly activating video tokens after the reshape and flatten operations shown in Fig. 3 (b), especially for videos with significant facial displacements. This mask, which shares the same dimensions as Attnout, can be directly employed for element-wise multiplication to compute the Audio-Face Cross Attention output, as formulated below:

As a DiT-based model, AnyTalker tokenizes the 3D VAE features fvideo through patchifying and flattening, whereas the text features ftext are generated by the T5 encoder [48]. Additionally, AnyTalker incorporates Reference Attention Layer, a cross-attention mechanism that leverages the CLIP image encoder [46] ECLIP to extract features fref from the first frame of the video. Wav2Vec2 [1] is also applied to extract the audio feature faudio. The overall input features finput can be written as

finput = [fvideo,ftext,fref,faudio]. (1)

Consistent with the Wan model, all attention layers are connected to the final output FFN layer (omitted in Fig. 2).

Mtoken = Patchify(Flatten(Mface)), AFCAout = Mtoken ⊙ Attnout.

##### 3.2. Audio-Face Cross Attention

(3)

To enable multi-person talking, the model must be capable of handling multi-stream audio inputs. Potential solutions may include the L-RoPE technique used in MultiTalk [31], which assigns unique labels and biases to different audio features. However, the range of these labels needs to be explicitly defined, which limits its scalability. Considering this, we design a more extensible structure to drive multiple IDs and enable accurate control in a scalable manner. As depicted in Fig. 2 (a) and (c), we introduce a specialized structure named Audio-Face Cross Attention (AFCA). The structure can iterate through a loop multiple times, contingent upon the number of input face-audio pairs. As illus-

Consequently, the hidden state Hi of each I2V DiT block can be formulated as

′

i = Hi + AFCA(1)out + ··· + AFCA(outn), (4)

H

where i represents the layer index of the attention block, and n denotes the number of IDs. Note that all AFCAout terms are produced by the same AFCA layer with shared parameters. The AFCA computation is applied n times iteratively, once for each individual. This architecture enables the number of drivable IDs to scale arbitrarily.

Motion = 0.42

[Figure 30]

- Motion = 0.62

[Figure 31]

Motion = 2.93

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

- Motion = 1.79

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

…

[Figure 40]

[Figure 41]

Motion = 0.58

[Figure 42]

[Figure 43]

[Figure 44]

…

spin

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

raise

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Motion = 0.36

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

…

[Figure 57]

[Figure 58]

[Figure 59]

High Interactivity Clip Low Interactivity Clip

- Figure 4. Two video clips from InteractiveEyes with Motion score (px): left shows original video, right shows cropped face and eye landmarks. Head turn toward the speaker or eyebrow raise will increase Motion and Interactivity; sustained stillness keeps both low.

##### 3.3. Training Strategy

AnyTalker explores the potential of single-person data for learning multi-person speaking patterns, with low-cost single-person data comprising the majority of training data. Single-Person Data Pretraining. We train the model using both standard single-person data and synthetic two-person data generated by horizontal concatenation. With an equal 50% probability, each batch of data is randomly configured to either two-person or single-person mode, as depicted in Fig. 2 (b). In two-person mode, each sample within the batch is horizontally concatenated with the data at the next index, along with its corresponding audio. This approach keeps the batch size identical across the two modes for every data batch. Additionally, we have predefined several generic text prompts that describe dual people speaking when data concatenation happens.

Although the above data construction strategy enhances the model’s ability to localize audio-visual features within local regions and learn speaking patterns of dual speakers, completely omitting the single-person data is not feasible. Doing so would significantly degrade the model’s performance on generating accurate lip movements, leading to unstable driving results, which we later discuss in Tab. 4.

Multi-person Data Refinement. In the next stage, we refine the model using a small amount of authentic multiperson data to enhance the interactivity within different IDs. Although our training data contains only interactions between two identities, we surprisingly find that our model equipped with the AFCA module naturally generalizes to scenarios with more than two IDs, as shown in Fig. 1. We speculate that this is because the AFCA mechanism enables

learning general patterns of human interaction, including not only accurate lip-syncing to the audio but also listening and responsive behaviors to other IDs’ speaking actions.

To construct high-quality multi-person training data, we construct a rigorous quality control pipeline, using InsightFace [12] to ensure two faces in most frames, audio diarization [45] to separate audio and ensure there is only one or two speakers, optical flow [27] to filter excessive motion, and Sync scores [10] to pair audio with faces. More details about this pipeline are in the supplementary material. This pipeline yields a total of 12 hours of high-quality dualperson data, which is a small amount compared to previous methods [31, 39, 63]. As the design of AnyTalker’s AFCA Layer inherently supports multi-ID inputs, two-person data is fed into the model in the same format as the concatenated data in the first stage, with no extra processing required.

To summarize, the single-person data training process enhances the model’s lip-syncing capability and generation quality, while also learning a generalized multi-person speaking pattern. Subsequently, lightweight multi-person data refinement compensates for the real interactions that cannot be fully covered by single-person data.

#### 4. Interactivity Evaluation

Despite progress, the prevailing evaluation benchmarks [65, 80, 81] for single-person talking head generation are inadequate for assessing natural interactions among characters. Although InterActHuman [63] introduces a comparable benchmark, its test set is limited to scenarios with only one speaker, which is not conducive to evaluating interactions among multiple characters. To fill this gap, we have

fore, evaluating during listening periods is more targeted and valuable. The lengths of the listening and speaking periods of each person are described in Fig. 5, denoted as L1,L2,L3,L4, respectively. To quantify the responsiveness of the generated avatars, we compute the average motion intensity during the listening phases L2 and L3:

L2

[Figure 60]

[Figure 61]

[Figure 62]

Time

[Figure 63]

L1

Speakerleft

[Figure 64]

[Figure 65]

[Figure 66]

L4

[Figure 67]

Time

Speakerright

L3

L2 · MotionL2 + L3 · MotionL3 L2 + L3

. (6)

Interactivity =

Figure 5. Listening and speaking periods of each speaker.

This metric measures the interactivity effectively in the generated multi-character videos. As Fig. 4 suggests, the proposed metric aligns well with human perception: static or sluggish eye movements receive low Motion scores, while head turns and eyebrow raises increase the score thus indicating higher interactivity. Moreover, to avoid misjudging abnormal eye movements, we implemented an exclusion algorithm detailed in the supplementary materials.

curated a collection of videos featuring two distinct identities, sourced from the web for evaluation purposes.

##### 4.1. Dataset Construction

We select interactive two-person videos to construct the video dataset, named InteractiveEyes. Two clips of these videos are illustrated in Fig. 4. Each video is approximately 10 seconds in duration and showcases exactly two faces throughout the entire segment. Furthermore, through a meticulous manual process, we segment the audio of each video to ensure that the majority of the videos capture scenes of both individuals engaging in speaking and listening, as well as a variety of rich eye interaction scenarios, as shown in Fig. 5. We have also ensured that each video includes instances of mutual gaze and head movements to provide authentic references.

#### 5. Experiments

Dataset. We expand single-person datasets [9, 11, 65, 80, 81] with internet-collected data, yielding roughly 1,000 hours for first-stage training, and also gather two-person conversation clips for the second-stage training, retaining only about 12 hours after filtering. Evaluations are conducted on two types of benchmarks: (i) standard talkinghead benchmarks HDTF [80] and VFHQ [65], and (ii) our self-collected multi-person conversation dataset (head-andbody, both identities speak). We then select 20 videos from each benchmark, rigorously ensuring that their identities do not appear in the training set.

##### 4.2. Proposed Interactivity Metric

In addition to this dataset, we introduce a novel metric, the eye-focused Interactivity, designed to assess the natural interaction between speakers and listeners. Since eye interaction is a fundamental and spontaneous behavior in conversational contexts, we use it as a key indicator of interactivity. Drawing inspiration from the Hand Keypoint Variance (HKV) metric employed in CyberHost [34], we propose a quantitative evaluation of the interaction by tracking the motion amplitude of eye keypoints.

Implement Details. To comprehensively evaluate our method, we train two models of different sizes: Wan2.11.3B-Inp and Wan2.1-I2V-14B [59], which serve as the foundational video diffusion models for our experiments. In all stages, the text [48], audio [1], and image [46] encoders, as well as the 3D VAE, remain frozen. The DiT main network, including the newly added AFCA layers, has all its parameters open for training. Stage 1 pretrains at 2 × 10−5 learning rate; stage 2 fine-tunes at 5×10−6. All models are optimized with AdamW [36] on 32 NVIDIA H200 GPUs.

To achieve this, we define Motion on the sequence of face-aligned eye keypoints extracted from the generated frames, where S denotes the frame sequence and E the eye keypoints. The Motion is calculated as follows

Evaluation Metrics. For the single-person benchmark, we employ several commonly used metrics: the Fr´echet Inception Distance (FID) [18] and the Fr´echet Video Distance (FVD) [57] to assess the quality of the generated data, SyncC [10] to measure the synchronization between audio and lip movements, and ID similarity [12] calculated between the first frame and the remaining frames.

  1

 .

|S|−1

|E|

1 |S| − 1

|Ei,j+1 − Ei,j|

Motion =

|E|

j=1

i=1

(5) Here, i and j denote the eye keypoint index and the frame index, while Ei,j denotes eye keypoints present in each frame. This formula intuitively computes the displacement and rotation of the eye region. We then calculate the motion during listening periods. The reason is, most generation methods perform well when activating the speaking subject, but the listening subject often appears rigid. There-

For the multi-person benchmark, we evaluate from different dimensions. The newly introduced metric, termed Interactivity, serves as the primary metric for assessment. For the FVD metric, the calculation is similar to that in the single-person benchmark. For the Sync-C metric, we refine

- Table 1. Quantitative comparison with other competing methods on HDTF [80] and VFHQ [65] benchmark. Here, OmniHuman-1.5∗ [26] refers to its “Master Mode” version accessed via the JiMeng platform [5], which currently does not support multi-person generation.

Method

Supported Generation Scope HDTF VFHQ multi-person body Sync-C↑ FID↓ FVD↓ ID↑ Sync-C↑ FID↓ FVD↓ ID↑

AniPortrait [64] × × 3.44 18.74 241.84 0.94 2.63 28.54 269.24 0.95 FantasyTalking [61] × ✓ 3.97 14.93 166.79 0.93 3.57 24.83 272.13 0.94 StableAvatar [56] × ✓ 4.11 14.67 166.44 0.91 3.53 22.91 275.73 0.88 EchoMimic [8] × × 5.23 61.53 381.55 0.95 4.87 58.72 486.75 0.86 Hallo3 [11] × × 7.53 17.12 195.61 0.91 6.32 41.26 371.24 0.91 Sonic [24] × × 7.81 52.96 286.12 0.95 7.71 36.68 385.37 0.89 OmniHuman-1.5∗ [26] × ✓ 7.23 35.26 173.23 0.90 7.67 35.36 283.39 0.90 MuitiTalk [31] ✓ ✓ 8.91 13.54 162.58 0.93 7.77 24.25 243.66 0.94

AnyTalker-1.3B ✓ ✓ 6.85 14.47 218.01 0.91 5.81 21.88 267.08 0.91 AnyTalker-14B ✓ ✓ 9.05 13.84 160.87 0.94 7.79 20.99 290.73 0.94

- Table 2. Quantitative comparison with other competing methods on the multi-person benchmark, InteractiveEyes.

Moreover, the 1.3B model of AnyTalker significantly outperforms AniPortrait [64], EchoMimic [8], and StableAvatar [56] in terms of lip synchronization, even though they have a similar number of parameters. These results demonstrate the excellent and comprehensive driving capabilities of the AnyTalker framework.

Method Interactivity↑ Sync-C∗↑ FVD↓ Ground Truth 0.77 6.01 0

Bind-Your-Avatar [23] 0.45 3.03 695.58

Subsequently, we evaluate AnyTalker’s ability to drive multiple IDs while maintaining both accurate lip synchronization and natural interactivity using the multi-person dataset, InteractiveEyes, described in Sec. 5, along with relevant metrics. In this comparison, we contrast AnyTalker with the available open-source multi-person driving methods, MultiTalk [31] and Bind-Your-Avatar [23]. The results depicted in Tab. 2 demonstrate that both the 1.3B and 14B models of AnyTalker achieve the best performance in terms of the Interactivity metric. Additionally, the 14B model achieves the best results across all metrics, thereby validating the effectiveness of our proposed training pipeline. We further illustrate AnyTalker’s capability to generate videos rich in interactivity through quantitative evaluation.

MuitiTalk [31] 0.49 6.88 500.03 AnyTalker-1.3B 0.97 4.56 467.84 AnyTalker-14B 1.01 6.99 424.15

its calculation as Sync-C∗ to focus only on the lip synchronization during each character’s speaking periods, thereby avoiding the influence of long listening segments on the final lip synchronization score, specifically,

L1 · Sync-CL1 + L4 · Sync-CL4 L1 + L4

Sync-C∗ =

. (7)

Here, L1 and L4 denotes the speaking phases depicted in Fig. 5.

Comparsion Methods. We compare AnyTalker with several state-of-the-art talking video generation methods. For single-person generation, we compare with AniPortrait [64], EchoMimic [8], Hallo3 [11], Sonic [24], FantasyTalking [61], StableAvatar [56], OmniHuman-1.5 [26], and MultiTalk [31]. For multi-person generation, we choose Bind-Your-Avatar [23] and MultiTalk [31] for quantitative and qualitative comparison.

Qualitative Comparsion. We then select an authentic human input from the InteractiveEyes dataset and use an input generated by an AIGC mode [54], both accompanied by corresponding text prompts and dual audio streams, to conduct a quantitative evaluation comparison using BindYour-Avatar [23], MultiTalk [31], and AnyTalker. As shown in Fig. 6, AnyTalker generates more natural videos with eye and head interactions compared to the other methods. MultiTalk exhibits weaker eye interaction, while Bind-YourAvatar tends to produce more static expressions. This trend further validates the effectiveness of the Interactivity metric proposed in Sec. 5. AnyTalker not only generates natural, two-person interactive speaking scenarios but also scales well to multiple IDs, as demonstrated in Fig. 1, where it effectively handles interactions among four IDs. The qualitative results of the single-person benchmarks [65, 80] will

##### 5.1. Comparison with SOTA methods

Quantitative Comparison. To begin with, we compare AnyTalker with several single-person generation methods to verify its excellent single-person driving capability. The quantitative results are shown in Tab. 1. Despite not being specifically designed for driving talking faces, AnyTalker achieves the best or competitive results across all metrics.

###### Real Human Input Case AIGC Input Case

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

AnyTalker(ours)MultiTalkBind-Your-Avatar

[Figure 72]

These people are talking to each other.

Input Prompt

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

|[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]|
|---|

Reference Images

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Audio Streams

Figure 6. Qualitative comparison of multiple multi-person driving methods. With the same text prompt, reference images, and multiple audio streams as input, we compare the generation results of Bind-Your-Avatar, MultiTalk, and AnyTalker. The left case uses the input image from the InteractiveEyes dataset, while the right case uses the image produced by a text-to-image generative model [54].

- Table 3. Ablation study about AnyTalker’s components on the HDTF dataset using the 1.3B model. “Baseline” denotes the model equipped solely with the basic audio attention layers. “AFCA” indicates the inclusion of the Audio-Face Cross Attention mechanism. “Single” indicates that authentic multi-person data is not utilized in this stage.

Metrics Sync-C↑ FID↓ FVD↓ ID↑

Setting

Baseline 5.42 13.01 170.96 0.90

w/o AFCA 6.71 14.97 207.47 0.88 w/o Mask Token 5.84 14.81 193.78 0.89

w/o Concatenated Data 6.21 14.73 202.01 0.91 AnyTalker-1.3B (Single) 6.97 15.58 166.27 0.91

be included in the supplementary material.

##### 5.2. Ablation Study

Components. We conduct ablation studies on the three important components mentioned in Fig. 2, including AudioFace Cross Attention, Mask Token for attention output, and concatenated multi-person data. Starting from the full 1.3B model in the first stage, which utilizes only single-person data, we progressively remove the relevant components to evaluate their impact on lip synchronization, generation quality, and identity similarity in the generated videos. The results in Tab. 3 show that every component plays a significant role, with concatenated multi-person data most beneficial to lip-movement accuracy. Although the completed model in the initial phase exhibits a marginally higher FID score compared to the Baseline model, we attribute this to the Baseline model’s propensity for generating talking videos with minimal facial expressions or head movements.

Table 4. Ablation studies conducted on the InteractiveEyes dataset using the 1.3B model. “RS” denotes the use of authentic singleperson data in the first stage. “CM” indicates concatenated multiperson data in the first stage. “RM” represents authentic multiperson data in the second stage.

Setting Metrics RS CM RM Interactivity↑ Sync-C∗↑ FVD↓

× ✓ × 0.55 3.21 672.18 ✓ × × 0.47 4.13 475.31 ✓ ✓ × 0.58 4.89 393.86

✓ × ✓ 0.71 3.63 511.51 ✓ ✓ ✓ 0.97 4.56 467.84

Conversely, the completed model generates more dynamic actions, which inherently influence the FID calculation to some degree. We regard this trade-off as a natural consequence of the model’s design. Furthermore, the completed model demonstrates superior performance over the Baseline model across other evaluation metrics. Crucially, the firststage model already has the basic capability to drive multiple individuals, a feature that the Baseline model lacks.

Multi-Person Data. Subsequently, we focus on multiperson data and conduct more targeted ablation experiments on the InteractiveEyes benchmark. In the first stage, we create concatenated multi-person data using single-person data. As shown in Tab. 4, models utilizing this concatenated data outperform those without it across all evaluation dimensions, especially in lip synchronization and interactivity, highlighting the role of concatenated data in learning multi-person speaking patterns. It is worth noting that the results in the first row of the table also demonstrate the necessity of mixing single-person data in the first stage; oth-

erwise, the generated results would be unstable.

After fine-tuning with authentic multi-person data, models that use concatenated data in the first stage demonstrate even more superior performance, further proving their early adaptation to multi-person speaking patterns. Although fine-tuning with authentic multi-person data leads to a slight decrease in lip synchronization, it significantly improves interactivity, which we consider a reasonable trade-off.

#### 6. Conclusion

In this paper, we introduce AnyTalker, an audio-driven framework for generating multi-person talking videos. It presents an extensible multi-stream processing structure called Audio-Face Cross Attention that enables identity scaling while guaranteeing seamless cross-identity interactions. We further propose a generalizable training strategy that maximally leverages single-person data through concatenation-based augmentation for learning multiperson speaking patterns. Additionally, we propose the first interactivity evaluation metric and a tailored benchmark for comprehensive assessment. Extensive experiments suggest that AnyTalker balances lip synchronization, identity scalability, and interactivity in multi-person scenarios.

#### References

- [1] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33:12449–12460, 2020. 3, 4, 6, 1
- [2] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22875–22889, 2025. 6
- [3] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. 6
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [5] ByteDance. Jimeng platform. https://jimeng. jianying . com / ai - tool / home ? type = digitalHuman, 2025. 7
- [6] Ming Chen, Liyuan Cui, Wenyuan Zhang, Haoxian Zhang, Yan Zhou, Xiaohan Li, Songlin Tang, Jiwen Liu, Borui Liao, Hejia Chen, et al. Midas: Multimodal interactive digitalhuman synthesis via real-time autoregressive video generation. arXiv preprint arXiv:2508.19320, 2025. 2

- [7] Yi Chen, Sen Liang, Zixiang Zhou, Ziyao Huang, Yifeng Ma, Junshu Tang, Qin Lin, Yuan Zhou, and Qinglin Lu. Hunyuanvideo-avatar: High-fidelity audio-driven human animation for multiple characters. arXiv preprint arXiv:2505.20156, 2025. 2, 3
- [8] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, and Chenguang Ma. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2403–2410, 2025. 2, 7, 4
- [9] J Chung, A Nagrani, and A Zisserman. Voxceleb2: Deep speaker recognition. Interspeech 2018, 2018. 2, 6
- [10] Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Asian Conference on Computer Vision, pages 251–263. Springer, 2016. 5, 6
- [11] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21086–21095, 2025. 2, 3, 6, 7
- [12] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4690–4699, 2019. 5, 6, 1
- [13] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multilevel face localisation in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5203–5212, 2020. 4, 1, 2
- [14] Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, and Steven Hoi. Omniavatar: Efficient audio-driven avatar video generation with adaptive body animation. arXiv preprint arXiv:2506.18866, 2025. 2, 3
- [15] Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, et al. Wan-s2v: Audio-driven cinematic video generation. arXiv preprint arXiv:2508.18621, 2025. 2, 3
- [16] Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168, 2024. 3
- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. In The Twelfth International Conference on Learning Representations, 2024. 2
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems, 30, 2017. 6
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 2, 3

- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2
- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 6
- [22] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 3
- [23] Yubo Huang, Weiqiang Wang, Sirui Zhao, Tong Xu, Lin Liu, and Enhong Chen. Bind-your-avatar: Multi-talkingcharacter video generation with dynamic 3d-mask-based embedding router. arXiv preprint arXiv:2506.19833, 2025. 2, 3, 7, 4
- [24] Xiaozhong Ji, Xiaobin Hu, Zhihong Xu, Junwei Zhu, Chuming Lin, Qingdong He, Jiangning Zhang, Donghao Luo, Yi Chen, Qin Lin, et al. Sonic: Shifting focus to global audio perception in portrait animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 193–203, 2025. 2, 3, 7, 4
- [25] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3
- [26] Jianwen Jiang, Weihong Zeng, Zerong Zheng, Jiaqi Yang, Chao Liang, Wang Liao, Han Liang, Yuan Zhang, and Mingyuan Gao. Omnihuman-1.5: Instilling an active mind in avatars via cognitive simulation. arXiv preprint arXiv:2508.19209, 2025. 2, 3, 7, 6
- [27] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European Conference on Computer Vision, pages 18–35. Springer, 2024. 5, 1
- [28] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 3
- [29] Xianghao Kong, Hansheng Chen, Yuwei Guo, Lvmin Zhang, Gordon Wetzstein, Maneesh Agrawala, and Anyi Rao. Taming flow-based i2v models for creative video editing. arXiv preprint arXiv:2509.21917, 2025. 6
- [30] Xianghao Kong, Qiaosong Qi, Yuanbin Wang, Anyi Rao, Biaolong Chen, Aixi Zhang, Si Liu, and Hao Jiang. Profashion: Prototype-guided fashion video generation with multiple reference images. arXiv preprint arXiv:2505.06537,

2025. 3

- [31] Zhe Kong, Feng Gao, Yong Zhang, Zhuoliang Kang, Xiaoming Wei, Xunliang Cai, Guanying Chen, and Wenhan Luo. Let them talk: Audio-driven multi-person conversational video generation. arXiv preprint arXiv:2505.22647,

2025. 2, 3, 4, 5, 7, 6

- [32] Chunyu Li, Chao Zhang, Weikai Xu, Jingyu Lin, Jinghui Xie, Weiguo Feng, Bingyue Peng, Cunjian Chen, and Wei-

- wei Xing. Latentsync: Taming audio-conditioned latent diffusion models for lip sync with syncnet supervision. arXiv preprint arXiv:2412.09262, 2024. 1, 2
- [33] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7752–7762, 2025. 2
- [34] Gaojie Lin, Jianwen Jiang, Chao Liang, Tianyun Zhong, Jiaqi Yang, Zerong Zheng, and Yanbo Zheng. Cyberhost: A one-stage diffusion framework for audio-driven talking body generation. In The Thirteenth International Conference on Learning Representations, 2025. 2, 6
- [35] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, Chao Liang, Yuan Zhang, and Jingtuo Liu. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13847–13858,

2025. 2, 3

- [36] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6, 3
- [37] Yanzuo Lu, Manlin Zhang, Andy J Ma, Xiaohua Xie, and Jianhuang Lai. Coarse-to-fine latent diffusion for poseguided person image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6420–6429, 2024. 6
- [38] Yanzuo Lu, Yuxi Ren, Xin Xia, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Andy J Ma, Xiaohua Xie, and Jian-Huang Lai. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16818–16829, 2025. 6
- [39] Xingpei Ma, Shenneng Huang, Jiaran Cai, Yuansheng Guan, Shen Zheng, Hanfeng Zhao, Qiang Zhang, and Shunsi Zhang. Playmate2: Training-free multi-character audiodriven animation via diffusion transformer with reward feedback. arXiv preprint arXiv:2510.12089, 2025. 3, 5
- [40] Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, and Chenguang Ma. Echomimicv3: 1.3 b parameters are all you need for unified multi-modal and multi-task human animation. arXiv preprint arXiv:2507.03905, 2025. 3
- [41] Rang Meng, Xingyu Zhang, Yuming Li, and Chenguang Ma. Echomimicv2: Towards striking, simplified, and semibody human animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5489–5498,

- 2025. 2, 3, 4

[42] Xiangyu Meng, Zixian Zhang, Zhenghao Zhang, Junchao Liao, Long Qin, and Weizhi Wang. Identity-grpo: Optimizing multi-human identity-preserving video generation via reinforcement learning. arXiv preprint arXiv:2510.14256,

- 2025. 3

- [43] Fatemeh Nazarieh, Zhenhua Feng, Diptesh Kanojia, Muhammad Awais, and Josef Kittler. Portraittalk: Towards customizable one-shot audio-to-talking face generation. arXiv preprint arXiv:2412.07754, 2024. 2
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF inter-

national conference on computer vision, pages 4195–4205,

2023. 2, 3

- [45] Alexis Plaquet and Herv´e Bredin. Powerset multi-class cross entropy loss for neural speaker diarization. In Proc. INTERSPEECH 2023, 2023. 5, 1, 2
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 4, 6, 3
- [47] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023. 3, 1
- [48] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 4, 6, 1, 3
- [49] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. Advances in Neural Information Processing Systems, 37:117340–117362, 2024. 6
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2, 3
- [51] Steffen Schneider, Alexei Baevski, Ronan Collobert, and Michael Auli. wav2vec: Unsupervised pre-training for speech recognition. arXiv preprint arXiv:1904.05862, 2019. 3, 1, 2
- [52] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 3

- [53] Naoya Takahashi and Yuki Mitsufuji. Multi-scale multiband densenets for audio source separation. In 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pages 21–25. IEEE, 2017. 1
- [54] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint,

2024. 7, 8

- [55] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244–260. Springer, 2024. 2, 3
- [56] Shuyuan Tu, Yueming Pan, Yinming Huang, Xintong Han, Zhen Xing, Qi Dai, Chong Luo, Zuxuan Wu, and Yu-Gang Jiang. Stableavatar: Infinite-length audio-driven avatar video generation. arXiv preprint arXiv:2508.08248, 2025. 3, 7, 4
- [57] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 6

- [58] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017. 3
- [59] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 6
- [60] Cong Wang, Kuan Tian, Jun Zhang, Yonghang Guan, Feng Luo, Fei Shen, Zhiwei Jiang, Qing Gu, Xiao Han, and Wei Yang. V-express: Conditional dropout for progressive training of portrait video generation. arXiv preprint arXiv:2406.02511, 2024. 2
- [61] Mengchao Wang, Qiang Wang, Fan Jiang, Yaqi Fan, Yunpeng Zhang, Yonggang Qi, Kun Zhao, and Mu Xu. Fantasytalking: Realistic talking portrait generation via coherent motion synthesis. Proceedings of the 33th ACM International Conference on Multimedia, 2025. 7, 3, 4
- [62] Qiang Wang, Mengchao Wang, Fan Jiang, Yaqi Fan, Yonggang Qi, and Mu Xu. Fantasyportrait: Enhancing multicharacter portrait animation with expression-augmented diffusion transformers. arXiv preprint arXiv:2507.12956, 2025. 3
- [63] Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, and Dahua Lin. Interacthuman: Multi-concept human animation with layoutaligned audio conditions. arXiv preprint arXiv:2506.09984,

2025. 2, 3, 5

- [64] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024. 7
- [65] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan. Vfhq: A high-quality dataset and benchmark for video face super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 657–666, 2022. 2, 5, 6, 7, 4
- [66] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 2
- [67] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, and Baining Guo. Vasa-1: Lifelike audio-driven talking faces generated in real time. Advances in Neural Information Processing Systems, 37:660–684, 2024. 3
- [68] Jingyun Xue, Hongfa Wang, Qi Tian, Yue Ma, Andong Wang, Zhiyuan Zhao, Shaobo Min, Wenzhe Zhao, Kaihao Zhang, Heung-Yeung Shum, et al. Towards multiple character image animation through enhancing implicit decoupling. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [69] Shaoshu Yang, Zhe Kong, Feng Gao, Meng Cheng, Xiangyu Liu, Yong Zhang, Zhuoliang Kang, Wenhan Luo, Xunliang Cai, Ran He, et al. Infinitetalk: Audio-driven video generation for sparse-frame video dubbing. arXiv preprint arXiv:2508.14033, 2025. 2, 3

- [70] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3
- [71] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025. 6
- [72] Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, and Wenhan Luo. Stylemaster: Stylize your video with artistic generation and translation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2630–2640, 2025. 6
- [73] Phyo Thet Yee, Dimitrios Kollias, Sudeepta Mishra, and Abhinav Dhall. Synchrorama: Lip-synchronized and emotionaware talking face generation via multi-modal emotion embedding. arXiv preprint arXiv:2509.19965, 2025. 2
- [74] Delong Zhang, Qiwei Huang, Yang Sun, Yuanliu Liu, WeiShi Zheng, Pengfei Xiong, and Wei Zhang. Learning implicit features with flow-infused transformations for realistic virtual try-on. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18736–18745, 2025. 3
- [75] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 2
- [76] Ruihan Zhang, Borou Yu, Jiajian Min, Yetong Xin, Zheng Wei, Juncheng Nemo Shi, Mingzhen Huang, Xianghao Kong, Nix Liu Xin, Shanshan Jiang, Praagya Bahuguna, Mark Chan, Khushi Hora, Lijian Yang, Yongqi Liang, Runhe Bian, Yunlei Liu, Isabela Campillo Valencia, Patricia Morales Tredinick, Ilia Kozlov, Sijia Jiang, Peiwen Huang, Na Chen, Xuanxuan Liu, and Anyi Rao. Generative ai for film creation: A survey of recent advances. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 6266–6278,

2025. 3

- [77] Ruicheng Zhang, Jun Zhou, Zunnan Xu, Zihao Liu, Jiehui Huang, Mingyang Zhang, Yu Sun, and Xiu Li. Zero-shot 3daware trajectory-guided image-to-video generation via testtime training. arXiv preprint arXiv:2509.06723, 2025. 6
- [78] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8652–8661, 2023. 3
- [79] Yue Zhang, Zhizhou Zhong, Minhao Liu, Zhaokang Chen, Bin Wu, Yubin Zeng, Chao Zhan, Yingjie He, Junxin Huang, and Wenjiang Zhou. Musetalk: Real-time high-fidelity video dubbing via spatio-temporal sampling. arXiv preprint arXiv:2410.10122, 2024. 3
- [80] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with

- a high-resolution audio-visual dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3661–3670, 2021. 2, 5, 6, 7, 4
- [81] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. Celebvhq: A large-scale video facial attributes dataset. In European Conference on Computer Vision, pages 650–667. Springer,

- 2022. 2, 5, 6

[82] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4606–4615,

- 2023. 3

## AnyTalker: Scaling Multi-Person Talking Video Generation with Interactivity Refinement

### Supplementary Material

[Figure 87]

[Figure 88]

Audioleft Audioright

left-left left-right

[Figure 89]

[Figure 90]

right-left right-right Speaker

Speakerleft

right

SyncNet Score Matrix

Figure 7. SyncNet Score Matrix in two-person data.

#### Outline

This supplementary material provides two sets of additional information on AnyTalker.

##### A. Experimental Details

- • Data-processing pipeline used during training
- • Construction scheme for inference data
- • Training hyper-parameters and other technical details
- • Inference settings B. Extended Experiments
- • More analysis of the Interactivity metric
- • Additional experimental results
- • Effectiveness of interactivity refinement A. Experimental Details

##### A.1. Training Data Processing

Auxiliary Models. We first introduce the auxiliary models used during the data processing stage. To segment sentences and chunk the audio track extracted from each video, we employ the pre-trained speaker-diarization3.1 model [45] in conjunction with OpenAI’s Whisper [47]. For vocal separation and vocal feature extraction, we use pre-trained models from Kim Vocal 2 [53] and Wav2Vec2 [1, 51]. For face detection and facial feature processing, we adopt RetinaFace [13] and ArcFace [12],

https : / / huggingface . co / pyannote / speaker -

diarization-3.1 https://github.com/openai/whisper https : / / github . com / Anjok07 /

ultimatevocalremovergui

https://huggingface.co/facebook/wav2vec2-base960h

video_path: … n_frames: … width: … height: … audio_emb_path: […] text_emb_path: […] face_bbox_path: […] SyncNet_confidence: […]

Figure 8. Properties of training data.

both accessible via InsightFace. To measure audio-visual synchronization, we employ the pre-trained SyncNet model from LatentSync [32]. Text prompts are obtained with Gemini 2.5 Pro, and their features are extracted using the T5-encoder [48]. Finally, we filter videos with excessive camera motion by means of the CoTracker model [27].

Single-Person Data. A valid single-person clip is a 24-fps video that continuously exhibits one identifiable face and is accompanied by speech audio whose lip movements are perfectly synchronised with the soundtrack. We first eliminate hand-held recordings exhibiting pronounced camera shake or abrupt scene changes using CoTracker [27]. RetinaFace [13] is then applied to guarantee that most frame contains exactly one face. Videos that satisfy the above criteria are processed by speaker-diarization-3.1 [45] to obtain coarse sentence-level segments. Utterances longer than 5s are further subdivided into semantically coherent units with Whisper [47], after which any fragment containing overlapping speakers is discarded. The remaining clean single-speaker segments average ≈2s in duration. Because 2s clips cannot saturate GPU memory, we randomly concatenate consecutive segments to extend each sequence; the resulting clip lengths follow a Gaussian distribution with mean 4s and standard deviation 0.5s. Then, their vocal tracks are extracted with Kim Vocal 2 [53] and en-

https://github.com/deepinsight/insightface https://huggingface.co/ByteDance/LatentSync/

tree/main https://deepmind.google/models/gemini/pro/ https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-

720P/tree/main/google/umt5-xxl https://huggingface.co/facebook/cotracker

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Figure 9. Two cases from InteractiveEyes. Both of the speakers have speaking periods.

[Figure 95]

[Figure 96]

[Figure 97]

Figure 10. Input cases from the benchmark dataset, from left to right: HDTF [80], VFHQ [65], and EMTD [41].

coded with Wav2Vec-2 [51], while audio-visual synchrony is scored by SyncNet [32]. Clips whose synchrony score lies below a pre-defined threshold are removed from the training set. Finally, every retained clip is transcribed and annotated with frame-level face bounding boxes. We then use Gemini 2.5 Pro to generate textual descriptions for these clips and employ a T5 encoder to extract text features. All input text is uniformly truncated or padded to exactly 512 tokens. After being preceded by the above pipeline, it yields approximately 1,000 h of high-quality 480P single-person data.

Two-Person Data. Processing Two-person clips follows the single-person pipeline with three key modifications.

- 1. Face count: the video must contain exactly two faces in most frame.
- 2. Speaker activity: speaker-diarization-3.1 [45] is constrained to output only two valid states: (i) both speakers active or (ii) a single speaker active.
- 3. Spatial consistency: the left–right spatial ordering of the two faces must remain unchanged throughout the entire clip; identity swapping is detected and rejected via InsightFace [13].

To establish the correct voice–face correspondence, we compute a 2×2 SyncNet confidence matrix (Fig. 7) and requires the two largest scores to lie on the diagonal; otherwise, the clip is discarded. A minimum synchrony threshold identical to the single-person setting is further applied. After filtering, approximately 12 h of clean two-person data are retained.

Data Concatenation. During the first-stage training, we horizontally concatenate randomly selected single-person clips. Because the original videos are extremely highresolution (many 2K/4K), naively resizing them along the height axis and stacking would yield faces that occupy only

a tiny fraction of the frame. To avoid this, we adopt a special cropping strategy. First, we locate the face centre in each of the two candidate clips. For 480P footage, the frame size is (H, W)=(480,832), so we expand a minimal crop of (480,416) around each face centre. We then apply further augmentation: while keeping the crop inside the original image, we randomly enlarge the window while preserving the 480/416 aspect ratio. The resulting crops contain a much larger facial region, enabling the model to learn a more accurate lip-to-audio mapping.

Data Properties. Each processed sample is summarised by a dictionary-style item that stores the absolute paths to the decoded video, cleaned audio, and pre-extracted features. During training, the data loader indexes all information through this item. Single-person and two-person clips share an identical key structure; the number of face–speech entries in the list unambiguously indicates whether the sample contains one or two speakers. Further details are illustrated in Fig. 8.

##### A.2. Benchmark Data Processing

Single-Person Data. We adopt HDTF [80] and VFHQ [65] as the single-speaker evaluation benchmarks. Both datasets are de facto standards for talking-head generation; their images are tightly cropped around the face region. Additional experiments on half-body sequences that include hands are reported in Sec. B.2.

For each test video, we supply (i) a single reference image of the subject and (ii) the corresponding speech segment. To respect GPU-memory constraints, we fix the audio length to 6s—a duration that all competing methods process without overflow. Twenty clips are randomly selected from each dataset, ensuring that none of the identities appear in the AnyTalker training set. Reference images from the three evaluation splits are visualised in Fig. 10.

Two-Person Data. We provide additional details for InteractiveEyes, the test set introduced in Section 4 in the main text for two-speaker scenarios. Each video is shot with a stable camera and contains exactly two faces in every frame; the duration is approximately 10s. In 80% of the cases, both speakers produce speech, while in the remaining 20% only one speaker talks and the other maintains a listening pose. To avoid segmentation errors, we do not apply speaker-diarization-3.1 [45]; instead, speaking intervals

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Eyebrow Raise Head Turn

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Head Nod Gaze Shift

Figure 11. Four typical cases that will get a high Interactivity score.

Table 5. Quantitative result on EMTD [41] benchmark.

Metrics Sync-C↑ FID↓ FVD↓ ID↑

Method

FantasyTalking [61] 3.76 67.66 818.71 0.76 EchoMimic v2 [41] 6.27 63.43 671.18 0.76

MultiTalk [31] 8.38 64.71 787.99 0.79

AnyTalker-1.3B 5.83 56.01 789.59 0.74 AnyTalker-14B 8.45 50.61 664.58 0.77

are manually annotated so that the Interactivity metric proposed in the main paper can be computed accurately. Two representative cases are shown in Fig. 9.

##### A.3. Implement Details

Training Details. To comprehensively evaluate our method, we train two models of different sizes: Wan2.11.3B-Inp and Wan2.1-I2V-14B [59], which served as the foundational video diffusion models for our experiments. In all stages, the text [48], audio [1], and image [46] encoders, as well as the 3D VAE, remained frozen with their parameters unchanged. The DiT main network, including the newly added AFCA Layers, had all its parameters open for training. The first stage employs a higher learning rate of 2 × 10−5 for pretraining, while the second stage uses a lower learning rate of 5 × 10−6 for fine-tuning, incorporating a warm-up strategy and optimized using the AdamW optimizer [36]. The 14B model is trained using 32 NVIDIA H200 GPUs. In the first stage, the global batch size is set to 32 and the model is trained for 2.4M steps. In the second stage, the batch size is adjusted to 16, and the model is trained for an additional 50K steps. The 1.3B model is trained using 8 NVIDIA H200 GPUs. The global batch size

https://huggingface.co/alibaba-pai/Wan2.1-Fun1.3B-InP

is maintained at 48 throughout the training process, with the total number of training steps being consistent with that of the 14B model.

Inference Details. For all competing methods, we strictly follow the publicly released implementations and adopt their default recommended inference hyperparameters. Approaches that require textual input receive a fixed prompt for the single-person benchmark: “this person is talking”. For the multi-person benchmark, the prompts are automatically generated by Gemini 2.5 Pro as described in Sec. A.1. AnyTalker performs inference with classifierfree guidance (CFG) [19] using a guidance scale of 4.0; in the unconditional branch, both textual and audio features are set to zero. Face masks are extracted with InsightFace and uniformly dilated to provide a slightly enlarged region for generation.

#### B. Extended Experiments

##### B.1. More analysis about Interactivity Metric

Good Case. Fig. 11 visualises four representative listener behaviours that strongly signal conversational engagement: eyebrow raise, head nod, head turn, and gaze shift. The presence of such cues contributes substantially to the final Interactivity score. Because Interactivity is computed over the entire video, we do not report frame-level values in Fig. 11. As illustrated in Fig. 14, AnyTalker produces sequences rich in these interactive actions, and thus achieves a comparatively high Interactivity rating.

The Rubustness of Interactive Metric. Some generative baselines occasionally produce highly implausible motions. For example, Bind-Your-Avatar [23] generates an exaggerated “lying-down” action shown in Fig. 12. Without counter-measures, such artifacts would dramatically inflate the Motion score even though they are unrelated to interactivity. We therefore introduce a lightweight anomalysuppression rule: if the mean facial-landmark displacement

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Figure 12. A bad case generated by Bind-Your-Avatar [23].

Monica Escobar and I represent El Paso, Texas… the humanity and just even though in the world…

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

EchoMimic

StableAvatar

Sonic

MultiTalk

OmniHuman-1.5

AnyTalker (orus)

Figure 13. Qualitative results on HDTF [80] (left) and VFHQ [65] (right) benchmark. The positions of pronunciation have been highlighted in red and underlined.

between two consecutive frames exceeds 10 pixels (all faces are pre-aligned to a 256 × 256 canvas), the landmark positions are frozen until a subsequent displacement below 10 pixels is observed. As demonstrated in the right two frames of Fig. 12, this simple clamping prevents the vast majority of abnormal movements from entering the Interactivity computation.

##### B.2. Additional Experimental Results

Quantitative Results on the EMTD Benchmark. EMTD [41] is a half-body dataset that includes hands, as illustrated in Fig. 8. We compare AnyTalker against

three methods capable of generating half-body sequences: EchoMimicv2 [41], FantasyTalking [61], and MultiTalk [31]. The evaluation protocol strictly follows the metrics described in Sec.5.1 for the single-person benchmark. AnyTalker-14B achieves the best scores on three metrics and is only marginally behind MultiTalk on identity preservation (ID). These results demonstrate that AnyTalker is a comprehensive framework that handles not only tight-face inputs but also half-body scenarios.

Qualitative Result on Single-Person Benchmarks. As illustrated in Fig. 13, we present qualitative comparisons of EchoMimic [8], StableAvatar [56], Sonic [24], Mul-

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

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

###### Figure 14. More Results generated by AnyTalker.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Figure 15. Improvement in interaction among each identity after fine-tuning on authentic multi-person data (right).

tiTalk [31], OmniHuman-1.5 [26], and AnyTalker-14B on the HDTF [80] and VFHQ [65] benchmarks. AnyTalker consistently produces clear dentition and accurate lip movement.

More Multi-Person Results. Fig. 14 showcases additional multi-person animations generated by AnyTalker. The model gracefully handles a broad spectrum of inputs: real photographs, AIGC images, and cartoons. It produces natural, context-appropriate interactions regardless of the number of identities involved. Further compelling examples are available in the videos on the project homepage.

##### B.3. Effectiveness of Interactivity Refinement

As shown in Fig. 15, refining interactivity with authentic multi-person data lets identities exchange significantly more natural eye contact. A model trained solely on singleperson data can activate each face correctly, yet every iden-

tity remains blank whenever it is not speaking, an effect that looks highly unnatural in conversational scenes.

##### B.4. Future Work

At present, AnyTalker supports only rudimentary camera motions driven by textual prompts. Drawing inspiration from recent controllable video-generation approaches [37, 71, 72], we can incorporate additional conditional signals, such as camera trajectory [3]. By grafting lightweight, training-efficient modules [21, 29, 38, 49] into recent camera-trajectory-control techniques [2, 77], we expect to enrich the visual storytelling of the generated videos, automatically framing and tracking the active speaker without manual intervention.

