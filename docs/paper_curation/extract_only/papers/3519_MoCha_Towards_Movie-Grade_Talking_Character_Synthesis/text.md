

# arXiv:2503.23307v1[cs.CV]30Mar2025

## MoCha: Towards Movie-Grade Talking Character Synthesis

Cong Wei1,2, Bo Sun2, Haoyu Ma2, Ji Hou2, Felix Juefei-Xu2, Zecheng He2, Xiaoliang Dai2, Luxin Zhang2, Kunpeng Li2, Tingbo Hou2, Animesh Sinha2, Peter Vajda2, Wenhu Chen1 1University of Waterloo, 2GenAI, Meta https://congwei1230.github.io/MoCha

Talking Character

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

“Close-up shot of a doctor in a white lab coat over blue scrubs, speaking…”

Scene Control

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

“Close-up shot of a young woman sitting in the driver's seat of a car with the window rolled down…”

Emotion Control

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

“A woman standing in a modern house. Two distinct streams of tears trail down her cheeks as she speaks with an angry expression…”

###### Action Control

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

“A medium shot of a man interacting warmly with an elephant. the man talks to the camera…"

###### Multi-Character Talk

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

“A medium close-up shot of man puts a ring on a woman's finger. They speak to each other…”

###### Multi-Character Turn-based Talk

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

“Two video clips. Person1: Woman with short brown hair… Person2: Man with curly hair… First clip: Person1 near a circular window inside a space station… Second clip: Person2 in the same cabin…”

Figure 1. MoCha is an end-to-end talking character video generation model that takes only speech and text as input, without requiring any auxiliary conditions. More videos are available on our website: https://congwei1230.github.io/MoCha

[Figure 41]

[Figure 42]

[Figure 47]

[Figure 52]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

1

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

#### Abstract

Recent advancements in video generation have achieved impressive motion realism, yet they often overlook characterdriven storytelling, a crucial task for automated film, animation generation. We introduce Talking Characters, a more realistic task to generate talking character animations directly from speech and text. Unlike talking head, Talking Characters aims at generating the full portrait of one or more characters beyond the facial region. In this paper, we propose MoCha, the first of its kind to generate talking characters. To ensure precise synchronization between video and speech, we propose a speechvideo window attention mechanism that effectively aligns speech and video tokens. To address the scarcity of largescale speech-labeled video datasets, we introduce a joint training strategy that leverages both speech-labeled and text-labeled video data, significantly improving generalization across diverse character actions. We also design structured prompt templates with character tags, enabling, for the first time, multi-character conversation with turnbased dialogue—allowing AI-generated characters to engage in context-aware conversations with cinematic coherence. Extensive qualitative and quantitative evaluations, including human preference studies and benchmark comparisons, demonstrate that MoCha sets a new standard for AI-generated cinematic storytelling, achieving superior realism, expressiveness, controllability and generalization.

#### 1. Introduction

Automating film production holds immense commercial potential, promising to democratize cinematic-level storytelling by enabling content creators to effortlessly generate films through natural language [15, 36, 41, 47]. Ideally, creators should be able to specify rich narratives involving multiple characters—either realistic humans or stylized cartoons—that engage in meaningful dialogues, expressive emotional portrayals, synchronized speech, and realistic full-body actions. Crucially, talking characters serve as powerful mediums for delivering impactful messages, clearly communicating ideas, and deeply engaging audiences. In films especially, dialogue serves as a key vehicle to effectively convey narratives, with vast downstream applications including digital assistants, virtual avatars, advertising, and educational content.

However, existing video foundation models are far from achieving this vision. Despite significant advancements in visually compelling content and dynamic environments, models such as SoRA, Pika, Luma, Hailuo, and Kling [2, 3, 5–7, 19, 27, 42] primarily generate characters with limited speech capabilities. Typically, these models exhibit simplified mouth movements and emotional expressions de-

tached from meaningful dialogue, lacking control over actual spoken content. Consequently, their practical usability is severely restricted for speech-driven interactions essential to cinematic and interactive applications. On the other hand, recent speech-driven video generation methods, such as Loopy, Hallo3, and EMO [18, 24, 33–35, 40], predominantly focus on synthesizing talking-head videos confined to facial regions. These approaches neglect essential fullbody movements and multi-character interactions critical for expressive storytelling, thus significantly limiting their applicability in realistic and interactive cinematic scenarios.

To bridge these gaps, we introduce the novel task: Talking Characters, defined as generating characters from natural language and speech input that naturally express synchronized speech, realistic emotions and full-body actions. We further propose MoCha, the first-of-its-kind diffusion transformer (DiT) model trained end-to-end to achieve high-quality, movie-grade talking character generation.

MoCha introduces several key technical innovations tailored specifically for this task:

- • End-to-End Training Without Auxiliary Conditions: Unlike prior works such as EMO [33, 35], SONIC [17], Echomimicv2 [24], Loopy [18], and Hallo3 [40], which rely heavily on external control signals (e.g., reference images, skeletons, keypoints), MoCha is trained directly on text and speech without any auxiliary conditioning. This simplifies the model architecture and improves motion diversity and generalization.
- • Speech-Video Window Attention: We propose a novel attention mechanism that aligns speech and video inputs through localized temporal conditioning (see Sec. 3.2). This design significantly improves lip-sync accuracy and speech-video alignment.
- • Joint Speech-Text Training Strategy: To address the scarcity of large-scale speech-labeled video datasets, we introduce a joint training framework that leverages both speech-labeled and text-labeled video data. This strategy enhances the model’s ability to generalize across diverse character actions and enables universal controllability through natural language prompts, enabling nuanced control of character expressions, actions, interactions, and environments without auxiliary signals.
- • Multi-Character Conversation Generation: For the first time, MoCha enables coherent multi-character conversations in dynamic, turn-based dialogues, overcoming the single-character limitation of prior methods and supporting cinematic, story-driven video synthesis.

To evaluate MoCha’s performance, we curated MoChaBench, a benchmark tailored for Talking Characters generation tasks. Both human evaluations and automatic metrics demonstrate that MoCha set a new standard for talking character video generation and represents a significant step toward achieving controllable, narrative-driven video synthe-

[Figure 78]

[Figure 79]

[Figure 80]

|“A close-up shot of a woman standing in a modern house, speaking to the camera while<br><br>facing straight ahead … She has shoulder-length dark hair … The background features sleek, modern furniture … She holds a single rose in her hands, gently running her fingers along its petals … The static camera captures her face and upper body”|
|---|

3D VAE

[Figure 81]

[Figure 82]

Window Cross Attention

[Figure 83]

[Figure 84]

Text Encoder

[Figure 85]

[Figure 86]

Wav2Vec2

Cross Attention

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Projector

Projector

[Figure 91]

[Figure 92]

Self Attention

Speech Tokens

Text Tokens

###### xN

DiT block

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Speech

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Noisy Video Tokens

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

- Figure 2. MoCha Architecture. MoCha is a end-to-end Diffusion Transformer model that generates video frames from the joint conditioning of speech and text, without relying on any auxiliary signals. Both speech and text inputs are projected into token representations and aligned with video tokens through cross-attention.

sis, with broad applications in film production, animation, virtual assistants, and beyond.

5. Visual Quality: The entire video is visual consistency and temporal coherence without visual artifacts.

[Figure 120]

[Figure 121]

[Figure 122]

#### 3. Model: MoCha

[Figure 123]

[Figure 124]

[Figure 125]

#### 2. Task: Talking Characters

[Figure 126]

[Figure 127]

[Figure 128]

In this section, we introduce the MoCha model, the first model to generate talking characters. We begin by outlining its architecture in subsection 3.1, followed by the speechvideo window attention mechanism in subsection 3.2. Next, we describe the method of generating multiple clips in subsection 3.3. Finally we provide explanation of the training strategy in subsection 3.4.

We introduce the novel task of Talking Characters, which aims to generating characters from natural language and speech input that mimicking realistic human-like behaviors. In contrast to talking-head (Close-up shot), Talking Characters aims at generating digital characters at any camera shot size (From close-up shot to wide-shot), covering one or more characters beyond the facial region.

Input: A Talking Character system takes as input:

##### 3.1. Speech+Text to Video Diffusion Transformers

- 1. A text prompt describing the character, environment, actions, facing direction(optional), position in the frame(optional) and camera framing(optional).
- 2. A speech audio for driving the character mouth, facial expression and body motion.

Figure 2 presents the overall framework of MoCha. Unlike prior works that employ text-to-image (T2I) U-Net [10, 33, 35, 40] for talking head generation, MoCha is built on a diffusion transformer (DiT) [26]. By incorporating text and speech conditions sequentially via cross-attention, it effectively captures both semantics and temporal dynamics.

Output: The output is a video featuring one or more talking characters, which can be human, 3D cartoon, or animal. Evaluation: The generated characters are expected to perform well across the following five axes:

Model Architecture. Given an RGB video ν ∈ RT×H×W×3 with T frames, we encode it into a latent representation x0 ∈ Rτ×h×w×c using a 3D VAE, which downsamples the video spatially and temporally. We define the temporal down-sampling ratio as r = Tτ . Next, x0 is flattened into a sequence of tokens of size (τ × h × w) × c and passed to the DiT model fθ(·). Within each DiT block, the model first applies self-attention to the tokens, followed by sequential cross-attention with the text condition tokens c and audio condition tokens α. The audio condition α ∈ RT×c is derived from raw waveforms using Wav2Vec2 [1]

- 1. Lip-Sync Quality: Speak the provided audio with accurate and temporally aligned lip synchronization.
- 2. Facial Expression Naturalness: Express natural and coherent facial emotions that align with both the speech content and the text prompt.
- 3. Action Naturalness: Perform natural and fluid body movements corresponding to the actions described in the text, with gestures synchronized to the speech.
- 4. Text Alignment: Appear in a scene and context that are consistent with the descriptions provided in the prompt.

| |
|---|

| |
|---|

| |
|---|

f rame1

f rame5 f rame9

f rame12

Noisy Video

ν ∈ ℝ12×h×w×3

f rame1

[Figure 185]

Audio Token

Wav2Vec2 + Projector

[Figure 186]

α12

α4 α5 α8 α9 α12

α1

α

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

(Key/ Value)

[Figure 200]

- x1
- x2
- x3

[Figure 201]

f rame5

[Figure 202]

3D VAE

[Figure 203]

[Figure 204]

[Figure 205]

ν ∈ ℝ4×h×w×3

[Figure 206]

[Figure 207]

Temp Down

[Figure 209]

f rame9

Noisy Video Token x (Query)

r=4

[Figure 210]

[Figure 212]

[Figure 214]

f rame12

- Figure 3. MoCha’s Speech-Video Window Cross Attention MoCha generates all video frames in parallel using a window cross-attention mechanism, where each video token attends to a local window of audio tokens to improve alignment and lip-sync quality.

and processed through an single layer MLP to align its feature dimension with the latent video tokens.

- 1. Temporal Compression: Videos are compressed using a 3D VAE with a downsampling ratio r (typically r = 4 or 8 in modern T2V models [19, 27]), resulting in latent representations of length τ = T/r. While audio remains at the original resolution (T), video tokens operate at the compressed scale (τ), degrading lip synchronization.
- 2. Parallel Generation: Unlike autoregressive models, DiT generates all τ latent frames in parallel. However, na¨ıve cross-attention allows each video token to attend to all audio tokens. As a result, latent frames may incorrectly associate with phonemes from unrelated timesteps. To address this, we propose a Speech-Video Window

Attention mechanism that enforces localized conditioning. This design is motivated by the observation that lip movements depend on short-term audio cues (1–2 phonemes), whereas body motions align with longer-term text descriptions. To capture this distinction, we constrain each video token to attend only to a temporally bounded audio window. As illustrated in Figure 3, for each latent video frame x(i) ∈ Rh×w×c (i ∈ {1,...,τ}), attention is computed over audio tokens αj, where j spans:

j ∈ [max(1,(i − 1)r − 1),min(T,ir + 1)]. (4)

This window encompasses r + 2 audio tokens, covering the r frames corresponding to the latent x(i), plus one token on either side to ensure contextual continuity, thereby enhancing local smoothness between adjacent latents.

- 3.3. Multi-character Conversation

Training Objective. We adopt Flow Matching [21], which enables efficient simulation of continuous-time dynamics, to train our model. Given a latent video representation x1 ∈ Rτ×h×w×c (encoded from the input video), random noise ϵ ∼ N(0,I), and a continuous time step t ∈ [0,1], we construct an intermediate latent xt by interpolating between ϵ and x1:

xt = (1 − t)ϵ + tx1. (1)

The model is trained to predict the velocity, defined as the difference between the data and noise:

dxt dt

= x1 − ϵ. (2) The training loss is then:

vt =

2 2

L = Eϵ∼N(0,I), x

1, c, α, t∈[0,1] fθ xt,c,α,t − (x1 − ϵ)

,

(3) where x1 is the encoded latent video, c and α are text and audio conditions, and fθ(·) is the DiT model.

##### 3.2. Speech-Video Window Attention

Most talking head generation methods employ 2D diffusion models (e.g., U-Net) that auto-regressively generate T video frames conditioned on audio tokens α ∈ RT×c. When generating frame νi, the model is provided only with the corresponding audio token αi. This design inherently preserves speech-video synchronization, ensuring accurate alignment between lip movements and the corresponding speech. However, when using DiT-style architectures, two key differences emerge that disrupt this alignment:

MoCha generates multi-clip videos simultaneously in the same manner as single-clip generation, with no additional architectural modifications. As illustrated in Figure 4, instead of auto-regressive generation in video extension methods which requiring conditioning on previous generated results, MoCha leverages self-attention across video tokens to ensure consistency of characters appearing in multiple clips,

4

| |
|---|

[Figure 216]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

Video

Prompt

| |“ Two video clips. Characters:<br><br>- Person1: Woman with short grey hair, medium skin tone, wearing a maroon sweater and apron.<br>- Person2: Man with a beard, darker skin tone, wearing a black t-shirt and jeans.<br><br><br>First clip: Person1 stands in a cozy kitchen, facing left, speaking warmly while preparing food on the counter. Pots and pans hang in the background. The camera gradually zooms in on Person1’s face. Second clip: Person2 stands on the other side of the counter, facing left, smiling and replying to Person1. The camera remains static.”|
|---|---|
| | |

3D VAE 3D VAE 3D VAE

Text Encoder

Cross Attention

Projector

Window Cross Attention

Self Attention

VideoNoisyToken x1 x2 x3

Speech Token α1 α4 α5 α8 α9

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

Wav2Vec2 + MLP

Speech

Person1 Speech Person2 Speech

frame1 CLIP1 frame CLIP2 frame12

5 frame6

- Figure 4. Multi-character Conversation and Character Tagging. MoCha supports generates multi-character conversion with scene cut. We design a specialized prompt template: it first specifies the number of clips, then introduces the characters along with their descriptions and associated tags. Each clip is subsequently described using only the character tags, simplifying the prompt while preserving clarity. MoCha leverages self-attention across video tokens to ensures character and environment consistency. The audio conditioning signal implicitly guides the model on when to transition between clips.

- • “Two video clips” Specifies the number of clips.
- • “Characters” Introduces a list of characters, each described by visual attributes and assigned a unique tag [20] (e.g., Person1, Person2).
- • “First clip”, “Second clip”, etc. Each video segment is described using only the defined character tags.

as well as coherence in the surrounding environment. We assume that only one character speaks at a single time, so the changes in the speaker in the audio conditioning implicitly guide the MoCha on when to transition between clips without any additional guidance signal condition such as clip tokens [39].

Binding attributes and actions to the correct characters using only text is particularly challenging in multi-clip settings—especially when multiple characters interact or when the same character appears across clips. Naive captioning models typically rely on visual descriptions to refer to characters. As a result, they must repeat detailed appearance descriptions each time a character is mentioned, leading to long, redundant, and confusing prompts. For example:

This design significantly reduces redundancy and helps the model reliably associate visual attributes with character actions, even across multiple clips.

| |
|---|

[Figure 248]

##### 3.4. MoCha Training Strategy

Joint Training of (Speech+Text)-to-Video (ST2V) and Text-to-Video (T2V) Speech-annotated video datasets are significantly smaller in scale and less diverse compared to standard text-to-video (T2V) datasets, making it challenging to train high-quality (Speech+Text)-to-Video (ST2V) models directly. Relying solely on speech-annotated data limits the model’s ability to generalize across varied visual and semantic contexts. To address this, we propose a joint training strategy that integrates both speech-annotated and text-only video dataset:

The girl aged 10-15 in a yellow dress waves to another girl dressed in a green shirt with braided hair holding a book... The girl in the green shirt with braided hair responds with a smile... Nearby, another girl aged 10-15 in a blue hoodie points toward the whiteboard while looking at the girl dressed in a green shirt with braided hair...

- • 80% ST2V Data: The model is primarily trained on speech-conditioned video data, leveraging both speech and text modalities.
- • 20% T2V Data: To enhance diversity, we incorporate text-only video data, where speech conditioning is absent. In these cases, Wav2Vec2 embeddings are replaced with zero vectors, allowing the model to generalize across textonly prompts after training.

This verbosity not only increases the risk of exceeding token limits (e.g., 256 tokens) but also confuses the model during generation—especially in multi-clip scenarios.

As shown in Figure 4, we address this by introducing a structured prompt template with fixed keywords and a character tagging mechanism that promotes clarity, compactness, and consistency:

###### Multi-Stage Human Video Learning Speech condition-

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Prompt: “A close-up shot of a man sitting on a dark gray couch… Behind the man are three white cylindrical light fixtures with yellow lights inside them… the man continues to speak to the camera while he holds a lit cigar, the smoke curling gently into the air…”

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

Prompt: “A medium shot of a young man aged 25 to 35 is sitting in the living room in a leisurely environment… He is live-streaming, sitting in front of a desk with a laptop in front of him. His demeanor is relaxed and friendly, gesturing with his hands while speaking…”

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Prompt: “A close-up shot of a young blonde woman sitting in an airplane seat, facing slightly to the right as she talks on the phone with a worried expression. As the video progresses, she continues speaking and eventually turns to look out the window…”

- Figure 5. Qualitative results of MoCha on MoCha-Bench. MoCha not only generates lip movements that are well-synchronized with the input speech, but also produces natural facial expressions that reflect the prompt along with realistic hand gestures and action movements

[Figure 264]

- Stage 0.
- Stage 1.

text

Text 100%

20%

Close-Up Shot (1 Character) 80%

- Stage 2. 20%

Close-Up Shot

40% 40%

Medium Close-Up Shot

- Stage 3.

Close-Up

20% 20%

Medium Close-Up Medium Shot

1 Character (1~2) Character

20%

text 1 Character (1~2)Character

40%

>=1 Character

text

Speech Condition

Strong

Speech Condition

Weak

Easy Task

Hard Task

Difﬁculty

- Figure 6. Multi-Stage Training Strategy for MoCha. TextSpeech Joint training starts with close-up shots where speech conditioning has the strongest influence. At each stage, previous data is reduced by 50%, and harder tasks with weaker speech conditioning are introduced. Stage 0 uses text-only video data to establish a foundation for the future stages.

of speech-conditioned data simultaneously can lead to inefficiencies.

As illustrated in Figure 6, we address this challenge with a multi-stage training framework that categorizes data based on shot types, ranging from close-up to medium shots:

- • We begin training with close-up shots, which contain the strongest speech-video correlation.
- • At each subsequent stage, we reduce the data from the previous stage by 50% while introducing harder tasks with weaker speech conditioning.
- • We maintain the 80% ST2V and 20% T2V data ratio across all stages to ensure balanced training.

In Stage 0, we pretrain MoCha exclusively on textconditioned video data (T2V) to establish a strong foundational prior for video generation before incorporating speech-conditioning signals.

#### 4. Experiment

ing in human video generation exhibits a diminishing influence as we progress from low-level to high-level motion: it strongly governs lip movements and facial expressions, but its effect weakens for co-speech gestures and full-body actions. Meanwhile, generating these higher-level motions is inherently more difficult. As a result, training on all types

In this section, we first describe our training data processing pipeline in subsection 4.1, followed by the details of our model in subsection 4.2. We then introduce MoChaBench for Talking Characters task and benchmark MoCha against baseline methods in subsection 4.3, and finally, we present an ablation study to analyze the impact of key design choices in subsection 4.4.

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

MoCha(Ours)SadTalkerAniPortraitHallo3MoCha(Ours)

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

### /sʌm/ /ﬁːl/ /laɪk/ /taɪ/ /tʃɛs/

Prompt: “A close-up shot of a young woman embracing her cat outdoors. She is speaking while facing slightly to the right of the frame with an frustrating expression. The background is… she continues speaking, her expression remaining tense with anger while still facing slightly to the right…”

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

SadTalkerAniPortraitHallo3

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Prompt: “A medium shot of a man speaking while adjusting his glasses… As he talks, he first takes off his glasses, then puts them back on…”

- Figure 7. Qualitative comparison between MoCha and baselines on MoCha-Bench. MoCha not only produces lip movements that align closely with the input speech—enhancing the clarity and naturalness of articulation—but also generates expressive facial animations and realistic, complex actions that faithfully follow the textual prompt. In contrast, SadTalker and AniPortrait exhibit minimal head motion and limited lip synchronization. Hallo3 mostly follows the lip-syncing but suffers from inaccurate articulation, erratic head movements, and noticeable visual artifacts. Since the baselines operate in an image-to-video (I2V) setting, we provide them with the first frame generated by MoCha as input for comparison. The first frame is cropped and resized as needed to meet the requirements of each baseline.

##### 4.1. Training Data Processing Pipeline

To construct high-quality training data, we employ a multistage filtering and annotation pipeline.

• Speech Scene Filtering: We first segment videos into scenes using PySceneDetect [4]. Each scene undergoes speech detection, where non-speech segments and those heavily influenced by background noise or music are dis-

MoCha(Ours) Hallo3 SadTalker AniPortrait

- 0
- 1
- 2
- 3
- 4

3.85 3.82 3.82 3.85 3.72

2.95

Score

2.45

2.35 2.36

2.25 2.13

1.45 1.21 1.16 1.14 1.12 1.00 1.00

N/A N/A

Lip-Sync Quality Facial Expression Naturalness Action Naturalness Text Alignment Visual Quality

- Figure 8. Human evaluation scores on MoCha-Bench. Scores range from 1 to 4 across five evaluation axes, where a score of 4 reflects performance that is nearly indistinguishable from real video or cinematic production. MoCha significantly outperforms all baselines across all axes. SadTalker and AniPortrait consistently received a score of 1 for action naturalness, as these methods only perform head movements. Text alignment is marked as not applicable (N/A) for these baselines since they do not accept text input.

carded. For each valid segment, we perform music and noise removal and Wav2Vec2 [1, 29] is used to extract speech embeddings for valid segments.

- • Prominent Character Filtering: To ensure the dataset focuses on scenes with clear, prominent human characters, we employ an LLM-based filtering mechanism. The model analyzes each scene and removes those lacking a central characters.
- • Motion and Lip-Sync Filtering: We further refine the dataset by applying motion and lip-sync filters, ensuring that the extracted speech corresponds to meaningful human expressions and actions.
- • Scene Captioning: Each processed scene is captioned using an LLM [25], describing character appearances, positions, speech activity, emotions, and body language in highly detailed structured format.

Following this pipeline, we curate a high-quality dataset consisting of 300 hours (O(500)K samples) of speechconditioned video data.

##### 4.2. Implementation Details

Our model follows a design similar to MovieGen [27] and HunyuanVideo [19], utilizing a DiT architecture. MoCha is built on a pretrained 30-B DiT model. All models are trained with a spatial resolution of approximately 720×720, accommodating multiple aspect ratios. The model is optimized to generate 128-frame videos at 24 frames per second, resulting in outputs with a duration of 5.3 seconds. Our dataset [9] for text conditioning comprises approximately O(100)M samples, while the speech conditioning video dataset consists of around O(500)K samples. Training is conducted on 64 nodes to support the scale of our model.

##### 4.3. Evaluation

Baselines. We compare our approach against several representative audio-driven talking face generation methods with publicly available source code or implementations, in-

cluding SadTalker [44], AniPortrait [38], and Hallo3 [40]. These baselines span both end-to-end architectures and those utilizing intermediate facial representations, enabling a comprehensive evaluation of our method across diverse generation paradigms.

Benchmark. We introduce MoCha-bench, a benchmark tailored for the Talking Character generation task. It comprises 150 diverse examples, each consisting of a text prompt and corresponding audio clip. The dataset includes both close-up and medium-shot compositions: close-ups emphasize facial expressions and lip synchronization, while medium shots highlight hand gestures and broader body movements. The scenes span a wide range of human activities and interactions with objects (e.g., a chef chopping vegetables, a musician playing an instrument) and the character speaking with various emotions and facing directions. All text prompts were manually curated and further enriched using the publicly released LLaMA-3 [12] model to enhance expressiveness and variety. As MoCha directly generates videos from speech and text inputs, while all baseline models operate in an image-to-video (I2V) setting, we ensure a fair comparison by providing each I2V method with the first frame of the MoCha’s generation as the input. Qualitative Experiments We presents qualitative results of MoCha-30B in Fig. 1 and Fig. 5 showcasing its ability to generate diverse and realistic human motion while synchronizing speech with complex actions.

Fig. 7 presents a direct comparison between MoCha and baseline methods on MoCha-Bench. All baselines require a reference image as an auxiliary input. To ensure fairness, we first generate a video using MoCha and then use its first frame as the reference image for all baseline models. For models that do not support arbitrary aspect ratios, we crop the first frame to focus on the head region before feeding it into their networks. We provide two groups of qualitative comparisons: one featuring close-up shots and the

###### Method Lip-Sync Quality Facial Expression Naturalness Action Naturalness Text Alignment Visual Quality

Hallo3 [40] 2.45 2.25 2.13 2.35 2.36 SadTalker [44] 1.21 1.14 1.00 N/A 2.95 AniPortrait [38] 1.16 1.12 1.00 N/A 1.45

MoCha (Ours) 3.85 (+1.40) 3.82 (+1.57) 3.82 (+1.69) 3.85 (+1.50) 3.72 (+1.36)

- Table 1. Human evaluation scores on MoCha-Bench. Scores range from 1 to 4 across five evaluation axes, where a score of 4 reflects performance that is nearly indistinguishable from real video or cinematic production. Participants rated each method on five aspects: lip-sync quality, facial expression naturalness, action naturalness, text-prompt alignment, and visual quality. MoCha significantly outperforms prior methods across all categories. Green numbers indicate absolute improvements (∆) over the second-best method (underlined). SadTalker and AniPortrait consistently received a score of 1 for action naturalness, as these methods only perform head movements.

Method Sync-C ↑ Sync-D ↓

SadTalker [44] 4.727 9.239 AniPortrait [38] 1.740 11.383 Hallo3 [40] 4.866 8.963 Ours 6.037 (+1.17) 8.103 (-0.86)

- Table 2. Comparison with State-of-the-Art Methods on MoCha-Bench. We report synchronization metrics: Sync-C (higher is better) and Sync-D (lower is better). MoCha outperforms all baselines, indicating superior lip-sync quality.

Ablation Sync-C ↑ Sync-D ↓ Ours 6.037 8.103 w/o Joint ST2V + T2V Training 5.659 8.435 w/o Speech-Video Window Attention 5.103 8.851

- Table 3. Ablation Study of MoCha on MoCha-Bench We analyze the impact of different components by disabling them and measuring the effect on key metrics. Removing speech-video window attention degrades synchronization, joint ST2V and T2V training improves generalization.

Human Evaluations. We conduct a comprehensive human evaluation to compare MoCha against baseline methods on the MoCha-Bench dataset. The evaluation is based on five axes tailored for the Talking Characters task(See 2):

- • Lip-Sync Quality: Measures how accurately the character’s lip movements align with the spoken audio. Scale: 1 – Not aligned at all, 2 – Weak alignment, 3 – Mostly aligned, 4 – Perfectly aligned.
- • Facial Expression Naturalness: Evaluates whether the facial expressions and lip-sync appear natural and contextually coherent, without seeming robotic or exaggerated. Scale: 1 – Completely unnatural, 2 – Noticeably synthetic or stiff, 3 – Mostly natural and believable, 4 – Indistinguishable from real or cinematic performance.
- • Action Naturalness: Assesses how naturally the character’s body movements and gestures align with the audio. Scale: 1 – Completely unnatural, 2 – Noticeably unnatural, 3 – Mostly natural, 4 – Indistinguishable from real movie or TV characters.
- • Text Alignment: Measures how well the generated actions and expressions follow the behaviors described. Scale: 1 – No alignment, 2 – Partial alignment, 3 – Mostly aligned, 4 – Perfect alignment with the prompt.
- • Visual Quality: Evaluates visual quality by checking for issues such as artifacts, discontinuities, or glitches. Scale: 1 – Severe artifacts, 2 – Noticeable artifacts, 3 – Mostly artifact-free, 4 – Flawless visuals.

other medium shots. The close-up group emphasizes lipsync quality, head movement, and facial expressions, while the medium shot group focuses on hand movements during speech. MoCha not only produces lip movements that closely align with the input speech—enhancing both articulation and naturalness—but also generates expressive facial animations and realistic, coordinated actions that accurately follow the textual prompt. In contrast, SadTalker and AniPortrait exhibit minimal head motion and limited lip synchronization. While Hallo3 achieves mostly consistent lipsyncing, it suffers from inaccurate articulation and erratic head movements. In the medium shot comparisons, Hallo3 also introduces noticeable visual artifacts, particularly during complex actions.

Each model output received 5 independent ratings per example, resulting in over 750 responses per model. MoCha significantly outperforms all baselines across all five axes, with average scores approaching 4—indicating performance that is nearly indistinguishable from real video or cinematic production.

##### 4.4. Ablation Studies

We conduct ablation studies to analyze the contribution of key components in MoCha. Table 3 presents the impact of each design choice.

Quantitative Experiments We evaluate video quality using the automatic metrics to measure the lip-sync quality. Table 2 presents a comparison on the MoCha-Bench. Our model achieves the best scores across lip-sync metrics.

• Speech-Video Window Attention Ablation: We disable our speech-video window attention mechanism to analyze

its effect on speech-video alignment. This results in a noticeable drop in Sync-C (6.037 → 5.103) and increased Sync-D (8.103 → 8.851), confirming that our method significantly enhances lip synchronization.

• Joint ST2V and T2V Training Ablation: We train MoCha exclusively on ST2V data (removing text-only video training). This leads to This results in a noticeable drop in Sync-C (6.037 → 5.659) and increased Sync-D (8.103 → 8.435), indicating degraded generalization due to reduced dataset diversity.

These findings confirm that both our speech-video window attention and joint training strategy are essential for achieving high-quality motion, realistic speech alignment, and overall superior generation performance.

#### 5. Related Work

##### 5.1. Talking Head Generation

Given an audio sequence and a reference face, pioneer talking-head generation works typically utilize biometric signals such as facial keypoints [22, 28, 30, 45], or 3D priors [11, 13, 16, 23, 32, 43, 44] as intermediate motion representation to animate the reference face while ensuring lip synchronization. For example, SadTalker [44] first extracts 3DMM coefficients from audio and then renders the face in a 3D-aware manner. AniPortrait [38] predicts 2D facial landmarks from audio and then utilizes a diffusion models to generate a portrait video from the 2D landmarks maps. VLOGGER [8] predicts both 3D expression coefficient and 3D body pose from speech and enables the simultaneous generation of talking-face animations and upper-body gestures. Although effective, videos generated by these methods often lack expressiveness and naturalness due to the limited representation of 2D/3D priors.

Recently works, such as EMO [34] and Hallo [40], generate audio-driven portrait videos end-to-end using diffusion models, which eliminate intermediate facial representations and learn natural motion from data [10, 18, 34, 37, 40]. Hallo3 [9] builds upon pretrained transformer-based video diffusion models to animate faces with dynamic head poses and background elements. Although these methods can generate natural expressions, they rely on complex auxiliary signals—such as reference images or keypoints—which not only limit the naturalness and flexibility of facial expressions and body movements but also limit the generalization ablity of those methods.

##### 5.2. Diffusion-Based Video Generation

Diffusion models have emerged as a powerful approach for video synthesis, demonstrating state-of-the-art results in text-to-video generation. Methods like Make-A-Video [31], MagicVideo [46], and AnimateDiff [14] leverage pretrained text-to-image (T2I) models and extend them to

the temporal domain to synthesize coherent motion sequences. Recent advances in DiT-based architectures, such as CogVideoX [42] and MovieGen [27], have further improved video fidelity and controllability by integrating spatial and temporal constraints.

Despite these advancements, existing diffusion-based methods primarily focus on scene dynamics and global motion synthesis, lacking explicit modeling of speech-driven facial and body gestures of characters in the video. Our proposed MoCha framework extends diffusion models to jointly condition on speech and text, enabling the generation of lifelike character animations with natural conversation.

#### 6. Conclusion

In summary, our work pioneers the task of Talking Characters Generation, pushing beyond traditional talking head synthesis to enable full-body, multi-character animations directly driven by speech and text. We present MoCha, the first framework to address this challenging task, introducing key innovations such as the speech-video window attention mechanism for precise audio-visual alignment and a joint training strategy that leverages both speech- and text-labeled data for enhanced generalization. Additionally, our structured prompt design unlocks multi-character, turn-based dialogues with contextual awareness, marking a significant step toward scalable, cinematic AI storytelling. Through comprehensive experiments and human evaluations, we demonstrate that MoCha delivers state-of-the-art performance in terms of realism, expressiveness, and controllability, setting a solid foundation for future research in generative character animation.

#### 7. Acknowledgment

We thank Xinyi Ji, Tianquan Di, Anqi Xu, Matthew Yu, Emily Luo for providing speech samples used in the MoCha demo.

#### References

- [1] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020. 3, 8
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1:8, 2024. 2
- [4] Brandon Castellano. PySceneDetect. 7

- [5] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025. 2
- [6] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [7] Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, Yifei Hu, Ting-Che Lin, Shilong Zhang, Fu Li, Chuan Li, Xing Wang, Yanghua Peng, Peize Sun, Ping Luo, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Goku: Flow based video generative foundation models. arXiv preprint arXiv:2502.04896, 2025. 2
- [8] Enric Corona, Andrei Zanfir, Eduard Gabriel Bazavan, Nikos Kolotouros, Thiemo Alldieck, and Cristian Sminchisescu. Vlogger: Multimodal diffusion for embodied avatar synthesis. arXiv preprint arXiv:2403.08764, 2024. 10
- [9] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with diffusion transformer networks. arXiv preprint arXiv:2412.00733, 2024. 8, 10
- [10] Jiahao Cui, Hui Li, Yao Yao, Hao Zhu, Hanlin Shang, Kaihui Cheng, Hang Zhou, Siyu Zhu, and Jingdong Wang. Hallo2: Long-duration and high-resolution audio-driven portrait image animation. ICLR, 2025. 3, 10
- [11] Michail Christos Doukas, Stefanos Zafeiriou, and Viktoriia Sharmanska. Headgan: One-shot neural head synthesis and editing. In Proceedings of the IEEE/CVF International conference on Computer Vision, pages 14398–14407, 2021. 10
- [12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 8

- [13] Yuan Gan, Zongxin Yang, Xihang Yue, Lingyun Sun, and Yi Yang. Efficient emotional adaptation for audio-driven talking-head generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22634– 22645, 2023. 10
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 10
- [15] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2
- [16] Xinya Ji, Hang Zhou, Kaisiyuan Wang, Wayne Wu, Chen Change Loy, Xun Cao, and Feng Xu. Audio-driven emotional video portraits. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14080–14089, 2021. 10

- [17] Xiaozhong Ji, Xiaobin Hu, Zhihong Xu, Junwei Zhu, Chuming Lin, Qingdong He, Jiangning Zhang, Donghao Luo, Yi Chen, Qin Lin, et al. Sonic: Shifting focus to global audio perception in portrait animation. arXiv preprint arXiv:2411.16331, 2024. 2
- [18] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. In The Thirteenth International Conference on Learning Representations, 2025. 2, 10
- [19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 4, 8
- [20] Feng Liang, Haoyu Ma, Zecheng He, Tingbo Hou, Ji Hou, Kunpeng Li, Xiaoliang Dai, Felix Juefei-Xu, Samaneh Azadi, Animesh Sinha, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Movie weaver: Tuning-free multi-concept video personalization with anchored prompts. CVPR, 2025. 5
- [21] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [22] Yunfei Liu, Lijian Lin, Fei Yu, Changyin Zhou, and Yu Li. Moda: Mapping-once audio-driven portrait animation with dual attentions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23020–23029,

2023. 10

- [23] Yifeng Ma, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yingya Zhang, and Zhidong Deng. Dreamtalk: When expressive talking head generation meets diffusion probabilistic models. arXiv preprint arXiv:2312.09767, 2023. 10
- [24] Rang Meng, Xingyu Zhang, Yuming Li, and Chenguang Ma. Echomimicv2: Towards striking, simplified, and semi-body human animation. arXiv preprint arXiv:2411.10061, 2024. 2
- [25] AI Meta. Introducing meta llama 3: The most capable openly available llm to date. Meta AI, 2(5):6, 2024. 8
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. ICCV, 2023. 3
- [27] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 2, 4, 8, 10

- [28] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM international conference on multimedia, pages 484–492, 2020. 10
- [29] Steffen Schneider, Alexei Baevski, Ronan Collobert, and Michael Auli. wav2vec: Unsupervised pre-training for speech recognition. arXiv preprint arXiv:1904.05862, 2019. 8
- [30] Shuai Shen, Wenliang Zhao, Zibin Meng, Wanhua Li, Zheng Zhu, Jie Zhou, and Jiwen Lu. Difftalk: Crafting diffusion

- models for generalized audio-driven portraits animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1982–1991, 2023. 10
- [31] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 10

- [32] Xusen Sun, Longhao Zhang, Hao Zhu, Peng Zhang, Bang Zhang, Xinya Ji, Kangneng Zhou, Daiheng Gao, Liefeng Bo, and Xun Cao. Vividtalk: One-shot audio-driven talking head generation based on 3d hybrid prior. arXiv preprint arXiv:2312.01841, 2023. 10
- [33] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244–260,

2024. 2, 3

- [34] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244–260. Springer, 2024. 10
- [35] Linrui Tian, Siqi Hu, Qi Wang, Bang Zhang, and Liefeng Bo. Emo2: End-effector guided audio-driven avatar video generation. arXiv preprint arXiv:2501.10687, 2025. 2, 3
- [36] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1526–1535,

2018. 2

- [37] Cong Wang, Kuan Tian, Jun Zhang, Yonghang Guan, Feng Luo, Fei Shen, Zhiwei Jiang, Qing Gu, Xiao Han, and Wei Yang. V-express: Conditional dropout for progressive training of portrait video generation. arXiv preprint arXiv:2406.02511, 2024. 10
- [38] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024. 8, 9, 10
- [39] Ziyi Wu, Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Yuwei Fang, Varnith Chordia, Igor Gilitschenski, and Sergey Tulyakov. Mind the time: Temporally-controlled multi-event video generation. In CVPR, 2025. 5
- [40] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 2, 3, 8, 9, 10
- [41] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 2
- [42] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 10
- [43] Zhenhui Ye, Tianyun Zhong, Yi Ren, Jiaqi Yang, Weichuang Li, Jiawei Huang, Ziyue Jiang, Jinzheng He, Rongjie Huang,

- Jinglin Liu, et al. Real3d-portrait: One-shot realistic 3d talking portrait synthesis. ICLR, 2024. 10
- [44] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8652–8661, 2023. 8, 9, 10
- [45] Weizhi Zhong, Chaowei Fang, Yinqi Cai, Pengxu Wei, Gangming Zhao, Liang Lin, and Guanbin Li. Identitypreserving talking face generation with landmark and appearance priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9729–9738, 2023. 10
- [46] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 10
- [47] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. Advances in Neural Information Processing Systems, 37: 110315–110340, 2024. 2

