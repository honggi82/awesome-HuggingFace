# arXiv:2508.20088v2[cs.CV]2Oct2025

## AudioStory: Generating Long-Form Narrative Audio with Large Language Models

Yuxin Guo1,2,3, Teng Wang2†∗, Yuying Ge2, Shijie Ma1,2,3, Yixiao Ge2, Wei Zou1,3∗, Ying Shan2 1School of Artificial Intelligence, University of Chinese Academy of Sciences 2ARC Lab, Tencent PCG 3MAIS, Institute of Automation, CAS, Beijing † Project Lead ∗ Corresponding Authors https://github.com/TencentARC/AudioStory

### Abstract

Recent advances in text-to-audio (TTA) generation excel at synthesizing short audio clips but struggle with long-form narrative audio, which requires temporal coherence and compositional reasoning. To fill this gap, we propose AudioStory, a unified framework that integrates large language models (LLMs) with TTA systems to generate structured, long-form audio narratives. AudioStory possesses strong instruction-following reasoning generation capabilities. It employs LLMs to decompose complex narrative queries into temporally ordered sub-tasks with contextual cues, enabling coherent scene transitions and emotional tone consistency. AudioStory has two appealing features: (1) Decoupled bridging mechanism: AudioStory disentangles LLM-diffuser collaboration into two specialized components, i.e., a bridging query for intra-event semantic alignment and a residual query for inter-event coherence preservation. (2) End-to-end training: By unifying instruction comprehension and audio generation within a single end-to-end framework, AudioStory eliminates the need for modular training pipelines while enhancing synergy between components. Furthermore, we establish a benchmark AudioStory10K, encompassing diverse domains such as animated soundscapes and natural sound narratives. Extensive experiments show the superiority of AudioStory on both single-audio generation and narrative audio generation, surpassing prior TTA baselines in both instruction-following ability and audio fidelity.

### 1 Introduction

Audio content plays a pivotal role in modern media, from immersive storytelling and podcasts to interactive entertainment and educational applications. Recent advancements in text-to-audio (TTA) generation, exemplified by models such as TangoFlux [1], AudioLDM [2], and Stable Audio [3], have demonstrated remarkable capabilities in synthesizing high-quality, short-form audio clips from textual descriptions. However, a critical gap remains in generating long-form narrative audio, i.e., coherent, structured sequences of audio instances that unfold over extended durations, such as audiobooks, podcasts, or dynamic soundscapes for games.

Long-form narrative audio generation introduces unique challenges that extend beyond single-prompt synthesis. First, it requires temporal coherence: maintaining consistency in themes, sound effects, and emotional tone across the whole audio. Second, it demands narrative reasoning to decompose a complex instruction into logically ordered sub-events, characters, or environmental interactions. For instance, a prompt like “A suspenseful chase through a rainstorm: footsteps splash, thunder roars, a car skids, and a door slams shut” necessitates not only generating individual sounds but also orchestrating their timing, intensity, and interplay to build tension. Existing TTA models, while

Preprint. Under review.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

###### Long-Form Narrative Audio Generation

###### Narrative Audio Generation Divide-and-conquer Generation

[Figure 7]

Craft a audio composition for a Tom & Jerry scene showing tom corners jerry, who escapes into a mouse hole. tom waits, but jerry emerges cleverly from an electrical outlet cover. The total duration is 31.6 seconds.

The storyline is mainly about: Tom menacingly laughs as Jerry runs and hides in a mouse hole. Tom waits patiently while Jerry cautiously reemerges from behind the outlet cover.

###### Storyline summarization

[Figure 8]

[Figure 9]

The whole story can be divided into 4 sequential parts.

###### Event number

[Figure 10]

[Figure 11]

Event 1 Event 2 Event 4

Event 3

- Event 1: Timestamp: 0.0s - 1.1s. Tom corners Jerry, instilling fear in him. Story detail: Tom has a menacing appearance, suggesting a threatening situation. Sound effect: Sharp, sudden dissonant chord emphasizing Tom's terrifying presence and sinister laugh.
- Event 2: Timestamp: 1.1s -10.8s. Jerry escapes into a mouse hole, leading to a chase. Story detail: Jerry frantically runs away from Tom, diving into the safe haven of a mouse hole. Sound effect: Fast-tempo, high-energy orchestral piece matching the intensity of the chase and Jerry's hurried movements. … Event 4: Timestamp: 22.1s - 31.6s. Jerry cleverly emerges from an electrical outlet cover. Story detail: Jerry peeks out cautiously from an unexpected exit, ensuring it's safe before coming out. Sound effect: Light, inquisitive strings and careful woodwinds creating a tentative and curious atmosphere reflecting Jerry's emergence.

Audio Continuation

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Based on the provided audio in Tom & Jerry, continue the audio story about tom corners jerry, who escapes into a

[Figure 16]

[Figure 17]

###### Event decompose & Sound Reasoning

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

- • Scene transition Temporal duration
- • Emotional tone

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

mouse hole. tom waits, but jerry emerges cleverly from an electrical outlet cover. The total duration is 31.6 seconds.

[Figure 27]

Diffuser Diffuser Diffuser

Diffuser

[Figure 28]

[Figure 29]

[Figure 30]

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

Video Dubbing

Please watch a video and identify the temporal consecutive key events or scene changes in the video, and generate a Tom & Jerry-style background soundtrack to the entire video.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Caption token Semantic token Residual token

Divide Conquer

[Figure 53]

[Figure 54]

Multimodal LLM Reasoner

Instruction Storyline Reasoning Interleaved Reasoning Generation

- Figure 1: AudioStory effectively follows multimodal instructions, decomposing them into a sequence of coherent audio segments, capturing scene transitions, emotional tone, and segment timestamps. Unlike prior T5-based diffusion models, which struggle with complex queries, AudioStory empowers LLMs with high-level planning ability for instruction-followed and consistent long audio generation.

proficient at capturing isolated events, often struggle with such compositional and temporal reasoning, leading to fragmented or inconsistent outputs.

To address these challenges, we propose AudioStory, a novel multi-step framework for generating long-form narrative audio by integrating the reasoning capabilities of LLMs with audio generation. As shown in Fig. 1, we propose interleaved reasoning generation following a divide-and-conquer manner: reasoning for general narrative plans, decomposing plans into sequential generation actions, and generating interleaved audio events step-by-step. Specifically, AudioStory employs LLMs to decompose a narrative query (in language or multimodality) into a structured sequence of audiogenerative sub-tasks, each accompanied by contextual cues such as temporal offsets, emotional tone, and character interactions. These reasoning chains are then synthesized into audio events using a diffusion backbone, with explicit mechanisms to ensure style consistency, smooth transitions and temporal alignment. We streamline the narrative planning via LLMs and audio synthesis via diffusion models into an end-to-end framework, enabling the generation of rich, multi-scene audio stories that adhere to user intent while preserving coherence over time.

AudioStory introduces several technical innovations: First, unlike prior approaches [4, 5] that bridge LLMs with audio diffusers through predefined textual spaces (e.g., T5 [6]), we propose a decoupled bridging space consisting of two distinct tokens: (1) semantic tokens, which encode text-oriented audio semantics, and (2) residual tokens, which capture nuanced acoustic cues and cross-event correlations. This design effectively improves both audio fidelity and temporal consistency during generation. Second, unlike zero-shot integration of LLMs and diffusers, our framework supports end-to-end progressive training, enabling joint optimization of instruction understanding and audio synthesis. This synergistic training paradigm enhances both audio understanding and generation performance. Third, we introduce the first narrative audio generation benchmark, providing a comprehensive evaluation framework for assessing audio generation quality and consistency.

The contributions of the paper are as follows:

- • We introduce AudioStory for narrative audio generation, which integrates LLM-based reasoning and iterative diffusion-based generation in a unified framework, with strong multimodal instructionfollowing and audio generation abilities.
- • We propose decoupled bridging tokens for LLM-diffuser collaboration, using semantic tokens (text-oriented audio semantics) and residual tokens (nuanced acoustic cues) to improve audio fidelity and temporal consistency.
- • We introduce a synergistic training paradigm, facilitating collaboration and complementarity between LLM and diffusion models. Unlike zero-shot LLM-diffusion integration, our framework enables end-to-end joint training, enhancing both multimodal understanding and generation.
- • Experiments show AudoStory significantly surpasses prior diffusion-based and MLLM-based models by a large margin in narrative audio generation. We also uncover some important findings across multiple aspects, including reasoning formulation, bridging mechanism and training recipes.

### 2 Related Works

Text-to-audio generation (TTA). Recent advances in latent diffusion and flow-matching frameworks have significantly advanced text-to-audio generation. Diffusion/flow-based approaches, exemplified by Make-An-Audio [7] and AudioLDM [8, 2], synthesize audio through iterative denoising of textconditioned latent representations. Extensions like Tango [9, 10], Audio Flamingo [11], GenAu [12], Fugatto [13] further enhance design spaces of latent space, data quality and cross-modal alignments. Recently, Stable Audio series [3] employs hierarchical latent diffusion trained on large-scale datasets for high-fidelity output. Beyond diffusion-based priors, flow-matching techniques optimize probability density transport for audio synthesis. VoiceBox [14] enables zero-shot style transfer via continuous normalizing flows, while AudioBox [15] and FlashAudio [16] prioritize computational efficiency through rectified flow architectures. TangoFlux [1] introduces CLAP-ranked preference optimization to iteratively generates and optimizes preference data to enhance text-audio alignment. Existing methods align text and audio semantically but primarily target descriptive queries, limiting interactive control and adaptability to evolving instructions. They are also confined to short audio domains. These limitations demand TTA models to handle complex instructions over long durations.

Any-to-any multimodal LLMs. Within the rapidly evolving field of multimodal learning, any-to-any generation across vision, language, and audio modalities represents a significant frontier [17, 18, 4, 19, 5, 20, 21, 22, 23, 24, 25, 26]. This paradigm aims for models capable of accepting arbitrary input modalities and generating outputs in any desired modality. Pioneering efforts include CoDi [27] and CoDi-2 [18], which leveraged composable diffusion for diverse modality handling. Spider [5] further extended these capabilities by enabling the generation of multiple modalities in a single response. NExT-GPT [4] demonstrated the efficacy of lightweight alignment for adapting LLMs to multimodal tasks, while AnyGPT [19] showcased the potential of discrete sequence modeling for unified multimodal processing. Unified-IO2 [28] highlighted the impact of scale and unified architectures in achieving state-of-the-art performance across a broad spectrum of modalities and tasks. Despite these advancements, current methods exhibit limitations in long-context generation with complex instructions: First, they primarily focus on speech generation and simple caption-tomusic or caption-to-sound tasks, struggling to comprehend general and intricate human instructions beyond basic caption; Second, their audio generation is typically limited to single, short segments, hindering the generation of longer audio sequences.

Compositional audio generation. Agentic workflows employ multiple off-the-shelf expert tools and a controller for compositional audio synthesis. Works like WavJourney [29] and MM-StoryAgent [30] decomposed audio generation into a text-centric interface and employs separate text-to-speech, audio, and music decoders for audio creation and storytelling. While these agents could generate combinations of audio components, their zero-shot nature suffers from suboptimal planning and limited adaption of nuanced acoustic cues, degrading instruction-following ability and overall audio quality. Instead, we target on end-to-end training to integrate LLM-based chain-like reasoning and flux decoders for long-term, consistent audio generation.

### 3 Narrative Audio Generation

Problem definition. Narrative audio generation aims to generate long-form, structured and temporally coherent audio sequences A = {Am}Mm=1, given multimodal instruction xins (e.g., language, audio or vision), where M is the number of audio segments. The task shares a similar formulation with the text-to-audio generation, but is far more challenging due to two distinct capabilities: (1) Temporal coherence, i.e., maintaining consistency in themes, sound effects, and emotional tone across extended durations; (2) Compositional reasoning. i.e., decomposing high-level narrative instructions into logically ordered events (e.g., “footsteps splash, then thunder roars”) with precise timing and contextual interactions. Existing TTA systems, while effective for short clips, lack explicit mechanisms to model cross-segment dependencies or align audio events with evolving narrative structures, limiting their applicability to real-world scenarios.

The AudioStory-10k benchmark. Given the lack of quantitative evaluation, we establish the AudioStory-10k benchmark for the narrative audio generation task. AudioStory-10k comprises 10k annotated audios paired with narrative prompts. We collect videos from two primary sources:

- • Natural sounds: We carefully select 4,723 audio instances from UnAV-100 [31], covering a broad spectrum of real-world environmental recordings (e.g., rainstorms, animal calls, rustling leaves) and human activities (e.g., footsteps, door slams, and conversations). This collection ensures sufficient coverage of everyday acoustic events and ambient soundscapes.
- • Animated sounds: We curate 5,332 audio clips from 157 episodes of Tom & Jerry, capturing stylized background music (e.g., orchestral pieces, string sections) and sound effects (e.g., slapstick actions, cartoonish collisions and rapid movements). These animated sounds feature stylized and expressive audio content, distinct from natural sound recordings.

The annotation pipeline involves three stages. First, we filter the videos with sequential audio events, ensuring the storyline of the audio is visually-grounded for meaningful activities1. Then, we parse the video into several key audio events by Gemini-2.5-Pro [34], each of which is labeled with its timestamps, audio caption and visual captions. Next, given these text-based timestamped captions, we prompt GPT-4o [35] to generate diverse instructions and chain-like reasoning steps.

To be specific, we design diverse format of multimodal instructions, including text-only instructions for narrative audio generation, audio-text ones for audio continuation and video-text ones for video dubbing (shown in Fig. 1). For a flexible control of duration and semantic elements of generated audios, we make the intermediate reasoning encompass at least the following steps: storyline summarization for global summarization of general story, event decomposition for inferring the number of audio events, sound reasoning for predicting timestamp and key elements (e.g., emotional tone, scene transition) of each event. All detailed prompts and processing steps are in Appendix E.1.

Evaluation metrics. The AudioStory-10k dataset includes 5.3k samples of natural sounds and

- 4.7k samples of cartoon audios. We randomly divided the dataset into 85% for training and 15% for testing. We propose a comprehensive evaluation spanning three aspects: instruction-following, consistency, and generation quality. (1) Instruction-following ability is quantified through multimodal alignment between instructions and generated audio (Instruct), CLAP score for audio-caption similarity, and reasoning text quality for logical decomposition and event planning. (2) Consistency metrics evaluates internal consistency (timbre uniformity, entity persistence) and temporal coherence (acoustic transitions, emotional flow). (3) Generation quality metrics employs FD and FAD [36] against ground-truth audio. Except from CLAP, FD, FAD-based metrics, we employ Gemini-2.0-flash as the evaluator with a score range of 0-5. More details could be found in the Appendix E.2.

### 4 AudioStory

Overview. To achieve instruction-followed audio generation, the ability to understand the input instruction and reason about relevant audio sub-events is essential. To this end, AudioStory adopts a unified understanding-generation framework (Fig. 2). Specifically, given multimodal instructions, an LLM analyzes and decomposes it into structured audio sub-events with context. Based on the inferred sub-events, the LLM first performs interleaved reasoning generation (Sec. 4.1), sequentially producing captions and bridging tokens between the LLM and the audio generator (Sec. 4.2). Through progressive end-to-end training, AudioStory ultimately achieves both strong instruction comprehension and high-quality audio generation (Sec. 4.3).

#### 4.1 Interleaved Reasoning Generation

Directly generating long-form narrative audio that aligns with complex instructions is challenging. We take the spirit of “divide-and-conquer” and propose decoupling the input instruction into chronological short audio clips, which are then combined to form the complete long-form narrative audio.

Single-audio clip generation. The ability to generate individual audio clips from captions is a foundational step toward producing sequential audio events. For audio clip generation, the LLM generates bridge tokens from a given caption, which serve as conditions for the DiT. While this method works well for short audio generation based on simple captions, it becomes insufficient for complex instructions involving multiple events, temporal relationships, or narrative structures.

1For Tom & Jerry, where episodes typically consist of numerous discontinuous shots with fast transitions, we employ PySceneDetect [32] to detect preliminary shot boundaries. These boundaries are further refined by thresholding the similarity between frame-level DINOv2 [33] features, which retains only high-quality, temporally consistent shots as individual video instances. For UnAV-100, we keep the videos longer than 30s.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Interleaved Reasoning Generation

(a) AudioStory Framework (b) Audio Generator

Flow loss Flow loss

Alignment loss

Alignment loss

Flow loss

Instruction token

Language token Semantic Token Residual Token

FLUX

FLUX

Next-Token Prediction

FLUX

Cross

Large Language Model (LLM)

Attention Noisy Audio

Duration

[Figure 60]

[Figure 61]

Bridging Mechanism

[Figure 62]

[Figure 63]

Multimodal Instruction Reasoning Caption S-Token R-Token

Caption S-Token R-Token

S-Token R-Token

𝐿

(c) Progressive Training Strategy

𝐿

𝐿 + 𝐿 + 𝐿

[Figure 64]

[Figure 65]

𝐿

𝐿

MM-DiT

[Figure 66]

MM-DiT

𝐿 𝐿

[Figure 67]

MM-DiT

MM-DiT

...

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

T5

LLM

[Figure 72]

[Figure 73]

T5

LLM

###### LLM

LLM

End-to-End Joint Training

End-to-End Joint Training

End-to-End Joint Training

Multimodal Instruction

Multimodal Instruction

Caption

Caption

Stage-I: Single-Audio Generaion Stage-II: Single-Audio Unified Gen. Stage-III: Long-Audio Unified Gen.

- Figure 2: Overview of AudioStory, with three core components: (a) A unified framewrok: The reasoning-capable LLM processes the instruction input, decomposes the long audio into structured generation sub-tasks, and sequentially generates a caption, semantic tokens, and residual tokens for each audio clip. (b) Audio Generator: After fusing semantic and residual tokens, they are combined with the duration information as conditioning inputs to the DiT, which then generates each audio clip. (c) Training strategy: Training is conducted in three stages to progressively enhance generation fidelity, semantic understanding, and global coherence.

Interleaved reasoning generation for long-audio generation. We propose to decouple a complex, long-form audio into multiple audio segments for segment-by-segment generation. This divide-andconquer process consists of two components: (1) Storyline reasoning: LLMs reason through the entire instruction, inferring the number of audio events. Furthermore, LLMs analyze the start and end timestamps of each event, as well as the event description and corresponding audio content that should be included. (2) Interleaved generation: For each event, the LLM infers the caption, duration, and corresponding bridge queries (semantic tokens and residual tokens, as described in Sec. 4.2), enabling interleaved generation. These queries, along with duration information, are then provided as conditional inputs to the DiT-based audio generator. By accurately predicting durations and utilizing semantically rich bridging tokens, the model ensures both coherent audio semantics within each event and consistency across events. The training data is structured as:

[BOS][BOT]{#event}{storyline reasoning tokens}[EOT][BOG]{caption}{duration} TsemanticTresidual[EOG]···[BOG]{caption}{duration}TsemanticTresidual[EOG][EOS].

(1) The textual tokens in the entire reasoning process is supervised by the next token prediction loss:

L

Lreason = L#eventtext + Lcontenttext + Lcaptiontext , where Ltext =

p(xi|X<i,Xp,<i). (2)

i=1

#### 4.2 Decoupled Bridging Mechanism

Once the LLM is capable of effective reasoning, establishing a seamless bridge between the LLM and the DiT becomes crucial. However, text alone might not be the optimal bridge. Although it carries rich semantics, it fails to capture diverse low-level details of the audio modality, e.g., timbre, rhythm, and ambience. Consequently, we propose decoupled bridges queries, which could be divided into semantic Tsemantic and residual tokens Tresidual. The semantic tokens represent the audio’s high-level semantics, while the residual tokens carry low-level audio details. They complement each other, enabling the disentanglement of audio information. In practice, after producing the caption for each audio event, the LLM collectively generates semantic and residual tokens. For semantic tokens, we use the textual tokens from Flan-T5 [37] Tgtsemantic as the supervision and apply MSE loss:

Lmse = ∥Tgtsemantic − Tsemantic∥22. (3)

The residual tokens are employed to supplement the missing information of the semantic tokens. Then, both types of tokens are merged and fed into as the conditional inputs of DiT. Here, we adopt multi-head cross-attention to merge these two tokens and obtain the resultant bridge queries:

Hbridge = Cross-Attn(Tsemantic,Tresidual,Tresidual). (4) For audio generator with Hbridge as condition, we employ flow-matching [38] for generative modeling:

1,x0,t∥u(xt,t,c) − vt∥22, (5)

Lflow = Ex

where c is the condition and we choose c = Hbridge and t is uniformly sampled from [0,1]. Through the generative supervision, Tresidual can capture detailed information and complement Tsemantic.

#### 4.3 Progressive Training Strategy

After establishing an effective bridge between the LLM and DiT, it becomes essential to design an efficient end-to-end training mechanism to build synergy between the understanding and generation tasks. We propose a progressive training strategy, following a single-to-multi and generation-tounification paradigm. The training could be divided into three stages, where the model (1) learn to generate single audio segments, followed by (2) unified generation and understanding for single audios and (3) long-audio adaptation.

- Stage-I: Single-audio generation. There are two sub-stages. (1) Stage-I-Warm, AudioStory learns to generate semantic tokens with MSE supervision in Eq. (3). Only the LoRA of the LLM and the projector of Tsemantic are updated. (2) Stage-I-Whole, AudioStory regresses bridge queries based on the input caption, i.e., generating Tsemantic and Tresidual, respectively. They are subsequently merged via Eq. (4) and fed into DiT. Here, the regression of Tsemantic and the prediction of its beginning and end tokens are supervised. We tune LoRA of the LLM, all projectors, the attention layer and the generation model DiT. The learning objectives are shown below:

Lwarms1 = Lmse, Lwholes1 = Lmse + λ1Ltokentext + λ2Lflow, (6)

where Ltokentext is only applied to the start and the end tokens of Tsemantic. After this Stage-I, AudioStory possesses a strong capability for single-audio generation.

- Stage-II: Single-audio unified generation and understanding. Building upon Stage-I, we further introduce audio understanding data to enable unified generation and understanding of single-audio clips. The model takes audio as input for understanding. We freeze the audio encoder while the trainable parameters remain the same as Stage-I-Whole. The learning objectives are in Eq (7).

Ls2

= Lmse + λ1Ltext + λ2Lflow. (7) With this unified training, AudioStory’s generation abilities can be further enhanced.

- Stage-III: Long-audio unified generation and understanding. We extend the unified training in Stage-II to long-form audio. We further introduce Interleaved Reasoning Generation (Sec. 4.1) for narrative audio generation. We curate a high-quality multi-audio dataset to perform supervised fine-tuning. For the generation task, the model sequentially infers the number of audio events based on the input instruction, analyzes the audio content, and performs interleaved generation of captions, semantic tokens, and residual tokens. For the audio continuation task, given the input audio and instruction, the model comprehends the inputs, reasons the key events with story details, and finally generates several short audio segments in a clip-by-clip manner. The audio understanding data incorporates audio Q&A and instruction data. We keep the learnable components the same as Stage-II. The overall learning objectives are:

= Lmse + λ1Ltext + λ2Lflow + λ3Lreason. (8)

Ls3

### 5 Experiments

In this section, we first present the experimental setup (Sec. 5.1). Then, we compare AudioStory with existing TTA and unified models on long-form audio generation (Sec. 5.2). We also study the audio understanding and the audio generation (Sec. 5.3) ability of AudioStory in short audio clips, showing its superior fundamental ability. Finally, in Sec. 5.5, we conduct an in-depth exploration of reasoning forms, bridging query types, joint training strategies, and the synergy between understanding and generation, and provide several key insights.

- Table 1: Comparative results on long-audio generation. “Instruct” is short for instruction-following and “CLAP” denotes CLAP score, “gt” denotes ground-truth. “Consis.” and “Coher.” are short for consistency and coherence. Here, bold and underline indicate the best and the second-best results.

Instruction-Following Consistency Generation Quality

Model

Max. Duration ↑ Instruct. ↑ CLAP ↑ Reasoning ↑ Consis. ↑ Coher. ↑ FD ↓ FAD ↓

AudioLDM2 [2] 2.8 0.296 - 4.6 4.4 3.43 4.49 10s TangoFlux [1] 3.2 0.317 - 4.1 4.2 2.48 3.49 30s

Caps (gt)+TangoFlux [1] 4.0 0.348 - 2.4 2.0 1.79 3.59 30s LLM+TangoFlux [1] 3.5 0.322 3.5 2.1 1.9 2.55 3.82 30s LLM+CoDi [27] 3.2 0.286 3.5 1.4 1.4 3.39 4.04 10s LLM+NExT-GPT [4] 3.3 0.299 3.5 1.8 1.7 3.47 3.99 10s

AudioStory 4.1 0.392 4.2 4.1 3.9 1.43 3.00 150s

#### 5.1 Experimental Setup

Implementation details. We choose Qwen-2.5-3B-Instruct [39] as the LLM and employ DiT pretrained from TangoFlux [1] as the initialization. For encoding instruction for the audio continuation task, we employ Whisper-large-v3 [40] as the audio encoder. The projector has two layers with GeLU activations. In Stage-I, AudioStory is trained with lr= 2e−4 for 50 epochs with a per-device batch size of 32. In Stage-II, we use lr=1e-4 for 10 epochs. The ratio of understanding and generation data is 2:1. In Stage-III, we set different learning rates for LLM and DiT. We set λ1 = 1,λ2 = 0.2,λ3 = 0.4. The tunable parameters three-stage training are LoRAs in LLMs, projectors, the cross-attention fuser for bridging queries, and DiT. More details are shown in the Appendix.

Training datasets. The training dataset comprises the understanding dataset, single-audio generation and multi-audio (long-audio) generation datasets. For the understanding dataset, we integrated AudioSetCaps [41], VGGSound [42], MusicCaps [43], and Auto-ACD [44], converting their captions into QA format. Additionally, we incorporated AudioSetCaps-QA and VGGSound-QA datasets, resulting in 1M audio-QA pairs in total. For the single-audio generation dataset, we combined AudioSetCaps, VGGSound, MusicCaps [43], and Auto-ACD, resulting in 700k audio-caption pairs. For the multi-audio generation dataset, we curated the AudioStory-10k dataset, with details provided in Sec. 3. In Stage-I, we train the model on we train the model on single-audio generation datasets. Stage-II further incorporates the audio understanding dataset beyond Stage-I. As for Stage-III, our model is trained using multi-audio generation as well as understanding datasets.

Evaluation metrics. For single-audio generation, we employ the AudioLDM-eval toolkit2 to compute Frechet Distance (FD), Frechet Audio Distance (FAD) [36], KL-Divergence (KL), and stable-audiometrics3 for FDopenl3 [45], KLpasst [46], and CLAP score on AudioCaps testset [47]. For audio understanding, we consider the tasks of audio question answering (AQA), and audio captioning on AudioCaps and Clotho dataset [48], reporting SPIDEr, CIDEr, and ACC scores. The evaluation metrics for long-audio generation is presented in Sec. 3.

Baseline methods. Prior audio generation models could be divided into two groups: pure TTA models like AudioLDM2 [2] and TangoFlux [1] and LLM-based unified models, including CoDi [27] and NExT-GPT [4]. Both of them could only generate short audio clips. For long-form audio generation, we curated three classes of baselines: (1) Directly generating audios with maximum available durations using the whole textual caption. (2) Incorporating LLMs to reason and generate captions for each short audio clip, which are then fed into baseline models to generate multiple audio clips separately. These clips are then concatenated to constitute the final long-form audio. (3) Directly using the ground truth captions in the benchmark, serving as the oracle setting and upper bound for baseline models. In addition, we also report the performance of AudioStory on the Tom & Jerry and the audio continuation task.

- 2https://github.com/haoheliu/audioldm_eval
- 3https://github.com/Stability-AI/stable-audio-metrics

- Table 2: Single audio understanding performance.

Table 3: Single audio generation performance.

AudioCaps Test Set FDopenl3 ↓ KLpasst ↓ FD ↓ FAD ↓ KL ↓ CLAP ↑

ClothoCaps ClothoAQA AudioCaps SPIDEr CIDEr ACC B-ACC SPIDEr CIDEr

Model

Model

Make-An-Audio [7] 128.49 1.16 1.65 3.16 0.63 0.256 stable-audio-open [3] 103.68 1.12 1.63 2.98 0.61 0.298 AudioLDM2 [2] 87.74 1.01 1.59 2.63 0.57 0.252 TangoFlux [1] 83.58 0.95 1.57 2.34 0.52 0.385 CoDi [27] 121.66 1.17 1.69 9.61 0.60 0.228 NExT-GPT [4] 107.18 1.13 1.64 5.69 0.59 0.265

UIO-2 XXL [28] 5.7 6.5 - - - 48.9 CoDi [27] 6.2 7.3 - - 48.0 78.9 NExT-GPT [4] 13.8 20.3 26.4 39.5 53.4 80.7 Spider [5] - - - - 53.7 81.9

AudioStory-Base 24.1 37.7 42.8 60.6 54.8 83.2

AudioStory-Base 83.39 0.91 1.52 2.29 0.51 0.383

#### 5.2 Long-Form Narrative Audio Generation

Instruction-following ability. As shown in Table 1, considering the instruction-following aspect, AudioStory demonstrates a significant advantage in complex scenarios involving multiple events and sounding objectives. It outperforms the LLM-aided TTA models by 17.85% on the CLAP score, thereby demonstrating the superior instruction-following generation capability of our model. Our method effectively addresses the issue of overlooking sounding entities, which can be attributed to the enhanced understanding and decomposition of the instruction.

Generation quality. AudioStory demonstrates strong long-form audio generation performance across both natural scenes and the cartoon domain. Our approach achieves superior FD and FAD scores compared to diffuser-based and LLM+diffuser baselines. This improvement is reasonable: (1) we enhance long-form audio generation by single-audio clip training, effectively extending high-quality short-audio generation to longer sequences; (2) the duration of generated audio is longer and more closely matches the reference long audios than those generated by previous methods.

Consistency. Notably, consistency is meaningful only with strong instruction-following. For example, AudioLDM2, despite high consistency scores from short (10s) outputs, performs poorly on instructionfollowing, making it a weak baseline. In contrast, our method achieves substantial advantages in both consistency and coherence, reaching scores of 4.0 and 3.7, respectively, as in Table 1. It is worth noting that in the consistency evaluation, AudioStory achieves comparable performance despite generating significantly longer audio with richer narratives compared to TTA models.

#### 5.3 Single-Audio Generation

Joint audio generation & understanding. We also evaluate our model’s performance on short audio generation and understanding tasks, and conduct comparisons with TTA and LLM-based models. For the generation task in Table 3, AudioStory outperforms prior competitors on both suites of evaluation tools, even outperforming the state-of-the-art TTA model, i.e., TangoFlux [1], indicating the effectiveness of the proposed LLM and DiT bridging mechanism. As for the audio understanding task in Table 2, AudioStory outperforms advanced LLM-based models, which means that our method could competently handle both generation and understanding tasks.

#### 5.4 Qualitative Analysis

AudioStory exhibits strong reasoning capabilities, it can accurately divide the input instruction into several events based on narrative logic and temporal order, subsequently generating short audio clips segment by segment, and ultimately composing a coherent long-form audio. Besides, AudioStory could accurately infer the duration of each audio clip. Here, we provide a qualitative case in Fig. 3. More cases are presented in the Appendix B.

#### 5.5 Ablation Studies

Does interleaved reasoning generation help narrative audio generation? We progressively investigate which forms of reasoning are effective for long-form narrative audio generation. We ablate two model variants: (a) a model that skips instruction analysis and directly generates captions for audio clips, and (b) a model that performs instruction decomposition without explicitly generating

Table 4: Ablations of reasoning.

Variant Cons. ↑ Inst. ↑ FAD ↓ CLAP ↑ w/o reasoning 3.1 3.1 4.13 0.34

w/o interleaved 1.6 1.2 16.03 0.14 w/ reasoning 4.0 4.1 3.06 0.39

###### Instructional Audio Generation

[Figure 74]

###### Instruction：

[Figure 75]

[Figure 76]

Audio input Story

Propose a detailed, immersive audio that showcases a woman chews, examines marshmallows, then noise abruptly ends the scene. The total duration is 38.5 seconds.

[Figure 77]

###### Generated Reasoning：

The caption can be divided into four points:\n\n1. **[0.0, 9.8]**\n - **Story Details**: A woman chews food slowly and deliberately.\n -

**Sound Effects**: Wet, rhythmic chewing sounds.\n\n2. **[9.8, 12.8]**\n - **Story Details**: She examines a package or wrapper briefly.\n - **Sound Effects**: Crisp rustling of a plastic bag or wrapper.\n\n3. **[12.8, 25.5]**\n - **Story Details**: She taps and inspects the surface of a marshmallow delicately.\n - **Sound Effects**: Soft tapping and rubbing sounds.\n\n4. **[25.5, 38.5]**\n -

**Story Details**: She holds a marshmallow near her mouth, lips slightly moving.\n - **Sound Effects**: Soft lip-smacking and mouthopening sounds.

[Figure 78]

###### Generated Caption：

- • Wet, close-miked chewing sounds, distinct and slightly squelchy. Rhythmic, slow, deliberate mastication creates an intimate, amplified ASMR effect. <timestamp>9.8</timestamp>
- • A brief, crisp rustling sound of handling a plastic bag or wrapper, moderately loud, suggests movement and packaging interaction. <timestamp>3.0</timestamp>
- • Soft tapping and rubbing sounds with a muffled, airy quality, like fingers on a marshmallow. The subtle, delicate noise creates a gentle atmosphere, evoking tenderness and a light, rhythmic texture. <timestamp>12.7</timestamp>
- • Soft, subtle lip-smacking and mouth-opening sounds, quiet and slightly wet, as the woman holds a marshmallow near her mouth and moves her lips slightly. <timestamp>13.0</timestamp>

Figure 3: Qualitative case of long-form audio generation.

audio captions. As shown in Table 4, ablating without reasoning leads to simplified audio events with missing objects or actions, resulting in a significant drop in both instruction-following performance and CLAP score. When interleaved reasoning is removed, the model can still infer event content, but lacks contextual guidance when generating bridge queries, severely degrading audio quality. We conclude that reasoning is essential for narrative audio generation, with interleaved reasoning being the most critical component. Explicit caption generation for each audio clip is necessary to ensure generation quality.

Which type of features are suitable for bridging between the LLM and the DiT? Our analysis suggests that audio features, on one hand, have lower semantic density, and on the other hand, are more difficult for the LLM to fit compared to textual features, especially given the complex temporal structure in Whisper. Therefore, supervising the semantic tokens with textual features is more suitable and efficient. For residual tokens, Table 5 (c)–(g) shows that explicit or weak supervision using existing audio features significantly harms generation performance. In summary, text features with rich semantics are well-suited for supervising semantic tokens, while for residual tokens, applying weak supervision through the DiT loss is the most effective way to capture complementary low-level audio information.

Table 5: Ablations on bridging mechanism.

ID BQ Sup. Feat. Single Multi

- (a) Semantic AudioMAE [49] 9.55 11.39
- (b) Whipser [40] 10.26 12.31

- (c) Residual AudioMAE [49] 9.24 10.06
- (d) Whisper [40] 11.06 11.21

- (e) Residual

+guid.

AudioMAE [49] 3.60 4.21

- (f) Whisper [40] 3.71 4.39

- (g) Ours T5 w/o guid. 2.29 3.12

What are the key factors in end-to-end joint training of unified models? For unified models, prior arts typically train the LLM and DiT separately, connecting them via a zero-shot bridging mechanism, which results in a feature gap. To address this, we propose end-to-end joint training of the LLM and DiT. We begin by focusing on the end-to-end training paradigm, as in Table 6. Notably, when residual tokens are removed, overall performance drops significantly. Analysis reveals that the LLM and DiT focus on different types of information, and directly updating the LLM using the DiT loss severely impairs its performance. In contrast, residual tokens effectively alleviate this issue. Secondly, we explore how to configure DiT’s learnable parameters. As shown in (c)–(f), fully freezing DiT degrades performance, while fully unfreezing it achieves the best results. Notably, unfreezing MMDiT outperforms Single-DiT, since the latter focuses on low-level features that are more sensitive to noise, thus affecting generation quality. Thus, we can draw the following conclusions: (1) End-to-end joint training of the LLM and DiT is essential. (2) Residual tokens play a critical role, as they capture

- Table 6: Ablations on the end-to-end joint training strategy of DiT. Here “S-DiT” and “M-DiT” denote Single-DiT and MM-DiT. “Consis.” denotes consistency.

ID

Semantic Tokens

Residual Tokens

DiT Joint Training

Tunable Module

Single Audio Multi Audio FD ↓ FAD ↓ KL ↓ Consis. ↑ FAD ↓

- (a) ✓ ✗ ✗ - 1.57 2.33 0.52 3.2 5.23
- (b) ✓ ✗ ✓ open all 2.16 4.66 0.84 3.4 4.98
- (c) ✓ ✗ ✓ freeze 4.86 11.04 0.89 1.3 12.97
- (d) ✓ ✓ ✓ open S-DiT 2.37 5.84 0.64 2.1 6.28
- (e) ✓ ✓ ✓ open M-DiT 1.98 3.21 0.67 3.5 3.64
- (f) ✓ ✓ ✓ open all 1.53 2.29 0.51 4.3 3.00

- Table 7: Ablations on progressive training. “Gen.”, “Und.” and “BQ” denote generation, understanding and Bridge Queries. “SAG” and “LAG” are short for single and long-form audio generation.

SAG LAG Audio Und. FAD ↓ FAD ↓ CIDEr ↑ SPIDEr ↑

ID Order Stage-I Stage-II Stage-III

- (a) Und.→Gen.

Und. - - - - 35.7 23.1

- (b) Und. BQ - 7.42 9.53 36.9 23.8
- (c) Und. BQ DiT joint 6.50 7.26 38.6 24.9

- (d) Gen.→Und.

BQ - - 2.37 5.23 - -

- (e) BQ Und. - 2.35 4.98 31.5 19.5

- (f) BQ Und. DiT joint 3.61 6.50 24.6 16.4
- (g) BQ DiT joint Und. 2.29 3.00 37.7 24.1

- (h) N/A DiT joint + Und. 5.70 8.74 27.3 18.2

low-level complementary information and help mitigate conflicts between DiT and LLM during optimization. (3) Fully unfreezing DiT is necessary. Selectively unfreezing either the Single-DiT or MM-DiT alone leads to suboptimal performance.

How to progressively build the synergy between generation and understanding? Both understanding and generation training are essential to our model, making a progressive training strategy crucial. Here, we examine the effectiveness of various progressive training approaches. In Table 7, without progressive training, performance on both comprehension and generation drops significantly, even worse than training the tasks independently, which is primarily due to the inherent conflict and task interference between them. In contrast, a well-structured progressive strategy enables unified training to outperform isolated approaches, highlighting its necessity. Further exploration into synergizing generation and comprehension reveals key insights: training generation first, then adding comprehension, yields optimal overall performance, with strong comprehension accuracy. In contrast, reversing the order harms generation, and interleaved training similarly undermines overall optimization. Therefore, we conclude that generation and understanding exhibit inherent synergy, and their optimal training order depends on the primary objective. For unified generation-understanding models, training generation first and then introducing understanding is the most effective strategy. In contrast, omitting progressive training or interleaving both tasks impairs overall optimization.

#### 5.6 Human Evaluation

Evaluation protocol. Beyond API-based evaluation, we further conducted an anonymous user study on our model and baseline models. We employ 30 participants to manually score a total of 150 audio clips, generated from 50 instructions, by our model, Tangoflux, and Next-GPT, respectively. The participants listened to the long-form audio generated by different models based on the same instruction. They scored the audio on four criteria: instruction-following, consistency, generation quality, and reasoning logic. The scores were averaged to compute user consistency. As shown in the Table 8, AudioStory consistently outperforms other competitors in terms of instruction-following, consistency, quality and reasoning logic.

Correlation between Gemini-based & human-based evaluation. Qualitatively, human evaluation results show our model performs the best among all three models, with the LLM + TTA model

- Table 8: Human evaluation of the generated audios for different methods on: instruct-following, consistency, fidelity, and reasoning logic.

Table 9: Correlation of Gemini and human scores.

Method Instruct-Follow Consist. Fidelity Reason. Logic

Across model

Across model

LLM + TangoFlux 3.52 3.22 3.58 3.19 LLM + NExT-GPT 3.10 2.56 2.87 3.14 AudioStory (Ours) 4.23 4.68 4.37 4.22

Kappa Coef. 0.91 0.83

###### Video Dubbing（video-to-text-to-audio）

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Instruction：

[Figure 87]

###### Audio input Story

Generate a classic Tom and Jerry style soundtrack for a moment in the cartoon that includes Snoopy being awakened by Woodstock. Woodstock then tries to bury a sack, but Snoopy ignores him. Later, Snoopy struggles with a garden hose, which inflates and blasts him with water. After cautiously reappearing, Snoopy is again surprised and blasted by the hose, this time emerging from his own doghouse. The total duration is 41.6s.

[Figure 88]

Generated Reasoning：

The caption can be divided into five points.\n\n1. **Initial Wake-Up: ** [0.0s – 5.1s] \n - **Story Detail:** Snoopy is awakened from a deep sleep near his snuffing fire.\n - **Derived Sound Effects:** Deep chuckle, enthusiastic 'Aha!', toffee toot, sigh of relief.\n\n2. **Woodstock's Burial Attempt:** [5.1s – 19.5s] \n - **Story Detail:** Woodstock tries to bury a sack, causing Snoopy to wake up.\n - **Derived Sound Effects:** 'Ooh' from Snoopy, rustling of sack, brief jump.\n\n3. **First Water Blaster Incident:** [19.5s – 26.8s] \n - **Story Detail:** A hose inflates and blasts Snoopy with water.\n - **Derived Sound Effects:** 'Ah!' from Snoopy, airfilled hose blowing.\n\n4. **Cautious Reappearance:** [26.8s – 38.2s] \n - **Story Detail:** Snoopy cautiously reappears after being hit by the hose.\n - **Derived Sound Effects:** Brief reaction, sudden launch from hose, explosive water spray.\n\n5. **Final Appearance:** [38.2s – 41.6s] \n - **Story Detail:** Snoopy emerges again from his doghouse, surprised by another blast.\n - **Derived Sound Effects:** 'Ah!', sudden launch from hose, forceful water spray.

[Figure 89]

###### Generated Caption：

- • A soft and gentle orchestral score typical of classic cartoons, featuring strings, woodwinds, brass, and percussion. It synchronizes with the action, enhancing comedic timing, tension, and drama, supporting both characters and musical gags. <timestamp>5.1<\timestamp>
- • The orchestral score slows, becoming suspenseful with pizzicato strings or light woodwinds, subtly highlighting the comedic tension of WoodStorck removing the sack from Snoopy's mouth. <timestamp>14.4<\timestamp>
- • A sharp, dramatic brass hit emphasizes Snoopy‘s sudden awakening and struggle. Tense music builds anticipation as he descends the hose, with strings and brass more prominent, reflecting his predicament. <timestamp>7.3<\timestamp>
- • The music shifts to a lighter, faster - paced melody, accompanied by woodwinds or percussion, then suddenly takes on a comical tone, reflecting Snoopy’s prank by the water pipes. <timestamp>7.4<\timestamp>
- • A dramatic crescendo leads to a massive orchestral explosion! Crashing cymbals, a loud timpani roll, and a frantic string glissando accompany Snoopy's watery, chaotic exit from the house. <timestamp>3.4<\timestamp>

[Figure 90]

- Figure 4: Case of naive video dubbing: First, we extract captions from the video, then write the extracted captions as instructions and send them to AudioStory for audio generation.

outperforming the LLM + any-to-any model. This aligns with the results from our Gemini evaluation. Quantitatively, we analyze the correlation between the human subjective and Gemini-based objective evaluation. We calculate Cohen’s kappa coefficient between these two evaluation protocols. Specifically, we compute the correlation across two dimensions, i.e., different comparative methods in Table 8 and different test samples. The results in Table 9 indicate a high correlation between the human and Gemini scoring distributions across various models and samples, validating the correctness of the proposed Gemini-based evaluation.

### 6 Extended Applications

We present AudioStory’s extended applications, including instruction-conditioned narrative audio generation, audio continuation, and video dubbing. Specifically, AudioStory generates long-form narrative audio from diverse multimodal instructions, remains closely aligned with input video and directives, and exhibits robust instruction-following.

###### Video Dubbing ( video-to-audio )

[Figure 91]

###### Multimodal Instruction：

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Please watch a video and identify the temporal consecutive key events or scene changes in the video, and generate a Tom & Jerry-style background soundtrack to the entire video.

[Figure 99]

[Figure 100]

Audio Story

[Figure 101]

###### Generated Reasoning：

The storyline is mainly about: Jerry escapes from tom at a dining table using quick, clever maneuvers and using a spoon and a pot lid to knock Tom unconscious, amidst dramatic orchestral music. The video can be divided into 4 key events. \n1. Jerry‘s Arrival [0.0, 1.5]. **Story Details**: Jerry tiptoes across a dinner table. **Sound Effects**: Light, sneaky pizzicato strings and woodwinds. \n2. Tom’s Ambush [1.5, 6.5]. **Story Details**: Tom, under a cloche, tries to eat Jerry. **Sound Effects**: A sudden orchestral crash and tense music. \n3. Jerry‘s Counter [6.5, 9.5]. **Story Details**: Jerry ties Tom’s tongue and hits him. **Sound Effects**: Comical xylophone runs and a sharp percussive hit. \n4. Pain and Escape [9.5, 13.5]. **Story Details**: Tom is in pain; Jerry leaps away. **Sound Effects**: Dizzying slide whistle and a triumphant flourish.

[Figure 102]

###### Generated Caption：

- • Playful pizzicato strings and a sneaky clarinet melody. A light, tiptoeing rhythm matches Jerry's steps, creating a mischievous and anticipatory atmosphere as he explores the grand dinner table. <timestamp>1.5<\timestamp>
- • A sudden, loud orchestral crash with blaring brass. The music turns into a tense, dramatic chase theme, reflecting Jerry's shock and Tom's menacing, wide-mouthed ambush under the cloche. <timestamp>5.0<\timestamp>
- • Fast-paced, comical xylophone runs and a slide whistle for the tongue-tying. A sharp percussive hit syncs with the spoon strike, creating a chaotic, slapstick atmosphere for Jerry's clever counter-attack. <timestamp>3.0<\timestamp>
- • A dizzying, descending slide whistle and wobbling brass notes for Tom's pained reaction. The piece concludes with a triumphant, quick string flourish, celebrating Jerry's successful and comical escape. <timestamp>4.0<\timestamp>

[Figure 103]

- Figure 5: Case of video dubbing: We input both the video and the instruction into the model, which parses the narrative into segments, extracts story details with corresponding audio elements, and sequentially generates aligned audio clips..

###### Audio Continuation

[Figure 104]

###### Instruction：

[Figure 105]

[Figure 106]

[Figure 107]

Audio + input Story

Understand the input audio, infer the subsequent events, and generate the continued audio of the coach giving basketball lessons to the players. The total duration is 36.6 seconds.

[Figure 108]

###### Generated Reasoning：

Based on the known event of \“a man speaking with an informative and engaging tone, accompanied by the sound of basketball in the background\”, we can infer that an instructor is explaining basketball moves, with the sound of shoes squeaking on the court demonstrating their practical use. Following this, we can further infer that the coach is guiding players through these techniques, including sudden quick jabs and shots, evidenced by sounds like ‘thwack’ and ‘ping’. Building on the first two captions, we can finally infer that the coach is intensely practicing ball handling and shooting skills, accompanied by gentle friction sounds, indicating a continuous and focused interaction with the basketball.

[Figure 109]

[Figure 110]

|[Figure 111]|
|---|

###### Generated Continuous Caption：

- • The whistle blows as the coach calmly describes the next move or technique, accompanied by the sound of shoes squeaking on the floor.<timestamp>13.9</timestamp>
- • Shoes make rapid squeaking sounds on the court floor. The sharp, loud thud of the basketball echoes clearly, with a rhythmic “bang” each time the ball hits the rim.<timestamp>12.6</timestamp>
- • The coach continues guiding the players, with the sounds of shooting and shoes sliding on the court continuing without pause.<timestamp>4.5</timestamp>

Figure 6: Qualitative cases of audio continuation.

###### Audio Continuation

[Figure 112]

###### Instruction：

[Figure 113]

[Figure 114]

[Figure 115]

Audio + input Story

Understand the input audio, infer the subsequent events, and generate the continued audio of three bulls fight intensively in a muddy pen, then abruptly stop, resulting in a calm atmosphere. The total duration is 45.8 seconds.

[Figure 116]

###### Generated Reasoning：

Based on the known event—a clear female voice commanding the bulls to stop fighting—it can be inferred that their intense physical conflict continues, evident from their deep growls, snorts, and heavy collisions. This power struggle, accompanied by heavy dragging and wrestling on the muddy ground, shows the bulls are resisting her command.\n\nThe battle appears unchanged, with the sounds of growls, snorts, and hoof friction continuing, emphasizing the bulls’ ongoing struggle for dominance.\n\nThe third section logically continues with the same auditory theme, with frequent growls, snorts, and heavy hoof clashes, indicating the fight is still fierce and unresolved.\n\nFinally, the deep growls, snorts, and hoof friction maintain the consistent audio theme, highlighting the bulls’ continued struggle on the muddy ground.

[Figure 117]

###### Generated Continuous Caption：

[Figure 118]

- • A clear female voice calls out, amid frequent deep grunts, breathy snorts, and dull thuds from bulls fighting. The heavy shuffling and scuffling on muddy ground convey their immense power and dominance.<timestamp>12.6</timestamp>
- • Frequent guttural grunts and breathy snorts from the bulls punctuate their fight, alongside heavy shuffling, scuffling hooves, and intermittent dull thuds. The soundscape conveys immense power, aggression, and the physical struggle for dominance. <timestamp>8.9</timestamp>
- • Deep guttural grunts, breathy snorts, shuffling hooves, and dull thuds punctuate the bulls' intense physical struggle for dominance on muddy ground.<timestamp>18.4</timestamp>

|[Figure 119]|
|---|

Figure 7: Qualitative cases of audio continuation.

Video dubbing. While prior experiments focused on text-based instructions, we extend AudioStory to a more practical scenario: video dubbing. This enhancement lets the model thoroughly analyze video content, reason about event sequences with timestamps, and generate synchronized audio. An initial approach uses Gemini-2.5-pro to generate a video summary caption, then performs instructionbased long-form narrative audio generation for the given video, as shown in Fig. 4.

However, this method poorly aligns audio with visual content. Therefore, we further modified the model to accept both video data and instructions as input. Specifically, the LLM reasons about the video to generate bridging tokens: it first comprehends the video’s overall content, then sequentially decomposes it into temporally ordered events, inferring each event’s visual details and corresponding audio information. Technically, we replace the LLM with a pretrained video MLLM (i.e., Qwen2.5VL [50]) and jointly train the LLM and audio generator using LoRA tuning. The training data is from the animated sound partition of AudioStory-10k. We provide the video dubbing results in Fig. 5.

Audio continuation. Given an audio segment and an instruction, our model performs audio continuation. AudioStory first reasons about the content of the subsequent audio to be generated, then proceeds with segment-by-segment generation. The cases are shown in Fig. 6 and Fig. 7.

### 7 Conclusion

In this paper, we tackle the key limitations of existing text-to-audio and unified models in generating long-form narrative audio in complex scenarios. We introduce AudioStory, a unified understandinggeneration model endowed with robust multimodal instruction-following and reasoning capabilities. To achieve this, we design an interleaved reasoning generation process, a decoupled bridging mechanism, and a progressive training strategy that jointly leverage the reasoning power of LLMs and strengthen the synergy between understanding and generation. Additionally, we present AudioStory10k, the first benchmark for long-form narrative audio generation, which includes fine-grained annotations of audio and audio-visual events with timestamps and detailed reasoning trajectories. Our comprehensive analyses cover reasoning forms, bridge query types, end-to-end training strategies for LLM-DiT integration, and the collaborative dynamics between understanding and generation, providing practical insights for future model development.

Limitations and Future Work. Since multimodal instruction for long audio generation remains underexplored, future work can incorporate more sophisticated designs, e.g., integrating multiple audio generators to better address the issue of overlapping audio segments. We also consider blending text and audio generation within the same autoregressive multimodal LLM, as well as delve into the relationship and synergy between audio generation and understanding.

### References

- [1] Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Amir Ali Bagherzadeh, Chuan Li, Rafael Valle, Bryan Catanzaro, and Soujanya Poria. Tangoflux: Super fast and faithful text to audio generation with flow matching and clap-ranked preference optimization. arXiv preprint arXiv:2412.21037,

2024. 1, 3, 7, 8

- [2] Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D. Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining, 2024. 1, 3, 7, 8
- [3] Zach Evans, Julian D. Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Stable audio open,

2024. 1, 3, 8

- [4] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. NExt-GPT: Any-to-any multimodal LLM. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 7, 8
- [5] Jinxiang Lai, Jie Zhang, Jun Liu, Jian Li, Xiaocheng Lu, and Song Guo. Spider: Any-to-many multimodal llm. arXiv preprint arXiv:2411.09439, 2024. 2, 3, 8
- [6] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer,

2023. 2

- [7] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models, 2023. 3, 8
- [8] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D. Plumbley. Audioldm: Text-to-audio generation with latent diffusion models, 2023. 3
- [9] Navonil Majumder, Chia-Yu Hung, Deepanway Ghosal, Wei-Ning Hsu, Rada Mihalcea, and Soujanya Poria. Tango 2: Aligning diffusion-based text-to-audio generations through direct preference optimization,

2024. 3

- [10] Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish, and Soujanya Poria. Text-to-audio generation using instruction-tuned llm and latent diffusion model, 2023. 3
- [11] Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities, 2024. 3
- [12] Moayed Haji-Ali, Willi Menapace, Aliaksandr Siarohin, Guha Balakrishnan, and Vicente Ordonez. Taming data and transformers for audio generation. arXiv preprint arXiv:2406.19388, 2024. 3
- [13] Rafael Valle, Rohan Badlani, Zhifeng Kong, Sang-gil Lee, Arushi Goel, Sungwon Kim, Joao Felipe Santos, Shuqi Dai, Siddharth Gururani, Aya Aljafari, et al. Fugatto 1: Foundational generative audio transformer opus 1. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [14] Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, and Wei-Ning Hsu. Voicebox: Text-guided multilingual universal speech generation at scale, 2023. 3
- [15] Apoorv Vyas, Bowen Shi, Matthew Le, Andros Tjandra, Yi-Chiao Wu, Baishan Guo, Jiemin Zhang, Xinyue Zhang, Robert Adkins, William Ngan, Jeff Wang, Ivan Cruz, Bapi Akula, Akinniyi Akinyemi, Brian Ellis, Rashel Moritz, Yael Yungster, Alice Rakotoarison, Liang Tan, Chris Summers, Carleigh Wood, Joshua Lane, Mary Williamson, and Wei-Ning Hsu. Audiobox: Unified audio generation with natural language prompts, 2023. 3
- [16] Huadai Liu, Jialei Wang, Rongjie Huang, Yang Liu, Heng Lu, Wei Xue, and Zhou Zhao. Flashaudio: Rectified flows for fast and high-fidelity text-to-audio generation, 2024. 3
- [17] Yang Jin, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, et al. Unified language-vision pretraining with dynamic discrete visual tokenization. arXiv preprint arXiv:2309.04669, 2023. 3
- [18] Zineng Tang, Ziyi Yang, Mahmoud Khademi, Yang Liu, Chenguang Zhu, and Mohit Bansal. Codi-2: In-context, interleaved, and interactive any-to-any generation. arXiv preprint arXiv:2311.18775, 2023. 3

- [19] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yu-Gang Jiang, and Xipeng Qiu. Anygpt: Unified multimodal llm with discrete sequence modeling. arXiv preprint arXiv:2402.12226, 2024. 3
- [20] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218, 2023. 3
- [21] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 3
- [22] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3
- [23] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 3
- [24] Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422, 2025. 3
- [25] Shijie Ma, Yuying Ge, Teng Wang, Yuxin Guo, Yixiao Ge, and Ying Shan. Genhancer: Imperfect generative models are secretly strong vision-centric enhancers. arXiv preprint arXiv:2503.19480, 2025. 3
- [26] Yuxin Guo, Shuailei Ma, Shijie Ma, Xiaoyi Bao, Chen-Wei Xie, Kecheng Zheng, Tingyu Weng, Siyang Sun, Yun Zheng, and Wei Zou. Aligned better, listen better for audio-visual large language models. arXiv preprint arXiv:2504.02061, 2025. 3
- [27] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. arXiv preprint arXiv:2305.11846, 2023. 3, 7, 8
- [28] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. arXiv preprint arXiv:2312.17172, 2023. 3, 8
- [29] Xubo Liu, Zhongkai Zhu, Haohe Liu, Yi Yuan, Qiushi Huang, Meng Cui, Jinhua Liang, Yin Cao, Qiuqiang Kong, Mark D Plumbley, et al. Wavjourney: Compositional audio creation with large language models. IEEE Transactions on Audio, Speech and Language Processing, 2025. 3
- [30] Xuenan Xu, Jiahao Mei, Chenliang Li, Yuning Wu, Ming Yan, Shaopeng Lai, Ji Zhang, and Mengyue Wu. Mm-storyagent: Immersive narrated storybook video generation with a multi-agent paradigm across text, image and audio. arXiv preprint arXiv:2503.05242, 2025. 3
- [31] Tiantian Geng, Teng Wang, Jinming Duan, Runmin Cong, and Feng Zheng. Dense-localizing audio-visual events in untrimmed videos: A large-scale benchmark and baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22942–22951, 2023. 4
- [32] Brandon Castellano. PySceneDetect: Intelligent scene cut detection and video splitting tool, 2018. Accessed: [Insert last accessed date]. 4
- [33] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4
- [34] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 4
- [35] OpenAI. Addendum to gpt-4o system card: 4o image generation, 2025. Accessed: 2025-04-02. 4
- [36] Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi. Fr\’echet audio distance: A metric for evaluating music enhancement algorithms. arXiv preprint arXiv:1812.08466, 2018. 4, 7
- [37] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 5

- [38] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 6
- [39] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 7
- [40] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023. 7, 9
- [41] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017. 7
- [42] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 721–725. IEEE, 2020. 7
- [43] Andrea Agostinelli, Timo I Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023. 7
- [44] Luoyi Sun, Xuenan Xu, Mengyue Wu, and Weidi Xie. Auto-acd: A large-scale dataset for audio-language representation learning. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5025–5034, 2024. 7
- [45] Aurora Linh Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello. Look, listen, and learn more: Design choices for deep audio embeddings. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856. IEEE, 2019. 7
- [46] Khaled Koutini, Jan Schlüter, Hamid Eghbal-Zadeh, and Gerhard Widmer. Efficient training of audio transformers with patchout. arXiv preprint arXiv:2110.05069, 2021. 7
- [47] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. AudioCaps: Generating captions for audios in the wild. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, Minneapolis, Minnesota, June

2019. Association for Computational Linguistics. 7

- [48] Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 736–740. IEEE, 2020. 7
- [49] Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. Advances in Neural Information Processing Systems, 35:28708–28720, 2022. 9
- [50] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 13

Appendix

- A Implementation Details

We provide detailed hyper-parameters of three training stages in Table 10. In Stage-II and Stage-III, the ratio of generation and understanding samples is 2:1. For LLM, we choose Qwen2.5-VL-3BInstruct and only tune LoRA to avoid overfitting. TangoFlux is employed as the initialization of DiT for audio generation. For the weights of different loss functions, we set the weight of Lmse for T5 regression, Ltext for next-token-prediction and Lflow for DiT as 5, 2 and 1, respectively.

Table 10: Detailed hyper-parameters of three training stages. Here, “A” denotes audio, “proj.” and “lr” are short for the projector and learning rate. We use 16 GPUs and report the overall batch size.

|Dimension| |Stage-I<br><br>| |Stage-II|Stage-III|
|---|---|---|---|---|---|
| | |Warm-up|Whole| | |
|Task| |A→T5<br><br>|A→T5 with DiT.<br><br>|A→T5 with DiT + Und.<br><br>|A→T5 with DiT + Und. + Reasoning|
|Dataset<br><br>| |AudioCaps, WavCaps| |I+AudioSetCaps (Q&A), VGGSound (Q&A), MusicCaps, Auto-ACD<br><br>|AudioStory-10k|
|Model|Trainable<br><br>|LLM, proj. (Tsemantic)<br><br>|LLM, all proj., DiT<br><br>|LLM, all projectors, DiT<br><br>|LLM, all proj., DiT|
| |Frozen<br><br>|Whipser, DiT|Whisper<br><br>|Whisper|Whisper|
|Training Config|batch size|512<br><br>|256<br><br>|Gen.: 8, Und.: 16|Gen.: 8, Und.: 16|
| |lr|1e-3<br><br>| |1e-4<br><br>|LLM (2e-5), DiT (5e-5)|
| |epoch|25|25<br><br>|10|10|

- B More Qualitative Cases

we present more cases for long-form audio generation. Our model could automatically derive the duration of each audio segment to be generated, as shown in Fig. 8. It can be observed that AudioStory could accurately determine the number of events based on the instruction and provide precise descriptions for each audio segment, including both the events themselves and their associated sound effects. Finally, AudioStory can precisely reason out the caption for each segment and generate the corresponding audio clips accordingly.

###### Instructional Audio Generation

[Figure 120]

###### Instruction：

[Figure 121]

[Figure 122]

###### Audio input Story

Construct a detailed soundscape that tells the story of a young beatboxer performs energetically outdoors, introduces themselves as 'yoh!', and requests tips. The total duration is 41.8 seconds.

[Figure 123]

###### Generated Reasoning：

The caption can be divided into four points:\n\n1. [0.0, 2.5]\n **Story Details:** The beatboxing begins indoors and transitions outdoors with a performer giving a shout.\n **Sound Effects:** Energetic beatboxing, voice saying \"Yoh!\", camera movement indicating a change in location.\n\n2. [2.5, 21.5]\n **Story Details:** A young man starts beatboxing complex rhythmic patterns, intensifies with \"skrrt\", and ends with a punchline.\n **Sound Effects:** Complex rhythmic beatboxing, intense \"skrrt\", loud breaths, informal ending statement.\n\n3. [21.5, 27.5]\n **Story Details:** Audience responds with laughter and brief verbal appreciation, and the performer clarifies in a friendly manner.\n **Sound Effects:** Laughter, verbal appreciation, friendly clarification in response to an audience member's request.\n\n4. [27.5, 41.8]\n **Story Details:** Performer thanks the audience, introduces as Yoh!\", asks for tips, and expresses excitement.\n **Sound Effects:** Grateful thank-you, friendly self-introduction, direct question, background noise from phone call.

[Figure 124]

###### Generated Caption：

- • Energetic beatboxing performance outdoors, abruptly shifting to a voice shout of \"Yoh!\" as the camera pans, indicating a move outside. <timestamp>2.5</timestamp>
- • A young man enthusiastically beatboxes, creating complex and rapid rhythmic patterns that intensify with sharp 'skrrt' sounds, followed by loud, audible breaths, ending with a punchline-like statement. <timestamp>19.0</timestamp>
- • Audience laughter followed by a brief, appreciative word from the performer, \"Okay.\" <timestamp>6.0</timestamp>
- • The performer thanks the audience, introduces themselves as 'Yoh!' and explains they don't have a YouTube channel, then asks for tips directly in a friendly tone. <timestamp>14.3</timestamp>

Figure 8: Long-form audio generation case #2.

###### Instructional Audio Generation

[Figure 125]

###### Instruction：

[Figure 126]

[Figure 127]

###### Audio input Story

Construct a detailed soundscape that tells the story of an instructor demonstrates common push technique errors in table tennis, emphasizing power, wrist angle, and swing size issues. The total duration is 43.6 seconds.

[Figure 128]

###### Generated Reasoning：

The caption can be divided into 4 points.\n\n1. [0.0, 11.8]\n**Story Details**: Instructor demonstrates a weak push technique leading to hitting the net.\n**Sound Effects**: Instructor's voice explaining the issue, followed by a ball hitting the net.\n\n2. [11.8, 24.8]\n**Story Details**: Instructor emphasizes the need for an upright wrist during the push.\n**Sound Effects**: Consistent, fast ball hitting sounds while demonstrating wrist movement.\n\n3. [24.8, 37.5]\n**Story Details**: Demonstrates a weak push by hitting the table and foot repeatedly.\n**Sound Effects**: Loud ball hitting sounds, followed by a ball hitting the foot, and repeated crashing on the table.\n\n4. [37.5, 43.6]\n**Story Details**: Fails miserly by hitting the ball with the paddle.\n**Sound Effects**: Instructor exclaiming an error, single dramatic, high-pitched paddle-ball impact sound.

[Figure 129]

###### Generated Caption：

- • An instructor explains that pushing this way lacks power and emphasizes an upright wrist technique to prevent the ball hitting the net. The ball does not.go in the hole effectively. <timestamp>11.8</timestamp>
- • The instructor discusses the importance of using an upright wrist during the execution of a push, emphasizing how it affects the ball's direction while demonstrating hand movement slowly at a consistent speed. <timestamp>13.0</timestamp>
- • The instructor repeats a sequence three times, ending with a weak push. A loud banging sound from the ball on the table, followed by a dull thud where the ball hits the foot, accompanied by repeated crashing sounds. <timestamp>12.7</timestamp>
- • The instructor exclaims with surprise and amusement, \"Oh my god, you hit the ball with the paddle!\" followed by a single, dramatic, high-pitched paddle-ball impact sound. <timestamp>6.1</timestamp>

Figure 9: Long-form audio generation case #3.

###### Instructional Audio Generation

[Figure 130]

###### Instruction：

[Figure 131]

[Figure 132]

###### Audio input Story

Design a descriptive audio capturing the essence of two men perform a melancholic song, sharing a lighthearted moment, concluding with smiles. The total duration is 43.5 seconds.

[Figure 133]

###### Generated Reasoning：

The caption can be divided into 3 points.\n\n1. [0.0, 14.0]\n **Story Details**: Two men start performing a melancholic song.\n **Sound Effects**: Gentle, melancholic piano melody.\n\n2. [14.0, 36.0]\n **Story Details**: One man sings about loss, creating a reflective mood.\n

**Sound Effects**: Earnest singing with steady piano music.\n\n3. [36.0, 43.5]\n **Story Details**: A lighthearted moment with smiles concludes the performance.\n **Sound Effects**: Brief pause in singing, followed by laughter and piano continuation.

[Figure 134]

###### Generated Caption：

- • Gentle, slightly melancholic melody played on the piano, creating a calm and anticipatory atmosphere. <timestamp>14.0</timestamp>
- • A man sings earnestly about loss and remembrance, with lyrics mentioning 'yesterday morning' and 'you were gone', accompanied by steady piano music creating a reflective mood. <timestamp>22.0</timestamp>
- • Singer briefly falters on the lyrics, followed by the sound of piano continuing. Laughter and lighthearted murmurs are exchanged between the singer and piano player before the piece concludes. <timestamp>7.5</timestamp>

Figure 10: Long-form audio generation case #3.

### C More Explorations of Residual Tokens

For residual tokens, we not only explore their forms and training strategies, but also investigate hyperparameters such as their quantity and fusion methods with semantic tokens.

The number of residual tokens. Here, we study the impact of different numbers of residual tokens, and report both single- and long-form audio generation, as in Table 11. For single-audio generation, too few residual tokens lead to degraded performance. We attribute this to two factors: less low-level complementary information is captured. Additionally, residual tokens help mitigate conflicts between the LLM and the DiT, while too few tokens weaken this effect. Conversely, an excessive number of tokens also degrades performance, because they increase the difficulty for the LLM to regress. Similar patterns could also be observed in the long-form scenario. Overall, 8 residual tokens are most suitable for both single and long audio scenarios.

Merging mechanism of residual tokens. For the merging mechanism between semantic and residual tokens, we also conduct in-depth explorations. Here, we mainly consider concatenation and cross-attention. The results of long-form audio generation are reported in Fig. 11. From the results,

Table 11: Detailed ablations of the number of residual tokens

Single Audio Long Audio FD ↓ FAD ↓ KL ↓ Consistency ↑

# Tokens

1 4.01 5.02 0.93 3.2 4 3.64 3.95 0.96 3.9 8 1.53 2.29 0.51 4.1

16 3.51 3.75 0.94 3.9

###### residual tokens generation results semantic tokens generation results

[Figure 135]

[Figure 136]

[Figure 137]

- Sample A
- Sample B

- Sample A
- Sample B

[Figure 138]

Figure 12: Visualizations of residual tokens.

one can observe that compared to concatenation, cross-attention ensures more effective fusion of the two features. Additionally, zero-initializing the final layer of the cross-attention module is necessary to prevent excessive disturbance to the semantic tokens at the beginning of training.

### D What do Residual Tokens Learn?

To thoroughly explore the effect of residual tokens, we provide visualizations in Fig. 12 (left). Specifically, the DiT takes only the residual tokens as the input and generates its corresponding audio. We subsequently concatenate all audio clips to constitute the whole long-form audio. The results reveal that for the same audio sample, the residual tokens capture temporally consistent low-level information, primarily reflecting coherence across different audio clips. In contrast, for different samples, the learned residual characteristics vary distinctly. By contrast, semantic tokens learn the underlying global semantics of the input audio and represent the progression of events over time, as illustrated in Fig. 12 (right).

concat cross-attn (rand) cross-attn (zero)

4.0

| |
|---|

3.5

LongAudioGeneration

| |
|---|

3.0

2.5

2.0

1.5

1.0

0.5

0.0

FD FAD

Figure 11: Ablations of token merging.

### E AudioStory-10k Benchmark

#### E.1 Dataset construction pipeline

The dataset construction pipeline is illustrated as follows. First, we filter videos to select those containing continuous audio events with visually grounded storylines. Next, in the event parsing

Annotation prompt

###### UnAV annotation prompt

Please watch a video clip and provide a detailed analysis of the video, aiming to generate high-quality input for subsequent text-to-speech synthesis. The results should be returned in JSON format and should include the following sections: (1) Main Event Analysis: Identify key events or scene changes in the video. Output the event's timestamp (start and end timestamps, two decimal places) and description. The description should integrate visual and audio information, vividly depicting the scenario and feelings at the time the event occurs. (2) Sound Effect Analysis: Identify sound effects (non-musical) in the video. Output the timestamp (start and end timestamps, two decimal places), type (e.g., footsteps, door closing, bird chirping), sound_description (description of the sound), and visual_context (description of the related visual scene) for each sound effect. The sound_caption must be detailed, vivid, and expressive, capturing the characteristics, quality, source, intensity, duration of the sound, and any emotions or environmental information it may convey, to facilitate high-quality sound synthesis. The visual_context should succinctly describe the visual imagery directly related to the sound effect. (3) Music Analysis: Identify background music or significant musical segments in the video. Output the timestamp (start and end timestamps, two decimal places) , type (style or genre), sound_description (description of the sound), and visual_context (description of the related visual scene) for each music segment. The sound_caption should detail the characteristics of the music, including melody, rhythm, instruments used, emotional atmosphere, and how it complements the visuals, also aimed at high-quality sound synthesis. The visual_context should succinctly describe the visuals present when the music occurs.

###### Tom & Jerrry annotation prompt

Please watch a video clip of 'Tom and Jerry' and provide a detailed analysis of the video, aiming to generate highquality input for subsequent text-to-speech synthesis. The results should be returned in JSON format and should include the following sections: (1) Main Event Analysis: Identify key events or scene changes in the video. Output the event's timestamp (start and end timestamps, two decimal places) and description. The description should integrate visual and audio information, vividly depicting the scenario and feelings at the time the event occurs. (2) Sound Effect Analysis: Identify sound effects (non-musical) in the video. Output the timestamp (start and end timestamps, two decimal places), type (e.g., footsteps, door closing, bird chirping), sound_description (description of the sound), and visual_context (description of the related visual scene) for each sound effect. The sound_caption must be detailed, vivid, and expressive, capturing the characteristics, quality, source, intensity, duration of the sound, and any emotions or environmental information it may convey, to facilitate high-quality sound synthesis. The visual_context should succinctly describe the visual imagery directly related to the sound effect. (3) Music Analysis: Identify background music or significant musical segments in the video. Output the timestamp (start and end timestamps, two decimal places) , type (style or genre), sound_description (description of the sound), and visual_context (description of the related visual scene) for each music segment. The sound_caption should detail the characteristics of the music, including melody, rhythm, instruments used, emotional atmosphere, and how it complements the visuals, also aimed at high-quality sound synthesis. The visual_context should succinctly describe the visuals present when the music occurs.

Figure 13: AudioStory-10k annotation prompts.

stage, we use Gemini-2.0-flash to decompose each video into multiple key audio events, each annotated with a timestamp, audio caption, and visual caption, as in Fig. 13. Finally, we perform instruction generation: based on fine-grained textual annotations, GPT-4o is used to generate diverse narrative instructions, accompanied by reasoning steps including task decomposition, audio event timeline planning, scene transitions, and emotional tone inference.

#### E.2 Benchmark Construction

Dataset prompt. The constructed dataset consists of instructions, reasoning, and audio clips, each with its caption and duration. Specifically, after parsing videos into key audio events using Gemini-2.0-flash as described in Sec. 3, we obtain annotations for each event including timestamps, audio captions, visual captions, and audiovisual event captions. For instruction generation, we use audio-visual event captions as the source input. A prompt, shown in Fig. 14, is used to summarize the whole caption of the full audio, which is then incorporated into a predefined instruction template to produce the final instruction. For reasoning generation, we provide GPT-4o with the whole caption along with the individual captions for each audio clip. GPT-4o is then prompted to infer the reasoning structure. The reasoning consists of two levels: a high-level decomposition indicating how the whole caption can be divided into several parts, followed by detailed descriptions for each part, including the corresponding events and sound-producing content. An example is illustrated in Fig. 15.

Benchmark evaluation. Along with the curated dataset, we also construct the long-form narrative audio generation task and its associated benchmark.

instruction prompt

I will provide you with “Main Event Analysis”, which has multiple descriptions in chronological order. Please combine the captions in each description while considering the timestamp information, and summarize them into a brief story summary, namely whole caption. Note: Only output the final whole caption within 20 words.

reasoning prompt

I will provide you with an input in JSON format, containing a "Sound Analysis" with multiple "sound description" in chronological order. Meanwhile, I will also provide you with a "whole_caption", representing the complete storyline summarized from these descriptions. Imagine that you are given only the whole caption and required to generate a detailed story description. You need to output the reasoning process. When doing so, you should not directly rely on the given "sound_description". You can make reasonable inferences and deduce the details of the story events and details, then infer the sounds objects and events, ensuring that the expanded sound details align with the given "sound description". The reasoning process could be divided into several points (same as the number of given "sound description" in "Sound Analysis"). The format MUST be: first, state how many points the caption can be divided into, and then elaborate on each point. Each point should follow the reasoning style explicitly with clear inferences and logical conjunctions, and state the story details and the derived sound effects. Each point MUST be within 15 words.

Figure 14: The datasets construction prompt.

AudioStiory-10k Cases Instruction

Craft an immersive soundscape that slowly unfolds dog howls in response to an emergency siren, then stops and looks at the camera as the siren fades.

###### Reasoning

The can be divided into 3 points:\n\n1. An emergency siren wails loudly outside.\nInference: An undulating siren wail slices through the room, prompting the dog to howl.\nDerived sounds: Emergency siren, dog howl.\n\n2. The howling dog shifts its attention.\nInference: The dog's howling session ends with a brief, highpitched bark as it stares.\nDerived sounds: Dog howl, sharp dog bark/yip.\n\n3. Human reacts softly to the dog's actions.\nInference: A faint chuckle from the person recording responds to the dog's intense gaze.\nDerived sounds: Faint human chuckle.

###### Audio Clips

###### Caption 1

An undulating wail of an emergency siren penetrates the room, rising and falling rhythmically. The moderate volume captures the dog's attention, prompting its vocal mimicry.

###### Caption 2

After a prolonged howling session, the dog emits a sharp, high-pitched yip, punctuating the howling and potentially shifting its attention to the person filming.

###### Caption 3

A faint, brief chuckle or breathy reaction is heard from the person recording, reacting to the dog's sudden stop and stare.

Figure 15: AudioStory-10k cases.

(1) Evaluation with Gemini-2.0-flash API, assessing consistency, coherence, instruction following, and reasoning logic. (2) Evaluation with traditional metrics to measure audio generation quality, including FD, FAD, and CLAP score, among others.

For the Gemini-based evaluation, we design tailored scoring criteria for each metric:

#### (a) Consistency.

- • Timbre and Sonic Cohesion Evaluate whether the primary sound sources maintain a generally consistent timbre and unified sonic characteristics.
- • Sound-Producing Entity Consistency Assess whether the implied sound-producing entities remain consistent, or if changes feel natural and logical within the audio.
- • Acoustic Environment Consistency Evaluate the background ambience, reverberation, and spatial impression for overall consistency or reasonable progression.
- • Transition Smoothness Assess whether the transitions between segments are smooth and free of jarring disruptions..

#### (b) Coherence.

- • Intentional Transitions Check whether transitions between segments are smooth, purposeful, and naturally connected.
- • Dynamic and Emotional Flow Assess if the dynamic and emotional progression feels consistent or evolves logically, without unjustified sudden shifts.
- • Tempo and Textural Compatibility Evaluate whether tempo, rhythm, and sonic textures between segments are compatible and blend cohesively.
- • Transition Smoothness Judge if segment connections are fluid, without abrupt or disjointed

#### (c) Instruction following.

- • Overall Semantic Alignment Evaluate whether the generated audio broadly reflects the intended scene, actions, and atmosphere described in the instruction. Minor differences are acceptable if the main idea remains clear.
- • Key Element Presence Verify whether the important sound-producing entities, actions, and environmental elements mentioned in the instruction are reasonably represented. Missing a few non-central elements is acceptable if key parts are present. Additional sounds not specified in the instruction are acceptable if they logically fit the scene and do not disrupt coherence.
- • Event Sequence and Logical Development Assess whether the overall event progression is reasonable according to the instruction. Small deviations in order are acceptable if they do not break the logical flow.
- • Specific Sound Detail Accuracy Evaluate whether important sound features (such as types of sounds, tonal qualities, or intensities) are reasonably reflected. Natural variations are acceptable as long as they do not change the overall character of the audio.

#### (d) Reasoning logic.

- • Overall Reasoning Logic Evaluate whether the model demonstrates a coherent, logical process in interpreting the instruction and planning the audio scene.
- • Caption-Instruction Alignment Assess whether the generated audio caption accurately reflects the instruction’s key content, sound-producing elements, and described environment.
- • Event Coverage Completeness Determine whether the inferred and described audio events fully cover the instruction’s core elements, with no major omissions.
- • Semantic and Temporal Accuracy Evaluate whether the implied timeline and semantic structure of the generated audio align with the instruction’s flow and intent.

- E.3 Single-Audio Evaluation Details To evaluate the audio generation model, four key metrics assess different aspects of performance:

- • Frechet Distance (FD) measures the statistical similarity between log-Mel spectrogram distributions of generated and real audio, quantifying low-level spectral fidelity (e.g., pitch, timbre) through mean and covariance comparisons in the mel-spectral domain.

- • Frechet Audio Distance (FAD) extends FD using high-level embeddings from a pre-trained audio encoder (e.g., VGGish), evaluating perceptual and semantic realism by comparing abstract features like instrument timbre, musical structure, and environmental acoustics.
- • CLAP Score calculates the cosine similarity between audio and text embeddings from a cross-modal model, assessing how well generated audio aligns with semantic prompts (e.g., textual descriptions of sound content or context).
- • KL-Divergence (KL) measures the distributional dissimilarity between generated and real audio features (spectral, latent, etc.), identifying consistency in probability distributions and helping debug issues like mode collapse or over-dispersion in outputs. Collectively, these metrics ensure a comprehensive evaluation of spectral realism, perceptual quality, semantic accuracy, and distributional consistency in generated audio.

