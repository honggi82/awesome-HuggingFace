## UNIFIED MULTI-TASK AUDIO-VIDEO JOINT GENERATION

Jun Wang∗ Chunyu Qiang∗ Yuxin Guo Yiran Wang Xijuan Zeng Feng Deng Kuaishou Technology {wangjun06, qiangchunyu}@kuaishou.com

###### Framework Comparison High-Fidelity & Tightly Synchronized Multi-Task Audio–Video Joint Generation.

- (a) Text to Audio-Video (T2AV)
- (b) Image to Audio-Video (TI2AV)
- (c) Image to Video (TI2V) (d) Text to Video (T2V) (e) Text to Audio (T2A)

[Figure 1]

[Figure 2]

[Figure 3]

T2V V2A

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

In a warm, sunlit kitchen, an elderly woman joyfully prepares dough for baking. A voiceover, likely hers, explains that in her kitchen, "love cooks." She adds that every bite is "like a hug," beautifully illustrating the deep connection between homemade food, affection, and the comfort of home.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

# arXiv:2601.04151v2[cs.CV]13Jan2026

T2A A2V

(a) Cascade

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

In a museum or gallery, a pensive Asian woman stands before a display of vintage books and documents, with old framed photographs on the wall behind her. Looking at a large album, she says, "These photos remind me of the past days," evoking a sense of nostalgia and quiet reflection.

[Figure 20]

[Figure 21]

Video Tower

[Figure 22]

[Figure 23]

| | |
|---|---|
| | |

Unified Audio-Video Generation

[Figure 24]

[Figure 25]

Audio Tower

###### IMAGE

(b) Dual Tower (c) Unified Single Tower (ours)

In a bright kitchen, a joyful man in an apron celebrates his cooking success. He proudly looks at the camera while an omelette sizzles in a pan. With a smile, he exclaims, "Yes! I did it! My first perfect omelette!" and gives a triumphant thumbs-up.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Superior Performance

↑ ↑

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

Two young men perform a rap and beatbox routine on a city sidewalk as a crowd watches at sunset.

A middle-aged man said in standard American English, "What's wrong? Do you need any help?"

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

↑

↑

A man in a grey suit is singing passionately in the auditorium.

[Figure 51]

A joyful boy and girl energetically perform on a stage with balloons, holding a toy microphone and maraca.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Catchy pop music with lively guitar riffs creates a funky stage atmosphere.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

A boy rides a large cat through glowing mushrooms, feedng it a small mushroom.

N/A

Figure 1: We propose APOLLO, a unified audio–video generation framework which delivers high fidelity, strong semantic and temporal alignment, and reliable instruction following in both joint and unimodal settings, with robust OOD generalization. Across tasks (T2AV/TI2AV/TI2V/T2V/T2A), it attains performance comparable to Veo-3 among open-source models.

ABSTRACT

Audio–video joint generation has progressed rapidly, yet substantial challenges still remain. Non-commercial approaches still suffer audio-visual asynchrony, poor lip–speech alignment, and unimodal degradation, which can be stemmed from weak audio–visual correspondence modeling, limited generalization, and scarce high-quality dense-caption data. To address these issues, we introduce APOLLO and delve into three axes—model architecture, training strategy, and data curation. Architecturally, we adopt a single-tower design with unified DiT blocks and an Omni-Full Attention mechanism, achieving tight audio–visual alignment and strong scalability. Training-wise, we adopt a progressive multitask regime—random modality masking to joint optimization across tasks, and a multistage curriculum, yielding robust representations, strengthening A-V aligned world knowledge, and preventing unimodal collapse. For datasets, we present the first large-scale audio–video dataset with dense captions, and introduce a novel automated data-construction pipeline which annotates and filters millions of diverse, high-quality, strictly aligned audio–video–caption triplets. Building on this, APOLLO scales to large datasets, delivering high-fidelity, semantically and temporally aligned, instruction-following generation in both joint and unimodal settings while generalizing robustly to out-of-distribution scenarios. Across tasks, it substantially outperforms prior methods by a large margin and achieves performance comparable to Veo 3, offering a unified, scalable path toward next-generation audio–video synthesis.

∗Equal contribution

- 1 INTRODUCTION

In ancient Greek mythology, Apollo presides over music, poetry, and prophecy, embodying artistic creation; he was later imbued with solar iconography and depicted as driving the sun chariot to illuminate the world. While for human perception and narrative, “illumination” arises not only from visible light and shadow but also from audible sensation: vision furnishes structure and space, while hearing shapes rhythm and affect, and their interplay yields a fully immersive experience. Accordingly, vision and auditory are complementary and jointly indispensable modalities for depicting the real world. Consequently, audio–video joint generation has emerged as one of the most prominent trends in generative AI. Commercial systems such as Veo 3 and Sora 2 have achieved impressive performance with strong semantic alignment, while diverse open-source models with varying architectures are also emerging. Nevertheless, research on joint audio–visual generation remains nascent, and current models—even some commercial systems—still exhibit audio–visual asynchrony, lip–speech mismatches, and degradation in unimodal quality. We attribute several factors: (1) Architecture. Most T2AV models like JavisDiT (Liu et al., 2025), UniVerse-1 (Wang et al., 2025a) and Ovi (Low et al., 2025) employ a single-tower architecture with a cross-attention module, resulting in limited audio-video interaction and alignment. (2) Training strategies and data. Currently, mainstream methods perform single-task training, which might bring about biased representations and struggle to exploit underlying audio-video correlation and world knowledge.

To this end, informed by the above observations, we identify several key contributing factors. (1) Architecturally, prevailing designs hinder thorough cross-modal interaction: most existing models adopt dual-tower architectures with modality-specific initialization, learn each modality independently, and rely on shallow fusion via cross-attention or adapters, which fails to fully align audio–visual features. (2) For data construction, there is a pronounced lack of diverse, high-quality, and densely annotated audio–video aligned generation datasets, as well as scalable, high-quality annotation methodologies for constructing them. (3) From a learning-strategy perspective, most existing methods are trained exclusively on text-to-AV generation. This single-task regime induces overfitting and representation bias, which in turn hinders generalization and degrades unimodal performance.

To address these issues, we propose APOLLO, which introduces coordinated improvements at the architectural, learning-strategy, and data levels. Specifically, on the architectural side, we adopt a singletower backbone with unified DiT blocks. Each block integrates an Omni-Full Attention module that jointly attends to four streams—audio, audio captions, video, and video captions—thereby facilitating cross-modal fusion and interaction, achieving tight A-V alignment and stronger coupling to textual conditions, which offers a higher scaling ceiling. For training, we design a progressive multitask training regime: random modality masking sustains joint optimization over T2AV/TI2AV/TI2V/T2V/T2A. A performance-adaptive pretrain–post train curriculum tunes data mixtures and quality, yielding robust, generalizable representations that exploit A/V correlations and world knowledge. From the data construction, we introduce an automated annotation pipeline that enables efficient model scaling. Building on these three insights, our model achieves strong performance in both joint and unimodal generation, delivering high fidelity, tight audio–visual alignment, natural outputs, and robust instruction following with favorable scaling behavior. On the Verse-Bench, it surpasses prior methods by a large margin and generalizes well to out-of-distribution (OOD) scenarios. Our main contributions are summarized as follows:

- • We introduce APOLLO, a unified framework for multi-task audio–video joint generation that, to our knowledge, is the first model to achieve performance comparable to Veo 3. APOLLO effectively resolves semantic and temporal audio–visual misalignment while delivering highfidelity generation.
- • Our key technical novelties lie in the unified single-tower architecture with the omni-full attention mechanism for seamless audio–visual fusion, and a progressive multi-task training strategy that promotes generalizable representations and prevents unimodal performance degradation.
- • We propose a large-scale, high-quality audio-video dataset consisting of 81 million samples with accurate dense captions, along with an automated data generation pipeline and a high-quality audio-video generation dataset.
- • Extensive experiments demonstrate that APOLLO consistently excels in both unimodal and joint audio–video generation, outperforming prior state-of-the-art on the unimodal benchmark and the AV joint-generation benchmark consistently.

- 2 RELATED WORKS

Text-to-Video Generation (T2V). Diffusion models have revolutionized video generation, with AnimateDiff (Guo et al., 2023) and Video Diffusion Models (Ho et al., 2022) leading the way. Stable Video Diffusion (Blattmann et al., 2023) emphasized the importance of large, high-quality datasets for performance. Early models used U-Net backbones, but Sora (Liu et al., 2024d) introduced the Diffusion Transformer (DiT) architecture (Peebles & Xie, 2023), trained on extensive video corpora. Open-source models like CogVideoX (Yang et al., 2024), HunyuanVideo (Kong et al., 2024a), and the WAN series (Wan et al., 2025) joined closed-source systems like Kling (Wang et al., 2025b) and Veo 2 (Deepanway et al., 2023). Most models share a common architecture: a 3D VAE compresses videos into spatiotemporal latents, and a DiT performs denoising. Data quality and scale remain crucial for model success, highlighting the importance of data curation and processing.

Image-to-Video Generation (I2V). Early I2V systems extended T2V by conditioning on a single frame via latent concatenation or CLIP-based feature injection (Liu et al., 2023b). Later, cascaded designs like I2VGen-XL (Zhang et al., 2023) and dual-injection frameworks like DynamiCrafter (Xing

- et al., 2024) introduced structured pipelines for better motion and appearance fidelity. Adapter-based approaches like LAMP (Notomi et al., 2015) and I2V-Adapter (Guo et al., 2024) integrate cross-frame attention. Recent works on motion modeling and stable sampling include Motion-I2V (Shi et al.,

2024) and FrameBridge (Wang et al., 2024b). Despite these advancements, I2V still faces challenges with curated datasets, limited long-range motion modeling, and the trade-off between appearance fidelity and motion realism.

Text-to-Audio Generation (TTA). Recent advances in generative models have propelled text-to-audio generation. Models like Make-An-Audio (Huang et al., 2023) and AudioLDM (Liu et al., 2023a) synthesize audio via iterative denoising of text-conditioned latent representations. Tango (Liu et al., 2024a), Audio Flamingo (Kong et al., 2024b), and others expand latent spaces and enhance crossmodal alignment. Stable Audio (Evans et al., 2025) employs hierarchical latent diffusion for highfidelity output. AudioStory (Guo et al., 2025) introduces a unified generation framework, achieving long-form audio generation for the first time, while flow-matching techniques like VoiceBox (Le

- et al., 2023) enable zero-shot style transfer. TangoFlux (Hung et al., 2024) optimizes text-audio alignment with CLAP-ranked preferences. Though these methods excel at semantic alignment, they are limited to short durations and lack flexibility for complex, evolving instructions, underscoring the need for TTA models that handle long, complex tasks.

Audio-Video Joint Generation (T2AV). Pioneering efforts like MM-Diffusion (Ruan et al., 2023) use coupled U-Net backbones, while DiT-based approaches dominate. AV-DiT (Wang et al., 2024a) adapts pre-trained image DiTs with lightweight adapters, and UniForm (Zhao et al., 2025) uses a unified single-tower architecture for audio-video tokens. Key challenges include precise spatio-temporal synchronization, addressed by methods like JavisDiT (Liu et al., 2025) (hierarchical prior), Ovi (Low

- et al., 2025) (twin-backbone design), and SyncFlow (Liu et al., 2024b) (dual-DiT with Rectified Flow Matching). Other research orchestrates unimodal experts, such as MMDisCo (Hayakawa et al., 2024) and Universe-1 (Wang et al., 2025a), which combine specialized models at the block level. Despite advances in architecture and data, most models focus on sound effects or music, leaving synchronized speech and video synthesis an underexplored challenge.

- 3 APOLLO

- 3.1 PRELIMINARY

Problem Definition. Our goal is to enable the generation of both audio and video within a single model, given various prior conditions. We denote the denoising network as ϵθ(·), the text condition

- as c. Let {zta}t∈[0,1] and {ztv}t∈[0,1] denote the latent variables at timestamp t for audio and video, respectively. Here, t = 0 denotes the final timestamp of pure Gaussian noise. During inference, ϵθ(·)

recursively performs denoising from t = 0 to t = 1 to produce the final generation, zˆ1a,zˆ1v, as shown below:

### zˆta′,zˆtv′ = ϵθ(zta,ztv,t,c), (1)

Conditional Flow-Matching. We employ flow matching as the denoising objective. The model needs to learn the velocity field that transforms pure noise p0 = N(0,I) to the underlying data

[Figure 65]

(c) Random Modality Masking

(a) Single-Tower Architecture (b) Omni-Full Attention

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

❄

[Figure 71]

[Figure 72]

𝑲𝑽 𝑲𝑻𝑽 𝑲𝑻𝑨 𝑲𝑨 Frozen

[Figure 73]

[Figure 74]

𝑸𝑽

Add

FNN FNN FNN FNN

Concat

𝑸𝑻𝑽

[Figure 75]

❄ ❄

[Figure 76]

Video Decoder

Audio Decoder

Scale& Norm

Scale& Norm

Scale& Norm

Scale& Norm

Text2Video Attention

𝑸𝑻𝑨

Text2Audio Attention

𝑸𝑨

Omni-Full Attention

Multimodal Transformer Block

𝑵 ×

Query Key Value

(d) MixD RoPE

|Height<br><br>Width<br><br>Time<br><br>Video tokens<br><br>Audio tokens<br><br>| | |
|---|---|
| | |
<br><br>| | |
|---|---|
| | |
|
|---|

|A ... ...<br><br>(3)<br><br>A<br><br>(4)<br><br>A<br><br>(5)<br><br>A<br><br>[Figure 77]|
|---|

|[Figure 78]<br><br>[Figure 79]<br><br> 0, 3, 0 <br><br> 2, 0, 0 <br> 2, 1, 0 <br> 2, 2, 0 <br><br><br> 0, 3, 1 <br><br> 2, 0, 1 <br> 2, 1, 1 <br> 2, 2, 1 <br><br><br> 0, 3, 2 <br><br> 2, 0, 2 <br> 2, 1, 2 <br> 2, 2, 2 <br><br><br>[Figure 80]<br><br>[Figure 81]<br><br> 1, 3, 0 <br> 2, 0, 0 <br><br><br> 2, 1, 0 <br> 2, 2, 0 <br><br><br> 1, 3, 1 <br> 2, 0, 1 <br><br><br> 2, 1, 1 <br> 2, 2, 1 <br><br><br> 1, 3, 2 <br> 2, 0, 2 <br><br><br> 2, 1, 2 <br> 2, 2, 2 <br><br><br>Height<br><br>Width<br><br>Time<br><br>[Figure 82]<br><br>[Figure 83]<br><br> 2, 3, 0 <br><br> 2, 0, 0 <br> 2, 1, 0 <br> 2, 2, 0 <br><br><br> 2, 3, 1 <br><br> 2, 0, 1 <br> 2, 1, 1 <br> 2, 2, 1 <br><br><br> 2, 3, 2 <br><br> 2, 0, 2 <br> 2, 1, 2 <br> 2, 2, 2 <br>|
|---|

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

❄ ❄ ❄ ❄ ❄

Video Encoder

Caption Encoder

Caption Encoder

TTS Tokenizer

Audio Encoder

-  b  Audio position ids
-  c  Embedding dimension allocation

Scale& Norm

Scale& Norm

Scale& Norm

Scale& Norm

|Time<br><br>Height<br><br>Width<br><br>|
|---|

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

A lone sailor woman sits on the boat, pipe clenched between her teeth. The ocean presents in a dark shade.

A lone sailor woman sits on the boat, pipe clenched between her teeth, murmuring soft words.

"This ocean, it’s a force, a wild, untamed might. "

Flow of V Flow of TV Flow of TA Flow of A

 a  Video position ids

 d  MixD RoPE Coordinates

Video

Video Caption Audio Caption Text / Lyrics Audio

Figure 2: Overview of APOLLO. The model takes four inputs: video, video-related text, audio-related text, and audio. Each input is individually encoded by respective encoders, then fed into the MM-DiT. The MM-DiT module outputs the latent variables of video and audio, which are then decoded separately into video and audio.

distribution pdata. In practice, we perform linear interpolation xt = (1 − t)x0 + tx1 to construct a distribution at timestamp t. Here, x0 ∼ p0 and x1 ∼ pdata. Given the condition c, the model ϵθ(·) is trained to predict the target velocity, i.e., constantly as ut = x1 − x0:

2 2

LFM = Et,c,x

0,x1 (x1 − x0) − ϵθ tx1 + (1 − t)x0,t,c

, where t ∼ U(0,1), x0 ∼ N(0,I), x1 ∼ pdata.

(2)

Latent Encoding. The model takes four inputs: video, video-related text, audio-related text, and audio, where video-related text represents the video caption and audio-related text represents the audio caption and speech text. Video is encoded by the 3d casual visual encoder from CogVideoX (Yang

- et al., 2024), We use Qwen3-8B Embedding (Zhang et al., 2025) as the encoder for audio and video captions.

- 3.2 SINGLE TOWER WITH FULL ATTENTION

Single Tower DiT. To ensure a thorough audio-video fusion, we employ a single-tower architecture. As shown in Fig. 2, following Stable Diffusion 3 (Esser et al., 2024), we employ Multimodal Diffusion (MMDiT) to take the sequences of all modalities as input and perform full attention. Specifically, there are four inputs, i.e., video, video-related text, audio-related text, and audio. Each type of input is individually encoded into latents with respective encoders, then fed into the MM-DiT. The MM-DiT module outputs the latent variables of video and audio in two streams, which are then decoded separately to perform video and audio generation.

Mixed Dimension Rotary Position Embedding (MixD-RoPE). Another key architectural innovation is Mixed Dimension Rotary Position Embedding (MixD-RoPE). As shown in Fig. 2 (d), to enhance the positional information introduced by various aspect ratios and duration in videos, we apply 3D RoPE encoding across three dimensions, i.e., temporal, width and height for video embedding. This

- 3D RoPE incorporates both absolute and relative position dependency in videos. For audio modality, we employ compatible temporal 1D positional encodings, while its position number is initialized by incrementing the maximum temporal position ID of the video modality by one. As a result, we build MixD-ROPE with a shared temporal position ID between video and audio modalities.

Omni-Full Attention. Previous works may employ separated spatial and temporal attention to reduce computational complexity, like UniForm (Zhao et al., 2025). However, as in CogVideoX (Yang et al., 2024), this separate attention mechanism requires extensive implicit information transmission, significantly increasing the learning complexity. Other works tailor two transformer towers for audio and video generation separately, e.g., AV-DiT, SyncFlow, JavisDiT, TAVGBench.However, they often adopt a multi-stage training approach, which is complex and resource-intensive. The two towers must first be pretrained separately, then finetuned together, increasing training time and resource consumption. To achieve more efficient training and more effective modality fusion, we employ the 3D text-video-audio hybrid full attention mechanism. As shown in Fig. 2, within the MM-DiT

module, the hidden states of video, video-related text, audio-related text, and audio are first scaled and normalized, then concatenated together for the attention calculation.

### Q = QV ⊙ QV T ⊙ QAT ⊙ QA, (3) K = KV ⊙ KV T ⊙ KAT ⊙ KA, (4) V = VV ⊙ VV T ⊙ VAT ⊙ VA, (5)

QK⊤ √dk

Attn(Q,K,V ) = Softmax(

)V, (6)

The attention values are then split into separate hidden states, which undergo scaling and normalization, residual connection, and feedforward, and subsequently fed to the next MM-DiT module. As a result, we achieve the unification of all input modalities within joint full-attention.

- 3.3 MULTI-TASK PROGRESSIVE TRAINING STRATEGY

Random Modality Masking. To learn generalizable and robust audio-visual representations for joint generation, we train the generative model with a broad spectrum of tasks. As a result, we propose to selectively adjust the mask of query and key for audio and video modalities. If we restrict the query and key to video embedding and video caption embedding, the model degenerates to a T2V model. Similarly, limiting the query and key to audio embedding and audio text embedding results in a T2A model. In this way, the model could not only handle joint generation, but also maintain the abilities of single-modality generation. Considering the scarcity of high-quality audio-video paired data, our method offers an alternative for training the T2VA model. We first pre-train APOLLO on T2V and T2A tasks, and then finetune our model on audio-video paired data to finally construct a T2VA model. The learning objectives for audio and video generation are in Eq. equation 7 and Eq. equation 8:

LT2A = ||ϵat − Maska(ϵθ(zta,ztv,c))||22, (7) LT2V = ||ϵvt − Maskv(ϵθ(zta,ztv,c))||22, (8)

where Maska is used to extract the audio token from the combined noise representation and Maskv is used to extract the vision tokens. In summary, LT2A and LT2V denote the single-modality tasks of T2A and T2V. To learn the generalizable and robust world knowledge of audio-visual correlation, we also incorporate several tasks of T2AV, I2V and I2AV. Consequently, the overall multi-task learning objective is as follows:

Loverall = LT2A + LT2V + LT2AV + LI2V + LI2AV (9)

Progressive Training Strategy. To efficiently train AV joint generation, we adopt a progressive multi-task learning framework with random modality masking applied throughout all stages:

- Stage-I: Pre-training. We pretrain the model on the large-scale, multi-scene data corpus to acquire atomic generation capabilities across all tasks, including cross-modal semantic alignment, temporal synchronization, high-fidelity audio synthesis, and precise visual feature construction, which ensures basic abilities of both single modality generation and joint generation, and provides a solid foundation for subsequent post-training.
- Stage-II: Specialized Post-training. We then specialize the model on its weaker abilities and tasks. Guided by evaluation metrics, we adaptively rebalance data distributions across scenarios and tasks to strengthen underperforming capabilities while preserving overall competency.
- Stage-III: Quality-Refined Post-training. Finally, we fine-tune the model on the manually-curated, high-quality dataset to refine generation fidelity and enhance robustness in complex scenes, yielding improvements in perceptual realism and overall generation quality.

- 4 DATASET CONSTRUCTION

Dataset Overview. Our dataset comprises automatically annotated samples. The dataset contains single-speaker speech, multi-speaker speech, singing, and natural sound clips, with an overall postfiltering retention rate of 27%.

##### (a) Data Filtering (b) Data Splitting (c) Dense Annotation

[Figure 98]

[Figure 99]

[Figure 100]

Sound

Video Quality Filtering

Voice Activity Detection

[Figure 101]

[Figure 102]

Audio Captioner

[Figure 103]

[Figure 104]

Singing

[Figure 105]

[Figure 106]

Scene Splitting

Vocal Type Classification

[Figure 107]

[Figure 108]

original data

Vocal Transcriptioner

[Figure 109]

Single Speech

[Figure 110]

[Figure 111]

[Figure 112]

Mulit-Speaker Diarization Multi Speech

Audio Quality Filtering

[Figure 113]

[Figure 114]

Figure 3: Overview of our Dataset Annotation Pipeline.

- 4.1 DATASET FILTERING

Video Filtering and Scene Splitting. We first filter video quality by modeling dynamic quality (subject motion ratio, camera stability), static quality (sharpness, aesthetics, color saturation), content naturalness (no excessive effects/watermarks), and safety. We discard those videos with low resolution, low SNR/MOS, or over 20% silence. We then apply scene splitting to ensure each sample contains only one scene.

Audio Filtering and Post Processing. We filter audio data by removing samples with low SNR, MOS, abnormal clipping, distortion, or noise, ensuring less than 20% silence, high fidelity, and consistent formatting. We then assess audio–visual consistency, using Synchformer for temporal alignment and ImageBind for semantic alignment, ensuring high synchronization in both temporal and semantic dimensions.

- 4.2 AUDIO-GUIDED DATA SPLITTING

We partition the dataset by audio type, separating vocal from non-vocal clips to form a sound split. From the vocal subset, we create singing, single-speaker speech, and multi-speaker speech splits, then apply dense captioning to each.

- 4.3 DENSE ANNOTATION AND INTEGRATION

We annotate each split with specialized models for speech transcripts, audio captions, and video captions, including both meta information and detailed content. For speech and singing, we extract speaker attributes (e.g., gender, age), while the sound split receives only audio captions. We use Whisper-Large-v3, SenseVoice, and Qwen2.5-Omni for transcription, Qwen2.5-Omni and Gemini 2.5-Pro for audio captions, and a video expert model for detailed video labels. All annotations are merged into unified dense captions.

5 EXPERIMENTS

In this section, we first present the experimental setup and implementation details (Sec. 5.1), then compare APOLLO with diverse baselines across multiple tasks (Sec. 5.2), complemented by qualitative results (Sec. 5.3). We further conduct ablations on the unified single-tower architecture, multi-task versus progressive training, and the role of 3D RoPE for native FPS (Sec. 5.4), collectively validating the effectiveness of our approach.

- 5.1 EXPERIMENTAL SETUP

Implementation details. APOLLO comprises 26B parameters with a flow-matching feed-forward dimension of 4096. The architecture incorporates 32 joint diffusion transformer layers combined with multimodal RoPE. For text encoding, we employ a 1024-dimensional TTS text encoder, while the caption encoder utilizes Qwen2.5-7B (Yang et al., 2025). The Audio-VAE processes input waveforms

- at 44.1 kHz and generates embeddings at 43 Hz, achieving a 1024× downsample ratio relative to the input sampling rate. The Video-VAE handles input videos with varying resolutions and frame rates,

Table 1: Main comparisons of audio-visual joint generation.

Video Audio TTS AV Consistency MS ↑ AS ↑ ID ↑ FD ↓ KL ↓ CLAP ↑ WER ↓ AV-A ↓ SNC ↑ IB-Score ↑ AudioLDM+TemoTkn T2A+A2V 0.05 0.12 0.07 2.05 2.53 0.229 - 1.15 2.68 0.137 OpenSora+FoleyGen

Method Framework

0.42 0.44 0.56 3.69 3.08 0.212 - 0.92 2.76 0.159 OpenSora+See&Hear 0.42 0.44 0.56 2.26 2.97 0.206 0.337 0.86 2.85 0.164 JavisDiT

T2V+V2A

0.18 0.36 0.22 1.95 5.17 0.228 0.256 0.92 3.94 0.231 SVG 0.40 0.41 0.25 1.55 3.62 0.080 - 0.72 4.07 0.206 Universe-1 0.20 0.47 0.25 1.55 1.25 0.160 0.180 0.98 3.92 0.198 Ovi 0.58 0.48 0.46 1.50 1.19 0.224 0.035 0.82 4.28 0.214

T2AV

APOLLO (Ours) Unified T2AV 0.48 0.51 0.59 1.36 1.06 0.232 0.028 0.65 6.79 0.316

producing embeddings at 3 Hz with 16× compression applied to both height and width dimensions. We train the model using the Adam optimizer with an initial learning rate of 1e-4.

Baseline Methods. We select two canonical types of methods, including (1) Cascaded generation: these methods typically employ sequential T2V+V2A or T2A+A2V. Here, we employ AudioLDM2 (Liu et al., 2024c) to perform the T2A task, while OpenSora (Zheng et al., 2024) for T2V. (2) Joint generation. Recent works like JavisDiT (Liu et al., 2025), UniVerse-1 (Wang et al., 2025a) and Ovi (Low et al., 2025) employ a dual-tower architecture with dedicated interaction layers between them.

Evaluation Metrics. To evaluate the model’s generation capabilities, we test T2AV tasks on video, audio, and audio-video consistency. Following Universe-1, we use Verse-Bench for T2AV tasks. For video quality, we report the Motion Score (MS) based on RAFT (Teed & Deng, 2020) optical flow for dynamic realism and the Aesthetic Score (AS), a composite metric from MANIQA (Yang et al., 2022), aesthetic-predictor-v2-5 (discus0434, 2024), and Musiq (Ke et al., 2021) for visual fidelity. Identity preservation is measured by ID Consistency (ID), calculated via DINOV3 (Siméoni et al.,

- 2025) feature similarity. For audio quality, we use Fréchet Distance (FD) and KL Divergence on mel-spectrograms from PANNs (Kong et al., 2020). Semantic alignment is evaluated using the CLAP score (Wu et al., 2023). For synchronization, we report AV-A (Audio-Video Alignment) distance from Synchformer (Iashin et al., 2024) and SyncNet Confidence (SNC) score (Chung & Zisserman,

2016) for lip sync. Global cross-modal alignment is measured by ImageBind (IB).

- 5.2 COMPARISON WITH EXISTING METHODS

APOLLO demonstrates robust audio–video joint generation. In Table 1, APOLLO achieves stateof-the-art performance, surpassing the two prior methods by a large margin. Cascaded approaches perform poorly due to error accumulation and strong dependence on upstream generation quality, while existing joint models exhibit only moderate audio–video consistency and noticeable unimodal degradation. In contrast, APOLLO attains substantially better A/V consistency and synchronization, which we attribute to the unified single-tower architecture and omni-full attention mechanism.

APOLLO effectively guarantees unimodal performance. We then assess unimodal performance. As shown in Table 1, APOLLO delivers high audio and video quality in joint generation—surpassing cascaded and joint baselines by 34% and 18%, respectively—while multitask training yields more generalizable representations. Moreover, APOLLO outperforms specialized T2A and T2V models on their respective tasks, indicating that leveraging complementary audio–visual knowledge strengthens unimodal representations and further improves single-modality generation quality.

APOLLO consistently maintains a performance advantage across multiple tasks. Table 1 provides a comprehensive evaluation of APOLLO across a broad suite of tasks—including TI2AV, TI2V, T2V, and T2A—with comparisons to task-specialized baselines. APOLLO delivers consistently strong results, matching or surpassing specialized state-of-the-art models.

- 5.3 QUALITATIVE RESULTS

Lip-Sync Accuracy. Fig. 4 (a–b) highlights lip-sync performance. APOLLO achieves phoneme-level alignment between mouth movements and audio—covering mouth openings, lip–teeth shapes, and

- (a) Lip-Sync Accuracy

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

(c) Image to Video (TI2V)

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

- (b) Emotional Expressiveness
- (c) Singing, Rap or Other Motion Performance
- (d) Audio–Visual Synchronization and Audio Overlapping

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

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

- (e) Comparison with Veo 3

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Veo 3

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

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

ours

[Figure 197]

[Figure 198]

Figure 4: Qualitative evaluation of audio-video joint generation across various aspects.

Table 2: Comparison of different methods. The Dual Tower uses standard cross-attention, while the Single Tower utilizes our proposed Omni-Full Attention.

Method Video Audio TTS Audio-Video Consistency ID ↑ MOS ↑ CLAP ↑ WER ↓ DeSync ↓ Sync-conf ↑ IB ↑

Dual Tower 0.62 62.02 0.139 0.675 1.163 3.762 0.126 Single Tower 0.80 93.11 0.232 0.028 0.650 6.787 0.316

tongue positions, while Universe-1 and Ovi suffer from misalignment, delay, and clear audio–visual mismatch.

Emotional Expressiveness. From Fig. 4 (c), APOLLO generates characters with expressive emotions: facial cues (eyes, mouth curvature, muscle tension) are highly consistent with the audio’s affective tone (joy, sadness, excitement, lethargy), reflecting natural audio–visual fusion of prosody and dynamics. In contrast, Universe-1 and Ovi often produce distorted or emotional expressions.

Singing and Rap Performance. As shown in Fig. 4, APOLLO yields performances where pitch, rhythm, and breath control are tightly aligned across audio and visuals—vibrato, melisma, and dynamic changes naturally match breathing patterns and facial expressions, consistent with human expectations of singing. In contrast, Universe-1 and Ovi show clear lip-sync failures, resulting in pronounced incongruity and reduced realism, especially for rap.

Audio–Visual Synchronization and Audio Overlapping. As shown in Fig. 4, APOLLO jointly generates background music and sound effects that are emotionally consistent with the video, with synchronized timing, realistic acoustics, and high fusion, thereby enhancing immersion. In contrast, baseline methods struggle to produce overlapping sounds and exhibit poor audio–visual consistency.

Image to Audio-Video. Fig. 4 presents TI2AV and TI2V results. Our model preserves high identity consistency with the input image while generating plausible camera motion and dynamics, whereas baselines exhibit identity drift, large visual discrepancies, and mechanical movements.

- 5.4 ABLATION AND ANALYSIS

Architectural Effectiveness. To compare single- and dual-tower architectures for audio–video generation, we feed audio and video features into separate mm-DiT branches, with a randomly initialized audio tower due to the lack of a matching DiT backbone. Each block uses full attention

Table 3: Ablation of multi-task masking, with arrows indicating optimization directions.

Method Video Audio TTS Audio-Video Consistency ID ↑ MOS ↑ CLAP ↑ WER ↓ DeSync ↓ Sync-conf ↑ IB ↑

T2V 0.71 - - - - - T2V+T2AV 0.76 88.181 0.188 0.044 0.895 5.024 0.201 All Tasks(Ours) 0.80 93.106 0.232 0.028 0.650 6.787 0.316

[Figure 199]

Figure 5: Ablations of different training stages. Metrics include video, audio, TTS, and audio-video consistency, with arrows indicating optimization directions.

and cross-attention for feature alignment. We also conduct an ablation with a pretrained audio tower. As shown in Table 2, the single-tower model outperforms the dual-tower variant in audio quality, video quality, and audio–video consistency, with visualizations confirming better cross-modal alignment. Although the pretrained audio tower converges faster, its performance is suboptimal due to a distribution mismatch with video features, hindering alignment.

Advantages of Multi-Task Masking. As shown in Table 3, for T2AV joint generation, our multi-task model significantly outperforms a counterpart trained solely on T2AV. This approach captures crossmodal audio–video correlations and complementary cues, outperforming video-only models on T2V and I2V. The unified multi-task training also produces robust, generalizable representations that scale well with data and compute, and the model generalizes effectively to I2AV and I2V, demonstrating high transferability.

Gains from Progressive Training Strategy. As in Figure 5, the progressive training first equips the model with basic audio–video generation capabilities (a). In the post-training phase, the full model. The SP stage further boosts previously weak skills (e.g., IB score by 0.1), and post-training on high-quality data brings an additional overall gain over (b). Removing the entire progressive schedule instead causes a significant drop, confirming the effectiveness of our multi-stage training strategy.

- 6 CONCLUSION

We identify key failure modes in audio–video generation, such as asynchrony, lip–speech mismatch, and unimodal degradation, caused by suboptimal architectures, misaligned data, and single-task training. To address these, we propose APOLLO, a unified multi-task framework with a single-tower backbone and Omni-Full Attention for cross-modal interaction, alongside a progressive training strategy and an automated annotation pipeline. This pipeline produces a high-quality, annotated audio–video dataset for scalable training. APOLLO outperforms prior state-of-the-art on unimodal and joint audio-video benchmarks, becoming the superior model comparable to Veo 3. We hope this work could provide a clear direction and catalyze deeper research in audio–video generation.

ACKNOWLEDGEMENT

We acknowledge the contributions from (sorted by first name): Boyuan Jiang, Chen Zhang, Feng Deng, Haorui Zheng, Jiachen Zheng, Jiahao Wang, Jiahui Zhao, Jingbin He, Jingke Li, Jingru Zhao, Junjie Yan, Kang Yin, Le Wang, Liang Hou, Lingyu Zou, Ming Wen, Nan Li, Peihan Li, Pengfei Cai, Pengfei Wan, Qianyue Hu, Shiyao Wang, Teng Ma, Xiaopeng Wang, Xiaoyu Shi, Xin Tao, Xu Li, Yan Zhou, Youjun Chen, Yu Zhao, Yuan Gao, Yuejiao Wang, Yun Li, Yushen Chen, Yuzhe Liang, Zewen Song, Zhongliang Liu, Zihan Li, Zihao Ji, Ziyang Yuan, and Ziyu Zhang.

REFERENCES

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Asian conference on computer vision, pp. 251–263. Springer, 2016.

Ghosal Deepanway, Majumder Navonil, Mehrish Ambuj, and Poria Soujanya. Text-to-audio generation using instruction-tuned llm and latent diffusion model. arXiv preprint arXiv:2304.13731, 2023.

discus0434. aesthetic-predictor-v2-5, 2024. URL https://github.com/discus0434/ aesthetic-predictor-v2-5/.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Stable audio open. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2025.

Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, et al. I2v-adapter: A general image-to-video adapter for diffusion models. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–12, 2024.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Yuxin Guo, Teng Wang, Yuying Ge, Shijie Ma, Yixiao Ge, Wei Zou, and Ying Shan. Audiostory: Generating long-form narrative audio with large language models. arXiv preprint arXiv:2508.20088, 2025.

Akio Hayakawa, Masato Ishii, Takashi Shibuya, and Yuki Mitsufuji. Mmdisco: Multi-modal discriminator-guided cooperative diffusion for joint audio and video generation. arXiv preprint arXiv:2405.17842, 2024.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646,

- 2022.

Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. In International Conference on Machine Learning, pp. 13916–13932. PMLR,

- 2023.

Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Amir Ali Bagherzadeh, Chuan Li, Rafael Valle, Bryan Catanzaro, and Soujanya Poria. Tangoflux: Super fast and faithful text to audio generation with flow matching and clap-ranked preference optimization. arXiv preprint arXiv:2412.21037, 2024.

Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5325–5329. IEEE, 2024.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5148–5157, 2021.

Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2880–2894, 2020.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024a.

Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities. arXiv preprint arXiv:2402.01831, 2024b.

Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, et al. Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36:14005–14034, 2023.

Haiyang Liu, Xingchao Yang, Tomoya Akiyama, Yuantian Huang, Qiaoge Li, Shigeru Kuriyama, and Takafumi Taketomi. Tango: Co-speech gesture video reenactment with hierarchical audio motion embedding and diffusion interpolation. arXiv preprint arXiv:2410.04221, 2024a.

Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audioldm: Text-to-audio generation with latent diffusion models. arXiv preprint arXiv:2301.12503, 2023a.

Haohe Liu, Gael Le Lan, Xinhao Mei, Zhaoheng Ni, Anurag Kumar, Varun Nagaraja, Wenwu Wang, Mark D Plumbley, Yangyang Shi, and Vikas Chandra. Syncflow: Toward temporally aligned joint audio-video generation from text. arXiv preprint arXiv:2412.15220, 2024b.

Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:2871–2883, 2024c.

Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, et al. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. arXiv preprint arXiv:2503.23377, 2025.

Ruyang Liu, Jingjia Huang, Ge Li, Jiashi Feng, Xinglong Wu, and Thomas H Li. Revisiting temporal modeling for clip-based image-to-video knowledge transferring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6555–6564, 2023b.

Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024d.

Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin backbone cross-modal fusion for audio-video generation. arXiv preprint arXiv:2510.01284, 2025.

Tsugunori Notomi, Yasuyoshi Mori, Norihiro Tomita, and Hidetoshi Kanda. Loop-mediated isothermal amplification (lamp): principle, features, and future prospects. Journal of microbiology, 53(1): 1–5, 2015.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10219–10228, 2023.

Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–11, 2024.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pp. 402–419. Springer, 2020.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025a.

Jun Wang, Xijuan Zeng, Chunyu Qiang, Ruilong Chen, Shiyao Wang, Le Wang, Wangjing Zhou, Pengfei Cai, Jiahui Zhao, Nan Li, et al. Kling-foley: Multimodal diffusion transformer for high-quality video-to-audio generation. arXiv preprint arXiv:2506.19774, 2025b.

Kai Wang, Shijian Deng, Jing Shi, Dimitrios Hatzinakos, and Yapeng Tian. Av-dit: Efficient audiovisual diffusion transformer for joint audio and video generation. arXiv preprint arXiv:2406.07686,

- 2024a.

Yuji Wang, Zehua Chen, Xiaoyu Chen, Yixiang Wei, Jun Zhu, and Jianfei Chen. Framebridge: Improving image-to-video generation with bridge models. arXiv preprint arXiv:2410.15371,

- 2024b.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pp. 399–417. Springer, 2024.

An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, Jiandong Jiang, Jianhong Tu, Jianwei Zhang, Jingren Zhou, et al. Qwen2. 5-1m technical report. arXiv preprint arXiv:2501.15383, 2025.

Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1191– 1200, 2022.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

Lei Zhao, Linfeng Feng, Dongxu Ge, Fangqiu Yi, Chi Zhang, Xiao-Lei Zhang, and Xuelong Li. Uniform: A unified diffusion transformer for audio-video generation. arXiv e-prints, pp. arXiv– 2502, 2025.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

