# arXiv:2509.09595v2[cs.CV]17Sep2025

September 18, 2025

## Kling-Avatar: Grounding Multimodal Instructions for Cascaded Long-Duration Avatar Animation Synthesis

Yikang Ding∗ Jiwen Liu∗ Wenyuan Zhang Zekun Wang Wentao Hu Liyuan Cui Mingming Lao Yingchao Shao Hui Liu Xiaohan Li Ming Chen Xiaoqiang Liu Yu-Shen Liu Pengfei Wan

Kling Team, Kuaishou Technology Project Page: https://klingavatar.github.io/

|[Figure 1]<br><br>Expressive and Vivid Portrait Animation<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>Precise Lip-Sync<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>Zero-shot Open Scenarios<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>|
|---|

- Figure 1: Conditioned on audio, image, and user prompts, Kling-Avatar generates high-fidelity portrait animations through instruction grounding and semantic planning. The results exhibit vivid emotions, rich actions, and precise lip synchronization, while also showing strong generalization to open scenarios such as anime, cartoons, and stylized characters.

2.39

1.77

2.06

1.17

1.37

0.42

0.57

0.48

0.86

0.73

Overall Lip Sync

Visual Quality

Control Response

Identity Consistency

Ours OmniHuman-1

1.37

2.35

1.76

0.76

0.86 0.73

0.43

0.57

1.32

1.17

0.00

0.50

1.00

1.50

2.00

2.50

3.00

Overall Lip Sync

Visual Quality

Control Response

Identity Consistency

Ours Heygen

[Figure 46]

[Figure 47]

- Figure 2: Benchmark performance of Kling-Avatar against its counterparts in terms of GSB metrics. We achieve superior performance on the overall metric as well as across most of sub-dimensions.

∗Equal contribution. Sorted in alphabetical order by surname.

### Abstract

Recent advances in audio-driven avatar video generation have significantly enhanced audio-visual realism. However, existing methods treat instruction conditioning merely as low-level tracking driven by acoustic or visual cues, without modeling the communicative purpose conveyed by the instructions. This limitation compromises their narrative coherence and character expressiveness. To bridge this gap, we introduce Kling-Avatar, a novel cascaded framework that unifies multimodal instruction understanding with photorealistic portrait generation. Our approach adopts a two-stage pipeline. In the first stage, we design a multimodal large language model (MLLM) director that produces a blueprint video conditioned on diverse instruction signals, thereby governing highlevel semantics such as character motion and emotions. In the second stage, guided by blueprint keyframes, we generate multiple sub-clips in parallel using a first-last frame strategy. This global-to-local framework preserves fine-grained details while faithfully encoding the high-level intent behind multimodal instructions. Our parallel architecture also enables fast and stable generation of long-duration videos, making it suitable for real-world applications such as digital human livestreaming and vlogging. To comprehensively evaluate our method, we construct a benchmark of 375 curated samples covering diverse instructions and challenging scenarios. Extensive experiments demonstrate that Kling-Avatar is capable of generating vivid, fluent, long-duration videos at up to 1080p and 48 fps, achieving superior performance in lip synchronization accuracy, emotion and dynamic expressiveness, instruction controllability, identity preservation, and cross-domain generalization. These results establish Kling-Avatar as a new benchmark for semantically grounded, high-fidelity audio-driven avatar synthesis. Demonstration videos are available in our project page: https://klingavatar.github.io/.

### 1 Introduction

Avatar animation synthesis translates multimodal references into temporally coherent facial expressions, lip movements and body gestures, enabling interactions with machines that feel conversational and embodied. As a communicative medium, a speaking avatar can convey intent and affect with high fidelity, turning abstract ideas into vivid, situated performances that maintain user attention and improve comprehension. This capability opens broad opportunities across virtual assistants, education, media content creation, and immersive telepresence. Building such avatars requires models that couple realism with fine-grained controllability and reliable synchronization, which define the core challenge and motivate the approach developed in this work.

Recently, Video Diffusion Transformers (DiT) (Chen et al., 2025c; Cui et al., 2025b; Jiang et al., 2024; Tian et al., 2024; Peng et al., 2025; Wei et al., 2025) have emerged as a general paradigm for generating visually compelling content conditioned on multimodal signals such as images, speech, and prompts. Prior work has advanced precise facial expression and lip synchronization (Jiang et al., 2024; Fei et al., 2025; Tian

- et al., 2024), coordinated body motion (Wang et al., 2025a; Gan et al., 2025), and data scaling (Lin et al., 2025; Jiang et al., 2025). However, these advances remain insufficient for highly realistic portrait synthesis, as we want the system not only to hear and to read but also to understand these inputs, so that it can produce natural and empathetic videos aligned with user intents. Without such understanding, current approaches often treat each conditional signal independently and capture only shallow correlations, which leads to semantic conflicts across modalities and affect. For example, an avatar may sing a sorrowful song while smiling, which is visually polished yet inconsistent with human expectations. In addition, existing approaches often rely on motion frames for video continuation, which poses significant challenges for maintaining consistency and stability in long-duration generation.

To bridge this gap, we introduce Kling-Avatar, a novel cascaded framework for portrait animation that faithfully follows multimodal instructions and synthesizes high-quality and long-duration avatar videos. Drawing inspiration from the capabilities of unifying understanding and generation of multimodal large language models(MLLMs) (Team et al., 2023; Xu et al., 2025; Hong et al., 2025), we design an MLLM Director that consolidates multimodal instructions into a structured storyline. This storyline encodes high-level plans such as scene layout, camera positioning, character motions, as well as implicit emotions and atmosphere, ensuring that the generated content aligns with the intended narrative arc and expressive trajectory. A blueprint video is first generated conditioned on the global script, followed by

multiple sub-clips generation in parallel conditioned on the blueprint keyframes. The MLLM Director continuously provides fine-grained guidance based on multimodal context, ensuring local dynamics and visual details. By densely selecting anchor frames and enabling parallel generation, our cascaded framework supports fast and stable synthesis of arbitrarily long videos, offering a promising solution for long-term digital human video generation.

For data preparation, we collect a dataset covering diverse scenarios such as dialogues, films, and speeches. To ensure data quality, we employ a series of expert models for rigorous filtering, including mouth-clarity recognition, stage-cut detection, audio-lip synchronization checking, and video quality scoring. To validate the effectiveness of our method, we construct a unique benchmark containing 375 “reference frame–audio” pairs. For these cases, we carefully design challenging instructions that include images from diverse categories, audio spanning different languages and speech rates, and text prompts with explicit control over emotion and dynamics. This benchmark builds up a comprehensive evaluation of different methods across multiple dimensions. We highlight representative generation results in Fig. 1, where Kling-Avatar produces expressive, vivid, long-duration portrait animations with rich emotions and dynamics, while maintaining strong generalization to open-domain scenarios. As indicated by the comparisons in Tab. 2 against leading competitors OmniHuman-1 (Lin et al., 2025) and HeyGen (hey), Kling-Avatar achieves superior performance in terms of lip synchronization accuracy, visual fidelity, instruction conditioned expressiveness, and identity preservation over long-duration generation. These results establish Kling-Avatar as the new benchmark for controllable, high-fidelity digital portrait animation synthesis. We summarize our contributions as follows:

- • MLLM Director with unified instruction grounding. We introduce an MLLM Director that grounds multimodal instructions into unified global plan, providing a new perspective that lifts portrait video generation from tracking low-level cues to semantic and intent understanding.
- • Cascaded avatar animation synthesis framework. We design a two-stage generation pipeline that first establishes high-level semantic guidance and then refines local dynamics, enabling long-duration video generation with coherent and expressive performances.
- • Curated data construction pipeline. We develop a data filtering pipeline powered by expert models for quality control, and further construct a challenging benchmark to enable comprehensive evaluation of digital human generation systems.
- • High-fidelity performance and strong generalization. Kling-Avatar produces state-of-the-art coherent and vivid portrait animations with precise lip synchronization, rich facial expressions and accurate response for multimodal instructions across diverse scenarios.

### 2 Method

Given a conditioning image, audio, and text prompt, Kling-Avatar aims to generate fluent and lifelike portrait animations with precise lip synchronization, accurate instruction following, and support for long-term extrapolation. As illustrated in Fig. 3, our framework is a two-stage generation pipeline guided by an MLLM Director. In the following sections, we first present the motivation and implementation of using MLLMs for instruction grounding and control (Sec. 2.1). We then introduce the cascaded generation framework (Sec. 2.2) for long-duration video synthesis, followed by our efforts in data construction for training and benchmarking (Sec. 2.3). Finally, we describe several key strategies for training and inference (Sec. 2.4).

#### 2.1 Grounding Multimodal Instructions with MLLMs

Current digital human video generation methods focus on conditioning strategies such as sliding windows or multi-scale injection, to better align input signals with the denoising diffusion process (Gao et al., 2025; Fei et al., 2025; Wang et al., 2025a). However, this alignment is typically performed per modality, relying on local cues such as acoustic features or pixel structures, followed by shallow fusion at the generation stage. While effective at reproducing observable details, this paradigm lacks coordination across multimodal inputs, leading to semantic conflicts or impoverished camera language. For instance, when the input contains angry speech but the text imposes no such constraint, the emotion may be significantly weakened in the final output.

To enable the model to truly understand the intent behind the instructions, drawing inspiration from multimodal large language models (MLLMs) (Bai et al., 2025; Hong et al., 2025; Qi et al., 2025), we unify

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

. . .

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

| | | |
|---|---|---|

latents

[Figure 69]

[Figure 70]

MLLM Director

###### >>

###### >>

>>>

###### Theroom,catwithis sittinga guitarinbesidea brightit... Human Video DiT

>>>

[Figure 71]

###### Blueprint key frames

Self Attention

Image

>>

>>

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

Temporal Attention

Text

###### . . .

>>>

Deep Thinking

Audio

Text Cross Attention

T5

[Figure 87]

[Figure 88]

###### Human Video DiT

Parallel generation

Audio Cross Attention

Storyline

Whisper

>>>

>>>

>>

>>

>>

Text

[Figure 89]

FFN

Audio

The cat waves hands and reaches for guitar… Then he picks up the guitar horizontally… Lastly he strums his guitar while singing happily.…

× N

- Figure 3: Illustration of Kling-Avatar’s cascaded generation pipeline. An MLLM Director first interprets multimodal instructions into high-level semantics and tells a storyline. Guided by this global planning, the first stage generates a blueprint video. In the second stage, keyframes are extracted from the blueprint and used as first–last frame conditions for parallel sub-clip generation, refining local details and dynamics to synthesize long-duration videos.

evidence from multimodal inputs into a shared semantic space, producing high-level control signals as a global planning for the generation process. Specifically, we use Qwen2.5-Omni (Xu et al., 2025) to extract the transcription and emotion from audio as the audio caption, and Qwen2.5-VL (Bai et al., 2025) to generate descriptions from images as the image caption. These captions are then combined with the user prompt and processed by our MLLM Director, to output a coherent storyline. We explicitly specify the storyline template for the MLLM Director using a three-shot in-context learning manner. This storyline, prioritized by user knowledge, audio, and image references, tells key elements such as character features, background layout, actions, visual style, camera planning, and emotional shifts. All of these elements are organized into a unified textual prompt, which is injected into the video diffusion model through a text cross-attention layer to generate a blueprint video.

#### 2.2 Cascaded Generation for Long-Duration Generation

In the first stage, we generate a blueprint video that tells a storyline to reflect semantic user intent. The blueprint is then leveraged in the second stage to produce video sub-clips that refine local dynamics and visual details. To this end, we evenly segment the video according to the desired number of clips. Around each segmentation point, we then select a high-quality frame that preserves identity consistency, exhibits significant motion, avoids occlusion, and conveys expressive facial details. These frames serve as anchor keyframes for first-last-frame conditioned generation of adjacent sub-clips. During sub-clip synthesis, the MLLM Director decomposes the global storyline into temporally localized semantic plans. This localized narrative, combined with time-aligned audio conditioning, provide fine-grained guidance to ensure expressive coherence and visual consistency throughout the generated sequence. To avoid misalignment between anchor frames and the actual speech timing, we employ an audio-conditioned interpolation strategy to synthesize transition frames. This ensures precise frame synchronization with the input audio, enabling seamless and temporally coherent transitions across sub-clips.

The pipeline can be easily implemented in a parallel manner since the clips are generated independently. By increasing the anchor number, arbitrarily long videos can be generated with nearly the same runtime as producing a single clip. This cascaded framework with first-last-frame conditioned generation highlights our unique advantage in generating long-duration videos and provides a promising solution for downstream applications such as digital human podcasting, public speaking, and online education.

#### 2.3 Data Preparation

Training Data. We collect thousands of hours of audio–visual content from multiple sources, including publicly available datasets as well as self-collected videos such as film clips, speeches, monologues, interviews, and singing performances, covering a wide range of scenarios, linguistic styles and character dynamics. All videos are carefully processed with audio extraction and captioning. In our practice, we emphasize that data quality, rather than data scale, plays a decisive role in final performance: a smaller amount of high-quality talking segments proves more effective than indiscriminately enlarging

the dataset with long-tail samples. To this end, we design a suite of expert models to classify and filter low-quality data along multiple dimensions:

- • Lip-clarity filtering. We synthetically perturb mouth regions in high-quality talking-head videos to construct positive/negative pairs. A binary discriminator is trained to classify lip-region clarity and filter out videos with visually ambiguous or motion-blurred lip movements.
- • Temporal-continuity detection. We manually assemble different video segments to build negative samples, paired with original clips as positives. We then train a temporal coherence discriminator, along with PySceneDetect (Breakthrough, 2023), to identify and remove discontinuous clips.
- • Audio–visual synchronization. We employ SyncNet (Chung &Zisserman, 2016) to assess framelevel audio-visual synchronization confidence scores, and discard videos that fall below calibrated thresholds.
- • Aesthetic quality assessment. We adopt video aesthetic scoring methods (Schuhmann, 2022) to evaluate visual composition and appeal. Only videos exceeding a calibrated quality threshold are included in the final training set.

After filtering the data using expert models, we further perform manual curation on the retained samples, ultimately assembling hundreds of hours of high-quality human portrait videos, which provide reliable supervision for training our model.

Benchmark. To comprehensively evaluate the performance of Kling-Avatar, we construct a challenging benchmark comprising 375 image–audio-prompt pairs. The dataset is carefully designed with the following composition:

- • Images. Reference images are sourced equally from real videos and AI-generated content. The set includes 340 human portraits of different races in both full-body and half-body formats, as well as 35 non-human cases covering cartoon, anime, and animal characters. Image resolutions span vertical, horizontal, and square formats, ranging from 480p to 1080p.
- • Audio. Audio tracks are extracted from real videos and cover both speeches and songs. The collection includes 150 Chinese, 150 English, 35 Korean, and 40 Japanese samples, with clip lengths ranging from 8 seconds to 2 minutes. The audios span multiple speaking rates and expressive styles, ensuring diversity in linguistic and prosodic conditions.
- • Prompt. Text prompts are manually annotated with diverse and explicit specifications on emotional expression, character actions, camera movements, and background layouts. Emotion categories include calm, excitement, confusion, sadness, surprise, and anger, each with multiple intensity levels. Camera instructions specify operations such as panning and zooming. Action descriptions encompass turning, raising hands, head shaking, and other expressive gestures, ensuring broad coverage of dynamic behaviors.

This benchmark establishes a demanding testbed for existing methods by requiring vivid and coherent portrait generation under complex multimodal instruction control.

#### 2.4 Training and Inference Strategy

Training Strategy. We design several training strategies to strengthen the alignment between lip movements and the corresponding speech. First, we adopt a sliding window scheme to inject audio features into the audio cross-attention layer, where each video token attends only to its temporally aligned audio tokens with a small padding, thereby reinforcing local phase consistency. Second, we employ DWPose (Yang et al., 2023) to locate the mouth region and assign a higher weight to its diffusion denoising loss. Third, we randomly pad empty pixels around video frames during training to reduce the proportion of the face in the image, which encourages the model to remain robust under small-face and long-shot conditions. Finally, to preserve the text controllability of the base video generation model and concentrate on audio–visual interaction, we freeze the parameters of the text cross-attention layer during training, effectively preventing the base model from collapsing into overfitting the specific talking-head data. Collectively, these strategies substantially improve lip synchronization accuracy by enhancing visual-audio alignment.

Inference Strategy. Our first-last-frame conditioned parallel generation framework alleviates the identity drift problem that commonly arises in existing methods which rely on motion frames for long video

Table 1: Numerical evaluations on GSB metrics between our method and competitors.

Category GSB Overall Lip Sync Visual Quality Control Response ID Consistency Overall

Ours vs. OmniHuman 2.39 1.77 2.06 1.17 1.37 Ours vs. HeyGen 1.37 2.35 1.76 0.76 0.86

Ours vs. OmniHuman 1.41 1.00 2.18 1.06 1.27

Speech-En

- Ours vs. HeyGen 0.79 1.22 1.51 0.83 0.76

Speech-Ch

Ours vs. OmniHuman 4.53 3.90 2.44 1.13 1.47

- Ours vs. HeyGen 1.22 2.26 1.93 0.79 0.82

Sing-En/Ch

Ours vs. OmniHuman 2.69 2.03 1.72 1.35 1.38

- Ours vs. HeyGen 2.90 7.69 1.89 0.97 0.70

Ours vs. OmniHuman-1 Ours vs. HeyGen

Kling-Avatar preferred Same The other preferred

Kling-Avatar preferred Same The other preferred

###### Overall

65.5%

16.9%

17.6%

49.7%

19.3%

31.0%

46.3%

29.0%

- 23.7%

26.3%

14.7%

- 24.7%

63.3%

22.7%

14.0%

Lip Sync

47.7%

37.7%

52.0%

32.3%

15.7%

Visual Quality

26.0%

47.7%

9.3%

59.4%

31.3%

Control Response

ID Consistency

33.3%

43.0%

20.3%

48.0%

31.7%

0.0% 20.0% 40.0% 60.0% 80.0% 100.0%

0.0% 20.0% 40.0% 60.0% 80.0% 100.0%

- Figure 4: Overall GSB evaluation results on our benchmark across various dimensions against OmniHuman-1 and HeyGen.

continuation. To further improve identity consistency within each segment, we introduce a negative frame Classifier-Free Guidance (CFG) mechanism. Through statistical analysis, we find that identity drift artifacts typically manifest as texture distortions, blurring, exaggerated contrast and saturation, and color shifts. To counter this, we manually corrupt the reference image according to these observed patterns to simulate an enhanced identity drift. The degraded image is then used as a negative CFG signal to guide the denoising process toward identity-consistent directions. In addition, since no ground truth frames are available for mouth region masking during inference, we instead increase the audio cross-attention values to strengthen the lip-audio alignment.

### 3 Experiments

#### 3.1 Experimental Settings

Implementation Details. Our implementation is based on a Video Diffusion Transformer architecture which was pretrained on a large-scale dataset. We extend it with an audio cross-attention layer to support audio-to-video generation. Audio features are extracted via a pre-trained Whisper encoder (Radford et al., 2022), and text conditioning utilizes a T5 encoder (Raffel et al., 2020). The model is optimized using AdamW (Loshchilov &Hutter, 2017) with a learning rate of 1e-5. During training, our framework supports arbitrary video resolutions ranging from 480p to 1080p, and at inference it produces fluent videos with up to 1080p at 48 fps.

Evaluation Metrics. We design a human preference–based subjective evaluation protocol as our primary metric, aiming to better reflect user-perceived semantics and aesthetic quality. For each sample in the benchmark, three participants independently provide a Good/Same/Bad (GSB) judgment by comparing the results of our method against baseline methods, and the final GSB label is determined by majority vote. We report (G+S)/(B+S) as the main metric, reflecting the proportion of cases where our method is judged as "better or not worse" than the baseline. In addition to the overall evaluation, we also conduct

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

OursOmniHumanHeyGen

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

/ɒ/ /ɑ:/ /ɪ/ /tru:/

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

OursOmniHumanHeyGen

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

/ə/ /hʊ/ ∅ /ʃ/

- Figure 5: Comparison of lip synchronization between Kling-Avatar and baselines. We produce accurate lip movements for characters across different scenarios.

GSB assessments on four specific dimensions, including:

- • Lip Synchronization. Assesses the naturalness of lip movements, accuracy of audio–visual alignment, and plausibility of facial expressions.
- • Visual Quality. Evaluates overall aesthetic appeal, structural coherence, and visual clarity of the generated video.
- • Control Response. Examines whether emotions, actions, and camera movements in the generated video accurately reflect the textual instructions. Since OmniHuman-1 does not support prompt input, this metric is instead used to evaluate how effectively audio conditions control the body movements.
- • Identity Consistency. Measures how well the generated video preserves identity traits and dynamic characteristics that are consistent with the reference image.

This GSB protocol provides a unified and intuitive framework of evaluating key aspects such as multimodal instruction following, avatar expressiveness, and visual coherence, which better reflects user subjective experience in real-world scenarios. We plan to incorporate additional objective metrics in the future to complement and extend our assessment.

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

##### Figure 6: Our generated videos with multimodal instruction conditioning. We highlight our results in generating vivid and coherent portrait animations with strong control over emotions, camera movements, lip synchronization and motion dynamics.

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

[Figure 211]

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

[Figure 229]

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

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

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

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

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

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

0s 10s 20s 30s 40s 50s 60s

- Figure 7: Visualization of generated long-duration videos with high consistency, coherence and vividness.

Baselines. We select OmniHuman-1 (Lin et al., 2025) and HeyGen (hey) as our primary baselines, since they represent the most competitive state-of-the-art systems currently available on the market. In the future, we plan to extend our comparisons to other commercial solutions such as Higgsfield (hig) and Hedra (hed).

#### 3.2 Experimental Results

Comparison with Baselines. Tab. 1 summarizes the GSB evaluation results on the benchmark, comparing our method with OmniHuman-1 (Lin et al., 2025) and HeyGen (hey). In addition to the overall benchmark score, we report results for three sub-categories: English speeches (Speech-En), Chinese speeches (SpeechCh), and bilingual singing (Sing-En/Ch). Since the number of Japanese and Korean samples is relatively small, making their GSB statistics less reliable, we include them only in the overall scores. Fig. 4 further visualizes the GSB comparison on the full benchmark. Numerical results show that Kling-Avatar consistently outperforms OmniHuman-1 across all dimensions, highlighting our superior performance.

Compared with HeyGen, our method achieves notable improvements in Lip Synchronization and Visual Quality. Notably, HeyGen produces videos by repeatedly looping a five-second action pattern, which enhances motion stability and identity consistency but significantly harms vividness and diversity. In addition, HeyGen crops the reference image to fixed horizontal or vertical resolutions for generation, while our method supports arbitrary input and output resolutions, producing videos up to 1080p at 48 fps. Moreover, HeyGen is tailored for digital human scenarios, whereas our approach is developed on top of a general video generation foundation model, making it more extensible and adaptable to broader future applications. We further provide visual comparison of lip synchronization accuracy in Fig. 5. Our method demonstrates precise correspondence between lip shapes and various syllables, whereas the baseline methods struggle with accurate alignment and sometimes even fail to respond.

Results on Diverse Scenarios. Fig. 6 showcases our diverse generation results. Benefiting from the high-level planning produced by the MLLM Director through the understanding and integration of multimodal instruction intents, our method faithfully adheres to the input signals and delivers vivid character emotions, actions, camera movements, and accurate, fine-grained lip synchronization. Moreover, it demonstrates strong generalization to various open scenarios, including multi-persons, cartoon and anime styles, and even non-human characters. Please refer to our project page for more compelling video results.

Long-Duration Video Synthesis. We further demonstrate the advantages of our cascaded parallel generation framework in long-duration video synthesis. As shown in Fig. 7, we sample one frame every 10 seconds to visualize the results. The generated frames exhibit stable identity preservation, coherent visual quality, and rich character dynamics. Notable examples include the background lighting changes in the first line, the head movements in the second, and the hand gestures in the third.

### 4 Related Work

- 4.1 Video Generation

Breakthroughs in diffusion models for image synthesis (Ho et al., 2020; Dhariwal &Nichol, 2021; Rombach et al., 2022) have driven an evolution in video generation, where scalable training paradigms based on noise inversion and conditional denoising have made high-fidelity appearance and controllability attainable. Early video generation approaches typically extend pretrained image-based U-Nets by stacking temporal modules or inserting temporal attention into spatial backbones to capture crossframe relations (Ho et al., 2022; Singer et al., 2022; Blattmann et al., 2023). However, such designs face limitations in scalability in terms of resolution and sequence length. Recently, the growth of training data and computational resources has shifted the focus toward Diffusion Transformers (DiT) (Peebles &Xie, 2023; Yang et al., 2025; Wan et al., 2025). This paradigm compresses videos into spatiotemporal tokens via 3D VAEs, and leverages the large context capacity and scalable attention of transformers to capture temporal dynamics, thus supporting stable large-scale video generation and establishing itself as the emerging mainstream approach. It also demonstrates strong potential for long-term generation (Kong

- et al., 2024; Zhang &Agrawala, 2025; Teng et al., 2025; Chen et al., 2025b), real-time synthesis (Zhao et al., 2025; Gu et al., 2025), and world modeling (Yu et al., 2025; Team et al., 2025). These methods are primarily designed for general video generation yet remain inadequate for speech-driven digital portrait modeling.

4.2 Audio-Driven Digital Human Synthesis

Audio-driven digital human synthesis aims to generate realistic and expressive talking videos conditioned on speech signals and a reference portrait image. One line of work employs explicit intermediate representations, such as facial landmarks or 3D head models, to drive facial expressions and lip movements (Chen et al., 2025a; Hu et al., 2025; Guo et al., 2024; Cui et al., 2025c). However, these approaches are typically limited to facial animation and cannot produce natural upper-body motion or hand gestures. More recent studies leverage diffusion models for end-to-end audio-driven video generation. By directly injecting speech as a condition into diffusion transformers, these methods achieve joint alignment and control of audio, expressions, and motion within a unified attention framework (Jiang et al., 2025; Gan

- et al., 2025; Peng et al., 2025; Wang et al., 2025a; Guo et al., 2024), enabling realistic and coherent video synthesis without relying on 3D priors, and showing advantages in expression detail and lip–audio alignment. To further support hand motion and human–object interactions, methods such as Emo2 (Tian

- et al., 2025) and HunyuanVideo-HOMA (Huang et al., 2025) incorporate pose sequences as conditions alongside speech and body dynamics. Other approaches such as Mocha (Wei et al., 2025), MultiTalk (Kong et al., 2025) and InteractHuman (Wang et al., 2025b), learn identity information or memory-slot IDs to enable speaker switching and cross-shot localization. Additional efforts explore data scaling (Lin et al., 2025), audio–video alignment strategies (Gao et al., 2025), and direct performance optimization (Cui et al., 2025a). Despite these advances, existing methods still rely on local cues for alignment within each modality, and thus struggle with multimodal instruction understanding and consistent long-duration generation. To address these challenges, we explore the use of multimodal large language models for instruction grounding and propose a cascaded framework for fast synthesizing vivid, long-duration portrait animations.

### 5 Conclusion

In this paper, we introduce Kling-Avatar, a cascaded framework that unifies multimodal instruction understanding with long-duration generation of lifelike portrait videos. Our two-stage pipeline first employs an MLLM director to produce a blueprint video that encodes high-level semantic intent into a coherent storyline, and then synthesizes long videos through parallel sub-clip generation guided by blueprint keyframes to refine local dynamics. Coupled with carefully curated data and practical training and inference strategies, our framework preserves fine-grained details while faithfully realizing global semantics. To evaluate the effectiveness, we construct a 375-sample benchmark spanning diverse instructions and challenging scenarios. Experiments demonstrate that Kling-Avatar delivers vivid, fluent videos up to 1080p and 48 fps, with precise lip synchronization, strong controllability, and robust generalization to open scenarios. Human preference–based metric comparisons further confirm our superior performance. We believe our exploration of instruction-grounded, long-duration avatar video generation represents a promising step toward broad real-world applications and future research.

### References

Hedra AI. URL https://www.hedra-ai.com/. HeyGen. URL https://www.heygen.com/. Higgsfield AI. URL https://higgsfield.ai/avatars. Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie

Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Breakthrough. Pyscenedetect. 2023. URL https://github.com/Breakthrough/PySceneDetect. Hejia Chen, Haoxian Zhang, Shoulong Zhang, Xiaoqiang Liu, Sisi Zhuang, Pengfei Wan, Di ZHANG,

Shuai Li, et al. Cafe-talk: Generating 3d talking face animation with multimodal coarse-and fine-grained control. In The Thirteenth International Conference on Learning Representations, 2025a.

Ming Chen, Liyuan Cui, Wenyuan Zhang, Haoxian Zhang, Yan Zhou, Xiaohan Li, Xiaoqiang Liu, and Pengfei Wan. Midas: Multimodal interactive digital-human synthesis via real-time autoregressive video generation. arXiv preprint arXiv:2508.19320, 2025b.

Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, and Chenguang Ma. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 2403–2410, 2025c.

Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Asian conference on computer vision, pp. 251–263. Springer, 2016.

Jiahao Cui, Yan Chen, Mingwang Xu, Hanlin Shang, Yuxuan Chen, Yun Zhan, Zilong Dong, Yao Yao, Jingdong Wang, and Siyu Zhu. Hallo4: High-fidelity dynamic portrait animation via direct preference optimization and temporal motion modulation. arXiv preprint arXiv:2505.23525, 2025a.

Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21086–21095, 2025b.

Liyuan Cui, Xiaogang Xu, Wenqi Dong, Zesong Yang, Hujun Bao, and Zhaopeng Cui. Cfsynthesis: Controllable and free-view 3d human video synthesis. In Proceedings of the 2025 International Conference on Multimedia Retrieval, pp. 135–144, 2025c.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Zhengcong Fei, Hao Jiang, Di Qiu, Baoxuan Gu, Youqiang Zhang, Jiahua Wang, Jialin Bai, Debang Li, Mingyuan Fan, Guibin Chen, et al. Skyreels-audio: Omni audio-conditioned talking portraits in video diffusion transformers. arXiv preprint arXiv:2506.00830, 2025.

Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, and Steven Hoi. Omniavatar: Efficient audio-driven avatar video generation with adaptive body animation. arXiv preprint arXiv:2506.18866, 2025.

Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, et al. Wan-s2v: Audio-driven cinematic video generation. arXiv preprint arXiv:2508.18621, 2025.

Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pp. arXiv–2507, 2025.

Wentao Hu, Shunkai Li, Ziqiao Peng, Haoxian Zhang, Fan Shi, Xiaoqiang Liu, Pengfei Wan, Di Zhang, and Hui Tian. Ggtalker: Talking head systhesis with generalizable gaussian priors and identity-specific adaptation. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Ziyao Huang, Zixiang Zhou, Juan Cao, Yifeng Ma, Yi Chen, Zejing Rao, Zhiyong Xu, Hongmei Wang, Qin Lin, Yuan Zhou, et al. Hunyuanvideo-homa: Generic human-object interaction in multimodal driven human animation. arXiv preprint arXiv:2506.08797, 2025.

Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. arXiv preprint arXiv:2409.02634, 2024.

Jianwen Jiang, Weihong Zeng, Zerong Zheng, Jiaqi Yang, Chao Liang, Wang Liao, Han Liang, Yuan Zhang, and Mingyuan Gao. Omnihuman-1.5: Instilling an active mind in avatars via cognitive simulation. arXiv preprint arXiv:2508.19209, 2025.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Zhe Kong, Feng Gao, Yong Zhang, Zhuoliang Kang, Xiaoming Wei, Xunliang Cai, Guanying Chen, and Wenhan Luo. Let them talk: Audio-driven multi-person conversational video generation. arXiv preprint arXiv:2505.22647, 2025.

Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Ziqiao Peng, Jiwen Liu, Haoxian Zhang, Xiaoqiang Liu, Songlin Tang, Pengfei Wan, Di Zhang, Hongyan Liu, and Jun He. Omnisync: Towards universal lip synchronization via diffusion transformers. arXiv preprint arXiv:2505.21448, 2025.

Ji Qi, Yuan Yao, Yushi Bai, Bin Xu, Juanzi Li, Zhiyuan Liu, and Tat-Seng Chua. An lmm for efficient video understanding via reinforced compression of video cubes. arXiv preprint arXiv:2504.15270, 2025.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. URL https://arxiv.org/abs/2212.04356.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Christoph Schuhmann. Aesthetic score predictor. 2022. URL https://github.com/christophschuhmann/ improved-aesthetic-predictor.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, 2022.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, et al. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025.

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pp. 244–260. Springer, 2024.

Linrui Tian, Siqi Hu, Qi Wang, Bang Zhang, and Liefeng Bo. Emo2: End-effector guided audio-driven avatar video generation. arXiv preprint arXiv:2501.10687, 2025.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Mengchao Wang, Qiang Wang, Fan Jiang, Yaqi Fan, Yunpeng Zhang, Yonggang Qi, Kun Zhao, and Mu Xu. Fantasytalking: Realistic talking portrait generation via coherent motion synthesis. arXiv preprint arXiv:2504.04842, 2025a.

Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, and Dahua Lin. Interacthuman: Multi-concept human animation with layout-aligned audio conditions. arXiv preprint arXiv:2506.09984, 2025b.

Cong Wei, Bo Sun, Haoyu Ma, Ji Hou, Felix Juefei-Xu, Zecheng He, Xiaoliang Dai, Luxin Zhang, Kunpeng Li, Tingbo Hou, et al. Mocha: Towards movie-grade talking character synthesis. arXiv preprint arXiv:2503.23307, 2025.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4210–4220, 2023.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, 2025.

Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.

Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. In The Thirteenth International Conference on Learning Representations, 2025.

