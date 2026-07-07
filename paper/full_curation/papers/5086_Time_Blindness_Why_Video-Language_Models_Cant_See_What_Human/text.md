## Time Blindness: Why Video-Language Models Can’t See What Humans Can?

Ujjwal Upadhyay1,3,*, Mukul Ranjan2,∗, Zhiqiang Shen2,†, Mohamed Elhoseiny1,† 1KAUST 2VILA Lab, MBZUAI 3DocPanel Technologies

Project page: https://timeblindness.github.io/

# arXiv:2505.24867v2[cs.CV]28Apr2026

### Abstract

Recent advances in vision-language models (VLMs) have made impressive strides in understanding spatio-temporal relationships in videos. However, when spatial information is obscured, these models struggle to capture purely temporal patterns. We introduce SpookyBench, a benchmark where information is encoded solely in temporal sequences of noise-like frames, mirroring natural phenomena from biological signaling to covert communication. Interestingly, while humans can recognize shapes, text, and patterns in these sequences with over 98% accuracy, state-of-the-art VLMs achieve 0% accuracy. This highlights a critical limitation: an over-reliance on frame-level spatial features and an inability to extract meaning from temporal cues. Overcoming this limitation will require novel architectures or training paradigms that decouple spatial dependencies from temporal processing. Our systematic analysis shows that this issue persists across model scales and architectures. We release SpookyBench to catalyze research in temporal pattern recognition and bridge the gap between human and machine video understanding. Dataset and code are available on the project website.

### 1. Introduction

Large multimodal models have revolutionized visual understanding in both images [5, 20, 26, 28, 51, 80] and videos [3, 54, 84, 85, 99]. Recent Video-Vision Language Models (Video-VLMs) demonstrate impressive capabilities across action recognition [38, 89, 100], visual question answering [4, 58, 61, 96, 102], dense captioning [13, 17, 18, 39, 63, 91, 93], and temporal grounding [15, 79, 92]. Yet even in tasks labeled as temporal, strong per-frame spatial cues often allow models to succeed without genuinely reasoning over time [10, 25, 46]. This paper introduces SpookyBench, a benchmark that closes this loophole. Every stimulus is constructed so that individual frames appear as noise; the signal exists only in the

*Equal contribution; Author ordering determined by a coin toss. †Corresponding authors

temporal relationship between frames. Existing temporal benchmarks [10, 44, 47, 95] still entangle spatial and temporal cues, making it difficult to diagnose where a model’s understanding breaks down. SpookyBench eliminates this confound entirely. In spirit it parallels ARC-AGI [23], which uses synthetic puzzles to isolate core reasoning; here, controlled stimuli isolate a single fundamental capability, extracting meaning from change over time.

Current approaches to video understanding [59, 75] typically follow a hierarchical paradigm: extract frame-level features using ViTs [6, 29, 65], integrate these features temporally, and fuse them with language for downstream tasks [45, 81, 88, 98]. This paradigm has yielded significant advances in general video understanding [31, 43, 59, 75]. However, our findings reveal a critical blind spot: when information exists purely in the temporal domain without reliable frame-level features, state-of-the-art models fail catastrophically (Figure 1). The inability to decode temporal patterns has significant implications for real-world applications. In nature, organisms such as fireflies communicate through precise temporal sequences of bioluminescence [11, 60, 66], encoding information exclusively through timing rather than spatial arrangements. Similarly, in medical domains, preictal EEG patterns for epileptic seizure prediction emerge only through temporal analysis [73]. These natural examples demonstrate how temporal patterns can carry rich information even when individual observations contain minimal static content. Similarly, various human technologies from Morse code to digital communication protocols rely on temporal encoding, yet current Video-VLMs lack the fundamental mechanisms to process such information.

The human visual system has evolved mechanisms for processing temporal information without relying solely on spatial cues [1, 37, 76]. Neuroscience research has revealed that temporal processing is distributed across neural structures rather than centralized in a single area [56], and the brain uses intrinsic network dynamics to perform temporal computations [62]. Areas such as the parietal cortex integrate temporal information along with spatial and numeric magnitudes [8]. Our experiments confirm humans’ remarkable temporal perception: participants achieve over 98% accuracy

Spatial Features

Strong Focus

Object & Classes Scene Layouts

[Figure 1]

Video Input

Frame Sampling

Language Model

Visual Encoder

[Figure 2]

Time t Temporal Information Loss

Spatial Bias

[Figure 3]

Coherence Gap

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Temporal Features

[Figure 9]

[Figure 10]

Weak Focus Motion patterns Event Causality

[Figure 11]

[Figure 12]

- Figure 1. Illustration of the current video-language models’ limitations: over reliance on spatial visual features within individual frame. Frame sampling results in temporal information loss, while the visual encoder exhibits a strong spatial bias. This creates a coherence gap (×) between well-represented spatial features (objects, scene layouts) and under-represented temporal features (motion patterns, causality), limiting video understanding capabilities.

on SpookyBench tasks without training. In stark contrast, our evaluation of 15 state-of-the-art Video-VLMs, including closed-source commercial systems such as GPT-4o [36], and Gemini 2.0 Flash [27], reveals near-zero accuracy on these same tasks.

niques [70], and multimodal fusion [81, 88, 98]. All of these exhibit limited temporal reasoning, with hallucinations [44], grounding failures [79], and reliance on linguistic shortcuts [40] persisting across action recognition [38, 89, 100], question answering [4, 58, 96], and captioning [13, 39, 93]. Targeted interventions such as timestamp-aware encoding [68], segment-level reasoning [64], direct token processing [53], temporal separation tokens [15], specialized temporal streams [79, 83], and synthetic training paradigms [59, 75, 96, 99] have not closed this gap, and dedicated video architectures like VideoGPT+ [55], TimeChat [68], LinVT [35], LongVLM [85], and Baichuan-Omni [49] extract spatial features first and treat temporal integration as an afterthought.

This performance gap persists across model architectures, parameter scales, and pre-training strategies, from compact systems like VideoLLaMA3-2B [97] to large-scale ones such as GPT-4o [36] and Qwen-VL [80], and extends to models designed specifically for video understanding including LongVLM [85], LLaVA-NeXT-Interleave [45], and InternVideo2.5 [84]. Efforts to enhance temporal reasoning through specialized temporal modeling [64, 68, 83] and finegrained temporal localization [15, 53, 79] have not addressed the challenge of extracting meaning from purely temporal patterns without reliable spatial features.

These limitations are further corroborated by temporal understanding benchmarks; TemporalBench [10] reveals a significant gap between model and human performance, TVBench [25], VITATECS [47], and Fateh et al. [34] confirm that many datasets reward spatial analysis over temporal reasoning, and targeted evaluations surface specific failures in temporal hallucinations [44], streaming video reasoning [95], and temporal location, object tracking, and anomaly detection [48]. Across these analyses, models such as LLaVA-Video [99], Video-ChatGPT [54], TemporalVLM [34], and VidChain [42] exploit spatial shortcuts instead of temporal reasoning [15, 40, 44, 79]. SpookyBench addresses this by obscuring spatial information and forcing models to extract meaning from temporal dynamics alone, exposing a “time-blindness” that conventional benchmarks leave undetected.

Our findings suggest that achieving human-like video understanding requires rethinking how neural architectures process temporal information. Rather than treating temporal integration as secondary to spatial feature extraction, future models may need dedicated mechanisms for temporal pattern recognition, drawing inspiration from cognitive neuroscience research on distributed neural timing mechanisms [56, 62] and specialized brain regions for temporal processing [8, 57]. The gap between human and machine performance on SpookyBench indicates that current architectures remain “time-blind” despite strong performance on standard benchmarks. We release SpookyBench to catalyze research into temporal reasoning in Video-VLMs, with applications ranging from medical diagnostics to autonomous systems that must interpret temporal cues in complex environments.

##### 2.2. Neuroscience Insights on Temporal Processing

Human perception of temporally-defined objects relies on the Gestalt principle of common fate [86], whereby coherently moving elements are grouped together, enabling figureground segregation from motion alone [37]. Tangemann et al. [76] showed that a cortical motion energy model [71] matches human performance on random dot segmentation while 40 deep optical flow models fail, and relatedly, recent works [9, 12] encode motion into structured noise for video diffusion control. Beyond these perceptual grouping mecha-

### 2. Related Work

##### 2.1. Temporal Reasoning in Video-VLMs

Video-VLMs have advanced through several architectural families, including LLaVA variants [43, 45, 51, 52, 54, 99], the Qwen series [5, 80], InternVL models [20, 21, 84], dual encoders [55], interleaved tokens [3, 103], compression tech-

Model Direct Prompt CoT Params Human Performance 98.0% ± 0.6 N/A N/A

Open-Source Models VideoLLaMA3-7B [97] 0% ± 0.0 0% ± 0.0 7B VideoLLaMA3-2B [97] 0% ± 0.0 0% ± 0.0 2B TimeChat-7B [68] 0% ± 0.0 0% ± 0.0 7B MiniGPT4-Video [3] 0% ± 0.0 0% ± 0.0 7B MovieChat [74] 0% ± 0.0 0% ± 0.0 7B Video-ChatGPT-7B [54] 0% ± 0.0 0% ± 0.0 7B VideoGPT-plus-Phi3-mini-4k [55] 0% ± 0.0 0% ± 0.0 7B VILA1.5-13b[50] 0% ± 0.0 0% ± 0.0 13B ShareGPT4Video-8B [13] 0% ± 0.0 0% ± 0.0 8B VideoLLaMA2-7B [22] 0% ± 0.0 0% ± 0.0 7B Video-LLaVA [99] 0% ± 0.0 0% ± 0.0 7B LLaVA-NeXT-Video [45] 0% ± 0.0 0% ± 0.0 8B InternVL2-40B [20] 0% ± 0.0 0% ± 0.0 40B InternVL2-8B [20] 0% ± 0.0 0% ± 0.0 8B InternVL2.5-78B [19] 0% ± 0.0 0% ± 0.0 78B InternVL2.5-8B [19] 0% ± 0.0 0% ± 0.0 8B InternVideo2.5-Chat-8B [84] 0% ± 0.0 0% ± 0.0 8B InternVideo2-Chat-8B [82] 0% ± 0.0 0% ± 0.0 8B Qwen2-VL-2B-Instruct [80] 0% ± 0.0 0% ± 0.0 2B Qwen2-VL-7B-Instruct [80] 0% ± 0.0 0% ± 0.0 7B

- Qwen2-VL-72B-Instruct [80] 0% ± 0.0 0% ± 0.0 72B Qwen2.5-VL-3B-Instruct [5] 0% ± 0.0 0% ± 0.0 3B Qwen2.5-VL-7B-Instruct [5] 0% ± 0.0 0% ± 0.0 7B Qwen2.5-VL-72B-Instruct [5] 0% ± 0.0 0% ± 0.0 72B
- Qwen3-VL-8B-Instruct [78] 0% ± 0.0 0% ± 0.0 8B Closed-Source Models

Gemini 2.5 Pro [24] 0% ± 0.0 0% ± 0.0 N/A

- Gemini 1.5 Pro [77] 0% ± 0.0 0% ± 0.0 N/A
- Gemini 2.0 Flash[27] 0% ± 0.0 0% ± 0.0 N/A GPT-4o [36] 0% ± 0.0 0% ± 0.0 N/A

- Table 1. Benchmark results comparing model performance on SpookyBench across different prompting strategies along with model size. Human accuracy (98.0%) is the weighted average of accuracy across 3 different categories.

nisms, neuroscience research offers insights for addressing temporal limitations in Video-VLMs. Mauk and Buonomano [56] established that temporal processing is distributed across neural structures through intrinsic circuit properties, contrasting with current Video-VLMs’ sequential spatial processing. The human brain processes time at multiple granularities, with the cerebellum handling millisecond-tosecond timing [57], the parietal cortex integrating temporal, spatial and numerical magnitudes [8], and neural patterns encoding time through “population clocks” [62]. Distributed temporal representations that evolve over time [62, 87] offer an alternative to treating temporal integration as secondary. The performance gap on temporal tasks [10, 25, 44] and our SpookyBench findings demonstrate that current architectures lack mechanisms for processing purely temporal patterns, a natural capability in humans through neural systems representing time as intrinsic dynamics.

- 3. SpookyBench

We introduce SpookyBench, a novel synthetic dataset specifically designed to isolate and evaluate pure temporal understanding in video language models. The key innovation of our benchmark lies in its unique design: All mean-

ingful information is encoded exclusively in the temporal domain through dynamic patterns of texts, images and video depth maps, while individual frames contain only structured noise. Our dataset is fundamentally different from the existing datasets used for training, fine-tuning, and evaluation of video-VLMs. Many state-of-the-art video language models employ advanced techniques, such as dynamic resolution strategies [5, 20, 80], specialized temporal encoding methods [5, 68, 80], hierarchical token merging [84, 85], and joint video-motion training frameworks [14] to capture temporal dynamics. However, these methods still rely on spatial representations extracted from individual frames, which currently remain the only viable mechanism for inferring temporal information. In contrast, SpookyBench forces models to depend only on temporal cues, thereby creating the first benchmark that exclusively evaluates a model’s ability to process and understand pure temporal information.

##### 3.1. Dataset Generation

Figure 2 shows our proposed data generation framework. The dataset consists of specially designed videos that encode three types of content - words, images, and videos - using binary noise patterns with specific motion properties. In this approach, content is embedded within noise patterns such that individual frames appear as random noise, while the content becomes perceptible only when viewed as a temporal sequence. Our dataset encodes different types of content (Figure 3) through temporal noise animations in the following categories: 1) Words: Text rendered as masks in which the background noise and foreground noise move in opposite directions, making the text visible only through temporal movement. 2)Images: Binary masks generated using SAM2 [67] from single-object images generated using textto-image model Flux [41], encoded using the same content mask animation approach as words. 3) Dynamic Scenes: Depth maps extracted from videos in single-object tracking datasets LaSOT [32] and OTB2015 [90] using Video Depth Anything [16]. These are encoded using a technique in which pixels above a brightness threshold move while others remain static as shown in the algorithm 2.

##### 3.2. Temporal Encoding Framework

Our temporal encoding framework implements two distinct motion configurations as detailed in Algorithms 1 and 2. For words, and image masks (Algorithm 1), we employ opposing motion patterns between foreground and background. The content is first converted to a binary mask M where M(x,y) = 1 represents foreground pixels and M(x,y) = 0 represents background. We generate two separate noise patterns Nbg and Nfg consisting of random binary values (0 or 255). During animation, foreground pixels sample from Nfg with a positive offset that increases with time (y+vt mod h), while background pixels sample from Nbg with a negative

###### Animation States: Moving vs. Paused

###### Key Principle: Opposing Noise Movement

###### Moving (Content Visible)

Paused (Content Disappears)

###### Foreground Noise

###### Background Noise

Content Mask

(Moves Up/Left)

(Moves Down/Right)

(Text, Object, Shape)

Background Foreground

Foreground

Background

| |
|---|

| |
|---|

Content Mask Background Noise Foreground Noise

Down/Right Up/Left

| |
|---|
| |

Background Content

Brain groups pixels by motion direction Content becomes visible

[Figure 13]

[Figure 14]

[Figure 15]

No motion = No content

###### Brain Perceives Content Through Motion Brain groups motion → Content becomes visible

- Figure 2. Illustration of the temporal encoding framework used in SpookyBench. Left: Core mechanism showing how content becomes visible through opposing motion patterns. A content mask defines regions where foreground noise (moving up/left) and background noise (moving down/right) are applied. When animated, the human visual system groups pixels with similar motion, causing the content to emerge. Right: Comparison between moving and paused states, demonstrating how content is only perceptible during animation and disappears when static, as individual frames contain only structured noise.

Category Basic SNR (dB) Perceptual SNR Temporal Coherence Motion Contrast Images -46.95 ± 2.40 -47.28 ± 2.28 8.00 ± 2.08 7.17 ± 5.00 Dynamic Scenes -48.95 ± 3.64 -63.43 ± 5.74 21.91 ± 5.76 -3.18 ± 10.17 Text -39.27 ± 1.58 -49.18 ± 3.31 7.84 ± 0.65 8.26 ± 6.44

- Table 2. Signal-to-Noise Ratio (SNR) metrics across SpookyBench categories.

offset (y − vt mod h). This creates the perception of opposing motion within and outside the masked regions. For video depth maps (Algorithm 2), we employ a threshold-based approach. Using depth maps D extracted from videos, pixels with brightness values between lower and upper thresholds (tl ≤ d ≤ tu) are animated by sampling a noise pattern N with a time-varying offset (y + vt mod h), while pixels outside this range remain static. This creates the illusion that brighter regions (typically foreground objects) are moving while darker regions (typically background) remain static. The noise patterns are generated using binary values (0 or 255) in square blocks of variable size. We used different speckle sizes ranging from 1 × 1 to 3 × 3 pixels to investigate the effect of noise granularity on perception. For each speckle size, we also varied the noise density - the probability that a block is white versus black - using values of 10%, 30%, 50%, and 90%. These noise patterns arranged in pixel blocks create optimal perceptual conditions for human viewers while remaining challenging for vision language models. To ensure seamless animation, the noise patterns are made tileable by copying edge pixels to the opposite boundaries. All videos maintain consistent technical specifications: 960 × 540 pixel resolution, with an average duration of 7.11 seconds (ranging from 1.0 to 35.0 seconds) and an average

of 333.5 frames per video. Text videos have a consistent duration of around 4 seconds; however, videos of dynamic scenes are longer, ranging up to 35 seconds. Figure 2 illustrates the structure of the data set and the encoding patterns in categories. We used binary masks for the images using SAM2 [67]. For videos, depth maps are extracted using Depth Anything V2 [94] and Video Depth Anything [16] from the LaSOT [32] and OTB2015 [90] datasets.

##### 3.3. Data Statistics

SpookyBench comprises 451 videos in three distinct categories, each requiring purely temporal reasoning for content identification. The dataset is distributed as follows: Text (46.6%, 210 videos), Object Images (40.8%, 184 videos) and Dynamic Scenes (12.6%, 57 videos). Although Dynamic Scenes comprise only 12.6% of videos, they account for 42.5% of total frames due to their longer duration, balancing the frame-level distribution across categories. This distribution ensures comprehensive coverage of different temporal perception challenges while maintaining a natural frequency distribution that reflects real-world scenarios. Additionally, more dataset can be generated indefinitely through the data generator on our project page, thus the dataset size is essentially unlimited. The “Text” category contains common English words rendered through temporal noise patterns, enabling evaluation of models’ ability to identify linguistic content through purely temporal cues. The “Object Images” category presents single objects extracted from high-quality images using segmentation techniques [67], encoded with the same temporal animation approach. It also contains a synthetic silhouette of simple objects generated using DALLE 3 [7] and flux [41].

|[Figure 16]<br><br>|[Figure 17]|
|---|
<br><br>[Figure 18]<br><br>|[Figure 19]|
|---|
<br><br>|[Figure 20]|
|---|
<br><br>[Figure 21]<br><br>| |
|---|
<br><br>[Figure 22]<br><br>| |
|---|
<br><br>|[Figure 23]|
|---|
<br><br>|[Figure 24]|
|---|
<br><br>|[Figure 25]|
|---|
<br><br>|[Figure 26]|
|---|
<br><br>|[Figure 27]|
|---|
|
|---|

- Figure 3. Noise generation process: (Top) masks applied for dynamic noise video generation, (Mid) word-specific mask, and (Bottom) depth map of video frame used for constructing noise-overlaid stimulus.

###### 3.3.1. Analysis of Temporal Metrics

To ensure a rigorous quantification of the temporal information present in each video, we analyzed five key 2. SNR metrics that capture different aspects of the complexity and perceptibility of temporal patterns in SpookyBench, as shown in Table. These metrics provide insight into why temporal patterns might be visible to humans but challenging to detect by computational models.

Basic SNR measures signal-to-noise ratio in decibels:

SNRB = 10log10

PS PN

(1)

where PS = E[∥∇F∥2] is motion boundary energy derived from spatial gradients of optical flow field F(x,y) = (Fx,Fy), and PN = Var(I0) is variance of the static frame I0.

Perceptual SNR incorporates frequency-dependent visual sensitivity:

SNRP = 10log10 ∥H(B) ⊙ W∥2

(2)

∥H(N) ⊙ W∥2

- where B is the average motion boundary strength, N is the static noise frame, H is the 2D Fourier transform, ⊙ denotes

element-wise multiplication, and W(f) = f · e−f/f

0 is the contrast sensitivity weighting function with peak f0 ≈ 0.1 cycles/pixel. Temporal Coherence SNR quantifies motion consistency:

SNRT = 10log10

Var(C) E[Varlocal(C)]

(3)

- where C = e−Var

θ(F) · (∥F∥ > τ) is the directional coherence map, Varθ computes circular variance of flow direction

angles over time, is indicator function, τ is magnitude threshold, and Varlocal computes variance over small spatial neighborhoods.

Motion Contrast SNR measures foreground-background motion differentiation:

SNRM = 10log10 ∥µM − µB∥2

- 1

- 2(σM2 + σB2 )

(4)

where µM = E[F | M] and µB = E[F | ¬M] are mean flow vectors within mask region M and background region

¬M respectively, σM2 = E[∥F − µM∥2 | M] and σB2 = E[∥F − µB∥2 | ¬M] are corresponding motion variances. The mask M is estimated from the motion boundaries.

The distribution of these metrics reveals why current vision models struggle with SpookyBench: they lack mechanisms to leverage temporal coherence (particularly high in Dynamic Scenes, 21.91 ± 5.76 dB) and motion contrast (negative for Dynamic Scenes, -2.20 and -3.18 dB), while text stimuli benefit from higher basic SNR (-39.27 ± 1.58 dB), explaining the observed performance gap.

###### 3.3.2. Binary SNR Threshold Effect in Detection

Our analysis shows a binary threshold phenomenon in detecting text within dynamic noise videos. The words shows negligible detection (∼0%) below 2.5dB SNR, but jumped to 85.7% accuracy above this threshold, displaying an abrupt rather than gradual transition as show in figure 5. Similar threshold behavior appears for images (inflection at 6.0dB) and dynamic scenes (9.0dB). Prompts performed best (40% accuracy), with Chain-of-Thought reasoning improving general identification tasks compared to direct prompting. This phenomenon parallels medical imaging diagnostics, where pathologies like microcalcifications in mammography become either entirely visible or invisible based on specific SNR thresholds [69]. The implications are significant: unlike perceptual phenomena that degrade gradually with noise, text detection functions as a step function, creating vulnerabilities in safety-critical applications. Just as radiologists cannot diagnose what remains invisible, language models cannot identify text below certain noise thresholds, leading to false certainties and potential catastrophic performance drops with minimal noise increases. This characteristic creates particular concerns for autonomous vehicles reading road signs or medical systems interpreting labels, while also exposing systems to adversarial attacks where slight SNR manipulations could render text completely undetectable.

### 4. Experiments 4.1. Experimental Setup

Models. We evaluate SpookyBench on both open source models (Video-LLaVA [99], LLaVA-NeXT-Video [45], TimeChat [68], InternVL2 [20], Qwen2-VL [80], Qwen2.5VL [5] etc.) and closed source models (GPT-4o [36], Gemini

Algorithm 1 Content Mask Animation

Algorithm 2 Video Depth Map Animation

- 1: Input: Content mask M, velocity v
- 2: Output: Animated frame Ft
- 3: Generate noise patterns Nbg, Nfg
- 4: for each pixel (x,y) do

▷ Check pixel’s mask status

- 5: if M(x,y) then
- 6: Ft(x,y) ← Nfg(x,y + vt mod h)

▷ Foreground

- 7: else
- 8: Ft(x,y) ← Nbg(x,y − vt mod h)

▷ Background

- 9: end if
- 10: end for

- 1: Input: Depth map D, thresholds (tl,tu), velocity v
- 2: Output: Animated frame Ft
- 3: Generate noise pattern N
- 4: for each pixel (x,y) do
- 5: d ← brightness from D(x,y)
- 6: if tl ≤ d ≤ tu then
- 7: Ft(x,y) ← N(x,y + vt mod h)

▷ Moving noise

- 8: else
- 9: Ft(x,y) ← N(x,y)

▷ Static noise

- 10: end if
- 11: end for

2.0 Flash [27], and Gemini 1.5 Pro [77]. We design different prompts for each category. All the prompts are included in the Appendix. All prompts instruct models to respond with only 1-5 words identifying the content. We input sequences of multiple video frames simultaneously for models that do not directly support video input.

Setup. We evaluate model performance using exact match accuracy. For Text, each video has a single correct label yi. For Object Images and Dynamic Scenes, we define a set of acceptable labels Yi = {yi1,yi2,...,yin} to account for semantic ambiguity (e.g., a video showing “a man playing basketball” accepts “playing basketball”, “man”, “human”, or “woman playing basketball”). Accuracy is computed as Accuracy = N1 Ni=1 (r i ∈ Li), where Li = yi for Text and Li = Yi for the other categories. We also verified all responses manually and with LLM-as-a-judge evaluation, a standard practice in VLM evaluation pipelines [30]; both yield identical 0% performance (details in Appendix). Despite this flexible protocol, no model produced responses matching any acceptable option.

##### 4.2. Human Evaluation

To evaluate human performance against our benchmark, we designed and conducted a controlled experiment involving human participants. We recruited a total of six human participants for this study, each independently evaluating all videos. Participants were instructed to view each video carefully and subsequently record their responses on an anonymized website in the following structured form: 1) Perceptibility Rating (1-5): Participants rated how perceptible the presented word, shape, or object was, ranging from 1 (very difficult to perceive) to 5 (very clearly perceptible). This measure provided insights into the clarity and ease of visual grouping. 2) Words/Objects Identification: Participants typed out exactly what they identified in the video. This response directly tested the accuracy of their visual perception.

|Text|Images|Dynamic Scenes|
|---|---|---|
|Acc(%) Perc(1-5)<br><br>|Acc(%) Perc(1-5)<br><br>|Acc(%) Perc(1-5)|

Annotator

- Annotator 1 99.5 4.7 99.5 4.7 96.5 4.3

- Annotator 2 98.6 4.8 98.4 4.9 91.2 4.0

- Annotator 3 99.5 4.9 97.2 4.5 94.7 4.4

- Annotator 4 97.6 4.6 96.7 4.5 91.2 4.0

- Annotator 5 100.0 4.8 99.5 4.7 99.0 4.7

- Annotator 6 98.0 4.7 97.8 4.5 93.0 4.2 Mean 98.9±0.7 4.8±0.0 98.2±1.1 4.7±0.1 94.3±3.1 4.3±0.1

Table 3. Human evaluation results showing accuracy and perceptibility ratings across different visual categories in SpookyBench.

We collect and evaluate participant responses using exact match criteria based on our predefined labels. Similar to the evaluation accuracy of the video language models for the categories of Object Images and Dynamic Scenes, we accepted multiple correct responses to avoid ambiguity. Table 3 shows the average precision and the perception rating of different annotators for different categories. The results show high human performance across all categories: participants correctly identified Words with 98% accuracy, while Object Images had 92% accuracy. We also observe a very high perceptibility rating (4.8 for texts and 4.3 and 4.0 for Object images and Dynamic scenes, respectively) across all three categories. This shows that the human brain can easily extract coherent information in videos, which seems to be very difficult for video language models.

##### 4.3. Impact of Frame Rates

To examine whether temporal sampling affects performance, we evaluate both humans and VLMs across frame rates from 1 to 30 FPS. Three human participants tested 120 randomly sampled videos (40 per category) at 1, 5, 10, 20, and 30 FPS, while four VLMs (Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen2.5-VL-3B, and GPT-4o) were evaluate using identical temporal downsampling. As shown in Tables 4 and 6, human accuracy remains above 95% at 20-30 FPS, degrades to 59.4% at 10 FPS, and drops to 0% at 1 FPS. In contrast,

Category 1 FPS 5 FPS 10 FPS 20 FPS 30 FPS Images 0.0 12.5 80.0 95.8 97.5 Words 0.0 10.8 35.8 95.8 95.8 Videos 0.0 15.0 62.5 93.3 93.3 Average 0.0 12.8 59.4 95.0 95.6

Table 4. Human accuracy (%) across different content categories at varying frame rates. Results are averaged across 3 participants on 120 videos (40 per category).

VJEPA-2 train

DINOv3 val

60%

VJEPA-2 val

Random chance (50%)

DINOv3 train

55%

Accuracy

50%

45%

40%

1.0

Random guess loss ( ln0.5 0.693)

Cross-entropyloss

0.9

0.8

0.7

5 10 15 20 25 30

Epoch

- Figure 4. Training and validation loss over all batch steps for a random binary classifier tasked with detecting foreground noise in video clips. The loss oscillates around 0.7, consistent with random prediction on a balanced dataset.

all VLMs achieved 0% accuracy across all frame rates. This demonstrates that temporal sampling frequency does not explain the performance gap between humans and current video-language models, indicating that VLMs lack the architectural mechanisms to process information conveyed through temporal patterns regardless of temporal resolution.

- 4.4. Impact of Finetuning To investigate whether the performance gap stems from out-of-distribution data rather than architectural limitations, we finetuned two state-of-the-art video-language models on SpookyBench: InternVL2.5-8B and Qwen2-VL-7B. Both models were trained on 400 SpookyBench videos for up to 30 epochs using LlamaFactory [101]. Despite this targeted training on the exact task and data distribution, both models maintained 0% accuracy on the test set. This result demonstrates that the failure to decode temporal patterns is not attributable to domain mismatch or insufficient exposure to the task, but rather indicates a fundamental architectural inability to process information conveyed purely through motion without relying on spatial content. We refer the reader

Model Words Images Videos Overall

Qwen2-VL-7B (Baseline) 0.0% 0.0% 0.0% 0.0% Qwen2-VL-7B (Augmented) 50.95% 70.51% 1.75% 51.54%

GPT-4o (Baseline) 0.0% 0.0% 0.0% 0.0% GPT-4o (Augmented) 56.19% 83.33% 3.51% 59.10%

Table 5. Performance comparison with and without motion boundary augmentation. Pre-computing and highlighting motion boundaries dramatically improves accuracy, demonstrating that temporal information is computationally extractable but inaccessible to current VLM architectures.

to the supplementary material for detailed results.

Can SOTA Visual Models Tell There’s an Object? We finetuned two video classification models, VJEPA-2[2] and DINOv3[72], on the task of predicting foreground noise presence. After 30 epochs, training loss saturated near chancelevel accuracy (50%) on the validation set (Figure 4), indicating that neither model could learn a discriminative representation from the frame-level features. The inability to overfit confirms that temporal patterns in this data are inaccessible to architectures operating on individual frames.

- 4.5. Impact of Motion Boundary Augmentation To further validate our hypothesis that VLM failure stems from an inability to extract temporal patterns rather than task impossibility or OOD, we pre-computed motion boundaries using classical optical flow (Farneback method [33]) and overlaid them onto the noisy frames before feeding them to models. This converts implicit temporal information into explicit spatial cues. As shown in Table 5, Qwen2-VL-7B jumps from 0% to 51.54% overall (50.95% on text, 70.51% on images), and GPT-4o reaches 59.10%. The substantial performance gain when temporal information is converted to spatial representations provides computational evidence that: (1) the benchmark tasks are solvable when appropriate preprocessing is applied, (2) current VLM architectures can process motion information if presented spatially, and (3) the fundamental limitation lies in the absence of temporal integration mechanisms that perform frame differencing and motion extraction, operations that classical computer vision methods accomplish routinely.

The Videos category shows minimal improvement (1.75%), consistent with our optical flow analysis in the supplementary material: complex articulated motion in dynamic scenes produces fragmented boundaries that remain hard to interpret even when spatially highlighted.

- 5. Results and Discussion

Table 1 presents the accuracy scores on the SpookyBench benchmark. Human participants achieved 98% accuracy under all test conditions. In contrast, all Video-VLMs scored 0% regardless of the type, size, or origin of the model. This pattern was held across all three task categories in our bench-

Model Qwen2-VL-7B Qwen2.5-VL-7B Qwen2.5-VL-3B GPT-4o Accuracy (%) 0.0 0.0 0.0 0.0

Table 6. VLM accuracy (%) averaged across all tested frame rates (1-30 FPS).

mark: temporal symbol recognition, temporal sequence understanding, and temporal pattern reasoning. We test two different prompting strategies to determine if performance limitations could be overcome through interface modifications. First, we used direct prompts with basic instructions asking the models to identify content in the videos. Next, we implemented chain-of-thought prompts with explicit guidance to focus on temporal patterns rather than individual frames. As shown in Table 1, none of these approaches yielded improvements. All models maintained 0% accuracy across all prompting conditions, suggesting that the limitation is inherent in the model architectures rather than a matter of optimization or prompt design. Examination of model output revealed consistent failure modes when processing SpookyBench videos.

Across all models tested, we observed attempts to extract information from individual frames rather than temporal patterns. When explicitly prompted to consider temporal changes, the models acknowledged the instruction but still failed to identify the patterns. For instance, Qwen-series models consistently predict “clock” or “coffee cup” regardless of input, indicating pattern collapse rather than meaningful temporal processing. Fine-tuned models produced outputs that mimicked training examples without correctly identifying test patterns. In particular, specialized temporal models like TimeChat [68], which were specifically designed for fine-grained temporal understanding, failed at the same rate as general-purpose models. This suggests that the limitation extends beyond general Video-VLMs to models explicitly optimized for temporal tasks.

##### 5.1. Architectural Implications for Vision Models.

Distinctive signal profiles in SpookyBench demonstrate a fundamental gap between human and machine perception of temporal information. Current vision models struggle with SpookyBench stimuli primarily because they: (1) lack robust temporal integration mechanisms that could leverage high temporal coherence, (2) process information primarily through spatial rather than temporal channels, and (3) fail to perform motion-based figure-ground segregation effectively. The consistently high temporal coherence values in Dynamic Scenes, coupled with their poor static-frame metrics, suggest that successful models must implement recurrent processing or attention mechanisms that operate across extended temporal windows rather than focusing on frame-level feature extraction. The negative motion contrast observed in Dynamic Scenes further indicates that models require more so-

|Text (threshold 2.5 dB) Image (threshold 6.0 dB)<br><br>| |
|---|
<br><br>Dynamic scene (threshold 9.0 dB)<br><br>|
|---|

1.0

| |
|---|

| |
|---|

| |
|---|

0.8

9.0 dB

Detectionaccuracy

| |
|---|

6.0 dB

0.6

2.5 dB

0.4

| |
|---|

0.2

0.0

| |
|---|

20 15 10 5 0 5 10 15

SNR (dB)

Figure 5. Analysis of effects of SNR on detecting text, image and dynamic scenes with direct prompting and chain of thought prompting.

phisticated motion segregation capabilities to match human perceptual abilities in dynamic visual environments. These findings highlight the need for architectural innovations that specifically address temporal processing limitations. Future models should incorporate dedicated temporal coherence pathways, motion contrast analysis, and longer temporal integration windows to bridge the perception gap demonstrated by SpookyBench.

### 6. Conclusion

In this paper, we introduced SpookyBench, a novel benchmark designed to evaluate the temporal reasoning capabilities of video-language models by isolating temporal understanding from spatial comprehension. Our experiments revealed a striking performance gap: while humans effortlessly achieve 98% accuracy on tasks requiring pure temporal pattern recognition, all tested models, including state-of-the-art open and closed-source systems, fail completely with 0% accuracy. This consistent failure across different model architectures, scales, and prompting strategies highlights a fundamental limitation in current video understanding approaches, which typically process spatial features first and then establish temporal connections, rather than integrating spatio-temporal information simultaneously. Importantly, our additional analyses with foreground noise detection and providing explicit motion cues to the model further validate that the deficit arises not from task impossibility but from architectural inability to internally extract temporal structure. The benchmark effectively exposes the time blindness of current architectures that remain hidden in conventional evaluation settings where spatial features can provide shortcuts to correct answers. We hope that SpookyBench will inspire the development of next-generation temporalconnected models.

### References

- [1] Valtteri Arstila. Theories of apparent motion. Phenomenology and the Cognitive Sciences, 15(3):337–358, 2016. 1
- [2] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. 7
- [3] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. arXiv preprint arXiv:2404.03413, 2024. 1, 2, 3
- [4] Hammad Ayyubi, Junzhang Liu, Ali Asgarov, Zaber Ibn Abdul Hakim, Najibul Haque Sarker, Zhecan Wang, ChiaWei Tang, Hani Alomari, Md Atabuzzaman, Xudong Lin, et al. Enter: Event based interpretable reasoning for videoqa. arXiv preprint arXiv:2501.14194, 2025. 1, 2
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 2, 3, 5
- [6] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, page 4, 2021. 1
- [7] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 4
- [8] Domenica Bueti and Vincent Walsh. The parietal cortex and the representation of time, space, number and other magnitudes. Philosophical Transactions of the Royal Society B: Biological Sciences, 364(1525):1831–1840, 2009. 1, 2, 3
- [9] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, et al. Go-with-the-flow: Motioncontrollable video diffusion models using real-time warped noise. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13–23, 2025. 2
- [10] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. Temporalbench: Benchmarking finegrained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024. 1, 2, 3
- [11] Albert D Carlson and Jonathan Copeland. Flash communication in fireflies. The Quarterly review of biology, 60(4): 415–436, 1985. 1
- [12] Pascal Chang, Jingwei Tang, Markus Gross, and Vinicius C Azevedo. How i warped your noise: a temporallycorrelated noise prior for diffusion models. arXiv preprint arXiv:2504.03072, 2025. 2

- [13] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 1, 2, 3
- [14] Ling-Hao Chen, Shunlin Lu, Ailing Zeng, Hao Zhang, Benyou Wang, Ruimao Zhang, and Lei Zhang. Motionllm: Understanding human behaviors from human motions and videos. arXiv preprint arXiv:2405.20340, 2024. 3
- [15] Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211, 2024. 1, 2
- [16] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. arXiv preprint arXiv:2501.12375, 2025. 3, 4
- [17] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple crossmodality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 1
- [18] Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, and Tieniu Tan. Vidcapbench: A comprehensive benchmark of video captioning for controllable text-tovideo generation. arXiv preprint arXiv:2502.12782, 2025. 1
- [19] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 3
- [20] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101,

2024. 1, 2, 3, 5

- [21] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2
- [22] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 3
- [23] Fran¸cois Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2025: Technical report, 2026. 1
- [24] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blis-

- tein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3
- [25] Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees GM Snoek, and Yuki M Asano. Tvbench: Redesigning video-language evaluation. arXiv preprint arXiv:2410.07752, 2024. 1, 2, 3
- [26] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint arXiv:2409.11402, 2024. 1
- [27] Google DeepMind. Gemini flash, 2025. Accessed: 202502-24. 2, 3, 6
- [28] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024. 1
- [29] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1
- [30] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 6
- [31] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1

- [32] Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5374–5383,

2019. 3, 4

- [33] Gunnar Farneb¨ack. Two-frame motion estimation based on polynomial expansion. In Scandinavian conference on Image analysis, pages 363–370. Springer, 2003. 7
- [34] Fawad Javed Fateh, Umer Ahmed, Hamza Khan, M Zeeshan Zia, and Quoc-Huy Tran. Video llms for temporal reasoning in long videos. arXiv preprint arXiv:2412.02930, 2024. 2
- [35] Lishuai Gao, Yujie Zhong, Yingsen Zeng, Haoxian Tan, Dengjie Li, and Zheng Zhao. Linvt: Empower your imagelevel large language model to understand videos. arXiv preprint arXiv:2412.05185, 2024. 2
- [36] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2, 3, 5

- [37] Gunnar Johansson. Visual perception of biological motion and a model for its analysis. Perception & psychophysics, 14(2):201–211, 1973. 1, 2
- [38] Kumara Kahatapitiya, Anurag Arnab, Arsha Nagrani, and Michael S Ryoo. Victr: Video-conditioned text representations for activity recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18547–18558, 2024. 1, 2
- [39] Minkuk Kim, Hyeon Bae Kim, Jinyoung Moon, Jinwoo Choi, and Seong Tae Kim. Do you remember? dense video captioning with cross-modal memory retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13894–13904, 2024. 1, 2
- [40] Dohwan Ko, Ji Soo Lee, Wooyoung Kang, Byungseok Roh, and Hyunwoo J Kim. Large language models are temporal and causal reasoners for video question answering. arXiv preprint arXiv:2310.15747, 2023. 2
- [41] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 3, 4
- [42] Ji Soo Lee, Jongha Kim, Jeehye Na, Jinyoung Park, and Hyunwoo J Kim. Vidchain: Chain-of-tasks with metricbased direct preference optimization for dense video captioning. arXiv preprint arXiv:2501.06761, 2025. 2
- [43] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 2
- [44] Chaoyu Li, Eun Woo Im, and Pooyan Fazli. Vidhalluc: Evaluating temporal hallucinations in multimodal large language models for video understanding. arXiv preprint arXiv:2412.03735, 2024. 1, 2, 3
- [45] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 1, 2, 3, 5
- [46] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 1
- [47] Shicheng Li, Lei Li, Yi Liu, Shuhuai Ren, Yuanxin Liu, Rundong Gao, Xu Sun, and Lu Hou. Vitatecs: A diagnostic dataset for temporal concept understanding of videolanguage models. In European Conference on Computer Vision, pages 331–348. Springer, 2024. 1, 2
- [48] Yunxin Li, Xinyu Chen, Baotian Hu, Longyue Wang, Haoyuan Shi, and Min Zhang. Videovista: A versatile benchmark for video understanding and reasoning. arXiv preprint arXiv:2406.11303, 2024. 2
- [49] Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, et al. Baichuan-omni technical report. arXiv preprint arXiv:2410.08565, 2(3), 2024. 2
- [50] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF confer-

- ence on computer vision and pattern recognition, pages 26689–26699, 2024. 3
- [51] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1, 2
- [52] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2
- [53] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. St-llm: Large language models are effective temporal learners. In European Conference on Computer Vision, pages 1–18. Springer, 2024. 2
- [54] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 1, 2, 3
- [55] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024. 2, 3
- [56] Michael D Mauk and Dean V Buonomano. The neural basis of temporal processing. Annual review of neuroscience, 27: 307–340, 2004. 1, 2, 3
- [57] Hugo Merchant, Deborah L Harrington, and Warren H Meck. Neural basis of the perception and estimation of time. Annual review of neuroscience, 36:313–336, 2013. 2, 3
- [58] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid. Morevqa: Exploring modular reasoning models for video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13235–13245, 2024. 1, 2
- [59] Thong Nguyen, Yi Bin, Junbin Xiao, Leigang Qu, Yicong Li, Jay Zhangjie Wu, Cong-Duy Nguyen, See-Kiong Ng, and Luu Anh Tuan. Video-language understanding: A survey from model architecture, model training, and data perspectives. arXiv preprint arXiv:2406.05615, 2024. 1, 2
- [60] Avalon CS Owens, Mira Van den Broeck, Rapha¨el De Cock, and Sara M Lewis. Behavioral responses of bioluminescent fireflies to artificial light at night. Frontiers in Ecology and Evolution, 10:946640, 2022. 1
- [61] Jongwoo Park, Kanchana Ranasinghe, Kumara Kahatapitiya, Wonjeong Ryoo, Donghyun Kim, and Michael S Ryoo. Too many frames, not all useful: Efficient strategies for longform video qa. arXiv preprint arXiv:2406.09396, 2024. 1
- [62] Joseph J Paton and Dean V Buonomano. The neural basis of timing: distributed mechanisms for diverse functions. Neuron, 98(4):687–705, 2018. 1, 2, 3
- [63] Iqra Qasim, Alexander Horsch, and Dilip Prasad. Dense video captioning: A survey of techniques, datasets and evaluation protocols. ACM Computing Surveys, 57(6):1–36, 2025.

- 1

[64] Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, TatSeng Chua, Yueting Zhuang, and Siliang Tang. Momentor: Advancing video large language model with fine-grained temporal reasoning. arXiv preprint arXiv:2402.11435, 2024.

- 2

- [65] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1
- [66] GM Ram´ırez-Avila,´ J Kurths, and Jean-Louis Deneubourg. Fireflies: A paradigm in synchronization. Chaotic, Fractional, and Complex Dynamics: New Insights and Perspectives, pages 35–64, 2018. 1
- [67] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3, 4
- [68] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 2, 3, 5, 8
- [69] Berkman Sahiner, Heang-Ping Chan, Lubomir M Hadjiiski, Mark A Helvie, Jun Wei, Chuan Zhou, and Yao Lu. Computer-aided detection of clustered microcalcifications in digital breast tomosynthesis: a 3d approach. Medical physics, 39(1):28–39, 2012. 5
- [70] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 2
- [71] Eero P Simoncelli and David J Heeger. A model of neuronal responses in visual area mt. Vision research, 38(5):743–761,

1998. 2

- [72] Oriane Sim´eoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv´e J´egou, Patrick Labatut, and Piotr Bojanowski. Dinov3, 2025. 7
- [73] Itaf Ben Slimen, Larbi Boubchir, and Hassene Seddik. Epileptic seizure prediction based on eeg spikes detection of ictal-preictal states. Journal of biomedical research, 34(3): 162, 2020. 1
- [74] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 3
- [75] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. arXiv preprint arXiv:2312.17432, 2023. 1, 2
- [76] Matthias Tangemann, Matthias K¨ummerer, and Matthias Bethge. Object segmentation from common fate: Motion

- energy processing enables human-like zero-shot generalization to random dot stimuli. Advances in Neural Information Processing Systems, 37:137135–137160, 2024. 1, 2
- [77] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 3, 6
- [78] Qwen Team. Qwen3 technical report, 2025. 3
- [79] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024. 1, 2
- [80] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 2, 3, 5
- [81] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 2
- [82] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024. 3
- [83] Yueqian Wang, Xiaojun Meng, Yuxuan Wang, Jianxin Liang, Jiansheng Wei, Huishuai Zhang, and Dongyan Zhao. Videollm knows when to speak: Enhancing time-sensitive video comprehension with video-text duet interaction format. arXiv preprint arXiv:2411.17991, 2024. 2
- [84] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2.5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 1, 2, 3
- [85] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. In European Conference on Computer Vision, pages 453–470. Springer, 2024. 1, 2, 3
- [86] Max Wertheimer. On perceived motion and figural organization. Mit Press, 2012. 2
- [87] Marc Wittmann. The experience of time: neural mechanisms and the interplay of emotion, cognition and embodiment. Philosophical Transactions of the Royal Society B: Biological Sciences, 364(1525):1809–1813, 2009. 3
- [88] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 1, 2
- [89] Wenhao Wu, Zhun Sun, and Wanli Ouyang. Revisiting classifier: Transferring vision-language models for video recognition. In Proceedings of the AAAI conference on artificial intelligence, pages 2847–2855, 2023. 1, 2

- [90] Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang. Object tracking benchmark. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9):1834–1848, 2015. 3, 4
- [91] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 1
- [92] Yifang Xu, Yunzhuo Sun, Zien Xie, Benxiang Zhai, and Sidan Du. Vtg-gpt: Tuning-free zero-shot video temporal grounding with gpt. Applied Sciences, 14(5):1894, 2024. 1
- [93] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10714–10726, 2023. 1, 2
- [94] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2025. 4
- [95] Zhenyu Yang, Yuhang Hu, Zemin Du, Dizhan Xue, Shengsheng Qian, Jiahong Wu, Fan Yang, Weiming Dong, and Changsheng Xu. Svbench: A benchmark with temporal multi-turn dialogues for streaming video understanding,

2025. 1, 2

- [96] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. Advances in Neural Information Processing Systems, 36:76749–76771, 2023. 1, 2
- [97] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 2, 3
- [98] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024. 1, 2
- [99] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1, 2, 3, 5
- [100] Yue Zhao, Ishan Misra, Philipp Kr¨ahenb¨uhl, and Rohit Girdhar. Learning video representations from large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6586–6597,

2023. 1, 2

- [101] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. 7
- [102] Yaoyao Zhong, Junbin Xiao, Wei Ji, Yicong Li, Weihong Deng, and Tat-Seng Chua. Video question answer-

ing: Datasets, algorithms and challenges. arXiv preprint arXiv:2203.01225, 2022. 1

[103] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2

## Time Blindness: Why Video-Language Models Can’t See What Humans Can? Supplementary Material

### Contents

- A. Data Statistics 2
- B. Prompt Design for Evaluation 2

- B.1. Prompt Design Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- B.2. Direct vs. Chain-of-Thought Prompting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- B.3. Prompt Effectiveness Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2

- C. LLM-as-a-Judge Evaluation 2

- C.1. Evaluation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- C.2. Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- D. Binary Classifier Experiment Details 3

- D.1. Dataset Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- D.2. Experiment Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- D.3. Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- E. Overfitting Experiment with Varying Dataset Sizes 6

- E.1. Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- E.2. Additional Data Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- E.3. Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- F. Impact of FPS 6
- G. Temporal Motion Coherence Analysis 6

- G.1. Motion-Based Perception in Noisy Environments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- G.2. Signal-to-Noise Ratio Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- H. Qualitative Examples of Model Responses 8

- H.1. Direct Prompting Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- H.2. Chain-of-Thought Prompting Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- H.3. Analysis of Failure Modes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- I. Additional Images 8

- A. Data Statistics Figure 1 shows the data distribution of SpookyBench.

|46.6%| |
|---|---|
| | |

|40.8%| |
|---|---|
| | |

|12.6%| |
|---|---|
| | |

Text Images Dynamic Scenes Content Type Encoded in Videos

0

25

50

75

100

125

150

175

200

225

NumberofVideos

210

184

57

SpookyBench Video Categories

- Figure 1. Distribution of the SpookyBench dataset across three video categories. Each category represents a different type of content encoded through temporal noise patterns: Text, Object Images, and Dynamic Scenes.

- B. Prompt Design for Evaluation

Prompt design significantly affects the performance of vision-language models [3, 4]. We performed careful prompt engineering to ensure fair and comprehensive evaluation, developing a systematic prompting methodology that builds on established principles while introducing elements specific to temporal pattern recognition.

- B.1. Prompt Design Principles We designed our prompts based on three key principles:

- 1. Specificity: Each prompt explicitly states that the content is encoded through temporal patterns to direct attention to motion-based cues rather than static frame analysis.
- 2. Category targeting: We created specialized prompts for each content category (text, objects, dynamic scenes) to account for the different perceptual mechanisms involved in each.
- 3. Constrained response format: All prompts request brief, specific answers (1-3 words) to ensure objective evaluation and minimize the influence of language generation capabilities.

- B.2. Direct vs. Chain-of-Thought Prompting

We implemented two distinct prompting strategies shown below. Figure 2 and 3 present our category-specific prompts for both strategies.

- • Direct prompts test immediate pattern recognition without explicit guidance, similar to how humans naturally perceive temporal patterns without conscious step-by-step processing.
- • Chain-of-Thought (CoT) prompts provide explicit steps to guide attention and processing, testing whether models could benefit from structured reasoning about temporal patterns.

- B.3. Prompt Effectiveness Analysis

Neither prompt strategy improved model performance on SpookyBench. All tested models achieved 0% accuracy regardless of prompt type, indicating a fundamental architectural limitation rather than a prompt engineering issue. The complete ineffectiveness of even carefully engineered prompts across all tested models further supports our claim that current videolanguage models lack the mechanisms needed for processing purely temporal patterns.

- C. LLM-as-a-Judge Evaluation

To verify that the observed 0% accuracy is not an artifact of our exact-match evaluation protocol, we additionally employ an LLM-as-a-judge approach, a standard evaluation methodology widely adopted in VLM benchmarks [2]. Specifically,

###### Text-Specific Direct Prompt

This video contains text encoded through temporal patterns. What specific word or phrase do you see? The text is only visible through the temporal changes in the video. Please respond with just the text you identify.

###### Object-Specific Direct Prompt

This video contains a common object encoded through temporal patterns. Individual frames may appear as noise, but an object is visible through the temporal changes. What object do you see? Please respond with just the object name.

###### Dynamic Scenes-Specific Direct Prompt

This video contains movement or action encoded through temporal patterns. The content is only visible through temporal changes, not in individual frames. What is being shown? Please respond with just 1-3 words describing what you see.

- Figure 2. Category-specific direct prompts for the SpookyBench benchmark. These prompts test immediate pattern recognition without step-by-step guidance.

we prompt an LLM to assess whether each model’s output is consistent with the ground-truth label, rather than requiring a verbatim match. This supplementary evaluation addresses the possibility that models may exhibit partial understanding of temporal content yet express it in a form that diverges from the predefined label space.

##### C.1. Evaluation Pipeline

- Figure 4 illustrates our LLM-as-a-judge evaluation pipeline. For each video, we provide the judge model (GPT-4o) with: (1) the ground-truth label(s), (2) the model’s response, and (3) instructions to determine whether the response correctly identifies the content, allowing for paraphrases, synonyms, and partial matches.

##### C.2. Results

The LLM-as-a-judge evaluation yields identical 0% accuracy across all models and categories. This confirms that the failure is not due to strict string matching: models do not produce responses that are even semantically close to the correct answers. Instead, they hallucinate unrelated content (see Appendix H for examples), confirming that the evaluation protocol is robust.

### D. Binary Classifier Experiment Details

To test whether state-of-the-art visual models can detect even the presence of foreground objects in SpookyBench videos (without identifying them), we trained binary classifiers using VJEPA-2 [1] and DINOv3 [5].

- D.1. Dataset Construction We constructed a balanced binary classification dataset as follows:

- • Positive class: SpookyBench videos containing encoded foreground content (text, objects, or dynamic scenes).
- • Negative class: Videos generated with the same noise generation pipeline but without any foreground mask applied, i.e., uniform random noise with no embedded temporal pattern.

The dataset was split into 80% training and 20% validation sets, maintaining class balance in both splits.

- D.2. Experiment Details

To evaluate whether pretrained video and image representation models can discriminate foreground noise presence from raw video, we finetune two recent self-supervised models on our binary classification task: VJEPA-2 (ViT-L/16, 64 frames per

###### Text-Specific CoT Prompt

This video encodes text through temporal patterns. Let’s think step by step to identify it:

- 1. Observe how noise pixels move differently across regions of the frame
- 2. Look for areas where opposing motion patterns create visible boundaries forming letters
- 3. Focus on the overall word or phrase that emerges from these motion boundaries
- 4. Read the specific text content revealed by the temporal dynamics Please respond with just the text you identify.

###### Object-Specific CoT Prompt

This video encodes an object through temporal patterns. Let’s think step by step to identify it:

- 1. Notice that individual frames appear as random noise
- 2. Track how groups of pixels move coherently over time in different directions
- 3. Look for areas where motion patterns reveal object contours and silhouette
- 4. Focus on the overall form that emerges and determine what specific object is represented

Please respond with just the object name.

###### Dynamic Scenes-Specific CoT Prompt

This video encodes movement through temporal patterns. Let’s think step by step to identify it:

- 1. Observe that each frame looks like noise, but regions move differently over time
- 2. Look for areas where temporal changes reveal motion of a foreground object
- 3. Focus on the action or activity that emerges from the separation of moving and static regions
- 4. Identify the specific movement or object in motion Please respond with just 1-3 words describing what you see.

- Figure 3. Category-specific chain-of-thought prompts for the SpookyBench benchmark. These prompts provide explicit step-by-step guidance to test structured temporal reasoning.

clip) [1] and DINOv3 (ConvNeXt-S, pretrained on LV1689M) [5]. For both DINOv3 and VJEPA-2, we load their respective models with a randomly initialized attention pooler and classifier head, and finetuned the full model on 64-frame clips at 256×256 resolution. Both models were trained for 100 epochs using cosine-annealed learning rates (DINOv3: 10−3 with Adam; VJEPA-2: 10−4 with AdamW, weight decay 0.01) on a stratified 90/10 train/validation split of the 1,000-video dataset (500 per class). Training was performed on a single NVIDIA RTX PRO 6000 GPU per model (batch sizes of 64 and 4, respectively). Neither model learned a discriminative representation. DINOv3 training loss converged to ∼0.689 with validation accuracy fluctuating between 48–52%, barely above the 50% chance baseline. VJEPA-2 exhibited even stronger signs of failure: validation accuracy remained at exactly 50% for the majority of training, with the loss plateauing near ln2 ≈ 0.693 — the theoretical maximum-entropy baseline for balanced binary classification.

These results confirm that the temporal signal encoding foreground noise presence in our stimuli is inaccessible to architectures that operate on per-frame spatial features, even when those features are aggregated temporally (DINOv3 mean

SpookyBench Video Ground-Truth Label(s)

Video-Language Model

Model Response

LLM Judge (GPT-4o)

“Does the response correctly identify the content?”

Verdict: Correct / Incorrect

- Figure 4. LLM-as-a-Judge evaluation pipeline. The judge receives both the ground-truth label and the model’s response, and determines whether the response correctly identifies the temporal content, allowing for synonyms and paraphrases.

pooling) or processed by a dedicated spatiotemporal encoder (VJEPA-2). The foreground and background noise patterns are visually indistinguishable in any single frame; discrimination requires detecting coherent differential motion across frames, a signal that neither frozen per-frame representations nor end-to-end video encoders could extract from the raw pixel input under these training conditions.

- D.3. Results Both models converge to approximately 50% validation accuracy (random chance on a balanced dataset), as shown in Figure
- 4 of the main paper. VJEPA-2 reached 52.8% and DINOv3 reached 53.2% after 30 epochs, with training loss oscillating rather than decreasing. This confirms that the features extracted by these models do not encode the temporal motion patterns present in SpookyBench videos, even at the coarse level of detecting whether a foreground object exists. We also fine-tune Qwen3-VL-8B on the same binary classification task, which similarly achieved 0% meaningful accuracy, further confirming that VLM architectures cannot extract these temporal patterns even with supervised signal. Table 1 reports the training accuracy after 80 epochs across varying dataset sizes. VJEPA-2 and DINOv3 hover near chance level regardless of the number of training videos, while Qwen3-VL-8B fails entirely at 0%.

###### Model 100 videos 400 videos 1,100 videos

VJEPA-2 52.3% 56.1% 52.8% DINOv3 52.7% 52.9% 53.2%

Table 1. Training accuracy after 80 epochs for binary classification (foreground presence detection) across different dataset sizes. All models remain near or below chance level.

### E. Overfitting Experiment with Varying Dataset Sizes

A natural question is whether the 0% accuracy of fine-tuned VLMs is simply due to insufficient training data. To address this, we conducted finetuning experiments with varying dataset sizes.

- E.1. Setup We fine-tuned InternVL2.5-8B and Qwen2-VL-7B using LlamaFactory [6] on three dataset configurations:

Configuration Training Videos Epochs Test Accuracy

Small 100 30 0% Medium 400 30 0% Large 1,100 30 0%

Table 2. Fine-tuning results across different dataset sizes. All configurations yield 0% test accuracy.

- E.2. Additional Data Details

For the large configuration (1,100 videos), we generated additional videos beyond the original 451-video SpookyBench dataset using our data generation pipeline. The additional videos follow the same distribution across categories and use the same noise generation parameters (speckle sizes, noise densities, velocity values) as the original dataset. The test set remained fixed across all configurations to ensure comparability.

- E.3. Analysis

The consistent 0% accuracy across all dataset sizes demonstrates that the failure cannot be attributed to insufficient training data. Even with 1,100 training videos and 30 epochs (providing ample opportunity for overfitting), the models cannot learn to extract temporal patterns from the data. This rules out a simple out-of-distribution explanation and points to an architectural inability to process the temporal information encoded in SpookyBench videos.

### F. Impact of FPS

To test whether frame rate affects the ability to perceive temporally encoded information, we evaluate both human participants and VLMs across frame rates from 1 to 30 FPS. We test three human participants on 120 randomly sampled videos (40 per category) at frame rates of 1, 5, 10, 20, and 30 FPS. For VLMs, we evaluated Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen2.5-VL3B, and GPT-4o. Human accuracy remains above 95% at 20–30 FPS, degrades to 59.4% at 10 FPS, and drops to 0% at 1 FPS (see Table 3 in the main paper). In contrast, all VLMs achieved 0% accuracy at every frame rate tested (Table 6 in the main paper). Human performance degrades gracefully as frame rate decreases, consistent with the known temporal resolution limits of motion perception. VLMs show no sensitivity to frame rate at all, confirming that they are not engaging with temporal information in any meaningful way, regardless of how much temporal data is provided. This rules out temporal undersampling as an explanation for the performance gap.

### G. Temporal Motion Coherence Analysis

Temporal coherence and motion boundaries provide powerful cues that enable the human visual system to extract shapes from noisy stimuli. We present a comprehensive analysis of these phenomena using our SpookyBench dataset.

##### G.1. Motion-Based Perception in Noisy Environments

Even when individual frames have low SNR, temporal integration of motion information allows robust shape perception.

- Figure 5 shows the motion direction coherence map of an ant silhouette, revealing how consistent motion patterns across frames enable object identification despite significant noise. Figure 7 shows both the average motion boundary strength and its overlay on a noisy frame. Motion boundaries emerge clearly despite extreme noise levels in individual frames (measured at

-49.07 dB basic SNR).

[Figure 28]

- Figure 5. Motion Direction Coherence visualization for the ant silhouette video. Yellow regions (coherence value 1.0) indicate consistent motion direction across frames; blue regions (coherence value 0.0) represent the silhouette itself.

[Figure 29]

(a) Estimated object mask from temporal motion coherence.

[Figure 30]

(b) Estimated mask overlay (red) on a single noise frame.

- Figure 6. Shape extraction through temporal integration, demonstrating how object shape can be recovered from noisy video sequences.

[Figure 31]

(a) Average motion boundary strength across frames.

[Figure 32]

(b) Motion boundaries (teal) overlaid on a single noise frame.

- Figure 7. Motion boundary analysis showing how temporal integration extracts object boundaries despite extremely noisy individual frames.

##### G.2. Signal-to-Noise Ratio Analysis

Table 3 presents detailed SNR results for the ant silhouette video. While basic and perceptual SNR are extremely low (-49.07 dB and -55.02 dB), temporal coherence SNR and motion contrast SNR are positive (7.18 dB and 14.24 dB), showing

###### SNR Metric Value (dB)

Basic SNR -49.07 Perceptual SNR -55.02 Temporal Coherence SNR 7.18 Motion Contrast SNR 14.24 Combined SNR -20.61

Table 3. Signal-to-Noise Ratio Analysis for Ant Silhouette Video.

that temporal information provides signal enhancement that supports human perception. The contrast between negative frame-based SNR values and positive temporal SNR metrics directly supports our hypothesis: temporal integration is essential for perceiving content in SpookyBench stimuli, and current architectures lack this capability.

### H. Qualitative Examples of Model Responses

We present representative model responses to illustrate the failure modes observed across VLMs on SpookyBench. These examples show that models do not produce partially correct or semantically related answers; instead, they hallucinate entirely unrelated content.

- H.1. Direct Prompting Responses

Figure 8 presents four representative model responses to the text-specific direct prompt. These responses illustrate distinct failure modes, including timestamp hallucination, prompt echoing, outright refusal, and confident hallucination of unrelated content.

- H.2. Chain-of-Thought Prompting Responses

Figure 9 shows model responses when given explicit step-by-step reasoning guidance. Despite the structured CoT prompt, models generate elaborate but entirely fabricated scene descriptions that bear no relation to the actual video content.

- H.3. Analysis of Failure Modes Across all tested models, we observe four distinct failure patterns:

- 1. Timestamp hallucination: Models output numerical timestamps instead of content, suggesting they attempt to process temporal metadata rather than temporal visual patterns.
- 2. Prompt echoing: Models repeat parts of the prompt as their answer, indicating no actual visual processing occurred.
- 3. Confident hallucination: Models produce specific but incorrect answers with high confidence. For instance, Qwen-series models consistently predict “clock” or “coffee cup” regardless of input, indicating pattern collapse to high-frequency training concepts.
- 4. Scene fabrication: When given CoT prompts, models generate elaborate descriptions of imaginary scenes (cats, buildings, paper textures) that have no correspondence to the actual video content, suggesting they are generating plausible-sounding narratives from their language prior rather than processing visual input. None of these failure modes produce responses that are even partially related to the actual content encoded in the videos,

which is consistent with the 0% accuracy observed under both exact-match and LLM-as-a-judge evaluation.

#### I. Additional Images We present additional images from the analysis of temporal motion coherence across all categories in SpookyBench. Figures

- 10, 11, 12, and 13 show motion boundaries, boundary overlays, estimated masks, and mask overlays for different examples in our dataset, demonstrating the varying effectiveness of temporal integration across different content types. In Figure 13, the temporal motion coherence method does not perform effectively for dynamic scenes. Since this category contains real-life videos with complex motion patterns, several factors reduce clarity:

- 1. Distributed Motion Patterns: Human movement involves multiple articulated body parts moving in different directions simultaneously, creating competing motion signals that fragment coherent boundaries.
- 2. Non-rigid Deformation: Dynamic content involves continuous shape changes throughout motion sequences, making consistent boundary extraction more challenging than for static objects.

###### Prompt

“This video contains text encoded through temporal patterns. What specific word or phrase do you see? The text is only visible through the temporal changes in the video. Please respond with just the text you identify.”

- Response 1 (Timestamp Hallucination) “The text in the video is: 4.3, 4.5, 4.7, 4.8 seconds.”

- Response 2 (Prompt Echoing) “I see the phrase ‘This video contains text encoded through temporal patterns.’”

- Response 3 (Refusal / No Detection)

“I apologize, but I am unable to see any text or patterns in the provided video. It appears to be a random sequence of pixels with no discernible pattern or text.”

- Response 4 (Confident Hallucination) “The word ‘GOOD’ is visible in the video.”

- Figure 8. Direct prompting responses on SpookyBench. The prompt (top, blue) is shown together with four representative model responses (red), illustrating distinct failure modes.

###### 3. Complex Temporal Dynamics: Mechanical motions in examples like Plane 2 and Bicycle 10 create temporal discontinuities that disrupt the motion coherence essential for shape perception.

###### Prompt

“This video encodes text through temporal patterns. Let’s think step by step to identify it: 1. Observe how noise pixels move differently across regions of the frame. 2. Look for areas where opposing motion patterns create visible boundaries forming letters. 3. Focus on the overall word or phrase that emerges from these motion boundaries. 4. Read the specific text content revealed by the temporal dynamics. Please respond with just the text you identify.”

###### CoT Response 1 (Scene Fabrication)

“In the video, a black and white photo of a cat is shown first. Next, the scene shifts to a black and white texture, followed by a black and white photo of a building. Then, a black and white photo of a cat is shown again. The scene changes to a black and white photo of a building and a black and white photo of a cat again. Finally, a black and white texture of a television screen appears. The video continues to show a black and white photo of a cat. Throughout the video, the dense captions describe the appearance of various objects in each scene, including trees, buildings, people, and various textures. The video is black and white, and the scenes appear to be sequenced in a linear fashion.”

###### CoT Response 2 (Texture Description)

“First, the video shows a texture of an old sheet of paper in black and white. Then, it shows a series of lines and textures on a black and white background. Next, the video shows a black and white texture of old paper. After that, it shows a texture of a piece of paper on a black and white background. Then, it shows a black and white background of textured material. Finally, the video shows a black and white textured pattern. Throughout the video, there are also various objects, such as a dog, a chair, and a handwritten note, but they are not the main focus of the video.”

- Figure 9. Chain-of-thought prompting responses on SpookyBench. The CoT prompt (top, green) is shown together with two representative model responses (red), illustrating elaborate but entirely fabricated descriptions.

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

###### Figure 10. Temporal motion coherence analysis for Images category (Part 1). Each row shows motion boundaries, boundary overlay, estimated mask, and mask overlay for: Cycle, Deer, Dolphin, and Duck (top to bottom).

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

###### Figure 11. Temporal motion coherence analysis for Images category (Part 2). Each row shows motion boundaries, boundary overlay, estimated mask, and mask overlay for: Kangaroo, Hammer, T-Rex, and Mouse (top to bottom).

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

###### Figure 12. Temporal motion coherence analysis for Words category. Each row shows motion boundaries, boundary overlay, estimated mask, and mask overlay for: Gold, Laser Beams Cross, Ancient Olive Trees, and Artificial Minds Think (top to bottom).

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

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

###### Figure 13. Temporal motion coherence analysis for Videos category. Each row shows motion boundaries, boundary overlay, estimated mask, and mask overlay for: Human 6, Man 1, Plane 2, and Bicycle 10 video sequence (top to bottom).

### References

- [1] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. 3, 4
- [2] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 2
- [3] Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, and Philip Torr. A systematic survey of prompt engineering on vision-language foundation models. arXiv preprint arXiv:2307.12980, 2023. 2
- [4] Woojeong Jin, Yu Cheng, Yelong Shen, Weizhu Chen, and Xiang Ren. A good prompt is worth millions of parameters: Low-resource prompt-based learning for vision-language models. arXiv preprint arXiv:2110.08484, 2021. 2
- [5] Oriane Sim´eoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv´e J´egou, Patrick Labatut, and Piotr Bojanowski. Dinov3, 2025. 3, 4
- [6] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. 6

