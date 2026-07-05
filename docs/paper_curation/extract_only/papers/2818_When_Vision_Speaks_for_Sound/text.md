# arXiv:2605.16403v1[cs.CV]13May2026

## When Vision Speaks for Sound

Xiaofei Wend Wenjie Jacky Mod Xingyu Fup Rui Caid Tinghui Zhud Wendi Liw Yanan Xieu Muhao Chend Peng Qiu dUniversity of California, Davis pPrinceton University wUniversity of Wisconsin–Madison uUniphore

Website: when-vision-speaks-for-sound Code Model

[Figure 1]

[Figure 2]

### Abstract

Despite rapid progress in video-capable MLLMs, we find that their apparent audio understanding in videos is often vision-driven: models rely on visual cues to infer or hallucinate acoustic information, rather than verifying the audio stream. This issue appears across both state-of-the-art open-source omni models and leading closed-source models from providers such as Google and OpenAI. We characterize this failure mode as an audio-visual Clever Hans effect, in which models appear (falsely) audio-grounded, but actually exploit visual-acoustic correlations without verifying whether the audio and visual streams are truly aligned. To systematically study this behavior, we introduce THUD, an intervention-driven probing framework based on three counterfactual audio edits: Shift, which tests temporal synchronization; Mute, which tests sound existence; and Swap, which tests audio-visual consistency. Beyond diagnosis, we further study a two-stage alignment recipe: intervention-derived preference pairs teach audio verification, while event-level general video preferences regularize the model against over-specialization. Our best 10K-sample recipe improves average performance across the three intervention dimensions by 28 percentage points, while slightly improving performance on general video and audio-visual QA benchmarks.

### 1 Introduction

Multimodal Large Language Models (MLLMs) have rapidly advanced video understanding [35, 37, 74]. Powered by foundation models such as GPT [41], Gemini [22], and Qwen-VL [57], recent VideoLLMs [14, 30, 54, 71] can interpret dynamic scenes [18, 47], answer complex questions [31, 44], and follow instructions [27, 63]. Yet, in videos with both visual and acoustic signals, such capabilities can blur the boundary between genuine audio-visual grounding and visually driven narration. For example, when shown a skateboarder crashing onto concrete, a model may describe a heavy thud even when the audio evidence is absent or misaligned [8, 24, 34, 52]. Such behavior is often interpreted as multimodal perception, but it may instead reflect an illusion of audio-visual understanding: the model predicts what a video should sound like from what it sees. While static vision-language models are known to behave like “bags-of-words” driven by text priors [59, 61, 68], analogous prediction shortcuts in dynamic audio-visual contexts remain underexplored. This raises a central question: Are current video-capable multimodal models truly performing audio-visual grounding, or merely hallucinating acoustic events from visual-semantic shortcuts?

Clever Hans Effect in Audio-Visual Grounding

A video-capable multimodal model exhibits a Clever Hans effect when it appears audio-grounded but produces sound-related outputs primarily from visual cues rather than verified audio evidence.

Preprint.

[Figure 3]

[Figure 4]

- Figure 1: When vision speaks for sound. Given the same visual event but different audio tracks, current video-capable models produce nearly identical captions, suggesting visual-prior shortcutting rather than audio-grounded understanding.

We find that current video-capable MLLMs are often visually dominated when reasoning about audio-related information in sounded videos. As illustrated in Fig. 1, this shortcut can lead models to produce nearly unchanged descriptions even when the audio track changes substantially. This behavior resembles the famous Clever Hans effect [45], where apparent competence arises from exploiting unintended but correlated cues rather than performing the intended task. Such semantic laziness [19] allows models to exploit visual-semantic shortcuts and language priors instead of fine-grained audio-visual grounding that checks whether the audio and visual streams are temporally and semantically consistent [23, 68]. This failure often remains hidden because common audio-visual evaluations preserve the natural correlations that make such shortcuts effective [9, 11, 20]: barking dogs produce barks, falling objects produce impacts, and speaking faces produce speech [3, 43]. As a result, a model can appear grounded by recognizing the visual event and predicting its likely sound, without verifying whether that sound is actually present, synchronized, or physically consistent. This pseudo-alignment creates an illusion of multimodal understanding that current evaluations often fail to expose [32, 38]. To expose the Clever Hans effect, evaluation must move beyond naturally correlated videos and use controlled interventions that systematically break the audio-visual correspondences that allow visual-semantic shortcuts to succeed [28, 39].

To this end, we introduce THUD (Temporal and Hallucination Unmasking Diagnostics), an intervention-driven diagnostic protocol for probing audio-visual grounding in sounded videos. THUD constructs a dynamic probing space by counterfactually perturbing the audio-visual correspondences of natural videos across temporal synchronization, audio existence, and sound consistency, thereby neutralizing semantic shortcuts and exposing whether a model engages in genuinely grounded audio-visual reasoning or merely hallucinates from visual-semantic and language priors. Beyond diagnosis, we further study whether targeted post-training can mitigate these shortcuts through a family of alignment recipes that combine intervention-derived preference pairs with general video data. The best-performing recipe uses a 10K-sample mixture of counterfactual temporal preferences and event-level general video supervision, substantially improving the model’s ability to detect temporal interventions, including out-of-distribution synchronization tests, while avoiding an alignment tax [4, 42] on standard video understanding benchmarks. Additional targeted supervision on Mute and Swap further improves audio-existence and sound-consistency verification, showing that interventionbased training can be extended beyond temporal alignment. However, the same training yields only marginal gains without such targeted examples, suggesting that temporal synchronization, audio existence, and sound consistency are distinct failure modes of grounded audio-visual understanding rather than a single unified deficiency.

In summary, we make three contributions: 1) We identify and systematically expose a Clever Hans effect in current Video-LLMs, where models substitute genuine audio-visual grounding with visualsemantic shortcuts. Through controlled interventions, we quantify how strongly models rely on visual priors when answering sound-related questions. 2) We introduce THUD, a counterfactual diagnostic protocol that dismantles natural cross-modal correlations. By applying Mute, Shift, and Swap interventions, THUD audits existential, temporal, and material aspects of audio-visual grounding. 3) We evaluate preference-optimization recipes for mitigating audio-visual shortcuts. Our final 10K recipe improves average performance across Shift, Mute, and Swap interventions by 28%, while slightly improving general video and audio-visual understanding.

[Figure 5]

Q: What did you hear when the crash happened?

Ground-truth impact moment

[Figure 6]

[Figure 7]

[Figure 8]

| |
|---|

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Random music track 🎵 🎶

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Visual impact Visual impact Visual impact Visual impact

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

The crash is accompanied by a clear, loud impact sound.

A sudden metallic noise is heard as the crash happen.

People shouting “oh my god” at the background.

There is a dull thud when the sled collides with the tree.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

The bike crash with sound like breaking glass.

The crash is accompanied by a splash-like sound.

I hear what sounds like a child yelling.

There is a solid thud when the bike reaches the roof.

✅ Correct sound grounding ❌ Misses temporal shift ❌ Hallucinates sound ❌ Visual-biased prediction

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

- Figure 2: Representative failure cases under Shift, Mute, and Swap interventions. Gemini and Qwen3-Omni often rely on visual priors rather than verifying the audio stream, leading to missed temporal shifts, hallucinated sounds, and visually biased predictions.

### 2 How Can We Align Models Beyond Visual Shortcuts?

- Fig. 2 illustrates that even native multimodal models such as Gemini and Qwen3-Omni can produce plausible acoustic interpretation from visual actions alone, rather than verifying whether the corresponding sound is present, temporally aligned, or consistent with its visual source. These failures motivate our intervention-driven diagnostic protocol, which deliberately breaks natural audio-visual correlations to expose models’ reliance on visual-semantic shortcuts.

To align models beyond visual shortcuts, we construct training signals that task them to compare visible events against the actual audio stream rather than rely on visual priors. Our recipe turns physical audio-visual interventions into alignment data in three steps. First, we source videos with salient acoustic consequences and break natural correlations (§2.1). Second, we annotate event-time labels and construct chosen–rejected preference pairs (§2.2). And third, we combine intervention data with general video instruction data to preserve overall comprehension (§2.3).

#### 2.1 Data Sourcing and Physical Interventions

To build intervention data for audio-visual grounding, we use the Oops dataset [15], a collection of in-the-wild videos centered on unintentional human actions. As shown in Appx. §A.1, Oops contains many failure-centered events, such as slipping, skiing crashes, and objects breaking, that naturally induce strong expectations about the accompanying sound. This property makes it a suitable source for constructing Clever Hans-style cases: the visual content often suggests a plausible acoustic event, while the audio track determines whether that event is actually present, temporally aligned, and physically consistent with the observed action.

Formalizing interventions. Let a video be represented as v = (x1:T,a1:T), where x1:T denotes the visual stream and a1:T denotes the audio track. We construct intervened videos by applying one of three operators:

v˜ = Ik(v), k ∈ {Shift,Mute,Swap}. (1) For Shift, the audio track is displaced by a temporal offset ∆:

ISHIFT(v;∆) = (x1:T,a+∆1:T ), ∆ ∈ [−∆max,∆max]. (2) Here, ∆ < 0 corresponds to an early audio event, while ∆ > 0 corresponds to a delayed audio event. This intervention requires the model to compare the timing of the visible event with the timing of its acoustic consequence.

For Mute, the audio signal is replaced with silence:

IMUTE(v) = (x1:T,∅). (3) For Swap, the original audio is replaced with an audio track a′1:T from another video:

##### ISWAP(v,v′) = (x1:T,a′1:T), v′ = (x′1:T,a′1:T). (4)

The substituted audio is acoustically plausible but physically inconsistent with the visible event, forcing the model to verify audio-visual consistency rather than rely on the most likely sound implied by vision alone. Overall, these interventions convert naturally correlated videos into controlled counterfactual cases that target temporal synchronization, sound presence, and physical consistency; a detailed summary is provided in Appx. §A.2.

- 2.2 Annotation and Preference Pair Construction We annotate each source video with event-time labels used to evaluate audio-visual interventions:

##### zi = (evi ,tvi ,eai ,tai ), (5)

where evi and tvi denote the visual event and its timestamp, eai and tai denote the corresponding acoustic event and timestamp. These fields correspond to the visual event, visual time, audio event,

and audio time labels in Fig. 9 (Appx. §A.1).

Cross-model verification. We use Gemini to generate initial event-time annotations because it supports direct video ingestion and can inspect both visual and audio streams. For visual timestamps, we further verify Gemini’s annotations with GPT and Claude by decomposing each video into N temporally ordered frame units and asking the models to locate the visual event within the frame sequence. For audio timestamps, which require access to the acoustic stream, we cross-verify Gemini’s predictions with human inspection.

Let Mv denote the set of visual annotator models and let Ma = {Gemini,Human} denote the audio verification sources.

zi(m) = (ev,mi ,tv,mi ,ea,mi ,ta,mi ), (6)

where visual fields are available for m ∈ Mv and audio fields are available for m ∈ Ma. A sample is automatically retained when both visual and acoustic timestamps agree within strict tolerances:

′

′

tv,mi − tv,m

ta,mi − ta,m

##### i ≤ ϵa. (7)

i ≤ ϵv, max

max

m,m′∈Mv

m,m′∈Ma

Here, ϵv and ϵa denote the tolerance thresholds for visual and acoustic timestamps, respectively. Cases with model disagreement are manually inspected and corrected to ensure reliable event-time labels. We provide the annotation prompts, frame-unit construction details, agreement criteria, and manual verification protocol in Appx. §B.

Preference pair construction. The annotated intervention cases are converted into chosen–rejected preference pairs:

##### Dpref = v ˜i,qi,yi+,yi− Ni=1 , (8)

where v˜i is the intervened video, qi is the diagnostic prompt, yi+ is the chosen response, and yi− is the rejected response. The chosen response explicitly verifies the audio-visual relation, while the

rejected response is visually plausible but inconsistent with the audio evidence, approximating the shortcut behavior we aim to suppress. The overall annotation and intervention pipeline is summarized in Fig. 9 (Appx. §A.1).

For Shift, chosen responses detect early or delayed audio, while rejected responses claim synchronization or the wrong temporal direction. For Mute, chosen responses identify silence, while rejected responses hallucinate expected sounds. For Swap, chosen responses flag audio-visual source inconsistency, while rejected responses accept the mismatched sound. These pairs train the model to verify audio evidence rather than follow visually plausible shortcuts. Examples are provided in Appx. §D.

- Table 1: Paired diagnostic accuracy (%) of video-capable multimodal models. Orig. denotes naturally correlated controls, while Shift, Mute, and Swap denote counterfactual interventions. Avg Gap is the average accuracy drop, reflecting shortcut reliance.

Temporal Sync. Audio Existence Sound Consistency

Model Size

Avg Gap Orig. Shift Orig. Mute Orig. Swap

Gemini N/A 54.9 46.5 100.0 13.4 93.6 18.3 56.8 MiniCPM-o-4.5 9B 83.8 13.7 100.0 19.0 95.8 4.9 80.7 Nemotron-3-Omni 30B 35.9 26.8 66.2 4.2 88.7 19.9 46.6 Qwen3-Omni 30B 100.0∗ 1.4 95.1 0.0 75.4 37.3 77.3 Ming-Omni-2.0 100B 54.2 20.1 95.7 54.9 90.1 15.5 49.8 MiMo-V2.5 311B 73.9 9.9 99.3 2.1 89.4 15.3 78.4

#### 2.3 Two-Stage Alignment with General Video Data

Intervention data provides targeted supervision for detecting Shift, Mute, and Swap failures, but may over-specialize the model to counterfactual cases. We therefore mix it with general video instruction data, whose temporally segmented annotations expose ordinary audio-visual correspondences at the event level. Appx. §A.4 summarizes this two-stage alignment pipeline.

We use FineVideo [16] as the source of general video data because its annotations are organized around time segments, describing what occurs from one timestamp range to the next. We re-annotate selected FineVideo clips with Gemini and apply human agreement checks, enriching the original segment annotations with both visual and audible event-level information. The resulting annotations are used to construct four instruction types summarized in Appx. §E.

Our training follows the standard post-training recipe of Supervised Fine Tuning (SFT) followed by preference alignment [12, 42, 77]. We use SFT warm-up on intervention-derived data to establish audio-aware response patterns, and then apply DPO on intervention preference pairs mixed with general video data to favor audio-verified responses over visually plausible shortcuts. The general video mixture is included to reduce over-specialization to intervention cases and preserve broad video understanding. The overall two-stage alignment pipeline is summarized in Fig. 10 (Appx. §A.4).

### 3 Experiments

This section presents the experiments for diagnosing audio-visual shortcut reliance and evaluating targeted alignment, covering the setup (§3.1), shortcut analysis (§3.2), targeted alignment improvements (§3.3), and broader intervention results (§3.4).

#### 3.1 Experimental Setup

Evaluation conditions and metrics. We evaluate audio-visual grounding under four conditions: Original, Shift, Mute and Swap. Original videos serve as positive controls with natural audio-visual correspondence, while the interventions probe audio existence, temporal synchronization, and sound consistency. We report paired accuracy for each grounding dimension.

Models. We group evaluated models by access mode. The API-tested models include Gemini-3.1Pro [22], MiMo-V2.5 [67], and Nemotron-3-Nano-Omni [55]. We also query GPT-5.5 [41], but omit it from Tab. 1 because its tested interface does not support direct audio input for video; its outputs are provided in Appx. §F. The locally evaluated models include MiniCPM-o-4.5 [13], Qwen3-Omni [56], and Ming-flash-omni-2.0 [53].

Training and general capability evaluation. For controlled training experiments, we use Qwen3Omni-30B as the trainable backbone and compare checkpoints trained with different combinations of intervention data and general video data. To test whether intervention training incurs an alignment tax, we evaluate these checkpoints on Video-MME [17], LVBench [62], DailyOmni [75], and WorldSense [25], which measure general video and omni-modal understanding beyond our intervention distribution. We further evaluate on VGGSoundSync [10] to test out-of-distribution temporal synchronization beyond our constructed intervention set.

#### 3.2 Do Video-Capable Multimodal Models Rely on Visual Shortcuts?

We examine whether video-capable multimodal models verify the audio stream or infer plausible sounds from visual context. Tab. 1 reports paired diagnostic accuracy under naturally correlated Original controls and counterfactual interventions. Original videos serve as positive controls, while drops under Shift, Mute, or Swap reveal failures when natural audio-visual correlations are broken. Avg Gap measures the average accuracy drop from Original to intervention conditions, with larger values indicating a larger performance collapse under counterfactual interventions. Its formula and the LLM-judge protocol for free-form outputs are provided in Appx. §G.

Audio-aware multimodal failure modes

Overall, most models show large drops from Original to intervention settings, indicating that strong performance on naturally correlated videos is fragile. MiniCPM-o-4.5 and MiMo-V2.5 have the largest gaps, 80.7% and 78.4%. Qwen3-Omni is diagnostic: its perfect original temporal-sync accuracy drops to 1.4% under Shift, suggesting a synchronized-default prior rather than true temporal grounding. These results suggest that current models often rely on visual-semantic priors instead of verifying audio presence, timing, and source consistency.

###### Gemini-3.1-proMiniCPM-o-4.5Nemotron-3-OmniQwen3-OmniMing-Omni-2.0MiMo-V2.5

|0.87 0.81 0.68 0.99 0.39 0.99<br><br>0.82 0.95 0.80 0.63 0.85 0.85<br><br>0.00 0.00 0.00 0.00 0.01 0.00<br><br>0.06 0.04 0.11 0.25 0.10 0.11<br><br>0.00 0.00 0.31 0.03 0.05 0.00<br><br>0.30 0.74 0.31 0.98 0.51 0.71<br><br>0.33 0.47 0.61 0.43 0.59 0.66<br><br>0.45 0.16 0.64 0.00 0.46 0.26|
|---|

###### Mute Hallucination

1.0

(claims sounds on silent video)

[Figure 35]

###### Swap False-Match

(claims swapped audio matches)

0.8

###### False Silence

(claims silence on real audio)

###### Swap False-Mismatch

0.6

Failurerate

(claims real audio doesn't match)

###### Audio Dodge

(describes only visuals)

0.4

###### Offset Blindness

(misses ±2s sync offset)

0.2

###### Direction Confusion

(delay early flipped)

###### False Sync Alarm

0.0

- Fig. 3 exposes a uniform shortcut. Every model saturates on audio hallucination, with Mute Hallucination and Swap False-Match both above 0.63 across the board, while their symmetric counterparts (False Silence, Swap False-Mismatch) sit near zero: models invent audio that fits the visuals but rarely deny audio that is real. Temporal perception is worse. Qwen3-Omni misses 98% of ±2s offsets; MiniCPM and MiMo miss roughly three quarters; and even when an offset is flagged, the delay/early sign is wrong about half the time, close to a random label. Definitions for each axis are given in Appx. §H.
- Fig. 4 decomposes each model’s predictions on the three intervention tasks. On Mute and Swap, almost all errors collapse onto Hallucinated synced, with five of six models fabricating matching audio on over 80% of muted clips and the mismatched class recovered at most 37% of the time. Hallucinated shift is negligible everywhere, indicating that models hold a strong synced prior and rarely entertain temporal alternatives. The Shift panel makes the consequence concrete: Qwen3-Omni answers synced on 98% of inputs, while Gemini-3.1-Pro, Nemotron-3-Omni, and Ming-Omni-2.0 lose 19 to

(claims offset on synced original)

Figure 3: Failure-mode heatmap. Red indicates higher failure; audio hallucination dominates, while temporal failures are model-specific.

###### Mute task

###### Swap task

###### Shift task

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |0.27| | | | | |
| | | | | | | | | | | | | |
| | | | | |0.39| | | | | | | |
| | | | | | | | | | | | | |
| |0.87| |0.99| | | | | |0.81| |0.99| |
| | | | | | | | | | | | | |
| | | | | |0.55| |0.68| | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | |0.19| | | |
| |0.13| | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | |0.63| | | |0.80| | | | | |
| |0.82| | | |0.85| | | |0.95| |0.85| |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | |0.37| | | |0.20| | | | | |
| |0.18| | | | | | | | | | | |
| | | | | |0.15| | | | | |0.15| |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |0.15| | | |0.15| |0.21| | | |0.09| |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| |0.20| |0.65| |0.34| | | |0.50| |0.47| |
| | | | | | | |0.20| | | | | |
| | | | | | | | | | | | | |
| |0.15| | | | | | | | | | | |
| | | | | | | |0.28| | | | | |
| | | | | |0.19| | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | |0.08| | | |
| | | | | | | | | | | | | |
| |0.31| | | | | | | | | |0.13| |
| | | | | | | | | |0.09| | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | |0.13| | | | | | | |
| | | | | | | |0.18| | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | |0.33| | | | | |0.28| |0.25| |
| |0.18| | | |0.18| | | | | | | |
| | | | | | | |0.12| | | | | |

1.0

1.0

1.0

0.8

0.8

0.8

Shareofpredictions

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

Gemini-3.1-ProQwen3-OmniMing-Omni-2.0Nemotron-3-OmniMiniCPM-o-4.5MiMo-V2.5

Gemini-3.1-ProQwen3-OmniMing-Omni-2.0Nemotron-3-OmniMiniCPM-o-4.5MiMo-V2.5

Gemini-3.1-ProQwen3-OmniMing-Omni-2.0Nemotron-3-OmniMiniCPM-o-4.5MiMo-V2.5

Correct (muted) Hallucinated synced

Hallucinated shift Other

Correct (mismatched) Hallucinated synced

Hallucinated shift Other

Correct synced Correct shift Wrong direction

Missed shift False shift

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 4: Prediction breakdown per model on the three intervention tasks. Errors cluster around a synced default, evidencing shortcut reliance over genuine audio-video alignment.

- Table 2: Accuracy (%) under different alignment recipes on temporal synchronization, general video and audio-visual understanding benchmarks. We evaluate temporal grounding on Sync and VGGSync, video understanding on V-MME and LVB, audio-visual understanding on WS and DO. Avg. is the six-benchmark average. All DPO recipes are initialized from the SFT w/ OP checkpoint.

Recipe Sync VGGSync V-MME LVB WS DO Avg. Qwen3-Omni-30B 34.3 36.8 69.2 49.1 50.3 68.2 51.3 SFT w/ OP 73.9 – – – – – – SFT w/ CTP + FV-D + FV-AL 76.1 46.7 43.8 40.8 48.2 66.9 53.8 DPO w/ SP 75.4 55.7 69.3 50.9 49.8 69.0 61.7 DPO w/ OP + SP 76.5 56.4 69.9 47.7 49.7 68.5 61.5 DPO w/ SP + FV-D 82.2 55.4 69.1 51.5 49.8 68.0 62.7 DPO w/ OP + FV-D + LV-MCQA 83.0 56.6 69.2 50.4 49.9 67.6 62.8 DPO w/ CTP + FV-D 81.2 55.8 69.6 51.4 49.5 68.0 62.6 DPO w/ CTP + FV-D + LV-MCQA 82.2 55.7 69.2 51.1 49.8 67.8 62.6 DPO w/ CTP + FV-D + FV-A 82.6 55.9 69.1 50.8 49.9 67.3 62.6 Ours 83.1 56.4 70.1 52.1 50.3 67.9 63.3

OP: initial original-sync preference data; SP: SFT-policy negatives; CTP: counterfactual temporal preferences; FV-* and LV-MCQA denote general video preference data.

22% of predictions to Wrong direction, showing partial sensitivity to offsets without reliable sign recovery. Errors are systematically biased toward the synced prior rather than randomly distributed, indicating that current models rely on shortcut consistency rather than genuine cross-modal alignment.

#### 3.3 Targeted Alignment Improves Temporal Grounding Without Alignment Tax

###### VGGSoundSync accuracy across difficulty bands (harder = smaller offset; synced = no shift)

We next ask whether targeted intervention training can improve temporal grounding without hurting general capabilities. Starting from Qwen3Omni-30B, we compare alignment recipes using original synchronization preferences, self-sampled negatives, counterfactual temporal preferences, and general video preferences. Ours denotes our final 10K DPO recipe combining CTP, FV-D, and FV-A-L. Appx. §A.3 details each data source, including its construction, preference format, and intended training signal.

Qwen3-Omni (vanilla) MiniCPM-o 4.5 Gemini-3.1 Pro-preview Ours

| | | |desynced harder<br><br>| | | | | |
|---|---|---|---|---|---|---|---|---|
| |81<br><br>89| | | | | | | |
| | | |73<br><br>67<br><br>60 60| | | | | |
| |43| |51| | | | | |
| |40| |28<br><br>24 24| | | | | |
| | | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>12<br><br>14<br><br>10 10 9 10<br><br>7 8| | | | | |
| | | | | | | | | |

1.0

0.8

3-classaccuracy

0.6

0.4

0.2

0.0

Synced Very easy (±1.6 s)

Easy (±1.2 s)

Medium (±0.8 s)

Hard (±0.4 s)

Figure 5: Difficulty-band robustness. Smaller offsets are harder; our model remains robust while baselines collapse under desynchronization.

Tab. 2 shows that alignment training substantially improves temporal synchronization over the vanilla Qwen3Omni baseline. Our best 10K mixture improves Sync from 34.3% to 83.1% and VGGSync from 36.8% to 56.4%, suggesting that the model gains transferable temporal grounding rather than simply memorizing our intervention format. At the same time, it maintains or improves V-MME, LVB, and WS, remains competitive on DO, and raises the six-benchmark average accuracy from 51.3% to 63.3%. The contrast with the SFT-only mixture, which improves Sync but sharply hurts general benchmarks, indicates that preference alignment rather than supervised mixing is key to improving temporal grounding without incurring an alignment tax.

The recipe ablation further clarifies which data sources are responsible for this tradeoff. SFT with intervention and general video data already improves Sync, but substantially degrades V-MME and LVB, indicating that supervised mixing alone can over-specialize the model to interventionstyle supervision. In contrast, DPO recipes recover general capability while preserving temporal gains. Self-sampled preferences provide a strong general baseline, but the best temporal results arise when targeted temporal preferences are combined with general video preference data. This

Localization quality: how often the predicted offset is close to ground truth

###### Audio visual synchronization accuracy

Qwen3-Omni (vanilla)

MiniCPM-o 4.5

Gemini-3.1 Pro-preview

Ours

| |
|---|

| |
|---|

| |
|---|

| |
|---|

binary (sync vs. desync)

3-class (synced / delay / early)

direction acc. (desync subset)

| |
|---|

| |
|---|

| |
|---|

###### Sync

VGGSoundSync

###### Sync

VGGSoundSync

0.40

Shareofevaluatedtestsamples

1.0

1.0

0.7

89.9

0.35

32.3%

84.5

83.1

58.2%

0.6

0.8

0.8

0.30

69.5

49.5%

65.1

64.8

0.5

0.25

Accuracy

Accuracy

56.6

0.6

0.6

54.3

49.3

0.4

0.20

46.7

46.5

43.7

17.1%

40.9

36.8

0.4

0.4

36.2

28.4%

35.0

0.3

34.3

0.15

33.4

32.5

30.8

11.7%

25.4

0.2

8.6%

0.10

8.4%

15.5%

0.2

0.2

6.9%

10.6 9.5

3.7%

0.1

0.05

1.4

0.9%

0.2%0.0% 1.4%0.0%

0.0

0.0

0.0

0.00

MiniCPM-o 4.5

MiniCPM-o 4.5

Qwen3-Omni (vanilla)

Qwen3-Omni (vanilla)

Ours

Ours

Gemini-3.1 Pro-preview

Gemini-3.1 Pro-preview

0.5 s 1.0 s

0.2 s 0.5 s

(a) Audio-visual synchronization accuracy.

(b) Localization quality under offset tolerance.

Figure 6: Complementary synchronization results. Left: model accuracy on binary synchronization, three-way temporal classification, and direction prediction. Right: the fraction of samples whose predicted offset is close to the ground-truth temporal displacement.

suggests that counterfactual temporal supervision supplies the grounding signal, while FineVideo and LLaVA-Video preferences regularize the model toward broad video understanding.

- Fig. 5 evaluates synchronization across temporal-offset difficulty bands on VGGSync, using the Shift intervention from §2.1. Each band corresponds to a different offset magnitude |∆|. The high synced accuracy of vanilla Qwen3-Omni and MiniCPM-o should be read together with Fig. 4: both models strongly prefer answering “synced,” making them appear accurate only when no shift is applied. Once any nonzero offset is introduced, their accuracy collapses across all bands, including large |∆| values that should be easy to detect. Gemini-3.1-Pro follows a more expected trend, performing better on larger shifts and degrading as |∆| becomes smaller and subtler. Our model remains stronger across all shifted bands while also reflecting the expected pattern that smaller |∆| is harder. This suggests that temporal grounding should be judged not by synced-video accuracy alone, but by whether models show difficulty-sensitive verification under controlled audio displacement.

OursQwen3-OmniGemini-3.1-ProMiniCPM-O-4.5Ming-Omni-2.0Nemotron-3-NanoMiMo-V2.5

0.0

0.2

0.4

0.6

0.8

1.0

Combinedaccuracy

0.739

0.475

0.565

0.595

0.750

0.352

0.504

Ours: rank 2 / 7

+26.4 pp vs Qwen3-Omni

Mute intervention + original

OursQwen3-OmniGemini-3.1-ProMiniCPM-O-4.5Ming-Omni-2.0Nemotron-3-NanoMiMo-V2.5

0.0

0.2

0.4

0.6

0.8

1.0

Combinedaccuracy

0.651

0.563 0.558

0.503

0.528 0.543 0.527

Ours: rank 1 / 7

+8.8 pp vs Qwen3-Omni

Swap intervention + original

Combined accuracy on Mute and Swap benchmarks

Figure 7: Beyond temporal synchronization. Combined Mute and Swap accuracy over original and intervened conditions.

- Fig. 6 separates temporal grounding into label-level synchronization detection and fine-grained offset localization. In Fig. 6a, our model consistently outperforms Gemini-3.1-Pro across all synchronization metrics, including binary synced/desynced classification, three-way temporal classification, and direction prediction on desynced videos. This suggests that the improvement is not limited to coarse mismatch detection, but extends to the harder problem of identifying the temporal direction of the mismatch. Fig. 6b further sharpens this distinction: most baselines rarely predict offsets close to the ground truth, whereas our model achieves the strongest localization coverage on Sync and remains competitive on VGGSync. Together, these results show that audio-visual grounding should not be measured only by whether a model flags desynchronization, but also by whether it can localize the temporal mismatch with meaningful precision.

- 3.4 Beyond Temporal Synchronization

- Fig. 7 evaluates whether the recipe in Tab. 2 can extend beyond temporal synchronization. Starting from our best recipe, we add a small amount of Mute/Swap SFT. The resulting model ranks first on Swap and second on Mute, yielding a 28% average gain over vanilla Qwen3-Omni across Shift, Mute, and Swap. Fig. 8 further separates intervention detection from false alarms on original controls, showing that the gain is not merely higher combined accuracy: our model moves closer to the ideal top-left tradeoff, especially on Swap. This suggests that intervention-based training can mitigate multiple shortcut modes, while audio existence and cross-modal consistency still require targeted supervision beyond temporal alignment alone.

### 4 Related Work

Detection vs false-alarm trade-off

Native Omni Models and CrossModal Shortcuts Recent frontier multimodal models are shifting from frame-centric video-language pipelines toward native multimodal or omni-modal processing, where video, audio, images, and text are handled through a unified interface or architecture [26, 33, 58]. Although such integration suggests stronger audio-visual grounding [21, 66, 70], it does not ensure that models verify the audio stream. The shortcut behavior we observe reflects a long-standing assumption in audio-visual representation learning: natural videos provide supervision because visual and acoustic events often co-occur in synchronized and semantically aligned ways [2, 11, 28, 39, 50]. While effective for learning shared representations, these co-occurrence signals can conflate genuine grounding with statistical association [40, 60, 69]. Models may therefore rely on visual-semantic shortcuts [1, 48]: barking dogs imply barks, falling objects imply impacts, and speaking faces imply speech. Without negative cases that break these correlations [51], models can appear grounded without checking whether sound is present, synchronized, or physically consistent, producing a Clever Hans effect [29] in modern audio-visual models [7, 64]. We address this gap using controlled audio interventions that test cross-modal verification under broken audio-visual correlations.

Mute top-left is best

Swap top-left is best

1.0

1.0

Detectionrateonintervention

Detectionrateonintervention

ideal

ideal

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

False-alarm rate on control (1 orig. accuracy)

False-alarm rate on control (1 orig. accuracy)

Ours

Gemini-3.1-Pro

Ming-Omni-2.0

MiMo-V2.5

Qwen3-Omni

MiniCPM-O-4.5

Nemotron-3-Nano

Figure 8: Intervention-control tradeoff. Top-left indicates strong intervention detection with few false alarms on original controls.

Ours: Direct prompt. Baselines: Neutral prompt (rejudged).

Preference Alignment for Video-Capable Multimodal Models Video-capable multimodal models have evolved along two related directions: video-language instruction tuning, which connects visual encoders with LLMs for video understanding [14, 35], and native omni-modal modeling, which integrates video, audio, images, and text within unified interfaces or architectures [26, 33, 58]. Preference-based methods such as Direct Preference Optimization [46] have also been adapted to video-language modeling, often using detailed captions or language-model feedback as proxies for video-grounded rewards [72]. However, existing alignment data mainly targets helpfulness [5, 42], visual question answering [36], instruction following [65], and safety [76, 78], with limited attention to how models use, ignore, or misattribute the audio stream. Recent work has also observed visual dominance and video-driven audio hallucination in audio-visual LLMs [6, 49]. Our work instead decomposes audio-visual grounding into temporal synchronization, audio existence, and cross-modal material consistency, and studies how intervention data and preference optimization affect each dimension.

### 5 Conclusion

This work shows that apparent audio understanding in video-capable multimodal models can be strongly vision-driven. We identify this behavior as an audio-visual Clever Hans effect, where models answer sound-related questions by exploiting natural visual-acoustic correlations rather than verifying the observed audio stream. To make this failure measurable, we introduce THUD, which uses Shift, Mute, and Swap interventions to probe temporal synchronization, sound existence, and audio-visual consistency. Our experiments reveal systematic shortcut reliance across current open and closed models. We further show that counterfactual intervention data can be used not only for diagnosis, but also for alignment: a two-stage recipe combining intervention-derived preferences with event-level general video preferences improves audio-visual grounding while preserving broad video understanding. Overall, our findings suggest that future video-capable models should be evaluated and trained under counterfactual audio-visual conditions, not only naturally correlated videos.

### References

- [1] Aishwarya Agrawal, Dhruv Batra, and Devi Parikh. Analyzing the behavior of visual question answering models. In Jian Su, Kevin Duh, and Xavier Carreras, editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1955–1960, Austin, Texas, November 2016. Association for Computational Linguistics.
- [2] Humam Alwassel, Dhruv Kumar Mahajan, Lorenzo Torresani, Bernard Ghanem, and Du Tran. Selfsupervised learning by cross-modal audio-video clustering. ArXiv, abs/1911.12667, 2019.
- [3] Relja Arandjelovi´c and Andrew Zisserman. Look, listen and learn. 2017 IEEE International Conference on Computer Vision (ICCV), pages 609–617, 2017.
- [4] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Thomas Henighan, Andy Jones, Nicholas Joseph, Benjamin Mann, Nova Dassarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, John Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. ArXiv, abs/2112.00861, 2021.
- [5] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova Dassarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Thomas Henighan, Nicholas Joseph, Saurav Kadavath, John Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, Benjamin Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. ArXiv, abs/2204.05862, 2022.
- [6] Ami Baid, Zihui Xue, and Kristen Grauman. Don’t let the video speak: Audio-contrastive preference optimization for audio-visual language models. arXiv preprint arXiv:2604.14129, 2026.
- [7] S. Buch, Cristobal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. Revisiting the “video” in video-language understanding. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2907–2917, 2022.
- [8] Rui Cai, Bangzheng Li, Xiaofei Wen, Muhao Chen, and Zhe Zhao. Diagnosing and mitigating modality interference in multimodal large language models. ArXiv, abs/2505.19616, 2025.
- [9] João Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4724–4733, 2017.
- [10] Honglie Chen, Weidi Xie, Triantafyllos Afouras, Arsha Nagrani, Andrea Vedaldi, and Andrew Zisserman. Audio-visual synchronization in the wild. In BMVC, 2021.
- [11] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 721–725, 2020.
- [12] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 4299–4307, 2017.
- [13] Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyu Sun, Yingjin Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, Jiancheng Gui, Luoyuan Zhang, Xian Sun, Fuwei Huang, Moye Chen, Zhuohang Lin, Hanyu Liu, Qi Gui, Qing-Yuan Han, Yuyang Wen, Huiping Liu, Rongkang Wang, Yaqi Zhang, HongRui Wei, Chi Chen, You Li, Kechen Fang, Jie Zhou, Yuxuan Li, Guoyang Zeng, Chaojun Xiao, Yankai Lin, Xu Han, Maosong Sun, Zhiyuan Liu, and Yuan Yao. Minicpm-o 4.5: Towards real-time full-duplex omni-modal interaction. CoRR, 2026.
- [14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.
- [15] Dave Epstein, Boyuan Chen, and Carl Vondrick. Oops! predicting unintentional action in video. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 916–926, 2019.

- [16] Miquel Farré, Andi Marafioti, Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. Finevideo. https: //huggingface.co/datasets/HuggingFaceFV/finevideo, 2024.
- [17] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 24108–24118. Computer Vision Foundation / IEEE, 2025.
- [18] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. ArXiv, abs/2404.12390, 2024.
- [19] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard S. Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nat. Mach. Intell., 2(11):665– 673, 2020.
- [20] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780, 2017.
- [21] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind one embedding space to bind them all. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180–15190, 2023.
- [22] Google DeepMind. Gemini 3. https://deepmind.google/models/gemini/, 2026.
- [23] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 6325–6334. IEEE Computer Society, 2017.
- [24] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 14375–14385. IEEE, 2024.
- [25] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. CoRR, abs/2502.04326, 2025.
- [26] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [27] Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046, 2023.
- [28] Bruno Korbar, Du Tran, and Lorenzo Torresani. Cooperative learning of audio and video models from self-supervised synchronization. In Neural Information Processing Systems, 2018.
- [29] Sebastian Lapuschkin, Stephan Wäldchen, Alexander Binder, Grégoire Montavon, Wojciech Samek, and Klaus-Robert Müller. Unmasking clever hans predictors and assessing what machines really learn. Nature Communications, 10, 2019.
- [30] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wen Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: chat-centric video understanding. Science China Information Sciences, 68, 2023.
- [31] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Lou, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22195–22206, 2024.

- [32] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Lou, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 22195–22206. IEEE, 2024.
- [33] Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv preprint arXiv:2501.15368, 2025.
- [34] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 292–305. Association for Computational Linguistics, 2023.
- [35] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 5971–5984. Association for Computational Linguistics, 2024.
- [36] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc., 2023.
- [37] Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 12585–

12602. Association for Computational Linguistics, 2024.

- [38] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.
- [39] Pedro Morgado, Nuno Vasconcelos, and Ishan Misra. Audio-visual instance discrimination with crossmodal agreement. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12470–12481, 2020.
- [40] Pedro Miguel Morgado, Ishan Misra, and Nuno Vasconcelos. Robust audio-visual instance discrimination. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12929–12940, 2021.
- [41] OpenAI. Openai GPT-5 system card. CoRR, abs/2601.03267, 2026.
- [42] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.
- [43] Andrew Owens and Alexei A. Efros. Audio-visual scene analysis with self-supervised multisensory features. In European Conference on Computer Vision, 2018.
- [44] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens Continente, Larisa Markeeva, Dylan Sunil Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and Joao Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.
- [45] Oskar Pfungst. Clever Hans:(the horse of Mr. Von Osten.) a contribution to experimental animal and human psychology. Holt, Rinehart and Winston, 1911.
- [46] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

- [47] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14313–14323, 2024.
- [48] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4035–4045, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.
- [49] Ramaneswaran Selvakumar, Kaousheik Jayakumar, S Sakshi, Sreyan Ghosh, Ruohan Gao, and Dinesh Manocha. Do audio-visual large language models really see and hear? arXiv preprint arXiv:2604.02605, 2026.
- [50] Arda Senocak, Tae-Hyun Oh, Junsik Kim, Ming-Hsuan Yang, and In-So Kweon. Learning to localize sound source in visual scenes. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4358–4366, 2018.
- [51] Nikhil Singh, Chih-Wei Wu, Iroro Orife, and Mahdi M. Kalayeh. Looking similar, sounding different: Leveraging counterfactual cross-modal pairs for audiovisual representation learning. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26897–26908, 2023.
- [52] Kim Sung-Bin, Oh Hyun-Bin, JungMok Lee, Arda Senocak, Joon Son Chung, and Tae-Hyun Oh. AVHBench: A cross-modal hallucination benchmark for audio-visual large language models. In The Thirteenth International Conference on Learning Representations, 2025.
- [53] Inclusion AI Team. Ming-omni: A unified multimodal model for perception and generation. CoRR, abs/2506.09344, 2025.
- [54] InternVL Team. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. CoRR, abs/2504.10479, 2025.
- [55] Nemotron 3 Nano Omni Team. Nemotron 3 nano omni: Efficient and open multimodal intelligence. 2026.
- [56] Qwen Team. Qwen3-omni technical report. CoRR, abs/2509.17765, 2025.
- [57] Qwen Team. Qwen3-vl technical report. CoRR, abs/2511.21631, 2025.
- [58] Qwen Team. Qwen3.5-omni technical report, 2026.
- [59] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 5228–5238. IEEE, 2022.
- [60] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5228–5238, 2022.
- [61] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 9568–9578. IEEE, 2024.
- [62] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark. CoRR, abs/2406.08035, 2024.
- [63] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali Wang, and Limin Wang. Internvideo2: Scaling video foundation models for multimodal video understanding. ArXiv, abs/2403.15377, 2024.
- [64] Yuxuan Wang, Yueqian Wang, Dongyan Zhao, Cihang Xie, and Zilong Zheng. Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models. ArXiv, abs/2406.16338, 2024.
- [65] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022.

- [66] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. NExt-GPT: Any-to-any multimodal LLM, 2024.
- [67] Xiaomi MiMo Team. Xiaomi mimo-v2.5: A leap in agency and multimodality. https://mimo.xiaomi. com/mimo-v2-5/, April 2026. Accessed: 2026-05-04.
- [68] Mert Yüksekgönül, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
- [69] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Y. Zou. When and why vision-language models behave like bags-of-words, and what to do about it? ArXiv, abs/2210.01936, 2022.
- [70] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. Anygpt: Unified multimodal llm with discrete sequence modeling. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9637–9662, 2024.
- [71] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. In Yansong Feng and Els Lefever, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 543–553, Singapore, December 2023. Association for Computational Linguistics.
- [72] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander G Hauptmann, Yonatan Bisk, and Yiming Yang. Direct preference optimization of video large multimodal models from language model reward. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 694–717, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics.
- [73] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024.
- [74] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. Trans. Mach. Learn. Res., 2025, 2025.
- [75] Ziwei Zhou, Rui Wang, and Zuxuan Wu. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities. CoRR, abs/2505.17862, 2025.
- [76] Boyu Zhu, Xiaofei Wen, Wenjie Jacky Mo, Tinghui Zhu, Yanan Xie, Peng Qi, and Muhao Chen. Omniguard: Unified omni-modal guardrails with deliberate reasoning. ArXiv, abs/2512.02306, 2025.
- [77] Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul F. Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. CoRR, abs/1909.08593, 2019.
- [78] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy M. Hospedales. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. ArXiv, abs/2402.02207, 2024.

###### Source Videos

###### Annotation and verification

###### Physical interventions

Salient acoustic consequences

Break natural audio-visual correlations

Event and timestamp annotations

[Figure 36]

Shift Mute Swap

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Visual event

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Visual time

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Audio event

early delay

[Figure 56]

[Figure 57]

[Figure 58]

Cross-model verification

[Figure 59]

[Figure 60]

Audio time

[Figure 61]

Sound Removal

Acoustic Substitution

Temporal Displacement

Human review

###### Example preference pair

###### Intervened Input (Shift)

[Figure 62]

###### Visual event (fall)

Chosen

[Figure 63]

t=5.0s

Visible fall at ~5.0s; impact sound at ~3.1s → mismatch.

0s 1s 2s 3s 4s 5s 6s 7s

Video

[Figure 64]

Rejected

Δ

The sound is synchronized with the fall.

t = 1.9s mismatch

0s 1s 2s 3s 4s 5s 6s 7s

[Figure 65]

[Figure 66]

[Figure 67]

Audio

Audio event (impact)

t=3.1s

- Figure 9: Pipeline for intervention data construction. We create Shift, Mute, and Swap variants from source videos with salient acoustic events, annotate visual/audio events and timestamps via cross-model verification with human review, and construct chosen–rejected preference pairs for training. The bottom panel shows a representative Shift example.

### A Schematic Overviews of Data Construction and Alignment

#### A.1 Data Construction Pipeline

- Fig. 9 illustrates the systematic pipeline for constructing the intervention-driven preference dataset. The process begins with initial event-time labeling using Gemini, which are then rigorously crossverified: visual timestamps are validated through a consensus of GPT and Claude via frame-unit analysis, while acoustic timestamps undergo human inspection to ensure ground-truth reliability. After filtering samples based on strict agreement criteria, we apply three interventions, Shift, Mute, and Swap, to the validated source videos. This results in the final preference pairs, where the "chosen" response reflects true audio-visual grounding and the "rejected" response exposes the visually-plausible shortcuts we aim to mitigate during alignment.

#### A.2 Intervention Summary

Tab. 3 summarizes the three interventions in THUD. Each intervention keeps the visual stream fixed while perturbing the audio track to target one grounding dimension: Shift probes temporal synchronization, Mute probes sound existence, and Swap probes source consistency. These controlled cases test whether models verify the observed audio or simply infer plausible sounds from visual priors.

#### A.3 Preference Data Sources

This section describes the preference data sources used in our alignment recipe study. Each preference example is represented as a pair of responses, where the chosen response provides the desired behavior and the rejected response provides a shortcut-prone or incorrect alternative.

Original synchronization preferences (OP). Original synchronization preferences are constructed from the annotated audio-visual event tuples introduced in §2.2. For each video, we represent the aligned visual and acoustic event as

zi = (evi ,tvi ,eai ,tai ), (9)

where evi and tvi denote the visual event and its timestamp, while eai and tai denote the corresponding acoustic event and timestamp. The chosen response is the annotated answer derived from the original

aligned event. The rejected response is produced by perturbing one or more components of zi, such as the visual event, visual timestamp, acoustic event, or acoustic timestamp, creating a plausible but incorrect synchronization explanation.

SFT-policy negatives (SP). Self-sampled negatives are generated from the SFT model itself. Given the same video-question input, we use the reference annotation as the chosen response and treat the SFT model’s incorrect or shortcut-prone output as the rejected response. This data source encourages the model to correct its own post-SFT failure modes.

Counterfactual temporal preferences (CTP). Counterfactual temporal preferences are constructed by pairing original and shifted videos. For an original video, the chosen response corresponds to the original synchronized condition, while the response describing the shifted condition is used as the rejected response. For a shifted video, this assignment is reversed: the shifted-condition answer is chosen, and the original synchronized answer is rejected. This forces the model to distinguish true temporal alignment from visually plausible but temporally inconsistent audio.

FineVideo descriptive preferences (FV-D). FV-D is derived from the FineVideo data described in Appx. §E. We use the description, localization, and attribution tasks, which encourage the model to produce faithful video descriptions, localize relevant events, and attribute answers to appropriate visual or audio evidence.

FineVideo audio-visual QA preferences (FV-AVQA). FV-AVQA corresponds to the audiodependent QA subset in Appx. §E. These examples ask questions that require audio evidence. Candidate questions and answers are generated by Gemini and then manually filtered. We further retain examples where GPT-based text-only answering fails, since GPT does not receive the audio stream. This filtering emphasizes cases where the answer cannot be reliably inferred from visual or textual priors alone.

FineVideo audio-visual QA long-form preferences (FV-AVQA-L). FV-AVQA-L is a long-form version of FV-AVQA. Instead of only selecting an answer option, the chosen response includes both the answer and an explanation grounded in the audio-visual evidence. This data source encourages the model to justify audio-dependent answers rather than relying on short-form guesses.

LLaVA-Video multiple-choice QA (LV-MCQA). LV-MCQA is a multiple-choice video QA dataset titled LLaVA-Video-178K [73]. We include it as a general video preference source to regularize the model toward broad video understanding and reduce over-specialization to interventionstyle examples.

#### A.4 Alignment Pipeline

- Fig. 10 illustrates our two-stage post-training pipeline designed to detect Shift, Mute, and Swap failures while preserving general video understanding. The training process integrates our targeted intervention dataset and re-annotated general video instructions derived from FineVideo [16]. In Stage 1, an SFT warm-up on intervention data establishes basic audio-aware patterns. Stage 2 applies DPO using a mixture of intervention preference pairs and general video data. The preference pairs teach the model to reject visually plausible shortcuts, while the general data acts as a regularizer to preserve broad multimodal capabilities.

### B Annotation and Verification Details

Video-to-frame-unit conversion. For visual verification with GPT and Claude, we convert each video into N temporally ordered frame units. Given a video of duration T seconds, we split it into

Preserve general ability

[Figure 68]

[Figure 69]

Intervention data

[Figure 70]

###### Preference-tuned Omni Model

General video data

[Figure 71]

Shift/mute/swap Counterfactual AV Clips

Temporal Synchronization

###### Finevideo + AVQA Dense captions / Event

[Figure 72]

[Figure 73]

Events align with audio in time

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Temporal shifts

Audio mutes

Audio swaps

[Figure 80]

[Figure 81]

Audio Existence

Event localization

Dense captions

Preference optimization

Diverse video clips

Audio presence matches visuals

[Figure 82]

[Figure 83]

SFT warm-up

Mixed with general video data

| | |
|---|---|
| | |

[Figure 84]

###### Intervention pairs

Audio-aware initialization

###### Chosen vs. Rejected Audio verified vs. shortcuts

[Figure 85]

[Figure 86]

Learn base AV alignment

[Figure 87]

Sound Consistency

Pairwise preference learning

[Figure 88]

Stabilize multimodal input

[Figure 89]

[Figure 90]

Sound match visual events

[Figure 91]

[Figure 92]

Initialize from pertained model

Audio veriﬁcation

Counterfactual construction

Hard Negative mining

- Figure 10: Two-stage intervention-driven alignment pipeline. Counterfactual intervention data is first used for SFT warm-up, and intervention preference pairs are then mixed with general video data during preference optimization. This design encourages audio-verified responses while preserving general video understanding.

Table 3: Summary of our three physical interventions. Each intervention breaks a different natural audio-visual correlation and poses a diagnostic question about audio-visual grounding.

Intervention Operation Broken correlation Diagnostic question Shift a1:T → a+∆1:T temporal synchrony Is the sound synchronized? Mute a1:T → ∅ sound existence Is any sound present? Swap a1:T → a′1:T source consistency Does the sound match the event?

non-overlapping windows

uj = [sj,ej], j = 1,...,N,

where sj and ej denote the start and end time of the j-th unit. From each unit, we sample representative frames and present them in temporal order, together with the timestamp range of the unit. The verifier is asked to select the unit that contains the target visual event and optionally refine the timestamp within that unit. This frame-unit format allows models without direct video ingestion to perform temporal localization over visual evidence.

Gemini annotation prompt. We use Gemini to produce the initial audio-visual event annotation. The prompt is designed to avoid generic captioning and instead force event-level localization:

You are given a video with audio. Identify the most salient visible event that has a corresponding acoustic consequence. Return a JSON object with the following fields: visual_event: a short description of the visible event. visual_time: the timestamp in seconds when the visible event occurs. audio_event: a short description of the corresponding sound. audio_time: the timestamp in seconds when the sound occurs. confidence: high / medium / low. If the visual event or audio event cannot be localized reliably, return uncertain.

Frame-unit visual verification prompt. For GPT and Claude, we provide the ordered frame units and the candidate visual event proposed by Gemini:

You are given temporally ordered frame units from a video. Each unit contains representative frames and a timestamp range. Target visual event: [visual_event]. Select the frame unit where this event occurs. Return: unit_id, timestamp_range, and a one-sentence justification. If the event is not visible or cannot be localized, return uncertain.

Agreement and filtering rules. We retain a sample only if it satisfies the following conditions:

- 1. Visual agreement: Gemini, GPT, and Claude localize the visual event within ϵv = 0.8 seconds, or select overlapping frame units.
- 2. Audio verification: the acoustic event is audible and its timestamp can be verified by human inspection within ϵa = 0.5 seconds of the Gemini prediction.
- 3. Event clarity: the visual event has a clear onset or peak moment, such as impact, fall, collision, breakage, or contact.
- 4. Acoustic salience: the corresponding sound is not dominated by unrelated background music, speech, or noise.
- 5. Intervention validity: after applying Shift, Mute, or Swap, the correct answer remains unambiguous.

Manual review protocol. Samples failing automatic agreement are manually reviewed. We use the following decision rules. If the disagreement is due to a small boundary ambiguity, we correct the timestamp to the clearest event onset. If the visual event is partially occluded, spread over a long interval, or lacks a well-defined moment, we discard the sample. If the sound is too weak, masked by background noise, or not clearly tied to the visual event, we discard the sample. For Swap, we additionally discard cases where the substituted audio is trivially unrelated or too similar to the original audio; retained swaps must be acoustically plausible but inconsistent with the visible event.

### C Experimental Configuration

This section provides additional implementation details for our supervised fine-tuning (SFT), preference optimization, and evaluation experiments. All training experiments are conducted on 8*NVIDIA H200 GPUs, while evaluation experiments are conducted on either 8*NVIDIA H200 or 8*NVIDIA H100 GPUs. A single SFT run takes approximately 6 hours, and DPO training on 10K examples takes approximately 20 hours. For evaluation, the average inference time across the six datasets is approximately 5 hours per dataset.

Base model. We use Qwen3-Omni-30B-A3B-Instruct as the base omni-modal model. Video inputs are processed with audio enabled by setting use_audio_in_video=true.

Training configuration. We summarize the training configurations for supervised fine-tuning and preference optimization in Tab. 4. Both stages are launched with torchrun on a single node with 8 GPUs, using DeepSpeed ZeRO-3 for memory-efficient distributed training.

### D Preference Pair Examples

Examples of Preference Pair Construction

Shift Chosen: The visible fall occurs at ∼ 5.0s, while the impact sound is heard at ∼ 3.1s, indicating a synchronization mismatch. Rejected: The sound is synchronized with the fall.

Mute Chosen: The audio track is silent throughout the clip; no music, speech, ambient

noise, or sound effects are detected.

Rejected: The animated music-video scene is described as containing female vocals, a chaotic electronic rock beat, rain, thunder, TV static, and glass-shattering effects.

Swap Chosen: The visuals show an optics diffraction demonstration, but the audio describes how to use a centrifuge, indicating an audio-source mismatch. Rejected: The narrator explains the diffraction setup as the hand shines a smartphone light through the pen tip.

Table 4: Training configurations for the supervised fine-tuning and DPO stages.

Configuration SFT DPO

Initialization Qwen3-Omni-30B-A3B-Instruct SFT checkpoint Fine-tuning type Full-parameter tuning LoRA Epochs 3 1

Learning rate 2 × 10−6 1 × 10−6 Scheduler / warmup Cosine / 0.03 Cosine / 0.03 Weight decay / grad norm 0.01 / 1.0 0.0 / 1.0 Precision bf16 bf16

Cutoff length 131,072 131,072 Video max pixels 501,760 250,880 Audio in video Enabled Enabled

Batch size 1 per GPU; accum. 4; effective 32 1 per GPU; accum. 8; effective 64 Memory optimization DeepSpeed ZeRO-3 DeepSpeed ZeRO-3

Preference loss – Sigmoid DPO, β = 0.1 LoRA setting – rank 32; alpha 64; dropout 0.05

Workers 16 preprocessing; 8 dataloader 16 preprocessing; 8 dataloader Distributed setup 8 GPUs, single node 8 GPUs, single node Hardware H200 GPUs H200 GPUs

### E FineVideo-derived general instruction data

Description Describe visible events and audible cues.

Localization Locate visual/audio events in time.

Audio-dependent QA Answer questions that require audio evidence.

Attribution Infer the source or material of a cue.

### F Qualitative GPT-5.5 Outputs (Visual-Only Input)

We provide representative raw outputs of GPT-5.5 across the three tasks of our test data—Mute, Shift, and Swap—when the model is given visual frames only. Since the model does not have access to the audio track, these examples illustrate how it tends to respond when forced to reason about audio without being able to hear it.

###### Task: Mute — What does the audio sound like?

video: Bats & Balls Fail Compilation 47.mp4

#### GPT-5.5 raw output:

“The audio is likely from an indoor gym: basketball dribbles and sneaker squeaks as someone runs toward the hoop, followed by the thud/clank of a dunk attempt. Then there’s a sudden loud crash as the backboard shatters, with glass raining down onto the floor. You can hear startled reactions—shouting, laughter, and movement—as people react and run over afterward.”

###### Task: Shift — Is the audio in sync with the video?

video: Are You Serious! - Throwback Thursday Sept 2017 11_delay_2.39s.mp4

#### GPT-5.5 raw output:

“The audio and video appear to be synchronized.

The clearest sync points are the physical impact moments: the man’s jump/trip over the boards, the body hitting the ground, and the subsequent movement/laughter reaction. The impact sound lines up with the visual fall closely enough that I do not perceive a meaningful lead or lag. Any offset, if present, seems very small—likely under about 0.1–0.2 seconds—and not enough to call the tracks out of sync.”

###### Task: Swap — Does the audio match the video content?

video: Bats & Balls Fail Compilation 47.mp4 audio swapped from: FailArmy After Dark - That’s Going To Cost You Ep. 10 (103.mp4)

GPT-5.5 raw output: “yes”

### G Evaluation Prompts

This section documents the exact prompts used to elicit model responses on each of the three tasks, together with the GPT-based judge prompts used to parse those free-form responses into a structured prediction. For MUTE and SWAP, the numbers reported in the main text correspond to the neutral prompt setting, in which the model is asked an open-ended description question rather than being directly cued about the hypothesis under test. For SHIFT, a single structured prompt is used, which simultaneously asks the model to make an aligned/misaligned decision and estimate the offset.

#### G.1 Average Gap Calculation

We summarize shortcut reliance by the average accuracy drop from each non-intervened control to its paired counterfactual condition:

1 |D| d∈D

(AccOrig,d − AccInterv,d), D = {Sync,Exist.,Consist.}. (10)

∆shortcut =

Larger values indicate a larger performance collapse under counterfactual interventions. We report the gap only when all three dimensions are available. For free-form outputs, GPT-5.4 is used as an LLM judge to adjudicate predicted labels.

#### G.2 Inference Prompts

MUTE — Inference Prompt

“Describe the audio you hear in this video.”

SWAP — Inference Prompt

“Describe what you see in the video and what you hear in the audio.”

SHIFT — Inference Prompt

“Watch this video and listen to its audio carefully. Determine whether the audio and video tracks are synchronized. If they are not synchronized, identify the direction of the offset (audio delayed or audio early relative to video) and estimate the offset in seconds. Explain your reasoning.”

#### G.3 GPT Judge Prompts

Because the inference prompts above elicit free-form text, we use GPT as a structured-output extractor to convert each response into a discrete prediction. The judge is prompted with the system message

below and given the model’s raw response as the user message; it is required to return a single JSON object.

###### MUTE — Judge System Prompt

“You are evaluating a model’s response about a video whose audio has been secretly replaced with silence. The user will give you the model’s response. Classify the model’s assessment into exactly ONE of four categories and return ONLY valid JSON (no markdown, no explanation):

{"prediction": "synced"|"delay"|"early"|"muted", "explanation": "<one sentence>"} Rules:

- •“synced”: the model claims audio and video are synchronized (it is hallucinating audio).
- •“delay”: the model claims audio is delayed relative to video (hallucinating audio + claiming offset).
- •“early”: the model claims audio is early relative to video (hallucinating audio + claiming offset).
- •“muted”: the model states there is no audio, the audio is silent, muted, or absent.
- • If the model describes specific sounds (e.g. “thud”, “music”, “splash”) but does not explicitly say the video is silent, classify based on its sync/desync conclusion, not as “muted”.
- • If the model is ambiguous or gives no clear answer, default to “synced”.”

###### SWAP — Judge System Prompt

“You are evaluating a model’s response about a video whose audio has been secretly replaced with audio from a DIFFERENT unrelated video. The user will give you the model’s response. Classify the model’s assessment into exactly ONE of four categories and return ONLY valid JSON (no markdown, no explanation):

{"prediction": "synced"|"delay"|"early"|"mismatched", "explanation": "<one sentence>"} Rules:

- •“mismatched”: the model says the audio does NOT match the visual content, or the audio and video are unrelated, or the sounds don’t correspond to what’s shown.
- •“synced”: the model claims audio and video are well synchronized and matching.
- •“delay”: the model claims audio is delayed (but still believes they match semantically).
- •“early”: the model claims audio is early (but still believes they match semantically).
- • Prefer “mismatched” whenever the model expresses confusion, doubt about the audio matching the video, or says the audio seems unrelated.
- • Default to “synced” if the answer is ambiguous.”

###### SHIFT — Judge System Prompt

“You are a structured-output extractor. The user will give you a model’s free-text response about audio-video synchronization. Extract the following fields and return ONLY valid JSON (no markdown, no explanation):

{"synced": <bool>, "direction": "none"|"delay"|"early", "offset_sec": <float>, "t_v": <float or null>, "t_a": <float or null>, "explanation": "<one sentence>"} Rules:

- • synced: true if the model says audio and video are synchronized, false otherwise.
- • direction: “delay” means audio comes AFTER the visual event; “early” means audio comes BEFORE the visual event; “none” if synced is true.
- • offset_sec: estimated time gap in seconds. 0.0 if synced.
- • t_v: the timestamp (in seconds) the model attributes to the VISUAL event. null if not mentioned.
- • t_a: the timestamp (in seconds) the model attributes to the AUDIO event. null if not mentioned.
- • If you cannot determine a field, use the default (true / “none” / 0.0 / null / “”).”

### H Failure-mode definitions

This appendix provides the detailed definition and measurement protocol for each of the eight failure modes reported in Fig. 3. All rates lie in [0,1]; higher values indicate more frequent failures. Freeform (neutral-prompt) responses are classified by an independent OpenAI GPT-5.4 judge so that the mute/swap modes do not depend on a model self-reporting its own confusion.

- A. Audio Hallucination. Errors in which the model invents or accepts audio content that is incompatible with the input.

Mute Hallucination. On videos whose audio track has been replaced with silence, the prompt is

“Describe the audio you hear in this video.” The judge classifies each response into muted / audio_described / visual_only, and Mute Hallucination is the audio_described rate: the fraction of responses in which the model produces any concrete description of audio content (speech, music, ambient noise, impacts) instead of reporting silence.

Swap False-Match. On videos whose audio track has been replaced with the soundtrack of an unrelated video, the prompt is “Describe what you see in the video and what you hear in the audio.” Swap False-Match is the fraction of responses in which the judge concludes the model treated the (mismatched) audio as a plausible natural match for the visuals.

- B. Audio Denial. Symmetric errors on naturally paired (un-intervened) videos.

False Silence. On videos with their original audio, the same neutral mute prompt is used. False Silence is the fraction of responses in which the model claims silence or “no audible content” despite real audio being present.

Swap False-Mismatch. On videos with their original audio, the same neutral swap prompt is used. Swap False-Mismatch is the fraction of responses in which the model spuriously claims an audio–visual mismatch on a naturally synchronized pair.

- C. Question Avoidance.

Audio Dodge. Fraction of responses to the neutral mute prompt in which the model produces a visual-only description and never engages with the audio question (neither describes any sound nor claims silence). We report the mean of this rate across the intervention (silenced) and control (real audio) conditions because non-engagement is a property of the model, not of whether audio is real.

- D. Temporal Failures (sync task). The sync benchmark contains synced originals together with two intervention variants per video: an audio-delayed copy (DELAY) and an audio-advanced copy (EARLY), both with offsets of approximately ±2s. Each model’s free-form response is parsed by the judge into a boolean pred_synced and a categorical pred_direction ∈ {DELAY, EARLY, NONE}.

Offset Blindness. Fraction of desync samples (delay or early) for which the model judges the clip

to be synced. This isolates pure failure to perceive temporal misalignment.

Direction Confusion. Among the desync samples on which the model correctly judges the clip to be non-synced, the fraction for which it picks the wrong direction (calls a delay an early or vice versa). This isolates direction-of-offset perception from offset detection itself.

False Sync Alarm. Fraction of synced original samples for which the model claims the clip is desynced. Symmetric counterpart to Offset Blindness: false alarms on naturally aligned audio.

For Gemini-3.1-pro, per-row sync predictions were not retained, so its Offset Blindness, Direction Confusion, and False Sync Alarm are derived from the aggregate sync_desync_accuracy, direction_accuracy_on_desync, and per-category accuracies in the saved metrics.json, which are mathematically equivalent to the per-row counts.

### I Limitations.

Our training recipe is currently evaluated on a limited set of base models, so its effectiveness across broader omni-modal model families remains to be further studied. In addition, our recipe experiments primarily validate the effect of applying DPO after SFT for improving temporal synchronization. We have not yet conducted a complete training study for the Mute and Swap settings, which probe audio existence and cross-modal consistency. Extending the recipe to these intervention types is an important direction for future work.

### J Ethics and Broader Impacts

- J.1 Ethics.

Our research follows the NeurIPS Code of Ethics. The study is designed as a diagnostic and alignment analysis of audio-visual grounding in multimodal models, and does not involve humansubject experiments, crowdsourcing, or the collection of personally identifiable information. The video data used in our experiments comes from public or properly licensed sources, and we use the data only for model evaluation and training under controlled audio-visual interventions. Our analysis does not perform face recognition, identity inference, biometric classification, or any other person-level profiling.

- J.2 Broader Impacts

Positive impacts. This work aims to improve the reliability of video-capable multimodal and omnimodal models by revealing when they rely on visual-semantic shortcuts rather than genuine audiovisual verification. By introducing controlled Mute, Swap, and Shift interventions, our evaluation can help researchers diagnose pseudo-alignment and develop models that more faithfully check whether audio is present, synchronized, and consistent with the visual scene. Such improvements may benefit downstream applications where audio-visual grounding is important, including assistive technologies, video understanding, human-computer interaction, and safety-critical multimodal monitoring.

Potential risks and mitigations. The main risk is that intervention-based diagnostics could be used to construct adversarial examples or to optimize models specifically for benchmark performance rather than robust real-world grounding. In addition, improved audio-visual verification does not eliminate all hallucination risks, and deployed systems may still fail under out-of-distribution sounds, noisy environments, edited videos, or subtle cross-modal inconsistencies. To mitigate these risks, we frame our benchmark as a diagnostic tool rather than a deployment guarantee, report limitations of the evaluated settings, and encourage evaluating models under diverse interventions instead of relying on aggregate accuracy alone. If releasing data or code, we will document intended use, licenses, and limitations, and avoid releasing sensitive or personally identifying content.

### K New Assets

We introduce intervention-based evaluation assets for probing audio-visual grounding under three controlled settings: Mute, Swap, and Shift. These assets are intended for diagnostic evaluation, testing whether multimodal models verify the audio stream rather than relying on visual-semantic shortcuts.

For each selected video, we construct intervention variants by modifying only the audio stream. Mute removes the audio track to test audio-existence verification. Swap replaces the original audio with mismatched audio to test cross-modal consistency. Shift temporally displaces the audio to test synchronization and offset reasoning. Each example is paired with its intervention type, evaluation condition, and target label.

The assets are verified by members of the research team to ensure that the intervention is valid and the label is unambiguous. We reject examples with unclear visual events, corrupted or inaudible audio, failed interventions, or ambiguous labels. The assets are used only for model evaluation and alignment research, and are not intended as a guarantee of real-world robustness. Their limitations

###### include restricted intervention coverage, possible residual annotation noise, and limited coverage of all real-world audio-visual failure modes.

