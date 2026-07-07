# arXiv:2606.14702v2[cs.CV]17Jun2026

OmniVideo-100K: A Dataset for Audio-Visual Reasoning through Structured Scripts and Evidence Chains

Xinyue Cai1, Chaoyou Fu1∗, Yi-Fan Zhang2, Ran He2, Caifeng Shan1 1Nanjing University 2CASIA yzlmhzz@smail.nju.edu.cn, cyfu@nju.edu.cn

ABSTRACT

Current automated pipelines for audio-visual Question Answering (QA) generally adopt a “video-caption-QA” paradigm. However, these methods typically segment videos into short clips and generate separate descriptions for audio and visual modalities. This decoupled processing severs inherent associations between sounds and their visual sources, while independent clip processing often causes inconsistent descriptions of the same entity across segments. Furthermore, coupling long-text comprehension and QA synthesis into a single step often restricts models to localized events, yielding questions lacking long-term temporal connections and deep cross-modal reasoning. To address these issues, we propose an automated data engine featuring two mechanisms: (1) Entity-Anchored Video Scripting transforms videos into structured scripts, comprising summaries, main entity lists, and segment-wise audio-visual descriptions. The entity list serves as a global prior to ensure cross-segment referential consistency and reconstruct audio-visual associations. (2) Clue-Guided QA Generation prompts models to first mine cross-segment, multimodal clues from the script, and subsequently generate QA pairs based on these high-value clues. Leveraging this pipeline, we construct the instruction-tuning dataset OmniVideo-100K and a human-verified test set, OmniVideo-Test. Fine-tuning VITA-1.5, Qwen2.5-Omni-7B and Qwen3Omni-30B on OmniVideo-100K yields performance gains of up to 20.59% on OmniVideo-Test, demonstrating strong generalization (up to 12.64% improvements) across established benchmarks like Daily-Omni and JointAVBench. Code and data are available at https://github.com/MiG-NJU/OmniVideo-100K.

1 INTRODUCTION

In recent years, Multimodal Large Language Models (MLLMs) have evolved from vision-language to omni-modal. Specifically, in the domain of video understanding, integrating the audio modality enables the joint analysis of audio-visual cues, thereby facilitating a deeper comprehension of video content (Li et al., 2025b; Xu et al., 2025b; Tang et al., 2025; Ye et al., 2025b; Geng et al., 2025a). This advanced comprehension goes beyond merely processing visual and audio modalities in isolation; instead, it requires effectively modeling complex cross-modal interactions.

Achieving such cross-modal capability relies heavily on the availability of high-quality audio-visual data. Recent instruction-tuning datasets (Ye et al., 2024; Liu et al., 2025) are often constructed via automated pipelines that leverage the advanced reasoning capabilities of Large Language Models (LLMs) to synthesize Question-Answer (QA) pairs from intermediate video captions. Considering MLLMs’ modality bias and hallucinations during joint audio-visual understanding (Sung-Bin et al., 2024), most methods (Geng et al., 2025b; Ye et al., 2025a; Kurpath et al., 2025) generate separate descriptions for visual and audio modalities, where videos are split into short segments to capture more visual details. Afterwards, QA pairs are directly generated from the long, detail-heavy descriptions. However, this paradigm exhibits three limitations: (1) It breaks the inherent audio-visual associations within videos (Seth et al., 2026); (2) Independent segment descriptions lead to incoherent narratives. For instance, a “person in white” might be described as a “person with yellow hair” in the next segment, making it hard for LLMs to track entities; (3) Coupling long-text comprehension

∗Corresponding author.

## Global Information

## Segment-wise Description

[Figure 1]

[Figure 2]

|Summary<br><br>A Brunette Girl visit a Pop Mart store …|
|---|

[speaker]: words & non-speech sound visual caption

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>Brunette Girl<br><br>[Figure 10]<br><br>people<br><br>[Figure 11]<br><br>A girl with short brown hair ....<br><br>Main Entity List<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>Pop Mart<br><br>[Figure 15]<br><br>object<br><br>[Figure 16]<br><br>Small collectible plastic toys …|
|---|

## ··· ···

···

03:34 – 03:44 07:17 – 07:31

[Figure 17]

[Figure 18]

[Figure 19]

###### (03:34-03:39) [Brunette Girl]:

(07:17-07:17) Popping sound (07:17-07:25) [Brunette Girl]: I love it.

I hope I get one with a coke

Reference Speaker

bottle instead of a can though. … right stands Brunette Girl, who has short brown hair … The scene cuts to a wider shot …

··· ···

[Figure 20]

[Figure 21]

###### … Brunette Girl standing …

Pop Mart with a brown bottle

labeled Coca-Cola …

[Figure 22]

|Summary …<br><br>Main Entity List<br><br>- Pop Mart: …<br>- Brunette Girl: … 00:00 – 00:06 Audio: … Visual: … …<br><br><br>|
|---|

In [03:34 - 03:44], Audio Clue: I

Ques: Why Brunette Girl

07:17 – 07:31 03:34 – 03:34

[Figure 23]

hope I get one with a coke bottle

happy with her blind box? Ans: It is holding a bottle instead of a can.

[Figure 24]

instead of a can … In [07:17 - 07:31], Audio Clue: I love it. Visual Clue: Pop Mart with a brown bottle labeled Coca-Cola …

Mine Clue LLM

Focus On LLM

Ques: Why Brunette Girl happy with her blind box?

Relevant Segments

A. It is holding a bottle

… explains why she is specifically thrilled with this

instead of a can. Ans: A

outcome.

Clues & Connections Open-ended / Multi-Choices

Entity-Anchored Script

- Figure 1: Overview of our pipeline. The pipeline first transforms audio-visual videos into entityanchored scripts (upper part), comprising a summary, a main entity list, and structured segment-wise descriptions that integrate speech, sounds, and visual information. By utilizing consistent identifiers (represented by orange and green text) from the main entity list, we ensure narrative coherence across different segments and associate speech content with visual entities. Subsequently, the clueguided QA generation strategy (lower part) prompts LLMs to mine cross-segment and cross-modal clues from the script, followed by a locally-focused generation step to produce QA pairs based on the clues and relevant segments.

and QA synthesis into a single step tends to yield questions with weaker long-term temporal and cross-modal dependencies.

To address these problems, we introduce an automated audio-visual QA generation pipeline consisting of two main stages. First, to mitigate narrative incoherence and broken sound-source associations, we propose Entity-Anchored Scripting. This stage represents the video as a structured, script-like text, which includes a summary, a list of main entities, and segment-wise information with timestamps (covering speech transcriptions, non-speech sounds, and visual descriptions). The main entity list serves as a global prior to unify references across all segments. Meanwhile, we associate speech with the speaker, linking the visual and audio modalities through the same entity identifiers. Second, to generate questions with long-term temporal spans and strong cross-modal dependencies, we design Clue-Guided QA Generation. Building on the entity-anchored script, we first require the LLM to identify clues centered on specific audio-visual tasks, instead of letting it generate questions from the entire text. These clues from multiple segments and modalities collectively form comprehensive reasoning chains. The model then focuses on these clues to generate QA pairs featuring long-term temporal spans and strong cross-modal dependencies.

Leveraging this pipeline, we construct OmniVideo-100K, an instruction-tuning dataset containing 100K automatically generated audio-visual QA pairs, and OmniVideo-Test, a test set with 505 human-verified samples. Both datasets cover ten diverse audio-visual tasks. Experimental results on OmniVideo-Test reveal that while open-source MLLMs perform relatively well in basic audiovisual semantic association, they remain constrained in fine-grained temporal alignment and cross-

modal reasoning. Notably, after fine-tuning on OmniVideo-100K, models based on VITA-1.5 (Fu

- et al., 2025b), Qwen2.5-Omni-7B (Xu et al., 2025a) and Qwen3-Omni-30B-A3B-Instruct (Xu et al., 2025b) achieve performance gains of 20.59%, 17.82% and 13.86% on OmniVideo-Test, respectively. Furthermore, these improvements demonstrate strong generalization across multiple existing benchmarks, including Video-MME (Fu et al., 2025a), Daily-Omni (Zhou et al., 2025), and JointAVBench (Chao et al., 2025), among others. Qualitative analysis confirms that fine-tuned models effectively capture audio-visual clues, revealing a clear shift from a reliance on single-modality perception to cross-modal synergy.

2 METHOD

In this section, we detail our pipeline, as illustrated in Fig. 1. Our framework is organized into two primary modules: Entity-Anchored Video Scripting (Sec. 2.1) and Clue-Guided QA Generation (Sec. 2.2). Prompts are provided in Sec. C. Finally, we present an overview of our instruction-tuning dataset, OmniVideo-100K, and the test set, OmniVideo-Test (Sec. 2.3).

- 2.1 ENTITY-ANCHORED VIDEO SCRIPTING

We leverage MLLMs to transform audio-visual videos into structured scripts. A script consists of a summary, a main entity list, and a body of sequential segments. As shown in Fig. 1 (blue region), each segment includes its timestamp, visual description, and audio information (chronologically ordered speaker-labeled speech and non-speech sounds). We integrate discrete segments into a coherent script through unified entity identifiers across segment descriptions. Simultaneously, we tag speech with its speaker labels to associate audio and visual information.

Main Entity List. Before modality decoupling and video segmentation, we use MLLMs to identify main active entities (e.g., people, animals, and objects) that are highly relevant to the video content. For each entity, we generate a unique descriptive identifier (e.g., “Brunette Girl”) along with a detailed feature description. These identifiers and descriptions collectively form the main entity list, which serves as a global prior to guide and constrain the subsequent scripting process.

Audio Information with Timestamps. We process the audio stream extracted from the raw video. We use MLLMs to generate transcriptions with start and end timestamps, where segmentation is determined by natural pauses or semantic completeness. Meanwhile, the model also identifies common non-speech sounds and provides music descriptions, all with timestamps.

Coherent Segmented Visual Narratives. We first partition the video into primary segments to establish a temporal backbone for the script, setting a target duration (15 seconds in our implementation) to capture more visual details. We use speech timestamps as boundaries to prevent semantic gaps. Neighboring short segments are merged up to the target duration to improve efficiency. For videos without speech, the entire video is treated as a single primary segment.

Building upon this temporal backbone, we generate visual captions for each primary segment based on video frames. The main entity list is used as a prior to ensure consistent entity references across all descriptions. Specifically, if a primary segment exceeds the target duration, we split it evenly into multiple smaller sub-segments for sequential processing. The final visual caption for the primary segment is synthesized by concatenating the timestamps and corresponding descriptions of its constituent sub-segments.

For each individual segment (either a primary segment or a sub-segment), the MLLM analyzes every shot or significant scene change across four dimensions: Setting&Environment, Characters&Objects, Actions&Interactions, and Cinematography. The observed content is then synthesized into coherent paragraphs, each corresponding to a shot or scene.

Sound-Source Association. We use MLLMs to identify the speaker for each transcription, which is essential for associating speech with visual narratives in multi-person scenarios. We input the raw video, transcriptions, and the main entity list into the MLLM, requiring it to assign speaker labels by jointly analyzing visual and audio features. This process can also handle off-screen speakers, such as narrators, who may not have a visual presence. If a speaker is already listed in the main entity list, we use the existing identifier; otherwise, a unique identifier is generated for the new speaker.

To provide a broader narrative context, we also generate a summary of the raw video, with entity references constrained by the main entity list. Finally, all processed information is integrated into a complete audio-visual script.

- 2.2 CLUE-GUIDED QA GENERATION

Based on the coherent audio-visual script, we adopt a clue-guided strategy to construct QA pairs with long-term temporal spans and deep cross-modal dependencies. This strategy is divided into two steps: Global Clue Mining and Locally Focused Generation.

- Step 1: Global Clue Mining. We use LLMs to perform a scan of the full script to extract clues needed for task-specific QA generation. This process emphasizes the integration of information across multiple segments and modalities. For instance, in causal reasoning, we require the model to retrieve and link multimodal information distributed throughout the script to construct multiple causal chains. The true cause of an event can often only be understood by combining audio cues from some segments with the visual content of other segments.

For each identified chain, the model provides the relevant segment timestamps and a logical description of the audio-visual synergy. Compared to direct QA generation, this process converts the model’s implicit understanding of complex narratives into explicit, step-by-step reasoning.

- Step 2: Locally Focused Generation. After identifying high-value clue chains, we move to the QA generation phase. We use the logical descriptions and segment timestamps from the previous step as contextual prompts to guide the model to focus on the key segments during QA construction. This focused approach filters out irrelevant content from the full script, reducing the model’s cognitive load. Most importantly, since the generation is anchored on pre-verified clues, this strategy helps ensure that the resulting QA pairs meet the requirements for both long-term temporal spans and audio-visual synergy.

- 2.3 DATASET OVERVIEW

Task Taxonomy. We define ten general audio-visual QA tasks (detailed in Sec. A) organized into a three-level cognitive framework. This framework ranges from basic perception to advanced reasoning, ensuring the comprehensiveness of our data:

- • Alignment: This level includes Fine-Grained Perception and Scene Transformation Detection. As the foundation of the framework, it focuses on the ability to perceive and synchronize audio and visual information along the timeline.
- • Understanding: This level covers Context Understanding, Comparison, Sentiment Analysis, Event Sequence Ordering and Summarization. It emphasizes semantic analysis, requiring the model to link cross-modal information and grasp the narrative content.
- • Reasoning: This level comprises Causal Reasoning, Future Prediction and Hypothetical Reasoning. Representing advanced cognitive abilities, it requires the model to go beyond surface-level understanding to perform logical deduction and abstract thinking.

Video Curation. We collect videos from online video platforms. Based on the video categorization in OmniVideoBench (Li et al., 2025a), we initiate a search keyword pool comprising seven labels: vlog, news, cartoon, sports, documentary, tv, and ego. We iteratively expand this pool with newly collected video tags to ensure diversity. We filter out low-quality videos with resolutions below 480p and retain only English videos. Following Farr´e et al. (2024), we further filter videos based on visual dynamics and word density to ensure rich audio-visual information. Furthermore, to prevent models from relying on on-screen text rather than audio, we employ the video-subtitle-extractor tool1 to exclude videos containing hard-coded subtitles.

Dataset Composition. Based on the task taxonomy and curated videos, we construct datasets for training and evaluation using our pipeline. The models employed in this process include Gemini-2.5Pro Comanici et al. (2025) and Gemini-3-Pro DeepMind (2025). It is worth noting that since most queries in basic Alignment and Context Understanding tasks often target localized events within

1https://github.com/YaoFANGUK/video-subtitle-extractor

[Figure 25]

[Figure 26]

(b) Video length statistics.

[Figure 27]

(c) Word count statistics (mean±std) for Open-Ended and Multiple-Choice samples.

(a) Video category distribution.

- Figure 2: Statistics of the OmniVideo-100K dataset. (a) Video category distribution, illustrating a diverse range of real-world domains. (b) Video length distribution, showing that the content predominantly covers a range of 1–3 minutes. (c) Word count statistics (mean±std). The notably higher word count in Open-Ended (OE) answers stems from the inclusion of detailed explanations alongside final conclusions. Meanwhile, the lengths of the four Multiple-Choice (MCQ) options are balanced to mitigate potential length bias.

short clips, we directly prompt the model to mine clues and generate QA pairs in a single pass, rather than utilizing the full multi-step strategy.

The instruction-tuning set, OmniVideo-100K, comprises 100K automatically generated QA pairs from 5214 videos. The distributions of video categories and lengths are illustrated in Fig. 2a and Fig. 2b, respectively. The samples consist of both Open-Ended (OE) and Multiple-Choice Questions (MCQ) in a 7:3 ratio, with samples evenly distributed across all ten audio-visual tasks. We further analyzed the word count statistics in Fig. 2c. Notably, the OE answers are significantly longer than the MCQ options, as they provide a detailed explanation alongside the final conclusion. The lengths of the four MCQ options are carefully balanced to prevent models from exploiting length bias as a shortcut for learning.

The corresponding test set, OmniVideo-Test, contains 505 multiple-choice QA pairs from 264 videos. These samples are initially generated by our pipeline and then undergo manual review. Each QA pair is verified for factual accuracy, cross-modal dependency, and answer uniqueness. Specifically, we discarded questions that could be answered or easily guessed using only a single modality. Additionally, any pairs with factual errors or ambiguous options were removed. This filtering process results in a final pass rate of approximately 38.14%.

- Table 1: Comparison of OmniVideo-100K with existing audio-visual video datasets for instruction tuning. We highlight that our dataset provides complex temporal tasks, evidence-based QA, and structured narratives that can be used for broader application. * denotes the automated pipeline leverages annotations from other datasets rather than starting from raw video.

Datasets # Samples Domain Avg. Length Annotation Complex Temporal Evidence Chain Structured Narratives

AVSD (Alamri et al., 2019) 8K Open 30s Manual ✗ ✗ ✗ Pano-AVQA (Yun et al., 2021) 42.8K Panoramic 5s Manual ✗ ✗ ✗ Music-AVQA (Li et al., 2022) 32K Music 60s Manual ✗ ✗ ✗ AVQA (Yang et al., 2022) 40K Open 10s Manual ✗ ✗ ✗

JavisInst-Und (Liu et al., 2025) 110K Open 10s Automatic* ✗ ✗ ✗ EgoAVU-Instruct (Seth et al., 2026) 3M Ego 1–6min Automatic ✗ ✗ ✗

OmniVideo-100K 100K Open 103s Automatic ✔ ✔ ✔

Comparison with Existing Datasets. We compare OmniVideo-100K with several existing datasets, as summarized in Table 1. Early datasets, such as Music-AVQA (Li et al., 2022) and AVQA (Yang et al., 2022), rely on manual annotations, which limits scalability. While JavisInst-Und (Liu et al., 2025) employs an automated pipeline to expand the dataset size, it remains dependent on preexisting annotations and its video durations are relatively short. In contrast, EgoAVU-Instruct (Seth

- et al., 2026) covers longer video spans, but it is restricted to the egocentric domain.

Furthermore, whereas previous datasets mostly focus on simple temporal relationships, such as answering before/after queries of audio and visual events, OmniVideo-100K introduces more challenging objectives including fine-grained alignment and ordering tasks. Meanwhile, our clue-guided strategy anchors QA pairs to explicit audio-visual evidence chains. Consequently, our open-ended answers include detailed reasoning explanations grounded in cross-modal clues, rather than only the brief conclusions commonly found in prior datasets. Beyond these high-quality QA pairs, the scripts produced by our pipeline serve as a valuable byproduct of OmniVideo-100K. Compared with single-paragraph narratives that entangle audio and visual modalities, our structured scripts not only support QA generation but also provide reusable intermediate representations for broader applications like video editing.

- 3 EXPERIMENTS

To verify the effectiveness of OmniVideo-100K, we perform full-parameter fine-tuning on VITA1.5, Qwen2.5-Omni-7B and Qwen3-Omni-30B-A3B-Instruct. We employ LLaMA-Factory (Zheng et al., 2024) for the Qwen-series models, while VITA-1.5 is fine-tuned using its official implementation. Detailed parameter settings are shown in Table 13.

- 3.1 MAIN RESULTS

We evaluate and compare the performance of the fine-tuned MLLMs, the baseline models, and several other open-source models (video-SALMONN 2+ (Tang et al., 2025), OmniVinci (Ye et al., 2025a), Uni-MoE-2.0-Omni (Li et al., 2025c), and MiniCPM-o 4.5 (Yao et al., 2024)) on both OmniVideo-Test and multiple existing benchmarks. During inference, we sample 16 frames for VITA-1.5 and 64 for other models.

- 3.1.1 PERFORMANCE ON OMNIVIDEO-TEST

- Table 2: Performance comparison on OmniVideo-Test. We report results across task dimensions and video durations. Red values denote the performance gains achieved after fine-tuning on the OmniVideo-100K dataset.

Models Size Overall Alignment Understanding Reasoning (0, 2]min (2, 5]min Human - 100 - - - - Gemini-3.1-Pro - 83.96 83.62 84.50 83.21 82.61 84.59 VITA-1.5 7B 40.99 28.45 43.02 48.09 44.72 39.24 Qwen2.5-Omni 7B 42.77 39.66 46.51 38.17 38.51 44.77 video-SALMONN 2+ 7B 45.15 37.93 49.22 43.51 46.58 44.48 OmniVinci 9B 47.13 43.97 51.55 41.22 47.20 47.09 MiniCPM-o 4.5 9B 55.25 56.90 58.14 48.09 53.42 56.10 Qwen3-Omni 30B 49.70 43.10 55.04 45.04 49.07 50.00 Uni-MoE-2.0-Omni 30B 46.93 39.66 52.71 41.98 46.58 47.09

OmniVideo-7B(VITA-1.5) 7B 61.58+20.59 59.48+31.03 63.18+20.16 60.31+12.22 59.01+14.29 62.79+23.55 OmniVideo-7B(Qwen2.5-Omni) 7B 60.59+17.82 62.93+23.27 62.40+15.89 54.96+16.79 54.66+16.15 63.37+18.60 OmniVideo-30B(Qwen3-Omni) 30B 63.56+13.86 60.34+17.24 67.05+12.01 59.54+14.50 62.11+13.04 64.24+14.24

Performance Discrepancies Across Tasks. Table 2 shows the performance on OmniVideo-Test (refer to Table 14 for detailed per-task results). Existing models generally perform better on the Understanding tasks than on Alignment and Reasoning tasks. For example, video-SALMONN 2+ achieves 49.22% on Understanding, but only 37.93% and 43.51% on Alignment and Reasoning, respectively. This indicates that the ability to establish preliminary semantic connections between visual and audio content and achieve basic understanding is relatively well-handled by current models. In contrast, for the Alignment tasks, which involve Fine-Grained Perception and Scene Transformation Detection, the main difficulty lies in precisely aligning visual and audio information in

[Figure 28]

[Figure 29]

###### Question:

[Figure 30]

[Figure 31]

During the discussion, the bald pundit performs an action where he reaches up and removes the glasses resting on his forehead. What is he speculating about at that exact moment?

###### Qwen2.5-Omni:

... Maybe he was about to make a point or just needed to see better for a moment …

###### Ours:

… it’ll give them a decision in the coming weeks because they’re complaining, …

He is speculating that the club might be seeking a decision in the coming weeks because they are complaining.

[Figure 32]

[Figure 33]

###### Question:

[Figure 34]

[Figure 35]

What specific detail in the narrative motivated the speaker’s gesture near his head?

###### Qwen2.5-Omni:

... father was a soldier ... His gesture near his head might be a way of showing the deep emotional impact of that memory …

###### Ours:

The speaker raises his right hand while recounting the memory of a soldier returning with a "deep dent in his

… And I remember the man coming back with a deep dent in his military helmet …

military helmet." …

[Figure 36]

[Figure 37]

###### Question:

[Figure 38]

[Figure 39]

What specific aspect of the interview motivated the man in blue to advise that players need to "act" to avoid "predators"?

###### Qwen2.5-Omni:

... He used the term "predators" to mean the high expectations and scrutiny ... So, he advised that players need to "act" like they're okay …

###### Ours:

… I’ve been terrible, … You gotta make

… the man in black verbally admits that he has "been terrible" … players must "act" to "trick … because

sure that you don’t let your guard down

"predators are watching" … the man in blue, who is seen nodding and gesturing emphatically to stress …

because predators are watching and …

Figure 3: Qualitative comparison between the baseline (Qwen2.5-Omni-7B) and our fine-tuned model on OmniVideo-Test. We directly input the questions into the models to obtain open-ended responses. Green text highlights accurate answers grounded in specific evidence, while red text indicates vague speculation or erroneous content. Compared to the baseline, our model reduces reliance on unimodal speculation (Case 1), exhibits improved fine-grained temporal alignment (Case 2), and facilitates cross-temporal synergistic reasoning (Case 3).

the temporal dimension. Except for MiniCPM-o 4.5, most open-source models perform similarly, reflecting that existing models still have certain limitations in temporal alignment. For the Reasoning tasks, models not only need cross-modal semantic understanding but must integrate multimodal cues to perform logical reasoning and derive abstract conclusions. However, even for MiniCPM-o

- 4.5, which exhibits strong results in the first two dimensions, its accuracy in this dimension still does not exceed 50%, reflecting that current open-source MLLMs still have significant room for improvement in deep cross-modal reasoning.

Improvements from OmniVideo-100K. Models fine-tuned on OmniVideo-100K exhibit significant performance gains of 20.59%, 17.82% and 13.86% over their respective baselines, effectively enhancing synergetic understanding in audio-visual scenarios. Fig. 3 presents a qualitative comparison on OmniVideo-Test between the baseline Qwen2.5-Omni and its fine-tuned counterpart. To provide a more intuitive analysis, we conduct tests using the open-ended form. We find that the baseline often relies on single-modality cues to generate speculative responses, rather than grounding its answers in the actual audio-visual content (e.g., Case 1). In Case 2, the baseline only captures audio content (“father was a soldier”) proximate to the target timestamp, thereby producing an erroneous response which highlights limitations in fine-grained temporal alignment. Furthermore, in Case 3, the baseline fails due to its inability to associate information from the preceding segment. In contrast, the fine-tuned model successfully leverages both cross-temporal and cross-modal clues to complete the reasoning. It not only incorporates content from the earlier interview but also attends to the visual actions of the man in blue. These qualitative results demonstrate that OmniVideo-100K not only improves temporal alignment but also facilitates a transition from single-modality reliance to joint multimodal understanding.

Cross-modal Dependency Analysis. Based on the results in Table 2, we select two representative open-source models, namely MiniCPM-o 4.5, which achieves the highest overall performance among open-source models, and Qwen3-Omni, which has a larger parameter scale, to further analyze the cross-modal dependency of OmniVideo-Test. We compare their performance in audio-only and visual-only settings against the audio-visual mode. As shown in Table 3, both models exhibit a performance decline when limited to a single modality, demonstrating that the samples in our test set require cross-modal synergy. Notably, although MiniCPM-o 4.5 and Qwen3-Omni show com-

- Table 3: Performance comparison under different modality configurations on OmniVideo-Test. We evaluate two open-source models across audio-visual, audio-only, and visual-only settings to analyze cross-modal dependency. Green and red values denote performance changes relative to the full audio-visual setting.

Models Size Overall Alignment Understanding Reasoning (0, 2]min (2, 5]min

Audio-Visual

MiniCPM-o 4.5 9B 55.25 56.90 58.14 48.09 53.42 56.10 Qwen3-Omni 30B 49.70 43.10 55.04 45.04 49.07 50.00

Audio-Only MiniCPM-o 4.5 9B 45.74-9.51 36.21-20.69 51.94-6.20 41.98-6.11 45.34-8.08 45.93-10.17

- Qwen3-Omni 30B 46.14-3.56 37.07-6.03 52.71-2.33 41.22-3.82 48.45-0.62 45.06-4.94 Visual-Only

MiniCPM-o 4.5 9B 47.92-7.33 42.24-14.66 50.00-8.14 48.85+0.76 51.55-1.87 46.22-9.88

- Qwen3-Omni 30B 47.13-2.57 42.24-0.86 48.84-6.20 48.09+3.05 45.96-3.11 47.67-2.33

parable unimodal performance with gaps below 1%, MiniCPM-o 4.5 gains a larger increase when both modalities are provided. Specifically, MiniCPM-o 4.5 obtains an improvement of 7% to 9% while Qwen3-Omni improves by only about 3%, indicating a higher synergistic gain.

- 3.1.2 GENERALIZATION ON EXISTING BENCHMARKS

- Table 4: Performance comparison on audio-visual and general video benchmarks. Due to the 5minute audio truncation limit of Qwen2.5-Omni, results for Video-MME-v2, OmniVideoBench, JointAVBench, and FutureOmni are reported only for samples within this duration; the row “# Samples” denotes the number of samples retained. Red values denote the performance gains achieved by fine-tuned models.

Models Size Video-MMEshort Video-MME-v2 Daily-Omni OmniVideoBench JointAVBench FutureOmni # Samples - 900 328 1197 509 2153 960

VITA-1.5 7B 70.63 5.95 52.63 36.35 44.77 48.65 Qwen2.5-Omni 7B 75.56 10.28 62.41 36.54 54.44 48.85 video-SALMONN 2+ 7B 74.22 11.50 62.57 35.36 56.39 51.88 OmniVinci 9B 77.56 11.59 61.32 36.74 57.55 52.81 MiniCPM-o 4.5 9B 83.22 14.48 80.20 41.06 55.18 52.29 Qwen3-Omni 30B 82.00 14.31 74.27 43.84 63.17 53.44 Uni-MoE-2.0-Omni 30B 78.89 8.82 64.33 38.55 57.55 52.81

OmniVideo-7B(VITA-1.5) 7B 67.67-2.96 7.42+1.47 55.39+2.76 36.94+0.59 57.41+12.64 56.35+7.70 OmniVideo-7B(Qwen2.5-Omni) 7B 76.33+0.77 8.50-1.78 69.84+7.43 39.88+3.34 60.75+6.31 55.00+6.15 OmniVideo-30B(Qwen3-Omni) 30B 83.56+1.56 15.33+1.02 76.61+2.34 44.81+0.97 66.37+3.20 57.60+4.16

To evaluate the generalization of the fine-tuned models, we compare their performance across several established audio-visual benchmarks. As shown in Table 4, our fine-tuned models achieve consistent performance gains across Daily-Omni (Zhou et al., 2025), OmniVideoBench (Li et al., 2025a), JointAVBench (Chao et al., 2025), and FutureOmni (Chen et al., 2026). VITA-1.5 demonstrates a performance increase of 0.6%–12% across various metrics, while Qwen2.5-Omni and the larger Qwen3-Omni achieve improvements of 3%–7% and 1%–4%, respectively.

- Table 5: Performance breakdown on Daily-Omni. We compare the results of the baseline Qwen2.5Omni against our fine-tuned model across task types.

Models AV Event Alignment Comparative Context Understanding Event Sequence Inference Reasoning

Qwen2.5-Omni 51.26 68.70 56.48 57.19 77.27 75.43 OmniVideo-7B(Qwen2.5-Omni) 68.49 73.28 70.98 60.13 79.22 76.57

Performance Breakdown. A detailed analysis reveals both the strengths of our approach and areas for future refinement. Specifically, Qwen2.5-Omni achieves its most significant gain of 7.43% on Daily-Omni. A more granular analysis (Table 5) reveals that this growth is primarily driven by a breakthrough in cross-modal alignment capabilities. The model achieves a remarkable 17.23% increase in the AV Event Alignment task. However, we also observe that the overall performance gain

of Qwen3-Omni on OmniVideoBench is relatively limited. To investigate the underlying cause, we break down the performance by audio types following the benchmark’s settings, as shown in Table 6.

The results reveal that while our model improves on music and speech samples, performance on the sound category drops by

Table 6: Performance breakdown on OmniVideoBench. Results are compared across audio types between the baseline Qwen3-Omni and our fine-tuned model.

- 4.21%. We attribute this decline to the scripting process, where, except for music, nonspeech sounds are often represented by general, coarse-grained category descriptions to avoid over-speculation. Such descriptions can introduce noise into the QA generation process. This finding suggests that future work should integrate specialized audio models to provide more precise and granular acoustic signals, particularly for complex sounds.

Models Music Sound Speech

Qwen3-Omni 36.73 41.05 45.50 OmniVideo-30B(Qwen3-Omni) 46.94 36.84 46.59

General Video Understanding. In addition to specialized audio-visual benchmarks, we also evaluate our models on general video understanding tasks, specifically Video-MME (Fu et al., 2025a) and Video-MME-v2 (Fu et al., 2026). As shown in Table 4, despite being fine-tuned primarily for complex cross-modal comprehension, our models maintain their original baseline performance across both general benchmarks. Although some fluctuations are observed, there is no significant performance drop; several configurations even exhibit improvements. This demonstrates that fine-tuning on OmniVideo-100K effectively enhances audio-visual synergistic capabilities without compromising the models’ original general video understanding.

- 3.1.3 COMPARISON WITH EXISTING DATASETS

Table 7: Performance comparison of Qwen2.5-Omni fine-tuned on different audio-visual datasets across multiple benchmarks.

Data Video-MMEshort Video-MME-v2 OmniVideo-Test Daily-Omni OmniVideoBench JointAVBench FutureOmni

Qwen2.5-Omni 75.56 10.28 42.77 62.41 36.54 54.44 48.85 w. AVQA 68.11 9.69 6.28 55.14 34.38 50.16 43.65 w. JavisInst-Und 59.44 3.22 38.22 48.96 32.42 44.36 58.54 w. OmniVideo-100K 76.33 8.50 60.59 69.84 39.88 60.75 55.00

We select two open-domain audio-visual datasets, AVQA and JavisInst-Und, to fine-tune Qwen2.5Omni for a quantitative comparison with our dataset. To ensure fairness, all training parameters are kept identical to those used for OmniVideo-100K. The AVQA dataset is manually annotated and contains only multiple-choice questions. We exclude 243 samples due to unavailable videos, resulting in a final training set of 40182 samples. JavisInst-Und is constructed via an automated pipeline and comprises 110K samples with a mix of open-ended and multiple-choice QA, which is similar in scale and format to our dataset. Note that, for this baseline, we remove the initial videogeneration rounds within the multi-turn dialogues of the Gen2Und task and retain the remaining turns. As shown in Table 7, models fine-tuned on AVQA and JavisInst-Und show varying degrees of performance drops across most benchmarks compared to the original base model. In contrast, our OmniVideo-100K achieves consistent and significant gains on most benchmarks. These results show that, compared to other datasets, our dataset is more effective in enhancing the video understanding capabilities of MLLMs.

3.2 ABLATION ANALYSIS

- 3.2.1 CLUE-GUIDED QA GENERATION

Experimental Setup. To investigate the impact of our Clue-Guided Generation, we compare it against a Direct Generation baseline. In this baseline, the entire video script is fed into the model, which is then prompted to directly generate QA pairs by integrating cross-segment and cross-modal clues. Since our clue-mining step produces multiple clue chains, we similarly require the baseline to output multiple QA pairs. This experiment is conducted on 20 randomly sampled videos, yielding a total of 71 QA pairs per method, which are generated by Gemini-3-Pro across all videos and

encompass both Causal Reasoning and Future Prediction tasks. We ensure that the number of QA pairs per task for each video remains identical between the two methods.

Question Difficulty. As shown in Table 8, all models achieve lower accuracy on QA pairs generated by Clue-Guided Generation than on those from the Direct Generation baseline, indicating that our strategy produces more challenging questions. Moreover, while different models exhibit highly similar performance on Direct samples, the performance gaps become more pronounced on Clue-Guided QA pairs, suggesting that our strategy better distinguishes between varying model capabilities. Notably, after fine-tuning on OmniVideo-100K, Qwen2.5-Omni shows only a marginal gain on Direct samples but a substantial improvement on the more demanding Clue-Guided set, with accuracy increasing from 59.15% to 80.28%. The enhancement is further corroborated by the consistent performance gains observed across multiple established benchmarks in Table 4, thereby verifying the effectiveness of OmniVideo-100K in enhancing audio-visual understanding.

Table 8: Accuracy of various MLLMs on QA pairs generated by Direct Generation and ClueGuided Generation strategies.

Models Direct Clue-Guided

Qwen2.5-Omni 80.28 59.15 MiniCPM-o 4.5 78.87 74.65 OmniVideo-7B(Qwen2.5-Omni) 81.69 80.28

Temporal Scope. We compare the average temporal span of the content covered by the questions (measured from the first evidence clue to the last) across both sets of QA pairs. For Clue-Guided QA, we directly utilize the timestamps of relevant segments from the intermediate clue chains, calculating the span by subtracting the end time of the first segment from the start time of the last. For Direct Generation, we prompt Gemini-3-Pro to determine the temporal span based on the raw video. The results reveal that Clue-Guided Generation yields an average temporal span of 144.75s, far exceeding 76.24s of the Direct Generation baseline. This demonstrates that our two-stage clue-guided approach effectively directs the model to establish connections across longer temporal distances.

- 3.2.2 MAIN ENTITY LIST

To verify the effectiveness of the Main Entity List in maintaining narrative coherence and mitigating sound-source mismatch, we conduct comparative experiments on 20 randomly sampled videos.

Referential Consistency. We develop a variant that removes the main entity list constraint during the segmented visual description phase. To assess how this referential consistency impacts QA generation, we prompt Gemini-3-Pro to generate three QA pairs per video based on the scripts. Answering these questions requires integrating entity information across multiple segments. Each pipeline generates 60 QA pairs in total. We utilize Gemini-3-Pro to judge whether the generated pairs suffer from entity confusion by referencing the raw videos. The results show that without the global prior, the error rate regarding entity confusion is 36.7%, whereas the full pipeline reduces this error rate to 23.4%. This indicates that, compared to inconsistent referencing across segments, the constraints from the main entity list facilitate the construction of cohesive narrative logic, which ultimately supports the generation of more reliable QA pairs.

Audio-Visual Association. Similarly, we construct a variant by removing speaker labels to investigate their role in cross-modal grounding. We require the model to generate three QA pairs per video based on the scripts, necessitating the correct association of speech content with the visual information of the corresponding speaker. Both the full script and the label-free script pipelines produce 60 QA pairs each. Evaluation by Gemini-3-Pro against the raw videos reveals that the audio-visual mismatch rate is 20% without speaker labels, compared to only 10% when labels are included. This demonstrates that explicit speaker labeling within the script effectively helps the model establish more accurate audio-visual associations, thereby enhancing the validity of the resulting QA pairs.

- 3.2.3 VIDEO-BASED VS. SCRIPT-BASED

Experimental Setup. To evaluate the necessity of the intermediate script representation, we randomly sample 20 videos to compare QA pairs generated from raw video with those generated using the script. Each pipeline utilizes the clue-guided strategy to produce a total of 60 QA pairs across Causal Reasoning and Future Prediction tasks.

Contextual Breadth. The script-based approach inherently captures broader context. We compare the average temporal span of both sets of QA pairs, calculated by subtracting the end timestamp of the first relevant segment from the start timestamp of the last. As shown in Table 9, the average temporal span for the scriptbased approach is 37.88s higher than that of the video-based baseline. The results across different durations show that the advantage of integrating long-term information is more significant in longer videos, with the gap reaching 79.65s for videos ranging from 8 to 14 minutes.

Cross-Modal Synergy. This increased difficulty of script-based QA is corroborated by the results in Table 10, where we evaluate the performance of MLLMs specifically on the 8– 14min samples. Across all modality settings, models exhibit a consistent performance drop when switching from video-based to scriptbased QA pairs. Moreover, the performance degradation observed when removing a single modality is more severe for script-based QA. This disparity shows that our pipeline can generate more rigorous QA pairs that necessitate cross-modal synergy.

- 3.2.4 IMPACT OF DATA SCALING

- Table 9: Average temporal span of QA pairs generated by video-based and script-based clueguided pipelines across different video durations. The row “# Samples” denotes the number of QA pairs in each duration group.

Pipelines Overall [1,3]min [8,14]min # Samples 60 28 32

Video + Clue-Guided 131.57s 53.43s 190.85s Script + Clue-Guided 169.45s 53.96s 270.50s

- Table 10: Model accuracy on 8–14 minute QA pairs generated by video-based and script-based clue-guided pipelines under full audio-visual input and single-modality ablation settings. Green values denote the performance drop relative to the full audio-visual setting.

Models Video + Clue-Guided Script + Clue-Guided Audio-Visual

OmniVinci 78.12 68.75 MiniCPM-o 4.5 84.38 75.00

Visual-Only

OmniVinci 65.62-12.50 40.62-28.13 MiniCPM-o 4.5 78.12-6.26 46.88-28.12

Audio-Only

OmniVinci 75.00-3.12 62.50-6.25 MiniCPM-o 4.5 84.38-0.00 68.75-6.25

Table 11: Performance scaling of Qwen2.5-Omni-7B across different data volumes. “Avg.” denotes the average score across all listed benchmarks.

Data Avg. Video-MMEshort Video-MME-v2 Daily-Omni OmniVideoBench JointAVBench FutureOmni OmniVideo-Test

w/o SFT 47.26 75.56 10.28 62.41 36.54 54.44 48.85 42.77 OmniVideo-10K 52.64 77.22 13.84 69.92 39.10 60.57 52.60 55.25 OmniVideo-25K 53.98 76.56 13.74 70.68 42.63 60.15 54.27 59.80 OmniVideo-50K 54.14 76.33 11.38 69.84 42.83 59.45 57.60 61.58 OmniVideo-75K 54.32 76.89 10.47 72.26 41.26 61.03 55.73 62.57 OmniVideo-100K 52.98 76.33 8.50 69.84 39.88 60.75 55.00 60.59

To investigate how the volume of our generated data affects model performance, we fine-tune Qwen2.5-Omni on different subsets of OmniVideo ranging from 10K to 100K samples. All training settings remain consistent with those used in the main experiments. As shown in Table 11, compared to the baseline model (w/o SFT), incorporating merely 10K samples leads to a performance leap across almost all benchmarks. For instance, accuracy on Daily-Omni and JointAVBench jumps from 62.41% to 69.92% and 54.44% to 60.57%, respectively. Performance exhibits steady improvements as the data volume increases from 10K to 75K, with the average score (Avg.) peaking at 54.32% at the 75K mark. However, when the data volume is further expanded to 100K, we observe slight performance fluctuations or saturation across several metrics. Nonetheless, the overall scaling trajectory validates the effectiveness and efficiency of our automated data generation pipeline.

- 4 RELATED WORK

Multimodal Large Language Models. Building on the success of Multimodal Large Language Models (Li et al., 2024; Bai et al., 2025; Marafioti et al., 2025; Wang et al., 2025; Wu et al., 2025; Zhang et al., 2026; Li et al., 2026a), researchers (Zhan et al., 2024; Cheng et al., 2024; Fu et al.,

- 2024; Xie & Wu, 2024; Fu et al., 2025b; Li et al., 2025b; AI et al., 2025) expand model capabilities from visual-only analysis to audio-visual understanding by adding audio branches. These models typically concatenate visual and audio tokens directly before feeding them into the LLM. To better

- capture the synchronization within audio-visual videos, several works (Xu et al., 2025a; Tang et al.,
- 2025; Ye et al., 2025a; Li et al., 2025c) introduce specialized designs for temporal alignment. For example, Qwen3-Omni (Xu et al., 2025b) interleaves visual and audio tokens based on actual time. JavisGPT (Liu et al., 2025) employs a SyncFusion module to integrate audio information into visual features, explicitly modeling spatio-temporal synchrony.

Audio-Visual Video Datasets. Early studies (Alamri et al., 2019; Yun et al., 2021; Li et al., 2022; Yang et al., 2022; Liu et al., 2024b) establish the foundation for audio-visual question answering, followed by a surge of datasets and benchmarks (Liu et al., 2024a; Hong et al., 2025; Li et al., 2025a; Xie et al., 2026; Tao et al., 2026; Goel et al., 2026; Ye et al., 2024; Wang et al., 2024; Sung-Bin et al., 2024; Zhang et al., 2025; Yang et al., 2025; Liu et al., 2025; Chen et al., 2026; Seth et al., 2026; Li et al., 2025d) designed to enhance and evaluate the cross-modal capabilities of MLLMs across diverse scenarios. Most recent works (Geng et al., 2025b; Chao et al., 2025; Ye et al., 2025a; Kurpath et al., 2025) employing automated data engines typically segment videos and use specialized models to extract uni-modal information (such as visual captions, speech transcriptions, and audio descriptions). These texts are then jointly fed into LLMs to produce segment-level audiovisual captions, which are ultimately used to generate QA pairs. However, this method often leads to narrative incoherence. Although Daily-Omni (Zhou et al., 2025) attempts to address this by using the full video to revise and align descriptions across modalities, this post-process may fail to preserve all original details, leading to information loss.

Video Structured Scripting. Conventional video captioning typically provides a single descriptive text for each segment. Some efforts (Du et al., 2025; Li et al., 2026b) have explored multi-attribute or multi-dimensional structuring to yield more fine-grained captions. More recently, TimeChatCaptioner (Yao et al., 2026) proposes the Omni Dense Captioning task, generating script-like captions by utilizing a six-dimensional schema to ensure comprehensive coverage of audio-visual content. However, descriptions generated by such methods often remain logically isolated across segments. In contrast, we introduce a main entity list as a global prior to synthesize coherent audiovisual scripts. Among concurrent works, OmniScript (Pu et al., 2026) focuses on training MLLMs to generate scripts directly from video, while Script-a-Video (Team, 2026) adopts the script as a bridge that distills video content into structured evidence for understanding and provides a blueprint for video generation. Distinct from these works, we leverage structured scripts to generate audio-visual QA pairs with long-term temporal spans and cross-modal dependencies. These samples are used to fine-tune MLLMs, thereby enhancing their comprehension capabilities in audio-visual scenarios.

- 5 CONCLUSION

In this paper, we propose an automated data synthesis engine designed to enhance the audio-visual understanding capabilities of MLLMs. Our pipeline is built upon two stages: (1) We extract audio and visual information from video segments to construct a structured, script-like text, leveraging the main entity list prior to maintain narrative coherence across segments and explicitly reconstruct audio-visual associations. (2) We design a clue-guided QA generation strategy that prompts the model to first mine clues involving long-term temporal spans and cross-modal dependencies, and then generate QA pairs based on these clues. The resulting instruction-tuning dataset, OmniVideo100K, effectively addresses the shortcomings of existing open-source models in synergistic crossmodal comprehension, demonstrating consistent performance gains on both our test set, OmniVideoTest, and established benchmarks. In future work, we plan to explore more robust adaptive segmentation strategies and integrate specialized audio models to capture fine-grained sounds and paralinguistic features, thereby further elevating the quality of automatically generated data.

ACKNOWLEDGMENTS This work is funded by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM902).

REFERENCES

Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, et al. Ming-omni: A unified multimodal

model for perception and generation. arXiv preprint arXiv:2506.09344, 2025.

Huda Alamri, Vincent Cartillier, Abhishek Das, Jue Wang, Anoop Cherian, Irfan Essa, Dhruv Batra, Tim K Marks, Chiori Hori, Peter Anderson, et al. Audio visual scene-aware dialog. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7558–7567, 2019.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Jianghan Chao, Jianzhang Gao, Wenhui Tan, Yuchong Sun, Ruihua Song, and Liyun Ru. Jointavbench: A benchmark for joint audio-visual reasoning evaluation. arXiv preprint arXiv:2512.12772, 2025.

Qian Chen, Jinlan Fu, Changsong Li, See-Kiong Ng, and Xipeng Qiu. Futureomni: Evaluating future forecasting from omni-modal context for multimodal llms. arXiv preprint arXiv:2601.13836, 2026.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Google DeepMind. A new era of intelligence with gemini 3. https://blog.google/ products-and-platforms/products/gemini/gemini-3/, 2025.

Yang Du, Zhuoran Lin, Kaiqiang Song, Biao Wang, Zhicheng Zheng, Tiezheng Ge, Bo Zheng, and Qin Jin. Vc4vg: Optimizing video captions for text-to-video generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 1124–1138, 2025.

Miquel Farr´e, Andi Marafioti, Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. Finevideo. https://huggingface.co/datasets/HuggingFaceFV/finevideo, 2024.

Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Yuhang Dai, Meng Zhao, Yi-Fan Zhang, Shaoqi Dong, Yangze Li, Xiong Wang, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24108–24118, 2025a.

Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025b.

Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, et al. Video-mme-v2: Towards the next stage in benchmarks for comprehensive video understanding. arXiv preprint arXiv:2604.05015, 2026.

Tiantian Geng, Teng Wang, Jinming Duan, Yanfu Zhang, Weili Guan, Feng Zheng, and Ling Shao. Uniav: Unified audio-visual perception for multi-task video event localization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(11):10280–10294, 2025a. doi: 10.1109/TPAMI.2025.3593932.

Tiantian Geng, Jinrui Zhang, Qingni Wang, Teng Wang, Jinming Duan, and Feng Zheng. Longvale: Vision-audio-language-event benchmark towards time-aware omni-modal perception of long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18959–18969, 2025b.

Arushi Goel, Sreyan Ghosh, Vatsal Agarwal, Nishit Anand, Kaousheik Jayakumar, Lasha Koroshinadze, Yao Xu, Katie Lyons, James Case, Karan Sapra, et al. Mmou: A massive multi-task omni understanding and reasoning benchmark for long and complex real-world videos. arXiv preprint arXiv:2603.14145, 2026.

Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2025.

Mohammed Irfan Kurpath, Jaseel Muhammad Kaithakkodan, Jinxing Zhou, Sahal Shaji Mullappilly, Mohammad Almansoori, Noor Ahsan, Beknur Kalmakhanbet, Sambal Shikhar, Rishabh Lalla, Jean Lahoud, et al. A benchmark and agentic framework for omni-modal reasoning and tool use in long videos. arXiv preprint arXiv:2512.16978, 2025.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Wentao Wang, Zhenghao Song, Dingling Zhang, et al. Omnivideobench: Towards audio-visual understanding evaluation for omni mllms. arXiv preprint arXiv:2510.10689, 2025a.

Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19108–19118, 2022.

Juncheng Li, Minghe Gao, Xiangnan He, Siliang Tang, Wei-Shi Zheng, Jun Xiao, Meng Wang, Tat-Seng Chua, and Yueting Zhuang. Momentor++: Advancing video large language models with fine-grained long video reasoning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(6):6208–6224, 2026a. doi: 10.1109/TPAMI.2026.3656169.

Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv preprint arXiv:2501.15368, 2025b.

Yunheng Li, Hengrui Zhang, Meng-Hao Guo, Wenzhao Gao, Shaoyong Jia, Shaohui Jiao, Qibin Hou, and Ming-Ming Cheng. Towards universal video mllms with attribute-structured and quality-verified instructions. arXiv preprint arXiv:2602.13013, 2026b.

Yunxin Li, Xinyu Chen, Shenyuan Jiang, Haoyuan Shi, Zhenyu Liu, Xuanyu Zhang, Nanhao Deng, Zhenran Xu, Yicheng Ma, Meishan Zhang, et al. Uni-moe-2.0-omni: Scaling language-centric omnimodal large model with advanced moe, training and data. arXiv preprint arXiv:2511.12609, 2025c.

Zinuo Li, Xian Zhang, Yongxin Guo, Mohammed Bennamoun, Farid Boussaid, Girish Dwivedi, Luqi Gong, and Qiuhong Ke. Watch and listen: Understanding audio-visual-speech moments with multimodal llm. arXiv preprint arXiv:2505.18110, 2025d.

Jing Liu, Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, and Jinhui Tang. Valor: Vision-audio-language omni-perception pretraining model and dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(2):708–724, 2024a.

Kai Liu, Jungang Li, Yuchong Sun, Shengqiong Wu, Jianzhang Gao, Daoan Zhang, Wei Zhang, Sheng Jin, Sicheng Yu, Geng Zhan, et al. Javisgpt: A unified multi-modal llm for sounding-video comprehension and generation. arXiv preprint arXiv:2512.22905, 2025.

Xiulong Liu, Zhikang Dong, and Peng Zhang. Tackling data bias in music-avqa: Crafting a balanced dataset for unbiased question-answering. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 4478–4487, 2024b.

Andr´es Marafioti, Orr Zohar, Miquel Farr´e, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025.

Junfu Pu, Yuxin Chen, Teng Wang, and Ying Shan. Omniscript: Towards audio-visual script generation for long-form cinematic video. arXiv preprint arXiv:2604.11102, 2026.

Ashish Seth, Xinhao Mei, Changsheng Zhao, Varun Nagaraja, Ernie Chang, Gregory P Meyer, Gael Le Lan, Yunyang Xiong, Vikas Chandra, Yangyang Shi, et al. Egoavu: Egocentric audiovisual understanding. arXiv preprint arXiv:2602.06139, 2026.

Kim Sung-Bin, Oh Hyun-Bin, JungMok Lee, Arda Senocak, Joon Son Chung, and Tae-Hyun Oh. Avhbench: A cross-modal hallucination benchmark for audio-visual large language models. arXiv preprint arXiv:2410.18325, 2024.

Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, and Chao Zhang. video-salmonn 2: Caption-enhanced audio-visual large language models. arXiv preprint arXiv:2506.15220, 2025.

Keda Tao, Yuhua Zheng, Jia Xu, Wenjie Du, Kele Shao, Hesong Wang, Xueyi Chen, Xin Jin, Junhan Zhu, Bohan Yu, et al. Lvomnibench: Pioneering long audio-video understanding evaluation for omnimodal llms. arXiv preprint arXiv:2603.19217, 2026.

Tencent Hunyuan Team. Script-a-video: Deep structured audio-visual captions via factorized streams and relational grounding. arXiv preprint arXiv:2604.11244, 2026.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European conference on computer vision, pp. 396–416. Springer, 2024.

Wenhao Wu, Xiaohan Wang, Haipeng Luo, Jingdong Wang, Yi Yang, and Wanli Ouyang. Cap4video++: Enhancing video understanding with auxiliary captions. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(7):5223–5237, 2025. doi: 10.1109/TPAMI.2024. 3410329.

Tianyu Xie, Jinfa Huang, Yuexiao Ma, Rongfang Luo, Yan Yang, Wang Chen, Yuhui Zeng, Ruize Fang, Yixuan Zou, Xiawu Zheng, et al. Socialomni: Benchmarking audio-visual social interactivity in omni models. arXiv preprint arXiv:2603.16859, 2026.

Zhifei Xie and Changqiao Wu. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. arXiv preprint arXiv:2410.11190, 2024.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report, 2025a. URL https://arxiv.org/abs/2503.20215.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025b.

Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. Avqa: A dataset for audio-visual question answering on videos. In Proceedings of the 30th ACM international conference on multimedia, pp. 3480–3491, 2022.

Yudong Yang, Jimin Zhuang, Guangzhi Sun, Changli Tang, Yixuan Li, Peihan Li, Yifan Jiang, Wei Li, Zejun Ma, and Chao Zhang. Audio-centric video understanding benchmark without text shortcut. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 6580–6598, 2025.

Linli Yao, Yuancheng Wei, Yaojie Zhang, Lei Li, Xinlong Chen, Feifan Song, Ziyue Wang, Kun Ouyang, Yuanxin Liu, Lingpeng Kong, et al. Timechat-captioner: Scripting multi-scene videos with time-aware and structural audio-visual captions. arXiv preprint arXiv:2602.08711, 2026.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.

Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, et al. Omnivinci: Enhancing architecture and data for omni-modal understanding llm. arXiv preprint arXiv:2510.15870, 2025a.

Qilang Ye, Zitong Yu, Rui Shao, Xinyu Xie, Philip Torr, and Xiaochun Cao. Cat: Enhancing multimodal large language model to answer questions in dynamic audio-visual scenarios. In European Conference on Computer Vision, pp. 146–164. Springer, 2024.

Qilang Ye, Zitong Yu, Rui Shao, Yawen Cui, Xiangui Kang, Xin Liu, Philip Torr, and Xiaochun Cao. Cat+: Investigating and enhancing audio-visual understanding in large language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(10):8674–8690, 2025b. doi: 10.1109/TPAMI.2025.3582389.

Heeseung Yun, Youngjae Yu, Wonsuk Yang, Kangil Lee, and Gunhee Kim. Pano-avqa: Grounded audio-visual question answering on 360deg videos. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2031–2041, 2021.

Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. Anygpt: Unified multimodal llm with discrete sequence modeling. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9637–9662, 2024.

Kecheng Zhang, Zongxin Yang, Mingfei Han, Yunzhi Zhuge, Haihong Hao, Changlin Li, Zhihui Li, and Xiaojun Chang. Selongvlm: Empowering long video language models with self-corrective clip selection. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–16, 2026. doi: 10.1109/TPAMI.2026.3673141.

Yiman Zhang, Ziheng Luo, Qiangyu Yan, Wei He, Borui Jiang, Xinghao Chen, and Kai Han. Omnieval: A benchmark for evaluating omni-modal models with visual, auditory, and textual inputs. arXiv preprint arXiv:2506.20960, 2025.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

Ziwei Zhou, Rui Wang, and Zuxuan Wu. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2025.

- A AUDIO-VISUAL TASK TAXONOMY

Table 12: Ten audio-visual tasks covered by OmniVideo-100K and OmniVideo-Test.

Cognitive Level Task Definition Alignment

Fine-Grained Perception Perceive synchronized cross-modal details from given unimodal cues. Scene Transformation Detection Identify synchronized visual shifts using audio cues.

Understanding

Context Understanding Identify semantic associations between audio and visual elements. Comparison Compare target states or attributes across different timestamps. Sentiment Analysis Analyze character inner states via audio-visual cues. Event Sequence Ordering Restore the correct temporal order of jumbled events. Summarization Generate an audio-visual summary for specific events.

Reasoning

Causal Reasoning Infer underlying causes of specific events. Future Prediction Predict upcoming plot developments from current content. Hypothetical Reasoning Propose “What-if” scenarios to re-interpret video content.

- B DETAILED EXPERIMENTAL SETTINGS AND RESULTS

- Table 13: Parameter settings for full-parameter fine-tuning of VITA-1.5, Qwen2.5-Omni-7B and Qwen3-Omni-30B-A3B-Instruct.

Models Max Pixels FPS Max Frames Epochs Batch Size Learning Rate Warmup Ratio

VITA-1.5 default default default 1 32 1 × 10−5 0.03 Qwen2.5-Omni 448 × 448 1.0 180 1 32 1 × 10−5 0.1 Qwen3-Omni 256 × 256 1.0 150 1 32 5 × 10−6 0.1

- Table 14: Performance comparison on OmniVideo-Test. FGP: Fine Grained Perception, STD: Scene Transformation Detection, CU: Context Understanding, CP: Comparison, SA: Sentiment Analysis, ESO: Event Sequence Ordering, SM: Summarization, CR: Causal Reasoning, FP: Future Prediction, HR: Hypothetical Reasoning.

Models Size Overall FGP STD CU CP SA ESO SM CR FP HR

Gemini-3.1-Pro - 83.96 78.33 89.29 87.04 83.64 74.55 86.67 94.12 82.69 70.27 95.24 VITA-1.5 7B 40.99 30.00 26.79 46.30 49.09 47.27 35.00 35.29 48.08 43.24 52.38 Qwen2.5-Omni 7B 42.77 38.33 41.07 40.74 58.18 40.00 45.00 50.00 34.62 35.14 45.24 video-SALMONN 2+ 7B 45.15 45.00 30.36 51.85 50.91 41.82 46.67 58.82 34.62 45.95 52.38 OmniVinci 9B 47.13 45.00 42.86 53.70 56.36 49.09 43.33 58.82 38.46 32.43 52.38 MiniCPM-o 4.5 9B 55.25 55.00 58.93 74.07 50.91 43.64 61.67 61.76 40.38 54.05 52.38 Qwen3-Omni 30B 49.70 40.00 46.43 59.26 65.45 52.73 45.00 52.49 48.08 37.84 47.62 Uni-MoE-2.0-Omni 30B 46.93 48.33 30.36 68.52 69.09 45.45 30.00 52.94 48.08 43.24 33.33 OmniVideo-7B(VITA-1.5) 7B 61.58 55.00 64.29 64.81 65.45 63.64 53.33 73.53 65.38 62.16 52.38 OmniVideo-7B(Qwen2.5-Omni) 7B 60.59 61.67 64.29 61.11 63.64 63.64 60.00 64.71 50.00 56.76 59.52 OmniVideo-30B(Qwen3-Omni) 30B 63.56 60.00 60.71 70.37 67.27 65.45 61.67 73.53 57.69 59.46 61.90

- C PROMPTS FOR THE OVERALL PIPELINE

|Role<br><br>You are an AI expert in video content analysis.<br><br>Task Identify the main active entities (people, animals and objects) in the provided video and return the findings as a single, clean, and strictly valid JSON array. Instructions Follow these steps meticulously:<br><br>1. Process the entire video, including audio, to grasp its core content.<br>2. Strict Entity Filtering:<br><br>- Your list must only include entities that are critical to the video's narrative or primary action.<br>- You must ignore all minor elements, such as static backgrounds or irrelevant passersby.<br><br><br>3. Data Extraction for Each Entity:<br><br>- entity: Assign a descriptive but unique name.<br>- description: Write a brief, factual, and distinguishing description.<br>- label: Assign exactly one of the following lowercase strings: people, animal, object.<br><br><br>4. JSON Formatting: Your response must be a single, valid JSON array. Do not include any explanatory text, comments, or markdown formatting. Output Format [<br><br><br>{<br><br>"entity": "<name>", "description": "<clear and factual description>", "label": "<people/animal/object>"<br><br>}<br><br>]|
|---|

Figure 4: Prompt for instructing MLLMs to identify main active entities in the video.

|Role<br><br>You are an AI expert in video content analysis.<br><br>Task Generate a brief, high-level summary of the provided video. The description must be based only on information from both the video's visuals (what is seen) and its audio (what is heard). Constraints<br><br>1. You MUST NOT include any speculation.<br>2. You must use the exact name from the "Main Entities List" when referring to any listed entity. Main Entities List: {MAIN_ENTITIES} Output Format Directly provide the factual summary in a single, concise paragraph.<br>|
|---|

Figure 5: Prompt for instructing MLLMs to generate a video summary.

|Role<br><br>You are a professional transcriptionist AI, specialized in creating structured, machine-readable<br><br>transcripts. Task Generate a timestamped transcript of the audio. The output MUST be either a single, valid JSON array of objects or the literal value 'None'. Instructions<br><br>1. Segmentation: Break the speech into meaningful segments, such as sentences or complete phrases. Critically, a new segment must be started whenever the speaker changes.<br>2. Timestamp: Timestamps should mark the beginning of the first word and the end of the last word in the segment. The string format MUST be strictly "MM:SS" (minutes:seconds).<br>3. Content Purity: Transcribe ONLY spoken words. Omit all non-speech sounds (e.g., [music], [applause], [cough]). Constraints If no discernible speech is detected in the entire audio, your output MUST be the single literal value 'None', without any JSON formatting or quotes. Output Format (if speech is detected) [<br><br><br>{<br><br>"start_time": "00:01", "end_time": "00:04", "text": "This is the first sentence.",<br><br>}<br><br>]<br><br>Output Format (if no speech is detected) None|
|---|

### Figure 6: Prompt for instructing MLLMs to generate transcriptions for speech.

|Role<br><br>You are an AI Acoustic Event Logger. Task<br><br>Your task is to identify all sound events, excluding discernible dialogue, and generate a strictly objective, chronological log. The output MUST be either a single, valid JSON array of objects or the literal value 'None'.<br><br>Constraints<br><br>1. You must only describe the sound itself using general, objective categories. Strictly avoid inferring the source, cause, intent, or context. Specially, If a sound is identified as music, you must provide a more detailed description.<br><br>- Correct (Objective Description): "Glass breaking sound", "Footsteps", "Metallic impact", "Dog bark", "Laughter", "Heavy breathing sound"<br>- Incorrect (Inferred Context): "A window shattered", "Someone is running", "A wrench was dropped", "He is scared"<br><br><br>2. Do not log human dialogue. However, non-dialogue human sounds (like a laugh, cough, cry) should be logged.<br>3. Timestamps should mark the beginning and the end of the sound. The string format MUST be strictly "MM:SS" (minutes:seconds).<br>4. If no non-dialogue sound events are detected in the entire video, your output MUST be the single literal value 'None', without any JSON formatting or quotes. Output Format (if sounds are detected) [<br><br><br>{<br><br>"start_time": "00:15", "end_time": "00:16", "sound": "Metallic impact"<br><br>}<br><br>] Output Format (if no sounds are detected) None|
|---|

### Figure 7: Prompt for instructing MLLMs to identify non-speech sounds.

|Role<br><br>You are an advanced AI assistant specializing in Audio-Visual Speaker Diarization. Your expertise lies in combining vision and audio analysis to accurately identify speakers. Task Your primary task is to process a video and its corresponding JSON transcript. For each segment in the transcript, you must add a new speaker field and assign it the correct speaker's name. This identification MUST be based on a synthesized analysis of both the video's visual and audio track. Transcript {TRANSCRIPT}<br><br>Instructions<br><br>1. Focus on the Time Segment: For each object in the input JSON, precisely locate the segment in the video using its start_time and end_time.<br>2. Multi-Modal Speaker Identification: Within each time segment, perform the following analysis:<br><br>- Visual Analysis: Observe the video. Whose lip movements are synchronized with the spoken words (text)?<br>- Audio Analysis: Listen to the voice. Note its unique characteristics (pitch, tone, cadence) to help distinguish between speakers.<br>- Synthesize and Decide: Combine the visual and auditory evidence. Visual confirmation of synchronized lip movement is the strongest signal for attribution. If the speaker is off-screen, rely solely on voice identification.<br><br><br>3. Create and Assign Speaker Field: Based on your identification, add a new key-value pair, "speaker": "<name>", to the object.<br><br><br>- If the identified speaker is on the "Main Entities List", you MUST use their exact name.<br>- If you can consistently identify a speaker who is not on the list, assign a consistent label.<br>- If visual and audio cues are conflicting or insufficient for a confident identification, you MUST use the string "UNKNOWN". Main Entities List: {MAIN_ENTITIES} Constraints<br><br><br>1. You MUST NOT modify the start_time, end_time, or text fields.<br>2. The output MUST be a single, valid JSON array of objects, representing the completed transcript. Do not add any text or explanations outside of the JSON structure. Output Format [<br><br><br>{{ "start_time": "00:01", "end_time": "00:04", "text": "This is the first sentence.", "speaker": <name/UNKNOWN><br><br>}} ]|
|---|

### Figure 8: Prompt for instructing MLLMs to identify speakers based on audio-visual cues.

|Role<br><br>You are a professional video content analyst, specializing in creating exceptionally detailed and strictly objective visual descriptions. Task Generate a detailed, objective, and chronologically ordered description of the provided video's visual content. Your analysis must be strictly limited to what is directly observable on screen. You are forbidden from making any assumptions, interpretations, or inferences. Instructions For every new shot or significant scene change, you must internally analyze the visual information by<br><br>considering these four aspects. You will then synthesize these details into a single descriptive<br><br>paragraph as described in the Output Format.<br><br>- Setting & Environment: Location, objects, props, and lighting.<br>- Characters & Objects: Positioning, posture, and appearance.<br>- Actions & Interactions: Individual movements and interactions.<br>- Cinematography: Shot type, camera angle, and movement. Constraints<br><br><br>1. Exhaustive Detail: Capture all available visual information, including textures, colors, and background elements.<br>2. Prohibition of Inference: You are strictly forbidden from inferring relationships ("friend"), internal states ("he thinks..."), or off-screen events (sounds, dialogue).<br>3. You must use the exact name from the "Main Entities List" when referring to any listed entity. Main Entities List: {MAIN_ENTITIES} Output Format<br><br><br>- Your entire output must be a single, cohesive text composed of descriptive paragraphs.<br>- Each new shot or significant scene change must be described in a new paragraph.<br>- You MUST NOT use bullet points or any of the analytical headings (e.g., 'Setting & Environment') in your final output.<br>|
|---|

### Figure 9: Prompt for instructing MLLMs to generate visual descriptions.

|Role<br><br>You are an expert Video Content Analyst, specializing in Causal Chain Analysis. Your expertise lies in<br><br>uncovering the root causes of events by synthesizing disparate elements-dialogue, sounds, actions, and visual cues-from across an entire video. Task Analyze the provided detailed video description to exhaustively uncover all hidden or complex causal relationships. You will do this by identifying and explaining groups of interconnected video segments that collectively establish a clear cause-and-effect chain. You will demonstrate that the true reason for a key event or outcome is only understood by linking the audio of one segment with the visuals of another (or multiple others). Input You will be provided with the following information derived from a video:<br><br>Video Summary:<br><br>{VIDEO_SUMMARY} Main Entities: {MAIN_ENTITIES} Segments: {SEGMENTS} Instructions<br><br>1. Identify All Hidden Causal Chains: Meticulously scan the entire video description to pinpoint all events or outcomes whose root causes are not immediately apparent. Focus on cause-and-effect relationships that are revealed non-linearly or implicitly.<br>2. Group All Causally-Linked Segments: For each causal chain you identify, find and list all relevant video segments that, when combined, connect a cause to its effect. These segments, viewed together, should build a complete picture of why a specific outcome occurred.<br>3. Analyze the Cross-Modal Synthesis: For each group, your analysis must prove that the causal link is only correctly and fully understood when audio elements from one or more segments are combined with visual elements from another one or more segments. Constraints<br><br><br>1. Your analysis must be exhaustive. You are required to identify all distinct groups that meet the criteria within the entire video, ensuring that every significant causal chain is accounted for.<br>2. Each group you identify must include at least two non-consecutive video segments.<br>3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation. Output Format Format your response precisely as follows for each group you identify: Associated Group [Number]: [A concise title describing the group's central theme] Relevant Segments: [Timestamp of the first relevant segment] [Timestamp of the second relevant segment] [Add all other relevant timestamps as needed] Analysis of Connection: [Detail precisely how audio elements from one or more segments and visual elements from another one or more segments interact to reveal the complete causal chain.]<br>|
|---|

### Figure 10: Prompt for Global Clue Mining of Clue-Guided Generation (Causal Reasoning).

|Role<br><br>You are an expert analyst specializing in multimodal media interpretation and causal reasoning. Your<br><br>expertise lies in integrating visual and auditory information from videos (both presented in text format) to understand the cause-and-effect relationships between events. Task Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a highly insightful Q&A pair focused on causal reasoning. Input You will be provided with the following information extracted from a video: Detailed Script: {SEGMENTS}<br><br>Designated Segments:<br><br>{DESIGNATED_SEGMENTS} Connections {CONNECTIONS} Instructions<br><br>1. Internalize the Causal Chain: Examine the Designated Segments, referring to the provided connections, fully grasp the cause-andeffect relationship that has been identified. Pinpoint the "cause" and the "effect".<br>2. Formulate a Precise Causal Question: Based on the identified chain, craft a single question that asks why the event in the "effect" happened. The question must be unanswerable without referencing the events, dialogue, or visuals in the "cause" segments.<br>3. Provide Integrated Answers: In the answer, you must clearly use evidence from the different designated segments-both visual and auditory-and explicitly state how these clues string together to form a complete, logical explanation. Constraints:<br><br><br>1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.<br>2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.<br>3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.<br>4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer. Output Format Please adhere strictly to the following format, without any additional preamble or conclusion: Q: [Your Question] A: [Your Answer]<br>|
|---|

### Figure 11: Prompt for Locally Focused Generation of Clue-Guided Generation: generating openended QA pairs (Causal Reasoning).

|Role<br><br>You are an expert specializing in Multimodal Video Understanding, specializing in constructing<br><br>challenging multiple-choice questions (MCQs) that strictly test a user's understanding ability. Task Analyze the provided textual description of a video (detailed script, a designated set of segments, and auxiliary analysis) to generate a high-difficulty {TASK_TYPE} Multiple Choice Question (MCQ). {TASK_TYPE} Definition: {TASK_DEFINITION} Example: {EXAMPLE} Input<br><br>Detailed Script: {SEGMENTS} Designated Segments:<br><br>{DESIGNATED_SEGMENTS}<br><br>Connections {CONNECTIONS} Instructions<br><br>1. Analyze & Locate: Examine the Designated Segments, referring to the provided connections, to identify the specific Audio cues and Visual cues. Identify the Core Conclusion that satisfies the {TASK_TYPE} from these cues.<br>2. Construct Question: Based on the Core Conclusion, write a precise question {TASK_DEFINITION}. Do NOT include phrases like "Based on the audio...", "Considering the visual context...", or "In the final scene...". Do NOT refer to any "provided analysis", "evidence", or the process of deduction. It must be a direct question about the events in the video.<br>3. Generate Concise Options: Create 1 Correct Option (A) and 3 Hard Distractors (B,C,D) for your new question. Options must contain ONLY the answer label. Do NOT explain the reason or cite evidence within the option text. To ensure the MCQ is challenging, you must avoid random or obviously false distractors. You can refer to the following strategies to create "Hard Negatives":<br><br><br>- The Plausible Inference Trap (Over-Interpretation): Create an option that logically fits the scene context and seems reasonable, but is not explicitly supported by the evidence.<br>- Missing Modality (The Half-Blind): Construct an option that would be correct if you only relied on the Visuals (ignoring Audio) or only relied on the Audio (ignoring Visuals).<br>- The Semantic Drift (Nuance Distortion): Use a label that is closely related to the truth but differs in Intensity, Duration (State vs Trait), or Social Nuance (e.g., "Arrogance" vs "Confidence").<br>|
|---|

### Figure 12: Prompt for Locally Focused Generation of Clue-Guided Generation: generating multiplechoice questions.

|4. Validation & Iteration Loop (Strict Enforcement):<br><br>Before producing the final output, perform the following "Solvability Tests" on your draft. If the draft<br><br>fails ANY check, you MUST regenerate the question and options.<br><br>- Check 1: The Blind Test (Common Sense) Can the correct answer be guessed solely by reading the Question and Options without seeing the video?<br>- Check 2: The Simple Match Test Can the answer be found by simply matching a keyword from the video or a single visual frame?<br>- Check 3: The Unimodal Trap Is the answer derivable from Audio alone or Visual alone? Constraints<br><br><br>1. Fixed Position: Option A MUST always be the Correct Answer.<br>2. Uniqueness: Ensure the correct answer is unambiguously correct based on the provided video.<br>3. Conciseness: Avoid wordy phrases. Options must contain only the necessary differentiating keywords to remain concise.<br>4. Style & Length Uniformity: All four options (the correct one and the three distractors) must be of similar word count, sentence structure, and complexity.<br>5. No Timestamps or Time Ranges: Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.<br>6. JSON Formatting: Your response must be a single, valid JSON object. Do not include any explanatory text, comments, or markdown formatting. Output Format {{<br><br><br>"evidence": "Identify the Visual Clue and the Audio Clue from the script.",<br><br>"question": "Your reformulated question that requires combining these cues.", "options": {{<br><br>"A": "Correct Answer",<br>"B": "Distractor",<br>"C": "Distractor",<br>"D": "Distractor"<br><br><br>}}, "explanation": "Briefly explain why the correct option is correct, and how the other incorrect<br><br>options were derived." }}|
|---|

### Figure 12: Prompt for Locally Focused Generation of Clue-Guided Generation: generating multiplechoice questions (continued).

