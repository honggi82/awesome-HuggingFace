# arXiv:2602.06139v1[cs.CV]5Feb2026

## EgoAVU: Egocentric Audio-Visual Understanding

Ashish Seth1,2,∗, Xinhao Mei1, Changsheng Zhao1, Varun Nagaraja1, Ernie Chang1, Gregory P. Meyer1, Gael Le Lan1, Yunyang Xiong1, Vikas Chandra1, Yangyang Shi1, Dinesh Manocha2, Zhipeng Cai1,†

1Meta, 2University of Maryland, College Park ∗Work done during intership at Meta, †Project Lead

Understanding egocentric videos plays a vital role for embodied intelligence. Recent multi-modal large language models (MLLMs) can accept both visual and audio inputs. However, due to the challenge of obtaining text labels with coherent joint-modality information, whether MLLMs can jointly understand both modalities in egocentric videos remains under-explored. To address this problem, we introduce EgoAVU, a scalable data engine to automatically generate egocentric audio-visual narrations, questions and answers. EgoAVU enriches human narrations with multimodal context and generates audiovisual narrations through cross-modal correlation modeling. Token-based video filtering and modular, graph based curation ensure both the data diversity and quality. Leveraging EgoAVU, we construct EgoAVU-Instruct — a large scale training dataset of 3M samples, and EgoAVU-Bench — a manually verified evaluation split covering diverse tasks. EgoAVU-Bench clearly reveals the limitation of existing MLLMs: they bias heavily towards visual signals, often neglecting audio cues or failing to correspond audio with the visual source. Finetuning MLLMs on EgoAVU-Instruct effectively solves this issue, enabling up to 113% performance improvement on EgoAVU-Bench. Such benefit can also transfer to other benchmarks such as EgoTempo and EgoIllusion, achieving up to 28% relative performance gain. Code will be released to the community.

Date: February 9, 2026 Correspondence: Ashish Seth: aseth125@umd.edu, Zhipeng Cai: czptc2h@gmail.com Project Page: https://cs20s030.github.io/EgoAVU/ Code: https://github.com/facebookresearch/EgoAVU Data: https://huggingface.co/datasets/facebook/EgoAVU_data

1 Introduction

Egocentric videos capture rich and dynamic first person audio–visual information centered on daily human activities, such as cooking, painting, or assembling objects (Grauman et al., 2022; Nasirimajd et al., 2023; Grauman et al., 2024). Understanding such data plays a vital role in embodied intelligence and mixedrealities (Martins et al., 2023; Nagarajan et al., 2023; Puig et al., 2020; Savva et al., 2019).

The highly dynamic camera motion and the limited field of view make comprehensive egocentric video understanding challenging with solely visual cues (Chen et al., 2024a). This motivates the use of audio information, which provides persistent contextual signals tied to ongoing events. Though recent multi-modal language models (MLLMs) can accept both audio and visual inputs (Xu et al., 2025b,a; Team et al., 2024; Han et al., 2023; OpenBMB, 2024), whether they understand the joint dynamic of audio–visual signals in egocentric videos remains an open question.

The main bottleneck to study this problem arguably is limited data. For training, existing egocentric datasets, such as MultiHop-EgoQA (Chen et al., 2025) and MM-Ego (Ye et al., 2024), are derived mostly from the human narrations in Ego4D (Grauman et al., 2022). While these narrations provide valuable human supervision, they bias toward describing human–object interactions and lack broader environmental context or the diversity of auditory signals in egocentric recordings. For evaluation, existing benchmarks (Mangalam et al., 2023; Cheng et al., 2024a; Plizzari et al., 2025) focus mainly on visual cues, limiting their ability to assess integrated audio–visual reasoning. Though exocentric audio–visual benchmarks (Yang et al., 2022; Li et al., 2024b; Sung-Bin et al., 2024; Ma et al., 2024) exist, their multi-modal dynamics are fundamentally different.

###### Benchmarking MLLMs on EgoAVU-Bench

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Question (SSA): Identify the distinct sounds occurring between 320 seconds and 330 seconds, and describe the visible sources responsible for each in the video.

###### ...

...

Qwen2.5 Omni

At 320 seconds, you hear a \"click\" sound. It could be from a switch or a button being pressed. At 330 seconds, there's a \"squeak\" sound. This might be from a door hinge ...

[Figure 5]

[Figure 6]

###### ...

###### VideoLLaMA2

The video shows a person cutting up various ingredients in a kitchen. The distinct sounds from 320 seconds and 330 seconds are the crushing of of garlic and the chopping of of vegetables. These sounds are likely caused by the person using knife...

00:30 00:40 05:50 06:00

###### Egocentric Narrations

[Figure 7]

###### Ours

- 05:50 #C C held a knife
- 06:00 #C C cutting a orange

...

00:30 #C C opened the drawer 00:35 #C C picked a bag

During the segment, two distinct sounds were heard. First, a tapping or impact sound occurred when the person cut the pepper with the knife. Next, a water running sound was produced when the person turned on the tap.

| | | |
|---|---|---|
|En|hancing Egocentric Narrations with EgoAVU| |

###### Qwen2.5 Omni VideoLLaMA2 Ours

Athepersonrefrigeratorin a kitchen, revealingopenedvarious the itemsdrawersuchof QA Generation as a bag of onions, lettuce, and other vegetables. They picked up a shopping bag, specifically a white nylon containing lettuce, and placed it on the counter, producing a rustling and crinkling sound. The person then picked up another bag, this time containing onions, and put it on the counter as well. In the process, they dropped the shopping bag, causing a thumping sound ...

A person in a kitchen held an orange and a knife while watching something on a laptop on the counter. The kitchen had various items, including a sink, a stove with a pot, a cutting board, and a wall with a window. As they dropped the knife on the chopping board, it made an tapping sound. The person then started cutting the orange, creating a chopping sound. Nearby, a kitchen sink, laptop, plastic bags, bottle of oil, and other items were visible but not interacted with.

SSA

[Figure 8]

EgoAVU-Bench (3K QAs)

EgoAVU-Instruct (3M QAs)

AVDN

...

AVH

- (1) Source-Sound Association (SSA)
- (2) Temporal Reasoning (TR)
- (3) Audio-Visual Dense Narration (AVDN)
- (4) Audio-Visual Segment Narration (AVSN)
- (5) Audio-Visual Hallucination (AVH)

TR AVSN

Actions

Objects Sounds

- Figure 1 Overview of EgoAVU. We introduce EgoAVU, a scalable and automated data engine to enable egocentric audio–visual understanding. EgoAVU enriches existing egocentric narrations by integrating human actions with environmental context, explicitly linking visible objects and the sounds produced during interactions or surroundings. Leveraging this pipeline, we construct EgoAVU-Instruct (3M QAs) and EgoAVU-Bench (3K verified QAs), enabling systematic training and evaluation of MLLMs. Models finetuned with EgoAVU-Instruct exhibit high audio-visual grounding in egocentric settings.

To address these problems, we introduce EgoAVU, a fully automated data engine that can generate diverse and high quality audio-visual-language data from public egocentric datasets such as Ego4D (Grauman et al., 2022). EgoAVU comprises four key components: (i) Enhancing egocentric narrations by enriching human descriptions with environmental context, visual object details, and audio captions generated using diverse open-source MLLMs (Bai et al., 2025; Xu et al., 2025a; AI@Meta, 2024); (ii) Filtering videos with rich audio–visual dynamics by selecting egocentric clips featuring varied action sequences, human–object interactions, and a wide range of ambient and foreground sounds; (iii) Generating fine-grained event captions through audio–visual correlation by integrating modality-specific cues such as actions, objects, and sounds, and modeling their relationships to enhance multimodal reasoning; and (iv) Curating diverse audio–visual understanding tasks that span grounding, temporal reasoning, scene understanding, and audio–visual hallucination.

Leveraging EgoAVU, we construct EgoAVU-Instruct — a large scale training dataset of 9K egocentric videos with 3M audio–visual-language samples, and EgoAVU-Bench — an evaluation split of 900 videos containing 3K manually verified samples. The generated data features strong multi-modal correspondence, long average video durations (4min), open and close-ended questions capturing diverse aspects of egocentric audio-visual understanding.

Comprehensive experiments show that existing MLLMs perform poorly on EgoAVU-Bench, revealing significant limitations in their joint audio–visual reasoning capabilities. The seven models we tested show consistent bias towards the vision modality, often neglecting audio cues or struggle to connect them to the correct visual source. Fine-tuning MLLMs such as Qwen2.5-Omni (Xu et al., 2025a) on EgoAVU-Instruct can effectively close this gap, resulting in up to 113% relative performance boost on EgoAVU-Bench. More importantly, the performance gain is transferrable to other egocentric benchmarks, such as EgoTempo (Plizzari et al., 2025) and EgoIllusion (Seth et al., 2025), achieving up to 28% relative performance improvement.

- 2 Related Work

Multimodal-Large Language Models. Recent advances in Multimodal Large Language Models (MLLMs) (Team et al., 2024; Achiam et al., 2023; Xu et al., 2025b,a; Wang et al., 2024; OpenBMB, 2024; Cheng et al., 2024b)

have substantially extended the capabilities of large language models (Achiam et al., 2023; AI@Meta, 2024; Yang et al., 2025) beyond text, enabling unified understanding of visual and auditory inputs. Despite this progress, existing MLLMs face critical limitations when applied to egocentric audio-visual understanding. First, leading models developed for egocentric video understanding, such as MM-Ego (Ye et al., 2024) and EgoVLPv2 (Pramanick et al., 2023), lack the ability to incorporate audio cues, limiting them to perform visual only tasks. Second, even models capable of handling viusal and auditory signals, including Qwen2.5-Omni (Xu et al., 2025a) and Video-LLaVA2 (Cheng et al., 2024b), remain primarily trained and benchmarked on exocentric audio-visual data (Yang et al., 2022; Geng et al., 2025). As a result, they struggle to generalize to egocentric settings, which exhibit fundamentally different characteristics such as dynamic camera motion, frequent self-occlusions, and distinct audio profiles. These limitations highlight a significant gap in current MLLMs’ to jointly understand audio–visual cues in egocentric videos.

Egocentric-Video Understanding. The field of egocentric video understanding has gained increasing attention due to its relevance in augmented reality (AR) (Nagarajan et al., 2023; Martins et al., 2023) and embodied AI (Savva et al., 2019; Puig et al., 2020). Large-scale datasets such as Ego4D (Grauman et al., 2022), EPIC-KITCHENS (Nasirimajd et al., 2023), and Ego-Exo4D (Grauman et al., 2024) have driven progress by enabling the creation of egocentric video–language corpora (e.g., MultiHop-EgoQA (Chen et al., 2025), EgoTextVQA (Zhou et al., 2025), QaEgo4D (Bärmann and Waibel, 2022)) and benchmarks (e.g., EgoSchema (Mangalam et al., 2023), EgoTempo (Plizzari et al., 2025), EgoIllusion (Seth et al., 2025)). However, these datasets are predominantly constructed from textual narrations, focusing on human–object interactions while overlooking environmental and auditory cues, critical for accurately inferring actions, contextual dynamics, and scene semantics in egocentric settings. Recent benchmarks such as EgoTempo and EgoIllusion attempt to enhance contextual understanding through visual captioning, but their reliance on closed-source models (e.g., Gemini (Team et al., 2024), GPT-4o (Achiam et al., 2023)) makes large-scale, reproducible data generation challenging. In this work, we address these problems by introducing a scalable data-engine capable of generating time-aligned joint audio-visual generation while utilizing open-source MLLMs (Xu et al., 2025a; Bai et al., 2025; AI@Meta, 2024).

### 3 Method

Fig.2 provides an overview of EgoAVU. We begin by collecting and organizing egocentric data (Sec.3.1). Next, we enhance egocentric narrations using MLLM-generated multi-modal context (Sec.3.2), and use the tokens of enriched narration to filter videos for temporal diversity (Sec.3.3). We then introduce a multi-modal context graph (MCG) that captures complex cross-modal relations, and parse these graphs with open-source LLMs to fuse multi-modal information into a single dense narration (Sec.3.4). Finally, this fused narration is used to generate QA pairs for joint audio-visual understanding (Sec.3.5). To ensure the scalability, EgoAVU only utilizes open source models. Please refer to Appendix A for example outputs and prompts for each module.

- 3.1 Data Collection

We begin by collecting videos from the Ego4D dataset (Grauman et al., 2022), and filter out videos that lack audio tracks. Each video in Ego4D is accompanied by a set of narrations that provide first-person descriptions of the events, such as “#C C holds a cup,” where “#C C” refers to the camera wearer. These narrations can be represented as {Nj,tj}Kj=1, where Nj denotes the narration text, tj is the corresponding timestamp, and K is the total number of narrations in the video. To obtain the temporal boundries of each narration, we use the strategy proposed in prior works (Plizzari et al., 2025; Lin et al., 2022), where temporal boundries Tj for narration Nj is defined as:

βi 2α

βi 2α

. (1)

Tj = tj −

,tj +

βi represents the average interval between consecutive timestamps ({Tj,Tj+1}), and α denotes the global average of βi across the entire dataset. Since the segments associated with each narration are short (on average 3s), they are insufficient to capture fine-grained visual and auditory information. Following (Di and Xie, 2024), for each video v, we group consecutive segments and their corresponding narrations to form video clips vj of at least 10s and not more than 360s.

###### (1) Egocentric Narration Enhancement

- (3) Generating Audio-Visual Narration
- (4) QA Generation

Generating Multimodal Context Graph

|[Figure 9]<br><br>| |Video Caption<br><br>Well, you can see|a person in a|
|---|---|---|---|
| | |kitchen cutting oranges with a knife. Plastic bags can be seen...| |
| | | | |

Audio-Visual Narrations

A person in a kitchen held an orange and a knife while watching something on a laptop on the counter. The kitchen had various items, including a sink, a stove with a pot,... The person then started cutting the orange, creating rhythmic tapping sound. A rustling sound can be heard in background. Nearby, a kitchen sink, laptop, ...items were visible but not interacted with.

###### Prompt

.... Identify:

Enhanced Narrations

|Video Caption|
|---|

Video Clip

- 1.Interacted Objects
- 2.Fore-ground SourceSounds Mapping.
- 3.Background Object and Sounds ...

Image Caption You can see person hands, kitchen sink, plastic bags, a laptop, Knife, orange, pot, ...

[Figure 10]

|Image Caption|
|---|

|Audio Caption|
|---|

Keyframe

|Egocentric Narration|
|---|

Audio Caption

Output a structured JSON:

LLM

Well, it seems like there's a sound of rhythmic tapping followed by rustling sound heared in the ...

[Figure 11]

Prompt

Audio Clip

Examples

... Generate one objective, detailed audiovisual narration describing the scene.

LLM

| | |
|---|---|
|Egocentric Narration| |
|[#C C opened the drawer, #C C holds the knife, #C C ...]| |

###### MCG

- 1. Parse the Multi-modal Context Graph
- 2.Integrate actions, objects, and sounds .. what caused them
- 3.Use video caption coherent narration ...

Multimodal Context Graph (MCG)

Enhanced Narration

(2) Video Filtering

Orange

Tapping sound

| |[Figure 12]|
|---|---|
| | |

Knife

###### Tokenization

###### Enhanced Narration

cut

MATTR

Laptop

Xm

EgoAVU-Instruct (3M QAs)

hold

Person

Example

Sink

Audio-Visual Narrations Xm

LLM

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]<br><br>...|
|---|

|[Figure 16]|
|---|

Actions Objects Sounds

###### ...

EgoAVU-Bench (3K QAs)

Rustling Sound

Pot

t = 0 sec t = 120 sec

t = 0 sec t = 120 sec

Task-Specific Prompt

Background Objects/Sounds

MATTR = 0.34 MATTR = 0.18

- Figure 2 EgoAVU pipeline. EgoAVU consists of four key components. (1) For each egocentric video clip, EgoAVU enhances the raw narration with detailed multisensory context using open-source MLLMs (Bai et al., 2025; Xu et al., 2025a). (2) These enriched narrations are then used to select clips that exhibit diverse audio–visual dynamics. (3) Next, EgoAVU constructs a Multimodal Context Graph (MCG), automatically generated via open-source LLMs (AI@Meta, 2024), to capture complex cross-modal relations. The MCG is parsed alongside the enhanced narrations to produce coherent audio–visual narrations. (4) The generated audio-visual narrations are leveraged to generate high-quality audio–visual QA pairs, forming both the instruction-tuning dataset EgoAVU-Instruct and the evaluation benchmark EgoAVU-Bench.

- 3.2 Narration Enhancement

After data collection, we generate descriptions about audio and video frames to enrich the original egocentric action-centric narrations. The goal of this stage is to obtain detailed descriptions for both modalities, which paves the way for diverse and fine-grained QA generation. The most straight-forward approach is to input both video frames and audio tracks into an MLLM, and prompt it to generate the audio and visual descriptions. However, our initial experiments show that MLLMs, when provided with audio and video inputs together, cannot capture important details due to modality bias and hallucination.

Specifically, we evaluate open-source models, including Qwen2.5-Omni (Xu et al., 2025a) and MiniCPMo (OpenBMB, 2024), on 200 randomly sampled video clips. Each model is first used in a uni-modal setting, captioning visual and auditory information independently, i.e., visual captioning without audio input and audio captioning without video input. We then perform joint audio–visual captioning by providing both modalities simultaneously. After that, we manually compare the consistency for both visual and audio modalities by computing the ratio where an object/event captured in the uni-modal output is also correctly appearing in the multi-modal output. I.e., whether the multi-modal output can capture all the details as in the uni-modal settting.

We observe a consistent pattern where models either omit various sounds or associate audio cues with incorrect visual events. For example, Qwen2.5-Omni shows an error rate of 54.3% for audio and 25.4% for visual consistency, while MiniCPM-o yields 68.2% and 31.2%, respectively (see Fig 10 for example). These findings show that existing MLLMs struggle to maintain accurate and detailed cross-modal grounding in egocentric contexts, motivating a modular data engine that leverages specialized models to process different modalities independently.

With this observation, we leverage a collection of MLLMs to extract rich multisensory information from individual modalities separately. Specifically, for each video segment, we first capture detailed spatial

descriptions of objects by applying an image captioner such as Qwen2.5-VL (Bai et al., 2025) to the center frame. To model temporal dynamics, including camera motion, action sequences, and auditory events, we employ Qwen2.5-Omni in two complementary modes. First, we use it as a video captioner, processing video frames without audio, to generate coherent video-level narrations. Then, we use the same model as an audio captioner without visual inputs to produce detailed auditory narrations that capture both foreground sounds closely tied to human activities (e.g., impacts and hissing) and background sounds (e.g., bird chirping, wind blowing). This process results in time-aligned uni-modal narrations that encapsulate both visual and auditory aspects of the scene.

- 3.3 Video Filtering for Diversity Enhancement

After obtaining the enhanced narrations, we use them to compute lexical diversity, filtering videos with rich and diverse visual and auditory signals. Specifically, for each video v, we combine the segment-level narrations into a single narration that captures all objects, actions, and sounds present in the video. We then tokenize this narration into Tv = {t1,...,tn} and compute the Moving-Average Type–Token Ratio (MATTR) (Covington and McFall, 2010), which averages the proportion of unique words across sliding windows of size w:

MATTR(Tv) =

1 n − w + 1

n−w+1

i=1

|Uni.(ti,...,ti+w−1)| w

. (2)

Higher MATTR scores indicate narrations that describe a wider variety of objects, actions, and auditory events. We retain videos whose MATTR exceeds a threshold τ = 0.3, effectively removing the bottom 25% of our distribution to filter static or repetitive descriptions, leading to the final video count of 9,900.

- 3.4 Audio-Visual Narration Generation

Our next objective is to generate a unified narration that captures both auditory and visual information for each segmented video clip. We do this leveraging text-based LLMs. When synthetically merging unimodal narrations, the LLM must first implicitly retrieve key details across modalities, such as which objects the human interacts with, which remain in the background, and what actions or objects produce specific sounds. It must then integrate this information into a coherent and perceptually grounded narration. However, our preliminary experiments show that directly prompting open-source LLMs such as LLaMA-70B (AI@Meta, 2024) often fails to maintain consistent human–object interactions and sound–source associations in their responses (See Appendix A.3 for detailed comparison).

We design a two-stage pipeline that addresses this challenge. First, we organize audio-visual cues from uni-modal narrations into a structured representation called the Multi-modal Context Graph (MCG). As shown in Fig. 2, MCG captures relationships between visible actions, objects, and audible sounds. Each MCG is generated by prompting an LLM (LLaMA-70B) with enhanced narrations to extract the following information :

- • Interacted Objects: Objects with which the person physically interacts, along with interaction types, inferred from action narrations.
- • Background Objects: Objects visible in the environment but never interacted with, identified by comparing objects and action narrations.
- • Foreground Sounds: Human-induced action sounds correlated with specific actions (e.g., impact sound

→ “C places the phone on the table”), or ambient sounds grounded in visible scene elements (e.g., “dog barking” aligned with a visible dog).

- • Background Sounds: Sounds present in the audio track whose sources cannot be visually grounded.

We find that MCG makes multi-modal relationships explicit and easily inferable, which we utilize further to generate joint audio-visual narrations.

Next, we leverage MCGs to guide the generation of unified, high-coherence audio–visual narrations. The goal is to fuse both visual and auditory details into the same multi-modal coherent narration. As shown in Fig 2, we provide the LLM (LLaMA-70B) with enhanced narrations and MCGs, and prompt it to generate the combined

Category Example

Open-Ended Source–Sound Association (SSA) What is the source of the sizzling sound heard in the video? Audio–Visual Segment Narration (AVSN) Between 240 and 250 seconds, describe the person’s surroundings, actions,

and the sounds that can be heard?

Audio–Visual Dense Narration (AVDN) Describe in detail what the person sees, hears, and does throughout the video. Close-Ended Temporal Reasoning (TR) What happened before the person opened a kitchen cabinet?. Choose the

correct option from the following options: ... Audio–Visual Hallucination (AVH) Is there a beeping sound coming from the microwave in the video?

##### Table 1 Question Prompts. Examples of open-ended and close-ended questions in EgoAVU-Instruct and EgoAVU-Bench.

###### Video Duration Distribution

250

2500

200

2000

150

1500

Count

Count

100

1000

50

500

0

0

min

min

min

min

min

min

min

min

min

min

min

min

0-1

1-2

2-3

3-4

4-5

5-6

0-1

1-2

2-3

3-4

4-5

5-6

Video Duration for EgoAVU-Bench

Video Duration for EgoAVU-Instruct

- Figure 3 Video duration distribution. Our videos includes both short clips within 1 min and long videos of 6 min.

Dataset A&V # Test Avg. Dur. QA-Type # Ans.

EgoTaskQA (Jia et al., 2022) ✗ 8k 25s Open 13 EgoSchema (Mangalam et al., 2023) ✗ 500 3 min Close EgoThink (Cheng et al., 2024a) ✗ 750 45s Open 4 EgoTempo (Plizzari et al., 2025) ✗ 500 45s Open 10 EgoIllusion (Seth et al., 2025) ✗ 8k 3 min Open+Close 3

EgoAVU-Bench(Ours) ✓ 3k 4 min Open+Close 200

Table 2 Benchmark statistics. Beside having QAs with high audio-visual coherence, EgoAVU-Bench features large number of QA pairs, longer egocentric videos with audio track, support both open/closed QA type and consists of significantly longer and descriptive responses.

narration. The prompt asks the LLM to first extract explicit cues from the MCG, including interacted and background objects, grounded sound events, and their associations with actions and visible sources, and then align these cues with the corresponding temporal descriptions from the video and action-level narrations.

- 3.5 QA Generation

Using the unified audio–visual narrations, EgoAVU synthesizes question–answer pairs that probe diverse aspects of egocentric audio-visual understanding.

EgoAVU Taxonomy. We design five categories of QAs encompassing both open-ended questions, where models must produce descriptive responses integrating visual and auditory cues, and closed-ended questions, which test precise multi-modal perceptual understanding through multiple-choice or binary (Yes/No) formats. A detailed description of each category is provided below.

- • Open-endedQAs. These questions are primarily free-form, requiring the model to develop a comprehensive understanding of temporal and spatial dynamics across visual and auditory cues in egocentric videos. The specific categories include: (1) Sound–Source Association (SSN): Identify various foreground sounds in the video and determine their corresponding visible sources including human actions or various objects shown in the video. (2) Audio–Visual Segment Narration (AVSN): Answer segment-level questions by producing coherent and natural narrations that describe what the person is doing, seeing, and hearing, including both foreground and background sounds, grounded within the specified temporal range. (3) Audio–Visual Dense Narration (AVDN): Extend previous task to the entire video, assessing the model’s ability to maintain narrative coherence across the complete video.
- • Close-ended QAs. These questions are particularly useful for adversarial testing. Close-ended QAs allow the construction of fine-grained distractors and counterfactuals to probe a model’s susceptibility to spurious correlations in audio-visual understanding. We design two main categories: (1) Temporal

###### Visual Scenarios Distribution

1400

1200

1000

800

Count

600

400

200

0

games Blacksmith

Talking Making coﬀee Gardening

Eating Playing Cards Construction Farmer Carpenter

Bike mechanic Walking

games

Making Bricks Playing

Practicing Instruments

Shopping

muting

Cooking

Cleaning

Painting

on street com

board

Playing

- Figure 4 Distribution of 20 most common visual scenarios in EgoAVU-Instruct and EgoAVU-Bench.

###### Task Distribution

1.6%

3.8%

20% 20%

40.1% EgoAVU-Instruct 40.1%

EgoAVU-Bench

20%

20%

20%

14.6%

AVDN AVSN TR SSA AVH

Figure 5 Distribution of proposed tasks across EgoAVUInstruct and EgoAVU-Bench.

Reasoning (TR): Multiple-choice questions with four options that assess the model’s understanding of temporal relationships among multi-modal events such as human actions, visual objects, and auditory cues in egocentric videos. These questions focus on reasoning about event ordering, for example identifying which event occurred first or last, or answering before/after queries. (2) Audio–Visual Hallucination (AVH): Binary (“Yes”/“No”) questions that evaluate the model’s tendency to hallucinate when verifying the presence of actions, objects, or sounds throughout the video.

- 3.6 Dataset Overview

Leveraging EgoAVU, we construct the first egocentric audio–visual dataset suite for training and evaluating MLLMs. Our large-scale instruction-tuning corpus, EgoAVU-Instruct, comprises approximately 3M samples with 9K egocentric videos, while the evaluation benchmark, EgoAVU-Bench, contains 3K QA pairs with 900 distinct videos. As shown in Fig. 4, both EgoAVU-Instruct and EgoAVU-Bench encompass a wide range of real-world visual scenarios, such as cooking, painting, and other indoor/outdoor activities. Extensive manual verifications are conducted on EgoAVU-Bench to ensure that the audio-visual information is correctly grounded. As illustrated in Fig. 5, both EgoAVU-Instructand EgoAVU-Bench covers all five question categories introduced earlier, featuring a balanced mix of open- and closed-ended formats. As shown in Fig. 3, the video length in both datasets vary from 1 to 6 minutes, providing rich temporal diversity for model training and evaluation.

- Table 2 compares existing egocentric benchmarks with our EgoAVU-Bench. Besides filling the gap of egocentric audio-visual understanding, EgoAVU-Bench includes a large number of QAs, features longer videos with synchronized audio tracks, supports both open- and closed-ended question formats, and provides significantly longer and more descriptive responses.

### 4 Results

Implementation Details. We verify the effectiveness of our instruction-tuning dataset, EgoAVU-Instruct, by fine-tuning MLLMs such as Qwen2.5-Omni (7B) on it and compare the finetuned model with baselines on EgoAVU-Bench. Fine-tuning is performed using LLaMA-Factory (Zheng et al., 2024), under two settings: LoRA (Hu et al., 2022) and full fine-tuning. All experiments are conducted on 64 H100 GPUs with a global batch size of 64, training each model for 5 epochs. To ensure consistent visual coverage across samples, during training, we uniformly sample 300 frames per video. We also uniformly sample data from each of the 5 tasks to achieve a balanced performance (See Appendix D for additional details).

Evaluation Protocol. For close-ended QAs in EgoAVU-Bench, we follow (Yue et al., 2024) and use regex-based string matching, where we construct robust regular expressions and design a response-processing module to extract key phrases such as option IDs (A, B, C, D), binary indicators (yes/no), and conclusion phrases from long responses for accurate answer matching. For open-ended QAs, similar to prior work (Plizzari et al., 2025),

Models Size SSA AVDN AVSN TR AVH

S (↑) S (↑) M (↑) R (↑) S (↑) M (↑) R (↑) Acc. (↑) Acc. (↑)

Open-Source MLLMs

VideoLLaMA2 (Cheng et al., 2024b) 7B 1.51 1.88 3.65 8.32 1.71 7.50 13.89 37.00 20.32 Baichuan-Omni (Li et al., 2024a) 7B 1.49 1.92 4.82 9.75 1.79 8.34 14.21 39.85 21.10 Intern-Omni (Chen et al., 2024b) 8.7B 1.47 1.95 5.20 10.11 1.82 8.69 14.70 41.22 21.75 Phi4-mm (Abouelenin et al., 2025) 8B 1.42 1.59 8.79 13.13 1.69 12.17 16.90 45.04 22.89 MiniCPM-o (OpenBMB, 2024) 8B 1.43 2.27 10.84 14.77 2.06 9.68 12.19 26.44 21.76 Qwen2.5-Omni (Xu et al., 2025a) 3B 1.45 2.17 8.55 13.45 1.85 8.63 13.08 46.40 26.28 Qwen2.5-Omni (Xu et al., 2025a) 7B 1.50 2.37 10.69 14.74 1.99 9.99 13.39 53.20 42.69

MLLM trained with EgoAVU-Instruct

Ours (LoRA) 7B 3.15 2.60 12.20 17.19 2.45 22.53 28.34 64.31 61.69 Ours (Full) 7B 3.20 2.66 12.50 17.32 2.63 22.68 28.70 67.84 60.12 ∆(%) – +113.3 +12.2 +16.9 +17.2 +27.6 +86.5 +69.8 +27.2 +30.8

- Table 3 Main result on EgoAVU-Bench. We compare seven MLLMs with joint audio–visual understanding capabilities against our fine-tuned models across a diverse set of tasks in EgoAVU-Bench, including open-ended QAs: Source-Sound Association (SSA), Audio-Visual Dense Narration (AVDN), and Audio-Visual Segment Narration (AVSN), as well as closed-ended QAs: Temporal Reasoning (TR) and Audio-Visual Hallucination (AVH). For the open-ended tasks, we report LLM-as-Judge (S), METEOR (M), and ROUGE-L (R). For the closed-ended tasks, we report Accuracy (Acc.). Additionally for each task, we compute the relative performance gain (∆) between the best open-source model and our fine-tuned models.

we adopt the LLM-as-a-judge approach, employing Qwen3-235B-A22B-Instruct-2507 (Yang et al., 2025) as an open-source judge for reproducibility. The model rates MLLM-generated responses on a 1–5 scale (see Appendix D for the evaluation prompt). We additionally report standard metrics used for dense response evaluation, including ROUGE-L (Lin, 2004) and METEOR (Banerjee and Lavie, 2005).

- 4.1 Main Results Table 3 presents the main result on EgoAVU-Bench. The key findings are summarized below.

- • MLLMs struggle to associate sounds with their visual sources. This is evident from their low scores on the Source-Sound Localization (SSA) task, where LLM-as-a-judge (S) evaluations remain below 1.6 out of 5 across all baseline MLLMs.
- • MLLMs struggle to produce coherent and temporally aligned audio-visual narrations. In both the Audio-Visual Dense Narration (AVDN) and Audio-Visual Segment Narration (AVSN) tasks, even the best-performing model, Qwen2.5-Omni (7B), attains S scores of below 2.4, which is consistent with low caption quality metrics such as ROUGE-L (R) and METEOR (M).
- • MLLMs demonstrate limited temporal reasoning over joint audio-visual inputs. The highest Temporal Reasoning (TR) accuracy, achieved by Qwen2.5-Omni (7B), is merely 53.2%, with models such as VideoLLaMA2 and MiniCPM-o performing considerably worse.
- • MLLMs frequently hallucinate during audio-visual reasoning, as reflected by their below 43% accuracy on the Audio-Visual Hallucination (AVH) task.
- • Finetuning MLLMs on EgoAVU-Instruct effectively improves egocentric audio-visual understanding. Our finetuned model yields substantial and consistent performance gains across all tasks. Compared to the best performing baseline, we achieve up to 113.3% and 44.5% relative performance improvement on open and close ended tasks respectively. In addition, LoRA and full finetuning both provides considerable performance gain. This shows the possibility to achieve strong audio-visual understanding under resource-limited scenarios.

- Figure 6 compares the responses of VideoLLaMA2 and Qwen2.5-Omni (7B), with our fine-tuned model on both open-ended and close-ended tasks. In close-ended tasks like Audio-Visual Hallucination, when queried about a sound source absent from the video, Qwen2.5-Omni often fabricates a visually plausible

|[Figure 17]<br><br>|
|---|

|[Figure 18]<br><br>|
|---|

|[Figure 19]<br><br>|
|---|

Models Size Action Object Sound

Acc. (↑) Acc. (↑) Acc. (↑)

...

...

VideoLLaMA2 7B 21.36 20.32 19.27 Phi4-MM 8B 22.45 21.74 24.49 MiniCPM-o 8B 24.49 24.46 16.33 Qwen2.5-Omni 3B 25.00 29.35 24.49 Qwen2.5-Omni 7B 44.39 50.00 33.67

Audio-Visual Hallucination (AVH)

Determine if the shattering sound was caused by a person’s movement.

|Qwen2.5 Omni| |
|---|---|
| | |
|Well, it could be that someone accidentally dropped a glass or a ceramic dish. You know, sometimes in the kitchen, things can slip and fall. Maybe it was a bit of a clumsy moment. What do you think?| |

Ours (LoRA) 7B 60.20 61.09 63.78 Ours (Full) 7B 61.32 62.40 64.20

| | |
|---|---|
|Ours| |
|None, as no shattering sound was heard| |

Table 4 Error Analysis for Audio-Visual Hallucination (AVH).

Audio-Visual Segment Narration (AVSN)

Models Size Action Object Sound

Between 190 and 200 seconds, describe the person’s surroundings, actions, and the sounds that can be heard?

Acc. (↑) Acc. (↑) Acc. (↑)

|VideoLLaMA2| |
|---|---|
| | |
|The video shows a person preparing food in a kitchen. The main source of sound in the video is the crushing of of vegetables as the person chops them up.| |

VideoLLaMA2 7B 45.88 38.38 38.30 Phi4-MM 8B 48.81 48.48 46.74 MiniCPM-o 8B 40.48 25.00 37.50 Qwen2.5-Omni 3B 44.44 69.70 31.91 Qwen2.5-Omni 7B 43.53 64.65 36.17

| | |
|---|---|
|Ours| |
|A person wearing a white and black striped shirt stood in a kitchen with ... They held a blue plastic bag in their left hand. The person start opening the bag, producing a rustling sound. Nearby, a kitchen sink, dish rack, spoon, knife, ... were visible but not interacted with.| |

Ours (LoRA) 7B 54.31 68.90 52.45 Ours (Full) 7B 55.29 69.80 53.17

Figure 6 Qualitative Analysis on EgoAVU-Bench.

Table 5 Error Analysis for Temporal Reasoning (TR).

Model EgoTempo Acc. (↑) EgoIllusion Acc. (↑) EgoSchema Acc. (↑) VideoMME Acc. (↑) AVQA Acc. (↑) Qwen2.5-Omni 16.25 56.32 67.43 73.0 89.4

Ours (LoRA) 20.83+28.1% 60.36+7.2% 67.34−0.1% 72.4−0.01% 89.7+0.003% Ours (Full) 20.21+24.4% 60.24+7.0% 66.21−1.8% 72.0−0.01% 89.5+0.001%

Table 6 Results on Egocentric and Exocentric benchmarks. Finetuning on EgoAVU-Instruct benefits other egocentric benchmarks such as EgoTempo and EgoIllusion, achieving up to 28.1% accuracy gain. Our model also maintains strong performance on exocentric video QA benchmarks such as VideoMME and AVQA.

yet non-existent source, whereas our model effectively resists such misleading prompts through improved audio–visual grounding. For open-ended tasks such as Audio-Visual Segment Narration, our model exhibits stronger sound–source coherence and more accurate action-sequence understanding. Fig. 7 further shows examples on audio-visual dense narration (AVDN). Unlike our fine-tuned model, existing MLLMs produce sparse descriptions for Audio-Visual Dense Narration task in EgoAVU-Bench and further overlook audio cues or fail to ground sounds to their sources.

- 4.2 Evaluating Existing Egocentric Benchmarks

To further evaluate the generalizability of our fine-tuned model, we report its performance on additional egocentric video–language benchmarks, including EgoTempo, EgoSchema, and EgoIllusion. As shown in Table 6, fine-tuning on EgoAVU-Instruct leads to notable accuracy improvements on EgoTempo and EgoIllusion, with gains of up to 28.1%, while maintaining competitive performance on EgoSchema, showing only a marginal decrease of 0.1%. Note that LoRA performs slightly better than full finetuning in terms of improving other datasets. These results demonstrate that training on EgoAVU-Instruct enhances audio-visual understanding without causing overfitting, and further complements performance across diverse egocentric video QA tasks. We also evaluate our finetuned model on non-egocentric datasets.

In addition to egocentric benchmarks, we evaluate our fine-tuned model on popular exocentric Video QA benchmarks, including VideoMME (Short duration split w/o subtiles) (Fu et al., 2024) and AVQA (Yang et al., 2022). As shown in Table 6, despite being fine-tuned exclusively on egocentric QAs, our model almost retains its original performance on VideoMME and slightly outperforms the base model Qwen2.5-Omni on

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

###### ... ... ... ...

| |Audio-Visual Dense Narration (AVDN)| |
|---|---|---|
|Question: Provide a detailed account of what does the|person sees, does and hear throughout th|e video?|
|VideoLLaMA2<br><br>A man is working on a building. He is applying cement t background.|o the wall using a tool. While he works, th|ere are sounds of traffic and people talking in the|
|Qwen2.5-Omni<br><br>In the video, there's a guy in a pink shirt working on a can see another person in a red shirt standing a bit fu materials like wooden planks and other debris scattere in the pink shirt seems really focused on his work, care think about construction work?|wall. He's using a trowel to smooth out th rther away, just watching. The ground aro<br><br>d about. There are also some trees and a efully applying the trowel to the wall. It lo|e surface. There are some pipes on the wall too. You und them is a bit messy, with a lot of construction road with passing vehicles in the background. The guy oks like a typical construction scene. What do you|
|Ours<br><br>A man in a pink shirt stood on a ladder, using a trowel person in a plaid shirt watched him work from the side man in a red shirt stood on the ground, observing the w<br><br>materials like wooden planks and metal rods. As the smoothing sound, and at one point, he dropped the background can be heard throughout the videos.|to smooth out the cement on a wall, pr . The man in the pink shirt was focused on<br><br>ork. A tree was visible in the backgroun man in the pink shirt worked, another per<br><br>float on the ground, making a impact so|oducing a scraping or smoothing sound. Another<br><br>his task, carefully spreading the cement. Nearby, a d, and the ground was covered with construction son used a wooden float, creating a scraping or<br><br>und. Sounds of traffic and people talking in the|

- Figure 7 Qualitative comparison of various MLLMs on the Audio-Visual Dense Narration (AVDN) task. Our model fine-tuned on EgoAVU-Instruct captures significantly more dense visual details than Qwen2.5 Omni and VideoLLaMA2, while also identifying auditory cues related to human actions and background sounds in the video.

audio-visual QAs in AVQA.

- 4.3 Error Analysis

Intrigued by the weak performance of current MLLMs across all tasks in EgoAVU-Bench, we conduct a detailed error analysis separately for the close-ended and open-ended tasks to analyze their behavior patterns. Our key findings are summarized below:

Close-Ended Tasks. Firstly, as illustrated in Table 4 and 5, for close-ended tasks such as Audio-Visual Hallucination (AVH) and Temporal Reasoning (TR), we evaluate MLLMs ability to independently perceive multisensory inputs within egocentric videos, specifically, human actions, visual objects, and sounds, including both foreground and background audio cues. This is achieved by separately evaluate the accuracy on the 3 subsets of corresponding tasks. Overall, our analysis reveals a consistent trend: MLLMs struggle the most with identifying sounds, followed by human actions, while performing relatively better at recognizing visual objects. For instance, in the TR task, the best-performing model, Qwen2.5-Omni, achieves only 36.1% accuracy in identifying sounds, exhibiting substantial performance gaps of 28.5% and 7.4% relative to visual object and human action identification, respectively. Furthermore, we find that model fine-tuned on our dataset, achieves a significant boost in its ability to independently perceive multisensory inputs within egocentric videos. For example, in the AVH task, compared to Qwen2.5-Omni, our fine-tuned model shows a reduced hallucination rate of 15.9%, 11.0%, and 30.0% when identifying human actions, visual objects, and sounds, respectively.

Open-Ended Tasks. We further extend our analysis to open-ended tasks such as Source-Sound Association (SSA) in EgoAVUBench, aiming to determine which modality contributes more to the overall error rate when models are asked to produce joint audio–visual descriptions. Specifically, we randomly sample 200 data points from the SSA task and, for each incorrect response across different MLLMs, manually annotate whether the error stems from inaccurate sound perception or an incorrect source description (which may involve visible objects or human–object interactions in the egocentric video). As shown in Fig. 8, our model achieves a substantially lower error rate of 21.1% compared to the next best model, Qwen2.5-Omni. Interestingly, other

|83.2%<br><br>78.4%<br><br>74.5%<br><br>70.2%<br><br>68.3%<br><br>47.2%<br><br>46.7%<br><br>| |
|---|
<br><br>Sound<br><br>| |
|---|
<br><br>Source| | | | |
|---|---|---|---|---|
| | | | | |

| | | |
|---|---|---|
| | | |

MiniCPM-o 2.6

| | | |
|---|---|---|
| | | |

Phi4-mm

| | | |
|---|---|---|
| | | |

Qwen2.5-Omni (3B)

| | | |
|---|---|---|
| | | |

VideoLLaMA2

| | | |
|---|---|---|
| | | |

Qwen2.5-Omni (7B)

| | | |
|---|---|---|
| | | |

Ours (LoRA)

| | | |
|---|---|---|
| | | |

Ours (Full)

0 20 40 60 80

Error Rate (%)

- Figure 8 Error Analysis on Sound-Source Association (SSA).

|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>... ...<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>MATTR = 0.06<br><br>MATTR = 0.42<br><br>|
|---|

Figure 9 Examples of video filtering based on MATTR scores.

MLLMs such as MiniCPM-o and Phi4-mm, which exhibit much higher error rates, show that over 72% of their errors stems from incorrect or missed sound descriptions rather than misidentified human–object interactions. This indicates that MLLMs primarily fail due to their limited ability to accurately perceive and interpret sounds, leading to incorrect associations with their visible sources.

- 5 Conclusion

We presented EgoAVU, a novel data engine to enhance egocentric audio-visual understanding by addressing the data limitations. EgoAVU employed modularized MLLMs to enhance egocentric narrations, and leveraged multi-modal context graphs to generate diverse, high-quality audio-visual QA pairs. Our evaluation benchmark EgoAVU-Bench revealed for the first time that existing MLLMs exhibit consistent vision bias, often neglecting or hallucinating audio information in egocentric videos. Our large-scale training dataset EgoAVU-Instruct effectively mitigated this gap, significantly improved performance on both EgoAVU-Bench and existing egocentric benchmarks. Importantly, EgoAVU demonstrated the self-learning potential of MLLMs: using uni-modal capabilities to improve the joint-modal capability. In terms of limitation, our training data, though with carefully designed filtering techniques, still contains noise from open source MLLM outputs. We believe this problem can be continually alleviated as the uni-modal capabilities of MLLMs improve, and leave the development of more soficsticated approaches as future work.

References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743, 2025.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

AI@Meta. Llama 3 model card. 2024. https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. https://arxiv.org/abs/2502.13923.

Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pages 65–72, 2005.

Leonard Bärmann and Alex Waibel. Where did i leave my keys?-episodic-memory-based question answering on egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1560–1568, 2022.

Changan Chen, Puyuan Peng, Ami Baid, Zihui Xue, Wei-Ning Hsu, David Harwath, and Kristen Grauman. Action2sound: Ambient-aware generation of action sounds from egocentric videos, 2024a. https://arxiv.org/abs/2406. 09272.

Qirui Chen, Shangzhe Di, and Weidi Xie. Grounded multi-hop videoqa in long-form egocentric videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2159–2167, 2025.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024b.

Sijie Cheng, Zhicheng Guo, Jingwen Wu, Kechen Fang, Peng Li, Huaping Liu, and Yang Liu. Egothink: Evaluating first-person perspective thinking capability of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14291–14302, 2024a.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024b.

Michael A Covington and Joe D McFall. Cutting the gordian knot: The moving-average type–token ratio (mattr). Journal of quantitative linguistics, 17(2):94–100, 2010.

Shangzhe Di and Weidi Xie. Grounded question-answering in long egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12934–12943, 2024.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

Tiantian Geng, Jinrui Zhang, Qingni Wang, Teng Wang, Jinming Duan, and Feng Zheng. Longvale: Vision-audiolanguage-event benchmark towards time-aware omni-modal perception of long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18959–18969, 2025.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.

Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400, 2024.

Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, Xudong Lu, Shuai Ren, Yafei Wen, Xiaoxin Chen, Xiangyu Yue, Hongsheng Li, and Yu Qiao. Imagebind-llm: Multi-modality instruction tuning, 2023. https://arxiv.org/abs/2309.03905.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. Advances in Neural Information Processing Systems, 35:3343–3360, 2022.

Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, et al. Baichuan-omni technical report. arXiv preprint arXiv:2410.08565, 2024a.

Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272, 2024b.

Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. https://aclanthology.org/ W04-1013/.

Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Z Xu, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. Advances in Neural Information Processing Systems, 35:7575–7586, 2022.

Jie Ma, Min Hu, Pinghui Wang, Wangchun Sun, Lingyun Song, Hongbin Pei, Jun Liu, and Youtian Du. Look, listen, and answer: Overcoming biases for audio-visual question answering. Advances in Neural Information Processing Systems, 37:9507–9531, 2024.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023.

Nuno Cid Martins, Bernardo Marques, Paulo Dias, and Beatriz Sousa Santos. Extending the egocentric viewpoint in situated visualization using augmented reality. In 2023 27th International Conference Information Visualisation (IV), pages 83–89. IEEE, 2023.

Tushar Nagarajan, Santhosh Kumar Ramakrishnan, Ruta Desai, James Hillis, and Kristen Grauman. Egoenv: Humancentric environment representations from egocentric video. Advances in Neural Information Processing Systems, 36: 60130–60143, 2023.

Amirshayan Nasirimajd, Simone Alberto Peirone, Chiara Plizzari, and Barbara Caputo. Epic-kitchens-100 unsupervised domain adaptation challenge: Mixed sequences prediction. arXiv preprint arXiv:2307.12837, 2023.

OpenBMB. Minicpm-o 2.6: A gpt-4o level mllm for vision, speech, and multimodal live streaming on your phone,

2024. Accessed: 2025-03-07.

Chiara Plizzari, Alessio Tonioni, Yongqin Xian, Achin Kulshrestha, and Federico Tombari. Omnia de egotempo: Benchmarking temporal understanding of multi-modal llms in egocentric videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24129–24138, 2025.

Shraman Pramanick, Yale Song, Sayan Nag, Kevin Qinghong Lin, Hardik Shah, Mike Zheng Shou, Rama Chellappa, and Pengchuan Zhang. Egovlpv2: Egocentric video-language pre-training with fusion in the backbone. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5285–5297, 2023.

Xavier Puig, Tianmin Shu, Shuang Li, Zilin Wang, Yuan-Hong Liao, Joshua B Tenenbaum, Sanja Fidler, and Antonio Torralba. Watch-and-help: A challenge for social perception and human-ai collaboration. arXiv preprint arXiv:2010.09890, 2020.

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9339–9347, 2019.

Ashish Seth, Utkarsh Tyagi, Ramaneswaran Selvakumar, Nishit Anand, Sonal Kumar, Sreyan Ghosh, Ramani Duraiswami, Chirag Agarwal, and Dinesh Manocha. EGOILLUSION: Benchmarking hallucinations in egocentric video understanding. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng,

editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 28449– 28468, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1446. https://aclanthology.org/2025.emnlp-main.1446/.

Kim Sung-Bin, Oh Hyun-Bin, JungMok Lee, Arda Senocak, Joon Son Chung, and Tae-Hyun Oh. Avhbench: A cross-modal hallucination benchmark for audio-visual large language models. arXiv preprint arXiv:2410.18325, 2024.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025a.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025b.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. Avqa: A dataset for audio-visual question answering on videos. In Proceedings of the 30th ACM international conference on multimedia, pages 3480–3491, 2022.

Hanrong Ye, Haotian Zhang, Erik Daxberger, Lin Chen, Zongyu Lin, Yanghao Li, Bowen Zhang, Haoxuan You, Dan

Xu, Zhe Gan, et al. Mm-ego: Towards building egocentric multimodal llms. arXiv preprint arXiv:2410.07177, 2024. Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, et al. Mmmu: A massive

multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. http://arxiv.org/abs/2403.13372.

Sheng Zhou, Junbin Xiao, Qingyun Li, Yicong Li, Xun Yang, Dan Guo, Meng Wang, Tat-Seng Chua, and Angela Yao. Egotextvqa: Towards egocentric scene-text aware video question answering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3363–3373, 2025.

## Appendix

- A Additional Details on EgoAVU

- A.1 Prompts

Prompt Used. We describe the various prompts used in EgoAVU. These prompts cover the generation of multiple modules including the Multi-modal Context Graph (see Fig. 11), Audio-Visual Narration (see Fig.13), and task-specific QA generation (see Fig. 14, Fig. 15, Fig. 16, Fig. 17, Fig. 18, Fig. 19, Fig. 20). The general structure of each prompt includes: (i) defining the objective, (ii) specifying the input format, (iii) describing the task, (4) providing general instructions, (iv) including human-generated examples to enable in-context learning, and (v) specifying the output format.

- A.2 MATTR

For video filtering, we utilize the Moving-Average Type-Token Ratio (MATTR) to identify videos with rich multimodal diversity. We set the window size to w = 200 tokens based on the average token length of combined narrations across video segments in our dataset, ensuring the window captures sufficient multi-modal context for meaningful diversity measurement. We retain videos whose MATTR exceeds τ = 0.3, effectively removing the bottom 25% of our distribution to filter out static or repetitive descriptions. This threshold was determined through manual inspection of 100 randomly sampled videos across different MATTR ranges. Videos below τ = 0.3 predominantly featured repetitive actions with limited object diversity and minimal auditory variation, while videos above this threshold exhibited richer multimodal dynamics (refer to Fig 9 for more examples).

- A.3 Ablation on Multi-Modal Context Graphs

To validate the necessity of the Multi-Modal Context Graphs (MCG) component in EgoAVU, we conducted an ablation experiment on 200 randomly sampled video clips. We compared our MCG-based pipeline against a direct baseline where LLaMA-3-70B generates audio-visual narrations directly from enhanced narrations (video caption, image caption, audio caption, and action narration) without the intermediate MCG structure. We manually evaluated both output, assessing: (1) completeness of sound-source associations, (2) accuracy of action sequences, and (3) overall audio-visual coherence. We observed that the direct method produced errors in 82 out of 200 captions (41.0%), with the breakdown as follows: 48 captions (19.0%) missed or incorrectly associated sound sources, 31 captions (15.5%) omitted crucial action sequences or interaction details, and 17 captions (3.5%) exhibited both issues (refer to Fig. 10 for example). In contrast, the our MCG-based approach reduced errors to 21 out of 200 captions (10.5%), representing a 76.1% relative error reduction.

- B Manual Effort for EgoAVU-Bench Construction

To ensure the reliability of EgoAVU-Bench, we conducted extensive manual verification across all 3,000 question-answer pairs covering 900 egocentric videos. Each video was carefully reviewed by trained annotators, taking approximately 2-3 minutes per video to verify temporal alignment and audio-visual correspondence. Out of 3,000 QA pairs, 1,524 pairs (50.8%) were modified or corrected during this process. For open-ended tasks (SSA, AVDN, AVSN), corrections primarily addressed missing sounds, incorrect human-object interactions, and sound-source misalignments to ensure accurate audio-visual grounding. For close-ended tasks (TR, AVH), we verified answer correctness and enhanced distractor quality by ensuring multiple-choice options were sufficiently challenging and plausible while avoiding options that were too similar to correct answers or obviously incorrect. The complete manual verification process required approximately 225 hours of human annotation effort.

|[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>... ... ... ...<br><br>Well, in the video, there's a woman in a kitchen. She's got a plate in her hand and she's standing by the sink. The sink is right there by a window, and it's a double sink. The kitchen has a granite countertop, which looks really nice. There's a dish rack on the counter with some dishes in it. You can see a bottle of dish soap and a sponge there too. The woman seems to be in the middle of doing the dishes. She's holding the plate and it looks like she's either about to put it in the sink or has just taken it out. The kitchen is well - lit, probably by the window, and it gives off a warm, cozy feeling.<br><br>Well, it sounds like there's a lot of clattering going on. You know, like dishes and pots and pans being moved around. And then there's the sound of water running, probably from a faucet. It seems like someone might be washing dishes or something. There's also a bit of a thud, which could be a pot being set down or something similar<br><br>['#O woman F picks plates from the cabinet', '#O woman F opens a tap', '#O woman F sets the plates on a cabinet. ', '#O woman F adjusts a tap pipe with his right hand. ', '#O woman F places plates on the counter']<br><br>A woman stood by the sink in a kitchen with a granite countertop, holding a plate. She picked the plate from a cabinet, causing a clattering sound. The woman then opened the tap, resulting in a water running sound, and appeared to be in the middle of doing the dishes. A dish rack with dishes, a bottle of dish soap, and a sponge were visible on the counter. The woman set the plates on a cabinet, producing a thud sound. The kitchen was well-lit, probably by the window, giving off a warm, cozy feeling, with various cleaning supplies are present but not interacted with.<br><br>A woman stood by the sink in a kitchen with a granite countertop, holding a plate. She opened the tap, resulting in a water running sound, and appeared to be in the middle of doing the dishes. A dish rack with dishes, a bottle of dish soap, and a sponge were visible on the counter. The kitchen was well-lit, probably by the window, giving off a warm, cozy feeling.<br><br>Egocentric Narration<br><br>Video Caption<br><br>Audio Caption<br><br>Audio Visual Narration (w/o MCG)<br><br>Audio Visual Narration (MCG)<br><br>|
|---|

Figure 10 Qualitative comparison of audio-visual narrations generated with and without (w/o) Multi-modal Context Graph (MCG). Our MCG-based approach produces narrations with superior audio-visual coherence, accurately capturing action sequences and sound-source associations, while the direct method (w/o MCG) often misses critical sounds or action sequences.

- C Additional Details on Narration Enhancement

Prompts. To capture spatial details, we extract the center frame and prompt Qwen2.5-VL with “Identify all the objects visible in the image” to detail all objects present in the video clip. To capture temporal dynamics, we utilize Qwen2.5-Omni in two stages: first, we process only the video frames with the prompt “Describe the video in detail” to capture visible activities; then, we process the audio with the prompt “Describe all the sounds heard in detail” to capture auditory information.

- D Additional Experiment Details

Training Details. For both LoRA and full fine-tuning, we use a maximum context length of 30,000 tokens and sample videos at 1 FPS with a frame resolution of 256 × 256. Training is performed using DeepSpeed ZeRO-3 with a learning rate of 1 × 10−4 and a cosine schedule with 10% warmup over 5 epochs. We perform balanced sampling, i.e, sample with equal weights from each task, during training.

Evaluation Details. Fig. 21 presents the prompt used for our LLM-as-judge evaluation. Following prior work (Plizzari et al., 2025), we assess the reliability of LLM-based scoring by measuring its alignment with human judgments on 300 randomly sampled open-ended QA pairs from EgoAVU-Bench. The resulting human-alignment rate is 87.6%, indicating strong alignment between the two.

Objective: You are an AI assistant tasked with performing a high-fidelity analysis of video content. Your role is to function as an evidence extractor, not an open-world reasoner. You must strictly use the provided captions to identify object interactions and to analyze sounds, grounding every piece of information directly to the source text.

Inputs: You will be provided with a JSON object containing the following four keys:

- • Video Caption: Describes the overall visual scene, including events and actions of entities other than the narrator (e.g., animals, other people, environmental events).
- • Image Caption: Describes the diverse objects visible in the center frame.
- • Audio Caption: A transcript of sounds and audio events.
- • Action Narration: Describes the specific actions performed by the primary person (#CC denotes person)

Task: Analyze the provided captions to generate a structured multimodal context graph in the form of JSON that captures multimodal relationship. For generating the multimodal context graph, follow the instruction mentioned below:

Instructions:

- • Identify Interacted Objects: Parse the action narration to find all objects the primary person is described as touching, holding, using, or manipulating. Compile these into the "interacted objects" list with what action the person performed.
- • Identify Background Objects: Take the complete list of objects from image caption. Create a new list containing only the items from image caption that are NOT in your "interacted objects" list. This will be your "non interacted objects" list.
- • Identify Sound-Source Associations This is a strict, evidence-based process. Ideally there can be two type of sound, sound caused by action/object or background sound. Your task is to capture both of them for audio caption

- A. Find the Grounding Evidence for foreground sound: For each sound in audio caption, you must search action narration and video caption to find the specific text that describes the action or event causing it. Look in action narration for causes related to the primary person’s actions. Look in video caption for causes related to other entities (animals, other people) or general scene events. ambient sound include music, background noise, etc. Crucially: If no direct textual evidence can be found in either caption that explains the sound’s origin, the sound is a background sound.
- B. Exclude "Unquestionable" Sounds: Even if grounded, do NOT include a sound if it falls into these categories: Mundane Biological Sounds: Common sounds like "breathing," "sighing," "swallowing." Vague Ambient Noise: like "white noise" or "faint hum."
- C. Determine the sound category: Classify as Foreground Sound (from the human action or visible object) or Background Sound.
- D. Handle Empty Results: If no sounds pass the filtering and grounding process, the "sounds" list in your output must be an empty list ([]).

Human Generated Examples: Here are 5 human created examples for the correct execution <examples>. Important Note:

- • In the example how "giggle" was excluded because it had no grounding evidence in video caption or action narration.
- • Source description will contain either the corresponding action or the object that can produced the sound
- • Sound category can have foreground sound such as “Action Sound”, “Object Sound” or background such as “Ambient Sound”

Final Output Format: Your entire response MUST be a single, valid JSON object following the structure of the example. Do not include any text outside of the JSON structure. Here is the input: <input>

Figure 11 Prompt For Generating Multi-Modal Context Graphs.

{

"interacted_objects": [ ["sink", "#C C rinses both hands"], ["tap", "#C C turns on tap"], ["door", "#C C opens the door"],

], "background_objects": [

"oranges", "sponge", "red chair", "microwave", "cabinets"

], "sounds": [

{

"acoustic_description": "water flowing sound", "source": "#C C turns on tap", "evidence_source": "action_narration", "sound_category": "Foreground Sound"

}, {

"acoustic_description": "hands being rinsed sound", "source": "#C C rinses both hands", "evidence_source": "action_narration", "sound_category": "Foreground Sound"

}, {

"acoustic_description": "door opening and closing sound", "source": "#C C opens the door", "evidence_source": "action_narration", "sound_category": "Foreground Sound"

} ]

}

Figure 12 Example of MCG. Example of MCG generated in JSON format using the above-mentioned prompt.

Objective: Your job is to generate a single, detailed, and objective paragraph summarizing what can be seen and heard in the video clip. Input: You will be given:

- • A free-form natural language paragraph summarizing what happens in the video. This is typically derived from a loose transcription or human description of the scene.
- • A list of short, possibly overlapping or action tags extracted from the video. These may contain minor inconsistencies, but offer clues about human interactions and movements in the scene. Focus closely on the interaction starting with #C C.
- • A multi-modal context graph represented as a structured JSON object containing: "interacted_objects"

— objects the person interacted with, "non_interacted_objects" — objects present but not interacted with, and "sounds" — sound events grounded in the narration, with descriptions and causes.

Task: Write a single, coherent paragraph that summarizes the video scene in detail, following these rules: Instructions

- • Clearly describe all key actions in the scene, combining the raw description and narration tags.
- • Include all interacted objects, and describe how each was interacted with.
- • Mention all non-interacted objects that are visible or relevant once, integrated naturally into the scene description. Do not repeat them again at the end.
- • Integrate sound events by describing what caused them and when, grounded in the referenced actions.

- – Not all actions have corresponding sounds.
- – Only include sounds that are listed in the scene graph.
- – Use semantically appropriate or naturalistic descriptions for acoustic events. For instance, if the sound is caused by shaking a spray bottle, you may refer to it as “crunching or rattling”.

- • Use an objective and factual tone—avoid any emotional, subjective, or evaluative language (e.g., no “cute,” “interesting,” or “simple”).
- • Write in past tense.
- • Ensure the paragraph flows naturally and avoids redundancy.

Human Generated Examples: Here are 5 human created examples for the correct execution <examples>. Final Output Format: The final output must be in JSON format with key as "caption"

Here is the input to generate the caption: <input>

Figure 13 Prompt For Generating Audio-Visual Narration.

Objective: You are an AI assistant tasked with analyzing a video segment and performing three tasks:

- • Generate a single open-ended question about the sound-source association observed in the video.
- • Produce a natural, human-like narration that links sounds to the actions and objects responsible.
- • Generate a detailed, structured answer to the question, grounded entirely in the provided scene graph metadata.

Input: You will be provided with:

- • video description: description of the video segment
- • Multi-modal Context Graph: <Details on Multi-modal Context Graph>

Instructions: Follow the below instruction to complete the task:

- • Question Generation. If the "sounds" list is empty or missing, return this exact string as the only output: "No significant sound is present in the video clip." Otherwise, use the template below: <template>
- • When narrating egocentric data, a person is sometimes referred to by capital letters, such as "C." When writing the description, treat such IDs as referring to a person. For example, if a sound-producing evidence states "person C is clapping," it should be treated as "the person is clapping."
- • Detailed Answer Generation.

- – Structure the answer as follows: Begin with a sentence that clearly states how many distinct grounded sound events were present. Then provide one sentence for each sound, explaining what caused it by using the acoustic description and grounding evidence. Treat C as the person in the video.
- – Do not speculate or add interpretation beyond the metadata.
- – Do not include any text outside of the JSON structure.
- – Do not include any step by step explanations.

Human Generated Examples: Here are 5 human created examples for the correct execution <examples>. Output Format: Output must be in a JSON format with following key "question" and "answer". Here is the input to generate the question-answer pair: <input>

Figure 14 Prompt for generating Sound-Source Association Question-Answer pair.

Objective: Generate two sound-related question–answer pairs from an egocentric video caption that describes a person’s visible actions, sounds, and objects. The output should be formatted as JSON with one correct and one hallucinated sound question.

Input: You will be given an egocentric video narration containing descriptions of:

- • The person’s visible actions
- • Distinctive sounds (e.g., hissing, tapping, scraping)
- • Objects present in the scene
- • Temporal information about when events occur

Instructions: Follow the instruction below:

- • Focus on distinctive sounds such as foreground sounds related to human-object interaction such as hissing, tapping etc. or background sounds such as bird chirping etc.
- • Generate one correct question: Ask about a sound explicitly mentioned in the narration.
- • Generate one hallucinated question: Ask about a plausible sound that is not mentioned in the narration.
- • Answer format: Answers must be in binary format "Yes" or "No".

Output Format: The output must be in JSON format with following keys: "question", "question type" including "Factual", "Hallucinated" and "answers". Here is the input to generate the question-answer pair: <input>

Figure 15 Prompt for generating Audio-Visual Hallucination (Sound) Question-Answer pair.

Objective: Generate two action-related question–answer pairs from an egocentric video caption that describes a person’s visible actions, sounds, and objects. The output should be formatted as JSON with one correct and one hallucinated action question.

Input: You will be given an egocentric video narration containing descriptions of:

- • The person’s visible actions
- • Distinctive sounds (e.g., hissing, tapping, scraping)
- • Objects present in the scene
- • Temporal information about when events occur

Instructions: Follow the instruction below:

- • Focus on distinctive, non-trivial actions such as wiping, twisting, or squeezing. Avoid trivial actions such as breathing, walking, or placing.
- • Generate one correct question: Ask about an action explicitly mentioned in the narration.
- • Generate one hallucinated question: Ask about a plausible action that is not mentioned in the narration.
- • Answer format: Answers must be in binary format "Yes" or "No".

Output Format: The output must be in JSON format with following keys: "question", "question type" including "Factual", "Hallucinated" and "answers". Here is the input to generate the question-answer pair: <input>

Figure 16 Prompt for generating Audio-Visual Hallucination (Action) Question-Answer pairs.

Objective: Generate two object-related question–answer pairs from an egocentric video caption that describes a person’s visible actions, sounds, and objects. The output should be formatted as JSON with one correct and one hallucinated object question.

Input: You will be given an egocentric video narration containing descriptions of:

- • The person’s visible actions
- • Distinctive sounds (e.g., hissing, tapping, scraping)
- • Objects present in the scene
- • Temporal information about when events occur

Instructions: Follow the instruction below:

- • Focus on specific, manipulable objects. Avoid generic nouns like ’things’, ’stuff’, or ’material’.
- • Generate one correct question: Ask about an object explicitly mentioned in the narration.
- • Generate one hallucinated question: Ask about a plausible object that is not mentioned in the narration.
- • Answer format: Answers must be in binary format "Yes" or "No".

Output Format: The output must be in JSON format with following keys: "question", "question type" including "Factual", "Hallucinated" and "answers". Here is the input to generate the question-answer pair: <input>

Figure 17 Prompt for generating Audio-Visual Hallucination (Object) Question-Answer pairs.

##### Objective: Generate two temporal reasoning question–answer pairs from a list of chronological video narrations, focusing on the order of Action, Object, and Sound events. The output should be formatted as a JSON list containing one "before" and one "after" question.

Input:

- • A list of narration describing what happens in the video in chronological order ({caption_list}).
- • The specific question type to be generated ({type}: one of Action-Action, Action-Object, or Action-Sound).

Instructions: Follow the steps below:

- 1. Identify Distinct Events: Identify several unique, non-trivial, and non-repetitive events, each describing an Action, an Object, or a Sound.
- 2. Select Event Pair (E1, E2): Choose two events occurring at different times that match the required category ({type}). E1 must chronologically precede E2.
- 3. Generate Questions: Create one "before" question (referencing E2) and one "after" question (referencing E1) using the corresponding template:

- • Action–Action

- – Before: "What action was the person performing before <E2>?"
- – After: "What action did the person perform after <E1>?"

- • Action–Object

- – Before: "What objects can be seen before the person performed the <E2> action?"
- – After: "What objects can be seen after the person performed the <E1> action?"

- • Action–Sound

- – Before: "What sound can be heard before the person <E2>?"
- – After: "What sound can be heard after the person <E1>?"

- 4. Answer and options: Write a concise, naturalistic answer as if you watched the video. Include three plausible options that fit the context but are temporally incorrect.

Output Format: The output must be a JSON list of exactly two question objects (one "before" and one "after") with the following keys: "question", "answer", "type", and "options".

- Figure 18 Prompt for generating Temporal Reasoning (before/after) Question-Answer Pairs

Objective: Generate one multiple-choice question about the temporal order of four events derived from a sequence of chronological egocentric video narration. Input: A sequence of detailed narration in chronological order describing what happens in a egocentric video. Instructions: Follow the steps below:

- • Identify Four Grounded Events: Identify four unique, non-trivial events. Each event must include concise but meaningful details about the person’s activity, the visible surroundings, and any sounds mentioned.
- • Grounding Constraint: All events must be directly derived from the narrations. Do not hallucinate or invent any objects, sounds, or actions.
- • Create Temporal Question: Create one general multiple-choice question that asks about the temporal order of the four events (e.g., "Which event happened first?", "Which moment occurred last?").
- • Create Options: List the four events as options A, B, C, and D. Ensure the description for each option is the exact description provided in the events list.
- • Provide Correct Answer: Indicate the correct temporal order by selecting one of the options (A, B, C, or D) as the correct_answer.

Output Format: The output must be in JSON format with the following keys: "events" (a list of the four descriptions), "question", "options" (a map of A, B, C, D to the event descriptions), and "answer".

- Figure 19 Prompt for generating Temporal Reasoning (Event Ordering) Question-Answer Pairs

Objective: Write a single, coherent, dense narration summarizing the entire video based on a list of 10-second captions. Input: A list of narration, each describing a 10-second segment of a egocentric video, including start_time, end_time, and the caption text. Instructions: The final output must be a single, fluent paragraph that acts as a dense narration. The paragraph must adhere to the following rules:

- • Integrate all actions, objects, and sounds across the full video.
- • Use timestamps in seconds to indicate when key events occurred.
- • Group similar or adjacent events into continuous spans.
- • Avoid listing or repeating captions verbatim.
- • Use only the information in the input captions.
- • Be concise and fluent.
- • Do not invent any new information or context.

Human Generated Examples: Here are 3 human created dense narration: <examples> Output Format: A single paragraph, not a JSON object. Here is the input: <input>

- Figure 20 Prompt for generating Audio Visual Dense Narration

Objective: Act as an impartial grader to evaluate a PREDICTED_ANSWER against a GROUNDING_ANSWER with respect to a QUESTION. Input:

- • QUESTION: The question posed to the model.
- • GROUNDING_ANSWER: The authoritative reference answer.
- • PREDICTED_ANSWER: The model’s answer to be graded.

Instructions for Grading:

- 1. Comparison: Compare PREDICTED_ANSWER to GROUNDING_ANSWER with respect to the QUESTION.
- 2. Assign Rating (1-5, integer only):

- • 5: Fully correct, complete, and faithful to the grounding; no meaningful errors or omissions.
- • 4: Mostly correct; minor omissions or small inaccuracies that do not change the overall correctness.
- • 3: Partially correct; captures some key points but misses important details or includes notable inaccuracies.
- • 2: Largely incorrect; substantial errors, contradictions, or missing major required points.
- • 1: Incorrect/irrelevant; contradicts the grounding or fails to answer the question.

- 3. Provide Reasoning: Briefly explain the rating (1–4 concise sentences).

Judging Rules (Priorities):

- • Prioritize factual alignment with the GROUNDING_ANSWER. Contradictions result in heavy penalization.
- • Extra details are acceptable only if they do not conflict with the grounding and remain relevant to the QUESTION.
- • Penalize hallucinations, unverifiable claims, safety issues, and failure to address the core of the QUESTION.
- • Do not reward verbosity or style unless it improves factual accuracy or completeness with respect to the grounding.
- • If the grounding indicates the question is unanswerable, judge whether the prediction correctly reflects that.

Output Format: The output MUST be valid JSON (no markdown, no extra text) with the following keys: {

"rating": <int value between 1 to 5>, "reason": "<string of 1-2 lines explaining the rating>"

}

- Figure 21 Prompt for LLM-as-judge evaluation

