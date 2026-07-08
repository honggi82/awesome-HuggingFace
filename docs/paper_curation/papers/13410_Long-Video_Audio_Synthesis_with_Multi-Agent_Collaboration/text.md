## Long-Video Audio Synthesis with Multi-Agent Collaboration

Yehang Zhang1∗ Xinli Xu1∗ Xiaojie Xu1∗ Li Liu 1,2† Ying-Cong Chen1,2† 1HKUST(GZ) 2HKUST *

# arXiv:2503.10719v2[cs.CV]17Mar2025

[Figure 1]

[Figure 2]

[Figure 3]

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

[Figure 16]

[Figure 17]

Long Video

with multi-scene

[Figure 18]

Scene #2

Scene #N

Scene #1

|[Figure 19]<br><br>Scriptwriter<br><br>[Figure 20]<br><br>Storyboarder<br><br>[Figure 21]<br><br>[Figure 22]<br><br>Designer<br><br>[Figure 23]<br><br>Generator<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>LVAS-Agent|
|---|

Background: Explosion

Background: None

Background: Helicopter

Entities: Cat Actions: Running, Meowing

Entities: Building, Monster

Entities: Monster, Helicopter Actions: Helicopter hovering,

[Figure 28]

[Figure 29]

Actions: Building Explosion

Summary: A cat is running

Summary: The building explodes after being trampled by the

Monster Eating

[Figure 30]

while an explosion occurs

Summary: The monster is eating

behind it.

monster.

while a helicopter hovers above it.

‘building explosion’ : {

‘Helicopter’ : {

‘violent explosion’ : { ‘Layout’: ‘Background’,

‘Layout’: ‘Foreground’,

‘Layout’: ‘Background’,

[Figure 31]

‘Volume’: 30 db’

‘Volume’: 20 db’

‘Volume’: ’20 db’

}

}

}

[Figure 32]

[Figure 33]

‘Monster roaring’ : {

‘Chewing sound’ : {

‘cat meowing’ : {

‘Layout’: ‘Foreground’, ‘Volume’: ’25 db’

‘Layout’: ‘Background’, ‘Volume’: ’20 db’

‘Layout’: ‘Foreground’, ‘Volume’: ’25 db’

}

}

}

[Figure 34]

Tool use: Video-to-Audio & Text-to-Audio generation, Audio mixing & Enhancement, Volume adjustment,

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

Video with

sound Effect

|[Figure 45]<br><br>Long Video Process: Visual Agent Support<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Audio Design: Auto Description Generation & correction<br><br>Audio Generation: HierGen & Highly editable<br><br>Multi-Agent System Cooperation|
|---|

|Long-Video Process: Manual Caption and Trimming Audio Design: Manual configuration of prompts<br><br>Audio Generation: Non-editable, no multi-level generation.<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>Current Video-to-Audio Methods|
|---|

Figure 1. We introduce LVAS-Agent, a multi-agent collaborative framework for end-to-end long video audio synthesis. Built on VLM and LLM-based agents, it simulates real-world dubbing workflows, enabling automatic video script generation, audio design, and high-quality audio synthesis for long videos.

*Equal contribution. † Co-corresponding author.

### Abstract

Video-to-audio synthesis, which generates synchronized au-

dio for visual content, critically enhances viewer immersion and narrative coherence in film and interactive media. However, video-to-audio dubbing for long-form content remains an unsolved challenge due to dynamic semantic shifts, temporal misalignment, and the absence of dedicated datasets. While existing methods excel in short videos, they falter in long scenarios (e.g., movies) due to fragmented synthesis and inadequate cross-scene consistency. We propose LVAS-Agent, a novel multi-agent framework that emulates professional dubbing workflows through collaborative role specialization. Our approach decomposes longvideo synthesis into four steps including scene segmentation, script generation, sound design and audio synthesis. Central innovations include a discussion-correction mechanism for scene/script refinement and a generation-retrieval loop for temporal-semantic alignment. To enable systematic evaluation, we introduce LVAS-Bench, the first benchmark with 207 professionally curated long videos spanning diverse scenarios. Experiments demonstrate superior audio-visual alignment over baseline methods. Project page: https://lvas-agent.github.io.

### 1. Introduction

Recent advancements in generative AI, particularly diffusion models and large language models (LLMs), have significantly improved short-video dubbing, enabling synchronized audio that enhances viewer immersion. However, long-video dubbing presents unique challenges, including complex semantic shifts, cross-scene temporal alignment, and adaptation to dynamic content. Current models, optimized for short clips, struggle to maintain narrative coherence over extended durations, limiting their effectiveness in applications such as film dubbing, AIGC video voiceovers, and automatic narration for mute videos. Moreover, the lack of dedicated datasets for long-video audio synthesis has hindered progress, as current datasets and benchmarks focus on short-form content.

Existing video-to-audio methods fall into two categories: (1) training dedicated generators such as SpecVQGAN [13] and Diff-Audio [33], which capture short-term correlations but fail in long-term scene transitions, and (2) adapting text-to-audio models like SonicVisionLM [52] and V2AMapper [47], which heavily rely on textual descriptions and struggle with implicit visual cues in long videos. These methods encounter common issues:(i) they lack mechanisms to capture long-range dependencies across dynamically changing scenes, (ii) they fail to preserve contextual continuity in dialogue-rich videos, and (iii) they struggle to synthesize background sounds that evolve naturally over extended durations. Additionally, these methods rely on shortvideo datasets, which lack annotations for multi-sounds and cross-scene consistency with only 2-4 words for each audio

labels.

A fundamental question arises: How can we leverage short-video dubbing priors to enable long-video synthesis while ensuring semantic coherence, temporal alignment, and scalable synthesis without requiring large-scale longvideo training data? A naive approach is to split long videos into shorter segments and apply existing methods. However, this approach practically may lead to issues such as poor continuity, unnatural transitions, and unclear main voice due to the lack of understanding of long-sequence videos.

To address these challenges, we present LVAS-Agent, a multi-agent framework that mimics professional dubbing workflows through structured role collaboration. Our key innovation lies in decomposing the synthesis process into specialized stages with collaborative agents: semanticaware scene segmentation, context-sensitive script generation, ambiguity-resolved sound design, and knowledgeenhanced audio synthesis. The overall is shown in Figure 1.

Specifically, our method operates through four tightly coupled roles. The Storyboarder first segments videos into narrative-preserving scenes using shot transition detection and contrastive keyframe clustering. The Scriptwriter then generates time-aligned audio scripts by fusing visual semantics from CLIP-encoded [38] features with dialogue context analysis. Building on this, the Designer employs spectral saliency analysis to disentangle foreground dialogues from ambient sounds, refining annotations through agent-mediated ambiguity resolution. Finally, the Synthesizer orchestrates hybrid audio generation, blending neural text-to-speech with diffusion-based environmental effects. Central to the system are two collaborative mechanisms: a discussion-correction process for scene merging and script refinement, and a generation-retrieval-optimization loop that iteratively aligns sound design with retrievable audio knowledge, enabling precise temporal and semantic consistency.

To enable systematic evaluation, we introduce the LongVideo Audio Synthesis (LVAS) benchmark, comprising 207 professionally selected long videos. LVAS-Bench covers various scenes such as urban landscapes, combat simulations, and animation actions to ensure the accuracy of the evaluation.

Our contributions can be summarized as follows:

- • We introduce LVAS-Agent, a multi-agent framework that systematically addresses long-video dubbing challenges by structuring the synthesis process into role-specialized collaborative agents.
- • We release LVAS-Bench, the first dedicated long-video audio synthesis dataset, covering 207 professionally curated videos across diverse scenarios, enabling standardized benchmarking.
- • Experiments demonstrate that LVAS-Agent improves se-

mantic alignment, temporal alignment and distribution matching of audio-visual for long-video dubbing over existing baselines.

### 2. Related Work

#### 2.1. Video-to-Audio Generation

Video-to-audio generation, also known as dubbing, is a crucial audio technique for enhancing viewers’ auditory experience and has witnessed significant evolution with the advent of neural approaches. Early neural dubbing models demonstrated deep learning’s potential in sound effect creation, though limited to specific genres [1, 2, 30, 59]. Recent advancements in video-to-audio generation have followed two main directions. The first approach focuses on training generators from scratch, with notable works including SpecVQGAN [13], which employs a cross-modal Transformer for auto-regressive sound generation, Im2Wav [40], which conditions audio generation on CLIP features, Diff-Foley [27], which enhances alignment through contrastive pre-training and MMAudio [3], which use a flow matching-based multimodal joint training framework on large-scale data. The second approach adapts text-to-audio generators, with Xing et al. [53] utilizing ImageBind [8] for optimization-based alignment, SonicVisionLM [52] employing caption-based synthesis, V2A-Mapper [47] directly translating visual to text embeddings and FoleyCrafter [57] integrating a learnable module into text-to-audio models with end-to-end training. Despite these advances, existing methods primarily excel only with short videos, encountering issues such as noise artifacts and audio-scene inconsistencies in longer videos. Our method addresses these limitations by incorporating a video understanding segmentation module, providing an effective solution for audio generation in long-form videos.

#### 2.2. MLLMs for Video Understanding

Recent advances in vision foundation models [4, 15, 25, 38, 43, 44, 56] have led to the emergence of multimodal LLMs (MLLMs) [18, 23, 45, 55], which have demonstrated remarkable capabilities in language-guided visual understanding. This progress has naturally extended to video understanding, with pioneering works including VideoChat [20], Video-ChatGPT [28], Video-LLaMA [54], VideoLLaVA [22], LanguageBind [60], and Valley [26]. However, videos present unique challenges compared to static images, particularly due to their temporal nature and the substantial volume of visual information that, when tokenized, often exceeds MLLMs’ context limitations. While most existing approaches address this through frame sampling, some methods, such as Video-ChatGPT [28], have introduced more efficient video feature representations. The field has also witnessed significant progress in instance-

level video understanding, with works like PG-VideoLLaVA [31] for video grounding, and Artemis [36] for video referring, expanding the capabilities of video understanding systems.

The challenge becomes particularly acute in long video understanding, where effective keyframe selection becomes more crucial and complex. While some approaches like Kangaroo [24] and LLaVA-Video [58] leverage language models with expanded context windows to accommodate more frames, others have developed specialized techniques to handle this limitation. MovieChat [42], for instance, implements a dual-memory system with short-term and long-term memory banks for efficient video content compression and preservation. Similarly, MA-LMM [9] and VideoStreaming [35] employ a Q-former alongside a compact language model (Phi-2 [14]) for video data condensation. LongVLM [50] takes a different approach by utilizing token merging to reduce video token count.

#### 2.3. Multi-Agent System

Multi-agent systems have evolved significantly with the advent of large language models (LLMs). Early single-agent frameworks like ModelScope-Agent [19] and Toolformer [39] demonstrated the potential of LLMs in executing complex tasks through tool integration. HuggingGPT [41] and AudioGPT [12] further expanded this capability by incorporating domain-specific models and functionalities.

However, the limitations of single-agent systems led to the emergence of multi-agent frameworks. Inspired by the Society of Mind [29], works like Generative Agents [32] pioneered the development of “Simulated Society”, where multiple agents interact within defined environments. Practical implementations such as ChatDev [34], MetaGPT [10], and TransAgents [51] have successfully demonstrated collaborative problem-solving through simulated workflows, achieving superior reasoning and factuality compared to single-agent approaches. These systems effectively address complex tasks requiring meaningful collaborative interaction, which single agents typically struggle to accomplish.

For long-video audio synthesis, while AudioAgent [49] employs pre-trained diffusion models and GPT-4, it lacks explicit multi-agent collaboration and specific optimizations for long videos. In contrast, we propose LVAS-Agent: a multi-agent collaborative framework that mimics professional dubbing workflows. Our system comprises specialized agents for video segmentation, content understanding (leveraging advanced MLLM models for precise description), and audio tag generation (separating foreground and background elements). These components work in concert with MMAudio to produce high-quality synthesized audio.

### 3. Method

#### 3.1. Overview

By clearly defining the roles of agents, multi-agent systems can decompose complex tasks into smaller, more manageable ones. In LVAS-Agent, we define four main characters: Storyboarder, Scriptwriter, Designer, and Synthesizer. As show in Figure 2, each of these roles carries its own specific set of responsibilities. The Storyboarder is responsible for the creation of video storyboards. This includes planning the storyboard strategy, segmenting video scenes, and extracting keyframes. The Scriptwriter is in charge of writing the video script. Their responsibilities include understanding the video content, collaborating with the storyboard artist to generate a detailed video outline, and providing references for the sound designer. The Designer is tasked with annotating sound effects based on the video outline. This includes analyzing video descriptions, generating detailed sound effect annotations for each potential sound, classifying entity and environmental sounds, and collaborating with the voiceover artist to refine the sound annotations. Finally, the Generator is responsible for the actual sound effect synthesis. This includes transforming sound effect annotations into suitable sound labels and using professional tools to achieve step-by-step sound-video synthesis, combining the main sound and background sound.

#### 3.2. Multi-agent collaboration strategy

In this section, we present the two agent collaboration strategies employed in this work: Discussion-Correction (Algorithm 1) and Generation-Retrieval-Optimization (Algorithm 2).

Discussion-Correction This strategy, as illustrated by Algorithm 1, is executed through the collaboration between two agents . First, the Storyboarder agent P segments the video into distinct scenes, denoted as [v0,...,vn], based on shot transitions and extracts the corresponding keyframes lists {[kf1,1,...],...,[kfn,1,...]}. Next, the Scriptwriter agent Q performs a global analysis of the entire video, followed by a detailed examination of each segment based on its respective keyframes. The Storyboarder agent P and Scriptwriter agent Q then engage in a discussion to determine whether certain segments should be merged and whether the segment captions require refinement, based on both the global understanding and the detailed video captions. The final output is a structured video script.

Generation-Retrieval-Optimization This is accomplished through the collaboration between the Designer agent D and the Synthesizer agent S. First, the Designer agent D formulates an initial sound design based on the video script. The Synthesizer agent S then retrieves relevant knowledge from a sound synthesis database to generate a concrete implementation plan. This plan is reviewed by the Designer

Algorithm 1: Collaboration Strategy Input: Storyboarder agent P, Scriptwriter agent Q,

Video V Output: Structured video script T T() ← ∅ {v0,...,vn} ← P(V) // Shot

Change Detection

{kf0,...,kfn} ← P({v0,...,vn})

// Keyframe Extraction

UV ← Q(V) // Understand global

content, style features

for i = 0 to n do Ui ← Q(kfi) if i > 1 then

D ← P(Ui,Ui−1,UV) if D = MERGE then

vi−1 ← merge segments(vi−1,vi)

T ← T ∪ [Ui−1,Ui] else

T ← T ∪ [Ui]

T ← T ∪ UV return T

agent D, who decides whether further refinement of the sound design is needed or if the plan is ready for final synthesis. Specifically, this process begins with an in-depth understanding of the video script, followed by iterative exchanges of feedback between the Designer agent D and the Synthesizer agent S. Through multiple iterations, the final sound synthesis plan is determined.

Algorithm 2: Generation-Retrieval-Optimization Collaboration Strategy

Input: Designer agent D, Synthesizer agent S,

Video script T, Maximum iterations Nmax Output: Finalized sound synthesis plan Afinal Initialization: Ainit ← D(T);

Aretrieved ← S(Ainit); for i = 1 to Nmax do

Areviewed ← D(Aretrieved); if D determines Areviewed is FINAL then

break; // Exit early if

finalized

Amodified ← D(Areviewed); Aretrieved ← S(Amodified);

return Areviewed;

#### 3.3. Video Structure

As shown in Figure 2, this paper proposes a structured video script generation method to assist in generating sound ef-

###### Multi-Scene Long-Video

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

| | |
|---|---|
| | |

[Figure 57]

Chain-of-Thought

Designer

| | |
|---|---|
| | |

Video Script

|ID|Caption|Initial Audio|Layout|
|---|---|---|---|
|01|Tank<br><br>Moving…|Tanks engine roar|Foreground|
| | |Dusty wind|Background|
|02|Tank<br><br>reloading..|Friction of<br><br>mechanical devices|Foreground|
|03|Tanks<br><br>firing…|Tanks open fire|Foreground|

| |
|---|

###### Initial Video Processing

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Storyboarder

Initial Audio Script Generation

Shot Detection Keyframes Extraction

Correction: Tank firing

should be firing cannon Final Design: True

Judge

Video Script

[Figure 62]

Retrieval reference label

[Figure 63]

[Figure 64]

###### Discussion Caption

|ID|Caption|End Frame|
|---|---|---|
|01|Tank Moving…|48<br><br>[Figure 65]|
|02|Tank reloading…|216|
|03|Tanks firing|322|

Merge&

[Figure 66]

[Figure 67]

Optimize

Update

Correction :

[Figure 68]

Final Design

- ID #1 & #2

[Figure 69]

- ID #2 & #3

[Figure 70]

- ID #3 & #4

Designer

Generator

Pre-built audio

Critique

description knowledge

Correction

|ID|Caption|Final Audio|Layout|Volume|
|---|---|---|---|---|
|01|Tank Moving…|Engine roar, tires rotating|Foreground|5|
| | |Wind noise|Background|25|
|02|Tank reloading..|Friction of mechanical devices|Foreground|15|
|03|Tanks firing…|Firing cannon|Foreground|35|

Focus on:

- - Background: …
- - Entities: …
- - Actions: …

###### Visual Agent Caption

Scene #1

Scene #2

Tank Moving

Tank reloading Scene #3 Two tanks firing

Clipped

[Figure 71]

Video

Scene #4

[Figure 72]

[Figure 73]

One tank firing

…

[Figure 74]

Video to Audio Tools

Background: wind

Background: Birds

Global Understanding

Whole

Text to Audio Tools

[Figure 75]

[Figure 76]

Approximate timeline (Part1: …)

…

Video

Scriptwriter

Audio Editing Tools

Foreground: …

Foreground:engine roar

Stage1: Video Outline

Stage2: Audio Design & Synthesize

- Figure 2. Workflow of LVAS-Agent. Given the original video, Storyboarder and Scriptwriter collaborate through Discussion and Correction to create a structured video script. The Designer and Generator complete multi-layered, high-quality sound synthesis through the Generate-Retrieve-Optimize mechanism.

fects for full-length videos. The method addresses three core challenges: 1) existing audio synthesis tools have duration constraints; 2) current Video-To-Audio methods struggle with scene and content transitions, hindering semantic and temporal alignment; 3) ensuring consistency between video captions and audio descriptions for coherent synthesis. To overcome these, we introduce a fine-grained video structuring approach, supported by collaboration between storyboarder and scriptwriter agents, as outlined in Algorithm 1 The specific design of these agents is detailed as follows.

Storyboarder is an LLM-based agent responsible for finegrained video structuring in the VTA task. Its key functions include detecting shot transitions for coarse segmentation, extracting key frames using the K-Means clustering algorithm, and refining segment boundaries and captions based on the Scriptwriter’s Video Caption. Shot detection uses an HSV color space transition method for rapid, frameaccurate segmentation, enabling detailed video understanding. By extracting key frames from smaller segments, it captures more visual information compared to directly inputting the full video into a vision-language model, enhancing video comprehension. Storyboarder also collaborates with the Scriptwriter to decide whether segments should be merged or captions refined, considering both local and

global video context. Detailed implementation is in Appendix 1.

Scriptwriter is a visual support agent responsible for comprehending both the full video and individual video segments. Recent video understanding tasks achieve comprehension by extracting information from visual contexts to derive semantic features [21] or by directly generating descriptive text [5]. Textual descriptions of the video script make it easier to maintain consistency between video and audio descriptions. Furthermore, the textual format facilitates interaction among multiple agents. Notably, transforming the video into a structured script, independent of video frames, enhances processing speed and significantly reduces the number of tokens. The detailed implementation is provided in Appendix 2.

#### 3.4. Audio Design and Generation

This section presents the second stage of LVAS-Agent: audio design and generation, as shown in Figure 2. The design follows key principles: 1) Mimicking professional sound design workflows by analyzing video scripts for accurate audio descriptions, 2) Enhancing efficiency and quality using existing audio generation tools, and 3) Ensuring structured, editable audio planning for fine-grained control. This stage uses a collaborative framework with two LLM-based agents, integrating retrieval-augmented genera-

[Figure 77]

|[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]|
|---|

[Figure 96]

[Figure 97]

Fighting, Sword

Cooking, Cutting

[Figure 98]

[Figure 99]

Forest, Bicycle

(a) (b) (c)

- Figure 3. Our LVAS-Bench is presented in the following parts: (a) illustrates sample data from the benchmark, (b) provides statistical distributions of audio categories and sub-categories across the dataset, and (c) presents the statistics of video categories within the dataset.

gorithm 2, facilitating iterative refinement of audio synthesis. The LVAS-Agent employs MMAudio [3], an opensource framework supporting VTA and TTA synthesis, ensuring flexibility for final audio mixing, volume adjustment, and refinements.

tion (RAG) and audio synthesis tools to create high-quality, multi-layered audio.

Designer annotates audio in the video script and collaborates with the Synthesizer agent to finalize the audio design. Real-world dubbing often involves complex scenes with layered environmental sounds, diverse sound-producing actions, and varying audio levels. To address this, we introduce a Chain-of-Thought (CoT) reasoning mechanism, breaking the task into steps: identifying primary action sounds, analyzing background audio, and ensuring audio coherence. The Designer agent creates the initial audio design, covering foreground and background sounds, volume control, and sound descriptions, while verifying alignment with the video content. It then provides iterative feedback to the Audio Synthesizer to optimize the final audio plan.

### 4. LVAS-Bench

Collection. We construct the first specialized long-video audio synthesis benchmark(LVAS-Bench). The benchmark contains 207 professionally curated videos (with an average duration of 1 minute) sourced from three main origins: (1) film production archives with open licenses, (2) annotated documentary segments, and (3) procedurally generated synthetic scenes. Importantly, all videos in the dataset have pure sound effects, with no background noise or human speech. This creates a dataset of long-form, semantically rich videos with clear transitions, matched with corresponding pure sound-audio. Figure 3(a) illustrates representative video-audio cases.

Generator The Generator synthesizes audio based on the audio annotations obtained through collaboration with the Designer. It uses retrieval-augmented generation (RAG) with an audio label knowledge base, Video-to-Audio (VTA) and Text-to-Audio (TTA) models for synthesis, hierarchical mixing, and volume adjustment. RAG-based retrieval ensures high-quality synthesis, addressing the limitations of VTA models trained on the VGGSound dataset, which contains only 310 audio labels with 2-4 words each. When audio prompts match these predefined labels, the generated audio is more stable and higher quality.

Statistical Analysis. To ensure diversity, LVAS-Bench covers sufficient video and audio categories. Figure 3(b) visualizes the benchmark’s audio types, encompassing five major classes (e.g., human activities) with numerous finegrained subcategories. Figure 3(c) quantifies the distribution across 10 video-level categories, where instances such as the “cooking” category comprise 22 entries.

Benchmark Annotation. LVAS-Bench also offers detailed time-stamped annotations for each video-audio pair and comprehensive global descriptions. The time-stamped annotations indicate captions from specific seconds to specific seconds, while the global descriptions provide a detailed account of the entire long video. We implement a hybrid annotation protocol: initial annotations are generated by video understanding model, subsequently refined through manual verification by domain experts.

Building on this insight, all VGGSound labels were reorganized and reclassified into 20 common video scenarios. To enrich the labels, GPT-4 and human annotators added details such as typical scenarios and relevant objects or interactions. This resulted in 192 refined labels. The structured knowledge base allows the Generator to retrieve and modify predefined labels, rather than relying on open-ended prompts. This retrieval-based approach enables the “generation-retrieval-optimization” mechanism in Al-

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Scene: The train passes through the station rapidly. Then a steam train appears and whistles.

[Figure 107]

| |
|---|

Wrong sound: Train horn

MMAudio, Prompt: train passes, train horn

[Figure 108]

| |
|---|

LVAS-Agent design Background: wind

Ours LVAS-Agent: AutoGen

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

[Figure 120]

[Figure 121]

Scene: Light the fireworks. The fireworks soar into the air and explode multiple

times.

[Figure 122]

| |
|---|

| |
|---|

Missing explosion

MMAudio, Prompt: Fire cracker

Missing Explosion

[Figure 123]

LVAS-Agent: Align with the explosion scene.

Ours LVAS-Agent: AutoGen

- Figure 4. We visualize the spectrograms of generated audio (by prior works and our method). LVAS-Agent demonstrates superior performance in synthesizing long video audio, ensuring seamless scene transitions without errors or missing sounds.
- 5. Experiment 5.1. Experiment Setup.

ports audio generation for segments up to 10 seconds, while MMAudio performs better for videos around 10 seconds due to the duration of its training data. Consequently, we set the segment interval for the baseline to 10 seconds.

Metrics We assess the generation quality in four different dimensions: distribution matching, audio quality, semantic alignment, and temporal alignment. 1) Distribution matching assesses the similarity in feature distribution between ground-truth audio and generated audio, under some embedding models. We compute Fr´echet Distance (FD) and Kullback–Leibler (KL) distance. For FD, we adopt PaSST [17] ((FDPaSST), PANNs [16] (FDPANNs), and VGGish [6] as embedding models. For the KL distance, we adopt PANNs (KLPANNs) and PaSST (KLPaSST) as classifiers. 2) We use PANNs as the classifier, following Wang et al. [48], to assess generation audio quality without the need for comparison with the ground truth, utilizing the inception score. 3) Semantic alignment is measured using ImageBind [7], following Viertola et al. [46], by extracting visual features from the input video and audio features from the generated audio, then computing the average cosine similarity as the IB-score. 4) Temporal alignment: We use synchronization score (DeSync) to assess audio-visual synchrony. DeSync is predicted by Synchformer [11] as the misalignment (in seconds) between the audio and video.

Implementation Details. In our experiments, all LLMbased agents use the Qwen API [37] with the “qwen-max” model to simulate different agent roles. The visual support agent is implemented using the locally deployed “Qwen2.5VL-7B” model. The retrieval-augmented generation for the predefined audio description knowledge base is built on LlamaIndex 1 and powered by a “qwen-plus” model.

#### 5.2. Main Results

The evaluation metric comparison results are shown in Table 1, where LVAS-Agent outperforms the baseline methods across all metrics in four key dimensions, achieving state-of-the-art performance. Additionally, we visualize and compare the audio waveforms generated by different methods. The quantitative results demonstrate that our approach enables the existing VTA base models to generate higherquality audio in long videos with enhanced semantic and temporal consistency, all without additional training. As shown in the visualized spectrogram comparison in Figure 4, (a) reveals that LVAS-Agent exhibits adaptive capability to video content variations, ensuring a high level of alignment with the video while reducing the omission of key sound effects and minimizing incorrect audio generation. Furthermore, (b) shows that our method, by designing foreground and background audio layers, achieves a multilevel synthesis that enhances its off-screen capability.

Data. Since our method focuses on the task of sound effect synthesis for long videos, which consist of shorter videoaudio pairs, are not suitable for evaluation. Therefore, this paper uses the proposed LVAS-Bench to assess the performance of the Agent-System.

Baselines. To accommodate sound effect synthesis for videos of arbitrary length, the experimental baseline is designed to first segment the video, then apply Video-to-Text (VTA) on each segment, and finally combine the results. We use state-of-the-art open-source methods, FoleyCrafter [57] and MMAudio [3], as VTA tools. FoleyCrafter sup-

#### 5.3. Ablation Study

To validate the effectiveness of the Agent-Framework design, an ablation study was conducted, as shown in Table 2.

1LlamaIndex: https://www.llamaindex.ai/.

|Methods|Distribution Matching|Audio Quality|Semantic Align|Temporal Align|
|---|---|---|---|---|
| |FDVGG↓ FDPANN↓ FDPASST↓ KLPANNs↓ KLPASST↓<br><br>|ISPANNs↑ ISPASST↑<br><br>|IB-Score↑|DeSync↓|

Baseline (FoleyCrafter) 6.61 60.66 637.82 2.68 2.65 4.79 4.34 0.28 1.24 Baseline (MMAudio) 9.48 51.73 588.24 2.02 1.80 3.91 3.05 0.32 0.61 LVAS-Agent (Ours) 5.76 46.16 573.67 1.86 1.77 4.28 3.50 0.33 0.53

Table 1. Comparison of different methods on various evaluation metrics. Lower values (↓) indicate better performance, while higher values (↑) indicate better quality.

Key Components Distribution Matching Audio Quality Semantic Align Temporal Align Video-Structure Chain-of-Thought RAG FDVGG↓ FDPANNs↓ ISPNSS↓ IB-Score↑ DeSync↑

✓ 7.45 77.65 1.85 0.312 0.361 ✓ ✓ 7.41 76.84 1.82 0.319 0.346 ✓ ✓ ✓ 7.12 71.61 1.81 0.336 0.338

Table 2. Ablation Study. Ablating different key components of LVAS-Agent and evaluating performance on LVAS-Bench.

Key components of the LVAS-Agent that contribute to enhancing audio generation quality were identified, including: (1) generating sound effects after refined video segmentation, (2) a Chain-of-Thought process for structured sound effect description and hierarchical generation, and (3) an iterative optimization process leveraging retrieved audio reference documents. The experiment was conducted on 20 randomly selected video cases.

First, integrating the proposed video structuring method into the baseline significantly improved audio quality. This improvement is attributed to content-aware segmentation, which ensures consistency between video content and generated audio. Building on this, incorporating the CoT process for audio description further enhanced both audiovisual synchronization and quality. This is due to CoT’s ability to effectively transform video captions into detailed audio descriptions, systematically reasoning through possible sound effects and accurately identifying appropriate sources. Finally, the retrieval-augmented iterative optimization of audio descriptions further refined the VTA tool’s audio generation, leveraging a domain-specific knowledge base to translate LLM-generated generalized descriptions into precise audio prompts familiar to the VTA model.

#### 5.4. User Study

We conducted a user study involving 30 participants to evaluate our method in comparison with FoleyCrafter [57] and MMAudio [3]. Participants were asked to listen to 10 audio samples generated by each method and rate them on a scale of 1 to 5 across three dimensions: “Audio Quality,” “Video-Audio Consistency,” and “Overall Satisfaction.” Higher scores indicate better performance. As illustrated in Figure 5, the results of the user study demonstrate

that our method outperforms the two baseline approaches across all evaluated aspects.

Figure 5. User study comparing our method with baselines across different aspects. Higher values indicate greater user preference.

### 6. Conclusion

We present LVAS-Agent, a multi-agent framework that systematically tackles long-video dubbing challenges through role-specialized collaborative agents. By decomposing the workflow into scene segmentation, script generation, sound design, and hybrid synthesis, our method overcomes limitations in semantic continuity and temporal alignment inherent to existing approaches. We also release the first dedicated long-video audio synthesis dataset, covering 207 professionally curated videos, named LVAS-Bench. Experimental results demonstrate superior performance in distribution matching, audio quality, and alignment metrics on

LVAS-Bench.

For future work, we aim to develop a large-scale, finely annotated dataset of long-video audio to further advance the development of long-video dubbing models.

### References

- [1] Kan Chen, Chuanxi Zhang, Chen Fang, Zhaowen Wang, Trung Bui, and Ram Nevatia. Visually indicated sound generation by perceptually optimized classification. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pages 0–0, 2018. 3
- [2] Peihao Chen, Yang Zhang, Mingkui Tan, Hongdong Xiao, Deng Huang, and Chuang Gan. Generating visually aligned sound from videos. IEEE Transactions on Image Processing, 29:8292–8302, 2020. 3
- [3] Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. Taming multimodal joint training for high-quality video-to-audio synthesis. arXiv preprint arXiv:2412.15322, 2024. 3, 6, 7, 8
- [4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3
- [5] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision, pages 75–92. Springer, 2025. 5
- [6] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and humanlabeled dataset for audio events. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780, 2017. 7
- [7] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind one embedding space to bind them all. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180–15190, 2023. 7
- [8] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15180–15190, 2023. 3
- [9] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13504–13514, 2024. 3
- [10] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta pro-

- gramming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 3(4):6, 2023. 3
- [11] Jiawei Huang, Yi Ren, Rongjie Huang, Dongchao Yang, Zhenhui Ye, Chen Zhang, Jinglin Liu, Xiang Yin, Zejun Ma, and Zhou Zhao. Make-an-audio 2: Temporal-enhanced textto-audio generation, 2023. 7
- [12] Rongjie Huang, Mingze Li, Dongchao Yang, Jiatong Shi, Xuankai Chang, Zhenhui Ye, Yuning Wu, Zhiqing Hong, Jiawei Huang, Jinglin Liu, et al. Audiogpt: Understanding and generating speech, music, sound, and talking head. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 23802–23804, 2024. 3
- [13] Vladimir Iashin and Esa Rahtu. Taming visually guided sound generation. arXiv preprint arXiv:2110.08791, 2021. 2, 3
- [14] Mojan Javaheripi, S´ebastien Bubeck, Marah Abdin, Jyoti Aneja, Sebastien Bubeck, Caio C´esar Teodoro Mendes, Weizhu Chen, Allie Del Giorno, Ronen Eldan, Sivakanth Gopi, et al. Phi-2: The surprising power of small language models. Microsoft Research Blog, 2023. 3
- [15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment Anything. arXiv preprint arXiv:2304.02643, 2023. 3
- [16] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2880–2894, 2020. 7
- [17] Khaled Koutini, Jan Schl¨uter, Hamid Eghbal-zadeh, and Gerhard Widmer. Efficient training of audio transformers with patchout. In Interspeech 2022, 23rd Annual Conference of the International Speech Communication Association, Incheon, Korea, 18-22 September 2022, pages 2753–2757. ISCA, 2022. 7
- [18] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: Reasoning Segmentation via Large Language Model. arXiv preprint arXiv:2308.00692, 2023. 3
- [19] Chenliang Li, Hehong Chen, Ming Yan, Weizhou Shen, Haiyang Xu, Zhikai Wu, Zhicheng Zhang, Wenmeng Zhou, Yingda Chen, Chen Cheng, et al. Modelscope-agent: Building your customizable agent system with open-source large language models. arXiv preprint arXiv:2309.00986, 2023. 3
- [20] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2024. 3
- [21] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2024. 5
- [22] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection, 2023. 3
- [23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. arXiv preprint arXiv:2304.08485,

2023. 3

- [24] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542,

2024. 3

- [25] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 3
- [26] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Da Li, Pengcheng Lu, Tao Wang, Linmei Hu, Minghui Qiu, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability, 2023. 3
- [27] Simian Luo, Chuanhao Yan, Chenxu Hu, and Hang Zhao. Diff-foley: Synchronized video-to-audio synthesis with latent diffusion models. Advances in Neural Information Processing Systems, 36:48855–48876, 2023. 3
- [28] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models, 2023. 3
- [29] Marvin Minsky. Society of mind. Simon and Schuster, 1988. 3
- [30] Shentong Mo, Jing Shi, and Yapeng Tian. Text-toaudio generation synchronized with videos. arXiv preprint arXiv:2403.07938, 2024. 3
- [31] Shehan Munasinghe, Rusiru Thushara, Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, Mubarak Shah, and Fahad Khan. Pg-video-llava: Pixel grounding large videolanguage models, 2023. 3
- [32] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023. 3
- [33] Vadim Popov, Amantur Amatov, Mikhail Kudinov, Vladimir Gogoryan, Tasnima Sadekova, and Ivan Vovk. Optimal transport in diffusion modeling for conversion tasks in audio domain. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 2
- [34] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. arXiv preprint arXiv:2307.07924, 2023. 3
- [35] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. arXiv preprint arXiv:2405.16009, 2024. 3
- [36] Jihao Qiu, Yuan Zhang, Xi Tang, Lingxi Xie, Tianren Ma, Pengyu Yan, David Doermann, Qixiang Ye, and Yunjie Tian. Artemis: Towards referential understanding in complex videos. arXiv preprint arXiv:2406.00258, 2024. 3
- [37] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le

- Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. 7
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 3
- [39] Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551,

2023. 3

- [40] Roy Sheffer and Yossi Adi. I hear your true colors: Image guided audio generation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 3
- [41] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180,

2023. 3

- [42] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 3
- [43] Yunjie Tian, Lingxi Xie, Xiaopeng Zhang, Jiemin Fang, Haohang Xu, Wei Huang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Semantic-aware generation for self-supervised visual representation learning. arXiv preprint arXiv:2111.13163,

2021. 3

- [44] Yunjie Tian, Lingxi Xie, Zhaozhi Wang, Longhui Wei, Xiaopeng Zhang, Jianbin Jiao, Yaowei Wang, Qi Tian, and Qixiang Ye. Integrally Pre-Trained Transformer Pyramid Networks. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18610–18620. IEEE, 2023. 3
- [45] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Chatterbox: Multi-round multimodal referring and grounding. arXiv preprint arXiv:2401.13307, 2024. 3
- [46] Ilpo Viertola, Vladimir Iashin, and Esa Rahtu. Temporally aligned audio for video with autoregression, 2024. 7
- [47] Heng Wang, Jianbo Ma, Santiago Pascual, Richard Cartwright, and Weidong Cai. V2a-mapper: A lightweight solution for vision-to-audio generation by connecting foundation models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 15492–15501, 2024. 2, 3
- [48] Yongqi Wang, Wenxiang Guo, Rongjie Huang, Jiawei Huang, Zehan Wang, Fuming You, Ruiqi Li, and Zhou Zhao. Frieren: Efficient video-to-audio generation network with

- rectified flow matching. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 7
- [49] Zixuan Wang, Yu-Wing Tai, and Chi-Keung Tang. Audioagent: Leveraging llms for audio generation, editing and composition. arXiv preprint arXiv:2410.03335, 2024. 3
- [50] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. arXiv preprint arXiv:2404.03384, 2024. 3
- [51] Minghao Wu, Jiahao Xu, and Longyue Wang. Transagents: Build your translation company with language agents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 131–141, 2024. 3
- [52] Zhifeng Xie, Shengye Yu, Qile He, and Mengtian Li. Sonicvisionlm: Playing sound with vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26866–26875, 2024. 2, 3
- [53] Yazhou Xing, Yingqing He, Zeyue Tian, Xintao Wang, and Qifeng Chen. Seeing and hearing: Open-domain visualaudio generation with diffusion latent aligners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7151–7161, 2024. 3
- [54] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding, 2023. 3
- [55] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. GPT4RoI: Instruction Tuning Large Language Model on Region-ofInterest. arXiv preprint arXiv:2307.03601, 2023. 3
- [56] Xiaosong Zhang, Yunjie Tian, Lingxi Xie, Wei Huang, Qi Dai, Qixiang Ye, and Qi Tian. Hivit: A simpler and more efficient design of hierarchical vision transformer. In The Eleventh International Conference on Learning Representations, 2022. 3
- [57] Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, and Kai Chen. Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. arXiv preprint arXiv:2407.01494, 2024. 3, 7, 8
- [58] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 3
- [59] Yipin Zhou, Zhaowen Wang, Chen Fang, Trung Bui, and Tamara L Berg. Visual to sound: Generating natural sound for videos in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3550–3558, 2018. 3
- [60] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan. Languagebind: Extending video-language pretraining to nmodality by language-based semantic alignment, 2024. 3

## Long-Video Audio Synthesis with Multi-Agent Collaboration Supplementary Material

### A. System Prompts

Here we show the detailed prompts of our LVAS-Agent, including Storyboarder, Scriptwriter, Designer and Synthesizer. The prompt of Storyboarder includes a prompt for global understanding as shown in Figure 2, a prompt for understanding each of the divided small segments as shown in Figure 3. The Storyboarder’s prompt as shown in Figure 1. The Designer’s prompt as shown in Figure 4. The Synthesizer’s prompts includes a system prompt (as shown in Figure 5) and a prompt for rag (as shown in Figure 6).

## **Role & Task** You are a **professional video editor** responsible for determining whether two video clips should be merged based on their textual descriptions. Your decision must ensure narrative and audiovisual continuity in the final edit.

## **Decision Criteria for Merging** Analyze the relationship between the **last video clip**, the **current video clip**, and the **overall video timeline** by considering:

- 1. **Content Continuity**

- - Are the two clips depicting the same ongoing scene or event?
- - Does the transition between clips maintain a logical progression?

- 2. **Scene & Environment Consistency**

- - Do both clips occur in the same setting?
- - Example of **discontinuity**:
- - A clip in an **outdoor park** followed by a scene **inside a church** → **Not continuous**
- - A **lake with a single tree** vs. a **lake with snow-capped mountains** → **Different settings, not continuous**
- - Example of **continuity**:
- - A **dog standing on stairs** → A **dog jumping down the stairs** → **Same entity, continuous event**

- 3. **Action & Sound Consistency**

- - Are the actions and sounds from one clip naturally leading into the next?
- - Example of **discontinuity**:
- - **Chopping vegetables** followed by **sprinkling seasoning** → **Both are cooking-related but produce different sounds, not continuous**
- - Example of **continuity**:
- - **Tank firing** → **Tank shooting** → **Essentially the same action and sound, should be merged**
- - **Two people dueling with knives** from different perspectives → **Same fight, should be merged**

## **Processing Steps**

- 1. **Analyze the Text Descriptions:**

- - Compare the last clip’s caption, the current clip’s caption, and the whole video’s context.
- - Determine whether the clips belong to the same scene or action sequence.

- 2. **If merging is required:**

- Generate a **new unified description** combining both clips.

- 3. **If merging is NOT required:**

- - Assess whether the current video description needs adjustments for clarity and continuity based on the global context.
- - If no modifications are needed, retain the original description.

## **Input Format** ```plaintext Video Caption: [Last Video Clip Caption], [Current Video Clip Caption], [Whole Video Caption] ```

## **Output Format** If the clips should be merged: ```json {

"merge": "True", "caption": {

"Background": "...", "Entity": "...", "Action": "...", "Summary": "..."

}

} ```

If the clips should NOT be merged: ```json {

"merge": "False", "caption": {

"Background": "...", "Entity": "...", "Action": "...", "Summary": "..."

}

} ``` Figure 1. Storyboarder Prompt

**Role** You are a professional video editor with expertise in scene analysis, timeline construction, and event identification.

**Task Description** Given a video, analyze its content to provide a detailed summary of the following:

- 1. Identify the main scenes and their sequence, highlighting key events and actions.
- 2. Construct an approximate timeline of the video, emphasizing transitions between key moments or actions.
- 3. Summarize the video’s content in a structured and coherent manner.

**Input** A video clip.

**Output Format** ```

- - Scene Summary: (Describe the key scenes and their sequence in detail)
- - Timeline: (Provide a detailed summary of key events, highlighting transitions between scenes)

```

**Output Example** ```

- - Scene Summary: The video begins with a busy city street, showing a large crowd and moving vehicles. The next scene transitions to a park where a dog chases a ball and interacts with a child. The final scene cuts to the child throwing the ball into a pond while the dog watches.
- - Timeline: The video starts with the city street scene, then transitions to the park where the dog and child interact. It ends with the child throwing the ball into the pond. ```

Figure 2. Scriptwriter Prompt: full video understanding

**Role** You are a video analyst with expertise in understanding and interpreting various video clips.

**Task Description** Given a video clip, please perform the following tasks:

- - Analyze the clip, identifying entities, their actions, and the video scene. Provide text descriptions as required by the output format. The entities must be real and present in the video.
- - The full video description provided is rough and not entirely accurate. You need to first analyze the current clip and then summarize it, considering the existing full description.

**Constraints**

- - The analysis is strictly limited to the provided video clip; avoid speculating or using background information beyond the video.
- - The summary must be strictly based on the video content, without personal assumptions or creative additions.
- - Background sounds typically include weather conditions (e.g., rain, snow, thunderstorms) or real-world sounds (e.g., crowd parades, train horns).
- - Avoid using abstract atmospheres like those of a futuristic city, forest, or the universe as background sounds.

**Input Data** {Video}

**Output Format** ```

- - Background:
- - Entity:
- - Action:
- - Video caption:

```

Figure 3. Scriptwriter Prompt: video segment understanding

You are a sound effects specialist. Your task is to generate precise and realistic sound effect descriptions for a video clip based on its textual description, just like a professional foley artist. Your output should be structured in JSON format.

### **Instructions** Follow these steps carefully to ensure accurate and contextually appropriate sound effect descriptions:

- 1. **Identify Sound-Producing Entities & Actions**

- - Extract key entities (e.g., people, animals, objects) and their actions from the video description.
- - Only describe actions that naturally produce a sound. For example, "a car accelerating" makes a sound, but "a sunset" does not.
- - Format: `[Entity] makes [adjective] sound` or `[Action] makes sound`.

- 2. **Determine Background Ambience**

- - If the environment contributes to the soundscape (e.g., wind, rain, ocean waves), describe it as the background audio.
- - Avoid vague terms such as "tense atmosphere" or "futuristic hum"—use concrete environmental sounds.
- - Background audio should be clearly distinguishable from main sounds.

- 3. **Prioritize Primary vs. Secondary Sounds**

- If the scene has a dominant action sound (e.g., car racing), it should be the main audio, while secondary sounds (e.g., crowd cheering) should be background if necessary.

- 4. **Determine Sound Output Based on Reality** Choose the most accurate option based on the video description:

- - **Option 1:** If no entity or ambient sound is relevant → `"audio": []`, `"background": []`.
- - **Option 2:** If there is only an ambient sound → `"background": [ambient sound]`, `"audio": []`.
- - **Option 3:** If entities/zactions produce sound and there is ambient noise → `"audio": [entity sound]`, `"background": [ambient sound]`.

- 5. **Avoid Redundant Sounds**

- - Do not repeat the same sound in both `"audio"` and `"background"`.
- - Example: `"background": ["wind"], "audio": ["wind noise"]` is redundant—only keep `"audio": ["wind noise"]`.

### **Example Outputs**

- #### **Example 1**

**Video Description:** "Intense space battle with ships firing and dodging debris."

**Output:** ```json {

"background": [], "audio": ["explosions", "cannon fire"] “volume”: 40

} ```

- #### **Example 2**

**Video Description:** "Two warriors engage in an intense sword fight."

**Output:** ```json {

"background": [], "audio": ["swords clashing with sharp metallic sounds"]

} ```

### **Input & Output Format** #### **Input:** ```plaintext {video description} ``` #### **Output:** ```json {

"background": [ambient sounds], "audio": [main sounds]

} ```

Figure 4. Designer Prompt

**Role**:

- - You are an assistant responsible for providing accurate audio labels based on the input video description, audio description, and alternative audio label references.

**Tasks and Requirements**:

- - If the original label is ["None"], retain it without change.
- - If no suitable match is found, keep the original description.
- - Avoid using human voices in the "audio" label unless there's a clear context, like cheering or similar actions.
- - Estimate the background sound based on the entities in the video. For example, if a train is present, the background should be a train whistle. If the video depicts a war scene, gunfire could be the background sound.

**Input**:

- - Audio Label Alternative Reference: ...
- - Clip Video Caption: ...
- - Audio Description: ...

**Output**: Please output the following JSON string with no additional content: ```

{

"background audio": [audio],

"audio": [audio, …]} ```

Figure 5. Synthesizer Prompt

**Role & Task:**

You are an AI assistant specializing in **audio analysis and labeling**. Your task is to retrieve the most appropriate and specific **audio labels** from a predefined reference document

based on the provided **video descriptions** and **original audio descriptions**.

**Guidelines for Retrieval & Labeling:**

- 1. **Strict Compatibility:** The retrieved audio label must be highly compatible with both the **video description** and the**original audio description**. If no suitable label is found,

**do not** provide a replacement.

- 2. **Document Format:** The reference document follows the structure:

- - `"Audio Category": "Specific Label"`
- - Output only the **label** after `:` (do not include category titles).

- 3. **Replacement Strategy:**

- - Prioritize **semantic similarity** when suggesting replacements.
- - If no exact match is found, focus on the **type of sound produced**, disregarding the sound source.
- - Example adjustments:
- - Searching for "building explosion," but only "volcanic explosion" is available → Output **"explosion"**
- - Searching for "tank firing," but no exact match exists → Find related **artillery firing** labels
- - Searching for "airplane engine roar," but no exact match exists → Look for **airplane-related** sounds, as airplane noise originates from its engine

- 4. **Context Awareness:** Consider both **video captions** and **raw audio descriptions** for accurate label selection.
- 5. **Strict Label Set Adherence:** Stay strictly within the available labels in the reference document.
- 6. **Handling 'None' Labels:** If the raw audio description is `["None"]`, retain `["None"]` without suggesting alternatives.

**Response Format:** ```

- [Raw Audio Label 1]: The optimized labels that can be referred to are as follows:

- - [Suggested Label]: [Brief acoustic explanation]
- - ...

- [Raw Audio Label 2]: The optimized labels that can be referred to are as follows:

- - [Suggested Label]: [Brief acoustic explanation]
- - ... ```

Figure 6. Synthesizer Prompt: Retrieval Augmented Generation(RAG)

