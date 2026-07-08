# arXiv:2503.22952v1[cs.CV]29Mar2025

## OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts

Yuxuan Wang1,2, Yueqian Wang2,3, Bo Chen1,2,4, Tong Wu1,2, Dongyan Zhao2,3, Zilong Zheng1,2

1 Beijing Institute for General Artificial Intelligence 2 State Key Laboratory of General Artificial Intelligence 3 Wangxuan Institute of Computer Technology, Peking University 4 X-LANCE Lab, Shanghai Jiao Tong University {wangyuxuan1, zlzheng}@bigai.ai https://omnimmi.github.io

##### Abstract

The rapid advancement of multi-modal language models (MLLMs) like GPT-4o has propelled the development of Omni language models, designed to process and proactively respond to continuous streams of multi-modal data. Despite their potential, evaluating their real-world interactive capabilities in streaming video contexts remains a formidable challenge. In this work, we introduce OmniMMI, a comprehensive multi-modal interaction benchmark tailored for OmniLLMs in streaming video contexts. OmniMMI encompasses over 1,121 videos and 2,290 questions, addressing two critical yet underexplored challenges in existing video benchmarks: streaming video understanding and proactive reasoning, across six distinct subtasks. Moreover, we propose a novel framework, Multi-modal Multiplexing Modeling (M4), designed to enable an inference-efficient streaming model that can see, listen while generating. Extensive experimental results reveal that the existing MLLMs fall short in interactive streaming understanding, particularly struggling with proactive tasks and multi-turn queries. Our proposed M4, though lightweight, demonstrates a significant improvement in handling proactive tasks and real-time interactions.

##### 1. Introduction

The burgeoning field of multi-modal large language models (MLLMs), exemplified by GPT-4o [32] and Gemini Pro [36], marks a significant leap towards embodied agentic intelligence by incorporating multi-modal encoders within pre-trained large language models (LLMs), such as video understanding [25, 38, 53], audio comprehension [7], and speech-to-speech dialogue [8, 9, 46, 47], etc. The overarching goal is (i) to transcend the general capabilities of LLMs to process and respond to continuous streams of multi-modal dynamics, encompassing text, vision, and

Corresponding author: Zilong Zheng ⟨zlzheng@bigai.ai⟩

speech modalities, i.e., Omni Large Language Models (OmniLLMs), and (ii) to derive interactive systems that can take the first-person perspective to interact with the real world. However, this rapid development raises a crucial question: How can we effectively evaluate the real-world interactive capabilities of OmniLLMs in streaming video contexts? Addressing this question is pivotal to validating their design efficacy and enhancing their performance for comprehensive open-world multi-modal understanding.

In response, a vast number of benchmarks has been launched with different focuses on long-form video understanding [10, 34, 59], comprehensive video analysis [10], or audio-video understanding [22], etc. However, most of these benchmarks take in the entire video sequence as input, some of which use frame selection techniques with slight information scarification, to produce final answers. These tasks, though challenging, are far from the real-world interactive scenarios where videos are taken in as a streaming sequence; refer to Tab. 1 for benchmark comparisons. More recently, OmniBench [22] has been introduced to evaluate models’ capabilities on visual, acoustic, and textual inputs simultaneously. However, only the last image frame is considered visual input, while the video dynamics across streaming contexts and interactive features are overlooked.

To bridge the critical gap, we introduce Multi-modal Interaction Benchmark for OmniLLMs (OmniMMI), which aims at comprehensively evaluating the interactive capabilities of streaming video context in the open world (Fig. 1). We start by formalizing the task of streaming multi-modal understanding (§ 3.2). Apart from challenges identified by prior long-form video-audio benchmarks (e.g., temporal dynamics [10, 59] and multi-modal localization [37, 42]), OmniMMI considers two featured obstacles for real-time interactions.

• Streaming Temporal State Awareness. Streaming video understanding must build an understanding w.r.t. the current and historical temporal state incrementally, without

###### Streaming Video Understanding

Dynamic State Grounding I am tasked with Peel, chop and cook the second

Action Prediction

[Figure 1]

[Figure 2]

How many cars have shown up?

potato. What is next step?

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

00:05 00:10 00:17

A: 1

A: 2 A: 3

A: pick up potato

Multi-Turn Dependency Reasoning

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Q: Who is organizing the backpack? A: A woman

Q: Who is <a woman> talk to? A: A man

Q: What is <a man> doing? A: Sitting on the bed.

###### Proactive Reasoning

###### Proactive Alerting

###### Speaker Identification

[Figure 15]

Inform me when a cat is getting a shot.

A: Informed

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

00:05 00:15 00:20 00:25

This is Bob This is Wanita

###### Proactive Turn-Taking

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

00:05 00:15 00:20 00:25

[Figure 29]

[Figure 30]

What happens to the man in black clothes after he stands on the cliff? A: He is caught by a net laid by a helicopter.

[Figure 31]

Oh well, it is what it is. No point in overthinking it. A: <Silence>

Who wants to serve drink? A: Wanita

- Figure 1. OmniMMI consists of two categories of multi-modal interactive challenges: streaming video understanding (top) and proactive reasoning (bottom). Each query is processed into natural language text and synthetic audio as input.

accessing the future context. This contrasts with traditional MLLMs that can leverage the entire multi-modal contexts, posing challenges in our distinguished tasks of action prediction (AP), state grounding (SG) and multiturn dependencies (MD) (§3.2.1).

ics relevant to the multi-modal contexts (Fig. 2). Notably, to enhance the interactive feature of this benchmark, we designed multi-turn questions (up to 3 turns) where the next response is based on the answer from previous turns; see dynamic state grounding and multi-turn dependency reasoning as examples.

• Proactive Reasoning and Turn-Taking. Generating responses proactively and appropriately anticipating the turn-taking time spot w.r.t. user’s intentions and dynamic contexts is a crucial feature for general interactive agents. This typically requires models to identify speakers (SI), distinguish between noise or legitimate query (PT), and proactively initiate a response (PA) (§3.2.2).

Using OmniMMI, we make a thorough evaluation of various well-known video large language models (VideoLLMs) and accessible OmniLLMs across all tasks. Surprisingly, most models encounter challenges in multiturn tasks involving streaming video, where they struggle beyond a single reasoning step, thus revealing limitations in dynamic settings. In the realm of audio-visual interaction, models that process both audio and visual inputs do not outperform those that handle visual inputs alone, indicating inadequate modality alignment. Furthermore, increasing the model size does not lead to improved performance; instead, models capable of processing longer input lengths demonstrate superior results, highlighting the importance of balancing input length with memory efficiency.

In light of this, OmniMMI is crafted as the first-ever comprehensive OmniLLM benchmark to address the aforementioned streaming and interactive challenges. As exemplified in Fig. 1, we curate a dataset of 1,121 videos sourced from YouTube and open-sourced video-audio data with an average duration of 324 seconds. 2,290 questions are manually annotated and reviewed w.r.t. both visual and auditory information from video inputs, encompassing different top-

Table 1. Comparison with existing Video Benchmark OmniMMI is the first comprehensive benchmark that focuses on streaming and interactive VideoLLMs. * only the last frame of videos are taken as input. † is not a real-time setting.

Length

Multi-Hop Contain-Ego Streaming Proactive Interactive Video(s) #Turn

Benchmark Modality #Videos #Questions

MVBench [20] Video 3,641 4,000 16.0 1 ✗ ✓ ✗ ✗ ✗ Video-Bench [31] Video 5,917 17,036 56.0 1 ✗ ✗ ✗ ✗ ✗ EgoSchema [29] Video 5,063 5,063 180.0 1 ✗ ✓ ✗ ✗ ✗ VideoMME [10] Video, Audio 900 2,700 1017.9 1 ✗ ✗ ✗ ✗ ✗ LongVideoBench [44] Video 3,763 6,678 473.0 1 ✗ ✗ ✗ ✗ ✗ MLVU [59] Video 2,593 2,593 720.0 1 ✗ ✓ ✗ ✗ ✗ MultiHop-EgoQA [5] Video 360 1,080 180.0 1 ✓ ✓ ✗ ✗ ✗ OmniBench [22] Image*,Audio - 1,142 9.2 1 ✗ ✓ ✗ ✗ ✗ StreamingBench [26] Video, Audio 900 4,500 243.1 1 ✗ ✗ ✓ ✓† ✗ OvO-Bench [23] Video 644 2,814 428.9 1 ✗ ✓ ✓ ✓† ✗

OmniMMI (Ours) Video, Audio 1,121 2,290 324.3 1-3 ✓ ✓ ✓ ✓ ✓

To margin towards real-time interactive reasoning, in §4, we devise a novel and robust framework, Multi-modal Multiplexing Modeling (M4), taking inspirations from duplexing modeling of auditory models [27, 30, 46]. We crafted a small video-free SFT data, M4-IT, for proactive turn-taking awareness. As such, M4 can be built upon any pre-trained VideoLLMs, enabling an inference-efficient streaming model that can see, listen while generating.

##### 2. Related Work

Omni Large Language Models The advancement of OmniLLMs represents a notable achievement in MLLMs, aiming for real-time comprehension and processing of diverse modalities. Significant contributions in this domain include GPT-4o and Project Astra, which manage and generate multi-modal inputs and outputs encompassing text, audio, images, and videos. Concurrently, the opensource community has developed models like VITA [11] and Ocean-Omni [21], which integrate distinct models for enhanced non-awakening and interrupting capabilities. Additionally, audio-based conversational models such as LSLM [27] and mini-Omni [46, 47] have emerged, utilizing text-instructed speech generation to facilitate real-time speech interactions while maintaining strong language proficiency. Despite these advancements, existing models often lack proactive reasoning abilities (Sec. 3.2.2) without additional computational overhead. Addressing this gap, we introduce M4, an interactive framework for MLLMs that enables proactive reasoning without necessitating extra forward computation steps or video-specific training.

Video Understanding Benchmarks Early general video understanding benchmarks [16, 20, 31, 41, 48, 52] were introduced to evaluate models’ capabilities in general videolanguage understanding. Other works [43, 45] have focused on temporal grounding for dynamic video content. More recently, several benchmarks for long video understanding [10, 29, 44, 59] have been proposed. The MultiHopEgoQA dataset [5] introduces a multi-hop video questionanswering dataset with temporal evidence to assess models’

multi-hop reasoning abilities over relatively long videos. OmniBench [22] extends visual information to include audio, proposing an audio-focused benchmark complemented by visual information. However, there is a lack of comprehensive benchmarks for streaming video understanding and proactive reasoning for interactive VideoLLMs. To address this gap, we propose OmniMMI, aiming to encourage further advancements in this area.

Streaming Video Understanding The realm of MLLMs [34, 38, 40, 54] have achieved superior performance in various video-centric tasks by employing a progressive training paradigm that unifies different selfor weakly-supervised learning frameworks. Some recent VideoLLMs [15, 33, 42, 58, 60] enables video processing in an online manner and store past video information in a memory bank, facilitating long-term video analysis without exceeding computational constraints. VideoLLMonline [3] further addresses the challenge of integrating diverse data modalities by effectively introducing a special token after each frame with a binary classification task. These advancements underscore the transformative potential of multimodal and interactive video understanding technologies, promising innovative applications across various fields as models continue to improve in their ability to integrate and process diverse data modalities.

##### 3. The OmniMMI Dataset

###### 3.1. Dataset Construction

Data Source The existing datasets include Ego4d [13], COIN [35], Shot2Story20K [14], QVHighlight [18], and MLVU [59], which encompass both egocentric and opendomain videos across diverse topics. However, most of these datasets are not specifically aligned with typical interactive scenarios, such as interactions involving a camera. To address the issue and minimize data contamination, we integrated the test and validation sets from the aforementioned datasets with newly collected video footage. Specifically, we augmented these datasets by sourcing additional videos

Table 2. OmniMMI detailed statistics. Vid.(s): video duration. Que.: question words.

Streaming Proactive SG AP MP PT PA SI

Statistic

What is the person doing?

#Videos 300 200 300 78 200 200 #Queries 704 200 786 200 200 200 Avg. Turns 2.35 1.00 2.62 1.00 1.00 1.00 Avg. Vid.(s) 350.82 234.95 374.80 2004.10 149.82 549.64 Avg. Que. 16.00 25.99 26.27 8.45 17.49 60.91

Which team won in the video clip mentioned?

Who is working with

the interior designer? My job is to prepare a yoghurt. The progress is visible in the video, What’s the next step?

250

Streaming Understanding Proactive Reasoning

NumberofVideos

200

#### …

What colors are the cosmetics?

150

100

Has the man touched the box with both hands?

50

0

(0,5]s(5,10]s(10,15]s(15,20]s(20,30]s(30,60]s(2,4]min(4,6]min(6,8]min(8,10]min(10,121]min

Figure 3. Distribution of video duration length.

- Figure 2. Distribution and examples of different types of query prompts.

description of the annotation procedure and the annotation interface is provided in SM.

from YouTube, utilizing 425 keywords pertinent to frequently observed interactive environments. For each keyword, we downloaded a maximum of 50 videos, each with a duration of less than one hour. Ultimately, we conducted a manual review to filter out low-quality videos, resulting in a curated set of 78 videos. A comprehensive list of these keywords is provided in the supplementary material (SM).

###### 3.1.1. Statistics

In Tab. 1, we present a comparison of our benchmark with existing popular benchmarks in the field of video understanding. Our dataset consists of 1,121 videos and 2,290 queries, with the average video length exceeding five minutes. Notably, a segment of our benchmark includes multiple turns, requiring models to correctly answer all associated questions to be labeled as a hit. This approach simulates a streaming scenario, thereby introducing an additional challenge. Overall, our work is the first to provide a comprehensive benchmark specifically designed to evaluate the efficacy of streaming video understanding models. Tab. 2 presents comprehensive statistics of the videos and queries across all six subtasks. We further delineate topics of query prompts into different categories to demonstrate the diversity of questions. The distribution and examples are shown in Fig. 2. As seen, queries related to action/activity predominates in our benchmark, reflecting the dynamic nature of streaming video.

Annotation The annotation process requires annotators to analyze both visual and auditory elements of input videos. Initially, annotators review the video chronologically and annotate the time span of relevant actions or states, tailored to the specific task, particularly for streaming video understanding. We provide question-type prompts to guide annotators in focusing on various video aspects, such as actions or object attributes, detailed in SM. For tasks involving proactive alerting and turn-taking, annotators record timestamps for significant events or their conclusions. In speaker identification tasks, annotators mark time spans of human introductions and subsequently label individuals’ names following activities or special situations. To ensure benchmark quality, annotators must review the video again, considering the noted spans and original dataset information, to refine questions and answers based on their annotations.

###### 3.2. Benchmark Tasks

Streaming Video Formulation We formalize an input streaming video as an infinite video sequence V∞ = {v1,··· ,vt,···}, where vt corresponds to the temporal time t and Vt is the video history up to time t. ∆t is the minimum temporal unit that denotes the interval between consecutive frames. At timestamp t, let qtn denote the nth natural language user query in the form of text or audio, Htn = {(qt1,a1t),··· ,(qtn−1,ant −1)} denote the interaction history prior to t, the task is to generate next response w.r.t. the input streaming context, i.e., ant = f(Vt,Htn,qtn),

Quality Review To ensure accuracy, a second annotator reviews the initial annotations, focusing on the consistency of the questions, answers, and time spans. Any inconsistencies identified are documented and corrected. We then calculate the inter-annotator agreement to evaluate the consistency and reliability of the annotations. We re-used VIA tool1 for the annotation and reviewing process. A detailed

1https://www.robots.ox.ac.uk/˜vgg/software/via

where ant is the n-th predicted answer at time t, Of note, at could be an action to be executed in a short-term future; see Sec. 3.2.2 for exemplar tasks.

###### 3.2.1. Streaming Video Understanding

Dynamic State Grounding (SG): This task aims to ascertain the dynamic states of a streaming video at different timestamps. We repeatedly pose the same query at different temporal states, i.e., {qτ = qt;τ ∼ [1,t)}. The objective is to determine the correct answer aiτ for each timestamp, where aiτ depends on Vτ−µ∆t, with µ∆t being a short duration preceding the time τ.

Action Planning (AP): Given a natural language goal and a historical sequence from the streaming video, this task is to identify the correct next action to achieve the goal.

Multi-turn Dependency (MD) Reasoning: This task involves answering a series of questions where each subsequent question depends on the answer to the previous one. The requirement is that ai forms a part of qi+1, i.e., ai = F(qi,{a1,a2,...,ai−1}) and qi+1 = G(qi,ai), where G(·,·) generates the next question based on the current answer in a predefined question template. The correctness of each answer ai is contingent upon the accuracy of the preceding answers.

Metrics We use GPT4-o to assess each step of SG and MD, and the final accuracy is averaged over the correct responses to all questions in the sequence. For AP, the evaluation is computed as the accuracy of selecting a response from a predefined vocabulary set of action candidates.

###### 3.2.2. Proactive Reasoning

Speaker Identification (SI): In streaming videos featuring multiple individuals, a proficient model should accurately identify speakers to better comprehend multi-party dialogues. Given an introduction by oneself or others, the question qti pertains to the current situation and requires identifying the name of the corresponding speaker.

Proactive Alerting (PA): A critical application of streaming video understanding is in surveillance, where the model is expected to notify humans of potentially dangerous situations. The desired response is a proactive alerting function at = A(v) to be executed on the consecutive video sequences until appropriate altering information (e.g., “informed” in Fig. 1) is proactively announced.

Proactive Turn-taking (PT): Streaming videos often contain significant noise. A competent model should distinguish between queries that require a response and those that are merely noise, necessitating silence. We construct a series of queries that do not require a response to evaluate the model’s ability to resist responding to noise queries.

Metrics We employ GPT4-o to evaluate the accuracy of the SI metric. For PA, the timestamp of the model’s initial proactive response is recorded and considered a successful instance if it occurs within the designated timeframe. For

PT, accuracy is determined by calculating the percentage of instances where no response is generated.

##### 4. Multi-modal Multiplexing Modeling

We evaluate a number of popular open-source MLLMs on OmniMMI in Tab. 3. Surprisingly, the existing MLLMs are far from satisfactory in streaming video understanding. To fill the gap, we develop a robust OmniLLM baseline, Multimodal Multiplexing Modeling, which is dubbed as M4.

Motivated by recent advances in speech LMs [27, 46], M4 formulates the interactive challenges of real-time multimodal communications with multiplexing modeling [30], a technique that enables LMs process multiple inputs simultaneously with a single compact representation. Compared with traditional VideoLLMs and OmniLLMs, M4 presents advances in:

- • Proactive Generation. A critical aspect of streaming video understanding is the model’s ability to proactively generate the next response without human intervention.
- • Proactive Interruption. When presented with a new query, M4 determines whether it is legitimate or merely noise in a single forward step.
- • Efficient Parallel Decoding. With multiplexing inputs, M4 decodes the next token in parallel to the inputs.

Proactive Generation Most current methodologies [3, 11, 60] employ special tokens in conjunction with binary classification tasks and threshold settings to enable continuous narration. However, the efficacy of these special tokens is heavily contingent upon the chosen threshold settings, leading to significant obstacles for generalizing across various domains and video-language models. Building on insights from recent works [32, 47], we ask: whether it is possible to enable real-time proactive generation without supplementing time-consuming video-specific training?

Solution We answer the question with affirmation. In M4, we derive an attention-based inference method, namely highlight spot, by harnessing the potential within pretrained VideoLLMs. Given a streaming video V∞ and a query qt as in Sec. 3.2, the algorithm goes as follows; refer to SM for a formalized pseudo-code:

- ▷ Step I: Streaming KV Cache. For each in-coming frame

v, we pre-compute K = Wkv and V = WV v vectors to form a KV cache. The attention scores between the query q and the frames are calculated w.r.t. the KV cache, i.e., s = softmax qK

T

√dk , with their mean and variance as µ,σ.

- ▷ Step II: Highlight Spot Max-heap. Indices of frames whose attention scores exceed the Gaussian average µ+α× σ are stored in a max heap, where α is a Gaussian factor.
- ▷ Step III: Hit Computation. The peak index from maxheap is extracted. If a frame index has a higher occurrence frequency than a predetermined threshold, it is designated as an “alert”, triggering a response generation.

###### Proactive Interruption Highlight Spot Max-Heap Parallel Decoding

[Figure 32]

𝑣𝑣 𝑞𝑞1 𝑡𝑡1 𝑡𝑡2 𝑡𝑡3 𝑞𝑞2 𝑞𝑞3 𝑡𝑡1

[Figure 33]

𝑣𝑣𝑞𝑞𝑡𝑡𝑡𝑡𝑞𝑞𝑞𝑞𝑡𝑡𝑡𝑡1122313

𝑣𝑣 𝑡𝑡1 𝑡𝑡2 𝑡𝑡3 𝑛𝑛1 𝑡𝑡1 𝑡𝑡2

[Figure 34]

𝑣𝑣 𝑞𝑞1 𝑡𝑡1 𝑡𝑡2 𝑞𝑞2 𝑞𝑞3 𝑡𝑡1

[Figure 35]

- Figure 4. Multiplexing Modeling of M4. v is the streaming video, qi denotes the input query, ti indicates the generated token, ni denotes noise token which will be discarded from the KVCache. The streaming video KVCache is computed to trigger a highlight spot index for the next response generation. Proactive interruption is facilitated through the computation of specific tokens designed for noise and stop signals. The parallel decoding takes mask strategy with dynamic KVCache to process multiple queries in one forward step.

Interruption Detection Starting Detection In this process, we calculate the probability of the “<bos>” token as a reference point. Drawing inspiration from Cai et al. [2], we utilize the reciprocal of perplexity as the threshold for identifying this special token.

moved from the KVCache, allowing the continuation of the previous generation process. If the query is deemed legitimate, we proceed to decode the new query with the video context, while masking the interrupted sequence being decoded. This approach enables the processing of new queries at any forward step, thereby maintaining low latency.

p(xn+k | x1,x2,...,xn+k−1) > β · exp(−S (p(· | x1,x2,...,xn+k−1))), (1)

M4-IT Building on the aforementioned framework, we crafted a small video-free synthetic instruction finetuning dataset, M4-IT, with the assistance of GPT-4o. M4-IT comprises four components: (i) the original instruction, which is a data replay from the instruction data of our base model, in our work, we use the LLaVA-NeXT [54]; (ii) interleaved image-text instruction, which is created by reordering the question and image components of the original instruction; (iii) noise instruction, where GPT-4 is prompted to automatically generate statements that do not require a response; and (iv) stop instruction, where GPT-4 is prompted to generate stop phrases for the stop instruction. Detailed descriptions regarding the instruction construction pipeline and prompts are provided in SM. An overall cost of $4.91 was incurred to construct the instruction dataset.

where β is a scaling factor, S(·) is the entropy function. The threshold for noise detection is dependent on the perplexity of the model. When there is a larger perplexity, the threshold is reduced, indicating the query is more like a noise that does not need a response.

Stopping Detection Knowing when to stop is a critical feature of an interactive system, which we consider essential for developing a duplex system. Similar to noise detection, when presented with a new query, we assess whether to halt the generation process by calculating the probability of the “<eos>” token in a single forward pass. This decision is made using the same threshold employed in noise detection. Parallel Decoding We enhance inference speed to achieve real-time interactions through parallel decoding [12, 50, 55, 57]. As illustrated in Fig. 4, when the model is generating new tokens and a new input query arises, the model decodes the next token alongside the original token using a combination of causal masks, prefix masks, and block masks. Specifically, the causal mask is applied for the language model, the prefix mask pertains to the video context, and the block mask is designed to separate the decoding procedures of different queries in parallel. This method allows for the prediction of the next token while responding to a new input query in a single forward pass. To enhance the model’s resilience to noise, once the probability of the next token for the new input query is obtained, we evaluate whether it constitutes noise. If it does, the token is re-

##### 5. Experiments

Setup We perform exhaustive evaluations of existing video-language models on OmniMMI. We have meticulously selected three categories of baseline models for our analysis: commercial VideoLLMs, open-source VideoLLMs, and LAVLMs, which encompass a spectrum from visual to audio modalities. Within the open-source VideoLLMs, we further explore models that vary in scalability, context length, and real-time design capabilities.

###### 5.1. Main Results

Tab. 3 presents the evaluation results. In summary, our analysis yields three key observations.

- Table 3. Performance comparison of existing VideoLLMs on OmniMMI. The 1st, 2nd, 3rd of SG and MD tasks represent the cumulative accuracy up to and including these stages. The “avg.” indicates average accuracy across all data points.

Num Frames

SG

MD

AP

Models LLM

SI PA PT 1st 2nd 3rd avg. 1st 2nd 3rd avg.

Commercial Video LLMs

Gemini-1.5-Pro [36] - 128 52.33 19.67 9.35 16.33 43.00 35.00 16.26 7.14 12.00 38.50 ✗ ✗ GPT-4o [32] - 50 48.67 16.95 5.61 15.00 39.50 34.33 15.57 7.65 12.33 17.00 ✗ ✗

Open-source Video LLMs

VideoChatGPT [28] LLaMA-7B 100 35.33 4.7 1.87 3.33 33.50 18.00 3.11 0.51 3.00 3.50 ✗ ✗ VideoChat2 [20] Vicuna-7B 8 19.67 2.37 0.93 2.33 27.50 16.33 3.81 0.51 2.67 1.00 ✗ ✗ Video-LLaVA [25] Vicuna-7B 8 32.00 1.69 0.00 1.67 28.00 22.67 5.19 1.02 3.33 2.50 ✗ ✗ LLaMA-VID [24] Vicuna-7B 128 29.67 2.38 0.00 2.33 29.00 21.33 3.80 0.51 2.67 7.50 ✗ ✗ MiniGPT4-Video [1] Mistral-7B 45 25.00 4.75 1.87 4.00 23.00 12.67 2.08 0.51 1.67 3.00 ✗ ✗ PLLaVA [49] Vicuna-7B 16 37.33 3.73 0.93 3.33 30.00 21.00 3.46 0.00 1.33 3.00 ✗ ✗ LLaVA-NeXT-Video [56] Vicuna-7B 32 30.33 2.37 0.93 3.00 30.50 17.00 2.08 0.51 2.00 1.50 ✗ ✗ ShareGPT4Video [4] Llama3-8B 16 34.00 2.03 0.93 2.00 29.00 20.33 3.46 0.00 2.00 4.50 ✗ ✗ LLaMA-VID-13B [24] Vicuna-13B 128 33.33 2.03 0.00 1.33 30.50 22.67 3.46 0.51 3.33 8.50 ✗ ✗ PLLaVA-13B [49] Vicuna-13B 16 41.33 3.39 0.00 2.67 25.00 25.67 5.54 2.04 4.33 6.50 ✗ ✗ PLLaVA-34B [49] Yi-34B 16 29.00 4.07 0.00 3.67 28.50 18.67 4.50 0.00 3.00 5.00 ✗ ✗ LLaVA-NeXT-Video-34B [56] Yi-34B 32 30.33 2.71 0.00 2.67 32.50 14.67 2.08 0.51 1.67 1.50 ✗ ✗

LongVA [54] Qwen2-7B 32 33.33 4.07 0.00 3.33 37.50 33.33 4.07 0.00 2.33 3.00 ✗ ✗ LongVILA [51] Llama3-8B 128 39.00 4.41 0.93 4.33 39.50 39.00 4.41 0.93 3.00 10.00 ✗ ✗ LongLLaVA [39] Jamba-9B 128 36.33 3.73 0.00 3.33 29.00 36.33 3.73 0.00 3.67 10.00 ✗ ✗ VideoLLM-online [3] Llama3-8B 1 fps 18.00 4.75 0.00 4.67 35.00 18.00 4.75 0.00 1.33 0.00 0.50 ✗ VideoLLaMB [42] Vicuna-7B 32 / 1 fps 32.67 2.71 0.00 2.33 29.50 32.67 2.71 0.00 3.00 3.00 0.00 ✗ IXC2.5-OL [] Qwen2-1.5B 32 40.33 5.08 0.00 4.03 30.50 26.00 4.50 1.52 4.00 23.0 ✗ ?

OmniLLMs VideoLLaMA2 [6] Qwen2-7B 8 41.00 12.88 0.00 10.33 35.00 23.33 4.15 0.51 3.00 5.00 ✗ ✗ VITA [11] Mistrl-8×7B 16 8.67 0.00 0.00 0.00 39.00 11.33 3.11 1.52 2.00 1.50 ✗ 67.00 MiniOmini2 [47] Qwen2-0.5B 1 17.00 5.08 0.93 4.67 14.00 6.00 1.00 0.00 1.00 1.00 ✗ ✗ M4 (ours) Qwen2-7B 32 / 1 fps 35.67 6.44 1.87 5.67 33.5 35.67 6.44 1.87 1.67 9.00 25.50 62.00 M4-a(ours) Qwen2-7B 32 / 1 fps 28.33 2.37 0.00 2.00 13.00 19.33 3.11 0.51 3.00 7.50 1.50 68.5

- • Challenges in Multi-Turn Tasks for Streaming Video. In the context of multi-turn tasks such as State Grounding and the Multi-Turn Dependency task, models exhibit a notable decline in performance when required to handle more than a single state or reasoning step. When tasked with managing three states or reasoning steps, the majority of open-source models fail to accurately address all inquiries. These results underscore the limitations of current methodologies in managing dynamic environments and performing multi-turn reasoning, despite their demonstrated efficacy in static video scenarios.
- • Limitations in Audio-Visual Interaction. Although our benchmark is explicitly designed to evaluate visual-audio interaction, current open-source models equipped with both visual and audio inputs do not outperform those with solely visual inputs. This discrepancy highlights a deficiency in the alignment of audio and visual features. Moreover, models with specialized speech training [11, 47] perform significantly worse than text input models, emphasizing the critical need for effective alignment and integration of multimodal inputs.
- • Model Size vs. Input Length. Our experiments reveal that increasing model size does not necessarily enhance performance in streaming video tasks. Models with 7B parameters achieve performance levels comparable

to those of larger models while maintaining greater efficiency. Conversely, models designed with long context capacities demonstrate improved performance in streaming tasks. Although we faced memory constraints with these models, we posit that balancing input length with memory efficiency is essential for effective understanding of streaming video content.

###### 5.2. Analysis

Proactive Alerting We ablate M4 with two backbones, Qwen2 and Llama3.1. Aside from accuracy, the precision and Intersection-over-Union (IoU) for all responses are computed throughout the entire video stream. We evaluate its PA ability under various settings, with results presented in Tab. 4. Further evaluation of the mixed model on the general video understanding task, VideoMME [10], yields scores of 51.74 for M4-Qwen2 and 43.52 for M4-Llama3.1. Findings Our analysis reveals a significant performance gap between different LLMs. These results are generally consistent with the performance in Tab. 3. We conclude that leveraging a model with strong general ability can enhance proactive capabilities without necessitating the construction of new data, which could potentially compromise the model’s performance [3]. Moreover, interleaved data appear to improve the model’s grounding ability, aligning with

- Table 4. Ablation Study Results for M4 on the Proactive Alerting Task. “interleave”: the tuning data comprising interleaved text and images. “mix”: using M4-IT. “mean”: the attention weight is computed by averaging all tokens from the query.

Method Precision IoU Accuracy

M4-Qwen 31.60 13.90 25.00 M4-Qwen-interleave 32.65 14.65 26.00 M4-Qwen-mix 29.58 10.43 25.50 M4-Qwen-mix-mean 5.50 0.00 6.50

M4-Llama 8.38 2.47 10.50 M4-Llama-interleave 9.17 1.05 10.00 M4-Llama-mix 10.63 5.26 11.50 M4-Llama-mix-mean 0.50 0.00 0.50

- Table 5. Performance on general video understanding task from VideoMME [10].

Accuracy@Normal

- 61.5α
- 62.0

62.5

63.0

- 63.5

101.0

Accuracy@Noise

| | | |
|---|---|---|
| | | |
| | | |
| | | |

100.5

100.0

99.5

99.0

1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.5

Figure 6. Performance on the Proactive Turn-taking task for noise and normal query over different scaling factor.

Proactive Turn-taking To evaluate the efficacy of the instruction data, we applied the proactive turn-taking task to the M4, which was finetuned on M4-IT.

Findings The results in Fig. 6 indicate that, after tuning with M4-IT, our method successfully handled all legitimate queries. When subjected to noise input queries, the model demonstrated resilience across a broad range of the hyperparameter α. This suggests that our proposed instructional data effectively facilitates learning the format without compromising the model’s performance on standard queries.

Model Short Medium Long General

LongVA 61.1 48.3 45.4 52.4 LongVA (DataReplay) 60.9 50.7 45.0 52.2 M4 (Interleave) 60.3 50.6 43.9 51.6 M4 (Noise) 60.3 51.4 45.7 52.5 M4 (Stop) 60.3 60.8 44.0 51.7 M4 60.6 50.8 43.9 51.7

Influence on General Task To evaluate the effectiveness of our proposed M4 on general tasks, we further examine the model’s capability using the VideoMME benchmark for general video understanding. The results are listed in Tab. 5, where each configuration was trained on an identical dataset to ensure a fair comparison.

[Figure 36]

existing findings [17, 19]. Further investigation shows that the specific tokens in the query play a crucial role in achieving meaningful attention weights for grounding, where the tokens associated with the assistant role have demonstrated superior effectiveness. To effectively demonstrate the efficacy of our M4, we visualize the attention weights in Fig. 5, in which a strong correlation is presented between the elements within the relevant frames and the input query, validating the effectiveness of our approach.

[Figure 37]

Findings Our findings suggest that the introduction of an interleaved image-text instruction format and stop instructions has the most significant impact on the results, primarily due to the heterogeneous nature of the data format. When all these instructional data types are combined, there is a compromise in performance; however, it still surpasses the outcomes of the previously mentioned individual instruction types. This mixture achieves an effective balance, transitioning from general MLLM to interactive VideoLLMs without training on any video.

[Figure 38]

Please notify me when there is a pan.

high

[Figure 39]

##### 6. Conclusion

Please notify me when there is a knife.

In this work, we introduce OmniMMI, which evaluates the interactive capabilities of systems processing streaming video in open-world contexts. OmniMMI addresses challenges like streaming temporal state awareness and proactive reasoning with turn-taking. To advance real-time interactive reasoning, we propose a novel framework, M4, which enhances proactive turn-taking and efficient streaming capabilities. Our evaluations of previous MLLMs reveal significant limitations in handling multi-turn tasks and modality alignment. As such, we call for future research on efficient designs for open-world interactive OmniLLMs.

Q=textquery

Please notify me when there is a mixer.

low

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Acknowledgments The authors thank the reviewers for their insightful suggestions on improving the manuscript. This work presented herein is supported by the National Natural Science Foundation of China (62376031).

time

K = frames

- Figure 5. Attention feature map utilizes query as Q frames as K. The query consists of the last three tokens of the text query, while the key is represented by the mean-pooled frame.

##### References

- [1] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. arXiv preprint arXiv:2404.03413, 2024. 7
- [2] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024. 6
- [3] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 18407–18418, 2024. 3, 5, 7
- [4] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 7
- [5] Qirui Chen, Shangzhe Di, and Weidi Xie. Grounded multi-hop videoqa in long-form egocentric videos. ArXiv, abs/2408.14469, 2024. 3
- [6] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 7, 2
- [7] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024. 1
- [8] Alexandre D´efossez, Laurent Mazar´e, Manu Orsini, Am´elie Royer, Patrick P´erez, Herv´e J´egou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for realtime dialogue. arXiv preprint arXiv:2410.00037, 2024. 1
- [9] Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. Llama-omni: Seamless speech interaction with large language models. arXiv preprint arXiv:2409.06666, 2024. 1
- [10] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. ArXiv, abs/2405.21075, 2024. 1, 3, 7, 8
- [11] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, Ran He, Rongrong Ji, Yunsheng Wu, Caifeng Shan, and Xing Sun. Vita: Towards open-source interactive omni multimodal llm. ArXiv, abs/2408.05211, 2024. 3, 5, 7, 2
- [12] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of LLM inference using lookahead decoding. In International Conference on Machine Learning (ICML). OpenReview.net, 2024. 6

- [13] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 18995–19012, 2022. 3
- [14] Mingfei Han, Linjie Yang, Xiaojun Chang, and Heng Wang. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos. arXiv preprint arXiv:2312.10300, 2023. 3
- [15] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 13504– 13514, 2024. 3
- [16] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. TGIF-QA: toward spatio-temporal reasoning in visual question answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 3
- [17] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max W.F. Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multiimage instruction tuning. ArXiv, abs/2405.01483, 2024. 8
- [18] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34: 11846–11858, 2021. 3
- [19] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. ArXiv, abs/2407.07895, 2024. 8
- [20] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3, 7
- [21] Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, Song Chen, Xu Li, Da Pan, Shusen Zhang, Xin Wu, Zheng Liang, Jun Liu, Tao Zhang, Keer Lu, Yaqi Zhao, Yanjun Shen, Fan Yang, Kaicheng Yu, Tao Lin, Jianhua Xu, Zenan Zhou, and Weipeng Chen. Ocean-omni: To understand the world with omni-modality. arXiv preprint arXiv:2410.08565, 2024. 3
- [22] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Jian Yang, Siwei Wu, et al. Omnibench: Towards the future of universal omnilanguage models. arXiv preprint arXiv:2409.15272, 2024. 1, 3
- [23] Yifei Li, Junbo Niu, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, and Jiaqi Wang. Ovo-bench: How far is your video-llms from real-world online video understanding?, 2025. 3
- [24] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In

- European Conference on Computer Vision, pages 323–340. Springer, 2025. 7
- [25] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1, 7
- [26] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. CoRR, abs/2411.03628, 2024. 3
- [27] Ziyang Ma, Yakun Song, Chenpeng Du, Jian Cong, Zhuo Chen, Yuping Wang, Yuxuan Wang, and Xie Chen. Language model can listen while speaking. arXiv preprint arXiv:2408.02622, 2024. 3, 5
- [28] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 7
- [29] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 3
- [30] Vishvak Murahari, Carlos Jimenez, Runzhe Yang, and Karthik Narasimhan. Datamux: Data multiplexing for neural networks. Advances in Neural Information Processing Systems (NeurIPS), 35:17515–17527, 2022. 3, 5
- [31] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. ArXiv preprint, 2023. 3
- [32] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 5, 7, 2
- [33] Rui Qian, Xiao wen Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. ArXiv, abs/2405.16009, 2024. 3
- [34] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. arXiv preprint arXiv:2409.14485, 2024. 1, 3
- [35] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 1207–1216, 2019. 3
- [36] Gemini Team. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 7, 2
- [37] Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja Ganu, and Hao Wang. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks, 2024. 1
- [38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin

- Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 3
- [39] Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang, and Benyou Wang. Longllava: Scaling multi-modal llms to 1000 images efficiently via hybrid architecture. arXiv preprint arXiv:2409.02889, 2024. 7
- [40] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024. 3
- [41] Yuxuan Wang, Yueqian Wang, Dongyan Zhao, Cihang Xie, and Zilong Zheng. Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models. ArXiv, abs/2406.16338, 2024. 3
- [42] Yuxuan Wang, Cihang Xie, Yang Liu, and Zilong Zheng. Videollamb: Long-context video understanding with recurrent memory bridges. arXiv preprint arXiv:2409.01071,

2024. 1, 3, 7

- [43] Bo Wu and Shoubin Yu. Star: A benchmark for situated reasoning in real-world videos. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 3
- [44] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. ArXiv, abs/2407.15754,

2024. 3

- [45] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 3
- [46] Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024. 1, 3, 5
- [47] Zhifei Xie and Changqiao Wu. Mini-omni2: Towards opensource gpt-4o model with vision, speech and duplex. arXiv preprint arXiv:2410.11190, 2024. 1, 3, 5, 7, 2
- [48] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Association for Computing Machinery’s Annual Conference on Multimedia (ACM MM), 2017. 3
- [49] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 7
- [50] Wang Xu, Shuo Wang, Weilin Zhao, Xu Han, Yukun Yan, Yudi Zhang, Zhe Tao, Zhiyuan Liu, and Wanxiang Che. Enabling real-time conversations with minimal training costs. ArXiv, abs/2409.11727, 2024. 6
- [51] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 7
- [52] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for

- understanding complex web videos via question answering. In AAAI Conference on Artificial Intelligence (AAAI), 2019. 3
- [53] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 5579–5588, 2021. 1
- [54] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 3, 6, 7
- [55] Xinrong Zhang, Yingfa Chen, Shengding Hu, Xu Han, Zihang Xu, Yuanwei Xu, Weilin Zhao, Maosong Sun, and Zhiyuan Liu. Beyond the turn-based game: Enabling real-time conversations with duplex models. ArXiv, abs/2406.15718, 2024. 6
- [56] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 7
- [57] Weilin Zhao, Yuxiang Huang, Xu Han, Wang Xu, Chaojun Xiao, Xinrong Zhang, Yewei Fang, Kaihuo Zhang, Zhiyuan Liu, and Maosong Sun. Ouroboros: Generating longer drafts phrase by phrase for faster speculative decoding. In Annual Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024. 6
- [58] Yucheng Zhao, Chong Luo, Chuanxin Tang, Dongdong Chen, Noel Codella, and Zheng-Jun Zha. Streaming video model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14602– 14612, 2023. 3
- [59] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264,

2024. 1, 3

- [60] Xingyi Zhou, Anurag Arnab, Shyamal Buch, Shen Yan, Austin Myers, Xuehan Xiong, Arsha Nagrani, and Cordelia Schmid. Streaming dense video captioning. CoRR, abs/2404.01297, 2024. 3, 5

## OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts

### Supplementary Material

##### A. Audio Adaption Analysis

In this section, we further explore the adaptation of our methods to audio speech input. To adapt M4 to receive audio queries, we fine-tuned it on a randomly selected subset of the VoiceAssistant dataset [46], which comprises 30,000 audio instructions. To ensure a fair comparison, we maintained the same hyperparameters and other settings as those used in the tuning of M4. The results are presented in Table 6. Our findings indicate that tuning the model on purely audio instruction data, without incorporating visual data, does not enhance its proactive turn-taking ability. Consequently, we converted the queries in M4-IT to speech using CosyVoice and mixed them with the VoiceAssistant subset used during the tuning of M4-a. After integrating this audio data, we achieved a score of 68.5 on the PT task. Overall, the introduction of audio instruction data still limits the performance of tasks requiring both visual and audio inputs. We believe this limitation arises from the lack of mixed visual and audio data during the training phase. In future work, we aim to enhance the model’s audio understanding capabilities by incorporating more high-quality multimodal data.

##### B. Highlight Spot Algorithm

In this section, we present the pseudo-code of our proposed training-free highlight spot algorithm, as illustrated in Algorithm 1. For any transformer-based model, incoming streaming video frames are stored in the KVCache to avoid redundant computations. Subsequently, we compute the attention weights from the model’s final layer using the text query as the key and the video as the value. We then identify and save the frame indices whose attention weights exceed a threshold, determined by the mean and variance of the previous attention weights. These indices are labeled as consistently salient frames, signifying the frames that need to be highlighted. The consistency threshold is a hyperparameter, which is set to 4 in our experiments. Furthermore, we introduce an initial latency step to mitigate the challenges associated with calculating the mean and variance; in practice, this latency step is set to 2.

##### C. Single Question Analysis of Multi-turn Dependency Reasoning

In this section, we detail the accuracy of each step in the multi-turn dependency reasoning task. The results are presented in Table 7. Unlike the results presented in Table 3,

Algorithm 1 Highlight Spot Require: Video stream V∞, query q, threshold γ, Gaussian

factor α

- 1: highlight spot.init()

- 2: for all frame v in V∞ do
- 3: KVCache.update(WKv, WV v)
- 4: attn ← SelfAttn(v ⊕ q,KVCache)
- 5: (µ,σ) ← std mean(attn)

- 6: δ ← µ + α × σ
- 7: cands ← {t;attn[t] > δ}
- 8: for all ct in cands do
- 9: ct ← highlight spot.get(t) + 1

- 10: highlight spot.update(i,ci)

- 11: end for
- 12: if highlight spot.heap is not empty then

- 13: (i,c) ← highlight spot.peek()

- 14: if freq > γ then
- 15: send(i)
- 16: end if
- 17: end if
- 18: end for

this experiment focuses solely on the accuracy of individual reasoning steps. Generally, we observe a decline in accuracy as the number of steps increases. However, in certain instances, accuracy at a later step exceeds that of a previous one. We attribute this anomaly to potential hallucinations generated by the language models. Overall, there is a significant drop in accuracy across successive steps, underscoring the importance of multi-step reasoning in evaluation. This approach helps to mitigate errors introduced by language models, demonstrating the necessity of a step-by-step evaluation process.

##### D. Single Question Analysis of Dynamic State Grounding

In this section, we extend our analysis of the Dynamic State Grounding task by examining the performance on each individual question. The results, as detailed in Table 8, indicate a notable decline in performance as the number of states increases. This decline can be attributed to the increased length of the video context and dialogue history, which complicates the process of dynamically grounding the current state to derive the correct answer. Furthermore, our analysis did not reveal significant performance differences across different models at the initial state. However,

- Table 6. Performance comparison of existing OmniLLM on OmniMMI. The 1st, 2nd, 3rd of SG and MD tasks represent the cumulative accuracy up to and including these stages. The “avg.” indicates average accuracy across all data points.

Num Frames

SG

MD

Models LLM

SI PA PT 1st 2nd 3rd avg. 1st 2nd 3rd avg.

AP

Commercial Video LLMs

Gemini-1.5-Pro [36] - 128 52.33 19.67 9.35 16.33 43.00 35.00 16.26 7.14 12.00 38.50 ✗ ✗ GPT-4o [32] - 50 48.67 16.95 5.61 15.00 39.50 34.33 15.57 7.65 12.33 17.00 ✗ ✗

OmniLLMs VideoLLaMA2 [6] Qwen2-7B 8 41.00 12.88 0.00 10.33 35.00 23.33 4.15 0.51 3.00 5.00 ✗ ✗ VITA [11] Mistrl-8×7B 16 8.67 0.00 0.00 0.00 39.00 11.33 3.11 1.52 2.00 1.50 ✗ 67.00 MiniOmini2 [47] Qwen2-0.5B 1 17.00 5.08 0.93 4.67 14.00 6.00 1.00 0.00 1.00 1.00 ✗ ✗

M4 (ours) Qwen2-7B 32 / 1 fps 35.67 6.44 1.87 5.67 33.5 35.67 6.44 1.87 1.67 9.00 25.50 62.00 M4-a(ours) Qwen2-7B 32 / 1 fps 28.33 2.37 0.00 2.00 13.00 19.33 3.11 0.51 3.00 7.50 1.50 68.5

Table 7. Multi-turn Dependency Reasoning

Table 8. Dynamic State Grounding

Models Step=1 Step=2 Step=3 Overall Commercial Video LLMs

Gemini-1.5-Pro 52.33 34.24 36.45 16.33 GPT-4o 48.67 31.53 20.56 15.00

Open-source Video LLMs

VideoChatGPT 18.00 13.49 11.22 3.00 VideoChat2 16.33 13.15 12.24 2.67 Video-LLaVA 22.67 13.49 16.33 3.33 LLaMA-VID 21.33 15.22 13.78 2.67 MiniGPT4-Video 12.67 6.57 8.67 1.67 PLLaVA 21.00 13.49 17.35 1.33 LLaVA-NeXT-Video 17.00 10.03 10.71 2.00 ShareGPT4Video 20.33 15.57 14.80 2.00 LLaMA-VID-13B 22.67 14.88 14.29 3.33 PLLaVA-13B 25.67 17.80 16.84 4.33 PLLaVA-34B 18.67 17.30 10.20 3.00 LLaVA-NeXT-Video-DPO-34B 14.67 14.53 12.24 1.67

LongVA 20.67 16.27 13.78 2.33 LongVILA 22.33 14.19 14.29 3.00 LongLLaVA 26.33 18.69 20.41 3.67 VideoLLM-online 11.67 7.27 10.71 1.33 VideoLLaMB 18.67 13.15 17.86 3.00

OmniLLMs

VideoLLaMA2 23.33 15.92 18.78 5.00 VITA 11.33 12.80 8.63 2.00 MiniOmini2 6.00 3.11 2.03 1.00

M4 19.33 10.73 12.18 1.67

Models State=1 State=2 State=3 Overall Commercial Video LLMs

Gemini-1.5-Pro 35.00 37.02 38.78 12.00 GPT-4o 34.33 33.56 37.24 12.33

Open-source Video LLMs

VideoChatGPT 35.33 17.97 10.28 3.33 VideoChat2 19.67 14.23 6.54 2.33 Video-LLaVA 32.00 16.27 11.21 1.67 LLaMA-VID 29.67 13.56 7.48 2.33 MiniGPT4-Video 25.00 15.25 14.02 4.00 PLLaVA 37.33 13.56 10.29 3.33 LLaVA-NeXT-Video 30.33 12.20 6.54 3.00 ShareGPT4Video 34.00 13.22 10.28 2.00 LLaMA-VID-13B 33.33 14.24 6.54 1.33 PLLaVA-13B 41.33 13.90 12.15 2.67 PLLaVA-34B 29.00 14.24 10.28 3.67 LLaVA-NeXT-Video-DPO-34B 30.33 11.19 5.61 2.67

LongVA 33.33 15.59 8.41 3.33 LongVILA 39.00 16.95 14.02 4.33 LongLLaVA 36.33 11.53 7.48 3.33 VideoLLM-online 18.00 13.56 5.61 4.67 VideoLLaMB 32.67 14.58 10.28 2.33

Open-source Video LLMs VideoLLaMA2 41.00 26.78 10.28 10.33 VITA 8.67 8.14 2.80 0.00 MiniOmini2 17.00 14.92 10.28 4.67

M4 35.67 13.22 6.54 5.67

the performance gap widens as the number of states increases, underscoring the importance of a model’s ability to handle longer contexts while maintaining effective grounding capabilities.

##### E. Annotation Details

###### E.1. Raw Video Data Collection

To enhance our dataset, we specifically collect data from YouTube, concentrating primarily on videos that are particularly commonly useful in our real-life. We also focus on the videos which are in content involving personal introduc-

tions and interpersonal interactions.

###### E.2. Annotation Tool

The front-end interface for human annotation is depicted in Figure 7. In this interface, each question or statement is associated with the most relevant time span, which serves either as part of the label or as an aid for subsequent annotation tasks.

###### E.3. Annotation Guidelines

To ensure that annotators produce high-quality annotations that align with our specified standards, we provide detailed guidelines, including examples of various question types.

[Figure 51]

Figure 7. The Front-End Interface for Human Annotation

Category Question

Object State How many objects are in the scene?

How many people are in the room? What is the color of the car? Is the door open or closed?

Spatial Relations Where is the cat relative to the chair? Dynamic Spatial Relations Is the person walking towards or away from the camera?

Where is the ball relative to the player? Action State What is the person doing?

What activity is happening in the scene? Scene State Is the room well-lit or dim?

What is the weather like? Is the street busy or quiet? What is the context of the scene? Human Object Interaction Is the person holding the book? Human Human Interaction Are the two people shaking hands?

What is the interaction between the two characters? Group Dynamics How are the group members interacting?

Emotional State What is the person’s emotional state? Audio/Speech State What does the speaker mentioned?

Table 9. Annotation hints for annotators including category and example question.

The list of hints is demonstrated in Table 9.

##### F. M4-IT Construction Details

###### F.1. Noise data prompt

We employ GPT-4o to autonomously generate noise data for the purpose of instruction tuning. The prompt utilized for the generation of noise data is detailed below.

You are a sophisticated AI designed to simulate human-like conversation by generating ’noise.’ This noise consists of naturally flowing statements that mimic the user’s perspective. —— Review the user’s questions and the assistant’s responses carefully. Using this information, create coherent declarative statements that reflect the user’s voice. These should resemble everyday human dialogue and do not require a response from the assistant. Ensure your output is in the form of declarative sentences and avoid questions. Keep the noise brief and in casual, conversational English. But do NOT need response

###### F.2. Stop Words

We compile a set of frequently used stop words to incorporate into our instructional data, thereby serving as the designated stop words: “That’s a good point, and”, “Let me stop you there”, “Just a second”, “I don’t mean to be rude, but”, “If I could interject”, “Pardon me, but”, “Sorry to interrupt”, “Before you continue”, “Can we pause for a moment?”, “May I add something here?”, “I apologize for cutting in”, “Could I stop you for a second?”, “I’d like to add”, “Could I clarify something?”, “I have a quick question”, “This reminds me of”, “Let me add to that”, “Can I share my thoughts?”, “Hold on a moment”, “One moment, please”, “Allow me to explain”, “Excuse me”, “Can I jump in for a moment?”, “I see what you mean, but”, “I think it’s important to mention”

##### G. M4 Implementation Details

###### Hyperparam M4

α 2 β 0.2 γ 4 Model Max Length 32000 Learning Rate 1e-5 Warmup Ratio 0.03 Per Device Batch Size 1 Gradient Accumulation Steps 4 Epoch 1

Table 10. Hyperparameters for M4.

In practice, we conduct the training process using four Nvidia A800 GPUs, which requires approximately one hour to fine-tune the model. Table 10 presents a detailed account of the hyperparameters employed during both the training and inference procedures.

