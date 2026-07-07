## Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO

# arXiv:2511.16669v2[cs.CV]23Nov2025

Junhao Cheng1† Liang Hou2 Xin Tao2 Jing Liao1

1 City University of Hong Kong 2 Kling Team, Kuaishou Technology

https://video-as-answer.github.io/

How to … ? What will … ?

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Assuming she's in an exaggerated movie, what will she look like the moment after?

How to proceed with making my windmill?

[Think] ofthepaperTheuserpinwheel.has alreadyPinwheelscompletedusuallyneedtheamainhandle topart VA hold—often made from a stick…so the next step should be attaching the stick to the back of the pinwheel. [/Think][Ans]One hand secure a stick…[/Ans]

[Think] The input video shows a girl getting scalded. A common trope in exaggerated movie to show a character

##### VA

being overheated is having them emit steam from their

mouth[/Think][Ans] [Ans]The girl lets out a sharp gasp, a wisp of steam escaping her lips…[/Ans]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Hi, how to move on to the next step.

What will she most likely do next?

[Think]wrappingFromSamosa.theinputAccordingvideo,Icanto standardtellthat theprocedural,userare VA the next step is to pinch the edges of the wrapper together to seal it completely, and forming a triangular shape.[/Think][Ans]Pinch the…[/Ans]

[Think]The video shows a woman hammering a nail into her bedroom wall, she most likely to presumably hang a painting and complete the room's décor. [/Think][Ans]A woman takes carefully…[/Ans]

##### VA

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Figure 1. Given an input video and a procedural (e.g. “how to do it?”) or predictive question (e.g. “what will happen next?”), our proposed VANS is capable of predicting and generating the next video event as an intuitive and customized answer.

### Abstract

While language models have become impactful in many real-

†This work was conducted during the author’s internship at Kling Team, Kuaishou Technology.

world applications, video generation remains largely confined to entertainment. Motivated by video’s inherent capacity to demonstrate physical-world information that is difficult to convey through language alone (e.g., imagine teaching someone to tie a tie using only text), we identify an underuti-

lized opportunity to extend video as a new answer modality for Next-Event Prediction (NEP), formalized as Video-NextEvent Prediction (VNEP). While the established NEP task takes a video with a procedural or predictive question as input to predict the next event in text, VNEP requires dynamic video responses. This shift from telling to showing unlocks more intuitive and customized answers for procedural learning and creative exploration. However, this task remains challenging for existing models, as it demands an understanding of multimodal input, instruction-conditioned reasoning, and the generation of video with visual and semantic consistency. To address this, we introduce VANS, a model that leverages reinforcement learning to align a Vision-Language Model (VLM) with a Video Diffusion Model (VDM) for VNEP. The core of VANS is our proposed JointGRPO that orchestrates the VLM and VDM to function as a unit. Driven by a shared reward on their respective output, it optimizes the VLM to produce captions that are both accurate and friendly to visualize, while guiding the VDM to generate videos that are faithful to these captions and the input visual context. To enable this learning, we craft VANS-Data100K, a dedicated dataset for the VNEP task. Experiments on procedural and predictive benchmarks demonstrate that VANS achieves state-of-the-art performance in both video event prediction and visualization. Codes are released in https://github.com/KlingTeam/VANS.

### 1. Introduction

While generative AI [21, 36, 38, 48, 50] has revolutionized text-based tasks in real-life domains like healthcare [13] and education [10], video generation models remain largely confined to entertainment [7, 52]. This is a missed opportunity, as video encapsulates rich, dynamic information about the physical world that text alone struggles to convey [28, 49].

Motivated by this observation, we pioneer Video-NextEvent Prediction (VNEP), a novel paradigm that uses video as the answer modality for Next-Event Prediction (NEP). While the established NEP task takes a video with a procedural or predictive question as input to predict the next event in text [9, 16, 31, 43], VNEP instead generates dynamic video responses. The shift from telling to showing enables VNEP to offer more intuitive and customized answers by leveraging video’s ability to convey spatial layout, motion, and temporal ordering, while adapting the demonstration to the user’s current state. As shown in Figure 2, the video answer guides the user through the remaining steps of the Windsor knot based on his current tie configuration (e.g., color, orientation and tightness), providing clarity and personalization that a generic text-only description cannot achieve.

However, VNEP introduces significant challenges that go beyond mere visual continuation. Unlike existing video extension task [55] which predicts frames based on spatiotem-

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Help! I'm stuck on the next step of my Windsor knot.

[Figure 33]

Hold the wide end, then pass it from the inside to the outside

through the intersection point below to form a loop…

[Figure 34]

VA

[Think]… Pass the wide end from the loop inside out like this:

[Figure 35]

[Figure 36]

[Figure 37]

Figure 2. Video answer (our VANS) versus text-only answer (Gemini) on a procedural question. Video answer provides an intuitive and customized response by demonstrating the action directly, while text-only answer falls short in clarity.

poral patterns (e.g., forecasting a ball’s trajectory), VNEP focus on event-conditioned reasoning. It requires a model to first comprehend the video and question, reason about the subsequent event from causal or procedural logic (e.g., inferring that adding soap is needed after observing dirty dishes being scrubbed with water), and then generate a video that is both visually coherent and semantically faithful to this inferred event. A straightforward solution is to employ a VLM for prediction followed by a VDM for generation. However, this cascaded pipeline suffers from a semanticto-visual misalignment; the VLM’s textual output may be linguistically correct but visually unrealistic or unexecutable by the VDM, leading to semantically and visually divergent videos [44]. In contrast, unified models [23, 34, 42] attempt to align understanding and generation within a single model but face a capability trade-off, often excelling in one at the expense of the other and struggling to achieve optimal performance in both simultaneously [45]. Consequently, neither paradigm alone offers a satisfactory solution.

However, the limitations of these two paradigms point toward a more promising direction: a tight integration of specialized models that preserves their strengths while resolving their interoperability issues. To this end, we propose VANS, a model that employs reinforcement learning (RL) post-training as an effective alignment process to fully realize the complementary strengths of VLMs (in semantic reasoning) and VDMs (in visual synthesis), enabling them to operate in concert for VNEP. Central to our approach is the Joint-GRPO RL strategy, which orchestrates both models using a joint reward derived from their respective outputs. Through a two-stage optimization, Joint-GRPO trains the VLM to produce captions that are both accurate and friendly

Procedural Data

Predictive Data

1 1

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

Data Collection (From Internet

Data Collection (From NEP Dataset)

& NEP Dataset)

- 2

- 3

- 4

[Figure 48]

add scallions to

add cheese to the

[Figure 49]

Shot-Boundary Detection Model

mix by hand

[Figure 50]

the meat

meat

Split Shot by Model 00:00 00:02 00:02 00:09

00:09 00:14 00:14 00:20

- 2

- 3

- 4

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

00:00 00:01 00:01 00:15 00:15 00:25 00:30

[Figure 56]

[Figure 57]

Split Shot & Crop

[Figure 58]

[Figure 59]

Too short

Too short

[Figure 60]

[Figure 61]

[Figure 62]

Task: Select clip that best

Video Caption Time Step

Video Time Step

[Figure 63]

Caption Generation & Clip Selection

[Figure 64]

represents the video with

Vid Cap TS Task

[Figure 65]

[Figure 66]

detailed caption

Task: Select clip that best match the caption

[Figure 67]

Clip Selection

[Figure 68]

00:02 00:07

00:09 00:14

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

00:16 00:20

[Figure 73]

[Figure 74]

Video 1 Video 2

[Figure 75]

Video 3

[Figure 76]

[Figure 77]

00:15 00:20

00:09 00:14

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Empty room…

Next to a table…

A woman lifts a cell phone…

Video 1 Video 2

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Video 1 Caption 1

Video 1 Caption 1 Video 2 Caption 2

Video 2 Caption 2 Video 3 Caption 3

Video 2 Caption 2

Task: Simulate user instruction, model thinking and ground truth dense

Task: First evaluate logical and visual consistency, then

descriptions for the output video

simulate the user instruction, model reasoning, and ground truth, finally performing a self-check

Task

[Figure 86]

[Figure 87]

Instruction: I'm making meat patties. How to finish the next step? Thinking: From the input video, I can see that the user has added cheese to the minced meat. Logically, the next step should be to mix them together thoroughly. I'll show a person using their hands to combine the cheese and meat evenly. Description: A cook in a black apron and white sleeves stands at a kitchen counter. He uses his bare hand to thoroughly mix and knead the contents,

QA Pair Generation

QA Pair Generation

[Figure 88]

Instruction: what will appear on the phone?

[Figure 89]

Thinking: The video shows a girl raising her

[Figure 90]

[Figure 91]

[Figure 92]

phone with a smile… This suggests she is taking a selfie, her own face would appear on the screen. Check: information leaking ×logically sound √

The two videos lack

[Figure 93]

logical and visual

[Figure 94]

squeezing the meat and cheese together until they form a uniform mixture.

consistency.

- Figure 3. Data curation pipeline of VANS-Data-100K, which processes raw videos through shot splitting, clip selection, and QA generation to produce high-quality data for both procedural and predictive Video-Next-Event Prediction.

to visualize, while guiding the VDM to generate videos that are faithful to these captions and the input visual context.

To enable this learning, we construct VANS-Data-100K, a dedicated dataset with 100K video-question-answer triplets for supervised fine-tuning (SFT) on VNEP task. From this collection, we manually select 1K high-quality samples to support the subsequent RL post-training.

Experimental results on procedural and predictive VNEP benchmarks demonstrate that VANS performs favorably against state-of-the-art (SOTA) approaches in both event prediction accuracy and the quality of the generated videos.

We make the following contributions in this work:

- • We pioneer VNEP, advancing next-event reasoning from textual description to dynamic video demonstration.
- • We propose VANS and its core Joint-GRPO strategy, which aligns a VLM and a VDM through RL with a joint reward, yielding video answers that are both semantically faithful and visually coherent.
- • We construct VANS-Data-100K, a dataset of 100K (input video, question, output video) triplets for training and evaluating models on VNEP.

### 2. Related Work

Next-Event Prediction. The established NEP task requires predicting a future event given a video and a procedural or predictive question [9, 16, 32, 39]. Existing efforts predominantly address this as a textual NEP problem, focusing on generating descriptive answers. This line of work is supported by benchmarks like VLEP [17], MVP [33] and V1-33K [39], and leverages techniques ranging from event

understanding [3] to multiscale temporal modeling and commonsense reasoning [4]. The recent rise of VLMs has further propelled this field, with methods that fine-tune on largescale data [9] or utilize RL to elicit NEP capabilities from pre-trained models [39]. However, a fundamental limitation persists: the answer modality remains exclusively textual. Our work breaks from this paradigm by introducing VNEP, advancing next-event reasoning from textual description to dynamic video demonstration.

Group Relative Policy Optimization. Group Relative Policy Optimization (GRPO) was initially introduced by DeepSeek-Math [29] to enhance the reasoning capabilities of language models and align their outputs with human preferences. Its effectiveness has led to its adoption in video understanding and reasoning [5, 6, 11, 12, 18, 41], where it improves model performance on complex, open-ended queries. Beyond understanding, GRPO has also been applied to image and video generation [15, 25, 44, 46, 47]. In video generation, its primary role has been to enhance the alignment between the generated video and the text prompt [22, 46], as well as to improve consistency in reference-based generation tasks [30]. While these works apply GRPO to optimize a single model, our Joint-GRPO coordinates two models (a VLM for reasoning and a VDM for visualization) simultaneously, ensuring they are jointly aligned for the VNEP task.

### 3. VANS-Data-100K

Existing NEP datasets are unsuitable for direct use in VNEP due to suboptimal video quality and a lack of diverse instructional questions. To bridge this gap, we construct the

VANS-Data-100K dataset, comprising 30K procedural and 70K predictive samples. Each sample contains an input video, a question, and a multi-modal answer (text and video), tailored for the VNEP task. As illustrated in Figure 3, our curation pipeline involves four stages.

Raw Data Collection. We collect data from two distinct sources to cover both procedural and predictive scenarios. For procedural data, we source high-resolution videos from COIN [51] and YouCook2 [54] to ensure clear visual demonstrations of step-by-step tasks. For predictive data, we gather videos from general-scene datasets [2, 39] and shortfilms [8], which are rich in narrative and causal dynamics.

Shot Split. Raw videos are segmented into coherent clips. Procedural videos are segmented using ground-truth timestamps, while predictive videos employ a shot-boundary detection model. We filter out segments shorter than 3 seconds to ensure action completeness.

Clip Selection. We employ Gemini-2.5-Flash [35] as an automated quality filter to identify the optimal 3-5 second clip. For procedural data, it selects the clip that best aligns with the given caption. For predictive data, it first generates a detailed caption for each segment, ensuring the selected clip is both high-quality and semantically representative.

QA Pair Generation. Using Gemini-2.5-Flash, we generate QA pairs from video-caption sequences. The VLM simulates diverse questions—focusing on the logical next step for procedural tasks and hypothetical “what-if” scenarios for predictive ones. It also produces a chain-of-thought reasoning and ground-truth answer for each question, with a self-check to ensure logical soundness and prevent information leakage.

Please refer to Appendix A for more dataset details.

### 4. VANS

- Figure 4 introduces the overall architecture of VANS. The input question is tokenized and fed into the VLM alongside high-level ViT visual features from the input video. We task the VLM with performing instruction-grounded reasoning to generate a textual caption describing the predicted next event, which serves as the semantic guide for the VDM. To ensure visual consistency, the VDM is conditioned on both the generated caption and low-level visual cues, which are extracted by tokenizing n sampled input frames using a VAE [38]; these tokens are then concatenated into the VDM’s conditioning latent space. This enables fine-grained visual correspondence while generating novel scenes.

This design faces a fundamental limitation: the VLM and VDM are optimized in isolation. The VLM is trained for textual accuracy but receives no feedback on whether its descriptions lead to visually plausible videos. Conversely, the VDM faces the challenge of coordinating two conditioning signals: the VLM’s specific caption and the input’s visual

[Figure 95]

[Figure 96]

[Figure 97]

VAE Token Caption

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Thinking

[Figure 102]

Video Diffusion Model

[Figure 103]

[Figure 104]

[Figure 105]

JointGRPO

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Vision-Language Model

[Figure 117]

[Figure 118]

[Figure 119]

ViT

VAE

Tokenizer

Encoder

Encoder

[Figure 120]

[Figure 121]

[Figure 122]

“Hi, I'd like to know how to move on to the

next stage”

Figure 4. Overall architecture of VANS.

context. While SFT equips the VDM with basic capabilities, achieving consistent performance on both semantic accuracy and visual fidelity requires further refinement. This disconnect creates a semantic-to-visual gap where both models operate without awareness of each other’s constraints and capabilities. To resolve this, we introduce Joint-GRPO to orchestrate the two models into a cohesive unit for VNEP.

#### 4.1. Preliminary of GRPO

GRPO is an RL algorithm designed to align model outputs with human preferences or complex objectives. The core concept involves using a reward function to evaluate the quality of generated samples, and then adjusting the model’s policy to increase the likelihood of high-reward generations. For each input context c, the policy model πθ generates a group of G trajectories {oi}Gi=1. Each trajectory receives a reward ri reflecting its quality. GRPO computes a normalized advantage A˜i that measures how much better or worse each trajectory is compared to the group average:

G

G

ri − r¯ σr

1 G

1 G

A˜i =

(rj − r¯)2.

, r¯ =

rj, σr =

j=1

j=1

(1) The policy is then optimized using the GRPO objective:

Ti−1

G

1 Ti

1 G

min rti(θ)A˜i,

J(θ) = E

t=0

i=1

clip(rti(θ),1 − ϵ,1 + ϵ)A˜i − βDKL(πθ∥πref) , (2)

Stage 1 Stage 2

Video + Question

Video + Question

[Figure 123]

[Figure 124]

Policy Model (VLM)

Anchor Model (VLM)

Stage 1

Stage 2

s1 s2 s3 …

s

sn

Video + Question

Video + Caption

| | | | | |
|---|---|---|---|---|
| |Rewar|d Mod|el (VDM)<br><br>[Figure 125]| |
| | | | | |

[Figure 126]

[Figure 127]

[Figure 128]

Policy Model (VLM)

Policy Model (VDM)

Policy Model (VDM)

…

…

…

…

s1 s2 s3

sn

v1 v2 v3 vn v1 v2 v3 vn v1 v2 v3 vn

Video Reward Function

Joint Reward Function 1

Joint Reward Function 2

Text Reward Function

r1 r2 r3 … rn r1 r2 r3 … rn r1 r2 r3 … rn

…

r1 r2 r3 rn

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Compute Advantage Compute Advantage Compute Advantage Compute Advantage

(a) GRPO (b) Joint-GRPO

- Figure 5. Comparison of standard GRPO with Joint-GRPO. While standard GRPO optimizes a single model at a time, our Joint-GRPO coordinates their optimization under a joint reward function.

i t|c)

where rti(θ) = πθ(o

πθold(oit|c) is the probability ratio for the i-th trajectory. The clipping mechanism and KL divergence term ensure training stability by preventing drastic policy updates.

#### 4.2. Joint-GRPO

Standard GRPO, while effective for single-model alignment, faces a fundamental limitation in multi-model scenarios like VNEP: it optimizes models in isolation. Applying it separately to the VLM and VDM fails to bridge the semanticto-visual gap, as it does not encourage their outputs to be mutually reinforcing. Conversely, a one-stage joint training of both models is also problematic. This approach is prone to reward hacking and training instability, since when a generated video is of poor quality, it is ambiguous whether the VLM’s caption or the VDM’s generation is at fault, leading to conflicting gradient signals.

To address this attribution problem and enable effective co-steering, we propose Joint-GRPO. This approach coordinates the VLM and VDM using a joint reward function via a structured two-stage optimization process. Our key insight is that the two models must be co-steered such that the VLM’s reasoning becomes visually grounded to guide the VDM effectively, while the VDM’s generation remains faithful to the VLM’s prediction and visual context.

Stage 1: Visualization-Friendly VLM Tuning. We first align the VLM’s reasoning with the VDM’s generation results. We optimize the VLM policy πVLM while keeping the VDM frozen. For an input video vin and question Q, we sam-

ple G textual captions {si}Gi=1 from πVLM. Each caption si is then used by the frozen VDM to generate a corresponding

video vouti . The joint reward r1 for the VLM is computed as: r1(si,vouti ) = λfrf(si)

###### +λv1rv1(vouti ,vgt)

###### +λt1rt1(si,sgt)

###### ,

format

text fidelity

video fidelity

where λf,λt1,λv1 are weighting coefficients for each reward, which are defined as follows:

- • rf(si) ensures the output follows the specified instruction format. A reward of 1 is given if the response adheres to the “reason-then-answer” template, and 0 otherwise.
- • rt1(si,sgt) measures semantic similarity between generated and ground-truth captions using ROUGE-L [19].
- • rv1(vouti ,vgt) evaluates visual coherence of generated videos with ground-truth using CLIP Similarity [27].

This composite reward is designed to steer the VLM beyond mere linguistic correctness. Relying solely on rt1 can lead to captions that are linguistically correct but visually unrealistic or unexecutable by the VDM. Conversely, using only rv1 provides a reward that is too distal and ambiguous to effectively guide the VLM’s reasoning process. The joint reward guides the VLM to generate captions that are not merely semantically accurate, but also visually plausible and actionable for the VDM. This process effectively forces the VLM to internalize the VDM’s capabilities and constraints. Stage 2: Context-Faithful VDM Adaptation. Building upon the visually-grounded captions from Stage 1, this stage tackles the challenge of cross-modal alignment by adapting the VDM to render these captions faithfully while preserving visual consistency with the input visual context. We optimize the VDM policy πVDM using the VLM as a frozen anchor model. As shown in Figure 5, the ‘now-improved’ VLM from Stage 1 generates a candidate anchor caption (samples with low semantic similarity to the ground truth

Table 1. Quantitative comparison with baseline models on Video-Next-Event Prediction.

Model BELU@1↑ BELU@2↑ BELU@3↑ BELU@4↑ ROUGE-L↑ FVD↓ CLIP-V↑ CLIP-T↑ Procedural Benchmarks

Video-GPT - - - - - 105.32 0.7334 0.1997 Omni-Video 0.0948 0.0253 0.0040 0.0008 0.1075 236.38 0.6293 0.2323 Qwen-Wan 0.0981 0.0260 0.0046 0.0013 0.1530 148.75 0.6619 0.2448 TEMPURA-Wan 0.1984 0.1063 0.0336 0.0167 0.1915 143.80 0.6738 0.2498 Gemini-Wan 0.2432 0.1077 0.0448 0.0215 0.2802 120.34 0.6898 0.2547 Qwen-FilmWeaver 0.0981 0.0260 0.0046 0.0013 0.1530 129.44 0.6831 0.2532 TEMPURA-FilmWeaver 0.1984 0.1063 0.0336 0.0167 0.1915 120.34 0.6923 0.2562 Gemini-FilmWeaver 0.2432 0.1077 0.0448 0.0215 0.2802 110.54 0.7102 0.2773 VANS (SFT) 0.2524 0.1162 0.0501 0.0233 0.2812 85.34 0.7655 0.3202 VANS (Joint-GRPO) 0.3257 0.1834 0.1242 0.0987 0.3631 78.32 0.8021 0.3824

Predictive Benchmarks

Video-GPT - - - - - 170.32 0.7031 0.2124 Omni-Video 0.0885 0.0232 0.0035 0.0006 0.1012 252.47 0.6083 0.2218 Qwen-Wan 0.0927 0.0241 0.0040 0.0010 0.1453 158.92 0.6427 0.2349 TEMPURA-Wan 0.1639 0.0647 0.0132 0.0105 0.2142 152.86 0.6524 0.2398 Gemini-Wan 0.1981 0.0760 0.0182 0.0112 0.2298 128.65 0.6673 0.2446 Qwen-FilmWeaver 0.0927 0.0241 0.0040 0.0010 0.1453 137.84 0.6608 0.2431 TEMPURA-FilmWeaver 0.1839 0.0647 0.0132 0.0105 0.2142 128.32 0.6709 0.2463 Gemini-FilmWeaver 0.1981 0.0760 0.0182 0.0112 0.2298 118.27 0.6874 0.2663 VANS (SFT) 0.2247 0.0873 0.0206 0.0136 0.2435 94.12 0.7512 0.3038 VANS (Joint-GRPO) 0.2789 0.1351 0.0853 0.0694 0.3058 86.85 0.7872 0.3759

are discarded and regenerated to ensure quality). The resulting semantically-grounded caption sanchor is then used to condition the VDM. We then sample G output videos {vouti }Gi=1 from πVDM. The VDM’s core task is to generate a novel scene by dynamically attending to and preserving relevant visual elements (e.g., IDs, backgrounds) from the input video’s VAE tokens, as guided by the semantic content of sanchor. The reward function r2 is defined as:

r2(vouti ,sanchor) = λv2rv2(vouti ,vgt)

###### +λc2rc2(vouti ,sanchor)

###### ,

semantic alignment

video fidelity

where λv2,λc2 are balancing coefficients, and:

- • rv2(vouti ,vgt) maintains visual quality and coherence with the input video, using the same metric as in Stage 1.
- • rc2(vouti ,sanchor) measures semantic consistency between the output video and the anchor caption using CLIPScore.

This joint-reward design tackles the core challenge of cross-modal alignment. rv2 ensures the output remains visually plausible and continuous. rc2 compels the VDM to strictly adhere to the event described in sanchor, preventing it from ignoring the caption and merely reconstructing or slightly altering the input video.

Through this two-stage optimization, the VLM and VDM co-evolve into a synergistic unit. The distinct, complementary roles of each reward component, along with the training reward curves, are presented in Appendix B.

### 5. Experiments

We conduct experiments to evaluate the effectiveness of our VANS and to compare it with cutting-edge solutions.

#### 5.1. Settings

Benchmarks. We construct our evaluation benchmark by sampling 400 procedural and 400 predictive samples from our dataset with source videos from established benchmarks [2, 8, 51, 54] to ensure reliable ground-truth text & video answers. The evaluation set is strictly separated from the training data, with no video or question overlap.

Metrics. Following [16], we employ BELU@1/2/3/4 [26] and ROUGE-L [20] for textual prediction quality. For videos, we use Fr´echet Video Distance (FVD) [37], CLIP-Video Score (CLIP-V)↑ [27], and CLIP-Text Score (CLIP-T) [27] to assess visual quality and semantic alignment.

Baselines. As no existing methods are designed for VNEP, we adapt top-performing models from related fields to establish baselines. These include: (1) Video extension model Video-GPT [55]; (2) Cascaded pipelines formed by combining top-tire VLMs (Gemini-2.5-Flash [35], Qwen-2.5VL-3B [1], and its NEP-finetuned version TEMPURA [9]) with VDMs (Wan-2.1-1.3B [38], FilmWeaver [24]); and (3) Unified model Omni-Video [34].

Implementation Details. We initialize VANS with Qwen2.5-VL-3B as the VLM and Wan-2.1-1.3B as the VDM. For Video-GPT, we provide the input video and utilize its native capability for video continuation. For VANS and other baseline methods, we supply the input video and the corresponding question to perform NVEP. Implementation details of our VANS and baselines are presented in Appendix C.

Case 1 (Predictive Question) Case 2 (Procedural Question)

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

What will the sitting man do if he is irritable?

Show me the next step for baked chicken Parmesan.

The input video shows two fried cutlets in a pan being drizzled with red sauce. The most logical next step would be to move the cutlets from the pan to a plate and add the sauce.

The input video shows a man in white slapping a man in black (the sitting one). Based on the assumption, the man in black would likely retaliate by hitting back.

Omni-Video

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

| |
|---|

| |
|---|

The input video shows a man in white scolding a man in black. Given their potential superior-subordinate relationship, it might be assumed that the man in black would talk back.

From the video, you have already pan-fried the breaded chicken and you are now drizzling ketchup over it. The next step is to bake the chicken in the oven with cheese.

Gemini-FilmWeaver

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

| |
|---|

[Think]You‘ve pan-fried the breaded chicken until golden and are now drizzling it with ketchup...[/Think][Ans]A spoon adding a layer of cheese directly over the ketchup…

[Think]The video shows the man in black being interviewed. It is assumed that he would likely grow impatient with the

Ours(SFT)

process.[/Think][Ans]A man shouting to a camera…

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

| |
|---|

| |
|---|

[Think]A man in white (superior) verbally abusing a seated man (subordinate). Given the hierarchy…[/Think][Ans]the man in black would shout back to express his anger…

[Think] The video shows the breaded chicken is now crispy, golden, and being glazed with ketchup...[/Think][Ans]A hand sprinkle cheese over the ketchup-coated chicken…

Ours(Joint-GRPO)

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

| |
|---|

| |
|---|

###### Shower grated cheese over the chicken cutlet.

A man in black suit shouting back angrily.

GroundTruth

| |
|---|

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

| |
|---|

- Figure 6. Visual comparison on VNEP. Captions are color-coded: green (correct), red (incorrect), blue (semantically correct but visually unfriendly). Yellow boxes highlight key regions. Baselines often fail in event prediction or visual consistency. Our SFT model improves reasoning but retains errors like semantic hallucination (predicting non-existent inreview in Case 1) and action misalignment (“adding cheese” yields pouring in Case 2). Joint-GRPO addresses both issues, enhancing model capability (correctly identifying document relationships and maintaining character appearance in Case 1) and fine-grained alignment (“sprinkle cheese” matching the GT “shower” in Case 2).

#### 5.2. Main Results

Quantitative Comparisons. Table 1 shows that VANS performs favorably against all baselines. On procedural benchmarks, VANS (Joint-GRPO) achieves a ROUGE-L of 0.3631 and CLIP-V of 0.8021, outperforming the strongest cascaded baseline (Gemini-FilmWeaver at 0.2802 and 0.7102) and unified model (Omni-Video at 0.1075 and 0.6293). More importantly, Joint-GRPO brings a significant gain over the

SFT version (e.g., ROUGE-L from 0.2812 to 0.3631 and CLIP-V from 0.7655 to 0.8021), demonstrating the effectiveness of our Joint-GRPO strategy. The Video Extension model Video-GPT yields the lowest CLIP-T (0.1997), as it generates frames without event reasoning. Please refer to Appendix D for additional results.

Qualitative Comparisons. As shown in Figure 6, baseline models frequently produce errors in either event prediction

Table 2. Quantitative results of ablation study.

Model BELU@1↑ BELU@2↑ BELU@3↑ BELU@4↑ ROUGE-L↑ FVD↓ CLIP-V↑ CLIP-T↑ Procedural Benchmarks

SFT 0.2524 0.1162 0.0501 0.0233 0.2812 85.34 0.7655 0.3202 GRPO (VLM) 0.2831 0.1498 0.0987 0.0698 0.3190 83.88 0.7798 0.3224 GRPO (VDM) 0.2524 0.1162 0.0501 0.0233 0.2812 84.76 0.7671 0.3013 GRPO (VLM+VDM) 0.2831 0.1498 0.0987 0.0698 0.2894 83.14 0.7703 0.3398 Joint-GRPO Stage 1 0.3257 0.1834 0.1242 0.0987 0.3631 80.23 0.7803 0.3521 Joint-GRPO Stage 1 (w/o rt1) 0.3176 0.1623 0.1123 0.0889 0.3498 83.31 0.7762 0.3454 Joint-GRPO Stage 1 (w/o rv1) 0.3252 0.1828 0.1240 0.0978 0.3625 82.34 0.7668 0.3403 Joint-GRPO Stage 1 + 2 (w/o rc2) 0.3257 0.1834 0.1242 0.0987 0.3631 78.55 0.7921 0.3673 Joint-GRPO Stage 1 + 2 (w/o rv2) 0.3257 0.1834 0.1242 0.0987 0.3631 79.76 0.7887 0.3806 Joint-GRPO Stage 1 + 2 (Ours) 0.3257 0.1834 0.1242 0.0987 0.3631 78.32 0.8021 0.3824 Joint-GRPO (all-in-one) 0.3012 0.1773 0.1003 0.0632 0.3577 81.01 0.7800 0.3423

Predictive Benchmarks

SFT 0.2247 0.0873 0.0206 0.0136 0.2435 94.12 0.7512 0.3038 GRPO (VLM) 0.2521 0.1124 0.0412 0.0289 0.2758 92.54 0.7643 0.3218 GRPO (VDM) 0.2247 0.0873 0.0206 0.0136 0.2435 93.45 0.7525 0.3051 GRPO (VLM+VDM) 0.2521 0.1124 0.0412 0.0289 0.2552 91.83 0.7558 0.3342 Joint-GRPO Stage 1 0.2789 0.1351 0.0853 0.0694 0.3058 89.92 0.7654 0.3462 Joint-GRPO Stage 1 (w/o rt1) 0.2718 0.1203 0.0765 0.0627 0.2934 92.11 0.7613 0.3396 Joint-GRPO Stage 1 (w/o rv1) 0.2785 0.1346 0.0851 0.0685 0.3052 91.15 0.7529 0.3345 Joint-GRPO Stage 1 + 2 (w/o rc2) 0.2789 0.1351 0.0853 0.0694 0.3058 88.36 0.7772 0.3612 Joint-GRPO Stage 1 + 2 (w/o rv2) 0.2789 0.1351 0.0853 0.0694 0.3058 89.57 0.7738 0.3642 Joint-GRPO Stage 1 + 2 (Ours) 0.2789 0.1351 0.0853 0.0694 0.3058 86.85 0.7872 0.3759 Joint-GRPO (all-in-one) 0.2574 0.1308 0.0684 0.0452 0.3012 90.82 0.7651 0.3365

Question: What will she do after 15 minutes Question: Show me the next step for baked chicken Parmesan

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

| |
|---|

| |
|---|

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 7. Visualization Results of ablation studies. Key regions are highlighted with yellow boxes: the left example shows degradation in the “mask removal” action completion without rt1; the right example illustrates the loss of visual consistency without rv2 and semantic alignment (leading to static frames) without rc2.

or visual consistency. For instance, Omni-Video misinterprets a quarrel as a fight and generates characters that deviate from the input. Our VANS after SFT demonstrates improved reasoning but reveals two key limitations: individual component errors, such as the VLM hallucinating non-existent text like “inreview” in Case 1, and a semantic-visual misalignment where the instruction “adding cheese” results in a pouring action instead of the ground-truth “showering” in Case 2. VANS with Joint-GRPO enhances the capability of

each component and achieves semantic-visual alignment, as evidenced by the precise caption “sprinkle cheese” and its faithful visualization that matches the “shower” action.

#### 5.3. Ablation Study

We conduct ablation studies to validate the design of JointGRPO, with results in Table 2 and Figure 7.

Joint vs. Isolated Optimization. Joint-GRPO outperforms variants where GRPO is applied solely to the VLM or VDM,

or where their individually optimized versions are simply cascaded. This confirms the necessity of joint optimization for coherent caption-video generation, where the VLM and VDM are co-adapted to bridge the semantic-to-visual gap.

Effect of Staged Training. The two-stage design proves critical: using only Stage 1 often produces captions and videos that deviate semantically, while an all-in-one variant suffers from optimization instability due to reward ambiguity—it becomes unclear whether a poor reward stems from the VLM’s caption or the VDM’s video generation.

Reward Component Analysis. Further ablation tests validate the contribution of each reward component. In Stage

- 1, removing the text fidelity reward rt1 reduces caption accuracy (e.g., failing to predict “removing the mask”), while

removing the video fidelity reward rv1 harms visual consistency. In Stage 2, removing the semantic alignment reward rc2 causes reward hacking with static frames, and removing the video fidelity reward rv2 reduces output coherence. These findings validate our complete design with staged optimization and balanced reward components.

### 6. Conclusion

This work pioneers Video-Next-Event Prediction (VNEP), a novel task that advances next-event reasoning from textual description to dynamic video demonstration. To address its unique challenges, we introduce VANS, which synergizes a VLM and a VDM through Joint-GRPO—a two-stage RL strategy coordinating both models under a joint reward. We construct VANS-Data-100K dataset to provide the essential training and evaluation foundation for this task. Experiments on established benchmarks demonstrate that VANS achieves SOTA performance in both event prediction accuracy and video generation quality.

### Acknowledgement

This work was supported by Kuaishou Technology.

## Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO Supplementary Material

This Appendix is organized as follows:

- • Section A provides the dataset details.
- • Section B offers an intuitive illustration of the Joint-GRPO reward design, along with training details.
- • Section C describes the implementation details of models.
- • Section D reports additional experimental results.

### A. Details of VANS-Data-100K

Table 3 demonstrates the composition and key statistics of our VANS-Data-100K dataset. It contains a total of 100K samples, with 30K dedicated to procedural tasks and 70K to predictive scenarios. These are sourced from a diverse set of publicly available video datasets and Internet to ensure broad coverage of real-world dynamics and instructional content.

In terms of video characteristics, the input videos have an average duration of 9.43 seconds, providing sufficient context for event reasoning. The corresponding target videos, which depict the predicted next event, average 3.76 seconds in length, ensuring concise and focused demonstrations.

### B. Details of Joint-GRPO

#### B.1. Reward Design

- Figure 8 provides an intuitive illustration of how the individual reward components work in concert within our two-stage training process of Joint-GRPO.

- In Stage 1 (VLM Tuning), we examine the role of the text

fidelity reward (rt1) and the video fidelity reward (rv1). For the provided example, if only rt1 is used, Sample 2 receives a high score comparable to the Ground Truth (GT), as both captions correctly describe the action. However, Sample 2 exhibits poor visual consistency. Conversely, if only rv1 is used, both Sample 1 and Sample 2 receive similarly low scores, failing to reflect that Sample 1 is semantically worse due to an incorrect action prediction. Only when both rt1 and rv1 are combined does the composite reward correctly rank the samples, successfully identifying the GT as the best.

- In Stage 2 (VDM Adaptation), we analyze the video

fidelity reward (rv2) and the semantic consistency reward (rc2). Relying solely on rv2 results in Sample 1 receiving a score similar to the GT, even though Sample 1 depicts an incorrect semantic action (it should show two people pointing guns at each other). Using only rc2 causes Sample 2 to be scored similarly to the GT, despite its poor visual consistency. The joint reward effectively combines these signals to prioritize samples that are correct in both semantics and visual quality.

The final combined reward in each stage is a sum of the

Table 3. Statistics of VANS-Data-100K dataset.

Component Size/Duration Data Composition

Procedural (Total: 30K)

YouCook2 [54] 9K COIN [51] 21K

Predictive (Total: 70K)

Video-Holmes [8] 10K ActivityNet [2] 20K V1-33K [39] 10K YouTube Videos 30K

###### Video Duration (Avg.)

Input Video 9.43s Target Video 3.76s

normalized individual rewards. All weighting coefficients (λ) are set to 1, assigning equal importance to each objective. This design ensures a balanced optimization towards captions that are both semantically accurate and visually plausible (Stage 1), and videos that are both high-quality and semantically faithful (Stage 2).

#### B.2. Training Process

Figure 9 illustrates the training dynamics of Joint-GRPO. In Stage 1 (VLM Tuning), the format reward (rf) in Figure 9(a) quickly saturates, indicating rapid adoption of the instruction template. Both text fidelity (rt1, Figure 9b) and video fidelity (rv1, Figure 9c) rewards show progressive improvement, reflecting the VLM’s learning to generate captions that are both semantically accurate and visually plausible. The combined reward (Figure 9d) stabilizes after approximately 600 steps, demonstrating effective optimization. Concurrently, the increasing thinking length (Figure 9e) indicates more detailed reasoning chains.

In Stage 2 (VDM Adaptation), both video fidelity (rv2, Figure 9f) and semantic alignment (rc2, Figure 9g) rewards improve consistently, with convergence occurring after about 1000 steps. This demonstrates the VDM’s successful adaptation to generate videos that preserve visual consistency while faithfully rendering the semantically-grounded captions from Stage 1. The total reward (Figure 9h) also reaches a stable level, confirming effective cross-modal alignment.

Collectively, these training curves validate the effectiveness of our Joint-GRPO design, demonstrating coordinated improvement across both stages.

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Think] From the input video, I can see that the user is preparing to pour the gel…[Ans] A hand pours white gel from a glass into the oval

[Figure 204]

[Figure 205]

[Figure 206]

Format reward

[Figure 207]

Sample2Sample2Ground TruthSample1Sample1Ground Truth

Text reward Video reward

r1

RewardModel(VDM)

[Figure 208]

PolicyModel(VLM)

###### basin.[/Ans]

###### Stage2Stage1

[Think]The white gel has been added, along with red and blue

[Figure 209]

[Figure 210]

[Figure 211]

|[Figure 212]<br><br>Model|
|---|

[Figure 213]

r2

pigment…[/Thinnk][Ans]A hand

stir the blue and red pigment in the basin with a spoon[/Ans]

[Figure 214]

[Think]The white gel has been added, along with red and blue

[Figure 215]

[Figure 216]

rgt

pigment…[/Thinnk][Ans]A hand

stir the red and blue pigment in the oval basin with a spoon[/Ans]

Q: How to continue

with my slime?

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Consistency

[Figure 222]

[Figure 223]

reward

r1

[Figure 224]

[Figure 225]

AnchorModel(VLM)

[Figure 226]

PolicyModel(VDM)

[Think] Based on the input video, the man on the left is being threatened with a gun. In such a situation, he might also draw a gun to defend himself[/Think][Ans] Two men are in a standoff, aiming their

[Figure 227]

[Figure 228]

|[Figure 229]<br><br>[Figure 230]|
|---|

|[Figure 231]<br><br>Model|
|---|

[Figure 232]

r2

[Figure 233]

guns at one another. [/Ans]

[Figure 234]

[Figure 235]

rgt

Q: What will the man do to protect himself?

- Figure 8. Illustration of our Joint-GRPO reward design. Top: For a Stage-1 case, we simulate three reasoning samples during GRPO training. The text-only reward fails to penalize Sample 2’s visual inconsistency, while the video-only reward fails to penalize Sample 1’s semantic error. Bottom: For a Stage-2 case, the video-only reward fails to penalize Sample 1’s semantic inaccuracy, while the consistency-only reward fails to penalize Sample 2’s visual inconsistency.

(a) Stage 1: format reward rf (b) Stage 1: text fidelity reward rt1 (c) Stage 1: video fidelity reward rv1 (d) Stage 1: total reward

(e) Stage 1: thinking length (f) Stage 2: video fidelity reward rv2 (g) Stage 2: alignment reward rc2 (h) Stage 2: total reward

[Figure 236]

[Figure 237]

- Figure 9. Training curves of Joint-GRPO: (a) format reward (rf) in Stage 1; (b) text fidelity reward (rt1) in Stage 1; (c) video fidelity reward (rv1) in Stage 1; (d) total reward in Stage 1; (e) thinking length evolution in Stage 1; (f) video fidelity reward (rv2) in Stage 2; (g) semantic alignment reward (rc2) in Stage 2; (h) total reward in Stage 2.

### C. Implementation Details

#### C.1. Training of VANS

We initialize VANS with Qwen2.5-VL-3B as the VLM and Wan-2.1-1.3B as the VDM. The VDM is configured to use n = 6 reference frames.

In the SFT stage, the VLM is trained for 10K steps using LoRA [14] (rank=8, alpha=32) with a learning rate of 5 × 10−5, while the VDM is fully fine-tuned for 20K steps across all DiT blocks with the same learning rate of 5 × 10−5.

For Joint-GRPO post-training, Stage 1 is optimized for 800 steps with a learning rate of 5 × 10−5. In Stage 2, to ensure the quality of anchor captions, we filter out those with ROUGE-L scores below 0.6 before proceeding with VDM adaptation. Stage 2 is then trained for 1K steps with the same learning rate of 5 × 10−5. We equip the VLM with LoRA (rank=8, alpha=32). For the VDM, we adopt the method from [22] to convert a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) to enable GRPO training. We set the KL coefficient β = 0.004, clip range to 1 × 10−3, and sample group size to 8 per prompt.

#### C.2. Evaluation

Evaluation Protocol. All methods are evaluated under a unified protocol. For Video-GPT, we provide only the input video and utilize its native video continuation capability. For VANS and other baseline methods, we provide the input video along with the corresponding question and the following system prompt:

You will be given a video. Your task is to predict the next event based on the input video and the user’s instructions. Please begin by providing your detailed reasoning between the [Think][/Think] tags, followed by your detailed description of the next event within the [Ans][/Ans] tags.

Input Adaptation. To accommodate different model architectures, we adapt the video input accordingly: for models that can directly process video input (e.g., Gemini), we provide the original video; for other models (e.g., Qwen), we set the input video fps = 1 for their video processors.

Output Specification. All methods are required to generate a video answer with a resolution of 352 × 640 and a length of 33 frames to ensure consistent and fair comparison.

Metric Computation. The CLIP-Score for video consistency (CLIP-V) and semantic consistency (CLIP-T) is computed using a ViT-B/32 model. Specifically, each frame of the generated video is compared with the corresponding frame in the ground-truth video, and the scores are averaged across all frames.

Table 4. Results on procedural VNEP. The comparison with finetuned baselines (*) shows that our architectural design, rather than data advantage, is the primary source of improvement.

Model BELU@4↑ ROUGE-L↑ FVD↓ CLIP-V↑ CLIP-T↑

Qwen-Wan 0.0013 0.1530 148.75 0.6619 0.2448 Qwen*-Wan 0.0233 0.2812 140.32 0.6790 0.2466 Qwen*-Wan* 0.0233 0.2812 140.07 0.6795 0.2470 Gemini-Wan 0.0215 0.2802 120.34 0.6898 0.2547 VANS (SFT) 0.0233 0.2812 85.34 0.7655 0.3202 VANS (Joint-GRPO) 0.0987 0.3631 78.32 0.8021 0.3824

### D. Additional Results

#### D.1. Inference Time

The inference time of VANS is comparable to other cascaded pipelines, requiring approximately 4 seconds for caption generation and 35 seconds for video generation using the official VAN library. In contrast, unified models exhibit longer inference times: Omni-Video requires approximately 50 seconds, while VideoGPT needs about 60 seconds for complete generation.

#### D.2. Comparison with Fine-tuned Baseline

To analyze the source of performance improvements in VANS, we compare it with fine-tuned baselines. As shown in Table 4, the results indicate three main observations: data quality provides a foundation, architectural modification contributes to noticeable gains, and Joint-GRPO provides the decisive enhancement that pushes performance to the state-of-the-art level.

Data Quality as the Foundation. When fine-tuned on our VANS-Data-100K for 10K steps (denoted as Qwen*), the model achieves reasoning capability competitive with Gemini-2.5-Flash in a zero-shot setting (ROUGE-L: 0.2812 vs. 0.2802). This confirms that our high-quality dataset enables smaller models to learn sophisticated reasoning.

Architectural Modification Contributes to Gains. Finetuning both components of the Qwen-Wan pipeline (denoted as Qwen*-Wan*) yields limited video metric improvements over the base fine-tuned VLM (denoted as Qwen*-Wan). In contrast, VANS (SFT) with the same text input achieves better video results: FVD decreases from 140.07 to 85.34 and CLIP-V increases from 0.6795 to 0.7655, suggesting the proposed VAE reference feature aids visual consistency.

Joint-GRPO Delivers the Decisive Enhancement. The most striking improvement comes from Joint-GRPO, which elevates VANS to unprecedented performance levels across all metrics. Compared to VANS (SFT), Joint-GRPO boosts ROUGE-L from 0.2812 to 0.3631 (29.1% relative improvement) and CLIP-T from 0.3202 to 0.3824 (19.4% relative improvement), while further reducing FVD to 78.32. These results unequivocally demonstrate that Joint-GRPO is the

Prompt: Man picking up unused paper

Prompt: Leave the bananas for a week

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Omni-VideoOursGemini-WanI2V

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

Figure 10. Visual comparison results on UI2V-Bench.

Input Video

[Figure 262]

[Figure 263]

[Figure 264]

- Q1: What is her reaction if she gets burned?
- Q2: What is her reaction if she gets burned in an exaggerated movie?
- Q3: Show her reaction if she eats something spicy in an exaggerated movie.

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Figure 11. Multi-Future Prediction Results.

most critical component for achieving state-of-the-art performance, effectively aligning both textual and visual outputs with human preferences.

#### D.3. Generalization

Multi-Future Prediction. The established NEP task typically assumes a single, causal progression from the input context. In contrast, our VANS demonstrates a key generalization capability: multi-future prediction. This allows the model to generate semantically distinct and contextually

appropriate video answers based on different hypothetical questions applied to the same input video, moving beyond deterministic continuation.

As shown in Figure 11, when presented with a scene of a woman reacting to a hot object, VANS can generate fundamentally different yet plausible outcomes conditioned on the scenario: in a realistic everyday context, it predicts a natural reaction of “coughing”; whereas in a stylized cinematic context, it visualizes a dramatic effect of “smoke exhaling from the mouth”. This flexibility stems from our model’s ability to ground its predictions in both the visual evidence and the diverse textual hypotheses provided, effectively exploring multiple potential futures from a single starting point.

Reasoning Image-to-Video Generation. VANS generalizes effectively to the reasoning image-to-video (I2V) task by treating a single image as a static video clip. This generalization capability is attributed to the model’s training on mixed datasets including Koala-36M [40] for I2V tasks. Figure 10 demonstrates examples from UI2V-Bench [53], when given an image of a banana and the instruction ”leave the banana for a week,” our model accurately predicts the temporal evolution, generating a video where the banana skin darkens. In contrast, other strong baselines struggle to capture this causal-physical transformation correctly. This demonstrates the robustness of our approach in understanding static visual contexts and reasoning about their potential dynamic futures.

#### D.4. Human Evaluation

To complement automatic metrics, we conduct a human evaluation to assess the subjective quality of generated video answers. We recruit 30 evaluators (mean age = 25 years; all hold at least a bachelor’s degree, including 20 postgraduate/PhD students and 10 full-time professionals). Each

Table 5. Human evaluation results (scale: 1-5). Our VANS with Joint-GRPO achieves the highest scores across all criteria.

Model Semantic Correctness Visual Consistency Overall

Video-GPT 1.5 3.6 1.5 Omni-Video 2.1 3.2 2.2 Gemini-FilmWeaver 3.9 3.1 3.5 VANS (SFT) 3.8 3.9 3.7 VANS (Joint-GRPO) 4.7 4.6 4.8

evaluator is presented with 20 randomly selected examples (10 procedural and 10 predictive) and rates the results on three dimensions: semantic correctness, visual consistency, and overall satisfaction.

The results in Table 5 reveal that VANS (SFT) achieves semantic correctness comparable to the strong baseline GeminiFilmWeaver, while demonstrating superior visual consistency. Furthermore, VANS with Joint-GRPO receives the highest ratings across all three criteria, indicating that our full approach yields video answers that are not only semantically and visually accurate but also subjectively more satisfactory to human observers.

#### D.5. Video Results

All video results corresponding to the figures in this paper, along with additional examples, are provided in supplementary.html.

### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609,

2023. 6

- [2] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 4, 6, 1

- [3] Yu-Wei Chao, Zhan Wang, Yugeng He, Jiaxuan Wang, and Jia Deng. Hico: A benchmark for recognizing humanobject interactions in images. In Proceedings of the IEEE international conference on computer vision, pages 1017– 1025, 2015. 3

- [4] Brian Chen, Xudong Lin, Christopher Thomas, Manling Li, Shoya Yoshida, Lovish Chum, Heng Ji, and Shih-Fu Chang. Joint multimedia event extraction from video and article. arXiv preprint arXiv:2109.12776, 2021. 3

- [5] Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, and Xihui Liu. Grpo-care: Consistency-aware reinforcement learning for multimodal reasoning. arXiv preprint arXiv:2506.16141, 2025. 3

- [6] Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Lu Qiu, Ying Shan, and Xihui Liu. Exploring the effect of reinforcement learning on video understanding: Insights from seed-bench-r1. arXiv preprint arXiv:2503.24376, 2025. 3

- [7] Junhao Cheng, Yuying Ge, Yixiao Ge, Jing Liao, and Ying Shan. Animegamer: Infinite anime life simulation with next game state prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10875– 10885, 2025. 2

- [8] Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. Video-holmes: Can mllm think like holmes for complex video reasoning? arXiv preprint arXiv:2505.21374, 2025. 4, 6, 1

- [9] Jen-Hao Cheng, Vivian Wang, Huayu Wang, Huapeng Zhou, Yi-Hao Peng, Hou-I Liu, Hsiang-Wei Huang, Kuang-Ming Chen, Cheng-Yen Yang, Wenhao Chai, et al. Tempura: Temporal event masked prediction and understanding for reasoning in action. arXiv preprint arXiv:2505.01583, 2025. 2, 3, 6

- [10] Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S Yu, et al. Llm agents for education: Advances and applications. arXiv preprint arXiv:2503.11733, 2025. 2

- [11] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 3

- [12] Yuying Ge, Yixiao Ge, Chen Li, Teng Wang, Junfu Pu, Yizhuo Li, Lu Qiu, Jin Ma, Lisheng Duan, Xinyu Zuo, et al. Archunyuan-video-7b: Structured video comprehension of realworld shorts. arXiv preprint arXiv:2507.20939, 2025. 3

- [13] Sagar Goyal, Eti Rastogi, Sree Prasanna Rajagopal, Dong Yuan, Fen Zhao, Jai Chintagunta, Gautam Naik, and Jeff Ward. Healai: A healthcare llm for effective medical documentation. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pages 1167–1168, 2024. 2

- [14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3,

2022. 3

- [15] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025. 3

- [16] Chenghang Lai, Weifeng Ge, and Xiangyang Xue. Crossmodal complementary learning and template-based reasoning chains for future event prediction in videos. IEEE Transactions on Multimedia, 2025. 2, 3, 6

- [17] Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. What is more likely to happen next? video-and-language future event prediction. arXiv preprint arXiv:2010.07999, 2020. 3

- [18] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958,

2025. 3

- [19] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 5

- [20] Chin-Yew Lin and Franz Josef Och. Automatic evaluation of machine translation quality using longest common subsequence and skip-bigram statistics. In Proceedings of the 42nd annual meeting of the association for computational linguistics (ACL-04), pages 605–612, 2004. 6

- [21] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 2

- [22] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 3

- [23] Jiabin Luo, Junhui Lin, Zeyu Zhang, Biao Wu, Meng Fang, Ling Chen, and Hao Tang. Univid: The open-source unified video model. arXiv preprint arXiv:2509.24200, 2025. 2

- [24] Xiangyang Luo, Qingyu Li, Xiaokun Liu, Wenyu Qin, Miao Yang, Pengfei Wan, Di Zhang, Kun Gai, and Shao-Lun Huang. Filmweaver: Weaving consistent multi-shot videos with cache-guided autoregressive diffusion. AAAI, 2026. 6

- [25] Yapeng Mi, Hengli Li, Yanpeng Zhao, Chenxi Li, Huimin Wu, Xiaojian Ma, Song-Chun Zhu, Ying Nian Wu, and Qing Li. Milr: Improving multimodal image generation via test-time latent reasoning. arXiv preprint arXiv:2509.22761, 2025. 3

- [26] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318,

2002. 6

- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 5, 6

- [28] John R Searle. Minds, brains, and programs. Behavioral and brain sciences, 3(3):417–424, 1980. 2

- [29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3

- [30] Liao Shen, Wentao Jiang, Yiran Zhu, Tiezheng Ge, Zhiguo Cao, and Bo Zheng. Identity-preserving image-to-video generation via reward-guided optimization. arXiv preprint arXiv:2510.14255, 2025. 3

- [31] Tom´aˇs Souˇcek, Prajwal Gatti, Michael Wray, Ivan Laptev, Dima Damen, and Josef Sivic. Showhowto: Generating sceneconditioned step-by-step visual instructions. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 27435–27445, 2025. 2

- [32] Qile Su, Shoutai Zhu, Shuai Zhang, Baoyu Liang, and Chao Tong. Eventformer: A node-graph hierarchical attention transformer for action-centric video event prediction. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4698–4707, 2025. 3

- [33] Reuben Tan, Matthias De Lange, Michael Iuzzolino, Bryan A Plummer, Kate Saenko, Karl Ridgeway, and Lorenzo Torresani. Multiscale video pretraining for long-term activity forecasting. arXiv preprint arXiv:2307.12854, 2023. 3

- [34] Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025. 2, 6

- [35] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 4, 6

- [36] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2

- [37] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 6
- [38] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 4, 6

- [39] Haonan Wang, Hongfu Liu, Xiangyan Liu, Chao Du, Kenji Kawaguchi, Ye Wang, and Tianyu Pang. Fostering video reasoning via next-event prediction. arXiv preprint arXiv:2505.22457, 2025. 3, 4, 1

- [40] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, Fei Yang, Pengfei Wan, and Di Zhang. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content, 2024. 4
- [41] Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, and Mohit Bansal. Video-rts: Rethinking reinforcement learning and test-time scaling for efficient and enhanced video reasoning. arXiv preprint arXiv:2507.06485, 2025. 3

- [42] Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025. 2

- [43] Chi Hsuan Wu, Kumar Ashutosh, and Kristen Grauman. Stitch-a-recipe: Video demonstration from multistep descriptions. arXiv preprint arXiv:2503.13821, 2025. 2

- [44] Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, and Ying Shan. Mindomni: Unleashing reasoning generation in vision language models with rgpo. arXiv preprint

- arXiv:2505.13031, 2025. 2, 3

[45] Yicheng Xiao, Lin Song, Rui Yang, Cheng Cheng, Zunnan Xu, Zhaoyang Zhang, Yixiao Ge, Xiu Li, and Ying Shan. Haploomni: Unified single transformer for multimodal video understanding and generation. arXiv preprint

- arXiv:2506.02975, 2025. 2

- [46] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 3

- [47] Zhiyuan Yan, Kaiqing Lin, Zongjian Li, Junyan Ye, Hui Han, Zhendong Wang, Hao Liu, Bin Lin, Hao Li, Xue Xu, et al. Can understanding and generation truly benefit together–or just coexist? arXiv e-prints, pages arXiv–2509, 2025. 3

- [48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2

- [49] Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139, 2024. 2

- [50] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2

- [51] Tang Yansong, Ding Dajun, Rao Yongming, Zheng Yu, Zhang Danyang, Zhao Lili, Lu Jiwen, and Zhou Jie. Coin: A largescale dataset for comprehensive instructional video analysis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 4, 6, 1

- [52] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325,

2025. 2

- [53] Ailing Zhang, Lina Lei, Dehong Kong, Zhixin Wang, Jiaqi Xu, Fenglong Song, Chun-Le Guo, Chang Liu, Fan Li, and Jie Chen. Ui2v-bench: An understanding-based image-to-video generation benchmark. arXiv preprint arXiv:2509.24427,

2025. 4

- [54] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, 2018. 4, 6, 1

- [55] Shaobin Zhuang, Zhipeng Huang, Ying Zhang, Fangyikang Wang, Canmiao Fu, Binxin Yang, Chong Sun, Chen Li, and Yali Wang. Video-gpt via next clip diffusion. arXiv preprint arXiv:2505.12489, 2025. 2, 6

