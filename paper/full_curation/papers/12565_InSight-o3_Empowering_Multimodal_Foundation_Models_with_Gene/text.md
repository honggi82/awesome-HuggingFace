# arXiv:2512.18745v1[cs.CV]21Dec2025

## INSIGHT-O3: EMPOWERING MULTIMODAL FOUNDATION MODELS WITH GENERALIZED VISUAL SEARCH

##### Kaican Li1∗, Lewei Yao2∗, Jiannan Wu2∗, Tiezheng Yu2, Jierun Chen2, Haoli Bai2, Lu Hou2, Lanqing Hong2, Wei Zhang2, Nevin L. Zhang1† 1Hong Kong University of Science and Technology, 2Huawei

ABSTRACT

The ability for AI agents to “think with images” requires a sophisticated blend of reasoning and perception. However, current open multimodal agents still largely fall short on the reasoning aspect crucial for real-world tasks like analyzing documents with dense charts/diagrams and navigating maps. To address this gap, we introduce O3-BENCH, a new benchmark designed to evaluate multimodal reasoning with interleaved attention to visual details. O3-BENCH features challenging problems that require agents to piece together subtle visual information from distinct image areas through multi-step reasoning. The problems are highly challenging even for frontier systems like OpenAI o3, which only obtains 40.8% accuracy on O3-BENCH. To make progress, we propose INSIGHT-O3, a multi-agent framework consisting of a visual reasoning agent (vReasoner) and a visual search agent (vSearcher) for which we introduce the task of generalized visual searchlocating relational, fuzzy, or conceptual regions described in free-form language, beyond just simple objects or figures in natural images. We then present a multimodal LLM purpose-trained for this task via reinforcement learning. As a plugand-play agent, our vSearcher empowers frontier multimodal models (as vReasoners), significantly improving their performance on a wide range of benchmarks. This marks a concrete step towards powerful o3-like open systems. Our code and dataset can be found at https://github.com/m-Just/InSight-o3.

1 INTRODUCTION

Thinking with images is an important and very useful skill for multimodal agents (OpenAI, 2025c). The skill rests on two crucial and fundamental cognitive abilities: reasoning and perception. Recent efforts at developing such a skill based on open models mainly focus on the perception component, e.g., searching for a particular object or figure in natural images and then answering a simple visual query about them (Wu & Xie, 2024; Shen et al., 2024; Li et al., 2025; Zhang et al., 2025b; Su et al., 2025a; Zheng et al., 2025; Wang et al., 2025c; Zhu et al., 2025b; Wang et al., 2025a; Lai et al., 2025). While this feature is useful, it is still far from being able to handle many real-world tasks that require deeper and more abstract reasoning. Typical examples include extracting information from complex reports and navigating through intricate maps. Solving these tasks often require both organized reasoning and focused attention to visual details scattered across an image (or images).

Currently, the reasoning capability of open multimodal models is still relatively weak in comparison with frontier proprietary models (Yue et al., 2024a; Yuan et al., 2025; Hao et al., 2025). This makes it very difficult to replicate the kind of reasoning-driven image-thinking behavior demonstrated by OpenAI o3 (OpenAI, 2025c). In this work, we take a concrete step towards building such an intelligent system with open models. First, we propose a new multimodal benchmark, O3-BENCH, to help better evaluate the general capability of multimodal models to think with images. Complementary to most of the existing benchmarks which only deal with object attributes and spatial relations in natural images (Wu & Xie, 2024; Wang et al., 2025g; Lai et al., 2025; Wang et al., 2025a), O3-BENCH consists of a set of high-quality, reasoning-oriented questions on images of high information density. The questions involve real-world tasks such as map navigation and cross-chart/diagram analysis that are highly challenging even for frontier systems like OpenAI o3. Compared with benchmarks like

∗Equal contribution. †Corresponding author (lzhang@cse.ust.hk).

3

[Figure 1]

3

[Figure 2]

[Figure 3]

[Figure 4]

###### Question

2

Which attraction in Draken Valley can a 128cm child play alone without needing a guardian and is closest to the Draken Snake Meals food?

[Figure 5]

- A. Draken
- B. Dragon Race
- C. Sköll & Hati
- D. Klake
- E. Valkyrie
- F. No right choice

[Figure 6]

✅

2

[Figure 7]

[Figure 8]

1

[Figure 9]

###### 1

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Find the Draken Snack location and nearby attractions, then check height rules. The meals legend shows "Draken Snack" is item #2. On the map the #2 meal icon is positioned right next to the Valkyrie attraction. The attractions legend lists Valkyrie as 120cm+ (so a 128cm child can ride alone). Other nearby Draken Valley rides (Draken and Klake) require 135cm+, so the 128cm child cannot ride those alone. Therefore the closest ride that a 128cm child can ride solo next to Draken Snack is Valkyrie.

Need to inspect the zoomed-in meals list/map area to locate "Draken Snake Meals" (or similarly named) and then find closest attraction within Draken Valley region. After identifying the food location, check nearby attractions and their height/guardian requirements. I'll request the zoomed crop now.

I need to confirm the exact height/guardian rules for the Draken Valley attractions (Draken, Valkyrie, Klake, Dragon Race, Sköll & Hati) from the attractions legend. I'll request a crop of the left legend area showing the Draken Valley attraction list and their height requirements.

I need to locate the specific meal marker for "Draken Snack" (#2) on the full map and see which Draken Valley attraction is nearest. I will request a crop of the Draken Valley lower-right region showing attractions and nearby meal/snack markers.

The answer is \boxed{E}

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

lower-right Draken Valley crop around meal marker #2 ‘Draken Snack’, showing nearby meal/snack markers

[Figure 22]

left side legend listing "ATTRACTION Draken Valley" with heights and guardian requirements

[Figure 23]

[Figure 24]

zoomed-in map lower center showing Meals list and map around Draken Valley lowerright area

1 2 3

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

vReasoner vSearcher Closed View Region Description

- Figure 1: A multi-step visual reasoning example of INSIGHT-O3 on O3-BENCH. For clarity, the internal reasoning processes are omitted. More examples can be found in Appendix D.2.

MME-RealWorld (Zhang et al., 2024b), O3-BENCH is significantly harder, requiring the evaluated system to collect detailed visual information from multiple distinct image areas while performing complex, interleaved reasoning using the information collected in the process.

To make substantive progress on O3-BENCH, we introduce a multi-agent framework, INSIGHT-O3, that comprises a visual reasoning agent (vReasoner) and a visual search agent (vSearcher). The former is responsible for high-level reasoning and general image understanding, while the latter is to help vReasoner locate specific regions of interest and collect the visual information therein. As such, INSIGHT-O3 reduces the burden of a single agent, allowing us to build an o3-like system via divide-and-conquer. This kind of specialization has been shown to work in prior art (Dayan & Hinton, 1992; Zeng et al., 2023; Shen et al., 2023; Li et al., 2023a; Hong et al., 2023; Castrejon

- et al., 2024). In this work, we focus on vSearcher and how it should interact with a given vReasoner. Different from current practices (Wu & Xie, 2024; Lai et al., 2025) for natural images and discrete object references, we aim to solve generalized visual search, where the input image can be arbitrary, e.g., a map, a poster, or a screenshot; and the referring description may specify a relational, fuzzy, or conceptual region, e.g., “the area to the left of the wooden chair,” and “the chart showing the company’s revenue in the last decade,” rather than a specific object or figure. Such fuzzy descriptions are more in line with how humans reason and direct their attention to a general region of interest.

To address this broader challenge, we present InSight-o3-vS, a vSearcher model specialized in generalized visual search through reinforcement learning. InSight-o3-vS combines multimodal understanding with spatial reasoning to localize regions described in completely free-form language. The

name of our model, InSight-o3-vS, reflects its dual role: providing deeper insight into multimodal semantics while bringing the target region in sight through precise localization. Our model empowers existing multimodal foundation models (as vReasoners) in a plug-and-play fashion, significantly improving the performance of frontier models across a wide range of benchmarks, e.g., from 39.0% to 61.5% on O3-BENCH for GPT-5-mini (OpenAI, 2025a), and from 80.1% to 87.6% on V⋆-Bench for Gemini-2.5-Flash (Comanici et al., 2025).

To summarize, we make the following key contributions in this work:

- • We propose a new benchmark, O3-BENCH, to evaluate complex, reasoning-oriented visual tasks. This benchmark features challenges like map navigation and cross-chart analysis, which require collecting information from multiple image areas and performing interleaved reasoning, making it significantly harder than existing benchmarks.
- • We introduce INSIGHT-O3, a multi-agent framework that divides the task of “thinking with images” between a high-level reasoning agent (vReasoner) and a visual search agent (vSearcher). This divide-and-conquer design greatly simplifies the complex interleaved reasoning, allowing us to build o3-like systems that surpass OpenAI o3 across a variety of benchmarks.
- • We present InSight-o3-vS, a specialized vSearcher model that excels at generalized visual search. It is designed to be a “plug-and-play” component that empowers existing multimodal foundation models, demonstrably and significantly improving the performance of frontier systems on a wide range of benchmarks including O3-BENCH.

- 2 RELATED WORK

We provide a brief overview of the most relevant related work in this section. For a more comprehensive discussion, please refer to Appendix A.

Multimodal benchmarks. Classical multimodal benchmarks (Goyal et al., 2017; Saikh et al., 2022; Liu et al., 2023; Ge et al., 2024) mainly test coarse image-level or salient-attribute recognition, where modern MLLMs are near-saturated (Bai et al., 2025b; Wang et al., 2025e). Recent multimodal reasoning benchmark split into (i) cognition-centric STEM benchmarks (Lu et al., 2023; Yue et al., 2024a;b) that emphasize multi-step/world-knowledge reasoning but use visually simple images, and (ii) perception-centric datasets (Wu & Xie, 2024; Zhang et al., 2024b; Lai et al., 2025) that require fine-grained recognition in high-resolution, text-rich scenes yet often limited to singleregion lookups. Motivated by the “think with images” paradigm (OpenAI, 2025c), O3-BENCH jointly evaluates search/localization and higher-level reasoning on high-information-density charts and maps, requiring cross-region evidence aggregation via interleaved, multi-hop reasoning.

Multimodal reasoning models. Reinforcement learning (RL) has long been used to align model behavior with human preferences (Schulman et al., 2017). DeepSeek-R1 applies group relative policy optimization (GRPO) (Guo et al., 2025a; Shao et al., 2024b), reliably eliciting planning, reflection, and long chain-of-thought reasoning under simple rewards. Building on this idea, recent multimodal models (Yang et al., 2025c) adopt GRPO-style training and report strong gains, while cascaded RL stages (e.g., InternVL3.5 (Wang et al., 2025e), Keye-VL1.5 (Yang et al., 2025a)) further push reasoning, approaching proprietary models. There are also pioneering work utilizing Python code execution to call various vision tools to solve tasks via divide-and-conquer or to help with reasoning (Gupta & Kembhavi, 2023; Sur´ıs et al., 2023; Ke et al., 2024). Nevertheless, most multimodal reasoners remain text-centric, overlooking the distinctive demands of visual reasoning.

Visual search models. Visual search is a core multimodal capability, requiring active region perception for fine-grained understanding. Early methods relied on external detectors or scripted workflows to localize regions and triggered tools via instruction tuning, leading to rigid outputs and typically single-round search (Wu & Xie, 2024). The “think with images” paradigm (OpenAI, 2025c) internalizes zoom/crop operations and has inspired end-to-end search (DeepEyes (Zheng

- et al., 2025)), synthetic warm-starts (Pixel-Reasoner (Su et al., 2025a)), and multi-turn RL (Minio3 (Lai et al., 2025)). Nonetheless, most systems still emphasize finding a single region in natural images, with limited support for multi-hop reasoning. We broaden this scope by decoupling visual search from visual reasoning and enabling multi-region search on arbitrary images.

### 3 O3-BENCH

We conceptualize “thinking with images” as an iterative perception-reasoning process. Perception focuses on searching and localizing task-relevant visual details, while reasoning needs to organize these cues into structured facts and performs higher-order inference (e.g., planning, arithmetic, use of world knowledge) to complete the task. These two critical skills should be executed effectively and cooperate tightly to achieve strong performance. Existing benchmarks (Wu & Xie, 2024; Zhang et al., 2024b; Lai et al., 2025) primarily emphasize perception, where their questions hardly require multi-step reasoning and thus induce short reasoning chains. To bridge this gap, we introduce O3-BENCH, a benchmark that jointly assesses high-resolution perception and multi-hop visual reasoning. O3-BENCH is designed with two principles:

- • High resolution & high information density. Images are large, high-resolution, cluttered, and information-dense, making evidence gathering genuinely non-trivial.
- • Multi-hop solution paths. Questions require decomposing the goal, retrieving evidence from multiple regions, and composing it via intermediate steps before answering.

To instantiate these principles, O3-BENCH comprises two complementary domains: (1) Composite charts. Each image contains multiple heterogeneous charts (e.g., bar/line/pie/tables). Our crafted questions demand cross-chart retrieval (series, axes, units), lightweight calculations (differences, ratios, aggregates), and consistency checks (scale, legend, time ranges) to derive the final answer. (2) High-resolution digital maps. The images typically include a map along with auxiliary components such as legends and building indices. We meticulously design questions that require visual search for targets (e.g., matching symbols, categories, or toponyms) and spatial reasoning about relations and routes (e.g., proximity constraints or shortest paths), conditioned on the provided context.

Overall, O3-BENCH comprises 204 images (117 charts, 87 maps) and 345 QA samples (163 for charts, 182 for maps) in total. The majority of samples fall into the more challenging map category, underscoring our prioritization of complex visual perception and multi-hop reasoning. The questions of O3-BENCH are multi-choice questions with six choices and one correct answer. Among the six choices, there are four distractors that appear in the image or look similar to the correct one. We also include an option F as No Right Choice if there are no correct options provided. Below, we present the construction process of O3-BENCH. For other details about O3-BENCH, see Appendix B.1.

- 3.1 SOURCE DATA COLLECTION

Chart. The chart images in O3-BENCH are curated from the “Diagram and Table” subset from MME-RealWorld (Zhang et al., 2024b) and the Internet. To ensure high information density, we run a layout detection model, PP-DocLayout plus-L (Cui et al., 2025), on the candidate images and only keep those with at least 8 detected layouts. As a result, 256 of 2,539 images (from MME-RealWorld) that contain sufficient number of sub-figures and rich recognizable texts are left.

Map. We manually collect high-resolution digital maps from the Internet via keyword search. We center on the venue-level maps that require reading the provided legend/index and visually locating entities within the image to answer the question. We exclude all the country-, state-, or city-scale cartography that could be potentially answered with world knowledge. Through this process, we end up with 87 high-density map images spanning the categories over bus routes, campus, park, etc.

- 3.2 ANNOTATION PIPELINE

After the collection and initial filtering process, all images then undergo further manual screening to ensure clarity and completeness of key visual cues (e.g., axes, units and legends). Next, we combine automated machine pre-annotation with human verification and authoring to generate the question-answer (QA) instances. The detailed process of data annotation is presented as follows.

Machine pre-annotation. To relieve the burden of human annotators and increase the data diversity, we first apply a three-step automated data pipeline to generate five questions for each image.

- (1) Layout detection. We divide the high-resolution images into several structured layouts (e.g., tables, charts, legends) using PP-DocLayout plus-L (Cui et al., 2025). For map images, we review

the predictions, correct erroneous regions, and supplement missing areas via manual annotation. (2) Information extraction. For each layout, we prompt Qwen2.5-VL-32B (Bai et al., 2025b) to produce a detailed caption for the layout and extract OCR text from it. In addition, we obtain global context by generating a caption and extracting the OCR text for the full image. (3) Automated question synthesis. For each image, we provide the layout set (with captions and OCR texts) and the global context to GPT-5 (OpenAI, 2025a) to generate five questions (with answers and explanations) that compose evidence from the provided layout regions. Note that we do not provide the full image to GPT-5, compelling it to focus on region-level details and encourages multi-hop composition. More details about the whole pre-annotation process can be found in Appendix B.2.

Human annotation. (1) Filtering and validation. Annotators start by discarding ill-posed or lowquality machine-generated QAs (e.g., those with factual inconsistencies, ambiguous prompts, or spurious multi-hop reasoning). For the retained QAs, annotators verify that the six-option set contains exactly one unambiguous correct answer and confirm that the explanation faithfully, step by step, justifies the choice. The annotators also ensures that the target layouts are relevant to answering the question; these layouts are either derived from the explanation or added via manual annotation.

- (2) Human-authored questions. For information-dense images, machine-generated QAs often contain logical errors, wrong answers, or missed visual details. These QAs are reworked or completely rewritten by the annotators, adhering to our design principles: requiring fine-grained detail retrieval and multi-hop reasoning. Each QA includes a detailed explanation to aid verification and have exactly one unambiguous correct choice among the six.

Difficulty filtering and secondary review. We evaluate all candidate items with three strong proprietary MLLMs, i.e., GPT-5-mini (OpenAI, 2025a), Gemini-2.5-Flash (Comanici et al., 2025) and Doubao-Seed-1.6 (Bytedance, 2025), using the same evaluation prompt. We discard any items solved by all three models to ensure difficulty. Subsequently, independent reviewers (distinct from the original annotators) conducts cross-verification: a final pass over the QAs and the explanations to confirm factual correctness, clarity, and formatting consistency. Finally, we confirm with experiments that attention to visual details is vital to good performance on O3-BENCH (see Appendix B.3).

- 4 INSIGHT-O3

In the previous section, we introduced O3-BENCH, a meticulously-crafted benchmark that require problem-solving systems to have both good reasoning and perception capabilities, as well as the ability to integrate them in a natural, synergetic manner. Recent approaches towards such systems mostly build upon a single MLLM agent which handles both reasoning and perception workloads within a single context window (Su et al., 2025a; Zheng et al., 2025; Lai et al., 2025). While this is reasonable for tasks primarily focusing on either reasoning or perception, the agent may struggle when the workloads are heavy and intertwined.

To address the issue, we propose INSIGHT-O3, a two-agent system that largely decouples the aforementioned burden by a visual reasoning agent (vReasoner) and a visual search agent (vSearcher). The former specializes in high-level, abstract reasoning (with some general image understanding), while the latter is mainly responsible for locating detailed visual information and presenting them to vReasoner. For instance, given a question, vReasoner first decomposes the question via reasoning, and, if needed, issues relevant image region descriptions to vSearcher; vSearcher then localizes the requested evidence (with help from tools like image cropping) and returns it for subsequent rounds until a final answer is produced. This process is illustrated in Figure 2(a). However, jointly training both agents in a system like INSIGHT-O3 is notoriously difficult—their objectives differ yet are highly interdependent, causing difficulties such as credit assignment across calls and non-stationary updates when both policies learn. Additionally, in our case, even the frontier open MLLMs, e.g., Qwen2.5-VL (Bai et al., 2025b), tend to produce overly concise replies (Lai et al., 2025).

To avoid overcomplication, we consider a simpler, more manageable setting in this paper. Specifically, we delegate higher-order reasoning at training time to a strong external model (e.g., GPT-5mini as vReasoner) and focus on training vSearcher to cooperate with the given vReasoner effectively. This separation helps simplify optimization and improve data efficiency. Our experiment shows that a well-trained vSearcher can significantly improve the performance of a wide range of vReasoners as a plug-and-play callable agent. Furthermore, the resulting system may help syn-

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

|Query|
|---|

vReasoner system prompt vSearcher system prompt

Full image Cropped image

Full image w/ boxed region

[Figure 34]

Feedback to the prev. crop

User

[Figure 35]

[Figure 36]

[Figure 37]

|Respond| |CoT1|<Desc1>| |
|---|---|---|---|---|
| | | | | |

|CoT2|
|---|

|<Desc2>| |
|---|---|
| | |

|CoT3|
|---|

|<Ans>| |
|---|---|
| | |

|GT|
|---|

[Figure 38]

[Figure 39]

vReasoner

[Figure 40]

[Figure 41]

[Figure 42]

|CoT1|<Box1>|
|---|---|

|CoT2|<Box2>|
|---|---|

|Locate|
|---|

|Locate|
|---|

Correct?

vSearcher

(a) In-loop sub-agent RL with region descriptions generated on-the-fly during training

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

|GT Box| |
|---|---|
| | |

|<Desc>|
|---|

|GT Box| |
|---|---|
| | |

User

User

[Figure 47]

[Figure 48]

[Figure 49]

|Describe| |CoT|<Desc>|
|---|---|---|---|
| | | | |

|CoT|<Box>| |
|---|---|---|
| | | |

|Locate|
|---|

###### IoU

vSearcher

vReasoner

(b.1) Pre-generating region descriptions (b.2) Out-of-loop sub-agent RL with pre-generated region descriptions

- Figure 2: Training pipeline. We use a hybrid RL algorithm to train vSearcher. (a) In the in-loop component, vReasoner generates visual search tasks on-the-fly during training as it tries to answer a user query. We use vReasoner’s feedback and final answer correctness as supervision (denoted by dashed arrows) for vSearcher. (b) In the out-of-loop component, we use pre-generated descriptions with ground-truth bounding boxes, allowing us to train vSearcher efficiently via IoU supervision.

thesize multi-turn supervised-finetuning (SFT) traces with interleaved reasoning and visual search, paving the way for a larger, potentially unified model. While prior works (Su et al., 2025a; Jiang et al., 2025a) have explored role-playing to construct similar SFT traces, the data quality is often not very high as the role-playing agents are only loosely coordinated. In this respect, our approach helps strength this coordination with reinforcement learning (RL), as detailed next.

- 4.1 TRAINING ALGORITHM

We propose a hybrid sub-agent RL algorithm that consists of an in-loop component and an out-ofloop component to train vSearcher (see Figure 2). In the out-of-loop component, we pre-generate region descriptions with predefined bounding boxes, allowing us to train vSearcher very efficiently via direct IoU supervision. In the in-loop component, we use real descriptions generated on-thefly during training by vReasoner. Compared with pre-generated tasks, these dynamically generated tasks are more aligned in nature with the tasks that vSearcher will see during inference time.

Reward design. For the out-of-loop RL, we use the following reward function for vSearcher:

r = I[ntool > 0] · (λformat · rformat + λIoU · rIoU), (1)

where ntool is the number of tool calls made by vSearcher, rformat ∈ {0,1} is the format reward, and rIoU ∈ (0,1) is the IoU reward (λformat and λIoU are weighting coefficients for the rewards). In Eq.(1), the IoU reward rIoU encourages vSearcher to propose an accurate region that matches the description. In addition, we encourage vSearcher to use image cropping1 at least once to verify that the returned region matches the given region description. For every predicted box b and the groundtruth box b∗, we define the IoU reward as rIoU = max{0,IoU(b,b∗)−α}/(1−α) where α ∈ (0,1) controls the reward threshold. Boxes with an IoU less than α are not rewarded.

For the in-loop RL component, we replace the IoU reward rIoU in Eq.(1) with a pseudo IoU reward rˆIoU ∈ {0,1}. We obtain rˆIoU by asking vReasoner to rate each vSearcher’s prediction, with the rating criterion being whether the prediction is relevant to the assigned task and can help vReasoner answer the user query. The rating s ∈ {0,1} is a binary score indicating if the prediction is helpful. However, as this rating is not always reliable (vReasoner may sometimes make mistakes), we further

1For simplicity, we only allow vSearcher to use the most essential image cropping tool. Our framework, however, does not impose such a constraint. It is easy to incorporate other kinds of tools for vSearcher to use.

incorporate outcome supervision as a safeguard. Let c ∈ {0,1} stand for whether the final answer of vReasoner is correct. The pseudo IoU reward is defined as rˆIoU = I[s = c = 1].

Advantage estimation. We follow GRPO (Shao et al., 2024b) to estimate the advantages for the out-of-loop RL component. As for the in-loop component, we normalize the rewards with respect to the global mean and standard deviation instead of the group mean and standard deviation since the concept of “group” no longer exists for the dynamically generated tasks. Formally, the advantage of an output token ot at time step t is computed as Aˆt = [r − mean(r)]/std(r), where for the out-of-loop component, r = {ri}Gi=1 with G being the group size; and for the in-loop component, r = {ri}Ni=1 with N being the total number of visual search tasks generated on-the-fly by vReasoner.

Objective function. The objective function we use is based on GRPO (Shao et al., 2024b), with some modifications (e.g., global advantage estimations) to incorporate the in-loop RL component. Given a policy model πθ, the old policy πθ

, and a reference policy πref, the objective function for a batch of M vSearcher outputs (including both in-loop and out-of-loop ones) is defined as

old

|oi|

M

1 M

1 |oi|

min γt(θ)Aˆt,clip(γt(θ),1 − ϵ,1 + ϵ)Aˆt − βDKL[πθ∥πref] , (2)

J(θ) =

t=1

i=1

where γt(θ) = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t) is the importance ratio of the output token oi,t given a query q and all previous output tokens oi,<t including tool-response tokens. During training, we mask the loss for tool-response tokens as they are not generated by the policy model.

- 4.2 TRAINING DATA CONSTRUCTION

As high-resolution, information-dense images with challenging questions are difficult to collect, we construct training data by synthesizing collages (for in-loop RL) and generating pseudo visual search targets (for out-of-loop RL). The source data are mostly from existing VQA training datasets, which follow a largely different distribution from evaluation benchmarks we consider in our experiments.

In-loop RL data. A key criterion for in-loop RL data is that they must be difficult enough to incentivize visual search; otherwise vReasoner could simply answer on its own and vSearcher would receive no reward. To raise search difficulty and ensure meaningful credit assignment, we build image collages by stitching multiple low-to-medium-resolution images into a canvas. We construct collages from a filtered combination of Visual CoT (Shao et al., 2024a) and the V∗ training data (Wu & Xie, 2024), where each item provides a QA pair and a target bounding box. For each collage, we choose one target image (carrying the QA) and add several filler images as distractors. After difficulty filtering, we obtain 15,303 hard problems that vReasoner must rely on vSearcher to solve reliably. For more construction details and visualizations of the data, see Appendix C.1.

Out-of-loop RL data. We use InfographicVQA (Mathew et al., 2022) as the image source of the out-of-loop RL data. Most InfographicVQA images have high information density and feature more organic and diverse layouts than collages. We detect layout components in the source images with PP-DocLayout plus-L (Cui et al., 2025). The candidate layout boxes are filtered and further processed, resulting in 10,186 high-quality layout boxes. We then use GPT-5-nano to generate concise, high-level region descriptions for each box, as illustrated in Figure 2(b.1). Through prompting, we make GPT-5-nano mimic the style it would use when invoking vSearcher in the in-loop setting. This process yields a set of (image, region description, bbox) which enables the out-of-loop RL. More construction details and visualizations are provided in the Appendix C.2.

- 5 EXPERIMENT

In our main experiments, we train Qwen2.5-VL-7B-Instruct (Bai et al., 2025b) as vSearcher under GPT-5-mini-2025-08-07 (OpenAI, 2025a) as vReasoner for balanced efficiency and reasoning capability. The resulting vSearcher, named InSight-o3-vS, is evaluated under various vReasoners including Gemini-2.5-Flash (Comanici et al., 2025). For comparison, we evaluate these vReasoners normally as standalone models and as vReasoner with the untrained Qwen2.5-VL-7B-Instruct as

- Table 1: Performance comparison with frontier models/systems. All models/systems are evaluated under their default configurations unless specified otherwise. Performance of open models are mostly cited from the literature (Wu & Xie, 2024; Zheng et al., 2025; Wang et al., 2025a; Lai

et al., 2025). Other results are averaged over 3 trials, except MME-RWLite (single-trial). Small-size numbers indicate performance gaps between vReasoners w/ and w/o access to vSearchers. For more comprehensive benchmark results on O3-BENCH, see Appendix B.6.

Model/System V⋆-Bench HR-Bench4K Tree-Bench VProbeHard MME-RWLite O3-Bench Average

LLaVA-OV-7B 70.9 62.0 37.3 13.4 48.5 20.2 42.1 InternVL3.5-8B 64.0 64.5 40.5 11.0 48.0 24.3 42.1 Qwen2.5-VL-7B 75.5 68.2 37.0 23.9 46.7 27.4 46.5 Qwen3-VL-8B 86.4 78.9 48.3 31.6 53.0 43.6 57.0 Qwen3-VL-32B 86.0 81.1 48.2 28.6 54.2 60.4 59.8

Pixel Reasoner 86.3 74.0 28.8 28.8 - - DeepEyes 83.3 73.2 37.5 35.1 - 27.0 Mini-o3 88.2 77.5 - 48.0 - 29.1 OpenAI o3 76.4 74.3 52.3 23.6 55.2 40.8 54.0 GPT-4o 68.6 65.1 47.4 26.4 51.2 28.0 47.8

- + Qwen2.5-VL-7B 75.2 +6.6 69.7 +4.6 45.9 −1.6 15.4 −11.0 44.6 −6.6 29.5 +1.5 46.7 −1.1

+ InSight-o3-vS 80.4 +11.8 76.2 +11.1 49.5 +2.1 25.5 −1.1 50.1 −1.1 36.4 +8.4 53.0 +5.2 GPT-5-nano 64.0 60.6 45.4 21.7 47.7 26.5 44.3

+ Qwen2.5-VL-7B 70.1 +6.1 67.3 +6.7 45.7 +0.3 18.2 −3.5 44.9 −2.8 25.3 −1.2 45.3 +1.0 + InSight-o3-vS 75.1 +11.1 72.3 +11.7 47.7 +2.3 31.4 +9.7 48.4 +0.7 34.6 +8.1 51.6 +7.3

GPT-5-mini 73.8 72.0 54.6 26.4 56.1 39.0 53.7

+ Qwen2.5-VL-7B 80.6 +6.8 83.2 +11.2 53.1 −1.5 37.7 +11.3 58.1 +2.0 47.5 +8.5 55.4 +1.7

- + InSight-o3-vS 86.9 +13.1 86.7 +14.7 54.1 −0.5 41.2 +14.6 59.0 +2.9 61.5 +22.5 64.9 +11.2

+ InSight-o3-vS† 86.2 +12.4 85.7 +13.7 55.0 +0.4 39.6 +13.2 58.4 +2.3 59.9 +20.9 64.1 +10.4 Gemini-2.5-Flash# 72.8 75.0 48.9 17.9 55.6 49.8 53.4

+ Qwen2.5-VL-7B 76.3 +3.5 76.7 +1.7 51.3 +2.4 16.7 −1.2 50.9 −4.7 47.9 −1.9 53.3 −0.1 + InSight-o3-vS 80.8 +8.0 80.2 +5.2 52.1 +3.2 19.8 +1.9 55.1 −0.5 58.0 +8.2 57.7 +4.3 + InSight-o3-vS† 85.5 +12.7 82.7 +7.7 52.6 +3.7 26.4 +8.5 56.1 +0.5 61.1 +11.3 60.7 +7.3

Gemini-2.5-Flash 80.1 83.5 49.9 39.6 56.5 60.4 61.7

+ Qwen2.5-VL-7B 80.9 +0.8 79.0 −4.5 49.1 +0.8 31.4 −8.2 52.0 −4.5 55.6 −4.8 58.0 −3.7

- + InSight-o3-vS 87.6 +7.5 82.3 −1.2 50.1 +0.2 36.2 −3.4 56.3 −0.2 69.7 +9.3 63.7 +2.0

+ InSight-o3-vS† 88.3 +8.2 83.0 −0.5 53.6 +3.7 38.3 −1.3 56.4 −0.1 68.3 +7.9 64.7 +3.0

† Trained with Gemini-2.5-Flash as vReasoner. # Image-size constraint set to 1280×1280px, roughly the maximum supported size for OpenAI models/systems via API.

vSearcher. We use the default configuration for proprietary models/systems (per official API) except from setting image detail to high2. More implementation details can be found in Appendix D.1.

Evaluation datasets. We evaluate a range of open and proprietary models/systems on the following benchmarks: (1) Natural-image benchmarks: V⋆-Bench (Wu & Xie, 2024), Tree-Bench (Wang et al., 2025a), and VisualProbe-Hard (Lai et al., 2025). (2) Mixed benchmarks: HR-Bench (Wang et al., 2025f) and MME-RealWorld (Zhang et al., 2024b). For efficient evaluation, we use the lite version of MME-RealWorld, which has 1,919 questions, still much heavier than the other benchmarks. (3) Our O3-BENCH. More information about the benchmarks can be found in Appendix G.

- 5.1 MAIN RESULTS

Cross-domain performance improvement for frontier models. As shown in Table 1, INSIGHTO3 significantly improves frontier models such as GPT-5-mini and Gemini-2.5-Flash on most benchmarks. On average, the performance of GPT-5-mini has improved by 20.9% (relatively) with the help of our vSearcher (InSight-o3-vS). In particular, the accuracy of GPT-5-mini on O3-BENCH has improved from 39.0% to 61.5%. Meanwhile, INSIGHT-O3 also significantly outperforms their pre-RL counterparts, i.e., vReasoner + Qwen2.5-VL-7B, across all the benchmarks. The results suggest that InSight-o3-vS is able to generalize out-of-distribution across various domains since its training data distribution is distinct from the evaluation data distributions. To gain more insight on how INSIGHT-O3 improves the baselines, see Appendix D.3 for a comparative analysis.

2When image detail is high, OpenAI scales down oversize images to ∼1280×1280px, as per OpenAI API (https://platform.openai.com/docs/guides/images-vision#calculating-costs). This is the maximum supported image resolution for OpenAI models/systems via API. Gemini-2.5-Flash API does not impose such a constraint, so we use a much larger, 3500×3500px budget, which is ample for the task.

##### Table 2: Performance of Gemini-2.5-Flash (+ InSight-o3-vS†) under different maximum train-

ing/test image resolutions. All results are averaged over 3 trials, except for MME-RWLite (which is based on a single trial). Small-size numbers indicate performance gaps between settings with and without vSearcher. † Trained with Gemini-2.5-Flash as vReasoner.

Train res. Test res. V⋆-Bench HR-Bench4K Tree-Bench VProbeHard MME-RWLite O3-Bench Average

- 12802 72.8 75.0 48.9 17.9 55.6 49.8 53.3

12802 12802 85.5 +12.7 82.7 +7.7 52.6 +3.7 26.4 +8.5 56.1 +0.5 61.1 +11.3 60.7 +7.4 35002 12802 85.3 +12.5 81.3 +6.3 53.1 +4.2 22.6 +4.7 55.1 −0.5 58.8 +9.0 59.4 +6.1

- 35002 80.1 83.5 49.9 39.6 56.5 60.4 61.7

12802 35002 87.8 +7.7 84.3 +0.8 52.1 +2.2 39.6 +0.0 56.4 −0.1 67.8 +7.4 64.7 +3.0 35002 35002 88.3 +8.2 83.0 −0.5 53.6 +3.7 38.3 −1.3 56.4 −0.1 68.3 +7.9 64.7 +3.0

###### Table 3: Ablation study on reward design and advantage estimation. All results are averaged over 3 trials. Small-size numbers indicate performance changes w.r.t. the proposed setting.

Setting V⋆-B. VPHard O3-B. Avg. Proposed 86.9 41.2 61.5 63.2 w/o tool cond. 86.4 −0.5 39.3 −1.9 60.6 −0.9 62.1 −1.1 w/o feedback 86.5 −0.4 37.1 −4.1 58.1 −3.4 60.6 −2.6 w/o outcome 86.9 +0.0 38.7 −2.5 60.9 −0.6 62.2 −1.0 w/o GN 87.3 +0.4 36.8 −4.4 61.3 −0.2 61.8 −1.4

Table 4: Sensitivity analysis w.r.t. max. input resolution of vSearcher. “#vSearch” is the number of vSearcher calls made by vReasoner per QA.

Max. pixels V⋆-B. O3-B. #vSearch

- 0.8M 85.3 56.0 2.82
- 1.6M 86.7 60.5 2.75 3.2M 89.4 62.3 2.69 6.4M 86.4 61.9 2.66

12.8M 86.9 61.5 2.58

Generalization under different vReasoners. We observe that InSight-o3-vS, which was trained as a sub-agent under GPT-5-mini, generalizes under other vReasoner models as well. As shown in Table 1, InSight-o3-vS improves the performance of a much smaller model, GPT-5-nano, from 21.7% to 31.4% on VisualProbe-Hard, from 26.5% to 34.6% on O3-BENCH, and from 44.3% to 51.6% overall. Under Gemini-2.5-Flash (a different model family), the advantage remains significant, showing about 7–10% lead over the baselines on V⋆-Bench and O3-BENCH. We have also explored training InSight-o3-vS under Gemini-2.5-Flash instead of GPT-5-mini, and observed similar generalization (see “+ InSight-o3-vS†” rows in Table 1). In few cases where InSight-o3-vS fails to improve the performance of the vReasoner, e.g., GPT-4o and Gemini-2.5-Flash on VisualProbeHard, we see a sharp decrease in performance as we allow these models to call Qwen2.5-VL-7B. This suggests that these models are relatively weak at tool calling and multi-turn reasoning. In Appendix D.4, we present typical failure cases of INSIGHT-O3; we find that even GPT-5-mini (despite the good performance) still makes a lot of mistakes.

Performance gaps on O3-BENCH. Interestingly, on O3-BENCH, we observe that Gemini-2.5Flash has a huge edge over GPT-5-mini (when they have no access to vSearcher), but on the other benchmarks, the edge is not so prominent—Gemini-2.5-Flash is even slightly worse than GPT-5mini on Tree-Bench. This suggests that O3-BENCH is indeed quite different from the other benchmarks, and Gemini-2.5-Flash is particularly good at solving the kind of tasks in O3-BENCH on its own. Notably, with InSight-o3-vS, GPT-5-mini is able to drastically reduce the gap (from 21.4% to 8.2%) with Gemini-2.5-Flash, demonstrating the importance of thinking with images for addressing O3-BENCH, and also highlighting the effectiveness of our approach.

Effect of input image resolution. Comparing the results of Gemini-2.5-Flash in Table 1 under different maximum input image resolutions, we see that a much higher resolution offers clear advantages. However, the improvement brought by vSearcher is less when vReasoner can see clearer. In addition, we find that training under one resolution and evaluating under another seems to have little impact on the performance (see Table 2). Meanwhile, the input image resolution for vSearcher has less impact. Table 4 shows the performance of GPT-5-mini + InSight-o3-vS under varying maximum image resolution of InSight-o3-vS during evaluation. We can see that InSight-o3-vS is not sensitive to the resolution, maintaining decent performance on V⋆-Bench and O3-BENCH even when the maximum image resolution is only 0.8M (25% of that during training). When the resolution is low, the average number of vSearcher calls is relatively high. This is expected as low-resolution images often obscure fine details, making it harder for vSearcher to locate the targets.

vSearcher reward∗

vSearcher IoU

vReasoner accuracy

# of vReasoner calls

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

2.2

0.60

0.4

0.60

0.45

0.3

2.0

proposed

0.45

0.30

w/o feedback w/o outcome w/o GN

0.2

1.8

0.15

0.30

0.1

0 50 100 150 Step

0 50 100 150 Step

0 50 100 150 Step

0 50 100 150 Step

- Figure 3: Training dynamics of INSIGHT-O3. The rightmost chart, “# of vReasoner calls”, shows the average number of times vReasoner calls vSearcher per QA. ∗ For fair comparison, the reward curves are plotted under the same setting (“w/o feedback”) for all the settings.

- 5.2 ABLATION STUDY

Hybrid RL training. Table 5 shows the results of ablating the in-loop and the out-of-loop sub-agent RL components. Without the in-loop RL component, training is much faster (80%+ reduction in time per training step) but the final performance is worse on average. Dropping the out-of-loop RL component also hurts the performance; moreover, the training time increases due to more in-loop training. As mentioned in Section 4.2, the two RL components use different training data. The better performance of hybrid RL training can be partly explained by the combined use of the training data. Another contributing factor is the combination of two different sources of supervision (high-level vReasoner feedback and low-level IoU with groundtruth boxes). Overall, combining the two components leads to the best result.

Table 5: Ablation study on hybrid RL training. “I.” and “O.” stand for the in-loop and outof-loop RL components, respectively. “T/step” is the average time per training step. All results are averaged over 3 trials. Small-size numbers indicate performance changes w.r.t. the untrained baselines.

I. O. V⋆-B. VPHard O3-B. T/step

70.1 18.2 25.3 ✓ 74.9 +4.8 23.9 +5.7 27.9 +2.6 846s ✓ 73.7 +3.6 25.1 +6.9 31.5 +6.2 130s ✓ ✓ 74.5 +4.4 27.4 +9.2 32.4 +7.1 693s

GPT-5-nano

80.6 37.7 47.5 ✓ 86.4 +5.8 39.0 +1.3 59.6 +12.1 1223s ✓ 84.8 +4.2 41.2 +3.5 58.8 +11.3 105s ✓ ✓ 86.9 +6.3 41.2 +3.5 61.5 +14.0 941s

GPT-5-mini

Reward design and advantage estimation. In Table 3, we compare the setting we proposed in Section 4.1 on GPT-5-mini + InSight-o3-vS with the following ablated variants: “w/o tool cond.” drops the tool condition in the reward function; “w/o feedback” removes vReasoner feedback, only using outcome supervision for pseudo IoU reward; “w/o outcome” is the opposite of “w/o feedback”; and “w/o GN” drops the global normalization for advantage estimation. The originally proposed setting outperforms all the variants with a small average lead on the three benchmarks. The training dynamics under these settings are shown in Figure 3. As vSearcher learns to better locate the regions described by vReasoner, we observe that both the out-of-loop localization IoU and the in-loop vReasoner accuracy improve. The non-monotonic “# of vReasoner calls” shows two RL phases of INSIGHT-O3: vSearcher first learns to obey the formatting instructions, and then learns to localize more accurately (so vReasoner could solve the same problem with less vSearcher calls).

Although we encourage vSearcher to use the image-cropping tool, we find the average tool call count often ends up close to 1. There are two underlying reasons for this behavior. First, as mentioned by Zheng et al. (2025); Lai et al. (2025), Qwen2.5-VL-7B-Instruct is often reluctant to call the tool, and does not seem to know how to use the tool properly. Second, vReasoner usually describe a rough region that is not very hard for vSearcher to locate, so the tool has little utility for vSearcher.

- 6 CONCLUSION

In this work, we introduced O3-BENCH, a high-information-density benchmark that jointly evaluates visual localization and multi-hop reasoning. To advance the research on this challenging benchmark, we proposed INSIGHT-O3, a multi-agent framework that decomposes the “think with images” workflow into high-level reasoning (vReasoner) and visual search (vSearcher). We focus on the training of vSearcher via reinforcement learning to seamlessly cooperate with vReasoner. The specialized InSight-o3-vS can be used as a “plug-and-play” component for existing multimodal foundation models and helps significantly improve the performance of frontier models.

REFERENCES

Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 54

Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 126–135, 2017. 53

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a. 29, 43, 47, 54

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b. 3, 5, 7, 19, 21, 29, 43, 47

Bytedance. Doubao-Seed-1.6. https://console.volcengine.com/ark/region: ark+cn-beijing/model/detail?Id=doubao-seed-1-6, 2025. 5, 29

Lluis Castrejon, Thomas Mensink, Howard Zhou, Vittorio Ferrari, Andre Araujo, and Jasper Uijlings. Hammr: Hierarchical multimodal react agents for generic vqa. In NeurIPS 2024 Workshop on Compositional Learning: Perspectives, Methods, and Paths Forward, 2024. 2, 19

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025a. 19

Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025b. 19

Zihui Cheng, Qiguang Chen, Xiao Xu, Jiaqi Wang, Weiyun Wang, Hao Fei, Yidong Wang, Alex Jinpeng Wang, Zhi Chen, Wanxiang Che, et al. Visual thoughts: A unified perspective of understanding multimodal chain-of-thought. arXiv preprint arXiv:2505.15510, 2025. 20

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3, 5, 7, 19, 29

Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025. URL https://arxiv.org/abs/2507.05595. 4, 7, 21

Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, et al. Multi-agent collaboration via evolving orchestration. arXiv preprint arXiv:2505.19591, 2025. 19

Peter Dayan and Geoffrey E Hinton. Feudal reinforcement learning. Advances in neural information processing systems, 5, 1992. 2

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement, 2025. URL https://arxiv.org/abs/2503.17352. 19

Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491,

2025. 19

Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879, 2025. 20

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 19

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024. 3, 19

Google. Gemini 3. https://blog.google/products/gemini/gemini-3/, 2025. 29 Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa

matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017. 3, 19

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a. 3, 19, 47

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025b. 19

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14953–14962, 2023. 3, 20

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3608–3617, 2018. 19

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. In Forty-second International Conference on Machine Learning, 2025. 1

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2023. 2, 19

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. URL https://arxiv.org/abs/2507.01006. 19, 29

Zhipeng Hou, Junyi Tang, and Yipeng Wang. Halo: Hierarchical autonomous logic-oriented orchestration for multi-agent llm systems. arXiv preprint arXiv:2505.13516, 2025. 19

Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models. arXiv preprint arXiv:2501.03262, 2025. 19

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024. 20

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025. 19

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700–6709, 2019. 19

Xinyu Jia, Chuang Zhu, Minzhen Li, Wenqi Tang, and Wenli Zhou. Llvip: A visible-infrared paired dataset for low-light vision. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3496–3504, 2021. 53

Chaoya Jiang, Yongrui Heng, Wei Ye, Han Yang, Haiyang Xu, Ming Yan, Ji Zhang, Fei Huang, and Shikun Zhang. Vlm-r3: Region recognition, reasoning, and refinement for enhanced multimodal chain-of-thought. arXiv preprint arXiv:2505.16192, 2025a. 6, 20

Qing Jiang, Xingyu Chen, Zhaoyang Zeng, Junzhi Yu, and Lei Zhang. Rex-thinker: Grounded object referring via chain-of-thought reasoning. arXiv preprint arXiv:2506.04034, 2025b. 20

Fucai Ke, Zhixi Cai, Simindokht Jahangard, Weiqing Wang, Pari Delir Haghighi, and Hamid Rezatofighi. Hydra: A hyper agent for dynamic compositional visual reasoning. In European Conference on Computer Vision, pp. 132–149. Springer, 2024. 3, 20

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023. 52

Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025. 1, 2, 3, 4, 5, 8, 10, 19, 20, 52

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 29

Geng Li, Jinglin Xu, Yunzhen Zhao, and Yuxin Peng. Dyfo: A training-free dynamic focus visual search for enhancing lmms in fine-grained visual understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 9098–9108, 2025. 1, 20

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for” mind” exploration of large language model society. Advances in Neural Information Processing Systems, 36:51991–52008, 2023a. 2, 19

Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, et al. Coda: A real-world road corner case dataset for object detection in autonomous driving. In European conference on computer vision, pp. 406– 423. Springer, 2022. 53

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023b. 19

Jiaying Liu, Dong Liu, Wenhan Yang, Sifeng Xia, Xiaoshuai Zhang, and Yuanying Dai. A comprehensive benchmark for single image compression artifact reduction. IEEE Transactions on image processing, 29:7845–7860, 2020. 53

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 3, 19

Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual agentic reinforcement fine-tuning. arXiv preprint arXiv:2505.14246, 2025. 20

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 3, 19, 54

Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1697–1706, 2022. 7

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng

- Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 19

Minheng Ni, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Kevin Lin, Wangmeng Zuo, and Lijuan Wang. Point-rft: Improving multimodal reasoning with visually grounded reinforcement finetuning. arXiv preprint arXiv:2505.19702, 2025. 20

- OpenAI. GPT-4o. https://openai.com/index/hello-gpt-4o/, 2024. 29
- OpenAI. GPT-5. https://openai.com/index/introducing-gpt-5/, 2025a. 3, 5, 7, 19, 21, 29, 33, 45, 46, 52

OpenAI. GPT-5.2. https://openai.com/index/introducing-gpt-5-2/, 2025b. 29 OpenAI. OpenAI o3 and o4-mini. https://openai.com/index/introducing-o3-

and-o4-mini/, 2025c. 1, 3, 19, 20, 29

Yi Peng, Peiyu Wang, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, et al. Skywork r1v: Pioneering multimodal reasoning with chain-ofthought. arXiv preprint arXiv:2504.05599, 2025. 19

Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. Cogcom: Train large vision-language models diving into details through chain of manipulations. arXiv preprint arXiv:2402.04236, 2024. 20

Yusu Qian, Hanrong Ye, Jean-Philippe Fauconnier, Peter Grasch, Yinfei Yang, and Zhe Gan. Miabench: Towards better instruction following evaluation of multimodal llms, 2024. URL https: //arxiv.org/abs/2407.01509. 54

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 19

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents, 2024. URL https://arxiv.org/abs/2405. 14573. 54

Enna Sachdeva, Nakul Agarwal, Suhas Chundi, Sean Roelofs, Jiachen Li, Mykel Kochenderfer, Chiho Choi, and Behzad Dariush. Rank2tell: A multimodal driving dataset for joint importance ranking and reasoning. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 7513–7522, 2024. 53

Tanik Saikh, Tirthankar Ghosal, Amish Mittal, Asif Ekbal, and Pushpak Bhattacharyya. Scienceqa: A novel resource for question answering on scholarly articles. International Journal on Digital Libraries, 23(3):289–301, 2022. 3, 19

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3, 19

- Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Unleashing chain-of-thought reasoning in multi-modal language models. CoRR, 2024a. 7, 20, 28

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024b. 3, 7, 19

Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. arXiv preprint arXiv:2411.16044, 2024. 1

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025a. 19

Wei Shen, Jiangbo Pei, Yi Peng, Xuchen Song, Yang Liu, Jian Peng, Haofeng Sun, Yunzhuo Hao, Peiyu Wang, Jianhao Zhang, et al. Skywork-r1v3 technical report. arXiv preprint arXiv:2507.06167, 2025b. 19

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180, 2023. 2, 19, 20

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024. 33

Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025a. 1, 3, 5, 6, 20

Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025b. 20

Xian Sun, Peijin Wang, Zhiyuan Yan, Feng Xu, Ruiping Wang, Wenhui Diao, Jin Chen, Jihao Li, Yingchao Feng, Tao Xu, et al. Fair1m: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 184:116–130, 2022. 53

D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11888–11898, 2023. 3, 20

Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, et al. Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. arXiv preprint arXiv:2507.07999, 2025a. 1, 8, 19, 52

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vlrethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025b. 19

Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. arXiv preprint arXiv:2506.11991, 2025c. 1, 20

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024. 19, 54

Peiyu Wang, Yichen Wei, Yi Peng, Xiaokun Wang, Weijie Qiu, Wei Shen, Tianyidan Xie, Jiangbo Pei, Jianhao Zhang, Yunzhuo Hao, et al. Skywork r1v2: Multimodal hybrid reinforcement learning for reasoning. arXiv preprint arXiv:2504.16656, 2025d. 19

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025e. 3, 19, 29, 47

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial

- Intelligence, volume 39, pp. 7907–7915, 2025f. 8, 53

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial

- Intelligence, volume 39, pp. 7907–7915, 2025g. 1, 19

Ye Wang, Qianglong Chen, Zejun Li, Siyuan Wang, Shijie Guo, Zhirui Zhang, and Zhongyu Wei. Simple o3: Towards interleaved vision-language reasoning. arXiv preprint arXiv:2508.12109, 2025h. 20

Yana Wei, Liang Zhao, Jianjian Sun, Kangheng Lin, Jisheng Yin, Jingcheng Hu, Yinmin Zhang, En Yu, Haoran Lv, Zejia Weng, et al. Open vision reasoner: Transferring linguistic cognitive behavior for visual reasoning. arXiv preprint arXiv:2507.05255, 2025. 19

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094, 2024. 1, 2, 3, 4, 7, 8, 19, 20, 28, 52

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multiagent conversations. In First Conference on Language Modeling, 2024. 19

LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025. URL https://arxiv.org/abs/ 2506.03569. 19

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. 54

Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025a. 3, 19

Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025b. 19

Qinhong Yang, Dongdong Chen, Zhentao Tan, Qiankun Liu, Qi Chu, Jianmin Bao, Lu Yuan, Gang Hua, and Nenghai Yu. Hq-50k: A large-scale, high-quality dataset for image restoration. arXiv preprint arXiv:2306.05390, 2023. 53

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025c. 3, 19

Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, et al. Mme-reasoning: A comprehensive benchmark for logical reasoning in mllms. arXiv preprint arXiv:2505.21327, 2025. 1

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024a. 1, 3, 19

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b. 3, 19

Andy Zeng, Maria Attarian, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S Ryoo, Vikas Sindhwani, Johnny Lee, et al. Socratic models: Composing zero-shot multimodal reasoning with language. In The Eleventh International Conference on Learning Representations, 2023. 2, 19

Kaihao Zhang, Dongxu Li, Wenhan Luo, Wenqi Ren, Bj¨orn Stenger, Wei Liu, Hongdong Li, and Ming-Hsuan Yang. Benchmarking ultra-high-definition image super-resolution. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 14769–14778, 2021. 53

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024a. 19

Wentao Zhang, Ce Cui, Yilei Zhao, Rui Hu, Yang Liu, Yahui Zhou, and Bo An. Agentorchestra: A hierarchical multi-agent framework for general-purpose task solving. arXiv e-prints, pp. arXiv– 2506, 2025a. 19

Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025b. 1, 20

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024b. 2, 3, 4, 8, 19, 53

Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025c. 20

Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025. 20

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 1, 3, 5, 8, 10, 20, 51

Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308, 2025. 54

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025a. 19

Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, et al. Active-o3: Empowering multimodal large language models with active perception via grpo. arXiv preprint arXiv:2505.21457, 2025b. 1, 20

###### Pengfei Zhu, Longyin Wen, Dawei Du, Xiao Bian, Heng Fan, Qinghua Hu, and Haibin Ling. Detection and tracking meet drones challenge. IEEE transactions on pattern analysis and machine intelligence, 44(11):7380–7399, 2021. 53

- A ADDITIONAL DISCUSSION ON RELATED WORK

- A.1 MULTIMODAL BENCHMARKS

Classical multimodal benchmarks (Goyal et al., 2017; Hudson & Manning, 2019; Gurari et al., 2018; Saikh et al., 2022; Li et al., 2023b; Fu et al., 2023; Liu et al., 2023; Ge et al., 2024) primarily focus on coarse image-level understanding or target at salient-object attributes, on which current multimodal models (Bai et al., 2025b; Wang et al., 2025e; Yang et al., 2025a) show near-saturated performance. With growing attention to multimodal reasoning, more challenging benchmarks have emerged, which could be categorized as two groups. (1) Cognition-centric benchmarks. STEM (science, technology, engineering, and mathematics) benchmarks (Lu et al., 2023; Wang et al., 2024; Zhang et al., 2024a; Yue et al., 2024a;b) evaluate the model’s multi-step reasoning, integration of world knowledge, and complex calculations to solve scientific problems, whereas the accompanying images are generally straightforward to interpret. (2) Perception-centric benchmarks. These benchmarks (Wu & Xie, 2024; Wang et al., 2025g; Zhang et al., 2024b; Lai et al., 2025) require fine-grained perception in high-resolution images and strong OCR recognition on text-rich scenes. Nevertheless, many questions become routine once the model precisely localizes the target region, allowing for single-glance solutions. Though the recent TreeBench (Wang et al., 2025a) evaluates second-order reasoning over object spatial transformations, depth ordering, and etc, it still centers on a single region in natural images, leaving cross-region evidence aggregation largely underexplored. With the emergence of the “think with images” paradigm (OpenAI, 2025c), we argue that a well-designed benchmarks should evaluate the joint perceptual and cognitive skills. Our proposed O3-BENCH fills the research gap by meticulously constructing hard questions on high-information density images (e.g., composite graphs, maps), therefore requiring models to gather information from multiple, spatially distinct regions and to perform complex, interleaved reasoning.

- A.2 MULTIMODAL REASONING MODELS

Reinforcement learning (RL) (Schulman et al., 2017; Rafailov et al., 2023; Hu et al., 2025) has long been used to align the response of large language models (LLMs) and multimodal LLMs (MLLMs) with human preferences. Recently, DeepSeek-R1 (Guo et al., 2025a) creatively applied group relative policy optimization (GRPO) (Shao et al., 2024b) to LLMs, estimating the mean and variance of advantages across response groups under a simple reward signal. This strategy reliably elicits behaviors such as planning, thinking, and self-reflection, enabling long chain-of-thought (CoT) reasoning and moving toward more general-purpose reasoning capabilities. Building on this success, several works (Huang et al., 2025; Yang et al., 2025c; Meng et al., 2025; Chen et al., 2025a; Wang et al., 2025b; Shen et al., 2025a; Chen et al., 2025b; Deng et al., 2025; Wei et al., 2025; Peng et al., 2025; Wang et al., 2025d; Shen et al., 2025b) explore cold-start initialization and GRPO-based RL training for multimodal models (e.g., Qwen2.5-VL (Bai et al., 2025b)) and report substantial gains on science- and math-oriented benchmarks. Concurrently, InternVL3.5 (Zhu et al., 2025a; Wang et al., 2025e) and Keye-VL1.5 (Yang et al., 2025b;a) further leverage cascaded, iterative RL stages to push the frontier of reasoning ability, achieving performance competitive with proprietary models (OpenAI, 2025a; Comanici et al., 2025). Nevertheless, current multimodal reasoning models (Xiaomi, 2025; Hong et al., 2025; Guo et al., 2025b; Du et al., 2025) still focus on text-centric reasoning, neglecting the distinctive demands of visual reasoning in multimodal scenarios.

- A.3 HIERARCHICAL AGENT FRAMEWORKS AND TOOL-USING MULTIMODAL AGENTS

Recent works have shown that hierarchical collaboration among specialized agents can significantly improve performance on complex tasks. Socratic Models (Zeng et al., 2023) and HuggingGPT (Shen et al., 2023) demonstrate early examples of LLM-based orchestration over expert models via language. More structured frameworks like CAMEL (Li et al., 2023a), MetaGPT (Hong et al., 2023), and HAMMR (Castrejon et al., 2024) explore role-based or modular specialization with coordinated task decomposition. Others, including AutoGen (Wu et al., 2024), HALO (Hou et al., 2025), Puppeteer (Dang et al., 2025), and AgentOrchestra (Zhang et al., 2025a), further extend this paradigm with explicit planning hierarchies, adaptive execution, and learned orchestration, consistently outperforming flat-agent baselines across diverse domains.

There are also recent works exploring equipping multimodal models with tool-use and programmatic reasoning to tackle complex visual tasks. VisProg (Visual Programming) (Gupta & Kembhavi, 2023) and ViperGPT (Sur´ıs et al., 2023) are two pioneering approaches that use large language models to generate and execute code for orchestrating vision modules. Beyond static program generation, other agents use LLMs as high-level controllers. For example, HuggingGPT (Shen et al., 2023) demonstrates an LLM (ChatGPT) orchestrating numerous specialized models (for vision, language, etc.) More recently, HYDRA (Ke et al., 2024) introduces a dynamic multi-stage framework for visual reasoning: it integrates an LLM-based planner and reasoner with a reinforcement learning–based controller that adapts the sequence of operations via feedback loops, yielding more reliable step-by-step reasoning. These tool-augmented systems highlight the power of combining learned vision-language models with external modules or code execution to improve flexibility and compositional reasoning. In contrast, our work (INSIGHT-O3) targets a complementary gap by introducing a dedicated visual search agent that can be invoked by reasoning agents to locate finegrained, conceptually described regions within images. This specialized capability, absent in prior tool-using frameworks, allows an INSIGHT-O3-enabled system to pinpoint relevant visual details based on free-form descriptions, thereby enhancing multimodal reasoning with more precise visual understanding.

- A.4 VISUAL SEARCH MODELS

Visual search is an important functionality in the multimodal domain, requiring the models to perform active perception over regions of interest (RoIs) for fine-grained visual understanding. Early approaches (Wu & Xie, 2024; Shao et al., 2024a; Qi et al., 2024; Hu et al., 2024; Li et al., 2025) rely on external tools or predefined workflows for region localization and use instruction tuning to trigger tool use. These models exhibit rigid output patterns and typically support only a single round of visual search, which limits their effectiveness in complex scenes. Recently, the milestone OpenAI o3 (OpenAI, 2025c) established the “think with images” paradigm, in which image manipulations (e.g., zooming, cropping) are internalized as intrinsic capabilities, enabling image–text interleaved reasoning. The community has rapidly turn the attention to the promising field. DeepEyes (Zheng et al., 2025) exploits the inherent grounding ability of MLLMs and incentivizes visual search via end-to-end reinforcement learning. Pixel-Reasoner (Su et al., 2025a) improves search accuracy by warm-start instruction tuning on synthesized data with error-induced self-correction trajectories. Mini-o3 (Lai et al., 2025) introduces an over-turn masking technique during RL to encourage multiturn interaction, markedly enhancing reasoning adaptability and diversity. Other lines of work (Zhao et al., 2025; Zhang et al., 2025c; Liu et al., 2025) resort to write codes for executing multiple image manipulations (e.g., cropping, rotation, enhancement), pointing to an open-ended toolkit for visual reasoning. Although effective, recent advanced methods (Zhu et al., 2025b; Fan et al., 2025; Zhang et al., 2025b; Su et al., 2025b; Ni et al., 2025; Cheng et al., 2025; Jiang et al., 2025a;b; Wang et al., 2025c;h) still largely prioritize locating a single region on natural images, which sidelines the model’s capacity for reasoning. This paper broadens the capability scope of visual search models by decoupling visual reasoning and visual search agents, allowing the receiving of any images and searching of multiple distinct regions.

- B ADDITIONAL INFORMATION ON O3-BENCH

- B.1 BENCHMARK STATISTICS

Layout distribution

Target layout distribution

chart map

chart

map

75

| |
|---|

#ofsamples

#ofsamples

20

50

10

25

0

0

10 20 # of layouts

2 4 6 # of target layouts

Figure 4: Distribution of layout numbers in O3-BENCH.

Image resolution distribution

10000

chart

map

Height

5000

5000 10000 Width

###### Figure 5: Resolution distribution.

We summarize the benchmark statistics from following three aspects. (1) Distribution of layouts. The benchmark features 8.7 layouts and 2.4 target layouts for each sample on average, indicating high information density and the need for multi-step reasoning. And the layout distribution by category is displayed in Figure 4. It can be seen that chart images typically exhibit a larger set of total layouts, whereas map images require more target layouts for reasoning. (2) Distribution of resolution. We collect high-resolution imagery in O3-BENCH. As shown in Figure 5, most images have side lengths in the 2K–5K range, while some images reach up to ∼10K pixels on the longer side, yielding high information density. On average, image height and width are 3,967 and 4,602 pixels, respectively. (3) Distribution of options. We randomly shuffle options A–E, ensuring an approximately uniform distribution of correct-answer positions. And a small portion (7.2%) of samples use option F (No Right Choice) as correct answer, which compels models to aggregate evidence across the entire image and determine that none of the other options is valid.

- B.2 DETAILS OF MACHINE PRE-ANNOTATION

- (1) Layout detection. We first divide the high-resolution images into several structured layouts (e.g., tables, charts, legends). We use PP-DocLayout plus-L (Cui et al., 2025) to detect layout bounding

boxes in the image and construct the set L = {li}mi=1, where li ∈ R4 denotes layout coordinates. For chart images, we directly use the detector outputs. For map images, we review the predictions,

correct erroneous regions, and supplement missing areas via manual annotation.

- (2) Information extraction. For each detected layout li in image I, we obtain the cropped image Il

i

according to its coordinates. We then prompt Qwen2.5-VL-32B (Bai et al., 2025b) to produce a detailed caption ci and extract OCR text oi for Il

i

, thereby forming the layout triplet τi = (Il

i

,ci,oi). And then we aggregate all region triplets into T = {τi}mi=1. In addition, we obtain global context by generating a caption and OCR text for the full image, denoted as G = (cg,og). The prompts we use are provided in Appendix E.1.

- (3) Automated question synthesis. We provide the layout set T and the global context G to GPT-

5 (OpenAI, 2025a) and explicitly prompt it to generate five questions that compose evidence from multiple regions. For each question, GPT-5 must produce six options (A–F) with exactly one correct answer, and option F is reserved for No Right Choice. It also need to supply a step-by-step explanation that interprets the reasoning chain. It is noted that we do not provide the full image to GPT-5, which compels the model to focus on region-level details and encourages multi-hop composition. The prompts we use are provided in Appendix E.2.

- B.3 MORE EXPERIMENTAL RESULTS FOR O3-BENCH

Table 6: Ablation on target layouts in O3-BENCH.

Model

O3-BENCH chart map overall

GPT-5-mini 34.4 43.2 39.0 + target layouts 74.2 61.5 67.5

Qwen2.5-VL-7B 30.9 24.4 27.4 + target layouts 39.3 31.9 35.4

We present additional results for O3-BENCH in this section. Because our annotation pipeline supplies the target layouts most relevant to each question, we can provide these region crops alongside the original full image at test time. We evaluate GPT-5-Mini (OpenAI, 2025a) and Qwen2.5-VL-7B (Bai et al., 2025b), with results shown in Table 6. Both models exhibit significant performance gains when given the additional target layouts, underscoring the need for models to actively locate task-critical regions and perform interleaved visual reasoning.

- B.4 VISUALIZATION OF O3-BENCH

In Figures 6–11, we present six representative visualizations (four map items and two chart items), showing how O3-BENCH couples high-resolution perception with multi-step reasoning. Each annotation includes: (i) the multiple-choice question and answer; (ii) highlighted target layouts that mark the regions consulted along the solution path; and (iii) a concise, ordered explanation that composes the evidence, which allows readers to verify the answer quickly.

###### Annotation

###### Split: Map

Q&A: What is the building opposite the main entrance of the School of Business Postgraduate Annexe?

[Figure 50]

✅ Islands Centre; F. None of above

A. MlnT Study; B. Commerce Building; C. Psychology, Darwin House; D. Philosophy, Union St East; E. Pacific

Target Layouts:

Explanation: Step 1. Locate the School of Business Postgraduate Annexe in the building list , identified as No. 26 at F7. Step 2. Find this building at F7 on the East Campus map . Step 3. Refer to the Map Legend to confirm that the triangle represents the Main Entrance to the Building. Observe that the main entrance of the School of Business Postgraduate Annexe faces Building No. 27. Step 4. Look up Building No. 27 in the building list , identified as the Commerce Building, thus the answer is B.

[Figure 51]

[Figure 52]

Closed views of key regions:

- Step 3 :

- Step 1 :

|[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

- Step 2 :

- Step 4 :

| |
|---|

|[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]|
|---|

- Figure 6: Example from O3-BENCH (Map-1). Each annotation comprises a six-choice QA and a brief explanation with highlighted target layouts for quick verification; additionally, we also provide step-wise close-ups (outside the annotation) to reveal the evidence chain in large images where fine details may be hard to see.

###### Annotation

###### Split: Map

Q&A: Which floor has the Prayer room, and near which zone is it located?

[Figure 61]

✅

A. 2F, South zone; B. 2F, North zone; C. 3F, North zone; D. 4F, North zone; E. 4F, West zone; F. None of above

Target Layouts:

Explanation: Step 1. Refer to the Legend to identify the symbol for the Prayer room. Step 2. Use the Floor Guide in the bottom-right corner to find that the Prayer room is located on 2F. Step 3. On the 2F map , locate the Prayer room to the right of the H&M store in the North zone. Select option B accordingly.

[Figure 62]

| |
|---|

[Figure 63]

Closed views of key regions: Step 3 :

Step 2 :

|[Figure 64]|
|---|

- Step 1 :

|[Figure 65]|
|---|

|[Figure 66]|
|---|

- Figure 7: Example from O3-BENCH (Map-2). Each annotation comprises a six-choice QA and a brief explanation with highlighted target layouts for quick verification; additionally, we also provide step-wise close-ups (outside the annotation) to reveal the evidence chain in large images where fine details may be hard to see.

###### Annotation

###### Split: Map

Q&A: Which restaurant has an ATM located next to it?

A. Burger Federation; B. Fulton Street Café; C. Wolfgang Express; D. Goose Island

Beer Company; E. Facades Bar; ✅ F. The Bronze Tap

[Figure 67]

Target Layouts:

Explanation: Step 1. Locate the ATM symbol in PASSENGER SERVICES . Step 2. Match the boarding gate numbers of the restaurants listed in the options using the restaurant list : Burger Federation (L20), Fulton Street Cafe (H14), Wolfgang Express (K12), Goose Island Beer Company (L10A), Facades Bar (K15), The Bronze Tap (K5). Step 3. Compare the gate locations on the terminal map and identify which gate has an ATM symbol nearby. Only K5 has an ATM symbol, so the answer is The Bronze Tap, F

|[Figure 68]<br><br>|
|---|

[Figure 69]

| |
|---|

Step 3 :

Closed views of key regions:

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

| |
|---|

- Step 2 : Step 1 :

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

[Figure 80]

[Figure 81]

[Figure 82]

- Figure 8: Example from O3-BENCH (Map-3). Each annotation comprises a six-choice QA and a brief explanation with highlighted target layouts for quick verification; additionally, we also provide step-wise close-ups (outside the annotation) to reveal the evidence chain in large images where fine details may be hard to see.

###### Annotation

###### Split: Map

Q&A: After having lunch at Brookfield Cafe, I want to take a bus to visit Boole Library. Which route should I take?

[Figure 83]

A. 201; B. 202; ✅ C. 205; D. 208; E. 209; F. 216

Target Layouts:

Explanation: Step 1. Locate Brookfield Cafe in the Catering list of ; it is building 13 in Grid C7. Step 2. Locate Boole Library in ; it is building 11 in Grid F6. Step 3. On the map , find that Route 205 bus from College Road below building 13 leads to the vicinity of building 11.",

[Figure 84]

[Figure 85]

Closed views of key regions:

Step 3 :

[Figure 86]

|[Figure 87]|
|---|

| |
|---|
| |
| |

Step 1 :

[Figure 88]

|[Figure 89]|
|---|

Step 2 :

|[Figure 90]|
|---|

[Figure 91]

- Figure 9: Example from O3-BENCH (Map-4). Each annotation comprises a six-choice QA and a brief explanation with highlighted target layouts for quick verification; additionally, we also provide step-wise close-ups (outside the annotation) to reveal the evidence chain in large images where fine details may be hard to see.

###### Annotation

###### Split: Chart

Q&A: If the Condo unit's Purchase Cost is depreciated evenly over its stated operation period, what percentage does this depreciation amount account for of its Avg. Annual Rental Fee (annualized based on the Year 10 Avg. Daily Rental Fee and 30 days/month)?

[Figure 92]

✅

A. 35.8%; B. 36.0%; C. 37.5%; D. 38.5%; E. 39.0% F. No right choice

Target Layouts:

Explanation: Step 1: From , Condo Purchase Costs = 150,000 USD. Step 2: From , depreciation period = 15 years. Therefore, Annual depreciation = 150,000 / 15 = 10,000 USD/year. Step 3: From , Condo Year 10 Avg. Daily Rental Fee = 114 USD/day. Step 4: Annual rental income at Year 10 rate with 30 days/month = 114 * 30 * 12 = 41,040 USD/year. Step 5: Percentage = (10,000 / 41,040) * 100 ≈ 24.37%. Step 6: 24.37%\nis not listed in options A–E, so the correct choice is F.

|[Figure 93]<br><br>|
|---|

[Figure 94]

Closed views of key regions: Step 3 :

Step 1 :

Step 2 :

|[Figure 95]|
|---|

|[Figure 96]|
|---|

| |
|---|

[Figure 97]

| |
|---|

[Figure 98]

|[Figure 99]|
|---|

- Figure 10: Example from O3-BENCH (Chart-1). Each annotation comprises a six-choice QA and a

###### Annotation

###### Split: Chart

Q&A: What is the LTV in the first year when Net Income exceeds 4,700,000 USD? A. 65.8%; B. 62.1%; C. 58.5%; D. 54.8%; E. 51.2%; F. No right choice

[Figure 100]

✅

Target Layouts:

Explanation: Step 1: From the Income Statement table , find first year Net Income > 4,700,000: 2028F

= 4,711,353 is the earliest that exceeds. Step 2: From Bank Ratios , LTV in 2028F is shown as 58.5%

[Figure 101]

| | | | |
|---|---|---|---|

[Figure 102]

Step 1 ：

Closed views of key regions:

|[Figure 103]|[Figure 104]|
|---|---|

Step 2 ：

|[Figure 105]|[Figure 106]|
|---|---|

- Figure 11: Example from O3-BENCH (Chart-2). Each annotation comprises a six-choice QA and a

- B.5 COMPARISON WITH MME-REALWORLD (CHART)

The chart images of O3-BENCH are mostly selected from the “Diagram & Table” subset of MMERealWorld. The original questions from MME-RealWorld are relatively simple, usually focusing on a single value in a chart, which do not require any kind of multi-hop reasoning. For example, the original question of MME-RealWorld for the chart we show in Figure 10 is:

What is the cost inflation rate in the General Settings section of the General Assumptions table?

In comparison, the new question in O3-BENCH is:

If the Condo unit’s Purchase Cost is depreciated evenly over its stated operation period, what percentage does this depreciation amount account for of its Avg. Annual Rental Fee (annualized based on the Year 10 Avg. Daily Rental Fee and 30 days/month)?

As illustrated in Figure 10, answering this question requires piecing together detailed information from three different tables through multi-step reasoning and arithmetic.

Overall, the questions of O3-BENCH (chart) are much more challenging than the MME-RealWorld counterparts. This can also be seen from the following statistics:

- • The average accuracy of GPT-5-mini on O3-BENCH (chart) is about 38.2%, whereas on MMERealWorld (chart), the accuracy is about 82.4%.
- • The average number of vSearch steps of InSight-o3 on O3-BENCH (chart) is about 3.1, whereas on MME-RealWorld (chart), the number is about 1.1.
- • The average multi-turn response length (including reasoning tokens) of GPT-5-mini vReasoner on O3-BENCH (chart) is about 1942.3 characters, whereas on MME-RealWorld (chart), the average response length is about 730.0 characters.

- B.6 FULL BENCHMARK RESULTS

Table 7 shows our full benchmark results of frontier multimodal models/systems on O3-BENCH. For OpenAI models/systems, oversize input images are resized to 1280×1280px (this is roughly the maximum supported size per OpenAI API, as mentioned in the main paper) and image detail is set to high. For other models/systems, oversize input images are resized to 3500×3500px (this translates to about 16K tokens/image for Qwen2.5-VL). All models/systems are given a 16K tokens/response budget (including reasoning tokens), which should be more than enough to solve the problems in O3-BENCH. Incomplete responses beyond this budget are considered wrong without checking. Open models are evaluated via self-hosted vLLM instances. Proprietary models are evaluated via official APIs. Our evaluation code can be found at https://github.com/m-Just/ InSight-o3.

- C TRAINING DATA CONSTRUCTION DETAILS

- C.1 IN-LOOP RL DATA

Our collage sources come from the training split of Visual CoT (Shao et al., 2024a) and V∗ (Wu & Xie, 2024). We first filter both datasets by target bounding box size, retaining items with area(bbox)/area(image) < 0.04. From Visual CoT, we keep all Chart/OCR-centric subsets (dude, cub, textvqa, docvqa, infographicsvqa, sroie, vsr, textcap), and treat the naturalimage subsets (flickr30k, gqa, openimages, v7w) together with V∗ as a separate stream due to lower QA reliability (e.g., weaker question–image alignment and non-unique answers). To ensure stable RL rewards, we filter this stream with an MLLM check using Qwen2.5-VL-7B and GPT-5nano under a deterministic prompt. An item is retained only if both models return correct answer; otherwise it is discarded, including ambiguous or poorly aligned cases. After this pipeline, we retain ∼100K items as the source pool D for collage synthesis.

Table 7: Benchmark of frontier multimodal models/systems on O3-BENCH. Default model/system settings are used unless stated otherwise. All results are averaged over 3 random trials.

Chart Map Overall LLaVA-OneVision-7B (Li et al., 2024) 21.1±3.2 19.4±4.3 20.2±3.7 InternVL3.5-8B (Wang et al., 2025e) 26.2±2.5 22.7±0.7 24.3±1.1 InternVL3.5-30B-A3B (Wang et al., 2025e) 24.5±3.5 21.2±1.7 22.8±2.5 GLM-4.6V (Hong et al., 2025) 51.5±2.2 38.5±2.9 44.6±2.4 Qwen2.5-VL-7B-Instruct (Bai et al., 2025b) 30.9±1.8 24.4±1.1 27.4±0.3 Qwen2.5-VL-32B-Instruct (Bai et al., 2025b) 35.4±1.0 33.5±1.2 34.4±1.0 Qwen3-VL-8B-Instruct (Bai et al., 2025a) 54.4±0.3 33.9±4.3 43.6±0.4 Qwen3-VL-8B-Thinking (Bai et al., 2025a) 49.1±2.2 33.0±0.9 40.6±0.7 Qwen3-VL-30B-A3B-Instruct (Bai et al., 2025a) 49.3±1.4 32.1±1.9 40.2±0.4 Qwen3-VL-30B-A3B-Thinking (Bai et al., 2025a) 51.1±1.5 36.8±1.2 43.6±1.3 Qwen3-VL-32B-Instruct (Bai et al., 2025a) 73.7±1.3 48.5±2.1 60.4±1.7 Qwen3-VL-32B-Thinking (Bai et al., 2025a) 52.4±3.1 40.5±1.4 46.1±1.3 Qwen3-VL-235B-A22B-Instruct (Bai et al., 2025a) 73.4±1.9 53.8±2.0 63.1±0.8 Qwen3-VL-235B-A22B-Thinking (Bai et al., 2025a) 57.3±1.2 47.8±2.0 52.3±0.8

- GPT-4o (OpenAI, 2024) 22.1±0.9 33.3±1.0 28.0±0.8 GPT-5-nano (OpenAI, 2025a) 19.2±2.3 33.3±3.9 26.5±3.1 GPT-5-mini (OpenAI, 2025a) 34.4±3.5 43.2±2.0 39.0±0.6
- GPT-5 (OpenAI, 2025a) 30.9±0.8 52.6±0.7 42.3±0.0 GPT-5.2 (OpenAI, 2025b) 31.9±2.3 39.0±2.7 35.7±2.3 OpenAI o3 (OpenAI, 2025c) 27.8±1.3 52.4±2.0 40.8±0.9

Gemini-2.5-Flash#(Comanici et al., 2025) 46.6±1.3 52.6±3.0 49.8±1.4 Gemini-2.5-Flash (Comanici et al., 2025) 61.8±1.2 59.2±1.8 60.4±0.5 Gemini-2.5-Pro (Comanici et al., 2025) 67.3±2.5 63.7±2.5 65.4±2.5 Gemini-3-Flash (Google, 2025) 68.1±2.6 69.0±3.4 68.6±1.6 Gemini-3-Pro-Preview (Google, 2025) 67.7±2.0 69.6±3.6 68.7±2.7

doubao-seed-1-6-250615 (Bytedance, 2025) 55.4±1.5 48.5±4.4 51.8±2.7 INSIGHT-O3 (w/ GPT-4o) 34.4±0.7 38.3±0.8 36.4±0.2 INSIGHT-O3 (w/ GPT-5-nano) 35.3±2.2 34.1±1.6 34.6±1.9 INSIGHT-O3 (w/ GPT-5-mini) 67.3±1.4 56.4±2.1 61.5±0.4 INSIGHT-O3 (w/ Gemini-2.5-Flash) 75.6±2.0 64.4±3.0 69.7±0.7

# Image-size constraint set to 1280×1280px, roughly the maximum supported size for OpenAI models/systems via API.

Given the filtered source pool D, we synthesize collage-style training images around one primary target (the image the model should attend to) and auxiliary fills (other images used to occupy remaining space and control background complexity). The full procedure is presented in Algorithm 1. Specifically, we sample and grid-quantize a canvas, then determine a feasible target scale by intersecting global bounds with a bbox-to-canvas cap and a minimum short-edge constraint after light aspect jitter (Steps 1–2). We plan & place the target using a fit-then-shrink heuristic with a single enlarge-canvas fallback (Steps 3–4). Remaining area is panelized into grid-aligned regions under simple aspect/size guards (Step 5). Panels are then filled (largest-first) by sampling images t˜ ∈ D using usage-aware weights (favoring less-frequently used candidates) that also roughly match panel aspect; when needed, we apply a light center crop and bounded scaling (Step 6). If packing remains incomplete after a brief extra fill pass, we resample from the canvas; otherwise we finalize the collage (Steps 7–8). To avoid ambiguity when querying the target image, we annotate each collage tile with an ID and include this ID in the question as a reference. Figure 12 shows representative visualizations of synthesized collages.

The canvas is sized so that the target box occupies only a tiny fraction of the canvas area, enforcing area(bbox)/area(canvas) < 0.0002. We filter out items that vReasoner can already solve without calling vSearcher using a pass@3 check (three attempts; any success leads to removal).

Algorithm 1 Target-and-Fill Collage Synthesis (High-level)

Require: Metadata table D (image path, Wsrc,Hsrc, object bbox), target image t⋆∈D, grid G, min short edge M, canvas area/aspect ranges [Amin,Amax] and [amin,amax], target scale bounds [λmin,λmax], target aspect jitter τtgt, fill jitter τfill, fill scales [λfillmin,λfillmax], panel aspect range [ARpanelmin ,ARpanelmax], max effective source area Smaxeff , bbox coverage cap ρcap=2×10−4, placement retries R, max attempts T

Ensure: Canvas C and placements P = {target,fills}

- 1: Precompute target meta. From t⋆, read Wsrc,Hsrc, bbox ratio ρsrc; set S⋆ ← (WsrcHsrc)· min 1, Smaxeff /(WsrcHsrc) 2, and rsrc←Wsrc/Hsrc.

- 2: for attempt = 1 to T do ▷ rejection loop
- 3: Step 1 — Sample canvas. Draw Acanvas ∼ [Amin,Amax], a ∼ [amin,amax]; snap to grid to obtain (W,H).
- 4: Step 2 — Compute feasible target scale interval.

Bounds: λ ∈ [λmin,λmax]; occupancy: λS⋆ ≤ Acanvas; bbox:

ρsrcλS⋆ Acanvas ≤ ρcap.

Choose r⋆ by log-jittering rsrc within ±τtgt and raise the lower bound on λ so that min(√λS⋆r⋆, λS⋆/r⋆)≥M.

Let Iλ be the intersection of the above constraints; if Iλ = ∅, optionally enlarge Acanvas once and recompute; if still empty, continue ▷ reject → restart at Step 1

- 5: Step 3 — Plan target box. Pick λ ∈ Iλ (e.g., midpoint); set w⋆= (λS⋆)r⋆, h⋆= (λS⋆)/r⋆; snap (w⋆,h⋆) to multiples of G.

- 6: Step 4 — Place target. Best-fit on the free list (grid-aligned). If no fit, iteratively shrink

(w⋆,h⋆) and update λ=w⋆h⋆/S⋆, keeping feasibility in Iλ, up to R retries; if still not placed, optionally enlarge canvas once and re-plan; if it fails, continue ▷ reject → restart at Step 1

- 7: Step 5 — Normalize free space. Recursively split free regions into grid-aligned panels subject to aspect ARP ∈[ARpanelmin ,ARpanelmax] and minimal size.
- 8: Step 6 — Fill panels. For each panel (largest-first), sample a fill image t˜∈D within a ±τfill

aspect band around ARP; if needed, center-crop t˜to ARP; scale with λfill ∈ [λfillmin,λfillmax] and place.

- 9: Step 7 — Resample if needed. If residual free space remains after one extra fill pass, continue ▷ reject → restart at Step 1
- 10: Step 8 — Finalize. return C,P
- 11: end for
- 12: return None ▷ no feasible collage after T attempts

- C.2 OUT-OF-LOOP RL DATA

For PP-DocLayout plus-L, we use the following configuration: {"threshold": 0.01, "layout nms": True, "layout merge bboxes mode": "union"}.

To construct meaningful visual search targets from the boxes produced by PP-DocLayout plus-L, we start by dropping boxes of trivial layout classes such as header and footer, keeping only text, image, table, chart, figure_title, and paragraph_title boxes. We then drop boxes that are too large as they usually merge disparate things together. Boxes with area larger than a quarter of the whole image area are dropped, except for charts which are usually clean and we use a much higher threshold (0.8) for them. Next, we merge boxes that are very close to each other, measured by the effort required to enclose them in one box. The effort is computed as 1 - summed_area / min_enclosing_area. We start with the boxes that require the least effort to merge, and stop until the required effort reaches a threshold. For figure_title and paragraph_title boxes, the threshold is 0.15, while for other boxes, it is 0.1. This helps merging small, auxiliary boxes such as figure titles and chart legends with their closest neighbors in the vicinity, avoiding truncating important context and preventing trivial search targets dominating

[Figure 107]

Question: In image13, what is the sanctioned faculty strength in 13 IIMS? Answer: 908 Target Image Target Bbox

- Figure 12: Example of synthesized collage for the in-loop RL. Multiple low-resolution images are stitched to raise visual density. The blue dashed box highlights the target tile; the magenta box marks the target bbox. Remaining tiles are distractors.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

a small table-like block showing quarterly visitor arrival numbers, located at the lower-left portion of the image

[Figure 112]

a laptop screen displaying a worksheet/diagram, lower-right area of the image

[Figure 113]

a stacked bar chart titled about increasing use of renewable energy (wind, solar, hydropower), located in the lower-right area of the "We need an all-of-the-above approach" section

[Figure 114]

Layout Box

Region Description

| |
|---|

###### Figure 13: Examples of InfographicVQA images with pre-generated layout boxes and region descriptions for the out-of-loop RL.

the dataset. We skip a merge if the new box would be too large (box-to-image area ratio more than 0.25) or have an extreme aspect ratio (not within 1:5 and 5:1). We do not merge charts and tables.

After merging the boxes, we drop (1) charts/tables enclosing other charts/tables, (2) unmerged images that do not contain any text or titles, (3) unmerged titles, (4) unmerged texts, (5) boxes that are too small (box-to-image area ratio less than 0.001), and (6) boxes with extreme aspect ratios (not within 1:5 and 5:1). These boxes often contain little information (e.g., icons, short texts). In the end, an image may still have multiple visual search targets; they are treated as separate data entries.

To generate region descriptions for the visual search targets obtained earlier, we first draw a red box around the target on the image, and then prompt GPT-5-nano as follows:

[SYSTEM] You are a visual assistant. Your goal is to help the user to locate the region indicated by

the red bounding box in an image. When the user asks you to describe the region, you must follow the following rules:

- - Keep it super simple and short as if you can’t see clearly what is in the region.
- - Don’t mention any details, specific content, or small text in the region.
- - Use concise, visually grounded targets (e.g., a chart, an object, a text block, a distinct area).
- - Optionally include approximate location (e.g., top-left of the image, bottom-right of the big chart, center column).
- - Optionally include the title of the region (e.g., the table about XXX, the section titled XXX).
- - Avoid non-visual or ordinal references (e.g., "the third largest bar", "the second row’s number").
- - Don’t mention the red bounding box. Output format: region_description={...}

[USER] Describe the region in the red bounding box.

In Figure 13, three examples of the final data are shown. Note that unlike collages, these examples are not stitched together; they are simply displayed side-by-side to save space.

- D ADDITIONAL INFORMATION ON INSIGHT-O3

- D.1 INSIGHT-O3 IMPLEMENTATION DETAILS

The maximum image resolution of vSearcher is set to ∼3.2M pixels (4K tokens/image) during training, and ∼12.8M pixels (16K tokens/image) during evaluation. Oversize images are downsampled to meet the constraint. We allow both vReasoner and vSearcher to make at most 6 sub-agent/tool calls during both training and evaluation. Image crops returned by sub-agent/tool calls are obtained from original images, and then resized if they exceed the size limit. For vSearcher, we use a maximum response length (including results returned by sub-agent/tool calls) of 9K and 32K tokens (with sampling temperature 1 and 0) for training and evaluation, respectively. Other hyperparameters include: training batch size 24, rollout number 8, learning rate 10−6, KL loss coefficient 0.01, reward weights λformat = 0.2, λIoU = 0.8, and IoU reward threshold α = 0.25. The composition of in-loop/out-of-loop training data is 1:1. We train vSearcher fully on-policy for 150 steps. We freeze the vision tower and the adapter of Qwen2.5-VL-7B-Instruct during the whole training process. We use GPT-5-nano (OpenAI, 2025a) for evaluating answer correctness. Our code is based on verl (Sheng et al., 2024). The prompts we use can be found in Appendix F.

- D.2 MORE VISUALIZATIONS OF INSIGHT-O3 REASONING PROCESS

- Figure 14 and 15 show examples of inference process of INSIGHT-O3 (GPT-5-mini as vReasoner). The vReasoner issues natural-language target descriptions; the vSearcher localizes evidence and returns them. Across a few rounds, the pair composes multi-step evidence and produces the final answer, demonstrating that INSIGHT-O3-VS plugs in cleanly and supports effective reasoning.

[Figure 115]

[Figure 116]

[Figure 117]

1

3

###### Question

If a customer finishes shopping at H&M and wants to go to AEON STYLE, which Elevator or Escalator is the closest and provides direct access?

- A. Escalator G
- B. Escalator E
- C. Elevator D
- D. Escalator D
- E. Elevator F
- F. None of above

[Figure 118]

✅

[Figure 119]

2

2

[Figure 120]

[Figure 121]

[Figure 122]

1

[Figure 123]

[Figure 124]

3

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

H&M is on the 2nd floor at the top center. Just southeast of H&M there is a purple escalator labeled "E" that goes down directly toward the AEON STYLE entrance on 1F. Elevator F is inside AEON STYLE (not the closest exterior connector), Escalator G is farther right, and Elevator/Escalator D options are not the direct closest link. Therefore Escalator E is the closest direct access from H&M to AEON STYLE.

I need a closer view of the rightcenter area connecting H&M (2F) down to AEON STYLE (1F) to read the labels for escalators/elevators (letters D, E, F, G) and see which one provides a direct link. I'll crop the area between H&M and AEON STYLE on the right side of the map.

I need to confirm the exact escalator/elevator labels adjacent to H&M on 2F and the direct connection down to AEON STYLE on 1F. I'll crop the area around H&M (topright of the 2F map) to show the nearby vertical-transport icons and their letters.

I need a closer view of the area around H&M (on 2F) and the AEON STYLE (on 1F) to identify the nearest elevator/escalator that provides direct access between them. I'll crop the top-left section of the map where H&M and the adjacent floors are located.

The answer is Escalator E (B).

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Right-center area between H&M (2F) and AEON STYLE (1F), showing escalator/elevator labels and connecting paths

[Figure 137]

Top-right area of 2F showing H&M and immediate surrounding escalator/elevator icons and labels

[Figure 138]

[Figure 139]

Top-left area showing 2F H&M and surrounding elevators/escalators, and the 1F area below it

1 2 3

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

vReasoner vSearcher Closed View Region Description

- Figure 14: Qualitative example 1 with GPT-5-mini as vReasoner and INSIGHT-O3-VS as vSearcher. The reasoner requests venue-level cues (e.g., legend/index lookups); the searcher returns localized regions and snippets, iterating to a correct answer.

- 1

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

- 2 1

[Figure 149]

[Figure 150]

3

[Figure 151]

###### Question

If a disabled visitor finishes lunch at KFC in the park at 1 pm and plans to watch an animal show at a venue offering special viewing areas for disabled individuals, which venue and showtime should they choose?

- A. Rainforest KidzWorld Amphitheatre, 1.30 pm
- B. Rainforest KidzWorld Amphitheatre, 4.00 pm
- C. Shaw Foundation Amphitheatre, 2.30 pm
- D. Shaw Foundation Amphitheatre, 1.30 pm
- E. Shaw Foundation Amphitheatre, 5.00 pm
- F. No right choice

[Figure 152]

[Figure 153]

✅

4

4

[Figure 154]

3

[Figure 155]

2

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

The legend shows a wheelchair icon meaning "Viewing for Guests with Disabilities." The Shaw Foundation Amphitheatre (right center of the map) displays that wheelchair icon next to it, while the Rainforest KidzWorld Amphitheatre does not. The schedule shows Shaw Foundation Amphitheatre has a 2.30 pm show (Rainforest Fights Back) after 1.00 pm. Therefore the disabled visitor should choose the Shaw Foundation Amphitheatre at 2.30 pm.

I need to confirm which venue specifically has special viewing areas for disabled visitors. The legend on the map will show the symbol for "Viewing for Guests with Disabilities" and I need to locate that symbol near the amphitheatres (Shaw Foundation Amphitheatre and Rainforest KidzWorld Amphitheatre). I'll crop the legend area first to identify the symbol, then crop near the amphitheatres to see which one displays that symbol.

I need to read the animal presentation schedule and any notes about special viewing areas for disabled visitors. The schedule and icons are located in the top-left corner of the map, so I will crop that region to read the details clearly.

I need to check whether the Shaw Foundation Amphitheatre has the wheelchair "Viewing for Guests with Disabilities" icon. I'll crop the area around the Shaw Foundation Amphitheatre (centralleft of the map) to look for that symbol near the amphitheatre seating.

I will zoom into the Rainforest KidzWorld area near the topcenter of the map to look for the disabledviewing (wheelchair) icon at that amphitheatre.

The answer is \boxed{C}

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Shaw Foundation Amphitheatre area central-left of map including amphitheatre seating and nearby icons

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Rainforest KidzWorld area at top-center including the KidzWorld building and surrounding amphitheatre

bottom-left legend area showing map symbols and labels

1 2 3

corner showing the

###### 4

animal presentation schedule and icons}

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

vReasoner vSearcher Closed View Region Description

- Figure 15: Qualitative example 2 with GPT-5-mini as vReasoner and INSIGHT-O3-VS as vSearcher. The reasoner requests venue-level cues (e.g., legend/index lookups); the searcher returns localized regions and snippets, iterating to a correct answer.

- D.3 COMPARATIVE ANALYSIS BETWEEN INSIGHT-O3 AND BASELINES

In Figure 16–19, we compare the behavior of INSIGHT-O3 (GPT-5-mini + InSight-o3-vS) with two baselines: (1) GPT-5-mini and (2) GPT-5-mini + Qwen2.5-VL-7B. This comparative analysis is based on three examples of O3-BENCH. We rate each crop returned by vSearcher on three levels:

- • High-quality crops tightly enclose the visual search targets and the relevant context.
- • Medium-quality crops contain the visual search targets but include too much context.
- • Low-quality crops miss or truncate the visual search targets or relevant context.

In the most basic setting where vReasoner does not have access to vSearcher, it often uses similar reasoning patterns as follows to reach its conclusion: “I first locate ... I see ... I then look for ... There is ... Therefore ...” (see the top parts of Figure 17-19). During this process, vReasoner often hallucinates and makes factual errors about what it sees, suggesting that it does not really see the relevant visual details clearly but still pretends so anyway.

With the vanilla Qwen2.5-VL-7B vSearcher, vReasoner is able to take closer looks at regions of interest and makes less factual errors (see the bottom part of Figure 17 and the middle part of Figure 18-19). However, the vanilla vSearcher is often unreliable, returning inaccurate/wrong crops to vReasoner or simply concluding that the target is not in the image, usually after a minimal amount of (sometimes none) reasoning. In such cases, vReasoner would eventually give up and resort to its own perception after multiple failed visual search attempts, leading to wrong final answers.

In comparison, our visual search agent, InSight-o3-vS, would usually first reason about the visual search request and then crop the candidate region to verify before returning it to vReasoner. For example, in Figure 16, InSight-o3-vS first reasons about what the bounding box should cover:

Based on the description, the right section of the map includes the legend/index and the "Catering venues" list with numbered cafes. The legend/index is located at right of the map, and the "Catering venues" list is further down, under the "Catering venues" heading. The bounding box should cover these areas.

Then, after viewing the cropped region, it concludes:

Based on the tool response, the right section of the map showing the legend/index and the "Catering venues" list with numbered cafes is already covered by the bounding box provided. Therefore, no further zooming is necessary.

In some cases, InSight-o3-vS is able to correct an initial bad crop after reviewing the crop:

Based on the response, the zoomed-in area does not match the intended area around grid B4. The bounding box needs to be adjusted to better capture the car parks and nearby cafe icons and labels.

The returned crops are usually medium-to-high-quality crops as shown in Figure 16. From these cases, we can see that vSearcher mostly helps vReasoner by (i) precisely locating the regions requested by vReasoner and (ii) presenting them to vReasoner and directing its attention to those regions, thereby reducing hallucination and facilitating evidence-based reasoning.

- D.4 FAILURE CASES OF INSIGHT-O3

Figure 20-21 show typical failure cases of INSIGHT-O3. In the first three failure cases, vSearcher (InSight-o3-vS) provided the correct crops for the search targets but vReasoner (GPT-5-mini) answers incorrectly due to its own errors, e.g., ignoring visual evidence due to internal knowledge bias, and jumping to conclusion without examining key visual information. This suggests that existing frontier models, even the proprietary ones, are still not very good at thinking with images in complex scenarios. The last failure cases is due to vSearcher failing to understand the structure of the map and/or the relatively complicated region descriptions involving multiple visual cues. Finally, we note that, at least on O3-BENCH, most wrong answers are due to vReasoner, not vSearcher.

###### Whiteknights map key (numerical)

Whiteknights map key (A – Z)

Classics HumSS Economics HumSS English Language and Applied

Art TOB 1 Design and Print Studio TOB 2 National Centre for

Grounds Maintenance Of ce Estates and Facilities

Institute for Cardiovascular and Metabolic Research (ICMR)

Hopkins

3sixty bar Students’ Union

Crop Protection Unit Agriculture

IT Services Mathematics and IT Services

Southern Universities Management Services (SUMS)

Science and Technology Centre

|E<br><br>D<br><br>C<br><br>B<br><br>A<br><br>town cen and Mu|ALLCROFTROAD<br><br>Town centre and station<br><br>To<br><br>tre, Royal Berkshire Hosp seum of English Rural Lif<br><br>P*<br><br>Benyon Hall<br><br>Wan H<br><br>Hillside<br><br>White|NORTHC OU R T AV<br><br>ENUE<br><br>E<br><br>L<br><br>M<br><br>H<br><br>U<br><br>S<br><br>R E<br><br>D<br><br>L A N D S R O A<br><br>D<br><br>N<br><br>E<br><br><br>W<br><br>R<br><br>O<br><br>A<br><br>D<br><br>London Road<br><br>ital<br><br>P*<br><br>P*<br><br>Shinﬁeld Road<br><br>N   Health Centre<br><br>St Patrick’s Hall<br><br>Sherﬁeld Hall<br><br>tage all<br><br>St<br><br>knights|W<br><br>H<br>I<br><br><br>T<br><br>E<br><br>K<br><br>N<br><br>I<br><br>G<br>H<br><br><br>T<br><br>S R O A D<br><br>U<br><br>P<br><br>P<br><br>E<br><br>R<br><br>R<br><br>E<br><br>D<br><br>L<br><br>A<br><br>N<br><br>D<br><br>S<br><br>R<br><br>O<br><br>A<br><br>D<br><br>R<br>S<br>T<br><br><br>R<br><br>O<br><br>A<br><br>D<br><br>H<br>I<br><br><br>N<br><br>F<br><br>I<br><br>E<br><br>L D R O A D<br><br>Q<br><br>u<br><br>e<br><br>e<br><br>n<br><br>’<br><br>s<br><br>D<br><br>r<br><br>i<br><br>v<br><br>e<br><br>C<br><br>h<br><br>a<br><br>n<br><br>c<br><br>e<br><br>l<br><br>l<br><br>o<br><br>r<br><br>’<br><br>s<br><br><br>W<br><br>a<br><br>y<br><br>P8<br><br>P6<br><br>P<br><br>P7<br><br>P2<br><br>P<br><br>P*<br><br>P*<br><br>P*<br><br>P*<br><br>P*<br><br>P1 P P<br><br>P16<br><br>P5<br><br>P11<br><br>P14<br><br>P15<br><br>P4<br><br>P10<br><br>P9<br><br>P13<br><br>P12<br><br>P<br><br>P3<br><br>Pepper Lane<br><br>Pedestrian entrance<br><br>Sports grounds<br><br>Sports grounds<br><br>Foxhill<br><br>Harris Garden<br><br>Whiteknights Lake<br><br>Friends' Bridge<br><br>Mackinder Hall<br><br>Windsor Hall<br><br>George's Hall<br><br>Childs Hall<br><br>McCombie<br><br>Greenow<br><br>Stenton Hall & Townhouses<br><br>Dunsden Crescent<br><br>campus|Path Cafe<br><br>W O K I<br><br>W<br><br>H<br>I T<br><br><br>E K N<br><br>G<br>H T<br><br><br>S R O A D<br><br>R<br><br>O<br><br>A<br><br>D<br><br>W<br><br>P* P*<br><br>P*<br><br>P*<br><br>P22<br><br>P22<br><br>P23<br><br>P<br><br>P*<br><br>P*<br><br>P24<br><br>Halls of residence<br><br>Car park restricted to users of this building<br><br>Car parks<br><br>One-way system<br><br>P8<br><br>P*<br><br>Under construction<br><br>Wessex Hall<br><br>Bridges Hall<br><br>(open 2014)<br><br>0|CHUR<br><br>CH<br><br>N<br><br>G<br>H A<br><br><br>M R O A D<br><br>Town centre and station<br><br>Earley Gate|
|---|---|---|---|---|---|
| | | | | |L<br><br>D<br>E<br><br><br>R<br><br>N<br><br>E<br><br>S<br><br><br>R<br><br>O<br><br>D<br><br>|
|F| | |P<br><br>E<br><br>P<br><br>P<br><br>E<br><br>R L A N E|W<br><br>I<br><br>L<br><br>D<br>E<br><br><br>R<br><br>N<br><br>E<br><br>S<br><br>S<br><br>0|600ft|

1 2 3 4 5 6 7 8 9

Biological Sciences Microbiology

Knight

Accommodation Of ce Carrington

Design and Print Studio TOB 2

JJ Thomson JJ Thomson

Town centre Town centre

Pharmacy Hopkins

Southern Universities Purchasing Consortium (SUPC)

Science and Technology Centre

HumSS

TOB 2

Admissions Of ce Miller

Disability Advisory Service Carrington

Job Shop (students) Students’ Union Hub Knowledge Transfer Centre Systems Engineering Law Foxhill House

Bus stop

Academic, administrative and service buildings

Innovation Works Science and Technology Centre

Accommodation Of ce Carrington

Linguistics

Biotechnology Education

Pedestrian entrance

Agriculture, Policy and Development

Agriculture

Economics HumSS

London Road campus,

Speech and Language Therapy Clinic

Psychology

English Literature HumSS History HumSS History of Art HumSS Humanities and Social Sciences

Pre-school TOB 2

Careers, Placement and Experience Centre

Carrington

English Language and Applied Linguistics

HumSS

Science and Technology Centre

Science and Technology Centre

Typography and Graphic Communication

TOB 2

Allen Laboratory The Allen Laboratory

Library Library

SportsPark SportsPark

Carrington Carrington Counselling and Wellbeing Carrington Disability Advisory Service Carrington Examinations Of ce Carrington Faculty Of ces Carrington International Student Advice Carrington Peer Support Carrington Room Bookings and

English Literature HumSS

Alumni Relations Blandford Lodge

Management (School of) Henley Business School

Southern Universities Management Services (SUMS)

Science and Technology Centre

Statistical Services Centre Harry Pitt

Construction Management and Engineering

Engineering

ESSC (Environmental Systems Science Centre)

Harry Pitt

HumSS

Archaeology Archaeology

Mathematics and Statistics Mathematics and IT Services

Student Financial Support Team

Carrington

(HumSS)

Art TOB 1

Mathematics Support Centre Library (S@iL)

Systems Engineering Engineering

Estates and Facilities Estates and Facilities

Southern Universities Purchasing Consortium (SUPC)

Science and Technology Centre

In-sessional English Support Programme (ISLC)

HumSS

Student Helpdesk Team Carrington

Athletics Pavilion Athletics Pavilion

Meteorology Meteorology

Palmer Palmer Reading Film Theatre Palmer Communications Of ce Whiteknights House

Examinations Of ce Carrington Faculty Of ces Carrington Farm Management Unit Agriculture

Student Recruitment and Outreach

Blandford Lodge

Biological Sciences Harborne

Meadow Suite Park House

Campus Central (shop) Cedars Hotel and Conference Centre

ISLC (International Study and Language Centre)

HumSS

Biological Sciences Hopkins

Miller Miller

Student Services Centre Carrington

Cedars Hotel and Conference Centre

Cedars Hotel and Conference Centre

Biological Sciences Philip Lyle

Modern Languages and European Studies

HumSS

Minghella Minghella

Finance and Corporate Services

Whiteknights House

Carrington

AD

Film, Theatre and Televison Minghella

Students’ Union Students’ Union Hub

Biological Sciences Microbiology

Knight

Modern Languages and European Studies

HumSS

Timetabling

RO

Finance and Corporate Services

Whiteknights House

Study Advice Library (Room 103) Synthetic Turf Pitch Synthetic Turf Pitch Systems Engineering Systems Engineering

Centre for Integrative Neuroscience and Neurodynamics (CINN)

Harry Pitt

Philosophy HumSS

Graduation Of ce Whiteknights House Human Resources Whiteknights House Pensions Of ce Whiteknights House Postal Services Whiteknights House Receipts Of ce Whiteknights House Research and Enterprise Whiteknights House Security Of ce Whiteknights House Vice-Chancellor’s Of ce Whiteknights House Whiteknights House Whiteknights House 3sixty bar Students’ Union

Student Financial Support Team

Carrington

Biological Sciences Teaching Laboratories

AMS Tower

Politics and International Relations

HumSS

Muslim Centre Archway Lodge

Food and Nutritional Sciences Food Biosciences

National Centre for Biotechnology Education

TOB 2

Student Helpdesk Team Carrington Student Services Centre Carrington

ESSC (Environmental Systems Science Centre)

Harry Pitt

Foxhill House Foxhill House

Blandford Lodge Blandford Lodge

Ure Museum of Greek Archaeology

HumSS

Systems Engineering Engineering

Geography and Environmental Science

Miller

Campaigns and Supporter Engagement Of ce

Blandford Lodge

National Centre for Earth Observation

Meteorology

Technologies for Sustainable Built Environments Centre (TSBE)

JJ Thomson

Harry Pitt Harry Pitt Statistical Services Centre Harry Pitt Meteorology Meteorology

Centre for Entrepreneurship Henley Business School Henley Business School Henley Business School Informatics Research Centre Henley Business School Management (School of) Henley Business School Reading Real Estate

IT Helpdesk Library Library Library Mathematics Support Centre Library (S@iL)

Geography and Environmental Science

Russell

Campus Central (shop) Cedars Hotel and Conference Centre

Nursery Nursery

Typography and Graphic Communication

TOB 2

Palmer Palmer

Graduate Institute of International Development and Applied Economics

Agriculture

National Centre for Earth Observation

Meteorology

Careers, Placement and Experience Centre

Carrington

Study Advice Library (Room 103)

Park House Park House Peer Support Carrington Pensions Of ce Whiteknights House

Ure Museum of Greek Archaeology

HumSS

Henley Business School

Centre for Advanced Microscopy (CfAM)

JJ Thomson

Agriculture, Policy and Development

Agriculture

Carrington Carrington

Foundation

Graduate School Old Whiteknights House

Venue Reading Park House

Catering – Estates and Facilities

Park House

Real Estate and Planning Henley Business School

Health and Safety Services JJ Thomson JJ Thomson JJ Thomson Technologies for Sustainable Built Environments Centre (TSBE)

Job Shop (students) Students’ Union Hub Students’ Union Students’ Union Hub Biological Sciences Teaching

Graduation Of ce Whiteknights House Grounds Maintenance Of ce Estates and Facilities Grounds Maintenance Yard Estates and Facilities

Centre for Agricultural Strategy

Agriculture

Pharmacy Chemistry and Pharmacy

Veterinary Epidemiology and Economic Research Unit (VEERU)

Agriculture

Central Kitchen Central Kitchen

Cedars Hotel and Conference Centre

Cedars Hotel and Conference Centre

Pharmacy Food Biosciences

Film, Theatre and Televison Minghella Minghella Minghella Nursery Nursery

JJ Thomson

Centre for Agri-Environmental Research

Agriculture

AMS Tower

Pharmacy Hopkins

| | |
|---|---|
|Central Kit|chen Central Kitchen|
| | |

Laboratories

Vice-Chancellor’s Of ce Whiteknights House

Harborne Harborne Harry Pitt Harry Pitt Health and Dental Centre Health and Dental Centre

Philosophy HumSS

Crop Protection Unit Agriculture Farm Management Unit Agriculture Graduate Institute of

Cole Museum of Zoology AMS Tower

Wager Wager

Centre for Advanced Microscopy  CfAM 

JJ Thomson

IT Services Mathematics and IT Services Mathematics and Statistics Mathematics and IT Services Human and Environmental

Politics and International Relations

HumSS

Biological Sciences Harborne Harborne Harborne Biological Sciences Philip Lyle Construction Management and

Reading Enterprise Centre Reading Enterprise Centre

Walker Institute for Climate System Research

Agriculture

Agriculture

Synthetic Turf Pitch Synthetic Turf Pitch

Centre for Agricultural Strategy

Agriculture

Postal Services Whiteknights House

Health and Safety Services JJ Thomson

Wager

International Development and Applied Economics

Health and Dental Centre Health and Dental Centre

Whiteknights House Whiteknights House

Sciences

Pre-school TOB 2

Henley Business School Henley Business School

Centre for Agri-Environmental Research

Agriculture

URS

|Eat at HumSS<br><br>Café Libro<br><br>Dol.cHe Vita<br><br>Eat at SportsPark<br><br>Fusion Café<br><br>Agriculture snack bar<br><br>Café Index<br><br>Eat at Enterprise<br><br>Catering venues|
|---|

Wager Wager

Veterinary Epidemiology and Economic Research Unit (VEERU)

Agriculture

Procurement Park House

History HumSS History of Art HumSS Human and Environmental

Engineering

Catering venues

Chemical Analysis Facility (CAF)

Chemistry and Pharmacy

Psychology and Clinical Language Sciences

Psychology

Centre for the Development of Teaching and Learning

Miller

Grounds Maintenance Yard Estates and Facilities

Wager

Muslim Centre Archway Lodge

Walker Institute for Climate System Research

Agriculture

Chemistry Chemistry and Pharmacy Pharmacy Chemistry and Pharmacy Catering – Estates and

Reading Enterprise Centre Reading Enterprise Centre

Agriculture snack bar

Centre for Entrepreneurship Henley Business School

Sciences

Knowledge Transfer Centre Systems Engineering Systems Engineering Systems Engineering SportsPark SportsPark

Reading Film Theatre Palmer

Café Index Café Libro Dol.cHe Vita

Centre for Integrative Neuroscience and Neurodynamics (CINN)

Harry Pitt

Human Resources Whiteknights House

Charlie Waller Institute Psychology Clinical Language Sciences Psychology Psychology and Clinical Language

Park House

Reading Real Estate Foundation

Henley Business School

Humanities and Social Sciences (HumSS)

HumSS

Facilities

Psychology

Chaplaincy Centre Park House Lodge

Chaplaincy Centre Park House Lodge Athletics Pavilion Athletics Pavilion Reading Scienti c Services Ltd

Meadow Suite Park House Park House Park House Procurement Park House Senior Common Room Park House Venue Reading Park House Graduate School Old Whiteknights House

Reading Scienti c Services Ltd (RSSL)

The Lord Zuckerman Research Centre

ICMA Centre ICMA Centre

Sciences

Eat at Enterprise

200m

Charlie Waller Institute Psychology

Informatics Research Centre Henley Business School

Speech and Language Therapy Clinic

Psychology

Real Estate and Planning Henley Business School

Chemical Analysis Facility  CAF 

Chemistry and Pharmacy

The Lord Zuckerman

Eat at HumSS

Innovation Works Science and Technology Centre

Research Centre Food and Nutritional Sciences Food Biosciences Pharmacy Food Biosciences Allen Laboratory The Allen Laboratory

(RSSL)

Receipts Of ce Whiteknights House

Geography and Environmental Science

Russell

Eat at SportsPark

Chemistry Chemistry and Pharmacy

Research and Enterprise

Whiteknights House

In-sessional English Support Programme (ISLC)

HumSS

Fusion Café

Classics HumSS

Russell Russell

Room Bookings and Timetabling

Carrington

Clinical Language Sciences Psychology Cole Museum of Zoology AMS Tower Communications Of ce Whiteknights House

Insurance Of ce Blandford Lodge

ICMA Centre ICMA Centre Archaeology Archaeology Social Club (Employees) TOB 2

Alumni Relations Blandford Lodge Blandford Lodge Blandford Lodge Campaigns and Supporter

Admissions Of ce Miller

Institute for Cardiovascular and Metabolic Research (ICMR)

Hopkins

Eat at the Square

Russell Russell

Eat at the Square

Centre for the Development of Teaching and Learning

Miller

Blandford Lodge

Eat and Drink at Park House

Science and Technology Centre

Science and Technology Centre

International Of ce Blandford Lodge

Eat and Drink at Park House

Construction Management and Engineering

URS

Engagement Of ce

Foxhill House Foxhill House Law Foxhill House Biological Sciences Hopkins

Park Eat

Geography and Environmental Science

Miller

International Student Advice Carrington

Park Eat

Insurance Of ce Blandford Lodge International Of ce Blandford Lodge Student Recruitment and

Security Of ce Whiteknights House

Construction Management and Engineering

Engineering

ISLC (International Study and Language Centre)

HumSS

Senior Common Room Park House

Miller Miller

Blandford Lodge

Blandford’s

Social Club (Employees) TOB 2

Blandford’s

Counselling and Wellbeing Carrington

Estates and Facilities Estates and Facilities

IT Helpdesk Library

Outreach

###### Question

###### Explanation

If you are currently at P2 Car park, which is near B4, and want to find a nearby place for coffee, which cafe is the closest?

- Step 1. Locate P2 Car park near B4 on the map .
- Step 2. Check the building list in the bottom right corner under Catering venues for the listed cafes: Cafe Libro (2, D5), Eat at SportsPark (39, C3), Agriculture snack bar (59, D8), Cafe Index (71, C4), Fusion Cafe (54, C7).
- Step 3. Identify the closest cafe, which is building 71 , Cafe Index, and choose option D.

Repton Loudon

- A. Cafe Libro
- B. Eat at SportsPark ✅

C. Agriculture snack bar D. Cafe Index

- E. Fusion Cafe
- F. No right choice

Grey

[Figure 179]

Addington

Penn

Buildings

Bus stop Telephones Halls of residence Path

Winchcombe

T

Blagrave

T

Car parks Cafe

P8

Mitford Huntley

Highwood

Blackstone

P

[Figure 180]

###### GPT-5-Mini InSight-o3-vS

Highquality Crop

Mediumquality Crop

Lowquality Crop

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Hollins

[Figure 187]

Faringdon

InSight-o3 vSearcher

Zoom-in Tool

Region Description

To playingﬁelds

P

vReasoner

P

Tennis courts

 a

###### Round 1

 h

 b

T

 g

P

P

 c

P

 a

P

[Figure 188]

[Figure 189]

T

 f

 b

P

 i

 e

 k

I need to find the P2 car park located near grid B4 and identify the nearest cafe. I'll zoom into the map area around grid B4 (upper middle-left) to locate P2 and nearby cafes.

P

P

 d

 j

U

E

N

E

V

A

GPT

S

D

N

Entrance 2

A

L

D

O

O

W

Main Entrance

[Figure 190]

Entrance 3

P

0 100m

[Figure 191]

0 300ft

1 2 3 4 5 6 7 8 9

[Figure 192]

Locate area around grid B4 (upper middle-left of the campus map) showing car parks and nearby cafe icons and labels.

Town centre

Bus stop

Academic, administrative and service buildings

Pedestrian entrance

Town centre

London Road campus, town centre, Royal Berkshire Hospital and Museum of English Rural Life

Path Cafe

Halls of residence

[Figure 193]

[Figure 194]

P*

W

- H
- I

T

[Figure 195]

E

K

###### Under construction

N

[Figure 196]

I

- G
- H

Pedestrian entrance

Wantage Hall

T

S R O A D

Car parks

P8

N

A

D

L

D

S

E

R

R

O

R

E

A

P

P

D

U

[Figure 197]

ALLCROFTROAD

[Figure 198]

A

[Figure 199]

Based on the response, the zoomed-in area does not match the intended area around grid B4. The bounding box needs to be adjusted to better capture the car parks and nearby cafe icons and labels.

St George's Hall

Childs Hall

Based on the map, the area around grid B4 (upper middle-left of the campus map) includes car parks and nearby cafe icons and labels. The bounding box should be set to include these elements.

Car park restricted to users of this building

P*

P*

Based on the response, the zoomed-in area matches the intended area around grid B4, so no further adjustments are needed.

[Figure 200]

P*

Foxhill

Hillside

One-way system

P*

Dunsden Crescent

P*

Greenow

To

R E

London Road

D

Town centre and station

Mackinder Hall

D

A

McCombie

O

L A N D S R O A

Whiteknights Lake

R

AD

W

InSight-o3

- D

N

- E

D

A

P*

RO

O

Windsor Hall

P*

Stenton Hall & Townhouses

R

W O

Wessex Hall

Sports grounds

CH

- R
- S
- T

SIDMOUTH STREET

B

P*

P*

Pedestrian only entrance

K I N

W

U

CHUR

Crown Place

H

- H
- I T

M

- G
- H A

P2

LONDON ROAD (A4)

L

VEHICULAR ENTRANCE

E

Town centre and station

Town centre

P3

M R O A D

E K N I

P

AD

RO

Royal Berkshire Hospital

P16

- G
- H T

Bridges Hall

KENDRICK

e

v

i

r

D

s

’

n

(open 2014)

e

1 2 3 4 5 6 7 8 9

P P

e

P1

u

Q

S R O A D

Shinﬁeld Road

Town centre

| | |
|---|---|
| | |

Benyon Hall

Earley Gate

Bus stop

Academic, administrative and service buildings

Pedestrian entrance

Town centre

N  

Sports grounds

Health Centre

OILETS

AC

###### Round 2

P4

London Road campus, town centre, Royal Berkshire Hospital and Museum of English Rural Life

C

White Gates car park

WN PL

Path Cafe

Halls of residence

P*

St Patrick’s Hall

P*

S N F I E L D R O A D

| | |
|---|---|
| | |

W

- H
- I

- H
- I

T

E

K

P5

P*

REDL ND RO AD

###### Under construction

CR

NORTHC OU R T AV

N

I

P*

- G
- H

Pedestrian entrance

P*

C

h

a

Wantage Hall

n

T

c

e

l

l

o

- r

’

- s

W

S R O A D

a

y

[Figure 201]

Car parks

P8

[Figure 202]

N

A

D

L

D

S

E

R

R

O

R

E

A

P

P

D

U

ALLCROFTROAD

P

A

[Figure 203]

D

St George's Hall

Childs Hall

P9

Friends' Bridge

A

Car park restricted to users of this building

P*

P22

P6

P*

P

O

P15

R

P*

P*

Foxhill

P7

Hillside

S

- R

N

E

- S

I need a closer view of the area around the P2 car park (right-center of the zoom), showing the cup (cafe) icons and any nearby building numbers or labels so I can identify which named cafe is closest. I'll request a crop centered on the P2 marker and the surrounding cafes/buildings.

One-way system

ENUE

P*

P

Dunsden Crescent

P*

D

Greenow

P22

To

R E

P24

London Road

- D
- E

West cloister car park

Sherﬁeld Hall

D

Cafe

Town centre and station

Mackinder Hall

D

P10

A

P14

McCombie

GPT

O

L A N D S R O A

Whiteknights Lake

L

R

AD

I

Bus stop

W

W

- D

N

- E

D

P23

P8

A

P*

RO

Taxis

O

Windsor Hall

[Figure 204]

P*

Stenton Hall & Townhouses

R

W O

Wessex Hall

Sports grounds

CH

- R
- S
- T

B

P*

[Figure 205]

Bike rack

P*

K I N

W

U

CHUR

H

P12

- H
- I T

M

- G
- H A

P2

L

P13

P11

E

Town centre and station

P3

Pepper Lane

M R O A D

E K N I

P

Locate area around the P2 car park on the right-center showing nearby cup icons, building numbers 71, 34, 50, 138, and 800.

E

0 50m

P16

- G
- H T

Bridges Hall

Harris Garden

###### Whiteknights campus

0 80ft

D

e

v

i

r

D

s

’

n

[Figure 206]

(open 2014)

e

P P

e

P1

A

u

Q

S R O A D

O

[Figure 207]

0 200m

Shinﬁeld Road

[Figure 208]

R

Acacia Road

P

S

Benyon Hall

E

FootpathtoKendrickRd.andbusstop

[Figure 209]

AcaciaRoadcarpark

S

P

Earley Gate

[Figure 210]

P

E

0 600ft

E

N

Updated April 2013. © Copyright The University of Reading

Kendrick car park

N  

R L A N E

Sports grounds

Health Centre

MERL entrance

R

P4

C

- D
- E

L

P*

St Patrick’s Hall

I

Based on the zoomed-in image, the bounding box seems to cover the area around the P2 car park, including the nearby cup icons and building numbers 71, 34, 50, 138, and 800. The response from the tool seems to match the required area, so no further adjustments

W

S N F I E L D R O A D

Based on the map, the P2 car park is located in the central-right area of the image. The nearby cup icons, building numbers 71, 34, 50, 138, and 800 are also in this area. The bounding box should encompass these elements.

- H
- I

P5

P*

NORTHC OU R T AV

Museum of English Rural Life

P*

F

and Special Collections Service

P*

C

h

a

n

c

e

Museum car park

l

l

o

- r

’

- s

W

a

y

P

Whiteknights campus

D

InSight-o3 are needed.

P9

Friends' Bridge

A

P22

P6

P

O

P15

R

P*

P7

S

- R

N

E

- S

ENUE

P

D

P22

P24

- D
- E

Sherﬁeld Hall

P10

P14

L

I

W

P23

P8

###### Round 3

P12

P13

P11

Pepper Lane

E

[Figure 211]

Harris Garden

[Figure 212]

###### Whiteknights campus

D

A

O

0 200m

R

P

S

E

S

P

P

E

0 600ft

E

N

Updated April 2013. © Copyright The University of Reading

R L A N E

R

I need to read the label next to the coffee cup icon by P2 (the marker labeled 202 near Windsor/Stenton). I'll zoom into the top-left corner around Windsor Hall and the 202 cup icon to identify the cafe name/number and match it to the legend choices.

- D
- E

L

I

W

GPT

F

[Figure 213]

[Figure 214]

Repton Loudon

[Figure 215]

Grey

Locate close-up of Windsor Hall area at top-left showing the P2 marker, the coffee cup icon with '202' label, and nearby building labels.

1 2 3 4 5 6 7 8 9

Town centre

Addington

Penn

Bus stop

Academic, administrative and service buildings

Buildings

Bus stop Telephones Halls of residence Path

Pedestrian entrance

Town centre

[Figure 216]

London Road campus, town centre, Royal Berkshire Hospital and Museum of English Rural Life

[Figure 217]

Path Cafe

Halls of residence

[Figure 218]

Winchcombe

P*

T

[Figure 219]

W

[Figure 220]

- H
- I

T

Blagrave

E

K

###### Under construction

N

I

- G
- H

Pedestrian entrance

T

Car parks Cafe

Wantage Hall

T

P8

S R O A D

Car parks

Mitford Huntley

P8

Highwood

N

A

D

L

D

S

E

R

R

O

R

E

A

P

P

D

U

Based on the zoomed-in image, the P2 marker, the coffee cup icon with '202' label, and nearby building labels are clearly visible. The bounding box coordinates provided accurately capture this area, so

ALLCROFTROAD

A

Based on the description, the area of interest is the top-left part of the map, specifically around the Windsor Hall area. The P2 marker, the coffee cup icon with '202' label, and nearby building labels should be visible in this region. The bounding box coordinates provided will help zoom in on this specific area.

St George's Hall

Childs Hall

Car park restricted to users of this building

P*

P*

Blackstone

P*

Foxhill

P

Hillside

One-way system

P*

Dunsden Crescent

P*

Greenow

InSight-o3 no further zooming is necessary.

To

R E

London Road

D

Town centre and station

Mackinder Hall

D

A

McCombie

O

L A N D S R O A

Hollins

Whiteknights Lake

R

AD

W

Faringdon

- D

N

- E

D

A

P*

RO

To playingﬁelds

O

P

Windsor Hall

P*

Stenton Hall & Townhouses

R

W O

Wessex Hall

Sports grounds

CH

- R
- S
- T

B

P*

P

P*

Whiteknights map key (A – Z)

K I N

W

U

CHUR

H

- H
- I T

M

- G
- H A

P2

Tennis courts

L

E

Town centre and station

P3

Name and grid ref Building Name and grid ref Building Name and grid ref Building Name and grid ref Building

M R O A D

E K N I

P

###### Round 4

 a

- 29 3sixty bar C5 Students’ Union

135 Accommodation Of ce C5 Carrington 49 Admissions Of ce D4 Miller 59 Agriculture, Policy and

Development D8

Agriculture

48 Allen Laboratory D5 The Allen Laboratory 10 Alumni Relations D4 Blandford Lodge 74 Archaeology D5 Archaeology 20 Art D8 TOB 1 43 Athletics Pavilion B3 Athletics Pavilion

- 31 Biological Sciences E6 Harborne

118 Biological Sciences D6 Hopkins

- 32 Biological Sciences E6 Philip Lyle 53 Biological Sciences Knight

- 30

59 Crop Protection Unit D8 Agriculture 21 Design and Print Studio C8 TOB 2

4 IT Services D5 Mathematics and IT Services 3 JJ Thomson D5 JJ Thomson

54 Southern Universities

Science and Technology

P16

- G
- H T

Bridges Hall

 h

Centre 54 Southern Universities Purchasing

Management Services (SUMS) C7

e

v

i

r

D

 b

s

’

n

(open 2014)

e

P P

e

P1

u

Repton Loudon

Q

S R O A D

Shinﬁeld Road

Science and Technology

Benyon Hall

Earley Gate

135 Disability Advisory Service C5 Carrington 1 Economics C4 HumSS 1 English Language and Applied

29 Job Shop (students) C5 Students’ Union Hub 38 Knowledge Transfer Centre E5 Systems Engineering 84 Law A6 Foxhill House

[Figure 221]

T

[Figure 222]

N  

Centre 62 Speech and Language Therapy

Consortium (SUPC) C7

Sports grounds

Health Centre

P4

 g

C

Grey

P

P

P*

St Patrick’s Hall

Psychology

 c

P

 a

S N F I E L D R O A D

Addington

- H
- I

P

Penn

P5

HumSS

P*

NORTHC OU R T AV

Clinic D7

P*

Buildings

Bus stop Telephones Halls of residence Path

T

 f

Linguistics C5

P*

C

h

I need the legend that maps numbered points (like 202, 71, 138, etc.) to cafe names. I'll request a crop of the legend area at the right of the map where the catering venues and numbered index are listed.

a

n

c

 b

e

l

l

P

o

- r

’

- s

W

a

y

2 Library D5 Library

39 SportsPark C2 SportsPark 56 Statistical Services Centre D7 Harry Pitt

Winchcombe

 i

P

T

D

1 English Literature C5 HumSS 56 ESSC (Environmental Systems

Blagrave

P9

138 Management (School of) C5 Henley Business School 4 Mathematics and Statistics D5 Mathematics and IT Services 2 Mathematics Support Centre D5 Library (S@iL)

 e

Friends' Bridge

A

GPT

P22

P6

[Figure 223]

P

O

 k

P15

T

Harry Pitt

Car parks Cafe

R

P8

P*

135 Student Financial Support

Carrington

P7

Mitford Huntley

Highwood

S

Science Centre) D7

- R

N

E

- S

ENUE

[Figure 224]

P

P

Team C5

D

P22

P

 d

[Figure 225]

50 Estates and Facilities C4 Estates and Facilities

Blackstone

P24

- D
- E

Sherﬁeld Hall

135 Student Helpdesk Team C5 Carrington 10 Student Recruitment and

P10

P14

58 Meteorology D7 Meteorology 8 Meadow Suite C5 Park House 49 Miller D4 Miller

P

L

135 Examinations Of ce C5 Carrington 135 Faculty Of ces C5 Carrington

I

 j

W

U

E

N

Blandford Lodge

E

V

A

S

D

P23

N

Entrance 2

A

P8

L

D

O

Locate right section of the map showing the legend/index and the "Catering venues" list with numbered cafes.

O

W

Main Entrance

Outreach D4

Hollins

59 Farm Management Unit D8 Agriculture 195 Film, Theatre and Televison C5 Minghella

135 Student Services Centre C5 Carrington

[Figure 226]

Faringdon

P12

- 195 Minghella C5 Minghella

1 Modern Languages and European

Studies C5

HumSS

37 Muslim Centre D4 Archway Lodge 21 National Centre for

Biotechnology Education C8

TOB 2

58 National Centre for Earth

Observation D7

Meteorology

- 196 Nursery C5 Nursery 26 Palmer C5 Palmer

[Figure 227]

[Figure 228]

Entrance 3

P13

P

P11

To playingﬁelds

P

29 Students’ Union C5 Students’ Union Hub 2 Study Advice D5 Library (Room 103)

Pepper Lane

0 100m

[Figure 229]

P

E

27 Finance and Corporate

Whiteknights House

0 300ft

|Micr<br><br>the|obiology E6|
|---|---|
|Biolog Labo Bland|ical Sciences Teaching ratories C5<br><br>AMS Tower<br><br>ford Lodge D4 Blandford Lodge|
|Camp Enga<br><br>venues"|aigns and Supporter gement Of ce D4<br><br>Blandford Lodge<br><br>[Figure 230]|

Harris Garden

Services C4

###### Whiteknights campus

Tennis courts

Based on the description, the right section of the map includes legend/index and the “Catering venues” list with numbered cafes. The legend/index is located at right the of the map, and the "Catering venues" list is further down, under the "Catering ven heading. The bounding box should cover these areas.

D

800 Synthetic Turf Pitch B4 Synthetic Turf Pitch 38 Systems Engineering E5 Systems Engineering 25 Systems Engineering E5 Engineering

Based on the tool response, the right section of the map showing the legend/index and the "Catering venues" list with numbered cafes is already covered by the bounding box

A

O

0 200m

47 Food and Nutritional Sciences D6 Food Biosciences 84 Foxhill House A6 Foxhill House

R

P

S

 a

E

S

P

P

E

0 600ft

 h

E

N

Updated April 2013. © Copyright The University of Reading

R L A N E

R

###### 10 10

 b

- D
- E

InSight-o3 provided. Therefore, no further zooming is necessary.

L

- 49 Geography and Environmental Science D4

Miller

70 Geography and Environmental

Science D4

Russell

59 Graduate Institute of International Development and Applied Economics D8

Agriculture

9 Graduate School C4 Old Whiteknights House 27 Graduation Of ce C4 Whiteknights House

- 50 Grounds Maintenance Of ce C4 Estates and Facilities 34 Grounds Maintenance Yard B4 Estates and Facilities 31 Harborne E6 Harborne 56 Harry Pitt D7 Harry Pitt

I

W

3 Technologies for Sustainable Built Environments Centre (TSBE) D5

JJ Thomson

T

 g

F

P

P

55 Campus Central (shop) C5 Cedars Hotel and

 c

P

 a

Conference Centre 135 Careers, Placement and

P

21 Typography and Graphic

TOB 2

T

Communication C8

 f

Carrington

 b

P

8 Park House C5 Park House 135 Peer Support C5 Carrington

Experience Centre C5

 i

1 Ure Museum of Greek Archaeology C5

HumSS

 e

135 Carrington C5 Carrington 8 Catering –

 k

###### Round 5

27 Pensions Of ce D4 Whiteknights House

8 Venue Reading C5 Park House

Park House

P

Estates and Facilities C5

P

6 Pharmacy D6 Chemistry and Pharmacy 47 Pharmacy D6 Food Biosciences

59 Veterinary Epidemiology and Economic Research Unit (VEERU) D8

Agriculture

 d

55 Cedars Hotel and Conference

Cedars Hotel and

[Figure 231]

[Figure 232]

 j

Conference Centre 186 Central Kitchen E5 Central Kitchen

Centre C5

U

E

N

E

V

A

S

D

N

118 Pharmacy D6 Hopkins 1 Philosophy C5 HumSS 1 Politics and International

Entrance 2

A

L

D

O

O

W

Main Entrance

27 Vice-Chancellor’s Of ce C4 Whiteknights House

[Figure 233]

5 Wager D4 Wager 59 Walker Institute for Climate System Research D8

3 Centre for Advanced Microscopy

JJ Thomson

I need to read the catering venues list in the legend (bottom-right) to match the numbered cafe locations (like 202, 71, 138) to their names. I'll request a crop of the bottom-right area of the legend where the "Catering venues" and cafe names are listed.

HumSS

 CfAM  D5

Entrance 3

P

Agriculture

Relations C5

N15 Health and Dental Centre C2 Health and Dental Centre 3 Health and Safety Services D5 JJ Thomson 138 Henley Business School C5 Henley Business School

0 100m

SIDMOUTH STREET

GPT

59 Centre for Agricultural

Agriculture

Pedestrian only entrance

27 Postal Services C4 Whiteknights House 21 Pre-school C8 TOB 2

0 300ft

Strategy D8

Crown Place

27 Whiteknights House C4 Whiteknights House

[Figure 234]

LONDON ROAD (A4)

VEHICULAR ENTRANCE

59 Centre for Agri-Environmental

Agriculture

Town centre

[Figure 235]

8 Procurement C5 Park House 62 Psychology and Clinical Language

Research D8

1 History C5 HumSS 1 History of Art C5 HumSS 5 Human and Environmental

Catering venues

AD

RO

Psychology

Royal Berkshire Hospital

49 Centre for the Development of

Miller

Locate bottom-right corner of the legend showing the "Catering venues" section with the list of cafes and their numbers/names.

KENDRICK

Sciences D7

Teaching and Learning D4

Cafés 59 Agriculture snack bar D8 Agriculture 71 Café Index C4 ICMA Centre

Wager

Repton Loudon

201 Reading Enterprise Centre C8 Reading Enterprise Centre

138 Centre for Entrepreneurship C4 Henley Business School

Sciences D4

[Figure 236]

[Figure 237]

[Figure 238]

| | |
|---|---|
|Blagrave| |
| | |

26 Reading Film Theatre C5 Palmer 138 Reading Real Estate

56 Centre for Integrative Neuroscience and Neurodynamics (CINN) D7

Harry Pitt

27 Human Resources C4 Whiteknights House 1 Humanities and Social Sciences

[Figure 239]

Grey

[Figure 240]

Henley Business School

2 Café Libro D5 Library 26 Dol.cHe Vita C5 Palmer

HumSS

OILETS

AC

Addington

White Gates car park

Penn

Foundation C4

WN PL

(HumSS) C5

Buildings

Bus stop Telephones Halls of residence Path

Based on the tool response, the bounding box provided captures the "Catering venues" section with the list of cafes and their numbers/names. The coordinates are already correctly identified and

Based on the description, the "Catering venues" section is located at the bottom-right corner of the image. The section starts with the heading "Catering venues" and includes a list of cafes with their numbers and names. The bounding box should capture this section completely.

40 Chaplaincy Centre D5 Park House Lodge 62 Charlie Waller Institute D7 Psychology

46 Reading Scienti c Services Ltd

The Lord Zuckerman Research Centre

71 ICMA Centre C4 ICMA Centre 138 Informatics Research Centre C4 Henley Business School 54 Innovation Works C7 Science and Technology

Winchcombe

201 Eat at Enterprise C8 Reading Enterprise Centre

O

T

(RSSL) D6

REDL A

CR

T

Car parks Cafe

138 Real Estate and Planning C4 Henley Business School 27 Receipts Of ce C4 Whiteknights House 27 Research

P8

6 Chemical Analysis Facility

Chemistry and Pharmacy

InSight-o3 no further adjustment is needed.

1 Eat at HumSS C5 HumSS 39 Eat at SportsPark C3 SportsPark 54 Fusion Café C7 Science and Technology

Mitford Huntley

ND S

Highwood

 CAF  D6

Centre 1 In-sessional English Support Programme (ISLC) C5

RO AD

Blackstone

6 Chemistry D6 Chemistry and Pharmacy 1 Classics C5 HumSS

Whiteknights House

HumSS

P

and Enterprise C4

Centre

135 Room Bookings and

Carrington

62 Clinical Language Sciences D7 Psychology 30 Cole Museum of Zoology C5 AMS Tower 27 Communications Of ce C4 Whiteknights House 33 Construction Management and

10 Insurance Of ce D4 Blandford Lodge 118 Institute for Cardiovascular and

TOILETS

Hollins

Timetabling C5

Food and bar outlets

Faringdon

Hopkins

To playingﬁelds

70 Russell D4 Russell 54 Science and Technology

P

Metabolic Research (ICMR) D6

- 7 Eat at the Square D5 Eat at the Square

- 8 Eat and Drink at Park House C5 Park House

West cloister car park

Cafe

P

SIDMOUTH STREET

Science and Technology Centre

10 International Of ce D4 Blandford Lodge 135 International Student Advice C5 Carrington

Pedestrian only entrance

[Figure 241]

URS

Bus stop

[Figure 242]

Centre C7

Tennis courts

Crown Place

Engineering D4

202 Park Eat B4 Park Eat

LONDON ROAD (A4)

Taxis

27 Security Of ce C4 Whiteknights House

VEHICULAR ENTRANCE

Town centre

 a

25 Construction Management and

Engineering

- 1 ISLC (International Study and Language Centre) C5

HumSS

- 2 IT Helpdesk D5 Library

I located P2 on the campus map and zoomed into the surrounding area. The nearest coffee cup icon next to P2 is labeled 71. In the legend, 71 corresponds to "Café Index" (ICMA Centre). Therefore the closest cafe to P2 is Café Index (option D).

 h

8 Senior Common Room C5 Park House 76 Social Club (Employees) C8 TOB 2

Engineering E5

Restaurants 8 Blandford’s C5 Park House

Bike rack

AD

 b

RO

Royal Berkshire Hospital

135 Counselling and Wellbeing C5 Carrington

GPT Answer: \boxed{D. Café Index} ✓

KENDRICK

T

 g

P

P

 c

P

 a

P

|P| f|
|---|---|
| | |

0 50m

T

 b

OILETS

AC

 i

0 80ft

White Gates car park

WN PL

 e

 k

TOILETS

Acacia Road

FootpathtoKendrickRd.andbusstop

AcaciaRoadcarpark

| | |
|---|---|
| | |

O

REDL A

P

CR

Kendrick car park

P

MERL entrance

Figure 16: Example 1 (MAP case 1) of complete reasoning trace of our model (GPT-5-mini as vReasoner and INSIGHT-O3-VS as vSearcher). The vReasoner iteratively guides vSearcher through natural-language region descriptions. INSIGHT-O3-VS precisely retrieves high-quality image crops that match the described regions, effectively supporting vReasoner in locating the target area and producing the correct answer.

 d

ND S

 j

U

E

N

E

V

A

S

D

N

Entrance 2

A

L

D

O

O

W

RO AD

Main Entrance

Museum of English Rural Life

and Special Collections Service

Museum car park

Entrance 3

P

0 100m

TOILETS

Whiteknights campus

0 300ft

Westcloistercarpark

Cafe

Bus stop

Taxis

Bike rack

0 50m

0 80ft

TOILETS

Acacia Road

FootpathtoKendrickRd.andbusstop

AcaciaRoadcarpark

Kendrick car park

MERL entrance

Museum of English Rural Life

SIDMOUTH STREET

Pedestrian only entrance

and Special Collections Service

Museum car park

Crown Place

LONDON ROAD (A4)

VEHICULAR ENTRANCE

Town centre

Whiteknights campus

AD

RO

Royal Berkshire Hospital

37

KENDRICK

| | |
|---|---|
| | |

OILETS

AC

White Gates car park

WN PL

| | |
|---|---|
| | |

O

REDL A

CR

ND S

RO AD

TOILETS

West cloister car park

Cafe

Bus stop

Taxis

Bike rack

0 50m

0 80ft

TOILETS

Acacia Road

FootpathtoKendrickRd.andbusstop

AcaciaRoadcarpark

Kendrick car park

MERL entrance

Museum of English Rural Life

and Special Collections Service

Museum car park

Whiteknights campus

Whiteknights map key (numerical)

Whiteknights map key (A – Z)

Classics HumSS Economics HumSS English Language and Applied

Art TOB 1 Design and Print Studio TOB 2 National Centre for

Grounds Maintenance Of ce Estates and Facilities

Institute for Cardiovascular and Metabolic Research (ICMR)

Hopkins

3sixty bar Students’ Union

Crop Protection Unit Agriculture

IT Services Mathematics and IT Services

Southern Universities Management Services (SUMS)

Science and Technology Centre

|E<br><br>D<br><br>C<br><br>B<br><br>A<br><br>town cen and Mu|ALLCROFTROAD<br><br>Town centre and station<br><br>To<br><br>tre, Royal Berkshire Hosp seum of English Rural Lif<br><br>P*<br><br>Benyon Hall<br><br>Wan H<br><br>Hillside<br><br>White|NORTHC OU R T AV<br><br>ENUE<br><br>E<br><br>L<br><br>M<br><br>H<br><br>U<br><br>S<br><br>R E<br><br>D<br><br>L A N D S R O A<br><br>D<br><br>N<br><br>E<br><br><br>W<br><br>R<br><br>O<br><br>A<br><br>D<br><br>London Road<br><br>ital<br><br>P*<br><br>P*<br><br>Shinﬁeld Road<br><br>N   Health Centre<br><br>St Patrick’s Hall<br><br>Sherﬁeld Hall<br><br>tage all<br><br>St<br><br>knights|W<br><br>H<br>I<br><br><br>T<br><br>E<br><br>K<br><br>N<br><br>I<br><br>G<br>H<br><br><br>T<br><br>S R O A D<br><br>U<br><br>P<br><br>P<br><br>E<br><br>R<br><br>R<br><br>E<br><br>D<br><br>L<br><br>A<br><br>N<br><br>D<br><br>S<br><br>R<br><br>O<br><br>A<br><br>D<br><br>R<br>S<br>T<br><br><br>R<br><br>O<br><br>A<br><br>D<br><br>H<br>I<br><br><br>N<br><br>F<br><br>I<br><br>E<br><br>L D R O A D<br><br>Q<br><br>u<br><br>e<br><br>e<br><br>n<br><br>’<br><br>s<br><br>D<br><br>r<br><br>i<br><br>v<br><br>e<br><br>C<br><br>h<br><br>a<br><br>n<br><br>c<br><br>e<br><br>l<br><br>l<br><br>o<br><br>r<br><br>’<br><br>s<br><br><br>W<br><br>a<br><br>y<br><br>P8<br><br>P6<br><br>P<br><br>P7<br><br>P2<br><br>P<br><br>P*<br><br>P*<br><br>P*<br><br>P*<br><br>P*<br><br>P1 P P<br><br>P16<br><br>P5<br><br>P11<br><br>P14<br><br>P15<br><br>P4<br><br>P10<br><br>P9<br><br>P13<br><br>P12<br><br>P<br><br>P3<br><br>Pepper Lane<br><br>Pedestrian entrance<br><br>Sports grounds<br><br>Sports grounds<br><br>Foxhill<br><br>Harris Garden<br><br>Whiteknights Lake<br><br>Friends' Bridge<br><br>Mackinder Hall<br><br>Windsor Hall<br><br>George's Hall<br><br>Childs Hall<br><br>McCombie<br><br>Greenow<br><br>Stenton Hall & Townhouses<br><br>Dunsden Crescent<br><br>campus|Path Cafe<br><br>W O K I<br><br>W<br><br>H<br>I T<br><br><br>E K N<br><br>G<br>H T<br><br><br>S R O A D<br><br>R<br><br>O<br><br>A<br><br>D<br><br>W<br><br>To an<br><br>P* P*<br><br>P*<br><br>P*<br><br>P22<br><br>P22<br><br>P23<br><br>P<br><br>P*<br><br>P*<br><br>P24<br><br>Halls of residence<br><br>Car park restricted to users of this building<br><br>Car parks<br><br>One-way system<br><br>P8<br><br>P*<br><br>Under construction<br><br>Wessex Hall<br><br>Bridges Hall<br><br>(open 2014)<br><br>0|CHUR<br><br>CH<br><br>N<br><br>G<br>H A<br><br><br>M R O A D<br><br>wn centre d station<br><br>Earley Gate|
|---|---|---|---|---|---|
| | | | | |L<br><br>D<br>E<br><br><br>R<br><br>N<br><br>E<br><br>S<br><br><br>R<br><br>O<br><br>D<br><br>|
|F| | |P<br><br>E<br><br>P<br><br>P<br><br>E<br><br>R L A N E|W<br><br>I<br><br>L<br><br>D<br>E<br><br><br>R<br><br>N<br><br>E<br><br>S<br><br>S<br><br>0|600ft|

1 2 3 4 5 6 7 8 9

Biological Sciences Microbiology

Knight

Accommodation Of ce Carrington

Design and Print Studio TOB 2

JJ Thomson JJ Thomson

Town centre Town centre

Pharmacy Hopkins

Southern Universities Purchasing Consortium (SUPC)

Science and Technology Centre

HumSS

TOB 2

Admissions Of ce Miller

Disability Advisory Service Carrington

Job Shop (students) Students’ Union Hub Knowledge Transfer Centre Systems Engineering Law Foxhill House

Bus stop

Academic, administrative and service buildings

Innovation Works Science and Technology Centre

Accommodation Of ce Carrington

Linguistics

Biotechnology Education

Pedestrian entrance

Agriculture, Policy and Development

Agriculture

Economics HumSS

London Road campus,

Speech and Language Therapy Clinic

Psychology

English Literature HumSS History HumSS History of Art HumSS Humanities and Social Sciences

Pre-school TOB 2

Careers, Placement and Experience Centre

Carrington

English Language and Applied Linguistics

HumSS

Science and Technology Centre

Science and Technology Centre

Typography and Graphic Communication

TOB 2

Allen Laboratory The Allen Laboratory

Library Library

SportsPark SportsPark

Carrington Carrington Counselling and Wellbeing Carrington Disability Advisory Service Carrington Examinations Of ce Carrington Faculty Of ces Carrington International Student Advice Carrington Peer Support Carrington Room Bookings and

English Literature HumSS

Alumni Relations Blandford Lodge

Management (School of) Henley Business School

Southern Universities Management Services (SUMS)

Science and Technology Centre

Statistical Services Centre Harry Pitt

Construction Management and Engineering

Engineering

ESSC (Environmental Systems Science Centre)

Harry Pitt

HumSS

Archaeology Archaeology

Mathematics and Statistics Mathematics and IT Services

Student Financial Support Team

Carrington

(HumSS)

Art TOB 1

Mathematics Support Centre Library (S@iL)

Systems Engineering Engineering

Estates and Facilities Estates and Facilities

Southern Universities Purchasing Consortium (SUPC)

Science and Technology Centre

In-sessional English Support Programme (ISLC)

HumSS

Student Helpdesk Team Carrington

Athletics Pavilion Athletics Pavilion

Meteorology Meteorology

Palmer Palmer Reading Film Theatre Palmer Communications Of ce Whiteknights House

Examinations Of ce Carrington Faculty Of ces Carrington Farm Management Unit Agriculture

Student Recruitment and Outreach

Blandford Lodge

Biological Sciences Harborne

Meadow Suite Park House

Campus Central (shop) Cedars Hotel and Conference Centre

ISLC (International Study and Language Centre)

HumSS

Biological Sciences Hopkins

Miller Miller

Student Services Centre Carrington

Cedars Hotel and Conference Centre

Cedars Hotel and Conference Centre

Biological Sciences Philip Lyle

Modern Languages and European Studies

HumSS

Minghella Minghella

Finance and Corporate Services

Whiteknights House

Carrington

AD

Film, Theatre and Televison Minghella

Students’ Union Students’ Union Hub

Biological Sciences Microbiology

Knight

Modern Languages and European Studies

HumSS

Timetabling

RO

Finance and Corporate Services

Whiteknights House

Study Advice Library (Room 103) Synthetic Turf Pitch Synthetic Turf Pitch Systems Engineering Systems Engineering

Centre for Integrative Neuroscience and Neurodynamics (CINN)

Harry Pitt

Philosophy HumSS

Graduation Of ce Whiteknights House Human Resources Whiteknights House Pensions Of ce Whiteknights House Postal Services Whiteknights House Receipts Of ce Whiteknights House Research and Enterprise Whiteknights House Security Of ce Whiteknights House Vice-Chancellor’s Of ce Whiteknights House Whiteknights House Whiteknights House 3sixty bar Students’ Union

Student Financial Support Team

Carrington

Biological Sciences Teaching Laboratories

AMS Tower

Politics and International Relations

HumSS

Muslim Centre Archway Lodge

Food and Nutritional Sciences Food Biosciences

National Centre for Biotechnology Education

TOB 2

Student Helpdesk Team Carrington Student Services Centre Carrington

ESSC (Environmental Systems Science Centre)

Harry Pitt

Foxhill House Foxhill House

Blandford Lodge Blandford Lodge

Ure Museum of Greek Archaeology

HumSS

Systems Engineering Engineering

Geography and Environmental Science

Miller

Campaigns and Supporter Engagement Of ce

Blandford Lodge

National Centre for Earth Observation

Meteorology

Technologies for Sustainable Built Environments Centre (TSBE)

JJ Thomson

Harry Pitt Harry Pitt Statistical Services Centre Harry Pitt Meteorology Meteorology

Centre for Entrepreneurship Henley Business School Henley Business School Henley Business School Informatics Research Centre Henley Business School Management (School of) Henley Business School Reading Real Estate

IT Helpdesk Library Library Library Mathematics Support Centre Library (S@iL)

Geography and Environmental Science

Russell

Campus Central (shop) Cedars Hotel and Conference Centre

Nursery Nursery

Typography and Graphic Communication

TOB 2

Palmer Palmer

Graduate Institute of International Development and Applied Economics

Agriculture

National Centre for Earth Observation

Meteorology

Careers, Placement and Experience Centre

Carrington

Study Advice Library (Room 103)

Park House Park House Peer Support Carrington Pensions Of ce Whiteknights House

Ure Museum of Greek Archaeology

HumSS

Henley Business School

Centre for Advanced Microscopy (CfAM)

JJ Thomson

Agriculture, Policy and Development

Agriculture

Carrington Carrington

Foundation

Graduate School Old Whiteknights House

Venue Reading Park House

Catering – Estates and Facilities

Park House

Real Estate and Planning Henley Business School

Health and Safety Services JJ Thomson JJ Thomson JJ Thomson Technologies for Sustainable Built Environments Centre (TSBE)

Job Shop (students) Students’ Union Hub Students’ Union Students’ Union Hub Biological Sciences Teaching

Graduation Of ce Whiteknights House Grounds Maintenance Of ce Estates and Facilities Grounds Maintenance Yard Estates and Facilities

Centre for Agricultural Strategy

Agriculture

Pharmacy Chemistry and Pharmacy

Veterinary Epidemiology and Economic Research Unit (VEERU)

Agriculture

Central Kitchen Central Kitchen

Cedars Hotel and Conference Centre

Cedars Hotel and Conference Centre

Pharmacy Food Biosciences

Film, Theatre and Televison Minghella Minghella Minghella Nursery Nursery

JJ Thomson

Centre for Agri-Environmental Research

Agriculture

AMS Tower

Pharmacy Hopkins

| | |
|---|---|
|Central Kit|chen Central Kitchen|
| | |

Laboratories

Vice-Chancellor’s Of ce Whiteknights House

Harborne Harborne Harry Pitt Harry Pitt Health and Dental Centre Health and Dental Centre

Philosophy HumSS

Crop Protection Unit Agriculture Farm Management Unit Agriculture Graduate Institute of

Cole Museum of Zoology AMS Tower

Wager Wager

Centre for Advanced Microscopy  CfAM 

JJ Thomson

IT Services Mathematics and IT Services Mathematics and Statistics Mathematics and IT Services Human and Environmental

Politics and International Relations

HumSS

Biological Sciences Harborne Harborne Harborne Biological Sciences Philip Lyle Construction Management and

Reading Enterprise Centre Reading Enterprise Centre

Walker Institute for Climate System Research

Agriculture

Agriculture

Synthetic Turf Pitch Synthetic Turf Pitch

Centre for Agricultural Strategy

Agriculture

Postal Services Whiteknights House

Health and Safety Services JJ Thomson

Wager

International Development and Applied Economics

Health and Dental Centre Health and Dental Centre

Whiteknights House Whiteknights House

Sciences

Pre-school TOB 2

Henley Business School Henley Business School

Centre for Agri-Environmental Research

Agriculture

URS

Wager Wager

Veterinary Epidemiology and Economic Research Unit (VEERU)

Agriculture

Procurement Park House

History HumSS History of Art HumSS Human and Environmental

Catering venues

Engineering

Catering venues

Chemical Analysis Facility (CAF)

Chemistry and Pharmacy

Psychology and Clinical Language Sciences

Psychology

Centre for the Development of Teaching and Learning

Miller

Grounds Maintenance Yard Estates and Facilities

Wager

Muslim Centre Archway Lodge

Walker Institute for Climate System Research

Agriculture

Eat at HumSS

Chemistry Chemistry and Pharmacy Pharmacy Chemistry and Pharmacy Catering – Estates and

Reading Enterprise Centre Reading Enterprise Centre

Agriculture snack bar

Centre for Entrepreneurship Henley Business School

Sciences

Knowledge Transfer Centre Systems Engineering Systems Engineering Systems Engineering SportsPark SportsPark

Café Libro

Reading Film Theatre Palmer

Café Index Café Libro Dol.cHe Vita

Centre for Integrative Neuroscience and Neurodynamics (CINN)

Harry Pitt

Human Resources Whiteknights House

Charlie Waller Institute Psychology Clinical Language Sciences Psychology Psychology and Clinical Language

Dol.cHe Vita

Park House

Reading Real Estate Foundation

Henley Business School

Humanities and Social Sciences (HumSS)

HumSS

Facilities

Eat at SportsPark

Psychology

Chaplaincy Centre Park House Lodge

Chaplaincy Centre Park House Lodge Athletics Pavilion Athletics Pavilion Reading Scienti c Services Ltd

Meadow Suite Park House Park House Park House Procurement Park House Senior Common Room Park House Venue Reading Park House Graduate School Old Whiteknights House

Reading Scienti c Services Ltd (RSSL)

The Lord Zuckerman Research Centre

ICMA Centre ICMA Centre

Fusion Café

Sciences

Eat at Enterprise

200m

Charlie Waller Institute Psychology

Informatics Research Centre Henley Business School

Speech and Language Therapy Clinic

Psychology

Real Estate and Planning Henley Business School

Chemical Analysis Facility  CAF 

Chemistry and Pharmacy

The Lord Zuckerman

Agriculture snack bar

Eat at HumSS

Innovation Works Science and Technology Centre

Research Centre Food and Nutritional Sciences Food Biosciences Pharmacy Food Biosciences Allen Laboratory The Allen Laboratory

(RSSL)

Receipts Of ce Whiteknights House

Café Index

Geography and Environmental Science

Russell

Eat at SportsPark

Chemistry Chemistry and Pharmacy

Research and Enterprise

Whiteknights House

In-sessional English Support Programme (ISLC)

HumSS

Eat at Enterprise

Fusion Café

Classics HumSS

Russell Russell

Room Bookings and Timetabling

Carrington

Clinical Language Sciences Psychology Cole Museum of Zoology AMS Tower Communications Of ce Whiteknights House

Insurance Of ce Blandford Lodge

ICMA Centre ICMA Centre Archaeology Archaeology Social Club (Employees) TOB 2

Alumni Relations Blandford Lodge Blandford Lodge Blandford Lodge Campaigns and Supporter

Admissions Of ce Miller

Institute for Cardiovascular and Metabolic Research (ICMR)

Hopkins

Eat at the Square

Russell Russell

Eat at the Square

Centre for the Development of Teaching and Learning

Miller

Blandford Lodge

Eat and Drink at Park House

Science and Technology Centre

Science and Technology Centre

International Of ce Blandford Lodge

Eat and Drink at Park House

Construction Management and Engineering

URS

Engagement Of ce

Foxhill House Foxhill House Law Foxhill House Biological Sciences Hopkins

Park Eat

Geography and Environmental Science

Miller

International Student Advice Carrington

Park Eat

Insurance Of ce Blandford Lodge International Of ce Blandford Lodge Student Recruitment and

Security Of ce Whiteknights House

Construction Management and Engineering

Engineering

ISLC (International Study and Language Centre)

HumSS

Senior Common Room Park House

Miller Miller

Blandford Lodge

Blandford’s

Social Club (Employees) TOB 2

Blandford’s

Counselling and Wellbeing Carrington

Estates and Facilities Estates and Facilities

IT Helpdesk Library

Outreach

###### Question

###### Explanation

If you are currently at P2 Car park, which is near B4, and want to find a nearby place for coffee, which cafe is the closest?

- Step 1. Locate P2 Car park near B4 on the map .
- Step 2. Check the building list in the bottom right corner under Catering venues for the listed cafes: Cafe Libro (2, D5), Eat at SportsPark (39, C3), Agriculture snack bar (59, D8), Cafe Index (71, C4), Fusion Cafe (54, C7).
- Step 3. Identify the closest cafe, which is building 71 , Cafe Index, and choose option D.

Repton Loudon

- A. Cafe Libro
- B. Eat at SportsPark ✅

C. Agriculture snack bar D. Cafe Index

- E. Fusion Cafe
- F. No right choice

Grey

[Figure 243]

Addington

Penn

Buildings

Bus stop Telephones Halls of residence Path

Winchcombe

T

Blagrave

T

Car parks Cafe

P8

Mitford Huntley

Highwood

Blackstone

P

[Figure 244]

###### GPT-5-Mini

Hollins

Faringdon

To playingﬁelds

P

P

[Figure 245]

Tennis courts

 a

 h

[Figure 246]

 b

I first locate P2 car park on the map grid near B4. I then look for nearby café icons and labels in that area. The closest café symbol and label adjacent to the P2 car park sports grounds area is the one at the SportsPark building labeled "Eat at SportsPark." Therefore the nearest place for coffee from P2 (near B4) is Eat at SportsPark.

1 2 3 4 5 6 7 8 9

✘

T

 g

P

P

Town centre

 c

P

 a

P

###### GPT ✘

T

 f

 b

P

Bus stop

Academic, administrative and service buildings

 i

Answer: \boxed{Eat at SportsPark}

Pedestrian entrance

 e

Town centre

 k

London Road campus, town centre, Royal Berkshire Hospital and Museum of English Rural Life

P

P

Path Cafe

Halls of residence

 d

 j

P*

U

E

N

E

V

A

S

D

N

Entrance 2

A

L

D

O

O

W

W

Main Entrance

- H
- I

T

E

K

###### Under construction

N

I

- G
- H

Entrance 3

Pedestrian entrance

P

[Figure 247]

[Figure 248]

0 100m

Wantage Hall

T

GPT-5-Mini

S R O A D

Highquality Crop

Mediumquality Crop

Lowquality Crop

0 300ft

[Figure 249]

[Figure 250]

[Figure 251]

Car parks

[Figure 252]

[Figure 253]

P8

[Figure 254]

N

A

D

L

D

S

E

R

R

O

R

E

A

P

P

D

U

Qwen2.5-VL vSearcher

Region Description

ALLCROFTROAD

A

vReasoner

St George's Hall

Childs Hall

Car park restricted to users of this building

P*

P*

###### Round 1

P*

Foxhill

Hillside

One-way system

P*

Dunsden Crescent

P*

Greenow

To

R E

[Figure 255]

[Figure 256]

London Road

D

Town centre and station

Mackinder Hall

D

A

McCombie

O

L A N D S R O A

Whiteknights Lake

R

AD

I need a close-up of the map area around grid B4 where P2 car park is located, to identify nearby cafes. I'll request a crop around the upper-left section (around B4) showing P2 and surrounding labels and icons.

W

- D

N

- E

D

A

P*

RO

GPT

O

SIDMOUTH STREET

Windsor Hall

P*

Pedestrian only entrance

Stenton Hall & Townhouses

R

W O

Wessex Hall

Sports grounds

Crown Place

CH

- R
- S
- T

B

LONDON ROAD (A4)

P*

[Figure 257]

[Figure 258]

VEHICULAR ENTRANCE

Town centre

P*

[Figure 259]

K I N

AD

W

U

CHUR

RO

Royal Berkshire Hospital

H

KENDRICK

[Figure 260]

- H
- I T

M

- G
- H A

P2

Locate P2 car park area around grid B4, upper-left section showing parking P2, nearby buildings, paths and cafe icons/labels.

L

E

Town centre and station

P3

M R O A D

E K N I

P

[Figure 261]

OILETS

AC

White Gates car park

[Figure 262]

WN PL

P16

- G
- H T

REDL ND RO AD

CR

Bridges Hall

Based on the region description provided, the bounding box coordinates for the P2 car park area around grid B4, upper-left section showing parking P2, Qwen2.5-VL nearby buildings, paths, and cafe icons/labels are: <bbox coordinates>

e

v

i

r

D

s

’

n

(open 2014)

e

P P

e

P1

u

Q

S R O A D

Shinﬁeld Road

Benyon Hall

Earley Gate

N  

West cloistercarpark

Sports grounds

Health Centre

Cafe

P4

C

Bus stop

Taxis

P*

###### Round 2

St Patrick’s Hall

S N F I E L D R O A D

Bike rack

- H
- I

P5

P*

NORTHC OU R T AV

P*

P*

C

h

a

n

c

e

l

l

o

- r

’

- s

[Figure 263]

W

0 50m

a

[Figure 264]

y

0 80ft

P

D

Acacia Road

FootpathtoKendrickRd.andbusstop

AcaciaRoadcarpark

P9

The provided zoomed crop shows only the top title, not the P2 car park. I need a crop that includes the P2 parking label and nearby cafe icons/labels in the leftmiddle of the map (around grid B4). I'll request a focused crop of that area showing P2 and surrounding buildings/cafes.

Friends' Bridge

A

Kendrick car park

[Figure 265]

MERL entrance

P22

P6

P

O

P15

GPT

R

P*

Museum of English Rural Life

P7

S

and Special Collections Service

Museum car park

- R

N

E

- S

[Figure 266]

ENUE

[Figure 267]

P

Whiteknights campus

D

P22

P24

Locate P2 car park and immediate surroundings, left-middle of map around grid B4, showing parking label P2, nearby buildings, paths and cafe icons/labels.

- D
- E

Sherﬁeld Hall

P10

P14

L

[Figure 268]

I

W

[Figure 269]

P23

P8

Based on the region description provided, the bounding box coordinates for the "P2 car park and immediate surroundings, left-middle of map around grid Qwen2.5-VL B4" are approximately: <bbox coordinates>

P12

P13

P11

Pepper Lane

E

###### Round 3

Harris Garden

#### Whiteknights campus

D

A

O

0 200m

[Figure 270]

R

[Figure 271]

P

S

E

S

P

P

E

0 600ft

E

N

Updated April 2013. © Copyright The University of Reading

R L A N E

The last crop shows P8/P11 area near Pepper Lane, not P2. I still need the exact P2 location and nearby cafe icons/names. I'll request a crop focused on the upper-left/mid-left map where P2 (labelled "P2" or a parking icon) and surrounding cafes in grid B4 would appear. This should include any nearby cafe names/icons (e.g., Cafe Libro, Cafe Index, Fusion Cafe, Agriculture snack bar, Eat at SportsPark) to identify the closest one to P2.

R

- D
- E

GPT

L

I

W

[Figure 272]

[Figure 273]

F

Locate P2 car park label and immediate surroundings in grid B4, mid-left/upper-left area of the map showing P2 parking icon and nearby cafes and building labels.

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

<bbox coordinates>

Qwen2.5-VL

###### Round 4

[Figure 278]

[Figure 279]

The previous crops missed the P2 area. I need a clear close-up that includes the label "P2" (parking) and nearby cafe icons/names around the upper-left/midleft of the campus (near Allcroft Road/Shinfield Road area). I'll request a focused crop that clearly shows the P2 parking marker and adjacent buildings/cafés so I can identify the closest cafe to P2.

GPT

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Locate upper-left campus area around Allcroft Road and Shinfield Road showing parking label "P2" and nearby buildings/cafe icons.

[Figure 284]

[Figure 285]

<bbox coordinates>

Qwen2.5-VL

###### Round 5

Repton Loudon

[Figure 286]

[Figure 287]

Grey

[Figure 288]

The previous targeted crops missed the P2 area. I'll request a larger crop covering the entire upper-left quadrant (top-left quarter) of the map so the P2 parking label, nearby roads (Allcroft/Shinfield), and any cafe icons/names in that area are visible. This larger view should include grid B4 and surrounding labels to identify the closest cafe to P2.

Addington

Penn

Buildings

Bus stop Telephones Halls of residence Path

GPT

1 2 3 4 5 6 7 8 9

Town centre

Winchcombe

Bus stop

Academic, administrative and service buildings

Pedestrian entrance

Town centre

[Figure 289]

[Figure 290]

T

London Road campus, town centre, Royal Berkshire Hospital and Museum of English Rural Life

Path Cafe

Halls of residence

Blagrave

P*

W

- H
- I

T

E

K

###### Under construction

N

I

Locate entire upper-left quadrant of the campus map (top-left quarter) showing Allcroft Road, Shinfield Road, and car parks including P2 with surrounding buildings and cafe icons/labels

- G
- H

Pedestrian entrance

T

Car parks Cafe

Wantage Hall

T

P8

S R O A D

Car parks

P8

N

A

D

L

D

S

E

R

R

O

R

E

A

P

P

D

U

ALLCROFTROAD

A

Mitford Huntley

Highwood

St George's Hall

Childs Hall

Car park restricted to users of this building

P*

P*

[Figure 291]

[Figure 292]

P*

Foxhill

Hillside

One-way system

P*

Dunsden Crescent

P*

Greenow

To

Blackstone

R E

London Road

D

Town centre and station

Mackinder Hall

D

A

McCombie

O

L A N D S R O A

<bbox coordinates>

Whiteknights Lake

R

AD

P

W

- D

N

- E

D

Qwen2.5-VL

A

P*

RO

O

Windsor Hall

P*

Stenton Hall & Townhouses

R

W O

Wessex Hall

Sports grounds

CH

- R
- S
- T

B

P*

P*

K I N

W

U

CHUR

H

- H
- I T

M

- G
- H A

P2

L

E

Town centre and station

P3

M R O A D

E K N I

P

Hollins

P16

- G
- H T

Bridges Hall

[Figure 293]

e

v

i

r

D

Faringdon

s

’

[Figure 294]

n

(open 2014)

e

P P

e

P1

u

Q

S R O A D

Shinﬁeld Road

Benyon Hall

Earley Gate

To playingﬁelds

P

N  

Sports grounds

Health Centre

P4

C

P2 is by Allcroft Road/Wantage Hall in grid B4. The nearest cafe icon to that parking area corresponds to Cafe Libro. Answer: \boxed{Cafe\ Libro} ✘

P*

St Patrick’s Hall

P

S N F I E L D R O A D

- H
- I

P5

P*

NORTHC OU R T AV

P*

GPT

P*

C

h

a

n

c

e

l

l

o

- r

’

- s

W

a

y

Tennis courts

P

D

P9

Friends' Bridge

A

P22

P6

P

O

P15

R

P*

P7

S

- R

N

E

- S

ENUE

P

 a

D

P22

P24

- D
- E

Sherﬁeld Hall

 h

P10

P14

L

I

W

 b

- Figure 17: Example 1 (MAP case 1, continuation of Fig. 16): reasoning traces of GPT-5-mini and GPT-5-mini + Qwen2.5-VL-7B-Instruct. GPT-5-mini exhibits misperception and reasoning drift, while Qwen2.5-VL-7B-Instruct frequently fails to follow vReasoner’s instructions, producing lowquality crops misaligned with the described regions. Consequently, both baselines yield incorrect answers.

P23

P8

P12

P13

P11

Pepper Lane

T

E

 g

Harris Garden

P

###### Whiteknights campus

P

D

A

O

0 200m

 c

R

P

 a

P

S

E

S

P

P

P

E

0 600ft

E

N

Updated April 2013. © Copyright The University of Reading

R L A N E

R

- D
- E

L

I

T

W

 f

 b

P

F

 i

 e

 k

P

P

 d

 j

U

E

N

E

V

A

S

D

N

Entrance 2

A

L

D

O

O

W

Main Entrance

Entrance 3

P

0 100m

38

Repton Loudon

0 300ft

Grey

Addington

Penn

Buildings

Bus stop Telephones Halls of residence Path

Winchcombe

T

Blagrave

T

Car parks Cafe

P8

Mitford Huntley

Highwood

Blackstone

P

Hollins

Faringdon

To playingﬁelds

P

P

Tennis courts

 a

 h

 b

T

 g

P

P

 c

P

 a

P

T

 f

 b

P

 i

 e

 k

P

P

 d

 j

U

E

N

E

V

A

S

D

N

Entrance 2

A

L

D

O

O

W

Main Entrance

Entrance 3

P

0 100m

0 300ft

SIDMOUTH STREET

Pedestrian only entrance

Crown Place

LONDON ROAD (A4)

VEHICULAR ENTRANCE

Town centre

AD

RO

Royal Berkshire Hospital

KENDRICK

| |Town centre|
|---|---|
| | |

SIDMOUTH STREET

Pedestrian only entrance

Crown Place

LONDON ROAD (A4)

ACE

VEHICULAR ENTRANCE

re

TOILETS

White Gates car park

AD

WN PL

RO

Royal Berkshire Hospital

KENDRICK

| | |
|---|---|
| | |

O

| | |
|---|---|
| | |

REDL A

CR

OILETS

AC

White Gates car park

WN PL

ND S

O

REDL A

CR

ND S

RO AD

RO AD

TOILETS

TOILETS

West cloister car park

Cafe

Bus stop

Taxis

West cloister car park

Cafe

Bike rack

Bus stop

0 50m

Taxis

0 80ft

TOILETS

Acacia Road

FootpathtoKendrickRd.andbusstop

AcaciaRoadcarpark

Bike rack

Kendrick car park

MERL entrance

Museum of English Rural Life

and Special Collections Service

Museum car park

Whiteknights campus

0 50m

0 80ft

TOILETS

Acacia Road

FootpathtoKendrickRd.andbusstop

AcaciaRoadcarpark

Kendrick car park

MERL entrance

Museum of English Rural Life

and Special Collections Service

Museum car park

Whiteknights campus

[Figure 295]

###### Question

Assuming you live at UCD Village, which pedestrian-accessible campus entrance closest to your residence is open at 1:00 AM on Monday when returning from off-campus?

[Figure 296]

C. Roebuck Castle Gate ✅ D. Merville House Entrance

- A. Ownestown Park Entrance
- B. Roebuck Wicket Gate

- E. Newstead Gate
- F. No right choice

###### Explanation

- Step 1. From , locate UCD Village (Building 79) at Grid E10 .
- Step 2. Check for gate opening hours and confirm that both Roebuck Castle Gate and Greenfield Park Gate are pedestrian-accessible and open at 1:00 AM.
- Step 3. On the map, compare their distances to Building 79: Roebuck Castle Gate (H11) is closer than Greenfield Park (C4) . Answer: C

[Figure 297]

###### GPT-5-Mini

[Figure 298]

[Figure 299]

I first looked for the gate-opening times printed on the map (bottom-right section). The list of gate opening times shows which pedestrian entrances are open 24 hours. I then located the entrances near UCD Village on the campus map (east side of the map) and checked which of those nearby gates is listed as open at 01:00 (i.e. shown as 24 hours). Merville House Entrance is indicated as a 24-hour pedestrian entrance, and it is the nearest 24-hour pedestrian-

GPT

✘

accessible campus entrance to the UCD Village residences.

✘

Answer: \boxed{Merville House Entrance}

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

High-quality Crop

Low-quality Crop

Zoom-in Tool

Region Description

vReasoner

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

InSight-o3 vSearcher

Qwen2.5-VL GPT-5-Mini vSearcher

Medium-quality Crop

###### Round 1

[Figure 310]

[Figure 311]

I need to read the gate opening times/details (likely in the text box at the bottom-right of the map) to know which entrances are open at 1:00 AM on Monday and find the one nearest UCD Village. I'll crop the bottom-right area of the map where the contact info and gate opening times are located.

GPT

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Locate bottom-right corner text box with "Gate Opening Times" and contact info.

[Figure 316]

[Figure 317]

Qwen2.5-VL <bbox coordinates>

###### Round 2

[Figure 318]

[Figure 319]

I need to confirm which entrance is physically closest to UCD Village on the map (likely in the lower-left/southwest area). I'll crop the lowerleft portion of the campus map showing UCD Village and nearby entrances (Newstead, Ownestown Park, Roebuck gates, Merville House area).

GPT

[Figure 320]

[Figure 321]

Locate lower-left area of the main campus map showing UCD Village residences and nearby entrance labels (Newstead, Ownestown Park, Roebuck area).

[Figure 322]

[Figure 323]

✘

Qwen2.5-VL Target not found in the image.

[Figure 324]

[Figure 325]

The zoomed schedule shows Newstead Gate and Roebuck Castle Gate listed as 24 hours, while Merville House closes at 00:00 and the Roebuck (wicket) pedestrian entrance is only 07:00–18:00. UCD Village is located in the lower-left of the campus map, adjacent to the Newstead entrance, so the closest pedestrian entrance open at 1:00 AM Monday is Newstead Gate.

✘

GPT

Answer: \boxed{E. Newstead Gate}

[Figure 326]

###### GPT-5-Mini InSight-o3-vS

###### Round 1

[Figure 327]

[Figure 328]

[Figure 329]

I need to read the gate opening times legend, which is likely in the lower-right corner of the full map. I'll crop the bottom-right area that contains the timetable/contact info and gate opening hours.

[Figure 330]

GPT

[Figure 331]

[Figure 332]

Locate bottom-right corner of the map showing legend, contact info, and gate opening times.

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

Based on the provided bounding box, the bottomright corner of the map showing the legend, contact info, and gate opening times is accurately captured. The tool response confirms this area is zoomed in correctly.

Based on the description, the bottom-right corner of the map includes the legend, contact information, and gate opening times. The legend is located at the bottom left, and the contact information and gate opening times are on the right side of the bottom part of the image. The bounding box should encompass these areas.

InSight-o3

###### Round 2

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

I need to see the entrances on the right side of the campus near UCD Village (Roebuck, Ownestown, Merville, Newstead) and check the "Gates Opening Times" list in the legend that shows hours for each named entrance. I'll crop the mid-right section of the map showing the Roebuck/Ownestown/Merville area (east side) and the small gate icons, then if needed I'll zoom into the gates-opening-times text in the legend (right side).

GPT

[Figure 342]

[Figure 343]

Locate mid-right area of map showing Roebuck Castle Gate, Roebuck Wicket Gate, Ownestown Park Entrance, Merville House Entrance, and nearby paths (east side)

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

Based on the provided bounding box, the map area seems to cover the mid-right section of the map, including the Roebuck Castle Gate, Roebuck Wicket Gate, Ownestown Park Entrance, Merville House Entrance, and nearby paths. The zoomed-in area appears to match the description and requirements of the task.

Based on the description, the area of interest is located in the mid-right section of the map, specifically around the Roebuck Castle Gate, Roebuck Wicket Gate, Ownestown Park Entrance, Merville House Entrance, and nearby paths. The bounding box should capture these landmarks and the surrounding paths.

InSight-o3

###### Round 3

[Figure 349]

[Figure 350]

I need to read the "Gates Opening Times" text in the legend (right side) to see which entrances are open at 01:00 on Monday. I'll crop the right portion of the legend where the gates and opening times are listed.

GPT

[Figure 351]

[Figure 352]

[Figure 353]

Locate right side of the legend box showing "Gates Opening Times" and the list of entrances with hours

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

Based on the tool response, the "Gates Opening Times" section is already identified and the coordinates provided in the bbox_2d should capture this area. The task is complete as the required section is already zoomed in.

Based on the description, the "Gates Opening Times" section is located on the right side of the image, below the "Campus Information" section. The coordinates provided in the bbox_2d should capture this area.

InSight-o3

[Figure 360]

[Figure 361]

The gates schedule shows Ownestown (Ownestown Entrance) and Merville House entrances close at midnight, while Roebuck Castle Gate and Newstead Gate are listed as 24 hours. On the map UCD Village is adjacent to the Roebuck/Ownestown area, and the closest 24-hour pedestrian entrance to UCD Village is Roebuck Castle Gate. Therefore the closest pedestrian entrance that is open at 1:00 AM Monday is Roebuck Castle Gate (option C).

Answer: \boxed{C} ✓

GPT

- Figure 18: Example 2 (MAP case 2): reasoning traces of our model, GPT-5-mini, and GPT-5-mini + Qwen2.5-VL-7B-Instruct. Our INSIGHT-O3-VS accurately follows vReasoner’s instructions and returns high-quality crops aligned with the described regions, leading to a correct answer. In contrast, Qwen2.5-VL-7B fails to return a valid crop in the final reasoning round, resulting in an incorrect answer.

[Figure 362]

###### Question

Using the dashboard, define P = (2024 Revenue shown in the Income Statement table) − (2024 Revenue shown in the Income Statement chart), Q = (2024 Total Assets shown in the Balance Sheet table) − (2024 Total Assets shown in the Balance Sheet chart). What is P + Q?

- A. 4879k
- B. 817k

C. 942k D. 755k

- E. 3318k
- F. None of above

[Figure 363]

✅

###### Explanation

- Step 1. Locate Revenue in the Income Statement — the table and the chart . For 2024, the Revenue reported in the table is

- 2,315,250 (2315k), while in the chart, it is 1997k. Thus, P = 2315 − 1997 = 318k.

Step 2. Locate Total Assets in the Balance Sheet — the table and the chart . For 2024, the Total Assets reported in the table is

- 3,814,162 (3814k), while in the chart, it is 3377k. Thus, Q = 3814 − 3377 = 437k.

- Step 3. Calculate P + Q = 318 + 437 = 755k. Answer: D

[Figure 364]

###### GPT-5-Mini

[Figure 365]

- Step 1 — Identify where to read the 2024 values: - The Income Statement table is in the lower-left of the dashboard section. - The Income Statement chart (bars) is immediately to the right of that table. - The Balance Sheet table is just below the Income Statement table; the Balance Sheet chart is to the right of that (bottom-right of the dashboard).
- Step 2 — Read the 2024 Revenue values: - From the Income Statement table the 2024 Revenue is 2,563k (as shown in the table column for 2024). - From the Income Statement chart the 2024 Revenue is 1,721k (value shown above the 2024 bar). - Compute P = 2,563k − 1,721k = 842k.

- Step 3 — Read the 2024 Total Assets values: - From the Balance Sheet table the 2024 Total Assets is 3,500k (as shown in the table column for 2024). - From the Balance Sheet chart the 2024 Total Assets is 100k less, shown as 758k (value on the 2024 bar). [Note: the difference seen is 100k] - Compute Q = 100k (the visible difference).

- Step 4 — Sum P + Q: - P + Q = 842k + 100k = 942k. Therefore the best matching option is C (942k). Answer: \boxed{942k}

[Figure 366]

✘

✘

✘

GPT

✘

✘

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

High-quality Crop

Low-quality Crop

Region Description

Zoom-in Tool

vReasoner

[Figure 372]

[Figure 373]

###### GPT-5-Mini

[Figure 374]

[Figure 375]

[Figure 376]

InSight-o3 vSearcher

Qwen2.5-VL vSearcher

Medium-quality Crop

###### Round 1

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

I need to read the 2024 Revenue value from the Income Statement table. I'll zoom into the financial summary tables at the bottom left where the Income Statement (extract) is located.

GPT

[Figure 381]

[Figure 382]

###### Locate bottom-left financial tables area showing the Income Statement table (lower-left quadrant).

[Figure 383]

[Figure 384]

The region description "bottom-left financial tables area showing the Income Statement table (lower-left quadrant)" refers to the area in the bottom-left corner of the image, which contains the "Income Statement (Extract)" table. The bounding box coordinates for this region are approximately: <bbox coordinates>

Qwen2.5-VL

###### Round 2

[Figure 385]

[Figure 386]

[Figure 387]

I need to read the 2024 Revenue value shown in the Income Statement chart (the bar chart on the right). I'll request a crop of the Income Statement chart area (right side of the financial summary showing the Revenue bars for each year).

[Figure 388]

GPT

[Figure 389]

[Figure 390]

Income Statement chart area on the right side showing bars for Revenue 2022-2026 (upper-right of the financial summary)

[Figure 391]

[Figure 392]

The region description you provided is for the "Income Statement chart area on the right side showing bars for Revenue 2022-2026 (upper-right of the financial summary)." Based on the image, the chart that matches this description is the one located in the upper-right corner of the image, which shows the "Income

Qwen2.5-VL Statement" with bars for Revenue 2022-2026. The bounding box coordinates for this chart are approximately: <bbox coordinates>

###### Round 3

[Figure 393]

[Figure 394]

I need to read the 2024 Total Assets value shown on the Balance Sheet chart (the bar for 2024) to compute Q. I'll request a crop of the lower portion of the financial summary that contains the Balance Sheet chart bars and their numeric labels (directly below the Income Statement chart).

GPT

[Figure 395]

[Figure 396]

Locate area directly below the Income Statement chart showing the full Balance Sheet chart bars and numeric labels for Total Assets for years 2022-2026 (lowercenter/right of the summary)

[Figure 397]

[Figure 398]

✘

The region description you provided is quite specific, and it seems to be referring to a part of the image that is not present in the provided image. The image you've shared does not contain a "Balance Sheet chart" or "numeric labels for Total Assets for years 2022-2026" in the area described. The area you're

Qwen2.5-VL asking about appears to be in a different section of a different document or image.

[Figure 399]

[Figure 400]

From the Income Statement table the 2024 Revenue is 2,315,250 EUR; from the Income Statement chart the 2024 Revenue is 1,997k EUR (1,997,000). So P = 2,315,250 − 1,997,000 = 318,250 (~318k). From the Balance Sheet table the 2024 Total Assets is 3,814,162 EUR; from the Balance Sheet chart the 2024 Total Assets is about 3,315k EUR (3,315,000). So Q ≈ 3,814,162 − 3,315,000 = 499,162 (~499k). P + Q ≈ 318k + 499k = 817k. The matching answer is 817k.

✘

Answer: \boxed{817k} ✘

GPT

[Figure 401]

###### GPT-5-Mini InSight-o3-vS

[Figure 402]

###### Round 1

[Figure 403]

[Figure 404]

[Figure 405]

I need to read the numeric values for 2024 Revenue from the Income Statement table and the Income Statement chart, and similarly the 2024 Total Assets values from the Balance Sheet table and its chart. These items are in the bottom section of the dashboard: the left shows the detailed tables and the right shows the charts. I'll first request a zoom of the bottom-left region that contains the Income Statement and Balance Sheet tables.

GPT

[Figure 406]

[Figure 407]

Locate bottom-left region containing the Income Statement table and Balance Sheet table (bottom area, left of the charts).

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

Based on the description, the bottom-left region of the image contains the Income Statement table and Balance Sheet table, which are located in the lower-left corner of the image. The Income Statement table is above the Balance Sheet table, and both are situated in the bottom-left area of the image, to the left of the charts.

Based on the response from the tool, the bottom-left region containing the Income Statement table and Balance Sheet table has been successfully identified and zoomed in. The coordinates provided in the tool response match the area

InSight-o3 described in the question, so no further action is needed.

[Figure 413]

###### Round 2

[Figure 414]

[Figure 415]

[Figure 416]

The table shows 2024 Revenue = 2,315,250 (from column 31-Dec-24). I need the 2024 Revenue value shown in the Income Statement chart on the right (bar chart) to compute P. That chart is located to the right of the tables in the bottom-right area. I'll request a crop of the bottom-right region containing the Income Statement and Balance Sheet charts to read the 2024 bar values (Revenue and Total Assets).

GPT

[Figure 417]

[Figure 418]

Locate bottom-right region containing the Income Statement chart (bar chart) and the Balance Sheet chart with Total Assets bars (bottom area, right of the tables).

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

Based on the description, the bottom-right region of the image contains the Income Statement chart (bar chart) and the Balance Sheet chart with Total Assets bars. The Income Statement chart is a bar chart showing revenue, net profit, and net profit margin, while the Balance Sheet chart shows Total Assets, Total Liabilities, and Borrowings to Assets. The Total Assets section is located at the bottom of the Balance Sheet section.

Based on the tool response, the bottom-right region containing the Income Statement chart and the Balance Sheet chart with Total Assets bars has been successfully identified and zoomed in. The coordinates provided in the tool response match the described region, so no further action is needed.

InSight-o3

[Figure 424]

[Figure 425]

The Income Statement table shows 2024 Revenue = 2,315,250 (from the table). The Income Statement chart shows 2024 Revenue ≈ 1,997k (1,997,000). So P = 2,315,250 − 1,997,000 = 318,250. The Balance Sheet table shows 2024 Total Assets = 3,814,162. The Balance Sheet chart shows 2024 Total Assets ≈ 3,377k (3,377,000). So Q = 3,814,162 − 3,377,000 = 437,162. P + Q = 318,250 + 437,162 = 755,412 ≈ 755k, matching option D.

Answer: \boxed{755k} ✓

GPT

- Figure 19: Example 3 (CHART case): reasoning traces of our model, GPT-5-mini, and GPT-5-mini + Qwen2.5-VL-7B-Instruct. INSIGHT-O3-VS effectively follows vReasoner’s guidance and retrieves high-quality crops that fully capture the described regions. In contrast, Qwen2.5-VL-7B returns only partial crops in Round 2 and, in the Round 3, fails to produce a valid crop as requested by vReasoner, incorrectly concluding that the target region is absent, which leads to an incorrect answer.

Failure case 1 （ vReasoner — Knowledge–Perception Conflict ）

###### Round 1

JR Tadami Line JR Banestu West Line

Aizu Line

Route Map Legend

Aizu-tajima

Aizu-shimogo

Yunokami-onsen Tonohetsuri

Ashinomaki-onsen

Nishi-wakamatsu

Aizu-wakamatsu

Kitakata

Fukushima

Nakaarai

[Figure 426]

Isesaki Line TOBU SKYTREE Line Nikko Line Kameido Line Daishi Line TOBU URBAN PARK Line Sano Line Kiryu Line Koizumi Line Utsunomiya Line Kinugawa Line Tojo Line Ogose Line

Aizu-Kinugawa Line（Yagan Railway） Aizu Line JR Line Direct Limited Express Train Section Chichibu Line Hibiya Line（Tokyo Metro） Yurakucho Line（Tokyo Metro） Fukutoshin Line（Tokyo Metro） Hanzomon Line（Tokyo Metro） Tokyu Den-en-toshi Line Tokyu Toyoko Line Minatomirai Line JR Line

Aizu-arakai

Aizu-sanson-dojo

Aizukogen-ozeguchi

Nanatsugatake-tozanguchi

[Figure 427]

Ojika-kogen

Aizu-Kinugawa Line（Yagan Railway）

Kamimiyori-shiobaraonsenguchi

I need a closer view of the area around Shin-kiba and the lines heading toward central Tokyo, to identify the shortest path to Shibuya and count intermediate stations. I'll crop the region around Shin-kiba, the nearby lines (Yurakucho, Keiyo/Rinkai), and Shibuya area connections.

Nakamiyori-onsen

Yunishigawa-onsen

[Figure 428]

Kawaji-yumoto

Kawaji-onsen Ryuokyo

Shin-fujiwara

Kinugawa-koen

[Figure 429]

Kinugawa-onsen Tobu World Square

GPT

[Figure 430]

Kosagoe

- 51

- 52

- 53

- 54

- 56

55

- 57

- 58

Kinugawa Line

Shin-takatoku

Shimo-goshiroMyojin

Tochigi

Okuwa

[Figure 431]

Daiyamuko

Itaga

Lake Chuzenji

Tobu-nikko Tobu-utsunomiya

- 21

Kita-kanuma

19

- 22

Akagi

[Figure 432]

57

25

Gunma

Nikko Line

20 24 23

Kiryu Line

Aioi

Shimo-imaichi

Kami-imaichi

Minamiutsunomiya

Yabuzuka

56

Shin-kanuma

Jiroembashi

18

Locate Shin-kiba area and routes toward central Tokyo and Shibuya (center-right lower part of map).

Utsunomiya Line

Sammaibashi

Kuzu

Shin-kiryu Azami

Esojima

Momiyama

55

- 13

12 31 32

39

40

33

34

35

36

37

38

- 14

- 15

- 16

- 17

39

Yashu-yamabe

Nishikawada

Niregi

Tada

Ashikagashi

Tobu-izumi

38

54 53

Niragawa

Yasuzuka

Tobu-kanasaki

Isesaki

25

Tanuma

52

37 36 35

Omochanomachi

Ienaka

Ota

Yoshimizu

Tatebayashi

51

Fukui

Sano Line

Kuniya

Kassemba

Shin-isesaki

Isesaki Line

24

Agata

18

17 16 15 14

Tochigi 11 Shin-ohirashita 10

Horigome

Mibu

Goshi

23

Tatara

13

Sano

###### Round 2

34

22

21 20 19

12

Yashu-otsuka

Yashu-hirakawa

Shin-tochigi

Shizuwa

Ryumai

Sakaimachi

Serada

Kizaki

Hosoya

09

47

Sanoshi

33

11

Koizumi Line

Tajima

10

32

Fujioka

- 06 05

- 07

- 08

Nishi-koizumi

Watarase

31

46

Itakura-toyodaimae

Koizumimachi

Higashi-koizumi

Shinozuka

Hon-nakano

Narushima

45 44 43 42 41

[Figure 433]

Morinjimae

09

Ibaraki

[Figure 434]

Yagyu Shin-koga

Kawamata

08

Kurihashi

Washinomiya

Yorii

04

39

[Figure 435]

[Figure 436]

Hanasaki

Minami-kurihashi

- 01

- 02

- 03

I need to see the path from Shin-kiba along the Yurakucho Line toward central Tokyo and how to reach Shibuya (transfer points and intermediate stations). I'll crop a wider central map that includes Yurakucho, Ikebukuro, Shinjuku, Shibuya, and connecting lines to identify the shortest route and count stations between Shin-kiba and Shibuya (excluding endpoints).

Kumagaya

Gyoda-shi

Hanyu Minami-hanyu

07

Chichibu Line

Satte

38

06

Tamayodo

Sugito-takanodai

37

04 05

[Figure 437]

Kazo

03

Hachigata

36

Tobu-dobutsu-koen

30

Obusuma

35

- 01 02

51

03 04

44

43

42

41

29

- 02 Wado

01

Minami-yori〈HondaYorii-mae〉

34

Himemiya

Kuki

Tobu-takezawa

33

Kita-kasukabe

28

Kasukabe

Atago

Ogawamachi

32

16

10

- 25 24 23 22 21

- 26

- 27

JR Line Direct Limited Express Train Section

Shinrin-koen

30 29 28 27

GPT

Musashi-ranzan

31

Nodashi Umesato Unga

17

Higashi-matsuyama

Tsukinowa

11 12 13 14 15

05 06 07 08 09

Ichinowari

18

Nanasato

Iwatsuki

Higashi-iwatsuki

Toyoharu

Yagisaki

Fujino-ushijima

Minami-sakurai

Kawama

Nanakodai

Shimizu-koen

Takasaka

Takesato

Saitama

04

19

Owada

Kita-sakado

Sengendai

03

Edogawadai

20

Omiya-koen

Obukuro

Omiya

[Figure 438]

01 02

Chichibu

TOBU URBAN PARK Line

Kita-omiya

Kita-koshigaya

26 41 25 42

[Figure 439]

Ogose Line

Sakado

Koshigaya

Hatsuishi

24

- 21

- 22

- 23

- 24

- 25

- 26

Wakaba

Ippommatsu

Ogose

Shin-koshigaya

23

Kawagoe

47

Tsurugashima

- 18 17 16 15 14 13 12 11 10 09 08 07 06 05

- 19

- 20

Nishi-oya

43 46 45 44

21 20 19

22

Kasumigaseki

Nagareyama-otakanomori

Kawakado

Gamo

Shingashi Kami-fukuoka

Locate Central Tokyo area showing Yurakucho Line from Shin-kiba through Yurakucho, Ikebukuro, Shinjuku, and Shibuya with nearby transfer stations

Kawagoeshi

Bushu-karasawa

Higashi-moro

Bushu-nagase

TOBU SKYTREE Line

Urawa

Shinden

Mitsumineguchi

Toyoshiki

Dokkyodaigakumae〈Soka-Matsubara〉

Tojo Line

Shimo-akatsuka

Soka

18

Narimasu

Kashiwa

Wakoshi

Fujimino

17

Yatsuka

Tsuruse

16

Tobu-nerima

Kami-itabashi

Takenotsuka

Shin-kashiwa

Mizuhodai

15

Yanasegawa

14

Daishimae

Nishiarai

11

Naka-itabashi

Shiki

13

Tokiwadai

Masuo

10

Umejima

12

Asakadai

Daishi Line

Shimo-itabashi

09

Asaka

Gotanno

Kita-ikebukuro

08

Oyama

###### Round 3

Kosuge

07

Sakasai

27 28

06

Kita-senju

05

Takayanagi

Ushida

04

Asakusa

Mutsumi Shin-kamagaya

TOKYO SKYTREE

29

03

Horikiri

- 30

- 31

- 32

- 33

- 34

02

[Figure 440]

Kanegafuchi

Yurakucho Line

Ikebukuro

Kamagaya

01

Tokyo

Higashi-mukojima

Magomezawa

###### Chiba

[Figure 441]

Omurai

Kameido Line

[Figure 442]

Shinjuku

Oshiage

Tsukada Shin-funabashi

Nagatacho

Higashi-azuma

I need a closer view of the Hibiya Line segment between Yurakucho and Naka-meguro (showing station names and dots) to count how many stations are passed on the route Shin-kiba -> Yurakucho (transfer) -> Hibiya to Naka-meguro -> Toyoko/Tokyu to Shibuya. I'll crop the area including Yurakucho, Hibiya line westward to Naka-meguro and the connection to the Toyoko/Tokyo lines into Shibuya.

Hikifune

〈SKYTREE〉

[Figure 443]

Fukutoshin Line

Kameidosuijin

Hanzomon Line

Hibiya

Tokyu Den-en-toshi Line

Sangen-jaya

Shibuya

Funabashi

Kameido

Futako-tamagawa

35

[Figure 444]

Yurakucho Naka-meguro

Mizonokuchi

Hibiya Line

Shin-kiba

GPT

Tokyu Toyoko Line

Nagatsuta

Kanagawa

Yokohama

Jiyugaoka Tamagawa

Hiyoshi

Kikuna

Den-en-chofu

Chuo-rinkan

Motomachi-chukagai 〈Yamashita-koen〉

Musashi-kosugi

Tokyo Bay

[Figure 445]

Minatomirai Line

[Figure 446]

Locate area showing Yurakucho, Hibiya Line west from Yurakucho to Naka-meguro and the connection to Shibuya (including Naka-meguro, Shibuya transfer node)

###### Question

A tourist wants to travel from Shin-kiba to Shibuya by subway. How many stations must they pass through on the shortest route, excluding Shin-kiba and Shibuya?

[Figure 447]

- A. 1
- B. 2 ✅

C. 3 D. 4

- E. 5
- F. No right choice

From the zoomed images I can trace the shortest path: Shin-kiba (Yurakucho Line) goes directly to Yurakucho, transfer there to the Hibiya Line which goes to Naka-meguro, then transfer to the Tokyu Toyoko line (or the short connection) which stops at Daikanyama before Shibuya. The stations passed through between Shin-kiba and Shibuya are Yurakucho, Naka-meguro, and Daikanyama — three intermediate stations.

[Figure 448]

[Figure 449]

✘

✘

GPT

Explanation

✘

Answer: \boxed{3}

- Step 1. Identify Shin-kiba at the bottom of the image as the terminal station of the Yurakucho Line (blue-yellow).
- Step 2. Locate Shibuya at the lower left of the image on the Tokyu Den-entoshi Line.
- Step 3. Excluding the terminal stations, count the stations passed: pass through Yurakucho and Nagatacho, then transfer to the Hanzomon Line to reach Shibuya, traversing a total of 2 stations. Answer: B

Failure Reason

In this case, the vReasoner correctly decomposes the question and formulates a proper region description that identifies the target object and its approximate location. The vSearcher then successfully locates and returns a clear region crop. However, when interpreting the cropped image, the vReasoner hallucinates a nonexistent intermediate station “Daikanyama” between Nakameguro and Shibuya, leading to an incorrect count. This error likely arises from a conflict between the model’s external world knowledge and the actual image content — since in the real Tokyu Toyoko Line, Daikanyama exists between those two stations, the model’s pretrained knowledge overrides the visual evidence, producing a hallucinated interpretation.

[Figure 450]

Failure case 2 （ vReasoner — Visual Misperception ）

[Figure 451]

[Figure 452]

- Round 1

[Figure 453]

[Figure 454]

I need to find which bus route runs directly between the Amtrak Station (eastern area) and Vintage Faire Mall, then check that route's Saturday frequency. I'll zoom to the area around Vintage Faire Mall to identify the route number(s) serving it.

Locate area around Vintage Faire Mall and nearby route numbers, center-left of the full map

[Figure 455]

[Figure 456]

GPT

- Round 2

[Figure 457]

[Figure 458]

I need a closer view of the Amtrak Station area to read the route number(s) that stop there so I can check their Saturday frequency. I'll zoom into the Amtrak Station and the nearby route labels/stops.

Locate Amtrak Station and immediate surrounding routes and route numbers, eastern edge near Briggsmore and E. Orangeburg

[Figure 459]

[Figure 460]

GPT

- Round 3

Route 30 (magenta) serves the Amtrak Station and also serves Vintage Faire Mall. The bus frequency table shows Route 30 runs every 30 minutes on Saturdays. Therefore the maximum wait time is 30 minutes.

Answer: \boxed{B. 30 minutes}

[Figure 461]

GPT

[Figure 462]

[Figure 463]

[Figure 464]

I need to see the bus frequency table to read the Saturday frequency for route 30(the route that serves both Amtrak Station and Vintage Faire Mall). I'll zoom into the bus frequency table in the lower-left of the map.

Locate bus frequency table located lower-left of the full map.

[Figure 465]

[Figure 466]

GPT

- Round 4

[Figure 467]

[Figure 468]

[Figure 469]

I need to read the map near the Amtrak Station (eastern area) to identify which bus route goes directly from there to Vintage Faire Mall and then check the bus frequency (Saturday) for that route. I'll zoom into the eastern section around the Amtrak Station to see route numbers and any labels directing to Vintage Faire Mall.

GPT

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

Locate eastern area around Amtrak Station showing route numbers and nearby stops.

Question

If a traveler plans to depart from the Amtrak Station in the eastern part of the Modesto Urban Area on a Saturday and take a bus directly to Vintage Faire Mall, what is the maximum wait time for the bus at the Amtrak Station?

[Figure 475]

- A. 15 minutes
- B. 30 minutes

###### C. 45 minutes D. 60 minutes

- E. The relevant bus does not operate on that day
- F. No right choice

[Figure 476]

[Figure 477]

✅

[Figure 478]

Explanation

- Step 1. Identify the bus route passing through the Amtrak Station that can reach Vintage Faire Mall , which is Route 25.
- Step 2. Refer to the Bus Frequency table to determine that Route 25 operates every 60 minutes on Saturdays.
- Step 3. Conclude that the traveler may need to wait up to 60 mins for the bus. Answer: D

[Figure 479]

[Figure 480]

[Figure 481]

✘

Failure Reason

In this case, the vReasoner correctly decomposes the question and provides an appropriate region description specifying the relevant targets and their approximate locations. The vSearcher accurately identifies the corresponding region and returns a clear crop. However, when interpreting the cropped image, the vReasoner misreads the color-coded line: the route connecting Amtrak Station and Vintage Faire Mall should be the purple Route 25, but the model mistakes it for the light-purple Route 30. This visual misinterpretation leads to an incorrect reading of the timetable and a wrong final answer.

✘

✘

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

High-quality Crop

Medium-quality Crop

Low-quality Crop

Region crop from InSight-o3 vSearcher

Region Description

vReasoner

###### Figure 20: Failure cases 1 & 2 of INSIGHT-O3 (GPT-5-mini + InSight-o3-vS).

[Figure 488]

Failure case 3 （ vReasoner — Improper Search Delegation ）

[Figure 489]

[Figure 490]

[Figure 491]

###### Round 1

[Figure 492]

[Figure 493]

I need a closer view of the left-central area where the Parking Garage, Skybridge, Electric Vehicle Parking/Charging Stations, and the corridor to the terminal are located to trace the most direct route to the Corewell Health Kids’ Port. I'll crop the left side of the map showing the parking garage, skybridge, and adjacent terminal entrances.

GPT

[Figure 494]

[Figure 495]

Locate left-center area showing Parking Garage, Skybridge, Terminal Drive, and nearby terminal entrances.

[Figure 496]

###### Round 2

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

I need a close-up of the central terminal area (the main rotunda and adjoining concourses) that shows the Market Place, Corewell Health Kids’ Port, baggage claim locations, GRR Life Market & Cafe, ATM, and taxi stand so I can trace the most direct route from the EV charging area into the Kids’ Port.

GPT

[Figure 501]

[Figure 502]

Locate central terminal rotunda showing Market Place, Corewell Health Kids' Port, nearby baggage claim, GRR Life Market & Cafe, ATM icons.

###### Round 3

[Figure 503]

[Figure 504]

Question

I need a closer view of the concourse A area (right side of the central rotunda) to locate the Corewell Health Kids’ Port and see the exact path from the main terminal/security area so I can determine which listed facilities lie directly on the most direct route.

A person starting from the Electric Vehicle Parking/Charging Stations wants to reach the Corewell Health Kids’ Port. Which of the following locations/facilities would they pass directly along the most direct route to the Kids’ Port?

[Figure 505]

GPT

[Figure 506]

[Figure 507]

[Figure 508]

- A. Market Place
- B. Baggage Claim No.4

C. GRR Life Market & Cafe D. ATM

E. Taxi Stand

[Figure 509]

[Figure 510]

✅ F. No right choice

Locate right-central area including Security Checkpoint, Market Place, and Concourse A (gates A1-A4) showing icons along concourse.

Explanation

- Step 1. From the legend, identify the symbols for Electric Vehicle Parking/Charging Stations and Corewell Health Kids’ Port.
- Step 2. On the terminal map, locate the Electric Vehicle Parking/Charging Stations in the bottom-left corner, and the Corewell Health Kids’ Port at the center of the terminal.
- Step 3. Starting from the Electric Vehicle Parking/Charging Stations, cross the Skybridge into the terminal, pass the ATM near the entrance, then continue forward and turn left to reach the Corewell Health Kids’ Port. Answer: D (ATM)

###### Round 4

[Figure 511]

[Figure 512]

I'll zoom in on the area just inside the terminal near the GRR Life Market & Cafe

and the corridor leading to Concourse A to find the Corewell Health Kids' Port and confirm which facilities lie directly on the most direct path from the EV charging

GPT location (which enters through the skybridge near GRR Life Market & Cafe).

[Figure 513]

[Figure 514]

###### Locate entry corridor area showing GRR Life Market & Cafe, Gerald R. Ford Tribute Room, and the start of Concourse A with nearby icons.

Failure Reason

[Figure 515]

This failure stems from the vReasoner’s inability to localize the target region Corewell Health Kids’ Port. Instead of first consulting the legend to identify the target’s symbol, the vReasoner attempts direct visual search across multiple regions without delegating the task to the vSearcher. During this process (Rounds 2–4), it generates several broad region descriptions. Although the vSearcher returns crops that match these descriptions, the target occupies only a small portion of each large region and lacks textual labels, making recognition difficult. Consequently, even with the cropped images, the vReasoner fails to identify the target and produces an incorrect answer.

[Figure 516]

[Figure 517]

[Figure 518]

From the EV parking (shown at the left skybridge), the most direct path goes across the lower skybridge into the terminal right by the GRR Life Market & Cafe. From there you continue through the central area toward security and on to

[Figure 519]

✘

Fort George

Concourse A where the Corewell Health Kids’ Port is located. Other options like Baggage Claim No.4, the Taxi Stand, and the ATM are not directly on that straight path; Market Place is nearby but sits off to the side of the security/rotunda, so the clearly passed facility is the GRR Life Market & Cafe.

###### LEGEND

###### 604 - NOTL SHUTTLE 602 - BLUE LINE

MCFARLAND HOUSE

LAURA SECORD HOMESTEAD & MACKENZIE PRINTERY

GPT

- 602 - Blue Line - Fallsview/Clifton Hill

601 - Red Line - Lundy’s Lane

605 - Red Line Express - Lundy’s Lane

- 603 - Green Line - Niagara Parks

604 - Niagara-on-the-Lake Shuttle

###### QUEENSTON HEIGHTS PARK

✘

Daily April 1 to October 27

Restaurant & Heritage Sites

Lewiston BridgeQueenston

Niagara-On-The-Lake Shuttle From April 1 to October 27, the Niagara-on-the-Lake shuttle

A more effective strategy would be for the vReasoner to reference the legend first and delegate precise localization to the vSearcher.

Answer: \boxed{C}

WEGO Main Transfer

T Hub to All Lines

###### FLORAL CLOCK

- o ers service between Niagara-on-the-Lake and the village
- of Queenston. Visit niagaraparks.com/notlshuttle for more information.

###### BUTTERFLY CONSERVATORY

Botanical Gardens School of Horticulture The Garden Café

Transfer

###### NIAGARA GLEN

WEGO Stop

Nature Centre & Trails (north only)

ADULT CHILD

Whirlpool Adventure Course (north only)

Bus & Train Stop

ONE WAY $12.00 $6.00 ROUND TRIP $24.00 $12.00

via Bridge Street

WHIRLPOOL GOLF COURSE

(south only)

Niagara Parks Welcome Centre

Fort George

Major Parking Lot

[Figure 520]

Failure case 4 （ vSearcher — Region Localization Failure ）

###### LEGEND

###### 604 - NOTL SHUTTLE 602 - BLUE LINE

[Figure 521]

All Niagara Parks locations also include parking, including accessible spots for permit holders

WHIRLPOOL AERO CAR

MCFARLAND HOUSE

N a g a r a P a k w

Victoria Ave

MAP NOT TO SCALE

LAURA SECORD HOMESTEAD & MACKENZIE PRINTERY

DrummondRd

DorchesterRd

PortageRoad

Maple Leaf Place (south only)

MontroseRd

- 602 - Blue Line - Fallsview/Clifton Hill

601 - Red Line - Lundy’s Lane

605 - Red Line Express - Lundy’s Lane

- 603 - Green Line - Niagara Parks

###### Round 1

StanleyAve

KalarRd

WHITE WATER WALK

###### LEGEND

###### 604 - NOTL SHUTTLE 602 - BLUE LINE

MCFARLAND HOUSE

GO/VIA Station Bus Terminal

WhirlpoolBridge

[Figure 522]

604 - Niagara-on-the-Lake Shuttle

LAURA SECORD HOMESTEAD & MACKENZIE PRINTERY

- 602 - Blue Line - Fallsview/Clifton Hill

601 - Red Line - Lundy’s Lane

605 - Red Line Express - Lundy’s Lane

- 603 - Green Line - Niagara Parks

Downtown

[Figure 523]

###### QUEENSTON HEIGHTS PARK

Daily April 1 to October 27

Morrison St

Daily • April 1 - June 29

###### Daily • May 16 - September 2

Restaurant & Heritage Sites

604 - Niagara-on-the-Lake Shuttle

Bed & Breakfast

Lewiston BridgeQueenston

SkylonTower

LauraSecord

NiagaraFalls

QUEENSTON HEIGHTS PARK

Copacabana

Ave./Clifton

NiagaraCity

Daily April 1 to October 27

Homestead/

Niagara-On-The-Lake Shuttle From April 1 to October 27, the Niagara-on-the-Lake shuttle

FortGeorge

Convention

Turnaround

Bird Kingdom (south only)

TableRock

Ave/Living

Lewiston BridgeQueenston

Queenston

Mackenzie

McFarland

Niagara-On-The-Lake Shuttle From April 1 to October 27, the Niagara-on-the-Lake shuttle

Fallsview

Fallsview

Embassy

Niagara

Victoria

Printery

I need to zoom into the central map area showing the Marriott Hotel stop and the route up to Queenston Heights Park to see which lines serve each and whether a direct (no-walking) bus transfer sequence exists. I'll crop the map region around the downtown network and the routes leading north to Queenston Heights Park.

stonSt.

Heights

Cruises

Casino

Casino

Centre

Centre

Suites

House

Clock

Floral

WEGO Main Transfer

Sheraton at the Falls

Casino Niagara

RainbowBridge

Scott’sTrailerPark

KOAKampground

T Hub to All Lines

FLORAL CLOCK

AmericanaResort

Hill

- o ers service between Niagara-on-the-Lake and the village
- of Queenston. Visit niagaraparks.com/notlshuttle for more information.

DoubletreeHotel

[Figure 524]

HistoryMuseum

Victoria Ave

CarriageHouse

DrummondRd

NiagaraLodge

FactoryOutlet

DorchesterRd

BUTTERFLY CONSERVATORY

WEGO Main Transfer

RamadaNear

BestWestern

Falls Ave

MontroseRd

NiagaraFalls

Transfer

10:30am

10:50am

CanadaOne

10:38am

10:32am

11:00am

9:00am

9:20am

U.S.A.

9:24am

8:58am

9:35am

8:52am

9:27am

9:10am

9:18am

9:13am

Victoria Ave/ Clifton Hill

CairnCroft

Oneida Lane

T Hub to All Lines

Clifton Hill

Campark

###### FLORAL CLOCK

KalarRd

theFalls

&Suites

NIAGARA GLEN

- o ers service between Niagara-on-the-Lake and the village
- of Queenston. Visit niagaraparks.com/notlshuttle for more information.

WEGO Stop

CliftonHill

Howard Johnson

ADULT CHILD

Bus will arrive every 20 minutes until last bus at:

GPT

Bus & Train Stop

###### Bus will arrive every 60 minutes until last southbound bus at:

ONE WAY

$6.00 ROUND TRIP $ $12.00

Ferry St./ Fallsview Blvd.

|$12.00 $24.00|
|---|

###### BUTTERFLY CONSERVATORY

###### GRAND VIEW MARKETPLACE

via Bridge Street

6:00pm

6:04pm

6:30pm

8:40pm

8:44pm

8:30pm

8:20pm

8:38pm

8:47pm

8:33pm

8:55pm

Lundy’s Lane

Ferry St.

6:15pm

6:21pm

8:18pm

8:12pm

WHIRLPOOL GOLF COURSE

Botanical Gardens School of Horticulture The Garden Café

Transfer

Niagara Parks Welcome Centre

Niagara City Cruises Zipline to the Falls Shopping & Dining

Ramada Hotel

Totwer

Theare

Skylon

IMAX

Major Parking Lot

Daily • June 30 - September 2

11:40pm

11:44pm

11:30pm

11:20pm

11:49pm

11:38pm

11:47pm

11:33pm

11:18pm

11:12pm

Wyndham Garden

[Figure 525]

Scan the QR Code to downlaod the Transit App for real time estimated bus arrivals.

###### QUEEN VICTORIA PLACE

All Niagara Parks locations also include parking, including accessible spots for permit holders

10:00am

###### NIAGARA GLEN

9:30am

9:50am

WEGO Stop

9:38am

9:32am

Restaurant Shopping & Dining Illumination Tower

WHIRLPOOL AERO CAR

N

Fallsview Casino

Nature Centre & Trails (north only)

ADULT CHILD

Murray Hill

[Figure 526]

###### TABLE ROCK CENTRE

Hilton Hotel & Fallsview Casino

###### Bus will arrive every 60 minutes until last southbound bus at:

Victoria Ave

Table Rock House Restaurant Journey Behind the Falls Niagara’s Fury Shopping & Dining

Casino

MAP NOT TO SCALE

Whirlpool Adventure Course (north only)

Bus & Train Stop

T

ONE WAY $12.00 $6.00 ROUND TRIP $24.00 $12.00

7:00pm

7:04pm

7:30pm

Locate center-right map area showing downtown stops, Marriott Hotel stop, route lines extending north toward Queenston Heights Park (include the waterfront and northbound routes).

4 Point Sheraton

7:15pm

7:21pm

WEGO is fully accessible and operates year-round, giving easy and convenient services to travelers from around the world who visit Niagara Falls.

FOR UP-TO-DATE BUS SCHEDULES VISIT NIAGARAPARKS.COM/WEGO FOR UP-TO-DATE GO TRAIN SCHEDULES VISIT GOTRANSIT.COM

NIAGARA PARKS POWER STATION

Starbucks

DrummondRd

DorchesterRd

PortageRoad

MontroseRd

+ THE TUNNEL

StanleyAve

Oakes Hotel

Currents - Sound & Light Show

via Bridge Street

KalarRd

###### SPRING-SUMMER SCHEDULE

Daily • September 3 - October 27

Copacabana

WHIRLPOOL GOLF COURSE

FLORAL SHOWHOUSE

2024

WHITE WATER WALK

10:30am

10:50am

10:38am

10:32am

11:00am

Dunn St Embassy Suites Livingstone St

(south only)

Dunn St

CANADA

Niagara Parks Welcome Centre

Marineland

WhirlpoolBridge

Rapidsview Parking

Niagara also welcomes cyclists! Each WEGO bus can transport two bicycles on front-mounted racks.

Marriott Hotel

All times listed are estimates, tra c congestion and weather conditions may impact schedule and service may be delayed. Children two and under ride for free. Pricing in Canadian dollars. Tax included. Green Line fare not available through the Transit app. All rates, routes and schedules are subject to change without notice. 0:00pm indicates stop ends earlier than other stops on line.

###### Bus will arrive every 60 minutes until last southbound bus at:

McLeod Rd

Stanley Ave

Daily • April 1 - June 29

###### Daily • May 16 - September 2

6:00pm

6:04pm

6:30pm

6:21pm

6:!5pm

Major Parking Lot

PortageRd

SkylonTower

LauraSecord

NiagaraFalls

Copacabana

Ave./Clifton

NiagaraCity

Homestead/

FortGeorge

Convention

Turnaround

FORT ERIE

TableRock

Ave/Living

Queenston

Mackenzie

McFarland

Fallsview

Fallsview

Embassy

Niagara

Victoria

Printery

stonSt.

Heights

NIAGARAPARKS.COM/WEGO

Cruises

Casino

Casino

Centre

Centre

Suites

House

###### Round 2

Clock

Floral

RainbowBridge

Scott’sTrailerPark

KOAKampground

AmericanaResort

Hill

DoubletreeHotel

HistoryMuseum

CarriageHouse

DrummondRd

NiagaraLodge

FactoryOutlet

DorchesterRd

RamadaNear

BestWestern

Falls Ave

MontroseRd

NiagaraFalls

10:30am

10:50am

CanadaOne

10:38am

10:32am

11:00am

9:00am

9:20am

U.S.A.

9:24am

*All Niagara Parks locations also include parking, including accessible spots for permit holders

8:58am

9:35am

8:52am

9:27am

9:10am

9:18am

9:13am

CairnCroft

Campark

KalarRd

theFalls

&Suites

CliftonHill

WHIRLPOOL AERO CAR

[Figure 527]

Bus will arrive every 20 minutes until last bus at:

###### Bus will arrive every 60 minutes until last southbound bus at:

[Figure 528]

GRAND VIEW MARKETPLACE

6:00pm

6:04pm

6:30pm

[Figure 529]

8:40pm

N

8:44pm

8:30pm

8:20pm

8:38pm

8:47pm

8:33pm

8:55pm

6:15pm

6:21pm

8:18pm

8:12pm

[Figure 530]

i

[Figure 531]

a g a r a

Totwer

Theare

Skylon

[Figure 532]

IMAX

Daily • June 30 - September 2

11:40pm

11:44pm

11:30pm

11:20pm

11:49pm

11:38pm

11:47pm

11:33pm

11:18pm

11:12pm

Scan the QR Code to downlaod the Transit App for real time estimated bus arrivals.

QUEEN VICTORIA PLACE

10:00am

9:30am

9:50am

9:38am

9:32am

P a r k

Victoria Ave

I need to see the downtown hub where the Marriott Hotel stop is labeled and the lines connecting north to Queenston Heights Park, to determine if a single bus line runs directly or if transfers are required and how many. I'll crop the map area to the right-center where the downtown stops and the Marriott Hotel label and route interchanges are located.

TABLE ROCK CENTRE

###### Bus will arrive every 60 minutes until last southbound bus at:

MAP NOT TO SCALE

T

w

7:00pm

7:04pm

7:30pm

7:15pm

7:21pm

WEGO is fully accessible and operates year-round, giving easy and convenient services to travelers from around the world who visit Niagara Falls.

FOR UP-TO-DATE BUS SCHEDULES VISIT NIAGARAPARKS.COM/WEGO FOR UP-TO-DATE GO TRAIN SCHEDULES VISIT GOTRANSIT.COM

NIAGARA PARKS POWER STATION

a y

+ THE TUNNEL

###### SPRING-SUMMER SCHEDULE

Daily • September 3 - October 27

FLORAL SHOWHOUSE

2024

10:30am

10:50am

10:38am

10:32am

11:00am

DrummondRd

DorchesterRd

CANADA

PortageRoad

GPT

Marineland

Niagara also welcomes cyclists! Each WEGO bus can transport two bicycles on front-mounted racks.

Maple Leaf Place (south only)

MontroseRd

All times listed are estimates, tra c congestion and weather conditions may impact schedule and service may be delayed. Children two and under ride for free. Pricing in Canadian dollars. Tax included. Green Line fare not available through the Transit app. All rates, routes and schedules are subject to change without notice. 0:00pm indicates stop ends earlier than other stops on line.

StanleyAve

###### Bus will arrive every 60 minutes until last southbound bus at:

6:00pm

6:04pm

6:30pm

6:21pm

6:!5pm

PortageRd

KalarRd

FORT ERIE

NIAGARAPARKS.COM/WEGO

[Figure 533]

WHITE WATER WALK

[Figure 534]

Question

GO/VIA Station Bus Terminal

Locate downtown hub area with overlapping red/green/blue lines, labels like Marriott Hotel, Wyndham Garden, Ramada Hotel, and lines going north toward Queenston Heights Park (center-right of map).

WhirlpoolBridge

If I am staying at the Marriott Hotel and plan to go to QUEENSTON HEIGHTS PARK today, can I reach it directly by transferring between buses (without walking in between)? And how many lines must I take at minimum?

Downtown

Morrison St

Daily • April 1 - June 29

###### Daily • May 16 - September 2

###### 604 - NOTL SHUTTLE 602 - BLUE LINE

Bed & Breakfast

SkylonTower

LauraSecord

NiagaraFalls

Copacabana

Ave./Clifton

NiagaraCity

Homestead/

FortGeorge

Convention

Turnaround

Bird Kingdom (south only)

TableRock

Ave/Living

Queenston

###### Round 3

Mackenzie

McFarland

Fallsview

Fallsview

Embassy

Niagara

Victoria

Printery

stonSt.

Heights

Cruises

Casino

Casino

Centre

Centre

Suites

House

Clock

Floral

[Figure 535]

✅

- A. No, 2 lines
- B. No, 3 lines

C. Yes, 1 line D. Yes, 2 lines

- E. Yes, 3 lines
- F. None of the above

Sheraton at the Falls

Casino Niagara

RainbowBridge

Scott’sTrailerPark

KOAKampground

AmericanaResort

Hill

DoubletreeHotel

[Figure 536]

HistoryMuseum

Victoria Ave

CarriageHouse

DrummondRd

NiagaraLodge

FactoryOutlet

DorchesterRd

RamadaNear

BestWestern

Falls Ave

MontroseRd

NiagaraFalls

10:30am

10:50am

CanadaOne

10:38am

10:32am

[Figure 537]

11:00am

9:00am

9:20am

U.S.A.

9:24am

8:58am

9:35am

8:52am

9:27am

9:10am

9:18am

9:13am

Victoria Ave/ Clifton Hill

CairnCroft

Oneida Lane

Clifton Hill

Campark

KalarRd

theFalls

&Suites

I still need to see the exact stop labels around the downtown transfer hubespecially the "Marriott Hotel" stop—and the colored lines serving it and continuing north to "Queenston Heights Park" to determine if a direct bus transfer (no walking) is possible and how many lines are required. I'll crop the map area slightly lower-right where the hub and labeled hotels (Marriott, Ramada, Wyndham) and the colored route lines converge.

CliftonHill

Howard Johnson

Bus will arrive every 20 minutes until last bus at:

###### Bus will arrive every 60 minutes until last southbound bus at:

Niagara-On-The-Lake Shuttle From April 1 to October 27, the Niagara-on-the-Lake shuttle

Ferry St./ Fallsview Blvd.

###### GRAND VIEW MARKETPLACE

6:00pm

6:04pm

6:30pm

8:40pm

8:44pm

8:30pm

8:20pm

8:38pm

8:47pm

8:33pm

8:55pm

Lundy’s Lane

Ferry St.

6:15pm

6:21pm

8:18pm

8:12pm

Explanation

GPT

Niagara City Cruises Zipline to the Falls Shopping & Dining

- o ers service between Niagara-on-the-Lake and the village
- of Queenston. Visit niagaraparks.com/notlshuttle for more information.

Ramada Hotel

Totwer

Theare

Skylon

IMAX

Daily • June 30 - September 2

- Step 1. Locate the Marriott Hotel , near the area below the blue line.
- Step 2. Locate QUEENSTON HEIGHTS PARK , at the top of the map near the orange line.
- Step 3. Determine the route: take the blue line, transfer to the green line, and then transfer to the orange line.
- Step 4. Refer to the LEGEND and note that the Main Transfer allows transfers to all lines. Transfer from the blue line to the green line at Main Transfer , and then from the green line to the orange line at FLORAL CLOCK.
- Step 5. Conclude that it is possible to reach the destination directly via transfers, requiring three lines. Answer: E.

11:40pm

11:44pm

11:30pm

11:20pm

11:49pm

11:38pm

11:47pm

11:33pm

[Figure 538]

11:18pm

11:12pm

Wyndham Garden

Scan the QR Code to downlaod the Transit App for real time estimated bus arrivals.

###### QUEEN VICTORIA PLACE

10:00am

9:30am

9:50am

9:38am

9:32am

Restaurant Shopping & Dining Illumination Tower

[Figure 539]

[Figure 540]

ADULT CHILD

Fallsview Casino

[Figure 541]

Locate map hub area showing Marriott Hotel, Ramada Hotel, Wyndham Garden, and the colored route lines (red, green, blue) converging at downtown; include northbound routes towards Queenston Heights Park.

Murray Hill

ONE WAY $12.00 $6.00 ROUND TRIP $24.00 $12.00

###### TABLE ROCK CENTRE

Hilton Hotel & Fallsview Casino

###### Bus will arrive every 60 minutes until last southbound bus at:

Table Rock House Restaurant Journey Behind the Falls Niagara’s Fury Shopping & Dining

Casino

T

7:00pm

7:04pm

7:30pm

4 Point Sheraton

7:15pm

7:21pm

WEGO is fully accessible and operates year-round, giving easy and convenient services to travelers from around the world who visit Niagara Falls.

FOR UP-TO-DATE BUS SCHEDULES VISIT NIAGARAPARKS.COM/WEGO FOR UP-TO-DATE GO TRAIN SCHEDULES VISIT GOTRANSIT.COM

NIAGARA PARKS POWER STATION

Starbucks

+ THE TUNNEL

Oakes Hotel

Currents - Sound & Light Show

Fort George

###### SPRING-SUMMER SCHEDULE

Daily • September 3 - October 27

Copacabana

###### Round 4

###### LEGEND

###### 604 - NOTL SHUTTLE 602 - BLUE LINE

FLORAL SHOWHOUSE

MCFARLAND HOUSE

2024

10:30am

10:50am

10:38am

10:32am

11:00am

Dunn St Embassy Suites Livingstone St

Dunn St

CANADA

LAURA SECORD HOMESTEAD & MACKENZIE PRINTERY

Marineland

- 602 - Blue Line - Fallsview/Clifton Hill

601 - Red Line - Lundy’s Lane

605 - Red Line Express - Lundy’s Lane

- 603 - Green Line - Niagara Parks

Rapidsview Parking

[Figure 542]

Niagara also welcomes cyclists! Each WEGO bus can transport two bicycles on front-mounted racks.

Marriott Hotel

All times listed are estimates, tra c congestion and weather conditions may impact schedule and service may be delayed. Children two and under ride for free. Pricing in Canadian dollars. Tax included. Green Line fare not available through the Transit app. All rates, routes and schedules are subject to change without notice. 0:00pm indicates stop ends earlier than other stops on line.

[Figure 543]

###### Bus will arrive every 60 minutes until last southbound bus at:

604 - Niagara-on-the-Lake Shuttle

###### QUEENSTON HEIGHTS PARK

Daily April 1 to October 27

McLeod Rd

Stanley Ave

Restaurant & Heritage Sites

I need to see where the green (Niagara Parks) line runs through downtown and whether it shares a stop with the blue line near the Marriott so I can determine if a direct transfer (no walking) is possible and how many different lines are needed. I'll crop the area where the green and blue lines pass close to each other around Clifton Hill/Table Rock/Casino Niagara where transfers likely occur.

Lewiston BridgeQueenston

6:00pm

6:04pm

Niagara-On-The-Lake Shuttle From April 1 to October 27, the Niagara-on-the-Lake shuttle

6:30pm

6:21pm

6:!5pm

PortageRd

WEGO Main Transfer

T Hub to All Lines

###### FLORAL CLOCK

FORT ERIE

- o ers service between Niagara-on-the-Lake and the village
- of Queenston. Visit niagaraparks.com/notlshuttle for more information.

###### BUTTERFLY CONSERVATORY

Botanical Gardens School of Horticulture The Garden Café

NIAGARAPARKS.COM/WEGO

Transfer

GPT

###### NIAGARA GLEN

WEGO Stop

Nature Centre & Trails (north only)

ADULT CHILD

Whirlpool Adventure Course (north only)

Bus & Train Stop

ONE WAY $12.00 $6.00 ROUND TRIP $24.00 $12.00

via Bridge Street

WHIRLPOOL GOLF COURSE

[Figure 544]

(south only)

Niagara Parks Welcome Centre

Major Parking Lot

[Figure 545]

Failure Reason

Locate area showing green line along river through Table Rock / White Water Walk / Grand View / Queen Victoria Place and the nearby blue line stops at Casino Niagara / Clifton Hill / Table Rock intersection.

All Niagara Parks locations also include parking, including accessible spots for permit holders

WHIRLPOOL AERO CAR

N

i

a g a r a

Daily • April 1 - June 29

###### Daily • May 16 - September 2

P a r k

In this case, the vReasoner correctly decomposes the question and provides a clear region description that specifies both the target area and its approximate location. However, due to the complex visual structure of the image, the vSearcher fails to retrieve the correct region as instructed. Despite multiple clarifications and follow-up attempts from the vReasoner, the vSearcher still cannot locate the intended area. As a result, the vReasoner ultimately relies on its own limited visual perception to answer, leading to an incorrect localization of Queenston Heights Park.

Victoria Ave

SkylonTower

LauraSecord

MAP NOT TO SCALE

NiagaraFalls

Copacabana

Ave./Clifton

NiagaraCity

w

Homestead/

FortGeorge

Convention

Turnaround

TableRock

Ave/Living

Queenston

Mackenzie

McFarland

Fallsview

Fallsview

Embassy

DrummondRd

DorchesterRd

PortageRoad

Niagara

Victoria

Printery

stonSt.

Maple Leaf Place (south only)

Heights

Cruises

MontroseRd

Casino

Casino

Centre

Centre

StanleyAve

Suites

House

Clock

Floral

KalarRd

[Figure 546]

[Figure 547]

Hill

[Figure 548]

WHITE WATER WALK

GO/VIA Station Bus Terminal

10:30am

10:50am

10:38am

10:32am

11:00am

9:00am

9:20am

9:24am

8:58am

9:35am

8:52am

9:27am

9:10am

9:18am

9:13am

[Figure 549]

WhirlpoolBridge

The Marriott Hotel stop is on the blue line. The Queenston Heights Park stop is on the Niagara Parks (green) line up the river. The map shows a transfer hub where the blue and green lines share stops (transfer icons), so you can change buses directly at the hub without walking. That requires riding the blue line from Marriott and then transferring to the green line two lines total.

✘

Downtown

Morrison St

Daily • April 1 - June 29

###### Daily • May 16 - September 2

Bed & Breakfast

Bus will arrive every 20 minutes until last bus at:

###### Bus will arrive every 60 minutes until last southbound bus at:

SkylonTower

LauraSecord

NiagaraFalls

Copacabana

Ave./Clifton

NiagaraCity

Homestead/

FortGeorge

Convention

Turnaround

Bird Kingdom (south only)

TableRock

Ave/Living

Queenston

Mackenzie

McFarland

Fallsview

Fallsview

Embassy

Niagara

Victoria

6:00pm

Printery

6:04pm

stonSt.

6:30pm

8:40pm

8:44pm

Heights

8:30pm

8:20pm

Cruises

8:38pm

8:47pm

8:33pm

8:55pm

6:15pm

6:21pm

Casino

Casino

8:18pm

Centre

Centre

8:12pm

Suites

House

Clock

Floral

Sheraton at the Falls

Casino Niagara

RainbowBridge

Scott’sTrailerPark

KOAKampground

AmericanaResort

Hill

DoubletreeHotel

HistoryMuseum

Victoria Ave

CarriageHouse

GPT

DrummondRd

NiagaraLodge

FactoryOutlet

DorchesterRd

RamadaNear

BestWestern

Falls Ave

MontroseRd

NiagaraFalls

10:30am

10:50am

CanadaOne

10:38am

10:32am

11:00am

9:00am

9:20am

U.S.A.

9:24am

8:58am

9:35am

8:52am

9:27am

9:10am

9:18am

9:13am

Victoria Ave/ Clifton Hill

CairnCroft

Oneida Lane

Clifton Hill

Campark

KalarRd

theFalls

&Suites

CliftonHill

Howard Johnson

✘

Daily • June 30 - September 2

11:40pm

11:44pm

11:30pm

11:20pm

11:49pm

11:38pm

11:47pm

11:33pm

11:18pm

11:12pm

Bus will arrive every 20 minutes until last bus at:

###### Bus will arrive every 60 minutes until last southbound bus at:

Ferry St./ Fallsview Blvd.

Answer: \boxed{D. Yes, 2 lines}

Scan the QR Code to downlaod the Transit App for real time estimated bus arrivals.

###### GRAND VIEW MARKETPLACE

10:00am

6:00pm

6:04pm

6:30pm

8:40pm

8:44pm

8:30pm

8:20pm

8:38pm

8:47pm

8:33pm

8:55pm

Lundy’s Lane

9:30am

9:50am

Ferry St.

6:15pm

6:21pm

8:18pm

9:38am

8:12pm

9:32am

Niagara City Cruises Zipline to the Falls Shopping & Dining

Ramada Hotel

Totwer

Theare

Skylon

IMAX

Daily • June 30 - September 2

11:40pm

11:44pm

11:30pm

11:20pm

11:49pm

11:38pm

11:47pm

11:33pm

11:18pm

11:12pm

Wyndham Garden

Scan the QR Code to downlaod the Transit App for real time estimated bus arrivals.

###### QUEEN VICTORIA PLACE

10:00am

9:30am

9:50am

9:38am

9:32am

Restaurant Shopping & Dining Illumination Tower

###### Bus will arrive every 60 minutes until last southbound bus at:

Fallsview Casino

Murray Hill

7:00pm

7:04pm

7:30pm

###### TABLE ROCK CENTRE

Hilton Hotel & Fallsview Casino

7:15pm

7:21pm

|7:30pm<br><br>7:21pm<br><br>7:15pm<br><br>7:04pm<br><br>Bus will arrive every 60 minutes until last southbound bus at:<br><br>7:30pm<br><br>7:21pm<br><br>7:15pm<br><br>7:04pm<br><br>7:00pm<br><br>Low-quality FORU<br><br>[Figure 550]|7:00pm<br><br>P-TO-DATE BUS|
|---|---|
|Crop Daily•SeptemberDaily3-October• September27 3Crop- October 27FORUP-T|O-DATE GO|
|10:30am<br><br>10:32am<br><br>10:38am<br><br>10:50am<br><br>11:00am| |

[Figure 551]

###### B

Table Rock House Restaurant Journey Behind the Falls Niagara’s Fury Shopping & Dining

[Figure 552]

[Figure 553]

[Figure 554]

Casino

[Figure 555]

WEGO is fully accessible and operates year-round, giving easy and convenient services to travelers from around the world who visit Niagara Falls.

T

FOR UP-TO-DATE BUS SCHEDULES VISIT NIAGARAPARKS.COM/WEGO FOR UP-TO-DATE GO TRAIN SCHEDULES VISIT GOTRANSIT.COM

7:00pm

7:04pm

7:30pm

4 Point Sheraton

High-quality Crop

Medium-quality Cr

7:15pm

Region crop from InSight-o3 vSearcher

Region Description

7:21pm

WEGO is fully accessible and operates year-round, giving easy and convenient services to travelers from around the world who visit Niagara Falls.

SCHEDULES VISIT NIAGARAPARKS.COM/WEGO

NIAGARA PARKS POWER STATION

Starbucks

vReasoner

+ THE TUNNEL

Oakes Hotel

Currents - Sound & Light Show

###### SPRING-SUMMER SCHEDULE

###### SPRING-SUMMER SCHEDULE

Copacabana

FLORAL SHOWHOUSE

TRAIN SCHEDULES VISIT GOTRANSIT.COM

2024

10:30am

10:50am

10:38am

10:32am

11:00am

Dunn St Embassy Suites Livingstone St

Dunn St

2024

10:30am

10:50am

CANADA

10:38am

10:32am

11:00am

Marineland

Rapidsview Parking

Niagara also welcomes cyclists! Each WEGO bus can transport two bicycles on front-mounted racks.

Marriott Hotel

All times listed are estimates, tra c congestion and weather conditions may impact schedule and service may be delayed. Children two and under ride for free. Pricing in Canadian dollars. Tax included. Green Line fare not available through the Transit app. All rates, routes and schedules are subject to change without notice. 0:00pm indicates stop ends earlier than other stops on line.

###### Bus will arrive every 60 minutes until last southbound bus at:

Niagara also welcomes cyclists! Each WEGO bus can transport two bicycles on front-mounted racks.

McLeod Rd

Stanley Ave

6:00pm

6:04pm

6:30pm

6:21pm

6:!5pm

PortageRd

All times listed are estimates, tra c congestion and weather conditions may impact schedule and service may be delayed. Children two and under ride for free. Pricing in Canadian dollars. Tax included. Green Line fare not available through the Transit app. All rates, routes and schedules are subject to change without notice. 0:00pm indicates stop ends earlier than other stops on line.

###### Bus will arrive every 60 minutes until last southbound bus at:

FORT ERIE

NIAGARAPARKS.COM/WEGO

6:00pm

6:04pm

6:30pm

6:21pm

6:!5pm

###### Figure 21: Failure cases 3 & 4 of INSIGHT-O3 (GPT-5-mini + InSight-o3-vS).

NIAGARAPARKS.COM/WEGO

Table 8: Performance of Qwen3-VL-32B and INSIGHT-O3 with Qwen3-VL-32B vReasoner.

vReasoner vSearcher V⋆-Bench VisualProbe-Hard O3-Bench

Qwen3-VL-32B - 86.0 28.6 60.4 Qwen3-VL-32B Qwen2.5-VL-7B (before RL) 69.1 21.7 48.5 Qwen3-VL-32B Qwen2.5-VL-7B (after RL) 90.1 36.8 61.2

- D.5 INSIGHT-O3 WITH QWEN3-VL VREASONER

Apart from closed-source proprietary models, we also experiment with open models like Qwen3VL (Bai et al., 2025a) as the vReasoner of INSIGHT-O3. As shown in Table 8, INSIGHT-O3 (the last row) outperforms both the base Qwen3-VL-32B and the combination of Qwen3-VL-32B (vReasoner) and Qwen2.5-VL-7B (vSearcher) without RL. Although the advantage of INSIGHT-O3 on O3-BENCH is relatively small, we note that as open models continue to advance in tool use and general reasoning, the performance of INSIGHT-O3 with open vReasoners will further improve.

E PROMPTS FOR O3-BENCH

- E.1 PROMPTS FOR INFORMATION EXTRACTION

For both the full image and cropped images, we feed them into Qwen2.5-VL-32B (Bai et al., 2025b) for information extraction using the same prompt as below.

### System

You are acting as a **precise visual information extractor**. Given ONE image, you must (1) identify the image type, (2) write a **comprehensive, strictly factual** caption, and (3) extract **complete OCR text** when present (with special handling for tables). Follow the rules **exactly** and return the output in the three sections shown under ** Output Format**.

--### Global Principles

- 1) **No hallucinations.** Describe only what is visible. If something is unclear, write ‘[ illegible]‘ or ‘[partially obscured: ...]‘.
- 2) **Be exhaustive.** Do not omit small text, legends, tick labels, footnotes, watermarks, axis titles, subtitles, panel labels (e.g., ‘(a)‘, ‘(b)‘), or figure notes.
- 3) **Preserve fidelity.** Copy punctuation, capitalization, diacritics, signs (‘+‘, ‘-‘), units, and spacing **exactly**. Do not normalize numbers or rewrite text.
- 4) **Reading order.** When listing text outside of tables, use **top-to-bottom, left-toright** order.
- 5) **Language.** OCR text must remain in the **original language(s)**. The caption should be in English unless the visible UI/page language is clearly not English; in that case,

keep captions in that language. Do **not** translate OCR unless the image itself contains a translation.

- 6) **No extra sections.** Output **only** the three required sections and nothing else.

--### Image Type Identification (Section 1)

- - Classify the image using one or more of the following types (multiple allowed if appropriate):

‘chart‘, ‘table‘, ‘document/text page‘, ‘diagram/flowchart‘, ‘map‘, ‘UI/screenshot‘, ‘ form‘, ‘invoice/receipt‘, ‘poster/flyer‘, ‘scientific figure (multi-panel)‘, ‘natural scene‘, ‘legend‘, ‘infographic‘, ‘other (specify)‘.

- --### Detailed Caption (Section 2)

Write a **dense, structured** caption that covers all critical elements. Use clear, objective language and organize logically (left->right, top->bottom; or foreground-> background). Include the relevant sub-guidelines:

- **A. Charts / Plots / Scientific Figures**

- - State the figure title (if present), chart type(s), axes titles, **units**, tick labels, gridlines, data series, markers/line styles, **legend and color/shape mappings**, annotations, error bars, trend lines, and notable extrema/patterns (peaks, troughs, monotonic trends, outliers).
- - If multi-panel: identify panel labels ‘(a)‘, ‘(b)‘, ... and summarize each panel in order

.

- - Mention any insets, callouts, footnotes, or sources.

- **B. Tables / Forms / Receipts / Documents**

- Summarize what the table/document contains (topics/fields), approximate dimensions (e.g.,

‘˜12 rows x 6 columns‘), header rows, merged cells, checkboxes, stamps, signatures, page numbers, and footers/footnotes.

- Call out key sections (headings, lists, paragraphs), logos, and seals.

- **C. Maps**

- - Report title, compass/north arrow, scale/scale bar, coordinate grid or lat/long,

boundaries, regions, routes/lines with **color-to-meaning mapping** (from legend), symbols/ icons (e.g., hospitals, stations), labels for places/roads, insets, and any zoning/heat color ramp.

- - Include legend content (categories and their visual encodings).

- **D. UI / Screenshots**

- App/site name, window title, menus/toolbars, visible controls (buttons, toggles, checkboxes, dropdowns, search fields), selected/disabled states, scroll position, timestamps, status bars, notifications, dialogs, visible file paths, and version strings.

- **E. Natural / Real-World Scenes**

- Enumerate salient objects, text on signs/labels, relative positions (e.g., ‘‘a red sign above the doorway"), counts for repeated items, conditions (day/night, indoor/outdoor), and

visible brands/logos.

> Do **not** invent interpretations or causal explanations. Keep to what is visually supported.

--### OCR Extraction (Section 3) Extract **all visible text**. Follow these rules:

**General OCR Rules**

- - Use **natural reading order** (top->bottom, left->right).
- - Preserve original line breaks and spacing.
- - If text is repeated (e.g., in a watermark), list it once and note ‘(repeats)‘ if necessary.
- - If a character/word is uncertain, write it as ‘[illegible]‘ or ‘[?]‘ without guessing.

**Tables (very important)**

- - When an area is a table, extract it immediately using **Markdown table syntax**.
- - Preserve the **exact** row/column structure and header rows; if a cell has line breaks, use ‘<br>‘ or ‘\n‘.
- - For merged cells, repeat the visible text in each affected cell and note ‘(merged)‘ once after the table.
- - If multiple tables exist, label them sequentially as ‘Table 1‘, ‘Table 2‘, ... in the order they appear.

**Documents / Text Pages / Forms**

- - Extract headings, paragraphs, lists, captions, footnotes, headers/footers exactly as shown. Maintain indentation and list markers.

**Charts / Maps / Diagrams / UI / Natural Scenes**

- - Extract **all textual elements** present: titles, subtitles, axis labels, tick labels, legend entries, series labels, annotations, callouts, map labels (places/roads/lines), UI labels (menu items, buttons, tooltips), signs, badges, and watermarks. If **no text** is present, write ‘None‘.
- --### Output Format (return EXACTLY this Markdown structure; no code fences)

## Image Identification <one or more types from the allowed list; add brief justification if mixed>

## Detailed Caption <dense, strictly factual caption covering all visible elements per rules above>

## Extracted OCR (if any)

<EITHER: full Markdown tables + remaining text in reading order; OR: all non-table text in reading order; write "None" if no text>

- E.2 PROMPTS FOR CONSTRUCTING O3-BENCH

For chart images, we use the following prompt for GPT-5 (OpenAI, 2025a) to automatically generate QA instances.

You are an expert assessment-item author who designs rigorous **multi-hop visual-reasoning

** questions to benchmark "think-with-images" abilities on **dense diagrams, charts, tables , and schematics**. The goal is to generate items so challenging that today’s strongest MLLMs score **<= 50%** without external tools.

You will receive: ### INPUTS

- 1. **GLOBAL_OCR** - OCR or caption text describing the entire image/page. {GLOBAL_OCR}
- 2. **GLOBAL_CAPTION** - caption text describing the entire image/page. {GLOBAL_CAPTION}
- 3. **LAYOUTS** - a list of cropped regions. Each layout contains:

- - ‘layout_id‘ - unique numeric ID (for your internal reference only).
- - ‘caption_or_ocr‘ - OCR or descriptive caption of the cropped region. {LAYOUTS}

> **Important constraint:** > In the **question** and **options**, you must use only natural labels/text present in the

‘GLOBAL_OCR‘ or ‘GLOBAL_CAPTION‘ or ‘caption_or_ocr‘. > **Never** mention ‘layout_id‘, ‘region‘, ‘crop‘, ‘panel‘, ‘box‘, or any similar tokens. Layouts are **reference-only** to help you construct questions; they may be cited in the ** explanation** but not in the question or options.

--## TASK Create **3-5 independent multiple-choice questions** that each requires **>=2 distinct visual hops** across layouts or global text. These must be **multi-step items directly**, not derived from single-step seeds. All facts must be grounded in the provided OCR/captions

only.

--## QUESTION-DESIGN RULES

- 1. **Multi-hop inference (required)** Each question must integrate information from at least **two different layouts** or from

global + local text. Valid patterns include:

- - Cross-table lookup & join (match category in Layout A to code/key in Layout B, then filter by condition).
- - Table <-> chart alignment (map series/labels from one layout to another).
- - Diagram <-> table mapping (use schematic labels to query corresponding rows/values).
- - Temporal alignment (identify when a threshold is crossed in one chart, then fetch related info from another).
- - Proportions/ratios/ranks (compute shares from one region and compare with targets in another).
- - Exceptions/constraints (apply footnote conditions from one layout before interpreting values elsewhere).

- 2. **Analytical realism** Situate each question in a plausible scenario (finance, science, education, operations, product metrics, etc.) while remaining strictly grounded in the provided OCR/captions.
- 3. **Difficulty control** Questions should require careful scanning, cross-referencing, and light calculations ( differences, ratios, ranking). Avoid items that can be answered at a glance.
- 4. **Units, scales, rounding** Always follow the units/scales given in the OCR. If rounding is necessary, state the rounding rule in the **explanation**.
- 5. **Ambiguity guardrails** Ensure exactly **one correct choice** among A-E, unless ‘F‘ ("No right choice") is correct. Adjust conditions to avoid ties.
- 6. **Label fidelity** Copy text exactly as provided (case, spelling, diacritics). Never use external/world knowledge.

--## ANSWER-OPTION RULES

- 1. Provide **exactly six** options, one per line, labeled ‘A.‘ ... ‘F.‘
- 2. ‘F.‘ must always be exactly ‘No right choice‘.
- 3. Place the correct answer randomly among ‘A.‘-‘E.‘; <=10% of items may have ‘F‘ as the correct answer.
- 4. Distractors must be plausible and drawn from actual text/numbers in the OCR/captions.
- 5. Make options mutually exclusive’ no meta-options like "All of the above".

--## EXPLANATION RULES

- - Provide a **step-by-step chain**: ‘Step 1: ...‘, ‘Step 2: ...‘, etc.
- - Explicitly cite which layouts were used as ‘[layout X]‘.
- - Show all computations (e.g., "(132 - 95) / 95 = 0.389 = 38.9% [layout 4]").
- ---

## OUTPUT FORMAT Return **only** the following JSON array - no extra text, no markdown outside the code block, no commentary:

‘‘‘json [

{{

"question": "...", "options": "A. ...\nB. ...\nC. ...\nD. ...\nE. ...\nF. No right choice", "answer": "C", "explanation": "Step 1: ... [layout 2]\nStep 2: ... [layout 5]\nStep 3: ..."

}}, ...

] ‘‘‘

For map images, we use the following prompt for GPT-5 (OpenAI, 2025a) to automatically generate QA instances.

You are an expert evaluation-item author who designs rigorous **multi-hop visual-reasoning

** questions to benchmark "think-with-images" abilities on **maps** (bus/metro networks, terminals, malls, festivals, parks, etc.). Items should be challenging enough that today’s

strongest MLLMs achieve **<= 50%** accuracy. You will receive: ### INPUTS

- 1. **GLOBAL_OCR** - OCR or caption text describing the entire map. {GLOBAL_OCR}
- 2. **GLOBAL_CAPTION** - caption text describing the entire map. {GLOBAL_CAPTION}
- 3. **LAYOUTS** - a list of cropped map regions. Each layout contains:

- - ‘layout_id‘ - unique numeric ID (for your internal reference only).
- - ‘caption_or_ocr‘ - OCR or descriptive caption of the cropped region. {LAYOUTS}

> **Hard requirement:** In the **question** and **options**, you must use only natural labels/text found in ‘GLOBAL_OCR‘ or ‘GLOBAL_CAPTION‘ or ‘caption_or_ocr‘.

> Do **not** mention ‘layout_id‘, ‘region‘, ‘crop‘, ‘box‘, ‘panel‘, or similar. Layouts are

**reference-only** for your reasoning; you may cite them in the **explanation**, but never in the question or options.

--## TASK Generate **3-5 independent multiple-choice questions**. Each question must require **>=2 distinct reasoning hops** that combine information across different layouts or between global and local OCR. All facts must be image-grounded.

---

- ## 1) QUESTION-DESIGN RULES

- 1. **Multi-hop reasoning (mandatory).** Examples of valid hops:

- - Legend <-> line color <-> stop/zone.
- - Grid index <-> label <-> adjacency.
- - Level/floor marker <-> facility <-> inset.
- - Route <-> timetable <-> destination.
- - Symbol <-> restriction <-> path feasibility.

- - Distance/scale <-> number of segments <-> travel time.

- 2. **Image/OCR grounded only.** Do not use external/world knowledge.
- 3. **Diversity.** Vary question styles (routing, conditional reachability, transfer logic, adjacency/containment, count/compare, scale-based).
- 4. **Difficulty target.** Avoid "at-a-glance" items. Require cross-checking, counting, or lightweight calculation.
- 5. **Label fidelity.** Copy map labels exactly (case, spelling, diacritics).
- 6. **Uniqueness.** Ensure exactly **one correct answer** among A-E, unless F ("No right choice") is deliberately correct.
- 7. **Units & scale.** If computing length/time/segments, use the map’s own scales, symbols, or counts.

---

- ## 2) ANSWER-OPTION RULES

- 1. Provide **exactly six** options labeled ‘A.‘ ... ‘F.‘.
- 2. ‘F.‘ must always be ‘No right choice‘.
- 3. Normally, the correct answer is among A-E; only rarely (<10%) should ‘F‘ be correct.
- 4. Distractors must be plausible, drawn from real map text/numbers, and mutually exclusive.
- 5. No meta-options ("All of the above").

---

- ## 3) EXPLANATION RULES

- - Provide a **step-by-step reasoning chain**.
- - Explicitly cite layouts used as ‘[layout X]‘.
- - Make hops and computations explicit (e.g., "Count 5 stops along Red Line [layout 3] and compare to 4 stops in Zone B [layout 5]").
- ---

- ## 4) OUTPUT FORMAT Return **only** the following JSON array-no extra commentary or markdown:

‘‘‘json [

{{

"question": "...", "options": "A. ...\nB. ...\nC. ...\nD. ...\nE. ...\nF. No right choice", "answer": "B", "explanation": "Step 1: ... [layout 1]\nStep 2: ... [layout 4]\nStep 3: ..."

}}, ...

] ‘‘‘

- E.3 PROMPTS FOR EVALUATION

For proprietary models/systems, we adopt the following thinking prompt (Guo et al., 2025a) as their system prompt. The prompt is used across all benchmarks except O3-Bench.

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within < think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.

On O3-Bench, we use the models’ default system prompts as they usually lead to better performance. On other benchmarks, the performance difference is negligible (mostly within 1%) except only that GPT-5-nano performs much worse on VisualProbe-Hard when the default system prompt is used (accuracy dropping from 21.7% to 16.0%).

For open models, e.g., InternVL3.5 (Wang et al., 2025e), Qwen2.5-VL (Bai et al., 2025b), and Qwen3-VL models (Bai et al., 2025a), we use their default system prompts since they are relatively sensitive to the system prompt. We note that the thinking prompt above often leads to suboptimal performance of Qwen3-VL models.

- F PROMPTS FOR INSIGHT-O3

- F.1 VREASONER PROMPTS We use the following prompts for vReasoner during training.

[SYSTEM] You are a visual assistant. Your goal is to answer a question based on an image.

First, think step by step to identify which visual facts you need from the image to answer the question. If the visual information is insufficient or unclear, call the visual search tool by providing a concise region description: <tool_call>region_description={...}</tool_call>

The tool will search the image and return a cropped view of the target region. You may repeat this process until you have enough evidence to answer confidently. The tool is not always precise -- evaluate its output critically. If it looks incorrect or off-target, refine your description and try again.

Region description guidance:

- - Use concise, visually grounded targets (e.g., a chart, an object, a text block, a distinct area)
- - Optionally include approximate location (e.g., top-left, bottom-right, center)
- - Avoid non-visual or ordinal references (e.g., "the third largest bar", "the second row’s number")
- - Describe only one region per tool call; do not request multiple regions in a single description Output format:
- - Put your reasoning process inside <think>...</think>.
- - Immediately after </think>, output your assessment of the most recent tool result (if any ) formatted as <tool_feedback>helpful/unhelpful</tool_feedback>.

This should indicate whether the result returned by the previous tool call is relevant to

your prior region description and helpful to answering the question. If it misses the key information you are looking for, it is unhelpful. If no previous tool result exists ( e.g., the first turn), output <tool_feedback>NA</tool_feedback>.

- - Immediately after </tool_feedback>, do exactly one of:

- 1) Call the tool; or
- 2) Provide the final answer (no tool call) -- include the result in \\boxed{...}. Do not mix tool calls and answers in the same turn.

- - If you need to call the tool, provide the region description using the exact format < tool_call>region_description={...}</tool_call>. You must strictly follow the output format, otherwise your answer will be judged as wrong.

A multi-turn format example: Assistant: <think>{your step-by-step analysis; decide if more detail is needed}</think> <tool_feedback>NA</tool_feedback> <tool_call>region_description={concise, visually grounded target (optionally with location) }</tool_call>

User: [Zoomed-in image + guidance (e.g., "Based on your description, here is the zoomed-in image.

Please continue your analysis; you may call the tool again or provide your final answer if sufficient.")]

Assistant: <think>{updated analysis based on the zoomed-in view; decide whether to refine or answer}</ think> <tool_feedback>unhelpful</tool_feedback> <tool_call>region_description={next concise target (optionally with location)}</tool_call>

(Repeat the User -> Assistant pattern as needed until enough evidence is gathered.) Assistant (final turn): <think>{final reasoning; explain why the available visual evidence is sufficient}</think> <tool_feedback>helpful</tool_feedback> Answer: \\boxed{...} [USER] {question}<image> [ASSISTANT]

... [USER] Based on your description, here is the zoomed-in image.

Please continue your analysis. After the analysis, state your assessment of the previous tool result using <tool_feedback>helpful/unhelpful</tool_feedback>, then do one of the following:

- - Call the tool again if you believe more visual detail is needed; or
- - Provide your final answer if the current information is sufficient. <image>

[ASSISTANT]

... [USER] Based on your description, here is the zoomed-in image. You have reached the limit for using the visual tool and cannot call it again. In this turn, after reasoning step by step, output your assessment of the previous tool result using exactly <tool_feedback>helpful/unhelpful</tool_feedback>, and then, based on the available information, provide your final answer using the required format. <image> [ASSISTANT]

...

In case that vSearcher is unable to find a region that matches the region description provided by vReasoner, we use the following user prompt to notify vReasoner.

[USER] The visual searcher could not locate the requested target in the image based on your description.

Please adjust or refine your region description (for example, refer to a larger, clearly visible area) and continue your analysis. Think first, then state your assessment of the previous tool result using <tool_feedback>helpful/unhelpful</tool_feedback>. Finally, do exactly one of the following:

- - Call the tool again with a revised description; or
- - Provide your final answer if the current information is sufficient.

Occasionally, vReasoner may fail to follow the format instructions. When this happens, we use the following user prompt to ask vReasoner to generate a new response:

[USER] In your previous response, neither a tool call nor a final boxed answer was detected, or you didn’t output your assessment of the previous tool result in the correct format.

Think first, and then include your assessment of the previous tool result using exactly < tool_feedback>helpful/unhelpful</tool_feedback> (or <tool_feedback>NA</tool_feedback> if there is no previous result). Finally, do exactly one of the following:

- - If you still need more visual detail, call the tool using the exact format: <tool_call>region_description={...}</tool_call>
- - Otherwise, provide the final answer now and include the result in \\boxed{...}.

During inference, we do not ask vReasoner to provide any feedback to the tool. The prompts are slightly simplified as follows:

[SYSTEM] You are a visual assistant. Your goal is to answer a question based on an image.

First, think step by step to identify which visual facts you need from the image to answer the question. If the visual information is insufficient or unclear, call the visual search tool by providing a concise region description: <tool_call> region_description={...} </tool_call>

The tool will search the image and return a cropped view of the target region. You may repeat this process until you have enough evidence to answer confidently. The tool is not always precise -- evaluate its output critically. If it looks incorrect or off-target, refine your description and try again.

Region description guidance:

- - Use concise, visually grounded targets (e.g., a chart, an object, a text block, a distinct area)
- - Optionally include approximate location (e.g., top-left, bottom-right, center)
- - Avoid non-visual or ordinal references (e.g., "the third largest bar", "the second row’s number")

- - Describe only one region per tool call; do not request multiple regions in a single description Output format:
- - Put your reasoning process inside <think>...</think>.
- - When you need to call the tool, you need to provide the region description using the format <tool_call>region_description={...}</tool_call>.
- - Immediately after each </think>, do exactly one of:

- 1) Call the tool; or
- 2) Provide the final answer (no tool call) -- include the result in \\boxed{...}. Do not mix tool calls and answers in the same turn.

You must strictly follow the output format, otherwise your answer will be judged as wrong. A multi-turn format example: Assistant: <think>{your step-by-step analysis; decide if more detail is needed}</think> <tool_call> region_description={concise, visually grounded target (optionally with location )} </tool_call> User: [Zoomed-in image + guidance (e.g., "Based on your description, here is the zoomed-in image.

Please continue your analysis; you may call the tool again or provide your final answer if sufficient.")]

Assistant: <think>{updated analysis based on the zoomed-in view; decide whether to refine or answer}</ think> <tool_call> region_description={next concise target (optionally with location)} </tool_call >

(Repeat the User -> Assistant pattern as needed until enough evidence is gathered.) Assistant (final turn): <think>{final reasoning; explain why the available visual evidence is sufficient}</think> Answer: \\boxed{...} [USER] {question}<image> [ASSISTANT]

... [USER] Based on your description, here is the zoomed-in image. Please continue your analysis. You may:

- - Call the tool again if you believe more visual detail is needed; or
- - Provide your final answer if the current information is sufficient. <image>

[ASSISTANT]

... [USER] Based on your description, here is the zoomed-in image. You have reached the limit for using the visual tool and cannot call it again. In this turn, based on the available information, provide your final answer using the required format. <image> [ASSISTANT]

...

When vSearcher can’t find the target region, we use the following user prompt to notify vReasoner:

[USER] The visual searcher could not locate the requested target in the image based on your description.

Please adjust or refine your region description (for example, refer to a larger, clearly visible area) and continue your analysis. You may:

- - Call the tool again with a revised description; or
- - Provide your final answer if the current information is sufficient.

When vReasoner fails to follow the format instruction described in the system prompt, we the following user prompt to ask vReasoner to generate a new response:

[USER] In your previous response, neither a tool call nor a final boxed answer was detected.

Please do exactly one of the following:

- - If you still need more visual detail, call the tool using the exact format: <tool_call>region_description={...}</tool_call>
- - Otherwise, provide the final answer now and include the result in \\boxed{...}.

- F.2 VSEARCHER PROMPTS

We use the following prompts for vSearcher during both training and evaluation after training. The prompts are adapted from DeepEyes (Zheng et al., 2025). For the last turn, we notify vSearcher in the user prompt that it has reached tool call limit.

[SYSTEM] You are a helpful assistant.

# Tools You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags: <tools> {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items ":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region

to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the

object in the specified bounding box (optional)."}},"required":["bbox"]}}} </tools>

# How to call a tool Return a json object with function name and arguments within <tool_call></tool_call> XML tags: <tool_call> {"name": <function-name>, "arguments": <args-json-object>} </tool_call>

Example: <tool_call> {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}} </tool_call>

[USER] <image> Locate {target}. Think first, call image_zoom_in_tool if needed, then answer with the bbox coordinates in [ x1, y1, x2, y2] format (or [0, 0, 0, 0] if you can’t locate it). Format strictly as: < think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>[x1, y1, x2, y2]</ answer> (otherwise)

[ASSISTANT]

... [USER] <tool_response><image></tool_response> Think first, call image_zoom_in_tool if needed, then answer with the bbox coordinates in [ x1, y1, x2, y2] format (or [0, 0, 0, 0] if you can’t locate it). Format strictly as: < think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>[x1, y1, x2, y2]</ answer> (otherwise) [ASSISTANT]

... [USER] <tool_response><image></tool_response> You have reached the tool call limit and cannot call tools anymore. Think first, call image_zoom_in_tool if needed, then answer with the bbox coordinates in [ x1, y1, x2, y2] format (or [0, 0, 0, 0] if you can’t locate it). Format strictly as: < think>...</think> <answer>[x1, y1, x2, y2]</answer> [ASSISTANT]

...

Without RL, Qwen2.5-VL-7B has difficulty following the instructions given in the above prompts. So, for evaluating vReasoner + Qwen2.5-VL-7B, we use the following simple prompt:

[SYSTEM] You are a helpful assistant.

[USER] Given an image and a region description, locate the region that best matches the description and output its bounding box coordinates as [x_min, y_min, x_max, y_max].

If the target cannot be found, output [0, 0, 0, 0]. Region description: {target} Now, output the coordinates in format [x_min, y_min, x_max, y_max]: <image> [ASSISTANT]

...

- F.3 PROMPTS FOR ANSWER VERIFICATION For answer verification, we use GPT-5-nano (OpenAI, 2025a) and feed it with the following prompt.

You are given an image-based question, the ground truth (GT) answer, and a model’s answer. Compare the model’s answer with the GT answer:

- - If the model’s answer matches the GT answer visually or semantically, reply with <correct >.
- - If it doesn’t match, or if uncertain, reply with <wrong>. Only reply with <correct> or <wrong>, no explanations.

Question: {question} GT Answer: {gt_answer} Model Answer: {model_answer}

- G EVALUATION BENCHMARKS

- G.1 NATURAL-IMAGE BENCHMARKS

- • V⋆-Bench (Wu & Xie, 2024) is designed to test a model’s ability to attend to high-resolution, detail-rich images. V⋆-Bench consists of 191 challenging natural images (sourced from the SA1B Segment Anything dataset (Kirillov et al., 2023)) and focuses on two fine-grained visual tasks: attribute recognition (identifying specific object attributes like color or material) and spatial relationship reasoning (determining the relative positions of objects). By requiring accurate visual grounding of small details, V⋆-Bench exposes the limitations of models that rely on coarse image understanding.
- • Tree-Bench (Wang et al., 2025a), like V⋆-Bench, uses high-quality, object-dense natural images (drawn from the SA-1B dataset) to evaluate fine-grained visual reasoning. However, Tree-Bench places additional emphasis on traceable evidence and complex reasoning. Each of its 405 visual question-answer pairs is annotated with bounding-box evidence for the correct answer, and many questions require second-order reasoning about object interactions or spatial hierarchies rather than simple identification.
- • VisualProbe (Lai et al., 2025) pushes visual reasoning to an even harder regime. It features high-resolution images with very small target objects and many distractors, making it “super challenging” and necessitating iterative, trial-and-error search by the model. VisualProbe is organized into easy, medium, and hard subsets; VisualProbe-Hard denotes the toughest set of questions (106 in total) that often cannot be solved in a single glance. Compared to V⋆-Bench and Tree-Bench, VisualProbe-Hard scenarios demand an active visual search strategy: the model may need to zoom in on different regions or sequentially explore the image to find relevant details.

Table 9: Comparison of O3-BENCH and related benchmarks.

Avg. resolution (width×height)

Layout/target boxes

Detailed explanations

Multi-hop

Benchmark # of QAs Image domains

V⋆-Bench 191 natural (100%) ✓ ✗ ✗ 2246×1583 Tree-Bench 405 natural (100%) ✗ ✗ ✗ 2152×1615

VisualProbeHard 106 natural (100%) ✗ ✗ ✗ 4944×3980

natural (89%), chart (5%), map (6%)

✗ ✗ ✗ 4024×3503

HR-Bench4K 200∗

natural (>60%), chart (25%), map (6%), etc.

MME-RealWorld ∼29K

✗ ✗ ✗ 2708×1844

chart (47%), map (53%)

✓ ✓ ✓ 4602×3967

O3-Bench 345

∗ Roughly 200 distinct QA pairs. The original paper reported 800 but most are the same questions with scrambled options.

Table 10: Comparison of O3-BENCH and related benchmarks on chart & map.

Avg. # of vSearch steps HR-Bench4K 4032×2509 79.6 2.3

Avg. resolution (width×height)

Avg. acc. of GPT-5-mini

Benchmark

MME-RealWorld 1875×1269 83.8† 1.0 O3-Bench 4602×3967 39.0 2.9

† Based on 500 random samples.

- G.2 MIXED BENCHMARKS

- • HR-Bench (Wang et al., 2025f) is a benchmark deliberately designed to evaluate models on ultra high-resolution images (up to 4K-8K pixels). It addresses a key limitation of prior multimodal tests (which max out at ∼2K resolution) by presenting tasks that cannot be solved with downsampled images. HR-Bench is split into two sub-task categories: (1) Fine-grained Single-instance Perception (FSP), with tasks like identifying detailed attributes of a single object, reading text via OCR, or responding to visual prompts on an image; and (2) Fine-grained Cross-instance Perception (FCP), which includes more complex multi-object challenges such as analyzing maps, interpreting charts/graphs, and assessing spatial relationships among multiple items. Each sub-task contains 100 queries on 8K-resolution images (with a downsampled 4K version also provided for efficiency).
- • MME-RealWorld (Zhang et al., 2024b) is a large-scale, comprehensive benchmark that evaluates models across a wide spectrum of real-world visual tasks. It comprises 13,366 high-quality images (average ∼2000×1500 resolution) and 29,429 QA pairs, spanning 43 distinct task types grouped into five real-world scenarios, curated from various datasets (Agustsson & Timofte, 2017; Liu et al., 2020; Zhang et al., 2021; Yang et al., 2023; Li et al., 2022; Sun et al., 2022; Sachdeva et al., 2024; Zhu et al., 2021; Jia et al., 2021). These scenarios cover diverse applications such as autonomous driving (e.g., interpreting traffic scenes), video surveillance (e.g., counting vehicles in an overhead street video), remote sensing (e.g., identifying and counting tiny objects in satellite maps), sports and entertainment (e.g., reading a scoreboard in a broadcast image), and others. MME-RealWorld is notable as the largest fully human-annotated multimodal benchmark to date. Given the benchmark’s scale, we employ the official MME-RealWorld lite version3 for efficiency, which uses a subset of 50 samples per task to speed up evaluation without significantly altering the task distribution.

- G.3 COMPARISON OF O3-BENCH AND RELATED BENCHMARKS

In Table 9 and Table 10, we compare O3-BENCH with related benchmarks and highlight their key differences. Compared with the most closely related MME-RealWorld, we note that on the overlapping domains (i.e. chart & map): (i) the average image resolution of O3-BENCH is significantly higher (4602×3967 of ours vs 1875×1269 of MME-RealWorld); (ii) the average accuracy of GPT5-mini on O3-BENCH is much lower (39.0% of ours vs 83.8% of MME-RealWorld); and (iii) the average number of visual search steps produced by InSight-o3 for O3-BENCH is 2.9× that of MMERealWorld. In addition, O3-BENCH provides layout boxes and detailed explanations for each QA pair, while most of the benchmarks do not. The explanations can help the community easily verify the correctness of the answers. These differences show that O3-BENCH is of exceptional quality and much harder to solve than the other related benchmarks.

Apart from quality, the scale of O3-BENCH is on par with most of the fine-grained perception multimodal understanding benchmarks, e.g., V⋆-Bench (191), Tree-Bench (405), VisualProbe-Hard (106), HR-Bench-4K (200∗), and commonly-used benchmarks in other multimodal research areas; to list a few: Math: MathVision test-mini (304) (Wang et al., 2024), MathVista test-mini (1K) (Lu et al., 2023); VQA: RealWorldQA4 (765); Embodiment/Spatial Understanding: ERQA (400) (Abeyruwan et al., 2025), RefSpatial-Bench (277) (Zhou et al., 2025); Agent: MIA Bench (400) (Qian et al., 2024), OSWorld (389) (Xie et al., 2024), AndroidWorld (116) (Rawles et al., 2024); Fine-grained Perception: V⋆-Bench. These benchmarks were used to evaluate the performance of Qwen3-VL (Bai et al., 2025a). As with O3-BENCH, these benchmarks are relatively small mainly because of the difficulty in data collection. Nevertheless, they have served the timely purpose of evaluating frontier models in the respective fast-developing areas.

O3-BENCH only focuses on composite charts and digital maps for two main reasons. First, they are representative of most use cases of thinking with images in the digital domain (as opposed to the natural domain). More specifically, composite charts represent structured images (with clear delineations between different layout regions) and often require more abstract reasoning (e.g., computing the difference of two quantities). On the other hand, digital maps represent images with less structure and more organic layouts. They usually require more visual reasoning (e.g., finding the shortest route from location A to B). Together, we argue that O3-BENCH is generally sufficient for evaluating the thinking-with-image capability of current multimodal models in the digital domain. As for the natural domain, there are already a number of high-quality benchmarks for thinking with natural images (some are listed in Table 9). O3-BENCH precisely fills the gap of existing benchmarks while striking a balance between evaluation efficiency and generality.

- 3https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite
- 4https://huggingface.co/datasets/xai-org/RealworldQA

