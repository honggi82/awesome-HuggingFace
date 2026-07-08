## arXiv:2605.12497v1[cs.CV]12May2026

[Figure 1]

[Figure 2]

[Figure 3]

# From Web to Pixels: Bringing Agentic Search into Visual Perception

###### Bokang Yang1, Xinyi Sun2, Kaituo Feng3, Xingping Dong2, Dongming Wu3,†, Xiangyu Yue3,†

1Shenzhen Loop Area Institute 2Wuhan University 3CUHK MMLab Project Website: https://pixel-searcher.github.io/ GitHub: https://github.com/yangbokang/Pixel-Searcher

Visual perception connects high-level semantic understanding to pixel-level perception, but most existing settings assume that the decisive evidence for identifying a target is already in the image or frozen model knowledge. We study a more practical yet harder open-world case where a visible object must first be resolved from external facts, recent events, long-tail entities, or multi-hop relations before it can be localized. We formalize this challenge as Perception Deep Research and introduce WebEyes, an object-anchored benchmark with verifiable evidence, knowledge-intensive queries, precise box/mask annotations, and three task views: Search-based Grounding, Search-based Segmentation, and Search-based VQA. WebEyes contains 120 images, 473 annotated object instances, 645 unique QA pairs, and 1,927 task samples. We further propose Pixel-Searcher, an agentic search-to-pixel workflow that resolves hidden target identities and binds them to boxes, masks, or grounded answers. Experiments show that Pixel-Searcher achieves the strongest open-source performance across all three task views, while failures mainly arise from evidence acquisition, identity resolution, and visual instance binding.

### 1 Introduction

Visual perception is a foundation of multimodal intelligence, not only for recognizing visual entities but also for grounding language-level intent into boxes, masks, and region-level answers. Grounding and segmentation thus serve as key interfaces between semantic understanding and pixel-level perception. With the development of multimodal large language models (MLLMs) [1–6], recent progress has pushed visual perception from visible-category recognition toward grounding implicit targets inferred from internal model knowledge [7–10]. Yet open-world settings introduce a more practical but harder case: The object may be visible, while the evidence needed to identify it lies outside the image and beyond frozen model knowledge [11–14]. Inspired by the recent progress of Deep Research [15–17] in knowledge-intensive tasks, we revisit visual perception from a broader perspective. Recognizing that real-world perception queries often involve up-to-date or knowledge-intensive information rather than direct visual attributes, we ask a natural question: can we build a visual perception search agent that actively performs multi-hop web search and reasoning to gather external knowledge for grounded visual perception?

We formulate this setting as Perception Deep Research, where a model must resolve a target identity from external evidence and bind it to a concrete visual instance. Given an image and a knowledge-intensive query, the target is not directly specified by the image or the query text [11, 12, 18]. The query may refer to an entity through indirect factual clues, such as a role, creator, brand affiliation, release history, recent event, or relation to another entity [13, 14, 19]. Solving the task therefore requires two coupled steps: first turning these clues into an explicit target hypothesis, and then mapping that hypothesis back to the image [7, 20, 21]. This coupling makes the problem different from simply answering a knowledge question. Supporting clues may reveal the correct identity but provide only weak visual cues, while the image may contain multiple plausible instances, distractors, or objects with similar appearance. A model must therefore verify that the resolved identity is compatible with the observed region, rather than relying on knowledge or appearance alone. The key gap therefore lies in converting a resolved identity into a grounded visual output, not merely

†Corresponding author

###### Visual-Cue Segmentation

###### Perception Deep Research

###### Reasoning Segmentation

Inner knowledge → mask

Visual cues → mask

Web knowledge → box / mask / answer

Q: Which device has a launch bundle tied to a racing series whose protagonist was originally introduced under another name in a 1981 game?

Q: What can make the woman stand higher?

Q: Segment the red Car

Web Search & Evidence Chain

[Figure 4]

[Figure 5]

Resolved

Web clue 1 Web clue 2 Web clue n

target

[Figure 6]

Search-based Grounding

| |
|---|

Query→ box

[Figure 7]

Search-based Segmentation

Visual cues

Model knowledge

Research

to know

Query→ mask

- • The target can be resolved from parametric knowledge.
- • Infer what the model already knows.

- • The decisive cue is visible.
- • Locate what is explicitly shown.

[Figure 8]

Search-based

| |
|---|

VQA

Which describe is right?

Region→ Choice

A. Q1 C. Q3

B. Q2 D. Q4

[Figure 9]

Visible reference

Implicit concept

Hidden identity via web

Figure 1: Our Perception Deep Research extends grounded perception from visual-cue reference and internal-knowledge reasoning to web-knowledge search.

in recognizing the entity. This gap motivates Perception Deep Research for open-world visual perception, where models must actively seek external evidence, resolve the hidden identity behind a query, and ground it to concrete visual outputs.

To make Perception Deep Research measurable, we introduce WebEyes, an object-anchored benchmark for evidence-to-pixel visual perception. WebEyes starts from concrete visual instances and builds knowledgeintensive queries, verifiable external evidence, target identities, and spatial annotations around them. This design requires models to infer not only what the target is, but also where it appears. WebEyes supports three complementary task views: Search-based Grounding for box prediction, Search-based Segmentation for mask prediction, and Search-based VQA for region-level answer selection. Together, they evaluate whether external evidence can be reliably converted into grounded visual outputs.

We further introduce Pixel-Searcher, an agentic search-to-pixel workflow for Perception Deep Research. It decomposes knowledge-intensive queries, gathers external evidence, resolves target identities, matches them to visual candidates, and produces the required box, mask, or answer. Experiments show that direct perception models struggle on WebEyes when decisive clues are absent from the image and frozen knowledge is insufficient. Pixel-Searcher improves performance through external evidence and step-wise reasoning, while diagnostic results show that the main bottlenecks lie in evidence acquisition, identity resolution, and visual instance binding, rather than final mask refinement.

Our contributions are threefold:

- • We establish Perception Deep Research for open-world visual perception, where models must actively seek external evidence, resolve the hidden identity behind a query, and ground it to concrete visual outputs.
- • We construct WebEyes, an object-anchored benchmark with verifiable evidence, knowledge-intensive queries, precise box/mask annotations, and three task views: Search-based Grounding, Search-based Segmentation, and Search-based VQA.
- • We propose Pixel-Searcher, an agentic search-to-pixel workflow, and provide diagnostic experiments that reveal key bottlenecks in evidence acquisition, identity resolution, and visual instance binding.

###### (a) WebEyes Generation (b) Pixel-Searcher Visual Perception

###### Input

###### Input

[Figure 10]

[Figure 11]

[Figure 12]

Find the person who, in the month when Samsung

Obj_name: Nintendo Switch 2 Distractor_Obj: PS 5, XBOX S BBox: [727, 539, 537, 323]

Galaxy Z TriFold went on sale in the U.S.,

became global ambassador of the flagship skincare brand under the company acquired by Unilever for $2.7B.

| |
|---|

[Think]: I should decompose the question into sub-problems:

[Think]: First, I examine the target object and extract only its visible features, while avoiding shortcuts from surrounding objects.

- (1) identify the month when Samsung Galaxy Z TriFold went on official U.S. sale;
- (2) identify the skincare brand under Unilever's $2.7B acquisition clue;
- (3) determine who became that skincare brand's global ambassador in the same month.

Features: black screen, dual controllers, blue/orange ring accents.

Next, I search for external knowledge

- [Search 1]: [Info]: the device went on official U.S. sale on January...
- [Search 2]: [Info]: Unilever acquired Carver Korea, maker of AHC...

Samsung Galaxy Z TriFold official US release month

[Search 1]:

###### Nintendo Switch 2

[Info]: The Nintendo Switch 2 launch bundle includes Mario Kart World, which links the device to the Mario franchise...

Skincare brand under Unilever's $2.7B acquisition

[Single-hop Query]:

[Think]: The first two sub-problems are solved: the target month is January 2026 and the skincare brand is AHC. Now I should find the AHC ambassador announced in that month.

Which game console launched with a bundle including Mario Kart World?

[Search 3]: AHC global ambassador January 2026 Winter

[Think]: The current clue is still too direct, so I should add another knowledge hop related to Mario.

[Info]: WINTER was named global ambassador... [Summary]: Resolved entity: Winter Reasoning chain:

[Search 2]

###### Mario Game

Galaxy Z TriFold U.S. sale→ January 2026→ Unilever/Carver Korea→ AHC Pro Derm Aesthe→ Winter

[Info]: Mario (originally known as Jumpman) first appeared in the 1981 arcade game Donkey Kong.

[Search 4]:

[Figure 13]

[Figure 14]

[Figure 15]

[Summary]: I should keep only the reasoning chain relevant to the target and avoid mentioning distractor objects.

AHC Winter

1981 arcade game→ Mario → Mario Kart World bundle

→ Nintendo Switch 2

[Figure 16]

[Figure 17]

Final Mask Output

[Multi-hop Query]:

###### [Grounding]: [338, 309,

| |
|---|

Which device has a launch bundle tied to a racing series whose protagonist was originally introduced under another name in a 1981 arcade game?

654, 1532]

[Tool]: SAM3

Figure 2: Overview of WebEyes generation and Pixel-Searcher inference.

### 2 Related Work

Visual perception with language. Language-guided visual perception spans referring expression comprehension, phrase grounding, and segmentation. RefCOCO-style referring expression comprehension established a common setting where a model localizes an object from a natural-language expression and uses contextual relations among objects as key cues [20, 22]. MDETR extends grounding to sentence-level phrase-region alignment by training an end-to-end detector conditioned on text spans [7]. LISA further broadens languageguided perception by using an MLLM to interpret reasoning segmentation prompts and produce masks [9]. Other grounding and segmentation methods improve open-set detection, mask prediction, video grounding, and region-level multimodal grounding [8, 10, 23–28]. Despite these advances, existing methods usually assume that the target can be identified from the image, the prompt, or model-internal knowledge; our work instead study cases where the target identity must first be resolved from web evidence before it can be grounded or segmented.

Agentic multimodal search. Open-knowledge VQA benchmarks such as OK-VQA expose that visual questions often require factual knowledge beyond the image [12]. MMSearch evaluates whether large multimodal models can act as search engines by decomposing multimodal questions, issuing searches, and synthesizing answers from retrieved evidence [13]. WebWatcher pushes this direction toward browsing-centric vision-language deep research agents that inspect pages and visual evidence during multi-step reasoning [19]. Related work also studies fact-based VQA, search-augmented VQA, multimodal browsing [11, 29–34]. However, existing work mainly studies search as an answer-synthesis tool, a browsing ability, or an auxiliary signal for segmentation. In contrast, our work evaluates whether web evidence can be grounded to object-level outputs through shared annotations across grounding, segmentation, and VQA, while our method explicitly resolves the hidden entity before binding it to a visual instance.

|[Figure 18]<br><br>[Figure 19]<br><br>(a) Search-based Segmentation<br><br>[Question]: There is an outstanding woman in the image who reached the highest point on Earth in 2003, becoming the first woman from her<br><br>country to achieve this feat.<br><br>Segment her.<br><br>|[Figure 20]|
|---|
<br><br>[Input]<br><br>[Output]: Pixel mask|
|---|

|[Figure 21]<br><br>[Figure 22]<br><br>| |
|---|
<br><br>(b) Search-based Grounding<br><br>[Question]: Locate the car whose<br><br>name is identical to a Turkic-<br><br>speaking ethnic group primarily living in the Zagros Mountains, known for weaving exquisite carpets and their nomadic lifestyle.<br><br>| |
|---|
<br><br>[Input]<br><br>[Output]: Grounded region|
|---|

|[Figure 23]<br><br>(c) Search-based VQA<br><br>[Question]: Please choose the correct description for the target. [777,55,1108,697]<br><br>A. The device that replaced its titanium...<br>B. The smartphone in the image belong…<br>C. The "Ultra" level imaging smartphone…<br>D. The device whose GPU codename is inspired by the boy...(Correct)<br><br><br>[Input]<br><br>[Output]: Answer choice|
|---|

- Figure 3: Examples of WebEyes task views: Search-based Segmentation outputs a mask, Search-based Grounding outputs a grounded region, and Search-based VQA selects the correct description for a highlighted target.

### 3 WebEyes Benchmark

Perception Deep Research asks a model to find a hidden target using external evidence and connect it to a precise visual output. Given an image and a knowledge-intensive query, the model must identify the real-world entity referred to by the query, locate the matching instance, and return a task-specific result such as a box, mask, or answer choice. To make this setting measurable, we introduce WebEyes, a benchmark that keeps the full chain from annotated objects to web evidence, queries, and grounded targets. We next describe its tasks, annotation pipeline, quality control, and dataset statistics.

#### 3.1 Benchmark Format and Statistics

Data format. As shown in Figure 3, WebEyes supports three task views built from the same object-level annotation layer. Search-based Grounding (SearchGround) predicts a bounding box from the image and query, Search-based Segmentation (SearchSeg) predicts a mask from the same input, and Search-based VQA (SearchVQA) selects the correct knowledge-rich description for a highlighted grounded instance. These views evaluate grounded perception from complementary perspectives: SearchGround tests whether the resolved entity can be localized, SearchSeg further measures pixel-level shape recovery, and SearchVQA checks whether a grounded region can be matched to the correct external-knowledge description.

Scale and categories. The released benchmark contains 120 images, 473 annotated object instances, and 645 unique QA pairs. These QA pairs define 645 SearchGround samples and 645 SearchSeg samples, while 637 of them also include valid multiple-choice options for SearchVQA, yielding 1,927 task samples in total. Figure 4 shows the category distribution, which covers a wide range of real-world entities.

Comparison with existing benchmarks. As shown in Table 1, RefCOCO-style datasets mainly evaluate language-to-region alignment [20], while ReasonSeg focuses on reasoning-based segmentation without web-based identity resolution [9]. Search-oriented benchmarks such as MMSearch and BrowseComp-VL evaluate browsing ability, but their outputs are usually textual or image-level [13, 19]. WebEyes differs by requiring the searched evidence to be grounded as box-level, mask-level, and region-level verification outputs.

[Figure 24]

Vehicles

7.61%

ICON

27.06%

Product

- 12.69%

Anime

- 13.32%

###### 473

##### Total Objects

Celebrities 23.68%

Pop-IP 15.64%

- Figure 4: WebEyes Category distribution.

Benchmark

Knowledge Source Capability Fine-

Grained Common Web Ground. Seg. QA

RefCOCO [20] × × ✓ ✓ × ✓ ReasonSeg [9] ✓ × × ✓ × ✓ MMSearch [13] ✓ ✓ × × ✓ × BrowseComp-VL [19] ✓ ✓ × × ✓ ×

WebEyes ✓ ✓ ✓ ✓ ✓ ✓

Table 1: Comparison of WebEyes with representative benchmarks.

- 3.2 Annotation Pipeline

Figure 5 shows the construction process. WebEyes follows an object-first workflow: each annotated object is expanded into evidence paths, questions, and task instances, forming a traceable chain from mask/box supervision to external knowledge and grounded evaluation.

Stage 1: Multi-Instance Image Collection. We select images primarily based on multi-instance complexity. Candidate images are collected from web image search, news pages, and social-media posts, focusing on recent scenes involving icons, celebrities, pop-culture IPs, anime/game characters, products, and vehicles. An MLLM-assisted screening step keeps images with multiple recognizable foreground instances and plausible distractors, while removing low-quality, text-dominated, severely occluded, or insufficiently ambiguous images.

- Stage 2: Object Annotation and Visual Parsing. Annotators mark foreground instances, refine

masks with SAM3, and save the mask, box, object name, and category. The agent then summarizes each instance with visual feature text describing its appearance, context, and nearby objects. Each object therefore has visual supervision for evaluation and text cues for retrieval and question generation.

- Stage 3: Chained Evidence Retrieval and Path Discovery. For each annotated object, the agent

performs a three-round chained search, where the result of each round conditions the next round. The search starts by resolving the object into a searchable entity using its name, category, context, and image-checkable cues. The resolved entity is then used with the Google Search API to retrieve public evidence within a six-month window before annotation, focusing on non-visual facts such as recent events, roles, creators, brands, product details, release histories, reports, or entity relations. The retrieved facts are further expanded into connected evidence paths that support multi-hop questions rather than direct entity lookup. The output is an evidence record containing the resolved entity, source URLs, access dates, visual category, and image-checkable cues.

- Stage 4: Knowledge-Based QA Construction. Given an evidence record, the agent generates a

question by hiding the target entity name and direct visual attributes while preserving the factual clues needed to identify it. Single-hop questions use one non-visual fact, such as a creator, brand, role, release, or recent event. Multi-hop questions are built from the chained evidence path and require two or more facts before resolving the visible target.

- 3.3 Quality Control

Quality control combines automatic filtering of candidates solvable by shortcuts with manual verification. The agent filters three failure modes: Closed-book shortcuts, Vision-only shortcuts, and Text leakage or non-uniqueness. This step rejects 38.2% of automatically generated candidates.

The remaining candidates enter manual review. Human reviewers check evidence correctness, target uniqueness, text leakage, mask/box quality, and consistency across SearchGround, SearchSeg, and SearchVQA. Among candidates that pass automatic filtering, reviewers reject another 49.2%. Each retained sample keeps a clear chain from source image to annotated object, external evidence, question, and grounded answer.

Stage 1: Real World

STAGE 4: Knowledge Based

STAGE 2: Visual Parsing & Dense Annotation

STAGE 5 : Automated Filtering

Image Collection

QA Construction

[Figure 25]

Candidate QA Pool

Visual Features (-)

TXT

[Figure 26]

[Figure 27]

Visual Feature Text

[Figure 28]

FILTER1

[Figure 29]

Single-Hop Questions

[Figure 30]

[Figure 31]

Closed-Book Answerable?

Visual Feature Parsing Process

[Figure 32]

MASK

FILTER2

Recent Events (+)

[Figure 33]

Segmentation

[Figure 34]

Solvable from

Mask

Vision Alone?

[Figure 35]

[Figure 36]

Reasoning

FILTER3

STAGE 3: Knowledge Retrieval

[Figure 37]

& Path Discovery

Semantic Leakage?

[Figure 38]

Entry Relations

[Figure 39]

News Articles

[Figure 40]

IMAGE SOURCING & FILTERING

Multi-hop

[Figure 41]

Questions

[Figure 42]

[Figure 43]

Internal Knowledge

Human Review

[Figure 44]

| | |
|---|---|
| | |

- Filtered multi-object Images from web, news

Contract Details

& social media sources

[Figure 45]

Candidate

Entity Node

(e.g. X, Reddit)

Final Benchmark

QA Pool

Wikipedia

- Figure 5: Automated WebEyes construction pipeline. The workflow annotates objects, links entities, searches evidence, generates questions, and filters shortcuts.

### 4 Pixel-Searcher: An Agentic Search-to-Pixel Workflow

Pixel-Searcher is a reference workflow for Perception Deep Research. Instead of treating a knowledge-intensive query as a direct grounding prompt, it converts the task into an agentic search-to-pixel process. As shown in

- Figure 6, Pixel-Searcher contains two phases: Agentic Search & Target Resolution and Agentic Grounding & Tool Use. The first phase searches for missing identity evidence and summarizes the hidden target, while the second phase binds the resolved target to a visible instance and invokes visual tools for task-specific outputs.

- 4.1 Overview Given an image I and a query q, Pixel-Searcher first resolves the hidden target into a structured hypothesis:

h = {e,c,K}, (1)

where e is the resolved entity name, c is its visual category, and K = {kj}mj=1 denotes image-checkable cues distilled from external evidence. This hypothesis bridges web evidence and visual perception: it removes irrelevant reasoning paths from the original query and keeps the information needed for grounding, such as object type, appearance cues, identity clues, or reference evidence.

For forward tasks, Pixel-Searcher uses h to bind the resolved target to a visible region in the image. Search-based Grounding returns this verified region directly, while Search-based Segmentation further invokes a promptable segmentation tool to obtain the final mask. For Search-based VQA, the direction is reversed: given a highlighted region, Pixel-Searcher resolves each answer option into evidence-aware cues and selects the option best supported by the grounded visual evidence.

- 4.2 Agentic Search and Target Resolution

The first phase determines what the query is actually asking the system to find. WebEyes queries may describe targets through events, creators, brands, roles, release history, or recent news, so the target identity is often missing from the image itself. Pixel-Searcher therefore uses an adaptive search–reason loop rather than relying on the original query alone.

The agent first plans the query and decomposes it into searchable sub-goals when needed. It then alternates among three actions: Search, which retrieves external evidence; Reason, which connects retrieved facts and checks whether the current evidence is sufficient; and Resolve, which outputs the current target hypothesis.

###### Phase 1: Agentic Search & Target Resolution

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Query Planning]

###### [ ] [Target Summary]

###### [Search-Reason Loop]

[Figure 50]

[Figure 51]

E EntityName C Category

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Search] [Reason] [Resolve]

[ ] [ ]

Complex Question

[Figure 61]

Input:Multi-Hop K KeyCues Question & Image

[Figure 62]

[Figure 63]

[Figure 64]

[Tool-Assisted Mask Refinement]

[Instance Binding] [Evidence Verification]

| | |
|---|---|
| | |
| | |
| | |

E C

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

E

[Figure 69]

| | |
|---|---|
| | |

C K

Verify K

Box Prompt

SAM

MASK

Output:

Tool-Refined Mask

###### Phase 2: Agentic Grounding & Tool Use

- Figure 6: Pixel-Searcher overview. Forward tasks resolve the hidden entity and ground it to a box; Searchbased VQA matches answer options to the highlighted region.

The loop is bounded by a maximum number of rounds, but the path is adaptive: simple queries may require one factual lookup, while harder queries may require connecting multiple pieces of evidence.

Let E1:T denote the evidence collected within at most T rounds. The resolution agent produces: h = R(q,E1:T). (2)

Unlike a free-form textual answer, h is designed for visual grounding. It contains the final visible entity, its coarse category, and key cues that can be checked in the image. The agent also verifies that the resolved entity is not an intermediate clue, and repairs hypotheses that are unsupported, too generic, or inconsistent with the visual context.

#### 4.3 Agentic Grounding and Tool Use

The second phase turns the resolved target hypothesis into grounded outputs. Pixel-Searcher uses h rather than the original query to guide visual grounding. The workflow invokes grounding tools to obtain possible target regions, and then performs evidence verification to select the region most consistent with both the image and the resolved evidence:

b∗ = Abind(I,h). (3)

This makes grounding a tool-assisted decision process conditioned on external evidence, rather than a one-shot text-to-box prediction.

For Search-based Grounding, the verified region is returned as the final answer:

yˆgrd = b∗. (4) For Search-based Segmentation, the verified region is passed to a promptable segmentation tool:

yˆseg = Tseg(I,b∗), (5)

where Tseg is implemented with SAM3 in our experiments. Thus, Pixel-Searcher focuses on resolving and locating the correct instance, while the segmentation tool handles boundary refinement.

For Search-based VQA, the benchmark provides an image, a highlighted target region b, and answer options {ok}Kk=1. Pixel-Searcher applies the same evidence-integration process in reverse. It resolves each

Table 2: SearchGround results. Bold marks the best result among compared open-source methods.

Vehicles Pop-IP Anime ICON Celebrities PRODUCT Overall

Method

IoU R@0.5 IoU R@0.5 IoU R@0.5 IoU R@0.5 IoU R@0.5 IoU R@0.5 IoU R@0.5

Closed-source Models Doubao-Seed-1.6 52.16 63.89 18.07 20.78 36.69 37.97 24.26 36.00 36.37 39.20 45.77 55.84 31.53 38.98

- Doubao-Seed-1.8 55.87 69.44 30.25 35.06 31.72 32.91 24.98 36.80 37.35 41.60 47.73 57.14 33.28 41.30
- Doubao-Seed-2.0-Pro 60.73 75.00 29.04 32.47 36.32 39.24 27.37 40.00 38.32 43.20 52.76 63.64 35.69 44.41

- GPT-4.1 30.01 27.78 3.95 1.30 6.59 6.33 1.61 0.00 6.29 2.40 13.15 7.79 6.38 3.88
- GPT-5.4 34.36 41.67 19.64 22.08 13.90 7.59 18.89 19.60 20.08 17.60 19.46 18.18 19.53 19.10 GPT-4o 13.94 2.78 1.14 0.00 1.09 0.00 1.47 0.00 0.17 0.00 6.59 2.60 2.44 0.47 Gemini-3.1-Flash-Lite 53.57 63.89 26.05 23.38 30.94 22.78 15.08 18.00 37.71 37.60 40.33 41.56 27.90 28.42 Gemini-3.1-Pro 62.89 75.00 32.13 33.77 24.14 22.78 22.22 29.60 31.64 32.00 45.43 53.25 30.52 35.09

Open-Source Grounding Models

Perception-R1 [4] 16.07 11.11 9.61 6.58 22.16 21.52 4.08 4.00 23.50 22.40 13.15 12.99 14.76 13.10 UniVG-R1 [37] 29.29 33.33 19.02 22.37 31.45 31.65 14.63 19.92 34.68 37.60 25.67 28.57 25.79 28.91 Ground-R1 [38] 44.47 50.00 24.11 28.95 24.23 24.05 9.60 13.67 32.41 32.80 25.23 28.57 26.68 29.67

Open-Source General Models OneThinker-8B [39] 51.19 61.11 28.19 32.89 23.86 24.05 21.81 32.81 37.71 42.40 33.92 38.96 32.78 38.70 InternVL-3.5-8B [40] 10.58 2.78 3.49 0.00 3.39 1.27 1.45 0.00 5.92 1.60 7.75 1.30 4.07 0.78 Qwen3-VL-8B [36] 47.58 58.33 20.13 25.97 24.78 29.11 22.30 28.40 32.54 36.80 31.17 37.66 26.81 32.61

Proposed Method Pixel-Searcher (Ours) 54.74 58.33 23.53 24.68 38.81 43.04 29.68 42.80 35.94 38.40 42.12 48.05 34.17 41.30

option into an entity-level summary and selects the option whose identity and visual cues best match the highlighted region:

a∗ = arg max

Avqa(I,b,ok). (6) In this way, Pixel-Searcher provides an inspectable workflow for WebEyes. Failures can be traced to

k

search planning, evidence integration, target-instance binding, or tool-based mask refinement.

### 5 Experiments

We evaluate whether WebEyes is challenging and Pixel-Searcher improves open-source multimodal models across grounding, segmentation, grounded answer selection, ablations, and failure analysis.

#### 5.1 Experimental Setup

All methods use the same WebEyes inputs, splits, and task-specific output interfaces, without task-specific finetuning. Direct baselines predict boxes from the image and query, segmentation converts the box into a mask with SAM3 [35], and Search-based VQA uses the image, target box, and answer options; Pixel-Searcher differs by inserting hidden-entity search before grounding and mask refinement. We use Qwen3-VL-8B-Instruct [36] as the general Qwen baseline because the Qwen-3.5 showed weaker instruction following in preliminary grounding trials and often failed to output valid bounding boxes. We report percentage scores: IoU and Recall@0.5 for grounding, gIoU and cIoU for segmentation, and exact-match accuracy for Search-based VQA.

#### 5.2 Main Results on WebEyes

The main results show that WebEyes remains challenging and that resolving the hidden entity before visual prediction improves open-source models across all three task views.

Search-based Grounding. Table 2 reports Search-based Grounding results. Pixel-Searcher is the strongest open-source method, improving Qwen3-VL-8B from 26.81 to 34.17 IoU and from 32.61 to 41.30 R@0.5. The gains are clearest in ambiguity-heavy categories such as Anime and ICON, although translating external evidence into precise boxes remains difficult.

Table 3: SearchSeg results. Bold marks the best result among compared open-source methods.

Vehicles Pop-IP Anime ICON Celebrities PRODUCT Overall gIoU cIoU gIoU cIoU gIoU cIoU gIoU cIoU gIoU cIoU gIoU cIoU gIoU cIoU

Method

Closed-source Models Doubao-Seed-1.6 65.92 57.13 23.72 20.66 40.88 32.60 63.81 49.15 47.01 30.39 54.44 44.05 50.38 36.13 Doubao-Seed-1.8 71.40 62.57 38.10 28.45 42.05 23.02 62.67 47.76 46.66 31.20 55.45 44.42 52.64 36.06 Doubao-Seed-2.0-Pro 79.00 71.81 41.66 21.22 49.79 36.87 73.26 61.05 56.21 36.78 62.13 51.53 61.22 43.32 GPT-4.1 24.28 20.34 0.85 0.49 8.10 6.98 0.43 0.24 5.84 3.61 7.19 4.93 4.99 6.79 GPT-5.4 43.03 37.06 20.90 10.21 9.74 7.43 39.43 29.94 16.48 8.64 18.93 14.55 25.09 13.54 GPT-4o 2.18 2.81 0.45 0.44 6.23 5.14 1.25 0.70 0.14 0.08 13.65 8.47 3.42 4.85 Gemini-3.1-Flash-Lite 67.32 65.00 28.44 24.87 29.11 25.07 40.43 26.17 38.40 33.29 45.04 43.51 39.11 35.58 Gemini-3.1-Pro 79.79 72.27 43.80 27.46 38.14 26.75 61.19 48.89 43.89 28.26 65.06 49.86 54.56 38.76

Open-Source Segmentation Models

LISA-7B [9] 15.74 20.84 11.85 18.78 9.09 5.38 2.82 4.32 21.02 16.97 8.67 10.65 11.53 12.82 Seg-Zero-7B [41] 49.43 35.98 32.54 16.17 25.79 21.18 29.57 18.66 39.26 27.36 27.58 15.70 34.03 22.51 Seg-R1-7B [42] 56.54 45.06 26.72 18.93 17.57 14.78 27.89 17.10 36.32 24.75 27.21 21.68 32.04 23.72 Affordance-R1 [43] 11.84 9.49 13.28 6.93 14.15 14.28 30.08 19.27 9.87 7.98 15.82 11.02 15.84 11.50 SAM3-Agent [44] 38.22 37.35 14.51 24.98 13.76 23.58 12.14 9.64 19.88 26.54 20.49 24.94 19.83 24.50

Open-Source General Models OneThinker-8B [39] 48.35 35.41 27.87 17.64 32.32 24.52 31.66 20.57 38.37 23.20 35.01 25.43 35.60 24.46 InternVL-3.5-8B [40] 10.62 10.45 3.72 2.31 4.59 3.67 2.99 2.06 7.67 4.95 5.53 5.82 5.28 5.28 Qwen3-VL-8B [36] 57.63 46.43 33.82 11.23 32.12 26.28 32.84 20.85 36.81 24.70 37.43 25.59 35.78 25.94

Proposed Method Pixel-Searcher (Ours) 61.01 53.19 23.91 15.25 37.57 33.21 39.86 19.19 37.57 33.62 46.33 35.24 39.17 32.41

Table 4: SearchVQA results. Bold marks the best accuracy among compared open-source methods.

Method Vehicles Pop-IP Anime ICON Celebrities PRODUCT Overall Closed-source Models Doubao-Seed-1.6 63.89 29.87 41.77 66.80 53.60 53.25 54.97

- Doubao-Seed-1.8 69.44 28.57 41.77 78.80 54.40 64.94 61.34
- Doubao-Seed-2.0-Pro 69.44 42.86 45.57 81.20 56.80 68.83 65.37

- GPT-4.1 58.33 24.68 37.97 40.00 39.20 51.95 40.22
- GPT-5.4 44.44 28.57 35.44 60.00 48.80 53.25 49.38 GPT-4o 58.33 24.68 27.85 20.40 39.20 40.26 29.97 Gemini-3.1-Flash-Lite 55.56 23.38 43.04 40.40 40.80 49.35 40.68 Gemini-3.1-Pro 83.33 48.05 44.30 68.00 62.40 79.22 63.82

Open-Source QA Models

UniVG-R1 [37] 30.56 19.74 31.65 22.13 36.00 22.08 26.22 SophiaVL-R1 [6] 38.89 23.68 22.78 27.46 34.40 37.66 29.67 VL-Rethinker [45] 33.33 23.68 31.65 27.05 33.60 29.87 29.20 Vision-R1 [46] 13.89 22.37 17.72 27.87 12.80 19.48 21.19

Open-Source General Models

OneThinker-8B [39] 36.11 21.05 29.11 24.59 40.00 23.38 28.26 InternVL-3.5-8B [40] 25.00 29.87 32.91 42.00 36.00 31.17 36.02 Qwen3-VL-8B [36] 47.22 24.68 31.65 35.60 43.20 38.96 36.34

Proposed Method Pixel-Searcher (Ours) 38.89 24.68 34.18 50.40 42.40 42.86 42.24

Search-based Segmentation. Table 3 reports Search-based Segmentation results. Pixel-Searcher again ranks first among open-source methods, improving Qwen3-VL-8B from 35.78 to 39.17 gIoU and from 25.94 to 32.41 cIoU. Category-level gains are strongest in Vehicles, Anime, and Product, indicating that better hidden-entity grounding transfers to box-prompted SAM3 refinement.

Search-based VQA. Table 4 reports Search-based VQA accuracy. Pixel-Searcher improves Qwen3-VL-8B from 36.34 to 42.24 overall accuracy and performs best among open-source methods, with clear gains in Icons and Product; the smaller margin to closed-source models suggests that fine-grained semantic comparison also matters.

These gains are consistent with the benchmark design, where many samples require selecting one instance among several similar objects. WebEyes does not only ask whether a model can segment or localize, but

whether it can first recover the hidden target identity from external evidence. In these cases, the key decision is often instance-level verification: the model must reject visually plausible regions whose identity is inconsistent with the retrieved evidence.

The remaining gap to closed-source systems indicates that search-conditioned perception is still limited by evidence selection, entity resolution, and matching the entity to the right image region rather than by a single output format. In many samples, several plausible objects are visible, and the decisive clue only emerges after external evidence resolution, unlike standard referring perception where visual attributes usually identify the target directly. This makes errors in the search stage especially costly, since an incorrect or unclear entity can still lead to a visually reasonable but semantically wrong region.

#### 5.3 Ablation and Failure Analysis Table 5: Component ablation of Pixel-Searcher.

The ablation study asks which parts of PixelSearcher’s evidence-to-region process are responsible for the final gains. Table 5 removes or simplifies individual grounding cues while measuring both box quality and downstream mask quality, since Searchbased Segmentation depends heavily on whether the resolved entity is first mapped to the correct instance.

Variant IoU R@0.5 gIoU cIoU Full 34.17 41.30 39.17 32.41 w/o contradiction 31.34 37.58 36.51 28.41 w/o direct bonus 30.80 36.96 35.67 27.92 support only 29.84 35.40 34.64 27.68 w/o ref. match 29.00 34.01 33.05 26.47 direct only 22.28 27.48 26.49 24.48 w/o direct cand. 20.14 19.72 20.14 15.71

The largest drops come from removing direct localization cues. Without direct candidates, IoU falls from 34.17 to 20.14 and R@0.5 falls from 41.30 to 19.72, while gIoU/cIoU drop from 39.17/32.41 to 20.14/15.71. However, the direct-only variant is also much weaker than the full system, reaching only 22.28 IoU and 26.49 gIoU. Reference matching and contradiction checking add smaller but consistent gains, showing that direct grounding must be combined with resolved entity evidence and visual verification.

Most failed segmentation samples are search/entity errors: among 389 failures, 304 are search/entity errors, 75 are entity-correct region errors, and only 10 are box-to-mask transfer errors. Thus, about three quarters of the errors occur before the system selects the correct entity, while very few come from converting a localized box into a mask. The remaining entity-correct region errors show that visual matching is still nontrivial even after the external evidence is correct. Overall, WebEyes stresses search planning, entity disambiguation, and visual instance verification under realistic distractors more than mask generation alone.

### 6 Conclusion

This paper introduces perception deep research, where web-derived evidence must be converted into box-level and pixel-level predictions. We build WebEyes and instantiate the setting with Pixel-Searcher, showing that agentic search consistently improves Search-based Grounding, Search-based Segmentation, and Search-based VQA when visual appearance alone is insufficient. The failure analysis shows that the dominant bottleneck is not mask refinement after a correct box, but the earlier path from search planning to entity resolution and visual instance verification. WebEyes provides benchmark infrastructure for this direction, and Pixel-Searcher offers a simple starting workflow for studying how agentic search can identify the right entity and bind it to the right visual instance.

### References

- [1] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [2] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

- [3] Xinlei Yu, Zhangquan Chen, Yongbo He, Tianyu Fu, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, Zhe Cao, Jie Xu, et al. The latent space: Foundation, evolution, mechanism, ability, and outlook. arXiv preprint arXiv:2604.02029, 2026.
- [4] En Yu, Kangheng Lin, Liang Zhao, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, et al. Perception-r1: Pioneering perception policy with reinforcement learning. Advances in Neural Information Processing Systems, 38:94827–94853, 2026.
- [5] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. Advances in Neural Information Processing Systems, 38:143297–143330, 2026.
- [6] Kaixuan Fan, Kaituo Feng, Haoming Lyu, Dongzhan Zhou, and Xiangyu Yue. Sophiavl-r1: Reinforcing mllms reasoning with thinking reward. arXiv preprint arXiv:2505.17018, 2025.
- [7] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetrmodulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1780–1790, 2021.
- [8] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.
- [9] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9579–9589, 2024.
- [10] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024.
- [11] Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. Fvqa: Fact-based visual question answering. IEEE transactions on pattern analysis and machine intelligence, 40(10):2413–2427, 2017.
- [12] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019.
- [13] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Chaoyou Fu, Guanglu Song, et al. Mmsearch: Benchmarking the potential of large models as multi-modal search engines. arXiv preprint arXiv:2409.12959, 2024.
- [14] Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, and Ziwei Liu. Mmsearch-r1: Incentivizing lmms to search. arXiv preprint arXiv:2506.20670, 2025.
- [15] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [16] Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, Shuang Chen, Peng Pei, Xunliang Cai, and Xiangyu Yue. Exploring reasoning reward model for agents. arXiv preprint arXiv:2601.22154, 2026.
- [17] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- [18] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer, 2022.

- [19] Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, et al. Webwatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748, 2025.
- [20] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pages 69–85. Springer, 2016.
- [21] Ronghang Hu, Marcus Rohrbach, and Trevor Darrell. Segmentation from natural language expressions. In European conference on computer vision, pages 108–124. Springer, 2016.
- [22] Dongming Wu, Tiancai Wang, Yuang Zhang, Xiangyu Zhang, and Jianbing Shen. Onlinerefer: A simple online baseline for referring video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2761–2770, 2023.
- [23] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18155–18165, 2022.
- [24] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clip-driven referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11686–11695, 2022.
- [25] Chao Shang, Zichen Song, Heqian Qiu, Lanxiao Wang, Fanman Meng, and Hongliang Li. Prompt-driven referring image segmentation with instance contrasting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4124–4134, 2024.
- [26] Chaoyang Wang, Kaituo Feng, Dongyang Chen, Zhongyu Wang, Zhixun Li, Sicheng Gao, Meng Meng, Xu Zhou, Manyuan Zhang, Yuzhang Shang, et al. Adatooler-v: Adaptive tool-use for images and videos. arXiv preprint arXiv:2512.16918, 2025.
- [27] Tianming Liang, Kun-Yu Lin, Chaolei Tan, Jianguo Zhang, Wei-Shi Zheng, and Jian-Fang Hu. Referdino: Referring video object segmentation with visual grounding foundations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20009–20019, 2025.
- [28] Dongming Wu, Yanping Fu, Saike Huang, Yingfei Liu, Fan Jia, Nian Liu, Feng Dai, Tiancai Wang, Rao Muhammad Anwer, Fahad Shahbaz Khan, et al. Ragnet: Large-scale reasoning-based affordance segmentation benchmark towards general grasping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11980–11990, 2025.
- [29] Kaituo Feng, Manyuan Zhang, Shuang Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, and Xiangyu Yue. Gen-searcher: Reinforcing agentic search for image generation. arXiv preprint arXiv:2603.28767, 2026.
- [30] Weizhe Lin and Bill Byrne. Retrieval augmented visual question answering with outside knowledge. In Proceedings of the 2022 conference on empirical methods in natural language processing, pages 11238–11254, 2022.
- [31] Huanjin Yao, Qixiang Yin, Min Yang, Ziwang Zhao, Yibo Wang, Haotian Luo, Jingyi Zhang, and Jiaxing Huang. Mm-deepresearch: A simple and effective multimodal agentic search baseline. arXiv preprint arXiv:2603.01050, 2026.
- [32] Shuang Chen, Kaituo Feng, Hangting Chen, Wenxuan Huang, Dasen Dai, Quanxin Shou, Yunlong Lin, Xiangyu Yue, Shenghua Gao, and Tianyu Pang. Opensearch-vl: An open recipe for frontier multimodal search agents. arXiv preprint arXiv:2605.05185, 2026.
- [33] Song Tang, Guangquan Jie, Henghui Ding, and Yu-Gang Jiang. Rose: Retrieval-oriented segmentation enhancement. arXiv preprint arXiv:2604.14147, 2026.
- [34] Tianming Liang, Qirui Du, Jian-Fang Hu, Haichao Jiang, Zicheng Lin, and Wei-Shi Zheng. Seg-research: Segmentation with interleaved reasoning and external search. arXiv preprint arXiv:2602.04454, 2026.
- [35] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.

- [36] Qwen Team. Qwen3 technical report, 2025.
- [37] Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, and Yansong Tang. Univg-r1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231, 2025.
- [38] Meng Cao, Haoze Zhao, Can Zhang, Xiaojun Chang, Ian Reid, and Xiaodan Liang. Ground-r1: Incentivizing grounded visual reasoning via reinforcement learning. arXiv preprint arXiv:2505.20272, 2025.
- [39] Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025.
- [40] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [41] Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025.
- [42] Zuyao You and Zuxuan Wu. Seg-r1: Segmentation can be surprisingly simple with reinforcement learning. arXiv preprint arXiv:2506.22624, 2025.
- [43] Hanqing Wang, Shaoyang Wang, Yiming Zhong, Zemin Yang, Jiamin Wang, Zhiqing Cui, Jiahao Yuan, Yifan Han, Mingyu Liu, and Yuexin Ma. Affordance-r1: Reinforcement learning for generalizable affordance reasoning in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9738–9746, 2026.
- [44] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025.
- [45] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.
- [46] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

### A Dataset Samples

Table 6 shows five representative WebEyes source images. For each selected image, all annotated objects are listed with their overlaid target masks, object names, and corresponding knowledge-intensive queries.

Table 6: Five WebEyes source images with all annotated objects, overlaid target masks, object names, and queries.

Image Mask Object Query

[Figure 70]

Please find the person who became a brand ambassador for the South Korean brand NE:AR in 2025 in the image.

Sana

[Figure 71]

Please find the person who performed on stage as a member of a K-pop girl group at the 2025 fashion show held by the brand that collaborated with the film The Bride! to launch a limited four-piece capsule collection in the image.

[Figure 72]

Momo

[Figure 73]

Please find the person who endorses the brand whose name simultaneously encodes the five values of TOUCH, TECH, TEMPO, TREND, and TIME in the image.

Mina

[Figure 74]

Please find the youngest one among the five rabbits in the image.

Sanli

[Figure 75]

[Figure 76]

Please find the youngest female rabbit in the image.

Erli

[Figure 77]

Please find the rabbit that was paired with Clarence during the collaboration between Lovebrush Chronicles and this IP in the image.

Shidan

[Figure 78]

Apple Watch Series 11

Please find the product that comes in both 42mm and 46mm sizes in the image.

Please find the product that uses only one specific metal material for the frame, and there are no other metal versions for this model, and this frame material is the same as the frame material of the Galaxy S26 Ultra in the image.

[Figure 79]

[Figure 80]

Apple Watch SE 3

Continued on next page

###### Image Mask Object Query

[Figure 81]

Nick

Please find the character who knocked a snake unconscious with a frying pan in the film in the image.

[Figure 82]

[Figure 83]

Please find the character who was saved only by an antivenom pen being stabbed directly into the heart in the film in the image.

Judy

[Figure 84]

Please find the character who revealed the map hidden beneath the metal cover of the Lynxley Journal in the film in the image.

Gary

[Figure 85]

Please find the character referred to as the “Zootopia know-it-all” in the film in the image.

Libao

[Figure 86]

Please find the character whose voice actor also voiced the character in Pokemon Horizons who holds the mysterious Poke Ball that can summon the black Rayquaza in the image.

Mika Egashira

[Figure 87]

[Figure 88]

Please find the character whose voice actor also participated in the voice cast of the anime series directed by Daisuke Hiramaki that began airing in January 2026 in the image.

Makoto Kurume

[Figure 89]

Yuzuki Murashige

Please find the character who owns a dog named Chiffon in the image.

[Figure 90]

Please find the character played by Miisha Shimizu in the musical Skip and Loafer, which premiered in March 2026, in the image.

Mitsumi Iwakura

### B Prompt Templates

This appendix lists the main prompt templates used by Pixel-Searcher. The placeholders in braces are filled with the sample-specific question, evidence, entity hypothesis, visual candidates, or answer options at inference time.

#### B.1 Hidden-Entity Search

###### Question decomposition.

You are decomposing a multi-hop visual grounding question into simpler atomic sub-questions that can each be answered by a

single web search. Question: {question} Return strict JSON only: {"sub_questions": ["sub-question 1", "sub-question 2", ...]} Rules:

- 1. 1-3 sub-questions, ordered by reasoning dependency.
- 2. Each sub-question should target one hop of reasoning.
- 3. If the question is already simple, return it as the only sub-question.
- 4. Preserve the final target of the original question. If the question asks about the item/person in the image, the last sub-question must still ask about that final target, not about an intermediate clue.
- 5. Do not let a year, event, or historical clue replace the final grounded entity. Intermediate clues are for resolving the target, not for becoming the target.
- 6. Return only JSON.

###### Multi-round search agent.

You are a multi-round reasoning agent for visual grounding. Your goal is to identify the exact entity described by the question so it can be located in an image.

Original question: {question} Sub-questions: {sub_questions}

Accumulated evidence so far: {evidence}

Interaction round {round_num} of {max_rounds}. Only SEARCH / ANSWER consume an interaction round. THINK does not. Return strict JSON with ONE of these forms:

- 1. {"action": "SEARCH", "query": "your DuckDuckGo search query"}
- 2. {"action": "THINK", "reasoning": "your reasoning based on evidence so far"}
- 3. {"action": "ANSWER", "entity_name": "resolved entity", "visual_category": "phone/person/car/...", "entity_type": "device /person/character/vehicle/object", "key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0}

Guidelines:

- - Use SEARCH to gather information you don’t have yet.
- - Use THINK only to briefly consolidate evidence before the next action.
- - If evidence is still missing or ambiguous, prefer SEARCH over repeated THINK.
- - You may use at most one THINK before you must SEARCH with a different query or ANSWER.
- - Do not repeat the same reasoning across rounds.
- - Use ANSWER only when you are confident about the entity.
- - If ambiguity remains, issue a more targeted SEARCH instead of restating the same conclusion.
- - If this is interaction round {max_rounds}, you MUST use ANSWER.
- - Return only JSON.

###### Forced answer.

You have gathered the following evidence about this question. You must now give your best answer.

Question: {question} Evidence: {evidence}

Return strict JSON: {"entity_name": "best guess entity", "visual_category": "phone/person/car/...", "entity_type": "device/person/character/

vehicle/object", "key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0, "remaining_ambiguities": ["ambiguity1", " ambiguity2"]}

Rules:

- - Return only JSON.
- - Before answering, identify unresolved ambiguities from the evidence.
- - If the evidence already resolves the entity, remaining_ambiguities can be [].
- - Give your best guess even if uncertain.

###### Final target resolution.

Resolve the FINAL visible target of this grounding question. Question: {question} Evidence: {evidence} Return strict JSON: {"entity_name": "final visible target", "visual_category": "phone/person/car/object/...", "entity_type": "device/person/

character/vehicle/object", "key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0} Rules:

- 1. Answer the actual item/person that should be located in the image.
- 2. Do not answer with an intermediate clue entity, historical reference, designer, event, or source article unless that is also the visible target.
- 3. Prefer the concrete visible model/person/character over a generic series or franchise name.
- 4. Only return an exact model/person if the evidence explicitly supports that exact target; otherwise return the best supported visible target.
- 5. Return only JSON.

###### Entity verification.

You are checking whether a resolved entity is actually consistent with a visual grounding question and the gathered evidence.

Question: {question} Proposed entity: {entity_name} Visual category: {visual_category} Entity type: {entity_type} Key cues: {key_cues}

Evidence: {evidence}

Return strict JSON: {"is_consistent": true, "consistency_score": 0.0-5.0, "issues": ["issue 1", "issue 2"], "followup_queries": ["query 1", "

query 2"]} Rules:

- 1. Mark is_consistent false if the proposed entity seems to be the wrong product/person/character/model, too generic, unsupported by evidence, or an intermediate clue rather than the final visible target in the image.
- 2. consistency_score 5 means the entity is well supported and specific.
- 3. If inconsistent, provide 1-2 targeted followup_queries to resolve the remaining ambiguity.
- 4. For model-level answers, exact evidence matters. Do not mark an entity consistent unless the evidence explicitly supports that exact model/person, not just a nearby series, sibling model, platform, or speculative variant.
- 5. Return only JSON.

###### Entity repair.

The current resolved entity for a visual grounding question appears unreliable. Question: {question} Current entity: {entity_name} Known issues with the current entity: {issues} Evidence: {evidence} Return strict JSON: {"entity_name": "better entity", "visual_category": "phone/person/car/object/...", "entity_type": "device/person/character/

vehicle/object", "key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0} Rules:

- 1. Re-resolve the entity from the evidence. Do not stick to the current entity if it is unsupported.
- 2. Prefer the most concrete model/person/character/entity actually supported by the evidence.

- 3. If the question asks about the item/person in the image, answer that final visible target, not an intermediate clue used to identify it.
- 4. Prefer an exact model/person only when it is explicitly supported by the evidence; otherwise step back to the best supported visible target.
- 5. If evidence is insufficient, still return the best alternative guess.
- 6. Return only JSON.

#### B.2 Visual Grounding

###### Visual appearance extraction.

Given search results about the appearance of "{entity_name}", extract a concise visual description focusing on shape, color , size, logos, and distinguishing physical features.

Search results: {search_evidence}

Return strict JSON: {"visual_description": "1-3 sentence description of how it looks", "shape": "compact/tall/flat/...", "color": "primary

color(s)", "distinctive_features": ["feature1", "feature2"]} Rules: Return only JSON. Focus on external appearance.

###### Direct grounding.

You are locating a specific entity in the FIRST image. The FIRST image is the target scene. Any additional images are web reference images of the target entity.

Question: {reference_text} Entity name: {entity_name} Visual category: {visual_category} Key cues: {key_cues}

Visual appearance summary: {visual_description}

Return strict JSON in one of these forms: {"bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0, "reason": "short reason"} {"bbox": null, "confidence": 0.0-1.0, "reason": "short reason"}

Rules:

- 1. bbox must use absolute pixel coordinates in the FIRST image only.
- 2. Use the attached reference images to find the same object/person/icon/model.
- 3. If several similar instances exist, choose the one best matching the cues.
- 4. Return a tight box around one concrete instance only. Avoid broad boxes that cover multiple objects, large empty regions , or the center area between objects.
- 5. If no plausible match exists, return bbox null.
- 6. Return only JSON.

###### Candidate joint ranking.

You are selecting the best matching candidate in the FIRST image. The FIRST image is the full scene with all candidate boxes labeled. The next candidate images are crops in this order: {candidate_order} Any remaining images are web reference images for the target entity.

Question: {reference_text} Entity name: {entity_name} Visual category: {visual_category} Key cues: {key_cues}

Visual appearance summary: {visual_description}

Candidates: {candidate_lines}

Return strict JSON:

{"best_candidate_id": "candidate_x", "runner_up_candidate_id": "candidate_y or ’’", "confidence": 0.0-1.0, "reason": "short

reason"} Rules:

- 1. Compare the labeled boxes in the overview image and the candidate crops jointly.
- 2. Prefer exact instance-level matches, not just same coarse category.
- 3. Use reference images when available.
- 4. Prefer the tighter candidate when two boxes cover the same object but one is broader or contains more background.
- 5. The reason must only cite visible evidence or explicit web evidence. Do not invent jersey numbers, logos, text, colors, or details that are not clearly visible.
- 6. If two candidates are similar, explicitly choose the better one.
- 7. Return only JSON.

###### Candidate scoring.

You are scoring whether a highlighted visual candidate matches a text hypothesis. You will see two target images first: the

full image with the candidate highlighted in yellow, and a zoomed crop of the candidate region. You may also see additional web reference images of the entity after that.

Reference text: {reference_text} Entity name: {entity_name} Visual category: {visual_category} Key cues: {key_cues} Candidate id: {candidate_id}

Visual appearance from web search: {visual_description}

Return strict JSON: {"support_score": 0-5, "contradiction_score": 0-5, "confidence": 0.0-1.0, "reason": "short reason"}

Rules:

- 1. support_score: how much the visual evidence supports this being the entity.
- 2. contradiction_score: how much the visual evidence contradicts it.
- 3. Compare the candidate crop with the visual description and any attached reference images carefully.
- 4. Pay attention to shape, color, logos, text, layout, and distinctive features.
- 5. Broad boxes, multi-object boxes, and mostly-background boxes should receive lower support even if the right object is somewhere inside.
- 6. Only mention details that are actually visible or explicitly stated in the reference text. Do not invent hidden labels, numbers, or logos.
- 7. Return only JSON.

###### Reference-image matching.

You see two images:

- 1. A REFERENCE image of "{entity_name}" found on the web
- 2. A CANDIDATE crop from the target image Does the candidate show the same type/model of object as the reference? Return strict JSON: {"match_score": 0-5, "reason": "short reason"} Rules:

- 1. match_score 5 = definitely the same object type/model.
- 2. match_score 0 = completely different objects.
- 3. Focus on shape, color, brand logos, and distinctive features.
- 4. Ignore background differences; only compare the objects themselves.
- 5. Return only JSON.

###### Visual entity repair.

The current resolved entity does not visually match the objects found in the image. Re-resolve the target entity using both the question and the visible candidate summary.

Question: {question} Current entity: {entity_name} Current visual category: {visual_category}

Visible candidates in the image: {candidate_summary}

Scoring evidence:

{score_summary} Return strict JSON: {"entity_name": "better entity", "visual_category": "phone/person/car/object/...", "entity_type": "device/person/character/

vehicle/object", "key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0} Rules:

- 1. Use the visible candidates as a hard constraint: the answer should be an entity that could plausibly appear in this image.
- 2. Prefer specific model-level entities when the question implies them.
- 3. If the current entity is impossible given the visible candidates, replace it.
- 4. Return only JSON.

#### B.3 Grounding-Based VQA

Option mini-resolution.

Given this description of an entity, identify what it refers to. Description: {text} Return strict JSON: {"entity_name": "identified entity", "visual_category": "phone/person/car/...", "entity_type": "device/person/character/

vehicle/object", "key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0} Rules: Return only JSON.

Grounded option selection.

You see an image with one object highlighted in yellow. A cropped view of that object is also provided. Decide which option best describes this object.

Options: {options_text}

Entity info from search: {entity_info}

Return strict JSON: {"selected_index": 0, "confidence": 0.0-1.0, "reason": "short reason"}

Rules:

- 1. selected_index is 0-based index of the best matching option.
- 2. Use visible evidence from the highlighted object.
- 3. Return only JSON.

#### B.4 Fallback Candidate Generation

###### Object detection.

You are a visual object detector. Identify ALL visually salient foreground objects in this image. For each object output a tight bounding box.

Return strict JSON only: {"detections": [{"label": "short label", "bbox": [x1, y1, x2, y2]}]}

Rules:

- 1. bbox uses absolute pixel coordinates, format xyxy.
- 2. One entry per visible object instance.
- 3. Labels should be short and concrete, e.g. phone, person, car, router, laptop.
- 4. At most {max_boxes} boxes.
- 5. Boxes must be tight; do NOT include large background margins.
- 6. Do not output reasoning or markdown, only JSON.

###### Saliency ranking.

You are ranking detected objects by visual saliency (how visually prominent and attention-grabbing each object is). The image contains these detected objects:

{candidate_list} For each candidate, assign a saliency_score between 0.0 (background clutter) and 1.0 (most prominent foreground object). Return strict JSON only: {"scores": [{"id": "candidate_1", "saliency_score": 0.0}, ...]} Rules:

- 1. Return only JSON.
- 2. Every candidate id must appear exactly once.
- 3. Do not output reasoning.

