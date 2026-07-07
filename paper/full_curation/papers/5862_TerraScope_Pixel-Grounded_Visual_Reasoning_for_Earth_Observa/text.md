[Figure 1]

## TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation

# arXiv:2603.19039v1[cs.CV]19Mar2026

Yan Shu1 Bin Ren1,4* Zhitong Xiong3* Xiao Xiang Zhu3 Beg¨um Demir2 Nicu Sebe1 Paolo Rota1

1 University of Trento 2 BIFOLD and TU Berlin 3 Technical University of Munich 4 MBZUAI

https://shuyansy.github.io/terrascope/

[Figure 2]

[Figure 3]

|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|
|---|

[Figure 8]

###### What proportion of the image is occupied by water?

[Figure 9]

###### Ground Truth: 13%

Global Multi-source

Pixel-Level CoT

[Figure 10]

[Figure 11]

GPT-4o & EarthDial Qwen3-VL TerraScope

[Figure 12]

(b) Terra-CoT

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Water appears to occupy roughly the right third of the image, though it extends slightly more than that in some areas and less in others.

Approximately 36.3% of the image is occupied by water, based on color-based segmentation heuristics.

First, I identify all water area in the image [SEG].

Boundary Relationship Detection

###### Dual Evaluation Metrics

Coverage Percentage Analysis

[Figure 17]

[Figure 18]

Now I count the pixels: water occupies 8,490 pixels out of 65,536 total pixels.

Building Change Estimation

Distance Measurement

###### TerraScope Bench

Answer Accuracy

[Figure 19]

[Figure 20]

Approximately 30%-45% of the image is occupied by water.

Absolute Area Quantification

[Figure 21]

About 50% of the image is occupied by water.

Comparative Area Ranking

[Figure 22]

Therefore, water covers 13% of the entire image area.

Thinking Correctness

(a-1) VLM w/o reasoning

(a-2) VLM with textual reasoning

(a-3) VLM with pixel-grounded visual reasoning (Ours)

(c) TerraScope-Bench

Figure 1. (a-1): The most common Vision Language Model (VLM) without reasoning directly outputs the wrong results. (a-2): Some solutions tried reasoning via textual Chain-of-Thought (CoT). (a-3): Our TerraScope, which takes the pixel-level grounding masks together with textual input, forming the interleaved CoT. (b): Our Terra-CoT 1M dataset. (c): Our TerraScope benchmark.

#### Abstract

Vision-language models (VLMs) have shown promise in earth observation (EO), yet they struggle with tasks that require grounding complex spatial reasoning in precise pixel-level visual representations. To address this problem, we introduce TerraScope, a unified VLM that delivers pixel-grounded geospatial reasoning with two key capabilities: (1) modality-flexible reasoning: it handles singlemodality inputs (optical or SAR) and adaptively fuses different modalities into the reasoning process when both are available; (2) multi-temporal reasoning: it integrates temporal sequences for change analysis across multiple time points. In addition, we curate Terra-CoT, a large-scale dataset containing 1 million samples with pixel-level masks embedded in reasoning chains across multiple sources. We also propose TerraScope-Bench, the first benchmark for pixel-grounded geospatial reasoning with six sub-tasks that evaluates both answer accuracy and mask quality to ensure authentic pixel-grounded reasoning. Experiments show

*Correspondence to <zhitong.xiong@tum.de> and <bin.ren@mbzuai.ac.ae>

that TerraScope significantly outperforms existing VLMs on pixel-grounded geospatial reasoning while providing interpretable visual evidence.

#### 1. Introduction

Earth observation (EO) satellites continuously monitor our planet at unprecedented scales, generating vast imagery archives for environmental monitoring [49], disaster response [48], and resource management [16]. Traditional approaches [13, 19, 20, 45] to EO data analysis rely on taskspecific models, limiting flexibility across diverse applications. Vision-language models (VLMs) offer a paradigm shift: unified models that understand both visual content and natural language, enabling flexible analysis through text-based interaction. Recent domain-adapted VLMs have demonstrated strong performance on standard EO tasks, including image captioning [30], visual question answering [29, 31, 43, 52], and visual grounding [15, 51, 54, 58], leveraging large-scale instruction tuning on remote sensing data.

However, state-of-the-art VLMs struggle with finegrained geospatial reasoning requiring pixel-accurate spa-

tial analysis. As illustrated in Fig. 1, leading generalpurpose models (GPT-4o [34]), reasoning-capable models (Qwen3-VL [1]), and EO-specific variants (EarthDial [43]) all fail to provide accurate answers on tasks such as calculating coverage of a land-cover class given in an image. Recent multi-modal reasoning models [7, 56, 57] have shown promise by grounding visual regions before reasoning. However, they cannot directly transfer to EO due to two fundamental differences: (i) Unlike natural images with discrete objects, EO imagery depicts continuous spatial distributions where land cover types transition gradually. This continuous nature introduces substantial noise when coarsegrained grounding is used, hindering reasoning accuracy. (ii) EO analysis often involves multi-sensor and temporally evolving data. Optical imagery captures surface reflectance, SAR provides all-weather observation, and multitemporal sequences reveal dynamic changes. However, existing VLMs struggle to effectively integrate such modalityflexible, time-varying data for EO reasoning within a single, unified framework.

To address these challenges, we present TerraScope, a comprehensive framework for pixel-grounded visual reasoning in EO. Building upon the recent paradigm of “thinking with images” [44], TerraScope embodies the principle of “thinking with pixels”: it explicitly localizes taskrelevant regions and grounds each reasoning step in pixellevel visual evidence, rather than operating solely within the language domain. Prior VLMs for EO rely on external tools [4, 8, 25, 37] for reasoning. The incorporation of external tools substantially increases the model’s complexity and reduces controllability, making it difficult to achieve pixel-level, intrinsic reasoning. In contrast, TerraScope employs mixed decoders that jointly generate segmentation masks and reasoning traces. The language model autonomously decides when to trigger mask generation and interleave the resulting visual tokens into the reasoning process, enabling dynamic visual grounding throughout multi-step reasoning. Beyond single-date single-modality data, TerraScope supports two independent reasoning capabilities. First, for multi-temporal reasoning, it analyzes observations from multiple time points to deduce temporal changes based on evolving spatial patterns. Second, for multi-modal reasoning, when both optical and SAR data are available, it adaptively selects the most informative modality for each reasoning step through textguided cross-attention, leveraging optical for spectral information in clear regions while relying on SAR for cloudcovered areas. To enable pixel-grounded reasoning at scale, we curate Terra-CoT, a 1M instruction-tuning dataset with pixel-level masks embedded in reasoning traces generated via an automated pipeline, covering global scenes across multi-source EO data. Additionally, existing EO benchmarks [23, 29, 46] primarily focus on visual perception

tasks and lack evaluation of fine-grained visual reasoning capabilities. We introduce TerraScope-Bench, a benchmark specifically designed for pixel-grounded geospatial reasoning. It comprises 3,837 expert-verified questions supporting flexible evaluation with optical-only, SAR-only, or joint optical-SAR data, across both single-date and multitemporal scenarios. Beyond traditional VQA accuracy metrics, TerraScope-Bench introduces dual evaluation metrics that assess both answer correctness and segmentation mask quality, ensuring models genuinely ground reasoning in pixel-level visual evidence.

In summary, our contributions are threefold:

- • We introduce TerraScope, a unified framework for pixelgrounded visual reasoning in EO. It grounds each reasoning step in precise segmentation masks for fine-grained, interpretable spatial analysis, supports multi-temporal change reasoning, and adaptively uses optical or SAR imagery.
- • We curate Terra-CoT, a 1M instruction-tuning dataset with pixel-accurate masks embedded in reasoning traces, enabling scalable pixel-grounded training.
- • We propose TerraScope-Bench, a benchmark of 3,837 expert-verified samples with dual metrics for answer accuracy and mask quality. Experiments on 11 models expose current limitations and demonstrate the effectiveness of TerraScope.

#### 2. Related Works

Earth Observation VLMs. Recent advancements in general-purpose VLMs [18, 26, 59] have shown impressive capabilities across various tasks. However, their limited exposure to remote sensing imagery hinders performance on EO tasks. To address this gap, specialized EO-VLMs have emerged through domain-specific data curation and model adaptation. RSGPT [12] enriches captioning datasets to enhance LLaVA’s conversational abilities on satellite imagery. SkyEye-GPT [52] synthesizes 968K instruction samples for multi-task learning. Beyond image-level tasks, GeoChat [15], SkySenseGPT [31], and LHRS-Bot [33] incorporate visual grounding, region captioning, and reasoning. EarthGPT [54] introduces multi-sensor datasets spanning optical, SAR, and infrared modalities, while EarthMarker and EarthGPT-X [55] enable visual prompting interactions. GeoPixel [38] focuses on pixel-level grounding with grounded conversation datasets. EarthDial [43] scales multi-sensor data across multispectral, hyperspectral, and SAR to improve generalization. VHM [35] proposes datasets with both factual and deceptive questions to improve model honesty. Despite these advances, existing EOVLMs still lack pixel-grounded reasoning capabilities required for fine-grained spatial analysis.

Earth Observation Benchmarks. The rapid development of EO-VLMs has stimulated dedicated evaluation bench-

###### TerraScope Framework

###### Masked Token Selection

Multi-Modal Reasoning

[Figure 23]

[Figure 24]

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

[Figure 25]

[Figure 26]

What percentage of the image is covered by developed area?

Question Image

Visual Encoder Projector

Ground Encoder

Language Tokenizer

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Reasoning rationales

... ... ...

[Figure 31]

[Figure 32]

Opt + SAR

Large Language Model

Mask Decoder

( b ) ( c )

I identify the developed area [SEG]

###### Training

[Figure 33]

[SEG]

Grounding Pretrain

Instruction Tuning

2M Referring expression segmentation pairs

###### 1M Pixel-CoT instruction-tuning data

Large Language Model

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Projector

Projector

Visual Encoder Ground Encoder

LLM Visual Encoder Ground Encoder

LLM

Therefore, developed area occupies 62% of the entire image.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Mask Decoder

Mask Decoder

( a ) ( d )

- Figure 2. Overview of TerraScope. TerraScope generates textual reasoning tokens and segmentation masks in an interleaved manner, where masked visual features are injected at each reasoning step to ensure faithful pixel-grounded reasoning. TerraScope supports multimodal and multi-temporal reasoning across EO data.

marks. RSVQA [29], LHRS-Bench [33], RSIEval [12], and VLEO-Bench [53] evaluate conversational capabilities including classification, captioning, and VQA. VRSBench [23] and GeoChat-Bench [15] incorporate regionlevel grounding for localization evaluation. XLRS-Bench [46] focuses on ultra-high-resolution imagery understanding. GeoBench-VLM [6] is a comprehensive benchmark covering multi-task and multi-sensor EO scenarios. DisasterM3 [48] proposes a bi-temporal benchmark spanning multiple hazards, sensors, and tasks. While recent benchmarks broaden the scope of sensors, tasks, and temporal settings, they still do not rigorously assess models’ capacity for pixel-accurate geospatial inference, leaving a gap in evaluating the precision needed for detailed spatial analysis. Visual Chain-of-Thought. Recent works have explored grounding reasoning processes in visual content by interleaving visual evidence with textual reasoning chains. GRIT [7] interleaves the bounding box coordinates with the natural language reasoning for fine-grained counting. DeepEyes [57], Chain-of-Focus [56], and Mini-o3 [17] employ iterative zoom-in mechanisms that crop and analyze focused regions. VLM-R1 [39] and Visual-RFT [28] leverage reinforcement learning for visual grounding tasks. MintCoT [3] and ICoT [9] select relevant visual tokens through retrieval or attention mechanisms to compose multimodal rationales. However, these methods rely on coarse-grained spatial representations (bounding boxes, crops, or implicit token selection), which are inadequate for geospatial reasoning requiring pixel-level segmentation to capture continuous spatial distributions across multimodal data.

#### 3. Method

In this section, we present the core components of TerraScope and outline how pixel-grounded visual reasoning is

formulated and implemented within our framework.

##### 3.1. Overview

Geospatial reasoning demands fine-grained visual understanding that language-only reasoning cannot provide. In this context, we propose pixel-grounded visual reasoning, where models explicitly generate segmentation masks and ground reasoning in the selected masked visual space. Formally, let f(·) be a VLM composed of a text encoder fT and a vision encoder fV . Given a question Q and an image I, the text encoder produces q = fT(Q) and the vision encoder produces v = fV (I) ∈ RN×D,where N is the number of visual tokens and D is the feature dimension. Traditional VLMs then output an answer via language-only reasoning:

[r1,r2,...,rk,a] = f(v,q), (1)

where k is the number of reasoning steps, ri denotes the i-th textual reasoning step, and a is the final answer. Pixelgrounded visual reasoning interleaves masked visual features with textual reasoning:

[r1,(m1,v1),r2,(m2,v2),...,rk,(mk,vk),a] = f(v,q),

(2) where at each reasoning step i, the model generates a segmentation mask mi and selects masked visual features vi from the identified regions. In the rest of this section, we first present the TerraScope architecture that enables joint generation of masks and reasoning (Sec. 3.2), then describe Terra-CoT, our instruction dataset with interleaved visual and textual traces (Sec. 3.3).

##### 3.2. TerraScope Framework

As shown in Fig. 2, our TerraScope builds upon a visionlanguage architecture augmented with a pixel-level segmen-

Cap-CoT

L1-Level VQA

###### . . .

Question: Identify land cover types and their visual locations step-by-step.

[Figure 45]

[Figure 46]

###### Existence Counting Localization

[Figure 47]

[Figure 48]

Question: Does the crop exist?

Question: what does the water locate?

###### Cap-CoT

Rationale: First, I see water [SEG], then I see crops [SEG]...

Image

Mask

ClassInfo MLLMs

TerraScopeCap

water developed area bare land grass crop tree shrub

Rationale: I can see crops [SEG]...

Rationale: I can see water [SEG]...

###### Caption:

starting from the bottom left, there is a river, which is surrounded by crops and grass...

Terra-CoT

Answer: Yes

Answer: bottom-left

[Figure 49]

Cap-CoT L1-level VQA L2-level VQA

| | |
|---|---|
| | |
| | |

Optical

L2-Level VQA

- L1-level VQA
- L2-level VQA

[Figure 50]

Semantic reasoning Spatial reasoning

[Figure 51]

Captions

Question: Is the region suitable for farming?

Question: Is the water adjacent to the crops ?

[Figure 52]

SAR

Image VQA CoT

###### Rationale:

###### Rationale:

First, I identify water sources [SEG], providing irrigation. Then I observe existing crops [SEG] here...

According to the relative position of water [SEG] and crops [SEG], their boundaries do not touch...

Masks

Temporal

Answer: Yes Answer: No

- Figure 3. Terra-CoT curation pipeline. First, we generate Cap-CoT using ground truth masks and class labels to train an initial annotation model. Second, we use the trained model to annotate unlabeled data with pixel-accurate masks and captions. Third, based on the synthetic annotations, we apply hierarchical data synthesis to generate diverse reasoning questions with chain-of-thought traces at two levels: (L1) basic spatial grounding and (L2) complex multi-step reasoning including spatial and semantic tasks.

tation module, forming a unified framework that integrates visual grounding and language-based reasoning within a single model. Specifically, we leverage InternVL3 [59] as our base model, which dynamically splits single images into sub-tiles while processing multi-image inputs independently, thereby defining a unified pipeline to transform all data into a uniform format.

Pixel-Grounded Chain-of-Thought. The core innovation of TerraScope lies in the cooperative mechanism between dual decoders, which interleaves segmentation mask generation with text generation. Specifically, during the reasoning process, TerraScope monitors the language decoder’s autoregressive output and triggers the mask decoder upon detecting [SEG], which typically appears after mentions of key regions or objects. The mask decoder then predicts segmentation masks, from which masked visual tokens are selected and injected into the reasoning sequence to guide subsequent generation. For example, when answering “Which is larger, water or road?”, the model generates “I first identify water regions [SEG]...then road regions [SEG]” and derives the answer by comparing their masked visual features.

As shown in Fig. 2 (b), to inject high-quality visual representations corresponding to the generated mask into reasoning traces, we first align the mask mi with the visual encoder’s dynamic patch layout by resizing it to the token grid resolution (n · s) × (m · s), where the image is split into n × m patches with each patch producing s × s tokens (s = 16 for InternVL). To handle partial overlap between the pixel-level mask and token grid, we select a visual token if the mask covers more than 50% of its corresponding spatial region. For the masked region, we extract the selected visual features as:

vi = {vj | mtoki [j] = 1,j ∈ [1,N]} (3)

where vj denotes the j-th visual token in the feature map, and mtoki is the token-level mask derived from mi by resizing to the token grid. The selected visual features vi are then projected and flattened into a 1D sequence aligned with text embeddings, and fed into the LLM to resume autoregressive text generation conditioned on the KV cache of previously generated tokens.

Multi-Modal and Temporal Reasoning. Unlike singleimage understanding, EO data often involves multi sources including optical-SAR pairs and temporal sequences. TerraScope handles these diverse scenarios through its flexible pixel-grounded reasoning framework.

For optical-SAR pairs, the model must identify complementary features, leveraging optical imagery for spectral information under clear conditions while relying on SAR for cloud-covered regions. We achieve this through textguided, token-level modality selection. As shown in Fig. 2 (c), given optical and SAR images processed independently through the vision encoder to obtain visual features vopt and vSAR, and question embeddings q from the text tokenizer with length L, we compute cross-attention between text and each visual modality, then aggregate across text tokens to obtain text-relevance scores:

L

###### vµq⊤

1 L

βjµ =

Softmax

, µ ∈ {opt,SAR}

√

D jℓ

ℓ=1

(4)

where βjµ denotes the relevance score of the j-th visual token to the question for modality µ. When selecting masked

visual features vi, we select features from the modality with a higher relevance score for each token position:

vjopt if βjopt > βjSAR vjSAR otherwise

, ∀j where mtoki [j] = 1 (5)

vj =

This dynamic, spatially adaptive mechanism leverages the

complementarity of paired EO data to boost reasoning.

For temporal sequences, a critical challenge is temporal disambiguation: when reasoning involves multiple observations, each [SEG] token must specify (1) which temporal image the mask decoder should segment from, and (2) from which image to extract the masked visual tokens. To address this, we incorporate explicit temporal indicators in the format “Image: ti” before each [SEG] token. When the language decoder generates these signals, the mask decoder segments from image ti and the feature extraction module samples visual tokens from v(t

i). The model learns to generate timestamps from our Terra-CoT dataset, which contains temporally grounded reasoning traces paired with frame-specific masks (Sec. 3.3).

Training. We train TerraScope in two stages using supervised fine-tuning. We first train on 2M referring expression segmentation pairs to establish basic grounding capability. We then fine-tune on 1M Terra-CoT samples to incentivize pixel-grounded visual reasoning ability. During training, we extract masked visual features from ground truth masks and interleave them into the sequence at positions following [SEG] tokens. The training objective combines language modeling loss (cross-entropy on text and [SEG] tokens, excluding injected visual features) and segmentation loss (Dice loss and pixel-wise cross-entropy):

L = LLM + λLseg, (6) where we set λ = 0.5 to balance both objectives.

##### 3.3. Terra-CoT Dataset

Curating pixel-grounded visual CoT data is non-trivial: existing EO datasets provide either segmentation labels [5, 50] or VQA pairs [29], but not both with reasoning traces. We address this with a two-stage automated pipeline enabling large-scale pixel-grounded reasoning data.

Grounded Captioning with Chain-of-Thought. We leverage existing datasets with semantic annotations [5, 22, 50] to construct pixel-grounded captioning data with reasoning traces (Cap-CoT). As shown in Fig. 3, we prompt a large multimodal model with an image where distinct landcover categories are highlighted using colored masks and labeled accordingly. The model is instructed to produce detailed captions that explicitly reference these masked regions throughout its reasoning. This process yields 250K Cap-CoT samples, used to both train TerraScope and build an intermediate annotator, TerraScope-Cap, capable of generating pixel-grounded captions for unlabeled imagery.

Hierarchical Data Synthesis. Using TerraScope-Cap trained on Cap-CoT, we annotate images from diverse sources (optical, SAR, temporal) covering global regions with multi-category pixel-level labels (statistics in Appendix). Based on these annotations, we synthesize TerraCoT through a two-level hierarchical process.

Coverage Percentage Analysis Absolute Area Quantification

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Q: What is the total area of developed land measured in square meters?

Q: What fraction of the image is occupied by developed land? A. 13% B. 19% C. 25% D. 42%

A. 1420600 m2 B. 5682400 m2 C. 284120 m2 D. 2841200 m2

Comparative Area Ranking Boundary Relationship Detection

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Q: Which covers a larger area, grass or tree?

Q: Is there a direct boundary between the developed area and the tree?

A. grass B. tree A. No B. Yes

Distance Measurement

Building Change Estimation

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Q: What is the shortest distance separating water from grass?

Q: What percentage of buildings were destroyed in this disaster?

A. 70m B. 60m C. 50m D. 35m A. 31.6% B. 50% C. 31% D.68%

Figure 4. Examples of TerraScope-Bench.

- Level 1 (L1): Basic spatial grounding. We generate

template-based questions for randomly selected categories, covering fundamental spatial tasks such as existence verification, object counting, localization, area quantification, and boundary detection. For each question, we synthesize pixel-grounded reasoning traces using segmentation labels to explain the spatial analysis process.

- Level 2 (L2): Complex multi-step reasoning. We prompt

an LLM to compose multiple L1 questions into complex reasoning tasks of two types: (1) L2-Spatial requires crossentity spatial analysis such as relationship inference (e.g., “Is the water adjacent to the crops?”); (2) L2-Semantic requires domain knowledge beyond visual observation such as land suitability assessment (e.g., “Is the region suitable for farming?”). For both types, the LLM synthesizes reasoning traces combining visual evidence with spatial or semantic analysis. This hierarchical process produces 1M Terra-CoT samples with diverse reasoning abilities.

#### 4. TerraScope-Bench

EO imagery above 10m resolution presents unique challenges: individual objects span only a few pixels, and landuse boundaries become ambiguous, making precise pixellevel spatial reasoning essential. However, existing benchmarks (e.g., BigEarthNet [5], ChatEarthNet [50],) emphasize coarse-grained tasks such as scene classification and image captioning that depend primarily on global visual cues. As a result, they fail to adequately assess VLMs’ finegrained reasoning capabilities, allowing models to perform well without genuine spatial understanding.

To address these limitations, we introduce TerraScopeBench, a benchmark comprising 3,837 carefully curated

Model Size TerraScope-Bench Landsat30AU DisasterM3 CA AQ CR BRD DM BCE Avg. APR NUM SRI Avg. BDC DRE Avg.

General VLMs GPT-4o† [34] - 27.6 25.4 54.3 75.3 22.5 27.1 38.7 - - - - 24.2 21.4 22.8 LLaVA-OV [18] 7B 28.0 21.2 56.6 75.9 19.4 23.7 37.5 39.4 46.6 85.1 57.0 26.4 24.2 25.3 Qwen2.5-VL [1] 7B 25.3 33.5 55.7 67.7 23.3 25.7 38.5 29.8 53.1 92.8 58.6 34.2 29.3 31.8 InternVL3 [59] 8B 22.3 26.3 57.2 67.0 18.6 24.3 36.0 31.4 42.4 90.6 54.8 30.3 24.1 27.2 GLM-4.1V-Think‡ [11] 9B 24.8 57.1 55.2 58.4 23.3 29.5 41.4 45.7 58.6 70.0 58.1 - - Qwen3-VL-Think‡ [1] 8B 29.0 47.8 57.9 67.8 25.6 31.9 43.3 42.8 60.2 92.0 65.0 36.8 28.2 32.5

EO-Specific VLMs GeoChat [15] 7B 24.8 19.5 49.6 69.2 5.4 - 33.7 31.1 41.8 86.2 53.0 - - TeoChat [14] 7B 25.6 17.8 55.8 55.8 8.5 22.6 31.0 30.2 59.6 87.1 59.0 22.5 23.3 22.9 LHRS-bot [33] 7B 13.7 24.3 54.0 28.4 12.4 - 26.6 63.5 12.5 82.6 52.9 - - EarthDial [43] 4B 26.3 24.1 54.4 69.2 20.2 23.6 36.3 23.5 43.6 51.2 39.4 30.2 20.8 25.5 EarthMind [42] 4B 26.1 42.2 52.2 73.3 38.1 20.8 42.1 - - - - - - -

Fine-tuned VLMs InternVL3 [59] 8B 67.1 63.2 60.0 67.8 40.0 31.0 54.9 55.3 56.6 90.8 67.6 42.2 30.1 36.1 GLM-4.1V-Think‡ [11] 9B 67.8 68.1 65.5 70.2 51.1 34.7 59.6 63.4 60.5 80.0 68.0 45.6 32.0 38.8

TerraScope 8B 73.2 70.2 71.8 80.0 65.9 52.1 68.9 69.8 60.8 91.1 73.9 54.1 38.9 46.5

- Table 1. Quantitative performance of Terrascope on TerraScope-Bench (optical only), Landsat30AU and DisasterM3. “Avg” is the average performance of multiple-choice tasks. Bold means the best performance. † denotes proprietary models. ‡ denotes reasoning models. Fine-tuned VLMs are fine-tuned on our proposed Terra-CoT dataset.

Qwen2.5-VL GLM-4,1V GPT-40 TerraScope

| |
|---|

CA AQ

| |
|---|

65 65

| |
|---|

45 45

| |
|---|

25 25

BCE

CR

20

40

65 45 30

55

20 20

35

35

50 50

DM BRD

Figure 5. Grounding IoU performance of different models.

samples from the test sets of existing datasets [5, 10, 50]. As shown in Fig. 4, our benchmark encompasses six task categories: Coverage Percentage Analysis (855), Absolute Area Quantification (855), Distance Measurement (129), Comparative Area Ranking (855), Boundary Relationship Detection (855), and Building Change Estimation (288).

We leverage pixel-level segmentation annotations to automatically generate question-answer pairs. For each sample, we compute spatial properties from segmentation masks, including coverage ratios, absolute areas, interobject distances, and boundary relationships, to derive ground-truth answers. Questions are generated via templates to ensure diverse phrasing, then rephrased by an LLM to create natural variations and plausible distractors for multiple-choice format. Finally, human experts review the dataset to filter samples with erroneous masks. Unlike existing benchmarks that only assess final answer accuracy, TerraScope-Bench evaluates both response correctness and spatial reasoning quality using IoU-based segmentation metrics, verifying whether models attend to the correct regions during their reasoning process.

#### 5. Experiments

Implementation details. Following the two-stage training strategy in Sec. 3.2, we first perform grounding pretraining where the vision encoder, projector, and LLM are frozen, and only the mask decoder is trained (lr=2e-5, batch size=8). In the second stage, we unfreeze the projector and mask decoder for full training and fine-tune the LLM via LoRA (lr=1e-5, batch size=2). The vision encoder is kept frozen during training. All experiments run on NVIDIA H200-141GB GPUs, with additional dataset and hyperparameter details provided in the Appendix.

Benchmarks. Beyond our proposed TerraScope-Bench, we evaluate TerraScope on two representative EO benchmarks in zero-shot settings to demonstrate its generalization capability. LandSat30-AU [32] features 30-meter resolution imagery with challenging reasoning subtasks; we report results on four tasks requiring fine-grained geospatial reasoning: Agro-Phenology Reasoning (APR), Numerosity Estimation (NUM), and Spatial-Relationship Inference (SRI). DisasterM3 [48] is a bi-temporal disaster assessment benchmark with pre- and post-event image pairs covering multi-hazard scenarios across multiple sensors; we evaluate on Damaged Building Counting (DBC) and Damaged Road Area Estimation (DRE).

##### 5.1. Main Results

We present the performance of TerraScope on several EO benchmarks in Tab. 1, where we evaluate 11 VLMs on TerraScope-Bench, including proprietary models and both general and EO-specific models. Additionally, we fine-tune InternVL3 and GLM-4.1V-Think on our Terra-CoT dataset to show its effectiveness. We highlight several key findings: (1) Pixel-grounded reasoning remains challenging. Existing VLMs struggle with fine-grained geospatial reason-

###### Model TerraBen. Landsat. Disaster.

Original 33.8 45.7 23.6 Textual CoT w/o Seg. 58.7 56.5 32.9 Textual CoT with Seg. 60.6 58.9 35.8 Random-Mask CoT 43.2 53.8 32.6 Box CoT 62.8 70.5 43.9

TerraScope 68.9 73.9 46.5

- Table 2. Ablation study on the effect of different CoT strategies for pixel-grounded visual reasoning. “Original” denotes the base TerraScope model after pretraining, upon which we fine-tune with different CoT variations via SFT.

Figure 6. IoU distribution for correct vs. incorrect predictions.

ing, particularly on tasks requiring precise spatial analysis such as area percentage estimation. Both proprietary and open-source models achieve near-random performance, indicating the necessity of pixel-level grounding.

- (2) EO-specific models show limited advantages. Despite training on large-scale EO data, EO-specific VLMs do not significantly outperform general VLMs on TerraScopeBench. We hypothesize that this is because existing EO datasets predominantly feature high-resolution imagery (<5m), limiting models’ ability to handle lower-resolution data prevalent in real-world applications.
- (3) Reasoning models perform better but lack visual grounding. Models with explicit reasoning capabilities show stronger performance, especially on tasks requiring external knowledge like Absolute Area Quantification. However, their reasoning remains purely textual without grounding in pixel-level visual evidence, leading to hallucinations and insufficient fine-grained spatial perception.
- (4) Terra-CoT effectively improves model performance. Fine-tuning general VLMs (e.g., InternVL3, GLM-4.1VThink) on our Terra-CoT dataset leads to substantial performance gains across all tasks, demonstrating the effectiveness of our pixel-grounded reasoning data. However, challenging tasks like distance measurement (DM) and building change estimation (BCE) remain difficult, suggesting that data alone is insufficient and specialized architectural designs for pixel-grounded reasoning are necessary.
- (5) TerraScope achieves strong performance and generalization. Our framework, which grounds reasoning in fine-grained visual perception, achieves the best results on TerraScope-Bench while demonstrating strong generaliza-

Model CA AQ CR BRD DM No Fusion 73.2 70.2 71.8 80.0 65.9 Concat. 74.5 71.6 73.0 81.2 67.4 Text-guided (test only.) 72.3 69.0 66.7 78.8 63.6 Text-guided (train + test) 74.3 70.9 72.7 80.7 68.2

Table 3. Ablation study of multi-modal reasoning.

|First, I identify all crop areas in the image [SEG]. Now I count the pixels: crop occupies 43909 pixels out of 65536 total pixels. Therefore, crop occupies 67% of the entire image area.<br><br>First, I identify all crop areas in the image [SEG]. Now I count the pixels: crop occupies 26000 pixels out of 65536 total pixels. Therefore, crop occupies 40% of the entire image area.<br><br>What fraction of the image is occupied by crop?<br><br>A. 67% B. 35% C. 40% D. 50%<br><br>[Figure 67]<br><br>[Figure 68]<br><br>Opt SAR<br><br>[Figure 69]<br><br>Optical:<br><br>Optical + SAR:<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]|
|---|

Figure 7. Visualizations of multi-modal reasoning.

tion to LandSat30-AU and DisasterM3.

(6) TerraScope provides interpretable reasoning. Beyond answer accuracy, TerraScope-Bench evaluates the reasoning process by measuring segmentation IoU against ground truth. As shown in Fig. 5, TerraScope not only produces correct answers but also generates faithful reasoning traces with accurate spatial grounding, outperforming other grounding-capable models.

##### 5.2. Ablation Studies

We conduct extensive ablation studies to analyze TerraScope’s effectiveness regarding its pixel-grounded visual reasoning mechanism and multi-modal reasoning, in which more details can be seen in Appendix.

Effectiveness of pixel-grounded visual reasoning. To verify that pixel-level grounding benefits reasoning, we compare several variants in Tab. 2. First, we train models with textual chain-of-thought only, where visual tokens are not interleaved into reasoning steps: either freezing the mask decoder (Textual CoT w/o Seg.) or training it with ground truth masks as auxiliary supervision (Textual CoT w/ Seg.). Results show that auxiliary segmentation training implicitly improves reasoning even when visual tokens are absent from the reasoning sequence, demonstrating the benefit of our joint training design.

Second, we examine the importance of mask-guided token selection. Random-Mask CoT, which interleaves randomly selected visual tokens at each reasoning step without mask prediction, performs worse than textual CoT, likely due to irrelevant visual information hindering reasoning. Box CoT uses the minimal bounding rectangles of predicted masks to select visual tokens, rather than using pre-

What is the shortest distance from water to crop? A. 36m B. 21m C. 42m D. 30m

How many buildings were completely destroyed in this disaster? A. 3 B. 0 C. 2 D. 1

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Which is greater in size, water or tree? A. tree B. water

[Figure 78]

[Figure 79]

[Figure 80]

Initially, there are 5 buildings [SEG] in the area. Following the disaster, 2 destroyed building [SEG] is reduced to rubble, leaving the remaining structures partially damaged or intact. I count 2 totally destroyed buildings.

First, I identify water areas [SEG], then I identify crop areas [SEG]. Now I measure the distance: The closest points between these two land cover types are 3 pixels apart. Since each pixel represents 10.0m, the smallest distance is 3 × 10.0 = 30m.

First, I identify water areas [SEG], then I find tree areas [SEG]. Now I compare their sizes: water occupies 14933 pixels, while tree occupies 23033 pixels. Therefore, tree is greater in size.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

A. tree

D. 30m

C.2

Figure 8. Visualization of TerraScope.

cise segmentation masks. This coarser grounding also underperforms TerraScope, especially on TerraScope-Bench and LandSat30-AU where land cover regions have irregular boundaries and shapes. These results confirm that precise pixel-level grounding through segmentation masks is essential for effective visual reasoning in EO.

alone, cloud cover causes severe segmentation errors in the masked regions. By fusing SAR data, which penetrates clouds, TerraScope produces accurate segmentation masks. (2) Adaptive modality selection for reasoning: Our textguided selection mechanism adaptively chooses between optical and SAR based on data quality: it prioritizes optical tokens for cloud-free regions with reliable spectral information while selecting SAR tokens for cloud-covered areas where optical data is corrupted. This dual advantage enables accurate reasoning under challenging conditions.

Beyond final answer accuracy, we analyze the relationship between intermediate segmentation quality (measured by mean IoU with ground truth masks) and answer correctness. As shown in Fig. 6, samples with higher segmentation quality are significantly more likely to produce correct answers. Specifically, correct predictions achieve mean IoU of 0.628, substantially higher than incorrect predictions (0.443). The strong Pearson correlation (r = 0.607, p < 0.001) holds consistently across all task types (r = 0.70-0.80), demonstrating that accurate pixel-level visual grounding is essential for correct geospatial reasoning.

#### 6. Qualitative Results

Fig. 8 presents representative examples that demonstrating TerraScope’s pixel-grounded reasoning capabilities across three challenging tasks: area percentage estimation, distance measurement, and temporal counting VQA. These cases illustrate TerraScope’s dual strengths: (1) Structured reasoning: decomposing complex spatial questions into interpretable sub-steps through textual chain-of-thought, and (2) Accurate visual grounding: generating precise segmentation masks for relevant regions at each reasoning step. By grounding numerical computations in pixel-accurate visual evidence, TerraScope produces interpretable answers with transparent reasoning traces. Additional qualitative results and failure case analysis are provided in the Appendix.

Effectiveness of multi-modal reasoning. Beyond optical imagery, TerraScope can reason across optical and SAR modalities through text-guided modality selection. To validate its effectiveness, we compare several settings in Tab. 3: (1) No fusion: using only optical images; (2) Concat: concatenating optical and SAR features and interleaving the concatenated features into each reasoning step; (3) Textguided (test only): enabling modality selection only during inference; (4) Text-guided (train+test): enabling modality selection during both training and inference.

#### 7. Conclusion

Results show that any form of multi-modal fusion substantially improves over the optical-only baseline. While Concat achieves slightly higher accuracy than text-guided selection, our approach offers a critical advantage: it reduces context length by selecting only the relevant modality rather than processing both, improving efficiency while maintaining competitive performance. Importantly, training with modality selection is essential—enabling it only at test time yields no improvement, demonstrating that the model must learn when and how to leverage different modalities.

In this paper, we presented TerraScope, a unified visionlanguage framework for pixel-grounded geospatial reasoning in earth observation. By generating segmentation masks alongside reasoning traces, TerraScope achieves precise and interpretable spatial analysis, supporting multi-temporal change analysis and adaptive reasoning across optical and SAR modalities. We curated Terra-CoT, a 1M instructiontuning dataset with pixel-accurate masks embedded in reasoning chains, and introduced TerraScope-Bench, the first benchmark for pixel-grounded geospatial reasoning. Extensive experiments validate the effectiveness of our approach across diverse geospatial reasoning tasks.

Fig. 7 illustrates the effectiveness of our multi-modal reasoning through a cloud-contaminated case. The figure reveals two key advantages: (1) Improved segmentation through multi-modal fusion: When using optical imagery

#### Acknowledgements

This work was supported by the European Union Horizon projects ELIAS (No. 101120237) and ELLIOT (No. 101214398), and by the FIS project GUIDANCE (No. FIS2023-03251). Beg¨um Demir is supported by the European Research Council (ERC) through the ERC-2025POC Agent-BigEarth Project under Grant 101292498. This work has been supported by Mountain Maps s.r.l.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 6
- [2] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024. 1
- [3] Xinyan Chen, Renrui Zhang, Dongzhi Jiang, Aojun Zhou, Shilin Yan, Weifeng Lin, and Hongsheng Li. Mint-cot: Enabling interleaved visual tokens in mathematical chain-ofthought reasoning. arXiv preprint arXiv:2506.05331, 2025. 3, 1
- [4] Yuxing Chen, Weijie Wang, Sylvain Lobry, and Camille Kurtz. An llm agent for automatic geospatial data analysis. arXiv preprint arXiv:2410.18792, 2024. 2
- [5] Kai Norman Clasen, Leonard Hackel, Tom Burgert, Gencer Sumbul, Beg¨um Demir, and Volker Markl. reBEN: Refined bigearthnet dataset for remote sensing image analysis. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2025. 5, 6, 10
- [6] Muhammad Sohail Danish, Muhammad Akhtar Munir, Syed Roshaan Ali Shah, Kartik Kuckreja, Fahad Shahbaz Khan, Paolo Fraccaro, Alexandre Lacoste, and Salman Khan. Geobench-vlm: Benchmarking vision-language models for geospatial tasks. arXiv preprint arXiv:2411.19325, 2024. 3
- [7] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879, 2025. 2, 3, 1
- [8] Peilin Feng, Zhutao Lv, Junyan Ye, Xiaolei Wang, Xinjie Huo, Jinhua Yu, Wanghan Xu, Wenlong Zhang, Lei Bai, Conghui He, et al. Earth-agent: Unlocking the full landscape of earth observation with agents. arXiv preprint arXiv:2509.23141, 2025. 2
- [9] Jun Gao, Yongqi Li, Ziqiang Cao, and Wenjie Li. Interleaved-modal chain-of-thought. In CVPR, pages 19520– 19529, 2025. 3, 1
- [10] Ritwik Gupta, Richard Hosfelt, Sandra Sajeev, Nirav Patel, Bryce Goodman, Jigar Doshi, Eric Heim, Howie Choset, and Matthew Gaston. xbd: A dataset for assessing building damage from satellite imagery. arXiv preprint arXiv:1911.09296,

2019. 6

- [11] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Li-

- hang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 6, 9
- [12] Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, Yu Liu, and Xiang Li. Rsgpt: A remote sensing vision language model and benchmark. ISPRS Journal of Photogrammetry and Remote Sensing,, 224:272–286, 2025. 2, 3
- [13] Shiqi Huang, Shuting He, Huaiyuan Qin, and Bihan Wen. Score: Scene context matters in open-vocabulary remote sensing instance segmentation. In ICCV, pages 12559– 12569, 2025. 1
- [14] Jeremy Andrew Irvin, Emily Ruoyu Liu, Joyce Chuyi Chen, Ines Dormoy, Jinyoung Kim, Samar Khanna, Zhuo Zheng, and Stefano Ermon. Teochat: A large vision-language assistant for temporal earth observation data. arXiv preprint arXiv:2410.06234, 2024. 6
- [15] Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. Geochat: Grounded large vision-language model for remote sensing. In CVPR, pages 27831–27840, 2024. 1, 2, 3, 6
- [16] Sandeep Kumar, Ram Swaroop Meena, Seema Sheoran, Chetan Kumar Jangir, Manoj Kumar Jhariya, Arnab Banerjee, and Abhishek Raj. Remote sensing for agriculture and resource management. In Natural resources conservation and advances for sustainability, pages 91–135. Elsevier,

2022. 1

- [17] Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025. 3
- [18] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 6
- [19] Erzhu Li, Junshi Xia, Peijun Du, Cong Lin, and Alim Samat. Integrating multilayer features of convolutional neural networks for remote sensing scene classification. TGRS, 55(10): 5653–5665, 2017. 1
- [20] Kaiyu Li, Ruixun Liu, Xiangyong Cao, Xueru Bai, Feng Zhou, Deyu Meng, and Zhi Wang. Segearth-ov: Towards training-free open-vocabulary segmentation for remote sensing images. In CVPR, pages 10545–10556, 2025. 1
- [21] Qi Li and Xinchao Wang. Sponge tool attack: Stealthy denial-of-efficiency against tool-augmented agentic reasoning. arXiv preprint arXiv:2601.17566, 2026. 1
- [22] Xue Li, Guo Zhang, Hao Cui, Shasha Hou, Shunyao Wang, Xin Li, Yujia Chen, Zhijiang Li, and Li Zhang. Mcanet: A joint semantic segmentation framework of optical and sar images for land use classification. International Journal of Applied Earth Observation and Geoinformation, 106: 102638, 2022. 5
- [23] Xiang Li, Jian Ding, and Mohamed Elhoseiny. Vrsbench: A versatile vision-language benchmark dataset for remote sensing image understanding. arXiv preprint arXiv:2406.12384,

2024. 2, 3

- [24] Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin

- Shang, Kun Wang, et al. Llm-based agents suffer from hallucinations: A survey of taxonomy, methods, and directions. arXiv preprint arXiv:2509.18970, 2025. 1
- [25] Chenyang Liu, Keyan Chen, Haotian Zhang, Zipeng Qi, Zhengxia Zou, and Zhenwei Shi. Change-agent: Towards interactive comprehensive remote sensing change interpretation and analysis. TGRS, 2024. 2
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36:34892–34916, 2023. 2
- [27] Hao Liu, Yongjie Zheng, Yuhan Kang, Mingyang Zhang, Maoguo Gong, and Lorenzo Bruzzone. Balanced diffusionguided fusion for multimodal remote sensing classification. arXiv preprint arXiv:2509.23310, 2025. 1
- [28] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 3
- [29] Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia. Rsvqa: Visual question answering for remote sensing data. TGRS, 58(12):8555–8566, 2020. 1, 2, 3, 5, 10
- [30] Xiaoqiang Lu, Binqiang Wang, Xiangtao Zheng, and Xuelong Li. Exploring models and data for remote sensing image caption generation. TGRS, 56(4):2183–2195, 2017. 1
- [31] Junwei Luo, Zhen Pang, Yongjun Zhang, Tingzhu Wang, Linlin Wang, Bo Dang, Jiangwei Lao, Jian Wang, Jingdong Chen, Yihua Tan, et al. Skysensegpt: A fine-grained instruction tuning dataset and model for remote sensing visionlanguage understanding. arXiv preprint arXiv:2406.10100,

2024. 1, 2

- [32] Sai Ma, Zhuang Li, and John A Taylor. Landsat30-au: A vision-language dataset for australian landsat imagery. arXiv preprint arXiv:2508.03127, 2025. 6
- [33] Dilxat Muhtar, Zhenshi Li, Feng Gu, Xueliang Zhang, and Pengfeng Xiao. Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model. In ECCV, pages 440–457. Springer, 2024. 2, 3, 6
- [34] OpenAI. Gpt-4o, 2024. 2, 6
- [35] Chao Pang, Xingxing Weng, Jiang Wu, Jiayu Li, Yi Liu, Jiaxing Sun, Weijia Li, Shuai Wang, Litong Feng, Gui-Song Xia, et al. Vhm: Versatile and honest vision language model for remote sensing image analysis. In AAAI, pages 6381– 6388, 2025. 2
- [36] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2
- [37] Akashah Shabbir, Muhammad Akhtar Munir, Akshay Dudhane, Muhammad Umer Sheikh, Muhammad Haris Khan, Paolo Fraccaro, Juan Bernabe Moreno, Fahad Shahbaz Khan, and Salman Khan. Thinkgeo: Evaluating toolaugmented agents for remote sensing tasks. arXiv preprint arXiv:2505.23752, 2025. 2
- [38] Akashah Shabbir, Mohammed Zumri, Mohammed Bennamoun, Fahad S Khan, and Salman Khan. Geopixel: Pixel grounding large multimodal model in remote sensing. ICML,

2025. 2

- [39] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 3
- [40] Yan Shu, Hangui Lin, Yexin Liu, Yan Zhang, Gangyan Zeng, Yan Li, Yu Zhou, Ser-Nam Lim, Harry Yang, and Nicu Sebe. When semantics mislead vision: Mitigating large multimodal models hallucinations in scene text spotting and understanding. arXiv preprint arXiv:2506.05551, 2025. 1
- [41] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. In CVPR, pages 26160–26169, 2025. 1
- [42] Yan Shu, Bin Ren, Zhitong Xiong, Danda Pani Paudel, Luc Van Gool, Beg¨um Demir, Nicu Sebe, and Paolo Rota. Earthmind: Leveraging cross-sensor data for advanced earth observation interpretation with a unified multimodal llm. arXiv preprint arXiv:2506.01667, 2025. 6
- [43] Sagar Soni, Akshay Dudhane, Hiyam Debary, Mustansar Fiaz, Muhammad Akhtar Munir, Muhammad Sohail Danish, Paolo Fraccaro, Campbell D Watson, Levente J Klein, Fahad Shahbaz Khan, et al. Earthdial: Turning multi-sensory earth observations to interactive dialogues. arXiv preprint arXiv:2412.15190, 2024. 1, 2, 6
- [44] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 2
- [45] Hao Sun, Siyuan Li, Xiangtao Zheng, and Xiaoqiang Lu. Remote sensing scene classification by gated bidirectional network. TGRS, 58(1):82–96, 2019. 1
- [46] Fengxiang Wang, Hongzhen Wang, Mingshuo Chen, Di Wang, Yulin Wang, Zonghao Guo, Qiang Ma, Long Lan, Wenjing Yang, Jing Zhang, et al. Xlrs-bench: Could your multimodal llms understand extremely large ultrahigh-resolution remote sensing imagery? arXiv preprint arXiv:2503.23771, 2025. 2, 3
- [47] Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. arXiv preprint arXiv:2506.11991, 2025. 1
- [48] Junjue Wang, Weihao Xuan, Heli Qi, Zhihao Liu, Kunyi Liu, Yuhan Wu, Hongruixuan Chen, Jian Song, Junshi Xia, Zhuo Zheng, et al. Disasterm3: A remote sensing vision-language dataset for disaster damage assessment and response. arXiv preprint arXiv:2505.21089, 2025. 1, 3, 6
- [49] Weihao Xuan, Junjue Wang, Heli Qi, Zihang Chen, Zhuo Zheng, Yanfei Zhong, Junshi Xia, and Naoto Yokoya. Dynamicvl: Benchmarking multimodal large language models for dynamic city understanding. arXiv preprint arXiv:2505.21076, 2025. 1
- [50] Zhenghang Yuan, Zhitong Xiong, Lichao Mou, and Xiao Xiang Zhu. Chatearthnet: A global-scale image-text dataset empowering vision-language geo-foundation models. Earth System Science Data Discussions, 2024:1–24, 2024. 5, 6

- [51] Yang Zhan, Zhitong Xiong, and Yuan Yuan. Rsvg: Exploring data and models for visual grounding on remote sensing data. TGRS, 61:1–13, 2023. 1
- [52] Yang Zhan, Zhitong Xiong, and Yuan Yuan. Skyeyegpt: Unifying remote sensing vision-language tasks via instruction tuning with large language model. ISPRS Journal of Photogrammetry and Remote Sensing,, 221:64–77, 2025. 1, 2
- [53] Chenhui Zhang and Sherrie Wang. Good at captioning bad at counting: Benchmarking gpt-4v on earth observation data. In CVPR, pages 7839–7849, 2024. 3
- [54] Wei Zhang, Miaoxin Cai, Tong Zhang, Yin Zhuang, and Xuerui Mao. Earthgpt: A universal multi-modal large language model for multi-sensor image comprehension in remote sensing domain. TGRS, 2024. 1, 2
- [55] Wei Zhang, Miaoxin Cai, Yaqian Ning, Tong Zhang, Yin Zhuang, He Chen, Jun Li, and Xuerui Mao. Earthgptx: Enabling mllms to flexibly and comprehensively understand multi-source remote sensing imagery. arXiv preprint arXiv:2504.12795, 2025. 2
- [56] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025. 2, 3
- [57] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 2, 3
- [58] Yue Zhou, Mengcheng Lan, Xiang Li, Litong Feng, Yiping Ke, Xue Jiang, Qingyun Li, Xue Yang, and Wayne Zhang. Geoground: A unified large vision-language model for remote sensing visual grounding. arXiv preprint arXiv:2411.11904, 2024. 1
- [59] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2, 4, 6

[Figure 87]

## TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation

### Supplementary Material

#### Appendix Overview

- • A: Limitations and Future work.
- • B: Comparison to Concurrent Works.
- • C: Details of TerraScope.
- • D: Details of TerraScope-Bench.
- • E: Details of Training Data.
- • F: Experimental Settings.
- • G: Efficiency Analysis.
- • I: More Ablation Studies.
- • H: Additional Experiment Results.
- • J: Additional Visualizations and Failure Analysis.

#### A. Limitations and Future Work

TerraScope focuses on pixel-level grounding for earth observation data, but it has several limitations. First, like other multimodal large language models, TerraScope may produce hallucinated outputs, e.g., generating plausible but factually incorrect reasoning traces or inaccurate mask predictions that do not correspond to actual ground features [2, 40]. Mitigating such hallucinations through improved training strategies, verification mechanisms, or retrievalaugmented generation is an important direction for future work. Second, the interleaved generation of masks and reasoning traces increases context length during training and inference. We analyze its computational complexity in Sec. G. A potential solution is to compress masked visual tokens to reduce context length while retaining visual grounding capability. Third, although TerraScope supports multi-sensor reasoning, it currently handles only optical (RGB) and SAR data, with limited capability for multispectral and hyper-spectral imagery [27]. Future work will explore integrating these challenging data sources into the reasoning framework. Finally, the current temporal reasoning capability of TerraScope is limited to bi-temporal analysis (i.e., comparing two time points). Many real-world EO applications, such as urban expansion monitoring, deforestation tracking, and climate trend analysis, require reasoning over long temporal sequences [41]. Extending TerraScope to support multi-temporal and continuous time-series reasoning is an important direction for future work.

#### B. Comparison to Concurrent Works

TerraScope belongs to the category of “thinking with images” models. In Sec. 1 and Sec. 2, we described the distinction between our approach and agent-based models. In this section, we provide detailed comparisons with both unified

interleaved reasoning models and LLM-agent-based methods.

Comparison with Unified Interleaved Reasoning Models. Several concurrent works share similar interleaved reasoning mechanisms with TerraScope, including ICoT [9], GRIT [7], VGR [47], and Mint-CoT [3]. However, they differ from TerraScope in two key aspects. First, these models are designed for general vision tasks and have limited transferability to earth observation, as they lack multi-modal reasoning (optical/SAR) and multi-temporal reasoning capabilities essential for EO applications. Second, they employ different mechanisms for interleaved reasoning:

- • ICoT [9] proposes a training-free module that leverages text-image cross-attention maps in LLMs to select relevant tokens. However, this approach is limited to scenarios with salient objects and fails when queries are complex or involve high-level semantic reasoning not directly tied to visible objects.
- • GRIT and VGR [7, 47] use language to model object coordinates (bounding boxes), which is inadequate for representing pixel-level regions in EO data where spatial phenomena often lack clear boundaries.
- • Mint-CoT [3] overcomes bounding-box limitations by selecting relevant image tokens through similarity-based implicit selection. However, this approach may include tokens irrelevant to the current reasoning step. To validate this, we trained Mint-CoT on our Terra-CoT dataset following their official training paradigm, converting our pixel-level masks into their token indices. Experiments (Tab. A) show Mint-CoT underperforms TerraScope on TerraScope-Bench, confirming the importance of explicit mask generation for pixel-grounded reasoning.

Comparison with LLM-Agent-Based Methods. We further compare TerraScope with concurrent agentic approaches, including ThinkGeo and EarthAgent. As shown in Tab. A, these methods significantly underperform TerraScope. We attribute this to two main limitations: (1) Hallucination: the LLM orchestrator may misinterpret tool outputs or introduce reasoning errors during multi-step planning [21, 24]; (2) Weak perception: ThinkGeo relies on box-level grounding, while EarthAgent adopts SAM-based grounding with independently trained modules, limiting cross-module synergy. In contrast, TerraScope’s unified training paradigm enables bidirectional enhancement between reasoning and pixel-level grounding, which agentic pipelines with decoupled components cannot achieve.

Methods TerraBench. Landsat. Interleaved Reasoning Models

Mint-CoT (with SFT) 54.6 62.8 Mint-CoT (with SFT + RL) 55.7 63.2

LLM-Agent-Based Methods

ThinkGeo 28.5 – EarthAgent 37.6 –

###### TerraScope 68.9 73.9

Table A. Comparison of TerraScope with interleaved reasoning models and LLM-agent-based methods on TerraScope-Bench.

#### C. Details of TerraScope

Vision-Language Model. The VLM component of TerraScope is built upon InternVL-3 [59]. In InternVL-3, each image is divided into multiple patches at a pre-defined resolution (448 × 448). Each patch is processed by the vision encoder and encoded into 256 tokens. For instance, an image with 4 patches (plus one global thumbnail) yields (4 + 1) × 256 = 1,280 visual tokens in total. For multitemporal inputs, we do not split images into patches but directly feed independent images into the model. For example, for a multi-temporal sequence with T observations, the total number of visual tokens is T × 256.

Pixel-Grounding Module. TerraScope’s pixel-grounding module is initialized with the pre-trained SAM-2 model [36]. We connect SAM-2 and the LLM via the special token [SEG]. The hidden states of the [SEG] token from the last layer of LLM serve as a spatial prompt and are fed into SAM-2’s decoder, which generates segmentation masks. This design allows the LLM to control mask generation through learned prompt embeddings.

During training, the SAM-2 decoder is fine-tuned to understand the spatial prompts, and gradients are backpropagated through the [SEG] token to the LLM, enabling it to generate better prompts. During inference, if the LLM does not generate a [SEG] token, we interpret this as indicating that no segmentation is needed for the current reasoning step.

Masked Token Selection. To balance effectiveness and efficiency, we set a maximum threshold λ = 128 for the number of visual tokens in vi. If the number of selected tokens exceeds this threshold, we apply spatial uniform sampling to retain λ tokens while preserving spatial coverage. Specifically, we divide the masked region into a ⌈

√

√

λ⌉ grid and select one token from each grid cell, choosing the token closest to the cell center. This ensures representative spatial sampling across the entire masked region rather than biased concentration in any local area.

λ⌉×⌈

Inference Process. TerraScope performs autoregressive generation with pixel-grounded reasoning (Algorithm 1).

Algorithm 1 TerraScope Inference

- 1: Input: Question embeddings q, Visual features v (or vopt, vSAR), Mask decoder fmask, Max tokens λ, Stopping criteria SC
- 2: Output: Generated answer a with reasoning traces
- 3: predicted tokens ← [] ▷ Initialize as empty list

- 4: reasoning step ← 0 ▷ Track reasoning step index

- 5: inputs ← Initialize(q, v) ▷ Initialize inputs for prefilling

▷ Compute modality relevance scores if multi-modal

- 6: if both vopt and vSAR are available then
- 7: βjµ ← L1 Lℓ=1 Softmax v√µqD⊤

jℓ

for µ ∈ {opt, SAR}

- 8: end if
- 9: while SC not met do
- 10: next token, hidden state ← LLM(inputs)

- 11: Append next token to predicted tokens

▷ Check if [SEG] token is generated

- 12: if next token = [SEG] then

- 13: reasoning step ← reasoning step + 1

- 14: i ← reasoning step

▷ Generate segmentation mask

- 15: mi ← fmask(hidden state)

▷ Resize mask to token grid

- 16: mtoki ← ResizeToTokenGrid(mi)

▷ Select tokens with > 50% coverage

- 17: I ← {j | Coverage(mtoki , j) > 0.5}

▷ Apply spatial sampling if exceeds threshold

- 18: if |I| > λ then
- 19: I ← SpatialUniformSample(I, mtoki , λ)
- 20: end if

▷ Extract masked visual features

- 21: if both modalities available then
- 22: for j ∈ I do
- 23: if βjopt > βjSAR then
- 24: vj ← vjopt
- 25: else
- 26: vj ← vjSAR
- 27: end if
- 28: end for
- 29: else
- 30: Extract features from single modality
- 31: end if
- 32: vi ← {vj | j ∈ I}
- 33: Append vi to predicted tokens

- 34: end if
- 35: inputs ← Update(inputs, predicted tokens) ▷ Update KV cache for next generation

- 36: end while
- 37: a ← Tokenizer.decode(predicted tokens)

- 38: return a

The vision encoder processes input images to obtain visual features v (or vopt, vSAR for multi-modal inputs), which are cached for efficiency. At each step, the LLM generates the next token. When a [SEG] token is generated, TerraScope: (1) generates a segmentation mask mi via the mask

BCE

7.5

DM

CA

3.4

[Figure 88]

22.3

855

BRD

22.3

228

AQ CR

22.3

Single lmage (Optical)

Single lmage (SAR)

Temporal (Optical)

22.3

( a ) ( b ) ( c )

Figure A. Data distribution of TerraScope-Bench.

decoder conditioned on the [SEG] token’s hidden states; (2) extracts masked visual features vi by selecting tokens with > 50% coverage and applying spatial uniform sampling if the count exceeds λ = 128; (3) for multi-modal inputs, adaptively selects between optical and SAR based on text-relevance scores βjµ. The selected features vi are then injected into the generation sequence, and the LLM continues reasoning conditioned on both textual and visual contexts through KV cache updates.

#### D. Details of TerraScope-Bench

##### D.1. Overview

We present a more detailed analysis of TerraScope-Bench in Fig. A. Subfigures (a–c) illustrate the distribution of task categories, image source (multi-sensor and multi-temporal) and the visualization of word clouds of question, showing that TerraScope-Bench covers a wide variety of object types and semantics, enabling comprehensive evaluation across pixel-level grounded visual reasoning tasks.

##### D.2. Data Annotations for TerraScope-Bench

TerraScope-Bench consists of six task types requiring pixelgrounded reasoning. We construct the benchmark through a three-stage pipeline: (1) heuristic-based answer generation from pixel-level annotations, (2) GPT-4o-based question rephrasing and distractor generation, and (3) expert validation and quality control.

Stage 1: Heuristic-Based Answer Generation. We leverage existing pixel-level segmentation annotations to generate ground-truth answers using deterministic rules. The benchmark includes three data sources: ChatEarthNet and BigEarthNet for land cover analysis and xBD for building damage assessment. For each image, we process the segmentation mask to extract spatial information required for different task types. The specific rules for each task are:

• Absolute Area Calculation: For a given land cover class c, we count all pixels with label c in the segmentation mask. The area is computed as Ac = Nc × r2, where Nc is the pixel count and r is the spatial resolution (10m for Sentinel-2). Questions specify a single target class (e.g., “What is the area of forest?”), and the ground-truth

answer is the computed area in square meters or hectares. We only include classes with Ac > 0 to avoid trivial questions.

- • Coverage Percentage: For a target land cover class c, we

compute the percentage as Pc = N

c

Ntotal × 100%, where Nc is the pixel count of class c and Ntotal is the total number of valid pixels in the image (excluding background/void). Questions ask for the coverage of a specific class (e.g., “What percentage of the image is cropland?”). We require Pc ≥ 5% to ensure the class is visually significant and avoid questions about negligible regions.

- • Comparative Area Ranking: Given a set of land cover classes present in the image, we rank them by area in de-

scending order: c1,c2,...,cn where Ac

1 ≥ Ac

2 ≥ ··· ≥ Ac

n

. Questions ask for the largest class (e.g., “Which land cover type has the largest area?”) or relative ranking (e.g., “Is forest larger than grassland?”). We only include classes with Pc ≥ 5% in the ranking to ensure clear visual distinction. For binary questions, we require |Ac

i − Ac

j| > 0.1 × max(Ac

i

,Ac

j

) to avoid ambiguous comparisons between similar-sized regions.

- • Distance Measurement: To measure the minimum distance between two land cover classes ci and cj, we: (1) generate binary masks Mi and Mj for each class; (2) apply Euclidean distance transform (distance transform edt) to Mi to compute the distance from each pixel to the nearest ci pixel; (3) extract the minimum value within Mj, which gives the minimum distance d(ci,cj) in pixels; (4) convert to meters using spatial resolution (dmeters = d×r). Questions specify two distinct classes (e.g., “What is the distance between forest and water?”). We require both classes to form spatially connected components (removing isolated pixels via mor-

phological opening) and enforce d(ci,cj) > 10 pixels to avoid trivial adjacent cases. For classes with multiple disconnected regions, we report the minimum distance across all region pairs.

- • Boundary Relationship Detection: To determine if two land cover classes ci and cj are adjacent, we: (1) generate binary masks Mi and Mj; (2) apply morphological dilation (binary dilation) with a 3×3 structuring element to Mi, creating Midilated; (3) check if Midilated∩Mj ̸=

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Figure B. Examples of human expert verification of mask accuracy. The first row shows original images, the second row shows original masks, and the third row shows masks modified by human annotators.

∅. If the intersection is non-empty, the classes are considered adjacent (sharing a boundary). Questions ask binary yes/no queries (e.g., “Does forest border water?”). We filter out class pairs where either region is too small (Pc < 3%) or fragmented (more than 5 disconnected components) to ensure clear, unambiguous boundaries. For multi-component classes, adjacency is determined if any component pair satisfies the criterion.

• Building Change Estimation: Using the xBD dataset, we compare pre-disaster and post-disaster satellite imagery to identify destroyed buildings. The annotation process: (1) parse building footprint polygons from JSON files in WKT format using Shapely (wkt.loads); (2) filter polygons based on damage classification labels (only retain buildings labeled as “destroyed”); (3) rasterize polygon geometries to binary masks using OpenCV (cv2.fillPoly) at the image resolution; (4) count destroyed buildings Ndestroyed and total buildings Ntotal to compute damage rate R = NNdestroyed

×100%. Questions ask about building counts (e.g., “How many buildings were destroyed?”) or damage percentages (e.g., “What percentage of buildings were destroyed?”). We only include samples with Ntotal ≥ 10 buildings and Ndestroyed ≥ 3 to ensure statistically meaningful damage assessment. Poly-

total

gon parsing handles potential coordinate precision issues and self-intersecting geometries using Shapely’s built-in validation.

The implementation uses Python libraries including NumPy for array operations, SciPy for distance transforms (distance transform edt, binary dilation), Shapely for geometry processing (wkt.loads, Polygon), and OpenCV for mask rendering.

Stage 2: GPT-4o-Based Question Refinement. To ensure linguistic diversity and difficulty, we use GPT-4o to: (1) rephrase template questions into natural language variations, and (2) generate plausible distractors for multiplechoice format. For comparative area ranking and boundary relationship detection, we generate 2 options (binary choice). For other tasks (absolute area, coverage percentage, distance measurement, building change estimation), we generate 4 options. The rephrasing prompt is designed to maintain semantic equivalence while varying question structure and wording.

###### GPT-4o Rephrasing Prompt:

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

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

What portion of the image is occupied by developed?

What portion of the image is occupied by crop?

Which land cover class is larger, developed or crop?

Which covers a greater area, water or wetland?

A. 6% B. 13% C. 7% D. 22%

A. 62% B. 44% C. 48% D. 38%

- A. water
- B. wetland

- A. crop
- B. developed

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

What is the total area of tree cover measured in square meters?

What is the total area of water area measured in square meters?

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Is the crop area adjacent to grass?

Is the developed area adjacent to wetland?

- A. 943300 m2
- B. 1886600 m2
- C. 3773200 m2
- D. 188660 m2

- A. 184750 m2
- B. 36950 m2
- C. 739000 m2
- D. 369500 m2

A. Yes B. No

A. Yes B. No

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

What is the shortest distance from crop to water?

What is the closest distance between water to shrub?

What percenatge of buildings were destroyed in this disaster? A. 33.3% B. 58.3% C. 86.7% D. 43.3%

What percenatge of buildings were destroyed in this disaster? A. 66.7% B. 58.3% C. 13.3% D. 43.3%

A. 161m B. 276m C. 230m D. 322m

A. 42m B. 21m C. 36m D. 30m

Figure C. More examples of TerraScope-Bench, including the questions, answers and the masks involved in the CoT.

Question Rephrasing Prompt

Given the following question template and answer: Question: {original question} Answer: {ground truth answer}

Task: Rephrase the question to make it more natural and diverse while preserving the original meaning. Generate {num options} plausible but incorrect answer choices (distractors) that are numerically/semantically close to the ground truth but clearly distinguishable. Ensure distractors are realistic and challenging.

Output format: { "question": "rephrased question", "options": ["option A", "option B", "option C", "option D"], "answer": "correct option letter" }

Stage 3: Expert Validation. We recruit 4 domain experts in geoscience and disaster assessment to ensure annotation quality. Each expert is assigned to validate one or two specific task types. The validation process includes:

- 1. Mask accuracy check: Verify that segmentation masks correctly represent land cover boundaries or building footprints, as shown in Fig. B.
- 2. Answer correctness: Validate that ground-truth answers match the mask through manual calculation.
- 3. Distractor quality: Ensure distractors are plausible but clearly incorrect.
- 4. Question clarity: Check that questions are unambiguous and answerable from the image.

After initial annotation, experts cross-validate each other’s work and score sample quality on a 3-point scale (low/medium/high). Only samples with consensus (all experts agree on high quality) are retained. Samples with erroneous masks, ambiguous questions, or invalid distractors are filtered out. The final benchmark contains 3,837 expertverified samples across six task types. Sample visualizations are shown in Fig. C.

#### E. Details of Training Data

##### E.1. Pretraining Data

For Stage 1 grounded pretraining, we synthesize 2M referring expression segmentation (RES) samples from two sources: 1.5M from BigEarthNet and 0.5M from ChatEarthNet. Both datasets provide semantic segmentation annotations with pixel-level class labels. To convert them into RES format, we randomly select one land cover category from each image and construct the instruction as “Please segment the [class name]”, where [class name] is replaced with the specific land cover type (e.g., “forest”, “cropland”, “water”). The corresponding groundtruth masks are extracted from the original semantic labels and encoded in Run-Length Encoding (RLE) format for efficient storage. This synthetic RES data enables the mask decoder to learn foundational pixel-level grounding capabilities before instruction tuning.

##### E.2. Terra-CoT Dataset Construction

Cap-CoT Curation. We construct the Cap-CoT (Caption with Chain-of-Thought) dataset from four sources: ChatEarthNet, BigEarthNet, xBD, and TEOChat (regionbased change question answering). We employ an RoIbased summarization strategy where class information or original metadata, along with mask-overlaid images, are fed into Qwen3-VL-235B to generate captions with reasoning chains. The generation prompt instructs the model to produce chain-of-thought reasoning that explicitly refers to the provided segmentation semantic labels. This ensures that generated captions are grounded in precise spatial information rather than vague descriptions.

Caption Generation Prompt for Cap-CoT

System: You are an expert in remote sensing image analysis. Your task is to generate a detailed caption with stepby-step reasoning for the given satellite image.

###### Input:

- • Satellite image with mask overlay
- • Segmentation labels: {label 1, label 2, ..., label n}

- • Metadata: [resolution, sensor type, location] Instructions:

- 1. Analyze the spatial distribution of each land cover type shown in the segmentation masks
- 2. Generate a chain-of-thought reasoning process that:

- • Explicitly mentions each segmented region
- • Describes spatial relationships between different land cover types
- • Estimates approximate coverage or area for major land cover classes
- • Notes any significant patterns or features

- 3. Provide a final comprehensive caption summarizing the image

Output Format: <think> First, I observe [description of dominant land cover]. The segmentation shows [specific area/pattern]. [SEG for region 1] covers approximately [percentage/area]. Next, I notice [another land cover type]. [SEG for region 2] appears in [location/pattern]. The spatial relationship between these regions shows [description]. Additionally, [other observations]... </think> <caption> This satellite image shows [comprehensive summary including all major land cover types, their spatial distribution, and key characteristics]. </caption>

VQA-CoT Curation. Based on the 250K Cap-CoT dataset, we first train TerraScope-Cap, a caption-specialized variant of TerraScope. We then use TerraScope-Cap to annotate images from ChatEarthNet, BigEarthNet, RSVQA-LR, and xBD training sets, generating captions and predicted masks. For ground-truth mask refinement, we compute the intersection between predicted masks and available ground-truth annotations when available, ensuring higher quality.

Using these captions as context, we synthesize L1-level VQA samples covering six task types. We design predefined templates for each task type to ensure consistency and coverage:

L1-Level VQA Templates

- Task 1: Object Existence Template: “Is there any [class] in the image?” Example: “Is there any forest in the image?” Answer: “Yes” or “No”
- Task 2: Object Counting Template: “How many [object] are there in the image?” Example: “How many buildings are there in the image?” Answer: “[number] [object]” (e.g., “15 buildings”)
- Task 3: Localization Template: “Where is the [class] located in the image?” Example: “Where is the water body located in the image?” Answer: “[cardinal direction/relative position]” (e.g., “in the northeastern part”, “along the southern edge”)
- Task 4: Area Quantification Template 1: “What is the area of [class]?”

Template 2: “What percentage of the image is covered by [class]?”

- Example 1: “What is the area of cropland?”
- Example 2: “What percentage of the image is covered by forest?”

- Answer 1: “[number] square meters” or “[number] hectares”
- Answer 2: “[percentage]%”

- Task 5: Boundary Detection Template: “Does [class1] border [class2]?” Example: “Does forest border water?” Answer: “Yes” or “No”
- Task 6: Distance Measurement Template: “What is the distance between [class1] and [class2]?” Example: “What is the distance between cropland and water?” Answer: “[number] meters” Generation Strategy:

- • For each image, randomly select 2-4 task types
- • Ensure at least one task per image requires pixel-level reasoning
- • Classes are sampled from available segmentation labels
- • Answers are computed deterministically from groundtruth or refined masks

Building upon L1-level VQA, we use GPT-4o to synthesize more complex reasoning problems that require multistep spatial analysis. The synthesis prompt encourages GPT-4o to create questions involving comparative reasoning, spatial relationships, and compositional understanding.

Fig. D visualizes the composition and distribution of the Terra-CoT dataset from three perspectives. First, we show the geographic distribution of source images, demonstrating global coverage across diverse geographical regions and climatic zones. Second, we present the data source breakdown for Cap-CoT and VQA-CoT subsets, illustrating how different source datasets contribute to caption generation and question-answering components. Third, we provide sample quantity statistics across the three dataset tiers: CapCoT (caption with chain-of-thought), L1-level VQA (simple spatial queries), and L2-level VQA (complex multi-step reasoning).

L2-Level VQA Synthesis Prompt

System: You are an expert in designing complex spatial reasoning questions for satellite imagery analysis. Input:

- • Image caption with spatial information
- • L1-level QA pairs (simple questions and answers)
- • Available land cover classes: {class 1, class 2, ..., class n}

Task: Generate 2-3 complex reasoning questions from two categories:

- Category 1: Spatial Reasoning Questions These questions focus on geometric and spatial properties requiring pixel-level analysis, such as area comparison, distance measurement, boundary relationships, coverage quantification, and spatial distribution patterns.
- Category 2: Semantic Reasoning Questions These questions focus on understanding land cover semantics, ecological patterns, temporal changes, functional relationships, and overall landscape composition. Requirements:

- 1. Generate at least one question from each category
- 2. Questions must require multi-step reasoning
- 3. Answers should be deterministic and verifiable
- 4. Spatial reasoning questions must involve precise geometric analysis
- 5. Semantic reasoning questions must demonstrate understanding of land cover semantics

Output Format: For each question, provide:

- • Question text
- • Category: [Spatial Reasoning] or [Semantic Reasoning]
- • Ground-truth answer
- • Reasoning steps required (brief description)
- • Classes involved

###### Hyperparameter Value

Overall batch size 32 Learning rate 4e-5 LR Scheduler Cosine decay DeepSpeed ZeRO Stage ZeRO-2 Optimizer Adam Warmup ratio 0.3 Epoch 1 Weight decay 0 Precision bf16

Table B. Hyperparameters of TerraScope.

#### F. Experimental Settings

##### F.1. Training Details of TerraScope

We provide complete training hyperparameters for TerraScope in Tab. B. The model is fine-tuned for one epoch on Terra-CoT with a total batch size of 32, using the Adam optimizer with cosine learning rate decay. We employ DeepSpeed ZeRO-2 for memory-efficient training and use bf16 mixed precision to accelerate computation.

[Figure 153]

450K

450K

420K

Cap-CoT VQA-CoT

450K

300K

350K

350K

260K

250K

250K

250K

180K

150K

150K

57.2K

50K

50K

50K

20K 11.4K 1K 0K 0K

ChatEarthNet BigEarthNet xBD TeoChat RSVQA

Cap-CoT

L1-Level VQA L2-Level VQA

(a) (b) (c)

Figure D. Data distributions of Terra-CoT.

System Prompt for Reasoning Mode. To facilitate pixelgrounded reasoning, we modify the system prompt of InternVL-3 to encourage structured thinking with segmentation. The modified prompt instructs the model to: (1) generate reasoning steps within <think> tags, where it can insert [SEG] tokens to trigger mask generation; (2) provide the final answer within <answer> tags. The complete system prompt is:

System Prompt for TerraScope

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in their mind, generating segmentation masks when needed using [SEG] tokens, and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process with [SEG] for segmentation </think><answer> answer here </answer>.

###### Example Prompt with Pixel-Grounded Reasoning:

###### Example Output

User: What percentage of the image is covered by forest?

Assistant: <think>

To answer this question, I need to identify and measure the forest area in the image. Let me first segment the forest regions. [SEG]

[Mask generated for forest regions]

Now I can see the forest areas clearly marked. Let me calculate the coverage:

- The segmented forest region covers approximately

3,450 pixels

- - The total image area is about 10,000 pixels
- - Coverage percentage = (3,450 / 10,000) × 100% = 34.5%

Therefore, forest covers approximately 35% of the image. </think> <answer>35%</answer>

The full prompt template including additional few-shot examples is provided in our code repository.

##### F.2. Evaluation Details on TerraScope-Bench

Baseline Models. We evaluate TerraScope-Bench on three categories of vision-language models:

- • General-purpose VLMs: We include proprietary models (GPT-4o) and open-source models (LLaVA-OV), Qwen2.5-VL, InternVL-3. For GPT-4o, we utilize its multi-image API to handle multi-temporal inputs for the Building Change Estimation task.
- • Reasoning-enhanced VLMs: We evaluate two models with explicit reasoning capabilities: GLM-4.1V-Think and Qwen3-VL-Think, which generate intermediate reasoning steps before final answers.
- • EO-specific VLMs: We include domain-adapted models fine-tuned on earth observation data: GeoChat, TEOChat, LHRS-Bot, EarthDial, and EarthMind.

Evaluation Protocol. All tasks in TerraScope-Bench are formatted as multiple-choice questions with 2 or 4 options (A, B, C, D). We use a unified prompt template across all evaluated models, requesting them to select the correct option. To ensure reliable option extraction, we incorporate option prediction guidance in the prompt: ”Please respond with only the option letter (A, B, C, or D) corresponding to your answer.” Since some models have limited instruction-following ability and may generate verbose explanations instead of direct option letters, we implement post-processing using regex patterns (e.g., r’\b[A-D]\b’) to extract the predicted option from model outputs. If multiple option letters appear, we select

the first occurrence; if no valid option is found, the prediction is marked as incorrect.

Multi-temporal Handling. For the Building Change Estimation task, which requires comparing pre-disaster and post-disaster imagery:

- • Proprietary models (GPT-4o): Use multi-image input API
- • Open-source models: Concatenate images horizontally or process as separate frames
- • Models without multi-image support: Provide both images sequentially in the conversation

Evaluation Metrics. We compute accuracy by exact matching between predicted option letters and ground-truth answers. For each task type, we report:

- • Per-task accuracy: Percentage of correct predictions for each task
- • Overall accuracy: Macro-average across all six tasks Implementation Details.
- • For open-source models, we use their official repositories and recommended inference settings
- • For proprietary APIs (GPT-4o), we set temperature=0 for deterministic outputs
- • All evaluations use greedy decoding (top-p=1.0, temperature=0)

To ensure fair comparison, we fine-tune baseline models on our Terra-CoT dataset with appropriate adaptations:

- • InternVL-3: We remove all special tokens (<think>, </think>, [SEG]) from the training data and perform standard supervised fine-tuning using the official training scripts. The model is trained to directly predict answers without explicit reasoning traces or segmentation masks.
- • GLM-4.1V-Think: We preserve the thinking mode structure (<think>, </think>) but remove the [SEG] token, as this model does not support pixel-level grounding. We use the official training pipeline combining SFT (Supervised Fine-Tuning) and RLVR (Reinforcement Learning with Verifiable Rewards) as described in [11].

This design allows us to assess whether baseline models can benefit from our training data while maintaining their original architectures. The complete evaluation code, prompts, and output parsing scripts are available in our repository.

#### G. Efficiency Analysis

We analyze TerraScope’s computational efficiency from multiple perspectives, including inference time, memory consumption, parameter count, and the impact of pixelgrounded reasoning on computational cost.

##### G.1. Model Complexity

Tab. C compares TerraScope with mainstream baseline models in terms of model size.

TerraScope integrates the SAM-2 image encoder (224.4M parameters) and mask decoder (3.9M parameters)

Model Total Params Additional Modules

GPT-4o - Qwen2.5-VL-7B 7.6B InternVL-3-8B 8.1B GLM-4.1V-9B 9.4B LLaVA-OV-7B 7.2B -

###### TerraScope-8B 8.3B SAM-2 (0.228B)

- – Base InternVL-3 8.1B -
- – SAM-2 image encoder - 0.224B
- – SAM-2 mask decoder - 0.004B

Table C. Model complexity comparison. TerraScope adds a lightweight pixel-level grounding module on top of InternVL-3.

to enable pixel-level grounding. These two modules together introduce only about 0.228B additional parameters, increasing the overall model size from 8.1B (base InternVL3) to 8.3B. This corresponds to a parameter overhead of merely ∼2.8%.

Crucially, the added segmentation components are extremely lightweight compared to the backbone large multimodal model: the extra 0.228B parameters account for only a small fraction of the total parameter budget, while the vast majority of parameters still reside in the LLM. In other words, TerraScope incurs only a minimal parameter increase yet gains the substantial benefit of being able to produce verifiable, pixel-level segmentation masks at each reasoning step.

##### G.2. Inference Time Analysis

We measure inference time on a single NVIDIA A100 80GB GPU with batch size 1. Tab. D reports the average time per sample on TerraScope-Bench.

Model Avg. Time (s)

InternVL-3-8B 0.85 Qwen2.5-VL-7B 0.92

TerraScope-8B 2.48 GLM-4.1V-9B 2.60

Table D. Average inference time per sample (seconds).

TerraScope achieves faster inference than GLM-4.1V-9B (2.4s vs 2.6s) despite generating additional segmentation masks. We identify two key efficiency advantages: First, TerraScope performs deterministic reasoning with structured output (<think> and <answer> tags), while GLM4.1V tends to generate overly verbose reasoning traces with significantly more tokens. Second, our interleaved mask injection is highly efficient—masked visual features are directly inserted into the KV cache without re-encoding through the vision encoder, avoiding redundant visual pro-

cessing. InternVL-3 remains the fastest (0.85s) as it generates answers directly without reasoning, but lacks both reasoning transparency and pixel-level grounding capabilities that TerraScope provides.

##### G.3. Memory Consumption

We profile GPU memory usage during inference on a single NVIDIA A100 80GB GPU. Tab. E shows peak memory consumption with different numbers of generated masks.

Model 1 Mask 2 Masks 3+ Masks

InternVL-3-8B 18.2 18.3 18.2 Qwen2.5-VL-7B 16.8 17.0 17.0

TerraScope-8B 22.4 23.1 24.2

Table E. GPU memory consumption (GB) on NVIDIA A100.

TerraScope requires approximately 22% more memory than InternVL-3 (22.4GB vs 18.2GB for single-mask cases), primarily due to the SAM-2 decoder weights (3.9GB). Memory consumption increases with the number of generated masks, as each mask adds approximately 0.7GB for storing mask features and intermediate activations. In contrast, baseline models (InternVL-3, Qwen2.5VL) maintain constant memory usage regardless of output complexity, as they do not generate pixel-level grounding. The memory overhead is acceptable given TerraScope’s additional capability of producing verifiable segmentation masks.

#### H. Additional Experimental Results

Beyond the geospatial reasoning tasks reported in Sec. 5, we evaluate TerraScope on additional benchmarks to demonstrate its generalization ability across diverse earth observation tasks.

##### H.1. Comprehensive Results on Landsat30-AU

Tab. F presents complete results on all eight task types in Landsat30-AU. The benchmark includes Agro-Phenology Reasoning (APR) for agricultural growth stages, CloudOcclusion Assessment (COA) for detecting cloud coverage, Dominant Land-Cover (DLC) for identifying main land types, Fine-Object Detectability (FOD) for detecting small objects, Macro-Object Presence (MOP) for large-scale objects, Object Counting (NUM), Spatial Relationship (SRI) for spatial layout reasoning, and Urban Scale Recognition (USR) for classifying settlement scale. TerraScope achieves competitive performance across all task types, with particularly strong results on fine-grained visual tasks requiring precise spatial understanding, such as Cloud-Occlusion Assessment (COA) and Fine-Object Detectability (FOD).

This demonstrates that pixel-grounded reasoning capabilities transfer effectively to general earth observation understanding tasks.

##### H.2. Results on RSVQA and Scene Classification

Tab. G reports performance on RSVQA-LR [29] and BigEarthNet scene classification [5]. On RSVQA-LR, TerraScope performs slightly below EarthDial. We attribute this to the difference in training data scale—LHRS-Bot and EarthDial were trained on significantly larger VQA datasets, which benefits general question-answering tasks. On BigEarthNet scene classification, TerraScope achieves competitive accuracy comparable to EarthDial, demonstrating effective transfer learning despite being primarily designed for pixel-grounded reasoning.

##### H.3. Complete Results on DisasterM3

We report comprehensive results on DisasterM3, which includes both optical-optical and optical-SAR multi-modal evaluation. In the main paper (Sec. 5), we reported only optical-optical results as most baseline models do not support SAR imagery. Tab. H presents results on both modality configurations. TerraScope is the only model capable of handling optical-SAR multi-modal inputs through adaptive modality selection. On optical-optical pairs, TerraScope achieves competitive performance with EO-specific baselines. On optical-SAR pairs, TerraScope demonstrates its unique capability to leverage complementary information from heterogeneous modalities for damage assessment.

#### I. More Ablation Studies

Effectiveness of Two-Stage Training. TerraScope employs a two-stage training strategy: Stage 1 performs grounded pretraining on 2M referring expression segmentation pairs to train the mask decoder, and Stage 2 applies instruction tuning on Terra-CoT to jointly optimize the projector, LLM, and mask decoder. Tab. J compares models with and without Stage 1 pretraining on three benchmarks. The results demonstrate that grounded pretraining establishes foundational pixel-level grounding capability, which substantially improves performance on pixel-grounded reasoning tasks and also benefits general EO understanding and disaster assessment tasks.

Effectiveness of Terra-CoT data composition. Our TerraCoT dataset is synthesized using a hierarchical data synthesis strategy combining three data types: L1-level VQA, L2-Level VQA, and captioning. To validate the effectiveness of this composition, we train TerraScope with different data mixtures in Tab. I. First, training with Terra-Cap (captioning only) provides limited instruction-following capability, as the model struggles with both perception and reasoning tasks. Second, adding L1-level VQA establishes

Model Size APR COA DLC FOD MOP NUM SRI USR Overall

EarthDial 4B 23.49 10.34 75.27 99.00 61.16 43.62 51.24 15.52 48.29 RS-LLaVA 7B 68.57 80.88 71.24 87.00 63.09 49.85 26.17 10.34 57.24 MiMo 7B 40.00 45.77 92.47 93.33 84.30 61.42 94.21 88.97 75.55 GLM-4.1V 9B 45.71 36.36 72.85 62.67 67.49 58.63 69.97 88.28 62.87 Qwen2.5-V 7B 29.84 89.66 94.09 71.67 76.03 53.12 92.84 82.07 74.28 LLaVA-OV 8B 39.37 79.00 83.06 59.00 72.45 46.59 85.12 10.34 60.96

TerraScope 8B 69.84 98.12 83.06 87.67 61.98 60.82 91.12 85.52 79.36

Table F. Performance on the VQA task on Landsat30-AU. Bold indicates the best score.

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

Which land cover class is greater, water or developed?? A. water B. developed

Is the developed area adjacent to water? A. Yes B. No

What proportion of the image is occupied by developed areas? A. 19% B. 43% C. 30% D. 11%

[Figure 165]

[Figure 166]

[Figure 167]

First, I identify water areas [SEG] and developed areas [SEG]. Now I compare their sizes: water occupies 7333 pixels, while developed area occupies 53332 pixels. Therefore, developed area is greater.

First, I identify developed areas [SEG] and water areas [SEG]. Now I examine the boundaries: I can see that developed and water share a common boundary, meaning they are adjacent.

First, I identify the developed areas [SEG] in the image, which occupies about 12452 pixels, so the answer is about 19%.

A

A

B

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

What is the closest distance from crop to water? A. 180m B. 105m C. 150m D. 210m

What is the total area of tree cover measured in square meters? A. 4205100m2 B. 1962379m2 C. 3644420m2 D. 2803400m2

What percentage of the entire image is occupied by undamaged roads?? A. 0.57% B. 2.85% C. 6.88% D. 7.76% E. 6.01%

[Figure 179]

[Figure 180]

[Figure 181]

First, I identify crop areas [SEG] and water areas [SEG]. Now I measure the distance: The closest points between these two land cover types are 15 pixels apart. Since each pixel represents 10.0m, the closest distance is 15 × 10.0 = 150m.

Examining the pre-disaster image, I observe the complete road network [SEG] covering the area. In the post-disaster image, I identify intact road segments [SEG] that remain undamaged, accounting for 6.01% of the total image area.

First, I identify all tree areas in the image [SEG]. Now I calculate the area: tree occupies 28034 pixels, and each pixel represents 10.0m × 10.0m = 100.0m². Therefore, the total area is 28034 × 100.0 = 2803400m².

D

C

E

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Which cropping season best matches the vegetation state in the image?

- A. winter_crop: crops that are planted in autumn and grow through the cool months
- B. summer_crop: crops that are planted in spring and grow through the warm months

What is the spatial relationship between the river and the surrounding vegetation? A. The river is isolated on bare land B. The river is surrounded by buildings C. The river is next to a road corridor D. The river runs through the vegetation

Enumerate the total intact buildings that survived the disaster event? A. 35 B. 30 C. 25 D. 25 E. 20

[Figure 191]

Upon analyzing the satellite image captured in mid-winter, I identify several key agricultural land cover types including Arable land [SEG]. These areas exhibit vegetation patterns characteristic of winter cropping systems, where crops are planted in autumn and actively grow during the cooler months. The green, active vegetation visible in these fields during the winter period indicates winter crops rather than dormant summer cropland..

[Figure 192]

Initially, there are 34 buildings [SEG] visible before the disaster. After the event, I identify 25 intact buildings [SEG] with no signs of damage. I count 25 intact buildings.

[Figure 193]

Let me analyze this image step by step. I can observe Inland waters [SEG] in the image. I can observe Moors, heathland and sclerophyllous vegetation[SEG] in the image.

C

D

A

Figure E. More qualitative results of TerraScope.

Model RSVQA-LR BigEarthNet GeoChat 90.7 20.4 LHRS-Bot 89.2 EarthDial 92.7 68.8 TerraScope 91.4 69.2

Table G. Results on RSVQA and scene classification.

foundational pixel-grounded visual understanding, significantly improving performance on tasks requiring accurate segmentation. However, this perception-focused training still lacks complex reasoning capabilities, resulting in poor

Optical-SAR BDC DRE Avg

Model

LLaVA-OV 22.2 19.4 20.8 TEOChat 18.4 9.4 13.9 InternVL3-8B 20.7 18.4 19.6 EarthDial 19.5 10.2 14.9

###### TerraScope 50.4 32.6 41.5

Table H. Optical-SAR results on DisasterM3 benchmark.

performance on challenging tasks like those in LandSat30AU that require multi-step spatial reasoning. Third, incor-

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

What proportion of the image is occupied by developed ? A. 47% B. 26% C. 34% D. 15%

How many distinct road corridors can be seen in the upper right corner of the image? A. One B. Two C. None D. Three

Which thin linear structure appears most prominently in this image? A. Overhead powerline B. Pipeline C. Fence wire D. None of the above

[Figure 200]

First, I identify all crop areas [SEG] in the image, which occupies about 22282 pixels, so the answer is about 34%.

[Figure 201]

I first identigy the road in the image [SEG], I find there are more than two roads in the image.

[Figure 202]

Let me analyze this image step by step. I can observe Pastures [SEG] in the image.

C

D

C

Figure F. Failure cases of TerraScope.

Data TerraBench. Landsat. Disaster. Cap-CoT 42.8 50.1 26.9

Cap-CoT + L1-VQA 66.7 61.0 46.2 Cap-CoT + L1-VQA + L2-VQA 68.9 73.9 46.5

Table I. Ablations of Terra-CoT.

Training Strategy TerraScope-Bench Landsat30-AU DisasterM3

w/o Grounded Pretrain 65.4 71.8 43.0 w/ Grounded Pretrain 68.9 73.9 46.5

Table J. Ablation study on grounded pretraining.

Multi-modal Masked Feature TerraScope-Bench Efficiency Encoding Interleaving Mean IoU (%) Accuracy (%)

Optical only Optical only 53.4 65.0 High Optical only Concat Opt+SAR 53.5 67.6 Low Optical only Adaptive selection 53.1 67.4 High

Concat Opt+SAR Optical only 56.8 69.2 High Concat Opt+SAR Concat Opt+SAR 57.2 73.0 Low Concat Opt+SAR Adaptive selection 57.2 72.6 High

Table K. Ablation study on multi-modal reasoning. ”Efficiency” indicates inference efficiency: ”High” for methods with shorter context length (single modality or adaptive selection), ”Low” for concatenation methods that double the visual token count.

porating L2-Level data enables strong generalization across diverse task types. The full Terra-CoT mixture achieves the best overall performance, with improvements scaling consistently as we increase the proportion of reasoning data.

Ablations about multi-modal reasoning. We investigate how multi-modal data (optical and SAR) contributes to TerraScope’s performance. We design ablation experiments by controlling two aspects: (1) Multi-modal encoding: whether to concatenate optical and SAR features as input to the LLM during initial image encoding; (2) Masked feature interleaving: how to inject masked visual features during reasoning steps—using optical only, concatenating both modalities, or adaptively selecting based on relevance scores (Eq. 4-5).

Tab. K presents results on TerraScope-Bench, evaluated on both segmentation quality (mean IoU) and final answer accuracy.

Our ablation study reveals two key findings. First, multimodal encoding is essential for both accurate segmentation and reasoning. Concatenating optical and SAR features as initial input substantially improves performance compared to optical-only encoding, demonstrating that the LLM benefits from complementary multi-modal representations from the beginning of reasoning. Second, the masked feature injection strategy during reasoning steps also matters. Both concatenation and adaptive selection of masked features significantly outperform optical-only injection. While con-

catenation achieves slightly higher answer accuracy, adaptive selection demonstrates a favorable trade-off: it maintains comparable segmentation quality and nearly equivalent reasoning performance while significantly reducing context length by dynamically selecting only the most informative modality at each spatial location. This reduction in context length translates to substantial savings in memory consumption and inference time, making adaptive selection the more practical choice for deployment.

#### J. Additional Visualizations and Failure Analysis

##### J.1. Qualitative Results

Fig. E presents additional qualitative results demonstrating TerraScope’s capabilities across diverse scenarios. The visualizations show that TerraScope can perform pixelgrounded reasoning on: (1) single-modality optical imagery, generating accurate segmentation masks and spatial analysis; (2) multi-modal optical-SAR fusion, adaptively selecting the most informative modality for each spatial region; and (3) temporal change detection, providing chainof-thought reasoning traces that explain land cover changes with supporting visual evidence. These results validate TerraScope’s versatility in handling different data modalities and temporal information while maintaining pixel-level grounding throughout the reasoning process.

##### J.2. Failure Cases and Analysis

Fig. F presents typical failure cases to understand TerraScope’s limitations. We identify two primary failure modes:

- (1) Limited spectral information. TerraScope currently processes only RGB bands as input, discarding additional spectral channels available in multispectral sensors like Sentinel-2 (which provides 13 bands including nearinfrared, red-edge, and shortwave infrared). This limitation makes it challenging to distinguish spectrally similar land cover types that appear visually identical in RGB but exhibit distinct spectral signatures in other bands. For example, certain crop types or vegetation health conditions that are easily separable using NDVI or red-edge indices become ambiguous in RGB-only input, leading to incorrect segmentation and subsequent reasoning errors.
- (2) Error propagation from segmentation. For scenes containing small or low-contrast objects (e.g., narrow roads, sparse buildings, thin water channels), the mask decoder may produce inaccurate segmentation due to insufficient visual salience. These segmentation errors directly propagate to the reasoning stage: when spatial claims are grounded in incorrect masks, the derived answers become unreliable even if the reasoning logic is sound. This highlights the critical dependency of pixel-grounded reasoning on highquality segmentation, particularly for fine-grained objects in complex landscapes.

Future improvements could address these limitations by: (1) extending the vision encoder to process full multispectral inputs rather than RGB only, enabling better spectral discrimination; (2) incorporating uncertainty estimation in the segmentation module to flag low-confidence masks and trigger refinement; and (3) developing iterative refinement mechanisms that allow the model to correct initial segmentation errors through multi-step reasoning.

