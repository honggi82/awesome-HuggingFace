[Figure 1]

### See What I Mean: Aligning Vision and Language Representations for Video Fine-grained Object Understanding

|DescribeQibintheHou<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]|
|---|

Boyuan Sun1,2 Bowen Yin1,2 Yuanming Li2 Xihan Wei2

1,3*

1VCIP, CS, Nankai University 2Tongyi Lab, Alibaba Group 3NKIARI, Shenzhen Futian

|[Figure 6]|
|---|

[Figure 7]

#### Abstract

Video

# arXiv:2605.18018v1[cs.CV]18May2026

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

VisualPromptVideo

[Figure 12]

[Figure 13]

SeeWhatIMean

[Figure 14]

[Figure 15]

We present SWIM (See What I Mean), a novel training strategy that aligns vision and language representations to enable fine-grained object understanding solely from textual prompts. Unlike existing approaches that require explicit visual prompts, such as masks or points, SWIM leverages

[Figure 16]

Fine-grainedModel

Video

[Figure 17]

[Figure 18]

[Figure 19]

VisualPromptVideo

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Describe the

[Figure 25]

[Figure 26]

|[Figure 27]|
|---|

[Figure 28]

|mask|
|---|

Model

[Figure 29]

k supervision only during training to guide crossmodal attention, allowing the

Model

Text Prompt

|model to at inference.<br><br>[Figure 30]<br><br>[Figure 31]<br><br>Describe the man walk down the street.|
|---|

[Figure 32]

automatically attend

[Figure 33]

[Figure 34]

|to attention<br><br>[Figure 35]|
|---|

the user-specified object e. Our crossatt analysis of pretrained multimodal large language models (MLLMs) reveals a systematic discrepancy: Attri produce sharp, localized activations in the vis , whereas object nouns yield diffuse and scattered patterns due to semantic reference bias and distributed high-level representations. To address this misalignment, we construct NL-Refer, an enriched dataset, in which each object mask is paired with a precise natural language referring expression. SWIM extracts multilayer cross-attention maps from object nouns and enforces spatial consistency with ground-truth masks. Experimental results demonstrate that SWIM substantially improves text–visual alignment and achieves superior performance over visual-prompt-based methods on fine-grained object understanding benchmarks. The code and data are available at https://github.com/HumanMLLM/SWIM.

Text

[Figure 36]

Describe the

NO Visual Prompt Needed

[Figure 37]

Text Prompt

[Figure 38]

[Figure 39]

|tribute wor visual moda<br><br>Describe the marked <region>|ds<br><br>[Figure 40]<br><br>[Figure 41]|
|---|---|
| |lit|

(a) Classical Fine-grained model (b) See What I Mean (SWIM)

Text

[Figure 42]

NO Visual Prompt Needed

Figure 1. Pattern comparison between classical fine-grained model and SWIM. Our SWIM extracts object from pure natural language, requiring no visual prompt or any extra modules.

[Figure 43]

(a) Classical Fine-grained model (b) See What I Mean (SWIM)

|[Figure 44]<br><br>[Figure 45]<br><br>Describe the<br><br>[Figure 46]<br><br>[Figure 47]|
|---|

grained object understanding abilities.

To enhance fine-grained object perception and understanding, a typical paradigm [7, 23, 101, 103] is to introduce additional region-level encoders that produce object-level embeddings, thereby explicitly modeling individual object tokens. In the video field, several approaches [41, 50, 89] extend this idea by incorporating explicit visual prompts, such as points [50], masks [90], or bounding boxes [85], to guide the model toward specific object regions, as shown in Fig. 1(a). While these approaches can successfully identify target objects through explicit visual cues, their complex designs depend on extra visual inputs, increase complexity, and diverge from the way users most naturally interact with MLLMs. In fact, specifying objects through pure natural language [2, 48, 75] is both more intuitive and far more common in real-world scenarios.

#### 1. Introduction

With the rapid development of large language models [48, 78, 79], multimodal large language models (MLLMs) [1, 62, 70, 75, 109] that can jointly reason over visual and textual modalities have recently achieved remarkable progress. Benefiting from large-scale pretraining on massive multimodal datasets [28, 37, 45, 105], general-purpose MLLMs [38, 60, 112] have demonstrated outstanding performance in holistic scene understanding. However, despite these impressive capabilities, they often struggle to consistently focus on user-specific objects, limiting their fine-

Our motivation stems from this mismatch. As demonstrate in Fig. 1(b), we aim to design a model that can directly locate and attend to the correct object purely from the pure textual prompt to achieve natural and fine-grained crossmodal understanding without any extra visual input in inference. To achieve this, we first explore how existing models attend to objects mentioned in text prompt [66, 92]. Considering cross-attention between textual and visual tokens

*Corresponding author.

[Figure 48]

###### The young man with short, light brown hair and a beard

###### The striped shirt and blue baseball cap-wearing man

shirt baseball cap man

man brown beard

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Qwen2.5-VL

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

###### SWIM

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Object Nouns Attribute Words

Attribute Words Object Nouns

Figure 2. Visual comparisons of cross-attention maps for object nouns and attribute words between Qwen2.5-VL [2] and SWIM. The attribute words tied with low-level texture exhibit sharp and localized attention over correct regions, whereas the object nouns corresponding to high-level semantic concepts result in diffuse activations. This discrepancy motivates our explicit supervision strategy in SWIM.

Based on the NL-Refer dataset, we propose SWIM (See What I Mean), a simple yet effective training strategy that explicitly aligns vision and language representations to strengthen fine-grained object understanding in MLLMs. Specifically, during supervised fine-tuning, SWIM extracts cross-attention maps for object nouns from multiple intermediate layers and aligns them with ground-truth object masks, enforcing spatial consistency between textual identity and visual grounding. By providing explicit alignment signal throughout training, SWIM guides the model to preserve and utilize fine-grained object-level information, enabling more precise visual localization from purely textual prompts at inference. Extensive experiments across finegrained object understanding benchmarks demonstrate that SWIM enhances text–visual alignment and outperforms visual-prompt-dependent approaches.

is a direct indicator of multimodal interaction [16, 54, 83], it can reveal whether a text token successfully grounds in a relevant visual region [82, 93]. Therefore, by visualizing cross-attention maps for object-related words of Qwen2.5VL [2] in Fig. 2, we aim to uncover alignment patterns and weaknesses not apparent from standard accuracy metrics.

Interestingly, our cross-attention analysis reveals a systematic discrepancy: Attribute words [27, 36, 56] produce sharper and more localized activations in the visual modality, while object nouns [52] result in diffuse and scattered attention patterns. We attribute this discrepancy to biases in semantic reference. In large-scale multimodal corpora, attribute words, such as colors or textures, correspond to specific and spatially localized visual patterns, while object nouns occur in diverse contexts, diluting their spatial association. Moreover, attribute words naturally map to low-level visual features, while object nouns rely on high-level semantic representations often varied across instances, which leads to poor alignment without explicit supervision.

We summarize our contributions as follows.

- • We point out design limitations in existing fine-grained object understanding models and the insufficient vision–language alignment in general MLLMs. Based on the observed systematic discrepancy, we introduce NLRefer dataset, in which each object is referred with explicit natural language expressions.
- • We propose a novel training strategy, SWIM (See What I Mean), which explicitly enforces alignment between visual content and object nouns during training, resulting in a model that requires no visual prompt inputs at inference.
- • Through experiments on fine-grained object understanding benchmarks, SWIM demonstrates consistent improvements over visual-prompt-based approaches. Quantitative and qualitative analyses of text–visual alignment further corroborate our claim.

This finding suggests that improving fine-grained object understanding requires explicitly strengthening crossmodal correspondence for object nouns, and thus inspires us to pursue direct supervision between object words and their associated visual regions. To provide such supervision signals, an enriched video understanding dataset that pairs object-level visual annotations with natural language prompts containing clear references is required. Thanks to earlier visual-prompt-based approaches [50, 89], collecting training data with mask annotations is not difficult. We start from VideoRefer [89], a video fine-grained object understanding dataset providing frame-level object masks aligned with textual prompts via placeholder tokens used for visual prompting. While these masks are valuable, the associated text does not contain clear natural language references to the objects. We therefore design a GPT-4o-powered [49] data refinement pipeline and construct the NL-Refer dataset. Specifically, for each placeholder, we automatically replace it with a concise natural language description of the specific instance, informed by the context.

#### 2. Related Work

##### 2.1. Multimodal Large Language Model

Multimodal large language models (MLLMs) [8, 13, 21, 43, 81] integrate visual signals with textual inputs, leveraging the powerful reasoning and generative capabilities of

𝐻 𝐻

Attention Matrix

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

”Can you …

Large Language Model

[Figure 79]

”Can you discuss in detail the important elements of the marked area <region> in the video?”

| | |
|---|---|
| | |
| | |

[Figure 80]

[Figure 81]

walking on the street in the video?”

[Figure 82]

Transformer Layers

[Figure 83]

𝐻𝑊

[Figure 84]

[Figure 85]

[Figure 86]

Tokenizer

[Figure 87]

𝑙-th Layer

[Figure 88]

[Figure 89]

Rewrite 𝐻 with natural language according to 𝐺

[Figure 90]

[Figure 91]

[Figure 92]

…

[Figure 93]

[Figure 94]

[Figure 95]

Vision Encoder

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

”The young man with short, light brown hair and a beard is walking down the street in a white t-shirt featuring a graphic design ...”

## …

𝐀 𝑀

[Figure 101]

[Figure 102]

…

Text Loss

𝐺

Video Frame ℒ

[Figure 103]

- Figure 3. Training pipeline of SWIM. Explicitly supervision is applied on cross-attention between object noun and visual tokens, enable accurate fine-grained object grounding from pure neutral text prompts at inference without any extra visual prompt.

walking

LLMs [11, 18, 22, 42, 46, 61] to tackle a wide range of tasks [30, 69, 107]. Beyond image-based approaches [40, 44], recent advances in spatiotemporal architectures design [72, 96, 97] enables MLLMs to extend multimodal understanding into the video filed [47, 59, 100], achieving strong performance in real-world applications [17, 25].

which brings extra computational cost and deviates from typical user interaction patterns. Meanwhile, SWIM adopts a training paradigm with explicit su crossmodal alignment, enabling fine-grained understanding from natural language references architecture modifications or any extra visual prompts

𝐻 𝐻

Attention Matrix

[Figure 104]

[Figure 105]

”Can you …the young

Large Language Model

[Figure 106]

”Can you discuss in detail the important elements of the marked area <region> in the video?”

|supervision<br><br>object without<br><br>at infer|for<br><br>ence.|
|---|---|
| | |
| | |

[Figure 107]

[Figure 108]

walking on the street in the video?”

[Figure 109]

[Figure 110]

Transformer Layers

[Figure 111]

𝐻𝑊

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Tokenizer

[Figure 116]

𝑙-th Layer

[Figure 117]

[Figure 118]

Rewrite 𝐻 with natural language according to 𝐺

However, despite advances, MLLMs still face challenges in fine-grained object understanding, especially when identifying or describing user-specified targets from solely textual prompts [71]. One potential reason lies in the issue of vision–language alignment within such models [68, 99]. As shown in Fig. 2, the discrepancy that attribute words tend to produce clear attention patterns, while object nouns often result in diffuse and scattered activations motivates us to design SWIM, which applies supervision on cross-modal correspondence of specific textual tokens. Although prior studies [29, 31, 32, 58, 106] examined intermediate feature representations in MLLMs, and works like Cambrain [65] and VIRAL [84] explores reconstructing visual features from intermediate layers, they generally focus on visual embeddings and ignore the visual-language alignment.

[Figure 119]

[Figure 120]

#### 3. See What I Mean (SWIM)

[Figure 121]

…

[Figure 122]

[Figure 123]

[Figure 124]

Vision Encoder

##### 3.1. NL-Refer: Dataset Construction

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

”The young man with short, light brown hair and a beard is walking down the street in a white t-shirt featuring a graphic design ...”

To provide explicit supervision for aligning object nouns in text with their corresponding visual regions, we construct NL-Refer from the VideoRefer dataset, which can be represented as

## …

𝐀 𝑀

[Figure 130]

[Figure 131]

…

Text Loss

𝐺

Video Frame ℒ

[Figure 132]

DVideoRefer = (Vi,Hi, Gi, Mi) i = 1,...,N , (1)

where Vi denotes the video, Hi denotes the “human” message containing the placeholder token <region> , Gi denotes the paired “gpt” response describing the marked object in natural language, and Mi denotes the corresponding pixel-level instance masks for the target region.

While such placeholders support visual-prompt-based methods, they do not convey the explicit semantic identity of the referenced object, thereby inhibiting a model’s ability to learn robust and direct text–visual correspondences for object nouns. To mitigate this, we propose a refinement process leverages GPT-4o to replace each placeholder <region> in the human message Hi with a concise and unambiguous natural language referring expression ri that specifies the object instance, using descriptive details drawn from the paired “gpt” response Gi. This generation process maintains the original conversational structure while embedding explicit semantic content in the prompt. Within each ri, GPT-4o identifies the single most representative object noun wi, the lexical item that captures the core semantic identity of the target, and surrounds it with a special markup token <ins> to enable deterministic location

##### 2.2. Fine-grained Object Understanding

To enhance the fine-grained understanding capability of MLLMs, recent approaches [4, 6, 19, 26, 55, 73, 76, 80, 91, 95, 98, 102, 110] start to focus on object-centric perception and reasoning. The most common paradigm [23, 24, 64, 85, 86, 88, 101] is to involve explicit visual prompts (points/boxes/masks) as guidance and additional encoders to improve fine-grained comprehension of local regions. For example, VideoRefer [89] and PixelRefer [103] employ masks as visual indicators and leverage extra visual encoders to enhance the perception of objects. They also contribute valuable video fine-grained datasets with mask annotations.

In a word, these methods require additional encoders and extra visual prompt [5, 77, 111] even at inference time,

of the corresponding token for further training. This inline tagging is directly linked to Mi, so that the marked noun token is aligned with the ground-truth mask it denotes.

Formally, let Hi denote the original human message and Gi the corresponding model response in VideoRefer, paired with mask Mi. The refined human message Hˆi is given by

###### Hˆi = Mark(Replace(Hi, <region>, ri), wi), (2)

where Replace(·) substitutes the placeholder with the GPT-4o-generated referring expression ri, and Mark(·,wi) encloses wi in <ins> delimiters. The referring expression itself is obtained as

###### ri = NLRef(Gi), (3)

where NLRef(·) extracts salient object descriptors from Gi and composes them into a minimal, discriminative phrase.

The refined dataset is then defined as DNL−Refer = (Vi,Hˆi, Gi, Mi) i = 1,...,N , (4)

with Hˆi containing the linguistically explicit object reference, Gi providing the unchanged descriptive context, and Mi denoting the ground-truth mask of the target instance. By systematically embedding object identity into the text and marking its precise token span, DNL−Refer establishes a reliable mapping between lexical items and visual annotations, laying a solid foundation for subsequent crossattention supervision of object nouns.

##### 3.2. Attention Regularization

Leveraging DNL−Refer, in which each refined textual prompt Hˆi contains precisely one object noun wi tagged with <ins> delimiters and deterministically linked to the ground-truth mask Mi, we design an auxiliary supervision mechanism to explicitly align the visual grounding of noun tokens with their annotated object regions. As analyzed in Section 1, the cross-attention patterns in existing MLLMs exhibit an empirical vision–language misalignment: Object nouns often produce diffuse and scattered activations across the visual tokens. This systematic discrepancy motivates us to directly guide the model’s cross-modal attention when processing tagged object noun tokens, so that their activation is concentrated on the relevant visual region.

During training, we first tokenize Hˆi into a sequence of Lt tokens, yielding text embeddings Xt ∈ RL

t×d, where d denotes the hidden dimension. Let ji ∈ {1,...,Lt} be the index corresponding to the tagged noun token wi. Within the LLM decoder, these text embeddings interact with a sequence of Lv visual tokens Xv ∈ RL

v×d through crossattention layers. For a given cross-attention module at layer index l, let Qtl[ji] ∈ Rd be the query vector of the tagged noun token at position ji, and let Kvl ∈ RL

v×d denote the key vectors of the all visual tokens of one frame from Vi.

The cross-attention weights from wi to the visual tokens at layer l are computed as:

Al,i = softmax

Qtl[ji](Kvl )⊤

√

d

, (5)

where the softmax is applied over the Lv visual token positions. Each element indicates the degree to which the noun token attends to each visual token at layer index l.

To enable spatial supervision, the attention vector Al,i is mapped to the original feature grid of resolution (H,W) that aligns with Mi. This mapping follows the spatial correspondence between visual tokens and encoder patches. If (H,W) differs from the token grid resolution, bilinear interpolation is applied to match the mask resolution exactly. The resulting attention map for layer l is denoted as Al,i ∈ [0,1]H×W. Since attention patterns may vary across layers, we aggregate attention maps from a selected set of layers S by simple averaging:

1 |S| l∈S

A¯ i =

Al,i. (6)

The aggregated map A¯ i captures the stable cross-modal correspondence between the tagged object noun wi and its visual region after accounting for multi-layer variability.

Finally, we supervise A¯ i with the binary mask Mi using a pixel-wise binary cross-entropy loss:

H

W

1 HW

L(BCEi) = −

Mi(u,v) log A¯ i(u,v)

u=1

v=1

+ 1 − Mi(u,v) log 1 − A¯ i(u,v) ,

(7)

where Mi(u,v) ∈ {0,1} indicates whether pixel (u,v) belongs to the target object. By providing this explicit alignment signal at training time, the model learns to consistently concentrate cross-attention from object nouns onto their correct visual regions, bridging the alignment gap identified in our analysis and enhancing fine-grained understanding performance without modifying the base architecture.

Notably, unlike many existing fine-grained object understanding approaches that require the visual prompt mask Mi as part of the inference input, in our SWIM, Mi is only used for attention regularization during supervised fine-tuning, incurring no additional burden in inference.

#### 4. Experiments

##### 4.1. Experimental Settings

Implementation details. We implement SWIM on top of the widely used Qwen2.5VL-7B [2] framework, which employs SIGLIP (so400m-patch14-384) [94] as the visual encoder and Qwen2.5 [63] as the large language model (LLM)

- Table 1. Performance comparisons on fine-grained VideoRefer-Bench-D and VideoRefer-Bench-Q. The best results are bold and the second-best results are underlined.

VideoRefer-Bench-Q VideoRefer-Bench-D Method Prompt types Basic Seq. Rel. Rea. Fut. Avg. SC AD TD HD Avg.

Generalist Models

LongVU-7B [57] Text 47.2 61.3 57.5 85.3 65.8 61.0 2.33 1.80 2.39 1.68 2.05 LongVA-7B [100] Text 56.2 62.5 52.0 83.9 65.8 61.8 3.02 2.30 1.92 2.51 2.44 LLaVA-OV-7B [33] Text 58.7 62.9 64.7 87.4 76.3 67.4 3.09 1.94 2.50 2.41 2.48 Qwen2-VL-7B [62] Text 62.0 69.6 54.9 87.3 74.6 66.0 3.99 3.05 2.44 2.44 2.97 Qwen2.5-Omni-7B [74] Text 65.0 67.6 54.0 84.3 70.3 68.2 3.92 2.82 1.97 2.34 2.76 InternVL2-26B [10] Text 58.5 63.5 53.4 88.0 78.9 65.0 4.08 3.35 3.08 2.28 3.20 Qwen2.5-VL-7B [2] Text 78.0 69.7 58.2 79.9 73.2 71.8 4.33 3.19 2.88 2.58 3.24 GPT-4o-mini [49] Text 57.6 67.1 56.5 85.9 75.4 65.8 3.89 3.18 2.62 2.50 3.05 GPT-4o [49] Text 62.3 74.5 66.0 88.0 73.7 71.3 4.15 3.31 3.11 2.43 3.25

Specialist Models

Elysium-7B [67] Box – – – – – – 2.35 0.30 0.02 3.59 1.57 Artemis-7B [53] Box – – – – – – 3.42 1.34 1.39 2.90 2.26 Osprey-7B [88] Point, Mask 45.9 47.1 30.0 48.6 23.7 39.9 3.30 2.66 2.10 1.58 2.41 Ferret-7B [85] Point, Box, Mask 35.2 44.7 41.9 70.4 74.6 48.8 3.20 2.38 1.97 1.38 2.23 PAM-3B [41] Point, Box, Mask – – – – – – 3.92 2.84 2.88 2.94 3.14 DAM-8B [39] Point, Box, Mask – – – – – – 4.69 3.61 3.34 3.03 3.68 VideoRefer-7B [89] Mask 75.4 68.6 59.3 89.4 78.1 71.9 4.44 3.27 3.10 3.04 3.46 SWIM Text 83.8 75.0 66.7 93.7 80.7 78.3 4.92 3.85 3.43 2.96 3.78

- Table 2. Performance comparisons on general video benchmarks. Method MVBench VideoMME ActivityNet

ness of SWIM, we evaluate it from both fine-grained video object understanding and general video understanding perspectives. We adopt VideoRefer-Bench [89], a dedicated benchmark for object-level video understanding that comprises two sub-tasks. VideoRefer-Bench-D measures description generation for specified objects, containing 400 curated entries from Panda-70M [9]. Outputs are scored from 0-5 in four aspects: Subject Correspondence (SC, subject matches ground truth), Appearance Description (AD, accuracy of color/shape/texture), Temporal Description (TD, correctness of motion), and Hallucination Detection (HD, absence of invented details). VideoReferBench-Q evaluates object-level understanding and reasoning, consisting of 198 videos from DAVIS-2017 [51] and MeViS [14, 15], paired with 1,000 region-linked multiplechoice questions spanning Basic (simple factual queries), Sequential (temporal order reasoning), Relationship (relations between objects), Reasoning (context-based inference), and Future (predict future states).

VideoLLaMA2 [12] 54.6 47.9 50.2 VideoChat2-HD [34] 51.1 54.6 49.1 VideoLLaMA2.1 [12] 57.3 54.9 LLaVA-Next-Video [104] - 46.5 53.5 LLaVA-Octopus [108] 51.7 55.7 53.4 INST-IT [50] - 54.0 55.2 VideoRefer [89] 59.6 55.9 SWIM 62.1 55.9 55.6

decoder. Our training set is composed of two parts: (1) The proposed NL-Refer dataset, converted from the detailed caption subset of VideoRefer-700K dataset [89], containing 125K videos with refined textual annotations that explicitly refer to objects in natural language along with their corresponding instance masks; (2) A portion of general video-based QA data from LLaVA-Video-178K [105] and videorefer-qa-75k. For the LLaVA-Video-178K, we decompose multi-turn dialogues into single-turn QA pairs (1.3M in total), and sample 100K QA pairs (approximately 7.5%) for training. We also sample 10K QA pairs from videorefer-qa-75k to maintain its ability on multi-choice question. In total, our training data contains 235K examples, which is significantly smaller than that used for most generalist MLLMs and is less than 1/3 of the VideoRefer. All experiments are conducted on 8× NVIDIA A100 GPUs.

As for general video understanding, we adopt three representative benchmarks: MVBench [35], which offers diverse multi-aspect evaluations of video-language reasoning; Video-MME [20], a comprehensive suite covering spatiotemporal reasoning, event localization, and attribute recognition; and ActivityNet-QA [87], a large-scale QA dataset based on ActivityNet videos targeting a wide range of skills from event recognition to temporal reasoning. These benchmarks jointly examine SWIM’s generalization ability beyond fine-grained object grounding.

Evaluation benchmarks. To demonstrate the effective-

Table 3. Ablation study of attention layer selection among different layer number and layer index. Layer Number Layer Index VideoRefer-D Layer Number Layer Index VideoRefer-D

1 [1] 3.43 6 [1, 3, 5, 7, 9, 11] 3.73 1 [13] 3.48 6 [1, 6, 11, 16, 21, 26] 3.78 1 [27] 3.52 6 [2, 7, 12, 17, 22, 27] 3.78 3 [1, 13, 27] 3.72 6 [17, 19, 21, 23, 25, 27] 3.76 3 [9, 18, 27] 3.70 9 [1, 4, 7, 10, 13, 16, 19, 22, 25] 3.75 3 [1, 7, 13] 3.69 14 [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27] 3.77

Table 4. Ablation study of attention layer fusion methods on VideoRefer-Bench-D. Prod. denotes element-wise product.

Subject Correspondence

Temporal Description

Fusion

Avg.

Add 4.62 3.24 3.57 Pool 4.56 3.11 3.49 Prod. 4.81 3.21 3.55 Mean 4.92 3.43 3.78

##### 4.2. Main Results

###### 4.2.1. Results on Fine-grained Benchmarks

In Tab. 1, we summarize the performance of SWIM and a range of generalist and specialist state-of-art approaches on the fine-grained VideoRefer-bench [89], which evaluates video fine-grained object understanding in both questionanswering and description settings.

On VideoRefer-Q, which evaluates fine-grained object understanding through five sub-tasks, SWIM attains substantial gains in Basic (+5.8% over Qwen2.5-VL-7B [2]) and Sequential (+5.3%) cases, which demand precise object identification before answering. SWIM yields an average accuracy of 78.3%, exceeding the VideoRefer-7B [89] by +6.4%, and surpassing all generalist models such as Qwen2.5-VL-7B (71.8) and GPT-4o (71.3) [49].

On VideoRefer-D, which assesses spatial correspondence (SC), action description (AD), temporal description (TD), and higher-level human–object description (HD), SWIM achieves 4.92, 3.85, 3.43, and 2.96 respectively, for an average of 3.78, outperforming the best specialist baseline DAM-8B (3.68) and the strongest generalist system GPT-4o (3.25). The performance gain on SC (+0.23) and AD (+0.24) over DAM-8B [39] highlights the strength of SWIM in aligning object nouns to precise instance regions.

Overall, SWIM delivers consistent improvements across both QA and description tasks, indicating more precise referring capability between natural language and visual regions. By integrating explicit attention regularization alignment supervision at training only, SWIM enhances the finegrained grounding capability of MLLMs without incurring architectural changes or inference-time visual prompting, making it competitive across diverse evaluation scenarios that demand high-resolution text–visual alignment.

Table 5. Ablation study of attention loss function used in SWIM on VideoRefer-Bench-D.

Subject Correspondence

Temporal Description

Loss

Avg.

mIoU 4.88 3.34 3.71 Focal 4.80 3.24 3.69 Dice 4.90 3.38 3.74 BCE 4.92 3.43 3.78

###### 4.2.2. Results on General Benchmarks

Other than the fine-grained understanding benchmarks, we also evaluate SWIM on several representative general video understanding benchmarks, including MVBench [35], Video-MME [20], and ActivityNet-QA [87], as summarized in Tab. 2. The results indicate that while SWIM is primarily optimized for fine-grained video object understanding tasks, its performance on broader video-language understanding tasks remains within a competitive range compared to existing methods. This suggests that the proposed training strategy for cross-attention alignment does not substantially compromise general video understanding ability, allowing SWIM to retain acceptable performance in more comprehensive scenarios.

##### 4.3. Ablation Analysis 4.3.1. Effect of Attention Layer Selection

We first explore the influence of the number and positions of cross-attention layers used for supervision in SWIM. As shown in Tab. 3, we evaluate VideoRefer-D performance under configurations ranging from single-layer supervision to selecting up to 14 layers. We find that increasing the number of supervised layers yields consistent improvements initially. The performance rises from 3.43 at a single shallow layer to 3.78 when supervising six layers. Beyond six layers, results tend to be stable with all larger configurations remaining within 0.02 of each other. Furthermore, evenly spaced selection of supervision layers across the network produces better or comparable results than densely clustered layers in a narrow depth range, indicating that a balanced distribution from early to late stages fosters more stable cross-modal alignment. This observation suggests that SWIM achieves the best trade-off with moderately deep and uniformly spaced supervision layers.

SWIM Performance Trend with different data scale

SWIM (VideoRefer-D)

3.78

3.8

PerformanceScore(VideoRefer-D)

Qwen2.5VL-7B

VideoRefer-7B

3.69

DAM-8B

3.68

3.7

3.60

3.6

3.5

3.46

3.39

3.4

3.3

3.24

3.23

3.2

30K 50K 80K 100K 125K

Amount of Data with Mask

- Figure 4. Scalablity of SWIM. The performance of SWIM scales consistently with the increase in data scale.

###### 4.3.2. Effect of Attention Layer Fusion

We further study how attention maps extracted from multiple layers should be fused to provide the alignment signal in SWIM. Several fusion strategies are considered, including addition, pooling, mean, and element-wise product. As shown in Tab. 4, simple mean aggregation yields the highest average score (3.81), exceeding addition (3.57) and pooling (3.49) by a clear margin. This can be attributed to its ability to preserve consistent spatial patterns across layers without introducing bias toward any single depth, effectively smoothing noise and retaining salient activation peaks. In contrast, The element-wise product requires all layers to highlight the same locations to retain them, thus tends to over-suppress regions with moderate but meaningful attention, reducing overall coverage of the target object.

###### 4.3.3. Effect of Loss Function

We also examine the impact of the loss function used to supervise cross-attention in SWIM. As demonstrated in Tab. 5, compared to alternatives such as mIoU, Focal, and Dice losses, binary cross-entropy (BCE) consistently yields superior overall performance. This advantage can be attributed to the sparsity inherent in attention maps extracted from MLLMs. When aligning object nouns to visual regions, activations are highly localized and occupy only a small fraction of the spatial grid due to the softmax operation. Loss functions that emphasize overlap ratios or focus disproportionately on hard negatives may under-penalize diffuse activations, making it harder to enforce precise alignment. In contrast, BCE treats each pixel independently and applies a uniform probabilistic penalty across all spatial locations, encouraging suppression of irrelevant highactivation regions while reinforcing confident attention on the target area. This balanced penalization aligns well with the fine-grained supervision signals in SWIM, leading to more accurate and stable grounding of object nouns.

Table 6. GamePoint@P between Qwen2.5-VL and SWIM. This metric measures the visual region with the highest cross-attention score falls within the specified object mask.

###### Method G.P.@P-1 G.P.@P-5 G.P.@P-10

Qwen2.5-VL-7B 0.329 0.293 0.270 SWIM 0.392 0.348 0.317

##### 4.4. Scalability with Mask-Annotated Data Volume

Beyond the intrinsic performance of the model, scalability is also a critical property for multimodal models. In the field of fine-grained object understanding, scalability determines whether stronger alignment can be achieved simply by expanding high-quality annotated datasets, making it essential for long-term advancement. Therefore, to examine SWIM’s scalability, we vary the number of mask-annotated training videos from NL-Refer dataset. The dataset size is gradually expanded from a 30K subset to the maximum available 125K samples. Fig. 4 shows the overall alignment score at each scale, revealing a clear and monotonic upward trend.

The results demonstrate that SWIM can effectively leverage additional fine-grained supervision. Each increase in mask-annotated data yields measurable gains, and the improvement persists up to the largest scale tested. This sustained growth can be attributed to the explicit alignment between object nouns and masks in our training pipeline, which enables the model to refine cross-modal attention alignments across diverse examples without overfitting to narrower data distributions.

In addition, the absence of a plateau at 125K data scale indicates that SWIM is inherently capable of benefiting from larger-scale mask supervision. Although our experiments are bounded by the current data availability, the persistent upward trajectory suggests considerable untapped potential. SWIM may achieve stronger alignment if broader and more diverse mask-annotated corpora are provided.

##### 4.5. GamePoint-based Attention Localization

We evaluate spatial grounding using the GamePoint@P metrics, as reported in Tab. 6. GamePoint@P measures the proportion for the top P% attention pixels highest-attention pixels in A¯ i that fall within the object mask Mi:

N

|TopPerc(A¯ i,P) ∩ Pi| |TopPerc(A¯ i,P)|

1 N

, (8)

GamePoint@P =

i=1

where Pi = {(u,v) | Mi(u,v) = 1} denotes object-region pixels, TopPerc(·) denotes selecting the top P% elements.

From Tab. 6, SWIM consistently outperforms Qwen2.5VL-7B across all P. The improvement is most pronounced at P = 1 (+6.3%) and P = 5% (+5.5%), showing that SWIM’s most confident attention points and top regions are far more likely to land within the correct object area. This

[Figure 133]

[Figure 134]

[Figure 135]

Please describe the rectangular wooden cutting board on the table in the video.

What is the distinguishing feature of the goldfish on the right side near the plants?

What activity might the person in the red jacket and black helmet be engaged in?

The biscuit is a round, flat piece of bread with a light, soft texture. It is placed on a wooden tray and appears to be freshly baked, as it has a golden-brown crust and a lighter, soft interior. The biscuit is being moved around by a person's hands, which are seen picking up and placing it back down on the tray.

Qwen2.5-VL

- (A) Protecting another one
- (B) Taking a selfie or recording a video
- (C) Paragliding
- (D) Flying

- (A) It has a white stripe
- (B) It is the largest goldfish
- (C) It is the smallest goldfish
- (D) It has a pink hue

v

[Figure 152]

The wooden cutting board is rectangular and placed on a table. It is used to hold the biscuits. The board is stationary and not in motion.", SWIM

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

D

C

[Figure 158]

[Figure 159]

[Figure 160]

Qwen2.5-VL

Qwen2.5-VL

GT: The wooden cutting board is rectangular and placed on a table with a checkered tablecloth. It is brown and has a rustic appearance. The cutting board is stationary and not in motion

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

B

B

[Figure 165]

[Figure 166]

SWIM

SWIM

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

What activity might the person in the red jacket and black helmet be engaged in?

What is the distinguishing feature of the goldfish on the right side near the plants?

[Figure 176]

[Figure 177]

[Figure 178]

Please describe the rectangular wooden cutting board on the

table in the video.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

The biscuit is a round, flat piece of bread with a light, soft texture. It is placed on a wooden tray and appears to be freshly baked, as it has a golden-brown crust and a lighter, soft interior. The biscuit is being moved around by a person's hands, which are seen picking up and placing it back down on the tray.

Qwen2.5-VL

- (A) Protecting another one
- (B) Taking a selfie or recording a video
- (C) Paragliding
- (D) Flying

[Figure 185]

- (A) It has a white stripe
- (B) It is the largest goldfish
- (C) It is the smallest goldfish
- (D) It has a pink hue

v

[Figure 186]

[Figure 187]

The wooden cutting board is rectangular and placed on a table. It is used to hold the biscuits. The board is stationary and not in motion. SWIM

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

C

D

[Figure 192]

[Figure 193]

[Figure 194]

Qwen2.5-VL

Qwen2.5-VL

GT: The wooden cutting board is rectangular and placed on a table with a checkered tablecloth. It is brown and has a rustic appearance. The cutting board is stationary and not in motion

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

B

B

[Figure 199]

[Figure 200]

SWIM

SWIM

Figure 5. Qualitative comparisons between SWIM and Qwen2.5-VL [2].

suggests SWIM achieves sharper and more focused attention peaks on the target object, raising coverage across all top-P settings, and overcoming the diffuse attention patterns observed in the baseline Qwen2.5-VL model.

###### Qwen2.5-VL vs SWIM Metric Comparison

0.7

0.66

Qwen2.5-VL SWIM

| |
|---|

0.62

0.6

##### 4.6. Fine-Grained Text-Visual Alignment Metrics

0.49

0.5

To quantify visual–language alignment at a finer granularity, we further compare the attention maps A¯ i for object nouns against the corresponding masks Mi using four common metrics: Average Precision (AP), Area Under Curve (AUC), Normalized Scanpath Saliency (NSS), and Precision. For each metric, we derive binary predictions from A¯ i using a fixed threshold of 0.75 if confusion matrix components are required.

###### Score

0.40

0.39

0.4

0.30

0.3

0.28

0.26

0.2

AP AUC NSS Precision

As shown in Fig. 6, SWIM consistently outperforms the Qwen2.5-VL baseline across all four metrics (AUC: 0.62 → 0.67, NSS: 0.39 → 0.50, Precision: 0.28 → 0.39, AP: 0.26 → 0.30). These improvements indicate that SWIM generates attention maps with more precise and concentrated coverage of target regions, reduces false activations, and achieves stronger discriminability across thresholds, thereby enhancing fine-grained text–visual grounding.

Figure 6. Quantitative comparison of fine-grained text–visual alignment metrics. Evaluation includes AP, AUC, NSS, and Precision. SWIM consistently outperforms the Qwen2.5-VL baseline on all four metrics, indicating more accurate attention on target, fewer false activations, and stronger alignment stability.

#### 5. Conclusions

In this paper, we propose SWIM, a training paradigm that applies explicit supervision to improve cross-modal alignment between object nouns and visual regions, thereby enhancing fine-grained object understanding in MLLMs. To enable such supervision, we construct NL-Refer, a refined video dataset with natural language object references paired with mask annotations. SWIM requires no architectural changes and does not need any visual prompts during inference. Experiments show that it achieves state-of-theart results on fine-grained understanding benchmarks while maintaining competitive performance on general benchmarks. Extensive quantitative analysis further verifies that SWIM can achieve better fine-grained video understanding.

##### 4.7. Qualitative Comparisons

As demonstrated in Fig. 5, we conduct qualitative comparisons between SWIM and Qwen2.5-VL [2] on examples that require precisely object reference through natural language. For example, in the caption case, Qwen2.5-VL disregards the prompt’s explicit reference and instead describes the most visually salient object in the scene while SWIM adheres to the prompt and focus on the specified object. The other two examples also show cases where SWIM’s outputs match the prompt’s described referent. Fig. 5 indicates that SWIM’s outputs align more closely with the objects mentioned in the prompts.

#### 6. Acknowledgments

This work was supported by NSFC (62522607, 62495061, and 62276145), and the Fundamental Research Funds for the Central Universities (Nankai University).

#### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 2, 4, 5, 6, 8, 14
- [3] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 14
- [4] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In CVPR, pages 12914–12923, 2024. 3
- [5] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Vipllava: Making large multimodal models understand arbitrary visual prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12914–12923, 2024. 3
- [6] Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, and Yang Liu. Position-enhanced visual instruction tuning for multimodal large language models. arXiv preprint arXiv:2308.13437, 2023. 3
- [7] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 1
- [8] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 2
- [9] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 5
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou

- Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 5
- [11] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites. arXiv preprint arXiv:2404.16821, 2024. 3
- [12] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 5
- [13] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2
- [14] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2694–2703, 2023. 5
- [15] Henghui Ding, Chang Liu, Shuting He, Kaining Ying, Xudong Jiang, Chen Change Loy, and Yu-Gang Jiang. Mevis: A multi-modal dataset for referring motion expression video segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 5
- [16] Liwei Ding, Kowei Shih, Hairu Wen, Xinshi Li, and Qin Yang. Cross-attention transformer-based visual-language fusion for multimodal image analysis. International Journal of Applied Science, 8(1):p27–p27, 2025. 2
- [17] Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, et al. Docopilot: Improving multimodal models for document-level understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4026–4037, 2025. 3
- [18] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407,

2024. 3

- [19] Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. In NeurIPS, 2024. 3
- [20] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 5, 6, 14
- [21] Chaoyou Fu, Yi-Fan Zhang, Shukang Yin, Bo Li, Xinyu Fang, Sirui Zhao, Haodong Duan, Xing Sun, Ziwei Liu, Liang Wang, et al. Mme-survey: A comprehensive sur-

- vey on evaluation of multimodal llms. arXiv preprint arXiv:2411.15296, 2024. 2
- [22] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3
- [23] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13796– 13806, 2024. 1, 3
- [24] Miran Heo, Min-Hung Chen, De-An Huang, Sifei Liu, Subhashree Radhakrishnan, Seon Joo Kim, Yu-Chiang Frank Wang, and Ryo Hachiuma. Omni-rgpt: Unifying image and video region-level understanding via token marks. arXiv preprint arXiv:2501.08326, 2025. 3
- [25] Yusong Hu, Runmin Ma, Yue Fan, Jinxin Shi, Zongsheng Cao, Yuhao Zhou, Jiakang Yuan, Xiangchao Yan, Wenlong Zhang, Lei Bai, et al. Flowsearch: Advancing deep research with dynamic structured knowledge flow. arXiv preprint arXiv:2510.08521, 2025. 3
- [26] Xiaoke Huang, Jianfeng Wang, Yansong Tang, Zheng Zhang, Han Hu, Jiwen Lu, Lijuan Wang, and Zicheng Liu. Segment and caption anything. In CVPR, pages 13405– 13417, 2024. 3
- [27] Nikolai Ilinykh and Simon Dobnik. Attention as grounding: Exploring textual and cross-modal attention on entities and relations in language-and-vision transformer. In Findings of the association for computational linguistics: ACL 2022, pages 4062–4073, 2022. 2
- [28] Qing Jiang, Lin Wu, Zhaoyang Zeng, Tianhe Ren, Yuda Xiong, Yihao Chen, Liu Qin, and Lei Zhang. Referring to any person. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21667– 21678, 2025. 1
- [29] Zhangqi Jiang, Junkai Chen, Beier Zhu, Tingjin Luo, Yankun Shen, and Xu Yang. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 25004–25014, 2025. 3
- [30] Modi Jin, Yiming Zhang, Boyuan Sun, Dingwen Zhang, Ming-Ming Cheng, and Qibin Hou. Geoagent: Learning to geolocate everywhere with reinforced geographic characteristics. arXiv preprint arXiv:2602.12617, 2026. 3
- [31] Omri Kaduri, Shai Bagon, and Tali Dekel. What’s in the image? a deep-dive into the vision of vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14549–14558, 2025. 3
- [32] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. Your large vision-language model only needs a few attention heads for visual grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9339–9350, 2025. 3
- [33] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and

- Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 5
- [34] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 5
- [35] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 5, 6, 14
- [36] Peize Li, Qingyi Si, Peng Fu, Zheng Lin, and Yan Wang. Object attribute matters in visual question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 18545–18553, 2024. 2
- [37] Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. Tgif: A new dataset and benchmark on animated gif description. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4641–4650, 2016. 1
- [38] Yunheng Li, Jing Cheng, Shaoyong Jia, Hangyi Kuang, Shaohui Jiao, Qibin Hou, and Ming-Ming Cheng. Tempsamp-r1: Effective temporal sampling with reinforcement fine-tuning for video llms. arXiv preprint arXiv:2509.18056, 2025. 1
- [39] Long Lian, Yifan Ding, Yunhao Ge, Sifei Liu, Hanzi Mao, Boyi Li, Marco Pavone, Ming-Yu Liu, Trevor Darrell, Adam Yala, et al. Describe anything: Detailed localized image and video captioning. arXiv preprint arXiv:2504.16072, 2025. 5, 6
- [40] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, pages 26689–26699, 2024. 3
- [41] Weifeng Lin, Xinyu Wei, Ruichuan An, Tianhe Ren, Tingwei Chen, Renrui Zhang, Ziyu Guo, Wentao Zhang, Lei Zhang, and Hongsheng Li. Perceive anything: Recognize, explain, caption, and segment anything in images and videos. arXiv preprint arXiv:2506.05302, 2025. 1, 5
- [42] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 3
- [43] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. 2023. 2
- [44] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26296–26306, 2024. 3
- [45] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatialtemporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 1
- [46] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. arXiv preprint arXiv:2402.06196, 2024. 3

- [47] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating videobased large language models. Computational Visual Media,

2025. 3

- [48] OpenAI. ChatGPT, 2023. 1
- [49] OpenAI. Gpt-4o system card, 2024. 2, 5, 6
- [50] Wujian Peng, Lingchen Meng, Yitong Chen, Yiweng Xie, Yang Liu, Tao Gui, Hang Xu, Xipeng Qiu, Zuxuan Wu, and Yu-Gang Jiang. Inst-it: Boosting multimodal instance understanding via explicit visual prompt instruction tuning. arXiv preprint arXiv:2412.03565, 2024. 1, 2, 5
- [51] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 5
- [52] Jianing Qi, Jiawei Liu, Hao Tang, and Zhigang Zhu. Beyond semantics: Rediscovering spatial awareness in visionlanguage models. arXiv preprint arXiv:2503.17349, 2025. 2
- [53] Jihao Qiu, Yuan Zhang, Xi Tang, Lingxi Xie, Tianren Ma, Pengyu Yan, David Doermann, Qixiang Ye, and Yunjie Tian. Artemis: Towards referential understanding in complex videos. Advances in Neural Information Processing Systems, 37:114321–114347, 2024. 5
- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2
- [55] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In CVPR, pages 13009–13018, 2024. 3
- [56] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. Advances in Neural Information Processing Systems, 36:3536–3559, 2023. 2
- [57] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv:2410.17434, 2024. 5
- [58] Boyuan Sun, Modi Jin, Bowen Yin, and Qibin Hou. Depth anything at any condition. arXiv preprint arXiv:2507.01634, 2025. 3
- [59] Boyuan Sun, Jiaxing Zhao, Xihan Wei, and Qibin Hou. Llava-scissor: Token compression with semantic connected components for video llms. arXiv preprint arXiv:2506.21862, 2025. 3
- [60] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin,

- Rongyi Zhu, Ali Vosoughi, Chao Huang, Zeliang Zhang, Pinxin Liu, Mingqian Feng, Feng Zheng, Jianguo Zhang, Ping Luo, Jiebo Luo, and Chenliang Xu. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025. 1
- [61] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. 3
- [62] Qwen team. Qwen2-vl. 2024. 1, 5
- [63] Qwen Team. Qwen2.5: A party of foundation models,

2024. 4

- [64] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Chatterbox: Multi-round multimodal referring and grounding. arXiv preprint arXiv:2401.13307, 2024. 3
- [65] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 3
- [66] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024. 1
- [67] Han Wang, Yongjie Ye, Yanjie Wang, Yuxiang Nie, and Can Huang. Elysium: Exploring object-level perception in videos via mllm. In European Conference on Computer Vision, pages 166–185. Springer, 2024. 5
- [68] Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive visual instruction tuning. arXiv preprint arXiv:2410.09575, 2024. 3
- [69] Hao Wang, Limeng Qiao, Zequn Jie, Zhijian Huang, Chengjian Feng, Qingfang Zheng, Lin Ma, Xiangyuan Lan, and Xiaodan Liang. X-sam: From segment anything to any segmentation. arXiv preprint arXiv:2508.04655, 2025. 3
- [70] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing opensource multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1
- [71] Yufei Wang, Wanjun Zhong, Liangyou Li, Fei Mi, Xingshan Zeng, Wenyong Huang, Lifeng Shang, Xin Jiang, and Qun Liu. Aligning large language models with human: A survey. arXiv preprint arXiv:2307.12966, 2023. 3
- [72] Yuxuan Wang, Cihang Xie, Yang Liu, and Zilong Zheng. Videollamb: Long video understanding with recurrent memory bridges. arxiv, 2024. 3
- [73] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747,

2025. 3

- [74] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang,

- et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 5
- [75] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025. 1
- [76] Shiyu Xuan, Qingpei Guo, Ming Yang, and Shiliang Zhang. Pink: Unveiling the power of referential comprehension for multi-modal llms. In CVPR, pages 13838–13848, 2024. 3
- [77] An Yan, Zhengyuan Yang, Junda Wu, Wanrong Zhu, Jianwei Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Julian McAuley, Jianfeng Gao, et al. List items one by one: A new data source and learning paradigm for multimodal llms. arXiv preprint arXiv:2404.16375, 2024. 3
- [78] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [79] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1
- [80] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 3
- [81] Qize Yang, Shimin Yao, Weixuan Chen, Shenghao Fu, Detao Bai, Jiaxing Zhao, Boyuan Sun, Bowen Yin, Xihan Wei, and Jingren Zhou. Humanomniv2: From understanding to omni-modal reasoning with context. arXiv preprint arXiv:2506.21277, 2025. 2
- [82] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18155–18165, 2022. 2
- [83] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive languageimage pre-training. arXiv preprint arXiv:2111.07783,

2021. 2

- [84] Heeji Yoon, Jaewoo Jung, Junwan Kim, Hyungyu Choi, Heeseong Shin, Sangbeom Lim, Honggyu An, Chaehyun Kim, Jisang Han, Donghyun Kim, et al. Visual representation alignment for multimodal large language models. arXiv preprint arXiv:2509.07979, 2025. 3

- [85] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023. 1, 3, 5
- [86] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. In ECCV, pages 425–443. Springer, 2025. 3
- [87] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, pages 9127–9134, 2019. 5, 6
- [88] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28202–28211, 2024. 3, 5
- [89] Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, et al. Videorefer suite: Advancing spatialtemporal object understanding with video llm. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18970–18980, 2025. 1, 2, 3, 5, 6
- [90] Yuqian Yuan, Wenqiao Zhang, Xin Li, Shihao Wang, Kehan Li, Wentong Li, Jun Xiao, Lei Zhang, and Beng Chin Ooi. Pixelrefer: A unified framework for spatio-temporal object referring with arbitrary granularity. arXiv, 2025. 1
- [91] Tongtian Yue, Jie Cheng, Longteng Guo, Xingyuan Dai, Zijia Zhao, Xingjian He, Gang Xiong, Yisheng Lv, and Jing Liu. Sc-tune: Unleashing self-consistent referential comprehension in large vision language models. In CVPR, pages 13073–13083, 2024. 3
- [92] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why visionlanguage models behave like bags-of-words, and what to do about it? arXiv preprint arXiv:2210.01936, 2022. 1
- [93] Quan-Sheng Zeng, Yunheng Li, Qilong Wang, Peng-Tao Jiang, Zuxuan Wu, Ming-Ming Cheng, and Qibin Hou. A glimpse to compress: Dynamic visual token pruning for large vision-language models. arXiv preprint arXiv:2508.01548, 2025. 2
- [94] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 4
- [95] Yufei Zhan, Yousong Zhu, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon v2: Advancing multimodal perception with high-resolution scaling and visuallanguage co-referring. arXiv preprint arXiv:2403.09333,

2024. 3

- [96] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 3

- [97] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 3
- [98] Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, et al. Ferretv2: An improved baseline for referring and grounding with large language models. arXiv preprint arXiv:2404.07973,

2024. 3

- [99] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems, 36:45533–45547,

2023. 3

- [100] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 3, 5
- [101] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. In European conference on computer vision, pages 52–70. Springer, 2024. 1, 3
- [102] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, and Shuicheng Yan. Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding. Advances in neural information processing systems, 37:71737–71767, 2024. 3
- [103] Tao Zhang, Xiangtai Li, Zilong Huang, Yanwei Li, Weixian Lei, Xueqing Deng, Shihao Chen, Shunping Ji, and Jiashi Feng. Pixel-sail: Single transformer for pixel-grounded understanding. arXiv preprint arXiv:2504.10465, 2025. 1, 3
- [104] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model,

2024. 5

- [105] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1, 5
- [106] Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. Cross-modal information flow in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19781–19791,

2025. 3

- [107] Jiaxing Zhao, Boyuan Sun, Xiang Chen, and Xihan Wei. Facial dynamics in video: Instruction tuning for improved facial expression perception and contextual awareness. arXiv preprint arXiv:2501.07978, 2025. 3
- [108] Jiaxing Zhao, Boyuan Sun, Xiang Chen, Xihan Wei, and Qibin Hou. Llava-octopus: Unlocking instruction-driven adaptive projector fusion for video understanding. arXiv preprint arXiv:2501.05067, 2025. 5
- [109] Jiaxing Zhao, Qize Yang, Yixing Peng, Detao Bai, Shimin Yao, Boyuan Sun, Xiang Chen, Shenghao Fu, Xihan Wei,

- Liefeng Bo, et al. Humanomni: A large vision-speech language model for human-centric video understanding. arXiv preprint arXiv:2501.15111, 2025. 1
- [110] Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, et al. Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning. arXiv preprint arXiv:2307.09474, 2023. 3
- [111] Xiangyu Zhao, Xiangtai Li, Haodong Duan, Haian Huang, Yining Li, Kai Chen, and Hua Yang. Mg-llava: Towards multi-granularity visual instruction tuning. arXiv preprint arXiv:2406.17770, 2024. 3
- [112] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 1

Table 7. GamePoint@K comparison.

###### Method GamePoint@K-1 GamePoint@K-5 GamePoint@K-10 GamePoint@K-50 GamePoint@K-100

Qwen2.5-VL-7B 0.330 0.328 0.331 0.330 0.329 SWIM 0.373 0.375 0.374 0.373 0.374

Appendix

- A. Benchmarks

For completeness, we provide detailed descriptions of the general benchmarks used in the main paper.

ActivityNet-QA [3] is a large-scale video question answering benchmark constructed from the ActivityNet dataset. It contains human-annotated question–answer pairs focusing on action-related content, with an average video duration of about 2 minutes. The questions are designed to require understanding of dynamic scenes and temporal sequences rather than static visual cues.

VideoMME [20] collects videos from a wide range of domains, including sports, documentaries, instructional content, and entertainment. Video durations vary from minutes to hours, making it one of the most comprehensive and challenging benchmarks for holistic video understanding. The diversity in topic, style, and duration tests a model’s ability to handle long-context reasoning and adapt to varied visual–text scenarios.

MVBench [35] is a multi-choice video understanding benchmark comprising 20 distinct tasks. Each task presents a multiple-choice question targeting temporal comprehension, covering scenarios such as event ordering, cause–effect reasoning, motion tracking, and activity prediction. These tasks require sophisticated temporal reasoning and understanding of dynamic content that cannot be solved by analyzing a single frame, thereby evaluating a model’s capability to integrate information across time.

Together, these benchmarks provide a diverse evaluation landscape: ActivityNet-QA and VideoMME emphasize broad video understanding with varying domain coverage and length, whereas MVBench focuses on fine-grained temporal reasoning across multiple types of challenges.

- B. More Experimental Analysis B.1. GamePoint@K

We further evaluate retrieval accuracy using GamePoint@K, which measures the fraction of relevant elements among the top-K highest-scoring positions in the attention map:

N

|TopK(A¯ i,K) ∩ Pi| |TopK(A¯ i,K)|

1 N

, (9)

GamePoint@K =

i=1

where TopK(A¯ i,K) selects the K highest-scoring elements for sample i, and Pi denotes its ground-truth posi-

tions. Higher GamePoint@K scores indicate that relevant visual targets are ranked closer to the top, reflecting better alignment between textual references and visual regions.

As shown in Table 7, SWIM consistently outperforms Qwen2.5-VL across all K values. At K = 1, SWIM achieves 0.373 compared to 0.330 for Qwen2.5-VL, indicating stronger ability to position the correct target at the top rank. This advantage is maintained at broader retrieval depths, with SWIM reaching 0.375 at K = 5 (+4.7% over baseline) and retaining stable gains for K = 10, K = 50, and K = 100. The consistent margins across different K suggest that SWIM produces reliable ranking distributions, keeping relevant objects prominent even as the retrieval list expands.

##### B.2. Robustnessto Synonym-basedLinguistic Noise

To assess the robustness of SWIM to variations in referring expressions, we conduct an evaluation in which words enclosed in <ins> tags within the VideoRefer-Bench-D prompts are replaced by semantically equivalent synonyms. This modification leaves the overall meaning unchanged but alters the surface form of the text, introducing lexical noise that may challenge models relying on exact token matches. Such a setting reflects real-world scenarios where object references may vary due to differences in speaker style, domain-specific terminology, or translation artifacts, and tests whether a model can preserve grounding accuracy under these conditions. As shown in Table 8, the original SWIM achieves an average score of 3.78, while SWIM∗ obtains 3.74 under synonym perturbations—a marginal drop of 0.04. Compared to Qwen2.5VL, SWIM maintains strong performance under synonym substitutions, achieving an average accuracy of 3.74 against 3.43. These results indicate that SWIM’s alignment mechanism is resilient to changes in word choice, preserving its ability to ground natural language object references to the correct visual regions even under lexical variation.

Table 8. Performance comparisons on VideoRefer-Bench-D. ∗ denotes incorporating synonym-based noise.

###### VideoRefer-Bench-D

Method SC AD TD HD Avg. Qwen2.5-VL-7B [2] 3.99 3.05 2.44 2.44 2.97 Qwen2.5-VL-7B∗ [2] 4.78 3.49 3.27 2.18 3.43 SWIM 4.92 3.85 3.43 2.96 3.78 SWIM∗ 4.86 3.78 3.36 2.96 3.74

