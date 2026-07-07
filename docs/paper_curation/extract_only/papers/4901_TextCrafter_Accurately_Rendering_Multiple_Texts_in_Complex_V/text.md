# arXiv:2503.23461v7[cs.CV]27Feb2026

## Investigating Text Insulation and Attention Mechanisms for Complex Visual Text Generation

Ying Tai1†, Nikai Du1†, Rui Xie1*, Zhennan Chen1*, Qian Wang2, Zhengkai Jiang3, Kai Zhang1, Jian Yang1

1School of Intelligence Science and Technology, Nanjing University. 2Jiutian Research. 3The Hong Kong University of Science and Technology.

*Corresponding author(s). E-mail(s): ruixie0097@gmail.com; znchen@smail.nju.edu.cn; Contributing authors: yingtai@nju.edu.cn; 502024710004@smail.nju.edu.cn; †These authors contributed equally to this work.

Abstract In this paper, we present TextCrafter, a Complex Visual Text Generation (CVTG) framework inspired by selective visual attention in cognitive science, and introduce the “Text Insulation-and-Attention” mechanisms. To implement the selective-attention principle that selection operates on discrete objects, we propose a novel Bottleneck-aware Constrained Reinforcement Learning for Multi-text Insulation, which substantially improves text-rendering performance on the strong Qwen-Image pretrained model without introducing additional parameters. To align with the selective concentration principle in human vision, we introduce a text-oriented attention module with a novel Quotation-guided Attention Gate that further improves generation quality for each text instance. Our Reinforcement Learning based text insulation approach attains state-of-the-art results, and incorporating text-oriented attention yields additional gains on top of an already strong baseline. More importantly, we introduce CVTG-2K, a benchmark comprising 2,000 complex visual-text prompts. These prompts vary in positions, quantities, lengths, and attributes, and span diverse real-world scenarios. Extensive evaluations on CVTG-2K, CVTG-Hard, LongText-Bench, and Geneval datasets confirm the effectiveness of TextCrafter. Despite using substantially fewer resources (i.e., 4 GPUs) than industrial-scale models (e.g., Qwen-Image, GPT Image, and Seedream), TextCrafter achieves superior performance in mitigating text misgeneration, omissions, and hallucinations.

Keywords: Text-to-Image Generation, Diffusion Models, Text Rendering

### 1 Introduction

###### Diffusion models [1–12] have emerged as the forefront technology in image generation. Recent

Project: https://github.com/NJU-PCALab/TextCrafter

advances, represented by the general and powerful text-to-image models such as FLUX [13] and SD3 [14], have demonstrated the ability to render simple text through large-scale pretraining. However, when confronted with complex real-world visual text scenarios (e.g., multiple texts), these models often struggle with critical challenges,

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Fig. 1 TextCrafter enables precise complex visual text rendering, supports the generation of different languages, and optimizes key challenges such as text misgeneration, omission, and hallucination.

including text misgeneration, omission, and hallucination. While the complexity of scene text has been extensively categorized in the recognition literature [15], effectively generating such intricate structures remains an open challenge. These shortcomings significantly hinder their practical applicability in real-world scenarios.

For complex scenarios, recent studies [16–19] primarily focus on generating multiple instances within a single image simultaneously, aiming to precisely control factors such as quantity, position, and attributes. However, a crucial yet often overlooked category in the visual domain is visual text. Text is an indispensable component of the real world and is markedly more intricate and delicate than conventional objects (e.g., plants or animals). Even slight perturbations in its finegrained structure can dramatically alter its visual appearance, leading to misrecognition or complete illegibility [20]. Existing multi-instance generation methods, such as MIGC [17] and 3DIS [19], mainly operate at the sentence level, offering limited understanding of text at finer granularities. Moreover, RPG [21] and RAG-Diffusion [18] adjust

instance structures during attention fusion but lack precision required for visual text rendering.

Recent work on visual text rendering has largely focused on accurately generating a single text region. For instance, methods like AnyText [22], Glyph-byT5 [23], SceneTextGen [24], Diff-Text [25], and TextDiffuser [26] often employ specialized modules to encode visual text representations and condition the diffusion model accordingly. However, these methods face two major limitations: 1) They rely on predefined rules to synthesize visual text training data. Because highly accurate text annotations are required, manual verification is often needed, making the construction of complex visual text datasets both labor-intensive and costly. 2) Most approaches depend on fine-tuned text encoders (e.g., Glyphby-T5) or trained conditional control encoders (e.g., AnyText, TextDiffuser, SceneTextGen) to facilitate text generation. However, in complex multi-text generation, these approaches often suffer from interference among the control signals of different targets, making it difficult to balance fine-grained local control with global consistency across multiple texts.

Table 1 Comparison of CVTG-2K with existing public visual text Benchmarks in terms of sample size (Num), average word count (Word), average character count (Char), Attribute, and number of regions (Region). ‘EN’ and ‘ZH’ denote English and Chinese, respectively. Note that ‘Word’ and ‘Char’ are calculated based only on the English subset of the benchmarks. The symbol ‘-’ indicates that the benchmark does not explicitly model attributes or the number of regions.

Benchmark Num Word Char Language Attribute Region

CreativeBench 400 1.00 7.29 EN - MARIOEval 5,414 2.92 15.47 EN - DrawTextExt 220 3.75 17.01 EN - AnyText-benchmark 2,000 4.18 21.84 ZH&EN - LongText-Bench 320 26.98 158.09 ZH&EN - -

CVTG-2K (Ours) 2,000 8.10 39.47 EN ✓ (size/color/font) ✓ (2/3/4/5) CVTG-Hard (Ours) 400 8.61 40.79 ZH&EN ✓ (size/color/font) ✓ (2/3/4/5)

Inspired by selective visual attention in cognitive science [27, 28] that enhances relevant signals and manages distraction, we frame complex visual text generation as a capacity-limited problem where multiple text instances compete for accurate representation, leading to the issues of feature leakage. In particular, the object-based theory of selective visual attention [29] posits that selection operates on discrete objects (i.e., attention deals with only one object at a time), mitigating crossobject interference. Analogously, generative models often render one target correctly while omitting or corrupting others due to inter-object interference. To address this, we propose a novel method TextCrafter that explores the “Text Insulation and Attention” mechanisms for Complex Visual Text Generation (CVTG). Specifically, 1) Text Insulation: To implement the selectiveattention principle that selection operates on discrete objects, TextCrafter insulates each text as an independent object. A novel Bottleneckaware Constrained Reinforcement Learning for Multi-text Insulation is proposed, which incorporates an OCR-based reward model during post-training and leverages reinforcement learning to explicitly optimize the fidelity of each text instance. We introduce a bottleneck-sensitive aggregation term that explicitly emphasizes the worst-case instance performance in the reward model. 2) Text-oriented Attention: Moreover, to align with the selective enhancement principle in selective attention, TextCrafter introduces a text-oriented attention module that employs the proposed Quotation-guided Attention Gate. We observe that some quotation marks serve as robust spatial anchors, which can be converted as the precise spatial gates through a sequence of smoothing, primary peak retention, and soft

binarization operations. These text-centric formulations operationalize the principles of selective visual attention to improve robustness and overall quality for multi-text rendering. Our Reinforcement Learning based text insulation approach attains state-of-the-art results, and incorporating text-oriented attention yields additional gains on top of an already strong baseline. Visual examples can be found in Figure 1.

On the other hand, we introduce CVTG-2K, a dedicated benchmark specifically tailored to the CVTG task. Unlike previous datasets that predominantly focus on single-region or fixedtemplate scenarios, CVTG-2K comprises 2,000 high-quality prompts featuring diverse region quantities (ranging from 2 to 5) and rich visual attributes (e.g., color, font, and size). As shown in Table 1, with an average text length of 8.10 words and 39.47 characters, it significantly surpasses existing benchmarks in complexity, offering a rigorous testbed for evaluating robustness in realistic multi-text environments. In general, our contributions are summarized as follows:

- • We propose TextCrafter, a novel framework that introduces the “Text Insulation and Attention” mechanisms, incorporating text insulation and text-oriented attention to suppress crosstext interference and enable precise multi-text rendering.
- • We construct CVTG-2K, a benchmark of 2,000 complex visual-text prompts covering diverse positions, quantities, lengths, and attributes, providing a rigorous legibility- and completeness-oriented testbed for CVTG task.
- • We conduct extensive quantitative and qualitative experiments on CVTG-2K and LongText-Bench, demonstrating the superiority of TextCrafter over strong competitors from both industry and academia.

[Figure 6]

- Fig. 2 Comparison of different paradigms for visual text generation. Left: General pretrained models (e.g., Qwen-Image [30]) lack specific architectural designs for precise text rendering. Middle: Glyph-based control methods (e.g., AnyText [22], GlyphControl [31]) introduce an additional ControlNet [32] branch and require pre-rendered glyph images as conditions, increasing both structural and input complexity. Right: Our approach presents the Text Insulation and Attention mechanisms, where the trainable components are implemented as a lightweight LoRA [33] module. It preserves the original architecture and general capabilities of the base model while significantly enhancing text rendering performance without the need for auxiliary control branches or glyph inputs.

### 2 Related Works

Multi-instance Generation. Attend-andExcite [34] and BOX-Diffusion [35] guide pre-trained diffusion models to align generated instances with prompts. Recent work has further explored attribute-centric approaches to mitigate the overfitting of common compositions and improve the generation of underrepresented attribute combinations [36]. GLIGEN [37], MIGC [17], and CreativeLayout [38] enhance control by incorporating bounding box conditions into trainable layers. RPG [21], RAG-Diffusion [18] and DreamRenderer [39] break the generation process into regional tasks for compositional generation. However, these methods often overlook a critical kind of instance: visual text, a key component of visual scenes.

Visual Text Generation. To achieve precise structural control, methods like AnyText [22] and GlyphControl [31] introduce an additional ControlNet [32] branch to inject pre-rendered glyph images. As shown in Figure 2 (Middle), such methods substantially increase model complexity and reliance on additional inputs. Meanwhile, DiffSTE [40], Glyph-ByT5 [23], and UDiffText [41] use character-level encoders to incorporate word appearance into embeddings. TextDiffuser [26] creates text layout masks for injection into the latent space. Approaches like Diff-Text [25] improve text accuracy with attention map restrictions. GlyphDraw [42] uses two encoders for text position prediction and rendering. TextDiffuser-2 [43] leverages LLMs for layout planning. More recently, recent industrial foundation models prioritize massive scaling. QwenImage [30] explores the scalability of Diffusion Transformers (DiT) with 20B parameters, while

GLM-Image [44] adopts a hybrid architecture combining an autoregressive transformer with a DiT-based decoder. However, as illustrated in Figure 2 (Left), these general-purpose models lack specific architectural designs for text rendering. In contrast, TextCrafter introduces the novel “Text Insulation and Attention” mechanisms, with the trainable components realized as a lightweight LoRA module requiring only low-cost training resources (see Figure 2 (Right)).

Benchmark for Visual Text Generation. Earlier works like GlyphControl introduce SimpleBench and CreativeBench [31], but these fixedtemplate datasets offer limited diversity and contain only single-word scenes. MARIO-Eval [26] suffers from low-quality prompts and unclear semantics. Although DrawTextExt [42] collects prompts for generating natural scenes, the comprehensiveness and scale of the dataset are insufficient. AnyText-benchmark [22] allows multi-word prompts but treats each word separately and appends them directly to the caption, limiting data distribution. As shown in Table 1, existing visual text benchmarks focus on text generation within a single object or location, failing to capture the complexity of real-world scenes. In contrast, we introduce CVTG-2K, a robust benchmark that incorporates diverse positions, quantities, lengths, and attributes to comprehensively evaluate models in complex visual text scenarios.

### 3 Methodology

###### 3.1 Challenges in CVTG

In CVTG, users provide a global prompt P containing multiple visual text descriptions D = {d1,d2,...,dn}, where each description includes

- 1) Text Misgeneration: Sale In a fashion store, a window display reads 'Trendy', a shopping bag with 'Sale'.

2)Text Omission: Drive A mountain road with a large billboard reading 'Adventure'. A car rear window displays the text 'Drive'.

[Figure 7]

| |
|---|

| |
|---|

[Figure 8]

| |
|---|

3)Text Hallucination: Indicated by yellow bounding boxes At a lively amusement park featuring a roller coaster sign with 'Thrill Seekers Welcome', a ticket booth announces 'Buy Tickets Here', and a souvenir shop window displays 'Memories to Take Home'.

[Figure 9]

Fig. 3 Challenges in CVTG. 1) Text Misgeneration: Existing model [13] fails to render “Sale” correctly, generating erroneous characters. 2) Text Omission: The required text “Drive” is missing from the rear window. 3) Text Hallucination: The ticket booth exhibits a hallucinated “HEE”, and souvenir shop’s signboard and windows are cluttered with extensive illegible gibberish.

the visual text’s content and descriptors (position, attributes). The visual texts’ content is defined as V T = {vt1,vt2,...,vtn}, with each vti corresponding to description di. The model generates an image from prompt P where each vti appears with its corresponding description di. Complex prompts cause three kinds of degradation: 1) Text Misgeneration: Visual texts vti and vtj intertwine, generating duplicate or missing characters.

- 2) Text Omission: Only text from description di appears, neglecting vtj from dj. 3) Text Hallucination: The model generates unrequested textual artifacts, manifesting as redundant repetitions of target text or unintelligible gibberish in regions not specified by the V T. Figure 3 visualizes these representative failure cases.

Prompt Sampling

Policy Optimization

[Figure 10]

T2I Model RL Algorithm

Prompt：A cozy coffee shop interior with a chalkboard reading 'Brew', a mug with 'Love' written on it, a bag of beans labeled 'Fresh', and a poster on the wall saying 'Relax'.

Samples & Prompt

###### Bottleneck-aware Constrained Reinforcement Learning for Multi-text Insulation

Joint Attention Layer in MMDiT Blocks

- Step 1: Target Extraction & Instance Preprocessing
- Step 2: Isolated Fuzzy Matching
- Step 3: Insulation-aware Aggregation
- Step 4: Anti-interference Penalty

Key

Prompt Sample OCR Model

𝑉𝑉𝑇𝑇 = 𝑣𝑣𝑡𝑡𝑡,𝑣𝑣𝑡𝑡2,⋯,𝑣𝑣𝑡𝑡𝑛𝑛

Normalization

Query

Brew Love Fresh Relax

𝒪𝒪

|Text-to-Text Attention|Text-to-Image Attention|
|---|---|
|<br><br><br><br><br><br>|Image-to-Image Attention|

dist𝑣𝑣𝑡𝑡𝑖𝑖,𝑤𝑤

| | |⋯ 𝒪𝒪|
|---|---|---|

𝑠𝑠𝑖𝑖 = max

1 −

𝑣𝑣𝑡𝑡𝑖𝑖

𝑤𝑤⊆𝒪𝒪, 𝑤𝑤 = 𝑣𝑣𝑡𝑡𝑡𝑡

Sliding window: 𝑤𝑤

Quotation-guided

AttentionGate

[Figure 16]

Attention Modulation

𝑛𝑛

1 𝑛𝑛

𝑅𝑅𝑏𝑏𝑏𝑏𝑏𝑏𝑏𝑏 = 1 − 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅

𝑠𝑠𝑖𝑖 + 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅ min 𝑠𝑠𝑡,…,𝑠𝑠𝑛𝑛

𝑖𝑖=𝑡

Noised Latent

Decoder

Image-to-Text Attention

Denoised Latent

𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 = 1.0

× N

Y

| | |
|---|---|
| | |

|Yes| |
|---|---|
|No| |

Ratio: 𝑏𝑏𝑏𝑏𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝

𝑅𝑅𝑛𝑛𝑜𝑜𝑜𝑜 = 𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 𝑅𝑅𝑏𝑏𝑏𝑏𝑏𝑏𝑏𝑏

≤ 𝛿𝛿

𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 = Ratio−𝛿𝛿+𝑡𝑡

𝑡𝑡𝑡𝑡𝑝𝑝𝑡𝑡𝑝𝑝𝑡𝑡

###### 3.2 Framework

CVTG requires rendering multiple textual contents that interact through layout, scale, and style. TextCrafter leverages a text insulation module to mitigate cross-text interference, and a textoriented attention mechanism to enhance complex visual text generation. The latter employs a Quotation-guided Attention Gate to dynamically modulate the attention of text tokens, enforcing their concentration within the designated region defined by anchor quotation marks.

Fig. 4 Bottleneck-aware Constrained Reinforcement Learning for Multi-text Insulation: (1) Target Extraction & Instance Preprocessing: Normalizing prompt strings and OCR outputs. (2) Isolated Fuzzy Matching: Calculating similarity scores si using a sliding window to measure independent instance accuracy. (3) Insulation-aware Aggregation: Balancing average performance with a bottleneck-sensitive min(·) term to prevent the omission of any single text target. (4) Antiinterference Penalty: Applying a length-based decay λnoise to suppress over-generation and hallucinations.

[Figure 17]

[Figure 18]

T2I Model Text-orientedAttention

T2I Model

Prompt T2I Model

[Figure 21]

Prompt

Prompt

LoRA

|[Figure 23]|
|---|

[Figure 24]

Glyph ControlNet

Reward Function of Text Insulation: 𝑅𝑅𝑛𝑛𝑜𝑜𝑜𝑜 = 𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 [ 1 − 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅

Glyph-based Control (e.g. AnyText)

𝑛𝑛

Pretrained Model (e.g. Qwen-Image)

1 𝑛𝑛

Ours

𝑠𝑠𝑖𝑖 + 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅ min 𝑠𝑠𝑡,…, 𝑠𝑠𝑛𝑛 ]

Rendered Glyph

𝑖𝑖=𝑡

model such as Qwen-Image would further validate the effectiveness and generality of our text insulation concept. To demonstrate the validity of text insulation on Qwen-Image, we propose a Bottleneck-aware Constrained Reinforcement Learning for Multi-text Insulation. Specifically,

###### 3.2.1 Text Insulation

Bottleneck-aware Constrained Reinforcement Learning for Multi-text Insulation. Qwen-Image [30] is one of the most powerful open-sourced image generation model. Successfully adapting our approach to a state-of-the-art

[Figure 25]

- Fig. 5 Visualization of Phrase-Level Cross-Attention Maps. We visualize the aggregated attention maps for representative target phrases, specifically ‘Fashion’ and ‘New Year Special’, across different scenes. Left (Before RL): The baseline model suffers from significant attention leakage, where attention drifts to visually salient but unrelated regions (e.g., ‘New Year Special’ erroneously focuses on the central lantern, and ‘Fashion’ bleeds into surrounding signage). Right (After RL): After post-training, the attention maps become notably spatially disentangled and concentrated exclusively on their corresponding text regions. This confirms that our method effectively suppresses cross-text interference at the feature level. Red bounding boxes denote the manually annotated ground-truth phrase-level regions.

to operationalize the principle of Multi-text Insulation, we design a novel reward function Rocr to incorporate an OCR-based reward model during post-training, which is engineered to optimize the fidelity of each text instance while preventing inter-text interference. As shown in Fiugre 4, the reward follows a four-step pipeline:

- 1) Target Extraction & Instance Preprocessing. Given a multi-text prompt, we treat each target string vti as an isolated entity in the ground truth set V T = {vt1,vt2,...,vtn}. Both V T and the OCR-detected [45] results from the generated image are normalized (lowercase conversion and removal of non-alphanumeric characters) to ensure that the insulation performance is measured purely on semantic accuracy rather than formatting noise.
- 2) Isolated Fuzzy Matching. We utilize the Fuzzy Partial Ratio metric to compute an independent

similarity score si ∈ [0,1] for each target vti. Formally, given the global OCR output sequence O and a target string vti, the partial ratio score is defined as the maximum Levenshtein similarity [46] between vti and any substring w extracted from O that has the same length as vti:

si = max

w⊆O,|w|=|vti|

dist(vti,w) |vti|

1 −

, (1)

where dist(·,·) denotes the Levenshtein edit distance. By matching each target individually against the global OCR output in sliding-window

manner, we precisely monitor the “insulation” quality of each text instance, ensuring that artistic variations or minor recognition errors in one area do not unfairly penalize other text regions.

- 3) Insulation-aware Aggregation. To prevent the model from collapsing by generating one text while neglecting others (insulation failure), we

define the base reward Rbase to balance average performance and individual integrity:

Rbase = (1−λbal)·

1 n

n

i=1

si+λbal·min(s1,...,sn),

(2) where λbal (empirically set to 0.3) is a balancing coefficient that controls the sensitivity to the worst-case instance (i.e., the bottleneck). The min term is critical for multi-text insulation, as it explicitly penalizes the omission or corruption of any single text target, forcing the model to “insulate” and preserve all requested instances.

- 4) Anti-interference Penalty. A common pathology in reinforcement learning for text generation is the “text explosion” phenomenon, where the model generates excessive irrelevant text or repetitions to maximize the recall probability (i.e., reward hacking). To mitigate this, we introduce a

length-based noise penalty λnoise. Specifically, we define a tolerance threshold δ for the length ratio. If the total predicted length lpred exceeds the sum of insulated targets ltarget by this threshold, the reward is decayed:

[Figure 26]

- Fig. 6 Quantitative evaluation on a subset (30 samples) of CVTG-Hard. Left: The Effective Attention Efficiency (η) analysis. The values displayed in the legend represent the mean η scores, which increase from 0.3126 to 0.4523, corresponding to a 44.7% relative improvement in attention quality. Right: The Recall metric increases by 6.2 pts, verifying the method’s effectiveness in handling complex character structures.

λnoise =

1.0, if llpred

≤ δ,

target

1

(lpred/ltarget)−δ+1, otherwise.

(3)

We empirically set δ = 1.5 to accommodate minor OCR redundancies while strictly penalizing excessive gibberish. The final reward is formulated as Rocr = λnoise · Rbase. In this work, we employ the DiffusionNFT [47] to optimize this objective. However, it is worth noting that our reward design is algorithm-agnostic and can be seamlessly adapted to other advanced reinforcement learning frameworks [48, 49]. The proposed penalty ensures the model generates the required text in a clean, insulated manner.

Functionally, these two components establish a dual-constraint mechanism on the generation length. The Insulation-aware Aggregation acts as a lower bound constraint, encouraging the model to generate sufficient tokens to cover all targets (preventing omission), while the Anti-interference Penalty serves as an upper bound constraint, suppressing the tendency to over-generate (preventing hallucination). This “push-pull” dynamic forces the model to converge on the optimal, insulated text layout.

To further demonstrate the effectiveness of our insulated RL strategy, we provide qualitative

evidence by analyzing the aggregated phraselevel cross-attention maps across different scenes. As illustrated in Figure 5, the baseline model often suffers from feature leakage, where the activation regions of text targets drift to visually salient but unrelated areas (e.g., the attention for ‘New Year Special’ erroneously focusing on the central lantern, or ‘Fashion’ bleeding into surrounding signage). Conversely, after RL finetuning, the attention maps for target phrases become notably disentangled and spatially concentrated. This observation corroborates that our reward formulation successfully guides the model to allocate exclusive latent regions for each text target, effectively preventing “text explosion” or overlap phenomena at the feature level.

Discussion on RL for Multi-Text Generation. We randomly select 30 prompts from the CVTG-Hard and conduct a comparative evaluation between the baseline model and our RLfinetuned model. During inference, we selectively extract the cross-attention maps corresponding to the tokens of the target text. Subsequently, for each target instance enclosed in quotation marks, we aggregate the maps of its constituent tokens to derive a phrase-level attention map, denoted as Ap. We introduce two metrics to reflect the omission and hallucination degree, i.e. Recall and Effective Attention Energy (η), respectively.

Attention Map Smoothing Primary Peak Retention Normalization & Soft Binarization

Generated Image

#### Gate Construction Pipeline

[Figure 32]

| |
|---|

| |
|---|

- Fig. 7 Visualization of prompt tokenization for “On the table, a note that says ‘TextCrafter’. A coffee cup with the word ‘IJCV’ on it.” along with per-token attention maps. In Qwen-Image, each closing quotation mark functions as a spatial anchor, enforcing alignment between text token and its corresponding carrier token and yielding clean spatial disentanglement without cross-text interference.

- 1) Recall. To quantify the degree of text omission, we calculate the object-level generation success rate. Let Ntotal be the total number of target phrases in the evaluation set, and Nsucc be the count of generated phrases that are successfully detected and matched with valid bounding boxes. The Recall is defined as:

Recall =

Nsucc Ntotal × 100%. (4)

A higher Recall indicates a lower omission rate and robust instruction following.

- 2) Effective Attention Efficiency (η). To measure the degree of visual hallucination (i.e., attention leakage), we propose η to evaluate the concentration of effective attention energy. As illustrated in Figure 5, we construct the GroundTruth (GT) phrase-level regions using manually annotated red bounding boxes. The core intuition of this metric is to maximize the effective attention energy accumulated within the GT box while minimizing the energy leaking into the outside background. Given the phrase-level attention map Ap, we first apply an adaptive statistical threshold to filter out background noise. Let µ(Ap) and

σ(Ap) denote the mean and standard deviation of the map, respectively. We obtain a denoised map Aˆ p by retaining only the signals exceeding the mean plus one standard deviation:

Aˆ pi,j =

Api,j, if Api,j > µ(Ap) + σ(Ap) 0, otherwise.

(5)

Then, η is calculated as the ratio of effective energy inside the target bounding box R to the energy in the background region:

Aˆ pi,j

η = (i,j)∈R

, (6)

Aˆ pi,j + ξ

(i,j)∈R/

where ξ is a small constant for numerical stability. A higher η signifies that the model’s highconfidence attention is strictly insulated within the target region, reflecting suppressed hallucinations. The quantitative results are presented in Figure 6. As shown in the right bar chart, the RL fine-tuning yields a solid improvement in instruction following, increasing the Recall by 6.2 pts (from 89.6% to 95.8%). More importantly, the left line chart visualizes the per-sample attention

[Figure 33]

- Fig. 8 Attention concentration ratio over denoising steps in Qwen-Image. Following TextGuider [50], we compute the attention concentration ratio as the mean image-to-text cross-modal attention within the OCR-detected text bounding box divided by the global mean over the entire image, and averaged over 100 successful instances. The closing quotation mark shows the earliest and strongest attention concentration, acting as a region-level layout anchor. Visual text tokens follow a similar but weaker trend due to finer-grained attention and dilution from loose OCR bounding boxes.

Prompt Sampling

Policy Optimization

[Figure 34]

[Figure 35]

T2I Model RL Algorithm

Prompt：A cozy coffee shop interior with a chalkboard reading 'Brew', a mug with 'Love' written on it, a bag of beans labeled 'Fresh', and a poster on the wall saying 'Relax'.

quality. The Effective Attention Efficiency (η) demonstrates a substantial 44.7% relative lift. This significant boost confirms that our method actively suppresses background leakage, reallocating high-confidence attention to target text regions to mitigate visual hallucinations.

Samples & Prompt

###### Bottleneck-aware Constrained Reinforcement Learning for Multi-text Insulation

Joint Attention Layer in MMDiT Blocks

- Step 1: Target Extraction & Instance Preprocessing
- Step 2: Isolated Fuzzy Matching
- Step 3: Insulation-aware Aggregation
- Step 4: Anti-interference Penalty

[Figure 36]

Key

Prompt Sample OCR Model

𝑉𝑉𝑇𝑇 = 𝑣𝑣𝑡𝑡𝑡,𝑣𝑣𝑡𝑡2,⋯,𝑣𝑣𝑡𝑡𝑛𝑛

Normalization

Query

Brew Love Fresh Relax

𝒪𝒪

|Text-to-Text Attention|Text-to-Image Attention|
|---|---|
|[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]|Image-to-Image Attention|

dist𝑣𝑣𝑡𝑡𝑖𝑖,𝑤𝑤

| | |⋯ 𝒪𝒪|
|---|---|---|

𝑠𝑠𝑖𝑖 = max

1 −

𝑣𝑣𝑡𝑡𝑖𝑖

𝑤𝑤⊆𝒪𝒪, 𝑤𝑤 = 𝑣𝑣𝑡𝑡𝑡𝑡

Sliding window: 𝑤𝑤

Post-training Settings. We conducted the RL post-training on only 4 NVIDIA A100 GPUs. For detailed hyperparameters, we primarily follow the configuration of Edit-R1 [51]. Motivated by the curriculum learning strategy [47], we adopt a two-stage training process. In the first stage, we fine-tune the model on the MixGRPO [52] training dataset for 1,000 steps, utilizing a weighted combination of PickScore [53], CLIPScore [54], and HPSv2.1 [55] rewards to preserve both aesthetic quality and semantic consistency. Subsequently, to specifically optimize text generation, we continue training for another 1,

Quotation-guided

AttentionGate

[Figure 41]

Attention Modulation

𝑛𝑛

1 𝑛𝑛

𝑅𝑅𝑏𝑏𝑏𝑏𝑏𝑏𝑏𝑏 = 1 − 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅

𝑠𝑠𝑖𝑖 + 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅ min 𝑠𝑠𝑡,…,𝑠𝑠𝑛𝑛

𝑖𝑖=𝑡

Noised Latent

Decoder

Image-to-Text Attention

Denoised Latent

𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 = 1.0

× N

Y

| | |
|---|---|
| | |

|Yes| |
|---|---|
|No| |

Ratio: 𝑏𝑏𝑏𝑏𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝

𝑅𝑅𝑛𝑛𝑜𝑜𝑜𝑜 = 𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 𝑅𝑅𝑏𝑏𝑏𝑏𝑏𝑏𝑏𝑏

≤ 𝛿𝛿

𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 = Ratio−𝛿𝛿+𝑡𝑡

𝑡𝑡𝑡𝑡𝑝𝑝𝑡𝑡𝑝𝑝𝑡𝑡

Fig. 9 Text-oriented Attention with Quotationguided Attention Gate. Within the Joint Attention Layer of MMDiT blocks, we leverage closing quotation marks as spatial anchors. The Quotation-guided Attention Gate is constructed from these anchors to dynamically modulate the Image-to-Text attention maps. Applied during inference, this mechanism enforces the concentration of text-related visual tokens within their designated regions, effectively mitigating feature leakage.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

T2I Model Text-orientedAttention

Prompt T2I Model

Prompt T2I Model

[Figure 46]

[Figure 47]

Prompt

LoRA

|000<br><br>[Figure 48]|
|---|

[Figure 49]

Glyph ControlNet

steps with the proposed OCR reward Rocr. This phase utilizes the English and Chinese text rendering subsets from the Qwen-Image-Self-GeneratedDataset [56], which collectively comprise approximately 60,000 prompts, covering diverse scenarios for multi-text rendering.

Reward Function of Text Insulation: 𝑅𝑅𝑛𝑛𝑜𝑜𝑜𝑜 = 𝜆𝜆𝑛𝑛𝑛𝑛𝑖𝑖𝑏𝑏𝑏𝑏 [ 1 − 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅

Glyph-based Control (e.g. AnyText)

𝑛𝑛

Pretrained Model (e.g. Qwen-Image)

1 𝑛𝑛

Ours

𝑠𝑠𝑖𝑖 + 𝜆𝜆𝑏𝑏𝑏𝑏𝑏𝑏 ⋅ min 𝑠𝑠𝑡,…, 𝑠𝑠𝑛𝑛 ]

Rendered Glyph

𝑖𝑖=𝑡

embeddings directly, we leverage the attention map of the quotation mark as a robust spatial anchor to guide the generation of visual text tokens. As illustrated in Figure 7, anchor quotation marks (i.e., closing quotation marks) consistently span the entire rendered text region they govern, suggesting they capture holistic, regionlevel information about the visual text layout.

###### 3.2.2 Text-oriented Attention

We further validate this behavior quantitatively in Figure 8, following the attention concentration analysis in [50]. At denoising step t, we extract the image-to-text cross-modal atten-

Effectiveness of Anchor Quotation Marks. It is crucial to prevent the attention of visual text tokens from leaking into the background and ensure their precise concentration within the designated regions. Rather than modifying the text

√

tion A(t) = softmax QimgK⊤text/

d , and define

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Attention Map Smoothing Primary Peak Retention Normalization & Soft Binarization

Generated Image

Gate Construction Pipeline

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- Fig. 10 Visualization of the Gate Construction Pipeline (Top) and Examples in Complex Scenarios (Bottom). Top: Taking the anchor quotation marks of two visual text instances as examples, we observe that directly using raw attention maps as gates is challenging due to inherent noise and multiple high-activation regions. To address this issue, we refine the map through three steps: 1) Smoothing removes scattered noise points in non-target regions. 2) Primary Peak Retention retains only the single region with the highest activation. 3) Normalization & Soft Binarization ensures continuous gate values bounded within [0, 1], where the target region approaches 1 and other regions approach 0. Bottom: We provide additional examples in complex scenarios featuring varying numbers of text instances (ranging from

- 3 to 5), demonstrating the robustness and adaptability of the gate mechanism in complex scenarios. In each example pair, the left side shows the Generated Image, and the right side displays the visualization of all gates simultaneously.

the attention map of a target text token τ over image tokens as a(τt)(p) = A(p,τt) . Given the OCRdetected text bounding box mask B (a set of image tokens) and the full image token set I, the Attention Concentration Ratio (ACR) is computed as

|B| p∈B a(τt)(p)

1

ACR(t)(τ) =

. (7)

|I| p∈I a(τt)(p)

1

For the visual text content inside the quotation marks, we first average the token-wise attention maps a¯(textt) (p) = |T 1

text| τ∈Ttext a(τt)(p) and then compute ACR(t)(text) by Eq. (7). As shown in Figure 8, closing quotation marks in Qwen-Image demonstrates the earliest and strongest attention concentration. Visual text tokens follow a similar but weaker trend due to finer-grained attention and dilution from loose OCR bounding boxes. This observation motivates us to utilize the attention map of the dominant quotation mark to construct a spatial gate, which dynamically highlights the text region and suppresses cross-text interference for visual text tokens.

Quotation-guided Attention Gate. We propose a Text-oriented Attention mechanism (see Figure 9) that employs a quotation-guided gate to modulate image-to-text attention. Let vtk denote the k-th visual text instance. Let qk be its corresponding anchor quotation mark, and Ck be the set of visual text tokens within the quotation marks.

At step t, we utilize attention map from previous step t − 1 to construct a spatial gate G(kt), which enhances the attention of visual text tokens in Ck. 1) Gate Construction. First, we extract the crossattention map of the anchor quotation mark qk, averaged over all L layers and H heads, indexed by ℓ and h respectively. Let p denote the spatial index. The raw map is a¯(kt−1)(p) =

L·H ℓ,h A(h,p,qt−1,ℓ)

1

. To generate a robust gate, we apply a specific operator G (see Figure 10), consisting of smoothing, primary peak retention, and normalization & soft binarization.

k

- • Smoothing: We apply a 5×5 average pooling to smooth out scattered noise points: a˜k = AvgPool(¯a(kt−1)).
- • Primary Peak Retention: We identify the

peak position p∗k = (x∗k,yk∗) = arg maxp a˜k(p) and suppress secondary peaks using a Gaussian

proximity mask. The Gaussian widths σx,k,σy,k are adaptively determined by the second central moments of smoothed attention map a˜k, representing the spatial extent around the peak:

- σx,k = p

a˜k(p)(x − x∗

k)2 p a˜k(p)

,

- σy,k = p

a˜k(p)(y − y∗

k)2 p a˜k(p)

,

(8)

|Previous Benchmark (MARIOEval)|Ours (CVTG-2K)|
|---|---|
|A cover titled 'Bad Times'|On a travel brochure, the cover title says 'Explore Now' in large bold orange, a destination name reads 'Sunny Beach' in medium blue italic, a package deal displays 'All Inclusive' in small black, a price tag shows '299 USD' in large red, and a contact section contains 'Book Today' in cursive.|

- Fig. 11 Comparison of prompt complexity and granularity. Left (MARIOEval [26]): Prompts typically contain only a single text content with minimal context. Right (CVTG-2K): Our prompts feature global context descriptions (Purple), multiple text content (Red), specific position requirements (Green), and fine-grained visual attributes (Blue), providing a more challenging and realistic evaluation for generative models.

introduce CVTG-2K, a challenging public benchmark for complex visual text generation. All prompts are generated through the OpenAI’s O1mini API [57], which encompass a variety of scenes containing complex visual texts. Unlike previous datasets synthesized using fixed rules, CVTG-2K ensures the diversity and rationality of the data distribution. We first prompt the O1 model to conceive a scene and then imagine the possible visual texts that may appear in that scene, utilizing Chain-of-Thought (CoT) [58] to ensure the quality of the generated prompts. Further details can be found in the supplementary material. Through rigorous filtering rules and meticulous post-processing, we have generated 2,000 prompts containing complex visual texts. On average, the visual texts in CVTG-2K contain 8.10 words and 39.47 characters, surpassing all previously published visual text benchmark datasets in terms of visual text length. Table 1 presents a comparison of the data in CVTG-2K with previous benchmark datasets. Furthermore, CVTG-2K is the first benchmark dataset to incorporate multiple visual text regions, with the number of regions ranging from 2 to 5. The distribution across region counts is approximately 20%, 30%, 30%, and 20%, respectively. With diverse numbers of text regions and varying visual text lengths, CVTG-2K enables comprehensive evaluation of model performance on complex visual text generation. Following a careful review process, we ensure CVTG-2K contains no discriminatory or inflammatory content.

where (σx,k,σy,k) critically determine the spatial extent of the gate, thereby controlling whether the gate is overly concentrated or overly diffuse. The region-masked map is then

∗ k)2

computed as rk(p) = a˜k(p) · exp(−(x−x

2σx,k2 − (y−yk∗)2

2σy,k2 ).

• Normalization & Soft Binarization: To form the final gate G(kt)(p) ∈ [0,1], we first apply max-norm normalization: rˆk(p) = rk(p)/maxp rk(p). Then, we define dynamic thresholds vlow = Quantile0.8(ˆrk) and vhigh = Quantile0.99(ˆrk). The gate is formulated as:

r ˆk(p) − vlow vhigh − vlow

G(kt)(p) = smoothstep

, (9)

where smoothstep(z) = z2(3 − 2z) maps values to [0,1].

2) Attention Modulation. The constructed gate G(kt) indicates the precise region where the k-th text should appear. We enhance the attention of all visual text tokens τ ∈ Ck strictly within this gated region. Finally, the attention map is updated as:

A(p,τt) ← A(p,τt) · 1 + G(kt)(p) , ∀τ ∈ Ck. (10)

This mechanism ensures that visual text tokens focus intensely on the text region defined by the anchor quotation mark, significantly mitigating text misgeneration and blurriness without disrupting the global layout.

To further enhance the challenge of CVTG2K, we partition the benchmark into two subsets: CVTG and CVTG-Style. CVTG-Style contains half of the data, where each visual text is randomly augmented with natural-language attribute annotations, while the remaining half in CVTG has no attributes. The attributes

###### 3.3 CVTG Benchmark

Currently, there is no publicly available benchmark dataset specifically designed for complex visual text generation. To address this gap, we

[Figure 61]

[Figure 62]

Fig. 12 Scene distributions of CVTG-2K (Left) and CVTG-Hard (Right). Outer ring shows coarse category frequencies, and inner ring indicates the split between CVTG and CVTG-Style.

include size (large, medium, small), color, and font (regular, bold, italic, cursive). Each text in CVTG-Style is assigned one or more attributes, facilitating systematic evaluation of visual text stylization and customization. Additionally, we provide fine-grained information. We employ O1mini to decompose each prompt into descriptions D = {d1,d2,...,dn} containing multiple visual texts, and to concisely express the correspondence between each text instance and its position using critical words, which serve as the carriers of the visual text. Meticulous manual review ensured the accuracy of the fine-grained information.

Scene Distributions of CVTG-2K. Figure 11 contrasts prompt complexity and granularity between prior benchmark MARIOEval and CVTG-2K. Figure 12 (Left) shows the scene distribution of CVTG-2K. The distribution is obtained by assigning each prompt to one coarse category using a local Qwen3-30B-A3B-Instruct-2507 [59] classifier with a predefined taxonomy and aggregating category frequencies. CVTG-2K has been publicly released together with the code and adopted to evaluate the visual text generation capabilities of multiple academic and industrial text-to-image models, including Qwen-Image [30], Z-image [60], LongCat-Image [61], EMU3.5 [62], GLM-Image [44] and Ovis-Image [63], contributing to research on text rendering.

Construction of CVTG-Hard. We further introduce CVTG-Hard, a challenging subset comprising the most difficult prompts from CVTG2K, together with their Chinese translations, yielding 400 test samples. As shown in Table 1, the visual texts in CVTG-hard contain an average of 8.61 words and 40.79 characters. Figure 12 (Right) shows CVTG-Hard still maintains the diverse scene distribution.

### 4 Experiments

###### 4.1 Evaluation and Baselines

Evaluation Metrics and Datasets. The experiments employ five metrics: Word Accuracy and Normalized Edit Distance (NED) [22] are employed to evaluate the accuracy of text rendering. CLIPScore [54], VQAScore [64], and Aesthetics [65] are utilized to assess the quality of the generated images. For detailed configurations of the metrics calculation, please refer to the supplementary materials. We evaluate TextCrafter on four datasets: The proposed CVTG-2K, CVTG-Hard, LongText-Bench [66] and Geneval [67]. Unlike previous benchmarks that primarily focus on short text or single-scene scenarios, LongText-Bench is specifically designed to evaluate the accuracy of generating extended text content in both English and Chinese across diverse scenarios.

Baselines. We compare TextCrafter with extensive state-of-the-art Text-to-Image models. Representative baselines from academia include AnyText [22], TextDiffuser2 [43], RAG-Diffusion [18], 3DIS [19], and Stable Diffusion 3.5 Large [14]. For a comprehensive comparison, we additionally compare TextCrafter against a wide array of models/products from the industry, including GPT Image [68], Gemini-2.5 [69], Seedream [70], QwenImage [30], HunyuanImage-3.0 [71], LongcatImage [61], Z-Image [60], EMU3.5 [62], GLMImage [44], FLUX.1 [13]. This setup allows us to rigorously validate TextCrafter’s effectiveness in complex visual text generation tasks.

###### 4.2 Quantitative Results

Comparison on CVTG-2K. As demonstrated in Table 2, TextCrafter surpasses competing methods in both text accuracy and image quality

- Table 2 Quantitative results on CVTG-2K dataset. TextCrafter is compared against state-of-the-art models from both industry and academia. ‘*’ denotes results cited from the previous papers. Bold values denote the best performance, while underlined values indicate the second-best performance for each metric. The three parenthesized values in the last row (e.g., +13.4%) denote the relative gains over the baseline model Qwen-Image on the corresponding metrics.

Model Word Accuracy (↑) NED (↑) CLIPScore (↑) VQAScore (↑) Aesthetics (↑) FLUX.1 dev [Black Forest Labs 2024] 0.4965 0.6879 0.7401 0.8886 5.91 GPT Image 1 [High] [OpenAI 2025]* 0.8569 0.9478 0.7982 - Gemini 2.5 Flash Image [Google 2025]* 0.7364 0.8516 - - Seedream 4.5 [ByteDance 2025]* 0.8990 0.9483 0.8069 - Qwen-Image [Alibaba 2025]* 0.8288 0.9116 0.8017 - Z-Image [Alibaba 2025]* 0.8671 0.9367 0.7969 - HunyuanImage-3.0 [Tencent 2025]* 0.7650 0.8765 0.8121 - Longcat-Image [Meituan 2025]* 0.8658 0.9361 0.7859 - Emu3.5 [BAAI 2025]* 0.9123 0.9656 - - GLM-Image [Z.ai 2026]* 0.9116 0.9557 0.7877 - -

SD3.5 Large [ICML 2024] 0.6548 0.8470 0.7797 0.9297 5.56 AnyText [ICLR 2024] 0.1804 0.4675 0.7432 0.6935 4.53 TextDiffuser-2 [ECCV 2024] 0.2326 0.4353 0.6765 0.5627 4.51 RAG-Diffusion [ICCV 2025] 0.2648 0.4498 0.6688 0.6397 5.58 3DIS [ICLR 2025] 0.3813 0.6505 0.7767 0.8684 4.86

TextCrafter (Qwen-Image) 0.9400(+13.4%) 0.9757(+7.0%) 0.8305(+3.6%) 0.9570 5.90

- Table 3 Quantitative results on CVTG-Hard dataset. For ZH, due to the absence of explicit whitespace word boundaries in Chinese, evaluation is performed at the span level rather than the word level. To accommodate spans split across lines, we allow matches formed by concatenating consecutive OCR lines. Span Accuracy is the fraction of ground-truth spans exactly matched by such concatenations. NED is the average, over all spans, of the normalized edit distance between each ground-truth span and its best-matching line concatenation.

EN ZH

Model

Word Accuracy (↑) NED (↑) Span Accuracy (↑) NED (↑)

FLUX.1 dev [Black Forest Labs 2024] 0.2427 0.4612 0.0000 0.0104 SD3.5 [ICML 2024] 0.4623 0.7078 0.0014 0.0105 Qwen-Image [Alibaba 2025] 0.6312 0.7776 0.6526 0.8237 Z-Image [Alibaba 2025] 0.7218 0.8477 0.7125 0.8548 Longcat-Image [Meituan 2025] 0.7991 0.8919 0.6894 0.8415 HunyuanImage3.0 [Tencent 2025] 0.6719 0.8221 0.5821 0.7315 GLM-Image [Z.ai 2026] 0.8171 0.9000 0.8610 0.9164

TextCrafter (Qwen-Image) 0.8862(+40.4%) 0.9470(+21.8%) 0.8692(+33.2%) 0.9518(+15.6%)

on CVTG-2K, with TextCrafter (Qwen-Image) improving word accuracy by 13.4% relative to Qwen-Image. While Stable Diffusion 3.5 performs adequately in simple scenarios, its efficacy diminishes significantly with increased textual complexity. AnyText and TextDiffuser-2, despite being trained on rule-based data, fail to generalize to multi-region tasks. Similarly, RAG-Diffusion and 3DIS struggles with visual text generation despite its multi-instance capabilities. By comparison, industry models [30, 62, 70] achieve significantly greater performance improvements than those from academia. While TextCrafter is primarily tailored for complex multi-text generation, it nevertheless exhibits exceptional robustness in rendering extended single-text sequences.

Comparison on CVTG-Hard. We further evaluated TextCrafter on the CVTG-Hard dataset, comparing with state-of-the-art open-source models. Note that EMU3.5 is not included, as its

inference is relatively time-consuming and our computational resources are limited. As illustrated in Table 3, the performance of competitive baselines, such as Z-Image and Qwen-Image, declines sharply on this challenging benchmark. Conversely, our model demonstrates exceptional robustness, with its Word/Span Accuracy surpassing Qwen-Image by 40.4% and 33.2% in English and Chinese scenarios, respectively.

Comparison on LongText-Bench. As shown in Figure 13, TextCrafter (Qwen-Image) also achieves state-of-the-art performance on LongText-Bench [66], outperforming leading commercial systems (e.g., GPT Image [68], Seedream [70]) and strong open-source competitors (e.g., Qwen-Image [30], EMU3.5 [62], GLMImage [44]). Notably, our model secures the competitive text accuracy across both English and Chinese subsets, demonstrating its versatility and precision in handling long texts.

[Figure 63]

- Fig. 13 Quantitative comparisons on LongText-Bench. We report the Text Accuracy on English (EN), Chinese (ZH), and their Average (Avg). Only models from industry (e.g., Qwen-Image, GPT Image 1 [High], Z-image etc.) are used for comparison, as existing academic approaches [18, 19, 22, 43] exhibit substantially inferior performance, often achieving scores below 0.5.

###### Table 4 Quantitative results on Geneval. All results are cited from the previous papers.

Model Single Object(↑) Two Object(↑) Counting(↑) Colors(↑) Position(↑) Attribute Binding(↑) Overall(↑) FLUX.1 dev [Black Forest Labs 2024] 0.98 0.81 0.74 0.79 0.22 0.45 0.66 GPT Image 1 [High] [OpenAI 2025] 0.99 0.92 0.85 0.92 0.75 0.61 0.84 Seedream 4.0 [ByteDance 2025] 0.99 0.92 0.72 0.91 0.76 0.74 0.84 Qwen-Image [Alibaba 2025] 0.99 0.92 0.89 0.88 0.76 0.77 0.87 Longcat-Image [Meituan 2025] 0.99 0.98 0.86 0.86 0.75 0.73 0.87 Z-Image [Alibaba 2025] 1.00 0.95 0.78 0.93 0.62 0.77 0.84 Show-o [ICLR 2025] 0.95 0.52 0.49 0.82 0.11 0.28 0.53 PixArt-α [ICLR 2024] 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SD3.5 Large [ICML 2024] 0.98 0.89 0.73 0.83 0.34 0.47 0.71 Lumina-Image 2.0 [ICCV 2025] - 0.87 0.67 - - 0.62 0.73 TextCrafter(Qwen-Image) 0.99 0.97 0.90 0.92 0.73 0.83 0.88

Comparison on Geneval. We additionally evaluated the TextCrafter (Qwen-Image) model on a general-purpose text-to-image benchmark Geneval [67] in Table 4 to assess its performance in generic scenarios. TextCrafter achieved an overall score of 0.88, slightly outperforming the baseline model Qwen-Image (0.87). This demonstrates that our approach significantly enhances complex visual text generation while maintaining strong performance in general text-to-image tasks.

###### 4.3 Qualitative Results

Visualizations on CVTG-2K Dataset. Figure 14 illustrates comparative visual results between TextCrafter and several state-of-theart academic approaches on CVTG-2K. While SD3.5 [14] and FLUX.1 dev [13] generate visually appealing images, they exhibit deficiencies in text rendering as regional complexity increases. AnyText [22] demonstrates suboptimal performance with multi-word text instances, TextDiffuser2 [43] compromises background fidelity (resulting

FLUX.1 dev SD 3.5 Large AnyText TextDiffuser-2 3DIS

Prompt: RAG-Diffusion TextCrafter

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

In a bookstore, a poster hangs with 'Read More', a book on the shelf is labeled 'Best Seller'.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

A park bench has a plaque saying 'Relax Here' in small green letters, a nearby tree displays 'Growing Strong' in large italic, and a jogging path has a sign with 'Keep Moving' in medium bold blue.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

A coffee shop counter with a sign saying 'Hot Today', a chalkboard displaying 'Grab And Go', a cup label reading 'Morning Blend', and a fridge sticker saying 'Cold Brew'.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

A cheerful classroom scene with a blackboard saying 'Learn' in large white letters, a bulletin board displaying 'News' in colorful bold letters, a desk labeled 'Student' in medium blue, and a window with 'View' in italic green letters.

- Fig. 14 Qualitative comparison of TextCrafter with other baselines on CVTG-2K. In the prompts, red content denotes the required visual text, while blue content denotes the attributes of visual text. TextCrafter excels in delivering harmonious and aesthetically pleasing images. It also accurately renders multiple visual texts while maintaining stability in complex scenarios.

in diminished CLIPScore), and 3DIS [19] deteriorates textual information during layout-to-depth conversion. While RAG-Diffusion [18] excels at region-aware generation, it tends to neglect the rendering of visual text. Conversely, TextCrafter achieves superior visual harmony while accurately rendering multiple texts without attribute confusion or prompt deviation.

Visualizations on CVTG-Hard. Figure 15 presents the qualitative comparison on the CVTGHard dataset, which demands precise text generation across multiple distinct regions. We compare our model with the baseline Qwen-Image. When dealing with challenging tasks involving larger volumes, Qwen-Image frequently exhibits significant issues such as text rendering errors, omissions, and hallucinations. In contrast, our model handles such cases effectively. Figure 16 illustrates additional visual comparison between TextCrafter and other industrial text-to-image models, including Z-Image, Longcat-Image, HunyuanImage, and GLM-Image. TextCrafter consistently outperforms these strong industrial models, particularly in text accuracy and adherence to specified attributes. More visualizations are provided in the supplementary material.

Visualizations on LongText-Bench. Figure 17 illustrates the visual results on LongText-Bench, a dataset characterized by high-density textual content. In these scenarios, Qwen-Image is prone to generation degradation, characterized by severe character omissions, spelling inconsistencies, and hallucinatory content as sequence length increases. Conversely, our model maintains robust character integrity.

###### 4.4 Ablation Study

To validate the effectiveness of the proposed components, we conduct ablation studies on the CVTG-Hard (English) subset. Table 5 presents the quantitative results, while Figure 18 provides the qualitative comparisons.

Ablation on Text Insulation. We first evaluate the impact of Text Insulation. As evidenced in Table 5, enabling Text Insulation significantly boosts performance, the proposed RL-based insulation raises Qwen-Image from 0.6312 to 0.8792 on Word Accuracy metric. This improvement stems from the effective decoupling of complex visual texts, effectively mitigating cross-text interference. Qualitatively, as shown in the top rows

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

In a lively sports stadium, a scoreboard displaying 'Home Team Leads 3-1', a concession stand sign that reads 'Hot Dogs 5 dollars', a banner across the field showing 'Championship Finals Tonight', and a restroom door labeled 'Men/Women’.

In a cozy coffee shop, a chalkboard with 'Daily Special: Caramel Latte', a window display showing 'Open from 6 AM to 8 PM', a door hanger reading 'Reserved for Customers', and a menu board that says 'Fresh Pastries Every Morning'.

在健身房更衣室内，一面镜子上的 贴纸写着“你行的”，毛巾架上的 标签写着“干毛巾”，门上的标牌 显示“男更衣室”，计时器上则显 示“时间到”。

在一家杂货店内，冷冻柜门上的贴纸写 着“冷冻食品五折优惠”，货架标签上 写着“有机农产品在此”，收银台的标 牌上写着“晚上9点后不接受现金支

付”，而一个区域隔板上则写着“零食 新品上市”。

###### Fig. 15 Qualitative comparison with Qwen-Image on CVTG-Hard (Top: Qwen-Image; Bottom: Ours). Red contents in the prompts denote the required visual texts.

###### Table 5 Ablation studies on CVTG-Hard (English) subset with Qwen-Image.

Method Word Accuracy (↑) NED (↑) Baseline (Qwen-Image) 0.6312 0.7776

+ Text Insulation 0.8792(+39.3%) 0.9369(+20.5%) + Text-oriented Attention 0.7422(+17.6%) 0.8598(+10.6%) + Both (Text Insulation and Attention) 0.8862(+40.4%) 0.9470(+21.8%)

of Figure 18, the baseline model frequently suffers from severe degradation. Specifically, the first column displays Text Misgeneration (red boxes), for example the target “Buy 2 Get 1 Free” is erroneously rendered as “Buy 251” and “FIEE”. Meanwhile, the second column exhibits extensive unintelligible gibberish (yellow boxes) that was not requested in the prompt. By enforcing independent generation regions, Text Insulation not only guarantees textual correctness but also effectively eliminates these hallucinated artifacts, yielding clean images with sharp, legible text.

Ablation on Text-oriented Attention. This strategy effectively alleviates the Text Omission challenge by ensuring concentration on all

requested texts. Quantitative results in Table 5 show that this module further enhances the metrics. When combined with Text Insulation (the “+ Both” setting), the Word Accuracy and NED peak at 0.8862 and 0.9470, respectively. Visual comparisons in the bottom rows of Figure 18 demonstrate its specific utility. Without this module, texts in peripheral or small regions (e.g., the “LOUD”) are often omitted. Enabling Text-oriented Attention successfully recovers these missing texts (highlighted in blue boxes). Furthermore, the Primary Peak Retention strategy employed during the gate construction ensures that attention is concentrated within a single connected region. This spatial constraint effectively suppresses the

[Figure 105]

- Fig. 16 Visual comparison on CVTG-Hard against state-of-the-art industrial models. In the prompt, red indicates the target visual text, and blue indicates the required attributes.

text repetition hallucination, preventing the model from generating duplicate text artifacts.

Ablation on RL Algorithms. We further ablate our reward function Rocr using two RL algorithms: DiffusionNFT and Flow-GRPO. As shown in Table 6, both methods yield substantial improvements over the baseline (Word Acc. 0.6312). Specifically, DiffusionNFT achieves a Word Accuracy of 0.8792 (+39.3%), while FlowGRPO attains a comparable 0.8699 (+37.8%). The consistent performance gains across these disparate optimization frameworks demonstrate that our proposed reward model is robust and algorithm-agnostic, effectively guiding the optimization process regardless of the underlying RL framework.

Ablation on the Hyperparameters λbal and δ in Reward Model. We conduct ablation studies on two critical hyperparameters: the balancing coefficient λbal, which is designed to mitigate omissions by emphasizing worst-case instances, and the length tolerance threshold δ, which serves as a constraint to suppress hallucinations. Table 7 reveals an inverted U-shape trend for both parameters. λbal = 0.3 achieves peak accuracy (0.8792) by effectively recovering hard instances, whereas deviations lead to neglect or instability. Similarly, δ = 1.5 proves optimal; strict thresholds penalize valid noise, while looser constraints (δ = 1.8) cause a performance drop to 0.8554 due to redundancy. Figure 19 visually confirms these findings. The difficult word “QUIET” is omitted at both

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

An artistic poster …: …\"Discover Films That Will Change Your Vision\" is featured prominently in stylish vintage typography. …. Slightly below, centered boldly is written \"Indie Lens Festival\" in clear distinctive lettering. At the bottom section of the poster, arranged neatly and balanced visually, appears the festival date \"March 5–9\" along with the location \"Downtown Art Theater\". …

A spacious and vibrant contemporary art gallery …. A prominent label distinctly identifies the central artwork as \"Echoes of Time by

黄昏时分热闹的市中心十字路口， 上方高高悬挂着巨大的广告牌。广 告牌上醒目地展示着清晰的大字 “在山景水疗度假村享受宁静”， 下方稍小但十分显眼的字样写着 “尽情享受奢华护理，在令人惊叹 的景色中放松身心”。广告牌底部 另有简洁提示文字“立即预订，首 次可享20%折扣！”和“详情访问 山景水疗度假村官网”。这些文字 以柔和的邀请式字体呈现，在充满 自然气息的宁静背景下吸引着路人 的目光。

引人注目的木质边框粉笔板伫立在温馨 街边面包店入口外侧。板面以迷人手写 风格醒目书写着“甜蜜乐园面包店 每

日新鲜烘焙的幸福！”下方紧邻着更小 字号的友好字迹补充道“今日特供：巧 克力牛角包”，并附上温暖邀请“进店 可免费品尝任意热饮”。招牌底部以优 雅的连笔字宣告“营业时间 上午7点至 下午6点，每日清晨现烤新鲜面包！”

Amelia Hart, Oil on Canvas\", immediately drawing viewers' attention; …, \"Created in 2022\".

Surrounding artworks feature equally concise, clear, and informative labels, such as \"Beyond Horizons by J.

店铺橱窗周围散落着更小的告示牌，展 示着欢快信息如“各类庆典定制蛋糕！” 所有文字内容整齐美观、亲切易读，精 心设计以热情迎接路人。

Rivera\", \"Fragmented Realities by L. Tanaka\", and \"Silent Reflections by M. Weber\". …

###### Fig. 17 Qualitative comparison with Qwen-Image on LongText-Bench (Top: Qwen-Image; Bottom: Ours). Red contents in the prompts denote the required visual texts.

Table 6 Effectiveness of the proposed reward function across different RL algorithms. Evaluated on the CVTG-Hard (English) subset. Our reward model consistently improves performance regardless of the optimization strategy.

Method Word Acc. (↑) NED (↑) Baseline (Qwen-Image) 0.6312 0.7776 DiffusionNFT (with Rocr) 0.8792(+39.3%) 0.9369(+20.5%) Flow-GRPO (with Rocr) 0.8699(+37.8%) 0.9332(+20.0%)

λbal extremes (due to average-based optimization or instability) but correctly rendered at 0.3. For length tolerance, strict values (δ ≤ 1.2) cause over-suppression (missing “Time” on the clock), while loose values (δ ≥ 1.8) induce hallucinations (redundant “Work” on the table).

Ablation on Smoothing Kernel Size. The smoothing operation is pivotal for mitigating highfrequency noise in raw attention maps. We evaluate the impact of the average pooling kernel size k × k on the CVTG-Hard (English) subset, testing standard sizes k ∈ {3,5,7}. As presented in

Table 8, the 5 × 5 kernel yields the optimal performance. Consequently, we adopt a 5 × 5 kernel for all experiments.

Ablation on the Gaussian widths. To validate the effectiveness of our adaptive strategy, we conduct a comparative analysis on the CVTG-Hard (English) subset. We compare our proposed adaptive approach against a series of fixed widths: 1) Fixed widths. We set constant widths σx,k = σy,k for all text instances regardless of their size, testing values in {6,8,...,22}. 2) Second central moment (Adaptive). We adopt the adaptive strategy described in Section 3.2.2, where

w/o Text Insulation w/o Text-oriented Attention

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

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

with Text Insulation with Text-oriented Attention

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

| |
|---|

In a cozy bookstore featuring a window display with 'Best Sellers of the Year', a reading nook sign saying 'Quiet Please', a cashier counter banner reading 'Buy 2 Get 1 Free', a children section label 'Kids Stories and More', and a café area sign showing 'Enjoy Your Coffee Here'.

On a university website homepage with a header showing 'Welcome to State University', a navigation menu item labeled 'Academic Programs', a hero banner reading 'Discover Your Potential', a sidebar section titled 'Upcoming Events', and a footer note saying 'Contact Us for More Information'.

A lively music concert with a stage banner that says 'Live'. A guitar amplifier reads 'LOUD'.

At a bookstore, a large display says 'New Arrivals', a shelf label reads 'Best Sellers', a reading nook sign says 'Quiet Area', and a cashier window displays 'Checkout Fast'.

###### Fig. 18 Qualitative visualization of ablation studies using Qwen-Image as the baseline model. Red boxes indicate the occurrence of Text Misgeneration. Yellow boxes represent the occurrence of Text Hallucination, including unintelligible gibberish and repeated text. Blue boxes highlight the resolution of Text Omission.

Table 7 Ablation on hyperparameters λbal and δ evaluated on the CVTG-Hard (English) subset. Bold denotes the best performance.

λbal δ Word Acc. (↑) NED (↑)

- 0.0 1.5

0.8776 0.9362

- 0.1 0.8770 0.9366 0.3 0.8792 0.9369 0.5 0.8630 0.9325 0.8 0.8788 0.9358

1.0 0.8734 0.9333 1.2 0.8595 0.9266

0.3

- 1.8 0.8554 0.9243
- 2.0 0.8702 0.9337

Table 8 Ablation on Smoothing Kernel Size in Text-oriented Attention module on CVTG-Hard (English) subset. Bold denotes the best results.

Kernel Size Word Acc. (↑) NED (↑) Baseline (Qwen-Image) 0.6312 0.7776

3 × 3 0.7288 0.8504 5 × 5 0.7422 0.8598 7 × 7 0.7369 0.8563

the Gaussian widths are dynamically computed

- as the second central moments of the smoothed attention map, allowing the gate to adapt to the text’s actual coverage without manual tuning. The quantitative results are reported in Figure 20.

Fixed widths vary substantially across settings, indicating strong sensitivity to parameter choice. In contrast, the proposed second central moment consistently approaches the best-performing fixed setting. This attention-driven formulation is adaptive and eliminates the need for manual hyperparameter tuning. We therefore use it in all subsequent experiments.

Complexity Analysis. In TextCrafter (QwenImage) at 1024 × 1024 resolution, the Text Insulation module incurs no additional inference latency. The introduction of Text-oriented Attention increases the peak GPU memory usage from ∼55GB to ∼60GB and extends inference time due to extra attention computations. However, our framework is compatible with distillation techniques such as DMD2 [72] and sCM [73]. This integration can reduce the number of sampling steps from 50 to 8 (or even 4), delivering over a 6× speedup while enabling controllable quality–efficiency trade-offs.

### 5 Conclusion

In this paper, we present TextCrafter, a Complex Visual Text Generation (CVTG) framework

[Figure 122]

- Fig. 19 Visual ablation study on the balancing coefficient λbal and length tolerance threshold δ. The top row demonstrates the effect of λbal (with fixed δ = 1.5). The difficult text instance “QUIET” is omitted at both extremes (λbal = 0.0 and 0.8), while the optimal setting (λbal = 0.3) successfully renders it. The bottom row illustrates the impact of δ (with fixed λbal = 0.3). Strict thresholds (δ ≤ 1.2) lead to over-suppression (missing “Time” on the clock), whereas loose thresholds (δ ≥ 1.8) induce hallucinations (redundant “Work” on the table). The central column represents our default setting, achieving the best trade-off between completeness and cleanliness.

6 8 10 12 14 16 18 20 22

0.72

0.74

0.76

Gaussian Width

↑WordAccuracy()

Adaptive (Ours) Fixed Widths

6 8 10 12 14 16 18 20 22

0.84

0.86

Gaussian Width

↑NED()

Adaptive (Ours) Fixed Widths

- Fig. 20 Ablation of Gaussian widths in Text-oriented Attention module on the CVTG-Hard (English) subset. We compare the fixed width settings (Red line) against our proposed adaptive Second Central Moment strategy (Blue line). The adaptive method achieves performance competitive with the optimal fixed setting without requiring manual tuning.

inspired by selective visual attention in cognitive science. TextCrafter employs the “Text Insulation-and-Attention” mechanisms to address the challenges of text misgeneration, omissions, and hallucinations. Specifically, on one hand, TextCrafter presents a Text Insulation module that introduces a novel Bottleneck-aware Constrained Reinforcement Learning method to explicitly optimize the fidelity of each text instance. On the other hand, TextCrafter introduces a text-oriented attention module with a novel Quotation-guided Attention Gate to further enhance text rendering. Last but not least, we

introduce CVTG-2K, the first challenging benchmark for complex visual text generation, comprising 2,000 complex visual-text prompts. Extensive experiments demonstrate that TextCrafter significantly outperforms existing state-of-the-art models on several challenging datasets: CVTG2K, CVTG-Hard and LongText-Bench, including highly capable industry models (e.g., GPT Image, Qwen-Image, Seedream etc.) that are trained with substantial resources.

Limitations. An effective text rendering model should not only generate text accurately but also minimize hallucinated content. Although we

incorporate the “Insulation and Attention” mechanisms to improve text rendering performance, it remains challenging for current state-of-the-art industrial systems and our TextCrafter to ensure the complete absence of hallucinations or textual inaccuracies within a single generation. Consequently, enhancing the robustness of complex text rendering warrants further in-depth exploration.

### Acknowledgment

This work was supported by the Gusu Innovation and Entrepreneur Leading Talents: No. ZXL2024362, Natural Science Foundation of Jiangsu Province: BK20241198, and Natural Science Foundation of China: No. 62406135.

### Data Availability Statement

CVTG-2K dataset is available at: https:// huggingface.co/datasets/dnkdnk/CVTG-2K MixGRPO training dataset is available

- at: https://github.com/Tencent-Hunyuan/ MixGRPO/blob/main/data/prompts.txt Qwen-Image-Self-Generated-Dataset is available at: https://modelscope. cn/datasets/DiffSynth-Studio/ Qwen-Image-Self-Generated-Dataset

### References

- [1] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: Proceedings of Neural Information Processing Systems, vol. 33, pp. 6840–6851 (2020)
- [2] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Mu¨ller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. In: Proceedings of International Conference on Learning Representations (2024)
- [3] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10684– 10695 (2022)

- [4] Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever,

I.: Zero-shot text-to-image generation. In: Proceedings of International Conference on Machine Learning, pp. 8821–8831 (2021)

- [5] Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 (2022)
- [6] Zhao, C., Cai, W., Dong, C., Hu, C.: Waveletbased fourier information interaction with frequency diffusion adjustment for underwater image restoration. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 8281–8291 (2024)
- [7] Dhariwal, P., Nichol, A.: Diffusion models beat GANs on image synthesis. In: Proceedings of Neural Information Processing Systems, vol. 34, pp. 8780–8794 (2021)
- [8] Chen, Z., Gao, R., Xiang, T.-Z., Lin, F.: Diffusion model for camouflaged object detection. In: Proceedings of European Conference on Artificial Intelligence, pp. 445–452 (2023)
- [9] Nan, K., Xie, R., Zhou, P., Fan, T., Yang, Z., Chen, Z., Li, X., Yang, J., Tai, Y.: Openvid1m: A large-scale high-quality dataset for text-to-video generation. In: Proceedings of International Conference on Learning Representations (2025)
- [10] Fan, T., Nan, K., Xie, R., Zhou, P., Yang, Z., Fu, C., Li, X., Yang, J., Tai, Y.: Instancecap: Improving text-to-video generation via instance-aware structured caption. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 28974–28983

(2025)

- [11] Xie, R., Zhao, C., Zhang, K., Zhang, Z., Zhou, J., Yang, J., Tai, Y.: Addsr: Accelerating diffusion-based blind super-resolution with adversarial diffusion distillation. arXiv preprint arXiv:2404.01717 (2024)
- [12] Zhao, C., Ci, E., Xu, Y., Fan, T., Guan, S., Ge, Y., Yang, J., Tai, Y.: Ultrahr-100k:

- Enhancing uhr image synthesis with a largescale high-quality dataset. In: Proceedings of Neural Information Processing Systems (2025)
- [13] BlackForest: Black Forest Labs; Frontier AI Lab. https://blackforestlabs.ai/ (2024)
- [14] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Mu¨ller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Proceedings of International Conference on Machine Learning (2024)
- [15] Long, S., He, X., Yao, C.: Scene text detection and recognition: The deep learning era. International Journal of Computer Vision 129(1), 161–184 (2021)
- [16] Feng, W., He, X., Fu, T.-J., Jampani, V., Akula, A.R., Narayana, P., Basu, S., Wang, X.E., Wang, W.Y.: Training-free structured diffusion guidance for compositional text-toimage synthesis. In: Proceedings of International Conference on Learning Representations (2023)
- [17] Zhou, D., Li, Y., Ma, F., Zhang, X., Yang, Y.: Migc: Multi-instance generation controller for text-to-image synthesis. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6818–6828 (2024)
- [18] Chen, Z., Li, Y., Wang, H., Chen, Z., Jiang, Z., Li, J., Wang, Q., Yang, J., Tai, Y.: Region-aware text-to-image generation via hard binding and soft refinement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2025)
- [19] Zhou, D., Xie, J., Yang, Z., Yang, Y.: 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. In: Proceedings of International Conference on Learning Representations (2024)
- [20] Liu, Y., He, T., Chen, H., Wang, X., Luo, C., Zhang, S., Shen, C., Jin, L.: Exploring the capacity of an orderless box discretization network for multi-orientation scene text detection. International Journal of Computer

- Vision 129(6), 1972–1992 (2021)
- [21] Yang, L., Yu, Z., Meng, C., Xu, M., Ermon, S., Bin, C.: Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In: Proceedings of International Conference on Machine Learning

(2024)

- [22] Tuo, Y., Xiang, W., He, J.-Y., Geng, Y., Xie, X.: Anytext: Multilingual visual text generation and editing. In: Proceedings of International Conference on Learning Representations (2024)
- [23] Liu, Z., Liang, W., Liang, Z., Luo, C., Li, J., Huang, G., Yuan, Y.: Glyph-byt5: A customized text encoder for accurate visual text rendering. In: Proceedings of European Conference on Computer Vision, pp. 361–377

(2025)

- [24] Zhangli, Q., Jiang, J., Liu, D., Yu, L., Dai, X., Ramchandani, A., Pang, G., Metaxas, D.N., Krishnan, P.: Layout-agnostic scene text image synthesis with diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7496– 7506 (2024)
- [25] Zhang, L., Chen, X., Wang, Y., Lu, Y., Qiao, Y.: Brush your text: Synthesize any scene text on images via diffusion model. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 7215–7223 (2024)
- [26] Chen, J., Huang, Y., Lv, T., Cui, L., Chen, Q., Wei, F.: Textdiffuser: Diffusion models as text painters. In: Proceedings of Neural Information Processing Systems, vol. 36

(2024)

- [27] Desimone, R., Duncan, J., et al.: Neural mechanisms of selective visual attention. Annual review of neuroscience 18(1), 193–222

(1995)

- [28] Stevens, C., Bavelier, D.: The role of selective attention on academic foundations: A cognitive neuroscience perspective. Developmental cognitive neuroscience 2, 30–48 (2012)

- [29] Duncan, J.: Selective attention and the organization of visual information. Journal of experimental psychology: General 113(4), 501 (1984)
- [30] Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)
- [31] Yang, Y., Gui, D., Yuan, Y., Liang, W., Ding, H., Hu, H., Chen, K.: Glyphcontrol: Glyph conditional control for visual text generation. In: Proceedings of Neural Information Processing Systems, vol. 36 (2023)
- [32] Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision
- [33] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: Proceedings of International Conference on Learning Representations
- [34] Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attentionbased semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics 42(4), 1–10 (2023)
- [35] Xie, J., Li, Y., Huang, Y., Liu, H., Zhang, W., Zheng, Y., Shou, M.Z.: Boxdiff: Textto-image synthesis with training-free boxconstrained diffusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7452–7461 (2023)
- [36] Cong, Y., Min, M.R., Li, L.E., Rosenhahn, B., Yang, M.Y.: Attribute-centric compositional text-to-image generation. International Journal of Computer Vision 133(7), 4555– 4570 (2025)
- [37] Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Openset grounded text-to-image generation. In: Proceedings of the Computer Vision and

- Pattern Recognition Conference, pp. 22511– 22521 (2023)
- [38] Zhang, H., Hong, D., Wang, Y., Shao, J., Wu, X., Wu, Z., Jiang, Y.-G.: Creatilayout: Siamese multimodal diffusion transformer for creative layout-to-image generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18487– 18497 (2025)
- [39] Zhou, D., Li, M., Yang, Z., Yang, Y.: DreamRenderer: Taming multi-instance attribute control in large-scale text-to-image models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision

(2025)

- [40] Ji, J., Zhang, G., Wang, Z., Hou, B., Zhang, Z., Price, B.L., Chang, S.: Improving diffusion models for scene text editing with dual encoders. Transactions on Machine Learning Research (2023)
- [41] Zhao, Y., Lian, Z.: Udifftext: A unified framework for high-quality text synthesis in arbitrary images via character-aware diffusion models. In: Proceedings of European Conference on Computer Vision, pp. 217–233

(2025)

- [42] Ma, J., Zhao, M., Chen, C., Wang, R., Niu, D., Lu, H., Lin, X.: GlyphDraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation. arXiv preprint arXiv:2303.17870 (2023)
- [43] Chen, J., Huang, Y., Lv, T., Cui, L., Chen, Q., Wei, F.: TextDiffuser-2: Unleashing the power of language models for text rendering. In: Proceedings of European Conference on Computer Vision, pp. 386–402 (2025)
- [44] AI, Z.: Glm-image: https://docs.z.ai/guides/image/glm-image

(2026)

- [45] Cui, C., Sun, T., Lin, M., Gao, T., Zhang, Y., Liu, J., Wang, X., Zhang, Z., Zhou, C., Liu, H., Zhang, Y., Lv, W., Huang, K., Zhang, Y., Zhang, J., Zhang, J., Liu, Y., Yu, D., Ma, Y.: PaddleOCR 3.0 Technical Report (2025).

- https://arxiv.org/abs/2507.05595
- [46] Levenshtein, V.I.: Binary codes capable of correcting deletions, insertions, and reversals. Soviet physics doklady 10(8), 707–710 (1966)
- [47] Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., Liu, M.-Y.: Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117 (2025)
- [48] Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., Ouyang, W.: Flow-grpo: Training flow matching models via online rl. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS) (2025). https://arxiv.org/abs/2505.05470
- [49] Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al.: Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818 (2025)
- [50] Baek, K., Lee, S., Choi, J.Y., Song, J., Park, D., Choi, J., Shin, C., Han, B., Yoon, S.: TextGuider: Training-free guidance for text rendering via attention alignment. arXiv preprint arXiv:2512.09350 (2025)
- [51] Li, Z., Liu, Z., Zhang, Q., Lin, B., Yuan, S., Yan, Z., Ye, Y., Yu, W., Niu, Y., Yuan, L.: Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888 (2025)
- [52] Li, J., Cui, Y., Huang, T., Ma, Y., Fan, C., Yang, M., Zhong, Z.: MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE (2025). https://arxiv.org/ abs/2507.21802
- [53] Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems 36, 36652–36663 (2023)
- [54] Hessel, J., Holtzman, A., Forbes, M., Le Bras,

- R., Choi, Y.: CLIPScore: A reference-free evaluation metric for image captioning. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing, pp. 7514–7528 (2021)
- [55] Wu, X., Sun, K., Zhu, F., Zhao, R., Li, H.: Human preference score: Better aligning textto-image models with human preference. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2096– 2105 (2023)
- [56] DiffSynth-Studio: Qwen-Image-SelfGenerated-Dataset. https://www. modelscope.cn/datasets/DiffSynth-Studio/ Qwen-Image-Self-Generated-Dataset. Accessed: 2026-01-27 (2025)
- [57] OpenAI: O1Mini: Advancing cost-efficient reasoning. https://openai.com/index/ openai-o1-mini-advancing-cost-efficient-reasoning/

(2024)

- [58] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. In: Proceedings of Neural Information Processing Systems, vol. 35, pp. 24824–24837 (2022)
- [59] Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J.,

- Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K.,
- Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., Qiu, Z.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

- [60] Cai, H., Cao, S., Du, R., Gao, P., Hoi, S., Huang, S., Hou, Z., Jiang, D., Jin, X., Li, L., et al.: Z-image: An efficient image generation foundation model with singlestream diffusion transformer. arXiv preprint arXiv:2511.22699 (2025)

- [61] Team, M.L., Ma, H., Tan, H., Huang, J., Wu, J., He, J.-Y., Gao, L., Xiao, S., Wei, X., Ma, X., et al.: Longcat-image technical report. arXiv preprint arXiv:2512.07584 (2025)
- [62] Cui, Y., Chen, H., Deng, H., Huang, X., Li, X., Liu, J., Liu, Y., Luo, Z., Wang, J., Wang, W., et al.: Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583 (2025)
- [63] Wang, G.-H., Cao, L., Cui, T., Fu, M., Chen, X., Zhan, P., Zhao, J., Li, L., Fu, B., Liu, J., et al.: Ovis-image technical report. arXiv preprint arXiv:2511.22982 (2025)
- [64] Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., Ramanan, D.: Evaluating text-to-visual generation with image-totext generation. In: Proceedings of European Conference on Computer Vision, pp. 366–384

(2024)

- [65] LAION AI: LAION-Aesthetics Predictor V1. https://github.com/LAION-AI/ aesthetic-predictor (2022)
- [66] Geng, Z., Wang, Y., Ma, Y., Li, C., Rao, Y., Gu, S., Zhong, Z., Lu, Q., Hu, H., Zhang, X., et al.: X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058 (2025)
- [67] Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems 36, 52132–52152 (2023)
- [68] OpenAI: Gpt image 1: https://platform.openai.com/docs/models/gptimage-1 (2025)
- [69] Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

- [70] Seedream, T., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al.: Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427

(2025)

- [71] Cao, S., Chen, H., Chen, P., Cheng, Y., Cui, Y., Deng, X., Dong, Y., Gong, K., Gu, T., Gu, X., et al.: Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951 (2025)
- [72] Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, B.: Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems 37, 47455– 47487 (2024)
- [73] Lu, C., Song, Y.: Simplifying, stabilizing and scaling continuous-time consistency models. In: The Thirteenth International Conference on Learning Representations

### Supplementary Material

### A Prompt Design of O1-mini

We divided the system prompt into three parts (i.e., Character, Constraints and Examples), aiming to both impose constraints on the model and stimulate its creativity. Through extensive testing on challenging samples, including numerous visual texts, uncommon words, and complex scenes, we finalized the system prompt shown in Figure 21. We designed prompts to generate varying numbers of regions and word counts, with the number of regions ranging from 2 to 5, and word counts categorized into single words (short), 2 or 3 words (medium length), and more than 4 words (long). Figure 21 illustrates an example with 3 regions and medium-length visual texts.

Attribute Insertion. After generating the original benchmark, we selected half of the samples to introduce additional attributes, aiming to test the image generation model’s ability to control visual text attributes. We also divided the system prompt into three parts to maximize the language model’s reasoning capabilities. The attributes to be added are limited to three types: size, color, and font, with each visual text randomly assigned one or more attributes. Figure 22 illustrates the system prompt used for adding attributes.

Fine-grained Information. Additionally, we annotated more fine-grained information for CVTG-2K to assist future research on the CVTG task. We decoupled complex multiple visual texts, isolated clauses that describe each visual text, and used a key word to represent the carrier of the visual text. The results are stored in JSON format. Figure 23 illustrates the system prompt used for annotating granular information.

### B Example of CVTG-2K

Using the method introduced in Prompt Design of O1-mini, we assign different proportions to different numbers of regions. From 2 regions to 5 regions, the proportions are approximately 20%, 30%, 30%, and 20% respectively. We also assign different proportions to texts of different lengths, with medium-length texts accounting for the largest number, accounting for approximately 60%. Figure 24 shows randomly selected samples under different numbers of regions.

### C Detailed Configurations of the Metric Calculation

Notation and Preprocessing. For each prompt Pj, we use the same symbol definition as the main paper and denote the target visual-text set as V Tj = {vtj,1,vtj,2,...,vtj,n

j}. The evaluator extracts V Tj by collecting all strings enclosed by single quotes in Pj, then lowercasing and splitting by whitespace. Given image Ij, PPOCR-v4 outputs an OCR word list Oj. For each ground-truth text vtj,i, the matched OCR word is selected by top-1 difflib.get close matches:

vtˆ j,i = Match(vtj,i,Oj). (11)

Word Accuracy. At region count r (sample set Sr), a target word is counted as correct only if an exact match appears in Oj:

nj i=1 1[vtj,i ∈ Oj]

WordAccr = j∈Sr

. (12)

j∈Sr nj

Normalized Edit Distance (NED). Consistent with the script, NED is implemented as the normalized edit similarity (↑). For a subset Sr:

nj i=1 1 − maxdist((|vtvtj,ij,i|,|,vtvtˆˆj,ij,i|))+ϵ , ϵ = 10−5

NEDr = 1

j∈Sr nj j∈Sr

(13) where dist(·,·) is the Levenshtein distance.

CLIPScore. The input text is prefixed as Tj = “A photo depicts ”+Pj. Using ViT-L/14 features with L2 normalization (f(·)), the score for image Ij is:

CLIPScore(Ij,Pj) = 2.5 · max 0,f(Ij)⊤f(Tj) .

(14) The region-level result is the average over images:

1 |Sr| j∈S

CLIPScorer =

r

CLIPScore(Ij,Pj). (15)

VQAScore and Aesthetics. VQAScore is computed by t2v metrics (clip-flant5-xxl) and Aesthetics by LAION Aesthetics Predictor V1:

1 |Sr| j∈S

VQAScorer =

r

###### VQA(Ij,Pj), (16)

[Figure 123]

[Figure 124]

###### System Prompt

# Character You are now an expert in the field of image generation. You are testing the visual text generation capabilities of a text-to-image generation model. Next, you need to design prompts that include visual text. Please use your imagination to design a variety of different prompts, including but not limited to street scenes, book covers, advertisements, websites, posters, notes, memes, logos, movie scenes, etc.

# Constraints Additional requirements:

- •Each visual text should contain only English and Arabic numerals, avoiding complex or uncommon punctuation marks.
- •Each prompt should not exceed 77 words and use common words as much as possible.
- •Do not describe the style, color, font, and size of the visual text.
- •Visual text should appear only in four areas. That is to say, each prompt should have four objects with visual text.
- •For each area, the object is mentioned first, followed by the associated visual text enclosed in single quotes.
- •Single quotes are only used to surround visual text, do not use single quotes elsewhere.
- •Only design visual text with more than or equal to four words.

# Examples Reference examples: A busy city street, a sign with the text 'Welcome to Downtown' , a billboard displaying 'Grand Opening This Weekend', a storefront window showing 'Fresh Bakery Daily', and a bus stop sign that reads 'Next Bus Arrives at 3 PM'. In a vibrant marketplace scene, a banner saying 'Holiday Sale Now On', a restaurant window displaying 'Open 7 Days a Week', an entrance sign reading 'Express Line to Central', and a park bench with the text 'Enjoy Your Day'. A corporate office building with a sign that says 'Headquarters', a digital board displaying 'Quarterly Results Released', a reception desk note reading 'Welcome to Our Company', and a conference room door that shows 'Meeting in Progress'.

Fig. 21 System prompt of original CVTG-2K.

1 |Sr| j∈S

Aesr =

Aes(Ij). (17)

r

Overall Aggregation. For the final summary over all benchmark subsets/region groups u, Word Accuracy and NED are micro-averaged by total word count:

Cu u Nu

WordAccall = u

,

Nu k=1 nedu,k

NEDall = u

.

u Nu

(18)

where Cu is the count of correct words and Nu is the total number of words in subset u. Conversely, CLIPScore, VQAScore, and Aesthetics are

averaged by image count:

Mall = u |Su| · Mu u |Su|

, M ∈ {CLIPScore,VQAScore,Aes}.

### D Visual Comparison with Industrial Models

(19)

In this section, we present a comprehensive visual comparison between TextCrafter and state-ofthe-art industrial text-to-image models, including Qwen-Image, Z-Image, Longcat-Image, HunyuanImage, and GLM-Image. All comparisons are

[Figure 125]

[Figure 126]

##### System Prompt

# Character You are now an expert in the field of image generation. You are testing the visual text generation capabilities of a text-to-image generation model. Next, I will provide you with some initially designed prompts. Without changing the content of these prompts, add descriptive attributes to each visual text (i.e., the content within single quotes).

# Constraints The attributes should only include size(small, medium, large), color and font(regular, bold, italic, cursive). Please use your imagination and reasoning skills to add appropriate attributes to each visual text (i.e., the content within single quotes). Randomly add one to three of the size, color or font attributes to each visual text (for example, only add the size attribute or add both size and color or add size, color, and font attributes at the same time). Describe the added attributes using the most natural language, without using labels. Each prompt should not exceed 77 words. Provide the prompts with added attributes directly. Do not include any extra content, do not number them.

# Examples Reference examples: The cover of a magazine with the title 'Top Stories' in large bold black letters. In a modern office, a window reads 'Open' in large letters, a door has the text 'Enter' in green cursive. A city billboard shows 'Sale' in bright yellow large letters. A bus stop bench says 'Wait' in italic. A storefront window displays 'New' in bold blue.

Prompts: …

- Fig. 22 System prompt of inserting attributes to CVTG-2K.

conducted on the challenging CVTG-Hard subset proposed in our main paper, which involves complex layouts and rich attribute constraints. Figures 25-30 illustrate the qualitative results. For clarity in the prompts shown in the figures, the red content indicates the required visual text, while the blue content denotes the specific attributes (e.g., color, font, style) required for the text. As demonstrated, TextCrafter consistently outperforms these powerful industrial models, particularly in text accuracy and attribute adherence.

[Figure 127]

[Figure 128]

###### System Prompt

# Character Now you are a prompt processing engineer, good at processing all kinds of data, and you are processing prompts.

# Constraints Your current task is to extract the carrier corresponding to each visual text (i.e., the content within single quotes) and the sentence corresponding to the visual text (i.e., the content within single quotes) from the given prompt. The carrier corresponding to the visual text refers to what object the visual text in the prompt is written on, expressed by only one key word(for example, A busy coffee shop with a sign that reads 'Coffee' in bold brown letter, the visual text is 'Coffee', and the corresponding position is sign). The sentence corresponding to the visual text refers to the short sentence composed of the visual text and the carrier in the prompt (for example, A busy coffee shop with a sign that reads 'Coffee' in bold brown letter, the sentence corresponding to the visual text is "a sign that reads 'Coffee' in bold brown letter."). The carrier corresponding to the visual text is output with carriers, and the sentence corresponding to the visual text is output with sentence_list.

# Examples The output should follow the format of the following example: [

{ "prompt": "A laptop screen showing the text 'Login Now' in bold blue letters with a medium size.", "carriers": ["screen"], "sentence_list": [

"A laptop screen showing the text 'Login Now' in bold blue letters with a medium size." ]

}, {

"prompt": "In a bustling train station, a large banner says 'Faster, Greener, Smarter: The AI Train'. The departure board shows 'Express to Tech Valley: 8:30 AM'. A vending machine with the text 'Please Select'.", "carriers": ["banner", "board", "machine"], "sentence_list": [

"a large banner says 'Faster, Greener, Smarter: The AI Train'.", "The departure board shows 'Express to Tech Valley 8:30 AM'.", "A vending machine with the text 'Please Select'."

] }

] Next, I will provide you with some prompts. According to the above requirements, please return the result in JSON-formatted List, including only the three sections: "prompt", "carriers", and "sentence_list".

- Fig. 23 System prompt of fine-grained information.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

###### 2 Regions

###### 3 Regions

"prompt": "On a tech website, the headline says 'Latest Gadgets', a button reads 'Buy Now', and the footer says 'Contact Us'.", "carriers": ["headline", "button", "footer"], "sentence_list": [

"prompt": "On a vibrant city street, a billboard announces 'Join the Fitness Revolution' and a bus stop sign shows 'Next Bus Arrives in 7 Minutes'.", "carriers": ["billboard", "sign"], "sentence_list": [

"the headline says 'Latest Gadgets'.", "a button reads 'Buy Now'.", "the footer says 'Contact Us'."

"a billboard announces 'Join the Fitness Revolution'.", "a bus stop sign shows 'Next Bus Arrives in 7 Minutes'."

] "prompt": "At an art gallery, a wall sign says 'Explore Abstract Art Today' in elegant italic font and a nearby placard reads 'Artist Reception 5 PM Friday' in small cursive.", "carriers": ["sign","placard"], "sentence_list": [

]

"prompt": "In a busy office, a desk calendar shows 'Deadline' in bold black letters, a laptop screen displays 'Work' in medium regular font, and a wall clock reads '12:00' in large numbers.", "carriers": ["calendar", "screen", "clock"], "sentence_list": [

"a desk calendar shows 'Deadline' in bold black letters.", "a laptop screen displays 'Work' in medium regular font.", "a wall clock reads '12:00' in large numbers.“

"a wall sign says 'Explore Abstract Art Today' in elegant italic font.", "a nearby placard reads 'Artist Reception 5 PM Friday' in small cursive."

]

]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

###### 4 Regions

###### 5 Regions

"prompt": "In a high school hallway with lockers labeled 'Student Services Office', a bulletin board displaying 'Upcoming School Events', a classroom door sign reading 'Math 101: Algebra Basics', a gymnasium banner saying 'Annual Sports Day', and a nurse office sign showing 'Health Services Available'.", "carriers": ["lockers", "bulletin board", "sign", "banner", "sign"], "sentence_list": [

"prompt": "In a park, a kiosk sign says 'Information Here', a food stand banner reads 'Cold Drinks', a walking trail marker shows 'Next Stop', and a bench sign displays 'Take a Break'.", "carriers": ["sign", "banner", "marker", "sign"], "sentence_list": [

"a kiosk sign says 'Information Here'.", "a food stand banner reads 'Cold Drinks'.", "a walking trail marker shows 'Next Stop'.", "a bench sign displays 'Take a Break'."

"lockers labeled 'Student Services Office'.", "a bulletin board displaying 'Upcoming School Events'.", "a classroom door sign reading 'Math 101: Algebra Basics'.", "a gymnasium banner saying 'Annual Sports Day'.", "a nurse office sign showing 'Health Services Available'."

] "prompt": "A nature trail with a wooden signpost displaying 'Trail' in medium brown letters, a rock with 'Peak' carved on it in dark gray, a flag with 'Summit' written in bold green, and a backpack with 'Gear' on it in small black letters.", "carriers": ["signpost", "rock", "flag", "backpack"], "sentence_list": [

]

"prompt": "A music album cover, the title says 'Sound Waves' in large blue letters, a track list reads 'Track 1: Intro' in medium white, a label shows 'Produced by XYZ' in small gray, a release date displays 'April 15' in regular black, and a barcode says '123456789' in small black.", "carriers": ["title", "list", "label", "date", "barcode"], "sentence_list": [

"A music album cover, the title says 'Sound Waves' in large blue letters.", "a track list reads 'Track 1: Intro' in medium white.", "a label shows 'Produced by XYZ' in small gray.", "a release date displays 'April 15' in regular black.", "a barcode says '123456789' in small black."

"a wooden signpost displaying 'Trail' in medium brown letters.", "a rock with 'Peak' carved on it in dark gray.", "a flag with 'Summit' written in bold green.", "a backpack with 'Gear' on it in small black letters.“

]

]

###### Fig. 24 Randomly selected samples under different numbers of regions.

[Figure 137]

[Figure 138]

[Figure 141]

Qwen-Image Z-Image

Qwen-Image Z-Image

Qwen-Image Z-Image

[Figure 143]

[Figure 146]

[Figure 148]

Longcat-Image

Longcat-Image

Longcat-Image

Hunyuan-Image

Hunyuan-Image

Hunyuan-Image

[Figure 151]

[Figure 153]

[Figure 154]

GLM-Image TextCrafter

GLM-Image TextCrafter

GLM-Image TextCrafter

Prompt: In a park, a wooden bench has a plaque reading 'Rest Here', a playground sign says 'Play Safe', a park entrance board reads 'Open 6AM-8PM', and a walking trail marker shows 'Trail 3'.

Prompt: A park with a sign reading 'Nature' in green bold letters, a bench displaying 'Relax' in italic, a playground labeled 'Fun' in large colorful letters, a fountain with 'Water' in blue, and a path showing 'Walk' in medium letters.

Prompt: In a high school hallway, a locker door tag reads 'Class of 2025', a bulletin board poster displays 'Join the Club', a classroom door sign shows 'Science Lab', a trophy case label says 'Championships Won', and a hallway clock reads 'Passing Period'.

- Fig. 25 Visual comparison on CVTG-Hard (Sample 1). In the prompt, red indicates the target visual text, and blue

[Figure 155]

[Figure 157]

[Figure 159]

[Figure 160]

Qwen-Image Z-Image

Qwen-Image Z-Image

Qwen-Image Z-Image

[Figure 161]

[Figure 162]

[Figure 164]

[Figure 165]

Longcat-Image

Longcat-Image

Longcat-Image

Hunyuan-Image

Hunyuan-Image

Hunyuan-Image

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 172]

GLM-Image TextCrafter

GLM-Image TextCrafter

GLM-Image TextCrafter

Prompt: In a park, a wooden bench has a plaque reading 'Rest Here', a playground sign says 'Play Safe', a park entrance board reads 'Open 6AM-8PM', and a walking trail marker shows 'Trail 3'.

Prompt: A park with a sign reading 'Nature' in green bold letters, a bench displaying 'Relax' in italic, a playground labeled 'Fun' in large colorful letters, a fountain with 'Water' in blue, and a path showing 'Walk' in medium letters.

Prompt: In a high school hallway, a locker door tag reads 'Class of 2025', a bulletin board poster displays 'Join the Club', a classroom door sign shows 'Science Lab', a trophy case label says 'Championships Won', and a hallway clock reads 'Passing Period'.

- Fig. 26 Visual comparison on CVTG-Hard (Sample 2). In the prompt, red indicates the target visual text, and blue indicates the required attributes.

[Figure 173]

[Figure 174]

[Figure 176]

Qwen-Image Z-Image

Qwen-Image Z-Image

Qwen-Image Z-Image

[Figure 179]

[Figure 180]

[Figure 183]

Longcat-Image

Longcat-Image

Longcat-Image

Hunyuan-Image

Hunyuan-Image

Hunyuan-Image

[Figure 185]

[Figure 186]

[Figure 189]

GLM-Image TextCrafter

GLM-Image TextCrafter

GLM-Image TextCrafter

Prompt:在一处宁静的海滩上，一家冲浪店的橱窗招牌上写着“冲 浪板出售”，一座救生员瞭望塔上飘扬着一面写着“安全水域，请 小心游泳”的旗帜，一间租赁小屋上标有“全天提供摩托艇租赁”，

Prompt:在一间高科技办公室的大厅里，接待台的标牌上写着“欢 迎来到TechCorp”，一块数字屏幕显示着“2024年公司成就”，等 候区的宣传册架上标有“公司服务”，会议室的门上则显示“会议室 A - 使用中”。

Prompt: A bustling street corner with a neon sign saying 'Pizza' in bright red bold letters, a bus stop displaying '10:15' in large white digits, a store window featuring 'Sale' in vibrant yellow cursive, and a passing car with 'Fast' written on it in italic blue.

一座沙雕上刻着“度假模式：开启”，而一家海滨餐厅的横幅则写 着“每日新鲜海鲜”。

- Fig. 27 Visual comparison on CVTG-Hard (Sample 3). In the prompt, red indicates the target visual text, and blue

[Figure 193]

[Figure 195]

[Figure 196]

Qwen-Image Z-Image

Qwen-Image Z-Image

Qwen-Image Z-Image

[Figure 199]

[Figure 200]

[Figure 202]

Longcat-Image

Longcat-Image

Longcat-Image

Hunyuan-Image

Hunyuan-Image

Hunyuan-Image

[Figure 205]

[Figure 206]

[Figure 208]

GLM-Image TextCrafter

GLM-Image TextCrafter

GLM-Image TextCrafter

Prompt:在一处宁静的海滩上，一家冲浪店的橱窗招牌上写着“冲 浪板出售”，一座救生员瞭望塔上飘扬着一面写着“安全水域，请 小心游泳”的旗帜，一间租赁小屋上标有“全天提供摩托艇租赁”，

Prompt:在一间高科技办公室的大厅里，接待台的标牌上写着“欢 迎来到TechCorp”，一块数字屏幕显示着“2024年公司成就”，等 候区的宣传册架上标有“公司服务”，会议室的门上则显示“会议室 A - 使用中”。

Prompt: A bustling street corner with a neon sign saying 'Pizza' in bright red bold letters, a bus stop displaying '10:15' in large white digits, a store window featuring 'Sale' in vibrant yellow cursive, and a passing car with 'Fast' written on it in italic blue.

一座沙雕上刻着“度假模式：开启”，而一家海滨餐厅的横幅则写 着“每日新鲜海鲜”。

- Fig. 28 Visual comparison on CVTG-Hard (Sample 4). In the prompt, red indicates the target visual text, and blue

[Figure 209]

[Figure 211]

[Figure 212]

Qwen-Image Z-Image

Qwen-Image Z-Image

[Figure 213]

[Figure 214]

[Figure 215]

Longcat-Image

Longcat-Image

Hunyuan-Image

Hunyuan-Image

[Figure 218]

[Figure 219]

[Figure 220]

GLM-Image TextCrafter Prompt:一本书的封面，书名以醒目的亮蓝色大字写着“夏日逃离”，

GLM-Image TextCrafter

Prompt:在一个宽敞的体育馆内，一块记分牌以醒目的数字显示 “最终比分：主队 102 - 客队 98”，一扇更衣室门上用醒目的蓝色 字体标有“仅限运动员”，公告栏上用中等大小的绿色字体张贴着 “即将开班的健身课程”，饮水机旁的标牌则用小号斜体字写着“在 此处保持水分补充”。

作者名“A. 布鲁克斯”以小号常规字体位于底部，宣传语以中等大 小的绿色斜体写着“放松并享受”，价格标签以小号红色字体显示 “12.99 美元”，条形码以常规字体显示“978-1234567890”。

- Fig. 29 Visual comparison on CVTG-Hard (Sample 5). In the prompt, red indicates the target visual text, and blue

[Figure 221]

[Figure 222]

[Figure 223]

Qwen-Image Z-Image

Qwen-Image Z-Image

[Figure 226]

[Figure 227]

[Figure 228]

Longcat-Image

Longcat-Image

Hunyuan-Image

Hunyuan-Image

[Figure 229]

[Figure 231]

[Figure 232]

GLM-Image TextCrafter Prompt:一本书的封面，书名以醒目的亮蓝色大字写着“夏日逃离”，

GLM-Image TextCrafter

Prompt:在一个宽敞的体育馆内，一块记分牌以醒目的数字显示 “最终比分：主队 102 - 客队 98”，一扇更衣室门上用醒目的蓝色 字体标有“仅限运动员”，公告栏上用中等大小的绿色字体张贴着 “即将开班的健身课程”，饮水机旁的标牌则用小号斜体字写着“在 此处保持水分补充”。

作者名“A. 布鲁克斯”以小号常规字体位于底部，宣传语以中等大 小的绿色斜体写着“放松并享受”，价格标签以小号红色字体显示 “12.99 美元”，条形码以常规字体显示“978-1234567890”。

Fig. 30 Visual comparison on CVTG-Hard (Sample 6). In the prompt, red indicates the target visual text, and blue

