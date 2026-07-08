## Think Visually, Reason Textually: Vision-Language Synergy in ARC

# arXiv:2511.15703v2[cs.CV]26Nov2025

Beichen Zhang1,2, Yuhang Zang2 , Xiaoyi Dong1,2, Yuhang Cao2 Haodong Duan2, Dahua Lin1,2, Jiaqi Wang2,3

1The Chinese University of Hong Kong 2Shanghai AI Laboratory 3Shanghai Innovation Institute

{zhangbeichen,zangyuhang}@pjlab.org.cn Corresponding Authors.

### Abstract

Abstract reasoning from minimal examples remains a core unsolved problem for frontier foundation models such as GPT-5 and Grok 4. These models still fail to infer structured transformation rules from a handful of examples, which is a key hallmark of human intelligence. The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) provides a rigorous testbed for this capability, demanding conceptual rule induction and transfer to novel tasks. Most existing methods treat ARC-AGI as a purely textual reasoning task, overlooking the fact that humans rely heavily on visual abstraction when solving such puzzles. However, our pilot experiments reveal a paradox: naively rendering ARC-AGI grids as images degrades performance due to imprecise rule execution. This leads to our central hypothesis that vision and language possess complementary strengths across distinct reasoning stages: vision supports global pattern abstraction and verification, whereas language specializes in symbolic rule formulation and precise execution. Building on this insight, we introduce two synergistic strategies: (1) Vision-Language Synergy Reasoning (VLSR), which decomposes ARC-AGI into modality-aligned subtasks; and (2) Modality-Switch SelfCorrection (MSSC), which leverages vision to verify textbased reasoning for intrinsic error correction. Extensive experiments demonstrate that our approach yields up to a 4.33% improvement over text-only baselines across diverse flagship models and multiple ARC-AGI tasks. Our findings suggest that unifying visual abstraction with linguistic reasoning is a crucial step toward achieving generalizable, human-like intelligence in future foundation models. Source code is released at https://github.com/ InternLM/ARC-VL.

### 1. Introduction

The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) [4] has emerged as the lead-

[Figure 1]

Figure 1. We propose vision-text co-reasoning in abstract reasoning tasks. It integrates the unique advantages of visual and textual thinking, thereby outperforming uni-modal reasoning. All methods use o4-mini as the base model.

ing benchmark for evaluating machine intelligence beyond domain-specific skills. Unlike traditional AI benchmarks that focus on narrow tasks such as natural language processing or image understanding, ARC-AGI measures the ability to learn how to learn: a system must identify abstract transformation rules from minimal examples and apply them to entirely new scenarios. This evaluation paradigm mirrors human IQ tests and represents a fundamental shift toward assessing artificial general intelligence. Consequently, state-of-the-art models (e.g., GPT-5 [21] and Grok 4 [36]) now highlight ARC-AGI performance as a key indicator of their reasoning capabilities.

Despite this importance, progress on ARC-AGI remains limited. An intriguing observation is that existing approaches treat ARC-AGI as a purely textual task, representing the input-output matrix pairs as nested lists (e.g., [[0,1,2],[3,4,5]]) during both training and inference; see the left part of Fig. 2. This design choice, while computationally convenient, fundamentally contradicts human problemsolving intuition. As shown in the right part of Fig. 2, when humans approach ARC-AGI tasks, they naturally visualize the patterns: a color-coded 2D grid immediately reveals spatial relationships like symmetries, rotations, or shape transformations that are tedious to infer from textual coor-

[Figure 2]

[..., [..., 0, 4, 0, ...], [..., 0, 4, 0, ...], ...] [..., [..., 0, 4, 4, ...], [..., 0, 4, 0, ...], ...] [..., [..., 0, 2, 0, ...], [..., 0, 0, 0, ...], ...]

- [..., [..., 0, 2, 0, ...], [..., 2, 2, 2, ...], ...] [..., [..., 5, 5, 0, ...], [..., 5, 0, 0, ...], ...] [..., [..., 5, 5, 0, ...], [..., 5, 0, 5, ...], ...]

- [..., [..., 1, 0, 0, ...], [..., 1, 0, 0, ...], ...] [..., [..., ?, ?, ?, ...], [..., ?, ?, ?, ...], ...]

Input-Output Examples

Test

- Figure 2. Textual (left) vs. Visual (right) Thinking in the ARC-AGI Task. Previous work treats ARC-AGI as a pure text task for training and reasoning, as text allows for a precise representation of each element. However, this approach loses the intuitiveness of visual thinking and 2D structural information. In contrast, we organically integrate visual thinking and textual thinking into the ARC-AGI reasoning process, using the complementary strengths of different modalities.

dinate descriptions. We argue that this discrepancy between human visual intuition and machine text-centric processing represents a critical missed opportunity.

However, integrating visual information is not straightforward. Our preliminary experiments reveal a counterintuitive paradox: naively rendering ARC-AGI grids as images actually degrades performance compared to textonly baselines. While image-based representations capture global 2D structure effectively, they struggle with precise element-wise operations. For instance, when a 20 × 20 grid is presented as an image, models often fail to reliably identify or manipulate the value at a specific position such as (5,7), conflating it with nearby cells. This highlights a fundamental tension: vision excels at recognizing overall spatial patterns, whereas textual encodings naturally provide the discrete precision required for exact rule execution.

This observation leads to our core insight: visual and textual modalities have fundamentally complementary strengths in different stages of abstract reasoning. To validate this hypothesis, we systematically decompose ARC-AGI into two sub-tasks: rule summarization (extracting transformation patterns from examples) and rule application (applying the extracted rule to new inputs). We empirically evaluate each modality’s performance on these sub-tasks (detailed in Sec. 3.2). Our analysis of the OpenAI o4-mini model reveals striking differences: vision excels at rule summarization, providing a 3.0% improvement through its holistic perception of 2D spatial structures, while text excels at rule application, with vision causing a dramatic 20.5% performance drop due to imprecise element-wise manipulation. These findings demonstrate that the question is not whether to use vision or text, but rather when and how to strategically combine them.

Guided by our insights, we propose two synergistic strategies that strategically integrate visual and textual modalities throughout the abstract reasoning pipeline:

Vision-Language Synergy Reasoning (VLSR) matches each sub-task to its optimal modality. During rule summarization, VLSR visualizes the example input-output matrix pairs as color-coded 2D grids, enabling the model to use holistic spatial perception and efficiently encode global transformation patterns (e.g., “all shapes rotate 90 degrees clockwise”). During rule application, VLSR switches to textual representation, allowing the model to perform precise element-wise manipulations guided by the extracted rule. This modality-aware decomposition achieves improvements through two mechanisms: (1) divide-andconquer task decomposition reduces individual sub-task complexity, and (2) strategic modality selection exploits each modality’s inherent strengths.

Modality-Switch Self-Correction (MSSC) addresses a fundamental challenge in intrinsic self-correction: models struggle to identify errors when verifying their own reasoning in the same modality [12, 39]. MSSC breaks this limitation by employing different modalities for forward reasoning and backward verification. After generating a candidate output through text-based rule application, MSSC visualizes both the test input and predicted output as images, then uses the visual modality’s strength in pattern consistency verification to check whether the predicted transformation matches the pattern shown in the example images. If inconsistencies are detected, the model receives explicit feedback and performs another round of textual inference. This crossmodal verification enables effective intrinsic self-correction without any external information or ground truth.

Extensive experiments demonstrate that our combined approach achieves substantial improvements. When applied to flagship reasoning models such as Gemini-2.5-Pro and o4-mini, it delivers notable improvements of up to 7.25%and 4.5% accuracy respectively on the official ARCAGI evaluation set. On average, our approach yields up to a 4.33% improvement over text-only baselines across diverse

models (GPT-4o, Gemini-2.5-Pro, o4-mini, Qwen3-VL) and multiple ARC-AGI benchmarks (ARC-AGI, BARC, Re-ARC).

Furthermore, our analysis reveals that text-only selfcorrection often fails or degrades performance, while MSSC provides consistent iterative improvements. Our work makes the following contributions:

- 1) We provide the first systematic study of visual versus

textual reasoning in ARC-AGI, identifying four key characteristics of their complementary strengths: holistic vs. independent processing, 2D structure preservation, encoding efficiency, and element-wise precision trade-offs, which is a novel perspective for the exploration of visual intelligence.

- 2) We introduce VLSR and MSSC, two training-

free strategies that strategically combine visual and textual modalities to exploit their complementary strengths throughout the reasoning pipeline.

- 3) We demonstrate that our proposed vision-language

synergy principle extends naturally to the training paradigm. By fine-tuning separate models for visual rule summarization and textual rule application, our approach achieves a 3.5% improvement over text-only fine-tuning on the same training data, enabling small open-source models (Qwen3-8B) to surpass closed-source models like GPT-4o.

### 2. Related Work

ARC-AGI Tasks. ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence) is a benchmark task designed to evaluate the generalization capability of AI systems. Its core objective is to measure whether an AI can abstract rules from a minimal number of examples and solve entirely new problems, rather than relying on largescale datasets or pre-trained knowledge. Each task consists of several input-output matrix pairs. The model is required to infer rules from a small set of demonstrations and apply these rules to an unseen test input matrix. For humans, ARC-AGI tasks are not difficult, with an accuracy rate of over 97%. However, they remain highly challenging for AI systems and have become one of the most demanding benchmarks currently used to assess whether an AI possesses artificial general intelligence.

ARC-AGI Strategies. The Abstraction and Reasoning Corpus (ARC-AGI) task [4] has garnered widespread attention in the academic community since its proposal. Most strategies attempt to enhance ARC-AGI capabilities through training. A number of approaches [3, 8, 10, 15– 17, 19, 20, 23, 31, 32, 37] generate a large number of ARCAGI tasks by permuting and combining predefined transformation rules, and use this synthetic data to fine-tune large language models. Additionally, test-time training [27] is a prevalent strategy. These approaches [1, 5, 7, 24, 40, 41] treat the input-output examples provided during testing not only as context to support the models’ reasoning, but also

as data for additional fine-tuning of model before it generates answers. Other strategies attempt to introduce additional hints during the inference process to enhance the model’s performance on the ARC-AGI task. A common type of strategy [18, 25, 26, 30, 34] leverages the characteristics of ARC-AGI tasks by predefining some of the potential transformation rules and then provides prompts and guidance to the model during the reasoning process. Another common strategy [2, 9, 14, 24, 28, 29] involves utilizing the model’s memory: summarizing observations and insights from previous problems. This strategy consolidates previous attempts into referenceable concepts to guide the model in new reasoning tasks. However, previous strategies have treated ARC-AGI as a pure text task for both training and reasoning, neglecting image representations that demonstrate the rules more intuitively. Therefore, the core work of this paper is to integrate visual information into ARC-AGI and analyze the differences between visual thinking and textual thinking in abstract reasoning.

Aiding Reasoning with Images. Leveraging visual modalities to augment reasoning has emerged as a pivotal research direction. Visual Sketchpad [11] enhances geometric problem-solving by rendering diagrams and introducing auxiliary constructions. ViLaSR [35] further proposes a “Drawing to Reason in Space” paradigm that permits the model to visualize its intermediate spatial hypotheses, achieving measurable improvements on spatial-reasoning benchmarks like maze solving. Owing to its immediacy and spatial expressiveness, visual information is becoming an indispensable component of reasoning pipelines.

### 3. Method

The core of this work lies in integrating visual thinking into ARC-AGI tasks to use the complementary strengths of different modalities. While existing approaches treat ARCAGI as a pure text task, we argue that visual and textual modalities excel at different aspects of the reasoning process. Visual thinking provides global perception and 2D structural understanding that text-centric reasoning lacks, while textual thinking enables precise element-wise processing and manipulation.

To this end, we propose two core methods for ARC-AGI (Fig. 3): a) Vision-Language Synergy Reasoning (VLSR) and b) Modality-Switch Self-Correction (MSSC). VLSR summarizes rules from exemplar matrices in the visual modality and then applies the inferred rule to the test input in the text modality. MSSC verifies the textual output by re-encoding it as an image to assess visual consistency with the examples; if inconsistent, it triggers another textual pass with targeted feedback.

We first formalize the ARC-AGI task and establish notation in Sec. 3.1. Before presenting our method design, we conduct a systematic empirical analysis in Sec. 3.2 to under-

a) Vision-Language Synergy Reasoning b) Modality-Swith Self-Correction

Apply the rule

Forward Reasoning

[Figure 3]

Summarize the rule

[Figure 4]

[Figure 5]

[Figure 6]

Rule: The green ...

[Figure 7]

The Result is [0, 0, 3, 0, 0, ...], [0, 3, 4, 3, 3, ...],

Input: [0, 0, 3, 0, ...], [0, 3, 0, 3, ...],

[Figure 8]

The green regions have their enclosed areas changed to yellow, while the remaining parts unchanged

Visualize

edback

[0, 0, 3, 0, 0, ...], [0, 3, 4, 3, 3, ...],

[Figure 9]

In 1

In 2

Backward Reflection

F

Follows the same rule?

[Figure 10]

Result

Judge

The images demonstrate the rule of change clearly

The texts support perelement modification.

[Figure 11]

[Figure 12]

、

##### ... ...

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Rule Application

Rule Summarization

Example

Pred

Out 1

Out 2

- Figure 3. Overview of our method. a) Vision-Language Synergy Reasoning decomposes ARC-AGI into two subtasks: Rulesummarization and Rule-application. The former visualizes the provided example matrices as images, using global visual perception and 2D structure to summarize the rule. The latter requires element-wise processing, so rule-application is carried out in the textual modality. b) Modality-Switch Self-Correction visualizes the output matrix to judge rule consistency. The results are fed back to implement the self-correction strategy if necessary. As visual information is more informative in rule verification, the model can repeatedly refine its answers without relying on additional inputs.

stand the respective strengths of visual and textual modalities in ARC-AGI reasoning. This provides the empirical foundation for our design choices. Then we present the detailed design of VLSR in Sec. 3.3 and MSSC in Sec. 3.4.

###### 3.1. Problem Setup and Notation

Task Formulation. Each ARC-AGI task is associated with a specific transformation rule r that maps input matrices minput to output matrices moutput. The model is provided with K example pairs: {(minput1 ,moutput1 ),(minput2 ,moutput2 ),...,(minputK ,moutputK )},

where each matrix m ∈ ZH×W has H,W ≤ 30 and cell values in {0,1,...,9}. The goal is to infer the underlying rule r from these examples and apply it to a new test input minputtest to generate the corresponding output moutputtest .

Notation. We denote matrices in different modalities as follows: 1) m: the matrix, in any modality. 2) t = T (m): textual representation of the matrix (e.g., nested list: [[0,1,2],[3,4,5],[2,3,5]]). 3) i = V(m): visual representation of the matrix, where each cell value (0-9) is mapped to a distinct color in a grid layout. Details of the visualization process are provided in Sec. B in the supplementary materials.

The symbols T and V denote the text and visual transformation functions, respectively. Both transformations are invertible: T −1(t) = m and V−1(i) = m, allowing seamless conversion between modalities.

Previous Approaches. Prior works process ARC-AGI tasks purely in the textual modality. Besides, the transformation rule r is not explicitly derived, either implicitly outputting it in the chain-of-thought, or skipping it directly to output the transformed matrix.

Table 1. Quantitative experiments on using different modalities in rule summarization and application phase.

Rule-Sum. Rule-App. text vision text vision

Models Baseline

GPT-4o 8.25 10.5 13.5 13.5 6.25 Gemini-2.5 35.0 35.25 38.75 38.75 23.75 o4-mini 42.25 42.5 45.5 45.5 25.0

Formally, previous methods optimize a function f to directly predict the output matrix in textual form:

tpred = f(tinput1 ,toutput1 ,··· ,tinputK ,toutputK ,tinputtest ). (1)

This has inherent limitations: (1) textual representation loses crucial 2D structural information, and (2) conflating rule summarization and application in a single step prevents the model from fully leveraging modality-specific strengths.

###### 3.2. Comparative Analysis of Vision and Text

To inform our method design, we conduct a systematic empirical analysis of visual versus textual modalities in ARCAGI reasoning. We decompose the task into two subtasks: rule summarization and rule application, and evaluate each modality’s performance on these subtasks separately across multiple state-of-the-art models.

3.2.1. Quantitative Comparison Experimental Design. To isolate the effect of modality choice on each sub-task, we conduct experiments where we vary the modality used in either the rule summarization or rule application phase while keeping other factors constant. Rule Summarization Phase. We input the provided example matrix pairs {(minput1 ,moutput1 ),...,(minputK ,moutputK )}

into the same model using either textual representations {(tinputi ,toutputi )} or visual representations {(iinputi ,ioutputi )}, and require the model to summarize the transformation rule r. To fairly compare the quality of the extracted rules, we then uniformly apply both rules in the textual modality and compare the final accuracy on the test matrix.

Rule Application Phase. For this phase, we adopt the same high-quality rule r derived from the visual modality (as validated by the previous experiment). The core comparison is whether it is better to represent both the example matrices and the test input matrix as images or as text when applying the rule to generate the output.

Baseline. We also include the standard text-only approach commonly used in previous studies, which relies solely on the textual modality without explicit rule extraction, directly outputting the transformed test matrix.

Results. As shown in Tab. 1, using the visual modality for rule summarization provides a clear advantage, yielding an average improvement of 3.2% across models (e.g., 40.75% vs. 37.25% for Gemini-2.5). However, this effect is reversed in the rule application phase. When applying rules using visual representations instead of textual ones, performance drops dramatically by an average of 15.0% (e.g., from 40.75% to 23.75% for Gemini-2.5).

Our results clearly demonstrate that visual and textual modalities have complementary strengths: vision excels at global pattern recognition needed for rule summarization, while text excels at precise element-wise manipulation needed for rule application.

###### 3.2.2. Qualitative Analysis: Understanding the “Why”

To understand the underlying reasons behind our quantitative findings, we conduct an in-depth qualitative analysis of model outputs across both modalities. We identify four key characteristics that explain the performance differences:

- 1) Visual thinking adopts holistic perception; textual thinking processes elements independently. Visual reasoning demonstrates a systematic bias toward encoding relational properties anchored on contiguous spatial structures, such as central blocks, checkerboard patterns, or connected components. In contrast, textual reasoning relies more on type-level statistics (e.g., frequency counts) to identify patterns, treating each element more independently. For rule summarization that requires identifying global spatial relationships, the holistic nature of visual thinking aligns better with task requirements and mirrors human cognitive strategies.
- 2) Visual thinking preserves 2D structure; textual thinking may lose spatial information. In a matrix, two vertically adjacent elements in the same column are perceptually contiguous in a visual representation, yet may be separated by dozens of tokens in a textual representation (e.g., [[0,1,2],[3],4,5]] where elements in different rows are far apart in the token sequence). Consequently, rules extracted

- through textual reasoning tend to lack 2D structural characteristics and perform poorly when tasks require capturing inter-row or diagonal regularities. Moreover, when both input and output matrices are transposed, rules discovered via visual representation remain essentially invariant, while those derived from textual representation may be significantly affected by the change in token ordering.
- 3) Visual representation is more efficient for large matrices. Textual representation requires separately encoding each matrix element along with delimiters (brackets, commas). For large matrices (e.g., 30 × 30), textual representation may require thousands of tokens. In contrast, visual representation encodes even complex matrices using a single image with only a few hundred vision tokens. This enables faster reasoning without compromising or even enhancing the reasoning quality. Recent work DeepSeekOCR [33] also explores visual-based approaches for compressing complex document understanding.
- 4) Visual reasoning lacks fine-grained element-wise precision. Images represent matrices as an integrated whole rather than encoding individual elements separately. While this provides a global perspective advantageous for rule summarization, it becomes inadequate when element-wise processing is required during rule application. We observe that when images represent large matrices, models may make errors in basic element localization and value identification (e.g., confusing the value at position (5,7) with a nearby cell). This limitation leads to performance degradation when visual thinking is adopted for the rule application, which demands precise per-element manipulation. Summary. Our analysis reveals that visual and textual modalities have fundamentally different information processing characteristics. Visual thinking provides global perception, preserves 2D spatial structure, and offers efficient representation, making it ideal for rule summarization. Textual thinking enables precise element-wise access and manipulation, making it essential for rule application. These complementary strengths motivate our method design in the following subsections.

###### 3.3. Vision-Language Synergy Reasoning

Motivation. We now translate the empirical insights from Sec. 3.2 into our pipeline design. Rather than forcing a single modality throughout the reasoning process, we introduce Vision-Language Synergy Reasoning that strategically switches modalities between subtasks. Our key design principle is to route each sub-task to its optimal modality: visual reasoning for rule summarization (exploiting global pattern recognition and 2D structural understanding) and textual reasoning for rule application (exploiting precise elementwise manipulation). This modality-aware decomposition allows us to harness the full potential of LVLMs.

Method Overview. Fig. 3 (a) presents our Vision-

Language Synergy Reasoning (VLSR) pipeline, which strategically employs appropriate modalities at different stages. The complete pipeline consists of two phases:

- Phase 1: Visual Rule Summarization. We convert all example matrix pairs into visual representations. Each matrix m is visualized as an image i = V(m), where V is a visualization function that maps each cell value to a distinct color in a grid layout. The LVLM then analyzes these visualized examples to derive an explicit transformation rule:

rpred = fsumvision(iinput1 ,ioutput1 ,iinput2 ,ioutput2 ,...,iinputK ,ioutputK ),

(2) where fsumvision represents the LVLM operating in visual mode with a rule summarization prompt. The derived rule rpred in Eq. (2) is expressed in natural language, describing the transformation pattern (e.g., “rotate each connected component 90 degrees clockwise”).

- Phase 2: Textual Rule Application. Given the summarized

rule rpred, we apply it to the test input using textual representations. All matrices are converted to text format, and the model performs element-wise reasoning:

tpred = fapptext(rpred,tinput1 ,toutput1 ,...,tinputK ,toutputK ,tinputtest ), (3)

where fapptext represents the same LVLM operating in text mode with a rule application prompt. Note that fsumvision and fapptext are the same base model; only the input modality and prompting strategy differ.

Key Advantages. Compared with Eq. (1), our approach offers two key benefits: (1) Task decomposition reduces the complexity of individual subtasks through a divide-andconquer strategy, and (2) Modality matching ensures each sub-task uses the optimal modality: visual for global pattern recognition, textual for precise manipulation. As we will demonstrate in Sec. 4, our strategy yields consistent improvements across different models and benchmarks.

###### 3.4. Modality-Switch Self-Correction

Motivation. Self-correction is a cornerstone of human intelligence, yet it remains challenging for LLMs. The fundamental paradox is: if a model can identify and fix its own errors, why not generate the correct answer initially? Existing works [12, 39] have shown that intrinsic self-correction (without external feedback) is difficult because models struggle to distinguish correct from incorrect outputs when using the same reasoning modality.

We propose Modality-Switch Self-Correction (MSSC), which achieves effective intrinsic self-correction by employing different modalities for reasoning and verification. The key insight is that visual and textual modalities have complementary verification capabilities: while text excels at forward rule application, vision excels at pattern consistency verification. By switching modalities, the model gains a fresh perspective that enables it to identify errors that are imperceptible in the original modality.

Method Design. Fig. 3 (b) shows the pipeline. After obtaining a candidate output tpred from textual rule application in Eq. (3), we perform the following iterative refinement:

- Step 1: Visualization. Convert the test input-output pair to

visual form. Since the predicted output tpred from Eq. (3) is in textual format (nested list), we first parse it back to matrix form mpred, then apply the visualization function V:

iinputtest = V(tinputtest ), ipred = V(tpred), (4)

- Step 2: Visual Consistency Verification. Present the visualized test pair with the examples to the LVLM as a critic:

sconsistent = fcriticvision(iinput1 ,ioutput1 ,...,iinputK ,ioutputK ,iinputtest ,ipred), (5)

where fcriticvision assesses whether the test pair (iinputtest ,ipred) follows the same transformation pattern as the examples. The

output sconsistent ∈ {yes,no} indicates pattern consistency.

- Step 3: Iterative Refinement. If sconsistent = no, the model receives feedback about the inconsistency and performs another round of textual inference with error awareness:

tpred ← finftext(rpred,tinput1 ,toutput1 ,...,tinputtest ,feedbackprev),

(6) where feedbackprev contains information about the previous attempt. This process repeats until consistency is achieved or the iteration limit is reached (we use Nmax = 3).

Key Advantages. MSSC provides two critical benefits: (1) Fresh perspective: switching to visual verification breaks the model’s confirmation bias that occurs when checking its own textual reasoning, and (2) No external information needed: unlike traditional self-correction that requires ground truth or external critics, MSSC uses the model’s own multimodal capabilities. As we show in Sec. 4, text-only self-correction often fails or even degrades performance, while MSSC achieves consistent improvements.

### 4. Experiments

###### 4.1. Experimental setup

Models. We evaluate both open-source and closed-source models. For open-source models, we use Qwen3-VL-235BA22B-Instruct [38]. For closed-source models, we use GPT-4o [13], Gemini-2.5-pro-thinking-8192 [6], and o4mini [22].

Benchmarks. We evaluate on three ARC-AGI benchmarks. (1) the official 400-task ARC-AGI [4] evaluation set, (2) 100 randomly sampled tasks from Re-ARC [10] and (3) 100 randomly sampled tasks from BARC [17]. For Re-ARC and BARC, each sampled task contains four input-output pairs: three examples and one test instance.

Implementation Details. We report Pass@1 accuracy across all experiments at a temperature 0.7. Our prompts are provided in Sec. A in the supplementary materials.

- Table 2. Visual-Language co-reasoning outperforms singlemodality reasoning. Both Vision-Language Synergy Reasoning (VLSR) and Modality-Switch Self-Correction (MSSC) improve over text-only baseline reasoning across models and benchmarks. The combination of both strategies yields the largest gains.

Models ARC-AGI BARC-100 Re-ARC

GPT-4o 8.25 28.0 10.0 +VLSR 13.5 32.0 13.0 +MSSC 12.0 30.0 14.0 +both (Ours) 14.5 33.0 16.0

Gemini-2.5-Pro 35.0 56.0 30.0 +VLSR 38.75 58.0 32.0 +MSSC 36.5 57.0 30.0 +both (Ours) 42.25 60.0 33.0

o4-mini 42.25 59.0 36.0 +VLSR 45.5 64.0 38.0 +MSSC 44.75 62.0 38.0 +both (Ours) 46.75 65.0 39.0

Qwen3-VL-235B 20.25 52.0 20.0 +VLSR 22.0 52.0 21.0 +MSSC 21.75 54.0 21.0 +both (Ours) 22.25 54.0 23.0

- Table 3. Comparison with training-free inference-time methods. All methods use o4-mini as the base model. Our VLSR+MSSC approach outperforms memory-augmented textonly strategies that retrieve past problem-solving experiences.

Method ARC-AGI ARC-AGI-100 Re-ARC

Direct Reason 40.5 41.0 36.0 Cheatsheet 38.5 41.0 34.0 ArcMemo-PS 45.25 45.0 39.0 Ours 46.75 46.0 39.0

- 4.2. Main Results

Overall Performance. Tab. 2 shows that both VLSR and MSSC consistently improve performance across all models and benchmarks. VLSR, which uses visual reasoning for rule summarization, improves baseline text-only reasoning by an average of 3.02%. MSSC, which uses visual verification for self-correction, provides an additional 1.82% improvement on average. Combining both strategies yields the largest gains, improving baseline performance by up to 6.25% for GPT-4o and 7.25% for Gemini-2.5-Pro on ARCAGI. These consistent improvements across diverse models demonstrate that visual information provides complementary benefits for both the rule summarization and the iterative refinement stages of abstract reasoning.

Comparison with Training-Free Reasoning Methods. We compare against two recent training-free strategies that enhance ARC-AGI reasoning at inference time: 1) Dy-

[Figure 21]

<CoT>

Count how many times each number appears Select the numbers with the highest counts <CoT>

Text-Only Reasoning

<CoT> Identifying larger blocks of similar colors and consolidating them into a simplified grid pattern, removing the noise. <CoT>

Vision-Centric Rule Summarization

Visualized Example

Figure 4. Qualitative comparison of text-only vs. visionlanguage synergy reasoning on GPT-4o. Text-only reasoning processes elements without spatial context, leading to an incorrect rule. Vision-language synergy reasoning uses global 2D perception in the rule-summarization phase to identify the correct spatial pattern (“retain large connected color blocks”).

Table 4. Iterative self-correction comparison. We apply Text-Only Self-Correction (TOSC) and Modality-Switch SelfCorrection (MSSC) for up to three rounds (R1-R3) without external feedback. MSSC achieves consistent iterative improvement while TOSC stagnates or degrades.

TOSC MSSC R1 R2 R3 R1 R2 R3

Models Base

GPT-4o 8.25 8.25 8.0 8.75 10.25 11.5 12.0 Gemini 35.0 34.25 36.0 35.75 35.75 36.25 36.5

- o4-mini 42.25 42.5 42.0 43.25 43.5 44.25 44.75

namic Cheatsheet [28] stores strategies and findings from past problem-solving processes as memory and includes them in prompts for new problems. 2) ArcMemo-PS [9] builds concept-level external memory through program synthesis and extracts reusable modular abstract concepts, and selectively retrieves relevant concepts during reasoning. Both methods are text-centric.

Using o4-mini as the base model, we evaluate all meth-

- ods on three test sets: the full ARC-AGI-400, the ARCAGI-100 subset from ArcMemo [9], and the Re-ARC test set we divided. As shown in Tab. 3, our method achieves the highest accuracy across all three benchmarks, outperforming the strongest baseline ArcMemo-PS by 1.5% on ARCAGI. Notably, while ArcMemo-PS and Cheatsheet use retrieved past experiences for text-only reasoning, they cannot access the global 2D structure and spatial pattern information that visual representations provide. Our results suggest that visual information offers complementary benefits that text-based memory retrieval alone cannot capture. 4.3. Analysis

Modality-Switch vs. Text-Only Self-Correction Tab. 4 compares traditional text-only self-correction (TOSC) with our Modality-Switch Self-Correction (MSSC) across three iterative rounds. TOSC shows minimal improvement and even degrades performance in some iterations. For GPT4o, TOSC improves by only 0.5 points (8.25→8.75) across

Table 5. Experiments on fine-tuning Results. Vision-Language Synergy Fine-tuning outperforms both text-only fine-tuning (using the same training tasks) and several closed-source and open-source baseline models with much larger parameter sizes.

Models ARC-AGI BARC-100 Re-ARC Text-only Reasoning Baseline

GPT-4o 8.25 28.0 10.0 Gemini-2.5-Pro 35.75 56.0 30.0 o4-mini 42.25 59.0 36.0 Qwen3-235B-A22B 20.25 52.0 20.0 Qwen3-8B 3.25 13.0 2.0 +Text-only FT 9.75 37.0 7.0 Improvement +6.5 +24.0 +5.0

Vision-Language Synergy Reasoning

Qwen3-VL-8B + Qwen3-8B 3.5 13.0 3.0 +VL Synergy FT 13.25 43.0 9.0 Improvement +9.75 +30.0 +6.0

three rounds, with Round 2 degrading to 8.0. In contrast, MSSC achieves consistent monotonic gains at each round. For GPT-4o, MSSC improves from 8.25 to 10.25 (+2.0) in Round 1, then to 11.5 (+1.25) in Round 2, and finally to 12.0 (+0.5) in Round 3. Similar trends hold for Gemini (+1.5 total) and o4-mini (+2.5 total).

We attribute this difference to modality switching: verifying textual outputs with visual representations provides a fresh perspective that helps models identify spatial inconsistencies imperceptible when reasoning solely in text. When TOSC uses the same textual modality for both generation and verification, the model exhibits confirmation bias and cannot effectively spot its own errors. By contrast, MSSC’s visual verification stage detects pattern violations (e.g., missing symmetry, incorrect spatial relationships) that the textual reasoning stage overlooked, enabling genuine iterative improvement without external feedback.

Qualitative Analysis. Fig. 4 presents how visual reasoning corrects systematic errors made by text-only reasoning on GPT-4o. When presented with textual matrix representations, GPT-4o incorrectly summarizes the transformation rule as “select the number with the highest count”. This frequency-based heuristic fails because the task actually requires identifying large connected spatial structures. With visualized matrices, the model correctly identifies the rule as “retain large connected color blocks” by using global 2D perception to recognize spatial contiguity patterns. Results show that visual representation enables holistic pattern recognition. Additional examples are provided in Sec. C in supplementary materials.

###### 4.4. Extension to Model Fine-tuning

Background: Open-source models typically underperform on ARC-AGI due to limited capabilities on abstract reasoning tasks. Prior work [1, 7, 10, 17] constructs large-scale

synthetic training data to fine-tune open-source models, but treats ARC-AGI as a pure text task without using visual information. We investigate whether vision-language synergy during fine-tuning can improve open-source model performance beyond text-only fine-tuning.

Method. We apply the same VLSR task decomposition from Sec. 3.3 during training: a vision-language model (Qwen3-VL-8B-Instruct) for visual rule summarization and a text-only model (Qwen3-8B) for textual rule application. This decouples the two subtasks, allowing for specialized training of each component.

Training Setup. Training data comes from ARC-Heavy200k [17], which provides synthetic ARC-AGI tasks with ground-truth rules. We use approximately 200k training tasks (excluding a held-out 100-task test set), with each task split into three example pairs and one test sample. ARCHeavy-200k provides explicit rules for each task, enabling us to train the rule summarization and rule application modules separately. As a text-only baseline, we also fine-tune Qwen3-8B on the same 200k tasks using textual matrix representations from ARC-Heavy-200k.

Results. As shown in Tab. 5, VL synergy fine-tuning achieves 13.25% on ARC-AGI, outperforming text-centric fine-tuning (9.75%) by 3.5% and surpassing the closedsource baseline GPT-4o (8.25%) by 5.0%. Compared to the Qwen3-8B baseline before fine-tuning, text-only finetuning improves performance by 6.5% while VL synergy fine-tuning improves by 9.75%, which demonstrates the advantage of incorporating visual information during training. Similar trends hold on BARC-100 and Re-ARC. We attribute the advantage of VL synergy fine-tuning to two factors: (1) task decomposition reduces training complexity by separating rule extraction from rule application, and (2) visual information provides global 2D structural cues (e.g., spatial contiguity, symmetry patterns) that are difficult to learn from sequential textual representations alone.

### 5. Conclusion

This paper incorporates visual information into abstract reasoning tasks, which have traditionally been treated as purely textual. By proposing the Vision-Language Synergy Reasoning (VLSR) framework and the Modality-Switch SelfCorrection (MSSC) mechanism, the 2D intuitiveness inherent in the visual modality and the precision of element-wise representation offered by the textual modality can be effectively integrated. Leveraging the distinct advantages of both modalities, our method achieves an average performance improvement of 4.3% compared to text-only reasoning approaches across multiple abstract reasoning tasks and different base models. Furthermore, it outperforms other textcentric training-free strategies, thereby demonstrating the unique value of incorporating visual information. Additionally, this inference strategy can be extended to model fine-

tuning scenarios. The vision-language synergy fine-tuning strategy can yield a 3.5% improvement over text-only finetuning methods and outperform several open-source and closed-source models with much larger parameter sizes.

### References

- [1] Ekin Aky¨urek, Mehul Damani, Linlu Qiu, Han Guo, Yoon Kim, and Jacob Andreas. The surprising effectiveness of test-time training for abstract reasoning. arXiv e-prints, pages arXiv–2411, 2024. 3, 8
- [2] Natasha Butt, Blazej Manczak, Auke Wiggers, Corrado Rainone, David W Zhang, Micha¨el Defferrard, and Taco Cohen. Codeit: Self-improving language models with prioritized hindsight replay. arXiv preprint arXiv:2402.04858,

2024. 3

- [3] Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, et al. Enigmata: Scaling logical reasoning in large language models with synthetic verifiable puzzles. arXiv preprint arXiv:2505.19914, 2025. 3
- [4] Fran¸cois Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. 1, 3, 6
- [5] J. Cole. Community interview jack cole– lab42 lab42.global., 2024. 3
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6
- [7] Daniel Franzen, Jan Disselhoff, and David Hartmann. The llm architect: Solving arc-agi is a matter of perspective,

- 2024. 3, 8

[8] Daniel Franzen, Jan Disselhoff, and David Hartmann. Product of experts with llms: Boosting performance on arc is a matter of perspective. arXiv preprint arXiv:2505.07859,

- 2025. 3

- [9] Matthew Ho, Chen Si, Zhaoxiang Feng, Fangxu Yu, Yichi Yang, Zhijian Liu, Zhiting Hu, and Lianhui Qin. Arcmemo: Abstract reasoning composition with lifelong llm memory. arXiv preprint arXiv:2509.04439, 2025. 3, 7
- [10] Michael Hodel. Addressing the abstraction and reasoning corpus via procedural example generation. arXiv preprint arXiv:2404.07353, 2024. 3, 6, 8
- [11] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024. 3
- [12] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798, 2023. 2, 6
- [13] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 6

- [14] Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks. arXiv preprint arXiv:2510.04871,

2025. 3

- [15] Hosung Lee, Sejin Kim, Seungpil Lee, Sanha Hwang, Jihwan Lee, Byung-Jun Lee, and Sundong Kim. Arcle: The abstraction and reasoning corpus learning environment for reinforcement learning. arXiv preprint arXiv:2407.20806,

2024. 3

- [16] Solim LeGris, Wai Keen Vong, Brenden M Lake, and Todd M Gureckis. H-arc: A robust estimate of human performance on the abstraction and reasoning corpus benchmark. arXiv preprint arXiv:2409.01374, 2024.
- [17] Wen-Ding Li, Keya Hu, Carter Larsen, Yuqing Wu, Simon Alford, Caleb Woo, Spencer M Dunn, Hao Tang, Michelangelo Naim, Dat Nguyen, et al. Combining induction and transduction for abstract reasoning. arXiv preprint arXiv:2411.02272, 2024. 3, 6, 8
- [18] Matthew V Macfarlane and Cl´ement Bonnet. Searching latent program spaces. arXiv preprint arXiv:2411.08706,

2024. 3

- [19] Michael D Moffitt. Arc-gen: A mimetic procedural benchmark generator for the abstraction and reasoning corpus. arXiv preprint arXiv:2511.00162, 2025. 3
- [20] Arseny Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The conceptarc benchmark: Evaluating understanding and generalization in the arc domain. arXiv preprint

- arXiv:2305.07141, 2023. 3

[21] OpenAI. Gpt-5 system card, 2025. 1 [22] OpenAI. Openai o3 and o4-mini system card, 2025. 6

[23] Jaehyun Park, Jaegyun Im, Sanha Hwang, Mintaek Lim, Sabina Ualibekova, Sejin Kim, and Sundong Kim. Unraveling the arc puzzle: Mimicking human solutions with object-centric decision transformer. arXiv preprint

- arXiv:2306.08204, 2023. 3

- [24] Julien Pourcel, C´edric Colas, and Pierre-Yves Oudeyer. Self-improving language models for evolutionary program synthesis: A case study on arc-agi. arXiv preprint arXiv:2507.14172, 2025. 3
- [25] Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, et al. Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement. arXiv preprint arXiv:2310.08559, 2023. 3
- [26] Kartik Singhal and Gautam Shroff. Conceptsearch: Towards efficient program search using llms for abstraction and reasoning corpus (arc). In Proceedings of the AAAI Conference on Artificial Intelligence, pages 20506–20513, 2025. 3
- [27] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with selfsupervision for generalization under distribution shifts. In International conference on machine learning, pages 9229–

9248. PMLR, 2020. 3

- [28] Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Testtime learning with adaptive memory. arXiv preprint arXiv:2504.07952, 2025. 3, 7

- [29] Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model. arXiv preprint arXiv:2506.21734, 2025. 3
- [30] Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah D Goodman. Hypothesis search: Inductive reasoning with language models. arXiv preprint arXiv:2309.05660, 2023. 3
- [31] Yile Wang and Hui Huang. Improving abstract reasoning ability of large language models through mixture programbased data synthesis. In China National Conference on Chinese Computational Linguistics, pages 208–228. Springer,

2025. 3

- [32] Yang Wang and Zhejun Zhao. Advancing abstract reasoning in artificial general intelligence with a hybrid multicomponent architecture. In 2024 4th International Symposium on Artificial Intelligence and Intelligent Manufacturing (AIIM), pages 867–871. IEEE, 2024. 3
- [33] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression, 2025. 5
- [34] Johan S. Wind. Dsl solution to the arc challenge, 2020. 3
- [35] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965,

2025. 3

- [36] xAI Team. Grok 4, 2025. 1
- [37] Kevin Xu and Risto Miikkulainen. Neural cellular automata for arc-agi. arXiv preprint arXiv:2506.15746, 2025. 3
- [38] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 6
- [39] Qingjie Zhang, Di Wang, Haoting Qian, Yiming Li, Tianwei Zhang, Minlie Huang, Ke Xu, Hewu Li, Yan Liu, and Han Qiu. Understanding the dark side of llms’ intrinsic selfcorrection. arXiv preprint arXiv:2412.14959, 2024. 2, 6
- [40] Shaoting Zhu, Shuangyue Geng, and Un Lok Chen. T5-arc: Test-time training for transductive transformer models in arcagi challenge. 3
- [41] Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025. 3

## Think Visually, Reason Textually: Vision-Language Synergy in ARC Supplementary Material

### A. Prompts

###### Prompt for Text-only Reasoning in ARC-AGI:

I will provide you with several input and output matrices. You need to find the matrix-changing rule from it and apply it to the new input. Put the output matrix within \\boxed{}.

- Example Input 1: [[0, 1, 0, ...], [0, 1, 1, ...], ...]

- Example Output 1: [[1, 1, 0, ...], [1, 1, 1, ...], ...]

Example Input 2: [[0, 2, 4, ...], [0, 2, 2, ...], ...]

- Example Output 2: [[2, 2, 4, ...], [2, 2, 2, ...], ...]

... Example Input n: [[0, 4, 0, ...], [0, 6, 1, ...], ...] Example Output n: [[4, 4, 0, ...], [6, 6, 1, ...], ...] New Input: [[0, 2, 0, ...], [0, 5, 3, ...], ...]

###### Prompt for Vision-centric Rule Summarization:

I will now provide you with several input and output images about 2D grids. You need to summarize the grid-changing rule from it. Output the rule you learned within \\boxed{}.

- Example Input 1: <Input Image 1>

- Example Output 1: <Output Image 1>

Example Input 2: <Input Image 2>

- Example Output 2: <Output Image 2>

... Example Input n: <Input Image n> Example Output n: <Output Image n>

###### Prompt for Text-centric Rule Application:

I will provide you with several input and output matrices. You need to find the matrix-changing rule from it and apply it to the new input. Put the output matrix within \\boxed{}.

Here is a possible rule for your reference. Rule: The rule involves removing the colored cross ... Note that the rule is described in color and each color represents a value in the matrix: [0:black; 1:blue; 2:red; 3:green; 4:yellow; 5:grey; 6:pink; 7:orange; 8:light blue; 9:brown]. You need to first check the correctness of the rule based on the examples. If the rule is correct, apply it to the new input. Otherwise, summarize a new rule and apply it to the new input.

- Example Input 1: [[0, 1, 0, ...], [0, 1, 1, ...], ...]

- Example Output 1: [[1, 1, 0, ...], [1, 1, 1, ...], ...]

Example Input 2: [[0, 2, 4, ...], [0, 2, 2, ...], ...]

- Example Output 2: [[2, 2, 4, ...], [2, 2, 2, ...], ...]

Example Output n: [[4, 4, 0, ...], [6, 6, 1, ...], ...] New Input: [[0, 2, 0, ...], [0, 5, 3, ...], ...]

###### Prompt for Vision-centric Consistency Verification:

I will now provide you with several input and output example images, which follows a specific changing rule. Then, I will give you another input and output pair, determine whether the new pair also follows the same changing rule. Add your final judgment at the end of your replay: \\boxed{True} or \\boxed{False}.

- Example Input 1: <Input Image 1>

- Example Output 1: <Output Image 1>

Example Input 2: <Input Image 2>

- Example Output 2: <Output Image 2>

... Example Input n: <Input Image n> Example Output n: <Output Image n> New Input: <New Input Image> New Output: <Output Image Pred>

### B. Matrix-to-Image Visualization

We visualize the input-output matrices into color-coded 2D grids to provide 2D spatial information and global view. The detailed visualization process is listed as follows.

First, each value is mapped to a distinct color. The color map is: [ 0: black; 1: blue; 2: red; 3: green; 4: yellow; 5: grey; 6: pink; 7: orange; 8: light blue; 9: brown ]

Furthermore, between the two elements (small colored squares), we add white dividing lines to more clearly indicate the specific number and the structure of the elements contained within a block.

### C. Qualitative Examples

We conduct an in-depth analysis of the specific outputs of different models (GPT-4o, Gemini-2.5-Pro-thinking-8192, o4-mini) when employing visual thinking versus textual thinking in the ARC-AGI task. Visual thinking demonstrates numerous unique advantages, such as the integration of 2D structural information(Fig. 6), a global perspective(Fig. 5, Fig. 7), and long-range perception capabilities(Fig. 8). Therefore, using visual information during the global rule summarization phase enables more effective extraction of the underlying rules.

... Example Input n: [[0, 4, 0, ...], [0, 6, 1, ...], ...]

[Figure 22]

[Figure 23]

Visualized Task

To change the input matrix, remove all-zero rows and columns. Identify the non-zero submatrix and remove any surrounding zero rows and columns. Essentially, trim the matrix to its smallest size that contains all non-zero elements, preserving the order and positions of these elements relative to each other.

Text-Only Reasoning

The rule involves removing the colored cross in the middle and shifting the remaining colored shapes towards the center to fill the space evenly. Each quadrant's shape moves towards the center, maintaining its orientation.

Vision-Centric Rule Summarization

Visual thinking is holistic, enabling it to quickly identify the most critical feature (the colored cross) and subsequently summarize the correct rule: assembling the shapes in each quadrant toward the center. In contrast, textual thinking is independent and lacks such a holistic perspective and fails to capture the prominent feature of the colored cross, thereby erroneously interpreting the task as compressing the matrix to include all non-zero elements.

Main Difference

- Figure 5. Visual reasoning possesses a global perspective, enabling it to better capture the most critical feature (the colored cross) in the entire image and subsequently summarize the correct rule the underlying rules. Base model is both gpt-4o.

[Figure 24]

#### Visualized Task

Here’s the idea in a nutshell:

- 1. In the input, the 8’s mark “target” slots into which the other non-zero values will move. All original non-8 entries and all the 8’s are eventually zeroed out.
- 2. Work row by row. In each row a) read off its non-zero entries (ignoring any 8’s), keeping them in left-to-right order; b) find all contiguous runs of 8’s in that row (again, left to right); c) for the k_th run of 8’s, replace those 8’s by the k_th non-zero entry from (a)
- 3. Everything else becomes 0. Text-Only Reasoning

Take the ordered list of colored bars on the left, delete the bars, and then repaint the cyan “Tetris” blocks on the right by cycling through exactly those same colors, in the same left-toright order. In other words, remove the yellow/green/red (etc.) stripes, and color the first cyan block (or pixel) you encounter with the first stripe-color, the next cyan block with the second stripe-color, and so on, leaving the background black and discarding the original bars.

Vision-Centric Rule Summarization

Textual reasoning tends to process the input matrix "row by row", thus failing to summarize the correct rules. It incorrectly replace every element in a row based on the elements on the left, whereas in reality, the replacement of elements in each row is still related to the presence of blocks in the previous rows. In contrast, visual reasoning can flexibly process the matrix "block by block" and correctly summarize the variation rule: coloring is performed block by block from left to right.

Main Difference

- Figure 6. Visual reasoning possesses 2D information and can flexibly summarize rules in a “block-by-block” manner, whereas text reasoning tends to adopt a “row-by-row” processing approach, thus failing to derive the correct rules. Base model is both o4-mini.

[Figure 25]

[Figure 26]

#### Visualized Task

If the input value is 5, it remains 5. If the input value is 0, its new value is determined by its 8 neighbors (including diagonals): If it has 5s as neighbors both vertically (up/down) AND horizontally (left/right), it becomes 2. If it has 5s as neighbors either vertically OR horizontally (but not both), it becomes 1. Otherwise, it remains 0. Text-Only Reasoning

The rule is to recolor the black paths in the input image based on their position relative to the grey squares.

- 1. **Identify Internal Paths:** Any black path segment that is located between two grey squares is colored red. This applies to both horizontal paths that separate two rows of squares and vertical paths that separate two columns of squares.
- 2. **Identify External Paths:** Any black path segment that is located between a grey square on one side and the outer black background on the other side is colored blue. These are the paths on the periphery of the grid of squares.
- 3. **No Change:** The grey squares themselves and the black background remain unchanged. Vision-Centric Rule Summarization

Visual thinking possesses a superior global perspective, enabling the identification of the core feature (grey squares) in the figure and subsequent processing based on Internal Paths and External Paths. In contrast, textual thinking focuses more on local information, thus conducting processing through the 8 neighbors corresponding to a specific element.

Main Difference

- Figure 7. Visual thinking tends to adopt a global perspective and thus processes information based on internal and external paths; in contrast, textual thinking focuses more on local information and processes individual elements with reference to their 8-neighbor context. Base model is both Gemini-2.5-Pro-thinking-8192.

[Figure 27]

Visualized Task

First, find the largest contiguous non-zero area in the input grid and crop to its bounding box. For each cell in this new grid, its new number is determined by its four cardinal neighbors (up, down, left, right). The new number is the count of unique values among these four neighbors that are different from the cell's own original value. If a cell is on the border, only its existing

neighbors are counted. Text-Only Reasoning

The image transformation rule is a color replacement process guided by key-value pairs. In each input image, there is a central object composed of colored shapes within a larger colored frame, and several small two-color pairs scattered on the black background. These pairs act as a color legend. For each pair, the color of the right square indicates an original color to be found within the central object's inner shapes, and the color of the left square is the new color it should be changed to. The transformation consists of applying these color replacements to the shapes inside the frame, while the frame's color remains unchanged. The final output displays only the recolored central object, with the background and the color-key pairs removed.

Vision-Centric Rule Summarization

Textual thinking represents each element individually, thus lacking "long-range correlation" and potentially overlooking some detailed elements. In the context of this example, textual thinking fails to accurately grasp the value of small two-number pairs; in contrast, visual thinking can fully comprehend the function of two-color pairs and summarize the correct rules.

Main Difference

- Figure 8. Visual thinking possesses superior long-range correlation capabilities and can better capture detailed features (the 2-color pairs and the re-color strategy). Base model is both Gemini-2.5-Pro-thinking-8192.

