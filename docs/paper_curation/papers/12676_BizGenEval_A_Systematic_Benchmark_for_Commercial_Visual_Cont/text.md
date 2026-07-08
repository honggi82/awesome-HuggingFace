# arXiv:2603.25732v1[cs.CV]26Mar2026

## BizGenEval: A Systematic Benchmark for Commercial Visual Content Generation

Yan Li2∗, Zezi Zeng3∗, Ziwei Zhou4∗, Xin Gao4∗, Muzhao Tian4∗, Yifan Yang1†, Mingxi Cheng1, Qi Dai1, Yuqing Yang1, Lili Qiu1, Zhendong Wang1, Zhengyuan Yang1, Xue Yang2†, Lijuan Wang1, Ji Li1, and Chong Luo1

1 Microsoft Corporation

- 2 Shanghai Jiao Tong University
- 3 Xi’an Jiaotong University 4 Fudan University

https://aka.ms/BizGenEval

Abstract. Recent advances in image generation models have expanded their applications beyond aesthetic imagery toward practical visual content creation. However, existing benchmarks mainly focus on naturalimage synthesis and fail to systematically evaluate models under the structured and multi-constraint requirements of real-world commercial design tasks. In this work, we introduce BizGenEval, a systematic benchmark for commercial visual content generation. The benchmark spans five representative document types—slides, charts, webpages, posters, and scientific figures—and evaluates four key capability dimensions: text rendering, layout control, attribute binding, and knowledge-based reasoning, forming 20 diverse evaluation tasks. BizGenEval contains 400 carefully curated prompts and 8,000 human-verified checklist questions to rigorously assess whether generated images satisfy complex visual and semantic constraints. We conduct large-scale benchmarking on 26 popular image generation systems, including state-of-the-art commercial APIs such as Nano Banana 2 and GPT-Image-1.5 as well as leading opensource models. The results reveal substantial capability gaps between current generative models and the requirements of professional visual content creation. We hope BizGenEval serves as a standardized benchmark for real-world commercial visual content generation.

Keywords: Image Generation · Benchmark · Multimodal Evaluation

### 1 Introduction

Recent advances in large-scale image generation models, such as Nano Banana Pro [11] and GPT-Image-1.5 [27], have significantly expanded the capabilities of

∗ Equal contribution. This work was done during their internship at Microsoft. † Corresponding to: Yifan Yang <yifanyang@microsoft.com>, Xue Yang <yangxue2019-sjtu@sjtu.edu.cn>.

visual generation systems. Beyond producing aesthetically pleasing images, these models are increasingly being used as practical interfaces for complex visual content creation tasks. In many real-world scenarios, image generation systems can already produce professional materials such as presentation slides, web page layouts, scientific figures, posters, and data charts with minimal human intervention. Recent industry reports, including GPT-Image-1.5 and Qwen-Image 2, increasingly highlight commercial design scenarios, reflecting the growing practical and economic importance of such capabilities.

However, systematic evaluation of commercial visual generation remains largely

under-explored. Despite the growing practical importance of these tasks, there is currently no comprehensive benchmark that systematically evaluates how well image generation models perform on commercial visual content creation, and model capabilities are often demonstrated through selected examples as in [27,30,35,36,44] rather than standardized evaluation.

Evaluating such tasks is inherently challenging, as commercial visual documents typically contain dense textual content, complex layouts, numerous visual elements, and strict semantic correctness requirements. Existing image generation benchmarks [10, 14, 15] mainly focus on natural images, object compositionality, or aesthetic quality. OneIG-Bench [2] includes themes such as slides or poster generation, but mainly uses them as prompt sources for text-rendering evaluation with simple prompts and similarity-based metrics, rather than assessing holistic commercial visual generation capability. Although several recent works explore design-oriented tasks such as infographic generation [34] or poster rendering [28], these evaluations remain fragmented and domain-specific.

To address this gap, we introduce BizGenEval, the first comprehensive benchmark for evaluating image generation models on commercial visual content creation.

The benchmark covers 5 representative domains—Slides, Scientific Figures, Charts, Webpages, and Posters—which correspond to common visual artifacts in professional workflows. For each domain, we evaluate 4 key capabilities: Text Rendering, Layout Control, Attribute Binding, and Knowledge-based Reasoning. Together, the five domains and four capability dimensions form 20 evaluation tasks. Representative examples are shown in Fig. 1, illustrating realistic multiconstraint generation scenarios in professional design workflows.

For the content-based dimensions (Text Rendering, Layout Control, and Attribute Binding), prompts are constructed from real-world commercial materials. We manually collect authentic design examples from professional online sources and curate representative samples for each domain. Based on these references, task-specific prompts are created to simulate realistic requests that a human designer might issue when using an image generation system. These prompts test whether models can accurately control low-level design elements such as typography, arrow directions, text box placement, visual counts, and spatial arrangements.

Beyond element-level rendering, commercial visual generation often requires incorporating external domain knowledge. To evaluate this aspect, we introduce

Layout Attribute Text Knowledge

Alignment Count Body

History

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

1 2

1

1

Webpage

1

2

2

2

- Q1: Does the main ship illustration correctly depict the structural hull and rigging characteristics?
- Q2: Are the chronological events displayed on the timeline historically accurate for the 15th to 17th centuries?

- Q1: Is there one large circle on the left and two smaller ones on the right?
- Q2: Is a large brand wordmark centered at the bottom of the left page?

- Q1: Does the paragraph under the center-left hero heading read: “Inspira V 2.5 is a …”?
- Q2: Does the paragraph under “AI that senses ...” read: “In the new version of

- Q1: Are five dark blue step boxes arranged horizontally at the top right?
- Q2: Are four service cards in a 2×2 grid with orange line icons?

Body

Chemistry

Structure Icon

1

[Figure 5]

1

[Figure 6]

[Figure 7]

[Figure 8]

1 2

2

- 1
- 2

Slides

2

- Q1: Are the depicted visual color changes scientifically accurate for the specific chemical compounds produced?
- Q2: Do the table’s observation descriptions exactly match the visual outcomes shown in the corresponding comic panels?

- Q1: Does the paragraph in the upper right rounded rectangle beside the stethoscope icon match the quoted text exactly?
- Q2: In the wider right subcolumn of the left main column, are there exactly three rectangular text panels stacked vertically?

- Q1: Are three client markers vertically along a curved guide in the left column?
- Q2: Are three rectangular panels stacked vertically in the right subcolumn?

- Q1: Is there a wide burnt-orange rounded rectangle at the top with centered white text?
- Q2: Does the seventh pink square contain a centered black line icon of an eye with short rays around it?

Grid Proportion Body

Physics

###### 1

[Figure 9]

[Figure 10]

- 1
- 2

[Figure 11]

1

[Figure 12]

1

Chart

2

2

- Q1: Is the filament lamp's current-voltage curve correctly depicted as mathematically symmetric in both the forward and reverse bias regions?
- Q2: Does the stated reason for the filament lamp correctly identify thermal heating as the main cause of increased resistance?

2

- Q1: Does the subtitle read exactly Annual coffee consumption per person compared with GDP per hour worked for selected countries GDP?
- Q2: Does a callout box near Luxembourg contain the exact text Luxembourg very high GDP per hour and exceptionally high coffee?

- Q1:Is there a centered legend below the heading with five square swatches labeled "Four", ..., "One", and "Not affected"?
- Q2: Visual proportion, Relative size In this chart, are the relative sizes of the bars proportional to their values?

- Q1: Is there a fifth subplot centered below the

2x2 grid, not spanning the panel’s full width?

- Q2: Inside the panel, are four square subplots arranged in a 2x2 grid with equal spacing?

Region Color Title

Music

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

1

1

1

1

###### PosterScientificFigure

2

###### 2

2

2

- Q1: Do the visual representations of the vocal ranges accurately demonstrate realistic pitch overlaps between adjacent voice types?
- Q2: Is the image completely free of any explicit statement directly naming which voice type usually plays the villain?

- Q1: Does the large three-line title in the center-left tall panel read “6 REASONS”, “TO GET UP”, “EARLY” line by line?
- Q2: Does the paragraph directly below “More time to get things done” in the lower-middle-right panel read exactly: “Have…?

- Q1: Is the card connected to number 3 positioned on the right side of the timeline and filled with bright red?
- Q2: Are all connector lines between the numbered circles and the colored cards thin, solid, and without dashes?

- Q1: Are there exactly three rounded-rectangle content cards arranged side-by-side in a single horizontal row beneath the main title block?
- Q2: Does each lower card have an illustration on the left and a text block on the right?

Connection Shape Chart

Mathematics

[Figure 17]

[Figure 18]

[Figure 19]

2 1

[Figure 20]

[Figure 21]

2

1

- 1

- 2

1

2

- Q1: Are there four dashed rectangular regions arranged horizontally from left to right and labeled 'backbone', 'encoder', 'decoder', and 'prediction heads'?
- Q2: Is a rectangle labeled transformer decoder to the right of the encoder, connected by a rightward arrow from encoder?

- Q1:In the left panel, is there exactly one wide soft pink rounded rectangle labeled “Linear Projection of Flattened Patches” directly above the nine patches?
- Q2: In the right panel, are modules stacked bottom-to-top as "Embedded Patches", "Norm", "Multi-Head Attention", "Norm", then "MLP"?

- Q1: Is the filament lamp's current-voltage curve correctly depicted as mathematically symmetric in both the forward and reverse bias regions?
- Q2: Does the stated reason for the filament lamp correctly identify thermal heating as the main cause of increased resistance?

- Q1: Does the first data row show 'in' with '47%', '55.3%', and ‘9.40%'?
- Q2: Inside ... is the sentence exactly A’s productivity of 57.5% ...?

- Fig. 1: Real-world samples of BizGenEval from 5 content domains and 4 capability dimensions. For each capability dimension, different subtasks are highlighted with blue boxes. Each example covers distinct aspects, with the relevant regions highlighted in red and the corresponding simplified questions listed below.

the Knowledge-based Reasoning dimension covering five themes—physics, chemistry, mathematics, history, and arts. These tasks require models to generate coherent images while reflecting underlying factual or conceptual knowledge.

All prompts and evaluation rationales are carefully curated and rewritten by human experts to ensure task validity and difficulty. In total, the benchmark contains 300 content-based prompts and 100 knowledge-based prompts, enabling comprehensive evaluation of both visual design control and knowledgegrounded generation in realistic commercial content creation scenarios.

Evaluating commercial visual documents is challenging due to their dense textual content, complex layouts, and tightly coupled visual constraints. To address this, we design a structured checklist-based evaluation protocol. For each prompt, human experts construct 20 task-specific verification questions that directly correspond to the evaluated capability.

We further introduce an automated evaluation pipeline that employs state-ofthe-art multimodal large language models (MLLMs) as judges to answer these verification questions. The final score is computed based on the accuracy of the responses, enabling scalable and consistent evaluation. Human evaluation in Sec. 4.2 shows a strong agreement between MLLM evaluators and human judgments, demonstrating the reliability of the proposed protocol.

Using BizGenEval, we conduct a large-scale evaluation of both closedsource and open-source image generation models. The results reveal substantial performance differences across domains and capability dimensions, providing new insights into the strengths and limitations of current image generation systems for commercial visual content creation.

Contributions. 1) BizGenEval, the first comprehensive benchmark for commercial visual content generation, covering five representative domains and four capability dimensions, resulting in 20 evaluation tasks and 400 curated prompts, together with 8,000 human-verified checklist questions for rigorous evaluation. 2) We conduct a large-scale evaluation of state-of-the-art image generation systems, spanning both commercial APIs and open-source models, establishing strong baselines for commercial visual generation. 3) We provide extensive empirical analysis of current model capabilities and limitations, offering insights for future image generation systems in practical commercial applications.

### 2 Related Work

Benchmarks for Commercial Visual Documents. Recent advances in largescale image generation models have increasingly sparked interest in their applications to commercial visual generation, motivating several works to benchmark specific types of commercial visual documents. For example, SlidesGenBench [43] is restricted to language-driven slide creation, while IGenBench [34] and BizGen [28] focus on infographics with atomic verification and long-context layout constraints. For web interfaces, Design2Code [32] and WebSight [19] are designed specifically for benchmarking screenshot-to-code systems. In scientific figures, FigureBench [46] is tailored to scientific figure generation from long-form

technical descriptions. Unlike these domain-specific efforts, our BizGenEval provides a holistic perspective across five representative commercial domains, covering slides, charts, webpages, posters, and scientific figures.

Benchmarks for Core Image-Generation Capabilities. Beyond domainspecific applications, a parallel line of research focuses on evaluating the fundamental generative capabilities of text-to-image models. For text rendering, LongText-Bench [9] and TextCrafter with CVTG-2K [8] are restricted to longform or multi-region text on text-centric images. Instruction-following suites such as TIIF-Bench [41] and OneIG-Bench [2] include a text-rendering dimension, but evaluate it via automatic global semantic scores on relatively unconstrained scenes, and offer little coverage of dense, layout-aware typography in structured commercial documents. For spatial layout and attribute control, LayoutBench [4], 7Bench [16], and OverLayBench [22] operate on abstract boundingbox layouts or simplified layout-to-image settings, with layout specified by object boxes and category labels. They rarely couple layout with long textual content or complex document semantics. GenEval [10], VQAScore [23], and GenAIBench [21] further probe object-level attributes, counts, and compositional relations in natural or synthetic scenes, rather than design-centric properties such as color schemes, icon usage, and stylistic consistency inside structured commercial documents. For knowledge and reasoning, WISE [25], WorldGenBench [45], R2I-Bench [3], and MMMG [24] evaluate knowledge-grounded image synthesis on generic or educational-style scenes, typically without enforcing realistic document formats or multi-constraint design goals. GenExam [40] introduces exam-style prompts and checklist scoring, but still operates outside commercial document layouts and does not require simultaneous satisfaction of layout, dense text, attributes, and factual correctness.

### 3 BizGenEval

#### 3.1 Task Description

BizGenEval is designed to evaluate commercial visual generation across diverse real-world content creation scenarios and demanding capabilities. Specifically, the benchmark is structured along two orthogonal dimensions: 5 content domains and 4 capability dimensions, resulting in a total of 20 evaluation tasks, as illustrated in Fig. 1.

We first start with five representative types of commercial visual content that frequently arise in professional workflows: Webpage: webpage designs that integrate structured layouts, textual content, and functional interface elements such as headers, sections, buttons, etc. Slides: presentation slides used in reports, lectures, and business scenarios, characterized by hierarchical structure, bullet lists, and aligned visual elements. Chart: data visualization graphics such as bar charts, line charts, and multiseries plots, involving precise rendering of numeric values, axes, and legends.

Poster: promotional or informational posters combining typography, graphics, and layout design, often emphasizing visual hierarchy and balanced composition.

Scientific Figure: figures in academic papers including diagrams, pipelines, and illustrations, where components, arrows, and annotations form clear structure.

For each domain, we further design four capability-oriented evaluation tasks that capture key challenges in commercial visual generation: Layout: evaluates spatial and structural organization, including overall layout, complex flows, arrows, blocks, and hierarchical arrangement of elements. Attribute: assesses control over visual attributes such as color, shape, style, icons, and quantity, emphasizing fine-grained control. Text: evaluates textual content rendering, including short titles, long paragraphs, tables, and integration with other components. Knowledge: tests reasoning and domain knowledge, including applying world knowledge across diverse domains such as physics, chemistry, arts, history, etc.

BizGen-Bench: A Systematic Benchmark for Commercial Visual Content Generation v2

Content-based Tasks

Checklist

###### Evaluation

|Dim: Webpage × Layout|
|---|

[Figure 22]

- 1. Are there exactly two primary vertical columns?
- 2. Is there a 3 by 3 matrix inside the laptop screen?
- 3. ... (Total 20 Qs)

[Figure 23]

[Figure 24]

- 1. YES. Two primary vertical columns.
- 2. NO. 2 by 3 matrix inside the laptop screen. … Summary: 16 Correct 4 Mistakes

Detailed Prompt

Generated Images

Create a webpage with a two

column layout. The left side has a laptop hero area containing a 3

[Figure 25]

| |
|---|

Leading Model

by 3 matrix grid. The right side is a mosaic organized into three vertical columns…

Real World Reference Images

[Figure 26]

[Figure 27]

（Slides, Webpage, Poster, Chart, Scientific Figure)

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Knowledge-based Tasks

[Figure 32]

[Figure 33]

Detailed Prompt …

- 1. YES. The chemical equation is balanced.
- 2. NO. Incorrect relative catalyst speeds. … Summary: 13 Correct 7 Mistakes

Design an Elephant Toothpaste slide. Include a balanced chemical equation and a data table of catalysts with plausible reaction rates …

[Figure 34]

[Figure 35]

[Figure 36]

| |
|---|

Physics Math

[Figure 37]

[Figure 38]

Checklist

Chemistry

[Figure 39]

[Figure 40]

Hidden Rational

- 1. Does the image display the correct balanced chemical equation for the decomposition?
- 2. Does the data table accurately assign the fastest reaction rates to the most effective catalysts?
- 3. ... (Total 20 Qs)

Analysis

[Figure 41]

[Figure 42]

Manganese dioxide is a highly effective inorganic catalyst which yields faster reaction rates than yeast.

[Figure 43]

[Figure 44]

[Figure 45]

History

Art

|Dim: Slides × Knowledge|
|---|

- Fig. 2: Overview of the construction and evaluation pipeline. The system converts real world references and domain knowledge into structured prompts, and evaluates generated images using rigorous checklists to provide a comprehensive analysis.

#### 3.2 Benchmark Construction of BizGenEval

Content-based Data Collection. To support evaluation of the content-based capabilities (Text Rendering, Layout Control, and Attribute Binding), we first construct a reference pool of real-world commercial visual designs. We manually collect 1,819 candidate images from diverse professional sources, including UI/UX design repositories, corporate presentation archives, academic databases, and digital marketing portfolios. Unlike aesthetic-oriented datasets, our collection explicitly focuses on visuals created for authentic commercial scenarios.

Based on the predefined domain–task taxonomy, we then perform a targeted curation process to identify representative references for each domain–capability

combination. For every domain–task pair, we deliberately select images that exhibit clear structural patterns and realistic design constraints. Through multiple rounds of human-in-the-loop filtering, low-information or ambiguous samples are excluded while any personal or sensitive information in the original materials is removed to ensure privacy compliance.

Finally, we retain 20 representative references for each domain–task combination, ensuring balanced coverage across domains and capabilities. These curated references later serve as the foundation for prompt construction. In total, this process produces 300 content-based reference samples capturing key challenges in commercial visual content generation.

Knowledge-based Data Collection. Beyond structural rendering, commercial visual generation often requires models to incorporate domain knowledge and factual reasoning. To evaluate this capability, we construct a knowledge reference pool covering five themes: physics, chemistry, mathematics, history, and arts. For each theme, 20 representative knowledge points are curated from professional databases, educational resources, and academic curricula to ensure that the evaluation targets reflect reliable and expert-level content. For scientific domains such as physics and chemistry, we prioritize knowledge points involving experimental setups, molecular structures, and fundamental scientific principles that require precise symbolic correctness and conceptual consistency. The mathematics category emphasizes reasoning-based visual tasks such as geometric constructions, equations, and diagrammatic representations. For humanitiesrelated themes including history and arts, the selected knowledge points focus on chronological relationships, cultural symbols, and stylistic characteristics.

Through this process we obtain 100 curated knowledge references. These knowledge points are later expanded into generation prompts across the benchmark domains to evaluate models’ ability to integrate factual knowledge into visual content generation.

Prompt Construction. Based on the curated visual references and knowledge points, we construct prompts for both content-based and knowledge-based evaluation tasks. For content-based tasks, prompts are generated from reference images using a structured analysis pipeline. A vision–language model performs component-level analysis to identify key structural elements, visual attributes, and textual content, from which detailed generation instructions are produced to specify the required visual elements and their relationships. Different tasks emphasize distinct aspects of generation: layout control focuses on spatial organization and structural relationships, attribute binding targets visual properties such as color, shape, quantity, and style, and text rendering requires exact character-level reproduction of textual content.

For knowledge-based tasks, prompts are constructed by expanding curated knowledge references into generation instructions within the five content domains. To prevent answer leakage and encourage genuine reasoning, key knowledge facts are intentionally omitted from the prompts and retained as hidden rationales used only during evaluation. For example, as illustrated in Fig. 2, a task combining the Slides domain and Chemistry knowledge asks the model to

“Design an Elephant Toothpaste slide with a balanced chemical equation and a catalyst table with plausible reaction rates.” The hidden rationale that “Manganese dioxide is a highly effective inorganic catalyst that yields faster reaction rates than yeast” is withheld, forcing the model to rely on its internal knowledge to produce factually consistent visual content.

[Figure 46]

[Figure 47]

[Figure 48]

(a) Prompt Length Distribution (b) Dimension Composition (c) Checklist Word Clouds

- Fig. 3: Dataset Statistics of BizGenEval. (a) Prompt token length distribution by evaluation dimension, with scatter points color-coded by document type. (b) Hierarchical subcategory distribution across evaluation dimensions. The inner ring shows question proportions per dimension; the outer ring shows subcategory breakdowns. (c) Checklist Questions keyword clouds per evaluation dimension.

Evaluation Checklist Construction. Based on the prompt, we further construct 20 verification questions for each task through a human-in-the-loop process, forming a structured checklist for systematic evaluation. Due to the large variability in model performance, we divide the questions into 10 easy and 10 hard items to enhance discrimination across models of different capability levels. The easy questions mainly target fundamental properties such as basic spatial layout, prominent visual attributes, or high-level text correctness, while the hard questions require fine-grained reasoning and precise control over visual, structural, or textual elements.

Manual Verification. To ensure the reliability and quality of the benchmark, we conduct a multi-round manual verification process over the generated prompts and evaluation checklists. Annotators review each task to confirm that the prompt accurately reflects the requirements of its assigned content domain and capability dimension. All verification questions are further examined to ensure they are unambiguous, visually verifiable, and consistent with the prompt specifications. We also verify that all prompts and references contain no harmful, sensitive, or personally identifiable information.

#### 3.3 Statistics of BizGenEval

BizGenEval contains 20 evaluation tasks formed by the intersection of 5 commercial domains and 4 capability dimensions. Each task includes 20 curated

prompts, resulting in 400 generation samples in total. Every prompt is paired with a checklist of 20 binary verification questions (10 easy and 10 hard), yielding 8,000 human-designed verification questions for systematic evaluation.

Figure 3 illustrates the structural composition and textual characteristics of the benchmark. As shown in Figure 3(a), prompts for the three contentbased capabilities (Text, Layout, and Attribute) typically range from 200 to 1400 tokens. This is because these tasks require detailed descriptions of low-level visual constraints such as typography, spatial layout, element counts, and attribute specifications, which naturally lead to longer instructions. In contrast, prompts for Knowledge are shorter (around 100–200 tokens), since the primary goal is to evaluate whether models can correctly incorporate domain knowledge and complete the task semantically rather than reproduce complex visual layouts.

Figure 3(b) further demonstrates the diversity of BizGenEval, where each dimension encompasses a wide spectrum of sub-attributes, from spatial properties like Position to semantic aspects like Fact, ensuring comprehensive coverage of fine-grained visual and logical constraints within each dimension. Finally, the keyword frequency clouds in Figure 3(c) reveal that the most frequent terms in the verification questions align closely with the visual or semantic elements targeted by each capability dimension. This observation supports the validity of our checklist design and indicates that the evaluation questions effectively capture the intended capabilities of each task.

#### 3.4 Evaluation Criteria of BizGenEval

To evaluate complex commercial visual content, we adopt a checklist-based evaluation protocol. For each prompt in BizGenEval, we provide a structured checklist consisting of 20 binary (Yes/No) verification questions. To automatically assess these requirements, we employ a multimodal large language model (MLLM) as the evaluation judge. Given a generated image and its corresponding checklist, the evaluator performs visual reasoning to answer each verification question. Following the benchmark design, the 20 questions are divided into easy and hard subsets, and we report performance on these two tracks separately.

Score for each track is computed using a penalty-based strategy: Score = max(0,1−αNerrors), where Nerrors denotes the number of incorrectly answered questions within the track and α is the penalty coefficient. In our benchmark we set α = 0.2, meaning that when half of the questions in a track (5 out of 10) are answered incorrectly the score becomes zero. Empirically, we observe that only a small portion of models collapse to zero scores, allowing the remaining score range to exhibit clearer performance stratification among models.

### 4 Evaluation Results

#### 4.1 Experimental Setup

We evaluate 26 leading image generation models, including 10 closed-source models and 16 open-source models. The closed-source models are: Nano-Banana-

- Table 1: Performance of Image Generative Models Across Five Commercial Content Domains. Closed-source and open-source models are marked with purple and blue , respectively. Each cell reports scores on the hard/easy testsets.

Model Slides Webpage Poster Chart Sci. Figure Average

Nano-Banana-Pro [11] 82.2 / 94.8 77.5 / 96.5 76.5 / 94.8 73.0 / 92.2 74.2 / 90.0 76.7 / 93.7 Nano-Banana-2.0 [12] 73.8 / 95.8 71.2 / 94.5 67.5 / 91.2 60.2 / 89.2 69.5 / 92.0 68.5 / 92.5 Seedream-5.0 [30] 54.5 / 80.8 47.0 / 80.8 50.7 / 77.0 46.0 / 76.2 45.8 / 81.5 48.8 / 79.2 GPT-Image-1.5 [27] 40.8 / 89.2 41.0 / 86.0 42.0 / 83.5 28.2 / 76.5 27.8 / 72.8 35.9 / 81.6 Seedream-4.5 [29] 33.8 / 71.0 36.2 / 75.5 35.5 / 72.2 18.0 / 47.8 27.3 / 64.8 30.1 / 66.2 Wan2.6-T2I [39] 27.1 / 56.5 25.5 / 67.0 27.5 / 62.5 17.2 / 48.8 12.5 / 58.8 21.9 / 58.7

- Seedream-4.0 [31] 18.5 / 67.8 19.2 / 71.8 18.8 / 65.5 7.8 / 46.0 7.2 / 49.2 14.3 / 60.1 Emu3.5 [5] 14.5 / 44.8 20.0 / 48.8 20.3 / 53.0 4.8 / 20.3 6.5 / 34.0 13.2 / 40.2 HunyuanImage-3.0 [1] 19.3 / 47.0 21.0 / 52.0 19.8 / 53.5 2.0 / 18.8 2.8 / 29.0 13.0 / 40.1 GPT-Image-1.0 [26] 18.2 / 64.8 12.0 / 63.0 17.2 / 64.8 3.2 / 31.2 5.0 / 38.5 11.2 / 52.4

- HunyuanImage-2.1 [37] 11.0 / 36.2 16.5 / 40.0 11.8 / 39.0 1.2 / 8.0 2.2 / 15.0 8.6 / 27.7 Z-Image [38] 12.2 / 43.8 6.2 / 48.5 12.0 / 50.0 8.8 / 30.5 1.8 / 46.0 8.2 / 43.8 Qwen-Image-2512 [42] 10.2 / 45.0 7.2 / 48.0 11.5 / 47.0 2.2 / 28.0 0.2 / 37.0 6.3 / 41.0 FLUX.2-dev [18] 5.5 / 43.2 5.5 / 48.8 7.2 / 48.2 5.5 / 33.5 0.5 / 36.5 4.9 / 42.0 Z-Image-Turbo [38] 7.0 / 36.5 4.5 / 45.8 4.8 / 40.8 0.8 / 15.5 0.0 / 23.2 3.4 / 32.4 Qwen-Image [42] 3.5 / 28.5 2.5 / 27.5 5.8 / 32.2 2.2 / 15.8 0.0 / 15.2 2.8 / 23.8 FLUX.2-Pro [18] 1.8 / 23.0 2.0 / 27.0 2.8 / 26.2 1.3 / 14.7 0.0 / 14.3 1.6 / 21.1 GLM-Image [44] 1.5 / 22.8 1.8 / 27.3 3.8 / 24.8 0.0 / 0.5 0.0 / 1.5 1.4 / 15.3 Imagen-4 [6] 1.5 / 15.0 0.5 / 14.8 1.8 / 12.8 1.5 / 6.8 0.2 / 4.3 1.1 / 10.7 LongCat-Image [35] 0.8 / 15.0 1.8 / 22.3 0.8 / 21.3 0.0 / 2.5 0.0 / 3.8 0.7 / 12.0 X-Omni-EN [9] 0.8 / 9.0 0.2 / 14.2 1.5 / 15.8 0.0 / 6.5 0.0 / 1.8 0.5 / 9.4 SD3.5-Large [33] 0.0 / 0.5 0.2 / 2.2 0.0 / 2.8 1.2 / 3.2 1.2 / 1.5 0.5 / 2.1 Bagel [7] 0.3 / 4.8 0.0 / 5.0 0.5 / 8.5 0.0 / 0.0 0.0 / 0.0 0.3 / 3.7 FLUX.1-Krea-dev [20] 0.0 / 4.8 0.0 / 8.5 0.2 / 11.8 0.0 / 0.8 0.0 / 0.0 0.1 / 5.1

- FLUX.1-dev [17] 0.0 / 6.0 0.0 / 11.0 0.5 / 7.8 0.0 / 0.2 0.0 / 0.0 0.1 / 5.0

- FLUX.1-schnell [17] 0.0 / 8.5 0.0 / 8.2 0.0 / 8.2 0.0 / 0.8 0.0 / 0.0 0.0 / 5.1

Pro [11], Nano-Banana-2.0 [12], GPT-Image-1.5 [27], GPT-Image-1.0 [26], Seedream [29–31], Wan2.6-T2I [39], FLUX.2-Pro [18] and Imagen-4 [6]. The opensource models include: Z-Image [38], FLUX.2-dev [18], Qwen-Image [42], Emu3.5 [5], HunyuanImage [1,37], GLM-Image [44], X-Omni [9], FLUX.1-schnell, FLUX.1Krea-dev [20], FLUX.1-dev [17], Bagel [7], LongCat-Image [35] and SD3.5-Large [33]. For all models, we use the default inference settings provided by their official APIs or documentation. The generation resolution is adaptively set to the closest supported aspect ratio of the ground-truth images to ensure structural alignment with real-world designs. Gemini-3-Flash-Preview [13] is used as our automated evaluator. To reduce API overhead, all 20 checklist questions are answered in a single query. We run evaluation with a single pass per sample; despite MLLM stochasticity, repeated trials show very low variance. Gemini-3-Flash-Preview exhibits behavior similar to Gemini-3-Pro and performs better than GPT-5.2 on structured visual tasks such as chart interpretation. Detailed evaluator analysis is provided in the Appendix.

###### Chart Webpage Poster Slides Scientific Figure

- Q1:In (a), is there a row of seven soft pink rounded rectangles labeled “CLS”, ..., “6” from left to right?",
- Q2:In (a), is there one orange rounded rectangle labeled “MLP Head” to the right of the “Transformer Encoder”, connected by a solid black right arrow?
- Q3:In (b), are there two dashed crimson arrows from the first Norm to A1 and from the second Norm to A2?

- Q1:Does the main title located at ..., read exactly: 'THE COCA-COLA CHRONICLE'?
- Q2:Does the rightmost era callout in the top-right portion of the poster contain the heading '1990s-Present: Innovation & Sustainability' ...?
- Q3:Does the three-line caption beneath the third bottle from the left ... , read exactly: '1915'...?

- Q1:Does the top-left hero section have side-by-side text and images?
- Q2:Does the bottom-left learning section feature left bullets and a right circular photo?
- Q3:Does the audience section have three cards in one horizontal row?

- Q1:In the left legend, is "Encoder C" shown in crimson, and are its plotted marks also crimson?
- Q2:At CLIP Score 0.24 (left panel), do markers show 14 (A), 13 (B), 12 (C), 11 (D), 12 (E)?
- Q3:Check if the right panel labels (62%, 43%, 52%, 47%) align with bars and match axis scales.

- Q1:Is the left main column wider than the right main column within the central content area?",
- Q2:In the same narrow left subcolumn, are there exactly three client segment markers arranged vertically from top to bottom along a curved guide?
- Q3:Inside the charts container, are there two bar-chart graphics placed side-by-side in one row?

[Figure 49]

[Figure 50]

[Figure 51]

1

###### 1

- 1

| |
|---|

- 2

[Figure 52]

###### NanoBanan-Pro

###### 2

1

3

3

|[Figure 53]<br><br>| |
|---|
<br><br>2<br><br>| |
|---|
<br><br>3|
|---|

3

| |
|---|

| |
|---|

2

3

2

[Figure 54]

[Figure 55]

- 1

| |
|---|

- 2

[Figure 56]

1

[Figure 57]

[Figure 58]

###### GPT-Image-1.5

| |
|---|

- 1 3

- 2

3

1

2

| |
|---|

3

1

| |
|---|

| |
|---|

|| |
|---|
<br><br>2<br><br>| |
|---|
<br><br>3|
|---|

2

3

3

1

Qwen-Image-2512

[Figure 59]

| |
|---|

[Figure 60]

###### 1

| |
|---|

[Figure 61]

1 1

1

|3<br><br>[Figure 62]<br><br>| |
|---|
<br><br>2<br><br>| |
|---|
<br><br>3|
|---|

| |
|---|

3

2

2

| |
|---|

2

3

| |
|---|

###### 2

3

- Fig. 4: Qualitative evaluation of different commercial image generation models across five content domains. Columns represent content domains, and rows show model outputs. Evaluation questions are listed in the top row, with correct and incorrect regions highlighted in blue and red boxes.

#### 4.2 Main Results

Results for Content Domains. The results in Table 1 highlight a pronounced sensitivity of current models to the distinct requirements of various commercial scenarios. Closed-source models, particularly Nano-Banana-Pro [11] and Nano-Banana-2.0 [12], consistently achieve high scores across all domains, with hard-set averages around 68–77%. Slides, Webpages and Posters are the bestperforming domains, which appear well-captured by the models’ training distributions. In contrast, Charts and Scientific Figures present greater challenges, with a noticeable drop in hard-set scores (e.g., GPT-Image-1.5 [27] drops to 28.2–27.8%), reflecting the difficulty of precise numerical rendering and structured diagram generation. Open-source models generally lag behind, with ZImage [38], FLUX.2-dev [18], and Qwen-Image-2512 [42] achieving mid-range performance on Slides and Webpages (easy scores 43–49%) but near-zero results on Scientific Figures (hard scores 0–2%), indicating significant gaps in generating complex professional content.

As illustrated in Fig. 4, the qualitative gap between models is most evident in high-precision domains like Charts and Scientific Figures, where the ability to bind specific numerical values to spatial coordinates is a key differentiator. For example, in the Chart task (Q2), where models are asked to render “specific markers (14, 13, 12, 11, 12) at a CLIP Score of 0.24”. Nano-Banana-Pro [11] accurately plots and labels each individual value at its correct position. In contrast,

- GPT-Image-1.5 [26] exhibits a “homogenization” error, rendering all markers as

##### Table 2: Performance of Image Generative Models Across Four Capability Dimensions.

###### Model Layout Attribute Text Knowledge Average

Nano-Banana-Pro [11] 72.2 / 91.2 65.6 / 92.2 86.4 / 95.0 82.6 / 96.2 76.7 / 93.7 Nano-Banana-2.0 [12] 68.4 / 91.0 57.4 / 91.6 83.4 / 94.6 64.6 / 93.0 68.5 / 92.5

- Seedream-5.0 [30] 67.6 / 89.0 42.4 / 77.2 43.4 / 75.6 41.8 / 75.2 48.8 / 79.2

- GPT-Image-1.5 [27] 51.6 / 84.8 25.8 / 75.2 40.4 / 82.8 26.0 / 83.6 35.9 / 81.6 Seedream-4.5 [29] 35.4 / 71.6 22.4 / 62.8 41.4 / 72.4 21.4 / 58.2 30.1 / 66.2 Wan2.6-T2I [39] 46.4 / 80.6 16.6 / 60.6 12.6 / 52.6 12.2 / 41.0 21.9 / 58.7 Seedream-4.0 [31] 27.6 / 73.4 11.4 / 59.2 11.4 / 52.8 6.8 / 54.8 14.3 / 60.1 Emu3.5 [5] 30.4 / 63.4 14.2 / 52.6 7.0 / 33.6 1.2 / 11.0 13.2 / 40.2

- HunyuanImage-3.0 [1] 27.8 / 65.0 13.8 / 53.6 10.2 / 39.6 0.0 / 2.0 13.0 / 40.1 GPT-Image-1.0 [26] 21.4 / 60.2 6.8 / 48.6 8.6 / 41.0 7.8 / 60.0 11.2 / 52.4 HunyuanImage-2.1 [37] 29.0 / 68.4 5.2 / 39.8 0.0 / 1.4 0.0 / 1.0 8.6 / 27.7 Z-Image [38] 26.8 / 69.2 2.6 / 47.6 2.8 / 45.0 0.6 / 13.2 8.2 / 43.8 Qwen-Image-2512 [42] 22.2 / 70.6 1.2 / 47.8 1.8 / 39.2 0.0 / 6.4 6.3 / 41.0

- FLUX.2-dev [18] 17.2 / 67.8 1.2 / 49.2 1.0 / 43.0 0.0 / 8.2 4.9 / 42.0 Z-Image-Turbo [38] 11.0 / 60.6 1.2 / 35.0 1.2 / 29.8 0.2 / 4.0 3.4 / 32.4 Qwen-Image [42] 10.4 / 51.2 0.2 / 22.2 0.6 / 17.6 0.0 / 4.4 2.8 / 23.8

- FLUX.2-Pro [18] 6.1 / 36.2 0.0 / 22.9 0.0 / 13.7 0.2 / 11.7 1.6 / 21.1 GLM-Image [44] 5.4 / 43.2 0.0 / 13.4 0.2 / 4.4 0.0 / 0.4 1.4 / 15.3 Imagen-4 [6] 4.2 / 26.8 0.0 / 8.7 0.2 / 4.0 0.0 / 3.4 1.1 / 10.7 LongCat-Image [35] 2.4 / 35.8 0.2 / 11.6 0.0 / 4.4 0.0 / 0.0 0.7 / 13.0 SD3.5-Large [33] 2.2 / 6.6 0.0 / 0.4 0.0 / 0.0 0.0 / 1.2 0.5 / 2.1 X-Omni-EN [9] 2.0 / 22.8 0.0 / 5.6 0.0 / 8.0 0.0 / 1.4 0.5 / 9.4 Bagel [7] 0.6 / 12.8 0.0 / 1.6 0.0 / 0.0 0.0 / 0.2 0.2 / 3.7 FLUX.1-Krea-dev [20] 0.2 / 17.8 0.0 / 2.8 0.0 / 0.0 0.0 / 0.0 0.1 / 5.1 FLUX.1-dev [17] 0.4 / 15.8 0.0 / 2.8 0.0 / 0.0 0.0 / 1.4 0.1 / 5.0 FLUX.1-schnell [17] 0.0 / 16.8 0.0 / 2.6 0.0 / 0.0 0.0 / 1.2 0.0 / 5.1

the same incorrect value (“12”), while Qwen-Image-2512 [42] fails fundamentally by omitting the numerical markers entirely.

Results for Capability Dimensions. Table 2 reports model performance across the four capability dimensions: Layout Control, Attribute Binding, Text Rendering, and Knowledge-based Reasoning. Closed-source models generally perform well on Text Rendering and Knowledge-based Reasoning. For instance, Nano-Banana-Pro [11] achieves 86.4/95.0 on Text Rendering and 82.6/96.2 on Knowledge-based Reasoning, indicating strong textual fidelity and domain reasoning. In contrast, Layout Control and Attribute Binding remain more challenging, especially on the hard subsets (e.g., 72.2 for Layout and 65.6 for Attribute), revealing persistent limitations in fine-grained spatial reasoning and precise property control. This difficulty is more evident for open-source models. While Z-Image [38] and FLUX.2-dev [18] retain basic layout capability (easy scores around 67.8–69.2%), their performance on Knowledge-based Reasoning and Attribute Binding drops sharply, often approaching zero on hard questions.

Attribute Text

###### Knowledge

###### Layout

- Q1:Is each panel label ((A), ...) placed at the top-left inside its own panel rather than outside the panel frame?
- Q2: Is the top edge of panel (A) horizontally aligned with the top edge of panel (B)?
- Q3:Are the left edges of panels (B) and (C) vertically aligned with each other?
- Q4:In panel (D), are there four KPI cards arranged horizontally in a single row within the footer band?

- Q1:Between 'MLP Bottleneck' and 'MLP Head', are there 3 short vertical white rectangles ... side-by-side?
- Q2: In the encoder inset, is one purple diamond in 'Patch Merge' and a purple '×N' near the lower-right?
- Q3:Is there a light gray rounded rectangle labeled 'MLP Bottleneck' immediately to the right of 'Transformer Stage 4’?
- Q4:Is the black annotation 'H/4 × W/4 × Ncls' placed above the 'Class Scores'?

- Q1:Does the depicted definition of a solar calendar correctly described?
- Q2:Does the representation of seasons accurately link to solar phenomena?
- Q3:Does the depicted cumulative drift correctly demonstrate a purely lunar calendar progressively misaligning from the solar baseline?
- Q4:Are the historical civilizations assigned to the calendar categories factually distinct from one another and correctly matched?

- Q1: Does the subtitle include the exact text "stacked totals show overall scale over time"
- Q2: Does the legend contain the label exactly "Material recovery"?
- Q3:Is there a total label "2.40" shown at the top of year 2002?
- Q4:Does the bottom-right footnote read exactly "Footnote: Units are million tonnes per year; totals are the sum of the four methods at each year."

[Figure 63]

[Figure 64]

[Figure 65]

1 2

NanoBanan-Pro

1 2

[Figure 66]

4

3

1

- 3

- 4

3

- 1

- 2 3

4

2

4

[Figure 67]

[Figure 68]

[Figure 69]

1

[Figure 70]

1 2

4

###### GPT-Image-1.5

1

1 4

2

3

3

3

3

2

2

4

4

[Figure 71]

[Figure 72]

[Figure 73]

1 2

Qwen-Image-2512

1

[Figure 74]

1

4

2 3

4

3

2

3

4

3

2

4

1

- Fig. 5: Qualitative evaluation of different commercial image generation models across four capability dimensions.

These results suggest that although modern models can generate legible text and perform basic semantic reasoning, producing structurally precise and attributeconsistent designs remains a key challenge for commercial visual generation.

As illustrated in Fig. 5, qualitative examples further highlight these limitations. Even top-tier models struggle with complex layout constraints and precise attribute control. Although Nano-Banana-Pro [11] produces more coherent structures, it still fails in highly specific spatial instructions (red boxes). For example, the Layout task (Q1) requires “placing panel labels (e.g., ‘(A)’) strictly inside the top-left corner of each panel frame”, yet all three models fail to maintain the required boundary logic.

Human Evaluation. We conducted a human evaluation on a subset of benchmark tasks. From the benchmark pool, 2,000 questions were randomly sampled (400 per model) covering Nano-Banana-Pro [11], GPT-Image-1.5 [27], Seedream5.0 [30], Qwen-Image-2512 [42], and Z-Image [38]. For each participant, two questions per model were randomly selected (10 questions total). Two additional fixed, intentionally simple reference questions were included to detect unreliable responses, resulting in 12 questions per participant. The study involved 59

participants with prior experience in visual design or data interpretation. After removing 9 responses that failed the quality checks, the observed agreement was po = 90.88% with Cohen’s κ = 0.7692, indicating strong consistency between the automated evaluator and human judgments.

#### 4.3 Main Findings

1) Stylistic Generation Does Not Imply Precise Composition. Although current generative models capture the high-level stylistic patterns of commercial documents well, they still lack deterministic control required for precise composition. This limitation appears in both spatial geometry and attribute binding. As in Fig. 4, even top-tier models such as Nano-Banana-Pro [11] struggle with tasks requiring precise element localization (e.g., Layout Q1&Q2) and accurate attribute counting (e.g., Attribute Q1&Q2), indicating that many models approximate layouts stylistically rather than enforcing strict structural constraints.

- 2) Text and Knowledge Capabilities Are Highly Polarized Across Models.

Top-tier commercial APIs such as Nano-Banana-Pro [11] (86.4/95.0 in Text and 82.6/96.2 in Knowledge) achieve strong results on language-intensive tasks, likely benefiting from integration with multimodal foundation models. For example, Gemini-Image-Pro documentation states that Nano Banana Pro is built on Gemini 3 Pro and leverages its reasoning and world knowledge for visual generation. In contrast, most models perform dramatically worse: as shown in Table 2, 21 out of 26 evaluated models score below 12.6 in both Text and Knowledge, with several approaching zero. Notably, all open-source models fall into this regime, revealing both the difficulty of accurate text and knowledge grounding and a capability gap between open-source models and closed-source commercial APIs.

Table 3: Cross-benchmark comparison.

Model GenEval [10] OneIG-EN [2] BizGenEval

HunyuanImage-3.0 [1] 0.72 - 13.0 / 40.1 GPT-Image-1.0 [26] 0.84 0.533 11.2 / 52.4 Z-Image [38] 0.84 0.546 8.2 / 43.8 Qwen-Image [42] 0.87 0.539 2.8 / 23.8 LongCat-Image [35] 0.87 - 0.7 / 12.0

- 3) Natural Image Competence

Does Not Transfer to Commercial Documents. Models trained on natural images show clear limitations when applied to commercial visual documents. As shown in Table 3, GPT-Image-1.0 and Qwen-Image achieve strong scores on GenEval [10] (0.84 and 0.87 respectively), yet perform differently on BizGenEval (11.2/52.4 vs. 2.8/23.8). This discrepancy indicates that strong performance on natural-image benchmarks does not directly translate to professional design scenarios, which require dense text rendering, structured layouts, and multiple compositional constraints. Although OneIG-Bench [2] includes prompts related to slides or posters, its score differences remain relatively small and fail to reveal the substantial capability gaps exposed by BizGenEval.

### 5 Conclusion

In this work, we introduced BizGenEval, a systematic benchmark for evaluating image generation models on commercial visual content creation. By covering five representative document types and four capability dimensions, BizGenEval provides a structured evaluation framework with diverse tasks, carefully curated prompts, and rigorous checklist-based verification. Through large-scale benchmarking of both state-of-the-art commercial APIs and open-source models, our results reveal substantial gaps between current generative systems and the requirements of real-world commercial design tasks, particularly in structured layout composition, precise attribute control, and complex multi-constraint generation. We hope BizGenEval can serve as a standardized benchmark to facilitate future research on practical visual content generation and promote the development of generative models better aligned with real-world design applications.

### References

- 1. Cao, S., Chen, H., Chen, P., Cheng, Y., Cui, Y., Deng, X., Dong, Y., Gong, K., Gu, T., Gu, X., et al.: Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951 (2025) 10, 12, 14
- 2. Chang, J., Fang, Y., Xing, P., Wu, S., Cheng, W., Wang, R., Zeng, X., Yu, G., Chen, H.B.: Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arxiv:2506.07977 (2025) 2, 5, 14
- 3. Chen, K., Lin, Z., Xu, Z., Shen, Y., Yao, Y., Rimchala, J., Zhang, J., Huang, L.: R2I-Bench: Benchmarking reasoning-driven text-to-image generation. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 12595–12630. Association for Computational Linguistics (2025). https://doi.org/10.18653/v1/2025.emnlp-main.636, https://aclanthology. org/2025.emnlp-main.636/ 5
- 4. Cho, J., Li, L., Yang, Z., Gan, Z., Wang, L., Bansal, M.: Diagnostic benchmark and iterative inpainting for layout-guided image generation. arXiv preprint arXiv:2304.06671 (2023). https://doi.org/10.48550/arXiv.2304.06671, https: //arxiv.org/abs/2304.06671, LayoutBench 5
- 5. Cui, Y., Chen, H., Deng, H., Huang, X., Li, X., Liu, J., Liu, Y., Luo, Z., Wang, J., Wang, W., Wang, Y., Wang, C., Zhang, F., Zhao, Y., Pan, T., Li, X., Hao, Z., Ma, W., Chen, Z., Ao, Y., Huang, T., Wang, Z., Wang, X.: Emu3.5: Native multimodal models are world learners (2025), https://arxiv.org/abs/2510.26583 10, 12
- 6. DeepMind, G.: Imagen: Our leading text-to-image model, engineered for creativity. https://deepmind.google/models/imagen/ (2025), model: Imagen-4 10, 12
- 7. Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., Shi, G., Fan, H.: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025) 10, 12
- 8. Du, N., Chen, Z., Chen, Z., Gao, S., Chen, X., Jiang, Z., Yang, J., Tai, Y.: Textcrafter: Accurately rendering multiple texts in complex visual scenes (2025). https://doi.org/10.48550/arXiv.2503.23461, https://arxiv.org/abs/2503. 23461 5
- 9. Geng, Z., Wang, Y., Ma, Y., Li, C., Rao, Y., Gu, S., Zhong, Z., Lu, Q., Hu, H., Zhang, X., Linus, Wang, D., Jiang, J.: X-omni: Reinforcement learning

- makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058 (2025) 5, 10, 12
- 10. Ghosh, D., Hajishirzi, H., Schmidt, L.: GenEval: An object-focused framework for evaluating text-to-image alignment. arXiv preprint arXiv:2310.11513 (2023). https://doi.org/10.48550/arXiv.2310.11513, https://arxiv.org/abs/2310. 11513 2, 5, 14
- 11. Google: Introducing nano banana pro. https://blog.google/innovation-andai/products/nano-banana-pro/ (2025) 1, 10, 11, 12, 13, 14
- 12. Google: Nano banana 2: Combining pro capabilities with lightning-fast speed. https://blog.google/innovation-and-ai/technology/ai/nano-banana-2/

(2026) 10, 11, 12

- 13. Google DeepMind: Gemini 3 flash: Best for frontier intelligence at speed. https: //deepmind.google/models/gemini/flash (2025) 10
- 14. Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., Yu, G.: Ella: Equip diffusion models with llm for enhanced semantic alignment (2024), https://arxiv.org/abs/2403. 05135 2
- 15. Huang, K., Sun, K., Xie, E., Li, Z., Liu, X.: T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems 36, 78723–78747 (2023) 2
- 16. Izzo, E., Parolari, L., Vezzaro, D., Ballan, L.: 7Bench: a comprehensive benchmark for layout-guided text-to-image models. arXiv preprint arXiv:2508.12919 (2025). https://doi.org/10.48550/arXiv.2508.12919, https://arxiv.org/abs/2508. 12919 5
- 17. Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024) 10, 12
- 18. Labs, B.F.: FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2

(2025) 10, 11, 12

- 19. Laurençon, H., Tronchon, L., Sanh, V.: Unlocking the conversion of web screenshots into HTML code with the WebSight dataset (2024). https://doi.org/10.48550/ arXiv.2403.09029, https://arxiv.org/abs/2403.09029 4
- 20. Lee, S., Ebbecke, T., Millon, E., Beddow, W., Zhuo, L., García-Ferrero, I., Esparraguera, L., Petrescu, M., Saß, G., Menezes, G., Perez, V.: Flux.1 krea [dev]. https://github.com/krea-ai/flux-krea (2025) 10, 12
- 21. Li, B., Lin, Z., Pathak, D., Li, J., Fei, Y., Wu, K., Ling, T., Xia, X., Zhang, P., Neubig, G., Ramanan, D.: GenAI-Bench: Evaluating and improving compositional text-to-visual generation. arXiv preprint arXiv:2406.13743 (2024). https://doi. org/10.48550/arXiv.2406.13743, https://arxiv.org/abs/2406.13743 5
- 22. Li, B., Wang, C.Y., Xu, H., Zhang, X., Armand, E., Srivastava, D., Shan, X., Chen, Z., Xie, J., Tu, Z.: OverLayBench: A benchmark for layout-to-image generation with dense overlaps. arXiv preprint arXiv:2509.19282 (2025). https://doi.org/ 10.48550/arXiv.2509.19282, https://arxiv.org/abs/2509.19282 5
- 23. Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., Ramanan, D.: Evaluating text-to-visual generation with image-to-text generation. arXiv preprint arXiv:2404.01291 (2024). https://doi.org/10.48550/arXiv.2404.01291, https: //arxiv.org/abs/2404.01291 5
- 24. Luo, Y., Yuan, Y., Chen, J., Cai, H., Yue, Z., Yang, Y., Daha, F.Z., Li, J., Lian, Z.: MMMG: A massive, multidisciplinary, multi-tier generation benchmark for textto-image reasoning. arXiv preprint arXiv:2506.10963 (2025). https://doi.org/ 10.48550/arXiv.2506.10963, https://arxiv.org/abs/2506.10963 5
- 25. Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Feng, C., Ning, K., Zhu, B., Yuan, L.: WISE: A world knowledge-informed semantic evaluation for

- text-to-image generation. arXiv preprint arXiv:2503.07265 (2025). https://doi. org/10.48550/arXiv.2503.07265, https://arxiv.org/abs/2503.07265 5
- 26. OpenAI: Introducing 4o image generation. https : / / openai . com / index / introducing-4o-image-generation/ (2025), model: GPT-Image-1 10, 11, 12, 14
- 27. OpenAI: The new chatgpt images is here. https://openai.com/index/newchatgpt-images-is-here/ (2025) 1, 2, 10, 11, 12, 13
- 28. Peng, Y., Xiao, S., Wu, K., Liao, Q., Chen, B., Lin, K., Huang, D., Li, J., Yuan, Y.: BizGen: Advancing article-level visual text rendering for infographics generation. arXiv preprint arXiv:2503.20672 (2025). https://doi.org/10.48550/arXiv. 2503.20672, https://arxiv.org/abs/2503.20672 2, 4
- 29. Seed, B.: Seedream 4.5. https://seed.bytedance.com/en/seedream4_5 (2024) 10, 12
- 30. Seed, B.: Seedream 5.0 lite. https://seed.bytedance.com/en/seedream5_0_lite

(2026) 2, 10, 12, 13

- 31. Seedream, T., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al.: Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427 (2025) 10, 12
- 32. Si, C., Zhang, Y., Li, R., Yang, Z., Liu, R., Yang, D.: Design2code: Benchmarking multimodal code generation for automated front-end engineering. arXiv preprint arXiv:2403.03163 (2024). https://doi.org/10.48550/arXiv.2403.03163, https: //arxiv.org/abs/2403.03163 4
- 33. Stability-AI: sd3.5 GitHub repository. https://github.com/Stability-AI/sd3.5

(2026) 10, 12

- 34. Tang, Y., Liu, X., Zhang, B., Lan, T., Xie, Y., Lao, J., Wang, Y., Li, H., Gao, T., Pan, B., Weng, L., Huang, X., Zhu, M., Feng, Y., Luo, Y., Chen, W.: IGenBench: Benchmarking the reliability of text-to-infographic generation. arXiv preprint arXiv:2601.04498 (2026). https://doi.org/10.48550/arXiv.2601.04498, https: //arxiv.org/abs/2601.04498 2, 4
- 35. Team, M.L., Ma, H., Tan, H., Huang, J., Wu, J., He, J.Y., Gao, L., Xiao, S., Wei, X., Ma, X., Cai, X., Guan, Y., Hu, J.: Longcat-image technical report. arXiv preprint arXiv:2512.07584 (2025) 2, 10, 12, 14
- 36. Team, Q.: Qwen-image-2.0: Professional infographics, exquisite photorealism. https://qwen.ai/blog?id=qwen-image-2.0 (2025) 2
- 37. Team, T.H.: Hunyuanimage 2.1: An efficient diffusion model for highresolution (2k) text-to-image generation. https://github.com/Tencent-Hunyuan/ HunyuanImage-2.1 (2025) 10, 12
- 38. Team, Z.I.: Z-image: An efficient image generation foundation model with singlestream diffusion transformer. arXiv preprint arXiv:2511.22699 (2025) 10, 11, 12, 13, 14
- 39. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang,

- X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li,
- Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025) 10, 12

- 40. Wang, Z., Yin, P., Zhao, X., Tian, C., Qiao, Y., Wang, W., Dai, J., Luo, G.: Genexam: A multidisciplinary text-to-image exam. arXiv preprint arXiv:2509.14232 (2025) 5
- 41. Wei, X., Zhang, J., Wang, Z., Wei, H., Guo, Z., Zhang, L.: TIIF-Bench: How does your T2I model follow your instructions? arXiv preprint arXiv:2506.02161 (2025). https://doi.org/10.48550/arXiv.2506.02161, https://arxiv.org/abs/2506. 02161 5
- 42. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., Liu, Z.: Qwen-image technical report (2025), https://arxiv.org/abs/2508.02324 10, 11, 12, 13, 14
- 43. Yang, Y., Li, W., Ren, H., Lu, Z., Wang, K., Huang, Z., Zong, Z., Zhan, M., Li, H.: SlidesGen-Bench: Evaluating slides generation via computational and quantitative metrics. arXiv preprint arXiv:2601.09487 (2026). https://doi.org/10.48550/ arXiv.2601.09487, https://arxiv.org/abs/2601.09487 4
- 44. Z.ai: Glm-image: Auto-regressive for dense-knowledge and high-fidelity image generation. https://z.ai/blog/glm-image/ (2026), model: GLM-Image 2, 10, 12
- 45. Zhang, D., Jiang, C., Xu, R., Chen, B., Jin, Z., Lu, Y., Zhang, J., Yong, L., Luo, J., Luo, S.: WorldGenBench: A world-knowledge-integrated benchmark for reasoningdriven text-to-image generation. arXiv preprint arXiv:2505.01490 (2025). https: //doi.org/10.48550/arXiv.2505.01490, https://arxiv.org/abs/2505.01490 5
- 46. Zhu, W., Deng, Z., Chen, C., Chen, X., Jin, W., Xiong, H., Gao, J., Yuan, L.: AutoFigure: Towards autonomous figure generation and understanding in scientific papers. arXiv preprint arXiv:2602.03828 (2026). https://doi.org/10.48550/ arXiv.2602.03828, https://arxiv.org/abs/2602.03828 4

### Supplemental Materials for BizGenEval

In these supplemental materials, we provide additional experimental data, model performance rankings, comprehensive evaluator reliability studies and extended qualitative analyses. The contents are organized as follows:

- – Sec. A: Visual examples of BizGenEval samples, illustrating the diverse prompts and corresponding multi-dimensional checklist questions.
- – Sec. B: Extensive model performance rankings across all five content domains and four capability dimensions.
- – Sec. C: In-depth analysis of MLLM-based evaluation, including stability tests and alignment with human judgment.

### A Visual Examples of BizGenEval

We present representative samples from BizGenEval in Figures 6–9. Each sample consists of a detailed prompt and its corresponding human-verified verification questions. These examples span the four key capability dimensions: Knowledge-based Reasoning (Fig. 6), Attribute Binding (Fig. 7), Layout Control (Fig. 8), and Text Rendering (Fig. 9).

### B Model Performance Comparison Across Domains

We provide a comprehensive performance ladder visualization to detail the landscape of current model capabilities. Figure 10 shows the overall ranking, Figures 11-15 present performance rankings across five professional content domains, and Figures 16-19 show the aggregated results across four capability dimensions. These rankings establish a clear hierarchy of current state-of-the-art models and highlight the relative difficulty of realistic commercial design tasks.

#### B.1 Comparison across content domains

We evaluate the performance across specific application scenarios, ranking models based on their ability to handle domain-specific constraints in Webpage, Slides, Chart, Poster, and Scientific Figure as shown in Figures 11, 12, 13, 14 and 15, respectively. For each model presented in the rankings, the bar with deeper color represents the performance on the “hard” subset, while lighter color denotes the “easy” subset score.

#### B.2 Comparison across capability dimensions

To understand the underlying technical strengths of the models, we present results across four fundamental dimensions, including Layout Control (Fig. 16), Attribute Binding (Fig. 17), Text Rendering (Fig. 18), and Knowledge-based Reasoning (Fig. 19).

Prompt: Generate a publication-style figure (panels A–F) titled “Probability (Harder): Bayes’ Theorem and the Base Rate Fallacy”. Problem statement (Panel A): - A disease has a prevalence of 0.1% in the general population. A diagnostic screening test has a sensitivity of 99% and a false positive rate of 5%. If a random person tests positive, what is the exact probability they actually have the disease? Panel requirements: - Panel A: A complete probability tree diagram showing Disease (D) and No Disease (∼D) branches, followed by Test Positive (+) and Test Negative (-) sub-branches, with all edge weights labeled. - Panel B: The formal mathematical statement of Bayes’ Theorem for P(D | +). - Panel C: Calculation of the total marginal probability of a positive test P(+), explicitly separating the True Positive and False Positive components. - Panel D: Numeric substitution into the Bayes’ formula. - Panel E: The final calculated probability expressed as a percentage. - Panel F: An array diagram or pie chart visually demonstrating the ’Base Rate Fallacy’ (showing physically why the vast majority of positive results in this scenario are false positives). Content requirements: - The probability logic and arithmetic must be flawlessly executed. No placeholders; strict journal style.

[Figure 75]

- Q1: Is the total marginal probability of a positive test correctly calculated as exactly 0.05094?
- Q2: Do the Bayes formula values exactly match the corresponding probability tree edge weights?
- Q3: Is the final calculated probability of having the disease given a positive test mathematically evaluated to approximately 0.0194?
- Q4: Is the conditional probability for testing positive given no disease correctly labeled with the false positive rate of exactly 0.05 or 5%?

- 1

- 2

- 3

4

- Fig. 6: Example Task: Knowledge-based Reasoning. Assessment of a model’s ability to incorporate domain knowledge into visual generation.

### C Evaluator Stability Analysis

#### C.1 Human-Alignment and Case Studies

As shown in Table 4, the selected MLLM judge exhibits a high degree of correlation with expert human annotations. Our primary evaluator, Gemini-3-Flash, achieves the highest agreement and Cohen’s Kappa, significantly outperforming other models in the checklist-based protocol.

Figures 20-21 further provide qualitative comparisons between different MLLM

judges. These cases illustrate the superior visual grounding and reasoning depth of the selected evaluator. For instance, the judge accurately verifies complex spatial constraints, such as the placement of a two-line caption block below specific scene levels (Fig. 20), and performs precise element counting, such as verifying the exact presence of 13 vertical bars against 13 categories on a chart’s x-axis (Fig. 21). Such detailed rationales demonstrate the robustness of using state-ofthe-art MLLMs to identify fine-grained errors and spatial inconsistencies that baseline judges often overlook.

Prompt: Create a flat 2D infographic on a solid light cream background... near the top, draw a wide horizontal rounded rectangle... filled with a solid burnt orange color... center the white all-caps title “BUSINESS NAME GOES HERE”; directly below... add a second wide horizontal rounded rectangle... filled with solid dark teal... center within it all-caps white text “CUSTOMER SERVICE PILLARS”; beneath this... arrange a row of exactly seven rounded squares, each a bright warm pink tile... center in each a black line icon: 1. speech bubbles, 2. trophy, 3. silhouettes with heart, 4. human head, 5. light bulb with gear, 6. handshake, and 7. eye; directly below these... place a second row of seven vertical rounded rectangles... filled with solid soft tangerine orange; in the first rectangle, write heading “COMMUNICATION” and body text “Clear and professional verbal or written communication...”; in the remaining six... center the heading “PILLAR GOES HERE” and placeholder text “Supporting details and examples go here.”; ensure all text uses a modern geometric sans-serif... forming a precise seven-column grid with equal horizontal spacing.

[Figure 76]

1

- Q1: Is there a wide burnt-orange rounded rectangle at the top with centered white text?
- Q2: Are there exactly seven pink rounded squares arranged in a single horizontal row?
- Q3: Does the fourth pink square contain a centered black line icon of a side-profile human head with a brain-like pattern inside?
- Q4: Are all text elements throughout the image set in a clean sans-serif typeface with headings in bold weight and body text in regular weight, and no italics used?

2

3

- Fig. 7: Example Task: Attribute Binding. Evaluation of precise associations between specific visual elements and their described properties.

#### C.2 Stability Analysis

We further verify the stability of our protocol in Table 5 using Gemini-3-Flash as the judge. The minimal variance observed across three independent evaluation passes confirms that the checklist-based scoring system is stable.

Prompt: Create a clean two-panel scientific figure... on a white background... Panel (a) is on the left...caption ’(a) Model architecture’...panel (b) is on the right...caption ’(b) Detection example’... In panel (a)... four large rounded-rectangle regions... labeled: ’backbone’, ’encoder’, ’decoder’, and ’prediction heads’... Inside ’backbone’, place... overlapping light-gray rectangles labeled ’input image’... draw one black solid arrow... to a right-pointing trapezoid filled orange labeled ’CNN’... draw one black solid arrow... to a tall thin vertical rectangle labeled ’image features’... Below, draw... rounded rectangle with a teal gradient fill labeled ’positional encoding’... connect it to a small circle... containing a plus sign ’+’ using a black solid arrow, and also connect ’image features’ to the same plus-circle... From the plus circle, draw a short dotted line to a single horizontal row... of six small light-gray squares... from the right side... draw one black solid arrow into a large rectangle labeled ’transformer encoder’ located inside the ’encoder’ region... To the right inside ’decoder’, draw one large rectangle labeled ’transformer decoder’ and connect the encoder box to the decoder box... Under the decoder box, draw exactly six small square icons in one row for queries, colored... In the top-left corner... add a tiny legend... ’bird 1’ and... ’bird 2’.

1

- Q1: Are there four dashed rectangular regions arranged horizontally from left to right and labeled 'backbone', 'encoder', 'decoder', and 'prediction heads'?
- Q2: Is there a rectangle labeled transformer decoder to the right of the encoder, connected by a rightward arrow from encoder?
- Q3: Is there a tall thin rectangle labeled ‘set of image features' placed to the right of the CNN, with a rightward arrow from CNN to ‘set of image features’?
- Q4: Is there a plus-sign circle located between ‘set of image features' and 'positional encoding' with a solid arrow from 'positional encoding' to the plus circle and a solid line from 'image features' to the same plus circle?
- Q5: is there a small label box immediately to the right of each FFN, aligned horizontally with that FFN’s centerline?

2

[Figure 77]

- 3

- 4

5

- Fig. 8: Example Task: Layout Control. Measurement of spatial arrangement and structural organization within professional design contexts.

- Table 4: Human-Judge Alignment Results. Comparison of overall agreement (po) and Cohen’s Kappa (κ) across different MLLM judges.

|Evaluator Model<br><br>|po κ|
|---|---|
|Gemini-3-Flash OpenAI-GPT-5.2 OpenAI-GPT-5.1<br><br>|90.88% 0.7692 84.15% 0.6221 82.91% 0.5646|

- Table 5: Evaluator Stability Test. Variance of results across three independent evaluation trials for selected models.

Model Trial 1 Trial 2 Trial 3 Std. Dev (σ)

Nano-Banana-Pro 76.7 / 93.7 76.1 / 92.8 76.1 / 92.7 0.28 / 0.45 GPT-Image-1.5 35.9 / 81.6 35.9 / 80.6 35.8 / 80.4 0.05 / 0.68

Prompt: Create an infographic-style image laid out in two main vertical columns... tall left panel slightly below the top, place a large three-line title... reading: “6 REASONS”, “TO GET UP”, “EARLY”... At the upper-right of the entire image, create the first reason section... heading reads “More time for yourself”; directly underneath that heading place one continuous paragraph that reads exactly: “You might be able to get some much-needed (and desperately wanted) time for yourself...”. Below this, in the middle-right panel, add the second reason section... text with a single-line heading that reads “More time to exercise”; directly beneath this heading include one paragraph of body text that reads exactly: “Getting up early could help you find the time to work out...”. To the left of this central right panel... insert the third reason section panel... heading at the top... must read “Less time in traffic”... and the paragraph directly below it must read exactly: “Early mornings can help you beat the usual traffic...”. Under the middle-right panel... create the fourth reason section... text block with a heading that reads on one line “More time to get things done” and with a paragraph immediately below that reads exactly: “Have you ever wished there were more hours in the day?...”. On the lower-left of the entire image... insert the fifth reason panel... text block whose heading reads “Helps you concentrate”... with a paragraph directly below that reads exactly: “We tend to feel groggy and disoriented when

[Figure 78]

- 3

- 4

- Q1: Does the large three-line title in the center-left tall panel read “6 REASONS”, “TO GET UP”, “EARLY” line by line?
- Q2: Does the paragraph directly below “More time to get things done” in the lower-middle-right panel read exactly: “Have…?
- Q3: Does the heading at the top of the upper-right panel, to the left of the stretching person illustration, read exactly: "More time for yourself"?
- Q4: In the same upper-right paragraph under "More time for yourself", does the phrase "much-needed (and desperately wanted)" appear with a hyphen in "much-needed" and parentheses around "and desperately wanted"?
- Q5: Does the paragraph under "Helps you concentrate" in the lower-left panel read exactly: "We tend to feel groggy and disoriented when we first get up because our brain does not wake up immediately. You're more likely to stay focused if you wake up earlier."?

1

2

5

- Fig. 9: Example Task: Text Rendering. Focus on the model’s capacity for legible, accurate, and typographically coherent text generation.

[Figure 79]

Fig. 10: Overall Ranking.

[Figure 80]

- Fig. 11: Performance Ranking: Webpage.

[Figure 81]

##### Fig. 12: Performance Ranking: Slides.

[Figure 82]

##### Fig. 13: Performance Ranking: Chart.

[Figure 83]

- Fig. 14: Performance Ranking: Poster.

[Figure 84]

Fig. 15: Performance Ranking: Scientific Figure.

[Figure 85]

Fig. 16: Capability Ladder: Layout Control.

[Figure 86]

Fig. 17: Capability Ladder: Attribute Binding.

[Figure 87]

Fig. 18: Capability Ladder: Text Rendering.

[Figure 88]

Fig. 19: Capability Ladder: Knowledge-based Reasoning.

[Figure 89]

Question: Is there a two-line caption text

block placed near the bottom-left corner

of the canvas, below the Level 2 scene?

###### GPT-5.2: True.( )

"A two-line caption sits along the bottom left area, below the Level 2 car scene."

Gemini-3-flash: False.( ) "The caption at the bottom is a single line

of text, not two lines."

Fig. 20: Evaluator Case Study A. Qualitative comparison of visual grounding performance between different MLLM judges.

[Figure 90]

Question: Are there exactly 13 vertical bars corresponding to the 13 method categories on the x-axis?

###### GPT-5.2: True.( )

- "There are 13 bars labeled Method A through Method M on the x-axis.”

Gemini-3-Flash: False.( )

- "There are 14 vertical bars visible on the x-axis."

##### Fig. 21: Evaluator Case Study B.

