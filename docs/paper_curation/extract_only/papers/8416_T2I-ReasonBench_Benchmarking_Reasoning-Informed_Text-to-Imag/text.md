## T2I-ReasonBench: Benchmarking Reasoning-Informed Text-to-Image Generation

Kaiyue Sun1 Rongyao Fang2 Chengqi Duan1 Xian Liu2 Xihui Liu1 1The University of Hong Kong 2The Chinese University of Hong Kong

[Figure 1]

Instruction: A rubber duck and a metal ball in a water tank

# arXiv:2508.17472v1[cs.CV]24Aug2025

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Idiom Interpretation Textual Image Design

###### Entity-Reasoning Scientific-Reasoning

Instruction: Create an infographic on STEM's role in environmental sustainability

Instruction: A rubber duck and a metal ball in a water tank

Instruction: The final exam was a piece of cake

Instruction: The city hosting the Summer Olympics in 2021

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

SD-3.5-Medium

GPT-Image-1

Flux.1-Schnell

Gemini-2.0

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

n Evaluation

n Evaluation n Evaluation n Evaluation

[Figure 25]

Idiom Interpretation Image Quality

Intent Reflection Image Quality

Entity Reasoning Image Quality

Scientific Reasoning Image Quality

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Figure 1: Overview of T2I-ReasonBench. We propose T2I-ReasonBench, a benchmark evaluating reasoning capabilities of text-to-image (T2I) models. It consists of four dimensions: Idiom Interpretation, Textual Image Design, Entity-Reasoning and Scientific-Reasoning. We propose a two-stage evaluation protocol to assess the reasoning accuracy and image quality. We benchmark various T2I generation models, and provide comprehensive analysis on their performances.

[Figure 32]

[Figure 33]

Evaluation

### Abstract

Text-to-image (T2I) generative models have achieved remarkable progress, demonstrating exceptional capability in synthesizing high-quality images from textual prompts. While existing research and benchmarks have extensively evaluated the ability of T2I models to follow the literal meaning of prompts, their ability to reason over prompts to uncover implicit meaning and contextual nuances remains underexplored. To bridge this gap, we introduce T2I-ReasonBench, a novel benchmark designed to explore the reasoning border of T2I models. T2I-ReasonBench comprises 800 meticulously designed prompts organized into four dimensions: (1) Idiom Interpretation, (2) Textual Image Design, (3) Entity-Reasoning, and (4) Scientific-Reasoning. These dimensions challenge models to infer latent meaning, integrate domain knowledge, and resolve contextual ambiguities. To quantify the performance, we introduce a two-stage evaluation framework: a large language model (LLM) generates prompt-specific question-criterion pairs that evaluate if the image includes the essential elements resulting from correct reasoning; a multimodal LLM (MLLM) then scores the generated image against these criteria. Experiments across 14 state-of-the-art T2I models reveal that open-source models exhibit critical limitations in reasoning-informed generation, while proprietary models like GPT-Image-1 [34] demonstrate stronger reasoning and knowledge in-

Preprint. Under review.

tegration. Our findings underscore the necessity to improve reasoning capabilities in next-generation T2I systems. This work provides a foundational benchmark and evaluation protocol to guide future research towards reasoning-informed T2I synthesis. Code is available at https://github.com/KaiyueSun98/T2I-ReasonBench.

### 1 Introduction

Recent advancements in T2I generative models have enabled the creation of visually appealing images from textual prompts. However, these models often struggle with generating complex scenes that demand multistep reasoning. A key limitation lies in their design: most T2I systems process the prompts by mapping semantic concepts directly to visual elements without integrating reasoning mechanisms. This approach restricts their ability to infer implicit relationships or leverage the knowledge learned, thereby hindering progress towards achieving a real “world model”.

Current benchmarks [48, 21, 23, 14, 20, 26, 45, 22, 44], such as T2I-CompBench [22] and PartiPrompts [48], primarily evaluate literal prompt-image alignment, focusing on object attributes (e.g., color, attribute, count, spatial relationships) or world-knowledge coverage. While DPG-Bench [20] extends evaluation to long-text comprehension, it remains confined to multi-object composition tasks. These frameworks fail to test models’ ability to reason beyond explicit instructions. For instance, generating an image of “A beach ball and a marble in a swimming pool” requires not only object composition but also reasoning about physical laws (e.g., inferring the ball floats while the marble sinks). Such reasoning necessitates understanding the related scientific knowledge, such as material density and buoyancy, as well as integrating the reasoning process into T2I generation.

To address this gap, we propose T2I-ReasonBench, a novel benchmark designed to systematically evaluate the reasoning ability of T2I models in four dimensions: (1) Idiom Interpretation: Decipher the implicit meanings of idiomatic expressions with the context to generate appropriate images. (2) Textual Image Design: Understand the intention of design and effectively plan integrated visualtextual layouts. (3) Entity-Reasoning: apply and integrate the knowledge about world entities in image generation, and (4) Scientific-Reasoning: reason with scientific knowledge (e.g., physics, chemistry, biology, astronomy) to produce images adhering to the underlying scientific laws. T2IResonBench encompasses four dimensions with 800 meticulously designed prompts, all requiring deep reasoning. It aims to explore the reasoning border of current T2I models and to identify how effectively they can interpret implicit instructions. Ultimately, this work seeks to push the boundaries T2I generation and inspire future research in this direction.

To rigorously evaluate performances of T2I models, we introduce a two-stage evaluation framework. First, an LLM generates specific question-criterion pairs for each prompt. To evaluate the images, an MLLM then answers each question and assigns a score based on the paired criterion. By averaging these scores, we measure how faithfully the image reflects the implicit meaning of the prompt, effectively capturing the accuracy of model’s reasoning. The same two-stage process is also applied to evaluate the quality of image. These result in two scores for each model: Accuracy and Quality. Our approach allows for fine-grained and interpretable evaluation of models’ reasoning ability and addresses the limitation of previous benchmarks that focused solely on literal prompt following.

We evaluate 14 state-of-the-art T2I models, including 7 diffusion-based models, 5 auto-regressivebased models, and 2 proprietary models. The results reveal that open-source models significantly underperform proprietary counterparts, though all models have room for improvement, and unified models that effectively integrate understanding and generation demonstrate enhanced reasoninginformed T2I generation.

Our contributions are threefold:

- • We propose T2I-ReasonBench, a novel benchmark with meticulously designed tasks to explore the reasoning border of current T2I models.
- • Our prompt-specific evaluation framework enables fine-grained and interpretable evaluation of T2I tasks that require reasoning.
- • We evaluate 14 state-of-the-art T2I models, and provide a thorough analysis of their performances, highlighting notable limitations in reasoning ability of these models.

### 2 Related Work

#### 2.1 Text-to-image Generation.

Diffusion-based models. T2I generation has seen rapid advances in recent years. Diffusion-based models have emerged as the dominant approach, surpassing conventional GANs in scalability, training stability, and output quality [8]. These models formulate image generation as a progressive denoising process, gradually refining Gaussian noise into coherent visual outputs conditioned on textual prompts [19]. Early breakthroughs like GLIDE [32] and Imagen [37] set the stage, while later systems such as Stable Diffusion [36] pushed the boundaries of resolution, realism, and controllability. Recent models like FLUX [25] and HiDream [18] further extend this paradigm, achieving fine-grained, photorealistic synthesis and state-of-the-art zero-shot text-to-image performance, solidifying diffusion as the backbone of modern T2I systems.

Auto-regressive-based models. Auto-regressive-based (AR-based) models generate data by predicting the next token in a sequence. The introduction of VQ-VAE [41] enabled a two-stage approach: compressing images into discrete tokens followed by autoregressive modeling of these token distributions. Building upon this, models like DALL·E [35] and CogView [9] extended AR methods to T2I generation by training decoder-only transformers on concatenated text and image tokens.

Recent advancements have further enhanced the capabilities of AR models. Chameleon [39] introduces a unified transformer architecture that processes interleaved text and image sequences without separate modality-specific encoders, enabling mixed-modal generation. Lumina-mGPT [29] builds upon this framework by employing multimodal generative pretraining and flexible supervised fine-tuning, achieving high-quality photorealistic image synthesis and supporting a range of vision-language tasks. Models like GoT [12] and GoT-R1 [10] employ a unified auto-regressive MLLM in their architectures for explicit semantic-spatial reasoning, leading to improved performance on compositional tasks. Bagel [7] unifies an auto-regressive LLM and a diffusion model within a single transformer architecture. Trained on data with explicit reasoning chains, Bagel can generate detailed reasoning steps before producing images when its “Thinking” mode is enabled, effectively transferring its understanding capabilities into the generation process.

While diffusion models excel in iterative refinement processes for image generation, AR models offer a unified architecture that seamlessly integrates text and image modeling. This inherent compatibility makes AR models particularly well-suited for multimodal tasks, highlighting the complementary strengths of both paradigms in the evolving landscape of generative modeling.

#### 2.2 Benchmarks for Text-to-image Generation.

Evaluating the capabilities of T2I models requires diverse benchmarks that assess various aspects of understanding and generation. Most of current benchmarks [48, 21, 23, 14, 20, 26, 45, 22, 44] primarily evaluate literal prompt-image alignment. For example, GenEval [14] utilizes object detection techniques to test whether generated images correctly capture object co-occurrence, position, count, and color described in the prompts. Recently, more benchmarks have shifted focus from simple literal alignment to evaluating reasoning abilities of T2I generation. WISE [33] introduces 1000 prompts that require world knowledge and reasoning across cultural common sense, spatial-temporal understanding, and natural science to assess how well images align with real-world knowledge. PhyBench [31] addresses physical common sense reasoning using 700 prompts covering mechanics, optics, thermodynamics, and material properties, challenging models to generate images consistent with physical laws. Commonsense-T2I [13] tests models’ grasp of everyday commonsense through adversarial prompt pairs. More comprehensively, R2I-Bench [4] designs more than 3000 prompts to evaluate reasoning-driven T2I generation across seven categories, including commonsense, mathematical, logical, compositional, numerical, causal, and concept mixing. These benchmarks collectively provide a multifaceted evaluation landscape, pushing T2I generation towards a deeper and more sophisticated understanding of the world.

T2I-ReasonBench aims to explore the reasoning borders of T2I generation through 800 carefully designed prompts across four dimensions. While previous benchmarks require reasoning, their content to generate is well-defined, such as “Einstein’s favorite musical instrument” or “A bookshelf with some books, no books on the second shelf”. In our benchmark, two critical dimensions, i.e., Idiom Interpretation and Textual Image Design, are introduced. They assess not only idiom comprehension

表格 1-1

表格 1

| | | | | | | |
|---|---|---|---|---|---|---|
| |800|Idiom|200|A|100|23|
| | | | |B|50|18|
| | | | |C|50|17|
| | |Rich text|200|d|100|16|
| | | | |e|50|15|
| | | | |f|50|15|
| | |World|200|g|100|14|
| | | | |h|50|14|
| | | | |i|50|13|
| | |Physics|200|j|100|12|
| | | | |k|100|11|
| | | | | | |22|
| | | | | | |80|
| | | | | | |40|
| | | | | | |40|
| | | | | | |20|
| | | | | | |10|
| | | | | | |10|
| | | | | | |100|
| | | | | | |100|
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |800|Aa|200|A|100| |
| | | | |B|50| |
| | | | |C|50| |
| | |Bb|200|d|100| |
| | | | |e|50| |
| | | | |f|50| |
| | |Cc|200|g|100| |
| | | | |h|50| |
| | | | |i|50| |
| | |Dd|200|j|100| |
| | | | |k|100| |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

13%

13%

| | |
|---|---|
| | |

13%

13%

6%

6%

13%

13%

6%

6%

6%

6%

13%

13%

6%

6%

6%

6%

13%

6%

13%

6%

表格 1-1-1

表格 1-2

Work & Career 800 Idiom 200 32 32 A Emotions 27 27 B Social Interaction

800 Aa 200 A 100

25 25 C

- B 50

- C 50

- d Rich text 18 18 d

- e 16 16 e

- f 15 15 f

- g World 14 14 g

- h 13 13 h

- i 40 40 ———————

- j Physics 80 80 j

- k 200 40 40 k

Bb 200 d 100

- e 50

- f 50

Cc 200 g 100

- h 50

- i 50

Dd 100 j 100 50 k 100

- A 40 40 A

- B 20 20 B

- C 20 20 ——————-

- d 49

- e 27 49 e

- f 200 26 27 f

- g 23 26 g

- h 21 23 h

- i 20 21 i

- j 18 20 j

- k 16 18 ———————

- D 21 16 D D 17 21 D D 200 17 19 D

30 20

13%

13%

6%

13%

6%

6%

14 17 17 12 17 14 19 14 12

13%

6%

12 17 8 24 11

6%

13%

13%

13%

6%

10 8 8

6%

13%

15 15

13%

13%

6%

20

6%

6%

13%

13%

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Health&Lifestyle

6%

|[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>Reaction<br><br>Combustion<br><br>Corrosion<br><br>Indicator<br><br>Animal<br><br>Plant<br><br>Astro.|&<br><br>Career<br><br>Interaction Lifestyle|
|---|---|
|[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Electromag.<br><br>Biology<br><br>Astro.|Work<br><br>Emotion<br><br>&|
|[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>Mechanics Optics<br><br>Chem.|Social Health|

6%

###### Task Proposal

[Figure 53]

A Emotions Social Interaction d e f g h i j k A B C d e f g h i j k D D D

###### Prompt

[Figure 54]

[Figure 55]

|[Figure 56]<br><br>[Figure 57]<br><br>Idiom Interpretation Textual Image Design<br><br>[Figure 58]<br><br>[Figure 59]<br><br>Entity-Reasoning Scientific-Reasoning<br><br>[Figure 60]<br><br>Search<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>Prompt Creation<br><br>[Figure 67]<br><br>LLaVAR-2 InfographicVQA POSTA CoSyn-400k<br><br>[Figure 68]<br><br>[Figure 69]<br><br>Design Intention Generation<br><br>[Figure 70]<br><br>Rich-Text Image Datasets<br><br>[Figure 71]<br><br>Expert designed prompts<br><br>inspire<br><br>[Figure 72]<br><br>Prompt Knowledge verification Creation<br><br>|
|---|

[Figure 73]

6%

Astronomy

Plant & Fungus

Work & Career

Min. length Max. length Avg. length

6%

13%

Animal

[Figure 74]

Corrosion

Indicator

Social Interaction

Combustion

6%

D 3%

2 33 12.5words

Emotions

D 3%

Reaction

Finance Knowledge

[Figure 75]

Health&Lifestyle

A

[Figure 76]

[Figure 77]

Emotions 4%

D

[Figure 78]

Thermodyn.

Rich-Text Image Datasets LLaVAR-2: text-rich images InfographicVQA: infographics POSTA: Expert-designed posters CoSyn-400k: code-generated graphs

13%

k 3% j 3%

Astronomy

[Figure 79]

[Figure 80]

Material

[Figure 81]

Social Interaction 4%

Biology

[Figure 82]

Time Conﬂict

[Figure 83]

Scientiﬁc Reasoning

Idiom Interpretation

Finance

Electromagnetism

Chemistry

Physics

Material

Knowledge

d 3%

Mechanics

6%

i 3% h3%

Others

Fluid

Optics

Time Conﬂict

c Reasoning

Idiom Interpretation

Scientiﬁ

Nature

- 2%

e

- 3%

f

Physics

Thermodynamics

Plant

Real-world textual

Entity Reasoning

Textual Image Design

6%

g 2%

4% g

Animal

[Figure 84]

Others

Fluid

13%

Architecture

images

h 2%

6%

Design

Event

Textual Image

[Figure 85]

4% f 4%

Entity Reasoning

Infographic

Food

| | |
|---|---|
| | |

Real-world textualimages

Plant Nature

Artifact

Poster Document

i 6%

Animal

Celebrity

Table &

Architecture

Infographic

e 8%

Diagram

Food Event

Poster

Artifact

Document

Table & Diagram

Celebrity

j 13%

C 3%B 3% A 6%

[Figure 86]

[Figure 87]

k

Figure 2: Left: Prompt collection process. Middle: Subcategories in the four evaluation dimensions. Right: Prompt Suite Statistics.

and text synthesis but also the model’s ability to envision and construct complex scenarios, as well as to infer and fill in missing information. Most prompts in Idiom Interpretation describe daily scenes without strictly defined content, where the goal is for the generated image to accurately express both the scenario and meaning of the idiom. In Textual Image Design, some prompts do not specify all the text or visual elements needed in the image. Here, the model must creatively design and include these elements to reflect the prompt intention. These types of reasoning abilities, i.e., scenario imagination and information completion, have not been addressed in previous works.

[Figure 88]

### 3 Benchmark Construction

#### 3.1 Problem Definition

While modern T2I models are good at literal prompt-to-image translation, their capacity for knowledge-driven reasoning-informed generation, remains underexplored. Existing benchmarks focus predominantly on surface-level alignment (e.g., object existence, spatial arrangements), but fail to evaluate whether models possess reasoning abilities to uncover the deeper meaning behind the text and generate logically coherent visual content. To this end, we identify four critical scenarios that challenge T2I models to reason about the instructions before visualizing them:

- Scenario 1: An idiom is a phrase or combination of words with a figurative meaning that differs from its literal meaning. Idioms are common in everyday language, and their meanings usually cannot be deduced by analyzing individual words. For T2I models, prompts containing idioms demand reasoning to obtain the latent meaning before generating semantically faithful visual content. This process requires leveraging linguistic knowledge and effectively analyzing context.
- Scenario 2: Images with rich text combine visuals and text in various formats. These images are used to serve specific communicative goals, such as education, marketing, and promotion. Generating such content requires T2I models reasoning about the purpose behind the image and apply goal-oriented design skills like layout planning, information structuring, and harmonizing visuals and text.
- Scenario 3: In everyday life, people often forget specific entity names but remember related details. For example, the prompt “Generate an image of the team lifting the trophy at the 2022 FIFA World Cup” requires the T2I model to reason about the context and then retrieve relevant knowledge to generate the entities not explicitly stated.
- Scenario 4: Creating physically realistic images remains a significant challenge for current T2I models, which often produce counterintuitive results that violate common sense. This highlights the need to test whether models can apply scientific knowledge. For example, with the prompt “Iron filings scattered around a bar magnet”, the model needs to understand magnetism and show the iron filings curving between the magnet’s poles.

Based on these four scenarios, we define four dimensions to evaluate the reasoning abilities of T2I models: Idiom Interpretation, Textual Image Design, Entity-Reasoning and Scientific-Reasoning.

- 1. Fluid Mechanics & Properties: buoyancy (6), fluid dynamics (2), pressure (8), capillary action (2), density (3) → total 21
- 2. Thermodynamics & Phase Transitions: thermodynamics (2), combustion (5), phase transition (8), sublimation (2) → 17
- 3. Classical Mechanics: inertia (2), leverage (5), mechanism (3), gravity (4) → 14
- 4. Material Science: elasticity (6), material hardness (5) → 11
- 5. Electromagnetism: electricity (12)
- 6. Optics: 17
- 7. Transport Phenomena: diffusion (2) →6 + others= 8

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Idiom Interpretation Textual Image Design Entity Reasoning Scientific Reasoning

Idiom Interpretation Textual Image Design Entity Reasoning Scientific Reasoning

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

Figure 3: Word cloud to visualize the word distribution of each dimension in our prompt suite.

[Figure 102]

#### 3.2 Prompt Suite of T2I-ReasonBench

Idiom Interpretation. By sourcing from a book [1] and the internet, we collect 200 idioms that are commonly used in daily life but may be challenging for T2I models. We then use an LLM, Claude-3.5-Sonnet [6], to generate sentences containing the idioms but without explicitly revealing their meanings. These idioms span diverse topics such as social interactions, lifestyle, and emotions. For example, the sentence “He told a funny joke to break the ice at the start of the meeting” uses the idiom “break the ice”, which means to ease tension during a first meeting, rather than literally destroying the ice.

Textual Image Design. In this category, we first collected 200 images featuring rich text from different datasets. Using an MLLM, Qwen2.5-VL [2], we then extracted the underlying design intentions from these images, resulting in 200 design prompts. Each prompt focuses on the functional purpose of the image rather than describing visual details. For example, the prompt “Create a minimalist promotional poster for a workshop on simplicity in design” is an abstract, high-level design instruction. Based on the image sources, the prompts span categories like infographics, posters, documents, tables, diagrams, and other real-world images such as book covers and tickets.

Entity-Reasoning. In Entity-Reasoning, we begin by defining subdomains for various entities, such as celebrities, artifacts, and natural landscapes. We manually create several example prompts along with their explicit meanings to guide an LLM (DeepSeek-R1 [16]) in generating more pairs of prompt and its explicit meaning. After collecting 200 such pairs, we carefully review them to ensure overall consistency and confirm that each entity possesses unique visual features. For instance, the prompt “The first mammal successfully cloned from an adult somatic cell in 1996” refers to Dolly the sheep.

Scientific-Reasoning. The prompts in the Scientific-Reasoning are constructed in a similar manner. We first identify four key scientific disciplines: physics, chemistry, biology, and astronomy, then create several example pairs of prompt and corresponding explicit meaning. We use these examples to inspire the LLM to generate additional pairs of prompt and explicit meaning. Each prompt is manually validated to ensure it requires reasoning about scientific knowledge and the expected visual outcome is not explicitly stated. For instance, the prompt “A trampoline with an iron ball on it” implies that the heavy iron ball would deeply stretch the surface of the trampoline due to its weight.

Figure 2 demonstrates the prompt collection process (left), shows the subcategories in each dimension (middle) and provides the prompt suite statistics (right). We visualize the word distribution of each dimension in our prompt suite in Figure 3. For more information about the prompt suite, please refer to the appendix.

### 4 Evaluation Metrics

In recent years, MLLMs have demonstrated remarkable capabilities in understanding complex visual content, becoming the primary tool for evaluating images and videos. However, the prompts in our benchmark are highly complex, often involving multiple objects, intricate relationships, and challenging scenarios. As a result, using generic evaluation instructions that are identical for all prompts proved ineffective. This is because each image, generated from a unique prompt, has specific features that require targeted checks. Generic instructions cannot cover every detail, and MLLMs struggle to address all aspects when given long, broad guidelines. To address this, we develop a two-stage evaluation framework with customized evaluation instructions for each prompt. These instructions take into account the prompt category, the reasoning needed, and both the explicit content and implicit meaning the image should exhibit. Figure 4 illustrates the evaluation process.

Prompt-specific question-criterion pairs generation. In the first stage, we use the LLM DeepSeekR1 [16] to generate question-criterion pairs based on the given prompt and dimension-specific information (e.g., idiom meaning for Idiom Interpretation or explicit meaning for Entity and ScientificReasoning). For each dimension, two sets of questions are provided to separately examine the T2I models’ required reasoning and the image quality. For Entity and Scientific-Reasoning, where prompts may involve explicit details that do not need reasoning, an additional set of questions is provided to examine these details.

Image analysis and evaluation. In the second stage, we employ the MLLM Qwen2.5-VL [2] to evaluate the generated images with a Chain-of-Thought [43] (CoT) mechanism: the model first describes the image, then answers the specific questions from Stage 1. For each question, Qwen2.5VL provides an analysis before assigning a score, ensuring thorough and reasoned evaluation. Scores within each set are averaged to produce two main results: Reasoning Accuracy and Image Quality.

nr i=1 scorei

, (1)

Sreason =

nr

nd i=1 scorei

, (2)

Sdetail =

nd

nq i=1 scorei

, (3)

Squality =

nq

where nr, nd, and nq represent the number of questions in reasoning evaluation, other details evaluation and image quality evaluation.

Sreason, for Idiom Interp. and Textual Image Design w1Sreason + w2Sdetail, for Entity and Scientific-Reasoning

Reasoning Accuracy =

(4) Image Quality = Squality, (5)

Here, we set the weights [w1,w2] to [0.7,0.3] to prioritize reasoning and produce a balanced final score. In this way, our evaluation metrics reflect the reasoning challenges and provide a comprehensive assessment. For more details of our evaluation framework, please refer to the appendix.

#### 4.1 Human Evaluation Correlation Analysis

To validate the effectiveness of our evaluation metrics, we perform human evaluations and measure the correlation between these metrics and human scores for each dimension. We randomly select 20 prompts from each dimension and use five different T2I models to generate 100 images per dimension. This results in 400 images in total for evaluation. The evaluation is conducted with a group of college postgraduate participants, and the criterion is specific for each dimension. Three annotators independently score each image, and we average their scores for each prompt-image pair. We then calculate the correlation between the averaged human scores and the automatic metric scores using Kendall’s τ and Spearman’s ρ. Our metrics are compared against several widely-used T2I evaluation metrics, including CLIP [17] and VQA score [28]. The correlation results, shown in Table 1, demonstrate that our proposed metrics (Reasoning Accuracy) achieve the highest correlation with human judgments across all dimensions (highlighted in bold). While CLIP and VQA scores also show moderate correlations due to capturing surface-level alignment, they perform worse than our metrics because they cannot interpret the implicit meaning of the prompts as effectively.

5 Evaluation Results

#### 5.1 Experimental Setup

Evaluated models. We evaluate 14 state-or-the-art T2I models, including 7 diffusion-based T2I models, 5 AR-based models, and 2 proprietary models. The diffusion-based T2I models are HiDream-I1-full [18], FLUX.1-dev[25], FLUX.1-schnell [25], Playground-v2.5[27], StableDiffusion-3-Medium [11], Stable-Diffusion-3.5-Medium [11], and Stable-Diffusion-3.5-Large [11]. The AR-based models are: Bagel [7], Emu3 [42], Janus-Pro-7B [5], show-o-demo-512 [46], and GoT [12]. The proprietary models are Gemini-2.0 [40] and GPT-Image-1 [34].

Implementation Details. We adopt the default setting for all the models during inference.

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

###### Idiom Interpretation

Textual Image Design

Entity Reasoning

Scientific Reasoning

[Figure 111]

Idiom Interpretation

|Prompt: I volunteered for too many projects and quickly realized I'd bitten off more than I could chew. Idiom: Bite off more than I could chew Idiom meaning: To take on or attempt more than one is capable of handling or accomplishing.|
|---|

|Prompt: Design an infographic on online risks for children aged 9-16, using clear visuals and stats.|
|---|

|Prompt: A garlic has been put in a glass of water for a month. Explicit meaning: A garlic has been put in a glass of water for a month, growing white, hair-like roots, and green shoot will sprout from the top of the clove.|
|---|

|Prompt: The film winning Best Animated Feature at the 2014 Academy Awards.<br><br>Explicit meaning: Frozen winning Best Animated Feature at the 2014 Academy Awards.|
|---|

[Figure 112]

Task Proposal

|[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>Prompt Creation<br><br>[Figure 117]|
|---|

[Figure 118]

Evaluation Question-Criterion Pairs Generation

|Reasoning Evaluation<br><br>Q1: Does the image depict root growth (thin, white structures) specifically from the base of the garlic clove? C1: 1.0: Clear root growth from garlic base. 0.5: Roots present but ambiguous origin. 0.0: No roots or incorrect origin.<br>Q2: Are the roots shown submerged in water?...<br>Q3: Is a green shoot visibly emerging from the top of the garlic?...<br><br><br>[Figure 119]<br><br>|
|---|

|Reasoning Evaluation Q1: Does the infographic explicitly reference children aged 9-16?...<br><br>[Figure 120]<br><br>Q2: Does it depict at least 3 distinct online risks categories (e.g., cyberbullying, grooming, data privacy)? C2: 1.0: for ≥3 visually distinct risk categories. 0.5: for 2 clear categories. 0.0: for <2.<br>Q3: Are statistics visibly integrated through data visualization elements?...<br><br><br>|
|---|

|Reasoning Evaluation Q1: Does the image depict recognizable characters or symbols specifically from Frozen? C1: 1.0: Clear Frozen elements (Elsa/Anna/Olaf). 0.5: Generic snow/winter elements. 0.0: No Frozen elements.<br><br>[Figure 121]|
|---|

|Reasoning Evaluation Q1: Is there a clear central figure as the focal subject?...<br><br>[Figure 122]<br><br>Q2: Are multiple distinct tasks/projects visually represented (e.g., ≥3 unique items like stacks of files, tools, digital screens)? C2: 1.0: ≥3 clearly distinct task representations. 0.5: 1–2 items OR unclear representations. 0.0: No task representations.<br>Q3: Is the figure visibly overwhelmed or struggling (e.g., bent posture, sweating, distressed expression, chaotic surroundings)?...<br><br><br>|
|---|

|Other details Evaluation<br><br>Q1: Is an Oscar statuette clearly visible?...<br><br>Q2: Does the setting suggest Academy Awards ceremony?...<br>Q3: Is there indication of 2014?...<br>|
|---|

|Other details Evaluation<br><br>Q1: Is the container clearly identifiable as a transparent glass?...<br><br>Q2: Is water visibly present in the container?...<br>|
|---|

|Image Quality Evaluation<br><br>Q1: Is the design visually appealing to children/teens? C1: 1.0: for age-appropriate design (bright colors + relatable illustrations). 0.5: if overly childish/adult. 0.0 if inappropriate.<br>Q2: Is there harmful/frightening imagery?...<br>|
|---|

|Image Quality Evaluation<br><br>Q1: Are Frozen characters and trophies clearly visible without obstruction?...<br>Q2: Is lighting/color quality suitable for formal event?...<br>|
|---|

|Image Quality Evaluation<br><br>Q1: Is the image visually coherent without significant artifacts?...<br>Q2: Does the composition prioritize the figure and tasks through framing/contrast?...<br>|
|---|

|Image Quality Evaluation<br><br>Q1: Are key elements (roots, shoot, garlic) clearly visible without obstruction?...|
|---|

[Figure 123]

[Figure 124]

Analysis and Evaluation

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

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

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

SD3.5-

GPT-

[Figure 157]

Bagel

HiDream-I1

[Figure 158]

[Figure 159]

[Figure 160]

Image-1

Large

|Analysis Example: Reasoning Q1: The image does not show any root growth from the base of the garlic cloves. Therefore, the score is 0. Final Score: Reasoning Accuracy<br><br>= W1 x Avg (Reasoning Questions) + W2 x Avg (Other Detail Questions)<br><br>= 0.7 x 0 + 0.3 x 1 = 0.3 Image Quality<br><br>= Avg (Image Quality Questions) = 1.0|
|---|

|Analysis Example: Reasoning Q2: The infographic depicts four distinct online risks: seeing sexual content, cyberbullying, seeing hateful content, and online contact with a stranger. So it scores 1. Final Score: Reasoning Accuracy<br><br>= Avg (Reasoning Questions) = 1.0 Image Quality<br><br>= Avg (Image Quality Questions) = 1.0|
|---|

|Analysis Example: Image Quality Q2: the image has noticeable artifacts, particularly in the hands and fingers of the person, which appear distorted and unnatural. Therefore, it scores 0.5. Final Score: Reasoning Accuracy<br><br>= Avg (Reasoning Questions) = 1.0 Image Quality<br><br>= Avg (Image Quality Questions) = 0.5|
|---|

|Analysis Example: Reasoning Q1: The image does not depict any recognizable characters from Frozen. Therefore, it scores 0. Final Score: Reasoning Accuracy<br><br>= W1 x Avg (Reasoning Questions) + W2 x Avg (Other Detail Questions)<br><br>= 0.7 x 0 + 0.3 x 1 = 0.3 Image Quality<br><br>= Avg (Image Quality Questions) = 0.33|
|---|

Figure 4: Evaluation Framework of T2I-ReasonBench. We adopt a two-stage evaluation framework: prompt-specific evaluation question-criterion pairs generation by an LLM, then image analysis and scoring by an MLLM. This figure shows one evaluation example for each dimension.

#### 5.2 Quantitative Evaluation

Table 2 presents the quantitative evaluation results of T2I-ReasonBench. The results reveal that generating images correctly reflecting the underlying meaning which is not explicitly expressed in the prompt remains a significant challenge for current T2I models, especially for the open-source ones. This suggests a critical gap in text-to-image generation, emphasizing the limitations of current approaches in handling complex prompt reasoning that involves diverse types of knowledge.

Drawing from the scores, a clear distinction emerges between open-source models and proprietary models, as well as between diffusion-based models and AR-based models among the open-source group. In particular, the recently released T2I model, HiDream [18], achieves an overall accuracy score exceeding 50.0. Using Llama 3 [15] as one of its text encoders, Hidream demonstrates the potential of integrating powerful LLMs in T2I generation and utilizing their extensive pretraining knowledge. However, despite this progress, HiDream still works in the way of concept mapping, shows no evidence of prompt reasoning, which hinders it achieving truly high accuracy scores. This highlights the need for more advanced approaches.

- Table 1: The correlation between automatic evaluation metrics and human evaluation. Our proposed metrics show enhanced performance in Kendall’s τ and Spearman’s ρ.

Model Idiom Textual Entity Scientific

τ(↑) ρ(↑) τ(↑) ρ(↑) τ(↑) ρ(↑) τ(↑) ρ(↑)

CLIP 0.3186 0.4348 0.5372 0.7187 0.2732 0.3837 0.1905 0.2657 VQA score 0.4091 0.5672 0.4890 0.6590 0.4483 0.6133 0.3698 0.4939 Reasoning Acc. (ours) 0.5246 0.6732 0.5829 0.7426 0.4732 0.6148 0.4490 0.5777 Image Qual. (ours) 0.1544 0.1883 0.3457 0.4611 0.3141 0.3947 0.2080 0.2652

- Table 2: Evaluation results of T2I-ReasonBench. ‘Acc.’ represents the Reasoning Accuracy score. ‘Qual.’ represent the Image Quality score. Scores are normalized between 0 and 1. A higher score indicates better performance. Blue highlights the top score among diffusion-based models. Yellow highlights the top score among AR-based models. Bold signifies the highest score across all models.

Idiom Textual Entity Scientific Overall Acc. Qual. Acc. Qual. Acc. Qual. Acc. Qual. Acc. Qual.

Model

Diffusion-based T2I HiDream-I1-full 48.5 87.2 72.3 85.5 54.1 94.1 53.2 84.5 57.0 87.8

FLUX.1-dev 39.1 83.4 56.9 76.5 45.1 90.6 46.7 80.9 47.0 82.8 FLUX.1-schnell 40.9 83.1 65.1 74.5 44.8 91.5 50.7 83.0 50.4 83.0 Playground-v2.5 43.9 87.8 38.5 72.1 48.4 92.4 50.8 83.3 45.4 83.9

SD-3-Medium 35.9 81.4 60.9 71.3 42.4 90.1 50.9 81.7 47.5 81.1 SD-3.5-Medium 34.4 80.6 58.0 70.1 44.8 92.1 49.9 83.0 46.8 81.4

SD-3.5-Large 35.6 85.3 62.2 75.4 46.6 92.6 52.9 84.5 49.3 84.4

Auto-regressive-based T2I Bagel (Thinking) 44.6 84.3 44.0 73.7 52.4 91.6 57.7 88.3 49.7 84.5

Emu3 33.1 82.9 33.7 68.7 33.8 85.2 40.1 77.0 35.2 78.5 Janus-Pro-7B 25.5 78.0 37.2 70.9 38.5 87.6 44.9 77.8 36.5 78.6

show-o-demo-512 33.1 82.5 35.3 80.3 34.9 87.4 41.6 76.6 36.2 81.7 GoT 29.7 76.4 30.6 70.7 31.0 86.2 36.8 76.3 32.0 77.4

###### Proprietary T2I

Gemini-2.0 52.4 87.8 73.0 83.3 67.0 94.3 66.7 89.3 64.8 88.7 GPT-Image-1 75.7 94.5 86.9 97.6 77.5 96.6 74.7 94.3 78.7 95.8

AR-based models adopt a unified architecture for understanding and generation. Although these models excel in understanding and reasoning text, most of them do not easily transfer this strength to T2I generation, thus underperforming diffusion-based models in our benchmark. However, Bagel [7] demonstrates comparable performance to diffusion-based models when its “Thinking” mode is enabled. Trained with reasoning-augmented data, Bagel first reasons about the prompt’s actual content then generating the image. Moreover, its bottleneck-free architecture unifies LLM and diffusion models within a single transformer, allowing better interaction between generation and understanding modules.

In the proprietary models, reasoning capabilities are clearly demonstrated by Gemini-2.0 [40]. It first reasons about the input prompt and then plans the content to generate. For example, given the prompt “The city hosting the Summer Olympics in 2021”, Gemini-2.0 outputs text evidence of reasoning: “I will generate an image depicting a vibrant cityscape, clearly identifiable as Tokyo through prominent landmarks such as Tokyo Tower and the Skytree, bustling with celebratory Olympic banners and flags...”. This reasoning process highlights the model’s ability to bridge textual understanding with visual synthesis. Finally, GPT-Image-1 [34], provided by OpenAI, exhibits even better performances. Although its technical details have not yet been publicly released, the community anticipates that GPT-Image-1 employs a hybrid auto-regressive architecture combined with a diffusion-based head. This hybrid approach likely contributes to advanced knowledge retrieval and reasoning, and the

#### Table 3: Evaluation results with LLM-rewritten prompts. Blue highlights the top score among

diffusion-based models. Yellow highlights the top score among AR-based models. Bold signifies the highest score across all models. ↑ Red and ↓ Blue subscripts denote the amount of increase or decrease compared to results from the original prompt.

Idiom Textual Entity Scientific Overall Acc. Qual. Acc. Qual. Acc. Qual. Acc. Qual. Acc. Qual.

Model

###### Diffusion-based T2I

HiDream-I1-full 64.4↑15.9 91.9 ↑4.7 77.5 ↑5.2 87.0 ↑1.5 76.9↑22.8 96.8 ↑2.7 67.3↑14.1 89.9↑5.4 71.5↑14.5 91.4 ↑3.6

FLUX.1-dev 66.2↑27.1 90.5↑7.1 69.8↑12.9 80.5↑4.0 72.0↑26.9 96.1↑5.5 69.1 ↑22.4 92.3 ↑11.4 69.3↑22.3 89.9↑7.1 FLUX.1-schnell 68.2 ↑27.3 87.4↑4.3 71.6↑6.5 78.7↑4.2 72.6↑27.8 94.9↑3.4 66.1↑15.4 90.1↑7.1 69.6↑19.2 87.8↑4.8 Playground-v2.5 55.8↑11.9 88.7↑0.9 40.5↑2.0 76.5↑4.4 70.7↑22.3 94.7↑2.3 56.1↑5.3 87.0↑3.7 55.8↑10.4 86.7↑2.8

SD-3-Medium 65.7↑29.8 87.6↑6.2 70.9↑10.0 83.1↑11.8 73.3↑30.9 96.1↑6.0 65.5↑14.6 91.7↑10.0 68.9↑21.4 89.6↑8.5 SD-3.5-Medium 66.8↑32.4 88.5↑7.9 69.2↑11.2 79.9↑9.8 72.2↑27.4 95.9↑3.8 66.6↑16.7 89.9↑6.9 68.7↑21.9 88.6↑7.2

SD-3.5-Large 67.7↑32.1 90.4↑5.1 72.4↑10.2 84.4↑9.0 77.7 ↑31.1 95.6↑3.0 68.6↑15.7 92.3 ↑7.8 71.6 ↑22.3 90.7↑6.3

Auto-regressive-based T2I Bagel (w/o Thinking) 67.7 ↑23.1 87.8↑3.5 61.5 ↑17.5 79.7↑6.0 69.9 ↑17.5 94.7 ↑3.1 67.5 ↑9.8 90.3 ↑2.0 66.7 ↑31.5 88.1↑9.6

Emu3 56.0↑22.9 84.2↑1.3 41.5↑7.8 74.7↑6.0 62.9↑29.1 90.6↑5.4 48.5↑8.4 84.1↑7.1 52.2↑15.7 83.4↑4.8 Janus-Pro-7B 63.1↑37.6 82.9↑4.9 54.9↑17.7 80.5↑9.6 69.4↑30.9 93.0↑5.4 61.1↑16.2 87.5↑9.7 62.1↑25.6 86.0↑7.4

show-o-demo-512 64.2↑31.1 89.5 ↑7.0 42.9↑7.6 83.5 ↑3.2 66.5↑31.6 94.0↑6.6 59.9↑18.3 90.3 ↑13.7 58.4↑22.2 89.3 ↑7.6 GoT 51.8↑22.1 81.4↑5.0 36.4↑5.8 73.1↑2.4 51.5↑20.5 89.2↑3.0 43.6↑6.8 81.8↑5.5 45.8↑13.8 81.4↑4.0

###### Proprietary T2I

Gemini-2.0 67.1↑14.7 91.5↑3.7 78.4↑5.4 89.4↑6.1 77.9↑10.9 96.1↑1.8 72.1↑5.4 90.6↑1.3 73.9↑9.1 91.9↑3.2 GPT-Image-1 77.3↑1.6 93.8↓0.7 83.0↓3.9 97.5↓0.1 83.4↑5.9 98.0↑1.4 80.8↑6.1 95.4↑1.1 81.1↑2.4 96.2↑0.4

ability to generate accurate, contextually rich visual representations. Figure 5 shows more qualitative examples from the evaluated T2I models.

#### 5.3 Evaluation on Two-Stage Pipeline Setting

We conduct an additional experiment using a pipeline that separates reasoning from image generation. In this setup, GPT-4o [24] first reasons about the original prompt and converts it into a visually explicit description, which is then fed to a T2I model. The quantitative results for images generated with these detailed, rewritten prompts are shown in Table 3. This pipeline significantly improves the reasoning accuracy for almost all models. This suggests that when prompts clearly express their intended meaning, models can generate appropriate scenes effectively, but they struggle to infer the underlying meaning when the prompts are not that straightforward.

The improvement achieved with this pipeline demonstrates the maximum potential these models can reach. Among diffusion-based models, Flux and Stable Diffusion models show the largest improvements, with their performances becoming comparable to HiDream. All these models fall within a score interval of 5.0, indicating similar performance when given clear, direct prompts. In Textual Image Design, improvements are minimal because performance here depends mainly on text synthesis, which cannot be enhanced with rewriting prompts.

In the pipeline setting, the “Thinking” mode of Bagel [7] is disabled. Bagel shows a substantial overall increase in accuracy, indicating that the external expert LLM has stronger reasoning abilities than Bagel’s internal understanding module. Other AR-based models also see significant improvements, suggesting that their generative abilities have not yet fully incorporated reasoning from the prompt. With clearer prompts, the generated images are more coherent, leading to a moderate increase in quality scores. This is expected, as image quality is less dependent on the prompt itself.

GPT-Image-1 [34] shows only a slight improvement, as it already achieves high scores with the original benchmark prompts. Interestingly, its accuracy score in Textual Image Design decreases slightly. By comparing the images generated from original and rewritten prompts, it can be seen that with the original concise prompts, the model can imagine and generate more creative content, but with rewritten detailed prompts, it is constrained to depict only what is explicitly described, limiting its freedom in creativity.

Idiom

[Figure 161]

GPT-Image-1 Gemini 2.0 playground-v2.5 SD-3.5-large Emu3

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

He told a funny joke to break the ice at the start of the meeting.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

At parties, she felt like a wallflower, too shy to join in on the dancing.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Create an infographic summarizing American coffee consumption trends in 2020.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Design a vibrant promotional poster for a tropical theme park…

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

The US president giving a speech at Brandenburg Gate in 1987

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

The first mammal successfully cloned from an adult somatic cell in 1996

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

A trampoline with a bowling ball, a basketball, and a ping pong ball placed on it

[Figure 196]

A seesaw with a 1-kilogram cotton ball on one end and a 1-kilogram iron cube on the other.

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Figure 5: Qualitative examples.

### 6 Conclusion and Discussions

Conclusion. In this study, we introduce T2I-ReasonBench, a novel benchmark designed to evaluate the reasoning capabilities of T2I generative models. While existing benchmarks primarily assess models’ abilities to follow literal prompts, T2I-ReasonBench challenges models to interpret implicit meanings across four dimensions: Idiom Interpretation, Textual Image Design, Entity-Reasoning,

and Scientific-Reasoning. Our evaluation of 14 state-of-the-art T2I models reveals that open-source models have significant limitations in reasoning ability. While proprietary models such as GPTImage-1 [34] and Gemini-2.0 [40] demonstrate stronger reasoning and knowledge integration, there is still considerable room for improvement.

Future Work. To advance the field of reasoning-informed T2I generation, future research should focus on incorporating structured knowledge bases and integrating reasoning mechanisms within T2I models. Additionally, expanding benchmarks to encompass a broader spectrum of reasoning tasks can provide deeper insights into model capabilities. By addressing these areas, we aim to bridge the gap between current T2I capabilities and the goal of achieving models that can generate images with a deep understanding of context nuance, and implicit meaning. At the same time, the community should be aware of the potential negative social impact of image generation models being used to generate fake images that mislead people.

### References

- [1] The exhaustive list of american idioms. https://learn-esl.org/index.html, 2023.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [3] Haoyu Chen, Xiaojie Xu, Wenbo Li, Jingjing Ren, Tian Ye, Songhua Liu, Ying-Cong Chen, Lei Zhu, and Xinchao Wang. Posta: A go-to framework for customized artistic poster generation. arXiv preprint arXiv:2503.14908, 2025.
- [4] Kaijie Chen, Zihao Lin, Zhiyang Xu, Ying Shen, Yuguang Yao, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. R2i-bench: Benchmarking reasoning-driven text-to-image generation. arXiv preprint arXiv:2505.23493, 2025.
- [5] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [6] Claude. Claude 3.5 sonnet. https://claude.ai/, 2023.
- [7] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [9] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-to-image generation via transformers. arXiv preprint arXiv:2105.13290, 2021.
- [10] Chengqi Duan, Rongyao Fang, Yuqing Wang, Kun Wang, Linjiang Huang, Xingyu Zeng, Hongsheng Li, and Xihui Liu. Got-r1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning. arXiv preprint arXiv:2505.17022, 2025.
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [12] Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, et al. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639, 2025.
- [13] Xingyu Fu, Muyu He, Yujie Lu, William Yang Wang, and Dan Roth. Commonsense-t2i challenge: Can text-to-image generation models understand commonsense? arXiv preprint arXiv:2406.07546, 2024.
- [14] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. arXiv preprint arXiv:2310.11513, 2023.
- [15] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.
- [18] hidream. hidream. https://github.com/HiDream-ai/HiDream-I1, 2024.
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [20] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- [21] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20406–20417, 2023.
- [22] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation . IEEE Transactions on Pattern Analysis Machine Intelligence, (01):1–17, January 5555.
- [23] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.
- [24] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [25] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [26] Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, et al. Genai-bench: Evaluating and improving compositional text-to-visual generation. arXiv preprint arXiv:2406.13743, 2024.
- [27] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.
- [28] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024.
- [29] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024.
- [30] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.
- [31] Fanqing Meng, Wenqi Shao, Lixin Luo, Yahong Wang, Yiran Chen, Quanfeng Lu, Yue Yang, Tianshuo Yang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Phybench: A physical commonsense benchmark for evaluating text-to-image models. arXiv preprint arXiv:2406.11802, 2024.
- [32] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [33] Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.
- [34] OpenAI. Gpt-image-1. https://openai.com/index/image-generation-api/, 2023.
- [35] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. arXiv preprint arXiv:2102.12092, 2021.
- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.
- [37] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [38] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.

- [39] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [40] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [41] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [42] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.
- [44] Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, and Lei Zhang. Tiif-bench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.
- [45] Xindi Wu, Dingli Yu, Yangsibo Huang, Olga Russakovsky, and Sanjeev Arora. Conceptmix: A compositional image generation benchmark with controllable difficulty. Advances in Neural Information Processing Systems, 37:86004–86047, 2024.
- [46] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [47] Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris Callison-Burch, Ranjay Krishna, Aniruddha Kembhavi, et al. Scaling text-rich image understanding via code-guided synthetic multimodal data generation. arXiv preprint arXiv:2502.14846, 2025.
- [48] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [49] Shijie Zhou, Ruiyi Zhang, Yufan Zhou, and Changyou Chen. A high-quality text-rich image instruction tuning dataset via hybrid instruction generation. arXiv preprint arXiv:2412.16364, 2024.

Appendix

- A More details on prompt collection process

Idiom Interpretation. In idiom collection, we leverage a book titled “The Exhaustive List of American Idioms” [1], which systematically documents over 11k idioms. These idioms were collected from diverse sources, including TV shows, movies, and everyday conversations. Each idiom in the book is accompanied by its actual meaning in context. In addition to this resource, we also refer to idioms available on the Internet. From this extensive pool, we manually select 200 idioms that are commonly used in daily life and challenging for T2I models due to their figurative meanings. We input the selected idioms and their actual meanings into an LLM and prompt it to generate new sentences. These sentences are designed to describe visible scenes involving the idioms, providing contextual clues for reasoning while avoiding directly revealing the idiom’s meaning.

Textual Image Design. For textual image design, we collect 6 types of text-rich images from 4 distinct sources.

- (1) LLAVAR-2 Dataset [49]: This dataset contains 42k text-rich images sourced from LAION [38], representing various categories such as quotes, memes, book covers, posters, and product packaging. However, images in this dataset are of various quality and formats, so we filter out 80 aesthetically pleasing images that have a resolution greater than 384x384 and exhibit clear design intentions.
- (2) InfographicVQA Dataset [30]: This dataset comprises 5k high-quality infographics. We select 40 with normal height-width ratio that exemplify well-crafted layouts to convey structured information.
- (3) POSTA Dataset [3]: This dataset includes over 300 posters with professional background, layout, and text formats designed by experts. We select 40 posters that demonstrate a balance between text and visual design elements.
- (4) CoSyn-400k Dataset [47]: This dataset consists of 400k synthetic text-rich images, generated by LLM-drive codes. These images cover diverse formats, such as charts, diagrams, tables, documents (e.g., menus or business cards), math examples, and musical scores. From this dataset, we select 40 samples that exemplify structured text-visual integration, including 10 tables, 10 diagrams, and 20 documents.

The design intentions of the collected images are summarized using Qwen2.5-VL [2], yielding 200 prompts in this dimension.

- B Evaluation Framework

We adopt Qwen2.5-VL [2] as our evaluation tool due to its state-of-the-art performance in visualtextual grounding and fine-grained object recognition. The evaluation of T2I models on our benchmark focuses on two key aspects: reasoning accuracy and image quality. To assess this, we generate specific pairs of question and criterion for each prompt with DeepSeek-R1 [16]

Table 4, 5, 6, and 7 present the templates used to generate the prompt-specific question-criterion pairs for Idiom Interpretation, Textual Image Design, Entity-Reasoning and Scientific-Reasoning, respectively. Each template is tailored to focus on the unique aspects of its corresponding dimension.

Table 8 presents the template used to evaluate the generated images for all four dimensions. Only the evaluation question-criterion pairs need to be replaced for each prompt.

<USER>: I have a text-to-image generation model that can generate images based on given prompts. However, the model is not perfect and may fail to accurately capture the meaning of the prompt or depict it correctly. Your task is to evaluate the generated image based on a specific prompt that contains an idiom. Given the prompt: {‘id’: {prompt_id}, ‘prompt’: {prompt}, ‘idiom’: {idiom}, ’idiom_meaning’: {idiom_meaning}}, you need to:

- 1. identify what should be depicted in the image or the meaning the image should convey.
- 2. analyze the prompt and create a list of questions based on the key elements that the image should be checked against.
- 3. consider factors that could impact the aesthetics or visual quality of the image and list relevant questions. Please also design a scoring criterion for each question, where a score of 1 means “yes (to the question)”, 0 means “no”, and 0.5 means “partially yes”. Provide your answer in json format: {{‘id’: [prompt id], ‘prompt’: [the prompt], ‘image_content’: [what the image should convey], ‘reason_evaluation’: (here should be a dictionary with 3-5 pairs of question and criterion: ‘q1’: [question 1], ‘c1’: [criterion 1], ‘q2’: [question 2], ‘c2’: [criterion 2]...), ‘quality_evaluation’: (same format as ‘reason_evaluation’ with 1-3 pairs of question and criterion)}}. <ASSISTANT>: { “id”: 1, “prompt”: “He told a funny joke to break the ice at the start of the meeting.”, “image_content”: "The image should depict a person telling a joke in a meeting setting, with the audience showing visible signs of relaxation and social easing (e.g., smiling, laughing, or beginning to converse) as a result. It must convey the metaphorical meaning of ‘breaking the ice’—reducing tension and initiating interaction—without literal depictions of ice or breaking.”, “reason_evaluation”: {

- “q1”: “Does the image show a meeting environment (e.g., conference table, office setting, or group in professional attire)?”, “c1”: “1: Clear meeting setting; 0.5: Ambiguous setting (e.g., generic group); 0: No meeting context”,
- “q2”: “Is a person visibly acting as the joke-teller (e.g., speaking, gesturing, or commanding group attention)?”, “c2”: “1: Clear speaker with active role; 0.5: Speaker present but passive; 0: No identifiable speaker”,
- “q3”: “Do audience members display relaxed/post-icebreaking reactions (e.g., genuine smiles, laughter, or eased body language)?”, “c3”: “1: Clear reactions showing tension relief; 0.5: Mild reactions (e.g., slight smiles); 0: No reactions or negative expressions”,
- “q4”: “Is the idiom depicted metaphorically (avoiding literal ice/breaking)?”, “c4”: “1: Entirely metaphorical; 0.5: Partial literal elements (e.g., ice visuals but not dominant); 0: Literal depiction (e.g., shattered ice)” }, “quality_evaluation”: {

- “q1”: “Is the image visually clear with well-defined subjects (not blurry or distorted)?”, “c1”: “1: High clarity; 0.5: Minor blur/distortion; 0: Major quality issues”,
- “q2”: “Is the composition balanced (e.g., speaker and audience arranged cohesively, no awkward cropping)?”, “c2”: “1: Harmonious composition; 0.5: Minor imbalances; 0: Chaotic or disjointed layout”,
- “q3”: “Are colors/lighting appropriate and enhance the mood (e.g., warm tones for positivity, no visual clashes)?”, “c3”: “1: Lighting/colors support the scene; 0.5: Partially fitting; 0: Inconsistent or jarring”} }

#### Table 4: Template used by DeepSeek-R1 to generate evaluation question-criterion pairs for the dimension of Idiom Interpretation and example output. The text highlighted in cyan should be replaced with details from the specific prompt.

<USER>: I have a text-to-image generation model that can generate images based on given prompts. However, the model is not perfect and may fail to accurately reflect the prompt or depict the details correctly. Given a prompt which is a design intention for a text-rich image like infographic or poster, your task is to evaluate whether the generated image correctly fulfill the design intention. Here is the prompt: {{‘id’: {prompt_id}, ‘prompt’: {prompt}}, you need to:

- 1. identify what should be depicted in the image and its functional purposes.
- 2. analyze the design intention and create a list of questions based on the key elements that the image should be checked against, including presence of required text elements.
- 3. consider factors that could impact the aesthetics or visual quality of the image and list relevant questions. Please also design a scoring criterion for each question, where a score of 1 means “yes (to the question)”, 0 means “no”, and 0.5 means “partially yes”. Provide your answer in json format: {{‘id’: [prompt id], ‘prompt’: [the prompt], ‘image_content’: [what the image should convey], ‘reason_evaluation’: (here should be a dictionary with 3-5 pairs of question and criterion: ‘q1’: [question 1], ‘c1’: [criterion 1], ‘q2’: [question 2], ‘c2’: [criterion 2]...), ‘quality_evaluation’: (same format as ‘reason_evaluation’ with 1-3 pairs of question and criterion)}}.

#### Table 5: Template used by DeepSeek-R1 to generate evaluation question-criterion pairs for the dimension of Textual Image Design. The text highlighted in cyan should be replaced with details from the specific prompt.

<USER>: I have a text-to-image generation model that can generate images based on given prompts. However, the prompts given to the model may contain implicit meanings or entities that are not directly stated. Your task is to evaluate whether the generated image accurately represents the intended meaning of the prompt. Given the prompt: {{‘id’: {prompt_id}, ‘prompt’: {prompt}, ‘explicit_meaning’: {explicit_meaning}}, you need to:

- 1. identify what should be depicted in the image in order to fully and accurately reflect the explicit meaning of the prompt.
- 2. identify the entity that the model needs to infer from the prompt, and create a list of questions that check whether the image has correctly identified and depicted this entity.
- 3. Consider other elements or details in the prompt (apart from the implicit entity), create a list of questions that check if the image accurately reflects these additional key elements.
- 4. consider factors that could impact the aesthetics or visual quality of the image and list relevant questions. Please also design a scoring criterion for each question, where a score of 1 means “yes (to the question)”, 0 means “no”, and 0.5 means “partially yes”. Provide your answer in json format: {{‘id’: [prompt id], ‘prompt’: [the prompt], ‘explicit_meaning’: [the explicit meaning], ‘image_content’: [what the image should depict], ‘entity_evaluation’: (here should be a dictionary with 1-3 pairs of question and criterion: ‘q1’: [question 1], ‘c1’: [criterion 1], ‘q2’: [question 2], ‘c2’: [criterion 2]...), ‘other_details_evaluation’: (same format as ‘entity_evaluation’ with 1-3 pairs of question and criterion), ‘quality_evaluation’: (same format as ‘entity_evaluation’ with 1-3 pairs of question and criterion)}}.

#### Table 6: Template used by DeepSeek-R1 to generate evaluation question-criterion pairs for the dimension of Entity-Reasoning. The text highlighted in cyan should be replaced with details from the specific prompt.

<USER>: I have a text-to-image generation model that can generate images based on given prompts. However, the prompts given to the model imply scientific laws (e.g., physics, chemistry, biology, or astronomy) that can affect how the scene looks without explicit explanation. Your task is to evaluate whether the generated image accurately reflects the scientific law and correctly portrays the resulting scene. Given the prompt: {{‘id’: {prompt_id}, ‘prompt’: {prompt}, ‘explicit_meaning’: {explicit_meaning}}, you need to:

- 1. describe what should be depicted in the image in order to fully and accurately reflect the explicit meaning of the prompt.
- 2. identify any scientific law(s) that the model needs to infer from the prompt, and create a list of questions that check whether the image correctly demonstrates and complies with these scientific laws.
- 3. consider other elements or details in the prompt that are not directly affected by the scientific law(s), create a list of questions that check if the image accurately represents these additional key elements.
- 4. consider factors that could impact the aesthetics or visual quality of the image and list relevant questions. Please also design a scoring criterion for each question, where a score of 1 means “yes (to the question)”, 0 means “no”, and 0.5 means “partially yes”. Provide your answer in json format: {{‘id’: [prompt id], ‘prompt’: [the prompt], ‘explicit_meaning’: [the explicit meaning], ‘image_content’: [what the image should depict], ‘scientific_evaluation’: (here should be a dictionary with 2-4 pairs of question and criterion: ‘q1’: [question 1], ‘c1’: [criterion 1], ‘q2’: [question 2], ‘c2’: [criterion 2]...), ‘other_details_evaluation’: (same format as ‘scientific_evaluation’ with 1-3 pairs of question and criterion), ‘quality_evaluation’: (same format as ‘scientific_evaluation’ with 1-3 pairs of question and criterion)}}.

#### Table 7: Template used by DeepSeek-R1 to generate evaluation question-criterion pairs for the dimension of Scientific-Reasoning. The text highlighted in cyan should be replaced with details from the specific prompt.

<USER>: Describe this image. <ASSISTANT>: The image depicts a group of four people in a meeting setting, with one person standing and the others seated around a table. The standing individual, dressed in a dark suit and light blue shirt, is playfully pointing at the nose of one of the seated individuals, who is smiling broadly... <USER>: Based on the image and your previous description, answer the following questions: q1, q2, ... For each question, assign a score of 1, 0.5 or 0 according to the corresponding scoring criteria: c1, c2, ... Here are the questions and criteria: {accuracy_evaluation_qc}. Carefully consider the image and each question before responding, then provide your answer in json format: {{“reason”: [your detailed reasoning], “score”: [s1,s2, ...]}}

<ASSISTANT>: { “reason”: [ “q1: The image clearly shows a meeting environment with a conference table, office-like setting, and individuals in professional attire. This aligns well with the criteria for a clear meeting setting.”,

- “q2: One person is visibly acting as the joke-teller by making a playful gesture towards another person’s face, which suggests an active role in engaging the group. This meets the criteria for a clear speaker with an active role.”,
- “q3: The audience members are displaying relaxed and post-icebreaking reactions, such as genuine smiles and laughter, indicating that the tension has been relieved. This fulfills the criteria for clear reactions showing tension relief.”,
- “q4: The idiom ’break the ice’ is depicted entirely metaphorically through the playful interaction and the text overlay, without any literal elements like actual ice. This satisfies the criteria for an entirely metaphorical depiction.” ], “score”: [1, 1, 1, 1] } <USER>: Based on the image and your previous description, answer the following questions: q1, q2, ... For each question, assign a score of 1, 0.5 or 0 according to the corresponding scoring criteria: c1, c2, ... Here are the questions and criteria: {quality_evaluation_qc} Carefully consider the image and each question before responding, then provide your answer in json format: {{“reason”: [your detailed reasoning], “score”: [s1,s2, ...]}} <ASSISTANT>: { “reason”: [ “The image is visually clear with well-defined subjects. There is no noticeable blur or distortion, allowing for easy identification of the individuals and their actions. Therefore, it scores high on clarity.”, “The composition is balanced, with the speaker positioned centrally and the audience arranged cohesively around the table. There is no awkward cropping, and all elements contribute to a harmonious arrangement.”, “The colors and lighting are appropriate, enhancing the mood of the scene. Warm tones and even lighting create a positive and inviting atmosphere, supporting the lighthearted interaction depicted.” ], “score”: [1, 1, 1] }

- Table 8: Template used by Qwen2.5-VL to evaluate generated images in T2I-ReasonBench. The text highlighted in cyan should be replaced with the specific evaluation question and criterion pairs for the given prompt. An example output for Idiom Interpretation is also provided.

