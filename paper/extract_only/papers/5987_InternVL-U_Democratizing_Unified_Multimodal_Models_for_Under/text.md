# arXiv:2603.09877v1[cs.CV]10Mar2026

[Figure 1]

2026-3-11

## InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing

Changyao Tian1,2*, Danni Yang1,3*, Guanzhou Chen1,4*, Erfei Cui1,4*, Zhaokai Wang1,4*, Yuchen Duan1,2*, Penghao Yin1,5*, Sitao Chen1,3, Ganlin Yang1,6, Mingxin Liu4, Zirun Zhu4, Ziqian Fan7, Leyao Gu4, Haomin Wang1,4, Qi Wei1,8, Jinhui Yin1,8, Xue Yang4, Zhihang Zhong4, Qi Qin1, Yi Xin1, Bin Fu1, Yihao Liu1, Jiaye Ge1, Qipeng Guo1, Gen Luo9, Hongsheng Li2, Yu Qiao1,

Kai Chen1 and Hongjie Zhang1

1Shanghai AI Laboratory, 2CUHK MMLab, 3Fudan University, 4Shanghai Jiao Tong University, 5Tsinghua University, 6University of Science and Technology of China, 7South China University of Technology, 8Nanjing University, 9Xiamen University

##### Abstract

Unified multimodal models (UMMs) that integrate understanding, reasoning, generation, and editing face inherent trade-offs between maintaining strong semantic comprehension and acquiring powerful generation capabilities. In this report, we present InternVL-U, a lightweight 4B-parameter UMM that democratizes these capabilities within a unified framework. Guided by the principles of unified contextual modeling and modality-specific modular design with decoupled visual representations, InternVL-U integrates a state-of-the-art Multimodal Large Language Model (MLLM) with a specialized MMDiT-based visual generation head. To further bridge the gap between aesthetic generation and high-level intelligence, we construct a comprehensive data synthesis pipeline targeting high-semanticdensity tasks, such as text rendering and scientific reasoning, under a reasoning-centric paradigm that leverages Chain-of-Thought (CoT) to better align abstract user intent with fine-grained visual generation details. Extensive experiments demonstrate that InternVL-U achieves a superior performance - efficiency balance. Despite using only 4B parameters, it consistently outperforms unified baseline models with over 3× larger scales such as BAGEL (14B) on various generation and editing tasks, while retaining strong multimodal understanding and reasoning capabilities.

GitHub Repo https://github.com/OpenGVLab/InternVL-U

HuggingFace Model https://huggingface.co/InternVL-U/InternVL-U

GenEditEvalKit https://github.com/open-compass/GenEditEvalKit

TextEdit Benchmark https://github.com/open-compass/TextEdit

* Equal Contribution. Corresponding author. This work was done when Changyao Tian was an intern at Shanghai AI Laboratory.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

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

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

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

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

###### Figure 1: Showcases of InternVL-U for general text-to-image generation (top) and image editing (bottom). InternVL-U supports high-fidelity image generation and editing at any resolution. 2

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

###### Spatial-centricScience-centric

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

###### Reasoning-centricPerception

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Color the graph with two distinct colors such that no two connected nodes share the same color.

A Structural formulas of Methyl isonicotinate.

[Figure 75]

Draw a right triangle ABC with the right angle at vertex B. Label side AB with a length of 6.0 units. Ensure the triangle is drawn to scale, with vertex A at the top,

<think>Upon analyzing the graph structure, it is evident that the graph is bipartite. The nodes can be partitioned into two disjoint sets, Set A and Set B, with every edge connecting a node in Set A to a node in Set B.

<think><svg width="128" height="128" stroke="#000" stroke-linecap="round" stroke-width="0.92" fontfamily="sans-serif" viewBox="0 0 128 128"><line x1="64.01" x2="64.01"

- vertex B at the bottom-left, and
- vertex C at the bottom-right. Mark the right angle at B and label it as 90°. Clearly label all vertices A, B, and C.

- - Set A: [84] - Set B: [38, 41, 63, 82] Subsequently, to ensure that no two connected nodes share the same color, one color is asigned to al nodes in Set A, and a diferent color is asigned to al nodes in Set B.
- - Asign Color 1 (e.g., #FDE68A) to

- the nodes in Set A.

- Asign Color 2 (e.g., #FF8C8C) to

- the nodes in Set B. Proceeding to verify the constraints

- y1="21.33" y2="32" /><line x1="64.01" x2="76.8" y1="32"
- y2="42.67" /><line x1="64.01" x2= "51.2" y1="32" y2= "42.67" /><line x1= "76.8" x2="89.6" y1

[Figure 76]

[Figure 77]

[Figure 78]

="42.67" y2="32" /> <line x1="51.2 . </think>

.</think>

Turn this image into a meme to express your suprise for the gift.

将这张图变成一张表现 小猫试图安抚生气大哥 的搞笑梗图。

[Figure 79]

[Figure 80]

生成一张可爱又害羞的小柴 犬表情包。

生成一张爱因斯坦的卡通表 情包，吐舌表达滑稽的心情。

<think>为了表现小猫试图安 抚生气大哥的搞笑梗图，可 以在图像上方添加文字“大 哥，别生气了”，仅添加文 字，其他视觉内容保持不变。

<think>一只毛茸茸的白色小 柴犬，圆溜溜的大眼睛睁得 圆圆的，脸颊泛着淡淡的红 晕，耳朵竖起，表情呆萌又 无辜，仿佛在说“我好可爱 啊”。

<think>爱因斯坦的卡通形象， 留着标志性的胡须，面带得意 又略带狡黠的微笑，右手食指 竖起，似乎在强调或表达某种 观点。画面下方配有文字“爱 因斯坦”，点明了人物身份。

<think>To expres your surprise for the gift, add the text 'WHEN YOU GET A GIFT' at the top of the image, with only the text added, and no changes to the other visual content. </think>

Humor-centric

</think>

[Figure 81]

[Figure 82]

[Figure 83]

</think>

整体风格简洁 可爱，常用于 表达对某人或 某事的赞赏、 调侃或戏谑。 </think>

[Figure 84]

[Figure 85]

以2026年的生肖为主题，帮 我生成一张主题的图片。

Please write the solution in this 4x4 Sudoku game picture.

<think>以2026年生肖马为主题的 图片，画面中央是一匹昂首奔跑的 骏马，姿态向右上方跃起，象征着 积极向上的精神。骏马采用国潮插 画风与金属鎏金质感结合：马身以 深红渐变为底，边缘勾勒金线，鬃 毛与尾巴延展成祥云与火焰纹的流 线，营造出速度与力量感；背景为 浅蓝到天蓝的渐变，点缀细小金粉 与放射状光斑，仿佛晨曦或星河。 下方配有竖排书法风文字“2026 新春”，右下角有篆刻风格的“马” 字印章。整体设计将传统生肖意象 与现代艺术表现手法融合，既具文 化辨识度，又充满动感与仪式感。 </think>

[Figure 86]

<think>To solve the 4x4 Sudoku puzzle shown in the image, we need to fill in the empty cells such that each row, each column, and the two 2x2 subgrids contain the numbers 1 through 4 exactly once. Here is the step-by-step solution:

- Row 1: [3, 1, 2, 4]
- Row 2: [2, 4, 1, 3]
- Row 3: [1, 3, 4, 2]
- Row 4: [4, 2, 3, 1] The final grid is:

[Figure 87]

- 3 1 2 4 2 4 1 3 1 3 4 2
- 4 2 3 1</think>

###### Figure 2: Showcases of InternVL-U for spatial-centric, perception, science-centric, humor-centric, and reasoning-centric text-to-image generation or editing tasks. InternVL-U demonstrates such core multimodal capabilities across various visual domains. 3

### Contents

- 1 Introduction 6
- 2 Related Work 7

- 2.1 Multimodal Large Language Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.2 Visual Generative Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 2.3 Unified Multimodal Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 3 Method: InternVL-U 8

- 3.1 Model Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 3.1.1 Overall Design Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.1.2 Visual Generation Head . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 3.2 Training Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 3.2.1 Training Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 3.2.2 Training Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 4 Data Construction 12

- 4.1 Open-source Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.2 General Data Preprocessing and Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 4.2.1 General Preprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 4.2.2 Text-to-Image Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.2.3 Image Editing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4.3 Text-centric Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 4.3.1 Text-to-Image Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 4.3.2 Image Editing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 4.4 Science-centric Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 4.4.1 General Science Generation Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 4.4.2 SVG-based Physics Editing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 4.4.3 Computer Science Editing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 4.5 Spatial-centric Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 4.5.1 Solid Geometry Editing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 4.5.2 Multi-view CAD Editing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 4.5.3 Spatial Rotation Editing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- 4.6 Humor-centric Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 4.7 Reasoning-centric Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- 5 Experiments 25

- 5.1 Experimental Setups . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 5.2 Multimodal Understanding and Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 5.3 Text-to-Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- 5.3.1 General Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 5.3.2 Text-centric Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 5.3.3 Knowledge-informed Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

- 5.4 Image Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

4

- 5.4.1 General Image Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- 5.4.2 Text-centric Image Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- 5.4.3 Reasoning-informed Image Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- 5.5 More Qualitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- 6 Conclusion 39

- A TextEdit Benchmark 51

- A.1 Design Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- A.2 Design Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51

- A.2.1 Scenario Taxonomy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
- A.2.2 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53

- A.3 MiniSet-500 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59

- B Data Construction Details 60

- B.1 Filtering details for General Science Image Generation . . . . . . . . . . . . . . . . . . . . . . 60
- B.2 Chemistry Text-to-Image Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- B.3 Computer Science Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- B.4 Solid Geometry . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61

#### 1. Introduction

Unified multimodal models (UMMs) have witnessed rapid advancement in recent years [13, 68, 113, 114, 120]. The emergence of models like GPT-4o [56] demonstrates that integrating native image generation with advanced linguistic capabilities not only enables users to execute complex visual tasks via natural language but also paves the way for exploring Artificial General Intelligence (AGI) and World Models [24, 31]. While closedsource models have exhibited remarkable general-purpose performance, the research community has actively explored various architectural and representational strategies to construct such unified models. These efforts can generally be categorized into two paradigms: (1) Fully-native UMMs [24, 31, 114, 124, 144, 145], which is trained from scratch or initialized from unimodal components (e.g., ViT, LLM) and jointly trained on multimodal understanding and generation tasks from scratch; and (2) Fully-ensemble UMMs [76, 102, 112, 139], which construct a unified system by post-hoc aligning pre-trained multimodal understanding models with pre-trained image generation models. However, both paradigms face significant limitations.

For fully-native UMMs, the community has yet to reach a consensus on the optimal design across modeling, representation, and architecture [49]. Not only do theoretical divergences exist, but no single approach has demonstrated a decisive advantage in performance or efficiency [68]. Furthermore, jointly training multimodal understanding and generation capabilities from scratch presents substantial engineering challenges, particularly in balancing the conflicting data distributions of different modalities. Crucially, this paradigm often necessitates foregoing the benefits of state-of-the-art (SOTA) multimodal understanding models [3, 19] already available in the community, thereby incurring prohibitive training costs and risks. Conversely, fully-ensemble UMMs typically attach an external and separately pre-trained image generator as a visual generation head [63, 108, 142]. In practice, they face a recurring trade-off. They can scale the head to very large parameter counts to reach top-tier visual quality, as in Qwen-Image [137] and Hunyuan Image 3.0 [9], but this substantially increases training and deployment cost. Alternatively, they can retain a smaller head while introducing elaborate and often fragmented conditioning pipelines, such as requiring multi-encoder text conditioning in Stable Diffusion

- 3 [35] or designs that decouple text and image conditions in Z-Image [8]. Either way, the resulting interfaces are difficult to align cleanly with the hidden-state space of a single MLLM, which limits how much can be gained from post-alignment training under constrained resources.

To address these challenges, we first systematically analyze the design principles of unified models from three dimensions: modeling, architecture, and representation. We posit that within a unified semantic reasoning space, a model should employ hybrid modeling objectives to accommodate the statistical properties of different modalities, adhere to modality-specific modularity to enhance overall architectural efficiency, and utilize decoupled visual representations to balance high-level semantic understanding with low-level pixel reconstruction. Guided by these principles, we propose InternVL-U, a streamlined and efficient unified multimodal model. Built upon InternVL 3.5 [129], an open-source MLLM with SoTA performance, we integrate a custom MMDiT-based visual generation head with a unified semantic conditioning interface aligned to the MLLM hidden states. Through a three-stage progressive training strategy, InternVL-U not only inherits the robust understanding and reasoning capabilities of its predecessor but also acquires powerful multimodal generation and editing skills. Furthermore, InternVL-U leverages self-reflection reasoning to utilize world knowledge inherited from the MLLM, further enhancing these capabilities.

Design unification alone, however, does not guarantee a truly AGI-oriented UMM, because the capabilities a model ultimately acquires are strongly shaped by the objectives and data regimes it is trained on [108, 129]. A unified multimodal model is expected to be both visually competent and semantically reliable, yet today’s visual generation and multimodal understanding models are optimized for fundamentally different goals and use cases. Traditional generation models primarily target low-level perceptual quality, such as aesthetics and visual fidelity, whereas understanding models emphasize high-level intelligence, including knowledge injection and reasoning emergence. This objective mismatch poses a major obstacle to developing AGI-oriented UMMs. We argue that a key driver is the domain gap in training data distributions. Generation models are predominantly trained on natural-image corpora (e.g., portraits and landscapes) rich in texture and high-frequency details but relatively low in semantic density. In contrast, understanding models rely heavily on text-rich and structurally organized data, including synthetic images such as GUIs, infographics, and OCR-centric documents, which may exhibit simpler textures but contain dense semantics, abundant textual cues, and structured knowledge.

Consistent with this diagnosis, next-generation commercial models (e.g., Nano-Banana Pro [29]) have begun to actively narrow the gap by emphasizing typographic precision and knowledge-faithful content

creation beyond aesthetics alone. Inspired by this trend, and to unlock the potential of InternVL-U as an AGIoriented UMM, we construct a comprehensive multimodal data synthesis pipeline targeting various capabilities, including text rendering, scientific reasoning, spatial and humor generation. Specifically, for “high semantic density” text scenarios, we design a fully automated text rendering and editing pipeline covering bilingual typography and local consistency editing to address the lack of symbolic precision in generative models. For “knowledge-intensive” scientific scenarios, we leverage programmatic tools (e.g., GeoGebra, SVG) and academic corpora to construct structured visual-text data across disciplines like mathematics, physics, and computer science. Furthermore, to better capture the abstract and underspecified nature of user intent, we propose a “Reasoning-centric” data synthesis paradigm. By introducing explicit Chain-of-Thought (CoT), we transform vague instructions into executable steps containing planning and constraints, achieving a leap from simple instruction following to deep intent alignment in tasks such as meme generation, geometric transformation, and logically constrained editing. By integrating data from these pipelines, InternVL-U retains its powerful general-purpose generation capabilities while significantly enhancing its ability to generate accurate text rendering and editing, spatial reasoning, humor generation, and multidisciplinary scientific knowledge.

Extensive empirical evaluations demonstrate that InternVL-U achieves a superior balance between performance and efficiency. As discussed in Section 5, in text-to-image generation, it consistently outperforms existing unified models across general, text-centric, and knowledge-intensive benchmarks, approaching the capabilities of significantly larger specialized generation models. Specifically, it exhibits exceptional instruction following and effectively addresses the deficiency of previous unified architectures in legible text rendering. Crucially, integrating the CoT strategy serves as a vital catalyst for both generation and editing, enabling the model to excel in knowledge-rich generation and complex logic-dependent editing tasks while delivering remarkable performance gains. Furthermore, regarding multimodal understanding, InternVL-U retains the robust capabilities of its predecessor, surpassing comparable unified baselines without compromising its native visual-language comprehension. To facilitate efficient benchmarking for the community, we further introduce GenEditEvalKit [100] to streamline UMM evaluation and TextEdit Benchmark [101] to provide a more comprehensive text-editing benchmark.

To summarize, our contributions are threefold:

- • We propose InternVL-U, an efficient UMM built on Unified Contextual Modeling, Decoupled Visual Representations, and Modality-Specific Modularity. By integrating a customized MMDiT-based generation head with decoupled ViT and VAE representations, our architecture resolves the conflict between semantic understanding and pixel reconstruction, enabling powerful generative skills without compromising native understanding capabilities.
- • We construct a comprehensive data pipeline targeting high-semantic-density tasks, including text rendering, scientific reasoning, spatial manipulation, and humor generation. Furthermore, we introduce the “Reasoning-centric” paradigm that leverages Chain-of-Thought to transform abstract user instructions into executable plans, effectively bridging the gap between vague intent and precise visual execution.
- • Extensive evaluations demonstrate that InternVL-U consistently outperforms unified baselines in generation and editing, particularly in text-rich and knowledge-intensive scenarios. Crucially, it retains robust multimodal understanding capabilities, surpassing comparable unified models without compromising visual-language comprehension.

#### 2. Related Work

###### 2.1. Multimodal Large Language Models

Recent advancements in Multimodal Large Language Models (MLLMs) have revolutionized vision-language tasks. Representative open-source families, such as LLaVA [80–82], Qwen-VL [3–5, 128], and InternVL [17– 19, 89, 90, 129, 163], alongside proprietary models like GPT [1, 97] and Gemini [21, 28, 46, 115], have demonstrated exceptional capabilities in visual understanding. Standard MLLMs typically adopt a unified architecture connecting a vision encoder [32] with an LLM [2, 121, 147] via adapters [70, 80]. Furthermore, recent trends have expanded towards processing interleaved image-text sequences [25, 31, 118, 149] and video understanding [71, 77, 93, 130, 148, 151], pushing the boundaries of long-context multimodal interaction.

- 2.2. Visual Generative Models

Visual generation has evolved from early GANs [45, 57, 58] to dominant diffusion-based frameworks [51, 63, 108] and flow-matching paradigms [79, 85], which offer superior scalability and sample quality. Parallelly, discrete token-based approaches [10, 36, 107, 119] generate images autoregressively via VQ-style codecs, enabling a unified token space with LLMs. State-of-the-art text-to-image models, including Stable Diffusion

- 3.5 [108], FLUX.2 [64], Hunyuan Image 3.0 [9], and Qwen-Image [137], emphasize instruction following and complex scene generation. Meanwhile, the community has explored data-centric and architecture-centric approaches to improve rendering of complex structures and text, as well as generalization to long prompts and multi-concept descriptions [8, 116, 123, 137]. In addition, several closed-source state-of-the-art systems, such as Nano Banana Pro [29], GPT-Image-1.5 [98], and Seedream 4.0 [110], have demonstrated strong performance on instruction-following and complex multi-concept image generation tasks. Additionally, instruction-driven editing [6, 65, 83] has gained traction, requiring models to manipulate specific regions while preserving semantic consistency.

- 2.3. Unified Multimodal Models

Unified Multimodal Models (UMMs) aim to integrate understanding, generation, and editing within a single foundation model. By coupling a powerful LLM with visual tokenizers or latent representations, UMMs can understand and generate visual content in a unified manner. Existing approaches generally fall into two categories: (1) Auto-Regressive Discrete-token methods, such as Chameleon [114], Emu3 [24], and SynerGen-VL [69], treat image generation as next-token prediction, naturally unifying modalities but often facing challenges in visual fidelity. (2) Diffusion/Hybrid methods, such as BLIP-3o [14], BAGEL [31], Ovis-U1 [124], and others [49, 68, 73, 87, 111, 120, 127], combine the reasoning power of LLMs with the high-fidelity generation of diffusion (or flow matching) models. Recent works also explore other distinct unified paradigms [72, 95, 113, 145, 150]. Following this line of research, our work InternVL-U is built upon the open-source MLLM (i.e. InternVL3.5 [129]) to unify general-purpose understanding, generation, and editing, as well as capabilities for domain-specific scenarios (e.g. text rendering, science, memes), within a single framework.

#### 3. Method: InternVL-U

###### 3.1. Model Architecture

In this section, we elaborate on the overall architectural design principle of InternVL-U, along with the detailed architecture of our visual generation head.

###### 3.1.1. Overall Design Principles

As shown in Figure 3, unlike recent approaches that enforce a homogenized processing pipeline for all modalities [74], our architecture is driven by the philosophy that distinct modalities require tailored handling to maximize efficiency and performance. We articulate our design principles through three key dimensions: modeling paradigm, structural efficiency, and data representation.

Unified Contextual Modeling with Modality-Adaptive Generation. Our first principle addresses the dichotomy between multimodal understanding (context) and generation (prediction). We argue that while contextualization benefits from a unified representation to facilitate deep semantic fusion, generation shall respect the inherent statistical properties of each modality.

- • Unified Context, Adaptive Targets: In the context phase, we project both visual and linguistic tokens into a shared latent space, employing a unified autoregressive (AR) paradigm with causal masking. This ensures that the model captures the complex high-level semantic dependencies between modalities during the reasoning process.
- • Hybrid Generative Objectives: However, for the prediction targets, we diverge from the “tokenizationfor-all” approach [24]. Text, being inherently discrete and sequential, is best modeled via a categorical distribution over a finite vocabulary using cross-entropy loss. Conversely, visual signals are continuous and spatially correlated. While discrete visual tokenization is a viable alternative (as in VQ-VAE-based AR

Velocity Prediction

Next Token Prediction

Overall Design Principles

Text Branch

Visual Branch

Text Gen. Head Visual Gen. Head

Unified Contextual Modeling with Modality-Adaptive Generation

Add Noise

Multimodal Context Embeddings

Visual Gen. Encoder

Unified Backbone for Multimodal Context Modeling

Structural Efficiency via Modality-Specific Modular Design

Visual Und. Encoder

Text Tokenizer

Text Tokenizer

Text Prompt

Text Prompt

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Decoupled Visual Representations for Understanding and Generation

[Figure 93]

[Figure 94]

v

[Figure 95]

Training Images Training Images

- Figure 3: The architectural design of InternVL-U. The framework highlights three design principles: (1) unified contextual modeling supporting modality-adaptive generation targets, (2) structural efficiency via a unified backbone with modality-specific modular design, and (3) decoupled visual representations for understanding and generation tasks. Und. and Gen. denote Understanding and Generation, respectively.

models), it may introduce quantization bottlenecks and make fine-grained spatial modeling less direct. Therefore, we adopt a hybrid AR + Diffusion modeling paradigm. We model image generation in a continuous multivariate probability space using Flow Matching (a generalized formulation of diffusion), while retaining the AR objective for text. This design allows the model to preserve the strengths of autoregressive language modeling for text, while leveraging the high-fidelity generation capabilities of diffusion-based methods for images.

Structural Efficiency via Modality-Specific Modular Design. Our second principle addresses the computational inefficiency of fully modality-agnostic architectures (e.g., Mixture-of-Transformer (MoT) [74]), which treat all modalities as uniform token sequences. We argue that different modalities possess varying "semantic densities": text is semantically dense, whereas raw visual patches are sparse and redundant.

- • Encoder-Based MLLM Initialization: To mitigate the parameter and FLOPs wastage inherent in processing raw modalities with a generic transformer, we incorporate modality-specific encoding stems. We initialize our multimodal context modeling backbone with an encoder-based architecture (leveraging a pre-trained ViT [19]) rather than a more monolithic or native multimodal design [88, 117]. This design introduces a necessary inductive bias that efficiently aggregates visual information before it enters the unified latent space.
- • Modality-Specific Generation Head: Furthermore, recognizing that the decoding requirements for text and images differ, we extend the pre-trained MLLM with a dedicated generation head based on the Multimodal Diffusion Transformer (MMDiT) [35] architecture for image generation. Instead of burdening the context modeling backbone with pixel-level synthesis, the MMDiT serves as a dedicated generative module that takes the unified hidden states as conditioning signals and synthesizes images in a continuous visual latent space. This hierarchical design ensures that the backbone focuses on semantic reasoning, while the specialized stems and heads handle modality-specific translation, resulting in a more unified yet computationally efficient UMM.

Decoupled Visual Representations for Understanding and Generation. Our third principle challenges the assumption that the visual representation used for comprehending an image must be identical to the one used for generating it. We propose an asymmetric representation strategy motivated by the observation that image

Dual-Stream Attention Block Dual-Stream FFN Block

Text Branch Visual Branch

Gate Gate

Gate Gate

Projection Out

Predicted Velocity

Multi Head Gated Self Attention

FFN Expert FFN Expert

MMDiT Blocks

x N

QKV Expert QKV Expert

Scale, Shift Scale, Shift Norm Norm

Scale, Shift Scale, Shift Norm Norm

Dual-Stream FFN Block

[Figure 96]

[Figure 97]

(b) Architecture of Dual-Stream Attention and FFN Block

Dual-Stream Attention Block

Noisy Image Latent Multimodal Context Embeddings

Linear Linear Patching Norm

[Figure 98]

[Figure 99]

[Figure 100]

- 0, 0, -1 0, 0, 0 0, 0, 1 0, 0, 2

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

0,-1,-1 0, -1, 0 0, -1, 1 0, -1, 2

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

0, 2, -1 0, 2, 0 0, 2, 1 0, 2, 2

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

- 0, 1, -1 0, 1, 0 0, 1, 1 0, 1, 2 7, 2, -1 7, 2, 2

7, -1, -1 7, -1, 2

3, 3, 3 4, 4, 4 5, 5, 5 6, 6, 6 8, 8, 8 9, 9, 9 ……

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Noisy Image Latent

Multimodal Context Embeddings

Unified MSRoPE

Timestep Embedding

(a) Architecture of Visual Generation Head

(c) Unified MSRoPE

- Figure 4: Architecture of the Visual Generation Head. (a) Overview of the head with dual-stream MMDiT blocks. (b) Detailed structure of the Dual-Stream Attention Block and Dual-Stream FFN Block. (c) Illustration of the Unified MSRoPE (Multi-Scale Rotary Positional Embeddings) applied to VAE image latents and multimodal context embeddings.

understanding primarily relies on semantically informative features, whereas image generation additionally requires representations that preserve reconstructable low-level visual details, much as humans can perceive complex scenes they cannot necessarily draw.

- • Semantic Input for Context Understanding: For the understanding tasks (context), we only utilize high-level semantic features extracted directly from raw pixels via a pre-trained ViT. This helps preserve the semantic fidelity required for complex reasoning.
- • Compressed Output for Generation Target: For the generation tasks (target), we employ a separate Variational Autoencoder (VAE) trained specifically for image reconstruction. This VAE compresses images into a latent space suitable for synthesis.

By decoupling these representations, we not only avoid the “optimization trade-off”, where a single encoder struggles to balance the high-level abstraction needed for understanding with the low-level pixel details needed for generation, but also avoid the increased computation cost and infrastructure complexity incurred by inputting the generation target into the context backbone. This allows us to leverage the strongest available pre-trained encoders for understanding without compromising the generative quality.

###### 3.1.2. Visual Generation Head

Based on the proposed principles, in this section, we further detail the implementation of our custom-developed visual generation head, as shown in Figure 4.

Dual Projectors for Context and Target Input. The feature distributions of the multimodal hidden states (context) and the VAE’s image latents (target) exhibit substantially different feature distributions. To bridge this heterogeneity, we employ independent linear projectors to map them into the conditioning space of the visual generation module. Crucially, we observe that the multimodal context embeddings tend to exhibit larger magnitudes and more pronounced outliers than the VAE latents. To reduce this scale mismatch and improve training stability, we introduce an additional normalization layer on the VLM branch before projection, explicitly normalizing the variance of the context features to unity.

Dual-Stream MMDiT Block with Gated Attention. We adopt a fully Dual-Stream architecture to account for

the distinct statistical properties of multimodal context and generative targets. While both streams interact via joint self-attention to capture token-level dependencies, they utilize disentangled parameters for the QKVO projections and Feed-Forward Networks (FFNs). Furthermore, to enhance non-linearity and mitigate “attentionsink” phenomena in high-resolution, long-context scenarios, we integrate an element-wise Gating Mechanism [106] into the attention block. Formally, the modulated output O′ of the attention layer is:

###### O′ = O ⊙ 𝜎(XW𝑔) (1)

where 𝜎 denotes the sigmoid function, X, O denote the input and output of the attention layer, and W𝑔 denotes the learnable gating projection matrix, which is also disentangled for each stream. To the best of our knowledge, this is the first integration of a gating mechanism within the MMDiT architecture, offering improved expressivity with minimal parameter overhead.

Unified MSRoPE with Resolution Interpolation. We employ Multimodal Scalable RoPE (MSRoPE) [136] to encode positional information, ensuring rigorous preservation of spatial structures.

- • Unified 3D Encoding: Unlike previous works [136] that often treat visual tokens in the multimodal context as flattened 1D sequences, we apply unified 3D positional embeddings (temporal, height, width) to both the generative targets and the visual tokens within the context. This alignment significantly benefits tasks requiring precise spatial reasoning, such as image editing.
- • Positional Interpolation: To facilitate resolution scaling, we address the “tiling artifact” observed when directly extrapolating position indices during high-resolution fine-tuning. Instead, we adopt a Resolution Interpolation strategy. We define the position embedding range based on the maximum target resolution (e.g., 1024px). During the initial low-resolution pre-training (e.g., 512px), rather than using a smaller index range, we utilize the full range but increase the stride between adjacent tokens. This ensures that the model learns a consistent global spatial representation from the outset, minimizing the domain gap when scaling to higher resolutions.

- 3.2. Training Strategy

- 3.2.1. Training Objective

To endow the UMM with the capability to process and generate multimodal content, we formulate a joint optimization objective. Given a multimodal context sequence c, the model is trained to simultaneously predict discrete text tokens x and continuous image latent representations z.

Autoregressive Text Generation. For the textual component, we treat text generation as a sequence modeling problem over a discrete vocabulary. We employ the standard Next-Token Prediction (NTP) objective, minimizing the negative log-likelihood of the target tokens conditioned on the context and preceding tokens:

∑︁𝑇

1 𝑇

log 𝑝𝜃(𝑥𝑡 | 𝑥<𝑡,c) (2)

ℒNTP = −

𝑡=1

where 𝑥𝑡 denotes the 𝑡-th token in the text sequence of length 𝑇, 𝑥<𝑡 represents the preceding tokens, and 𝜃 parameterizes the unified model. This objective ensures the model retains the reasoning and instructionfollowing capabilities inherent in the MLLM backbone.

Flow Matching for Image Generation. For the visual component, we adopt the Flow Matching framework with velocity parameterization to model the continuous distribution of image latents. Unlike diffusion models that predict noise 𝜖, we regress the velocity vector field 𝑣𝜃 that transports the probability density from a Gaussian noise distribution to the data distribution. We assume the standard linear interpolation path between the noise z0 ∼ 𝒩(0,I) and the ground-truth image latent z1, following the common formulation used in Flow Matching and transport paths inspired by Optimal Transport. The intermediate state at time 𝑡 ∈ [0,1] is defined as z𝑡 = 𝑡z1 + (1 − 𝑡)z0. The objective is to minimize the mean squared error between the predicted velocity and the target drift:

[︀‖𝑣𝜃(z𝑡,𝑡,c) − (z1 − z0)‖2]︀

ℒFM = E𝑡∼𝒰[0,1],z

0∼𝒩(0,I),z1∼𝑝data

(3)

where 𝑣𝜃(z𝑡,𝑡,c) is the model output predicting the velocity vector at time 𝑡 conditioned on context c, and (z1 − z0) represents the ground-truth instantaneous velocity along the linear trajectory. Unified Training Objective. The final training objective is a weighted sum of the discrete and continuous losses:

ℒTotal = 𝛼 · ℒNTP + 𝛽 · ℒFM (4)

where 𝛼 and 𝛽 are scalar hyperparameters balancing the two modalities. In practice, we dynamically adjust these coefficients across different training stages (e.g., pre-training vs. supervised fine-tuning) to prioritize specific capabilities, such as visual fidelity or reasoning capabilities.

###### 3.2.2. Training Pipeline

To maximize training efficiency while adhering to the architectural principles outlined in Section 3.1.1, we initialize our UMM from a pre-trained MLLM solely optimized for understanding tasks. Since the base MLLM lacks visual generative capabilities, we design a three-stage curriculum that progressively unlocks visual synthesis skills before unifying them with semantic reasoning.

- Stage1: Generation Head Pre-training. In the initial phase, we focus on grounding the newly initialized visual generation head to the MLLM’s latent space. We freeze the MLLM to preserve its semantic representations and train only the generation head and projectors. Following previous work [142], we skip the 256px pre-training and utilize a fixed resolution of 512px to accelerate early convergence. Unlike prior approaches [124, 136] that rely solely on text-to-image data for initialization, we incorporate a mixture of text-to-image generation and image editing datasets from the outset. This multi-task strategy forces the generation head to attend to both textual instructions and visual context tokens simultaneously, establishing a robust foundation for multimodal condition alignment.
- Stage2: Any-resolution Continued Pre-training. Building upon the stable initialization, we advance to variable-resolution training to handle diverse aspect ratios and enhance visual fidelity. The MLLM backbone remains frozen. We perform a secondary filtration of the training corpus, retaining only high-aesthetic samples and discarding those with extreme aspect ratios that might induce training instability. The resolution of the generated images is controlled within the range of 512 to 1024 pixels, while the aspect ratio is maintained within the range of 0.5 to 2.0. For image editing tasks, maintaining pixel-level alignment between the input condition and the output is critical. To this end, we further explicitly inject the VAE latent of the condition image into the visual generation head to achieve better pixel-level consistency.
- Stage3: Unified Supervised Finetuning. The final stage aims to further synergize the visual generative capabilities acquired in previous stages with the reasoning capabilities of pre-trained MLLM. Therefore, the entire model is unfrozen, including the MLLM backbone, to enable end-to-end optimization. The training corpus is further filtered based on stricter criteria, along with additional CoT reasoning data, detailed in Section 4.7. By mixing these CoT data with image generation and editing data, the model is allowed to plan its generation via textual reasoning before executing it in the visual domain.

#### 4. Data Construction

To equip InternVL-U with multimodal generation and editing capabilities on top of InternVL’s strong multimodal understanding foundation, we construct a large-scale training corpus by combining publicly available datasets with synthetic data pipelines tailored to diverse generation and editing tasks.

###### 4.1. Open-source Data Collection

As shown in Table 1, we have collected a large number of high-quality image generation and image editing datasets as our initial data pool. To better address long-tail cases, particularly in the domains of human portraits and text-rich imagery, we further augment this open-source corpus with specialized datasets.

###### Table 1: Overview of collected open-source datasets.

Task Datasets Text-To-Image (T2I) LAION [109], BLIP-3o [14], ShareGPT-4o-Image [15], OSP [75], Echo-4o-

Image [152], OpenGPT-4o [20], FaceCaption [26], Flux-Reason-6M [37], HumanCaption [27], POSTER-TEXT [39], AutoPoster [78], CTW [155]

Image Editing (IT2I) InstructPix2Pix [6], AnyEdit [154], PIPE [133], ImgEdit [153], SEED-DataEdit [41], OmniEdit [134], UltraEdit [160], HQEdit [55], ShareGPT-4o-Image [15], OpenGPT-4o [20], X2Edit [91], X2I2 [138], UniWorld perception [76], NHR-Edit [62], GPT-hqedit [131], GPT-omniedit [131], GPT-ultraedit [131], Nano-consistent-150k [152], Pico Banana [103]

###### 4.2. General Data Preprocessing and Synthesis

Texd-to-Image Data

[Figure 118]

[Figure 119]

[Figure 120]

A bright and airy living room exudes summer charm with its light beige walls, large windows,... and welcoming summer living room design.

The image shows a young man with a well-groomed appearance... The overall look is clean, professional, and sporty.

A green hanging planter with trailing ivy.

[Figure 121]

[Figure 122]

A small, weathered wooden stool sits beside a rustic clay bowl filled with dried lavender sprigs, both placed on a woven mat spread across a stone floor. To the right of the bowl... that carries the scent of earth and rain.

这是一张以高原牧场风光为背景的乳制 品宣传海报，整体色调明亮清新，充满 自然与健康的视觉感受...带来愉悦生活 的品牌理念。

Image Editing Data

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Add a red and white striped awning above the motorcycle, extending from the top of the garage doors.

Transform the stone wall's material to polished glass.

Replace the rocky mountainous landscape in the background with a lush green forest.

- Figure 5: Examples of general data synthesized by our pipeline. The synthesized data features varied textual annotations and covers diverse visual domains, including portraits, posters, natural scenes, etc.

We first design the general data preprocessing and synthesis pipelines for image generation and image editing. Representative examples are shown in Figure 5.

- 4.2.1. General Preprocessing

Given the collected open-source datasets, we first apply a general preprocessing pipeline for filtering, expansion, and deduplication, as illustrated in Figure 6. Specifically, we begin with a rigorous multi-dimensional filtering protocol to exclude low-quality samples. This process includes filtering based on aesthetic scores, resolution thresholds, safety standards (e.g., NSFW detection), and watermark identification, yielding a pristine subset of high-quality samples.

To enrich domain coverage and intra-domain diversity, a dual-branch expansion workflow is then implemented: retrieval-based and synthesis-based expansion. In the retrieval-based branch, we capture long-tail concepts and real-world variations absent in standard datasets by utilizing both image and text queries in largescale search engines. Complementing this, the synthesis-based branch densifies the image manifold by creating realistic variants of existing samples. Finally, to ensure a non-redundant source pool, we compute perceptual

###### General Preprocessing

Text-to-Image Pipeline

[Figure 129]

Image Synthesis

[Figure 130]

Open-source Dataset

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Dense Captioning

Human-centric Captioning

Concise Captioning

Variant Captioning

[Figure 135]

Multi-step Filtering

[Figure 136]

Source Data Pool

[Figure 137]

Filtered Data

Image Editing Pipeline

[Figure 138]

[Figure 139]

Retrieval-based Expansion

Task Router

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Global-level

Object-level

Attribute-level

Compositional

[Figure 144]

Synthesis-based Expansion

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Instruction Generation

Instruction Generation

Instruction Generation

Instruction Generation

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Image Edit

Image Edit

Expanded Data

Image Edit

Image Edit

[Figure 154]

[Figure 155]

Task-aware Verification Instruction Following Editing Consistency Generation Quality

Deduplication

[Figure 156]

[Figure 157]

[Figure 158]

- Figure 6: Overview of our general data synthesis pipeline. First, the preprocessing stage applies filtering, expansion, and deduplication to construct a high-quality source pool. Building upon this, two parallel branches are deployed to generate text-to-image pairs and instruction-guided editing data, respectively.

hashes (p-hash) [157] for all collected samples, removing near-duplicates to maximize data efficiency.

###### 4.2.2. Text-to-Image Data

As shown in Figure 6, different captioning strategies are adopted to enhance the diversity and quality of the captions of the collected data. Specifically, a pre-trained MLLM (i.e., Qwen2.5-VL [5]) is adopted as the image captioning agent and is prompted to generate captions at varying levels of granularity, including:

- • Concise Captioning: Short, lucid descriptions of core visual elements, facilitating strong concept binding and prompt adherence.
- • Dense Captioning: Hierarchical descriptions to ensure the model attends to holistic scene structure and details, covering foreground subjects, background environments, and stylistic features.
- • Human-Centric Captioning: Designed specifically for portraiture, focusing on fine-grained attributes such as facial features, expressions, poses, and clothing details.

To address the long-tail scarcity of images containing legible text, we further introduce a targeted data expansion strategy tailored for text-rich data. Given specific textual content and background environments, variant captions are first generated by the image captioner. Then an image generation expert (e.g., Qwen-Image [136]) is employed to synthesize corresponding images. This process significantly densifies the distribution of high-quality text-image pairs. To equip the model with robust bilingual capabilities, a comprehensive English-to-Chinese translation pipeline is also conducted across the dataset, ensuring the model can interpret and generate content with equal proficiency in both languages.

###### 4.2.3. Image Editing Data

For image editing data synthesis, a more sophisticated pipeline is further introduced, as illustrated in Figure 6. We first categorize the editing tasks into four primary classes: (1) Global-level: Modifying overall style, tone, or background while preserving the structural layout. (2) Object-level: precise addition, removal, or replacement of objects with boundary handling. (3) Attribute-level: Adjusting fine-grained properties such as color, material, size, and count. (4) Compositional: Executing compound instructions that require coherent, multi-step operations.

Given the source data pool, we then adopt a multi-agent framework to generate instruction-edit pairs. An MLLM-based router is responsible for determining the specific editing tasks for each source image. Based on

the routing results, images are dispatched to a modular pool of specialized agents. To ensure high-quality synthesis, we stratify these agents into two functional categories: Instruction Generation and Image Editing.

- • Instruction Generation: Acknowledging that distinct editing tasks require capturing different levels of semantic granularity, we implement these agents using Qwen2.5-VL-72B [5], conditioned with taskspecific prompts to produce precise and context-aware instructions.
- • Image Editing: Since existing open-source models [65, 83, 136] exhibit varying strengths due in part to differences in their training data distributions, we integrate a heterogeneous ensemble of editing models. By assigning different tasks to the model best suited for that specific granularity, we ensure optimal visual fidelity across the entire dataset.

To ensure the reliability of the constructed corpus, an automated, task-aware verification module is also introduced, as shown in Figure 6. We define a tripartite evaluation protocol assessed by MLLMs: (1) Instruction Following: Verifying if the edited image faithfully executes the prompt. (2) Editing Consistency: Evaluating semantic and structural coherence with the source image. (3) Generation Quality: Assessing visual fidelity, realism, and artifact suppression. Each image-editing pair is scored against task-specific prompts. Only samples exceeding predefined thresholds across all three dimensions are retained. This adaptive filtering effectively removes misaligned or low-quality examples, resulting in a robust dataset characterized by strong instruction alignment and visual coherence.

###### 4.3. Text-centric Data Synthesis

Semantically Related Text Rendering on a Natural Image Text Rendering on a Pure Color Background

[Figure 159]

[Figure 160]

The background is kraft, and the black text is '嗯， 这道题说的是，至少存在一个x≥0，使得不等式x² ≤4 |2x + m|成立，求实数m的取值范围。选项是A到D四个。

[Figure 161]

[Figure 162]

在图像的左下角，有一张覆盖在背 景上的白色纸张，上面用蓝色的中 文文字书写着：“简·贝塔的蓝色 儿童肖像”。文字的方向是正常的 水平方向，清晰地呈现在纸张上。 背景环境是一幅水彩风格的画作， 整体色调以蓝色和白色为主，营造 出一种梦幻而宁静的氛围。画面中， 一个穿着深蓝色围巾的小孩占据了 画面的中心位置，小孩的头发是棕 色的，眼神清澈，脸颊微微泛红， 显得非常可爱。背景中的蓝色和白 色水彩渲染出一种朦胧的云雾效果， 仿佛小孩置身于一片梦幻的天空之 中，增添了一种诗意和温暖的感觉。 整体画面通过柔和的色彩和细腻的 笔触，展现了儿童的纯真与美好， 同时也突出了“简·贝塔的蓝色儿 童肖像”这一主题。

那我得仔细分析一下怎么解这个题了。首先，题目是 说存在至少一个非负的x，使得x² ≤4 - |2x + m|成立。 也就是说，我们需要找到所有m的值，使得这个不等式 在x≥0的范围内至少有一个解。反过来，如果对于某个 m，不管取什么样的x≥0，这个不等式都不成立，那么 这个m就不在所求的范围内。所以，我们需要确定这样 的m，使得存在x≥0满足x² + |2x +m| ≤4。''.

在图像的底部，有一条横向铺开的牛皮纸条，上面用黑色的中 文文字书写着：“南方泰国考帕通岛的猴子观河”。文字的方 向是正常的水平方向，清晰地呈现在牛皮纸条上。背景环境是 一片宁静的自然景观，一条清澈的河流蜿蜒穿过郁郁葱葱的热 带丛林，两岸被茂密的绿植和高大的树木环绕，树木的枝叶在 阳光下显得生机勃勃。河面上停泊着几艘传统的小船，船身颜 色鲜艳，船桅高耸，显得古朴而富有特色。远处的山丘在薄雾 中若隐若现，增添了一丝神秘的氛围。整体画面充满了热带雨 林的自然气息，给人一种宁静而祥和的感觉。

On a black background, displaying white text '### Algorithmic Explanation of Insertion in a HashTable#### Primary Objectives:1. Data Storage: The primary goal is to store key-valuepairs in the hash table.2. Efficiency: Ensure that the insertion operation is asfast as possible, ideally with a time complexity of O(1).3. Consistency: Maintain the integrity of the hash tableby avoiding collisions or handling them effectively.#### Procedural Steps:1. Hash Function Calculation: - The hash function takes the key as input and returnsan index (hash value) where the key-value pair should bestored in the array. - The hash function should be designed to distribute thekeys uniformly across the array to minimize collisions.

Text Editing within an Image

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Replace the text “TMP 548F” with “ASTON MARTIN”.

Replace the text “MARKET PLAZA” with “STOCK YARDS”.

Replace the text “Happy” with “Special”.

Replace the text “06:03” with “12:45”.

- Figure 7: Three types of text-centric data synthesized by our pipeline. The first type overlays semantically relevant text onto natural images using a masked background image. The second type renders text on solid-color backgrounds, focusing on clean and aesthetically pleasing layouts. The third type performs text editing within existing images, such as modifying text on license plates, mobile interfaces, signboards, and similar surfaces.

Textual elements in visual media are highly semantically dense and crucial for communication, making accurate text rendering and fine-grained editing essential for real-world applications. Despite recent advances in text-related tasks [22, 66, 104, 122, 136], general multimodal models still struggle with text-centric generation and editing, often exhibiting spelling errors, poor support for non-alphabetic languages, layout misalignment, and unintended visual artifacts. Considering its significance, we further introduce a text-centric data synthesis pipeline aimed at further improving text rendering and editing capabilities. As shown in Figure 7, we cover three representative data types: (1) semantically related text rendering on natural images, (2) text rendering on pure-color backgrounds, and (3) text editing within images, which together strengthen a model’s integrated ability to understand, generate, and modify textual content in visual contexts.

###### Mask Image Selections

###### Ready Data

Rendered Result

###### Text Color Selections

###### Image Caption

Rendered Image

###### Attribute

Image

###### Kraft Paper

Wood Material

White Paper

[Figure 171]

"In the top-center of the image, there is a rectangular piece of kraft paper with black text writen in a horizontal orientation. The text reads: '汤姆·霍珀，电影艺术与科学学院

[Figure 172]

[Figure 173]

mask image font type text color layout

[Figure 174]

Adaptive

[Figure 175]

[Figure 176]

[Figure 177]

Kaiti.ttf

Rendering Module

Font Type Selections

[Figure 178]

[Figure 179]

[Figure 180]

的治理奖.' The background features a formal seting with a large, golden Oscar statuete prominently ."

[Figure 181]

Ch En

[Figure 182]

[Figure 183]

[Figure 184]

Kaiti.ttf、FangSong.ttf、 Simhei.ttf、weiruanyahei.ttf 、AdobeSongStd.otf

Arial_Unicode.ttf、times.ttf happyschool-v4pzw.ttf、 sattriahandsome-m2lga.ttf

Image

Initial Setting Intermediate Result

[Figure 185]

[Figure 186]

[Figure 187]

Adaptive Layout selections

[Figure 188]

[Figure 189]

푚   , ℎ

Wrap Text Function

h 퐹표  푚   =

[Figure 190]

퐹표  푚   = 푚  ( , ℎ) ∗ 퐹표       = 퐹표  푚  

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Adaptive

w

Rendering Module

Rendered Image

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Condition

✅ Searching for Font ..

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

    ℎ 푒  +2 * padding ≦ w ℎ푒  ℎ  푒  +2 * padding ≦ h

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

퐹표       퐹표       − 1

- Figure 8: Text rendering data construction pipeline. For synthetic text rendering, we prepare mask images, font colors, font styles, and adaptive layout options. During the rendering process, these attributes are randomly sampled, and the text is rendered with typography that adapts to its length.

###### 4.3.1. Text-to-Image Data

To equip the model with a stronger visual text rendering capability, we design a comprehensive automatic textrendering data synthesis pipeline that can produce high-quality and diverse textual data. This pipeline supports both semantically related text rendering on natural images and text rendering on pure-color backgrounds, covering different languages (i.e., both Chinese and English).

Specifically, as shown in Figure 8, for semantically related text rendering on natural images, we directly render the original paired caption annotation onto the source image. To increase the text diversity, we also render pure text data on a randomly picked pure-color background. During rendering, the mask of the rendering region, the text color and font type, as well as an adaptive layout design scheme, are also taken into account. The text size can be further adaptively adjusted with line breaks inserted automatically to ensure that the text is arranged aesthetically on the image. Finally, an image captioner (i.e., Qwen2.5-VL-72B [5]) is used to re-caption the rendered images.

###### 4.3.2. Image Editing Data

For synthesizing text-aware image editing, we design a three-stage pipeline, as shown in Figure 9. This pipeline can generate high-quality paired samples covering text edits in both natural and virtual scenes. First, we employ OCR tools (i.e., PaddleOCR [22]) to detect text regions and extract the recognized text, confidence scores, and bounding polygons. Second, an MLLM-based instruction agent (i.e., Qwen2.5-VL-72B [5]) is prompted to filter candidate regions, verify textual relevance and visual consistency, and produce semantically explicit editing instructions. Finally, these selected texts, polygons, and editing instructions are then taken as inputs for a text-editing agent (e.g., Flux-Text [66]) to perform precise, context-aware text editing. This workflow yields high-quality, semantically aligned image-text editing pairs suitable for text-aware applications.

- 4.4. Science-centric Data Synthesis

Scientific images are of great importance to both the science academia and the AI industry. While the understanding of science-centric images has attracted wide attention [52, 146, 156, 162], their generation is still in a relatively preliminary stage [12, 132]. To enhance the model’s capability for generating images with strict structuredness, high semantic coherence, rigid knowledge dependence, and deep reasoning, we design various rigorous pipelines to curate science-centric data for text-to-image and image editing. These data cover various disciplines such as physics, chemistry, biology, and computer science. Specifically, text-to-image data are mainly curated from existing understanding datasets and web images. For editing data, which are more difficult to collect as they require input-output image pairs with diverse and meaningful editing prompts, we design multiple data engines to synthesize editing data for physics and computer science. The pipeline of

###### Stage 1 Stage 2 Stage 3

Identify candidate text regions for editing Select the editing area and generate edited text Obtain the condition and generate the edited image

[Figure 215]

[Figure 216]

Original Image

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Text2 score2 polys2

SelectedPolys Edit Text

Selected Text

[Figure 221]

[Figure 222]

Text1 Score1 Polys1

Image Caption Instruction

[Figure 223]

PaddlePaddleOCR

[Figure 224]

[Figure 225]

Masked Image

Text Rendering

[Figure 226]

Editing Prompt

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Text1 Score1 Polys1

Text2 Score2 Polys2

Text3 Score3 Polys3

Glyph Image

[Figure 231]

###### Qwen2.5VL-72B

[Figure 232]

[Figure 233]

Original Image

[Figure 234]

FLUX-TEXT

Filter Flow

[Figure 235]

[Figure 236]

[Figure 237]

Selected Text

Selected Polys Edit Text

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Text2 Score2 Polys2

Text1 Score1 Polys1

Edited Image

- Figure 9: Text editing data construction pipeline. First, we use an OCR tool to extract candidate text regions for editing. Second, we generate the editing instructions. Third, we use a generative model to produce the edited ground truth. Through these three steps, we synthesize high-quality text-editing triplets.

[Figure 242]

Open-source Datasets

[Figure 243]

Qwen3-VL-8B Rating & Filtering

[Figure 244]

PaddleOCR Image Extraction

[Figure 245]

Documents

General Science T2I Physics

[Figure 246]

Web Images

[Figure 247]

Filtered Images

[Figure 248]

Qwen3-VL-32B Annotating

[Figure 249]

Generation Prompts

Computer Science

[Figure 250]

Rendering & Validation

[Figure 251]

Image Pairs Editing Prompts

[Figure 252]

[Figure 253]

Matplotlib Graphviz

[Figure 254]

[Figure 255]

Tree Tasks

[Figure 256]

Graph Tasks

[Figure 257]

FSM Tasks

[Figure 258]

Qwen3-VL-8B Filtering

[Figure 259]

Raw Images

[Figure 260]

Text Context

[Figure 261]

Filtered Raw Images

[Figure 262]

Gemini-3-Flash SVG Conversion & Annotating

[Figure 263]

Gemini-3-Flash Filtering & Prompt Simplifying

[Figure 264]

SVG-Rendered Image Pairs

[Figure 265]

CoT Editing Prompts

[Figure 266]

Filtered Raw Images

[Figure 267]

Text Context

[Figure 268]

Image Pairs

[Figure 269]

Simplified Editing Prompts

[Figure 270]

Traditional Solver Solving & Qwen3-Max Rewriting

[Figure 271]

CoT Editing Prompts

Step1: Raw Image Acquisition Step2: SVG-Based Data Synthesis

- Figure 10: Science data generation pipeline. For general science T2I, we collect web images and open-source datasets and design automatic filtering and annotating with open-source models. For physics, we obtain images from documents by PaddleOCR, and propose an SVG-based pipeline for high-quality and affordable image-pair generation. For computer science, we define tasks and render images with Python libraries.

science data generation is shown in Figure 10, and examples of science data are given in Figure 11.

- 4.4.1. General Science Generation Data

For text-to-image generation, the images are collected from various sources, including open-source multimodal scientific understanding datasets ([48, 54, 59, 60, 84, 126, 156, 159, 164]), textbooks, and competitions (e.g., IPHO, Gaokao, Kaoyan). After that, a multi-stage filtering strategy is adopted to obtain high-quality data. In the first stage, images with resolutions below 256p are removed, and the resulting images are deduplicated based on p-hash. Images identical to the benchmarks in Section 5 are also removed to avoid data contamination. In the second stage, we employ an open-source MLLM (i.e., Qwen3-VL-8B [4]) to rate and filter the images from multiple dimensions, including image types, subjects, text length, image complexity, and subject knowledge density. Please refer to Section B.1 for more filtering details. An image captioner (i.e., Qwen3-VL-32B [4]) is then prompted to synthesize the corresponding caption.

General Science T2I

llustrate a diagram of a cell membrane showing both primary and secondary active transport mechanisms. For primary active transport, depict an antiport protein pump using ATP to move three sodium ions (Na+) out of the cell and two potassium ions (K+) into the cell. For secondary active transport, depict a symport protein that utilizes the established sodium gradient to co-transport one glucose molecule into the cell along with a sodium ion. Label the lipid bilayer, the specific ions/molecules, the transport proteins, and the energy source.

Draw a diagram illustrating Earth as a giant magnet, showing the magnetic field lines extending from the magnetic south pole to the magnetic north pole. Clearly label the North Magnetic Pole, South Magnetic Pole, North Geographic Pole, and South Geographic Pole. Ensure the magnetic field lines are depicted as loops that emerge from the magnetic south pole and enter the magnetic north pole, illustrating the orientation of Earth's magnetic field relative to its geographic poles.

[Figure 272]

[Figure 273]

Physics

Transform the DC parallel circuit diagram into an AC series-parallel impedance network where the DC voltage source U is converted to an AC source, an inductor L1 is added in series to the branch containing bulb L, and a capacitor C1 is added in series to the branch containing resistor R, while ensuring all current labels and component identifiers are accurately maintained.

Modify the schematic to show the block of mass $m$ at position D, midway between the wall at A and the spring's (constant $k$) natural length at point O, featuring a compressed spring, the position label D, and a complete free-body diagram with vectors for gravity ($G$), normal force ($F_N$), spring force ($F_s$), and kinetic friction ($f$) that accurately reflect the physics of the block moving leftward.

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Computer Science

Perform extraction from the Max-Heap: extract the root and display the resulting heap structure after all sifting operations are complete.

Identify and highlight the closed loop (cycle) in the graph. Please mark all nodes and edges on the cycle in red. The output image should clearly show which nodes and edges form this cycle.

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

- Figure 11: Examples of science-centric data. General science T2I features dense and detailed instructions and requires depicting multidisciplinary concepts. Physics and computer science focus on image editing with disciplinary reasoning.

###### 4.4.2. SVG-based Physics Editing Data

Although abundant physics images are abundant on the Internet, it is difficult to obtain paired physics images for editing from open-source datasets or synthesizing with existing libraries and software. Using an existing proprietary image editing model (e.g., Nano Banana Pro [29]) to generate a paired image from a given image can lead to extremely high costs and inconsistent quality. Inspired by recent advances in Scalable Vector Graphics (SVG) understanding and generation [105, 125], as well as the strong capability of state-of-the-art multimodal models [46, 97], we design an SVG-based pipeline to synthesize physics image pairs, as shown in Figure 10. SVG enables high-quality and resolution-independent generation of target images in a cost-effective manner, by manipulating structured SVG code rather than editing raster images directly.

In the first stage, we collect heterogeneous-source documents of physics textbooks and exams, and use PaddleOCR [23] to extract images and their textual context. We then use Qwen3-VL-8B [4] to filter out images of undesirable domains, e.g., illustrative figures in textbooks, real-world photographs in exam questions, mathematical formulas, and complex images that are difficult to convert into SVG format. These images cover sub-disciplines such as electromagnetism, mechanics, optics, circuits, thermology, and atomic physics.

Subsequently, an image and its context are fed into Gemini-3-Flash [46] to generate the corresponding SVG code of the original image. The model also decides whether the original image should serve as the input or output image, and generates an editing prompt and the SVG code of the other image. The two SVG codes are rendered into a pair of images. The original prompts generated by Gemini-3-Flash [46] are detailed and explicit and can serve as a step-by-step reasoning prompt. To evaluate subject-specific knowledge and competence, we perform key information extraction and summarization on these prompts to obtain a simplified, implicit prompt as the final data.

Finally, another filtering is performed on each triplet of <input_image, prompt, output_image> to examine

the correctness and quality of the data. We use Gemini-3-Flash [46] to filter and remove images containing rich text formatting and images with physical structural errors or annotation errors. Compared to directly using Nano Banana Pro [29] to generate the output image, this SVG-based approach greatly reduces the cost from $0.16 to $0.03 per sample.

###### 4.4.3. Computer Science Editing Data

We construct a data engine for computer science editing data based on several Python libraries. These data focus on operations and algorithms on data structures like trees, graphs and finite state machines (FSM). Specifically, the tasks include:

- • Tree: Topology editing and node manipulation, traversal visualization, binary search tree (BST) operations, heap operations with dual views, Huffman coding trees, and lowest common ancestor (LCA) and path highlighting.
- • Graph: K-hop neighborhood identification, degree identification, cycle detection, bipartite graph coloring, shortest path reasoning, and directed graph reachability.
- • FSM: String tracing, state role identification, and transition logic completion.

Definitions of these tasks are in Section B.3.

To ensure high-quality data generation, we first select task-specific rendering engines based on task complexity to balance efficiency and visual fidelity. For structurally simpler tasks such as trees and graphs, we employ matplotlib to expedite the generation process. In contrast, for state machines requiring dense information display, we utilize Graphviz (with circo layouts) to ensure optimal topological clarity. To maintain spatial consistency across image pairs (e.g., the position of a node in the images should not change), we define fixed anchor points for nodes. This constraint ensures that invariant components remain visually consistent.

We then perform validation to detect and eliminate samples with node or edge overlaps by calculating the distance between two nodes or between a node and an edge. Any generated instance failing this occlusion check is discarded, ensuring the overall structural integrity of the dataset. Finally, we generate CoT editing prompts by solving with a traditional solver and rewriting, detailed in Section 4.7.

###### 4.5. Spatial-centric Data Synthesis

To enhance the spatial understanding capability of our model, we synthesize spatial-centric data from three distinct domains: solid geometry, multi-view CAD, and general spatial rotation. Example images are shown in Figure 12.

- 4.5.1. Solid Geometry Editing Data We use GeoGebra [43] and matplotlib to render solid geometry editing data. The tasks include:

- • Solid of Revolution: Draw a solid formed by rotating a given shape around a given axis.
- • Plane Symmetry: Draw the solid that is centrally symmetric to a given solid with respect to a given plane.
- • Point Symmetry: Draw the solid that is centrally symmetric to a given solid with respect to a given point.
- • Solid Translation: Translate a given solid by a given translation vector.
- • Solid Projection: Draw the orthogonal projection of a given solid onto the 𝑥𝑜𝑦-plane.

Implementations of these tasks are in Section B.3.

###### 4.5.2. Multi-view CAD Editing Data

To further enhance the model’s spatial understanding capabilities, we constructed corresponding three-view editing data based on an open-source CAD dataset. The model was required to predict other corresponding views based on the input isometric view and editing instructions. Specifically, we used the open-source ABC

Solid Geometry

Draw the solid formed by rotating the figure one full turn around the axis shown. Draw the reflection of the solid across the plane shown in the diagram, and keep the original solid in the result.

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Multi-view CAD

Generate the corresponding front view based on the isometric view of the object. Generate the corresponding top view based on the isometric view of the object.

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Spatial Rotation

Rotate the blue bicycle with the basket by 150 degrees clockwise. Generate an image where the street lamp post is rotated 330 degrees counterclockwise, and the background should also look good.

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

- Figure 12: Examples of spatial-centric data. We consider three spatial-centric scenarios: solid geometry (e.g., revolution, symmetry), multi-view CAD (three-view drawing), and spatial rotation of 3D objects.

dataset [61] and the OCC Python library to render the CAD files in the dataset into images of corresponding perspectives, including isometric views, front views, side views, and top views. We also increased the diversity of data distribution by randomly setting the colors and materials of objects during the rendering process. Examples of the final constructed editing dataset are shown in Figure 11.

###### 4.5.3. Spatial Rotation Editing Data

We leverage Objaverse [30], an open-source 3D model dataset, to build high-quality object rotation data. We begin by rotating the objects at uniform angles and rendering. While covering a wide range of daily objects, it typically lacks rich environmental contexts. As illustrated in Figure 13, to endow these objects with suitable backgrounds, we initially generate four distinct reference images featuring backgrounds for each object. These candidates then undergo a rigorous multi-step filtering mechanism:

- • Bounding Box Detection: By detecting the object’s bounding box, we assess deformation, retaining only instances where the aspect ratio falls within the range of [0.9,1.1]. Additionally, to ensure the feasibility of the Background-First strategy, we exclude candidates where the object would exceed the image boundaries during rotation.
- • Object Consistency: We evaluate the consistency of essential visual attributes, with a particular emphasis on orientation, and retain only those instances that demonstrate high consistency to the original input.
- • Generation Quality: We assess the plausibility of object-background integration, stylistic consistency with the scene, and overall visual quality, and select the highest-scoring candidate for each object.

Subsequently, we design two different strategies for the final data synthesis:

• Object-First strategy, which prioritizes the plausible integration of the object within its context. We first generate context-rich instructions based on the selected reference image. The instructions together with the reference image are then used by Qwen-Image [136] to synthesize the edited image. To ensure the

###### Stage 1

###### Stage 2

Reference Images & Metadata Preparation

Object-first Strategy Background-first Strategy

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Reference Images

Brief Caption

Reference Images

Brief Caption

[Figure 298]

Open-source 3D Models

[Figure 299]

Rendering

[Figure 300]

[Figure 301]

FLUX.1 Kontext Object Removal

Qwen2.5-VL-7B Prompt Generation

[Figure 302]

[Figure 303]

1 2

[Figure 304]

[Figure 305]

Raw Images

[Figure 306]

[Figure 307]

3 4

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Background Prompts

Raw Images

... a grassy field with rolling hills, a few trees, and...

Background Images

[Figure 312]

Qwen-Image-Edit Background Addition

[Figure 313]

Qwen2.5-VL-7B Removal Validation

[Figure 314]

[Figure 315]

[Figure 316]

Qwen-Image-Edit Background Addition

1 2

[Figure 317]

Reference Image Candidates

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

- a b

c d

[Figure 322]

- b d

a b

[Figure 323]

[Figure 324]

3 4

[Figure 325]

[Figure 326]

Filtered Background Images

Raw Images

[Figure 327]

Image Pool

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

c d

[Figure 332]

Qwen2.5-VL-7B Filtering & Labeling

[Figure 333]

Bounding Box

[Figure 334]

GPT-5.1 Pairing & Filtering

[Figure 335]

Background Paste & Pairing

[Figure 336]

[Figure 337]

[Figure 338]

Reference Images

Bounding Box

Brief Caption

90 degrees clockwise... 30 degrees clockwise...

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

3

C D

blue bicycle with the basket

[Figure 344]

[Figure 345]

Filtered Editing Data

Filtered Editing Data

- Figure 13: Overview of our spatial rotation editing data synthesis pipeline. Stage 1 prepares a pool of filtered reference images. Subsequently, Stage 2 generates the final editing pairs via either the Object-First strategy for object-context integration, or the Background-First strategy for strict background preservation.

high quality of these editing pairs, we utilize GPT-5.1 to further filter results based on two dimensions: the accuracy of the object’s new orientation and the consistency of the background.

• Background-First strategy, which prioritizes maximal background consistency across edits. Leveraging the strong consistency capabilities of Flux.1 Kontext [65], we perform object removal to obtain a clean background image. The success of this removal operation is verified using Qwen2.5-VL [5]. Next, utilizing the bounding boxes detected before, we paste objects with alternative orientations onto this clean background, thereby yielding editing pairs with perfectly consistent background scenery.

Meme Data for Image Generation

Meme Data for Image Editing

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

生成一张表情 包，表现起床 痛苦又无奈的 日常状态。

生 成 一 张 因 害 怕 而 恳 求 对 方 “ 正 常 点 ” 的 搞笑表情包。

生成一张表达 强烈佩服或惊 叹的表情包。

生 成 一 张 表 情 包 ， 表 现 不 想 听 别 人 说 话 的 烦躁抗拒感。

生成一张撒娇 式可爱表情包， 询问“我可爱 吗”。

[Figure 360]

生成一张表情 包，表现面对 压力时的无奈 与自嘲。

[Figure 361]

生成一张可爱 想要拥抱的表 情包。

Generate a meme showing sarcastic self-satisfaction after quitting a job to avoid being fired.

将这张图变成一张表达质 疑和不满情绪的梗图，带 点讽刺意味。

把这张图变成一张表现无奈又 带点搞笑的梗图，头顶西瓜的 滑稽感要突出。

让这张图传达出一种兴奋和欢呼的情绪， 仿佛角色在庆祝周末的到来。

将这张图变成一张表现深 夜孤独、情绪低落的网抑 云梗图。

- Figure 14: Meme data example for image generation and editing. The meme data here captures elements of everyday human humor and expressive nuances commonly found in daily life.

###### 4.6. Humor-centric Data Synthesis

Memes have become an important form of expression on the Internet, conveying humor, satire, and cultural information through the combination of visual and textual elements. To enhance the meme generation capability of our model, we synthesize two types of training data tailored to memes: text-to-image generation data and image-to-image editing data. The examples are shown in Figure 14. The generation data targets scenarios where users provide short and abstract intent prompts, and the system produces an image that conveys concrete or implicit meanings, such as humor, sarcasm, surprise, helplessness, or other affective intentions. The editing data targets scenarios where a user supplies an input image and modifies it via natural-language instructions, while preserving the original content and style as much as possible. A common form is adding concise, punchy text (e.g., subtitles, labels, or dialogue) to create contrast or irony.

Stage 2

Meme Image Stage 1

Stage 3

Model-Style Prompt Generation

User-Style Prompt Generation

Text Presence Detection

[Figure 362]

[Figure 363]

[Figure 364]

original image

四个天线宝宝角色在开满野花的草 地上兴奋地跳跃，它们张大嘴巴、 眼睛圆睁、双臂高举，表情充满喜 悦和期待。背景是晴朗的蓝天和绿 意盎然的田野，营造出轻松愉快的 氛围。图片下方配有文字“休息日 要来了”，精准传达了人们对周末 即将到来的激动心情，幽默地捕捉 了工作日结束时对放松时光的强烈 渴望。

with text set

[Figure 365]

[Figure 366]

[Figure 367]

四 个 天 线 宝 宝 角 色 在....图片下方配有文 字“休息日要来了”，

[Figure 368]

[Figure 369]

...

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

Captioner

[Figure 376]

[Figure 377]

精准传达了....。

[Figure 378]

[Figure 379]

[Figure 380]

CoT Captioner

VLM

without text

[Figure 381]

[Figure 382]

...

[Figure 383]

Captioner

[Figure 384]

生成一张图片表达我对休息日马上要到来的喜悦 之情。

Stage 4

Meme Data for Training

Stage 5

Image Generation

Paired Image Construction

Editing Instruction Generation

[Figure 385]

User-style prompt: 生成一张委屈撒娇的表情包，表现被冤枉或被责备时的无辜抗议。

Model-style Prompt: 一个紫色的卡通角色，外形类似宝可梦中的耿鬼，双眼含泪， 泪珠呈弯曲的浅蓝色弧线从眼角滑落，表情委屈，嘴巴微张成“O”形，流露出无 辜和被冤枉的神情。背景是模糊的黄绿色调，带有轻微的噪点和飞溅的墨点效果， 营造出一种情绪化的氛围。图片下方配有文字“干嘛凶我”，直接点明角色正在质 问对方为何责备自己，整体传达出一种委屈、撒娇并寻求安慰的幽默情绪。

image pair

with text set

image pair

image pair

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

...

[Figure 390]

Remove Text Agent

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

Editing Instruction Generater

[Figure 396]

[Figure 397]

Image Editing

image pair

without text

[Figure 398]

[Figure 399]

User-style prompt：

User-style prompt：把这张图变成一张表现焦急等待心情的梗图。 Model-style Prompt: 原图中一只站立的仓鼠，表情呆萌，双手自 然垂下。为了表现焦急等待的心情，可以在图像上方添加文字 “我..?”，在下方添加文字“在线等，挺急的”，仅添加文字， 其他视觉内容保持不变，让仓鼠看起来像是在表达一种急切又 带点无奈的等待状态，使得这张图变成一张梗图。

[Figure 400]

Add Text Agent

[Figure 401]

[Figure 402]

[Figure 403]

让这张图表达我马 上要过周末的喜悦 的情绪。

...

[Figure 404]

Spoken Language Template

[Figure 405]

Figure 15: Meme data synthesis pipeline. It comprises five stages, leveraging chain-of-thought reasoning to process internet memes and synthesize high-quality training data for meme generation and editing.

To build these synthetic datasets, we first crawl and collect a large number of meme images from the Internet and open-source datasets, and then apply an automated pipeline (as illustrated in Figure 15) to generate high-quality paired images and aligned instructions:

- • Stage 1 (Text Presence Detection): Perform per-sample text detection to determine whether the image contains visible text using VLM.
- • Stage 2 (Model-Augmented Instruction Generation): Use image captioner VLM to generate modelaugmented instructions based on the input image and its original caption, providing fine-grained descriptions of visual details and semantic focus.
- • Stage 3 (User-Style Prompt Generation): Use CoT captioner to generate shorter, user-style prompts that are closer to real user queries.
- • Stage 4 (Paired Image Construction): Branch by text presence: if the image contains text, invoke “Text Removing Agent” to erase on-image text and obtain a text-free version; otherwise, use “Text Adding Agent” to add the instructed content or perform instruction-driven modifications on the original image, thereby constructing image pairs.
- • Stage 5 (Editing Instruction Generation): Use editing instruction generator to generate editing instructions that are strictly aligned with the paired images produced in Stage 3, yielding training triplets of (source image, target image, editing instruction).

###### 4.7. Reasoning-centric Data Synthesis

In real-world applications, user-provided generation and editing instructions are often brief, high-level, and abstract, especially in complex or domain-specific scenarios. While such concise prompts are natural and convenient for users, they frequently omit critical details, such as explicit attribute specifications, spatial relationships, executable editing steps, and domain-specific constraints, making it difficult for models to accurately interpret intent and consistently produce reliable results.

To address this challenge, we propose reasoning-centric data synthesis, which introduces an explicit reasoning module as an interpreter between raw user instructions and the final multimodal supervision signals. Given a short and abstract instruction, this module automatically derives a more structured, concrete, and actionable specification, including refined objectives, decomposed sub-tasks, verifiable constraints, and ordered editing operations. Such structured interpretations provide clearer learning signals during training. The synthesized data jointly organizes abstract instructions, their explicated reasoning traces, and corresponding execution targets, enabling models to better follow underspecified instructions, improve robustness in challenging domains, and enhance controllability in both generation and editing tasks. In particular, we focus on four core application settings: (1) General Images, (2) Knowledge-infused Images, (3) Meme Images, and (4) Science Images.

General T2I CoT Data Example

Enhanced T2I Instruction After CoT Reasoning

Original T2I Instruction

Original Image

"A cat is leaping over a stack of books and competing in a creative obstacle course."

"A tabby cat with a striped brown and black coat, white paws and chest, and green eyes is captured mid-leap, extending its body horizontally as it jumps over a stack of eight books. The books are stacked vertically with varied spine colors including beige, red, black, and blue, and they are positioned on a grassy field. The books support a horizontal bar that is white with red and yellow segments, forming part of an obstacle course. The bar is attached to a white frame with blue and yellow panels. In the background, a blurred green landscape suggests an outdoor setting, and the lighting is bright and even, casting soft shadows on the ground."

[Figure 406]

[Figure 407]

not detailed

[Figure 408]

Detailed

General Editing CoT Data Example

Edited Image

Original Editing Instruction

CoT Reasoning

[Figure 409]

"The image features a person wearing a Evaluation result sleeveless, ribbed knit top with vertical stripes in rich colors such as deep red, royal blue, burnt orange, and teal. To extract this top, I must first precisely identify its boundaries, including the V-neckline, the armholes, and the hemline where it meets the white pants. This edit makes sense for purposes like product isolation, fashion design, or creating a digital asset for e-commerce. The extraction should preserve the texture and color fidelity of the knit fabric while cleanly separating it from the background and the wearer’s body. The expected outcome is a standalone image of the top with a transparent or white background, accurately representing its shape, pattern, and material details."

[Figure 410]

"Extract the colorful striped top worn by the person in the image."

Original Image

Brief reasoning: The extracted object includes the necklace, and the shape and print of the top are distorted.\nObject Identity: 3 \nMask Precision: 3 \nVisual Quality: 3"

Enhanced Editing Instruction

[Figure 411]

"Extract the colorful striped sleeveless knit top worn by the person in the image. Ensure the extraction includes the full garment from the V-neckline down to the waistband, preserving the ribbed texture and vibrant vertical stripes in red, blue, orange, and teal. Remove all surrounding elements, including the person’s arms, torso below the top, and background, to isolate the top cleanly. Maintain the original lighting and color accuracy for a realistic and high-quality result."

[Figure 412]

[Figure 413]

"Brief reasoning: Entire top extracted with full silhouette, colors and stripes intact. Clean mask, pure white background, sharp details.\nObject Identity: 5 \nMask Precision: 5 \nVisual Quality: 5"

- Figure 16: Examples of CoT reasoning and enhancement for general image generation and editing. The original prompt is enriched through our chain-of-thought reasoning, adding finer-grained details that enable the model to perform generation and editing with greater accuracy and improved fidelity.

General Images. In general image generation and editing, user-provided instructions are often short and underspecified. Ambiguous descriptions of scene composition, target regions, or modification attributes can cause models to misinterpret user intent, leading to low-fidelity generation or inaccurate fine-grained edits. To mitigate this issue, reasoning-based prompt rewriting and refinement are crucial. Accordingly, we apply CoT augmentation to all instructions in our constructed dataset. For generation tasks, we expand abstract concepts into detailed visual descriptions of objects, backgrounds, and styles; for editing tasks, we enrich instructions with clearer references for localizing target regions, explicit attributes to modify or preserve, and necessary visual-consistency constraints. This process produces more grounded and informative prompts without altering the original intent, thereby improving the learnability of supervision signals and the controllability of outcomes. As shown in Figure 16, using CoT-enhanced instructions at inference time yields noticeably more accurate generation and editing results that are better aligned with user requirements.

Knowledge-infused Images. Knowledge-infused image generation and editing require models to not only accurately comprehend user instructions, which are often concise and abstract, but also to possess deep capabilities for knowledge-based reasoning and analysis. The core of this task lies in deconstructing the implicit background knowledge behind the instructions and translating it into concrete visual concepts or detailed scene descriptions, thereby effectively bridging the informational gap in user inputs. Subsequently, this enriched and explicit semantic information is fed into the generative model, guiding it to produce high-quality images that align with the user’s underlying intent while maintaining logical consistency. As illustrated in Figure 17, we employ open-source models to perform detailed-level reasoning enhancement on the instructions within our curated knowledge-based image generation and editing dataset. For instance, the model can automatically establish the cultural association between the “Mid-Autumn Festival” and its traditional food, “mooncakes,” or link the state of a “banana after one week” with the visual feature of “brown spots”, thus achieving a concrete representation of abstract concepts.

Meme Images. In Section 4.6, we have introduced the tasks of meme generation and editing. In meme scenarios, user instructions are often brief and highly abstract (e.g., “generate an image that expresses my happiness”),

Knowledge-infused T2I CoT Data Example

Enhanced T2I Instruction After CoT Reasoning Generated Image

Original T2I Instruction

[Figure 414]

"A close-up of traditional mooncakes, round pastries with a golden-brown crust, filled with smooth lotus seed paste or red bean paste. The surface shows subtle texture details, with a warm lighting that enhances the richness of the filling and the crispness of the dough. The background is simple and neutral, focusing

"Traditional food of the MidAutumn Festival."

[Figure 415]

[Figure 416]

Abstract

attention on the traditional food." Detailed

[Figure 417]

Knowledge-infused Editing CoT Data Example

Enhanced Editing Instruction After CoT Reasoning Edited Image

Original Editing Instruction

Original Image

" Add realistic brown spots and streaks across the yellow peel, darken and slightly shrivel the stem. Maintain the original lighting, perspective, and position on the wooden cutting board, ensuring the changes blend naturally with the scene without any

[Figure 418]

[Figure 419]

"Draw what it will look like after being kept in a daily environment for a week."

[Figure 420]

[Figure 421]

[Figure 422]

Abstract

signs of mold or decay." Detailed

[Figure 423]

- Figure 17: Examples of CoT reasoning and enhancement for knowledge-infused T2I and editing. After CoT reasoning, abstract knowledge concepts are concretized, enabling more precise generation and editing.

and typically lack descriptions of concrete visual details. To address this, we introduce reasoning-based prompt enhancement, converting short instructions into explicit and controllable specifications. Without altering the original intent, we enrich the prompts with (1) concrete scene and visual element details, (2) the intended emotional stance and expression mechanism (i.e., the humor structure), and (3) template and typography constraints (e.g., caption placement postion). For editing tasks, we further specify localized operations and clearly state which regions and attributes should remain unchanged. The resulting pipeline is illustrated in Figure 15. This reasoning-centric instruction formulation provides stronger and more stable supervision signals, improving both controllability and alignment with user intent for meme generation and editing.

Science Images. Science is a typical domain that requires detailed reasoning to generate a semantically correct image. For science T2I tasks, especially those involving structured symbols such as mathematical formulas and chemical molecular notations, we use LLMs first parses the scientific concepts to produce intermediate reasoning steps, including conceptual analysis and layout planning, and then generates the final image description. For science image editing, the model interprets the source image and instructions, converting abstract editing intents into executable text directives that follow correct scientific logic For physics editing data, we use the prompt before the summarization stage as the reasoning-informed prompt, which includes explicit and detailed step-by-step instructions. For computer science editing data, we first define the data structures for problem-solving based on the input and output images, and employ traditional algorithms to solve the problem. Then we record each step of the solution process in detail and map them into 13 predefined templates prepared by human. Qwen3-Max [4] is utilized to rewrite these templates to increase the diversity of CoT data.

[Figure 424]

[Figure 425]

"Illustrate the process of photosynthesis in a plant, showing the role of light energy from the sun, the intake of carbon dioxide, and the release of oxygen. Label the green plant cells where food is made, the glucose (sugars) produced, and the water absorbed from the soil. Ensure the diagram clearly depicts the flow of these elements and the transformation within the plant."

Original T2I Instruction

Enhanced T2I Instruction After CoT Reasoning

"Illustrate the process of photosynthesis in a stylized green plant with a single large leaf, a stem, and roots embedded in brown soil. Show a bright yellow sun in the upper right corner. Include a yellow arrow labeled 'Light energy' pointing from the sun to the leaf, a brown arrow labeled 'Carbon dioxide in' entering the leaf, and a blue arrow labeled 'Water' moving upward from the soil into the roots. Show a green arrow labeled 'Oxygen out' exiting the leaf, and a red arrow labeled 'Glucose (sugars)' traveling downward from the leaf along the stem. Within the leaf, depict small green dots labeled 'Green plant cells where food is made'. Use black text for all labels on a plain white background, with clear, simple arrows indicating the direction of material flow. The entire scene should be evenly lit with no shadows, emphasizing clarity for educational purposes."

Science Image

[Figure 426]

[Figure 427]

clear

abstract

Science Orig Image

Science Edit Image

[Figure 428]

[Figure 429]

Given the leaf nodes with frequencies A: 12, B: 76, C: 23, D: 84, E: 88, Construct the complete Huffman Tree by adding internal nodes and edges.

Original Editing Instruction

abstract

Enhanced Editing Instruction After CoT Reasoning

- 1. Create a new internal node with weight 35. Connect leaf node 'A' (weight 12) as its left child (edge '0') and leaf node 'C' (weight

23) as its right child (edge '1').

- 2. Create a new internal node with weight 111. Connect internal node (weight 35) as its left child (edge '0') and leaf node 'B' (weight 76) as its right child (edge '1').
- 3. Create a new internal node with weight 172. Connect leaf node 'D' (weight 84) as its left child (edge '0') and leaf node 'E' (weight

88) as its right child (edge '1').

- 4. Create a new internal node with weight 283. Connect internal node (weight 111) as its left child (edge '0') and internal node (weight 172) as its right child (edge '1'). clear

[Figure 430]

Science T2I CoT Data Example Science Editing CoT Data Example

- Figure 18: Examples of CoT reasoning and enhancement for Science T2I and Editing. Through CoT reasoning, scientific knowledge is injected into the generation process in a more explicit and detailed manner (e.g., how elements should be depicted and how operations should be executed).

#### 5. Experiments

###### 5.1. Experimental Setups

InternVL-U is implemented upon InternVL3.5-2B [129], using its weight to initialize the visual understanding encoder and the multimodal context backbone. The text tokenizer and conversation format are also the same. For image generation, we use the same VAE as Qwen-Image [136]. The visual generation head is randomly initialized, containing 1.7B parameters. The total number of parameters of InternVL-U is 4B. Detailed configuration is shown in Table 2. Following previous works [6], we also adopt the classifier-free guidance (CFG) for both image and text conditions. During training, for text-to-image generation data, the condition is dropped with a 10% probability; for image editing data, the multimodal condition (including text and image) is dropped with a 5% probability, and there is also a 5% probability of dropping only the text while retaining the image input. During inference, Flow-DPM-Solver [142] is adopted with 20 inference steps. The CFG scales for dropping the entire condition and text condition only (for editing tasks) are set to 3.5 and 1.5, respectively.

- Figure 19 illustrates the system and user prompts used under different settings, of which the embeddings are truncated when fed into the visual generation head. The detailed training settings of each stage are shown in Table 3. We use VLMEvalkit [34] to evaluate the multimodal understanding and reasoning benchmarks. For the evaluation on image generation and editing tasks, we use our self-developed evaluation toolkit named GenEditEvalKit†, which is also open-sourced.

- Table 2: Detailed configuration of InternVL-U architecture.

|Configuration|Visual Und. Encoder|Context Backbone|Visual Gen. Head<br><br>|
|---|---|---|---|
|# Layers # Num Heads (Q / KV) Head Size Intermediate Size Patch / Scale Factor|24 16 / 16 64 4,096 14|28 16 / 8 128 6,144 -<br><br>|20 12 / 12 128 6,144 2|
|# Parameters|0.3B|2B|1.7B|

- Table 3: Hyperparameters for different training stages.

Hyperparameters Stage 1 Stage 2 Stage 3 Trainable modules

|Backbone Visual gen. head<br><br>|× × ✓ ✓ ✓ ✓|
|---|---|
|Learning rate LR scheduler Weight decay Gradient norm clip|3e-04 1e-04 1e-05<br><br>Constant Cosine Cosine 0.0 0.0 0.01 0.2 0.2 1<br><br>|
|Optimizer Warm-up steps|AdamW (𝛽1 = 0.9,𝛽2 = 0.999,𝜖 = 10−8) 1000<br><br>|
|Training steps Batch size Gen. resolution (min, max) Und. resolution (min, max) Diffusion timestep shift<br><br>|250,000 60,000 20,000 2048 1024 1024 (512, 512) (512, 1024) (512, 1024)<br><br>(448, 448) 3.0|
|Data / tasks Data ratio Loss weight (NTP:VP)|T2I + IT2I T2I + IT2I T2I + IT2I + Und<br><br>4:1 3:4 1:1:2 0:1 0:1 1:20|

† https://github.com/open-compass/GenEditEvalKit

System Prompt for T2I task

<|im_start|>system

你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的 多模态大语言模型。请通过详细描述图像中物体和背景的颜色、形状、大小、纹理、数量、文字内容以 及空间位置关系等来对图像进行全面描述: <|im_end|>

System Prompt for IT2I task

<|im_start|>system

你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的 多模态大语言模型。请描述输入图像的关键特征（颜色、形状、大小、纹理、物体、背景等），然后解 释用户的文本指令应该如何改变或修改图像，从而生成一个满足用户要求的新图像，并适当保持与原始 输入的一致性: <|im_end|>

User Prompt for dropping image and text conditions

<|im_start|>user Here is a random image <img_uncond>:<|im_end|>

User Prompt for dropping text conditions only

<|im_start|>user Generate an image based on reference images.<|im_end|>

- Figure 19: System prompts and user prompts adopted during training for different tasks, where <img_uncond> is a learnable special token for image generation.

- 5.2. Multimodal Understanding and Reasoning

To assess multimodal understanding and reasoning capabilities, we evaluate InternVL-U on 7 widely recognized MLLM benchmarks, including MME-P [38], SEED [67], ChartQA [94], OCRBench [86], MMMU [156], MathVerse [158], and LogicVista [141]. As shown in Table 4, InternVL-U demonstrates robust performance on multimodal understanding and reasoning benchmarks, significantly surpassing comparable-sized UMMs such as Janus-Pro [16] and Ovis-U1 [124] across key metrics like MME-P [38] (1607.5) and OCRBench [86](83.9). Remarkably, despite its compact architecture (2B+1.7B), it delivers reasoning capabilities comparable to the significantly larger BAGEL [31](7B+7B) model, particularly on MMMU [156](54.7 vs. 55.3). These results indicate that our unified training strategy effectively retains the strong visuallanguage comprehension of understanding-only baselines, showing minimal degradation while achieving a superior balance between understanding and generation.

- Table 4: Comparisons between InternVL-U and baseline models on multimodal understanding and reasoning benchmarks. * indicates the results based on our evaluation scripts. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

|Model|#Params|Understanding MME-P [38] SEED [67] ChartQA [94] OCRBench [86]<br><br>|Reasoning MMMU [156] MathVerse [158] LogicVista [141]|
|---|---|---|---|
| | | | |

MLLMs w/o Generator LLaVA-1.5V [80] 7B 1510.7 65.8 17.8 31.8 35.7 7.6 – Qwen2.5-VL [5] 3B 1574.9 73.7 84.0 79.7 53.1 31.2 40.3 InternVL3.5 [129] 2B 1552.1 75.3 80.7 83.6 59.0 53.4 47.7 UMMs w/ Generator JanusFlow [92] 1.3B 1333.1 70.5 42.4 53.2 29.3 – – Janus-Pro [16] 1.5B 1444.0 68.3 23.4 48.7 36.3 – – Show-o2 [144] 1.5B 1450.9 65.6 40.0 24.5 37.1 – – TUNA [87] 1.5B 1461.5 69.3 82.1 71.9 39.1 – – MetaQuery-L [102] 3B+1B 1574.3 73.8 – – 53.1 – – Ovis-U1 [124] 2.4B+1.2B 1508.0* 75.5* 76.4* 88.3 51.1 30.6* 32.4* Emu3 [24] 8B – 68.2 – 68.7 31.6 – – BAGEL [31] 7B+7B 1687.0 78.5 78.5 73.3 55.3 48.1* 44.3* InternVL-U 2B+1.7B 1607.5 75.2 76.6 83.9 54.7 45.6 40.3

###### 5.3. Text-to-Image Generation

To comprehensively evaluate the text-to-image generation capabilities, we adopt GenEval [44], DPG-Bench [53], TIIF [135], and OneIG [11] for general assessment, LongText [42] and CVTG-2k [33] for text rendering quality, and WISE [96] and GenExam [132] for knowledge-intensive generation.

###### 5.3.1. General Image Generation

GenEval. GenEval [44] is an object-focused framework for evaluating compositional image properties such as object co-occurrence, position, count, and color. In Table 5, InternVL-U achieves highest overall score (0.85) among existing unified models such as BAGEL [31] with only half or even less number of parameters. It also surpasses most of the specialized generation models.

- Table 5: Evaluation of general text-to-image generation ability on GenEval [44]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

Model #Params Single Object Two Object Counting Colors Position Color Attribution Overall Generation Models

FLUX.1 [dev] [63] 12B 0.98 0.81 0.74 0.79 0.22 0.45 0.66 SD3-Medium [35] 2B 0.99 0.94 0.72 0.89 0.33 0.60 0.74 Seedream 3.0 [40] - 0.99 0.96 0.91 0.93 0.47 0.80 0.84 GPT Image 1 [High] [99] - 0.99 0.92 0.85 0.92 0.75 0.61 0.84 Z-Image [8] 6B 1.00 0.94 0.78 0.93 0.62 0.77 0.84 Qwen-Image 20B 0.99 0.92 0.89 0.88 0.76 0.77 0.87 Unified Models

Show-o2 [144] 7B 1.00 0.87 0.58 0.92 0.52 0.62 0.76 Janus-Pro [16] 7B 0.99 0.89 0.59 0.90 0.79 0.66 0.80 UniWorld-V1 [76] 7B+13B 0.99 0.93 0.79 0.89 0.49 0.70 0.80 OmniGen2 [138] 3B+4B 1.00 0.95 0.64 0.88 0.55 0.76 0.80 BAGEL [31] 7B+7B 0.99 0.94 0.81 0.88 0.64 0.63 0.82 InternVL-U 2B+1.7B 0.99 0.94 0.74 0.91 0.77 0.74 0.85

DPG-Bench. Table 6 shows the results on DPG-Bench [53], which provides dense prompts that describe multiple objects to assess the intricate semantic alignment capabilities of text-to-image models. In Table 6, our model exhibits stronger performance than other unified models, especially on Global and Entity dimensions.

- Table 6: Evaluation of general text-to-image generation ability on DPG-Bench [53]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

Model #Params Global Entity Attribute Relation Other Overall Generation Models

FLUX.1 [dev] [63] 12B 82.10 89.50 88.80 91.10 89.40 84.00 SD3-Medium [35] 2B 87.90 91.01 88.83 80.70 88.68 84.08 GPT Image 1 [High] [99] - 88.89 88.94 89.84 92.63 90.96 85.15 Nano Banana Pro [29] - 91.00 92.85 91.56 92.39 89.93 87.16 Z-Image [8] 6B 93.39 91.22 93.16 92.22 91.52 88.14 Qwen-Image [136] 20B 91.32 91.56 92.02 94.31 92.73 88.32 Seedream 4.5 [7] - 89.24 94.30 92.14 92.23 93.83 88.63 Unified Models

UniWorld-V1 [76] 7B+13B 83.64 88.39 88.44 89.27 87.22 81.38 OmniGen2 [138] 3B+4B 88.81 88.83 90.18 89.37 90.27 83.57 Ovis-U1 [124] 2.4B+1.2B 82.37 90.08 88.68 93.35 85.20 83.72 Janus-Pro [16] 7B 86.90 88.90 89.40 89.32 89.48 84.19 BAGEL [31] 7B+7B 88.94 90.37 91.29 90.82 88.67 85.07 InternVL-U 2B+1.7B 90.39 90.78 90.68 90.29 88.77 85.18

TIIF. TIIF [135] aims to systematically evaluate the ability to follow intricate instructions. As shown in Tables 7 and 8, our InternVL-U achieves strong performance among unified models, especially on advanced instruction following. There are still notable differences between unified models and generation models, indicating potential future improvements in instruction following.

- Table 7: Evaluation of general text-to-image generation ability on TIIF [135] (Short Prompts). Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. Abbreviations: Attr=Attribute, Rel=Relation, Reas=Reasoning, A+R=Attribute+Relation, A+Re=Attribute+Reasoning, R+Re=Relation+Reasoning, RW=Real World.

|Model|#Params|Basic Following Avg Attr Rel Reas<br><br>|Advanced Following Avg A+R A+Re R+Re Style Text|Designer RW|Overall|
|---|---|---|---|---|---|
| | | | | | |

Generation Models

FLUX.1 [dev] [63] 12B 83.1 87.1 87.3 75.0 65.8 67.1 73.8 69.1 66.7 43.8 70.7 71.1 Z-Image [8] 6B 78.4 79.5 80.5 75.1 72.9 72.9 67.0 73.9 90.0 94.8 88.1 80.2 Seedream 3.0 [40] - 87.1 90.5 89.6 80.9 79.2 79.8 77.2 75.6 100.0 97.2 83.2 86.0 Qwen-Image [136] 20B 86.2 90.5 88.2 79.8 79.3 79.2 78.9 75.6 100.0 92.8 90.3 86.1 Unified Models

Show-o [143] 1.3B 73.1 74.8 78.8 65.6 53.7 61.0 68.6 66.5 63.3 3.8 55.0 59.7 Janus-Pro [16] 7B 79.3 79.3 78.3 80.3 59.7 66.1 70.5 67.2 60.0 28.8 65.8 65.5 Ovis-U1 [124] 2.4B+1.2B 77.8 83.5 80.1 69.9 67.4 71.8 66.8 69.0 83.3 8.1 67.2 66.7 BAGEL [31] 7B+7B 81.8 82.5 83.0 79.9 70.2 74.4 67.4 72.0 86.7 29.4 68.3 71.5 Lumina-DiMOO [145] 8B 84.9 87.0 87.6 79.8 72.8 74.8 76.8 69.8 70.0 51.1 75.0 74.7 InternVL-U 2B+1.7B 82.3 86.0 84.1 76.7 73.5 75.3 70.4 75.5 93.3 47.5 65.3 74.9

- Table 8: Evaluation of general text-to-image generation ability on TIIF [135] (Long Prompts). Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. Abbreviations: Attr=Attribute, Rel=Relation, Reas=Reasoning, A+R=Attribute+Relation, A+Re=Attribute+Reasoning, R+Re=Relation+Reasoning, RW=Real World.

|Model|#Params|Basic Following Avg Attr Rel Reas<br><br>|Advanced Following Avg A+R A+Re R+Re Style Text|Designer RW|Overall|
|---|---|---|---|---|---|
| | | | | | |

Generation Models

FLUX.1 [dev] [63] 12B 78.7 83.2 80.4 72.4 68.5 73.7 73.3 71.6 66.7 52.8 71.5 71.8 Z-Image [8] 6B 82.8 86.5 79.9 81.9 77.0 77.6 73.8 75.6 93.3 93.2 85.5 83.0 Seedream 3.0 [40] - 84.9 90.1 85.9 78.9 80.6 81.8 78.9 78.6 93.3 87.8 83.6 84.3 Qwen-Image [136] 20B 87.2 91.5 90.8 79.4 80.9 79.8 81.7 78.6 100.0 89.1 91.4 86.8 Unified Models

Show-o [143] 1.3B 75.8 79.8 78.3 69.3 50.4 56.8 69.0 56.2 66.7 2.8 50.9 58.9 Janus-Pro [16] 7B 78.3 82.3 73.3 79.1 58.8 56.2 70.8 60.0 70.0 33.8 60.3 65.0 Ovis-U1 [124] 2.4B+1.2B 79.4 81.5 81.4 75.2 67.8 68.3 73.8 65.9 86.7 12.7 68.7 68.2 Lumina-DiMOO [145] 8B 78.0 81.5 79.8 72.6 68.5 74.1 69.1 66.4 63.3 40.7 72.0 68.8 BAGEL [31] 7B+7B 80.1 83.5 79.9 76.8 72.2 75.0 70.1 74.9 83.3 33.9 67.9 71.7 InternVL-U 2B+1.7B 81.5 81.5 82.2 80.9 72.7 76.2 67.6 75.8 83.3 50.7 66.8 73.9

OneIG-Bench. In Tables 9 and 10, we assess our InternVL-U on OneIG-Bench [11], which is designed for finegrained evaluation across subject-element alignment, text rendering precision, reasoning-generated content, stylization, and diversity. InternVL-U shows highest overall score among open-source unified models with a small parameter scale, demonstrating its fine-grained alignment capability with multi-lingual robustness.

- Table 9: Evaluation of general text-to-image generation ability on OneIG-EN [11]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

Model #Params Alignment Text Reasoning Style Diversity Overall Generation Models

SDXL [35] 2.6B 0.69 0.03 0.24 0.33 0.30 0.32 FLUX.1 [dev] [63] 12B 0.79 0.52 0.25 0.37 0.24 0.43 Qwen-Image [136] 20B 0.88 0.89 0.31 0.42 0.18 0.54 Z-Image [8] 6B 0.88 0.99 0.28 0.39 0.19 0.55 Seedream 4.5 [7] - 0.89 1.00 0.35 0.43 0.21 0.58 Nano Banana Pro [29] - 0.89 0.94 0.33 0.48 0.25 0.58 Unified Models

Janus-Pro [16] 7B 0.55 0.00 0.14 0.28 0.37 0.27 Show-o2 [144] 7B 0.82 0.00 0.23 0.32 0.18 0.31 Ovis-U1 [124] 2.4B+1.2B 0.81 0.03 0.22 0.45 0.18 0.34 BAGEL [31] 7B+7B 0.77 0.24 0.17 0.37 0.25 0.36 Lumina-DiMOO [145] 8B 0.82 0.55 0.28 0.40 0.23 0.46 OmniGen2 [138] 3B+4B 0.80 0.68 0.27 0.38 0.24 0.47 InternVL-U 2B+1.7B 0.82 0.74 0.27 0.40 0.25 0.50

- Table 10: Evaluation of general text-to-image generation ability on OneIG-ZH [11]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

Model #Params Alignment Text Reasoning Style Diversity Overall Generation Models

Qwen-Image [136] 20B 0.83 0.96 0.27 0.41 0.21 0.53 Z-Image [8] 6B 0.79 0.99 0.27 0.39 0.24 0.54 Seedream 4.5 [7] - 0.83 0.99 0.30 0.43 0.21 0.55 Nano Banana Pro [29] - 0.84 0.98 0.31 0.46 0.24 0.57 Unified Models

Janus-Pro [16] 7B 0.32 0.15 0.10 0.26 0.36 0.24 BLIP-3o [13] 8B 0.61 0.09 0.21 0.37 0.23 0.30 Lumina-DiMOO [145] 8B 0.68 0.15 0.23 0.37 0.24 0.33 Ovis-U1 [124] 2.4B+1.2B 0.72 0.15 0.21 0.43 0.20 0.34 BAGEL [31] 7B+7B 0.67 0.37 0.19 0.36 0.27 0.37 InternVL-U 2B+1.7B 0.75 0.90 0.23 0.37 0.26 0.50

Qualitative Results. To further illustrate the practical strengths beyond quantitative metrics, we provide additional qualitative comparisons. As shown in Figure 20, InternVL-U demonstrates exceptional visual fidelity in general image generation, particularly in rendering intricate textures and nuanced lighting effects, while precisely capturing the intent of each instruction.

###### InternVL-U BAGEL Ovis-U1 Qwen-Image Nano Banana Pro

Prompt: A minimalist metal wolf head emblem in side profile, mouth open with sharp teeth, rendered in cold-toned metallic textures. The background is a dark black gradient with subtle lighting focused on the emblem, creating a grim and solemn atmosphere. Epic fantasy house sigil or wallpaper style.

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Prompt: 一张吉卜力风格的插画。画面描绘了一个充满魔法的森林茶会，主角是一只毛茸茸的树懒，正捧着一块比它脸 还大的圆形手工饼干准备咬一口。饼干表面烤得焦黄酥脆，中间的巧克力酱像熔岩一样诱人地流淌出来。阳光透过树叶 的缝隙洒下斑驳的光影，桌上摆满了野果和冒着热气的茶杯。画风色彩饱和度高，充满手绘的温暖质感，让人感到极致 的宁静与治愈。

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Prompt: A Victorian noblewoman radiating poise and sophistication, adorned with an exquisite wide-brimmed hat decorated with silk ribbons, ostrich feathers, and a cluster of deep velvet roses in shades of burgundy and dusty pink. Her ensemble features a highcollared gown made of rich emerald green taffeta, embellished with intricate black lace overlay on the bodice and cuffs, evoking a sense of historical grandeur. A sheer, dotted tulle veil is pinned loosely to the hat, casting a soft shadow over her eyes while adding an air of mystery. Her hair is swept up into an intricate chignon, revealing a pair of antique gold drop earrings set with garnets and a delicate gold chain necklace featuring a garnet pendant resting gracefully against her collarbone. The makeup is subtle yet defined, with a porcelain complexion and a soft rose lip. The composition is set against a muted, charcoal-grey background, drawing full attention to the luxurious textures of the fabric and the harmonious color palette, capturing the essence of 19th-century elegance in a photorealistic and painterly artistic style.

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

- Figure 20: Visualizations of general image generation. Compared to other open-source models, InternVL-U demonstrates exceptional fidelity in rendering intricate textures and nuanced lighting effects, capturing the precise intent of each instruction.

###### 5.3.2. Text-centric Image Generation

CVTG-2k. In Table 11, we provide results on CVTG-2k [33], which is specialized designed for complex visual text generation. InternVL-U achieves state-of-the-art performance among unified models, with an average word accuracy of 0.623.

- Table 11: Evaluation of text-centric text-to-image generation ability on CVTG-2k [33]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

|Model|#Params|NED|CLIPScore|Word Accuracy 2 regions 3 regions 4 regions 5 regions average<br><br>|
|---|---|---|---|---|
| | | | | |

Generation Models

FLUX.1 [dev] [63] 12B 0.688 0.740 0.609 0.553 0.466 0.432 0.497 Nano Banana Pro [29] - 0.875 0.737 0.737 0.775 0.786 0.793 0.779 Qwen-Image [136] 20B 0.912 0.802 0.837 0.836 0.831 0.816 0.829 Z-Image [8] 6B 0.937 0.797 0.901 0.872 0.865 0.851 0.867 Seedream 4.5 [7] - 0.948 0.807 0.878 0.895 0.908 0.901 0.899 Unified Models

Ovis-U1 [124] 2.4B+1.2B 0.477 0.725 0.133 0.109 0.091 0.065 0.093 BAGEL [31] 7B+7B 0.657 0.779 0.498 0.391 0.332 0.291 0.356 Lumina-DiMOO [145] 8B 0.805 0.831 0.723 0.646 0.571 0.505 0.590 InternVL-U 2B+1.7B 0.804 0.816 0.729 0.660 0.618 0.549 0.623

LongText-Bench. LongText-Bench [42] assesses the ability to render longer texts on images. As shown in Table 12, InternVL-U demonstrates robust multilingual text generation with scores of 0.738 in English and 0.860 in Chinese, surpassing previous unified models by large margins. These results show that our model effectively addresses the previous deficiency of unified models in rendering legible text.

- Table 12: Evaluation of text-centric text-to-image generation ability on LongText-Bench [42]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

Model #Params LongText-Bench-EN LongText-Bench-ZH Generation Models

FLUX.1 [dev] [63] 12B 0.607 0.005 Z-Image [8] 6B 0.943 0.946 Qwen-Image [136] 20B 0.943 0.946 Nano Banana Pro [29] - 0.981 0.949 Seedream 4.5 [7] - 0.989 0.987 Unified Models

Janus-Pro [16] 7B 0.019 0.006 BLIP-3o [13] 8B 0.021 0.018 Ovis-U1 [124] 2.4B+1.2B 0.030 0.051 BAGEL [31] 7B+7B 0.373 0.310 Lumina-DiMOO [145] 8B 0.437 0.047 OmniGen2 [140] 3B+4B 0.561 0.059 InternVL-U 2B+1.7B 0.738 0.860

- Qualitative Results. As shown in Figure 21, InternVL-U shows excellent capability in rendering Chinese and English characters as well as numerical and mathematical symbols with higher readability and fewer artifacts. Compared with open-source unified multimodal baselines such as BAGEL [31] and Ovis-U1 [124], it achieves better text rendering quality, and remains competitive with the 20B large-scale model Qwen-Image [136] and the closed-source model Nano-Banana-Pro [29].

###### InternVL-U BAGEL Ovis-U1 Qwen-Image Nano Banana Pro

Prompt: Draw a dark green classroom blackboard with three lines of chalk-written text centered on the board. The first line shows a cleanly written math fraction formula: "y=a*x+(b+c)/d", written in clear chalk fraction style. The second line, written in neat Chinese characters, reads: "认真学好数学". The third line, in simple English handwriting, reads: "Study math well". All three lines are aligned and evenly spaced, with natural chalk texture and slightly rough edges. The blackboard surface shows a soft matte finish. The scene is simple and quiet, with an academic atmosphere.

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

Prompt:为春季文学沙龙设计的清新简约风格海报。图像中的文字元素及其空间布局如下：顶部中央位置以简洁的无衬线 字体优雅呈现短语“聆听春日文字之语”。文字下方由柔和的花瓣纹样自然围合，引导视觉流向。海报中心偏下位置， 以流畅的圆体字突出活动主题“春声”。主题下方整齐排列着时间“3月1日至15日”和地点“绿韵书屋”，布局协调。 淡雅的春日色调与细腻纹理相得益彰，整体设计和谐统一。

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

Prompt:A whimsical cartoon-style illustration showcasing two pandas discussing playfully within a cozy bamboo grove setting, surrounded by towering bamboo stalks, leafy trees, and scattered panda toys. In the upper-left corner, Panda One sits upright, ears perked up, with a worried expression and raised paws, speaking in a jagged oval speech bubble with bold comic font: "Our bamboo stash is gone! How will we survive?". On the lower-right side, Panda Two lounges relaxed on a soft log, smiling broadly with arms folded, replying in a smooth, rounded rectangular speech bubble with gentle italics: "Don’t worry! We’ll just find a new forest and restock!". Panda Two's speech bubble slightly overlaps the bottom edge of Panda One’s dialogue bubble, guiding the reader's eye downward and diagonally through the conversation.

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

- Figure 21: Visualizations of text-centric image generation. The results show that our InternVL-U has excellent ability in rendering symbols of Chinese, English, and numerical formulas. Compared with the open-source unified multimodal models BAGEL and Ovis-U1, it has better rendering ability and comparable performance with Qwen Image, a large parameter and commercial closed source model Nano Banana Pro.

- Table 13: Evaluation of knowledge-informed text-to-image generation ability on WISE [96]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

Model #Params Cultural Time Space Biology Physics Chemistry Overall Generation Models

SD3-Medium [35] 2B 0.43 0.50 0.52 0.41 0.53 0.33 0.45 FLUX.1 [dev] [63] 12B 0.48 0.58 0.62 0.42 0.51 0.35 0.50 Qwen-Image [136] 20B 0.63 0.62 0.76 0.60 0.72 0.39 0.63 Unified Models

Janus-Pro [16] 7B 0.30 0.37 0.49 0.36 0.42 0.26 0.35 Lumina-DiMOO [145] 8B 0.35 0.43 0.59 0.31 0.49 0.34 0.40 Ovis-U1 [124] 2.4B+1.2B 0.36 0.46 0.64 0.35 0.52 0.28 0.42 BAGEL [31] 7B+7B 0.44 0.52 0.65 0.42 0.62 0.41 0.49 UniWorld-V1 [76] 7B+13B 0.53 0.55 0.73 0.45 0.59 0.41 0.55 InternVL-U 2B+1.7B 0.37 0.51 0.68 0.39 0.62 0.39 0.46 InternVL-U (w/ CoT) 2B+1.7B 0.55 0.57 0.74 0.51 0.72 0.46 0.58

- Table 14: Evaluation of knowledge-informed text-to-image generation ability on GenExam [132] (Relaxed Scores). Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. Abbreviations: Math=Mathematics, Phy=Physics, Chem=Chemistry, Bio=Biology, Geo=Geography, Comp=Computer Science, Eng=Engineering, Econ=Economics, Hist=History.

Model #Params Math Phy Chem Bio Geo Comp Eng Econ Music Hist Overall Generation Models

FLUX.1 [dev] [63] 12B 12.2 14.4 12.5 22.8 36.4 11.0 14.0 9.2 21.3 21.7 17.6 HunyuanImage-3.0 [9] - 17.0 17.2 18.8 18.7 30.4 15.5 16.9 11.7 23.9 20.4 19.1 Qwen-Image [136] 20B 18.9 26.3 15.3 32.1 49.6 18.9 32.0 20.3 23.4 38.6 27.5 Seedream 4.5 [7] - 44.7 63.4 48.9 75.8 67.6 57.9 69.7 67.3 38.0 55.0 58.8 GPT-Image-1.5 [136] - 65.8 85.4 78.1 91.9 92.5 75.8 86.4 85.5 70.8 90.9 82.3 Nano Banana Pro [29] - 86.3 95.1 88.7 95.9 96.5 91.7 95.1 97.2 91.0 99.9 93.7 Unified Models

BLIP-3o [13] 8B 6.4 5.5 4.7 7.0 16.7 3.6 8.4 2.5 6.0 11.2 7.2 Janus-Pro [16] 7B 13.7 8.8 8.2 7.2 18.8 3.9 10.5 4.2 14.5 6.6 9.6 Ovis-U1 [124] 2.4B+1.2B 12.2 10.8 6.6 10.0 25.4 6.1 8.8 5.4 13.2 15.1 11.4 BAGEL [31] 7B+7B 14.7 10.6 7.9 10.8 24.5 6.8 10.2 5.3 13.7 14.4 11.9 Show-o2 [144] 7B 10.8 11.9 4.8 12.8 33.3 4.7 11.8 7.0 8.8 14.5 12.0 InternVL-U 2B+1.7B 21.5 22.2 19.3 20.0 31.2 9.9 19.6 21.5 17.8 24.9 20.8 InternVL-U (w/ CoT) 2B+1.7B 25.6 24.2 23.5 23.6 35.6 12.0 21.4 24.4 18.4 20.3 22.9

###### 5.3.3. Knowledge-informed Image Generation

WISE. WISE [96] evaluates whether models can integrate world knowledge into text-to-image generation. As shown in Table 13, InternVL-U with CoT yields significant performance gains (from 0.46 to 0.58 overall scores) and surpasses other unified baselines like BAGEL [31] and UniWorld-V1 [76], suggesting high capabilities in cultural commonsense, spatio-temporal reasoning and natural science.

GenExam. GenExam [132] assesses the ability of text-to-image models to understand reasoning with disciplinary knowledge through exam-style instructions. In Table 14, our model achieves the highest scores among unified models, especially on physics, chemistry, and biology. Boosted with CoT, InternVL-U attains an overall score of 22.9 with only 3.7B parameters. This validates InternVL-U’s on science-centric image generation and its integrated ability of understanding, reasoning and generating.

- Qualitative Results. As shown in Figure 22, for prompts that require the model to understand world knowledge, InternVL-U delivers stronger knowledge-grounded rendering, producing visually faithful results for complex instructions and markedly outperforming baselines without explicit knowledge integration.

InternVL-U BAGEL Ovis-U1 Qwen-Image Nano Banana Pro Prompt: A Structural formulas of 2-Bromo-4-methylbenzoic acid.

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Prompt: The animal revered in some cultures as a sacred messenger of the sky god.

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

Prompt: A significant form of timekeeping before mechanical clocks were invented.

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

- Figure 22: Visualizations of knowledge-informed image generation. InternVL-U exhibits superior capability in accurate knowledge rendering. By effectively integrating domain knowledge, our model produces visually faithful results for complex prompts, significantly outperforming baselines that lack specific world knowledge.

###### 5.4. Image Editing

For image editing, we employ existing benchmarks of ImgEdit [153], GEdit-Bench [83], and RISEBench [161]. Furthermore, given the extensive application scenarios for text editing, we have constructed an additional text-centric image editing benchmark, namely TextEdit, to evaluate models’ accuracy in performing text editing across virtual and real-world scenarios.

###### 5.4.1. General Image Editing

ImgEdit. ImgEdit [153] covers diverse range of single- and multi-turn editing tasks. As shown in Table 15, InternVL-U demonstrates competitive editing proficiency among unified models, where the CoT models achieves 3.82 overall score.

GEdit-Bench. GEdit-Bench [83] contains prompts with both real-world editing requirements and high diversity. In Table 16, InternVL-U achieves an average score of 6.66, surpassing baselines like BAGEL [31] (6.52) and Ovis-U1 [124] (6.42). Notably, applying the CoT strategy further enhances performance, raising the score to 6.88. While specialized editing models like Qwen-Image-Edit [136] still hold a lead in certain metrics, InternVL-U’s performance confirms the viability of a unified architecture for diverse editing tasks, particularly when augmented with explicit reasoning steps.

- Qualitative Results. As shown in Figure 23, InternVL-U excels at producing realistic textures and styles while preserving the source image’s lighting and structural details, resulting in more natural and coherent edits across a wide range of scenarios compared with other open-source models.

- Table 15: Evaluation of general image editing ability on ImgEdit [153]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters.

Model #Params Add Adjust Extract Replace Remove Background Style Hybrid Action Overall Generation Models

FLUX.1 Kontext [65] 12B 4.25 4.15 2.35 4.56 3.57 4.26 4.57 3.68 4.63 4.00 GPT-Image-1 [High] [99] - 4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20 Qwen-Image-Edit [136] 20B 4.38 4.16 3.43 4.66 4.14 4.38 4.81 3.82 4.69 4.27 Z-Image-Edit [8] 6B 4.40 4.14 4.30 4.57 4.13 4.14 4.85 3.63 4.50 4.30 Unified Models

Lumina-DiMOO [145] 8B 3.41 2.38 1.90 3.26 2.21 2.11 4.19 2.26 3.17 2.77 BAGEL [31] 7B+7B 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 UniWorld-V1 [76] 20B 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 OmniGen2 [138] 3B+4B 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 Ovis-U1 [124] 2.4B+1.2B 3.99 3.73 2.66 4.38 4.15 4.05 4.86 3.43 4.68 3.97 InternVL-U 2B+1.7B 4.13 3.40 2.27 4.13 3.39 3.84 4.77 3.03 4.05 3.67 InternVL-U (w/ CoT) 2B+1.7B 4.24 3.80 2.58 4.36 3.51 3.92 4.69 3.00 4.31 3.82

- Table 16: Evaluation of general image editing ability on GEdit-Bench [83]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. Abbreviations: BC=Background Change, CA=Color Alteration, MM=Material Modification, MC=Motion Change, PB=Portrait Beautification, ST=Style Transfer, SA=Subject Addition, SR=Subject Removal, SRp=Subject Replacement, TM=Text Modification, TT=Tone Transfer.

Models #Params BC CA MM MC PB ST SA SR SRp TM TT Avg/G_O Generation Models

GPT Image 1 [99] - 6.96 6.85 7.10 5.41 6.74 7.44 7.51 8.73 8.55 8.45 8.69 7.49 Qwen-Image-Edit [136] 20B 8.23 8.30 7.33 8.05 7.49 6.74 8.57 8.09 8.29 8.48 8.50 8.01 Unified Models

Lumina-DiMOO [145] 8B 3.43 4.27 3.08 2.77 4.74 5.19 4.44 3.80 4.38 2.68 4.20 3.91 Ovis-U1 [124] 2.4B+1.2B 7.49 6.88 6.21 4.79 5.98 6.46 7.49 7.25 7.27 4.48 6.31 6.42 BAGEL [31] 7B+7B 7.32 6.91 6.38 4.75 4.57 6.15 7.90 7.16 7.02 7.32 6.22 6.52 InternVL-U 2B+1.7B 7.08 7.05 6.38 7.02 6.03 6.27 7.13 6.55 6.33 6.59 6.85 6.66 InternVL-U (w/ CoT) 2B+1.7B 7.05 7.87 6.50 6.99 5.77 6.10 7.33 7.16 7.12 7.36 6.46 6.88

Original Image InternVL-U BAGEL Ovis-U1 Qwen-Image Nano Banana Pro

Editing Instruction : Change the material of the tabletop to frosted glass.

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

Editing Instruction : Replace the green flower bed in the center with a fountain.

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

Editing Instruction : 把这张素描变成油画风格。

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

- Figure 23: Visualizations of general image editing. InternVL-U excels in generating realistic textures and styles while maintaining high fidelity to the original image’s lighting and structural details, demonstrating superior performance across various tasks compared to other open-source models.

###### 5.4.2. Text-centric Image Editing

TextEdit. To rigorously evaluate text editing capabilities, we introduce TextEdit†, a novel benchmark containing 2,148 samples featuring diverse editing scenarios and high-quality edited image ground truth (see Section A for more details about the benchmark construction). As shown in Tables 17 and 18, InternVL-U demonstrates superior performance on this benchmark, achieving a F1 score of 0.71 in classic metrics, matching the Nano Banano Pro [29] and drastically outperforming unified counterparts like Ovis-U1 [124] (0.35). This advantage is further corroborated by MLLM-based evaluations, where InternVL-U attains an average score of 0.88 on the images from real scene, significantly surpassing BAGEL [31] (0.53) and exhibiting competitive capability against closed-source commercial models like GPT-Image-1.5 [98].

- Table 17: Evaluation of text-centric image editing on TextEdit (Classic Metrics). Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. “Real” refers to source images from real-world scene, while “Virtual” refers to images from virtual scene. Abbreviations: OA=OCR Accuracy, OP=OCR Precision, OR=OCR Recall, F1=OCR F1-Score, NED=ROI-Aware NED, CLIP=CLIPScore, AES=Aesthetic Score. For detailed evaluation metric, please refer to Appendix Section A.

|Models|# Params|Real OA OP OR F1 NED CLIP AES<br><br>|Virtual OA OP OR F1 NED CLIP AES|
|---|---|---|---|
| | | | |

Generation Models

Qwen-Image-Edit [136] 20B 0.75 0.68 0.66 0.67 0.71 0.75 5.72 0.78 0.75 0.73 0.74 0.75 0.81 5.21 GPT-Image-1.5 [98] - 0.74 0.69 0.67 0.68 0.68 0.75 5.78 0.73 0.72 0.71 0.71 0.70 0.80 5.28 Nano Banana Pro [29] - 0.77 0.72 0.70 0.71 0.72 0.75 5.79 0.80 0.78 0.77 0.78 0.78 0.81 5.28 Unified Models

Lumina-DiMOO [145] 8B 0.22 0.23 0.19 0.20 0.19 0.69 5.53 0.22 0.25 0.21 0.22 0.20 0.72 4.76 Ovis-U1 [124] 2.4B+1.2B 0.40 0.37 0.34 0.35 0.35 0.72 5.32 0.37 0.40 0.38 0.39 0.33 0.75 4.66 BAGEL [31] 7B+7B 0.60 0.59 0.53 0.55 0.55 0.74 5.71 0.57 0.60 0.56 0.57 0.54 0.78 5.19 InternVL-U 2B+1.7B 0.77 0.73 0.70 0.71 0.72 0.75 5.70 0.79 0.77 0.75 0.75 0.77 0.80 5.12

- Table 18: Evaluation of text-centric image editing on TextEdit (MLLM-based Metrics). Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. “Real” refers to source images from real-world scene, while “Virtual” refers to images from virtual scene. Abbreviations: TA: Text Accuracy, TP: Text Preservation, SI: Scene Integrity, LR: Local Realism, VC: Visual Coherence, Avg: MLLM Overall Average. For detailed evaluation metric, please refer to Appendix Section A.

|Models|# Params|Real TA TP SI LR VC Avg<br><br>|Virtual TA TP SI LR VC Avg|
|---|---|---|---|
| | | | |

Generation Models

Qwen-Image-Edit [136] 20B 0.92 0.82 0.75 0.57 0.80 0.77 0.57 0.79 0.92 0.80 0.77 0.77 GPT-Image-1.5 [98] - 0.96 0.94 0.86 0.80 0.93 0.90 0.82 0.93 0.96 0.91 0.87 0.90 Nano Banana Pro [29] - 0.96 0.95 0.85 0.88 0.93 0.91 0.87 0.92 0.96 0.94 0.89 0.92 Unified Models

Lumina-DiMOO [145] 8B 0.17 0.06 0.04 0.02 0.05 0.09 0.02 0.06 0.16 0.05 0.03 0.08 Ovis-U1 [124] 2.4B+1.2B 0.31 0.12 0.12 0.07 0.18 0.18 0.06 0.16 0.31 0.14 0.13 0.19 BAGEL [31] 7B+7B 0.68 0.60 0.38 0.35 0.56 0.53 0.38 0.51 0.68 0.62 0.42 0.54 InternVL-U 2B+1.7B 0.94 0.90 0.71 0.80 0.80 0.88 0.87 0.86 0.91 0.82 0.62 0.83

- Qualitative Results. As shown in Figure 24, we visualize the performance of representative top-tier open-source and commercial models on our proposed benchmark TextEdit, our InternVL-U, which achieves strong results across a wide range of text-editing scenarios. In particular, InternVL-U can accurately localize the text to be replaced in the image and substitute it with the target text, while preserving both visual aesthetics and textual correctness. These results provide a clear picture of the current state of the art and effectively mark the upper bounds of existing text editing capabilities, highlighting how our benchmark delineates the performance frontier of text-centric image editing.

† https://github.com/open-compass/TextEdit

Original Image InternVL-U BAGEL Ovis-U1 Qwen-Image Nano Banana Pro

Editing Instruction : Replace the text 'Nestle' with 'Aquafina'.

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

Editing Instruction : Replace the text 'DELTA' with 'AIRLINES'.

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

Editing Instruction : Replace the text 'BAUME&MERCIER' with 'ROLEX'.

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

- Figure 24: Visualizations of text-centric image editing. InternVL-U demonstrates more accurate and faithful text editing capabilities, while maintaining strong consistency for both textual and visual content outside the target editing region.

- 5.4.3. Reasoning-informed Image Editing

RISEBench. We further assessed the model’s ability to handle complex editing instructions that require logical deduction on the RISEBench [161] benchmark. As detailed in Table 19, the introduction of the CoT strategy yields a remarkable performance boost, elevating InternVL-U’s overall score from 3.6 to 9.4. This enhancement allows it to surpass both open-source unified baselines (e.g., BAGEL [31] at 6.1) and specialized generation models like Qwen-Image-Edit [136] (8.9). Notably, CoT significantly improves Instruction Reasoning (IR) and Appearance Consistency (AC), demonstrating that explicit reasoning is essential for the accurate execution of complex, logic-dependent editing tasks.

- Table 19: Evaluation of reasoning-informed image editing ability on RISEBench [161]. Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. Abbreviations: IR=Instruction Reasoning, AC=Appearance Consistency, VP=Visual Plausibility.

Models #Params Temporal Causal Spatial Logical Overall IR AC VP Generation Models

FLUX.1 Kontext [65] 12B 2.3 5.5 13.0 1.2 5.8 26.0 71.6 85.2 Qwen-Image-Edit [136] 20B 4.7 10.0 17.0 2.4 8.9 37.2 66.4 86.9 Seedream 4.0 [110] - 12.9 12.2 11.0 7.1 10.8 58.9 67.4 91.2 Nano Banana Pro [29] - 41.2 61.1 48.0 37.6 47.2 77.0 85.5 94.4 GPT-Image-1.5 [98] - 54.1 60.0 62.0 21.2 50.0 69.7 92.5 94.9 Unified Models

Lumina-DiMOO [145] 8B 2.4 1.1 4.0 1.2 2.2 34.0 50.7 72.3 Ovis-U1 [124] 2.4B+1.2B 1.2 3.3 4.0 2.4 2.8 33.9 52.7 72.9 BAGEL [31] 7B+7B 2.4 5.6 14.0 1.2 6.1 36.5 53.5 73.0 InternVL-U 2B+1.7B 3.5 2.2 5.0 3.5 3.6 35.6 52.7 75.9 InternVL-U (w/ CoT) 2B+1.7B 4.7 7.8 1.8 5.9 9.4 43.9 64.4 79.7

- Qualitative Results. As shown in Figure 25, InternVL-U handles complex editing instructions that require multi-step reasoning and strict logical constraints more reliably than prior methods. It can accurately interpret and execute diverse constraints, including temporal calculations (e.g., updating calendar dates), spatial and cultural understanding (e.g., retrieve contextually appropriate poetry given an image), and precise algorithmic rules (e.g., binary search tree insertion).

###### Original Image InternVL-U BAGEL Ovis-U1 Qwen-Image Nano Banana Pro

Editing Instruction: Draw what it will look like tomorrow.

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

Editing Instruction: 给图中李白人物形象加两句诗竖着排列在画面右上角.

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

Editing Instruction: Insert value 88 into the BST, and highlight the inserted nodes with red rectangular boxes.

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

- Figure 25: Visualizations of reasoning-informed image editing. InternVL-U outperforms state-of-the-art models in handling complex prompts requiring multi-step reasoning. The results demonstrate our model’s superior ability to accurately interpret and execute diverse logical constraints, spanning temporal calculations for updating calendar dates, spatial and cultural understanding for poem placement, and precise algorithmic rules for binary search tree insertion.

Original Image InternVL-U BAGEL Ovis-U1 Qwen-Image Nano Banana Pro Editing Instruction: Identify all nodes with a degree of 1, using a green bounding box to highlight them.

Editing Instruction: 将这张图变成一张表现自信登场、略带傲娇语气的梗图。

Editing Instruction: Move the pyramid in the drawing along vector [0, -6, 0], and keep the source geometry unchanged.

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

- Figure 26: Visualizations of more special image editing examples. InternVL-U demonstrates its advanced spatial reasoning and precise control in complex editing tasks. The results highlight the model’s superior ability to accurately identify graph properties (e.g., node degrees), generate humor-centric content with appropriate expressions, and execute precise 3D geometric transformations based on coordinate vectors, showing its broad applicability in specialized domains.

###### 5.5. More Qualitative Results

As shown in Figure 26, we present specialized editing examples that showcase InternVL-U’s distinctive capabilities beyond standard editing, enabling it to handle uncommon and challenging requirements robustly, including computer-science knowledge, humor-centric content creation, and mathematics-related edits, demonstrating robust controllability and broad applicability beyond standard editing settings.

#### 6. Conclusion

In this work, we presented InternVL-U, a unified multimodal model that effectively democratizes the capabilities of understanding, reasoning, generation, and editing. By adhering to the principles of unified context modeling with modality-specific modularity and decoupled visual representations, our architecture seamlessly integrates generative capabilities into a strong understanding backbone. To further bridge the gap between high-level intelligence and visual generation, we introduced a comprehensive data synthesis pipeline with the Chainof-Thought (CoT) paradigm, enabling the model to align abstract user intent with precise visual execution. Empirical results confirm that InternVL-U not only excels in knowledge-intensive generation and editing but also retains competitive performance in multimodal understanding and reasoning benchmarks. We hope InternVLU serves as a robust baseline and accelerates the community’s progress toward developing comprehensive, omni-capable AGI-oriented UMMs.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2.1
- [2] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 2.1
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1, 2.1
- [4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 4.4.1, 4.4.2, 4.7, A.2.2
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923,

2025. 2.1, 4.2.2, 4.2.3, 4.3.1, 4.3.2, 4.5.3, 4

- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 2.2, 1, 5.1
- [7] ByteDance. Seedream 4.5. https://seed.bytedance.com/en/seedream4_5, 2025. Accessed: 2026-02-10. 6, 9, 10, 11, 12, 14
- [8] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Shijie Huang, Zhaohui Hou, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 1, 2.2, 5, 6, 7, 8, 9, 10, 11, 12, 15
- [9] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951,

2025. 1, 2.2, 14

- [10] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June

2022. 2.2

- [11] Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arXiv:2506.07977, 2025. 5.3, 5.3.1, 9, 10
- [12] Yifan Chang, Yukang Feng, Jianwen Sun, Jiaxin Ai, Chuanhao Li, S Kevin Zhou, and Kaipeng Zhang. Sridbench: Benchmark of scientific research illustration drawing of image generation model. arXiv preprint arXiv:2505.22126, 2025. 4.4
- [13] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 1, 10, 12, 14

- [14] Jiuhai Chen, Le Xue, Zhiyang Xu, Xichen Pan, Shusheng Yang, Can Qin, An Yan, Honglu Zhou, Zeyuan Chen, Lifu Huang, et al. Blip3o-next: Next frontier of native image generation. arXiv preprint arXiv:2510.15857, 2025. 2.3, 1
- [15] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025. 1
- [16] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025. 5.2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14
- [17] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 2.1
- [18] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024.
- [19] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visuallinguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 1, 2.1, 3.1.1
- [20] Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, Pengfei Wan, and Yi-Fan Zhang. Opengpt-4o-image: A comprehensive dataset for advanced image generation and editing, 2025. 1
- [21] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2.1
- [22] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025. 4.3, 4.3.2
- [23] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025. 4.4.2
- [24] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025. 1, 2.3, 3.1.1, 4
- [25] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners, 2025. 2.1
- [26] Dawei Dai, YuTang Li, YingGe Liu, Mingming Jia, Zhang YuanHui, and Guoyin Wang. 15m multimodal facial image-text dataset. arXiv preprint arXiv:2407.08515, 2024. 1
- [27] Dawei Dai, Xu Long, Li Yutang, Zhang Yuanhui, and Shuyin Xia. Humanvlm: Foundation for human-scene vision-language model, 2024. 1
- [28] Google Deepmind. Gemini 2.5: Our most intelligent ai model. https://blog.google/ technology/google-deepmind/gemini-model-thinking-updates-march-2025/ #gemini-2-5-thinking/, 2025. Accessed: 2025-03-26. 2.1

- [29] Google DeepMind. Gemini3pro image model card. https://storage.googleapis.com/ deepmind-media/Model-Cards/Gemini-3-Pro-Image-Model-Card.pdf, Nov 2025. 1, 2.2, 4.4.2, 6, 9, 10, 11, 12, 5.3.2, 14, 5.4.2, 17, 18, 19, 23, 24
- [30] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153,

2023. 4.5.3

- [31] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 2.1, 2.3, 5.2, 4, 5.3.1, 5, 6, 7, 8, 9, 10, 11, 12, 5.3.2, 13, 14, 5.3.3, 5.4.1, 15, 16, 5.4.2, 17, 18, 5.4.3, 19, 23, 24
- [32] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2.1
- [33] Nikai Du, Zhennan Chen, Shan Gao, Zhizhou Chen, Xi Chen, Zhengkai Jiang, Jian Yang, and Ying Tai. Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461, 2025. 5.3, 5.3.2, 11, A.1, 20
- [34] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201,

2024. 5.1

- [35] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 1, 3.1.1, 5, 6, 9, 13
- [36] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2.2
- [37] Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025. 1
- [38] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 5.2, 4
- [39] Yifan Gao, Jinpeng Lin, Min Zhou, Chuanbin Liu, Hongtao Xie, Tiezheng Ge, and Yuning Jiang. Textpainter: Multimodal text image generation with visual-harmony and text-comprehension for poster design. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7236–7246, 2023. 1
- [40] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025. 5, 7, 8
- [41] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024. 1
- [42] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025. 5.3, 5.3.2, 12, A.1, 20
- [43] GeoGebra Team. Geogebra: Dynamic mathematics software. https://www.geogebra.org, 2024. 4.5.1

- [44] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152,

2023. 5.3, 5.3.1, 5

- [45] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139– 144, 2020. 2.2
- [46] Google. Gemini 3 flash: frontier intelligence built for speed. https://blog.google/products/ gemini/gemini-3-flash/, 2025. 2.1, 4.4.2
- [47] Google. Gemini 3 pro: Best for complex tasks and bringing creative concepts to life. https:// deepmind.google/models/gemini/pro/, 2025. A.2.2, A.2.2
- [48] Janna Hastings, Gareth Owen, Adriano Dekker, Marcus Ennis, Namrata Kale, Venkatesh Muthukrishnan, Steve Turner, Neil Swainston, Pedro Mendes, and Christoph Steinbeck. Chebi in 2016: Improved services and an expanding collection of metabolites. Nucleic acids research, 44(D1):D1214–D1219, 2016. 4.4.1
- [49] Xin He, Longhui Wei, Jianbo Ouyang, Minghui Liao, Lingxi Xie, and Qi Tian. Emma: Efficient multimodal understanding, generation, and editing with a unified architecture, 2025. 1, 2.3
- [50] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021. A.2.2, A.2.2
- [51] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arxiv:2006.11239, 2020. 2.2
- [52] Ming Hu, Chenglong Ma, Wei Li, Wanghan Xu, Jiamin Wu, Jucheng Hu, Tianbin Li, Guohang Zhuang, Jiaqi Liu, Yingzhou Lu, Ying Chen, Chaoyang Zhang, Cheng Tan, Jie Ying, Guocheng Wu, Shujian Gao, Pengcheng Chen, Jiashi Lin, Haitao Wu, Lulu Chen, Fengxiang Wang, Yuanyuan Zhang, Xiangyu Zhao, Feilong Tang, Encheng Su, Junzhi Ning, Xinyao Liu, Ye Du, Changkai Ji, Cheng Tang, Huihui Xu, Ziyang Chen, Ziyan Huang, Jiyao Liu, Pengfei Jiang, Yizhou Wang, Chen Tang, Jianyu Wu, Yuchen Ren, Siyuan Yan, Zhonghua Wang, Zhongxing Xu, Shiyan Su, Shangquan Sun, Runkai Zhao, Zhisheng Zhang, Yu Liu, Fudi Wang, Yuanfeng Ji, Yanzhou Su, Hongming Shan, Chunmei Feng, Jiahao Xu, Jiangtao Yan, Wenhao Tang, Diping Song, Lihao Liu, Yanyan Huang, Lequan Yu, Bin Fu, Shujun Wang, Xiaomeng Li, Xiaowei Hu, Yun Gu, Ben Fei, Zhongying Deng, Benyou Wang, Yuewen Cao, Minjie Shen, Haodong Duan, Jie Xu, Yirong Chen, Fang Yan, Hongxia Hao, Jielan Li, Jiajun Du, Yanbo Wang, Imran Razzak, Chi Zhang, Lijun Wu, Conghui He, Zhaohui Lu, Jinhai Huang, Yihao Liu, Fenghua Ling, Yuqiang Li, Aoran Wang, Qihao Zheng, Nanqing Dong, Tianfan Fu, Dongzhan Zhou, Yan Lu, Wenlong Zhang, Jin Ye, Jianfei Cai, Wanli Ouyang, Yu Qiao, Zongyuan Ge, Shixiang Tang, Junjun He, Chunfeng Song, Lei Bai, and Bowen Zhou. A survey of scientific large language models: From data foundations to agent frontiers, 2025. 4.4
- [53] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 5.3, 5.3.1, 6
- [54] Yuqing Huang, Rongyang Zhang, Xuesong He, Xuyang Zhi, Hao Wang, Xin Li, Feiyang Xu, Deguang Liu, Huadong Liang, Yi Li, et al. Chemeval: A comprehensive multi-level chemical evaluation for large language models. arXiv preprint arXiv:2409.13989, 2024. 4.4.1
- [55] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990,

2024. 1

- [56] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276,

2024. 1

- [57] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017. 2.2
- [58] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2.2
- [59] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, pages 235–251, 2016. 4.4.1
- [60] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. pages 4999–5007, 2017. 4.4.1
- [61] Sebastian Koch, Albert Matveev, Zhongshi Jiang, Francis Williams, Alexey Artemov, Evgeny Burnaev, Marc Alexa, Denis Zorin, and Daniele Panozzo. Abc: A big cad model dataset for geometric deep learning. in 2019 ieee. In CVPR, 2018. 4.5.2
- [62] Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, and Aleksandr Gordeev. NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining. arXiv preprint arXiv:2507.14119, 2025. 1
- [63] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. 1, 2.2, 5, 6, 7, 8, 9, 11, 12, 13, 14
- [64] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025. 2.2
- [65] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. 2.2, 4.2.3, 4.5.3, 15, 19
- [66] Rui Lan, Yancheng Bai, Xu Duan, Mingxing Li, Dongyang Jin, Ryan Xu, Dong Nie, Lei Sun, and Xiangxiang Chu. Flux-text: A simple and advanced diffusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329, 2025. 4.3, 4.3.2, A.1, 20
- [67] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 5.2, 4
- [68] Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. Onecat: Decoder-only auto-regressive model for unified understanding and generation. arXiv preprint arXiv:2509.03498, 2025. 1, 2.3
- [69] Hao Li, Changyao Tian, Jie Shao, Xizhou Zhu, Zhaokai Wang, Jinguo Zhu, Wenhan Dou, Xiaogang Wang, Hongsheng Li, Lewei Lu, et al. Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29767–29779, 2025. 2.3
- [70] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 2.1
- [71] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2.1
- [72] Shufan Li, Jiuxiang Gu, Kangning Liu, Zhe Lin, Zijun Wei, Aditya Grover, and Jason Kuen. Lavida-o: Elastic large masked diffusion models for unified multimodal understanding and generation. arXiv preprint arXiv:2509.19244, 2025. 2.3

- [73] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Feize Wu, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, et al. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025. 2.3
- [74] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024. 3.1.1
- [75] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 1
- [76] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 1, 1, 5, 6, 13, 5.3.3, 15
- [77] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 2.1
- [78] Jinpeng Lin, Min Zhou, Ye Ma, Yifan Gao, Chenxi Fei, Yangjian Chen, Zhang Yu, and Tiezheng Ge. Autoposter: A highly automatic and content-aware design system for advertising poster generation,

2023. 1

- [79] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code, 2024. 2.2
- [80] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 2.1, 4
- [81] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [82] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 2.1
- [83] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761,

2025. 2.2, 4.2.3, 5.4, 5.4.1, 16

- [84] Wentao Liu, Qianjun Pan, Yi Zhang, Zhuo Liu, Ji Wu, Jie Zhou, Aimin Zhou, Qin Chen, Bo Jiang, and Liang He. Cmm-math: A chinese multimodal math dataset to evaluate and enhance the mathematics reasoning of large multimodal models. arXiv preprint arXiv:2409.02834, 2024. 4.4.1
- [85] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2.2
- [86] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 5.2, 4
- [87] Zhiheng Liu, Weiming Ren, Haozhe Liu, Zijian Zhou, Shoufa Chen, Haonan Qiu, Xiaoke Huang, Zhaochong An, Fanny Yang, Aditya Patel, et al. Tuna: Taming unified visual representations for native unified multimodal models. arXiv preprint arXiv:2512.02014, 2025. 2.3, 4
- [88] Gen Luo, Wenhan Dou, Wenhao Li, Zhaokai Wang, Xue Yang, Changyao Tian, Hao Li, Weiyun Wang, Wenhai Wang, Xizhou Zhu, et al. Mono-internvl-1.5: Towards cheaper and faster monolithic multimodal large language models. arXiv preprint arXiv:2507.12566, 2025. 3.1.1
- [89] Gen Luo, Wenhan Dou, Wenhao Li, Zhaokai Wang, Xue Yang, Changyao Tian, Hao Li, Weiyun Wang, Wenhai Wang, Xizhou Zhu, Yu Qiao, and Jifeng Dai. Mono-internvl-1.5: Towards cheaper and faster monolithic multimodal large language models. arXiv preprint arXiv:2507.12566, 2025. 2.1

- [90] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. arXiv preprint arXiv:2410.08202, 2024. 2.1
- [91] Jian Ma, Xujie Zhu, Zihao Pan, Qirong Peng, Xu Guo, Chen Chen, and Haonan Lu. X2edit: Revisiting arbitrary-instruction image editing through self-constructed data and task-aware representation learning,

2025. 1

- [92] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751, 2025. 4
- [93] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 2.1
- [94] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022. 5.2, 4
- [95] TIMODAL MODEL. Manzano: Asimple and scalable unified mul-timodal model with a hybrid vision tokenizer. 2.3
- [96] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 5.3, 13, 5.3.3
- [97] OpenAI. Gpt-5 system card. https://cdn.openai.com/pdf/ 8124a3ce-ab78-4f06-96eb-49ea29ffb52f/gpt5-system-card-aug7.pdf, 2025. 2.1, 4.4.2
- [98] OpenAI. Gpt-image-1.5. https://openai.com/index/new-chatgpt-images-is-here/, 2025. 2.2, 5.4.2, 17, 18, 19, 23, 24
- [99] OpenAI. Introducing our latest image generation model in the api, 2025. 5, 6, 15, 16
- [100] OpenCompass. GenEditEvalKit. https://github.com/open-compass/GenEditEvalKit. 1
- [101] OpenCompass. TextEdit. https://github.com/open-compass/TextEdit. 1
- [102] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 1, 4
- [103] Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing, 2025. 1
- [104] Zhipeng Qian, Pei Zhang, Baosong Yang, Kai Fan, Yiwei Ma, Derek F Wong, Xiaoshuai Sun, and Rongrong Ji. Anytrans: Translate anytext in the image with large scale models. arXiv preprint arXiv:2406.11432,

2024. 4.3

- [105] Zeju Qiu, Weiyang Liu, Haiwen Feng, Zhen Liu, Tim Z Xiao, Katherine M Collins, Joshua B Tenenbaum, Adrian Weller, Michael J Black, and Bernhard Schölkopf. Can large language models understand symbolic graphics programs? arXiv preprint arXiv:2408.08313, 2024. 4.4.2
- [106] Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, et al. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. arXiv preprint arXiv:2505.06708, 2025. 3.1.2
- [107] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 2.2

- [108] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2.2
- [109] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 1
- [110] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 2.2, 19
- [111] Tao Shen, Xin Wan, Taicai Chen, Rui Zhang, Junwen Pan, Dawei Lu, Fanding Lei, Zhilin Lu, Yunfei Yang, Chen Cheng, Qi She, Chang Liu, and Zhenbang Sun. Mammothmoda2: A unified ar-diffusion framework for multimodal understanding and generation. arXiv preprint arXiv:2511.18262, 2025. 2.3
- [112] Yuxin Song, Wenkai Dong, Shizun Wang, Qi Zhang, Song Xue, Tao Yuan, Hu Yang, Haocheng Feng, Hang Zhou, Xinyan Xiao, et al. Query-kontext: An unified multimodal model for image generation and editing. arXiv preprint arXiv:2509.26641, 2025. 1
- [113] Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278,

2025. 1, 2.3

- [114] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 1, 2.3
- [115] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2.1
- [116] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025. 2.2
- [117] Changyao Tian, Hao Li, Gen Luo, Xizhou Zhu, Weijie Su, Hanming Deng, Jinguo Zhu, Jie Shao, Ziran Zhu, Yunpeng Liu, et al. Navil: Rethinking scaling properties of native multimodal large language models under data constraints. arXiv preprint arXiv:2510.08565, 2025. 3.1.1
- [118] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, Hongsheng Li, Yu Qiao, and Jifeng Dai. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. arXiv preprint arXiv:2401.10208, 2024. 2.1
- [119] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 2.2
- [120] Rui Tian, Mingfei Gao, Haiming Gang, Jiasen Lu, Zhe Gan, Yinfei Yang, Zuxuan Wu, and Afshin Dehghan. Unigen-1.5: Enhancing image generation and editing through reward unification in reinforcement learning. arXiv preprint arXiv:2511.14760, 2025. 1, 2.3
- [121] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2.1
- [122] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. 2023. 4.3, A.1, 20
- [123] Guo-Hua Wang, Liangfu Cao, Tianyu Cui, Minghao Fu, Xiaohao Chen, Pengxin Zhan, Jianshan Zhao, Lan Li, Bowen Fu, Jiaqi Liu, and Qing-Guo Chen. Ovis-image technical report. arXiv preprint arXiv:2511.22982, 2025. 2.2

- [124] Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, et al. Ovis-u1 technical report. arXiv preprint arXiv:2506.23044, 2025. 1, 2.3, 3.2.2, 5.2, 4, 6, 7, 8, 9, 10, 11, 12, 5.3.2, 13, 14, 5.4.1, 15, 16, 5.4.2, 17, 18, 19, 23, 24
- [125] Haomin Wang, Jinhui Yin, Qi Wei, Wenguang Zeng, Lixin Gu, Shenglong Ye, Zhangwei Gao, Yaohui Wang, Yanting Zhang, Yuanqi Li, et al. Internsvg: Towards unified svg tasks with multimodal large language models. arXiv preprint arXiv:2510.11341, 2025. 4.4.2
- [126] Peijie Wang, Zhong-Zhi Li, Fei Yin, Dekang Ran, and Cheng-Lin Liu. Mv-math: Evaluating multimodal math reasoning in multi-visual contexts. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19541–19551, 2025. 4.4.1
- [127] Peiyu Wang, Yi Peng, Yimeng Gan, Liang Hu, Tianyidan Xie, Xiaokun Wang, Yichen Wei, Chuanxin Tang, Bo Zhu, Changshi Li, et al. Skywork unipic: Unified autoregressive modeling for visual understanding and generation. arXiv preprint arXiv:2508.03320, 2025. 2.3
- [128] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2.1
- [129] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1, 2.1, 2.3, 5.1, 4
- [130] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 2.1
- [131] Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gptimage-edit-1.5m: A million-scale, gpt-generated image dataset, 2025. 1
- [132] Zhaokai Wang, Penghao Yin, Xiangyu Zhao, Changyao Tian, Yu Qiao, Wenhai Wang, Jifeng Dai, and Gen Luo. Genexam: A multidisciplinary text-to-image exam. arXiv preprint arXiv:2509.14232, 2025. 4.4, 5.3, 14, 5.3.3
- [133] Navve Wasserman, Noam Rotstein, Roy Ganz, and Ron Kimmel. Paint by inpaint: Learning to add image objects by removing them first. arXiv preprint arXiv:2404.18212, 2024. 1
- [134] Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. arXiv preprint arXiv:2411.07199, 2024. 1
- [135] Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, and Lei Zhang. Tiif-bench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161, 2025. 5.3, 5.3.1, 7, 8
- [136] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 3.1.2, 3.2.2, 4.2.2, 4.2.3, 4.3, 4.5.3, 5.1, 6, 7, 8, 9, 10, 11, 12, 5.3.2, 13, 14, 5.4.1, 15, 16, 17, 18, 5.4.3, 19, 23, 24
- [137] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 2.2
- [138] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 1, 5, 6, 9, 15

- [139] Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661, 2025. 1
- [140] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 12
- [141] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 5.2, 4
- [142] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. Sana: Efficient high-resolution image synthesis with linear diffusion transformer, 2024. 1, 3.2.2, 5.1
- [143] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 7, 8
- [144] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 1, 4, 5, 9, 14
- [145] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025. 1, 2.3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 23, 24
- [146] Wanghan Xu, Yuhao Zhou, Yifan Zhou, Qinglong Cao, Shuo Li, Jia Bu, Bo Liu, Yixin Chen, Xuming He, Xiangyu Zhao, et al. Probing scientific general intelligence of llms with scientist-aligned workflows. arXiv preprint arXiv:2512.16969, 2025. 4.4
- [147] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2.1
- [148] Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025. 2.1
- [149] Chenyu Yang, Xizhou Zhu, Jinguo Zhu, Weijie Su, Junjie Wang, Xuan Dong, Wenhai Wang, Bin Li, Jie Zhou, Yu Qiao, and Jifeng Dai. Vision model pre-training on interleaved image-text data via latent compression learning. arXiv preprint arXiv:2406.07543, 2024. 2.1
- [150] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025. 2.3
- [151] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025. 2.1
- [152] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, Conghui He, and Weijia Li. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. https://arxiv.org/abs/2508.09987, 2025. 1
- [153] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 1, 5.4, 5.4.1, 15
- [154] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024. 1
- [155] Tai-Ling Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 34(3):509–521, 2019. 1

- [156] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024. 4.4, 4.4.1, 5.2, 4, 1
- [157] Christoph Zauner. Implementation and benchmarking of perceptual image hash functions. 2010. 4.2.1
- [158] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024. 5.2, 4
- [159] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739, 2024. 4.4.1
- [160] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale, 2024. 1
- [161] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025. 5.4, 5.4.3, 19
- [162] Yuhao Zhou, Yiheng Wang, Xuming He, Ruoyao Xiao, Zhiwei Li, Qiantai Feng, Zijie Guo, Yuejin Yang, Hao Wu, Wenxuan Huang, Jiaqi Wei, Dan Si, Xiuqi Yao, Jia Bu, Haiwen Huang, Tianfan Fu, Shixiang Tang, Ben Fei, Dongzhan Zhou, Fenghua Ling, Yan Lu, Siqi Sun, Chenhui Li, Guanjie Zheng, Jiancheng Lv, Wenlong Zhang, and Lei Bai. Scientists’ first exam: Probing cognitive abilities of mllm via perception, understanding, and reasoning, 2025. 4.4
- [163] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2.1
- [164] Shang Zhu, Xuefeng Liu, and Ghazal Khalighinejad. Chemqa: a multimodal question-and-answering dataset on chemistry reasoning. https://huggingface.co/datasets/shangzhu/ChemQA, 2024. 4.4.1

#### A. TextEdit Benchmark

###### A.1. Design Motivation

With the increasing adoption of text-to-image and image editing models in real-world applications, text-centric image editing has become a frequent requirement in advertising design, poster revision, UI localization, and commercial asset updates. However, existing general-purpose image editing models remain unreliable when processing text content. On the one hand, the generated text often suffers from misspellings, distorted glyphs, broken multi-line layouts, or unnatural blending with the background. On the other hand, while replacing text, models frequently inadvertently alter non-target regions (e.g., material textures, facial details, or background structures), effectively turning “edit the text” into “edit the whole image”. Moreover, as is shown in Table 20, existing text-centric benchmarks (e.g., AnyText [122], LongText [42], and CVTG-2K [33]) primarily focus on the accuracy of text generation, lacking adequate modeling for specific text editing scenarios. Although text editing benchmarks like MARIO-Eval-edit [66] exist, their image sources cover only a limited set of text scenarios, failing to fully reflect the diverse text carriers and layout forms encountered in both real-world and synthetic design materials. Furthermore, these benchmarks lack a systematic evaluation of edit faithfulness and visual preservation.

- Table 20: Key attributes comparison of open-source text generation or editing benchmarks.

Benchmarks Type Size Human Filter GT Ann. Sub-class Traditial Eval. LLM Eval.

AnyText [122] Text Generation 2,000 ✗ ✗ - ✓ ✗ LongText [42] Text Generation 320 ✓ ✗ 8 ✗ ✓ CVTG-2K [33] Text Generation 2,000 ✓ ✗ 2 ✓ ✗ MARIO-Eval-edit [66] Text Edit 4,000 ✗ ✗ - ✓ ✗

TextEdit (Ours) Text Edit 2,148 ✓ ✓ 18 ✓ ✓

To comprehensively evaluate the capabilities of image editing models, we introduce TextEdit, a novel and meticulously structured benchmark. TextEdit addresses the limitations of existing text-centric editing benchmarks by providing a more systematic, fine-grained, and human-curated evaluation framework. It is distinguished by the following key advantages:

- • Human-curated Data Pipeline: Unlike prior benchmarks that rely heavily on synthetic or automatically collected data, TextEdit adopts a human-filtered pipeline to ensure high-quality, realistic editing scenarios and reduce noisy or ambiguous samples.
- • Manually Annotated Ground Truth: We provide manually edited Ground Truth (GT) images, enabling precise quantitative evaluation. This supports reliable computation of pixel-level fidelity metrics and accurate assessment of background preservation.
- • Fine-grained Scenario Taxonomy: TextEdit covers 18 diverse sub-classes of text editing scenarios, offering a more systematic and detailed evaluation compared to existing benchmarks with limited or coarse categorizations.
- • Alignment with LLM-based Instructions: The benchmark follows a pure text-instruction paradigm aligned with modern LLM-based interaction. It removes the need for auxiliary inputs such as glyph maps or segmentation masks, enabling a more natural evaluation of instruction-following ability.
- • Hybrid Evaluation Protocol: We combine classic OCR, image-fidelity metrics and modern multimodal LLM-based evaluation across target accuracy, text preservation, scene integrity, local realism and visual coherence. This dual-track protocol enables comprehensive assessment.

###### A.2. Design Details

Grounded in practical user scenarios, TextEdit bridges the gap between diverse real-world needs and model assessment through a systematic Scenario Taxonomy and a robust Evaluation Protocol. This integrated design enables a comprehensive evaluation of both technical text-editing performance and practical usability.

###### A.2.1. Scenario Taxonomy

We organize text-centric editing scenarios into two overarching domains: Virtual Scenes and Real-world Scenes. This fine-grained taxonomy, detailed with definitions and statistics in Table 20, captures the nuances of diverse text carriers, ranging from digital formats like posters, comic, slide and GUIs to real-world environments such as products, buliding, board-like media, personal accessorie, transport, watermarks and paper media. Figure 27 further illustrates the distribution of these subcategories. To construct this dataset, we curated high-quality source images aligning with these scenarios and synthesized editing instructions via a rule-based mechanism. The resulting Ground Truth images underwent rigorous manual verification to ensure visual naturalness and instruction faithfulness, establishing a reliable gold standard for quantitative evaluation.

[Figure 551]

[Figure 552]

2.7%

3.4%

Activities & Promotions Posters Product & Advertising Posters Movie & Art Posters Dialogue & Narration Onomatopoeia & Special-effects Text

6.0%

5.0% 3.0%

3.2%

10.3%

1.2%

2.5%

- 3.3%
- 3.4%

Titles & Subtitles Charts & Explanatory Text Game Interfaces Browser Interfaces

%

12.0%

6.3

6.4%

App Interfaces (Mobile/TV)

%

Product & Advertising Posters

4.2

2.1% 2.1%

1

8.9%

Objects Surface Signage Surface Board-like Media Surface Personal Accessories Surface

2.8%

7.8%

10.9% 15.8%

Transport Surface Watermarks

Paper Media Surface

- Figure 27: Data Distribution of TextEdit benchmark.

- Table 21: Unified benchmark taxonomy with data statistics.

Major ID Category (Mid) ID Specific Scene (Sub) Count

- 1.1 Poster Scenes

- 1.1.1 Activities & Promotions Posters 57
- 1.1.2 Product & Advertising Posters 74
- 1.1.3 Movie & Art Posters 107

- 1.2 Comic Scenes

- 1.2.1 Dialogue & Narration 65
- 1.2.2 Onomatopoeia & Special-effects Text 26

- 1.3 Slide / Presentation

- 1.3.1 Titles & Subtitles 71
- 1.3.2 Charts & Explanatory Text 73

- 1.4 GUI Scenes

VirtualScenes

- 1.4.1 Game Interfaces 138
- 1.4.2 Browser Interfaces 44
- 1.4.3 App Interfaces (Mobile/TV) 45
- 1.4.4 Operating-System Desktops 61

- 2.1 Objects Surface - e.g., Packages, Bottles, Boxes, Coins 168
- 2.2 Signage Surface - e.g., Building Signs, Storefronts, Billboards 339
- 2.3 Board-like Media Surface - e.g., Blackboards, Whiteboards 235
- 2.4 Personal Accessories Surface - e.g., Clothing Prints, Badges 192
- 2.5 Transport Surface - e.g., Cars, Buses, Trains, Ships 257
- 2.6 Watermarks - e.g., Photo watermarks, Brand Marks, Corner Stamps 69
- 2.7 Paper Media Surface - e.g., Papers, Books, Newspapers, Menus 127

Real-world

###### Total 2148

- Table 22: Overview of evaluation metrics. We categorize metrics into objective (Text-Centric, General) and perceptual (MLLM-based) dimensions. Detailed formulations are provided in Section A.2.2.

Category Metric What it Measures

OCR Accuracy Maximum similarity between the generated text in the target region and

the ground-truth string.

OCR Precision Accuracy of background text preservation (penalizes hallucinated or incor-

rect background text).

Classic (Text-Centric)

OCR Recall Completeness of background text preservation (penalizes missing back-

ground text). OCR F1-Score Harmonic mean of OCR Precision and OCR Recall. ROI-Aware NED Normalized Edit Distance specifically within the source text bounding box.

CLIPScore Semantic alignment between the edited image and the target caption. Aesthetic Score Visual appeal score predicted by a CLIP-based aesthetic predictor.

Classic (General)

Target Accuracy Evaluation of spelling correctness and erasure quality of the target text. Text Preservation Assessment of whether non-target background text remains intact. Scene Integrity Stability of background geometry and objects (checking for distortions). Local Realism Quality of inpainting edges, checking for artifacts like blurring or seams. Visual Coherence Harmony of font style, lighting, and texture with the original scene. MLLM Overall Avg Weighted average of the MLLM-based sub-scores (40/30/10/10/10).

MLLM-based

###### A.2.2. Evaluation Metrics

Quantitative evaluation of text editing is challenging due to the dual requirement of manipulating specific text content while strictly preserving the background. To provide a holistic assessment, we employ a hybrid evaluation strategy combining Classic Metrics and MLLM-based Metrics.

Classic Metrics focus on the existence and correctness of text. We utilize standard OCR tools to measure edit distance and detection rates. Specifically, we decouple the evaluation into Target Region (measuring editing success) and Background Region (measuring preservation capabilities). Additionally, we use CLIPScore [50] to assess general image quality and semantic alignment, along with aesthetic quality evaluation.

MLLM-based Metrics are further introduced to better capture visual nuances such as “ghosting” artifacts, lighting inconsistencies, or partial erasure. By simulating an expert forensic analysis, we use the most powerful multimodal understanding model Gemini-3-Pro [47] as judge to provide fine-grained scoring on dimensions like local realism and scene integrity, offering an evaluation that aligns closer with human preference.

Below are the detailed implementations of each evaluation metric, organized following the structure in Table 22.

###### (a) Classic Metrics (Text-Centric)

We first define a normalized similarity function 𝑆(𝑠1,𝑠2) based on the Levenshtein distance 𝐷𝑙𝑒𝑣(·,·), which forms the foundation for our text-centric metrics:

𝐷𝑙𝑒𝑣(𝑠1,𝑠2) max(|𝑠1|,|𝑠2|,1)

(5)

𝑆(𝑠1,𝑠2) = 1 −

where 𝑠1 and 𝑠2 denote two comparison strings, and |𝑠1|,|𝑠2| represent their respective lengths. The term max(|𝑠1|,|𝑠2|,1) serves as a normalization factor to ensure the denominator is non-zero, bounding the similarity score 𝑆 within the range [0,1], where 1 indicates a perfect match.

- (i) OCR Accuracy. This metric evaluates whether the target text is correctly rendered in the editing region. Let

𝒯𝑔𝑒𝑛 be the set of text strings detected in the generated image that significantly overlap with the target editing region (Intersection over Union, IoU > 0.5). The accuracy is defined as the maximum similarity between the

detected texts and the ground truth target text 𝑡𝑡𝑔𝑡, adjusted by a penalty factor: Acc = max

###### 𝑆(𝑡,𝑡𝑡𝑔𝑡) × P𝑓𝑎𝑖𝑙 (6)

𝑡∈𝒯𝑔𝑒𝑛

Here, 𝑡 represents a candidate string within the set 𝒯𝑔𝑒𝑛. The term P𝑓𝑎𝑖𝑙 is a penalty coefficient designed to punish failed editing attempts. Specifically, P𝑓𝑎𝑖𝑙 is set to 0.2 if the original source text 𝑡𝑠𝑟𝑐 is still detected in the region while the target text 𝑡𝑡𝑔𝑡 is absent; otherwise, P𝑓𝑎𝑖𝑙 = 1.0.

- (ii) OCR Precision. This metric measures the accuracy of background text preservation, penalizing hallucinated or incorrect background text. For each detected text item 𝑡𝑖 in the background region (those with IoU < 0.5

with the target region), we find its best match among the original background texts 𝒯𝑏𝑔𝑜𝑟𝑖𝑔 from the source image:

Precision =

1 |𝒯𝑏𝑔𝑔𝑒𝑛|

∑︁

𝑡∈𝒯𝑏𝑔𝑔𝑒𝑛

max

𝑡′∈𝒯𝑏𝑔𝑜𝑟𝑖𝑔

𝑆(𝑡,𝑡′) (7)

where 𝒯𝑏𝑔𝑔𝑒𝑛 denotes the set of background text detected in the generated image, and |𝒯𝑏𝑔𝑔𝑒𝑛| is its cardinality. This metric achieves a high score when all detected background texts closely match the original ones, thus

avoiding spurious text generation.

- (iii) OCR Recall. This metric assesses the completeness of background text preservation, penalizing missing

background text. For each original background text 𝑡′ in 𝒯𝑏𝑔𝑜𝑟𝑖𝑔, we measure how well it is preserved in the generated image:

Recall =

1 |𝒯𝑏𝑔𝑜𝑟𝑖𝑔|

∑︁

𝑡′∈𝒯𝑏𝑔𝑜𝑟𝑖𝑔

max

𝑡∈𝒯𝑏𝑔𝑔𝑒𝑛

𝑆(𝑡,𝑡′) (8)

A high recall indicates that most of the original background texts are successfully retained in the edited image without being accidentally removed or altered.

- (iv) OCR F1-Score. The F1-score provides a balanced metric by computing the harmonic mean of OCR Precision and OCR Recall:

F1 = 2 ×

Precision × Recall Precision + Recall

(9)

This unified metric captures both the accuracy and completeness of background text preservation, offering a comprehensive assessment of text-level editing quality.

- (v) ROI-Aware NED. This metric strictly evaluates the editing quality within the specific Region of Interest (ROI) defined by the original source text bounding box. We calculate the similarity between the predicted text

string 𝑡𝑝𝑟𝑒𝑑 extracted from the ROI and the target string 𝑡𝑡𝑔𝑡. A failed erasure penalty is applied if the residual similarity to the source text remains high:

###### NED = 𝑆(𝑡𝑝𝑟𝑒𝑑,𝑡𝑡𝑔𝑡) × I𝑟𝑒𝑠𝑖𝑑 (10)

where 𝑡𝑝𝑟𝑒𝑑 denotes the OCR recognition result directly cropped from the ROI. I𝑟𝑒𝑠𝑖𝑑 is a residual indicator function acting as a penalty term: I𝑟𝑒𝑠𝑖𝑑 = 0.2 if 𝑆(𝑡𝑝𝑟𝑒𝑑,𝑡𝑠𝑟𝑐) > 0.9 (implying the source text 𝑡𝑠𝑟𝑐 was not effectively erased), and I𝑟𝑒𝑠𝑖𝑑 = 1.0 otherwise. This metric ensures that the target region contains the correct new text while completely removing the original text.

###### (b) Classic Metrics (General)

- (i) CLIPScore. We use CLIPScore [50] to measure the semantic alignment between the predicted edited image and the caption text. First, we employ Qwen3-VL [4] to generate a concise caption text for the labeled GT image, with a length that satisfies the input constraints of the CLIP text encoder. Let v𝑖𝑚𝑔 denote the CLIP visual embedding of the predicted edited image, and v𝑡𝑒𝑥𝑡 denote the CLIP text embedding of the generated caption. The CLIPScore is computed as the cosine similarity:

v𝑖𝑚𝑔 · v𝑡𝑒𝑥𝑡 |v𝑖𝑚𝑔| · |v𝑡𝑒𝑥𝑡|

CLIPScore =

(11)

A higher CLIPScore indicates better semantic consistency between the visual content and the textual description, reflecting successful integration of the edited text into the scene context.

- (ii) Aesthetic Score. We use a CLIP-based aesthetic predictor to evaluate the overall visual appeal of the generated

image. Let 𝑓aes(·) denote the aesthetic prediction model that maps a CLIP image embedding to an aesthetic quality score:

AesScore = 𝑓aes(v𝑖𝑚𝑔) (12)

The aesthetic score typically ranges from 1 to 5 (or 0 to 1 after normalization), where higher values indicate better visual quality, composition, and perceptual appeal. This metric helps ensure that text editing does not degrade the overall image quality.

###### (c) MLLM-based Metrics

We utilize a strong commercial MLLM (i.e. Gemini-3-Pro-Preview [47]) to simulate expert forensic analysis across 𝑁 = 5 dimensions (𝐷1 to 𝐷5). Each dimension is evaluated on a Likert scale from 1 to 5, where higher scores indicate better quality. The raw scores are then normalized and aggregated as follows.

- (i) Target Accuracy (𝐷1). This dimension evaluates the spelling correctness and erasure quality of the target text in the edited region. The MLLM assigns a score 𝑠1 ∈ [1,5] based on:

- • Whether the target text is rendered with correct spelling
- • Whether the original source text has been completely erased
- • The visual clarity and legibility of the new text

- A score of 5 indicates perfect text replacement with no residual artifacts from the source text.

- (ii) Text Preservation (𝐷2). This dimension assesses whether non-target background text remains intact and unaffected by the editing operation. The MLLM evaluates:

- • Completeness of background text retention
- • Absence of unintended modifications to surrounding text
- • Preservation of text layout and positioning

- The score 𝑠2 ∈ [1,5] reflects the degree to which the model successfully preserves the original background text elements.

(iii) Scene Integrity (𝐷3). This dimension measures the stability of background geometry and objects, checking for structural distortions or artifacts introduced by the editing process. The evaluation criteria include:

- • Preservation of architectural elements, object boundaries, and spatial relationships
- • Absence of geometric distortions or warping effects
- • Maintenance of scene perspective and depth cues

- The score 𝑠3 ∈ [1,5] indicates how well the overall scene structure is maintained.

(iv) Local Realism (𝐷4). This dimension evaluates the quality of inpainting in the edited region, focusing on:

- • Edge cleanness and seamlessness between edited and original regions
- • Absence of visible artifacts such as blurring, ghosting, or seams
- • Natural integration of the new text with its immediate surroundings

- The score 𝑠4 ∈ [1,5] reflects the photorealistic quality of the local editing region.

- (v) Visual Coherence (𝐷5). This dimension assesses the harmony of font style, lighting, and texture with the original scene context. The MLLM evaluates:

- • Font style consistency with surrounding text or scene aesthetics
- • Lighting direction, intensity, and color temperature matching
- • Shadow casting and texture patterns that blend naturally with the background

- The score 𝑠5 ∈ [1,5] measures how well the edited text integrates into the overall visual style of the image.

- (vi) Score Normalization and Cutoff Mechanism. The raw Likert scores 𝑠𝑖 ∈ [1,5] for each dimension 𝑖 are first

normalized to 𝑠′𝑖 ∈ [0,1] using the mapping: 𝑠′𝑖 = (𝑠𝑖 − 1)/4. To ensure evaluation validity, we implement a Cutoff Mechanism: if the primary editing task (Target Text Accuracy, 𝐷1) fails significantly (i.e., 𝑠1 < 4), the scores for secondary dimensions are penalized to zero, as a failed text edit renders other quality assessments meaningless. The final weighted score 𝑉𝑠𝑐𝑜𝑟𝑒 is formulated as:

𝑉𝑠𝑐𝑜𝑟𝑒 = 𝑤1𝑠′1 + I(𝑠

1≥4) ·

∑︁5

𝑖=2

𝑤𝑖𝑠′𝑖 (13)

In this formulation:

- • 𝑠′𝑖 denotes the normalized score for the 𝑖-th dimension.
- • 𝑤𝑖 represents the weight assigned to the 𝑖-th dimension, satisfying

∑︀5

𝑖=1 𝑤𝑖 = 1. By default, we use 𝑤1 = 0.4, 𝑤2 = 0.3, 𝑤3 = 0.1, 𝑤4 = 0.1, 𝑤5 = 0.1.

- • 𝑠1 is the raw score for the primary text accuracy dimension.
- • I(𝑠

1≥4) is an indicator function that equals 1 if the condition 𝑠1 ≥ 4 is met, and 0 otherwise. This ensures that if the textual content is incorrect (score < 4), the visual quality of the background (dimensions 𝐷2 to 𝐷5) does not contribute to the final score.

- (vii) MLLM Overall Average. The overall MLLM-based metric is defined as the weighted average across all five dimensions with the cutoff mechanism applied:

∑︁5

MLLM Overall Avg = 𝑉𝑠𝑐𝑜𝑟𝑒 = 𝑤1𝑠′1 + I(𝑠

𝑤𝑖𝑠′𝑖 (14)

1≥4) ·

𝑖=2

This comprehensive metric provides a single scalar value that captures both the primary text editing success and the secondary visual quality factors, with appropriate penalties for fundamental editing failures. The metric is computed separately for Virtual (synthetic scenes, category 1.x.x) and Real (real-world scenes, category 2.x) subsets of the benchmark to enable fine-grained performance analysis across different scene types.

System Prompt for T2I task

You are an expert Forensic Image Analyst and Design QA Specialist. Your task is to evaluate the quality of an AI-edited image by comparing three images. Images Provided (in order):

- 1. Original Image: The unedited source image containing the text “{raw_text}”.
- 2. Ground Truth Image: A human-created reference showing the ideal result with text “{target_text}”.
- 3. Edited Image: The AI-generated result to be evaluated. Editing Task Information:

- • Text to Remove: “{raw_text}”
- • Text to Add: “{target_text}” EVALUATION RUBRIC (1-5 SCORING SYSTEM) Please evaluate the Edited Image based on the following 5 dimensions. Use the strict criteria below to assign a score from 1 to 5.

Text Accuracy

- Q1. [Target Text Accuracy] Focus: Spelling, erasure correctness, and legibility of “{target_text}”.

- • 5 (Perfect): Exact spelling match (case-sensitive). Old text completely erased. No ghosting.
- • 4 (Minor Flaw): Text is correct but has 1 character error/typo, OR slight casing issue, OR extremely faint ghosting visible only on close inspection.
- • 3 (Readable but Flawed): 2–3 character errors but word is recognizable. OR visible ghosting/remnants of old text that affect cleanness.
- • 2 (Major Error): >3 character errors (misspelled heavily). OR old text is still clearly readable (failed erasure).
- • 1 (Failed): Text is missing, gibberish, or completely wrong word. Old text remains fully intact.

Text Preservation

- Q2. [Non-Target Text Preservation] Focus: Preservation/legibility of background text other than the edited target.

- • 5 (Perfect): All non-target text is 100% preserved and legible, identical to Original/GT.
- • 4 (Good): Main background text is preserved. Minor distant text is slightly softened/blurred but still readable.
- • 3 (Fair): One or two secondary text elements are blurred, damaged, or missing.
- • 2 (Poor): Critical nearby text (directly adjacent to target) is damaged, erased, or hallucinated.
- • 1 (Destructive): Widespread destruction or hallucination of background text.

Scene Integrity

- Q3. [Global Scene Integrity] Focus: Geometric stability of non-edited areas (background, objects, people).

- • 5 (Perfect): Pixel-perfect preservation of background geometry. No distortions.
- • 4 (Good): Almost perfect, but very minor shift (<1%) in background lines or perspective.
- • 3 (Noticeable): Visible distortion in straight lines (wavy), or slight warping of objects/faces.
- • 2 (Severe): Major structural damage (e.g., a person’s face is melted, a building collapsed).
- • 1 (Chaos): The scene structure is completely changed or nonsensical compared to Original.

Local Realism

- Q4. [Local Realism & Artifacts] Focus: Inpainting quality, edge cleanliness, and seamlessness around the edited area.

- • 5 (Excellent): Invisible edit. Clean edges, no halos, no smudges. Professional quality.
- • 4 (Good): Very minor artifacts (e.g., slight pixelation on zoom-in), but looks natural at a glance.
- • 3 (Fair): Visible seams, blurry rectangular patch, or “smudged” look around the text.
- • 2 (Poor): Obvious artifacts, messy edges, or white/black box artifacts.
- • 1 (Garbage): The edited area looks like a corrupted file or pure noise.

- Figure 28: The system prompt template used for MLLM-based automated evaluation, covering the analyst persona, task definition, and the first four scoring dimensions (Q1–Q4).

Visual Coherence

Q5. [Aesthetic & Lighting Harmony] Focus: Style matching (font), lighting, shadow, and texture harmony.

- • 5 (Seamless): Font style matches the GT/Context perfectly. Lighting/shadows are physically correct. Texture (grain) matches the photo.
- • 4 (Integrated): Good style match. Lighting is mostly correct. Texture is slightly too smooth but acceptable.
- • 3 (Artificial): Text looks “pasted on” (digital sticker look). Font style is generic (e.g., Arial) and clashes with the scene.
- • 2 (Disjointed): Wrong color, wrong perspective, or no shading where needed.
- • 1 (Mismatch): Text floats awkwardly, completely ignoring the scene’s physics and style.

Final Output Format

##### FINAL OUTPUT FORMAT (JSON ONLY)

You must output a valid JSON object containing two dictionaries:score (integers) and reason (strings). Example Output: Instruction: "Replace the text 'MUSIC' with 'PARTY'."

[Figure 553]

{

"score": {

- "Q1": 5,
- "Q2": 1,
- "Q3": 2,
- "Q4": 5,
- "Q5": 4

}, "reason": {

- "Q1": "The target text 'PARTY' is spelled correctly and

is clearly legible. The specific target text to remove ( 'MUSIC' ) is completely gone with no ghosting.",

- "Q2": "The model caused widespread destruction of non-target text. 'NIGHT CLUB' , '31 OCT' , 'FREE DRINKS' , 'LIVE' , and 'PRICE' were all erroneously removed, and '10$' was corrupted into the hallucinated text '1TY' .",
- "Q3": "Global scene integrity is severely compromised. The skeleton's arm holding the maraca was erased, leaving the maraca floating in mid-air, which breaks the physical logic of the illustration.",
- "Q4": "Despite the semantic failures, the technical quality of the image is excellent. The edges are sharp, the background inpainting is smooth, and there are no visible pixel artifacts, blur, or noise.",
- "Q5": "The font style selected for 'PARTY' integrates well with the hand-drawn vector aesthetic of the poster, although the color is a darker maroon compared to the bright red of the original text."

Original Image

[Figure 554]

} }

Generated Image

Do not output any markdown or conversational text outside the JSON block.

- Figure 29: The continuation of the evaluation prompt, detailing the final scoring dimension (Q5) and the strict JSON output schema required for parsing results.

###### A.3. MiniSet-500 Results

To provide a lightweight and standardized evaluation subset for the open-source community, we construct a MiniSet-500 from the full TextEdit benchmark. It is built by randomly sampling instances from each of the 18 subcategories to ensure a balanced distribution across scenario types. The MiniSet-500 contains a total of 500 image editing pairs, preserving the diversity of tasks while significantly reducing evaluation cost. It serves as an efficient protocol for rapid benchmarking and ablation studies, while the full benchmark remains the standard for comprehensive evaluation. In Table 23 and Table 24, we report the performance of various models on the MiniSet-500 TextEdit benchmark.

- Table 23: Evaluation of text-centric image editing on TextEdit MiniSet-500 (Classic Metrics). Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. “Real” refers to source images from real-world scenes, while “Virtual” refers to images from synthetic virtual. Abbreviations: OA=OCR Accuracy, OP=OCR Precision, OR=OCR Recall, F1=OCR F1-Score, NED=ROI-Aware NED, CLIP=CLIPScore, AES=Aesthetic Score.

|Models|# Params|Real Virtual OA OP OR F1 NED CLIP AES OA OP OR F1 NED CLIP AES<br><br>| |
|---|---|---|---|
| | | | |

Generation Models

Qwen-Image-Edit [136] 20B 0.76 0.69 0.67 0.67 0.70 0.75 5.81 0.74 0.71 0.70 0.70 0.70 0.80 5.27 GPT-Image-1.5 [98] - 0.72 0.68 0.66 0.67 0.67 0.75 5.85 0.68 0.69 0.68 0.68 0.65 0.80 5.32 Nano Banana Pro [29] - 0.76 0.71 0.69 0.70 0.70 0.75 5.86 0.77 0.76 0.75 0.75 0.76 0.81 5.32 Unified Models

Lumina-DiMOO [145] 8B 0.20 0.22 0.18 0.19 0.19 0.70 5.58 0.22 0.25 0.21 0.22 0.19 0.73 4.87 Ovis-U1 [124] 2.4B+1.2B 0.37 0.34 0.32 0.32 0.33 0.72 5.39 0.39 0.41 0.38 0.39 0.33 0.74 4.75 BAGEL [31] 7B+7B 0.61 0.59 0.52 0.54 0.54 0.74 5.79 0.53 0.58 0.53 0.55 0.51 0.78 5.25 InternVL-U 2B+1.7B 0.77 0.74 0.70 0.71 0.71 0.76 5.79 0.74 0.72 0.69 0.70 0.72 0.79 5.14

- Table 24: Evaluation of text-centric image editing on TextEdit MiniSet-500(MLLM-based Metrics). Sizes of unified models in “A + B” indicate separate understanding (A) and generation (B) parameters. “Real” refers to source images from real-world scenes, while “Virtual” refers to images from virtual scene. Abbreviations: TA: Text Accuracy, TP: Text Preservation, SI: Scene Integrity, LR: Local Realism, VC: Visual Coherence, Avg: MLLM Overall Average.

|Models|# Params|Real Virtual TA TP SI LR VC Avg TA TP SI LR VC Avg<br><br>| |
|---|---|---|---|
| | | | |

Generation Models

Qwen-Image-Edit [136] 20B 0.93 0.85 0.77 0.55 0.78 0.80 0.60 0.82 0.91 0.81 0.74 0.76 GPT-Image-1.5 [98] - 0.97 0.94 0.86 0.79 0.92 0.91 0.85 0.93 0.95 0.92 0.83 0.88 Nano Banana Pro [29] - 0.96 0.95 0.85 0.86 0.92 0.91 0.87 0.92 0.96 0.93 0.87 0.92 Unified Models

Lumina-DiMOO [145] 8B 0.16 0.04 0.04 0.02 0.06 0.08 0.02 0.05 0.19 0.07 0.03 0.10 Ovis-U1 [124] 2.4B+1.2B 0.29 0.11 0.11 0.08 0.20 0.17 0.04 0.16 0.35 0.18 0.15 0.22 BAGEL [31] 7B+7B 0.68 0.61 0.38 0.34 0.59 0.53 0.36 0.52 0.69 0.64 0.40 0.54 InternVL-U 2B+1.7B 0.94 0.91 0.72 0.73 0.75 0.89 0.88 0.87 0.90 0.78 0.57 0.79

- B. Data Construction Details In this section, we provide details of our data construction process.

- B.1. Filtering details for General Science Image Generation The second stage of filtering is based on the following dimensions:

- 1. Image Type: images are categorized into one of the 30 image types defined in MMMU [156], e.g. Posters, Diagrams, Screenshots. Certain image types are removed depending on the subject, like Microscopic_Images for biology or Tables for all subjects.
- 2. Subject: images are classified into a list of give subjects and removed if they belong to subjects like Literature, Art, Design, etc., since those images are usually natural images like paintings or photographs.
- 3. Text Length: all text in the images is extracted. Images with text longer than 200 characters are removed.
- 4. Image Complexity: images are rated in terms of their complexity from 1 (very complex) to 10 (very simple), i.e. whether the image is too complex to draw, considering the number and complexity of components, objects, and text. We set different ranges (e.g. 5-7) for each subject to filter images of moderate complexity.
- 5. Subject Knowledge Density: similar to Image Complexity, images are rated by whether the model needs subject knowledge and reasoning to generate them, from 1 (least knowledge) to 10 (very dense knowledge). Different ranges are also applied for each subject, e.g. 7-8.

###### B.2. Chemistry Text-to-Image Data Synthesis

In addition to the filtered general science text-to-image data from existing datasets and the Internet, we also devised an automated pipeline to acquire large-scale complex organic compound data for chemistry images. In the initial phase, we employed automated acquisition protocols to harvest 800,000 raw entries from PubChem, a publicly accessible and authoritative chemical repository. This raw corpus encompasses essential physicochemical descriptors, including unique Compound IDs (CIDs), chemical nomenclature, molecular formulas, SMILES representations, and 2D structural diagrams. Subsequently, we implemented a rigorous filtration mechanism to sanitize the dataset by eliminating erroneous or incomplete entries, yielding a curated set of 600,000 high-fidelity chemical instances. In the second phase, we focused on the construction of instruction-following data. By establishing diverse QA templates that integrate chemical names, SMILES strings, and their corresponding 2D visualizations, we successfully synthesized 600,000 visual-textual QA pairs specifically optimized for chemical image generation tasks.

###### B.3. Computer Science Editing The definitions of each task are given below:

- 1. Tree Topology Editing & Node Manipulation: perform completion (complete into a full binary tree), insertion (insert a new node at a specific position) or pruning (delete a node and its subtree or nodes within a specific region) on a tree.
- 2. Tree Traversal Visualization: visualize the traversal path or label the traversal order of a binary tree, including pre-order, in-order and post-order.
- 3. Binary Search Tree (BST) Operations: insert a node into a BST, or validate a BST and fix the swapped nodes.
- 4. Heap Operations & Dual Views: perform 1-3 steps of insert or extract-root operations on heaps and visualize the memory array. Sift-up and sift-down processes should be included.
- 5. Huffman Coding Tree: given the frequency of characters, construct a Huffman coding tree or perform node merging.
- 6. Tree Lowest Common Ancestor (LCA) & Path Highlighting: given two nodes, highlight their LCA or draw the connected path between them.

- 7. Graph 𝑘-hop Neighborhood: given a central node, visually label all nodes with an exact distance of 𝑘.
- 8. Graph Degree Identification: identify and box all nodes whose degree equals a specific value.
- 9. Graph Cycle Detection: depict the unique simple cycle in the graph.
- 10. Bipartite Graph Coloring: color an uncolored bipartite graph with two colors with the constraint that adjacent nodes must have different colors.
- 11. Graph Shortest Path: given a start node and an end node, draw the shortest connecting path between the two nodes.
- 12. Directed Graph Reachability: in a directed graph, box the complete set of downstream nodes reachable in the forward direction from the source node.
- 13. FSM String Trace: given an input string, draw the complete state transition path in the FSM.
- 14. FSM State Role Identification: identify and color the start state and accepting states in the FSM.
- 15. FSM Transition Logic Completion: complete the missing arrowed edge and its input label in the incomplete FSM.

- B.4. Solid Geometry Implementations of each task are as follows:

- 1. Solid of Revolution: We first construct a family of 3D solids of revolution using GeoGebra, parameterized by three possible rotation axes (x, y, or z). Taking the z-axis as an example, we generate a planar polygon in the xoz plane whose one edge lies on the rotation axis. Specifically, we fix the y-coordinate to 0 and sample a sequence of vertices with monotonically increasing integer z-coordinates. The x-coordinates are sampled as random integers that are either strictly positive (polygon lies on the positive x half-axis) or strictly negative (polygon lies on the negative x half-axis). The first and last vertices are constrained to lie exactly on the z-axis (x = 0), ensuring that one side of the polygon coincides with the rotation axis. The number of vertices in the polygon is randomly chosen in the range [3,6]. This polygon then serves as the generator curve (meridian), which defines the rotational surface when swept around the specified axis.
- 2. Plane Symmetry: Plane symmetry is also implemented with GeoGebra. To maintain visually pleasing layouts, we restrict the symmetry plane to be perpendicular to the xoy plane and ensure that the original solid lies entirely on one side of this plane prior to reflection. To enrich the diversity of the dataset, we support multiple configurable geometric primitives, including regular prisms, regular pyramids, cylinders, cones, and spheres. For prisms and pyramids, the number of edges of the regular base polygon is sampled in the range [3,6], and the edge length is configurable. In addition, both the color of the solid and the color of the symmetry plane are parameterized, allowing us to generate samples with varied appearance while preserving geometric clarity and visual aesthetics.
- 3. Point Symmetry: For point symmetry, we implement two parallel sample generation pipelines based on GeoGebra and matplotlib, respectively. In both pipelines, the type of geometric primitive (e.g. prisms, pyramids, cylinders, cones, spheres) and its color are randomly configurable, enabling a wide variety of instance appearances. To further increase the diversity of the dataset, we apply a random initial rotation to each base solid and allow the camera viewpoint to be varied across samples.
- 4. Solid Translation: The translation task is implemented using a matplotlib-based rendering pipeline, in which the type and color of the 3D solids are fully configurable. To keep the visual effect both aesthetically pleasing and controllable, we restrict the translation vector to three canonical configurations: along the x-axis, y-axis, or z-axis. For each sample, the translation magnitude is randomly chosen as an integer in the range [4,10], which introduces sufficient variation in the relative positions of the original and translated solids while ensuring that both remain clearly visible within the same view.
- 5. Solid Projection: The projection task is implemented in matplotlib, with the type and color of 3D solids randomly sampled. Each solid is first rotated by a configurable angle to avoid axis-aligned degeneracies, and then orthographically projected onto the xoy plane.

