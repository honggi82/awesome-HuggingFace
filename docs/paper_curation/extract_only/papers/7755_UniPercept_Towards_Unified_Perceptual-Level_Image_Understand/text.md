###### UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture

## arXiv:2512.21675v1[cs.CV]25Dec2025

Shuo Cao1,2,∗,♢, Jiayang Li3,∗, Xiaohui Li2,4, Yuandong Pu2,4, Kaiwen Zhu2,4 Yuanting Gao5, Siqi Luo2,4, Yi Xin2,6, Qi Qin2, Yu Zhou7 Xiangyu Chen8, Wenlong Zhang2, Bin Fu2, Yu Qiao2, Yihao Liu2,†

1 University of Science and Technology of China 2 Shanghai AI Laboratory 3 Peking University 4 Shanghai Jiao Tong University 5 Tsinghua University 6 Nanjing University 7 Sun Yat-sen University 8 Tele-AI Website https://thunderbolt215.github.io/Unipercept-project Code https://github.com/thunderbolt215/UniPercept Benchmark & Checkpoint https://hf.co/collections/Thunderbolt215215/unipercept

[Figure 1]

[Figure 2]

[Figure 3]

Q: On a scale, how strong is the Visual Elements & Structure displayed here? Higher level reflects more skillful use of color, geometry, and

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Q: How strongly do distortions impact the perception of this image?

A: Slight (barely noticeable but present)

[Figure 8]

Q: How does the image maintain viewer engagement through its use of lighting and composition? A: By emphasizing the beam saber's glow

Q: Which element in the image best exemplifies compositional rhythm through repetition and spacing? A: Overhead lights pattern

[Figure 9]

Q: How does the

A: It

[Figure 10]

spatial organization for coherent

Q: Is the license plate of the car obscured due to motion blur in the image? A: Yes

overexposure in the image affect the visibility of the central signage?

obscures the text details with glare.

contrast and structure. A: High

[Figure 11]

[Figure 12]

Overall

Gestalt

Originality & Creativity

Q: Does the image exhibit a distinctive concept or narrative beyond typical icy fantasy landscapes? A: Yes

Q: Which part of the image shows the most noticeable motion blur?

Theme & Communication

Distortion Location

A: The people seated on the

roller coaster

Technical Execution

[Figure 13]

Q: Please judge the image’s aesthetics and output a score from 0 to 100. A: 59

[Figure 14]

[Figure 15]

# IAA IQA ISTA

[Figure 16]

[Figure 17]

Distortion Severity

Visual Elements & Structure

Q: What distortion types are evident in the image of the windmill? A: Noise / Speckle

Q: Why does the image evoke a tranquil emotional tone despite its lack of vibrant colors? A: The mist softens contrast, creating a serene atmosphere.

[Figure 18]

Composition

[Figure 19]

Q: Provide a quality rating for this image between 0 and 100. A: 83

Distortion Type

[Figure 20]

& Design

Material

Representation

Physical Structure

[Figure 21]

Semantic Perception Geometric

Composition

Q: What area of the image

shows the most noticeable distortion? A: Cat's fur texture

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Q: Please rate the structure and texture richness of this

Q: What stylistic classification best describes the church's architectural style? A: Classical Architecture

Q: How can the 2D contour

Q: What describes the surface behavior of the plant's ceramic pot? A: Glossy and smooth finish

Q: Which material

Q: What 3D volumetric form best describes the mountain slope's contour?

of the red mesh structure be classified based on its repeating shape? A: Diamond

category best describes the surface of the chequered floor tiles? A: Polished Stone

image and provide a score

between 0 and 100. A: 64

A: Paraboloid

Figure 1. Overview of UniPercept-Bench. We propose a unified benchmark for perceptual-level image understanding, covering three perceptual domains: Image Aesthetics Assessment (IAA), Image Quality Assessment (IQA), and Image Structure and Texture Assessment (ISTA). Each domain is organized hierarchically into Domain–Category–Criterion levels, enabling fine-grained perceptual evaluation. The benchmark contains both Visual Rating (VR) and Visual Question Answering (VQA) tasks, facilitating comprehensive assessment of models’ abilities to understand and reason about perceptual-level image understanding.

1

###### Abstract

Human visual perception involves much more than object recognition: it includes nuanced judgments about how an image looks and feels. Such perceptual cues play a crucial role in many downstream applications (e.g., content creation, image enhancement, and generative model alignment). Nevertheless, current MLLMs often struggle with these aspects, producing unstable or inconsistent predictions when evaluating aesthetic quality, perceptual degradation, or structural coherence. This gap highlights the need for a unified framework that can explicitly define, evaluate, and improve perceptual-level understanding in MLLMs, as perceptual attributes remain far less standardized and less explored than semantic ones. Addressing this missing layer is essential for building models that align more closely with human judgments, thereby achieving higher visual quality.

Multimodal large language models (MLLMs) have achieved remarkable progress in visual understanding tasks such as visual grounding, segmentation, and captioning. However, their ability to perceive perceptual-level image features remains limited. In this work, we present UniPerceptBench, a unified framework for perceptual-level image understanding across three key domains: Aesthetics, Quality, Structure and Texture. We establish a hierarchical definition system and construct large-scale datasets to evaluate perceptual-level image understanding. Based on this foundation, we develop a strong baseline UniPercept trained via Domain-Adaptive Pre-Training and Task-Aligned RL, enabling robust generalization across both Visual Rating (VR) and Visual Question Answering (VQA) tasks. UniPercept outperforms existing MLLMs on perceptuallevel image understanding and can serve as a plug-andplay reward model for text-to-image generation. This work defines Perceptual-Level Image Understanding in the era of MLLMs and, through the introduction of a comprehensive benchmark together with a strong baseline, provides a solid foundation for advancing perceptual-level multimodal image understanding.

[Figure 28]

###### Semantic-level Understanding

[Figure 29]

###### Image Captioning

Q: Generate a natural-language description of the scene. A: The image shows a grand cathedral with intricate spires and stone carvings under a clear blue sky, surrounded by people in the open square.

[Figure 30]

###### Object Visual Reasoning

Q: What are the people looking at near the shore? A: The sunset over the ocean.

Interpreting Context & Meaning

[Figure 31]

###### Perceptual-level Understanding

[Figure 32]

###### Aesthetics Assessment

Q: Evaluate visual balance. A: The central placement of the cat is a key factor in achieving visual balance in this image. The cat acts as the focal point, drawing immediate attention and anchoring ...

###### 1. Introduction

###### Structure Assessment

[Figure 33]

Recent years have witnessed the rapid advancement of multimodal large language models (MLLMs), which now achieve impressive performance across a variety of vision–language tasks including segmentation, visual grounding, image captioning, and visual reasoning [1, 31, 46]. These advancements are largely driven by their strong capability to learn and align semantic-level representations, allowing models to identify objects and scenes, capture their relationships and perform visual reasoning [3, 52, 75].

Q: Assess surface texture. A: The image depicts a surface with a crumpled texture, characterized by irregular folds, creases, and a matte finish. The characteristics ...

Perceiving Appearance & Quality

Figure 2. Semantic-level vs. Perceptual-level understanding.

To bridge this gap, we propose UniPercept, the first unified framework for perceptual-level image understanding across three core domains: Image Aesthetics Assessment (IAA), Image Quality Assessment (IQA), and Image Structure and Texture Assessment (ISTA). Our contributions are summarized as follows:

However, despite extensive progress on semantic understanding, the perceptual-level comprehension of images—how humans perceive aesthetics, quality, structure, and texture—remains substantially underexplored. As illustrated in Fig. 2, semantic-level tasks focus on high-level interpretation of visual entities (e.g., object attributes or contextual reasoning), whereas perceptual-level tasks require assessing fine-grained, low-level visual appearance, such as aesthetic harmony, degradation severity, structural regularity, or surface texture. These perceptual attributes are inherently subtle, often subjective, and closely tied to human visual experience, making them fundamentally different from typical semantic-level tasks.

• UniPercept-Bench. We establish a comprehensive hierarchical taxonomy of perceptual attributes, consisting of three progressive layers: Domain–Category–Criterion. Building upon this taxonomy, we construct UniPerceptBench, a systematically designed benchmark for evaluating perceptual-level image understanding in MLLMs. The benchmark covers fine-grained perceptual attributes and supports both Visual Rating (VR) and Visual Question Answering ( VQA) tasks for unified perceptual-level image understanding.

∗Equal contribution. ♢This work was done during his internship at Shanghai AI Laboratory. †Corresponding author.

|I A A|Composition & Design|Emotion & Viewer Response|Technical Execution|
|---|---|---|---|
| |Q: What structural element guides the viewer's eye in the image’s spatial organization?<br><br>A. Curved lines of rocks<br>B. Horizontal water surface<br>C. Vertical rock formations<br>D. Diagonal shoreline line<br><br><br>[Figure 34]|Q: Does the ethereal glow of the antlers efectively enhance the emotional resonance of a mystical atmosphere?<br><br>A. Yes<br><br>B. No<br><br><br>[Figure 35]|Q: Which element most efectively conveys cultural insight in the artwork?<br><br>A. Samurai armor and stance<br><br>B. Monochromatic color palette<br>C. Towering trees and cliffs<br>D. Cascading lightning in background<br><br><br>[Figure 36]|
| |Criterion: Structural Organization QA Template: What|Criterion: Emotional Resonance QA Template: Yes-No|Criterion: Cultural Insight QA Template: Which|
|I Q A|Distortion Location|Distortion Severity|Distortion Type|
| |Q: What spatial region in the image shows the most motion blur?<br><br>A. Basketbal in mid-air<br><br>B. Substitutes sitting on chairs<br>C. Floor near the green towel<br>D. Wall in the background<br><br><br>[Figure 37]|Q: To what degree are distortions present throughout the image?<br><br>A. None (no visible distortion)<br>B. Slight (barely noticeable but present)<br>C. Obvious (clearly visible and significantly impacts perception)<br><br><br>[Figure 38]|Q: How does pixelation afect the texture visibility on the tomato surfaces?<br><br>A. Smooths textures uniformly<br>B. Creates blocky visual appearance<br><br>C. Enhances edge definition<br>D. Introduces color bleeding<br><br><br>[Figure 39]|
| |Criterion: Location Description QA Template: What|Criterion: Severity Level QA Template: Level Prediction|Criterion: Types Present QA Template: How|
|I<br><br>S<br>T A<br>|Geometric Composition|Material Representation|Semantic Perception|
| |Q: How can the 2D contour of the chequered fabric patern be classified?<br><br>A. As a series of circles<br>B. As a grid of squares<br><br>C. As a wave-like pattern<br>D. As a series of triangles<br><br><br>[Figure 40]|Q: What is the primary material of the spray paint cans visible in the scene?<br><br>A. Metal<br><br>B. Fabric<br>C. Ceramic<br>D. Glass<br><br><br>[Figure 41]|Q: Which component’s surface texture most suggests a reflective functional implication?<br><br>A. Sky, smooth, light blue<br>B. Clouds, bubbly, white<br>C. Sea, smooth, gradient teal-slate<br><br>D. Beach, matte, brown<br><br><br>[Figure 42]|
| |Criterion: 2D Contour QA Template: How|Criterion: Material Identification QA Template: What|Criterion: Functional Suggestion QA Template: Which|

- Figure 3. Representative QA examples in UniPercept-Bench. Questions follow a three-level hierarchy of Domain–Category–Criterion, defining perceptual scope, specific visual aspects, and fine-grained criteria for constructing diverse, perception-oriented VQA tasks.

###### 2. Related Works

- • UniPercept. We develop UniPercept as a strong baseline MLLM through large-scale Domain-Adaptive PreTraining and task-aligned reinforcement learning. Without relying on additional human feedback, the model learns to reliably assess perceptual attributes across diverse visual domains. UniPercept demonstrates strong generalization across both VR and VQA tasks, and achieves consistent gains in all three perceptual domains (IAA, IQA, and ISTA), substantially outperforming state-of-the-art generalized and specialized MLLMs.
- • Applications of UniPercept. UniPercept serves as a strong plug-and-play reward model for post-training T2I models [32], enabling direct optimization of perceptuallevel signals such as aesthetic quality, structural richness, and scene diversity. This integration yields clear and controllable improvements in the perceptual quality of generated images. Beyond reward optimization, UniPercept also functions as a unified perceptual metric for evaluating images and as a tool for characterizing perceptual distributions in large-scale datasets.

###### 2.1. MLLM Benchmark

With the rapid development of MLLMs, evaluating their performance has gone far beyond simple semantic understanding tasks such as image recognition or segmentation. In recent years, researchers have begun to assess whether a model can achieve a deeper level of understanding and reasoning about visual content.

For instance, MMMU [71] focuses on university-level exam questions across diverse disciplines, while MMMUPro [72] extends this idea with more complex, cross-domain reasoning challenges. MEGA-Bench [6] emphasizes largescale multimodal comprehension and knowledge integration, and MMStar [8] targets general reasoning and factual understanding in visual contexts. MMBench [33] evaluates comprehensive perception and reasoning across everyday images, MathVista [35] centers on mathematical and geometric reasoning within visual scenes, and OCRBench [34] specifically tests a model’s capability to recognize and interpret text embedded in images.

However, these benchmarks rely on converting visual content into text representations before reasoning, empha-

[Figure 43]

[Figure 44]

Prior Definition

Trained Volunteers

Initial Datasets

<IAA><Composition & Design><Visual Balance> Assess the distribution of visual elements such as shapes, tones, and colors ... <IQA><Distortion Location><Object Association> Specify the semantic or structural objects in the image that are impacted... <ISTA><Material Representation><Surface Behavior> Describe optical surface properties including glossiness, translucency ...

[Figure 45]

Image

High‑Quality QA Pairs

Judger MLLM

[Figure 46]

IAA IQA ISTA

Dimension: IAA Category: Visual Elements & Structure Criterion: Form Realization Question Template: which Q: Which element most effectively conveys depth through perspective cues?

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Question Validity

[Figure 53]

Generator MLLM

Answer Validity

[Figure 54]

Professional Evaluation

[Figure 55]

Reasoning Validity Criterion Relevance

Question Template Bank

<IAA> The attribute focuses on harmonic unity, specifically visual consistency in shape proportions ... <IQA> The image assessment highlights issues such as motion blur and focus problems. These are most evident in the tables and ... <ISTA> The attribute focuses on material identification, and altar's material is described as wood with smooth, matte properties ...

<Template: Yes_Or_No> Create ONE concise Yes/No question that references visible elements... <Template: What> Create ONE multiple‑choice question that starts with "What" and focuses on concrete ... <Template: Which> Create ONE multiple‑choice question that starts with "Which" or clearly frames a selection context ...

- A. Pavilion details and roof shading
- B. Gradual fading of distant mountain details
- C. Pine tree line thickness variation
- D. Water surface line patterns

(b) Reject Sampling (c) Human Refinement

(a) Initial QA Generation

- Figure 4. Constuction pipeline of UniPercept-Bench. A three-stage process initial QA generation, reject sampling, and human refinement to produce high-quality perceptual-level QA pairs across aesthetics, quality, structure, and texture.

[Figure 56]

- Figure 5. Distribution of UniPercept-Bench across (a) Domain, (b) Category, and (c) Criterion. Zoom in for best view.

45] touch upon aspects of structural or textural perception, they do not provide a unified or comprehensive definition of ISTA, leaving this important component of perceptual-level understanding still insufficiently explored.

In addition, most existing datasets focus on a single aspect such as numerical scoring or question answering, without providing a comprehensive and multi-dimensional evaluation framework. As multimodal large models continue to improve, many existing models have already achieved very high accuracy on prior benchmarks [17, 58, 73], reducing the ability of these benchmarks to effectively distinguish stronger models. In contrast, our UniPercept-Bench addresses these limitations by covering multiple perceptual dimensions, offering diverse and detailed evaluation data, and presenting a more comprehensive challenge for MLLMs.

###### 3. UniPercept-Bench

###### 3.1. Definition

As shown in Fig. 1, UniPercept-Bench targets perceptuallevel image understanding across Image Aesthetics Assessment (IAA), Image Quality Assessment (IQA), and Image Structure and Texture Assessment (ISTA). IAA focuses on the perceived aesthetic attributes of an image, such as composition, style, emotion, and overall visual appeal. IQA targets the perceived fidelity and degradation factors, including noise, blur, compression artifacts, and overall distortion levels. ISTA evaluates the structural and textural characteristics of a scene, emphasizing geometry, material properties, and local detail richness. Although all three domains assess images, the tasks focus on fundamentally different aspects. For example, a high-quality image may not possess strong aesthetic value, while an aesthetically pleasing image may contain only simple or sparse textures.

sizing language-based inference over genuine visual understanding. In contrast, UniPercept-Bench directly evaluates perceptual-level visual properties—such as technical execution, distortion location, and material depiction—bridging the gap between perceptual and semantic understanding.

###### 2.2. Image Assessment

For perceptual-level image assessment, prior research has primarily focused on two major areas: Image Aesthetics Assessment (IAA) and Image Quality Assessment (IQA). Extensive benchmarks and methods have been developed for these two categories—for example, Q-Align [60], UNIAA [74], and ArtiMuse [4] for IAA, and MUSIQ [21], DepictQA [67, 68], DeQA [69], and Q-Insight [27] for IQA. In contrast, another crucial perceptual dimension—Image Structure and Texture Assessment (ISTA)—has received far less systematic attention. Although a few prior works [9,

While IAA and IQA have been widely studied in prior works such as Q-Align [60], the DepictQA series [67–69], and ArtiMuse [4], ISTA remains largely unexplored, with little prior effort to systematically define or evaluate it. In-

[Figure 57]

[Figure 58]

[Figure 59]

###### Domain-Adaptive Pre-Training

###### Task-Aligned RL for VR & VQA

Text Annotation: The lighting accentuates the bird's intricate feather patterns ... The composition effectively centers the subject ...

[Figure 60]

[Figure 61]

###### Adaptive Gaussian Soft Reward

Answer-Matching Reward Q: How does the cliff face's base morphology influence the climber's grip?

[Figure 62]

Pred Error=5, Reward=0.827

Visual Rating (IQA) GT 33.6/100 Pred 38.6/100 Reward 0.827

###### Structured Annotation:

[Figure 63]

- A. Smooth surfaces enhance grip stability.
- B. Cracked and veined textures provide varied handholds.
- C. Matte surfaces reduce friction for climbing.
- D. Clustered arrangements hinder climber movement.

Pred Error=10, Reward=0.475

Reward

{{\"PhysicalStructure\": {\"BaseMorphology\": [\"smooth\"], \"Arrangement\": [\"Radial\"]}, {\"MaterialClass\": [\"Fabric\"], ...

Reward=0

[Figure 64]

- Reward=0
- Reward=1

Pred Error=20, Reward=0.059

[Figure 65]

Reward=0

Visual Question Answering GT B Pred B Reward 1

Visual Rating (IAA): 78.5/100

Pred Error=50, Reward=0.000

Pred Error

- Figure 6. Training pipeline of UniPercept. A two-stage framework combining domain-adaptive pre-training for perceptual understanding and task-aligned RL to jointly optimize Visual Rating and Visual Question Answering.

spired by recent advances in image generation and low-level vision [25, 56, 70], we reorganize and unify the existing definition systems for IAA and IQA, and propose the first systematic, operational definition of ISTA. Together, these three domains form a coherent perceptual-level framework that enables comprehensive and fine-grained assessment of IAA, IQA ans ISTA cues aligned with human perception. More details are provided in the Appendix.

###### 3.2. Benchmark Construction

We organize UniPercept-Bench using a three-tier taxonomy of Domain–Category–Criterion. For IAA and IQA, we consolidate expert-agreed definitions from prior literature. For ISTA, we conduct structured interviews with domain practitioners to derive precise, consensus-driven perceptual criteria. Building on this taxonomy, UniPercept-Bench introduces two complementary task forms:

Visual Rating (VR). Models output a continuous score representing perceptual alignment across IAA (aesthetics quality) and IQA (image quality), serving as a quantitative measure of perceptual understanding. We additionally define the ISTA Rating task to evaluate structure and texture richness, where a higher score indicates a scene with more complex geometry and richer textures. Following the definition framework of UniPercept-Bench, we design a scoring method that computes the ISTA score by aggregating and weighting the counts of structured attributes extracted from each image, as detailed in the Appendix.

Visual Question Answering (VQA). Question sets are constructed at the Domain–Category–Criterion levels to assess perceptual-level understanding and reasoning. Together, VR and VQA form a unified evaluation protocol that jointly measures quantitative judgment and explanatory consistency, advancing comprehensive perceptual-level understanding, as illustrated in Fig. 3.

Initial QA Generation. As illustrated in Fig. 4, We begin by collecting images from three perceptual dimensions to ensure diversity across aesthetic quality, distortion severity, and structural complexity. For IAA and IQA, we adopt expert-annotated datasets [4, 5, 67] as professional

evaluations. For ISTA, we introduce structured annotations guided by UniPercept-Bench’s taxonomy, describing scene composition, geometric layout, and material representation. These annotations are paired with predefined Domain–Category–Criterion definitions and matched templates from a curated Question Template Bank. The Generator MLLM (GPT-4o [1]) combines the image, annotation, and template to produce candidate QA pairs along with brief reasoning rationales.

Reject Sampling. A heterogeneous Judger MLLM (Qwen2.5-VL-78B-Instruct [3]) evaluates each QA pair on Question Validity, Answer Validity, Reasoning Validity, and Criterion Relevanceusing a five-point scale. Only samples rated good or above on all aspects are retained, removing about 40% of candidates.

Human Refinement. Finally, trained volunteers with expertise in Image Assessment conduct manual validation to ensure alignment with human reasoning and perception. Borderline cases are revised for clarity and consistency, while invalid samples are removed. The resulting dataset consists of high-quality QA pairs that are both perceptually grounded and semantically coherent.

###### 4. UniPercept

###### 4.1. Domain-Adaptive Pre-Training

We aim to train a model with perceptual-level understanding ability across Aesthetics, Quality, Structure, and Texture. To endow the model with a foundational capability for perceptual-level image understanding, we first conduct a Domain-Adaptive Pre-Training stage. Specifically, we select and process large-scale existing datasets, resulting in two types of pre-training data as follows:

Text-based QA pairs. We construct large-scale QA pairs for IAA, IQA, and ISTA. For ISTA, we additionally design structured-output QA pairs (Fig. 6) to support fine-grained structural reasoning.

VR-based QA pairs. We additionally incorporate VR data for IAA, IQA, and ISTA, also visualized in Fig. 6, which

- Table 1. Performance comparison of different models on UniPercept-Bench-VR. The best results are highlighted with dark blue cells, and the second-best results with light blue cells. Models with * indicate those retrained on ArtiMuse-10K, KonIQ-10K, and ISTA-10K. ”-/-” denotes cases where the model refuse to respond or fail to provide valid answers due to functional limitations. Metrics: SRCC/PLCC.

VR - IAA VR - IQA VR - ISTA ArtiMuse-10K [4] AVA [38] TAD66K [14] FLICKR-AES [40] Avg. KonIQ-10K [15] SPAQ [11] KADID [30] PIPAL [13] Avg. ISTA-10K

Models

Proprietary Models Large-scale Pretraining, Cross domain testing GPT-4o [1] 0.333/0.276 0.509/0.485 0.278/0.282 0.605/0.597 0.431/0.410 0.695/0.744 0.874/0.881 0.677/0.646 0.325/0.349 0.643/0.655 -0.003/0.116 Llama-4-Scout [37] 0.204/0.147 0.345/0.329 0.236/0.210 0.548/0.506 0.333/0.298 0.503/0.653 -0.041/0.007 -0.099/-0.004 -0.007/0.023 0.089/0.170 -0.025/0.047 Gemini-2.5-pro [46] 0.187/0.035 0.248/0.100 0.143/0.037 0.357/0.206 0.234/0.095 0.582/0.316 0.087/0.212 0.436/0.274 0.225/-0.019 0.333/0.196 -0.230/-0.118 Claude-Sonnet-4.5 [31] 0.041/0.027 0.003/0.013 0.040/0.047 0.037/0.049 0.030/0.034 -0.037/-0.043 0.036/0.085 0.223/0.273 -0.131/-0.088 0.023/0.057 0.125/0.089 Claude-Sonnet-4.5-Think [31] 0.066/0.103 0.018/0.019 0.026/0.039 -/- 0.037/0.054 -/- -/- -/- -/- -/- -/-

Open-Source Models Large-scale Pretraining, Cross domain testing LLaVA-OneVision-1.5-Instruct-8B [2] 0.274/0.212 0.381/0.378 0.213/0.224 0.586/0.541 0.364/0.339 0.639/0.744 -/- 0.505/0.534 0.417/0.407 0.520/0.562 -0.094/0.027 GLM-4.5-V-106BA12B [47] 0.346/0.249 0.464/0.420 0.289/0.278 0.651/0.597 0.438/0.386 0.721/0.765 -0.040/-0.038 -0.142/-0.128 0.013/0.020 0.138/0.155 0.083/0.117 InternVL3-8B [75] 0.245/0.211 0.372/0.344 0.205/0.191 0.547/0.476 0.342/0.306 0.574/0.646 0.828/0.800 0.496/0.475 0.435/0.459 0.583/0.595 -0.127/0.046 InternVL3-78B [75] 0.223/0.206 0.385/0.344 0.221/0.220 0.518/0.433 0.337/0.301 0.635/0.676 0.849/0.852 0.579/0.553 0.415/0.457 0.619/0.634 -/InternVL3.5-8B [52] 0.135/0.104 0.308/0.295 0.180/0.182 0.519/0.448 0.286/0.257 0.663/0.660 0.783/0.777 0.541/0.478 0.351/0.386 0.585/0.575 -0.096/-0.025 InternVL3.5-38B [52] 0.219/0.175 0.359/0.357 0.201/0.208 0.559/0.529 0.334/0.317 0.578/0.652 0.840/0.831 0.568/0.537 0.448/0.457 0.608/0.619 0.262/0.345 QwenVL-2.5-Instruct-7B [3] 0.223/0.143 0.359/0.324 0.208/0.195 0.588/0.520 0.345/0.296 0.708/0.762 -/- 0.521/0.517 0.350/0.361 0.526/0.547 -0.046/0.076 QwenVL-2.5-Instruct-72B [3] 0.233/0.197 0.408/0.387 0.232/0.235 0.626/0.589 0.375/0.352 0.762/0.820 -/- 0.606/0.570 0.381/0.407 0.583/0.599 0.091/0.148 QwenVL-3-Instruct-8B [3] 0.156/0.094 0.280/0.170 0.191/0.121 0.507/0.388 0.283/0.193 0.761/0.822 0.612/0.604 0.723/0.696 0.434/0.427 0.633/0.637 0.033/0.044 QwenVL-3-Instruct-32B [3] 0.227/0.130 0.353/0.198 0.200/0.095 0.572/0.413 0.338/0.209 0.796/0.838 0.690/0.657 0.673/0.682 0.414/0.402 0.643/0.644 0.084/0.106

Specialized Models In domain Cross domain Avg. In domain Cross domain Avg. In domain ArtiMuse [4] 0.614/0.627 0.397/0.385 0.230/0.232 0.349/0.334 0.398/0.395 -/- -/- -/- -/- -/- -/DeQA [69] -/- -/- -/- -/- -/- 0.953/0.941 0.895/0.896 0.694/0.687 0.472/0.478 0.753/0.750 -/Q-Align* [60] 0.551/0.573 0.398/0.386 0.194/0.197 0.137/0.123 0.320/0.320 0.941/0.940 0.886/0.887 0.674/0.684 0.403/0.419 0.726/0.733 -/Q-Insight [27] -/- -/- -/- -/- -/- 0.933/0.916 0.907/0.905 0.742/0.736 0.486/0.474 0.767/0.758 -/Q-Insight* [27] 0.228/0.175 0.405/0.376 0.212/0.217 0.617/0.537 0.366/0.326 0.733/0.750 0.800/0.938 0.580/0.548 0.369/0.368 0.621/0.651 0.060/0.152 UniPercept (Ours) 0.746/0.738 0.589/0.577 0.336/0.346 0.688/0.681 0.590/0.586 0.940/0.949 0.904/0.895 0.872/0.870 0.581/0.594 0.824/0.827 0.778/0.767

allows the model to directly associate perceptual-level attributes with quantitative ratings.

In total, the dataset in domain-adaptive pre-training contains approximately 800K samples. Through DomainAdaptive Pre-Training, the model acquires the essential capability to handle diverse perceptual-level image understanding tasks across different domains.

###### 4.2. Task-Aligned RL for VR & VQA

To achieve precise alignment of the model across both Visual Rating (VR) and Visual Question Answering (VQA), we employ the GRPO algorithm [44] to perform TaskAligned RL with task-specific reward functions for VR and VQA. For the VQA task, we adopt a binary reward:

rvqa =

1, if the predicted answer is correct, 0, otherwise.

(1)

For the VR task, we design an Adaptive Gaussian Soft Reward, which continuously evaluates the prediction according to its numerical deviation from the ground truth:

(|pi − gi|)2 2σdyn2

, σdyn = σ0 1 + α|pi − gi| 100

rvr = exp −

(2) where pi and gi denote the predicted and ground-truth scores (mapped to [0,100]), σ0 is the base smoothing coefficient, and α controls the degree of adaptive Gaussian smoothing. This soft reward offers smoother gradients and avoids threshold-induced discontinuities. Following prior works [4, 60], we adopt a Token As Score strategy for VR, deriving ratings from the predicted token distribution. We then incorporate this task-specific reward into the GRPO objective to enable perceptual-level policy optimization:

JGRPOB (θ) = EB

G

1

G i=1 |oi|

i=1

|oi|

ri · min rti(θ)Aˆit,

t=1

clip rti(θ),1 − ϵ,1 + ϵ A ˆit .

(3) where rti(θ) = πθ(o

i t|qi,oi<t)

πold(oit|qi,oi<t) is the ratio between current and old policies, Aˆit is the estimated advantage, and ri is the task-specific reward from Eq. 1 or Eq. 2. This unified formulation enables GRPO to align model behavior with both discrete correctness (VQA) and continuous perceptual consistency (VR).

,

###### 5. Experiments

###### 5.1. Implementation

Evaluated Models. We evaluate a total of 18 models, encompassing three categories. (1) Proprietary Models: GPT-4o [1], Llama-4-Scout [37], Gemini-2.5Pro [46], Claude-Sonnet-4.5 [31] and Claude-Sonnet-4.5Think [31]. (2) Leading Open-Source Models: the InternVL3 and InternVL3.5 series [52, 75], QwenVL2.5-Instruct and QwenVL-3-Instruct series [3], GLM-4.5V-106BA12B [47], as well as LLaVA-OneVision-1.5Instruct [2]. (3) Specialized Models for IAA and IQA: QAlign [60], ArtiMuse [4], DeQA [69], and Q-Insight [27].

Evaluation Settings. For VQA, all models were provided with identical prompts corresponding to each question, and their generated answers were compared against the groundtruth options. For VR, we designed task-specific prompts for models lacking a dedicated scoring interface to elicit

- Table 2. Performance comparison of different models on UniPercept-Bench-VQA (IAA). Category names are abbreviated; full definitions are provided in the Appendix. Results follow the same notation throughout the paper.

IAA Categories QA Templates

Models

Overall Comp. VisStr. Tech. Creat. Theme. Emo. Gest. CompEv. Lv.Pred How What Which Why Yes-No

Random Guess 23.08% 27.27% 21.95% 29.63% 25.93% 22.86% 23.68% 32.56% 24.14% 21.28% 30.43% 25.32% 24.00% 29.49% 25.17% Proprietary Models

GPT-4o 64.62% 59.57% 57.58% 60.19% 65.19% 67.62% 51.95% 30.23% 38.86% 78.17% 72.46% 62.66% 72.67% 70.51% 60.04% Llama-4-Scout 62.56% 68.45% 59.76% 61.11% 57.78% 70.48% 48.68% 32.56% 43.97% 70.92% 69.57% 61.39% 77.33% 70.51% 60.91% Gemini-2.5-pro 71.79% 68.45% 61.59% 76.85% 67.41% 63.81% 61.84% 37.21% 45.98% 78.72% 73.91% 67.72% 84.67% 84.62% 66.44% Claude-Sonnet-4.5 70.26% 70.05% 62.20% 71.30% 64.44% 67.62% 50.00% 46.51% 46.84% 77.30% 76.09% 65.19% 86.00% 69.23% 65.45% Claude-Sonnet-4.5-Think 71.28% 69.52% 61.21% 68.52% 62.22% 66.67% 53.25% 41.86% 44.57% 75.89% 77.54% 67.09% 86.00% 66.67% 64.73%

###### Open-Source Models

LLaVA-OneVision-1.5-Instruct-8B 67.18% 68.62% 61.21% 62.96% 67.41% 62.86% 53.25% 20.93% 34.86% 85.21% 79.71% 65.82% 83.33% 69.23% 62.60% GLM-4.5-V-106BA12B 67.18% 65.78% 60.98% 75.00% 64.44% 68.57% 51.32% 46.51% 45.40% 71.63% 78.26% 65.82% 84.67% 70.51% 64.46% InternVL3-8B 65.64% 67.55% 59.39% 67.59% 69.63% 62.86% 50.65% 25.58% 36.00% 81.69% 73.91% 67.72% 86.00% 71.79% 62.60% InternVL3-78B 71.79% 73.26% 61.21% 73.15% 74.81% 74.29% 53.25% 37.21% 45.14% 85.82% 81.16% 72.15% 86.00% 75.64% 68.28% InternVL3.5-8B 32.31% 29.41% 30.30% 26.85% 28.89% 26.67% 23.38% 9.30% 17.14% 41.13% 26.81% 19.62% 36.00% 58.97% 28.18% InternVL3.5-38B 37.44% 40.11% 27.88% 39.81% 34.81% 38.10% 45.45% 6.98% 34.00% 47.52% 26.09% 28.48% 37.33% 50.00% 35.67% QwenVL-2.5-Instruct-7B 67.18% 70.74% 56.36% 66.67% 68.89% 63.81% 48.05% 37.21% 38.86% 76.76% 75.36% 67.09% 87.33% 71.79% 63.19% QwenVL-2.5-Instruct-72B 22.05% 24.60% 25.45% 29.63% 30.37% 18.10% 19.48% 6.98% 14.00% 19.86% 17.39% 24.05% 41.33% 51.28% 23.74% QwenVL-3-Instruct-8B 31.28% 32.09% 32.12% 37.04% 34.07% 22.86% 37.66% 25.58% 35.43% 14.89% 17.39% 34.81% 28.67% 73.08% 31.92% QwenVL-3-Instruct-32B 23.08% 26.74% 32.12% 26.85% 32.59% 20.95% 33.77% 20.93% 33.43% 9.22% 13.77% 31.01% 18.67% 66.67% 27.39%

Specialized Models

ArtiMuse 67.69% 68.45% 64.85% 74.07% 71.85% 64.76% 61.04% 32.56% 39.14% 88.65% 76.81% 72.78% 85.33% 79.49% 66.31% UniPercept (Ours) 80.00% 77.54% 69.70% 80.56% 79.26% 80.95% 67.53% 69.77% 63.71% 92.20% 81.88% 75.32% 86.67% 84.62% 76.55%

quantitative predictions, while specialized models for visual rating were directly evaluated through their native interfaces. It is worth noting that all specialized models are trained exclusively on In-domain datasets. For representative models (Q-Align and Q-Insight), we further retrain them on a mixed dataset comprising ArtiMuse-10K [4], KonIQ-10K [15], and ISTA-10K to ensure consistent perceptual alignment across domains.

Training Details. Based on InternVL3-8B[75], UniPercept is trained in two stages as described in Sec. 4. The Domain-Adaptive Pre-Training stage adopts multiple public datasets, including APDDv2 [19] and Impressions [24] for IAA, Q-Ground-100K [5] and DataDepictQA [67] for IQA, and DTD [9], FMD [45], Flickr2K [29], and LSDIR [28] for ISTA, among others. After preprocessing and filtering, the resulting corpus contains approximately 800K samples in total. For Task-Aligned RL, we adopt the VR datasets ArtiMuse-10K [4], KonIQ-10K [15], and ISTA-10K, along with ∼30K VQA samples generated as described in Sec. 3.2. Training is performed on 16 NVIDIA A100 GPUs for 2 epochs per each stage with a batch size of 128. In GRPO, we sample n=8 responses per query and set β=0.001, ε=0.2, and σ=0.8.

###### 5.2. Benchmark Results with Analysis 5.2.1. Visual Rating

General Models vs. Specialized Models. Visual Rating is a challenging task that requires models to output continuous, high-precision perceptual scores. As shown in Table 1, most general-purpose MLLMs without task-specific training exhibit significantly lower performance compared with specialized models. This gap arises from the inherent difficulty of directly generating numerical outputs in text form, which often leads to hallucinated or unstable predictions—

a phenomenon also discussed in prior works [4, 26, 60]. In contrast, specialized methods adopt structured strategies such as Token As Score, effectively stabilizing the regression behavior of large models on continuous scales. [4, 60] Domain-wise Analysis. Model performance varies significantly across different domains, reflecting differences in task objectivity. IQA achieves the highest correlations, followed by ISTA, while IAA remains the most challenging due to its subjective and ambiguous aesthetic judgments. These results suggest that perceptual-level visual rating becomes increasingly difficult as the task shifts from objective assessment toward subjective reasoning.

###### 5.2.2. Visual Question Answering

Domain-wise Analysis. Across domains, the bestperforming generalized models can reach accuracies of up to 68.28%, 72.15%, and 81.13% on IAA, IQA, and ISTA, respectively. However, their average accuracies drop to only 51.62%, 51.45%, and 60.56%. This disparity suggests that perceptual-level visual question answering remains highly challenging for current MLLMs, as they often rely on unstable domain-specific heuristics rather than robust perceptual reasoning. Among the three domains, ISTA appears slightly easier, as it centers on objective and physically grounded properties—such as geometry, structure, and material composition—that align better with the visual–text priors learned during large-scale pretraining. In contrast, IAA and IQA require subjective reasoning over aesthetics, emotion, and perceptual quality—nuanced judgments that general models are not explicitly optimized for.

Category-wise Analysis. Models generally perform better on holistic perception categories such as Composition & Design and Theme & Communication, which focus on the overall aesthetic and semantic coherence of an image. Most models achieve accuracies above 60% in these cate-

###### Table 3. Performance comparison of different models on UniPercept-Bench-VQA (IQA).

81%

IQA Categories QA Templates

72%

84%

70%

Models

Overall Loc. Sev. Type. Lv.Pred How What Which Why Yes-No

66%

81%

Random Guess 23.67% 24.75% 20.08% 24.75% 27.03% 16.05% 25.00% 21.39% 22.99% 23.16% Proprietary Models

60%

79%

63%

68%

Q

A

I

77%

I

75%

A

- S
- T A

GPT-4o 71.74% 53.18% 70.49% 53.18% 83.78% 59.26% 61.31% 80.21% 67.82% 66.36% Llama-4-Scout 60.18% 58.19% 52.05% 58.19% 82.16% 37.04% 38.69% 66.31% 62.07% 57.81% Gemini-2.5-pro 32.84% 52.84% 40.98% 52.84% 40.54% 32.72% 29.17% 41.18% 28.74% 40.17% Claude-Sonnet-4.5 71.19% 51.51% 66.80% 51.51% 90.81% 50.00% 50.60% 82.89% 71.26% 64.80% Claude-Sonnet-4.5-Think 71.19% 55.52% 66.80% 55.52% 89.19% 50.00% 51.79% 82.89% 72.41% 65.90%

t

n

io

es

A

u

n

A

Q

s w

I

al

e

u

rin g

Vis

A

I

A

0.35

V

g

- S
- T

n

i

s

A

i

u

at

a

R

l

0.77

I

0.30

0.40

I

A

Q

0.11

0.06

0.42

###### Open-Source Models

0.59

LLaVA-OneVision-1.5-Instruct-8B 76.51% 59.87% 77.46% 59.87% 91.35% 70.37% 61.31% 82.35% 75.86% 72.15% GLM-4.5-V-106BA12B 70.09% 35.79% 54.51% 35.79% 88.11% 48.77% 44.05% 74.33% 68.97% 57.17% InternVL3-8B 71.56% 52.84% 59.43% 52.84% 87.03% 59.88% 48.81% 71.12% 71.26% 63.69% InternVL3-78B 75.41% 51.84% 81.56% 51.84% 93.51% 66.67% 63.10% 88.24% 66.67% 70.31% InternVL3.5-8B 38.17% 44.82% 38.11% 44.82% 35.14% 41.98% 30.36% 36.36% 56.32% 39.98% InternVL3.5-38B 38.90% 49.83% 45.08% 49.83% 46.49% 41.36% 31.55% 33.16% 62.07% 43.29% QwenVL-2.5-Instruct-7B 74.13% 48.83% 66.39% 48.83% 88.65% 60.49% 53.57% 78.61% 77.01% 65.44% QwenVL-2.5-Instruct-72B 31.01% 4.68% 16.39% 4.68% 35.14% 14.81% 11.31% 22.99% 66.67% 20.50% QwenVL-3-Instruct-8B 34.68% 55.18% 16.39% 55.18% 20.54% 18.52% 27.38% 25.67% 77.01% 36.21% QwenVL-3-Instruct-32B 29.54% 14.38% 16.80% 14.38% 11.89% 18.52% 25.60% 22.46% 74.71% 22.52%

0.64

0.65

0.83

0.75

UniPercept ArtiMuse LLaVA−OneVision−1.5−Instruct−8B DeQA InternVL3−78B InternVL3.5−38B Q−Insight* GPT−4o

Figure 7. Results on UniPerceptBench. VQA and VR are evaluated by Acc. and (SRCC + PLCC)/2.

UniPercept (Ours) 77.43% 79.60% 90.98% 79.60% 87.03% 80.86% 75.60% 83.42% 79.31% 81.07%

###### Table 4. Performance comparison of different models on UniPercept-Bench-VQA (ISTA).

###### Models ISTA Categories QA Templates Overall

Scene. Phys. Mat. Geo. Sem. How What Which Why Yes-No Random Guess 26.50% 23.63% 24.73% 30.30% 30.58% 26.28% 23.84% 24.29% 33.77% 33.33% 26.60%

###### Proprietary Models

GPT-4o 75.64% 79.12% 73.48% 33.33% 77.27% 71.79% 78.78% 69.23% 77.92% 72.46% 74.64% Llama-4-Scout 73.50% 75.27% 71.68% 72.73% 67.77% 75.64% 69.77% 69.64% 77.27% 69.57% 71.86% Gemini-2.5-pro 76.50% 82.42% 77.06% 66.67% 77.69% 78.21% 78.20% 75.71% 82.47% 71.01% 77.73% Claude-Sonnet-4.5 76.92% 78.57% 74.91% 90.91% 77.69% 76.92% 77.03% 74.49% 81.82% 79.71% 77.32% Claude-Sonnet-4.5-Think 77.35% 78.02% 73.12% 87.88% 75.21% 76.28% 74.71% 74.09% 81.82% 76.81% 76.08%

###### Open-Source Models

LLaVA-OneVision-1.5-Instruct-8B 78.63% 85.16% 82.44% 72.73% 80.17% 83.33% 81.40% 75.30% 84.42% 88.41% 81.13% GLM-4.5-V-106BA12B 81.20% 79.67% 74.55% 72.73% 75.21% 80.77% 76.74% 73.68% 79.87% 78.26% 77.22% InternVL3-8B 75.64% 79.12% 73.48% 33.33% 77.27% 71.79% 78.78% 69.23% 77.92% 72.46% 74.64% InternVL3-78B 79.06% 85.16% 77.42% 69.70% 78.51% 81.41% 79.65% 73.68% 84.42% 81.16% 79.28% InternVL3.5-8B 54.27% 50.55% 58.42% 39.39% 36.36% 46.79% 56.69% 48.58% 29.87% 71.01% 49.79% InternVL3.5-38B 50.00% 55.49% 61.29% 30.30% 35.95% 50.64% 59.30% 42.91% 37.01% 57.97% 50.10% QwenVL-2.5-Instruct-7B 74.79% 72.53% 74.91% 51.52% 73.55% 73.72% 77.33% 66.80% 74.03% 73.91% 73.30% QwenVL-2.5-Instruct-72B 14.10% 29.12% 19.71% 12.12% 18.60% 20.51% 12.21% 14.57% 31.17% 46.38% 19.59% QwenVL-3-Instruct-8B 27.78% 32.42% 25.45% 39.39% 24.79% 14.74% 23.26% 28.34% 25.32% 81.16% 27.63% QwenVL-3-Instruct-32B 26.50% 24.73% 19.00% 15.15% 18.60% 11.54% 18.31% 22.67% 17.53% 66.67% 21.65%

UniPercept (Ours) 89.74% 85.71% 82.44% 93.94% 78.51% 82.69% 89.24% 78.54% 83.12% 85.51% 84.23%

gories, indicating that global scene perception is relatively well captured by vision–language pretraining. In contrast, performance drops substantially on fine-grained perceptual categories such as Overall Gestalt, Material Representation, and Geometric Composition, where most models remain below 50%. These results suggest that while highlevel holistic understanding is well learned, precise reasoning over local structures cues remains a key limitation.

QA Template Analysis. The Level Prediction questions require models to provide fine-grained evaluations of images along specific dimensions (as shown in Fig. 3). Due to the demand for precise quantitative reasoning, most MLLMs struggle with this type, achieving only around 36% and 45% average accuracy on the IAA and IQA domains, respectively. For other QA templates, “Yes-No” and “Why” questions—closer to higher-level reasoning and causal inference—allow generalized MLLMs to reach average accuracies above 60%. In contrast, “What” and “Which” questions, which require detailed and localized visual analysis, remain challenging, reflecting the current models’ limited

capacity for fine-grained perceptual understanding.

###### 5.3. Further Discussion on UniPercept

###### 5.3.1. Performance

As shown in Fig. 7 and Tabs. 1, 2, 3, and 4, UniPercept is capable of handling both major types of perceptuallevel image understanding tasks (VR & VQA) across the three perceptual domains of IAA, IQA, and ISTA. Benefiting from the proposed Domain-Adaptive Pre-Training and Task-Aligned RL strategies, UniPercept consistently outperforms both generalized and specialized models. It not only achieves strong in-domain results but also exhibits remarkable cross-domain generalization, demonstrating its capability as a unified perceptual understanding baseline.

###### 5.3.2. UniPercept Reward

We investigate the use of UniPercept as a multidimensional perceptual reward model for text-to-image generation. UniPercept provides three types of visual ratings (Aesthetics Quality (IAA), Image Quality (IQA), and

Table 5. More Performance Results of FLUX.1-dev w/ UniPercept Reward.

Preference Score Image Quality Image Aesthetics UniPercept Score

Models

PickScore [22] HPSv3 [36] DeQA [69] LAION-Aes [42] ArtiMuse [4] IAA IQA ISTA

Baseline 22.46 10.71 4.32 5.77 59.02 65.18 73.59 46.64 w/ UniPercept IAA Reward 22.47 10.09 4.09 6.19 67.02 76.20 76.39 54.83 w/ UniPercept IQA Reward 22.63 11.21 4.37 6.02 63.64 72.16 76.87 52.34 w/ UniPercept ISTA Reward 22.72 11.09 4.37 6.16 63.75 72.23 76.17 59.61 w/ UniPercept Reward (All) 22.67 10.93 4.33 6.19 65.52 74.24 77.04 59.08

Prompt w/ UniPercept IAA Reward

###### Baseline (FLUX.1-dev)

w/ UniPercept IQA Reward

w/ UniPercept ISTA Reward

w/ UniPercept Reward (All)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

A modern office space featuring a sleek desk with a computer set up, including a monitor, keyboard, and mouse. Beside the computer, there's a printer with a stack of paper next to it. An ergonomic office chair is positioned in front of the desk, ready for someone to sit down and start working.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

A young child with brown hair, focused intently, sits at a wooden table scattered with colorful crayons and paper. In their small hand is a bright red pencil, with which they are diligently drawing a vibrant blue flower that's taking shape on the white sheet before them. Sunlight filters through a nearby window, casting a warm glow on the child's artwork.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

A striking black bird with glossy feathers sits atop the vibrant orange petals of a Bird of Paradise flower. The unique flower is positioned in the midst of an arid desert landscape, with various cacti and sparse vegetation dotting the sandy ground. In the background, the sun casts a warm glow on the distant rolling dunes.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

A vibrant yellow 2017 Porsche 911 is captured in motion, navigating a winding mountain road with its sleek body hugging the curve. The sports car's headlights are piercing through the overcast weather, illuminating the path ahead. In the background, a lush green valley stretches out beneath a sky filled with grey clouds, hinting at the vast expanse beyond the road's edge.

Figure 8. More results of FLUX.1-dev w/ UniPercept Reward. Different reward signals emphasize distinct perceptual attributes, while UniPercept Reward (All) achieves the best overall performance by integrating complementary perceptual cues.

Structure & Texture Richness (ISTA)) which can be employed as independent reward signals or jointly as an integrated perceptual reward. All reward configurations are incorporated into the Flow-GRPO [32] fine-tuning pipeline based on FLUX.1-dev [25]. Quantitative results in Tab. 5 show that each reward captures a distinct facet of perceptual evaluation and thus leads to different characteristic improvements. The IAA reward mainly enhances aestheticsrelated metrics, whereas the IQA reward yields stronger gains on quality-oriented metrics such as sharpness and clarity. The ISTA reward, in contrast, provides more balanced improvements across all dimensions due to its emphasis on structural and textural richness. When combining

all three as a unified UniPercept reward (All), the model integrates the benefits of each dimension and achieves the best overall performance.

Corresponding qualitative results in Fig. 8 further demonstrate that UniPercept-guided training improves visual fidelity, perceptual quality, and human preference. These results confirm that UniPercept serves effectively not only as a reward model but also as a unified set of perceptual metrics for evaluating generated images.

###### 5.3.3. UniPercept Metrics

For Models. UniPercept can serve as an perceptual-level metric that assesses the quality of outputs from any model producing images, covering three complementary dimen-

Table 6. Evaluation of T2I Models on DPG [16] Metrics and UniPercept Metrics.

DPG Metrics [16] UniPercept Metrics Global Entity Attribute Relation Other Overall IAA IQA ISTA Avg.

Models

OmniGen [62] – – – – – – 62.83 72.22 45.09 60.04 OmniGen2 [57] 88.81 88.83 90.18 89.37 90.27 83.57 58.51 71.89 43.31 57.90 BAGEL [10] 88.94 90.37 91.29 90.82 88.67 85.07 60.20 70.52 45.78 58.83 SANA-1.6B [63, 64] 86.00 91.50 88.90 91.90 90.70 84.80 40.33 42.89 42.41 41.87 Lumina-DiMOO [65] 81.46 92.08 88.98 94.31 82.00 86.04 61.00 71.14 44.83 58.99 FLUX.1-dev [25] 74.35 90.00 88.96 90.87 88.33 83.84 65.18 73.59 46.64 61.80 GPT-Image-1 [1] 88.89 88.94 89.84 92.63 90.96 85.15 62.27 72.87 44.88 60.00 Qwen-Image [56] 91.32 91.56 92.02 94.31 92.73 88.32 62.89 72.15 47.23 60.76

Table 7. Evaluation of T2I Models on GenEval [12] Metrics and UniPercept Metrics.

GenEval [12] Metrics UniPercept Metrics Single Obj. Two Obj. Counting Colors Position Attr. Bind. Overall IAA IQA ISTA Avg.

Models

OmniGen [62] 0.99 0.86 0.64 0.85 0.31 0.55 0.70 58.84 75.62 41.00 58.49 OmniGen2 [57] 0.99 0.96 0.74 0.98 0.71 0.75 0.86 54.20 75.16 34.48 54.61 BAGEL [10] 0.99 0.94 0.81 0.88 0.64 0.63 0.82 58.68 71.24 38.35 56.09 SANA-1.6B [63, 64] 0.99 0.77 0.62 0.88 0.21 0.47 0.66 34.34 35.11 31.22 33.56 Lumina-DiMOO [65] 1.00 0.94 0.85 0.89 0.85 0.76 0.88 51.93 71.98 30.86 51.59 FLUX.1-dev [25] 0.98 0.81 0.74 0.79 0.22 0.45 0.66 64.24 74.96 41.14 60.11 GPT-Image-1 [1] 0.99 0.92 0.85 0.92 0.75 0.61 0.84 69.07 76.74 51.26 65.69 Qwen-Image [56] 0.99 0.92 0.89 0.88 0.76 0.77 0.87 52.02 74.44 34.13 53.53

sions: IAA, IQA, and ISTA. Here, we take text-to-image models as an example and apply UniPercept as an evaluation metric on both the DPG [16] and GenEval [12] benchmarks to measure the perceptual quality of generated images. Except for special cases (e.g., GPT-Image-1 [1], where output resolution cannot be controlled), all models are evaluated with a fixed output resolution of 1024×1024. The results are reported in Tab.6 and Tab.7.

We observe that while current models generally achieve strong performance in IQA, there remains substantial room for improvement in IAA and ISTA. Moreover, as model capability advances, we see consistent improvements in both instruction-following ability (as reflected by DPG and GenEval metrics) and perceptual image quality (as reflected by UniPercept). For instance, GPT-Image-1 [1] and Qwen-Image [56] exhibit strong performance across both Dataset Metrics and UniPercept Metrics. More detailed analyses will be presented in future work.

For Datasets. We evaluate a variety of natural-image and AIGC-image datasets using UniPercept as the assessment metric, with results summarized in Tab.8. Among all datasets, Unsplash [51] and Blip3o-60K[7] achieve the strongest overall performance across the three domains of IAA, IQA, and ISTA. A more in-depth investigation of these distributional characteristics is left for future work.

Table 8. UniPercept Metrics on Various Datasets.

Dataset UniPercept-IAA UniPercept-IQA UniPercept-ISTA Avg. Natural Images

ImageNet [41] 53.88 61.90 36.79 50.85 Unsplash [51] 62.49 69.19 43.32 58.33 DF2K [18, 48–50] 45.99 52.92 34.78 44.56 LAION-5B [43] 60.56 69.21 38.85 56.21

###### AIGC Images

Blip3o-60K [7] 63.81 73.88 49.38 62.36 ImgEdit [66] 55.83 59.77 36.88 50.83

###### 6. Conclusion

We provide UniPercept-Bench, a unified benchmark built on a hierarchical definition for perceptual-level image understanding. We also develop a strong baseline UniPercept through Domain-Adaptive Pre-training and Task-Aligned RL, which generalizes well across perceptual domains and outperforms existing MLLMs. UniPercept further serves as a plug-and-play reward model for perceptually aligned post-training of T2I models, enabling controllable improvements in perceptual attributes. Beyond model optimization, it also offers a unified perceptual diagnostic tool that reveals systematic behavioral and dataset-level patterns, highlighting its broader utility for future perceptual-level research.

Limitations. Although UniPercept-Bench is sufficiently large for current perceptual-level tasks, it remains smaller than typical semantic-level benchmarks. Further expansion in scale will be explored in future work.

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2, 5, 6, 10, 21

- [2] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, Ziyong Feng, Ziwei Liu, Bo Li, and Jiankang Deng. Llava-onevision-1.5: Fully open framework for democratized multimodal training. In arXiv, 2025. 6
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 5, 6
- [4] Shuo Cao, Nan Ma, Jiayang Li, Xiaohui Li, Lihao Shao, Kaiwen Zhu, Yu Zhou, Yuandong Pu, Jiarui Wu, Jiaquan Wang, Bo Qu, Wenhai Wang, Yu Qiao, Dajuin Yao, and Yihao Liu. Artimuse: Fine-grained image aesthetics assessment with joint scoring and expert-level understanding, 2025. 4, 5, 6, 7, 9, 15, 21, 22, 23
- [5] Chaofeng Chen, Sensen Yang, Haoning Wu, Liang Liao, Zicheng Zhang, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Q-ground: Image quality grounding with large multi-modality models, 2024. 5, 7, 15, 22
- [6] Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, et al. Mega-bench: Scaling multimodal evaluation to over 500 real-world tasks. arXiv preprint arXiv:2410.10563, 2024. 3
- [7] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3o: A family of fully open unified multimodal modelsarchitecture, training and dataset, 2025. 10
- [8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024. 3
- [9] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi. Describing textures in the wild. In Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2014. 4, 7, 22
- [10] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 10
- [11] Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, and Zhou Wang. Perceptual quality assessment of smartphone photography. In IEEE Conference on Computer Vision and Pattern Recognition, pages 3677–3686, 2020. 6, 22
- [12] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt.

- Geneval: An object-focused framework for evaluating textto-image alignment, 2023. 10
- [13] Jinjin Gu, Haoming Cai, Haoyu Chen, Xiaoxing Ye, Jimmy Ren, and Chao Dong. Pipal: a large-scale image quality assessment dataset for perceptual image restoration, 2020. 6, 22
- [14] Shuai He, Yongchang Zhang, Rui Xie, Dongxiang Jiang, and Anlong Ming. Rethinking image aesthetics assessment: Models, datasets and benchmarks. IJCAI, 2022. 6, 22
- [15] Vlad Hosu, Hanhe Lin, Tamas Sziranyi, and Dietmar Saupe. Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing, 29:4041–4056, 2020. 6, 7, 21, 22, 23
- [16] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment, 2024. 10
- [17] Yipo Huang, Quan Yuan, Xiangfei Sheng, Zhichao Yang, Haoning Wu, Pengfei Chen, Yuzhe Yang, Leida Li, and Weisi Lin. Aesbench: An expert benchmark for multimodal large language models on image aesthetics perception. arXiv preprint arXiv:2401.08276, 2024. 4, 22
- [18] Andrey Ignatov, Radu Timofte, et al. Pirm challenge on perceptual image enhancement on smartphones: report. In European Conference on Computer Vision (ECCV) Workshops,

2019. 10

- [19] Xin Jin, Qianqian Qiao, Yi Lu, Huaye Wang, Heng Huang, Shan Gao, Jianfei Liu, and Rui Li. Apddv2: Aesthetics of paintings and drawings dataset with artist labeled scores and comments, 2024. 7, 22
- [20] Emilie L Josephs, Haoyun Zhao, and Talia Konkle. The world within reach: an image database of reach-relevant environments. Journal of Vision, 21(7):14–14, 2021. 22
- [21] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 4
- [22] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation.

2023. 9

- [23] Talia Konkle and Aude Oliva. A real-world size organization of object responses in occipitotemporal cortex. Neuron, 74

(6):1114–1124, 2012. 22

- [24] Julia Kruk, Caleb Ziems, and Diyi Yang. Impressions: Understanding visual semiotics and aesthetic impact, 2023. 7, 22
- [25] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 5, 9, 10

- [26] Mingxing Li, Rui Wang, Lei Sun, Yancheng Bai, and Xiangxiang Chu. Next token is enough: Realistic image quality and aesthetic scoring with multimodal large language model,

2025. 7

- [27] Weiqi Li, Xuanyu Zhang, Shijie Zhao, Yabin Zhang, Junlin Li, Li Zhang, and Jian Zhang. Q-insight: Understanding image quality via visual reinforcement learning. arXiv preprint arXiv:2503.22679, 2025. 4, 6, 22
- [28] Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, Rakesh Ranjan, Radu Timofte, and Luc Van Gool. Lsdir: A large scale dataset for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1775–1787, 2023. 7, 22
- [29] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops,

2017. 7, 22

- [30] Hanhe Lin, Vlad Hosu, and Dietmar Saupe. Kadid-10k: A large-scale artificially distorted iqa database. In 2019 Tenth International Conference on Quality of Multimedia Experience (QoMEX), pages 1–3. IEEE, 2019. 6, 22
- [31] Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, Andrew Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas Conerly, Chris Olah, and Joshua Batson. On the biology of a large language model. Transformer Circuits Thread, 2025. 2, 6
- [32] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl, 2025. 3, 9
- [33] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 3
- [34] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 3
- [35] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 3
- [36] Yuhang Ma, Yunhao Shui, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score, 2025. 9
- [37] Meta Platforms, Inc. Llama 4 scout: 17b active parameters, 16 experts — a natively multimodal large language model. https://huggingface.co/meta-llama/Llama4-Scout-17B-16E, 2025. Model release. Available at https://huggingface.co/meta-llama/Llama4-Scout-17B-16E (accessed YYYY-MM-DD). 6

- [38] Naila Murray, Luca Marchesotti, and Florent Perronnin. Ava: A large-scale database for aesthetic visual analysis. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 2408–2415, 2012. 6, 22
- [39] Soojin Park, Talia Konkle, and Aude Oliva. Parametric coding of the size and clutter of natural scenes in the human brain. Cerebral cortex, 25(7):1792–1805, 2015. 22
- [40] Jian Ren, Xiaohui Shen, Zhe Lin, Radomir Mech, and David J. Foran. Personalized image aesthetics. In The IEEE International Conference on Computer Vision (ICCV), 2017. 6, 22
- [41] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge,

2015. 10

- [42] Christoph Schuhmann and Romain Beaumont. Laionaesthetics. LAION. AI, 2022. 9
- [43] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. 10
- [44] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. 6
- [45] Lavanya Sharan, Ruth Rosenholtz, and Edward H. Adelson. Accuracy and speed of material categorization in real-world images. Journal of Vision, 14(10), 2014. 4, 7, 22
- [46] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 6
- [47] V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang.

- Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. 6
- [48] Radu Timofte, Eirikur Agustsson, Luc Van Gool, MingHsuan Yang, Lei Zhang, Bee Lim, et al. Ntire 2017 challenge on single image super-resolution: Methods and results. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2017. 10
- [49] Radu Timofte, Shuhang Gu, Jiqing Wu, Luc Van Gool, Lei Zhang, Ming-Hsuan Yang, Muhammad Haris, et al. Ntire 2018 challenge on single image super-resolution: Methods and results. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018.
- [50] Radu Timofte, Shuhang Gu, Jiqing Wu, Luc Van Gool, Lei Zhang, Ming-Hsuan Yang, Muhammad Haris, et al. Ntire 2018 challenge on single image super-resolution: Methods and results. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018. 10
- [51] Unsplash. Unsplash data / unsplash dataset. https:// unsplash.com/data, 2025. Accessed: 2025-11-21. 10
- [52] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. 2, 6
- [53] Wikipedia contributors. Aesthetics. https://en. wikipedia.org/wiki/Aesthetics, 2025. Accessed: 2025-11-21. 15
- [54] Wikipedia contributors. Image quality. https://en. wikipedia.org/wiki/Image_quality, 2025. Accessed: 2025-11-21. 15
- [55] Wikipedia contributors. Image texture. https://en. wikipedia.org/wiki/Image_texture, 2025. Accessed: 2025-11-21. 15
- [56] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 5, 10

- [57] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation, 2025. 10
- [58] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181, 2023. 4, 15, 22
- [59] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Kaixin Xu, Chunyi Li, Jingwen Hou, Guangtao Zhai, Geng Xue, Wenxiu Sun, Qiong Yan, and Weisi Lin. Q-instruct: Improving low-level visual abilities for multi-modality foundation models, 2023. 22
- [60] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. Q-align: Teaching LMMs for visual scoring via discrete text-defined levels. In Proceedings of the 41st International Conference on Machine Learning, pages 54015–54029. PMLR, 2024. 4, 6, 7
- [61] Haoning Wu, Hanwei Zhu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Annan Wang, Wenxiu Sun, Qiong Yan, Xiaohong Liu, Guangtao Zhai, Shiqi Wang, and Weisi Lin. Towards open-ended visual quality comparison, 2024. 22
- [62] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation, 2024. 10
- [63] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. Sana: Efficient high-resolution image synthesis with linear diffusion transformer, 2024. 10
- [64] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer,

2025. 10

- [65] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, Jinbin Bai, Qian Yu, Dengyang Jiang, Yuandong Pu, Haoxing Chen, Le Zhuo, Junjun He, Gen Luo, Tianbin Li, Ming Hu, Jin Ye, Shenglong Ye, Bo Zhang, Chang Xu, Wenhai Wang, Hongsheng Li, Guangtao Zhai, Tianfan Xue, Bin Fu, Xiaohong Liu, Yu Qiao, and Yihao Liu. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding, 2025. 10
- [66] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark, 2025. 10
- [67] Zhiyuan You, Jinjin Gu, Zheyuan Li, Xin Cai, Kaiwen Zhu, Chao Dong, and Tianfan Xue. Descriptive image quality assessment in the wild. arXiv preprint arXiv:2405.18842,

2024. 4, 5, 7, 15, 22

- [68] Zhiyuan You, Zheyuan Li, Jinjin Gu, Zhenfei Yin, Tianfan Xue, and Chao Dong. Depicting beyond scores: Advancing image quality assessment through multi-modal language models. In European Conference on Computer Vision, 2024. 4, 15, 22
- [69] Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. In IEEE Conference on Computer Vision and Pattern Recognition, 2025. 4, 6, 9
- [70] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photorealistic image restoration in the wild, 2024. 5
- [71] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 3
- [72] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multidiscipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. 3
- [73] Zicheng Zhang, Haoning Wu, Erli Zhang, Guangtao Zhai, and Weisi Lin. Q-bench++: A benchmark for multi-modal foundation models on low-level vision from single images to pairs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 4
- [74] Zhaokun Zhou, Qiulin Wang, Bin Lin, Yiwei Su, Rui Chen, Xin Tao, Amin Zheng, Li Yuan, Pengfei Wan, and Di Zhang. Uniaa: A unified multi-modal image aesthetic assessment baseline and benchmark. arXiv preprint arXiv:2404.09619,

2024. 4

- [75] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2, 6, 7

###### UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture

###### Supplementary Material

###### 7. Unifying Perceptual-Level Image Understanding

###### 7.1. From Semantics to Perception

High-level image understanding concerns what an image depicts—objects, actions, scenes, and their semantic relations (e.g., “a dog running on grass,” “a busy street at night”). In contrast, perceptual-level image understanding concerns how an image looks and feels to human observers. It captures intrinsic perceptual properties of the visual signal, largely independent of whether semantic content is correctly recognized.

An image may be semantically understandable yet perceptually flawed due to poor composition, compression artifacts, or unrealistic textures. Perceptual-level understanding emphasizes visual cues and their human-perceived effects—beauty and preference, fidelity and distortion, and structural/textural realism—rather than categorical recognition or reasoning. This distinction is well aligned with prior research separating perceptual judgments from semantic interpretation.

We view perceptual-level image understanding as a unified family of human-aligned, low-level perceptual tasks, consisting of:

- • Image Aesthetics Assessment (IAA)
- • Image Quality Assessment (IQA)
- • Image Structure & Texture Assessment (ISTA) Together, these tasks form a top-down to bottom-up char-

acterization of perceptual attributes:

- • Aesthetics captures holistic human preference and artistic visual appeal, reflecting composition, mood, and overall expressive quality.
- • Quality captures visual fidelity and perceived degradation, encompassing distortions and signal-level clarity.
- • Structure & Texture capture local cues shaping realism and richness, including fine-grained patterns, materials, and small-scale structural details.

This tripartite formulation provides a more complete perceptual description than treating aesthetics, quality, or texture as isolated benchmarks.

###### 7.2. Detailed Explanation of IAA, IQA, and ISTA

###### 7.2.1. Image Aesthetics Assessment (IAA)

IAA represents the highest level of perceptual abstraction. Its goal is to predict an image’s holistic aesthetic appealhow pleasing, expressive, or artistically valuable it appears.

The task draws on both philosophical aesthetics and empirical studies of human visual preference [53]. Common aesthetic factors include: [4]

- • Composition: balance, framing, symmetry, rule-ofthirds, leading lines;
- • Visual Elements: harmony, exposure, atmosphere;
- • Creativity: intent, novelty, genre coherence;
- • Emotion: theme communication, storytelling. Because aesthetic judgments depend strongly on viewer

psychology, IAA is the most subjective perceptual dimension.

###### 7.2.2. Image Quality Assessment (IQA)

IQA bridges subjective perception and objective image formation. Its goal is to estimate perceived visual fidelity, i.e., the severity of degradations relative to an ideal image. Such degradations commonly arise from sensor noise, blur, compression, transmission errors, and artifacts [54].

Whereas IAA asks “Is this beautiful?”, IQA asks “Is this technically clean?” Partially, IQA evaluates: [5, 58, 67, 68]

- • Signal Fidelity: sharpness, exposure, color naturalness;
- • Artifacts: blur, noise, compression artifacts, aliasing;
- • Perceptual Faithfulness: how “undistorted” the image appears.

These properties have both measurable signal correlates and human-dependent thresholds, placing IQA between IAA and ISTA in subjectivity.

###### 7.2.3. Image Structure & Texture Assessment (ISTA)

ISTA measures local perceptual primitives that determine whether an image appears structurally coherent and texturally rich [55]. Unlike the global emphasis of IAA and IQA, ISTA focuses on pixel- and patch-level cues, including:

- • Local Structure: edges, contours, geometric consistency;
- • Texture Statistics: granularity, repetitiveness, roughness/smoothness;
- • Material-like Cues: surface micro-patterns conveying realism (e.g., fabric weave, wood grain).

ISTA connects to classic studies of texture perception and structural similarity, but reframes them as a unified perceptual assessment dimension that many vision models underrepresent.

###### 7.3. The Perceptual Hierarchy

IAA, IQA, and ISTA form a complementary perceptual hierarchy:

• IAA: holistic beauty and preference (highly subjective),

- • IQA: technical fidelity and distortion (mixed objective/subjective),
- • ISTA: fine-grained structure and texture realism (more objective).

These dimensions are complementary but nonredundant. High quality does not imply high aesthetics; rich texture does not imply high quality; high aesthetics does not imply rich texture.

Thus, no single dimension adequately explains human perceptual judgments.

###### 7.4. Why Unify Perceptual Understanding?

Unifying IAA, IQA, and ISTA provides key scientific and practical benefits:

- • Shared Perceptual Representations: joint learning encourages a universal perceptual embedding.
- • Better Evaluators for Generative/Restoration Models: unified perceptual signals diagnose model failures beyond semantic metrics.
- • Controllable Generation/Editing: separate perceptual axes support targeted improvements (e.g., enhancing texture without harming quality).
- • Dataset Curation: perceptual scoring allows filtering or reweighting data by aesthetics, fidelity, or structure.
- • Human-centric Applications: real-world systems optimize perceived quality, not just semantics.

###### 7.5. Our Contributions

We introduce a unified paradigm for perceptual-level image understanding. Our contributions include:

- • Unified Task Definition: a coherent taxonomy integrating IAA, IQA, and ISTA. More details are provided in Tab. 10, Tab. 11, Tab. 12 and Sec. 9.1.
- • Holistic Evaluation Protocol: performance measurement across three perceptual dimensions.
- • Unified Baseline Model: a multi-task perceptual model predicting aesthetics, quality, and structure/texture jointly.
- • Downstream Validation: unified perceptual signals improve applications such as reward modeling for posttraining.

###### 8. Details of ISTA

###### 8.1. Structural Annotation

We establish a systematic definition of image structure and texture, as presented in Tab. 12. Building upon this definition framework, we design Structural Annotation scheme for ISTA as shown in Fig. 9. We also employ the following prompt to guide the MLLM in performing the Structural Annotation for ISTA.

{

"SceneType": "<SceneType>", "SceneName": "<SceneName>", "Components": [

{

"ComponentName": "<Component_1>", "DescriptionContent": {

"PhysicalStructure": { "BaseMorphology": ["<

Morphology_1>"], "Arrangement": ["<

Arrangement_1>"]

}, "MaterialRepresentation": {

"MaterialClass": ["< MaterialClass_1>"],

"SurfaceProperties": ["< SurfaceProperty_1>"]

}, "GeometricComposition": {

"PlanarContour": ["<

PlanarContour_1>"], "VolumetricForm": ["<

VolumetricForm_1>"]

}, "SemanticPerception": {

"FunctionalInference": ["<

FunctionalInference_1>"], "StyleType": ["<StyleType_1

>"] }

}

}, {

"ComponentName": "<Component_2>",

... }

} ]

}

Figure 9. Template of ISTA Structural Annotation.

###### Prompt for ISTA Structural Annotation [PRIOR KNOWLEDGE BASE]

- - Base Morphology blotchy, braided, bubbly, bumpy, chequered, cobwebbed, cracked, crosshatched, crystalline, dotted, fibrous, flecked, freckled, frilly, grid, grooved, honeycombed, interlaced, knitted, lacelike, lined, marbled, matted, meshed, paisley, perforated, pitted, pleated, porous, scaly, smeared, spiralled, sprinkled, stratified, striped, studded, swirly, veined, woven, wrinkled, zigzagged, smooth
- - Material Type

1. Natural Materials: Foliage, Grass, Skin, Stone, Wood, Water, Hair

- 2. Man-Made Materials: Brick, Carpet, Ceramic, Fabric, Glass, Leather, Metal, Mirror, Painted Surface, Paper, Plastic, Polished Stone, Tile, Wallpaper, Concrete, Food Surface
- 3. Environmental / Background Textures: Sky, Clouds, Fog / Mist

- - Two-Dimensional Shape Rectangle, Square, Circle, Ellipse / Oval, Triangle, Equilateral Triangle, Isosceles Triangle, Scalene Triangle, Right Triangle, Trapezoid / Trapezium, Parallelogram, Rhombus, Pentagon, Hexagon, Heptagon, Octagon, Nonagon, Decagon, Star, Pentagram, Hexagram, Cross, Arrow, Semicircle, Sector, Crescent, Annulus / Ring, Heart, Lemniscate, Lune / Bow Shape, Spiral, Waveform, Teardrop
- - Three-Dimensional Shape Categories Sphere, Ellipsoid, Cube, Cuboid, Cylinder, Cone, Pyramid, Tetrahedron, Octahedron, Dodecahedron, Icosahedron, Prism, Triangular Prism, Rectangular Prism, Pentagonal Prism, Hexagonal Prism, Torus, Annular Torus, Paraboloid, Hyperboloid, Elliptic Cylinder, Hyperbolic Cylinder, Truncated Cone, Truncated Pyramid, Capsule, Dome, Lens, Bipyramid, Frustum, M¨obius Strip, Knot, Klein Bottle
- - Style Semantics Embossed, Engraved, Rough, Smooth, Matte, Glossy, Brushed, Honeycomb, Geometric, Fractal, Tile Mosaic, Chinese Cloud Pattern, Dragon Scale, Cyberpunk Holographic, Steampunk Mechanical [STRUCTURE TEMPLATE]
- - Scene Decomposition Principles

- A. Single Scene: (Please introduce the description object.)
- B. Composite Scene: (Please introduce the different objects. Describe the object separately) (e.g., street → architectural cluster + pavement system

+ sky background)

- Description Content (In Composite Scene mode, Please indicate the object in a single description.) (e.g., Description Content [street])

- 1. Physical Structure Base Morphology (*) → Select 1-3 terms from the lexicon (Basically comes from Base Morphology) to describe the surface texture (Focus on texture form rather than shape) Arrangement (!) → Describe spatial layout or directionality of texture: orientation (e.g., diagonal, radial), distribution pattern (e.g., clustered, layered), or density changes (optional) Dynamics (!) → Motion/transition states (optional)
- 2. Material Representation Material Class (*) → Select from Material Type

- Surface Properties (!) → Reflectivity/translucency (optional)
- 3. Geometric Composition Planar Contour (!) → 2D shape terms from TwoDimensional Shape (where applicable) (optional) Volumetric Form (!) → 3D form terms from ThreeDimensional Shape Categories (where applicable) (optional)
- 4. Semantic Perception Functional Inference (!) → Only!!! with text/icons present (e.g. The icon is: no traffic, and the text is: XX Street...) (optional) Style Type (!) → Must use style semantics terms (e.g. Baroque style, Xiangyun style...) (optional) [Execution Standards]

- 1. Terminology Enforcement: (*)-marked fields require exact term matches
- 2. Format Purity: Output only structured content, no explanations
- 3. Hierarchy Preservation: Apply complete template per independent unit
- 4. Complexity Adaptation: Single description for simple objects, multi-unit decomposition for complex scenes
- 5. Lexicon Flexibility: For *-marked fields, use official lexicon terms where possible. Free-form extensions are allowed if they add necessary clarity or express phenomena outside the vocabulary.
- 6. Mixed Mode Expression: Structured descriptions may combine fixed taxonomy terms with precise natural language when facing edge cases or fine-grained observations.

[Example] {

"SceneType": "Composite Scene", "SceneName": "Urban skyscraper cluster", "Components": [

{

"ComponentName": "Buildings", "DescriptionContent": {

"PhysicalStructure": { "BaseMorphology": ["grid "],

"Arrangement": [" Vertical,Symmetrical "],

"Dynamics": ["N/A"]

}, "MaterialRepresentation": {

"MaterialClass": ["Glass ","Metal"],

"SurfaceProperties": [" Glossy","Reflective "]

}, "GeometricComposition": {

"PlanarContour": ["

Rectangle"], "VolumetricForm": ["

Cuboid"]

}, "SemanticPerception": {

"FunctionalInference": ["N/A"], "StyleType": ["Modern

Architecture"] }

}

}, {

"ComponentName": "Sky Background ", "DescriptionContent": { "PhysicalStructure": { "BaseMorphology": ["

smooth"], "Arrangement": ["N/A"], "Dynamics": ["N/A"]

}, "MaterialRepresentation": {

"MaterialClass": ["Sky "], "SurfaceProperties": ["N /A"]

}, "GeometricComposition": {

"PlanarContour": ["N/A "], "VolumetricForm": ["N/A "]

}, "SemanticPerception": {

"FunctionalInference": ["N/A"],

"StyleType": ["N/A"] }

} }

] }

Please use the above foundational knowledge and the provided example to perform a texture and structural analysis of the image.

###### 8.2. Definition of ISTA-10K

ISTA-10K is our curated visual rating dataset for evaluating the structure–texture dimension of images. The ISTA score quantifies an image’s structure–texture richness, reflecting both (1) the complexity and diversity of its visual textures, and (2) the richness and organization of its structural components. Images exhibiting more varied textures, materials, geometric forms, and semantically distinct elements are assigned higher ISTA scores.

Texture Intensity Mapping. To quantify the complexity of base morphological patterns, each texture term t is assigned a discrete weight reflecting its perceived structural richness.

Lower weights correspond to visually simple and uniform textures, whereas higher weights indicate more irregular, high-frequency, or structurally complex surface patterns. The weighting function is defined as:

 

- 1, t ∈ Tweak,
- 2, t ∈ Tmedium,
- 3, t ∈ Tstrong, 0, otherwise.

(4)

w(t) =



The three texture sets used in Eq. 4 are summarized in Table 9. These categories are derived from commonly observed visual morphologies and are grouped by increasing structural complexity. Weak textures are visually simple and low-variation patterns with minimal geometric or material irregularities. Medium textures exhibit moderate complexity with richer local variations and more structured arrangements. Strong textures reflect high morphological complexity, featuring irregular, multi-scale, or highly distinctive patterns often associated with heterogeneous or fine-grained natural structures.

Table 9. Texture categories used for texture intensity mapping. Higher groups indicate richer and more complex morphology patterns.

Category (Weight)

Texture Terms

Weak (1) smooth, plain, uniform, lined, grid,

striped, chequered, dotted, freckled

Medium (2) braided, woven, crosshatched, meshed, cobwebbed, lacelike, knitted, spiralled, swirly

Strong (3) bumpy, blotchy, bubbly, cracked, crystalline, flecked, frilly, grooved, honeycombed, marbled, matted, paisley, perforated, pitted, pleated, porous, scaly, smeared, sprinkled, stratified, studded, veined, wrinkled, zigzagged

Component-Level ISTA Rating. For each component c, we compute:

S(c) = SPS(c) + SMR(c) + SGC(c) + SSP(c). (5) These terms quantify different dimensions of structural

and perceptual richness:

- • Physical Structure (PS): describes the surface morphology and spatial arrangement of structural patterns;
- • Material Representation (MR): captures the diversity of material classes and their surface properties;
- • Geometric Composition (GC): reflects variations in planar contours and volumetric forms;

• Semantic Perception (SP): accounts for functional cues and stylistic categories associated with the component Each sub-score is defined as follows (all “N/A” entries excluded):

w(t) + Arrangement(c) ,

###### SPS(c) =

t∈BaseMorphology(c)

- (6)

SMR(c) = MaterialClass(c) + SurfaceProperties(c) ,

- (7)

SGC(c) = PlanarContour(c) + VolumetricForm(c) ,

- (8)

SSP(c) = FunctionalInference(c) + StyleType(c) .

- (9)

Image-Level ISTA Rating. If the image contains a set of components C,

###### SISTA = |C| +

S(c). (10)

c∈C

For images without explicit components, the whole image is treated as a single component:

SISTA = 1 + S(cimage). (11)

Finally, the score is clipped to match the 0–100 rating range:

SISTA ← min(SISTA, 100). (12)

This formulation yields a deterministic and interpretable measure of structure–texture richness that aligns with the hierarchical annotation schema in UniPercept-Bench.

###### 9. Details of UniPercept-Bench

###### 9.1. Definition System

The UniPercept definition system is organized into three levels: Domain–Category–Criterion. The complete set of fine-grained attributes and their brief descriptions are provided in Tab. 10, Tab. 11 and Tab. 12.

###### 9.2. Evaluation Details

For the VQA task, we directly feed the image, question, and answer options into the MLLM during evaluation. For the VR task, if a model provides dedicated interfaces for IAA or IQA scoring, we invoke them directly. For models without such interfaces, we evaluate them using the following prompts:

###### Prompt for Visual Rating (IAA)

Please rate the aesthetics of this image and provide a score between 0 and 100, where 0 represents the lowest quality and 100 represents the highest. Your response should contain only an integer value.

###### Prompt for Visual Rating (IQA)

Please rate the quality of this image and provide a score between 0 and 100, where 0 represents the lowest quality and 100 represents the highest. Your response should contain only an integer value.

###### Prompt for Visual Rating (ISTA)

Please rate the structure and texture richness of this image and provide a score between 0 and 100, where 0 represents the lowest quality and 100 represents the highest. Your response should contain only an integer value.

###### 9.3. Comparison with Other Benchmarks

As shown in Tab. 13, we compare UniPercept-Bench with other widely-used benchmarks in image assessment. UniPercept-Bench provides finer-grained QA categories, supports both rating and textual formats, and employs a scalable annotation pipeline that combines human expertise with MLLM assistance to achieve example-level annotation. It further covers all three perceptual-level domains and includes both VQA and VR tasks, making it the most comprehensive and well-defined benchmark for perceptuallevel image understanding to date.

###### 9.4. Relations Among Different Perceptual Domains

As discussed in Sec.3.1 of the main paper, the three domains (IAA, IQA, and ISTA) characterize distinct dimensions of image assessment and are therefore largely independent. As illustrated in Fig. 10, an image that receives a high score in IQA may perform poorly in IAA. Likewise, images that achieve strong aesthetic scores may exhibit weaker performance in structural or textural assessment.

###### 10. Further Discussion on UniPercept

###### 10.1. Training Dataset

For Domain-Adaptive Pre-Training. As shown in Tab. 14, our Domain-Adaptive Pre-Training is conducted on approximately 800K samples spanning the IAA, IQA, and ISTA domains. For the IAA and IQA domains, the data include both text and rating types: the text data con-

###### Table 10. Definition details of IAA domain.

###### No. Category Criterion Description

- 1 Composition & Design (Comp.)

Visual Balance Assess the distribution of visual elements such as shapes, tones, and colors to determine equilibrium across the frame.

- 2 Composition & Design Hierarchical Emphasis Evaluate the relative prominence of visual elements based on size, contrast, or positioning.
- 3 Composition & Design Structural Organization Examine spatial alignment, grid conformity, and grouping of elements within the image frame.
- 4 Composition & Design Compositional Rhythm Assess repetition, spacing, and directional continuity of elements that suggest visual pacing.
- 5 Composition & Design Harmonic Unity Evaluate visual consistency in shape proportions, relative sizes, and orientation patterns.
- 6 Composition & Design Composition & Design Level Indicates the overall compositional quality — reflecting balance, rhythm, and structural harmony.

- 7 Visual Elements & Structure (VisStr.)

Line Dynamics Assess structure, orientation, and density of linework contributing to visual form.

- 8 Visual Elements & Structure Shape Clarity Assess clarity of 2D shape boundaries and separation from background.
- 9 Visual Elements & Structure Form Realization Evaluate rendering of 3D form through shading, lighting gradients, and perspective cues.
- 10 Visual Elements & Structure Spatial Illusion Judge depth cues such as occlusion, scale variation, and linear perspective.
- 11 Visual Elements & Structure Light Modeling Assess consistency and realism of lighting, highlights, and shadows.
- 12 Visual Elements & Structure Visual Elements & Structure Level Measures mastery of fundamental visual elements including lines, shapes, form, and spatial coherence.

- 13 Technical Execution (Tech.) Material Proficiency Evaluate control and precision in using the medium, including brushstroke discipline.
- 14 Technical Execution Rendering Precision Assess refinement in edge definition, gradient transitions, and micro-level detailing.
- 15 Technical Execution Focus Control Judge sharpness, blur, or depth-of-field usage to structure visual hierarchy.
- 16 Technical Execution Tonal and Exposure Control Evaluate luminance distribution and tonal range.
- 17 Technical Execution Technical Execution Level Represents technical proficiency — tonal control, precision, and rendering quality.

- 18 Originality & Creativity (Creat.)

Concept Innovation Assess distinctiveness of concept or narrative.

- 19 Originality & Creativity Creative Problem-Solving Evaluate ingenuity in visual execution or compositional decision-making.
- 20 Originality & Creativity Originality & Creativity Level Expresses creative strength — innovation, imagination, and conceptual uniqueness.

- 21 Theme & Communication (Theme.)

Subject Clarity Evaluate clarity of subject or message.

- 22 Theme & Communication Narrative Depth Assess symbolic or narrative layering.
- 23 Theme & Communication Cultural Insight Judge engagement with cultural, historical, or social ideas.
- 24 Theme & Communication Theme & Communication Level Evaluates thematic clarity and communication effectiveness.

- 25 Emotion & Viewer Response (Emo.)

Emotional Resonance Evaluate emotional tone and viewer response.

- 26 Emotion & Viewer Response Viewer Engagement Assess long-term viewer compellingness.
- 27 Emotion & Viewer Response Interpretive Openness Judge clarity–ambiguity balance inviting interpretation.
- 28 Emotion & Viewer Response Emotion & Viewer Response Level Measures emotional and psychological impact.

- 29 Overall Gestalt (Gest.) Holistic Cohesion Evaluate integration of visual and conceptual components into a unified whole.
- 30 Overall Gestalt Overall Gestalt Level Represents holistic integration across all components.

- 31 Comprehensive Evaluation (CompEv.)

Comprehensive Evaluation Level Synthesizes all artistic dimensions — perceptual quality and conceptual depth.

###### Table 11. Definition details of IQA domain.

###### No. Category Criterion Description

- 1 Distortion Location (Loc.) Location Description Precisely identify and describe specific spatial regions within the image where distortions are visible or concentrated. The question must explicitly reference a concrete part of the image, not the image as a whole.
- 2 Distortion Location Object Association Specify the semantic or structural objects in the image that are impacted or altered by the distortions. The question must explicitly reference a concrete object, not just a general distortion type.

- 3 Distortion Severity (Sev.) Severity Level Evaluate the severity of distortion in the image as None, Slight, or Obvious, reflecting visibility and perceptual impact.

- 4 Distortion Type (Type.) Distortion Types Present Identify distortion types present in the scene from a comprehensive taxonomy, including blur, noise, compression, brightness, contrast, saturation, sharpening, quantization, exposure, pixelation, color diffusion, jitter, transmission errors, and multidistortion combinations.

###### Table 12. Definition details of ISTA domain.

###### No. Category Criterion Description

- 1 Scene Decomposition Principles (Scene.)

Scene Classification Classify the scene as (A) Single-Object Scene with one primary subject, or (B) Composite Scene containing multiple distinguishable components. Describe the main object(s) clearly.

- 2 Physical Structure (Phys.) Base Morphology Describe surface texture using perceptual descriptors such as fibrous, grooved, marbled, veined, smooth, etc.
- 3 Physical Structure Spatial Arrangement Describe texture orientation, distribution pattern, and density variation across regions (e.g., horizontal, clustered, layered, radial, uniform).

- 4 Material Representation (Mat.) Material Identification Identify the perceived material category using a standardized taxonomy (natural, man-made, or environmental materials) present in the image.
- 5 Material Representation Surface Behavior Describe optical surface properties such as glossiness, translucency, or matte finish.

- 6 Geometric Composition (Geo.) 2D Contour Classify the 2D outline shape using a standardized lexicon including basic shapes, polygons, special forms, or organic/curved forms.
- 7 Geometric Composition 3D Volume Describe the implied 3D volumetric form using a taxonomy of basic solids, polyhedra, prisms, or complex mathematical shapes.

- 8 Semantic Perception (Sem.) Functional Suggestion Infer functional or symbolic implications of textures/motifs based on appearance, referencing standardized functional texture/style descriptors.
- 9 Semantic Perception Stylistic Classification Assign a stylistic category (e.g., Minimalist, Gothic, Art Deco, Futuristic, Cyberpunk, Chinese Cloud Pattern) based on visual elements and decorative cues.

sist of high-quality image–text pairs curated from the source datasets using an MLLM (GPT-4o [1]), while the rating data are directly taken from ArtiMuse-10K [4] and KonIQ10K [15]. For the ISTA domain, due to the lack of highquality annotations, we first extract images that meet the domain requirements from the raw datasets and then, following the definition in Sec. 8.2, employ GPT-4o to generate structured annotations, which are further converted into textual descriptions. Using all the above data, we perform

Domain-Adaptive Pre-Training to equip UniPercept with an initial capacity for perceptual-level image understanding.

For Task-Aligned RL. As shown in Tab. 15, we summarize the data composition used in the Task-Aligned RL stage. For the three domains (IAA, IQA, and ISTA), we combine data from both the VR and VQA tasks. For the VR task, we use the training sets from ArtiMuse-10K, KonIQ10K, and ISTA-10K. For the VQA task, we construct training data following the same pipeline described in Fig. 4

###### Table 13. Comparison with existing benchmarks & datasets.

Perceptual-Level Domain Task IAA IQA ISTA VQA VR

Benchmark # Test # QA Category Data Format Annotation Level Annotator

VQA Benchmarks & Datasets

Q-Bench [58] ∼1.5K – Text Category-Level Human – ✓ – ✓ – AesBench [17] ∼10K 10 Text Example-Level Human ✓ – – ✓ – DQ-495K [67] ∼56K – Text Category-Level Human & MLLM – ✓ – ✓ – Q-Instruct-DB [59] – – Text Category-Level Human & MLLM – ✓ – ✓ – Co-Instruct [61] – – Text Category-Level Human & MLLM – ✓ – ✓ – Q-Ground-100K [5] ∼1K – Text Category-Level Human & MLLM – ✓ – ✓ –

###### VR Benchmarks & Datasets

ArtiMuse-10K [4] ∼1K 15 Rating & Text Example-Level Human ✓ – – – ✓ AVA [38] ∼20K – Rating Example-Level Human ✓ – – – ✓ TAD66K [14] ∼15K – Rating Example-Level Human ✓ – – – ✓ KonIQ-10K [15] ∼2K – Rating Example-Level Human – ✓ – – ✓ SPAQ [11] ∼1K – Rating Example-Level Human – ✓ – – ✓ KADID [30] ∼1K – Rating Example-Level Human – ✓ – – ✓

###### UniPercept-Bench (Ours) ∼6K 44 Rating & Text Example-Level Human & MLLM ✓ ✓ ✓ ✓ ✓

Table 14. Data Overview for Domain-Adaptive Pre-Training.

[Figure 86]

[Figure 87]

Domain Type Size Source

APDDv2 [19] Impressions [24] AVA [38] TAD66K [14] FLICKR-AES [40] Rating ∼9K ArtiMuse-10K [4]

Text ∼360K

IAA

High clarity, but poor composition.

Elegant visuals, but texturally simple.

IAA:30 IQA:89 ISTA: 49

###### IAA:75 IQA: 85 ISTA:39

[Figure 88]

[Figure 89]

Q-Ground-100K [5] DQ-495K [68] DataDepictQA [67, 68] SPAQ [11] KADID [30] PIPAL [13]

Text ∼380K

IQA

Good clarity and lighting, but lack visual appeal。

Modern aesthetics, but limited structural complexity.

Rating ∼7K KonIQ-10K [15]

IAA:41 IQA:81 ISTA: 61

###### IAA:76 IQA: 70 ISTA:39

DTD [9] FMD [45] Big and Small Objects [23] Scene Size x Clutter Database [39] Reachspaces [20] Flickr2K [29], LSDIR [28] Structure ∼40K Same as Text

Figure 10. Relationship between different perceptual domains. Ratings are provided by UniPercept.

Text ∼40K

of the main paper, and denote the resulting training set as UniPercept Data-VQA (train). By jointly leveraging all six types of training sources, the Task-Aligned RL stage enables UniPercept to effectively handle both two task formats and all three perceptual domains.

ISTA

###### 10.2. Ablation Studies

Pre-Training leads to a substantial performance drop on both the VR and VQA tasks. This indicates that a vanilla MLLM initialization exhibits limited perceptual-level understanding, and that training on sufficiently large, domainrelevant data is necessary to equip the model with fundamental perceptual capabilities.

We further investigate the training of UniPercept and evaluate its performance on both the VR and VQA dimensions of UniPercept-Bench. The results are reported in Tab.16 and Tab.17.

###### 10.2.1. Training Strategy

Reward Design. For the VR task, we also consider a threshold-based reward, following the formulation used in Q-Insight [27], which determines correctness solely based

Domain-Adaptive Pre-Training. We empirically verify the importance of Domain-Adaptive Pre-Training. As shown in Tab.16 and Tab.17, removing Domain-Adaptive

Table 15. Data Overview for Task-Aligned RL.

###### Domain Type Size Source

VR ∼9K ArtiMuse-10K [4] VQA ∼10K UniPercept Data-VQA (train)

IAA

VR ∼7K KonIQ-10K [15] VQA ∼10K UniPercept Data-VQA (train)

IQA

VR ∼10K ISTA-10K VQA ∼10K UniPercept Data-VQA (train)

ISTA

on the numerical deviation between the prediction and the ground truth. Let pi and gi denote the predicted and groundtruth scores. A prediction is considered correct if its absolute error falls within a predefined tolerance ϵ, and incorrect otherwise. This produces a binary reward signal that avoids extreme reward magnitudes and encourages predictions to stay within an acceptable range:

rthr(i) =

1, if |pi − gi| < ϵ, 0, otherwise.

(13)

In our experiments, we set the threshold to match the one used by the Adaptive Gaussian Soft Reward in UniPercept. As shown in Tab.16 and Tab.17, the Adaptive Gaussian Soft Reward consistently outperforms the threshold-based formulation and even leads to improvements on the VQA task, where rating rewards are not directly applied. These results highlight the advantages of the Adaptive Gaussian Soft Reward in more accurately capturing the deviation between predicted and ground-truth scores. They also indicate that the VQA and VR tasks share underlying correlations, such that advances in one task can benefit the other.

###### 10.2.2. Training Data

Multi-Task vs. Single-Task. We further investigate the relationship between the VR and VQA tasks. During both Domain-Adaptive Pre-Training and Task-Aligned RL, we separate the VR and VQA training data while keeping all other text data unchanged, and conduct two settings: VQAonly and VR-only. As shown in Tab.16 and Tab.17, the VQA-only model performs poorly on the VR task, and the VR-only model similarly performs poorly on the VQA task. Moreover, both settings underperform UniPercept (VR & VQA) even on their respective target tasks. These results demonstrate that jointly training on both VR and VQA tasks provides substantial mutual benefits and leads to stronger perceptual understanding.

Multi-Domain vs Single-Domain. We also study the relationships among the three perceptual domains: IAA, IQA, and ISTA. During both Domain-Adaptive Pre-Training and Task-Aligned RL, we separate the training data of the three

domains while keeping all other data unchanged, and conduct three single-domain settings: IAA-only, IQA-only, and ISTA-only. As shown in Tab.16 and Tab.17, models trained on a single domain perform well on their corresponding domain but fall short in overall performance. In contrast, UniPercept (trained with a mixture of all three domains) achieves substantially stronger overall results and even surpasses the single-domain models on certain tasks. These findings indicate that, although IAA, IQA, and ISTA focus on different aspects of perceptual assessment, jointly training on all three domains enhances the model’s holistic perceptual understanding.

Table 16. Ablation studies on UniPercept-Bench-VR. The best results are highlighted with dark blue cells, and the second-best results with light blue cells. Metrics: SRCC/PLCC.

IAA IQA ISTA

Models

ArtiMuse-10K [4] Avg. KonIQ-10K [15] Avg. ISTA-10K Ablation on Training Strategy

w/ Threshold Reward 0.604/0.556 0.617/0.596 0.882/0.888 0.801/0.790 0.303/0.334 w/o Adaptive Pre-Training 0.546/0.510 0.481/0.421 0.851/0.817 0.733/0.700 0.755/0.732

###### Ablation on Training Tasks

VQA-Only 0.591/0.582 0.585/0.598 0.816/0.847 0.769/0.774 0.206/0.206 VR-Only 0.629/0.596 0.558/0.509 0.907/0.828 0.794/0.749 0.767/0.767

###### Ablation on Training Domains

IAA-Only 0.621/0.608 0.508/0.464 0.641/0.644 0.706/0.680 0.197/0.197 IQA-Only 0.369/0.352 0.468/0.435 0.901/0.839 0.786/0.726 0.341/0.337 ISTA-Only 0.351/0.319 0.275/0.288 0.595/0.575 0.611/0.570 0.771/0.782

UniPercept (Ours) 0.746/0.738 0.590/0.586 0.940/0.949 0.824/0.827 0.778/0.767

###### Table 17. Ablation studies on UniPercept-Bench-VQA.

Experiments IAA IQA ISTA Avg. Ablation on Training Strategy

w/ Threshold-based Reward 72.32% 76.29% 81.65% 76.75% w/o Adaptive Pre-Training 69.16% 75.09% 80.00% 74.75%

###### Ablation on Training Tasks

VQA-Only 71.92% 76.29% 81.44% 76.55% VR-Only 68.57% 68.38% 75.15% 70.70%

###### Ablation on Training Domains

IAA-Only 73.69% 69.67% 75.57% 72.98% IQA-Only 64.73% 76.01% 77.53% 72.76% ISTA-Only 69.56% 69.58% 82.27% 73.80%

UniPercept (Ours) 76.55% 81.07% 84.23% 80.62%

###### 11. More Examples of UniPercept-Bench

We provide additional examples from UniPercept-Bench in Fig. 11.

###### 12. UniPercept-Constructed Image Profiles

UniPercept is capable of performing comprehensive perceptual-level analysis of images, providing accurate visual-rating evaluations across the IAA, IQA, and ISTA dimensions, together with fine-grained, multi-dimensional analytical outputs. This enables UniPercept to generate a detailed profile for each image. We present examples of UniPercept-Constructed Image Profiles in Fig. 12, Fig. 13, and Fig. 14.

|I A A|Composition & Design|Emotion & Viewer Response|Technical Execution|
|---|---|---|---|
| |Q: What visual element is most<br><br>prominent due to hierarchical emphasis?<br><br>A. Floral design above the circle<br><br>B. Text below the circle<br>C. Cultural attire within the circle<br>D. Historical architecture background<br><br><br>[Figure 90]|Q: What is your<br><br>assessment of the Emotion & Viewer Response quality in this picture?<br><br>A. Low<br>B. Medium<br>C. High<br><br><br>[Figure 91]|Q: Why does the artist use layered brushstrokes on the peony petals?<br><br>A. Simulate natural petal surfaces<br><br>B. Emphasize the golden background<br>C. Obscure imperfections<br>D. To reduce the visual prominence<br><br><br>[Figure 92]|
| |Criterion: Hierarchical Emphasis QA Template: What|Criterion: Emotion & Viewer Response Level QA Template: Level Prediction|Criterion: Material Proficiency QA Template: Why|
|I A A|Originality & Creativity|Visual Elements & Structure|Comprehensive Evaluation|
| |Q: What compositional decision most impacts the sense of grandeur in the image?<br><br>A. Warm wooden tones against red walls<br>B. The pagoda structure<br>C. For scale and context<br>D. Emphasizing vertical layers<br><br><br>[Figure 93]|Q: Why do the line dynamics enhance the monkey's playful expression and pose?<br><br>A. Lines create movement and flow<br><br>B. Lines emphasize facial details<br>C. Lines add structural complexity<br>D. Lines increase color contrast<br><br><br>[Figure 94]|Q: What is your assessment of the Comprehensive Evaluation quality in this picture?<br><br>A. Low<br>B. Medium<br>C. High<br><br><br>[Figure 95]|
| |Criterion: Creative Problem-Solving QA Template: What|Criterion: Line Dynamics QA Template: Why|Criterion: Comprehensive Evaluation Level QA Template: Level Prediction|
|I Q A|Distortion Location|Distortion Location|Distortion Type|
| |Q: How does the lighting affect texture visibility in the foreground of the image?<br><br>A. Enhances stone texture clarity<br><br>B. Causes noticeable blurring<br>C. Creates strong shadow contrasts<br>D. Reduces texture detail visibility<br><br><br>[Figure 96]|Q: How does the spatial clarity differ between the wooden floor and the cat's fur?<br><br>A. Floor grain is sharper than fur texture<br><br>B. Fur texture is sharper than floor grain<br><br><br>[Figure 97]|Q: Which distortion type is evident in the image, affecting color realism?<br><br>A. Gaussian YCbCr noise<br>B. JPEG compression artifacts<br>C. Saturate strengthen YCrCb distortion<br><br><br>[Figure 98]|
| |Criterion: Location Description QA Template: How|Criterion: Location Description QA Template: How|Criterion: Distortion Types Present QA Template: Which|
|I Q A|Distortion Type|Distortion Severity|Distortion Location|
| |Q: What distortion type is most evident in<br><br>this image's color representation?<br><br>A. Gaussian Blur<br>B. JPEG Compression<br>C. Saturation Weaken HSV<br><br>D. Over-Exposure<br><br><br>[Figure 99]|Q: Overall, how would you rate the severity of distortions in this image?<br><br>A. None (no visible distortion)<br>B. Slight (barely noticeable but present)<br><br>C. Obvious (clearly visible and significantly impacts perception)<br><br><br>[Figure 100]|Q: What specific distortion is most noticeable on the lantern’s surface?<br><br>A. Overexposure causing loss of detail<br>B. Blurring obscuring texture details<br><br>C. High contrast creating harsh edges<br><br><br>[Figure 101]|
| |Criterion: Distortion Types Present QA Template: What|Criterion: Severity Level QA Template: Level Prediction|Criterion: Location Description QA Template: What|
|I<br><br>S<br>T A<br>|Geometric Composition|Geometric Composition|Physical Structure|
| |Q: What is the primary 2D contour shape visible in the honeycomb structure?<br><br>A. Square<br>B. Hexagon<br><br>C. Circle<br>D. Pentagon<br><br><br>[Figure 102]|Q: Why is the contour in the image classified as a spiral form?<br><br>A. It has multiple loops.<br><br>B. It forms a closed loop.<br>C. It has a consistent thickness.<br>D. It radiates from a central point.<br><br><br>[Figure 103]|Q: What term best describes the surface texture of the sandy scene?<br><br>A. Grooved<br>B. Pitted<br><br>C. Crystalline<br>D. Braided<br><br><br>[Figure 104]|
| |Criterion: 2D Contour QA Template: What|Criterion: 2D Contour QA Template: Why|Criterion: Base Morphology QA Template: What|
|I<br><br>S<br>T A<br>|Semantic Perception|Material Representation|Material Representation|
| |Q: Which stylistic classification best describes the overall visual theme of the limousine interior?<br><br>A. Futuristic Minimalism<br>B. Modern Luxury<br><br>C. Art Deco<br>D. Cyberpunk<br><br><br>[Figure 105]|Q: Which component exhibits the highest glossiness in surface properties?<br><br>A. Salmon<br>B. Stuffed Pepper<br><br>C. Rice<br>D. Spinach Leaves<br><br><br>[Figure 106]|Q: Does the texture in the image visually correspond to Metal?<br><br>A. Yes<br>B. No<br><br><br>[Figure 107]|
| |Criterion: Stylistic Classification QA Template: Which|Criterion: Surface Behavior QA Template: Which|Criterion: Material Identification QA Template: Yes-No|

###### Figure 11. More Examples in UniPercept-Bench.

[Figure 108]

### UniPercept-Constructed Image Profile

[Figure 109]

[Figure 110]

IAA Rating: 29/100

[Figure 111]

IQA Rating: 35/100

[Figure 112]

ISTA Rating: 46/100

###### Image Aesthetics Assessment (IAA)

###### Compostion & Design

- • The composition is nonexistent, with a completely blurred and indistinct subject ...
- • There is no balance or rhythm to speak of. The elements are randomly scattered ... Technical Execution
- • The image exhibits catastrophic technical execution.
- • Every possible mistake is present, with extreme blurriness rendering all details invisible.
- • The focus is nonexistent, resulting in a complete lack of sharpnessness.

... Comprehensive Evaluation

- • The image suffers from extreme blurriness, rendering any aesthetic evaluation impossible.
- • The lack of focus eliminates any potential visual impact, technical merit, or depth. Every element, from the pergola to the lanterns, is indistinct, failing to convey any meaningful ...
- • In summary, the image's total absence of visual coherence results in a zero aesthetic merit.

###### Distortion Location

- • The entire scene, including the pergola, lanterns, and background foliage, is out of focus, resulting in a complete loss of sharpnessness. This blurring affects every element uniformly, making it impossible to identify ...
- • The image fails to convey any meaningful content due to its total lack of clarity. Distortion Severity
- • The image suffers from extreme blurriness.
- • This catastrophic blurring negates any possible evaluation of other distortion types, as nothing is visible.
- • The final distortion severity is the maximum imaginable.

... Overall Quality

- • The image is severely out of focus, rendering all elements indistinct and lacking any discernible detail.
- • The lack of sharpnessness, exposure, and any visible content results in a complete failure in image quality. This extreme blurriness negates any possible merit, making it impossible to evaluate any positive attributes.
- • In summary, the image quality is catastrophic, with zero merit due to total blurriness.

Image Quality Assessment (IQA)

Scene Decomposition Principles

- • Scene Type: Blurred Composite Scene
- • Scene Name: Garden with pergola and hanging lights Componet: Pergola
- • Physical Structure: [Base Morphology: Lined] [Arrangement: Horizontal]
- • Material Representation: [Material Class: Glass] [Surface Properties: Glossy]
- • Geometric Composition: [Volumetric Form: Sphere] Componet: Background Foliage
- • Physical Structure: [Base Morphology: Matted] [Arrangement: Layered]
- • Material Representation: [Material Class: Foliage] [Surface Properties: Matte]
- • Geometric Composition: [Volumetric Form: Naturalistic] Componet: Hanging Lights
- • Physical Structure: [Base Morphology: Smooth] [Arrangement: Clustered]
- • Material Representation: [Material Class: Glass] [Surface Properties: Glossy]
- • Geometric Composition: [Volumetric Form: Sphere]

Image Structure & Texture Assessment (ISTA)

...

[Figure 113]

#### UniPercept-Constructed Image Profile

[Figure 114]

[Figure 115]

IAA Rating: 87/100

[Figure 116]

IQA Rating: 75/100

[Figure 117]

ISTA Rating: 56/100

###### Image Aesthetics Assessment (IAA)

###### Originality & Creativity

- • The image depicts a well-preserved historical house with a straightforward and classic presentation.
- • While the composition and clarity are excellent, the photograph adheres closely to traditional documentation styles, lacking experimental or imaginative elements.

###### Emotion & Viewer Response

- • The image evokes a strong sense of nostalgia and admiration for historical architecture, effectively capturing ...
- • The well-preserved colonial house and lush green create a tranquil, inviting atmosphere, likely sparking curiosity and appreciation for historical preservation. While the composition is perfect and visually pleasing, it may ...

###### ... Comprehensive Evaluation

- • The image showcases a well-preserved historical house with strong visual impact due to its symmetrical ...
- • Overall, the image excels in historical representation but slightly lacks dynamic engagement. The aesthetic merit is strong, reflecting both historical and visual excellence.

###### Distortion Type / Location / Severity

- • There is no distortion in the image.

... Overall Quality

- • The image showcases a well-preserved historical building with excellent clarity and vibrant colors, capturing the architectural details and surrounding green effectively.
- • The composition is balanced, with well-lit, sharp focus, and excellent exposure.
- • The cobblestone path and lush trees enhance the scene's authenticity.
- • There are no visible flaws, indicating perfect sharpnessness and color accuracy.
- • This image excels in every aspect, achieving perfection.
- • Overall, this is a flawless, high-quality photograph.

Image Quality Assessment (IQA)

Scene Decomposition Principles

- • Scene Type: Composite Scene
- • Scene Name: Historic house with surrounding environment Componet: House
- • Physical Structure: [Base Morphology: Lined, Smooth] [Arrangement: Horizontal]
- • Material Representation: [Material Class: Wood] [Surface Properties: Matte]
- • Geometric Composition: [Planar Contour: Rectangle] [Volumetric Form: Cuboid]
- • Semantic Perception: [Style Type: Colonial Architecture] Componet: Roof
- • Physical Structure: [Base Morphology: Tiled] [Arrangement: Layered]
- • Material Representation: [Material Class: Tile] [Surface Properties: Matte]
- • Geometric Composition: [Planar Contour: Rectangle]
- • Semantic Perception: [Style Type: Traditional] Componet: Chimneys
- • Physical Structure: [Base Morphology: Brick] [Arrangement: Vertical]
- • Material Representation: [Material Class: Brick] [Surface Properties: Matte]
- • Geometric Composition:[Planar Contour: Rectangle] [Volumetric Form: Cuboid]
- • Semantic Perception: [Style Type: Traditional]

Image Structure & Texture Assessment (ISTA)

...

[Figure 118]

##### UniPercept-Constructed Image Profile

[Figure 119]

[Figure 120]

IAA Rating: 75/100

[Figure 121]

IQA Rating: 81/100

[Figure 122]

ISTA Rating: 82/100

###### Image Aesthetics Assessment (IAA)

###### Visual Elements & Structure

- • The image excels in visual elements and structure, showcasing a harmonious blend of vibrant colors ...
- • The lush green of the jungle contrasts beautifully with the bright orange background, enhancing visual impact. The composition is meticulously balanced, with dynamic focal points like the pyramid and characters, creating a cohesive and engaging scene.

###### Theme & Communication

- • The image masterfully communicates the theme of a jungle tomb raid through its vivid and intricate LEGO ...
- • Every element, from the pyramid to the characters and animals, is meticulously crafted to evoke a sense of adventure and danger. The narrative is clear and engaging, effectively capturing the essence of Indiana Jones.

###### Overall Gestalt

- • The image showcases a masterpiece of LEGO artistry, seamlessly blending every element to perfection.
- • The overall gestalt is nothing short of perfection, with every detail meticulously crafted to create a cohesive, enchanting jungle scene. The vibrant colors, intricate details, and harmonious composition evoke a sense ...

...

###### Image Quality Assessment (IQA)

###### Distortion Type / Location / Severity

- • There is no distortion in the image.

... Overall Quality

- • The image is sharp and well-composed, showcasing a detailed LEGO jungle scene without any distortion.
- • The vibrant colors and intricate details are intact, with no signs of blurring, overexposure, or compression artifacts. The scene's perfection is evident in its clarity and balance. There are no visual flaws.
- • The lighting and composition are flawless, enhancing the intricate details and depth.
- • There are no visible flaws, making it a flawless representation.
- • This image excels beyond perfection, meriting a perfect score. In summary, the image is flawless in every aspect, showcasing impeccable quality.

Image Structure & Texture Assessment (ISTA)

###### Scene Decomposition Principles

- • Scene Type: Composite Scene
- • Scene Name: Lego Jungle Temple Componet: Temple Structure
- • Physical Structure: [Base Morphology: Blocky, Grid] [Arrangement: Layered]
- • Material Representation: [Material Class: Plastic] [Surface Properties: Matte]
- • Geometric Composition: [Planar Contour: Rectangle] [Volumetric Form: Cuboid] Componet: Palm Trees
- • Physical Structure: [Base Morphology: Fibrous, Frilly] [Arrangement: Vertical]
- • Material Representation: [Material Class: Plastic] [Surface Properties: Matte]
- • Geometric Composition: [Volumetric Form: Cylinder] Componet: Flora
- • Physical Structure: [Base Morphology: Matted, Frilly] [Arrangement: Clustered]
- • Material Representation: [Material Class: Plastic] [Surface Properties: Matte] Componet: Figures
- • Physical Structure: [Base Morphology: Blocky] [Arrangement: Clustered]
- • Material Representation: [Material Class: Plastic] [Surface Properties: Matte]

...

