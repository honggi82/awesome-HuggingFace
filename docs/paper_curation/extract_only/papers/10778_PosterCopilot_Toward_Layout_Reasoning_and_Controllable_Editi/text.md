## PosterCopilot: Toward Layout Reasoning and Controllable Editing for Professional Graphic Design

# arXiv:2512.04082v1[cs.CV]3Dec2025

Jiazhe Wei1∗, Ken Li1∗, Tianyu Lao2, Haofan Wang2, Liang Wang1,3, Caifeng Shan1, Chenyang Si1† 1PRLab, Nanjing University 2LibLib.ai 3 Institute of Automation, Chinese Academy of Sciences Project Page: https://postercopilot.github.io/

[Figure 1]

Figure 1. Generated results from our PosterCopilot. PosterCopilot exhibits exceptional graphic design capabilities by creating artworks with professional-grade layout, compelling visuals, and cohesive themes.

#### Abstract

Graphic design forms the cornerstone of modern visual communication, serving as a vital medium for promoting cultural and commercial events. Recent advances have explored automating this process using Large Multimodal Models (LMMs), yet existing methods often produce geometrically inaccurate layouts and lack the iterative, layer-specific editing required in professional workflows. To address these limitations, we present PosterCopilot, a framework that advances

*Equal Contribution †Corresponding author (chenyang.si@nju.edu.cn)

layout reasoning and controllable editing for professional graphic design. Specifically, we introduce a progressive three-stage training strategy that equips LMMs with geometric understanding and aesthetic reasoning for layout design, consisting of Perturbed Supervised Fine-Tuning, Reinforcement Learning for Visual-Reality Alignment, and Reinforcement Learning from Aesthetic Feedback. Furthermore, we develop a complete workflow that couples the trained LMMbased design model with generative models, enabling layercontrollable, iterative editing for precise element refinement while maintaining global visual consistency. Extensive experiments demonstrate that PosterCopilot achieves geomet-

rically accurate and aesthetically superior layouts, offering unprecedented controllability for professional iterative design.

#### 1. Introduction

Graphic design serves as a fundamental medium for visual communication [36], translating abstract ideas into clear and engaging visuals. It brings together images, text, and graphic elements in a deliberate way to create layouts that are both informative and visually appealing, bridging creativity with effective communication [1]. Recently, growing interest has emerged in automating the graphic design process through artificial intelligence. One major line of work explores diffusion-based generative models, which leverage their strong image synthesis capabilities to create visually rich posters [17, 31, 55]. However, because these models generate all image regions simultaneously, they struggle to preserve the structural integrity, texture fidelity, and stylistic consistency of user-provided assets, making local refinements prone to distortion [16, 35]. Another line leverages Large Multimodal Models (LMMs) to reason over design elements and predict their spatial and layer-wise arrangements, determining each element’s position, scale, and ordering within the composition [18, 43, 57]. These methods preserve the authenticity of visual assets and introduce interpretability and controllability into the design process, representing a promising step toward layout-centric and automationoriented graphic design.

Despite these advances, current LMM-based methods still exhibit notable limitations when applied to professional design workflows: 1) when handling complex and numerous assets, existing methods often produce inaccurate and unaesthetic layouts [34, 56] as shown in Fig. 2. We identify that existing methods rely on supervised fine-tuning (SFT) over discrete textual tokens to represent continuous spatial coordinates, creating a mismatch between the model’s symbolic representation and the true Euclidean geometry of layout design [24, 30]. This mismatch leads to misalignment, distortion, and suboptimal compositions. Moreover, these models lack visual feedback during training, which limits their ability to perceive and reason about aesthetic layouts [43, 44]. 2) More critically, current LMM-based approaches merely generate initial drafts and lack interactive editing capabilities [26, 37]. However, professional designers refine the drafts through multiple rounds of precise, layer-specific adjustments [3, 20, 39]. Therefore, enabling iterative refinement is a crucial requirement for advancing AI-assisted graphic design toward practical applications [25, 40].

To address these challenges, we propose PosterCopilot, which advances the field toward layout reasoning and controllable editing for professional graphic design. Specifically, to mitigate the inaccurate and unaesthetic layouts resulting

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

Figure 2. Some failure cases created by existing design models in real-world, multi-asset scenarios, producing severe misalignments and visual discord.

from token-based coordinate representations, we propose Perturbed Supervised Fine-Tuning (PSFT), which reformulates coordinate regression into a distribution-based learning paradigm by introducing controlled perturbations to groundtruth coordinates. Compared to point-wise regression, learning a coordinate distribution allows the model to reason over continuous spatial relationships rather than memorizing discrete positions [29, 61], leading to more coherent and aesthetically balanced layouts. To further address the lack of visual feedback and aesthetic understanding, we introduce a two-stage reinforcement learning (RL) strategy. In the first stage, Reinforcement Learning for Visual-Reality Alignment (RL-VRA) introduces verifiable geometric reward signals to explicitly correct residual spatial inaccuracies after SFT. In the second stage, Reinforcement Learning from Aesthetic Feedback (RLAF) employs a learned aesthetic reward model to encourage the model to generate aesthetically coherent and diverse compositions that extend beyond the ground truth.

Finally, to enable iterative and controllable refinement beyond initial generation, we develop a complete workflow that couples the trained LMM-based design model with the generative models, seamlessly integrating asset creation with precision editing. This workflow supports layer-specific, iterative editing, allowing precise modification of individual elements while maintaining global visual consistency. It empowers designers with multi-round, high-fidelity editing capabilities, enabling flexible adjustments to specific layers without altering surrounding content. Experimental results indicate that the design model trained via our three-stage method produces layouts that are both accurate and visually appealing, even rivaling or surpassing the Nano-Banana. More significantly, PosterCopilot’s integration of a generative agent provides precise layer-wise editing. This transforms it into a powerful assistant, allowing designers to take a well-composed draft as a starting point and have it further optimized for enhanced aesthetics and practical application.

Our main contributions are summarized as follows:

- • We propose PosterCopilot, the first framework to decouple complex poster design into layout reasoning and multiround lossless editing, demonstrating exceptional capabilities in both aspects.
- • We introduce a progressive alignment training paradigm (PSFT, RL-VRA, RLAF) that enables LMMs to reason over continuous spatial relationships while instilling design principles and human aesthetics.
- • We design a generative agent that supports iterative, controllable refinement beyond the initial generation, empowering PosterCopilot to serve as a powerful assistant for real-world editing scenarios.
- • We contribute a large-scale, high-quality multi-layer poster dataset with rational granularity, along with its construction pipeline, addressing critical gaps in data scarcity and layer segmentation to benefit future research and applications.

#### 2. Related Work

Multi-layer Graphic Layout Planning prioritizes real-world practicality by first inferring layouts, then assembling layers for optimal flexibility. LMM-assisted approaches (LayoutPrompter [32], LayoutNUWA [52], PosterLLaVA [63]) employed in-context learning, while others specialized in asset integration (Graphist [9]), typography (POSTA [4]), or external generation (CreatiPoster [68], COLE [23]). Crucially, these methods mimic static datasets rather than learning from aesthetic outcomes. Our approach transcends limitations by internalizing layout principles and visual aesthetics through direct generative feedback. More discussion is in supplementary material.

#### 3. Methodology

In this section, we will first detail the training paradigm for the design model, and subsequently present the complete PosterCopilot pipeline. Our three-stage design model training paradigm is illustrated in Fig. 3.

- 3.1. Task Formulation Our objective is to automatically arrange user-provided ele-

ments E = {e1,...,eN} of types T = {image,text,shape} on a canvas, achieving aesthetic coherence while preserving asset fidelity. Text elements are rasterized into image layers for unified processing. The input elements and canvas dimensions (Hc,Wc) are encoded into a multimodal prompt PHE

c,Wc, which our design model M processes to generate the final layout:

M(PHE

c,Wc) → G (1)

where G = {(bi,li)}Ni=1 specifies each element’s bounding box bi and layer order li.

###### Perturbed SFT Loss

###### Design Model

Layers + Prompt Tokenization Prediction GroundPerturbedTruth

(a) Perturbed Supervised Fine-Tuning (PSFT)

###### GRPO

[Figure 12]

###### Verifiable Reward )

"layers": {

###### Design Model

"image_id": 2,

- "x": 967,
- "y": 1654, "w": 1361, "h": 106

}

Layers + Prompt

Tokenization Prediction

……

(b) Reinforcement Learning for Visual-Reality Alignment (RL-VRA)

GRPO

[Figure 13]

Aesthetic Reward

Prompt: You are a graphic design reviewer. Your task is to evaluate ……

###### Design Model

Aesthetic Layers + Prompt Feedback

Rendering To Image

Tokenization Prediction

(c) Reinforcement Learning from Aesthetic Feedback (RLAF)

Figure 3. Overview of the training paradigm of PosterCopilot. Rather than formulating the training process as a simple regression task, we endow PosterCopilot with outstanding layout capabilities and human-like aesthetics through a three-stage training paradigm.

##### 3.2. Perturbed Supervised Fine-Tuning

We posit that the standard LMM practice of quantizing continuous coordinates into discrete text tokens fundamentally warps the optimization space’s geometry [11, 19, 48], hindering precise localization. To validate this, we visualize the local geometric uniformity using det(S), the determinant of the Structure Tensor S [2, 15]. As shown in Fig. 4, the ideal Euclidean space (a) has det(S) ≡ 1, whereas the text-represented numerical space (b) is geometrically broken. Critically, (c) confirms that neighborhood averaging—our core insight—effectively repairs this distortion and recovers a stable optimization signal.

Based on this finding, we propose Perturbed Supervised Fine-Tuning (PSFT). Instead of point-wise regression on ground-truth layout Ggt = {(bi,li)}Ni=1, we sample n perturbed variants Gpert(i) by injecting Gaussian noise specifically on the bounding box values bi:

###### G(perti) ∼ N(Ggt,σ2I), i = 1,2,...,n (2)

[Figure 14]

- Figure 4. Geometric instability of text-based coordinate representations. (a) Euclidean Space: The ideal baseline, showing perfect, uniform geometry (det(S) ≡ 1). (b) Text-Based Space: Suffers from signal collapse (near-zero det(S)) and geometric noise, creating a chaotic landscape unstable for optimization. (c) Reconstructed Space via Neighborhood Averaging: This method suppresses noise, recovering a smooth, uniform geometry that is far more stable than (b).

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Aspect ratio distortion

Text Overlay

Aspect ratio distortion

Aspect ratio distortion

Bounding box offset

- Figure 5. Our motivation for visual-reality alignment and aesthetic feedback stems from the observation that design models frequently produce works that violate fundamental graphic design principles, as well as exhibit serious aesthetic flaws. We use red, green, and blue boxes to mark the error areas in the figure.

where σ is a small standard deviation. Our training objective, LPSFT, combines the standard cross-entropy loss on the original layout with an averaged loss over n perturbations:

###### LPSFT = LCE(Gˆ ,Ggt)

1 n

+ λPerturbed ·

n

LCE(Gˆ ,G(perti) )

i=1

(3)

where Gˆ is the model’s prediction. This formulation compels the model to learn a continuous spatial distribution centered on the ground truth, rather than memorizing discrete token positions, thereby mitigating the limitations of text-based regression.

##### 3.3. Reinforcement Learning for Visual-Reality Alignment

While PSFT offers a robust spatial prior, its dependence on supervised learning without visual feedback results in geometric flaws, such as bounding box drift and aspect ratio distortion. Critically, these rendering-stage errors, evident in Fig. 5, cannot be captured easily within the SFT paradigm itself. To bridge this visual-reality gap and align model outputs with graphic design principles, we introduce the Reinforcement Learning for Visual-Reality Alignment (RL-VRA) phase.

We frame RL-VRA as an online policy optimization task under a single-step Markov Decision Process (MDP). The state s corresponds to the input prompt PHE

c,Wc, while the action a represents the layout generation G = {(bi,li)}Ni=1. Our objective is to refine the pre-trained SFT policy πref(G | s) into an enhanced policy πθ(G | s) by maximizing the expected return under a geometry-aware reward signal:

JV RA(θ) = EG∼π

θ(·|s) [r(G)] − βDKL(πθ(· | s)||πref(· | s))

(4)

where JV RA(θ) balances reward maximization against policy conservatism, with πref serving as the frozen reference policy, β controlling the KL regularization strength [47], and r(G) providing dense verifiable geometric visual feedback. To ensure stable policy updates for high-dimensional discrete action spaces, we employ Group Relative Policy Optimization (GRPO) [49], which operates without explicit value function estimation. For each group of K policy rollouts, we compute:

K

1 K

Ai = r(Gi) −

r(Gj)

(5)

j=1

πθ(Gi | s) πθ

ri(θ) =

(Gi | s)

old

where Ai represents the advantage of action Gi relative to the group, and ri(θ) is the probability ratio between the new and old policies. Our reward function r(G) = rSpatial + rElement + rformat provides multi-scale geometric supervision, decomposing layout quality into spatial coherence and element-level fidelity components.

The spatial reward rSpatial addresses layout misalignment through Distance Intersection over Union (DIoU) [71]:

rSpatial = rDIoU =

i

ρ2(bi,bgti ) c2

IoU(bi,bgti ) −

(6)

where ρ denotes the center distance, c represents the diagonal of the minimal enclosing box, and bgti is the ground-truth box from Ggt.

The element-level reward rElement = rAR + rsize penalizes geometric distortions that compromise visual integrity. The aspect ratio reward:

rAR = −

i

log

wi/hi wigt/hgti

preserves element proportions, while the size reward:

rsize = −

i

smoothδ

wi − wigt wigt

+ smoothδ

hi − hgti hgti

(7)

(8)

Various Input Modes Generative Agent

###### Design Master

[Figure 21]

[Figure 22]

- 1-1 : Change hair style and shave

: 1. Color layer_1

- 2. Shave the man
- 3. Dye his hair yellow

: Planning new layout

2

3

Draft Design

Final Design

- 1-2

- 1-3

Canvas:

[Figure 23]

[Figure 24]

###### Reception Model

[Figure 25]

[Figure 26]

Requirement: I want to create a modern, minimalist poster for an online launch……

W: 2480 H: 3508

[Figure 27]

[Figure 28]

- 1

- 2

- 3

Planning Layout Prompt

Background Prompt

It should feature a dark, subtle gradient transitioning from a deep charcoal grey at the top to a slightly lighter……

[Figure 29]

###### Design Model

Part-provided assets/ Prompt Only

Foreground Prompt 1

……

[Figure 30]

A prominent portrait of a young man……

Generating Layout

[Figure 31]

Layout:

Canvas_Size: width: 2480, height: 3508 { image_id: 1, x: 144, y: 99……

Fully--provided assets

[Figure 32]

T2I Model

[Figure 33]

image_id: 8, x: 712, y: 1972…… }

[Figure 34]

Generated Full Layers

[Figure 35]

[Figure 36]

……

Rendering to Image

[Figure 37]

###### Frozen Weights Trainable Weights

[Figure 38]

- Figure 6. Overview of PosterCopilot’s Inference and Editing Pipeline. The standard inference and multi-round editing pipelines are marked by blue and red numbers, respectively. Before layout design, PosterCopilot can supplement new assets when design materials are insufficient. Generative Agent first processes user requirements, undergoes professional planning, and delivers complete assets. Design Master then generates optimal compositions based on the assets and requirements, ultimately rendering the Draft design. The draft design will be revised into the final design after multiple rounds of editing by the collaboration of both generative agent and design master.

maintains original dimensions using the Huber loss [13]:

0.5d2/δ |d| < δ |d| − 0.5δ otherwise

smoothδ(d) =

(9)

where δ controls the transition between quadratic and linear regimes, preventing reward domination during extreme size distortions.

We further incorporate rformat to enforce JSON-structured outputs. The complete reward formulation:

+rformat (10)

###### r(G) = rDIoU

+λsizersize + λARrAR

Element Fidelity

Spatial Coherence

where λsize,λAR > 0 balance reward components. This geometrically-grounded reward structure injects explicit visual-reality constraints directly into the policy gradient updates, enabling the model to learn corrective behaviors that transcend the limitations of previous methods that lack visual feedback during training.

##### 3.4. Reinforcement Learning from Aesthetic Feedback

While prior stages enforce graphic design rules based on a single ground-truth, this is just one of many aesthetically valid solutions. To align with broader human aesthetic preferences, we introduce the Reinforcement Learning from Aesthetic Feedback (RLAF) stage. This stage explores a wider design space using a new subjective reward, raes(G),

provided by a pre-trained LMM (acting as an aesthetic judge) that evaluates the final rendered image. This aesthetic score is combined with our format reward rformat:

rRLAF(G) = rformat + λaesraes(G) (11)

where λaes > 0. This stage encourages the model to discover novel, high-appeal layouts that may surpass the ground-truth.

##### 3.5. Unleashing the Creative Flow: Generative Asset Synthesis and Iterative Refinement

With our design model, we now unleash its creative potential by integrating a generative agent that completes the PosterCopilot framework. This integration transforms the model from a pure layout planner into a comprehensive design partner, capable of both asset synthesis and iterative editing. As shown in Fig. 6, this agent first addresses the issue of missing assets: when provided with only partial assets, it can adaptively generate new, style-consistent elements to complete the layout. Specifically, we utilize a trained LMM called the reception model to generate textual descriptions for each missing layer, which are then combined with existing assets as style reference images to be fed together into a text-to-image (T2I) model to generate the corresponding assets. More importantly, the generative agent supports fine-grained, multi-round editing required in professional workflows by accepting user instructions to perform targeted modifications on corresponding layers. This enables designers to perform stable, iterative cycles be-

[Figure 39]

|[Figure 40]<br><br>[Figure 41]<br><br>Shape 1 Image 1<br><br>[Figure 42]<br><br>Image 2<br><br>[Figure 43]<br><br>Text 4<br><br>[Figure 44]<br><br>Shape 2<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>Text 1<br><br>Text 2<br><br>Text 3<br><br><br>Text 5<br><br>Text 6|
|---|

[Figure 50]

Canvas: W: 1753 H: 2480

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Prompt:

[Figure 55]

Subject & Theme: A vibrant, playful, and youthful promotional poster for a youth event or celebration. The overall aesthetic is cheerful, slightly whimsical, and modern, targeting a young demographic. Layout & Composition: A vertically oriented poster with a dynamic and slightly cluttered (in a stylized way) layout. A dominant central figure occupies the lower twothirds, with text and graphic elements arranged around her.

###### Canvas size + Prompt Layers Layout Design by PosterCopilot

- Figure 7. Poster generated from fully-provided assets by PosterCopilot.

tween ’precise single-layer asset editing’ and ’global layout re-arrangement’, while effectively mitigating common challenges in traditional editing methods, such as asset distortion and uncontrollable edit scopes.

#### 4. Application

Harnessing its powerful reasoning capability and finegrained layer-wise architecture, PosterCopilot unlocks diverse applications in professional design scenarios.

- 4.1. Poster Generation from Fully-provided Assets As shown in Fig. 7, PosterCopilot excels at arranging a complete set of user-provided assets into an aesthetically pleasing, professional-grade design, while guaranteeing every asset is faithfully preserved without alteration.
- 4.2. Poster Generation from insufficient Assets PosterCopilot’s generative agent handles incomplete assets by synthesizing missing layers, such as background or foreground layers, with stylistic consistency. This capability, as shown in Fig. 8, accelerates the initial design phase by enabling rapid drafts generation where synthesized elements blend harmoniously with user-provided assets.
- 4.3. Multi-round fine-grained Edit PosterCopilot supports precise, multi-round editing of poster drafts. This functionality encompasses a diverse range of edit types, which we demonstrate in the following.

- 4.3.1. Single Layer Edit As shown in Fig. 9, PosterCopilot supports multiple, varied edits on a single, fine-grained layer (e.g., modifying a camera’s material or a character’s pose). This high-fidelity process strictly confines the edit scope to the target layer, ensuring precise modification while preserving all other elements. This approach avoids the distortion common in diffusion-based methods that edit the entire poster.
- 4.3.2. Theme Switch Fig. 10 demonstrates the ”Theme Switch” capability, enabling holistic theme migration through targeted, multiround edits. For instance, users can swap ”lollipop” elements for ”ice cream,” transforming the poster’s theme (e.g.,

”lollipop sale” to ”ice cream promotion”) while perfectly preserving the original layout and decorative elements.

- 4.3.3. Poster Reframe Leveraging the design model’s powerful reasoning capability, PosterCopilot can intelligently reframe and regenerate appropriate layouts simply by modifying the canvas size specification in the input requirements. Fig. 11 presents examples of poster reframing by PosterCopilot.
- 5. Experimental Details

- 5.1. PosterCopilot Datasets A long-standing challenge in constructing high-quality, multi-layer poster datasets is over-segmentation, where a single visual element is fragmented across multiple independent layers [73] (e.g., a shoe decomposed into separate layers for its laces, sole, and body). To solve this, we developed a novel construction pipeline. As illustrated in Fig. 12, we employ OCR-based fine-granularity bounding box to merge overly fine-grained layers and filter out redundant ones. The refined dataset comprises 160K posters, encompassing a total of 2.6M layers (1.2M text and 1.4M image/decorative).
- 5.2. Experimental Setup Implementation: Our design model employs Qwen-2.5VL-7B-Instruct [54] as backbone; the generative agent employs Qwen-Image-Edit-2509 [58] as T2I model; the reception model uses Qwen-2.5-7B [53]; and RLAF utilizes VisualQuality-R1 [59] as reward model. All experiments run on 8×RTX H20 GPUs. Baselines: We compare against: (1) commercial platforms (Microsoft Designer, Nano-Banana); (2) academic SOTAs (LaDeCo [33], CreatiPoster [68]); and (3) reasoning models (Gemini 2.5 Pro [10], Qwen-VL-2.5-72B-Instruct [54]). Metrics: Following expert consultation, we evaluate the quality of the posters generated via ratings on key metrics for graphic design: Layout Rationality [12, 69], Text Legibility [5], Element Preservation [31], Style Consistency [51], Instruction Following [46] and Visual Appeal [46] for holistic poster quality evaluation, complemented by quantitative IoU, Inverse order pair ratio (IOPR) [9], and Aspect Ratio Distortion (ARD) [72] for ablation study. Evaluation Procedure: We performed human evaluation, supplemented by GPT-5 [41] as an extra reliable evaluator. For human evaluation, we conducted pairwise, binary-choice comparisons against each baseline. We sampled 25 examples per baseline, all generated from identical prompts and fully-provided assets. We collected 5 judgments per example, totaling 750 responses from over 40 evaluators with graphic design backgrounds. For GPT-5 evaluation, we used in-context learning to align the model with our scoring criteria, ensuring a strict and fair assessment of all designs. We prompted GPT-5 to evaluate all results ten times, taking the average of its ratings as the final score for each method.

[Figure 56]

|Text 2<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>Text 4<br><br>[Figure 61]<br><br>Text 3<br><br><br>[Figure 62]<br><br>Text 5<br><br>Text 1<br><br>[Figure 63]<br><br>Image 1<br><br>Gen Image 1<br><br>[Figure 64]<br><br>Shape 1|
|---|

[Figure 65]

|Text1<br><br>Text 4<br><br>Text 3<br><br>Text 5<br><br>Text 2<br><br>Image 1<br><br>Gen Image 1<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]|
|---|

Text1

[Figure 73]

|Text 6<br><br>Text 3<br><br>Gen Image 1 Gen Image 2<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>Text 4<br><br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>Text 1<br><br>[Figure 81]<br><br>Text 8<br><br>Text 2<br><br><br>Shape 1<br><br>[Figure 82]<br><br>Text 5<br><br>[Figure 83]<br><br>Text 7<br><br>[Figure 84]<br><br>Text 9<br><br>[Figure 85]|
|---|

Add Background Layer Layout Design

Add Foreground Layer Layout Design

Add Multiple Layers Layout Design

- Figure 8. Posters generated from insufficient assets by our PosterCopilot.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

- Figure 9. Multi-round refinement for a single layer by our PosterCopilot.

[Figure 92]

|Text 2<br><br>Text 1<br><br>Text 3<br><br>Text 4<br><br>Image 1<br><br>Shape 1<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>Shape 2 Shape 4<br><br><br>[Figure 103]<br><br>Shape 3<br><br>Image 3<br><br>Image 2|
|---|

|Text 2 Text 1<br><br>Text 3<br><br><br>Image 1<br><br>Shape 1<br><br>Image 3 Image 4 Image 5<br><br>Image 6<br><br>Image 2<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]|
|---|

[Figure 114]

[Figure 115]

[Figure 116]

- Gen Image 1

[Figure 117]

- Gen Image 2

bicycle to motorcycle, shift to center

- Gen Image 1

- Gen Image 2

Lollipop to ice cream，add a cute cat

[Figure 118]

[Figure 119]

Lollipop to ice cream

bicycle to motorcycle

[Figure 120]

All other assets remain unchanged

All other assets remain unchanged

###### Theme Switch(Bicycle to motorcycle) Final Design

Craft Design Theme Switch(Lollipop to ice cream) Final Design Craft Design

Figure 10. Multi-round refinement for theme switch by our PosterCopilot.

More information about Experimental Details can be found in supplementary material.

#### 6. Results and Analysis

- 6.1. Comparison with baselines Results of human evaluation is as shown in Fig. 13, PosterCopilot’s average win rate is well above 74% across all baselines. While LMM-based methods such as LaDeCo perform poorly on Layout Rationality and T2I models like Nanobanana struggle with Element Preservation, PosterCopilot preserves all user-provided elements while delivering harmonious, aesthetically pleasing designs. For GPT-5 evaluation, while GPT-5 excels at holistic quality assessment, it struggles with ”instruction following” and ”element preservation” as it cannot reliably process the source assets for these tasks. Consequently, these metrics were omitted from our GPT-

- 5 evaluation. PosterCopilot’s superiority in these specific areas was instead validated through our user study, which confirmed its high-fidelity performance with a dominant win

rate exceeding 87% on both. The results of GPT-5 evaluation is shown in Fig. 14 . We can see that PosterCopilot decisively outperforms other methods across most metrics. PosterCopilot is slightly deficient in Text Legibility compared to Nano-Banana, because PosterCopilot prioritizes faithfully preserving all user-requested text, scaling it as needed for a harmonious layout. Nano-Banana, conversely, often achieves its legibility by simply discarding user elements—a flaw confirmed by its low Element Preservation score in our user study.

##### 6.2. Ablation Study

The RL-VRA and RLAF phases instill professional design principles to address SFT-stage issues, including bounding box drift, element distortion, and aspect ratio errors. Evaluated using IoU, IOPR, and ARD metrics (Tab. 1a), RL-VRA significantly improves layout accuracy over PSFT, with further IOPR/ARD gains in RLAF. The slight IoU drop in RLAF reflects its shifted focus from ground-truth fitting to aesthetic exploration. As detailed in Sec. 3.3, the RL-

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Original: 902 X 1280

Reframe: 1280 X 902

Original: 1200 X 628

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Reframe: 628 X 1046

Original: 1280 X 904

Reframe: 904 X 1280

- Figure 11. PosterCopilot intelligently reframes posters to new canvas sizes while maintaining layout harmony. All figures are scaled to a uniform height for presentation in this paper.

[Figure 133]

- Figure 12. Dataset construction pipeline for our PosterCopilot. We merged numerous scattered layers with OCR-based fine-granularity bounding box rather than simply parsing the original PSD file.

Training Stages

ID

IOU↑ IOPR↓ ARD↓ PSFT RL-VRA RLAF

- I ✓ 0.311 3.38 0.699
- II ✓ ✓ 0.347 1.72 0.061

- III ✓ ✓ ✓ 0.342 0.56 0.045 (a) Ablation on training stages.

Layout Rewards

ID

IOU ↑ IOPR ↓ ARD ↓ rformat rDIOU rAR + rsize

- I ✓ 0.317 3.29 0.707
- II ✓ ✓ 0.339 1.95 0.734
- III ✓ ✓ ✓ 0.347 1.72 0.061 (b) Ablation on reward components of RL-VRA.

Table 1. Comprehensive ablation studies for training stages and reward components. We highlight the best results in red.

VRA reward comprises three components: Spatial Coherence (rDIoU), Element Fidelity (rsize + rAR), and format reward. Our ablation study results in Tab. 1b on the first two rewards reveal their distinct contributions: the Spatial Coherence reward substantially enhances layout accuracy, while the Element Fidelity reward improves preservation of element sizes and proportions. Their combination yields optimal performance.

#### 7. Conclusion

PosterCopilot revolutionizes automated poster design by decoupling creation into layout design and multi-round editing. Our progressive training paradigm forges the design

Metrics Style Consistency Visual Appeal

[Figure 134]

[Figure 135]

Element Preservation

PosterCopilot Win/Lose

Layout Rationality Text Legibility

Design Utility

Instruction Following Average Win Rate

PosterCopilot VS:

[Figure 136]

|96.8%|99.2%|96.0%|98.4%|98.4%|96.8%|99.2%|
|---|---|---|---|---|---|---|

Qwen-2.5-vl-72B

[Figure 137]

|92.8%|69.6%|66.4%|74.4%|76.8%|70.4%|69.6%|
|---|---|---|---|---|---|---|

Creatiposter

[Figure 138]

|92.8%|88.0%|88.0%|85.6%|91.2%|87.2%|100%|
|---|---|---|---|---|---|---|

Microsoft Designer

[Figure 139]

|83.2%|82.4%|83.2%|80.0%|85.6%|83.2%|88.8%|
|---|---|---|---|---|---|---|

LaDeCo

[Figure 140]

|80.8%|83.2%|77.6%|82.4%|84.8%|81.6%|80.8%|
|---|---|---|---|---|---|---|

Gemini 2.5 Pro

[Figure 141]

|93.6%|76.8%|72.8%|84.0%|80.8%|80.0%|88.0%|
|---|---|---|---|---|---|---|

Nano-Banana

Win Rate on all metrics

Figure 13. Results of User-study.

###### Metrics

Style Consistency Visual Appeal

Layout Rationality Text Legibility

Design Utility

| | |
|---|---|
| |3.85|
| | |
| ||6.52|
|---|
<br><br>|6.50|
|---|
<br><br>|5.50|
|---|
<br><br>|5.23|
|---|
<br><br>|5.57|
|---|
<br><br>|6.19|
|---|
<br><br>|
| | |
| | |
| | |
| | |
| | |
| | |

|3.27|
|---|

|4.42|
|---|

|4.58|
|---|

|3.42|
|---|

Qwen-2.5-vl-72B

Creatiposter

|5.38|
|---|

|5.96|
|---|

|6.23|
|---|

|5.23|
|---|

Microsoft Designer

|5.65|
|---|

|5.91|
|---|

|5.91|
|---|

|4.74|
|---|

LaDeCo

|4.96|
|---|

|5.27|
|---|

|5.38|
|---|

|4.00|
|---|

Gemini 2.5 Pro

|4.58|
|---|

|5.88|
|---|

|5.5|4.96|
|---|---|

|6.50|
|---|

|6.08|
|---|

|6.23|
|---|

|6.27|
|---|

|5.46|
|---|

Nano-Banana PosterCopilot

|6.52|
|---|

|5.96|
|---|

|6.68|
|---|

|6.6|6.16|
|---|---|

GPT-5 Score

Figure 14. Results of GPT-5 evaluation.

model with geometric precision and human-like aesthetics, while a generative agent enables multi-round, layer-wise editing mirroring professional workflows. Limitations include the lack of a poster-specific aesthetic reward model and the use of standard blend modes, pointing to future work.

#### References

- [1] Jonathan Baldwin and Lucienne Roberts. Visual communication: from theory to practice. Ava Publishing, 2006. 2
- [2] Josef Bigun, Goesta H. Granlund, and Johan Wiklund. Multidimensional orientation estimation with applications to texture analysis and optical flow. IEEE Transactions on pattern analysis and machine intelligence, 13(8):775–790, 2002. 3
- [3] Inha Cha and Richmond Y Wong. Understanding sociotechnical factors configuring ai non-use in ux work practices. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–17, 2025. 2
- [4] Haoyu Chen, Xiaojie Xu, Wenbo Li, Jingjing Ren, Tian Ye, Songhua Liu, Ying-Cong Chen, Lei Zhu, and Xinchao Wang. Posta: A go-to framework for customized artistic poster generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28694–28704, 2025. 3, 1
- [5] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems, 36:9353–9387, 2023. 6
- [6] Liuqing Chen, Qianzhi Jing, Yixin Tsang, and Tingting Zhou. Iris: a multi-constraint graphic layout generation system. Frontiers of Information Technology & Electronic Engineering, 25(7):968–987, 2024. 1
- [7] Yan Chen, Long Li, Teng Xi, Long Zeng, and Jingdong Wang. Perception before reasoning: Two-stage reinforcement learning for visual reasoning in vision-language models. arXiv preprint arXiv:2509.13031, 2025. 1
- [8] Yutao Cheng, Zhao Zhang, Maoke Yang, Nie Hui, Chunyuan Li, Xinglong Wu, and Jie Shao. Graphic design with large multimodal model. arXiv preprint arXiv:2404.14368, 2024. 4
- [9] Yutao Cheng, Zhao Zhang, Maoke Yang, Hui Nie, Chunyuan Li, Xinglong Wu, and Jie Shao. Graphic design with large multimodal model. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2473–2481, 2025. 3, 6, 1
- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6, 5
- [11] Alex O Davies, Roussel Nzoyem, Nirav Ajmeri, et al. Language models do not embed numbers continuously. arXiv preprint arXiv:2510.08009, 2025. 3
- [12] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36, 2024. 6
- [13] Ross Girshick. Fast R-CNN. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 1440–1448, 2015. 5

- [14] Kamal Gupta, Justin Lazarow, Alessandro Achille, Larry S Davis, Vijay Mahadevan, and Abhinav Shrivastava. Layouttransformer: Layout generation and completion with selfattention. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1004–1014, 2021. 1
- [15] Chris Harris, Mike Stephens, et al. A combined corner and edge detector. In Alvey vision conference, pages 10–5244. Manchester, UK, 1988. 3
- [16] Chen Hou, Guoqiang Wei, and Zhibo Chen. High-fidelity diffusion-based image editing. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2184–2192, 2024. 2
- [17] Runhui Huang, Kaixin Cai, Jianhua Han, Xiaodan Liang, Renjing Pei, Guansong Lu, Songcen Xu, Wei Zhang, and Hang Xu. Layerdiff: Exploring text-guided multi-layered composable image synthesis via layer-collaborative diffusion model. In European Conference on Computer Vision, pages 144–160. Springer, 2024. 2, 1
- [18] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2
- [19] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. Layoutdm: Discrete diffusion model for controllable layout generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10167–10176, 2023. 3, 1
- [20] Mikołaj Janusz, Tomasz Wojnar, Yawei Li, Luca Benini, and Kamil Adamczewski. One shot vs. iterative: Rethinking pruning strategies for model compression. arXiv preprint arXiv:2508.13836, 2025. 2
- [21] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9307–9315,

2024. 3

- [22] Binbin Ji, Siddharth Agrawal, Qiance Tang, and Yvonne Wu. Enhancing spatial reasoning in vision-language models via chain-of-thought prompting and reinforcement learning. arXiv preprint arXiv:2507.13362, 2025. 1
- [23] Peidong Jia, Chenxuan Li, Yuhui Yuan, Zeyu Liu, Yichao Shen, Bohan Chen, Xingru Chen, Yinglin Zheng, Dong Chen, Ji Li, et al. Cole: A hierarchical generation framework for multi-layered and editable graphic design. arXiv preprint arXiv:2311.16974, 2023. 3, 1
- [24] Qing Jiang, Junan Huo, Xingyu Chen, Yuda Xiong, Zhaoyang Zeng, Yihao Chen, Tianhe Ren, Junzhi Yu, and Lei Zhang. Detect anything via next point prediction. arXiv preprint arXiv:2510.12798, 2025. 2
- [25] Pegah Karimi, Jeba Rezwana, Safat Siddiqui, Mary Lou Maher, and Nasrin Dehbozorgi. Creative sketching partner: an analysis of human-ai co-creativity. In Proceedings of the 25th international conference on intelligent user interfaces, pages 221–230, 2020. 2
- [26] Abidullah Khan, Atefeh Shokrizadeh, and Jinghui Cheng. Beyond automation: How designers perceive ai as a creative

- partner in the divergent thinking stages of ui/ux design. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–12, 2025. 2
- [27] Xiang Kong, Lu Jiang, Huiwen Chang, Han Zhang, Yuan Hao, Haifeng Gong, and Irfan Essa. Blt: Bidirectional layout transformer for controllable layout generation. In European Conference on Computer Vision, pages 474–490. Springer,

2022. 1

- [28] Hsin-Ying Lee, Lu Jiang, Irfan Essa, Phuong B Le, Haifeng Gong, Ming-Hsuan Yang, and Weilong Yang. Neural design network: Graphic layout generation with constraints. In European conference on computer vision, pages 491–506. Springer, 2020. 1
- [29] Chen Li, Xiaoling Hu, Shahira Abousamra, Meilong Xu, and Chao Chen. Spatial diffusion for cell layout generation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 481–491. Springer,

2024. 2

- [30] Sha Li. Llms as layout designers: A spatial reasoning perspective. arXiv e-prints, pages arXiv–2509, 2025. 2
- [31] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22511–22521, 2023. 2, 6, 1
- [32] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-Guang Lou, and Dongmei Zhang. Layoutprompter: Awaken the design ability of large language models. Advances in Neural Information Processing Systems, 36:43852–43879, 2023. 3, 1
- [33] Jiawei Lin, Shizhao Sun, Danqing Huang, Ting Liu, Ji Li, and Jiang Bian. From elements to design: A layered approach for automatic graphic design composition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8128–8137, 2025. 6, 5
- [34] Xiao Liu, Tianjie Zhang, Yu Gu, Iat Long Iong, Yifan Xu, Xixuan Song, Shudan Zhang, Hanyu Lai, Xinyi Liu, Hanlin Zhao, et al. Visualagentbench: Towards large multimodal models as visual foundation agents. arXiv preprint arXiv:2408.06327,

2024. 2

- [35] Ishaan Malhi, Praneet Dutta, Ellie Talius, Sally Ma, Brendan Driscoll, Krista Holden, Garima Pruthi, and Arunachalam Narayanaswamy. Preserving product fidelity in large scale image recontextualization with diffusion models. arXiv preprint arXiv:2503.08729, 2025. 2
- [36] Philip B Meggs, Alston W Purvis, Sandra Maxa, and Mark Sanders. Meggs’ history of graphic design. John Wiley & Sons, 2025. 2
- [37] Marie Muehlhaus and J¨urgen Steimle. Interaction design with generative ai: An empirical study of emerging strategies across the four phases of design. arXiv preprint arXiv:2411.02662, 2024. 2
- [38] Thanh Thi Nguyen, Campbell Wilson, and Janis Dalins. Aligning large vision-language models by deep reinforcement learning and direct preference optimization. arXiv preprint arXiv:2509.06759, 2025. 1
- [39] Rodolfo Ocampo Blanco and Oliver Bown. Integrating generative ai into creative workflows: Dealing with consistency,

- scene control, and refinement in a professional image generation case study. In International Conference on Computational Creativity, 2024. 2
- [40] Joel Oksanen. Bridging the integrity gap: Towards ai-assisted design research. In Extended abstracts of the CHI conference on human factors in computing systems, pages 1–5, 2024. 2
- [41] OpenAI. Introducing gpt-5, 2025. 6
- [42] Peter O’Donovan, Aseem Agarwala, and Aaron Hertzmann. Learning layouts for single-pagegraphic designs. IEEE transactions on visualization and computer graphics, 20(8):1200– 1213, 2014. 1
- [43] Sohan Patnaik, Rishabh Jain, Balaji Krishnamurthy, and Mausoom Sarkar. Aesthetiq: Enhancing graphic layout design via aesthetic-aware preference alignment of multi-modal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23701–23711, 2025. 2, 1
- [44] Leigang Qu, Haochuan Li, Wenjie Wang, Xiang Liu, Juncheng Li, Liqiang Nie, and Tat-Seng Chua. Silmm: Selfimproving large multimodal models for compositional text-toimage generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18497–18508, 2025. 2
- [45] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 1
- [46] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 6
- [47] John Schulman, Filip Wolski, Prafulla Dhara, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 4
- [48] Karthick Panner Selvam. Why large language models fail at precision regression, 2025. 3
- [49] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 4
- [50] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024. 2
- [51] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983, 2023. 6
- [52] Zecheng Tang, Chenfei Wu, Juntao Li, and Nan Duan. Layoutnuwa: Revealing the hidden layout expertise of large language models. arXiv preprint arXiv:2309.09506, 2023. 3, 1

- [53] Qwen Team. Qwen2.5: A party of foundation models, 2024. 6
- [54] Qwen Team. Qwen2.5-vl, 2025. 6, 5
- [55] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 2, 1
- [56] Yuqing Wang, Zhijie Lin, Yao Teng, Yuanzhi Zhu, Shuhuai Ren, Jiashi Feng, and Xihui Liu. Bridging continuous and discrete tokens for autoregressive visual generation. arXiv preprint arXiv:2503.16430, 2025. 2
- [57] Zhenyu Wang, Enze Xie, Aoxue Li, Zhongdao Wang, Xihui Liu, and Zhenguo Li. Divide and conquer: Language models can plan and self-correct for compositional text-to-image generation. arXiv preprint arXiv:2401.15688, 2024. 2
- [58] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 6
- [59] Tianhe Wu, Jian Zou, Jie Liang, Lei Zhang, and Kede Ma. Visualquality-r1: Reasoning-induced image quality assessment via reinforcement learning to rank. arXiv preprint arXiv:2505.14460, 2025. 6, 1, 3
- [60] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 1

- [61] Xiao-Kun Wu, Min Chen, Wanyi Li, Rui Wang, Limeng Lu, Jia Liu, Kai Hwang, Yixue Hao, Yanru Pan, Qingguo Meng, et al. Llm fine-tuning: Concepts, opportunities, and challenges. Big Data and Cognitive Computing, 9(4):87,

2025. 2

- [62] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023. 1
- [63] Tao Yang, Yingmin Luo, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. Posterllava: Constructing a unified multi-modal layout generator with llm. arXiv preprint arXiv:2406.02884, 2024. 3, 1
- [64] Xuyong Yang, Tao Mei, Ying-Qing Xu, Yong Rui, and Shipeng Li. Automatic generation of visual-textual presentation layout. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 12(2):1–22,

2016. 1

- [65] Ning Yu, Chia-Chih Chen, Zeyuan Chen, Rui Meng, Gang Wu, Paul Josel, Juan Carlos Niebles, Caiming Xiong, and Ran Xu. Layoutdetr: detection transformer is a good multimodal layout designer. In European Conference on Computer Vision, pages 169–187. Springer, 2024. 1

- [66] Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv preprint arXiv:2503.18013, 2025. 1
- [67] Hui Zhang, Dexiang Hong, Maoke Yang, Yutao Cheng, Zhao Zhang, Jie Shao, Xinglong Wu, Zuxuan Wu, and YuGang Jiang. Creatidesign: A unified multi-conditional diffusion transformer for creative graphic design. arXiv preprint arXiv:2505.19114, 2025. 1
- [68] Zhao Zhang, Yutao Cheng, Dexiang Hong, Maoke Yang, Gonglei Shi, Lei Ma, Hui Zhang, Jie Shao, and Xinglong Wu. Creatiposter: Towards editable and controllable multi-layer graphic design generation. arXiv preprint arXiv:2506.10890,

2025. 3, 6, 1, 5

- [69] Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22490–22499, 2023. 6, 1
- [70] Xinru Zheng, Xiaotian Qiao, Ying Cao, and Rynson WH Lau. Content-aware generative modeling of graphic design layouts. ACM Transactions on Graphics (TOG), 38(4):1–15, 2019. 1
- [71] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-IoU loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12993– 13000, 2020. 4
- [72] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-iou loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI conference on artificial intelligence, pages 12993– 13000, 2020. 6, 4
- [73] Xingxing Zou, Wen Zhang, and Nanxuan Zhao. From fragment to one piece: A review on ai-driven graphic design. Journal of Imaging, 11(9):289, 2025. 6, 3

## PosterCopilot: Toward Layout Reasoning and Controllable Editing for Professional Graphic Design

### Supplementary Material

#### 8. Related Work

##### 8.1. Intelligent Graphic Design System

Single-layer Graphic Design Generation initially relied on rule-based methods and human aesthetic constraints [42, 64, 70], or framed the task as a constrained optimization problem [6, 28]. The paradigm shifted with the advent of text-to-image (T2I) models, driving research into enhancing the compositional capabilities of diffusion models by integrating layout information. Examples include GLIGEN [31], LayerDiff [17], and MS-Diffusion [55], with LayoutDiffusion [69] specifically using layout as a conditioning modality. CreatiDesign [67] integrates user assets but requires pre-defined layouts. However, this single-layer approach inherently limits iterative refinement and editability, often leading to visual inconsistency and distortion in unmodified regions [19], which increases user burden and limits usability.

Multi-layers Graphic Layout Planning has gained attention due to its focus on real-world practicality, operating by first inferring a layout and then assembling multiple layers to offer high flexibility and editability. Early Transformerbased methods, including LayoutTransformer [14], BLT [27], and LayoutDETR [65], reframed generation as a layout prediction task, but their flat, sequential representations lacked the necessary hierarchical structure for complex designs. The subsequent rise of Vision-Language Models (VLMs) led to VLM-assisted approaches like LayoutPrompter [32] and LayoutUWNA [52], which use in-context learning for layout inference. PosterLLAVA [63] guides generation through Vision Supervised Fine-Tuning on layered designs. Other methods focus on asset integration (Graphist [9]), typography (POSTA [4]), or external generative capabilities (CreatiPoster [68], COLE [23]). Crucially, these models primarily mimic static datasets rather than learning from the aesthetic quality of their own outputs. Our strategy moves beyond simple mimicry, internalizing fundamental principles of layout generation and visual aesthetics from direct generative feedback.

##### 8.2.ReinforcementLearningforVisuallyGrounded Layout Generation

Reinforcement Learning (RL) has significantly advanced the alignment of Vision-Language Models (VLMs) with human preferences [38, 66] and enhanced their reasoning capabilities [7, 22]. Various VLM-based visual reward models, such as HPSv2 [60], ImageReward [62], and VisualQuality-R1

[59], are trained on human preference datasets to provide aesthetic feedback. However, the feedback signals from these models are often overly general and holistic, lacking specific assessments of crucial graphic design elements like layout and alignment. While AesthetiQ [43] utilizes DirectPreference-Optimization (DPO) [45] to embed preferences in layout models, its basic feedback mechanism struggles with complex, nuanced preference signals. Our approach addresses these limitations by proposing a multi-stage Reinforcement Learning framework that directly integrates finegrained layout principles and human aesthetic feedback into the model’s learning process.

#### 9. Implementation details for three-stage training process

Training is conducted in three stages: (1) initial PSFT phase, training for 3 epochs on 160K high-quality samples from our PosterCopilot datasets; (2) RL-VRA phase on 20K samples exhibiting complex layout rules; and (3) the final RLAF phase on 1k expert-validated samples.

##### 9.1. Implementation details for PSFT phase

As shown in Eq.5 in the main text, we perturb the bounding box values of each element in the ground truth layout to conduct our PSFT training phase. This process transforms them from single, precise values into a Gaussian-like distribution, using the original value as the mean and a small parameter as the variance. Subsequently, we sample n values from this distribution and then calculate the PSFT loss.

Prior to the PSFT training, we conducted a grid analysis on the hyperparameters: (1) the standard deviation σ of the applied perturbation, and (2) the PSFT sampling number n. We evaluated a wide range of σ and n combinations. The quality of layouts generated by design models trained with these different parameter combinations was measured using IoU, ARD, and IOPR. The resulting impact of these parameters on the PSFT stage is illustrated in Fig. 15. It is evident that when the standard deviation σ of the added perturbation is below 3.0, the model’s overall performance in the PSFT phase improves as the perturbation magnitude increases. This is because adding perturbation effectively mitigates the numerical gap caused by text tokens performing regression tasks. Learning a distribution (rather than a single point) allows the design model to better grasp key layout patterns. When the standard deviation σ exceeds 3.0, the model’s performance shows a slight degradation as σ increases (for a fixed sampling number n). This is

[Figure 142]

[Figure 143]

[Figure 144]

Figure 15. Visualization of the hyperparameter analysis for the PSFT phase.

because the spread of the perturbed distribution becomes excessively large, which interferes with the model’s learning of the ground truth layout.Conversely, increasing the sampling number n consistently improves the design model’s performance, regardless of the σ value. However, this improvement becomes marginal once n exceeds 5, and a larger n also incurs a significant computational burden. Ultimately, to strike a balance between model performance and computational efficiency, we adopt σ = 2.5 and n = 5 as our final parameters.

##### 9.2. Implementation details for RL-VRA

###### 9.2.1. Implementation details for reward design

We use verl [50] for our reinforcement learning training phase. In RL-VRA phase we design a verifiable geometric reward as shown in Eq.10 in the main text:

+rformat (12)

###### r(G) = rDIoU

+λsizersize + λARrAR

Element Fidelity

Spatial Coherence

where λsize,λAR > 0. In practice, we empirically set the weights λsize = 0.6 and λAR = 0.4. In future work, a more fine-grained method for automatically determining individual reward weights based on the training stage may further improve the training effectiveness of RL-VRA. The specific calculations of several rewards in RL-VRA during the actual training process are as follows:

For rDIoU, the raw DIoU metric is calculated for each element in each data sample, with a native value range of [-1.0, 1.0]. These values are then averaged to get Mean − DIoU. This average is transformed using the formula:

rDIoU = (Mean − DIoU + 1)/2) × 10. (13)

This mapping scales the original [-1.0, 1.0] range directly to the [0, 10] reward range, where a value of -1.0 (worst) corresponds to 0 points and +1.0 (perfect) corresponds to 10 points.

For rAR, it’s calculated from a normalized penalty. The function first computes the absolute log-difference between the predicted and ground truth aspect ratios for each layer, capping this penalty value at 1.0 (defined as cap in the following illustration). It then calculates the average negative

penalty as shown in Eq.7 in the main text to get rARoriginal, which lies in the range [-1.0, 0]. This penalty is converted

into the score using the formula:

rAR = ((rARoriginal + cap)/cap) × 10 (14)

This inverts the rARoriginal, mapping the worst-case penalty (-1.0) to a score of 0.0 and the no-penalty case (0.0) to a full score of 10.0.

The computation of the size accuracy reward (rsize) parallels the methodology used for the aspect ratio reward. First, the size inaccuracy for each layer in each data sample is quantified as shown in Eq.8 in the main text. This resulting penalty is capped at a maximum value of 1.0 (denoted as cap). The average of these individual penalties is then calculated across all layers, and its negative is taken, yielding rsize smooth. This ensures rsize smooth is bounded within the range [−1.0,0], where -1.0 represents the maximum penalty. Finally, rsize smooth is linearly transformed from its penalty-based range to the final 0–10 reward scale. This transformation is expressed in the following equation:

(rsize smooth + cap)

cap × 10 (15)

rsize =

The format reward rformat is a binary score designed to ensure the prediction layout Gˆ is a valid JSON. It receives a full score of 10.0 if Gˆ can be successfully parsed as a JSON object. If the string is malformed and results in a JSONDecodeError or other parsing failure, the function immediately returns 0.0, effectively penalizing any syntactically incorrect outputs.

In summary, we have obtained a reward function that is dense, provides multi-dimensional geometric feedback, and has a maximum score of 30. This balanced reward structure is designed to provide effective visual feedback while simultaneously mitigating reward hacking. Furthermore, it prevents any single component from dominating the optimization process, which would otherwise lead to the neglect of other crucial objectives.

9.2.2. GRPO Hyperparameter Settings for RL-VRA As shown in Tab. 2.

Hyperparameter Value Learning Rate 1 × 10−6 KL Loss Coefficient 0.01 Clip Ratio 0.2 Actor Entropy Coefficient 0.01 Training Batch Size 96 GRPO Group Size 8 Total Epochs 1 Learning Rate Optimizer Adam

Table 2. GRPO hyperparameter settings for RL-VRA

##### 9.3. Implementation details for RLAF

###### 9.3.1. Implementation details for reward design rRLAF(G) is defined as Eq.11 in the main text:

rRLAF(G) = rformat + λaesraes(G) (16) where λaes > 0. We adopt the same calculation method for rformat as in the RL-VRA stage. We employ VisualQualityR1 [59], an evaluation model meticulously trained to align with human aesthetic preferences, as the judge model for the RLAF stage. Similarly, we modulate the contribution of raes(G) via the hyperparameter λaes to ensure a balanced configuration of reward scores. In our experiments, we set λaes = 2.

###### 9.3.2. GRPO hyperparameter settings for RLAF

Hyperparameter Value Learning Rate 5 × 10−7 KL Loss Coefficient 0.01 Clip Ratio 0.4 Actor Entropy Coefficient 0.01 Training Batch Size 64 GRPO Group Size 4 Total Epochs 1 Learning Rate Optimizer Adam

Table 3. GRPO hyperparameter settings for RLAF

#### 10. More Details For Evaluation Metrics

##### 10.1. Aesthetic Evaluation Metrics

It has become a prevailing consensus among researchers in the field of graphic wdesign that relying on traditional AIGC metrics to gauge design quality is fundamentally unreasonable. While metrics like Fr´echet Inception Distance (FID) and Structural Similarity Index (SSIM) are highly effective in natural image synthesis tasks, they prove inadequate when assessing the quality of graphic design [19, 21, 68]. This inadequacy stems from a fundamental divergence in evaluation dimensions: FID and other metrics focus primarily on

pixel-level fidelity and the statistical similarity of feature distributions. However, the core value of poster design lies not in the pixel-level replication of training data, but in layout topology, visual hierarchy, typographic aesthetics, and the semantic interaction among multi-modal elements [73].

Specifically, a vast majority of existing literature in the domain has critically argued that traditional AIGC metrics suffer from severe limitations. First, they lack the capability to perceive design rules. A generated poster might exhibit texture and color distributions highly consistent with the training set (yielding a favorable FID score), yet contain severe design accidents such as text occluding key image subjects, misalignment of elements, or imbalanced white space. While these errors are intolerable to human designers, they are often overlooked by evaluation systems based on convolutional features. Second, the calculation of these metrics is heavily influenced by the generative model’s fit to the training data distribution. A low FID score merely indicates that the generated images are statistically similar to the training set, without measuring whether they are good in terms of visual appeal, the core of the graphic design domain. If the training data itself contains mediocre designs, traditional metrics may even reward outputs that mimic this mediocrity while penalizing high-quality designs that are innovative but deviate from the statistical mean. Consequently, directly applying traditional AIGC metrics fails to objectively evaluate the aesthetic value and layout quality of poster designs. There is an urgent need in this field to establish a novel evaluation system based on geometric constraints and human aesthetic perception.

Building upon the foundation of numerous distinguished prior works, we further consulted a diverse panel of experts—spanning from professional graphic designers to AI researchers. Through this process, we finalized a set of human evaluation metrics that are most suitable for assessing poster design. While the metrics are enumerated in Sec. 5.2 in the main text, owing to the limited space, their detailed descriptions are presented in Tab. 4. These metrics cover all critical aspects of poster quality assessment, enabling a fair and comprehensive measurement of the final design quality.

##### 10.2. Layout evaluation metrics

In the training phase, our method takes multiple layers decomposed from a complete poster as input, generates a layout in JSON format, and subsequently renders this layout into a poster image using rendering code. Following mainstream practices in prior literature, we employed three metrics—IoU, IOPR, and ARD—in the ablation study of the main text to directly measure the discrepancy between the predicted JSON layout and the ground truth layout. Here, we first provide the detailed calculation methods for these three metrics.

###### For IoU metric, we clarify that all references to this

Criterion Description Layout Rationality

Layout Rationality evaluates the global compositional coherence, rational element placement, clarity of visual hierarchy, and minimal occlusion of critical content.

Text Legibility assesses the readability of the text design (determined by font choice, size, line spacing, and color) and the faithfulness of its rendering (sharp edges, no distortion, artifacts, or garbled characters).

Text Legibility

Asset Preservation evaluates if all user-provided visual assets are fully retained and unaltered in the final result.

Asset Preservation

Style Consistency assesses the coherence of stylistic treatment across all elements and the appropriateness of the overall visual style to the stated theme.

Style Consistency

Instruction Following

Instruction following evaluates the fidelity to the textual specification, including the requested theme, style, layout, color scheme, and any required elements.

Overall Visual Appeal assesses the immediate aesthetic appeal and the ability to attract attention at first glance.

Visual Appeal

Design Utility assesses the suitability of the poster to be adopted as an initial design when facing the same practical brief (e.g., promoting the same product or theme.

Design Utility

Table 4. Aesthetic evaluation metrics.

metric throughout both the main text and the supplementary material denote the average IoU. Specifically, we compute the IoU between each element in the layout generated by the design model and its corresponding element in the ground truth layout. The final IoU score for a poster sample is then derived by averaging the IoU values of all its constituent elements. The calculation of IoU is formally defined as follows:

Area(Bpred ∩ Bgt) Area(Bpred ∪ Bgt)

IoU(Bpred,Bgt) =

(17)

where N denotes the number of elements in the poster, and Bpred(i) and Bgt(i) represent the predicted and ground truth bounding boxes of the i-th element, respectively.

For the IOPR [8] metric, we evaluate the correctness of the predicted layer order, which is essential for maintaining visual hierarchy. IOPR quantifies the ratio of overlapping element pairs that violate the ground truth depth sequence. For a single sample with n layers, it is calculated as:

n−1 i=1

n j=i+1 (Oj < Oi ∧ overlap(i,j))

IOPR =

,

n−1 i=0

n j=i+1

(18) where n is the number of layers in the hierarchical structure.

is an indicator function that returns 1 if the argument condition is true and 0 otherwise. O denotes the output order or predicted order of the layers as determined by the model. Oi and Oj correspond to the predicted order positions of the ith and jth layers, respectively. overlap(i,j) is a predicate function that determines whether the ith and jth layers overlap.

For ARD metric, it’s utilized to measure the aspect ratio distortion of the predicted bounding boxes relative to the ground truth. It is derived from the v term of the Complete IoU (CIoU) [72] metric, which is widely adopted in the industry:

4 π2

v =

wgt hgt − arctan

w h

arctan

2

(19)

where wgt and hgt denote the ground truth bounding box values, w,h denote the predicted bounding box values, and arctanh is one of the three tangent functions. In practice, we omitted the leading normalization term π42 to make the metric differences more pronounced:

2

wgt hgt − arctan

w h

(20)

ARD = arctan

Although these quantitative layout evaluation metrics are less suited for assessing overall poster image quality compared to the aesthetic metrics introduced earlier, and are not directly applicable to single-layer generation or text-toimage models, they provide a more direct quantification of the discrepancy between generated layouts and the ground truth. Consequently, we employ these metrics specifically in the ablation study of the main text, rather than calculating them for all baselines.

#### 11. Supplementary Ablation Study

Due to the limited space, the ablation study in the main text details only the primary training procedure and the reward component ablation results, demonstrating the necessity of each component and training phase. Here, we present additional ablation studies to directly validate the superiority of

Method IOU↑ IOPR↓ ARD↓ SFT 0.285 4.12 0.851 PSFT (Ours) 0.311 3.38 0.699

Table 5. Quantitative comparison between standard SFT and our proposed PSFT. Best results are highlighted in red.

our PSFT phase over conventional SFT paradigms. Furthermore, we conduct a human evaluation to verify that RLAF guides the model to generate layouts more aligned with human aesthetics. This serves as an intuitive complement to the quantitative metrics presented in the main text. Except for the specific modules being ablated, all experimental settings for the ablation studies in both the main text and the supplementary material are identical to the training procedure described in Sec. 9.

##### 11.1. Ablation study for PSFT phase

We evaluated the design model trained solely with PSFT against the one trained with standard SFT. The latter was trained exclusively on ground truth layouts without the introduction of perturbations or other augmentation measures. The results is as shown in Tab. 5. We can see that the design model trained via PSFT significantly outperforms the standard SFT baseline across IoU, IOPR, and ARD metrics. This demonstrates that the PSFT strategy, by incorporating perturbations, effectively mitigates the numerical-semantic gap caused by treating numerical coordinates as text tokens for regression.

##### 11.2. Ablation study for RLAF phase

Fig. 16 visually demonstrates the critical role of RLAF. Given that poster design is inherently driven by human aesthetics, training a design model solely to replicate ground truth layouts is insufficient. The model often generates layouts that deviate significantly from the ground truth yet remain aesthetically pleasing. In fact, layouts exhibiting greater divergence from the ground truth can sometimes yield superior aesthetic quality. We conducted an human evaluation on models trained via three progressive stages: only PSFT, PSFT + RL-VRA, and PSFT+RL-VRA+RLAF (PosterCopilot). We collected 10 inference poster samples, each of which was assessed by a panel of 15 ranging from professional graphic designers to individuals with diverse interdisciplinary backgrounds. The assessment was strictly based on the human evaluation metrics defined in the main text. Fig. 17 presents the evaluation results across various metrics. It is evident that the RL-VRA stage significantly enhances the layout quality and consistency of the generated designs. Building upon the previous stages, RLAF further substantially improves the visual appeal. Regarding the instruction-following capability, since the design model

[Figure 145]

[Figure 146]

[Figure 147]

- Figure 16. Poster samples generated by the design model via multiple inference runs. The IoU scores against the ground truth layout are 0.87, 0.43, and 0.21, respectively. Notably, despite the varying degrees of deviation from the ground truth, all three posters align well with human aesthetics.

has already achieved a satisfactory level via training on the high-quality large-scale PosterCopilot dataset, the improvement in this metric is relatively marginal compared to other key indicators.

Layout Rationality Text Legibility

Metrics Style Consistency Visual Appeal

Win Rate on all metrics

PSFT+RL-VRA

PSFT

PosterCopilot VS:

61.2% 59.4% 63.0% 72.4% 64.9% 69.5% 55.3%

83.7% 84.2% 64.1% 84.9% 71.3% 81.3% 60.9%

[Figure 148]

[Figure 149]

Design Utility

Element Preservation

Instruction Following Average Win Rate

[Figure 150]

PosterCopilot Win/Lose

[Figure 151]

- Figure 17. Human evaluation comparison of design quality metrics across different stages of our training paradigm. PosterCopilot is trained via complete three stages.

#### 12. More Details About Evaluation Procedure

In the field of poster design, recent, open-source, and highperforming baselines capable of handling user-supplied assets are notably scarce. To ensure methodological diversity and comparison against state-of-the-art (SOTA) solutions from both academia and industry, we selected the following baselines: (1) commercial platforms (Microsoft Designer, Nano-Banana); (2) academic SOTAs (LaDeCo [33], CreatiPoster [68]); and (3) reasoning models (Gemini 2.5 Pro [10], Qwen-VL-2.5-72B-Instruct [54]). As demonstrated in the main text, our comparative analysis conditions all models on identical user assets input and design prompts to generate posters. Since the baselines encompass both text-to-image (T2I) models and non-end-to-end layout generation frameworks (similar to PosterCopilot), their inference pipelines exhibit slight variations. In this section, we provide a detailed elaboration of these specific testing protocols.

##### 12.1. Evaluation procedure for T2I models

Among the selected baselines, Microsoft Designer and NanoBanana (formally known as Gemini 2.5 Flash Image) belongs to the T2I category. Notably, since its debut, Nano-

Banana has garnered widespread attention within the graphic design community, distinguished by its unparalleled capabilities in multi-asset conditioned generation and multi-turn iterative editing. The evaluation procedure for T2I models is relatively straightforward. We condition the models on all provided user assets, specify the target canvas dimensions, and input the design prompt to generate the corresponding poster samples for comparison.

##### 12.2. Evaluation procedure for layout generation models

The remaining methods—CreatiPoster [68], LaDeCo [33], Qwen-VL-2.5-72B-Instruct [54], Gemini 2.5 Pro [10], and our own PosterCopilot—fall under the category of Layout Generation models. For these models, consistent with the T2I evaluation, we provide user assets and design prompts. However, we explicitly instruct the models to output the layout in JSON format. Upon obtaining the generated JSON files, we employ a unified high-precision lossless rendering script to convert the text-based layouts into final poster images for each test sample.

It’s worth noting that CreatiPoster requires precise predefined layouts for foreground elements. To accommodate this, we provided the ground truth foreground layouts during its evaluation. Although this setup places our method at a comparative disadvantage, PosterCopilot still achieved a significant lead across all metrics in both GPT-5 evaluations and multi-dimensional human assessments. This further demonstrates PosterCopilot’s robust layout reasoning capabilities while requiring minimal manual input.

#### 13. More Qualitative Comparisons

We provide in Fig. 18 some examples of the setting where various methods assemble posters based on complete assets. Fig. 19 presents additional examples of precise single-layer editing.

#### 14. More details about PosterCopilot datasets

The main text provided a key description of the PosterCopilot dataset construction pipeline. Here, we further offer more details regarding the dataset construction process and the dataset composition.

Our datasets construction pipeline begins with the ingestion of approximately 160,000 professionally designed PSD source files collected from online stock platforms. In the initial phase, OCR Document Parsing, each PSD is exhaustively analyzed to extract all valid layers as independent PNG files. Concurrently, a JSON annotation is generated for each poster, capturing low-level metadata such as bounding boxes, stacking order, and layer type, which provides the foundation for structured supervision.

To mitigate the fragmentation problem, the pipeline proceeds to the Parse stage, where the initial raw layers are prepared for semantic grouping. This is followed by the core Layers Merger phase. Here, the semantic cues provided by the initial OCR-based document parsing are leveraged as a data-cleaning mechanism. The merger process intelligently groups and combines excessively fine layers and concurrently discards visually insignificant ones. This crucial refinement step effectively aligns the fragmented raw layers with human visual perception, resulting in a refined annotation space focused on genuine visual elements. We present key statistics of the PosterCopilot dataset in Fig. 20. To facilitate understanding, we also provide an example of a parsed JSON file for a representative poster instance:

Example of parsed JSON file

{

"psd_file": "c:/desktop/dataset-images/ freepik/freepik/Medical Poster/40858 9341-world-cancer-day-awarenesstemplate/11575324.psd",

"ocr_file": "c:/desktop/user-workspace/ anonymous/psd-parsed-with-ocr/ Medical Poster-408589341-11575324/ ocr/11575324_ocr.json",

"canvas_size": { "width": 1748, "height": 2480

}, "layers": [

{

"src": "World cancer day", "category": "type",

- "x": 144,
- "y": 537, "w": 1468, "h": 368, "order": 0, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": [

{

"text": "WORLD CANCER DAY", "text_type": "PARAGRAPH", "font_size_px": 50.31, "font_family": "’Jost-ExtraBold

’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "center", "leading": 0.99, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0, "warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal",

[Figure 152]

[Figure 153]

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

[Figure 165]

[Figure 166]

[Figure 167]

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

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

###### PosterCopilot Microsoft Designer LaDeCo Gemini 2.5 Pro CreatiPoster Nano-Banana Qwen-VL-2.5-72B-Instruct

- Figure 18. Visual comparison of poster composition results across all methods. Each column corresponds to a specific method, demonstrating its generation performance based on various user assets and prompts.

"tracking": 0.0, "transform": [

4.166666666666667, 0.0, 0.0, 4.166666666666667,

- -33255.49755600113,
- -32887.51407877605

] }

], "group": [

[

"Text", "World cancer day"

]

], "merged_layers_names": [

"World cancer day"

], "merged_layers_num": 1, "merged_layers_indices": [

0 ],

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

- (a) The first set of qualitative comparisons on single-layer editing between PosterCopilot and Nano-Banana.

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

- (b) The second set of qualitative comparisons on single-layer editing between PosterCopilot and Nano-Banana.

- Figure 19. Comparison of single-layer editing performance between PosterCopilot and Nano-Banana. Among all the baselines, only Nano-Banana and our PosterCopilot support the precise editing of arbitrary layers within a poster. Others either lack editing capabilities entirely or are limited to manual repositioning via dragging. Both PosterCopilot and Nano-Banana are fed with identical user assets and prompts for poster generation and multi-round edit. In each comparison, the top row shows the generation and multi-turn editing results of PosterCopilot, while the bottom row displays those of Nano-Banana. In the cases presented, the objective is to exclusively modify the background layer or the woman’s appearance while leaving the rest of the poster intact. As observed, PosterCopilot faithfully preserves non-target regions throughout multi-turn editing sessions while precisely modifying the target layer. In contrast, although Nano-Banana produces impressive results initially, severe distortion occurs in other parts of the poster after just one or two refinement iterations, and unintended attributes of the subject are also altered.

"is_single_layer": true, "files": {

"layer": "c:/desktop/userworkspace/anonymous/psd-parsed

[Figure 211]

[Figure 212]

Figure 20. Key statistics of the PosterCopilot dataset.

-with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 11_merged.png"

}, "ocr_info": { "bbox": [ 136, 534, 1613, 907

], "category": "Title", "text": "# WORLD CANCER DAY"

}

}, {

"src": "entry free", "category": "type",

- "x": 74,
- "y": 119, "w": 156, "h": 117, "order": 1, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": [

###### {

"text": "entry free", "text_type": "PARAGRAPH", "font_size_px": 11.4, "font_family": "’Montserrat-

SemiBold’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "start", "leading": 1.2, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0, "warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal", "tracking": 0.0, "transform": [

4.166666666666667, 0.0,

0.0, 4.166666666666667,

- -33253.49994542471,
- -32886.160441080734

] }

], "group": [

[

"Text", "entry free"

]

], "merged_layers_names": [

"entry free"

], "merged_layers_num": 1, "merged_layers_indices": [

1

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 10_merged.png"

}, "ocr_info": { "bbox": [ 69, 112, 234, 241

], "category": "Text", "text": "entry\nfree"

}

}, {

"src": "4/02", "category": "type",

- "x": 697,
- "y": 122, "w": 319, "h": 128, "order": 3, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": [

###### {

"text": "4/02", "text_type": "PARAGRAPH", "font_size_px": 32.59, "font_family": "’Montserrat-

ExtraBold’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "start", "leading": 1.2, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0,

"warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal", "tracking": 0.0, "transform": [

4.166666666666667, 0.0, 0.0, 4.166666666666667,

- -33255.50039401008,
- -32886.97428385417

] }

], "group": [

[

"Text", "4/02"

]

], "merged_layers_names": [

"4/02"

], "merged_layers_num": 1, "merged_layers_indices": [

3

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 8_merged.png"

}, "ocr_info": { "bbox": [ 693, 118, 1021, 251

], "category": "Text", "text": "4/02"

}

}, {

"src": "@cancer_day", "category": "type",

- "x": 364,
- "y": 2119, "w": 295, "h": 52, "order": 5, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": [

###### {

"text": "@cancer_day", "text_type": "PARAGRAPH", "font_size_px": 12.29, "font_family": "’Jost-Medium’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "start", "leading": 1.2, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0, "warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal", "tracking": 0.0, "transform": [

4.166666666666667, 0.0, 0.0, 4.166666666666667,

- -33254.49951986482,
- -32887.92683919271

] }

], "group": [

[

"Text", "@cancer_day"

]

], "merged_layers_names": [

"@cancer_day"

], "merged_layers_num": 1, "merged_layers_indices": [

5

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 6_merged.png"

}, "ocr_info": { "bbox": [ 361, 2116, 658, 2172

], "category": "Text", "text": "@cancer_day"

}

}, {

"src": "Healthy Life Avenue, 8842

Melrose st., LA,California", "category": "type",

- "x": 71,
- "y": 2248, "w": 982, "h": 104, "order": 6, "blend_mode": "BlendMode.NORMAL",

"opacity": 255, "text_info": [

###### {

"text": "Healthy Life Avenue, 88 42 Melrose st., LA, California",

"text_type": "PARAGRAPH", "font_size_px": 12.29, "font_family": "’Jost-Medium’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "start", "leading": 1.2, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0, "warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal", "tracking": 0.0, "transform": [

4.166666666666667, 0.0, 0.0, 4.166666666666667,

- -33255.49974608525,
- -32887.958521327215

] }

], "group": [

[

"Text", "Healthy Life Avenue, 8842

Melrose st., LA,California" ]

], "merged_layers_names": [

"Healthy Life Avenue, 8842 Melrose st., LA,California"

], "merged_layers_num": 1, "merged_layers_indices": [

6

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 5_merged.png"

}, "ocr_info": { "bbox": [ 69, 2245, 1054, 2357

], "category": "Text", "text": "Healthy Life Avenue, 8842

Melrose st., LA,-\nCalifornia "

}

}, {

"src": "www.cancerday.com", "category": "type",

- "x": 1222,
- "y": 2118, "w": 459, "h": 52, "order": 7, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": [

###### {

"text": "www.cancerday.com", "text_type": "PARAGRAPH", "font_size_px": 12.29, "font_family": "’Jost-Medium’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "right", "leading": 1.2, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0, "warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal", "tracking": 0.0, "transform": [

4.166666666666667, 0.0, 0.0, 4.166666666666667,

- -33254.49970463595,
- -32889.040771484375

] }

], "group": [

[

"Text", "www.cancerday.com"

]

], "merged_layers_names": [

"www.cancerday.com"

], "merged_layers_num": 1, "merged_layers_indices": [

7

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 4_merged.png"

},

"ocr_info": { "bbox": [ 1219, 2116, 1680, 2170

], "category": "Text", "text": "www.cancerday.com"

}

}, {

"src": "cancer knows no gender or

age. get a check-up regulary.", "category": "type",

- "x": 366,
- "y": 362, "w": 1004, "h": 97, "order": 8, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": [

###### {

"text": "CANCER KNOWS NO GENDER OR AGE. GET A CHECK-UP REGULARY.",

"text_type": "PARAGRAPH", "font_size_px": 12.0, "font_family": "’Montserrat-

BoldItalic’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "center", "leading": 1.2, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0, "warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal", "tracking": 0.0, "transform": [

4.166666666666667, 0.0, 0.0, 4.166666666666667,

- -33255.501571969085,
- -32886.8654327771 ]

###### }

], "group": [

[

"Text", "cancer knows no gender or age.

get a check-up regulary." ]

], "merged_layers_names": [

"cancer knows no gender or age. get a check-up regulary."

], "merged_layers_num": 1, "merged_layers_indices": [

8

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 3_merged.png"

}, "ocr_info": { "bbox": [ 361, 355, 1374, 463

], "category": "Text", "text": "CANCER KNOWS NO GENDER OR

AGE.\nGET A CHECK-UP REGULARY

." }

}, {

"src": "Anual Scientific Cancer congress", "category": "type",

- "x": 373,
- "y": 993, "w": 920, "h": 116, "order": 9, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": [

###### {

"text": "ANNUAL SCIENTIFIC

CANCER CONGRESS", "text_type": "PARAGRAPH", "font_size_px": 14.24, "font_family": "’Montserrat-

ExtraBold’", "color_css": "rgba(96, 0, 146, 1

.0)", "text_align": "center", "leading": 1.2, "warp": {

"warpStyle": "b’warpNone’", "warpRotate": "b’Hrzn’", "warpValue": 0.0, "warpPerspective": 0.0, "warpPerspectiveOther": 0.0

}, "font-weight": "normal", "font-style": "normal", "tracking": 0.0, "transform": [

4.166666666666667, 0.0, 0.0, 4.166666666666667,

-33255.50023252936,

-32886.372521938516 ]

###### }

], "group": [

[

"Text", "Anual Scientific Cancer

congress" ]

], "merged_layers_names": [

"Anual Scientific Cancer congress"

], "merged_layers_num": 1, "merged_layers_indices": [

9

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 2_merged.png"

}, "ocr_info": { "bbox": [ 370, 986, 1296, 1113

], "category": "Text", "text": "## ANNUAL SCIENTIFIC

CANCER CONGRESS" }

}, {

"src": "Vector Smart Object", "category": "smartobject",

- "x": 129,
- "y": 1161, "w": 1355, "h": 889, "order": 10, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": {}, "group": [

[

"Design", "Vector Smart Object"

]

], "merged_layers_names": [

"Vector Smart Object"

], "merged_layers_num": 1, "merged_layers_indices": [

11

], "is_single_layer": true, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 1_merged.png"

}, "ocr_info": { "bbox": [ 127, 1156, 1489, 2051

], "category": "Picture"

}

}, {

"src": "Background Layer", "category": "background",

- "x": 0,
- "y": 0, "w": 1748, "h": 2480, "order": 11, "blend_mode": "BlendMode.NORMAL", "opacity": 255, "text_info": {}, "group": [

[

"Social Media", "Vector Smart Object"

], [

"Design", "Vector Smart Object"

], [

"Background", "Background"

]

], "merged_layers_names": [ "Vector Smart Object", "Vector Smart Object", "Background"

], "merged_layers_num": 3, "merged_layers_indices": [

10,

- 12,
- 13

], "is_single_layer": false, "files": {

"layer": "c:/desktop/user-

workspace/anonymous/psd-parsed -with-ocr/Medical Poster-40858 9341-11575324/merged/11575324_ 0_merged.png"

} }

], "statistics": {

"original_layers": 17, "valid_layers": 14, "merged_groups": 12,

"excluded_layers": 0, "out_of_bounds_layers": 3

} }

