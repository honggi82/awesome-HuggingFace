# arXiv:2512.01816v1[cs.CV]1Dec2025

[Figure 1]

#### Envision: Benchmarking Unified Understanding & Generation for Causal World Process Insights

###### Juanxi Tian1, Siyuan Li1, Conghui He1, Lijun Wu1, Cheng Tan1 1Shanghai Artificial Intelligence Laboratory

Current multimodal models aim to transcend the limitations of single-modality representations by unifying understanding and generation, often using text-to-image (T2I) tasks to calibrate semantic consistency. However, their reliance on static, single-image generation in training and evaluation leads to overfitting to static pattern matching and semantic fusion, while fundamentally hindering their ability to model dynamic processes that unfold over time. To address these constraints, we propose Envision—a causal event progression benchmark for chained text-to-multi-image generation. Grounded in world knowledge and structured by spatiotemporal causality, it reorganizes existing evaluation dimensions and includes 1,000 four-stage prompts spanning six scientific and humanities domains. To transition evaluation from single images to sequential frames and assess whether models truly internalize world knowledge while adhering to causal-temporal constraints, we introduce Envision-Score—a holistic metric integrating multi-dimensional consistency, physicality, and aesthetics. Comprehensive evaluation of 15 models (10 specialized T2I models, 5 unified models) uncovers: specialized T2I models demonstrate proficiency in aesthetic rendering yet lack intrinsic world knowledge. Unified multimodal models bridge this gap, consistently outperforming specialized counterparts in causal narrative coherence. However, even these unified architectures remain subordinate to closed-source models and struggle to overcome the core challenge of spatiotemporal consistency. This demonstrates that a focus on causally-isolated single images impedes multi-frame reasoning and generation, promoting static pattern matching over dynamic world modeling—ultimately limiting world knowledge internalization, generation.

Date: December 2, 2025 Correspondence: Cheng Tan, tancheng@pjlab.org.cn

[Figure 2]

[Figure 3]

[Figure 4]

Code Dataset Website

#### 1 Introduction

Current text-to-image (T2I) models are capable of rendering images with remarkable realism and diversity [30, 31, 29]. Efforts are also being made to further push the limits of semantic representation by unifying visual understanding and generation capabilities [42, 12, 43]. Additionally, tasks of T2I benchmarks are used to calibrate the semantic consistency of world knowledge and the physical plausibility of visual representations across models. However, this reliance on static pattern matching of single-frame images may lead to overfitting limitations in the models, where semantics become fused. Inherently, static images lack temporal directionality, creating causal ambiguity where the genesis and consequence of a visual state are indistinguishable. Models trained solely on such singleimage data may excel at constructing static scenes, yet they lack the capacity to govern the dynamic processes of real-world events.[5, 46]. Therefore, the ability to generate photorealistic frames does not necessarily imply an understanding of the generative rules of reality. This paradox compels us to ask: Does outstanding performance in generating static, isolated images fully demonstrate that the model has truly internalized world knowledge?

[Figure 5]

World Simulation

z

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Temporal Weaving

、

[Figure 12]

[Figure 13]

[Figure 14]

Spatial Deconstruction

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Semantic Anchoring

A lion.

A lion.

A zebra.

A zebra. A zebra.

Figure 1: Envision Vision: (1) Semantic Anchoring, (2) Spatial Deconstruction, (3) Temporal Weaving, and (4) World Simulation. Progressive stages of cognitive development in generative models.

Table 1: Requirements for Different Text-to-Visual Generation Modalities.

Modality Core Requirements Additional Requirements T2I Image Aesthetics

—

Object Texts Position Texts

T2I to T2MI All T2I Requirements + Chain of Events

+ Consistent Attributes

+ Event Causality

T2MI to T2V All T2MI Requirements + Chain of Actions

+ Object & Attribute Consistency

+ Temporal Continuity

Note: T2I: Text-to-Image; T2V: Text-to-Video; T2MI: Text-to-Multi-Image. Core requirements are fundamental capabilities; ‘+‘ denotes extensions beyond preceding modalities.

Building on this, our idealized vision posits that a perfected generative model must possess the capacity to truly internalize and govern world knowledge, enabling it to construct an internal world model through the dynamic interleaving of understanding and generation (as illustrated in Figure 1). This vision, termed Envision-Vision, is concretely embodied in four stages: (1) Semantic Anchoring (mapping visual features to conceptual entities), (2) Spatial Deconstruction (deconstructing 2D spatial relationships using implicit 3D cognition), (3) Temporal Weaving (constructing cross-temporal causal chains and state transitions), and (4) World Simulation (building an internal world model to predict spatiotemporal states and simulate process dynamics).

Under the Envision-Vision, various visual modalities are intricately interconnected, ultimately advancing toward a comprehensive visual intelligence capable of representing the complete world. As an optimal intermediate representation, multi-image sequences bridge the spatiotemporal boundary between images and videos, employing continuous or discrete image sequences to depict event processes under arbitrary spatiotemporal constraints, as detailed in Table 1. In summary, we introduce Envision—a causal event progression benchmark built upon world knowledge, centered on causal spatiotemporal constraints, and generating multi-image sequences from chained textual descriptions. We meticulously design Envision-Score, a quantitative evaluation metric integrating multidimensional aesthetic, consistency, and physical considerations, compelling models to simulate coherent causal trajectories rather than merely matching static patterns. This prompts profound reflection on the transition from individual event frames to the entire process of the event.

Envision evaluates 5 UMMs and 10 specialize T2I models across 1,000 four-stage prompt sequences spanning six subdomains in natural sciences and humanities history. The benchmark assesses whether models can genuinely internalize the world knowledge introduced during training and maintain the capabilities reflected in previous benchmarks, within dynamically constrained causal spatiotemporal “event processes.” In Envision, understanding and generation are tightly coupled: failures in generating any event frame within the overall event sequence not only highlight the limitations of prior static single-image benchmarks but also reveal disconnections or conflicts between these two capacities in UMMs. These critical shortcomings expose the substantial gap between ambitious ideals and actual capabilities. Our main contributions are as follows:

- • Novel Benchmark. We introduce Envision, a novel benchmark designed to evaluate T2I models’ ability to produce dynamic causal event sequences through multiple image sequences.
- • Event-Level Evaluation. We propose Envision-Score, a dedicated metric for comprehensively evaluating event-level multi-image sequences across aesthetics, consistency, and physical.
- • Analysis and Insights. We reveal the performance gap between current T2I models in static, isolated single-image generation and dynamic, causal reasoning-based multi-image generation.

#### 2 Why the Multiple-Image Sequence?

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

###### EventProcesses

“Continuous”

Multi-Image

Frame

Multiple Image

Single Image

Video

Sequence Frame

Temporal

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

“Discrete”

Figure 2: Overview of Envision’s Multi-Image Framework.

###### 2.1 Multi-Image as a Flexible Framework for Visual Narrative Construction.

The multi-image framework (Figure 2) introduced by Envision establishes a flexible framework for modeling event processes across diverse spatiotemporal constraints. By moving beyond singleimage generation—which captures only isolated moments—this approach enables the construction of coherent visual narratives that seamlessly adapt to both continuous and discrete event representations. In continuous temporal evolution, models are required to generate smooth, physically consistent transitions that adhere to conservation laws, while in discrete contexts, they must preserve logical progression despite leaps in temporal resolution. This dual capacity facilitates rigorous evaluation of a model’s ability to handle both fine-grained dynamic processes and high-level event sequencing. Spatially, the methodology offers complementary versatility: sequences can maintain consistent viewpoints to document progressive changes within fixed scenes, or employ varying perspectives to trace developments across different spatial contexts. This spatial adaptability, combined with temporal flexibility, enables the representation of intricate world processes characterized by complex spatiotemporal interactions. For a detailed description, please refer to the Appendix D.

###### 2.2 Bidirectional Verification Between Understanding and Generation.

The Envision establishes a bidirectional evaluation paradigm for T2I models, rigorously examining the interaction between understanding and generation within UMMs. It constructs visual narrative scenes through progressively generated image outputs. This serves to demonstrate whether the UMM outperforms specialized T2I models in internalizing world knowledge and generating expression, and whether understanding and generation are truly unified, dynamically interleaved, and mutually reinforcing. In the forward direction of (Understanding → Generation), a model’s internalized world knowledge and causal reasoning capabilities are systematically probed. This process mandates the generation of a causally coherent multi-image sequence from an abstract conceptual understanding, and must strictly adhere to spatiotemporal and physical constraints. The occurrence of an interruption in the sequence generation—such as producing an illogical state transition or violating physical laws—constitutes a critical diagnostic indicator, revealing a fundamental deficiency in the model’s capacity to internalize world knowledge and deploy it for visual generative expression.

Conversely, (Generation → Understanding) highlights how the act of generation itself functions as an analytical mechanism that deepens understanding. Whole process is fundamentally an interleaving cycle: When confronted with ambiguities or inconsistencies during sequential frame generation, the model must internally refine its causal representations based on this visual state. By treating visual generation as the result of interleaved reasoning processes of understanding and generation, this approach examines the model’s dynamic conceptual and structural comprehension of an event sequence during its step-by-step generation. This enables analysis of whether the integration of understanding and generation within UMMs exhibits disconnects or conflicts, and whether this architectural paradigm offers advantages over specialized T2I models.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

###### Case

Biology

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Event Process

Discrete

Causal Structure

“Chameleon color adaptation”

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Physics

Case

[Figure 44]

[Figure 45]

Event Process

[Figure 46]

[Figure 47]

Continuous

Causal Structure

“Pendulum momentum transfer”

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Case

Chemistry

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Event Process

Continuous

Causal Structure

“Thermal reaction in crucible”

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

###### Case

Geography

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Event Process

Discrete

Causal Structure

“Volcanic eruption and restoration”

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Meteorology

Case

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Event Process

Discrete

Causal Structure

“Thunderstorm formation–dispersal”

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Case

History

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Event Process

Discrete

Causal Structure

“Industrial Revolution transitions”

- Figure 3: Spanning six disciplines, representative cases showcase the causal structure of processes as discrete (dashed) or continuous (solid) spatial relations over time.

###### Dataset Suite

###### Evaluation Suite

[Figure 80]

[Figure 81]

[Figure 82]

System Instruction:

[Figure 83]

Robustness and stability

You are an AI Quality auditor for text-to-image generation.

###### Prompt Template

Science

###### SearchFilter

Physics Chemistry Biological

[Figure 84]

[Figure 85]

Subject Textbooks

###### Physicality Consistency Aesthetics

20%40%40%

"step": 1/2/3/4, "prompt": "This is the xth

Human evaluate relevance

Dimensions

Evaluation

[Figure 86]

event frame of a continuous four-stage event sequence showing...",

[Figure 87]

Geography

[Figure 88]

Metrics attribute mining

Meteorology

[Figure 89]

"explanation": "This step depicts ..."

Culture & History

Encyclopedia Website

History

###### Dimension Structure

VLM Setting

Metric Analysis

[Figure 90]

[Figure 91]

[Figure 92]

###### Instruction

VLM ScoringCriteria

[Figure 93]

T2I / UMM

Generation

[Figure 94]

(0-5)

[Figure 95]

[Figure 96]

[Figure 97]

Prompt

- 0 (Failure):
- 1 (Very Poor):
- 2 (Poor):

- 3 (Fair):
- 4 (Good):
- 5 (Excellent):

“You’re a ... Clearly define...”

Text Multi-Image

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Need to refine.

[Figure 102]

Quality passed.

VLM as Judge

Image Generation

- Figure 4: Overview of the Event Process in generating and evaluating multi-image sequences for Envision. The process involves constructing a Dataset Suite using multi-domain knowledge sources (Science, Culture & History) and structured prompt generation. This feeds into an Image Generation process via a Text-to-Image model, followed by an Evaluation Suite which includes both Physicality, consistency, and aesthetics, and VLM as Judge and Metric Analysis using specific scoring criteria (0-5).

#### 3 Envision

Envision is founded on a fundamental premise: genuine visual intelligence must transcend the limitations of static isolation to encompass the understanding and simulation of dynamic world processes. Current T2I evaluation paradigms, predominantly focused on single-image tasks, fail to capture the causal relationships and spatiotemporal continuity inherent in real-world phenomena. To address this limitation, we propose Envision—a comprehensive benchmark centered on multi-image generation that compels models to generate the event process image by image, thereby demonstrating their ability to represent the true event sequences of the world.

###### 3.1 World Knowledge Dataset Suite

Envision is constructed on two core domains—natural sciences and humanities history—drawing on carefully curated information from academic textbooks and online sources that center on real-world events, covering a total of six disciplines. To balance the difficulty of generating visual narratives with the pressure of context length in evaluation, we employ a sequential prompt template consisting of four-stage prompts. Under human experts supervision, pre-collected and filtered information was utilized by GPT-4o to generate and refine four-stage narrative prompts for each event. This process ultimately produced 1,000 high-quality event sequences, comprising 4,000 text-to-image prompts, as illustrated in Figure 5 and Figure 3.

###### 3.1.1 Nature Science

This category evaluates the model’s internalization and representation of natural science phenomena and fundamental laws, comprising 75% of the dataset. Its five sub-dimensions each contain 150 sequences. For specific examples, see Figure 8a.

- • Physics: The tasks in this domain aim to examine the model’s reasoning capabilities regarding core principles such as mechanics, thermodynamics, and electromagnetism.
- • Chemistry: Assess the level of understanding of fundamental chemical principles, including chemical reaction kinetics, molecular interactions, and phase transitions, etc.

###### Step 2 Step 3 Step 4 Understanding

Step 1

###### Generation

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

###### Consistency

Physicality Aesthetics

Physics

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Envision Score

Chemistry

Evaluation Dimension

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Dataset Domain

Biology

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

e

u r

Geogrephy

u l t

###### e

###### c

C

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

###### n

###### c i e

Meteorology

###### S

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Culture & History

Figure 5: Overview of the Envision Data Domain & Evaluation Dimension.

- • Biology: Evaluations focus on biological processes such as ecological succession, predator-prey dynamics, organismal life cycles, and evolution.
- • Meteorology: Focusing on dynamic atmospheric systems, specific process deductions are made based on principles of meteorology and thermodynamics to infer the development and changes in space and time.
- • Geography: Study geology, geomorphology and environmental systems. Its tasks include geological processes such as river meandering and coastal erosion, and reasoning about the topographic evolution driven by factors such as erosive forces, tectonic activities.

###### 3.1.2 History & Cultural

This category encompasses human social development and historical evolution. Grounded in the context of human history and cultural knowledge, it collectively accounts for one-quarter of the total data volume. For specific examples, please refer to 8b.

• World History & Cultural Commonsense: This category requires models not only to comprehend the human intent and social logic underlying event processes, but also to discern the intricate connections between scientific and technological advancements and sociocultural phenomena during periods of social transformation.

###### 3.1.3 Causal Structure

Across all domains, Envision sequences are fundamentally grounded in temporal causality, requiring models to generate events that unfold logically over time. However, to rigorously test different facets of world modeling, we stratify these sequences into two distinct spatial-temporal structures: Continuous and Discrete (Figure3).

- • Continuous Causality: These sequences maintain a consistent spatial context with smooth, uninterrupted temporal progression, typical of fine-grained physical dynamics (e.g., chemical reactions, immediate mechanical processes). This structure compels models to strictly adhere to physical conservation laws and accurately simulate subtle state transitions.
- • Discrete Causality: These sequences involve significant spatial leaps or substantial temporal jumps between frames, commonly found in macro-scale processes (e.g., geological evolution, multi-stage life cycles, historical narratives). This structure necessitates abstracting high-level logic and maintaining long-range coherence across discontinuous contexts, evaluating its capacity for abstract causal reasoning.

###### 3.2 Evaluation Suite

Envision’s evaluation is built upon a meticulously designed workflow, aiming to achieve an objective and reasonable assessment of multi-image and text pairs that constitute event-level data. To ensure objectivity, reproducibility, and scoring efficiency, moving beyond subjective human judgments, we adopt a composite methodology centered on three pillars: (1) Event Sequence-level Evaluation, (2) Deconstruction and Integration of Evaluation Dimensions, and (3) VLM-as-Judge Evaluation.

###### 3.3 Evaluation Dimensions

To fulfill the stringent requirements of event-level visual narrative, our evaluation framework is founded upon three necessary dimensions: Consistency, Physicality, and Aesthetics. Furthermore, to ensure comprehensive and granular reliability, particularly given the inherent limitations of VLM-asJudge models in differentiating subtly similar criteria, we meticulously delineate each dimension into distinct, non-overlapping sub-metrics. The specific evaluation criteria are as follows:

- • Consistency: Ensuring an unbroken multi-image sequence, this metric evaluates the preservation of logical, factual, and narrative coherence throughout the sequence. Sub-dimensions include: Semantic Consistency, Spatial-Temporal Consistency, and Factual Consistency.
- • Physicality: Focused on the plausibility of dynamic processes, this metric quantifies a model’s internalization of physical laws and its capacity for reliable simulation. Sub-dimensions include: Basic Properties, Dynamics and Interactivity, Physical Reliability.
- • Aesthetics: Ensure that narrative expression does not come at the expense of aesthetic quality. Sub-dimensions include: Expressiveness, Aesthetic Quality, and Authenticity.

###### 3.4 Scoring Rubric

Each metric is scored by GPT-4o on a discrete integer scale from 0 to 5. This discrete scale serves to mitigate both extreme scoring tendencies and central tendency bias, thereby ensuring a statistically discriminative distribution of scores. This rubric provides a standardized measure of quality, with clear criteria for each score level to ensure inter-rater reliability.

- • 5 (Excellent): The generated sequence of images is flawless regarding the specific sub-dimension. It perfectly adheres to the prompt and the principles of narrative.
- • 4 (Good): The generated sequence of images exhibits only minor, inconsequential deviations. The core causal logic and visual intent are fully preserved.
- • 3 (Fair): There are noticeable errors, but they do not fundamentally violate the causal chain or the primary objective of the prompt.
- • 2 (Poor): Contains significant errors that undermine the causal narrative or visual consistency. The generated state transition is illogical or physically implausible.

- • 1 (Very Poor): The generated sequence of images are fundamentally flawed, bearing little resemblance to a causally or semantically correct outcome.
- • 0 (Failure): The model fails to produce a relevant output, or the output is completely nonsensical, artifact-ridden, or non-compliant with the task.

###### 3.4.1 Envision-Score

During the evaluation process, GPT-4o serves as the scoring model, processing input pairs composed of four-stage narrative prompts and model-generated multi-image sequences for t2I evaluation. Based on the predefined evaluation dimensions and scoring criteria, scores are calculated for three evaluation dimensions—consistency, physicality, and aesthetics—along with their respective sub-metrics. These scores are then consolidated through a weighted average calculation to derive the Envision score.

Weighted Scoring Formulation. Let D = {C, P, A} denote the set of primary evaluation dimensions, where C represents Consistency, P represents Physicality, and A represents Aesthetics. Each primary

dimension d ∈ D comprises a set of sub-dimensions Sd = {sd,1, sd,2, . . . , sd,nd}, where nd is the number of sub-dimensions for dimension d, and each sd,i ∈ [0, 5] represents the score for the i-th sub-dimension.

The score for each primary dimension is computed as the arithmetic mean of its constituent subdimensions:

|Sd|

1 |Sd|

##### ∑

sd,i for d ∈ D (1)

Sd =

i=1

This method ensures equal weighting within each dimension, reflecting our assessment that all sub-dimensions within Consistency and Physicality contribute equally to their respective constructs, and similarly for Aesthetics.

The comprehensive Envision (Overall) Score, SOverall, is then computed as a weighted combination of the three primary dimension scores:

SOverall = βC · SC + βP · SP + βA · SA (2) where the weighting coefficients are defined as βC = 0.4, βP = 0.4, and βA = 0.2, satisfying the

constraint βC + βP + βA = 1.

This formulation explicitly prioritizes causal reasoning by assigning 80% of the total weight to Consistency and Physicality dimensions, while maintaining a balanced assessment of visual quality through the Aesthetics dimension.

Multi-trial Evaluation Protocol. To ensure the statistical reliability of our automated evaluation framework, we implement a comprehensive multi-trial evaluation protocol. For each generated sequence comprising the quadruple {(It, Pt}4t=1, where It denotes the image at step t, Pt represents the corresponding prompt, we conduct K independent evaluation trials under identical conditions.

For each trial k ∈ {1,2, . . . , K}, the evaluation framework produces a complete evaluation tuple:

Tk = (sk,rk) (3) where:

- • sk ∈ [0,5]9 represents the fine-grained score vector across all nine sub-dimensions.
- • rk denotes the complete textual rationale generated by the evaluator model.

This multi-trial data collection enables a dual-faceted analysis of evaluator reliability. First, we perform metric analysis by computing the empirical mean and standard deviation (or upper/lower bounds) for each sub-dimension score across the K trials:

K

1 K

s(dk,i) (4)

##### ∑

µsd,i =

k=1

Metric Analysis for Evaluation. Furthermore, we engaged five domain human experts (all holding PhDs) and models from the Gemini[10], GPT[21], and Qwen[1, 45] families for evaluation.Each expert selected 50 prompt sequences for each data dimension, with GPT-4o generating corresponding images. The resulting image-text pairs underwent five rounds of evaluation by VLMs and human experts. After analyzing metric outcomes and validating output visualizations, GPT-4o was selected as the automated evaluation tool due to its highly consistent alignment with human expert judgments and robust, stable distribution. This minimizes random fluctuations and ensures assessment consistency. For detailed information, please refer to the Appendix A and Figure 9.

#### 4 Experiments

###### 4.1 Experiment Setup

Our experimental evaluation encompasses 15 models, categorized into three groups: 8 Open-Source T2I Models, 2 Closed-Source T2I Models and 5 UMMs.

- • Open-Source T2I Models: The open-source T2I model group includes Stable-Diffusion-3.5flash, Stable-Diffusion-3.5-medium, Stable-Diffusion-3.5-large[29], FLUX-dev, FLUX-pro-1.1, FLUX-pro-1.1-ultra, FLUX-kontext-pro and FLUX-kontext-max[2].
- • Closed-Source T2I Models: The closed source T2I model group includes GPT-4o[21], Gemini2.5-Flash-Image (nano-banana)[10].
- • Unified Multimodal Models (UMMs): The UMMs under evaluation are Janus-Pro-7B[6], Hunyuan Image 3.0[3], Bagel[11], Seedream 4.0[32], Qwen-Image[40].

Evaluation Protocol. For the “VLM-as-Judge” component of our evaluation pipeline (see Section 3), we employed GPT-4o as the scorer. To ensure fair and reproducible comparisons, image generation was performed using the official default configuration for each model with a fixed random seed. All experiments were conducted on a cluster of eight NVIDIA A800 GPUs.

Table 2: Comparing Envision Scores for T2I Models in Science and Culture Domains.

Domain-Specific Performance

Model

Overall Physics Chemistry Biology Geography Meteorology Culture

Open-Source T2I Models

FLUX-dev 37.62 58.86 57.12 57.27 58.75 51.01 53.44 FLUX-pro-1.1 39.52 58.52 56.15 54.29 57.97 57.62 54.01 FLUX-pro-1.1-ultra 39.69 55.08 56.51 54.54 53.15 54.27 52.21 FLUX-kontext-pro 43.78 61.72 61.36 55.00 58.41 63.45 57.29 FLUX-kontext-max 42.82 58.72 62.96 60.99 62.40 57.76 57.61 SD-3.5-flash 35.61 40.43 53.73 50.72 49.12 51.69 46.88 SD-3.5-medium 36.89 41.30 51.61 57.47 53.68 47.13 48.01 SD-3.5-large 36.07 42.32 50.24 51.12 55.43 47.34 47.09 Closed-Source T2I Models

GPT-4o 58.87 66.55 78.55 78.40 78.69 81.83 73.81 Gemini-2.5-Flash-Image 57.47 62.91 67.63 75.38 69.74 69.94 67.18 Unified Multimodal Models

Seedream 4.0 51.06 57.27 76.92 66.09 67.35 65.55 64.04 Qwen-Image 47.98 56.22 76.40 63.81 58.94 66.01 61.56 Hunyuan Image 3.0 37.84 49.76 51.27 70.49 67.74 62.10 56.53 Bagel 39.40 56.25 57.65 51.00 58.20 72.40 55.82 Janus-Pro-7B 36.24 44.08 53.09 55.05 62.70 50.52 50.28

Table 3: Comparison of performance in each dimension of the T2I models (All Domains Average).

Evaluation Dimensions SC FC STC CS Exp AQ Auth AS PR BP DI PS Open-Source T2I Models

Model

FLUX-dev 54.11 56.30 40.98 50.37 71.05 75.66 53.02 66.60 45.23 58.97 45.84 49.96 FLUX-pro-1.1 55.20 55.89 40.56 50.45 72.95 76.73 52.67 67.30 46.96 58.18 47.62 50.88 FLUX-pro-1.1-ultra 52.04 54.53 40.71 49.01 68.69 72.68 50.29 63.75 44.88 58.28 45.82 49.61 FLUX-kontext-pro 57.09 60.33 48.46 55.22 64.63 70.09 56.10 63.53 52.70 66.04 49.94 56.19 FLUX-kontext-max 60.85 63.46 46.06 56.48 68.42 71.94 55.83 65.30 51.24 64.38 49.07 54.86 SD-3.5-flash 48.12 46.05 36.56 42.44 62.92 71.76 48.08 60.79 41.08 52.20 39.87 44.35 SD-3.5-large 40.82 50.52 33.84 43.30 63.02 67.32 45.89 58.62 41.01 53.64 40.73 45.08 SD-3.5-medium 49.79 50.89 35.64 44.08 61.65 67.86 48.86 59.35 41.66 54.88 42.29 46.23 Closed-Source T2I Models

GPT-4o 75.76 78.65 67.42 73.88 77.05 81.37 71.81 76.70 70.56 79.90 66.44 72.28 Gemini-2.5-Flash-Image 69.12 73.20 58.71 66.92 73.29 75.65 64.10 70.95 62.68 74.45 59.50 65.52 Unified Multimodal Models

Seedream 4.0 66.15 66.78 56.79 63.18 73.54 76.12 58.61 69.32 60.05 69.83 56.91 62.24 Qwen-Image 61.16 59.88 53.20 58.03 79.54 82.57 58.05 73.23 55.97 67.11 54.69 59.22 Hunyuan Image 3.0 57.42 58.14 46.03 53.78 73.08 76.05 53.48 67.40 49.54 61.34 50.70 53.81 Bagel 58.61 57.50 51.11 55.70 65.84 69.17 48.89 61.17 51.94 61.94 46.11 53.32 Janus-Pro-7B 52.29 55.05 41.51 49.54 63.26 64.86 45.96 57.90 44.40 53.64 44.63 47.52

Note: Evaluation dimension abbreviations: SC: Semantic Consistency, FC: Factual Consistency, STC: Spatial-Temporal Consistency, CS: Consistency Score, Exp: Expressiveness, AQ: Artistic Quality, Auth: Authenticity, AS: Aesthetic Score, PR: Physical Reliability, BP: Basic Properties, DI: Dynamics & Interactivity, PS: Physicality Score.

###### Consistency Score

###### Aesthetic Score

###### Physicality Score

50.37

49.01

56.48

43.3

73.88

63.18

53.78

49.54

66.6

63.75

65.3

58.62

76.7

69.32

67.4

57.9

49.96

49.61

54.86

45.08

72.28

62.24

53.81

47.52

80

80

80

75

75

75

ConsistencyScore

70

70

70

PhysicalityScore

AestheticScore

65

65

65

60

60

60

55

55

55

50

50

50

45

45

45

40

40

40

50.45

55.22

42.44

44.08

66.92

58.03

55.7

67.3

63.53

60.79

59.35

70.95

73.23

61.17

50.88

56.19

44.35

46.23

65.52

59.22

53.32

FLUX-pro-1.1-ultraFLUX-devFLUX-pro-1.1FLUX-kontext-proFLUX-kontext-maxSD-3.5-flashSD-3.5-largeSD-3.5-mediumGemini-2.5-FlashGPT4oSeedream4.0HunyuanImage3.0Qwen-ImageJanus-Pro-7BBagel

FLUX-pro-1.1-ultraFLUX-devFLUX-pro-1.1FLUX-kontext-proFLUX-kontext-maxSD-3.5-flashSD-3.5-largeSD-3.5-mediumGemini-2.5-FlashGPT4oSeedream4.0HunyuanImage3.0Qwen-ImageJanus-Pro-7BBagel

FLUX-pro-1.1-ultraFLUX-devFLUX-pro-1.1FLUX-kontext-proFLUX-kontext-maxSD-3.5-flashSD-3.5-largeSD-3.5-mediumGemini-2.5-FlashGPT4oSeedream4.0HunyuanImage3.0Qwen-ImageJanus-Pro-7BBagel

Open-Source T2I

Closed-Source T2I

UMM

| |
|---|

| |
|---|

- Figure 6: Comparison of Model Performance on Key Evaluation Dimensions, shows the average scores of UMMs (Red), Open-Source T2I Models (Blue) and Closed-Source T2I Models (Yellow) across metrics.

###### 4.2 Evaluation Results

The comprehensive evaluation of 15 models on the Envision benchmark, summarized in Table 3 and visualized in Figure 6, reveals a fundamental performance dichotomy between UMMs, Open-Source T2I Models and Closed-Souced T2I Models. Closed-Source T2I Models, exemplified by GPT-4o and Gemini-2.5-Flash-Image, establish dominant superiority across all evaluation axes, achieving peak scores in Consistency (73.88), Aesthetic (76.70), and Physicality (72.28). This advantage appears to stem from its massive data scale, along with scalable model sizes and proprietary training methods,

endowing it with formidable generative capabilities. Open-Source T2I Models, represented by the FLUX and SD-3.5 series, demonstrate specialized proficiency in static visual fidelity, attaining high scores in Artistic Quality (FLUX-pro-1.1: 76.73). While excelling in texture rendering and stylistic composition, these models reveal a critical disconnect between surface-level aesthetics and underlying plausibility, as evidenced by substantial deficits in Factual Consistency (FLUX-kontext-pro: 60.33) and Physical Reliability (52.70). This fully reflects its lack of internalization of world knowledge. UMMs—architecturally unified models like Seedream and Qwen—occupy an intermediate position, effectively translating broad multimodal knowledge into enhanced scene understanding. This enables strong performance in knowledge-intensive domains such as Biology (Seedream: 76.92) and consistent advantages over Open-Source T2I Models in causal consistency-related metrics. Nevertheless, even these conceptually advanced UMMs currently trail behind state-of-the-art Closed-Source T2I Models in maximizing their potential for causal and physical reasoning. Most critically, the Spatial-Temporal Consistency dimension emerges as a universal challenge transcending all model categories, with the highest achieved score (GPT-4o: 67.42) remaining substantially limited—thereby validating the benchmark’s core thesis that contemporary T2I generation models, despite their impressive static synthesis capabilities, maintain fundamentally gaps in coherent spatiotemporal reasoning.

### 4.3 Key Insights from Envision Continuous Discrete

###### Flux-Kontext-max GPT-4o Bagel Flux-Kontext-max GPT-4o Bagel

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Step2Step3Step1Step4

Step2Step4Step1Step3

Element missing.

Position is correct, but status is incorrect.

A red billiard ball was stationary at the center of the pool table, while a white billiard ball was approaching it from the left.

In a rural area of England at the end of the 18th century, the scene shows farmers working manually in the fields, while nearby there is a steam engine brought by a carriage.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Incorrect deformation.

The details are unclear.

Exaggerated expression.

Exaggerated expression.

The white ball collided and came into contact with the stationary red ball, resulting in a force interaction.

In the countryside, the workers are assembling the machines and laying the tracks near the steam engine. Smoke is slowly rising from the chimney of the steam engine.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Exaggerated expression.

Exaggerated expression.

The white ball stopped moving, while the red ball continued to move forward with its kinetic energy.

The busy railway under construction in the countryside is now in operation with steam locomotives. Workers are loading and unloading goods, while in the background, factories are emitting smoke.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

No actual movement.

The white ball was almost stationary near the impact point, while the red ball was moving outward.

The original rural area now presents a highly industrialized scene, featuring busy factories, densely populated urban residential areas and crowded railway stations.

- Figure 7: Visualization of Causal Event Progression and Failure Analysis. This figure compares the multi-step generative outputs of various generative model, including Flux-Kontext-max (Open-Source T2I Model), GPT-4o (Closed-Source T2I Model), Qwen-Image (UMM) and Bagel (UMM), across two distinct causal scenarios.

Foundational Deficits in Dynamic Event Modeling. The Envision benchmark reveals a fundamental limitation in contemporary multimodal T2I models: their inability to conceptualize and represent events as coherent spatio-temporal processes. Despite extensive training in large-scale static image datasets, these models demonstrate a systematic failure to internalize the causal mechanisms that govern dynamic evolution in the physical world. This deficit manifests as a striking performance

dichotomy (as shown in Table 2 and Figure 7)—while models exhibit reasonable competence in handling commonsense scenarios that rely on static pattern recognition, they show profound deficiencies in scientific domains requiring strict adherence to physical laws and temporal constraints. The underlying challenge originates from the inherent tension between discrete and continuous event representation. Discrete event sequences can be partially addressed through static pattern matching and interpolation, whereas continuous processes demand deeply integrated world knowledge and explicit causal reasoning to maintain coherence across state transitions. Critically, our findings point to a breakdown in temporal reciprocal reasoning: the models fail to use the currently generated visual state as a validated memory scaffold to guide and constrain the subsequent causal evolution.

The Understanding-Generation Paradox: Fundamental Disconnect. Our investigation establishes the Understanding-Generation Paradox as a critical failure of UMMs to execute effective reciprocal temporal reasoning. This paradox is not merely a performance gap but a systemic architectural conflict: the knowledge optimized for static, discrete representation (Understanding) is decoupled from the policy required for dynamic, continuous manipulation (Generation). Specifically, the generated visual state fails to serve as a rigorously validated causal memory trace for the next step, and the latent causal understanding fails to impose sufficient predictive constraints on the sequential generation policy. This breakdown is evidenced across two axes (Figure 7): in discrete event scenarios, models exhibit “visual fluency without causal fidelity,” suggesting their success relies on statistical pattern interpolation without the underlying generative self-verification demanded by true causality. Conversely, in continuous process scenarios, models show “nascent understanding with generative collapse,” where the inability to maintain spatio-temporal consistency leads to the catastrophic breakdown of physical laws. Moreover, this divergence and the performance gap between UMMs and closed-source T2I models highlight a critical insight: the unification of understanding and generation remains significantly disjointed and conflicted. Resolving this paradox may require a paradigm shift toward explicitly demanding learning frameworks that interweave understanding and generation. This involves exploring more efficient and scalable architectures and training paradigms, ultimately validating and optimizing their internalization and representation of world knowledge through large-scale data.

Beyond Performance Gaps: Architectural and Conceptual Barriers. The deficiencies revealed by the Envision are not limited to the shortcomings quantified by the surface Enscore but fundamentally expose a core issue in current UMMs: the disconnection between understanding and generation. The consistent inability of diverse architectures to enforce causal continuity and perform complex spatio-temporal reasoning suggests that these barriers stem not from architectural limitations but from a foundational conceptual flaw: the inductive bias primarily based on static, causally isolated individual images within the data and the training paradigm built upon it. When confronted with the need to simulate dynamic processes, the generation module often degrades into shallow pattern matching or semantic mapping, failing to effectively leverage the deeper, reasoned world knowledge potentially stored in the understanding module. Crucially, this limitation stems from the lack of architectural mechanisms that enforce the interleaving of understanding and generation, where semantic consistency and logical deductions explicitly constrain and guide the visual output across sequential steps. Conceptually, this requires incorporating video frame sequences or causally linked image sequences as native multi-image data modalities during training. This explicitly injects necessary spatio-temporal induction biases, ensuring that deeper world knowledge understanding is tightly interleaved with generation and genuinely manifested in multi-image visual narratives. To bridge the chasm between capturing isolated “world states” and simulating continuous “world processes,” future progress necessitates a fundamental shift from static pattern matching to integrated world models, even as the explicit Chain-of-Thought (CoT) mechanism serves as a promising, if external, universal bridge to enforce understanding’s constraints and guide the multi-step causal generation process in the interim.

#### 5 Related Work

- 5.1 Unified Multimodal Models (UMMs)

Unified Multimodal Models (UMMs) aim to integrate diverse tasks like visual understanding and generation within a single architecture, promoting cross-modal synergy and reducing complexity. Recent works fall into several paradigms: Autoregressive (AR) Models like Chameleon [36], Janus [41], and Emu3 [38] tokenize visual inputs for sequential generation, while Show-O [43] incorporates a discrete-diffusion schedule for refined token prediction. AR with Diffusion Head approaches, such as Transfusion, often keep a pre-trained MLLM frozen and route its features to an external generator [37, 33] to enable complex multimodal interactions. Alternatively, Integrated Transformers [50, 4] unify different paradigms in one backbone to eliminate bottlenecks. To enhance scalability, the Mixture-ofTransformers (MoT) paradigm [26, 11] introduces a sparse, modular design, as seen in Bagel. However, progress in model architecture must be gauged through rigorous evaluation. Current evaluations, however, still rely heavily on static single-image benchmarks, which can mask fundamental limitations like the disconnection and opposition between understanding and generation.

5.2 Multimodal Understanding and Generation Benchmarks

Landscape of evaluation benchmarks for multimodal T2I model is evolving, shifting from isolated assessments of singular capabilities toward more integrated, comprehensive evaluations. In the domain of understanding, the research focus has moved from basic perception to an “thinking-withimages” paradigm that incorporates generative abilities[48, 34]. Concurrently, generative evaluation has progressed beyond foundational metrics of semantic fidelity[22]. It now involves calibrating the performance of unified models against traditional generative counterparts in text-to-image[30] and image editing[47], specifically targeting capability gaps across key dimensions such as consistency[39, 16, 20], world knowledge[28, 23, 13, 27, 9]. While there is a burgeoning effort to illustrate the benefits of unified interaction through reasoning tasks[35] and the analysis of coupled versus decoupled generation-understanding frameworks[44, 51, 25, 14, 19, 49], a significant limitation persists: these benchmarks remain predominantly anchored to single-image generation tasks, which inherently fail to capture the causal and temporal dynamics of event processes. We address this limitation by introducing multi-image sequential scenarios, which transition evaluation from a conventional “static” paradigm to a “dynamic”. By meticulously decomposing and reconstructing previous evaluation metrics, Envision enables a more intuitive, authentic, and multifaceted assessment of a model’s actual performance across all established dimensions.

#### 6 Conclusion

In conclusion, Envision establishes that prevailing paradigms in multimodal T2I model, anchored in static single-image generation and evaluation, are fundamentally insufficient for achieving genuine world understanding. The Envision benchmark and its findings expose a critical “UnderstandingGeneration Paradox”, revealing that proficiency in depicting isolated scenes does not equate to an internalized model of dynamic, causal processes. The pronounced performance gap between UMMs and T2I models, coupled with the universal bottleneck in spatiotemporal reasoning, underscores a systemic limitation of current architectures, data and training regimens. Our results compellingly argue that future progress hinges on a foundational shift—from optimizing for static pattern matching to architecting models capable of world simulation.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506, 2025.
- [3] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.
- [4] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [5] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12501–12511, 2025.
- [6] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [7] Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, and Tieniu Tan. Vidcapbench: A comprehensive benchmark of video captioning for controllable text-to-video generation. arXiv preprint arXiv:2502.12782, 2025.
- [8] Yubin Chen, Xuyang Guo, Zhenmei Shi, Zhao Song, and Jiahao Zhang. T2vworldbench: A benchmark for evaluating world knowledge in text-to-video generation. arXiv preprint arXiv:2507.18107, 2025.
- [9] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. arXiv preprint arXiv:2501.16411, 2025.
- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [11] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [12] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.
- [13] Xingyu Fu, Muyu He, Yujie Lu, William Yang Wang, and Dan Roth. Commonsense-t2i challenge: Can text-to-image generation models understand commonsense? arXiv preprint arXiv:2406.07546, 2024.
- [14] Sara Ghazanfari, Francesco Croce, Nicolas Flammarion, Prashanth Krishnamurthy, Farshad Khorrami, and Siddharth Garg. Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning. arXiv preprint arXiv:2506.00318, 2025.
- [15] Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, and Jiale Zhao. T2vphysbench: A firstprinciples benchmark for physical consistency in text-to-video generation. arXiv preprint arXiv:2505.00337, 2025.
- [16] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.
- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative

- models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807– 21818, 2024.
- [18] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024.
- [19] Ziqi Huang, Ning Yu, Gordon Chen, Haonan Qiu, Paul Debevec, and Ziwei Liu. Vchain: Chain-of-visualthought for reasoning in video generation. arXiv preprint arXiv:2510.05094, 2025.
- [20] Ziwei Huang, Wanggui He, Quanyu Long, Yandi Wang, Haoyuan Li, Zhelun Yu, Fangxun Shu, Long Chan, Hao Jiang, Fei Wu, et al. T2i-factualbench: Benchmarking the factuality of text-to-image models with knowledge-intensive concepts. arXiv preprint arXiv:2412.04300, 2024.
- [21] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [22] Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. Genai arena: An open evaluation platform for generative models. Advances in Neural Information Processing Systems, 37: 79889–79908, 2024.
- [23] Jialuo Li, Wenhao Chai, Xingyu Fu, Haiyang Xu, and Saining Xie. Science-t2i: Addressing scientific illusions in image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2734–2744, 2025.
- [24] Ouxiang Li, Yuan Wang, Xinting Hu, Huijuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, and Fuli Feng. Easier painting than thinking: Can text-to-image models set the stage, but not direct the play? arXiv preprint arXiv:2509.03516, 2025.
- [25] Yi Li, Haonan Wang, Qixiang Zhang, Boyu Xiao, Chenchang Hu, Hualiang Wang, and Xiaomeng Li. Unieval: Unified holistic evaluation for unified multimodal understanding and generation. arXiv preprint arXiv:2505.10483, 2025.
- [26] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.
- [27] Fanqing Meng, Wenqi Shao, Lixin Luo, Yahong Wang, Yiran Chen, Quanfeng Lu, Yue Yang, Tianshuo Yang, Kaipeng Zhang, Yu Qiao, et al. Phybench: A physical commonsense benchmark for evaluating text-to-image models. arXiv preprint arXiv:2406.11802, 2024.
- [28] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.
- [29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller,¨ Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [30] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.
- [31] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [32] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

- [33] Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024.
- [34] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025.
- [35] Kaiyue Sun, Rongyao Fang, Chengqi Duan, Xian Liu, and Xihui Liu. T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472, 2025.
- [36] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [37] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.
- [38] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [39] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025.
- [40] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [41] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025.
- [42] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.
- [43] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [44] Wulin Xie, Yi-Fan Zhang, Chaoyou Fu, Yang Shi, Bingyan Nie, Hongkai Chen, Zhang Zhang, Liang Wang, and Tieniu Tan. Mme-unify: A comprehensive benchmark for unified multimodal understanding and generation models. arXiv preprint arXiv:2504.03641, 2025.
- [45] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [46] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 1(2):6, 2023.
- [47] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.
- [48] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [49] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025.

- [50] Chuyang Zhao, Yuxing Song, Wenhao Wang, Haocheng Feng, Errui Ding, Yifan Sun, Xinyan Xiao, and Jingdong Wang. Monoformer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280, 2024.
- [51] Kai Zou, Ziqi Huang, Yuhao Dong, Shulin Tian, Dian Zheng, Hongbo Liu, Jingwen He, Bin Liu, Yu Qiao, and Ziwei Liu. Uni-mmmu: A massive multi-discipline multimodal unified benchmark. arXiv preprint arXiv:2510.13759, 2025.

## Appendix

#### A More Details about Envision

This appendix serves as a supplementary compendium to the main text, providing exhaustive methodological details, formal definitions, and extended analyses that underpin the Envision.

#### B Definition of Categories in Envision

This appendix provides a formal definition of each category and sub-category within the benchmark, elucidating the specific cognitive capabilities they are designed to probe.

###### B.1 Natural Science

This category of evaluation models assesses the internalized understanding of fundamental laws governing nature. Achieving success requires not only maintaining robust semantic consistency under spatiotemporal constraints, but also internalizing world knowledge and deducing the sequential natural scientific processes—including physical, chemical, and biological phenomena—that compose each event frame and the entire multi-image sequence of events within a complete event progression.

- • Physics: This sub-category evaluates qualitative and semi-quantitative reasoning about core principles from classical mechanics, thermodynamics, optics, and electromagnetism. Scenarios are engineered to have unambiguous, visually verifiable outcomes based on physical laws. The model must demonstrate an understanding of state transitions governed by forces, energy, and conservation laws. Exemplar Task: As shown in Figure 5. “A white billiard ball rolls across a table and strikes a stationary red billiard ball. Show the sequence of what happens during and after the collision.” A correct sequence must depict the transfer of momentum between the balls, with the white ball slowing down or stopping while the red ball moves away, thereby illustrating the principles of conservation of momentum and energy in elastic collisions.
- • Chemistry: Tasks in this domain probe the model’s comprehension of molecular-level interactions and their macroscopic consequences. This includes reasoning about reaction kinetics (precipitation, combustion), stoichiometry, and phase transitions. The benchmark requires models to infer the visual outcomes of chemical processes, moving beyond symbolic representations to dynamic visualizations of transformation. Exemplar Task: As shown in Figure 5. “Clear lead nitrate solution and potassium iodide solution are mixed together in a beaker. Show the sequence of what happens immediately after mixing.” A correct sequence must depict the instantaneous formation of a bright yellow precipitate of lead iodide, demonstrating a double displacement precipitation reaction and the transition from soluble reactants to insoluble products.
- • Biology: Evaluations focus on quintessential biological processes that operate across various scales, from individual to ecological. This includes modeling life cycles (e.g., metamorphosis), species evolution, predator-prey population dynamics, and ecosystem succession. The model must reason about temporal progressions driven by biological imperatives like growth, reproduction, natural selection, and resource competition. Exemplar Task: As shown in Figure 5. “A whale carcass sinks to the deep ocean floor. Show the sequence of its decomposition over time.” A coherent sequence should illustrate the ecological succession: from initial consumption by mobile scavengers like sharks, through colonization by smaller organisms on the remaining tissues and bones, to the final stage where specialized bone-eating worms and bacteria break down the skeleton, completing the nutrient cycle.

- • Geography: This sub-category focuses on long-term geomorphological processes and the spatial relationships between physical and human features on Earth’s surface. It challenges models to extrapolate the slow, yet deterministic, evolution of landscapes and human-environment interactions. Reasoning is based on causal drivers like tectonic forces, erosion, deposition, and human modification. Exemplar Task: As shown in Figure 5. “An island volcano erupts. Show the sequence from the eruption to the ecological recovery over an extended period.” A coherent sequence should visually demonstrate the causal stages: (1) the initial volcanic eruption with lava flows and ash covering the landscape; (2) the gradual cooling and solidification of volcanic materials, forming new landforms; (3) the pioneering stage where initial plant life and organisms colonize the barren terrain; (4) the eventual establishment of a recovered ecosystem with diverse flora and fauna, illustrating long-term ecological succession following a major disturbance.
- • Meteorology: This sub-category focuses on short-to-medium-term atmospheric processes and weather phenomena. It evaluates a model’s ability to reason about the formation, progression, and dissipation of weather systems based on thermodynamic principles and fluid dynamics. Tasks require understanding the visual manifestations of atmospheric states and their sequential evolution. Exemplar Task: As shown in Figure 5. “Over a Gobi desert landscape, show the sequence from the formation of rain clouds to the end of a thunderstorm.” A correct sequence must depict the causal stages: (1) the initial formation and gathering of cumulus clouds in the sky; (2) the development into dark, towering cumulonimbus clouds; (3) the occurrence of rainfall and potential lightning over the arid land; (4) the final dissipation of clouds and the clearing of the sky, leaving a moistened ground.

###### B.2 Cultural and History

This category aims to evaluate a model’s alignment with shared human knowledge, social conventions, and historical narratives. It assesses the model’s comprehension of intent, cultural logic, and social causality, and deduces the core semantic alignment components within multi-image narrative processes alongside other relevant variations across the entire pictorial narrative—elements that are often implicit and context-dependent.

• Cultural & History: This unified area probes a model’s knowledge of stereotypical human activities and their evolution. At the micro-level, this involves understanding the script-like sequences of everyday events. At the macro-level, it requires modeling the impact of pivotal historical developments on material culture, social organization, and daily life. The core challenge is to ground abstract historical narratives or social conventions in their concrete, visual manifestations across time.

Exemplar Task: As shown in Figure 5. “Show the founding and early growth of Apple Computer in a garage during the 1970s.” A coherent sequence must visualize the key stages: (1) the initial scene of a suburban garage with two young entrepreneurs surrounded by electronic components and early blueprints; (2) the development phase with assembled circuit boards and intensified prototyping activities; (3) the small-scale production stage with multiple finished units and packaging materials; (4) the transition to a recognized company, symbolized by the garage’s eventual transformation into a historic landmark.

#### C Prompt Structure

To ensure the generation of causally coherent and visually plausible multi-image sequences for evaluation, we designed a structured JSON-based prompt template. This template has been manually designed and validated, clearly defining the sequence of phases while maintaining strict narrative

constraints between each stage. Additionally, we developed specialized prompt styles to facilitate multi-step prompt combinations and their corresponding explanations.

JSON Schema Definition Each event process scenario is represented as a JSON array containing four sequential steps. The schema is defined as follows:

{

"prompts": [ {

"step": 1, "prompt": "", "explanation": ""

}, {

"step": 2, "prompt": "", "explanation": ""

}, {

"step": 3, "prompt": "", "explanation": ""

}, {

"step": 4, "prompt": "", "explanation": ""

} ]

}

- C.1 Event-Level Prompt Consistency Constraints The prompt structure enforces several critical constraints to ensure evaluation validity:

- • Narrative Logic of Events: For the prompt design of each step, we remind that “this is the Xth event frame in a four-stage event progression, ...” and the explanation for that stage provides a detailed description and hints for the events occurring in the prompt of that step. Ultimately, the prompts of the four steps serve as four event frames, forming a multi-image event sequence.
- • Viewpoint Consistency: All four frames must maintain a reasonable camera position, angle, and field of view relative to the scene being depicted, unless a change in perspective is explicitly motivated by the narrative. This ensures that the model can focus on the dynamic evolution of the event itself, rather than being distracted by inconsistent visual framing. It is crucial for establishing a coherent spatial context and enabling accurate comparison of states across different temporal stages, thereby facilitating the assessment of a model’s understanding of spatiotemporal relationships.
- • Environmental Stability: Throughout the entire process, particularly in natural science scenarios such as physics and chemistry, lighting conditions, background elements, and experimental apparatus must remain consistent to maintain narrative coherence. For other scenarios, emphasis is placed on lighting, scene composition, and the arrangement of elements within the scene.

- • Temporal Progression: Each step must represent a logical, causally-driven, and progressive stage within the entire event process. The sequence should visually articulate a clear “before-and-after” relationship between consecutive frames, demonstrating a plausible evolution of states rather than a mere substitution of scenes. This requires the model to internalize not just individual states, but the dynamic mechanisms—that govern the transitions between them.

- C.2 Event-Level Prompt Framework The four-stage multi-image sequence follows a carefully designed causal narrative framework:

- 1. Step 1 - Initial State: This frame establishes the narrative’s baseline, introducing the core entities, environment, and all initial conditions. A well-defined initial state provides the necessary foundation from which the entire causal chain can unfold, anchoring the sequence in a specific and stable starting point.
- 2. Step 2 - Early Interaction: This frame depicts the inciting incident or the initial catalyst that disrupts the initial state. It captures the first dynamic interaction, a key decision by an agent, or the onset of a natural process. This step is critical for initiating the causal chain, showing the immediate consequences of the interaction and setting the narrative in motion.
- 3. Step 3 - Progressive Transformation: This frame represents the core development phase of the event. It visualizes the significant intermediate state where the initial interactions lead to pronounced changes, cascading effects, or complex developments. This step tests the model’s ability to simulate the ongoing process, maintain logical state transitions, and depict the evolving relationships between entities as the narrative progresses toward its culmination.
- 4. Step 4 - Final Resolution: This frame concludes the event sequence by showcasing the ultimate outcome and steady state resulting from the preceding transformations. It demonstrates the final configuration of the system, the long-term consequences of the actions, and the fulfillment of the causal chain. A coherent resolution should align with established principles, providing a logically and visually satisfying endpoint to the narrative.

This structured approach ensures that the generated sequences maintain subject plausibility while providing a standardized basis for evaluating model performance on T2MI tasks, with the JSON format further facilitating automated processing and analysis of evaluation results across diverse data domains. It is noteworthy that, whereas early CLIP-based models constrained prompt tokens to a limit of 77, our current focus has shifted toward advanced T2I models and UMMs; accordingly, the original prompts have been appropriately enriched with descriptive details—guided by expert annotation and supplemented by GPT-4o—to enhance expressiveness without compromising fairness, with each prompt strictly limited to within 100 tokens.

#### D Quadripartite Event Frame Structure

The Envision benchmark employs a standardized four-frame sequence to represent a complete causal event. This specific structural choice is grounded in a deliberate trade-off between representational expressivity, computational feasibility, and evaluative rigor.

###### D.1 Multiple images in Visual Narrative Generation

Visual narrative generation exists on a spectrum defined by temporal granularity. At one extreme lies single-image generation (T2I), which captures a solitary, static moment. This paradigm is inherently limited for evaluating dynamic processes, as it lacks temporal extension and cannot express state

transitions. At the other extreme lies video generation (T2V), which models continuous, high-framerate sequences. While powerful, T2V introduces significant computational complexity and often focuses on short-term motion dynamics rather than high-level, causal event progression.

The multi-image sequence (T2MI), and specifically the four-frame structure, positions itself as an optimal intermediate representation:

- • Sufficiently Discrete to Avoid Complexity: A sequence of four images is computationally less intensive to generate and evaluate than a high-frame-rate image sequence or video. It avoids the context window limitations and prohibitive memory costs associated with processing long visual sequences in current transformer-based or diffusion-based models.
- • Sufficiently Continuous to Model Causality: Crucially, four frames provide the minimal yet sufficient number of distinct states to define a complete causal narrative arc. As established in our prompt structure, these frames map directly onto the canonical stages of a narrative: Initial State (S1), Inciting Interaction (S2), Progressive Transformation (S3), and Final Resolution (S4). This structure forces the model to conceptualize an event not as a collection of independent scenes, but as a chain of causally linked states, thereby discouraging mere associative pattern matching and promoting genuine causal reasoning.

###### D.2 Cognitive and Evaluative Advantages

The four-frame paradigm is aligned with cognitive theories of event perception, where humans naturally segment continuous experience into discrete, meaningful units bounded by causal boundaries.

From an evaluation perspective, this structure provides several critical advantages:

- 1. Amplification of Consistency Errors: A sequence of four frames provides multiple transition points (S1→S2, S2→S3, S3→S4) where inconsistencies in object attributes, spatial relationships, and temporal logic can be detected. A three-frame sequence might be too concise to reveal progressive errors, while a longer sequence (e.g., five or six frames) might introduce redundant states without providing proportional diagnostic value, thereby increasing evaluation cost without a commensurate gain in insight.
- 2. Mitigation of Contextual Interference: When a long sequence of prompts or images is input into a model as a single context, there is a risk of “context dilution” or “forgetting,” where early context influences the generation of later frames in an uncontrolled manner. The four-frame sequence is short enough to fit comfortably within the context windows of modern large models, ensuring that all prompts are considered concurrently during generation.
- 3. Structured Diagnostic Granularity: The four-stage arc allows for precise fault localization. A failure in S2 indicates a model’s inability to initiate a causal chain from a stable state. A breakdown between S3 and S4 suggests a failure in projecting a process to its logical conclusion. This granularity is essential for providing actionable insights into model limitations.

#### E Scoring Criteria of Metrics

To ensure a rigorous and nuanced evaluation of model performance on the Envision benchmark, we have established a comprehensive scoring framework. This framework is built upon three core dimensions—Consistency, Physicality, and Aesthetics—each decomposed into specific, actionable sub-dimensions. Each sub-dimension is scored on a discrete 0-5 scale, where 0 represents a catastrophic failure and 5 represents flawless execution. The following sections detail the rationale and scoring rubrics for each dimension.

###### E.1 Consistency Dimension

The Consistency dimension evaluates the logical, semantic, and factual consistency of the generated sequence. It ensures that the narrative is stable, meaningful, and grounded in reality (see Figure 10).

- • Spatio-Temporal Consistency: Evaluates the coherence of spatial relationships and their logical evolution over time across the image sequence. It ensures that objects, characters, and the environment maintain a stable presence, follow plausible motion trajectories, and undergo changes in a continuous and causally sound manner. A high score (4-5) requires smooth, logical transitions where all spatial changes (e.g., movement, deformation) are physically plausible and temporally coherent, adhering to a consistent narrative timeline. A low score (0-1) indicates severe discontinuities, such as objects teleporting, erratically changing size or shape, backgrounds inconsistently shifting, or a complete breakdown in the logical progression of events.
- • Semantic Consistency: Assesses the alignment between the visual output and the conceptual meaning of the prompt. It judges whether the generated images correctly convey the intended story, action, or concept. A high score (4-5) demands that both explicit and implicit meanings are captured, whereas a low score (0-1) signifies a fundamental misunderstanding or contradiction of the core theme.
- • Factual Consistency: Gauges the adherence to empirical facts and established logical relationships. This sub-dimension is critical for scenarios involving scientific, historical, or commonsense knowledge. A top score of 5 indicates complete empirical accuracy with no detectable errors, while a score of 0 reflects violations of fundamental facts or physical laws.

###### E.2 Aesthetic Dimension

While secondary to causal reasoning, the Aesthetic dimension remains essential for evaluating the overall quality and usability of the generated content. It ensures that the sequences are not only correct but also visually compelling (see Figure 11).

- • Expressivenes: Evaluates the effectiveness and emotional impact of the visual storytelling. It assesses how well the composition, character expressions, lighting, and color are used to convey the narrative’s mood, action, and underlying causality. A high score (4-5) indicates a dynamic and evocative sequence that powerfully communicates the event’s drama and progression, while a low score (0-1) reflects a flat, inert, or emotionally disconnected portrayal that fails to engage the viewer in the narrative.
- • Aesthetic Quality: Assesses the overall artistic merit, style coherence, and emotional impact. It judges whether the visual style is appropriate, consistent, and effectively supports the narrative through masterful execution of composition, color, and lighting. A high score (4-5) indicates a visually stunning and stylistically harmonious sequence that deeply enhances the storytelling. A low score (0-1) reflects a poorly executed, inconsistent, or aesthetically unappealing result that fails to engage the viewer artistically.
- • Authenticity: Measures the believability and naturalness of the generated imagery, judging how closely it approximates the intended reality (whether photorealistic or stylistically consistent). A high score (4-5) signifies that the sequence is visually convincing and free of jarring artificial artifacts, making it indistinguishable from a real photograph or a masterfully consistent artistic rendering. A low score (0-1) is given for obvious synthetic flaws, unrealistic textures, unnatural lighting, or a cartoonish appearance that breaks immersion and undermines plausibility.

###### E.3 Physicality Dimension

The Physicality dimension is the cornerstone of our causal reasoning evaluation. It scrutinizes the model’s adherence to the principles of the physical world, from macroscopic geometry to microscopic material properties (see Figure 12).

- • Basic Properties: Evaluates the accuracy and stability of fundamental scene attributes throughout the sequence. This includes the precise maintenance of object counts, the preservation of fundamental geometric shapes and sizes, and the consistency of scale and proportions between objects. A high score (4-5) requires that all entities retain their core attributes without spurious appearance, disappearance, or morphing, while a low score (0-1) indicates severe errors such as objects vanishing, replicating, or exhibiting impossible and unstable shapes.
- • Dynamics & Interactivit: Assesses the plausibility of physical movements and interactions between entities. This dimension scrutinizes the realism of motion trajectories, force transmissions, collision outcomes, and the behavior of materials (e.g., fluids, rigid bodies). A high score (4-5) depicts interactions that are governed by coherent physics, such as a falling object following a parabolic arc or a collision resulting in momentum transfer, whereas a low score (0-1) reveals physically impossible motions, such as objects passing through each other or moving without applied force.
- • Physical Reliability: Gauges the adherence to fundamental physical laws and principles across the entire sequence. This serves as the ultimate test of the model’s internalized world model, checking for violations of thermodynamics, conservation laws, gravity, and other invariant principles. A high score (4-5) demonstrates a sequence that is entirely plausible within our physical reality, while a low score (0-1) is assigned for catastrophic failures, such as energy being created from nothing, perpetual motion, or blatant defiance of gravity.

###### E.4 Scoring Rubric and Aggregation

Each of the aforementioned sub-dimensions is scored by human experts or a qualified large visionlanguage model (VLM) using a standardized rubric:

- • 5 (Excellent): Flawless execution regarding the specific sub-dimension. No detectable errors.
- • 4 (Good): Minor, inconsequential deviations. The core intent and logic are fully preserved.
- • 3 (Fair): Noticeable errors that do not fundamentally undermine the causal chain or primary objective.
- • 2 (Poor): Significant errors that violate causal narrative or visual consistency.
- • 1 (Very Poor): Fundamentally flawed, bearing little resemblance to a correct outcome.
- • 0 (Failure): Irrelevant, nonsensical, or non-compliant output.

To emphasize the importance of causal reasoning, the final Envision (Overall) Score is calculated as a weighted sum of the primary dimension scores: 40% for Physicality, 40% for Consistency, and 20% for Aesthetics. This formulation ensures that a model’s ability to reason about event processes is the dominant factor in its final evaluation.

#### F VLM Score Reliability Validation

The Envision-Score metric, designed to evaluate T2I models and UMMs on their ability to generate temporally consistent, physically plausible, and aesthetically coherent multi-image sequences, requires rigorous validation to ensure its reliability. This section outlines the validation process used to assess

the consistency and robustness of the scores produced by the scoring framework, as well as the mechanisms that underpin the trustworthiness of these evaluations.

###### F.1 Multi-Trial Evaluation

To ensure statistical validity, we conducted K independent evaluation trials for each generated sequence, maintaining consistent conditions across all trials. For each sequence, comprising four steps image in a event progression, the scoring process was repeated multiple times (K=5), providing a comprehensive data set that allows for a detailed reliability analysis. (see the Figure 9m)

Each trial resulted in an evaluation tuple Tk = (sk,rk), where sk ∈ [0,5]n represents the fine-grained score vector across all nine sub-dimensions, and rk denotes the textual rationale provided by the evaluator model, explaining the score assignment for each sub-dimension.

This multi-trial method ensures that the scoring process is robust to any inconsistencies that might arise from random variations in evaluation conditions, model interpretation, or inherent ambiguities in the generated sequences.

###### F.2 Quantitative Stability Analysis

For each sub-dimension score, the upper and lower bounds smaxd,i , smind,i and standard deviation σsd,i were computed across the K trials. This step provides a quantitative measure of the stability and variance of the scores, reflecting the reliability of the evaluation process. The lower the standard deviation, the more consistent the model’s performance is, indicating a high level of reliability in the scoring framework.

The standard deviation as:

σsd,i =

K

1 K

∑

(sk,d,i − µsd,i)2 (5)

k=1

The upper and lower bounds are defined as:

smaxd,i = max(sk,d,i : k = 1, . . . , K) (6) smind,i = min(sk,d,i : k = 1, . . . , K) (7)

By analyzing the distribution of scores across multi-trials, we can assess whether the scoring metric reliably reflects the quality of model outputs or whether variability in evaluation might be due to inconsistencies in the generated sequences or the evaluation model itself.

###### F.3 Qualitative Analysis for Evaluation

To complement the quantitative analysis, a qualitative consistency analysis was conducted with the involvement of five domain experts (PhD-level evaluators) who assessed 50 GPT-4o-generated multiimage sequences per domain (as shown in Figure 9). This step provided a human-driven evaluation of the coherence of the generated sequences, focusing on how well they aligned with real-world causal processes and spatiotemporal dynamics.

These human judgments were compared to the scores assigned by GPT-4o, the automated evaluator. The results showed a strong agreement between human experts and the GPT-4o assessments, demonstrating that the model’s scoring methodology is not only reliable but also consistent with the expert-level judgment. Any discrepancies in scores were systematically reviewed, contributing to

refining the scoring algorithm and ensuring that the evaluations capture the true causal coherence of the sequences.

###### F.4 Selection of GPT-4o as Final Evaluator

Based on the results of the qualitative and quantitative analyses, GPT-4o was chosen as the final automated evaluator. This decision was driven by its superior alignment with expert judgments and its ability to produce semantically stable patterns across multiple evaluation dimensions. By minimizing random variance in scoring and providing consistent rationales for its scores, GPT-4o was confirmed to be the most reliable tool for evaluating the causal and physical consistency of the generated sequences.

- Table 4: Comparative Analysis of Text-to-Visual Generation Modalities: Requirements and Benchmarks.

Modality Core Requirements Evaluation Benchmarks T2I (Text-to-Image)

Single-frame Synthesis

- • PhysBench[9]
- • T2I-ReasonBench [35]
- • WISE [28]
- • T2I-CoReBench [24]
- • T2I-CompBench [16]
- • MME-Unify [44]
- • Uni-MMMU [51]

- • Visual Fidelity: High-resolution, artifact-free image generation, image aesthetic quality.
- • Semantic Consistency: Precise matching between prompts and visual content.
- • Compositional Integrity: Correct spatial relationships among multiple objects.
- • Physical Plausibility: Adherence to basic physical laws and commonsense.

Multi-Frame Causal Modeling

T2MI (Text-to-MultiImage)

###### • Envision

- • Cross-Frame Consistency: Stable entities and environments across sequential frames.
- • Causal Progression: Logically connected event transitions with temporal coherence.
- • State Transformation: Plausible evolution from initial to final states.
- • World Process Simulation: Internal modeling of dynamic causal mechanisms.

Continuous Temporal Dynamics

T2V (Text-to-Video)

- • VBench [17]
- • VBench++ [18]
- • VidCapBench [7]
- • T2VPhysBench [15]
- • T2VWorldBench [8]

- • Motion Fluidity: Smooth, continuous object movement and camera motion.
- • Temporal Coherence: Consistent storytelling over extended time durations.
- • Dynamic Realism: Physically accurate motion patterns and interactions.
- • Multi-Scale Consistency: Coherent content at both frame and sequence levels.

###### Scientific Case: Biology Flux-kontext

###### -max GPT-4o Qwen-Image Bagel

"prompts": [ {

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

"step": 1, "prompt": "This is the first key frame of a continuous four-part visual event sequence showing a whale fall. A

massive whale carcass has just settled on the dark, barren abyssal plain of the deep ocean floor. The water is clear, with minimal sediment disturbed.",

"explanation": "This establishes the initial state of a whale fall, a significant source of organic material in the

nutrient-poor deep sea. The absence of scavengers indicates the carcass has just arrived. " }, {

"step": 2, "prompt": "This is the second key frame of a continuous four-part visual event sequence showing a whale

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

fall. From the same fixed viewpoint, the massive whale carcass is now being consumed by a swarm of mobile scavengers. Large sleeper sharks, hagfish, and crabs tear at the flesh, initiating the first stage of decomposition.",

"explanation": "This frame depicts the 'mobile scavenger' stage of a whale fall. Large, highly mobile deep-sea fauna are attracted to the carcass and rapidly consume the soft tissues. This is the first major step in transferring the whale's biomass to the deep-sea ecosystem."

}, {

"step": 3, "prompt": "This is the third key frame of a continuous four-part visual event sequence showing a whale fall.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

The whale skeleton is now mostly bare, with small patches of remaining tissue. A diverse community of smaller organisms, like polychaete worms and crustaceans, has colonized the bones and surrounding sediment.",

"explanation": "This 'enrichment opportunist' stage follows the removal of most soft tissue. Smaller, less mobile species colonize the bones and enriched sediment, continuing the nutrient transfer. This stage can last for months to years. The camera perspective highlights the shift in community structure from large mobile scavengers to smaller, stationary organisms."

}, {

"step": 4, "prompt": "This is the final key frame of a continuous four-part visual event sequence showing a whale fall.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

The last remnants of bone are being consumed by bone-eating Osedax worms and chemosynthetic bacteria. The surrounding sediment shows a lingering patch of unique life, but the main resource is nearly gone.",

"explanation": "Chemosynthetic bacteria metabolize sulfides from the decaying bones, supporting a unique community including Osedax worms that bore into the bone. This final stage completes the nutrient cycle, recycling the whale's biomass back into the deep-sea ecosystem."

} ]

###### (a) A four-stage visual narrative of a whale fall event in the deep ocean.

###### Cultural Case: History Flux-kontext

###### -max GPT-4o Qwen-Image Bagel

"prompts": [ {

"step": 1, "prompt": "This is the first key frame of a continuous four-part visual event sequence showing a quiet

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

American garage in the 1970s, with two young men surrounded by circuit boards, a soldering iron, and handwritten blueprints on a cluttered workbench.",

"explanation": "This frame depicts the early days of the personal computing revolution, showing Steve Jobs and Steve Wozniak in their garage workspace before Apple Computer was founded. The scene captures the causal trigger: passionate innovation and experimentation by hobbyists."

}, {

"step": 2, "prompt": "This is the second key frame of a continuous four-part visual event sequence showing the garage

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

workspace now busier, with early Apple I components assembled on a cluttered table as Jobs and Wozniak engage with prototype boards and wiring.",

"explanation": "This frame illustrates the early transformation driven by the causal trigger—intense innovation. However, now the workspace is more active, displaying tangible progress such as early Apple I hardware. Jobs and Wozniak are visibly engaged in hands-on electronics work, signaling the shift from experimentation to product development."

}, {

[Figure 175]

[Figure 176]

"step": 3, "prompt": "This is the third key frame of a continuous four-part visual event sequence showing the same garage, now bustling with intensified activity, with multiple assembled Apple I computers, packaging materials, and a growing stack of circuit boards as Jobs and Wozniak coordinate production and prepare for distribution.",

[Figure 177]

[Figure 178]

"explanation": "This frame captures the scaling phase of Apple’s origin story. The garage remains identical in layout and lighting to prior frames, reinforcing spatial continuity. The scene now shows Jobs and Wozniak transitioning from prototyping to small-scale manufacturing, a direct result of early technical success and market interest." }, {

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

"step": 4, "prompt": "This is the final key frame of a continuous four-part visual event sequence showing the same

suburban garage now abandoned, with a commemorative plaque outside and modern visitors taking photos, marking the completed transformation from startup to historical landmark.",

"explanation": "This frame reflects the long-term outcome of Apple’s founding, where the original garage—

unchanged in structure, lighting, and camera angle—is now a memorial site. The visitors and plaque represent recognition of its historic role, directly resulting from the earlier innovations and global impact initiated there."

} ]

(b) A four-stage historical narrative showing the evolution of Apple Computer’s founding in a 1970s garage.

Figure 8: Exemplar four-stage visual narratives from the Envision benchmark.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

(a) AQ (b) Auth (c) Exp (d) AS

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

(e) SC (f) STC (g) FC (h) CS

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

(i) BP (j) DI (k) PR (l) PS

[Figure 195]

(m) Overall

- Figure 9: Comprehensive evaluation of model performance across multiple dimensions: (a-d) aesthetic metrics (AQ, Auth, Exp, AS); (e-h) consistency metrics (SC, STC, FC, CS); (i-l) physicality metrics (BP, DI, PR, PS); (m) overall performance score.

- Table 5: Comprehensive Performance Comparison of Text-to-Image Generation Models across Multiple Evaluation Dimensions for Envision’s Physics Evaluation.

Model

Evaluation Dimensions SC FC STC CS Exp AQ Auth AS PR BP DI PS Open-Source T2I Models

FLUX-dev 35.29 37.65 24.71 32.47 60.00 71.76 43.53 58.28 24.71 42.35 30.59 32.47 FLUX-pro-1.1 38.00 36.00 26.00 33.26 64.00 74.00 44.00 60.50 28.00 42.00 36.00 35.26 FLUX-pro-1.1-ultra 36.92 36.92 26.15 33.26 61.54 70.77 43.08 58.31 30.77 44.62 35.38 36.86 FLUX-kontext-pro 41.67 43.33 35.00 39.95 50.83 64.17 45.83 53.53 36.67 52.50 39.17 42.72 FLUX-kontext-max 43.33 45.00 30.00 39.35 55.00 70.00 48.33 57.68 33.33 46.67 36.67 38.83 SD-3.5-flash 44.72 36.43 23.57 31.82 54.29 65.71 42.14 53.93 25.00 37.86 27.86 30.19 SD-3.5-medium 32.00 36.00 25.33 31.05 57.33 69.33 41.33 55.85 28.00 40.00 32.00 33.28 SD-3.5-large 36.47 38.82 22.35 32.45 52.94 62.35 43.53 52.85 24.71 41.18 28.24 31.31 Closed-Source T2I Models

GPT-4o 60.07 63.11 49.32 57.42 64.32 75.74 60.61 66.83 51.89 66.08 51.15 56.33 Gemini-2.5-Flash-Image 58.87 63.05 47.87 56.51 64.11 69.22 56.03 63.05 51.28 65.60 50.07 55.61 Unified Multimodal Models

Seedream 4.0 51.90 52.39 41.52 48.53 63.67 69.48 49.13 60.64 44.29 56.75 45.40 48.77 Qwen-Image 37.97 41.89 38.51 39.45 68.92 81.62 54.59 68.24 41.08 56.35 41.89 46.39 Hunyuan Image 3.0 35.00 36.25 26.25 32.44 63.75 72.50 42.50 59.41 26.25 38.75 32.50 32.44 Bagel 46.67 46.67 26.67 39.87 46.67 66.67 40.00 51.00 26.67 46.67 26.67 33.27 Janus-Pro-7B 34.93 36.67 25.60 32.33 53.20 62.27 38.93 51.34 27.87 38.93 31.07 32.57

Note: The table compares model performance across thirteen evaluation dimensions for Envision’s Physics evaluation. Models

are categorized into Open-Source Text-to-Image Models (blue background), Closed-Source T2I Models (yellow background), and Unified Multimodal Models (red background).

- Table 6: Comprehensive Performance Comparison of Text-to-Image Generation Models across Multiple Evaluation Dimensions for Envision’s Biology Evaluation.

Evaluation Dimensions SC FC STC CS Exp AQ Auth AS PR BP DI PS Open-Source T2I Models

Model

FLUX-dev 59.26 64.44 44.44 55.93 71.85 76.30 57.78 68.53 50.37 60.74 46.67 52.57 FLUX-pro-1.1 57.33 65.33 42.00 54.76 72.67 76.00 54.67 67.65 51.33 58.00 46.00 51.77 FLUX-pro-1.1-ultra 57.60 66.40 45.60 56.42 68.80 74.40 52.80 65.21 48.80 61.60 46.40 52.23 FLUX-kontext-pro 62.76 70.34 53.79 62.21 66.21 69.66 60.69 65.47 57.24 68.28 49.66 58.38 FLUX-kontext-max 65.33 76.00 52.00 64.32 70.67 72.00 64.00 68.84 58.67 66.67 50.67 58.67 SD-3.5-flash 51.89 57.84 44.32 51.28 65.95 76.76 56.22 66.21 49.73 56.76 43.24 49.91 SD-3.5-medium 50.43 60.87 40.00 50.33 59.13 66.09 54.78 59.95 45.22 58.26 42.61 48.66 SD-3.5-large 50.00 57.78 41.11 49.54 63.33 63.33 47.78 58.04 45.56 53.33 42.22 47.02 Closed-Source T2I Models

GPT-4o 83.01 85.55 73.98 80.78 82.88 83.68 71.64 79.32 77.46 83.68 66.49 75.89 Gemini-2.5-Flash-Image 71.99 77.04 57.62 68.77 77.69 77.62 60.58 71.85 62.45 73.65 57.04 64.36 Unified Multimodal Models

Seedream 4.0 81.81 82.68 73.54 79.29 82.83 82.44 67.95 77.64 76.22 81.42 64.72 74.14 Qwen-Image 78.53 78.60 70.93 75.97 89.93 90.27 69.13 82.97 74.27 79.67 66.67 73.54 Hunyuan Image 3.0 50.00 56.67 40.00 48.80 70.00 73.33 46.67 63.17 43.33 56.67 43.33 47.73 Bagel 60.00 65.00 55.00 59.95 65.00 65.00 45.00 58.20 55.00 65.00 45.00 55.00 Janus-Pro-7B 58.82 64.71 42.94 55.36 65.88 65.88 48.24 59.88 47.65 55.88 44.71 49.39

Note: The table compares model performance across thirteen evaluation dimensions for Envision’s Biology evaluation. Models are categorized into Open-Source Text-to-Image Models (blue background), Closed-Source T2I Models (yellow background), and Unified Multimodal Models (red background).

- Table 7: Comprehensive Performance Comparison of Text-to-Image Generation Models across Multiple Evaluation Dimensions for Envision’s Chemistry Evaluation.

Model

Evaluation Dimensions SC FC STC CS Exp AQ Auth AS PR BP DI PS Open-Source T2I Models

FLUX-dev 62.96 62.22 46.67 57.18 74.81 78.52 56.30 69.74 48.89 63.70 52.59 55.00 FLUX-pro-1.1 61.76 61.18 44.71 55.77 77.06 80.59 54.12 70.42 50.59 61.76 53.53 55.25 FLUX-pro-1.1-ultra 56.43 53.57 43.57 51.11 70.71 75.71 52.14 66.05 47.86 60.71 52.14 53.51 FLUX-kontext-pro 58.57 60.00 53.81 57.42 66.67 73.81 59.52 66.60 59.05 73.33 58.57 63.60 FLUX-kontext-max 65.00 65.00 48.33 59.33 68.33 71.67 55.00 64.90 48.33 66.67 50.00 54.93 SD-3.5-flash 43.33 34.29 30.00 32.36 54.29 72.86 42.86 56.53 35.71 48.57 37.14 40.43 SD-3.5-medium 50.43 33.33 26.67 29.97 63.33 80.00 43.33 62.03 36.67 46.67 43.33 42.17 SD-3.5-large 42.00 40.00 24.00 31.92 64.00 76.00 40.00 59.80 40.00 48.00 44.00 43.96 Closed-Source T2I Models

GPT-4o 69.83 74.61 58.17 67.44 64.70 73.83 67.30 68.60 61.22 73.30 59.39 64.60 Gemini-2.5-Flash-Image 67.95 71.69 54.25 64.52 63.47 67.03 60.00 63.47 56.99 71.32 54.70 60.96 Unified Multimodal Models

Seedream 4.0 62.87 62.67 49.13 58.13 61.03 69.33 51.90 60.66 50.87 64.51 48.82 54.70 Qwen-Image 57.21 53.49 47.91 52.82 75.35 80.00 50.70 68.50 49.30 60.93 50.23 53.45 Hunyuan Image 3.0 49.74 48.95 39.21 45.90 66.32 71.32 45.53 60.90 42.63 53.42 48.16 48.02 Bagel 65.00 60.00 45.00 56.55 70.00 70.00 55.00 64.90 50.00 60.00 45.00 51.65 Janus-Pro-7B 43.17 43.49 35.87 40.80 58.10 59.68 40.32 52.57 38.41 47.62 43.49 43.13

Note: The table compares model performance across thirteen evaluation dimensions for Envision’s Chemistry evaluation. Models are categorized into Open-Source Text-to-Image Models (blue background), Closed-Source T2I Models (yellow background), and Unified Multimodal Models (red background).

- Table 8: Comprehensive Performance Comparison of Text-to-Image Generation Models across Multiple Evaluation Dimensions for Envision’s Geography Evaluation.

Evaluation Dimensions SC FC STC CS Exp AQ Auth AS PR BP DI PS Open-Source T2I Models

Model

FLUX-dev 57.69 61.54 46.15 55.04 72.31 78.46 53.85 69.06 49.23 63.85 49.23 54.05 FLUX-pro-1.1 55.14 55.68 42.70 51.09 71.89 78.38 54.05 67.97 47.03 58.92 45.95 50.59 FLUX-pro-1.1-ultra 53.33 59.17 44.17 52.14 67.50 73.33 55.00 65.17 47.50 60.00 47.50 51.62 FLUX-kontext-pro 54.74 60.00 45.79 53.43 64.21 69.47 55.26 62.91 47.89 63.16 46.84 52.58 FLUX-kontext-max 65.71 67.37 50.53 59.91 70.53 73.68 57.89 67.27 54.74 70.53 51.58 58.91 SD-3.5-flash 50.00 50.77 41.54 47.38 68.46 75.38 50.00 64.47 44.62 53.85 43.08 47.15 SD-3.5-medium 61.43 67.14 47.14 58.46 61.43 65.71 55.71 60.90 51.43 65.71 47.14 54.73 SD-3.5-large 52.31 60.00 38.46 50.14 66.15 69.23 50.77 61.94 41.54 58.46 40.00 46.62 Closed-Source T2I Models

GPT-4o 82.33 84.33 75.00 80.50 79.00 79.73 73.53 77.38 76.60 84.47 69.20 76.75 Gemini-2.5-Flash-Image 78.16 83.33 69.79 77.02 76.81 77.73 74.11 76.20 72.48 82.13 65.25 73.28 Unified Multimodal Models

Seedream 4.0 70.60 71.40 60.70 67.50 74.40 74.70 59.80 69.54 61.60 70.10 57.10 62.92 Qwen-Image 65.79 66.32 54.74 62.21 80.00 81.58 60.53 73.90 56.84 68.95 55.26 60.32 Hunyuan Image 3.0 74.86 77.43 62.57 71.53 76.57 78.29 66.57 73.74 66.29 76.00 61.14 67.79 Bagel 46.67 46.67 46.67 46.67 66.67 66.67 46.67 59.87 46.67 60.00 46.67 51.07 Janus-Pro-7B 56.19 62.86 48.57 55.80 63.81 65.71 49.52 59.58 49.52 59.05 47.62 52.04

Note: The table compares model performance across thirteen evaluation dimensions for Envision’s Geography evaluation. Models are categorized into Open-Source Text-to-Image Models (blue background), Closed-Source T2I Models (yellow background), and Unified Multimodal Models (red background).

- Table 9: Comprehensive Performance Comparison of Text-to-Image Generation Models across Multiple Evaluation Dimensions for Envision’s Meteorology Evaluation.

Model

Evaluation Dimensions SC FC STC CS Exp AQ Auth AS PR BP DI PS Open-Source T2I Models

FLUX-dev 55.38 63.08 46.15 54.78 77.69 80.77 59.23 72.43 51.54 66.15 50.00 55.85 FLUX-pro-1.1 55.56 60.00 42.78 52.68 76.67 80.56 58.89 71.91 51.11 65.56 52.22 56.24 FLUX-pro-1.1-ultra 48.78 56.10 41.46 48.71 70.24 72.68 51.22 64.58 46.83 61.95 46.83 51.82 FLUX-kontext-pro 56.67 62.92 48.33 55.90 69.58 72.08 55.83 65.73 53.75 67.08 50.83 57.19 FLUX-kontext-max 60.00 68.33 51.67 59.92 71.67 76.67 58.33 68.78 60.00 70.00 55.00 61.65 SD-3.5-flash 44.72 48.33 38.33 43.74 68.06 73.33 49.72 63.56 42.78 57.22 41.94 47.27 SD-3.5-medium 54.00 60.00 44.00 52.58 62.00 62.00 54.00 59.28 46.00 64.00 46.00 51.94 SD-3.5-large 55.38 61.54 44.62 53.75 69.23 69.23 50.77 62.95 49.23 64.62 46.15 53.29 Closed-Source T2I Models

GPT-4o 80.67 82.60 72.27 78.45 84.13 86.00 76.67 82.21 74.47 83.27 73.80 77.15 Gemini-2.5-Flash-Image 71.48 74.58 60.49 68.77 78.66 80.99 68.10 75.84 63.87 75.63 63.52 67.64 Unified Multimodal Models

Seedream 4.0 68.16 68.10 58.44 64.83 83.27 82.52 61.63 75.66 61.29 73.13 62.72 65.67 Qwen-Image 56.28 57.21 48.84 54.06 81.86 83.72 56.28 73.78 50.23 65.58 53.49 56.37 Hunyuan Image 3.0 66.15 70.77 56.92 64.54 83.08 84.62 64.62 77.31 60.00 76.92 61.54 66.09 Bagel 53.33 53.33 60.00 55.60 66.67 66.67 46.67 59.87 60.00 66.67 53.33 60.00 Janus-Pro-7B 62.50 70.00 53.75 62.00 72.50 73.75 55.00 66.96 58.75 68.75 56.25 61.22

Note: The table compares model performance across thirteen evaluation dimensions for Envision’s Meteorology evaluation. Models are categorized into Open-Source Text-to-Image Models (blue background), Closed-Source T2I Models (yellow background), and Unified Multimodal Models (red background).

- Table 10: Comprehensive Performance Comparison of Text-to-Image Generation Models across Multiple Evaluation Dimensions for Envision’s Culture & History Evaluation.

Evaluation Dimensions SC FC STC CS Exp AQ Auth AS PR BP DI PS Open-Source T2I Models

Model

FLUX-dev 54.07 48.89 37.78 46.82 69.63 68.15 47.41 61.59 46.67 57.04 45.93 49.84 FLUX-pro-1.1 63.43 57.14 45.14 55.14 75.43 70.86 50.29 65.37 53.71 62.86 52.00 56.17 FLUX-pro-1.1-ultra 59.17 55.00 43.33 52.41 73.33 69.17 47.50 63.17 47.50 60.83 46.67 51.62 FLUX-kontext-pro 68.11 65.41 54.05 62.44 70.27 71.35 59.46 66.95 61.62 71.89 54.59 62.69 FLUX-kontext-max 65.71 59.05 43.81 56.07 74.29 67.62 51.43 64.31 52.38 65.71 50.48 56.15 SD-3.5-flash 54.05 48.65 41.62 48.04 66.49 66.49 47.57 60.05 48.65 58.92 45.95 51.15 SD-3.5-medium 50.43 48.00 30.67 42.11 66.67 64.00 44.00 58.08 42.67 54.67 42.67 46.63 SD-3.5-large 48.75 45.00 32.50 41.99 62.50 63.75 42.50 56.11 45.00 56.25 43.75 48.30 Closed-Source T2I Models

GPT-4o 78.67 81.68 75.79 78.68 87.30 89.26 81.12 85.85 81.75 88.63 78.60 82.98 Gemini-2.5-Flash-Image 66.25 69.50 62.24 65.96 79.00 81.31 65.79 75.27 69.03 78.38 66.41 71.25 Unified Multimodal Models

Seedream 4.0 61.59 63.43 57.41 60.77 76.07 78.24 61.26 71.75 66.03 73.05 62.68 67.24 Qwen-Image 71.18 61.76 58.24 63.67 81.18 78.24 57.06 72.01 64.12 71.18 60.59 65.28 Hunyuan Image 3.0 68.75 58.75 51.25 59.50 78.75 76.25 55.00 69.85 58.75 66.25 57.50 60.81 Bagel 80.00 73.33 73.33 75.53 80.00 80.00 60.00 73.20 73.33 73.33 60.00 68.93 Janus-Pro-7B 58.14 52.56 42.33 50.92 66.05 61.86 43.72 57.07 44.19 51.63 44.65 46.80

Note: The table compares model performance across thirteen evaluation dimensions for Envision’s Culture & History evaluation. Models are categorized into Open-Source Text-to-Image Models (blue background), Closed-Source T2I Models (yellow background), and Unified Multimodal Models (red background).

###### SCORING CRITERIA -- CONSISTENCY DIMENSION

Semantic Consistency (0-5): Alignment between visual representation and conceptual meaning throughout the sequence, considering both explicit and implicit elements.

- 0 (Failure): Visual representation completely contradicts the core concept; the subject matter is wrong or misrepresented; the intended meaning is entirely lost or reversed; major misunderstanding of the explanation.
- 1 (Very Poor): Core theme is present but significantly distorted; key elements are misinterpreted or omitted; contradictory elements disrupt understanding; the overall narrative is unclear or misleading.
- 2 (Poor): Basic concept is recognizable but flawed in key details; the main idea is conveyed but with significant errors; some elements align with the intended meaning while others are incorrectly represented.
- 3 (Fair): Core concept is accurately captured, and the meaning is mostly clear; most elements align with the intended idea, but some minor semantic inconsistencies or unclear representations persist.
- 4 (Good): Both explicit and implicit meanings are captured well; a nuanced understanding of the concept is conveyed, and even subtle or abstract aspects are represented accurately; rich semantic content with few inconsistencies.
- 5 (Excellent): Every conceptual nuance is perfectly expressed; the visual representation fully aligns with the intended meaning, conveying deep understanding of all semantic layers; flawless conceptual execution with no distortions.

Spatial-Temporal Consistency (0-5): Coherence in spatial evolution and temporal progression across the sequence, considering both short-term and long-term changes in the context of a causal narrative.

- 0 (Failure): Spatial and temporal evolution is entirely incoherent; there are no logical relationships between the states; abrupt, random shifts with no narrative consistency; objects or environments teleport without any narrative justification.
- 1 (Very Poor): Significant spatial and temporal inconsistencies that hinder narrative flow; poor understanding of progression and transitions; movement and positioning are illogical, with abrupt and unmotivated changes that disrupt continuity.
- 2 (Poor): Some logical progression is present, but spatial evolution and temporal transitions are often unclear or inconsistent; movements are somewhat plausible but lack cohesion in complex scenarios or fail to respect causal logic.
- 3 (Fair): Spatial and temporal transitions are mostly logical and plausible, with smooth progression overall; occasional disruptions or inconsistencies in movement or timing, but the general evolution remains coherent and the causal narrative holds.
- 4 (Good): Complex spatial evolution and temporal changes are well-executed; transitions are smooth and consistent; sophisticated understanding of causal relationships is applied with few minor inconsistencies that do not disrupt the overall flow.
- 5 (Excellent): Spatial and temporal evolution is flawlessly consistent with the narrative's causal logic; all changes in position, movement, and timing align with the story's progression and adhere to both immediate and long-term narrative consistency; impeccable execution of spatial-temporal relationships throughout the sequence.

Factual Consistency (0-5): Adherence to empirical facts, scientific knowledge, and logical relationships, ensuring the representation is grounded in reality and subject to coherent reasoning.

- 0 (Failure): Violates fundamental physical laws or scientific principles; contains logical contradictions; depicts completely impossible scenarios; misrepresents core historical or factual knowledge.
- 1 (Very Poor): Core factual relationships are incorrect or distorted; frequent contradictions with known facts or logical reasoning; significant inaccuracies in science, history, or other factual domains; implausible scenarios that undermine credibility.
- 2 (Poor): While most factual relationships are generally correct, there are notable inaccuracies or misrepresentations; the core facts are plausible but some details are wrong or incomplete; minor factual errors that don't severely affect the overall portrayal.
- 3 (Fair): Majority of factual elements are accurate, and logical relationships are well-maintained; only minor inaccuracies in specific domains or specialized knowledge; generally plausible depiction with slight factual discrepancies.
- 4 (Good): The depiction shows a high level of factual accuracy; complex relationships are correctly represented; specialized knowledge is precisely applied, with only minor technical inaccuracies or inconsistencies.
- 5 (Excellent): Complete empirical accuracy; there are no detectable factual errors or logical contradictions; every aspect aligns perfectly with the relevant knowledge domains, demonstrating exemplary research and execution.

- Figure 10: Scoring Criteria for the Consistency dimension, evaluating the spatiotemporal, semantic, and factual consistency of generated multi-image sequences. Each is scored on a 0–5 scale, where 0 indicates catastrophic failure and 5 represents flawless execution.

###### SCORING CRITERIA -- PHYSICALITY DIMENSION

Basic Properties (0-5): Accuracy in fundamental scene properties including object quantities, basic shapes, geometric relationships, and size proportions across the sequence.

- 0 (Failure): Wrong number of major objects; impossible geometric relationships; completely unrealistic proportions; no understanding of basic spatial organization.
- 1 (Very Poor): Significant errors in object counts (>30% wrong); major shape distortions; poor geometric understanding; clearly wrong size relationships.
- 2 (Poor): Approximate object counts generally correct within 20%; recognizable but imperfect shapes; basic geometric relationships mostly maintained; roughly plausible proportions.
- 3 (Fair): Object counts within 15% of intended; shapes accurately represented with minor flaws; good geometric consistency; generally correct size relationships.
- 4 (Good): Precise object quantities; well-defined shapes; sophisticated geometric relationships; excellent proportional accuracy; professional-level spatial organization.
- 5 (Excellent): Perfect object counts and distributions; mathematically precise shapes and forms; flawless geometric relationships; exact proportional accuracy throughout.

Dynamics and Interactivity (0-5): Realism in physical dynamics including motion trajectories, force interactions, fluid/rigid body behaviors, and object interactions across the sequence.

- 0 (Failure): Motion trajectories completely unrealistic; impossible force interactions; no understanding of dynamics; objects interact in physically impossible ways; fluid/rigid body behaviors completely wrong.
- 1 (Very Poor): Major motion trajectory errors; unrealistic force applications; poor understanding of dynamics; clearly wrong interaction patterns; fluid behaviors significantly distorted.
- 2 (Poor): Basic motion trajectories generally plausible; simple force interactions mostly correct; adequate but imperfect dynamics; generally believable interactions with noticeable flaws.
- 3 (Fair): Good motion trajectory realism; proper force applications; technically sound dynamics; convincing interaction patterns with minor flaws; fluid/rigid body behaviors mostly correct.
- 4 (Good): Sophisticated motion trajectories; complex force interactions handled well; professional-level dynamics; realistic and nuanced object interactions; accurate fluid/rigid body simulations.
- 5 (Excellent): Perfect motion physics throughout; flawless force interactions; exemplary dynamics in all aspects; indistinguishable from real physical interactions; professional-grade fluid/rigid body behaviors. Physical Reliability (0-5): Adherence to fundamental physical laws across the sequence.

- 0 (Failure): Spatial evolution is completely illogical or random; objects appear, disappear, or teleport without cause; no discernible progression; chaotic and physically impossible movements.
- 1 (Very Poor): Major inconsistencies in object movement or scene evolution; poor understanding of physical progression; frequent illogical transitions; severe continuity errors in position or timing.
- 2 (Poor): Basic spatial progression is generally logical but contains noticeable flaws; some object movements or transitions lack smoothness; plausible overall but with clear inconsistencies in complex actions.
- 3 (Fair): Spatial evolution is mostly logical and coherent; object movements show reasonable continuity; minor issues in timing or complex transitions; generally smooth but not fully polished.
- 4 (Good): Spatial progression is consistently logical and well-executed; complex movements and transitions are handled adeptly; strong temporal coherence with only subtle imperfections.
- 5 (Excellent): Spatial-temporal evolution is perfectly logical, fluid, and physically plausible; all movements and transitions demonstrate flawless continuity and natural progression; exemplary understanding of spatiotemporal dynamics.

- Figure 11: Scoring Criteria for the Physicality dimension, assessing the model’s adherence to physical laws and plausibility of dynamic processes. Scores range from 0 (failure) to 5 (excellent).

###### SCORING CRITERIA -- AESTHETIC DIMENSION

Expressiveness (0-5): The emotional impact and visual storytelling conveyed through the arrangement of visual elements, balance, and flow across the sequence.

- 0 (Failure): No emotional depth or clear visual narrative; elements lack cohesion or meaning; chaotic and incoherent design with no emotional connection.
- 1 (Very Poor): Minimal emotional expression; poorly communicated visual narrative; conflicting elements and arrangement that detract from the intended message.
- 2 (Poor): Simple visual arrangement with limited emotional expression; some effort at narrative is apparent, but the flow feels basic or incomplete; uneven balance in visual elements.
- 3 (Fair): Clear and effective emotional expression; visual flow supports the intended narrative; balanced elements that guide the viewer's attention with good use of space and composition.
- 4 (Good): Strong emotional engagement; dynamic arrangement that supports a compelling visual narrative; sophisticated design choices that enhance emotional tone and storytelling.
- 5 (Excellent): Exceptional emotional resonance; perfectly executed visual narrative that captures and amplifies the theme; masterful expressiveness in the arrangement of elements that tells a compelling story.

Aesthetic Quality (0-5): Overall aesthetic appeal, style coherence, and artistic merit across the sequence.

- 0 (Failure): No coherent style; elements clash severely; visually painful to view; complete lack of artistic consideration.
- 1 (Very Poor): Inappropriate style choices; poor visual appeal; unbalanced aesthetic elements; amateurish execution; distracting aesthetic flaws.
- 2 (Poor): Functional but unrefined aesthetic; adequate visual appeal; simple style application; limited artistic effectiveness.
- 3 (Fair): Coherent style with good aesthetic relationships; appropriate emotional tone; technically sound artistic execution; generally pleasing appearance.
- 4 (Good): Sophisticated artistic execution; strong emotional impact through style; complex aesthetic relationships handled well; professional artistic quality.
- 5 (Excellent): Perfect aesthetic harmony throughout; innovative and emotionally resonant style; flawless artistic execution; exemplary aesthetic storytelling; world-class artistic design.

Authenticity (0-5): The believability and naturalness of the elements, ensuring that the sequence aligns with the intended reality or narrative.

- 0 (Failure): The sequence looks completely artificial and lacks any attempt at realism; obvious digital artifacts; cartoonish appearance with no connection to reality.
- 1 (Very Poor): Major unrealistic elements dominate the sequence; obvious digital artifacts; unnatural lighting, textures, or distortions; the image feels clearly synthetic.
- 2 (Poor): Generally believable but with noticeable artificial elements; some aspects appear unnatural or inconsistent in realism, but the overall image is somewhat convincing.
- 3 (Fair): Good level of realism with only minor artificial elements; generally convincing depiction of reality; some minor issues in complex areas (lighting, texture, etc.).
- 4 (Good): Very believable image approaching photographic quality; subtle details enhance realism, and artificial elements are barely detectable.
- 5 (Excellent): Indistinguishable from real photography; perfect realism in all aspects; flawless material representation and expert-level rendering that conveys authenticity seamlessly.

- Figure 12: Scoring Criteria for the Aesthetics dimension, capturing the visual quality and narrative expressiveness of generated sequences. Scores range from 0 (failure) to 5 (excellent).

