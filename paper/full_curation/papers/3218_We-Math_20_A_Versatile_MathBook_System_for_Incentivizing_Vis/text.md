# arXiv:2508.10433v1[cs.AI]14Aug2025

## WE-MATH 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning

Runqi Qiao1∗†, Qiuna Tan1∗†, Peiqing Yang1, Yanzi Wang3, Xiaowan Wang1, Enhui Wan1, Sitong Zhou1, Guanting Dong1, Yuchen Zeng1, Yida Xu1, Jie Wang1, Chong Sun2, Chen Li2‡, Honggang Zhang1‡ 1BUPT 2WeChat Vision, Tencent Inc. 3Tsinghua University qrq@bupt.edu.cn qiunatan@bupt.edu.cn chaselli@tencent.com

https://we-math2.github.io/

### Abstract

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various tasks, but still struggle with complex mathematical reasoning. Existing research primarily focuses on dataset construction and method optimization, often overlooking two critical aspects: comprehensive knowledgedriven design and model-centric data space modeling. In this paper, we introduce WE-MATH 2.0, a unified system that integrates a structured mathematical knowledge system, model-centric data space modeling, and a reinforcement learning (RL)-based training paradigm to comprehensively enhance the mathematical reasoning abilities of MLLMs. The key contributions of We-Math 2.0 are fourfold: (1) MathBook Knowledge System: We construct a five-level hierarchical system encompassing 491 knowledge points and 1,819 fundamental principles. (2) MathBook-Standard & Pro: We develop MathBook-Standard, a dataset that ensures broad conceptual coverage and flexibility through dual expansion. Additionally, we define a three-dimensional difficulty space and generate 7 progressive variants per problem to build MathBook-Pro, a challenging dataset for robust training. (3) MathBook-RL: We propose a two-stage RL framework comprising: (i) Cold-Start Fine-tuning, which aligns the model with knowledge-oriented chain-of-thought reasoning; and (ii) Progressive Alignment RL, leveraging averagereward learning and dynamic data scheduling to achieve progressive alignment across difficulty levels. (4) MathBookEval: We introduce a comprehensive benchmark covering all 491 knowledge points with diverse reasoning step distributions. Experimental results show that MathBook-RL performs competitively with existing baselines on four widely-used benchmarks and achieves strong results on MathBookEval, suggesting promising generalization in mathematical reasoning.

### 1 Introduction

Large Language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks [1, 9, 22, 62, 54, 60, 24, 56, 53, 63, 25, 49, 69]. Building on this foundation, Multimodal Large Language Models (MLLMs) have shown impressive performance in visual question answering (VQA) [3, 74, 18, 41, 44], optical character recognition (OCR) [68, 67, 30, 61], and object detection [31, 28, 45]. However, MLLMs still face difficulties with complex reasoning tasks,

∗Equal contribution. †Work done as interns at WeChat, Tencent Inc. ‡Corresponding author. zhhg@bupt.edu.cn

Preprint. Under review.

- Table 1: Comparison between MathBook and several multimodal mathematical datasets. To achieve comprehensive coverage of mathematical knowledge and ensure that each data sample is retrievable with precise knowledge labels, all data are manually constructed using GeoGebra.

Data Annotation

Knowledge-level Annotation

Principle-level Annotation

Difficulty Levels

Dataset Usage Image Source

Geometry3K [33] Training Data & Benchmark Collection Manual - - MathV360K [47] Training Data Collection Collection - - 4 We-Math [43] Benchmark Manually Created Manual 67 - GeoSense [64] Benchmark Collection Manual 148 ✓ MathBook (Ours) Training Data & Benchmark Manually Created Manual 491 ✓ 8

particularly in visual mathematical problem-solving, where generalization remains a fundamental challeng [32, 71, 57].

Recent efforts to enhance mathematical reasoning in MLLMs have primarily focused on three directions: dataset construction [33, 72, 47, 59], preference optimization [75, 35], and reinforcement learning (RL) [21, 36]. Foundation approaches aggregated datasets from diverse mathematical domains [48]. Subsequent efforts introduce structured supervision formats (e.g., Chain-of-Thought (CoT)) combined with preference optimization to guide step-by-step reasoning [76]. More recently, RL-based studies with curriculum-based training have been employed to further improve model performance on complex reasoning tasks [21, 5, 65, 55]. Despite this progress, several fundamental challenges remain (in Table 1):

- (1) Lack of a comprehensive knowledge system: Existing MLLMs show uneven performance across different subfields of math reasoning. [32, 57] Unfortunately, current datasets suffer from limited coverage of knowledge points and domain diversity, underscoring the necessity of establishing a more systematic knowledge system.
- (2) Lack of model-centric difficulty modeling: Existing multimodal training datasets primarily perform difficulty annotation based on human learning stages [36]. However, recent studies [26, 42, 32] reveal that MLLMs do not exhibit learning patterns that align well with these human-defined levels. This highlights the need for a more model-centric approach to modeling data difficulty.
- (3) Lack of emphasis on reasoning generalization: MLLMs are capable of solving complex problems, but perform poorly on corresponding subproblems [42] as well as on similar, same-type tasks [77]. This underscores the current training methods’ focus on problem memorization rather than fostering reasoning generalization.

To address these limitations, we introduce WE-MATH 2.0, a versatile framework that combines a structured mathematical knowledge system, model-centric data space modeling, and a reinforcement learning-based training paradigm to comprehensively improve MLLM’s reasoning capabilities. In detail, we begin by establishing the MathBook Knowledge System, a five-level hierarchy comprising 491 knowledge points and 1,819 fundamental principles (see Figure 1). This structure is systematically derived from sources such as Wikipedia and open-source textbooks, refined through hierarchical clustering, and further revised by human experts.

Building on this foundation, we introduce MathBook-Standard, a dataset featuring comprehensive annotations at the level of 1,819 knowledge principles, along with carefully curated problems to ensure broad, balanced coverage, particularly in underrepresented mathematical domains. To foster deeper conceptual understanding, MathBook-Standard employs dual expansions: "multi-images per question" and "multi-questions per image", enabling diverse problem sets that achieve conceptual flexibility. Crucially, we propose a pivotal three-dimensional difficulty modeling framework that redefines mathematical problem construction. By explicitly modeling "step complexity", "visual complexity" and "contextual complexity", each problem is systematically expanded into seven difficulty levels to form MathBook-Pro. This design enables structured, progressive learning for MLLMs, laying a strong foundation for improved reasoning across difficulty levels.

Notably, all images in our training data are meticulously handcrafted using GeoGebra software4. This approach ensures that our images are not only newly created and precise, but also demonstrate a

4https://www.geogebra.org/

level of rigor and complexity in image construction that surpasses commonly used Python-based rendering methods.

To further improve the general mathematical reasoning capabilities of MLLMs, we propose MathBook-RL, a two-stage reinforcement learning framework for progressive and robust mathematical reasoning:

(1) Cold-Start Fine-tuning: We first adopt a supervised fine-tuning that guides the MLLM to learn knowledge-oriented CoT reasoning, internalizing it to acquire conceptual understanding and structured problem-solving paradigms. (2) Progressive Alignment RL: We propose a curriculumbased RL paradigm. Leveraging the "one-question-multi-image" and knowledge-point features in MathBook-Standard, we first align the model’s analogical reasoning by introducing an average reward mechanism. Building on this foundation, we progressively train the MLLM on MathBook-Pro and further introduce two dynamic scheduling strategies: i) Knowledge Increment Scheduling: When errors occur due to complex reasoning steps, the model is adaptively redirected to relevant incremental-step samples in MathBook-Standard. ii) Modality Increment Scheduling: When errors stem from increased modality complexity, the model is guided through single-modality incremental problems. This targeted curriculum enables effective knowledge transfer across difficulty levels.

To comprehensively evaluate MLLM’s reasoning capability, we introduce MathBookEval, a benchmark covering all 491 knowledge points with diverse step distributions. Experimental results show that MathBook-RL performs competitively with existing baselines on four widely used benchmarks, showing substantial improvements in generalization and robustness. In summary, our main contributions are as follows:

- • We propose the MathBook Knowledge System, a five-level hierarchical framework with 491 knowledge points and 1,819 fundamental principles, enabling systematic and comprehensive mathematical knowledge supervision.
- • We develop MathBook-Standard and MathBook-Pro, two novel datasets that combine comprehensive step-wise annotation, dual expansions for conceptual flexibility, and a principled three-dimensional difficulty modeling framework for structured and progressive learning.
- • We introduce MathBook-RL, a two-stage RL framework that integrates structured knowledge supervision and dynamic data scheduling, improving the reasoning capabilities of MLLMs.
- • We present MathBookEval, a benchmark designed to comprehensively evaluate model reasoning across diverse knowledge points and step distributions. Extensive experiments demonstrate that our approach achieves remarkable performance in both generalization and robustness.
- • All images in our training set are manually and meticulously crafted using GeoGebra software, ensuring they are newly created, precise, and surpass common Python-based rendering methods in spatial geometric rigor and complexity.

### 2 Related Work

Visual Mathematical Reasoning. Recently, research on visual mathematical reasoning has made notable progress [47, 72, 75, 19, 35, 13]. For benchmarks, MathVista [32] and MathVision [57] evaluate overall model performance, while MathVerse [71], We-Math [42], and Dynamath [77] focus on analyzing reasoning mechanisms and offer directions for optimization. Specifically, MathVerse finds that models benefit from richer textual information. Dynamath investigates model robustness by evaluating performance on near-duplicate questions.

Methodologically, substantial efforts have been devoted to improving the alignment between visual and textual modalities, as demonstrated by Math-LLaVA [47], MAVIS [72] and MathCoder-VL [59]. In addition, Math-PUMA [75] and URSA [35] introduce step-wise reasoning processes to better mimic human problem solving. Furthermore, with the growing interest in RL-based approaches, a series of recent works explore reward optimization as a means to further enhance visual reasoning abilities [21, 70, 36, 5, 29, 2, 55, 73, 65, 50, 20, 52]. Notably, these RL-based methods have shown promising improvements by explicitly optimizing model behavior for complex reasoning tasks.

Building upon these encouraging results, we contribute an alternative perspective by developing a systematic, model-centric knowledge framework, combining it with RL-based alignment training,

and releasing a new dataset. We hope these efforts will provide the community with new insights and resources to support ongoing advancements in visual mathematical reasoning.

### 3 WE-MATH 2.0

Overview. In this section, we introduce WE-MATH 2.0, a unified system designed to advance visual mathematical reasoning in MLLMs. WE-MATH 2.0 is developed from three key aspects:

- • We construct a five-level MathBook Knowledge System (§3.1), systematically organizing 491 knowledge points and 1,819 fundamental principles for comprehensive mathematical supervision.
- • We propose a Multi-Dimensional data construction pipeline (§3.2), incorporating seed problem construction, variant expansion, and principled three-dimensional difficulty modeling.
- • We introduce MathBookEval (§3.3), a benchmark aligned with our knowledge system for systematic evaluation.

##### 3.1 MathBook Knowledge System

System Overview. We construct a five-level hierarchical MathBook Knowledge System organized by the “Definition–Theorem–Application” paradigm [15]. The core is a set of knowledge points K = {k1,k2,...,kN}, N = 491, spanning primary to university mathematics. Each ki is associated with a set of fundamental principles Pi = {pi1,...,pim

i}, mi ∈ [1,7], where P = Ni=1 Pi and |P| = 1,819. These principles systematically cover definitions, theorems, and applications. (Concrete examples of knowledge points and fundamental principles can be found on our project page.)

Hierarchical Construction via Human-AI Collaboration. We construct K through a hybrid process. Human experts first design an initial structure Khuman based on authoritative sources, including textbooks, Wikipedia, and national curriculum standards. In parallel, we sample 30K problems from the existing math dataset [34, 23, 16, 47, 39, 35, 75, 72], merging them into a unified dataset. We then use GPT-4o [37] to assign multi-level topic tags T = {t1,...,tn}, followed by hierarchical clustering on the semantic similarity matrix S ∈ Rn×n to obtain an AI-generated structure Kauto. The final knowledge point set K is produced by expert-guided integration of Khuman and Kauto, with independent review for quality assurance.

Fine-Grained Principle Annotation. Given the constructed K, we employ GPT-4o to annotate the step-level knowledge points for each problem qj ∈ Q = {q1,...,qM} by mapping each step in its chain-of-thought solution to the corresponding ki ∈ K. This yields a mapping M1 : qj  → (ki

,...), forming a set of step-level solution paths for each knowledge point. Next, for each ki, GPT-4o summarizes the set of theorems and principles used across all associated solution paths, resulting in a mapping M2 : ki  → {pi1,...,pim

,ki

1

2

i}. Finally, these AI-extracted principles are consolidated and cross-checked with those written by human experts, with iterative refinement to ensure completeness and accuracy of P. Detailed guideline of our system are listed in Appendix.

##### 3.2 Multi-Dimensional Data Construction

In this section, we introduce the data construction pipeline of our versatile datasets: MathBookStandard and MathBook-Pro.

- 3.2.1 MathBook-Standard: Seed and Variant Problem Construction Seed Problem Construction.

To ensure rich coverage and high-quality design, we construct problems based on the knowledge system following 3 guidelines:

- (1) All diagrams are rendered with GeoGebra for precise geometric representation;
- (2) Problems focus on math essence, avoiding reliance on superficial visual cues;
- (3) Each problem strictly corresponds to its designated principle set Pi.

To achieve these, we adopt a “model-assisted, expert-led” workflow. Given a knowledge point ki ∈ K and its associated principle set Pi, an LLM first generates a draft problem, including the question,

###### We-Math 2.0

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Examples

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

|FundamentalSkills Numbers andQuantities Basic Calculation Methods<br><br>[Figure 17]<br><br>[Figure 18]<br><br>· Definition of integers · Definition of fractions · Understanding the number axis…<br><br>· Four operations on complex numbers · Conversion between exponential and logarithmic expressions · Definition of square…<br><br>· Observe natural images · Measure · Count · Find patterns…<br><br>- Plane shape recognition: Identify basic geometric shapes from natural or architectural images.<br>- Angles and proportions: Measure or estimate angles, length proportions…<br><br><br>- A fraction is composed of a numerator, a denominator, and a fraction line, representing a part of a whole<br>- Basic properties of fractions: The value of a numerator and denominator remains unchanged when multiplied together unless the number is zero. a÷b=a/b (b≠0)…<br><br><br>- Integer addition definition: the operation of combining two numbers into one number, addend + addend = sum, knowing the sum and one addend, we can find the other addend<br>- Integer subtraction definition: the operation of finding the other addend when the sum of two addends and one of the addends is known, minuend - subtrahend = difference, subtraction and addition are inverse operations…<br><br><br>Knowledge<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]|
|---|

###### Probability

[Figure 22]

Statistics

[Figure 23]

· Definition of random events · Relationships between random events · Calculation of random events · Definition of random variables · Definition of distribution functions ·Calculation of geometric probability models · One-dimensional continuous variables and their distribution…

- - The formula for calculating the number of combinations:

C𝒏𝒎 = 𝒎!(𝒏 𝒎)!𝒏!

- - The properties of the number of combinations: symmetry

·Differences between different sampling methods · Choose appropriate survey and sampling methods · Classification addition counting principle · Step-by-step multiplication counting principle · Calculation of the number of combinations…

- - Core features: infinite possible results (such as a dart landing anywhere on a target), uniform distribution of results
- - One-dimensional geometric probability: use length ratio to calculate probability (such as the probability that a random number in the interval [0,30] falls on [10,15] is 𝟑𝟎𝟓 = 𝟏𝟔…

C𝒏𝒎 = C𝒏𝒏 𝒎, boundary value

C𝒏𝟎 = C𝒏𝒏 = 𝟏 …

[Figure 24]

[Figure 25]

[Figure 26]

###### 491

gePoints

###### 1,819

|[Figure 27]<br><br>· Definition of cone · Structural features of cone · Section of cone · Volume of cylinder…<br><br>- The surface area of a cube is 𝑺 = 𝟔𝒂𝟐<br>- Comparative analysis: Difference from the surface area of a sphere<br><br><br>𝑺 = 𝟒𝝅𝒓𝟐<br><br>the surface area of a cube is larger under the same volume…<br><br>· Definition of polar coordinate system · Definition of plane rectangular coordinate system · Representation and equation of straight line · Distance formula between two points…<br><br>· Definition of line segment · Definition of straight line · Definition of ray · Calculating the area of a circle…<br><br>Plane geometry Solid geometry Analytical geometry<br><br>[Figure 28]<br><br>[Figure 29]<br><br>- Circle area calculation formula: 𝑺 = 𝝅𝒓𝟐 in polar coordinates<br><br>𝑺 =<br><br>𝟎<br><br>𝟐𝝅<br><br>𝟎<br><br>𝒓<br><br>𝝆𝒅𝝆𝒅𝜽<br><br>- Area and circumference relationship:<br><br>𝑺 =<br><br>𝑪𝟐 𝟒𝝅<br><br>…<br><br>- The distance formula between two points in the rectangular coordinate system: The distance between two points in the plane A(x₁, y₁) and B(x₂, y₂)<br><br>𝒅 = (x₂ − x₁)𝟐+(y₂ − y₁)𝟐<br><br>- The distance formula between two points in the rectangular coordinate system in space: The distance between two points in three dimensions A(x₁, y₁, z₁) and B(x₂, y₂, z₂)<br><br><br>𝒅 = (x₂ − x₁)𝟐+(y₂ − y₁)𝟐+(z₂ − z₁)𝟐 …<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]|
|---|

Graph Vectors Logic theory

###### Linear algebra

Functions and equations

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Principles

· Definition of matrix Basic operations on matrix · Complex operations on matrix …

· Definition of sufficient condition · Definition of necessary condition · Definition of necessary and sufficient condition …

· Basic concepts of graphs · Basic concepts of trees · Graph traversal …

· Concept of vectors · Modulus of vectors · The magnitude of a vector …

· Summation of a sequence · Definition of a geometric sequence · Definition of a linear function …

- Vector modulus: the size (length) of a vector, denoted as ∥ 𝜶 ∥,The modulus of an n-dimensional vector is

- Matrix addition: Add two matrices and add corresponding elements. Let

- A tree is an undirected graph that is connected. There are no cycles in a tree. When a tree has n vertices, it has exactly (n-1\ edges. …

- Sufficient condition: If condition A holds, then conclusion B must hold, denoted as

- The basic form of a linear function: a function of the form

𝜶 = 𝒂𝟏,𝒂𝟐,⋯,𝒂𝒏 ∥ 𝜶 ∥ is

𝑨 = (𝒂𝒊𝒋)𝒎×𝒏 𝑩 = (𝒃𝒊𝒋)𝒎×𝒏

𝒂𝟏𝟐 + 𝒂𝟐𝟐 + ⋯+ 𝒂𝒏𝟐

𝒚 = 𝒌𝒙 + 𝒃 where k and b are constants and 𝒌 ≠ 𝟎 …

- Equal vectors: vectors with the same modulus and direction. …

A ⇒B …

###### then

𝑨 + 𝑩 = (𝒂𝒊𝒋 + 𝒃𝒊𝒋)𝒎×𝒏 …

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Figure 1: Overview diagram of MathBook, including examples of knowledge points, fundamental principles, and sample problems.

answer, and XML script. We then use GeoGebra, a software that renders diagrams from XML-based scripts, to automate the generation of draft images: GLM(ki,pij) → (qidraft, adrafti , xxmli draft). The resulting visual drafts serve as references to guide human experts in constructing problems and diagrams via GeoGebra scripting. In practice, almost all drafts were revised or reworked by experts,5 in order to avoid reliance on superficial visual cues and ensure proper alignment with the underlying mathematical principles. The final seed problem set is Dseed = {(ki,pij,qi,ai,Ii,xxmli )}, covering all knowledge points and principles. The detailed GeoGebra-based diagram generation guidelines can be found in the Appendix C.2.1.

Variant Problem Expansion. To further enhance the diversity and generalization ability of the dataset, we systematically construct two types of variants based on each seed problem:

- (1) One-Problem-Multi-Image Variants (DImgVar): Given a seed problem (qi,ai,Ii) ∈ Dseed, we fix the problem statement qi and knowledge annotation (ki,pij), and generate a set of im-

ages {Ii(1),Ii(2),...,Ii(m)} by varying the parameters in GeoGebra while maintaining the underlying geometric construction. Each image corresponds to a different geometric instantiation (e.g.,

acute/obtuse/right triangle), resulting in distinct answers a(it): {(qi,a(it),Ii(t)) ∈ DImgVar, t = 1,...,m}. This approach enriches the visual data diversity while preserving semantic consistency.

- (2) One-Image-Multi-Problem Variants (DQstVar): Given a seed image Ii, we construct multiple new problems qi(s) targeting different knowledge points ki(s) and principles p(ijs), curated by experts

with language model assistance: {((qi(s),a(is),Ii) ∈ DQstVar, s = 1,...,n} This strategy leverages the reusability of high-quality diagrams to generate diverse problem variants.

By systematically applying these variant construction methods to each seed problem, we build the MathBook-Standard dataset with rich semantic and visual diversity.

##### 3.2.2 MathBook-Pro: Three-Dimensional Difficulty Modeling

To systematically characterize problem complexity from a model-centric perspective, we define a three-dimensional difficulty space for each seed problem along three orthogonal axes:

- (1) Step Complexity (ϕs): Knowledge-oriented reasoning depth is quantified by the number of involved knowledge points, which are sourced from the MathBook knowledge system. Given a

5Only 1.2% of the drafts were directly adopted by experts.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

###### (a) Pipeline

[Figure 53]

SFT Pre-aligned RL Dynamic Scheduling RL

###### (b) MathBook-Standard

(c) MathBook-Pro

[Figure 54]

[Figure 55]

[4K]

[1.8K]

Seed

Three-Dimensional-Difficulty

[Figure 56]

[Figure 57]

Find the measure of angle ∠BOC.

[4.5K]

One-Image-Multi-Problem

[Figure 58]

[Figure 59]

- 1. Find the measure of angle ∠OBC.
- 2. What type of angle is ∠BOC?

3. Find the measure of angle ∠DOB.

[5.8K]

One-Problem-Multi-Image

Find the measure of angle ∠BOC.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Figure 2: Overview of MathBook dataset and the corresponding training phase.

seed problem with l process-oriented knowledge points, we construct variants that require at least six, satisfying l′ > l.

- (2) Visual Complexity (ϕv): We increase complexity by adding auxiliary elements (e.g., lines) to the original image via GeoGebra, while preserving the core structure.
- (3) Contextual Complexity (ϕc): Captures the contextualization of the problem statement. We vary the textual context from concise mathematical descriptions to complex linguistic scenarios.

Each seed problem (q0,a0,I0) ∈ Dseed serves as the origin in a structured difficulty space. To enable controlled and interpretable expansion, we generate derived problems by varying a single dimension

d ∈ {ϕs,ϕv,ϕc} at a time, yielding variants (qi(d),ai(d),Ii(d)). Through multiple rounds of such singleaxis transformations, we progressively construct more complex problems by composing changes

across multiple dimensions. Formally, the most advanced variant takes the form: (q∗,a∗,I∗) = ϕs ◦ ϕv ◦ ϕc(q0,a0,I0). In MathBook-Pro, the expansion along each dimension is implemented as:

- (1) Along the ϕs dimension, we introduce intermediate conclusions as new conditions, enabling a knowledge-driven, progressive deepening of reasoning, expressed as Ki+1 = Ki + 1, where Ki denotes the number of knowledge points involved at step i. In MathBook-Pro, the most complex step variants involve at least 6 knowledge points.
- (2) Along the ϕv, we increase visual complexity by adding auxiliary lines, altering geometric configurations or introducing new spatial constructs via GeoGebra, while preserving the core structure.
- (3) Along ϕc, we embed the mathematical core into real-world contexts or linguistically abstract scenarios, increasing the semantic and contextual demands of the problem statement.

By expanding along the defined dimensions, we generate a set of difficulty-controlled problem variants for each knowledge point, forming the difficulty modeling subset Ddifficulty of MathBook-Pro. Detailed dataset statistics are presented in Figure 2.

##### 3.3 MathBookEval

Design Principles. To ensure the quality and interpretability of annotations in visual math reasoning tasks, MathBookEval is designed based on the following principles: (1) Comprehensive Knowledge Coverage: Problems involve 491 knowledge points, spanning primary to university level, demonstrating broad coverage. (2) Multi-level Reasoning Depth: Each problem integrates 1–10 knowledge points, compared to 1–3 (Level 1) in existing process-oriented benchmarks, as illustrated in Figure 24.

Notably, our annotation adheres to three principles: (1) integrating public and newly constructed problems under a unified guideline; (2) expert step-by-step annotation with explicit knowledge-point mapping; and (3) independent cross-validation, retaining only consistently annotated items.

Data Statistics and Evaluation Protocol. MathBookEval contains 1,000 fully annotated problems, covering all 491 knowledge points in the unified knowledge system K, with 600 problems collected from existing benchmarks and 400 newly curated (Detailed statistics are presented in Table 6). We provide detailed statistics and splits along two key dimensions:

- (1) Reasoning Dimension: Problems are divided by reasoning steps into three levels: 1-3 (Level 1), 4-6 (Level 2), and 7-10 (Level 3), reflecting different reasoning depths.
- (2) Knowledge Dimension: The 491 knowledge points are grouped into 4 domains and 13 subdomains, covering primary to university level. Figure 24 demonstrates superior coverage of knowledge points and reasoning depth. All problems are in multiple-choice or fill-in-the-blank format.

### 4 Methodology

In this section, we introduce MathBook-RL, a two-stage framework that progressively guides MLLMs to develop reasoning capabilities from easy to hard. The first stage is a cold-start fine-tuning phase that establishes a knowledge-driven reasoning paradigm (§4.1); the second is a dynamic reinforcement learning phase that enhances the model’s generalization ability (§4.2).

##### 4.1 Cold-Start Fine-tuning

The cold-start supervised fine-tuning (SFT) stage aims to instill explicit awareness of knowledge system and a knowledge-driven reasoning paradigm, avoiding rote memorization. The initial training set Dinit is built from MathBook-Standard, which fully covers all 491 knowledge points. To improve rationale interpretability, we use GPT-4o [37] to rewrite each sample with natural language explanations that explicitly reference the relevant knowledge. The model is then trained using standard supervised fine-tuning: LSFT(θ) = E(x,y)∼D

[−log Pθ(y | x)]. This stage enhances the model’s ability to internalize the knowledge system and follow knowledge-guided reasoning chains.

init

##### 4.2 Progressive Alignment Reinforcement Learning

- (1) Pre-aligned RL. Prior to the dynamic scheduling stage, we perform initial RL training on MathBook-Standard dataset to ensure that the model develops genuine understanding of mathematical knowledge. Specifically, we utilize the DImgVar subset, where each group contains multiple

variants of the same knowledge principle: (qi,a(it),Ii(t)) ∈ DImgVar, t = 1,...,m. To encourage consistent and robust performance across different formulations, we adopt a mean-based reward

function: r = m1 mt=1 r(t), where r(t) = 0.9 if the answer is correct, 0.1 if only the format is correct, and 0 otherwise. Specifically, for problems corresponding to the same knowledge principle,

rollout rewards are first sorted within each problem. Next, the mean reward at each sorted position is calculated across these problems and subsequently employed in the calculation of Ai. Instead of focusing on individual problems, this design integrates rewards across all problems corresponding to the same knowledge principle, thereby providing a more comprehensive critic.

- (2) Dynamic Scheduling RL. In this section, we introduce a dynamic RL algorithm based on MathBook-Pro. The training process is organized as a curriculum along a main trajectory of increasing difficulty, primarily centered on the knowledge dimension. For each base problem (q0,a0,I0), denoted as x0, we construct a sequence of increasingly challenging variants as follows:

x0 → ϕs(x0) → ϕs ◦ ϕv(x0) → ϕs ◦ ϕc(x0) → ϕs ◦ ϕv ◦ ϕc(x0) (1)

where ϕs denotes increasing the number of knowledge points, ϕv and ϕc denotes increasing visual complexity and contextual abstraction. This forms a progressive path from basic to advanced reasoning for each knowledge anchor.

Incremental Learning Mechanism. At each curriculum transition x → ϕ(x), if the model fails on ϕ(x) after succeeding on x, we introduce an incremental learning step. Specifically, we define the incremental set ∆(x,ϕ) as a collection of samples that isolate the new knowledge or modality

introduced by ϕ. The model is first trained on ∆(x,ϕ) to address the incremental challenge, then reattempts ϕ(x). Concretely:

- • Knowledge Increment Scheduling: For x0 → ϕs(x0), if the model fails on ϕs(x0), we construct ∆(x0,ϕs), comprising auxiliary problems x′0 that target the new knowledge point(s) from ϕs.
- • Modality Increment Scheduling: For ϕs(x0) → ϕs ◦ ϕv(x0) (or ϕs ◦ ϕc(x0)), if the model fails on the more complex sample, we construct ∆(ϕs(x0),ϕv) (or ∆(ϕs(x0),ϕc)), which contains samples isolating the new visual or contextual complexity.

This incremental adaptation, denoted by ∆(x,ϕ) at each step, ensures that the model can efficiently bridge the gap between curriculum stages. Notably, our overall RL objective is optimized using Group Relative Policy Optimization (GRPO) [46], which extends PPO by estimating the baseline from group scores instead of a separate critic. The GRPO objective is:

|oi|

G

1 G

1 |oi|

J (θ) = E[q ∼ P(Q), {oi}Gi=1 ∼ πθold(O|q)]

(2)

t=1

i=1

πθ(oi,t|q, oi,<t) πθold(oi,t|q, oi,<t)

πθ(oi,t|q, oi,<t) πθold(oi,t|q, oi,<t)

, 1 − ϵ, 1 + ϵ A ˆi,t − βDKL [πθ||πref] ,

Aˆi,t, clip

min

where ϵ and β are hyperparameters, q denotes the input, {oi}Gi=1 are sampled outputs, and ri is the corresponding reward. Aˆi,t is the normalized advantage value for the i-th trajectory in the group.

This curriculum-driven RL process, augmented with explicit incremental adaptation at each stage, enables the MLLM to progressively master complex, multi-dimensional reasoning tasks while preserving stability and generalization across knowledge, visual, and contextual variations.

### 5 Experiments

##### 5.1 Experimental Setup

Datasets. All training data are sourced from WE-MATH 2.0 in compliance with copyright and licensing requirements, and all expert-constructed problems will be released under appropriate CC licenses. We use 1K, 5.8K and 4K samples for SFT, pre-aligned RL, and dynamic scheduling RL stages, respectively. Experiments are conducted on four standard mathematical reasoning benchmarks: MathVista [32], MathVision [58], MathVerse [71], and We-Math [43].

Baselines. We conduct our training based on Qwen2.5-VL-7B [3] and compare our method with three categories of baselines: (1) Closed-source models (e.g., GPT-4o [37]); (2) Open-source general models (e.g., InternVL2.5 series [6], Qwen2.5-VL series [3]); (3) Open-source reasoning models (e.g., R1-VL [70]). Our evaluation is based on VLMEvalKit [14], with some modifications to the API-calling components due to network constraints. Details of baselines and evaluation metrics are listed in supplementary.

##### 5.2 Main Results

- Table 2 displays the performance of our MathBook-7B across various mathematical reasoning benchmarks. Overall, our method achieves remarkable performance in all benchmarks, clearly demonstrating its superiority. Further analysis reveals the following observations:

- (1) Overall superiority of MathBook. Compared to the backbone model Qwen2.5-VL-7B, MathBook-7B achieves over a 5% improvement across all benchmarks, validating the effectiveness of our approach.
- (2) Effectiveness of progressive alignment reinforcement learning on knowledge generalization. Focusing on the We-Math benchmark, which requires models to correctly solve both complex multistep questions and their corresponding subproblems, MathBook-7B outperforms strong RL-based baselines. This demonstrates the effectiveness of progressive alignment reinforcement learning for improving knowledge generalization.
- (3) Less is More: Efficiency with limited training data. Notably, MathBook-7B achieves consistently strong performance using only 9.8K training samples. We attribute this to the high-quality,

Table 2: Performance comparison across four widely-used mathematical reasoning benchmarks. Each benchmark follows its standard evaluation metric: MathVista and MathVision use accuracy, We-Math reports the strict score, and MathVerse evaluates on the vision-only subset with accuracy. Data sizes used for SFT and RL are annotated in blue and red, respectively.

Model #Data Avg. MathVista MathVision We-Math MathVerse Closed-source

GPT-4o-latest [37] - 54.0 71.6 43.8 50.6 49.9 Gemini-1.5-Pro [51] - 53.6 67.9 41 50.5 54.8

Open-source (General)

Qwen2.5-VL-7B [3] - 42.6 68.2 25.1 36.0 41.1 InternVL2.5-8B-BoN-8 [6] - 41.7 68.2 25.6 38.6 34.5

Open-source (Reasoning)

Math-PUMA-7B [76] 1.88M - 47.9 - 19.2 26.0 URSA-8B [35] 2.96M 37.8 58.8 28.7 32.8 31.0 R1-OneVision-7B [66] 155K+10K - 64.1 29.9 30.1 R1-VL-7B [70] 260K+10K - 63.5 24.7 22.7 MM-Eureka-7B [36] 15K 45.2 73.0 26.9 34.5 46.2 WeThink-7B [65] 120K+20K 47.5 71.6 26.0 48.0 44.2 VLAA-Thinker-7B[5] 25K 46.0 68.0 26.4 41.5 48.2 OpenVLThinker-7B [10] 35K+15K - 72.3 25.9 - MathBook-7B (Ours) 1K+9.8K 48.7 73.0 28.0 48.4 45.2

structured mathematical knowledge system we constructed, which enables efficient alignment and generalization even with limited data.

##### 5.3 Results on MathBookEval

To investigate the abilities of existing MLLMs in terms of reasoning depth and knowledge coverage breadth, we conduct MathBookEval evaluation and identify the following findings (see Table 5):

Table 3: Results of the ablation study. MVt: MathVista; MVs: MathVision; WM: We-Math)

Method SFT RL-Pre RL-Dyn MVt MVs WM M0 ✓ ✓ ✓ 73.0 28.0 48.4 M1 ✓ ✓ - 72.4 27.0 47.2 M2 ✓ - ✓ 72.0 26.3 43.3 M3 - ✓ ✓ 71.5 26.3 46.7 M4 ✓ - - 65.8 25.7 38.3

- (1) MLLMs performance negatively correlates with the number of required knowledge points. As the "reasoning step dimension" increases, model accuracy steadily declines. In particular, problems requiring 7–10 knowledge points result in the lowest performance for most MLLMs (below 50%). These findings underscore the ongoing challenge of multi-step reasoning, further validating the number of knowledge points as a robust indicator of problem difficulty.
- (2) MLLMs perform well on algebra but struggle with geometry. Along the knowledge dimension, most MLLMs demonstrate strong performance in algebra, achieving accuracies above 50%. However, their consistently poor performance in geometry highlights ongoing challenges in spatial reasoning.
- (3) Larger models yield more consistent improvements across all dimensions. Within the InternVL2.5 and Qwen2.5-VL families, increasing model size leads to consistent gains across all dimensions and in overall scores, emphasizing the role of scale in enhancing reasoning capabilities.

##### 5.4 Quantitative Analysis

Ablation Study. As shown in Table 3, we conduct ablation studies on the training stages. M0 denotes MathBook-7B, while M1–M4 represent models at different training stages (RL-Pre: Prealigned RL; RL-Dyn: Dynamic Scheduling RL). We lead to following two key findings:

(i) Both RL stages contribute significantly. Each RL stage (M0-M3) yields progressive improvements over M4. In particular, pre-aligned reinforcement learning (RL) in the first stage yields

Table 5: The performance of different MLLMs on MathBookEval for reasoning evaluation. Acc.: Accuracy; FS.: Foundational skills; PS.: Probability and statistics; Geo.: Geometry; Alg.: Algbra

Reasoning Knowledge Level1 Level2 Level3 FS. PS. Geo. Alg.

Models Acc.

Closed-source MLLMs

GPT-4o 50.8 52.8 48.9 41.7 33.8 57.6 44.2 67.2 GPT-4V 42.8 44.0 43.0 31.9 36.8 56.6 33.5 59.4

Open-source MLLMs

InternVL2.5-78B 51.8 52.5 51.8 45.8 50.0 64.2 42.6 67.6 Qwen2.5-VL-72B 57.1 58.3 56.4 50.0 52.9 58.5 52.1 68.8

LLaVA-OneVision-72B 43.0 44.8 42.0 31.9 38.2 52.8 37.0 53.5 InternVL2.5-8B 37.9 40.7 34.5 27.8 33.8 46.2 31.4 50.0 Qwen2.5-VL-7B 46.7 50.1 43.0 33.3 44.1 58.5 38.8 60.2 Qwen2.5-VL-3B 36.9 38.7 34.2 33.3 35.3 49.1 29.1 49.6

LLaVA-OneVision-7B 31.6 34.3 28.0 23.6 36.8 41.5 24.9 41.0 GLM-4V-9B 22.2 23.7 20.5 16.7 26.5 23.6 18.4 28.9 MathBook-7B 50.4 52.0 48.2 45.8 57.4 67.9 40.5 63.3

impressive results on MathVista and We-Math benchmarks, highlighting the crucial role of knowledge learning in enhancing mathematical reasoning abilities. (ii) SFT alone offers limited gains, but is crucial for unlocking RL potential. Comparing M0, M3, M4, we find that SFT alone yields marginal improvements over the Qwen2.5-VL backbone. However, when combined with RL, SFT version substantially boosts overall performance, highlighting its critical role in shifting the model’s reasoning paradigm and enabling more effective RL optimization.

Analysis of SFT Data Paradigm and Scale. In this section, we explore the impact of both data paradigm and data scale during the SFT stage. Based on the M0 setting, we consider two variants: (1) Replacing the natural language CoT format with a structured, step-wise CoT format [75] aligned with K; (2) Increasing the SFT data scale with larger datasets (from 1K to 10K). We identify following findings.

Table 4: Results of the analysis experiments. SFT (Str.) refers to structured SFT training, while SFT (Lar.) indicates training with a larger dataset.

Setting MVt MVs WM M0 73.0 28.0 48.4 SFT(Str.) 71.9 26.0 46.7 SFT(Lar.) 72.1 27.0 47.8

(i) Natural language CoT outperforms the structured step-wise format in SFT. As evidenced in Table 4, the structured SFT format results in lower RL performance relative to natural language CoT. This highlights the advantage of natural prompts in cultivating flexible reasoning, which in turn strengthens visual mathematical reasoning skills. (ii) Minimal SFT suffices to unlock RL potential. Scaling up SFT data does not consistently lead to better performance—models trained on minimal, well-curated SFT data perform comparably or even better than those trained on larger datasets. This suggests that a small, high-quality SFT set is sufficient to establish the reasoning paradigm required for effective RL.

### 6 Conclusion

In this work, we present WE-MATH 2.0, a unified framework for multi-modal mathematical reasoning. It comprises: (1) MathBook Knowledge System, a five-level hierarchy covering 491 knowledge points and 1,819 fundamental principles for comprehensive supervision; (2) MathBook-Standard and MathBook-Pro, two richly annotated datasets with conceptual expansions and principled difficulty modeling for structured learning; (3) MathBook-RL, a two-stage reinforcement learning framework that leverages knowledge-guided supervision and dynamic data scheduling; and (4) MathBookEval, a benchmark for evaluating reasoning across diverse knowledge and step distributions. Extensive experiments validate MathBook’s effectiveness in enhancing generalization of MLLMs.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Inclusion AI, Fudong Wang, Jiajia Liu, Jingdong Chen, Jun Zhou, Kaixiang Ji, Lixiang Ru, Qingpei Guo, Ruobing Zheng, Tianqi Li, et al. M2-reasoning: Empowering mllms with unified general and spatial reasoning. arXiv preprint arXiv:2507.08306, 2025.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [5] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025.
- [6] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. CoRR, abs/2412.05271, 2024.
- [7] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.
- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.
- [9] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [10] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative selfimprovement. arXiv preprint arXiv:2503.17352, 2025.
- [11] Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. Tool-star: Empowering llm-brained multi-tool reasoner via reinforcement learning. arXiv preprint arXiv:2505.16410, 2025.
- [12] Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, et al. Agentic reinforced policy optimization. arXiv preprint arXiv:2507.19849, 2025.
- [13] Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, and Ji-Rong Wen. Progressive multimodal reasoning via active retrieval. arXiv preprint arXiv:2412.14835, 2024.
- [14] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024.

- [15] Richard Fitzpatrick. Euclid’s elements of geometry. 2008.
- [16] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.
- [17] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.
- [18] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.
- [19] Xiaotian Han, Yiren Jian, Xuefeng Hu, Haogeng Liu, Yiqi Wang, Qihang Fan, Yuang Ai, Huaibo Huang, Ran He, Zhenheng Yang, et al. Infimm-webmath-40b: Advancing multimodal pre-training for enhanced mathematical reasoning. arXiv preprint arXiv:2409.12568, 2024.
- [20] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.
- [21] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [22] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [23] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.
- [24] Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Stepdpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629, 2024.
- [25] Shanglin Lei, Guanting Dong, Xiaoping Wang, Keheng Wang, Runqi Qiao, and Sirui Wang. Instructerc: Reforming emotion recognition in conversation with multi-task retrieval-augmented large language models. arXiv preprint arXiv:2309.11911, 2023.
- [26] Zhikai Lei, Tianyi Liang, Hanglei Hu, Jin Zhang, Yunhua Zhou, Yunfan Shao, Linyang Li, Chenchui Li, Changbo Wang, Hang Yan, et al. Gaokao-eval: Does high scores truly reflect strong capabilities in llms? arXiv preprint arXiv:2412.10056, 2024.
- [27] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [28] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024.
- [29] Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and Michael Qizhe Shieh. Noisyrollout: Reinforcing visual reasoning with data augmentation. arXiv preprint arXiv:2504.13055, 2025.
- [30] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473, 2024.

- [31] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [32] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [33] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.
- [34] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021.
- [35] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Yiyao Yu, Xinzhe Ni, Zicheng Lin, Jin Zeng, and Yujiu Yang. Ursa: Understanding and verifying chain-of-thought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686, 2025.
- [36] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.
- [37] OpenAI. Hello gpt-4o, 2024.
- [38] R OpenAI. Gpt-4v (ision) system card. Citekey: gptvision, 2023.
- [39] Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024.
- [40] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. arXiv preprint arXiv:2504.13958, 2025.
- [41] Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Jiapeng Wang, YiFan Zhang, Zhuoma GongQue, Chong Sun, Yida Xu, Yadong Xue, Ye Tian, Zhimin Bao, Lan Yang, Chen Li, and Honggang Zhang. V-oracle: Making progressive reasoning in deciphering oracle bones for you and me. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20124–20150, Vienna, Austria, July 2025. Association for Computational Linguistics.
- [42] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.
- [43] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma Gongque, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Yifan Zhang, Xiao Zong, Yida Xu, Muxi Diao, Zhimin Bao, Chen Li, and Honggang Zhang. We-math: Does your large multimodal model achieve human-like mathematical reasoning? CoRR, abs/2407.01284, 2024.
- [44] Runqi Qiao, Lan Yang, Kaiyue Pang, and Honggang Zhang. Making visual sense of oracle bones for you and me. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12656–12665, June 2024.
- [45] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the" edge" of open-set object detection. arXiv preprint arXiv:2405.10300, 2024.

- [46] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [47] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-LLaVA: Bootstrapping mathematical reasoning for multimodal large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4663–4680, November 2024.
- [48] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.
- [49] Xiaoshuai Song, Muxi Diao, Guanting Dong, Zhengyang Wang, Yujia Fu, Runqi Qiao, Zhexu Wang, Dayuan Fu, Huangxuan Wu, Bin Liang, et al. Cs-bench: A comprehensive benchmark for large language models towards computer science mastery. arXiv preprint arXiv:2406.08587, 2024.
- [50] Core Team, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xinzhe Xu, Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, and Bingquan Xia. Mimo-vl technical report, 2025.
- [51] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [52] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025.
- [53] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024.
- [54] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.
- [55] Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, et al. Srpo: Enhancing multimodal llm reasoning via reflectionaware reinforcement learning. arXiv preprint arXiv:2506.01713, 2025.
- [56] Ziyu Wan, Xidong Feng, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. In Forty-first International Conference on Machine Learning, 2024.
- [57] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset.

- Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

[58] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset.

- Advances in Neural Information Processing Systems, 37:95095–95169, 2025.

- [59] Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, et al. Mathcoder-vl: Bridging vision and code for enhanced multimodal mathematical reasoning. arXiv preprint arXiv:2505.10557, 2025.

- [60] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. arXiv preprint arXiv:2312.08935, 2023.
- [61] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. 2024.
- [62] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 December 9, 2022, 2022.
- [63] Huajian Xin, ZZ Ren, Junxiao Song, Zhihong Shao, Wanjia Zhao, Haocheng Wang, Bo Liu, Liyue Zhang, Xuan Lu, Qiushi Du, et al. Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. arXiv preprint arXiv:2408.08152,

- 2024.

[64] Liangyu Xu, Yingxiu Zhao, Jingyun Wang, Yingyao Wang, Bu Pi, Chen Wang, Mingliang Zhang, Jihao Gu, Xiang Li, Xiaoyong Zhu, et al. Geosense: Evaluating identification and application of geometric principles in multimodal reasoning. arXiv preprint arXiv:2504.12597,

- 2025.

- [65] Jie Yang, Feipeng Ma, Zitian Wang, Dacheng Yin, Kang Rong, Fengyun Rao, and Ruimao Zhang. Wethink: Toward general-purpose vision-language reasoning via reinforcement learning. arXiv preprint arXiv:2506.07905, 2025.
- [66] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.
- [67] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023.
- [68] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023.
- [69] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.
- [70] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025.
- [71] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.
- [72] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, et al. Mavis: Mathematical visual instruction tuning with an automatic data engine. arXiv preprint arXiv:2407.08739, 2024.
- [73] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

- [74] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [75] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. arXiv preprint arXiv:2408.08640, 2024.
- [76] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 26183–26191, 2025.
- [77] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024.

### Contents

- 1 Introduction 1
- 2 Related Work 3
- 3 WE-MATH 2.0 4

- 3.1 MathBook Knowledge System . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2 Multi-Dimensional Data Construction . . . . . . . . . . . . . . . . . . . . . . . . 4

- 3.2.1 MathBook-Standard: Seed and Variant Problem Construction . . . . . . . 4
- 3.2.2 MathBook-Pro: Three-Dimensional Difficulty Modeling . . . . . . . . . . 5

- 3.3 MathBookEval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 4 Methodology 7

- 4.1 Cold-Start Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.2 Progressive Alignment Reinforcement Learning . . . . . . . . . . . . . . . . . . . 7

- 5 Experiments 8

- 5.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 5.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 5.3 Results on MathBookEval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 5.4 Quantitative Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 6 Conclusion 10

- A Broaden Impact 18
- B Ethical Considerations 18
- C Details of WE-MATH 2.0 19

- C.1 MathBook Knowledge System . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- C.1.1 Hierarchical Structure of Knowledge Points . . . . . . . . . . . . . . . . . 19
- C.1.2 Knowledge Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- C.2 MathBook-Standard . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- C.2.1 GeoGebra-based Diagram Generation. . . . . . . . . . . . . . . . . . . . . 26
- C.2.2 Dataset Diversity and Variant Construction. . . . . . . . . . . . . . . . . . 26

- C.3 MathBook-Pro . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- C.4 MathBookEval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

- C.4.1 Dataset Construction and Annotation Protocol. . . . . . . . . . . . . . . . 35
- C.4.2 Task Dimensions. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- C.4.3 Dataset Statistics. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- C.4.4 Evaluation Protocol and Metrics. . . . . . . . . . . . . . . . . . . . . . . . 36
- C.4.5 Experiment Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- C.4.6 Additional Results on MathBookEval . . . . . . . . . . . . . . . . . . . . 37

##### D More Details on MathBook-RL 38

- D.1 Implementation details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- D.2 Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- D.3 Experiment Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- D.3.1 Details of the Evaluation. . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- D.3.2 Details of the Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

### A Broaden Impact

Towards principled and generalizable mathematical model training. WE-MATH 2.0 provides a comprehensive and structured mathematical knowledge system, which fills the gap left by previous works that lack a complete and systematic framework. By introducing a five-level hierarchy with 491 knowledge points and 1,819 fundamental principles, MathBook enables more principled and interpretable mathematical learning for MLLMs. The dual expansion strategy ("multi-images per question" and "multi-questions per image") and the three-dimensional difficulty modeling not only enrich the diversity of training data but also facilitate robust and progressive learning. This systematic approach can inspire future research to adopt knowledge-driven and model-centric data space modeling, leading to more reliable and generalizable mathematical reasoning models. Furthermore, the fine-grained annotations and progressive difficulty levels provide a valuable resource for benchmarking and analyzing the strengths and weaknesses of different MLLMs, promoting transparency and interpretability in model development.

Bridging AI and Education: high-quality datasets for teaching and learning. WE-MATH 2.0’s datasets are not only designed for model training but also have significant educational value. Each problem is accompanied by a GeoGebra (GGB) file, which can serve as high-quality teaching material for educators and students. The hierarchical knowledge system and step-wise annotations make it easier to design personalized learning paths and targeted exercises, supporting adaptive learning and formative assessment. The multi-modal and multi-perspective problem sets encourage students to develop flexible thinking and deepen their conceptual understanding. By bridging the gap between AI research and educational practice, MathBook has the potential to enhance mathematics education, facilitate interactive and engaging learning experiences, and support the development of intelligent tutoring systems.

Enhancing RL generalization through progressive and dynamic training. WE-MATH 2.0 introduces a novel, model-centric curriculum for RL-based training, where problems are systematically organized from easy to hard based on explicit difficulty modeling. This approach provides a new perspective for designing RL curricula, enabling more effective and efficient learning. The "onequestion-multi-image" and "one-image-multi-question" strategies, together with dynamic scheduling mechanisms, enhance the robustness and generalization of RL-trained models. These innovations can inspire the broader RL community to explore curriculum learning, dynamic data scheduling, and multi-modal data augmentation for complex reasoning tasks. Moreover, these hierarchical knowledge approaches also offers a new solution for tool learning [40, 12, 11]. MathBook thus serves as a valuable testbed for advancing RL methods in the context of mathematical reasoning and beyond.

### B Ethical Considerations

Licensing and Open Access. For all referenced or incorporated data in WE-MATH 2.0, we only use existing datasets with clear and appropriate licenses. All data curated by our team will be released under the CC BY 4.0 license, ensuring open access for the research community. The entire MathBook dataset, including both external and newly constructed components, will be made publicly available to facilitate further research and development.

Data Sources and Privacy. All data in WE-MATH 2.0 are either sourced from publicly available datasets or generated by our expert team, and do not contain any personal user information. Therefore, there are no privacy concerns related to our dataset.

Expert Compensation. Experts involved in annotation are compensated on a per-task basis, with payment issued only after cross-validation and quality assurance. All compensation meets or exceeds the local minimum wage standards.

### C Details of WE-MATH 2.0

- C.1 MathBook Knowledge System

- C.1.1 Hierarchical Structure of Knowledge Points

As illustrated in Figures 3–6, we present a partial example of the hierarchical structure of knowledge points in the MathBook Knowledge System. The system is organized as a five-level hierarchical structure of knowledge points K = {k1,k2,...,kN}, where N = 491 denotes the total number of knowledge points at the lowest level of the hierarchy. The first level consists of four categories: Geometry, Fundamental Skills, Algebra, and Probability and Statistics.

The construction of K follows a two-track, human-AI collaborative process. First, an initial version Khuman is constructed by collecting and merging knowledge point lists from authoritative sources, including Wikipedia, open-source mathematics textbooks, and national curriculum standards. This initial structure is deduplicated, reorganized, and refined for logical consistency and comprehensive coverage.

In parallel, a large-scale problem set Q is collected, including 30,000 sampled from existing math datasets. GPT-4o is used to assign multi-level topic tags T = {t1,...,tn} to each problem, and a semantic similarity matrix S ∈ Rn×n is computed. Hierarchical clustering is then applied to S to generate an AI-assisted hierarchical structure of knowledge points Kauto.

Finally, the AI-assisted structure Kauto is integrated with the initial structure Khuman through systematic comparison, merging, and revision. The Kauto serves as a reference for revising and refining the manually constructed hierarchical structure of knowledge points, resulting in the final knowledge point set K.

[Figure 64]

###### Figure 3: Overview of the hierarchical structure of knowledge points in MathBook (1).

[Figure 65]

###### Figure 4: Overview of the hierarchical structure of knowledge points in MathBook (2).

[Figure 66]

###### Figure 5: Overview of the hierarchical structure of knowledge points in MathBook (3).

[Figure 67]

###### Figure 6: Overview of the hierarchical structure of knowledge points in MathBook (4).

##### C.1.2 Knowledge Principles

As shown in Figures 7–10, we provide several examples of knowledge principles, which include definitions, theorems, and other foundational statements associated with each knowledge point.

The annotation of principles P = Ni=1 Pi (|P| = 2,157) also follows a two-track, human-AI collaborative approach.

Based on the constructed knowledge hierarchy K, a set of core principles for each knowledge point ki is first drafted, referencing authoritative sources such as Wikipedia, textbooks, and international curriculum standards.

In parallel, for each ki, a set of representative problems from Q is selected and their chain-of-thought (CoT) solutions are annotated. Each step in the CoT is mapped to the corresponding knowledge point

using GPT-4o, and the relevant CoT steps for each ki are extracted. The CoT steps associated with each knowledge point are then aggregated and reviewed to supplement, refine, and validate the set of

principles for ki.

This process is repeated iteratively, consolidating both the expert-written and data-driven principles, cross-checking against original sources and annotated solution paths, until the set of principles Pi for each knowledge point is comprehensive and precise.

###### Geometry

Geometry-Plane Geometry-Basic Plane Figures-Triangles -Classification of triangles

Geometry-Plane Geometry-Relationships Between Figures -Intersection and Perpendicularity-Adjacent supplementary angles

Principle:

Principle:

- 1. Definition of adjacent supplementary angles: Two adjacent angles (sharing one common side) whose measures sum to 180°are called adjacent supplementary angles (e.g., lines AB and CD intersect at O, then∠AOC and∠COB are adjacent supplementary angles).
- 2. Characteristics of adjacent supplementary angles: (1) A common side is shared and lies between the angles; (2) the other sides are extensions of each other in opposite directions;

(3) their sum is a straight angle (mathematical expression:∠α +∠β = π). Used in geometric proofs to complete angle relationships.

- 3. Judgment of adjacent supplementary angles: If two angles share one side and their noncommon sides form a straight line, then they are adjacent supplementary angles (e.g., if∠1 and∠2 share side OA and OB lies on straight line AB). Applications: constructing interior angles of parallelograms, computing hinge opening angles in doors/windows.
- 4. Vector expression of adjacent supplementary angles: If the angle between two vectors is θ, then the sum of the angle and its adjacent supplement is θ + (180°− θ) = 180°.

- 1. Definition of an acute triangle: A triangle where all three interior angles are less than 90°(e.g., angles of 50°, 60°, 70°), with all altitudes located inside the triangle and the orthocentre inside as well; used in stable structural designs (e.g., Eiffel Tower trusses).
- 2. Definition of a right triangle: A triangle with one 90°angle (e.g., sides 3, 4, 5 satisfy 3²

+ 4² = 5²), the other two angles are complementary (α + β = 90°), and the Pythagorean theorem a² + b² = c² is used in GPS triangle positioning for coordinate calculations.

- 3. Definition of an obtuse triangle: A triangle with one interior angle greater than 90° and less than 180°(e.g., angles of 100°, 40°, 40°), where the side opposite the largest angle is longest; its corresponding altitude lies inside the triangle, but the other two altitudes need to be extended outside the triangle (e.g., for structural stress distribution in engineering blueprints).

Geometry-Solid Geometry-Basic Solid Figures-Cones -Section of a cone

Geometry-Solid Geometry-Basic Solid Figures -Volume of irregular solids

Principle:

Principle:

- 1. Cross-section of a cone: a cross-section parallel to the base is a circle (radius r' = r * h'/h, where h' is the distance from the apex), used in layered volume calculation.
- 2. A vertical cross-section along the height is an isosceles triangle (base = 2r, height = h, area = r * h), used in symmetric cutting of cone sculptures.
- 3. An oblique section results in a conic section: depending on the angle, the section may be an ellipse (angle < angle between generator and height), a parabola (parallel to generator), or a hyperbola (angle > angle between generator and height). These are used in classifying conic sections and applied in satellite dish design and spotlight reflection paths.

- 1. Object combination: Combine known basic solid figures into a new object and calculate the volume of the new object. Usually, the volume of the basic solid figures as the combination elements can be calculated separately, and finally the volume of the irregular figure can be obtained by adding them up.
- 2. Object disassembly: Remove a part of the volume from the known basic solid figure and calculate the volume of the new object. Usually, the volume of the original solid figure and the removed volume can be calculated separately, and finally the volume of the irregular figure can be obtained by subtracting them.
- 3. …

Geometry-Analytic Geometry-Circles -Problem of a circle passing through a fixed point

Geometry-Analytic Geometry-Analytic Geometry -Definition of the slope of a line

Principle:

Principle:

- 1. Parametric equation processing: For the equation with parameters x2+y2+Ax+By+C=0, the fixed point satisfies A,B,C and the coefficients are independent of the parameters (for example: the equation x2+y2+λx+(1-λ)y-5=0 needs to solve λ(x - y +1) + (x2+y2 + y -5)=0 to get x=1,y=2)
- 2. Geometric condition method: Use the equation of the circle system (such as the intersection line of two circles and the third circle) to determine the fixed point; Easy to make mistakes: Failure to completely eliminate parameters leads to missed solutions, and parameter-related items are mistakenly regarded as independent conditions; Question type: Find all the fixed points that the circle passes through (for example: k(x2+y2)+x+y+1=0 passes through the fixed point), prove that the circle system passes through the fixed point, and find the equation parameters at a known fixed point
- 3. …

- 1. Conditions for two lines to be parallel: When the slopes of the two lines are equal, that is, k1 = k2 , and the intercepts are different (such as y=2x+3 and y=2x-5 are parallel); Parallel determination questions (verifying whether the two lines are parallel) and equation construction questions (finding equations parallel to the known lines); Easy mistakes: Ignoring the same intercepts leads to the coincidence of the lines
- 2. Conditions for two lines to be perpendicular: When the product of the slopes of the two lines is -1, that is, k1 , k2 = -1 , or one line is perpendicular (slope does not exist) and the other is horizontal k=0; vertical determination questions (verifying the orthogonality of lines) and geometric construction questions (finding the equation of the perpendicular line to a given line); easy mistakes: ignoring the special cases of vertical and horizontal lines (such as x=3 and y=5 are perpendicular)
- 3. …

- Figure 7: Examples of knowledge principles corresponding to specific knowledge points in MathBook

- (1).

###### Fundamental Skills

Fundamental Skills-Understanding Numbers and Quantities-Compare sizes in images-Definition of integers

Fundamental Skills-Compare sizes in images

Principle:

Principle:

- 1. Definition of natural numbers: Non-negative integers, used for counting or ordering.
- 2. Definition of positive integers, negative integers, and zero: Positive integers: Greater than 0, typically used for quantity. Zero: Neither positive nor negative, represents “none” or a reference point. Negative integers: Less than 0, representing opposite quantities.
- 3. Set notation: The set of integers is denoted by Z. The set of natural numbers is denoted by N.

- 1. Comparison of function values: Compare y-values of different functions at the same x-value.
- 2. Comparison of rate of change: Compare slopes (rate of increase/decrease) over the same interval.
- 3. Comparison of extrema: Compare maximum/minimum values of different functions.
- 4. Comparison of definite integrals: Compare the area enclosed between the curve and the xaxis.

Fundamental Skills-Basic Calculation Methods-Squaring and Square RootingDefinition of square

Fundamental Skills-Basic Calculation Methods-Four Arithmetic OperationsDefinition of division

Principle:

Principle:

- 1. Definition of square: Square refers to the result of multiplying a number by itself, that is, a number multiplied by itself. For example, the square of 2 is 2×2=4. In mathematics, the symbol “²” is used to represent a square, for example, 2²=4
- 2. Non-negativity of square: The square of any real number is non-negative, that is, the square root does not have a negative solution (such as (-3)²=9≥0)
- 3. Parity of square: If a number is even, then its square is also an even number (such as 4²=16); if it is odd, its square is also an odd number (such as 5²=25)
- 4. Definition of square number: A square number is the square of an integer, which can be written in the form of a perfect square (such as 1=1², 4=2², 9=3², 16=4²)

- 1. The essential definition of division: the inverse operation of multiplication, that is, to find another factor given a product and a factor. If a×b=c, then c÷b=a (where b≠0), represented by "÷" or fraction line "/", such as a÷b or a/b, the divisor cannot be zero (undefined in mathematics) does not satisfy the commutative law (6÷2≠2÷6)
- 2. The concept of remainder: when it cannot be divided evenly, the remaining part is called the remainder (such as 7÷3=2 remainder 1)
- 3. Practical application: allocate resources , calculate the average. Easy to make mistakes: the remainder unit is not marked, ignore the divisibility condition
- 4. Reciprocal association: a÷b=a×(1/b), division is converted into multiplication by the reciprocal

- Figure 8: Examples of knowledge principles corresponding to specific knowledge points in MathBook

- (2).

- 1. Random event: An experiment satisfies the condition that it can be repeated under the same conditions, and all possible results are clearly known. If the results of each experiment are uncertain, then the experiment is a random experiment, such as shooting a basketball and flipping a coin. Results that may or may not appear in an experiment are called random events. Events that must occur in each experiment are called inevitable events, and events that must not occur are called impossible events, denoted as empty set.
- 2. Sample space: Each possible result of a random experiment is called a sample point, and the set of all sample points is called the sample space. Random events are always composed of several basic events.
- 3. …

Principle:

Probability and Statistics-Probability-Random Events -Definition of random events

Probability and Statistics

- 1. The principle of frequency estimation probability: approximate the true probability by the frequency of events in a large number of repeated experiments (the mathematical basis is the law of large numbers: when the number of experiments is ! → ∞, the frequency is

$!(&) → ((&)

- 2. Application scenarios: weather forecast (historical rainfall frequency predicts the probability of precipitation tomorrow), gambling game winning rate calculation (long-term statistics of roulette)
- 3. Formula example: The frequency of the number 6 appearing in 600 dice tossing is 98 times, and the estimated probability is () = $%%"# ≈ 0.163

Principle:

Probability and Statistics-Probability-Probability Calculation -Estimating probability using frequency

- 1. To complete a task with multiple steps, multiply the number of ways for each step (formula:

N = m1 * m2 * ... * mk, e.g., 4-digit password, each digit has 10 choices → 10^4 = 10,000). Key condition: steps must be ordered and independent (e.g., “choose shirt” then “choose pants”).

- 2. Application scenarios: password combinations (letters + numbers), travel routes (3 paths from A to B, 2 from B to C → total = 3×2 = 6)
- 3. Special cases: repetition allowed (e.g., repeated license plate letters), partial restrictions (e.g., phone number cannot start with 0). Question types: restricted permutations (e.g., 3digit numbers formed from 1–9 with no repeats)

Principle:

Probability and Statistics-Statistics-Principles of Counting-Addition and Multiplication Principles-Step-by-step multiplication counting principle

- 1. At least/at most problems: "at least m" is total combinations minus combinations less than m (e.g., “at least 3 qualified items” = total − C(n,0) − C(n,1) − C(n,2), formula:

C(n , k)at_least_m = Σ&'(! C(n , i), e.g., probability of at least 2 defective items in 10 = 1 − (C(10,0)+C(10,1)) / 210

- 2. Element distribution problems: same items to different people → use divider method (e.g., 10 identical balls to 3 people, each gets ≥1: C(9,2) = 36), allowing empty box: C(n+k−1,k−1) (e.g., 7 balls into 4 boxes: C(10,3) = 120)
- 3. Balls into boxes problem: distinguishable balls into distinguishable boxes (m choices per ball → mn); indistinguishable balls into distinguishable boxes (divider method); indistinguishable balls into indistinguishable boxes (integer partitions, e.g., Stirling numbers)

Principle:

Probability and Statistics-Statistics-Principles of Counting-Combinations -Combination-based counting problems

Figure 9: Examples of knowledge principles corresponding to specific knowledge points in MathBook

- (3).

###### Algebra

Algebra-Functions and Equations-Understanding Functions -Analytical expression of functions

Algebra-Functions and Equations-Fundamentals of Sequences -Definition and representation of sequences

Principle:

Principle:

- 1. Concept of analytical expression: A way to express a function using mathematical formula (e.g., uniform motion s = vt, direct proportionality y = kx); may be explicit (e.g., y = x³) or implicit (e.g., x² + y² = 1).
- 2. Types of expressions: Linear (e.g., y = 3x–2), quadratic (e.g., y = x²+2x+1), exponential (e.g., y = 2^x), logarithmic (e.g., y = ln x), trigonometric (e.g., y = A sin(ωx+φ)), piecewise (e.g., y = {x, x ≥ 0; –x, x < 0}).
- 3. Determining the expression: Given type, solve for parameters: use method of

- 1. Definition of a sequence: A sequence is a collection of numbers arranged in a specific pattern, where each number is called a term, denoted as a1, a2 , a3… (e.g., the natural number sequence 1, 2, 3, …)
- 2. Expression of terms: a1is the first term (initial term), anis the nth term (e.g., in the sequence 2, 4, 6, …, a1= 2, a2= 4)
- 3. Methods of representing a sequence: Listing method: directly listing finite terms (e.g., 1, 3, 5, 7, 9) or infinite terms (e.g., 1, 1/2, 1/3, …); General term formula: using an= f(n) (e.g., an= 2n - 1 generates the odd number sequence 1, 3, 5, …); Recursive formula: defining a term based on previous terms (e.g., Fibonacci sequence an= a{n-1}+ a{n-2}with initial values a1= 1, a2= 1)

undetermined coefficients (e.g., let y = kx + b, plug in (2,5) and (3,7) → solve for k = 2, b = 1); given graph, derive expression using vertex (e.g., parabola with vertex (1,–3) → y = a(x–1)²–3, then plug in a point to solve a), or asymptotes (e.g., hyperbola y = k/x → determine k); construct expression from conditions (e.g., profit model y = –x² + 50x – 100, or physical law y = A cos(ωt)).

Algebra-Logic-Necessary and Sufficient Conditions -Definition of true and false propositions

Algebra-Vectors-Concept of vectors

Principle:

Principle:

- 1. True propositions: Statements consistent with facts or logic (e.g., “In an isosceles triangle, the base angles are equal” – true, “2+2=4” – true, “The Earth is a planet” – true); False propositions: Statements that contradict facts or logic (e.g., “1 > 2” – false, “The Sun revolves around the Earth” – false, “All prime numbers are odd” – false because 2 is an even prime)
- 2. Truth of conditional propositions: In statements like “If A then B” (A → B), it is false only when A is true and B is false; in all other cases, it is true (e.g., “If it rains, the ground is wet” is false when it rains but the ground is dry)
- 3. Compound propositions: Formed by logical connectors like and, or, not (e.g., “2+2=4 and 3>5” is false, “x>1 or x<0” is true when x=2)
- 4. Role of counterexamples: A single counterexample disproves a universal statement (e.g., “All birds can fly” is disproved by the ostrich)

- 1. Definition of vector: a quantity that has both magnitude (modulus) and direction (such as velocity, force), as opposed to scalars (such as temperature, mass). Two elements of a vector: magnitude (such as velocity rate) and direction (such as the direction of force).
- 2. Representation of vector: geometrically represented by directed line segments (such as the vector from the starting point (") to the end point ($) is denoted by AB, algebraically represented by coordinates (such as two-dimensional vector (( =

(3,4)). Symbols of vectors: bold lowercase letters (such as () or symbols with arrows (such as *), zero vector is denoted by *) (direction is arbitrary, modulus is 0)

- 3. Basics of vector operations: addition (parallelogram law or triangle law), subtraction (adding opposite vectors), scalar multiplication (changing modulus, direction may be reversed)

Algebra-Linear Algebra-Vectors and Matrices-Matrices -Definition of matrices

Algebra- Graph Theory-Tree -Basic concepts of trees

Principle:

Principle:

- 1. Definition of a tree: A tree is an undirected graph that is connected. There are no cycles in a tree. When a tree has n vertices, it has exactly (n-1) edges.
- 2. Basic terms for a tree: Root, the starting vertex of the tree, has no parent vertex. Parent node, the direct predecessor of a vertex. Child node, the direct successor of a vertex. Sibling node, a vertex with the same parent node. Ancestor, all vertices on the path from the root to the vertex. Descendants, all vertices from the vertex to the leaves. Leaves, vertices with no child nodes. Internal nodes, non-leaf nodes. Depth, the length

of the path from the root to a vertex. Height, the length of the path from a vertex to the farthest leaf.

- 3. …

- 1. In linear algebra, a matrix is a rectangular array of numbers, which can be real numbers or complex numbers. Matrices are usually denoted by capital letters, such as A. Each number in a matrix is called an element or entry of the matrix.
- 2. The size of a matrix is determined by its number of rows and columns, typically expressed as m×n, where m is the number of rows and n is the number of columns. An m×n matrix is a rectangular array consisting of m rows and n columns, with a number at each position. A matrix can be expressed as A=(aij)m×n, where aij is an element of the matrix, i represents the row index, and j represents the column index.
- 3. …

- Figure 10: Examples of knowledge principles corresponding to specific knowledge points in MathBook (4).

- C.2 MathBook-Standard

- C.2.1 GeoGebra-based Diagram Generation.

As described in the main text, all diagrams in MathBook-Standard are rendered using GeoGebra, a dynamic mathematics software that enables precise and reproducible geometric constructions. GeoGebra supports both interactive design and programmatic generation of diagrams, making it highly suitable for large-scale dataset creation. In our workflow, each problem is paired with a high-quality diagram constructed in GeoGebra, ensuring mathematical rigor and visual clarity. For automated and scalable generation, we leverage GeoGebra’s ability to encode diagrams as scripts (e.g., XML, as shown in Figure 11), which allows for efficient parameter variation and reproducibility, but the core advantage lies in GeoGebra’s expressive power and accuracy for mathematical figures. Compared to general-purpose plotting libraries such as Python’s matplotlib, GeoGebra offers richer geometric primitives and more precise control over mathematical relationships, supporting a broader range of problem types and visual styles. As shown in Figures 12, 13, and 14, diagrams generated by GeoGebra exhibit higher fidelity and better alignment with mathematical conventions, which is essential for both algorithmic evaluation and educational use. It is evident that the complexity and precision of these diagrams would be difficult to achieve using Python-based plotting tools alone.

- C.2.2 Dataset Diversity and Variant Construction.

Building on the GeoGebra-based pipeline, we systematically construct a diverse dataset as detailed in the main text. For each knowledge point and principle, a seed problem is designed with a corresponding diagram. To further enhance diversity, we introduce two types of variants: oneproblem-multi-image (generating multiple diagram instances for the same problem by varying

[Figure 68]

GGB

<command name="Polygon"> <input a0="A" a1="B" a2="C" a3="D" /> <output a0="PolygonABCD" a1="AB" a2="BC" a3="CD" a4="AD" />

</command> <command name="Prism">

XML

<input a0="PolygonABCD" a1="5" /> <output a0="Prism" a1="E" a2="F" a3="G" a4="H" a5="faceABFE" a6="faceBCGF"

a7="faceCDHG" a8="faceADHE" a9="faceEFGH" a10="AE" a11="edgeBF" a12="CG" a13="edgeDH" a14="edgeEF" a15="edgeFG" a16="edgeGH" a17="edgeEH" /> </command>

- Figure 11: Overview of the GeoGebra interface and part of the corresponding XML script, showing core commands for defining geometric objects.

parameters in GeoGebra) and one-image-multi-problem (curating multiple questions for a single diagram, each derived from different knowledge points or mathematical principles). Representative examples of seed problems and their variants are shown in Figure 15– 22, demonstrating the flexibility and extensibility of our approach.

By leveraging GeoGebra’s capabilities, MathBook-Standard achieves both high-fidelity geometric representation and systematic dataset expansion, ensuring rich semantic and visual diversity for mathematical reasoning tasks.

###### Examples (1)

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

- Figure 12: Overview of a group of GeoGebra-generated images in MathBook (1).

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Examples (2)

- Figure 13: Overview of a group of GeoGebra-generated images in MathBook (2).

- Examples (3)
- Figure 14: Overview of a group of GeoGebra-generated images in MathBook (3).

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

###### Example 1

[Figure 97]

Seed

[Figure 98]

How many triangles are there in the picture?

One-Image-Multi-Problem

[Figure 99]

- 1. How many triangles are there in the picture?
- 2. How many line segments are there in the picture?

[Figure 100]

3. How many angles are there in the picture?

One-Problem-Multi-Image

How many triangles are there in the picture?

[Figure 101]

[Figure 102]

[Figure 103]

Figure 15: An example of MathBook-Standard data instance (1).

[Figure 104]

###### Example 2

[Figure 105]

Seed

[Figure 106]

[Figure 107]

Compare the magnitudes of the fractions represented by the shaded parts in circles A and B.

One-Image-Multi-Problem

- 1. What is the difference between the fractions indicated by the shaded areas of Circle B and Circle A?
- 2. What is the sum of the fractions indicated by the shaded areas of Circle B and Circle A?

[Figure 108]

[Figure 109]

[Figure 110]

3. Compare the size of the fractions represented by the blank parts in circle A and circle B.

One-Problem-Multi-Image

Compare the magnitudes of the fractions represented by the shaded parts in circles A and B.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Figure 16: An example of MathBook-Standard data instance (2).

###### Example 3

[Figure 119]

Seed

[Figure 120]

What three-dimensional geometric solid is generated by rotating the figure about the dotted line?

One-Image-Multi-Problem

1. What three-dimensional geometric solid is generated by rotating the figure about the dotted line?

[Figure 121]

[Figure 122]

2. What type of figure represents the side profile of a solid generated by rotation?

3. What is the shape of the base of the solid generated by rotation?

One-Problem-Multi-Image

What three-dimensional geometric solid is generated by rotating the figure about the dotted line?

[Figure 123]

[Figure 124]

[Figure 125]

Figure 17: An example of MathBook-Standard data instance (3).

[Figure 126]

###### Example 4

[Figure 127]

Seed

[Figure 128]

What is the Angle between the two straight lines in the figure?

One-Image-Multi-Problem

[Figure 129]

- 1. What is the slope of the straight line l1?
- 2. What type of triangle is formed by the two lines and the y-axis in the figure?

[Figure 130]

3. What is the intercept of the straight line l1?

One-Problem-Multi-Image

What is the Angle between the two straight lines in the figure?

[Figure 131]

[Figure 132]

[Figure 133]

Figure 18: An example of MathBook-Standard data instance (4).

[Figure 134]

###### Example 5

[Figure 135]

[Figure 136]

Seed

What are the in-degree and out-degree of vertex B respectively?

One-Image-Multi-Problem

[Figure 137]

- 1. What are the in-degree and out-degree of vertex A respectively?
- 2. What is the length of the shortest path from C to B?

[Figure 138]

3. Does a cycle exist in the figure?

One-Problem-Multi-Image

What are the in-degree and out-degree of vertex B respectively?

[Figure 139]

[Figure 140]

[Figure 141]

Figure 19: An example of MathBook-Standard data instance (5).

[Figure 142]

###### Example 6

[Figure 143]

Seed

[Figure 144]

Determine the monotonically increasing interval of the function in the graph.

One-Image-Multi-Problem

- 1. Determine the monotonically decreasing interval of the function in the graph.
- 2. Determine the zero point of the function in the graph.

[Figure 145]

[Figure 146]

3. Determine the symmetry points of the functions in the graph.

One-Problem-Multi-Image

Determine the monotonically increasing interval of the function in the graph.

[Figure 147]

[Figure 148]

[Figure 149]

Figure 20: An example of MathBook-Standard data instance (6).

[Figure 150]

###### Example 7

[Figure 151]

Seed

[Figure 152]

Divide the circles in the following picture into 3 equal parts. How many are there in each part?

One-Image-Multi-Problem

- 1. Divide the circles in the following picture into 2 equal parts. How many are there in each part?
- 2. If the circles are arranged in a rectangular grid with 3 rows, how many columns are there?

[Figure 153]

[Figure 154]

3. If all circles are tightly packed into a square shape, how many circles are on the outermost layer?

One-Problem-Multi-Image

Divide the circles in the following picture into 3 equal parts. How many are there in each part?

[Figure 155]

[Figure 156]

[Figure 157]

Figure 21: An example of MathBook-Standard data instance (7).

[Figure 158]

###### Example 8

[Figure 159]

Seed

[Figure 160]

The vector OA can be decomposed into the vectors i and j shown in the figure. What are the coordinates of vector OA?

One-Image-Multi-Problem

[Figure 161]

[Figure 162]

- 1. What is the abscissa of vector OA?
- 2. What is the ordinate of vector OA?

3. What is the modulus of vector OA?

One-Problem-Multi-Image

The vector OA can be decomposed into the vectors i and j shown in the figure. What are the coordinates of vector OA?

[Figure 163]

[Figure 164]

[Figure 165]

Figure 22: An example of MathBook-Standard data instance (8).

##### C.3 MathBook-Pro

- As shown in Figure 23, we present a concrete example from MathBook-Pro to illustrate the construction and expansion of problem variants within the three-dimensional difficulty space. The seed problem, positioned at the origin, focuses on the arc length formula in plane geometry, and involves knowledge points such as definition of angle, definition of circles, four arithmetic operations of integers and four arithmetic operations of fractions. We first demonstrate how the seed problem is expanded along each individual dimension:

Step Complexity: The number of required knowledge points is increased by introducing new intermediate conclusions as conditions. In this example, the extended variant requires not only the arc length formula, but also incorporates circumference of a circle and area of a circle as additional knowledge points. The solution to the new problem depends on the answer to the seed problem, reflecting a progressive deepening of reasoning.

Visual Complexity: The original diagram is enhanced by adding shaded regions, which increases the visual and interpretive demands while keeping the core mathematical focus unchanged.

Contextual Complexity: The problem statement is recontextualized from a direct geometric description to a real-world application scenario. Although the narrative becomes more complex, the essential assessment of the arc length formula remains at the core.

By systematically combining these single-dimension expansions, we further generate multidimensional variants that integrate increased step, visual, and contextual complexity. In total, starting from the seed problem, we construct 7 variants corresponding to all possible combinations of the three dimensions. Each new variant is constructed through progressive modifications to both the problem statement and the accompanying image, resulting in a diverse and interpretable set of difficulty-controlled problems.

[Figure 166]

Step Complexity

[Figure 167]

[Figure 168]

[Figure 169]

Find the area of a circle whose circumference is equal to the length of arc CD.

As shown in the figure, arc CD is the fence of a circular farm. Now this fence is to be used to enclose a regular hexagonal flower bed. What is the area of the flower bed?

[Figure 170]

[Figure 171]

Find the area of the shaded region.

For the farm shown in the figure, if the shaded area needs to be fully paved with tiles, how much tile area should be purchased in advance?

[Figure 172]

[Figure 173]

Contextual Complexity

[Figure 174]

What is the arc length of CD?

[Figure 175]

[Figure 176]

As shown in the figure, arc CD is the fence of a circular farm. Now the fence needs to be rebuilt. What length of fencing material should be purchased?

Find the length of the arc corresponding to the shaded sector in the figure.

[Figure 177]

[Figure 178]

As shown in the figure, arc CD is the fence of a circular farm. Find the length of the arc fence corresponding to the shaded area.

Visual Complexity

Figure 23: An example from MathBook-Pro in the difficulty space.

- C.4 MathBookEval

- C.4.1 Dataset Construction and Annotation Protocol.

To ensure both comprehensive knowledge coverage and rigorous, interpretable annotations for visual mathematical reasoning, MathBookEval is constructed through a multi-stage, process-oriented pipeline. We begin by integrating representative samples collected from five open-source benchmarks: MathVista [32], MathVerse [71], MathVision [57], We-Math [43] and DynaMath [77], systematically filtering out redundant or highly similar items to maximize diversity in knowledge point combinations and reasoning patterns. All problems are re-annotated under unified guidelines, ensuring consistency in annotation style and granularity. For each problem, at least two human experts independently provide a complete, step-by-step solution, where each step is explicitly decomposed according to the underlying knowledge point(s) from the unified knowledge system K. This knowledge-point-based decomposition is fundamental: it enables precise and systematic quantification of reasoning depth, as each reasoning step directly corresponds to a specific knowledge point. Only those problems for which the set of annotated knowledge points at each step is exactly consistent across expert annotations are retained in the final dataset, ensuring high reliability and objectivity. To address knowledge points and reasoning depths insufficiently covered by existing benchmarks, additional samples are newly constructed by human experts following the same process-oriented, knowledgepoint-level annotation protocol. As a result, MathBookEval comprises both collected samples from open-source benchmarks and newly constructed samples, achieving balanced and comprehensive coverage across mathematical domains and reasoning complexities, with every annotation step tightly aligned to the relevant knowledge points for robust reasoning depth modeling.

- C.4.2 Task Dimensions.

MathBookEval is organized along two dimensions, capturing both the diversity of mathematical knowledge and the depth of reasoning required for problem solving.

- (1) Reasoning Dimension: Problems are categorized by the number of reasoning steps required to reach the solution. Each step is explicitly defined and directly mapped to a specific knowledge point in the unified knowledge system K. We define three levels: Level 1 (1–3 steps, basic reasoning), Level 2 (4–6 steps, intermediate reasoning), and Level 3 (7–10 steps, complex reasoning). This step-by-step annotation based on knowledge points allows for clear and objective quantification of reasoning depth, enabling detailed analysis of model performance across different levels of reasoning complexity.
- (2) Knowledge Dimension: The 491 knowledge points in K are distributed across 4 top-level domains and 13 subdomains, all of which are covered in the benchmark. Each problem is annotated with all the knowledge points involved in its solution, and every reasoning step is aligned with a specific knowledge point. This enables fine-grained evaluation of model capabilities across various mathematical topics and educational stages.

- C.4.3 Dataset Statistics.

Table 6 summarizes the key statistics of MathBookEval. The benchmark contains 1,000 fully annotated problems, covering all 491 knowledge points in the unified knowledge system. Of these, 600 are sourced from open-source benchmarks and 400 are newly constructed to address coverage gaps and increase diversity. Problems are distributed across multiple formats, including multiplechoice and fill-in-the-blank, and span a wide range of reasoning depths and knowledge domains.

- As shown in Figure 24, we compare MathBookEval and existing benchmarks. In the right panel, the y-axis represents the percentage of problems at each reasoning level. Note that for some benchmarks, the total does not reach 100% because we exclude problems that experts annotate as belonging to other subjects such as physics, chemistry, or biology, in order to ensure a rigorous comparison. It is evident from the figure that existing benchmarks contain less than 3% of problems at Level 2 (4–6 steps), and none at Level 3 (7–10 steps). In contrast, MathBookEval substantially supplements these two categories, providing a more comprehensive evaluation of multi-step reasoning abilities.

Table 6: Key statistics of MathBookEval.

Statistics Number Total Problems 1,000

- - Open-source Benchmarks 600

- - MathVista 150
- - MathVerse 150
- - MathVision 100
- - We-Math 100
- - DynaMath 100

- - Newly Constructed 400

Knowledge Points Covered 491

- - Domains 4
- - Subdomains 13

Reasoning Depth

- - Level 1 (1–3 steps) 62.0%
- - Level 2 (4–6 steps) 30.2%
- - Level 3 (7–10 steps) 7.8%

[Figure 179]

[Figure 180]

Figure 24: Comparison of MathBookEval and open-source benchmarks

##### C.4.4 Evaluation Protocol and Metrics.

MathBookEval incorporates existing evaluation protocols, including both LLM-as-a-judge and rulebased approaches. To ensure consistency, MathBookEval adopts the LLM-as-a-judge protocol as the evaluation rule. Specifically, we adopt the LLM-as-a-judge protocol, following MathVista and MathVerse by employing GPT-4o as the judge model and reporting overall accuracy as the evaluation metric. The detailed prompt used for evaluation is shown in Table 7.

Table 7: Prompt templates for evaluation on MathBookEval.

Type Prompt Template

Now, we require you to solve a math question. Please briefly describe your thought process and provide the final answer. For multiple-choice questions, return the selected option and its content. For direct answer selection, return only the chosen result. For fill-in-the-blank questions, answer directly. Question: <Question> Regarding the format, please answer following the template below, and be sure to include two <> symbols: <Thought process>: <<your thought process>> <Answer>: <<your answer>>

Evaluation Prompt

Table 8: The release time and model source of MLLMs used in MathBookEval

Model Release Time Source GPT-4o [37] 2024-05 https://gpt4o.ai/ GPT-4V [38] 2024-04 https://openai.com/index/gpt-4v-system-card/ InternVL2.5-78B [7] 2024-12 https://huggingface.co/OpenGVLab/InternVL2_5-78B

- InternVL2.5-8B [7] 2024-12 https://huggingface.co/OpenGVLab/InternVL2_5-8B Qwen2.5-VL-72B [4] 2025-01 https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct Qwen2.5-VL-7B [4] 2025-01 https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct Qwen2.5-VL-3B [4] 2025-01 https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct LLaVA-OneVision-72B [27] 2024-08 https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov-chat LLaVA-OneVision-7B [27] 2024-08 https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov GLM-4V-9B [17] 2024-06 https://huggingface.co/THUDM/glm-4v-9b

Table 9: Model architecture of 10 MLLMs evaluated on MathBookEval. Models LLM Vision Encoder

GPT-4o - GPT-4V - InternVL2.5-78B Qwen2.5-72B-Instruct InternViT-6B-448px-V2_5 InternVL2.5-8B internlm2_5-7b-chat InternViT-6B-448px-V2_5 Qwen2.5-VL-72B Qwen2.5-72B-Instruct CLIP ViT-bigG-P14 Qwen2.5-VL-7B Qwen2.5-7B-Instruct CLIP ViT-bigG-P14 Qwen2.5-VL-3B Qwen2.5-3B-Instruct CLIP ViT-bigG-P14 LLaVA-OneVision-72B Qwen2-72B SigLip-so400m-P14-384 LLaVA-OneVision-7B Qwen2-7B SigLip-so400m-P14-384 GLM-4V-9B GLM-9B EVA_02_CLIP-E-P14

##### C.4.5 Experiment Setup

Details of the Evaluated Models. To evaluate the performance of various multimodal large models on mathematical tasks, we include a diverse set of recent models in our benchmark. Table 8 lists the release dates and official sources of all evaluated models. Additionally, Table 9 provides an overview of their architectural designs to support a comprehensive comparison.

Details of the Model Hyperparameters. For all closed-source models accessed via API, we adopt the standard generation settings and perform inference on CPUs, with the process typically completing within a day. For open-source models, inference is conducted on a cluster equipped with 8 NVIDIA A800-SXM4-80GB GPUs, using the hyperparameter configurations provided in the official inference examples. If no specific instructions are available, default settings are applied. The detailed generation parameters are summarized in Table 10 and Table 11.

##### C.4.6 Additional Results on MathBookEval

As shown in Figure 28 to Figure 37, we present the performance of different models on various subdomains in MathBookEval, where subdomains belonging to the same domain are indicated with the same color. It can be observed that models generally perform worse on geometry-related problems, especially in subdomains such as solid geometry and analytic geometry, which involve higher visual complexity and require more advanced reasoning. In contrast, models tend to achieve better results on algebra and fundamental skills, particularly in subdomains related to computational methods.

- Table 10: Generating parameters for Open-Source MLLMs.

|Model<br><br>|Generation Setup|
|---|---|
|InternVL2.5-78B<br><br>|do_sample = False, max_new_tokens = 1024|
|InternVL2.5-8B|do_sample = False, max_new_tokens = 1024<br><br>|
|Qwen2.5-VL-72B<br><br>|do_sample = False, max_new_tokens = 1024|
|Qwen2.5-VL-7B|do_sample = False, max_new_tokens = 1024<br><br>|
|Qwen2.5-VL-3B|do_sample = False, max_new_tokens = 1024<br><br>|
|LLaVA-OneVision-72B<br><br>|do_sample = False, max_new_tokens = 1024|
|LLaVA-OneVision-7B|do_sample = False, max_new_tokens = 1024|
|GLM-4V-9B|do_sample = False|

- Table 11: Generating parameters for Closed-Source MLLMs.

Model Generation Setup GPT-4o "model" : "gpt-4o", "temperature" : 0, "max_tokens" : 1024 GPT-4V "model" : "gpt-4-turbo", "temperature" : 0, "max_tokens" : 1024

### D More Details on MathBook-RL

##### D.1 Implementation details

We use Qwen2.5-VL-7B-Instruct as the base model and conduct all experiments on 8×A800 GPUs. The training process consists of two stages.

In the first stage, we perform cold-start supervised fine-tuning (SFT) to help the model develop explicit awareness of knowledge system and a knowledge-driven reasoning paradigm. The SFT stage uses a learning rate of 1.0 × 10−5, is trained for 1 epoch, and adopts a warmup ratio of 0.1.

In the second stage, we apply dynamic reinforcement learning (RL) to further improve the model’s generalization ability. For RL, we set the rollout temperature to 1.0 and generate 8 rollouts per sample. The learning rate is set to 1 × 10−6, and the maximum completion length is 1024 tokens. The system prompt used in this stage is illustrated in Table 12. The reward function combines answer accuracy (weight 0.9) and response format compliance (weight 0.1). Specifically, we use MathVerify to extract and compare answers for accuracy, while format compliance ensures the output follows the required structure.

Table 12: The system prompt template for response generation in the RL stage.

Type Prompt Template

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first outputs the thinking process in <think> </think> and then provides the final answer(number, option, phrase, or LaTeX expression as appropriate) in <answer> </answer> tags. User: {question} Assistant:

System Prompt

##### D.2 Case Study

To further illustrate the strengths of our approach, we present several representative case studies comparing our model with the Qwen2.5-VL-7B baseline across different benchmarks.

- Case 1: Conciseness and Reasoning Process. Figure 25 compares the response patterns of our model and the Qwen2.5-VL-7B baseline on We-Math. Our model produces more concise answers,

with a reduced average response length, while still retaining all necessary formulas and reasoning steps. This demonstrates that our approach effectively mitigates the issue of overthinking, resulting in more focused and efficient solutions without sacrificing completeness or mathematical rigor.

- Case 2: Spatial Reasoning Enhancement. Figure 26 highlights the performance of our model on MathVision. Compared to the baseline, which often fails to correctly interpret or solve such problems, our model (MathBook-7B) shows clear improvements in spatial reasoning. This is particularly evident in questions requiring the understanding of geometric relationships or positional logic, indicating that our training strategy significantly enhances the model’s ability to handle spatially complex scenarios.
- Case 3: Knowledge-Oriented and Context-Aware Mathematical Reasoning. In Figure 27, we compare the responses of our model and the Qwen2.5-VL-7B baseline on MathVista. Our model not only applies the relevant mathematical concepts correctly but also demonstrates improved integration of mathematical knowledge with real-world problem contexts. This case exemplifies the model’s strengthened ability to bridge abstract mathematical reasoning and practical application, reflecting the benefits of our knowledge-oriented and context-aware training paradigm.

Overall, these case studies provide qualitative evidence that our model achieves more concise, accurate, and contextually appropriate reasoning compared to strong baselines, particularly in scenarios demanding spatial understanding and knowledge-oriented problem solving.

Comparison of MathBook-7B and Qwen2.5-VL-7B on We-Math

Question:

Now, we require you to solve a multiple-choice math question. Please briefly describe your thought process and provide the final answer(option). Question: As shown in the diagram, what is the arc length of this sector? Options: A. 2π; B. 4π;C. 8π; D. No correct answer Regarding the format, please answer following the template below, and be sure to include two <> symbols:<Thought process>: <<your thought process>> <Answer>: <<your option>>

#### 4 90°

###### Response from Qwen2.5-VL-7B:

<Thought process>: The diagram shows a quarter circle (90 degrees) with a radius of 4 units. The formula for the arc length of a sector is (θ/360) * 2πr, where θ is the angle in degrees and r is the radius. For a 90-degree sector, the arc length is (90/360) * 2π * 4 = (1/4) * 2π * 4 = π * 4 = 4π. Therefore, the arc length of this sector is 4π.

[Figure 181]

<Answer>: B.4π

###### Response from MathBook-7B:

<Thought process>: The arc length of a sector is calculated using the formula \\( \\frac{\\theta}{360} \\times 2\\pi r \\), where \\( \\theta \\) is the angle in degrees and \\( r \\) is the radius. Here, \\( \\theta = 90^\\circ \\) and \\( r = 4 \\). Plugging in the values, we get \\( \\frac{90}{360} \\times 2\\pi \\times 4 = \\frac{1}{4} \\times 8\\pi = 2\\pi \\). Thus, the arc length is \\( 2\\pi \\). <Answer>: A. 2π

[Figure 182]

Figure 25: Comparison of MathBook-7B and Qwen2.5-VL-7B on We-Math.

- D.3 Experiment Setup

- D.3.1 Details of the Evaluation.

We evaluate our model on four representative benchmarks: MathVista [32], MathVerse [71], MathVision [57], and We-Math [42]. During evaluation, we strictly follow the official scoring protocols provided in the respective benchmark GitHub repositories to ensure fair and consistent comparison.

Comparison of MathBook-7B and Qwen2.5-VL-7B on MathVision

[Figure 183]

Question:

What do you see if you look at the tower, which is made up of two building blocks, exactly from above?

- <image1>
- <image2>

###### Response from Qwen2.5-VL-7B:

To determine what you would see if you looked at the tower from above, we need to analyze the shapes of the building blocks that make up the tower.

- 1. The tower consists of two building blocks: a cylinder and a cone.
- 2. When viewed from above, the cylinder and the cone will appear as their respective twodimensional shapes.

- - The cylinder, when viewed from above, appears as a circle.
- - The cone, when viewed from above, appears as a triangle. Since the question asks for the shape you would see from above, we need to identify the shape that corresponds to the top view of the cone. The options provided are: (A) Circle (B) Triangle (C) Square (D) Triangle (repeated) (E) Square The correct answer is the shape that represents the top view of the cone, which is a triangle. Therefore, the answer is \boxed{B}.

[Figure 184]

###### Response from MathBook-7B:

<think>The top view of the tower, which consists of a cone and a cylinder, is a circle. This is because the top surface of the cone is a circle, and the top surface of the cylinder is also a circle. Therefore, the top view is a circle.</think><answer>: A</answer>

[Figure 185]

Figure 26: Comparison of MathBook-7B and Qwen2.5-VL-7B on MathVision.

Specifically, we report results on the testmini split for MathVista, MathVerse, MathVision, and WeMath. For We-Math, we adopt the main evaluation metric as defined in the original paper, reporting “Score (Strict)” as the primary metric. For the other benchmarks, we report the average accuracy as the main evaluation result.

##### D.3.2 Details of the Baselines

We compare our method with a comprehensive set of baselines from three perspectives: closed-source models (e.g., GPT-4o [37], Gemini-1.5-Pro [51]), open-source general models (e.g., Qwen2.5VL-7B [4], InternVL2.5-8B [8]), and open-source reasoning models (e.g., Math-PUMA-7B [75], URSA-8B [35], R1-Onevision-7B [66], R1-VL-7B [70], MM-Eureka-7B [36]), which enables a thorough and multi-faceted comparison to highlight the advantages of our approach.

GPT-4o [37]: GPT-4o (“o” for omni) is OpenAI’s 2024 flagship multilingual, multimodal large language model that accepts text, images, and audio as input for unified cross-modal understanding. It is designed for broad adaptability and seamless integration of visual and linguistic information, supporting complex reasoning across modalities and languages.

Gemini-1.5-Pro [51]: Gemini 1.5 Pro, developed by Google DeepMind, is a multimodal model capable of processing text, images, audio, and video inputs, with an extremely long context window (up to 1–2 million tokens). It is optimized for multi-task proficiency and structured tool integration, enabling analysis of lengthy and diverse content.

Qwen2.5-VL-7B [4]: Qwen2.5-VL-7B is an open-source vision-language model with 7 billion parameters, designed to generate both free-form text and structured outputs such as bounding boxes

Comparison of MathBook-7B and Qwen2.5-VL-7B on MathVista

Question:

[Figure 186]

Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: An administrator at the Department of Motor Vehicles (DMV) tracked the average wait time from month to month. According to the table, what was the rate of change between August and September? (Unit: minutes per month)

###### Response from Qwen2.5-VL-7B:

<Thought process>: To find the rate of change between August and September, we need to calculate the difference in waiting time and divide it by the number of months between August and September. The waiting time in August is 17 minutes. The waiting time in September is 14 minutes. The difference in waiting time is: 17 minutes (August) - 14 minutes (September) = 3 minutes Since there is only one month between August and September, the rate of change is: 3 minutes / 1 month = 3 minutes per month Therefore, the rate of change between August and September is 3 minutes per month.

[Figure 187]

###### Response from MathBook-7B:

<think>To find the rate of change between August and September, we calculate the difference in waiting times and divide by the number of months between them. The waiting time in August is 17 minutes, and in September it is 14 minutes. The change is \\( 17 - 14 = 3 \\) minutes. Since the time decreased, the rate of change is \\(-3\\) minutes per month.</think> <answer>-3</answer>

[Figure 188]

Figure 27: Comparison of MathBook-7B and Qwen2.5-VL-7B on MathVista.

and JSON. It emphasizes fine-grained visual understanding and multi-modal alignment, supporting tasks like document analysis and event detection.

- InternVL2.5-8B [8]: InternVL2.5-8B is an 8B-parameter open-source multimodal model that employs progressive scaling and co-training strategies to align its vision and language components. It incorporates training optimizations and a curated dataset to enhance cross-modal reasoning and reduce hallucinations.

Math-PUMA-7B [75]: Math-PUMA-7B is a vision-language model focused on mathematical reasoning with visual inputs, introducing a three-stage curriculum for aligning textual and visual modalities. The model is optimized for visual math benchmarks and aims to ensure consistent problem-solving across formats such as text and diagrams.

URSA-8B [35]: URSA-8B is an 8B-parameter multimodal model targeting chain-of-thought reasoning in visual mathematical problems, trained on large-scale multimodal CoT datasets. It employs a reward model for stepwise verification, emphasizing both the generation and validation of reasoning chains for reliable solutions.

R1-Onevision-7B [66]: R1-Onevision-7B is a 7B-parameter multimodal reasoning model that converts images into structured textual representations for symbolic reasoning. It is trained on step-by-step multimodal reasoning annotations and refined with reinforcement learning, enabling precise multi-hop visual-textual inference.

R1-VL-7B [70]: R1-VL-7B is an open-source 7B vision-language model designed for stepwise reasoning, applying Step-wise Group Relative Policy Optimization (StepGRPO) for dense intermedi-

ate rewards. This approach improves logical coherence and multi-step problem-solving, especially in mathematical and logical tasks.

MM-Eureka-7B [36]: MM-Eureka-7B is a vision-language model based on Qwen2.5-VL-7B, fine-tuned with the MMK12 dataset and a rule-based reinforcement learning strategy. It is designed for multidisciplinary visual reasoning in math and science, using rule-based rewards to guide the learning of complex reasoning steps.

[Figure 189]

- Figure 28: Detailed performance of GPT-4o across 13 subdomains.

[Figure 190]

- Figure 29: Detailed performance of GPT-4V across 13 subdomains.
- Figure 30: Detailed performance of InternVL2.5-78B across 13 subdomains.

[Figure 191]

[Figure 192]

- Figure 31: Detailed performance of Qwen2.5-VL-72B across 13 subdomains.

[Figure 193]

- Figure 32: Detailed performance of LLaVA-OneVision-72B across 13 subdomains.
- Figure 33: Detailed performance of InternVL2.5-8B across 13 subdomains.

[Figure 194]

- Figure 34: Detailed performance of Qwen2.5-VL-7B across 13 subdomains.

[Figure 195]

[Figure 196]

- Figure 35: Detailed performance of LLaVA-OneVision-7B across 13 subdomains.
- Figure 36: Detailed performance of GLM-4V-9B across 13 subdomains.

[Figure 197]

- Figure 37: Detailed performance of Qwen2.5-VL-3B across 13 subdomains.

[Figure 198]

