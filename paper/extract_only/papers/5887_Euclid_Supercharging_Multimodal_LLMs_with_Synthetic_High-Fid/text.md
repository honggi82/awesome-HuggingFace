# arXiv:2412.08737v1[cs.CV]11Dec2024

## Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions

###### Jiarui Zhang♢, Ollie Liu♢, Tianyu Yu♠, Jinyi Hu♠, Willie Neiswanger♢

♢University of Southern California, ♠Tsinghua University

Multimodal large language models (MLLMs) have made rapid progress in recent years, yet continue to struggle with low-level visual perception (LLVP)—particularly the ability to accurately describe the geometric details of an image. This capability is crucial for applications in areas such as robotics, medical image analysis, and manufacturing. In this paper, we first introduce Geoperception, a benchmark designed to evaluate an MLLM’s ability to accurately transcribe 2D geometric information from an image. Using this benchmark, we demonstrate the limitations of leading MLLMs, and then conduct a comprehensive empirical study to explore strategies for improving their performance on geometric tasks. Our findings highlight the benefits of certain model architectures, training techniques, and data strategies, including the use of high-fidelity synthetic data and multi-stage training with a data curriculum. Notably, we find that a data curriculum enables models to learn challenging geometry understanding tasks which they fail to learn from scratch. Leveraging these insights, we develop Euclid, a family of models specifically optimized for strong low-level geometric perception. Although purely trained on synthetic multimodal data, Euclid shows strong generalization ability to novel geometry shapes. For instance, Euclid outperforms the best closed-source model, Gemini-1.5-Pro, by up to 58.56% on certain Geoperception benchmark tasks and 10.65% on average across all tasks.

Website: euclid-multimodal.github.io

Model Weights & Datasets: huggingface.co/euclid-multimodal

Code Repository: github.com/euclid-multimodal/Euclid

#### 1. Introduction

Multimodal large language models (MLLMs) have rapidly progressed in recent years, demonstrating remarkable potential in understanding and reasoning about the visual world through the powerful capabilities of large language models (LLMs) (Liu et al., 2024c,a, Achiam et al., 2023, Team et al., 2023, Hu et al., 2023, Tong et al., 2024a, Wang et al., 2024a). These models have showcased strong performance in tasks such as visual question answering (VQA) (Goyal et al., 2017), image captioning (Lin et al., 2014), and multimodal reasoning (Liu et al., 2023). As one recent example, LLaVA-NeXT-34B (Liu et al., 2024b) achieves an impressive 83.7% accuracy on the VQAv2 benchmark (Goyal et al., 2017), a comprehensive benchmark on natural image question answering.

While MLLMs achieve impressive results on tasks like VQA, their performance relies on high-level semantic extraction (Tong et al., 2024b); in contrast, they often fall short on low-level visual perception (LLVP)—i.e., the ability to accurately describe the geometric details of an image, such as the points, lines, angles, shapes, and spatial relationships among its constituent objects. This limitation becomes especially apparent in tasks requiring precise descriptions, such as mathematical visual problem solving (Zhang et al., 2024a, Lu et al., 2023), scientific visual understanding (Yue et al., 2024, Fu et al., 2024a), abstract visual reasoning (Jiang et al., 2024, Ahrabian et al., 2024), and even simple visual comprehension (Rahmanzadehgervi et al., 2024, Wang et al., 2024b). For example, when interpreting a graph diagram, precise recognition of edges is essential for extracting reliable information, and in geometry problem-solving, accurate identification of

Corresponding author(s): Jiarui Zhang, jzhang37@usc.edu; Willie Neiswanger, neiswang@usc.edu

relationships between line segments and points is fundamental (Fu et al., 2024a). Beyond abstract tasks, LLVP is also vital in real-world applications, including spatial understanding for robotics, medical image analysis for accurate diagnosis, quality control in manufacturing to detect subtle defects, autonomous driving systems that rely on exact object localization or distance estimation, and augmented reality applications that demand precise overlay of virtual objects onto the real world.

In this paper, we aim to study the challenges of LLVP in MLLMs, take steps to understand the root cause of their performance, and improve the models’ capabilities in this area. We begin by developing a benchmark dataset specifically designed to evaluate precise geometric perception, which we call Geoperception. As a focused test bed, this benchmark targets 2D geometry tasks. Using this benchmark, we demonstrate the limitations of leading closed and open MLLMs, followed by a comprehensive empirical study to explore strategies for significantly improving MLLM’s performance on geometric perception tasks. Our findings show the benefits of key factors such as model architecture, training techniques, and data strategies, including the use of synthetic data and multi-stage training with a data curriculum. Notably, we find that a data curriculum enables models to learn challenging geometry LLVP tasks, which they fail to learn from scratch, even when trained on a very large dataset. Using these lessons learned, we then train a family of models—using a carefully designed curriculum of synthetic data—that are specifically optimized for strong LLVP, which we call Euclid. We evaluate this family of models, and show that it excels on a variety of low-level geometric perception tasks.

Our main technical contributions are as follows:

- • Geoperception Benchmark: We introduce a new benchmark dataset, Geoperception, derived from the Geometry-3K corpus (Lu et al., 2021), specifically designed to evaluate MLLMs’ ability to accurately perceive surface-level geometric information without requiring complex inference or reasoning. Our benchmark reveals shortcomings in precise geometric perception across all leading vision-language MLLMs, both closed and open-source.
- • Empirical Study and Synthetic Data Engine: To investigate the root cause of this performance, we conduct a detailed empirical exploration of MLLM architecture and training strategies. To aid in our investigation, we develop a synthetic data engine capable of generating high-fidelity visual descriptions of geometric elements. This study leads to key insights, such as the importance of certain architectural choices and the use of curriculum-based, multi-stage training with progressively more complex visual descriptions for improving low-level visual perception.
- • Euclid Model Family: Leveraging the insights from our exploration and our synthetic data engine, we train Euclid, a series of MLLMs tailored for high-quality geometric LLVP. Although purely trained on synthetic multimodal data with simple geometry shapes, Euclid generalizes strongly to the real-world geometry images from Geoperception benchmark, for instance, outperforming the best closed-source model, Gemini-1.5-Pro, by up to 58.56% on certain benchmark tasks and 10.65% across the tasks.

#### 2. Background and Related Work

We provide an overview of prior efforts that assess and improve low-level perception and geometric reasoning in MLLMs, and highlight our contributions in data synthesis, evaluation, and training.

Vision-Language MLLMs. While recent iterations of LLMs feature a standardized model architecture and pretraining recipe, MLLMs still often differ in design choices for infusing visual inputs. One popular design

is to align continuous visual features with the embedding space of a backbone LLM (Liu et al., 2024a,b, Dubey et al., 2024, McKinzie et al., 2024, Tong et al., 2024a, Beyer et al., 2024, AI, 2023, Wang et al.,

- 2024a); another approach involves tokenizing visual inputs to be trained jointly with language tokens (Team et al., 2023, Team, 2024a). These modules are often infused with a decoder-only LLM, but others have explored encoder-decoder architectures to integrate a more varied collection of modalities (Alayrac et al.,

- 2022, Mizrahi et al., 2024, Ormazabal et al., 2024, Bachmann et al., 2024). Our study focuses on decoder MLLMs with a continuous visual encoder, and we carry out an empirical study to explore the effect of synthetic dataset mixture, training recipe, and encoder design (Liu et al., 2022, Radford et al., 2021, Zhai et al., 2023, Oquab et al., 2023).

Geometry-Oriented MLLMs. At the core of these choices is the hardness in designing a module adept in general visual reasoning (McKinzie et al., 2024, Tong et al., 2024a). In this work, we explore the optimal design of MLLMs specialized in low-level visual perception, a crucial aspect for (among other applications) multimodal mathematical understanding (Lu et al., 2023, Zhang et al., 2024a). This paper supplements prior efforts in improving mathematical reasoning (Gao et al., 2023, Zhang et al., 2024b, Zhuang et al., 2024, Li et al., 2024, Peng et al., 2024, Shi et al., 2024b) with a detailed study on the effect of dataset mixture, curriculum, and visual encoder, to reach a recipe that elicits strong performance on geometric tasks (Kazemi et al., 2023) that require low-level perception.

Evaluating LLVP. Many benchmarks (Rahmanzadehgervi et al., 2024) have reported that frontier-class MLLMs struggle with visual perception tasks, which are prerequisites for applications that emphasize low-level geometric perception (Chen et al., 2024, Fu et al., 2024c), including mathematical (Yue et al., 2024, Lu et al.,

- 2023, Zhang et al., 2024a, Jiang et al., 2024) and spatial reasoning (Chen et al., 2024, Fu et al., 2024b). These findings collectively identify that MLLMs exhibit a language prior (Lin et al., 2023)—a preference of textual inputs over visual inputs—leading to a performance gap between modalities (Wang et al., 2024b, Zhang et al., 2024a, Fu et al., 2024a). Meanwhile, there lacks a high-quality benchmark that evaluates low-level geometric perception in MLLMs, and the Geoperception benchmark represents a first effort to narrow this gap. This type of efforts have led to significant improvements in certain capabilities of MLLMs, such as compositionality of objects (Yuksekgonul et al., 2022, Kong et al., 2023).

Improving LLVP. Many prior works study data-driven approaches to improve low-level perception skills. For example, Gao et al. (2023), Li et al. (2024), Zhuang et al. (2024) employ a standardized supervised finetuning recipe, and optionally adjust the training data mixture. This type of training data is often synthesized from text-only math problems (Lu et al., 2021, Trinh et al., 2024) or via rule-based systems (Kazemi et al., 2023). In parallel, Vishniakov et al. (2023), Shi et al. (2024a), Tong et al. (2024b) have explored the design space of visual encoders for general-purpose vision-language reasoning. We identify best practices over the union of these design spaces, and then train small MLLMs with strong performance in low-level perception tasks.

Lastly, several works (Schick et al., 2024, Surís et al., 2023, Hu et al., 2024) have opted to augment an MLLM with external APIs that process low-level features with specialized vision modules, such as object detection (Redmon et al., 2016), segmentation (Kirillov et al., 2023), and depth estimation (Yang et al., 2024). While these agentic frameworks (Wu et al., 2023) present a promising alternative that directly addresses the shortcomings of visual encoders, they are limited by their scalability to novel use cases, and may be insufficient for precise tool routing that requires low-level perception as a primer (Picard et al., 2023, Wu et al., 2024, Buehler, 2024).

###### PointLiesOnLine

###### PointLiesOnCircle

###### AngleClassification

LineComparison

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Q: What is the point lying o-

Q: What is the point lying on circle with center G? A: L, H, F

Q: Is angle BAC acute or obtuse? A: acute

Q: Which line is longer, QS -

n line YB? A: D, G

or SP? A: QS

Figure 1: Four examples from our Geoperception dataset. The questions are sourced from the Geometry-3K corpus (Lu

- et al., 2021), which compiles problems from two widely-used high school textbooks. We perform filtering, validation, and generate question-and-answer text for each image.

#### 3. Geoperception Benchmark

Recently, there has been a growing number of multimodal benchmarks across diverse domains beyond natural image understanding, including mathematical reasoning (Zhang et al., 2024a, Lu et al., 2023) and abstract visual reasoning (Jiang et al., 2024, Chia et al., 2024). Many of these prior works have realized the importance of accurate low-level visual perception. Specifically, Marvel (Jiang et al., 2024) introduces perception questions for various abstract reasoning patterns, and finds that the main bottleneck of MLLMs’ performance on abstract visual reasoning is that they fail to accurately transcribe visual information into concepts; Mathverse (Zhang et al., 2024a) and IsoBench (Fu et al., 2024a) both test MLLMs on equivalent question represented by language and visual modalities, respectively. Both works find that language-only input always outperforms vision-language input, and that the vision component of MLLMs always fails to utilize low-level visual features. VDLM (Wang et al., 2024b) transcribes raster images into vector graphics and uses LLMs to reason over the SVG code. They find that although SVG code is not straightforward to understand, using LLMs to reason over SVG is consistently more effective than directly using MLLMs on original raster images. Blind-test (Rahmanzadehgervi et al., 2024) and BLINK (Fu et al., 2024c) also share similar findings with the works above.

A Benchmark for Geometric LLVP. Although such shortcomings of MLLMs are commonly recognized, there is a lack of comprehensive benchmark that purely focuses on these abilities of MLLMs. Our goal is to construct a benchmark focusing solely on the perception ability of MLLMs, which is also representative enough of real-world applications. When humans perceive and memorize visual information, it is wellrecognized that this procedure relies crucially on searching for the closest and simplest corresponding geometric shapes (Sablé-Meyer et al., 2022). We posit that geometric perception is a fundamental and broadly representative LLVP ability in many applications. Hence, we select geometry understanding as our domain of dataset construction.

Benchmark Tasks. Over two thousand years ago, Euclid introduced five axioms that underpin all further geometric reasoning. These axioms involve establishing and extending lines using points (Axioms 1 and 2), constructing circles from a point and a radius (Axiom 3), and defining perpendicularity (Axiom 4) and parallelism (Axiom 5). Additionally, Euclid provided common notions regarding the properties of equality. To capture these aspects, we define five tasks in our Geoperception dataset: PointLiesOnLine,

PointLiesOnCircle, Parallel, Perpendicular and Equal, and additionally define AngleClassification and LineComparison tasks to assess the model’s understanding of angle and length measurements, resulting in a total of seven tasks. In geometric diagrams, perpendicularity, parallelism, and equality are often indicated by annotation symbols. Thus, we classify Parallel, Perpendicular, and Equal as annotated geometry understanding. Meanwhile, PointLiesOnLine, PointLiesOnCircle, AngleClassification, and LineComparison fall under primitive geometry shape understanding, which includes both logical (PointLiesOnLine, PointLiesOnCircle) and numerical (AngleClassif-ication, LineComparison) tasks.

Table 1: Statistics of the seven tasks in our Geoperception dataset, including the number of questions and images.

Data Filtering. Geoperception is sourced from the Geometry3K (Lu et al., 2021) corpus, which offers precise logical forms for geometric diagrams, compiled from popular high-school textbooks. However, certain points in these logical forms are absent in the corresponding diagrams. To resolve this, we use GPT-4o-mini MLLM to confirm the presence of all points listed in the logical forms. This process filters the 3,002 diagrams to retain 1,584, where at least one logical form fully represents its points in the diagram. A random inspection of 100 annotations reveals only two errors, indicating high annotation accuracy.

Predicate # Q # I

PointLiesOnLine 1901 924 PointLiesOnCircle 359 322

Parallel 106 101 Perpendicular 1266 456

Equals 4436 1202 AngleClassification 2193 1389

LineComparison 1394 1394

Converting Logical Forms Into Questions. We convert logical forms into question-and-answer pairs for each of the seven tasks in Geoperception. In the Equals task, for example, we directly convert the logical form (e.g., Equals(LengthOf(Line(Q, T)), 86)) into a question-answer pair (e.g., Q: What is the length of line QT as annotated? A: 86). For PointLiesOnLine, two points on the line are chosen to form the question, with the remaining points on the line as the answer. Similarly, for PointLiesOnCircle, we ask which points lie on the circle, using its center as the basis for the question. For Parallel and Perpendicular, we represent each line by two points and query which other lines are parallel or perpendicular to it. In AngleClassification, we ensure the queried angle is in the range of [10,80] ∪ [100,170] degrees to avoid ambiguity. For LineComparison, we ensure that the shorter line is less than 70% of the length of the longer line. Since multiple equivalent questions can be generated for a single logical form (e.g., a line containing five points generates 5P2 equivalent questions), we randomly select one to avoid redundancy. Table 1 summarizes the question statistics for each task, as well as the number of images involved. Four examples from Geoperception are illustrated in Fig. 1.

Evaluation Details. We evaluate seven leading MLLMs, both open source and closed source. The open source models include Molmo-7B-D (Deitke et al., 2024), Cambrian-1-8B (Tong et al., 2024a), Qwen2VL-7B (Wang et al., 2024a), Llama-3.2-11B (Dubey et al., 2024), and Pixtral-12B (AI, 2023). The closedsource models include GPT-4o-mini (Achiam et al., 2023), GPT-4o (Achiam et al., 2023), Claude-3.5-Sonnet (Anthropic, 2024), Gemini-1.5-flash (Team et al., 2023), and Gemini-1.5-pro (Team et al., 2023). Additionally, GPT-4o-mini without image input is used for generating the random baseline, employing the same textual instructions. To prevent stretching, all images are padded to square dimensions before being fed into the models. During evaluation of a given question by an MLLM, let G denote the ground truth set of answers,

and let P denote the predicted set of answers; then the evaluation score is defined as

⎧ ⎨ ⎪⎩

∣P∣ ∣G∣

if P ⊆ G, 0 otherwise.

Evaluation score =

(1)

Current MLLMs struggle to perceive low-level geometry annotations and relationships. We show a comparison of all models on Geoperception in Table 2. Despite the simplicity of Geoperception for humans, it remains a considerable challenge for even the most advanced commercial MLLMs. Notably, all models fall short of achieving 30% accuracy on the PointLiesOnLine task and do not outperform the text-only GPT-4o mini model in AngleClassification task. Closed source models generally outperform open source ones, with Gemini-1.5-pro attaining the highest average score of 56.98%, followed by gemini-1.5-flash at 54.76%. Among open source models, Pixtral-12B achieves the best performance with an overall score of 41.95%. It is worth noting that Cambrian-1 (Tong et al., 2024a), which is reported to be trained on Geo-170K (Gao et al.,

- 2023), a geometry multimodal instruction tuning dataset built on the logical annotation of Geometry-3K, the same source with our Geoperception, still faces challenges in our Geoperception task, despite being trained on the dataset having the same images and augmented text annotations.

Table 2: Performance (average evaluation score) of different models on Geoperception benchmark tasks. POL: PointLiesOnLine, POC: PointLiesOnCircle, ALC: AngleClassification, LHC: LineComparison, PEP: Perpendicular, PRA: Parallel, EQL: Equals. As the Random Baseline method, we use GPT-4o-mini, given the same textual instruction but without an image. The best model for each task is bolded.

Logical Numerical Annotations Model POL POC ALC LHC PEP PRA EQL Overall Random Baseline 1.35 2.63 59.92 51.36 0.23 0.00 0.02 16.50 Open Source Molmo-7B-D (Deitke et al., 2024) 11.96 35.73 56.77 16.79 1.06 0.00 0.81 17.59 Llama-3.2-11B (Dubey et al., 2024) 16.22 37.12 59.46 52.08 8.38 22.41 49.86 35.08 Qwen2-VL-7B (Wang et al., 2024a) 21.89 41.60 46.60 63.27 26.41 30.19 54.37 40.62 Cambrian-1-8B (Tong et al., 2024a) 15.14 28.68 58.05 61.48 22.96 30.74 31.04 35.44 Pixtral-12B (AI, 2023) 24.63 53.21 47.33 51.43 21.96 36.64 58.41 41.95 Closed Source GPT-4o-mini (Achiam et al., 2023) 9.80 61.19 48.84 69.51 9.80 4.25 44.74 35.45 GPT-4o (Achiam et al., 2023) 16.43 71.49 55.63 74.39 24.80 60.30 44.69 49.68 Claude 3.5 Sonnet (Anthropic, 2024) 25.44 68.34 42.95 70.73 21.41 63.92 66.34 51.30 Gemini-1.5-Flash (Team et al., 2023) 29.30 67.75 49.89 76.69 29.98 63.44 66.28 54.76 Gemini-1.5-Pro (Team et al., 2023) 24.42 69.80 57.96 79.05 38.81 76.65 52.15 56.98

#### 4. Empirical Study on MLLM Design Space

Although large-scale web-crawled image-text pairs cover a variety of domains, including geometry, the textual descriptions often lack the necessary specificity and depth. To address this issue, current studies in this domain (Gao et al., 2023, Shi et al., 2024b, Zhang et al., 2024b) typically construct a geometry or

mathematical domain dataset and apply the same training strategy used for general-purpose MLLMs. For example, Math-LLaVA (Shi et al., 2024b) and multi-math (Peng et al., 2024) rely on GPT-4V or GPT-4o’s vision ability to generate most of the question and answer pairs and image captions, which is essentially model distillation. However, as evidenced by Table 2, GPT-4o and Gemini-1.5-Pro often struggle to answer even basic perception questions, limiting the performance potential of resulting models. Furthermore, while works such as G-LLaVA (Gao et al., 2023), MAVIS (Zhang et al., 2024b), and Math-PUMA (Zhuang et al.,

- 2024) utilize human crafted logical forms or synthetic multimodal data to ensure the reliability of textual annotations, they often conflate low-level perception with problem-solving, and train models to directly solve multimodal geometry problems, without verifying if the model’s low-level perception abilities are sufficient. As an evidence, the best models in MAVIS (Zhang et al., 2024b) and Math-PUMA (Zhuang et al.,

2024) evaluation results on Mathverse (Zhang et al., 2024a) still have a substantial gap of 26.8% and 28.7% between text-dominant versions and vision-only versions of problems1, respectively. Furthermore, attempts to train MLLMs on low-level visual perception tasks (Wang et al., 2024b, Rahmanzadehgervi et al., 2024) have also struggled to achieve satisfactory in-domain performance or generalize effectively. In this section, we aim to address these challenges.

We hypothesize the inability of today’s MLLMs to effectively perceive basic geometric annotations and relationships stems from two factors: 1. The lack of high-fidelity geometric visual perception training data. 2. The problem of their model architectures and training strategy. Next, we will introduce our geometry dataset generation engine to overcome the lack of data, and then use generated dataset to study the optimal training strategy.

A B C = triangle A B C;

A B C = triangle A B C; D = midpoint B C

A B C = triangle A B C;

- D = midpoint A B;
- E = midpoint A C

- D = midpoint B C;
- E = midpoint A C;
- F = intersection_ll A D B E

Geometry Dataset Generation Engine

[Figure 5]

[Figure 6]

[Figure 7]

##### … … …

Geometry Image Generation Engine. To provide sufficient high-fidelity training datasets, we develop a synthetic dataset generation engine to programmatically produce geometry shapes. Our geometry shape generation engine is built on AlphaGeometry (Trinh et al.,

Figure 2: Three geometry logical shapes, of increasing complexity, used in our empirical study. Our geometry image generation engine is able to produce infinite visual instances for each of these logical shapes. All letters are randomly sampled from the alphabet and reassigned to each of the points before drawing.

2024). Given an input formal language describing a geometry shape, the geometry engine will first check the validity of the geometry shape. Then it will create numerical positions for all points following the restrictions given by the input. After the creation of all points, it will connect the line as specified in the input. To avoid inductive bias during training (e.g., point A is always on top of a triangle), letters are first picked from a letter pool (e.g., all 26 capital letters) and then randomly assigned to each point. In addition to the original image generation engine, we introduce three visualization enhancements: (1) additional inputs to control the connections between points, number of letters in the letter pool, presence of each points, and annotation about length and angles; (2) increased randomness in creating numerical instances from conceptual shapes;

1In Mathverse, text-dominant is the version where the problem is mainly represented by text, while in the vision-only version an equivalent problem is represented purely by image.

###### LineComparison PointLiesOnLine

Testing Accuracy

Training Loss

Testing Accuracy

| |Training Loss| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

1.0

100

100

0.8

0.8

10 1

10 1

0.6

0.6

10 2

10 2

0.4

0.4

10 3

10 3

0.2

0.2

10 4

10 4

0.0

0.0

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct

- Figure 3: LLM size experiments. Training loss and testing accuracy curve comparing three choices of LLM size with a fixed visual encoder and multimodal connector. Training losses are window-smoothed using a window size of 10 for better visibility.

and (3) adjustments to the canvas range to ensure visibility of all geometry components. Examples of our geometry generation engine, showing three geometries of increasing complexity, are shown in Fig. 2

Exploration of MLLM design space. With sufficient training dataset, we now revisit the MLLMs architectural and training design space. We choose 2 primitive geometry tasks from Geoperception as the test bed for the exploration: logical task, PointLiesOnLine and numerical task, LineComparison. For each task, we carefully create three tasks with incremental difficulty levels. We name them as difficulty level easy, medium and hard. Based on the insight from our preliminary experiments, to increase the difficulty levels, for PointLiesOnLine, we increase the complexity of geometry shapes as is shown in Fig. 18, for LineComparison, we increase the total number of letters in letter pool while mixing geometry shapes. During our preliminary experiments, we find that sometimes the model fails to converge due to instability. To this end, for all experiment moving forward, we run the training for three times and report the best run among them (i.e., having the lowest overall training loss or testing accuracy).

We start with a typical setting of MLLMs: CLIP-ViT-L/14 (Radford et al., 2021) as the visual encoder and a two layer MLP as multimodal connector and the latest Qwen-2.5 series (Team, 2024b) as LLM. During training, we actively tune the MLP and LLM, while keeping visual encoder frozen. We use the mixture of three difficulty levels as the training set.

- Lesson 1: Under the same training dataset, scaling LLM sizes does not lead to better performance. It is commonly acknowledged that under the same training dataset, scaling up LLM can lead to better MLLM performance (Liu et al., 2024a). To this end, we first vary the sizes of LLMs, Qwen-2.5 (Team,

- 2024b) in a range of 0.5B, 1.5B, and 3B while keep other components consistent. The result is shown in Fig. 3. For LineComparison, Qwen-2.5-1.5B performs the best while Qwen-2.5-3B learns most slowly. For PointLiesOnLine, Qwen-2.5-1.5B and Qwen-2.5-3B performs almost the same. Qwen-2.5-0.5B learns relatively slower, but still reach almost the same final performance with two other models. In conclusion, we do not observe an obvious trend that larger LLMs can learn such low-level visual perception task faster or

###### LineComparison PointLiesOnLine

Training Loss

Test Accuracy

Training Loss

Test Accuracy

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

1.0

100

100

0.8

0.8

10 1

10 1

0.6

0.6

10 2

10 2

0.4

0.4

10 3

10 3

0.2

0.2

10 4

10 4

0.0

0.0

0 25 50 75 100 # Training Instances (k)

0 25 50 75 100 # Training Instances (k)

0 25 50 75 100 # Training Instances (k)

0 25 50 75 100 # Training Instances (k)

ConvNeXT-L@512

ViT-L@224

ViT-g@224 ViT-H@224

DINO-G@224

ConvNeXT-XXL@512

SigLIP@224

DINO-L@224

- Figure 4: Vision encoder experiments. Training loss and testing accuracy (on a 1500 instances holdout test set) curve comparing eight visual encoders, with a fixed multimodal encoder and LLM. For a fair comparison, all visual encoder transcribe an image into 256 visual tokens. Training losses are window-smoothed using a window size of 10 for better visibility.

better. Moving forward, we will use Qwen-2.5-1.5B to continue our exploration.

- Lesson 2: CNN architecture performs better than ViT. We then study the choice of visual encoder architectures, including two families of architectures: Vision Transformer (ViT) (Dosovitskiy, 2020) and ConvNeXT (Liu

et al., 2022); as well as two visual representation learning objectives: language-supervised learning (Radford et al., 2021) and self-supervised learning (Oquab et al., 2023). We control the number of visual tokens to 256 for all of our vision encoders. The result is shown in Fig. 4. We find that ConvNeXt-XXLarge and ConvNeXt-Large consistently learns the geometric concept the fastest among all of the visual encoders. Notably, ConvNeXT-Large shows superior learning performance with the vision transformers which are 3-5 times larger. We hypothesize that CNN architecture extract visual features globally, effectively preserving low-level visual features. In contrast, ViT architectures split images into discrete patches, making it more challenging to retain the original low-level visual information. Self-supervised learning (SSL) visual encoders, DINO-v2, struggles to learn the geometry concept; we hypothesis this is due to the weak vision-language representation in these models. Surprisingly, although the SigLIP-family is widely-recognized as a better visual encoder (Tong et al., 2024a), we find that their performance in learning basic visual geometry attributes is limited.

- Lesson 3: Tuning vision encoder does not provide significant help. We next study the effect of tuning versus freezing the visual encoder. In Fig. 5, we show the testing accuracy curves of tuning and freezing visual encoders. We find that compared with using a frozen encoder, tuning the visual encoder does not help

Table 3: Summary of Visual Encoders

Model Params Objective

ConvNeXt Large@512 200M CLIP ConvNeXt XXLarge@512 847M CLIP ViT-g/14@224 1.01B CLIP ViT-H/14@224 632M CLIP ViT-L/14@224 303M CLIP SigLIP@224 (ViT) 428M CLIP-like DINOv2 Giant@224 (ViT) 1.14B Self-Sup DINOv2 Large@224 (ViT) 304M Self-Sup

###### LineComparison

###### PointLiesOnLine

ConvNeXT-L@512

ConvNeXT-XXL@512

ConvNeXT-L@512

ConvNeXT-XXL@512

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

freeze Vi-Enc

freeze Vi-Enc

freeze Vi-Enc

freeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

0.0

0.0

0.0

0.0

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

- Figure 5: Tuning/freezing vision encoder experiments. Testing accuracy (on a 1500 instances holdout test set) curve comparing freezing versus tuning the visual encoder during training.

the model learn low-level geometry relationships faster or better. In what follows, we will freeze the encoder for simplicity.

- Lesson 4: Curriculum learning unleashes full potential. Finally, we study training data composition. In our preliminary experiment Fig. 19, we observe that the model fails to converge on difficulty level 3 of PointLiesOnLine and difficulty level 2 and 3 of LineComparison. However, when using mixed training set of all three difficulty levels, the model achieves convergence, despite using the same amount of data for each difficulty levels. We hypothesize that including easier levels aids the model in learning more complex levels. To test this hypothesis, we report the test accuracy for three difficulty levels separately during the mixed training of ConvNeXtXXLarge, in Fig. 6, on both tasks. We notice that the testing accuracy for easier tasks increase earlier and more quickly than difficulty tasks. In PointLiesOnLine tasks, we notice an apparent plateau for hard level tasks until the model has trained on approximately 20K samples. During this period, the testing accuracy for easy and medium continue to increase. This suggests that learning easier shapes can significantly help the model tackle more challenging shapes, comparing with directly learning the challenging ones, this finding align with the principles of curriculum learning.

###### LineComparison

###### PointLiesOnLine

| | | | | | | |
|---|---|---|---|---|---|---|
| | || |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
| | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
| | | |
| | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
| | | |
| | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| | | | |
| | || |
|---|
| | | | |
| | | | | | | |

1.0

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

| |
|---|

| |
|---|

| |
|---|
| |
| |

| |
|---|
| |

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

0.8

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

| |
|---|

| |
|---|

| |
|---|

TestAccuracy

0.6

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|
| |

0.4

0.2

0.0

0 25 50 75 100 # Training Instances (k)

0 25 50 75 100 # Training Instances (k)

Difficulty: Easy Difficulty: Medium Difficulty: Hard

Figure 6: Separate testing accuracy curves on difficulty levels easy, medium, and hard, shown over the course of training on a mixture of all difficulty levels.

While mixed training enables effective spontaneous curriculum learning, we investigate whether a structured curriculum can further enhance model efficiency on challenging shapes. To this end, we train the model sequentially from simple to more complex shapes and compare testing accuracy just on hard level tasks. During training, we monitor the model’s performance and dynamically adjust the distribution of

LineComparison

ConvNeXT-L@512

ConvNeXT-XXL@512

ViT-L@224

ViT-H@224

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

PointLiesOnLine

ConvNeXT-L@512

ConvNeXT-XXL@512

ViT-L@224

ViT-H@224

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

0 25 50 75 100 # Training Samples (k)

Purely Training on Hard Level Mixed Training Curriculum Training

Figure 7: Curriculum learning experiments. Test accuracy on difficulty level hard of three training strategies: purely training on difficulty level hard, mixed training of difficulty levels easy/medium/hard, and curriculum training.

training data (i.e., the curriculum stage) based on this performance. Specifically, the model starts by training on the easy level data. and is evaluated when it finishes a training round, using testing accuracy from the current level of data. Upon evaluation, if the model achieves an accuracy exceeding a predefined threshold θ, the framework advances the level to the next. Formally, the update rule for advancing stages is given by:

if accuracys > θ ⇒ c ← c + 1. (2)

The model is trained on a total of M rounds and K steps within each round. To avoid forgetting, we apply data smoothing at each stage. Specifically, we smooth our dataset distribution over all stages using an exponential attenuation function:

ratios = exp(−α ⋅ ∣stages − c∣), (3)

where α denotes the attenuation rate. Eq. (3) ensures that stages proximal to the current stage receive higher sampling probabilities.

We refer to this as our curriculum training strategy. Specifically, the accuracy threshold for advancing training stage θ is set to 0.99. We train all the models for M = 30 rounds, each round with K = 50 steps. The results are shown in Fig. 7. Firstly, we find that all of the models fail to converge when trained purely on hard level for PointLiesOnLine task. In contrast, the mixed training strategy shown by the red curve, consistently reaches faster convergence on hard level. Curriculum training strategy, shown by the purple curve, proves more efficient than mixed training.

#### 5. Euclid: a Family of MLLMs for Geometric Visual Perception

In this section, we take all of the lessons we learned in the previous sections and train Euclid, a family of MLLMs specifically designed for strong geometric LLVP.

On-the-fly progressive training. We use the same strategy as the curriculum training in Section 4, but scale our training to all tasks in Geoperception. For each task, we create N stages of training dataset shapes with progressively increasing geometric complexity.

Specifications. For models, we select the best visual encoder architecture we found in our investigation, ConvNeXt, including ConvNeXt-Large@512 and ConvNeXt-XXLarge@512, and keep the same multimodal connector (2 layers MLP) and LLM (Qwen2.5-1.5B-instruct). The accuracy threshold for advancing training stage θ is set to 0.99. All models are trained on N = 3 stages with manually curated geometry shapes and M = 50 rounds with K = 500 steps in each round, and the batch size is 64 for each training step. The total training dataset volume for both of the models is 1.6M.

- Table 4: Performance comparison between Euclid and the best leading open source and closed source MLLMs on the seven tasks. Note that Euclid is not trained on any of the in-distribution data from the benchmark tasks below. The best model for each task is bolded.

Logical Numerical Annotations

Model POL POC ALC LHC PEP PRA EQL Average

Random Baseline 0.43 2.63 59.92 51.36 0.25 0.00 0.02 16.37 Pixtral-12B (AI, 2023) 24.63 53.21 47.33 51.43 21.96 36.64 58.41 41.95 Gemini-1.5-Pro (Team et al., 2023) 24.42 69.80 57.96 79.05 38.81 76.65 52.15 56.98

Euclid-ConvNeXt-Large 80.54 57.76 86.37 88.24 42.23 64.94 34.45 64.93 Euclid-ConvNeXt-XXLarge 82.98 61.45 90.56 90.82 46.96 70.52 31.94 67.89

Evaluation results. The results are shown in Table 4. Overall, although only trained on very simple synthetic geometry shapes, and using only a 1.5B language model, Euclid significantly outperforms current leading MLLMs in most of the tasks, showing strong generalization abilities on real-world geometry LLVP. Notably, in the PointLiesOnLine task, which is particularly challenging for existing MLLMs, Euclid achieves up to 82.98% accuracy, more than three times the performance of Gemini-1.5-Pro. On all both numerical tasks, LineComparison and AngleClassification, Euclid keeps superior performance. However, on three annotation tasks, Euclid’s performance is limited. We hypothesis this is due to the limited setting of our annotation types and styles, making the model hard to generalize to diverse human geometry annotations.

Error analysis. We take a deep look into Euclid’s prediction on Geoperception, we find that our model’s performance is hindered when diagrams are heavily annotated. An example is shown in Fig. 8, where a line is annotated by “x”, confusing the model from choosing the correct point. Incorporating training data with more diverse annotation types, geometry shape and can better distinguish different diagram annotation types could potentially help the model with such scenarios.

[Figure 8]

Question: What is the point lying on line TY?

#### 6. Conclusion and Future Work

Ground truth: W Prediction: X

In this work, we highlight the importance of accurate low-level visual perception(LLVP) in MLLMs. To this end, we first introduce Geoperception, a large-scale multimodal benchmark focused exclusively on geometry-domain visual perception. We evaluate leading MLLMs on Geoperception, find that even top models such as Gemini-1.5-Pro struggle significantly it, although it is straightforward for humans. We then conduct an empirical study to explore the design space of MLLM training and architectures using the dataset generated by a geometric high-fidelity synthetic-data engine that we develop. Our study indicate that convolutional neural network visual encoders outperform vision transformers in our tasks; tuning the visual encoder generally enhances performance; and employing a curriculum-based training approach yields much more model potential than direct task training. Based on insights from this study, we develop Euclid, a model trained purely on highfidelity synthetic generated data, which generalizes effectively to real-world geometric shape understanding tasks, surpassing the leading MLLMs by a substantial margin.

Figure 8: An error case where Euclid fails to predict the correct point on a line, potentially distracted by the annotation “x”.

Future work. Our work examines the potential of using synthetic multimodal data to improve MLLM performance in low-level geometric perception tasks. However, there are still directions that remain underexplored: (1) Automatic curriculum learning. Incorporating a more diverse dataset, including varied geometric shapes and different domain dataset, introduces challenges in defining the learning order. Rule based definition and manual curation may become impractical, necessitating automated strategies like hard negative sampling to organize the curriculum based on training loss or testing accuracy. This approach could streamline the process, reduce human effort, provide more suitable and efficient curriculum learning orders. (2) Using a more-diverse training dataset. Currently, the text portion of our synthetic multimodal training data uses a restricted set of templates, and the model trained on such templates could fail to generalize to other question types; it could therefore be beneficial to increase the diversity of our training images as well as the instruction-following formats. (3) Generalizing to other task domains. In this work, our study is focused on data from 2D geometry, as it provides a focused test bed of fundamental tasks. We believe the lessons we learn from this domain can be effectively generalized to a broader set of downstream domains that benefit from high-quality LLVP.

#### Reproducibility Statement

In Section 3, we provide a comprehensive description of the procedure for generating the Geoperception benchmark. This includes employing GPT-4o-mini for dataset filtering and detailing the conversion of logical forms into questions and answers. Evaluation prompts for MLLMs on different types of Geoperception questions are presented in Appendix B. For model architecture exploration, we specify the visual encoders and provide corresponding Hugging Face links in Table 3. Additionally, we outline the LLMs and multimodal connector architectures used. For our Euclid model, we include all geometry shape code used for training, along with demonstration diagrams and pseudo-code for generating training questions and answers.

#### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Kian Ahrabian, Zhivar Sourati, Kexuan Sun, Jiarui Zhang, Yifan Jiang, Fred Morstatter, and Jay Pujara. The curious case of nonverbal abstract reasoning with multi-modal large language models. arXiv preprint arXiv:2401.12117, 2024.

Mistral AI. Pixtral 12b. https://mistral.ai/news/pixtral-12b/, 2023. Accessed: 2024-09-27. Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur

Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

Anthropic. The claude 3 model family: Opus, Sonnet, Haiku, March 2024. URL https: //www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_ Claude_3.pdf.

Roman Bachmann, Oğuzhan Fatih Kar, David Mizrahi, Ali Garjani, Mingfei Gao, David Griffiths, Jiaming Hu, Afshin Dehghan, and Amir Zamir. 4m-21: An any-to-any vision model for tens of tasks and modalities. arXiv preprint arXiv:2406.09406, 2024.

Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Markus J Buehler. Cephalo: Multi-modal vision-language models for bio-inspired materials analysis and design. Advanced Functional Materials, page 2409531, 2024.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.

Yew Ken Chia, Vernon Toh Yan Han, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. arXiv preprint arXiv:2403.13315, 2024.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Jen Dumas, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models, 2024. URL https://arxiv.org/abs/2409.17146.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Deqing Fu, Ruohao Guo, Ghazal Khalighinejad, Ollie Liu, Bhuwan Dhingra, Dani Yogatama, Robin Jia, and Willie Neiswanger. Isobench: Benchmarking multimodal foundation models on isomorphic representations. In First Conference on Language Modeling, 2024a. URL https://openreview.net/forum?id= KZd1EErRJ1.

Deqing Fu, Tong Xiao, Rui Wang, Wang Zhu, Pengchuan Zhang, Guan Pang, Robin Jia, and Lawrence Chen. Tldr: Token-level detective reward model for large vision language models. arXiv preprint arXiv:2410.04734, 2024b.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024c.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.

Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, et al. Large multilingual models pivot zero-shot multimodal learning across languages. arXiv preprint arXiv:2308.12038, 2023.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. arXiv preprint arXiv:2406.09403, 2024.

Yifan Jiang, Jiarui Zhang, Kexuan Sun, Zhivar Sourati, Kian Ahrabian, Kaixin Ma, Filip Ilievski, and Jay Pujara. Marvel: Multidimensional abstraction and reasoning through visual evaluation and learning. arXiv preprint arXiv:2404.13591, 2024.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

Xianghao Kong, Ollie Liu, Han Li, Dani Yogatama, and Greg Ver Steeg. Interpretable diffusion via information decomposition. arXiv preprint arXiv:2310.07972, 2023.

Zhihao Li, Yao Du, Yang Liu, Yan Zhang, Yufang Liu, Mengdi Zhang, and Xunliang Cai. Eagle: Elevating geometric reasoning through llm-empowered visual instruction tuning. arXiv preprint arXiv:2408.11397, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

Zhiqiu Lin, Xinyue Chen, Deepak Pathak, Pengchuan Zhang, and Deva Ramanan. Revisiting the role of language priors in vision-language models. arXiv preprint arXiv:2306.01879, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b. URL https://llava-vl.github.io/ blog/2024-01-30-llava-next/.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024c.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.

David Mizrahi, Roman Bachmann, Oguzhan Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, and Amir Zamir. 4m: Massively multimodal masked modeling. Advances in Neural Information Processing Systems, 36, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Aitor Ormazabal, Che Zheng, Cyprien de Masson d’Autume, Dani Yogatama, Deyu Fu, Donovan Ong, Eric Chen, Eugenie Lamprecht, Hai Pham, Isaac Ong, et al. Reka core, flash, and edge: A series of powerful multimodal language models. arXiv preprint arXiv:2404.12387, 2024.

Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024.

Cyril Picard, Kristen M Edwards, Anna C Doris, Brandon Man, Giorgio Giannone, Md Ferdous Alam, and Faez Ahmed. From concept to manufacturing: Evaluating vision-language models for engineering design. arXiv preprint arXiv:2311.12668, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. arXiv preprint arXiv:2407.06581, 2024.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016.

Mathias Sablé-Meyer, Kevin Ellis, Josh Tenenbaum, and Stanislas Dehaene. A language of thought for the mental representation of geometric shapes. Cognitive Psychology, 139:101527, 2022.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.

Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024a.

Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024b.

Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11888–11898, 2023.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024a.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Qwen Team. Qwen2.5: A party of foundation models, September 2024b. URL https://qwenlm.github. io/blog/qwen2.5/.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024a.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024b.

Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024.

Kirill Vishniakov, Zhiqiang Shen, and Zhuang Liu. Convnet vs transformer, supervised vs clip: Beyond imagenet accuracy. arXiv preprint arXiv:2311.09215, 2023.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Zhenhailong Wang, Joy Hsu, Xingyao Wang, Kuan-Hao Huang, Manling Li, Jiajun Wu, and Heng Ji. Textbased reasoning about vector graphics. arXiv preprint arXiv:2404.06479, 2024b.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023.

Sifan Wu, Amir Khasahmadi, Mor Katz, Pradeep Kumar Jayaraman, Yewen Pu, Karl Willis, and Bang Liu. Cadvlm: Bridging language and vision in the generation of parametric cad sketches. arXiv preprint arXiv:2409.17457, 2024.

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why visionlanguage models behave like bags-of-words, and what to do about it? arXiv preprint arXiv:2210.01936,

- 2022.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986,

- 2023.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024a.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739, 2024b.

Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. arXiv preprint arXiv:2408.08640, 2024.

### Appendix

#### A. Geoperception Benchmark Details

In Table 5, we provide more details on the Geoperception benchmark, such as the number of logic forms present before and after filtering, the number of questions, and the number of images. AngleClassification and LineComparison are directly derived from points coordinates without filtering.

Predicate # LF Before Filter # LF After Filter # Q # I

PointLiesOnLine 6988 2567 1901 924 PointLiesOnCircle 1966 1240 359 322

Parallel 222 123 106 101 Perpendicular 1111 680 1266 456

Equals 6434 4123 4436 1202

- Table 5: Statistics of the five predicates in our Geoperception dataset. Including number of logic forms before filter, after filter and the number of questions and images.

###### PointLiesOnLine

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Q: What is the point lying o-

Q: What is the point lying o-

Q: What is the point lying o-

Q: What is the point lying o-

n line JL? A: R

n line ZX? A: N

n line AB? A: E

n line RN? A: Q

###### PointLiesOnCircle

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Q: What is the point lying on circle with center P? A: T, S, R, Q

Q: What is the point lying on circle with center K? A: L, J

Q: What is the point lying on circle with center Z? A: X, C

Q: What is the point lying on circle with center F? A: A, C, B, D, E

###### Parallel

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Q: What is the line parallel-

Q: What is the line parallel-

Q: What is the line parallel-

Q: What is the line parallel-

to line BE? A: CD

to line NQ? A: OP

to line EB? A: CD

to line CD? A: BE, AB, AE

###### Perpendicular

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Q: What is the line perpendi-

Q: What is the line perpendi-

Q: What is the line perpendicular to line LF? A: LM, KM, GH, HJ, KL, GJ

Q: What is the line perpendi-

cular to line ZW? A: YZ

cular to line CB? A: AC

cular to line VS? A: RT, TV, RV

###### Equals

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Q: What is the length of lin-

Q: What is the measure of an-

Q: What is the measure of an-

Q: What is the line in the diagram that is equal to line VU?

e NM as annotated? A: 39

gle ABC as annotated? A: 2x

gle JKL as annotated? A: 70

A: ZV, VZ

###### AngleClassification

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Q: Is angle SUV acute or obtuse? A: obtuse

Q: Is angle JKL acute or obtuse? A: obtuse

Q: Is angle CBD acute or obtuse? A: acute

Q: Is angle WVX acute or obtuse? A: acute

###### LineComparison

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Q: Which line is longer, AB -

Q: Which line is longer, AE -

Q: Which line is longer, JM -

Q: Which line is longer, RQ -

or AC? A: AC

or ED? A: AE

or JL? A: JL

or QT? A: RQ

Figure 9: Examples of our Geoperception dataset.

#### B. Prompts for the Geoperception Dataset Evaluation

Prompt Template for the PointLiesOnLine Task

Answer me directly just with the all points lie on the line mentioned in the question (do not include the point mentioned in the question). Answer template:

(If only one point) The other point is: "your point". Or

(if multiple points) The other points are: "your points". For example:

The other point is: A Or

The other points are: A, B, C

- Figure 10: Template for the PointLiesOnLine tasks

Prompt Template for the PointLiesOnCircle Task

Answer me directly just with the all points lie on the circle mentioned in the question. Answer template:

(If only one point) The point is: "your point". Or

(If multiple points) The points are: "your points". For example:

The point is: A Or:

The points are: A, B, C

- Figure 11: Template for the PointLiesOnCircle tasks

Prompt Template for the Parallel Task

Answer me directly just with the all lines which are parallel to the line mentioned in the question (do not include the line mentioned in the question). Answer template:

(If only one line) The line is: "your line". Or

(If multiple lines) The lines are: "your lines". For example:

The line is: BC Or:

The lines are: BC, DE

- Figure 12: Template for the Parallel tasks

Prompt Template for the Perpendicular Task

Answer me directly just with the all lines which are perpendicular to the line mentioned in the question (do not include the line mentioned in the question). Answer template:

(If only one line) The line is: "your line". Or

(If multiple lines) The lines are: "your lines". For example:

The line is: BC Or:

The lines are: BC, DE

- Figure 13: Template for the Perpendicular tasks

Prompt Template for the Equals Task

Answer me directly just with the annotations presented on the image. Answer template:

The annotation is: "your annotation". For example:

The annotation is: 2x+4 Or:

The annotations is: 90

- Figure 14: Template for the Equals tasks

Prompt Template for the Angle Classification Task

Answer me directly just with the classification of the angle mentioned in the question. Answer template:

The angle is: "your angle". For example:

The angle is: acute Or:

The angle is: obtuse

- Figure 15: Template for the Angle Classification tasks

Prompt Template for the LineComparison Task

Answer me directly just with the longer line mentioned in the question. Answer template:

The longer line is: "your line". For example:

The longer line is: BC Or:

The longer line is: DE

- Figure 16: Template for the LineComparison tasks

#### C. Details for Training Data Engine

In this section, we provide all geometry shapes we use for Euclid training, including the pseudocode for generating text describing the geometry shapes and diagram examples.

###### C.1. Pseudocode for Training Textual Dataset Synthesis

- Algorithm 1 Data Synthesis for the PointLiesOnLine Task

- 1: Input: data_info, points_set
- 2: Output: data
- 3: for points_set ∈ data_info do
- 4: for (A, B) ∈ permutations(points_set, 2) do
- 5: all_rest_points ← [p for p in points_set if p not in [A, B]]
- 6: for rest_points ∈ permutations(all_rest_points) do
- 7: verb_agreement ← ’is’ if len(rest_points) == 1 else ’are’
- 8: rest_points ← [f"{p}" for p in rest_points]
- 9: rest_points ← sorted(rest_points)
- 10: question ← ’What is the point lying on line ’ + A + B + ’?’
- 11: answer ← ’The point lying on line ’ + A + B + ’ ’ + verb_agreement + ’ ’ + ’, ’.join(rest_points)
- 12: gt ← ”.join(rest_points)
- 13: data ← {’question’: question, ’answer’: answer, ’gt’: gt}
- 14: end for
- 15: end for
- 16: end for

- Algorithm 2 Data Synthesis for the PointLiesOnCircle Task

- 1: Input: data_info
- 2: Output: data
- 3: point_set ← random.choice(list(data_info.items()))
- 4: center_point ← point_set[0]
- 5: target_points ← point_set[1]
- 6: target_points ← sorted(target_points)
- 7: question ← ’What are the point lying on circle ’ + center_point + ’?’
- 8: answer ← ’The point lying on circle ’ + center_point + ’ are ’ + ’, ’.join(target_points)
- 9: gt ← ”.join(target_points)
- 10: data ← {’question’: question, ’answer’: answer, ’gt’: gt}

- Algorithm 3 Data Synthesis for the AngleClassification Task

- 1: Input: data_info
- 2: Output: data
- 3: angle ← data_info
- 4: angle_options ← [f’{angle[1][0]}{angle[1][1]}{angle[1][2]}’, f’{angle[1][2]}{angle[1][1]}{angle[1][0]}’]
- 5: angle_letter ← random.choice(angle_options)
- 6: angle_class ← ’acute’ if angle[0] < 90 else ’obtuse’
- 7: question ← ’Is angle ’ + angle_letter + ’ acute or obtuse?’
- 8: answer ← ’Angle ’ + angle_letter + ’ is ’ + angle_class
- 9: gt ← angle_class
- 10: data ← {’question’: question, ’answer’: answer, ’gt’: gt}

- Algorithm 4 Data Synthesis for the LineComparison Task

- 1: Input: data_info
- 2: Output: data
- 3: names ← [data_info[0][1], data_info[1][1]]
- 4: lengths ← [data_info[0][0], data_info[1][0]]
- 5: if lengths[0] > lengths[1] then
- 6: longer_name, shorter_name ← names[0], names[1]
- 7: else
- 8: longer_name, shorter_name ← names[1], names[0]
- 9: end if
- 10: data ← [
- 11: { ’question’: ’Which line is longer, ’ + longer_name + ’ or ’ + shorter_name + ’?’,
- 12: ’answer’: ’The longer line is ’ + longer_name,
- 13: ’gt’: longer_name },
- 14: { ’question’: ’Which line is longer, ’ + shorter_name + ’ or ’ + longer_name + ’?’,
- 15: ’answer’: ’The longer line is ’ + longer_name,
- 16: ’gt’: longer_name }
- 17: ]

- Algorithm 5 Data Synthesis for the Parallel Task

- 1: Input: data_info
- 2: Output: data
- 3: points_set ← data_info
- 4: for line_points ∈ points_set do
- 5: for (A, B) ∈ permutations(line_points, 2) do
- 6: all_rest_lines ← [p for p in points_set if p != line_points]
- 7: gts ← [‘’.join(
- 8: f‘{p}’ for line in all_rest_lines for p in line)
- 9: ]
- 10: rest_point_pairs ← []
- 11: for rest_line ∈ all_rest_lines do
- 12: C, D ← random.sample(rest_line, 2)
- 13: rest_point_pairs.append([C, D])
- 14: end for
- 15: all_possible_answer ← ‘, ’.join(
- 16: [f‘{C}{D}’ for C, D in rest_point_pairs]
- 17: )
- 18: verb_agreement ← ‘is’ if len(rest_point_pairs) == 1 else ‘are’
- 19: question ← ‘What is the line parallel to line ’ + A + B + ‘?’
- 20: answer ← (
- 21: ‘According to the diagram, the line parallel to ’ +
- 22: A + B + verb_agreement + all_possible_answer
- 23: )
- 24: gt ← ‘, ’.join(gts)
- 25: data ← {
- 26: ‘question’: question, ‘answer’: answer, ‘task’: task, ‘gt’: gt
- 27: }
- 28: end for
- 29: end for

- Algorithm 6 Data Synthesis for the Perpendicular Task

- 1: Input: data_info
- 2: Output: data
- 3: source_lines, target_lines ← data_info
- 4: all_possible_answer ← []
- 5: gts ← target_lines ▷ Randomly choose two points from each target line
- 6: for target_line ∈ target_lines do
- 7: C, D ← random.sample(target_line, 2)
- 8: all_possible_answer.append(f‘{C}{D}’)
- 9: end for
- 10: verb_agreement ← ‘is’ if len(all_possible_answer) == 1 else ‘are’
- 11: for (A, B) ∈ permutations(source_line, 2) do
- 12: question ← ‘What is the line perpendicular to line ’ + A + B + ‘?’
- 13: answer ← (
- 14: ‘According to the diagram, the line perpendicular to ’ +
- 15: A + B + verb_agreement + ‘, ’.join(all_possible_answer
- 16: )
- 17: gt ← ‘, ’.join(gts)
- 18: data ← {
- 19: ‘question’: question, ’, ‘answer’: answer, ’‘ask’: task, ‘gt’: gt
- 20: }
- 21: end for

- Algorithm 7 Data Synthesis for the Equal Task

- 1: Input: data_info
- 2: Output: data
- 3: statement, content ← data_info.split(‘;’)
- 4: if statement == ‘angles_value’ then
- 5: angle_letter, angle_measure ← content.split(‘=’)
- 6: angle_letter ← random.choice([angle_letter, angle_letter[::-1]])
- 7: question ← ‘What is the measure of angle ’ + angle_letter + ‘ as annotated?’
- 8: answer ← ‘Angle ’ + angle_letter + ‘ is annotated as ’ + angle_measure
- 9: gt ← angle_measure
- 10: else if statement == ‘segments_value’ then
- 11: segment_letter, segment_length ← content.split(‘=’)
- 12: segment_letter ← random.choice([segment_letter, segment_letter[::-1]])
- 13: question ← ‘What is the length of line ’ + segment_letter + ‘ as annotated?’
- 14: answer ← ‘Line ’ + segment_letter + ‘ is annotated as ’ + segment_length
- 15: gt ← segment_length
- 16: else if statement == ‘angles’ then
- 17: angle1, angle2 ← content.split(‘=’)
- 18: angle1 ← random.choice([angle1, angle1[::-1]])
- 19: angle2 ← random.choice([angle2, angle2[::-1]])
- 20: query_angle ← random.choice([angle1, angle2])
- 21: answer_angle ← angle2 if query_angle == angle1 else angle1
- 22: question ← ‘What is the angle in the diagram that is equal to angle ’ + query_angle
- 23: answer ← ‘Angle ’ + query_angle + ‘ is equal to angle ’ + answer_angle
- 24: gt ← answer_angle
- 25: else if statement == ‘segments’ then
- 26: segment1, segment2 ← content.split(‘=’)
- 27: segment1 ← random.choice([segment1, segment1[::-1]])
- 28: segment2 ← random.choice([segment2, segment2[::-1]])
- 29: query_segment ← random.choice([segment1, segment2])
- 30: answer_segment ← segment2 if query_segment == segment1 else segment1
- 31: question ← ‘What is the segment in the diagram that is equal to segment ’

+ query_segment

- 32: answer ← ‘Segment ’ + query_segment + ‘ is equal to segment ’ + answer_segment
- 33: gt ← answer_segment
- 34: end if
- 35: data ← {
- 36: ‘question’: question, ‘answer’: answer, ‘task’: task, ‘gt’: gt
- 37: }

###### C.2. Geometry Shapes Used for Euclid Training

Geometry Shape Generation Code

###### PointLiesOnLine

- (stage 1) A B C = triangle A B C; D = midpoint B C

- (stage 1) A B C = triangle A B C; D = midpoint B C; O = circle O A B C
- (stage 2) A B C = triangle A B C; D = midpoint A B; E = midpoint A C

(stage 2) A B C = triangle A B C; D = midpoint A B; E = midpoint A C; O = circle O A B C (stage 3) A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E

- (stage 3) A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E; O = circle O A B C

###### PointLiesOnCircle

- (stage 1) A B = segment A B; C = on_circle C A B

- (stage 1) A B = segment A B; C = on_circle C A B; D = on_circle D A B

- (stage 1) A B = segment A B; C = on_circle C A B; D = on_circle D A B; E = on_circle E A B

- (stage 1) A B = segment A B; C = on_circle C A B; D = on_circle D A B; E = on_circle E A B; F = on_circle F A B

- (stage 1) A B = segment A B; C = on_circle C A B; D = on_circle D A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B
- (stage 2) A B = segment A B; C = on_circle C A B; D = midpoint A B

- (stage 2) A B = segment A B; C = on_circle C A B; D = midpoint A B

- (stage 2) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B

- (stage 2) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B

- (stage 2) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B

- (stage 2) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B; H

= on_circle H A B

- (stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C (stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B (stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B; G = on_circle G A B (stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B; G = on_circle G A B; H =

on_circle H A B

(stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B; G = on_circle G A B; H = on_circle H A B; I = on_circle I A B

(stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B; H

= on_circle H A B; I = midpoint B C (stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint B C (stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = lc_tangent E C A (stage 3) A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B; H

###### = lc_tangent H C A AngleClassification

(stage 1) A B C = triangle A B C (stage 3) A B C = triangle A B C; D = midpoint B C (stage 3) A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll F A D B E

###### LengthComparison

- (stage 1) A B C = triangle A B C
- (stage 2) A B C = triangle A B C; D = midpoint B C
- (stage 3) A B C = triangle A B C; D = midpoint A B; E = midpoint A C

###### Parallel

(stage 1) A B C = triangle A B C; D = midpoint A B; E = midpoint A C (stage 1) A B C = triangle A B C; D = midpoint A B; E = midpoint A C

- (stage 1) A B C = triangle A B C; D = midpoint A B; E = midpoint A C
- (stage 2) A B C = triangle A B C; D = parallelogram A B C D
- (stage 3) A B C = triangle A B C; D = midpoint A B; E = midpoint A C; F = midpoint B C

###### Perpendicular

- (stage 1) A B C = triangle A B C; D = foot A B C

- (stage 1) A B C = r_triangle A B C

- (stage 1) A B = segment A B; C = eq_triangle C A B; D = eq_triangle D A B; E = on_circle E A B
- (stage 2) A B C = triangle A B C; D = foot A B C; E = foot C A B

(stage 2) A B C = r_triangle A B C; D = foot A B C (stage 2) A B C = triangle A B C; O = circle A B C; D = foot O A B; E = foot O C A (stage 3) A B C D = rectangle A B C D; E = intersection_ll A C B D (stage 3) A B C = triangle A B C; O = incenter A B C; D = foot O A C; E = foot O B C; F = foot O A B

- (stage 3) A B C = r_triangle A B C; D = foot A B C; E = foot D A B (stage 3) A B C = triangle A B C; D = foot A B C; E = foot C A B; F = foot B A C

###### Equal

- (stage 1) A B C = triangle A B C; D = midpoint C B

- (stage 1) A B C = triangle A B C; D = midpoint C B; O = circle O A B C

- (stage 1) A B C = triangle A B C; D = angle_bisector B A C, on_line D C B
- (stage 2) A B C = triangle A B C; D = midpoint A B; E = midpoint A C

- (stage 2) A B C = triangle A B C; D = midpoint A B; E = midpoint A C; O = circle O A B C

- (stage 2) A B C = triangle A B C; D = midpoint A B; E = midpoint A C
- (stage 3) A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B

Figure 17: Geometry Shape Generation Code for Euclid training

PointLiesOnLine

Stage 1 Stage 2 Stage 3

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

PointLiesOnCircle

Stage 1 Stage 2 Stage 3

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

AngleClassification

Stage 1 Stage 2 Stage 3

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

LineComparison

Stage 1 Stage 2 Stage 3

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Parallel

Stage 1 Stage 2 Stage 3

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Perpendicular

Stage 1 Stage 2 Stage 3

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Equal

Stage 1 Stage 2 Stage 3

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

- Figure 18: Examples of the geometry diagrams used to train Euclid, the diagrams are generated by our dataset engine.

#### D. Additional Experimental Results

###### LengthComparison PointLiesOnLine

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 10 2

- 10 1

100

Training Loss

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 10 20 30

0.0

0.2

0.4

0.6

0.8

1.0

Testing Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 10 20 30

10 3

10 2

10 1

100

Training Loss

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 10 20 30

0.0

0.2

0.4

0.6

0.8

1.0

Testing Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 30 60 90 # Training Samples (k)

10 3

- 10 2

10 3

0 10 20 30

1.0

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

100

100

0.8

0.8

10 1

10 1

0.6

0.6

10 2

0.4

0.4

0.2

0.2

10 3

0.0

0.0

0 30 60 90 # Training Samples (k)

0 30 60 90 # Training Samples (k)

0 30 60 90 # Training Samples (k)

Difficulty: Easy Difficulty: Medium Difficulty: Hard Mixed Difficulty

- Figure 19: Result of our preliminary experiments, we use a standard setting of MLLMs: an OpenAI-CLIP@224 as visual encoders (Radford et al., 2021), two-layer MLP as multimodal connector and Qwen-2.5-1.5B as language model. We find that the model can reach convergence in some of the easy tasks, while struggle to learn hard tasks. We also find mixed training is better than separate training, given the same amount of training data in each difficulty level.

Difficulty: easy

Difficulty: medium

Difficulty: hard

mix

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100

100

100

100

TrainingLoss

10 1

10 1

10 1

10 1

10 2

10 2

10 2

10 2

10 3

10 3

10 3

10 3

10 4

10 4

10 4

10 4

0 10 20 30

0 10 20 30

0 10 20 30

0 30 60 90

1.0

1.0

1.0

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 10 20 30

0 10 20 30

0 10 20 30

0 30 60 90

###### PointLiesOnLine

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100

100

100

100

TrainingLoss

10 1

10 1

10 1

10 1

10 2

10 2

10 2

10 2

10 3

10 3

10 3

10 3

10 4

10 4

10 4

10 4

0 10 20 30

0 10 20 30

0 10 20 30

0 30 60 90

1.0

1.0

1.0

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 10 20 30 # Training Samples (k)

0 10 20 30 # Training Samples (k)

0 10 20 30 # Training Samples (k)

0 30 60 90 # Training Samples (k)

Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct

- Figure 20: The complete result of the effect of LLM size. The finding is similar with Fig. 3.

LineComparison

Difficulty: easy

Difficulty: medium

Difficulty: hard

mix

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100

100

100

100

TrainingLoss

10 1

10 1

10 1

10 1

10 2

10 2

10 2

10 2

10 3

10 3

10 3

10 3

10 4

10 4

10 4

10 4

0 10 20 30

0 10 20 30

0 10 20 30

0 30 60 90

1.0

1.0

1.0

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 10 20 30

0 10 20 30

0 10 20 30

0 30 60 90

PointLiesOnLine

100

100

100

100

TrainingLoss

10 1

10 1

10 1

10 1

10 2

10 2

10 2

10 2

10 3

10 3

10 3

10 3

10 4

10 4

10 4

10 4

0 10 20 30

0 10 20 30

0 10 20 30

0 30 60 90

1.0

1.0

1.0

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 10 20 30 # Training Samples (k)

0 10 20 30 # Training Samples (k)

0 10 20 30 # Training Samples (k)

0 30 60 90 # Training Samples (k)

ConvNeXT-L@512

ViT-L@224

ViT-g@224 ViT-H@224

DINO-G@224

ConvNeXT-XXL@512

SigLIP@224

DINO-L@224

- Figure 21: The complete result of the effect of different visual encoders. The finding is similar with Fig. 4.

ConvNeXT-L@512

ConvNeXT-XXL@512

ViT-L@224

ViT-H@224

freeze Vi-Enc

freeze Vi-Enc

100

100

100

100

unfreeze Vi-Enc

unfreeze Vi-Enc

10 1

10 1

10 1

10 1

TrainingLoss

10 2

10 2

10 2

10 2

10 3

10 3

10 3

10 3

freeze Vi-Enc

freeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

0 30 60 90

0 30 60 90

0 30 60 90

0 30 60 90

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

freeze Vi-Enc

freeze Vi-Enc

freeze Vi-Enc

freeze Vi-Enc

0.2

0.2

0.2

0.2

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

0.0

0.0

0.0

0.0

0 30 60 90

0 30 60 90

0 30 60 90

0 30 60 90

PointLiesOnLine

freeze Vi-Enc

100 freeze Vi-Enc

100 freeze Vi-Enc

100 freeze Vi-Enc

100

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

10 1

10 1

10 1

10 1

TrainingLoss

10 2

10 2

10 2

10 2

10 3

10 3

10 3

10 3

0 30 60 90

0 30 60 90

0 30 60 90

0 30 60 90

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

TestingAccuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

freeze Vi-Enc

freeze Vi-Enc

freeze Vi-Enc

freeze Vi-Enc

0.2

0.2

0.2

0.2

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

unfreeze Vi-Enc

0.0

0.0

0.0

0.0

0 30 60 90 # Training Samples (k)

0 30 60 90 # Training Samples (k)

0 30 60 90 # Training Samples (k)

0 30 60 90 # Training Samples (k)

- Figure 22: The complete result of the effect of tuning visual encoders. The finding is similar with Fig. 5.

