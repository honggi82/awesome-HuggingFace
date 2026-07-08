## MVI-Bench: A Comprehensive Benchmark for Evaluating Robustness to Misleading Visual Inputs in LVLMs

# arXiv:2511.14159v3[cs.CV]27May2026

#### Huiyi Chen1 Jiawei Peng*1 Dehai Min*2 Changchang Sun*2 Kaijie Chen3 Yan Yan2 Xu Yang1 Lu Cheng2

https://github.com/chenyil6/MVI-Bench

### Abstract

- (a) Misleading Textual Input (MAD-Bench) (Normal Question) How many dogs are there?

[Figure 1]

There are two dogs in image. (Misleading Question) What color is the cat in the image? The cat in the image is black and white.

- (b) Misleading Visual Input (MVI-Bench)

[Figure 2]

Evaluating the robustness of Large VisionLanguage Models (LVLMs) is essential for their continued development and responsible deployment. However, existing robustness benchmarks largely focus on hallucination or misleading textual inputs, overlooking the critical challenge posed by misleading visual inputs in assessing visual understanding. To fill this gap, we introduce MVI-Bench, the first comprehensive benchmark specially designed for evaluating how Misleading Visual Inputs undermine the robustness of LVLMs. Grounded in fundamental visual primitives, the design of MVI-Bench centers on three hierarchical levels of misleading visual inputs: Visual Concept, Visual Attribute, and Visual Relationship. Using this taxonomy, we curate six representative categories and compile 1,248 expertly annotated VQA instances. To facilitate fine-grained robustness evaluation, we further introduce MVI-Sensitivity, a novel metric that characterizes LVLM robustness. Empirical results across 18 state-of-the-art LVLMs uncover pronounced vulnerabilities to misleading visual inputs, and our in-depth analyses on MVI-Bench provide actionable insights that can guide the development of more reliable and robust LVLMs.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

(Normal Image) What is in the image?

(Misleading Image) What is in the image?

[Figure 10]

[Figure 11]

###### mushroom

stool

Figure 1. (a) Misleading Textual Input: misleading questions are created by injecting inaccurate or irrelevant information into otherwise normal queries. (b) Misleading Visual Input: misleading visual cues arise from real-world scenes, causing models to misinterpret the image content (e.g., stools mistaken for mushrooms).

across a wide range of multimodal tasks, including image understanding (Chen et al., 2015; Wang et al., 2024a; Yang et al., 2023), visual question answering (Antol et al., 2015; Kuang et al., 2025; Li et al., 2024c; Peng et al., 2024), and complex visual reasoning (Wang et al., 2024c; Yue et al.,

- 2024). With these rapid developments comes an urgent need for rigorous evaluation of LVLM robustness, motivating the creation of numerous robustness benchmarks (Fu et al., 2023; Agarwal et al., 2025; Jiang et al., 2025a).

While existing robustness benchmarks have provided valuable insights, they largely focus on specific challenges, such as hallucination (Bai et al., 2024; Wang et al., 2023) or adversarial robustness (Jiang et al., 2025a; Agarwal et al.,

- 2025), and thus offer only a partial view of LVLM robustness. To broaden this scope, recent efforts (Dang et al., 2024; Deng et al., 2025; Qian et al., 2024) have begun exploring LVLM behavior under misleading scenarios, typically by introducing misleading textual inputs. For instance, MADBench (Qian et al., 2024) constructs misleading questions by injecting inaccurate information into textual prompts (Fig. 1(a)). However, misleading information is not confined to the language modality. In real-world settings, misleading

### 1. Introduction

Recent advances in Large Vision-Language Models (LVLMs) (Yin et al., 2024) have driven remarkable progress

*Equal contribution 1School of Computer Science & Engineering, Southeast University, Nanjing, China 2Department of Computer Science, University of Illinois Chicago, Chicago, USA 3Guohao School, Tongji University, Shanghai, China. Correspondence to: Huiyi Chen <kitty22hy@gmail.com>, Lu Cheng <lucheng@uic.edu>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

[Figure 12]

[Figure 13]

###### (a) Advertisement billboard (b) Vehicle's perception view

- Figure 2. A real-world example of misleading visual inputs that deceive machines: the vehicle’s perception system misidentifies the 2D content on an advertisement billboard (a) as actual road objects (b), triggering unnecessary emergency braking.

visual inputs naturally arise and can deceive both machines (see Fig. 2) and humans.

This phenomenon has long been examined in philosophy under the notion of the Problem of Perception (Smith, 2005), which emphasizes that human visual cognition is prone to systematic errors induced by perceptual biases. As illustrated in Fig. 1(b), humans may initially misinterpret visually ambiguous scenes, such as perceiving stools as mushrooms, but can often correct these misconceptions through contextual reasoning and prior knowledge.

This observation reveals a fundamental challenge for LVLMs: unlike humans, who can correct initial misperceptions through contextual reasoning and prior knowledge, models often rely on superficial visual patterns and struggle to disambiguate misleading cues. Yet, this critical dimension remains underexplored. Moreover, prior works on misleading scenarios typically focus on textual inputs with deliberately injected misinformation, which rarely occurs in real-world interactions and differs fundamentally from naturally arising visual misleading cues.

This gap motivates us to introduce the Misleading Visual Input Benchmark (MVI-Bench), the first comprehensive benchmark specifically designed to evaluate the robustness of LVLMs against misleading visual inputs. Grounded in fundamental visual primitives (Han et al., 2025), the design of MVI-Bench is organized around three hierarchical levels of misleading cues: Visual Concept, Visual Attribute, and Visual Relationship. Each level further comprises several representative misleading categories that closely mirror the types of perceptual errors humans encounter in real-world visual experiences, as detailed in Section 3.1. MVI-Bench includes a total of 624 pairs of VQA instances (i.e., 1,248 instances in total), where each pair consists of a normal image and its corresponding misleading counterpart. The two images within a pair share nearly identical semantic

content, differing only by the introduction of subtle misleading visual cues (see Fig. 3 for details). To complement this paired design, we further propose MVI-Sensitivity, a novel evaluation metric that quantifies the relative performance degradation from normal to misleading visual inputs. This metric provides a fine-grained measure of LVLM robustness, enabling a deeper understanding of model sensitivity to misleading visual inputs.

In summary, our main contributions are as follows:

- • We establish the first comprehensive taxonomy of misleading visual inputs. Grounded in fundamental visual primitives, this taxonomy systematically characterizes Visual Concept, Visual Attribute, and Visual Relationship as the core dimensions for evaluating LVLM behavior under misleading visual conditions.
- • We introduce MVI-Bench, a carefully curated benchmark comprising six representative categories and 624 expertly annotated VQA instance pairs. Each pair is designed to isolate the effects of misleading visual cues while preserving the underlying semantic content.
- • We propose MVI-Sensitivity, a novel evaluation metric that enables fine-grained robustness assessment by quantitatively measuring the performance degradation of LVLMs when exposed to misleading visual inputs.
- • Through extensive experiments and in-depth analyses on 18 open- and closed-source LVLMs, we uncover three key observations: limited LVLMs robustness, visual perception and reasoning are complementary, and the presence of spurious correlation. These findings offer actionable insights for developing more reliable and robust LVLMs.

### 2. Related Work

Large Vision-Language Models (LVLMs). LVLMs unify visual perception and visual reasoning through an architecture comprising a vision encoder, a large language model (LLM), and a connector module that aligns visual and textual feature spaces, enabling seamless multimodal understanding (Awadalla et al., 2023; Dai et al., 2023; Bai et al., 2023; Liu et al., 2023; Zhu et al., 2023; Dong et al., 2025; Abdin et al., 2024; Chen et al., 2024; Team et al., 2025). Leveraging this architecture, LVLMs have rapidly evolved from early multimodal frameworks such as Flamingo (Awadalla et al., 2023) and InstructBLIP (Dai et al., 2023) to a new generation of high-performing models, such as LLaVA (Liu et al., 2023) and Qwen-VL (Bai et al., 2023). When deployed in real-world settings, LVLMs exhibit remarkable advancements across multiple multimodal tasks, including image understanding (Chen et al., 2015; Wang et al., 2024a; Chen et al., 2025), visual question answering (Antol et al., 2015; Kuang et al., 2025; Jiang et al., 2025b), and complex visual reasoning (Wang et al., 2024c;

###### Visual Resemblance

###### Representation Confusion

How many edible cookies are in the image?

What is in the picture?

What is in the picture?

How many people are in the image?

[Figure 14]

[Figure 15]

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

French fries cake 3 6

[Figure 25]

3

cake

3

French fries

[Figure 26]

[Figure 27]

[Figure 28]

2

3

handbag

[Figure 29]

Umbrella

###### Material Confusion

###### Mirror Reflection

How many fragile plates are shown in the image?

How many chairs are there in the image?

How many spoons in the picture are conductive?

How many toys are there in the picture?

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

4 5 2 4

[Figure 39]

[Figure 40]

[Figure 41]

0

1

1

0

[Figure 42]

[Figure 43]

2

1

3

[Figure 44]

[Figure 45]

1

###### Occlusion Confusion Visual Illusion

[Figure 46]

[Figure 47]

Which sentence best describes the scene in the picture?

How many stools are there in the image?

How many balloons is the little girl holding in her hand?

Which of the following options is in the picture?

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

A bottle is placed on 2 4 Big hole top of a pillar.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

A bottle is placed on top of a pillar.

2

Carpet

4

Carpet A bottleground.is on the

[Figure 58]

[Figure 59]

[Figure 60]

4

6

[Figure 61]

- Figure 3. Examples from six misleading categories defined in MVI-Bench. Each pair contains a normal image (left) and misleading image (right) with the same MCQ and corresponding ground-truth answer. Answer choices are omitted for brevity (see Fig. 5 for full format).

Yue et al., 2024; Fu et al., 2026). This progress is reflected across both open-source and proprietary LVLM families. For example, open-source series such as InternVL (Chen

- et al., 2024) and SAIL-VL (Yin et al., 2025) have made substantial progress, while proprietary models like GPT-

- 4 (Achiam et al., 2023) and Gemini (Team et al., 2023) further push the frontier through large-scale multimodal training and enhanced reasoning capabilities.

2024; Deng et al., 2025; Qian et al., 2024). However, robustness against misleading visual inputs, a challenge that naturally arises in real-world environments, remains substantially underexplored. Although a few recent efforts, such as IllusionVQA (Shahgir et al., 2024) and O-Bench (Liu et al., 2025c), have begun to investigate this direction, they focus on a single type of misleading phenomenon and rely on coarse metrics such as accuracy, falling short of fine-grained robustness analysis. MVI-Bench addresses these limitations through comprehensive category coverage and a paired evaluation design, enabling unique insights unattainable from prior benchmarks.

Robustness Benchmarks for LVLMs. Existing studies on LVLM robustness have primarily focused on two perspectives. On the one hand, prior works examine hallucination issues, which evaluates the discrepancy between generated responses and the visual content (Bai et al., 2024; Wang et al., 2023; Jiang et al., 2024). On the other hand, a separate line of research investigates adversarial vulnerabilities, analyzing how models behave under carefully crafted perturbated inputs (Jiang et al., 2025a; Agarwal et al., 2025; Zhao et al., 2023). However, these evaluations illuminate only a partial view of LVLM robustness. To broaden this scope, recent researches have begun to investigate LVLM behavior under misleading scenarios, most commonly through misleading textual inputs (Liu et al., 2025b; Dang et al.,

### 3. MVI-Bench

To comprehensively evaluate LVLM robustness under misleading visual inputs, we propose MVI-Bench, a paired VQA benchmark. Specifically, each instance pair shares the same multiple-choice question (MCQ) with four options (A-D) but differs in visual input: a normal image and its misleading counterpart (Fig. 3). The two images are visually similar in overall composition, but the misleading one

- Table 1. Comparison between MVI-Bench and prior visually misleading benchmarks. MVI-Bench uniquely covers six well-defined misleading categories and includes diverse image sources with a paired design, enabling comprehensive and controlled robustness evaluation. (Abbreviations: Res.-Visual resemblance; Rep.-Representation confusion; Mat.-Material confusion; Mir.-Mirror reflection; Occ.-Occlusion confusion; Ill.-Visual illusion.)

Image Source

Dataset Venue #Image Res. Rep. Mat. Mir. Occ. Ill.

Paired Natural Synthetic Edited

IllusionVQA (Shahgir et al., 2024) COLM 2024 374 ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✗ ✗ IllusionBench+ (Zhang et al., 2025b) arXiv 2025.6 1,051 ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✗ ✗ O-Bench (Liu et al., 2025c) arXiv 2025.8 1,365 ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✓ ✗ ✗ MVI-Bench (ours) – 1,248 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

introduces subtle cues designed to confuse the model. This paired design enables controlled analysis of how visual misleading affects LVLMs’ behavior: for each image, only one option is correct, and at least one distractor is intentionally crafted to appear plausible given the misleading visual cues. Overall, MVI-Bench comprises 624 such pairs (1,248 instances total) across six categories, enabling systematic evaluation under visually misleading conditions. Tab. 1 presents a detailed comparison of MVI-Bench with existing related benchmarks.

#### 3.1. Taxonomy of Visually Misleading Types

We ground our taxonomy in the principles of fundamental visual primitives (Han et al., 2025), namely, Visual Concept, Visual Attribute, and Visual Relationship1. These primitives form the core of visual cognition and offer a structured lens through which misleading phenomena can be characterized. Accordingly, we define three corresponding levels of misleading visual inputs and their associated categories.

- 1. Visual Concept Misleading. Misleading effects at this level arise from coarse-grained visual similarity between semantically different entities (e.g., objects, people, scenes). Addressing such misleading inputs requires finer visual discrimination and modest reasoning that goes beyond superficial pattern matching. This level includes two categories:

- • Visual Resemblance. Cases where models may confuse an object with a visually similar but semantically different object or an object from a different category.
- • Representation Confusion. Cases where models can fail to distinguish between an actual object and its two-dimensional representation (e.g., photograph or painting).

- 2. Visual Attribute Misleading. This level captures misleading cases from fine-grained ambiguity in visual attributes (e.g., texture, gloss, material). Correctly resolving these cases requires refined perceptual discrimination to differentiate materials that may look similar at a coarse level but differ fundamentally. This level includes one category:

1We exclude “visual combination”, since our aim is to isolate and analyze the effect of each individual misleading level.

• Material confusion. Cases where models sometimes confuse items based on their texture, material, or other physical attributes.

3. Visual Relationship Misleading. This level encompasses misleading cases that arise from incorrect interpretation of visual relationships, which describe spatial arrangements, part–whole structures, or contextual dependencies among entities. Unlike concept- or attribute-level misleading, relationship-level misleading requires the model to reason about higher-order spatial arrangements and interactions between objects. This level includes three categories:

- • Mirror Reflection. Cases where models can be misled by virtual images from mirror reflections, confusing them with actual, real-world objects.
- • Occlusion Confusion. Cases where models struggle to recognize an object when part of it is visually occluded, sometimes leading to incorrect identification.
- • Visual Illusion. Cases where models are susceptible to visual illusions arising from complex spatial arrangements, lighting conditions, or deceptive perspectives.

Above all, these three levels and six categories form a comprehensive taxonomy for evaluating LVLMs’ susceptibility to misleading visual inputs. This taxonomy spans the full spectrum of visual understanding from low-level recognition and mid-level attribute discrimination to high-level spatial and contextual reasoning. Representative examples from each category are illustrated in Fig. 3.

#### 3.2. Data Curation

Data Collection. Following the taxonomy of visually misleading types described above, three trained human experts with at least a bachelor’s degree manually curated an initial set of image pairs from three complementary sources to provide broad topical coverage:

- (1) Natural images, collected from multiple international social media platforms to ensure broad and diverse realworld content coverage;
- (2) Synthetic images, generated using the generative model Seedream (Seedream et al., 2025) to supplement misleading

[Figure 62]

[Figure 63]

1.00

[Figure 64]

0.95

0.9188

[Figure 65]

[Figure 66]

[Figure 67]

0.8851 0.8916

[Figure 68]

[Figure 69]

0.90

[Figure 70]

[Figure 71]

Edited Images

0.85

0.8255

210

[Figure 72]

0.8099

Natural Images

0.7776

0.80

475 (38.06%)

0.75

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

MVI-Bench Books MVI-Bench Lighting

[Figure 79]

653 (52.32%)

0.70

[Figure 80]

[Figure 81]

0.65

0.60

Synthetic Images

208

Average Cosine Similarity for Visual Resemblance

Average Cosine Similarity for Mirror Reflection

Average Cosine Similarity for Representation Confusion

Average Cosine Similarity for Occlusion Confusion

120 (9.62%)

Average Cosine Similarity for Material Confusion

Average Cosine Similarity for Visual Illusion

(a) Misleading Categories of MVI-Bench

(b) Image Sources of MVI-Bench

(c) Object Elements included in MVI-Bench

(d) Image Similarity among Categories

Figure 4. Overview of MVI-Bench statistics.

cases that are rare and low-quality in natural data;

compassing a wide range of categories such as food, furniture, nature, and decorations. Moreover, each normal and misleading image pair is meticulously designed to isolate the effect of visual misleading cues while preserving other semantic content, ensuring minimal differences between paired images. Fig. 4 (d) shows empirical evidence for this design by showing high cosine similarity (using features obtained using CLIP-Large (Radford et al., 2021)) between paired images across all six categories.

(3) Edited images, created by human experts using advanced image editing tools to modify natural or synthetic images, typically by removing misleading visual cues to serve as normal counterparts.

Normal images in our benchmark primarily originate from natural (23.56%) and edited images (76.44%), while misleading images are mainly sourced from natural (81.09%) and synthetic ones (18.91%). For each image, human experts craft a MCQ based on its visual content, with exactly one correct answer and at least one distractor that plausibly leverages the misleading visual cues. Additionally, each annotated VQA pair including its label (“misleading” or “normal”) and the answer is independently reviewed by additional human experts, and any controversial cases are discarded to ensure quality.

#### 3.3. Evaluation Metrics

We employ two evaluation metrics. First, we report Accuracy on normal and misleading examples to directly measure model performance under subtly different visual conditions. Second, we introduce MVI-Sensitivity, a new metric designed to quantify the extent to which an LVLM’s performance degrades when exposed to misleading visual inputs, compared to corresponding normal images:

Data Filtering and Refinement. To further ensure the discriminative value of MVI-Bench, we evaluate the initially collected data using two strong LVLMs: InternVL3-2B (Chen et al., 2024) and Qwen2.5-VL-3B (Bai et al., 2025). Based on the evaluation results, we then identify instances where both models correctly answer the questions for both normal and misleading images. Since these instances pose little challenge and provide limited ability to differentiate model robustness, we treat them as lowdifficulty samples. To maintain dataset diversity while preventing such easy cases from dominating the benchmark, we randomly retain only a portion of them. After this refinement process, the dataset is reduced from 1,578 to 1,248 high-quality, discriminative VQA instances.

MVI-Sensitivity = |Accn − Accm|

, (1)

Accn

where Accn and Accm denote the accuracy on normal and misleading samples, respectively. A lower MVI-sensitivity indicates that the model is less affected by misleading cues and exhibits stronger robustness.

### 4. Experiments

We employ MVI-Bench to comprehensively study LVLMs’ behavior under misleading visual inputs. Our experimental design centers on the following research questions:

Data Statistics. Our final benchmark comprises 1,248 VQA instances. Fig. 4 illustrates the overall statistics of MVIBench from four complementary perspectives. Fig. 4 (a) shows the data distribution across our taxonomy, with roughly balanced instances per category for unbiased evaluation. Fig. 4 (b) depicts the composition of our three image sources: natural images (52.32%), synthetic images (9.62%), and edited images (38.06%). Fig. 4 (c) shows that the benchmark exhibits broad semantic coverage, en-

- • RQ1: How do LVLMs perform on MVI-Bench across different categories of misleading visual inputs?
- • RQ2: How do visual perception and reasoning abilities contribute respectively to LVLMs robustness in visually misleading scenarios?
- • RQ3: What insights can be derived from counterintuitive cases where LVLMs behave unexpectedly under misleading visual inputs?

#### 4.1. Models and Experimental Settings

We evaluate 18 state-of-the-art LVLMs on MVI-Bench to systematically assess their robustness against misleading visual inputs. These models span a diverse range of architectures and scales. Specifically, it includes six closed-source models: GPT-4o (Hurst et al., 2024), GPT-4.1 (Achiam et al., 2023), Claude-3.7-Sonnet (Anthropic, 2025), Gemini2.5-Flash (Team et al., 2023), Gemini-2.5-Pro (Team et al.,

- 2023) and GPT-5-Chat (OpenAI, 2025); and twelve opensource models, including Qwen2.5-VL series (Bai et al., 2025), the InternVL-3 series (Zhu et al., 2025a), the SAILVL2 series (Yin et al., 2025), LLaVA-OneVision (Li et al.,
- 2024b) and Molmo (Deitke et al., 2025). For closed-source models, the temperature is fixed at 0.0, and for open-source models, the decoding is performed using a greedy strategy. Meanwhile, modern LLMs are known to be vulnerable to option-position variations in MCQs due to their inherent ‘selection bias’ (Zheng et al., 2023), namely a tendency to prefer specific option IDs as answers. To mitigate this bias, we randomly shuffle the answer options each time the LVLMs are queried.

#### 4.2. Main Results (RQ1)

We report the evaluation results of 18 LVLMs in Tab. 2 across 6 misleading categories in MVI-Bench. The main findings are summarized as follows.

Existing LVLMs remain highly susceptible to misleading visual inputs, especially open-source ones. As shown in Tab. 2, all evaluated LVLMs exhibit substantial performance degradation when exposed to misleading visual cues. On the closed-source side, even the latest state-of-the-art models are not immune. For instance, GPT-5-Chat’s accuracy drops from 90.00% to 61.00% when visual illusion is introduced, and Gemini-2.5-Pro’s accuracy decreases from 86.54% to 54.81% under mirror reflection. All closed-source LVLMs yield overall MVI-Sensitivity above 20%, indicating substantial vulnerability. Notably, Claude-3.7-Sonnet, despite its strength on reasoning-intensive tasks such as coding, performs the worst among closed-source models, achieving only 42.13% accuracy on misleading images (Accm) and an MVI-Sensitivity of 42.10%.

In contrast, open-source models demonstrate greater susceptibility. The strongest, Qwen2-VL-72B, reaches 32.38% MVI-Sensitivity, yet lags noticeably behind top closedsource counterparts. Molmo-7B, a fully-open model releasing both its training data and recipes, performs worst at 48.69% MVI-Sensitivity, meaning nearly half of its responses are affected by misleading cues. The performance gap between open- and closed-source models may stem from several factors (Li et al., 2024a; Laurenc¸on et al., 2024; Chen et al., 2024): closed-source models typically benefit from larger-scale, higher-quality proprietary training

data (Sun et al., 2024; Ye et al., 2025; Min et al., 2025) and more extensive computational budgets enabling sophisticated post-training refinement techniques (Yu et al., 2025; Li et al., 2024a).

LVLMs demonstrate stronger mastery of coarse-grained visual concepts than fine-grained visual attributes. Within the visual concept level, both visual resemblance and representation confusion involve coarse-grained semantic discrimination. LVLMs consistently show high and stable robustness on these two categories: most LVLMs models exceed or approach 50% Accm in visual resemblance, and all models except Molmo maintain over 40% Accm in representation confusion. These results indicates that current LVLMs handle high-level semantic discrimination more reliably than other types of misleading visual inputs. For example, Qwen2-VL-72B achieves highest 76.19% Accm and 15.79% MVI-sensitivity in visual resemblance, and GPT-4.1 reaches an 74.77% Accm with the lowest MVI-Sensitivity of 8.04% in representation confusion. However, performance drops considerably in material confusion, where the best model, Gemini-2.5-Pro, attains only 66.00% Accm and GPT-4o falls even further to 43.00%.

Spatial reasoning remains a critical weakness of current LVLMs. Mirror reflection is common in everyday life, yet it remains highly challenging for LVLMs. Even Qwen2-VL-72B, achieves only 50.00% Accm, and GPT-4o reaches merely 52.88%. This weakness is more pronounced among smaller models: InternVL3-2B, Qwen2.5-VL-3B, and LLaVA-OneVision-7B all exhibit MVI-Sensitivity exceeding 70%. This indicates that while LVLMs perform well on images without mirror reflections (Accn often exceeds 80%), their performance drops sharply once virtual reflections appear, suggesting fundamental difficulties in distinguishing real objects from mirror images.

Similarly, occlusion confusion reveals another spatial reasoning limitation. Only Gemini-2.5-Flash, Gemini-2.5-Pro and GPT-5-Chat surpass 50% Accm, while all other models fall below, highlighting a substantial deficiency in occlusion perception—a capability that underpins human spatial understanding. Notably, performance differences across models on visual illusion are less pronounced than in other categories. For example, InternVL3’s Accm varies only between 48% and 52% as model size increases from 2B to 78B. This may stem from the scarcity of visual illusion examples in multimodal pre-training datasets, suggesting targeted data augmentation as a promising direction.

4.3. Assessing the Impact of Visual Perception and Reasoning (RQ2)

To understand what drives robustness against misleading visual inputs, we separately investigate the impact of visual perception and visual reasoning capabilities through

- Table 2. Performance comparison on MVI-Bench. Each category reports model and human performance on normal (Accn) and misleading (Accm) images, as well as MVI-Sensitivity (Sens). For each category and the overall results, the best score is in bold and the second best is underlined. Accm evaluates performance under misleading visual cues (↑: higher is better), while Sens measures the relative performance drop from normal to misleading inputs (↓: lower is better).

Misleading Level Concept Attribute Relationship

Overall Misleading Type Resemblance Representation Material Mirror Occlusion Illusion

Model Accn → Accm Sens Accn → Accm Sens Accn → Accm Sens Accn → Accm Sens Accn → Accm Sens Accn → Accm Sens Accm Sens

Open-source Models ↑ ↓ LLaVA-OV (7B) 89.52 → 66.67 25.53 82.24 → 50.47 38.63 77.67 → 45.63 41.25 88.46 → 23.08 73.91 70.48 → 29.52 58.12 87.00 → 47.00 45.98 43.75 46.99

Molmo (7B) 82.52 → 49.52 40.23 70.09 → 23.36 66.67 65.05 → 34.95 46.27 80.77 → 21.15 73.81 62.86 → 48.57 22.73 79.00 → 49.00 37.97 37.66 48.69 SAIL-VL2 (2B) 91.43 → 53.33 41.67 79.44 → 65.42 17.65 73.79 → 46.60 36.85 93.27 → 36.54 60.85 68.57 → 39.05 43.06 85.00 → 48.00 43.53 48.23 41.10 SAIL-VL2 (8B) 90.48 → 57.14 36.85 81.31 → 63.55 21.84 72.82 → 45.63 37.34 92.31 → 39.42 57.29 72.38 → 38.10 47.36 85.00 → 51.00 40.00 49.19 40.28 InternVL3 (2B) 83.81 → 40.00 52.27 70.09 → 45.79 34.67 65.05 → 42.72 34.33 86.54 → 25.96 70.00 64.76 → 32.38 50.00 84.00 → 51.00 39.29 39.58 47.67 InternVL3 (8B) 88.57 → 60.95 31.18 75.70 → 58.88 22.22 71.84 → 45.75 36.32 93.27 → 30.77 67.01 67.62 → 41.90 38.04 86.00 → 49.00 43.02 48.23 40.04

InternVL3 (78B) 89.52 → 66.67 25.53 83.18 → 64.49 22.47 78.64 → 62.14 20.98 89.42 → 47.12 47.30 78.12 → 48.57 37.83 86.00 → 52.00 39.53 56.89 32.38 Qwen2-VL (72B) 90.48 → 76.19 15.79 81.31 → 69.16 14.94 84.47 → 51.46 39.10 91.35 → 50.00 45.26 75.24 → 40.95 45.57 87.00 → 61.00 29.89 58.17 31.52 Qwen2.5-VL (3B) 87.62 → 62.86 28.26 71.96 → 49.53 31.17 68.93 → 38.83 43.67 84.62 → 23.08 72.73 60.95 → 31.43 48.43 79.00 → 43.00 45.57 41.51 45.01 Qwen2.5-VL (7B) 87.62 → 53.33 39.14 77.57 → 60.75 21.68 77.67 → 46.60 40.00 92.31 → 43.72 52.64 71.43 → 29.52 58.67 85.00 → 42.00 50.59 45.99 43.84

Qwen2.5-VL (32B) 91.43 → 60.00 34.38 79.44 → 59.81 24.72 73.79 → 43.69 40.78 89.42 → 37.50 58.08 74.29 → 30.48 58.97 87.00 → 54.00 37.93 47.59 42.35 Qwen2.5-VL (72B) 92.38 → 63.81 30.92 78.50 → 67.29 14.28 79.61 → 51.46 35.36 89.42 → 49.04 45.16 80.95 → 48.57 40.00 87.00 → 63.00 27.57 57.21 32.38

Closed-source Models ↑ ↓

Claude-3.7-Sonnet 90.48 → 54.29 40.00 72.90 → 43.93 39.74 59.22 → 38.83 34.43 73.08 → 29.81 59.21 57.14 → 39.05 31.66 86.00 → 48.00 44.19 42.13 42.10 GPT-4o 90.48 → 74.29 17.89 71.96 → 54.21 24.67 58.00 → 43.00 25.86 77.88 → 52.88 32.10 59.05 → 42.86 27.42 85.00 → 54.00 36.47 53.37 27.28 GPT-4.1 86.67 → 73.33 15.39 81.31 → 74.77 8.04 68.00 → 55.00 19.12 86.54 → 67.31 22.22 67.62 → 49.52 26.77 88.00 → 58.00 34.09 62.82 20.80

Gemini-2.5-Flash 86.67 → 61.90 28.58 72.90 → 49.50 32.13 74.76 → 55.34 25.98 72.12 → 36.54 49.33 60.95 → 55.23 9.38 80.00 → 54.00 32.50 52.08 30.11 Gemini-2.5-Pro 90.48 → 67.62 25.26 79.44 → 64.49 18.82 81.00 → 66.00 18.52 86.54 → 54.81 36.67 78.10 → 65.71 15.86 91.00 → 58.00 36.26 62.50 22.00 GPT-5 Chat 83.81 → 70.48 15.91 84.11 → 70.09 16.67 84.47 → 59.22 29.89 85.58 → 70.19 17.98 69.52 → 51.43 26.02 90.00 → 61.00 32.22 63.78 23.02

Human Performance ↑ ↓ Human 100.0 → 98.10 1.99 100.0 → 98.43 1.57 99.03 → 97.09 1.96 100.0 → 98.08 1.92 100.0 → 100.0 0.00 100.0 → 98.00 2.00 98.24 1.63

- Table 3. Performance of Qwen2.5-VL-7B with caption-assisted inference. Baseline (caption model set to None) uses no caption.

Caption Model Res. Rep. Mat. Mir. Occ. Ill. Overall Accm None 53.33 60.75 46.60 43.72 29.52 42.00 45.99 Qwen2.5-VL 54.29 61.68 40.78 40.38 26.67 52.00 45.99 (+0.0)

GPT-4.1 58.10 71.96 49.51 53.85 33.33 56.00 53.85 (+7.86)

- Table 4. Comparison of model performance without and with the thinking process across misleading visual categories.

perceive and encode visual details may improve its robustness. To test this hypothesis, we employ a caption-assisted inference approach (Wan et al., 2025; Liang et al., 2025), where image captions are used as a proxy for enhanced perception. Since captions describe what is present in the image rather than involving multi-step reasoning, they primarily capture perceptual information (Vinyals et al., 2015; Tong et al., 2024). Specifically, we use a strong auxiliary LVLM (the caption model) to generate a detailed image caption, which is then concatenated with the image and question as input to the inference model (see Appendix D.1 for the prompt format).

Model Res. Rep. Mat. Mir. Occ. Ill. Overall Accm w/o thinking process

SAIL-VL-2-8B 57.14 63.55 45.63 39.42 38.10 51.00 49.19 Qwen2-VL-72B 76.19 69.16 51.46 50.00 40.95 61.00 58.17

As shown in Tab. 3, when Qwen2.5-VL-7B answers directly without any additional captions, its Accm is only 45.99%. When the model itself serves as the caption model, performance remains nearly unchanged. However, replacing the caption model with a stronger GPT-4.1 yields substantial gains, increasing overall Accm from 45.99% to 53.85% with improvements across all categories. Notably, this performance surpasses Qwen2.5-VL-32B (47.59%, as shown in Tab. 2) and approaches the results of Qwen2.5-VL-72B (57.21%). These results suggest that the visual perceptual capacity may be a key factor of LVLM robustness under visually misleading conditions.

Gemini-2.5-Flash 61.90 49.50 55.34 36.54 55.23 54.00 52.08 GPT-5-Chat 70.48 70.09 59.22 70.19 51.43 61.00 63.78

w/ thinking process

SAIL-VL2-8B 49.52 55.14 54.37 47.12 34.29 53.00 48.08 (-1.11) QVQ-72B-Preview 61.90 66.36 60.19 54.81 21.90 45.00 51.76 (-6.41)

Gemini-2.5-Flash 64.76 72.90 65.05 63.46 59.05 57.00 63.14 (+11.06) GPT-5-thinking 67.62 73.81 66.99 57.69 55.24 65.00 64.42 (+0.64)

controlled experiments, yielding following two key findings. Finding 1: Enhanced visual perception substantially improves robustness against misleading visual inputs. The vision encoder serves as the “eye” of an LVLM, determining its perceptual upper bound (Tong et al., 2024; Neo et al., 2024; Zhang et al., 2025a). Enhancing the model’s ability to

#### Finding 2: Enhanced visual reasoning improves robust-

|[Figure 82]|
|---|

|(non-think) The little girl is holding six balloons in her hand. The balloons are a mix of pastel<br><br>colors, including pink, white, yellow, and light blue. They are clustered together in her hand, and the image clearly shows her holding them. So the final answer should be B.<br><br>[Figure 83]|
|---|

[Figure 84]

[Figure 85]

|(think) <think>Got it, let's see. The question is asking how many balloons the little girl is holding.<br><br>The options are A. Five, B. Six, C. Four, D. Seven.<br><br>First, I need to look at the image description provided. The description says, "holding a bunch of balloons" and then lists the colors: "white, pink, yellow, light blue, and green". Let me count those. White is one, pink is two, yellow three, light blue four, green five. So that's five balloons.<br><br>Wait, the original COT mentioned the description lists five colors, so that would be five. The answer was A. But let me make sure I didn‘t miss any. The description says “a bunch of<br><br>balloons” with colors: white, pink, yellow, light blue, green. That’s five distinct colors, so<br><br>five balloons. So the correct answer should be A.</think><br><br>[Figure 86]|
|---|

|How many balloons is the<br><br>little girl holding in her hand?<br><br>A. Five B. Six C. Four D. Seven|
|---|

- Figure 5. Comparison between the “non-think” and “think” modes of SAIL-VL. In the non-think mode, the model answers directly based on visual evidence, while in the think mode, the model is guided by historical thoughts and tend to overemphasize fine details.

ness against misleading visual inputs, yet the improvement remains inconsistent and unstable. There are currently two mainstream approaches to strengthen visual reasoning in LVLMs: (1) scaling the language model (Wei et al., 2022a; Berti et al., 2025), and (2) enabling explicit reasoning via long-form chain-of-thought (CoT) (Wei et al., 2022b; Bi et al., 2025; Liu et al., 2026) through reinforcement learning (RL) (Guo et al., 2025; Peng et al., 2025; Wu

confusion and mirror reflection, where models need to carefully examine and analyze visual details.

In contrast, proprietary models (like Gemini-2.5-Flash and GPT-5) show gains with explicit thinking, suggesting better training to effectively leverage extended reasoning without sacrificing perception. However, the thinking-enhanced Gemini-2.5-Flash still underperforms the non-reasoning GPT-5-Chat by 0.64% in Accm. This indicates that visual perception and reasoning are complementary in resisting misleading visual inputs, where visual perception serves as a fundamental prerequisite for extended reasoning in LVLMs.

- et al., 2025). We examine both approaches to understand their impacts on robustness.

Scaling LLMs. As shown in Tab. 2, when scaling from Qwen2.5-VL-3B to Qwen2.5-VL-72B with a fixed vision encoder, overall Accm increases and MVI-Sensitivity decreases. This trend suggests that stronger reasoning capacity can partially compensate for limited visual perceptual ability. However, the improvement is non-monotonic: within the mid-scale range (e.g., from 7B to 32B), performance on misleading images degrades in certain categories, including representation confusion and mirror reflection.

4.4. Case Study: When LVLMs Outperform on Misleading Visual Inputs (RQ3)

While most models perform worse on misleading images, we observe a small but revealing set of counterintuitive instances (about 4% of total cases) where models answer misleading images correctly but fail on normal ones. To investigate, we employ an attention-guided masking approach inspired by recent interpretability studies (Jiang et al., 2025c; Stan et al., 2024; Zhu et al., 2025b; Kang et al., 2025; Tang et al., 2025): we visualize attention maps, identify highattention regions, and mask them to test whether the model’s answer changes (see Appendix D.4 for details).

Enabling explicit reasoning via long-form CoT. Next, we examine how explicit “long-thinking” influences model performance under misleading visual inputs. Unexpectedly, long-thinking variants of open-source models (e.g., SAILVL2 and Qwen2-VL) exhibit lower Accm on MVI-Bench, primarily due to declines in perception-heavy categories such as visual resemblance and representation confusion. This aligns with prior works (Tian et al., 2025; Liu et al., 2025a) showing that multimodal reasoning models’ perceptual abilities weaken as reasoning processes. Analysis of paired instances (Fig. 5) reveals two failure patterns. First, reasoning becomes increasingly guided by historical thoughts rather than image content, a phenomenon termed “visual forgetting” (Tian et al., 2025) (More cases are provided in Appendix E.). Second, models increasingly overattend to fine-grained details (e.g., shape or color), which drives them away from correct answers. Interestingly, this behavior can be beneficial in categories such as material

Findings. Our analysis reveals that these cases arise from spurious correlations between visual artifacts and target labels. Fig. 6 shows a representative example. In the misleading image (Fig. 6 (b)), Qwen2.5-VL-7B misidentifies overlapping books as one and treats a receipt as another book, accidentally producing the correct answer (“2 books”). Masking the receipt (Fig. 6 (c)) immediately flips the prediction, confirming reliance on this false cue. This also explains failure on the normal image (Fig. 6 (a)): the model predicts “2” despite only one book present.

We speculate that this phenomenon stems from the lack of fine-grained supervision in current VQA training paradigms,

copyright regulations, collecting data only from publicly accessible sources. Generated images underwent manual review to mitigate potential biases. We believe this work contributes positively to responsible AI development and foresee no direct negative societal consequences.

(a) Normal Image (b) Misleading Image (c) Misleading Image + Mask

[Figure 87]

[Figure 88]

[Figure 89]

receipt mask

book two book

Question: How many books are there in the image?

[Figure 90]

[Figure 91]

[Figure 92]

### References

Abdin, M., Aneja, J., Behl, H., Bubeck, S., Eldan, R., Gunasekar, S., Harrison, M., Hewett, R. J., Javaheripi, M., Kauffmann, P., et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

[Figure 93]

[Figure 94]

[Figure 95]

Answer: 2 Answer: 2 Answer: 1

- Figure 6. Attention-guided masking for a counterintuitive instance. Qwen2.5-VL-7B spuriously associates a receipt with a book. (a) On the normal image with one book, it answers incorrectly. (b) On the misleading image, it coincidentally answers “2” by counting the receipt as an extra book. (c) Masking the receipt flips the prediction, confirming the spurious correlation.

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

where models are supervised only on final answers without explicit rationale guidance. This weak supervision encourages models to exploit shortcut cues rather than learning true causal associations. Our findings also suggest that future VQA evaluation should consider not only answer correctness but also the underlying reasoning process.

Agarwal, A., Panda, S., Charles, A., Patel, H. L., Kumar, B., Pattnayak, P., Rafi, T. H., Kumar, T., Meghwani, H., Gupta, K., et al. Mvtamperbench: Evaluating robustness of vision-language models. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 18804– 18828, 2025.

### 5. Conclusion

This paper introduces MVI-Bench, the first comprehensive benchmark specifically designed to evaluate how misleading visual inputs undermine the robustness of LVLMs. In addition, we propose MVI-Sensitivity, a metric that characterizes LVLM robustness at a fine-grained level. Our evaluation reveals that existing LVLMs exhibit pronounced vulnerabilities to misleading visual inputs, and in-depth analysis on MVI-Bench provides actionable insights for developing more reliable and robust LVLMs.

### Acknowledgements

We thank Yuxin Wang and Yuhua Xie for their contributions to image collection and annotation of MVI-Bench. We also thank Yuqin Wang and Yuxuan Xie for participating in the human performance evaluation.

### Impact Statement

This paper presents MVI-Bench, a benchmark for evaluating the robustness of Large Vision-Language Models (LVLMs) against misleading visual inputs. By examining model behavior from visual perception and reasoning perspectives, our work provides actionable insights for developing more reliable LVLMs. The analysis of counterintuitive cases further offers guidance for future training and evaluation methodologies. In developing MVI-Bench, ethical considerations were carefully addressed. We strictly complied with

Anthropic. Claude 3.7 sonnet system card, 2025. Accessed: 2025-02-04.

Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., and Parikh, D. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pp. 2425–2433, 2015.

Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Bai, Z., Wang, P., Xiao, T., He, T., Han, Z., Zhang, Z., and Shou, M. Z. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

Berti, L., Giorgi, F., and Kasneci, G. Emergent abilities in large language models: A survey. arXiv preprint arXiv:2503.05788, 2025.

Bi, Z., Chen, K., Wang, T., Hao, J., Peng, B., and Song, X. Cot-x: An adaptive framework for cross-model chainof-thought transfer and optimization. arXiv preprint arXiv:2511.05747, 2025.

Chen, H., Peng, J., Tang, K., Geng, X., and Yang, X. Enhancing multimodal in-context learning for image classification through coreset optimization. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 5130–5139, 2025.

Chen, X., Fang, H., Lin, T.-Y., Vedantam, R., Gupta, S., Doll´ar, P., and Zitnick, C. L. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024.

Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P. N., and Hoi, S. Instructblip: Towards generalpurpose vision-language models with instruction tuning. Advances in neural information processing systems, 36: 49250–49267, 2023.

Dang, Y., Gao, M., Yan, Y., Zou, X., Gu, Y., Li, J., Wang, J., Jiang, P., Liu, A., Liu, J., et al. Exploring response uncertainty in mllms: An empirical evaluation under misleading scenarios. arXiv preprint arXiv:2411.02708, 2024.

Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J. S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 91–104, 2025.

Deng, A., Cao, T., Chen, Z., and Hooi, B. Words or vision: Do vision-language models have blind faith in text? In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 3867–3876, 2025.

Dong, H., Kang, Z., Yin, W., Liang, X., Feng, C., and Ran, J. Scalable vision language model training via high quality data curation. arXiv preprint arXiv:2501.05952, 2025.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Fu, J., Wang, Y., Han, S., Fan, J., and Yang, X. Graphic: A graph-based in-context example retrieval model for multistep reasoning. In Proceedings of the AAAI Conference

on Artificial Intelligence, volume 40, pp. 30753–30761, 2026.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Han, J., Tong, S., Fan, D., Ren, Y., Sinha, K., Torr, P., and Kokkinos, F. Learning to see before seeing: Demystifying llm visual priors from language pre-training. arXiv preprint arXiv:2509.26625, 2025.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Jiang, C., Jia, H., Dong, M., Ye, W., Xu, H., Yan, M., Zhang, J., and Zhang, S. Hal-eval: A universal and fine-grained hallucination evaluation framework for large vision language models. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 525–534, 2024.

Jiang, C., Wang, Z., Dong, M., and Gui, J. Survey of adversarial robustness in multimodal large language models. arXiv preprint arXiv:2503.13962, 2025a.

- Jiang, Y., Fu, J., Hao, C., Hu, X., Peng, Y., Geng, X., and Yang, X. Mimic in-context learning for multimodal tasks. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 29825–29835, 2025b.
- Jiang, Z., Chen, J., Zhu, B., Luo, T., Shen, Y., and Yang, X. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 25004– 25014, 2025c.

Kang, S., Kim, J., Kim, J., and Hwang, S. J. See what you are told: Visual attention sink in large multimodal models. arXiv preprint arXiv:2503.03321, 2025.

Kuang, J., Shen, Y., Xie, J., Luo, H., Xu, Z., Li, R., Li, Y., Cheng, X., Lin, X., and Han, Y. Natural language understanding and inference with mllm in visual question answering: A survey. ACM Computing Surveys, 57(8): 1–36, 2025.

Landis, J. R. and Koch, G. G. The measurement of observer agreement for categorical data. biometrics, pp. 159–174, 1977.

Lauren¸con, H., Tronchon, L., Cord, M., and Sanh, V. What matters when building vision-language models? Advances in Neural Information Processing Systems, 37: 87874–87907, 2024.

Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., and Shan, Y. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13299–13308, 2024a.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024b.

Li, L., Peng, J., Chen, H., Gao, C., and Yang, X. How to configure good in-context sequence for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26710– 26720, 2024c.

Liang, H., Wu, R., Zeng, B., Niu, J., Zhang, W., and Dong,

- B. Multimodal reasoning for science: Technical report and 1st place solution to the icml 2025 seephys challenge. arXiv preprint arXiv:2509.06079, 2025.

Liu, C., Xu, Z., Wei, Q., Wu, J., Zou, J., Wang, X. E., Zhou, Y., and Liu, S. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. arXiv preprint arXiv:2505.21523, 2025a.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Liu, W., Min, D., and Cheng, L. Latent thoughts tuning: Bridging context and reasoning with fused information in latent tokens. arXiv preprint arXiv:2602.10229, 2026.

- Liu, Y., Liang, Z., Wang, Y., Wu, X., Tang, F., He, M., Li, J.,
- Liu, Z., Yang, H., Lim, S., et al. Unveiling the ignorance of mllms: Seeing clearly, answering incorrectly. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 9087–9097, 2025b.

Liu, Z., Gao, K., Liang, S., Xiao, B., Qiao, L., Ma, L., and Jiang, T. Beyond the visible: Benchmarking occlusion perception in multimodal large language models. arXiv preprint arXiv:2508.04059, 2025c.

Min, D., Zhang, K., Wu, T., and Cheng, L. Quco-rag: Quantifying uncertainty from the pre-training corpus for dynamic retrieval-augmented generation. arXiv preprint arXiv:2512.19134, 2025.

Neo, C., Ong, L., Torr, P., Geva, M., Krueger, D., and Barez, F. Towards interpreting visual information processing in vision-language models. arXiv preprint arXiv:2410.07149, 2024.

OpenAI. Gpt-5 system card, 2025. Accessed: 2025-08-13.

Peng, Y., Hao, C., Hu, X., Peng, J., Geng, X., and Yang, X. Live: Learnable in-context vector for visual question answering. Advances in Neural Information Processing Systems, 37:9773–9800, 2024.

Peng, Y., Zhang, G., Zhang, M., You, Z., Liu, J., Zhu, Q., Yang, K., Xu, X., Geng, X., and Yang, X. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.

Qian, Y., Zhang, H., Yang, Y., and Gan, Z. How easy is it to fool your multimodal llms? an empirical analysis on deceptive prompts. arXiv preprint arXiv:2402.13220, 2 (7), 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Seedream, T., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

Shahgir, H. S., Sayeed, K. S., Bhattacharjee, A., Ahmad, W. U., Dong, Y., and Shahriyar, R. Illusionvqa: A challenging optical illusion dataset for vision language models. arXiv preprint arXiv:2403.15952, 2024.

Smith, A. D. The problem of perception. Motilal Banarsidass Publishe, 2005.

Stan, G. B. M., Aflalo, E., Rohekar, R. Y., Bhiwandiwalla, A., Tseng, S.-Y., Olson, M. L., Gurwicz, Y., Wu, C., Duan, N., and Lal, V. Lvlm-interpret: an interpretability tool for large vision-language models. arXiv preprint arXiv:2404.03118, 2024.

Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L., Wang, Y.-X., Yang, Y., et al. Aligning large multimodal models with factually augmented rlhf. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 13088–13110, 2024.

Tang, F., Liu, C., Xu, Z., Hu, M., Huang, Z., Xue, H., Chen, Z., Peng, Z., Yang, Z., Zhou, S., et al. Seeing far and clearly: Mitigating hallucinations in mllms with attention causal decoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26147–26159, 2025.

Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Team, K., Du, A., Yin, B., Xing, B., Qu, B., Wang, B., Chen, C., Zhang, C., Du, C., Wei, C., et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.

Tian, X., Zou, S., Yang, Z., He, M., Waschkowski, F., Wesemann, L., Tu, P., and Zhang, J. More thought, less accuracy? on the dual nature of reasoning in vision-language models. arXiv preprint arXiv:2509.25848, 2025.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024.

Vinyals, O., Toshev, A., Bengio, S., and Erhan, D. Show and tell: A neural image caption generator. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3156–3164, 2015.

Wan, Q., Wang, Z., Zhou, J., Wang, W., Geng, Z., Liu, J., Cao, R., Cheng, M., and Cheng, L. Som-1k: A thousand-problem benchmark dataset for strength of materials. arXiv preprint arXiv:2509.21079, 2025.

Wang, F., Fu, X., Huang, J. Y., Li, Z., Liu, Q., Liu, X., Ma, M. D., Xu, N., Zhou, W., Zhang, K., et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024a.

Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., Wang, J., Xu, H., Yan, M., Zhang, J., et al. Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023.

Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b.

Wang, Y., Chen, W., Han, X., Lin, X., Zhao, H., Liu, Y., Zhai, B., Yuan, J., You, Q., and Yang, H. Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning. arXiv preprint arXiv:2401.06805, 2024c.

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022a.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022b.

Wu, Y., Zhou, Y., Ziheng, Z., Peng, Y., Ye, X., Hu, X., Zhu, W., Qi, L., Yang, M.-H., and Yang, X. On the generalization of sft: A reinforcement learning perspective with reward rectification. arXiv preprint arXiv:2508.05629, 2025.

Yang, X., Wu, Y., Yang, M., Chen, H., and Geng, X. Exploring diverse in-context configurations for image captioning. Advances in Neural Information Processing Systems, 36:40924–40943, 2023.

Ye, M., Rong, X., Huang, W., Du, B., Yu, N., and Tao, D. A survey of safety on large vision-language models: Attacks, defenses and evaluations. arXiv preprint arXiv:2502.14881, 2025.

Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., and Chen, E. A survey on multimodal large language models. National Science Review, 11(12):nwae403, 2024.

Yin, W., Ye, Y., Shu, F., Liao, Y., Kang, Z., Dong, H., Yu, H., Yang, D., Wang, J., Wang, H., et al. Sail-vl2 technical report. arXiv preprint arXiv:2509.14033, 2025.

Yu, T., Zhang, H., Li, Q., Xu, Q., Yao, Y., Chen, D., Lu, X., Cui, G., Dang, Y., He, T., et al. Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19985–19995, 2025.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhang, J., Khayatkhoei, M., Chhikara, P., and Ilievski, F. Mllms know where to look: Training-free perception of small visual details with multimodal llms. arXiv preprint arXiv:2502.17422, 2025a.

Zhang, Y., Zhang, Z., Wei, X., Liu, X., Zhai, G., and Min, X. Illusionbench: A large-scale and comprehensive benchmark for visual illusion understanding in vision-language models. arXiv preprint arXiv:2501.00848, 2025b.

Zhao, Y., Pang, T., Du, C., Yang, X., Li, C., Cheung, N.M. M., and Lin, M. On evaluating adversarial robustness of large vision-language models. Advances in Neural Information Processing Systems, 36:54111–54138, 2023.

Zheng, C., Zhou, H., Meng, F., Zhou, J., and Huang, M. Large language models are not robust multiple choice selectors. arXiv preprint arXiv:2309.03882, 2023.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding

with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025a.

Zhu, J., Wu, Y., Zhu, W., Cao, J., Zheng, Y., Chen, J., Yang, X., Schiele, B., Fischer, J., and Hu, X. Layercake: Tokenaware contrastive decoding within large language model layers. arXiv preprint arXiv:2507.04404, 2025b.

### A. Limitations and Future Work

Although spurious-correlation cases constitute only a very small fraction of our benchmark, their implications may be more substantial in real-world settings. Modern LVLMs are trained on extremely large-scale datasets, often containing millions to billions of image–text pairs and even small proportions of these examples can accumulate into systematic biases. However, our analysis of spurious correlations relies fundamentally on the paired design of MVI-Bench, which enables counterfactual comparisons between normal and misleading images. As such, the methodology does not directly generalize to other datasets without further adaptations. Furthermore, our findings underscore broader challenges in current LVLM training and evaluation pipelines: shortcut learning remains difficult to mitigate under weak supervision that provides only answer labels without rationales, and existing VQA evaluations emphasize answer correctness without assessing whether predictions are causally grounded in visual evidence. Future work should explore training objectives and evaluation protocols that discourage shortcut exploitation and promote faithful, causally aligned visual reasoning.

### B. Ethics Statement

During the collection of major instances, we strictly complied with the copyright and licensing regulations of each social media platform, ensuring that data was collected only from publicly accessible posts and that no images were downloaded from sources explicitly prohibiting data reuse or redistribution. A portion of the dataset was additionally generated using Seedream, a powerful image generation model. While this model produces high-quality outputs, the generated content inevitably reflects the biases and limitations inherent in its training data. We recognize the ethical concerns associated with such models, including the potential to reinforce stereotypes or generate inappropriate content.

To minimize these risks, we applied careful dataset curation and conducted manual reviews throughout the construction process. We encourage future research to explore stronger methods for identifying and mitigating such biases, ensuring that both collected and generated content align with ethical standards and societal norms.

### C. More Details about MVI-Bench

- C.1. Data Curation Pipeline The data curation pipeline is illustrated in Fig. 7.

###### (c) Data Filtering and Refinement

|(a) Image Collection| |
|---|---|
|Natural Images<br><br>[Figure 96]<br><br>[Figure 97]<br><br>Synthetic Images<br><br>[Figure 98]<br><br>Web<br><br>MLLM<br><br>[Figure 99]<br><br>[Figure 100]<br><br>Edited Images<br><br>Edited Images<br><br>[Figure 101]<br><br>[Figure 102]| |
| | |

|(b) VQA Annotation| |
|---|---|
|[Figure 103]<br><br>[Figure 104]<br><br>Normal Images<br><br>Misleading Images<br><br>Normal Images<br><br>Misleading Images<br><br>[Figure 105]<br><br>[Figure 106]<br><br>Annotator Annotator<br><br>|question|
|---|
<br><br>|answer|
|---|
<br><br>|answer|
|---|
<br><br>|question|
|---|
<br><br>|answer|
|---|
<br><br>|answer|
|---|
| |
| | |

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Normal Images

Misleading Images

Normal Images

Misleading Images

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

filter retain

- Figure 7. Benchmark Curation Pipeline. The pipeline starts with image collection, followed by VQA annotation, data filtering, and ultimately results in MVI-Bench. To ensure data quality, human verification is performed at each key stage to eliminate low-quality data, annotations, and ambiguous evaluation questions.

- C.2. Annotation Quality Control This section provides additional details on the annotation process and quality control measures outlined in Section 3.

Annotator qualifications. As mentioned in Section 3, all annotators are trained human experts with at least a bachelor’s degree. Before annotation, each annotator received detailed guidelines with category definitions, examples of valid/invalid instances, and common pitfalls to avoid.

Verification process. The annotation process involved multiple verification stages:

• Initial annotation: Three trained experts independently curated image pairs and crafted MCQs following the taxonomy.

|[Figure 119]|
|---|

|(non-think) The image depicts a unique and artistic piece of furniture that resembles a chair. It has a distinctive design with a green, leaf-like structure that appears to be made of a soft, plush material. The chair's form is reminiscent of a lotus leaf, but it is clearly a piece of furniture rather than a natural object. So the final answer should be D.<br><br>[Figure 120]|
|---|

[Figure 121]

[Figure 122]

|(think) <think>Got it, let's try to figure out this question. The user is asking what's in the picture, with options A to D. The original answer was B, which is Lotus leaf. Let me start by looking at the image description provided.<br><br>The image is described as a chair with a green, leaf-like design. The key here is the shape and context. The original COT mentions the chair has a leaf-like design, which is a lotus leaf. Let's check the options: A is Lotus flower, B is Lotus leaf. A flower is a bloom, while a leaf is the flat, broad part. The image is a chair, which is a piece of furniture, not a flower. So the main object is the chair, but the description says it's leaf-like. The question is asking what's in the picture, and the options include "Leaves" (C) as a choice. Wait, the original answer was B, which is Lotus leaf. But the image is a chair, not a leaf. Maybe the original COT was focusing on the leaf shape of the chair, which is a lotus leaf. The original answer was B, so perhaps the correct answer is B. Let me confirm: the image is a chair with a leaf-like design. The options include "Leaves" (C), but the question's options are A to D. The original answer was B, which is Lotus leaf. So the final answer should be B.</think><br><br>[Figure 123]|
|---|

|What is in the picture?<br><br>A. Lotus flower<br>B. Lotus leaf<br>C. Leaves<br>D. Chair<br>|
|---|

|[Figure 124]|
|---|

|(non-think) It's a pink-colored beverage in a clear cup with a yellow McCafé logo on it. The drink is placed on a white surface, and there's a smartphone nearby displaying the same image of the drink. There are no other drinks visible in the image. So the final answer should be C.<br><br>[Figure 125]|
|---|

[Figure 126]

[Figure 127]

|[Figure 128]<br><br>(think) <think> Got it, let's try to figure out how many drinks are in the image. First, the image description mentions a "McCafé" cup with a pink drink and whipped cream. Then, there's a "red cup with whipped cream" on a book. Wait, the original COT mentions "a red cup with whipped cream" and "a pink one". But wait, the image description in the input might have more details. Let me check the user's input again. The user's image description (from the SOI to EOI) says: "a red cup with whipped cream" and "a pink one".<br><br>Wait, the original COT says: "the image description mentions a red cup with whipped cream and a pink one". Oh, the original COT says "the image description mentions a red cup with whipped cream and a pink one". So that would be two drinks: red and pink. Then the final answer should be D(Two).</think>|
|---|

|How many drinks are in the image? A. Four B. Three C. One D. Two|
|---|

- Figure 8. Comparison between the “non-think” and “think” modes of SAIL-VL. In the non-think mode, the model answers directly based on visual evidence, while in the think mode, the model is guided by historical thoughts and tend to overemphasize fine details.

- • Cross-review: Each annotated VQA instance was then independently reviewed by additional experts to verify: (1) the image pair correctly reflects the intended misleading category, (2) the question is unambiguous, and (3) the ground-truth answer is correct.
- • Final check: A final quality review was conducted by the first author to filter out any remaining ambiguous instances.

Inter-annotator agreement. To quantify annotation reliability, we computed Fleiss’ Kappa on a randomly sampled subset of 200 instances, yielding κ = 0.93, indicating near-perfect agreement (Landis & Koch, 1977). This confirms that the annotation criteria were well-defined and consistently applied.

Disagreement resolution. When reviewers disagreed on an instance (approximately 5% of cases), the instance was discussed among annotators. If consensus could not be reached, the instance was discarded, as noted in Section 3. This conservative approach prioritizes annotation reliability over dataset size.

- C.3. More Examples Fig. 9 presents additional examples from each category of MVI-Bench.

### D. Experiment Details

#### D.1. Prompt Format for Caption-Assisted Inference

We describe the prompt format used for caption-assisted inference. After prompting a strong auxiliary LVLM with the instruction “Please describe the image.”, we obtain a detailed caption that compensates for the limitations of the inference model’s vision encoder. We then concatenate this caption with the corresponding image and question as input to the inference model, as illustrated below.

<image> Here are some additional information which are text descriptions based on the image to assist you for answering the later question:{Caption} {Question} Answer with the letter from the given choices directly.

- D.2. Model Details Tab. 5 presents the release time and model sources of LVLMs used in MVI-Bench.
- D.3. Human Performance Evaluation

To establish a human performance baseline, we conducted a controlled study with four participants: two high school students and two university students. This selection covers different educational backgrounds to verify that MVI-Bench requires no specialized knowledge and remains solvable for the general population.

Procedure. Each participant independently completed all 1,248 VQA instances (624 pairs across six categories) in a randomized order. To simulate rapid visual perception in real-world scenarios, we imposed a 5-second time limit per question. If a participant failed to respond within 5 seconds, the response was marked as incorrect. This constraint also aligns with the fast inference setting of LVLMs in practical applications. Prior to the formal evaluation, each participant completed a practice session of 10 instances (not included in MVI-Bench) to familiarize themselves with the interface and time constraint.

Results. The reported human performance in Tab. 2 represents the average accuracy across all four participants. Despite the time constraint, humans achieve 98.24% Accm with an MVI-Sensitivity of only 1.63%, demonstrating strong resilience to misleading visual cues. This result confirms that the visual misleading scenarios in MVI-Bench, while challenging for LVLMs, remain easily resolvable for humans through rapid perceptual processing and contextual reasoning.

- D.4. Implementation of Attention Visualization

To visualize where the model attends when generating an answer, we implement the relative answer-to-image attention. Specifically, for a given image x and question q, we first feed the multimodal input sequence into the LVLM and extract the cross-attention weights from the first generated answer token to all image tokens between the <|vision start|> and <|vision end|> markers in each layer. The attention matrices are averaged across all attention heads, yielding the answer-to-image-token attention Ast(x,q).

We then normalize this attention by its counterpart obtained from a generic instruction q′ =“Write a general description of the image.”, resulting in the relative attention:

Ast(x,q) Ast(x,q′)

Arel(x,q) =

.

This normalization removes the model’s default visual bias and highlights the image regions whose attention increases specifically in response to the question.

- E. Additional Case Studies on Long-form CoT More cases illustrating the comparison between the Non-think and Think modes are presented in Fig. 8.

###### Visual resemblance

What is in the picture?

What is in the picture?

What is in the picture?

What is in the picture?

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

Donuts Sandwich Cake Cat

Donuts

Cake

Sandwich

Cat

Wood Carving

Cup

Harmonica

Sponges

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

###### Representation Confusion

How many edible desserts are in the image?

[Figure 145]

[Figure 146]

[Figure 147]

- 1

[Figure 148]

[Figure 149]

How many edible snacks are there in the picture?

5

5

How many drinks are in the image?

1

1

How many drinks are in the picture?

2

2

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

- 2 8 2 4

[Figure 156]

1

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

###### Material Confusion

How many plastic chairs are in the image?

- 1

How many wooden cutting boards are in the image?

3

2

How many towels are on the metal rack?

1

1

How many fragile bowls are in the image?

2

1

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

- 2 3 2 2

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

2

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

###### Mirror Reflection

How many notebooks are in the image?

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

- 1

How many mats are on the floor?

1

1

How many apples are in the image?

1

1

How many balls are in the image?

1

1

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

- 2 2 4 8

1

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

###### Occlusion Confusion

How many dumplings are in the image?

How many cups are in the image?

How many biscuits are in the image?

How many cushions are in the image?

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

1 5 3 6

[Figure 202]

[Figure 203]

[Figure 204]

1

6

4

3

[Figure 205]

4

8

[Figure 206]

[Figure 207]

[Figure 208]

6

2

###### Visual Illusion

Which of the following options is in the picture?

Which sentence best describes the scene in the picture?

###### Which sentence best describes the scene in the picture?

###### Which sentence best describes the scene in the picture?

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

There is smoke rising from the top of the building.

[Figure 218]

[Figure 219]

[Figure 220]

A crane is lifting

A bird stands on an a moon. External staircase

iron chain.

Balcony

There is a building under the blue sky.

A crane is lifting a white box.

A bird stands by an iron chain.

There is a building under the blue sky and white clouds.

A bird happens to stand on the shadow of an iron chain.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

A crane is not lifting anything.

Balcony

###### Figure 9. More Examples from six misleading categories defined in MVI-Bench.

Table 5. The Release Time and Model Source of LVLMs Evaluated in MVI-Bench.

Model Release Time

Source URL

LLaVA-OneVision-7B (Li et al.,

- 2024b)

2024-08 ByteDance https://github.com/LLaVA-VL/LLaVA-NeXT

Molmo-7B-D-0924 (Deitke et al., 2025)

2025-04 allenai (Ai2) https:

//huggingface.co/allenai/Molmo-7B-D-0924 SAIL-VL2-2B (Yin et al., 2025) 2025-09 ByteDance https://huggingface.co/

BytedanceDouyinContent/SAIL-VL2-2B SAIL-VL2-8B (Yin et al., 2025) 2025-09 ByteDance https://huggingface.co/

BytedanceDouyinContent/SAIL-VL2-8B

InternVL3-2B (Zhu et al., 2025a)

2025-08 OpenGVLab https: //huggingface.co/OpenGVLab/InternVL3-2B

InternVL3-8B (Zhu et al., 2025a)

2025-08 OpenGVLab https: //huggingface.co/OpenGVLab/InternVL3-8B

InternVL3-78B (Zhu et al., 2025a)

2025-08 OpenGVLab https:

//huggingface.co/OpenGVLab/InternVL3-78B Qwen2-VL-72B-Instruct (Wang

- et al., 2024b)

2024-09 Alibaba https://huggingface.co/Qwen/

Qwen2-VL-72B-Instruct Qwen2.5-VL-3B-Instruct (Bai

- et al., 2025)

2025-01 Alibaba https://huggingface.co/Qwen/Qwen2. 5-VL-3B-Instruct

Qwen2.5-VL-7B-Instruct (Bai et al., 2025)

2025-01 Alibaba https://huggingface.co/Qwen/Qwen2. 5-VL-7B-Instruct

Qwen2.5-VL-32B-Instruct (Bai et al., 2025)

2025-01 Alibaba https://huggingface.co/Qwen/Qwen2. 5-VL-32B-Instruct

Qwen2.5-VL-72B-Instruct (Bai et al., 2025)

2025-01 Alibaba https://huggingface.co/Qwen/Qwen2.

5-VL-72B-Instruct Claude-3.7-Sonnet (Anthropic,

- 2025)

2025-01 Anthropic https://www.anthropic.com/news/ claude-3-7-sonnet

- GPT-4o (Hurst et al., 2024) 2024-05 OpenAI https://platform.openai.com/ GPT-4.1 (Achiam et al., 2023) 2024-05 OpenAI https://platform.openai.com/

Gemini-2.5-Flash (Team et al., 2023)

2025-06 Google https://gemini.google.com/app

Gemini-2.5-Pro (Team et al., 2023)

2025-06 Google https://gemini.google.com/app

- GPT-5 Chat (OpenAI, 2025) 2025-08 OpenAI https://platform.openai.com/

