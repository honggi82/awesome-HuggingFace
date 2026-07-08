## ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement

Zhihang Liu1*, Xiaoyi Bao2*, Pandeng Li1,7†, Junjie Zhou3, Zhaohe Liao4, Yefei He5, Kaixun Jiang6, Chen-Wei Xie7, Yun Zheng7, Hongtao Xie1 1 USTC 2 CASIA 3 NJU 4 SJTU 5 ZJU 6 FDU 7 Tongyi Lab

Project Page: https://lntzm.github.io/showtable-page/

# arXiv:2512.13303v2[cs.CV]26Mar2026

#### Abstract

[Figure 1]

|Creative Table Visualization Task|
|---|

Ablation Study of Refinement Module:

While existing generation and unified models excel at general image generation, they struggle with tasks requiring deep reasoning, planning, and precise data-to-visual mapping abilities beyond general scenarios. To push beyond the existing limitations, we introduce a new and challenging task: creative table visualization, requiring the model to generate an infographic that faithfully and aesthetically visualizes the data from a given table. To address this challenge, we propose ShowTable, a pipeline that synergizes MLLMs with diffusion models via a progressive selfcorrecting process. The MLLM acts as the central orchestrator for reasoning the visual plan and judging visual errors to provide refined instructions, the diffusion execute the commands from MLLM, achieving high-fidelity results. To support this task and our pipeline, we introduce three automated data construction pipelines for training different modules. Furthermore, we introduce TableVisBench, a new benchmark with 800 challenging instances across 5 evaluation dimensions, to assess performance on this task. Experiments demonstrate that our pipeline, instantiated with different models, significantly outperforms baselines, highlighting its effective multi-modal reasoning, generation, and error correction capabilities.

[Figure 2]

[Figure 3]

S2

S1 & S2

S3 & S4 S3 & S4

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

ShowTable Pipeline

S1: rewriting → S2: generation → S3: reflection → S4: refinement

Figure 1. The illustration of our proposed creative table visualization task and ShowTable pipeline. Given a table about a specific topic, our task requires the model to produce a visualization infographic that is aesthetic and faithful to the data points. The ShowTable pipeline employs four steps: rewriting, generation, reflection, and refinement, thus achieving high-fidelity visualization. We use Wan2.5-Preview [51] here for generation and refinement.

creasingly shifting towards more complex scenarios such as graphic design (e.g., poster generation [19, 24, 35] and text rendering [7, 38, 56]). These tasks present a higher challenge as they demand not photorealism, but precise textual alignment and adherence to aesthetic principles of graphics.

#### 1. Introduction

However, within this structured domain, data-centric visualizations (e.g., charts and graphs derived from tables) present an even more formidable challenge. These tasks require not only understanding and reasoning of data but also a rigorous, high-fidelity mapping of quantitative data, where errors in visual proportions (e.g., bar heights, pie chart angles) or data labels render the output useless. To push beyond this limitation from unified reasoning and synthesis, we introduce a new and challenging task for unified models: creative table visualization, requiring a model to generate an infographic that is aesthetic and faithful to the data points within a given table. This task is defined by its

Image generation models have shown significant growth in quality in recent years [5, 15, 28, 43, 46, 64]. Recent advancements, particularly in unified models [8, 10, 14, 34, 50, 51, 58, 65, 66, 70], have leveraged MLLMs for fine-grained understanding of complex multi-modal instructions [39], leading to more semantically coherent synthesis in general-purpose image generation and editing. Despite this rapid progress in the general domain, research is in-

*Equal contribution, interns at Tongyi Lab. †Corresponding author.

dual requirements: 1) sophisticated reasoning for graphic design and aesthetic layout, and 2) strict, high-fidelity datato-visual mapping from a source table. The ability to perform such synthesis could significantly streamline workflows in poster design, automatic slide generation, scientific communication, and other data-driven reporting tasks.

As shown in Figure 1, current models still struggle with directly completing this task, as various errors about layout and data mapping may occur. Therefore, we propose ShowTable, a novel pipeline that synergizes MLLMs and diffusion models in a progressive, self-correcting loop, better addressing the task. As illustrated in Figure 1, this process is inherently suited for achieving faithful data-to-image mapping, since the self-correcting loop iteratively refine initial inaccuracies. The framework relies on two core components: an MLLM acting as a central orchestrator and a diffusion model as the executor. The MLLM performs two key roles: 1) Rewriting, where it first reasons over the tabular data to plan an aesthetic visual sketch and rewrites the user prompt accordingly; and 2) Reflection, where it assesses the generated output and provides precise editing instructions. Correspondingly, the diffusion model executes two stages: 1) Generation, creating an initial figure based on the MLLM’s sketch; and 2) Refinement, editing the figure based on the MLLM’s reflective feedback.

To optimize the effectiveness of the pipeline, we conduct a pioneering exploration into training specific rewriting and refining capabilities. First, as Figure 3 illustrates, the rewriting module undertakes the crucial responsibility of reasoning and planning, its quality thus largely dictates the final outcome. Therefore, we produce 30K supervisedfinetuning (SFT) data by captioning collected table visualizations and train this module. Second, as shown in Figure 4, incapable refining models may degrade performance. This observation confirms that the refinement module could be a bottleneck in the pipeline. Therefore, we first train a reward model (RM) using our constructed 30K comprehensive preference pairs. We then leverage this trained RM to optimize the diffusion model’s refinement capability using reinforcement learning (RL) on a filtered set of 5K samples.

Furthermore, to evaluate the performance of models on this task, we introduce a comprehensive benchmark called TableVisBench. The benchmark features 800 challenging table instances, which are meticulously selected, filtered, labeled, and manually verified to ensure high quality. We design five key evaluation dimensions: data accuracy, text rendering, relative relationship, additional information accuracy, and aesthetic quality, comprehensively assessing the performance of different baselines.

Extensive experiments have demonstrated that our ShowTable pipeline significantly boosts the performance of all base models on TableVisBench. In conclusion, our contribution can be summarized as: 1) New task: We propose

the creative table visualization task, which requires detailed reasoning and precise alignment ability, to challenge existing unified models. 2) Data: We construct three automatic data construction pipelines to produce data for training, and present a TabelVisBench benchmark, comprehensively evaluating the models’ capability for creative table visualization task. 3) Method: We design a progressive self-correcting pipeline ShowTable that cooperate an MLLM as central orchestrator with a diffusion model as executor. Experiments demonstrate the effectiveness of our framework.

#### 2. Related Work

Image generation for graphic design. Recent advances in image generation have significantly enhanced the quality and semantic controllability of synthesized content in general-purpose scenarios [5, 8, 10, 14, 15, 28, 34, 43, 46, 51, 58, 65, 66, 70]. This progress has spurred growing interest in more complex generation tasks such as graphic design. AnyText [55, 56], Glyph-ByT5 [38], and TextDiffuser [7] have made significant efforts in text rendering, focusing on accurate textual element incorporation. AutoPoster [35], PosterMaker [19], and DreamPoster [24] focus on poster generation, which addresses aesthetic layout planning. The recently proposed Qwen-Image [65] also emphasizes capabilities in complex text rendering and infographic design. Compared to these scenarios, our proposed creative table visualization task presents a more formidable challenge. Our task demands not only reasoning about graphic design for aesthetic layout but also a precise data-to-visual mapping for faithful representation of tabular content.

Reasoning and reflection paradigm of MLLMs. The reasoning and reflection abilities of MLLMs are gaining increasing attention for both understanding and generation. Some studies leverage the reasoning capability of MLLMs to achieve better image understanding, often termed thinking with images [3, 11, 25, 80, 81]. For image generation, some works [14, 17, 28, 69] introduce a text-based reasoning step, enhancing image generation performance. Recently, emerging research has begun to leverage the MLLM reflection process to refine the image generation itself, aiming to enhance instruction-following in complex, generalpurpose scenes [26, 27, 60]. However, this category of research remains focused on general domains. The high information density and specific structural constraints of our task pose a greater challenge for MLLMs with combining reasoning, planning, and reflection abilities.

#### 3. Method 3.1. ShowTable Pipeline

Overview. To address the challenge of creative table visualization and push model reasoning in the infographic domains, we propose a base pipeline, ShowTable. To

Average Time Spent On Social Networks Per Day

Refinement Refinement

|[Figure 8]| |
|---|---|
| | |

|[Figure 9]| |
|---|---|
| | |

|Social Network|Avg Time (Mins)|
|---|---|
|Facebook|40|
|Tumblr|34|
|Instagram|21|
|Pinterest|21|
|Twitter|17|
|Snapchat|17|
|Reddit|13|
|LinkedIn|10|
| | |

Diffusion Executor​

Diffusion Executor​

Reflection: The image contains critical errors ... First, fix the bar heights, increase… Second, replace the incorrect logo under the 'Tumblr' bar…

Reflection: The image… First, change the label ‘Linkecin’ to ‘LinkedIn’

… Second, adjust the height of the ‘Snapchat’ bar…

###### Central Orchestrator (MLLM)​

| | |
|---|---|
|[Figure 10]| |

| | |
|---|---|
|[Figure 11]| |

Rewriting: An infographic…, titled 'Average Time Spent On Social Networks Per Day'. The chart features eight colored bars... Each bar has the social network's icon at its base…

Reflection: The image… Correct the bar heights to… Make the Pinterest bar the exact same height as the Instagram bar…

Reflection : The infographic accurately represents the provided data table…

Diffusion Executor​

Diffusion Executor​

[Figure 12]

✅ Done.

Generation Refinement

- Figure 2. The proposed ShowTable pipeline, which synergizes an MLLM as the central orchestrator with a diffusion model as the executor. Given a table, the MLLM first rewrites a detailed prompt for the diffusion model’s initial generation. The MLLM then iteratively reflects on the output to identify errors (marked in red) and provides precise instructions for refinement (corrected results shown in green).

confront the limitations of existing models in preserving visual relational reasoning and information consistency, ShowTable introduces a structured Rewriting → Generation → Reflection → Refinement workflow. This iterative selfcorrection loop is specifically designed to address the challenges of creative table visualization, where standard generation models often fail, producing logical inconsistencies or text rendering errors. As shown in Figure 2, ShowTable progressively improves the quality and detail fidelity by repeated reflection and refinement.

heights), axis label misalignments, or suboptimal text rendering. This imperfect initial generation serves as the foundation for the subsequent reflection and refinement stages.

Reflection. Given the strict fidelity required for table visualization, we employ a reflection module. This module uses an MLLM to perform a critical audit of the generated image. By cross-referencing the original table with the generated image, the MLLM identifies inconsistencies and inaccuracies. It then formulates a set of precise, actionable editing instructions to correct these errors.

Rewriting. There exists a significant distinction between raw tables and typical image generation prompts. Table inputs (e.g., in markdown format) possess high information density, with each data point encapsulating complex relational semantics. Visualizing such dense and non-redundant data necessitates deliberate reasoning about data presentation and layout. When a markdown-formatted table is directly used as a prompt, the generation model tends to misinterpret the task, often trying to render the table itself rather than visualizing its underlying data, as illustrated in Figure 3. To bridge this gap, we use an MLLM to perform semantic and structural reasoning with compositional planning. It translates the data-dense table into a detailed descriptive prompt that generation models can execute.

Refinement. Finally, as shown in Figure 2, these edit instructions are provided to an image editing model, which executes the corrections on the initial image to produce the refined, high-fidelity visualization.

##### 3.2. Training Details

Rewriting module. Although general LLMs have demonstrated powerful reasoning abilities on various tasks, their performance on table reasoning and compositional planning can be further enhanced. As Figure 3 shows, they may miss data points or plan poorly when encountering complex data (e.g., multi-layered structures). As this is the critical step responsible for basic layout design and content planning, fine-tuning the rewrite module is expected to significantly enhance the overall performance. Therefore, we fine-tune a specific rewrite module from Qwen3-8B [73] to address this challenge. With our constructed 30K rewriting training data (Section 4.2), we train the module with the standard

Generation. Upon obtaining the rewritten prompt, we feed it into a pre-trained text-to-image (T2I) model to produce an initial image. As shown in Figure 2, this preliminary result generally captures the overall layout and aesthetic. However, it often contains critical errors in high-fidelity details, such as incorrect data-to-visual correspondence (e.g., bar

Original Table (w/o Rewriting)

[Figure 13]

[Figure 14]

80

[Figure 15]

A clean infographic about Increasing Multi-Screen Behavior: Go Digital Go Mobile – Hong Kong media consumption presenting the following table. trends, showing data, chart style, professional design, high quality, 4k

[Figure 16]

|Main Category|Mobile Use|Subcategory|Usage|
|---|---|---|---|
|Internet|98%|Smartphone|97%|
| | |PC|61%|
| | |Tablet|27%|
|TV|44%|TV Set|94%|
| | |Digital|51%|
|Print|45%|Hard Copy|78%|
| | |Digital|54%|

[Figure 17]

70

62.8 63.4

61.3

60

Qwen3-8B (General LLM)

[Figure 18]

A modern, clean infographic-style image depicting Hong Kong's media consumption trends, focusing on the shift to multi-screen behavior. The central visual features a circular layout with icons representing key platforms: smartphone (97%), tablet (27%), PC (61%), TV set (94%), and digital print (54%). Each icon is accompanied by a percentage label in bold, white text on a dark blue background. The smartphone is highlighted as the dominant platform with a glowing effect, symbolizing the "Go Digital Go Mobile" trend. A secondary section shows a split between traditional print (78%) and digital print (54%), with a subtle cityscape background in the distance to represent Hong Kong. The color palette includes deep blues, bright whites, and soft grays for contrast, with a minimalist sans-serif font for clarity. The overall style is flat and tech-forward, emphasizing data accuracy and visual hierarchy.

Missing Data, Wrong Category

Score

54.3

51.8

50

50.1 49.4

Missing Data

[Figure 19]

[Figure 20]

[Figure 21]

40

Qwen3-8B* (Trained)

[Figure 22]

An infographic with a clean, minimalist design and a white background, titled “Go Digital Go Mobile” in a large, bold, red sans-serif font. Below the title, a subtitle reads “Increasing Multi-Screen Behavior: Hong Kong media consumption trends” in a smaller, grey font. The infographic is divided into three main vertical sections, each with a red header bar and white text.

30

**First Section (Internet):*** Header: "Internet"* Text: "98% Mobile Use" in large, bold, grey text.* List below: "Smartphone 97%", "PC 61%", "Tablet 27%".

[Figure 23]

✅ ✅ ✅

Generation Round 1 Round 2 Round 3

**Second Section (TV):*** Header: "TV"* Text: "44% Mobile Use" in large, bold, grey text.* List below: "TV Set 94%", "Digital 51%".

[Figure 24]

Qwen-Image-Edit-2509 Wan2.5-I2I-Preview

**Third Section (Print):*** Header: "Print"* Text: "45% Mobile Use" in large, bold, grey text.* List below: "Hard Copy 78%", "Digital 54%".At the bottom of the infographic, there is a solid red banner with the text: "Mobile is the dominant screen, but digital is also a big part of the story".

[Figure 25]

Figure 4. Demonstrating the refinement model’s capability as a critical pipeline bottleneck. The base model’s performance degrades with each correction, indicating an inability to process iterative feedback. In contrast, the Wan2.5-I2I-Preview shows consistent improvement. This confirms our pipeline structure is sound and that the bottleneck is the model’s capability, motivating our specialized training for the refinement module.

- Figure 3. Generation comparison among different rewriting. The model tends to directly render the table text, failed to visualize when disabling rewriting. General LLMs also tend to miss some data or wrongly classify data, especially for complex tables.

next-token prediction format:

preference dataset (Section 4.2), we fine-tune a Qwen2.5VL-3B [2] model, fθ, as our quality assessor. The model is trained to distinguish positive (xw) and negative (xl) samples for a given prompt p using the Bradley-Terry loss [6]:

N

1 N

) (1)

Lrewrite = −

log(ˆyn,k

n

n=1

where N denotes the sequence length, and yˆn,k

is the predicted probability for the true class kn at position n.

n

LBT = −E[log σ (fθ(xw,p) − fθ(xl,p))], (2) where σ(·) is the Sigmoid function. To enhance training efficiency, the scalar reward score is computed by extracting the probabilities corresponding to digits 0–9 from the output logits and averaging their sum. Finally, using our constructed 5K refinement data (Section 4.2), we follow Flow-GRPO [37] to perform the RL. The full reward signal combines our trained fθ with an existing aesthetic reward model, ImageReward [71], to optimize the training.

Refinement module. The refinement stage is critical for correcting errors, but it may also present a significant potential bottleneck. As illustrated in Figure 4, our preliminary experiments revealed this challenge. When employing a recent SOTA Qwen-Image-Edit [65] within our iterative loop, we observed a performance degradation with each correction round. This raised a critical question: is the pipeline’s self-correction logic flawed, or is the editing model’s capability insufficient? To investigate this variable, we introduced Wan2.5-I2I-Preview [51], another editing model known for fine-grained controllability, and found that performance indeed increased with each iteration. This indicates that our pipeline structure is sound, but its effectiveness is fundamentally constrained by the refinement model’s ability to execute precise edits. Therefore, to resolve this bottleneck, we train the refinement module using RL, specifically employing the Group Relative Policy Optimization (GRPO) algorithm [21]. This approach requires an accurate reward signal. However, evaluating rendering quality is complex, integrating multiple dimensions. As recent pretrained MLLMs struggle to directly provide consistent, accurate scalar scores for such assessments [41] (see Appendix), there remains a need to develop a specialized reward model (RM). Using our constructed 30K pairwise

#### 4. Dataset and Benchmark

##### 4.1. Data Collection and Filtering

To support both training and benchmarking for our proposed task, we construct a dataset of 30K high-quality tableimage pairs and an additional benchmark of 800 evaluation samples. We initiate this by collecting raw images from diverse public datasets, including SlideVQA [49], OpenImages [31], and Cambrian-10M [54]. We then applied the rigorous filtering and annotation pipeline shown in Figure 5. First, we discard low-resolution images (under 200 × 200) and images lacking text, as identified by PaddleOCR [13]. Following this, we utilize SOTA MLLMs (i.e., Gemini-2.5pro[12] and GPT-5 [48]) to perform a crucial filtering and

Data Collection and Filtering Rewriting Training Data Refinement Training Data

[Figure 26]

Retain Discard

GPT-5Gemini-2.5-pro

Wan-2.5 Qwen-Image

[Figure 27]

[Figure 28]

[Figure 29]

Generated image

[Figure 30]

Comparison

TableAnnotation Image

TableAnnotation

Table-based description

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Image

ImagePool

[Figure 35]

[Figure 36]

Refine command

[Figure 37]

Rollout samples

ImagePool

[Figure 38]

[Figure 39]

Rollout samplesRollout

[Figure 40]

[Figure 41]

[Figure 42]

Base Model

[Figure 43]

[Figure 44]

…

samples

[Figure 45]

[Figure 46]

Discard

Reward Training Data

[Figure 47]

Rationale to description

Wan-2.5 Refined image

[Figure 48]

Image represented inappropriately

[Figure 49]

Discard

Image w/o statistic data

Pos-Neg Pairs

D

Text-free image

[Figure 50]

[Figure 51]

Comparison

[Figure 52]

[Figure 53]

vs

D Low resolution image

- Figure 5. Dataset construction pipeline. We initially collect and filter images from public datasets with SOTA MLLMs, and then propose three different kinds of training data construction pipelines: rewriting training data, refinement training data, and reward training data.

annotation step: the models filter out images that do not feature statistical data as the main body, and simultaneously annotate the table information in markdown format for all remaining images. To ensure the quality of these annotations, we implemented a consensus-based verification process. Both MLLMs independently annotate the filtered images. We then retained only the samples for which the annotations from both models were consistent and mutually approved, resulting in our final set of 30K high-quality table-image pairs.

sure RL training stability. For each sample (initial image + instruction), we instruct our base editing model (QwenImage-Edit [65]) to generate five refined candidates. These candidates are compared against the initial image by a powerful MLLM assessor (GPT-5). We discard samples where all five attempts are judged as worse, or all five as better, than the original. This filtering isolates samples that are too hard or too easy for base model, yielding 5K challenging samples ideal for refinement training.

Reward training data. Our RL approach requires a reliable RM. Given that MLLMs are unstable in providing direct point-wise scores [41], we construct a preference dataset for converting the preference into a specific score assessor. We use GPT-5 and Gemini-2.5-pro to compare each image pair, followed by voting, and finally generate 30K positive-negative image pairs for training the RM, as shown in Figure 5. These pairs are sourced from three comparisons: 1) images refined by Wan2.5 versus images generated by Wan2.5 or Qwen; 2) images generated by Wan2.5 versus those by Qwen; and 3) collected ground-truth images versus generated or refined images.

##### 4.2. Training Data Construction

Based on the collected 30K table-image pairs, we propose three automated training data construction pipelines to support the distinct stages of our method.

Rewriting training data. As the rewriting module is critical for semantic reasoning and compositional planning, high-quality SFT data is essential. As shown in Figure 5, we first prompt Gemini-2.5-pro to generate a detailed description of the ground-truth image based on the annotated table, covering data points, layout, color, and background. We then prompt Gemini-2.5-pro again, providing both the table and the new description, to generate a chain-of-thought rationale that explains the conversion process. This results in 30K data pairs ({table, rationale} → {description}), which are used to fine-tune the rewriting module.

##### 4.3. Benchmark Design

To accurately evaluate performance on our creative table visualization task, we construct TableVisBench, a benchmark containing 800 challenging table-based instances. The collection and filtering process is similar to that of the training data, but with an additional step of manual verification and correction for all samples. Detailed statistics about the benchmark are provided in the Appendix.

Refinement training data. To generate data for RL, we first use the descriptions from the previous step to generate initial images using Wan2.5-t2i-preview [51] and QwenImage [65]. We then use Gemini-2.5-pro to audit these generated images and produce corresponding refinement instructions. We apply a rigorous filtering strategy to en-

For comprehensive benchmarking, we conduct multiview evaluation [40] with five well-designed dimensions,

Table 1. Performance comparison of recent strong open-sourced baselines on our TableVisBench. “RW” refers to the rewriting module, “REF” refers to the reflection and refinement process. We mark the improvement of our proposed pipeline for each base generation model.

Methods DA TR RR AA AQ Score

Reference Image 97.7 99.5 86.4 96.6 4.2 84.4 Flux [5] 12.1 46.7 28.9 18.7 4.0 29.3 RW+Flux 12.0 52.3 27.0 25.3 4.4 32.1 RW+Flux+REF 20.3 63.1 31.8 24.0 4.3 36.4 Improvement +8.2 +16.4 +2.9 +5.3 +0.3 +7.1 Bagel [14] 0.1 1.6 14.2 7.7 2.7 10.1 RW+Bagel 3.4 18.3 28.9 13.0 3.4 19.5 RW+Bagel+REF 18.3 54.8 36.7 15.9 3.8 32.7 Improvement +18.2 +53.2 +22.5 +8.2 +1.1 +22.6

Blip3o-Next [9] 0.4 18.0 4.4 6.2 2.5 10.8 RW+Blip3o-Next 0.5 14.5 19.1 7.6 2.9 14.1 RW+Blip3o-Next+REF 21.3 63.9 33.4 19.2 3.6 34.8 Improvement +20.9 +45.9 +29.0 +13.0 +1.1 +24.0

UniWorld-V1 [34] 3.0 18.3 14.7 2.9 3.5 14.8 RW+UniWorld-V1 4.0 20.8 23.7 11.6 3.3 18.6 RW+UniWorld-V1+REF 18.7 54.6 37.6 18.8 3.8 33.5 Improvement +15.7 +36.3 +22.9 +15.9 +0.3 +18.7

OmniGen2 [66] 3.1 17.8 13.5 2.6 3.5 14.4 RW+OmniGen2 4.0 32.1 25.0 9.5 3.9 21.9 RW+OmniGen2+REF 16.2 49.8 30.6 13.8 3.9 29.9 Improvement +13.1 +32.0 +17.1 +11.2 +0.4 +15.5

Qwen-Image [65] 47.5 90.9 26.1 14.1 4.3 44.3 RW+Qwen-Image 51.2 83.1 50.1 40.9 4.6 54.3 RW+Qwen-Image+REF 52.4 82.9 54.3 40.0 4.5 54.9 Improvement +4.9 -8.0 +28.2 +25.9 +0.2 +10.6

focusing not only on factual accuracy but also on logical coherence and visual aesthetics. Instead of using an MLLM as a subjective scorer, we leverage it as a quality assurance analyst. For the first four dimensions, the MLLM is prompted to identify and quantify specific errors within the generated chart. The final score is then deterministically calculated based on the number of reported errors, thereby mitigating the instability and bias associated with direct LLM-based scoring. The final dimension is quantitatively assessed using a dedicated aesthetic scoring model. The five dimensions are as follows:

Data Accuracy (DA). This dimension verifies that every single data point from the source table is accurately represented in the generated image, ensuring none are missing, incorrect, or simply rendered as raw table text.

Text Rendering (TR). This dimension focuses on the legibility and correctness of all textual elements in the image.

Relative Relationship (RR). This dimension assesses the core visualization logic, i.e., whether the visual proportions of chart elements (e.g., bar heights, slice angles) correctly reflect the quantitative relationships between data points.

Additional information Accuracy (AA). This dimension

Table 2. Ablation of the rewriting module of our ShowTable. All results reported in the table are the generated results after the rewriting process without reflection and refinement. The generation module utilized is Qwen-Image (8 steps distilled).

###### Rewrite DA TR RR AA AQ Score

False 47.5 90.9 26.1 14.1 4.3 44.3 Qwen3-8B [73] 30.6 71.5 46.6 34.1 5.1 46.8 GPT-5 [48] 35.9 78.5 47.8 41.8 5.2 51.2 Gemini-2.5-pro [12] 40.8 79.9 53.9 41.1 5.1 53.3 Qwen3-8B* 51.2 83.1 50.1 40.9 4.6 54.3

Reference-Caption 50.3 83.4 55.1 42.8 4.5 55.3

inspects the accuracy and appropriateness of contextual information added by the model (not present in the source table), such as axes, ticks, gridlines, and extraneous artifacts. Aesthetic Quality (AQ). Independent of factual correctness, this dimension evaluates the overall visual appeal of the generated chart, including its layout, color palette, typography, and design creativity.

The scores for the first four dimensions (DA, TR, RR, AA) range from 0 to 100, while AQ ranges from 0 to 10. We calculate the final score by:

Score = (DA + TR + RR + AA + 10 × AQ)/5 (3)

We provide detailed information about these dimensions in the Appendix. To validate the reliability of our benchmark, we test our dimensions on the collected ground-truth images. As shown in Table 1, these high-quality images achieve very high scores, confirming that our evaluation metrics are well-aligned with human-annotated quality.

#### 5. Experiments

##### 5.1. Implementation Setup

The modular design of our ShowTable pipeline allows for flexible combinations of MLLMs and diffusion models. In our default configuration, the rewriting module is trained based on Qwen3-8B [73], and the refinement module is trained based on a distilled 8-step version [42] of QwenImage-Edit-2509 [65] to accelerate RL training. For the reflection module, we employ GPT-5-2025-08-07 [48]. For fair comparison, all baselines that use Qwen-Image for generation or Qwen-Image-Edit-2509 for refinement also use these same distilled versions [42]. We set the maximum self-correction round to 3, though the process can terminate early if the reflection module deems an image satisfactory. More detailed settings can be found in the Appendix.

##### 5.2. Main Results

We comprehensively evaluate the effectiveness of the ShowTable pipeline by applying it to several advanced T2I

Table 3. Ablation of the different models of the reflection module. We use our trained Qwen3-8B as the rewriting module, QwenImage (distilled) as the generation module, and Qwen-Image-Edit2509 (distilled) / our trained model as the refining module here.

Reflection DA TR RR AA AQ Score Refinement: Qwen-Image-Edit-2509

Qwen3-VL-235B [1] 37.7 78.1 45.1 31.3 4.4 47.2 Gemini-2.5-pro [12] 43.2 81.2 46.6 41.1 4.4 51.2 GPT-5 [48] 42.6 79.7 45.0 35.9 4.4 49.4

Refinement: Qwen-Image-Edit-2509* (trained by ours)

Qwen3-VL-235B [1] 46.9 81.5 48.3 37.1 4.5 51.8 Gemini-2.5-pro [12] 48.1 83.0 48.8 44.8 4.5 53.9 GPT-5 [48] 52.4 82.9 54.3 40.0 4.5 54.9

generation models, including Flux [5], Bagel [14], Blip3oNext [9], UniWorld-V1 [34], OmniGen2 [66], and QwenImage [65]. The evaluation is conducted on our TableVisBench based on the five dimensions. We systematically compare performance under three configurations: 1) The base generation model alone (Base); 2) The base model prefixed with our rewriting module (RW+Base); 3) The full pipeline, integrating all modules (RW+Base+REF).

Quantitative analysis. The results, presented in Table 1, demonstrate three clear findings. First, base models alone are incapable of our proposed challenging task. Some models, such as Bagel and Blip3o-Next, score near zero (0.1 and 0.4, respectively) on Data accuracy, indicating a fundamental failure to translate table data into visual components. Second, the rewriting (RW) module is critical for reasoning and planning. Simply adding the RW module significantly boosts performance, especially in logical coherence. For instance, with Qwen-Image, the RR score jumps from 26.1 to 50.1. This shows that converting raw markdown into a reasoned, descriptive prompt is an essential first step. Third, the reflection and refinement (REF) loop is essential for accuracy. The full pipeline (RW+Base+REF) achieves the best overall score in all cases. This step yields the most significant gains in correctness-based metrics. With Blip3o-Next, the full pipeline improves DA from 0.5 to 21.3 and TR from 14.5 to 63.9. Moreover, our approach also shows strong adaptability. For models with weaker baseline capabilities (e.g., Bagel), our pipeline provides a substantial boost, improving the final score by +22.6 points, respectively. For strong base models like Qwen-Image, the full pipeline unlocks their potential, achieving the highest scores in key metrics and demonstrating a powerful synergistic effect, raising the overall score from 44.3 to 54.9.

Qualitative analysis. Qualitative results in Figure 6 also demonstrate ShowTable’s ability to produce creative yet accurate visualizations across diverse table structures. The reflection-refinement mechanism effectively corrects vari-

Table 4. Ablation of different models for the refinement module. We use our trained Qwen3-8B as the rewriting module, QwenImage (distilled) as the generation module, and GPT-5 as the reflection module here. * donates the method trained by ours.

###### Refining DA TR RR AA AQ Score

Qwen-Image-Edit-2509 42.6 79.7 45.0 35.9 4.4 49.4 Qwen-Image-Edit-2509* 52.4 82.9 54.3 40.0 4.5 54.9 Improvement +9.8 +3.2 +9.3 +4.1 +0.1 +5.5

Wan2.5-I2I-Preview 64.2 84.7 64.6 59.7 4.4 63.4

ous error types: misrendered text and numbers (Row 1Left, Row 2), incorrect proportional relationships (Row 1Left), and improper visual element representations (Row 3). Cases requiring zero refinement (Row 1-Right) confirm the pipeline’s adaptive efficiency. These results validate the robustness of ShowTable in achieving faithful and creative table-to-image translation.

##### 5.3. Ablation Studies

Rewriting module. We evaluate different rewriting strategies using the Qwen-Image generation model. As shown in Table 2, compared settings include: 1) no rewriting (False), 2) general-purpose LLMs (Qwen3-8B, GPT5, Gemini-2.5-pro), 3) our fine-tuned model (Qwen3-8B*), and 4) an upper-bound using reference captions (ReferenceCaption). All results are based on the initial generation (no refinement). Our fine-tuned module (Qwen3-8B*) achieves the highest overall score (54.3) among all evaluated rewriting methods, demonstrating a strong, well-balanced performance. Notably, it attains the best DA (51.2), confirming that specialized training is more effective at preserving data integrity than general-purpose LLMs (30.6-40.8). While all rewriting methods slightly reduce the TR score compared to the baseline (which often just renders the table text directly), our model’s substantial gains in DA (+3.7 vs base) and RR (+24.0 vs base) justify this trade-off. Notably, our fine-tuned model even surpasses the Reference-Caption’s DA (51.2 vs. 50.3), further underscoring the advantage of training. While Gemini-2.5-pro achieves the highest RR score, our model remains highly competitive and delivers the best overall performance, highlighting its robustness.

Reflection module. We evaluate the effectiveness of different MLLMs as the reflection module. As shown in Table 3, we use our fine-tuned rewriting module (Qwen3-8B*) and the Qwen-Image generation model. We then compare the final performance with Qwen3-VL-235B, Gemini-2.5-pro, and GPT-5 as reflection modules. The results demonstrate the significant impact of the reflection module on output quality. When using our trained refinement model (bottom half of table), GPT-5 achieves the strongest performance (54.9), excelling in Data Accuracy (52.4) and Relative Re-

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

|Public perception of sustainability efforts in different travel industries|
|---|
|Doin g…<br><br>Goo d…<br><br>Littl e…<br><br>No …<br><br>Industry|
|Hotels… 8 52 35 5|
|Train… 15 47 31 7|
|Bus… 23 52 22 5|
|Airlines 25 49 21 6|
|Cruise… 32 45 19 4|

Refinement Prompt 1

Update … Hotels … to show …52%... 35%... Update Train operators … 47% Little Effort…, with segment lengths proportionally adjusted.

| |
|---|
| |

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Refinement Prompt 1

Refinement Prompt 2

Change the text label for 'Availability' in the infographic from '10.7%' to '19.0%’… while keeping the same font, color, and style.

Update the Environmental Impact label in the infographic from '7.7%' to

'10.7%' to match the table data…, while keeping the same font…

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Refinement Prompt 1

Refinement Prompt 2

Correct the … three segments labeled… Remove duplicate

Adjust ‘Important’ … occupies about 64% ...Change its label to ‘Important – 64%’ and position.. Adjust the size of remaining slices so that 'Moderately important' remains at 27% and...

‘Important’… Adjust the segment sizes so they visually represent 64%, 27%, and 9% of the circle…

- Figure 6. Case studies of the ShowTable pipeline with four examples: top row shows one case requiring one-round reflection and refinement (left) and one proper initial generation (right), while other rows display results refined through multiple reflection rounds. Examples demonstrate adaptive correction of text, proportions, and visual elements. We mark the error parts with red boxes for better reading.

Table 5. Ablation of the maximum refining rounds. We keep the same setting as Table 4, and compare results of each round.

Methods Num DA TR RR AA AQ Score w/o refine 0 51.2 83.1 50.1 40.9 4.6 54.3

- 1 45.6 81.7 48.3 38.3 4.5 51.8

- 2 42.6 80.7 46.2 36.2 4.5 50.1

- 3 42.6 79.7 45.0 35.9 4.4 49.4

Qwen-Image-Edit-2509

- 1 50.0 82.6 51.3 39.4 4.5 53.7

- 2 50.4 83.0 52.9 41.5 4.6 54.8

- 3 52.4 82.9 54.3 40.0 4.5 54.9

Qwen-Image-Edit-2509*

- 1 60.7 85.2 61.8 53.8 4.5 61.3

- 2 63.3 85.1 64.1 57.5 4.4 62.8

- 3 64.2 84.7 64.6 59.7 4.4 63.4

Wan2.5-I2I-Preview

lationship (54.3). When paired with the base Qwen-ImageEdit module (top half of table), Gemini-2.5-pro delivers the best results (51.2). These findings confirm that more capable reflection models consistently enhance output quality, providing practical guidance for model selection.

Refinement module. We evaluate different refining modules under fixed rewriting (our fine-tuned Qwen3-8B*) and reflection (GPT-5) conditions, with results in Table 4. Our trained refinement model (Qwen-Image-Edit-2509*) achieves significant improvements over its base model, increasing the overall score from 49.4 to 54.9 (+5.5). The model proves notable gains in DA (+9.8) and RR (+9.3), validating our specialized RL-based training approach. As shown in Table 4, while the powerful Wan2.5 achieves a

higher performance (63.4), our trained model’s substantial enhancement proves that our method effectively boosts open-source base models, offering a viable path for customized refinement solutions.

Refining rounds. We analyze the impact of iterative refinement rounds in Table 5. The results demonstrate that the refinement model’s capability is critical. The base Qwen-Image-Edit model shows performance degradation with each round (Score 54.3 → 49.4), indicating an inability to process iterative corrections. In contrast, our trained model (Qwen-Image-Edit-2509*) maintains stable improvement (53.7 → 54.9), validating that our specialized training successfully addresses this error accumulation. Furthermore, the powerful Wan2.5 model achieves continuous improvement (61.3 → 63.4), confirming that more capable models are essential for effective multi-round refinement. This underscores the necessity of our training, which successfully enhances the open-source model to reliably support the iterative process.

#### 6. Conclusion

This work introduces the creative table visualization task that demands both aesthetic graphic reasoning and highfidelity data mapping, addressing the critical challenge of generating faithful and aesthetic data visualizations. To address this, we propose ShowTable, a novel pipeline that synergizes MLLMs with diffusion models through iterative reflection and refinement, significantly improving visualdata mapping alignment. To support this task, we also

present three automated data construction pipelines for different module training. Moreover, we propose a new comprehensive benchmark TableVisBench with 5 evaluation dimensions. Experiments demonstrate our approach’s effectiveness in producing accurate and aesthetically coherent table visualizations, establishing a foundation for future research in multi-modal reasoning and visual synthesis.

#### Acknowledgment

This work is supported by the National Nature Science Foundation of China (62425114, 62121002, U23B2028), and the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM103). We acknowledge the support of Alibaba Group, the GPU cluster built by MCC Lab of Information Science and Technology Institution, USTC, and USTC super-computing center for providing computational resources for this project.

#### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 7
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4, 1, 3
- [3] Xiaoyi Bao, Siyang Sun, Shuailei Ma, Kecheng Zheng, Yuxin Guo, Guosheng Zhao, Yun Zheng, and Xingang Wang. Cores: Orchestrating the dance of reasoning and segmentation. In European Conference on Computer Vision, pages 187–204. Springer, 2024. 2
- [4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2
- [5] BlackForest. Flux. https://github.com/blackforest-labs/flux, 2024. 1, 2, 6, 7
- [6] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 4
- [7] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser-2: Unleashing the power of language models for text rendering. In European Conference on Computer Vision, pages 386–402. Springer, 2024. 1, 2
- [8] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 1, 2
- [9] Jiuhai Chen, Le Xue, Zhiyang Xu, Xichen Pan, Shusheng Yang, Can Qin, An Yan, Honglu Zhou, Zeyuan Chen, Lifu Huang, et al. Blip3o-next: Next frontier of native image generation. arXiv preprint arXiv:2510.15857, 2025. 6, 7

- [10] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 1, 2

- [11] Ethan Chern, Zhulin Hu, Steffi Chern, Siqi Kou, Jiadi Su, Yan Ma, Zhijie Deng, and Pengfei Liu. Thinking with generated images. arXiv preprint arXiv:2505.22525, 2025. 2
- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 4, 6, 7
- [13] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025. 4
- [14] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 2, 6, 7
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1, 2

- [16] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023. 2
- [17] Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, et al. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639, 2025. 2
- [18] James Ford and Anthony Rios. Does it run and is that enough? revisiting text-to-chart generation with a multiagent approach. arXiv preprint arXiv:2506.06175, 2025. 1
- [19] Yifan Gao, Zihang Lin, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, and Hongtao Xie. Postermaker: Towards high-quality product poster generation with accurate text rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8083–8093, 2025. 1, 2
- [20] Google. Nano banana pro. https://blog.google/ innovation-and-ai/products/nano-bananapro/, 2025. 2
- [21] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 4, 2
- [22] Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A mul-

- timodal llm for chart understanding and generation. arXiv preprint arXiv:2311.16483, 2023. 1
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [24] Xiwei Hu, Haokun Chen, Zhongqi Qi, Hui Zhang, Dexiang Hong, Jie Shao, and Xinglong Wu. Dreamposter: A unified framework for image-conditioned generative poster design. arXiv preprint arXiv:2507.04218, 2025. 1, 2
- [25] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024. 2
- [26] Minbin Huang, Yanxin Long, Xinchi Deng, Ruihang Chu, Jiangfeng Xiong, Xiaodan Liang, Hong Cheng, Qinglin Lu, and Wei Liu. Dialoggen: Multi-modal interactive dialogue system for multi-turn text-to-image generation. arXiv preprint arXiv:2403.08857, 2024. 2
- [27] Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, et al. Interleaving reasoning for better text-to-image generation. arXiv preprint arXiv:2509.06945, 2025. 2
- [28] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025. 1, 2
- [29] Kaixun Jiang, Zhaoyu Chen, Hao Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, and Wenqiang Zhang. Efficient decision-based black-box patch attacks on video recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4379–4389, 2023. 2
- [30] Kaixun Jiang, Zhaoyu Chen, Haijing Guo, Jinglun Li, Jiyuan Fu, Pinxue Guo, Hao Tang, Bo Li, and Wenqiang Zhang. Enhancing diffusion-based unrestricted adversarial attacks via adversary preferences alignment. arXiv preprint arXiv:2506.01511, 2025. 2
- [31] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7):1956–1981, 2020. 4
- [32] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flowbased grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025. 2
- [33] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025. 2
- [34] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic en-

- coders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 1, 2, 6, 7
- [35] Jinpeng Lin, Min Zhou, Ye Ma, Yifan Gao, Chenxi Fei, Yangjian Chen, Zhang Yu, and Tiezheng Ge. Autoposter: A highly automatic and content-aware design system for advertising poster generation. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1250–1260,

2023. 1, 2

- [36] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2023. 1
- [37] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 4, 2
- [38] Zeyu Liu, Weicong Liang, Zhanhao Liang, Chong Luo, Ji Li, Gao Huang, and Yuhui Yuan. Glyph-byt5: A customized text encoder for accurate visual text rendering. In European Conference on Computer Vision, pages 361–377. Springer,

2024. 1, 2

- [39] Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, and Hongtao Xie. Hybrid-level instruction injection for video token compression in multi-modal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8568–8578, 2025. 1
- [40] Zhihang Liu, Chen-Wei Xie, Bin Wen, Feiwu Yu, Jixuan Chen, Pandeng Li, Boqiang Zhang, Nianzu Yang, Yinglu Li, Zuan Gao, Yun Zheng, and Hongtao Xie. Capability: A comprehensive visual caption benchmark for evaluating both correctness and thoroughness, 2025. 5
- [41] Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, et al. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909, 2025. 4, 5
- [42] ModelTC. Qwen-image-lightning. https://github. com/ModelTC/Qwen-Image-Lightning, 2025. 6, 4
- [43] OpenAI. Dall·e 3. https://openai.com/index/ dall-e-3/, 2023. 1, 2
- [44] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [45] Md Mahinur Rashid, Hasin Kawsar Jahan, Annysha Huzzat, Riyasaat Ahmed Rahul, Tamim Bin Zakir, Farhana Meem, Md Saddam Hossain Mukta, and Swakkhar Shatabda. Text2chart: A multi-staged chart generator from natural language text. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pages 3–16. Springer, 2022. 1
- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [47] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li,

- Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2
- [48] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2026. 4, 6, 7
- [49] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13636–13645, 2023. 4
- [50] Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025. 1
- [51] Wan Team. Wan2.5. https://www.wan-ai.co/wan2-5, 2025. 1, 2, 4, 5, 8
- [52] Wan Team. Wan2.6. https : / / wan . video / introduction/wan2.6, 2025. 2
- [53] Yuan Tian, Weiwei Cui, Dazhen Deng, Xinjing Yi, Yurun Yang, Haidong Zhang, and Yingcai Wu. Chartgpt: Leveraging llms to generate charts from abstract natural language. IEEE Transactions on Visualization and Computer Graphics, 31(3):1731–1745, 2024. 1
- [54] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 4
- [55] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. arXiv preprint arXiv:2311.03054, 2023. 2
- [56] Yuxiang Tuo, Yifeng Geng, and Liefeng Bo. Anytext2: Visual text generation and editing with customizable attributes. arXiv preprint arXiv:2411.15245, 2024. 1, 2
- [57] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 2
- [58] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 2
- [59] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025. 2
- [60] Zhenyu Wang, Aoxue Li, Zhenguo Li, and Xihui Liu. Genartist: Multimodal llm as an agent for unified image generation and editing. Advances in Neural Information Processing Systems, 37:128374–128395, 2024. 2

- [61] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024. 2
- [62] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, et al. Dreamvideo-2: Zero-shot subjectdriven video customization with precise motion control. arXiv preprint arXiv:2410.13830, 2024.
- [63] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Biao Gong, Longxiang Tang, Xiang Wang, Haonan Qiu, Hengjia Li, Shuai Tan, Yingya Zhang, et al. Dreamrelation: Relation-centric video customization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12381–12393,

2025. 2

- [64] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Yujin Han, Zhekai Chen, Jiayu Wang, Difan Zou, Xihui Liu, Yingya Zhang, Yu Liu, et al. Routing matters in moe: Scaling diffusion transformers with explicit routing guidance. arXiv preprint arXiv:2510.24711, 2025. 1
- [65] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 2, 4, 5, 6, 7
- [66] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 1, 2, 6, 7
- [67] Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al. Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025. 2
- [68] Shishi Xiao, Suizi Huang, Yue Lin, Yilin Ye, and Wei Zeng. Let the chart spark: Embedding semantic context into chart with text-to-image generative model. IEEE Transactions on Visualization and Computer Graphics, 30(1):284–294, 2023. 1
- [69] Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, and Ying Shan. Mindomni: Unleashing reasoning generation in vision language models with rgpo. arXiv preprint arXiv:2505.13031, 2025. 2
- [70] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 1, 2
- [71] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023. 4, 2
- [72] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo,

- Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 2
- [73] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 3, 6
- [74] Fatemeh Pesaran Zadeh, Juyeon Kim, Jin-Hwa Kim, and Gunhee Kim. Text2chart31: Instruction tuning for chart generation with automatic feedback. arXiv preprint arXiv:2410.04064, 2024. 1
- [75] Songheng Zhang, Lei Wang, Toby Jia-Jun Li, Qiaomu Shen, Yixin Cao, and Yong Wang. Chartifytext: Automated chart generation from data-involved texts via llm. arXiv preprint arXiv:2410.14331, 2024. 1
- [76] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025. 1
- [77] Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025.
- [78] Xuanle Zhao, Xuexin Liu, Haoyue Yang, Xianzhen Luo, Fanhu Zeng, Jianling Li, Qi Shi, and Chi Chen. Chartedit: How far are mllms from automating chart analysis? evaluating mllms’ capability via chart editing. arXiv preprint arXiv:2505.11935, 2025. 1
- [79] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372, 2024. 2
- [80] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 2, 1
- [81] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 37:131278–131315, 2024. 2

## ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement

### Supplementary Material

Overview

- A. More Related Works 1

- A.1. Chart Generation with Agentic Tools . . . . 1
- A.2. Reinforcement Learning for Image Generation 2

- B. More Method Details 2

- B.1. Prompts in Pipeline . . . . . . . . . . . . . . 2
- B.2. Rewriting Training Details . . . . . . . . . . 2
- B.3. Refinement Training Details . . . . . . . . . 2

- C. More Dataset and Benchmark Details 4

- C.1. Training Data Format . . . . . . . . . . . . . 4
- C.2. Benchmark Statistics . . . . . . . . . . . . . 5
- C.3. Benchmark Evaluation Details . . . . . . . . 5

- D. More Experiments 8

- D.1. Results of Wan2.5-Preview . . . . . . . . . . 8
- D.2. More Visualization Results . . . . . . . . . . 8
- D.3. User Study . . . . . . . . . . . . . . . . . . 8

- E. Limitations and Future Work 10

#### A. More Related Works

##### A.1. Chart Generation with Agentic Tools

With rapid development of current MLLMs [2, 36, 39, 80], the strong tool calling ability provides a promising way to complete multi-modal tasks. Current approaches for chart generation predominantly rely on LLMs coupled with agentic tools, generally falling into three main categories. The first category [45, 53, 75] employs LLMs in an end-toend manner to produce charts directly from text. These methods typically parse input to identify axes, map entities, and classify chart types, subsequently generating structured specifications for rendering engines like Vega-Lite. While automated, their expressiveness is strictly confined by the predefined grammatical rules of the underlying visualization language. The second category [22, 74, 76–78] focuses on generating executable plotting code (e.g., via Matplotlib/Python). Recent works have incorporated multiagent frameworks and reflection mechanisms to improve the syntactic correctness of the generated code [18]. Although these methods offer precise control, they heavily depend on

Table A1. The comparison between our ShowTable pipeline and code-based methods. For our ShowTable pipeline, we use Gemini2.5-pro for rewriting, Wan2.5-T2I-Preview for generation, GPT-5 for Reflection, and Wan2.5-I2I-Preview for refinement here. For code-based methods, we use Gemini-2.5-pro to write python code with the matplotlib library.

###### Methods DA TR RR AA AQ Score

ShowTable Pipeline 69.6 84.9 67.6 68.4 4.9 67.9 Code (Gemini-2.5-pro) 83.8 89.2 85.3 97.7 4.1 79.4

Code ShowTable

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Figure A1. The qualitative comparison between code-based methods and our ShowTable pipeline. The code-based methods often prioritize “correctness” but may fail at “presentation”

external rendering engines and typically lack the capability to handle complex, artistic visual designs beyond standard plots. The third category [68] adopts a retrieval-editing pipeline that selects visual templates from an image corpus and adapts them to new data. While this benefits from template reuse, it is limited by the diversity of the database and often faces challenges in accurately aligning new data with retrieved visual structures.

Discussion. While existing methods excel in structural automation and factual plotting, they fundamentally struggle with creativity and aesthetics. Code-based and templatebased approaches are bound by rigid rendering logic, making it difficult to produce visually striking infographics suit-

able for professional poster design, slide generation, or data-driven storytelling. They prioritize “correctness” but often fail at “presentation”. To explicitly validate this, we conducted a quantitative comparison between a strong code-based baseline (e.g., using Gemini-2.5-Pro to generate executable plotting code) and our ShowTable pipeline, as shown in Tab. A1 and Fig. A1. As expected, code generation methods inherently achieve superior factual accuracy across structural metrics such as Data Accuracy and Relative Relationship. However, they fall significantly short in Aesthetic Quality (AQ: 4.1 vs. our 4.9) and frequently produce visual artifacts, such as overlapping text. This rigidity restricts their utility in design-centric, creative scenarios.

Importantly, our objective is not to replace traditional code-based rendering tools where absolute numerical precision is the sole priority. Instead, our work proposes the Creative Table Visualization task to explore the untapped potential of generative models and unified models in this domain. We frame our contribution as a technical advancement in multi-modal generation control: we aim to retain the unparalleled creative flexibility of generation models while actively mitigating their primary weakness, data fidelity, through our progressive self-correcting pipeline. We argue that generative models offer a significantly higher ceiling for flexibility and aesthetic quality, capable of seamlessly integrating data into artistic compositions. Furthermore, equipping text-to-image models with rigorous data fidelity is a vital frontier for achieving more general capabilities in visual synthesis (including scientific reporting), as seen in the trajectory of recent models like Wan2.6 [52] and Nano Banana [20] after the CVPR submission. By validating the feasibility of this approach, we aim to break the traditional boundaries of rule-based rendering and pave the way for more robust, unified, and creative visual synthesis systems.

##### A.2. Reinforcement Learning for Image Generation

Diffusion models have established themselves as the predominant framework for text-to-image (T2I) generation [5, 23, 30, 44, 46, 65] and text-to-video (T2V) generation [29, 61–63]. The integration of reinforcement learning (RL) into this paradigm began with works utilizing policy gradient optimization to guide the denoising process [4, 16, 67, 71]. The field subsequently expanded to include preferencebased alignment methods, which achieve competitive performance without explicit reward modeling [57]. A significant recent development is the adoption of Group Relative Policy Optimization(GRPO) [21, 47], an efficient alternative that has inspired numerous adaptations for T2I generation. These include pioneering works [37, 72] which unified diffusion and flow matching under an SDE-based formulation. This line of inquiry further explores the design of specialized reward models and data curation strategies

to enhance the framework’s capability for producing highquality, preference-aligned visual outputs [32, 33, 59].

#### B. More Method Details

##### B.1. Prompts in Pipeline

In our ShowTable pipeline, the MLLM acts as the central orchestrator, performing two key roles: rewriting and reflection. We detail the specific prompts used for these modules. The rewriting prompt, shown in Figure A2, instructs the MLLM to reason over the input table and translate its dense data into a detailed descriptive prompt suitable for the diffusion executor. The reflection prompt, shown in Figure A3, guides the MLLM to critically audit the generated image against the original table, identify inaccuracies, and formulate precise, actionable editing instructions for the refinement stage. To ensure fair comparison and consistency across experiments, we use the same system and user prompts for all models tested in these roles.

##### B.2. Rewriting Training Details

As discussed in Section 3.2, we fine-tune a specialized rewriting module based on Qwen3-8B to handle the critical task of reasoning and compositional planning. This module is trained on our 30K SFT data with both thinking and rewriting result, and we construct various kinds of instruction templates during training to ensure the diversity. For the implementation, we utilize the LLaMA-Factory [79] library. We train the model for 3 epochs with a total batch size of 256. The training employs a learning rate of 1e-5, combined with a cosine learning rate decay strategy to stabilize the training process.

Through targeted fine-tuning, the resulting rewrite model acquires the ability to infer appropriate data visualization and layout strategies. The thinking process enables the model to generate significantly improved prompts, thereby enhancing both data integrity and accuracy throughout the prompt rewriting and subsequent image generation pipeline.

##### B.3. Refinement Training Details

To empower the refinement model with the capability for precise, fine-grained edits on dense data infographics, we employ an on-policy reinforcement learning approach utilizing the Group Relative Policy Optimization (GRPO) [21] algorithm. In this framework, rendering accuracy serves as the critical reward signal.

Reward model. Evaluating the rendering accuracy of infographics is complex, requiring the integration of multiple dimensions—such as textual correctness, data-to-visual alignment, and spatial layout. Our preliminary experiments revealed that utilizing state-of-the-art pretrained VisionLanguage Models (VLMs) directly for point-wise reward assessment is suboptimal. As shown in Figure A4, the in-

Rewriting

system_prompt = "You are a helpful assistant. Your task is to carefully understand the user’s input structured data (e.g., Markdown tables) and transform it into a more detailed, precise, and well-structured description, making it fully optimized for image generation models to produce outputs that strictly align with the user’s requirements. When presented with a table, you deeply interpret its content, then craft a natural, narrative think process that envisions how the information should be transformed into a compelling visual representation. This includes reasoning about the suitable visualization style, layout structure, color palette, typography, iconography, and compositional balance, while ensuring data accuracy and aesthetic clarity.”

user_prompt = f"You are an expert prompt engineer. Your client wants an image about ‘{topic}', and has provided this data. Write the perfect prompt to get a stunning result from a model like Stable Diffusion.\n{table}\nThink first in <think> tag, then directly output your prompt in <answer> tag. Format strictly as: <think>your thinking</think><answer>put your refined prompt here.</answer>"

- Figure A2. The system prompt and user prompt for the rewriting module. We use the same prompts for all models.

Reflection

system_prompt = r'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table and verify **numeric accuracy, geometric proportionality, and labeling fidelity**. If any discrepancy exists, you must explain it and produce a precise, actionable instruction for an image editing model.

# Tools When necessary, call functions defined in <tools></tools>: <tools>{"type": "function", "function": {"name": "image_editing_tool", "description": "Edit the provided image based on the provided prompt", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A descriptive prompt for the image to be edited"}}, "required": ["prompt"]}}}</tools>

Return each function call as a JSON object wrapped in <tool_call>: <tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>

# Mandatory Checks (apply all relevant)

- 1) **Data vs. Marks**

- - Bars/points/slices and data labels must equal the table values.
- - Stacked totals equal the sum of segments; percent series sum to ~100%.

- 2) **Axis & Scale Consistency (CRITICAL)**

- - Identify axis min/max, tick marks, units, and scale type.
- - **Linear interpolation rule**: a value v must map to a position linearly between its surrounding ticks.
- - **Overflow/undershoot rule (must flag)**:
- - If a mark’s top/point **visually exceeds the max tick** while its value ≤ max tick (e.g., a bar labeled 565.8 on a Y-axis with a 600 tick but the bar top is above 600), **this is an error**.
- - If a mark’s top/point is **below** where v should be (e.g., 565.8 drawn near 520) beyond tolerance, **this is an error**.
- - **Label–position agreement**: a data label’s anchor height must be consistent with the axis mapping for the labeled value.
- - **Zero baseline**: Columns/bars start at 0 unless explicitly truncated and labeled as such.
- - **Tick spacing** must be uniform for linear scales (or powers for log scales).
- - **Tolerance**: Flag if the observed position/length deviates by > **2% of the full axis span** or **>3 px**, whichever is larger. For values near the max tick, also flag if the mark crosses the max tick line at all.

- 3) **Pie/Donut Proportionality (CRITICAL)**

- - Slice angle ∝ value; 78% must be visibly larger than 20%; 30% > 25%.
- - **Angle tolerance**: deviation > **5°** or totals ≠ 100% ±0.5%.

- 4) **Completeness**

- All categories/series from the table are present; nothing extra appears.

- 5) **Labeling & Mapping**

- Titles, axes, units, legends, series names, colors, and ordering claims match the table and encodings.

- 6) **Insufficient Information = Discrepancy**

- - Missing/illegible ticks or units that block verification must be corrected (instruct the editor). # How to Reason (be numeric and explicit)
- - State the axis range and tick step you infer (e.g., “Y-axis ticks at 0, 200, 400, 600 M$”).
- - Compare **expected vs. observed** positions using linear interpolation.
- - Call out **overflow/undershoot** explicitly (e.g., “565.8 should sit just below the 600 tick, but the bar top is above the 600 line by ~5–10 px”).
- - Be specific about where the problem occurs (series/category/color/axis/legend/slice). # Output Requirements
- - `<assessment>`: one-sentence verdict (e.g., “The image contains critical axis-position errors.”).
- - `<analysis>`: bullet or paragraph list of issues stating **what**, **where**, **why** (reference table and axis logic/tolerance).
- - `<tool_call>` (optional): If any issue exists, provide **one** concise paragraph of directives that fix **all** issues (e.g., “Lower the ‘Category X’ bar so its top aligns with 565.8 on a 0–600–800 scale; ensure Y ticks at 0, 200, 400, 600, 800 M$; move the data label to sit just below 600; …”).
- - If **no issues**, do **not** call any tool and end with `<answer>done</answer>`. Tone: objective, precise, analytical. No conversational filler.‘’’

user_prompt = f"Your primary goal is to generate a high-quality prompt for an image editing model if, and only if, the infographic for \"{topic}\" has errors. First, find and analyze the errors. Then, craft the image editing tool prompt. If no errors are found, your only final output after analysis should be \"<answer>done</answer>\".\n Table data: {table}\nThink first in <think> and </think> tag. Your output must strictly follow the format: <think>your thinking</think><assessment>...</assessment><analysis>...</analysis><tool_call>...</tool_call> or <think>your thinking</think><assessment>...</assessment><analysis>...</analysis><answer>done</answer>."

- Figure A3. The system prompt and user prompt for the reflection module. We use the same prompts for all models.

consistency in VLM scoring often leads to training instability or collapse. Furthermore, the high inference latency of VLMs significantly hampers the efficiency of the on-policy GRPO loop.

from the same prompt, as detailed in Section 4.2. We finetune a Qwen2.5-VL-3B [2] model, denoted as fθ, to serve as the quality assessor. For a given prompt p, with xw denoting the preferred (positive) sample and xl the dispreferred (negative) sample, the model is optimized using the Bradley-

To address this, we develop a specialized, efficient Reward Model (RM). We construct a pairwise dataset D consisting of positive and negative graphic samples generated

- Figure A4. The reward comparison between our RM and direct LLM scoring, we achieve a stable reward increase.

Terry (BT) loss [6]:

LBT = −E(p,x

w,xl)∼D [log σ (fθ(xw,p) − fθ(xl,p))],

(A1) where σ(·) represents the Sigmoid function. To stabilize the output, the reward score is computed by extracting and averaging the probabilities corresponding to the tokens for digits 0–9 from the output logits. The final reward signal used in RL is a weighted combination: R = 0.8·fθ(x,p)+

- 0.2 · ImageReward(x,p) [71]. Policy optimization. The refinement policy is updated using the GRPO algorithm. The objective function maximizes the expected reward while constraining policy divergence via a clipped surrogate objective:

JGRPO(θ) =

G

πθ(oi|y) πθ

1 G

Ey∼D min

Ai,

(oi|y)

old

i=1

πθ(oi|y) πθ

clip

,1 − ε,1 + ε Ai

(oi|y)

old

− βDKL (πθ ∥ πref) ,

(A2)

where ε and β are hyperparameters, and G is the group size. The advantage Ai is computed by normalizing the rewards {r1,r2,··· ,rG} within each group:

ri − mean({r1,··· ,rG}) std({r1,··· ,rG}) + ϵ

. (A3)

Ai =

Implementation. Following the Flow-GRPO framework [37], we train our refinement model using a distilled 8-step version of Qwen-Image-Edit-2509 [42] to accelerate training efficiency. The model is trained on our constructed 5K refinement dataset for 1 epoch using 32 GPUs. We set the image resolution to 1024×1024 and perform 16 rollouts per prompt. We utilize 8 sampling steps for both the inference rollout and the training backward pass. As illustrated in Figure A4, compared to the unstable baseline using raw LLM scores, our approach with the trained Reward Model achieves stable and consistent performance gains.

#### C. More Dataset and Benchmark Details

##### C.1. Training Data Format

As illustrated in Figure 5, our data construction pipeline transforms raw collected images into specialized datasets tailored for the rewriting, refinement, and reward modules. Below, we detail the specific input and output formats for each training stage.

Rewriting Training Data (SFT). The goal of the rewriting module is to convert a raw markdown table into a comprehensive visual plan. To support this, we construct Supervised Fine-Tuning (SFT) data that teaches the model to first reason about the data structure (Rationale) and then describe the visual elements (Description).

- • Input: The raw table data in markdown format, annotated from the collected image pool.
- • Output: A composite text sequence consisting of a Chain-of-Thought Rationale followed by a Detailed Image Description.
- • Construction: As shown in the green section of Figure 5, we first prompt Gemini-2.5-pro to generate a descriptive caption (“Table-based description”). Then, we feed both the table and the description into Gemini-2.5pro again to reverse-engineer the reasoning process (“Rationale to description”), forming a complete training sample: Table → Rationale + Description.

Refinement Training Data (RL). For the reinforcement learning stage, the training data consists of challenging scenarios where an initial generation with correction instructions, and no ground-truth image is needed. This data is formatted as prompt-response pairs for the policy model.

- • Input: A pair consisting of an Initial Generated Image (containing errors) and a precise Refinement Instruction.
- • Output: No refined image is needed.
- • Construction: As shown in the blue section of Figure 5, we generate initial images from our descriptions and use Gemini-2.5-pro to compare them against the table, producing a “Refine command.” To ensure the data is valid for training, we perform a “Rollout” check: we discard samples where the base model either fails to improve the image over multiple attempts (too hard) or solves it trivially (too easy), retaining only those suitable for learning stable policy gradients.

Reward Training Data (Preference Pairs). To train the reward model fθ as a reliable quality assessor, we construct a dataset of preference image pairs, focusing on the data fidelity.

- • Input: A text condition (the table) and two candidate images (xw,xl).
- • Output: A binary label indicating which image is the “Winner” (xw) and which is the “Loser” (xl).
- • Construction: As shown in the purple section of Figure 5, we source candidates from three comparisons: (1)

[Figure 70]

[Figure 71]

- Figure A5. The Statistical information about our proposed TableVisBench. (a) (Left): the data length distribution of TableVisBench. (b) (Right): The word cloud of the topic in TableVisBench.

Refined vs. Initial images, (2) Strong (Wan2.5) vs. Weak (Qwen) model outputs, and (3) Ground-truth vs. Generated images. MLLMs (GPT-5 and Gemini-2.5-pro) act as judges to vote on the pair, establishing a high-confidence “Pos-Neg Pair” dataset for training the reward model to discriminate fine-grained visual differences.

##### C.2. Benchmark Statistics

To demonstrate the diversity and complexity of our proposed TableVisBench, we present detailed statistical characteristics in Figure A5.

Data Length Distribution. Figure A5 (a) illustrates the distribution of table lengths, defined as the number of key data points per instance. The benchmark covers a broad spectrum of information density, ranging from concise tables (fewer than 5 rows) to highly complex ones. The distribution shows a natural concentration between 5 to 15 data points, reflecting common real-world infographic scenarios. Notably, we also include some “long-tail” instances, with the final bin aggregating tables containing over 40 data points. This design ensures that the benchmark rigorously evaluates models not only on standard visualizations but also on their stability and layout planning capabilities when handling high-density data.

Topic Diversity. Figure A5 (b) presents a word cloud visualization derived from the topics of the collected tables. The dataset encompasses a wide array of domains, with prominent keywords including “Social Media,” “Distribution,” “Percentage,” “User,” “Market,” and “Mobile.” This semantic diversity confirms that TableVisBench covers various distinct fields—such as business reports, sociological statistics, and technology usage—thereby assessing the model’s generalization ability across different contexts and terminologies.

##### C.3. Benchmark Evaluation Details

To ensure a rigorous and reproducible evaluation, we design a deterministic scoring mechanism for our TableVisBench. Instead of asking the MLLM to directly output a subjective

score (e.g., 1-10), we employ the MLLM as a Quality Assurance Analyst to identify and count specific errors based on strict definitions. The final scores are calculated deterministically from these counts. Below are the detailed calculation protocols for the four accuracy-based dimensions.

Data Accuracy (DA). This dimension measures the completeness and correctness of the data points rendered. The MLLM identifies the total number of data points in the source table (Ntotal) and counts the number of incorrect data points (Nerror) in the image (including missing values, wrong numbers, or incorrect legend mappings). The specific prompt is shown in Figure A6. The score is calculated as:

Ntotal − Nerror Ntotal

(A4)

SDA =

Text Rendering (TR). This dimension evaluates the character-level correctness of textual elements. The MLLM extracts all visible text strings from the image and identifies specific substrings or characters that contain errors (e.g., typos, garbled text). The specific prompt is shown in Figure A7. Let Ltotal be the total character length of all text in the image, and Lerror be the total character length of the identified incorrect text. The score is defined as:

Ltotal − Lerror Ltotal

STR =

(A5)

If no text is present (Ltotal = 0), the score is set to 0.

Relative Relationship (RR). This dimension assesses the visual proportionality of the infographic (e.g., whether bar heights or pie slice angles correspond to the data values). Similar to DA, the MLLM counts the total data points (Ntotal) and identifies the number of points (Nerror) that violate visual logic relative to other elements. The specific prompt is shown in Figure A8. The score is calculated as:

Ntotal − Nerror Ntotal

SRR =

(A6)

Additional information Accuracy (AA). This dimension evaluates contextual elements such as axes, ticks, and extra

Data Accuracy (DA)

system_prompt = r'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension. Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Data Accuracy**

- * **Focus:** Verify that every single data point from the source table is correctly represented in the image, and not missing, rendered as raw table data, or containing errors.
- * **Process:** For each row/data point in the table, verify its presence, rendering, and correctness in the image. This includes the numerical value, the associated label (e.g., category name), and its mapping to legends, series names, or colors. If the image only simply prints the raw table data without any data visualization or visual elements, you should consider this as a failure.
- * **Exclusion:** Do NOT consider the relative size/position in this step. Only focus on the existence and correctness of the data labels, values, and series mapping.
- **Mandatory Output Format:** You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure. ```json { "total_data_points": <integer>, // The total number of data points in the source table. "incorrect_data_points": <integer>, // The number of data points that are missing, have wrong values, or incorrect labels/legends in the image. "detailed_explain": "<string>" // Output your specific reason or detailed explain here. } ```’’’

user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Data Accuracy.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

- Figure A6. The system prompt and user prompt of the Data Accuracy dimension.

Text Rendering (TR)

system_prompt = '''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension. Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Text Rendering**

- * **Focus:** Checks the correctness of all textual elements in the image.
- * **Process:** First, identify and list ALL text visible in the image, including titles, annotations, labels, numbers, and legends. Then, compare this text against common knowledge and the source table to identify any errors (e.g., typos, misspellings, garbled characters, incorrect numbers). When listing incorrect texts, you should only list all error text characters in the image, rather than the whole string containing the error.
- **Mandatory Output Format:** You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure. ```json { "all_text_in_image": [ "<string>", // List all text strings found in the image. "<string>",

... ], "incorrect_text_in_image": [ "<string>", // List only the text characters that are incorrect (typos, wrong values, garbled characters, etc.).

...

] } ```’’’

user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Text Rendering.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

- Figure A7. The system prompt and user prompt of the Text Rendering dimension.

Table A2. Performance of Wan2.5-Preview on our TableVisBench.

Rewriting Generation Reflection Refinement DA TR RR AA AQ Score

Reference Image 97.7 99.5 86.4 96.6 4.2 84.4 - Wan2.5-T2I-Preview - - 62.8 82.5 62.8 56.5 4.8 62.5 Qwen3-8B* Wan2.5-T2I-Preview - - 73.3 92.9 70.5 79.7 4.4 72.1 Qwen3-8B* Wan2.5-T2I-Preview Gemini-2.5-pro Qwen-Image-Edit-2509 53.3 88.3 53.5 59.5 4.3 59.5 Qwen3-8B* Wan2.5-T2I-Preview Gemini-2.5-pro Qwen-Image-Edit-2509* 68.9 91.7 64.9 71.2 4.4 68.1 Qwen3-8B* Wan2.5-T2I-Preview Gemini-2.5-pro Wan2.5-I2I-Preview 76.9 91.1 74.3 74.3 4.4 72.1

annotations. The score is an average of up to three submetrics, depending on the elements present in the image:

Then Slbl = 1 − Perr.

2. Axis Alignment (Salign): If axes exist, the model checks the visual alignment of data points against axis ticks. Let Nmis be the number of misaligned points.

- 1. Label Logic (Slbl): If axis indicators exist, the model estimates the percentage of incorrect tick labels (Perr).

Relative Relationship (RR)

system_prompt = r'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension. Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Relative Relationship**

- * **Focus:** Checks if the visual proportions between data points are correct.
- * **Process:** For charts like bar, column, or line charts, a larger data value must correspond to a taller/longer bar or a higher point. For pie or donut charts, a larger value must correspond to a larger slice area. Check each data point against others to ensure its visual scale is logically correct relative to them.
- * **Notation:** If the image only simply prints the raw table data without any data visualization or visual elements, you should consider this as a failure, beacause the relative relationship cannot be checked.
- **Mandatory Output Format:** You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure. ```json { "total_data_points": <integer>, // The total number of data points in the source table. "incorrect_data_points": <integer>, // The number of data points whose visual size is incorrect relative to other data points. "detailed_explain": "<string>" // Output your specific reason or detailed explain here. } ```’’’

user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Relative Relationship.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

Figure A8. The system prompt and user prompt of the Relative Relationship dimension.

Additional information Accuracy (AA)

system_prompt = r'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension. Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Additional Information Accuracy**

- * **Focus:** Checks the correctness of non-data elements (not appearing in the topic and table) that provide context, such as annotations, axes, ticks, gridlines, and some unreadable marks (including some markdown marks, table delimiters, \n mark).
- * **Process:**
- * **Existence:** First, determine if any additional information (like a Y-axis or X-axis with ticks, lines, additional annotations and marks) is present.
- * **Axis Indicator/Label/Tick Logic:** If axis marks with scale indicators exist, check their labels. For a numerical axis, the tick mark labels must be logical and sequential (e.g., monotonically increasing: 0, 100, 200, 300, not 0, 200, 100, 300). Calculate the proportion of incorrect tick labels, and give the result ranged from 0 to 1 in the 'percentage_of_incorrect_indicator' field.
- * **Data-to-Axis Alignment:** If axis marks with scale indicators exist, for each data point from the table, verify its visual alignment with the axis ticks. Complete the 'total_data_points' and 'misaligned_data_points_vs_axis' field based on the following criteria:
- * An error occurs if a data mark (e.g., the top of a bar) is visually **above** a tick mark when its value is less than or equal to that tick's value (e.g., a bar for value 565 is drawn above the 600 tick).
- * An error occurs if a data mark is drawn **significantly lower** than its value suggests on the scale (e.g., a value of 565.8 is drawn down near the 520 level on an axis that goes up to 600).
- * **Additional Mark:** Except the label or tick above, if any additional annotations or marks (like markdown delimiters or \n) are present, judge whether they are appropriate to exist in the image. Give the percentage of how many of them are inappropriate to exist ranged from 0 to 1 in the 'percentage_of_inappropriate_mark' field.
- **Mandatory Output Format:** You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure. ```json { "detailed_explain": "<string>" // Output your specific reason or detailed explain here. "has_additional_indicator": <integer>, // 1 if elements like axes or gridlines with scale indicators exist, otherwise 0. "percentage_of_incorrect_indicator": <float>, // (Only if has_additional_indicator is True) The percentage of axis tick labels that are logically incorrect (e.g., not sequential). E.g., 0.25 for 1 out of 4 wrong labels. "total_data_points": <integer>, // (Only if has_additional_indicator is True) The total number of data points in the source table. "misaligned_data_points_vs_axis": <integer>, // (Only if has_additional_indicator is True) The number of data points from the table whose visual position is incorrectly aligned with the axis scale. "has_additional_mark": <integer>, // 1 if additional marks or annotations (not including the indicators above) exist, otherwise 0. "percentage_of_inappropriate_mark": <float>, // (Only if has_additional_mark is True) The level of inappropriate marks, ranged from 0 to 1.0. } ```’’’

user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Additional Information Accuracy.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

Figure A9. The system prompt and user prompt of the Additional information Accuracy dimension.

total−Nmis

Aesthetic Quality (AQ). Unlike the strictly factual dimensions above, this dimension evaluates the overall visual appeal, including layout harmony, color palette suitability, and typographic quality. Since aesthetic judgment relies on subjective perception rather than rule-based error counting, we employ a dedicated pre-trained aesthetic scoring model to quantitatively assess this dimension. The model provides a scalar score SAQ ranging from 0 to 10, reflecting the artistic quality of the infographic independent of its data fidelity.

Then Salign = N

Ntotal .

- 3. Artifact Appropriateness (Smark): If other marks (e.g., Markdown delimiters, random symbols) exist, the model estimates the percentage of inappropriate marks (Pinapp). Then Smark = 1 − Pinapp.

The specific prompt is shown in Figure A9. The final SAA is the arithmetic mean of the valid sub-metrics. If no additional information is present, this dimension is excluded from the calculation.

#### D. More Experiments

##### D.1. Results of Wan2.5-Preview

In the main paper, we primarily focused on open-source models to ensure reproducibility. Here, we extend our evaluation to the recently released Wan2.5-Preview series [51] to assess the ShowTable pipeline’s performance with stateof-the-art (SOTA) generation capabilities. The results are presented in Table A2.

Impact of rewriting. Consistent with our findings on opensource models, the rewriting module provides a substantial performance boost for Wan2.5-T2I-Preview. By converting the raw table into a reasoned visual plan (RW), the overall score increases significantly from 62.5 to 72.1 (+9.6). This improvement is particularly evident in Data Accuracy (62.8 → 73.3) and Text Rendering (82.5 → 92.9), confirming that our rewriting strategy is model-agnostic and effective even for top-tier generation models.

Analysis of refinement. The refinement stage reveals the critical importance of the editing model’s underlying capacity. When applying the open-source Qwen-Image-Edit2509 to refine images generated by the powerful Wan2.5T2I-Preview, we observe a performance degradation (Score 72.1 → 59.5) with the base editor. This is attributed to the significant capacity gap between the strong generator and the relatively weaker editor. Essentially, the editor struggles to maintain the high-fidelity details produced by the SOTA generator. However, our proposed RL training demonstrates clear effectiveness in this challenging scenario. Our trained model (Qwen-Image-Edit-2509*) significantly recovers performance compared to the base editor, raising the score from 59.5 to 68.1 (+8.6). Finally, when utilizing Wan2.5-I2I-Preview as the executor, matching the generator’s capacity, the pipeline achieves the highest performance in key structural metrics, with Data Accuracy improving further to 76.9 and Relative Relationship to 74.3. This underscores that while our training effectively boosts open-source editors, the ceiling of the refinement stage is ultimately determined by the base model’s capability.

##### D.2. More Visualization Results

Qualitative comparisons. To further qualitatively demonstrate the universality and effectiveness of our approach, we present a comprehensive case comparison across different base models in Figure A10. As evident in the first column of Figure A10, without the rewriting module, most base models fail to fundamentally grasp the visualization task. General unified models (e.g., Blip3o-Next, UniWorld-V1, OmniGen2) often suffer from severe hallucinations, producing chaotic layouts that lack any statistical meaning. Strong text-rendering models like Qwen-Image and Flux also fail to render the structure logits among data points. The second column (RW + gen) highlights the critical role of our

Table A3. User study results showing human preference rates (%) across five evaluation dimensions. Our full pipeline achieves the highest human preference in almost all dimensions, particularly dominating in Data Accuracy (DA).

Method DA TR RR AA AQ Qwen-Image 2.25 38.50 12.00 2.50 7.75 RW + Qwen-Image 24.00 29.25 41.25 48.25 44.50 RW + Qwen-Image + REF 73.75 32.25 46.75 49.25 47.75

rewriting module. By translating the data into a visual plan, all models successfully transition from unstructured chaos to a coherent infographic layout (specifically, a donut chart structure in this example), though most of them still suffer from heavy logical errors. Note that Wan2.5-Preview successfully generates a promising result, leading to no need of further refinement. The subsequent columns (REF round 13) demonstrate the power of the progressive self-correction loop. Visual inaccuracies, such as incorrect data segmentation, garbled text, and layout misalignments, are iteratively repaired. For example, in the Bagel and UniWorld-V1 rows, the text legibility and data mapping precision improve noticeably with each round. Additionally, the Wan2.5-Preview case (bottom row) showcases the efficiency of our pipeline; due to its high initial quality, the reflection module triggers the early-stopping mechanism (“Done”), avoiding unnecessary computation.

Detailed pipeline visualization. To provide a transparent view of the internal mechanisms of our ShowTable pipeline, we present a series of detailed case visualizations in the following figures. These figures explicitly display the stepby-step intermediate outputs, including the source table, the MLLM-generated rewriting prompt, the initial generation, and the iterative reflection instructions that guide the refinement. To demonstrate the pipeline’s versatility across different base models, the first three figures (Figure A11, Figure A12, Figure A13) utilize Qwen-Image as the generation module, while the subsequent two figures (Figure A14, Figure A15) employ Wan2.5-Preview. As shown in the examples, the visualization highlights the orchestrator’s ability to formulate precise geometric corrections (For example, “adjust the height of the bar to... resulting in a ratio of approximately 2.69:1” or “ensure the green ’Yes’ slice occupies exactly 81% (291.6◦)”), effectively guiding the executor to achieve high-fidelity data alignment.

##### D.3. User Study

To further validate the effectiveness of our proposed pipeline from a human perspective, we conducted a comprehensive user study. We randomly selected 20 table instances and generated visualizations using three configurations: the base generation model (Qwen-Image), the model prefixed with the rewriting module (RW+Qwen-Image), and our full

w/o RW RW + gen REF round1 REF round2 REF round3

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Flux

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Bagel

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Blip3o-Next

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

UniWorld-V1

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

OmniGen2

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Qwen-Image

[Figure 102]

[Figure 103]

Wan2.5-Preview

Done.

Figure A10. The qualitative comparison of different generation baselines of the same case on our proposed creative table visualization task. The first column presents the results without rewriting. The results via our ShowTable pipeline are shown from the second column.

ShowTable pipeline (RW + Qwen-Image + REF). We invited 20 human evaluators to independently assess the 20

sets of images. Participants were asked to vote for the best generation in each set across our five evaluation dimen-

Advertising Effectiveness by Medium

Survey Results: Yes vs No Responses

|Medium|Percentage|
|---|---|
|Magazine/Newspaper|35%|
|Billboard|13%|

|Response|Percentage|
|---|---|
|Yes|81%|
|No|19%|

Rewriting: A professional, ultra-detailed infographic visualizing survey results.

The centerpiece is a sleek, 3D isometric donut chart, presented on a clean, dark charcoal background with a subtle gradient.The chart is clearly divided into two segments to represent the data:- A dominant, large segment representing

**81%**, rendered in a vibrant, optimistic **teal green**.- A smaller, contrasting segment representing **19%**, rendered in a soft **coral red**.Elegant, bold white sans-serif typography is placed directly on or beside each segment, clearly displaying the percentages "81%" and "19%".The entire composition has a glossy, polished finish. The scene is illuminated by soft studio lighting, casting realistic soft shadows and creating subtle highlights on the chart's surface, enhancing its dimensionality. Minimalist icons float around their respective segments to add context: a dense cluster of glowing green checkmarks near the 81% slice, and a few sparse red 'X' marks near the 19% slice.Style: data visualization, corporate presentation, Behance HD, 4K, sharp focus, high contrast, modern UI design.

Rewriting: A stunning 3D infographic visualizing "Advertising Effectiveness by Medium," presented in a clean, minimalist, and professional corporate style. The composition is balanced on a light grey studio background with soft, diffused lighting.**On the left side:** A stylized, isometric 3D model of a stack of modern magazines next to a neatly folded newspaper. Emanating upwards from these publications is a thick, vibrant, glowing bar of bright blue light, representing high effectiveness. The bar reaches a significant height. Clearly legible, bold white text floats next to it, reading "**Magazine/Newspaper**" with the large number "**35%**" prominently displayed.**On the right side:** A sleek, miniature 3D model of a modern billboard. Emanating upwards from the billboard is a much shorter and thinner glowing bar of muted orange light. This bar is visually about one-third the height of the blue bar, accurately representing its lower effectiveness. Adjacent to this bar is clear, bold white text reading "**Billboard**" and the number "**13%**".**Overall Style:** The aesthetic is clean and sophisticated, with a focus on data clarity. The glowing bars have a subtle emissive quality, casting a soft light on the objects below. The entire scene is rendered with extreme detail and realism.**Keywords:** 3D render, data visualization, infographic, isometric perspective, minimalist, professional, corporate branding, octane render, trending on Behance, studio lighting, soft shadows, high detail, 4K, clean design, legible typography.

[Figure 104]

Reflection: Adjust the donut chart so that the green 'Yes' slice occupies exactly 81% of the circle (291.6°) and the red 'No' slice occupies exactly 19% of the circle (68.4°). Ensure the slice boundaries are repositioned to reflect these proportions accurately while keeping the existing colors, labels, and 3D style intact.

[Figure 105]

[Figure 106]

Reflection: Adjust the height of the Magazine/Newspaper blue bar so that it is proportionally accurate to the Billboard orange bar based on their labeled percentages (35% vs 13%), resulting in a height ratio of approximately 2.69:1. Ensure both bars retain their current colors, positions, and labels, but correct the geometric scaling to match the data.

Reflection: Adjust the donut chart so that the red 'No' segment occupies exactly 19% of the circle (about 68°) and the green 'Yes' segment fills the remaining 81% (about 292°). Keep the existing colors, labels, and 3D style consistent.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Done! ✅

Done! ✅

Figure A11. Some detailed pipeline visualizations. We use Qwen-Image here for base generation module with our pipeline.

sions. The preference rates are reported in ??. Consistent with our quantitative automated metrics, the full ShowTable pipeline significantly outperforms the baselines in human preference. Notably, ShowTable achieves an overwhelming 73.75% preference rate in Data Accuracy, confirming that our progressive reflection-refinement loop effectively corrects data-to-visual mapping errors that humans easily perceive. For Text Rendering (TR), the base model occasionally receives higher preference (38.50%); this aligns with our observation, where the base model often defaults to merely rendering the raw table text directly without executing complex layout reasoning or data visualization. While this “lazy rendering” avoids text artifacts, it completely fails

the core visualization task (reflected in its dismal 2.25% DA score). Overall, the human study strongly corroborates that our pipeline produces the most factually faithful and visually appealing infographics.

#### E. Limitations and Future Work

While our ShowTable pipeline demonstrates significant improvements in creative table visualization, there remain several limitations that open avenues for future research:

Full-pipeline training. First, our current work primarily explores the training of the rewriting and refinement modules. The generation and reflection modules still rely

Smartphone Ownership and AI Assistant Users

Advertiser Outlook and Growth in Paid Social Advertising Budgets

|Advertiser Outlook for Paid Social Advertising Budgets| |Paid Social Ad Budgets Will Grow By...| |
|---|---|---|---|
|Outlook|Percentage|Growth Range|Percentage|
|Increase|64%|1-10%|41%|
|Stay the Same|34%|11-20%|15%|
|Decrease|2%|21%+|11%|

|Number of smartphone owners|169 million|
|---|---|
|Number of smartphones with personal assistant|72 million|
|Number of monthly AI assistant users|71 million|

Rewriting: An infographic about smartphone ownership and AI assistant users, in a clean, modern, flat design style with a light blue background. The infographic should have three sections arranged horizontally, each in a white box. From left to right:1. A section with a simple icon of a smartphone inside a dark blue circle. Below the icon, the text reads "169 million people own a smartphone" in dark blue, with "169 million" in a large, bold, orange font.2. A section with a simple icon of a person's head inside a dark blue circle. Below the icon, the text reads "72 million smartphones have a personal assistant" in dark blue, with "72 million" in a large, bold, orange font.3. A section with a simple icon of a robot head inside a dark blue circle. Below the icon, the text reads "71 million people use an AI assistant monthly" in dark blue, with "71 million" in a large, bold, orange font.

Rewriting: A professional and visually engaging infographic titled "Advertiser Outlook: Paid Social Advertising Budgets". The design is clean, modern, and minimalist, with a clear two-part layout on a light gray background.**On the left side, under the heading "Advertiser Outlook":**A large, elegant donut chart visualizes the overall sentiment.* A dominant **64% segment** is colored in a **vibrant teal**, clearly labeled "**Increase**".* A second **34% segment** is a professional

**medium blue**, labeled "**Stay the Same**".* A tiny **2% sliver** is a neutral **dark gray**, labeled "**Decrease**".In the center of the donut, the number "**64%**" is displayed in a large, bold, dark font, highlighting the key statistic.**On the right side, under the heading "Paid Social Ad Budgets Will Grow By...":**A subtle arrow or flow line connects the "Increase" segment of the donut chart to this section. This area features three clean vertical bar charts of varying heights.* The tallest bar represents **41%**, labeled underneath as "**1-10% Growth**".* The middle bar represents **15%**, labeled "**11-20% Growth**".* The shortest bar represents **11%**, labeled "**21%+ Growth**".All three bars are rendered in the same vibrant teal as the "Increase" segment to maintain visual consistency.**Overall Style and Details:*** **Aesthetics:** Corporate style, data visualization, UI/UX design principles, high-quality vector art.* **Color Palette:** A sophisticated palette of vibrant teal, medium blue, dark gray, and a clean light gray background.*

[Figure 111]

**Typography:** Uses a clean, legible sans-serif font. Headings are bold, and all data labels and percentages are sharp and easy to read.* **Iconography:** Subtle, minimalist icons related to finance (upward arrow, dollar sign) and social media (megaphone) are tastefully placed in the corners or background to add context without cluttering the data.* **Quality:** High resolution, sharp focus, perfect composition, studio lighting.

Reflection: Replace the Hebrew word ' 'םעin the middle card's descriptive text with the English word 'with', so the text reads 'smartphones with a personal assistant'. Ensure font, size, and color match the existing style.

[Figure 112]

[Figure 113]

Reflection: Replace the text '7D million' in the second card (smartphones with a personal assistant) with '72 million' in the same bold orange font and style as the other numbers, ensuring alignment, spacing, and typography match the existing design.

Reflection: Correct the infographic as follows: In the pie chart, relabel the light blue segment from 'Increase' to 'Stay the Same' while keeping the proportions 64% (Increase), 34% (Stay the Same), and 2% (Decrease). In the bar chart, update the bars so that '1–10% Growth' shows 41%, '11–20% Growth' shows 15%, and '21%+ Growth' shows 11%. Adjust bar heights proportionally to reflect these corrected percentages and update the numeric labels accordingly.

[Figure 114]

[Figure 115]

Reflection: Correct the text on the middle card of the infographic so that it clearly reads 'smartphones with a personal assistant' instead of the incomplete phrase 'smartphones with a personal assistant with'. Keep the existing numerical value (72 million) and overall design, font, colors, and layout unchanged.

Reflection: Correct the infographic as follows: In the donut chart, change the inner label on the 64% segment from 'Stay the Same' to 'Increase' to match the table data. In the bar chart, swap the positions of the first two bars so that the tallest bar labeled 41% corresponds to '1–10% Growth' and the shorter bar labeled 15% corresponds to '11–20% Growth'. Ensure all colors and percentages remain consistent with the legend and table.

[Figure 116]

[Figure 117]

… (Max Rounds)

Reflection: Adjust the donut chart in the 'Advertiser Outlook' section so that the 'Increase' slice is reduced to accurately represent 64% (~230° of the circle), add the missing 'Stay the Same' slice at 34% (~122°) in the blue color per legend, and keep the small 'Decrease' slice at 2%. Ensure slice angles match percentages within ±2° and labels are correctly positioned. In the 'Paid Social Ad Budgets Will Grow By...' bar chart, change the first bar’s category label from '11–20% Growth' to '1–10% Growth', keeping the 41% value, and maintain proportional bar heights consistent with their percentages.

[Figure 118]

… (Max Rounds)

###### Figure A12. Some detailed pipeline visualizations. We use Qwen-Image here for base generation module with our pipeline.

How does social media popularity affect your life?

Comparison of Financial Year Values and Percentage Change

|Financial Year|Value|% Change|
|---|---|---|
|18/19|247.5| |
|19/20|211.5|-14.5%|

|Effect|Number|
|---|---|
|Limits attention span when completing assignments|75|
|Makes me proud to realize I am known online|25|
|I hide behind my social media|10|
|Decreases my mental health and creates toxicity|35|

Rewriting: A clean, minimalist infographic comparing financial data for two years. The infographic is split into two sections side-by-side on a white background.On the left, for the "18/19" financial year, there is a large circle with the value "247.5" inside. Below the circle is the text "18/19".On the right, for the "19/20" financial year, there is a smaller circle with the value "211.5" inside. Below the circle is the text "19/20".Between the two circles, there is a red speech bubble containing the text "-14.5%".The style is modern and uses a sans-serif font.

Rewriting: An infographic with the title "How does social media popularity affect your life?". The image features four items arranged horizontally on a light gray background. Each item consists of a colored circle with a gradient, a large number inside, and a line connecting the circle to a text description below.1. The first item on the left is a pink circle with the number "75" in large white text. The text below reads: "Limits attention span when completing assignments".2. The second item is a purple circle with the number "25" in large white text. The text below reads: "Makes me proud to realize I am known online".3. The third item is a teal circle with the number "10" in large white text. The text below reads: "I hide behind my social media".4. The fourth item on the right is an orange circle with the number "35" in large white text. The text below reads: "Decreases my mental health and creates toxicity".The overall style is clean, modern, and uses flat design. The text is in a clear, sans-serif font.

[Figure 119]

[Figure 120]

Reflection: Correct the geometric proportionality of the smaller circular arc on the right, which represents the value 211.5. Its length is currently too long compared to the arc for 247.5. Shorten the arc for 211.5 so its angle is approximately 85.5% of the larger arc's angle. This involves moving the arc's endpoint counter-clockwise by about 13.5 degrees, so it ends roughly halfway between the 3 and 4 o'clock positions instead of at the 4 o'clock position.

Reflection: Correct the geometric proportionality of the circles so their sizes

[Figure 121]

accurately represent the numbers they contain. Specifically, resize the orange circle (35) to be visibly larger than the purple circle (25). The final size order of the circles, from largest to smallest, must be: pink (75), orange (35), purple (25), and teal (10).

[Figure 122]

[Figure 123]

Done! ✅

[Figure 124]

Done! ✅

- Figure A13. Some detailed pipeline visualizations. We use Qwen-Image here for base generation module with our pipeline.

on off-the-shelf models. We believe that extending supervised fine-tuning (SFT) or reinforcement learning (RL) to all components of the pipeline could further enhance the system’s robustness and domain adaptability. Future work will investigate a more holistic training strategy to optimize the entire pipeline jointly.

refinement experiments, the upper bound of visualization fidelity is often determined by the precision of refinement module. We hope that our proposed task and benchmark will encourage the community to focus more on enhancing these foundational capabilities, particularly in precise fine-grained editing and logical reasoning, thereby advancing the field closer to more general and capable Artificial General Intelligence (AGI).

Towards a unified model. Second, the current implementation operates as a cascade of distinct models for rewriting, generation, reflection, and refinement. While effective, this multi-model approach increases deployment complexity and inference latency. A promising future direction is to explore a unified multi-modal architecture capable of performing all four sub-tasks end-to-end. Integrating these capabilities into a single model could significantly streamline the workflow and improve efficiency.

Dependency on foundation models. Finally, the performance of our pipeline is inevitably constrained by the capabilities of the underlying base models. As observed in our

Forecast of Mobile Communications Device Open OS Sales to End Users by OS (Millions of Units) from 2009 to 2014

Breakdown of Pension Pot

|Category|Amount|
|---|---|
|Some of your tax-free cash|£10,000|
|Equivalent sum to drawdown|£30,000|
|Remaining pension pot untouched|£60,000|

|Year|Symbian|Android|Research In Motion|iOS|Windows Phone|Other Operating Systems|
|---|---|---|---|---|---|---|
|2009|80|10|35|25|10|15|
|2010|115|40|50|45|15|25|
|2011|125|85|70|70|20|40|
|2014|260|250|110|120|70|60|

Rewriting: A pie chart infographic illustrating the "Breakdown of Pension Pot" on a white background.The chart is divided into three distinct slices:* A dark teal slice representing "Some of your tax-free cash (£10,000)", which is the smallest portion.* A light blue slice representing "Equivalent sum to drawdown (£30,000)".* A light green slice representing "Remaining pension pot untouched (£60,000)", which is the largest portion.To the right of the chart, there is a legend with colored squares corresponding to each slice:* A dark teal square next to the text "Some of your tax-free cash (£10,000)".* A light blue square next to the text "Equivalent sum to drawdown (£30,000)".* A light green square next to the text "Remaining pension pot untouched (£60,000)".The overall design should be clean, modern, and professional, using a sans-serif font. The labels should be clear and point to their respective slices.

Rewriting: A clustered bar chart titled "Forecast of Mobile Communications Device Open OS Sales to End Users by OS (Millions of Units)".The x-axis shows the years 2009, 2010, 2011, and 2014.The yaxis is labeled in millions of units, ranging from 0 to 300.For each year, there is a group of six bars representing different operating systems:- **2009:** Symbian (blue) at 80, Android (red) at 10, Research In Motion (green) at 35, iOS (purple) at 25, Windows Phone (orange) at 10, Other Operating Systems (light blue) at 15.- **2010:** Symbian (blue) at 115, Android (red) at 40, Research In Motion (green) at 50, iOS (purple) at 45, Windows Phone (orange) at 15, Other Operating Systems (light blue) at 25.- **2011:** Symbian (blue) at 125, Android (red) at 85, Research In Motion (green) at 70, iOS (purple) at 70, Windows Phone (orange) at 20, Other Operating Systems (light blue) at 40.- **2014:** Symbian (blue) at 260, Android (red) at 250, Research In Motion (green) at 110, iOS (purple) at 120, Windows Phone (orange) at 70, Other Operating Systems (light blue) at 60.Each bar has its corresponding numerical value displayed on top.A legend at the bottom indicates the color for each operating system: blue for Symbian, red for Android, green for Research In Motion, purple for iOS, orange for Windows Phone, and light blue for Other Operating Systems.The chart has a clean, professional design with a white background.

|[Figure 125]|
|---|

|[Figure 126]|
|---|

Reflection: Recreate the pie chart to consist of exactly three slices based on a total of £100,000. First, a dark teal slice for 'Some of your tax-free cash' at £10,000 (10% of the total area). Second, a medium blue slice for 'Equivalent sum to drawdown' at £30,000 (30% of the total area). Third, a light green slice for 'Remaining pension pot untouched' at £60,000 (60% of the total area). Remove the 'Equivalent ouxfree cash' category and its label entirely. Ensure all labels are correctly spelled and clearly point to their respective, proportionally accurate slices.

Reflection: Correct the bar chart to accurately reflect the provided data table. First, update the legend text from 'Research In Motim' to 'Research In Motion'. For the year 2009, adjust the 'Research In Motion' (green) bar and its label to a value of 35. For the year 2010, correct the following bars and their labels: Symbian (blue) to 115, Android (red) to 40, iOS (purple) to 45, Windows Phone (orange) to 15, and Other Operating Systems (light blue) to 25. For the year 2011, change the 'Research In Motion' (green) bar and label to 70, and the 'Windows Phone' (orange) bar and label to 20. Ensure all corrected bar heights are geometrically proportional to their new numeric values on the existing 0-300 Y-axis.

|[Figure 127]|
|---|

|[Figure 128]|
|---|

Reflection: Correct the proportions of the pie chart slices to accurately reflect the data. The total value is £100,000. The teal slice for 'Some of your tax-free cash' (£10,000) must be 10% of the circle (36 degrees). The blue slice for 'Equivalent sum to drawdown' (£30,000) must be 30% of the circle (108 degrees). The green slice for 'Remaining pension pot untouched' (£60,000) must be 60% of the circle (216 degrees), making it visibly more than half of the entire pie. Retain all original colors, labels, callout lines, and the title.

Reflection: Correct the bar chart to accurately represent the provided data. First, fix the legend: remove duplicate entries for 'iOS' and 'Windows Phone', and correct the typo 'Windoms Phone' to 'Windows Phone'. Then, adjust the bars and their data labels for each year to match the correct values. For 2009, change the Android (red) bar to 10 and the iOS (purple) bar to 25. For 2010, change the Research In Motion bar to be green with a value of 50, and the Windows Phone (orange) bar to 15. For 2011, change the Symbian (blue) bar to 125, the Android (red) bar to 85, and the Other Operating Systems (light blue) bar to 40. For 2014, change the Research In Motion bar to be green with a value of 110, the iOS (purple) bar to 120, and the Windows Phone (orange) bar to 70. Ensure all bar heights are geometrically proportional to their new values on the existing 0-300 Y-axis.

|[Figure 129]|
|---|

|[Figure 130]|
|---|

Reflection: Correct the proportions of the pie chart slices to accurately reflect the labeled monetary values. The total is £100,000. The green slice ('Remaining pension pot untouched', £60,000) must be 60% of the total circle (a 216-degree angle). The blue slice ('Equivalent sum to drawdown', £30,000) must be 30% of the circle (a 108-degree angle). The teal slice ('Some of your tax-free cash', £10,000) must be 10% of the circle (a 36-degree angle). Keep all colors, labels, title, and callout lines, adjusting the lines to point to the correctly resized slices.

Reflection: Correct the bar chart to accurately represent the provided data table. First, fix the legend to correctly map the colors: Symbian (blue), Android (red), Research In Motion (green), iOS (orange), Windows Phone (purple), and Other Operating Systems (light blue). Second, redraw all bars and update their data labels to match the correct table values for each year. For 2009, values are: Symbian 80, Android 10, RIM 35, iOS 25, Windows Phone 10, Other OS 15. For 2010: Symbian 115, Android 40, RIM 50, iOS 45, Windows Phone 15, Other OS 25. For 2011: Symbian 125, Android 85, RIM 70, iOS 70, Windows Phone 20, Other OS 40. For 2014: Symbian 260, Android 250, RIM 110, iOS 120, Windows Phone 70, Other OS 60. Ensure all required bars are present for each year and their heights are proportional to their value on the 0-300 Y-axis.

|[Figure 131]|
|---|

|[Figure 132]|
|---|

… (max Rounds)

… (max Rounds)

###### Figure A14. Some detailed pipeline visualizations with more complex and difficult cases. We use Wan2.5-Preview here for generation module and refinement module with our pipeline.

Vulnerability Severity Distribution and Scores

Survey Results: Yes vs No Responses

|Priority|Percent|
|---|---|
|Finding Effective Treatments or Cures for Diseases (like Cancer and Alzheimer's)|79%|
|Reducing Poverty and Hunger|61%|
|Improving Education and How Our Children Learn|56%|
|Solving Energy Problems (Renewable Sources, Efficiency)|55%|
|Improving Environmental Health and Sustainability|54%|
|Improving Public Health|53%|
|Improving the Lives of People with Disabilities|43%|
|Driving Economic Growth|35%|
|Developing Defense and Military Technology|32%|
|Improving Safety Devices (such as in Cars and Sports)|30%|
|Understanding Space and Our Place in the Universe|16%|

|Score Category|Percentage|
|---|---|
|Scores of 9 or higher|28%|
|Scores of 7 or higher|84%|
| | |
|Severity|Percentage|
|LOW|17.2%|
|MEDIUM|12.4%|
|HIGH|65.3%|
|CRITICAL|5.1%|

Rewriting: A professional and sleek infographic dashboard visualizing "Vulnerability Severity Distribution and Scores". The background is a dark, textured charcoal gray (#2c3e50), creating a modern, high-contrast look.**Main Feature: Donut Chart**On the left, the centerpiece is a large, clean, and vibrant donut chart representing the 'Severity' distribution. The chart is segmented with precise data labels:- **CRITICAL:** 5.1%, colored in a deep, alarming red (#e74c3c).- **HIGH:** 65.3%, colored in a bright, warning orange (#f39c12).- **MEDIUM:** 12.4%, colored in a cautionary amber yellow (#f1c40f).- **LOW:** 17.2%, colored in a calm, informative blue (#3498db).Each segment has a clean, white, sans-serif label indicating its name and percentage. In the hollow center of the donut chart, there is a minimalist, glowing white shield icon, reinforcing the security theme.**Side Panel: KPI Score Cards**To the right of the donut chart, there are two vertically stacked, modern KPI cards with rounded corners and a subtle glassmorphism effect.- **Top Card:** Features the large, bold number "28%" in white. Below it, in a smaller font, is the text "Scores of 9 or higher".- **Bottom Card:** Features the large, bold number "84%" in white. Below it, in a smaller font, is the text "Scores of 7 or higher".**Typography and Title**- A bold, clear title at the top of the image reads: **Vulnerability Severity Distribution and Scores**.- All text uses a clean, highly readable sans-serif font like Inter or Lato.**Overall Style**- **Aesthetics:** Data visualization, UI/UX design, corporate report, professional, minimalist, sleek, modern.- **Lighting:** Soft, subtle shadows are cast by the chart and KPI cards to give a sense of depth and dimension. The colored segments of the chart have a slight inner glow.- **Composition:** Balanced, uncluttered, and focused on data clarity.**Negative Prompts:**- Avoid: 3D pie charts, clutter, messy text, handwritten fonts, photographic elements, distracting background images, gradients.

Rewriting: An ultra-detailed, professional infographic data visualization titled "Public Priorities for Scientific Advancements".**Layout & Style:**The infographic features a clean, modern, vertical bar chart layout on a light gray background with a subtle grid pattern. The style is minimalist flat vector art with soft, subtle shadows to give elements depth.**Data Representation:**There are 11 horizontal bars, meticulously ordered from top to bottom based on percentage. The length of each bar is precisely proportional to its value.**Color Palette:**A sophisticated and harmonious color palette of cool blues, teals, and greens is used for the bars. The top priority, "Finding Effective Treatments or Cures for Diseases (79%)", is highlighted in a distinct and optimistic warm orange to make it stand out as the clear public priority.**Iconography:**To the left of each bar is a clean, minimalist, and universally understood icon representing its category:*

**Diseases (79%):** A DNA helix.* **Poverty & Hunger (61%):** A stalk of wheat.* **Education (56%):** A graduation cap.* **Energy (55%):** A wind turbine.* **Environment (54%):** A green leaf.* **Public Health (53%):** A stethoscope.* **Disabilities (43%):** The international symbol of access.* **Economic Growth (35%):** A rising arrow graph.* **Defense (32%):** A shield.* **Safety (30%):** A seatbelt.* **Space (16%):** A planet with rings.**Typography:**Crisp, highly legible sans-serif typography is used throughout. The main title is large and bold. Each bar is clearly labeled with its priority description, followed by its bolded percentage value (e.g., "Improving Education and How Our Children Learn - **56%**").**Overall Quality:**Award-winning graphic design, professional studio quality, sharp focus, high resolution, 8K. The final image should be suitable for a prestigious science journal or a modern museum exhibit.

[Figure 133]

[Figure 134]

Reflection: Remove the extraneous alphanumeric string 'f3512' and any distorted characters from the orange HIGH slice in the donut chart. Ensure only the correct 'HIGH 65.3%' label remains, clearly visible and unobstructed. Preserve all other chart elements, colors, slice proportions, and data labels exactly as they are.

Reflection: Replace the second bar's category text and value to match the table: change the label from 'Improving Education and How Our Children Learn – 56%' to 'Reducing Poverty and Hunger' and ensure the numeric value displayed is 61%. Adjust the bar length to visually match 61% proportion relative to the 79% bar above. Keep styling, color, and layout consistent with the rest of the chart.

[Figure 135]

[Figure 136]

[Figure 137]

Done! ✅

[Figure 138]

Done! ✅

###### Figure A15. Some detailed pipeline visualizations with more complex and difficult cases. We use Wan2.5-Preview here for generation module and refinement module with our pipeline.

