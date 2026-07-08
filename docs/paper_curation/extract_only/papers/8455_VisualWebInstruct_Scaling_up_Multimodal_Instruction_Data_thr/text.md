# arXiv:2503.10582v2[cs.CV]15Mar2025

## VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search

k,tYiming Jia*, nJiachen Li, mXiang Yue, zBo Li, xPing Nie, yKai Zou, kWenhu Chen kUniversity of Waterloo, tUniversity of Toronto, nUniversity of California, Santa Barbara, mCarnegie Mellon University, zNanyang Technological University, xIndependent, yNetmind.ai {yiming.jia@mail.utoronto.ca, wenhuchen@uwaterloo.ca} https://tiger-ai-lab.github.io/VisualWebInstruct

### Abstract

Vision-Language Models have made significant progress on many perception-focused tasks. However, their progress on reasoning-focused tasks remains limited due to the lack of high-quality and diverse training data. In this work, we aim to address the scarcity of reasoning-focused multimodal datasets. We propose VisualWebInstruct, a novel approach that leverages search engines to create a diverse and high-quality dataset spanning multiple disciplines, including mathematics, physics, finance, and chemistry, etc. Starting with a meticulously selected set of 30,000 seed images, we employ Google Image Search to identify websites containing similar images. We collect and process HTML data from over 700K unique URLs. Through a pipeline of content extraction, filtering, and synthesis, we construct a dataset of approximately 900K question-answer (QA) pairs, with 40% consisting of visual QA pairs and the remaining comprising text-based QA pairs. Models fine-tuned on VisualWebInstruct demonstrate significant performance improvements: (1) fine-tuning on Llava-OV results in 10-20 absolute points improvement across benchmarks, and (2) fine-tuning from MAmmoTH-VL yields a 5 absolute points gain across benchmarks. Our best model, MAmmoTH-VL2, achieves state-of-the-art performance within the 10B parameter class on MMMU-Pro (40.7), MathVerse (42.6), and DynaMath (55.7). These results highlight the effectiveness of our dataset in enhancing the reasoning capabilities of vision-language models for complex multimodal tasks.

### 1. Introduction

Vision-Language Models (VLMs), such as Llava [34] and Gemini [51], are designed to process multimodal inputs,

*Published during an internship at University of Waterloo

including images, videos, and text. While VLMs have recently demonstrated significant progress in straightforward perceptual tasks such as VQA [4], DocVQA [42], and VizWiz [21], they often struggle with more complex tasks such as MMMU [65], MathVista [39], and MEGABench [9], which require multi-step, deliberate reasoning [16, 56]. One major bottleneck for existing VLMs is the scarcity of reasoning-focused training datasets. Current multimodal reasoning datasets exhibit several limitations: (1) Many datasets, such as FigureQA [26], MapQA [7], GeoQA [8], and ChartQA [41], focus narrowly on specific types of scientific images. (2) Some datasets rely on synthetic images generated through predefined rules, such as CLEVR [25] and Geo170K [17], which often result in poor generalization to real-world visual reasoning tasks. (3) Other training datasets, such as AI2D [27] and ScienceQA [47], are relatively small and simplistic, primarily covering elementary-level visual knowledge. Due to these limitations, VLMs fail to acquire diverse reasoning skills, leading to slower progress on reasoning-intensive benchmarks compared to language models.

Given the difficulty of human annotation, we draw inspiration from WebInstruct [67] to mine naturally existing reasoning-focused instruction data from the internet. While WebInstruct retrieves reasoning-focused text data from Common Crawl1, their approach is infeasible for the multimodal domain due to two key challenges: (1) the lack of a comparable large-scale multimodal dataset, and (2) the unreliability of current multimodal retrieval models. To address these challenges, we leverage commercial web image search tools, such as Google Image Search, which offer high coverage and accuracy.

We begin by collecting approximately 30,000 seed images across multiple disciplines, including Accounting, Chemistry, Mathematics, and Physics. These images serve

1https://commoncrawl.org/

Dataset Size Source & Domains Coverage ScienceQA [47] 21K Elementary and high school science Science Q&A, diagrams, K-12 Exam IconQA [38] 107K Abstract diagrams and visual reasoning Visual reasoning, diagrams Geo170K [17] 170K Synthesized from LLMs Geometry CLEVR [25] 700K Synthesized from rules Shapes FigureQA [26] 1.3M Synthesized from rules Bar, Line, Pie ChartQA [41] 23K Charts from Staista, Pew, etc Charts Math360V [49] 260K FigureQA [26], CLEVR [25], IconQA [38], etc Math reasoning, diagrams Mulberry [63] 260K Geo3K [37], IconQA [38], ChartQA [41], ScienceQA [47], etc Geo, Figure, Medical, K-12 Exam Llava-CoT [60] 100K ChartQA [41], AI2D [27], GeoQA [8], CLEVR [25], etc Geo, General VQA, K-12 Exam VISUALWEBINSTRUCT 906K Internet (Homework Website, Forums, etc) All Above + College Exams

Table 1. Comparison between our dataset and the existing datasets. VISUALWEBINSTRUCT is the most diverse dataset with very broad coverage of disciplines and image types.

Table 1 compares VISUALWEBINSTRUCT with other datasets in terms of source and coverage. Our dataset comprises highly diverse, human-created scientific images spanning multiple disciplines and levels of complexity. Its broad coverage and increased difficulty make it particularly well-suited for improving VLM performance on realworld tasks requiring multi-step reasoning. To evaluate the effectiveness of VISUALWEBINSTRUCT, we perform supervised fine-tuning on MAmmoTH-VL [20] and LlavaOV-mid [29]. Comprehensive evaluations across seven visual reasoning benchmarks, including MMMU [65], MathVista [39], and Dyna-Math [73], demonstrate substantial performance gains. When fine-tuning LlavaOV-mid, we observe an absolute improvement of 10– 20 percentage points across these benchmarks. When fine-tuning MAmmoTH-VL, our model MAmmoTH-VL2 achieves state-of-the-art performance (within the 10B parameter range) on several benchmarks, including MMMUPro-std (40.7%), MMVet (64.5%), MathVerse (42.6%), and Dyna-Math (55.7%). MAmmoTH-VL2’s average performance across seven benchmarks surpasses strong competitors such as InternVL2.5 [10] and Phi-4-Mini [1], underscoring the effectiveness of VISUALWEBINSTRUCT in enhancing VLMs’ reasoning capabilities.

Scaling the Visual Reasoning Data Through Google Lens

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Extract

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Seed images Google Lens

Web pages VisualWebInstruct

Llava-CoT Qwen2-VL MAmmoTH-VL2

68.1

58.2

55.7

54.7

54.8

52.1

50.1

42.6

42.1

40.7

40.5

37

30.2

28.2

33

MMMU MMMU-PRO MATHVISTA MATHVERSE DYNA-MATH

- Figure 1. Overview of our automated data curation approach and major experimental results.

as queries for Google Image Search [70] to identify websites containing similar images. We then download the HTMLs from these websites and extract their accessibility trees, which are processed by an LLM to extract QA pairs (if any) for an initial dataset. However, we found that over half of the extracted questions lack annotated answers due to three primary reasons: (1) these websites do not provide answers, (2) some require membership to access, and (3) some necessitate user interaction to reveal the answers. To address this, we use GPT-4o [23] to synthesize multiple candidate solutions for each question, filtering for consistency among responses. Finally, we align the selected answers with the content from original webpage to remove potential inaccurate ones. Through this sophisticated process, we construct VISUALWEBINSTRUCT, a dataset containing approximately 900K QA pairs, where 40% are visual QA pairs associated with 163,743 unique images, while the remaining 60% are text-only QA pairs. Most of them are exam-like problems requiring deliberate reasoning.

Our contributions can be summarized as follows:

- • We propose a scalable pipeline for acquiring high-quality multimodal reasoning data from the internet, ensuring both scalability and quality.
- • We introduce VISUALWEBINSTRUCT, a diverse and comprehensive multimodal instruction dataset, which we will publicly release to the research community.
- • We develop MAmmoTH-VL2, a 7B-parameter visionlanguage model fine-tuned on VISUALWEBINSTRUCT, achieving state-of-the-art performance among models of comparable size and excelling in complex reasoning tasks requiring multi-step deliberation with visual context.

In the following sections, we will first talk about how we

mine the data from the Internet in section 2 and then talk about how to refine it in section 3. Finally, we show our experimental results in section 4.

### 2. Stage 1: Mining Data from the Internet

Our data mining pipeline follows a systematic approach to extract image-rich QA pairs from the internet. We begin with approximately 30K scientific images as seed data spanning multiple disciplines. We employ Google Image Search to identify visually similar content, gathering 758,490 unique URLs. After filtering out irrelevant domains, we construct accessibility trees for the relevant websites to extract meaningful content, preserving both textual and visual information while eliminating non-essential elements. We then leverage the Gemini 1.5 Flash model in a two-stage process: first to automatically extract QA pairs from the accessibility trees and then to filter these pairs based on comprehensive quality criteria, including question validity and image relevance, ensuring the educational value and integrity of the final dataset.

#### 2.1. Seed Data collecting

Due to the limited availability of image-rich QA datasets and the predominant focus on mathematics in existing datasets, creating a comprehensive QA dataset that incorporates diverse subjects and abundant visual content is essential. Our seed dataset consists of approximately 30,000 images, which were crawled from Stemez2 in compliance with copyright regulations. These images span multiple disciplines, including mathematics, physics, accounting, chemistry, engineering, and biology, ensuring both subject diversity and visual richness.

#### 2.2. Google Image Searching

Using the seed images, we conducted Google Image searches to find visually similar content across the web. Leveraging Google Lens (Figure 3), we collected approximately 60 URLs per image, resulting in a total of 1,747,634 URLs containing visually similar content. Many websites with non-permissive licenses implement anti-crawling mechanisms, and we ensured compliance by avoiding data collection from such sources. We applied rigorous deduplication and filtering, removing URLs from domains unlikely to contain educational content (e.g., video platforms and image repositories). This refinement yielded 758,490 unique, high-quality URLs for further processing. By using images as primary search keys, we ensured strong visual and contextual connections between the collected data and our seed dataset, effectively preserving the original distribution while significantly expanding its coverage.

2https://stemez.com/subjects/science/

#### 2.3. Accessibility Tree Building

After filtering out irrelevant domains, we processed the HTML content of each remaining URL to construct accessibility trees that capture essential textual and visual information. As illustrated in Figure 4, our implementation focuses on extracting meaningful text content and image elements while filtering out non-essential components such as navigation menus, advertisements, and auxiliary elements. We developed a tree-based structure where each node represents either textual content or an image, preserving the hierarchical relationships present in the original HTML while removing unnecessary markup and styling information. The resulting accessibility trees provide a clean, hierarchical representation of each webpage’s content, making subsequent QA pair extraction more efficient and reliable.

#### 2.4. QA Pairs Extraction

After constructing accessibility trees, we prompt the Gemini 1.5 Flash model to identify and extract high-quality QA pairs from webpage content. We designed a structured prompt instructing the model to extract complete question text, identify relevant question-related images, and extract comprehensive solution details while preserving mathematical notations and step-by-step explanations. This approach maintains the educational integrity of the extracted content by preserving its original formatting, mathematical expressions, and logical structure, ensuring technical accuracy throughout the extraction process. Through this method, we extracted a total of 421,320 raw QA pairs from the webpages, with approximately 60% containing images.

We then implemented a post-processing stage using the Gemini 1.5 Flash model to ensure dataset quality by evaluating both textual content and images. Our evaluation framework assessed two key criteria: question validity and meaningfulness, as well as the relevance and clarity of question-related images. By prompting Gemini to verify whether images are properly referenced, clear, visible, and contribute to understanding the question, we established strict validation criteria for retaining QA pairs.

This post-processing step significantly improved dataset quality by removing incomplete, unclear, or irrelevant content while preserving educational integrity and effectiveness. Our analysis shows that out of 421,320 processed pairs, 361,015 (85.7%) were valid, while 60,305 were filtered out as invalid. Similarly, out of 449,859 total images processed, 331,818 (73.76%) were deemed valid and relevant to their corresponding questions.

### 3. Stage 2: Dataset Refinement

After Stage 1, we obtain a large amount of raw data from the Internet. However, this data contains a notable level of noise. For instance, more than half of the questions lack cor-

Seed Image Webpage Accessibility Tree Initial Dataset

Post-Process

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

- Stage 1:
- Stage 2:

[Figure 16]

[Figure 17]

VisualWebInstruct Verified Answer

Consistency Filtering

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Y/F

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- Figure 2. Comprehensive Pipeline for VISUALWEBINSTRUCT Dataset Generation. The workflow illustrates our multi-stage approach for creating high-quality multimodal instruction data. Stage 1: starting with seed images, we leverage Google Image search to identify relevant webpages, which are processed into accessibility trees. The raw QA pairs are extracted from the trees and refined through a post-processing step to ensure the vadality the data. Stage 2: we first generat multiple synthesized answers for consistency filtering, then align these with original web-sourced content to enhance the accuracy of the answers.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

|[Figure 37]|
|---|

|[Figure 38]|
|---|
|[Figure 39]|

[Figure 40]

[Figure 41]

Query

- Figure 3. Example of Google Lens search functionality for circle geometry problems.

[Figure 42]

Text

Question: In the following figure...

[Figure 43]

Image

Root

Text

Option: A: 75, B: 60, C: 30, D: 45

responding answers due to various issues, such as (1) membership requirements, (2) interaction requirements, and (3) the absence of an answer. Thus, a second round of refinement is necessary to further improve the dataset quality.

Text Answer: Chord AB subtends...

Figure 4. Example of an accessibility tree structure extracted from an educational website.

#### 3.1. Answer Refinement

erate four different answer variations. This approach allowed us to obtain multiple perspectives on each question. Next, we employed GPT-4o as an LLM judge to determine whether the synthesized responses aligned with each other. As illustrated in Figure 5, we evaluated whether the conclusions were mutually consistent across these responses. This evaluation was particularly important for questions in domains such as mathematics and physics, where precision and correctness are paramount. Only when more than half of the synthesized responses demonstrated consistency did we retain the question along with the consistent responses.

We implemented a comprehensive refinement process to ensure consistency and quality in our dataset. This step was critical in addressing potential variations or inconsistencies in the extracted answers, thereby creating a high-fidelity dataset for model training.

Our refinement methodology leveraged GPT-4o’s capabilities in a two-stage process. First, for each question and its associated images, we prompted GPT-4o [23]3 to gen-

3We compared GPT-4o and Gemini-1.5 and found that GPT-4o’s outputs were significantly more reliable. Therefore, we adopted GPT-4o.

This rigorous consistency check served as an additional quality filter, ensuring that our dataset contained highly accurate and unambiguous answers that could be reliably used for model training.

sion through two main stages. Our approach effectively scaled the initial 30,000 seed images into a comprehensive multimodal instruction dataset containing 900K instruction data. The final dataset includes 347,313 image-associated QA pairs (approximately 38% of the total) supported by 163,743 unique images.

Through this refinement process, we successfully created a dataset in which all responses were systematically generated by GPT-4o, ensuring a consistent style and level of quality throughout the collection. The resulting dataset comprises 1.04 million QA pairs spanning multiple disciplines, representing one of the largest collections of consistency-verified multimodal instruction data available.

We also conducted thorough decontamination checking to ensure our training dataset does not contain any data from the evaluation benchmarks, thereby maintaining the integrity of our experimental results.

Category Percentage Num of QA Pairs Math 62.50% 566K Physics 14.50% 132K Finance 7.25% 66K Chemistry 4.80% 43K Engineering 4.35% 39K Others 6.60% 60K

#### 3.2. Answer Alignment

The final step in our quality assurance process involved answer alignment to further enhance accuracy. While the previous refinement step generated consistent answers using GPT-4o, we recognized the importance of validating these against authoritative content from the original web sources.

Synthesized Answera

Output

Table 2. Distribution of Categories in VISUALWEBINSTRUCT

Is the majority of the answers consistent?

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Yes!

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

### 4. Experiments

[Figure 53]

[Figure 54]

We detail the training and evaluation details of our experiments in this section.

No

[Figure 55]

[Figure 56]

[Figure 57]

LLM Judge

Nothing Here

[Figure 58]

[Figure 59]

[Figure 60]

#### 4.1. Experimental Setup

For our experiments, we directly fine-tuned an existing MAmmoTH-VL checkpoint on our VISUALWEBINSTRUCT dataset. We refer to our resulting model as MAmmoTH-VL2. The architecture consists of a language tower based on Qwen2.5-7B-Instruct [62], a vision tower using SigLip [68], and a projector module connecting these components, following Llava-OneVision [29, 34].

Figure 5. Illustration of our consistency checking methodology.

In this step, we used Gemini-2.0-Flash to measure the alignment between GPT-generated responses and the original extracted answers, if available. In cases where the comparison indicated inconsistency, we preserved the original web-sourced answer. Conversely, when the Gemini model determined strong alignment between the generated and web-sourced answers, we retained the GPT-generated version. Through this alignment process, we combined the consistency of model-generated content with the authority of original educational materials in a balanced manner.

To enhance data diversity, we employed a data mixing strategy that combined our VISUALWEBINSTRUCT dataset with modified LLaVA-CoT [60] (with CoT prompting tags removed) in a 9:1 ratio, resulting in approximately 900K samples from VISUALWEBINSTRUCT and 100K samples from the modified LLaVA-CoT dataset. This mixing strategy empirically improved our model’s performance across diverse visual reasoning tasks.

#### 3.3. Dataset Statistics

The statistics presented in Table 2 illustrate the distribution of knowledge domains in our dataset, VISUALWEBINSTRUCT. While the major categories are shown in the table, the ”Others” category (6.60%) comprises General Knowledge (2.45%), Computer Science (2.25%), Biology (1.40%), and humanities subjects, including Language/Literature (0.25%), Social Sciences (0.20%), and Arts (0.05%). This distribution reflects the dataset’s strong quantitative orientation while ensuring sufficient breadth.

We employed a supervised fine-tuning (SFT) approach with a batch size of 256. The learning rate was set to 1 × 10−5 for the language model and projector components, while the vision encoder was fine-tuned with a lower rate of 2 × 10−6 to preserve its pre-trained visual recognition capabilities. The model was trained for a single epoch, which proved sufficient given the high quality and diversity of our dataset. Input images were processed at a resolution of 384 × 384 with appropriate adjustments for varied aspect ratios. We limited input sequences to a maximum

- Table 3 summarizes the statistics after each step of the VISUALWEBINSTRUCT pipeline, showing the data progres-

Processing Stage Total QA Pairs Image-Associated QA Unique Questions Total Images Unique Images Stage 1: Mining Data from the Internet

Seed Data Collection - - - 30,000 30,000 QA Pairs Extraction 421,320 248,643 421,320 552,269 362,728 Post-Processing 361,015 159,059 361,015 331,818 212,530

Stage 2: Dataset Refinement

Answer Refinement 1,041,598 407,218 257,201 577,455 167,493 Answer Alignment 906,160 347,313 257,201 475,099 163,743

Table 3. Statistics of different milestones in the data processing pipeline of VISUALWEBINSTRUCT.

Model Size MMMU MMMU-Pro MMMU-Pro MathVista MMVet MathVerse Dyna-Math Avg

val standard vision testmini test testmini test Closed-sourced Models

GPT-4o - 69.1 54.0 49.7 63.8 76.2 50.2 63.7 61.0 Gemini-1.5-Pro - 59.1 49.4 65.8 63.9 64.0 41.2 64.8 58.3 Claude-3.5-Sonnet - 68.3 55.0 48.0 67.7 75.4 44.2 60.5 59.9

Open-source General Vision-Language Models

Molmo 8B 45.3 28.3 18.9 51.6 58.0 18.9 41.6 37.5 Llava-OV 7B 48.8 29.5 18.7 63.2 58.6 26.2 40.3 40.8 Llama-3.2-Inst 11B 50.7 33.0 23.7 51.5 59.3 31.6 40.5 41.5 Qwen2-VL 7B 52.1 37.0 26.9 58.2 62.0 28.2 42.1 43.8 MAmmoTH-VL 7B 50.8 33.2 25.3 66.0 62.3 34.2 44.7 45.2 InternVL2.5 7B 55.8 38.2 30.4 64.4 62.8 39.5 49.8 48.7 Phi-4-mini 5.6B 55.1 39.7 31.2 62.4 60.5 37.6 51.4 48.6 DeepSeek-VL2 27B 51.1 31.4 24.3 62.8 - - - Llava-CoT-L 11B 50.1 31.6 20.4 54.8 60.3 30.2 44.8 41.7 Llava-CoT-M 7B 51.4 33.0 23.7 63.8 58.6 39.4 48.3 45.5 LlamaV-o1 11B 49.1 31.5 22.4 54.4 63.6 - - Mulberry 7B 55.0 36.8 23.6 63.1 60.9 31.0 45.1 45.0 Insight-V 8B 50.2 30.7 20.5 59.9 60.8 28.7 47.8 42.6 MM-Eureka 8B 49.2 - - 67.1 60.7 40.4 - -

MAmmoTH-VL2 7B 54.7 40.7 26.3 68.1 64.5 42.6 55.7 50.4 ∆ over SoTA -1.1 +1.0 -4.9 +2.1 +0.9 +3.1 +4.3 +1.7

- Table 4. Evaluation Results of our model and other baseline models. Most of the baseline results are taken from other papers. The best and second-best results across all open-source models are highlighted in bold and underlined, respectively.

of 8,192 tokens to accommodate detailed reasoning chains while maintaining computational efficiency.

This fine-tuning approach enabled MAmmoTH-VL2 to leverage the strong multimodal reasoning foundation of MAmmoTH-VL while enhancing its performance on our targeted visual reasoning tasks that require multi-step deliberation with visual context.

#### 4.2. Evaluation Setup

To assess the capabilities of MAmmoTH-VL2, we conducted a comprehensive evaluation across multiple multimodal benchmarks that specifically test visual reasoning and knowledge application. Our evaluation framework focuses on benchmarks that require complex reasoning with visual context. We evaluate our model on seven key bench-

marks that collectively provide a comprehensive assessment of multimodal reasoning capabilities:

- • MMMU [65]: Tests multimodal understanding across university-level domains, requiring integration of visual and textual information.
- • MMMU-Pro [66]: Advanced versions of MMMU with more challenging problems and more distractor options that require sophisticated visual reasoning.
- • MathVista [39]: Evaluates mathematical reasoning with visual inputs, testing the model’s ability to process visual information for solving complex math problems.
- • MMVet [64]: Assesses general multimodal understanding across diverse tasks and contexts.
- • MathVerse [71]: Focuses on mathematical reasoning with visual components and relies less on text hints, re-

quiring complex visual reasoning.

• Dynamath [73]: Tests dynamic mathematical reasoning capabilities with visual context.

For all evaluations, we used greedy decoding in a zero-shot setting to ensure fair comparison with existing models. We categorize the comparison models into three groups:

- • Closed-source Models: GPT-4o [23], Gemini-1.5Pro [51], Claude-3.5-Sonnet [3].
- • Open-source Vision-Language Models: Molmo [14], LLaVA-OV [29], Llama-3.2 [19], Qwen2-VL [55], MAmmoTH-VL [20], InternVL2.5 [10], Phi-4-mini [1], DeepSeek-VL2 [58].
- • Reasoning Vision-Language Models: SFT models like Llava-CoT-L (from Llama-3.2) and Llava-CoT-M (from MAmmoTH-VL) [60], LLama-V-o1 [53], Mulberry [63], Insight-V [15]. We include a recent work MMEureka [44] trained with RL.

To ensure standardized and reproducible evaluations, we employed LMMsEval [69], a comprehensive evaluation framework for multimodal language models. For all evaluations, we used greedy decoding in a zero-shot setting to ensure fair comparison with existing models. Our approach allows for direct comparison with models of comparable size, providing insights into the value of the VISUALWEBINSTRUCT dataset. Performance is reported using accuracy scores for each benchmark, with an average score across all benchmarks to indicate overall model capability.

#### 4.3. Experimental Results

Here we evaluate our results from different perspectives.

Quantitative Results The table 4 presents the performance of MAmmoTH-VL2 compared to various multimodal models across seven benchmarks. Our analysis reveals several important findings regarding the effectiveness of models fine-tuned on VISUALWEBINSTRUCT.

Overall Performance. MAmmoTH-VL2 achieves an average accuracy of 50.4% across all benchmarks, outperforming other open-source vision-language models of comparable size (7B-11B parameters). This represents a significant improvement over standard vision-language models like Qwen2-VL (43.8%), LLaVA-OV (40.8%), and Molmo (37.5%). It even beats the very recent model like InternVL2.5 [10] and Phi-4-mini-Multimodal [1].

Mathematical Reasoning Capabilities. MAmmoTH-VL2 demonstrates particularly strong performance on mathematical reasoning tasks. On MathVista, our model achieves 68.1% accuracy, surpassing all the open-source and closedsource models. The model’s performance on MathVerse (42.6%) and Dyna-Math (55.7%) further confirms its enhanced capability for visual reasoning.

Complex Reasoning Tasks. On MMMU-Pro-std with 10 options, MAmmoTH-VL2 achieves 40.7% accuracy, showing a significant improvement over other 7B models such as

LLaVA-OV (29.5%) and Qwen2-VL (37.0%). This demonstrates that our approach effectively enhances the model’s ability to perform complex reasoning across diverse domains beyond mathematics.

Gap with Larger and Closed-Source Models. While MAmmoTH-VL2 outperforms open-source models of comparable size, there remains a gap with closed-source models such as GPT-4o, Gemini-1.5-Pro, and Claude-3.5-Sonnet. This indicates potential for further improvements through scaling or enhanced training methodologies.

Comparison with Reasoning-Enhanced Models. Among the reasoning-enhanced vision-language models like LlavaCoT, Mulberry [63], LlamaV-o1 [53] and Insight-V [15], MAmmoTH-VL2 demonstrates competitive performance, achieving results comparable to or better than specialized models like LLaVA-CoT and Mulberry. For instance, on MMMU-Pro Vision, our model achieves 26.3% accuracy, outperforming LLaVA-CoTM’s 23.7%. Notably, other reasoning-enhanced models often utilize complex methodologies in either the training or inference stage to enhance their chain-of-thought abilities, which makes the development process and deployment more complicated. In contrast, MAmmoTH-VL2 achieves much better reasoning capabilities through our straightforward fine-tuning approach on VISUALWEBINSTRUCT, offering a simpler yet effective solution compared to the other approaches.

These results confirm that fine-tuning on VISUALWEBINSTRUCT significantly enhances the model’s reasoning capabilities. The consistent performance improvements across diverse benchmarks from non math-related and math-related domains demonstrate the effectiveness of our approach in developing more capable multimodal reasoning models. We believe our dataset can be utilized to augment future vision-language models.

#### 4.4. Ablation Study

The ablation study in Table 5 demonstrates the impact of different training datasets and their combinations on model performance across multiple visual reasoning benchmarks. Two base models were evaluated: Llava-OV-mid and MAmmoTH-VL.

For Llava-OV-mid, the baseline starts at 26.3% average score across benchmarks. Training with Llava-CoT data improves this to 33.6%, while training on VISUALWEBINSTRUCT yields an even better 38.1%, with with MMVet performance notably jumping from 32.1% to 57.6%. The combined training approach (VISUALWEBINSTRUCT +LlavaCoT) achieves the best overall performance at 39.7%.

The stronger MAmmoTH-VL model begins with an average score of 45.4%. Training with VISUALWEBINSTRUCT improves the average to 49.0%, showing gains across multiple benchmarks, particularly in MMMU-Pro vision and Dyna-Math tests. As with Llava-OV-mid, the com-

Training Data MMMU MMMU-Pro MMMU-Pro MathVista MMVet MathVerse Dyna-Math Avg

val standard vision testmini test testmini test Training from LLava-OV-mid

- - 40.1 21.2 12.2 36.0 32.1 18.1 24.4 26.3 Llava-CoT 40.8 25.8 14.6 45.7 47.5 27.2 33.9 33.6 Ours 45.3 31.5 20.9 43.9 57.6 27.4 40.3 38.1 Ours+Llava-CoT 47.6 31.6 20.9 48.8 51.7 34.9 42.3 39.7

Training from MAmmoTH-VL

- - 50.8 34.8 25.3 66.0 62.3 34.2 44.7 45.4 Llava-CoT 51.4 35.2 24.6 63.8 58.7 39.4 48.3 45.9 Ours 52.6 38.6 29.0 65.9 61.8 39.4 55.7 49.0 Ours+Llava-CoT 54.7 40.7 26.3 68.1 64.5 42.6 55.7 50.4

- Table 5. Ablation Results of our experiments. We show experimental results from different backbones to show the impact of consistency filtering and data mixing with Llava-CoT. For each base model, the best performance is highlighted in bold.

bined training approach works best, reaching 50.4% average score, with notable improvements in MMMU (54.7%), MMMU-Pro standard (40.7%), and Dyna-Math (55.7%).

The key findings indicate strong data complementarity between VISUALWEBINSTRUCT and Llava-CoT, with their combination consistently delivering the best results. We also observe that weaker base models show larger relative improvements from training. Overall, the ablation study confirms that our VISUALWEBINSTRUCT dataset significantly boosts model performance across all benchmarks, demonstrating its effectiveness in enhancing visual reasoning capabilities regardless of the base model.

### 5. Related Works

#### 5.1. Multimodal Instruction Data

Creating high-quality multimodal datasets remains a significant challenge in advancing MLLMs. Current approaches face critical limitations, particularly in balancing quality and scale. Human-annotated datasets provide high-precision, contextually appropriate data [14, 43, 50, 61] but suffer from prohibitive costs and scalability constraints. Meanwhile, methods leveraging existing academic datasets [35, 54] offer more cost-effective alternatives but lack the diversity and reasoning complexity needed for advanced multimodal reasoning tasks. This limitation is particularly evident in the scarcity of large-scale, reasoningfocused multimodal datasets that can be efficiently produced. Our work addresses these challenges by proposing a novel, scalable methodology for constructing multimodal instruction datasets that maintain both the quality and reasoning complexity.

#### 5.2. Multimodal Large Language Models

Multimodal Large Language Models (MLLMs) have advanced AI by integrating text and visual processing capa-

bilities. While proprietary models such as GPT-4o [23] and Gemini [51, 52] achieve state-of-the-art performance, they remain inaccessible to the broader research community. To address this gap, connector-based approaches [13, 31] have emerged, linking visual encoders to language models through lightweight projection modules.

Recent open-source MLLMs, such as LLAMA [19], LLaVA [28, 33], MiniGPT-4 [72], and Deepseek-VL [36], have contributed to advancements in vision-language understanding. Additionally, Qwen-VL [55] and InternVL [10] have demonstrated strong performance through efficient design and diverse pre-training.

Meanwhile, various approaches have been developed to enhance MLLM reasoning capabilities, including neural symbolic methods [2, 11], optimized visual encoding strategies [24, 32], plan-based prompting [40, 45], structured reasoning frameworks [60], and sequential instruction tuning [22]. Despite these advancements, these models face a critical challenge: the scarcity of publicly available large-scale visual reasoning datasets necessary for enhancing model reasoning capabilities [5]. Our work addresses this supervised fine-tuning data bottleneck while building on the connector-training paradigm, aiming to bridge the gap between proprietary and open-source multimodal models to foster more accessible vision-language systems.

#### 5.3. Chain-of-Thought in Large Language Models

Chain-of-Thought (CoT) prompting [57] has revolutionized how large language models tackle complex reasoning challenges. This technique enables LLMs to navigate difficult problems—including commonsense scenarios [18, 48] and logical puzzles [30, 59]—by following explicit reasoning pathways. At its core, CoT methodically decomposes complex questions into manageable sequential steps, creating a structured framework that guides models toward systematic solutions [12]. Evidence consistently

demonstrates significant improvements in reasoning performance through this approach. Notable advancements include Prism [46], which implements a distinctive dual-stage architecture that separates initial perception from subsequent reasoning operations, and MSG [6], which pioneered the forced Chain-of-Thought methodology—establishing a foundational paradigm shift in structured prompting approaches that continues to shape current research.

### 6. Conclusion

In this paper, we explore the possibility of constructing large-scale multimodal reasoning datasets without relying on human annotation. We are the first paper to utilize Google Image Search for mining high-quality visual reasoning dataset. Our approach has been highly effective to achieve state-of-the-art performance on 5 out of 7 evaluated benchmarks. In the future, we plan to work on multiple round of search to further expand the dataset size.

### Acknowledgement

This research was supported by NetMind.Ai for providing cloud compute. Also, we also want to thank Google DeepMind for generous support for Gemini credits. A large part of our data processing pipeline is benefited from the credits.

### References

- [1] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, YenChun Chen, Yi ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, and Xiren Zhou. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras, 2025. 2, 7
- [2] Saeed Amizadeh, Hamid Palangi, Oleksandr Polozov, Yichen Huang, and Kazuhito Koishida. Neuro-symbolic visual reasoning: Disentangling ”visual” from ”reasoning”,

2020. 8

- [3] Anthropic. Claude 3.5 sonnet model card addendum, 2024. 7

- [4] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 1
- [5] Tianyi Bai, Hao Liang, Binwang Wan, Yanran Xu, Xi Li, Shiyu Li, Ling Yang, Bozhou Li, Yifan Wang, Bin Cui, Ping Huang, Jiulong Shan, Conghui He, Binhang Yuan, and Wentao Zhang. A survey of multimodal large language model from a data-centric perspective, 2024. 8
- [6] Franz Louis Cesista. Multimodal structured generation: Cvpr’s 2nd mmfm challenge technical report, 2025. 9
- [7] Shuaichen Chang, David Palzer, Jialin Li, Eric FoslerLussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps. arXiv preprint arXiv:2211.08545, 2022. 1
- [8] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning, 2022. 1, 2
- [9] Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, et al. Mega-bench: Scaling multimodal evaluation to over 500 real-world tasks. arXiv preprint arXiv:2410.10563, 2024. 1
- [10] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 2, 7, 8
- [11] Minkyu Choi, Harsh Goel, Mohammad Omama, Yunhao Yang, Sahil Shah, and Sandeep Chinchali. Towards NeuroSymbolic Video Understanding, page 220–236. Springer Nature Switzerland, 2024. 8
- [12] Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future, 2024. 8
- [13] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 8

- [14] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris CallisonBurch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna,

- Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models, 2024. 7, 8
- [15] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. arXiv preprint arXiv:2411.14432, 2024. 7
- [16] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems, 36:70757–70798, 2023. 1
- [17] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-modal large language model, 2023. 1, 2
- [18] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies, 2021. 8
- [19] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 7, 8
- [20] Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. arXiv preprint arXiv:2412.05237,

2024. 2, 7

- [21] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617,

2018. 1

- [22] Hanxu Hu, Simon Yu, Pinzhen Chen, and Edoardo M. Ponti. Fine-tuning large language models with sequential instructions, 2024. 8
- [23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2, 4, 7, 8
- [24] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding, 2024. 8
- [25] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 1, 2
- [26] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos´ K´ad´ar, Adam Trischler, and Yoshua Bengio. Fig-

- ureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017. 1, 2
- [27] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images, 2016. 1, 2
- [28] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 8
- [29] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 5, 7
- [30] Hao Li, Jinfa Huang, Peng Jin, Guoli Song, Qi Wu, and Jie Chen. Weakly-supervised 3d spatial reasoning for text-based visual question answering. IEEE Transactions on Image Processing, 32:3367–3382, 2023. 8
- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. 8
- [32] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm, 2024. 8
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 8
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1, 5
- [35] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too, 2023. 8
- [36] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world visionlanguage understanding, 2024. 8
- [37] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021. 2
- [38] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021. 2
- [39] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 1, 2, 6
- [40] Bozhi Luan, Hao Feng, Hong Chen, Yonghui Wang, Wengang Zhou, and Houqiang Li. Textcot: Zoom in for enhanced multimodal text-rich image understanding, 2024. 8
- [41] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 1, 2

- [42] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 1
- [43] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, Karanjeet Singh, Doug Kang, Ankur Jain, Hongyu H`e, Max Schwarzer, Tom Gunter, Xiang Kong, Aonan Zhang, Jianyu Wang, Chong Wang, Nan Du, Tao Lei, Sam Wiseman, Guoli Yin, Mark Lee, Zirui Wang, Ruoming Pang, Peter Grasch, Alexander Toshev, and Yinfei Yang. Mm1: Methods, analysis & insights from multimodal llm pretraining, 2024. 8
- [44] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 7
- [45] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models, 2024. 8
- [46] Yuxuan Qiao, Haodong Duan, Xinyu Fang, Junming Yang, Lin Chen, Songyang Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Prism: A framework for decoupling and assessing the capabilities of vlms, 2024. 9
- [47] Tanik Saikh, Tirthankar Ghosal, Amish Mittal, Asif Ekbal, and Pushpak Bhattacharyya. Scienceqa: A novel resource for question answering on scholarly articles. International Journal on Digital Libraries, 23(3):289–301, 2022. 1, 2
- [48] Maarten Sap, Vered Shwartz, Antoine Bosselut, Yejin Choi, and Dan Roth. Commonsense reasoning for natural language processing. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts, pages 27–33, Online, 2020. Association for Computational Linguistics. 8
- [49] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Mathllava: Bootstrapping mathematical reasoning for multimodal large language models, 2024. 2
- [50] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. Aligning large multimodal models with factually augmented rlhf, 2023. 8
- [51] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 7, 8
- [52] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 8
- [53] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed

- Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamavo1: Rethinking step-by-step visual reasoning in llms. arXiv preprint arXiv:2501.06186, 2025. 7
- [54] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms,

2024. 8

- [55] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024. 7, 8
- [56] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 1
- [57] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. 8
- [58] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 7
- [59] Siheng Xiong, Yuan Yang, Ali Payani, James C Kerce, and Faramarz Fekri. Teilp: Time prediction over knowledge graphs via logical reasoning, 2024. 8
- [60] Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason stepby-step, 2025. 2, 5, 7, 8
- [61] Zhiyang Xu, Chao Feng, Rulin Shao, Trevor Ashby, Ying Shen, Di Jin, Yu Cheng, Qifan Wang, and Lifu Huang. Vision-flan: Scaling human-labeled tasks in visual instruction tuning, 2024. 8
- [62] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 5
- [63] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search, 2024. 2, 7
- [64] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International Conference on Machine Learning, pages 57730–57754. PMLR, 2024. 6
- [65] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference

- on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 1, 2, 6
- [66] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multidiscipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. 6
- [67] Xiang Yue, Tianyu Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. Advances in Neural Information Processing Systems, 37:90629–90660,

2025. 1

- [68] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 5
- [69] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 7
- [70] Lei Zhang and Yong Rui. Image search—from thousands to billions in 20 years. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 9 (1s):1–20, 2013. 2
- [71] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186, 2024. 6
- [72] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023. 8
- [73] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. In The Thirteenth International Conference on Learning Representations, 2024. 2, 7

### 7. Supplementary Material

- 7.1. Image Number Distribution

0 images 1 image 2 images 3 images 4 images 5+ images

0

10

20

30

40

50

60

PercentageofEntries

558847 (61.7%)

283516 (31.3%)

34830 (3.8%)

12059 (1.3%) 8402 (0.9%) 8506 (0.9%)

Number of Images Distribution in VisualWebInstruct Total Entries: 906160

- 7.2. MAmmoTH-VL2 Training Configuration

|Model Architecture| | |Data Processing| |
|---|---|---|---|---|
|Base Language Model Vision Encoder Vision-Language Connector Vision Select Layer Patch Merge Type Starting Checkpoint<br><br>|Qwen/Qwen2.5-7B-Instruct google/siglip-so400m-patch14-384 MLP-based projector (2-layer with GELU)<br><br>-2 (second-to-last layer) spatial unpad MAmmoTH-VL<br><br>| |Image Aspect Ratio Image Grid Pinpoints Group by Modality Image Start/End Tokens Image Patch Token Lazy Preprocessing<br><br>|anyres max 4 (1x1),...,(6x6) Enabled Disabled Disabled Enabled<br><br>|
|Training Configuration| | |Dataset Configuration| |
|Training Epochs Batch Size Maximum Sequence Length Learning Rate Vision Tower Learning Rate Weight Decay Warmup Ratio LR Scheduler|1<br><br>256<br><br>8,192 tokens<br><br>1e-5 (language and projector)<br><br>2e-6<br><br><br>0.0<br><br>0.03<br><br>Cosine<br><br>| |Primary Dataset Additional Dataset Prompt Template<br><br>|VisualWebInstruct LLaVA-CoT (9:1 ratio) qwen 2 5<br><br>|
|Tunable Components| | |Optimization| |
|Language Model Vision Tower MLP Adapter Gradient Checkpointing Torch Compile<br><br>|Enabled Enabled Enabled Enabled Enabled (inductor)| |Distributed Training TF32 Precision Mixed Precision TF32 Precision<br><br>|DeepSpeed Zero-3 Enabled BF16 Enabled|

#### 7.3. Prompt for Each Stage

|QA Pairs Extraction<br><br>"""Analyze this webpage content and extract questions, images, and complete solution details in Markdown format. Please format your response as follows:<br><br>**Question 1:** [complete question text]<br><br>**Images:**<br><br>* [First image URL if available]<br>* [Second image URL if available] [continue for each additional image...]<br>**Solution:** [Copy the complete solution text from the webpage, including all steps, explanations, and calculations] **Images in Solution:**<br>* [First image URL if available]<br>* [Second image URL if available] [continue for each additional image...] [repeat for each additional question...] Requirements:<br><br><br>- Keep the complete solution text exactly as shown in the webpage<br>- Use Markdown formatting throughout the response<br>- Mark missing content as "Not found"<br>- For images, include URL only<br>- For multiple questions, number them sequentially<br>- Do not summarize or modify the solution text<br>- Preserve all mathematical notations and formulas<br>- Keep all step-by-step explanations intact<br>- Preserve all line breaks and indentation in solution text<br>- If there is no question in the content, mark it as "Not found"<br>- If the webpage is empty or missing, return nothing Webpage content: {Accessibility Tree} """<br>|
|---|

|QA Pairs Validation<br><br>"""Please analyze this question-answer pair and its images: Question: complete question text Solution: complete solution text Your tasks:<br><br>1. Determine if the question is meaningful and valid.<br>2. For the question images (if any), determine if each is:<br><br>- Properly referenced in the question<br>- Clear and visible<br>- Actually helps understand the question<br><br><br>3. For the solution images (if any), determine if each is:<br><br><br>- Helps explain the solution Notes:<br>- Image indices start from 0 (e.g., first image is index 0, second is index 1, etc.)<br>- Images should be marked as valid if they show the actual content being discussed<br>- Images should be marked as invalid only if they are:<br><br>* Completely irrelevant to the question/solution<br>* Corrupted or unreadable<br>* Duplicate or redundant<br><br><br>Question Images: [Images loaded here] Solution Images (starting a new section, indexes reset to 0): [Images loaded here] Please respond in this exact format: QUESTION VALID: [yes/no] ANALYSIS: [Brief explanation of why the question is valid/invalid] QUESTION IMAGES: [comma-separated list of valid image indices starting from 0] QUESTION IMAGES REASON: [Brief explanation for each image decision] SOLUTION IMAGES: [comma-separated list of valid image indices starting from 0] SOLUTION IMAGES REASON: [Brief explanation for each image decision]<br><br>CRITICAL RESPONSE FORMAT INSTRUCTIONS:<br><br>- You MUST respond using EXACTLY this format with no additional text<br>- Use ONLY numeric indices for images, starting from 0<br>- If no images are valid, use an empty string<br>- Be precise and use actual numbers<br>- Always use numeric indices (0,1,2...)<br>- Use empty string for no images (e.g., "SOLUTION IMAGES: ")<br><br>- Do not add explanatory text in the indices field """<br>|
|---|

|Answer Alignment<br><br>"""Given the question and the provided image(s), compare these two answers and determine if they are aligned. Question: question GPT’s Answer: gptanswer Real Answer: realanswer Example of Aligned Answers: Question: What is 2 + 2? GPT Answer: 4 Real Answer: 4 Example of Misaligned Answers: Question: What is derivative of x2? GPT Answer: 2x + 1 Real Answer: 2x Are these answers aligned? Respond with just ’Yes’ or ’No’ on the first line. Provide a brief explanation on the second line."""|
|---|

#### 7.4. Example of QA Pair

Question: How many angles does a quadrilateral have? A. 1 B. 2 C. 3 D. 4

Web Answer: Right Answer is: D SOLUTION An angle is made up of two rays starting from a common initial point (or a common vertex). A quadrilateral has 4 angles. Consider quadrilateral ABCD. It can be seen here that there are 4 angles here; ∠ A , ∠ B , ∠ C and ∠ D .

- GPT Answer1: To determine how many angles a quadrilateral has, let's reason through it step by step:

1. **Definition of a Quadrilateral**: A quadrilateral is a polygon with four sides. 2.

**Understanding Angles**: In any polygon, the number of angles is equal to the number of sides because each angle is formed where two sides meet. 3. **Applying the Definition**: Since a quadrilateral has four sides, by definition, it will also have four angles. 4. **Double-check with Examples**: Consider common examples of quadrilaterals like squares, rectangles, and diamonds—all these shapes have four angles. So, based on the above reasoning, the quadrilateral depicted in the image has four angles. Answer: D. 4

[Figure 61]

- GPT Answer2: To determine how many angles a quadrilateral has, let's break it down step by step: 1.

**Definition of a Quadrilateral**: A quadrilateral is a polygon with four sides. 2. **Interior Angles in a Quadrilateral**: By definition, any polygon with four sides (a quadrilateral) will have four interior angles. 3. **Examples and Properties**: - Consider a square, rectangle, or any general quadrilateral. Each has four corners, and at each corner, there is an angle formed between two sides. - A rectangle, for example, has four right angles. A rhombus has four angles, the sum of which is also 360 degrees. 4. **Summation of Angles**: The sum of the interior angles of any quadrilateral is 360 degrees. This sum indicates there must be four angles, as each angle contributes to this total. 5. **Checking against other Shapes**: - A triangle, which is a threesided polygon, has three angles. - Extending to more sides, a pentagon (five-sided polygon) has five angles. - This shows the number of angles is directly determined by the number of sides of the polygon. Based on the above reasoning, a quadrilateral has 4 angles. Hence, the correct answer is: `Answer: 4`

- GPT Answer3: A quadrilateral is a polygon with four sides. By definition, a polygon has the same number of angles as it does sides. Therefore: 1. A quadrilateral has four sides. 2. Since it has four sides, it will also have four angles. Thus, a quadrilateral has four angles. Answer: D. 4

- GPT Answer4: A quadrilateral is a polygon with four sides. By definition, any polygon with \( n \) sides will also have \( n \) angles. Since a quadrilateral has 4 sides, it must also have 4 angles. Let's break this down: 1. Identify the shape: The given shape is a quadrilateral. 2. Determine the number of sides: A quadrilateral has 4 sides. 3. Determine the number of angles: In any polygon, the number of angles is equal to the number of sides. Therefore, a quadrilateral has 4 angles. Answer: D. 4

