arXiv:2506.07977v3[cs.CV]26Jun2025

# OneIG-Bench: Omni-dimensional Nuanced Evaluation for Image Generation

Jingjing Chang1,2 Yixiao Fang2,† Peng Xing2 Shuhan Wu2 Wei Cheng2 Rui Wang2 Xianfang Zeng2 Gang Yu2,‡ Hai-Bao Chen1,‡

1 SJTU 2 StepFun † Project lead ‡ Corresponding author

[Figure 1]

[Figure 2]

[Figure 3]

Dataset Code Project Page

## Abstract

Text-to-image (T2I) models have garnered significant attention for generating highquality images aligned with text prompts. However, rapid T2I model advancements reveal limitations in early benchmarks, lacking comprehensive evaluations, for example, the evaluation on reasoning, text rendering and style. Notably, recent stateof-the-art models, with their rich knowledge modeling capabilities, show promising results on the image generation problems requiring strong reasoning ability, yet existing evaluation systems have not adequately addressed this frontier. To systematically address these gaps, we introduce OneIG-Bench, a meticulously designed comprehensive benchmark framework for fine-grained evaluation of T2I models across multiple dimensions, including prompt-image alignment, text rendering precision, reasoning-generated content, stylization, and diversity. By structuring the evaluation, this benchmark enables in-depth analysis of model performance, helping researchers and practitioners pinpoint strengths and bottlenecks in the full pipeline of image generation. Specifically, OneIG-Bench enables flexible evaluation by allowing users to focus on a particular evaluation subset. Instead of generating images for the entire set of prompts, users can generate images only for the prompts associated with the selected dimension and complete the corresponding evaluation accordingly. Our codebase and dataset are now publicly available to facilitate reproducible evaluation studies and cross-model comparisons within the T2I research community.

## 1 Introduction

Recent years have witnessed remarkable advancements in text-to-image (T2I) models across image quality, semantic alignment, text rendering precision, and knowledge-driven reasoning in image generation [55, 46, 19, 60, 38, 53, 45]. However, the development of evaluation systems has significantly lagged behind model progress: most existing benchmarks remain confined to single-dimensional assessments, lacking comprehensiveness. For instance, T2ICompBench [34], GenEval [25], and DSG-1k [16] focus on short-text semantic understanding, while DPG-Bench [31] introduces dense prompt evaluation but lackly coverage of limited dimensions like style and text. Although WorldGenBench [80] addresses world knowledge and reasoning, a more comprehensive multi-dimensional evaluation framework is urgently needed to provide scientific model assessments and guide technological development.

To drive the advancement of text-to-image models, we posit that the development of a holistic benchmark framework capable of evaluating models across multiple interconnected dimensions is imperative for fostering rigorous, comprehensive assessment. We introduce OneIG-Bench shown

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

longtextrendering

[Figure 8]

nitaolf arf-g

natural-sce ne

elyts-itiffarg txe t

black b o

“System update Now supports gardening mode!”

advertising imagery

Generated image Reference images

[Figure 9]

ard text

watercolor

em txe t

sionism

“Freshly baked bread daily”

Three adventures embark on a journey…

[Figure 10]

PPT

text

men u

m

cubis

generation

[Figure 11]

impre

fauvism

poster design

“Dreamland Cat …”

other styles…

[Figure 12]

artistic renderings

Princess with …

Text Rendering

Anime & Stylization

[Figure 13]

geography

Five highest mountains…

[Figure 14]

The animated series figure stands …

Anime

computer science

###### OneIG-Bench

Knowledge & Reasoning

biology

[Figure 15]

[Figure 16]

Animal and plant cells…

mathematics

端午节，龙舟竞 渡，鼓声震天...

Multilingualism

physics

[Figure 17]

chemistry

[Figure 18]

中秋节，一家人围 坐在一起赏月...

mon sense

Hands washing steps…

General Object

com

Portrait

[Figure 19]

[Figure 20]

Night descends the city light …

Close-up portrait of an old woman…

[Figure 21]

A man in gym…

[Figure 22]

A snow leopard…

- Figure 1: Overview of OneIG-Bench. OneIG-Bench comprises six core categories, each designed to evaluate targeted capabilities across distinct generative dimensions, with approximately 200 carefully curated prompts per category to ensure comprehensive coverage of diverse scenarios.

in Figure 1, comprising over 1000 omni-dimensional prompts, primarily sourced from real-world user inputs, systematically designed to comprehensively evaluate text-to-image models across diverse generative capabilities. Based on distinct generative themes, we classify the evaluation dataset into six core assessment categories: General Object, Portrait, Anime and Stylization, Text Rendering, Knowledge and Reasoning and Multilingualism. By leveraging our framework for the taxonomic differentiation of generative themes, we enable a more nuanced evaluation of models’ multidimensional capabilities. Consequently, this systematic assessment allows users to identify models with tailored capabilities that align precisely with their specific application requirements.

For different evaluation dimensions, we have carefully devised quantitative indicators, taking into account various factors to ensure the comprehensiveness and objectivity of the evaluation. These indicators are designed to precisely measure the model’s performance in different aspects related to specific subject domains, enabling a more accurate and in-depth assessment. Specifically for the evaluation of General object, Portrait, Anime and Stylization, we have developed an evaluation system for the ability to comply with input prompts. Regarding Anime and Stylization, we also incorporate the stylistic similarity into our evaluation system, specifically designed to assess models’ capabilities in reproducing diverse artistic styles. In the Text Rendering evaluation segment, we focus on three core metrics: the edit distance between generated text and ground truth, the text completion rate in one visual output, and the overall text generation accuracy. For the Knowledge and Reasoning part, our assessment centers on whether the model possesses the required domain knowledge and can accurately interpret user intent, thereby enabling the generation of semantically coherent and logically consistent images. For the Multilingualism part, we evaluate the alignment between the images generated from cultural element prompts and the corresponding cultural elements. Additionally, we also evaluate the diversity across all six dimensions.

We summarize our key contributions in the following three points:

- • We present OneIG-Bench, which consists of six prompt sets, with the first five — 245 Anime and Stylization, 244 Portrait, 206 General Object, 200 Text Rendering, and 225 Knowledge and Reasoning prompts — each provided in both English and Chinese, and 200 Multilingualism prompts, designed for the comprehensive evaluation of current text-to-image models.
- • A systematic quantitative evaluation is developed to facilitate objective capability ranking through standardized metrics, enabling direct comparability across models. Specifically, our evaluation framework allows T2I models to generate images only for prompts associated with a particular evaluation dimension, and to assess performance accordingly within that dimension.

- • State-of-the-art open-sourced methods as well as the proprietary model are evaluated based on our proposed benchmark to facilitate the development of text-to-image research.

## 2 Related Works

- 2.1 Text to Image Models

Text-to-image (T2I) generation aims to develop models that produce images semantically consistent with given text descriptions. Early explorations during the generative adversarial network (GAN) [26, 57] era laid foundational work, but these methods suffered significant limitations due to mode collapse, often failing to generate even simple subject matters accurately. In recent years, generative approaches based on the diffusion model paradigm have emerged as a dominant trend [30, 59]. Notable advancements include Unet-based architectures like Stable Diffusion XL [46], Diffusion Image Transformer (DiT)-based models [9, 10, 42], double-stream MMDiT frameworks [60, 19], and hybrid designs combining double-stream and single-stream networks [78, 73, 38]. These models have achieved remarkable progress in generating high-quality images that closely align with textual semantics. With technological advancements, the evaluation framework for T2I models must evolve from single-dimensional attribute assessments (e.g., color, shape) to multi-layered evaluations, encompassing semantic alignment, stylistic consistency, and text rendering accuracy. Meanwhile, autoregressive-based models [12, 51, 54, 61, 77, 63, 68, 74] have demonstrated unique strengths in knowledge modeling and complex text comprehension, necessitating the integration of systematic reasoning capability evaluations into the assessment framework. In summary, the rapid development of T2I generation underscores the urgent need for a comprehensive and rigorous evaluation system to accurately measure model performance, identify strengths and weaknesses, and foster sustainable progress in the field.

- 2.2 Text to Image Evaluation

In the early stages of text-to-image development, researchers typically employed some metrics to evaluate image generation quality, such as FID [28](Fréchet Inception Distance), SSIM [69](Structural Similarity Index Measure), PSNR(Peak Signal to Noise Ratio), etc. However, these methods do not provide a comprehensive understanding of the model’s capabilities and fail to capture the model’s ability to comprehend higher-order semantics.

In recent years, the technology of text-to-image models has been evolving rapidly, and the corresponding evaluation system urgently needs to be innovated and upgraded. Taking the evaluation of Stable Diffusion 1.5 [55] and Stable Diffusion XL [46] as examples, most of the existing benchmark evaluations (Attend [7], Hrs-bench [5], CC500 [20]) focus on judging the degree of restoration of the core elements in the prompts. In the face of the rapid evolution of model technology, it has become difficult to comprehensively and accurately measure the actual performance and innovative potential of the models. Subsequently, evaluation methods such as PartiPrompt [77], DrawBench [56], TIFA [32], Gecko [70], EvalAlign [62], T2ICompBench [34], T2ICompBench++ [33], GenEval [25], EvalMuse [27], DPG-Bench [31], and GenAI-Bench [39] introduced Visual Language Models (CLIP [49],BLIP [41], MLLMs [3, 67, 14]) as evaluators. These approaches aim to maximize the utilization of model capabilities for jointly assessing prompts and images. However, these evaluation methods primarily focus on the prompt following ability of text-to-image models, often neglecting other critical aspects. Some alternative methods(MJHQ-30K [40], HPSv2 [72], Pick-a-pic [37]) have attempted to assess the aesthetic quality of images generated by these models. Recently, to keep up with the advances in text-to-image models, evaluation frameworks such as WISE [43], WorldGenBench [80], Commonsense-T2I [22], and PhyBench [48] have been developed to assess the models’ knowledge and reasoning capabilities.

- 3 Benchmark

- 3.1 Benchmark Overview

With the rapid advancement of text-to-image (T2I) models, the existing evaluation frameworks for T2I models urgently require improvement to measure model strengths and weaknesses comprehensively. Early studies have explored evaluation methods from diverse perspectives, such as compositional

text-to-image generation tasks [7, 34, 22], including concept correlation, attribute binding (focusing on color attributes), and spatial relationship modeling. However, evaluations solely focusing on compositional content exhibit significant limitations [25, 31, 27], failing to adequately address broader natural language understanding and other quantitative image assessment dimensions.

In response, recent evaluation frameworks like GenEval [25], EvalMuse [27], and DPG-Bench [31] adopt an object-centric structured paradigm to quantify T2I model performance on specific tasks. Nevertheless, these methods predominantly rely on vision-language models (VLMs) [2, 4], object detection models [15, 11] or visual question answering (VQA) [1] models for element-level alignment assessment, suffering from notable deficiencies in evaluating style consistency and text rendering accuracy, with a lack of high-precision metrics. Additionally, human-based evaluation is prohibitively costly in terms of time and resources, making automated evaluation with limited prompts increasingly critical. Notably, with the rapid development of reasoning-oriented models [45], this study proactively introduces reasoning task evaluation to accurately measure models’ knowledge representation and reasoning-driven image generation capabilities.

Table 1 systematically reviews the advantages and disadvantages of recent evaluation methods, presenting a comprehensive framework named OneIG-Bench from six dimensions: scene coverage, prompt distribution diversity, evaluation content, multilingualism support, automation level and leaderboard availability. OneIG-Bench covers core T2I scenarios (e.g., style image generation, text rendering, reasoning-based drawing), supports multi-format prompt inputs (including long/short texts and phrase/tag-based prompts), and employs customized automated metrics for different scenarios. Experimental results demonstrate that this framework effectively identifies performance bottlenecks and strengths of current models, providing a scientific basis for T2I model optimization. Hereafter, unless otherwise specified, OneIG-Bench refers to OneIG-Bench-EN.

- Table 1: Comparison between OneIG-Bench and other previous benchmarks. In the column of prompt diversity, L denotes long prompt, S denotes the short prompt, NP denotes the natural language prompt, T denotes the tag-based prompt, and P denotes the phrase-based prompt.

|Scenes|Prompt Diversity<br><br>|Evalution|
|---|---|---|
|General<br><br>Style<br><br>Text<br><br>Reason|Length<br><br>Type<br><br>Count|Alignment<br><br>Text<br><br>Style<br><br>Diversity|

Leaderboard

Multilingualism

AutoEval

Benchmark

PartiPrompt [77] ✓ ✗ ✗ ✗ L, S NP 1,600 ✗ ✗ ✗ ✗ ✗ ✗ ✗ DrawBench [56] ✓ ✗ ✓ ✗ S NP 200 ✗ ✗ ✗ ✗ ✗ ✗ ✗ TIFA [32] ✓ ✗ ✗ ✗ S NP 4,000 ✓ ✗ ✗ ✗ ✗ ✓ ✗ T2ICompBench [34] ✓ ✗ ✗ ✗ S NP 6,000 ✓ ✗ ✗ ✗ ✗ ✓ ✗ GenEval [39] ✓ ✗ ✗ ✗ S NP 553 ✓ ✗ ✗ ✗ ✗ ✓ ✗ EVALALIGN [62] ✓ ✓ ✓ ✓ S NP 100 ✓ ✗ ✗ ✗ ✗ ✓ ✗ WISE [43] ✓ ✗ ✗ ✓ S NP 1,000 ✓ ✗ ✗ ✗ ✗ ✓ ✗ EvalMuse [27] ✓ ✗ ✗ ✗ S NP 199 ✓ ✗ ✗ ✗ ✗ ✓ ✓ DPG-Bench [31] ✓ ✗ ✗ ✗ L NP 1,065 ✓ ✗ ✗ ✗ ✗ ✓ ✗

OneIG-Bench ✓ ✓ ✓ ✓ L, S NP,T,P 2,440∗ ✓ ✓ ✓ ✓ ✓ ✓ ✓

∗:OneIG-Bench consists of two subsets: OneIG-Bench-EN and OneIG-Bench-ZH. OneIG-Bench-EN includes 245 Anime and Stylization prompts, 244 Portrait prompts, 206 General Objectz prompts, 200 Text Rendering prompts, and 225 Knowledge and Reasoning prompts. OneIG-Bench-ZH comprises manually translated and verified Chinese versions of the prompts in OneIG-Bench-EN, along with additional prompts from the Multilingualism category, totaling 1, 320 prompts. In total, OneIG-Bench contains 2, 440 prompts.

### 3.2 Benchmark Construction

In constructing the evaluation prompt set, as illustrated in Figure 2, we have established five core steps to ensure the diversity and comprehensiveness of the prompt set, aligning with the design principles of our benchmark framework.

###### Prompt Collection Cluster and Deduplication Prompt Rewrite Human Review

[Figure 23]

[Figure 24]

[Figure 25]

Run clustering

Realistic portrait, slender bald man, seated on chair, injury with blood, war outside

A dolphin is swimming in the sea, ultra-high resolution, dive and breach

…

Prompt set

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Filter out similar prompts

Remove Revision

GPT

Prompt subset

A dolphin is swimming in the sea, ultra-high resolution, dive and breach

Realistic portrait, slender bald man, seated on chair, bright pristine white room.

The entire body of the dolphin is meticulously crafted to resemble an entity with…

…

- Figure 2: The construction pipeline of OneIG-Bench. The construction pipeline comprises four methodical steps to derive our final assessment prompts, ensuring the diversity and comprehensiveness of the benchmark.

In the first step, we curated prompts and generation scenes by filtering publicly accessible internet data, user inputs, and some established datasets, thereby ensuring that the benchmark focuses on content aligned with real-world user needs rather than rare or specialized contexts.

In the second step, we apply a clustering approach to balance the distribution of prompts across different scenes and semantic dimensions, ensuring that no single category dominates. Within each cluster, we identify some prompts with high semantic overlap, which can compromise evaluation diversity; thus, we implement a deduplication pipeline that filters out redundant prompts based on cosine similarity to the cluster center embeddings. Besides, we endeavor to maintain a relatively consistent proportion of prompts across the five defined dimensions during prompt selection.

Following deduplication and subset curation, we employ a large language model (LLM) to rewrite the original prompts. Concurrently, constraints are applied to the word-level length distribution of prompts, enabling a structured analysis of model performance across varying text complexities. The prompt corpus is intentionally structured into three length categories: concise texts (fewer than 30 words), mid-complexity scenarios (30–60 words), and elaborate texts (exceeding 60 words). The corresponding ratio of three categories is around 1:2:1.

Finally, we performed manual reviews to filter out prompts containing sensitive content or conflicting semantics, ensuring the rationality of all benchmark prompts. This critical step not only enhances the dataset’s quality and reliability but also guarantees its suitability for fair and unbiased model evaluation across diverse generative scenarios.

Through a rigorously designed construction pipeline, this structured evaluation framework facilitates a granular assessment of model performance across multiple dimensions.

## 4 Evaluation

### 4.1 Metrics

To illustrate the metrics used in our benchmark, we provide the following general definitions. Given the text prompt set T = {T1,T2,...,Tn}, for each Tk ∈ T, the generated images are defined as Gk = {Gk1,Gk2,...,Gkm}, where m denotes the number of generated images, and a total of m × n images are generated as evaluation images.. To evaluate the style score, for each style-specific prompt Tk ∈ T, we define R = {R1k,R2k,R3k} as the set of corresponding style reference images, where l denotes the number of reference images. For text rendering, the original target text string in Tk is defined as sk, and the generated string in the corresponding image is defined as sˆk.

Semantic alignment. We follow the method introduced in DSG [16] on General Object, Portrait, Anime and Stylization to assess the semantic matching degree of each text-image sample. For each prompt, we initially leverage GPT-4o [44] to generate a question dependency graph. In the process of constructing the graph, our focus lies in formulating questions related to the overall information, spatial relationships, and the attributes of diverse objects. During the evaluation, Qwen2.5-VL-7B [4] is utilized to answer questions derived from the corresponding prompt and the generated image. A score of 1 is assigned for each correctly answered question. However, when calculating the aggregate score for a prompt, leaf node scores are conditionally validated: they contribute to the total score only if the root node question is answered correctly; otherwise, leaf node scores are reset to 0. The

final score for each prompt is computed as the sum of validated scores divided by the total number of questions.

Text Rendering. To accurately evaluate the text-generation capability of text-to-image models, we designed specialized text evaluation metrics. To extract the generated string sˆk, we first use a state-of-the-art Vision-Language Model (VLM, e.g., Qwen2.5-VL-7B [4]) to parse the text string and then clean it by removing symbols and consecutive spaces The metrics are as follows:

- (1) Edit Distance (ED): It is defined as the average edit distance between the generated text of evaluation images and the ground-truth text to be generated. We define the edit distance score of the i-th evaluation image as EDi = L(ˆsi,si), where L(·) denotes the Levenshtein distance function. Therefore, the overall edit distance score of the model is: ED = n×1m i n=1×m EDi.

- (2) Completion Rate (CR): It is defined as the proportion of the number of evaluation images with completely correct generated text to the total number of evaluation images. We define that the score for the i-th generated image in this criterion is CRi = 1 if and only if the edit distance score of the i-th image is 0, i.e., EDi = 0. Therefore, the overall CR score of the model is defined as CR = mi=1×n CRi/(m × n).
- (3) Word Accuracy (WAC): It is defined as a metric representing the ratio of all correctly generated words to the total number of words in the original target text strings among all prompts.

Based on our analysis on the evaluation results, we define the edit distance upper bound as ϕ, and edit distance exceeding ϕ indicate deficiencies in the model’s text-rendering capability. To facilitate metric ranking and readability, we integrated three metrics into a composite metric and defined the text score(Stext) as follows:

Stext = 1 − min(ϕ,ED) ∗ (1 − CR) ∗ (1 − WAC)/ϕ (1) where ϕ = 100 in OneIG-Bench. Considering that Chinese characters typically occupy twice as many bytes as English letters, we use ϕ = 50 in Equation 1 when computing the text score for OneIG-Bench-ZH, in order to maintain a comparable normalization scale.

Knowledge and Reasoning. We perform the evaluations using GPT-4o [44] and LLM2CLIP [35]. Specifically, GPT-4o is responsible for generating the textual reasoning answers, which serve as the core reference for evaluation. LLM2CLIP then measures the alignment between text and image by calculating the cosine similarity between the GPT-4o-generated answer and the corresponding generated image.

Style. For stylization evaluation, we curated multiple reference images per style and employed a dualstyle extraction framework to mitigate bias and enhance robustness. Specifically, the CSD [58] model and OneIG style image encoder(fine-tuned from CLIP [49], using images generated by CSGO [76]) are leveraged to encode the images and generate the corresponding embeddings. Subsequently, for each encoder, we quantitatively assess the model’s style capacity by computing the cosine similarity between the style embeddings of generated images and those of reference images. For each generated image, we choose the maximum similarity as the score of the image. The style similarity(S[csd,oneig]) of one style image encoder is defined as:

S[csd,oneig] =

n

1 n

k=1

m

1 m

i=1

cos(F(Gki ),F(Rjk)) (2)

max

j

where F(·) denotes the corresponding style image encoder. The final style score Sstyle is defined as Sstyle = (Scsd + Soneig)/2.

Diversity. In addition, we apply a form of similarity calculation to evaluate the diversity of the generation of the model introduced in [23]. The calculation of diversity is defined as follows: for a given model, we first compute the pairwise cosine similarity between every pair of images generated from the same prompt within a set of multiple generated outputs. These similarities are averaged per prompt to yield an intra-prompt similarity score. We then aggregate these intra-prompt averages across all prompts in the evaluation dataset using a global mean, resulting in an overall diversity metric. Following [23], we also applied DreamSim [21] to compute the cosine similarity and the formula is as follows:

#### SIMkij = cos(F(Gki ),F(Gkj)) (3)

where SIMkij denotes the cosine similarity between images generated by one text prompt, F(·) represents DreamSim [21] model. Thus, the diversity score(Sdiversity) can be defined:

  1

 . (4)

n

m

m

1 n

1 − SIMkij

Sdiversity =

Cm2

i=1

j=i+1

k=1

### 4.2 Results and Analysis

We evaluate a range of well-known image generation models on our benchmark, including unified multimodal models (Janus-Pro [13], BLIP3-o [8], BAGEL [18], Show-o2 [75], OmniGen2 [71]), open-source models (Stable Diffusion 1.5 [55], Stable Diffusion XL [46], Stable Diffusion 3.5 [60], Flux.1-dev [38], CogView4 [78], SANA [73], Lumina-Image 2.0 [47], and HiDream-I1-Full [29]) with A800 GPUs, as well as closed-source models (Imagen3 [36], Recraft V3 [66], Kolors 2.0 [64], Seedream 3.0 [24], Imagen4 [17] and GPT-4o [45]). To present a comprehensive comparison, we aggregate evaluation metrics across multiple dimensions, as summarized in Table 2. We define the sets of images generated based on the OneIG-Bench prompt categories General Object O, Portrait P, Anime and Stylization A (prompts without stylization), S (prompts with stylization), Text Rendering T , Knowledge and Reasoning KR and Multilingualism L. A more detailed, finegrained analysis of individual dimensions on OneIG-Bench follows in the subsequent sections and the appendix.

- Table 2: Overall quantitative comparison of different methods on OneIG-Bench. The table showcases the results of five core metrics for various methods. indicate the first, second, third, fourth, and fifth performance, respectively.

Method Alignment ↑ Text ↑ Reasoning ↑ Style ↑ Diversity ↑ Assessment Sets O, P, A, S T KR S O, P, A, S, T , KR

Janus-Pro [13] 0.553 0.001 0.139 0.276 0.365 BLIP3-o [8] 0.711 0.013 0.223 0.361 0.229 BAGEL [18] 0.769 0.244 0.173 0.367 0.251 BAGEL+CoT [18] 0.793 0.020 0.206 0.390 0.209 Show-o2-1.5B [75] 0.798 0.002 0.219 0.317 0.186 Show-o2-7B [75] 0.817 0.002 0.226 0.317 0.177 OmniGen2 [71] 0.804 0.680 0.271 0.377 0.242

Stable Diffusion 1.5 [55] 0.565 0.010 0.207 0.383 0.429 Stable Diffusion XL [46] 0.688 0.029 0.237 0.332 0.296 Stable Diffusion 3.5 Large [60] 0.809 0.629 0.294 0.353 0.225 Flux.1-dev [38] 0.786 0.523 0.253 0.368 0.238 CogView4 [78] 0.786 0.641 0.246 0.353 0.205 SANA-1.5 1.6B (PAG) [73] 0.762 0.054 0.209 0.387 0.222 SANA-1.5 4.8B (PAG) [73] 0.765 0.069 0.217 0.401 0.216 Lumina-Image 2.0 [47] 0.819 0.106 0.270 0.354 0.216 HiDream-I1-Full [29] 0.829 0.707 0.317 0.347 0.186

- Imagen3 [36] 0.843 0.343 0.313 0.359 0.188 Recraft V3 [66] 0.810 0.795 0.323 0.378 0.205 Kolors 2.0 [64] 0.820 0.427 0.262 0.360 0.300 Seedream 3.0 [24] 0.818 0.865 0.275 0.413 0.277

- Imagen4 [17] 0.857 0.805 0.338 0.377 0.199 GPT-4o [45] 0.851 0.857 0.345 0.462 0.151

Meanwhile, we evaluate a relatively smaller set of well-known image generation models on OneIGBench-ZH, including unified multimodal models (Janus-Pro [13], BLIP3-o [8], BAGEL [18]), opensource models (CogView4 [78], Lumina-Image 2.0 [47], and HiDream-I1-Full [29]) using A800 GPUs, as well as closed-source models (Kolors 2.0 [64], Seedream 3.0 [24], and GPT-4o [45]). In OneIG-Bench-ZH, Multilingualism part consists of 100 culture-related prompts and 100 portraitrelated prompts. To provide a clear overall comparison, the evaluation results across multiple dimensions are summarized in Table 3. On OneIG-Bench-ZH, GPT-4o [45] demonstrates outstanding performance, ranking first across most evaluation dimensions in Table 3 and Figure 3. In contrast, Seedream 3.0 [24] excels particularly in Chinese text rendering, significantly outperforming GPT4o. However, most models show limited capability in generating Chinese text, with many nearly incapable of producing legible Chinese characters. It is also worth noting that for most models, performance on alignment, reasoning, and style dimensions is slightly weaker on OneIG-Bench-ZH

- Table 3: Overall quantitative comparison of different methods on OneIG-Bench-ZH. The table showcases the results of five core metrics for various methods. indicate the first, second, third, fourth, and fifth performance, respectively.

Method Alignment ↑ Text ↑ Reasoning ↑ Style ↑ Diversity ↑

Assessment Sets Ozh, Pzh,

Ozh, Pzh, Azh, Szh, Tzh, KRzh, Lzh

Tzh KRzh Szh

Azh, Szh, Lzh

Janus-Pro [13] 0.324 0.148 0.104 0.264 0.358 BLIP3-o [8] 0.608 0.092 0.213 0.369 0.233 BAGEL [18] 0.672 0.365 0.186 0.357 0.268 BAGEL+CoT [18] 0.719 0.127 0.219 0.385 0.197

Cogview4 [78] 0.700 0.193 0.236 0.348 0.214 Lumina-Image 2.0 [47] 0.731 0.136 0.221 0.343 0.240 HiDream-I1-Full [29] 0.620 0.205 0.256 0.304 0.300

Kolors 2.0 [64] 0.738 0.502 0.226 0.331 0.333 Seedream 3.0 [24] 0.793 0.928 0.281 0.397 0.243 GPT-4o [45] 0.812 0.650 0.300 0.449 0.159

than on OneIG-Bench, indirectly reflecting that their ability to understand and generate Chinese semantics still requires further improvement.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

###### GPT-4o Seedream 3.0 Kolors 2.0 Lumina-Image 2.0 BAGEL+CoT

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

1.00 1.00 0.78 0.78 0.56

PROMPT: 端午节，龙舟竞渡，鼓声震天，选手们奋力划桨，江水被船桨拍打得浪花四溅。

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Seedream 3.0 GPT-4o Korlors 2.0 BAGEL HiDream-I1-Full

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

ED:0 CR:1 WAC:1.0 ED:59 CR:0 WAC:0.64 ED:3 CR:0 WAC:0.72 ED:11 CR:0 WAC:0.00 ED:28 CR:0 WAC:0.00

PROMPT: 这是一张名为“校园趣事：图书馆搞笑活动”的海报。笑声充满了图书馆，一名学生模仿了一个滑稽的场景。

- Figure 3: An illustration of the generation results and the corresponding scores on OneIG-Bench-ZH. The first row shows the alignment results and the second row shows the the text rendering results. And the evaluation scores are displayed in the upper left corner of the image.

### 4.2.1 Semantic Alignment and Diversity

In the alignment dimension shown in Table 2, Imagen4 [17], GPT-4o [45] and Imagen3 [36] consistently outperform other models, with Imagen4 and GPT-4o showing superior alignment performance. Furthermore, as indicated in Table 4, most models achieve significantly higher alignment accuracy when responding to natural language prompts than to tag-based or phrase-based prompts. A possible explanation is that tag and phrase prompts may introduce ambiguity, such as attribute confusion between entities or logical inconsistencies, which complicates semantic alignment. Longer prompts tend to involve greater semantic complexity and structural variation, often resulting in lower alignment scores. Notably, models incorporating T5 [52] or other large language models appear more robust in handling long prompts, exhibiting less semantic degradation.

Diversity is informative when assessed among models with comparable levels of alignment. In real-world applications, generative models are generally expected to produce varied outputs while maintaining close adherence to the input prompts. Although Stable Diffusion 1.5 [55] and Janus-

- Table 4: The alignment and diversity evaluation results. NP denotes the natural language prompt. T&P denotes the tag-based and phrase-based prompt. Short, Medium and Long represent the length of the prompts, where Short denote the number of words is less than 30, Medium denotes the number between 30 and 60, and Long denotes the number exceeding 60. indicate the first, second, third, fourth, and fifth performance respectively.

|Alignment↑<br><br>| |Diversity↑| |
|---|---|---|---|
|NP T&P<br><br>|Short Medium Long<br><br>|NP T&P<br><br>|Short Medium Long|

Method

Janus-Pro [13] 0.557 0.533 0.609 0.548 0.515 0.372 0.304 0.407 0.332 0.347 BLIP3-o [8] 0.719 0.671 0.754 0.712 0.674 0.237 0.161 0.283 0.192 0.198 BAGEL [18] 0.776 0.734 0.782 0.769 0.759 0.257 0.197 0.344 0.190 0.194 BAGEL+CoT [18] 0.798 0.767 0.824 0.793 0.767 0.214 0.164 0.244 0.184 0.189 Show-o2-1.5B [75] 0.805 0.760 0.800 0.799 0.793 0.191 0.142 0.226 0.162 0.157 Show-o2-7B [75] 0.825 0.778 0.825 0.819 0.807 0.182 0.130 0.216 0.152 0.151 OmniGen2 [71] 0.812 0.768 0.809 0.805 0.799 0.250 0.174 0.329 0.185 0.190

Stable Diffusion 1.5 [55] 0.570 0.541 0.616 0.558 0.537 0.434 0.381 0.481 0.389 0.403 Stable Diffusion XL [46] 0.688 0.685 0.732 0.685 0.657 0.303 0.239 0.337 0.263 0.278 Stable Diffusion 3.5 Large [60] 0.818 0.762 0.826 0.808 0.795 0.229 0.194 0.267 0.194 0.206 Flux.1-dev [38] 0.791 0.759 0.794 0.785 0.780 0.243 0.190 0.302 0.194 0.199 CogView4 [78] 0.796 0.737 0.792 0.788 0.777 0.211 0.153 0.277 0.158 0.159 SANA-1.5 1.6B(PAG) [73] 0.769 0.726 0.770 0.764 0.752 0.231 0.153 0.284 0.177 0.192 SANA-1.5 4.8B(PAG) [73] 0.773 0.729 0.782 0.767 0.749 0.223 0.154 0.264 0.181 0.191 Lumina-Image 2.0 [47] 0.825 0.788 0.829 0.819 0.812 0.224 0.149 0.282 0.171 0.180 HiDream-I1-Full [29] 0.834 0.806 0.849 0.834 0.806 0.192 0.142 0.260 0.139 0.140

- Imagen3 [36] 0.849 0.809 0.859 0.841 0.832 0.189 0.173 0.246 0.146 0.153 Recraft V3 [66] 0.816 0.781 0.838 0.809 0.788 0.209 0.178 0.246 0.178 0.180 Kolors 2.0 [64] 0.824 0.798 0.847 0.814 0.807 0.308 0.230 0.359 0.261 0.259 Seedream 3.0 [24] 0.818 0.815 0.838 0.825 0.789 0.280 0.246 0.342 0.235 0.227

- Imagen4 [17] 0.860 0.843 0.875 0.854 0.847 0.199 0.197 0.276 0.147 0.149 GPT-4o [45] 0.857 0.820 0.869 0.851 0.838 0.154 0.124 0.177 0.134 0.134

Pro [13] achieve notably high diversity scores, these results are less indicative of true generative quality, as they largely stem from the models’ inconsistent preservation of semantic alignment in the generated images. And Kolors 2.0 [64] exhibits outstanding diversity without compromising its alignment performance, and is regarded as a model with excellent diversity performance.

While stylization can be viewed as a specific facet of semantic alignment, performance in this dimension does not entirely coincide with overall alignment outcomes. GPT-4o [45] retains a clear lead in stylization, while both Seedream 3.0 [24] and the SANA series methods also exhibit strong performance Notably, Stable Diffusion 1.5 [55] demonstrates impressive stylization capabilities despite its relatively poor performance in semantic alignment. This may be attributed to its data cleaning process, which likely preserved a broad range of stylistic patterns and enabled the model to generate images with distinct stylistic characteristics.

### 4.2.2 Text Rendering

We evaluate the performance of text-to-image models with a particular focus on text rendering, which assesses how accurately these models reproduce textual content within generated images. For clearer comparison across varying textual complexities, the images are categorized by prompt length.

As shown in Table 5, Seedream 3.0 [24] achieves the best performance across nearly all subdimensions of completion ratio and word accuracy count, as well as in edit distance for short and medium-length prompts. Closer inspection of the generated images reveals that the relatively higher edit distances observed for Seedream 3.0 are mainly associated with the difficulty of rendering long text inputs, particularly in structured formats such as articles and PowerPoint slides. Images generated with long text inputs may contain more frequent textual deviations. While Seedream 3.0 performs well on certain long prompts, the overall complexity of long-text rendering contributes to an increase in edit distance.

As shown in Figure 4, although GPT-4o [45] demonstrates strong visual accuracy, it shows no particular advantage in quantitative evaluation. This is mainly due to our strict evaluation criterion:

- Table 5: The text rendering evaluation results. In the table, Short, Medium and Long represent the length of the prompts, where Short denote the number of words is less than 30, Medium denotes the length between 30 and 60, and Long denotes the length exceeding 60. indicate the first, second, third, fourth, and fifth performance respectively.

|Edit Distance (ED)↓<br><br>|Completion Rate (CR)↑|Word Accuracy (WAC)↑|
|---|---|---|
|Short Medium Long<br><br>|Short Medium Long<br><br>|Short Medium Long|

Method

Janus-Pro[13] 33.041 59.695 295.020 0.000 0.000 0.000 0.001 0.001 0.000 BLIP3-o[8] 31.627 59.584 258.510 0.005 0.000 0.000 0.014 0.022 0.012 BAGEL[18] 25.650 44.453 242.415 0.005 0.079 0.000 0.128 0.288 0.148 BAGEL+CoT[18] 30.550 57.397 255.220 0.000 0.005 0.000 0.024 0.033 0.011 Show-o2-1.5B [75] 32.205 59.232 296.580 0.000 0.000 0.000 0.002 0.004 0.001 Show-o2-7B [75] 31.709 59.179 294.720 0.000 0.000 0.000 0.004 0.006 0.001 OmniGen2 [71] 17.000 36.145 181.525 0.255 0.145 0.000 0.543 0.614 0.534

Stable Diffusion 1.5 [55] 36.227 60.480 290.245 0.000 0.011 0.000 0.004 0.020 0.004 Stable Diffusion XL [46] 35.045 61.824 290.470 0.000 0.008 0.000 0.020 0.056 0.029 Stable Diffusion 3.5 Large [60] 19.459 38.711 248.280 0.432 0.255 0.005 0.749 0.740 0.512 Flux.1-dev [38] 26.855 44.189 227.565 0.223 0.161 0.000 0.387 0.577 0.430 CogView4 [78] 17.173 29.437 193.420 0.200 0.150 0.010 0.437 0.593 0.517 SANA-1.5 1.6B (PAG) [73] 31.573 61.566 288.225 0.059 0.003 0.000 0.143 0.090 0.031 SANA-1.5 4.8B (PAG) [73] 25.027 55.634 268.025 0.086 0.000 0.000 0.228 0.079 0.030 Lumina-Image 2.0 [47] 26.259 54.547 270.530 0.059 0.013 0.000 0.199 0.199 0.083 HiDream-I1-Full [29] 14.464 31.026 177.765 0.195 0.166 0.005 0.435 0.602 0.576

- Imagen3 [36] 34.005 83.171 239.565 0.068 0.071 0.000 0.279 0.447 0.371 Recraft V3 [66] 17.050 26.945 74.181 0.050 0.061 0.006 0.267 0.371 0.430 Kolors 2.0 [64] 18.236 37.930 235.953 0.125 0.118 0.000 0.374 0.483 0.257 Seedream 3.0 [24] 7.204 20.596 169.307 0.699 0.451 0.109 0.893 0.822 0.688

- Imagen4 [17] 18.273 42.918 121.625 0.181 0.184 0.060 0.364 0.575 0.745 GPT-4o [45] 18.255 37.850 85.430 0.150 0.171 0.070 0.323 0.495 0.673

any mismatch in capitalization (e.g., uppercase vs. lowercase) is counted as one unit of edit distance. This rule has a direct impact on GPT-4o’s overall rendering scores.

Compared to Imagen3 [36], Imagen4 [17] shows a clear improvement in text rendering, with nearly double the performance on metrics such as ED and CR. Nonetheless, its overall visual appeal and the clarity of rendered text for long prompts still leave room for enhancement.

Notably, Recraft V3 [66] consistently achieves excellent edit distance performance across all prompt lengths, with its values for long prompts showing a large gap from the second-best result. However, its performance in completion ratio and word accuracy count is relatively less impressive. This discrepancy may be attributed to its layout-first strategy [65], in which a layout is generated prior to text insertion. This approach effectively reduces severe errors and prevents chaotic outputs by decomposing the original text rendering task into several relatively simpler subtasks, thereby significantly enhancing the reliability of text rendering in the final images.

### 4.2.3 Knowledge and Reasoning

In the knowledge and reasoning dimension, shown in Table 6 and in Figure 4, GPT-4o [45] demonstrates substantially stronger capabilities than other models. It consistently outperforms its counterparts in both knowledge retention and reasoning ability across nearly all subject categories evaluated. In general, closed-source models outperform open-source models in knowledge and reasoning capabilities. Notably, no single model has shown a particularly outstanding performance in specific subjects, indicating that the reasoning abilities of current models are largely derived from a balanced and conventional training dataset.

The models’ reasoning abilities can be categorized into five tiers as follows: the first tier includes GPT-4o [45], Imagen4 [17], the second tier consists of Recraft V3 [66], HiDream-I1-Full [29], and

- Imagen3 [36], the third tier includes only Stable Diffusion 3.5 Large [60], and the fourth tier includes Seedream 3.0 [24], Lumina-Image 2.0 [47], and Kolors 2 [64], other models form the last tier. The following figure visualizes the reasoning scores, which correspond closely with the aforementioned ranking.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Seedream 3.0 GPT-4o Imagen4 Recraft V3 HiDream-I1-Full

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

ED:22 CR:0 WAC:1.0 ED:43 CR:0 WAC:0.58 ED:30 CR:0 WAC:0.96 ED:17 CR:0 WAC:0.66 ED:38 CR:0 WAC:0.75

PROMPT: Designed for tech-savvy young professionals, this modern and vibrant advertisement features Smartphone over a gradient blue and purple background. The text elements include "Experience Speed Like Never Before" as the main hook, "Unlock a new level of convenience." for added detail, "Early Bird Discount Ends Soon." to highlight the offer, and "Download the app and get started today." driving action. The logo "TechNova" ties it all together in a visually appealing manner.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

GPT-4o Imagen4 Recraft V3 HiDream-I1-Full Imagen3

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

0.32

0.35 0.33 0.33 0.31

PROMPT: Draw a diagram showcasing the movements in plate tectonics. Tips: Continental drift, subduction zones, tectonic plates.

- Figure 4: An illustration of the generation results and the corresponding scores. The first row shows the text rendering results and the second row shows the the knowledge and reasoning results. And the evaluation scores are displayed in the upper left corner of the image.

- Table 6: The Knowledge and Reasoning evaluation results. indicate the first, second, third, fourth, and fifth performance respectively.

Method Geography Computer Science Biology Mathmatics Physics Chemistry Common Sense

Janus-Pro [13] 0.153 0.134 0.123 0.130 0.144 0.131 0.189 BLIP3-o [8] 0.210 0.225 0.219 0.223 0.219 0.232 0.275 BAGEL [18] 0.191 0.152 0.183 0.149 0.168 0.169 0.248 BAGEL+CoT [18] 0.206 0.184 0.206 0.200 0.212 0.208 0.273 Show-o2-1.5B [75] 0.218 0.197 0.235 0.201 0.222 0.206 0.295 Show-o2-7B [75] 0.222 0.206 0.244 0.217 0.227 0.212 0.297 OmniGen2 [71] 0.266 0.270 0.271 0.284 0.257 0.278 0.314

Stable Diffusion 1.5 [55] 0.217 0.203 0.211 0.212 0.207 0.185 0.251 Stable Diffusion XL [46] 0.246 0.216 0.244 0.235 0.234 0.239 0.289 Stable Diffusion 3.5 Large [60] 0.291 0.299 0.283 0.306 0.292 0.292 0.319 Flux.1-dev [38] 0.239 0.257 0.247 0.265 0.247 0.253 0.298 CogView4 [78] 0.223 0.251 0.237 0.279 0.239 0.252 0.296 SANA-1.5 1.6B(PAG) [73] 0.207 0.203 0.222 0.218 0.211 0.193 0.280 SANA-1.5 4.8B(PAG) [73] 0.214 0.206 0.224 0.224 0.214 0.203 0.282 Lumina-Image 2.0 [47] 0.208 0.206 0.225 0.222 0.215 0.197 0.278 HiDream-I1-Full [29] 0.324 0.324 0.305 0.312 0.318 0.305 0.347

- Imagen3 [36] 0.304 0.319 0.298 0.303 0.320 0.315 0.338 Recraft V3 [66] 0.323 0.337 0.303 0.320 0.319 0.328 0.344 Kolors 2.0 [64] 0.255 0.252 0.256 0.263 0.258 0.277 0.314 Seedream 3.0 [24] 0.246 0.295 0.313 0.253 0.297 0.270 0.277

- Imagen4 [17] 0.334 0.346 0.314 0.343 0.342 0.346 0.351 GPT-4o [45] 0.351 0.348 0.323 0.334 0.350 0.355 0.364

- 5 Conclusion

We introduce a comprehensive text-to-image benchmark, namely OneIG-Bench, which establishes a systematic framework for omni-dimensional nuanced evaluation through categorization of generation themes. Specifically, we have meticulously designed general scenarios including human figures and conventional objects, text rendering scenarios, and anime/style scenarios, and have crafted evaluation metrics for each scenario to comprehensively measure text-to-image performance. By decomposing evaluation into these discrete dimensions, the benchmark facilitates in-depth comparative analysis of models’ strengths and limitations. This approach not only provides researchers with a rigorous evaluation framework but also serves as a guiding tool for identifying technical bottlenecks and prioritizing methodological innovations in the field.

Limitation: While this study presents a novel and systematic benchmark, several limitations should be acknowledged. (1) Knowledge and reasoning represents a relatively novel task in the image generation domain, and most existing models currently lack robust reasoning capabilities. While we have confirmed that our metric rankings align closely with human evaluations, there may exist more rational and effective evaluation approaches yet to be explored. (2) Moreover, aesthetic models tend to exhibit unexpected biases, while body quality assessment models often lack sufficient discriminative power and generalizability. We will further investigate both dimensions to develop a more robust and precise evaluation method.

## Acknowledgements

We are particularly grateful to Bizhu Huang, Shuli Gao, Kang An, and Wen Sun for their invaluable support throughout the course of this research.

## References

- [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015.
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [5] Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, and Mohamed Elhoseiny. Hrs-bench: Holistic, reliable and scalable benchmark for text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20041–20053, 2023.
- [6] black-forest labs. The official api of flux-1.dev. https://api.us1.bfl.ai/scalar#tag/tasks/POST /v1/flux-dev, 2024.
- [7] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attentionbased semantic guidance for text-to-image diffusion models. ACM transactions on Graphics (TOG), 42(4):1–10, 2023.
- [8] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [9] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024.
- [10] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [11] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019.
- [12] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020.
- [13] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [14] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.
- [15] Bowen Cheng, Anwesa Choudhuri, Ishan Misra, Alexander Kirillov, Rohit Girdhar, and Alexander G Schwing. Mask2former for video instance segmentation. arXiv preprint arXiv:2112.10764, 2021.
- [16] Jaemin Cho, Yushi Hu, Jason Baldridge, Roopal Garg, Peter Anderson, Ranjay Krishna, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In ICLR, 2024.
- [17] Google deepmind Imagen4 team. Imagen4. https://storage.googleapis.com/deepmind-media /Model-Cards/Imagen-4-Model-Card.pdf, 2025.
- [18] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [19] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [20] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. arXiv preprint arXiv:2212.05032, 2022.
- [21] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data, 2023.
- [22] Xingyu Fu, Muyu He, Yujie Lu, William Yang Wang, and Dan Roth. Commonsense-t2i challenge: Can text-to-image generation models understand commonsense? arXiv preprint arXiv:2406.07546, 2024.
- [23] Rohit Gandikota and David Bau. Distilling diversity and control in diffusion models. arXiv preprint arXiv:2503.10637, 2025.
- [24] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.
- [25] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [26] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139– 144, 2020.
- [27] Shuhao Han, Haotian Fan, Jiachen Fu, Liang Li, Tao Li, Junhui Cui, Yunqiu Wang, Yang Tai, Jingwei Sun, Chunle Guo, and Chongyi Li. Evalmuse-40k: A reliable and fine-grained benchmark with comprehensive human annotations for text-to-image generation model evaluation, 2024.
- [28] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [29] HiDream-ai. Hidream-i1. https://github.com/HiDream-ai/HiDream-I1, 2025.
- [30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [31] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.
- [32] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20406–20417, 2023.
- [33] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [34] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.
- [35] Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Liang Hu, Qi Dai, Xiyang Dai, Dongdong Chen, Chong Luo, et al. Llm2clip: Powerful language model unlock richer visual representation. arXiv preprint arXiv:2411.04997, 2024.
- [36] Imagen-Team-Google. Imagen 3, 2024.
- [37] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-apic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:36652–36663, 2023.
- [38] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [39] Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, et al. Genai-bench: Evaluating and improving compositional text-to-visual generation. arXiv preprint arXiv:2406.13743, 2024.
- [40] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation, 2024.
- [41] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.
- [42] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

- [43] Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.
- [44] OpenAI. Gpt-4o system card. https://openai.com/index/gpt-4o-system-card/, 2024.
- [45] OpenAI. Introducing 4o image generation. https://openai.com/index/introducing-4o-image

-generation/, 2025.

- [46] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [47] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Will Beddow, Erwann Millon, Wenhai Wang Victor Perez, Yu Qiao, Bo Zhang, Xiaohong Liu, Hongsheng Li, Chang Xu, and Peng Gao. Lumina-image 2.0: A unified and efficient image generative framework, 2025.
- [48] Shi Qiu, Shaoyang Guo, Zhuo-Yang Song, Yunbo Sun, Zeyu Cai, Jiashen Wei, Tianyu Luo, Yixuan Yin, Haoxu Zhang, Yi Hu, et al. Phybench: Holistic evaluation of physical perception and reasoning in large language models. arXiv preprint arXiv:2504.16074, 2025.
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [51] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [52] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [53] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [54] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.
- [55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.
- [56] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [57] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pages 30105–30118. PMLR, 2023.
- [58] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024.
- [59] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [60] Stability-AI. stable-diffusion-3.5-large. https://github.com/Stability-AI/sd3.5, 2024.
- [61] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [62] Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, Mengping Yang, Cheng Zhang, and Hao Li. Evalalign: Evaluating text-to-image models through precision alignment of multimodal large models with supervised fine-tuning to human annotations. arXiv e-prints, pages arXiv–2406, 2024.

- [63] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [64] Kuaishou Kolors team. Kolors2.0. https://app.klingai.com/cn/, 2025.
- [65] Recraft team. How to create sota image generation with text recrafts ml team insights. https://www.re craft.ai/blog/how-to-create-sota-image-generation-with-text-recrafts-ml-team-i nsights, 2024.
- [66] Recraft team. Recraft v3. https://www.recraft.ai/blog/recraft-introduces-a-revolutio nary-ai-model-that-thinks-in-design-language?utm_source=ai-bot.cn, 2024.
- [67] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [68] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [69] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.
- [70] Olivia Wiles, Chuhan Zhang, Isabela Albuquerque, Ivana Kaji´c, Su Wang, Emanuele Bugliarello, Yasumasa Onoe, Pinelopi Papalampidi, Ira Ktena, Chris Knutsen, et al. Revisiting text-to-image evaluation with gecko: On metrics, prompts, and human ratings. arXiv preprint arXiv:2404.16820, 2024.
- [71] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [72] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [73] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer, 2025.
- [74] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [75] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.
- [76] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766, 2024.
- [77] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [78] THUKEG Z.ai. Cogview4. https://github.com/THUDM/CogView4, 2025.
- [79] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [80] Daoan Zhang, Che Jiang, Ruoshi Xu, Biaoxiang Chen, Zijian Jin, Yutian Lu, Jianguo Zhang, Liang Yong, Jiebo Luo, and Shengda Luo. Worldgenbench: A world-knowledge-integrated benchmark for reasoning-driven text-to-image generation, 2025.

## A Appendix

### A.1 Word Count Distribution Statistics

The word count distribution of our OneIG-Bench prompts, as shown in Table 7 and Figure 5, follows a Short : Middle : Long ratio of approximately 1:2:1. The choice of 30 and 60 words as the thresholds for distinguishing short, middle, and long prompts is based on the following reasoning: According to common rules for understanding token lengths, 1 word is approximately equal to 4/3 tokens, and 1-2 sentences are roughly equivalent to 30 tokens. This means that a simple 1-2 sentence prompt has a length of about 20-25 words. To ensure diversity in sentence structure and accuracy in stylization or portraiture in the image, we set the boundaries for short and middle prompts at 30 words. Furthermore, since some text encoders, such as CLIP [50], SigLIP [79], support a maximum of 77 tokens, a prompt of up to 60 words can generally be processed directly by these encoders.

- Table 7: Word count distribution of OneIG-Bench prompts in different categories. In the table, Avg represents the average word count of prompts in different categories (including total and total w/o Knowledge & Reasoning). Short, Medium and Long represent the length of the prompts, where Short denote the number of words is less than 30, Medium denotes the number between 30 and 60, and Long denotes the number exceeding

60. "K & R" is the abbreviation for "Knowledge & Reasoning".

Category Avg Short Middle Long Portrait 56.4 0.184 0.443 0.373 General Object 46.5 0.330 0.422 0.248 Anime & Stylization 50.6 0.212 0.522 0.265 Text Rendering 51.2 0.275 0.475 0.250 Knowledge & Reasoning 20.5 0.960 0.018 0.022 Total Distribution 45.2 0.389 0.377 0.234 Total Distribution w/o K & R 51.3 0.246 0.467 0.287

The Portrait category, however, shows a slight deviation from this 1:2:1 distribution in Figure 6 due to the explicit requirement for portraits in the prompts, ensuring that the generated characters do not include stylized figures like those found in anime. As a result, the average word count for prompts in this category is higher than in other categories. On the other hand, the Knowledge & Reasoning category, which focuses on reasoning tasks, does not revise the prompts to conform to the word count ratio, leading to a noticeably lower average word count compared to other categories. In general, excluding the Knowledge & Reasoning category, OneIG-Bench prompts’ word count results align closely with the 1:2:1 ratio.

All Categories Excluding Knowledge & Reasoning

175

| |
|---|

150

125

Frequency

100

75

50

25

0

0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200

Word Count

- Figure 5: Word Count of the Overall Prompts of OneIG-Bench. The word count distribution of OneIGBench’s prompts ranges from 0 to 200.

1.00

| |Portrait<br><br>General Object<br><br>Anime & Stylization<br><br>Text Rendering<br><br>Knowledge & Reasoning| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.75

Proportion

0.50

0.25

0.00

Short Middle Long

Prompt Length

### Figure 6: The distribution of prompt word counts across Short, Middle, Long categories.

- A.2 Implementation

Our experiments on image generation with unified multimodal methods and open-source methods are configured according to Table 8. For all methods, the CFG and step parameters follow the methods’ default settings. To ensure consistency and ensure the quality of image generation, we increased the default steps for Stable Diffusion 3.5 Large [60] from 40 to 50. The number of inference steps for Flux.1-dev [38] is set to be consistent with that used in the official API [6]. With the exception of Stable Diffusion 1.5 [55], which cannot generate images with a resolution of 1024 × 1024, all other methods are configured to generate images at a resolution of 1024 × 1024.

Table 8: Configurations for unified multimodal and open-source methods. Size represents the parameter size of the corresponding method. CFG represents the guidance scale of the corresponding method. Resolution represents the resolution of the image generated by the corresponding method. Step represents the number of inference steps during the image generation process.

Method Size CFG Resolution Step

Janus-Pro [13] 7B 5.0 384 × 384 BLIP3-o [8] 8B 3.0 1024 × 1024 BAGEL [18] 7B 4.0 1024 × 1024 50 BAGEL+CoT [18] 7B 4.0 1024 × 1024 50 Show-o2-1.5B [75] 1.5B 5.0 432 × 432 50 Show-o2-7B [75] 7B 5.0 432 × 432 50 OmniGen2 [71] 3B+4B 4.0 1024 × 1024 50

Stable Diffusion 1.5 [55] 0.9B 7.5 512 × 512 50 Stable Diffusion XL [46] 2.6B 5.0 1024 × 1024 50 Stable Diffusion 3.5 Large [60] 8.1B 4.5 1024 × 1024 50 Flux.1-dev [38] 12B 3.5 1024 × 1024 28 CogView4 [78] 6B 3.5 1024 × 1024 50 SANA-1.5 1.6B (PAG) [73] 1.6B 5.0 1024 × 1024 20 SANA-1.5 4.8B (PAG) [73] 4.8B 5.0 1024 × 1024 20 Lumina-Image 2.0 [47] 2.6B 4.0 1024 × 1024 50 HiDream-I1-Full [29] 17B 5.0 1024 × 1024 50

For closed-source methods, we present the corresponding release or update dates of the methods in Table 9 to facilitate alignment with subsequent experimental results.

Table 9: Release/Update date of closed-source methods. Release/Update date represents the version of the corresponding method when generating images.

Method Imagen3 [36] Recraft V3 [66] Kolors 2.0 [64] Seedream 3.0 [24] Imagen4 [17] GPT-4o [45] Release/Update Date 2025-01-23 2024-10-30 2025-04-15 2025-04-15 2025-05-20 2025-04-29

- A.3 The Details on Prompts Rewriting

Algorithm 1: Initial Prompts Rewritten by GPT-4o Input: n: the length of the initial prompts list,

Pinit: the initial prompts [p1,p2,...,pn]. Output: Prewritten : the rewritten prompts [p′1,p′2,...,p′n] Psorted ← sorted_by_word_count(Pinit) R ← 100 ∗ sorted(beta.rvs(2.37,2.86,n)) for i ← 1 to n do

initial_prompt ← Psorted[i] target_word_count ← R[i] rewritten_prompt ← GPT-4o_API(Prompt Template, initial_prompt, target_word_count)

Prewritten[i] ← rewritten_prompt return Prewritten

As shown in Algorithm 1, the initial prompts are first sorted based on their word count. Then, using a Beta distribution with parameters (2.37, 2.86), which roughly follows the ratio 0-0.3:0.3-0.6:0.6-1

≈ 1:2:1, a list of desired prompt lengths is generated and subsequently sorted. The sorted prompts are then matched with the corresponding desired lengths, and the GPT-4o API is called to rewrite each prompt according to its specified length, resulting in the rewritten prompts. Without loss of generality, lengths in the range of 60-100 can be mapped to a range of greater, thereby generating longer prompts. In this process, the corresponding prompt for rewriting is as follows:

Prompt of Prompts Rewriting You are a precise rewriting assistant. Task Description:

- • Rewrite the <initial_prompt > according to the <target_word_count> of the prompt.
- • For <initial_prompt > longer than the <target_word_count>

- – Shorten the prompt by carefully removing specific but non-essential details.
- – Do not simply delete words or generalize the description.

- • For <initial_prompt > shorter than the <target_word_count>

- – Expand the prompt by adding specific, meaningful, and vivid details that enhance the scene.
- – Do not introduce abstract or generalized commentary.

- • Ensure the rewritten prompt

- – The prompt should be coherent, natural, fluent, logically structured.
- – Please maintain the initial tone and intent as much as possible.

### Important:

• Only output the final rewritten prompt without any additional words.

### A.4 The Details and Analysis on Stylization Table 10: The styles in OneIG-Bench corresponding to specific categories.

Category Style

Traditional abstract expressionism, art nouveau, Baroque, Chinese ink painting, cubism, fauvism, impressionism, line art, minimalism, pointillism, pop art, Rococo, Ukiyo-e

Media clay, crayon, graffiti, LEGO, pencil sketch, stone sculp-

ture, watercolor Anime Celluloid, Chibi, comic, Cyberpunk, Ghibli, Impasto,

Pixar, pixel art, 3d rendering,

The Anime & Stylization category encompasses a variety of styles, which are systematically grouped into three subcategories in Table 10: Traditional, Media, and Anime. The Traditional category primarily includes styles rooted in classical and historical art movements from around the world. The Media category includes styles defined by specific artistic media and material-based techniques. The Anime category represents a collection of stylized and detailed visual aesthetics commonly associated with animation and pop culture. And Table 11 presents the scores of different methods across various style categories. The calculation process is as follows: for each method, the average style score is first calculated based on the images generated for each prompt within each style. Then, the score for each style category is obtained by averaging the scores of the individual styles within that category. It is clear that GPT-4o [45] demonstrates exceptional style-following ability across most categories, significantly outperforming other methods.

### A.5 Visualization Results

We first give the best normalized polar visualization of SOTA methods on OneIG-Bench in Table 12. It can be observed that the closed-source methods outperform the other two categories of methods overall. And we selected some methods based on their overall performance across non-style metrics and showcased representative examples for each. For the style dimension, we further selected images

- Table 11: The style scores on different categories of styles. indicate the first, second, third, fourth, and fifth performance, respectively.

###### Method Traditional Media Anime

Janus-Pro [13] 0.224 0.212 0.412 BLIP3-o [8] 0.381 0.287 0.390 BAGEL [18] 0.363 0.297 0.440 BAGEL+CoT [18] 0.372 0.329 0.499 Show-o2-1.5B [75] 0.269 0.288 0.410 Show-o2-7B [75] 0.283 0.295 0.380 OmniGen2 [71] 0.360 0.314 0.458

Stable Diffusion 1.5 [55] 0.483 0.298 0.349 Stable Diffusion XL [46] 0.316 0.307 0.339 Stable Diffusion 3.5 Large [60] 0.356 0.315 0.335 Flux.1-dev [38] 0.367 0.298 0.391 CogView4 [78] 0.376 0.294 0.369 SANA-1.5 1.6B(PAG) [73] 0.438 0.331 0.370 SANA-1.5 4.8B(PAG) [73] 0.443 0.340 0.379 Lumina-Image 2.0 [47] 0.351 0.325 0.360 HiDream-I1-Full [29] 0.331 0.295 0.368

- Imagen3 [36] 0.378 0.309 0.371 Recraft V3 [66] 0.418 0.347 0.332 Kolors 2.0 [64] 0.370 0.336 0.360 Seedream 3.0 [24] 0.383 0.365 0.524

- Imagen4 [17] 0.336 0.365 0.452 GPT-4o [45] 0.532 0.404 0.411

from three finer-grained subcategories to illustrate the results. In all visualizations, the methods are broadly selected based on overall performance. In the following figures, each image tile is labeled in the top-left corner with the score achieved by the corresponding method under the current evaluation metric.

Figure 7 shows that Imagen4 [17] and GPT-4o [45] demonstrate strong capabilities in semantic alignment. Notably, in multi-person generation tasks, many methods struggle to accurately fulfill requirements at the individual level and often exhibit confusion in assigning attributes to the correct subjects. In addition, some methods tend to overlook fine-grained details while focusing on the primary generation task, which significantly hinders their ability to achieve high alignment scores.

Text rendering is an important task for current generative methods. As shown in Figure 8, Seedream 3.0 [24] demonstrates high accuracy and aesthetic quality. Some methods, such as GPT-4o [45], demonstrate limitations in adhering to case sensitivity (distinguishing between uppercase and lowercase letters), which compromises the textual accuracy of the rendered content. While the generated images may appear visually impressive, these subtle errors can lead to a noticeable divergence between subjective visual quality and objective evaluations based on metrics such as ED and WAC.

- Imagen4 [17] demonstrates good accuracy in text generation, but the overall visual quality of the images is relatively poor. Recraft V3 [66], although rarely producing major errors in text generation, tends to make mistakes at the word level and suffers from inconsistencies in typography and layout coherence. Overall, HiDream-I1-Full [29] performs reasonably well in text rendering—it may not outperform Seedream 3.0, Recraft V3, or GPT-4o, but it is able to fulfill the basic prompt requirements.

From a reasoning perspective, only GPT-4o [45] demonstrates both logical coherence and textual accuracy in Figure 9. Although Imagen4 [17] and Recraft V3 [66] fall short of GPT-4o in terms of clarity and correctness, they produce text that is generally readable. HiDream-I1-Full [29] provides limited textual and visual content but manages to convey a certain degree of knowledge and reasoning, albeit with insufficient accuracy. In contrast, Imagen3 [36] tends to generate overly redundant outputs, with excessive textual and graphical elements that obscure the intended message, and often includes incorrect information. Therefore, Knowledge and Reasoning remains a critical area that warrants further investigation and refinement for generative methods.

The visualization of diversity results is presented in Figure 10, where the ranking of diversity scores aligns well with visual inspection, supporting the validity of our proposed diversity metrics, although the diversity observed in some methods may partly stem from insufficient alignment capabilities.

Figures 11, 12, and 13 show that GPT-4o [45] performs well across most styles, though it struggles with specific ones especially some anime styles. Stable Diffusion 1.5 [55], despite its lower visual quality, effectively captures traditional style features. Although Imagen4 [17] does not achieve a high overall score in style, it performs notably well in media and anime styles. It is worth noting that unified multimodal methods, such as BAGEL+CoT [18], OmniGen2 [71], BAGEL [18] and Janus-Pro [13], exhibit particularly strong performance in the anime style. Overall, Seedream 3.0[24] and SANA-1.5 4.8B (PAG)[73] also demonstrate strong stylistic consistency, ranking just behind GPT-4o.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

###### GPT-4o Imagen3 HiDream-I1-Full Kolors 2.0

###### Imagen4

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- 1.00 0.88 0.50 0.68

- 1.00 0.89 1.00 0.89

0.89

PROMPT: 1girl, 2boys, multiple boys, anime, blonde hair, purple hair, sunglasses, logo.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

1.00

PROMPT: Create an image of a blob fish in a surreal underwater garden with vibrant corals and unusual sea creatures.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

1.00

1.00 0.95 0.95 0.85

PROMPT: An elderly woman shares cherished childhood stories with a robot, displaying a collection of faded photographs meticulously arranged in an album. Accompanying her are a young woman and an inquisitive boy, all portrayed as lifelike individuals, capturing the essence of real-life interactions through a realistic photographic style.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

1.00

0.85 0.75 0.80 0.65

PROMPT: Create a vivid and realistic depiction of ancient farmers laboring tirelessly in expansive fields beneath a scorching, intense sun. Capture the determination and hard work etched into their facial expressions. Accentuate the intricate details of their traditional clothing, sturdy farming tools, and the lush crops that thrive around them. Highlight the sweat glistening on their brows and dripping onto the soil, emphasizing their relentless dedication in a vibrant, warm environment.

- Figure 7: Visualization results of SOTA methods. Alignment of Imagen4 [17], GPT-4o [45], Imagen3 [36], HiDream-I1-Full [29] and Kolors 2.0 [64] are evaluated respectively. Row 1 corresponds to tag/phrase prompt: The variation in the scores of the visual samples are mainly influenced by: "2 boys" and "logo". Row 2 corresponds to short prompt: The variation in the scores of the visual samples are mainly influenced by: "blob fish", "surreal underwater" and "unusual sea creatures". Row 3 corresponds to middle prompt: The variation in the scores of the visual samples are influenced by: "inquisitive boy", "realistic photography style" and whether each individual mentioned in the prompt is accurately and uniquely generated in the image. Row 4 corresponds to long prompt: The variation in the scores of the visual samples are influenced by: "ancient farmers", "sweat", "facial expressions", "relentless dedication". The mentioned keywords may correspond to more than one question–answer pair.

- Table 12: Best normalized polar visualization of SOTA methods on OneIG-Bench. Anti-Clockwisely,

Alignment:

Long; Text:

ED-Long, CR-Short,

NP,

T&P,

Short,

Middle,

ED-Short,

ED-Middle,

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

WAC-Long; Reasoning:

Geography, Computer Science,

CR-Middle,

CR-Long,

WAC-Short,

WAC-Middle,

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

Common Sense; Style: Traditional,

Biology,

Mathematics,

Physics,

Chemistry,

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

Anime; Diversity:

Long. The absolute average values of each evaluation metric are list on legends.

Media,

NP,

T&P,

Short,

Middle,

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

Janus-Pro BLIP3-o BAGEL

Alignment: 0.553 Text: 0.001

Alignment: 0.711 Text: 0.013

Alignment: 0.769 Text: 0.244

| |
|---|

| |
|---|

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.276 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.361 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.367 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.139

0.223

0.173

0.365

0.229

0.251

BAGEL+CoT Show-o2-1.5B(432) Show-o2-7B(432)

Alignment: 0.793 Text: 0.020

Alignment: 0.798 Text: 0.002

Alignment: 0.817 Text: 0.002

| |
|---|

| |
|---|

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.390 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.317 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.317 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.206

0.219

0.226

0.209

0.186

0.177

OmniGen2 Stable Diffusion 1.5 Stable Diffusion XL

Alignment: 0.804 Text: 0.680

Alignment: 0.565 Text: 0.010

Alignment: 0.688 Text: 0.029

| |
|---|

| |
|---|

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.377 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.383 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.332 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.271

0.207

0.237

0.242

0.429

0.296

Stable Diffusion 3.5 Large Flux.1-dev CogView4

Alignment: 0.809 Text: 0.629

Alignment: 0.786 Text: 0.523

Alignment: 0.786 Text: 0.641

| |
|---|

| |
|---|

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.353 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.368 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.353 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.294

0.253

0.246

0.225

0.238

0.205

Table 12: [Continued] Best normalized polar visualization of SOTA methods on OneIG-Bench. AntiClockwisely, Alignment:

Long; Text:

ED-Middle, ED-Long,

NP,

T&P,

Short,

Middle,

ED-Short,

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

WAC-Long; Reasoning: Geography,

CR-Short,

CR-Middle,

CR-Long,

WAC-Short,

WAC-Middle,

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

Common Sense; Style:

Computer Science,

Biology,

Mathematics,

Physics,

Chemistry,

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

Anime; Diversity:

Long. The absolute average values of each evaluation metric are list on legends.

Traditional,

Media,

NP,

T&P,

Short,

Middle,

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

SANA-1.5 1.6B (PAG) SANA-1.5 4.8B (PAG) Lumina-Image 2.0

Alignment: 0.762 Text: 0.054

Alignment: 0.765 Text: 0.069

Alignment: 0.819 Text: 0.106

| |
|---|

| |
|---|

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.387 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.401 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.354 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.209

0.217

0.270

0.222

0.216

0.216

HiDream-I1-Full Imagen3 Recraft V3

Alignment: 0.829 Text: 0.707

Alignment: 0.843 Text: 0.343

Alignment: 0.810 Text: 0.795

| |
|---|

| |
|---|

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.347 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.359 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.378 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.317

0.313

0.323

0.186

0.188

0.205

Kolors 2.0 Seedream 3.0 Imagen4

Alignment: 0.820 Text: 0.427

Alignment: 0.818 Text: 0.865

Alignment: 0.857 Text: 0.805

| |
|---|

| |
|---|

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.360 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.413 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.377 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.262

0.275

0.338

0.300

0.277

0.199

GPT-4o

Alignment: 0.851 Text: 0.857

| |
|---|

| |0.2<br><br>0.6<br><br>1.0<br><br>| |
|---|
<br><br>Reasoning: Style: 0.462 Diversity:<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |

0.345

0.151

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

###### Seedream 3.0 GPT-4o Imagen4 Recraft V3 Hidream-I1-Full

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

###### ED:0 CR:1 WAC:1.00 ED:0 CR:1 WAC:1.00 ED:24 CR:0 WAC:0.25 ED:0 CR:1 WAC:1.00

###### ED:9 CR:0 WAC:0.75

PROMPT: A red umbrella connects two hearts on a rainy street. Center text: "Rainy Encounter: Love Blossoms in a Chance Meeting".

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

###### ED:0 CR:1 WAC:1.00 ED:0 CR:1 WAC:1.00 ED:0 CR:1 WAC:1.00 ED:1 CR:0 WAC:0.50 ED:1 CR:0 WAC:0.50

PROMPT: Display "Not again..." inside a visible text box, hovering above a mage casting a swirling spell in a fantasy setting.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

###### ED:0 CR:1 WAC:1.00 ED:57 CR:0 WAC:0.46 ED:0 CR:1 WAC:1.00 ED:10 CR:0 WAC:0.65 ED:39 CR:0 WAC:0.62

PROMPT: Capture the festive spirit of a lantern-lit street market glowing with vibrant colors during Lunar New Year celebrations. The poster highlights "Spring Festival Fair 2025" in elegant, traditional script. Beneath, a cheerful slogan says "Celebrate with dazzling lights, authentic flavors, and timeless traditions." Event details follow: "February 1 to 10 at Old Town Plaza — Free entry for all visitors."

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

###### ED:1 CR:0 WAC:0.88 ED:26 CR:0 WAC:0.13 ED:21 CR:0 WAC:0.50 ED:11 CR:0 WAC:0.38 ED:27 CR:0 WAC:0.75

PROMPT: Design a creative square business card for a freelance photographer, featuring a monochrome palette and a camera icon watermark. Highlight "Emma Zhao", add "Visual Storyteller", include "hello@emmazhao.com", and phone "+44 7700 900123".

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

###### ED:2 CR:0 WAC:0.93 ED:0 CR:1 WAC:1.00 ED:14 CR:0 WAC:0.93 ED:18 CR:0 WAC:0.90 ED:24 CR:0 WAC:0.93

PROMPT: Imagine a PPT slide where Global Tourism Recovery takes center stage against a clean white space accented by soft geometric shapes. The title reads "Global Tourism Recovery" and is complemented by a paragraph stating "Post-pandemic travel is witnessing a surge, driven by digital nomadism, eco-tourism, and flexible booking options.". A visual chart labeled "Tourist Arrivals by Continent" includes categories like "Europe", "Asia", and "Americas". Decorative icons such as an airplane, a suitcase, and a globe add context. The slide concludes with a footer note: "World Tourism Organization, 2025".

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

ED:126 CR:0 WAC:0.50 ED:70 CR:0 WAC:0.59 ED:66 CR:0 WAC:0.72 ED:50 CR:0 WAC:0.72 ED:76 CR:0 WAC:0.75

PROMPT: A bold presentation visualizing The Impact of 5G Technology with a pastel-colored layout with modern design cues. The title reads "The Impact of 5G Technology" and is complemented by a paragraph stating "5G networks are enabling faster connectivity, supporting innovations in autonomous vehicles, remote surgeries, and smart devices.". A visual chart labeled "5G Coverage Expansion" includes categories like "Urban Areas", "Suburban", and "Rural". Decorative icons such as a 5G icon, a satellite dish, and a smartphone add context. The slide concludes with a footer note: "Telecom Industry Report, 2025".

- Figure 8: Visualization results of SOTA methods. Text of Seedream 3.0 [24], GPT-4o [45], Imagen4 [17], Recraft V3 [66], HiDream-I1-Full [29] are evaluated respectively. Row 1, 2 correspond to short prompts. Row 3, 4 correspond to middle prompts. Row 5, 6 correspond to long prompts.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

###### GPT-4o Imagen4 Recraft V3 HiDream-I1-Full Imagen3

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

PROMPT: How about designing a diagram that illustrates how coral reefs are formed? Tips: coral polyps, limestone skeleton.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

- 0.39 0.37 0.35 0.35 0.34

0.36 0.34 0.32 0.31 0.29

PROMPT: Tha task is to give the structural diagrams of butanone and butane-2,3-diol. Tips: butanone, butane-2,3-diol

- 0.40 0.36 0.34 0.34 0.29

PROMPT: Visually compare series and parallel circuits with a diagram. Tips: Current flow, Resistors, Voltage distribution.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

- Figure 9: Visualization results of SOTA methods. Reasoning of GPT-4o [45], Imagen4 [17], Recraft V3 [66], HiDream-I1-Full [29] and Imagen3 [36] are evaluated respectively. Row 1 aims to illustrate how coral reefs are formed, highlighting key steps such as the growth of coral polyps and the gradual accumulation of calcium carbonate structures. Row 2 demonstrates the circuit diagrams of series and parallel circuits. Row 3 focuses on the structural diagrams of butanone and butane-2,3-diol.

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

PROMPT: Please create a realistic photograph capturing a woman in a brown shirt and jeans, joyfully smiling as she lies on a bed of fresh snowflakes. Her hair is gently tousled, and the scene is set against a backdrop of towering evergreen trees.

PROMPT: We need you to show a diagram depicting Bernoulli's principle in fluid mechanics. Tips: Fluid dynamics, Pressure, Velocity.

0.74 0.55 0.52 0.38

0.37 0.35 0.30 0.24

0.60

0.31

[Figure 164]

Stable Diffusion 1.5 Janus-Pro Kolors 2.0 Stable Diffusion XL Seedream 3.0

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

- Figure 10: Visualization results of SOTA methods. Diversity of Stable Diffusion 1.5 [55], Janus-Pro [13], Kolors 2.0 [64], Stable Diffusion XL [46] and Seedream 3.0 [24] are evaluated respectively.

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

###### GPT-4o Stable Diffusion 1.5 SANA-1.5 4.8B (PAG) SANA-1.5 1.6B (PAG) Recraft V3

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

0.71 0.70 0.54 0.53 0.52

PROMPT: A smiling figure dressed in a vibrant blue outfit holds a sunny demeanor. One eye is playfully shut. The entire scene is depicted in pointillism style.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

0.57 0.51 0.36 0.36 0.40

PROMPT: A character with distinct sharp facial features, pierced ear, wavy hair, colorful jacket, unique tie, glove holding cigarette, watch on wrist, contrasted deep blue background. The scene is depicted in minimalism.

- Figure 11: Visualization results of SOTA methods. Traditional styles of GPT-4o [45], Stable Diffusion 1.5 [55], SANA-1.5 4.8B (PAG) and 1.6B (PAG) [73], and Recraft V3 [66] are evaluated respectively. The styles are pointillism and minimalism.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

PROMPT: A muscular man carries an unconscious individual with blonde hair in his strong arms. A girl in a camo outfit, sporting short hair, has a butterfly on her shoulder. The entire scene is depicted in pencil sketch style.

PROMPT: A stone gargoyle carries a joyful red panda with vibrant orange fur, surrounded by intricate stone carvings of a gothic cathedral under deep, moonlit shadows. This scene is rendered entirely in clay style.

0.51 0.41 0.41 0.37 0.48

0.53 0.54 0.45 0.47 0.42

[Figure 194]

GPT-4o Imagen4 Seedream 3.0 Recraft V3 SANA-1.5 4.8B (PAG)

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

- Figure 12: Visualization results of SOTA methods. Media styles of GPT-4o [45], Imagen4 [17], Seedream 3.0 [24], Recraft V3[66] and SANA-1.5 4.8B (PAG) [73] are evaluated respectively. The styles are pencil sketch and stone sculpture.

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

PROMPT: A blonde woman with long, flowing hair and a bandaged eye is trapped within the confines of a glowing light bulb. Surrounded by fluttering moths and drifting leaf-like debris, her legs are shackled, emphasizing her captivity. The scene evokes a dreamy atmosphere. This striking image is created in a fusion of Cyberpunk and anime styles.

PROMPT: Designed in a distinctly 3d rendering look, this underwater scene showcases an acrylic pour painting style with dynamic colors of purple, blue, green, and yellow. Intricate textures outline coral reefs amid dreamy bubbles and sparkles.

0.58 0.49 0.51 0.30 0.42

0.62 0.58 0.39 0.35 0.39

[Figure 209]

Seedream 3.0 BAGEL+CoT Imagen4 BAGEL Janus-Pro

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

- Figure 13: Visualization results of SOTA methods. Anime styles of Seedream 3.0 [24], BAGEL+CoT [18], Imagen4 [17], BAGEL [18] and Janus-Pro [13] are evaluated respectively The styles are cyberpunk and 3d rendering.

