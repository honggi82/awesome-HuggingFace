# arXiv:2511.01340v1[cs.CV]3Nov2025

## | |: A Large and Diverse Multimodal Benchmark for evaluating the ability of Vision-Language Models to understand Rebus Puzzles

Trishanu Das∗

Abhilash Nandy∗

dastrishanu01@gmail.com Tredence Inc. India

nandyabhilash@gmail.com Indian Institute of Technology Kharagpur India

Khush Bajaj

Deepiha S

Indian Institute of Technology Kharagpur India

Indian Institute of Technology Kharagpur India

### 1 Introduction

### Abstract

Understanding Rebus Puzzles (Rebus Puzzles use pictures, symbols, and letters to represent words or phrases creatively) requires a variety of skills such as image recognition, cognitive skills, commonsense reasoning, multi-step reasoning, image-based wordplay, etc., making this a challenging task for even current Vision-Language Models. In this paper, we present | | (Rebus Puzzle for the Word “Rebus”, consisting of the “Re” - and “Bus” - symbols), a large and diverse benchmark of 1, 333 English Rebus Puzzles containing different artistic styles and levels of difficulty, spread across 18 categories such as food, idioms, sports, finance, entertainment, etc. We also propose RebusDescProgICE, a model-agnostic framework which uses a combination of an unstructured description and code-based, structured reasoning, along with better, reasoningbased in-context example selection, improving the performance of Vision-Language Models on | | by 2.1−4.1% and 20−30% using closed-source and open-source models respectively compared to Chain-of-Thought Reasoning1.

Rebus Puzzles are a form of wordplay that uses images, letters, and symbols to represent words or syllables. They serve as a creative tool to spark lateral thinking, challenge conventional patterns, and invite solvers to uncover hidden meanings through visual clues. Understanding such puzzles requires a plethora of capabilities such as image recognition, commonsense knowledge and reasoning, multi-step reasoning, and understanding the creator’s intent [13]. Fig. 1 shows an example of a Rebus Puzzle containing images and letters. The images in the puzzle are that of a “Mill” and “Lime”, followed by letters that read “Ters”. Combining “Mill”, “Lime”, and “Ters” creatively (by adding/subtracting/replacing letters), we get a meaningful word of “Millimeters” as the final answer of the puzzle. Note that the choice of the words for images are very crucial - for instance, if the word “Turbine” is used instead of “Mill”, we would not arrive at a meaningful final answer.

[Figure 1]

### Keywords

Rebus, puzzles, multimodal, benchmark

ACM Reference Format:

Trishanu Das, Abhilash Nandy, Khush Bajaj, and Deepiha S. 2018. | |: A Large and Diverse Multimodal Benchmark for evaluating the ability of Vision-Language Models to understand Rebus Puzzles. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 7 pages. https: //doi.org/XXXXXXX.XXXXXXX

∗Both authors contributed equally to this research. 1The dataset and code are available at https://github.com/abhi1nandy2/Re-Bus

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

#### Figure 1: Example of a Rebus Puzzle in | |

Puzzle-solving and reasoning abilities of Vision-Language Models have been evaluated previously. For instance, M3Exam [42] evaluates multimodal multiple choice exam questions. MATH-Vision [37]evaluatesthemathematicalreasoning ability of Vision-Language Models on math questions spanning several topics. There is also prior work on evaluating Rebus Puzzles in English [13] and in Italian [32] Languages. However, prior work on Rebus Puzzles neither

proposes a diverse benchmark having different levels of difficulty, nor does any such work provide a model-agnostic solution that can be applied on top of both open as well as closed-source models with minimal or no training for improved performance.

The development of Vision-Language Models (VLMs) [1, 5, 15, 17, 22, 33] has witnessed a substantial rise in recent years. These models have demonstrated outstanding state-of-the-art (SOTA) performance across various downstream tasks, including Image Captioning and Visual Question Answering. These models are pretrained such that images and text share a common embedding space, ensuring that images and their corresponding textual descriptions have similar representations within that space.

In this paper, we first inspect whether VLMs are able to understand and solve Rebus Puzzles - Given an image of a Rebus Puzzle (like in Fig. 1), the VLM is expected to generate a natural language answer to the puzzle as a word/phrase (“Millimeters” in case of Fig. 1). This is a challenging task that extends beyond mere image analysis and linguistic comprehension, as it involves a layered process that draws on factual knowledge, contextual insight, language skills, and logical reasoning within specific boundaries—core abilities essential for tackling many real-world challenges [32].

To evaluate the task, we curated a large and diverse multimodal dataset | | comprising of 1, 333 English Rebus Puzzles2, where each dataset sample contains an image of a Rebus Puzzle which contains a combination of images and/or texts, along with the answer to the puzzle, and a rich suite of meticulously annotated metadata such as a hint to solve the the puzzle, difficulty of the puzzle, whether the spelling of the text/objects in the image is varied in order to get the puzzle’s answer, is color of text in the puzzle relevant in solving the puzzle, etc., making our proposed |

| superior to that of the previous works on Rebus Puzzles [13, 32]. Also, to increase the diversity and difficulty of the puzzles, some samples in | | are generated using ControlNet [41], which adds an ambient background as a backdrop while preserving the core content of the Rebus puzzle. This added background serves as a visual distraction, thereby increasing the difficulty of solving the puzzle.

Additionally, we propose a compute-efficient RebusDescProgICE framework, which combines structured (code-based) and unstructured reasoning in an in-context learning setup, along with a lightweight example selection strategy requiring only minimal training. Unlike baselines that rely on a single reasoning style, RebusDescProgICE consistently improves puzzle-solving performance. For GPT-4o, it yields steady gains (Word-Level F1 rising from 0.489 in zero-shot normal prompting to 0.512 in three-shot RebusDescProgICE). The benefits are more striking for open-source models: Qwen2-VL-7B achieves up to a 20–30% relative improvement over description-only and VisProg baselines (e.g., from 0.2 to 0.264 F1). These results highlight that the synergy of structured and unstructured reasoning, coupled with informed example selection, is key to unlocking better performance, particularly for weaker open-source VLMs.

We make the following contributions in this paper - (1) We introduce | |, a large, diverse dataset of 1, 333 annotated English Rebus Puzzles (2) We enhance puzzle difficulty using ControlNet

2The answer to every puzzle is in English.

to add distracting yet realistic visual backgrounds (3) We propose RebusDescProgICE, a compute-efficient framework combining structured and unstructured reasoning in-context (4) We design a novel in-context example selection strategy aligned with anticipated VLM reasoning patterns.

### 2 Background

Linguistic Puzzle Solving. Linguistic Puzzles have emerged as an intriguing benchmark for evaluating the reasoning and language capabilities of large language models (LLMs) [12, 20, 30]. While early research predominantly explored English-language challenges like crosswords [8, 16, 31, 35], recent efforts have expanded to include a richer variety of puzzle types, including popular games such as Wordle [2] and the New York Times Connections [34]. Beyond English, automated puzzle solvers have been developed for other languages as well—such as French [4], German [43], and Italian [3, 44]. Additionally, educational puzzle generators are available in languages like Italian [39] and Turkish [40], highlighting the growing global interest in computational approaches to linguistic games.

Code-based reasoning using VLMs and LLMs. Structured, codebased reasoning shows improvements in performance in complex, commonsense reasoning tasks. Recent approaches like VISPROG [14] extend this paradigm to vision-language tasks, where VLMs and LLMs collaborate by generating modular code that orchestrates vision models and logical operations to solve complex visual problems. Code-based reasoning methods like PoT (Program of Thoughts) [7] help LLMs tackle math by writing and running code, separating thought from calculation. PAL (Program-Aided Language Models) [11] boost LLM performance by turning text problems into code, allowing a Python Interpreter handle the computation. Madaan et al. [18] show that even without code in the task, code LLMs do better when commonsense problems are framed as code generation.

### 3 | | Dataset 3.1 Our Annotation Pipeline

The entire data collection and annotation pipeline is shown in Fig. 2. We curated a collection of Rebus Puzzles with meticulously annotated metadata in this section in 3 stages.

3.1.1 Stage 1: Collecting Rebus Puzzles from multiple Internet Sources. We collect Rebus Puzzle Images along with the corresponding ground truth answers from 3 different sources https://eslvault.com/free-printable-rebus-puzzles/ (contains a diverse set of rebus puzzles that are mostly in black and white), https://kids.niehs.nih.gov/games/brainteasers/rebus-puzzles (contains mostly text-based rebus puzzles), and http://flashbynight.com/ rebus (contains a diverse set of rebus puzzles that are mostly colored). After removing duplicate Rebus Puzzle Images, we end up with 722 Rebus Puzzles. We also verified the answers collected for each Rebus Puzzle and made manual corrections to the answer wherever necessary.

[Figure 2]

Figure 2: Annotation Pipeline of | | Dataset

[Figure 3]

Figure 3: Breakdown of some important Rebus Puzzle Metadata Characteristics

- 3.1.3 Stage 3: Using ControlNet to obtain modified versions of Rebus Puzzles. The Rebus Puzzles collected in Stage 1 contain a mostly white background, making the puzzle potentially easier to solve. One way to make a rebus Puzzle difficult to solve could be to add a distracting background to the Rebus Puzzle Image, without affecting the Rebus Puzzle in itself. To do so, we use ControlNet [41] on all the Rebus Puzzles collected in Stage 1, treating the puzzles as Canny Edge Maps 3. These generated images are verified by a qualified annotator to keep only those images in the dataset that are meaningful and would have the same answer as the original puzzle (from Stage 1). Among the 722 generated images, 611 puzzle images are filtered, and are added to the | | dataset (along with the Stage 1 puzzles), making the total number of Rebus Puzzles in | | Dataset as 1, 333.
- 3.2 Dataset Description and Details

- 3.1.2 Stage 2: Annotation of Rebus Puzzle Metadata. Several types of metadata are annotated for the rebus Puzzles using 4 annotators, all of whom were at least in their second year of undergraduate study and enrolled in institutions where English is the primary language of instruction. The annotated metadata includes

- (1) features related to the process of solving the puzzle, such as puzzle difficulty (Easy/Hard), whether spelling of individual objects/text is different from that in the ground truth answer, hint for solving it, number of units of reasoning to solve it (2) image feature, like whether any text is present in the image (3) binary feature of importance of different attributes such as color, position, orientation, aspect ratio, size, style/texture, and number of object(s)/text when answering the puzzle. Fig. 3 shows the distribution of the puzzles across 3 types of binary metadata as a Venn Diagram. This shows that Rebus Puzzles are highly diverse, as they are spread out across different combinations of the binary values of the metadata.

[Figure 4]

#### Figure 4: Distribution of Rebus Puzzles based on their category

3API Calls to https://huggingface.co/spaces/hysts/ControlNet were performed during implementation

The | | dataset has a total of 1, 333 images of Rebus Puzzles, 611 of which are generated using ControlNet [41] and are therefore of a different artistic style.

The Rebus puzzles included in the | | Dataset encompass a diverse range of linguistic and conceptual features. To better understand these underlying patterns, we employ ChatGPT [15] to systematically categorize the ground truth answers into distinct and meaningful classes by designing an appropriate prompting strategy. Fig. 4 shows the distribution of the Rebus Puzzles across the 18 finegrained categories predicted by ChatGPT. These categories belong to varied aspects of Language and Expression Usage, Knowledge and Facts, Culture and Society, Activities and Hobbies, and Nature and Living Things.

To illustrate the diversity among these Rebus Puzzle Images, we project their pre-trained CLIP [27] (MIT License) image embeddings into a 2D space using UMAP [21], as shown in Fig. 5. We color-code the image samples according to their artistic styles. Interestingly, despite sharing identical puzzle answers, the Rebus Puzzle images generated via ControlNet [41] are semantically far apart from their original counterparts. This results in a highly diverse set of Rebus Puzzle images that span multiple categories.

[Figure 5]

Figure 5: 2D UMAP Representations of CLIP Image representations of Rebus Puzzle Images

### 3.3 Proposed approach: RebusDescProgICE

Our proposed approach RebusDescProgICE introduces a novel LLM-agnostic reasoning module in an in-context learning setting. The reasoning to be generated contains 2 components - (1) an unstructured and detailed image description (2) a structured, code-based reasoning component which elaborates the steps to be followed create the Rebus Puzzle Image. The unstructured and codebased reasoning components provide explicit factual knowledge and procedural logic, which are necessary in order to solve a Rebus Puzzle correctly. Also, we use a novel in-context example selection method based on the similarity between the code-based reasoning components (similar to Poesia et al. [26]).

In-Context Example Selection in RebusDescProgICE. In our approach, a VLM is guided using task-specific examples and instructions provided within the same session, without modifying its underlying parameters. This setup enhances the VLM’s output by leveraging relevant contextual information. To ensure the selected

in-context examples are useful for the target task, we employ a vector embedding-based similarity technique to retrieve training examples whose embeddings closely match that of the test input. Additionally, we propose a novel technique (inspired by Poesia et al. [26]) to learn a unified embedding that effectively represents a Rebus Puzzle Image.

4 Experiments and Results

- 4.1 Experimental Setup

All experiments using open-source models are carried out on 4 L40 GPUs, each having a VRAM of 48GB.

- 4.2 Baselines

We use the following prompting strategies as prompts - (1) ZeroShot. This follows the naive zero-shot prompting strategy inspired by Gritsevskiy et al. [13] (2) Few-Shot CoT (Chain-of-Thought) [38]. In addition to mentioning the task instructions, this baseline uses In-Context Learning [6], where each in-context example contains the Rebus Puzzle image and a corresponding hint (annotated as metadata) as the input, and the corresponding ground truth answer as the output (3) Few-shot with Descriptions. This uses a similar prompt template as the previous baseline, where instead of using an annotated hint, an image description generated using GPT4o [15] in a zero-shot setting is used as the intermediate reasoning in the in-context examples. Note that for the last two baselines, the in-context example(s) for each test sample are randomly sampled from examples in the holdout set.

We benchmarked three closed-source models—GPT-4o, GPT-4omini, and GPT-4 Turbo—and three open-source vision-language models—Phi-3.5-Vision, Pixtral-12B, and Qwen2-VL-7B. GPT-4o stands out with its seamless support for text, audio, images, and video, achieving state-of-the-art multilingual, vision, and audio understanding while operating faster and more cost-efficiently than GPT-4 Turbo [23, 25]. The lighter GPT-4o-mini, obtained through model distillation, preserves much of GPT-4o’s multimodal capabilities at significantly lower cost and latency, excelling in reasoning and coding benchmarks [9, 24]. GPT-4 Turbo similarly offers strong performance in text and code tasks but with comparatively less advanced multimodal integration [19]. On the open-source side, Phi-3.5-Vision isalightweight, multimodal model adept at dense reasoning and efficient image processing, even enabling high-quality OCR and text extraction in resource-constrained environments [10, 29]. Pixtral-12B, a 12-billion-parameter model, delivers strong instruction-following performance in both text and vision, outperforming larger open models in multimodal benchmarks thanks to its high-resolution vision encoder and long-context support [1]. Lastly, Qwen2-VL-7B introduces dynamic-resolution visual tokenization and multimodal rotary embeddings (M-ROPE), enhancing its ability to process variable-resolution images and fuse visual and textual information—reaching performance comparable to leading closedsource models in some benchmarks [36]. Each model thus presents a distinct balance between capability, modality support, and computational accessibility, offering a diverse testbed for evaluating rebus puzzle solving.

### 4.3 Automated Evaluation Metrics

For automated evaluation, we employ two lexical text-matching metrics. The word-level F1 score is computed as the harmonic mean of precision and recall over the tokenized prediction and reference answers, providing a balanced measure of both answer completeness and correctness [28]. In addition, we report substring accuracy, which measures whether the predicted answer occurs as a contiguous substring within the reference. This metric is particularly relevant for ReBus puzzle-solving, where reference answers may permit multiple surface realizations, and a prediction that partially overlaps with the ground truth can still capture essential semantic content.

### 4.4 RebusDescProgICE vs. Baselines

Tables 1 and 2 show the substring accuracy and word-level F1 scores respectively across closed and open-source models with varying number of in-context examples and prompting methods. We can infer that - (1) The closed-source models (GPT-4o, GPT-4o-mini, GPT-4 turbo) consistently outperform open-source models (Phi-3.5, Pixtral, Qwen2-VL-7B) across both metrics. For instance, in Table 2 (Word-Level F1 Score), GPT-4o reaches 0.536 (three-shot, only description), while open-source models peak around 0.270 (Qwen2VL-7B, three-shot, RebusDescProgICE). This highlights the superior reasoning and alignment capabilities of closed-source VLMs, particularly GPT-4 variants, which show more robust performance across prompting strategies. (2) Our method RebusDescProgICE shows notable improvements, particularly when compared to simpler prompting methods like "only description." For example, in GPT-4o (Table 1), three-shot RebusDescProgICE achieves 0.422 substring accuracy, comparable to or better than most other settings. Similarly, in Table 2, GPT-4o reaches 0.512 F1, showing stable gains. Even for weaker open-source models like Qwen2-VL-7B, RebusDescProgICE boosts performance substantially (e.g., from 0.200 to 0.241 in one-shot F1, and up to 0.264 in three-shot F1), suggesting that the synergy of visual program + description generalizes across model families. (3) Increasing the number of in-context examples generally leads to modest but consistent improvements, especially in F1 scores. For example, GPT-4 turbo F1 improves from 0.382 (zero-shot normal) to 0.442 (three-shot normal). Substring accuracy shows smaller gains, but still some improvements (e.g., GPT-4o from 0.420 zero-shot normal to 0.416–0.422 three-shot variants). However, gains plateau beyond two or three examples, indicating diminishing returns. (4) Importance of combining VisProg and Description (our method). Comparing isolated prompting methods highlights why both components are essential. VisProg alone (e.g., GPT-4o three-shot VisProg: 0.383 substring acc, 0.506 F1) or only description (GPT-4o three-shot: 0.434 substring acc, 0.536 F1) do well individually, but RebusDescProgICE consistently balances both to achieve competitive performance (0.422 substring acc, 0.512 F1). In open-source models, this effect is even clearer: for Qwen2-VL-7B, VisProg (0.214 substring acc, 0.248 F1) and only desc (0.111 substring acc, 0.250 F1) underperform compared to RebusDescProgICE (0.107 substring acc, 0.264 F1). This demonstrates that combining structured visual reasoning (VisProg) with descriptive context leads to more reliable gains than either alone.

Performance on Augmented Test Data.

Tables 3 and 4 display the Substring Accuracy and word-level F1 scores on Augmented Test Data with varying number of incontext examples and prompting methods. We can infer that - (1) The overall low scores across models arise from the complexity of our dataset, where solving Rebus puzzles requires layered semantic reasoning; this is further amplified by the ControlNet-augmented noisy backgrounds, resulting in best scores remaining modest (e.g., maximum F1 of only 0.402 for GPT-4o). (2) GPT-4o consistently achieves the highest performance across substring accuracy (0.280 with two in-context examples, only desc) and word-level F1 (0.402 with two in-context examples, VisProg), reaffirming the challenging nature of the dataset even for state-of-the-art closed-source models. (3) Open-source models such as Phi-3.5 and Pixtral struggle considerably, with word-level F1 mostly below 0.20, reflecting their limited capacity for abstract reasoning under noisy conditions, whereas Qwen2-VL-7B-Instruct shows relatively better resilience (F1 up to 0.253). (4) Our proposed RebusDescProgICE framework provides consistent gains over standard prompting, particularly for weaker models — for instance, boosting Pixtral’s F1 from 0.186 (cot, one example) to 0.201 (two examples, RebusDescProgICE). (5) Increasing the number of in-context examples does not uniformly improve performance, confirming that puzzle-solving is not driven by pattern-matching; instead, structured reasoning guidance through RebusDescProgICE is crucial for robustness. Overall, these results establish our dataset as a strong benchmark for evaluating the reasoning capabilities of vision-language models under challenging, real-world-like conditions.

### 5 Conclusion

Closed-source VLMs remain far ahead of open-source ones in this challenging rebus puzzle-solving task. Nevertheless, our proposed method, RebusDescProgICE, proves robust across settings and especially beneficial for open-source models that otherwise struggle. While additional in-context examples help, the real performance boost comes from integrating both code-based reasoning (VisProg) and descriptive grounding, validating our design choice.

### References

- [1] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. 2024. Pixtral 12B. arXiv preprint arXiv:2410.07073

(2024).

- [2] Benton J Anderson and Jesse G Meyer. 2022. Finding the optimal human strategy for wordle using maximum correct letter probabilities and reinforcement learning. arXiv preprint arXiv:2202.00557 (2022).
- [3] Giovanni Angelini, Marco Ernandes, and Marco Gori. 2005. Solving italian crosswords using the web. In Congress of the Italian Association for Artificial Intelligence. Springer, 393–405.
- [4] Giovanni Angelini, Marco Ernandes, Tommaso Iaquinta, Caroline Stehlé, Fanny Simões, Kamyar Zeinalipour, Andrea Zugarini, and Marco Gori. 2023. The webcrow french crossword solver. In International Conference on Intelligent Technologies for Interactive Entertainment. Springer, 193–209.
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025).
- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners.

#### Table 1: Substring Accuracy across models with varying number of in-context examples and prompting methods.

Zero One Two Three normal cot normal cot VisProg only desc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE

Model

GPT-4o 0.420 0.449 0.416 0.426 0.387 0.402 0.414 0.414 0.423 0.380 0.428 0.411 0.416 0.415 0.383 0.434 0.422 GPT-4o-mini 0.213 0.428 0.223 0.215 0.224 0.230 0.208 0.211 0.238 0.221 0.227 0.230 0.223 0.202 0.229 0.188 0.208 GPT-4 turbo 0.279 0.410 0.346 0.319 0.315 0.303 0.324 0.342 0.327 0.307 0.321 0.328 0.342 0.319 0.328 0.313 0.343 Phi-3.5 0.169 0.114 0.075 0.163 0.128 0.086 0.110 0.066 0.152 0.099 0.071 0.098 0.062 0.154 0.087 0.062 0.096 Pixtral 0.107 0.096 0.102 0.065 0.075 0.108 0.096 0.086 0.066 0.075 0.093 0.078 0.095 0.083 0.093 0.092 0.093 Qwen2-VL-7B 0.342 0.161 0.268 0.209 0.208 0.160 0.101 0.241 0.182 0.185 0.139 0.068 0.343 0.146 0.214 0.111 0.107

#### Table 2: Word-Level F1 Score across models with varying number of in-context examples and prompting methods

Zero One Two Three normal cot normal cot VisProg onlydesc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE

Model

GPT-4o 0.489 0.467 0.503 0.516 0.507 0.514 0.513 0.511 0.549 0.506 0.521 0.517 0.519 0.517 0.506 0.536 0.512 GPT-4o-mini 0.330 0.457 0.355 0.356 0.355 0.346 0.352 0.358 0.370 0.362 0.374 0.366 0.348 0.361 0.366 0.360 0.352 GPT-4 turbo 0.382 0.451 0.433 0.421 0.424 0.413 0.398 0.440 0.432 0.423 0.426 0.439 0.442 0.438 0.426 0.422 0.431 Phi-3.5 0.153 0.130 0.093 0.161 0.130 0.191 0.198 0.112 0.178 0.177 0.196 0.205 0.106 0.198 0.172 0.196 0.177 Pixtral 0.151 0.185 0.161 0.189 0.216 0.209 0.209 0.170 0.207 0.214 0.199 0.225 0.180 0.209 0.238 0.213 0.239 Qwen2-VL-7B 0.179 0.120 0.176 0.206 0.219 0.200 0.241 0.211 0.230 0.235 0.222 0.270 0.264 0.237 0.248 0.250 0.264

#### Table 3: Substring Accuracy on Augmented Test Data across models with varying number of in-context examples and prompting methods.

One Two Three normal cot VisProg only desc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE

Model

GPT-4o 0.257 0.264 0.268 0.254 0.268 0.262 0.248 0.246 0.280 0.246 0.262 0.254 0.245 0.259 0.259 Phi-3.5 0.060 0.118 0.093 0.053 0.058 0.079 0.120 0.081 0.039 0.033 0.079 0.104 0.079 0.025 0.039 Pixtral 0.065 0.074 0.065 0.074 0.062 0.079 0.041 0.060 0.046 0.058 0.086 0.058 0.072 0.067 0.048 Qwen2-VL-7B-Instruct 0.276 0.282 0.123 0.116 0.097 0.238 0.165 0.107 0.083 0.083 0.222 0.129 0.090 0.067 0.070

#### Table 4: Word-Level F1 score on Augmented Test Data across models with varying number of in-context examples and prompting methods.

One Two Three normal cot VisProg only desc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE normal cot VisProg only desc RebusDescProgICE

Model

GPT-4o 0.366 0.400 0.379 0.372 0.383 0.384 0.395 0.397 0.395 0.374 0.381 0.385 0.402 0.400 0.391 Phi-3.5 0.085 0.134 0.135 0.152 0.164 0.125 0.161 0.169 0.163 0.164 0.125 0.173 0.154 0.156 0.158 Pixtral 0.149 0.186 0.188 0.192 0.195 0.150 0.190 0.190 0.174 0.201 0.161 0.209 0.206 0.195 0.215 Qwen2-VL-7B-Instruct 0.209 0.237 0.185 0.183 0.218 0.212 0.234 0.207 0.207 0.235 0.213 0.233 0.208 0.212 0.253

In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 1877–1901. https://proceedings.neurips.cc/paper_files/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf

- [7] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. Transactions on Machine Learning Research (2023). https://openreview.net/forum?id=YfZ4ZPt8zd
- [8] Marco Ernandes, Giovanni Angelini, and Marco Gori. 2005. Webcrow: A webbased system for crossword solving. In AAAI. 1412–1417.
- [9] EverydayAI. 2024. GPT-4o Mini: lighter, faster, and more affordable. Review distinguishing GPT-4o mini’s efficiency versus GPT-4o.
- [10] Dyland Freedman. 2024. Phi-3.5 Vision excels at OCR and text extraction. Reports strong OCR/text extraction, including handwriting.
- [11] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Pal: Program-aided language models. In International Conference on Machine Learning. PMLR, 10764–10799.
- [12] Panagiotis Giadikiaroglou, Maria Lymperaiou, Giorgos Filandrianos, and Giorgos Stamou. 2024. Puzzle Solving using Reasoning of Large Language Models: A Survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 11574–11591.
- [13] Andrew Gritsevskiy, Arjun Panickssery, Aaron Kirtland, Derik Kauffman, Hans Gundlach, Irina Gritsevskaya, Joe Cavanagh, Jonathan Chiang, Lydia La Roux, and Michelle Hung. 2024. REBUS: A Robust Evaluation Benchmark of Understanding Symbols. arXiv preprint arXiv:2401.05604 (2024).
- [14] Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual Programming: Compositional Visual Reasoning without Training. In Proceedings of the IEEE/CVF

- Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 14953– 14962.
- [15] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- [16] Michael L Littman, Greg A Keim, and Noam Shazeer. 2002. A probabilistic approach to solving crossword puzzles. Artificial Intelligence 134, 1-2 (2002), 23–55.
- [17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruction Tuning. arXiv:2304.08485 [cs.CV]
- [18] Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig.

2022. Language Models of Code are Few-Shot Commonsense Learners. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 1384–1403. doi:10.18653/v1/2022.emnlp-main.90

- [19] Time Magazine. 2024. OpenAI Announces a More Powerful, Cheaper GPT-4 Turbo. Describes GPT-4 Turbo enhancements and cost reduction.
- [20] Raffaele Manna, Maria Pia Di Buono, and Johanna Monti. 2024. Riddle me this: Evaluating large language models in solving word-based games. In Proceedings of the 10th Workshop on Games and Natural Language Processing@ LREC-COLING

2024. 97–106.

- [21] Leland McInnes, John Healy, Nathaniel Saul, and Lukas Großberger. 2018. UMAP: Uniform Manifold Approximation and Projection. Journal of Open Source Software 3, 29 (2018).
- [22] OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL]
- [23] OpenAI. 2024. GPT-4o achieves state-of-the-art results in multilingual, audio, and vision benchmarks. Includes multilingual benchmarks like 88.7 MMLU.

- [24] OpenAI. 2024. GPT-4o mini: Advancing cost-efficient intelligence. Details GPT-4o mini’s reasoning, math, and multimodal performance.
- [25] OpenAI. 2024. GPT-4o System Card. (2024). Describes GPT-4o capabilities across text, vision, and audio.
- [26] Gabriel Poesia, Alex Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, and Sumit Gulwani. 2022. Synchromesh: Reliable Code Generation from Pre-trained Language Models. In International Conference on Learning Representations. https://openreview.net/forum?id=KmtVD97J43e
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.
- [28] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, Jian Su, Kevin Duh, and Xavier Carreras (Eds.). Association for Computational Linguistics, Austin, Texas, 2383–2392. doi:10.18653/v1/D16-1264
- [29] Microsoft Research. 2025. Phi-3.5-Vision: a lightweight state-of-the-art open multimodal model. Describes Phi-3.5-Vision’s reasoning and multimodal capabilities.
- [30] Josh Rozner, Christopher Potts, and Kyle Mahowald. 2021. Decrypting cryptic crosswords: Semantically complex wordplay puzzles as a target for nlp. Advances in Neural Information Processing Systems 34 (2021), 11409–11421.
- [31] Abdelrahman Sadallah, Daria Kotova, and Ekaterina Kochmar. 2024. Are LLMs Good Cryptic Crossword Solvers? arXiv preprint arXiv:2403.12094 (2024).
- [32] Gabriele Sarti, Tommaso Caselli, Malvina Nissim, and Arianna Bisazza. 2024. Non Verbis, Sed Rebus: Large Language Models Are Weak Solvers of Italian Rebuses. In Proceedings of the 10th Italian Conference on Computational Linguistics (CLiC-it 2024), Felice Dell’Orletta, Alessandro Lenci, Simonetta Montemagni, and Rachele Sprugnoli (Eds.). CEUR Workshop Proceedings, Pisa, Italy, 888–897. https://aclanthology.org/2024.clicit-1.96/
- [33] Gemini Team. 2023. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805 [cs.CL]
- [34] Graham Todd, Tim Merino, Sam Earle, and Julian Togelius. 2024. Missed connections: Lateral thinking puzzles for large language models. In 2024 IEEE Conference on Games (CoG). IEEE, 1–8.
- [35] Eric Wallace, Nicholas Tomlin, Albert Xu, Kevin Yang, Eshaan Pathak, Matthew Ginsberg, and Dan Klein. 2022. Automated Crossword Solving. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume

- 1: Long Papers). 3073–3085.
- [36] et al. Wang. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception at Any Resolution. arXiv preprint arXiv:2409.12191 (2024). Introduces dynamic resolution and M-ROPE in Qwen2-VL.
- [37] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li.

2024. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804 (2024).

- [38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 24824–24837. https://proceedings.neurips.cc/paper_files/paper/2022/file/ 9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf
- [39] Kamyar Zeinalipour, Tommaso Iaquinta, Asya Zanollo, Giovanni Angelini, Leonardo Rigutini, Marco Maggini, and Marco Gori. 2023. Italian crossword generator: Enhancing education through interactive word puzzles. (2023).
- [40] Kamyar Zeinalipour, Yusuf Gökberk Keptiğ, Marco Maggini, Leonardo Rigutini, and Marco Gori. 2024. A turkish educational crossword puzzle generator. In International Conference on Artificial Intelligence in Education. Springer, 226–233.
- [41] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 3836–3847.
- [42] Wenxuan Zhang, Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. 2023. M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36. Curran Associates, Inc., 5484–5505. https://proceedings.neurips.cc/paper_files/paper/2023/file/ 117c5c8622b0d539f74f6d1fb082a2e9-Paper-Datasets_and_Benchmarks.pdf
- [43] Andrea Zugarini, Thomas Röthenbacher, Kai Klede, Marco Ernandes, Bjoern M Eskofier, and Dario Zanca. 2023. Die rätselrevolution: Automated german crossword solving. (2023).
- [44] Andrea Zugarini, Kamyar Zeinalipour, Surya Sai Kadali, Marco Maggini, Marco Gori, and Leonardo Rigutini. 2024. Clue-Instruct: Text-Based Clue Generation for Educational Crossword Puzzles. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 3347–3356.

