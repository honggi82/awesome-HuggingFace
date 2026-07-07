# arXiv:2409.12568v1[cs.CV]19Sep2024

## InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning

Xiaotian Han1∗ Yiren Jian1∗ Xuefeng Hu1∗ Haogeng Liu1,2,∗ Yiqi Wang1∗ Qihang Fan1,2 Yuang Ai1,2 Huaibo Huang2 Ran He2 Zhenheng Yang1

Quanzeng You1† 1ByteDance, Inc 2Chinese Academy of Sciences

### Abstract

Pre-training on large-scale, high-quality datasets is crucial for enhancing the reasoning capabilities of Large Language Models (LLMs), especially in specialized domains such as mathematics. Despite the recognized importance, the Multimodal LLMs (MLLMs) field currently lacks a comprehensive open-source pre-training dataset specifically designed for mathematical reasoning. To address this gap, we introduce InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents. It comprises 24 million web pages, 85 million associated image URLs, and 40 billion text tokens, all meticulously extracted and filtered from CommonCrawl. We provide a detailed overview of our data collection and processing pipeline. To demonstrate the robustness of InfiMM-WebMath-40B, we conducted evaluations in both text-only and multimodal settings. Our evaluations on textonly benchmarks show that, despite utilizing only 40 billion tokens, our dataset significantly enhances the performance of our 1.3B model, delivering results comparable to DeepSeekMath-1.3B, which uses 120 billion tokens for the same model size. Nevertheless, with the introduction of our multi-modal math pre-training dataset, our models set a new state-of-the-art among open-source models on multimodal math benchmarks such as MathVerse and We-Math. We release our data at https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B.

### 1 Introduction

Recently, Large Language Models (LLMs) have exhibited significant advancements, enhancing their capabilities in complex reasoning and multi-step mathematical problem-solving. This progress is driven by larger training scales, innovative inference techniques like Chain-of-Thought (CoT) prompting [69], and the rich diversity of training data. These developments have significantly advanced both proprietary models like GPT-4o [1], Claude 3.5 Sonnet [2], and open-source variants such as Llama 3.1 [17]. Notably, these models demonstrate enhanced understanding and reasoning, evident in their ability to tackle challenges from basic grade-school word problems in GSM8K [12] to high school competition-level tasks in MATH [24].

In parallel to general-purpose models, specialized smaller LLMs have achieved significant progress. Models such as DeepSeekMath-7B [62] and InternLM-Math [74] have specialized capabilities in mathematics, emphasizing their effectiveness within focused domains. Furthermore, LLM-based formal mathematics proving systems such as Alpha-Proof [16] and DeepSeek-Prover [72] have shown notable success in solving complex International Mathematics Olympiad (IMO)-level problems. These achievements further highlight the potential of LLMs, both large and small, to tackle sophisticated reasoning tasks that approximate human-level capabilities.

∗Equal contributions. †Corresponding author.

Preprint. Under review.

While most mathematical knowledge is typically encoded in language, visual elements like figures, diagrams, and geometric plots are pivotal for intuitively understanding abstract mathematical concepts. Recognizing this, several Multimodal Large Language Models (MLLMs) such as G-LLaVA [19], Math-LLaVA [63], LLaVA-Next [41], and MAVIS [81] have been developed to enhance reasoning capabilities by integrating these multimodal inputs. These models integrate vision modalities using visual embeddings from pre-trained models such as CLIP [59] and SigLIP [77], enhancing their ability to process and reason with visually represented mathematical concepts. They refine their capabilities through multi-modal instruction tuning with specialized datasets focused on mathematical instruction, such as Geo170k [20, 9], MathV360K [64], and MAVIS-Instruct [82].

While many of these models show promising progress, studies suggest that introducing new knowledge during the instruction fine-tuning stage can be challenging [85], often increasing the risk of hallucinations [21], particularly due to limitations in data scale, format, and training strategies. In contrast, large corporations often have access to high-quality, large-scale proprietary pre-training datasets, giving them an advantage in developing MLLMs with strong mathematical reasoning capabilities. However, despite growing demand, publicly available large-scale pre-training datasets that integrate both textual and visual mathematical data remain scarce, significantly hindering progress within the open-source community, which depends on accessible data to drive research and develop truly effective MLLMs.

In this work, we introduce InfiMM-WebMath-40B, the first publicly available large-scale multimodal mathematics pre-training dataset. This dataset marks a significant milestone for the open-source community, addressing the long-standing gap in publicly available multimodal math data. InfiMMWebMath-40B comprises 24 million mathematics and science-related web documents, including 85 million image URLs and approximately 40 billion text tokens, offering an unprecedented resource for training and fine-tuning Multimodal Large Language Models (MLLMs).

Inspired by previous efforts like OpenWebMath [53] and OBELICS [33], we construct our dataset by filtering mathematical and scientific content from the CommonCrawl [15] repository, preserving full multimodal web documents with interleaved images and text. We start with 44 snapshots of CommonCrawl data from 2019 to 2023. After filtering for Chinese and English webpages, we are left with 57.2 billion web documents. Through a rigorous series of filtering processes—beginning with model-based language filtering and deduplication—we apply both rule-based and model-based filtering using fastText [27], which narrows the collection down to 24 million high-quality mathematics and science-related web documents.

To showcase the potential of InfiMM-WebMath-40B, we conduct extensive experiments using recent benchmarks such as MathVerse [80] and WeMath [58]. Our preliminary results demonstrate the effectiveness of the dataset in enhancing multimodal mathematical reasoning, further validating its importance for both open-source research and the broader AI community.

Our contributions in this work are as follows:

- • First, we introduce InfiMM-WebMath-40B, the first publicly available, large-scale multimodal dataset specifically designed for mathematical pre-training. This dataset significantly contributes to filling a critical gap in the open-source community by providing a vast collection of high-quality text and image data to advance the mathematical reasoning capabilities of Multimodal Large Language Models (MLLMs).
- • Second, we detail the comprehensive preprocessing pipeline used to filter and align mathematics and science-related content from CommonCrawl, ensuring the quality and relevance of the dataset.
- • Lastly, we evaluate the impact of continuing pre-training on InfiMM-WebMath-40B through extensive experiments. Our InfiMM-Math models, trained on top of this dataset, demonstrate strong performance on mathematical reasoning benchmarks, particularly for complex multimodal problems. We believe future research can further unlock MLLM mathematical reasoning capabilities based on our InfiMM-WebMath-40B dataset.

### 2 Related Work

#### 2.1 LLM for Math

The use of LLMs for mathematical reasoning has been explored in several studies. GPT-3 [8] already demonstrated the ability to solve basic arithmetic and algebraic problems. However, it was observed that the model may produce incorrect or misleading explanations. To better assess the progress of LLMs in mathematical reasoning and encourage their improvement, several math-specific evaluation benchmarks and training datasets have been introduced.

Evaluation Benchmarks The GSM8K [13] dataset focuses on grad school-level math problems, highlighting models’ ability in basic arithmetic and reasoning. In contrast, the MATH dataset [25] targets more advanced topics, covering high school and undergraduate-level mathematics. MMLU [23] assesses LLM performance across multiple subjects, with its STEM section providing insights into the models’ capabilities in scientific and mathematical reasoning. Additionally, benchmarks such as SAT [4] and OCW [52] evaluate LLMs on standardized test problems, further probing their mathematical capabilities. Recent research has also introduced specialized benchmarks for evaluating logical and counterfactual reasoning in mathematical contexts, such as MalAlgoQA [44] and MathCheck-GSM [83]. Together, these benchmarks offer a comprehensive framework for assessing the strengths and limitations of LLMs in mathematical reasoning.

Training Datasets Several large-scale math-specific pre-training datasets have been introduced to enhance the mathematical reasoning capabilities of LLMs. Notable proprietary datasets include WebMath [57], developed by OpenAI, which comprises 35 billion tokens from Github, arXiv, and Math StackExchange. MathMix [40], containing 1 billion high-quality mathematical tokens, has been used to fine-tune GPT-4. Math Web Pages [35] contributes 17.5 billion LaTex-based tokens for training models like Minerva. Similarly, the DeepSeekMath Corpus [61] offers 120 billion tokens used in training the DeepSeekMath model. Among the open-source datasets, AMPS [24] includes over 100,000 Khan Academy problems and 5 million problems generated using Mathematica. NaturalProofs [70] offers 32,000 theorem statements and proofs, along with 14,000 definitions sourced from ProofWiki and other platforms. OpenWebMath [54] filters 14.7 billion tokens of mathematical content from CommonCrawl. MathPile [68], which aggregates data from textbooks, arXiv, Wikipedia, ProofWiki, StackExchange, and other web pages, contains 9.5B tokens. Proof-Pile-2 [5] combines several sources, including OpenWebMath [54], AlgebraicStack (10.3B tokens of mathematical code), and arXiv papers (28.0B tokens) [14]. MathInstruct [76] compiles data from 13 math datasets, incorporating intermediate reasoning steps to facilitate supervised fine-tuning of LLMs.

These datasets, both proprietary and open-source, play a critical role in advancing the mathematical reasoning capabilities of LLMs, enabling more robust and accurate problem-solving across a wide range of mathematical domains.

#### 2.2 MLLM for Math

The rapid evolution of MLLMs has sparked significant interest in enhancing their capabilities for multi-modal reasoning [67]. This section outlines the key benchmarks and training datasets developed to evaluate and improve MLLMs’ mathematical reasoning abilities.

Evaluation Benchmarks Benchmarks such as GeoEval [78], Geometry3K [47], and GeomVerse [30] focus on evaluating MLLM performance in plane geometry problems. While ChartX [71], ChartQA [50], and ChartBench [73] target the understanding and reasoning of charts and graphical plots. Comprehensive math benchmarks like MathVista [46], MathVerse [80], MathVision [66], and We-Math [58] combine tasks across plane geometry, solid geometry, and chart interpretation. These benchmarks provide a robust framework for assessing the strengths and limitations of MLLMs in mathematical reasoning.

Training Datasets The development of multimodal instruction fine-tuning datasets has been crucial in advancing MLLMs’ capabilities for mathematical reasoning. Key contributions include GeoGPT4V [9], Geo170k [20], MathV360K [64], and MAVIS [82]. Most of these instruction finetuning datasets are synthesized by ChatGPT using existing public datasets. While large-scale open source interleaved multi-modal pre-training datasets, such as Multimodal-C4 [84] and OBLICS [33], have been constructed, these datasets are not specifically designed for MLLM mathematical reasoning. MINT-1T [3], which includes parsed PDFs and arXiv data, aims to enhance MLLMs’ reasoning

capability. Multimodal ArXiv [39] builds high-quality captioning and instruction tuning datasets from arXiv papers, providing further opportunities for improving reasoning in mathematical contexts.

The aforementioned datasets have been proposed for instruction tuning of MLLMs to tackle mathematical reasoning problems, leading to the development of several advanced math-focused MLLMs. However, despite these advances, a critical gap remains in the community’s efforts: the lack of high-quality, large-scale, multimodal pre-training datasets specifically designed for MLLMs. In this work, we address this gap by constructing and publicly releasing a high-quality, multimodal, interleaved mathematical dataset specifically designed for MLLM pre-training. This resource aims to catalyze further progress in the mathematical reasoning capabilities of MLLMs by providing a more robust foundation for pre-training.

### 3 Dataset Construction

In this section, we outline the methodology employed in constructing the InfiMM-WebMath-40B dataset. Our dataset comprises approximately 24 million webpages, encompassing 40 billion text tokens and 85 million image URLs. The objective of this study is to develop a large-scale multimodal math dataset that integrates interleaved image and text data.

To develop our dataset, we source content from the CommonCrawl archives, adhering to methodologies established by previous large-scale data constructions for pre-training language models as discussed in works such as RefinedWeb [56], DataComp [37] and further refined by FineWeb [55]. Building upon the methodology established in the construction of the OBELICS dataset [33], we enhance our dataset with interleaved text and corresponding image URLs. The final stage involves the parallel downloading of images based on these URLs.

#### 3.1 Text-only Data Curation Pipeline

Figure 1: InfiMM-WebMath-40B data curation pipeline.

#### 3.1.1 Text Extraction and Language Filtering

After careful investigation, we chose Trafilatura 3 as the library for extracting text from web pages, as it is a widely adopted Python library for web scraping . This library has also been used extensively in the construction of other open-source, text-only datasets, such as RefinedWeb [56] and FineWeb [55]. However, Trafilatura’s design focuses on plain text, omitting math-related equations and symbols—crucial elements for our dataset. Consequently, the subsequent section will outline our development of a specialized extraction tool tailored for math-related content. We initially use Trafilatura to streamline the data extraction process, substantially reducing the volume of data for further processing in our data pipeline.

Following the methodology used in DeepSeekMath [62], we focus on retaining only Chinese and English content when constructing our dataset. To achieve this, we apply language filtering to the CommonCrawl repositories, which originally comprise approximately 122 billion webpages, as shown in Figure 1.

3https://trafilatura.readthedocs.io/en/latest/

For language detection, we employ a fastText language identification model [28], which offers an effective balance between speed and accuracy, making it well-suited for the initial filtering phase. This language filtering process significantly reduces the dataset size, lowering the number of pages from 122 billion to 57.2 billion.

#### 3.1.2 Mathematical Content Extraction

Extracting text from HTML in the field of mathematics presents unique challenges, particularly due to the limited availability of specialized tools that can accurately handle mathematical content, such as LaTeX equations. Most standard extraction tools are not equipped to process this specialized content, resulting in gaps in accurate content retrieval. Additionally, capturing image URLs and their corresponding positions in the text is equally important, as both are critical for constructing our dataset comprehensively.

To address these issues, we evaluate several open-source HTML extraction tools to balance highquality text extraction, accurate math content retrieval, and precise extraction of image URLs. Ultimately, we choose to build on top of Resiliparse 4, leveraging its extensibility and robust data extraction capabilities as a foundation for our development.

Figure 2 illustrates a comparison of extraction results between Trafilatura and our enhanced version of Resiliparse. Our tool successfully extracts both the mathematical equations and image URLs, as highlighted in the red boxes in the screenshot from a Wikipedia webpage.

[Figure 1]

From trafilatura From Ours

Integral form\n[edit]Gauss's law may be expressed as:[6]\nwhere ΦE is the electric flux through a closed surface S enclosing any volume V, Q is the total charge enclosed within V, and ε0 is the electric constant. The electric flux ΦE is defined as a surface integral of the electric field:\nwhere E is the electric field, dA is a vector representing an infinitesimal element of area of the surface,[note 2] and · represents the dot product of two vectors.

Integral form\n\n[Image_Link]//upload.wikimedia.org/wikipedia/commons/thumb/3/32/Electricflux-surface-example.svg/220px-Electric-flux-surface-example.svg.png [Image_Link] Electric flux through an arbitrary surface is proportional to the total charge enclosed by the surface. \n\nGauss's law may be expressed as:\n\n$\\Phi _{E}={\\frac {Q}{\\varepsilon _{0}}}$\n\n\nwhere ΦE is the electric flux through a closed surface S enclosing any volume V, Q is the total charge enclosed within V, and ε0 is the electric constant. The electric flux ΦE is defined as a surface integral of the electric field:\n\n$\\Phi _{E}=$ $\\scriptstyle _{S}$ $\\mathbf {E} \\cdot \\mathrm {d} \\mathbf {A}$\n\nwhere E is the electric field, dA is a vector representing an infinitesimal element of area of the surface,[note 2] and · represents the dot product of two vectors.

Figure 2: A comparative illustration of extraction results from a Wikipedia webpage using Trafilatura and our enhanced version of Resiliparse, highlighting the successful retrieval of mathematical equations and image URLs.

#### 3.1.3 High-Recall Filtering for Mathematical Content

Building upon the math content extracted previously, we have developed an initial filtering methodology inspired by DeepSeek-Math [62]. We begin by randomly sampling half a million data points from the OpenWebMath [53] dataset, complementing these with an equal number of negative samples

4https://github.com/chatnoir-eu/chatnoir-resiliparse

from our earlier extracted math content. This creates a well-balanced training set, which is crucial for effective classifier development. Notably, we removed image URL links from the negative samples to standardize the data format across both positive and negative samples.

Leveraging this balanced dataset, we trained a fastText classifier specifically designed to filter mathematical content. This model significantly streamlined our dataset, reducing the volume from 57.2 billion samples to 9.5 billion. We set a probability threshold of 0.4 to prioritize recall over precision, allowing for a more inclusive filtering process. This lower threshold ensures that we capture a broader range of mathematical content, providing a foundation for subsequent refinement while minimizing the risk of excluding relevant data too early in our data curation pipeline.

#### 3.1.4 Deduplication

Removing duplicate webpages from training data is known to positively impact model performance [34]. To address this, we implement comprehensive deduplication strategies, including both content and URL deduplication, to eliminate redundant content.

Content Deduplication We employ MinHash [7], a fuzzy hashing technique widely recognized as a standard in data deduplication [56, 55]. Following the FineWeb methodology [55], we apply MinHashbased deduplication within each individual snapshot instead of global deduplication across all data snapshots, as suggested by experimental results. This approach significantly reduces computational demands and data volume. Subsequently, we extend our deduplication to encompass neighboring snapshot pairs, aiming to further compress the dataset size. This two-step process is projected to decrease the total dataset volume by approximately 43%, which equates to a substantial reduction in absolute data size from 9.5 billion to 5.4 billion samples.

URL Deduplication Following content deduplication, we apply an exact matching technique for URL deduplication, which further reduces the sample volume from 5.4 billion to 3.9 billion. This process is conducted within snapshots from the same year, balancing resource efficiency with data quality. When identical URLs are identified, we selectively retain content from the more recent snapshot, ensuring that the dataset not only eliminates duplicates but also maintains the most current and relevant information.

#### 3.1.5 Rule-based Filtering

For rule-based filtering methods, we initially explore several filtering rules used in the C4 [60] and OBELICS [33] datasets. However, we discover that most of these rules are not well-suited for a mathematical corpus, as they inadvertently filter out valuable content. Ultimately, we retain only the most essential rules to preserve the integrity of the mathematical data. The rules we apply include:

- • Remove Lorem Ipsum We remove all short-length documents containing “lorem ipsum” while preserving longer documents that may still contain useful content.
- • Punctuation Ratio The “punctuation ratio” rule proved unsuitable for the Chinese corpus, so we restricted its use to English documents, removing only those with a punctuation ratio exceeding 0.3.
- • NSFW Content We filtered out any documents containing NSFW words to ensure the dataset’s appropriateness for academic and research use.
- • Unicode Errors Documents containing Unicode errors were excluded to maintain the overall quality and usability of the dataset.

Due to the thorough data filtering applied in the earlier steps, we end up removing only about 3% of the samples based on the above rules, which resulted in 3.8 billion remaining samples.

#### 3.1.6 High-Precision Filtering for Mathematical Content

Our initial fastText classifier (Sec. 3.1.3), employed positive samples from the OpenWebMath dataset [53] and negative samples from our own data extraction from CommonCrawl. To enhance the accuracy of our labeling process, we leveraged the LLaMA3-70B-Instruct model [17], employing prompt formats inspired by the FineWeb-Edu dataset [45]. This allowed us to score the mathematical quality of each sample on a scale from 0 to 10. Table 6 in Appendix A displays the full prompt.

From the data remaining after rule-based filtering, we randomly sample approximately one million entries. Using the vLLM inference engine [32], we assign math quality scores and apply a threshold of 6 to select positive samples for training our updated fastText classifier. This process yielded 640,000 positive samples. We then randomly select an equivalent number of 640,000 negative samples as outlined in Sec. 3.1.3. These positive and negative samples are combined to train the new fastText classifier.5

During fastText training, we implement data cleaning rules to optimize the model’s performance for mathematical content. Mathematical texts pose unique challenges due to specialized terminology, symbols, formulas, and numeric data, which differ from typical natural language and require more refined preprocessing techniques.

Our goal is to standardize and simplify the input training data while preserving essential mathematical information. Key considerations include maintaining consistency in token representation, minimizing noise from extraneous characters, and standardizing numeric values. The following steps reflect this approach:

- • Utilizing the SpaCy English language model (en_core_web_sm), we preprocess the input text, tokenize it, and process each token by converting it to its lowercase and lemmatized form. Common placeholders are replaced, certain non-alphanumeric characters are removed, and patterns of special characters like dashes and underscores are normalized. We also strip any unnecessary whitespace, ensuring the text is well-prepared for downstream processing.
- • All numeric values are replaced with the <NUM> placeholder to standardize the representation, and line breaks along with carriage returns are removed. Tokens exceeding 100 characters in English are discarded.

These preprocessing steps notably enhance the classifier’s performance. For evaluation, we use all samples in the Geometry3K [47] benchmark as positive examples of mathematical content. With our refined preprocessing techniques, fastText’s accuracy increases from 48.74% to 72.15%.

#### 3.1.7 Evaluation on Text-Only Filtering

To provide a preliminary evaluation of the quality of our filtered dataset, we continue pretraining a deepseek-coder-1.3b-base model for one epoch using the filtered mathematical content in Sec. 3.1.6, excluding image URLs. We validate the effectiveness of our math-related filtering with a few-shot evaluation using the GSM8K [12] and the STEM sections of the MMLU [23] benchmark.

- Table 1: Evaluation of models on GSM8K and MMLU (STEM). The baseline is the deepseek-coder1.3b-base without any training.

Training Corpus GSM8K MMLU (STEM)

Baseline 4.8 25.6 OpenWebMath [54] 11.0 29.6 DeepSeekMath Corpus [61] 23.8 33.1 InfiMM-WebMath-40B (text) 26.1 35.6

As shown in Table 1, the model trained on our InfiMM-WebMath-40B text-only dataset demonstrates competitive performance compared to OpenWebMath and the DeepSeekMath Corpus, highlighting the high quality of our dataset and the effectiveness of our filtering procedures.

#### 3.2 Multimodal Data Construction

In Sec. 3.1.2, we successfully extract image URLs from each webpage as part of the multimodal dataset construction process. Following the OBELICS format [33], we organize the data into two parallel lists—one for text and one for image URLs—to preserve the original sequence of content on the webpages. After applying our filtering pipeline in Sec. 3.1, we obtain 24 million interleaved mathematical documents, consisting of 85 million image URLs and 40 billion text tokens. For the

5We also employ an LLM-based classifier for high-precision filtering, Appendix B shows the comparison.

release of the InfiMM-WebMath-40B dataset, we provide the interleaved text and image URLs, adhering to the OBELICS dataset format.

Next, we proceed to download the images from the remaining 85 million image URLs to construct our version of the multimodal dataset for experiments in subsequent sections. Upon initial inspection, we find that many URLs are duplicates, often linking to the same equations, mathematical diagrams, graphical plots, or common background images. To optimize this process, we first implement deduplication and filtering to focus on unique and relevant images.

To avoid downloading duplicates, we retain only unique image URLs. Additionally, we remove URLs that appear more than 10 times across samples or originate from documents containing over 100 images, as these are likely to be noisy or irrelevant. This refinement reduces the total to 23 million unique image URLs. We further apply keyword filtering, keeping only URLs that begin with “https” and excluding those containing terms such as “logo”, “banner”, “avatar”, or “icon”, reducing the set to 22 million URLs.

From this filtered set, we successfully download 14 million unique images6. These images are then reintegrated into their original positions within the interleaved image-text documents, ultimately yielding 24 million records with a total of 28 million downloaded images, due to the shared use of images across multiple webpages.

### 4 Experiments

We start by introducing our model architectures in Sec. 4.1, followed by a discussion of continue pre-training and instruction fine-tuning in Sec. 4.2. Our primary goal is to verify the effectiveness of the collected pre-training data. We conduct experiments to evaluate the trained model, using selected architectures and training stages. The evaluation results are discussed in Sec. 4.3.

#### 4.1 Model Architectures

Our model architecture design aligns with the latest advancements in vision-language learning [43, 38], incorporating a visual encoder followed by a vision-to-language connector and an LLM decoder. Specifically, we employ the SigLip model siglip-so400m-patch14-384 to extract visual features. To balance computational efficiency with performance, we use a 3-layer Perceiver Resampler [26] with 64 latents to reduce the number of tokens/features per image to 64. These visual token/feature embeddings are then concatenated with text embeddings before being fed into the LLMs. In this study, we experiment with two different LLMs from DeepSeek-Coder [22]: deepseek-coder-1.3b-base and deepseek-coder-7b-v1.5.

#### 4.2 Training Details

In this section, we detail the training data and processes involved in our three-stage training approach: modality alignment, continue pre-training using InfiMM-WebMath-40B, and instruction fine-tuning.

#### 4.2.1 Modality Alignment Stage

In this stage, we utilize general-purpose image-text pairs to align the visual encoder and the LLM via Perceiver Resampler. The primary objective is to minimize the domain gap between visual and linguistic modalities. To achieve this, we sample a 8 million image-text pair subset from the DFN-2B dataset [18] for the alignment training. During this stage, the vision encoder and LLM backbone are frozen, and training is focused on the Perceiver Resampler module. Training is conducted for one epoch using DeepSpeed Zero2, with the AdamW optimizer, configured with a cosine learning rate scheduler, a maximum learning rate of 1e−4, betas of (0.9,0.95), and a weight decay of 0.1.

#### 4.2.2 Continue Pre-training Stage

We further continue pre-training our models using the InfiMM-WebMath-40B dataset to enhance the model’s mathematical knowledge acquisition in a multi-modal setting. The training is conducted for

6While additional nodes, IPs, or proxies could improve download success, this dataset version strikes a good balance between resource use and size for our experiments.

- Table 2: Evaluation of models on MathVerse. Following the official MathVerse recommendation, we report the “w/o” scores based on the correctness of final answers.

Vision Only Human - 64.9 71.2 70.9 61.4 68.3 66.7

Base LLM

Text Dominant

Text Lite

Vision Intense

Vision Dominant

Model

All

Proprietary Models

GPT-4V N/A 39.4 54.7 41.4 34.9 34.4 31.6 Gemini-Pro N/A 23.5 26.3 23.5 23.0 22.3 22.2

Qwen-VL-Max N/A 25.3 30.7 26.1 24.1 24.1 21.4

Open-sourced Models SPHINX-Plus LLaMA2-13B 14.0 16.3 12.8 12.9 14.7 13.2

G-LLaVA LLaMA2-7B 15.7 22.2 20.4 16.5 12.7 6.6

InternLM-XC2 InternLM2-7B 16.5 22.3 17.0 15.7 16.4 11.0 Math-LLaVA Vicuna-13B 19.0 21.2 19.8 20.2 17.6 16.4 ShareGPT4V Vicuna-13B 17.4 21.8 20.6 18.6 16.2 9.7 LLaVA-NeXT LLaMA3-8B 19.3 24.9 20.9 20.8 16.1 13.8 LLaVA-NeXT Qwen-1.5-110B 24.5 31.7 24.1 24.0 22.1 20.7

MAVIS Mammoth2-7B 27.5 41.4 29.1 27.4 24.9 14.6 Our Models

InfiMM-Math DS-Coder-1.3B 26.9 37.1 30.2 29.2 24.4 13.7 InfiMM-Math DS-Coder-1.5-7B 34.5 46.7 32.4 38.1 32.4 15.8

one epoch using DeepSpeed Zero2, with the AdamW optimizer, configured with a cosine learning rate scheduler, a maximum learning rate of 5e−5, betas of (0.9,0.95), and a weight decay of 0.1. The context length for training examples is set to 4096, with a maximum of 32 images per example. During this stage, the visual encoder remains frozen, and training focuses on learning the Perceiver Resampler module (the visual-language connector) and the LLM.

#### 4.2.3 Instruction Fine-tuning Stage

In this stage of training, we fine-tune our models using instruction datasets, including PGPS9K [79], Geo170k [20], TABMWP [49], ScienceQA [48], Vflan [10], VisualWebInstruct, AI2D [31], ChartQA [50], DocVQA [51], DVQA [29], GeoQA [11], and MAVIS [82]. We find that incorporating unimodal text instruction datasets is crucial for enhancing the models’ instruction-following capabilities. Therefore, we also include pure text instruction datasets such as Math[36], MetaMathQA [75], DART-Math [65], and NuminaMath [6]. The objective of this stage is to acclimate the models to the common chat templates used in math VQA settings, thereby enabling them to better utilize the mathematical knowledge acquired in the previous stage.

We freeze the vision encoder and update the parameters of the Perceiver Resampler and LLMs. As in the previous stages, training is conducted using DeepSpeed Zero2 for one epoch, with the AdamW optimizer, configured with 2000 warmup steps, a maximum learning rate of 5e−6, betas of (0.9,0.95), a weight decay of 0.1, and cosine decay to 5e−7. The batch size is set to one per GPU, and the context length of the training examples is set to 4096. We utilize 32 A100-80G GPUs for the 1.3b models and 64 A100-80G GPUs for the 7b models. We refer to the resulting model as InfiMM-Math.

#### 4.3 Evaluations

In the following sections, we discuss the evaluation of our models on two widely adopted multi-modal math-only benchmarks: MathVerse [80] and We-Math [58].

- Table 3: Datasets ablations (CPT and IFT) using Deepseek-coder-1.3B.

MathVerse w/o score

CPT IFT

DSC-1.3B Mavis 20.2 DSC-1.3B ✓ Mavis 25.1 (+4.9) DSC-1.3B Extended 22.3 DSC-1.3B ✓ Extended 26.9 (+4.6)

Table 4: Datasets ablations (CPT and IFT) using Deepseek-coder-1.5-7B.

MathVerse w/o score

CPT IFT

DSC-1.5-7B Mavis 22.8 DSC-1.5-7B ✓ Mavis 27.1 (+4.3) DSC-1.5-7B Extended 23.8 DSC-1.5-7B ✓ Extended 29.1 (+5.3)

#### 4.3.1 Evaluations on MathVerse

We evaluate our models on the testmini set of MathVerse, which contains nearly 4,000 instances. Each problem in MathVerse is constructed with varying levels of multi-modal information, including text-dominant, text-lite, vision-intensive, vision-dominant, and vision-only categories. The questions can be either open-ended or multiple-choice.

We follow the three-stage evaluation pipeline outlined in MathVerse. In the first stage, models generate answers based on the provided prompts and images. In the second stage, GPT-4 is employed to extract answers from the models’ outputs. Finally, in the third stage, GPT-4 is used to determine whether the extracted answers match the ground truth.

Following the official MathVerse recommendations, we report the “w/o” score (based on the correctness of final answers) using the suggested CoT prompts. The results are shown in Table 2. Our 7B model outperforms all open-source models, including the 110B LLaVA-NeXT. It also surpasses Gemini-Pro and Qwen-VL-Max, trailing only behind GPT-4V. Our model excels in the Text-Dominant, Text-Lite, Vision-Intense, and Vision-Dominant sections, demonstrating its strong multi-modal understanding when processing both text and visual inputs. Our model underperforms in the Vision-Only section, likely due to the limitations of the vision encoder we employed, which processes input images at 384 × 384 resolution, whereas LLaVA-NeXT supports 336 × [(2,2),(1,2),(2,1),(1,3),(3,1)] resolutions using the AnyRes techniques [42]. Additionally, our vision encoder remains frozen during training, which restricts its ability to learn and adapt. In future work, we plan to develop improved learning algorithms and models to enhance the model’s visual understanding capabilities.

#### 4.3.2 CPT and IFT Dataset Ablations on MathVerse

In this section, we conduct ablation studies on models (1) trained with and without continue pretraining (CPT), and (2) models fine-tuned on the MAVIS dataset versus a more extensive instruction fine-tuning (IFT) dataset. Specifically, we compare models trained with and without our own mathematical multi-modal pre-training dataset, InfiMM-WebMath-40B. Additionally, we evaluate two IFT dataset configurations: (a) a combination of MAVIS-Caption-to-QA, MAVIS-ExistingDataset-Augment, MAVIS-Caption, MAVIS-DataEngine-Geometry, and MAVIS-Meta-Question (referred to as the MAVIS dataset); and (b) a broader set consisting of the MAVIS datasets along with Vflan, VisualWebInstruct, AI2D, CHARTQA, DOCVQA, DVQA, GEOQA, DART-Math, and Numina-Math (referred to as the Extended dataset).

As shown in Table 3, in the 1.3B model, CPT improves the MathVerse scores by 4.9 and 4.6 points when IFT is performed with MAVIS and Extended datasets, respectively. Similarly, Table 4 shows that in the 7B model, CPT improves the MathVerse scores by 4.8 and 5.3 points with MAVIS and Extended datasets, respectively. In contrast, using broader IFT datasets typically enhances model performance by approximately 2 points. These results highlight the significant mathematical capabilities imparted to the models through our InfiMM-WebMath-40B for CPT.

#### 4.3.3 Evaluations on We-Math

In this section, we compare models evaluated on the We-Math benchmarks. We-Math consists of 6.5K visual math questions and employs a four-dimensional metric based on the different levels of knowledge required to answer each question: Insufficient Knowledge (IK), Inadequate Generalization

- Table 5: Evaluation of models on the We-Math benchmark. AVG represents the primary metric of interest.

Model Base LLM AVG ↑ IK ↓ IG ↑ CM ↑ RM ↓ Proprietary Models

Qwen-VL-Max N/A 10.5 65.1 7.6 6.7 75.5 Gemini-1.5-Pro N/A 26.4 42.7 11.2 20.8 54.8

GPT-4V N/A 31.1 39.8 14.5 23.8 47.9 GPT-4o N/A 42.9 31.2 15.2 35.2 34.2

Open-sourced Models

- LLaVA-1.5 Vicuna-7B 6.5 - - - 85.6

- LLaVA-1.5 Vicuna-13B 8.4 - - - 78.1

- LLaVA-1.6 Vicuna-7B 3.3 78.3 2.5 2.1 89.1

- LLaVA-1.6 Vicuna-13B 5.2 69.1 3.2 3.6 86.9

LLaVA-NeXT Mammoth2-7B 13.4 - - - 71.0 LLaVA-NeXT Qwen-1.5-110B 19.2 - - - 66.0 DeepSeek-VL DeepSeek-7B 6.3 69.1 4.6 4.0 84.8

G-LLaVA Vicuna-13B 6.5 64.2 4.6 4.2 86.6 Math-LLaVA Vicuna-13B 11.1 - - - 72.8

InternLM-XC2 InternLM2-7B 12.7 56.4 10.5 7.4 77.6 Our Models

InfiMM-Math DeepSeek-Coder-1.3B 13.1 56.2 9.1 9.3 73.7 InfiMM-Math DeepSeek-Base-7B 20.6 48.8 12.2 15.2 61.7

(IG), Complete Mastery (CM), and Rote Memorization (RM). We report results on the We-Math testmini set using all four metrics.

As shown in Table 5, our model, InfiMM-Math, surpasses all open-source models. Notably, our model with a 7B base LLM outperforms LLaVA models with 72B and 110B LLMs. When compared to math-specific models like LLaVA, Math-LLaVA, G-LLaVA and InternLM-XC2 of similar size, our model demonstrates significant improvements. For instance, compared to LLaVA-NeXT-7B and InternLM-XC2, we achieve an improvement of 7.2 and 7.9 points in AVG scores, respectively.

### 5 Conclusions

In this work, unlike most current works that focus on instruction-following datasets to enhance LLMs’ mathematical reasoning, we introduce InfiMM-WebMath-40B, the first open source largescale multimodal interleaved mathematical pretraining dataset, addressing a critical gap in the multimodal research community. We provide detailed descriptions of our data collection process and publicly release the dataset to support open-source research. Our model, InfiMM-Math, demonstrates exceptional performance on the MathVerse and We-Math benchmarks, highlighting the effectiveness of the proposed dataset.

For future work, we aim to further advance mathematical reasoning in Multimodal Large Language Models (MLLMs) by exploring more sophisticated components specifically designed for mathematical content. This includes developing enhanced vision encoders tailored to effectively process mathematical symbols, diagrams, and equations. Additionally, we plan to integrate reinforcement learning techniques to improve reasoning capabilities in mathematical contexts. Beyond these efforts, we will explore other innovative approaches to push the boundaries of mathematical understanding in MLLMs, addressing the unique complexities inherent in multimodal mathematical reasoning.

### References

- [1] Open AI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024.
- [2] Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet, 2024.
- [3] Anas Awadalla, Le Xue, Oscar Lo, Manli Shu, Hannah Lee, Etash Kumar Guha, Matt Jordan, Sheng Shen, Mohamed Awadalla, Silvio Savarese, Caiming Xiong, Ran Xu, Yejin Choi, and Ludwig Schmidt. Mint-1t: Scaling open-source multimodal data by 10x: A multimodal dataset with one trillion tokens, 2024.
- [4] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. In The Twelfth International Conference on Learning Representations, 2024.
- [5] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics, 2023.
- [6] Edward Beeching, Shengyi Costa Huang, Albert Jiang, Jia Li, Benjamin Lipkin, Zihan Qina, Kashif Rasul, Ziju Shen, Roman Soletskyi, and Lewis Tunstall. Numinamath 7b tir. https://huggingface. co/AI-MO/NuminaMath-7B-TIR, 2024.
- [7] Andrei Z Broder. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171), pages 21–29. IEEE, 1997.
- [8] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
- [9] Shihao Cai, Keqin Bao, Hangyu Guo, Jizhi Zhang, Jun Song, and Bo Zheng. Geogpt4v: Towards geometric multi-modal large language models with geometric image generation, 2024.
- [10] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024.
- [11] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 513–523, 2021.
- [12] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [13] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.
- [14] Together Computer. Redpajama: An open source recipe to reproduce llama training dataset, April 2023. Software.
- [15] Common Crawl. Common crawl - open repository of web crawl data. https://commoncrawl.org/, 2008.
- [16] Google DeepMind. Ai achieves silver-medal standard solving international mathematical olympiad problems. https://deepmind.google/discover/blog/ ai-solves-imo-problems-at-silver-medal-level/, 2024.
- [17] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [18] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander T Toshev, and Vaishaal Shankar. Data filtering networks. In The Twelfth International Conference on Learning Representations, 2024.
- [19] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.
- [20] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-modal large language model, 2023.
- [21] Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. Does fine-tuning llms on new knowledge encourage hallucinations? arXiv preprint arXiv:2405.05904, 2024.
- [22] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.

- [23] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.
- [24] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [25] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.
- [26] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–

4664. PMLR, 2021.

- [27] Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hérve Jégou, and Tomas Mikolov. Fasttext. zip: Compressing text classification models. arXiv preprint arXiv:1612.03651, 2016.
- [28] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016.
- [29] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656, 2018.
- [30] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning, 2023.
- [31] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.
- [32] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [33] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelics: An open web-scale filtered dataset of interleaved image-text documents, 2023.
- [34] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499, 2021.
- [35] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022.
- [36] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large scale language model society, 2023.
- [37] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al. Datacomp-lm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024.
- [38] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.
- [39] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models, 2024.
- [40] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023.
- [41] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024.
- [42] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [43] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [44] Naiming Liu, Shashank Sonkar, Myco Le, and Richard Baraniuk. Malalgoqa: A pedagogical approach for evaluating counterfactual reasoning abilities, 2024.
- [45] Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu, May 2024. Software.
- [46] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024.
- [47] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning, 2021.

- [48] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.
- [49] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In The Eleventh International Conference on Learning Representations, 2023.
- [50] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning, 2022.
- [51] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [52] Saeid Naeini, Raeid Saqur, Mozhgan Saeidi, John Giorgi, and Babak Taati. Large language models are fixated by red herrings: Exploring creative problem solving and einstellung effect using the only connect wall dataset, 2023.
- [53] Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. arXiv preprint arXiv:2310.06786, 2023.
- [54] Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text, 2023.
- [55] Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024.
- [56] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023.
- [57] Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving, 2020.
- [58] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Yifan Zhang, Xiao Zong, Yida Xu, Muxi Diao, Zhimin Bao, Chen Li, and Honggang Zhang. We-math: Does your large multimodal model achieve human-like mathematical reasoning?, 2024.
- [59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [60] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019.
- [61] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.
- [62] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [63] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.
- [64] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models, 2024.
- [65] Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. arXiv preprint arXiv:2407.13690, 2024.
- [66] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024.
- [67] Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin, Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan, Quanzeng You, and Hongxia Yang. Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning, 2024.
- [68] Zengzhi Wang, Rui Xia, and Pengfei Liu. Generative ai for math: Part i – mathpile: A billion-token-scale pretraining corpus for math. arXiv preprint arXiv:2312.17120, 2023.
- [69] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [70] Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hannaneh Hajishirzi, Yejin Choi, and Kyunghyun Cho. Naturalproofs: Mathematical theorem proving in natural language. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021.

- [71] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, and Yu Qiao. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning, 2024.
- [72] Huajian Xin, ZZ Ren, Junxiao Song, Zhihong Shao, Wanjia Zhao, Haocheng Wang, Bo Liu, Liyue Zhang, Xuan Lu, Qiushi Du, et al. Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. arXiv preprint arXiv:2408.08152, 2024.
- [73] Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, and Jian Guo. Chartbench: A benchmark for complex visual reasoning in charts, 2024.
- [74] Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, et al. Internlm-math: Open math large language models toward verifiable reasoning. arXiv preprint arXiv:2402.06332, 2024.
- [75] Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024.
- [76] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning, 2023.
- [77] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.
- [78] Jiaxin Zhang, Zhongzhi Li, Mingliang Zhang, Fei Yin, Chenglin Liu, and Yashar Moshfeghi. Geoeval: Benchmark for evaluating llms and multi-modal models on geometry problem-solving, 2024.
- [79] Ming-Liang Zhang, Fei Yin, and Cheng-Lin Liu. A multi-modal neural geometric solver with textual clauses parsed from diagram. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 3374–3382, 2023.
- [80] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?, 2024.
- [81] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739, 2024.
- [82] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, Peng Gao, and Hongsheng Li. Mavis: Mathematical visual instruction tuning, 2024.
- [83] Zihao Zhou, Shudong Liu, Maizhen Ning, Wei Liu, Jindong Wang, Derek F. Wong, Xiaowei Huang, Qiufeng Wang, and Kaizhu Huang. Is your model really a good math reasoner? evaluating mathematical reasoning with checklist, 2024.
- [84] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text, 2023.
- [85] Zeyuan Allen Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. arXiv preprint arXiv:2309.14316, 2023.

### A Using Prompting with Llama-3-70B for Mathematical Annotation

Table 6: Prompt for evaluating mathematical content using Llama-3-70B following FineWebEdu [45].

Below is an extract from a web page. Evaluate the mathematical value of the extract and its potential utility as a teaching resource in a mathematical context using the additive 10-point scoring system described below. Points accumulate based on the satisfaction of each criterion, with special attention to the presence and quality of mathematical equations:

- - 0 points if the extract includes no mathematical content, such as only provides historical context, summarizes an article’s abstract, or exclusively features a person’s resume.
- - 1-2 points if the extract offers rudimentary information on mathematical subjects, even if interspersed with irrelevant material such as advertisements or non-academic content.
- - 2-4 points if the extract touches upon mathematical topics without rigorous adherence to academic standards and contains a mix of mathematical and non-mathematical content, or if the presentation is haphazard and the writing lacks clarity.
- - 4-6 points if the extract presents key concepts pertinent to educational curricula and includes mathematical equations, albeit potentially non-comprehensive or alongside superfluous information. It should resemble a mathematical text, such as an introductory section of a textbook or a basic tutorial.
- - 6-8 points if the extract is highly relevant to mathematics, is well-structured, and offers a clear exposition, including a significant number of mathematical equations and solutions. It should be akin to an in-depth textbook chapter or tutorial, with a strong focus on mathematical content and minimal unrelated information.
- - 8-10 points if the extract exhibits exceptional mathematical merit, characterized by detailed explanations, a comprehensive array of mathematical equations, and a coherent, accessible writing style that provides profound insights into mathematical theories and applications. The extract: <EXAMPLE>. After examining the extract: - Briefly justify your total score. - Conclude with the score using the format: "mathematical score: <total points>"

### B Ablation Studies on High-Precision Mathematical Content Filtering

In this section, we examine the efficacy of two classifiers—LLM-based and fastText-based—focusing on high-precision mathematical content filtering. The comparison utilizes the DeepSeek-Coder 1.3B model, which we trained on a dataset previously introduced in Sec. 3.1.3 with a sequence length of 4096. This model was trained to score documents based on their relevance to mathematical content on a scale from 0 to 10.

We conduct the continue pretraining of the DeepSeekCoder 1.3B model using datasets filtered by both the LLM- and fastText-based classifiers. Table 7 shows the performance results. The results highlight a length bias in the LLM-based method, which tends to favor longer documents, averaging 2,500 tokens, compared to 1,700 tokens for the FastText filter. The length bias associated with the LLM-based classifier has adversely impacted the dataset’s performance on the GSM8K dataset. As indicated in the table, the LLM-filtered dataset achieved lower accuracy (17.5%) on the GSM8K dataset compared to the fastText-filtered dataset (20.2%). This decrease in performance indicates that the LLM’s preference for longer documents may not align well with the requirements of datasets like GSM8K, which demand concise and precise mathematical descriptions.

Given these insights, we have decided to continue utilizing the fastText classifier for high-precision filtering in our ongoing research. Nonetheless, the implications of the LLM-based classifier require further investigation to fully understand and address its biases.

- Table 7: Ablations on the high-precision filtering. The “Text Avg Length” column indicates the averaged document length after filtering by each respective classifier.

MMLU (STEM) GSM8K Text Avg Length LLM-Classifier 32.8 17.5% 2500

FastText-Classifier 31.1 20.2% 1700

