# arXiv:2408.07246v6[cs.LG]26Oct2025

## ChemVLM: Exploring the Power of Multimodal Large Language Models in Chemistry Area

Junxian Li1,2Di Zhang2,3Xunzhi Wang2,4Zeying Hao2,5Jingdi Lei2, Qian Tan2,5, Cai Zhou2, Wei

Liu1,2, Yaotian Yang2, Xinrui Xiong2, Weiyun Wang2,3, Zhe Chen2, Wenhai Wang2, Wei Li2, Mao Su2, Shufei Zhang2, Wanli Ouyang2, Yuqiang Li2†, Dongzhan Zhou1*

1Shanghai Jiao Tong University, 2Shanghai Artificial Intelligence Laboratory, 3Fudan University,

4Nankai University, 5University of Science and Technology of China, lijunxian0531@sjtu.edu.cn, di.zhang@ustc.edu, xunzhi@mail.nankai.edu.cn, haozeying@mail.ustc.edu.cn, {liyuqiang, zhoudongzhan}@pjlab.org.cn Abstract

reaction equations, and other related phenomena. These limitations can impede the performance of comprehensive multimodal chemical reasoning (MMCR) tasks, leading to potential misinterpretations or misleading hallucinations. Moreover, chemists often spend significant time manually redrawing chemical images using software like ChemDraw (Li et al. 2004) to convert them into machine-readable formats such as SMILES (Weininger 1988) or IUPAC names (Favre and Powell 2013). Although traditional Chemical Optical Character Recognition (OCR) models, especially transformer-based ones like MolScribe (Qian et al. 2023b) and Decimer (K Rajan 2020), have achieved remarkable success in converting chemical images into SMILES (Qian et al. 2023b; Rajan, Zielesny, and Steinbeck 2021), their capabilities are limited to modality conversion and thus fails to address multimodal chemical reasoning (MMCR) tasks. For MMCR tasks, naive modality conversion was inadequate indeed, a more sophisticated capability that can offer a deeper insight into the non-text data is still required for the community. Therefore, a model that can seamlessly bridge visual and textual information in chemistry is desired.

Large Language Models (LLMs) have achieved remarkable success and have been applied across various scientific fields, including chemistry. However, many chemical tasks require the processing of visual information, which cannot be successfully handled by existing chemical LLMs. This brings a growing need for models capable of integrating multimodal information in the chemical domain. In this paper, we introduce ChemVLM, an open-source chemical multimodal large language model specifically designed for chemical applications. ChemVLM is trained on a carefully curated bilingual multimodal dataset that enhances its ability to understand both textual and visual chemical information, including molecular structures, reactions, and chemistry examination questions. We develop three datasets for comprehensive evaluation, tailored to Chemical Optical Character Recognition (OCR), Multimodal Chemical Reasoning (MMCR), and Multimodal Molecule Understanding tasks. We benchmark ChemVLM against a range of open-source and proprietary multimodal large language models on various tasks. Experimental results demonstrate that ChemVLM achieves competitive performance across all evaluated tasks.

While existing multimodal Large Language Models (MLLMs) (Liu et al. 2023, 2024; OpenAI 2023; Bai et al. 2023a; Young et al. 2024; Wang et al. 2023a) excel at processing diverse data types, their generic nature falls short in the chemical domain due to limited specialized knowledge. This gap necessitates a model specifically designed for chemistry that integrates textual and visual information, thereby enhancing research and analytical efficiency for chemists, overcoming the limitations of generalist approaches by providing precise, domain-specific insights.

Code — https://github.com/lijunxian111/ChemVlm Training Data — https://huggingface.co/datasets/di-zhang-

fdu/chemvlm-sft-datasets

Test Data — https://github.com/lijunxian111/ChemVlm

### Introduction

Large Language Models (LLMs) have been widely adopted in various scientific domains due to their high potential to accelerate scientific discovery (Zhang et al. 2024a; M. Bran et al. 2024; Jablonka et al. 2024; Zhao et al. 2024; Dagdelen et al. 2024). However, despite their promising potential to advance research, purely text-based models are limited in handling the diverse multimodal data encountered in the fields of chemistry, including molecular structures,

To address this issue, we introduce ChemVLM*, an opensource multimodal large language model specifically designed for the chemical domain, built upon the ViT-MLPLLM architecture (Liu et al. 2023). Our model combines a state-of-the-art Vision Transformer (ViT) (Dosovitskiy et al. 2020) for robust image encoding with ChemLLM (Zhang et al. 2024a), a domain-specific Large Language Model

*Corresponding author Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

*This work was done during their internship at Shanghai Artificial Intelligence Laboratory.

trained on billions of tokens from high-quality chemical data. This architecture effectively bridges the gap in domainspecific knowledge needed to process multimodal chemical tasks. A Multi-Layer Perceptron (MLP) layer is adopted as a projector to perform image-text alignment. We implement a two-stage supervised fine-tuning strategy to enhance model performance. In the first stage, we freeze the LLM component to focus on modality alignment. In the second stage, we unfreeze and update all parameters to adapt the model to downstream tasks. This structured approach ensures that ChemVLM achieves competitive performance across a variety of chemical tasks.

To rigorously evaluate the performance of chemical domain models, we introduce three specialized datasets: ChemOCR, MMCR-Bench, and MMChemBench. Each dataset is designed to assess distinct aspects of model capabilities. Specifically, ChemOCR focuses on modality conversion, offering bilingual questions that challenge models to recognize the SMILES format from molecular images, directly testing OCR capabilities within a chemical context. MMCR-Bench is mainly derived from the chemistry section of the Chinese college entrance examination and expands to assess models on their ability to solve complex multimodal chemical reasoning (MMCR) problems. MMChemBench, extended from ChemBench (Zhang et al. 2024a), includes molecule caption and molecule property prediction tasks with multimodal information, providing a comprehensive view of a model’s capability on Multimodal Molecule Understanding. Additionally, we further validate our model’s performance using established open-source benchmarks (Lu et al. 2022; He et al. 2024b), ensuring a comprehensive assessment across diverse chemical tasks.

In our assessment, ChemVLM demonstrated significant improvements over baseline models and achieved state-ofthe-art (SOTA) performance on several tasks, surpassing GPT4 vision models (OpenAI 2023, 2024c). These results underscore the model’s substantial advantages in chemical image recognition and image-text MMCR tasks. Our key contributions are as follows:

- 1. A multimodal LLM dedicated to chemistry. We propose ChemVLM, an open-source MLLM for the chemistry area. In detail, We present and implement a Vision Transformer-Multi-Layer Perceptron-Large Language Model (ViT-MLP-LLM) architecture specifically tailored for chemical domain visual-language processing.
- 2. A comprehensive dataset suite. We develop three new datasets (ChemOCR, MMCR-Bench, and MMChemBench) with diverse tasks to evaluate the capability of visual-language models in the chemical domain.
- 3. A variety of reliable evaluation. We assess the performance of different models on various chemical tasks, including benchmarks collected by ourselves and the opensourced ones. Results show that our model exhibits desirable performance in multimodal chemistry understanding and reasoning.

We hope ChemVLM can advance chemical research by bridging the gap between multimodal data interpretation and

domain-specific knowledge and inspire future works in this field.

|ASSISTANT: I think the chemical structure in this image is C1=C[Se]N[Se]1. So there were two Selenium atoms.|
|---|

Chemical Large Language Model Projector

Embedding

Vision

| | |
|---|---|
|USER: How many <image><br><br>atoms are in this molecule<br><br><image>?| |

Transformer

[Figure 1]

[Figure 2]

Figure 1: Overall architecture of ChemVLM. ChemVLM combines the advantage of an advanced vision transformer and a large language enriched with chemical knowledge, ensuring the strong ability of multimodal chemical knowledge understanding and reasoning.

### Related Work

Overview of Multimodal Large Language Models. In addition to textual content, visual information encompasses a wealth of data, and integrating these modalities through large language models (MLLMs) has demonstrated superior comprehension, reasoning, and generation capabilities compared to traditional models. Notable MLLMs include GPT-4V (OpenAI 2023), which extends GPT-4 with a visual module, and the Gemini series (Team et al. 2023; Reid et al. 2024), which processes audio, video, and text inputs with enhanced longcontext understanding. Proprietary models like QwenVLPlus/Max (Bai et al. 2023a), MM1 (McKinzie et al. 2024), and Claude-3V (Anthropic 2024) also exhibit exceptional multimodal performance. Open-source MLLMs, such as the LLaVA series (Liu et al. 2023, 2024), VisionLLM (Wang et al. 2024a), CogVLM (Wang et al. 2023b), LLaMA-Adapter V2 (Gao et al. 2023), and ShareGPT4V (Chen et al. 2023), continue to evolve but still lag behind their proprietary counterparts in performance.

Overview of Chemical OCR and MMCR. Various ways have been attempted by prior works to extract visual information from the chemical images. Works on diagram parsing focused on the segmentation of molecular images (Joshua et al. 2019; EJ Beard 2020) and the recognition of their chemical structures (Qian et al. 2023b; IV Filippov 2009; K Rajan 2020). Some researchers want to understand the relationships between the molecules, i.e., reaction schemes. Wilary and Cole proposed ReactionDataExtractor (DM Wilary 2021) to extract reaction schemes from the diagrams. A method called Chemgrapher (M et al. 2020) suggests dealing with the problem in a modular fashion by using a segmentation algorithm to segment the images containing chemical graphs to detect atom locations, bonds, and charges. Only a few try to predict the properties of molecules. Chithrananda et al. make one of the first attempts to systematically evaluate transformers on

Multimodal Molecule Understanding tasks such as molecular property prediction (Seyone Chithrananda 2020). We provide an MLLM method to help solve the problem of chemical OCR and can work as a chemical properties answer assistant.

### Architecture of ChemVLM

As shown in Figure 1, ChemVLM follows the architecture from LLaVA (Liu et al. 2023, 2024) in the fashion of "ViTMLP-LLM". This framework integrates a Vision Transformer (ViT) followed by Multi-Layer Perceptron (MLP) components, seamlessly connected to a Large Language Model (LLM). We adopt InternViT-6B (Chen et al. 2024c) as the vision encoder and ChemLLM-20B (Zhang et al. 2024a) as the language model, as it demonstrates strong chemical capabilities. The projector is an MLP to convert the visual feature into the language embedding space, whose weights are randomly initialized. The visual encoder takes images of 448 × 448 resolution as input and extracts high-level features from them. The visual features are then transformed into the same dimension of the language tokens, aligning them with the text space. Simultaneously, the textual input passes through the tokenizer to be converted into a sequence of tokens. The image and text tokens are then concatenated to form a unified token sequence and then fed into the LLM. The LLM synthesizes the multimodal tokens to generate a cohesive and contextually relevant response. This pipeline enhances the model’s capability in complex chemical environments.

### Data Composition

This section details our approach to enhancing our model’s recognition and understanding of images of molecules and reactions, and multimodal reasoning tasks in chemistry. We aim to improve the model’s capability in real-world multimodal scenarios by leveraging diverse datasets and educational resources. We visualize the data preparation work in Figure 3. The data production process is as follows. (1). We collect raw data encompassing various tasks from reliable open-source chemical datasets and databases. (2). We apply diverse image transformation techniques for data augmentation, including style transformations (e.g., handwritten styles, graph paper backgrounds), rotations, blurring, and other enhancement techniques. (3). A variety of templates are carefully designed to cover different task scenarios and improve diversity (please refer to the Appendix for template details). (4). These templates are filled with raw data to create high-quality QA pairs, which will be used to train and evaluate MLLMs. Here we show the data distribution of our train data and benchmarks in Figure 2. Detailed explanation can be found in the Appendix.

#### Molecule Subset

To improve the model’s ability on Multimodal Molecule Understanding and Recognition, we collect images from different sources to cover a broad range of styles. The sources are demonstrated as follows.

• Hand-drawn style. We utilize the DECIMER HDM (Brinkhaus et al. 2023) dataset that contains over 7,000 hand-drawn molecular structure images. This

17.67%

MMChem OCR

Chem Exam General

35.06%

25.93%

3.45%

37.04%

MMCRbench

Benchmark

Train Data

General Chinese

20.13%

MMChem Bench

Chem OCR

37.04%

18.60%

Chem

Exam(text)

5.09%

WebChem(text)

Figure 2: Data distribution of our train data and benchmarks.

dataset includes drawings created with different inks and types of pens.

- • Scanned and photographed style. This subset encompasses images with potential artistic enhancements and photographic distortions. We used the dataset compiled by Molscribe Project (Qian et al. 2023b), including molecular images sourced from published literature.
- • Electronic document style. This subset is characterized by images generated by chemical drawing tools, including ChemDraw (Li et al. 2004), RDKit (Landrum 2014) and Indigo†. The images are created with varying line styles and color presets.

#### Reaction Subset

To enhance our model’s visual understanding of a broad spectrum of chemical reaction schemes and synthesis processes, we have gathered a diverse collection of images, ranging from simple reaction images to complex synthesis pathway diagrams. The sources are demonstrated as follows.

- • Simple chemical reaction images. This subset contains two types of images. The first consists of inorganic chemical reaction equations, primarily sampled from the PEACE (Zhang et al. 2024b) dataset, which includes reaction equation image fragments extracted from the literature. The second focuses on organic chemical reactions, which are mainly based on reaction scheme images collected from organic chemistry literature (Qian et al. 2023a). Additionally, we synthesize a batch of reaction scheme images using RDKit (Landrum 2014) with data from the USPTO-50K (Maziarz 2024) dataset.
- • Synthesis pathway diagrams. This subset also includes two types of images. The first features simpler synthesis routes with fewer than five steps gathered from real literature (Qian et al. 2023a). The second involves more complex diagrams called Total Synthesis, which aims to create complex large molecules, typically natural products found in biological organisms, from commercially available small molecules artificially. The data are collected from The Organic Chemistry Data (Myers 2024) and SynArchive website‡.

†A universal cheminformatics toolkit, utilities and database search tools can be found in https://github.com/epam/Indigo

‡https://synarchive.com/

| |High quality data collection| |
|---|---|---|
| | | |

| | | |
|---|---|---|
| |Preprocessing| |

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Molecule： Image, SMILES, molecule caption…

[Figure 7]

Molecule image style transfer

[Figure 8]

Reaction: Image, description…

[Figure 9]

[Figure 10]

Exclusive

RDKIT filter

Valid SMILES and expressions checking

database and

Multimodal Chemistry Exam questions

[Figure 11]

open-source data

Text cleaning

| | | |
|---|---|---|
| |Well-prepared data| |

| | | |
|---|---|---|
| |Different styles of templates designing| |

|Human: Can you identify the molecule in the image in SMILES format? <image> AI assistant: From my vantage point, the molecular<br><br>formula in the image, if represented with SMILES, would<br><br>be CCOC(=O)C1=C(N)SC=C1C1=CC=C(C2=CC=CC=C2)C=C1.|
|---|

[Figure 12]

Generated templates

“Please answer the following question:

[Figure 13]

{}” “please give a SMILES format.” “What can you tell me about {}?” “From my perspective, we {}” “The answer is {}, solution: {}”

|Human: 请根据指令和图片回答问题。 <image> 某铁氮化合物晶胞如图1, 下列说<br><br>法错误的是() A… B… C… D… AI assistant: 答案：A<br><br>解析：….|
|---|

[Figure 14]

Q-A bilingual template designing

“I think, the molecule in the image,

represented with SMILES, is {SMILES}”

- Figure 3: Overview of our data composition work. This multi-step process ensures our model’s good performance and a comprehensive evaluation.

#### Multimodal Chemical Reasoning (MMCR) Tasks

Existing multimodal chemical models (Qian et al. 2023b; IV Filippov 2009; K Rajan 2020; Qian et al. 2023a; Zhang et al. 2024b) primarily focused on modality conversion tasks, such as OCR and molecule captioning, and have achieved a remarkable progress. However, with large language models increasingly being used as tools for knowledge reasoning and scientific discovery (Mirza et al. 2024), the need for a deeper, more integrated understanding of multimodal knowledge has become critical. Some multimodal questionanswering datasets (Mirza et al. 2024; Luo et al. 2023) have been involved in the field of chemistry, however, these efforts often focus solely on evaluating the model’s ability to recognize and statistically analyze local structures or sub-patterns within non-text modalities, such as identifying specific functional groups within molecular structures. For tasks like solving chemistry problems in high school chemistry exams, the model needs to possess strong logical reasoning skills, be able to capture critical information from the questions, and combine it with extensive domain-specific knowledge to provide accurate and reasonable answers.

Given the importance of the problem above, we utilize examination data within the chemistry discipline to build a Q&A dataset. Inspired by C-MHChem (Zhang et al. 2024a), we carefully curate and deduplicate a dataset containing 200,000 high-quality multimodal chemistry questions from the OpenDataLab platform (He et al. 2024a), covering Chi-

nese education from secondary school to graduate levels§. It includes various problem types such as multiple choice, fillin-the-blank, and short answer. The questions are designed to assess diverse skills of test-takers, including error correction, knowledge-based Q&A, complex reasoning, and experiment protocol design. This dataset aims to equip the model with a deeper understanding of chemistry, enabling it to tackle complex multimodal challenges in real-world scenarios.

#### Data Quality Enhancement

To ensure higher data quality and improve the effectiveness of training, we implement a series of data quality improvement techniques, as depicted in Figure 3. Considering the importance of diverse training data for learning meaningful image-text interactions, we apply RanDepict (Brinkhaus et al. 2022) to incorporate variation in image styles. This includes style transformations (e.g., handwriting style, grid paper backgrounds), rotations, blurring, and other data augmentation techniques for molecular images. Additionally, we perform validation checks on the SMILES fields in the text information and remove any irrelevant symbols to maintain accuracy. Furthermore, we diversify the question-answer prompt templates by interacting with GPT-4 (OpenAI 2024a) to generate prompts in different linguistic styles, both in Chinese and

§The chemistry QA pairs from OpenDataLab are originally in Chinese, and we translate 10,704 of them into English using the Bing translator from pypi/translators.

English. These prompts are then used to construct the final question-answer pairs for supervised finetuning.

### Training

#### Training Strategy

ChemVLM’s training follows a two-stage paradigm inspired by InternVL (Chen et al. 2024a). The first stage focuses on aligning the image and text modalities, while the second stage involves supervised finetuning. During training, we segment images into pixel tiles with a maximum of 12, depending on the image’s aspect ratio and resolution. A context length of 2048 tokens is utilized to accommodate detailed responses, and the response formatting follows the prompt structure used in LLaVA 1.5 (Liu et al. 2024).

Image-Text Modal Alignment Training. In the first stage, we utilize a diverse multimodal dataset, encompassing both a general-purpose corpus and a specialized chemical corpus, to enhance the alignment of visual and textual representations. Specifically, we freeze the weights of LLM and the base visual encoder, training only the randomly initialized projector and the additional LoRA (Hu et al. 2021) layers in the visual encoder to imbue it with an understanding of chemical space. Finetuning with LoRA effectively reduces the number of trainable parameters and computational costs, while simultaneously lowering the risks of model overfitting and catastrophic forgetting.

Supervised FineTuning Training. In the second stage, we finetune the model using a combination of multimodal data and pure text data. The training dataset includes both specialized data from the field of chemistry and general-purpose corpora. In this process, since ViT and LLM account for a significant proportion of the parameters, LoRA is applied to finetune the LLM and ViT components to reduce training costs. Meanwhile, the projector, which has a much smaller parameter footprint, is fully trained using all parameters. The parameters used in this stage are derived from those merged and processed in the first stage.

#### Training Details

For each stage, the model is trained on 16 NVIDIA A100×80G GPUs for one epoch. The batch size is set to

- 4 and gradients are accumulated over 4 iterations. We use AdamW (Loshchilov and Hutter 2017) as the optimizer and

|Model&Method|Avg Sim. (%↑)|Tani@1.0 (%↑)<br><br>|
|---|---|---|
|Decimer Molscribe<br><br>|85.0 92.0<br><br>|77.3 89.1|
|Qwen-VL-Chat LLaVA-v1.5-13B InternVL-v1.5 Yi-VL-Plus GPT-4V ChemVLM-26B (ours)<br><br>|5.0 1.0 9.0 5.0 15.0 71.0|0.0 0.0 0.0 0.0 2.1 42.9<br><br>|

Table 1: Results on ChemOCR. Tanimoto similarities are written as Avg Sim. (%↑), and Tanimoto@1.0 written as Tani@1.0 (%↑). High is better for both metrics.

|Model &Method|ScienceQA Chem (%↑)<br><br>|CMMU Chem (%↑)|MMCRBench (%↑)<br><br>|
|---|---|---|---|
|Qwen-VL-Chat LLaVA-v1.5-13B InternVL-v1.5 Yi-VL-Plus GPT-4V ChemVLM-26B (ours)|65.3 43.6 74.0 68.1 71.9 71.2<br><br>|22.1 18.6 26.9 29.1 24.2 31.6|28.3 23.1 36.5 35.3 40.1 41.7<br><br>|

Table 2: Results on Chemisty part of CMMU, ScienceQA, and MMCR-Bench. We compare our model’s performance with several MLLMs and report the accuracy. The evaluation is conducted in a zero-shot manner.

Deepspeed bfloat16 (bf16) (Rasley et al. 2020) precision for efficient training. To handle distributed training, we consistently apply the Deepspeed ZeRO-3 (Rajbhandari et al. 2020) strategy. We adopt the chat template from InternLM2 (Cai et al. 2024) as the dialogue schema for the LLMs. Please refer to the supplementary material for more detailed information.

### Experiments

In this section, we conduct extensive experiments to assess the chemical capabilities of ChemVLM and other competing models on benchmarks of various tasks.

#### Evaluation Settings

To assess the multimodal capabilities of models on chemical tasks, we evaluate their performance on both publicly available and self-curated benchmarks. Our evaluation focuses on the three critical chemical tasks, including Chemical Optical Character Recognition (Chemical OCR), Multimodal Chemical Reasoning (MMCR), and Multimodal Molecule Understanding.

We utilize the chemistry part of open-source multimodal benchmarks, CMMU (He et al. 2024b) and ScienceQA (Lu et al. 2022) for MMCR evaluation. To evaluate the model’s capability of Multimodal Molecule Understanding, we construct a multimodal chemical benchmark named MMChemBench, derived from ChemBench (Zhang et al. 2024a) with two image-text tasks: molecule caption and molecular property prediction. Meanwhile, we employ other scientific disciplines within the CMMU (He et al. 2024b) framework to assess ChemVLM’s generalization capabilities in different areas. We also adopt Scibench (Wang et al. 2024b), a collegelevel chemistry test set consisting of purely textual data, to evaluate the model’s problem-solving abilities in more complex tasks.

Moreover, we build two datasets for a more comprehensive benchmarking called ChemOCR and MMCRBench. Specifically, ChemOCR includes 1000 chemical OCR image-text pairs collected from an open-source chemical database (Qian et al. 2023b). And MMCR-Bench includes 1000 high-quality multimodal chemistry exam questions from the Chinese college entrance examination.

[Figure 15]

- Figure 4: In the left figure, we compare ChemVLM with three other MLLMs on other subjects aside from chemistry on CMMU. In the right figure, we show results on the subsets related to chemistry on Scibench. The numbers represent the performance of ChemVLM.

#### Results on Chemical OCR Task

In this task, the models are expected to generate a corresponding SMILES string for each molecular image. We compare our model with previous end-to-end models dedicated to this task, including Decimer (Rajan, Zielesny, and Steinbeck 2021) and Molscribe (Qian et al. 2023b). These models are limited to generating SMILES and lack natural language processing capabilities. Therefore, the comparison is also conducted on various multimodal LLMs (MLLMs) including Qwen-VL-Chat (Bai et al. 2023b), LLaVA-v1.5-13B (Liu et al. 2023), InternVL-v1.5 (Chen et al. 2024b), Yi-VLplus (Young et al. 2024), and GPT-4V (OpenAI 2024b). We employ 1000 image-text pairs in ChemOCR to evaluate the model’s overall chemical OCR capabilities. We compute the Tanimoto similarity between the generated molecules and ground truth molecules with RDKIT (Landrum 2014), and report the average Tanimoto similarity and Tanimoto hit 1.0 (tanimoto@1.0), which measures the percentage of structures with 1.0 similarity.

The evaluation results are shown in Table 1. Our model exhibits strong performance on this task, outperforming all competing MLLM models. While ChemVLM’s performance is behind specialized models like Decimer and MolScribe, it offers greater versatility in handling a wider range of tasks, not limited to Chemical OCR. Notably, a 59% increase in average similarity and a 40.8 increase in tanimoto@1.0 can be found when compared with GPT-4V.

#### Results on MMCR Tasks

In addition to the OCR-related tasks, we test our model on question-answer tasks compiled from exams. The task involves data from three different sources. We utilize the chemical sections of the CMMU and ScienceQA datasets, which are collected from exams across various grade levels. Additionally, we include 1000 multimodal chemistry exam questions from our custom test set, MMCR-Bench. The total score is calculated based on the following criteria.

An answer gets one point when it matches all the right choices of a multiple-choice problem, else it gets zero point.

For a fill-in-the-blanks problem, an answer should make all the blanks right, and then get one point. A wrong blank means zero points for the whole problem. We complete this

process by prompting the Qwen-max API (Yang et al. 2024). Please refer to the supplement for more details.

At last, the total score is calculated by dividing the number of questions into points a model gets.

Results on the MMCR tasks are summarized in Table 2. Our model achieves state-of-the-art (SOTA) for the chemical part on the CMMU benchmark. Additionally, ChemVLM demonstrates strong performance in the chemistry section of ScienceQA, closely matching that of GPT-4V. Upon analyzing the ScienceQA dataset, we find that many of the questions rely on common sense, making them easier for MLLMs to solve. When it comes to MMCR-bench, ChemVLM shows more promising performance, which surpasses GPT-4V by 1.7% and achieves state-of-the-art performance.

|Model &Method<br><br>|molecule caption(%)<br><br>|property prediction(%)|
|---|---|---|
|ChemLLM* GPT-4*|92.6 96.3<br><br>|72.2 68.4|
|Qwen-VL-Chat LLaVA-v1.5-13B InternVL-v1.5 Yi-VL-Plus GPT-4V ChemVLM-26B (ours)|86.6 75.9 78.7 89.3 95.2 98.2<br><br>|56.1 19.6 32.2 27.4 38.6 80.9<br><br>|

Table 3: Results on MMChemBench. We report the total score using the method of calculating a score for choice problems. Notably, we report text-only LLMs’(*) performance on MMChemBench’s corresponding data source: ChemBench.

#### Results on Multimodal Molecule Understanding Tasks

We present MMChemBench, an extension of ChemBench, designed to evaluate our models’ performance in Multimodal Molecule Understanding tasks which include molecule caption and molecule property prediction. We also compare ChemVLM with ChemLLM and GPT-4 to assess the impact of incorporating visual modality into the model. The results are presented in Table 3, which demonstrate that ChemVLM achieves state-of-the-art performance in both tasks. A significant improvement is observed in the property prediction task compared to other MLLMs. The findings highlight that incorporating visual information greatly enhances the model’s ability to understand domain-specific knowledge, such as molecular structures.

#### Study on Multidisciplinary Competence

Our proposed ChemVLM is primarily trained on a proprietary chemistry dataset. To evaluate the model’s generalization capabilities, we tested it on test sets from other scientific fields within the CMMU framework. As shown in Figure 4, the experimental results indicate that ChemVLM performs

Question: 取1mL0.1 mol/L的AgNO3溶液进行如下实

验(实验中所用试剂浓度均为 0.1 mol/L) 下列说法不正确的

取 $1 mL 0.1 mol·L^{-1}$AgNO_3$ 溶液进行如下实验

是( )

(实验中所用试剂浓度均为 $0.1 mol·L^{-1}$)下列说法不

[Figure 16]

- A. 实验1.白色沉淀是难溶的 AgCl;
- B. 由实验2.说明 AgI 比 AgCl 更难溶;
- C. 若按1.3.顺序实验，看不到黑色沉淀;
- D. 若按2.1.顺序实验，看不到白色沉淀;

正确的是( ) 实验1.白色沉淀是难溶的 $textAgCl$; 由实验2.说明 $textAgI$ 比 $textAgCl$ 更难溶; 若按1.3.顺序实验，看 不到黑色沉淀; 若按2.1.顺序实验，看不到白色沉淀;

###### Ground truth：C

GPT-4v: 这道题目中，根据所提供的实验顺 序和反应物，我们来分析各个选项的正确性。

##### ChemVLM(ours): 【答案】C

首先……

【解析】A. ……

- C. 若按1.3.顺序实验，实验1生成的 AgCl在实验 3中会溶解，不会生成黑色沉淀Ag2S。因此看 不到黑色沉淀。这个描述也是正确的。
- D. ……

- B. ……
- C. 实验3.中，AgI 黄色沉淀与Na2S 溶液反应 生成Ag2S 黑色沉淀，若按1.3.顺序实验，也 能看到黑色沉淀，故C错误；
- D. ……故D正确；故选C。

综上所述，所有选项A、B、C和D都是正确 的，因此没有“不正确”的选项。……

[Figure 17]

Question: Take 1 mL of 0.1 mol/L AgNO3 solution and perform the following experiments (the concentration of reagents used in the experiments is all 0.1 mol/L). The incorrect statement among the following is ( )

[Figure 18]

- A. Experiment 1: The white precipitate is the insoluble AgCl.
- B. Experiment 2: This demonstrates that AgI is less soluble than AgCl.
- C. No black precipitate with sequence 1.3.
- D. No white precipitate with sequence 2.1.

Ground truth：C

[Figure 19]

[Figure 20]

[Figure 21]

GPT-4v:

ChemVLM(ours): 【Answer】C

In this problem, based on the provided experimental sequence and reactants, we will analyze the correctness of each option. First…… C. If the experiments are conducted in the order of 1,3, the AgCl produced in experiment 1

【Analysis】

- A. ……
- B. ……
- C. In Experiment 3, the yellow AgI precipitate reacts with Na2S solution to form black Ag2S precipitate. If experiments are conducted in the order of 1.3, black precipitate can still be observed, hence C is incorrect. D…… hence D is correct. Therefore，Correct answer is C

[Figure 22]

will dissolve in 3, and thus will not form the

black precipitate, Ag2S. Therefore, no black precipitate will be observed. This description is also correct. D. …… In summary, since all options A, B, C, and D are

correct, there are no incorrect choices. ……

- Figure 5: A qualitative comparison of answers on MMCR-Bench between GPT-4V and our ChemVLM. Mistakes within the answers are highlighted in red, whereas detailed and accurate parts are emphasized in green. Since this is a Chinese exam question, we prepare the original Chinese text and the English translation of it. This shows the strong MMCR capability of ChemVLM.

取 $1 mL 0.1 mol·L^{-1}$AgNO_3$ 溶液进行如下实验

(实验中所用试剂浓度均为 $0.1 mol·L^{-1}$)下列说法不

正确的是( ) 实验1.白色沉淀是难溶的 $textAgCl$; 由实验2.说明 $textAgI$ 比 $textAgCl$ 更难溶; 若按1.3.顺序实验，看 不到黑色沉淀; 若按2.1.顺序实验，看不到白色沉淀;

Question: Can you tell me what is the

molecule in this image, using SMILES format？

Ground truth：

CC1C=C(C=CN=1)C1C=CC2N=C(NC3C=C(C=C N=3)N3CCN(CC3)C(=O)COC)SC=2N=1

GPT-4v:

To identify the molecule in the figure and

generate its SMILES molecular formula,

we can analyze …… we can derive the following SMILES molecular formula： CCOC(=O)N1CCC(CC1)Nc2ncc3n(nc(c

3s2)c4ccccn4)C

This SMILES molecular formula represents the chemical structure of the molecule pictured.

ChemVLM(ours): From my perspective，the molecule shown in this image，when represented with SMILES, should be： CC1C=C(C=CN=1)C1C=CC2N=C(N

C3C=C(C=CN=3)N3CCN(CC3)C(=O

)COC)SC=2N=1

[Figure 23]

- Figure 6: A qualitative comparison of answers on ChemOCR on GPT-4V and our ChemVLM. Red and green parts are the same as Figure 5.

#### Visualization of Answers From Different MLLMs and Our ChemVLM

In this part, we present qualitative results to compare the outputs of our model with those from a strong proprietary MLLM, GPT-4V, as shown in Figure 5 and Figure 6. The results indicate that our model exhibits a better understanding of the questions and provides more accurate answers for chemical OCR and examination questions. This highlights ChemVLM’s proficiency in leveraging chemical knowledge to address multimodal problems.

### Conclusion and Future Work

In this study, we introduce ChemVLM, an open-source, multimodal model tailored for applications in chemistry. Comprehensive evaluations indicate that ChemVLM exceeds most existing methods, advancing the AI and chemistry fields.

Currently, ChemVLM primarily incorporates image modalities and lacks processing capabilities for molecular graph and time-series data, limiting its effectiveness in capturing complex chemical phenomena. Additionally, the high computational costs may hinder its wider adoption.

exceptionally well in other disciplines, surpassing renowned multimodal LLMs and InternVL-v1.5-26B. This suggests that large-scale training on high-quality problems enables our model to exhibit remarkable capabilities across different scientific domains.

In the future, we aim to enhance ChemVLM by adding new modalities, such as graph and time-series data. We also plan to explore more efficient training methods and develop a range of models with varying parameter sizes to suit diverse scenarios, thereby advancing LLM’s role in chemistry research.

#### Study on More Difficult Questions

To test the model’s reasoning capability on complex chemistry problems, we evaluated various models using the Scibench (Wang et al. 2024b) dataset. Scibench is a text-only benchmark designed for university-level science questions extracted from exercise books, including topics like quantum chemistry and physical chemistry. As shown in Figure 4, our model outperformed all models mentioned in their paper, except for the GPT-4 series. Notably, in the "Chemmic" and "Matter" subtasks, our model achieved better results than GPT-4. This highlights ChemVLM’s reasoning ability on purely textual, more difficult questions, despite being designed to address multimodal chemistry problems.

### Acknowledgements

This work is supported by Shanghai Artificial Intelligence Laboratory.

### References

Anthropic. 2024. The claude 3 model family: Opus, sonnet, haiku.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023a. Qwen-vl: A frontier large

vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023b. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Brinkhaus, H. O.; Rajan, K.; Zielesny, A.; and Steinbeck, C. 2022. RanDepict: Random chemical structure depiction generator. Journal of Cheminformatics, 14(1): 31.

Brinkhaus, H. O.; Zielesny, A.; Steinbeck, C.; and Rajan, K.

- 2023. DECIMER - Hand-drawn molecule images dataset.

Cai, Z.; Cao, M.; Chen, H.; Chen, K.; Chen, K.; Chen, X.; Chen, X.; Chen, Z.; Chen, Z.; Chu, P.; et al. 2024. Internlm2 technical report. arXiv preprint arXiv:2403.17297.

Chen, L.; Li, J.; Dong, X.; Zhang, P.; He, C.; Wang, J.; Zhao, F.; and Lin, D. 2023. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793.

Chen, Z.; Wang, W.; Tian, H.; Ye, S.; Gao, Z.; Cui, E.; Tong, W.; Hu, K.; Luo, J.; Ma, Z.; Ma, J.; Wang, J.; Dong, X.; Yan, H.; Guo, H.; He, C.; Shi, B.; Jin, Z.; Xu, C.; Wang, B.; Wei, X.; Li, W.; Zhang, W.; Zhang, B.; Cai, P.; Wen, L.; Yan, X.; Dou, M.; Lu, L.; Zhu, X.; Lu, T.; Lin, D.; Qiao, Y.; Dai, J.; and Wang, W. 2024a. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv:2404.16821.

Chen, Z.; Wang, W.; Tian, H.; Ye, S.; Gao, Z.; Cui, E.; Tong, W.; Hu, K.; Luo, J.; Ma, Z.; et al. 2024b. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; et al. 2024c. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 24185–24198.

Dagdelen, J.; Dunn, A.; Lee, S.; Walker, N.; Rosen, A. S.; Ceder, G.; Persson, K. A.; and Jain, A. 2024. Structured information extraction from scientific text with large language models. Nature Communications, 15(1): 1418.

DM Wilary, J. C. 2021. A Tool for Automated Extraction of Information from Chemical Reaction Schemes. Journal of chemical information and modeling, 61: 4962–4974.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

EJ Beard, J. C. 2020. ChemSchematicResolver: A toolkit to decode 2D chemical diagrams with labels and R-groups into annotated chemical named entities. Journal of chemical information and modeling, 60: 2059–2072.

Favre, H. A.; and Powell, W. H. 2013. Nomenclature of Organic Chemistry. The Royal Society of Chemistry. ISBN 978-0-85404-182-4.

Gao, P.; Han, J.; Zhang, R.; Lin, Z.; Geng, S.; Zhou, A.; Zhang, W.; Lu, P.; He, C.; Yue, X.; et al. 2023. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010.

He, C.; Li, W.; Jin, Z.; Xu, C.; Wang, B.; and Lin, D. 2024a. OpenDataLab: Empowering General Artificial Intelligence with Open Datasets. arXiv:2407.13773.

He, Z.; Wu, X.; Zhou, P.; Xuan, R.; Liu, G.; Yang, X.; Zhu, Q.; and Huang, H. 2024b. CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning. arXiv preprint arXiv:2401.14011.

Hu, E. J.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W.; et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

IV Filippov, M. N. 2009. Optical structure recognition software to recover chemical information: OSRA, an open source solution.

Jablonka, K. M.; Schwaller, P.; Ortega-Guerrero, A.; and Smit, B. 2024. Leveraging large language models for predictive chemistry. Nature Machine Intelligence, 6(2): 161–169. Joshua, S.; Kyle, M.; Robert, A.; and Carolyn, M. 2019. Molecular Structure Extraction from Documents Using Deep Learning. Journal of chemical information and modeling, 59: 1017–1029.

K Rajan, C. S., A Zielesny. 2020. DECIMER: towards deep learning for chemical image recognition. Journal of Cheminformatics.

Landrum, G. A. 2014. RDKit: Open-source cheminformatics. Release 2014.03.1.

Li, Z.; Wan, H.; Shi, Y.; and Ouyang, P. 2004. Personal Experience with Four Kinds of Chemical Structure Drawing Software: Review on ChemDraw, ChemWindow, ISIS/Draw, and ChemSketch. Journal of Chemical Information and Computer Sciences, 44(5): 1886–1890.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024. Llava-next: Improved reasoning, ocr, and world knowledge.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. In Thirty-seventh Conference on Neural Information Processing Systems.

Loshchilov, I.; and Hutter, F. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Lu, P.; Mishra, S.; Xia, T.; Qiu, L.; Chang, K.-W.; Zhu, S.-C.; Tafjord, O.; Clark, P.; and Kalyan, A. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35: 2507–2521.

Luo, Y.; Zhang, J.; Fan, S.; Yang, K.; Wu, Y.; Qiao, M.; and Nie, Z. 2023. BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine. arXiv:2308.09442. M, O.; A, A.; Y, M.; and J, S. 2020. ChemGrapher: optical graph recognition of chemical compounds by deep learning. Journal of chemical information and modeling, 60(10): 4506– 4517.

M. Bran, A.; Cox, S.; Schilter, O.; Baldassari, C.; White,

- A. D.; and Schwaller, P. 2024. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 1–11. Maziarz, K. 2024. USPTO-50K (raw). McKinzie, B.; Gan, Z.; Fauconnier, J.-P.; Dodge, S.; Zhang,
- B.; Dufter, P.; Shah, D.; Du, X.; Peng, F.; Weers, F.; et al.

- 2024. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611.

Mirza, A.; Alampara, N.; Kunchapu, S.; Emoekabu, B.; Krishnan, A.; Wilhelmi, M.; Okereke, M.; Eberhardt, J.; Elahi, A. M.; Greiner, M.; et al. 2024. Are large language models superhuman chemists? arXiv preprint arXiv:2404.01475.

Myers, B. 2024. The organic division launches a new organic chemistry data website.

- OpenAI. 2023. Gpt-4v(ision) system card.
- OpenAI. 2024a. GPT-4 Technical Report. arXiv:2303.08774.

- OpenAI. 2024b. GPT-4o.
- OpenAI. 2024c. Hello GPT-4o.

Qian, Y.; Guo, J.; Tu, Z.; Coley, C. W.; and Barzilay, R. 2023a. RxnScribe: A Sequence Generation Model for Reaction Diagram Parsing. Journal of chemical information and modeling.

Qian, Y.; Guo, J.; Tu, Z.; Li, Z.; CW, C.; and R, B. 2023b. MolScribe: Robust Molecular Structure Recognition with Image-to-Graph Generation. Journal of chemical information and modeling, 63: 1925–1934.

Rajan, K.; Zielesny, A.; and Steinbeck, C. 2021. DECIMER

- 1.0: deep learning for chemical image recognition using transformers. Journal of Cheminformatics, 13: 1–16.

Rajbhandari, S.; Rasley, J.; Ruwase, O.; and He, Y. 2020. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arXiv:1910.02054.

Rasley, J.; Rajbhandari, S.; Ruwase, O.; and He, Y. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In SIGKDD, 3505–3506.

Reid, M.; Savinov, N.; Teplyashin, D.; Lepikhin, D.; Lillicrap, T.; Alayrac, J.-b.; Soricut, R.; Lazaridou, A.; Firat, O.; Schrittwieser, J.; et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Seyone Chithrananda, B. R., Gabriel Grand. 2020. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv preprint arXiv:2010.09885.

Team, G.; Anil, R.; Borgeaud, S.; Wu, Y.; Alayrac, J.-B.; Yu, J.; Soricut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Wang, W.; Chen, Z.; Chen, X.; Wu, J.; Zhu, X.; Zeng, G.; Luo, P.; Lu, T.; Zhou, J.; Qiao, Y.; et al. 2024a. Visionllm: Large language model is also an open-ended decoder for visioncentric tasks. Advances in Neural Information Processing Systems, 36.

Wang, W.; Lv, Q.; Yu, W.; Hong, W.; Qi, J.; Wang, Y.; Ji,

- J.; Yang, Z.; Zhao, L.; Song, X.; et al. 2023a. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079.

- Wang, W.; Lv, Q.; Yu, W.; Hong, W.; Qi, J.; Wang, Y.; Ji,

J.; Yang, Z.; Zhao, L.; Song, X.; et al. 2023b. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079.

- Wang, X.; Hu, Z.; Lu, P.; Zhu, Y.; Zhang, J.; Subramaniam, S.; Loomba, A. R.; Zhang, S.; Sun, Y.; and Wang, W. 2024b. SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models. In Proceedings of the Forty-First International Conference on Machine Learning.

Weininger, D. 1988. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1): 31–36.

Yang, A.; Yang, B.; Hui, B.; Zheng, B.; Yu, B.; Zhou, C.; Li, C.; Li, C.; Liu, D.; Huang, F.; et al. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Young, A.; Chen, B.; Li, C.; Huang, C.; Zhang, G.; Zhang, G.; Li, H.; Zhu, J.; Chen, J.; Chang, J.; et al. 2024. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652.

Zhang, D.; Liu, W.; Tan, Q.; Chen, J.; Yan, H.; Yan, Y.; Li, J.; Huang, W.; Yue, X.; Zhou, D.; et al. 2024a. Chemllm: A chemical large language model. arXiv preprint arXiv:2402.06852.

Zhang, N.; Heaton, C.; Okonsky, S. T.; Mitra, P.; and Toraman, H. E. 2024b. PEaCE: A Chemistry-Oriented Dataset for Optical Character Recognition on Scientific Documents. In Calzolari, N.; Kan, M.-Y.; Hoste, V.; Lenci, A.; Sakti, S.; and Xue, N., eds., Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 12679– 12689. Torino, Italia: ELRA and ICCL.

Zhao, Z.; Ma, D.; Chen, L.; Sun, L.; Li, Z.; Xu, H.; Zhu, Z.; Zhu, S.; Fan, S.; Shen, G.; Chen, X.; and Yu, K. 2024. ChemDFM: Dialogue Foundation Model for Chemistry. arXiv:2401.14818.

### Technical Appendix

Some details of our framework are listed here.

### Training settings

Here we list out the training settings of our two-stage training, for better reproduction.

Table 4: Training settings of ChemVLM’s stage 1 and stage

- 2.

|Settings<br><br>|Stage 1 Stage 2|
|---|---|
|freeze llm llm lora rank freeze mlp freeze vit vit lora rank learning rate learning rate schedule optimizer optimizer hyper-parameters weight decay input size total batch size drop rate numerical precision GPUs for training<br><br>|True True 0 16 False False True True 32 32 5e-5 5e-5 cosine decay cosine decay AdamW AdamW β1, β2 = 0.9, 0.999 β1, β2 = 0.9, 0.999 1e-5 0.00 448×448 448×448 16×4×4 16×4×4 0.0 0.0 DeepSpeed bf16 DeepSpeed bf16 16×A100 (80G) 16×A100 (80G)<br><br>|

### Data generation prompts

In the main content, we introduce how we generate the chemical data we need. Here we list out the sample prompts, helping understand our work.

Prompts for generating question-answer templates(calling GPT-4):

Can you offer me ten(or more) different expressions of {}? Remember that you should use as much different words and lingual styles as you can.

In {} we add one simple expression of templates we need, like "Can you tell me what is this molecule in the image?".

Samples of generated templates:

1. (Chemical OCR) Human: Can you explain the composition of the molecule in the image in SMILES style?

Assistant: My interpretation is that the molecular formula in this image, according to SMILES format, is {SMILES}.

2. (Molecule caption and property prediction) Human: What can you tell me about this molecule in the image?

Assistant: As far as I’m concerned, the molecule in the image, is {Name}. It has {Properties}.

3. (Exam question answering) Human: {A certain exam question}. Understand the question and the image, and then give your answer.

Assistant: I believe, the answer is {Answer}. The analysis to the question is {Detailed solutions}.

4. (Reaction analysis) Human: This diagram appears to illustrate a synthesis process; what is the product, and which reagents are utilized?

Assistant: This diagram, as I understand it, aims to synthesize {target}, covering the path {path} and using reagents such as {reagent}.

... Here we only show some samples of our generated tem-

plates on certain tasks. Our total data is a mixture of multitask data in different language styles.

### Dataset details

Here we list out the data details we use for training and evaluation in Table 2 and 3. These data consists of our own data, data downloading from huggingface, science databases, etc. Proper citations are added in the References part. Note that our model use the pretrained parameters from ChemLLM-20B, so it has abilities of understanding chemical knowledge itself. Additionally, all the data we mention here are multimodal data if a ’(*)’ is not used. Data with ’(*)’ are text-only data.

- Table 5: Training data details. ’Chem’ means this data is chemical data. General means common sense data. WebChem means a set of chemical datasets we download online. Datasets printed in italic type are open-source datasets.

|Dataset name|Amount<br><br>|Dataset Description|
|---|---|---|
|Chem Exam General General-Chinese Chem OCR Chem Exam(*) Web-Chem(*)<br><br>|122k 625k 712k 180k 658k 1240k|exam questions English general questions Chinese general questions Chemical OCR questions text-only exam questions web multi-task data<br><br>|

- Table 6: Evaluation data details. Datasets printed in italic type are open-source datasets.

|Dataset name<br><br>|Amount<br><br>|Dataset Description|
|---|---|---|
|CMMU ScienceQA MMChemOCR MMCR-bench MMChemBench|about 240 about 200 1000 1000 700<br><br>|Chinese exam questions English general questions SMILES OCR questions Chinese exam questions English general questions|

### Evaluation details and prompts

In this section, we list out the details and prompts we use for evaluating various MLLMs on some task.

#### Evaluating chemical OCR

In this task, MLLMs need to recognize what the molecule is in a image. And then, they should generate a SMILES name for it. Many MLLMs’ answers do not only contain SMILES strings, but analysis for the question. However, the evaluation metrics need SMILES only. Thus, we design prompts

(Zh) 你 是 一 位 熟 悉 化 学SMILES表 示 的 专

家 。 请提取下列语句中的SMILES：“‘{}“‘ 注 意，你的回答格 式 为 ：{’SMILES’: xxx}， 其

中 ，xxx是 你 要 返 回的SMILES串。

(En) You are a chemical scientist familiar with

SMILES. Please extract the SMILES string in the following sentences: “‘{}“‘ Notably, Your

answer should follow the for-mat: {’SMILES’:

xxx}, and xxx is the SMILES string you extract.

(Zh) 你是一位熟悉化学题目答案和评分的专家, 下面有 一个填空题“‘{}“‘和对于填空题的回答“‘{}“‘, 请根据标准 答案“‘{}“‘来逐空给这道题目打分。注意, 这道题目满分 为1分, 请按照正确的空的数目按比例给分, 注意各个空 用’ ’或’；’隔开。请只回答一个0-1(包括0,1)之间的数字 而不输出任何其他内容。

Figure 7: Prompts for extracting SMILES.

to extract the SMILES they generate and use SMILES for score calculation. Note that we design the Chinese(Zh) and English(En) version. Since extracting SMILES is a text-only task, we call the Qwen-max API with this prompt and keep the answers from Qwen-max. The prompts are in figure 1:

(En) You are an expert in answering and grading chem-istry questions, here is a fill-in-the-blank question “‘{}“‘and an answer to the fill-in-the-blank question “‘{}“, pleasegrade the question on a space-by-space basis according tothe

These prompts all means ’SMILES extraction’. Fill in the {} with answer from MLLMs and the prompts are completed. Additionally, we choose Qwen-max for its strong ability for extracting certain texts and convenience during testing various LLMs.

standard answer “‘{}“. Note that this question is worth1

mark, so please award marks proportionally to the number of correct blanks, and note that blanks are separated by’ ’ or ’;’. Please answer only one number between 0 and

#### Evaluating exam questions

When we evaluate MMCR-bench, CMMU and ScienceQA, we may face problems. There are quite a few fill-in-the-blanks problems in the benchmarks, so manually counting the number of right blanks is impossible. Thus, we also choose to design special prompts and call Qwen-max API to do these tasks. The prompts are in figure 2:

1(including 0,1) and do not return anything else.

Figure 8: Prompts for fill-in-the-blank questions scoring.

