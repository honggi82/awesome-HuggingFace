# arXiv:2410.18558v2[cs.CL]6Jan2025

## Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data

Shuhao Gu1*, Jialing Zhang1,2*, Siyuan Zhou1,3*, Kevin Yu1,4*, Zhaohu Xing1,5, Liangdong Wang1, Zhou Cao1, Jintao Jia1,4, Zhuoyi Zhang1,4, Yixuan Wang1,4, Zhenchong Hu1,4, Bo-Wen Zhang1, Jijie Li1, Dong Liang1, Yingli Zhao1, Songjing Wang1, Yulong Ao1, Yiming Ju1, Huanhuan Ma1, Xiaotong Li1,6, Haiwen Diao1,7, Yufeng Cui1, Xinlong Wang1, Yaoqi Liu4, Fangxiang Feng3, Guang Liu1†

1BAAI, 2BJTU, 3BUPT, 4ICT/CAS, 5HKUST(GZ), 6PKU, 7DLUT

### Abstract

Recently, Vision-Language Models (VLMs) have achieved remarkable progress in multimodal tasks, and multimodal instruction data serves as the foundation for enhancing VLM capabilities. Despite the availability of several open-source multimodal datasets, limitations in the scale and quality of open-source instruction data hinder the performance of VLMs trained on these datasets, leading to a significant gap compared to models trained on closed-source data. To address this challenge, we introduce Infinity-MM, a largescale multimodal instruction dataset. We collected the available multimodal instruction datasets and performed unified preprocessing, resulting in a dataset with over 40 million samples that ensures diversity and accuracy. Furthermore, to enable large-scale expansion of instruction data and support the continuous acquisition of high-quality data, we propose a synthetic instruction generation method based on a tagging system and open-source VLMs. By establishing correspondences between different types of images and associated instruction types, this method can provide essential guidance during data synthesis. Leveraging this high-quality data, we have trained a 2-billion-parameter Vision-Language Model, Aquila-VL-2B, which achieves state-of-the-art (SOTA) performance among models of similar scale. The data is available at: https://huggingface.co/datasets/BAAI/Infinity-MM.

### 1. Introduction

Recently, Vision-Language Models (VLMs) [7, 14, 16, 33, 35, 41, 51, 63, 64, 68, 70, 80] have made significant progresses and found important applications across numerous fields, drawing increasing attention. With the development

*Core contributors with equal contributions. †Project Lead, liuguang@baai.ac.cn

[Figure 1]

Figure 1. Average score of different VLMs on benchmarks. The Aquila-VL-2B model, trained with Infinity-MM, not only outperforms models trained on other open-source datasets (OneVision-SI) but also surpasses models trained on closed-source datasets.

in foundational language models, multimodal architectures, multimodal training data, and evaluation benchmarks, the capabilities of multimodal models have greatly improved. Among these elements, multimodal instruction data forms the foundation of a multimodal model’s capabilities. Expanding training data, enhancing data quality, and increasing data diversity play crucial roles in advancing model performance [33, 34, 40, 41, 60].

Many works have focused on exploring more effective ways to generate and utilize training data. For instance, Liu et al. [40] leverages GPT-4 to generate various types of instructions based on textual descriptions of images. Building on this, Li et al. [34] further expands the data scale, leading to performance improvements. Tong et al. [60] and Li et al.

- Table 1. The comparision between Infinity-MM and other multimodal instruction datasets. ”Size” denotes the overall volume of data in the dataset, ”Data Synthesis Method” denotes to the specific approach used to generate synthetic data within the dataset, and ”Data Composition” denotes the types of instruction data included in the dataset. Compared to existing open-source datasets, Infinity-MM has a significant advantage in data scale. Besides, this is the first time to use an open-source VLM for large-scale, high-quality instruction data synthesis.

Data Synthesis Methods Data Composition

Datasets Size

from GPT4(o/v) from Open VLM General Instruction OCR Doc/Chart/Screen Math/Reasoning Text Instruction LVIS-Instruct[25] 223K

Sharegpt4[12] 3.2M MMC-INST[38] 500K

ALLaVA[10] 1.7M MathV360K[55] 360K DocStruct4M[66] 4M DocDownstream [71] 700K

Llava-1.5[40] 660K Cambrain-1[60] 10M

Llava-OneVision[33] 7.2M Infinity-MM 44.8M

[33] enhance model performance by increasing the dataset size and adjusting the data type ratios. Besides, several works explore using GPT-4 series models to generate synthetic instruction data, such as captions [10, 12, 36] , OCR data [9], math questions [55], and conversation data [10, 38, 61, 71]. Despite these advancements, existing open-source instruction datasets remain insufficient to support models in achieving best performance. Models trained solely on open-source data [33, 60] still significantly lag behind SOTA closedsource models [4, 51] or open-source models trained on closed-source data [14, 63, 70]. Compared to closed-source data, existing open-source data is significantly limited in scale, restricting the foundational capability enhancement of models [47]. Besides, current methods for acquiring highquality open-source data often rely on commercial closedsource models, such as GPT-4v and GPT-4o, which can be costly, making large-scale data generation a challenge. Furthermore, the synthesized instruction types are typically confined to predefined categories or are generated freely by the model without sufficient guidance on instruction types, leading to limitations in correctness and diversity. The limitations in both the quantity and quality of open-source data are key factors constraining open model performance.

To enhance the performance of open-source models, this work explores improving data effectiveness by expanding the scale of instruction data and increasing the diversity of instruction types. We have extensively collected existing opensource multimodal instruction data, constructing a dataset of over 40 million samples, and applied rigorous quality filtering and deduplication processes. Besides, we propose a data synthesis method based on a tagging system. We establish tagging systems for images and instructions, modeling the relationship between image and instruction types through this system. This provides essential guidance for data synthesis, ensuring both the accuracy and diversity of the synthesized

data. We also employed open-source VLMs for large-scale, high-quality data synthesis, providing detailed and diverse annotations for images to cover all the information within them as comprehensively as possible, thereby enhancing the model’s performance. Ultimately, we successfully trained a 2-billion-parameter model Aquila-VL-2B based on the proposed dataset Infinity-MM, which not only outperforms models trained on other open-source datasets but also surpasses models trained on closed-source datasets, as shown in Figure 1. The key contributions are summarized as follows:

- • We collected and organized large-scale multimodal instruction dataset, Infinity-MM, consisting of tens of millions of samples. Through quality filtering and deduplication, we ensured the dataset’s high quality and diversity.
- • We propose a data synthesis method based on a tagging system, which provides essential guidance for data synthesis. We employed open-source VLMs to perform large-scale, high-quality instruction data synthesis.
- • Based on Infinity-MM, we have successfully trained a 2billion-parameter VLM model, Aquila-VL-2B, achieving SOTA performance among models of the same scale.

### 2. Related Work

Multi-modal Instruction Data Currently, the construction of multimodal instruction datasets primarily employs three methods: manual annotation, model synthesis, and opensource data collection. The first method is manual annotation. Das et al. [17] involves two annotators per image engaging in up to 10 rounds of question-answer interactions, generating multi-turn dialogues associated with each image. Agrawal et al. [2] employs one set of workers to create up to three QA pairs based on given images and text, followed by verification by a second set of workers to ensure accuracy. Havard et al. [26] combines manual annotation and OCR techniques to build a dataset with detailed text detection and recognition

[Figure 2]

New Image

[Figure 3]

RAM++

…

…

Woman | Face | Laugh …

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Instruction 1: Describe the woman’s emotion.

Instruction 1: Describe the sexual of the human.

Instruction 1: Describe the woman’s emotion.

Question Filter

Image Labeling

Question Generation

Seed Data

…

[Figure 8]

RAM++

Man | Face | Smile ...

Retrieval

Answers Generation

[Figure 9]

[Figure 10]

- Instruction 1: Describe this man’s emotion

- Instruction 2: The sexual of the human

[Figure 11]

…

Instruction

…

[Figure 12]

Image-Instruction Correspondence

[Figure 13]

[Figure 14]

Instructions Classification

The woman is surprised.

The woman is surprised.

Answers Filter

- Instruction 1: Coarse Perception->Image Emotion-> Read emotions from faces
- Instruction 2: …

…

Image-Instruction pair

Retrieved Instruction 1: Coarse Perception->Image Emotion->Read emotions from faces

- Figure 2. Illustration data synthesis method. We propose a tagging system to model the correspondence between image types and instructions, providing essential guidance for data synthesis to ensure both accuracy and diversity in instruction generation. Besides, we introduce the use of open-source VLM for large-scale, high-quality instruction data synthesis.

### 3. The proposed Infinity-MM Dataset

annotations. Mathew et al. [49] adopts a three-stage manual annotation process to create a visual question-answering dataset tailored for document images, featuring diverse question types and a wide range of document images. The second kind of methods utilize VLM model to synthesize specific types of data, such as captions [10, 12] , OCR data [9], math questions [55], chart description [31, 71], and conversation data [10, 40, 61]. The third method is open-source data collection. Liu et al. [38] collected 210K chart-title pairs by downloading academic articles from 2010 to 2020. Ye et al. [71] aggregates multiple publicly available document image datasets into a unified, diverse document dataset. Tong et al. [60] and Li et al. [33] both compile a variety of available open-source instruction datasets to create a collection encompassing multiple data types. Our proposed InfinityMM dataset combines model synthesis with open-source data collection. As shown in Table 1, compared to existing datasets, Infinity-MM includes various types of data and has a substantial advantage in data scale.

In this section, we provide a detailed explanation of the process to construct the Infinity-MM dataset. First, we extensively collect available open-source multimodal datasets of various tasks and types and categorize them based on task and quality. Then, we propose a data synthesis method based on a tagging system. We establish tagging systems for both images and instructions within the multimodal instruction dataset. By modeling the relationship between image and instruction tags, we provide essential guidance during instruction synthesis, ensuring that the generated instructions align with the content of the images while also increasing diversity of the synthesized instructions. Finally, we standardize the formats of data from different sources and apply deduplication and quality filtering to the dataset.

#### 3.1. Collection of Datasets

We systematically gather available open-source multimodal datasets and categorize them. These datasets are classified into four categories, as outlined in Table 2. The specific sources of the data are given in Appendix B.

Vision-Language Model VLMs can be categorized into three types based on their capabilities. The first type focuses on understanding multimodal information, such as videos and images [3, 18, 35, 41, 53]. These models typically take multimodal data as input and produce natural language output, characterized by their ability to integrate and process information from different modalities in a unified manner. The second type emphasizes visual generation, primarily aimed at producing high-resolution images and videos [19, 52, 54, 56]. The third type combines both visual understanding and generation capabilities [58, 59, 65, 69, 79]. In this work, we focus on enhancing the model’s ability to comprehend multimodal information.

- • Image-Caption Data We collect the Image-Caption dataset generated by Emu2 [58], which performs well in image caption task. The image-caption data has detailed descriptions of image content, making this it well-suited for vision-language alignment training in VLMs.
- • Comprehensive Visual Instruction Data This part of the data is collected from open-source datasets and includes various types of data. The data distribution has not been specifically adjusted, and its effectiveness has not been rigorously validated. Therefore, this part of data is suitable for enhancing the foundational capabilities of VLMs.

- Table 2. The data quantity and types contained in Infinity-MM. We segmented the dataset based on data type and quality, allowing the model to perform targeted training at different stages.

Data Category Size Data Composition

Image Caption Data 10M General Instruction Data 10M

Visual Instruction Data

|Comprehensive Data|25.8M<br><br>|General Instruction Data 7.1M OCR Data 2.6M Doc/Chart/Screen Data 5.8M Math/Reasoning Data 1.3M Text Instruction Data 9M|
|---|---|---|
|Selective Data|6M<br><br>|General Instruction Data 1.3M OCR Data 0.3M Doc/Chart/Screen Data 1.9M Math/Reasoning Data 0.7M Text Instruction Data 1.8M|
|GPT4 & Synthetic Data<br><br>|3M|General Instruction Data 1M OCR Data 0.5M Doc/Chart/Screen Data 0.1M Math/Reasoning Data 0.3M Text Instruction Data 0.3M Our Synthetic Data 0.8M|

- • Selective Visual Instruction Data This part of the data is also sourced from open-source datasets, with an adjusted data distribution that increases the proportion of tasks involving mathematical reasoning, chart analysis, and complex instructions—areas where VLMs generally perform poorly. The quality and effectiveness of these data have been validated, making them suitable for enhancing the instruction-following ability of VLMs.
- • GPT4 & Synthetic Visual Instruction Data We collect instruction data synthesized using GPT-4 series models. This data generally exhibits high quality, but the task types are relatively homogeneous, and the data volume is limited. We randomly sample a subset of images from this data and synthesized instructions based on these images with the method introduced in Section 3.2.

- 3.2. Data Synthesis

##### 3.2.1. Image and Instruction Tagging System

We randomly select a portion of the open-source data as seed data. Then we utilize the RAM++ model [29] to automatically annotate images by extracting key information such as objects, actions, and scenes. These tags form the semantic foundation of the images, providing a critical basis for subsequent instruction generation. The RAM++ model demonstrates excellent performance when processing large-scale image datasets, accurately capturing essential details in multimodal scenes. This lays a solid foundation for generating precise and contextually relevant multimodal instructions.

To systematize the instruction generation process, we design a three-level instruction tagging system that covers different types of instructions. Following Liu et al. [42], the first-level tags of the instruction tagging system are divided into six categories, which are:

- • Coarse Perception
- • Fine-grained Perception (single-instance)
- • Fine-grained Perception (cross-instance)
- • Relation Reasoning
- • Attribute Reasoning
- • Logic Reasoning We extend the tagging system based on the first-level tags with the GPT-4 model. The middle level further refines task characteristics, while the bottom level provides a detailed classification based on specific task requirements. After constructing the initial tagging system, we use the GPT-4o model to annotate the instructions in the seed data. During the annotation process, we refined the tagging system by adding

In this section, we introduce our multimodal instruction data synthesis method. We aim that the generated instructions are closely aligned with the content of the images while maintaining diversity in instruction types and ensuring the accuracy of instruction responses. The overall process of the method is shown in Figure 2. The images of the synthetic data are extracted from the instruction dataset synthesized using the GPT-4 series models, which are of high quality. However, the instruction type and data quantity of the original data are limited. Therefore, we aim to leverage open-source models to synthesize more high-quality data, combining it with the original data to further enhance model performance.

[Figure 15]

- Figure 3. This is an example of our synthesized data. Compared to the original data on the left, our synthesized data on the right offers greater diversity in instructions (Q&A 1-3) and covers a broader range of image content (Q&A 4,5).

or removing tags. The resulting tagging system covers 199 sub-tasks, ensuring its comprehensiveness and rationality. The complete tagging system can be found in Appendix C.

##### 3.2.2. Image-Instruction Mapping

We annotate both the images and instructions in the seed data with the method in the previous section. Following this, we establish the mapping relationships between image tags and instruction tags. Specifically, we begin by counting the frequency of image tags associated with each instruction tag in the seed data. Then, treating each instruction tag as a distinct unit, we calculate the TF-IDF values for the image tags within these units. A higher TF-IDF value for an image tag within a given instruction type indicates that images with this tag are more suitable for generating that specific type of instruction. Leveraging these TF-IDF-based mappings allows us to automatically determine the most appropriate instruction type to generate for new images. This approach significantly improves the alignment between the generated instructions and the actual image content.

##### 3.2.3. Question Generation

After establishing the mapping relationships between images and instruction types, we proceed with instruction synthesis. Balancing data synthesis quality and efficiency, we employ the MiniCPM-V2.6 model [70] in this work. First, we need to generate appropriate questions for each image. For each candidate image, we identify the suitable instruction types.

Figure 4. The distribution of first-level tags for synthetic instruction data. The tagging system enables more effective targeted synthesis and analysis of instruction data.

Then, we input both the image and the target instruction type into the VLM model, prompting it to generate questions based on these information. Besides, we select two data examples of the same instruction type from our annotated seed data and input their images and questions into the VLM for reference, enabling few-shot generation. For the questions generated by the VLM, we further re-input both the image and question into the VLM to evaluate the relevance of each question to the image, filtering out lower-quality questions.

##### 3.2.4. Answer Generation

After generating the questions, we proceed to generate the corresponding instruction answers. We ensure not only the accuracy of the answers but also the diversity of their formats. To achieve this, we employ different prompts during

Table 3. Configuration for training Aquila-VL-2B across various stages.

Stage-2

Stage-1

Stage-3 Stage-4 a b c

Resolution 384 384×{(1×1),...,(2×2)} 384×{(1×1),...,(3×3)} 384×{(1×1),...,(4×4)} 384×{(1×1),...,(6×6)} 384×{(1×1),...,(6×6)} #tokens 729 Max 5×729 Max 6×729 Max 7×729 Max 10×729 Max 10×729

Vision

Data

Samples 10M 8.6M 8.6M 8.6M 6M 3M

Trainable Projector Full Model Full Model Full Model Full Model Full Model #para counts 4.13M 2B 2B 2B 2B 2B

Model

Batch Size 512 512 512 512 512 512 LR 1.00E-03 1.00E-05 1.00E-05 1.00E-05 1.00E-05 1.00E-05 Epoch 1 1 1 1 1 1

Training

the answer generation process. Specifically, we employ three different types of prompts: one instructed the model to provide short answers using single words or phrases; another prompted the model to first generate a simple explanation before giving the answer; and the third prompted the model to provide a detailed explanation followed by the answer.

We re-input the image, question, and answer into the VLM model, allowing it to score the instruction quality on a scale of 1 to 10, where a higher score indicates better quality. Instructions with scores below 8 were filtered out. Additionally, we input the synthesized instruction data into the Qwen2-VL-7B model [63] model to compute the loss, filtering out data with excessively high loss values. Finally, we select approximately 3 million filtered data points for training, prioritizing data related to reasoning and document analysis, as these tasks were relatively scarce in the previous training phases. We combine multiple QA pairs corresponding to the same image into multi-turn instruction data, resulting in approximately 800K training instructions. Figure 3 presents an example of our synthesized data. The distribution of instruction types in the final synthetic data is shown in Figure 4.

- 3.3. Data Processing

After collecting all the data, we proceed with data processing. First, we standardize the format of data from various sources. Then, we remove duplicate Image-Text pairs and filter out images with high similarity based on their pHash values [75]. Besides, we use Qwen2-VL-2B [63] to calculate the loss for each sample and exclude the top 5% data with the highest loss, as high loss in well-trained multimodal models often indicates noisy data.

- 4. Training of Aquila-VL-2B

- 4.1. Model Architecture

To validate the effectiveness of Infinity-MM, we used it to train Aquila-VL-2B, a vision-language model with about

2 billion parameters. Aquila-VL-2B builds upon LLaVAOneVision architecture[33], comprising a language tower, a vision tower, and a projector.

- • Language Tower We choose Qwen-2.5-Instruct [6] as the language tower for its outstanding performance among open-source models.
- • Vision Tower We utilize SigLIP [76], with approximately 400 million parameters, as the vision tower to extract visual features from input images and videos.
- • Projector We utilize a two-layer MLP [41] with a GELU [27] activation to project visual features into the word embedding space.

#### 4.2. Training Details

We use the official codes of LLaVA-OneVision1 and the training setup is given in Table 3. Followling Li et al. [34], the training is divided into different stages, progressively increasing the difficulty, image resolution, and data quality:

- • Stage 1: We train the projector using 10M image-caption data to align the visual feature space with the word embedding space. Both the vision tower and language tower are frozen during this phase.
- • Stage 2: We utilize general visual instruction data for further training to equip the model with fundamental capabilities for solving multimodal tasks. The data is divided into three subsets, and during each stage of training, the maximum visual resolution was progressively increased to enhance the model’s comprehension of visual information.
- • Stage 3: We employ selective visual instruction data for training and further increase the maximum resolution to improve performance.
- • Stage 4: We fine-tune the model using training data from GPT-4 and synthetic data. Experiments demonstrate that this part of data can further enhance model performance.

1https://github.com/LLaVA- VL/LLaVA- NeXT/tree/ main/scripts/train

- Table 4. Comprehensive benchmark Comparisons of Aquila-VL-2B Model and other models. Results for models marked with * are sourced directly from the VLMEvalKit leaderboard, while other results are measured by us using default configurations. In all experimental results, higher values indicate better performance. The Aquila-VL-2B model, trained with Infinity-MM, achieved SOTA performance on average among models of the same scale.

Model MiniMonkey*[28] PaliGemma*[8] DeepSeekVL*[44] H2OVL*[23] Phi3-V*[1] Vintern*[20] MiniCPM-V2[70] InternVL2[14] XinYuanVL[15] Qwen2VL[63] Aquila-VL Pars (B) 2.2 2.9 2.0 2.1 4.2 3.7 2.8 2 2 2 2

Open Data

MMB-ENtest[42] 72.4 71.0 66.4 72.1 73.6 70.6 69.4 73.4 78.9 74.9 78.8 MMB-CNtest[42] 70.3 63.6 62.9 62.9 62.4 69.4 65.9 70.9 76.1 73.9 76.4

MMBV1.1[42] 68.9 65.6 63.8 64.8 65.2 66.6 65.2 69.7 75.4 72.7 75.2 MMT-Benchall[72] - - - - - - 54.5 53.3 57.2 54.8 58.2 RealWorldQA[67] 57.1 55.2 49.7 62.9 58.8 58.2 55.4 57.3 63.9 62.6 63.9

HalluB[24] 38.7 32.2 27.6 35.9 39.0 43.2 36.8 38.1 36.0 41.5 43.0 S-Bench2Plus[32] - 49.8 43.7 59.5 64.2 64.1 51.8 60.0 63.0 62.4 63.0 LLaVABench[39] 61.2 36.9 51.1 65.7 63.9 62.2 66.1 64.8 42.4 52.5 68.4

GeneralVQA

MMStar[13] 50.4 48.3 39.9 48.9 47.7 47.5 41.6 50.2 51.9 47.8 54.9 POPE[37] - 87.5 85.9 86.6 83.7 87.4 86.6 85.3 89.4 88.0 83.6 MMVet[73] 38.0 33.1 29.2 41.1 44.1 37.8 44.0 41.1 42.7 50.7 44.3

MME[22] 1881.6 1686.1 1531.6 1767.5 1508.0 1782.6 1788.6 1863.0 1854.9 1890.0 1799.3

MMMUval[74] 35.0 34.9 33.8 36.3 46.1 46.7 39.6 34.9 43.6 41.7 47.4 S-QAtest[45] - 94.3 68.4 92.1 90.0 75.0 80.4 94.1 86.6 78.1 95.2 AI2Dtest[30] 74.8 68.3 51.5 70.9 78.4 69.1 64.8 74.4 74.2 74.6 75.0

Knowledge&Math

MathVista[46] 45.2 28.7 29.8 56.5 44.6 43.4 39.0 45.0 47.1 47.9 59.0 MathVerse[77] - - - - - - 19.8 24.7 22.2 21.0 26.2

MathVision[62] - - - - - - 15.4 12.6 16.3 17.5 18.4 DocVQAtest[49] - - - - - - 71.0 86.9 87.6 89.9 85.0 InfoVQAtest[49] - - - - - - 40.0 59.5 59.1 65.4 58.3 ChartQAtest[48] - 33.7 47.4 59.4 81.8 68.3 59.6 71.4 57.1 73.5 76.5 TextVQAval[57] - 68.1 57.8 74.8 72.4 67.2 74.3 73.5 77.6 79.9 76.4 OCRVQAtest[50] - 57.8 58.1 70.3 61.9 56.8 54.4 40.2 67.6 68.7 64.0

Text-rich

VCRen easy[78] - - - - - - 27.6 51.6 67.7 68.3 70.0

OCRBench [43] 804 614 413 778 637 618 613 784 782 810 772 Average - - - - - - 53.9 59.1 61.1 62.3 64.1

### 5. Evaluation

In this section, we first compared the performance of AquilaVL-2B with similarly sized models across different benchmarks, showing that Aquila-VL-2B achieves SOTA performance. Next, we assess the impact of training with InfinityMM versus other open-source datasets. Finally, we analyze the impact of our synthetic data on model performance and examine how performance varies with data scale.

#### 5.1. Main Results

We assessed the visual capabilities of Aquila-VL-2B using a range of visual benchmarks provided by the VLMEvalKit [21]. Experimental results are shown in Table 4. Aquila-VL-2B demonstrates highly competitive performance at the same scale, achieving new state-of-the-art results. Specifically, we evaluated the capabilities of AquilaVL-2B across three task categories.

General Visual Question Answering We conducted extensive evaluations across a diverse array of general visual question answering benchmarks: MMBench-ENtest (MMBENtest), MMBench-CNtest (MMB-CNtest), MMBench-1.1 (MMBV1.1) [42], MMT-Benchall [72], RealWorldQA [67], HallusionBench (HalluB) [24], SEEDBench2plus (SBench2plus) [32]), LLaVABench [39], MMStar [13],POPE [37] ,MMVet [73], and MME [22] datasets. Aquila-VL2B demonstrates strong performance across these benchmarks, achieving or surpassing state-of-the-art results in most cases at the same scale. Specifically, Aquila-VL-2B achieves the best performance on MMB-CNtest, MMBV1.1,

MMT-Benchall, RealWorldQA, LLaVABench, and MMSar. It also demonstrates top-tier results on MMB-ENtest, HalluB, and S-Bench2Plus. Additionally, it achieves competitive results on POPE, MMVet, and MME, with room for further improvement. These results strongly indicate Aquila-VL-2B’s remarkable capability in handling general visual questionanswering tasks and validate the effectiveness of using the Infinity-MM dataset to enhance visual understanding.

Knowledge and Mathematical Reasoning We conducted experiments on MMMUval [74], ScienceQAtest (S-QAtest) [45], AI2Dtest [30], MathVista [46], MathVerse [77], and MathVision [62] to evaluate the model’s capabilities in knowledge and mathematical reasoning. Aquila-VL-2B achieves state-of-the-art performance on MMMUval, SQAval, MathVista, MathVerse, and MathVision, while securing top-tier results on AI2Dtest. These outcomes underscore Aquila-VL-2B’s exceptional capabilities in visual knowledge comprehension and mathematical reasoning. The success can be attributed to Infinity-MM, which has collected over 2 million high-quality mathematical reasoning data points, further demonstrating the dataset’s quality and effectiveness. Text Reading We assessed Aquila-VL-2B’s capabilities in text reading and diagram comprehension using DocVQAtest [49], InfoVQAtest [49], ChartQAtest [48], TextVQAval [57], OCRVQAtestcore [50] and VCRen easy [78] and OCRBench [43]. The experimental results indicate that Aquila-VL-2B’s text comprehension abilities are generally among the top tier, with potential for further improvement. We will continually supplement high-quality, text-rich data to further refine the Infinity-MM dataset.

- Table 5. Performance comparison of models trained with Infinity-MM versus other open-source datasets. Compared to existing open-source datasets, Infinity-MM demonstrates a clear performance advantage.

Model Params (B) Training Data Average MMBenchV1.1test MMStar MMMUval MathVistatestmini HallusionBench AI2Dtest OCRBench MMVet

LLaVA-OneVision-7B [33] 8 LLaVA-OneVision-SI 60.1 80.9 61.9 47.9 62.3 31.6 82.4 622 51.9 Cambrian-34B [60] 34 Cambrian-7M 58.3 77.8 54.2 50.4 50.3 41.6 79.5 591 53.2

Aquila-VL-2B 2

LLaVA-OneVision-SI 52.1 70.3 49.9 42.2 49.4 33.6 73.1 617 36.7 Infinity-MM 59.5 75.2 54.9 47.4 59.0 43.0 75.0 772 44.3

- Table 6. Comparison of the model performance with and without synthetic data. The average performance of the model significantly declined without synthetic data, validating the effectiveness of synthetic data.

Models Average MMBenchV1.1test MMStar MMMUval MathVistatestmini HallusionBench AI2Dtest OCRBench MMVet Aquila-VL-2B 59.5 75.2 54.9 47.4 59.0 43.0 75.0 772 44.3

w/o Synthetic Data 57.25 75.09 54.53 45.56 57 35.89 75.03 766 38.3

#### 5.2. Infinity-MM vs. Other Open Datasets

To further validate the performance advantages of InfinityMM and its importance for research in the open-source community, we compared Infinity-MM with several widely used, high-performance open-source datasets. We selected eight of the most representative datasets for testing, and the experimental results are shown in Table 5. Specifically, we first compare Aquila-VL-2B, trained on Infinity-MM, with LLaVA-OneVision-7B, trained on LlaVA-OneVision data, and Cambrian-34B, trained on Cambrian-10M. The results indicate that, despite its smaller scale relative to LLaVAOneVision-7B and Cambrian-34B, Aquila-VL-2B outperforms Cambrian-34B and performs comparably to LLaVAOneVision-7B. This underscores the critical role of training data in determining model performance. Furthermore, to better assess the relative impact of Infinity-MM versus LLaVAOneVision data, we trained Aquila-VL-2B using each dataset independently. The results reveal that training on InfinityMM yields superior performance, providing strong evidence of the high quality of the Infinity-MM dataset.

#### 5.3. Effects of Synthetic Data

To assess the impact of synthetic data on model performance, we conducted an ablation study. In this experiment, we removed all synthetic data and trained the model using only the original GPT-generated data. The results, as shown in Table 6, reveal a significant decline in overall model performance after removing the synthetic data. This demonstrates that the synthetic data played a crucial role in enhancing the model’s performance, further validating the effectiveness of our approach in data augmentation and diversity.

#### 5.4. Data Scaling

To further analyze the impact of data size scaling on model performance, we conducted a detailed study on how model performance varies with the amount of training data. The results, shown in Figure 5, indicate a consistent improvement in performance as the training data increases. This trend

[Figure 16]

Figure 5. As the amount of instruction data used for training increased, the model’s average performance continued to improve, validating the effectiveness of scaling up instruction data.

clearly demonstrates that expanding the scale of instruction data has a significant positive effect on model performance. This observation indicates that incorporating more diverse instructional data enhances the Aquila-VL’s capability to tackle complex tasks. Consequently, expanding the volume of instructional data proves to be an effective strategy for boosting overall model performance.

### 6. Conclusion

In this work, we proposed the Infinity-MM multimodal instruction dataset, comprising tens of millions of samples to substantially increase data volume and enhance model effectiveness. Furthermore, we introduced a novel method for synthesizing instruction data based on open-source models, which enabled the generation of high-quality instruction data and further expanded the dataset. Finally, we trained the Aquila-VL-2B model on Infinity-MM, achieving stateof-the-art performance for models of a similar scale.

Limitations Despite the impressive performance of InfinitiyMM, there is still room for enhancement in our work: (1)

Due to limited computational resources, we have currently conducted experiments only on the 2B-scale model; (2) Due to time constraints, our dataset has not yet incorporated additional multi-image, video, and multilingual data. The future work will involve training on larger-scale models to further validate the quality of Infinity-MM. Additionally, we will continuously incorporate diverse high-quality data into Infinity-MM to support advancements in related research.

### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, S´ebastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai, Matthew Dixon, Ronen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg, Allie Del Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Wenxiang Hu, Jamie Huynh, Dan Iter, Sam Ade Jacobs, Mojan Javaheripi, Xin Jin, Nikos Karampatziakis, Piero Kauffmann, Mahoud Khademi, Dongwoo Kim, Young Jin Kim, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui Lin, Zeqi Lin, Ce Liu, Liyuan Liu, Mengchen Liu, Weishung Liu, Xiaodong Liu, Chong Luo, Piyush Madan, Ali Mahmoudzadeh, David Majercak, Matt Mazzola, Caio C´esar Teodoro Mendes, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Liliang Ren, Gustavo de Rosa, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia Song, Masahiro Tanaka, Andrea Tupini, Praneetha Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel Ward, Wen Wen, Philipp Witte, Haiping Wu, Xiaoxia Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Jilong Xue, Sonali Yadav, Fan Yang, Jianwei Yang, Yifan Yang, Ziyi Yang, Donghan Yu, Lu Yuan, Chenruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. Phi-3 technical report: A highly capable language model locally on your phone, 2024. 7
- [2] Mayank Agrawal, Anand Singh Jalal, and Himanshu Sharma. Enhancing scene-text visual question answering with relational reasoning, attention and dynamic vocabulary integration. Comput. Intell., 40(1), 2024. 2, 14
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Kar´en Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in Neural

- Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. 3
- [4] Anthropic. Claude 3.5 sonnet, 2024. 2
- [5] BAAI. Infinity instruct. arXiv preprint arXiv:2406, 2024. 14
- [6] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 6
- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. CoRR, abs/2308.12966, 2023. 1
- [8] Lucas Beyer*, Andreas Steiner*, Andr´e Susano Pinto*, Alexander Kolesnikov*, Xiao Wang*, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Boˇsnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai*. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726, 2024. 7
- [9] Jimmy Carter. Textocr-gpt4v. https://huggingface. co / datasets / jimmycarter / textocr - gpt4v,

2024. 2, 3

- [10] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4vsynthesized data for a lite vision-language model, 2024. 2, 3, 14
- [11] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, pages 513–523. Association for Computational Linguistics, 2021. 14
- [12] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 2, 3, 14
- [13] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? CoRR, abs/2403.20330, 2024. 7

- [14] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 1, 2, 7
- [15] Cylingo. Xinyuan-vl-2b, 2024. 7
- [16] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 16, 2023, 2023. 1
- [17] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jos´e M. F. Moura, Devi Parikh, and Dhruv Batra. Visual dialog. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 1080–1089. IEEE Computer Society, 2017. 2, 14
- [18] Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. CoRR, abs/2406.11832, 2024. 3
- [19] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-to-image generation via transformers. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 19822–19835, 2021. 3
- [20] Khang T. Doan, Bao G. Huynh, Dung T. Hoang, Thuc D. Pham, Nhat H. Pham, Quan T. M. Nguyen, Bang Q. Vo, and Suong N. Hoang. Vintern-3b-beta, 2024. 7
- [21] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An opensource toolkit for evaluating large multi-modality models,

2024. 7

- [22] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. CoRR, abs/2306.13394, 2023. 7
- [23] Shaikat Galib, Shanshan Wang, Guanshuo Xu, Pascal Pfeiffer, Ryan Chesler, Mark Landry, and Sri Satish Ambati. H2ovlmississippi vision language models technical report, 2024. 7
- [24] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In CVPR, pages 14375–14385. IEEE, 2024. 7
- [25] Agrim Gupta, Piotr Doll´ar, and Ross B. Girshick. LVIS: A dataset for large vocabulary instance segmentation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 5356–5364. Computer Vision Foundation / IEEE, 2019. 2, 14

- [26] William Havard, Laurent Besacier, and Olivier Rosec. SPEECH-COCO: 600k visually grounded spoken captions aligned to MSCOCO data set. CoRR, abs/1707.08435, 2017. 2, 14
- [27] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023. 6
- [28] Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, and Xiang Bai. Mini-monkey: Multi-scale adaptive cropping for multimodal large language models. arXiv preprint arXiv:2408.02034, 2024. 7
- [29] Xinyu Huang, Yi-Jie Huang, Youcai Zhang, Weiwei Tian, Rui Feng, Yuejie Zhang, Yanchun Xie, Yaqian Li, and Lei Zhang. Open-set image tagging with multi-grained text supervision. arXiv e-prints, pages arXiv–2310, 2023. 4
- [30] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min Joon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV (4), pages 235–251. Springer,

2016. 7

- [31] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions. CoRR, abs/2408.12637, 2024. 3, 14
- [32] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. CoRR, abs/2404.16790, 2024. 7
- [33] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. CoRR, abs/2408.03326, 2024. 1, 2, 3, 6, 8, 14
- [34] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 1, 6, 14
- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, ICML 2023, 2329 July 2023, Honolulu, Hawaii, USA, pages 19730–19742. PMLR, 2023. 1, 3
- [36] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Ling-Yu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. CoRR, abs/2407.08303, 2024. 2
- [37] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, pages 292–305. Association for Computational Linguistics, 2023. 7
- [38] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. MMC: advancing multimodal chart understanding with largescale instruction tuning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 1287–1310. Association for Computational Linguistics, 2024. 2, 3, 14

- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 7
- [40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 1, 2, 3
- [41] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 26286–26296. IEEE, 2024. 1, 3, 6
- [42] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multimodal model an all-around player? CoRR, abs/2307.06281,

2023. 4, 7

- [43] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, and Xiang Bai. On the hidden mystery of OCR in large multimodal models. CoRR, abs/2305.07895,

2023. 7

- [44] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world visionlanguage understanding, 2024. 7
- [45] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS,

2022. 7

- [46] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 7
- [47] Run Luo, Haonan Zhang, Longze Chen, Ting-En Lin, Xiong Liu, Yuchuan Wu, Min Yang, Minzheng Wang, Pengpeng Zeng, Lianli Gao, Heng Tao Shen, Yunshui Li, Xiaobo Xia, Fei Huang, Jingkuan Song, and Yongbin Li. Mmevol: Empowering multimodal large language models with evol-instruct. CoRR, abs/2409.05840, 2024. 2
- [48] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. CoRR, abs/2203.10244, 2022. 7
- [49] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for VQA on document images. In IEEE Winter Conference on Applications of Computer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021, pages 2199–2208. IEEE, 2021. 3, 7, 14
- [50] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: visual question answering by reading text in images. In ICDAR, pages 947–952. IEEE,

2019. 7

- [51] OpenAI. Gpt-4v system card, 2024. 1, 2
- [52] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4172–4182. IEEE, 2023. 3
- [53] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748–8763. PMLR, 2021. 3
- [54] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8821–8831. PMLR, 2021. 3
- [55] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Mathllava: Bootstrapping mathematical reasoning for multimodal large language models. CoRR, abs/2406.17294, 2024. 2, 3, 14
- [56] Zhan Shi, Xu Zhou, Xipeng Qiu, and Xiaodan Zhu. Improving image captioning with better use of caption. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7454–7464, Online, 2020. Association for Computational Linguistics. 3
- [57] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In CVPR, pages 8317–

8326. Computer Vision Foundation / IEEE, 2019. 7

- [58] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are incontext learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 14398–14409. IEEE, 2024. 3
- [59] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 3, 14
- [60] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. CoRR, abs/2406.16860, 2024. 1, 2, 3, 8, 14
- [61] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting GPT-4V for better visual instruction tuning. CoRR, abs/2311.07574, 2023. 2, 3
- [62] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical

reasoning with math-vision dataset. CoRR, abs/2402.14804,

2024. 7

- [63] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. CoRR, abs/2409.12191, 2024. 1, 2, 6, 7
- [64] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. CoRR, abs/2311.03079, 2023. 1
- [65] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need, 2024. 3
- [66] Zilong Wang, Mingjie Zhan, Xuebo Liu, and Ding Liang. Docstruct: A multimodal method to extract hierarchy structure in document for general form understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, pages 898–908. Association for Computational Linguistics, 2020. 2, 14
- [67] X.AI. Grok-1.5 Vision Preview. https://x.ai/blog/ grok-1.5v, 2024. 7
- [68] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 4818–4829. IEEE, 2024. 1
- [69] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. CoRR, abs/2408.12528, 2024. 3
- [70] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 1, 2, 5, 7
- [71] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplugowi2: Revolutionizing multi-modal large language model with modality collaboration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 13040–13051. IEEE,

2024. 2, 3, 14

- [72] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao,

- Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask AGI. In ICML. OpenReview.net, 2024. 7
- [73] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. CoRR, abs/2308.02490, 2023. 7
- [74] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. CoRR, abs/2311.16502,

2023. 7

- [75] Christoph Zauner. Implementation and benchmarking of perceptual image hash functions. 2010. 6
- [76] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 6
- [77] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. Mathverse: Does your multimodal LLM truly see the diagrams in visual math problems? CoRR, abs/2403.14624, 2024. 7
- [78] Tianyu Zhang, Suyuchen Wang, Lu Li, Ge Zhang, Perouz Taslakian, Sai Rajeswar, Jie Fu, Bang Liu, and Yoshua Bengio. VCR: visual caption restoration. CoRR, abs/2406.06462,

2024. 7

- [79] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. CoRR, abs/2408.11039, 2024. 3
- [80] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 1

Benchmark MiniCPM-V-2 InternVL2-2B Qwen2-VL-2B-Instruct Aquila-VL-2B Aquila-VL-2B-video Video-MME(w/o subs) 38.6 45.9 55.6 48.4 51.5

Table 7. Performance of Aquila-VL-2B and other models on video benchmarks.

[Figure 17]

[Figure 18]

[Figure 19]

(a) Comprehensive Data (b) Selective Data (c) GPT4 & Synthetic Data

[Figure 20]

Figure 6. Distribution of Visual Instruction Data

### A. Video Understanding

To enhance Aquila-VL-2B’s ability to process multi-image and video data, we extracted a total of 937K multi-image and video samples from the LLaVA-OneVision dataset, and combined them with 1M single-image samples drawn from Stage 4 for further training. The results, as shown in Table 7, demonstrate that even prior to incorporating the multi-image and video data, our model already exhibited a solid ability to handle video imagery with satisfactory performance. After introducing the additional multi-image and video data for further training, the model’s capacity to process such data was significantly improved.

### B. Data composition

For Image Caption Data, it consists entirely of General Instruction Data. But for Visual Instruction Data, due to its complex components, we further divide it into 3 parts, and visualize the data types of the three parts respectively, as shown in Figure 6. The composition of Comprehensive Data and Selective Data is General Instruction Data, OCR Data, Doc/Chart/Screen Data, Math/Reasoning Data and Text Instruction Data, for GPT4 & Synthetic Data, we add our Synthetic Data. Besides, we listed the sources, sizes, and types of all our data in Table 8.

### C. Instruction Tagging System

#### C.1. Coarse Perception

- • Image Scene

- – Identify structures

- – Identify geographic location
- – Identify weather condition
- – Identify presence of people
- – Identify event type
- – Identify activity
- – Identify location
- – Identify time
- – Identify buildings
- – Identify people
- – Other scene descriptions
- – Identify background
- – Identify diagram
- – Identify action
- – Identify season
- – Identify vegetation type
- – Other
- – Identify objects in scene
- – Identify overall theme
- – Identify natural elements
- – Identify objects
- – Identify time of day
- – Identify activities
- – Identify number of people
- – Identify environment type
- – Count people
- – Identify main subject
- – Identify clothing
- – Identify geometric properties
- – Identify vegetation presence
- – Identify animals
- – Identify furniture
- – Describe background

|Data Source<br><br>|Size|Type|
|---|---|---|
|Emu2 [59]<br><br>|10M|Caption|
|LVIS-Instruct[25]|223K<br><br>|General|
|LLaVA-CC3M-Pretrain-595K[34]<br><br>|595K|General|
|Visdial[17]|116K<br><br>|General|
|Sharegpt4[12]<br><br>|3.2M|General|
|STVQA[2]|43K|General|
|MMC-INST[38]|500K|Doc/Chart/Screen|
|MathV360K[55]|338K|Math/Reasoning|
|MMC-Alignment[38]|250K<br><br>|Doc/Chart/Screen|
|DocReason[71]<br><br>|26K|Doc/Chart/Screen|
|ALLaVA[10]|1.7M|General|
|Cocotext[26]<br><br>|163K|General|
|Docvqa[49]|16K<br><br>|Doc/Chart/Screen|
|Geoqa+[11]|72K|Math/Reasoning|
|DocDownstream[71]|700K<br><br>|Doc/Chart/Screen|
|Cambrian [60]<br><br>|8.3M|General, General OCR, Math/Reasoning Doc/Chart/Screen, Text Instruct|
|DocStruct4M[66]<br><br>|4M|General OCR, Doc/Chart/Screen|
|LLaVA-onevision [33]|4M<br><br>|General, General OCR, Math/Reasoning Doc/Chart/Screen, Text Instruct|
|Docmatix[31]<br><br>|1.2M<br><br>|Doc VQA|
|Infinity-Instruct [5]<br><br>|7M|Text Instruct|
|Our Synthetic Data<br><br>|0.8M|Fine-grained Perception(single-instance) Attribute Reasoning Fine-grained Perception(Cross-instance) Relation Reasoning Coarse Perception, Logic Reasoning|

Table 8. Data Source, Size and Type of Training Data

- – Identify key elements
- – Identify transportation
- – Identify background details
- – Identify presence of objects
- – Identify natural environment scenery
- – Other image scenes
- – Identify stage
- – Identify indoor scene
- – Other image scene descriptions
- – Identify temperature state
- – Identify presence
- – Describe scene

- • Image Quality

- – Assess color and balance
- – Assess focus
- – Other image quality assessments
- – Identify quality issues
- – Assess brightness/ contrast
- – Assess color
- – Assess overall quality
- – Assess lighting

- – Assess overall clarity
- – Assess composition
- – Assess clarity
- – Detect noise
- – Assess sharpness

• Image Topic

- – Identify food
- – Identify book-related content
- – Identify animals
- – Identify medical condition
- – Identify geometric properties
- – Identify people
- – Identify portrait
- – Identify objects
- – Identify main subject
- – Other image topics
- – Identify text
- – Identify event
- – Identify diagram content
- – Identify book
- – Identify color

- – Identify content
- – Identify caption
- – Identify infographic/ cartoon style
- – Identify life cycle stage
- – Identify chart content
- – Identify image content
- – Identify book content
- – Identify vehicles
- – Describe image
- – Identify plant
- – Identify sports

- • Image Emotion

- – Detect overall emotion
- – Other image emotion
- – Read emotions from faces

- • Image Style

- – Other image styles
- – Identify image category

#### C.2. Fine-grained Perception (single-instance)

- • Object Localization

- – Locate object
- – Determine coordinates
- – Count objects
- – Identify specific object
- – Describe region
- – Detect presence
- – Provide bounding box
- – Determine orientation
- – Provide bounding box coordinates
- – Count people
- – Other object localization tasks
- – Provide descriptions
- – Count animals
- – Provide region description
- – Provide short description
- – Identify region
- – Other localization tasks

- • Attribute Recognition

- – Recognize texture
- – Recognize material
- – Recognize pattern
- – Recognize clothing
- – Recognize geometric properties
- – Recognize object presence
- – Recognize appearance characteristics
- – Recognize size
- – Recognize objects
- – Recognize color
- – Other attributes
- – Recognize formulas/tables/charts
- – Recognize orientation
- – Recognize shape

- – Recognize category
- – Count objects

- • OCR

- – Recognize printed text
- – Recognize text
- – Transcribe text from image
- – Extract text from image
- – Recognize text in images
- – Transcribe text in image
- – Extract text from images
- – Recognize formulas/ tables/ charts
- – Key Information Extraction
- – Transcribe text
- – Other OCR tasks

- • Identify specific object

– direct

- • Detect presence

– direct

#### C.3. Fine-grained Perception (cross-instance)

- • Spatial Relationship

- – Determine relative position
- – Determine spatial arrangement
- – Other spatial relationships
- – Determine coordinates
- – Count objects

- • Action Recognition

- – Recognize actions in video and text
- – Recognize sequence of actions
- – Recognize human-human interactions
- – Recognize human actions
- – Recognize animal actions
- – Recognize human-object interactions
- – Recognize actions

- • Attribute Comparison

- – Compare text
- – Other attribute comparison
- – Compare preferences
- – Compare ages
- – Compare materials
- – Compare values
- – Compare material
- – Compare shapes
- – Compare shapes/ colors/ textures/ sizes
- – Compare quantities
- – Compare sizes
- – Compare colors

- • Determine relative position

– direct

#### C.4. Relation Reasoning

• Social Relation

– Other social relations

- – Identify family/ friendship/ professional/ hostile relationships

- • Physical Relation

- – Identify spatial/ mechanical/ cause-effect relationships
- – Identify cause-effect relationships
- – Other physical relations

- • Nature Relation

– Other nature relations

#### C.5. Attribute Reasoning

- • Identity Reasoning

- – Other identity reasoning
- – Predict occupation/ role/ social status

- • Function Reasoning

- – Predict function of objects
- – Other function reasoning
- – New tag

- • Physical Property Reasoning

- – Other physical properties
- – Recognize geometric properties
- – Other physical property reasoning
- – Attribute Reasoning
- – Recognize formulas/ tables/ charts

#### C.6. Logic Reasoning

- • Structuralized Image-Text Understanding

- – Parse tables
- – Other image-text understanding
- – Parse geometric diagrams
- – Other Structuralized Image-Text Understanding
- – Parse sales data
- – Parse bar charts
- – Parse line charts
- – Parse text
- – Other charts
- – Parse other charts
- – Parse diagrams
- – Parse mathematical problem
- – Parse bar/ pie/ line charts
- – Parse formulas
- – Parse function plots
- – Parse charts
- – Parse word problems

- • Future Prediction

- – Predict trend/ social interaction/ physical movement/environmental changes
- – Other future predictions
- – Predict action sequence
- – Action Prediction
- – Predict series of actions

