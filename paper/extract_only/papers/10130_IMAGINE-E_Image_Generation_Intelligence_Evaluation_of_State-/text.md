### arXiv:2501.13920v1[cs.CV]23Jan2025

#### IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models

Jiayi Lei1,2∗, Renrui Zhang3∗, Xiangfei Hu1,2, Weifeng Lin3, Zhen Li3, Wenjian Sun1 Ruoyi Du2, Le Zhuo2, Zhongyu Li2, Xinyue Li2, Shitian Zhao2 Ziyu Guo3, Yiting Lu2, Peng Gao2†, Hongsheng Li3†

1Shanghai Jiaotong University, 2Shanghai AI Laboratory 3CUHK MMLab

∗ Equal Contribution † Corresponding Author

###### Abstract

With the rapid development of diffusion models, text-to-image (T2I) models have made significant progress, showcasing impressive abilities in prompt following and image generation. Recently launched models such as FLUX.1 and Ideogram2.0, along with others like Dall-E3 and Stable Diffusion 3, have demonstrated exceptional performance across various complex tasks, raising questions about whether T2I models are moving towards general-purpose applicability. Beyond traditional image generation, these models exhibit capabilities across a range of fields, including controllable generation, image editing, video, audio, 3D, and motion generation, as well as computer vision tasks like semantic segmentation and depth estimation. However, current evaluation frameworks are insufficient to comprehensively assess these models’ performance across expanding domains. To thoroughly evaluate these models, we developed the IMAGINE-E and tested six prominent models: FLUX.1, Ideogram2.0, Midjourney, Dall-E3, Stable Diffusion 3, and Jimeng. Our evaluation is divided into five key domains: structured output generation, realism, and physical consistency, specific domain generation, challenging scenario generation, and multi-style creation tasks. This comprehensive assessment highlights each model’s strengths and limitations, particularly the outstanding performance of FLUX.1 and Ideogram2.0 in structured and specific domain tasks, underscoring the expanding applications and potential of T2I models as foundational AI tools. This study provides valuable insights into the current state and future trajectory of T2I models as they evolve towards general-purpose usability. Evaluation scripts will be released at https://github.com/jylei16/Imagine-e.

###### Contents

###### 1 Introduction 4

- 1.1 Task Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 1.2 Quantitative Evaluation Criteria . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

###### 2 Evaluation 6

- 2.1 Structured Output Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 2.1.1 Code2Table . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.1.2 Language2Table . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 2.1.3 Code2Chart . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 2.1.4 Language2Chart . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 2.1.5 Equation Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 2.1.6 Language2Newspaper . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 2.1.7 Language2Paper . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 2.1.8 Json2Image . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 2.1.9 UI Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 2.1.10 Code Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 2.2 Realism and Physical Consistency Tasks . . . . . . . . . . . . . . . . . . . . . . . 20

- 2.2.1 Multi-Person . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 2.2.2 Human body . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 2.2.3 Photographic Image Generation . . . . . . . . . . . . . . . . . . . . . . . 23
- 2.2.4 Perspective Relation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 2.2.5 Physical understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- 2.3 Specific Domain Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- 2.3.1 Math . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 2.3.2 Fractal . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 2.3.3 Medical . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 2.3.4 3D Point Cloud . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- 2.3.5 3D Mesh . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- 2.3.6 Chemistry . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- 2.3.7 Biology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- 2.3.8 Robotics and Simulation Tasks . . . . . . . . . . . . . . . . . . . . . . . . 36
- 2.3.9 Autonomous Driving . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- 2.4 Challenging Scenario Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

- 2.4.1 Image with Mark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- 2.4.2 Set of Mark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- 2.4.3 Multilingual . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- 2.4.4 Dense OCR . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- 2.4.5 Emoji . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- 2.4.6 Irrational Scene Generation . . . . . . . . . . . . . . . . . . . . . . . . . 47
- 2.4.7 LLM QA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- 2.4.8 Watermark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- 2.4.9 Low Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- 2.4.10 Multi-image . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
- 2.4.11 Text Writing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61

- 2.5 Multi-style Creation Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63

###### 3 Conclusion 69

- 3.1 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 69
- 3.2 Task Complexity Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 69
- 3.3 Model Performance Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 69
- 3.4 Quantitative Benchmark Assessment . . . . . . . . . . . . . . . . . . . . . . . . . 70

###### 1 Introduction

With the rapid development of large models [21, 87, 90, 45, 44, 88, 28, 24], text-to-image (T2I) diffusion models [62, 63, 67] have emerged, showcasing impressive abilities in prompt following and high-quality image generation, including Imagen[67], Dall-E3 [6], the Stable Diffusion series [64], and Lumina-T2I [22] models, among others. Recently, Black Forest Lab released FLUX.1 [20], and Ideogram2.0 [36] also made its debut, showcasing exceptional performance. Existing evaluation methods [13, 40, 91] often suffer from issues such as overly simple tasks and a significant gap between evaluation results and human intuitive perceptions. In contrast, we designed IMAGINEE with detailing and challenging tasks, and scored models using a variety of scientific methods for quantitative evaluation. we delve deeply into the capabilities and performance of FLUX.1, Ideogram2.0, and other state-of-the-art T2I models to address the following question: Have T2I models entered a new era, and can these breakthroughs lead T2I models toward becoming generalpurpose models?

###### 1.1 Task Overview

As more powerful models emerge, T2I models are no longer limited to traditional image generation tasks. They demonstrated remarkable performance in various fields, ranging from text-to-image generation [64, 66, 7, 37, 89], controllable generation [85, 83, 12], and image editing [3, 8, 38] to video [30, 9], audio [42, 32], 3D [26, 27, 25], and motion [71, 86] generation. Beyond generation, recent works have also exhibited diffusion models’ capabilities in computer vision tasks, such as semantic segmentation [5, 79], depth estimation [39, 43], and image restoration [77].

To this end, we introduced IMAGINE-E, a comprehensive evaluation framework designed to benchmark text-to-image (T2I) generation models. Using IMAGINE-E, we selected six representative T2I models for comparison, including FLUX.1, Ideogram2.0, Midjourney, Dall-E3, Stable Diffusion 3, and Jimeng. These models were chosen based on their maturity, industry recognition, and diversity, encompassing both open-source and closed-source approaches. To scientifically and systematically evaluate these models, we designed five domains to rigorously assess and compare their capabilities. These domains include structured output generation, realism and physical consistency tasks, specific domain generation, challenging scenario generation, and different style image generation.

- • Structured Output Generation: In this task, we focus on evaluating the model’s ability to generate structured outputs such as tables, figures, and documents. These domains have rarely been specifically tested, making this a highly challenging task. It provides a substantial measure of the current level of alignment between T2I models and instructions, as well as their generation capabilities. Structured output tasks demand high-level understanding from models, requiring them to comprehend complex structured or natural language inputs while maintaining precise formatting in their output. These tasks also demand that models accurately extract and reproduce textual or numerical information from inputs into outputs. Structured output generation has immense practical applications in design, academic research, education, and more. This is also a crucial step for T2I models on their path to becoming foundation models, highlighting their potential as a universal visual output interface.
- • Realism and Physical Consistency Tasks: A critical criterion for assessing the quality of T2I models is whether the generated images adhere to the fundamental laws and requirements of the physical world. In this task, we rigorously test different T2I models’ understanding of human anatomy and physical laws. This task seeks to answer a broad question: Can AI truly understand the physical world? Do T2I models represent a world that abides by the laws of physics, with generated images merely reflecting a fragment of that world?
- • Specific Domain Generation: In this task, we carefully design a series of prompts from underrepresented academic or research fields to test the models’ breadth of knowledge. We gather prompts from specialized domains such as mathematics, 3D modeling, and medical fields to evaluate T2I models’ expertise in these areas. FLUX.1 and Ideogram2.0’s remarkable performance in this domain illustrates the expanding utility of T2I models, which hold the potential to contribute significantly to scientific research.
- • Challenging Scenario Generation: To further diversify the difficulty of our evaluations, we have collected a wide array of highly challenging tasks. These prompts enhance the

- diversity of prompt types and complexity, offering a more comprehensive assessment of the models’ abilities and performance.
- • Multi-style Creation Task: In this task, we have meticulously selected over thirty distinct artistic styles and crafted detailed prompts to evaluate the capabilities of T2I models in handling such fundamental tasks. This task assesses the T2I models’ understanding of various styles, their ability to generalize by integrating elements with significantly different styles, and the aesthetic quality of the images they generate.

###### 1.2 Quantitative Evaluation Criteria

In recent years, the development of text-to-image (T2I) models has significantly advanced the field of image generation. To evaluate the quality of these generated results, researchers have proposed various automated evaluation metrics. Among these, the following methods are commonly used:

- • CLIPScore [60]: This method leverages OpenAI’s CLIP model to assess image quality by computing the similarity between generated images and their corresponding text descriptions. Its advantage lies in the ability to directly compare text and images, providing contentrelevant evaluations. However, it has limitations, such as a lack of sensitivity to subtle artistic styles and compositions, which may lead to inaccurate scoring of high-quality images.
- • HPSv2 [76]: This newer visual quality assessment method aims to combine multiple evaluation dimensions to enhance the accuracy of image quality measurement. Although HPSv2 offers a comprehensive quality assessment, there is currently limited literature on the method, and its generalizability and effectiveness are yet to be fully validated.
- • Aesthetic Score [68]: This approach focuses on assessing the aesthetic quality of images by utilizing deep learning models to analyze aspects such as composition and color [81]. While it effectively captures aesthetic features, it is constrained by the limitations of its training data, potentially introducing biases in images with high stylistic diversity.
- • GPT-4o [56]: This study incorporates a scoring method based on GPT-4o, utilizing a prompt that evaluates the quality of generated images from four aspects: aesthetic appeal and alignment with human preferences, conformance to physical laws and realism, safety, and the degree of matching between the image and the text description. This method leverages the reasoning capabilities of the language model to score the generated results, addressing the shortcomings of the aforementioned methods.
- • Human: Our researchers use the same evaluation criteria as GPT-4o, focusing on four aspects: aesthetic appeal and alignment with human preferences, conformance to physical laws and realism, safety, and the degree of matching between the image and the text description. We conduct detailed scoring of the generation results from six models based on human aesthetic judgments. Additionally, we test the reliability of different evaluation systems by comparing and analyzing the differences and similarities between other evaluation methods and human evaluations.

Additionally, this study compares these automated scoring methods with human subjective ratings to assess their validity and consistency.

###### 2 Evaluation

In this section, we will conduct a systematic evaluation of six models across five domains: structured output generation, realism and physical consistency tasks, specific domain generation, challenging scenario generation, and multi-style creation. Each domain is further divided into specific sub-tasks to assess model performance in various detailed aspects.

We will visually compare the model outputs for an intuitive comparison and conduct quantitative evaluations using metrics such as CLIPScore, HPSv2, Aesthetic Score, and GPT-4o scores. Additionally, these quantitative evaluations will be compared with human perceptual ratings to assess the alignment between model evaluation metrics and human judgment. For CLIPScore, HPSv2, and Aesthetic Score, we have sampled a small set of carefully selected prompts, which are displayed in the images to allow direct comparison with human perception. However, these results may exhibit some degree of randomness. In the future, we plan to perform extensive sampling and evaluations to further refine the benchmarking process.

For the GPT-4o and human evaluations, the generated images will be assessed on the following aspects:

- • Aesthetic appeal and alignment with human preferences
- • Conformance to physical laws and realism
- • Safety (no copyright infringement, no NSFW content)
- • Alignment with the text description, including the accuracy of generated text and charts

Each of these four aspects will be rated on a three-level scale: A ("Highly meets the requirements"),

- B ("Moderately meets the requirements"), and C ("Does not meet the requirements"). A, B, and
- C correspond to scores of 2, 1, and 0, respectively. The final score is calculated as follows, with a maximum score of 10.

(Aesthetic score × 1 + Realism score × 2 + Safety score × 1 + Matching score × 2)/1.2

In the article’s subtask, we present the prompts used for testing the image output by each model. To visually represent the quality of the model outputs, we label images with a green smiley face if they are aesthetically pleasing, adhere to the physical world logic, and perfectly match the prompt requirements. Images with chaotic outputs that deviate significantly from the prompt are labeled with a red sad face. If the output images meet the aesthetic and prompt requirements to some extent but have minor flaws, we do not label them with either a smiley or sad face.

###### 2.1 Structured Output Generation

In the context of text-to-image models, structured output generation refers to the task where the model processes structured or natural language input and generates structured image outputs that meet the given requirements. The ability to produce structured outputs can, to some extent, reflect the model’s proficiency in following instructions, providing direction for the further development of text-to-image models toward becoming more comprehensive and versatile models.

In Sections 2.1.1, 2.1.3, 2.1.2, and 2.1.4, we will explore the tasks of code2table, code2figure, language2table, and language2figure, where different types of code or natural language inputs are used to generate tables or figures. In Section 2.1.5, we examine the models’ ability to generate complex equations. Sections 2.1.6 and 2.1.7 focus on the models’ capability to generate newspaper articles and academic papers from natural language descriptions. In Section 2.1.8, we introduce a new input format using JSON to describe a scene. In Section 2.1.9, we will investigate the models’ ability to design user interfaces based on code or language input. Finally, in section 2.1.10, we test T2I models’ ability to generate code.

###### 2.1.1 Code2Table

In previous work, several studies [80, 4] have made significant strides in the task of generating tables from code. In our study, we investigate the potential of text-to-image models for generating tables based on code inputs. Markdown2Table. We investigated the models’ ability to comprehend

markdown text and generate tables from input. The results are shown in right subplot of Figure 1. Using a simple 3 × 3 table as a test, we found that FLUX.1 [20] almost generated the table accurately, with only minor errors in specific data. However, Midjourney did not recognize the task as table generation. Ideogram2.0 [36], Dall-E3, Stable Diffusion 3, and Jimeng understood the intent to generate a table but were unable to produce it with complete accuracy.

LaTeX2Table. As shown in the left and middle sublplot of Figure 1, we used LaTeX format instead of markdown to test the models’ ability to generate more complex tables with 9 rows and 4 columns. We found that FLUX.1 demonstrated an extraordinary ability to process complex tables, almost perfectly generating the table as described in the prompt. Similar to the Markdown2Table task, Midjourney did not recognize the task as table generation. Ideogram2.0, Dall-E3 [6], and Stable Diffusion 3 [64] were able to generate images that resembled tables but lacked accurate content, while Jimeng struggled with handling certain special characters in the LaTeX format.

Sec. 2.1.1 Code2Table

[Figure 1]

[Figure 2]

Prompt

###### Prompt

\begin{table}[h] \centering \begin{tabular}{|c|c|c|c|} \hline Student & Math & English &Physics \\ \hline Mary & 98 & 99 & 85 \\ John & 95 & 90 & 95 \\ Jolin & 77 & 78 & 90 \\ Mike & 90 & 99 & 95 \\ Alice & 95 & 98 & 93 \\

\begin{table}[h] \centering \begin{tabular}{|c|c|c|c|} \hline Student & Math & English &Physics \\ \hline Mary & A & A+ & B \\ Jolin & C & B & A \\ Mike & A & A+ & A+ \\ Alice & A+ & A+ & A \\ Bona & A+ & B- & A+ \\ Amy & A+ & A+ & A+ \\ Chansder & A & A- & A+ \\ Joey & C+ & B+ & A- \\ Edward & A+ & A & A+ \\ \hline \end{tabular} \caption{A table of the test scores of the students in the class.}

[Figure 3]

###### Prompt

This is a table: | Class | Boys | Girls | |---------|--------|--------|

- | A | 20 | 10 |
- | B | 10 | 20 |
- | C | 11 | 19 |

Bona & 97 & 80 & 100 \\

Amy & 97 & 96 & 95 \\ Chansder & 99 & 99 & 98 \\ Joey & 95 & 78 & 99 \\ \hline \end{tabular} \caption{A table of the test scores of the students in the class.} \end{table}

\end{table}

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

FLUX.1

Ideogram2.0

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Midjourney

DALLE3

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Jimeng

Stable Diffusion 3 Stable Diffusion 3 Stable Diffusion 3

Jimeng

Jimeng

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### Illegal Illegal

Figure 1: Results on code2table task. Refer to Section 2.1.1 for detailed discussions.

- Table 1: The scoring of generation results by six models on code2table under different evaluation

- systems. Refer to Section 2.1.1 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 26.48 0.20 4.73 5.56 8.89 Ideogram2.0 29.17 0.23 5.30 4.44 7.50 Dall-E3 30.17 0.25 5.21 4.45 7.22 Midjourney 22.70 0.23 5.67 4.17 5.00 SD3 20.86 0.17 4.94 1.39 4.72 Jimeng 27.39 0.19 4.11 2.50 8.33

Score. The results are shown in Table 1. By comparing and observing the ratings of model outputs across four metrics, we found that the scores from CLIPScore, HPSv2 and Aesthetic Score did not align with the actual results. Through visual inspection of the generated images, FLUX.1 produced outputs most consistent with the format and content of the table in the prompt. However, the results obtained by these three metrics were not consistent with human observations. The scores GPT-4o were more in line with the actual situation.

###### 2.1.2 Language2Table

In this experiment, we aimed to explore the T2I models’ ability to transform natural language descriptions into tables. We described three tables with increasing levels of complexity. The results of all experiments are presented in Figure 2. It was observed that only FLUX.1, Ideogram2.0, Dall-E3, and Stable Diffusion 3 consistently grasped the intent to generate a table. However, Ideogram2.0 tended to generate more columns than described in the prompt, while Dall-E3 often produced blurry text in the tables. FLUX.1 outperformed all other models in this task, demonstrating superior text accuracy and an exceptional understanding of prompts, particularly with the third, the most complex prompt.

###### Sec. 2.1.2 Language2Table

[Figure 47]

[Figure 48]

Prompt

Prompt

###### The table has two columns and six rows.

The table has three columns and six rows. The first row contains the headers: "Name","Age", "Occupation“.

The first row contains the headers: “Name” and

“Occupation”. The subsequent rows contain data for five individuals. In the second row, in the first cell:"Alice", and in the second cell:"Engineer". In the third row, in the first cell:"Bob", and in the second cell:"Designer". In the fourth row, in the first cell:"Charlie", and in the second cell:"Teacher". In the fifth row, in the first cell:"Emy", and in the second cell:"Student". In the sixth row, in the first cell:"Jolin", and in the second cell:"Dancer". Each cell in the table contains text that is aligned to the left. The table has simple borders separating the rows and

The subsequent rows contain data for five individuals.

First row: Contains the headers: "Name," "Occupation," and "Age." Second row: in the first cell:"Alice," in the second cell:"Engineer," and in the third cell:"30." Third row: in the first cell:"Bob", in the second cell:"Designer," and in the third cell:"28." Fourth row: in the first cell:"Charlie", in the second cell:"Teacher," and in the third cell:"35." Fifth row: in the first cell:"Emy", in the second cell:"Student," and in the third cell:"22." Sixth row: in the first cell:"Jolin", in the second cell:"Dancer," and in the third cell:"26." Each cell in the table contains text that is aligned to the left, with simple borders separating the rows and columns.

columns.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

DALLE3

Midjourney

[Figure 65]

Midjourney

DALLE3

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Jimeng

###### Stable Diffusion 3 Stable Diffusion 3

Jimeng

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Figure 2: Results on language2table task. Refer to Section 2.1.2 for detailed discussions.

Score. The scores of output are shown in Table 2. The scoring results of CLIPScore and GPT-4o are consistent with human intuition, but the numerical results of GPT-4o differ significantly from human intuitive judgments.

- Table 2: The scoring of generation results by six models on language2table under different evaluation

- systems. Refer to Section 2.1.2 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human FLUX.1 32.99 0.21 4.89 3.61 7.78 Ideogram2.0 31.39 0.20 5.40 3.33 7.50 Dall-E3 30.03 0.20 5.10 3.61 6.11 Midjourney 28.86 0.20 4.66 3.33 5.56 SD3 29.99 0.23 5.09 3.33 5.28 Jimeng 31.17 0.23 6.21 3.34 5.00

2.1.3 Code2Chart

Several studies [65, 4, 18] have made significant strides in the task of generating charts from code. In our study, we investigate the potential of text-to-image models for generating charts based on code inputs. Bar chart. We conducted an experiment to evaluate T2I models’ ability to understand Matplotlib code and generate a corresponding chart. We began by designing a simple bar chart code, with the results presented in the left subplot of Figure 3. FLUX.1, Ideogram2.0, Dall-E3, and Jimeng were able to grasp the intent to generate a bar chart. Among these, FLUX.1, Ideogram2.0, and Dall-E3 successfully generated labels for all bars. However, only FLUX.1 and Ideogram2.0 produced the correct format for the bar chart. None of the models, however, generated the correct numerical values for the bars.

Line chart. We also conducted an experiment with a line chart, designed to show an increasing trend. The results are displayed in the right subplot of Figure 3. Except for Midjourney, all other models grasped the intent to generate a line chart. However, Stable Diffusion 3 and Jimeng failed to produce the correct line chart format. FLUX.1 and Ideogram2.0 understood the increasing trend, but none of the models were able to generate an accurate chart that strictly followed the prompt.

- Score. The scores of the model results in this task are shown in the Table 3. Only the results of HPSv2 are consistent with human intuition; however, all scores are relatively low, suggesting that these metrics may not effectively understand prompts with structured outputs.

Table 3: The scoring of generation results by six models on code2chart under different evaluation

- systems. Refer to Section 2.1.3 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 25.86 0.19 4.79 1.67 7.50 Ideogram2.0 30.32 0.24 5.08 2.08 7.92 Dall-E3 27.78 0.18 5.01 2.08 6.67 Midjourney 30.47 0.23 4.60 2.50 5.83 SD3 28.80 0.23 5.06 1.67 5.00 Jimeng 24.56 0.22 5.44 2.50 5.42

###### 2.1.4 Language2Chart

Bar chart. In this task, we assess T2I models’ ability to transform natural language descriptions into visual charts. As illustrated in the left subplot of Figure 4, we describe a simple bar chart and evaluate how well the models can reconstruct it. All models, except Midjourney, are capable of generating a bar chart format. However, only FLUX.1, Ideogram2.0, and Dall-E3 are able to accurately generate both the x-axis labels and the overall title of the chart. None of these three models, however, can precisely generate the correct values for each bar, though FLUX.1 performs the best, producing the bar heights closest to the target values.

###### Sec. 2.1.3 Code2Chart

[Figure 80]

Prompt

[Figure 81]

Prompt

import matplotlib.pyplot as plt plt.xlabel('Fruit’)

import matplotlib.pyplot as plt plt.figure(figsize=(20, 8), dpi=100)

fruits = ['Apple', 'Banana', 'Watermelon', 'Lemon', 'Orange’]

plt.ylabel('Number’) numbers = [10, 24, 36, 48, 20] plt.bar(fruits, numbers) plt.title('Number of Fruits’) plt.show()

- x = [1, 2, 3]
- y = [4, 5, 6] plt.plot(x, y)

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

DALLE3

[Figure 98]

Midjourney

Midjourney

DALLE3

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Stable Diffusion 3 Jimeng

###### Stable Diffusion 3

Jimeng

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Figure 3: Results on code2chart task. Refer to Section 2.1.3 for detailed discussions.

Pie chart. We also describe a simple pie chart to evaluate the models’ capabilities, with the results shown in the right subplot of Figure 4. While all models successfully generate the pie chart format, none are able to produce the correct ratios for the chart segments.

- Score. The scores of the model results in this task are shown in the Table 4. The results of several metrics are relatively consistent, with only CLIPScore differing from human intuitive judgments.

Compared to the Code2chart task, we observe that models perform better when the input is in natural language rather than code. This suggests that the models’ training data may lack sufficient multi-format input.

- Table 4: The scoring of generation results by six models on language2chart under different evaluation

- systems. Refer to Section 2.1.4 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 33.55 0.28 5.43 7.08 7.50 Ideogram2.0 32.87 0.27 4.75 3.34 6.25 Dall-E3 34.55 0.27 4.85 4.58 7.08 Midjourney 35.05 0.28 5.29 2.92 6.25 SD3 33.68 0.26 4.83 5.00 4.58 Jimeng 30.12 0.26 5.61 4.58 4.58

###### Sec. 2.1.4 Language2Chart

[Figure 113]

[Figure 114]

Prompt

Prompt

This bar chart is titled "Number of Fruits" and compares the quantities of different types of fruits. The x-axis, labeled "Fruit," lists five types of fruits: Apple, Banana, Watermelon, Lemon, and Orange. The y-axis, labeled "Number“, represents the quantity of each fruit. The heights of the bars correspond to the number of each fruit: Apple: 10 Banana: 20 Watermelon: 30

The pie chart is a circle divided into four slices, each representing a different portion. The first slice is red and takes up 50% of the pie.

The second slice is blue and takes up 30% of the pie.

The third slice is green and takes up 10% of the pie. The fourth slice is yellow and takes up 10% of the pie. Each slice has a label inside or next to it, showing its percentage. The colors help to easily identify each section, and the pie chart gives a clear visual representation of how the portions compare to each other.

###### Lemon: 40

Orange: 50 The chart visually shows that Lemon has the highest quantity at 48, while Apple has the lowest at 10.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Midjourney

DALLE3

DALLE3

Midjourney

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Figure 4: Results on language2chart task. Refer to Section 2.1.4 for detailed discussions.

###### 2.1.5 Equation Generation

Logically connected equations. The understanding and generation of mathematical formulas have long been a focus of research [57, 73, 49]. With the emergence of text-to-image models, we explore their ability to comprehend mathematical formulas and output them in image form. We conducted an equation generation experiment to evaluate the T2I models’ ability to generate equations. We used a set of logically connected equations, drawn from the derivation process of a linear equation in two variables, with the results shown in left, middle subplot of Figure 5. Only FLUX.1 and Jimeng were able to generate a roughly correct set of equations, with FLUX.1 generally outperforming the other models.

Independent equations. In right subplot of Figure 5, we observe that, aside from Midjourney, the other models can generate images containing mathematical symbols resembling equations. However, FLUX.1 is the most accurate. In particular, for the second set of equations, FLUX.1 almost perfectly reproduces all the equations.

- Score. The scores of the model results in this task are shown in the Table 5. The results of several metrics are relatively consistent, with only CLIPScore differing from human intuitive judgments.

- Table 5: The scoring of generation results by six models on equation generation under different

- evaluation systems. Refer to Section 2.1.5 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 19.47 0.17 5.09 4.72 7.50 Ideogram2.0 21.86 0.15 4.28 1.95 5.84 Dall-E3 21.83 0.16 4.56 4.44 5.83 Midjourney 21.68 0.17 4.15 2.78 4.44 SD3 20.65 0.17 4.70 4.17 3.89 Jimeng 19.02 0.15 4.94 2.78 4.72

Sec. 2.1.5 Equation Generation

[Figure 146]

[Figure 147]

[Figure 148]

Prompt

Prompt

Prompt

- a_1x + b_1y = c_1
- a_2x + b_2y = c_2 (a_1b_2 - a_2b_1)x = c_1b_2 - c_2b_1 x = \frac{c_1b_2 - c_2b_1}{a_1b_2 a_2b_1}

- a=b+c
- b=a+g
- c=b+h h=f+j w=e+x

ax^2 + bx + c = 0, a^2 + b^2 = c^2 y = mx + b y = e^{x} \log_b{a} = c

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

FLUX.1

FLUX.1

[Figure 158]

Ideogram2.0

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

Midjourney

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3

[Figure 185]

Jimeng

Stable Diffusion 3

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Figure 5: Results on equation generation task. Refer to Section 2.1.5 for detailed discussions.

###### 2.1.6 Language2Newspaper

We evaluated the ability of these models to generate newspaper images based on natural language descriptions. We simply specified the layout and headlines for different sections of the newspaper and guided the models to generate a newspaper page, and the results are shown in the left subplot of Figure 6. Among the models tested, Ideogram2.0’s results were significantly better than the others, successfully generating the corresponding layout and headlines in the specified positions and adhering to the artistic style of a newspaper. Dall-E3, Stable Diffusion 3, and Jimeng model were able to generate newspaper-style images, but their text generation had significant flaws. FLUX.1 produced mostly correct text and layout, but the style did not match that of a newspaper. Midjourney’s generation was unsatisfactory in both newspaper style and textual content.

In in the right subplot of Figure 6, We present a more complex example. We describe in greater detail the titles, style, content, and the placement of inserted images for each section of the newspaper. Although none of the models perfectly met the requirements of the prompt, Ideogram2.0 still outperformed the others, correctly generating the required layout and main titles. FLUX.1 was able to generate some of the titles correctly, but the layout had errors. Dall-E3, Stable Diffusion 3, Jimeng, and Midjourney barely generated any correct text or layout.

- Score. The scores of the model results in this task are shown in the Table 6. CLIPScore aligns relatively well with human intuition, while the other three metrics show significant discrepancies from human judgments, possibly because scoring in this task requires examining the specific text content within the images.

- Table 6: The scoring of generation results by six models on language2newspaper under different

- evaluation systems. Refer to Section 2.1.6 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 30.27 0.19 4.45 3.34 8.34 Ideogram2.0 32.95 0.26 5.21 4.58 9.16 Dall-E3 29.57 0.27 5.29 3.33 7.50 Midjourney 29.92 0.25 4.88 4.17 6.67 SD3 31.78 0.28 5.14 3.75 6.25 Jimeng 30.88 0.24 5.27 4.59 5.42

Sec. 2.1.6 Language2Newspaper

[Figure 192]

[Figure 193]

Prompt

Prompt

Newspaper Layout Overview Main Headline:"Technological Innovation Driving the Future: How AI is Changing Lives“ Subheadline:"From Labs to Everyday Life, the Endless Possibilities of AI“ Left Column: "AI in Healthcare: Revolutionizing Patient Care“ Right Column: Feature Article: "The Ethical Implications of AI: Balancing Innovation and Responsibility“ Bottom Section: Editorial: "The Future of AI: A Path Forward“ Page Number and Date:"August 24, 2024 | Page 1" (bottom right corner)

Newspaper Layout Overview Format: Traditional broadsheet with a large, sectioned front page. Header: Bold title in classic serif font with the newspaper’s name, "The Daily Times." Date, issue number, and logo are beneath the title.

Main Headline: Spanning the top third, written in large font: "Global Summit Yields Historic Agreement." This headline is accompanied by a large image of world leaders shaking hands, with a brief caption: "Leaders Celebrate

###### New Trade Deal." Columns: Left Column (Secondary Headline): "Local Elections: Candidates Face Off in

Final Debate." Includes a smaller image of the debate. Middle and Right Columns: Contain snippets of various articles, such as "Tech Giants Report Record Profits" and "Sports Update: Championship Game Results." These summaries direct readers to the full stories inside. Sidebar: Located on the right side, listing key contents with titles like "World News," "Sports Highlights," and "Entertainment This Week." Footer: A horizontal strip at the bottom for brief ads, with text like "Weekend Weather: Rain Expected" or "Flash Sale at Local Mall." Inside Layout: Consistent layout with text-heavy articles, images, and infographics. Major sections like "Business" and "Sports" have their own headers and may include full-page ads such as "Holiday Sale: 50% Off All Items!"

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Jimeng

###### Stable Diffusion 3

Stable Diffusion 3 Jimeng

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

- Figure 6: Results on language2newspaper task. Refer to Section 2.1.6 for detailed discussions.

###### 2.1.7 Language2Paper

In Figure 7, we evaluated the ability of these models to generate academic paper images based on natural language descriptions. We specified the paper’s title, author, abstract outline, and date to guide the models in generating the first page of an academic paper. Among the models tested, only FLUX.1 and Stable Diffusion 3 were able to correctly produce the layout of an academic paper, while the other models mistakenly generated a large number of decorative images. In terms of text accuracy, Ideogram2.0 and FLUX.1 performed the best, being able to accurately generate titles and subtitles.

Dall-E3 followed closely, while Midjourney and Stable Diffusion 3 almost failed to generate correct text.

Sec. 2.1.7 Language2Paper

[Figure 223]

Prompt

[Figure 224]

Prompt

Generate the first page of an academic paper

Title:"Impact of Climate Change on Coastal Erosion" Positioned at the top, centered, in large, bold font (e.g., 16-18 point).

ACademic paper: First Page Layout:

###### Title: The Impact of Artificial Intelligence on

Author(s):"Jane Doe" Listed below the title, centered, in standard font (e.g., 12-point). Institutional Affiliation: "Department of Environmental Science, XYZ University" Placed directly under the author’s name.

Modern Healthcare (Centered, large, bold) Authors: John Doe Jane Smith (Centered, affiliations below names: University A, University B) Date: August 24, 2024 (Centered below author information)

Date:"August 23, 2024" Centered below the author’s information. Abstract:"This study examines the effects of climate change on coastal erosion, analyzing data from various coastal regions over the past 50 years. The findings indicate a significant increase in erosion rates correlated with rising sea levels and increased storm frequency." Positioned below the author info, often italicized, approximately 150-250 words.

Abstract: (Brief summary of AI's impact on healthcare, covering diagnostics, treatment, and ethics) (Italicized, below the date)

Keywords: Artificial Intelligence, Healthcare, Diagnostics, Ethics (Below the abstract) Page Number: 1 (Upper right corner) This version

Keywords:"Climate Change, Coastal Erosion, Sea Level Rise, Environmental Impact" Listed below the abstract, indicating 3-6 key terms.

is more concise, focusing on the essential layout

and content.

Page Number:"1" Typically in the upper right corner of the page.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Jimeng

###### Stable Diffusion 3

Stable Diffusion 3 Jimeng

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

- Figure 7: Results on language2paper task. Refer to Section 2.1.7 for detailed discussions.

Score. The results of this task are shown in the Table 7. We can observe that all metrics differ somewhat from human intuitive judgments. This discrepancy arises because accurate evaluation requires a thorough understanding of the basic format of academic papers and a detailed comparison of the specific text content in the images, leading to insufficient accuracy of these evaluation metrics.

- Table 7: The scoring of generation results by six models on language2paper under different evaluation

- systems. Refer to Section 2.1.7 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 26.67 0.20 4.12 2.92 9.59 Ideogram2.0 33.87 0.21 4.26 3.75 8.33 Dall-E3 29.06 0.21 4.51 3.75 7.50 Midjourney 29.62 0.26 4.91 5.84 4.58 SD3 26.04 0.19 5.08 3.75 5.42 Jimeng 31.87 0.22 4.29 5.00 3.34

###### 2.1.8 Json2Image

We used a new prompt format to evaluate the ability of T2I models to understand the relationships between objects and generate images correctly. The prompt was designed in a JSON format, which is divided into three parts: objects, attributes, and relations. The objects section describes the items that appear in the image, the attributes section details the characteristics and specifics of each object, and the relations section describes the spatial or logical relationships between different items. An example is shown in Figure 8. In the first example, we found that, except for Midjourney, which cannot process this format, both Jimeng and Stable Diffusion 3 could only understand the main objects and combine them together, lacking logical coherence. Dall-E3 generated a green lens, while FLUX.1 and Ideogram2.0 performed the best. In the second example, except for Midjourney, the output from Jimeng failed to show the woman sitting down. In Stable Diffusion 3 and Dall-E3’s results, the bicycle’s tire was incomplete. FLUX.1 and Ideogram2.0 excelled in this task as well.

###### Sec. 2.1.8 Json2Image

[Figure 255]

Prompt

[Figure 256]

Prompt

{ "objects": [ "woman",

{ "objects": [ "man", "crocodile", "camera",

"dog",

"bicycle", "bench", "lake", "mountain", "sun"

"water", "tree",

"sky" ],

], "attributes": {

"attributes": { “man”: ["smiling", "brown hair", "beard","blue shirt"], "crocodile": ["open mouth", "sharp teeth", "green eyes",

"woman": [ "wearing hat", "blonde hair", "red jacket", "smiling" ], "dog": [ "brown fur", "wagging tail", "small size", "wearing collar" ], "bicycle": [ "blue frame", "black tires", "basket attached" ], "bench": [ "wooden", "old", "near lake" ], "lake": [ "clear water", "reflection of mountains" ], "mountain": [ "snow-capped",

"scales"],

"camera": ["silver body", "black lens", "green button" ], "water": [ "calm", "reflection visible"], "tree": [ "tall", "in background"], "sky": ["partly cloudy"]

"far distance" ],

}, "relations": {

"sun": [ "setting", "orange hue" ] },

"man_near_water": ["man", "water"], "man_near_crocodile": ["man", "crocodile"], "man_holding_camera": ["man", "camera"], "crocodile_in_water": ["crocodile", "water"]

"relations": {

"woman_sitting_on_bench": ["woman", "bench"], "dog_near_woman": ["dog", "woman"], "bicycle_leaning_on_bench": ["bicycle", "bench"], "lake_reflecting_mountains": ["lake", "mountain"], "sun_setting_behind_mountain": ["sun", "mountain"]

} }

} }

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

DALLE3

Midjourney

Midjourney

DALLE3

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Stable Diffusion 3 Jimeng

Jimeng

###### Stable Diffusion 3

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Figure 8: Results on json2image task. Refer to Section 2.1.8 for detailed discussions.

Score. The results of this task are shown in the Table 8, where we found that the evaluations from CLIPScore and GPT-4o are closer to human intuition, while Ideogram2.0 performs better in these two metrics.

- Table 8: The scoring of generation results by six models on json2image under different evaluation

- systems. Refer to Section 2.1.8 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human FLUX.1 18.96 0.22 6.44 5.84 10.00 Ideogram2.0 20.14 0.22 5.66 7.50 9.16 Dall-E3 16.39 0.24 6.11 5.84 7.50 Midjourney 14.00 0.18 5.85 7.08 9.16 SD3 18.55 0.26 6.75 4.16 6.25 Jimeng 19.40 0.26 6.40 6.25 8.75

2.1.9 UI Design

Previous work has explored how to use AI as an assistive tool for UI design [74, 35, 17, 75].In this work, we explore the potential of using text-to-image models for automating UI design. Code2UI. UI design is a common task for evaluating T2I models’ ability to follow instructions. We input HTML code into the models, and the results are shown in Figure 9. While all models generate some form of a web interface, Stable Diffusion 3 produces output that appears as meaningless gibberish. In comparison to the ground truth, only FLUX.1, Ideogram2.0, and Dall-E3 follow the instructions more accurately, generating web layouts containing the sections "About Me", "My Work", and "Contact."

Language2UI. In this task, we assess models’ ability to convert natural language into web interfaces, as shown in Figure 10. In the first example, both Jimeng and Stable Diffusion 3 produce unreadable text, while Dall-E 3 fails to generate a typical web interface. In contrast, FLUX.1, Ideogram2.0, and Midjourney generate legible text, with FLUX.1 and Ideogram2.0 excelling in instruction-following. In the second example, Jimeng, Stable Diffusion 3, and Midjourney produce blurry outputs, while Ideogram2.0 and Dall-E3 contain some chaotic text. FLUX.1 outperforms the other models, demonstrating better instruction-following.

Score. The results of this experiment are shown in the Table 9. Among the scores for CLIPScore, HPSv2, and Aesthetic score, FLUX.1 achieved a higher score. In human intuitive perception, the outputs of FLUX.1 and Ideogram2.0 are also better, while the results of GPT-4o are inconsistent with human intuitive perceptions in this experiment.

Table 9: The scoring of generation results by six models on UI design task under different evaluation

- systems. Refer to Section 2.1.9 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 28.19 0.22 4.94 4.45 8.89 Ideogram2.0 25.82 0.20 4.64 5.00 9.44 Dall-E3 27.32 0.21 4.77 4.72 6.67 Midjourney 24.68 0.18 4.56 5.28 6.94 SD3 27.98 0.19 4.88 5.28 5.00 Jimeng 26.27 0.21 4.77 4.72 6.67

###### 2.1.10 Code Generation

The use of LLMs for code generation has long been a focus of research [14, 19, 2, 72]. With the emergence of diffusion models, the question arises: Can text-to-image models also be used to generate code? In Figures 11 and 12, we examine the models’ capability to generate various types of code, including Python and C programs, as well as barcodes and QR codes, in order to explore the potential for generalizing text-to-image (T2I) models into more fundamental models. In Figure 11, the models are expected to generate images containing correct program code. However, none of the models

###### Sec. 2.1.9 UI Design

[Figure 286]

Prompt

<!DOCTYPE html> <html lang="en"> <head>

<meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0"> <title>Simple Webpage</title>

</head> <body>

- <h1>Welcome to My Webpage</h1> <p>This is a simple webpage layout.</p>
- <h2>About Me</h2> <p>A brief introduction about yourself or the topic of the webpage.</p>

<h2>My Work</h2> <p>Details about your work, projects, or content you want to showcase.</p>

<h2>Contact</h2> <p>Provide contact details or a form for visitors to reach out to you.</p>

</body> </html>

| |
|---|

###### GT

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

FLUX.1

Ideogram2.0 DALLE3

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Midjourney Stable Diffusion 3 Jimeng

[Figure 297]

[Figure 298]

[Figure 299]

- Figure 9: Results on UI design. Refer to Section 2.1.9 for detailed discussions.

Sec. 2.1.9 UI Design

[Figure 300]

[Figure 301]

Prompt

Prompt

The webpage features a modern and sleek design with a dark-themed background. At the top, there's a prominent header with the brand's logo on the left, and a navigation menu on the right, including links to "Home," "Products," "Features," "Support," and "Contact“. Below the header, there's a large hero section showcasing the latest digital product, with a high-resolution image of the product on one side and a

The shopping website features a clean and user-friendly interface. At the top, there’s a navigation bar with categories. On the right side of the navigation bar, there’s a search bar where users can quickly find products, and next to it are icons for the shopping cart and user

###### account.

Below the navigation bar, a large banner showcases current promotions or featured products, with eye-catching images and discount offers. Scrolling down, the main section displays products in a grid layout, with each item showing an image, name, price, and an "Add to Cart" button.

###### brief description with a "Buy Now" button on the other.

Scrolling down, the page highlights various product categories, such as "Smartphones," "Laptops," "Wearables," and "Accessories," each represented by an image and a short description. There's also a section dedicated to customer reviews and testimonials, featuring user ratings and feedback.

On the left side, there’s a filter sidebar where users can sort products by categories, price range, brand, and ratings.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

DALLE3

Midjourney

Midjourney

DALLE3

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

- Figure 10: Results on UI design. Refer to Section 2.1.9 for detailed discussions.

produce accurate outputs. Instead, they generate images depicting computer screens with code-like visuals. Similarly, in Figure 12, where the tasks are to generate valid barcodes and QR codes, all models fail to produce correct results. These findings suggest that significant further development is required before t2i models can evolve into foundational models capable of handling such tasks.

Score. The experimental results of this task are shown in Table 10. It can be observed that due to the difficulty of the task, the performance of several models is not high.

- Table 10: The scoring of generation results by six models on code generation under different evaluation systems. Refer to Section 2.1.10 for detailed discussions.

Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 28.10 0.22 4.50 4.38 5.00 Ideogram2.0 27.41 0.25 4.99 4.58 4.38 Dall-E3 25.70 0.23 4.44 5.42 4.17 Midjourney 25.84 0.24 5.00 5.63 4.17 SD3 26.49 0.24 5.02 4.59 2.50 Jimeng 30.53 0.26 4.89 4.79 2.50

Sec. 2.1.10 Code Generation

[Figure 326]

Prompt

[Figure 327]

A highly detailed technical illustration of Python code being written on

Prompt

a modern computer screen. The code on display is a simple Python script that prints "Hello world" to the console. The screen shows a clean, organized coding environment, with the text editor highlighting the Python syntax in vibrant colors. A line of code reads `print("Hello world")`, perfectly formatted with correct indentation. The background features a modern workspace with subtle lighting, emphasizing the simplicity of coding.

Generate a simple Python script that uses the print function to output the phrase "Hello, World." in the console, highlighting Python syntax.

Created Using: technical precision, clean design, modern aesthetics, vibrant code highlights

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

DALLE3

Midjourney

Midjourney

DALLE3

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

- Figure 11: Results on code generation. Refer to Section 2.1.10 for detailed discussions.

[Figure 352]

FLUX.1

Midjourney

[Figure 353]

Stable Diffusion 3 Jimeng

[Figure 354]

[Figure 355]

Prompt

Ideogram2.0

[Figure 356]

DALLE3

[Figure 357]

Prompt

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

FLUX.1

Midjourney

[Figure 365]

Stable Diffusion 3 Jimeng

[Figure 366]

[Figure 367]

Ideogram2.0

[Figure 368]

DALLE3

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

A close-up of an EAN-13 barcode featuring the code 6974336958790, centered on a clean white background. The black bars are crisp and clearly defined, with the numbers "6974336958790" displayed prominently underneath in bold, black font. The overall image is minimalist, ensuring the barcode is easily scannable and legible.

Created Using: high contrast, sharp focus, minimalist design, realistic barcode depiction

Generate a QR code that links to this URL: https://chatgpt.com/

Sec. 2.1.10 Code Generation

- Figure 12: Results on code generation. Refer to Section 2.1.10 for detailed discussions.

###### 2.2 Realism and Physical Consistency Tasks

In the image and video generation task, achieving realism and physical consistency is crucial. Previous works have made significant contributions to this area of research [54, 46, 47, 33]. We aim for models to generate images that are not only visually compelling but also believable and grounded in the physical world. To assess a model’s ability to understand and replicate real-world dynamics, we have designed a set of tasks that evaluate its grasp of physical laws.

In Section 2.2.1, we evaluate the models’ ability to generate credible human figures in complex multi-person settings. Section 2.2.2 focuses on assessing the models’ capability to accurately render human bodies and poses. In Section 2.2.3, we incorporate various photographic terminologies into the prompts to test the models’ understanding of photography techniques. Section 2.2.4 examines the models’ ability to interpret and generate correct perspective relationships within realistic scenes. Section 2.2.5 explores the extent to which T2I models understand the fundamental physical laws of the real world.

###### 2.2.1 Multi-Person

Generating images with multiple characters has always been a highly challenging task [84]. Figure 13 depicts the visualization results of six models in generating images based on prompts involving multiple persons. FLUX.1 demonstrates a strong ability to capture overall details from the prompts. And Stable Diffusion 3 [64], Midjourney, and Jimeng struggle with handling the overlapping and nonoverlapping aspects of multiple persons. Midjourney often cuts off half of a face, and Jimeng produces disjointed upper body parts. In the second part of Figure 13, Jimeng and FLUX.1 successfully generate images of multiple persons on a crowded subway with minimal distortion. FLUX.1, in particular, handles facial features and overlapping boundaries well, though its color palette is somewhat monotonous, and the depicted actions are limited. Conversely, Stable Diffusion 3 introduces significant distortions, notably a visible distortion at the junction of a blonde woman’s hair and another man’s face. Midjourney also exhibits distortion, particularly in the distant background of the subway scene.

###### Sec. 2.2.1 Multi-Person

[Figure 378]

[Figure 379]

Prompt

Prompt

Photo of a hospital waiting room with people of various ages and conditions, some anxiously waiting, others reading or looking at their phones.

Photo a crowded subway car during rush hour, with people of all ages and backgrounds standing and sitting, each absorbed in their own world.

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

Stable Diffusion 3 Jimeng

Jimeng

###### Stable Diffusion 3

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

Figure 13: Results on multi-person task. Refer to Section 2.2.1 for detailed discussions.

Score. The results of this experiment are shown in Table 11. Both CLIPScore, HPSv2, and GPT-4o consider Midjourney’s output to be superior; however, upon our careful observation, the human forms in FLUX.1 appear more realistic.

- Table 11: The scoring of generation results by six models on multi-person under different evaluation

- systems. Refer to Section 2.2.1 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human FLUX.1 26.31 0.29 5.69 5.00 9.17 Ideogram2.0 25.60 0.31 6.04 7.08 5.00 Dall-E3 25.05 0.29 5.51 7.01 7.08 Midjourney 27.64 0.31 5.36 7.08 7.92 SD3 23.60 0.30 5.98 6.25 7.08 Jimeng 26.87 0.30 5.03 6.67 5.84

2.2.2 Human body

In Figure 14-15, we examine the models’ ability to accurately generate human body, with a particular focus on the hands and feet, which are difficult tasks in image synthesis.

Hands. For instance, in the left subplot of Figure 14, all the models produce extra fingers except Dall-E3 [6]. Specifically, the Stable Diffusion 3 and Jimeng models exhibit entirely irrational hand structures. The image generated by Ideogram2.0 looks fake. Midjourney demonstrates a capacity to capture significant hand details, and FLUX.1 [20] achieves the most accurate body structure.

Feet. In the right subplot of Figure 14, both Midjourney and Stable Diffusion 3 generate the totally wrong foot structures, and Dall-E3 even result illegal, whereas FLUX.1, Jimeng and Ideogram2.0 produce more anatomically correct feet, despite Jimeng and Ideogram2.0 displaying oddly legs. Overall, FLUX.1 exhibits superior human body structure generation compared to the other models, though it still requires improvements in rendering the correct number of fingers.

Pose. In Figure 15, we examine the models’ ability to generate accurate human poses. In the first example, the desired pose is the Tree Pose (Vrksasana) from yoga. FLUX.1, Ideogram2.0, Stable Diffusion 3, and Jimeng successfully generate a woman in the correct pose. However, the poses generated by Dall-E3 and Midjourney are inaccurate, possibly due to a lack of understanding of Vrksasana. While their outputs fit the general prompt description, even they do not accurately capture the specific yoga position. In the second example, only Jimeng precisely follows the Warrior II Pose, but it overlooks the prompt detail of "facing the ocean". Ideogram2.0 and Dall-E3 fail to depict the correct yoga pose but align more closely with the general description of the pose.

Score. All scoring results for this task are shown in Table 12. The output of FLUX.1 is closer to reality, GPT-4o’s evaluation aligns with human perception, while CLIPScore, HPSv2, and Aesthetic Score differ significantly from human intuition.

Table 12: The scoring of generation results by six models on human body under different evaluation

- systems. Refer to Section 2.2.2 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 27.16 0.27 5.70 7.91 8.12 Ideogram2.0 30.86 0.26 5.92 6.66 6.25 Dall-E3 28.42 0.28 5.77 8.75 7.08 Midjourney 30.26 0.26 5.66 7.07 5.42 SD3 30.08 0.27 6.23 4.79 6.46 Jimeng 28.77 0.27 5.76 6.04 6.46

###### Sec. 2.2.2 Human Body

[Figure 407]

[Figure 408]

Prompt

Prompt

A hyper-realistic photograph of a person, with a close-up

A hyper-realistic photograph of a person, with a close-up focus on their hands. The hands are elegantly posed, showing intricate details of the skin, veins, and texture. The background is softly blurred, drawing attention to the hands. Gentle natural light enhances the depth and realism, highlighting the fine lines and subtle movements.

focus on their feet. The feet are bare, showing intricate details of the skin, toes, and subtle texture. The background is softly blurred, drawing all attention to the feet. The lighting is natural and soft, highlighting the contours and fine details of the feet, capturing a sense of realism and life.

Created Using: realism, intricate detail, soft focus, natural lighting

Created Using: realism, intricate detail, soft focus, natural lighting

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Stable Diffusion 3 Jimeng

Jimeng

###### Stable Diffusion 3

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

- Figure 14: Results on human body task. Refer to Section 2.2.2 for detailed discussions.

[Figure 440]

FLUX.1

Midjourney

[Figure 441]

Stable Diffusion 3 Jimeng

[Figure 442]

[Figure 443]

Prompt

Ideogram2.0

[Figure 444]

DALLE3

[Figure 445]

Prompt

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

FLUX.1

Midjourney

[Figure 453]

Jimeng

[Figure 454]

Ideogram2.0

[Figure 455]

DALLE3

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

A detailed photograph of a woman in the Tree Pose (Vrksasana) yoga position, standing tall with one foot placed against her opposite thigh, her arms extended above her head. She is practicing on a grassy hilltop, surrounded by lush green trees and distant mountains. The soft sunlight illuminates her, casting long shadows on the

ground, and a gentle breeze moves through the scene.

Created Using: realism, natural elements, soft lighting, serene atmosphere

A serene painting of a man performing the Warrior II Pose (Virabhadrasana II) on a beach at sunrise. His stance is strong, with his front knee bent and arms extended parallel to the ground, facing the ocean. The background is filled with vibrant hues of the rising sun reflecting on the water, and the calm waves gently roll onto the shore.

Created Using: vibrant colors, peaceful setting,

dynamic composition, natural lighting

[Figure 465]

[Figure 466]

Sec. 2.2.2 Human Body

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

Stable Diffusion 3

[Figure 471]

[Figure 472]

- Figure 15: Results on human body task. Refer to Section 2.2.2 for detailed discussions.

###### 2.2.3 Photographic Image Generation

In Figure16-19, we explore the model’s ability to generate images that meet specific requirements based on photographic terminology.

- Setting 1: We tested blurred bokeh backgrounds and depth of field. The models generally understood these concepts, with FLUX.1, Midjourney, and Jimeng performing best. Results are shown in the left subplot of Figure 16.
- Setting 2: We tested long exposure, specifically time-lapse photography represented as star trails. Dall-E3’s images had excessive star trails that appeared unnatural, followed by Ideogram2.0 and Midjourney. FLUX.1 and Stable Diffusion 3 produced the best overall results. Results are shown in the middle subplot of Figure 16.
- Setting 3: We examined macro photography and the concept of copy space. All models managed macro photography. Copy space refers to large blank areas in images for adding text, graphics, or other design elements. Dall-E3 mistakenly added unspecified text directly. Results are shown in the right subplot of Figure 16.

###### Sec. 2.2.3 Photographic Image Generation

[Figure 473]

[Figure 474]

[Figure 475]

Prompt

Prompt

Prompt

A long exposure photo of a clear night sky in a remote location, capturing the trails of stars as they move across the sky. Include a foreground of silhouetted trees or a mountain range.

Close up of dew on green leaf, macro photo, copy space concept, banner with copyspace area for text. Background with blurred background. Water droplets on tulip leaves, closeup.

Photo of a sunset by beach with a sharp foreground flower and a beautifully blurred bokeh background, showcasing the depth of field.

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

Midjourney

Midjourney

DALLE3

DALLE3

Midjourney

DALLE3

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Jimeng

Stable Diffusion 3 Jimeng

###### Stable Diffusion 3 Stable Diffusion 3

Jimeng

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

- Figure 16: Results on photographic image generation. Refer to Section 2.2.3 for detailed discussions.

- Setting 4: The setting involved terms like close-up shots, which refer to capturing above the chest. FLUX.1 and Stable Diffusion 3 missed this detail, while Midjourney performed best in overall style. Ideogram2.0’s images were darker, and Dall-E3’s output conflicted with photographic styles. Results are shown in the left subplot of Figure 17.
- Setting 5: Tilt-shift photography, used to alter the focus and depth of field, typically for creating miniature scenes, was tested. Dall-E3 performed best with this keyword, and all models could accurately generate images as prompted. Results are shown in the middle subplot of Figure 17.
- Setting 6: For golden tones, FLUX.1 and Dall-E3 excelled, while other models failed to achieve the effect. For symmetrical composition, only Stable Diffusion 3 missed the mark. For telephoto lens, backlighting, and soft light, FLUX.1 failed to deliver the telephoto effect but had the best lighting.

Dall-E3’s lighting was decent, while others only achieved soft light. Results are shown in the right subplot of Figure 17.

###### Sec. 2.2.3 Photographic Image Generation

[Figure 517]

[Figure 518]

Prompt

[Figure 519]

Prompt

Prompt

close up shot, A beautiful 18 years old Chinese girl, short hair, walking in the summer reed marshes, with green light ray, filled with summer vibe, film grain, portrait shot, in the style of renko kawauchi’s photography, no reflection, rich detail, White chinese dress.

Autumn colours of the Forbidden City, maple leaves falling, golden tones, symmetrical composition, telephoto lens, backlighting, soft light, stillness, sense of history, red walls and yellow tiles, Beijing city background.

Tilt-shift photography, a corgi standing on the grass,

realistic photography, shallow depth of field, natural light, wildlife photography, extreme details

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

- Figure 17: Results on photographic image generation. Refer to Section 2.2.3 for detailed discussions.

- Setting 7: We tested stunning photorealism, cinematic composition, and minimalist style. Midjourney and Ideogram2.0 had the most realistic images, followed by FLUX.1. Minimalist style was harder to judge, but FLUX.1 and Stable Diffusion 3 had the fewest elements. For the ’shot on Fujifilm’ look, only Jimeng and Dall-E3 struggled to achieve the retro film style with subtle contrasts. In terms of professional photography techniques, atmospheric lighting, natural gradients, and cinematic depth, Jimeng’s contrast was too intense. FLUX.1 had the best gradient effect, while Dall-E3’s gradients felt forced and ineffective. Results are shown in the left subplot of Figure 18.
- Setting 8: Double exposure, intended to capture reflections of people on glass, was tested. Only FLUX.1 and Midjourney met expectations; Ideogram2.0 and Dall-E3 partially achieved the effect, while Jimeng had clear issues, and Stable Diffusion 3 completely failed to recognize the keyword. For soft focus, delicate light play, cinematic quality, soft shadows, and artistic composition, the overall softness was best in Midjourney and FLUX.1. Results are shown in the middle subplot of Figure 18.
- Setting 9: The 28mm lens, a wide-angle lens that maintains background clarity, was tested. Ideogram2.0 and Jimeng did not achieve this effect. For studio lighting, interpreted as artificial lighting typical of a studio, Ideogram2.0 only captured regular artificial light. FLUX.1, Midjourney, and Jimeng performed best in high-definition photography, professional lighting, cinematic depth, and soft focus, which emphasized facial contours and details, while others were slightly weaker. Results are shown in the right subplot of Figure 18.
- Setting 10: Involving aerial environment photography and the blue hour, Dall-E3’s results were slightly inferior; others performed well. The requirement for a high-resolution image from a high vantage point with the Sony A7R IV was best met by Ideogram2.0 and Midjourney, with more harmonious and softer color tones. Results are shown in the left subplot of Figure 19.
- Setting 11: Under cinematic lighting, Dall-E3 produced the best facial lighting, Midjourney achieved a dreamy effect, and FLUX.1 had the most realistic lighting. For surrealism, vibrant colors, and professional photography techniques—using methods like distortion, collage, and supernatural elements to create dreamlike atmospheres. FLUX.1 failed to capture this, Stable Diffusion 3 had issues with hand and scene generation, Jimeng generated anime-style images, and Ideogram2.0

[Figure 560]

[Figure 561]

[Figure 562]

Prompt

Prompt

Prompt

A photoshoot of a handsome 25-year-old Asian man posing confidently, wearing gold 'Davis' spectacles and white clothes. He has black short-cut hair and detailed facial features. The image is captured with a 28mm lens, using studio lighting with soft silver and brown tones. The style is realistic with a romantic, Asian-inspired aesthetic. Created

Double exposure of a mysterious woman, silhouette. The forest landscape is seen in the double exposure, blending seamlessly with her face, highlighting the natural elements. Stunning portrait. Moody, ethereal tones. High resolution, soft focus, delicate light play, cinematic quality. Created Using: fine detail, surreal atmosphere, soft shadows,

A green hillside with a forest, captured in stunning photorealism, cinematic composition, and minimalistic style. The scene is shot on Fujifilm, with a wide-angle lens to emphasize the vastness of the landscape. The colors are rich and natural, with a subtle grain that enhances the organic feel. Created Using: professional photography techniques, atmospheric lighting, natural gradients, cinematic depth.

Using: high-definition photography, professional lighting,

artistic composition.

cinematic depth, soft focus

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

Jimeng

Jimeng

Stable Diffusion 3 Jimeng

###### Stable Diffusion 3 Stable Diffusion 3

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

- Figure 18: Results on photographic image generation. Refer to Section 2.2.3 for detailed discussions.

mistook kite shapes for fish. Midjourney depicted kites as butterflies but with excellent overall style, while Dall-E3 performed best. Results are shown in the middle subplot of Figure 19.

- Setting 12: In street photography, soft and diffused light was required; Dall-E3 notably violated this, and Ideogram2.0’s tone was too cool. For cinematic framing, dynamic composition, natural reflections, urban realism, and soft lighting, reflections were best captured by FLUX.1 and Midjourney. Jimeng, Stable Diffusion 3, and Ideogram2.0 showed varying issues, with Ideogram2.0’s water ripple effects being notably problematic, and Dall-E3’s composition defying logic. Results are shown in the right subplot of Figure 19.

- Score. The results of this experiment are shown in Table 13. These metrics differ significantly from human intuition, with only GPT-4o and CLIPScore’s scores being relatively consistent with human evaluations. This may be due to the presence of numerous technical terms related to photography in the prompts, which the other metrics may not fully comprehend.

- Table 13: The scoring of generation results by six models on photographic image generation under different evaluation systems. Refer to Section 2.2.3 for detailed discussions.

Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 28.49 0.28 6.09 7.29 9.38 Ideogram2.0 28.88 0.30 6.22 7.57 7.08 Dall-E3 29.31 0.28 6.13 6.87 5.90 Midjourney 30.70 0.29 6.27 8.61 8.68 SD3 29.56 0.30 6.38 7.01 6.87 Jimeng 29.93 0.30 6.46 6.66 7.56

[Figure 606]

[Figure 607]

[Figure 608]

Prompt

Prompt

Prompt

Aerial environment photography of snow-covered peaks and frozen lakes in Banff National Park during winter. Captured during the Blue hour, highlighting the icy landscape and tranquil mood. Shot on a Sony A7R IV from a high vantage point, emphasizing the serene atmosphere.

A striking portrait of a handsome 18-year-old Chinese boy with black, very short hair, flying blue fishing kites in an open field. The scene captures the dynamic motion of the kites against a sky-blue and white background, using cinematic lighting to enhance the atmosphere. The image combines hyperrealism with a Japanese-inspired style, focusing on the intricate details of the marinethemed kites. Created Using: ultra-high definition, surrealism, vibrant colors, professional photography techniques

A high-resolution street photography of a rainy day in Glasgow's city centre, bustling with crowds and many cars. The wet pavement gleams with reflections from buildings, people, and headlights, forming intricate patterns in the puddles. Raindrops gently ripple across the surface, adding

dynamic texture. Soft, diffused light filters through the

overcast sky, enhancing the moody atmosphere. Cars leave wet trails as they move through the street, while pedestrians hold umbrellas and rush through the city. Created Using: cinematic framing, dynamic composition, natural reflections, urban realism, soft lighting

Created Using: detailed composition,

natural lighting, crisp clarity, dynamic range

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

Midjourney

[Figure 635]

DALLE3

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

Stable Diffusion 3 Jimeng

[Figure 647]

Jimeng

Jimeng

###### Stable Diffusion 3 Stable Diffusion 3

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

- Figure 19: Results on photographic image generation. Refer to Section 2.2.3 for detailed discussions.

###### 2.2.4 Perspective Relation

In Figure 20 and Figure 21, we evaluated the models’ ability to correctly handle perspective relationships. Most of the tested models demonstrated excellent performance, whether dealing with simple track scenes or more complex urban streets and library settings, generally aligning well with real-world perspective. However, Stable Diffusion 3 produced images with a certain degree of distortion, performing the worst in terms of matching real-world perspective relationships.

- Score. The results of this experiment are shown in Table 14. We can see that in this task, only GPT-4o’s scores align relatively well with human ratings. This may be because the experiment involves physical relationships such as perspective, requiring the evaluation metrics to have a certain understanding of the fundamental principles of the physical world.

- Table 14: The scoring of generation results by six models on perspective relation under different

- evaluation systems. Refer to Section 2.2.4 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 25.42 0.27 5.82 7.50 8.61 Ideogram2.0 27.31 0.29 6.38 5.28 5.56 Dall-E3 25.95 0.29 6.32 7.78 7.50 Midjourney 26.58 0.26 6.00 8.89 8.05 SD3 27.38 0.28 5.90 7.22 7.50 Jimeng 28.26 0.30 6.34 8.89 8.33

###### Sec 2.2.4 Perspective Relation

[Figure 654]

[Figure 655]

Prompt

Prompt

###### Generate an image of a bustling city street viewed from

Photo of a set of train tracks leading into the horizon, using perspective lines to create a sense of depth and distance.

ground level, where the buildings rise high above,

converging towards a vanishing point in the distance. Include details like cars, streetlights, and pedestrians to emphasize the sense of scale.

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

Jimeng

###### Stable Diffusion 3

Stable Diffusion 3 Jimeng

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

- Figure 20: Results on perspective relation task. Refer to Section 2.2.4 for detailed discussions.

Sec 2.2.4 Perspective Relation

Prompt

[Figure 684]

Create a scene of an old, grand library with tall bookshelves lining the walls, receding into the distance. The perspective should make the shelves appear to converge, with a central reading area at the far end of the room.

[Figure 685]

FLUX.1

Midjourney Stable Diffusion 3 Jimeng

[Figure 686]

[Figure 687]

[Figure 688]

Ideogram2.0 DALLE3

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

- Figure 21: Results on perspective relation task. Refer to Section 2.2.4 for detailed discussions.

###### 2.2.5 Physical understanding

In the T2I pipeline, we give the image caption to the model, then the model generates an image reflecting the caption content, visually correct. In this process, does the model do understand the world’s physical law [52]? To test this point, we describe a real-world physical scene in the prompt. To generate an image that conforms to the laws of physics, the models need to truly understand the physical law. Here we describe two scenes: a glass cup falling to the ground and the water’s temperature is over 100 celsius degrees.

The results are shown in Figure 22. In the first scene, only Ideogram2.0 and Jimeng can generate the physically correct image: the glass cup shattered into pieces. In the second scene, all models perform well: the water boiled, except FLUX.1.

Score. The results of this task are shown in the Table 15. It can be observed that HPSv2, GPT-4o, and human perception are largely consistent. Ideogram2.0 achieved the highest score in the Aesthetic Score, which also aligns with human perception. However, the CLIPScore differs significantly from human perception.

- Table 15: The scoring of generation results by six models on physical understanding under different

- evaluation systems. Refer to Section 2.2.5 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 22.10 0.26 5.67 2.92 4.17 Ideogram2.0 20.79 0.27 5.99 8.34 9.16 Dall-E3 24.94 0.26 5.91 7.08 6.66 Midjourney 25.94 0.26 5.76 5.42 7.50 SD3 22.78 0.24 5.99 4.17 5.00 Jimeng 23.58 0.20 5.22 6.25 6.25

###### Sec. 2.2.5 Physical Understanding

[Figure 697]

[Figure 698]

Prompt

Prompt

If the temperature of the water is over 100 Celsius, what would happen?

If a glass cup falls from a high place to the ground, what would happen?

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

Jimeng

###### Stable Diffusion 3 Stable Diffusion 3

Jimeng

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

Figure 22: Results on physical understanding task. Refer to Section 2.2.5 for detailed discussions.

###### 2.3 Specific Domain Generation

With the advancement of T2I models, their usefulness has expanded to various domains. These models hold the potential to generate high-quality, domain-specific data, paving the way for significant contributions to technological innovation and interdisciplinary research.

- In Section 2.3.1, we assess the models’ understanding of mathematical terminology and their capability to generate math-related images based on given descriptions. Section 2.3.2 explores the models’ performance in generating images within fractal settings. In Section 2.3.3, we evaluate the models’ capability to produce medical images with potential applications in medical research. In Sections

- 2.3.4 and 2.3.5, we prompt the models to generate 3D images. Lastly, in Sections 2.3.6 and 2.3.7, we assess the models’ ability to generate images related to chemistry and biology. Section 2.3.8 explores the working environments of robots in embodied intelligence, while Section 2.3.9 investigates tasks in autonomous driving scenarios.

###### 2.3.1 Math

In Figure 24, we explore the models’ mathematical ability, especially geometrical concepts. For the first example of a right-angled triangle in Figure 24, FLUX.1 [20], Stable Diffusion 3 [64] and Jimeng try to present the outputs in a mathematical format. However, FLUX.1 fails to accurately depict the correct geometric relationships, and the output from Stable Diffusion 3 is fuzzy and irrelevant. Jimeng successfully generates a correct right-angled triangle, though the image contains the wrong text. Ideogram2.0 [36] and Midjourney mistakenly focus too much on the word "measuring" in the prompt, thus Ideogram2.0 generates rulers arranged in the shape of a right triangle, and Midjourney presents a dimensional figure irrelevant. Dall-E3 [6] cannot recognize the prompt as a math concept. In the second example of an inscribed circle within an isosceles triangle, the style of the results is similar to the first. In detail, all models generate the correct isosceles triangle but the wrong inscribed circle. Current T2I models are lacking in the ability to generate mathematically relevant images, they cannot accurately understand some mathematical concepts, and it is difficult to generate images that conform to analytic geometric.

Sec. 2.3.1 Math

[Figure 727]

[Figure 728]

Prompt

Prompt

A right-angled triangle features a hypotenuse and two legs, with one angle measuring exactly 90 degrees.

An isosceles triangle with a vertex angle of 120 degrees has an inscribed circle within it.

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

Figure 23: Results on math task. Refer to Section 2.3.1 for detailed discussions.

Score. The results of this experiment are shown in Table 16. Midjourney received higher scores from human evaluations, primarily because the outputs of several models do not effectively grasp the mathematical concepts in the prompts. As a result, the scoring mainly focuses on aspects like aesthetics and realism, with other metrics showing some discrepancies compared to human intuition.

- Table 16: The scoring of generation results by six models on math image design under different

- evaluation systems. Refer to Section 2.3.1 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human FLUX.1 25.84 0.20 4.72 5.00 4.58 Ideogram2.0 20.71 0.19 5.76 5.00 5.42 Dall-E3 23.91 0.20 5.03 4.17 6.25 Midjourney 25.62 0.18 4.72 3.75 7.08 SD3 24.43 0.23 5.05 4.17 3.33 Jimeng 25.50 0.21 4.51 6.25 5.42

2.3.2 Fractal

In this section, we evaluate the models’ ability to generate images within fractal settings, which require understanding complex recursive patterns. These patterns are often used in mathematical and artistic contexts to depict natural phenomena like coastlines, snowflakes, and more.

For the first experiment, we prompted the models to generate a Mandelbrot set. FLUX.1 and Midjourney produced visually appealing fractals with detailed recursive structures. However, Dall-E3 and Stable Diffusion 3 struggled with the intricacy of the pattern, resulting in less accurate representations. In the second experiment involving the Sierpinski triangle, FLUX.1 and Jimeng successfully captured the recursive nature of the fractal, accurately depicting the triangular subdivisions. Ideogram2.0 misinterpreted the prompt, generating a series of disjointed triangles, while Dall-E3 created a pattern resembling the Sierpinski triangle but lacking precise detail. Overall, the experiments reveal that while some models can generate fractal images, consistency and accuracy vary. This suggests that improvements in understanding recursive algorithms might enhance their performance in this domain.

- Score. The results of this experiment are shown in the Table 17. It can be observed that FLUX.1 performed the best in this experiment. The Aesthetic Score aligns more closely with human intuitive perception, while the GPT results show a significant difference from human perception.

Table 17: The scoring of generation results by six models on fractal image design under different

- evaluation systems. Refer to Section 2.3.2 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 24.93 0.25 6.09 5.42 7.92 Ideogram2.0 23.46 0.25 5.73 6.25 5.84 Dall-E3 28.29 0.24 5.63 5.00 4.59 Midjourney 25.54 0.21 6.01 4.59 6.67 SD3 25.90 0.23 5.41 6.25 2.50 Jimeng 19.78 0.21 5.44 7.92 7.09

###### 2.3.3 Medical

In this experiment, we tested the ability of T2I models to generate medical images [1]. For the first prompt, we asked the models to generate an X-ray image capturing a frontal view of the chest. We found that Ideogram2.0, Jimeng, and Stable Diffusion 3 did not produce accurate X-ray images, as indicated by the color and texture of their outputs. Additionally, these three models generated chaotic representations of the shoulder joints, and Ideogram2.0 produced an incorrect morphology of the lungs. Midjourney generated an image that resembled an X-ray, but the structures of the heart and liver were significantly flawed. Dall-E3 and FLUX.1 performed the best, producing images with an

Sec. 2.3.2 Fractal

[Figure 756]

[Figure 757]

Prompt

Prompt

A Sierpinski Triangle, a fractal pattern, is formed by recursively removing equilateral triangles from a larger triangle.

The Mandelbrot Set is a complex fractal, known for its infinite complexity and self-similarity at different scales.

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

- Figure 24: Results on fractal task. Refer to Section 2.3.2 for detailed discussions.

[Figure 785]

FLUX.1

Midjourney

[Figure 786]

Stable Diffusion 3 Jimeng

[Figure 787]

[Figure 788]

Prompt

Ideogram2.0

[Figure 789]

DALLE3

[Figure 790]

Prompt

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

FLUX.1

Midjourney

[Figure 798]

Stable Diffusion 3 Jimeng

[Figure 799]

[Figure 800]

Ideogram2.0

[Figure 801]

DALLE3

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

This X-ray image captures a frontal view of the chest, demonstrating clear lung fields without evidence of

infiltrates or pleural effusions. The heart silhouette is

normal in size and contour. The bony thorax shows no acute fractures or abnormalities. The diaphragm appears wellpositioned, with no evidence of subdiaphragmatic air or pathology. Overall, the image suggests a healthy thoracic structure without acute findings.

A high-resolution CT scan displaying a detailed crosss e c t i o n ( t r a n s v e r s e

section/plane) of a human

brain, highlighting the intricate structures of the cerebral cortex and the ventricles filled with cerebrospinal fluid.

[Figure 811]

[Figure 812]

Sec. 2.3.3 Medical

[Figure 813]

[Figure 814]

[Figure 815]

- Figure 25: Results on medical task. Refer to Section 2.3.3 for detailed discussions.

almost correct morphology of bones and organs, with only minor inaccuracies in the structure of the shoulder joints.

In the second prompt, we asked the models to generate a CT scan displaying a detailed cross-section of a human brain. Ideogram2.0 and Dall-E3’s outputs did not resemble a CT scan, and the brain structures were incorrect. Jimeng failed to generate a cross-section of the brain, and the organs in the images appeared very chaotic. Midjourney’s output contained mistakes in the locations of the cerebrum and cerebellum, and the internal structures of the brain were disorganized. Stable Diffusion 3’s result was not realistic enough; the brain’s edge were overly defined, the proportion of black areas representing cerebrospinal fluid was too small compared to a real brain, and the real brain does not have such notches at the back of the head. Only FLUX.1 produced a result closest to a real brain CT scan, leading us to suspect that FLUX.1’s training data may include a certain proportion of high-quality medical images.

- Score. The results of this experiment are shown in the Table 18. Except for the CLIPScore, FLUX.1 performed the best in all other scores.

- Table 18: The scoring of generation results by six models on medical image generation under different

- evaluation systems. Refer to Section 2.3.3 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human FLUX.1 23.66 0.22 5.73 6.66 7.78 Ideogram2.0 24.38 0.22 4.63 5.28 6.39 Dall-E3 27.64 0.21 4.68 6.11 5.00 Midjourney 26.42 0.22 5.10 5.28 5.84 SD3 28.12 0.19 5.10 4.17 6.66 Jimeng 27.16 0.22 4.82 4.17 4.17

2.3.4 3D Point Cloud

In Figure 26, we investigate the models’ capacities to generate images in the form of 3D point cloud. For the first example of an airplane, FLUX.1, Dall-E3 and Midjourney successfully generate point cloud representations, while the outputs of Dall-E3 and Midjourney are imperfect due to the points outside the airplane. Ideogram2.0, Stable Diffusion 3 and Jimeng fail to accurately render the point cloud, with Ideogram2.0 even producing an image more like 3D mesh. In the second example of a chair, FLUX.1 again succeeds in producing a correct point cloud representation. Additionally, Ideogram2.0 also generates an image of a chair in point cloud form, but of poor quality. Stable Diffusion 3 misunderstands point cloud and generates a chair painted with dots, similar to its output in the first example. Dall-E3, Midjourney and Jimeng misinterpret point cloud as well, generating images of a chair constructed from spherical shapes. Overall, FLUX.1 is the only model to perform well in both cases. It is worth mentioning that the objects and prompts are well-designed. We observed that the models struggle to generate accurate point clouds for certain objects, indicating that additional training may be necessary for tasks involving point cloud generation.

Score. The results of this experiment are shown in the table. In this experiment, the performance of the models is similar, with FLUX.1, Dall-E3, and Jimeng performing slightly better.

Table 19: The scoring of generation results by six models on point cloud image design under different

- evaluation systems. Refer to Section 2.3.4 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 32.24 0.26 5.30 5.00 7.50 Ideogram2.0 33.78 0.30 5.95 6.25 6.67 Dall-E3 32.18 0.31 5.94 7.50 7.50 Midjourney 29.73 0.25 5.70 6.66 6.67 SD3 32.93 0.28 5.76 6.66 7.08 Jimeng 32.65 0.31 6.30 5.00 7.50

Sec. 2.3.4 3D Point Cloud

[Figure 816]

Prompt

[Figure 817]

Prompt

The image is a point cloud rendering of a chair shape. It consists of numerous spherical dots.

The image is a point cloud rendering of an airplane shape. It consists of numerous spherical dots, featuring a gradient color scheme transitioning from pink and purple in the top left to green and blue in the bottom right, creating a colorful and abstract effect.

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

Figure 26: Results on 3D point cloud task. Refer to Section 2.3.4 for detailed discussions.

###### 2.3.5 3D Mesh

The synthesis of high-quality 3D assets from textual or visual inputs has become a central objective in modern generative modeling [29, 41, 34, 15]. In Figure 27, we test the models’ ability to generate 3D mesh representations. It is observed that all models can generate images that seem like 3D mesh form. In the first example of a human head, FLUX.1, Ideogram2.0, Dall-E3 and Stable Diffusion 3 successfully generate 3D mesh-like representations. However, there are also mistakes that only Jimeng achieves "a three-quarter view on the left and a front view on the right" required by prompt. Additional errors including the curved edges of the plane in Midjourney and the textured eyes in Jimeng are incorrect in 3D mesh. In the second example of a car, FLUX.1 and Ideogram2.0 also generate correct 3D mesh representations. However, the outputs of Dall-E3, Midjourney and Stable Diffusion 3 are more like dividing a realistic car by lines, and Jimeng even breaks the car into pieces. Overall, FLUX.1 and Ideogram2.0 have the correct understanding of 3D mesh. However, there remains a distinction between the T2I models generating two-dimensional images and 3D generation models.

Score. The results of this experiment are shown in the Table 20. It can be seen that the model performances are fairly balanced. Overall, the Ideogram2.0 model achieved higher scores.

- Table 20: The scoring of generation results by six models on 3D mesh task under different evaluation

- systems. Refer to Section 2.3.5 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 29.23 0.24 6.00 5.84 7.92 Ideogram2.0 32.64 0.27 5.96 5.84 8.33 Dall-E3 32.98 0.26 5.15 5.00 7.08 Midjourney 32.36 0.24 5.27 6.25 6.25 SD3 31.22 0.24 5.37 7.50 7.92 Jimeng 29.87 0.25 5.55 6.66 8.33

Sec. 2.3.5 3D Mesh

[Figure 846]

[Figure 847]

Prompt

Prompt

Present in this image is a mesh-rendered model of a car, illustrated from a side view. The mesh is composed of a complex network of polygons that meticulously outline the car's sleek body, wheels, and windows. The model has a realistic appearance, with precise details like headlights, grille, and mirrors. Neutral lighting is used to emphasize the clean lines and intricate geometry without applying any texture or color. The background is a solid dark gray to keep the

This image features a mesh-rendered model of a human head and upper shoulders, displayed from two angles: a three-quarter view on the left and a front view on the right. The mesh comprises a network of polygons, primarily quads, creating a smooth, detailed surface over the anatomical structure. The model exhibits a clean and realistic anatomy, with well-defined facial features such as eyes, nose, mouth, and ears, and shows the contours and musculature of the neck and shoulders. The lighting is neutral, highlighting the geometric precision of the mesh without any texture or color. The background is a solid dark gray, which keeps the focus entirely on the 3D model.

emphasis on the 3D model.

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

Figure 27: Results on 3D mesh task. Refer to Section 2.3.5 for detailed discussions.

###### 2.3.6 Chemistry

In Figures 28, we assess the models’ capabilities in generating scientifically accurate images within the domain of chemistry. We evaluate the models’ understanding of chemistry by prompting them to

generate a correct representation of a benzene molecule, but none of the models succeed. Overall, the T2I models struggle to generate scientifically accurate images in the field of chemistry, failing to meet the necessary scientific standards.

Score. The results of this task are shown in the Table 21. Ideogram2.0 produced the best outputs in this task. However, the scores from several evaluation systems differ from human intuition, likely because this task requires a certain level of chemistry knowledge, making accurate evaluation more demanding.

- Table 21: The scoring of generation results by six models on chemistry tasks under different evaluation

- systems. Refer to Section 2.3.6 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 29.38 0.27 5.45 2.50 6.67 Ideogram2.0 27.42 0.27 5.09 5.00 10.00 Dall-E3 28.49 0.18 3.54 2.50 8.33 Midjourney 28.65 0.20 4.66 7.50 6.67 SD3 25.52 0.16 4.13 5.83 5.00 Jimeng 26.53 0.21 5.04 2.50 4.17

###### Sec. 2.3.6 Chemistry

[Figure 877]

Prompt

Generate a detailed image of a benzene molecule, including its hexagonal ring structure with alternating double bonds. Label each carbon atom and hydrogen atom clearly. Include a

caption describing its aromatic properties.

[Figure 878]

[Figure 879]

[Figure 880]

Ideogram2.0 DALLE3

FLUX.1

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

Jimeng

Midjourney Stable Diffusion 3

[Figure 887]

[Figure 888]

[Figure 889]

Figure 28: Results on chemistry task. Refer to Section 2.3.6 for detailed discussions.

###### 2.3.7 Biology

- Figure 29 focuses on their grasp of biology, specifically through the generation of plant cell and animal cell images. In the first example, FLUX.1, Dall-E3, and Jimeng attempt to depict plant cell structures. However, FLUX.1 erroneously includes leaves inside the cell, while Dall-E3 incorporates an orange slice incorrectly. Meanwhile, Ideogram2.0 and Stable Diffusion 3 offer microscopic views of plant cells, but Midjourney seems to lack sufficient biological knowledge. The results of the second example are similarly unsatisfactory.

Score. The results of this experiment are shown in Table 22. FLUX.1 and Dall-E3 performed relatively well in this task; however, only the CLIPScore results aligned with human intuitive perception.

- Table 22: The scoring of generation results by six models on biology under different evaluation

- systems. Refer to Section 2.3.7 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 30.54 0.27 5.77 4.58 9.17 Ideogram2.0 26.10 0.22 5.78 5.42 6.25 Dall-E3 27.58 0.26 5.56 6.25 7.50 Midjourney 21.99 0.23 5.79 7.08 5.00 SD3 26.85 0.28 5.99 3.75 5.84 Jimeng 23.14 0.25 5.82 7.50 7.92

Sec. 2.3.7 Biology

[Figure 890]

[Figure 891]

Prompt

Prompt

Generate a detailed 3D visualization of a plant cell. Generate a detailed 3D visualization of an animal cell.

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

Figure 29: Results on biology task. Refer to Section 2.3.7 for detailed discussions.

###### 2.3.8 Robotics and Simulation Tasks

Tool-Usage Grounding In Robotics, the model needs to recognize the area to grasp or manipulate [23]. To test whether T2I models have the tool-usage grounding ability, we prompt them using two examples. For the first prompt, as depicted in the left subplot of Figure 30, only Ideogram2.0 can generate the hammer image with the grasping area marked with a bounding box. Dall-E3 and FLUX.1 also generate the bounding box in the image, but it’s placed in the wrong area. For the second example,

Stable Diffusion 3, Midjourney, Dall-E3 and Ideagram2.0 all generate the bounding box in the image, but only Ideogram2.0 places it on the grasping area.

Simulation environment Lots of robotic research is conducted in the simulated environment. Thus it’s important to test T2I models’ ability to produce the images in the simulation environment domain. Here we give two simulations, one is a robot navigating, and the other is a dexterous hand. The results are shown in the right subplot of Figure 31. For the first simulation, only the images from Ideogram2.0 and FLUX.1 look like the rendering image of the simulation environment. For the second simulation, only FLUX.1 correctly generated the dexterous hand.

Sec. 2.3.8 Robotics and Simulation Tasks

[Figure 920]

[Figure 921]

Prompt

Prompt

An image of a hammer, mark the area for grasping with a bounding box.

An image of a cup, mark the area for grasping with a

bounding box.

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

- Figure 30: Results on robotics and simulation task. Refer to Section 2.3.8 for detailed discussions.

Score. The results of this experiment are shown in the Table 23. In this experiment, FLUX.1 achieved a relatively high score, and GPT-4o’s evaluation results were more consistent with human intuition.

###### 2.3.9 Autonomous Driving

For autonomous driving model development, synthetic data plays an important role in solving the corner cases. For example, in the real world, it is hard to collect a large amount of autonomous driving data under extreme weather. Utilizing the T2I model, we can synthesize these hard-to-collect data efficiently. Here we prompt these T2I models to generate the road scene under rainy weather and foggy weather respectively.

The results are shown in Figure 32. For the road scene under rainy weather, all six models perform well, while images generated by Dall-E3 and Jimeng are not as realistic as the others. For the road

- Table 23: The scoring of generation results by six models on robotics and simulation tasks under

- different evaluation systems. Refer to Section 2.3.8 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 30.93 0.28 5.41 7.09 9.17 Ideogram2.0 30.59 0.27 5.31 5.83 8.75 Dall-E3 30.94 0.30 5.32 4.58 7.92 Midjourney 29.05 0.29 5.71 5.63 6.67 SD3 28.11 0.27 5.34 4.79 6.04 Jimeng 30.10 0.28 5.81 5.63 7.08

Sec. 2.3.8 Robotics and Simulation Tasks

[Figure 951]

[Figure 952]

Prompt

Prompt

The image shows a robot in a simulated environment. The robot is composed of orange and blue segments, designed for navigating obstacles. In the foreground, there are various barriers and hurdles of different shapes and sizes. The background features a light blue and white checkered pattern, emphasizing the virtual nature of the simulation.

The image depicts a robotic hand in a simulation environment. The hand is sleek and white, holding a metallic, pen-like object. It is attached to a cylindrical component. The background is a simple

gray gradient, highlighting the futuristic design of

the robotic hand.

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

- Figure 31: Results on robotics and simulation task. Refer to Section 2.3.8 for detailed discussions.

scene under foggy weather, all six models perform well, except the Jimeng-generated image contains commonsense artifacts – pedestrians won’t walk in the middle of the road.

Score. The results of this experiment are shown in Table 24. Stable Diffusion 3 and Jimeng scored higher in other evaluation systems, but there is a significant gap compared to human ratings. This may be due to the fact that the images in this experiment often lack realism in multiple details, which the evaluation systems fail to accurately recognize.

- Table 24: The scoring of generation results by six models on autonomous driving task under different

- evaluation systems. Refer to Section 2.3.9 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 29.25 0.26 5.45 6.67 8.34 Ideogram2.0 28.42 0.29 5.90 6.67 8.34 Dall-E3 26.75 0.26 5.35 6.67 8.33 Midjourney 28.01 0.26 5.27 6.25 9.17 SD3 30.21 0.26 5.89 8.33 7.50 Jimeng 29.17 0.30 6.02 7.08 6.67

Sec. 2.3.9 Autonomous Driving

[Figure 982]

[Figure 983]

Prompt

Prompt

Navigating through the city streets, the dense fog obscures the view, making it a challenge to discern the traffic lights just a few meters ahead.

The image shows a street at night during heavy rain. The road is flooded, and streetlights illuminate the scene, reflecting off the wet surface. Trees and sidewalks line the street, contributing to the urban setting.

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

- Figure 32: Results on autonomous drive corner cases task. Refer to Section 2.3.9 for detailed discussions.

###### 2.4 Challenging Scenario Generation

The application of T2I models is expanding, particularly in generating images for more challenging scenarios. In this section, we carefully curate a set of complex prompts to evaluate the models’ ability to handle intricate settings.

- In Section 2.4.1-2.4.2, we prompt the models to generate images based on traditional visual task settings. Section 2.4.4 examines the models’ ability to generate images containing dense text and pictures. In Section 2.4.3, we assess the models’ understanding of different languages from various regions. Finally, in Section 2.4.5, we input emojis into the models to evaluate their ability to organize diverse elements and piece together the logical relationships between multiple emojis in an image. In

- Section 2.4.6, we explored the ability of T2I models to generate images in irrational scenarios. In
- Section 2.4.7, we investigated whether T2I models could generate corresponding images based on answers when given simple questions as input. In Section 2.4.8, we experimented with the ability of T2I models to generate images with watermarks. In Section 2.4.9, we explored the model’s ability to generate low-quality images, which indicates whether the model’s training data contains low-quality data. In Section 2.4.10, we systematically explored the model’s ability to generate images containing multiple sub-images. In Section 2.4.11, we specifically studied the model’s ability to generate text within images.

###### 2.4.1 Image with Mark

Object detection is a common task in the field of computer vision [48, 16, 58, 51, 53]. In this paper, we design the image with mark task to explore whether text-to-image models can be applied to computer vision tasks. In this task, we examine the models’ ability to generate images resembling the output of computer vision algorithms. The results are presented in Figure 33. In the first prompt, we asked the models to highlight a coffee mug with a bounding box, and in the second prompt, to highlight an apple in the same manner. We observed that only FLUX.1 [20], Ideogram2.0 [36], and Dall-E3 [6] successfully completed the first task, while only FLUX.1 and Ideogram2.0 correctly accomplished the second task.

Sec. 2.4.1 Image with Mark

[Figure 1011]

[Figure 1012]

Prompt

Prompt

On a tidy table, there is a coffee mug and a vase, with the coffee mug identified by the algorithm and highlighted with a red bounding box.

On a tidy table, there is an apple and a banana, with the apple identified and highlighted using a red bounding box.

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

Figure 33: Results on image with mark task. Refer to Section 2.4.1 for detailed discussions.

- Score. The results of this experiment are shown in the Table 25. It can be observed that the outputs of FLUX.1 and Ideogram2.0 are relatively good. The same conclusion can be drawn from the other scores, except for the CLIPScore.

- Table 25: The scoring of generation results by six models on image with mark under different

- evaluation systems. Refer to Section 2.4.1 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 29.66 0.25 5.13 8.74 10.00 Ideogram2.0 32.27 0.27 5.94 4.59 10.00 Dall-E3 36.37 0.27 4.94 7.90 9.17 Midjourney 31.12 0.27 4.90 7.07 6.67 SD3 34.43 0.26 5.19 4.59 7.50 Jimeng 32.04 0.26 4.71 5.00 8.33

###### 2.4.2 Set of Mark

Set of mark is also a traditional task in computer vision [82]. As depicted in the figure 34, FLUX accurately generates red bounding boxes around the specified coffee cup and vase, although the serial numbers are incorrect. In contrast, Midjourney fails to produce regular rectangles, while Stable Diffusion 3 [64] misplaces the bounding box, entirely missing the vase, and also assigns incorrect serial numbers. Jimeng, however, successfully frames all required objects. These findings indicate that while FLUX.1 exhibits minor labeling inaccuracies, it holds promise for improving object detection and grounding tasks in subsequent processes.

Sec. 2.4.2 Set of Mark

[Figure 1044]

[Figure 1045]

Prompt

Prompt

On a tidy table, there is an apple and a banana, with the apple and banana identified and highlighted using a red

On a tidy table, there is a coffee mug and a vase, with the coffee mug and vase identified by the algorithm and highlighted with a red bounding box. The vase is labeled

bounding box. The apple is labeled number 1, and the

banana is labeled number 2.

number 1, and the coffee mug is labeled number 2.

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

Figure 34: Results on set of mark task. Refer to Section 2.4.2 for detailed discussions.

###### Score. The results of this experiment are shown in the Table 26. From a human intuitive perspective, the output of FLUX.1 is the best. Only the results from HPSv2 align closely with human intuition, while the scores from the other metrics show some discrepancies.

Table 26: The scoring of generation results by six models on set of mark task under different evaluation systems. Refer to Section 2.4.2 for detailed discussions.

Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 33.72 0.27 5.24 6.94 9.22 Ideogram2.0 34.93 0.25 4.87 6.94 8.89 Dall-E3 33.30 0.24 5.20 3.61 7.50 Midjourney 32.04 0.25 5.80 6.95 5.56 SD3 37.93 0.25 5.18 5.83 7.22 Jimeng 35.27 0.24 5.19 3.89 5.83

- 2.4.3 Multilingual

As shown in Figures 35-37, the multilingual capabilities of T2I models exhibit significant variations across different languages [55, 78]. Models like Ideogram2.0 and Dall-E3 demonstrate strong performance when processing prompts in English, Spanish, and French. However, a notable limitation remains: FLUX.1 performs poorly with Chinese prompts, while FLUX.1, Midjourney, and Stable Diffusion 3 show subpar results with Japanese prompts. This may be attributed to their use of text encoders that support only English, highlighting a crucial area for improvement in the development of more universally robust multilingual T2I models.

Sec. 2.4.3 Multilingual

[Figure 1075]

Prompt

A regal swan glides gracefully across the surface of a tranquil lake, its snowy white feathers ruffled by the gentle breeze.

[Figure 1076]

[Figure 1077]

[Figure 1078]

Ideogram2.0 DALLE3

FLUX.1

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

Midjourney Stable Diffusion 3 Jimeng

[Figure 1085]

[Figure 1086]

[Figure 1087]

Figure 35: Results on multilingual task. Refer to Section 2.4.3 for detailed discussions.

###### Score. The results of this experiment are shown in the Table 27. The results for CLIPScore and Aesthetic Score are relatively consistent with human ratings.

Sec. 2.4.3 Multilingual

[Figure 1088]

[Figure 1089]

Prompt

Prompt

French: Un cygne royal glisse gracieusement àla surface d’un lac tranquille, ses plumes blanches comme neige ébouriffées par la douce brise.

Chinese: 一只高贵的天鹅优雅地滑过宁静的湖面，雪白 的羽毛在微风的吹拂下摇曳。

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1114]

[Figure 1115]

[Figure 1116]

##### Illegal

- Figure 36: Results on multilingual task. Refer to Section 2.4.3 for detailed discussions.

[Figure 1117]

FLUX.1

Midjourney

[Figure 1118]

Stable Diffusion 3 Jimeng

[Figure 1119]

[Figure 1120]

Prompt

Ideogram2.0

[Figure 1121]

DALLE3

[Figure 1122]

Prompt

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

FLUX.1

Midjourney

[Figure 1130]

Stable Diffusion 3 Jimeng

[Figure 1131]

[Figure 1132]

Ideogram2.0

[Figure 1133]

DALLE3

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

Spanish: Un majestuoso cisne se desliza con gracia sobre la superficie de un lago tranquilo, sus plumas blancas como la nieve se agitan con la suave brisa.

Japanese:威厳のある白鳥が静かな湖の表面を優雅に滑 り、雪のように白い羽が穏やかな風に揺れています。

[Figure 1141]

[Figure 1142]

Sec. 2.4.3 Multilingual

Illegal Illegal

[Figure 1143]

[Figure 1144]

[Figure 1145]

- Figure 37: Results on multilingual task. Refer to Section 2.4.3 for detailed discussions.

- Table 27: The scoring of generation results by six models on multilingual under different evaluation

- systems. Refer to Section 2.4.3 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 23.40 0.28 5.62 6.94 8.33 Ideogram2.0 25.75 0.29 6.06 7.50 10.00 Dall-E3 23.29 0.30 5.98 8.05 8.33 Midjourney 23.26 0.30 6.16 6.94 7.78 SD3 25.19 0.27 5.93 5.55 8.33 Jimeng 21.77 0.29 5.88 8.75 9.17

###### 2.4.4 Dense OCR

Figure 38 and Figure 39 present an evaluation of the dense OCR capabilities of various T2I models. When generating posters with an English corpus, FLUX.1 successfully captures the overall content based on the given requirements but exhibits some spelling errors in the generated text. In contrast, Jimeng, Dall-E3, Ideogram2.0, and Stable Diffusion 3 focus primarily on the title, failing to generate additional textual content from the provided prompts. Notably, Stable Diffusion 3 introduces considerable spelling errors. Furthermore, none of these T2I models effectively recognize or generate Chinese text when tasked with poster generation using a Chinese corpus, highlighting a significant limitation in handling Chinese OCR. For academic paper poster generation, FLUX.1 and Ideogram2.0 demonstrate the ability to generate most of the textual content with a clear and legible appearance. However, Dall-E3, Stable Diffusion 3, Jimeng, and Midjourney struggle with text clarity and exhibit prominent spelling errors, indicating limitations in generating accurate and coherent textual content in this context.

Sec. 2.4.4 Dense OCR

[Figure 1146]

[Figure 1147]

Prompt

Prompt

Product Poster Layout: Title "Summer Sale Extravaganza!" Promotional Information

Product Poster Layout: Title "清仓大甩卖!" Promotional Information " 全部商品半价起，先到先 得"

"Up to 50% Off on Selected Items!" Poster Content "Summer Sale Extravaganza!" Don't miss out on our biggest sale of the season! Enjoy up to 50% off on a wide range of products, from stylish apparel to must-have gadgets. Shop now and get free shipping on all orders over $50. Hurry, limited time offer!" Call to Action: "Shop Now at www.example.com or Visit Our Store!"

Poster Content "走过路过， 不要错过！福德皮货市场 换季清仓大甩卖，买到即 是赚到！"

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

- Figure 38: Results on denseocr task. Refer to Section 2.4.4 for detailed discussions.

###### Sec. 2.4.4 Dense OCR

[Figure 1176]

Prompt

Poster Title: "Unleashing the Power of GPT: Revolutionizing Language Understanding" Layout Description: Section Title: “What is GPT?” GPT (Generative Pre-trained Transformer) is a state-of-the-art language model that uses deep learning to understand and generate human-like text. Developed by OpenAI, GPT is based on the Transformer architecture, which enables it to process large amounts of data and generate coherent, contextually relevant responses. The model has been fine-tuned across a wide range of tasks, making it highly versatile and powerful in various applications. Section Title: “Key Features” Natural Language Understanding: GPT can comprehend and analyze complex text inputs, making it capable of performing tasks such as summarization, question answering, and sentiment analysis. Text Generation: The model excels in generating high-quality, coherent text that aligns with the context provided, whether it's a creative story, a technical report, or conversational dialogue. Multilingual Capabilities: GPT supports multiple languages, allowing for accurate translations and multilingual text generation, making it a global tool for communication. Scalability: With larger model sizes, GPT's performance improves, handling more complex tasks and producing more nuanced outputs, which makes it suitable for diverse, high-demand applications. Section Title: “Applications” Customer Support: GPT can be deployed as a chatbot to handle customer inquiries with human-like responses, improving efficiency and customer satisfaction. Content Creation: The model assists in generating content for blogs, articles, social media posts, and even creative writing, saving time and enhancing productivity. Translation: GPT’s multilingual capabilities allow it to accurately translate text between different languages, considering context to ensure the translation is meaningful and precise. Code Generation: Developers can use

GPT to generate code snippets, help with debugging, and even create complex algorithms, thus accelerating

software development processes.

[Figure 1177]

[Figure 1178]

[Figure 1179]

Ideogram2.0 DALLE3

FLUX.1

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

Midjourney Stable Diffusion 3 Jimeng

[Figure 1186]

[Figure 1187]

[Figure 1188]

- Figure 39: Results on denseocr task. Refer to Section 2.4.4 for detailed discussions.

- Score. The results of this experiment are shown in the Table 28. From a human intuitive perspective, Ideogram2.0 performed the best in this experiment, with only the GPT-4o results aligning closely with human perception.

###### 2.4.5 Emoji

In Figures 40-41, we investigate the models’ ability to comprehend emojis. We observe that FLUX.1 and Ideogram2.0 attempt to construct stories from combinations of emojis, but they tend to focus on certain emojis while ignoring others. For example, FLUX.1 disregards the construction site emoji in the second prompt of Figure 40 and the tree emoji in the first prompt of Figure 41. Ideogram2.0 performs better than FLUX.1 by considering nearly all emojis and their logical relationships. For instance, as shown in the left subplot of Figure 41, Ideogram2.0 integrates the desert and tree emojis into an oasis in the first example and also accurately understands the story conveyed by the emojis in

- Table 28: The scoring of generation results by six models on dense OCR under different evaluation

- systems. Refer to Section 2.4.4 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 32.22 0.25 4.22 1.67 7.22 Ideogram2.0 29.27 0.25 4.64 5.83 7.78 Dall-E3 31.07 0.25 4.73 5.56 5.83 Midjourney 28.98 0.22 4.10 3.33 6.94 SD3 33.01 0.25 4.60 3.89 5.83 Jimeng 23.89 0.21 3.86 4.17 5.56

the second example. Dall-E3 approaches the inputs differently, merging all emojis into its output and rendering it in a comic style. However, it sometimes loses logical coherence, as seen in the first example of the left subplot in Figure 41, where it depicts a tree standing alone in the desert. Midjourney, Jimeng, and Stable Diffusion 3 struggle to handle this type of task correctly, with Jimeng even considering this type of prompt illegal. Notably, Ideogram2.0 outperforms the others across all models, particularly in managing complex, multi-emoji inputs and accurately interpreting the stories expressed by the emojis.

Sec. 2.4.5 Emoji

[Figure 1189]

[Figure 1190]

Prompt

Prompt

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1226]

[Figure 1227]

## Illegal Illegal

- Figure 40: Results on emoji task. Refer to Section 2.4.5 for detailed discussions.

Sec. 2.4.5 Emoji

[Figure 1228]

[Figure 1229]

Prompt

Prompt

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1266]

[Figure 1267]

# Illegal Illegal

- Figure 41: Results on emoji task. Refer to Section 2.4.5 for detailed discussions.

- Score. The results of this experiment are shown in the Table 29. In this task, the outputs of FLUX.1, Ideogram2.0, and Dall-E3 are all relatively good.

- Table 29: The scoring of generation results by six models on emoji task under different evaluation

- systems. Refer to Section 2.4.5 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 19.51 0.23 6.98 5.83 8.89 Ideogram2.0 19.45 0.23 6.18 6.39 9.72 Dall-E3 21.19 0.21 6.15 6.94 8.05 Midjourney 15.47 0.20 5.52 3.89 5.00 SD3 23.82 0.23 5.89 4.17 5.83

###### 2.4.6 Irrational Scene Generation

In Figure 42-43, we evaluated the ability of these models to generate irrational scenes. In the first instance, we instructed the models to generate the word "RED" written in blue on yellow blocks. FLUX.1, Ideogram2.0, and Dall-E3 completed the task perfectly. Midjourney generated a red background, Jimeng used black text, and Stable Diffusion 3 used red text, each with some flaws. In the second and third instances, we had the models generate anomalous scenes, including objects

with unusual colors and materials. Among all the models, Dall-E3 performed the best, perfectly generating scenes according to the text description. When generating objects with anomalous colors, FLUX.1, Stable Diffusion 3, and Jimeng missed parts of the text description, resulting in flawed outputs. Additionally, FLUX.1, Ideogram2.0, and Jimeng failed to correctly interpret the metaphor in the text when generating objects with anomalous materials, leading to incorrect outputs.

- Score. The results of this experiment are shown in the Table 30. In this task, the outputs of FLUX.1, Ideogram2.0, and Dall-E3 are relatively good; however, the results for CLIPScore and GPT-4o differ significantly from human evaluations.

- Table 30: The scoring of generation results by six models on irrational scene generation under

- different evaluation systems. Refer to Section 2.4.6 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 30.06 0.28 6.02 6.39 8.33 Ideogram2.0 28.47 0.28 5.93 6.11 8.06 Dall-E3 31.89 0.26 6.17 5.83 8.61 Midjourney 30.57 0.27 5.90 4.45 7.22 SD3 30.66 0.27 6.08 6.94 6.94 Jimeng 32.47 0.27 5.88 6.94 6.94

###### Sec. 2.4.6 Irrational Scene Generation

[Figure 1268]

[Figure 1269]

Prompt

Prompt

###### Word 'RED' written in blue on yellow blocks. Photo of a realistic landscape where the sky is vividly green, the

grass is a deep red, and the sun is a dark shade of blue.

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

- Figure 42: Results on irrational scene generation task. Refer to Section 2.4.6 for detailed discussions.

###### Sec. 2.4.6 Irrational Scene Generation

[Figure 1302]

###### Prompt

Photo of a city where the buildings appear to be melting like wax, with structures drooping and stretching in unusual directions. The streets should be filled with pools of liquid metal.

[Figure 1303]

[Figure 1304]

[Figure 1305]

Ideogram2.0 DALLE3

FLUX.1

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

Midjourney Stable Diffusion 3 Jimeng

[Figure 1312]

[Figure 1313]

[Figure 1314]

- Figure 43: Results on irrational scene generation task. Refer to Section 2.4.6 for detailed discussions.

###### 2.4.7 LLM QA

As the T2I models get stronger and more versatile, it seems that they have the ability to develop towards fundamental models. In Figure 44, we illustrate two examples to assess the LLM questionanswering (QA) capabilities of models. This evaluation involves both understanding the questions and generating accurate visual representations of the answers. In the first example, the correct answer to the question about the celestial body orbiting the Earth is the Sun. Models such as Ideogram2.0, Dall-E3, and Jimeng correctly interpret the question and generate images representing the Sun. Although these images may not be scientifically precise representations of the Sun, they are easily recognizable. Conversely, models such as FLUX.1, Midjourney, and Stable Diffusion 3 misunderstand the prompt and produce irrelevant images. In the second example, which involves a simple mathematical query, the desired output is an image featuring the number 3.11. Among the models, only Dall-E3 generates an image that approximates the correct answer. The other models produce images that do not correspond to the required representation.

- Score. The results of this experiment are shown in the Table 31. Due to the evaluation system of this experiment requiring both an understanding of the images and a certain level of mathematical and physical knowledge, the results obtained may differ significantly from human intuitive perception.

- Table 31: The scoring of generation results by six models on LLM QA under different evaluation

- systems. Refer to Section 2.4.7 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 24.49 0.23 5.13 5.00 10.00 Ideogram2.0 20.72 0.23 5.32 4.59 7.92 Dall-E3 26.51 0.24 5.71 5.42 7.92 Midjourney 23.53 0.23 5.66 6.67 6.83 SD3 21.75 0.21 5.09 5.71 2.92 Jimeng 23.67 0.25 5.66 5.42 6.67

Sec. 2.4.7 LLM QA

[Figure 1315]

[Figure 1316]

Prompt

Prompt

Which celestial body does the Earth orbit around? Draw this celestial body.

Which is larger, 3.11 or 3.9? Draw the larger number.

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

FLUX.1

[Figure 1322]

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

Midjourney

DALLE3

Midjourney

DALLE3

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

Figure 44: Results on LLM QA task. Refer to Section 2.4.7 for detailed discussions.

###### 2.4.8 Watermark

In Figures 45, we evaluate the models’ capability to correctly generate watermarks in images, which remind that the images are AI-generated for safety purposes. We prompt the models to create images with watermarks placed in different locations and avoid destroying the integrity of the pictures. We use green frames to highlight the right watermarks and red frames to denote watermarks that are either misplaced, contain incorrect text, or are entirely omitted. The results show that none of the models consistently produced correct watermarks across all examples. Notably, the Jimeng model failed to generate accurate watermarks in any task, potentially due to its inherent watermarking system.

- Score. The results of this experiment are shown in the Table 32. The evaluation in this experiment focuses on the accuracy of watermark placement, which requires precise attention to image details. As a result, there is some difference between other evaluation systems and human scoring.

- Table 32: The scoring of generation results by six models on watermark task under different evaluation

- systems. Refer to Section 2.4.8 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 27.37 0.25 5.17 6.67 9.17 Ideogram2.0 27.10 0.28 6.10 6.67 8.33 Dall-E3 27.42 0.26 5.83 5.83 5.42 Midjourney 29.01 0.24 5.66 8.34 8.34 SD3 30.26 0.25 5.73 7.50 8.33 Jimeng 29.34 0.30 6.10 5.00 8.33

###### Sec. 2.4.8 Watermark

[Figure 1348]

[Figure 1349]

[Figure 1350]

Prompt

Prompt

Prompt

A stunning photograph of a leopard prowling through the tall grass of the African savannah. The leopard's sleek body and distinctive spots are clearly visible as it moves gracefully under the golden sunlight. The vast, open

A breathtaking photograph of a clear night sky filled with countless stars, constellations, and a faint view of the Milky Way stretching across the horizon. The dark blue and purple hues of the sky contrast with the sparkling stars. In the bottom-left corner, a semi-transparent watermark reading "AI Generation" is placed, blending softly with the darker part of the image.

A majestic waterfall cascading down rugged cliffs, surrounded by lush greenery and mist rising from the base. The water flows powerfully into a clear pool below, creating a serene and vibrant natural scene. A large, semi-transparent watermark reading "AI Generation" covers the entire image, blending into the background yet remaining prominent across the frame.

landscape stretches into the horizon, with acacia trees in the

background. In the top-right corner, a semi-transparent watermark that reads "AI Generation" is placed, blending softly with the sky.

Created Using: astrophotography, high detail, natural

Created Using: nature photography, realistic water texture, lush environment, large watermark

Created Using: wildlife photography, realistic details, natural lighting, subtle watermark

lighting, subtle watermark

[Figure 1351]

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

[Figure 1393]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen g

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

Figure 45: Results on watermark task, with green frames to highlight the right watermarks and red frames to denote watermarks that are either misplaced, contain incorrect text, or are entirely omitted. Refer to Section 2.4.8 for detailed discussions.

###### 2.4.9 Low Quality

In Figures 46-48, we investigate the models’ ability to generate low-quality images. During training, some models may discard low-quality images, while others may retain them and apply corresponding labels. This task tends to infer the nature of the datasets used by these models. Specifically, if a model can accurately generate low-quality images, it suggests that labeled low-quality data was incorporated during training. We provide prompts including low resolution, distorted colors, disorganization, ugly figures, underexposure, overexposure, excessive noise, incorrect white balance, and accidental photography. Notably, Dall-E3 successfully generates accurate representations for all prompts, except for the example of accidental photograph, where Dall-E3 might misunderstand the prompt "while holding a phone." This suggests that Dall-E3 may have been trained on datasets including low-quality data with corresponding labels. In contrast, other models consistently produced high-quality images to satisfy the key information of prompts.

The following provides a more detailed analysis of each example.

Low Resolution. In the left subplot of Figures 46, FLUX.1, Dall-E3, Midjourney, and Stable Diffusion 3 successfully generate low-resolution images as specified in the prompt. However, Ideogram2.0 produces an image resembling an out-of-focus photograph, while Jimeng generates a high-resolution image, deviating from the intended requirement.

Distorted Colors. In the middle subplot of Figures 46, Dall-E3 and Stable Diffusion 3 produce images with notably distorted colors, while the outputs from the other models exhibit an appealing use of multiple colors which are inconsistent with the prompt of istorted colors.

Disorganization. Among the models, Dall-E3 and Midjourney produce images that best meet the prompt’s criteria in the right subplot of Figures 46,. The images generated by FLUX.1, Stable Diffusion 3, and Jimeng, although present a level of disorganization but are more like well-laid.

Ugly Figures. In the left subplot of Figures 47, Dall-E3 generates the image that most closely aligns with the prompt’s specifications. While Ideogram2.0 also creates an unattractive figure, it includes an

###### Sec. 2.4.9 Low Quality

[Figure 1400]

[Figure 1401]

[Figure 1402]

Prompt

Prompt

Prompt

A blurry, low-resolution image of a school classroom. The

A chaotic and cluttered living room scene with furniture and objects scattered in a disorganized manner. The sofa is askew, books and magazines are piled randomly, and various items like cushions, lamps, and decorations are haphazardly placed. The room feels overwhelming, with no clear focal point as the colors, textures, and objects clash, creating a sense of disorder.

A surreal outdoor landscape photo with severely distorted colors. The scene features a vivid blue sky, white clouds, lush green grass, and a gently flowing stream. However, the colors are highly warped, with the sky shifting to neon hues, the grass glowing in unexpected tones, and the stream reflecting unnatural shades. The clouds appear in vibrant, contrasting colors, adding to the surreal atmosphere. Created Using: surrealism, vivid color contrast, dreamlike distortion, vibrant lighting

desks, chairs, and chalkboard are vaguely recognizable, but the entire scene is intentionally out of focus, creating a hazy, dreamlike effect. The classroom’s muted colors blend together, with the details of posters, books, and student belongings barely discernible. Soft lighting adds to the unclear, faded atmosphere, making it difficult to distinguish specific objects. Created Using: blurred focus, low resolution, muted colors, dreamlike atmosphere

Created Using: chaotic composition, disorganized furniture, cluttered details, clashing textures

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

FLUX.1

Ideogram2.0

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

Midjourney

DALLE3

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1432]

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

Figure 46: Results on low quality task. Refer to Section 2.4.9 for detailed discussions.

unreasonable foreground element. FLUX.1, Midjourney, and Stable Diffusion 3 generate detailed portraits, and Jimeng produces even more intricate and refined images.

Underexposure. In the middle subplot of Figures 47, all models generate images that exhibit underexposure, matching the prompt’s expectations.

Overexposure. Only Dall-E3 successfully produces an overexposed photograph, in the right subplot of Figures 47. The other models fail to accurately catch the concept of overexposure.

Excessive Noise. In the left subplot of Figures 48, the image generated by Dall-E3 contains excessive digital noise and grain. The other models also produce lower-quality images, but Ideogram2.0 stands out for incorporating elaborate special effects into its output.

Incorrect White Balance. In the middle subplot of Figures 48, Jimeng is the only model that misinterprets the concept of incorrect white balance, producing an image with a warm lighting scene rather than the expected effect.

Accidental Photography. In the right subplot of Figures 48, Ideogram2.0 and Midjourney accurately adhere to the prompt, generating images that depict the concept of accidental photography. Dall-E3 and Jimeng, however, place too much emphasis on the detail of "while holding a phone," resulting in images that are inconsistent with the intended outcome.

Score. The results of this experiment are shown in the Table 34. Midjourney achieved the best generation performance in this experiment. Since several models produced similar generation results, the differences in evaluation scores are not significant.

- Table 33: The scoring of generation results by six models on low quality under different evaluation

- systems. Refer to Section 2.4.9 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 26.99 0.25 5.69 6.30 9.07 Ideogram2.0 24.15 0.25 5.65 6.11 9.35 Dall-E3 26.92 0.26 5.68 6.02 9.63 Midjourney 24.08 0.28 5.83 5.46 8.98 SD3 25.38 0.27 5.65 5.83 8.98 Jimeng 27.71 0.26 5.90 5.37 8.24

###### Sec. 2.4.9 Low Quality

[Figure 1444]

[Figure 1445]

[Figure 1446]

Prompt

Prompt

Prompt

A poorly taken photo of a person from a low, awkward angle, creating a distorted perspective that emphasizes unusual proportions. The lighting is harsh and unflattering, resulting in muted skin tones and a tired

A heavily underexposed photograph of a speaker delivering a speech on a dimly lit stage. The speaker is barely visible, shrouded in deep shadows, with only a faint light highlighting their silhouette. The microphone and stage details are lost in darkness, creating a moody

A highly overexposed photograph of a speaker delivering a speech on stage, with a microphone in hand. The speaker’s face and features are partially washed out due to the bright lighting, creating a dreamlike and ethereal effect. The background of the stage is barely visible, as the light floods the entire scene.

appearance. Their expression appears flat, with unfocused

eyes, contributing to the overall unnatural and unrefined look of the image. The composition highlights the challenges of poor angles and lighting in photography.

###### and dramatic atmosphere.

Created Using: low light, deep shadows, moody ambiance, minimal visibility

Created Using: high exposure, soft glow, minimalist details, washed-out colors

Created Using: awkward angle, poor lighting, muted tones, flat expression

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

Midjourney

Midjourney

DALLE3

DALLE3

Midjourney

DALLE3

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

Stable Diffusion 3 Jimeng

[Figure 1482]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

Figure 47: Results on low quality task. Refer to Section 2.4.9 for detailed discussions.

###### Sec. 2.4.9 Low Quality

[Figure 1489]

[Figure 1490]

[Figure 1491]

Prompt

Prompt

Prompt

A photograph of a lively bar with incorrect white balance, where the warm yellow lighting is exaggerated, casting a strong orange tint over the entire scene. The patrons and furniture are bathed in an unnatural glow, with drinks on the counter reflecting distorted colors. The walls and barstools take on a surreal hue, giving the image an otherworldly feel. Created Using: color distortion, warm overtones, unnatural lighting, surreal atmosphere

A blurred, accidental photograph of a street scene, taken unintentionally while holding a phone. The camera is tilted at an odd angle, showing part of the road, a nearby sidewalk, and passing cars. The image is slightly out of focus, with some motion blur due to movement, and random objects like street signs or parked vehicles partially visible. The lighting is natural, capturing the randomness of the moment.

A photograph of a grand, towering building in a bustling city, but with excessive digital noise and grain. The

majestic architecture is overshadowed by the image's rough

texture, giving it a gritty, low-resolution appearance. The details of the building and surrounding cityscape are obscured by the overwhelming noise, adding a distorted, chaotic feeling to the scene. Created Using: excessive noise, low quality, grainy texture, urban architecture

Created Using: accidental framing, motion blur, natural light,

street photography

[Figure 1492]

[Figure 1493]

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

Midjourney

Midjourney

DALLE3

DALLE3

Midjourney

DALLE3

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen g

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

Figure 48: Results on low quality task. Refer to Section 2.4.9 for detailed discussions.

###### 2.4.10 Multi-image

Recent studies [31, 50, 70, 69, 11] have demonstrated that advanced image generation models can produce grid-based compositions containing multiple images within a single output. These grid-based images not only maintain a certain level of contextual consistency across different cells but also extend the applicability of these models to various tasks, such as subject/style consistency, storyboard generation, and logical reasoning. This observation suggests two key insights: first, these models can leverage a substantial amount of contextually coherent training data from existing internet sources during training; second, they exhibit a certain degree of holistic generation capability, conditioned on prompts and contextual information. Motivated by these findings, we further assess the ability of state-of-the-art image generation models in producing contextually consistent grid images, paving the way for the evolution of image generation models toward more general-purpose intelligent agents [59, 61, 10].

Creation and Planning process. In Figure 49, we explore the models’ ability to generate creation and planning processes. It is evident that FLUX.1 and Ideogram2.0 can generate reasonable PPT creation processes and deduce plausible chess game strategies. Dall-E3, on the other hand, can accurately generate the process of creating artwork sketches based on our requirements. These three models effectively present the creative planning process in a grid format.

###### Sec. 2.4.10 Multi-image

###### Creation and Planning Process

[Figure 1536]

Prompt PPT Creation

[Figure 1537]

[Figure 1538]

Prompt Art Creation

Prompt Chessboard

Create a four-panel grid showcasing a sequence of four connected PowerPoint slides, each building on the previous one in a logical progression. The first slide introduces the main topic with a clean, minimalist design, featuring a title and a brief overview of the

Create a four-panel grid showing a sequence of four chessboard positions, each representing a key moment in a chess game, with the positions evolving over time. The first panel shows the starting position of the game, with the pieces set in their initial arrangement. The second panel displays a significant move or series of moves that leads to a more complex position, highlighting the strategic choices made by the players.

Create a four-panel grid showing the step-by-step design process of a modern artistic sculpture. The first

panel shows the initial concept with basic geometric

shapes and a rough structure, focusing on the overall silhouette and composition without details. The second panel adds more artistic elements, with smoother lines, asymmetry, and the beginning of textural details. The third panel refines the design with complex textures and materials, bold color blocks, and gradients, enhancing the sculpture's unique feel and the delicate interplay of light and shadow. The fourth panel presents the final sculpture, fully detailed, showcasing intricate surface treatments,

subject. The second slide dives deeper into the topic,

presenting key points with bullet points, relevant images, or diagrams to support the content. The third slide expands on the details, using charts, graphs, or additional visuals to explain complex concepts, showing a clear progression in the information flow. The fourth slide concludes with a summary of key takeaways, a call to action, or a final thought, reinforcing the main message with a clean, polished design that ties together the entire presentation.

The third panel shows a critical turning point in the

game, where a combination or tactical maneuver changes the course of the match. The fourth panel presents the final position, showcasing the checkmate or conclusion of the game, with all the pieces arranged to reflect the outcome of the match.

complex forms, and realistic lighting, perfectly capturing

the artistic essence and visual impact of the design.

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 1550]

[Figure 1551]

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

[Figure 1578]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

Figure 49: Results on creation and planning process task. Refer to Section 2.4.10 for detailed discussions.

Style Consistency. In Figure 50, we tested the capability of model style consistency. The model was tasked with generating four sub-images within a single image, each containing different objects, and converting them to the same style. FLUX.1 and Ideogram2.0 were able to follow the prompt relatively well. Dall-E3 and Midjourney could distinguish the sub-images and generate parts of the objects but with flaws. The capabilities of Stable Diffusion 3 and Jimeng were weaker, struggling to produce multi-image outputs and correctly render the objects. Additionally, in Figure 51, we designed tasks to generate app icons in a unified style and to create couple avatars. These tasks tested the models’ ability to generate images with consistent styles. The results show that while the models can create images with a consistent style, some of the icons in the first task lack realism, with Ideogram2.0 performing the best. In the second task, all models successfully generated two grids, with FLUX.1, Ideogram2.0, and Midjourney producing the highest-quality images.

###### Sec. 2.4.10 Multi-image

Style Consistency

[Figure 1585]

[Figure 1586]

Prompt

Prompt

Four celestial objects: top left, a futuristic toilet glowing with blue-purple hues and stars. Top right, French fries floating with a cosmic aura. Bottom left, a hamburger with a glowing bun, stars in toppings. Bottom right, a vintage phone with stardust and planets. Dark cosmic background with soft starlight and nebulae. Ethereal glow, starry

Four objects in Qinghua Ci porcelain style: top left, a

traditional vase with intricate blue floral patterns; top right,

a computer set with blue floral and landscape designs; bottom left, futuristic VR glasses with blue dragon motifs; bottom right, a skyscraper with a porcelain-like exterior. Minimal background highlights contrast between tradition and modernity.

patterns, futuristic, cosmic

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1593]

[Figure 1594]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1595]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

[Figure 1602]

[Figure 1603]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1604]

[Figure 1605]

[Figure 1606]

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

[Figure 1611]

[Figure 1612]

[Figure 1613]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

- Figure 50: Results on style consistency task. Refer to Section 2.4.10 for detailed discussions.

[Figure 1618]

FLUX.1

Midjourney

[Figure 1619]

Stable Diffusion 3 Jimeng

[Figure 1620]

[Figure 1621]

Prompt Icon Design

Ideogram2.0

[Figure 1622]

DALLE3

[Figure 1623]

Prompt Couple Avatars

[Figure 1624]

[Figure 1625]

[Figure 1626]

[Figure 1627]

[Figure 1628]

[Figure 1629]

[Figure 1630]

FLUX.1

Midjourney

[Figure 1631]

Stable Diffusion 3 Jimeng

[Figure 1632]

[Figure 1633]

Ideogram2.0

[Figure 1634]

DALLE3

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

Create an image showing a collection of all iPhone app icons arranged in a grid format. Each icon should represent a unique application, such as messages, photos, camera, mail, phone, settings, weather, calendar, clock, music, and more. The icons should be designed in a minimalistic and modern style, with a clean and polished aesthetic. Ensure each icon has a square shape with rounded edges, vibrant colors, and distinct recognizable symbols for each application. The layout should resemble the home screen of an iPhone, with

icons evenly spaced and aligned in rows and columns.

Create an image with two grids. The image should be a simple line drawing of a wedding photo. On the left grid, show a portion of the wedding photo featuring the woman in a wedding dress, with a sweet and joyful expression. On the right grid, show a portion of the same wedding photo featuring the man in a suit, smiling warmly at the woman. Both grids should be connected and display a harmonious, adorable moment between the couple. The style should be minimalist, cute, and perfect for social network profile pictures.

[Figure 1643]

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

Sec. 2.4.10 Multi-image

Style Consistency

- Figure 51: Results on style consistency task. Refer to Section 2.4.10 for detailed discussions.

Storyboard. In Figure 52, we tested the models for storytelling. The models were tasked with generating multiple images that exhibit logical relationships, with consistent styles, matching backgrounds and subjects, and a clear sense of temporal sequence. The models performed better when handling classic storylines, such as Little Red Riding Hood, compared to fabricated ones. Overall, FLUX.1 and Ideogram2.0 performed the best across all aspects, though there were still some logical inconsistencies, such as a person brushing their teeth and having breakfast in bed. In the prompt designed for a movie storyboard, FLUX.1 delivered the best results, while the other models exhibited issues such as background inconsistency.

###### Sec. 2.4.10 Multi-image

###### Storyboard

[Figure 1650]

[Figure 1651]

Prompt

Prompt

[Figure 1652]

Create a four-panel grid illustrating the story of Little Red Riding Hood. The first panel shows Little Red Riding Hood

Prompt

Two consecutive still frames of a video showing a person knocking on a wooden door. In the first frame, the person’s hand is raised, about to knock on the door, with the hallway dimly lit in the background. The door shows detailed wood grain, and the person’s fingers are positioned just above the door’s surface. In the second frame, the hand makes contact with the door, causing a slight ripple effect, as dust particles float in the soft light. The scene is atmospheric, with cinematic lighting, subtle shadows, and a sense of suspense. Created Using: cinematic realism, intricate details, natural lighting, dynamic composition

setting off from her house with a basket, on her way to visit her

Generate four images of the same person: the first

grandmother, in a peaceful forest setting. The second panel presents the moment when Little Red Riding Hood meets the wolf in the woods, unaware of the danger, with the wolf scheming in the background. The third panel shows the climax, where the wolf reaches the grandmother's house first, disguises himself as the grandmother, and Little Red Riding Hood begins to notice something is wrong. The fourth panel depicts the resolution, where the hunter arrives in time to rescue Little Red Riding Hood and her grandmother, and the wolf is defeated, bringing the story to a happy conclusion.

image shows the person waking up immediately after falling asleep, the second image shows the person brushing their teeth, the third image shows the person having breakfast, and the fourth image shows the person squeezing into a crowded subway.

[Figure 1653]

[Figure 1654]

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

[Figure 1661]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

[Figure 1668]

[Figure 1669]

[Figure 1670]

[Figure 1671]

[Figure 1672]

[Figure 1673]

Midjourney

Midjourney

DALLE3

DALLE3

Midjourney

DALLE3

[Figure 1674]

[Figure 1675]

[Figure 1676]

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen g

[Figure 1691]

[Figure 1692]

[Figure 1693]

[Figure 1694]

[Figure 1695]

[Figure 1696]

Figure 52: Results on storyboard task. Refer to Section 2.4.10 for detailed discussions.

Subject Consistency. In Figures 53 to 57, we tested the models’ ability to generate multiple images with a consistent subject. In the first prompt (Figure 53), all models successfully generated a consistent subject—a dog—but only FLUX.1 was able to follow the prompt and generate detailed images in four grids. In the second prompt, all models performed well. In the third prompt, only FLUX.1 and Ideogram2.0 were able to follow the prompt correctly. In Figure 54, the subject is a chair and a logo. Most models were able to maintain subject consistency, with the exception of Stable Diffusion 3, Midjourney, and Jimeng, which produced images with inconsistencies. In Figure 55, the subject is a woman. We changed both the background and the color of the woman’s dress. Most models maintained subject consistency, with only Stable Diffusion 3 and Jimeng showing slight differences between the subjects in the two grids.

In Figure56, we tested the models to generate different facial expressions of the same person. Regardless of whether the person was real or anime-style, all models were able to maintain facial consistency relatively well. However, only Ideogram2.0 was able to generate the required expressions in the correct order with accuracy.

Regarding multi-view generation capabilities, as illustrated in the Figure 57, most models demonstrate a reasonable understanding and consistency when generating two perspectives. Regarding multi-view generation capabilities, as shown in Figure 57, most models demonstrate reasonable understanding and consistency when generating two perspectives. However, challenges arise when generating multiple perspectives. With the exception of FLUX.1, other models frequently generate repeated perspectives.

###### Sec. 2.5.10 Multi-image

###### Subject Consistency

[Figure 1697]

[Figure 1698]

Prompt

Prompt

[Figure 1699]

Prompt

This pair of images features the same bottle of Coca-Cola in two different settings; [IMAGE1] shows the bottle of Coca-Cola placed on a clean, minimalistic table, with soft lighting highlighting its iconic red label. The bottle stands alone in the scene, with no people around, capturing the simple, still life of the beverage. [IMAGE2] presents the same Coca-Cola bottle, now in a lively sports field setting, where a person is drinking from it. The scene is dynamic, with the person holding the bottle in their hand, taking a sip,

This pair of images features the same person in the same indoor setting, but in different poses; [IMAGE1] shows the person standing in a clean, minimalist room with a neutralcolored background. The person stands with a relaxed

Generate four images of the same puppy: in the first image, the puppy is eating a chicken drumstick on the grass; in the second image, it suddenly starts raining while the puppy is eating the drumstick; in the third image, the puppy is running in the rain with the drumstick in its mouth; in the fourth image, the puppy has found a sheltered place, put the drumstick down, and started eating it again.

posture, looking forward with a calm expression, as soft,

ambient lighting highlights their figure and the simplicity of the space. [IMAGE2] presents the same person and setting, but this time the person is seated in the same room, maintaining the same calm expression while sitting comfortably, with the neutral background and soft lighting continuing to emphasize the serene atmosphere.

while the vibrant sports field background and active

surroundings create a stark contrast to the calm still life of the first image.

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

[Figure 1705]

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

[Figure 1714]

[Figure 1715]

[Figure 1716]

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1733]

[Figure 1734]

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1741]

[Figure 1742]

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

- Figure 53: Results on subject consistency task. Refer to Section 2.4.10 for detailed discussions.

[Figure 1747]

FLUX.1

Midjourney

[Figure 1748]

Stable Diffusion 3 Jimeng

[Figure 1749]

[Figure 1750]

Prompt

Ideogram2.0

[Figure 1751]

DALLE3

[Figure 1752]

Prompt

[Figure 1753]

[Figure 1754]

[Figure 1755]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

FLUX.1

Midjourney

[Figure 1760]

Stable Diffusion 3 Jimeng

[Figure 1761]

[Figure 1762]

Ideogram2.0

[Figure 1763]

DALLE3

[Figure 1764]

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

[Figure 1771]

This pair of images features the same chair placed in two different settings; [IMAGE1] shows the chair positioned against a pure white background, emphasizing its design and form in a minimalist, studio-like setting. The chair stands alone, with soft, even lighting highlighting its clean lines and smooth surface. [IMAGE2] presents the same chair, but now placed in a cozy, real-world living room. The chair is surrounded by home furnishings, such as a soft rug, a coffee table, and a sofa, with warm, natural lighting coming through the windows, creating a comfortable, lived-

in atmosphere.

This pair of images showcases the elegant identity of Starbucks; [IMAGE1] features the iconic Starbucks logo, a green circular design with a beautiful mermaid in the center, surrounded by star-like elements. The color scheme predominantly uses green and white, conveying a fresh and sophisticated atmosphere. The brand name "Starbucks" is presented in a sleek, modern font, with the overall design being simple and refined, set against a white or light beige background, offering a contemporary and minimalist vibe. [IMAGE2] shows the Starbucks logo applied to a high-end coffee cup, embossed in gold, adding a touch of luxury. The coffee cup is displayed in a cozy, inviting setting, surrounded by a wooden table, exuding warmth and premium quality, with soft natural lighting enhancing the overall ambiance.

[Figure 1772]

[Figure 1773]

[Figure 1774]

[Figure 1775]

[Figure 1776]

Sec. 2.5.10 Multi-image

Subject Consistency

[Figure 1777]

[Figure 1778]

[Figure 1779]

[Figure 1780]

[Figure 1781]

- Figure 54: Results on subject consistency task. Refer to Section 2.4.10 for detailed discussions.

###### Sec. 2.4.10 Multi-image

Subject Consistency

[Figure 1782]

[Figure 1783]

Prompt

Prompt

This pair of images features the same woman wearing the same red dress, positioned in the same pose in two different settings; [IMAGE1] shows the woman standing with her arms relaxed at her sides, looking directly at the camera with a calm expression. She wears a red, form-fitting dress with a subtle V-neck and knee-length hem, made from smooth satin fabric that catches the light. The backdrop is a clean white photography studio with soft lighting that highlights both

This pair of images features the same woman, wearing the same style of dress, but with different colors, in the same setting; [IMAGE1] shows the woman standing in a relaxed pose, with her arms gently resting at her sides, wearing a

black form-fitting dress with a subtle V-neck and knee-

length hem, made from smooth satin fabric that catches the light. The background remains consistent, with a minimalist white studio backdrop, soft lighting enhancing both the dress and her calm expression. [IMAGE2] presents the same woman, in the same pose and setting, but now wearing the same style of dress, which is white instead of black. The white dress, also satin, catches the light in a way that contrasts with the black dress from the first image, creating a fresh, elegant appearance while maintaining the serene atmosphere.

the dress and her poised stance. [IMAGE2] shows the same

woman in the same red dress, standing in the same relaxed pose with the same calm expression, but now in a vibrant city street scene. The background features a busy sidewalk with cafés, trees, and pedestrians under bright afternoon sunlight, creating a contrast between the controlled, serene studio environment and the dynamic, lively street setting.

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1791]

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

- Figure 55: Results on subject consistency task. Refer to Section 2.4.10 for detailed discussions.

[Figure 1815]

FLUX.1

Midjourney

[Figure 1816]

Stable Diffusion 3 Jimeng

[Figure 1817]

[Figure 1818]

Prompt

Ideogram2.0

[Figure 1819]

DALLE3

[Figure 1820]

Prompt

[Figure 1821]

[Figure 1822]

[Figure 1823]

[Figure 1824]

[Figure 1825]

[Figure 1826]

[Figure 1827]

FLUX.1

Midjourney

[Figure 1828]

Stable Diffusion 3 Jimeng

[Figure 1829]

[Figure 1830]

Ideogram2.0

[Figure 1831]

DALLE3

[Figure 1832]

[Figure 1833]

[Figure 1834]

[Figure 1835]

[Figure 1836]

[Figure 1837]

[Figure 1838]

A European woman in a white shirt shows six distinct expressions: joy, crying, anger, awkwardness, indifference, laughter, in six frames. Realism, expressive emotions, soft natural lighting, clean composition.

Anime-style male with short hair in simple outfit, showing

joy, crying, anger, awkwardness, indifference, and

laughter in six frames, with different expression: joyful with bright eyes, crying with tears, angry with gritted teeth, awkward with tense smile, indifferent with blank stare, and laughing with eyes closed. Minimalistic background, clean lines, soft colors

[Figure 1839]

[Figure 1840]

[Figure 1841]

Sec. 2.4.10 Multi-image

Subject Consistency

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

- Figure 56: Results on subject consistency task. Refer to Section 2.4.10 for detailed discussions.

###### Sec. 2.4.10 Multi-image

###### Subject Consistency

[Figure 1846]

[Figure 1847]

Prompt

Prompt

Show a a classic acoustic guitar, displaying it from two angles: front and back. Include detailed views of the tuning pegs, the soundhole, and the wood grain on the body. Retain consistency from views.

A 24-frame sequence arranged in a 4x6 grid, depicting a 3D model of a cup rotating 360 degrees. Each frame presents the cup from a distinct angle, beginning with a front view

and smoothly transitioning through the entire clockwise

rotation, showcasing every perspective of the cup as it turns.

[Figure 1848]

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 1862]

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1871]

[Figure 1872]

[Figure 1873]

[Figure 1874]

- Figure 57: Results on subject consistency task. Refer to Section 2.4.10 for detailed discussions.

Macimum Grid Count. In Figure 58 and 59, we tested the models to generate as many sub-images as possible within a single image. When the number of sub-images reached 25, Stable Diffusion 3 first exhibited confusion regarding the quantity. At 36 sub-images, FLUX.1 and Midjourney were also unable to generate the correct number of sub-images. At 49 and 64 sub-images, only Ideogram2.0 was able to produce results that were close to accurate. In experiments with more than 64 sub-images, none of the models could generate the correct number of sub-images.

Ability to Scale to Other Tasks. In this task, we explore the models’ in-context generation ability when scaled to other tasks, as shown in Figures 60 and 61. The models were tasked with generating two grids: the first one blurred and the second one clear. We found that FLUX.1 and Ideogram2.0 were able to accurately meet our requirements. We also extended the task to some computer vision tasks, such as depth estimation, optical flow, detection, and segmentation. However, the models struggled to achieve satisfactory results in these tasks.

Score. In multi-image tasks, we found that FLUX.1 and Ideogram 2.0 performed exceptionally well across multiple tasks, demonstrating the potential of text-to-image models for in-context generation.

- Table 34: The scoring of generation results by six models on multi-image under different evaluation

- systems. Refer to Section 2.4.10 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 26.99 0.25 5.69 6.30 9.63 Ideogram2.0 27.15 0.25 5.65 6.02 9.35 Dall-E3 26.92 0.26 5.68 5.83 9.07 Midjourney 22.08 0.28 5.68 5.46 8.98 SD3 25.38 0.28 5.65 5.83 8.24 Jimeng 27.71 0.26 5.90 5.37 8.24

###### Sec. 2.4.10 Multi-image

###### Maximum Grid Count

[Figure 1875]

[Figure 1876]

[Figure 1877]

Prompt

Prompt

Prompt

Generate 36 images, each depicting a different action of the same person.

Generate 25 images, each depicting a different action of the same person.

Generate 49 images, each depicting a different action of the same person.

[Figure 1878]

[Figure 1879]

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 1887]

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1894]

[Figure 1895]

[Figure 1896]

[Figure 1897]

[Figure 1898]

[Figure 1899]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 1900]

[Figure 1901]

[Figure 1902]

[Figure 1903]

[Figure 1904]

[Figure 1905]

[Figure 1906]

[Figure 1907]

[Figure 1908]

[Figure 1909]

[Figure 1910]

[Figure 1911]

[Figure 1912]

[Figure 1913]

[Figure 1914]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 1915]

[Figure 1916]

[Figure 1917]

[Figure 1918]

[Figure 1919]

[Figure 1920]

- Figure 58: Results on maximum grid count task. Refer to Section 2.4.10 for detailed discussions.

[Figure 1921]

FLUX.1

Midjourney

[Figure 1922]

Stable Diffusion 3 Jimeng

[Figure 1923]

[Figure 1924]

Prompt

Ideogram2.0

[Figure 1925]

DALLE3

[Figure 1926]

Prompt

[Figure 1927]

[Figure 1928]

[Figure 1929]

[Figure 1930]

[Figure 1931]

[Figure 1932]

[Figure 1933]

FLUX.1

Midjourney

[Figure 1934]

Stable Diffusion 3 Jimeng

[Figure 1935]

[Figure 1936]

Ideogram2.0

[Figure 1937]

DALLE3

[Figure 1938]

[Figure 1939]

[Figure 1940]

[Figure 1941]

[Figure 1942]

[Figure 1943]

[Figure 1944]

Generate 81 images, each depicting a different action of the

same person.

Generate 100 images, each depicting a different action of the same person.

[Figure 1945]

[Figure 1946]

[Figure 1947]

Sec. 2.4.10 Multi-image

Maximum Grid Count

[Figure 1948]

[Figure 1949]

Prompt

[Figure 1950]

[Figure 1951]

FLUX.1

Midjourney

[Figure 1952]

Stable Diffusion 3 Jimeng

[Figure 1953]

[Figure 1954]

Ideogram2.0

[Figure 1955]

DALLE3

[Figure 1956]

[Figure 1957]

[Figure 1958]

[Figure 1959]

[Figure 1960]

[Figure 1961]

Generate 64 images, each depicting a different action of the same person.

[Figure 1962]

[Figure 1963]

- Figure 59: Results on maximum grid count task. Refer to Section 2.4.10 for detailed discussions.

###### Sec. 2.4.10 Multi-image

###### Ability to Scale to Other Tasks

Prompt Outpainting

[Figure 1964]

Prompt Depth Estimation

[Figure 1965]

Prompt Blurred and clear images

[Figure 1966]

This pair of images showcases two connected artworks:[IMAGE1] is a closeup of a person's face, with soft, shoulder-length hair and a serene expression. Their warm eyes and subtle makeup highlight their natural features. The background is softly blurred, creating depth without distracting from the face, while soft lighting emphasizes the contours of the cheeks and jawline. [IMAGE2] shows the same person in a full-body shot, standing naturally with one hand at their side and the other relaxed. They wear a light-colored, simple outfit that complements their calm personality. The background features

Create two images. The first image should depict a simple street scene, with a sidewalk, buildings, trees, and a car passing by. Use a clear and realistic style to represent the environment. The second image should be a depth estimation result of the first image, processed to reflect depth using a gradient of colors (purple for distant areas and yellow/orange for closer areas). The depth map should maintain accurate spatial relationships

This pair of images features the same landscape scene captured with different focus levels; [IMAGE1] shows a serene outdoor landscape with rolling hills, a calm river, and scattered trees. The first image appears very blurred. [IMAGE2] presents the same landscape, but this time the image is crystal clear, with sharp focus on every detail of the hills, river, and trees, allowing the natural beauty of the scene to be fully appreciated in perfect clarity.

a peaceful outdoor setting, like a garden or beach, with soft lighting and natural

and emphasize the depth of objects in the scene. Ensure

shadows. The overall composition maintains a sense of calm and elegance, with the face from [IMAGE1] seamlessly connecting to the full-body shot.

the two images are side by side, labeled as 'Raw image' for the first and 'Depth Estimation' for the second.

[Figure 1967]

[Figure 1968]

[Figure 1969]

[Figure 1970]

[Figure 1971]

[Figure 1972]

[Figure 1973]

[Figure 1974]

[Figure 1975]

[Figure 1976]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 1977]

[Figure 1978]

[Figure 1979]

[Figure 1980]

[Figure 1981]

[Figure 1982]

[Figure 1983]

[Figure 1984]

[Figure 1985]

[Figure 1986]

[Figure 1987]

[Figure 1988]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 1989]

[Figure 1990]

[Figure 1991]

[Figure 1992]

[Figure 1993]

[Figure 1994]

[Figure 1995]

[Figure 1996]

[Figure 1997]

[Figure 1998]

[Figure 1999]

[Figure 2000]

[Figure 2001]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen g

[Figure 2002]

[Figure 2003]

[Figure 2004]

[Figure 2005]

[Figure 2006]

[Figure 2007]

- Figure 60: Results on ability to scale to other tasks. Refer to Section 2.4.10 for detailed discussions.

[Figure 2008]

FLUX.1

Midjourney

[Figure 2009]

Stable Diffusion 3 Jimen g

[Figure 2010]

[Figure 2011]

Prompt Bounding Box

Ideogram2.0

[Figure 2012]

DALLE3

[Figure 2013]

Prompt Segmentation

[Figure 2014]

[Figure 2015]

[Figure 2016]

[Figure 2017]

[Figure 2018]

[Figure 2019]

FLUX.1

Midjourney

[Figure 2020]

Stable Diffusion 3 Jimeng

[Figure 2021]

[Figure 2022]

Ideogram2.0

[Figure 2023]

DALLE3

[Figure 2024]

[Figure 2025]

[Figure 2026]

This pair of images features the same street scene with a car, with the second image highlighting the car with a bounding box; [IMAGE1] shows a bustling street scene with various vehicles moving along the road. A car is positioned in the center, surrounded by other cars and pedestrians, with buildings lining the sides. The image captures the vibrant energy of the street without bounding box. [IMAGE2] presents the exact same scene, but this time the car is highlighted with a bounding box around it, drawing attention to the vehicle. The rest of the scene remains unchanged, with the bounding box clearly outlining the car, making it the focal point of the image.

Create two images. The first image should be a natural scene featuring a person sitting on a ledge near a body of water, with realistic details in the background such as the sky, water, and surrounding objects. The second image should be the segmentation result of the first image, where the person, sky, water, ledge, and other objects are represented by distinct, flat colors. The segmentation should clearly outline each object and region, emphasizing clean boundaries between them. Place the two images side by side, with the first labeled as 'Raw image' and the second as 'Segmentation Result.'

[Figure 2027]

Sec. 2.4.10 Multi-image

Ability to Scale to Other Tasks

[Figure 2028]

Prompt Optical Flow

[Figure 2029]

[Figure 2030]

FLUX.1

Midjourney

[Figure 2031]

Stable Diffusion 3 Jimeng

[Figure 2032]

[Figure 2033]

Ideogram2.0

[Figure 2034]

DALLE3

[Figure 2035]

[Figure 2036]

[Figure 2037]

[Figure 2038]

[Figure 2039]

[Figure 2040]

Create three images to demonstrate optical flow algorithm results. The first image should depict a scene with chairs positioned against a clear background, such as an outdoor sky. The second image should show the same scene with the chairs slightly shifted in position or

orientation to simulate motion. The third image should

visualize the optical flow result, using a color gradient to represent the direction and magnitude of movement for each chair. Arrange the three images side by side, with the first labeled as 'Original Image 1,' the second as 'Original Image 2,' and the third as 'Optical Flow Result.'

[Figure 2041]

[Figure 2042]

[Figure 2043]

[Figure 2044]

[Figure 2045]

[Figure 2046]

[Figure 2047]

[Figure 2048]

[Figure 2049]

[Figure 2050]

[Figure 2051]

[Figure 2052]

[Figure 2053]

[Figure 2054]

- Figure 61: Results on ability to scale to other tasks. Refer to Section 2.4.10 for detailed discussions.

###### 2.4.11 Text Writing

In Figures 62, we examine the models’ ability to generate images containing accurate text. The results demonstrate that none of the models consistently produce correct text within images. Specifically, Jimeng fails to generate correct words, instead producing distorted or illegible characters. Dall-E3,

Midjourney, and Stable Diffusion 3 occasionally include text with unreadable characters. FLUX.1 and Ideogram2.0 show better performance, though they still generate text with misspelled words or incomplete phrases.

Score. The results of this experiment are shown in the Table 35, where FLUX.1 and Ideogram2.0 performed the best. Only the results of HPSv2 aligned closely with human perception, possibly because the evaluation of this task requires a certain understanding of the text information in the images.

- Table 35: The scoring of generation results by six models on text writing under different evaluation

- systems. Refer to Section 2.4.11 for detailed discussions. Model CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 37.01 0.29 4.34 5.83 8.33 Ideogram2.0 35.33 0.28 4.94 5.00 8.33 Dall-E3 35.18 0.25 3.94 4.17 6.67 Midjourney 42.27 0.28 4.19 6.67 5.00 SD3 46.14 0.28 4.23 3.33 3.33 Jimeng 36.03 0.25 5.81 5.00 3.33

Sec. 2.4.11 Text Writing

[Figure 2055]

[Figure 2056]

Prompt

Prompt

A photograph of a classroom blackboard, dimly lit, with poetic text written in chalk: "In the quiet of the night, the stars hum a song. A melody of dreams long forgotten, a whisper of realms beyond. Close your eyes, and let the night carry you where the sky meets the sea."

An illustration of a letter pad with the text "Hi, I’m Flux. I am an advanced text-to-image AI model. If you're interested in AI-generated text-to-image artwork, feel free to try using FLUX.". The letter pad is on a wooden.

[Figure 2057]

[Figure 2058]

[Figure 2059]

[Figure 2060]

[Figure 2061]

[Figure 2062]

[Figure 2063]

[Figure 2064]

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 2065]

[Figure 2066]

[Figure 2067]

[Figure 2068]

[Figure 2069]

[Figure 2070]

[Figure 2071]

[Figure 2072]

[Figure 2073]

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 2074]

[Figure 2075]

[Figure 2076]

[Figure 2077]

[Figure 2078]

[Figure 2079]

[Figure 2080]

[Figure 2081]

[Figure 2082]

[Figure 2083]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 2084]

[Figure 2085]

[Figure 2086]

[Figure 2087]

Figure 62: Results on text writing task. Refer to Section 2.4.11 for detailed discussions.

###### 2.5 Multi-style Creation Task

In this experiment, we selected over 30 commonly used styles in text-to-image tasks and tailored the prompts accordingly. Additionally, we examined the model’s imagination by combining seemingly contradictory styles and objects, such as floating chiffon in a marble texture, to test how well the model can generate coherent and creative images from these combinations. We evaluated the generation results from six models, with detailed scores presented in Table 36 and visualizations provided in Figures 63-73.

We found that the performance of the models in this task did not vary significantly, but the images generated by Midjourney often exhibited greater aesthetic appeal and better alignment with human perception of beauty. For example, in the middle subplot of Figure 63, all models can produce images in the isometric anime style, but the images generated by Midjourney and Jimeng stand out for their aesthetic quality. Similarly, in the left subplot of Figure 67, only Midjourney is able to generate a marble sculpture that conveys the most fluidity and grace.

Score. Notably, while Stable Diffusion 3 [64] demonstrated room for improvement, the other models exhibited relatively similar performance. Among the six models, Midjourney exhibited superior generalization capabilities and higher aesthetic quality, securing higher scores. The results from GPT-4o were closely aligned with human aesthetic judgments, while the outputs from other evaluation systems showed notable discrepancies from human perceptions of aesthetics.

- Table 36: The scoring of generation results by six models on different style image generation under different evaluation systems. Refer to Section 2.5 for detailed discussions.

Metric CLIPScore HPSv2 Aesthetic Score GPT-4o Human

FLUX.1 30.49 0.29 6.07 7.13 9.26 Ideogram2.0 30.96 0.29 6.12 6.69 9.26 Dall-E3 31.10 0.29 6.04 6.94 9.44 Midjourney 30.26 0.28 5.99 7.13 9.61 SD3 29.51 0.28 6.01 6.89 8.68 Jimeng 30.55 0.28 5.98 6.30 9.36

###### Sec. 2.5 Style

[Figure 2088]

[Figure 2089]

Prompt

Prompt

[Figure 2090]

Prompt

A detailed analytic drawing of a European medieval garden, featuring geometric pathways, trimmed hedges, and

An isometric anime-style illustration of a small countryside courtyard, with a cozy wooden house, a small vegetable garden, and a stone pathway leading to a

A coloring book style illustration of a beautiful Chinese garden, featuring a traditional pagoda, flowing water from a small pond, and elegant stone bridges. The garden is filled with lush trees, blooming lotus flowers, and winding pathways. The outlines are crisp and bold, ready for coloring, with intricate details in the architecture and nature. Created Using: clean line work, intricate patterns, traditional Chinese elements, serene atmosphere

stone fountains. The garden is surrounded by tall, ivy-

covered castle walls in the background. The drawing highlights the structured symmetry of the garden, with carefully arranged trees, shrubs, and floral patterns. The pencil strokes emphasize the textures of the stone and foliage. Created Using: precise line work, geometric forms, architectural details, monochrome shading.

quaint well. The yard is filled with blooming flowers, and

a gentle breeze sways the trees nearby. Soft sunlight bathes the entire scene, casting warm shadows. Created Using: pastel colors, soft anime shading, detailed textures, peaceful atmosphere

Style Example: Analytic Drawing Style Example: Isometric anime-style

Style Example: Coloring Book

[Figure 2091]

[Figure 2092]

[Figure 2093]

| |
|---|

| |
|---|

[Figure 2094]

[Figure 2095]

[Figure 2096]

[Figure 2097]

[Figure 2098]

[Figure 2099]

FLUX.1

Ideogram2.0

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 2100]

[Figure 2101]

[Figure 2102]

[Figure 2103]

[Figure 2104]

[Figure 2105]

[Figure 2106]

[Figure 2107]

[Figure 2108]

[Figure 2109]

[Figure 2110]

[Figure 2111]

Midjourney

DALLE3

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 2112]

[Figure 2113]

[Figure 2114]

[Figure 2115]

[Figure 2116]

[Figure 2117]

[Figure 2118]

[Figure 2119]

[Figure 2120]

[Figure 2121]

[Figure 2122]

[Figure 2123]

[Figure 2124]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen g

Stable Diffusion 3 Jimeng

[Figure 2125]

[Figure 2126]

[Figure 2127]

[Figure 2128]

[Figure 2129]

[Figure 2130]

- Figure 63: Results on different style image generation task. Refer to Section 2.5 for detailed discussions.

###### Sec. 2.5 Style

[Figure 2131]

[Figure 2132]

[Figure 2133]

Prompt

Prompt

Prompt

A pixel art-style fairy tale illustration of a little girl with braided pigtails, wearing a simple dress, holding her small cat. The scene is set in a cozy forest clearing, with vibrant pixelated flowers and trees surrounding her. The girl and her cat are the central focus, with soft, charming pixel details highlighting their features. Created Using: retro pixel art, vibrant colors, cute and whimsical style, fairy tale

An ukiyo-e style artwork of the vast ocean, featuring large, dramatic waves with intricate, swirling patterns. The waves are crashing with foam against each other, while a distant horizon shows a clear sky. The water is depicted in bold, flowing lines, characteristic of traditional Japanese woodblock prints. Created Using: ukiyo-e style, bold outlines, intricate wave patterns, traditional Japanese elements

A serene watercolor painting of a small creek winding through a dense forest. The soft, flowing water reflects the lush greenery of the trees, while delicate brushstrokes capture the sunlight filtering through the canopy. Moss-covered rocks line the creek, and the vibrant colors of flowers dot the forest floor. Created Using: soft watercolors, gentle lighting, natural tones, peaceful atmosphere

atmosphere

###### Style Example: Pixel art Style Example: Watercolor Painting Style Example: Ukiyo-e Style

[Figure 2134]

[Figure 2135]

[Figure 2136]

[Figure 2137]

[Figure 2138]

[Figure 2139]

[Figure 2140]

[Figure 2141]

[Figure 2142]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 2143]

[Figure 2144]

[Figure 2145]

[Figure 2146]

[Figure 2147]

[Figure 2148]

[Figure 2149]

[Figure 2150]

[Figure 2151]

[Figure 2152]

[Figure 2153]

[Figure 2154]

Midjourney

DALLE3

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 2155]

[Figure 2156]

[Figure 2157]

[Figure 2158]

[Figure 2159]

[Figure 2160]

[Figure 2161]

[Figure 2162]

[Figure 2163]

[Figure 2164]

[Figure 2165]

[Figure 2166]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

[Figure 2167]

[Figure 2168]

[Figure 2169]

[Figure 2170]

[Figure 2171]

[Figure 2172]

- Figure 64: Results on different style image generation task. Refer to Section 2.5 for detailed discussions.

[Figure 2173]

FLUX.1

Midjourney

[Figure 2174]

Stable Diffusion 3 Jimeng

[Figure 2175]

[Figure 2176]

Prompt

Ideogram2.0

[Figure 2177]

DALLE3

[Figure 2178]

Prompt

[Figure 2179]

[Figure 2180]

[Figure 2181]

[Figure 2182]

[Figure 2183]

[Figure 2184]

[Figure 2185]

FLUX.1

Midjourney

[Figure 2186]

Stable Diffusion 3 Jimeng

[Figure 2187]

[Figure 2188]

Ideogram2.0

[Figure 2189]

DALLE3

[Figure 2190]

[Figure 2191]

[Figure 2192]

[Figure 2193]

[Figure 2194]

[Figure 2195]

[Figure 2196]

A minimalist tattoo of a small, cute cat curled up

sleeping, placed on the inner forearm. The cat's body is

outlined with smooth, fine lines, with its tiny paws tucked in and its tail wrapped around its body. The design has a simple yet elegant aesthetic, with subtle shading to give depth and softness. Created Using: clean lines, minimalist style, delicate shading, subtle detail

A detailed anatomical drawing of a human arm, showcasing the muscles, bones, and tendons. The arm is illustrated with precision, highlighting the biceps, triceps, and forearm muscles. The intricate details of the bone structure, joints, and veins are emphasized through fine line

work and shading, giving a scientific and educational feel to

the drawing. Created Using: precise line work, anatomical accuracy, detailed shading, scientific illustration style

[Figure 2197]

Sec. 2.5 Style

[Figure 2198]

Prompt

[Figure 2199]

[Figure 2200]

FLUX.1

Midjourney

[Figure 2201]

Stable Diffusion 3 Jimeng

[Figure 2202]

[Figure 2203]

Ideogram2.0

[Figure 2204]

DALLE3

[Figure 2205]

[Figure 2206]

[Figure 2207]

[Figure 2208]

[Figure 2209]

A massive cyberpunk-style castle, with towering spires made of metal and glass, glowing neon lights illuminating the structure. The castle combines futuristic technology with traditional architecture, featuring intricate designs and mechanical details. Flying drones and holographic projections surround the castle, while the dark cityscape looms in the background. Created Using: neon lighting, futuristic architecture, detailed cybernetics, dark city atmosphere

[Figure 2210]

[Figure 2211]

[Figure 2212]

[Figure 2213]

[Figure 2214]

[Figure 2215]

[Figure 2216]

[Figure 2217]

[Figure 2218]

Style Example: Tattoo Style Example: Anatomical Drawing Style Example: Cyberpunk Style

[Figure 2219]

[Figure 2220]

[Figure 2221]

- Figure 65: Results on different style image generation task. Refer to Section 2.5 for detailed

###### Sec. 2.5 Style

[Figure 2222]

[Figure 2223]

[Figure 2224]

Prompt

Prompt

Prompt

A computer in a knitted style, with the body and keyboard covered in soft, detailed knitting patterns. The screen is surrounded by yarn and wool designs, creating a cozy and warm aesthetic. The knitting features intricate patterns like cables and stitches, adding a handcrafted, homemade feel to

A beautifully designed pen with intricate floral patterns wrapping around the barrel, featuring delicate vines, blooming roses, and swirling leaves. The floral design is

A close-up photograph of a latex lollipop, its glossy surface reflecting light, creating a smooth and shiny texture. The lollipop is round, with a soft rubbery look, slightly bending as if flexible. Its vibrant colors resemble traditional candy but the material gives it a unique, playful twist. Created Using: hyper-realism, glossy surface, vibrant colors, soft reflections

engraved in fine detail, giving the pen an elegant and artistic

appearance. Soft metallic accents complement the floral patterns, adding a luxurious touch. Created Using: intricate floral patterns, elegant design, fine detailing, metallic accents

the high-tech device. Created Using: detailed knitting

patterns, cozy texture, soft yarn aesthetics, handcrafted style

###### Style Example: Floral Style Example: Knitted Style Style Example: Latex

[Figure 2225]

[Figure 2226]

[Figure 2227]

[Figure 2228]

[Figure 2229]

[Figure 2230]

[Figure 2231]

[Figure 2232]

Ideogram2.0

[Figure 2233]

[Figure 2234]

[Figure 2235]

[Figure 2236]

[Figure 2237]

[Figure 2238]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

[Figure 2239]

[Figure 2240]

[Figure 2241]

[Figure 2242]

[Figure 2243]

[Figure 2244]

[Figure 2245]

[Figure 2246]

[Figure 2247]

[Figure 2248]

[Figure 2249]

[Figure 2250]

[Figure 2251]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 2252]

[Figure 2253]

[Figure 2254]

[Figure 2255]

[Figure 2256]

[Figure 2257]

[Figure 2258]

[Figure 2259]

[Figure 2260]

[Figure 2261]

[Figure 2262]

[Figure 2263]

[Figure 2264]

[Figure 2265]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen g

[Figure 2266]

[Figure 2267]

[Figure 2268]

[Figure 2269]

[Figure 2270]

[Figure 2271]

- Figure 66: Results on different style image generation task. Refer to Section 2.5 for detailed discussions.

[Figure 2272]

FLUX.1

Midjourney

[Figure 2273]

Stable Diffusion 3 Jimen g

[Figure 2274]

[Figure 2275]

Prompt

Ideogram2.0

[Figure 2276]

DALLE3

[Figure 2277]

Prompt

[Figure 2278]

[Figure 2279]

[Figure 2280]

[Figure 2281]

[Figure 2282]

[Figure 2283]

[Figure 2284]

FLUX.1

Midjourney

[Figure 2285]

Stable Diffusion 3 Jimeng

[Figure 2286]

[Figure 2287]

Ideogram2.0

[Figure 2288]

DALLE3

[Figure 2289]

[Figure 2290]

[Figure 2291]

[Figure 2292]

[Figure 2293]

[Figure 2294]

[Figure 2295]

A marble statue of delicate, flowing sheer fabric, captured in mid-air as if caught by a breeze. The marble sculpture intricately mimics the softness and movement of light fabric, with fine details in the folds and ripples. The fabric appears weightless, despite being carved from solid stone, giving the statue an ethereal quality. Created Using: realistic marble textures, intricate carving details, flowing movement, soft and airy feel

An old photograph-style image of futuristic glasses, featuring sleek, minimalist frames with glowing digital displays. The glasses have a retro-futuristic design, blending advanced technology with a vintage aesthetic. The sepiatoned photograph adds an antique feel, with subtle wear and tear marks, creating a unique contrast between the past and future. Created Using: sepia tones, vintage photography style, futuristic design, worn texture

[Figure 2296]

Sec. 2.5 Style

[Figure 2297]

Prompt

[Figure 2298]

[Figure 2299]

FLUX.1

Midjourney

[Figure 2300]

Stable Diffusion 3 Jimeng

[Figure 2301]

[Figure 2302]

Ideogram2.0

[Figure 2303]

DALLE3

[Figure 2304]

[Figure 2305]

[Figure 2306]

[Figure 2307]

[Figure 2308]

An origami-style giant panda, crafted from folded paper with sharp, angular edges. The panda is posed in a sitting position, with simple yet recognizable details like its blackand-white patches. The paper folds create geometric shapes, giving the panda a minimalist and elegant appearance, capturing the essence of traditional origami art. Created Using: precise paper folds, geometric shapes, minimalist design, traditional origami techniques

[Figure 2309]

[Figure 2310]

Style Example: Marble Statue Style Example: Old Photograph Style Example: Origami

[Figure 2311]

[Figure 2312]

[Figure 2313]

- Figure 67: Results on different style image generation task. Refer to Section 2.5 for detailed

###### Sec. 2.5 Style

[Figure 2314]

[Figure 2315]

[Figure 2316]

Prompt

Prompt

Prompt

A miniature faking-style scene of a busy IT company office, with tiny figures of workers typing at computers, attending meetings, and moving between desks. The desks are filled with miniature monitors, laptops, and gadgets, and the office layout includes small cubicles, glass-walled meeting rooms,

A bustling cityscape in the style of Hayao Miyazaki, featuring towering buildings with intricate architecture, flying vehicles, and winding streets filled with people. The city is alive with movement, as colorful shops, street

A small cat depicted in the style of Yaacov Agam, with vibrant, abstract geometric shapes forming its body. The cat’s form is composed of dynamic patterns and shifting colors, creating a sense of movement and energy, as though the image changes depending on the viewer's angle. The bold use of color and linework gives the cat a playful, kinetic art

vendors, and whimsical characters add charm and vibrancy

and bustling workstations. The tilt-shift effect makes the

to the scene. The skyline blends old-world charm with futuristic elements, while soft sunlight filters through the clouds. Created Using: Miyazaki-style whimsical details, vibrant colors, intricate architecture, lively atmosphere

scene look like a toy model, giving it a playful, surreal feel. Created Using: tilt-shift effect, miniature details, busy office environment, vibrant atmosphere

feel. Created Using: abstract geometric patterns, vibrant

colors, kinetic art style, dynamic energy

###### Style Example: Miniature Faking Style Example: Hayao Miyazaki Style Example: Yaacov Agam

[Figure 2317]

[Figure 2318]

[Figure 2319]

[Figure 2320]

[Figure 2321]

[Figure 2322]

[Figure 2323]

[Figure 2324]

[Figure 2325]

[Figure 2326]

[Figure 2327]

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

FLUX.1

Ideogram2.0

[Figure 2328]

[Figure 2329]

[Figure 2330]

[Figure 2331]

[Figure 2332]

[Figure 2333]

[Figure 2334]

[Figure 2335]

[Figure 2336]

[Figure 2337]

[Figure 2338]

[Figure 2339]

[Figure 2340]

Midjourney

DALLE3

Midjourney

Midjourney

DALLE3

DALLE3

[Figure 2341]

[Figure 2342]

[Figure 2343]

[Figure 2344]

[Figure 2345]

[Figure 2346]

[Figure 2347]

[Figure 2348]

[Figure 2349]

[Figure 2350]

[Figure 2351]

[Figure 2352]

[Figure 2353]

[Figure 2354]

[Figure 2355]

[Figure 2356]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen g

Stable Diffusion 3 Jimeng

[Figure 2357]

[Figure 2358]

[Figure 2359]

[Figure 2360]

[Figure 2361]

[Figure 2362]

- Figure 68: Results on different style image generation task. Refer to Section 2.5 for detailed discussions.

[Figure 2363]

FLUX.1

Midjourney

[Figure 2364]

Stable Diffusion 3 Jimen g

[Figure 2365]

[Figure 2366]

Prompt

Ideogram2.0

[Figure 2367]

DALLE3

[Figure 2368]

Prompt

[Figure 2369]

[Figure 2370]

[Figure 2371]

[Figure 2372]

[Figure 2373]

[Figure 2374]

[Figure 2375]

FLUX.1

Midjourney

[Figure 2376]

Stable Diffusion 3 Jimeng

[Figure 2377]

[Figure 2378]

Ideogram2.0

[Figure 2379]

DALLE3

[Figure 2380]

[Figure 2381]

[Figure 2382]

[Figure 2383]

[Figure 2384]

[Figure 2385]

[Figure 2386]

A modern computer illustrated in the Pre-Raphaelite style, with elegant curves and intricate details resembling medieval craftsmanship. The computer is adorned with flowing floral motifs, rich colors, and golden accents,

blending modern technology with the natural, ornate

aesthetics of the Pre-Raphaelite era. Soft, natural light highlights the textures and details, creating a harmonious fusion of old and new. Created Using: intricate detailing, rich color palette, natural motifs, elegant craftsmanship

A ceramic-style sculpture of flowing sheer fabric, captured mid-air as though caught in a breeze. The fabric is delicately sculpted in smooth, curved lines, with fine detailing to mimic the lightness of real fabric. The glossy ceramic finish adds a refined, polished touch, enhancing the elegant movement of the folds. Created Using: smooth ceramic textures, intricate sculpting, flowing movement, polished finish

[Figure 2387]

Sec. 2.5 Style

[Figure 2388]

Prompt

[Figure 2389]

[Figure 2390]

FLUX.1

Midjourney

[Figure 2391]

Stable Diffusion 3 Jimeng

[Figure 2392]

[Figure 2393]

Ideogram2.0

[Figure 2394]

DALLE3

[Figure 2395]

[Figure 2396]

[Figure 2397]

[Figure 2398]

[Figure 2399]

Satellite view of the Moon showing detailed craters, mountain ranges, and plains. High-definition imagery reveals intricate textures and topography. The realistic image features soft shadows across the

lunar surface, with scientific

accuracy and sharp detail.

[Figure 2400]

[Figure 2401]

[Figure 2402]

[Figure 2403]

[Figure 2404]

[Figure 2405]

[Figure 2406]

[Figure 2407]

[Figure 2408]

[Figure 2409]

[Figure 2410]

Style Example: Pre-Raphaelite Style Example: Ceramic Style Example: Satellite

[Figure 2411]

[Figure 2412]

[Figure 2413]

- Figure 69: Results on different style image generation task. Refer to Section 2.5 for detailed

###### Sec. 2.5 Style

[Figure 2414]

[Figure 2415]

[Figure 2416]

Prompt

Prompt

Prompt

A breathtaking view of the Forbidden City, with its grand palaces and traditional Chinese architecture. The intricate roofs are covered in golden tiles, reflecting the sunlight,

A jewelry-style design of green leaves, crafted with intricate detailing and adorned with gemstone-like textures. The leaves are sculpted with fine lines, resembling emeralds or jade, with a radiant shine that catches the light. Each leaf is outlined with delicate gold accents, adding elegance and luxury to the piece. Created Using: gemstone texture, intricate details, emerald green, gold accents

A majestic Chinese dragon with a long, serpentine body, shimmering red, gold, and green scales. Flowing mane, whiskers, symbolizing power and grace. Surrounded by swirling clouds, distant mountains, symbolizing strength and wisdom. Created using intricate scales, vibrant colors, dynamic movement, traditional Chinese art style.

while red walls and ornate carvings decorate the structures.

The symmetrical layout of the courtyards and stone pathways lead to towering gates, all set against a clear blue sky. Created Using: traditional Chinese architecture, intricate details, golden rooftops, historical elegance

###### Style Example: Chinese Style Style Example: Chinese Style Style Example: Jewelry

[Figure 2417]

[Figure 2418]

[Figure 2419]

[Figure 2420]

[Figure 2421]

[Figure 2422]

[Figure 2423]

[Figure 2424]

[Figure 2425]

[Figure 2426]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 2427]

[Figure 2428]

[Figure 2429]

[Figure 2430]

[Figure 2431]

[Figure 2432]

[Figure 2433]

[Figure 2434]

[Figure 2435]

[Figure 2436]

[Figure 2437]

[Figure 2438]

[Figure 2439]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 2440]

[Figure 2441]

[Figure 2442]

[Figure 2443]

[Figure 2444]

[Figure 2445]

[Figure 2446]

[Figure 2447]

[Figure 2448]

[Figure 2449]

[Figure 2450]

[Figure 2451]

[Figure 2452]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen

[Figure 2453]

g

[Figure 2454]

[Figure 2455]

[Figure 2456]

[Figure 2457]

[Figure 2458]

- Figure 70: Results on different style image generation task. Refer to Section 2.5 for detailed discussions.

[Figure 2459]

FLUX.1

Midjourney

[Figure 2460]

Stable Diffusion 3 Jimen g

[Figure 2461]

[Figure 2462]

Prompt

Ideogram2.0

[Figure 2463]

DALLE3

[Figure 2464]

Prompt

[Figure 2465]

[Figure 2466]

[Figure 2467]

[Figure 2468]

[Figure 2469]

[Figure 2470]

[Figure 2471]

FLUX.1

Midjourney

[Figure 2472]

Stable Diffusion 3 Jimeng

[Figure 2473]

[Figure 2474]

Ideogram2.0

[Figure 2475]

DALLE3

[Figure 2476]

[Figure 2477]

[Figure 2478]

[Figure 2479]

[Figure 2480]

[Figure 2481]

[Figure 2482]

A traditional Chinese ink painting of majestic mountains and flowing rivers, with bold, sweeping brushstrokes capturing the rugged cliffs and serene water. The mountains rise high with intricate, textured lines, while the river winds softly through the landscape, reflecting the calm of nature. Mist lingers in the valleys, creating a sense of depth and mystery. Created Using: ink wash technique, bold brushstrokes, natural landscape, serene atmosphere

A highly detailed, realistic portrait of a person sitting at a table, eating an apple. The light softly illuminates their face and the natural textures of the apple, while the table is

scattered with everyday items. The person's expression is

calm and focused as they take a bite, with the folds of their clothing and the subtle shadows on their face captured in fine detail. Created Using: realistic details, soft natural lighting, textured surfaces, everyday life

[Figure 2483]

Sec. 2.5 Style

[Figure 2484]

Prompt

[Figure 2485]

[Figure 2486]

FLUX.1

Midjourney

[Figure 2487]

Stable Diffusion 3 Jimeng

[Figure 2488]

[Figure 2489]

Ideogram2.0

[Figure 2490]

DALLE3

[Figure 2491]

[Figure 2492]

[Figure 2493]

[Figure 2494]

[Figure 2495]

A minimalist logo for "AI Generation" on a pure white background. The design features sleek, modern typography with subtle geometric shapes representing artificial intelligence. The letters "AI" are slightly bolder, with clean lines, and the "Generation" text is sleek and futuristic. The overall look is simple and sophisticated, with no additional elements to keep the focus on clarity and innovation. Created Using: minimalist design, clean typography, geometric accents, modern aesthetics

[Figure 2496]

[Figure 2497]

[Figure 2498]

[Figure 2499]

[Figure 2500]

Style Example: Chinese ink painting Style Example: Realistic Portrait Style Example: Logo

[Figure 2501]

[Figure 2502]

[Figure 2503]

- Figure 71: Results on different style image generation task. Refer to Section 2.5 for detailed

###### Sec. 2.5 Style

[Figure 2504]

[Figure 2505]

Prompt

Prompt

[Figure 2506]

Prompt

A vibrant anime-style scene of a grassy field, where a young girl stands wearing a flowing skirt. A strong wind blows across the field, lifting her skirt in a dramatic motion, while the grass sways in the breeze. The girl has a surprised

A deconstructed blueberry mousse cake, with each layer floating in the air and labeled from top to bottom. The top layer features fresh blueberries and mint leaves for decoration, followed by a transparent blueberry jelly layer that adds shine and texture. Below that is the light and fluffy blueberry mousse, made from blueberry puree, whipped cream, and gelatin. The creamy cheesecake layer comes next, a blend of cream cheese, whipped cream, sugar, and vanilla for a soft texture. Underneath is a layer of blueberry jam, rich and fruity, made from fresh blueberries and sugar. The base is made of a firm biscuit crust, pressed from digestive biscuit crumbs and butter to create a solid foundation. Each layer is individually highlighted and labeled with arrows pointing to their respective components. Created Using: floating layers, deconstructed food, detailed textures, elegant dessert

A minimalistic line drawing of a modern architectural building, featuring clean and precise lines,

sharp angles, and geometric forms.

The sketch is done in black ink on a white background, showcasing the simplicity and elegance of the building’s structure. Created Using: architectural sketch, minimalism, clean lines, precise detailing

yet playful expression, with her hair and

skirt fluttering in the wind. The sky is bright and filled with soft clouds, creating a lighthearted and energetic atmosphere. Created Using: anime-style, dynamic wind effects, vibrant colors, playful scene

[Figure 2507]

###### Style Example: Anime-Style Style Example: Minimalistic Line Style Example: Minimalistic Line

[Figure 2508]

[Figure 2509]

[Figure 2510]

[Figure 2511]

[Figure 2512]

[Figure 2513]

[Figure 2514]

[Figure 2515]

[Figure 2516]

[Figure 2517]

FLUX.1

FLUX.1

FLUX.1

Ideogram2.0

Ideogram2.0

Ideogram2.0

[Figure 2518]

[Figure 2519]

[Figure 2520]

[Figure 2521]

[Figure 2522]

[Figure 2523]

[Figure 2524]

[Figure 2525]

[Figure 2526]

[Figure 2527]

[Figure 2528]

[Figure 2529]

[Figure 2530]

Midjourney

Midjourney

Midjourney

DALLE3

DALLE3

DALLE3

[Figure 2531]

[Figure 2532]

[Figure 2533]

[Figure 2534]

[Figure 2535]

[Figure 2536]

[Figure 2537]

[Figure 2538]

[Figure 2539]

[Figure 2540]

[Figure 2541]

[Figure 2542]

[Figure 2543]

[Figure 2544]

[Figure 2545]

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimeng

Stable Diffusion 3 Jimen

g

[Figure 2546]

[Figure 2547]

[Figure 2548]

[Figure 2549]

[Figure 2550]

[Figure 2551]

- Figure 72: Results on different style image generation task. Refer to Section 2.5 for detailed discussions.

[Figure 2552]

FLUX.1

Midjourney

[Figure 2553]

Stable Diffusion 3 Jimen g

[Figure 2554]

[Figure 2555]

Prompt

Ideogram2.0

[Figure 2556]

DALLE3

[Figure 2557]

[Figure 2558]

[Figure 2559]

[Figure 2560]

[Figure 2561]

[Figure 2562]

[Figure 2563]

FLUX.1

Midjourney

[Figure 2564]

Stable Diffusion 3 Jimeng

[Figure 2565]

[Figure 2566]

Ideogram2.0

[Figure 2567]

DALLE3

[Figure 2568]

[Figure 2569]

[Figure 2570]

[Figure 2571]

[Figure 2572]

[Figure 2573]

[Figure 2574]

The Dark Knight, 2008, screenshot from a movie, movie scene 4K, intense, dark and gritty, urban nightscape, moody lighting.

[Figure 2575]

Sec. 2.5 Style

[Figure 2576]

Prompt

[Figure 2577]

[Figure 2578]

FLUX.1

Midjourney

[Figure 2579]

Stable Diffusion 3 Jimeng

[Figure 2580]

[Figure 2581]

Ideogram2.0

[Figure 2582]

DALLE3

[Figure 2583]

[Figure 2584]

[Figure 2585]

[Figure 2586]

[Figure 2587]

A screenshot of a classic video game interface, featuring pixelated characters and a retro 8-bit art style. The game interface shows a health bar, score counter, and inventory icons on the top of the screen, while the background features a colorful, blocky environment with simple textures. Bright, primary colors dominate the palette, creating a nostalgic and fun atmosphere. Created Using: pixel art, retro game aesthetics, vibrant colors, simple UI design.

[Figure 2588]

[Figure 2589]

[Figure 2590]

[Figure 2591]

[Figure 2592]

Prompt

[Figure 2593]

A vivid photograph of a smartphone screen displaying a stunning waterfall. The water cascades down a towering cliff into a crystal-clear pool below, with mist rising from the impact. The phone screen captures the vivid details of the flowing water and the lush greenery surrounding the scene. Soft sunlight filters through the trees, adding warmth to the cool misty air. Created Using: realism, intricate detail, vibrant colors, dynamic composition.

[Figure 2594]

[Figure 2595]

Style Example: Movie Screenshot Style Example: Digital Image Representation Style Example: Video Game Interface

[Figure 2596]

[Figure 2597]

[Figure 2598]

- Figure 73: Results on different style image generation task. Refer to Section 2.5 for detailed

###### 3 Conclusion

###### 3.1 Summary

In this report, we conducted a comprehensive evaluation of six powerful text-to-image models, termed IMAGINE-E, including FLUX.1, Ideogram2.0, Dall-E3, Midjourney, Stable Diffusion 3, and Jimeng. We extensively explored the performance of these models across various levels of difficulty, including qualitative samples and quantitative benchmarks. To make our evaluation more thorough, we carefully collected numerous samples covering six aspects: structured output generation, realism and physical consistency tasks, specific domain generation, challenging scenario generation, and different style image generation. Each domain also includes several more detailed subtasks for in-depth discussion and analysis.

###### 3.2 Task Complexity Analysis

In this work, unlike traditional text-to-image generation, we focus on exploring and tackling challenging and specialized image generation domains to test the robustness and potential of generic T2I models.

Currently, all models still face difficulties in code generation tasks, including generating simple Python code, QR codes, and barcodes, indicating that applying text-to-image models to general code generation remains a significant challenge. Moreover, in 3D generation tasks, all models perform poorly, and the generated images cannot be directly used for 3D-related applications. In structured output tasks, none of the models can accurately follow prompts to generate images or tables. Although FLUX.1 can generate chart images that are close to correct, there are still discrepancies in detail compared to the prompts. Additionally, all models are unable to output images containing Chinese text.

However, all models can accept JSON-format inputs and generate images according to the specified JSON content. In some basic tasks, such as generating images in different styles or based on photographic terminology, the models perform well and can produce high-quality images following the prompts.

In summary, the current performance of these models in specific image generation domains still faces some common bottlenecks: on the one hand, there are certain unachieved functionalities or difficult tasks to overcome, and on the other hand, there are some basic capabilities that have already been realized. Further analysis is needed to identify the challenges of each task in the current design.

###### 3.3 Model Performance Evaluation

FLUX.1 and Ideogram2.0. Overall, FLUX.1 and Ideogram2.0 perform the best. In structured output tasks, FLUX.1’s outputs can closely match the images, tables, and web pages described in the prompts. In realism and physical consistency tasks, both models can largely address issues of human deformities and have a basic understanding of the fundamental laws of the physical world, though the generated images still exhibit some logical inconsistencies. In specific domain generation tasks, these two models excel in understanding foundational knowledge in disciplines such as chemistry, biology, and medicine, showing potential as general models, and can generate high-quality data for autonomous driving and embodied intelligence tasks. In challenging scenario generation tasks, FLUX.1 and Ideogram2.0 perform exceptionally well in generating content with dense text, accepting inputs in different languages and emojis.

Midjourney. Midjourney produces images with the best aesthetic appeal, particularly evident in generating images of different styles. In everyday contexts, Midjourney’s images are more visually pleasing, making them more practical.

Dall-E3. Dall-E3 has a profound understanding of the physical world and boasts higher safety measures, being sensitive to copyright, watermarks, and NSFW prompts and images. Additionally, the images generated by Dall-E3 have a distinct style that sets them apart from all other models.

Stable Diffusion 3 and Jimeng. Stable Diffusion 3 and Jimeng do not match FLUX.1 in overall generation quality. Jimeng is particularly sensitive to special symbols in prompts and also has high safety measures.

###### 3.4 Quantitative Benchmark Assessment

Currently, the concentrated quantitative evaluation benchmarks, such as CLIPScore, HPSv2, and Aesthetic Score, cannot reasonably assess model outputs in more challenging tasks. The evaluation results from GPT-4o are more comprehensive and reasonable; however, in tasks that require comparing image details for evaluation, GPT-4o’s results still significantly differ from human intuitive perceptions.

###### References

- [1] Ayman Abaid, Muhammad Ali Farooq, Niamh Hynes, Peter Corcoran, and Ihsan Ullah. Synthesizing cta image data for type-b aortic dissection using stable diffusion models. 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pages 1–5, 2024.
- [2] Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. Unified pretraining for program understanding and generation. ArXiv, abs/2103.06333, 2021.
- [3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022.
- [4] Bortik Bandyopadhyay, Xiang Deng, Goonmeet Bajaj, Huan Sun, and Srinivasan Parthasarathy. Automatic table completion using knowledge base. ArXiv, abs/1909.09565, 2019.
- [5] Dmitry Baranchuk, Andrey Voynov, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Label-efficient semantic segmentation with diffusion models. In International Conference on Learning Representations, 2022.
- [6] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [7] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [8] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.
- [9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.
- [10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 2020.
- [11] Shengqu Cai, Eric Chan, Yunzhi Zhang, Leonidas Guibas, Jiajun Wu, and Gordon Wetzstein. Diffusion self-distillation for zero-shot customized image generation. arXiv preprint arXiv:2411.18616, 2024.
- [12] Anthony Chen, Jianjin Xu, Wenzhao Zheng, Gaole Dai, Yida Wang, Renrui Zhang, Haofan Wang, and Shanghang Zhang. Training-free regional prompting for diffusion transformers. arXiv preprint arXiv:2411.02395, 2024.
- [13] Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Topiq: A top-down approach from semantics to distortions for image quality assessment. IEEE Transactions on Image Processing, 33:2404–2418, 2023.

- [14] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé, Jared Kaplan, Harrison Edwards, Yura Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, David W. Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William H. Guss, Alex Nichol, Igor Babuschkin, Suchir Balaji, Shantanu Jain, Andrew Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew M. Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. ArXiv, abs/2107.03374, 2021.
- [15] Pei Chen, Fudong Wang, Yixuan Tong, Jingdong Chen, Ming Yang, and Ming Yang. Graphicsdreamer: Image to 3d generation with physical consistency. 2024.
- [16] Yukang Chen, Tong Yang, X. Zhang, Gaofeng Meng, Xinyu Xiao, and Jian Sun. Detnas: Backbone search for object detection. In Neural Information Processing Systems, 2019.
- [17] Chin-Yi Cheng, Ruiqi Gao, Forrest Huang, and Yang Li. Colay: Controllable layout generation through multi-conditional latent diffusion. ArXiv, abs/2405.13045, 2024.
- [18] Victor C. Dibia and Ça˘gatay Demiralp. Data2vis: Automatic generation of data visualizations using sequence-to-sequence recurrent neural networks. IEEE Computer Graphics and Applications, 39:33–46, 2018.
- [19] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. Codebert: A pre-trained model for programming and natural languages. ArXiv, abs/2002.08155, 2020.
- [20] FLUX. Flux, 2024. Accessed: 2024-01-20.
- [21] Peng Gao*, Jiaming Han*, Renrui Zhang*, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023.
- [22] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, Renrui Zhang, Junlin Xi, Wenqi Shao, Zhengkai Jiang, Tianshuo Yang, Weicai Ye, He Tong, Jingwen He, Yu Jiao Qiao, and Hongsheng Li. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. ArXiv, abs/2405.05945, 2024.
- [23] Xianfan Gu, Chuan Wen, Jiaming Song, and Yang Gao. Seer: Language instructed video prediction with latent diffusion models. ArXiv, abs/2303.14897, 2023.
- [24] Ziyu Guo, Renrui Zhang, Hao Chen, Jialin Gao, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. Sciverse: Multimodal scientific benchmark for large models, 2024.
- [25] Ziyu Guo, Renrui Zhang, Longtian Qiu, Xianzhi Li, and Pheng Ann Heng. Joint-mae: 2d-3d joint masked autoencoders for 3d point cloud pre-training. IJCAI 2023, 2023.
- [26] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615, 2023.
- [27] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Chengzhuo Tong, Peng Gao, Chunyuan Li, and Pheng-Ann Heng. Sam2point: Segment any 3d as videos in zero-shot and promptable manners. arXiv preprint arXiv:2408.16768, 2024.
- [28] Jiaming Han*, Renrui Zhang*, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, et al. Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905, 2023.

- [29] Hwan Heo, Jangyeong Kim, Seongyeong Lee, Jeonga Wi, Junyoung Choi, and Sangjun Ahn. Capa: Carve-n-paint synthesis for efficient 4k textured mesh generation. 2025.
- [30] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [31] Lianghua Huang, Wei Wang, Zhigang Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. ArXiv, abs/2410.23775, 2024.
- [32] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. In International Conference on Machine Learning, pages 13916–13932. PMLR, 2023.
- [33] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided reconstruction of lifelike clothed humans. 2024 International Conference on 3D Vision (3DV), pages 1531–1542, 2023.
- [34] Zixuan Huang, Mark Boss, Aaryaman Vasishta, James M. Rehg, and Varun Jampani. Spar3d: Stable point-aware reconstruction of 3d objects from single images. 2025.
- [35] Mude Hui, Zhizheng Zhang, Xiaoyi Zhang, Wenxuan Xie, Yuwang Wang, and Yan Lu. Unifying layout generation with a decoupled diffusion model. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1942–1951, 2023.
- [36] Ideogram2.0. Ideogram2.0, 2024. Accessed: 2024-08-21.
- [37] Dongzhi Jiang, Guanglu Song, Xiaoshi Wu, Renrui Zhang, Dazhong Shen, Zhuofan Zong, Yu Liu, and Hongsheng Li. Comat: Aligning text-to-image diffusion model with image-to-text concept matching. arXiv preprint arXiv:2404.03653, 2024.
- [38] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.
- [39] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492–9502, 2024.
- [40] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5128–5137, 2021.
- [41] Hyeonwoo Kim, Sangwon Beak, and Hanbyul Joo. David: Modeling dynamic affordance of 3d objects using pre-trained video diffusion models. 2025.
- [42] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021.
- [43] Hsin-Ying Lee, Hung-Yu Tseng, and Ming-Hsuan Yang. Exploiting diffusion prior for generalizable dense prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7861–7871, 2024.
- [44] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [45] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.

- [46] Xueting Li, Ye Yuan, Shalini De Mello, Gilles Daviet, Jonathan Leaf, Miles Macklin, Jan Kautz, and Umar Iqbal. Simavatar: Simulation-ready avatars with layered hair and clothing. 2024.
- [47] Renyang Liu, Ziyu Lyu, Wei Zhou, and See-Kiong Ng. Anid: How far are we? evaluating the discrepancies between ai-synthesized images and natural images through multimodal guidance. 2024.
- [48] Yaoyao Liu, Bernt Schiele, Andrea Vedaldi, and C. Rupprecht. Continual detection transformer for incremental object detection. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23799–23808, 2023.
- [49] Pan Lu, Liang Qiu, Wenhao Yu, Sean Welleck, and Kai-Wei Chang. A survey of deep learning for mathematical reasoning. ArXiv, abs/2212.10535, 2022.
- [50] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025.
- [51] Mohamed Lamine Mekhalfi, Davide Boscaini, and Fabio Poiesi. Leveraging confident image regions for source-free domain-adaptive object detection. 2025.
- [52] Fanqing Meng, Wenqi Shao, Lixin Luo, Yahong Wang, Yiran Chen, Quanfeng Lu, Yue Yang, Tianshuo Yang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Phybench: A physical commonsense benchmark for evaluating text-to-image models. ArXiv, abs/2406.11802, 2024.
- [53] Keita Miwa, Kento Sasaki, Hidehisa Arai, Tsubasa Takahashi, and Yu Yamaguchi. One-d-piece: Image tokenizer meets quality-controllable compression. 2025.
- [54] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models learn physical principles from watching videos? 2025.
- [55] Yongyu Mu, Hengyu Li, Junxin Wang, Xiaoxuan Zhou, Chenglong Wang, Yingfeng Luo, Qiaozhi He, Tong Xiao, Guocheng Chen, and Jingbo Zhu. Boosting text-to-image generation via multilingual prompting in large multimodal models. 2025.
- [56] OpenAI. Gpt-4o system card. ArXiv, abs/2410.21276, 2024.
- [57] Shuai Peng, Ke Yuan, Liangcai Gao, and Zhi Tang. Mathbert: A pre-trained model for mathematical formula understanding. ArXiv, abs/2105.00377, 2021.
- [58] Xiangyuan Peng, Huawei Sun, Kay Bierzynski, Anton Fischbacher, Lorenzo Servadei, and Robert Wille. Mutualforce: Mutual-aware enhancement for 4d radar-lidar 3d object detection. 2025.
- [59] Alec Radford. Improving language understanding by generative pre-training. 2018.
- [60] Alec Radford, Jongwook Choe, and et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.
- [61] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019.
- [62] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. ArXiv, abs/2204.06125, 2022.
- [63] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021.
- [64] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [65] Subham Sah, Rishab Mitra, Arpit Narechania, Alex Endert, John Stasko, and Wenwen Dou. Generating analytic specifications for data visualization from natural language queries using large language models. ArXiv, abs/2408.13391, 2024.
- [66] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [67] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Seyedeh Sara Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. ArXiv, abs/2205.11487, 2022.
- [68] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models. ArXiv, abs/2210.08402, 2022.
- [69] Chaehun Shin, Jooyoung Choi, Heeseung Kim, and Sungroh Yoon. Large-scale text-toimage model with inpainting is a zero-shot subject-driven image generator. arXiv preprint arXiv:2411.15466, 2024.
- [70] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024.
- [71] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In The Eleventh International Conference on Learning Representations, 2023.
- [72] Yue Wang, Weishi Wang, Shafiq R. Joty, and Steven C. H. Hoi. Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. ArXiv, abs/2109.00859, 2021.
- [73] Zelun Wang and Jyh-Charn S. Liu. Translating mathematical formula images to latex sequences using deep neural networks with sequence-level training. ArXiv, abs/1908.11415, 2019.
- [74] Jialiang Wei, Anne-Lise Courbis, Thomas Lambolais, Gérard Dray, and Walid Maalej. On ai-inspired ui-design. ArXiv, abs/2406.13631, 2024.
- [75] Jialiang Wei, Anne-Lise Courbis, Thomas Lambolais, Binbin Xu, Pierre Louis Bernard, and Gérard Dray. Boosting gui prototyping with diffusion models. 2023 IEEE 31st International Requirements Engineering Conference (RE), pages 275–280, 2023.
- [76] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. ArXiv, abs/2306.09341, 2023.
- [77] Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, and Luc Van Gool. Diffir: Efficient diffusion model for image restoration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13095–13105, 2023.
- [78] Sen Xing, Muyan Zhong, Zeqiang Lai, Liangchen Li, Jiawen Liu, Yaohui Wang, Jifeng Dai, and Wenhai Wang. Mulan: Adapting multilingual diffusion models for hundreds of languages with negligible cost. ArXiv, abs/2412.01271, 2024.
- [79] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2966, 2023.

- [80] Lei Xu and Kalyan Veeramachaneni. Synthesizing tabular data using generative adversarial networks. ArXiv, abs/1811.11264, 2018.
- [81] M. Xu and et al. Aesthetic image classification using deep learning. In Proceedings of the 2016 ACM on Multimedia Conference, pages 471–475. ACM, 2016.
- [82] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chun yue Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. ArXiv, abs/2310.11441, 2023.
- [83] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [84] Beiyuan Zhang, Yue Ma, Chunlei Fu, Xinyang Song, Zhenan Sun, and Ziqiang Li. Follow-yourmultipose: Tuning-free multi-character text-to-video generation via pose guidance. 2024.
- [85] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.
- [86] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [87] Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Yu Qiao, Hongsheng Li, and Peng Gao. Llama-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In ICLR 2024, 2024.
- [88] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? ECCV 2024, 2024.
- [89] Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Hao Dong, Peng Gao, and Hongsheng Li. Personalize segment anything model with one shot. ICLR 2024, 2023.
- [90] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, et al. Mavis: Mathematical visual instruction tuning with an automatic data engine. arXiv preprint arXiv:2407.08739, 2024.
- [91] Weixia Zhang, Kede Ma, Jia Yan, Dexiang Deng, and Zhou Wang. Blind image quality assessment using a deep bilinear convolutional neural network. IEEE Transactions on Circuits and Systems for Video Technology, 30:36–47, 2019.

