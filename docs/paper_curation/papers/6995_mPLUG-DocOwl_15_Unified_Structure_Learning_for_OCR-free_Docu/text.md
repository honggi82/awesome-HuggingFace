# arXiv:2403.12895v1[cs.CV]19Mar2024

## mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding

Anwen Hu1, Haiyang Xu1∗, Jiabo Ye 1, Ming Yan1∗ Liang Zhang2, Bo Zhang1, Chen Li1, Ji Zhang1, Qin Jin2, Fei Huang1, Jingren Zhou1 1Alibaba Group 2Renmin University of China {huanwen.haw,shuofeng.xhy,ym119608}@alibaba-inc.com

[Figure 1]

[Figure 2]

Figure 1: Compared with similar-size generalists, our DocOwl 1.5 achieves state-of-the-art OCR-free performance on 10 Visual Document Understanding benchmarks.

### Abstract

Structure information is critical for understanding the semantics of text-rich images, such as documents, tables, and charts. Existing Multimodal Large Language Models (MLLMs) for Visual Document Understanding are equipped with text recognition ability but lack general structure understanding abilities for text-rich document images. In this work, we emphasize the importance of structure information in Visual Document Understanding and propose the Unified Structure Learning to boost the performance of MLLMs. Our Unified Structure Learning comprises structureaware parsing tasks and multi-grained text localization tasks across 5 domains: document, webpage, table, chart, and natural image. To better encode structure information, we design a simple and effective vision-to-text module H-Reducer, which can not only maintain the layout information but also reduce the length of visual features by merging horizontal adjacent patches through convolution, enabling the LLM to understand high-resolution images more efficiently. Furthermore, by constructing structure-aware text sequences and multi-grained pairs of texts and bounding boxes for publicly available text-rich images, we build a comprehensive

∗Corresponding authors

Preprint. Under review.

[Figure 3]

|[Figure 4]<br><br>|
|---|

[Figure 5]

(b) which edition has unlimited remote desktop services and virtulization rights? Datacenter

|[Figure 6]|
|---|

|[Figure 7]<br><br>| |
|---|
|
|---|

(c) What is the title of the paper in the website? Balance Control in Obese Subjects during Quiet Stance: A state-of-the Art.

(a) What is the assigned response code? W24

[Figure 8]

|[Figure 9]|
|---|

(e) What is the forecast for the increase in customs duty revenue in 2030? 100

(f) What is the percentage of Iraqi dependents citizen? 0.19.

(d) What percentage of teenagers from the age group 12-17 didn't use the Internet? 5%

- Figure 2: Illustrations of the importance of structure information in Visual Document Understanding on documents (a), tables (b), webpages (c), infographics (d), and charts (e-f).

training set DocStruct4M to support structure learning. Finally, we construct a small but high-quality reasoning tuning dataset DocReason25K to trigger the detailed explanation ability in the document domain. Our model DocOwl 1.5 achieves state-of-the-art performance on 10 visual document understanding benchmarks, improving the SOTA performance of MLLMs with a 7B LLM by more than 10 points in 5/10 benchmarks. Our codes, models, and datasets are publicly available at https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5.

### 1 Introduction

Leveraging the strong language understanding and generation ability of Large Language Models (LLM) [5, 46, 48, 62], some recent works [57, 58, 27, 26, 64, 24] have developed Multimodal Large Language Models (MLLMs) for general vision-and-language understanding. By aligning a pre-trained visual encoder (e.g. the ViT/L-14 [12] from CLIP [36]) and the LLM with a Vision-toText (V2T) module, these models present promising performance on understanding general images. However, they still face great challenges with images with rich text information, such as documents, webpages, tables, and charts [28]. This is mainly because the visual encoder and V2T module are trained on general image-text pairs and not specifically optimized to represent the textual and structural information in text-rich images.

Textual information in images manifests with a multitude of visual structures, spanning the simplicity of plain text to the systematic grid layouts of tables and incorporating a spectrum of graphical

representations such as pie, line, and bar charts. These elements may appear in isolation or be intricately interwoven within the framework of documents and webpages, reflecting a rich diversity of informational architecture across posters, invoices, infographics, scientific reports, academic and news websites, etc. As shown in Fig. 2, besides the basic textual content, structure information also plays a big role in Visual Document Understanding [53, 18, 45, 23]. With basic abilities to understand general images and comprehend structured texts through the LLM decoder, MLLM has the potential to achieve unified structure learning on text-rich images. For better Visual Document Understanding with MLLMs, some works [55, 56, 3, 13] attempt to design text-reading tasks to strengthen the text recognition ability, but either ignore the structure comprehension or only cover limited domains of text-rich images, such as just webpages [23] or documents [13]. In this work, we first propose to perform unified structure learning on text-rich images for MLLMs across 5 domains: document, webpage, table, chart, and natural image.

For better structural understanding, we first design a simple and effective vision-to-text module, namely H-Reducer. Unlike the Resampler [1] or Q-former [24] which fuses visual features with learnable queries but affects spatial information, the H-Reducer accumulates neighborhood visual features through convolution to keep the relative positional relationships. Compared with V2T modules with only linear layers [27, 26], it produces much fewer visual features, which is more efficient for LLM to understand high-resolution document images. Considering texts in document images are most organized from left to right, H-Reducer merges visual features at the horizontal level. Our Unified Structure Learning comprises structure-aware parsing tasks and multi-grained text localization tasks. To learn the organization of text contents, the former mainly teaches the model to parse the texts in the image in a structure-aware style, such as using line feeds and spaces to represent the structure of documents or webpages, and using extended Markdown syntax to represent the structure of tables and charts. Multi-grained text localization tasks further enhance the ability to correlate visually situated texts and concrete positions in the image. To support unified structure learning, based on publicly available datasets, we carefully build a comprehensive training set DocStruct4M by constructing structure-aware sequences and multi-grained pairs of text and bounding boxes. The DocOwl 1.5 is trained in a two-stage framework, starting with the Unified Structure Learning and then followed by the Multi-task Tuning among downstream tasks. Finally, to trigger the reasoning ability of MLLM in Visual Document Understanding, we construct a high-quality instruction tuning dataset DocReason25K. By performing joint training on DocReason25K and downstream datasets, DocOwl 1.5-Chat well balance giving a simple answer or detailed explanations.

Our contributions in this work are four-fold:

- • We first propose Unified Structure Learning on text-rich images for MLLMs and design both structure-aware parsing tasks and multi-grained text localization tasks across 5 domains. A comprehensive dataset DocStruct4M is carefully built to support Unified Structure Learning.
- • We design a simple and effective vision-to-text module for structure learning and perform extensive experiments to validate its effectiveness.
- • We construct a high-quality instruction tuning set to trigger the reasoning ability of MLLMs on Visual Document Understanding.
- • DocOwl 1.5 and DocOwl 1.5-Chat achieves state-of-the-art OCR-free performance on 10 Visual Document Understanding tasks, achieving improvement of more than 10 points on 5/10 tasks among similar-sized models.

### 2 Related Work

Visual Document Understanding(VDU), also known as Visually-situated Language Understanding [23, 56], aims to comprehend images with rich text information. Such images range from documents [30, 31, 42, 41, 60], tables [34, 8, 63], charts [29, 19, 32, 21, 44, 17], natural images [39, 40, 16] to webpage screenshots [43, 9], where diverse composition of text and visual objects contains a wealth of information. To evaluate the multimodal document understanding performance, the task formats include low-level recognition, e.g. information extraction [42, 41], and high-level semantic understanding, such as visual question answering [30, 31, 34, 29, 43, 40], image captioning [39, 21, 44], and natural language inference [8]. According to whether relying on an off-the-shelf OCR system to recognize texts in the image, models for Visual Document Understanding can be categorized into OCR-dependent models [45, 53, 18, 54] and OCR-free ones [22, 23]. To leverage recognized texts

from an OCR system, OCR-dependent models are always trained to align textual and visual inputs. For example, UDOP [45] is pre-trained to recover masked text and layout information given image and retained text as inputs. As for OCR-free methods, training with tasks about text recognition is indispensable. Dount [22] design the text reading task to output continuous text sequences that ignore structure information. To leverage structure information, Pix2Struct [23] designs a Screenshot Parsing Task to generate the HTML DOM tree for webpage screenshots but is hard to apply to other types of images. In this work, we first propose Unified Structure Learning for all image types and carefully build a comprehensive dataset to support layout learning.

Multimodal Large Language Models(MLLM) have shown strong vision understanding and openended conversation abilities [57, 58, 64, 10, 3, 15, 59] for natural images. They follow the architecture paradigm of connecting a vision encoder,e.g. ViT [12, 36], with a Large Language Model(LLM) [46, 48, 2] by a vision-to-text module, such as simple linear layers [27, 26] or a Q-Former [24]/Resampler [1]/Abstractor [57, 58] with learnable queries. To enable MLLMs to comprehend images with rich texts, there are major two challenges: how to encode high-resolution images and how to understand visually-situated texts. To tackle high-resolution images, most works choose to further train [3, 13] or extraly add a high-resolution vision encoder [15]. UReader [56] first proposes to keep the low-resolution vision encoder and use a shape-adaptive cropping module to crop raw images into multiple sub-images with low resolution. To enhance the visually-situated text understanding, some work design tasks of reading texts from top-left to bottom-right without taking into account the importance of structure [56, 3]. CogAgent [15] and DocPedia [13] further try strengthening the layout understanding for documents, webpages, and natural images with text grounding tasks. However, the comprehension of the overall structure is ignored, and tables and charts are not covered. In this work, we follow UReader to process high-resolution images. To strengthen structure understanding, we design structure-aware praising and multi-grained text localization tasks for all types of images, covering documents, tables, charts, webpages, and natural images. We propose a vision-to-text architecture to better maintain spatial information of visual features by convolution. Finally, to support unified structure learning, we build a comprehensive training dataset DocStruct4M and greatly improve the visual document understanding performance.

### 3 DocOwl 1.5

DocOwl 1.5 follows the typical architecture of Multimodal Large Language Models, which consists of a visual encoder, a vision-to-text module, and a large language model as the decoder. To better keep the textual and layout information in text-rich images of high resolution, we design an H-Reducer as the vision-to-text module to ensemble horizontal visual features. As shown in Fig. 3(a), to enhance the text recognition and structure understanding abilities, we first perform Unified Structure Learning with structure-aware parsing and multi-grained text localization tasks for all types of images. Then, the model is jointly tuned on multiple downstream tasks of Visual Document understanding.

#### 3.1 Model Architecture

High-resolution Image Encoding. As proved by previous works [22, 23, 56], the ability to encode high-resolution images is critical to ensuring that the decoder can use rich text information from document images. As shown in Fig. 3(b), following UReader [56] , we utilize a parameter-free Shape-adaptive Cropping Module to crop a shape-variable high-resolution image I into multiple fixed-size sub-images (I1,I2,...,IC), where C is the number of crops. To keep the overall layout information, the raw image is also resized to a low-resolution one as the global image I0. Then, each image Ii in (I0,I1,...,IC) is independently encoded to a sequence of visual features Vi = (vi1,vi2,...,viL),0 ≤ i ≤ C by a transformer-based Visual Encoder, where vij,1 ≤ j ≤ L is a D-dimension vector, L is the length of visual features for each image.

Spatial-aware Vision-to-Text Module: H-Reducer. There are two kinds of popular vision-to-text modules for Multimodal Large Language Models: a MLP [27, 26, 64] or a cross-attention module with learnable queries [57, 3, 1, 24]. Both two are not quite suitable for representing high-resolution text-rich images. The former projects complete visual features into the language embedding space. It maintains all spatial information in the document image but keeps the sequence length of raw visual features, which is too long when processing high-resolution images. For example, encoding a 1,344x1,344 image with the ViT/L-14 results in 9,216 visual tokens. The cross-attention module

[Figure 10]

[Figure 11]

[Figure 12]

Document Parsing

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

VQA

|ViT<br><br>LLM<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>H-Reducer|
|---|

|ViT<br><br>LLM<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>H-Reducer|
|---|

[Figure 24]

Table Parsing Chart Parsing Natural Image Parsing

[Figure 25]

Information Extraction

[Figure 26]

[Figure 27]

Image Captioning

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Multi-grained Text Localization

Natural Language Inference

Stage 1: Unified Structure Learning Stage 2: Multi-task Tuning

- (a)

- (b)

[Figure 34]

|Twitter. According to the image, there are 560 million and 70 million active users for Twitter and Pinterest. Thus, Twitter has more active users.|
|---|

[Figure 35]

Large Language Model MAM

𝑇 𝑇 𝑇 𝑇 𝑇 𝑇 𝑇 𝑋

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |

H-Reducer

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |

<row1-col1> <row1-col2> <row1-col3> <row2-col1> <row2-col2> <row2-col3>

img>

Convolution (1x4)

𝑉 𝑉 𝑉 𝑉 𝑉 𝑉 𝑉

<global-

H-Reducer

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

𝑉 𝑉 𝑉 𝑉 𝑉 𝑉 𝑉

FC

Visual Encoder

[Figure 36]

[Figure 37]

[Figure 38]

flatten

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

𝐼 𝐼 𝐼 𝐼 𝐼 𝐼 𝐼

| | |
|---|---|
|[Figure 43]| |

Shape-Adaptive Cropping Module

𝑉

𝑉

𝑉

[Figure 44]

|Who has more active users, Pinterest or Twitter?|
|---|

- Figure 3: The two-stage training framework (a) and overall architecture (b) of DocOwl 1.5. The global image and cropped images are processed independently by the Visual Encoder and H-Reducer. <rowx-coly> is the special textual token to indicate that the position of the cropped image in the original image is the xth row and yth column.

could greatly reduce the length of the visual sequence to the number of learnable queries, but may lose spatial information during semantic fusion.

In this work, we design a more appropriate vision-to-text module for Visual Document Understanding, namely H-Reducer, which not only reduces visual sequence length but also keeps the spatial information. As shown in Fig. 3(b), the H-Reducer is comprised of a convolution layer to reduce sequence length and a fully-connected layer to project visual features to language embedding space. Since most textual information in document images is arranged from left to right, the horizontal text information is usually semantically coherent. Thus, the kernel size and stride size in the convolution layer are set as 1x4 to ensemble horizontal 4 visual features. The output channel is set equal to the input channel D. The convolution calculation is as follows:

Vi = (vi1,vi2,...,viL) (1) vji = f(vi4j−3,vi4j−2,vi4j−1,vi4j),1 ≤ j ≤ L/4, (2)

V i = (v1i,v2i,...,vL/i 4), (3) where f represents the dot product with kernel weights on multiple channels. After the convolution layer, the visual features of image Ii are converted to the V i, the feature length of which is L/4.

Then, with a fully connected layer to align visual features to the language embedding space, the V i are transferred to Vˆi = (ˆvi1,vˆi2,...,vˆiL/4).

Multimodal Modeling with LLM. As the decoder of MLLM, large language models should understand both the visual features of images and the textual features of language instructions. Following

Document Parsing Natural Image Parsing

|emotional players celebrate at the final whistle of their game against<br><br>football team <ocr> QATAR AIRWAYS QATAR 15 CAP </ocr>|
|---|

|[Figure 45]|
|---|

|<doc> Universities and departments \n Students are members of the \n University, a department and a \n College.\n<br><br>•Course content \n<br>•Lectures,…</doc><br>|
|---|

[Figure 46]

Describe the content and text within the image.

Read text in the image.

Table Parsing

|<md> | | <COLSPAN=2> Factors levels in actual values | | | | \n<br><br>| Formula | Pluronic to drug ratio | P123 percentage (%) | EE% + SD (%)a | PS + SD (nm)a | PDI + SDa | \n<br><br>| --- | --- | --- | --- | --- | --- | \n | M1 | 10 | 10 | 7.27 ± 0.47 | 58.73 ± 1.56 | 0.23 ± 0.01 | \n … | M11 | 30 | 50 | 99.43 ± 0.73 | 23.74 ± 0.95 | 0.13 ± 0.03 | </md>|
|---|

[Figure 47]

|H-Reducer Viusal Encoder<br><br>LLM MAM<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>DocOwl 1.5|
|---|

Parse the table in Markdown style.

Chart Parsing

[Figure 53]

Convert the chart into Markdown format.

|<md> | Country | 1960 | 1961 | 1962 | 1963 | \n | --- | --- | --- | --- | --- | \n | Ecuador | 45 | 44.1 | 43.4 | 43.4 | \n … | Iran | 45.2 | 44.3 | 43.7 | 43.8 | </md>|
|---|

Text Recognition

Text Grounding

|[Figure 54]|
|---|

|<ocr> September 15, 1972 \n DATE BIOGRAPHICAL DATA \n NAME Mr. Milovan Bosnjak \n … PROFESSIONAL EXPERIENCE (In decending chronological order; position </ocr>|
|---|

Predict the bounding box of the text <ocr> 17. Loans Given 75.64 55.24 \n

[Figure 55]

Identify the text within the bounding box <bbox>212, 52, 896, 418</bbox>

- 18. Receipt towards Loan Repayment 64.11 4.64 0.13 0.11 \n
- 19. Advances Given 26.27 0.88 6.50 </ocr>

|<bbox>120, 538, 753, 584 </bbox>|
|---|

Figure 4: The illustration of Unified Structure Learning of DocOwl 1.5.

mPLUG-Owl2 [58], we apply the Modality-adaptive Module(MAM) in LLM to better distinguish visual and textual inputs. During self-attention, MAM utilizes two sets of linear projection layers to separately perform the key/value projection for visual features and textual features. To help the LLM correlate multiple cropped sub-images, UReader [56] designs learnable crop position embeddings to denote the row and column position in the raw image. In this work, we simply add special textual tokens ‘<rowx_coly>’ before the visual features of each cropped image, where x and y refer to the row and column index respectively. For the global image, the textual indicator token is ‘<global_img>’. This design eliminates the need to introduce additional parameters and is more friendly to the LLM decoder. Our experiments validate that it achieves comparable effects as the crop position embedding. Overall, the decoding of the LLM is as follows:

Y = LLM([T0;Vˆ0,T1;Vˆ1,...,TC;VˆC;X]) (4)

where [;] means the concatenation operation, C is the crop number of the image, Tj,0 ≤ j ≤ C is the textual embeddings of the special textual indicator for the global image or positions of cropped

images, Vˆj is the visual features of a global or cropped image, X is the textual embeddings of the instruction, Y is the predicted answer.

#### 3.2 Unified Structure Learning

Most Multimodal Large Language Models [27, 58, 50] are trained with image-text pairs of natural images to align the visual encoder with the LLM, such as Conceptual Captions [7], LAION [37] and COYO [6]. Initializing from such models could inherit the shallow text recognition ability, but is far from understanding complex textual and structural information in various text-rich images. In this work, to empower the comprehensive document understanding abilities of MLLM, we design a Unified Structure Learning across 5 domains, including natural images, documents, tables, charts, and webpages. It involves both structure-aware parsing tasks and multi-grained text localization tasks, as shown in Fig. 4.

Document Parsing. For representing the structure information, Pix2Struct [23] parses webpage screenshots with condensed HTML DOM trees, which are built based on the HTML source codes and are not available for other formats of documents or webpage screenshots, e.g. PDF. In documents or webpages, horizontal and vertical distances between texts form the main layout information. Therefore, to make the structure-aware parsing task applicable to most documents and webpage

screenshots, we choose to add extra line feeds(‘\n’) and spaces into the text sequence to denote different lines and horizontal distances. The greater the horizontal distance, the more space characters.

We choose CCpdf [47], RVL-CDIP [14], VisualMRC [43] and datasets encapsulated in DUEBenchmark [4] (DocVQA [30], InfoVQA [31], DeepForm [42], KLC [41], WTQ [34], TabFact [8]) to support the Document Parsing task. CCpdf [47] is a multi-lingual PDF dataset built upon webpages from Common Cramwl2, covering diverse domains of documents, such as industry, academic, and medical. In this work, we mainly focus on English Document Understanding and drop PDFs detected as other languages. RVL-CDIP contains 16 categories of industry documents, such as ‘letter’, ‘email’, and ‘scientific reports’. We further remove some categories with flipping and blurring texts, such as ‘handwritten’ and ‘form’. DUE-Benchmark is a collection of available and reformulated datasets over various document domains and layouts featuring tables, graphs, lists, and infographics. VisualMRC is a webpage screenshot dataset across 35 websites. OCR annotations in VisualMRC are aligned with local regions, thus, we follow them to utilize crops of a screenshot as input for this parsing task. For CCpdf and DUE-Benchmark, a PDF-parsing tool pdfplumber3 can be directly used to generate structure-aware text sequence with a PDF page as the input. For RVL-CDIP and VisualMRC, there are no PDF files, just annotations of bounding boxes of texts. As an alternative, akin to the LATIN-Prompt [51], we insert the line feeds and spaces by calculating and comparing the horizontal and vertical distances of bounding boxes. To avoid too many space characters resulting in sparse texts, we further limit the maximum number of consecutive spaces to 4. This strategy allows us to construct structure-aware text sequences in the same style as pdfplumber.

Table Parsing. Different from documents or webpages, tables are structured in a more standardized way, where row and column correspondences represent key-value pairs. HTML and Markdown codes are mainly two kinds of text sequences used to represent a table. HTML codes can represent all kinds of tables, with or without cells spanning multiple rows and grids, but they contain too many paired labels (e.g. ‘<tr></tr>’ and ‘<td></td>’), causing text sequences to be too long. Markdown codes can represent a table with concise text sequence, but they cannot represent cells spanning multiple rows and columns. To represent all tables with concise text sequence, we follow the main grammar of Markdown to represent table structure with ‘|’ and line feeds(‘\n’). To represent cells spanning multiple rows and columns, we add special text tokens ‘<COLSPAN=x>’ and ‘<ROWSPAN=y>’ before the value, as shown in Fig. 4.

We choose TURL [11] and PubTabNet [63] to do the structure-aware table parsing task, where tables are collected from Wikipedia pages and scientific articles, respectively. Without cells across rows and columns, tables in TURL can be directly represented with Markdown codes. Due to lacking table images in TURL, we transfer tables into HTML codes and render table images with variations in background color and font size. PubTabNet contains pairs of table images and HTML codes. We convert HTML codes into Markdown style and add ‘<ROWSPAN=x>’ or ‘<COLSPAN=y>’ before the value when attributes ‘rowspan=x’ or ‘colspan=y’ are set in the ‘<td>’ label.

Chart Parsing. Unlike documents and tables, organizing texts in reading order cannot represent the structure of charts. Considering that the chart is a visualization form of the table, parsing charts to tables could best maintain the mathematical characteristics of the chart. This requires the model to understand the structure of the chart and the alignment of the x/y axis. Besides, to keep consistent with the Table Parsing task, we also use Markdown codes to represent the data tables of charts, as shown in Fig. 4.

We adopt PlotQA [32], FigureQA [20], DVQA [19], and ChartQA [29] to support the structureaware chart parsing task. These datasets cover charts on both synthetic [20, 19] data and data from real-world sources [32, 29]. Chart types include vertical bar, horizontal bar, line, dot line, and pie chart. Source data of the chart is provided in the JSON [32, 20, 32] or CSV format [29], both can be conveniently converted to Markdown codes. However, some raw values are not suitable as standard answers for parsing because there are too many significant digits to be represented on the chart. Therefore, to reduce the difficulty of estimating values and make the model focus more on structural understanding, we keep 4 significant digits for all values.

Natural Image Parsing. Quite different from text-dominant images mentioned above, the semantics of natural images is a combination of natural objects and scene texts. Thus, parsing natural images is

- 2https://commoncrawl.org
- 3https://github.com/jsvine/pdfplumber

[Figure 56]

[Figure 57]

Figure 5: Detailed statistics of DocStruct4M.

necessary to organize scene texts and mention the main image content. Manually annotating captions to describe the relationship between objects and scene texts is labour- and financial-intensive. Like TAP [54], we concatenate the general caption with OCR texts to form the target parsing sequence.

We utilize OCR-CC [54] to support the Natural Image Parsing task. OCR-CC is a subset of Conceptual Caption [38], which contains images with scene texts detected by the Microsoft Azure OCR system.

Multi-grained Text Localization. As proved in previous works [52, 49, 35] on general image understanding, semantic comprehension and object grounding tasks can be well unified in a single model. For Visual Document Understanding, structure-aware parsing tasks mainly focus on organizing texts according to the overall structure, while neglecting the correspondence between specific texts and local positions. Correlating texts with the concrete position in images is another basic structure understanding ability for visual documents. To support text position learning, we design two symmetrical tasks, namely Multi-grained Text Grounding and Multi-grained Text Recognition. The former aims to predict the bounding box given the visually-situated texts, while the latter does the opposite. We set four granularities of texts for these two tasks: word, phrase, line, and block. The ‘word’ is the smallest granularity of the bounding box, referring to only 1 word. To ensure that the word is visible and the answer is unique, words that are too small (normalized area < 0.001) and words that appear multiple times in the same image are excluded from candidates. The ‘line’ consists of texts that are judged to be horizontally parallel by vertical distance, and the ‘phrase’ is comprised of multiple adjacent words within the same line. The ‘block’ is a combination of multiple successive lines, ranging from 2 to half of the total lines. The text sequences of word-level and phrase-level question answering are much shorter than the other two. Therefore, in order to learn localization more efficiently, each word-level or phrase-level sample consists of up to 5 question-answer pairs for the same image. As for the representation of bounding boxes, we transfer each continuous value in the normalized bounding box into a discrete position token, ranging from 0 to 999.

The bounding box annotation is necessary for constructing samples for Multi-grained Text Localization tasks. Therefore, we take DocVQA, InfoVQA, WTQ, TabFact, DeepForm, KLC, ChartQA, VisualMRC, and TextVQA [40] for this task, across domains of the document, table, chart, webpage, and natural image.

Overall, to support the unified structure learning for text-rich images, we build a DocStruct4M dataset by ensembling multiple training sets of publicly available datasets and constructing structure-aware text sequences or text-position pairs as the targets. The form of instructions for each task is very diverse for developing the general instruction-following ability of the model. Fig. 5 shows the detailed statistics of DocStruct4M.

#### 3.3 Multi-task Fine-tuning

Through Unified Structure Learning, models could well understand the structure of diverse document images but cannot follow users’ instructions to do different types of tasks, such as information

- Table 1: The detailed statistics of DocReason25K. The ‘Avg Length’ refers to the average token length of the answer.

DocVQA InfoVQA WTQ VisualMRC ChartQA TextVQA ALL

Image 1,491 1,614 850 1,927 1,252 1,612 8,746 Sample 5,119 5,421 5,994 5,263 1,827 2,253 25,877 Avg Length 79.2 95.4 77.7 103.4 106.9 88.0 89.9

extraction or image captioning. So, we further perform multi-task fine-tuning to train a generalist of visual document understanding as UReader [56].

#### 3.4 Training Paradigm

As shown in Fig. 3(a), DocOwl 1.5 is trained in a two-stage framework. Considering the LLM has strong comprehension abilities for structured text [51, 61], we argue that the main limitation of MLLM in visual document understanding is the representation ability of the Visual Encoder and Vision-to-Text module for visually-situated text and structure information. Thus, during the Unified Structure Learning, we freeze the LLM parameters and tune the Visual Encoder and H-Reducer. The MAM is also optimized to help the LLM better distinguish visual features and texts parsed from the image. During the stage of Multi-task Fine-tuning, the model mainly learns how to follow the user’s instructions to give answers based on visually-situated text and structure understanding capabilities acquired in the first stage. Therefore, the Visual Encoder is frozen and other modules are tuned.

### 4 DocOwl 1.5-Chat

Existing benchmarks mainly evaluate the document understanding ability by answering the question with simple phrases and neglect detailed explanations. In this work, to better leverage the strong language reasoning ability of Large Language Models on Visual Document Understanding, we build a small instruction-tuning set with detailed explanations on text-rich image understanding, namely DocReason25K. Based on raw questions from DocVQA [30], InfoVQA [31], WTQ [34], VisualMRC [43], ChartQA [29] and TextVQA [40], we collect detailed explanations with ChatGPT4. Text contents are dominant information on documents, tables or webpage screenshots. Therefore, for DocVQA, InfoVQA, WTQ, and VisualMRC, we take the structure-aware text sequence of the image as the input to gpt-3.5-turbo-0301 and prompt it to answer the question with simple answers and detailed explanations. As for ChartQA and TextVQA, we take the image as the input and utilize the gpt-4-vision-preview to answer the question with detailed explanations. In order to filter out samples where ChartGPT answers incorrectly, we further prompt gpt-3.5-turbo-0301 to judge whether the answer given by ChartGPT is consistent with the concise human-annotated ground-truth answer. Compared with raw questions in benchmark datasets, questions in DocReason25K are added with a prompt ‘Answer the question with detailed explanation’. Detailed statistics of DocReason25K are presented in Table 1. DocOwl 1.5-Chat is trained by combining downstream datasets with DocReason25K and performing multi-task tuning after Unified Structure Learning.

### 5 Experiments

#### 5.1 Implementation Details

DocOwl 1.5 is initialized from mPLUG-Owl2 [58], which utilizes the ViT/L-14 [12] as the Visual Encoder and a 7B Large Langauge Model with the Modality Adaptive Module as the language decoder. According to the aspect ratio and resolution, each image is cropped into up to 9 sub-images with a fixed resolution of 448x448. Each sub-image is encoded to 1,024 features by the ViT/L-14 and then reduced to 256 features by the H-Reducer. The model is trained with 12,000 iterations on DocStruct4M, with the learning rate and batch size set as 1e-4 and 1,024. It costs about 128 A100 days. During the Multi-task finetuning, the model is trained for 6,500 iterations with the batch size set as 256 and the learning rate set as 2e-5. This further costs about 24 A100 days.

4https://openai.com/chatgpt

- Table 2: Different settings of OCR-free Visual Document Understanding models. ‘Open’ refers to whether all OCR learning data is open-source.

Model Init Resolution

OCR Learning Text Bbox Size Domain Open

Donut [22] - 2560x1920 ✓ × 13M Synthetic, Doc ✓ Pix2Struct [23] - 219(shape variable) ✓ × 80M Web × QwenVL [3] - 448x448 ✓ × 24.8M Synthetic, Doc, Web × Monkey [25] QwenVL [3] 896x896 × × - - UReader [56] Owl [57] 224x224(x20 crops) ✓ × 0.1M Doc, Table, Chart, Web, Natural ✓ DocPedia [13] - 2560×2560 ✓ ✓ 0.9M Doc × CogAgent [15] CogVLM [50] 1120×1120 ✓ ✓ 107M Synthetic, Nature, Doc, Web ×

DocOwl 1.5 Owl2 [58] 448x448(x9 crops) ✓ ✓ 4M Doc, Table, Chart, Web, Natural ✓

- Table 3: Comparison with OCR-free methods on various types of text-rich image understanding tasks. The superscript ‘∗’ refers to models separately fine-tuned on each downstream task, rather than generalists. The underline means the best performance among models with <10B parameters.

Tab Chart Text Text Visual VQA VQA Form Fact QA VQA Caps MRC

Doc Info Deep

Model Size

KLC WTQ

Dessurt∗ <1B 63.2 - - - - - - - - Donut∗ <1B 67.5 11.6 61.6 30.0 18.8 54.6 41.8 43.5 74.4 93.91 Pix2Struct∗

base <1B 72.1 38.2 - - - - 56.0 - 88.0 Pix2Struct∗

large 1.3B 76.6 40.0 - - - - 58.6 - 95.5 -

DocPeida 7.0B 47.1 15.2 - - - - 46.9 60.2 - DocOwl 7.1B 62.2 38.2 42.6 30.3 26.9 60.2 57.4 52.6 111.9 188.8 QwenVL 9.6B 65.1 35.4 - - - - 65.7 63.8 - UReader 7.1B 65.4 42.2 49.5 32.8 29.4 67.6 59.3 57.6 118.4 221.7 Monkey 9.8B 66.5 36.1 40.6 32.8 25.3 - - 67.6 93.2 CogAgent 17.3B 81.6 44.5 - - - - 68.4 76.1 - -

DocOwl-1.5 8.1B 81.6 50.4 68.8 37.9 39.8 80.4 70.5 68.8 132.0 239.5 DocOwl-1.5-Chat 8.1B 82.2 50.7 68.8 38.7 40.6 80.2 70.2 68.6 131.6 246.4

#### 5.2 Main Results

We evaluate the Visual Document Understanding performance on 10 text-rich image benchmarks, covering documents (DocVQA [30], InfoVQA [31], DeepForm [42], KLC [41]), tables (WTQ [34], TabFact [8]), charts (ChartQA [29]), natural images (TextVQA [40], TextCaps [39]), and webpage screenshots (VisualMRC [43]). We compare DocOwl 1.5 with state-of-the-art OCR-free models, including both Multimodal Large Language Models adapted for recognizing texts and much smaller models trained only for document understanding. The detailed comparison of model settings can be found in Table 2. As shown in Table 3, previous MLLMs with more than 7B parameters underperform domain-specific models with less than 1B parameters, showing that the document understanding is still a shortcoming for existing MLLMs. Our DocOwl 1.5 outperforms both domain-specific models and MLLMs with similar sizes on all 10 benchmarks. This validates that DocOwl 1.5 is much stronger on visual document understanding across 5 domains, covering visual question answering, information retrieval, natural language inference, and image captioning tasks. Besides, with much fewer unnatural data (3M vs 9M) and parameters (8.1B vs 17.3B), DocOwl 1.5 outperforms CogAgent [15] on InfoVQA and ChartQA, and achieves comparable performance on DocVQA. This suggests that our unified structure learning with DocStruct4M is more efficient in learning printed text recognition and how to analyze documents. However, our model still underperforms CogAgent on TextVQA, which requires the ability of scene text recognition and general knowledge about natural objects. The primary reason is that scene texts are more diverse in shapes than printed texts and CogAgent is trained on 98M samples of scene text recognition from LAION-2B [37] and COYO-700M [6], much more than the natural images (1M) in DocStruct4M. In this work, we mainly focus on improving the unified structure comprehension of visual documents and leave further scaling up data on natural scenes as future work. Finally, DocOwl 1.5-Chat can also be evaluated on these concise-answer benchmarks by removing the prompt of detailed explanation. It achieves comparable or slightly better performance than DocOwl 1.5, showing that a small amount of detailed explanatory data may better help the model understand the semantics of text-rich images.

- Table 4: Ablation study of model setting. ‘Crop’ refers to the maximum number of cropped images. ‘CropPos’ means using learnable embeddings (‘Emb’) or textual tokens (‘Text’) to represent the position of cropped images. ‘Parsing’ and ‘MTL’ refer to structure-aware parsing tasks and the Multigrained Text Location task, respectively. ‘Owl(224)’ and ‘Owl2(448)’ refer to mPLUG-Owl [57] with 224 resolution and mPLUG-Owl2 [58] with 448 resolution, respectively.

Model Architecture Structure Multi-task Tuning

DocVQA TabFact ChartQA

Init V2T Crop CropPos Learning ViT LLM

- r1 Owl(224) Abstractor 20 Emb × × × 65.4 67.6 59.3

- r2 Owl2(448) Abstractor 20 Emb × × × 66.3 69.8 60.6

- r3 Owl2(448) Abstractor 20 Emb × ✓ × 71.4 70.3 64.2

- r4 Owl2(448) Abstractor 9 Emb × ✓ × 68.0 70.0 64.2

- r5 Owl2(448) H-Reducer(1x4) 9 Emb × ✓ × 72.8 72.9 65.0

- r6 Owl2(448) H-Reducer(2x2) 9 Emb × ✓ × 71.8 72.1 65.2

- r7 Owl2(448) H-Reducer(2x4) 9 Emb × ✓ × 71.4 71.1 66.0

- r8 Owl2(448) H-Reducer(1x8) 9 Emb × ✓ × 69.9 71.2 64.4

- r9 Owl2(448) H-Reducer(2x8) 9 Emb × ✓ × 69.2 70.2 65.6

- r10 Owl2(448) H-Reducer(1x4) 9 Emb Parsing × × 77.7 76.5 67.5

- r11 Owl2(448) H-Reducer(1x4) 9 Emb Parsing × ✓ 78.9 78.1 68.1

- r12 Owl2(448) H-Reducer(1x4) 9 Text Parsing × ✓ 79.8 77.7 69.1

- r13 Owl2(448) H-Reducer(1x4) 9 Text Parsing+MTL × ✓ 81.6 80.4 70.5

#### 5.3 Ablation Study

As shown in Table 4, we further perform a comprehensive ablation study to validate the effectiveness of our H-Reducer and Unified Structure Learning.

Firstly, initializing from a stronger general MLLMs brings better performance on text-rich images (r2 vs r1), showing general vision-and-language knowledge benefits visual document understanding. Tuning the visual encoder during multi-task fine-tuning significantly improves the document understanding performance (r3 vs r2). This suggests that the visual representation of document images may be the main shortcoming of MLLMs and inspires us to design Unified Structure Learning to enhance the representation ability of the visual encoder for visually situated texts and structure.

Effectiveness of H-Reducer. When using the Shape-adaptive Cropping Module, the image resolution supported by the MLLM is the product of the cropping number and basic resolution of each crop. With the Abstractor as the vision-to-text module, reducing the cropping number causes an obvious performance decrease (r4 vs r3) on documents. However, with a smaller cropping number, the H-Reducer achieves better performance than the Abstractor (r5 vs r3), showing that 4482 × 9 ≈ 221 is an acceptable resolution for existing benchmarks and the H-Reducer is stronger on maintaining rich text information during vision-and-language feature alignment. Besides, we further compare different settings of the merging shape in the convolution layer. With the same number of merged tokens, the model with the 1x4 merging shape achieves better performance than the one with the 2x2 merging shape on document and table datasets but slightly worse performance on chart understanding (r6 vs r5). This is consistent with the common sense that documents and tables mainly organize texts in the left-to-right order while the semantic structures of charts are much more flexible. A square merging shape is more suited to encode visual features in the form of bars, lines, or pies while the 1x4 merging shape is more appropriate for general document understanding. As shown in r7-r9, further extending the 1x4 merging shape horizontally and vertically decreases the length of visual features but at the cost of performance degradation. Considering the overall performance on all text-rich images, we finally choose the 1x4 as the merging shape in H-Reducer.

Effectiveness of Unified Structure Learning. After determining the vision-to-text module, we perform two-stage training with Unified Structure Learning. With only the structure-aware parsing tasks, there is significant improvement across different domains (r10 vs r5). This validates that finetuning the visual encoder and H-Reducer with structure-aware parsing tasks greatly helps MLLMs understand text-rich images. Further tuning the parameters of LLM brings slight improvement (r11 vs r10), suggesting that general language knowledge is not the main obstacle to visual document understanding. By replacing the learnable crop position embeddings with special textual tokens, the model achieves better performance (r12 vs r11), showing that the LLM can well understand the relative positions of multiple cropped images with just simple textual indicators. Finally, by introducing Multigrained Text Localization tasks, DocOwl 1.5 achieves the best performance, validating that correlating visually situated texts with concrete positions helps comprehend documents more accurately.

- Table 5: The comparison of two-stage training and one-stage joint training with increasing samples from DocStruct4M. For a fair comparison, the LLM is frozen for both two-stage and one-stage training. The bath size of one-stage training is always set as 256, the same as the Multi-task Tuning in two-stage training.

One-Stage Two-Stage

DocStruct4M samples 0.0M 0.5M 1.0M 2.0M 4.0M 4.0M Benchmark samples 0.6M 0.6M 0.6M 0.6M 0.6M 0.6M Epoch/iteration 7/18k 6/25k 6/37k 4/40k 3/54k 3/12k + 3/6.5k Cost (A100 days) 60.0 83.3 123.3 133.3 180.0 144.8

DocVQA 72.8 75.5 78.6 78.8 78.9 79.9

Table 6: The detailed statistic of DocLocal4K.

Text Granularity Image Domain

Task

Word Phrase Line Block Doc Table Chart Web Natural Text Recognition 622 499 522 482 1,004 491 229 267 134

Text Grounding 595 542 503 485 1,011 524 240 242 108

Effectiveness of the Two-stage Training. As shown in Table 5, instead of two-stage training, we also try one-stage joint training of the structure learning and downstream tasks and gradually increase the samples from DocStruct4M. The epoch is gradually reduced because we didn’t observe performance improvements with more iterations. For joint training, the model improves significantly on DocVQA as the samples of Unified Structure Learning increase when it is below 1M. However, as the Unified Structure Learning samples are further increased, the improvement of the model becomes subtle and its performance is not as good as the one using two-stage training. This shows that the two-stage training could better enhance basic text recognition and structure parsing abilities and is more beneficial and efficient for downstream document understanding.

#### 5.4 Text Localization Evaluation

Besides proving the effectiveness of H-Reducer through downstream text-rich image understanding performance in Table 4, we further directly compare the text localization performance after the Unified Structure Learning to validate its superiority in preserving spatial features. We build a text localization evaluation set DocLocal4K with 4,250 samples balanced on 4 granularities and covering both text recognition and text grounding tasks. The detailed statistics of DocLocal4K are shown in Table 6. Considering that document images are much more diverse and complex than other images, there are more samples in this domain than others. The IOU@0.5 is used to evaluate the text grounding performance. As for text recognition, the word, phrase, line, and block granularity is evaluated with BLEU1, BLEU2, BLEU3, and BLEU4 [33], respectively. As shown in Table 7, when trained with the same iterations, the H-Reducer achieves much better performance on both Text Recognition and Text Grounding tasks, showing that H-Reducer with the 1x4 merging shape helps the LLM better understand concrete positions in images.

#### 5.5 Qualitative Results

Question Answering with Simple Phrases. Besides quantitative results, we further present some qualitative results of visual document understanding on different domains of images. As shown in

Table 7: Multi-grained text localization performance of models with different vision-to-text modules.

Text Grounding Text Recognition Word Phrase Line Block ALL Word Phrase Line Block ALL

Module Iter

Abstractor 1,800 10.92 25.83 34.59 87.01 37.69 30.68 28.58 40.12 32.73 33.03 H-Reducer(2x2) 1,800 14.19 34.87 43.94 89.07 43.94 37.20 38.33 48.68 41.99 41.55 H-Reducer(1x4) 1,800 17.82 39.30 53.28 90.52 48.28 39.60 41.84 55.37 49.84 46.66

H-Reducer(1x4) 12,000 70.42 76.38 85.88 91.34 80.38 70.10 67.86 73.88 70.70 70.63

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

|[Figure 62]<br><br>| |
|---|---|
| | |

[Figure 63]

Human: What is the forecast for the increase in customs duty revenue in 2030? UReader: 90.5 (×) DocOwl 1.5: 100 (√)

[Figure 64]

Human: What is the Dept.No?

Human: Who did clinical research? UReader: Steve Haas / Art O‘Neal (×) DocOwl 1.5: Steve peoples (√)

UReader: 76/77-142 (×) DocOwl 1.5: 218-12 (√)

（a） （b） （c）

[Figure 65]

| |
|---|

|[Figure 66]|
|---|

[Figure 67]

Human: which edition has unlimited remote desktop services and virtulization rights?

UReader: Enterprise edition (×)

DocOwl 1.5: Standard (×) Ground Truth: Datacenter

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

（d）

Figure 6: Qualitative results of DocOwl 1.5 and UReader on different domains of images.

Fig. 6(a) and (b), both models answer the question with texts in the image. DocOwl 1.5 can better understand the structure of two documents and give correct answers. In Fig. 6(c), due to the learning of parsing chart with Markdown codes, DocOwl 1.5 can better understand the chart and successfully correlate the x/y axis. Fig. 6(d) shows that although inconsistent with the ground truth, DocOwl 1.5 gives another correct answer with the help of stronger structure understanding on tables.

Question Answering with Detailed Explanations. Fig. 7 and Fig. 8 present qualitative results of detailed explanations. Through a small amount of reasoning training, DocOwl 1.5-Chat can well inherit the reasoning ability of LLM and provide detailed explanations about the answer. However, as presented in Fig. 8(c), like most general Multimoal large Language Models [57, 58, 3], DocOwl 1.5-Chat may also suffer from the hallucination problem in Visual Document Understanding. In this work, we mainly focus on enhancing the unified structure understanding ability of MLLMs and leave how to resolve the hallucination problem in OCR-free document understanding as future work.

Structure-aware Parsing. As shown in Fig. 9, DocOwl 1.5 could parse a document image by using line feeds and spaces to represent the structure of text contents. Besides parsing the whole document, as shown in Fig. 10, it could also parse texts from the middle of the image according to human instruction. Fig. 11 presents qualitative results of structure-aware table parsing through extended Markdown syntax on tables with cells spanning multiple columns or not. Furthermore, Fig. 12 shows some cases of parsing different types of charts into Markdown codes, including vertical bar, horizontal bar, pie, and line charts. When all data points are presented in the chart, DocOwl 1.5 can accurately align statistic objects with corresponding numbers. It makes some mistakes in Fig. 12(d) because estimating the concrete numbers is quite challenging when no data points are provided. Finally, as shown in Fig. 13, DocOwl 1.5 can both describe the content of natural images and read scene texts.

Multi-grained Text Localization. Fig. 14 and Fig. 15 show qualitative results of text grounding and text recognition at granularities of word, phrase, line and block. The image domains range from documents, webpages, charts, and tables to natural images.

### 6 Conclusion

To enhance the Visual Document Understanding performance of Multimodal Large Language Models, we first propose Unified Structure Learning across 5 domains of text-rich images, including both structure-aware parsing tasks and multi-grained text localization tasks. To better maintain structure and spatial information during vision-and-language feature alignment, we design a simple and effective vision-to-text module, named H-Reducer. It mainly utilizes a convolution layer to aggregate horizontally neighboring visual features. To support the Unified Structure Learning, we build a training dataset DocStruct4M by collecting publicly available images and carefully constructing structure-aware text sequences and multi-grained pairs of texts and bounding boxes. With Unified Structure Learning, our model DocOwl 1.5 achieves state-of-the-art OCR-free performance on 10 visual document understanding benchmarks.

[Figure 71]

[Figure 72]

What is the SIS # of L. MacWilliams? Answer the question with detailed explanation.

[Figure 73]

DocOwl 1.5-Chat:

SIS # 158120. The SIS # of L. MacWilliams is 158120, as stated in the text under the section "L. MacWilliams SIS # Direct Account Name Information Required PO #".

- (a)

[Figure 74]

| |
|---|
| |
|[Figure 75]|

[Figure 76]

According to the text, 95% of teenagers ages 12-17 use the Internet. Therefore, the percentage of teenagers from the age group 12-17 who didn't use the Internet would be 100% - 95% = 5%.

DocOwl 1.5-Chat:

[Figure 77]

[Figure 78]

What percentage of teenagers from the age group 12-17 didn't use the Internet? Answer the question with detailed explanation.

- (b)

[Figure 79]

[Figure 80]

what is the last date and time on the table? Answer the question with detailed explanation.

[Figure 81]

DocOwl 1.5-Chat:

The last date and time on the chart is January 1, 2013 at 4:10 PM. The chart lists all the games played by a certain team, along with the date, time, opponent, rank, site, and result. The last game listed on the chart is against Stanford, which took place on January 1, 2013 at 4:10 PM. Therefore, the last date and time on the chart is January 1, 2013 at 4:10 PM.

(c)

- Figure 7: Qualitative results of question answering with detailed explanations. Some regions are enlarged for better visualization.

[Figure 82]

| |
|---|

|[Figure 83]|
|---|

[Figure 84]

What's the value of the Very well bar in the 65+ age group? Answer the question with detailed explanation.

[Figure 85]

[Figure 86]

DocOwl 1.5-Chat:

68%. In the image, which appears to be a chart from the Pew Research Center, the bar representing "Very well" for the age group 65 and older is marked at 68%. This indicates that 68% of Republicans and Republican leaners in this age group believe that Trump is fighting for their beliefs "Very well."

- (a)

Fly Emirates. The blue shirt on the left has the word "Fly Emirates"

written across the chest. The logo is in white lettering against a blue background, which is the color scheme typically used for the airline's branding.

DocOwl 1.5-Chat:

[Figure 87]

[Figure 88]

what is written on the blue shirt on the left? Answer the question with detailed explanation.

- (b)

[Figure 89]

| |
|---|

[Figure 90]

| |
|---|
|[Figure 91]|
| |
| |

[Figure 92]

[Figure 93]

what nationality of rappers are lazy? Answer the question with detailed explanation.

[Figure 94]

DocOwl 1.5-Chat:

South African rappers are lazy. The text in the image states that "South African rappers are lazy." This is a reference to a quote from an article titled "Rap Economix" in the magazine Hype, which is South Africa's No. 1 hip hop magazine. The article discusses the perception that South African rappers are lazy and that they rely on the success of Kool & the Gang's "Stayin' Alive" instead of creating their own music. The article also mentions that this perception is changing, with new artists emerging and breaking away from the stereotype.

(c)

- Figure 8: Qualitative results of question answering with detailed explanations. Hallucination in answers are marked in red.

[Figure 95]

Extract words from the graphic. DocOwl 1.5:

|[Figure 96]|
|---|

[Figure 97]

<doc> Tropical Birding Trip Report NAMIBIA NOV DEC 2021 Tinkling Cisticola Cisticola rufilatus Seen near Kayova. Chirping Cisticola Cisticola pipiens Seen at Kayova. Piping Cisticola Cisticola fulvicapilla Seen near Kayova. Zitting Cisticola Cisticola juncidis Seen at Etosha & Windhoek. Desert Cisticola Cistcola aridulus Seen at Etosha. Reed Warblers and Allies (Acrocephalidae) Icterine Warbler Hippolais icterina Seen at Etosha & Windhoek. Sedge Warbler Acrocephalus schoenobaenus Seen at Kayova. African Reed Warbler Acrocephalus baeticus Seen at Walvis Bay & Windhoek. Lesser Swamp Warbler Acrocephalus gracilirostris Seen at Kayova & Windhoek. Swallows (Hirundinidae) Plain Martin Riparia paludicola Seen at Kayova. Bank Swallow Riparia riparia Seen at Caprivi. www.tropicalbirding.com +1-409-515-9110 info@tropicalbirding.com </doc>

##### Figure 9: A qualitative result of structure-aware document parsing.

[Figure 98]

|[Figure 99]|
|---|

Continue reading the text: <doc> 5.4. Quality Criteria for a Blended Learning Platform 5.4.1. Introduction Our objective is to create a list of absolutely necessary features for the eLearning platform (LMS) used in a blended Learning course. These features are focused on the technical and pedagogical background as well as on the learner’s situation. In addition, recommends for useful features and tools are defined that used either inside the LCMS or as additional tools. The target of the blended learning platform is the provision of high quality learning that will lead to the development of the 21st century skills. These are characterized, according to Kong et al.(2014) by three emphases: First emphasis is on skills development in both formal and informal learning contexts (Cox, 2013; Huang, Kinshuk. & Spector. 2013). The learners will be engaged in a seamless learning environment to coherently apply various generic skills for in-school teacher-led learning process initiated in digital classrooms and after-school learner-initiated learning process in social learning platforms/tools according to individual needs (Milrad, Wong, Sharpies, Hwang, Looi, & Ogata. 2013; Otero, Milrad, Rogers, Santos, Verissimo, & Tones. 2011; Wong & Looi. 2011). Second emphasis is on skills development through both individualized and collaborative learning approaches. On their own or with peers, learners take responsibilities to apply various generic skills to plan goals, implement tasks, monitor progress and evaluate outcomes in their learning process (Kicken, Brand-Gruwel, Merrienboer, & Slot. 2009; Norris & Soloway. 2009). The feedback for learners in a minimal but sufficient amount identifies individual needs and directions for future improvement (Caballero, van Riesen, Alvarez, Nussbaum, De Jong. 2014; Sims. 2003: Van Merrienboer. &

[Figure 100]

DocOwl 1.5:

Sluisjmans. 2009). Figure 5-24: Learning with Tech trends31 Third emphasis is on skills development supported by evidence of improvement and awareness of progress. The learning process in the e-learning environment can be designed in a range of activities in authentic learning contexts. Rich evidence of improvement and productive failure is collected by learners performance during the learning process; indications on applying 21st century skills for processing real-life information, reflecting on problem-solving ways, articulating tacit knowledge and 31 Adapted from Olenka Villavicencio (2013): Learning with Tech trends (Retrieved from http://olevilla.blogspot.gr/2013_07_01_archive.html) and Planning for Personalization,(retrieved from http://education.vermont.gov/plo-working-group/main). A guidance to Blended Learning 79 </doc>

- Figure 10: A qualitative result of structure-aware document parsing from the middle of the image. The red dotted box is only used to mark the location of the answers for better visualization and is not included in the input image.

|[Figure 101]|
|---|

[Figure 102]

Convert the figure into Markdown language.

[Figure 103]

DocOwl 1.5:

<md> | | <COLSPAN=2> N (%)** | <COLSPAN=2> Crude results (n = 61,129) | <COLSPAN=2> Adjusted results** (n = 47,817) | | LAN exposure | Cases | Referents | Odds ratios | 95% CI | Odds ratios | 95% CI | | --- | --- | --- | --- | --- | --- | --- | | Low | 27,121 (63) | 10,970 (60) | | Ref. | | Ref. | | Medium | 5,974 (14) | 2,623 (14) | 1.13 | (1.07-1.19) | 1.06 | (0.97-1.16) | | High | 9,659 (23) | 4,782 (26) | 1.22 | (1.18-1.28) | 1.12 | (1.04-1.20)† | </md>

(a)

[Figure 104]

| | |
|---|---|
| | |

[Figure 105]

Parse the picture content behind with markdown code: <md> | region | population centers | notes | | --- | --- | --- | | Bačka | Novi Sad , Subotica | part of Vojvodina ; partially in Hungary (Bácska) | | Gornji Breg | Gornji Breg | sub-region of Bačka | | Potisje | Bečej , Senta | sub-region of both Bačka and Banat | | Šajkaška | Temerin , Žabalj | sub-region of Bačka | | Telečka | Vrbas , Kula | sub-region of Bačka | | Banat | Zrenjanin , Pančevo | part of Vojvodina ; partially in Romania , Hungary (Bánság) and Central Serbia | | Gornje Livade | | sub-region of Banat | | Ilandžanski Rit | Ilandža , Lokve

[Figure 106]

DocOwl 1.5:

| sub-region of Banat | | Pančevački Rit | Borča , Knjača | sub-region of Banat; part of Central Serbia | | Pomorišje | Novi Kneževac , Srpski Krstur | today considered sub-region of Banat; partially in Hungary and Romania | | Repište | Knjačanin | sub-region of Banat | | Veliki Rit | north-west of Vršac | sub-region of Banat | | Srem | Belgrade , Sremska Mitrovica | part of Vojvodina; partially in Central Serbia and Croatia (Srijem) | | Crmi Lug | Progar , Boljevci | sub-region of Podlužje | | Kupinovski Kut | Kupinovo | sub-region of Podlužje | | Podlužje | Jakovo , Hrtkovci | sub-region of Srem; partially in Central Serbia | </md>

(b)

- Figure 11: Qualitative results of structure-aware table parsing on the table with cells spanning multiple columns (a) and structure-aware table parsing from the middle of the image (b). The red dotted box is only used to mark the location of the answers for better visualization and is not included in the input image.

|[Figure 107]|
|---|

[Figure 108]

Translate the chart into Markdown format.

[Figure 109]

DocOwl 1.5:

<md> | Age group | November 2010 | mid-December 2011 | mid-January 2012 | | --- | --- | --- | --- | | 18-29 | 6% | 7% | 18% | | 30-49 | 5% | 12% | 24% | | 50-64 | 9% | 11% | 19% | | 65 and older | 4% | 8% | 12% | </md>

(a)

|[Figure 110]|
|---|

[Figure 111]

Render the figure in Markdown format.

[Figure 112]

DocOwl 1.5:

<md> | Entity | No confidence | Confidence | | --- | --- | --- | | U.S | 63 | 20 | | Lebanon | 64 | 23 | | Turkey | 68 | 14 | | Tunisia | 69 | 18 | | Israel | 83 | 6 | </md>

(b)

|[Figure 113]|
|---|

[Figure 114]

Convert the image into Markdown language.

[Figure 115]

DocOwl 1.5:

<md> | Teal | Yellow Green | Light Salmon | Chartreuse | Chocolate | | --- | --- | --- | --- | --- | | 19.5% | 31.1% | 4.17% | 24.2% | 21.9% | </md>

(c)

[Figure 116]

|[Figure 117]|
|---|

Parse the figure content behind with markdown code: <md> | Years | Dominican Republic | Kenya | Zambia |

| --- | --- | --- | --- |

[Figure 118]

DocOwl 1.5:

| 2007 | 1 | 10 | 9.9 | | 2010 | 3.47 | 4 | 6 | | 2012 | 2 | 4 | 7 | | 2014 | 3 | 4 | 4 | </md>

(d)

- Figure 12: Qualitative results of structure-aware chart parsing on the charts of the vertical bar (a), horizontal bar (b), pie (c), and line (d). Incorrect words in the answer are marked in red.

[Figure 119]

[Figure 120]

| |
|---|

[Figure 121]

[Figure 122]

(a a a)

| |[Figure 123]|
|---|---|
| | |

( NIMA RESTAURNT & BAR PARO TOWN Specialist in SHUTNIS DI SM PSTAIRS )

| |
|---|

(PEMA DORA GENERAL SHOP)

| |
|---|

(a alamy stock photo E7BDR1 www.alamy.com)

[Figure 124]

Detail the information and textual components depicted in the image.

[Figure 125]

DocOwl 1.5:

[Figure 126]

the main street in the town <ocr> a a a alamy NIMA RESTAURANT & BAR Special TOURS & DEL PSTAIRS PERIA DORA GENERAL SHOP SONAR ENTERPRISE a a a a a a alamy photo stock E7BDR1 a www.alamy.com </ocr>

(a)

[Figure 127]

| |
|---|

(PORTERHIRE)

[Figure 128]

Detail the information and textual components depicted in the image.

[Figure 129]

DocOwl 1.5:

a digger works to clear trees from the path of the railway line . <ocr> PORTERHIRE </ocr>

(b)

- Figure 13: Qualitative results of natural image parsing. For better visualization, some regions are enlarged and labeled with corresponding scene texts. Incorrect words in the answer are marked in red.

[Figure 130]

Predict the bounding box of the text <ocr> 54 </ocr>

[Figure 131]

DocOwl 1.5:

<bbox>329,341,371,381</bbox>

- (a)

(d)

|[Figure 132]|
|---|

[Figure 133]

<bbox>564,214,696,272</bbox>

DocOwl 1.5:

[Figure 134]

[Figure 135]

Give the bounding box of the text <ocr> Losing You" </ocr>

[Figure 136]

- (b)

[Figure 137]

[Figure 138]

[Figure 139]

Predict the bounding box of the text <ocr> networks in 2012 to find talent and potential </ocr>"

[Figure 140]

DocOwl 1.5:

<bbox>569,172,949,182</bbox>

(c)

|[Figure 141]|
|---|

[Figure 142]

Predict the bounding box of the text <ocr> particular to: Nicholas Capaldi

[Figure 143]

- • observe the Accounts Direction issued by the Accounting Officer Secretary of State for Culture, Media and Sport, including the relevant accounting and 7 July 2017 disclosure requirements, and apply suitable accounting policies on a consistent basis; Endorsed on behalf of Council:
- • make judgements and estimates on a reasonable basis;
- • state whether applicable accounting standards </ocr>

[Figure 144]

DocOwl 1.5:

<bbox>71,437,761,658</bbox>

- Figure 14: Qualitative results of Multi-grained Text Grounding. Some regions are enlarged for better visualization. Bounding boxes predicted by DocOwl 1.5 are drawn in images as solid red boxes.

[Figure 145]

[Figure 146]

[Figure 147]

Detect the text in the bounding box <bbox>859,2,962,36 </bbox>

[Figure 148]

[Figure 149]

DocOwl 1.5:

<ocr> Emirates </ocr>

- (a)

(d)

<ocr> Produced capital </ocr>

DocOwl 1.5:

[Figure 150]

[Figure 151]

Detect the text in the bounding box <bbox>128,430,256,455</bbox>

- (b)

[Figure 152]

[Figure 153]

[Figure 154]

|[Figure 155]|
|---|

Detect the text in the bounding box <bbox>76,674,833,682</bbox>

[Figure 156]

DocOwl 1.5:

<ocr> SPOT 1:30:00AM- 2:35:00APMRPer week (M),Tu,We 02/14/20 1:45AM (Mo) 01:00 DATELINE 1:35A MRBPBK6033EH $150.00 </ocr>

| |
|---|

(c)

[Figure 157]

| |
|---|
|[Figure 158]|
| |

| |
|---|

[Figure 159]

[Figure 160]

Detect the text in the bounding box <bbox>14,532,981,850</bbox>

DocOwl 1.5:

<ocr> Ludwig Wittgenstein [edit] \n Born 1889, died 1951. An Austrian born Cambridge scholar, Wittgenstein was involved in his early years with the Logical Positivists of the Vienna Circle. \n Wittgenstein left Philosophy for a while but was persuaded to return in the 1920s. Wittgenstein's return prompted a different way of looking at religious \n language, he no longer sought to prove "whether" language refers, but looked at the way in which language refers to things. \n He decided that the best way to determine a word's meaning was to look at how it is used, and not to come up with a theoretical definition. Wittgenstein \n said this was about: "Getting back to the rough ground". </ocr>

- Figure 15: Qualitative results of Multi-grained Text Recognition. Some regions are enlarged for better visualization. Input bounding boxes are drawn in images as solid blue boxes. Incorrect words in answers are marked in red.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. ArXiv, abs/2204.14198, 2022.
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.
- [4] Lukasz Borchmann, Michal Pietruszka, Tomasz Stanislawek, Dawid Jurkiewicz, Michal Turski, Karolina Szyndler, and Filip Gralinski. DUE: end-to-end document understanding benchmark. In NeurIPS Datasets and Benchmarks, 2021.
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [6] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022.
- [7] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pages 3558–3568. Computer Vision Foundation / IEEE, 2021.
- [8] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. Tabfact : A large-scale dataset for table-based fact verification. In International Conference on Learning Representations (ICLR), Addis Ababa, Ethiopia, April

- 2020.

[9] Xingyu Chen, Zihan Zhao, Lu Chen, JiaBao Ji, Danyang Zhang, Ao Luo, Yuxuan Xiong, and Kai Yu. Websrc: A dataset for web-based structural reading comprehension. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4173–4185,

- 2021.

- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. CoRR, abs/2305.06500, 2023.
- [11] Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. TURL: table understanding through representation learning. SIGMOD Rec., 51(1):33–40, 2022.
- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR. OpenReview.net, 2021.
- [13] Hao Feng, Qi Liu, Hao Liu, Wengang Zhou, Houqiang Li, and Can Huang. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. CoRR, abs/2311.11810, 2023.

- [14] Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis. Evaluation of deep convolutional nets for document image classification and retrieval. In ICDAR, pages 991–995. IEEE Computer Society, 2015.
- [15] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for GUI agents. CoRR, abs/2312.08914, 2023.
- [16] Anwen Hu, Shizhe Chen, and Qin Jin. Question-controlled text-aware image captioning. In ACM Multimedia, pages 3097–3105. ACM, 2021.
- [17] Anwen Hu, Yaya Shi, Haiyang Xu, Jiabo Ye, Qinghao Ye, Ming Yan, Chenliang Li, Qi Qian, Ji Zhang, and Fei Huang. mplug-paperowl: Scientific diagram analysis with the multimodal large language model. arXiv preprint arXiv:2311.18248, 2023.
- [18] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document AI with unified text and image masking. In ACM Multimedia, pages 4083–4091. ACM, 2022.
- [19] Kushal Kafle, Brian L. Price, Scott Cohen, and Christopher Kanan. DVQA: understanding data visualizations via question answering. In CVPR, pages 5648–5656. Computer Vision Foundation / IEEE Computer Society, 2018.
- [20] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. In ICLR (Workshop). OpenReview.net, 2018.
- [21] Shankar Kantharaj, Rixie Tiffany Ko Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq R. Joty. Chart-to-text: A large-scale benchmark for chart summarization. In ACL (1), pages 4005–4023. Association for Computational Linguistics, 2022.
- [22] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In ECCV (28), volume 13688 of Lecture Notes in Computer Science, pages 498–517. Springer, 2022.
- [23] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 18893–18912. PMLR, 2023.
- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. CoRR, abs/2301.12597, 2023.
- [25] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. CoRR, abs/2311.06607, 2023.
- [26] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. CoRR, abs/2310.03744, 2023.
- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. CoRR, abs/2304.08485, 2023.
- [28] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023.
- [29] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL (Findings), pages 2263–2279. Association for Computational Linguistics, 2022.

- [30] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for VQA on document images. In WACV, pages 2199–2208. IEEE, 2021.
- [31] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In WACV, pages 2582–2591. IEEE, 2022.
- [32] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In WACV, pages 1516–1525. IEEE, 2020.
- [33] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.
- [34] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In ACL (1), pages 1470–1480. The Association for Computer Linguistics, 2015.
- [35] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. CoRR, abs/2306.14824, 2023.
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021.
- [37] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In NeurIPS, 2022.
- [38] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL (1), pages 2556–2565. Association for Computational Linguistics, 2018.
- [39] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: A dataset for image captioning with reading comprehension. In ECCV (2), volume 12347 of Lecture Notes in Computer Science, pages 742–758. Springer, 2020.
- [40] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In CVPR, pages 8317–8326. Computer Vision Foundation / IEEE, 2019.
- [41] Tomasz Stanislawek, Filip Gralinski, Anna Wróblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemyslaw Biecek. Kleister: Key information extraction datasets involving long documents with complex layouts. In ICDAR (1), volume 12821 of Lecture Notes in Computer Science, pages 564–579. Springer, 2021.
- [42] S Svetlichnaya. Deepform: Understand structured documents at scale, 2020.
- [43] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In AAAI, pages 13878–13888. AAAI Press, 2021.
- [44] Benny J. Tang, Angie Boggust, and Arvind Satyanarayan. Vistext: A benchmark for semantically rich chart captioning. In ACL (1), pages 7268–7298. Association for Computational Linguistics, 2023.
- [45] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19254–19264, 2023.
- [46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [47] Michal Turski, Tomasz Stanislawek, Karol Kaczmarek, Pawel Dyda, and Filip Gralinski. Ccpdf: Building a high quality corpus for visually rich documents from web crawl data. In ICDAR (3), volume 14189 of Lecture Notes in Computer Science, pages 348–365. Springer, 2023.
- [48] Vicuna. Vicuna: An open chatbot impressing gpt-4. https://github.com/lm-sys/ FastChat, 2023.
- [49] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. OFA: unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In ICML, volume 162 of Proceedings of Machine Learning Research, pages 23318–23340. PMLR, 2022.
- [50] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. CoRR, abs/2311.03079, 2023.
- [51] Wenjin Wang, Yunhao Li, Yixin Ou, and Yin Zhang. Layout and task aware instruction prompt for zero-shot document image question answering. CoRR, abs/2306.00526, 2023.
- [52] Haiyang Xu, Ming Yan, Chenliang Li, Bin Bi, Songfang Huang, Wenming Xiao, and Fei Huang. E2E-VLP: end-to-end vision-language pre-training enhanced by visual learning. In ACL/IJCNLP (1), pages 503–513. Association for Computational Linguistics, 2021.
- [53] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei A. F. Florêncio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. In ACL/IJCNLP (1), pages 2579–2591. Association for Computational Linguistics, 2021.
- [54] Zhengyuan Yang, Yijuan Lu, Jianfeng Wang, Xi Yin, Dinei Florêncio, Lijuan Wang, Cha Zhang, Lei Zhang, and Jiebo Luo. TAP: text-aware pre-training for text-vqa and text-caption. In CVPR, pages 8751–8761. Computer Vision Foundation / IEEE, 2021.
- [55] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-docowl: Modularized multimodal large language model for document understanding. CoRR, abs/2307.02499, 2023.
- [56] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Fei Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In EMNLP (Findings), pages 2841–2858. Association for Computational Linguistics, 2023.
- [57] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality. CoRR, abs/2304.14178, 2023.
- [58] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. CoRR, abs/2311.04257, 2023.
- [59] Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. Mmllms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601, 2024.
- [60] Liang Zhang, Anwen Hu, Jing Zhang, Shuo Hu, and Qin Jin. MPMQA: multimodal question answering on product manuals. CoRR, abs/2304.09660, 2023.
- [61] Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. Tablellama: Towards open large generalist models for tables. CoRR, abs/2311.09206, 2023.
- [62] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. CoRR, abs/2303.18223, 2023.

- [63] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno-Yepes. Image-based table recognition: Data, model, and evaluation. In ECCV (21), volume 12366 of Lecture Notes in Computer Science, pages 564–580. Springer, 2020.
- [64] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023.

