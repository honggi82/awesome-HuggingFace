## Enhancing Vision-Language Pre-training with Rich Supervisions

# arXiv:2403.03346v2[cs.CV]13Mar2025

Yuan Gao1∗† Kunyu Shi2∗ Pengkai Zhu2 Edouard Belval2 Oren Nuriel2

Srikar Appalaraju2 Shabnam Ghadar2 Vijay Mahadevan2 Zhuowen Tu2 Stefano Soatto2 1Stanford University 2AWS AI Labs

y1gao@stanford.edu {kunyus, zhpengka, belvae, onuriel, srikara, shabnam, ztu, soattos}@amazon.com

### Abstract

We propose Strongly Supervised pre-training with ScreenShots (S4) - a novel pre-training paradigm for Vision-Language Models using data from large-scale web screenshot rendering. Using web screenshots unlocks a treasure trove of visual and textual cues that are not present in using image-text pairs. In S4, we leverage the inherent tree-structured hierarchy of HTML elements and the spatial localization to carefully design 10 pre-training tasks with large scale annotated data. These tasks resemble downstream tasks across different domains and the annotations are cheap to obtain. We demonstrate that, compared to current screenshot pre-training objectives, our innovative pre-training method significantly enhances performance of image-to-text model in nine varied and popular downstream tasks - up to 76.1% improvements on Table Detection, and at least 1% on Widget Captioning.

### 1. Introduction

In recent years, there has been significant progress in Language Models (LMs) [7, 11, 43, 48] and Vision Language Models (VLMs) [1, 2, 6, 8, 10, 15, 16, 19, 22–26, 35– 37, 40, 41, 45, 46, 52, 53, 56–59, 61–71, 73–76] exhibiting strong zero-shot generalization and adaptability to a wide range of tasks. Though they may differ in architecture, data and task formulation, such foundational models predominantly rely on large-scale pre-training on giant corpora of web scraped data which serves as the source of generalization capability - C4 [47], The Pile [18], Laion 5B [50].

The pre-training of LMs and VLMs were mostly studied separately. For LMs, the inputs and outputs reside within a homogeneous space, and pre-training tasks that reconstruct inputs such as Masked Language Modeling (MLM) [13, 48] and Casual Language Modeling (CLM) [7, 44] have exhib-

*Equal contribution †Work conducted during an internship at Amazon.

[Figure 1]

Figure 1. We propose a novel pre-training paradigm - S4, composed of ten carefully designed tasks on large scale webscreenshots. Compared to image-to-text pretraining objectives on screenshots, which mainly utilized HTML[32] or its subset like raw texts[27, 33], our paradigm utilizes rich and diverse supervisions generated from web rendering that is also cheap to obtain.

ited the capability of learning knowledge from large corpora of text extracted from web crawls, which translates well to downstream task performance. On the other hand, although the input reconstruction type of pre-training for VLMs have shown performance improvements in certain settings, in general they are less effective compared to what is observed in language domain [10, 30] due to heterogeneity of vision tasks [12, 31, 39]. In addition to self-supervised learning tasks, many VLMs [10, 57] use a mixture of supervised pre-training tasks (e.g. object detection, VQA, captioning, etc), relying on human manually labeled datasets such as COCO [39], Object365 [51], VQA [20], etc as well

- as datasets generated in automated fashion such as LAION5B [50], WebLi-10B [10].

Advancements of supervised datasets have powered the advancements of VLMs. Increasing amounts of human annotated datasets were released [28, 29], which benefit the performance of similar or relevant downstream tasks, albeit

- at a steep cost. Approaches that can automatically generate supervisions at scale have also been explored [32, 50].

Notably, the use of massive amount of image-caption pairs, which are automatically generated using images and their associated Hypertext Markup Language (HTML) alt-text has enabled the development of some important VLMs such as CLIP models [46] and diffusion models [49]. Similarly, the use of screenshots and simplified HTML text pairs powered the Pix2Struct models [32]. However, methods capable of producing automatically annotated data beyond basic image-text pairs are currently under explored. Consequently, the effects of employing explicit, automatically generated, fine-grained supervision for pre-training have been understudied.

Therefore, in this work, we extend the use of web crawl corpuses and propose a novel pre-training framework that utilizes rich and diverse supervisions generated from web rendering. Modern websites are built using a combination of technologies such as HTML, CSS, JavaScript, that enable website creators to design dynamic content and interactive elements with diverse layouts.

To leverage such information, our solution renders crawled web-pages into screenshot images. We also have access to textual content, position, attribute and relationship of HTML elements - all of which can be obtained cheaply and utilized in pre-training. Building on this extracted data, we propose a set of pre-training tasks (see details in 3.2) that are highly synergistic to downstream tasks. Our results demonstrate significant performance improvements compared to the image-to-text pre-training baseline. On average, we observed an improvement of +2.7% points across 5 datasets (ChartQA, RefExp, Widget Captioning, Screen Summarization and WebSRC) with language outputs, and a notable average increase of +25.3% points on 4 datasets (PubLayNet, PubTables1M, RefExp candidate free and ICDAR 2019 modern) with localization outputs. See more in Tables 1 and 2. Our key contributions:

- • We develop an automatic data annotation pipeline that is able to render web crawls and create rich labels. Coupled with our carefully designed data cleaning process, we create a high-quality and large-scale vision language pre-training dataset.
- • We propose a novel pre-training paradigm - S4, composed of ten carefully designed tasks on large scale webscreenshots showing the effectiveness on a wide range of benchmarks.
- • Comparing to current screenshot pre-training objectives, our innovative pre-training method significantly enhances performance of image-to-text model in nine varied and popular downstream tasks - up to 76.1% improvements on Table Detection, and at least 1% on Widget Captioning.

### 2. Related Work

Next, we discuss in detail the difference to previous pretraining approaches.

Masked Signal Modeling. Pre-training through selfsupervision has revolutionized the field of natural language processing (NLP). Pioneering models like BERT [48] and GPTs [7, 44] demonstrated the profound impact of selfsupervised learning in enhancing generalization across a variety of language tasks. The success in NLP spurred analogous research in computer vision, leading to innovations in Masked Image Modeling (MIM) with approaches such as BEiT [5], SimMIM [60], and MAE [21] that recover masked pixels or patches, and improvements on classic vision tasks such as classification, semantic segmentation, etc are observed. In the domain of Vision Language(VL), MLIM [3] and MaskVLM [30] propose to integrate MLM and MIM and conduct VL pretraining in a joint manner.

Supervised Pre-training In supervised pre-training, image-caption pair annotations are generated on a large scale automatically from web crawls. This enables the training of models that generalize well in tasks like classification, retrieval, and captioning, as seen in works like CLIP, OFA, PaLi [10, 46, 57]. Donut [27] proposes a OCR-free model that relies on text reading pre-training of documents. SPOTLIGHT uses [33] region to text pre-training task on website and UI datasets. Pix2Struct [32] leverages screenshot and image pairs with a screen parsing pre-training task that converts webpage screenshots to HTML text. Our work proposes a pre-training paradigm that goes beyond image-text pairing type of tasks. We develop a suite of diverse, heterogeneous tasks specifically crafted to mirror the nature of downstream applications.

### 3. S4 Pre-training

In this section, we propose a novel pre-training paradigm for Vision-Language Models — Strongly Supervised pretraining with ScreenShots (S4) from large scale website rendering. We will first describe the creation procedure of our dataset for S4 pretraining, which we will call S4 Data, and then go through our proposed pre-training tasks enabled by our novel preprocessing method.

#### 3.1. Dataset

##### 3.1.1 Dataset Description

CommonCrawl† provides access to a large-scale web page corpus spanning over a decade. We download the web crawls from the Registry of Open Data on AWS† and we filter content with an explicit copyright notice. We execute our rendering and extraction pipeline (described in 3.1.2) and data pre-processing and cleaning procedure (described in 3.1.3) to obtain 15M screenshots enriched with supervisions. We applied deduplication based on urls to make sure our screenshots are unique. Each page is rendered at a

†http://commoncrawl.org/ †https://registry.opendata.aws/commoncrawl/

[Figure 2]

- Figure 2. Compared to traditional pre-training paradigms, our rich supervised pre-training leverages much more information that is also cheap to acquire (i.e via browser). We can then utilize the rich semantic and structural annotations to construct novel pre-training tasks that are naturally and directly aligned with downstream tasks. We use green words to refer to the words contained (visible) in the screenshot. We use red words to refer to the words that are not visible in the screenshot. For instance, “price” is not shown on the screenshot, but is the id of an element (refer to picture). We use brown words in the format of <x><y><x><y> to denote the bounding box.

resolution of 1280x1280 and is paired with matching annotations that enable the proposed pre-training tasks described in 3.2.

##### 3.1.2 Efficient Rendering and Supervision Extraction

We use Playwright †, which provides a programmatic interface to headless browsers that we use to render raw HTML files into screenshots. For each web page, we retrieve and cache the associated CSS, JavaScript fonts and images needed to render the page accurately. Caching those assets avoids making unnecessary requests to the originating website and quicker rendering, allowing us to create 5M parsed screenshot per day with 500 CPUs.

We build the annotations by traversing through the document object model (DOM) tree and collecting annotations for every leaf node of type Text, Image, Table or Input. More information about the dataset and example annotation can be found in the supplementary material.

†https://github.com/microsoft/playwright

##### 3.1.3 Pre-processing and Cleaning

During data rendering, we found that directly traversing through the DOM tree and collecting information on each node would lead to the inclusion of elements that were not visible in the page. We solve this issue by inspecting their CSS property for visibility and verifying the alignment to their corresponding bounding box. Specifically, if the element elem b returned by clicking on the center elem a’s bounding box is not a descendent of elem a, then elem a is pruned. This simple heuristics helps us get rid of most of the annotation that contains invisible elements. Also, we implemented recursive pre-order traversal to filter out overflow words in a textnode where the texts are overflowing outside of it’s ancestor’s bounding box. Without such a filter, words that are occluded by other elements would be included in the final annotation. Finally, we get rid of all <iframe> tags since the Same Origin Policy prohibits direct access of the nodes in <iframe>.

#### 3.2. Pre-training Task construction

Using the rich information provided by the HTML document structure, we design ten diverse supervised objec-

tives, which improve upon previous supervision like Screen Parsing. Our tasks include: Screen Parsing, OCR, Image Grounding, Element Grounding, Attribute Prediction, Node Relation Prediction, Table Detection, Table Parsing, Screen Titling, and Layout Analysis. We describe the objctives in the sections below, as well as in Figure 2.

##### 3.2.1 Screen Parsing

Similar to Pix2Struct, our Screen Parsing objective aims to reconstruct both the texts and their underlying structure. As shown in Figure 2, the input is simply a screenshot with a bounding box drawn on a region, with words 50% masked words, and the target is the cleaned HTML. We obtain the cleaned and simplified HTML as described in Pix2Struct by removing invisible nodes, and recursively remove chained nesting.

##### 3.2.2 Optical Character Recognition - OCR

The OCR objective aims to train the model with the ability for spatial understanding. It takes in a screenshot with drawn bounding box specifying a region, and outputs a “word0<x0><y0><x0><y0>word1<x1><y1><x1><y1>...” sequence. To limit the sequence length, we choose bounding box for a random region that contains at most 50 words. The OCR objective empowers the model with the ability to spatially understand the screenshot, bringing benefits for general detection and grounding tasks.

##### 3.2.3 Image Grounding

Image grounding is an important aspect to enhance imagetext alignment, and we want to empower our model with the ability to understand the semantics of the images in the context of a screenshot. We formulate the Image grounding objective as follow: First, we randomly pick an <img> element from the screenshot. Then, we obtain two captions for the image: one is from the alt attribute, which we call alt caption, and second is from the text node that is closest to the <img> element in the HTML Tree, which we call neighbor caption. We then randomly choose from {alt caption, neighbor caption} and ask the model to predict the bounding box of the image described. Note that since the neighbor caption appears in the screenshot, the model can, instead of learning the image-text relation, simply cheat to locate the neighbor caption first and then predict the bounding box for the image closest to it. Therefore, to avoid leaking spatial information from texts, we mask out 90% of the texts for the input screenshot to the model.

##### 3.2.4 Element Grounding

Element grounding is a generalization of image grounding to other elements in the HTML DOM tree. To build a better

representation of the meaning and functionality of each elements shown in the screenshot, we ask the model to localize their position based on a text description. We obtain the text description by concatenating the element tag and attributes from {class, id, label, for, alt, title, type}. However, values of the attributes are often noisy as the id and class label of an element can be randomized (i.e, in web frontend frameworks such as React.js). We address this issue by adding a post-processing step that filters out words that are numerical, single characters or that combines letters and numbers, as they are unlikely to useful labels. As a final step we use the T5 tokenizer to get rid of strings that map to < unk > tokens.

##### 3.2.5 Attribute Prediction

Beyond elements grounding from descriptions, we also ask the model to predict a matching description for a region in HTML. We group the visible elements into groups where they contain the same tag and attributes within {class, id, label, for, alt, title, type}, and randomly specify a group by rendering its bounding box to the input screenshot. The model is then asked to predict the tag and attributes in the following format: “{tag} {tag.class} {tag.id} {tag.label} {tag.for} {tag.alt}”. We apply the same post-processing described in 3.2.4 to filter out noise in the attribute values. The Attribute Prediction task forces the model to reason about the semantic meaning of each element, which could bring benefits downstreams tasks that involves element-level understanding.

##### 3.2.6 Node Relation Prediction (NRP)

This task is a pixel-only adaptation of the Node Relation Prediction objective introduced by MarkupLM[34], which takes the tree-structure of HTML and labels the relationship as either {self, parent, child, sibling, ancestor, descendent, others}. Given two elements outlined with bounding boxes in the input image, the model has to predict their node-level relationship. This task is expected to force the model to learn the relationships between the various layout components and how they interact.

##### 3.2.7 Table Detection

To closely mimic the downstream task for table detection, we construct table detection on our screenshot data. The construction is as simple as merging the bounding box for the elements with <table[id]> contained in their Xpaths, which results in the ground truth bounding box for each table. We then ask the model to predict the following sequence:<xtable0><ytable0><xtable0><ytable0><xtable1>

<ytable1><xtable1><ytable1>...

[Figure 3]

- Figure 3. Visualization of layout parsed from a screenshot. Corresponding HTML tags like <h1> are visualize on top-left corner of the bounding box.

##### 3.2.8 Table Parsing

The original Screen Parsing objective, although encouraging structure-level understanding, does not emphasize the semantics of those structures, as the pre-processing replaces tags with empty brackets. We argue that the information contained in the tags is also useful signal for pre-training, especially for well-structured elements like <table>. Therefore, we design a table parsing objective which contains the original tag name as well as the text contents for tables inside a page, as shown in Figure 2.

##### 3.2.9 Screenshot Titling

To encourage the model to summarize the content in the screenshot and improve its ability on image captioning, we propose a screen titling task. Specifically, the main title in the screenshot is masked and the model is asked to generate the title text by only looking at the rest of the web page. The ground truth title is obtained from the < title > node of the HTML DOM tree. The Screenshot Titling task closely resembles the screen summarization task for UI understanding,

##### 3.2.10 Layout Analysis

Obtaining layout from a screenshot is realized by grouping elements under the same sub-tree in the HTML. Specifically, for each element we obtain its cleaned Xpath by only keeping tags in [<p>,<table>,<form>,<dl>,<button>,<ol>, <ul>,<nav>,<img>,<object>] as they represent the semantic abstraction of the element. Then, we group each elements according to the value of their cleaned Xpath

to form layout of the screenshot. A visualization of the layout from a screenshot is shown in Figure 3.

#### 3.3. Architecture

We adopt a simple architecture with an image encoder followed by a text decoder, same as Pix2Struct [32] and similar to Donut [27]. The Image encoder is ViT [14] and text decoder is transformer decoder, where the vocabulary is extended with 1000 coordinate tokens (representing discrete positions in images, normalized between 0-1000) to support localization tasks such as object detection and visual grounding. Such image-encoder-text-decoder models don’t need text input and have the advantage of being OCR-free, which leads to reduced latency [27]. On the other hand, in order to read textual content that are typically small, input image resolution has to be high for good performance, which leads to increased memory usage. Our proposed S4 pre-training paradigm is not limited to this architecture and can be applied to other approaches as well.

### 4. Experiments

We validate the effectiveness of our ten proposed pretraining tasks by fine-tuning the model on nine downstream tasks and compare its performance to a Pix2Struct baseline model that was only pre-trained with screen parsing. Based on the output format, we also divide the downstream tasks into two groups.

#### 4.1. Implementation Details

Pre-training schema. We propose 2 pre-training schemes, S4NL for natural language generation and S4Loc for localization, targeting on different downstream tasks. Specifically, S4NL includes the baseline screen parsing task and all the tasks on natural language generation, including Attribute Prediction, Table Parsing, Title Generation, and Node Relation Prediction. S4Loc comprises of tasks with bounding box generations, including OCR, Image Grounding, Element Grounding, Table Detection and Layout Analysis, in addition to the screen parsing task. During pretraining, we randomly sample one task for each image with uniform distribution.

##### 4.1.1 Pretraining Settings

We conducted pretraining on both 2 million and 15 million subsets of our S4 dataset, during which we set the screenshot’s viewport to 1280*1280. We initialize our model with weights from Pix2struct-base and set our batch size to 32 for each node and use 4 A100 nodes during pretraining. The maximum sequence length for the pretraining targets is 128 and the patch size for the input image is 2048. Our optimizer is AdamW with learning rate set to 1e-4 with cosine decay, and for both 2M and 15M subsets we pretrain with

Pre-training Pre-training Finetune

ChartQA ↑ RefExpcls↑ Dataset Objectives Batchsize WidgetCap. ↑ ScreenSum. ↑ WebSRC ↑

Methods

Pix2Struct [32] Google Priv. Data - 80M Screen Parsing 32 to 256 56.0 92.2 133.1 107.0 Pix2Struct† Google Priv. Data - 80M Screen Parsing 8 54.3 91.7 131.1 105.5 60.4

Donut [27] SynthDoG - 37M OCR 64 41.8 - 127.4 56.4 Pix2Struct* S4 Data - 2M Screen Parsing 8 47.4 87.9 129.5 101.3 58.7 Pix2Struct* S4 Data - 15M Screen Parsing 8 52.1 88.1 129.2 104.5 60.1 S4* (Ours) S4 Data - 2M S4NL 8 50.5 92.4 130.5 103.2 60.5 S4* (Ours) S4 Data - 15M S4NL 8 55.0 94.9 130.6 105.7 61.1

- Table 1. Results for Chart, Web, and UI Understanding datasets. * denotes that we load Pix2Struct’s pre-trained weight and further pre-train on our S4 dataset with corresponding objectives. † denotes the reproduced results on downstream tasks with Pix2Struct-base’s pre-trained weight and smaller batch size. Results from gray rows are not directly comparable to our S4 model since we don’t have access to their nonreleased pre-training datasets. The results from last 4 rows show that in addition to the Pix2Struct’s pre-training objective, our supervised pre-training extracted from HTML DOM tree brings consistent improvement on various of downstream tasks. Note that the OCR objective for donut doesn’t include bounding box prediction.

- 1 epoch per pretraining task. For instance, for S4NL pretraining with the 2M subset, there are 5 tasks so the total training sample is 5*2M = 10M. Note that since for each screenshots we can obtain multiple tasks, the models sees the same subset of screenshots regardless of the number of pretraining tasks.

#### 4.2. Chart, Web, and UI Understanding

In this section we evaluate on tasks that require generating natural language responses from image inputs. We focus on Chart & Web VQA, UI summarization and UI widget captioning.

##### 4.2.1 Datasets

ChartQA: ChartQA[42] is a VQA dataset for different types of charts (bar charts, line graphs, etc.). It includes both extractive and reasoning questions, which requires analyzing the visual data in charts to extract the relevant information. We follow the convention and report the Relaxed Match metric on ChartQA.

WebSRC: WebSRC[9] is a web-based VQA dataset. It contains both cleaned HTML and the screenshot of web pages, and the task is to answer the question about the content in the web page. Prior arts mostly tackle this problem by taking the ground truth cleaned HTML code as inputs, which is an unrealistic setting as real-word applications often have much more complex HTML codes than the cleaned data. Instead, our model only takes the screenshot as inputs and predicts the answer from pure-vision information. On WebSRC the Exact Match metric is reported.

Screen2words: Screen2words[55] is a dataset for extracting summarization from screenshots of mobile app screens. We use Bleu and Cider scores as the evaluation metrics.

Widget Captioning: Widget Captioning[38] is a dataset for generating descriptive captions for UI widgets. The task is to generate captions that accurate describe the purpose or

function of a widget in a bounding box, such as a button, slider, or a checkbox. In the input screenshot, the target widget is specified through the rendered bounding box. Bleu and Cider scores are used as evaluation metrics.

UI RefExpcls UI Referential Expression (RefExp) [4] is a dataset specifically designed for grounding referring expressions for UI elements in screenshots. Current SOTA usually approach this problem in a simplified classification formulation: given a question and a candidate widget from annotation, the model is asked to predict whether the widget and question are related. This setting requires little localization ability from the model as the candidates widget bounding boxes are provided as inputs. We call this classification setting RefExpclsand report the classification accuracy as the metric.

##### 4.2.2 Settings

Following the same training and evaluation protocol [32], our model was pre-trained with S4NL objectives and finetuned on the Chart, Web, and UI Understanding tasks. Since there’s no access to Google’s 80M private data, we compare to two PixStruct variations. The first one is with the weights pre-trained on its private data using screen parsing released by the original author. The second one is initialized with the former’s weights, and is further pre-trained on our 2M and 15M S4 data with only screen parsing objective. To have a fair comparison, our model is initialized with the same weights, and pre-trained on the same amount of data (2M and 15M S4 data) but with extra tasks. We also compare to Donut[27], which is another model uses pure-vision inputs and produces text predictions.

###### 4.2.3 Results We tabulate the results in Tab. 1. Our method consistently outperforms Pix2Struct on all downstream tasks with significant margins when pre-trained with the same

RefExp ↑ PublayNet ↑ PubTables1M ↑ ICDAR 2019 ↑

Pre-training Dataset

Pre-training Objectives

Methods

cand free (Table Det.) (Modern Subset)

30k samples 1M samples 400k samples 600 samples

DETR - - - - 99.5 DiT-B (Cascade RCNN) IIT-CDIP - 42M MIM - 95.4 - 97.2

Pix2Struct [32] Google Priv. Data - 80M Screen Parsing 55.1 91.1 97.0 3.6 Pix2Struct* S4 Data - 2M Screen Parsing 52.7 91.0 97.1 3.3

S4* (Ours) S4 Data - 2M S4Loc 83.6 92.5 98.4 70.7 S4* (Ours) S4 Data - 15M S4Loc 84.3 93.1 99.0 79.4

- Table 2. Results on Detection and Grounding datasets. We train all of the models with batch size = 32 and patch size = 2048.* denotes that we load Pix2Struct-base’s pre-trained weight and further pre-train on our S4 dataset with corresponding objectives. Models denoted gray are specialist detection models that cannot parse language input (i.e cannot do grounding tasks). With our pre-training objectives, autogressive models can get significant boosts on various detection & grounding tasks. MIM refers to Masked Image Modeling.

data. Specifically, when pre-trained with 15M image-text pairs, our method achieves 2.9, 6.8, 1.4, 1.2, and 1.0 improvement over Pix2Struct on ChartQA, RefExpcls, Widget Captioning, Screen Summarization, and WebSRC, respectively. Notice that the largest improvement is obtained on RefExpcls, because the pre-training tasks we proposed, such the attribute prediction and node relation prediction, help the model build robust connections between the UI elements and their referring expressions. In addition, when more data (15M) is available, our pre-training scheme S4NL gets improved accuracy compared to less training data (2M). The outstanding results comparing to the baseline and Donut demonstrate the efficacy of the proposed S4 pre-training scheme for downstream tasks that involve chart, web, and UI understanding.

We also noticed that, comparing to the original Pix2struct, further pre-training it on our 2M data with screen parsing objective harms the performance across different datasets. This is expected as our data collected from Common Crawl has different distribution against the original Pix2struct data due to the data size (2M vs. 80M) and potentially different website filtering strategies. Specifically, we filter out website whose CSS and JS files failed to download, while the filtering process remain unclear for Pix2struct. In addition, we also used a much smaller batch size (128 vs. 2048) due to computational constraints. Therefore, the pre-train on 2M data might drive the model weight to a less optimal state. With 15M pre-training data, we observed performance improvements over 2M data, implying that more data might compensate this distribution shift.

#### 4.3. Detection and Grounding

We further investigate the effect of S4 pre-training on tasks that require spatial information understanding, such as image grounding and localization. While current SOTA models adopt specific architectures for detection, we show that

with sufficient pre-training tasks on localization, an autogressive model can close the gap towards detection-specific architectures.

##### 4.3.1 Datasets

PubLayNet: PubLayNet [72] is a large-scale dataset for document layout analysis, containing more than 360,000 pages of scientific articles. We evaluate on the bounding box prediction task and report AP 50 as the metric.

ICDAR2019: ICDAR2019 [17] is a table detection dataset that contains 1200 modern and archived documents. We only evaluate on the modern document split as the archived documents do not have bounding boxes. We use AP 50 as our evaluation metric.

PubTables-1M PubTables-1M [54] has around 400k images with 947,642 tables from PMCOA scientific articles and we use it for table detection experiments. AP 50 is used as the evaluation metric.

UI RefExpcand free As mentioned earlier, current works mostly treat the UI RefExp task as a binary classification problem using the ground truth bounding boxes as candidates, making it less challenging as it does not measure whether the model can localize the UI elements. In this work, we propose a new task, UI RefExpcand free, to include the element grounding into the challenges. In this task, the input is only the screenshot with the text description, and the model is asked to predict the bounding box of the related UI element directly, thus ”candidate free”. For evaluation, the predicted bounding box will be matched to the closest ground truth box to compute the accuracy.

##### 4.3.2 Settings

Our model was pre-trained with S4Loc for the benefits on localization related tasks. The model is then fine-tuned and evaluated on each downstream task dataset. We compare to

Titling Attibute Pred. NRP Table Parsing. Screen Parsing ChartQA Widget Cap. Screen Sum. RefExpcls

✓ 47.4 129.5 101.3 87.9 ✓ ✓ 48.7 128.3 101.4 87.8 ✓ ✓ ✓ 50.7 128.3 101.3 89.4 ✓ ✓ ✓ ✓ 50.1 130.7 101.1 92.7 ✓ ✓ ✓ ✓ ✓ 50.5 130.5 103.2 92.4

Table Detection Layout Analysis Image & Element Grounding OCR Screen Parsing ICDAR RefExpcand free

✓ 3.3 52.7 ✓ ✓ 50.1 68.6 ✓ ✓ ✓ 52.9 66.2 ✓ ✓ ✓ ✓ ✓ 70.7 83.6

Table 3. Ablation study on adding different pre-training objectives using 2M S4 data.

two PixStruct variations. The first one is the weights pretrained on its private data using screen parsing released by the original author. The second one is initialized with the former weights and is further pre-trained on our S4 data 2M with only screen parsing, to be compared to our model pretrained on the same amount of data.

##### 4.3.3 Results

The evaluation results on all detection and grounding related tasks are tabulated in Tab. 2. Our method shows clear advantages over the baseline Pix2Struct model that was only pre-trained with the screen parsing task. Specifically, on RefExpcand free, when pre-trained with 2M data, our method outperforms Pix2Struct by a significant margin of 30.9 (83.6 vs. 52.7). This is because our pre-training tasks like OCR prediction and element grounding, help the model learn to localize items in the image given their semantic descriptions. Similarly, on ICDAR, which only has 600 training samples, our method achieves 70.7 AP with

- 2M pre-training data, while the baseline Pix2Struct only obtains 3.3, due to its lack of localization ability. When there are more fine-tuning data for the downstream tasks (PublayNet 1M & PubTables 400K), Pix2Struct can learn to localize the objects and obtain decent performance, but our

training scheme S4Loc still benefits the model and improves the performance by 1.5 on PublayNet and 1.3 on PubTables.

The benefits of S4Loc pre-training becomes more prominent when more pre-train data is available. Our model pretrained with 15M data consistently improves the accuracy on all four downstream tasks compared to 2M pre-train data. In particular, on ICDAR the 15M pre-train data improves the accuracy from 70.7 to 79.4, showing that the pretraining task benefits more when the downstream task has less data. It is worth noting that, as a generic auto-regressive text generation model, our method with 15M pre-training data achieves comparable performance on PublayNet and PubTables to detection specific models like DeTR and DitB, showing that sufficient pre-training with proper tasks

helps close the gap between auto-regressive models and detection-specific architectures.

#### 4.4. Contribution of each task

We conducted ablative studies to show the effectiveness of each individual pre-training tasks besides screen parsing. For S4NL, we evaluate on ChartQA, Widget Captioning, Screen Summarization, and RefExpcls, by adding the natural language related tasks gradually. For S4Loc, we also add the localization related tasks incrementally and evaluate on RefExpcand free and ICDAR. The results are shown in Tab. 3. Observe that the downstream task usually benefits from the addition of the most related pre-training task. For example, Screen Summarization gets 2.1 performance improvement when the screen tilting pre-training task is added, while the other tasks have little effect on the performance. The attribute prediction task encourages the model to associate website elements to their text description. Therefore, adding it to the pre-training scheme significantly improves the performance on both Widget Captioning and RefExpcls, which requires the model to associate UI elements to texts. Similarly, adding all the localization related pre-train task substantially improves the model’s ability on grounding elements, resulting in higher accuracy on both ICDAR and RefExpcand free.

### 5. Conclusions

We introduced a novel pre-training framework for visionlanguage models, with which models are exposed with a variety of supervised tasks on diverse and massive amount website screenshots. This innovation is enabled by our proposed data pipeline, in which the web pages are rendered, extracted, and cleaned automatically to generate screenshots and corresponding annotations. The tasks in our pretraining scheme are designed to maximize the utilization of annotations in our data as well as the similarities between the downstream tasks. Through extensive experiments, we demonstrated the efficacy of our method on boosting the downstream tasks performance on 9 different datasets.

### References

- [1] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. Docformer: End-to-end transformer for document understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 993–1003, 2021. 1
- [2] Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R. Manmatha. Docformerv2: Local features for document understanding. AAAI, abs/2306.01733,

2024. 1

- [3] Tarik Arici, Mehmet Saygin Seyfioglu, Tal Neiman, Yi Xu, Son Train, Trishul Chilimbi, Belinda Zeng, and Ismail Tutar. Mlim: Vision-and-language model pre-training with masked language and image modeling. arXiv preprint arXiv:2109.12178, 2021. 2
- [4] Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, et al. Uibert: Learning generic multimodal representations for ui understanding. arXiv preprint arXiv:2107.13731, 2021. 6
- [5] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 2
- [6] Ali Furkan Biten, Ron Litman, Yusheng Xie, Srikar Appalaraju, and R. Manmatha. Latr: Layout-aware transformer for scene-text vqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16548–16558, 2022. 1
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1, 2
- [8] Feilong Chen, Duzhen Zhang, Minglun Han, Xiuyi Chen, Jing Shi, Shuang Xu, and Bo Xu. Vlp: A survey on visionlanguage pre-training. Machine Intelligence Research, 20: 38–56, 2022. 1
- [9] Xingyu Chen, Zihan Zhao, Lu Chen, Danyang Zhang, Jiabao Ji, Ao Luo, Yuxuan Xiong, and Kai Yu. Websrc: A dataset for web-based structural reading comprehension, 2021. 6
- [10] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointlyscaled multilingual language-image model. arXiv preprint

- arXiv:2209.06794, 2022. 1, 2

[11] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint

- arXiv:2210.11416, 2022. 1

- [12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 1
- [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 1

- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 5
- [15] Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, Jianfeng Gao, and Lijuan Wang. Coarseto-fine vision-language pre-training with fusion in the backbone. ArXiv, abs/2206.07643, 2022. 1
- [16] Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao. A survey of vision-language pre-trained models. In International Joint Conference on Artificial Intelligence, 2022. 1
- [17] Liangcai Gao, Yilun Huang, Herv´e D´ejean, Jean-Luc Meunier, Qinqin Yan, Yu Fang, Florian Kleber, and Eva Lang. Icdar 2019 competition on table detection and recognition (ctdar). In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1510–1515, 2019. 7
- [18] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020. 1
- [19] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Jiao Qiao. Clip-adapter: Better vision-language models with feature adapters. ArXiv, abs/2110.04544, 2021. 1
- [20] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 1
- [21] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 2
- [22] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pretraining for image captioning. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17959–17968, 2021. 1
- [23] Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu. Seeing out of the box: Endto-end pre-training for vision-language representation learning. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12971–12980, 2021.
- [24] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, 2021.
- [25] Woojeong Jin, Yu Cheng, Yelong Shen, Weizhu Chen, and Xiang Ren. A good prompt is worth millions of parameters: Low-resource prompt-based learning for visionlanguage models. In Annual Meeting of the Association for Computational Linguistics, 2021.

- [26] Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, and Nicolas Carion. Mdetr - modulated detection for end-to-end multi-modal understanding. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 1760–1770, 2021. 1
- [27] Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer, 2022. 1, 2, 5, 6
- [28] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 1
- [29] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. 1
- [30] Gukyeong Kwon, Zhaowei Cai, Avinash Ravichandran, Erhan Bas, Rahul Bhotika, and Stefano Soatto. Masked vision and language modeling for multi-modal representation learning. arXiv preprint arXiv:2208.02131, 2022. 1, 2
- [31] Justin Lazarow, Kwonjoon Lee, Kunyu Shi, and Zhuowen Tu. Learning instance occlusion for panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10720–10729, 2020. 1
- [32] Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding, 2023. 1, 2, 5, 6, 7
- [33] Gang Li and Yang Li. Spotlight: Mobile ui understanding using vision-language models with a focus. 2023. 1, 2
- [34] Junlong Li, Yiheng Xu, Lei Cui, and Furu Wei. Markuplm: Pre-training of text and markup language for visually-rich document understanding. 2021. 4
- [35] Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, 2022. 1
- [36] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10955–10965, 2021.
- [37] Xiujun Li, Xi Yin, Chunyuan Li, Xiaowei Hu, Pengchuan Zhang, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, and Jianfeng Gao. Oscar: Objectsemantics aligned pre-training for vision-language tasks. ECCV 2020, 2020. 1
- [38] Yang Li, Gang Li, Luheng He, Jingjie Zheng, Hong Li, and Zhiwei Guan. Widget captioning: Generating natural language description for mobile user interface elements, 2020. 6

- [39] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 1
- [40] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Neural Information Processing Systems, 2019. 1
- [41] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. ArXiv, abs/2206.08916, 2022. 1
- [42] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning, 2022. 6
- [43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022. 1
- [44] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 1, 2
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. 1
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2
- [47] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019. 1
- [48] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020. 1, 2
- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [50] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training

- next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 1
- [51] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019. 1
- [52] Kunyu Shi, Qi Dong, Luis Goncalves, Zhuowen Tu, and Stefano Soatto. Non-autoregressive sequence-to-sequence vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13603–13612, 2024. 1
- [53] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Testtime prompt tuning for zero-shot generalization in visionlanguage models. ArXiv, abs/2209.07511, 2022. 1
- [54] Brandon Smock, Rohith Pesala, and Robin Abraham. PubTables-1M: Towards comprehensive table extraction from unstructured documents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4634–4642, 2022. 7
- [55] Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman, and Yang Li. Screen2words: Automatic mobile ui summarization with multimodal learning, 2021. 6
- [56] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, 2022. 1
- [57] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR, 2022. 1, 2
- [58] Wenhui Wang, Hangbo Bao, Li Dong, and Furu Wei. Vlmo: Unified vision-language pre-training with mixture-ofmodality-experts. ArXiv, abs/2111.02358, 2021.
- [59] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. ArXiv, abs/2208.10442, 2022. 1
- [60] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9653–9663, 2022. 2
- [61] Haiyang Xu, Ming Yan, Chenliang Li, Bin Bi, Songfang Huang, Wenming Xiao, and Fei Huang. E2e-vlp: End-toend vision-language pre-training enhanced by visual learning. ArXiv, abs/2106.01804, 2021. 1
- [62] Jinyu Yang, Jiali Duan, S. Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul M. Chilimbi, and Junzhou Huang. Vision-language pre-training with triple contrastive learning. 2022 IEEE/CVF Conference on Com-

- puter Vision and Pattern Recognition (CVPR), pages 15650– 15659, 2022.
- [63] Xu Yang, Hanwang Zhang, Guojun Qi, and Jianfei Cai. Causal attention for vision-language tasks. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9842–9852, 2021.
- [64] Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, Tat seng Chua, and Maosong Sun. Cpt: Colorful prompt tuning for pre-trained vision-language models. ArXiv, abs/2109.11797, 2021.
- [65] Andy Zeng, Adrian S. Wong, Stefan Welker, Krzysztof Choromanski, Federico Tombari, Aveek Purohit, Michael S. Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Peter R. Florence. Socratic models: Composing zero-shot multimodal reasoning with language. ArXiv, abs/2204.00598, 2022.
- [66] Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. ArXiv, abs/2111.08276, 2021.
- [67] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, JunJuan Zhu, Lionel Ming shuan Ni, and Heung yeung Shum. Dino: Detr with improved denoising anchor boxes for endto-end object detection. ArXiv, abs/2203.03605, 2022.
- [68] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. ArXiv, abs/2206.05836, 2022.
- [69] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Making visual representations matter in visionlanguage models. ArXiv, abs/2101.00529, 2021.
- [70] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Making visual representations matter in visionlanguage models. CVPR 2021, 2021.
- [71] Zhaoyang Zhang, Yantao Shen, Kunyu Shi, Zhaowei Cai, Jun Fang, Siqi Deng, Hao Yang, Davide Modolo, Zhuowen Tu, and Stefano Soatto. Musketeer: Joint training for multitask vision language model with task explanation prompts. arXiv preprint arXiv:2305.07019, 2023. 1
- [72] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1015–1022. IEEE, 2019. 7
- [73] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130:2337 – 2348,

2021. 1

- [74] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16795–16804, 2022.
- [75] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, and Jianfeng Gao. Unified visionlanguage pre-training for image captioning and vqa. ArXiv, abs/1909.11059, 2019.

[76] Mingchen Zhuge, Dehong Gao, Deng-Ping Fan, Linbo Jin, Ben Chen, Hao Zhou, Minghui Qiu, and Ling Shao. Kaleidobert: Vision-language pre-training on fashion domain. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12642–12652, 2021. 1

## Enhancing Vision-Language Pre-training with Rich Supervisions Supplementary Material

### A. Quantitative Results on Joint Training

We provide some additional discussions for the impact of joint training language and location tasks. Directly combining S4NL objectives and S4Loc objectives, which we denote as S4Joint, harms the performance on most downstream tasks. For S4Joint, we use the following weighted loss as it gives the best average performance across all tasks:

loss_weights = { ’screen2html’: 1.0, ’attribute_prediction’: 0.5, ’title_generation’: 0.5, ’node_relation_prediction’:0.1, ’table_parsing’:0.1, ’ocr’:0.1, ’table_detection’:0.1, ’layout_analysis’:0.1, ’image_grounding’:0.1, ’element_grounding’:0.1

}

Pre-training Pre-training Finetune

Dataset Objectives Batchsize WidgetCap. ↑ ScreenSum. ↑ WebSRC ↑

Methods

ChartQA ↑ RefExpcls↑

S4* (Ours) S4 Data - 15M S4NL 8 55.0 94.9 130.6 105.7 61.1 S4* (Ours) S4 Data - 15M S4Joint 8 52.1 94.9 128.4 101.5 60.3

RefExp ↑ PublayNet ↑ PubTables1M ↑ ICDAR 2019 ↑

Pre-training Dataset

Pre-training Objectives

cand free (Table Det.) (Modern Subset)

Methods

30k samples 1M samples 400k samples 600 samples

S4* (Ours) S4 Data - 15M S4Loc 84.3 93.1 99.0 79.4 S4* (Ours) S4 Data - 15M S4Joint 79.2 91.6 97.7 76.7

### B. Qualitative Results during Pre-training

We visualize qualitative pre-training results in below figures. For table detection, image grounding, and element grounding, red boxes denotes ground truth boxes and blue boxes denotes predicted boxes. We present four images for each of these three tasks, where the bottom right is a failure case and the others are good cases. For layout analysis, we present a pair of prediction (blue) and ground truth layout (red).

[Figure 4]

###### Figure 4. Attribute Prediction

[Figure 5]

###### Figure 5. Element Grounding

[Figure 6]

###### Figure 6. Image Grounding

[Figure 7]

###### Figure 7. Layout Analysis

[Figure 8]

###### Figure 8. Node Relation

[Figure 9]

###### Figure 9. OCR

[Figure 10]

###### Figure 10. Screen2html

[Figure 11]

###### Figure 11. Screen Tiltling

[Figure 12]

###### Figure 12. Table Detection

[Figure 13]

###### Figure 13. Table Parsing

