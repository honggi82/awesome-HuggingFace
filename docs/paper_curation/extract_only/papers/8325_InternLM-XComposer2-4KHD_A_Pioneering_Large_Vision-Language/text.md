## InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD

# arXiv:2404.06512v1[cs.CV]9Apr2024

Xiaoyi Dong∗1,2, Pan Zhang∗1, Yuhang Zang∗1, Yuhang Cao1,2, Bin Wang1, Linke Ouyang1, Songyang Zhang1, Haodong Duan1, Wenwei Zhang1, Yining Li1, Hang Yan1, Yang Gao1, Zhe Chen1 Xinyue Zhang1, Wei Li1, Jingwen Li1, Wenhai Wang1,2, Kai Chen1, Conghui He3, Xingcheng Zhang3,

Jifeng Dai4,1, Yu Qiao1, Dahua Lin1,2, Jiaqi Wang1,

1Shanghai Artificial Intelligence Laboratory, 2The Chinese University of Hong Kong, 3SenseTime Group, 4Tsinghua University

internlm@pjlab.org.cn

### Abstract

DocVQA

OCRBench

ChartQA

[Figure 1]

90.0

81.0

67.5

MathVista

TextVQA

The Large Vision-Language Model (LVLM) field has seen significant advancements, yet its progression has been hindered by challenges in comprehending fine-grained visual content due to limited resolution. Recent efforts have aimed to enhance the high-resolution understanding capabilities of LVLMs, yet they remain capped at approximately 1500 × 1500 pixels and constrained to a relatively narrow resolution range. This paper represents InternLMXComposer2-4KHD, a groundbreaking exploration into elevating LVLM resolution capabilities up to 4K HD (3840 × 1600) and beyond. Concurrently, considering the ultrahigh resolution may not be necessary in all scenarios, it supports a wide range of diverse resolutions from 336 pixels to 4K standard, significantly broadening its scope of applicability. Specifically, this research advances the patch division paradigm by introducing a novel extension: dynamic resolution with automatic patch configuration. It maintains the training image aspect ratios while automatically varying patch counts and configuring layouts based on a pre-trained Vision Transformer (ViT) (336 × 336), leading to dynamic training resolution from 336 pixels to 4K standard. Our research demonstrates that scaling training resolution up to 4K HD leads to consistent performance enhancements without hitting the ceiling of potential improvements. InternLM-XComposer2-4KHD shows superb capability that matches or even surpasses GPT4V and Gemini Pro in 10 of the 16 benchmarks. The InternLM-XComposer2-4KHD model series with 7B parameters are publicly available at https://github. com/InternLM/InternLM-XComposer.

57.8

77.2

AI2D

InfoVQA

80.9

68.6

2204.9

74.7

MME

SEEDI

71.8

77.7

QBenchT

MMBenchCN

54.9

39.9

40.9

80.2

MM-Vet

MMBench

54.1

HallB

MMMU

MMStar

GPT-4V Gemini-Pro IXC2-4KHD

Previous open-source SOTA

Figure 1. Overview of InternLM-XComposer2-4KHD performance on benchmarks with different resolutions. Our model based on InternLM2-7B [91] matches or even surpasses GPT-4V [74] and Gemini Pro [90] in 10 of the 16 benchmarks.

### 1. Introduction

In recent years, the progress in Large Language Models (LLMs) [10, 21, 29, 39, 73, 78, 91–93] has provoked the development of Large Vision-Language Models (LVLMs). These models have demonstrated proficiency in tasks such as image captioning [14, 17] and visual-question-answering (VQA) [31, 33, 57, 107]. Nevertheless, due to their limited resolution, they struggle with processing images containing fine details, such as charts [68], tables [87], documents [70], and infographics [69]. This limitation constrains their prac-

* indicates equal contribution.

tical applicability in real-world scenarios.

Recent advancements have aimed at enhancing the resolution of Large Vision-Language Models (LVLMs). Some approaches [36, 48, 66, 97] involve adapting highresolution vision encoders directly. However, the Vision Transformer (ViT) architecture falls short when dealing with images of varying resolutions and aspect ratios, thereby restricting its ability to handle diverse inputs effectively. Alternatively, some methods [37, 46, 50, 51, 55, 59, 99] maintain the vision encoder’s resolution, segmenting high-resolution images into multiple low-resolution patches. Yet, these methods are constrained by an inadequate resolution, typically around 1500 × 1500, which does not satisfy the demands of daily content, e.g., website screenshots [85], document pages [70], and blueprints [69]. Furthermore, they are confined to either a few predefined high-resolution settings [36, 46, 48, 50, 51, 55, 59, 66, 97] or a limited range of resolutions [37, 99], thereby restricting their utility across a variety of applications.

In this work, we introduce InternLM-XComposer24KHD, a pioneering model that for the first time expands the resolution capabilities of Large Vision-Language Models (LVLMs) to 4K HD and even higher, thereby setting a new standard in high-resolution vision-language understanding. Designed to handle a broad range of resolutions, InternLM-XComposer2-4KHD supports images with any aspect ratio from 336 pixels up to 4K HD, facilitating its deployment in real-world contexts.

InternLM-XComposer2-4KHD follows patch division [46, 50] paradigm and enhances it by incorporating an innovative extension: dynamic resolution with automatic patch configuration. To be specific, scaling the resolution of Large Vision-Language Models (LVLMs) to 4K HD and even higher standard is far beyond merely increasing the number of patches. It involves a nuanced approach to overcoming specific challenges: (1) Dynamic Resolution and Automatic Patch Configuration: Addressing the scarcity of high-resolution training data, our framework introduces a strategy that dynamically adjusts resolution alongside an automatic layout configuration. During training, it maintains the original aspect ratios of images while adaptively altering patch (336 × 336) layouts and counts. This results in a training resolution that exceeds the original image resolutions, reaching up to 4KHD, addressing the shortfall of high-resolution data. (2) Handling Variability in Patch Configurations: Despite the apparent simplicity of dynamic resolution training, the variability in patch configurations can heavily confuse LVLMs. To mitigate this, we introduce a newline token after each row of patch tokens to clearly delineate patch layouts, reducing training ambiguity and significantly boosting performance. (3) Inference Beyond 4K Resolution: Our observations reveal that, even when trained on images

up to 4K resolution, the model can achieve additional performance improvements during inference by processing images at higher resolutions.

Furthermore, scaling the training resolution up to 4K standard results in a consistent improvement in performance, highlighting the potential for training even beyond 4K resolution. This underscores the capacity for further enhancing model capabilities and suggests a promising trajectory for advancing the frontiers of high-resolution image processing within the domain of large vision-language models.

We evaluate our InternLM-XComposer2-4KHD on 16 diverse benchmarks spanning various domains, including 5 challenging HD-OCR datasets (DocVQA[70], ChartQA[68], InfographicVQA[69], TextVQA[87] and OCRBench[58]). Compared to previous open-source LVLM models and closed-source APIs, our approach achieves SOTA results in 6 of 16 benchmarks, demonstrating competitive performance despite only 7B parameters. As shown in Figure 1, InternLM-XComposer2-4KHD even surpasses the performance of GPT4V [74] and Gemini Pro [90] across ten benchmarks. Notably, our method exhibits excellent performance on 5 HD-OCR datasets, over existing open-source LVLMs by a substantial margin.

### 2. Related Works

Large Vision-Language Models (LVLMs). Large Language Models (LLMs) [6, 9, 10, 23, 39, 41, 73, 76, 78, 91– 93, 108] have gained significant attention due to their impressive performance in various language-related tasks such as text generation and question answering. Following this enthusiasm, recent Large Vision-Language Models (LVLMs) have emerged[4, 7, 16, 18, 19, 25, 28, 32, 47, 74, 77, 102, 110, 113], combining LLMs with vision encoders [79, 89, 109] to leverage the complementary strengths of language and vision modalities. By fusing textual and visual representations, LVLMs can ground language in visual contexts, enabling a more comprehensive understanding and generation of multimodal content [5, 11, 14, 20, 27, 51, 60, 95].

LVLMs for High-Resolution Understanding. Large Vision-Language Models (LVLMs) often employ CLIPViT as the visual encoder for vision-dependent tasks. However, the visual encoder’s reliance on low resolutions, such as 224 × 224 or 336 × 336 pixels, limits its effectiveness for high-resolution tasks like OCR and document/chart perception. To enhance high-resolution understanding, recent works have primarily employed the following strategies: (1) High-resolution (HR) visual encoders or dual encoders catering to HR and low-resolution (LR) inputs [36, 48, 66, 97]. For instance, Vary [97] introduces a new image encoder supporting HR inputs, which are then concatenated with LR embeddings from the orig-

[Figure 2]

[Figure 3]

[Figure 4]

Analyze this figure in detail

[Figure 5]

The image is a scientific paper titled "ShareGPT4V: Improving Large Multi-Modal Models with Better Captions" from the University of Science and Technology of China and Shanghai AI Laboratory. The paper discusses the comparison between the captions in their proposed ShareGPT4V dataset and those utilized by recent large multi-modal models (LMMs). The captions are compared in terms of their quality and performance. The paper also highlights the remarkable performance of the proposed LMMs based on a simple architecture that has remarkable performance across a majority of the multi-modal benchmarks.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

| |
|---|

| |
|---|

| |
|---|

[Figure 10]

Which model has the best performance on the MMBench and what is its exact number?

| |
|---|

[Figure 11]

According to the paper, ShareGPT4V-7B has the best performance on the MMBench with a score of 68.8.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Where can I get this project?

[Figure 17]

You can get this project from the following GitHub link: https://ShareGPT4V.github.io

The HD image (2550 x 3300)

[Figure 18]

[Figure 19]

[Figure 20]

What is the algorithm of this flow chart？

[Figure 21]

The flow chart in the image is an algorithm for finding the maximum element in an array. Here is a detailed explanation of the algorithm:

- 1. Declare an integer array having size m.
- 2. Input the m numbers into the array.
- 3. Initialize max as the first element of the array.
- 4. For each element in the array, starting from the second element: If the current element is greater than max, set max to the current element.
- 5. At the end of the loop, max will hold the maximum element in the array.
- 6. Output the maximum element.

This algorithm iterates through the array, comparing each element to the current maximum. If a larger element is found, it becomes the new maximum. At the end of the loop, the maximum element is stored in the variable max, and it is then outputted

The HD image (1888 x 3172)

[Figure 22]

[Figure 23]

[Figure 24]

Conduct an in-depth analysis of the chart.

[Figure 25]

[Figure 26]

The line chart presents the operations cost by department in US dollars for various categories from May 2018 to Mar 2019. The categories include Human Resources (HR), Information Technology (IT), Payroll, Support, Engineering, Facilities, Finance, and Sales. Each category is represented by a different color and line on the graph. The y-axis represents the cost in thousands of US dollars, while the x-axis represents the months from May 2018 to Mar 2019. The graph shows fluctuations in the cost for each category throughout the months, with some categories experiencing peaks and troughs at different times.

[Figure 27]

| |
|---|

| |
|---|

| |
|---|

[Figure 28]

Which department has the highest cost in most cases?

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Payroll has the highest cost in most cases, as indicated by the purple line on the graph.

The HD image (1475 x 675)

- Figure 2. Chat with InternLM-XComposer2-4KHD. Some regions of the input HD images are zoomed in for better visualization.

[Figure 34]

[Figure 35]

[Figure 36]

Illustrate the fine details present in the image.

[Figure 37]

[Figure 38]

The image is a vibrant and colorful infographic that provides an overview of 7 graphic design trends that are expected to dominate in 2021. The infographic is divided into 7 sections, each representing a different trend.

[Figure 39]

[Figure 40]

[Figure 41]

At the top, there's a section titled "Muted Color Palettes", which introduces the concept of muted colors as a trend for 2021.

| |
|---|

This is followed by a section on "Simple Data Visualizations", highlighting the goal of making complex data easy to understand.

[Figure 42]

[Figure 43]

| |
|---|

The third section focuses on "Geometric Shapes

Everywhere", showcasing how designers are using geometric shapes in their designs.

The fourth section delves into "Flat Icons and Illustrations", explaining how brands are using flat icons and illustrations in their social media graphics, website design, and more.

| |
|---|

[Figure 44]

[Figure 45]

The fifth section introduces "Classic Serif Fonts", discussing how these fonts are seen as classic, elegant, and trustworthy.

| |
|---|

The sixth section explores "Social Media Slide Decks", illustrating how slide decks have become a visual way of communicating longer messages on Instagram and LinkedIn.

[Figure 46]

[Figure 47]

| |
|---|

Finally, the seventh section discusses "Text Heavy Videos", predicting that people will likely continue to work remotely well into 2021, leading to an increase in video content that uses text onscreen to communicate messages.

| |
|---|

Overall, the infographic provides a comprehensive overview of the 7 graphic design trends that are expected to shape the design landscape in 2021.

[Figure 48]

[Figure 49]

[Figure 50]

What is the detailed explanation of the third part?

| |
|---|

[Figure 51]

The third part of the image is titled "Geometric Shapes Everywhere". It explains that last year, designers used a lot of flowing and abstract shapes in their designs. However, this year, they have been replaced with rigid, hard-edged geometric shapes and patterns. The hard edges of a geometric shape create a great contrast against muted colors.

[Figure 52]

[Figure 53]

| |
|---|

[Figure 54]

Where can I find full guide?

[Figure 55]

[Figure 56]

[Figure 57]

You can find the full guide on vennage.com/blog/graphic-design-trends.

The HD image (816 x 5133)

###### Figure 3. Chat with InternLM-XComposer2-4KHD on ultra-high HD image with the 816 × 5133 resolution. Some regions of the input HD images are zoomed in for better visualization.

inal CLIP visual encoder. Similarly, CogAgent [36] and Mini-Gemini [48] also separate HR and LR images using distinct vision encoders, subsequently merging their features using a cross-attention module. In contrast, our approach offers a more simplified solution and shows advantages for varying resolutions and aspect ratio inputs. (2) Cropped image patches [37, 46, 50, 51, 59, 99, 101]. For example, Monkey [50] employs sliding windows to segment images into patches, subsequently processing them with LoRA fine-tuning. TextMonkey [59] further proposes shifted window attention and token resampler to consider the connections among different patches. These approaches are confined to either a few predefined high-resolution settings [36, 46, 48, 50, 51, 55, 59, 66, 97] or a limited range of resolutions [37, 99]. Conversely, our method devises a dynamic image partition strategy to support the scaling from 336 pixels to 4K resolution, and the maximum resolution is larger than previous approaches (e.g., 1.5k for Monkey [50] and 2k for UReader [101]).

LVLMs for Document Understanding. Document understanding involves analyzing and comprehending various digital documents, such as figures, tables, and academic papers. Many document understanding tasks require models to handle high-resolution inputs, complex layouts, various aspect ratios, and diverse document formats. To enhance the capabilities of LVLMs for document understanding, several works have collected and constructed high-quality document instruction tuning data, including LLaVAR [112], mPLUG-DocOwl [100] and TGDoc [96]. DocPediaDocPedia [30] processes document inputs in the frequency domain. Some previous works have improved document understanding ability by designing special modules for highresolution inputs, such as HR and LR encoders [36, 97] or cropped image patches [59, 99, 101]. Our InternLMXComposer2-4KHD first scales to 4K resolution inputs and demonstrates strong document understanding ability on OCR-related benchmarks. Also, our approach also achieves comparable results on other general LVLM benchmarks like perception and reasoning [15, 33, 57, 61].

### 3. Method

#### 3.1. Model Architecture.

The model architecture of InternLM-XComposer2-4KHD mainly follows the design of InternLM-XComposer2[27] (XComposer2 in the following for simplicity.), including a light-weight Vision Encoder OpenAI ViT-Large/14, Large Language Model InternLM2-7B, and Partial LoRA for efficient alignment. We recommend the readers to the XComposer2 paper for more details.

[Figure 58]

[Figure 59]

[Figure 60]

|\n| |
|---|---|
|\n| |

| | |
|---|---|
| | |

Flatten

Encode & Merge

Resize

Dynamic Image Partition

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|\n|
|---|
|\n|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Flatten

Encode & Merge

|\n|
|---|
|\n|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

Figure 4. The illustration of processing high-resolution input.

#### 3.2. High-Resolution Input.

Dynamic Image Partition. Utilizing a static input image size for processing high-resolution images, particularly those with varying aspect ratios, is neither efficient nor effective. To overcome this limitation, we introduce a dynamic image partitioning approach, as shown in Figure 4. Our method strategically segments the image into smaller patches, while maintaining the integrity of the original image’s aspect ratio.

Given a maximum partition number H, the image x with size [h,w] is resized and padded to the new image xˆ with size [ph × 336,pw × 336]. This process is subject to the following constraints:

pw × ph ≤ H; ph = ⌈pw × h/w⌉ (1)

here pw and ph represent the number of patches in each row and column, respectively. We then split the xˆ into ph × pw non-overlapped patches. Each patch is a small image with 336×336 size and we treat these patches as individual inputs for the ViT.

In the following, we use ‘HD-H’ to represent our highresolution setting with the constraint of H patches. For example, the ’HD-9’ allows up to 9 patches, including a range of resolutions such as 1008×1008, 672×1344, 336×3024, etc.

Global-Local Format. For each input image, we present it to the model with two views. The first is the global view, where the image is resized to a fixed size (in our case, 336 × 336). This provides a macro understanding of the image. Empirically, we have found this to be crucial for the LVLM to correctly understand the image. The second view is the local view. We divide the image into patches using the previously mentioned Dynamic Image Partition strategy and extract features from each patch. Following feature extraction, the patches are reassembled into a large feature map. The feature map is then flattened to the final local features after a straightforward token merging process.

##### Image 2D Structure Newline Indicator. Given that an

Task Dataset General Semantic Alignment ShareGPT4V-PT [14], COCO [17], Nocaps [1], TextCaps [86], LAION400M [80], SBU [75], CC 3M [83] World Knowledge Alignment Concept Data [110] Vision Capability Enhancement WanJuan [35], Flicker[103], MMC-Inst[54], RCTW-17[84], CTW[106], LSVT[88], ReCTs[111], ArT[22]

Table 1. Datasets used for Pre-Training. The data are collected from diverse sources for the three objectives. The newly added data is highlighted with red.

image has a 2D structure and the image ratio is dynamic, the number of tokens for each row can vary across different images. This variation can potentially confuse the LVLM, making it difficult to determine which tokens belong to the same row of the image and which ones belong to the next row. This confusion may hinder the LVLM’s ability to understand the 2D structure of the image, which is crucial for comprehending structural image content such as documents, charts, and tables. To address this issue, we introduce a learnable newline (‘\n’) token at the end of each row of the image features before the flattening. Finally, we concatenate the global and local views, inserting a special ‘separate’ token between them to distinguish the two views.

#### 3.3. Pre-Training

During the pre-training phase, the LLM is frozen while both the vision encoder and Partial LoRA are fine-tuned to align the visual tokens with the LLM. The pre-training data mainly follow the design in XComposer2 which is curated with three objectives in mind: 1) general semantic alignment, 2) world knowledge alignment, 3) vision capability enhancement. In this paper, we focus on high-resolution and structural image understanding. Therefore, we have collected more related data to enhance this specific capability. As shown in Table.1, we have utilized a diverse OCR dataset for this purpose.

In practice, we employ the OpenAI CLIP ViT-L-14-336 as the vision encoder. Different from XComposer2, We keep the ViT resolution as 336×336 and increase the input resolution with more patches. For the Dynamic Image Partition strategy, we use ‘HD-25’ for the pertaining. For each image or patch, the image token number is decreased to 1/4 with a simple merge operation. We concatenate the nearby

- 4 tokens into a new token through the channel dimension, then align it with the LLM by an MLP. The ‘separate’ and ‘\n’ token are randomly initialized. For the Partial LoRA, we set a rank of 256 for all the linear layers in the LLM decoder block. Our training process involves a batch size of 4096 and spans across 2 epochs. The learning rate linearly increases to 2 × 10−4 within the first 1% of the training steps. Following this, it decreases to 0 according to a cosine decay strategy. To preserve the pre-existing knowledge of the vision encoder, we apply a layer-wise learning rate (LLDR) decay strategy, and the decay factor is set to 0.90.

Task Dataset Caption ShareGPT4V [14], COCO [17],Nocaps [1] General QA VQAv2 [3], GQA [38], OK-VQA [67]

VD [26], RD[13], VSR[53], Science QA AI2D [42], SQA [63], TQA[43], IconQA[65] Chart QA DVQA [40], ChartQA, ChartQA-AUG [68] Math QA MathQA [104], Geometry3K[62], TabMWP[64],

CLEVR-MATH[52]/Super[49] World Knowledge QA A-OKVQA [81],KVQA [82], ViQuAE[44] OCR QA TextVQA[87], OCR-VQA[72], ST-VQA[8] HD-OCR QA InfoVQA[69], DocVQA[70] Conversation LLaVA-150k [56], LVIS-Instruct4V [94]

ShareGPT-en&zh [21], InternLM-Chat[91]

Table 2. Datasets used for Supervised Fine-Tuning. We collect data from diverse sources to empower the model with different capabilities. The newly added data is highlighted with red.

#### 3.4. 4KHD Supervised Fine-tuning

After the pre-training, we empower the model to understand high-resolution images and solve diverse challenges. Different from previous perception tasks (e.g., VQAv2, GQA) which typically answer questions based on the noticeable object in the image. OCR-related tasks depend on a detailed understanding of text within a high-resolution image. For instance, in InfoVQA, the length of the longer side of 50% of the images exceeds 2000 pixels. Low-resolution inputs can distort the dense text information, causing the model to fail in its understanding. However, we have observed a resolution saturation problem with the aforementioned perception tasks, where the influence of resolution becomes negligible.

To address this, we introduce a mixed-resolution training strategy for more efficient training. For tasks requiring high resolution, we employ the ‘HD-55’ setting during training. This allows for the input of 4K (3840 × 1600) images without necessitating additional image compression. These tasks are referred to as the HD-OCR QA tasks in Table 2. For other tasks, we implement a dynamic-resolution strategy. Images are resized to fall within a range between their original size and the size specified by the ‘HD25’ setting. This dynamic approach enhances the robustness of the LVLM against differences in input resolution, thereby enabling the LVLM to utilize a larger resolution during inference. For instance, we have observed that using the ‘HD30’

Doc Chart Info Text OCR MM Math

MMB MMB SEED QBench MM- Hall

Method

AI2D MMMU MME

VQA QA VQA VQA Bench Star Vista EN CN Image Test Vet Bench

Open-Source [37] [37] [37] [36] [36] [55] [55] [55] [20] [2] [55] [55] [55] [110] [95] [50] Previous SOTA 8B 8B 8B 18B 18B 35B 35B 35B 40B 34B 35B 35B 35B 8B 17B 10B

82.2 70.2 44.5 76.1 59.0 52.1 39.0 78.9 51.6 2050.2 81.1 79.0 75.7 64.4 54.5 39.3 Closed-source API

GPT-4V 88.4 78.5 75.1 78.0 51.6 57.1 47.8 75.5 56.8 1,926.5 77.0 74.4 69.1 74.1 56.8 46.5 Gemini-Pro 88.1 74.1 75.2 74.6 68.0 42.6 45.8 70.2 47.9 1,933.3 73.6 74.3 70.7 70.6 59.2 45.2

IXC2-VL 57.7 72.6 34.4 70.1 53.2 55.4 57.6 81.2 41.4 2,220.4 80.7 79.4 74.9 72.5 46.7 41.0 IXC2-4KHD 90.0 81.0 68.6 77.2 67.5 54.1 57.8 80.9 39.7 2,204.9 80.2 77.7 74.7 71.8 54.9 40.9

- Table 3. Comparison with closed-source APIs and previous open-source SOTAs. Our InternLM-XComposer2-4KHD gets SOTA results in 6 of the 16 benchmarks with only 7B parameters, showing competitive results with current closed-source APIs. The best results are bold and the second-best results are underlined.

Method LLM MMStar MathVista AI2D MMEP MMEC MMB MMBCN SEEDI QBenchT MM-Vet

Qwen-VL-Chat Qwen-7B 37.5 33.8 63.0 1,487.5 360.7 60.6 56.7 58.2 61.7 47.3 ShareGPT4V Vicuna-7B 33.0 25.8 58.0 1,567.4 376.4 68.8 62.2 69.7 - 37.6 Monkey Qwen-7B 38.3 34.8 62.5 1,522.4 401.4 72.4 67.5 68.9 - 33.0 CogVLM-17B Vicuna-7B 36.5 34.7 63.3 - - 65.8 55.9 68.8 - 54.5 LLaVA-XTuner InernLM2-20B - 24.6 65.4 - - 75.1 73.7 70.2 - 37.2 LLaVA-1.5 Vicuna-13B 32.8 26.1 61.1 1,531.3 295.4 67.7 63.6 68.2 61.4 35.4 LLaVA-Next Vicuna-13B 38.3 32.4 72.2 1,445.0 296.0 70.0 68.5 71.4 - 44.9 InternLM-XC InernLM-7B - 29.5 56.9 1,528.4 391.1 74.4 72.4 66.1 64.4 35.2

IXC2-VL InernLM2-7B 55.4 57.6 81.2 1,712.0 530.7 80.7 79.4 74.9 72.5 46.7 IXC2-4KHD InernLM2-7B 54.1 57.8 80.9 1,655.9 548.9 80.2 77.7 74.7 71.8 54.9

- Table 4. Comparison with open-source SOTA methods. IXC2-4KHD outperforms competitors in most benchmarks. The best results are bold and the second-best results are underlined.

setting yields better results on most OCR-related tasks when the LVLM is trained under the ‘HD25’ setting.

In practice, we jointly train all the components with a batch size of 2048 over 3500 steps. Data from multiple sources are sampled in a weighted manner, with the weights based on the number of data from each source. As the ‘HD55’ setting has double image tokens than the ‘HD-25’, we adjust the data loader to enable different batch sizes for them and adjust their weight accordingly. The maximum learning rate is set to 5 × 10−5, and each component has its own unique learning strategy. For the vision encoder, we set the LLDR to 0.9, which aligns with the pretraining strategy. For the LLM, we employ a fixed learning rate scale factor of 0.2. This slows down the update of the LLM, achieving a balance between preserving its original capabilities and aligning it with vision knowledge.

### 4. Experiments

In this section, we validate the benchmark performance of our InternLM-XComposer2-4KHD (IXC2-4KHD in the following for simplicity) after supervised fine-tuning.

#### 4.1. LVLM Benchmark results.

In Table 3 and Table 4, we compare our IXC24KHD on a list of benchmarks with both SOTA

open-source LVLMs and closed-source APIs. Here we report results in DocVQA[70], ChartQA[68], InfographicVQA[69], TextVQA[87], OCRBench[58], MMStar[15], MathVista[61], MMMU[107], AI2D[42], MME [31], MMBench (MMB) [57], MMBench-Chinese (MMBCN) [57], SEED-Bench Image Part (SEEDI)[45], QBench-Testset (QBenchT)[98], MM-Vet [105], HallusionBench (HallB)[34]. The evaluation is mainly conducted on the OpenCompass VLMEvalKit[24] for the unified reproduction of the results.

Comparison with Closed-Source APIs. As demonstrated in Table 3, IXC2-4KHD exhibits competitive performance across a variety of benchmarks, rivaling that of ClosedSource APIs. Owing to its high-resolution input, IXC24KHD achieves a score of 90.0% on DocVQA and 81.0% on ChartQA, thereby surpassing GPT-4V and Gemini-Pro with a non-trivial margin. In the challenging InfographicVQA task, our model is the first open-source model that is close to the performance of Closed-Source APIs, exceeding the performance of previous open-source models by nearly 20%. In addition to OCR-related tasks, IXC2-4KHD is a general-purpose Large Vision-Language Modal that excels in semantic-level tasks, demonstrating competitive results.

Comparison with Open-Source Models. We also conduct a comprehensive comparison with open-source LVLMs un-

|50<br><br>55<br><br>60<br><br>65<br><br>70<br><br>75<br><br>80<br><br>85<br><br>90<br><br>InfoVQA DocVQA TextVQA ChartQA MMBench MME Seed AI2D<br><br>HD-9 (1561 Tokens) HD-16 (2653 Tokens) HD-25 (4057 Tokens) 4K HD (8737 Tokens)<br><br>Val Val Val Test EN-Test * Test<br><br>Figure 5. Influence of Training Resolution. High-resolution training is critical for HD-OCR tasks, while its gain on other tasks is minor.|
|---|

Model Model Size Max Resolution DocVQATest ChartQATest InfoVQATest TextVQAV al OCRBench TextMonkey[59] 9B 896x896 73.0 66.9 28.6 65.6 55.8 LLaVA-UHD [99] 13B 1008x672 — — — 67.7 CogAgent [36] 17B 1024x1024 81.6 68.4 44.5 76.1 59.0 UReader [101] 7B 2240x2240 65.4 59.3 42.2 57.6 DocOwl 1.5 [37] 8B 1344x1344 82.2 70.2 50.7 68.6 IXC2-4KHD 8B 3840x1600 90.0 (+7.8) 81.0 (+10.8) 68.6 (+17.9) 77.2 (+1.2) 67.5 (+8.5)

- Table 5. High-resolution Evaluation. IntenrLM-XComposer2-4KHD has the largest input resolution and outperforms open-source LVLMs which are specifically tuned for document understanding.

der a similar model scale. As shown in Table 4, our model significantly outperforms existing open-source models, achieving competitive results across all benchmarks. Notably, the InternLM-XComposer2 series is the only method that achieves a higher than 50% score on the challenging MMStar benchmark.

High-resolution Understanding Evaluation. Then we compare IXC2-4KHD with models that are specifically designed for high-resolution understanding tasks. We report the results of 5 high-resolution benchmarks in Table 5, as a general LVLM, IXC2-4KHD shows superb performance on these tasks and outperforms competitors with a large margin. For example, IXC2-4KHD gets 68.6% on InfographicVQA, surpassing recent DocOwl 1.5 with +17.9%. For the OCRBench, IXC2-4KHD gets 67.5%, outperforms CogAgent with +8.5%.

#### 4.2. Dive into Resolution

High-Resolution Training is Critical for HD-OCR tasks. We study four resolution settings: HD-9 (1561 image tokens at most, we simply the statement if the following), HD16 (2653 tokens), HD-25 (4057 tokens), and 4KHD (8737 tokens). Here we report the validation set of InfoVQA, DocVQA, and TextVQA, test set of ChartQA and AI2D, MMBench EN-Test, and a 2k subset of SEEDBench (we denote it as SEED∗). In the following experiments, we report results on the above benchmarks by default.

As illustrated in Fig.5, we note a significant improvement in the HD-OCR tasks as the resolution increases. For instance, the model achieves only a 50.5% score on the InfographicVQA with the HD-9 setting. However, when we

Train Eval Doc Info Text Chart MMB MME SEED∗ HD9

HD9 79.4 50.5 73.8 78.2 79.5 2,201 76.6

- HD16 83.0 58.6 74.3 75.8 79.3 2,198 76.7

HD16

- HD16 84.9 60.8 75.7 80.1 80.2 2,129 75.7 HD25 85.9 62.1 75.8 79.1 80.1 2,100 75.4

HD25 87.0 63.6 76.0 80.3 78.5 2,209 74.9 HD30 87.4 64.6 76.2 79.4 78.9 2,173 74.3

HD25

Table 6. Influence of Inference Resolution. The model achieves better performance on text-related tasks when the inference resolution is higher than its training resolution.

switch to the HD-16 setting, we observe a performance gain of +10.2%. The performance continues to improve as the resolution increases, with saturation not observed even for the 4KHD setting. Due to computational constraints, we defer the exploration of the upper bound of improvement to future work. In terms of other OCR-related tasks, the performance gain attributable to increased resolution is relatively minor. For the perception-related benchmarks, performance is saturated on the resolution that only has negligible difference between the four settings.

Higher Inference Resolution Leads to better results on Text-related Tasks. An intriguing observation from our experiments is that our model, when inferring with a slightly higher resolution, tends to yield improved results on textrelated tasks. We present the results of HD-9, HD-16, and HD-25 in Table 6. For instance, IXC2-HD9 achieves a 50.5% score on InfographicVQA. When we infer with HD16, we see a performance gain of +8.1%, without additional training. Similar improvements are also observed with IXC2-HD16 and IXC2-HD25. We posit that the dy-

Model Doc Info Text Chart MMB MME SEED∗

HD9 79.4 50.5 73.8 78.2 79.5 2201 76.6 + w/o global-view 78.1 47.9 71.2 77.9 75.1 2019 76.2

Table 7. Influence of Global-View in the Input. Global-view is critical for most benchmarks.

namic image token length used in training enhances the robustness of the LVLM, leading to better results when the text in the image is more ‘clear’ in the higher resolution input. Conversely, the results on ChartQA consistently degrade under this setting. This could be due to the model becoming confused about the chart structure when the resolution is altered. Additionally, similar to the observation from Figure 5, the impact of resolution on perception-related benchmarks appears to be quite minor.

Visualization Results. We provide the visualization results on ultra-high HD images in Figure 2 and Figure 3. Please refer to the appendix for more results.

#### 4.3. High-Resolution Strategy Ablation

The Role of Global-View. We first examine the impact of the global view in our Global-Local Format. As indicated in Table 7, we find that the global view is essential for the LVLM to accurately comprehend the input image. When it is removed, the model performs worse across all benchmarks. For instance, the model experiences a −4.4% drop in performance on the MMBench EN-Test without the global view. We contend that the global view offers a general macro understanding of the image, which the model struggled to derive from the large number of tokens in the local view.

The Role of the Newline Token. We incorporate a special newline token at the end of each row of the image features before the flattening operation. This token serves as an indicator of the image’s 2D structure. We examine its impact on both the HD-9 and 4KHD strategies in Table 8. When a fixed high-resolution strategy HD-9 is employed, we observe that the benefit derived from the newline token is minor. This could be attributed to the LVLM’s ability to handle limited differences in image ratios after training. However, when we implement a more challenging 4KHD (HD-25 + HD-55) strategy, which exhibits significant diversity in both image ratio and token number, the LVLM demonstrates a notable decline in performance on OCR-related tasks without the newline indicator. This finding supports our hypothesis that the LVLM struggles to comprehend the shape of the image when the image tokens are directly flattened into a 1D sequence. The newline token can assist the model in better understanding the structure of the image.

Influence of Token Merging Strategy. In practice, we employ a simple merging strategy that concatenates four adjacent tokens along the channel dimension. We have

Model ‘\n’ Doc Info Text Chart MMB MME SEED∗

HD9 × 79.5 50.3 74.0 78.2 79.1 2206 75.9 HD9 ✓ 79.4 50.5 73.8 78.2 79.5 2201 76.6

4KHD × 88.1 67.4 75.9 80.4 79.9 2232 76.4 4KHD ✓ 89.0 69.3 77.2 81.0 80.2 2205 76.2

- Table 8. Influence of Indicator ‘\n’ in the Image Features. ‘\n’ helps LVLM understand structural images when the input resolution is dynamic and large.

Strategy Doc Info Text Chart MMB MME SEED∗

Re-Sampler 86.2 67.1 75.3 78.8 79.6 2124 74.2 C-Abstractor 88.6 69.5 77.1 80.6 80.4 2236 76.7 Concat 89.0 69.3 77.2 81.0 80.2 2205 76.2

- Table 9. Ablation on Token Merging Strategy. Both the simple concatenation operation and the C-Abstractor works well.

found this approach to be effective in reducing the number of image tokens efficiently. Here we study the influence of different token-merging strategies under the 4KHD setting. In Table 9, we study two additional strategies: ReSampler[5] and C-Abstractor[12], with their default setting and the same compressing rate 0.25, i.e., reducing an image with 576 tokens to 144 tokens. Results show that both concatenation and C-Abstractor work well and get similar results on most benchmarks, this observation is also consistent with the study in MM-1[71] that the influence of the connector is minor. However, the Re-Sampler performs worse than the other methods with a noticeable margin. We argue this is caused by the learnable queries used for gathering information requiring a great number of data for training, our pre-training data is somewhat lightweight for it to converge fully.

### 5. Conclusion

In this paper, we propose the InternLM-Xcomposer24KHD that exceeds the performance of previous opensource models on OCR-related tasks and also achieves competitive results on general-purpose LVLM benchmarks. Thanks to our dynamic resolution and automatic patch configuration, our model supports a maximum training resolution of up to 4K HD. We also integrate a global view patch to support the macro understanding and a learnable newline token to handle the various input image resolutions. Our model’s performance continues to improve as the training resolution increases for HD-OCR tasks. Notably, we do not observe any performance saturation even for the 4KHD setting, and we have not explored the upper bound due to the computational burden increasing with higher-resolution inputs. In future work, we plan to explore efficient solutions for accurate LVLM training and inference, enabling our model to handle even higher resolutions while maintaining computational efficiency.

### References

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8948–8957,

2019. 6

- [2] 01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024. 7
- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In International Conference on Computer Vision (ICCV), 2015. 6
- [4] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv.org, 2023. 2
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv.org, 2023. 2, 9
- [6] Baichuan. Baichuan 2: Open large-scale language models. arXiv.org, 2023. 2
- [7] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 2
- [8] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marc¸al Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4291–4301, 2019. 6
- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems (NeurIPS), 33:1877– 1901, 2020. 2
- [10] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng

- Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024. 1, 2
- [11] Yuhang Cao, Pan Zhang, Xiaoyi Dong, Dahua Lin, and Jiaqi Wang. DualFocus: Integrating macro and micro perspectives in multi-modal large language models. arXiv preprint arXiv:2402.14767, 2024. 2
- [12] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. arXiv preprint arXiv:2312.06742,

2023. 9

- [13] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv.org, 2023. 6
- [14] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 1, 2, 6
- [15] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 5, 7
- [16] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, Siamak Shakeri, Mostafa Dehghani, Daniel Salz, Mario Lucic, Michael Tschannen, Arsha Nagrani, Hexiang Hu, Mandar Joshi, Bo Pang, Ceslee Montgomery, Paulina Pietrzyk, Marvin Ritter, AJ Piergiovanni, Matthias Minderer, Filip Pavetic, Austin Waters, Gang Li, Ibrahim Alabdulmohsin, Lucas Beyer, Julien Amelot, Kenton Lee, Andreas Peter Steiner, Yang Li, Daniel Keysers, Anurag Arnab, Yuanzhong Xu, Keran Rong, Alexander Kolesnikov, Mojtaba Seyedhosseini, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali-x: On scaling up a multilingual vision and language model, 2023. 2
- [17] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server, 2015. 1, 6
- [18] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Soricut. Pali-3 vision language models: Smaller, faster, stronger, 2023. 2
- [19] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman,

- Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali: A jointly-scaled multilingual languageimage model, 2023. 2
- [20] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 2, 7
- [21] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. 1, 6
- [22] Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1571–1576. IEEE, 2019. 6
- [23] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv.org, 2022. 2
- [24] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass,

2023. 7

- [25] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 2

- [26] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jos´e M.F. Moura, Devi Parikh, and Dhruv Batra. Visual Dialog. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2017. 6

- [27] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024. 2, 5
- [28] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch,

- and Pete Florence. Palm-e: An embodied multimodal language model. In arXiv preprint arXiv:2303.03378, 2023. 2
- [29] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, 2022. 1
- [30] Hao Feng, Qi Liu, Hao Liu, Wengang Zhou, Houqiang Li, and Can Huang. DocPedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. arXiv preprint arXiv:2311.11810, 2023. 5
- [31] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 1, 7
- [32] Chaoyou Fu, Renrui Zhang, Zihan Wang, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, Yunhang Shen, Mengdan Zhang, Peixian Chen, Sirui Zhao, Shaohui Lin, Deqiang Jiang, Di Yin, Peng Gao, Ke Li, Hongsheng Li, and Xing Sun. A challenger to gpt-4v? early explorations of gemini in visual expertise. arXiv preprint arXiv:2312.12436, 2023. 2
- [33] Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041, 2023. 1, 5
- [34] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models, 2023. 7
- [35] Conghui He, Zhenjiang Jin, Chaoxi Xu, Jiantao Qiu, Bin Wang, Wei Li, Hang Yan, Jiaqi Wang, and Da Lin. Wanjuan: A comprehensive multimodal dataset for advancing english and chinese large models. ArXiv, abs/2308.10755,

2023. 6

- [36] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914,

2023. 2, 5, 7, 8

- [37] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895, 2024. 2, 5, 7, 8
- [38] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 6
- [39] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne

- Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. 1, 2
- [40] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656,

2018. 6

- [41] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 2

- [42] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 6, 7

- [43] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 4999–5007, 2017. 6
- [44] Paul Lerner, Olivier Ferret, Camille Guinaudeau, Herv´e Le Borgne, Romaric Besanc¸on, Jos´e G Moreno, and Jes´us Lov´on Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3108–3120, 2022. 6
- [45] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023. 7
- [46] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multimodality model, 2023. 2, 5
- [47] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv.org, 2023. 2
- [48] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-Gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 2, 5

- [49] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14963–14973, 2023. 6
- [50] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607, 2023. 2, 5, 7
- [51] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin

- Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 2, 5
- [52] Adam Dahlgren Lindstr¨om and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. arXiv preprint arXiv:2208.05358, 2022. 6
- [53] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 2023. 6
- [54] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. arXiv preprint arXiv:2311.10774, 2023. 6
- [55] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. 2, 5, 7
- [56] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv.org, 2023. 6
- [57] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhnag, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023. 1, 5, 7
- [58] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models, 2024. 2, 7
- [59] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473, 2024. 2, 5, 8
- [60] Ziyu Liu, Zeyi Sun, Yuhang Zang, Wei Li, Pan Zhang, Xiaoyi Dong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. RAR: Retrieving and ranking augmented mllms for visual recognition. arXiv preprint arXiv:2403.13805, 2024. 2
- [61] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024. 5, 7
- [62] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021. 6
- [63] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022. 6
- [64] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, SongChun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint

- arXiv:2209.14610, 2022. 6
- [65] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021. 6
- [66] Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. Kosmos-2.5: A multimodal literate model, 2023. 2, 5
- [67] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 6
- [68] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 1, 2, 6, 7
- [69] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 1, 2, 6, 7
- [70] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 1, 2, 6, 7
- [71] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024. 9
- [72] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, 2019. 6
- [73] OpenAI. Chatgpt. https://openai.com/blog/ chatgpt, 2022. 1, 2
- [74] OpenAI. Gpt-4 technical report, 2023. 1, 2
- [75] Vicente Ordonez, Girish Kulkarni, and Tamara L. Berg. Im2text: Describing images using 1 million captioned photographs. In Neural Information Processing Systems (NIPS), 2011. 6
- [76] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems (NeurIPS), 35:27730–27744, 2022. 2
- [77] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv.org,

2023. 2

- [78] Qwen. Introducing qwen-7b: Open foundation and humanaligned models (of the state-of-the-arts), 2023. 1, 2
- [79] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine learning (ICML), pages 8748–8763. PMLR, 2021. 2
- [80] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 6
- [81] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pages 146–162. Springer, 2022. 6
- [82] Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. Kvqa: Knowledge-aware visual question answering. In Proceedings of the AAAI conference on artificial intelligence, 2019. 6
- [83] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 6
- [84] Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu, and Xiang Bai. Icdar2017 competition on reading chinese text in the wild (rctw-17). In 2017 14th iapr international conference on document analysis and recognition (ICDAR), volume 1, pages 1429–1434. IEEE, 2017. 6
- [85] Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: How far are we from automating front-end engineering?, 2024. 2
- [86] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer, 2020. 6
- [87] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 1, 2, 6, 7
- [88] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1557–1562. IEEE, 2019. 6
- [89] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Alpha-CLIP: A clip model focusing on wherever you want. arXiv preprint arXiv:2312.03818, 2023. 2
- [90] Gemini Team. Gemini: A family of highly capable multimodal models, 2023. 1, 2
- [91] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https://

- github.com/InternLM/InternLM, 2023. 1, 2, 6
- [92] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv.org, 2023.
- [93] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models,

2023. 1, 2

- [94] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv preprint arXiv:2311.07574, 2023. 6
- [95] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2023. 2, 7
- [96] Yonghui Wang, Wengang Zhou, Hao Feng, Keyi Zhou, and Houqiang Li. Towards improving document understanding: An exploration on text-grounding via mllms. arXiv preprint

- arXiv:2311.13194, 2023. 5

[97] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. arXiv preprint

- arXiv:2312.06109, 2023. 2, 5

- [98] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181, 2023. 7
- [99] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703, 2024. 2, 5, 8
- [100] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mPLUG-DocOwl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023. 5
- [101] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023. 5, 8
- [102] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv.org, 2023. 2
- [103] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. 6

- [104] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 6
- [105] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 7
- [106] Tai-Ling Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 34(3):509–521, 2019. 6
- [107] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023. 1, 7
- [108] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130b: An open bilingual pretrained model. In The Eleventh International Conference on Learning Representations (ICLR), 2023. 2
- [109] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-CLIP: Unlocking the long-text capability of clip. arXiv preprint arXiv:2403.15378, 2024. 2
- [110] Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlmxcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023. 2, 6, 7
- [111] Rui Zhang, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, Minghui Liao, Mingkun Yang, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In 2019 international conference on document analysis and recognition (ICDAR), pages 1577–1581. IEEE, 2019. 6
- [112] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. LLaVAR: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 5
- [113] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv.org, 2023. 2

[Figure 67]

[Figure 68]

Give an in-depth description of the visual.

[Figure 69]

[Figure 70]

The image is an infographic from Public Health England that explains the concept of contact tracing in the context of preventing the spread of coronavirus (COVID-19). Contact tracing is a fundamental part of outbreak control used by public health professionals around the world to prevent the spread of infections.

The infographic is divided into several sections, each providing a different aspect of the contact tracing process. The first section explains that a contact is assessed to see how close they were to a confirmed case, which could include someone living with the case, or someone who has been in direct contact with them or their bodily fluids. Following this assessment, the contacts are categorised into low or high risk and provided with advice on what they should do.

The second section explains that if a contact is at high risk of infection, they may be asked to selfisolate, remaining in their home for 14 days and staying away from work, school or public places. If a contact is at low risk, they don't require self-isolation.

The third section explains that if the person develops symptoms, they would be tested and provided with specialist care if they have the novel coronavirus (COVID-19).

Finally, the infographic emphasizes the importance of recovery time, as contact displays no symptoms of coronavirus (COVID-19) the patient makes a full recovery, they can go back to daily lives as normal and pose no risk to

The HD image (986 x 2560) others.

Figure 6. Chat with InternLM-XComposer2-4KHD. Please refer to the link for better visualization.

[Figure 71]

[Figure 72]

[Figure 73]

Detect all the word in the given image

[Figure 74]

American Heart Association. VennGAGE. How to Develop Healthy Eating Habits. And Reduce the Risk of Heart Disease. Your best defense is a healthy lifestyle filled with physical activity and nutritious meals, while cutting down or avoiding drinking and smoking. The Current Trend 75% of the US population is low in its fruit and vegetables consumption. 1/10 adults meets the recommendation for fruit and vegetable consumption. Men and young adults in general eat the fewest fruits and vegetables compared to any other group. 3 Steps to Eating Healthy 1. Add produce of various colors to your diet to diversify your produce consumption. 2. Use hacks to incorporate more veggies and fruits to your meal, and throughout the day. 3. Take steps to improve access to fruits and vegetables for all.

- 1. Adding Color to Your Diet

Eating 2 to 2% cups (4 to 5 servings) daily of fruits and vegetables may reduce your risk for cardiovascular disease, including heart attack and stroke.

Examples of a single serving of fruit: 1 medium apple, 1 small banana, 1 medium avocado, 4 large strawberries. Examples of a single serving of vegetables: 5 - 8 broccoli florets, 2/3 large sweet potato, 6 baby carrots, 1 cup of raw spinach.

Adding a variety of produce to your diet has a number of health benefits. It helps you: Manage your weight, Control your blood pressure, Support healthy digestion, Reduce the risk of some cancers, such as colon cancer, Reduce the risk of chronic health problems, such as diabetes.

- 2. Use Healthy Eating Hacks

1. Fresh, frozen, canned and dried all count toward your daily servings, so you have plenty of options. 2. Always reserve half of your plate for fruits and vegetables. 3. Choose whole fruits and vegetables. 4. Try to eat a variety of vegetables, instead of the same thing all the time. 5. Look for fruit packed in its own fruit juice and 100% vegetable juice, with no added sugar. 6. Look for low/no-sodium options for canned vegetables, and 100% vegetable juice.

- 3. Take Action for Better Access

In many communities, fruits and vegetables are limited/expensive and only available in corner stores, convenience stores, bodegas and gas stations.

Lack of access can profoundly impact diet. Here's how: Contact your city and state leaders to let them know what food access is in your community.

Find ways to spread the word about nutrition assistance programs, such as SNAP, WIC and school meals.

Create a petition for more variety, improve affordability and advocate for better signage/placement.

Meet with an after-school or daycare program representative to discuss serving more fruits and vegetables for snacks. Organize a letter-writing campaign and set up a meeting with state leaders.

For example, ask for funding to host a farmers market in an underserved community. Sign up for "You're the Cure" and send a note to your Congressperson advocating for healthier meals at school.

It's American Heart Month! Share this infographic with your family, friends and network today. You can carry out all of these actions or just a few. But whichever actions you choose, they pave the way for greater access to nutritious food for your community.

The HD image (816 x 3813)

Figure 7. Chat with InternLM-XComposer2-4KHD. Please refer to the link for better visualization.

