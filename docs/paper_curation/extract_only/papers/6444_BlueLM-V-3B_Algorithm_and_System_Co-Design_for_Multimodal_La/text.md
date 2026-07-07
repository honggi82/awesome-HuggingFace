# arXiv:2411.10640v1[cs.CV]16Nov2024

## BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices

Xudong Lu∗2,1†, Yinghao Chen∗1, Cheng Chen∗1, Hui Tan∗1, Boheng Chen1, Yina Xie1, Rui Hu1, Guanxin Tan1, Renshou Wu1, Yan Hu1, Yi Zeng1, Lei Wu1, Liuyang Bian1, Zhaoxiong Wang1, Long Liu1, Yanzhou Yang1, Han Xiao2,1†,

Aojun Zhou2, Yafei Wen1, Xiaoxin Chen1, Shuai Ren1‡ , Hongsheng Li2

1vivo AI Lab 2CUHK MMLab {luxudong@link,hsli@ee}.cuhk.edu.hk shuai.ren@vivo.com

###### Abstract

The emergence and growing popularity of multimodal large language models (MLLMs) have significant potential to enhance various aspects of daily life, from improving communication to facilitating learning and problem-solving. Mobile phones, as essential daily companions, represent the most effective and accessible deployment platform for MLLMs, enabling seamless integration into everyday tasks. However, deploying MLLMs on mobile phones presents challenges due to limitations in memory size and computational capability, making it difficult to achieve smooth and real-time processing without extensive optimization. In this paper, we present BlueLM-V-3B, an algorithm and system co-design approach specifically tailored for the efficient deployment of MLLMs on mobile platforms. To be specific, we redesign the dynamic resolution scheme adopted by mainstream MLLMs and implement system optimization for hardware-aware deployment to optimize model inference on mobile phones. BlueLM-V-3B boasts the following key highlights: (1) Small Size: BlueLM-V-3B features a language model with 2.7B parameters and a vision encoder with 400M parameters. (2) Fast Speed: BlueLMV-3B achieves a generation speed of 24.4 token/s on the MediaTek Dimensity 9300 processor with 4-bit LLM weight quantization. (3) Strong Performance: BlueLM-V-3B has attained the highest average score of 66.1 on the OpenCompass benchmark among models with ≤ 4B parameters and surpassed a series of models with much larger parameter sizes (e.g., MiniCPM-V-2.6, InternVL2-8B).

###### 1. Introduction

Large language models (LLMs) [10, 118, 119] have gained significant attention due to their potential to solve various complex tasks [120, 128]. Multimodal large language

∗Equal contribution Corresponding author ‡Project lead †Interns at vivo.

MMVet

AI2D (Test)

DocVQA (Test)

61.8

87.8

85.3

MTVQA

MMMU (Val)

32.7

45.1

RealWorldQA

MMStar

66.7

62.3

80.4

48

ChartQA (Test) HallusionBench

829

60.8

MathVista (TestMini)

OCRBench

82.3 83.1

MMBench v1.1 (Test_CN)

MMBench v1.1 (Test_EN)

Phi-3-Vision

InternVL2-4B Qwen2-VL-2B

BlueLM-V-3B (Ours)

MiniCPM-V-2

Figure 1. Comparison with mainstream MLLMs. We compare the performance of several mainstream MLLMs with a parameter count similar to that of BlueLM-V-3B across multiple benchmarks. BlueLM-V-3B leads in the majority of datasets.

models (MLLMs) extend the capabilities by processing and integrating various forms of data—such as text, images, and audio—enabling richer interaction and a deeper understanding of context, which lead to more intuitive user experiences [4, 8, 22, 23, 86, 116, 125, 142]. As research and applications of LLMs and MLLMs continue to evolve, more studies are exploring the feasibility of deploying the models on a variety of devices, including cloudbased platforms [123], desktop PCs [109], and even edge devices [28, 40, 97, 132, 134]. This trend emphasizes the need for optimizing model performance and resource efficiency to ensure accessibility across diverse platforms.

Among available platforms, mobile phones stand out as the most efficient and accessible tool for deploying

MLLMs. Firstly, it enables real-time, on-device processing, allowing users to interact with the model offline. This enhances privacy and reduces latency [28, 97]. Secondly, mobile deployment improves accessibility, allowing users to leverage advanced models anytime and anywhere, such as augmented reality or real-time translation [24, 40, 134]. Lastly, it drives research toward minimizing computational and memory demands to ensure efficient operation on resource-constrained hardware [123, 132].

However, deploying LLMs and MLLMs on mobile phones remains challenging. Firstly, the limited memory capacity of mobile phones restricts the deployment of largeparameter models. For example, a 4-bit quantized LLaMA 7B model requires approximately 4.5 GB of memory, which can impact system fluency due to high memory usage. Secondly, the limited computational power of mobile processors constrains inference speed. For instance, on the MediaTek Dimensity 9300 processor, a 4-bit quantized LLaMA 7B model generates around 10-15 tokens per second, limiting its suitability for real-time applications. Thirdly, mainstream MLLMs [22, 70] often use dynamic image resolution strategies to enhance high-resolution image understanding, leading to multiple inferences of ViT and excessive image tokens. This hinders image processing speed and affects overall latency for end-side deployment.

To address these challenges, we propose BlueLM-V-3B, an algorithm and system co-design approach that enables more efficient deployment of MLLMs on mobile devices. Specifically, we train a state-of-the-art MLLM with only 3B parameters and effectively deploy it on the NPU of smartphones. In terms of algorithm design, we find that traditional dynamic resolution schemes [22, 70] lead to exaggerated image enlargement, resulting in longer image tokens and complicating mobile deployment. We propose a relaxed aspect ratio matching method, which effectively reduces the number of image tokens without sacrificing model accuracy. In terms of system design, different from previous papers on end-side MLLM [24, 134], we incorporate a detailed system optimization for hardware-aware deployment. To accelerate image encoding, we design batched image encoding together with pipeline parallelism processing for the image patches generated by the dynamic resolution processor. To address the inefficiency of NPU when processing long input tokens, we adopt a token downsampling method and implement a chunked computing approach. We also enhance deployment efficiency by using mixed-precision deployment and carefully designing the overall inference framework.

Compared to previous efforts for efficient MLLM deployment on mobile phones [24, 25, 134], BlueLM-V-3B achieves higher model performance and features a more detailed algorithm and system co-design, paving the way for more powerful and efficient MLLMs optimized for mobile environments. The features of BlueLM-V-3B and the con-

tributions of our work are summarized as follows:

###### 1) Algorithm and System Initiative: We identify and

address the excessive image enlargement issue in the dynamic resolution scheme used by classical MLLMs. Additionally, we implement a series of system designs and optimizations for hardware-aware deployment, resulting in more efficient inference of MLLMs on mobile devices.

###### 2) State-of-the-art MLLM Performance: BlueLM-V-

3B achieves SOTA performance (e.g., 66.1 on the OpenCompass benchmark) among models with similar parameter sizes, even surpassing a series of MLLMs with much more parameters (e.g., MiniCPM-V-2.6, InternVL2-8B).

###### 3) High Deployment Efficiency: BlueLM-V-3B is

highly efficient when deployed on mobile phones. Take the MediaTek Dimensity 9300 processor as an example, with a memory requirement of just 2.2GB, it can encode images with a resolution of 768×1536 in approximately 2.1 seconds and achieves a token throughput speed of 24.4 token/s.

###### 2. Related Works

###### 2.1. Multimodal Large Language Models

Large language models (LLMs) [6, 10, 118, 119] have demonstrated impressive success in tackling various complex tasks [120, 128] in recent years. Building upon these advancements, multimodal large language models (MLLMs) incorporate visual inputs into LLMs [23, 71, 91, 143] to handle multimodal scenarios. Various methods have been designed to integrate visual knowledge into language models, such as the linear projector approach [22, 71, 125], the Q-Former approach [57], and the perceiver resampler approach [5, 8, 134]. To further enhance MLLMs’ capability to comprehend high-resolution images, a dynamic resolution scheme has recently been proposed [22, 69, 70]. This scheme enables the model to adaptively process images at different resolutions while capturing more detailed information [41]. However, when deploying on mobile devices, the dynamic resolution approach presents two challenges: firstly, an excessive number of image patches can significantly slow down the processing speed of the image encoder, and secondly, long sequences of image tokens can result in increased latency in the language model [64].

###### 2.2. On-device Large Language Models

As application scenarios for large models continue to expand, small-scale large language models (SLMs) are now attracting increasing attention as consumers seek more costeffective and efficient solutions [7]. A series of SLMs have emerged to meet these demands, including language models [1, 39, 87] and multimodal language models [22, 56, 81, 125, 134]. The compact size of these models (2-3B parameters) enables deployment on user devices, such as personal computers and mobile phones. In addition to de-

will be introduced in Sec. 3.2 and Sec. 3.3 respectively.

BlueLM-3B

MLP Projector

BlueLM Tokenizer

Training and Inference: During training, the Image Encoder receives input images processed by the Dynamic Resolution Processor (for multiple input images, we simply concatenate them). The output features are passed through the Token Downsampler and MLP Projector to produce the corresponding image tokens. These image tokens are then concatenated with the language instruction tokens provided by the user. The resulting token sequences are used for model training. For inference, image and text tokens are similarly obtained (with user instructions in the audio format being converted to text first), and the model generates subsequent tokens in an autoregressive manner.

Token Downsampler Language Instruction

[Figure 3]

SigLIP-400M

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Dynamic Resolution Processor

Overall Architecture

- Figure 2. Model architecture of BlueLM-V-3B. The architecture of BlueLM-V-3B follows the classical LLaVA approach. We integrate a dynamic resolution processing module (as in LLaVANeXT [70] and InternVL 1.5 [22]) to enhance model capabilities and apply token downsampling to reduce deployment complexity.

###### 3.2. Dynamic Image Resolution

The original ViT of LLaVA [71] directly resizes the input images to a fixed resolution (e.g., 336×336 or 384×384), which is not well-suited for high-resolution images. To address this issue, BlueLM-V-3B adopts a dynamic image resolution design, which has been proven effective in InternVL 1.5 [22] and LLaVA-NeXT [70]. We observe exaggerated image enlargement issues in these two methods and make improvements for better training and easier deployment. Additionally, we design a batched image patch encoding with pipeline parallelism to further enhance the efficiency of both training and inference.

veloping smaller LLMs and MLLMs with higher performance, recent studies from a system perspective have also introduced various methods for deploying these SLMs on user devices, such as personal computers [129], and mobile phones [58, 134]. Our proposed BlueLM-V-3B is an algorithm and system co-design approach that not only achieves state-of-the-art model capability but also enables efficient deployment of MLLMs on mobile devices.

###### 3. BlueLM-V-3B

We provide a detailed introduction of BlueLM-V-3B in this section, emphasizing its model architecture and highlighting the algorithm and system co-design that optimizes efficiency during both the training and deployment stages.

Exaggerated Image Enlargement: LLaVA-NeXT [70] and InternVL 1.5 [22] both propose dynamic image resolution approaches to tackle high-resolution images. For the SigLIP encoder, both approaches use 384×384 as the base resolution (1:1), then select an appropriate resolution aspect ratio (m:n) to resize (and pad) the original image to a size of 384m×384n. The image is subsequently divided into patches of 384×384. LLaVA-NeXT tends to select aspect ratios that result in a larger image than the original one but with a smaller total area, while InternVL 1.5 opts for the ones that match the original image’s width-to-height ratio.

###### 3.1. Basic Network Components

Model Architecture: Our architecture is modified from the classical LLaVA approach [71], as it has been shown effective in prior works such as InternVL 1.5 [22] and LLaVA-NeXT [70]. The overall architecture is shown in Fig. 2. It is composed of the following components. Image Encoder: To process multimodal (image and language) inputs, we utilize the SigLIP [141] ViT for 384×384 input images, as described in [64, 134], which has 400M parameters. MLP Projector: A 2-layer MLP is employed to map the space of image tokens to LLM tokens. LLM: We employ an in-house 2.7B BlueLM model as the core language model to design BlueLM-V-3B. To further enhance the model’s ability to understand high-resolution images, a Dynamic Resolution Processor module is integrated. We reflect on the exaggerated image enlargement issue observed in InternVL 1.5 [22] and LLaVA-NeXT [70], then introduce a new approach that improves both training and deployment efficiency. Given the limited performance of NPUs in handling long tokens, we leverage a Token Downsampler module to reduce deployment complexity, which

We use some examples in Fig. 3 to demonstrate the exaggerated image enlargement by the two methods. For LLaVA-NeXT in Fig. 3A, given an image with 394×390 resolution, it will choose a ratio of 2:2, then resize and pad the original image to 768×768 (4× area enlargement). For InternVL 1.5 in Fig. 3B, given an image with 380×76 resolution (setting max num=6, i.e., m × n ≤ 6), it will choose a ratio of 5:1, then directly resize the original image to 1920×384 (25× enlargement)1. The enlargement does not necessarily alter image information, but increases de-

1The recently proposed adaptive gridding method in Ferret-UI 2 [62] shares a similar issue with InternVL 1.5, as strictly preserving the aspect ratio (e.g., choosing 5:1 instead of 1:1 for a 380 × 76 image) results in ∆aspect = 0, according to the pseudocode provided in the paper.

(A) LLaVA-NeXT Ours (B) InternVL 1.5 (max_num=6) Ours

394

394

380

380

| | | |
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| |
|---|
| |

76

76

390

390

- Figure 3. Existing methods overly enlarge images. (A) For LLaVA-NeXT, an image with resolution 394×390 selects a 2:2 aspect ratio and is resized and padded to 768×768 (4× area enlargement). (B) For InternVL 1.5, an image with resolution 380×76 chooses a 5:1 aspect ratio and is directly resized to 1920×384 (25× area enlargement). BlueLM-V-3B, in contrast, selects a 1:1 aspect ratio for both resolutions, resulting in the minimum number of image tokens after ViT encoding, which can facilitate both model training and deployment.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

ployment difficulty on mobile devices, as higher resolutions lead to more image patches and thus longer image tokens. Therefore, we redesign the dynamic resolution approach, as will be discussed in the following paragraph.

[ ]

ViT

| |
|---|
| |
| |

Batch

Concat

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[ ]

ViT

Batch

[Figure 22]

[ ]

ViT

Batch

Relaxed Aspect Ratio Matching: We propose a relaxed aspect ratio matching method based on LLaVA-NeXT to mitigate the exaggerated image enlargement. LLaVANeXT defines effective resolution Re and wasted resolution Rw. For aspect ratios (m:n) that lead to image resolution 384m×384n smaller than the original image in either dimension, Re is defined as the maximum area of the image scaled proportionally to fit within the dimensions 384m×384n. Otherwise, Re is set to the area of the original image. After getting Re, Rw is calculated by 384m×384n−Re. LLaVA-NeXT explores feasible aspect ratios and searches for the largest Re while minimizing Rw. Generally, larger aspect ratio options are available because m and n are always set to bigger values to accommodate high-resolution images (e.g., from 1:1 to 6:6 in LLaVAOneVision [56]). In this context, LLaVA-NeXT often leads to image enlargement by selecting an aspect ratio that offers a higher resolution than the original image, resulting in the cases shown in Fig. 3A. Considering that larger resolutions (4×) do not necessarily provide additional information but rather increase training and deployment complexity, downscaling the image is a more appropriate choice.

Figure 4. Batched image encoding on NPU. We design a parallel processing scheme for image patches on the NPU. The figure illustrates the case of 4 patches being processed in parallel.

and record the according aspect ratio. In our design, smaller Re with smaller Rw will have a chance to be chosen. To further increase the likelihood of selecting a smaller Rw, we enumerate the candidate aspect ratios in descending order, e.g., from 3:3 to 1:1 (assuming that a smaller aspect ratio leads to a smaller total area, thus a smaller Rw). Our relaxed aspect ratio matching method allows for more flexible handling of dynamic resolution. As shown in Fig. 3, when facing these two extreme cases, our solution can still select the appropriate aspect ratio (1:1). The pseudocode of the relaxed aspect ratio matching method is shown in Alg. 1.

Batched Image Patch Encoding: We also propose system optimization to achieve more efficient hardware-aware training and deployment. After the dynamic resolution processing, an image is divided into several local patches, together with a thumbnail image (global patch). In the training process, we batch the image patches before inputting them into the ViT, leveraging GPU parallelism to accelerate the process and achieving a 10% speedup. For inference, we adopt a similar parallel strategy to exploit the NPU’s computing capabilities. Unlike high-level languages (e.g., Python), hardware acceleration design requires lowlevel control over computing resources, such as memory layout and computational optimizations based on register size. Due to the NPU’s limited computational power, all patches cannot be effectively processed simultaneously; instead, we handle a fixed batch size at a time. Fig. 4 illustrates the concurrent processing of 4 patches for a 2:4 aspect ratio (the ratio we use to process mobile phone

We propose a relaxed aspect ratio matching method by leveraging a threshold to prevent the trend of always selecting larger resolutions. To be specific, we add a parameter α2 such that when:

Re − Re,max > α ⋅ Re,max, (1) or

(Re,max − Re)< α ⋅ Re,max and Rw < Rw,min, (2) we then update

Re,max ← Re, Rw,min ← Rw, (3)

2α = 0.1 in our implementation.

Conv2d x 4

Conv2d x 4 Conv2d ViT (4 Patches / Batch) ViT (1 Patch)

ViT (4 Patches / Batch)

40ms

160ms 800ms

300ms

- Figure 5. Pipeline parallelism in image encoding. We design a pipeline parallelism scheme for image encoding. The Conv2D layer in the vision embedding module of SigLIP (on the CPU) and the vision transformer blocks (on the NPU) for different image patches run parallel to improve inference speed. This image illustrates the pipeline parallelism scheme combined with batched image patch encoding.

screens) following a 4 + 4 + 1 approach. This concurrent patch processing notably reduces overall processing time.

computing techniques to process all input tokens simultaneously. However, the excessive length of these tokens (due to the extended length of image tokens or the contextual information involved), combined with the limited computational capacity of NPUs, renders the parallel processing of all input tokens inefficient. Conversely, sequential processing of individual tokens (t1) is also suboptimal. Consequently, we implement a chunking strategy on mobile devices, processing 128 input tokens in parallel (t128) per iteration, and then combining the results. This approach strikes a balance between parallel processing and the computational resources available on the NPU.

Pipeline Parallelism in Image Patch Encoding: During the model inference process, we implement a pipeline parallelism scheme to optimize image patch encoding. Specifically, for different patches extracted from a single image, we design a parallel pipeline for the Conv2D layer in SigLIP’s vision embedding module (on the CPU) and the vision transformer blocks (on the NPU). We show the encoding pipeline on MediaTek Dimensity 9300 processor with a 2:4 aspect ratio in Fig. 5. This approach helps to conceal the execution latency of the Conv2D operation.

###### 3.4. Model Quantization and Overall Framework

With the above design and optimization, we deploy the BlueLM-V-3B model on the MediaTek Dimensity 9300 processor. We hope to take full advantage of the device’s capabilities, offering a powerful yet efficient solution for running the model in a mobile environment.

###### 3.3. Token Downsampler

Although we have designed a relaxed aspect ratio matching method to mitigate the exaggerated image enlargement, the dynamic image resolution strategy still results in a significant number of image tokens, posing challenges for the deployment on mobile phone processors and potentially exceeding the maximum context length of the language model. For instance, with an image that selects a 2:4 aspect ratio (the resolution we use for processing mobile phone screens), we obtain a total of 9 image patches (calculated as 2×4+1). This results in 729×9 = 6561 image tokens from SigLIP after the dynamic resolution processor, making it too long to deploy on the NPU.

Mixed Parameter Precision: We apply mixed-precision quantization to further reduce memory usage and improve inference speed. We employ INT8 precision for the ViT and the MLP projector weights, and INT4 precision for the LLM weights. This combination strikes a balance between computational efficiency and model accuracy. However, we find that the activation values are more sensitive to quantization; therefore, we maintain INT16 precision for LLM activation and FP16 for ViT and projector activation to ensure the model’s performance remains robust. During inference, we store the KV cache in INT8 precision.

Basic Practice: To reduce the excessive number of image tokens, we apply the downsampler block proposed in VILA [64]. Specifically, it concatenates every 2 × 2 tokens into a single token and then employs a linear layer to fuse the information. This effectively reduces the number of image tokens generated by SigLIP from 729 to 196, resulting in a total of 196 × 9 = 1764 image tokens for the 2:4 aspect ratio setting. However, the approximately 2k length of the image tokens, when combined with the user instruction, still poses challenges for deployment on the NPU, as will be discussed in the following paragraph.

Decoupling Image Encoding and Instruction Processing: To improve overall efficiency during deployment, we decouple image processing from user input handling. At model initialization, we load both the ViT and LLM models simultaneously. Users begin by uploading an image, and since the MLLM is deployed locally, this upload takes virtually no time. The ViT starts processing the image immediately after the upload is complete. Meanwhile, users can input their instructions simultaneously; for instructions in audio format, we convert them to text first. Once the image processing is finished, the user’s commands are submitted to the LLM for response generation, and the ViT can

Chunked Computing of Input Tokens: During the inference process of a LLM, to accelerate the computation of input tokens, traditional GPUs frequently employ parallel

###### Type Public (M) In-House (M) In-House / Public

Image Uploading

Pure Text 2.2 64.7 29.4 Caption 10.0 306.3 30.6 VQA 20.3 44.4 2.2 OCR 23.3 173.9 7.5

ViT

LLM Processing

| |
|---|

| |
|---|

Text to Audio

Text Response

User Instruction

###### t

Total 55.8 589.3 10.6

- Figure 6. Overall framework of deploying BlueLM-V-3B. We decouple ViT image processing from user instruction (text or audio) handling to enhance overall efficiency. The text responses by LLM can be further converted on the fly to audio responses.

Table 1. Detailed statistics of the fine-tuning dataset. Summary of dataset types, counts (in millions), and in-house/public ratios for each category used in fine-tuning.

be freed from memory. This parallel process, illustrated in Fig. 6, reduces the waiting time for the first token generation, improves overall responsiveness, and limits the peak memory usage of BlueLM-V-3B to 2.2GB.

to convert them into image-text pairs. For formula data, we use Matplotlib4 to render them into necessary representations. For table contents, we convert the data to Markdown format and render it with the IMGKit5 library. For problem-solving data, we similarly convert the text into Markdown format and render it using the IMGKit library. Additionally, we manually render a substantial amount of multilingual OCR data by converting texts in various languages into image-text pairs, which helps to enhance the model’s multilingual understanding and capabilities.

###### 4. Training Recipe

In this section, we detail the training process and training data for BlueLM-V-3B.

###### 4.1. Training Process

In addition to image rendering, we also utilize GPT4o [91] and Gemini Pro [115] to create and revise image captions and question-answering pairs. The combination of open-source and proprietary data significantly enhances the model’s capabilities, allowing it to learn from a diverse range of examples and improve its performance across various tasks and modalities.

We begin with the BlueLM-3B language model and train the model in two stages. In the first stage, We pre-train the MLP projection layer while keeping the ViT and LLM frozen. In the second stage, we fully fine-tune the model with a large set of image-text pairs.

###### 4.2. Training Data

Pre-training Stage: The pre-training stage aims to equip the model with basic multimodal capabilities. In this stage, we utilize open-source datasets, creating a comprehensive pre-training dataset composed of 2.5 million imagecaption pairs drawn from LLaVA 558k [69], ShareGPT4V 1200k [17], and ALLaVA 708k [14].

###### 5. Experiments

In this section, we conduct a series of experiments to validate the effectiveness of our proposed approaches and to demonstrate the capabilities of BlueLM-V-3B in terms of benchmark accuracy and deployment efficiency.

Fine-tuning Stage: During the fine-tuning process, we meticulously construct a dataset containing 645 million image-text pairs, including both open-source and in-house datasets. This dataset covers a variety of downstream tasks and diverse dataset types, such as captioning, VQA, OCR, and pure text. Tab. 1 summarizes the distribution of data types, along with the proportions of public and in-house data in our fine-tuning dataset.

To better illustrate the data we use, we present the opensource datasets utilized at this stage in Tab. 6. In addition to these open-source datasets, we also incorporate in-house data to enhance the model’s capabilities. We crawl a significant amount of pure text data and image-text pairs from various websites. For different data categories, we also manually create a large number of image-text pairs to enrich the diversity of the training data.

For PDF documents, we utilize the PyMuPDF3 library

3https://github.com/pymupdf/PyMuPDF

###### 5.1. Relaxed Aspect Ratio Matching

In BlueLM-V-3B, we alleviate the issue of ineffective image upscaling present in InternVL 1.5 and LLaVA-NeXT by applying a relaxed aspect ratio matching approach. In this section, we validate the enhancements by measuring the improvements in both deployment efficiency and benchmark accuracy, with both open-source and in-house models.

Deployment Efficiency: We conduct a statistical analysis on the multimodal LLaVA [69] 665k training dataset. We omit the 41k text-only data and compare the aspect ratio selected by our approach with those chosen by the LLaVANeXT [70] and InternVL 1.5 [22] approaches. A larger aspect ratio selection results in higher image resolution, which leads to a longer image token sequence during deployment. We provide 9 candidate aspect ratios (from 1:1 to 3:3) for

- 4https://github.com/matplotlib/matplotlib
- 5https://github.com/csquared/IMGKit

Language Model Vision Model Params Method VQAv2val TextVQAval DocVQAval OCRBench ChartQAtest

InternVL 1.5 70.5 46.9 26.2 327 15.7 LLaVA-NeXT 70.1 44.2 24.3 324 14.8 Ours 71.8 49.4 27.3 343 16.9

MiniCPM-2B [39] SigLIP-400M [141] 3B

InternVL 1.5 78.3 52.7 28.7 338 16.8 LLaVA-NeXT 77.7 51.4 29.6 351 16.4 Ours 79.5 56.2 31.3 360 17.5 Ours (fully-trained) 82.7 78.4 86.6 829 80.4

BlueLM-3B SigLIP-400M [141] 3B

- Table 2. Comparison results of different dynamic resolution methods. We compare the performance of models trained using different dynamic resolution methods. We use the LLaVA [69] 558k dataset for pre-training, and the LLaVA 665k dataset for fine-tuning. To better demonstrate our improvements, we conduct experiments on both our in-house BlueLM-3B language model and the open-sourced MiniCPM-2B language model, which have similar parameter counts (2.7B). Our dynamic image processing method achieves the best performance. †We also provide the results of the fully trained BlueLM-V-3B model for reference.

Model Params Avg. MMBench MMStar MMMU MathVista HallusionBench AI2D OCRBench MMVet Qwen2-VL [125] 8B 67 81 60.7 53.7 61.4 50.4 83 843 61.8 MiniCPM-V-2.6 [134] 8B 65.2 78 57.5 49.8 60.6 48.1 82.1 852 60 InternVL2 [22] 8B 64.1 79.4 61.5 51.2 58.3 45 83.6 794 54.3 POINTS-Qwen2.5 [74] 8.3B 62.5 78 60.9 51.4 63 45.6 81.2 717 47.9 BlueLM-V (Ours) 3B 66.1 82.7 62.3 45.1 60.8 48 85.3 829 61.8

- Table 3. OpenCompass benchmark. Comparison results on the OpenCompass benchmark for models with parameter sizes less than or equal to 10B. BlueLM-V-3B achieves state-of-the-art performance on 4 out of 8 tasks, with an average performance ranking of second.

Model Params TextVQAval DocVQAtest MTVQA Phi-3-Vision [2] 4.2B 72.4 84.6 13.9 MiniCPM-V-2 [134] 2.8B 73.2 71.9 9.3 InternVL2 [22] 4B 74.7 89.2 15.5 Qwen2-VL [125] 2B 79.9 90.1 20.7 BlueLM-V (Ours) 3B 78.4 87.8 32.7

- Table 4. Text-centric/OCR benchmarks. Comparison on textcentric/OCR benchmarks shows that BlueLM-V-3B achieves performance comparable to SOTA MLLMs with similar parameter sizes, while significantly enhancing multilingual capability. †We evaluate TextVQA and MTVQA on VLMEvalKit [29] for a fair comparison. OCRBench has been included in OpenCompass.

665k data for fine-tuning. Due to the slower learning capacity of the 3B model compared to the 7B/13B models, we train each stage for two epochs.

We conduct experiments on both our in-house BlueLM3B model and the open-sourced MiniCPM-2B language model [39]. The MiniCPM-2B model has 2.7B parameters, matching the parameter count of BlueLM-3B. The introduction of dynamic resolution is primarily aimed at handling high-resolution images, especially in OCR tasks. For evaluation, we use VQAv2 [31] for general vision question answering. For OCR tasks, we include TextVQA [107], DocVQA [84], and OCRBench [73]. Additionally, we evaluate on ChartQA [83] dataset for multimodal mathematical reasoning. Comparison results are shown in Tab. 2. As can be seen, our proposed relaxed aspect ratio matching approach not only reduces the number of image tokens but also significantly improves accuracy. This indicates that simply enlarging images does not always lead to performance gains, as in LLaVA-NeXT and InternVL 1.5. The results demonstrate the effectiveness of our approach in both training outcomes and deployment efficiency.

LLaVA-NeXT and our approach. For InternVL 1.5, we fix max num=9 and propose the initial candidate aspect ratios using the method described in the original paper. This will result in the same maximum area (9×384×384) for all approaches, and ensure a fair comparison.

Compared with LLaVA-NeXT, in 29k cases, our method selects a smaller aspect ratio. Concerning InternVL 1.5, we select smaller aspect ratios in 523k cases and larger aspect ratios in 25k cases. This leads to a significant improvement in efficiency during the inference on NPU.

###### 5.2. Accuracies on Different Benchmarks

After extensive fine-tuning, we evaluate the performance of BlueLM-V-3B on various mainstream benchmarks.

Benchmark Accuracy: To further evaluate the performance impact of reducing image tokens, we train on the LLaVA 1.5 [69] dataset with 3 dynamic image resolution approaches. We use the 558k data for pre-training and the

OpenCompass Benchmark: To evaluate the overall performance of BlueLM-V-3B, we use the OpenCompass benchmark [26], including MMbench [75], MMStar [18],

Model Name Params Processor Solution Image Processing LLM Prefilling Throughput MiniCPM-V 2.5 [134] 8B MediaTek Dimensity 9300 CPU (llama.cpp) 4.0s 13.9s 4.9 token/s BlueLM-V-3B (Ours) 3B MediaTek Dimensity 9300 NPU 2.53s (0.47+2.06) 2.7s 24.4 token/s

- Table 5. Deployment efficiency comparison with MiniCPM-V. MiniCPM-V deploys an 8B model on the CPU, leading to longer image processing latency, LLM prefilling latency, and lower throughput. †MiniCPM-V calculates encoding latency by including both model loading time and encoding time. In our setting, we need 0.47s to simultaneously load the ViT and LLM once during system initialization.

| |2 patches 4 patches<br><br>6 patches<br><br>1 patch| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2500

2000

Time(ms)

1500

1000

500

0

1 2 4 6 Patches Per Batch

- Figure 7. ViT inference time for 2:4 resolution aspect ratio. We experiment with 1, 2, 4, and 6 image patches per batch on the NPU, using a 2:4 resolution aspect ratio (comprising one global patch and 8 local patches). Overall, processing 4 patches per batch delivers the fastest performance.

MMMU [140], MathVista [80], HallusionBench [33], AI2D [49], OCRBench [72], and MM-Vet [138]. BlueLMV-3B demonstrates significantly improved performance over models with a similar number of parameters (≤ 4B), as illustrated in Fig. 1. We also compare BlueLM-V-3B to models with up to 10B parameters, with detailed results presented in Tab. 3. BlueLM-V-3B achieves SOTA performance on 4 out of 8 tasks, with an average performance ranking of second. With only 3B parameters, BlueLM-V3B outperforms a series of MLLMs with significantly more parameters, such as InternVL2-8B [22], MiniCPM-V-2.6 (8B) [134]. This demonstrates that MLLMs with a smaller number of parameters can still achieve strong capacity.

Text-centric/OCR Capacity: We compare BlueLM-V-3B with popular MLLMs of similar parameter sizes on textcentric/OCR benchmarks, including TextVQA [107] and DocVQA [84] (OCRBench [72] has been included in OpenCompass). For MLLMs deployed on mobile devices for everyday use, especially in overseas markets or among foreign users, multilingual capabilities are essential. We also evaluate the multilingual multimodal capabilities using the MTVQA [114] dataset. The comparison results are shown in Tab. 4. BlueLM-V-3B achieves performance comparable to state-of-the-art MLLMs with similar parameter sizes, while significantly enhancing multilingual capability.

Latency and Output Speed Comparison

5.0

25.0

| |Latency (s) Output Speed (token/s)<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

4.5

24.5

OutputSpeed(token/s)

4.0

24.0

Latency(s)

3.5

23.5

3.0

23.0

2.5

22.5

t32/t1 t128/t1 t512/t1 t2048/t1 Tokens Per Trunk

Figure 8. Latency and output speed comparison. We compare the latency and output generation speed with processing different numbers of input tokens in parallel. t{x}/t1 implies processing x input tokens in parallel. The output token is fixed to 1 per trunk as the LLM can only generate one token for each forward process.

###### 5.3. Deployment Efficiency Evaluation

In this section, we discuss the deployment efficiency of our proposed algorithm and system co-design in BlueLM-V3B. We deploy BlueLM-V-3B on vivo X100 mobile phone with MediaTek Dimensity 9300 processor as an example.

Batched Image Patch Encoding on NPU: We design a parallel processing method for encoding image patches generated by the dynamic resolution processor. Fig. 7 shows the ViT inference latency when processing an image with a 2:4 aspect ratio (9 patches in total). As illustrated, parallelizing the inference with an increasing number of patches initially reduces latency, but after a certain point, latency begins to increase again. The fastest encoding speed is achieved with 4 patches processed in parallel.

Pipeline Parallelism in Image Patch Encoding: We design a pipeline parallelism scheme for the Conv2D layer and the vision transformer block in the SigLIP model. The processing pipeline for an image with a 2:4 aspect ratio is shown in Fig. 5, where we need 1.9 + 0.16 = 2.06s to encode the image and can hide a total latency of 200ms.

Chunked Computing of Input Tokens: We implement a chunking strategy on the NPU to process 128 input tokens in parallel per iteration (t128), balancing parallel

processing with NPU performance. Here, we present the LLM prefilling latency when processing different numbers of input tokens in parallel: t32, t128, t512, and t2048. For additional information, we also report the speed of output token generation, where only the t1 case is shown, as the LLM processes one token at a time for output. We fix the input token length to 2048 and set the KV cache length to 2048. As shown in Fig. 8, t128/t1 achieves the lowest latency and the fastest generation speed. Theoretically, the output token speed for t1 should remain consistent; however, we observe slight variations due to possible fluctuations in the testing environment.

Compasison with MiniCPM-V: The MiniCPM-V paper [134] only reports the deployment statistics for the 8B MiniCPM-V 2.5 model, which is deployed on the CPU of the MediaTek Dimensity 9300 using llama.cpp6. However, the paper does not specify the exact testing configuration, such as image resolution, context length, cache size, etc. Here, we provide a direct comparison between BlueLM-V-3B and the statistics reported in the MiniCPMV paper [134], as shown in Tab. 5. We utilize an image with a resolution of 768×1536, fix the whole input token length to 2048, and set the KV cache length to 2048. BlueLM-V3B achieves shorter latency and faster token throughput.

###### 6. Conclusion and Future Discussions

This paper introduces BlueLM-V-3B, an algorithm and system co-design approach tailored for MLLMs on mobile platforms. Our goal is to maximize usability and performance while ensuring a positive user experience. We emphasize algorithm-system co-design and hardware-aware optimization for training MLLMs and deploying them on mobile phones. Extensive experiments and statistical analyses demonstrate the strong capability and high efficiency of BlueLM-V-3B on mobile devices. Future work will focus on optimizing the scalability of BlueLM-V-3B for a broader range of mobile devices and exploring advanced algorithms to enhance performance and usability. We hope our work will contribute to the advancement of this field of study.

###### References

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219,

2024. 2

- [2] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model

6https://github.com/ggerganov/llama.cpp

locally on your phone. arXiv preprint arXiv:2404.14219,

2024. 7

- [3] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, pages 8076–8084, 2019. 17
- [4] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [5] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: A visual language model for few-shot learning. NeurIPS, 35:23716–23736, 2022. 2
- [6] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 2
- [7] Saleh Ashkboos, Iman Mirzadeh, Keivan Alizadeh, Mohammad Hossein Sekhavat, Moin Nabi, Mehrdad Farajtabar, and Fartash Faghri. Computational bottlenecks of training small-scale large language models. arXiv preprint arXiv:2410.19456, 2024. 2
- [8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966,

2023. 1, 2

- [9] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Mar¸cal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4291–4301, 2019. 17
- [10] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. 1, 2
- [11] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. COYO-700M: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 16, 17
- [12] Sungguk Cha, Jusung Lee, Younghyun Lee, and Cheoljong Yang. Visually dehallucinative instruction generation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5510–5514. IEEE, 2024. 17
- [13] Shuaichen Chang, David Palzer, Jialin Li, Eric FoslerLussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps. arXiv preprint arXiv:2211.08545, 2022. 17
- [14] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. ALLaVA: Harnessing GPT4V-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 6, 17
- [15] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying

- geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746, 2022. 17
- [16] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal LLM’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 17
- [17] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. ShareGPT4V: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 6
- [18] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 7
- [19] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. Tabfact: A large-scale dataset for table-based fact verification. 2019. 17
- [20] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 17
- [21] Xingyu Chen, Zihan Zhao, Lu Chen, Danyang Zhang, Jiabao Ji, Ao Luo, Yuxuan Xiong, and Kai Yu. Websrc: A dataset for web-based structural reading comprehension. arXiv preprint arXiv:2101.09465, 2021. 17
- [22] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites. arXiv preprint arXiv:2404.16821, 2024. 1, 2, 3, 6, 7, 8, 16
- [23] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 1, 2
- [24] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023. 2
- [25] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024. 2
- [26] OpenCompass Contributors. OpenCompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass,

2023. 7

- [27] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jos´e MF Moura, Devi Parikh, and Dhruv Batra. Visual dialog. In Proceedings of the IEEE con-

- ference on computer vision and pattern recognition, pages 326–335, 2017. 17
- [28] Yucheng Ding, Chaoyue Niu, Fan Wu, Shaojie Tang, Chengfei Lyu, and Guihai Chen. Enhancing on-device llm inference with historical cloud-based llm interactions. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 597–608,

2024. 1, 2

- [29] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models, 2024. 7
- [30] Haoyuan Gao, Junhua Mao, Jie Zhou, Zhiheng Huang, Lei Wang, and Wei Xu. Are you talking to a machine? dataset and methods for multilingual image question. Advances in neural information processing systems, 28, 2015. 17
- [31] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913,

2017. 7, 17

- [32] Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale Chinese cross-modal pre-training benchmark. NeurIPS, 35: 26418–26431, 2022. 17
- [33] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. arXiv preprint arXiv:2310.14566, 2023. 8
- [34] Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 18135–18143, 2024. 17
- [35] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. VizWiz Grand Challenge: Answering visual questions from blind people. In CVPR, pages 3608–3617, 2018. 17
- [36] Danna Gurari, Yinan Zhao, Meng Zhang, and Nilavra Bhattacharya. Captioning images taken by people who are blind. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16, pages 417–434. Springer, 2020. 17
- [37] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286,

2020. 17

- [38] Yu-Chung Hsiao, Fedir Zubach, Gilles Baechler, Victor Carbune, Jason Lin, Maria Wang, Srinivas Sunkara, Yun Zhu, and Jindong Chen. Screenqa: Large-scale questionanswer pairs over mobile app screenshots. arXiv preprint arXiv:2209.08199, 2022. 17
- [39] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang,

- Weilin Zhao, et al. MiniCPM: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024. 2, 7
- [40] Wenyue Hua, Mengting Wan, Shashank Vadrevu, Ryan Nadel, Yongfeng Zhang, and Chi Wang. Interactive speculative planning: Enhance agent efficiency through codesign of system and user interface, 2024. 1, 2
- [41] Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, and Xiang Bai. Mini-monkey: Multi-scale adaptive cropping for multimodal large language models. arXiv preprint arXiv:2408.02034, 2024. 2
- [42] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019. 17
- [43] Drew A Hudson and Christopher D Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, pages 6700–6709, 2019. 17
- [44] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. In 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW), pages 1–6. IEEE, 2019. 17
- [45] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR, pages 2901–2910, 2017. 17
- [46] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. DVQA: Understanding data visualizations via question answering. In CVPR, pages 5648–5656, 2018. 17
- [47] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos´ K´ad´ar, Adam Trischler, and Yoshua Bengio. FigureQA: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017. 17
- [48] Shankar Kantharaj, Rixie Tiffany Ko Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq Joty. Chart-to-text: A large-scale benchmark for chart summarization. arXiv preprint arXiv:2203.06486, 2022. 17
- [49] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 8

- [50] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocrfree document understanding transformer. In European Conference on Computer Vision (ECCV), 2022. 17
- [51] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 17

- [52] Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1):1–10, 2018. 17
- [53] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions., 2024. 17
- [54] Hugo Laurenc¸on, L´eo Tronchon, and Victor Sanh. Unlocking the conversion of web screenshots into html code with the websight dataset, 2024. 17
- [55] Paul Lerner, Olivier Ferret, Camille Guinaudeau, Herv´e Le Borgne, Romaric Besanc¸on, Jos´e G Moreno, and Jes´us Lov´on Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3108–3120, 2022. 17
- [56] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 4, 17
- [57] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 2

- [58] Luchang Li, Sheng Qian, Jie Lu, Lunxi Yuan, Rui Wang, and Qin Xie. Transformer-lite: High-efficiency deployment of large language models on mobile phone gpus. arXiv preprint arXiv:2403.20041, 2024. 3
- [59] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231,

2024. 17

- [60] Shengzhi Li and Nima Tajbakhsh. Scigraphqa: A largescale synthetic multi-turn question-answering dataset for scientific graphs. arXiv preprint arXiv:2308.03349, 2023. 17
- [61] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14963–14973, 2023. 17
- [62] Zhangheng Li, Keen You, Haotian Zhang, Di Feng, Harsh Agrawal, Xiujun Li, Mohana Prasad Sathya Moorthy, Jeff Nichols, Yinfei Yang, and Zhe Gan. Ferret-ui 2: Mastering universal user interface understanding across platforms. arXiv preprint arXiv:2410.18967, 2024. 3
- [63] Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and ”Teknium”. OpenOrca: An open dataset of GPT augmented FLAN reasoning traces. https://https://huggingface.co/ Open-Orca/OpenOrca, 2023. 17
- [64] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi,

- and Song Han. Vila: On pre-training for visual language models, 2023. 2, 3, 5
- [65] Yiqi Lin, Conghui He, Alex Jinpeng Wang, Bin Wang, Weijia Li, and Mike Zheng Shou. Parrot captions teach clip to spot text. In European Conference on Computer Vision, pages 368–385. Springer, 2025. 17
- [66] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651, 2023. 17
- [67] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 17
- [68] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. arXiv preprint arXiv:2311.10774, 2023. 17
- [69] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 2, 6, 7, 17
- [70] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge, 2024. 2, 3, 6, 16
- [71] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2024. 2, 3, 17
- [72] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of OCR in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 8
- [73] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 7
- [74] Yuan Liu, Zhongyin Zhao, Ziyuan Zhuang, Le Tian, Xiao Zhou, and Jie Zhou. Points: Improving your visionlanguage model with affordable strategies. arXiv preprint arXiv:2409.04828, 2024. 7
- [75] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multimodal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025. 7
- [76] Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Towards end-to-end unified scene text detection and layout analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1049–1059, 2022. 17
- [77] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165,

2021. 17

- [78] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021. 17
- [79] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35: 2507–2521, 2022. 17
- [80] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 8
- [81] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. arXiv preprint arXiv:2410.08202, 2024. 2
- [82] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 17
- [83] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 7, 17
- [84] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. DocVQA: A dataset for VQA on document images. In WACV, pages 2200–2209, 2021. 7, 8, 17
- [85] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. InfographicVQA. In WACV, pages 1697–1706, 2022. 17
- [86] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. MM1: Methods, analysis & insights from multimodal LLM pretraining. arXiv preprint arXiv:2403.09611, 2024. 1
- [87] Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Seyed Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, et al. Openelm: An efficient language model family with open training and inference framework. In Workshop on Efficient Systems for Foundation Models II@ ICML2024, 2024. 2
- [88] Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1527–1536, 2020. 17
- [89] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: Visual question answering by reading text in images. In ICDAR, pages 947–952,

2019. 17

- [90] Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the poten-

- tial of slms in grade school math. arXiv preprint arXiv:2402.14830, 2024. 17
- [91] OpenAI. Gpt-4 technical report. 2023. 2, 6
- [92] Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. Cord: a consolidated receipt dataset for post-ocr parsing. In Workshop on Document Intelligence at NeurIPS 2019, 2019. 17
- [93] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. arXiv preprint arXiv:1508.00305, 2015. 17
- [94] Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024. 17
- [95] B Pfitzmann, C Auer, M Dolfi, AS Nassar, and PWJ Staar. Doclaynet: A large humanannotated dataset for documentlayout analysis (2022). URL: https://arxiv. org/abs/2206,

1062. 17

- [96] Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pages 647–664. Springer, 2020. 17
- [97] Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and Kaibin Huang. Mobile edge intelligence for large language models: A contemporary survey. arXiv preprint arXiv:2407.18921, 2024. 1, 2
- [98] Juan A Rodriguez, David Vazquez, Issam Laradji, Marco Pedersoli, and Pau Rodriguez. Ocr-vqgan: Taming textwithin-image generation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3689–3698, 2023. 17
- [99] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. NeurIPS, 35: 25278–25294, 2022. 17
- [100] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015. 17
- [101] Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. KVQA: Knowledge-aware visual question answering. In AAAI, pages 8876–8884, 2019. 17
- [102] Aditya Sharma, Aman Dalmia, Mehran Kazemi, Amal Zouaq, and Christopher J Pal. Geocoder: Solving geometry problems by generating modular code through visionlanguage models. arXiv preprint arXiv:2410.13510, 2024. 17
- [103] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cˆot´e, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2021. 17

- [104] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer, 2020. 17
- [105] Stˇˇ ep´an Simsa,ˇ Milan Sulc,ˇ Michal Uˇriˇc´aˇr, Yash Patel, Ahmed Hamdi, Matˇej Koci´an, Maty´aˇs Skalick`y, Jiˇr´ı Matas, Antoine Doucet, Micka¨el Coustaty, et al. Docile benchmark for document information localization and extraction. In International Conference on Document Analysis and Recognition, pages 147–166. Springer, 2023. 17
- [106] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In CVPR, pages 8317–8326, 2019. 17
- [107] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 7, 8
- [108] Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner. Textocr: Towards largescale end-to-end reasoning for arbitrary-shaped scene text. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8802–8812, 2021. 17
- [109] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. Powerinfer: Fast large language model serving with a consumer-grade gpu. arXiv preprint arXiv:2312.12456,

2023. 1

- [110] Tomasz Stanisławek, Filip Grali´nski, Anna Wr´oblewska, Dawid Lipi´nski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemysław Biecek. Kleister: Key information extraction datasets involving long documents with complex layouts. In ICDAR, pages 564–579. Springer,

2021. 17

- [111] Hongbin Sun, Zhanghui Kuang, Xiaoyu Yue, Chenhao Lin, and Wayne Zhang. Spatial dual-modality graph reasoning for key information extraction. arXiv preprint arXiv:2103.14470, 2021. 17
- [112] S Svetlichnaya. Deepform: Understand structured documents at scale. 2020. 17
- [113] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. VisualMRC: Machine reading comprehension on document images. In AAAI, pages 13878–13888, 2021. 17
- [114] Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, Yanjie Wang, Yuliang Liu, Hao Liu, Xiang Bai, and Can Huang. Mtvqa: Benchmarking multilingual text-centric visual question answering, 2024. 8
- [115] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 6
- [116] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang,

- Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 1, 17
- [117] Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct1: A 1.8 million math instruction tuning dataset. arXiv preprint arXiv:2402.10176, 2024. 17
- [118] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 2
- [119] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 2
- [120] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024. 1, 2
- [121] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. arXiv preprint arXiv:1601.07140, 2016. 17
- [122] Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman, and Yang Li. Screen2words: Automatic mobile ui summarization with multimodal learning. In The 34th Annual ACM Symposium on User Interface Software and Technology, pages 498–510, 2021. 17
- [123] Guanqun Wang, Jiaming Liu, Chenxuan Li, Yuan Zhang, Junpeng Ma, Xinyu Wei, Kevin Zhang, Maurice Chong, Renrui Zhang, Yijiang Liu, et al. Cloud-device collaborative learning for multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12646–12655, 2024. 1, 2
- [124] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv preprint arXiv:2311.07574, 2023. 17
- [125] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 2, 7
- [126] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. CogVLM: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 16, 17
- [127] Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the general value of evidence, and bilingual scene-text visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10126–10135, 2020. 17
- [128] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten

- Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022. 1, 2
- [129] Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, and Mao Yang. T-mac: Cpu renaissance via table lookup for low-bit llm deployment on edge. arXiv preprint arXiv:2407.00088, 2024. 3
- [130] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023. 17
- [131] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, and Furu Wei. Xfund: a benchmark dataset for multilingual visually rich form understanding. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3214–3224, 2022. 17
- [132] Zhenliang Xue, Yixin Song, Zeyu Mi, Le Chen, Yubin Xia, and Haibo Chen. Powerinfer-2: Fast large language model inference on a smartphone. arXiv preprint arXiv:2406.06282, 2024. 1, 2
- [133] Zhibo Yang, Rujiao Long, Pengfei Wang, Sibo Song, Humen Zhong, Wenqing Cheng, Xiang Bai, and Cong Yao. Modeling entities as semantic points for visual information extraction in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15358–15367, 2023. 17
- [134] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 1, 2, 3, 7, 8, 9
- [135] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023. 17
- [136] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023. 17
- [137] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 17
- [138] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 8
- [139] Wenwen Yu, Chengquan Zhang, Haoyu Cao, Wei Hua, Bohan Li, Huang Chen, Mingyu Liu, Mingrui Chen, Jianfeng Kuang, Mengjun Cheng, et al. Icdar 2023 competition on structured text extraction from visually-rich document images. In International Conference on Document Analysis and Recognition, pages 536–552. Springer, 2023. 17

- [140] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024. 8
- [141] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of ICCV, pages 11975–11986, 2023. 3, 7
- [142] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, Sam Dodge, Keen You, Zhen Yang, Aleksei Timofeev, Mingze Xu, Hong-You Chen, Jean-Philippe Fauconnier, Zhengfeng Lai, Haoxuan You, Zirui Wang, Afshin Dehghan, Peter Grasch, and Yinfei Yang. Mm1.5: Methods, analysis & insights from multimodal llm fine-tuning, 2024. 1
- [143] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199,

2023. 2

- [144] Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415, 2023. 17
- [145] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. LLaVAR: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 17
- [146] Bo Zhao, Boya Wu, and Tiejun Huang. SVIT: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087,

2023. 17

- [147] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. Towards complex document understanding by discrete reasoning. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4857–4866, 2022. 17

###### 7. Relaxed Aspect Ratio Matching

Algorithm 1 Relaxed Aspect Ratio Matching

- 1: function RELAXED ASPECT RATIO MATCHING(original size: (Worig,Horig), possible ratios: List of (m,n))

- 2: Initialize: best fit ← None, Re,max ← 0, Rw,min ←∞

- 3: (Worig,Horig)← original size

- 4: for each (m,n) in possible ratios do

- 5: (W,H)←(384 × m,384 × n)
- 6: scale ← min(WW

orig

, HH

orig

)

- 7: δW int(Worig ⋅ scale)
- 8: δH int(Horig scale)
- 9: Re min(δW ⋅ δH,Worig ⋅ Horig)
- 10: Rw ← W ⋅ H − Re
- 11: if (Re − Re,max)> α ⋅ Re,max or ((Re,max − Re)< α ⋅ Re,max and Rw < Rw,min) then
- 12: Re,max Re
- 13: Rw,min Rw
- 14: best fit ←(m,n)

- 15: end if
- 16: end for
- 17: return best fit

- 18: end function

To further expand the content of the main text, here we provide more about the relaxed aspect ratio matching method in this section.

|[Figure 35]|
|---|
| |

- (A)
- (B)

Pseudocode: We present the pseudocode for our proposed relaxed aspect ratio matching method, as shown in Alg. 1. To be specific, we change the updating logic of LLaVANeXT by adding a parameter α such that when:

Re − Re,max > α ⋅ Re,max, (4) or

(Re,max − Re)< α ⋅ Re,max and Rw < Rw,min, (5) we then update

| |[Figure 36]| | | | |
|---|---|---|---|---|---|
| | | | | | |

Re,max ← Re, Rw,min ← Rw, (6)

Figure 9. Case study. (A) LLaVA-NeXT chooses resolution 384×768 for an image with the original size of 380×393. (B) InternVL 1.5 chooses resolution 1920×384 for an image with the original size of 500×102.

and record the according aspect ratio. This increases the likelihood of selecting aspect ratios with smaller Re but also smaller Rw.

Case Study: Here we present real cases where LLaVANeXT [70] and InternVL 1.5 [22] result in significant image enlargement, as illustrated in Fig. 9. In Fig. 9A (from COYO-300M Dataset [11]), LLaVA-NeXT selects a resolution of 384×768 for an image originally sized 380×393. Similarly, Fig. 9B (from CogVLM-SFT dataset [126]) shows InternVL 1.5 selecting a resolution of 1920×384 for an image initially sized 500×102. In contrast, our proposed relaxed aspect ratio matching selects 384×384 for the 380×393 image and 768×384 for the 500×102 image.

###### 8. Open-source Training Dataset

Here we provide the open-source dataset used to train BlueLM-V-3B in the fine-tuning stage in Tab. 6.

Task Dataset Textonly

ALLaVA [14], ScienceQA [79], Orca-Math [90], OpenOrca [63], MetaMathQA [137], WizardLM [130], MathInstruct [117]

Caption TextCaps [104], Screen2Words [122], VizWiz [36], Laion [99], COCO [20], LLaVA [71], ALLaVA [14], SVIT [146], SA1B [51], VSR [66], Chart2Text [48], MultiMath [94], ArXivCap [59], COYO [11]

OCR Wukong [32], HierText [76], TextOCR [108], WildReceipt [111], DocILE [105], SVRD [139], DocLayNet [95], XFUND [131], COCO-Text [121], SROIE [42], FUNSD [44], CORD [92], Paper2Fig100k [98], Docmatix [53], LAION-2B-OCR [65], SynthDoG [50], WebSight [54], DeepForm [112], Kleister [110], TabFact [19]

VQA LVIS-Instruct4V [124], CLEVR [45], TallyQA [3], LNQA [96], Geo170K [102], ALLaVA [14], DocVQA [84], ChartQA [83], ArxivQA [59], GEOS [100], PMC-VQA [144], KVQA [101], Geometry3K [77], MapQA [13], PlotQA [88], ViQuAE [55], VQA-RAD [52], ST-VQA [9], TextVQA [106], LLaVAR [145], SIBR [133], MMC-Inst [68], IconQA [78], GQA [43], SciGraphQA [60], LRV-Instruction [67], DVQA [46], InfographicVQA [85], FigureQA [47], WikiTableQuestions [93], TAT-DQA [147], VisualMRC [113], ScienceQA [79], OCR-VQA [89], WebSRC [21], PathVQA [37], UniGeo [15], ScreenQA [38], VizWiz [35], SVIT [146], CogVLM [126], FM-IQA [30], VQAv2 [31], OK-VQA [82], EST-VQA [127], VisDial [27], Shikra [16], Super-CLEVR [61], LLaVA [69], IDK [12], AlfWorld [103], M-HalDetect [34], Cambrian7M [116], LLaVA-OneVision [56], mPLUG-DocOwl [135], UReader [136]

Table 6. Training data. This table presents the open-source datasets used in the fine-tuning stage, corresponding with the categories and data volume in Tab. 1 of the main text.

Please note that some datasets may belong to more than one category, and there may be overlapping data among these datasets.

###### 9. Hyper-parameters for Training

We list the hyper-parameters for the pre-training stage (stage 1) and fine-tuning stage (stage 2) in Tab. 7 and Tab. 8 respectively.

Configuration Stage 1 LLM Sequence Length 4096 Dynamic Resolution None (384×384) Optimizer AdamW Optimizer Hyperparams β1 = 0.9, β2 = 0.98, ϵ = 10−6 Peak LR 10−3 LR Schedule Cosine Decay Weight Decay 0.05 Training Steps 3.434k Warm-up Steps 34 Global Batch Size 720 Gradient Accumulation 1 Numerical Precision bfloat16

- Table 7. Hyper-parameters. Hyper-parameters for the pretraining stage (stage 1).

Configuration Stage 2 LLM Sequence Length 4096 Dynamic Resolution Up to 16 patches (1536×1536) Optimizer AdamW Optimizer Hyperparams β1 = 0.9, β2 = 0.98, ϵ = 10−6 Peak LR 10−4 LR Schedule Cosine Decay Weight Decay 0.05 ViT Layer-wise LR Decay 0.9 Training Steps 131k Warm-up Steps 1310 Global Batch Size 5760 Gradient Accumulation 8 Numerical Precision bfloat16

- Table 8. Hyper-parameters. Hyper-parameters for the finetuning stage (stage 2).

Please note that due to the upsampling of certain datasets with smaller data volumes, the product of Training Steps and Global Batch Size may exceed the total data volume.

###### 10. Visualization on Practical Cases

We present some demos of BlueLM-V-3B on mobile devices.

##### Math Ability

Human: Solve this problem.

[Figure 39]

Response: To solve this problem, we need to simplify the given expression and compare it to the options provided.

The expression is: \[ \frac{10x^8}{5x^4} = ? \]

First, we simplify the expression by dividing the coefficients and subtracting the exponents of the same base:

\[ \frac{10}{5} \cdot \frac{x^8}{x^4} = 2 \cdot x^{8-4}

= 2 \cdot x^4 \] So the simplified expression is: \[ 2x^4 \] Now, we compare this to the options:

- A) \( 2x^2 \)
- B) \( 2x^{12} \)
- C) \( 2x^4 \)
- D) \( 2x^{32} \)

The correct answer is C) \( 2x^4 \), as it matches the simplified expression.

##### Chart Understanding

Human: What is the percentage difference between Natural Gas and Petroleum Systems and MSW Landfills?

[Figure 40]

Response: The pie chart shows that Natural Gas and Petroleum Systems make up 30% of methane emissions, while MSW Landfills account for 14%. The percentage difference between Natural Gas and Petroleum Systems and MSW Landfills is 30% - 14% = 16%.

### Scientific Understanding

Human: Which car in the picture is more aerodynamic?

[Figure 42]

Response: The car on the right appears more aerodynamic due to its streamlined shape with a pointed front end and a sloped rear end, which reduces air resistance and could potentially allow it to move more efficiently through the air.

### Document Understanding

Human: What is the image about? Response: The image depicts "The Nobel Prize in Chemistry" for the year 2024, featuring three scientists: David Baker, Demis Hassabis, and John M. Jumper. Each is recognized for their contributions to computational protein design or structure prediction in science. The poster highlights their significant achievements in the field of chemistry and celebrates their awards.

[Figure 43]

### Multilingual Ability

[Figure 44]

Human: What are all the scene text in the image, and what do they mean?

Response: The scene text in the image is "Welcome bienvenida Bienvenue Willkommen benvenuto". It is a multilingual sign that says "Welcome" in English, Spanish, French, German and Italian respectively.

### Screen Understanding

[Figure 45]

Human: 我有⼏个闹钟?分别提醒我做什么?

Response: 有两个闹钟。⼀个是明天早上8点30分起床 的闹铃，另⼀个是今天下午16点00分的快递取件。

###### Image Description

Human: 详细描述照⽚中的内容。 Response: 这张图⽚展示了⼀个卡通⻛格的男孩雕塑。 他闭着眼睛，嘴巴张开，似乎在笑或者打哈⽋。他的 头发是⿊⾊的，有点凌乱。他穿着⼀件红⾊的上⾐， 上⾯有⻩⾊的纽扣和装饰性的图案，还有⼀条蓝⾊的 裤⼦，裤脚处有⽩⾊的⽑边装饰。男孩的脚上穿着棕 ⾊的鞋⼦，鞋带也是⿊⾊的。这个雕塑放置在⼀个圆 形的底座上，底座看起来像是⽤⽯头或类似材料制成， 颜⾊为浅灰⾊。背景是模糊的，没有太多细节，主要 是⽩⾊和灰⾊的⾊调，这使得雕塑更加突出。整体来 看，这个雕塑给⼈⼀种轻松愉快的感觉。

[Figure 47]

###### Chinese OCR

[Figure 48]

Human: 告诉我图⾥⾯写了什么。 Response: 若能避開猛烈的狂喜 ⾃然也不會有悲痛的來襲 太宰治|《⼈間失格》

###### Common Sense

[Figure 49]

Human: 我能吃这些蘑菇么? Response: 不，这些红盖⽩点的蘑菇可能有毒，不 建议⻝⽤。

###### Knowledge

Human: Introduce me the author of this painting.

[Figure 50]

Response: This painting is by Claude Monet, a renowned French painter who was a founding member of the Impressionist movement. Monet is celebrated for his innovative use of light and color to capture the fleeting moments of life and landscapes.

#### Multi-turn Conversation

Human: What are the departure point and destination of this train, and where is the next stop?

[Figure 52]

Response: The departure point is Praha hl.n. and the destination is Wien Hbf. The next stop is BrnoŽidenice.

Human: What is the train speed now? Response: The train speed is 160 km/h.

