## arXiv:2605.20147v1[cs.CV]19May2026

### PixVerve: Advancing Native UHR Image Generation to 100MP with a Large-Scale High-Quality Dataset

Haojun Chen1,⋆, Haoyang He1,⋆, Chengming Xu2,⋆, Qingdong He1, Junwei Zhu3, Yabiao Wang1, Zhucun Xue1, Xianfang Zeng1, Zhennan Chen3, Xiaobin Hu4, Hao Zhao5, Yong Liu1, Jiangning Zhang1,†, , Dacheng Tao6

1Zhejiang University, 2Fudan University, 3Nanjing University, 4National University of Singapore, 5Tsinghua University, 6Nanyang Technological University ⋆Joint first authors, †Corresponding author

Text-to-Image (T2I) models have recently seen notable progress around 1K and 2K resolution. With the extreme desire for better visual experience and the rapid development of imaging technology, the demand for Ultra-High-Resolution (UHR) image generation has grown significantly. However, UHR image generation poses great challenges due to the scarcity and complexity of high-resolution content. In this paper, we first introduce PixVerve-95K, a high-quality, open-source UHR T2I dataset curated with a carefully designed data pipeline, which contains 95K images across diverse scenarios (each image has a minimum pixel-count of 100M) and seven-dimensional annotations. Based on our large-scale image-text dataset, we take a pioneering step to extend various T2I foundation models to native 100MP generation with three training schemes. Finally, leveraging both conventional metrics and multimodal large language model-based assessments, our proposed PixVerve-Bench benchmark establishes a comprehensive evaluation protocol for UHR images encompassing visual quality and semantic alignment. Extensive experimental results on our benchmark and the constructive exploration of training strategies collaboratively provide valuable insights for future breakthroughs.

Date: May 20, 2026

Correspondence: 186368@zju.edu.cn Code: https://github.com/HaojunChen663/PixVerve-95K Data: https://modelscope.cn/datasets/APRIL6AIGC/PixVerve-95K Project: https://haojunchen663.github.io/projects/PixVerve/

[Figure 1]

[Figure 2]

#### 1 Introduction

In recent years, Text-to-Image (T2I) models have made remarkable advancements in synthesis quality and controllability [4, 29], underscoring their exceptional potential to revolutionize the paradigm of content creation. Despite substantial progress, most existing models focus on training and generation at fixed low-to-moderate resolutions (typically 1K and 2K). Directly extrapolating these models to Ultra-High-Resolution (UHR) scenarios inevitably leads to degradations such as structural artifacts, content repetition, and a pervasive loss of high-frequency details (see Fig. 2, top-right), which significantly hinder real-world applications that necessitate photorealistic visual fidelity.

With the extreme desire for better visual experience of the next-generation media [22, 52, 53, 57, 62] and empowered by computing resources, the demand for high-quality gigapixel-scale content has grown continuously in fields such as digital cinematography, immersive entertainment, and commercial design. Additionally, recent advancements in imaging technology and display devices have driven native 100-Megapixel (100MP) imaging a standard specification in modern smartphones of many brands and no longer confined to specialized domains. Furthermore, the theoretical resolution of the Human Visual System (HVS) is estimated to be 576 megapixels when integrating information across the 120-degree field of view [10]. This capacity implies that 100MP T2I generation is not merely a pursuit of larger dimensions, but a valuable quest to bridge the gap between digital synthesis and human perception. To this end, this work seeks to first advance UHR image generation to 100MP.

[Figure 4]

[Figure 5]

[Figure 6]

① 394×690

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

2K

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

② 555×431

[Figure 22]

[Figure 23]

PixVerve -Bench

[Figure 24]

[Figure 25]

[Figure 26]

4K

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

|①|
|---|

[Figure 32]

|②|
|---|

[Figure 33]

[Figure 34]

8K

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

③ 308×555

[Figure 41]

[Figure 42]

| |
|---|

[Figure 43]

⑤

[Figure 44]

③

[Figure 45]

|④|
|---|

[Figure 46]

- ④ 604×1140

[Figure 47]

[Figure 48]

- ⑤ 197×308

[Figure 49]

PixVerve-95K (Ours)

[Figure 50]

[Figure 51]

High-quality 100MP

[Figure 52]

8320×12480

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

- ① Basic Visual Scores (6 Metrics)
- ② Tags List
- ③ BBoxes Coordinates
- ④ Aesthetic Analysis (6 Dimensions)
- ⑤ Instance-level Description
- ⑥ Short Caption Overall Content Fine-grained Details Spatial Relations

A breathtaking aerial vista at golden hour captures the iconic landscape of Cappadocia, where multiple hot air balloons drift serenely above a tapestry of eroded volcanic rock formations and a clustered village nestled in the valley. Dominating the upper-left foreground, a massive balloon emblazoned with 'Traveller's' and 'Turkey' in bold script glows with warm amber light, its basket filled with silhouetted passengers, while a second balloon bearing a pixelated heart design and the name 'PASHA BALLOONS' floats centrally, its checkered pattern of orange, blue, and beige catching the low sun. To the far right, a third balloon with vertical black-and-white stripes and Chinese characters hovers near the horizon. Below, the village sprawls across the middle ground, its terracotta-roofed houses and a slender minaret rising from the cluster, surrounded by winding dirt roads and scattered white vans and cars, likely supporting the balloon operations. The background reveals undulating, sun-baked hills and towering fairy chimneys, their soft, eroded surfaces rendered in rich ochre and sienna tones by the directional sunlight, which casts long, dramatic shadows across the valley floor. Two dark silhouettes of birds soar in the hazy, golden sky, adding a sense of scale and motion. The atmosphere is thick with a warm, dusty haze that softens distant ridges, enhancing the dreamlike, almost surreal quality of the scene. The composition layers the foreground balloon, the mid-ground village and fields, and the distant geological formations into a harmonious, cinematic panorama, rendered with a rich, warm color palette and a shallow depth of field that gently blurs the immediate foreground, drawing the eye toward the luminous balloons and the sculpted earth below.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

###### Figure 1 PixVerve-95K is a large-scale, high-quality dataset for Ultra-High-Resolution (UHR) image generation, first advancing Text-to-Image (T2I) generation to the 100MP scale. Featuring high visual fidelity (right) and comprehensive annotations (bottom), it can meet the growing demand for next-generation T2I applications. PixVerve-Bench is a comprehensive benchmark suite comprising 8 metrics for the systematic evaluation of UHR T2I methods (top-right).

Recently, training-based methods for native image generation have demonstrated promising results at the 4K (∼16MP) resolution [55, 58, 59, 61]. Compared to training-free strategies [3, 11, 12, 23] which often exhibit excessive smoothing and implausible details (see Fig. 2, bottom-left), these approaches enable model backbones to explicitly capture the long-range correlations within UHR images, thus attaining better performance in detail synthesis. However, extending UHR image generation to native 100MP is not simply about resolution scaling and faces three core challenges: 1) The primary bottleneck for native 100MP T2I training and generation lies in the lack of suitable data. Existing UHR T2I datasets are modest in resolution (typically limited to 4K [59, 61]) due to the data scarcity and the difficulty in curating suitable data. Furthermore, public image-text corpora lack specialized captioning protocols for the UHR setting and rarely provide multi-dimensional, structured annotations which benefit precise control over various visual attributes. 2) The immense semantic complexity and vast pixel space of 100MP data make it challenging to design effective training schemes, which is largely unexplored in the current landscape. 3) Standard T2I evaluation protocols are inadequate for UHR scenarios, making it difficult to provide reliable feedback for training and model selection, as conventional metrics such as FID [20] and CLIPScore [19] fail to capture fine-grained details.

[Figure 67]

[Figure 68]

|[Figure 69]|
|---|

|[Figure 70]|
|---|

[Figure 71]

[Figure 72]

FLUX.2-Ⅰ Full

[Figure 73]

[Figure 74]

FLUX.2-klein-base-4B

(Ours)

(Base Model)

| |[Figure 75]|
|---|---|
| | |

|[Figure 76]|
|---|

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

HiFlow (FLUX.1-12B) Diffusion-4K (FLUX.1-12B)

Figure 2 Visual comparison of 4K image generations. Please zoom in for clearer details.

To bridge multi-faceted gaps, we propose a comprehensive methodology framework spanning dataset, model, and benchmark. Concretely, our core contributions are threefold:

- • We introduce PixVerve-95K, the first large-scale, high-quality T2I dataset to push image resolution to 100MP. With a five-stage, automated data pipeline, we curate 95,735 100MP images with fine-grained annotations (5 types of metadata and 2 comprehensive captions), directly supporting the training or fine-tuning of T2I models at high resolutions.
- • Based on our proposed PixVerve-95K, we first explore the attempt of generating 100MP images natively. Specifically, we extend existing T2I foundation models (including both latent diffusion models and pixel diffusion models) with three distinct training schemes, providing valuable insights and paving the way for future breakthroughs.
- • To address the limitations of conventional T2I benchmarks, we construct PixVerve-Bench, a systematic, hierarchical evaluation protocol incorporating both traditional metrics and assessments based on Multimodal Large Language Models (MLLMs).

#### 2 Related Work

##### 2.1 Text-to-Image Datasets

The evolution of Text-to-Image (T2I) generation has been fundamentally driven by the availability and quality of large-scale image-text datasets. The release of the early web-scale corpora such as LAION-400M [41] and LAION-5B [42] has significantly facilitated T2I foundation model training. As the field further matures, the focus of dataset construction starts to shift from mere volume toward high quality [27]. With the growing demand for higher resolution and visual fidelity, Diffusion-4K [59] introduces the first open-source 4K T2I dataset for native UHR image training. More recently, Aesthetic-Train-V2 [58] and UltraHR-100K [61] further expand the 4K T2I corpora. Despite these advances, most existing datasets are primarily constrained to the 1K-4K regime and often rely on global, superficial descriptions that lack the structural granularity and instance-level detail required to supervise the synthesis of exceptionally complex Ultra-High-Resolution (UHR) scenes.

##### 2.2 Text-to-Image Foundation Models

Mainstream T2I foundation models include the Generative Adversarial Network (GAN) [15], autoregressive (AR) models [37], and diffusion models (DMs) [21]. With this evolution of architectures, DMs have recently emerged as the prevailing paradigm, pushing T2I generation to an unprecedented level [28, 29, 32, 40, 45, 50]. A pivotal milestone is the introduction of latent diffusion models (LDMs) [40], which perform the diffusion process in a compressed latent space, alleviating computational burdens while maintaining high perceptual fidelity [13, 35]. More recently, Diffusion Transformers (DiTs) [34] have made remarkable progress within the LDM framework, offering superior scalability compared to traditional U-Net backbones. Parallel to the paradigm of LDMs, pixel diffusion models perform the diffusion process directly in the raw pixel space, which have regained attention for image generation these days [7, 8, 31, 48]. While LDMs are often preferred for their computational efficiency at moderate resolutions, pixel diffusion models offer a distinct advantage by bypassing the potential information loss and reconstruction artifacts inherent in Variational Autoencoder-based compression. Nevertheless, most current T2I foundation models are constrained to fixed low-to-moderate resolutions (typically 1024×1024), leaving UHR T2I generation a relatively under-explored field.

##### 2.3 Ultra-High-Resolution Image Generation

Beyond the 2K resolution threshold, image generation is currently dominated by LDMs. Existing solutions can be categorized into two main paradigms: training-free strategies for UHR scaling [3, 11, 18, 23, 44] and training-based methods for native UHR image generation [6, 39, 46, 55, 58, 59, 61]. Despite being more resource-friendly, the former approaches often suffer from object repetition, texture degradation, and unrealistic details. To enhance synthesis quality, the alternative direction curates UHR T2I corpora and trains or fine-tunes models at native high resolutions. However, current training-based frameworks remain

[Figure 85]

[Figure 86]

[Figure 87]

Ⅰ Image Data Collection Ⅱ Preliminary Data Purification Ⅳ Final Data Filtering

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

(Tailored for Synthetic Data)

High-Quality Real Data

Diverse T2I Data

Exposure Detection Sharpness Detection Flatness Detection Content Richness Detection Aesthetics Detection

[Figure 92]

Pixel Grayscale

[Figure 93]

[Figure 94]

Original Prompt Generation

Patch Seam Continuity Inspection

Internet Data (Pexels, Unsplash)

[Figure 95]

[Figure 96]

Seamless

# +

[Figure 97]

Laplacian Var.

Existing UHR T2I Datasets (Aesthetic-Train-V2,

Tile Stride: 384 Pixels Pixel Grad. Ratio

HR Image Synthesis

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

UltraHR-100K)

Sobel Var.

[Figure 102]

Candidate Set

Pixel-level, Structural, Perceptual Consistency

Post-SR Consistency Validation

[Figure 103]

(All Passed)

Entropy

[Figure 104]

Real Data Pool

T2I Data Pool

Down-sampled SR & Original LR PSNR SSIM LPIPS

4K

[Figure 105]

[Figure 106]

LAION & ArtiMuse

[Figure 107]

Region-level Artifact Assessment

Reso. Filtering

[Figure 108]

[Figure 109]

[Figure 110]

Ⅲ 100-Megapixel Data Curation

10 Non-overlapping Crops

Qwen3-VL-30B-A3B-Instruct

[Figure 111]

Multi-scale Fidelity

[Figure 112]

[Figure 113]

[Figure 114]

Super-Resolution

[Figure 115]

Total Pixels < 25M Total Pixels > 25M

|Prompt: Ultra-high-resolution photo-realistic image with the preservation of high-frequency spatial details and authentic micro-textures, maintaining consistent geometric perspective. Emphasize authentic dermal microscopic texture and fine follicular detail, avoid plastic-like smoothing. Extreme structural clarity and crisp edge definition without halo artifacts, eliminate aliasing on fine grids. … Ensure global semantic coherence and pixel-level precision for a professional-grade 100-megapixel output.<br><br>|
|---|

ODTSR (Built on Qwen-Image)

Total Pixels > 100M

Instance-level Artifact Assessment

2× SR

4× SR

Instance-centered Crops

Qwen3-VL-30B-A3B-Instruct

|[Figure 116]|
|---|

|[Figure 117]|
|---|

[Figure 118]

Native 100MP Data

[Figure 119]

[Figure 120]

All Passed

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

PixVerve-95K

… …

Directly Archived

Discarded

[Figure 125]

Ⅴ Stage-wise Data Caption

[Figure 126]

Qwen3-30B-A3BInstruct-2507

[Figure 127]

Qwen3-VL-235B-A22B-Instruct

Qwen3-VL-235B-A22B-Instruct

[Figure 128]

[Figure 129]

RAM++

[Figure 130]

[Figure 131]

[Figure 132]

["carpet", "couch", "dog", "man", "woman", "plant", "laptop", "light", "curtain"]

Crop with Padding Add Visual Prompts

Tags

[Figure 133]

[Figure 134]

[Figure 135]

Comprehensive Caption

Dense Instance-level Descriptions

[Figure 136]

Rex-Omni "carpet” [0, 4931, 4476, 6720] “couch” [991, 3350, 4471, 5186] “dog” …

[Figure 137]

[Figure 138]

BBox Coords.

[Figure 139]

Long Caption

[Figure 140]

[Figure 141]

[Figure 142]

carpet’s description & importance score

[Figure 143]

[Figure 144]

Macroscopic Perception Microscopic Details Aesthetics-aware Des.

[Figure 145]

[Figure 146]

… … …

[Figure 147]

SAM 2 Mask-based NMS

[Figure 148]

carpet’s description & importance score

[Figure 149]

Masks

[Figure 150]

[Figure 151]

…

[Figure 152]

[Figure 153]

Short Caption

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

ArtiMuse Holistic

② Visual Elements & Structure

③ Technical Execution

① Composition & Design

Summarized by Qwen3-VL235B-A22B-Instruct

[Figure 159]

Aesthetics-level Analysis

[Figure 160]

[Figure 161]

[Figure 162]

④ Originality & Creativity

⑤ Theme & Communication

⑥ Emotion & Viewer Response

Figure 3 Overview of our PixVerve-95K curation pipeline that includes: 1) High-quality and diverse raw image data acquisition (Sec. 3.1.1). 2) Preliminary data purification comprising five parallel detection procedures (Sec. 3.1.2). 3) 100MP data curation via super-resolution (Sec. 3.1.3). 4) Final data filtering to ensure the quality of our synthetic data (Sec. 3.1.4). 5) Stage-wise data caption pipeline carefully designed for UHR images (Sec. 3.1.5).

confined to the sub-4K [6] or 4K [55, 58, 59, 61] scale, still falling short of the gigapixel-scale fidelity required for real-world applications. In this paper, we aim to take a pioneering step and push the frontier of T2I to the 100MP scale.

#### 3 Methodology: Dataset, Model, and Benchmark

In this work, we operationalize Native 100MP Text-to-Image Generation as a dedicated training and evaluation regime, significantly distinct from approaches of training-free resolution upscaling. Training-based methods treat UHR image generation as an end-to-end task that requires intrinsic high-resolution priors, while executing this regime necessitates addressing two fundamental challenges: i) high-quality 100MP T2I datasets and ii) training recipes. Also, the absence of a systematic T2I benchmark designed for UHR scenarios hinder further research on this valuable topic. Resolving these challenges requires a holistic methodology that integrates data, model training, and evaluation.ression.

##### 3.1 Curating PixVerve-95K Dataset

To facilitate direct training at native 100MP resolution, we curate the first large-scale 100MP T2I dataset, addressing the critical deficit of UHR corpora in the current landscape. Beyond the pursuit of extreme resolution, we prioritize high image quality and caption comprehensiveness. To this end, we carefully design and implement a five-stage data pipeline, which is intuitively shown in Fig. 3.

###### 3.1.1 Raw Image Data Collection

High-quality real data collection. To establish a large-scale image corpus for UHR T2I generation, we begin by collecting high-resolution real imagery from diverse sources. We harvest high-quality photography from platforms Pexels [25] and Unsplash [9] via official APIs, while also integrating a subset from Aesthetic-TrainV2 [58] and UltraHR-100K [61]. Both data collection streams are subjected to a deduplication procedure and

notably, we apply the following resolution-based screening criteria to construct a data pool prior to 100MP upscaling: i) total pixels exceeding 25M with a minimum dimension of 3,000 pixels, or ii) total pixels ranging from 10M to 25M with a minimum dimension of 1,500 pixels. Detailed clarification on image licensing is provided in Sec. C.

Diverse T2I data generation. To further enhance semantic diversity and ensure the comprehensiveness of visual concepts, we complement the real data with synthesized data. Specifically, we leverage GPT-5.1 [47] to generate a set of wide-ranging, expressive prompts, which are subsequently sent to the advanced Nano Banana Pro [16] to generate high-quality 4K images. Together with the real data, these diverse synthesized images constitute our raw data pool (approximately 300K).

###### 3.1.2 Preliminary Data Purification

Large-scale image corpora collected from diverse sources inevitably contain subpar samples suffering from technical degradation (e.g., exposure anomalies, blurriness, etc.), which can undermine the learning efficacy of T2I models. Therefore, to establish a baseline of visual excellence, we comprehensively evaluate each image in our raw data pool across five fundamental dimensions:

Exposure detection. Overexposure and underexposure degrade the image quality greatly. Taking 5 as the threshold, we calculate the cumulative proportion of pixels with values above 250 or below 5 for each image. Any image of which the proportion exceeds 20% is deemed anomalous and excluded.

Sharpness detection. To eliminate the presence of out-of-focus or motion-blurred images, we utilize the Laplacian variance as an interpretable metric for image sharpness assessment. Images yielding a score below the threshold of 10 are identified as insufficiently clear and discarded from the corpus.

Flatness detection. To suppress images dominated by textureless regions, we partition each image into local patches and compute the proportion of overly smooth patches based on the Sobel variance. Images are considered to severely lack texture and then removed if the proportion exceeds 97.5%.

Content richness detection. Beyond basic physical properties, superior content richness is another defining characteristic of a high-quality image. We employ the classical signal, Shannon entropy [43], to quantify the informational density, retaining the top 60% highest-entropy images in the raw pool.

Aesthetics detection. Aesthetic appeal plays an important role in high-quality image generation. For aesthetics detection, we adopt a coupling approach which combines the LAION Aesthetic Predictor [30] and ArtiMuse [5], a modern MLLM-based aesthetics evaluator. We utilize both predictors to assess the aesthetic quality of each image in the raw pool with the score SL and SA respectively. Images of which SL or SA ranks among the top 60% are preserved.

Exposure Sharpness Flatness Content Richness Aesthetics

[Figure 164]

|[Figure 165]<br><br>[Figure 166]<br><br>0.422|
|---|

|[Figure 167]<br><br>[Figure 168]<br><br>5.712|
|---|

|[Figure 169]<br><br>[Figure 170]<br><br>1.000|
|---|

|[Figure 171]<br><br>[Figure 172]<br><br>5.274|
|---|

|[Figure 173]<br><br>[Figure 174]<br><br>4.944 48.485|
|---|

Discarded

[Figure 175]

|[Figure 176]<br><br>[Figure 177]<br><br>0.034|
|---|

|[Figure 178]<br><br>[Figure 179]<br><br>190.310|
|---|

|[Figure 180]<br><br>[Figure 181]<br><br>0.100|
|---|

|[Figure 182]<br><br>[Figure 183]<br><br>7.893|
|---|

|[Figure 184]<br><br>[Figure 185]<br><br>6.364 71.988|
|---|

Qualified

Figure 4 Representative discarded and qualified examples for each dimension. Green scores represent disqualification, while red scores represent passing the detection.

By taking the intersection of the subsets retained from the five detection procedures above, the final candidate pool is derived. We present representative discarded and qualified examples for each dimension along with their corresponding scores in Fig. 4, demonstrating the necessity and effectiveness of our preliminary data purification.

###### 3.1.3 100MP Data Curation

Given the scarcity of native 100MP image data in our candidate pool, we employ ODTSR [14], a novel super-resolution (SR) framework based on Qwen-Image [50] which considers both fidelity and controllability, to bridge this gap and expand the scale of our final corpus. Notably, we employ a tiling strategy to tackle the high-resolution nature, which incorporates overlapping strides and feathering matrices to facilitate smooth

###### Table 1 Data Flow and Refinement Details.

Pipeline Stage Resulting Subset Scale Image Data Collection Raw Data Pool 300,316 (5,000 Synthesized Images) Preliminary Data Purification Candidate Data Pool 122,866 Final Data Filtering Final Data 95,935 (95,735+200)

transitions. We implement distinct upscaling intensities for different source resolutions to reach the uniform 100MP threshold, leveraging textual prompts as conditional guidance: i) native 100MP images are directly archived; ii) images with a total pixel count exceeding 25M are elevated via 2× SR; and iii) for the remaining images with total pixels in the 10M-25M range, a 4× SR process is performed. This tiered production pipeline ensures that all samples achieve a minimum resolution of 100MP with high perceptual fidelity, establishing a sound data foundation for UHR T2I training and generation.

###### 3.1.4 Final Data Filtering

To guarantee the quality of our synthetic 100MP data, we rigorously implement a four-tiered filtering pipeline, which specially targets different problems potentially introduced during the SR process.

Patch seam continuity inspection. To eliminate color discontinuities and geometric misalignments, we compute the pixel gradient ratio across all horizontal and vertical seams defined by the 384-pixel tile stride used in Sec. 3.1.3. An image is considered defective and strictly excluded if any detected seam exhibits a ratio exceeding the threshold rt = 2.5.

Post-SR consistency validation. To ensure pixel-level, structural, and perceptual fidelity, each synthetic 100MP image is down-sampled to its original resolution and compared against its initial input via Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and LPIPS [60]. Any candidate image that fails to satisfy the tri-metric thresholds is consequently discarded.

Region-level artifact assessment. To prevent local degradations such as geometric deformations and warped human features, we partition each synthetic 100MP image into non-overlapping patches of size 768 and employ a hybrid sampling strategy to select ten representative patches: six with the highest texture complexity (via the Sobel variance) and the remaining four sampled randomly. All selected patches are then evaluated by Qwen3-VL-30B-A3B-Instruct [1]. An image is strictly discarded if more than one of its sampled patches is identified as containing noticeable artifacts.

Instance-level artifact assessment. We further scrutinize key instances leveraging the image crops obtained in Sec. 3.1.5. Similarly, we employ Qwen3-VL-30B-A3B-Instruct [1] to evaluate each crop, adopting a stringent criterion where an image is excluded if any instance is flagged as defective.

Tab. 1 illustrates the specific data flow and the data scale at each major stage, tracing the refinement process from the initial collection to the final curated corpus.

###### 3.1.5 Stage-wise Data Caption

Detailed captions are crucial for fine-grained controllable image generation, which is widely recognized [2, 55, 61]. However, standard zero-shot MLLM prompting often fails to encapsulate the intricate details present in UHR images. To address this challenge, we propose a hierarchical stage-wise pipeline which decouples the captioning process into three distinct layers:

Dense instance-level descriptions generation. To facilitate precise alignment at the instance level, we design a cascaded pipeline utilizing the capabilities of foundation models and MLLMs. We first employ RAM++ [24] for open-vocabulary tagging to generate semantic tags, which are pruned by Qwen3-30B-A3BInstruct-2507 [54] to retain only tangible object tags. Rex-Omni [26] predicts bounding boxes (bboxes) for these filtered tags, followed by a step where SAM 2 [38] performs instance segmentation and generates high-fidelity masks. We further apply Non-Maximum Suppression (NMS) based on IoU to deduplicate overlapping masks and remove trivial objects with an area threshold. For context-aware captioning, we generate a visual pair

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

10461×23907

10622×15548

###### 14016×9344

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

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

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

13260×10608

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

11008×16512

- 1st Zoom-in
- 2nd Zoom-in

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

- ① Basic Visual Scores (6 Metrics)
- ② Tags List
- ③ BBoxes Coordinates
- ④ Aesthetic Analysis (6 Dimensions)
- ⑤ Instance-level Description
- ⑥ Short Caption Long Caption

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Long Caption: A dramatic, overcast sky heavy with textured gray and white cumulus clouds looms over a picturesque European riverside town, casting a moody, diffused light that accentuates the rich, saturated colors of the architecture below. In the foreground, a weathered stone bridge with multiple rounded arches spans a dark, gently flowing river, its moss-covered abutments and rough-hewn blocks showing centuries of erosion and resilience. Lush, untamed green foliage spills over the riverbank and creeps up the bridge’s supports, adding a wild, organic contrast to the man-made structures. Along the river’s edge, a row of vibrant, multi-story buildings lines the embankment: a bright ochre-yellow house with white-trimmed windows and a small balcony, a salmon-pink building adorned with flower boxes, and a striking half-timbered structure with exposed dark wood beams and warm orange stucco. To the left, a white building with green shutters features a faded fresco above its windows, while a distinctive green-domed clock tower with a weather vane rises behind it. Dominating the mid-ground, a white church with a tall, pointed spire and red brick accents stands proudly on a hill, flanked by dense trees and a larger, pale-yellow institutional building further uphill. The composition layers depth masterfully, from the textured stone parapet in the immediate foreground to the distant forested hills under the brooding sky, creating a scene that balances historical charm with atmospheric tension, where every surface—from the peeling paint on window frames to the damp sheen on the river—contributes to a richly detailed, lived-in narrative.

Overall Content

Figure 5 Qualitative samples in our PixVerve-95K dataset. The zoomed-in regions highlight the fine-grained and high-fidelity details.

Fine-grained Des. of Key Instances

Spatial Relations between Entities

for each identified instance. Specifically, we crop out a sub-image centered on the target instance with 5% padding and incorporate a highlighted prompt on the original image using its mask. These visual pairs are finally sent to Qwen3-VL-235B-A22B-Instruct [1] to generate comprehensive instance-level descriptions and assign a semantic importance score to each instance.

Holistic aesthetics-level analysis. Beyond instance details, a high-quality image caption should encompass an aesthetic depiction spanning multiple dimensions. To this end, we adopt ArtiMuse [5] to provide an expert-style aesthetic analysis across six key dimensions (composition & design, visual elements & structure, technical execution, originality & creativity, theme & communication, and emotion & viewer response), which serves as a vital reference for final caption summarization.

Comprehensive caption summarization. Based on key instances’ detailed descriptions and the aesthetic analysis, we employ Qwen3-VL-235B-A22B-Instruct [1] as a caption synthesis expert. With the original image and all aggregated metadata, it first generates a coherent long caption encompassing the overall content and style, fine-grained details of instances, and clear relations between objects (as shown in Fig. 1). Paired with the original image, this long caption is subsequently distilled by the same MLLM into a short caption that encapsulates the core semantic essence in a concise and fluid narrative, which can meet diverse task requirements together with the long version.

###### 3.1.6 Statistical Comparison and Analysis

With our five-stage data pipeline, we construct PixVerve-95K, comprising 95,735 100MP images with comprehensive annotations. Fig. 5 presents some qualitative samples, which are best viewed zoomed-in. As summarized in Tab. 2, PixVerve-95K is the first to push open-source T2I data to 10K resolution (∼100MP), providing five-dimensional metadata (basic visual scores, tags, bboxes, aesthetics-level analysis, and instancelevel description) beyond long and short captions. These structured annotations offer versatile utility for the

- Table 2 Comparison with open-source UHR T2I datasets. Our proposed PixVerve-95K is the first premium T2I dataset to push image resolution to 10K (∼100MP), providing multi-dimensional, fine-grained metadata and significantly longer comprehensive captions.

Variable-length Caption PixArt-30k [6] 2531 × 2656 30,000 71.3 words ✘ ✘ ✘ ✘ ✘ ✘

Avg. Resolution (height × width)

Avg. Caption Length

Basic Visual Scores

Aesthetics-level Analysis

Instance-level Description

Dataset

Number

Tags BBoxes

Aesthetic-Train [59] 4578 × 4838 12,015 24.2 words ✘ ✘ ✘ ✘ ✘ ✘ Aesthetic-Train-V2 [58] 4861 × 5127 105,288 38.1 words ✘ ✘ ✘ ✘ ✘ ✘

UltraHR-100K [61] 3654 × 5143 100,486 109.2 words ✘ ✘ ✘ ✘ ✘ ✘ PixVerve-95K (Ours) 13031 × 15348 95,735

234.1 words (Long)

✔ (6 Metrics)

✔ (6 Dimensions)

✔ ✔

✔ ✔

[Figure 251]

4734

[Figure 252]

10000

[Figure 253]

|Real Data|
|---|
| |
|Native 2× SR 4× SR<br><br>|

4500

8000

[Figure 254]

47892

Long Caption Short Caption Instance-level Description

[Figure 255]

Avg. Instances: 43 Max Instances: 818

3000

6000

Num

Num

4000

1500

|T2I Data|
|---|

42932

2000

0

Tech Human

177

30

0 10 20 40 50 60

Instance Num

0

(c) Instance Num Distribution

Art

0

100 200 300 400 500

(b) Data Source Distribution

Word Length (grouped by 2)

[Figure 256]

(d) Caption Word Length

[Figure 257]

Nature

30955

25000

Aspect Ratio

30000

[Figure 258]

Interior

- [1/3, 2/3)
- [2/3, 1) [1, 1.5) [1.5, 3) Others

|Datasets Aesthetic-4K<br><br>UltraHR-100K PixVerve-95K (Ours)|
|---|

26171

25000

20000

21605

Flora & Fauna

Height(pixels)

Urban

20000

15000

Num

15000

10000

16K

10000

8804

8200

5000

8K 4K

5000

2K

0

0

0 5000 10000 15000 20000 25000 30000 35000 40000

100M 150M 200M 250M ≥300M

Width (pixels)

Total Pixels

(a) Image Topic Distribution

(e) Image Resolution and Aspect Ratio Distribution

(f) Image Resolution and Aspect Ratio Comparison

(g) Long Caption Word Cloud

Figure 6 Statistical distributions of our PixVerve-95K.

community, enabling granular control over data quality and facilitating adaptive sampling strategies tailored to specialized training objectives. Fig. 6 visualizes the statistical distributions of PixVerve-95K from multiple perspectives, highlighting the scenario diversity, balanced aspect ratios, and the exceptional expressiveness of aggregated captions.

##### 3.2 Extending Text-to-Image Foundation Models to Native 100MP Generation

Training T2I foundation models at native 100MP poses great challenges due to the immense semantic complexity and vast pixel space. Bridging this gap necessitates a collaborative exploration of diverse architectural and optimization strategies rather than a singular fix. Based on PixVerve-95K, we conduct a multi-faceted exploration to retrofit T2I foundation models for native 100MP synthesis. Specifically, we investigate three distinct schemes to identify the optimal path:

- • Scheme I: Full-Attention LDM Fine-tuning. As direct baselines, we perform full-parameter training and use LoRA for parameter-efficient fine-tuning (PEFT) on FLUX.2-klein-base-4B [29], respectively. While maximizing the retention of the pre-trained model’s semantic prior and structural integrity, this approach encounters severe hardware constraints. The latent representation dimensions lead to an exponential surge in memory requirements, mandating model parallelism for inference at the 100MP scale, which limits the flexibility for general-purpose deployment in real world.
- • Scheme II: Window-Attention Retrofitting and LDM Fine-tuning. Inspired by EMOv2 [56], we refine the training strategy by introducing a local-to-global attention mechanism. We retrofit the joint attention in FLUX.2 into a dual-branch window-attention, without altering the core architecture of full-attention pretrained models. Specifically, text queries attend to all text and image tokens, while each image query attends to all text tokens and two complementary image-token neighborhoods. The close branch partitions the latent grid into contiguous spatial windows, preserving high-frequency local structure; while the remote branch groups tokens with the same modulo offset under the same partition stride, providing sparse long-range communication across the canvas. The outputs of the close and remote branches are averaged to approximate the original full image attention. For a latent grid consisting of N = HW image tokens, a

- standard full self-attention mechanism incurs a quadratic computational cost of O(N2). By retrofitting the attention with partition factors (a,b), this scheme effectively reduces the image-image attention complexity to approximately O(2N2/(ab)), while the complexity for text-image conditioning remains linear with respect to N since the number of text tokens is small. Following T3-Video [57], we further cycle layer-wise partition schedules with different window size so different blocks exchange information under different receptive-field shapes.
- • Scheme III: Patch-based Diffusion in Pixel Space. Motivated by recent pixel diffusion models [7, 8, 31], we explore a paradigm bypassing the latent space entirely. Following L2P [8], which is built on DiP [7], we adopt a patch-based pixel diffusion framework that decouples global structure from local refinement: a transformer backbone operates on large image patches for long-range semantics and spatial layout, while a lightweight head leverages contextual features and original noisy patches to reconstruct fine details. From the theoretical perspective of L2P, large-patch tokenization preserves global low-frequency information efficiently, but high-frequency components are only weakly recovered during denoising unless explicit local inductive bias is introduced, making dedicated patch refinement crucial for faithful pixel-space reconstruction. However, scaling this scheme to UHR scales exposes a severe sequence-length and memory bottleneck. To enable training on a single 96 GB GPU card, we adaptively adjust the patch size to control the token count at the cost of coarser patch-level representations at higher resolutions.

Progressive Training Strategy. To mitigate the training instability inherent in the transition from standard resolutions to 100MP, we implement a three-stage progressive training strategy across all schemes. Models are fine-tuned through three graduated resolution tiers: 4K (∼16MP), 8K (∼64MP), and finally the target 10K (100MP). Concretely, these constructive experimental routes collaboratively provide critical insights into the resolution scalability of current T2I foundation models.

##### 3.3 PixVerve-Bench Construction and Evaluation

For systematic and universal evaluation of UHR T2I models, we introduce PixVerve-Bench, comprising 200 manually picked images across diverse scenarios with an average resolution of 12369×14377. The benchmark framework (see Fig. 1, top-right) leverages both conventional metrics and novel MLLM-as-a-judge protocols, providing a holistic evaluation across two complementary aspects: 1) Visual Quality Assessment, comprising four critical dimensions, including distribution consistency, aesthetic quality, texture granularity, and multi-scale fidelity; and 2) Semantic Alignment Evaluation, which assesses instructional adherence across scene-level correspondence and instance-centric compliance. Detailed evaluation procedures and scoring formulas are provided in Sec. F.

###### 3.3.1 Visual Quality Assessment

Visual Quality evaluates the intrinsic physical attributes and perceptual realism of the generated images across the following four dimensions:

Distribution Consistency. Following common practice, we employ FID [20] to evaluate the overall distribution fidelity of the generated images. Considering that FID is calculated on down-sampled (299×299) images and neglects details, we incorporate FIDpatch to scrutinize local patches.

Aesthetic Quality. Utilizing the LAION Aesthetic Predictor [30], we map each generated image into an aesthetic feature space and score its aesthetic quality.

Texture Granularity focuses on the richness and complexity of micro-patterns, which are among the most significant characteristics of UHR images. Using the Gray Level Co-occurrence Matrix (GLCM) [17] Score, we provide a rigorous diagnosis of whether the generated images suffer from monotonous flatness. Higher scores indicate richer texture and greater details.

Multi-scale Fidelity. The fidelity of UHR images is susceptible to both global artifacts (e.g., structural incoherence and physical distortion) and local artifacts (e.g., unrealistic noise and pattern repetition), which are difficult to capture using conventional metrics. Therefore, we employ Qwen3.5-35B-A3B [36] to perform a rigorous multi-scale fidelity assessment. Specifically, we systematically categorize the UHR-specific artifacts into two main dimensions and nine fine-grained sub-dimensions. The model is instructed to assign a score on a

five-point scale for each sub-dimension based on the severity of the artifact, followed by a unifying step where these individual scores are integrated into an interpretable metric, the Multi-scale Fidelity Index (MSFI), to reflect overall performance.

###### 3.3.2 Semantic Alignment Evaluation

Semantic Alignment evaluates how well the generated visual content adheres to the provided textual prompts. Given the expansive canvas and semantic complexity in UHR T2I generation, we hierarchically assess instructional adherence across two levels of granularity:

Scene-level Correspondence. For this foundational granularity, we first utilize CLIPScore [19] to measure the global semantic correlation using the short captions. Furthermore, we incorporate FG-CLIP2 Score [51] to better capture fine-grained details, which is computed on the long captions.

Instance-centric Compliance. To complement global correlation metrics, we propose the Instance-centric Compliance Score (ICS). ICS leverages the advanced capabilities of Qwen3.5-35B-A3B [36] to assess semantic alignment across three hierarchical dimensions: Instance Existence Verification, Appearance Attribute Alignment, and Spatial Relation Accuracy, which provides a fine-grained and interpretable metric for measuring whether visual elements adhere to textual prompts.

#### 4 Experiments

- 4.1 Experimental Setup Overall training settings. As introduced in Sec. 3.2, we investigate three training schemes in our experiments.

- Scheme I: We fine-tune the pretrained FLUX.2-klein-base-4B [29] model using both full-parameter tuning and LoRA adaptation. For full-parameter fine-tuning, the learning rate is fixed at 1×10−5 across all resolution tiers. For LoRA fine-tuning, we use a learning rate of 1 × 10−4 and set the LoRA rank to 32. Scheme II: For the window-attention retrofitting, we adopt window aspect ratios of 1:1, 1:2, 2:1, 1:8, and 8:1. The window size is scaled linearly with the input resolution. Scheme III: To enable training on a single 96 GB GPU card at different scales, we adjust the patch size used in L2P [8] accordingly. Specifically, the patch sizes are set to 64, 128, and 320 for 4K, 8K, and 10K resolution, respectively. Unless otherwise specified, the learning rate is uniformly set to 5 × 10−5. More training details including the fine-tuning epochs and computational expenditures are provided in Sec. E.1.

Baselines. Corresponding with our proposed training schemes, we denote our fine-tuned variants as FLUX.2-I (Full), FLUX.2-I (LoRA), FLUX.2-II, and L2P-III, respectively. To comprehensively evaluate our approaches, we conduct extensive comparisons against different baselines across three UHR scales: 4K (4096×4096), 8K (8192×8192), and 10K (10240×10240). The compared methods encompass: i) direct extrapolation of pre-trained T2I models (FLUX.2-klein-base-4B [29], Qwen-Image [50], and L2P [8]), ii) representative training-free strategies (DemoFusion [11], LinFusion [33], and HiFlow [3]), and iii) recent training-based models (UltraPixel [39], UltraFlux [55], and Diffusion-4K [59]). All baselines are evaluated with their official implementations and parameter settings.

- 4.2 Experimental Results, Observations, and Analysis

Tab. 3 presents the quantitative performance across three different resolutions on PixVerve-Bench, with detailed sub-dimension performance of the MSFI shown in Tab. A4). Fig. 7 as well as Fig. 2 illustrates a qualitative comparison at 4K resolution. In this section, we report our key observations and analysis regarding the resolution scalability of current methods and different T2I foundation models.

Base Model and Existing Methods. Directly extrapolating the base model FLUX.2-klein-base-4B to UHR generation leads to severe degradation: at 8K, FID exceeds 422, and CLIPScore drops to 18.345. Training-free strategies such as DemoFusion [11] and LinFusion [33] can maintain relatively good local visual statistics beyond 4K, e.g., DemoFusion obtains the best FIDpatch at 8K and 10K. However, their performance on semantic alignment remains limited; DemoFusion achieves ICS below 3.7 across all resolutions, indicating

- Table 3 Quantitative comparison on PixVerve-Bench. The best result is highlighted in bold, while the second-best result is underlined. – indicates complete failures such as producing meaningless textures or black images, which are not applicable to the semantics-agnostic GLCM Score and MSFI.

Visual Quality Semantic Alignment

Resolution

|(height × width)<br><br>Method<br><br>|FID ↓ FIDpatch ↓ Aesthetics ↑<br><br>GLCM Score ↑ MSFI ↑|CLIPScore ↑<br><br>FG-CLIP2 Score ↑ ICS ↑|
|---|---|---|
|4K (4096 × 4096)<br><br>FLUX.2-klein-base-4B [29] Qwen-Image [50] L2P [8] DemoFusion [11] LinFusion [33] HiFlow [3] UltraPixel [39] UltraFlux [55] Diffusion-4K [59] FLUX.2-I (Full) FLUX.2-I (LoRA) FLUX.2-II L2P-III<br><br>|167.234 76.794 5.498 0.873 7.408 140.740 53.200 5.707 0.611 8.293<br><br>126.089 90.803 5.852 0.818 6.904 142.981 55.164 6.167 0.733 8.246 142.933 68.184 6.302 0.394 8.171 130.337 49.842 6.189 1.068 8.779 144.859 66.878 6.260 0.732 9.153 121.337 49.902 6.068 1.037 8.712 134.702 78.323 5.848 0.668 8.377 128.897 45.204 5.804 0.987 8.911<br><br>127.436 40.433 5.798 0.977 8.977 161.729 76.460 5.530 0.593 8.125 118.183 98.704 5.792 0.896 6.990<br><br><br>|31.058 17.049 5.376<br><br>33.041 18.492 6.753<br><br>33.890 19.501 7.954<br><br>32.608 18.105 3.619 32.246 17.694 3.849<br><br>34.190 19.643 6.978<br><br><br>32.430 17.836 3.741<br><br>34.909 20.084 8.530<br><br>33.421 18.749 6.423<br><br>34.161 19.683 8.533 34.178 19.767 8.420 31.663 18.119 5.340 34.264 19.676 7.870<br><br><br><br><br><br><br>|
|8K (8192 × 8192)<br><br>FLUX.2-klein-base-4B [29] L2P [8] DemoFusion [11] LinFusion [33] FLUX.2-I (Full) FLUX.2-I (LoRA) L2P-III<br><br>|422.737 350.331 4.121 – – 140.396 87.167 5.590 0.363 7.592 176.068 58.480 6.031 0.514 6.966 143.429 62.658 6.271 0.177 7.797 197.690 66.571 5.289 – – 277.410 99.414 4.752 – – 134.635 133.453 5.569 1.122 5.310<br><br>|18.345 0.503 0.318 33.261 18.548 6.394<br><br>31.529 17.057 2.122<br><br>32.097 17.530 3.809 28.676 10.765 0.420 24.250 8.086 0.411 32.788 17.802 5.504<br><br><br>|
|10K (10240 × 10240)<br><br>L2P [8] DemoFusion [11] LinFusion [33] L2P-III|156.158 116.379 5.438 0.533 7.397 179.063 61.854 5.895 0.538 6.710 152.964 70.718 6.215 0.156 7.525 159.212 192.286 5.569 1.100 5.567<br><br>|32.440 17.866 5.548<br><br>30.930 15.920 2.671<br>31.937 17.380 4.825 31.810 17.057 3.586<br><br><br>|

that tiled or progressive inference struggles to preserve prompt consistency. At 4K, UltraFlux [55] is a strong baseline with FID 121.337 and ICS 8.530, but our FLUX.2-I achieves better local fidelity with lower FIDpatch.

###### Scheme I: Strong 4K Adaptation but Poor Scalability. Fine-tuning the base model with Scheme I achieves

the best balance at 4K. Compared with the base FLUX.2-klein-base-4B, the LoRA variant reduces FIDpatch from 76.794 to 40.433, while the full-parameter variant improves ICS from 5.376 to 8.533. These results show that fine-tuning effectively adapts local image statistics while preserving the semantic prior. Qualitatively, it generates coherent layouts and fine details without over-sharpening or texture collapse. However, this advantage comes with heavy computational overheads. As shown in Tab. A1, full fine-tuning at 4K costs over 20,000 H20 GPU hours, and only 0.25 epoch at 8K costs over 10,000 GPU hours. Inference also scales poorly: Tab. A2 shows that FLUX.2-I variants require 8 GPUs and over 100s at 4K, over 1,200s at 8K, and nearly 3,000s at 10K. This confirms that full-attention LDM fine-tuning is effective at 4K but impractical for native 100MP synthesis.

###### Scheme II: Faster Attention with Optimization Difficulty. FLUX.2-II reduces computational cost through window-attention retrofitting. As shown in Tab. A2, the inference time at 4K is reduced by approximately 32s on 8 GPUs, giving 30% speedup. Its 4K training cost is also lower than that with Scheme I, requiring 9,216 H20 GPU hours for 3 epochs. However, we observe that performance drops noticeably: FLUX.2-II

obtains FIDpatch 76.460 and ICS 5.340, close to the base model but much worse than both FLUX.2-I variants. This suggests a mismatch between the pretrained full-attention structure and the retrofitted local attention pattern. In practice, the model requires more optimization steps to recover global communication, and the final quality is sensitive to the window size and aspect-ratio schedules.

###### Scheme III: Best Scalability with a Patch-Size Trade-off. It is surprisingly observed that L2P-III shows the strongest scalability among our explored schemes. It achieves the best FID at 4K and 8K, with scores of 118.183 and 134.635, and remains functional at 10K. More importantly, Tab. A2 shows that L2P-III needs one GPU and 58s, 70s, and 88s for 4K, 8K, and 10K inference. Compared with DemoFusion, it is up to

FLUX.2-Ⅰ(Full) FLUX.2-Ⅱ L2P-Ⅲ FLUX.2-klein-base-4B DemoFusion HiFlow UltraPixel Diffusion-4K

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

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

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

Figure 7 Qualitative comparison of different methods at 4K (4096×4096) resolution.

156× faster. Compared with FLUX.2-I LoRA at 10K, it is over 33× faster while using 1 GPU instead of 8. The main limitation is the patch-size trade-off: to fit training into a single 96 GB GPU card, the patch size increases from 64 to 128 and 320, reducing memory cost but weakening fine-detail reconstruction. This explains its less competitive FIDpatch and MSFI at high resolutions. Overall, patch-based pixel diffusion is currently the most practical path toward native 100MP generation.

##### 4.3 Ablation Study and Discussion

Ablation on image caption quality. We investigate the impact of caption granularity on image synthesis by comparing 4K image generations under short and long prompting settings on PixVerve-Bench. Evaluations across UltraFlux, Diffusion-4K, and FLUX.2-I (Full) reveal consistent performance gains when using long captions, as summarized in Tab. 4, indicating that increased descriptive granularity and semantic density are of great significance for UHR image generation. The visual comparison is shown in Fig. A4.

Discussion on downstream applications. Beyond T2I generation, our dataset supports diverse downstream tasks such as UHR image quality assessment, fine-grained understanding in UHR contexts, image outpainting, and benchmarking for image compression. To validate one of these potential utilities, we conduct a performance evaluation of different lossless image compressed formats using our 100MP images. Compared to small images for which program startup time accounts significantly during testing, time consumption on large images can better reflect the true performance of the codecs. Detailed experimental setup and results are provided in Sec. H.

Table 4 Ablation on image caption quality.

Method Caption FID ↓ FIDpatch ↓ Aesthetics ↑ GLCM Score ↑ UltraFlux

Short 126.316 55.732 6.058 1.008 Long 121.337 49.902 6.068 1.037

Short 142.628 90.728 5.748 0.606 Long 134.702 78.323 5.848 0.668

Diffusion-4K

Short 135.837 51.173 5.793 0.893 Long 128.897 45.204 5.804 0.987

FLUX.2-I (Full)

#### 5 Conclusion

In this paper, we present a comprehensive framework to explore the frontier of native 100MP T2I generation, encompassing data, training, and evaluation. To address key challenges in data scarcity and semantic

granularity, we introduce PixVerve-95K, a high-quality, open-source 100MP T2I dataset curated with a five-stage automated pipeline which ensures data excellence. Based on PixVerve-95K, we explore distinct training schemes to extend current T2I foundation models to native 100MP generation. Finally, our proposed PixVerve-Bench tailored for UHR scenarios further provides reliable feedback for model evaluation and selection.

Limitations and broader impact. As a work primarily contributing a novel dataset and evaluation benchmark, there remains significant scope for investigating UHR-specific architectural adaptations and more efficient, robust training recipes. And compared to existing general-purpose T2I datasets, the corpus size of PixVerve95K remains still limited, though our highly scalable curation process. Considering broader impacts, while photorealistic UHR image generation greatly empowers real-world applications, the extreme realism poses heightened ethical risks concerning the proliferation of misinformation and the misuse of AI-generated content. We emphasize vigilance regarding these societal implications within the research community and advocate for multi-dimensional regulations and technical response solutions.

#### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 6, 7
- [2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 6
- [3] Jiazi Bu, Pengyang Ling, Yujie Zhou, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Hiflow: Training-free high-resolution image generation with flow-aligned guidance. arXiv preprint arXiv:2504.06232, 2025. 2, 3, 10, 11
- [4] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 1
- [5] Shuo Cao, Nan Ma, Jiayang Li, Xiaohui Li, Lihao Shao, Kaiwen Zhu, Yu Zhou, Yuandong Pu, Jiarui Wu, Jiaquan Wang, et al. Artimuse: Fine-grained image aesthetics assessment with joint scoring and expert-level understanding. arXiv preprint arXiv:2507.14533, 2025. 5, 7
- [6] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024. 3, 4, 8
- [7] Zhennan Chen, Junwei Zhu, Xu Chen, Jiangning Zhang, Xiaobin Hu, Hanzhen Zhao, Chengjie Wang, Jian Yang, and Ying Tai. Dip: Taming diffusion models in pixel space. arXiv preprint arXiv:2511.18822, 2025. 3, 9
- [8] Zhennan Chen, Junwei Zhu, Xu Chen, Jiangning Zhang, Jiawei Chen, Zhuoqi Zeng, Wei Zhang, Chengjie Wang, Jian Yang, and Ying Tai. L2p: Unlocking latent potential for pixel generation. arXiv preprint arXiv:2605.12013,

2026. 3, 9, 10, 11

- [9] Mikael Cho. Unsplash. https://unsplash.com/images, 2013. 4, 17
- [10] Roger N Clark. Notes on the resolution and other details of the human eye. Clarkvision. com, 2005. 1
- [11] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising high-resolution image generation with no $. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6159–6168, 2024. 2, 3, 10, 11, 19
- [12] Ruoyi Du, Dongyang Liu, Le Zhuo, Qin Qi, Hongsheng Li, Zhanyu Ma, and Peng Gao. I-max: Maximize the resolution potential of pre-trained rectified flow transformers with projected flow. 2024. 2
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 3

- [14] Yushun Fang, Yuxiang Chen, Shibo Yin, Qiang Hu, Jiangchao Yao, Ya Zhang, Xiaoyun Zhang, and Yanfeng Wang. One-step diffusion transformer for controllable real-world image super-resolution. arXiv preprint arXiv:2511.17138,

2025. 5

- [15] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 3
- [16] Google. Gemini. https://gemini.google.com/, 2025. 5
- [17] Robert M Haralick, Karthikeyan Shanmugam, and Its’ Hak Dinstein. Textural features for image classification. IEEE Transactions on systems, man, and cybernetics, (6):610–621, 2007. 9, 19
- [18] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higher-resolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations, 2023. 3
- [19] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021. 2, 10
- [20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 2, 9
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [22] Teng Hu, Jiangning Zhang, Zihan Su, and Ran Yi. Ultragen: High-resolution video generation with hierarchical attention. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 4923–4931, 2026. 1
- [23] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. In European conference on computer vision, pages 196–212. Springer, 2024. 2, 3
- [24] Xinyu Huang, Yi-Jie Huang, Youcai Zhang, Weiwei Tian, Rui Feng, Yuejie Zhang, Yanchun Xie, Yaqian Li, and Lei Zhang. Open-set image tagging with multi-grained text supervision. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4117–4126, 2025. 6
- [25] Ingo, Bruno Joseph, and Daniel Frese. Pexels images. https://www.pexels.com/images/, 2014. 4, 17
- [26] Qing Jiang, Junan Huo, Xingyu Chen, Yuda Xiong, Zhaoyang Zeng, Yihao Chen, Tianhe Ren, Junzhi Yu, and Lei Zhang. Detect anything via next point prediction. arXiv preprint arXiv:2510.12798, 2025. 6
- [27] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023. 3
- [28] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. 3
- [29] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025. 1, 3, 8, 10, 11
- [30] LAION-AI. aesthetic-predictor. https://github.com/LAION-AI/aesthetic-predictor, 2022. 5, 9
- [31] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025. 3, 9
- [32] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 3
- [33] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024. 10, 11
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 3
- [35] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3

- [36] Qwen Team. Qwen3.5: Towards Native Multimodal Agents. https://qwen.ai/blog?id=qwen3.5, February 2026. 9, 10, 20, 21
- [37] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 3
- [38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 6
- [39] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra high-resolution image synthesis to new peaks. Advances in Neural Information Processing Systems, 37:111131–111171, 2024. 3, 10, 11
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [41] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 3
- [42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294,

2022. 3

- [43] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3): 379–423, 1948. 5, 19
- [44] Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. Resmaster: Mastering high-resolution image generation via structural and fine-grained guidance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 6887–6895, 2025. 3
- [45] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4733–4743, 2024. 3
- [46] Luigi Sigillo, Shengfeng He, and Danilo Comminiello. Latent wavelet diffusion for ultra-high-resolution image synthesis. arXiv preprint arXiv:2506.00433, 2025. 3
- [47] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 5
- [48] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025. 3
- [49] WangXuan95. Image-compression-benchmark. https://github.com/WangXuan95/ Image-Compression-Benchmark, 2025. 23
- [50] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 3, 5, 10, 11
- [51] Chunyu Xie, Bin Wang, Fanjing Kong, Jincheng Li, Dawei Liang, Ji Ao, Dawei Leng, and Yuhui Yin. Fg-clip 2: A bilingual fine-grained vision-language alignment model. arXiv preprint arXiv:2510.10921, 2025. 10
- [52] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024. 1
- [53] Zhucun Xue, Jiangning Zhang, Teng Hu, Haoyang He, Yinan Chen, Yuxuan Cai, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, et al. Ultravideo: High-quality uhd video dataset with comprehensive captions. arXiv preprint arXiv:2506.13691, 2025. 1
- [54] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 6

- [55] Tian Ye, Song Fei, and Lei Zhu. Ultraflux: Data-model co-design for high-quality native 4k text-to-image generation across diverse aspect ratios. arXiv preprint arXiv:2511.18050, 2025. 2, 3, 4, 6, 10, 11
- [56] Jiangning Zhang, Teng Hu, Haoyang He, Zhucun Xue, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, and Dacheng Tao. Emov2: Pushing 5 m vision model frontier. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 8
- [57] Jiangning Zhang, Junwei Zhu, Teng Hu, Yabiao Wang, Donghao Luo, Weijian Cao, Zhenye Gan, Xiaobin Hu, Zhucun Xue, and Chengjie Wang. Transform trained transformer: Accelerating naive 4k video generation over 10×. arXiv preprint arXiv:2512.13492, 2025. 1, 9
- [58] Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Ultra-high-resolution image synthesis: Data, method and evaluation. arXiv preprint arXiv:2506.01331, 2025. 2, 3, 4, 8, 17
- [59] Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Diffusion-4k: Ultra-high-resolution image synthesis with latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23464–23473, 2025. 2, 3, 4, 8, 10, 11
- [60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [61] Chen Zhao, En Ci, Yunzhe Xu, Tiehan Fan, Shanyan Guan, Yanhao Ge, Jian Yang, and Ying Tai. Ultrahr-100k: Enhancing uhr image synthesis with a large-scale high-quality dataset. arXiv preprint arXiv:2510.20661, 2025. 2, 3, 4, 6, 8, 17
- [62] Yushen Zuo, Qi Zheng, Mingyang Wu, Xinrui Jiang, Renjie Li, Jian Wang, Yide Zhang, Gengchen Mai, Lihong V Wang, James Zou, et al. 4kagent: agentic any image to 4k super-resolution. arXiv preprint arXiv:2507.07105,

2025. 1

#### Appendix

The appendix presents the following sections to strengthen the main manuscript:

- — Sec. A provides implementation details of flatness detection.
- — Sec. B provides a further frequency-domain analysis to confirm the quality of PixVerve-95K.
- — Sec. C provides a detailed clarification on the licensing for our proposed dataset to ensure transparency and ethical compliance.
- — Sec. D presents more qualitative samples in PixVerve-95K.
- — Sec. E provides more training and inference details of different approaches.
- — Sec. F provides additional details on metrics of PixVerve-Bench, including specific evaluation procedures, scoring formulas, and human alignment analysis.
- — Sec. G provides more quantitative and qualitative results of model performance.
- — Sec. H provides benchmark results of image compression using our 100MP images.
- — Sec. I shows the prompts used for MLLM evaluation in PixVerve-Bench.

#### A Implementation Details of Flatness Detection

In the stage of preliminary data purification (Sec. 3.1.2), we conduct a flatness detection procedure based on the Sobel variance. Each image in the raw data pool is first converted to grayscale and partitioned into

240 × 240 non-overlapping patches. We then compute the gradient magnitude Gmag = G2x + G2y for each patch, using Sobel operators with a kernel size of 3. A patch is categorized as “textureless” if the variance of its gradient magnitude falls below a predefined threshold of 750, and an image is discarded if the proportion of textureless patches exceeds 97.5%. Notably, both thresholds are intentionally conservative and determined empirically through manual visual audits, which ensures the efficient elimination of overly flat images while still preserving legitimate low-texture content.

#### B Further Analysis of PixVerve-95K Dataset

We conduct an extended frequency-domain quality analysis of our PixVerve-95K dataset, utilizing the Radially Averaged Power Spectrum (RAPS). This analysis intuitively presents the power distribution across spatial frequencies, providing insights into the realism of synthesized high-frequency textures. As illustrated in Fig. A1 (a), the power spectral density of our synthetic 100MP data closely matches the native distribution across the entire frequency range; notably, no significant energy attenuation is observed in the high-frequency regime, confirming the micro-texture fidelity of our rigorously preserved synthetic data. Furthermore, we analyze the consistency between down-sampled SR images and their original low-resolution (LR) counterparts. The alignment of the power spectrum shown in Fig. A1 (b) indicates that our data curation pipeline maintains global structural consistency and adheres to the underlying distribution of the original images. Overall, this analysis further substantiates that our PixVerve-95K provides high-quality, ultra-high-resolution data, ensuring its reliability for downstream large-scale generative modeling.

#### C Licensing and Dataset Release

This section provides a detailed account of the licensing for our proposed dataset. As illustrated in the main manuscript, a significant portion of the images was sourced from Pexels [25] and Unsplash [9]. These platforms operate under permissive licenses that grant broad permissions for downloading, using, and modifying visual content for both commercial and non-commercial purposes without financial obligation. Additionally, a subset is from Aesthetic-Train-V2 [58] and UltraHR-100K [61], which are governed by the MIT License and the Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) License, respectively. Both frameworks

[Figure 301]

[Figure 302]

Down-sampled SR Original LR

OriginalLR Data Down-sampledSR Data

Native 100MP Data Synthetic 100MP Data

1013

1014

PowerSpectralDensity(Log)

PowerSpectralDensity(Log)

1011

1012

109

[Figure 303]

1010

107

108

105

10-3 10-2 10-1

10-3 10-2 10-1

Normalized Frequency (cycles/pixel) Normalized Frequency (cycles/pixel)

(a) RAPS of Native and Synthetic 100MP Data (b) RAPS of Down-sampled SR Data and Original Data

Figure A1 RAPS comparisons.

permit the utilization of data for non-commercial research purposes. Therefore, we affirm that our dataset is in full compliance with current copyright laws and privacy regulations, and this dataset is released under the CC BY-NC 4.0 license to prevent unauthorized commercial exploitation.

#### D Qualitative Samples in PixVerve-95K Dataset

Fig. A5 and Fig. A6 show two qualitative samples in our PixVerve-95K dataset. Best viewed zoomed-in.

#### E More Training and Inference Details

- E.1 Training Details

Table A1 Training details of different schemes.

Training Resolution Number of Epochs Total H20 GPU Hours Scheme I (Full) 4K 6 21,888 8K 0.25 10,752 Scheme I (LoRA) 4K 3 10,368 8K 0.25 10,752

- 10K 0.32 28,416

- Scheme II

4K 3 9,216

- Scheme III

4K 4.3 8,448 8K 2.9 18,432

- 10K 1.7 23,040

For each training scheme, we report the finetuning epochs at each resolution scale and the corresponding NVIDIA H20 GPU hours in Tab. A1. GPU hours are computed as the product of the number of NVIDIA H20 GPUs used and the wall-clock training time. Certain schemes do not complete the full three-stage training process since they have exhibited unsatisfactory performance at low resolutions. These computational details are provided to ensure the reproducibility of our main experimental results.

For Scheme I, we consider both full-parameter fine-tuning and LoRA for parameter-efficient fine-tuning. Specifically, the full-parameter trained 8K model is initialized from the 4K checkpoint trained for 3 epochs, and is further fine-tuned for 0.25 epochs at 8K resolution. All LoRA models at different resolutions are independently fine-tuned from the same base model, FLUX.2-klein-base-4B, rather than being initialized from lower-resolution LoRA checkpoints. For Scheme II, we report the cost of full-parameter fine-tuning with window-attention retrofitting at 4K resolution. For Scheme III, the training also follows a progressive curriculum: the 8K model is initialized from the 4K checkpoint, while the 10K model is initialized from the 8K checkpoint after 1 epoch of training.

- E.2 Inference Details

In this section, we report the per-sample inference cost of different methods in Tab. A2. The number of GPUs denotes the number of devices jointly used to generate one sample, rather than the batch size, and the inference time is measured as wall-clock latency in seconds.

Table A2 Inference details of different methods.

Inference Resolution Number of GPUs Inference time (s) Scheme I (Full) 4K 8 103 8K 8 1,234

Scheme I (LoRA) 4K 8 103 8K 8 1,252

For Scheme I, as we expect, full fine-tuning and LoRA fine-tuning show nearly identical inference latency, since LoRA changes the adaptation parameterization but does not alter the dominant full-attention computation over the high-resolution latent grid. Therefore, the latency of both variants grows rapidly as the inference resolution increases. Increasing the resolution from 4K to 8K raises the inference time from 103s to over 1,200s, and the LoRA variant requires nearly 3,000s at 10K. Moreover, all these runs require 8 GPUs for generating a single sample, revealing the substantial memory and deployment cost of full-attention latent diffusion at the 100MP scale.

10K 8 2,977

- Scheme II

4K 8 71

- Scheme III

4K 1 58 8K 1 70

10K 1 88

DemoFusion [11] 4K 1 945 8K 1 6,366 10K 1 13,689

- Scheme II reduces the 4K inference time from 103s to 71s under the same 8-GPU setting, which verifies that the window-attention retrofitting effectively lowers the attention cost while retaining compatibility with the pretrained latent diffusion backbone. However, it still inherits the multi-GPU inference requirement of the latent-space pipeline, making it an efficiency improvement rather than a complete solution to the deployment bottleneck.

In contrast, Scheme III exhibits the most favorable inference behavior. It runs on a single GPU card at all evaluated resolutions, with inference times of 58s, 70s, and 88s at 4K, 8K, and 10K, respectively. The nearly flat latency derives from the adaptive patch-size design, which controls the transformer sequence length as the image resolution increases. Compared to the representative training-free method, DemoFusion, Scheme III is 16.3×, 90.9×, and 155.6× faster at 4K, 8K, and 10K, respectively.

Overall, these results reveal a clear efficiency and deployability trade-off. Full-attention LDM fine-tuning preserves the original computation of the foundation model but becomes prohibitively expensive at ultra-high resolutions. Window-attention retrofitting provides a useful intermediate point by reducing latent attention cost. Patch-based pixel diffusion, although relying on coarser patch-level representations at higher resolutions, is the only explored scheme that enables native 100MP image generation with a single GPU card and sub-minute latency.

#### F Additional Details on Metrics of PixVerve-Bench

##### F.1 GLCM Score

In this section, we provide the detailed computation procedure of the GLCM Score, which is introduced to quantitatively evaluate the texture granularity in PixVerve-Bench. The GLCM Score is computed by first quantizing the grayscale intensities of the generated image into 64 levels and partitioning it into a set of 64 × 64 non-overlapping local patches {p1,p2,...,pP}. For each patch pi, a normalized Gray Level Cooccurrence Matrix [17] Gp

is constructed across multiple predefined distances δ ∈ {1,2,3,4} and orientations

i

θ ∈ {0, π4, π2, 34π} to capture spatial correlations. Subsequently, the detail richness of pi is measured via the average Shannon entropy [43], which is calculated over all spatial parameters and denoted as H(Gp

). Finally, the GLCM Score S for the entire image is defined as the arithmetic mean of the entropy values across all P

i

|[Figure 306]<br><br>UltraPixel<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>0.280 1.853 0.566<br><br>[Figure 311]<br><br>[Figure 312]|
|---|

|[Figure 313]<br><br>HiFlow<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]<br><br>[Figure 317]<br><br>0.377 1.827 1.376<br><br>[Figure 318]<br><br>[Figure 319]|
|---|

UltraPixel

HiFlow

|[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]<br><br>[Figure 323]<br><br>UltraFlux<br><br>[Figure 324]<br><br>0.440 1.705 1.700<br><br>[Figure 325]<br><br>[Figure 326]|
|---|

|[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>Diffusion-4K<br><br>[Figure 331]<br><br>0.298 1.073 1.135<br><br>[Figure 332]<br><br>[Figure 333]|
|---|

Diffusion-4K

UltraFlux

- Figure A2 Case visualization of the GLCM Score. We present representative 4096×4096 images generated by UltraPixel, HiFlow, UltraFlux, and Diffusion-4K along with their corresponding scores, demonstrating that higher scores indicate richer texture. Prompts are from PixVerve-Bench.

patches:

P

1 P

S =

H(Gp

),

i

i=1

providing an objective statistical assessment of the micro-structural complexity which is essential for UHR image generation. The case visualization of the GLCM Score is shown in Fig. A2.

##### F.2 Multi-scale Fidelity Index (MSFI)

In this section, we provide the detailed procedure and formulas of computing the Multi-scale Fidelity Index (MSFI).

Taxonomy of evaluation dimensions. As described in the main manuscript, the MSFI systematically assesses UHR image fidelity across two complementary dimensions, spanning nine fine-grained sub-dimensions that each target a distinct artifact category. Tab. A3 summarizes the evaluation dimensions, sub-dimensions, corresponding descriptions, and criteria.

Detailed evaluation procedures. For each predefined sub-dimension, we instruct Qwen3.5-35B-A3B [36] to assign a score on a five-point scale reflecting artifact severity. Specifically, for global-scale assessment, the MLLM evaluates the complete generated image resized to a standardized resolution. For local-scale assessment, we adopt the hybrid strategy used in Sec. 3.1.4 to sample ten representative local patches per image, with the patch size set to 512 × 512 for images below 8K resolution and 1024 × 1024 otherwise; during each scoring round, the target local patch is evaluated at its native resolution, with the down-sampled complete image provided for contextual reference and the patch’s relative spatial coordinates explicitly specified in the prompt; finally, the local-scale fidelity score of each image is derived by averaging the ratings assigned to the ten sampled patches. Furthermore, we provide detailed definitions for each sub-dimension and descriptions for each score level to ensure that the model maintains consistent criteria across multiple rounds of evaluations. The case visualization of each sub-dimension is shown in Fig. A3. The prompts used for global-scale and local-scale fidelity evaluation are provided in Sec. I.

Unified scoring formulation. For a given evaluation dimension D (global-scale or local-scale), let the set of scores for different sub-dimensions be {s1,s2,...,sn

D}. The score for dimension D is defined as:

D} with corresponding weights {w1,w2,...,wn

Σn

i=1wi · si Σn

D

SD =

.

i=1wi

D

The weights are determined through a user study, where participants provided importance ratings (1-5) for all sub-dimensions. These ratings were averaged, rounded to the nearest integer, and applied directly as the

Table A3 Description and evaluation criteria for each sub-dimension of the MSFI. We detail the hierarchical framework designed to assess the fidelity of UHR images across dual scales.

Dimension Sub-dimension Description and Evaluation Criteria

Structural Coherence Evaluates the physical plausibility of overall spatial arrangements and global geometric integrity of entities, ensuring anatomical correctness (e.g., no missing or redundant limbs).

Perspective Integrity Assesses whether vanishing points and the relative scale of objects at varying depths conform to perspective principles, identifying any geometric distortions against common sense.

Global-scale

Lighting Consistency Inspects global illumination for naturalism, checking for consistent light direction and the absence of artificial luminance gradients or disjointed shading.

Color Harmony Examines chromatic transitions for smoothness, checking for quantization artifacts (e.g., color banding) or unnatural boundary blurring between distinct color blocks.

Noise & Grain Existence Detects stochastic high-frequency chroma noise and unnatural graininess

within local patches that deviate from the expected sensor behavior. Generative Artifacts Scrutinizes local patches for typical synthesis flaws, including

checkerboard patterns, aliasing, and edge halos.

Texture Fidelity Differentiates between realistic surface roughness and “plastic-like” oversmoothing, ensuring natural materials exhibit stochastic randomness rather than mechanical repetition.

Local-scale

Micro-geometry Coherence Analyzes the continuity of local contours at the pixel level, penalizing unacceptable jitter, “staircase” effects, or jagged edges in high-contrast regions.

Sharpness Consistency Validates the consistency of the focal plane, ensuring that areas within the same depth of field maintain uniform clarity without abnormal, localized blurring.

weights. The overall MSFI for image I is given by:

MSFI(I) = Sglobal(I) + wl ·

1 10

Σ10i=1Slocal(I,i),

where Sglobal(I) and Slocal(I,i) denote the aggregated global fidelity score of I and the local fidelity score of the ith patch sampled from I. Notably, we set the weighting factor wl as Sglobal5 (I), claiming that global structural integrity is a fundamental prerequisite for microscopic realism. This formulation effectively penalizes “structurally incoherent” images that might otherwise attain misleadingly high scores due to sharp local textures. Consequently, the MSFI ranges from 1.2 to 10, where a score approaching 10 signifies superior multi-scale fidelity.

##### F.3 Instance-centric Compliance Score (ICS)

In this section, we provide the specific evaluation procedure and scoring formulas of the Instance-centric Compliance Score (ICS), which is proposed to quantify how precisely visual instances adhere to complex textual descriptions. We implement the ICS evaluator leveraging the Qwen3.5-35B-A3B model [36].

Evaluation dimensions. The assessment of the ICS is performed across three distinct dimensions, each scored by MLLM on a ten-point scale based on specific rubrics:

• Instance Existence Verification (IEV): This dimension serves as the gatekeeper, identifying whether all primary and secondary instances specified in the long caption are present. It focuses purely on presence or absence rather than quality.

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

SC-global: 3 PI: 2 LC: 2

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

CH: 2 NGE: 2.1 GA: 2.7

|[Figure 348]|
|---|

|[Figure 349]|
|---|

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

TF: 2.8 MGC: 3.1 SC-local: 1.9

- Figure A3 Case visualization of the MSFI. We present typical examples of low-quality generations across the nine sub-dimensions, accompanied by their corresponding scores. Blue scores denote sub-dimension ratings for global-scale fidelity, while purple scores indicate those for local-scale fidelity, which are the arithmetic means of scores across ten sampled patches. SC-global, PI, LC, CH, NGE, GA, TF, MGC, and SC-local denote the nine MLLM-as-a-judge sub-dimensions described in Tab. A3, respectively. Red bounding boxes are utilized to highlight specific degradations identified within the examples.

- • Appearance Attribute Alignment (AAA): For identified instances, this dimension evaluates whether the corresponding visual attributes (e.g., color, texture, material, shape, etc.) are compliant with the textual description.
- • Spatial Relation Accuracy (SRA): This dimension assesses whether the relative positioning (e.g., “left of”, “behind”, “in the foreground”, etc.) and the logical perspective are accurately depicted in the synthesized images.

We also provide detailed descriptions for each score level to ensure consistent criteria across multiple rounds of evaluations. The prompt used for instance-centric compliance evaluation is provided in Sec. I.

Unified scoring formulation. For each generated image, we instruct the MLLM to evaluate the aforementioned three dimensions, yielding a set of raw scores denoted as {SIEV ,SAAA,SSRA}. Considering the hierarchical dependency where appearance attributes and spatial relations are contingent upon the existence of the instances themselves, we employ a gated weighted average strategy to synthesize these three-dimensional

scores into the final ICS:

ICS =

SIEV 10 × (α · SAAA + β · SSRA),

where the term SIEV /10 acts as a penalty for instance omissions. In our implementation, we set α = 0.6 and β = 0.4 to prioritize appearance attribute fidelity. Consequently, an ICS nearing 10 indicates superior instance-centric compliance, while a low score reflects either obvious entity omissions or a misalignment of visual details.

##### F.4 Human Alignment for Metric Validation

To validate that our proposed automated MSFI and ICS align with human-centric preferences, this section provides a quantitative analysis. We select four representative text-to-image models, M = {MA,MB,MC,MD}, to generate 4K and 8K images. For each resolution, 30 unique prompts are utilized, resulting in a total of C42 × 60 = 360 pair-wise comparison sets. We recruit 8 participants to conduct this human preference study, each tasked with evaluating 90 image pairs to ensure that every pair receives two independent annotations. Participants are provided with detailed definitions and illustrative examples for two evaluation dimensions: Image Fidelity and Instance-centric Semantic Alignment, which correspond to the MSFI and ICS, respectively. For each comparison pair (Ii,Ij) generated by Mi and Mj, annotators are instructed to indicate a preference or label the pair as “indistinguishable”. Let si,j be the score assigned to model Mi in a single comparison:

 

1.0 if Mi is preferred, 0.5 if indistinguishable, 0 if Mj is preferred.

si,j =



For each evaluation dimension, the final human preference score for each model is computed as the total score divided by the number of comparisons. We observe that the performance rankings of the four models yielded by our proposed MSFI and ICS perfectly match the rankings derived from human preference scores. This consistency demonstrates that the MSFI and ICS are well-aligned with human subjective judgments in distinguishing the quality and alignment capabilities of different T2I models.

#### G More Quantitative and Qualitative Results

- G.1 More Quantitative Results

Tab. A4 details the performance of different methods on the MSFI. SC-global, PI, LC, CH, NGE, GA, TF, MGC, and SC-local denote the nine MLLM-as-a-judge sub-dimensions described in Tab. A3, respectively.

- G.2 More Qualitative Results

Fig. A4 visualizes the qualitative comparison of 4K image generations under short and long prompting settings. All case pairs are generated by our FLUX.2-I (Full).

#### H Benchmarking for Image Compression at 100MP Scale

Following the benchmark specifications in [49], we convert 25 images from our dataset into PNM format (a simple image format which stores raw pixels) for standard testing. A total of 20 image compressed formats are evaluated under strict lossless settings. The testing environment consists of a 12th Gen Intel i7-12700H CPU @2.30 GHz, 16 GB of RAM, and the Windows 11 operating system, and all codecs run in a single thread. The detailed benchmark results, including compressed size, compression time, and decompression time, are summarized in Tab. A5.

Table A4 Detailed MSFI comparison across various UHR scales, dimensions, and sub-dimensions. – indicates complete failures such as producing meaningless textures or black images, which are not applicable to the semanticsagnostic MSFI. Higher values indicate better performance.

Dimension Performance Sub-dimension Performance

Resolution (height × width)

Method

Global Fidelity

Local Fidelity

SC-global PI LC CH NGE GA TF MGC SC-local

FLUX.2-klein-base-4B 3.955 4.366 3.485 4.260 4.240 4.385 4.326 4.381 4.332 4.409 4.426 Qwen-Image 4.343 4.547 4.010 4.470 4.620 4.710 4.546 4.560 4.515 4.573 4.583

L2P 4.572 2.539 4.255 4.820 4.785 4.780 2.508 2.550 2.531 2.556 2.556 DemoFusion 4.237 4.731 3.685 4.510 4.665 4.780 4.786 4.713 4.704 4.724 4.776

LinFusion 4.410 4.264 3.935 4.685 4.750 4.845 4.315 4.291 4.202 4.295 4.301

HiFlow 4.471 4.848 4.180 4.910 4.915 4.930 4.889 4.831 4.828 4.847 4.885 UltraPixel 4.688 4.761 4.315 4.945 4.950 4.975 4.783 4.762 4.741 4.766 4.786 UltraFlux 4.669 4.330 4.260 4.945 4.965 4.980 4.314 4.341 4.312 4.354 4.351

4K (4096 × 4096)

Diffusion-4K 4.577 4.151 4.165 4.860 4.875 4.885 4.125 4.171 4.121 4.186 4.187 FLUX.2-I (Full) 4.561 4.762 4.120 4.860 4.870 4.905 4.760 4.765 4.747 4.770 4.786

FLUX.2-I (LoRA) 4.575 4.801 4.155 4.865 4.855 4.910 4.815 4.796 4.785 4.807 4.830 FLUX.2-II 4.150 4.784 3.650 4.530 4.365 4.615 4.791 4.792 4.761 4.797 4.814 L2P-III 4.493 2.764 4.165 4.815 4.660 4.665 2.750 2.768 2.753 2.782 2.785

FLUX.2-klein-base-4B – – – – – – – – – – –

L2P 4.558 3.317 4.240 4.830 4.735 4.770 3.216 3.361 3.292 3.377 3.358 DemoFusion 3.647 4.550 3.075 3.770 4.110 4.430 4.763 4.615 4.313 4.647 4.735

8K (8192 × 8192)

LinFusion 4.250 4.173 3.755 4.510 4.615 4.730 4.230 4.193 4.121 4.193 4.200 FLUX.2-I (Full) – – – – – – – – – – –

FLUX.2-I (LoRA) – – – – – – – – – – – L2P-III 3.490 2.580 3.255 4.020 3.410 3.365 2.542 2.594 2.563 2.601 2.618

L2P 4.301 3.573 3.970 4.585 4.495 4.510 3.468 3.617 3.548 3.627 3.627 DemoFusion 3.448 4.730 2.890 3.525 3.945 4.230 4.810 4.704 4.697 4.722 4.785

10K (10240 × 10240)

LinFusion 4.139 4.090 3.620 4.400 4.545 4.640 4.145 4.113 4.041 4.107 4.108 L2P-III 3.577 2.746 3.204 4.017 3.744 3.680 2.711 2.745 2.721 2.757 2.835

Short Captions

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Long Captions

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

###### Figure A4 Qualitative comparison of 4K image generations under short and long prompting settings. Top: images generated by our FLUX.2-I (Full) using short captions; bottom: generated images using corresponding long captions.

Table A5 Performance comparison of different lossless image compressed formats at the 100MP scale.

###### Compressed Format Compressed Size (bytes) ↓ Compression Time (s) ↓ Decompression Time (s) ↓

fNBLI 2152173140 68.343 44.276

NBLI (-g) 2122763920 483.106 429.641 HALIC 2327457195 23.266 33.072 HALICfast 2501106140 16.154 25.109

KVICK 2467300114 60.579 43.413 QIC 2675297097 32.500 30.960 QLIC2 2415123224 40.452 38.795

LEA 1898470283 772.283 870.666 BMF 2260500400 554.801 254.198 BIM 2399426455 301.153 339.631 QOI 3685837372 28.764 22.864

QOIR 3421495512 28.557 26.181 ZPNG 3003246621 36.658 29.638

PNG (optimizing=False) 3207101968 760.569 65.350 PNG (optimizing=True) 3138083188 5417.594 74.167

- JPEG-XL (-q 100 -e 1) 2765116877 77.526 107.456
- JPEG-XL (-q 100 -e 2) 2557728595 233.997 112.632
- JPEG-XL (-q 100 -e 3) 2315011289 511.114 314.402 WEBP (lossless m1) 2387603822 2536.114 75.605

JPEG-LS 2969890552 234.688 199.415

#### I Prompts for MLLM Evaluation

###### Global-scale Fidelity Evaluation Prompt

# ROLE AND TASK FORMULATION:

You are a professional image quality assessment (IQA) and scoring expert specializing in T2I evaluation. Your task is to carefully evaluate the **global-scale fidelity** of the provided synthetic image across 4 dimensions. Please focus on the overall composition and structure, geometric and physical logic, and macro-scale consistency.

# CRITICAL SCORING RULES (Must Strictly Follow):

- 1. **Objectivity & Fairness:** Maintain an objective stance throughout the evaluation process and base your judgement on visual evidence with the same standard instead of subjective preference.
- 2. **Focus Solely on Fidelity:** Consider the image category and expected characteristics while avoiding any bias towards the content of the image. Score based on the visual quality and fidelity aspects solely.
- 3. **Independence:** Evaluate each dimension independently without any halo effects.
- 4. **Rigor:** Apply strict criteria and any noticeable artifact should be reflected in the scoring. Maintain a high standard for what constitutes a “5” (Excellent). # EVALUATION RUBRICS:

- ## 1. Structural Coherence (SC-global) Check whether the geometric structure of the entities is correct, whether there are any missing or redundant limbs, and whether the overall spatial relations are consistent with physical common sense.

- - 5 (Excellent): Flawless physical logic. Objects are perfectly formed; no missing/extra parts.
- - 4 (Good): Minor structural oddities that don’t distract from the main subject.
- - 3 (Fair): Noticeable structural errors (e.g., slightly deformed limbs or merged background objects).
- - 2 (Poor): Obvious physical failures; objects are partially collapsed or mutated.
- - 1 (Very Poor): Severe structural collapse; chaotic composition; unrecognizable forms.

- ## 2. Perspective Integrity (PI) Check whether the perspective relationship between objects at different distances conforms to the principles of perspective, and whether there is any distorted perspective.

- - 5 (Excellent): Flawless perspective and geometric projection. Vanishing points and horizon lines align accurately.
- - 4 (Good): Slight perspective tilt, but geometric projection still feels natural.
- - 3 (Fair): Distorted depth; objects at different distances feel “stacked” or misaligned.

- - 2 (Poor): Severe geometric warping; architectural lines curve unnaturally or conflict.
- - 1 (Very Poor): Multiple conflicting vanishing points or warped architectural lines; total perspective failure.

- ## 3. Lighting Consistency (LC) Check whether the overall lighting has consistency with that of a natural image, and whether there are obviously artificial brightness gradients.

- - 5 (Excellent): Unified light source. Shadows, highlights, and reflections follow ray-tracing logic.
- - 4 (Good): Consistent lighting, but subtle mismatch in shadow softness or intensity.
- - 3 (Fair): Ambiguous light source. Some objects appear “self-lit” without casting shadows.
- - 2 (Poor): Contradictory lighting directions. Shadows cast in different ways for nearby objects.
- - 1 (Very Poor): Complete lighting failure; flat “sticker-like” objects with zero interaction with the environment.

- ## 4. Color Harmony (CH) Check whether the overall color transitions are smooth and natural, and whether there are issues such as blurred edges of color blocks and color banding.

- - 5 (Excellent): Natural color gamut. Smooth gradients; no banding or abnormal color noise.
- - 4 (Good): High-quality color, though slight over-saturation or other issues in small areas.
- - 3 (Fair): Visible color banding in gradients. Slight chromatic aberrations.
- - 2 (Poor): Patchy color blocks; unnatural “neon” artifacts or gray/dull patches in vibrant areas.
- - 1 (Very Poor): Severe color corruption; massive chromatic noise or broken color channels. # OUTPUT RULES (Must Strictly Follow):

- 1. You MUST follow a strict 5-point scale and provide a score as an **INTEGER from 1 to 5 only** for each dimension.
- 2. Provide the final output strictly **in a JSON object inside the <json> tag**.
- 3. The <json> block MUST contain ONLY the valid JSON object. No markdown code blocks or extra text.
- 4. Keys: “SC-global”, “PI”, “LC”, “CH” represent the scores for the 4 dimensions respectively, and “reasoning” is a concise explanation justifying the scores. Ensure the JSON property names are enclosed in double quotes and there are no trailing commas in the JSON object.

# OUTPUT FORMAT: <json> {{

“SC-global”: int, “PI”: int, “LC”: int, “CH”: int, “reasoning”: “A concise explanation justifying the four-dimensional scores.”

}} </json>

Now, please evaluate the **global-scale fidelity** of the provided image based on the above criteria and output the scores and reasoning in the specified **JSON** format.

###### Local-scale Fidelity Evaluation Prompt

# ROLE AND TASK FORMULATION: You are a professional image quality assessment (IQA) and scoring expert specializing in local fine-grained details evaluation. Your task is to carefully evaluate the **local-scale fidelity** of the provided **local patch** of an ultra-high-resolution synthetic image across 5 dimensions. To ensure high-quality evaluation, please adhere to the following guidelines:

# IMPORTANT CONTEXT:

- - **Image 1 (First):** This is the **LOCAL PATCH (target for scoring)**. Ignore incomplete objects or composition issues. Focus only on the visual quality of the visible area.
- - **Image 2 (Second):** This is the **FULL GLOBAL IMAGE (for contextual reference)**. **DO NOT** use this image for direct visual analysis or evaluation. Use this image ONLY for **understanding the original image’s theme and global context**.
- - **Patch Location:** The local patch (Image 1) corresponds to the area defined by the relative coordinates {relative_coords}

in the global image (Image 2). The relative coordinates are normalized [xmin, ymin, xmax, ymax], where [0,0] is the top-left corner and [1,1] is the bottom-right corner of the global image.

# CRITICAL SCORING RULES (Must Strictly Follow):

- 1. **Objectivity & Fairness:** Maintain an objective stance throughout the evaluation process and base your judgement on visual evidence with the same standard instead of subjective preference.
- 2. **Focus Solely on Fidelity:** Consider the image category and expected characteristics while avoiding any bias towards the content of the image. Score based on the visual quality and fidelity aspects solely.
- 3. **Local-to-Global Evaluation:** Evaluate the details in Image 1, and use Image 2 to distinguish between “intended bokeh/blur” and “accidental artifacts”.

- 4. **Coordinates Reference:** Use the rectangular bounding box only to understand the local patch’s location in the overall image context, but DO NOT directly compare the local patch to the global image for pixel-level details.
- 5. **Independence:** Evaluate each dimension independently without any halo effects.
- 6. **Rigor:** Apply strict criteria and any noticeable artifact should be reflected in the scoring. Maintain a high standard for what constitutes a “5” (Excellent).

# EVALUATION RUBRICS: Please evaluate the microscopic details and fidelity of the **Local Patch (Image 1)** across the 5 dimensions below, while using the Global Image (Image 2) and the relative coordinates {relative_coords} as reference.

- ## 1. Noise and Grain Existence (NGE) Check whether there is random high-frequency color noise and obvious color graininess in the local patch.

- - 5 (Excellent): Crystal clean and realistic cinematic grain. Zero digital noise or compression blocks. Grain looks like natural film if present.
- - 4 (Good): Slight luminance noise, barely visible at 100% zoom.
- - 3 (Fair): Noticeable noise or grain that distracts from the details, especially in shadow areas.
- - 2 (Poor): Heavy salt-and-pepper noise or distracting grain.
- - 1 (Very Poor): Image details are buried under severe noise or compression corruption.

- ## 2. Generative Artifacts (GA)

- - 5 (Excellent): No AI-specific artifacts. Details look photo-realistic and like they were captured by a high-end CMOS sensor.
- - 4 (Good): Minor generative patterns that require close inspection to find.
- - 3 (Fair): Noticeable AI “melting” or “waxy” textures where details should be sharp.
- - 2 (Poor): Hallucinated textures or “ghosting” artifacts typical of diffusion models.
- - 1 (Very Poor): Massive generative collapse; “AI-soup” textures.

- ## 3. Texture Fidelity (TF) Check whether the local patch presents plastic-like oversmoothing, and for natural objects such as wood and fabric, whether the texture has randomness rather than mechanical repetition.

- - 5 (Excellent): Tactile realism. Skin pores, fabric weaves, or surface grit are ultra-sharp and authentic.
- - 4 (Good): High detail, but slightly “soft” or over-regularized texture.
- - 3 (Fair): Texture is visible but “flat”; lacks the micro-depth of real-world surfaces.
- - 2 (Poor): Over-smoothed “plastic” look; details are smeared out.
- - 1 (Very Poor): Completely blurred or “mushy” surfaces with zero recognizable texture.

- ## 4. Micro-geometry Coherence (MGC) Check at a local scale whether the lines show unacceptable jitter or jagged edges.

- - 5 (Excellent): Perfect edge continuity. Fine lines (e.g., hair, wires) are smooth at the pixel level.
- - 4 (Good): Sharp edges, though very minor aliasing (stair-stepping) visible on diagonals.
- - 3 (Fair): Jagged edges or slight shimmering; fine lines appear broken in some places.
- - 2 (Poor): Severe aliasing; pixelated edges; micro-structures look “broken”.
- - 1 (Very Poor): Total geometric chaos at the micro-level; edges are unrecognizable.

- ## 5. Sharpness Consistency (SC-local) Check whether there are unnatural blurry patches (i.e., within the same focal plane some areas is clear enough while others are abnormally blurry).

- - 5 (Excellent): Natural optical sharpness variation consistent with depth of field. No inconsistent blur patches within the same focal plane.
- - 4 (Good): Slightly soft, but no inconsistent blur patches within the same focal plane.
- - 3 (Fair): Noticeable inconsistency in sharpness; some areas look artificially sharpened while others are blurry without a natural depth-of-field reason.
- - 2 (Poor): Obvious sharpness inconsistency; “cut-and-paste” feel where the patch looks like it was taken from a different image with different focus.
- - 1 (Very Poor): Severe sharpness failure; the patch looks like a low-quality thumbnail pasted into the global image. # OUTPUT RULES (Must Strictly Follow):

- 1. You MUST follow a strict 5-point scale and provide a score as an **INTEGER from 1 to 5 only** for each dimension.
- 2. Provide the final output strictly **in a JSON object inside the <json> tag**.
- 3. The <json> block MUST contain ONLY the valid JSON object. No markdown code blocks or extra text.
- 4. Keys: “NGE”, “GA”, “TF”, “MGC”, “SC-local” represent the scores for the 5 dimensions respectively, and “reasoning” is a concise explanation justifying the scores. Ensure the JSON property names are enclosed in double quotes and there are no trailing commas in the JSON object.

# OUTPUT FORMAT: <json> {{

“NGE”: int, “GA”: int, “TF”: int, “MGC”: int,

“SC-local”: int, “reasoning”: “A concise explanation justifying the five-dimensional scores.”

}} </json>

Now, please evaluate the **local-scale fidelity** of the provided local crop based on the above criteria and output the scores and reasoning in the specified **JSON** format.

###### Instance-centric Compliance Evaluation Prompt

# ROLE AND TASK FORMULATION:

You are a professional image quality assessment and scoring expert specializing in Text-to-Image (T2I) semantic alignment evaluation. Your task is to carefully perform a fine-grained, instance-centric assessment of the provided synthesized image based on its initial detailed long caption.

**Input Long Caption:** “{long_caption}” # METRIC DIMENSIONS AND BOUNDARIES:

Please evaluate the image across the following three distinct, hierarchical dimensions, strictly adhering to the defined criteria and boundaries:

- 1. **IEV (Instance Existence Verification):** Inspect whether all instances explicitly mentioned in the long caption are present. Focus strictly and solely on presence or absence rather than quality.
- 2. **AAA (Appearance Attribute Alignment):** For each instance that exists, assess whether its visual attributes (color, texture, material, size, shape) align with the description in the long caption. This requires detailed cross-referencing between the caption and the visual content.
- 3. **SRA (Spatial Relation Accuracy):** Evaluate whether the relative positioning (e.g., left/right, top/bottom, foreground/background) and the logical perspective between multiple instances are accurately depicted in the image. # CRITICAL SCORING RULES (Must Strictly Follow):

- 1. **Hierarchical Dependence:** **IEV** is the gatekeeper. If any critical instance is missing (IEV below 4), the corresponding AAA and SRA for the image must be penalized accordingly, as attributes and relations cannot exist without the entity.
- 2. **Detail Awareness:** Since this is a high-resolution image evaluation task, you must meticulously scan **the entire canvas**, including corners and background, to identify all mentioned instances and their micro-details.
- 3. **Strict Adherence to Explicit Constraints:** Judge the image ONLY based on what is explicitly stated in the long caption. Do not impose imaginary constraints or personal aesthetic preferences. For any visual aspects NOT mentioned (e.g., specific lighting, background nuances, or artistic style), the generation model is allowed creative autonomy. Do not penalize the model for “making choices” where the prompt is silent.
- 4. **Hallucination Penalty:** If the synthesized image contains prominent instances that are NOT mentioned in the long caption and significantly distract from the caption’s content (severe hallucination), deduct 1-2 points from **IEV**.
- 5. **No Middle Ground Bias:** Avoid giving 7/10 by default. Be decisive based on the visual evidence.
- 6. **Objectivity & Fairness:** Maintain an objective stance throughout the evaluation process and base your judgement on visual evidence with the same standard instead of subjective preference. # SCORING RUBRICS (10-Point Scale):

- ## 1. IEV (Instance Existence Verification)

- - **[9-10]:** All instances (primary and secondary) from the long caption are present and clearly identifiable in the image. No significant omissions.
- - **[7-8]:** Primary instances are present; only minor, non-essential background elements are missing or extremely unclear.
- - **[5-6]:** At least one primary instance is missing, or multiple instances are severely obscured; but the image still captures the general theme and content of the caption.
- - **[3-4]:** Multiple primary instances are missing or unrecognizable, significantly detracting from the caption’s description.
- - **[1-2]:** Total mismatch; the image fails to depict the core content of the caption.

- ## 2. AAA (Appearance Attribute Alignment)

- - **[9-10]:** Perfect alignment. All visible attributes (color, texture, material, size, shape) of the instances matches the caption exactly.
- - **[7-8]:** Most attributes are correct; slight deviations in secondary details (e.g., shade of color or minor texture mismatch) that do not significantly affect the overall perception.
- - **[5-6]:** Noticeable attribute misalignment in primary instances (e.g., wrong color or material), but the image still somewhat reflects the caption’s content.
- - **[3-4]:** Severe attribute misalignment; primary instances look totally different from the text description.
- - **[1-2]:** Complete attribute failure; objects lack any descriptive fidelity or are rendered as generic blobs.

- ## 3. SRA (Spatial Relation Accuracy)

- - **[9-10]:** All relative positions between instances and depth cues perfectly match the spatial prepositions in the caption.
- - **[7-8]:** Relative positions are correct, but there are minor errors in scale/perspective or overlapping logic that do not cause major confusion.

- - **[5-6]:** Noticeable spatial relation errors (e.g., left/right flipped, foreground/background confusion).
- - **[3-4]:** Obvious spatial relation failures; instances are positioned in a way that contradicts the caption’s logic.
- - **[1-2]:** Chaotic layout; objects are floating or positioned randomly without regard for the caption’s logic. # OUTPUT RULES (Must Strictly Follow):

- 1. You MUST follow a strict 10-point scale and provide a score as an **INTEGER from 1 to 10 only** for each dimension.
- 2. Provide the final output strictly **in a JSON object inside the <json> tag**.
- 3. The <json> block MUST contain ONLY the valid JSON object. No markdown code blocks or extra text.
- 4. Keys: “reasoning”, “IEV”, “AAA”, and “SRA”. “reasoning” is a concise explanation justifying the scores; “IEV”, “AAA”, and “SRA” represent the scores for the 3 dimensions respectively. Ensure the JSON property names are enclosed in double quotes and there are no trailing commas in the JSON object.

**Output example for reference (do not copy this exact content, just an example of the structure):** <json> {{

“reasoning”: “The cat and the bench is present but the straw hat is missing; the absence of the hat results in a deduction in IEV. The cat’s color and texture are reasonably well-aligned with the caption, though the fur appears more orange than described, leading to a moderate AAA score. The spatial relations are mostly accurate, with the cat sitting on bench and positioned on the left as specified, but the background elements are somewhat jumbled, resulting in a high but not perfect SRA score.”, “IEV”: 6, “AAA”: 8, “SRA”: 9

}} </json>

Now, please evaluate the **instance-centric compliance** of the provided image based on the above criteria and output the scores and reasoning in the specified **JSON** format.

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

###### Figure A5 One qualitative 15289×8600 sample in our PixVerve-95K dataset.

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

###### Figure A6 One qualitative 11656×8742 sample in our PixVerve-95K dataset.

