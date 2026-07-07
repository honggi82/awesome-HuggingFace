#### RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models

## arXiv:2603.25502v1[cs.CV]26Mar2026

Yufeng Yang1,2 Xianfang Zeng2,† Zhangqi Jiang2 Fukun Yin2 Jianzhuang Liu3 Wei Cheng2 Jinghong Lan2 Shiyu Liu2 Yuqi Peng3 Gang Yu2,‡ Shifeng Chen3,4,‡

1Southern University of Science and Technology 2StepFun 3Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences 4Shenzhen University of Advanced Technology

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### Project Page Models RealIR-Bench Code

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

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Before After

Before After

Before After

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Deblurring

Moiré Pattern Removal

###### Compression Restoration

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

Before After

Before After

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

###### Low-light Enhancement

###### Denoise

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Before After

Before After

Before After

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Deflare

Reflection Removal

###### Haze Removal

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Before After

Before After

Before After

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Rain Removal

Deflare

Moiré Pattern Removal

Figure 1. RealRestorer effectively restores diverse real-world image degradations, including deblurring, moiré pattern removal, compression restoration, reflection removal, hazing removal, rain removal, deflare, and low-light enhancement.

##### Abstract

Image restoration under real-world degradations is critical for downstream tasks such as autonomous driving and object detection. However, existing restoration models are often limited by the scale and distribution of their training data, resulting in poor generalization to real-world scenarios. Recently, large-scale image editing models have shown strong generalization ability in restoration tasks, especially for closed-source models like Nano Banana Pro, which can restore images while preserving consistency. Nev-

† leads this project; ‡Corresponding authors.

ertheless, achieving such performance with those large universal models requires substantial data and computational costs. To address this issue, we construct a large-scale dataset covering nine common real-world degradation types and train a state-of-the-art open-source model to narrow the gap with closed-source alternatives. Furthermore, we introduce RealIR-Bench, which contains 464 real-world degraded images and tailored evaluation metrics focusing on degradation removal and consistency preservation. Extensive experiments demonstrate our model ranks first among open-source methods, achieving state-of-the-art performance.

##### 1. Introduction

Image restoration [15, 31, 35, 37, 70] aims to recover highquality images from degraded observations and serves as a fundamental building block for downstream applications such as autonomous driving [4, 23], remote sensing [62], detection [22, 27], and 3D reconstruction [68]. However, realworld images often suffer from diverse and co-existing degradations [1, 3, 10, 11, 13, 16, 19, 21, 25, 26, 36, 40, 60, 71], including blur, rain, noise, low-light, moiré patterns, haze, compression artifacts, reflection, and flare. This complexity goes beyond the single degradation and single model paradigm.

To address this, recent all-in-one restoration methods [24, 33, 48, 69] attempt to handle multiple degradations within a unified framework. Nevertheless, they often rely on a limited set of synthetic degradation distributions, while collecting large-scale real degraded-clean pairs remains expensive and difficult. As a result, these models can generalize poorly to real-world scenarios. In parallel, large image editing models trained on massive editing datasets have recently demonstrated strong restoration capabilities [74], such as Nano Banana Pro [56] and GPT-Image-1.5 [45]. However, these models are typically trained with closed-source data and compute, which makes them hard to reproduce and limits their utility for the research community. Despite this, leveraging the strong priors learned by image editing models provides a promising path to overcome the key limitation of traditional restoration approaches.

However, conventional restoration datasets often focus on a narrow degradation distribution that is not representative of real-world conditions. Evaluation protocols that emphasize only reference-based metrics further exacerbate this issue, as they may not reflect perceptual quality, robustness across diverse degradations, or detail consistency in real scenes.

To bridge these gaps, we design a comprehensive degradation synthesis pipeline to generate high-quality training data, aiming to narrow the gap between synthetic and realworld degradations. Based on this dataset, we fine-tune an open-source image editing model RealRestorer across nine restoration tasks, and further introduce a new benchmark RealIR-Bench to evaluate restoration performance under real-world degradations.

In summary, our contributions are threefold:

- • We develop RealRestorer, an open-source real-world image restoration model that sets a new state of the art and achieves performance highly comparable to closed-source systems. We will release the model to facilitate future research in real-world restoration.
- • We propose a data generation pipeline to produce highquality restoration training data with diverse and representative degradations. This pipeline provides a valuable resource for developing more robust restoration models.
- • We develop a new benchmark, RealIR-Bench, grounded

in real-world cases, to evaluate both degradation restoration and consistency preservation. By addressing the lack of reliable evaluation protocols for real-world restoration, it enables more authentic and comprehensive assessment of restoration models.

##### 2. Related Work

###### 2.1. Single-Degradation Restoration

Single-degradation restoration methods typically focus on removing one specific type of degradation under constrained and well-defined scenarios. With the rapid development of deep learning, numerous works [5, 24, 32, 44, 73] have achieved impressive performance on individual tasks such as deblurring, haze removal, low-light enhancement, deflare, and reflection removal. These approaches often rely on carefully designed architectures and degradation-specific priors, enabling strong performance.

However, most single-degradation models are built upon task-specific assumptions, where the degradation type is predefined and relatively homogeneous, which makes models trained for a single degradation tend to generalize poorly and may even introduce secondary artifacts when encountering unseen or compound degradations.

Moreover, many existing methods are trained and evaluated primarily on synthetic datasets with simplified degradation models, which may not faithfully represent the complexity of real-world data distributions. This gap between synthetic training data and real-world testing scenarios further limits their robustness and practical applicability. Consequently, while single-degradation methods achieve strong performance on benchmark datasets, their effectiveness in real-world applications remains constrained.

###### 2.2. All-in-One Image Restoration

All-in-one approaches [7, 17, 33, 34, 38, 42, 48, 69] aim to handle multiple degradations within a unified network by balancing shared representations and task-specific components. Nevertheless, many of these methods still rely heavily on synthetic datasets with limited and overly simplified degradation patterns. Such a narrow training distribution often results in weak robustness and poor generalization to real-world degradations, where corruption characteristics are diverse, complex, and domain-dependent.

Meanwhile, large diffusion or flow-matching image editing models [12, 39, 46, 53] have recently demonstrated strong semantic priors for image enhancement and restoration. Trained on massive image–text pairs, these image editing models [30, 41, 57, 65] can leverage semantic conditioning and often generalize better to real-world data than small specialized restoration networks. Therefore, transferring and exploiting the priors of large image editing models

[Figure 98]

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

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

###### Blur

Compression

###### Moiré

Low-light

###### Noise

###### Flare

###### Reflection

###### Haze

###### Rain

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

###### T

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|
|[Figure 131]|
|[Figure 132]|

|[Figure 133]|
|---|
|[Figure 134]|

|[Figure 135]|
|---|

|[Figure 136]|
|---|
|[Figure 137]|
|[Figure 138]|

|[Figure 139]|
|---|
|[Figure 140]|

Clean Images

Clean Images

Clean Images

|[Figure 141]|
|---|

|[Figure 142]|
|---|

[Figure 143]

Clean Images

Clean Images

|[Figure 144]|
|---|

|[Figure 145]|
|---|

Clean Images

Clean Images

Flare Patterns

Haze Patterns

[Figure 146]

Clean Video Clips

Clean images

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Moiré Patterns

Retinexformer

|[Figure 151]|
|---|

Reflection

Depth Prediction

[Figure 152]

SAM2 Segment Granularity Noise Web-Style Degradation

Web-Style Degradation

[Figure 153]

Light Adjustment

Depth Prediction

VLMs Filter

Light Adjustment

[Figure 154]

Real Moiré Image

[Figure 155]

|[Figure 156]|
|---|
|[Figure 157]|
|[Figure 158]|

[Figure 159]

|[Figure 160]|
|---|
|[Figure 161]|
|[Figure 162]|

Rain streaks

[Figure 163]

Flare Blender Random Flip

Haze Synthesize

SynNet

|[Figure 164]|
|---|
|[Figure 165]|

Unidemoire´

[Figure 166]

|[Figure 167]|
|---|
| |
|[Figure 168]|

[Figure 169]

|[Figure 170]|
|---|

|[Figure 171]|
|---|

Haze Blender

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

Degraded Images

Degraded Images

Degraded Images

Degraded Images

Degraded Images

Degraded Images

Degraded Images

Degraded Images

Degraded Images

- Figure 2. Overview of our large-scale Synthetic Degradation Data pipeline. We construct nine representative degradation types, including blur, compression artifacts, moiré patterns, low-light, noise, flare, reflection, haze, and rain. Compared with previous synthetic-only pipelines, our upgraded framework incorporates granular noise modeling, segment-aware perturbations, and web-style degradation processes, significantly narrowing the gap between synthetic and real-world distributions. This comprehensive pipeline enables more robust and generalizable restoration learning.

provides a promising direction for building restoration systems with stronger real-world generalization.

tion distributions, and they are often not robust enough for large-scale training that aims at strong generalization.

Motivated by this observation, we develop a high-quality and realistic degradation synthesis pipeline covering nine major degradations and use it to fine-tune open-source image editing models for robust real-world restoration while maintaining strong content consistency. Furthermore, to evaluate real-world restoration performance in the absence of clean references, we curate a benchmark of 464 real images spanning nine single-degradation categories, and propose new evaluation metrics that measure both degradation removal ability and consistency with the input content. Based on the proposed dataset and metrics, our fine-tuned model achieves state-of-the-art performance among open-source methods and is competitive with closed-source systems, while qualitative results further demonstrate strong generalization to real-world degradations.

To address this limitation, we develop a new dataset collection pipeline that produces more realistic degradation patterns while keeping the paired clean images highly consistent with their degraded counterparts.

In general, we adopt two main ways to obtain high-quality paired data for image restoration of nine tasks:

Synthetic Degradation Data: Start from clean images and synthesize degradations. This approach is highly scalable as long as sufficient clean images can be collected from the internet. However, even with increasingly sophisticated degradation synthesis, it remains challenging to fully capture the diversity and complexity of real-world degradations. Nevertheless, such synthetic data can still be valuable, as it provides a convenient way to transfer general image editing priors to image restoration models and helps them acquire foundational restoration knowledge. We leverage several powerful open-source models to support the synthetic data generation process, including SAM-2 [52], and MiDaS [51]. These models are used to filter unsuitable samples and provide essential structural and geometric information required for realistic degradation synthesis, such as semantic masks and depth cues.

##### 3. RealRestorer

###### 3.1. Data Construction

Existing image restoration datasets [17, 34] often rely on a single degradation model to synthesize degraded images and use a fixed composition strategy to explicitly disentangle degradation features for representation learning. These modeling approaches are effective for specific degradation settings. However, in real-world scenarios, degradations are far more complex and diverse. Simple synthetic degradation models are usually insufficient to approximate real degrada-

In our pipeline, to ensure high data quality, we employ the Vision-Language Models (VLMs) and quality assessment models [43] to filter out low-quality or unsuitable images like watermarked images. After forming pairs, we further

examine the degree of degradation alignment between the degraded and restored images to ensure that the degradation patterns are learnable from the paired data. Specifically, the synthetic pairing data construction is illustrated as follows.

Blur: The motion blur dataset is primarily synthesized using temporal averaging over video clips to simulate realistic motion trajectories. Both the target and source images are filtered to ensure consistent blur patterns. In addition, webstyle degradation, including common blur operations, such

- as Gaussian blur and standard motion blur, is incorporated to better approximate real-world motion blur characteristics.

Compression Artifacts: We simulate compression artifacts using JPEG compression and image resizing to approximate common web compression effects. In addition to standard JPEG degradation, we also incorporate web-style compression processes to better reflect the wide range of compression artifacts found in online images.

Moiré Patterns: Following UniDemoiré [67], we generate 3,000 moiré patterns at multiple scales and randomly fuse one to three patterns into clean images. This strategy substantially improves the diversity and generalization capability of the model for moiré pattern removal.

Low-Light: We simulate low-light conditions by applying brightness attenuation and gamma correction to reduce pixel intensity. Moreover, we train a separate model [5] using paired datasets such as LOL [66] and LSRW [18], reversing the low-exposure and high-exposure image pairs. This trained model is then applied to clean images to better mimic realistic low-light distributions.

Noise: We adopt web-style degradation as the primary noise synthesis pipeline. Compared with the degradation strategy used in Real-ESRGAN [61], we further introduce granular noise for web images. Additionally, we incorporate segment-aware noise, which significantly improves performance on real-world denoising tasks.

Flare: We collect more than 3,000 glare patterns and adapt them to clean images for realistic blending. In addition, random horizontal and vertical flipping is applied to further enhance the diversity of the generated data pairs.

Reflection: For reflection degradation synthesis, we collect two sources of clean images. The first source mainly consists of portrait images, which are treated as transmission layers. The second source contains diverse scenes with human faces, which are used as reflection layers. To increase the diversity of the paired data, we randomly swap a few portions of the image pairs, using human portraits as reflection layers instead of transmission layers. The overall synthesis pipeline follows SynNet [64].

Haze: We synthesize hazy images based on the classic atmospheric scattering model by estimating depth from clean images and generating fog accordingly [20]. To better simulate real haze, we collect nearly 200 haze patterns and randomly blend them with the synthesized haze, making the

results closer to real-world haze distributions.

Rain: To synthesize realistic rain degradation, we not only add rain streaks but also incorporate splashes and simulate physical effects such as perspective distortion and droplet sputtering. Furthermore, we collect 200 real rain patterns and randomly blend them into clean images to enhance diversity and realism. Besides, we also adopt the rain category from the FoundIR dataset [34], which contains about 70K paired samples.

Real-World Degradation Data: Collect real degraded images and generate corresponding clean images by removing degradations using high performance restoration models. Compared with synthetic pairing, this approach is more likely to preserve the true degradation statistics of real-world data, enabling restoration models trained on such pairs to generalize better to real scenarios. To bridge the gap between synthetic and real-world degradations, we collect real degraded images from the web and pair them with high-quality references.

During web data collection, we first employ the CLIP model [49] to filter images based on degradation-related semantic cues. While this approach effectively removes a portion of irrelevant samples, it still introduces noisy cases, such as watermarked images or visually similar but nondegraded content. To further refine the dataset, we apply a watermark detection filter and leverage Qwen3-VL-8BInstruct [58] to assess and verify the degree of degradation. After generating clean references using high-performance image generation models, we further examine the consistency of the paired data by employing low-level metrics to detect potential content shifts. A subset of the filtered pairs is then manually reviewed to ensure that the degradation type and severity are properly aligned between degraded inputs and their corresponding clean references. These curated realworld degradation samples enable the model to better adapt its parameters to realistic data distributions. Such adaptation helps the model converge more effectively toward real-world scenarios, consistent with prior findings in large-scale generative modeling [47, 57, 59].

Additional details and qualitative demonstrations are provided in Appendix A.

###### 3.2. Method and Training Strategy

We fine-tune the base model Step1X-Edit [41] built on a large Diffusion in Transformer (DiT) backbone [46], which is effective for generation. It is equipped with QwenVL [2] as a text encoder that injects high-level semantic extraction into the DiT denoising pathway. Inside the diffusion network, a dual-stream design is used to jointly process semantic information together with noise and the conditional input image. The reference image and output image are both encoded into latent space by Flux-VAE [29]. During training, all the components are initialized from the officially released

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

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Degraded Image

Nano Banana Pro

GPT-Image1.5

FLUX.1Kontext-dev

Qwen-ImageEdit-2511

LongCatImage-Edit

[Figure 190]

[Figure 191]

[Figure 192]

Ours

Seedream 4.5

Step1X-Edit

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

HazeNoiseFlareCompressionBlurReflectionRain

Bl ur

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

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

tterns𝐏𝐏𝐏𝐏

[Figure 222]

Low-lightMoiré

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

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

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

- Figure 3. Comparison with state-of-the-art image editing models across nine real-world degradations, including blur, compression artifacts, moiré patterns, low-light, noise, flare, reflection, haze, and rain. We compare our method with large-scale image editing models, such as Seedream 4.5, Nano Banana Pro, GPT-Image-1.5, Step1X-Edit, FLUX.1-Kontext-dev, Qwen-Image-Edit-2511, and LongCat-Image-Edit.

checkpoint of Step1X-Edit, and we freeze the Flux-VAE and text encoder, only fine-tune the DiT. Starting from the original image editing model, we fine-tune on nine restoration tasks in two stages: a Transfer-training stage for large-scale restoration transfer and a Supervised Fine-tuning stage for constraining the manifold of the final model distribution.

Transfer Training Stage: In the first stage, we use synthetic paired data to transfer high-level knowledge and priors from image editing to image restoration. Since we initialize from a pretrained backbone, we eschew progressive resolution schedules [41] for training. Instead, we adopt a high-

resolution setting of 1024×1024 throughout the entire training process. The learning rate is kept constant at 1e−5, and the global batch size is set to 16. Since most of our training data has a resolution higher than 1024×1024, no additional upsampling is required, which helps preserve fine-grained details and maintain training stability. For each degradation of nine, we adopt single and fixed prompts, which are also the same for the second training stage. For multi-task learning, we adopt an average sampling ratio across all tasks during training. After several steps of transfer training, RealRestorer begins to exhibit signs of knowledge transfer from

high-level image editing tasks to image restoration tasks, which is insufficient in the base model.

Although RealRestorer gradually acquires the basic capability to handle simple degradation patterns across all nine tasks, its ability to distinguish and model diverse real-world degradation patterns remains limited. In particular, the model still struggles to capture fine-grained details in complex scenarios. In some cases, noticeable artifacts are present, and the model fails to respond effectively to certain types of degradations. This observation motivates us to introduce a second training stage aimed at improving generalization and restoration quality under real-world degradation scenarios. Moreover, we observe that different task types exhibit distinct learning dynamics and require varying training durations. Therefore, we select a balanced trade-off checkpoint

- at the end of the first stage to preserve both generation capability and cross-task generalization. Supervised Fine-tuning Stage: For the second training stage, we incorporate real-world degradation data to further enhance restoration fidelity and improve generalization under real-world degradation scenarios [57, 59, 65]. Compared with the first stage, this stage emphasizes adaptation to complex and authentic degradation patterns. We adopt a cosine annealing learning rate schedule, where the learning rate is gradually decayed to zero, using the same initial learning rate as in the first stage. This smooth decay strategy stabilizes the transition between training stages and encourages the model to progressively adapt to the real-to-clean paired data. By gradually reducing the optimization step size, the model is guided to converge toward a parameter configuration that better aligns with the distribution represented by the highquality real-world dataset, thereby improving restoration fidelity and robustness under realistic degradations.

Importantly, instead of completely replacing synthetic data, we adopt a Progressively-Mixed training strategy, which retains a small proportion of synthetic paired samples during the second stage. RealRestorer is first exposed to diverse synthetic degradations to build broad generalization, and then gradually adapted to real-world degradations while maintaining exposure to synthetic distributions. Such a hybrid curriculum helps prevent overfitting to specific real degradation patterns and preserves cross-task robustness. More detailed discussions and quantitative analyses of this training strategy are provided in the ablation study. In addition, we introduce a web-style degradation data augmentation strategy throughout the training process to enhance robustness to images collected from the web. Such images typically suffer from low visual quality, compression artifacts, and other degradations. By simulating these practical degradation patterns during training, the model becomes better equipped to handle real-world inputs and produce better restoration results under challenging conditions.

Throughout the two-stage training process, we select the

intermediate checkpoint with the best generalization capability to maintain a balanced performance across multiple tasks and ensure strong overall performance of the final model. All our experiments are conducted on 8 NVIDIA H800 GPUs. More implementation details can be found in Appendix B.

##### 4. Benchmark and Evaluation

###### 4.1. RealIR-Bench

Traditional image restoration benchmarks primarily focus on single-degradation tasks with synthetic corruptions or limited degradation patterns, which makes them insufficient for evaluating model performance in real-world applications [14, 17, 34, 50]. Such benchmarks often fail to capture the complexity, diversity, and unpredictability of degradations encountered in practical scenarios.

To properly evaluate restoration performance under realworld degradations, we construct a new benchmark composed entirely of internet-sourced, naturally degraded images. The proposed benchmark spans nine common restoration tasks and covers a wide range of degradation types frequently observed in real-world photography, including blur, rain, noise, low-light, moiré patterns, haze, compression artifacts, reflection, and flare, which collectively represent the most common forms of real-world image degradation. To preserve the authentic real-world degradation distribution, we directly curate images from web sources rather than synthesizing degradations. We further conduct manual filtering to ensure both quality control and diversity across degradation types, scene content, and severity levels. This human-in-the-loop curation process helps preserve realistic degradation characteristics while avoiding overly biased, repetitive, or low-quality samples. By combining automatic collection with manual verification, we ensure that the benchmark better reflects the complexity and diversity of degradations encountered in real-world scenarios, rather than artifacts introduced by purely synthetic construction.

In total, the benchmark contains 464 non-reference degraded images for testing. To ensure a fair and consistent evaluation protocol, we adopt a fixed enhancement instruction for all samples. This design minimizes the influence of instruction variation and allows the evaluation to focus more directly on a model’s restoration capability and its ability to preserve image consistency.

The collected images cover a variety of common realworld degradation scenarios, including complex and mixed degradations that are often challenging for restoration models. As a result, the benchmark provides a practical and demanding testbed for assessing real-world restoration performance. More details about the benchmark construction and data statistics are provided in Appendix C.

- Table 1. Quantitative comparison on the Rain Removal, Deblurring, Low-light Enhancement, Haze Removal, and Reflection Removal tasks. We compared state-of-the-art (SOTA) image editing models. For each task, we reported LPS (↓), RS (↑), and FS (↑). The best result is marked in bold, and underline indicates the second-best result. The best and second-best open-source results are highlighted with yellow and blue backgrounds, respectively.

Method Open-source

Rain Removal Deblurring Low-light Enhancement Haze Removal Reflection Removal

LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ Image Editing Methods

Nano Banana Pro [56] No 0.429 2.063 0.236 0.326 1.068 0.144 0.467 0.720 0.077 0.492 1.920 0.195 0.358 1.368 0.176 GPT–Image–1.5 [45] No 0.535 2.120 0.197 0.532 1.667 0.156 0.523 1.048 0.100 0.558 1.840 0.163 0.468 2.320 0.247 Seedream 4.5 [55] No 0.438 1.500 0.169 0.254 0.255 0.038 0.423 0.600 0.069 0.418 1.140 0.133 0.291 1.156 0.164 LongCat–Image–Edit [57] Yes 0.381 1.302 0.161 0.200 0.000 0.000 0.158 0.120 0.020 0.236 0.000 0.000 0.254 1.060 0.158 Qwen–Image–Edit–2511 [65] Yes 0.435 1.736 0.196 0.170 0.240 0.040 0.122 0.080 0.014 0.337 0.060 0.008 0.333 1.820 0.243 FLUX.1-Kontext-dev [30] Yes 0.244 0.673 0.102 0.090 0.104 0.019 0.108 0.160 0.029 0.058 0.020 0.004 0.048 0.127 0.024 Step1X–Edit [41] Yes 0.282 0.019 0.003 0.321 0.906 0.123 0.306 0.340 0.047 0.194 0.190 0.031 0.247 0.080 0.012

RealRestorer (ours) Yes 0.371 1.076 0.135 0.582 1.900 0.159 0.597 1.360 0.110 0.339 0.680 0.090 0.290 1.620 0.230

- Table 2. Quantitative comparison on the Deflare, Demoiré, Denoise, and Compression-restoration tasks. The average results of all 9 tasks are reported in the last column. We compared state-of-the-art (SOTA) image editing models. For each task, we reported LPIPS (↓), RS (↑), and FS (↑). The best result is marked in bold, and underline indicates the second-best result. The best and second-best open-source results are highlighted with yellow and blue backgrounds, respectively.

Deflare Moiré Patterns Removal Denoise Compression Restoration Avg Total (9)

Method Open-source

LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ LPS↓ RS↑ FS↑ Image Editing Methods

Nano Banana Pro [56] No 0.214 1.222 0.192 0.562 1.560 0.137 0.386 0.712 0.087 0.483 1.122 0.116 0.413 1.306 0.153 GPT–Image–1.5 [45] No 0.336 1.415 0.188 0.646 1.633 0.116 0.496 0.993 0.100 0.633 1.167 0.086 0.525 1.578 0.150 Seedream 4.5 [54] No 0.225 1.104 0.171 0.548 1.600 0.145 0.387 0.770 0.094 0.529 1.136 0.107 0.390 1.029 0.125 LongCat–Image–Edit [57] Yes 0.241 1.717 0.261 0.420 1.200 0.139 0.350 0.471 0.061 0.188 0.083 0.014 0.270 0.661 0.097 Qwen–Image–Edit–2511 [65] Yes 0.222 1.660 0.258 0.595 1.660 0.135 0.429 0.824 0.094 0.242 0.300 0.046 0.320 0.931 0.127 FLUX.1-Kontext-dev [30] Yes 0.064 0.264 0.049 0.348 0.540 0.070 0.429 0.628 0.072 0.429 0.628 0.072 0.202 0.349 0.056 Step1X–Edit [41] Yes 0.173 0.000 0.000 0.654 0.410 0.028 0.409 0.098 0.012 0.344 0.383 0.050 0.325 0.270 0.036

RealRestorer (ours) Yes 0.239 1.623 0.247 0.563 1.620 0.142 0.478 0.863 0.090 0.547 1.067 0.097 0.445 1.312 0.146

###### 4.2. Experimental Results on RealIR-Bench

Based on RealIR-Bench, we evaluate a diverse set of large image editing models’ ability in image restoration towards the real world, covering state-of-the-art closed-source systems such as GPT-Image-1.5 [45], Nano Banana Pro [56], Seeddream 4.5 [55], as well as strong open models including Qwen-Image-Edit-2511 [65], FLUX.1-Kontext-dev [30], LongCat-Image-Edit [57] and Step1X-Edit [41]. We provide nine major degradation tasks for evaluation: deblurring, rain removal, denoise, low-light enhancement, moiré patterns removal, haze removal, compression restoration, reflection removal, and deflare, with task-specific English instructions for each model to remove the corresponding degradation.

Unlike full-reference metrics such as PSNR and SSIM [63], which require paired clean reference images for evaluation, RealIR-Bench is built entirely from nonreference images collected from diverse real-world scenarios. In these cases, obtaining perfectly aligned clean targets is infeasible, making conventional full-reference evaluation protocols unsuitable. Therefore, instead of relying on pixelwise fidelity measures, we adopt a non-reference evaluation framework to assess how well image editing models handle

real-world degradations, with particular emphasis on both degradation removal capability and consistency preservation.

To characterize both restoration effectiveness and the trade-off with content fidelity, we report two metrics: Restoration Score (RS), LPIPS (LPS) [72]. We convert the LPIPS distance into a perceptual similarity score so that higher values indicate better perceptual consistency. After normalizing both RS and LPS to the same scale, the Final Score (FS) is defined as:

###### FS = 0.2(1 − LPS)RS (1)

FS jointly reflects restoration improvement and content preservation, and poor performance in either aspect will directly lead to a lower overall score.

Inspired by non-reference evaluation methods such as VIEScore [28], we leverage VLMs to assess Restoration Score. Specifically, we employ Qwen3-VL-8B-Instruct [58] to rate the degradation severity of both degraded images and restored images on a scale from 0 to 5, where 5 indicates no visible degradation, and 0 corresponds to the most severe degradation. The Restoration Score (RS) is defined as the improvement in degradation level after restoration. In other words, it is computed as the difference between the degra-

- Table 3. Quantitative comparison on the FoundIR dataset across various real-world degradations. We report PSNR (↑) and SSIM (↑). The best results are highlighted in bold, and the second-best results are underlined.

Blur Rain Raindrops Noise Low-light Haze Compression Average

Method

PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑

Nano Banana Pro [56] 20.16 0.68 21.55 0.65 23.29 0.66 24.78 0.89 15.73 0.73 17.63 0.50 19.48 0.58 20.37 0.67 GPT–Image–1.5 [45] 13.25 0.41 12.52 0.34 13.89 0.31 14.15 0.68 11.00 0.55 12.39 0.28 13.06 0.39 12.89 0.42 Seedream 4.5 [55] 17.94 0.61 17.61 0.57 18.01 0.58 15.57 0.78 13.64 0.68 15.65 0.44 14.29 0.54 16.10 0.60 LongCat–Image–Edit [57] 18.53 0.65 17.09 0.52 18.22 0.48 19.77 0.81 15.24 0.70 10.52 0.36 15.64 0.47 16.43 0.57 Qwen–Image–Edit–2511 [65] 15.60 0.55 14.68 0.46 15.19 0.42 19.73 0.81 7.13 0.17 7.20 0.34 18.29 0.61 13.98 0.48 FLUX.1-Kontext-dev [30] 10.73 0.41 10.48 0.35 11.25 0.31 15.10 0.69 10.34 0.53 10.74 0.29 10.74 0.34 11.34 0.42 Step1X–Edit [41] 16.38 0.63 19.91 0.63 19.42 0.58 27.18 0.91 15.72 0.70 14.20 0.45 22.81 0.75 19.37 0.66

RealRestorer (ours) 18.99 0.59 23.72 0.71 23.64 0.71 28.15 0.90 17.59 0.77 17.66 0.56 20.40 0.66 21.45 0.70

dation score of the restored image and that of the degraded image. A higher RS indicates greater perceived restoration improvement according to the VLM evaluator.

For consistency evaluation, we aim to measure the model’s ability to preserve the original scene structure, semantic content, and fine-grained details throughout the restoration process. To this end, we employ LPIPS as the evaluation metric to measure the perceptual similarity between the restored images and the degraded inputs. Unlike traditional pixel-level metrics, LPIPS is more sensitive to perceptually relevant discrepancies, including structural deviations and semantic inconsistencies, making it particularly suitable for assessing content preservation before and after restoration.

Table 1 and Table 2 demonstrate the strong restoration capability of RealRestorer. It consistently outperforms existing open-source image editing models and achieves performance comparable to leading closed-source systems. Across all nine tasks, RealRestorer achieves the best performance on deblurring and low-light enhancement and ranks second on moiré pattern removal. Among open-source models, it ranks first on five tasks and second on two, and remains highly competitive on the remaining tasks. Overall, it ranks first among open-source models and third overall, narrowing the gap with Nano Banana Pro (first place) to only 0.007 points and surpassing Qwen-Image-Edit-2511 (the second-best opensource model) by 0.019 points. These results indicate that RealRestorer not only effectively removes real-world degradations but also maintains high consistency and fidelity in the restored outputs. As an open-source model, RealRestorer significantly narrows the performance gap between open-source and closed-source systems, while exhibiting strong generalization ability across diverse real-world scenarios. Figure 3 presents qualitative comparisons, further demonstrating that RealRestorer produces visually cleaner and more consistent restoration results compared to other state-of-the-art image editing methods. On real-world degraded images from RealIR-Bench, our model shows strong performance across diverse scenarios. In particular, when handling complex and irregular real-world degradations such as blur and flare, RealRestorer remains highly competitive with leading closed-source models, achieving comparable

visual quality and structural fidelity.

###### 4.3. Extra Benchmark Evaluation and Zero-shot Generalization

To further evaluate restoration performance on a traditional all-in-one benchmark, we additionally evaluate the same set of image editing models on the FoundIR test set [34]. FoundIR contains 20 real-world degradation settings with paired clean references, including 7 isolated degradations (blur, rain, noise, low-light, raindrops, haze, and compression artifacts) and 13 coupled degradation combinations. We report results on the 7 isolated degradation subsets, which also overlap with RealIR-Bench, resulting in a total of 750 paired image pairs with an average resolution of 2514 × 1516. For the editing prompt, we use the same prompt set as RealIR-Bench.

Table 3 shows that RealRestorer achieves strong restoration performance on these 7 tasks, obtaining the best PSNR and SSIM on 5 out of 7 degradations. Notably, all image editing models tend to achieve relatively low reference-based metrics, which is consistent with the generative nature of these models that may introduce perceptually plausible yet non-identical details. Benefiting from high-quality synthetic degradation data, RealRestorer achieves a better trade-off while improving content consistency. We further evaluate the generalization ability of RealRestorer via zero-shot experiments on real-world restoration scenarios, including snow removal and old photo restoration. RealRestorer also generalizes well to unseen restoration tasks. Although it is only finetuned on a limited set of degradation types, it can still handle other unseen tasks by benefiting from the restoration priors learned during training, while retaining part of the original model’s general image editing capability. More qualitative results, evaluations on additional public benchmarks, and detailed comparisons are provided in Appendix D, together with further visualizations and analysis.

###### 4.4. Ablation and User Studies We conduct an ablation study on the training data and training stages to examine the necessity of the proposed two-stage training strategy. Specifically, we first train the model on the Synthetic Degradation Data (about 1M samples). As shown in Figure 14, the model acquires basic restoration

[Figure 283]

Figure 4. Model performance with varying training steps and training data on RealIR-Bench. The blue line shows transfer training on synthetic degradation data, where the model gradually acquires basic restoration capability. The blue dashed line indicates performance degradation after prolonged training due to the limited diversity of synthetic data. The purple line represents supervised fine-tuning with real-world degradation data, which rapidly improves performance and generalization. The purple dashed segment indicates the onset of overfitting after around 2.5K steps.

capability and reaches a peak FS of 0.122 during the first stage, but still lacks sufficient generalization ability and fails on some rare cases. Moreover, its performance drops significantly after 2.5K steps, which we attribute to the limited diversity of the synthetic data. We further investigate the impact of the Real-World Degradation Data (about 100K samples) in the second stage. After entering this stage, the model quickly surpasses the peak score achieved in the transfer training stage and continues to improve its generalization ability, eventually achieving strong performance at around 2.5K steps. However, beyond this point, the model begins to overfit the Real-World Degradation Data, which motivates us to adopt early stopping. Overall, the two-stage training strategy, together with the combination of synthetic and realworld data, leads to a final model with strong restoration performance and better consistency preservation.

Furthermore, we conduct an ablation study on the Progressively-Mixed training strategy. Without this component, the final FS score decreases by 0.004 points under the same training configuration, confirming its effectiveness. And from a qualitative perspective, the Progressive-Mixed strategy also leads to better preservation of structural consistency and content fidelity, resulting in more visually stable and coherent restoration results. Additional ablation results and analyses are provided in the supplementary materials.

We conduct a user study to evaluate both the reliability of the proposed RealIR-Bench metrics and the perceptual performance of our model from a human perspective. Specifically, we recruit 32 participants to rank 3,200 groups of generated images produced by five high-performing models according to two criteria: restoration quality and content con-

sistency. Specifically, Nano Banana Pro achieves the highest first-ranking rate of 32.02%, followed by GPT-Image-1.5 with 23.83%, while our method attains 21.54%. This trend is consistent with the average overall scores reported in Table 2. Moreover, we perform a statistical analysis of the proposed metrics and observe a moderate alignment with human judgments across all evaluation measures (p < 0.01). Further details on the study design, ranking protocol, and statistical analysis are provided in the Apeendix F.

##### 5. Limitations and Discussion

Although RealRestorer demonstrates strong generalization across both seen and unseen restoration tasks, we still observe several limitations. First, since the base image editing model relies on a 28-step denoising process, its computational cost remains substantially higher than that of smallerscale models, which is a common limitation of large-scale image editing models. Second, in cases with strong semantic and physical ambiguity, such as mirror selfies, the model may fail to distinguish true scene content from undesired reflections, a challenge that is also common in other image editing methods. Third, RealRestorer still struggles with extremely severe degradations where reliable pixel evidence is largely missing, and may fail to preserve physically consistent structures such as water reflections.

##### 6. Conclusion

In this paper, we introduce RealRestorer, a robust opensource image editing model for complex real-world image restoration. To reduce the synthetic-to-real domain gap, we propose a comprehensive data generation pipeline and a twostage progressively mixed training strategy that combines synthetic and real-to-clean pairs. We further present RealIRBench, a non-reference benchmark with authentic degraded images and a VLM-based evaluation framework for realworld restoration. Extensive experiments on many evaluation sets demonstrate that RealRestorer achieves open-source state-of-the-art performance across nine restoration tasks, with results highly comparable to leading closed-source commercial systems, and exhibits strong zero-shot generalization to unseen degradations. We will release our model, data synthesis pipeline, and benchmark to support future research in real-world image restoration.

##### References

- [1] SM Al-Salem, G Abraham, OA Al-Qabandi, and AM Dashti. Investigating the effect of accelerated weathering on the mechanical and physical properties of high content plastic solid waste (psw) blends with virgin linear low density polyethylene (lldpe). Polymer Testing, 46:116–121, 2015. 2

- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 4

- [3] Arnold O Benz. Flare observations. Living reviews in solar physics, 14(1):2, 2017. 2

- [4] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621– 11631, 2020. 2

- [5] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinex-based transformer for low-light image enhancement. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12504–12513,

2023. 2, 4

- [6] Ke Cheng, Yifan Zhang, Xiangyu He, Weihan Chen, Jian Cheng, and Hanqing Lu. Skeleton-based action recognition with shift graph convolutional network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 183– 192, 2020. 14

- [7] Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, and Fahad Shahbaz Khan. Adair: Adaptive all-in-one image restoration via frequency mining and modulation. In 13th International Conference on Learning Representations, ICLR 2025, pages 57335–57356. International Conference on Learning Representations, ICLR, 2025. 2

- [8] Peng Dai, Xin Yu, Lan Ma, Baoheng Zhang, Jia Li, Wenbo Li, Jiajun Shen, and Xiaojuan Qi. Video demoireing with relation-based temporal consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 15

- [9] Yuekun Dai, Chongyi Li, Shangchen Zhou, Ruicheng Feng, Yihang Luo, and Chen Change Loy. Flare7k++: Mixing synthetic and real datasets for nighttime flare removal and beyond. 2023. 15
- [10] Niranjan Damera-Venkata, Thomas D Kite, Wilson S Geisler, Brian L Evans, and Alan C Bovik. Image quality assessment based on a degradation model. IEEE

- transactions on image processing, 9(4):636–650, 2000. 2
- [11] Alain Dufaux, Laurent Besacier, Michael Ansorge, and Fausto Pellandini. Automatic sound detection and recognition for noisy environment. In 2000 10th European Signal Processing Conference, pages 1–4. IEEE, 2000. 2

- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 2

- [13] Jan Flusser, Sajad Farokhi, Cyril Höschl, Tomáš Suk, Barbara Zitova, and Matteo Pedone. Recognition of images degraded by gaussian blur. IEEE transactions on Image Processing, 25(2):790–806, 2015. 2

- [14] Qiyuan Guan, Qianfeng Yang, Xiang Chen, Tianyu Song, Guiyue Jin, and Jiyu Jin. Weatherbench: A real-world benchmark dataset for all-in-one adverse weather image restoration. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12607–12613, 2025. 6

- [15] Bahadir Gunturk and Xin Li. Image restoration. CRC Press, 2018. 2

- [16] Xiaojie Guo and Qiming Hu. Low-light image enhancement via breaking down the darkness. International Journal of Computer Vision, 131(1):48–66, 2023. 2

- [17] Yu Guo, Yuan Gao, Yuxu Lu, Huilin Zhu, Ryan Wen Liu, and Shengfeng He. Onerestore: A universal restoration framework for composite degradation. In European conference on computer vision, pages 255–

272. Springer, 2024. 2, 3, 6

- [18] Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. R2rnet: Lowlight image enhancement via real-low to real-normal network. Journal of Visual Communication and Image Representation, 90:103712, 2023. 4

- [19] Feng He, Yongjian Zhou, Zefang Ye, Sang-Hyeok Cho, Jihoon Jeong, Xianghai Meng, and Yaguo Wang. Moiré patterns in 2d materials: a review. ACS nano, 15(4):5944–5958, 2021. 2

- [20] Kaiming He, Jian Sun, and Xiaoou Tang. Single image haze removal using dark channel prior. IEEE transactions on pattern analysis and machine intelligence, 33(12):2341–2353, 2010. 4

- [21] Klaus Hermann. Periodic overlayers and moiré patterns: theoretical studies of geometric properties. Journal of Physics: Condensed Matter, 24(31):314210,

2012. 2

- [22] Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In Proceedings of the IEEE conference on computer

- vision and pattern recognition, pages 3588–3597, 2018. 2
- [23] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17853–17862, 2023. 2

- [24] Yi Huang, Jiancheng Huang, Jianzhuang Liu, Mingfu Yan, Yu Dong, Jiaxi Lv, Chaoqi Chen, and Shifeng Chen. Wavedm: Wavelet-based diffusion models for image restoration. IEEE Transactions on Multimedia, 26:7058–7073, 2024. 2

- [25] Yukun Huang, Zheng-Jun Zha, Xueyang Fu, Richang Hong, and Liang Li. Real-world person reidentification via degradation invariance learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14084– 14094, 2020. 2

- [26] Kui Jiang, Zhongyuan Wang, Peng Yi, Chen Chen, Zheng Wang, Xiao Wang, Junjun Jiang, and ChiaWen Lin. Rain-free and residue hand-in-hand: A progressive coupled network for real-time image deraining. IEEE Transactions on Image Processing, 30:7404– 7418, 2021. 2

- [27] KJ Joseph, Salman Khan, Fahad Shahbaz Khan, and Vineeth N Balasubramanian. Towards open world object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5830–5840, 2021. 2

- [28] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12268–12290, 2024. 7

- [29] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 4
- [30] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025. 2, 7, 8

- [31] Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, and Timo Aila. Noise2noise: Learning image restoration without clean data. arXiv preprint arXiv:1803.04189, 2018. 2

- [32] Boyi Li, Wenqi Ren, Dengpan Fu, Dacheng Tao, Dan Feng, Wenjun Zeng, and Zhangyang Wang. Benchmarking single-image dehazing and beyond. IEEE transactions on image processing, 28(1):492–

505, 2018. 2

- [33] Boyun Li, Xiao Liu, Peng Hu, Zhongqin Wu, Jiancheng Lv, and Xi Peng. All-in-one image restoration for unknown corruption. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17452–17462, 2022. 2

- [34] Hao Li, Xiang Chen, Jiangxin Dong, Jinhui Tang, and Jinshan Pan. Foundir: Unleashing million-scale training data to advance foundation models for image restoration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12626–12636, 2025. 2, 3, 4, 6, 8

- [35] Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1775–1787, 2023. 2

- [36] Jie Liang, Hui Zeng, and Lei Zhang. Efficient and degradation-adaptive network for real-world image super-resolution. In European Conference on Computer Vision, pages 574–591. Springer, 2022. 2

- [37] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833–1844, 2021. 2

- [38] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Yu Qiao, Wanli Ouyang, and Chao Dong. Diffbir: Toward blind image restoration with generative diffusion prior. In European conference on computer vision, pages 430–

448. Springer, 2024. 2

- [39] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2

- [40] Jiaying Liu, Dejia Xu, Wenhan Yang, Minhao Fan, and Haofeng Huang. Benchmarking low-light image enhancement and beyond. International Journal of Computer Vision, 129(4):1153–1184, 2021. 2

- [41] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 2, 4, 5, 7, 8

- [42] Ziwei Luo, Fredrik K Gustafsson, Zheng Zhao, Jens Sjölund, and Thomas B Schön. Controlling visionlanguage models for universal image restoration. arXiv preprint arXiv:2310.01018, 3(8), 2023. 2

- [43] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection,

2023. 3, 14

- [44] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In CVPR, July 2017. 2

- [45] OpenAI. Introducing 4o image generation, 2025. 2, 7, 8
- [46] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 2, 4

- [47] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 4

- [48] Vaishnav Potlapalli, Syed Waqas Zamir, Salman H Khan, and Fahad Shahbaz Khan. Promptir: Prompting for all-in-one image restoration. Advances in Neural Information Processing Systems, 36:71275– 71293, 2023. 2

- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 4, 14

- [50] Sudarshan Rajagopalan, Nithin Gopalakrishnan Nair, Jay N. Paranjape, and Vishal M. Patel. Gendeg: Diffusion-based degradation synthesis for generalizable all-in-one image restoration, 2024. 6
- [51] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(3), 2022. 3

- [52] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3

- [53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684– 10695, 2022. 2

- [54] Team Seedream, :, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, Xiaowen Jian, Huafeng Kuang, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, Wei Liu, Yanzuo Lu, Zhengxiong Luo, Tongtong Ou, Guang Shi, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng

- Wang, Rui Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Wenxu Wu, Yonghui Wu, Xin Xia, Xuefeng Xiao, Shuang Xu, Xin Yan, Ceyuan Yang, Jianchao Yang, Zhonghua Zhai, Chenlin Zhang, Heng Zhang, Qi Zhang, Xinyu Zhang, Yuwei Zhang, Shijia Zhao, Wenliang Zhao, and Wenjia Zhu. Seedream 4.0: Toward next-generation multimodal image generation, 2025. 7
- [55] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 7, 8

- [56] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 7, 8

- [57] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, and Jie Hu. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025. 2, 4, 6, 7, 8

- [58] Qwen Team. Qwen3 technical report, 2025. 4, 7, 14
- [59] Z-Image Team. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 4, 6

- [60] Li Tong and JR White. Photo-oxidation of thermoplastics in bending and in uniaxial compression. Polymer degradation and stability, 53(3):381–396, 1996. 2

- [61] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In International Conference on Computer Vision Workshops (ICCVW). 4

- [62] Yan Wang, Ling Yang, Xinzhan Liu, and Pengfei Yan. An improved semantic segmentation algorithm for high-resolution remote sensing images based on deeplabv3+. Scientific reports, 14(1):9716, 2024. 2

- [63] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 7

- [64] Qiang Wen, Yinjie Tan, Jing Qin, Wenxi Liu, Guoqiang Han, and Shengfeng He. Single image reflection removal beyond linearity. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 4

- [65] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai,

- Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 2, 6, 7, 8
- [66] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. IEEE Transactions on Image Processing, 30:2072–2086, 2021. 4

- [67] Zemin Yang, Yujing Sun, Xidong Peng, Siu Ming Yiu, and Yuexin Ma. Unidemoiré: Towards universal image demoiréing with data generation and synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9354–9362, 2025. 4

- [68] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pages 767– 783, 2018. 2

- [69] Eduard Zamfir, Zongwei Wu, Nancy Mehta, Yuedong Tan, Danda Pani Paudel, Yulun Zhang, and Radu Timofte. Complexity experts are task-discriminative learners for any image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12753–12763, 2025. 2

- [70] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-stage progressive image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14821– 14831, 2021. 2

- [71] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4791–4800, 2021. 2

- [72] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7, 15

- [73] Hao Zhao, Mingjia Li, Qiming Hu, and Xiaojie Guo. Reversible decoupling network for single image reflection removal. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26430– 26439, 2025. 2

- [74] Jialong Zuo, Haoyou Deng, Hanyu Zhou, Jiaxin Zhu,

Yicheng Zhang, Yiwei Zhang, Yongxin Yan, Kaixing Huang, Weisen Chen, Yongtai Deng, et al. Is nano banana pro a low-level vision all-rounder? a comprehensive evaluation on 14 tasks and 40 datasets. arXiv preprint arXiv:2512.15110, 2025. 2

### Appendix

##### A. Data Construction Details

- A.1. Synthetic Degradation Data

For the Synthetic Degradation Data, we collect the clean image from the internet and synthesize the nine major degradation patterns: blur, rain, noise, low-light, moiré patterns, haze, compression artifacts, reflection, and flare. We will release the pipeline.

- A.2. Real-World Degradation Data

For the real-world degradation data, we collect clean images from high-quality open-source image websites, including Pexels and Pinterest, covering six types of degradation: blur, rain, low light, haze, reflection, and flare. These degradation types often exhibit a substantial gap between real-world degradations and synthesized patterns.

- Table 4. Semantic prompts used for CLIP-based degradation filtering. Degradation Type CLIP Text Prompt

Flare a photo with lens flare, bright streaks of light Haze a hazy photo, foggy atmosphere, low contrast Rain a rainy photo with rain streaks or raindrops Low-light a dark photo, underexposed, low illumination Blur a blurry photo with motion blur or out-of-focus regions Reflection a photo with glass or mirror-like reflection artifacts

To construct a high-quality real-world degradation dataset, we first employ the CLIP model [49] to filter images based on degradation-related semantic cues, as shown in Table 4. Second, we apply a watermark detection filter [43] together with Qwen3-VL-8B-Instruct [58] to remove watermarked images and images with insufficient degradation, thereby retaining samples suitable for obtaining paired clean data. After selecting appropriate editing models to generate a large amount of raw paired data, additional filtering is required to remove failure cases. Specifically, we use Qwen3-VL-8BInstruct to estimate the degradation scores of both the clean and degraded images, and then filter pairs with inconsistent or insufficient score differences, while a skeleton-shift-based method [6] is adopted to remove pixel pairs with alignment errors. Finally, after strictly filtering the raw dataset, we further performed human curation on the remaining subset to construct the final dataset. Three trained human experts participated in the annotation and verification process.

A.3. Training Dataset statistics

- Table 5 summarizes the statistics of the two components of our training data across different degradation types. Visual

examples of the data are presented in Figure 5.

Table 5. Statistics of our dataset across different degradation types. The table reports the number of training image pairs from Synthetic Degradation Data and Real-World Degradation Data. The total column indicates the combined number of samples for each degradation category.

Degradation Synthetic Real Total Rain 84,968 43,415 128,383 Blur 1,014,229 13,458 1,027,687 Low-light 5,000 7,005 12,005 Hazy 103,971 8,147 112,118 Reflection 68,227 7,604 75,831 Flare 59,520 7,956 67,476 Moire 99,085 0 99,085 Noise 64,492 0 64,492 Compression 68,000 0 68,000 Total 1,567,492 87,585 1,655,077

##### B. Implementation Details

During training, we treat the DiT blocks as trainable components, while freezing both the VAE and text encoders. In the Transfer Training Stage, we train the model using the Synthetic Degradation Dataset, which covers nine degradation types. To balance the learning across tasks, we adopt an average sampling strategy over all nine degradation categories. In this stage, the bucket resolution is fixed at 1024 × 1024, and the global batch size is set to 16.

After approximately 500 training steps, the model begins to transfer knowledge from high-level editing capabilities to low-level restoration tasks. However, it still struggles to handle more complex degradations, often producing artifacts in the restored results. To address this limitation, we introduce a Supervised Fine-Tuning Stage. In this stage, we adopt a Progressively-Mixed training strategy, combining Real-World Degradation Data with a small portion of Synthetic Degradation Data. This strategy helps constrain the model toward the data manifold of real-world restoration tasks while retaining the robustness learned from synthetic degradations.

Additionally, we freeze the first one-fourth of the SingleStreamBlocks in the DiT architecture to stabilize training. The global batch size is increased to 32, and a cosine annealing learning rate schedule is applied, where the learning rate gradually decays to zero while maintaining the same initial learning rate as in the first stage. This stage lasts for 1.5K training steps, allowing the model to converge to a balanced and generalizable checkpoint. All experiments are conducted on NVIDIA H800 GPUs, and the entire training process takes approximately one day on 8 H800 GPUs. More detailed training hyperparameters are provided in Table 6.

Table 6. Training hyperparameters used in the two training stages.

Hyperparameters Transfer Training Stage Supervised Fine-Tuning Stage Learning rate 1 × 10−5 1 × 10−5 → 0 LR scheduler Constant Cosine Weight decay 0.0 0.01 Gradient norm clip 1.0 1.0 Optimizer AdamW (β1 = 0.9,β2 = 0.95,ϵ = 1e−8) Warm-up steps 100 100 Frozen layers None First 1/4 SingleStreamBlocks Training steps 500 1500 Training samples 1.5M 80K Resolution 1024 × 1024 1024 × 1024 Synthetic : Real data ratio 1 : 0 2 : 8

##### C. RealIR-Bench and Metrics Details

RealIR-Bench covers diverse real-world degradation scenarios, including blur, rain, noise, low-light, moiré patterns, haze, compression artifacts, reflection, and flare. Example cases from the benchmark are shown in Figure 8.

Specifically, we evaluate models using two complementary metrics: Restoration Score (RS), which reflects the perceptual restoration quality, and LPIPS [72], which measures perceptual similarity to assess consistency after restoration.

###### C.1. Restoration Score

The Restoration Score (RS) is designed to evaluate the ability of a model to remove degradations without explicitly considering content consistency. Inspired by VIEScore, we employ Qwen3-VL-8B-Instruct as a vision-language evaluator to assess the degradation severity of both degraded images and restored images. The Restoration Score (RS) is then defined as the improvement in the degradation level after restoration. The detailed system instruction for Qwen3-VL-8B-Instruct is shown in the Figure 9.

##### D. More Qualitative Results and Benchmark Evaluation

We present additional visualization results in the following pages to further demonstrate the strong restoration capability of our model compared with other image editing models on RealIR-Bench across nine degradation types.

Furthermore, to evaluate RealRestorer on public benchmarks for deflare, reflection removal, and demoiré, we conduct additional experiments on several widely used datasets. For deflare evaluation, we use the Flare-R subset from the Flare7K++ dataset [9], which contains 100 paired real-world flare images. The Flare7K++ dataset combines synthetic and real flare data and provides a comprehensive benchmark for nighttime flare removal tasks.

For moiré pattern removalevaluation, we adopt the

UHDM test set [8], which contains 500 paired real moiré images captured at ultra-high resolution. For reflection removal, we evaluate on the SIR²+ benchmark, which includes three subsets: SolidObjectDataset, PostcardDataset, and WildScene, containing 50, 50, and 101 paired images respectively. These datasets include real-world scenes with complex reflective patterns and are widely used for evaluating single-image reflection removal methods. The quantitative comparison results are presented in Table 7. The results show that RealRestorer achieves the second-best performance in PSNR and the third-best performance in SSIM on average across the five evaluation datasets.

##### E. Ablation Study Details

Besides comparing the proposed two-stage training strategy composed of the synthetic degradation transfer training stage and the real-world degradation SFT stage, we further analyze an alternative setting where the model is trained using only real-world degradation data. For a fair comparison, we train the model for the same number of iterations as in the synthetic transfer training stage. At the peak point, the model trained only on real-world data tends to overfit the degradation patterns, which harms the structural consistency of the restored images. This often results in artifacts such as object deformation, body shifting, and unrealistic enhancement. These observations further confirm the importance of the proposed two-stage training strategy.

Specifically, at 2.5K training steps, the model trained with only synthetic degradation data still shows limited ability to handle complex real-world degradations, while the model trained solely on real-world degradation data can partially restore degradations but often fails to preserve content consistency. The model produces overly enhanced results, such as removing natural light sources, as shown in Figure 14. In contrast, our two-stage training strategy effectively balances restoration capability and structural consistency, leading to more stable and generalizable performance.

- Table 7. Quantitative comparison on extra public benchmarks for real-world image restoration tasks. We report PSNR (↑) and SSIM (↑). The evaluation is conducted on multiple datasets including Flare-R from Flare7K++ for deflare, UHDM for moiré pattern removalremoval, and the SIR2+ reflection removal benchmark with three subsets: PostcardDataset, SolidObjectDataset, and WildScene. Flare-R contains real captured flare images, while UHDM provides ultra-high-definition paired moiré images. The SIR2+ subsets represent different reflection scenarios with diverse scene contents. The best results are highlighted in bold, and the second-best results are underlined.

Method

Flare Moiré Refl (Postcard) Refl (Solid) Refl (Wildscene) Average

PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑

Nano Banana Pro 25.28 0.889 18.86 0.726 20.11 0.821 24.53 0.880 22.27 0.835 22.21 0.830 GPT–Image–1.5 16.86 0.526 11.84 0.549 13.51 0.465 13.97 0.451 14.59 0.527 14.15 0.504 Seedream 4.5 23.88 0.831 15.30 0.643 21.38 0.807 22.16 0.809 21.45 0.783 20.83 0.775 LongCat–Image–Edit 22.57 0.809 15.79 0.670 20.28 0.706 17.12 0.632 21.28 0.805 19.41 0.724 Qwen–Image–Edit–2511 22.45 0.869 16.07 0.673 21.07 0.839 20.17 0.864 23.77 0.896 20.71 0.828 FLUX.1-Kontext-dev 23.27 0.859 10.36 0.516 13.43 0.428 11.60 0.374 14.62 0.520 14.66 0.539 Step1X–Edit 18.74 0.772 14.31 0.583 19.29 0.841 18.56 0.783 21.12 0.864 18.40 0.769

Ours 22.05 0.745 17.62 0.743 22.67 0.904 20.35 0.797 21.72 0.826 20.88 0.803

F. User Study Details

Our user study evaluates the results of the five bestperforming models on RealIR-Bench. All the 32 participants receive a brief tutorial beforehand to ensure they understand the task and the evaluation criteria. The interface used in the user study is illustrated in Figure 15. To further analyze the reliability of the proposed metric, we compute the Kendall’s τb, Spearman Rank Correlation Coefficient (SRCC), and Pearson Linear Correlation Coefficient (PLCC) between the metric score (FS) and human judgments. These correlation measures are widely used to evaluate the consistency between automatic metrics and human evaluation. The results demonstrate that the proposed metric achieves moderate statistical alignment with human judgments (p < 0.01) across all evaluation settings, as shown in Table 8.

- Table 8. Consistency evaluation between RS metric and subjective human perception.

###### Metric Correlation Coefficient p-value

Kendall’s τb 0.2493 2.96 × 10−124 SRCC 0.3010 4.62 × 10−128 PLCC 0.2919 3.21 × 10−120

Degraded Image Clean Image Degraded Image Clean Image Degraded Image Clean Image

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Blur Compression Moiré

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Low-light Haze Rain

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Reflection Noise Flare

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

###### Blur

Low-light

Haze

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Rain

Reflection

Flare

[Figure 341]

[Figure 342]

[Figure 343]

- Figure 5. Examples from our training dataset containing both synthetic and real-world degradation pairs. The upper rows with gray labels show synthesized degradations generated by our pipeline, while the bottom rows highlighted with orange labels correspond to real-world degraded images paired with clean references.

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

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

[Figure 395]

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

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

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

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

###### Figure 6. Additional qualitative results of RealRestorer under real-world degradations. Please zoom in for better visualization of details.

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

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

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

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

[Figure 554]

[Figure 555]

###### Figure 7. Additional qualitative results of RealRestorer under real-world degradations. Please zoom in for better visualization of details.

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Blur] Please deblur the image and make it sharper. 请将图像去模糊，变得更清晰。

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Compression] Please restore the image clarity and artifacts. 请修复图像清晰度和伪影。

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Moiré] Please remove the moiré patterns from the image. 请将图像中的摩尔条纹去除。

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Low-light] Please restore this low-quality image, recovering its normal brightness and clarity. 请修复这张低质量图像，恢复其正常的亮度和清晰度。

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Noise] Please remove noise from the image. 请修复请去除图像中的噪声。

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Flare] Please remove the lens flare and glare from the image. 请去除图像中的光晕和炫光。

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Haze] Please dehaze the image. 请将图像去雾。

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Rain] Please remove the rain from the image and restore its clarity. 请去除图像中的雨水并恢复图像清晰度。

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Reflection] Please remove the reflection from the image. 请移除图像中的反光。

Figure 8. Examples from our RealIR-Bench. Each degradation category is evaluated using a fixed bilingual prompt.

#### Instruction for Degradation Evaluation

- • Degradation Evaluation: You will evaluate whether an image exhibits the degradation type {task} and determine its severity level. The assessment should focus exclusively on the specified degradation type, while ignoring unrelated image-quality issues, semantic content, and aesthetic preference. Your goal is to provide a consistent and objective judgment of how strongly the target degradation affects the image.
- • Evaluation Procedure:

- 1. Divide the image into several local regions (e.g., a 3×3 grid) to ensure a systematic inspection.
- 2. Examine each region carefully and identify whether the degradation type {task} is present.
- 3. Estimate both the severity of the degradation and the proportion of the image area that is affected.
- 4. Summarize the regional observations into a single overall score that best reflects the image-level degradation.
- 5. If the degradation is spatially uneven, place greater emphasis on the regions that are more strongly affected.
- 6. For borderline cases, choose the nearest score level based on the dominant visual impression.
- 7. Do not output any intermediate reasoning, explanation, or analysis.

- • Scoring Scale (1–5): Use the following scale to measure the severity of the target degradation. Higher scores indicate cleaner images with less visible degradation, while lower scores indicate stronger and more widespread corruption.

- – 5 = No {task}; the image is essentially clean with no noticeable degradation.
- – 4 = Mild degradation; only a small portion of the image is affected (≤ 20% area), and the overall visual quality remains largely intact.
- – 3 = Moderate degradation; the degradation is clearly visible and affects a noticeable part of the image (20–50% area).
- – 2 = Severe degradation; the corruption is strong and influences a large portion of the image (50–80% area).
- – 1 = Extreme degradation; the degradation dominates most of the image (> 80% area) and seriously harms visual quality.

- • Output Format: Return only in the format “Degradation Score: <1–5>”.

Figure 9. System instruction used for degradation evaluation.

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Blur] Please deblur the image and make it sharper.

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Compression] Please restore the image clarity and artifacts.

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Flare] Please remove the lens flare and glare from the image.

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Moiré] Please remove the moiré patterns from the image.

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

###### [Haze] Please dehaze the image.

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Low-light] Please restore this low-quality image, recovering its normal brightness and clarity.

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Noise] Please remove noise from the image.

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Rain] Please remove the rain from the image and restore its clarity.

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Blur] Please deblur the image and make it sharper.

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Compression] Please restore the image clarity and artifacts.

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Flare] Please remove the lens flare and glare from the image.

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Moiré] Please remove the moiré patterns from the image.

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

###### [Haze] Please dehaze the image.

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Low-light] Please restore this low-quality image, recovering its normal brightness and clarity.

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Noise] Please remove noise from the image.

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Rain] Please remove the rain from the image and restore its clarity.

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Blur] Please deblur the image and make it sharper.

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Compression] Please restore the image clarity and artifacts.

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Flare] Please remove the lens flare and glare from the image.

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Moiré] Please remove the moiré patterns from the image.

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

###### [Haze] Please dehaze the image.

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Low-light] Please restore this low-quality image, recovering its normal brightness and clarity.

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Noise] Please remove noise from the image.

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Rain] Please remove the rain from the image and restore its clarity.

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Blur] Please deblur the image and make it sharper.

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Compression] Please restore the image clarity and artifacts.

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Flare] Please remove the lens flare and glare from the image.

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Moiré] Please remove the moiré patterns from the image.

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

###### [Haze] Please dehaze the image.

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Low-light] Please restore this low-quality image, recovering its normal brightness and clarity.

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Noise] Please remove noise from the image.

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Rain] Please remove the rain from the image and restore its clarity.

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

Synthetic Degradation Data Training

Real-World Degradation Data Training

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

Degraded Image

Ours

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

- Figure 14. Qualitative comparison of different training strategies. Models trained only with synthetic degradation data show limited ability to restore complex real-world degradations. In contrast, models trained solely on real-world degradation data tend to overfit, which may harm structural consistency. Our two-stage training strategy effectively balances restoration capability and content consistency.

###### Instruction: Please restore this low-quality image, recovering its normal brightness and clarity.

Scoring Dimensions Enhancement Capability:

Does the output image show clear enhancement compared to the low-quality input image?

Scene Consistency: Does the enhanced image preserve the original scene content, such as characters, objects, and overall layout?

Overall Quality: Holistic visual quality and usability of the result (clarity, naturalness, and absence of artifacts).

Scoring Rules (1–5, 1 = Poor, 5 = Excellent) Enhancement Capability:

1: No meaningful enhancement or the image quality becomes worse. 2–3: Partial improvement, but noticeable degradation or incomplete enhancement remains. 4–5: Clear and effective enhancement with substantial improvement over the input image.

Scene Consistency: 1: Major inconsistency; the scene or key elements are significantly altered or replaced. 2–3: Generally similar but with noticeable inconsistencies (e.g., identity changes, object shifts, or missing elements). 4–5: The scene and key elements remain consistent with the input image.

Overall Quality: 1: Very poor visual quality or severe artifacts. 2–3: Acceptable but with visible flaws or unnatural effects. 4–5: High visual quality, natural appearance, and minimal artifacts.

General Scoring Principles Absolute Quality: Focus on the actual visual quality of the result. Do not intentionally spread scores just to create differences between methods. Use the Full Scale: Please utilize the entire 1–5 rating range when appropriate. Weakest-Link Principle: If a critical failure occurs in any key aspect (e.g., the degradation is removed but the scene or identity changes completely), the final score should be significantly reduced.

A generated image B generated image

Input image Image

[Figure 941]

[Figure 942]

[Figure 943]

C generated image D generated image E generated image

[Figure 944]

[Figure 945]

[Figure 946]

# A score: B score: C score: D score: E score:

- Figure 15. User study interface used to evaluate the restoration results. Participants are presented with one degraded input image and five restored results generated by different models and are asked to rate them based on restoration quality and consistency.

