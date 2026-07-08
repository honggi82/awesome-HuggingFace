## EdgeFusion: On-Device Text-to-Image Generation

# arXiv:2404.11925v1[cs.LG]18Apr2024

Thibault Castells† Hyoung-Kyu Song† Tairen Piao† Shinkook Choi† Bo-Kyeong Kim† Hanyoung Yim‡ Changgwun Lee‡ Jae Gon Kim‡ Tae-Ho Kim†

{thibault, hyoungkyu.song, tairen.piao, shinkook.choi, bokyeong.kim, thkim}@nota.ai {hanyoung.yim, cgwun.lee, jgon.kim}@samsung.com

†Nota Inc., Korea ‡Samsung Electronics, Korea

### Abstract

The intensive computational burden of Stable Diffusion (SD) for text-to-image generation poses a significant hurdle for its practical application. To tackle this challenge, recent research focuses on methods to reduce sampling steps, such as Latent Consistency Model (LCM), and on employing architectural optimizations, including pruning and knowledge distillation. Diverging from existing approaches, we uniquely start with a compact SD variant, BK-SDM. We observe that directly applying LCM to BK-SDM with commonly used crawled datasets yields unsatisfactory results. It leads us to develop two strategies: (1) leveraging highquality image-text pairs from leading generative models and (2) designing an advanced distillation process tailored for LCM. Through our thorough exploration of quantization, profiling, and on-device deployment, we achieve rapid generation of photo-realistic, text-aligned images in just two steps, with latency under one second on resource-limited edge devices.

### 1. Introduction

Stable Diffusion (SD) [23] models have emerged as powerful tools in text-to-image (T2I) synthesis, acclaimed for their ability to transform text into high-quality images. These powerful generative models have widespread applications across various domains, from creative arts to practical solutions. However, their real-world deployment is hindered by immense computational and memory requirements due to complex iterative processes and large parameter sizes, making them difficult to deploy on resourceconstrained devices like Neural Processing Units (NPUs).

To tackle this, existing studies focus on three directions: (1) architectural reduction [2, 4, 9, 14, 27, 33] to reduce the model size, (2) few-step inference [14, 17, 18, 25] to accelerate the inference speed, and (3) leveraging AI-generated data [3, 9] to improve the training efficiency. In particular, employing high-quality image-text pairs from advanced

SD-v1.4 (25 steps)

BK-SDM-Tiny (25 steps)

EdgeFusion (4 steps) w/o improved data

EdgeFusion (4 steps) w/ improved data

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

“a cute fox, hyper detailed, cinematic light, ultra high res, best shadow, RAW”

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

“a child sporting a yellow raincoat, vibrant against the gray of a rainy day, high resolution, 4k”

Figure 1. T2I generation results. When trained with improved data, our EdgeFusion can produce high-quality images from challenging prompts in just a few denoising steps.

generative models [8, 10, 12] overcomes limitations of realworld datasets’ scarcity and biases while improving models’ ability to produce visually compelling text-aligned images. However, the holistic integration of these approaches remains underexplored, especially regarding the practical deployment of compact SD models (see Tab. 1).

Adapting SD models in memory-constrained environments (e.g., NPUs), necessitates a targeted and optimized approach. This involves addressing the unique computational limitations and memory constraints of NPUs through a comprehensive strategy that includes architectural modifications and specialized optimizations specifically designed for these operational constraints [6, 15].

We propose EdgeFusion, a method that advances the field by optimizing SD for execution on NPUs. EdgeFusion enhances the model efficiency by employing Blockremoved Knowledge-distilled SDM (BK-SDM) [13], a foundational effort towards lightweight SD, and significantly improve its generation performance with superior image-text pairs from synthetic datasets. Furthermore, we refine the step distillation process of Latent Consistency Model (LCM) [17] through empirical practices, achieving

|Model|Arch. Reduc.†<br><br>AI-Gen. Data†<br><br>Few-Step Gen.†<br><br>Edge Deploy.†|Param.†<br><br>|
|---|---|---|
|Choi et al. [4] SnapFusion [14] LCM [17] SSD-1B [9] Pixart-alpha [3] Pixart-delta [2] UFOGen [32] SD-Turbo [27] MobileDiff. [33] EdgeFusion (Ours)<br><br>|✓ ✗ ✗(20) ✓a ✓ ✗ ✗(8) ✓b<br><br>✗ ✗ ✓(2, 4) ✗<br><br>✓ ✓ ✗(25) ✗<br><br>✗ ✓ ✗(20) ✗<br><br>✓ - ✓(1, 4) ✗<br><br>✗ ✗ ✓(1) ✗<br><br><br>✓ ✗ ✓(1) ✗<br><br>✓ ✗ ✓(1, 8) ✓c ✓ ✓ ✓(1, 2, 4) ✓d<br><br>|1B 1.3B+ 1.3B+ 0.6B 0.9B 0.86B<br><br>0.4B<br><br>0.5B<br>|

†: Architectural Reduction; AI-Generated Data; Few-Step Generation (#Steps); Edge Deployment; #Parameters. a: Galaxy S23. b: iPhone14 Pro. c: iPhone15 Pro. d: Samsung Exynos 2400.

Table 1. Comparison with existing work on efficient T2I synthesis.

high-quality few-step inference. For deployment, EdgeFusion adopts model-level tiling, quantization, and graph optimization to generate a 512×512 image under one second on Samsung Exynos NPU [26].

### 2. Proposed approach

#### 2.1. Advanced distillation for LCM

To accelerate the generation speed, we collaboratively integrate multiple strategies. By adopting the compressed U-Net design of BK-SDM-Tiny [13], we overcome the computational bottleneck inherent in the inference processes of diffusion-based T2I models. Additionally, by utilizing a recent step distillation method of LCM, plausible text-aligned images can be generated in as few as 2 or 4 denoising steps. However, we observed that the vanilla application of LCM’s step reduction to the publicly accessible BK-SDM-Tiny checkpoint [1, 13] (see Fig. 2(a)) resulted in unsatisfactory outcomes.

To address this issue, we present our advanced distillation approach (see Fig. 2(b)). In Kim et al. [13], the BKSDM-Tiny was trained via feature-level knowledge distillation [11, 24], using the original SD-v1.4’s U-Net [22, 23] as its teacher. We empirically find that leveraging an advanced teacher model, namely Realistic Vision V5.1 [30], enhances the generation quality. This process leads to an improved student model, dubbed BK-SDM-Adv-Tiny.

Starting with BK-SDM-Adv-Tiny as the initialization, we fine-tune it with the scheduler in LCM to develop our fewer-step model, called EdgeFusion. Our experiments reveal notable improvements when utilizing the original large model as the teacher, instead of a model identically structured with the student. In both stages, the quality of the training data is crucial to achieving superior T2I diffusion models, which will be described further in the next section.

#### 2.2. Improved data quality

This section describes how we improve training data quality (dataset size and further details in Supplementary).

- (a) Vanilla Approach
- (b) Advanced Distillation (Ours)

[Figure 9]

: frozen weights

|U-Net Original SD-v1.4<br><br>[Figure 10]|
|---|

|U-Net BK-SDM-Tiny<br><br>[Figure 11]|
|---|

TeacherStudent

TeacherStudent

|LAION<br><br>Data|
|---|

|LAION<br><br>Data|
|---|

Knowledge Distillation Loss

Consistency Distillation Loss

|U-Net BK-SDM-Tiny|
|---|

|U-Net (few steps) BK-LCM-Tiny|
|---|

gradient

gradient

[Figure 12]

Initialization

|U-Net Realistic_Vision_V5.1<br><br>[Figure 13]|
|---|

|U-Net Realistic_Vision_V5.1<br><br>[Figure 14]|
|---|

TeacherStudent

TeacherStudent

Improved Data

Improved Data

Knowledge

Consistency

|U-Net<br><br>BK-SDM-Adv-Tiny|
|---|

|U-Net (few steps)<br><br>EdgeFusion|
|---|

Distillation

Distillation

Loss

Loss

gradient gradient Initialization

[Figure 15]

Phase 1. Improved Student Initialization

Phase 2. Improved LCM Training

Figure 2. A compact SD with step reduction. (a) Vanilla application of LCM: we initialize BK-LCM-Tiny with the weight from BK-SDM-Tiny and train with distillation to reduce sampling steps. (b) Our approach: improving the initialization of the LCM’s student with a better teacher is beneficial. Moreover, in the LCM training phase, employing the original teacher enhances performance. Leveraging high-quality data is crucial in both phases.

Initial training with LAION subset (L-orig). When the model was trained using the LAION data subset [28, 29] used in Kim et al. [13], the results were sub-optimal. We recognize a significant challenge with the LAION dataset: the prevalence of low-quality images and irrelevant captions. We hypothesize that such issues could greatly impede the training processes since capturing text-image relationships from noisy data is challenging and might demand considerably more extended training periods. It highlights the critical need to enhance the quality of the data.

Data preprocessing (L-pp). To address the limitations of the LAION dataset, we perform data deduplication [20], remove samples smaller than 300 pixels, and utilize a model to optimize image cropping by selecting the best option among center crop and two crops from the image’s longer side extremes, reducing the occurrence of cropped subjects. While these techniques yield better results, the gains in data quality are likely offset by the 44% reduction in dataset size. Synthetic caption generation (L-pps). To enhance the caption informativeness, we use Sphinx [16], a large language model (LLM), to generate detailed, relevant captions for our preprocessed images, improving text-image correspondence. Caption generation is performed after image cropping to ensure prompts accurately reflect visual content. While synthetic captions have positive effects, they do not address image quality concerns from the initial data collection stage.

Fully synthetic data generation (synt). To address persistent image quality issues in our dataset, we synthetically generate both images and corresponding prompts, providing higher control over training sample quality. Using GPT4 [19] for prompt generation and SDXL [21] for image synthesis, we create a wide range of image-text pairs with notable improvements in image quality and text-image alignment. This fully synthetic approach allows explicit control

over generated image attributes like gender, ethnicity, age, and other demographic factors, creating a more diverse, inclusive dataset that mitigates potential biases and ensures comprehensive representation. We generate 62.3k synthetic images using this approach. Including these high-quality, diverse, semantically aligned samples in our training set substantially improves our model’s performance.

Manual data curation (synt-cur). Despite significant improvements in image quality through fully synthetic data generation, some issues persist in the dataset, such as visual artifacts and suboptimal text-image alignment, particularly for complex or unusual prompts. To determine if the substantial human effort of manually curating synthetic data is justified, we curate 29.9k synthetic images by correcting inexact prompts for improved image description accuracy and removing images containing artifacts.

#### 2.3. Deployment on NPU

Model-level tiling (MLT). Given the limited memory capacity of NPUs, dividing the model into smaller segments, called MLT, is crucial for efficient inference. FlashAttention [5] introduces an algorithm capable of diminishing the data transfers between High Bandwidth Memory (HBM) and on-chip Static Random-Access Memory (SRAM), thereby enhancing performance within GPU environments. Motivated by this, we develop an MLT method to leverage heterogeneous computing resources, including tensor and vector engines, in edge devices. The primary objective of our tiling method is to reduce Dynamic RandomAccess Memory (DRAM) access in handling Transformer blocks [7] (e.g., due to large input dimensions of the Softmax layer). The secondary aim is to maximize SRAM utilization, as the smallest tile size could incur significant communication overhead [31] between computing engines, necessitating an optimal tile count. We develop a method to determine the suitable number of tiles, based on calculations for the required size and utilization of SRAM.

Quantization. To efficiently deploy EdgeFusion to Samsung Exynos NPU, we apply mixed-precision post-training quantization to the model. We apply FP16 precision for weights and activations to encoder and decoder modules, and apply INT8 weight quantization, INT16 activation quantization to the U-Net module. Sec. 4.2 provides detailed quantization performance.

### 3. Experimental setup

All training stages are conducted on a single NVIDIA A100 GPU. For BK-SDM model training, a synthetic dataset is integrated alongside the existing training dataset while adhering to the established training recipe derived from prior studies. Likewise, the training process of applying LCM scheduler follows previous methods, ensuring

fidelity to the established methodologies.

For deployment on NPU devices, we fuse operators using a kernel fusion method [5] and apply the MLT method. Experiments are conducted on Exynos 2400 [26], which features a 17K MAC NPU (2-GNPU + 2-SNPU) and DSP, to evaluate latency on edge devices.

Additionally, to assess the potential of the speed enhancements on cloud service, we provide a benchmark on a single NVIDIA A100 GPU.

### 4. Results

#### 4.1. Main results

Tab. 2 shows that despite being finetuned with just 29.9k samples (synt-cur), BK-SDM-Adv-Tiny outperforms BKSDM-Tiny by 5.28 (+17.4%) IS and 0.021 CLIP scores. The lower FID score is likely due to the synthetic data distribution differing from COCO and Realistic Vision v5.1 having a worse FID than SD-v1.4. However, our human evaluation (Tab. 4) showing a strong preference for BK-SDMAdv-Tiny (60.2% win rate) suggests that while comparing to COCO is common practice, assuming COCO as the ideal data distribution may not align well with human preference. Our final model, EdgeFusion, LCM-distilled from BK-SDM-Adv-Tiny on 182.2k samples (L-pps & synt), has close performance to BK-SDM-Tiny and even outperforms it on CLIP (Tabs. 2 and 3). Two steps results and more output examples can be found in the supplementary.

KD teacher Dataset IS ↑ FID ↓ CLIP ↑ SD-v1.4 [23] L-orig a, [13] 30.39 17.43 0.266

L-orig b 32.11 21.66 0.278 L-pp 32.20 20.98 0.278 L-pps 32.32 21.65 0.283 L-pps & synt 34.82 23.37 0.288 L-pps & synt-cur 34.54 23.34 0.287 synt 35.11 25.14 0.288 synt-cur c 35.67 26.15 0.287

Realistic Vision v5.1 [30]

Termed a: w/o finetuning; b: w/o improved data; c: BK-SDM-Adv-Tiny.

Table 2. Results of the BK-SDM-Tiny architecture trained on different datasets. Evaluated on COCO dataset with 25 steps.

#### 4.2. Ablation studies

Significance of high-quality training data. Tabs. 2 and 3 show that synthetic prompts consistently improve textimage alignment (CLIP), but also affect the image quality: for LCM training, re-captioning LAION data improves both IS (+1.99) and FID (−2.49). Interestingly, the qualityquantity trade-off favors quality when finetuning BK-SDMTiny, with best results using synt-cur data only, while favoring quantity for LCM training, where combining L-pps and synt yields optimal results. Tab. 3 demonstrates the results of our advanced distillation pipeline (Fig. 2) with and

without improved data, obtaining +3.27 IS, −5.69 FID, and +0.018 CLIP. Human evaluation in Tab. 4 confirms these observations with 62.8% win rate for EdgeFusion.

Pretrained student Dataset IS ↑ FID ↓ CLIP ↑

BK-SDM-Tiny [13] L-pps & synt a 26.11 22.06 0.262 L-orig L-orig b 25.51 25.03 0.254 L-pp L-pp 25.26 24.48 0.254 L-pps L-pps 27.25 21.99 0.266 L-pps & synt L-pps & synt 28.41 19.33 0.276 L-pps & synt-cur L-pps & synt-cur 28.11 20.07 0.273 synt synt 28.23 19.91 0.272 synt-cur synt-cur 27.87 20.21 0.270 synt-cur L-pps & synt c 28.78 19.34 0.272

Termed a: w/o finetuning; b: w/o improved data; c: EdgeFusion.

- Table 3. Results for LCM-distillation of the BK-SDM-Tiny architecture from different pretrained students, with different datasets. Evaluated on COCO dataset with 4 inference steps.

Impact of fine-tuning the student first. As shown in Tab. 3, removing BK-SDM-Tiny finetuning from our advanced distillation pipeline results in a decrease in performance, with −2.67 IS, +2.72 FID and −0.01 CLIP. Again, this observation is confirmed by our human evaluation (Tab. 4), with 61.7% win rate for EdgeFusion.

Model w/o finetuning w/o improved data

BK-SDM-Adv-Tiny 67.3% 60.9% EdgeFusion 61.7% 62.8%

- Table 4. Human preference evaluation. The win rate of our models against the same architecture without improved data and without student finetuning is reported (1500 comparisons, 21 participants).

Quantization. We compare the performance between FP32 and W8A16 quantized EdgeFusion. Fig. 3 shows the results of 3 prompts. There is no performance degradation after applying the W8A16 quantization.

4.3. Model benchmark

Inference speed on GPU Tab. 5 compares the inference speed of EdgeFusion (without quantization) to BK-SDMTiny and SD-v1.4, demonstrating a remarkable ×10.3 speedup compared to SD-v1.4.

Model Inference Steps Time (ms) SD-v1.4 25 930 BK-SDM-Tiny 25 540 EdgeFusion 4 / 2 / 1 150 / 90 / 90

- Table 5. Benchmark results on NVIDIA A100 GPU (50 runs).

NPU benchmark Tab. 6 shows the relative time ratio results for the cross attention block, without and with the MLT method. The results demonstrate a latency gain of approximately 73% when using MLT, effectively reducing Direct

[Figure 16]

[Figure 17]

[Figure 18]

FP32

[Figure 19]

[Figure 20]

[Figure 21]

W8A16

##### Figure 3. Comparison between FP32 and W8A16 quantized EdgeFusion (2 steps). Prompts can be found in Supplementary.

Operation† w/o MLT w/ MLT DMA load for Query, Key 0.6% 0.6% Tensor calculation: Query × KeyT = Sin 6.9% 7.2% DMA store: Sin 19.1% DMA load: Sin 18.2% Vector calculation: Softmax(Sin) = Sout 11.0% 11.6% DMA store: Sout 18.5% DMA load: Sout, Value 19.5% 0.8% Tensor calculation: Sout × Value = A 5.4% 5.7% DMA store: A 0.7% 0.7% Total 100% 27%

†: Sin and Sout indicate the input and output of the softmax layer, respectively. A is the output of the cross attention block.

- Table 6. Inference time reduction using MLT. For the cross attention block, the relative time ratio is computed by comparing operations without MLT to those with MLT.

Model MLT

Time (ms) Encoder UNet Decoder Total

SDM v1.5 ✗ 52.5 1350.3 483.4 1885.8 EdgeFusion ✗ 51.9 454.3 484.7 990.9 EdgeFusion ✓ 52.2 200.2 486.1 738.5

- Table 7. Benchmark on Exynos 2400 with and without MLT.

Memory Access (DMA) accesses. Tab. 7 shows the inference efficiency of architectural elements in EdgeFusion.

### 5. Conclusion

We introduce EdgeFusion to optimize SD models for efficient execution on resource-limited devices. Our findings show that utilizing informative synthetic image-text pairs is crucial for improving the performance of compact SD models. We also design an advanced distillation process for LCM to enable high-quality few-step inference. Through detailed exploration of deployment techniques, EdgeFusion can produce photorealistic images from prompts in one second on Samsung Exynos NPU.

### References

- [1] Nota AI. BK-SDM-Tiny. https://huggingface.co/ nota-ai/bk-sdm-tiny, 2023. 2
- [2] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-delta: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024. 1, 2
- [3] Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024. 1, 2
- [4] Jiwoong Choi, Minkyu Kim, Daehyun Ahn, Taesu Kim, Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Jae-Joon Kim, and Hyungjun Kim. Squeezing large-scale diffusion models for mobile. In ICML Workshop, 2023. 1, 2
- [5] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. In NeurIPS, 2022. 3
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 1
- [7] Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In ICML, 2021. 3
- [8] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 1
- [9] Yatharth Gupta, Vishnu V Jaddipal, Harish Prabhala, Sayak Paul, and Patrick Von Platen. Progressive knowledge distillation of stable diffusion xl using layer level loss. arXiv preprint arXiv:2401.02677, 2024. 1, 2
- [10] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate post-training quantization for diffusion models. In NeurIPS, 2024. 1
- [11] Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, and Jin Young Choi. A comprehensive overhaul of feature distillation. In ICCV, 2019. 2
- [12] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, 2023. 1
- [13] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. Bk-sdm: A lightweight, fast, and cheap version of stable diffusion. arXiv preprint arXiv:2305.15798,

2023. 1, 2, 3, 4, 6

- [14] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. In NeurIPS, 2023. 1, 2
- [15] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. In NeurIPS, 2023. 1
- [16] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Ji-

- aming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Qiao. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 2
- [17] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 1, 2
- [18] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023. 1
- [19] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2
- [20] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon LLM: Outperforming curated corpora with web data only. In NeurIPS Workshop,

2023. 2

- [21] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 2
- [22] Robin Rombach and Patrick Esser. Stable Diffusion v1-

4. https://huggingface.co/CompVis/stablediffusion-v1-4, 2022. 2

- [23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2, 3
- [24] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. In ICLR, 2015. 2
- [25] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 1
- [26] Samsung Semiconductor. Samsung exynos 2400. https: //semiconductor.samsung.com/processor/ mobile-processor/exynos-2400/, 2024. 2, 3
- [27] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023. 1, 2
- [28] Christoph Schuhmann and Romain Beaumont. Laionaesthetics. https://laion.ai/blog/laionaesthetics, 2022. 2
- [29] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS Workshop,

2022. 2

- [30] SG161222. Realistic-Vision-V5.1. https : / / huggingface . co / SG161222 / Realistic _ Vision_V5.1_noVAE, 2023. 2, 3
- [31] Haluk Topcuoglu, Salim Hariri, and Min-You Wu. Performance-effective and low-complexity task scheduling for heterogeneous computing. IEEE transactions on parallel and distributed systems, 13(3):260–274, 2002. 3

###### [32] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. arXiv preprint arXiv:2311.09257,

2023. 2

###### [33] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023. 1, 2

## EdgeFusion: On-Device Text-to-Image Generation Supplementary Material

### A. Data description

Tab. 8 summarize the datasets used in this paper, and precise the size of each dataset. Notably, our synthetic dataset (synt) is composed of 62.3k image-text pairs, and our manually curated synthetic dataset (synt-cur) is composed of 29.9k high quality image-text pairs.

### B. Teacher models

In Tab. 9, we compare the performance of Realistic Vision 5.1, used as a teacher model in this work, with SD-v1.4. While both models share the same architecture, their different training results in a gap in performance. Notably, Realistic Vision 5.1 outperform SD-v1.4 on both IS and CLIP when evaluated on COCO dataset.

### C. Supplementary results on data curation

To better understand the effect of data curation, Tab. 10 provides a comparison of training using a subset of 10k samples of our synthetic dataset (synt), and a manually curated version of this subset. The data curation removes 17.4% of the data, resulting in 8.26k samples in the curated dataset. The evaluation is performed for BK-SDM-Tiny (25 steps), and BK-LCM (2 and 4 steps), on COCO dataset.

### D. Supplementary output examples

Fig. 4, 5, 6 and 7 give additional output examples for BK-SDM-Adv-Tiny and EdgeFusion with 1, 2 and 4 diffusion steps, respectively. Prompts can be found in the figures descriptions.

### E. EdgeFusion 2-steps evaluation

Tab. 11 provides the evaluation on COCO dataset for BK-LCM-Tiny with 2 inference step.

### F. Prompts for FP32 and W8A16 Comparison

In this section, we show the prompts used in Fig. 3. Prompts: “cyclops fighter, white grey blue color palette, male, d&d, fantasy, intricate, elegant, highly detailed, long silver hair, digital painting, artstation, octane render, concept art, matte, sharp focus, illustration, hearthstone, art by artgerm, alphonse mucha johannes voss” (left), “Max Headroom in a Perfume advertisement, magical, science fiction, symmetrical face, large eyes, Chanel, Calvin Klein, Burberry, Versace, Gucci, Dior, hyper realistic, digital art, octane render, trending on artstation, artstationHD, artstationHQ, unreal engine, 4k, 8k” (middle), “oil painting of

holocaust LANDSCAPE, diffuse lighting, intricate, highly detailed, lifelike, photorealistic, illustration, concept art, smooth, sharp focus, art by francis bacon” (right).

Dataset Description Size L-orig LAION subset used to train BK-SDM-Tiny in Kim et al. [13] 212.8k pairs L-pp L-orig, plus additional preprocessing (deduplication, low resolution filtering, automatic

119.9k pairs

cropping, manual removal)

L-pps Recaptioned L-pp with synthetic prompts (generated using Sphinx [16]) 119.9k pairs synt Synthetic text-image dataset, generated using GPT-4 and SDXL. Includes 21.6k manually

62.3k pairs

curated pairs.

synt-cur Manually curated synthetic text-image dataset, generated using GPT-4 and SDXL. 29.9k pairs

Table 8. Summary of datasets used in this study

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

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

- Figure 4. Output examples from BK-SDM-Adv-Tiny. Prompts (left to right, top to bottom): “a cute cat, hyper detailed, cinematic light, ultra high res, best shadow, RAW”, “a teenager in a black graphic tee, the design a splash of neon colors”, “photo portrait of Will Smith, hyper detailed, ultra high res, RAW”, “photo portrait of an old man with blond hair”, “painting of fruits on a table, next to a bottle of wine and a glass”, “a beautiful pizza”, “A sleek black sedan parked in a garage.”, “a fox, photo-realistic, high-resolution, 4k”, “photo portrait of a man wearing a charcoal gray suit, crisp and meticulously tailored”, “photo portrait of a young woman with blond hair”, “flowers in a golden vase”, “a lion is reading a book on the desk, photo-realistic”, “an astronaut in the jungle”, “A classic convertible parked on a sunny beach.”, “photo portrait of Brad Pitt wearing a black hat”.

Teacher model IS ↑ FID ↓ CLIP ↑

SD-v1.4 36.90 13.2 0.297 Realistic Vision v5.1 38.13 17.63 0.307

- Table 9. Evaluation results for SD-v1.4 (used in BK-SDM [13]) and for Realistic Vision v5.1 (used in this work). The evaluation is performed on COCO dataset, with 25 inference steps.

Model Training dataset Inference Steps IS ↑ FID ↓ CLIP ↑ BK-SDM-Tiny L-pps & synt subset (not curated) 25 33.37 22.58 0.285

L-pps & synt subset (curated) 25 32.54 22.86 0.285 BK-LCM-Tiny L-pps & synt subset (not curated) 4 27.52 21.33 0.270

L-pps & synt subset (curated) 4 27.57 21.48 0.269 BK-LCM-Tiny L-pps & synt subset (not curated) 2 26.31 22.52 0.265 L-pps & synt subset (curated) 2 26.60 22.46 0.265

- Table 10. IS, FID, and CLIP results for BK-SDM-Tiny and BK-LCM-Tiny models trained on L-pps and a subset of the synthetic image-pair dataset. The subset size is 10k samples before being curated, and 8.26k samples after.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

- Figure 5. Output examples from EdgeFusion with 1 step inference. Prompts (left to right, top to bottom): “a bear wearing a hat and sunglasses, photo-realistic, high-resolution, 4k”, “photo portrait of an old man with blond hair”, “flowers in a golden vase”, “a cute rabbit”, “a frozen tundra under the aurora borealis, the night sky alive with color, ice crystals sparkling like diamonds”, “close up on colorful flowers”, “a beautiful sports car”, “a mountain peak during golden hour”, “photo portrait of an old woman dressed in red”, “photo portrait of a young man with blond hair”, “beautiful photo portrait of the king, high resolution, 4k”, “a cute dog wearing a hat”, “photo portrait of an old man dressed in white”, “a cute cat wearing sunglasses, photo-realistic”, “photo portrait of an old woman with blond hair”.

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

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

- Figure 6. Output examples from EdgeFusion with 2 step inference. Prompts (left to right, top to bottom): “photo portrait of a young man with blond hair”, “a cute rabbit wearing sunglasses, photo-realistic”, “photo portrait of an old man wearing a suit”, “a lion playing chess, photo-realistic”, “photo portrait of Johnny Depp, perfect skin, hyper detailed, ultra high res, RAW”, “close up on colorful flowers”, “an enchanted forest, trees aglow with bioluminescent fungi, mythical creatures flitting through the shadows”, “a beautiful car in a mountain”, “beautiful photo portrait of the queen, high resolution, 4k”, “a bustling city street at night, neon lights reflecting off wet pavement, the hum of activity never ceasing”, “photo portrait of an old man with blond hair”, “a cow swimming under the sea, photo-realistic”, “photo portrait of a black woman with short hair, nature in the background”, “A yellow taxi driving through the city streets.”, “photo portrait of a teenager with a pink beanie, a pop of color on a snowy day”.

[Figure 73]

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

- Figure 7. Output examples from EdgeFusion with 4 step inference. Prompts (left to right, top to bottom): “photo portrait of Tom Cruise wearing a black hat”, “a cute dog making a sandcastle, photo-realistic”, “photo portrait of a young woman dressed in red”, “a beautiful sports car in a forest”, “a cute dog wearing sunglasses, photo-realistic”, “a cabin in snowy forest”, “photo portrait of an astronaut, high resolution, 4k”, “a cute rabbit playing chess, photo-realistic”, “a field of lavender swaying, the scent perfuming the air, bees buzzing lazily from flower to flower”, “photo portrait of a woman in a golden yellow summer dress, capturing the essence of sunny days”, “a woman in a purple silk scarf, adding a touch of elegance to her outfit, high resolution, 4k”, “a cute cat, hyper detailed, cinematic light, ultra high res, best shadow, RAW”, “a beautiful sports car under the snow”, “photo portrait of a woman in a bright red cardigan, buttoned up over a white blouse”, “painting of a group of people eating next to a lake”.

Pretrained student Dataset IS ↑ FID ↓ CLIP ↑

BK-SDM-Tiny [13] L-pps & synt a 25.15 23.49 0.258 L-orig L-orig b 23.72 27.30 0.248 L-pp L-pp 23.97 26.79 0.249 L-pps L-pps 25.93 23.36 0.261 L-pps & synt L-pps & synt 26.99 20.57 0.271 L-pps & synt cur L-pps & synt cur 26.63 21.20 0.268 synt synth 26.62 21.36 0.267 synt cur synt cur 26.53 21.41 0.264 synt cur L-pps & synt c 27.11 20.47 0.268

Termed a: w/o finetuning; b: w/o improved data; c: EdgeFusion.

- Table 11. Results for LCM-distillation of the BK-SDM-Tiny architecture from different pretrained students, with different datasets. Evaluated on COCO dataset with 2 inference steps.

