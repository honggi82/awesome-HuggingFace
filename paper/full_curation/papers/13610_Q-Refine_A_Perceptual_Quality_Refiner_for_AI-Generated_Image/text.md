# arXiv:2401.01117v1[cs.CV]2Jan2024

Q-REFINE: A PERCEPTUAL QUALITY REFINER FOR AI-GENERATED IMAGE Chunyi Li1,2 Haoning Wu3 Zicheng Zhang1 Hongkun Hao1 Kaiwei Zhang1 Lei Bai2 Xiaohong Liu1 Xiongkuo Min1 Weisi Lin3 Guangtao Zhai1 Shanghai Jiao Tong University1, Shanghai AI Lab2, Nanyang Technological University3

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|

| |
|---|

[Figure 7]

Original AIGI

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

| |
|---|

Traditional

[Figure 14]

| |
|---|

| |
|---|

| |
|---|

Refiner

| |
|---|

| |
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

| |
|---|

Q-Refine

| |
|---|

| |
|---|

[Figure 21]

| |
|---|

| |
|---|

| |
|---|

(1) (2) (3) (4) (5) (6)

Fig. 1. The original AIGIs from AGIQA-3K [1], optimized by Traditional Refiners and Q-Refine we proposed. As a qualityaware metric, the Q-Refine can add details on the blurred part, to better optimize low-quality regions of (1)(2); improve clarity in medium-quality regions of (3)(4) without changing the whole image; and avoid degrading the high-quality regions of (5)(6).

ABSTRACT With the rapid evolution of the Text-to-Image (T2I) model in recent years, their unsatisfactory generation result has become a challenge. However, uniformly refining AI-Generated Images (AIGIs) of different qualities not only limited optimization capabilities for low-quality AIGIs but also brought negative optimization to high-quality AIGIs. To address this issue, a quality-award refiner named Q-Refine1 is proposed. Based on the preference of the Human Visual System (HVS), Q-Refine uses the Image Quality Assessment (IQA) metric to guide the refining process for the first time, and modify images of different qualities through three adaptive pipelines. Experimental shows that for mainstream T2I models, QRefine can perform effective optimization to AIGIs of different qualities. It can be a general refiner to optimize AIGIs from both fidelity and aesthetic quality levels, thus expanding the application of the T2I generation models.

### 1. INTRODUCTION

AI-Generated Content (AIGC) refers to the creation of content, such as images, videos, and music, using AI algorithms [1]. Since vision is the dominant way for humans to perceive the external world, AI-Generated Images (AIGIs) [2] have become one of the most representative forms of AIGC. The development of Text-to-Image (T2I) models is a crucial step in the advancement of AIGIs, as it allows for the creation of high-quality images that can be used in a variety of applications [3], including advertising, entertainment, and even scientific research. The importance of AIGI in today’s internet cannot be overstated, as it has the potential to revolutionize the way we consume and interact with visual content.

With the rapid technological evolution of T2I generation techniques, there have been at least 20 representative T2I models coexisting up to 2023, whose generation quality varies widely [1]. Coupled with confusing prompt input, unreasonable hyper-parameter settings, and insufficient iteration epochs, the quality of today’s AIGIs is still not satisfying.

Index Terms— AI-Generated Content, Image Quality Assessment, Image Restoration

Considering the wide application of AIGIs, their qual-

1The code will be released on https://github.com/Q-Future/Q-Refine

Quality-Related Module LQ Pipeline MQ Pipeline HQ Pipeline

[Figure 22]

[Figure 23]

ROIPollingFaster-RCNNQualityMap

MQ Bound

HQ Bound

LQ Bound

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Mode Selector

Weight

[Figure 28]

[Figure 29]

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 30]<br><br>[Figure 31]<br><br>| | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 32]<br><br>[Figure 33]<br><br>| | |
| | | | |

Blind Restoration Original

Gaussian Generator

| | | | |
|---|---|---|---|
| | | | |
|[Figure 34]<br><br>[Figure 35]<br><br>| | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | |[Figure 36]<br><br>[Figure 37]<br><br>| |
| | | | |

HQ Prompt

Noise

Mask

Prompt

[Figure 38]

[Figure 39]

[Figure 40]

…

…

## +

[Figure 41]

[Figure 42]

Denoise

Denoise

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

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Input Stage 1 Output Stage 2 Output Final Output

- Fig. 2. Framework of Q-Refine, including a quality pre-prossess module, and three refining pipelines for low/medium/high quality (LQ/MQ/HQ) regions. The refining mechanisms for each pipeline are inspired by the predicted quality.

ity needs to be further optimized. However, this task is extremely challenging as shown in Fig. 1. Firstly, positive optimization is difficult to achieve for Low-Quality (LQ) regions. If their quality falls into a local optimum, they won’t be modified as a global optimum; secondly, local negative optimization is a hidden danger of Medium-Quality (MQ) regions. Since the quality distribution of images varies, refiners need to change only the LQ/MQ without affecting other regions; finally, global negative optimization is common in High-Quality (HQ) regions. Since the performance of refiners has a certain limit, blindly modifying an already highquality image can easily lead to a decrease in quality.

### 2. RELATED WORK AND CONTRIBUTIONS

Existing AIGI quality refiners are mainly divided into two types. The most commonly used method is to treat AIGI as a Natural Sense Image (NSI) and use a large-scale neural network for Image Restoration [4–6]; the other is to use the prompt as guidance, then put the AIGI back into a generative model for several epochs [7, 8]. However, both refiners ignore image quality. Using the same pipeline for LQ/MQ/HQ will lead to insufficient enhancement in the LQ regions and negative optimization in the HQ regions, essentially bringing all images to the MQ level as Fig. 1 shows.

Therefore, the quality of AIGIs needs to be computed in advance as refining guidance. However, Image Quality As-

sessment (IQA) [9, 10] and Refiner cannot be directly combined. Existing IQA works [11–13] usually consider the overall quality of the image, instead of a quality map, making it difficult for the refiner to implement local optimization.

To enhance positive while avoiding negative optimization, we found a way to combine IQA with refiners named QRefine, the first quality-aware refiner for AIGIs based on the preference of the Human Visual System (HVS) with the following contribution: (i) We introduce the IQA map to guide the AIGI refining for the first time. A new paradigm for AIGI restoration, namely using quality-inspired refining is proposed. (ii) We establish three refining pipelines that are suitable for LQ/MQ/HQ regions respectively. Each pipeline can self-adaptively determine the executing intensity according to the predicted quality. (iii) We extensively conduct comparative experiments between existing refiners and QRefine on mainstream AIGI quality databases. The result proved the strong versatility of Q-Refine.

### 3. PROPOSED METHOD 3.1. Framework

Since perceptual quality has been widely recognized as a decisive role for Generative AI [14–16], Q-Refine is designed to refine AIGIs with separated pipelines according to the quality. Our framework is shown in Fig. 2 with an IQA module to predict a quality map and three pipelines include: (1) Gaussian

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

De-

noise

Quality Score: 55

Quality Score: 48

[Figure 63]

[Figure 64]

[Figure 65]

Add noise

Denoise

Quality Score: 48 Quality Score: 30 Quality Score: 81

(a) Patch quality map (b) Flattened quality map

- Fig. 3. The refining result by only denoise / add noise + denoise from SDXL [8]. Adding noise reduces quality [19], but it lays the foundation for global optimality before denoising.

Fig. 4. Using original patch quality map / flattened map to guide the inpainting. (a) suffers from block effects and unexpected artifacts while (b) has a smooth and natural result.

Noise: encouraging changing the LQ region by adding noise; (2) Mask Inpainting: generating a mask from the quality map to reserve HQ region; (3) Global Enhancement: setting an enhancement threshold to fine-tune the final output.

generate blurry images. Such a problem may be due to the initial few denoising steps, causing the image to fall into a local optimum. In such cases, the model will stubbornly retain some LQ regions, causing the image to remain unchanged even after iterating hundreds of epochs. To solve this problem, such LQ regions should rewind to previous steps, to trigger the model’s denoising mechanism. Since Sec. 3.2 provides a quality map, the LQ region can be identified and then modified. As the starting noise image before denoising, we superimpose Gaussian noise in the LQ region to obtain the first stage output Is1:

### 3.2. IQA Module

Splitting the image into patches [17], evaluating them separately [18], and then combining them is a commonly used [19] IQA pipeline in recent years. It can evaluate the overall quality while providing a rough quality map through patches. By dividing an AIGI into n × n, a patch P with index (i,j) ∈ [0,n − 1] has:

W = max(BLQ − Q,0) Is1 = WG(h,w) + (1 − W)I

(3)

nh:i+1n h,nj w:j+1n w)) (1) where (h,w) are the height/width of the input image I. Since extracting the quality map requires a network sensitive for both global regression and local perception, the dual-task structure for image classification/detection, namely FasterRCNN [20], is utilized as our CNN model backbone. For local quality Q(i,j), referring to previous quality map extractor [19], we use the largest value in each patch as its quality score, to obtain a n × n quality map Q. However, for global quality q, to avoid excessive complexity affecting the subsequent three refining pipelines, we abandoned all global extractors and directly averaged the patch scores as:

P(i,j) = CNN(I( i

where the noise weight map W is determined by LQ bound BLQ, a region with lower quality has higher weight while quality larger than BLQ leads to zero weight. The size of Gaussian noise G is (h,w). As Fig. 3 shows, though the noise from the stage 1 pipeline may temporarily reduce the image quality, it can help the following two pipelines to change the LQ region. By refining the final output, it can move the local quality optimum toward the global optimum.

### 3.4. Stage 2 Pipeline: Mask Inpainting

Since different regions of images have different quality, this pipeline aims to retain HQ and modify other regions. This operation can be completed through the inpainting method, by taking LQ regions as a mask. However, as the edges between patches are un-discontinuous, directly using the quality map with n×n patches to generate this mask will cause some unsatisfying results like Fig. 4 shows. First, a discontinuous quality map may require the inpainting model to retain a certain patch and modify adjacent patches. The result will have obvious block effects at the edge of the patches. Second, the inpainting model tends to redraw the main object with a regular rectangle mask. Though we only want some detail on

Q(i,j) = RoIPool(P(i,j)) q = Avg(Q(i,j))

(2)

where Avg and RoIPool are the average and average-maxpooling layers. The global quality/quality map will guide refining pipelines.

### 3.3. Stage 1 Pipeline: Gaussian Noise

Existing T2I generation models cannot always ensure a HQ result, even the most advanced model [21] may occasionally

|Objective Quality<br><br>OriginalQualityRange<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Original Image<br><br>Blind Enhancer<br><br>Prompt Enhancer<br><br>(+2.91, +2.05)<br><br>(+1.40, +0.34)<br><br>(-1.28, +0.08)<br><br>Fig. 5. Using blind enhancer or prompt-guided enhancer to refine images in different quality groups in AGIQA-3K [1]. Blind enhancer shows better refining results for LQ groups but causes negative optimization for HQ groups.<br><br>a plain background, it will generate unexpected main objects here instead. Thus the patch quality map Q needs to be flat-<br><br>SOTAs of the two and the refining results enhancer is suitable prompt-guided enhancer based on global quality,<br><br>If = {<br><br>where EB stands for a similar mechanism strength (to avoid mask. The HQ bound while, considering improve the generation the original prompt as of whether the input refine its quality by|
|---|

enhancers are DiffBIR [5] and SDXL [8], s are in Fig. 5. Considering the blind for LQ, but performs worse than the

er on HQ, we implement the enhancer y, with the final output If:

OriginalQualityRange

EB,EP∥q < BHQ}(Is2) (7)

a blind enhancer while EP performs as (6), but in smaller hyper-parameter

negative optimization for HQ) without a

BHQ determines such selection. Meansome positive words [23] will significantly

quality, we combine these words with

the input of EP. Therefore, regardless belongs to LQ/MQ/HQ, our model can

providing an HQ result.

### 4. EXPRIMENT

tened before inpainting. Considering smoothness is our first requirement, we use the smoothest interpolation method BiCubic [22], to convolve each pixel with 16 adjacent pixels:

### 4.1. Expriment Settings

Our Q-Refine is validated on three AIGI quality databases, including AGIQA-3K, AGIQA-1K, and AIGCIQA [1,27,28]. The quality of AIGIs before/after Q-Refine is compared to prove the general optimization level. Moreover, since AGIQA-3K [1] includes five T2I models [7, 8, 24–26] with remarkable quality differences, their performances are listed respectively to prove Q-Refine’s versatility on LQ/MQ/HQ regions. Besides the original image, the image quality generated by Q-Refine is compared with three latest image restoration refiners [4–6] and two representative generative refiners [7,8] as Sec. 2 reviewed.

2

Q(⌊xnh⌋+r,⌊ywn⌋+c)Cub(r−x,c−y) (4)

Q(x,y) =

r,c=−1

where pixel ( xnh , ywn ) from the original quality map is the corresponding pixel (x,y) from the flattened map and

Cub stands for the Bi-Cubic [22] matrix. From this, the probability density function z of each step is:

z = QKV(prompt,mask = {Q − BMQ}) (5)

where we set quality region below the threshold BMQ as mask. QKV stands for multi-head attention, which depends on the input prompt and mask. Set the starting point of denoising to x0 = Is1, we have the second stage output Is2:

To measure the image quality, since FID [29] is inconsistent with human subjective preferences, we use IQA methods to represent HVS’s perceptual quality. The image quality consists of two different levels. Signal-fidelity characterizes low-level quality including factors like blur or noise, which is the traditional definition of image quality. Thus, we use the classic Brisque [30] as its index. Aesthetic, however, represents high-level quality, which depends on the overall appeal and beauty of the image. Here we take the HyperIQA [31] as the index since it best correlates human subjective preference on AIGIs. Moreover, for a more intuitive performance comparison, we also take CLIPIQA [32] as an overall quality indicator for both levels.

Is2 = xm = Dm(xm−1) = Dm(Dm−1 ···D1(Is1)) (6)

where Dm represents the diffusion operation at the m-th iteration and x stands for this intermediate state. From this, we used masks to modify the LQ/MQ region through the smoothed quality map without affecting the HQ region.

### 3.5. Stage 3 Pipeline: Global Enhancement

After local inpainting, to further improve the image quality, this pipeline fine-tunes the image’s low-level attributes, rather than adding new objects. Low-level quality enhancers include the following two types. One is the traditional image superresolution/restoration method, which ignores the prompt as a blind enhancer, using prior knowledge from NSIs to achieve image-to-image reconstruction. The other is the generative model, which uses the original prompt to guide the diffusion for several steps, namely prompt-guided enhancer. The

### 4.2. Expriment Result and Discussion

The experimental performance on the AGIQA-3K [1] database and five subsets is shown in Table 1. In the general perspective, Q-Refine achieved the best aesthetic, fidelity, and overall quality. On a total of 18 indexes in six sets, Q-Refine reached SOTA on 16 of them. It is worth mentioning that Q-Refine never negatively optimized any index

- Table 1. Refined result of AGIQA-3K [1] database and five subsets from different generators. The refined results with the best quality are noted in red. The refined quality below the original data is noted in underline.

|Refiner<br><br>|Mean<br><br>|GLIDE [24]<br><br>|SDXL [8]|
|---|---|---|---|
| |Overall↑ Aesthetic↑ Fidelity↓<br><br>|Overall↑ Aesthetic↑ Fidelity↓|Overall↑ Aesthetic↑ Fidelity↓|
|Original DASR [4] DiffBIR [5] RFDN [6] SD1.5 [7] SDXL [8] Q-Refine<br><br>|0.5710 0.4890 38.975 0.4987 0.5507 45.252 0.5829 0.5935 35.049 0.5704 0.4885 38.831 0.6461 0.5359 39.649 0.6489 0.5418 32.999 0.7232 0.6021 22.463<br><br>|0.2901 0.2895 71.331 0.2384 0.3007 63.922 0.4104 0.3982 60.728 0.2900 0.2886 71.178 0.5852 0.4749 67.669 0.5609 0.4416 52.711 0.6333 0.4986 31.722<br><br>|0.7559 0.6173 24.816 0.7298 0.7011 22.728 0.7400 0.7273 26.309 0.7532 0.6164 24.522 0.6917 0.5632 29.996 0.7111 0.5842 24.589 0.8007 0.6640 18.145<br><br>|

|Refiner|DALLE2 [25]<br><br>|MidJourney [26]|SD1.5 [7]|
|---|---|---|---|
| |Overall↑ Aesthetic↑ Fidelity↓<br><br>|Overall↑ Aesthetic↑ Fidelity↓<br><br>|Overall↑ Aesthetic↑ Fidelity↓|
|Original DASR [4] DiffBIR [5] RFDN [6] SD1.5 [7] SDXL [8] Q-Refine<br><br>|0.6193 0.4884 29.264 0.5686 0.5884 38.917 0.5947 0.6118 27.658 0.6191 0.4875 29.120 0.6543 0.5425 33.885 0.6692 0.5654 29.789 0.7350 0.6133 20.763<br><br>|0.5340 0.4751 42.6938 0.4521 0.5562 39.6575 0.5543 0.5706 31.3149 0.5337 0.4755 42.6950 0.6295 0.5305 39.8227 0.6307 0.5359 34.2414 0.7384 0.6097 19.4677<br><br>|0.6555 0.5749 26.7706 0.5045 0.6069 61.0347 0.6153 0.6598 29.2356 0.6561 0.5745 26.6402 0.6696 0.5686 26.8717 0.6726 0.5819 23.6660 0.7084 0.6249 22.2168<br><br>|

- Table 2. Three AIGI quality databases [1,27,28] before/after Q-Refine. The best result is noted in red.

Table 3. The AGIQA-3K [1] refining result after abandoning different Q-Refine pipelines. The best result is noted in red.

|Databases<br><br>|Overall↑ Aesethic↑ Fidelity↓|
|---|---|
|AGIQA-3K [1] AGIQA-3K + Q-Refine<br><br>|0.5710 0.4890 38.975 0.7232 0.6021 22.463|
|AGIQA-1K [27] AGIQA-1K + Q-Refine|0.6454 0.5896 42.288 0.7258 0.6511 27.767<br><br>|
|AIGCIQA [28] AIGCIQA + Q-Refine|0.5720 0.5213 31.443 0.6639 0.6196 23.365<br><br>|

|Pipelines<br><br>|Overall↑ Aesethic↑ Fidelity↓|
|---|---|
|(1)+(2)+(3)<br><br>(1)+(2)<br>(2)+(3)<br><br><br>(1)+(3)<br><br><br>(2)<br>(3)<br>|0.7232 0.6021 22.463 0.6604 0.5610 32.079 0.6897 0.5884 24.373 0.6315 0.5445 29.917 0.6165 0.5147 34.299 0.6852 0.5571 29.332<br><br>|

that other Refiners never achieved. From a detailed perspective, Q-refine has a satisfying performance on all subsets as we stated in our contributions. Firstly, for the worst quality GLIDE [24] model, the significant improvement of the three indexes proves that Q-Refine can effectively refine LQ. Secondly, for the strongest SDXL [8] model, each index after Q-Refine does not drop like other methods certified the robustness on HQ. Thirdly, in the remaining three subsets with average performance, the rise in all indexes indicated that QRefine can identify and modify the LQ/MQ region and retain the HQ. Table 2 also proved in databases constructed by different T2I generation metrics with different performance, QRefine can provide an HQ refining result for all AIGIs.

### 4.3. Ablation Study

To quantify the contributions of three pipelines of Q-Refine, we abandon its stage (1)/(2)/(3) pipelines respectively in this section. As a side-effect module, (1) does not appear alone.

The result in Table 3 indicates the positive effect of add-noise on subsequent denoising, as the noise from (1) greatly improves the image quality refined by (2). Both (2) and (3) have a positive effect on the refining task , which are responsible for high-level and low-level optimization respectively. When the two are combined, the image quality is further improved. Thus, all pipelines contribute to the final result.

### 5. CONCLUSION

In this study, targeting AIGI’s unsatisfying quality, a qualityaware refiner is proposed. To enhance positive while avoiding negative optimization in the LQ/HQ region, IQA is innovatively introduced into the image refiner to provide guidance. Inspired by quality maps, three well-designed pipelines work collaboratively to optimize the LQ/MQ/HQ regions. Experimental data shows that Q-Refine improves the quality of AIGIs at both fidelity and aesthetic levels, which enables a better viewing experience for humans in the AIGC era.

#### 6. REFERENCES

- [1] Chunyi Li, Zicheng Zhang, Haoning Wu, Wei Sun, Xiongkuo Min, Xiaohong Liu, Guangtao Zhai, and Weisi Lin, “Agiqa-3k: An open database for ai-generated image quality assessment,” IEEE TCSVT, 2023.
- [2] Stanislav Frolov, Tobias Hinz, Federico Raue, J¨orn Hees, and Andreas Dengel, “Adversarial text-to-image synthesis: A review,” Neural Networks, 2021.
- [3] Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon, “Text-to-image diffusion model in generative ai: A survey,” arXiv:2303.07909, 2023.
- [4] Yunxuan Wei, Shuhang Gu, Yawei Li, Radu Timofte, Longcun Jin, and Hengjie Song, “Unsupervised real-world image super resolution via domain-distance aware training,” in IEEE/CVF CVPR, 2021.
- [5] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Ben Fei, Bo Dai, Wanli Ouyang, Yu Qiao, and Chao Dong, “Diffbir: Towards blind image restoration with generative diffusion prior,” arXiv:2308.15070, 2023.
- [6] Jie Liu, Jie Tang, and Gangshan Wu, “Residual feature distillation network for lightweight image super-resolution,” in ECCV, 2020.
- [7] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer, “High-resolution image synthesis with latent diffusion models,” in IEEE CVPR, 2022.
- [8] Robin Rombach, Andreas Blattmann, and Bj¨orn Ommer, “Text-guided synthesis of artistic images with retrievalaugmented diffusion models,” arXiv:2207.13038, 2022.
- [9] Tengchuan Kou, Xiaohong Liu, Wei Sun, Jun Jia, Xiongkuo Min, Guangtao Zhai, and Ning Liu, “Stablevqa: A deep noreference quality assessment model for video stability,” in ACM MM, 2023.
- [10] Yixuan Gao, Yuqin Cao, Tengchuan Kou, Wei Sun, Yunlong Dong, Xiaohong Liu, Xiongkuo Min, and Guangtao Zhai, “Vdpve: Vqa dataset for perceptual video enhancement,” in IEEE/CVF CVPR, 2023.
- [11] Chunyi Li, May Lim, Abdelhak Bentaleb, and Roger Zimmermann, “A real-time blind quality-of-experience assessment metric for http adaptive streaming,” in IEEE ICME, 2023.
- [12] Chunyi Li, Zicheng Zhang, Wei Sun, Xiongkuo Min, and Guangtao Zhai, “A full-reference quality assessment metric for cartoon images,” in IEEE MMSP, 2022.
- [13] Xinhui Huang, Chunyi Li, Abdelhak Bentaleb, Roger Zimmermann, and Guangtao Zhai, “Xgc-vqa: A unified video quality assessment model for user, professionally, and occupationallygenerated content,” in IEEE ICMEW, 2023.
- [14] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, and Weisi Lin, “Q-bench: A benchmark for general-purpose foundation models on low-level vision,” arXiv:2309.14181, 2023.
- [15] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Kaixin Xu, Chunyi Li, Jingwen Hou, Guangtao Zhai, Geng Xue, Wenxiu Sun, Qiong Yan, and Weisi Lin, “Q-instruct: Improving low-level visual abilities for multi-modality foundation models,” arXiv:2311.06783, 2023.
- [16] Zicheng Zhang, Haoning Wu, Zhongpeng Ji, Chunyi Li, Erli Zhang, Wei Sun, Xiaohong Liu, Xiongkuo Min, Fengyu Sun,

- Shangling Jui, et al., “Q-boost: On visual quality assessment ability of low-level multi-modality foundation models,” arXiv:2312.15300, 2023.
- [17] Zicheng Zhang, Wei Sun, Yingjie Zhou, Haoning Wu, Chunyi Li, Xiongkuo Min, Xiaohong Liu, Guangtao Zhai, and Weisi Lin, “Advancing zero-shot digital human quality assessment through text-prompted evaluation,” arXiv:2307.02808, 2023.
- [18] Zicheng Zhang, Wei Sun, Houning Wu, Yingjie Zhou, Chunyi Li, Xiongkuo Min, Guangtao Zhai, and Weisi Lin, “Gms-3dqa: Projection-based grid mini-patch sampling for 3d model quality assessment,” arXiv:2306.05658, 2023.
- [19] Zhenqiang Ying, Haoran Niu, Praful Gupta, Dhruv Mahajan, Deepti Ghadiyaram, and Alan Bovik, “From patches to pictures (paq-2-piq): Mapping the perceptual space of picture quality,” in IEEE/CVF CVPR, 2020.
- [20] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” NIPS, 2015.
- [21] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li, “Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,” arXiv:2310.00426, 2023.
- [22] Dianyuan Han, “Comparison of commonly used image interpolation methods,” in ICCSEE, 2013.
- [23] Nikita Pavlichenko and Dmitry Ustalov, “Best prompts for text-to-image models and how to find them,” in ACM SIGIRl, 2023.
- [24] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” in ICML, 2022.
- [25] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv:2204.06125, 2022.
- [26] David Holz, “Midjourney,” https://www.midjourney. com/, 2023.
- [27] Zicheng Zhang, Chunyi Li, Wei Sun, Xiaohong Liu, Xiongkuo Min, and Guangtao Zhai, “A perceptual quality assessment exploration for aigc images,” in IEEE ICMEW, 2023.
- [28] Jiarui Wang, Huiyu Duan, Jing Liu, Shi Chen, Xiongkuo Min, and Guangtao Zhai, “Aigciqa2023: A large-scale image quality assessment database for ai generated images: from the perspectives of quality, authenticity and correspondence,” in CICAI, 2023.
- [29] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” in NIPS, 2017.
- [30] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik, “No-reference image quality assessment in the spatial domain,” IEEE TIP, 2012.
- [31] Shaolin Su, Qingsen Yan, Yu Zhu, Cheng Zhang, Xin Ge, Jinqiu Sun, and Yanning Zhang, “Blindly assess image quality in the wild guided by a self-adaptive hyper network,” in IEEE/CVF CVPR, 2020.
- [32] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy, “Exploring clip for assessing the look and feel of images,” in AAAI, 2023.

