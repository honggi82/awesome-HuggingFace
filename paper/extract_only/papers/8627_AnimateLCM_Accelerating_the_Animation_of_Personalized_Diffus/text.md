# arXiv:2402.00769v3[cs.CV]16Oct2024

## AnimateLCM: Computation-Efficient Personalized Style Video Generation without Personalized Video Data

Fu-Yun Wang

Zhaoyang Huang

WeiKang Bian

fywang@link.cuhk.edu.hk MMLab, CUHK

zhaoyanghuang@avolutionai.com Avolution AI China

wkbian@outlook.com MMLab, CUHK

Hong Kong SAR

Hong Kong SAR

Xiaoyu Shi

xiaoyushi@link.cuhk.edu.hk MMLab, CUHK

Hong Kong SAR

Keqiang Sun

kqsun@link.cuhk.edu.hk MMLab, CUHK

Hong Kong SAR

Guanglu Song

guanglusong@foxmail.com SenseTime Research China

Yu Liu

liuyuisanai@gmail.com SenseTime Research China

### ABSTRACT

This paper introduces an effective method for computation-efficient personalized style video generation without requiring access to any personalized video data. It reduces the necessary generation time of similarly sized video diffusion models from 25 seconds to around 1 second while maintaining the same level of performance. The method’s effectiveness lies in its dual-level decoupling learning approach: 1) separating the learning of video style from video generation acceleration, which allows for personalized style video generation without any personalized style video data, and 2) separating the acceleration of image generation from the acceleration of video motion generation, enhancing training efficiency and mitigating the negative effects of low-quality video data.

### CCS CONCEPTS

• Computing methodologies → Animation.

### KEYWORDS

Consistency Models, Video Generation

ACM Reference Format:

Fu-Yun Wang, Zhaoyang Huang, WeiKang Bian, Xiaoyu Shi, Keqiang Sun, Guanglu Song, Yu Liu, and Hongsheng Li. 2024. AnimateLCM: ComputationEfficient Personalized Style Video Generation without Personalized Video Data. In Proceedings of Technical Communications (SA’24). ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SA’24, Dec 2024, Tokyo, Japan © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

Hongsheng Li

hsli@ee.cuhk.edu.hk MMLab, CUHK Centre for Perceptual and Interactive Intelligence (CPII) Hong Kong SAR

### 1 INTRODUCTION

Over the past few years, the field of video generation has made significant strides, thanks to the utilization of video diffusion models [Ho et al. 2022; Shi et al. 2024; Singer et al. 2022]. Currently, commonly applied video diffusion models can generate short video clips of about 2 seconds with relatively high-quality and reasonable motions. Nevertheless, those video generation models still have two significant shortcomings:

- (1) Slowgenerationspeed.The high-qualitygenerationachieved by the diffusion model relies on the iterative denoising process that gradually transforms high-dimensional noises into real data. However, the nature of iterative sampling leads to slow generation and high computational burdens of the diffusion model whose generation is much slower than other generative models (e.g., GAN) [Goodfellow et al. 2014; Yu et al. 2023]. For example, even testing on a high-performance GPU A100, it still takes 25 seconds to generate a 2-second short video clips in 512p×512p.
- (2) Inflexibility of generation style. In general, the quality of video data is inferior to that of image data, and accurately annotating video data with textual information is more challenging. Consequently, high-quality video data is difficult to obtain. Using low-quality video data typically results in suboptimal generation outcomes. Furthermore, users tend to prefer generating videos with higher quality and diverse styles, such as 2D animation, 3D animation, ink painting, etc. However, collecting high-quality videos in these styles is often very difficult.

Our approach effectively addresses the aforementioned issues without requiring complex steps. The core of our method lies in independently solving the problems of style learning and video generation acceleration, and then integrating them through weight fusion. By doing so, we only need to collect high-quality image data of specific styles for content learning, while utilizing lower-quality video

datasets to learn motion characteristics and accelerate video generation. Additionally, it is worth noting that a video can essentially be regarded as a series of images over time, connected through motion relationships. Therefore, we further decouple the acceleration of video generation into two parts: the generation acceleration of images and the generation acceleration of video motion. Our experimental results demonstrate that this decoupled acceleration method significantly enhances training efficiency. We illustrates the high-level idea of our methods in Fig. 1.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Animate LCMDDIM-Diffusion

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

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Style video Fast

Style video Fast

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

(LCM-Motion)

LCM

Video LCM

Low-quality Video Data

(LCM-LoRA)

Image Data

(Combination)

#### Figure 2: In the given denoising time budget, our model completes three high-quality generations, while video diffusion models are still in the process of denoising.

(Style-LoRA)

Base LDM

Personalized LDM

Animate LCM

High-quality Personalized Image Data

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Style video Fast

Style video Fast

Style video Fast

Figure 1: High-level overview of the pipeline of AnimateLCM.

- 1) Fine-tune the base LDM on the high-quality personalized style image data for stylized image generation. 2) Accelerate the base LDM into LCM for fast image generation. 3) Accelerate and extend the LCM into video LCM for fast video generation. 4) Combine the weights of personalized LDM and video LCM into AnimateLCM for computation-efficient personalized style video generation without any personalized video data.
- 2 RELATED WORKS

Diffusion Models have gradually dominated the filed of image and video generation, though suffering from low generation speed. LCM-LoRA [Geng et al. 2024; Luo et al. 2023; Song et al. 2023; Wang et al. 2024], working as a versatile acceleration module for image diffusion models, attracted huge attention. This work explores an versatile module, enabling the off-the-shell image diffusion models for computation-efficient personalized style video generation.

- 3 METHOD

data samples to re-fine-tune the model, transforming the base LDM into a personalized LDM that can generate high-quality images in a specified style. Generally, since personal users have limited training resources, such as GPUs, they often adopt parameter-efficient finetuning methods, with LoRA [Hu et al. 2021] being the most widely used. Specifically, the model’s weight update can be expressed as 𝑤 = 𝑤0 +𝐴𝐵, where 𝑤0 ∈ R𝑑×𝑘 is the original weight of the model, 𝐴 ∈ R𝑑×𝑟, and 𝐵 ∈ R𝑟×𝑘, with 𝑟 ≪ min(𝑑,𝑘). We can denote the 𝐴𝐵 as 𝜏𝑝𝑒𝑟𝑠𝑜𝑛𝑎𝑙𝑖𝑧𝑒, functioning as a specific weight residual for stylized generation.

AnimateLCM as a universal efficient video generation module. Our motivation is that the process of accelerating the model through consistency distillation can still be seen as a fine-tuning process of the pretrained model. Therefore, the distillation acceleration process of the base LDM can still be viewed as learning a weight residual for the base LDM. Specifically, 𝑤accelerated = 𝑤0 + 𝜏accelerated, where 𝑤accelerated,𝑤0,𝜏accelerated ∈ R𝑑×𝑘. In this way, we obtain two weight residuals, 𝜏accelerated and 𝜏personalized. We can linearly combine these residuals with the original weights for joint functionality. In practice, we use scaling factors 𝛼 and 𝛽 to control the influence of different weight residuals, combining them as𝑤combined = 𝑤0+𝛼𝜏accelerated+𝛽𝜏personalized. It’s important to note that since these residuals are directly integrated with the original weights, they do not affect the actual computation speed. The decoupling learning approach eliminates the need for high-quality personalized style video data collection. Overall, in the process described above, the stylized weight parameters are fine-tuned using a high-quality image dataset, while the weight residuals for acceleration are trained on general images and lowerquality video datasets, since high-quality stylized videos are hard to obtain. This approach allows us to combine the advantages of both methods, thereby eliminating the need for high-quality stylized video collections.

Our model supports high-quality personalized style video generation without learning from any personalized video data. It also reduces the generation time by around 10–25 times compared to similarly sized diffusion models. Its effectiveness benefits from its dual-level decoupled learning strategy: 1) separating video style learning from generation acceleration, and 2) separating image generation acceleration and video motion generation acceleration.

### 3.1 Decoupling Style Learning and Acceleration

Fine-tuning base LDM on a personalized image dataset. The base LDM is trained on a vast amount of text-image pairs that have not been thoroughly filtered. It can accept text inputs and generate corresponding images. Due to issues such as data quality and model capacity, this base model often struggles to accurately generate images that match the style described by the text. Fortunately, this pretrained base model has a good capability for fine-tuning. Typically, individuals can collect a few hundred or more private

### 3.2 Decoupling Image and Video Acceleration

Videos can generally be viewed as sequences of images over time, with motion relationships between temporally adjacent frames.

AnimateLCM: Computation-Efficient Personalized Style Video Generation without Personalized Video Data SA’24, Dec 2024, Tokyo, Japan

With this in mind, our motivation is that the acceleration weight residual mentioned earlier can be decomposed into two parts: one for learning the acceleration residuals in image generation, and the other for video motion generation. On one hand, learning from image data is typically easier and less costly than learning from video data. That is,

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

NFE=2NFE=4NFE=8NFE=12NFE=25NFE=50NFE=2NFE=4NFE=8NFE=2NFE=4NFE=8

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

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

𝜏𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑 = 𝜏𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑖𝑚𝑎𝑔𝑒 + 𝜏𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑣𝑖𝑑𝑒𝑜 , (1) where 𝜏𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑖𝑚𝑎𝑔𝑒 aims to image generation acceleration while 𝜏𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑣𝑖𝑑𝑒𝑜 aims to video motion generation acceleration. On the other hand, the content of a video forms the basis for its motion; without clear spatial content, any temporal relationships become meaningless. Therefore, we propose first accelerating the base LDM for image generation to obtain the base LCM. From there, we extend the base LCM to accept video inputs and continue acceleration training on readily available low-quality video datasets. We found that this approach significantly speeds up the training process. In practice, we implement the 𝑣𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑖𝑚𝑎𝑔𝑒 as the LoRA and implement the 𝑣𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑣𝑖𝑑𝑒𝑜 as the motion module composed of temporal attention blocks.

Animate LCM

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

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

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

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Teacher Model (DDIM)

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Thereby, the final weight is written as

Teacher Model (DPM)

- Figure 3: Qualitative comparison under different number of inference steps (NFE).

[Figure 159]

Text-to-VideoImage-to-VideoControl-Video

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

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

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

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

- Figure 4: 4-step generation results. AnimateLCM supports text-to-video, image-to-video, and controllable video generation.

𝑤′ = 𝑤0 + 𝛼𝜏𝑝𝑒𝑟𝑠𝑜𝑛𝑎𝑙𝑖𝑧𝑒𝑑 + 𝛽𝜏𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑖𝑚𝑎𝑔𝑒 +𝛾𝜏𝑎𝑐𝑐𝑒𝑙𝑒𝑟𝑎𝑡𝑒𝑑𝑣𝑖𝑑𝑒𝑜 , (2) where 𝛼, 𝛽,𝛾 are all scaling factor. In practice, we find we generally have to set 𝛾 = 1 considering that the it is the only weight enables the video generation ability of base LDM. For 𝛼 and 𝛽, users can scaling them to control the impact of different weight residuals.

- 4 EXPERIMENTS

- 4.1 Benchmarks.

To evaluate our approach, we follow previous works, utilizing the widely used UCF-101 [Soomro et al. 2012] for validation. For each category, we generate 24 videos with 16 frames in resolution 512 × 512 and thus generate 24 × 101 videos in total. We apply FVD [Unterthiner et al. 2018] and CLIPSIM [Hessel et al. 2021] as the validation metric. For CLIPSIM, we rely on the CLIP ViT-H/14 LAION-2B [Radford et al. 2021] to compute the mean value of the similarities of the brief caption and all the frames in the video. Following the validation choice in LCM [Luo et al. 2023], we compare AnimateLCM with the teacher model using the DDIM [Song et al. 2020] and DPM-Solver++ [Lu et al. 2022].

### 4.2 Experimental Results

Table 1: Zero-shot video generation comparision on UCF-101.

Qualitative results. Fig. 4 demonstrates the 4-step generation results of our method in text-conditioned video generation with different personalized style models including styles of realistic, 2D anime, and 3D anime, image-conditioned video generation, and layoutconditioned video generation. We also demonstrate the generation results under different numbers of function evaluation (NFEs) in Fig. 3. We demonstrate good visual quality with only 2 inference steps. As the NFE increases, the generation quality increases accordingly, achieving competitive performance with the teacher model with 25 steps.

|Methods<br><br>|FVD ↓ 1-Step 2-Step 4-Step 8-Step|CLIPSIM ↑ 1-Step 2-Step 4-Step 8-Step|
|---|---|---|
| | | |
|DDIM [Song et al. 2020] DPM++ [Lu et al. 2022] Ours Ours-R|4940.83 3218.74 1944.82 1209.88<br><br>2731.37 2093.47 1043.82 932.43 1256.50 1081.26 925.71 910.34 1071.50 790.99 929.79 1081.72<br><br>|4.43 5.26 14.87 24.38 10.48 18.04 26.82 29.50 22.16 25.99 28.89 30.03 25.41 29.39 30.62 30.71<br><br>|

DDIM [Song et al. 2020], and DPM++ [Lu et al. 2022]. AnimateLCM significantly surpasses the baseline methods, especially in the low step regime (1∼4). Additionally, all these metrics of AnimateLCM are evaluated without requiring classifier-free guidance (CFG) [Ho and Salimans 2022] instead of 7.5 CFG strength applied for other baselines, thus saving half of the inference peak memory cost and inference time. Additionally, we show Ours-R, which we replace the

Quantitative Comparison. Table 1 illustrates quantitative metrics comparison for AnimateLCM and strong baseline methods

LDM weights with new weights finetuned on the high-quality personalized image datasets, can achieve even superior performance. It further indicates the effectiveness of our decouple learning approach.

Advanced quantitative comparison. For a more comprehensive evaluating of the ability of AnimateLCM, we apply Vbench [Huang et al. 2024] for a more advanced metric comparision, which including measurements from dozens of perspectives. We can observe from the Table. 2 that, our model as the only video generation support fast generation (typically at lease 5 times faster than all compared methods), still achieves very competitive totoal score.

#### Table 2: Advanced evaluation with Vbench.

Model Name Fast Total Score Quality Score Semantic Score

Pika-1.0 (2024-06) N/A 80.69% 82.92% 71.77% Gen-2 (2023-12) N/A 80.58% 82.47% 73.03% VideoCrafter-1.0 [He et al. 2022] × 79.72% 81.59% 72.22% AnimateLCM (Ours) ✓ 79.42% 82.36% 67.65% OpenSora V1.2 × 79.23% 80.71% 73.30% Show-1 [Zhang et al. 2023] × 78.93% 80.42% 72.98% OpenSoraPlan V1.1 × 77.99% 80.90% 66.38% AnimateDiff-V1 [Guo et al. 2023] × 77.46% 80.24% 66.32% Latte-1 [Ma et al. 2024] × 77.29% 79.72% 67.58% Open-Sora [Zheng et al. 2024] × 75.91% 78.82% 64.28%

Effectiveness of decoupled consistency learning. We validate the effectiveness of our proposed decoupled distillation strategy. For a fair comparison of convergence speed, we train the spatial LoRA weights for 4 hours on an 8 A100 GPUs. We then train our strategy on the video dataset for an additional 4 hours. We train the baseline without decoupled distillation for 8 hours. Our strategy achieves FVD 985.9 and CLIPSIM 27.7 within the training budget while the baseline without the decoupled distillation strategy achieves FVD 1060.6 and CLIPSIM 18.8.

Inference time comparision. Diffusion models require 50 steps with proper CFG values for high-quality generation (50 × 2 model evaluations). Our model can generate videos in 4 steps without CFG (4 model evaluations). Theoretically, our model can achieve acceleration by 504×2 = 25 times. Testing on a single A800 with fp16 mixed precision, our model generates 2-second videos in 963ms, whereas normal diffusion models take 23564ms (24.47 times slower). Note that for the time computation, we exclude the VAE decoding time since it does not belong to the denoising process.

Denoising process visualization. In Fig. 2, we visualize the denoising process of our model as well as that of a conventional video model. Within the same time frame, our model has generated three high-quality videos, while the compared video diffusion model has yet to complete the denoising of a single video.

### 5 CONCLUSIONS AND LIMITATIONS

We present AnimateLCM, achieving computation-efficient personalized style video generation without personalized video data. Its decoupling strategies from two perspectives allows us to achieve fast stylized video generation with smaller training budget and alleviating the need to collect high-quality stylized video data. It might fail to generate samples with good quality with very low steps (e.g., one-step) though.

### ACKNOWLEDGMENTS

This project is funded in part by National Key R&D Program of China Project 2022ZD0161100, by the Centre for Perceptual and

Interactive Intelligence (CPII) Ltd under the Innovation and Technology Commission (ITC)’s InnoHK, by General Research Fund of Hong Kong RGC Project 14204021. Hongsheng Li is a PI of CPII under the InnoHK.

### REFERENCES

Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J Zico Kolter. 2024. Consistency Models Made Easy. arXiv preprint arXiv:2406.14548 (2024).

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial networks. NeurIPS (2014).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai.

2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023).

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent Video Diffusion Models for High-Fidelity Video Generation with Arbitrary Lengths. arXiv preprint arXiv:2211.13221 (2022).

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021).

Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022. Video diffusion models. arXiv:2204.03458 (2022).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21807– 21818.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022. Dpmsolver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems 35 (2022), 5775–5787.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. 2023. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378 (2023).

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. 2024. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048 (2024).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In ICML. PMLR, 8748–8763.

Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. 2024. Motioni2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020). Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency models. arXiv preprint arXiv:2303.01469 (2023).

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. 2012. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402

(2012).

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018).

Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al.

2024. Phased Consistency Model. arXiv preprint arXiv:2405.18407 (2024).

Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. 2023. Magvit: Masked generative video transformer. In CVPR. 10459–10469.

David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. 2023. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818 (2023).

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. 2024. Open-Sora: Democratizing Efficient Video Production for All. https://github.com/hpcaitech/Open-Sora

