# arXiv:2404.01367v2[cs.CV]10Dec2024

## Bigger is not Always Better: Scaling Properties of Latent Diffusion Models

Kangfu Mei∗ kmei1@jhu.edu Johns Hopkins University

Zhengzhong Tu† tzz@tamu.edu Texas A&M University

Mauricio Delbracio mdelbra@google.com Google

Hossein Talebi htalebi@google.com Google

Vishal M. Patel vpatel36@jhu.edu Johns Hopkins University

Peyman Milanfar milanfar@google.com Google

Reviewed on OpenReview: https: // openreview. net/ forum? id= 0u7pWfjri5

### Abstract

We study the scaling properties of latent diffusion models (LDMs) with an emphasis on their sampling efficiency. While improved network architecture and inference algorithms have shown to effectively boost sampling efficiency of diffusion models, the role of model size—a critical determinant of sampling efficiency—has not been thoroughly examined. Through empirical analysis of established text-to-image diffusion models, we conduct an in-depth investigation into how model size influences sampling efficiency across varying sampling steps. Our findings unveil a surprising trend: when operating under a given inference budget, smaller models frequently outperform their larger equivalents in generating high-quality results. Moreover, we extend our study to demonstrate the generalizability of the these findings by applying various diffusion samplers, exploring diverse downstream tasks, evaluating post-distilled models, as well as comparing performance relative to training compute. These findings open up new pathways for the development of LDM scaling strategies which can be employed to enhance generative capabilities within limited inference budgets.

### 1 Introduction

Latent diffusion models (LDMs) (Rombach et al., 2022), and diffusion models in general, trained on largescale, high-quality data (Lin et al., 2014; Schuhmann et al., 2022) have emerged as a powerful and robust framework for generating impressive results in a variety of tasks, including image synthesis and editing (Rombach et al., 2022; Podell et al., 2023; Delbracio & Milanfar, 2023; Ren et al., 2023; Qi et al., 2023), video creation (Mei & Patel, 2023; Mei et al., 2023; Wu et al., 2023; Singer et al., 2022), audio production (Liu et al., 2023a), and 3D synthesis (Lin et al., 2023; Liu et al., 2023b). Despite their versatility, the major

∗This work was done during an internship at Google. †This work was done while at Google.

barrier against wide deployment in real-world applications (Du et al., 2023; Choi et al., 2023) comes from their low sampling efficiency. The essence of this challenge lies in the inherent reliance of LDMs on multistep sampling (Song et al., 2021b; Ho et al., 2020) to produce high-quality outputs, where the total cost of sampling is the product of sampling steps and the cost of each step. Specifically, the go-to approach involves using the 50-step DDIM sampling (Song et al., 2021a; Rombach et al., 2022), a process that, despite ensuring output quality, still requires a relatively long latency for completion on modern mobile devices with post-quantization. In contrast to single shot generative models (e.g., generative-adversarial networks (GANs) (Goodfellow et al., 2020)) which bypass the need for iterative refinement (Goodfellow et al., 2020; Karras et al., 2019), the operational latency of LDMs calls for a pressing need for efficiency optimization to further facilitate their practical applications.

Recent advancements in this field (Li et al., 2023; Zhao et al., 2023; Peebles & Xie, 2023; Kim et al., 2023b;a; Choi et al., 2023) have primarily focused on developing faster network architectures with comparable model size to reduce the inference time per step, along with innovations in improving sampling algorithms that allow for using less sampling steps (Song et al., 2021a; Dockhorn et al., 2022; Karras et al., 2022; Lu et al.,

- 2022a; Liu et al., 2023c; Xu et al., 2023). Further progress has been made through diffusion-distillation techniques (Luhman & Luhman, 2021; Salimans & Ho, 2022; Song et al., 2023; Sauer et al., 2023b; Gu et al.,
- 2023; Mei et al., 2024a), which simplifies the process by learning multi-step sampling results in a single forward pass, and then broadcasts this single-step prediction multiple times. These distillation techniques leverage the redundant learning capability in LDMs, enabling the distilled models to assimilate additional distillation knowledge. Despite these efforts being made to improve diffusion models, the sampling efficiency of smaller, less redundant models has not received adequate attention. A significant barrier to this area of research is the scarcity of available modern accelerator clusters (Jouppi et al., 2023), as training high-quality text-to-image (T2I) LDMs from scratch is both time-consuming and expensive—often requiring several weeks and hundreds of thousands of dollars.

In this paper, we empirically investigate the scaling properties of LDMs, with a particular focus on understanding how their scaling properties impact the sampling efficiency across various model sizes. We trained a suite of 12 text-to-image LDMs from scratch, ranging from 39 million to 5 billion parameters, under a constrained budget. Example results are depicted in Fig. 1. All models were trained on TPUv5 using internal data sources with about 600 million aesthetically-filtered text-to-image pairs. Our study reveals that there exist a scaling trend within LDMs, notably that smaller models may have the capability to surpass larger models under an equivalent sampling budget. Furthermore, we investigate how the size of pre-trained text-to-image LDMs affects their sampling efficiency across diverse downstream tasks, such as real-world super-resolution (Saharia et al., 2022; Sahak et al., 2023) and subject-driven text-to-image synthesis (i.e., Dreambooth) (Ruiz et al., 2023).

##### 1.1 Summary

Our key findings for scaling latent diffusion models in text-to-image generation and various downstream tasks are as follows:

Pretraining performance scales with training compute. We demonstrate a clear link between compute resources and LDM performance by scaling models from 39 million to 5 billion parameters. This suggests potential for further improvement with increased scaling. See Section 3.1 for details.

Downstream performance scales with pretraining. We demonstrate a strong correlation between pretraining performance and success in downstream tasks. Smaller models, even with extra training, cannot fully bridge the gap created by the pretraining quality of larger models. This is explored in detail in Section 3.2.

Smaller models sample more efficient. Smaller models initially outperform larger models in image quality for a given sampling budget, but larger models surpass them in detail generation when computational constraints are relaxed. This is further elaborated in Section 3.3.1 and Section 3.3.2.

Sampler does not change the scaling efficiency. Smaller models consistently demonstrate superior sampling efficiency, regardless of the diffusion sampler used. This holds true for deterministic DDIM (Song

- et al., 2021a), stochastic DDPM (Ho et al., 2020), and higher-order DPM-Solver++ (Lu et al., 2022b). For more details, see Section 3.4.

Smaller models sample more efficient on the downstream tasks with fewer steps. The advantage of smaller models in terms of sampling efficiency extends to the downstream tasks when using less than 20 sampling steps. This is further elaborated in Section 3.5.

Diffusion distillation does not change scaling trends. Even with diffusion distillation, smaller models maintain competitive performance against larger distilled models when sampling budgets are constrained. This suggests distillation does not fundamentally alter scaling trends. See Section 3.6 for in-depth analysis.

### 2 Related Work

Scaling laws. Recent Large Language Models (LLMs) including GPT (Brown et al., 2020), PaLM (Anil et al., 2023), and LLaMa (Touvron et al., 2023) have dominated language generative modeling tasks. The foundational works (Kaplan et al., 2020; Brown et al., 2020; Hoffmann et al., 2022) for investigating their scaling behavior have shown the capability of predicting the performance from the model size. They also investigated the factors that affect the scaling properties of language models, including training compute, dataset size and quality, learning rate schedule, etc. Those experimental clues have effectively guided the later language model development, which have led to the emergence of several parameter-efficient LLMs (Hoffmann et al., 2022; Touvron et al., 2023; Zhou et al., 2023; Alabdulmohsin et al., 2024). However, scaling generative text-to-image models are relatively unexplored, and existing efforts have only investigated the scaling properties on small datasets or small models, like scaling UNet (Nichol & Dhariwal, 2021) to 270 million parameters and DiT (Peebles & Xie, 2023) on ImageNet (14 million), or less-efficient autoregressive models (Chen et al., 2020). Different from these attempts, our work investigates the scaling properties by scaling down the efficient and capable diffusion models, i.e. LDMs (Rombach et al., 2022), on internal data sources that have about 600 million aesthetics-filtered text-to-image pairs for featuring the sampling efficiency of scaled LDMs. We also scale LDMs on various scenarios such as finetuning LDMs on downstream tasks (Wang et al., 2021; Ruiz et al., 2023) and distilling LDMs (Mei et al., 2024a) for faster sampling to demonstrate the generalizability of the scaled sampling-efficiency.

Efficient diffusion models. Nichol et al. (Nichol & Dhariwal, 2021) show that the generative performance of diffusion models improves as the model size increases. Based on this preliminary observation, the model size of widely used LDMs, e.g., Stable Diffusion (Rombach et al., 2022), has been empirically increased to billions of parameters (Ramesh et al., 2022; Podell et al., 2023). However, such a large model makes it impossible to fit into the common inference budget of practical scenarios. Recent work on improving the sampling efficiency focus on improving network architectures (Li et al., 2023; Zhao et al., 2023; Peebles & Xie, 2023; Kim et al., 2023b;a; Choi et al., 2023; Mei et al., 2024b) or the sampling procedures (Song et al., 2021a; Dockhorn et al., 2022; Karras et al., 2022; Lu et al., 2022a; Liu et al., 2023c; Xu et al., 2023; Mei et al., 2025). We explore sampling efficiency by training smaller, more compact LDMs. Our analysis involves scaling down the model size, training from scratch, and comparing performance at equivalent inference cost.

Efficient non-diffusion generative models. Compared to diffusion models, other generative models such as, Variational Autoencoders (VAEs) (Kingma & Welling, 2014; Rezende & Mohamed, 2015; Makhzani et al., 2015; Vahdat & Kautz, 2020), Generative Adversarial Networks (GANs) (Goodfellow et al., 2020; Mao et al., 2017; Karras et al., 2019; Reed et al., 2016; Miyato et al., 2018), and Masked Models (Devlin et al., 2019; Raffel et al., 2020; He et al., 2022; Chang et al., 2022; 2023), are more efficient, as they rely less on an iterative refinement process. Sauer et al. (Sauer et al., 2023a) recently scaled up StyleGAN (Karras et al., 2019) into 1 billion parameters and demonstrated the single-step GANs’ effectiveness in modeling text-toimage generation. Chang et al. (Chang et al., 2023) scaled up masked transformer models for text-to-image generation. These non-diffusion generative models can generate high-quality images with less inference cost,

[Figure 1]

[Figure 2]

[Figure 3]

(a) 39M model (b) 83M model (c) 145M model

[Figure 4]

[Figure 5]

[Figure 6]

(d) 223M model (e) 318M model (f) 430M model

[Figure 7]

[Figure 8]

[Figure 9]

(g) 558M model (h) 704M model (i) 2B model

- Figure 1: Text-to-image results from our scaled LDMs (39M - 2B), highlighting the improvement in visual quality with increased model size (note: 39M model is the exception). All images generated using 50-step DDIM sampling and CFG rate of 7.5. We use representative prompts from PartiPrompts Yu et al. (2022), including “a professional photo of a sunset behind the grand canyon.”, “Dogs sitting around a poker table with beer bottles and chips. Their hands are holding cards.”, ‘Portrait of anime girl in mechanic armor in night Tokyo.”, “a teddy bear on a skateboard.”, “a pixel art corgi pizza.”, “Snow mountain and tree reflection in the lake.”, “a propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese.”, “a store front that has the word ‘LDMs’ written on it.”, and “ten red apples.”. Check our supplement for additional visual comparisons.

Params 39M 83M 145M 223M 318M 430M 558M 704M 866M 2B 5B

Filters (c) 64 96 128 160 192 224 256 288 320 512 768 GFLOPS 25.3 102.7 161.5 233.5 318.5 416.6 527.8 652.0 789.3 1887.5 4082.6

Norm. Cost 0.07 0.13 0.20 0.30 0.40 0.53 0.67 0.83 1.00 2.39 5.17

FID ↓ 25.30 24.30 24.18 23.76 22.83 22.35 22.15 21.82 21.55 20.98 20.14 CLIP ↑ 0.305 0.308 0.310 0.310 0.311 0.312 0.312 0.312 0.312 0.312 0.314

Table 1: We scale the baseline LDM (i.e., 866M Stable Diffusion v1.5) by changing the base number of channels c that controls the rest of the U-Net architecture as [c,2c,4c,4c] (See Fig. 2). GFLOPS are measured for an input latent of shape 64 × 64 × 4 with FP32. We also show a normalized running cost with respect to the baseline model. The text-to-image performance (FID and CLIP scores) for all scaled LDMs is evaluated on the COCO-2014 validation set with 30k samples, using 50-step DDIM sampling and Classifier-free Guidance (CFG) with a rate of 7.5. It is worth noting that all the model sizes, and the training and the inference costs reported in this work only refer to the denoising UNet in the latent space, and do not include the 1.4B text encoder and the 250M latent encoder and decoder.

scaled UNet ( ) scaled UNet ( )

scaled UNet ( )

Prompt CLIP

Noise

[Figure 10]

UNet ( )

#### ...

Decoder

Image 5B model

2B model

39M model

- Figure 2: Our scaled latent diffusion models vary in the number of filters within the denoising U-Net. Other modules remain consistent. Smooth channel scaling (64 to 768) within residual blocks yields models ranging from 39M to 5B parameters. For downstream tasks requiring image input, we use an encoder to generate a latent code; this code is then concatenated with the noise vector in the denoising U-Net.

which require fewer sampling steps than diffusion models and autoregressive models, but they need more parameters, i.e., 4 billion parameters.

- 3 Scaling LDMs

We developed a family of powerful Latent Diffusion Models (LDMs) built upon the widely-used 866M Stable Diffusion v1.5 standard (Rombach et al., 2022)1. The denoising UNet of our models offers a flexible range of sizes, with parameters spanning from 39M to 5B. We incrementally increase the number of filters in the residual blocks while maintaining other architecture elements the same, enabling a predictably controlled scaling. Table 1 shows the architectural differences among our scaled models. We also provide the relative cost of each model against the baseline model. Fig. 2 shows the architectural differences during scaling. Models were trained using the web-scale aesthetically filtered text-to-image dataset, i.e., WebLI (Chen

- et al., 2022). All the models are trained for 500K steps, batch size 2048, and learning rate 1e-4. This allows for all the models to have reached a point where we observe diminishing returns. Fig. 1 demonstrates the consistent generation capabilities across our scaled models. We used the common practice of 50 sampling

1We adopted SD v1.5 since it is among the most popular diffusion models https://huggingface.co/models?sort=likes.

52

0.315

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

39M 83M

48

0.310

44

145M 223M 318M 430M 558M 704M 866M

0.305

Performance(CLIP)

Performance(FID)

40

0.300

36

0.295

32

0.290

0.285

2B 5B

28

0.280

24

0.275

0.270

10 M 100 M 1 G

10 M 100 M 1 G

Training Compute

Training Compute

- Figure 3: In text-to-image generation using 50-step DDIM sampling and CFG rate of 7.5, we observe consistent trends across various model sizes in how quality metrics (FID and CLIP scores) relate to training compute (i.e., the total GFLOPS spend on training). Under moderate training resources, training compute is the most relevant factor dominating quality.

1 M 10 M 100 M

Training Compute

18

21

24

27

30

33

36

Performance(FID)

1 M 10 M 100 M

Training Compute

0.270

0.285

0.300

0.315

0.330

0.345

0.360

Performance(LPIPS)

83M

145M 223M 318M 430M 558M 704M 866M 2B

- Figure 4: In 4× real image super-resolution using 50-step DDIM sampling, FID and LPIPS scores reveal an interesting divergence. Model size drives FID score improvement, while training compute most impacts LPIPS score. Despite this, visual assessment (Fig. 5) confirms the importance of model size for superior detail recovery (similarly as observed in the text-to-image pretraining).

steps with the DDIM sampler, 7.5 classifier-free guidance rate, for text-to-image generation. The visual quality of the results exhibits a clear improvement as model size increases.

In order to evaluate the performance of the scaled models, we test the text-to-image performance of scaled models on the validation set of COCO 2014 (Lin et al., 2014) with 30k samples. For downstream performance, specifically real-world super-resolution, we test the performance of scaled models on the validation of DIV2K with 3k randomly cropped patches, which are degraded with the RealESRGAN degradation (Wang et al., 2021).

##### 3.1 Training compute scales text-to-image performance

We find that our scaled LDMs, across various model sizes, exhibit similar trends in generative performance relative to training compute cost, especially after training stabilizes, which typically occurs after 200K iterations. These trends demonstrate a smooth scaling in learning capability between different model sizes. To elaborate, Fig. 3 illustrates a series of training runs with models varying in size from 39 million to 5 billion parameters, where the training compute cost is quantified as the product of relative cost shown in Table 1 and training iterations. Model performance is evaluated by using the same sampling steps and sampling parameters. In scenarios with moderate training compute (i.e., < 1G, see Fig. 3), the generative performance of T2I models scales well with additional compute resources.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

83M 145M 223M 318M 430M

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

LR 558M 704M 866M 2B HR

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

83M 145M 223M 318M 430M

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

LR 558M 704M 866M 2B HR

- Figure 5: In 4× super-resolution using 50-step DDIM sampling, visual quality directly improves with increased model size. As these scaled models vary in pretraining performance, the results clearly demonstrate that pretraining boosts super-resolution capabilities in both quantitative (Fig 4) and qualitative ways. Additional results are given in supplementary material.

- 3.2 Pretraining scales downstream performance

Using scaled models based on their pretraining on text-to-image data, we finetune these models on the downstream tasks of real-world super-resolution (Saharia et al., 2022; Sahak et al., 2023) and DreamBooth (Ruiz et al., 2023). The performance of these pretrained models is shown in Table. 1. In the left panel of Fig. 4, we present the generative performance FID versus training compute on the super-resolution (SR) task. It can be seen that the performance of SR models is more dependent on the model size than training compute. Our results demonstrate a clear limitation of smaller models: they cannot reach the same performance levels as larger models, regardless of training compute.

While the distortion metric LPIPS shows some inconsistencies compared to the generative metric FID (Fig. 4), Fig. 5 clearly demonstrates that larger models excel in recovering fine-grained details compared to smaller models.

The key takeaway from Fig. 4 is that large super-resolution models achieve superior results even after short finetuning periods compared to smaller models. This suggests that pretraining performance (dominated by the pretraining model sizes) has a greater influence on the super-resolution FID scores than the duration of finetuning (i.e., training compute for finetuning).

Furthermore, we compare the visual results of the DreamBooth finetuning on the different models in Fig. 6. We observe a similar trend between visual quality and model size. Please see our supplement for more discussions on the other quality metrics.

- 3.3 Scaling sampling-efficiency

- 3.3.1 Analyzing the effect of CFG rate.

Text-to-image generative models require nuanced evaluation beyond single metrics. Sampling parameters are vital for customization, with the Classifier-Free Guidance (CFG) rate (Ho & Salimans, 2022) directly

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

83M 145M 223M 318M

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Inputs 430M 558M 866M 2B

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

83M 145M 223M 318M

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Inputs 430M 558M 866M 2B

- Figure 6: Visualization of the Dreambooth results (using 50-step DDIM sampling and CFG rate of 7.5) shows two distinct tiers based on model size. Smaller models (83M-223M) perform similarly, as do larger ones (318M-2B), with a clear quality advantage for the larger group. Additional results are given in supplementary material.

[Figure 57]

- (a) 50-step sampling results of the 145M model

[Figure 58]

- (b) 50-step sampling results of the 866M model

- Figure 7: Visualization of text-to-image results with 50-step DDIM sampling and different CFG rates (from left to right in each row: (1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0)). The prompt used is “A raccoon wearing formal clothes, wearing a top hat and holding a cane. Oil painting in the style of Rembrandt.”. We observe that changes in CFG rates impact visual quality more significantly than the prompt semantic accuracy. We use the FID score for quantitative determination of optimal sampling performance (Fig. 8) because it directly measures visual quality, unlike the CLIP score, which focuses on semantic similarity.

influencing the balance between visual fidelity and semantic alignment with text prompt. Rombach et al. (Rombach et al., 2022) experimentally demonstrate that different CFG rates result in different CLIP and FID scores.

Optimal CFG Rate

8.0

80

[Figure 59]

56 64 72 1.5 2.0 3.0 4.0 5.0 6.0 7.0 8.0 optimal

1.5 2.0 3.0 4.0 5.0 6.0 7.0 8.0 optimal

50

72

7.0

64

56

20

Performance(FID)

48

6.0

samplingsteps

48

40

10

5.0

40

32

4.0

32

8

24

3.0

24

6

2.0

16

4

1.5

16

4 6 8 10 20 50

4 6 8 10 20 50

39M83M145M223M318M430M558M704M866M

Sampling Steps 145M model

Sampling Steps 558M model

model size

- Figure 8: The impact of the CFG rate on text-to-image generation depends on the model size and sampling steps. As demonstrated in the left and center panels, the optimal CFG rate changes as the sampling steps increased. To determine the optimal performance (according to the FID score) of each model and each sampling steps, we systematically sample the model at various CFG rates and identify the best one. As a reference of the optimal performance, the right panel shows the CFG rate corresponding to the optimal performance of each model for a given number of sampling steps.

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

0 100 200 300 400 500

Sampling Steps

0

25

50

75

100

125

150

175

200

SamplingCost

3 6 10 14 20 28 38 88 200

Sampling Cost

15

18

21

24

27

30

33

36

Performance(FID)

39M 83M

145M 223M 318M 430M 558M 704M 866M

2B 5B

- Figure 9: Comparison of text-to-image performance of models with varying sizes. The left figure shows the relationship between sampling cost (normalized cost × sampling steps) and sampling steps for different model sizes. The right figure plots the optimal text-to-image FID score among CFG rates of (1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0) as a function of the sampling cost for the same models. Key Observation: Smaller models achieve better FID scores than larger models for a fixed sampling cost. For instance, at a cost of 3, the 83M model achieves the best FID compared to the larger models. This suggests that smaller models can be more efficient in achieving good results with lower costs.

In this study, we find that CFG rate as a sampling parameter yields inconsistent results across different model sizes. Hence, it is interesting to quantitatively determine the optimal CFG rate for each model size and sampling steps using either FID or CLIP score. We demonstrate this by sampling the scaled models using different CFG rates, i.e., (1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0) and comparing their quantitative and qualitative results. In Fig. 7, we present visual results of two models under varying CFG rates, highlighting the impact on the visual quality. We observed that changes in CFG rates impact visual quality more significantly than prompt semantic accuracy and therefore opted to use the FID score for quantitative determination of the optimal CFG rate. performance. Fig. 8 shows how different classifier-free guidance rates affect the FID scores in text-to-image generation (see figure caption for more details).

##### 3.3.2 Scaling efficiency trends.

Using the optimal CFG rates established for each model at various number of sampling steps, we analyze the optimal performance to understand the sampling efficiency of different LDM sizes. Specifically, in Fig. 9, we present a comparison between different models and their optimal performance given the sampling cost

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

83M 145M 223M 318M

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

430M 558M 704M 866M

- (a) Prompt: “A corgi’s head depicted as a nebula.”. Sampling Cost ≈ 6.

[Figure 68]

83M 145M 223M 318M

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

430M 558M 704M 866M

[Figure 73]

[Figure 74]

[Figure 75]

- (b) Prompt: “A pineapple surfing on a wave.”. Sampling Cost ≈ 12.

- Figure 10: Text-to-image results of the scaled LDMs under approximately the same inference cost (normalized cost × sampling steps). Smaller models can produce comparable or even better visual results than larger models under similar sampling cost.

(normalized cost × sampling steps). By tracing the points of optimal performance across various sampling cost—represented by the dashed vertical line—we observe a consistent trend: smaller models frequently outperform larger models across a range of sampling cost in terms of FID scores. Furthermore, to visually substantiate better-quality results generated by smaller models against larger ones, Fig. 10 compares the results of different scaled models, which highlights that the performance of smaller models can indeed match their larger counterparts under similar sampling cost conditions. Please see our supplement for more visual comparisons.

39

39

39M 83M

36

36

33

33

145M 223M 318M 430M 558M 704M 866M

Performance(FID)

Performance(FID)

30

30

27

27

24

24

21

21

18

18

15

15

1 2 3 4 6 8 11 16 23 32 45 64

1 2 3 4 6 8 11 16 23

Sampling Cost

Sampling Cost

- Figure 11: Left: Text-to-image performance FID as a function of the sampling cost (normalized cost × sampling steps) for the DDPM sampler (solid curves) and the DDIM sampler (dashed curves). Right: Textto-image performance FID as a function of the sampling cost for the second-order DPM-Solver++ sampler (solid curves) and the DDIM sampler (dashed curves). Suggested by the trends shown in Fig. 9, we only show the sampling steps ≤ 50 as using more steps does not improve the performance.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

1 2 3 4 6 10

Sampling Cost

20

22

24

26

28

30

32

34

Performance(FID)

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

2 3 4 6 10 20 40 60 100 200

Sampling Cost

- 18
- 19
- 20
- 21
- 22
- 23
- 24

Performance(FID)

83M

145M 223M 318M 430M 558M 704M 866M

- Figure 12: Super-resolution performance vs. sampling cost for different model sizes. Left: FID scores of super-resolution models under limited sampling steps (less than or equal to 20). Smaller models tend to achieve lower (better) FID scores within this range. Right: FID scores of super-resolution models under a larger number of sampling steps (greater than 20). Performance differences between models become less pronounced as sampling steps increase.

##### 3.4 Scaling sampling-efficiency in different samplers

To assess the generalizability of observed scaling trends in sampling efficiency, we compared scaled LDM performance using different diffusion samplers. In addition to the default DDIM sampler, we employed two representative alternatives: the stochastic DDPM sampler (Ho et al., 2020) and the high-order DPMSolver++ (Lu et al., 2022b).

Experiments illustrated in Fig. 11 reveal that the DDPM sampler typically produces lower-quality results than DDIM with fewer sampling steps, while the DPM-Solver++ sampler generally outperforms DDIM in image quality (see the figure caption for details). Importantly, we observe consistent sampling-efficiency trends with the DDPM and DPM-Solver++ sampler as seen with the default DDIM: smaller models tend to achieve better performance than larger models under the same sampling cost. Since the DPM-Solver++ sampler is not designed for use beyond 20 steps, we focused our testing within this range. This finding demonstrates that the scaling properties of LDMs remain consistent regardless of the diffusion sampler used.

##### 3.5 Scaling downstream sampling-efficiency

Here, we investigate the scaling sampling-efficiency of LDMs on downstream tasks, specifically focusing on the super-resolution task. Unlike our earlier discussions on optimal sampling performance, there is limited

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

42

39M 83M

undistill 10-step undistill 20-step undistill 50-step distilled 4-step

24

39

36

145M 223M 318M 430M 558M 704M 866M distilled

33

Performance(FID)

22

Accelerationfactor

Performance(FID)

30

27

20

24

18

21

18

16

15

0.5 1.0 2.0 4.0 6.0 12.0 26.0 48.0

200 400 600 800 Model Size (million parameters)

Sampling Cost

- Figure 13: Distillation improves text-to-image performance and scalability. Left: Distilled Latent Diffusion Models (LDMs) consistently exhibit lower (better) FID scores compared to their undistilled counterparts across varying model sizes. The consistent acceleration factor (approx. 5×) indicates that the benefits of distillation scale well with model size. Right: Distilled models using only 4 sampling steps achieve FID scores comparable to undistilled models using significantly more steps. Interestingly, at a sampling cost of 7, the distilled 866M model performs similarly to the smaller, undistilled 83M model, suggesting improved efficiency.

literature demonstrating the positive impacts of SR performance without using classifier-free guidance. Thus, our approach directly uses the SR sampling result without applying classifier-free guidance. Inspired from Fig. 4, where the scaled downstream LDMs have significant performance difference in 50-step sampling, we investigate sampling efficiency from two different aspects, i.e., fewer sampling steps [4,20] and more sampling steps (20,250]. As shown in the left part of Fig. 12, the scaling sampling-efficiency still holds in the SR tasks when the number of sampling steps is less than or equal to 20 steps. Beyond this threshold, however, larger models demonstrate greater sampling-efficiency than smaller models, as illustrated in the right part of Fig. 12. This observation suggests the consistent sampling efficiency of scaled models on fewer sampling steps from text-to-image generation to super-resolution tasks.

##### 3.6 Scaling sampling-efficiency in distilled LDMs.

We have featured the scaling sampling-efficiency of latent diffusion models, which demonstrates that smaller model sizes exhibit higher sampling efficiency. A notable caveat, however, is that smaller models typically imply reduced modeling capability. This poses a challenge for recent diffusion distillation methods (Luhman & Luhman, 2021; Salimans & Ho, 2022; Song et al., 2023; Sauer et al., 2023b; Gu et al., 2023; Mei et al., 2024a; Luo et al., 2023; Lin et al., 2024) that heavily depend on modeling capability. One might expect a contradictory conclusion and believe the distilled large models sample faster than distilled small models. In order to demonstrate the sampling efficiency of scaled models after distillation, we distill our previously scaled models with conditional consistency distillation (Song et al., 2023; Mei et al., 2024a) on text-to-image data and compare those distilled models on their optimal performance.

To elaborate, we test all distilled models with the same 4-step sampling, which is shown to be able to achieve the best sampling performance; we then compare each distilled model with the undistilled one on the normalized sampling cost. We follow the same practice discussed in Section 3.3.1 for selecting the optimal CFG rate and compare them under the same relative inference cost. The results shown in the left part of Fig. 13 demonstrate that distillation significantly improves the generative performance for all models in 4-step sampling, with FID improvements across the board. By comparing these distilled models with the undistilled models in the right part of Fig. 13, we demonstrate that distilled models outperform undistilled models at the same sampling cost. However, at the specific sampling cost, i.e., sampling cost ≈ 8, the smaller undistilled 83M model still achieves similar performance to the larger distilled 866M model. The observation further supports our proposed scaling sampling-efficiency after diffusion distillation.

### 4 Conclusion

In this paper, we investigated scaling properties of Latent Diffusion Models (LDMs), specifically through scaling model size from 39 million to 5 billion parameters. We trained these scaled models from scratch on a web-scale text-to-image dataset and then finetuned the pretrained models for downstream tasks. Our findings unveil that, under identical sampling costs, smaller models frequently outperform larger models, suggesting a promising direction for accelerating LDMs in terms of model size. We further show that the sampling efficiency is consistent in multiple axes. For example, it is invariant to various diffusion samplers (stochastic and deterministic), and also holds true for distilled models. We believe this analysis of scaling sampling efficiency would be instrumental in guiding future developments of LDMs, specifically for balancing model size against performance and efficiency in a broad spectrum of practical applications.

Limitations and future work. This work utilizes visual quality inspection alongside established metrics like FID and CLIP scores. We opted to avoid human evaluations due to the immense number of different combinations needed for the more than 1000 variants considered in this study. However, it is important to acknowledge the potential discrepancy between visual quality and quantitative metrics, which is actively discussed in recent works (Zhang et al., 2021; Jayasumana et al., 2024; Cho et al., 2023).

Claims regarding the scalability of latent diffusion models are made specifically for the particular model family studied in this work (Rombach et al., 2022). Extending this analysis to other model families, particularly those incorporating transformer-based backbones such as DiT (Peebles & Xie, 2023; Mei et al., 2023), SiT (Ma et al., 2024), MM-DiT (Esser et al., 2024), and DiS (Fei et al., 2024), and cascaded diffusion models such as Imagen3 (Baldridge et al., 2024) and Stable Cascade (Pernias et al., 2023), would be a valuable direction for future research.

### 5 Acknowledgments

Vishal M. Patel was supported by NSF CAREER award 2045489. We are grateful to Keren Ye, Jason Baldridge, Kelvin Chan for their valuable feedback. We also extend our gratitude to Shlomi Fruchter, Kevin Murphy, Mohammad Babaeizadeh, and Han Zhang for their instrumental contributions in facilitating the initial implementation of the latent diffusion models.

### References

Ibrahim M Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, and Lucas Beyer. Getting vit in shape: Scaling laws for compute-optimal model design. Advances in Neural Information Processing Systems, 2024. 3

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. ArXiv preprint,

- 2023. 3

Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Kelvin Chan, Yichang Chen, Sander Dieleman, Yuqing Du, Zach Eaton-Rosen, et al. Imagen 3. arXiv preprint arXiv:2408.07009,

- 2024. 13

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 3

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, 2022. 3

Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. ArXiv preprint, 2023. 3

Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, Proceedings of Machine Learning Research, 2020. 3

Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. ArXiv preprint, 2022. 5

Jaemin Cho, Yushi Hu, Roopal Garg, Peter Anderson, Ranjay Krishna, Jason Baldridge, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-image generation. ArXiv preprint, 2023. 13

Jiwoong Choi, Minkyu Kim, Daehyun Ahn, Taesu Kim, Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Jae-Joon Kim, and Hyungjun Kim. Squeezing large-scale diffusion models for mobile. ArXiv preprint, 2023. 2, 3

Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. Featured Certification. 1

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019. 3

Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Genie: Higher-order denoising diffusion solvers. Advances in Neural Information Processing Systems, 2022. 2, 3

Hongyang Du, Ruichen Zhang, Dusit Niyato, Jiawen Kang, Zehui Xiong, Dong In Kim, Xuemin Sherman Shen, and H Vincent Poor. Exploring collaborative distributed diffusion-based ai-generated content (aigc) in wireless networks. IEEE Network, (99), 2023. 2

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 13

Zhengcong Fei, Mingyuan Fan, Changqian Yu, and Junshi Huang. Scalable diffusion models with state space backbone. arXiv preprint arXiv:2402.05608, 2024. 13

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, (11), 2020. 2, 3

Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference {\&} Generative Modeling, 2023. 2, 12

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, 2022. 3

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. ArXiv preprint, 2022. 7

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 2, 3, 11

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. ArXiv preprint, 2022. 3

Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. ArXiv preprint, 2024. 13

Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, et al. Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. In Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023. 2

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. ArXiv preprint,

2020. 3

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, 2019. 2, 3

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 2022. 2, 3

Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. On architectural compression of text-to-image diffusion models. ArXiv preprint, 2023a. 2, 3

Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. Bk-sdm: Architecturally compressed stable diffusion for efficient text-to-image generation. In Workshop on Efficient Systems for Foundation Models@ ICML2023, 2023b. 2, 3

Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. 3

Hang Li, Chengzhi Shen, Philip Torr, Volker Tresp, and Jindong Gu. Self-discovering interpretable diffusion latent directions for responsible text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12006–12016, 2024. 27

Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. ArXiv preprint,

2023. 2, 3

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 1

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. ArXiv preprint, 2024. 12

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 1, 6

Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audioldm: Text-to-audio generation with latent diffusion models. ArXiv preprint, 2023a. 1

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023b. 1

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. ArXiv preprint, 2023c. 2, 3

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 2022a. 2, 3

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. ArXiv preprint, 2022b. 3, 11

Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. ArXiv preprint, 2021. 2, 12

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. ArXiv preprint, 2023. 12

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. ArXiv preprint, 2024. 13

Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial autoencoders. ArXiv preprint, 2015. 3

Xudong Mao, Qing Li, Haoran Xie, Raymond Y. K. Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, 2017. 3

Kangfu Mei and Vishal Patel. Vidm: Video implicit diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, number 8, 2023. 1

Kangfu Mei, Mo Zhou, and Vishal M Patel. T1: Scaling diffusion probabilistic fields to high-resolution on unified visual modalities. ArXiv preprint, 2023. 1, 13

Kangfu Mei, Mauricio Delbracio, Hossein Talebi, Zhengzhong Tu, Vishal M Patel, and Peyman Milanfar. Codi: Conditional diffusion distillation for higher-fidelity and faster image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024a. 2, 3, 12, 23

Kangfu Mei, Luis Figueroa, Zhe Lin, Zhihong Ding, Scott Cohen, and Vishal M Patel. Latent feature-guided diffusion models for shadow removal. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 4313–4322, 2024b. 3

Kangfu Mei, Nithin Gopalakrishnan Nair, and Vishal M Patel. Improving conditional diffusion models through re-noising from unconditional diffusion priors. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2025. 3

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 23

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings, 2018. 3

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, Proceedings of Machine Learning Research, 2021. 3

Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, and Youngjung Uh. Understanding the latent space of diffusion models through the lens of riemannian geometry. Advances in Neural Information Processing Systems, 36:24129–24142, 2023. 27

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 2, 3, 13

Pablo Pernias, Dominic Rampas, Mats L Richter, Christopher J Pal, and Marc Aubreville. Würstchen: An efficient architecture for large-scale text-to-image diffusion models. arXiv preprint arXiv:2306.00637,

2023. 13

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. ArXiv preprint, 2023. 1, 3

Chenyang Qi, Zhengzhong Tu, Keren Ye, Mauricio Delbracio, Peyman Milanfar, Qifeng Chen, and Hossein Talebi. Tip: Text-driven image processing with semantic and restoration instructions. ArXiv preprint,

2023. 1

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 2020. 3

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. ArXiv preprint, 2022. 3

Scott E. Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, JMLR Workshop and Conference Proceedings, 2016. 3

Mengwei Ren, Mauricio Delbracio, Hossein Talebi, Guido Gerig, and Peyman Milanfar. Multiscale structure guided diffusion for image deblurring. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1

Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, JMLR Workshop and Conference Proceedings, 2015. 3

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022. 1, 2, 3, 5, 8, 13

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2, 3, 7

Hshmat Sahak, Daniel Watson, Chitwan Saharia, and David Fleet. Denoising diffusion probabilistic models for robust image super-resolution in the wild. ArXiv preprint, 2023. 2, 7

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, (4), 2022. 2, 7

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022,

2022. 2, 12, 23 Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. ArXiv preprint, 2023a. 3

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. ArXiv preprint, 2023b. 2, 12

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open largescale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 2022. 1

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. ArXiv preprint, 2022. 1

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021a. 2, 3

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021b. 2

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. 2, 12, 23 Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bash-

lykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, 2023. 3

Arash Vahdat and Jan Kautz. NVAE: A deep hierarchical variational autoencoder. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 3

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind superresolution with pure synthetic data. In IEEE/CVF International Conference on Computer Vision Workshops, ICCVW 2021, Montreal, BC, Canada, October 11-17, 2021, 2021. 3, 6

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1

Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. ArXiv preprint, 2023. 2, 3

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-toimage generation. ArXiv preprint, 2022. 4

Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive learning for text-to-image generation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, 2021. 13

Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. ArXiv preprint, 2023. 2, 3

Yanqi Zhou, Nan Du, Yanping Huang, Daiyi Peng, Chang Lan, Da Huang, Siamak Shakeri, David So, Andrew M Dai, Yifeng Lu, et al. Brainformers: Trading simplicity for efficiency. In International Conference on Machine Learning. PMLR, 2023. 3

### A Scaling the text-to-image performance

In order to provide detailed visual comparisons for Fig. 1 in the main manuscript, Fig. 14, Fig. 15, and Fig. 16 show the generated results with the same prompt and the same sampling parameters (i.e., 50-step DDIM sampling and 7.5 CFG rate).

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

83M 145M 223M 318M 430M

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

558M 704M 866M 2B 5B

(a) Prompt: “a professional photo of a sunset behind the grand canyon.”

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

83M 145M 223M 318M 430M

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

558M 704M 866M 2B 5B

(b) Prompt: “Dogs sitting around a poker table with beer bottles and chips. Their hands are holding cards.”

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

83M 145M 223M 318M 430M

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

558M 704M 866M 2B 5B

(c) Prompt: ‘Portrait of anime girl in mechanic armor in night Tokyo.”

- Figure 14: Text-to-image results from our scaled LDMs (83M - 5B), highlighting the improvement in visual quality with increased model size. 19

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

83M 145M 223M 318M 430M

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

558M 704M 866M 2B 5B

(a) Prompt: “a teddy bear on a skateboard.”

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

83M 145M 223M 318M 430M

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

558M 704M 866M 2B 5B

(b) Prompt: “a pixel art corgi pizza.”

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

83M 145M 223M 318M 430M

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

558M 704M 866M 2B 5B

(c) Prompt: “Snow mountain and tree reflection in the lake.”

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

83M 145M 223M 318M 430M

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

558M 704M 866M 2B 5B

(a) Prompt: “a propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese.”

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

83M 145M 223M 318M 430M

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

558M 704M 866M 2B 5B

(b) Prompt: “a store front that has the word ‘LDMs’ written on it.”

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

83M 145M 223M 318M 430M

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

558M 704M 866M 2B 5B

(c) Prompt: “ten red apples.”

### B Scaling downstream performance

To provide more metrics for the super-resolution experiments in Fig. 4 of the main manuscript, Fig. 18 shows the generative metric IS for the super-resolution results. Fig. 18 shows the visual results of the superresolution results in order to provide more visual results for the visual comparisons of Fig. 5 in the main manuscript.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

- 23
- 24
- 25
- 26
- 27

Performance(IS)

1 M 10 M 100 M

Training Compute

36

33

Performance(FID)

30

27

24

21

18

1 M 10 M 100 M

Training Compute

- Figure 17: For super-resolution, we show the trends between the generative metric IS and the training compute still depend on the pretraining, which is similar to the trends between the generative metric FID and the training compute.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

83M 145M 223M 318M 430M

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

LR 558M 704M 866M 2B HR

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

83M 145M 223M 318M 430M

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

LR 558M 704M 866M 2B HR

- Figure 18: In 4× super-resolution, visual quality directly improves with increased model size. As these scaled models vary in pretraining performance, the results clearly demonstrate that pretraining boosts superresolution capabilities.

### C Scaling sampling-efficiency in distilled LDMs

Diffusion distillation methods for accelerating sampling are generally derived from Progressive Distillation (PD) Salimans & Ho (2022) and Consistency Models (CM) Song et al. (2023). In the main paper, we have shown that CoDi Mei et al. (2024a) based on CM is scalable to different model sizes. Here we show other investigated methods, i.e., guided distillation Meng et al. (2023), has inconsistent acceleration effects across different model sizes. Fig. 19 shows guided distillation results for the 83M and 223M models respectively, where s16 and s8 denote different distillation stages. It is easy to see that the performance improvement of these two models is inconsistent.

Fig. 20 shows the visual results of the CoDi distilled models and the undistilled models under the same sampling cost to demonstrate the sampling-efficiency.

38

83M s16

83M s8 83M s4 83M undistill

36

34

Performance(FID)

32

30

28

26

24

22

4 5 6 7 8 9 10

Sampling Steps

40

38

36

Performance(FID)

34

32

30

28

26

223M s16

223M s8 223M s4 223M undistill

24

4 5 6 7 8 9 10

Sampling Steps

- Figure 19: Left: Guided distillation on the 83M model for text-to-image generation. Right: Guided distillation on the 224M model for text-to-image generation.

[Figure 188]

866M 4-step distilled

[Figure 189]

866M 4-step undistilled

[Figure 190]

83M 50-step undistilled

(a) Prompt: “A corgi’s head depicted as a nebula.”. Sampling Cost ≈ 7.

[Figure 191]

866M 4-step distilled

[Figure 192]

866M 4-step undistilled

[Figure 193]

83M 50-step undistilled

(b) Prompt: “a cat drinking a pint of beer.”. Sampling Cost ≈ 7.

- Figure 20: We visualize text-to-image generation results of the tested LDMs under approximately the same inference cost.

### D Scaling the sampling-efficiency

To provide more visual comparisons additional to Fig. 10 in the main paper, Fig. 21, Fig.22, and Fig. 23 present visual comparisons between different scaled models under a uniform sampling cost. This highlights that the performance of smaller models can indeed match their larger counterparts under similar sampling cost.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

83M 145M 223M 318M

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

430M 558M 704M 866M

(a) Prompt: “a cat drinking a pint of beer.”. Sampling Cost ≈ 10.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

83M 145M 223M 318M

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

430M 558M 704M 866M

(b) Prompt: “a teddy bear on a skateboard.”. Sampling Cost ≈ 10.

- Figure 21: We visualize text-to-image generation results of the tested LDMs under approximately the same inference cost. We observe that smaller models can produce comparable or even better visual results than larger models under similar sampling cost (model GFLOPs × sampling steps).

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

83M 145M 223M 318M

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

430M 558M 704M 866M

(a) Prompt: “a propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese.”. Sampling Cost ≈ 14.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

83M 145M 223M 318M

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

430M 558M 704M 866M

(b) Prompt: “Dogs sitting around a poker table with beer bottles and chips.”. Sampling Cost ≈ 14.

- Figure 22: We visualize text-to-image generation results of the tested LDMs under approximately the same inference cost. We observe that smaller models can produce comparable or even better visual results than larger models under similar sampling cost (model GFLOPs × sampling steps).

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

83M 145M 223M 318M

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

430M 558M 704M 866M

(a) Prompt: “a pixel art corgi pizza.”. Sampling Cost ≈ 18.

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

83M 145M 223M 318M

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

430M 558M 704M 866M

(b) Prompt: “Snow mountain and tree reflection in the lake.”. Sampling Cost ≈ 18.

- Figure 23: We visualize text-to-image generation results of the tested LDMs under approximately the same inference cost. We observe that smaller models can produce comparable or even better visual results than larger models under similar sampling cost (model GFLOPs × sampling steps).

### E Scaling interpretability of text prompt interpolatation

Text prompt interpolation is widely recognized as a way to evaluate the interpretability of text-to-image models in recent works (Li et al., 2024; Park et al., 2023). In Figure 24, we show the text-prompt interopolation results of models in different sizes and visualize their sampling results. Specifically, we use two distinct prompts A and B and interpolate their CLIP embeddings as αA + (1 − α)B,α ∈ [0,1], to generate intermediate text-to-image results. A clear pattern emerges: larger models leads to more semantically coherent and visually plausible interpolations compared to their smaller counterparts. The figure demonstrates the 2B model’s superior ability to accurately interpret interpolated prompts, as evidenced by its generation of a tablet computer with a touch pen.

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

145M α = 1.0

145M α = 0.8

145M α = 0.5

145M α = 0.2

145M α = 0.0

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

318M α = 1.0

318M α = 0.8

318M α = 0.5

318M α = 0.2

318M α = 0.0

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

430M α = 1.0

430M α = 0.8

430M α = 0.5

430M α = 0.2

430M α = 0.0

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

866M α = 1.0

866M α = 0.8

866M α = 0.5

866M α = 0.2

866M α = 0.0

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

2B α = 1.0 2B α = 0.8 2B α = 0.5 2B α = 0.2 2B α = 0.0

(a) Prompt A: “a old computer”. Prompt B: “a fancy mobile phone”.

- Figure 24: We visualize the text-prompt interpolation results of scaled models in different sizes. Each row shows the results of the same model with different interpolation fraction αA + (1 − α)B. All results are sampled with the same 20-step DDIM sampler and CFG of 7.5.

