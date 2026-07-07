# arXiv:2504.01017v1[cs.CV]1Apr2025

## Scaling Language-Free Visual Representation Learning

David Fan1,∗, Shengbang Tong1,2,∗, Jiachen Zhu1,2, Koustuv Sinha1, Zhuang Liu1,3, Xinlei Chen1, Michael Rabbat1, Nicolas Ballas1, Yann LeCun1,2, Amir Bar1,†, Saining Xie2,†

1FAIR, Meta, 2New York University, 3Princeton University ∗equal contribution, †equal advising

Visual Self-Supervised Learning (SSL) currently underperforms Contrastive Language-Image Pretraining (CLIP) in multimodal settings such as Visual Question Answering (VQA). This multimodal gap is often attributed to the semantics introduced by language supervision, even though visual SSL and CLIP models are often trained on different data. In this work, we ask the question: “Do visual self-supervised approaches lag behind CLIP due to the lack of language supervision, or differences in the training data?” We study this question by training both visual SSL and CLIP models on the same MetaCLIP data, and leveraging VQA as a diverse testbed for vision encoders. In this controlled setup, visual SSL models scale better than CLIP models in terms of data and model capacity, and visual SSL performance does not saturate even after scaling up to 7B parameters. Consequently, we observe visual SSL methods achieve CLIP-level performance on a wide range of VQA and classic vision benchmarks. These findings demonstrate that pure visual SSL can match language-supervised visual pretraining at scale, opening new opportunities for vision-centric representation learning.

Date: April 1, 2025 Project Page: https://davidfan.io/webssl/

1 Introduction

Visual representation learning has evolved along two distinct paths with different training approaches. Language-supervised methods such as Contrastive Language-Image Pretraining (CLIP) (Radford et al., 2021; Zhai et al., 2023) use paired image-text data to learn representations that are enriched with linguistic semantics. Self-Supervised Learning (SSL) methods (Zhang et al., 2016; Chen et al., 2020a; He et al., 2022; LeCun, 2022; Oquab et al., 2023) learn from images alone, without language.

Despite SSL models outperforming languagesupervised models on classic vision tasks such as classification and segmentation (Oquab et al., 2023), they are less commonly adopted in recent multimodal large language models (MLLMs) (Liu et al., 2023a, 2024a; Agrawal et al., 2024; Tong et al., 2024a; Beyer et al., 2024; Li et al., 2024; AI@Meta, 2024). This difference in adoption is partially due to a performance gap in visual question answering (see Figure 1), particularly for OCR & Chart interpretation tasks (Tong et al., 2024a; Shi et al., 2024).

Beyond methodology differences, these approaches

Diameter 1B 7B Params

CLIP DINOv2

+0.7%

Web-SSL

+4.6%

Avg.VQAAccuracy

LVD

(142M samples)

MetaCLIP Web Data

(2B samples)

ImageNet

(1.28M samples)

<1B Data >1B Data

Figure 1 We compare the scaling behavior of visual SSL and CLIP on 16 VQA tasks from the Cambrian-1 suite under different data and model size regimes. Prior visual SSL methods achieved strong performance on classic vision tasks, but have underperformed as encoders for multimodal instruction-tuned VQA tasks. Our results show that with appropriate scaling of models and data, visual SSL can match the performance of language-supervised models across all evaluated domains—even OCR & Chart.

have also been separated by data scale and distribution (Figure 1). CLIP models typically train on

Data: < 1B Model: ≤ 1B Eval: Linear probing / E2E Tuning

ImageNet-1k

CLS

ImageNet/LVD-142M ≤ 1B Parameters

COCO ADE20K

SEG DET

SEG DET

[Figure 1]

[Figure 2]

Data Scaling Model Scaling More Evals

Data: > 1B Model: > 1B Eval: Instruction Tuning w/ LLM

CV-Bench MMVP

MMB AI2D

Web-scale Data

SQA VQA …

> 1B Parameters

RealWorldQA

(MetaCLIP Web Data)

SEED

- Figure 2 Visual SSL 2.0 changes. In this work, we adopt three improvements to the visual SSL pipeline: 1) Training on billion-scale web data, curated through the MetaCLIP pipeline, to move beyond “conventional” datasets; 2) Scaling model architecture from sub-billion parameter models to models exceeding 1 billion parameters; and 3) Incorporating VQA as a complementary evaluation protocol to comprehensively assess visual features. These changes enable us to study visual SSL at a larger scale and observe scaling trends previously unobserved in smaller-scale experiments.

billion-scale image-text pairs from the web (Schuhmann et al., 2022; Chen et al., 2023; Xu et al., 2024b), while SSL methods use million-scale datasets such as ImageNet (Deng et al., 2009) or hundred-million scale data with ImageNet-like distributions (Ridnik et al., 2021; Oquab et al., 2023).

to CLIP. As a result of our empirical study, we contribute several insights:

- • Visual SSL can match and even surpass languagesupervised methods for visual pretraining, on a wide range of VQA tasks—even on languagerelated tasks such as OCR & Chart understanding (Figure 3).
- • Visual SSL scales well with respect to model capacity (Figure 3) and data (Figure 4), indicating that SSL has significant untapped potential.
- • Visual SSL can maintain competitive traditional vision performance on classification and segmentation, even while improving at VQA (Figure 7).
- • Training on a higher ratio of images containing text is especially effective for improving OCR & Chart performance (Question 4). Exploring data composition is a promising direction.

In this work, we investigate a fundamental question: Is language supervision necessary to pretrain visual representations for multimodal modeling? Rather than seeking to replace language-supervised approaches, we aim to understand the intrinsic capabilities and limitations of visual self-supervision at scale for multimodal applications. To conduct a fair comparison, we train SSL models on the same billionscale web data used for state-of-the-art CLIP modelsspecifically the MetaCLIP dataset (Xu et al., 2024b). This approach controls for data distribution differences when comparing visual SSL and CLIP.

For evaluation, we primarily use visual question answering (VQA) as a framework to evaluate SSL models across a diverse set of capabilities at scale. VQA evaluation suites span vision-centric, visual reasoning, and OCR & Chart tasks, and have been shown to be a more diverse testbed for assessing vision encoders (Tschannen et al., 2024; Wan et al., 2024; Fini et al., 2024; Tong et al., 2024a), reflecting the broader perception challenges found in real-world distributions. We adopt the evaluation suite proposed in Cambrian-1 (Tong et al., 2024a), which evaluates performance across 16 tasks spanning 4 distinct categories of VQA: General, Knowledge, OCR & Chart, and Vision-Centric.

This work serves as a proof of concept that offers a compelling vision-centric alternative to the recent CLIP-dominated trend, and opens new opportunities for future research. We plan to open-source our WebSSL vision models, and we hope to inspire the broader community to unlock the full potential of visual SSL in the multimodal era.

2 From Visual SSL 1.0 to 2.0

In this section, we describe our experimental setup, which extends previous SSL works by (1) scaling dataset size to billion-scale images (Section 2.1), (2) scaling model size beyond 1B parameters (Section 2.2), and (3) evaluating vision models using open-ended VQA tasks (Section 2.3), in addition to

We train Web-SSL, a family of visual SSL models ranging from 1 to 7 billion parameters, using the above setting for direct and controlled comparison

classic vision benchmarks such as ImageNet-1k (Deng et al., 2009) and ADE20k (Zhou et al., 2019).

- 2.1 Beyond ImageNet Pretraining

To study whether visual SSL can match the performance of CLIP, we start by adopting the same data that drove CLIP’s success. We thus leverage the MetaCLIP dataset (Xu et al., 2024b,a), which has enabled the most successful open-source reproduction of CLIP to-date.1 We use 2 billion samples from MetaCLIP, which we refer to as MC-2B. We train SSL methods on only the images, and CLIP on the image-text pairs.

This controls for data distribution and size as confounding variables, and enables a fairer comparison of the pretraining methods themselves, while ensuring sufficient data diversity and scale.

- 2.2 Scaling Up Vision Models to Billion Scale

We can also increase model size. Inspired by advancements in scaling language models (Brown et al., 2020; Kaplan et al., 2020; OpenAI, 2022), we train Vision Transformers (ViTs) with 1B, 2B, 3B, 5B, and 7B parameters, on only the images from MC-2B, to study the properties of larger-scale visual SSL models trained on web-scale data. We adapt ViT-g from Oquab et al. (2023) as ViT-1B, and define new configurations for ViT-2B to 7B (Table 1); see Appendix A for model details.

Model Width Depth Heads MLP

- ViT-1B 1536 40 24 6144
- ViT-2B 2688 24 21 10752
- ViT-3B 3072 26 24 12288 ViT-5B 3584 32 28 14336 ViT-7B 4096 32 32 16384

- Table 1 Model architecture details. For consistency, we denote ViT-g from Oquab et al. (2023) as ViT-1B.

- 2.3 Multimodal LLMs as an Evaluation Protocol

In addition to conventional evaluation protocols, such as ImageNet-1k linear probe, we also evaluate our vision encoders using VQA, a flexible and robust evaluation protocol that reflects the diversity of realworld perceptual challenges (Tschannen et al., 2024; Tong et al., 2024a), as shown in Figure 2.

Here, we study all vision encoders using the same controlled setting to ensure fair comparison. Specifically,

1The data used to train the original CLIP is closed-source.

we use the same two-stage visual instruction tuning procedure and data as Cambrian-1 (Tong et al., 2024a). First, a lightweight MLP adapter is added to project the vision encoder features into the same dimensionality as the LLM, and only this MLP adapter is trained. In the second stage, both the MLP adapter and LLM are finetuned. To enable controlled comparison, the vision encoder remains frozen in both stages, and all experiments use the same training recipe as well as Llama-3 8B Instruct (Touvron et al., 2023) backbone. We provide detailed training datasets and hyperparameters in Appendix A.

We then report results on the Cambrian-1 (Tong et al., 2024a) evaluation suite, which is comprised of 16 VQA benchmarks spanning four established domains: General, Knowledge, OCR & Chart, and Vision-Centric. The average VQA performance is the average of the four subcategories. Each subcategory has 4 benchmarks and is equally weighted.

### 3 Scaling Visual SSL

In this section, we explore the scaling behavior of visual SSL models with respect to both model and data size, as a result of training on only images from MC-2B. We focus on DINOv2 (Oquab et al., 2023)

- as the visual SSL method in this section, and discuss MAE (He et al., 2022) in Section 4.

- In Section 3.1, we increase model size from 1B to 7B while keeping the training data fixed at 2 billion MC2B images—unless otherwise denoted. We use the off-shelf training code and recipe for each method, and do not change the recipe for different model sizes in order to control for confounding variables.
- In Section 3.2, we shift our focus to scaling total data seen for a fixed model size, and analyze how performance evolves as the number of images seen during training increases from 1 billion to 8 billion.

3.1 Scaling Model

The intention of scaling model size is both to find the ceiling of visual SSL under this new data regime, and to identify any unique behavior that emerges in larger models.

We thus pretrain DINOv2 ViT models, ranging from 1B to 7B parameters, using 2 billion unlabeled images

- at 224×224 resolution from MC-2B—without highresolution adaptation (Oquab et al., 2023)—to ensure fair comparison with CLIP. We refer to these models as Web-DINO throughout the paper. For a controlled comparison, we also train CLIP models of the same sizes on the same data.

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| |
|---|

- Figure 3 Scaling behavior of Web-DINO and CLIP ViTs trained on MC-2B. The x-axis shows model sizes from 1B to 7B parameters on a log scale. We observe novel “scaling behavior” with Web-DINO models across all categories, with particularly pronounced improvements in the OCR & Chart and Vision-Centric domains as model size increases. In contrast, CLIP models demonstrate limited scaling benefits, with performance saturating at moderate model sizes. The two model families exhibit complementary strengths: CLIP models excel at OCR & Chart VQA, and Web-DINO models are superior at Vision-Centric VQA, while remaining competitive in all other categories.

We evaluate each model with VQA and present the results in Figure 3. We will first discuss the overall performance trend and then turn to specific category performance. To the best of our knowledge, this is the first instance of a vision encoder trained purely with visual self-supervision achieving performance parity with language-supervised encoders on VQA—even in the OCR & Chart category, which is traditionally considered to be highly text-dependent.

Performance trend. We compare the performance trend as model capacity increases in Figure 3. WebDINO’s Average, OCR & Chart, and Vision-Centric VQA performance improves nearly log-linearly with increasing model size, while General and Knowledge improve to a smaller degree. In contrast, CLIP’s performance in all VQA categories largely saturates after 3B parameters. This suggests that while smaller CLIP models may be more data-efficient, this advantage largely dissipates for larger CLIP models. The continual improvement from increasing Web-DINO model capacity also suggests that visual SSL benefits from larger model capacity, and that scaling visual SSL past 7B parameters is a promising direction.

Category-specific performance. In terms of categoryspecific performance, DINO also increasingly outperforms CLIP on Vision-Centric VQA and largely closes the gap with CLIP on OCR & Chart and Average VQA (Figure 3), as model size increases. At 5B parameters and above, DINO can exceed the Average VQA performance of CLIP, despite being trained solely on images and without language supervision. These results suggest that vision-only models, when trained on CLIP-distribution images, can develop strong visual features that are comparable to those of language-supervised vision encoders.

3.2 Scaling Examples Seen

Previously, we focused on single-epoch training, where each of the 2B unique images in MC-2B is seen only once. Here, we investigate the impact of increasing the number of examples seen by training Web-DINO ViT-7B on data ranging from 1 billion to 8 billion images from MC-2B.

As shown in Figure 4, General and Knowledge VQA performance improves incrementally with more examples seen, saturating at 4B and 2B examples respectively. Vision-Centric VQA performance improves sharply from 1B to 2B examples, and saturates beyond 2B examples. In contrast, OCR & Chart is the only category that shows consistent improvement with more examples seen. This suggests that as the model sees more data, it learns a representation that is increasingly well-suited for text-related tasks, yet without marked degradation on other capabilities.

Furthermore, when compared to a CLIP model of the same size (ViT-7B), Web-DINO consistently outperforms CLIP on average VQA performance given the same number of samples seen (Figure 4). Notably, after seeing 8B samples, Web-DINO closes the performance gap with the CLIP model on OCR & Chart VQA tasks. This provides further evidence suggesting that visual SSL models have the potential to scale better than language-supervised models.

Collectively, the results in Figure 3 and 4 indicate that as model size and examples seen increase, visual SSL learns features that are increasingly effective for VQA in general, but especially on OCR & Chart. Our results suggest that CLIP-based models do not hold an absolute advantage compared to visual SSL. In Section 4, we delve deeper into the underlying mechanisms driving this trend.

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| |
|---|

- Figure 4 Scaling up examples seen when training Web-DINO-7B. Performance across different VQA categories as training data increases from 1B to 8B images. While General and Vision-Centric tasks show diminishing returns after 2B images, OCR & Chart tasks demonstrate continued improvement, contributing to steady gains in average performance. Further, Web-DINO consistently outperforms same-size (ViT-7B) CLIP models with different training samples seen. The x-axis plots training data size on a log-scale.

### 4 Scaling Analysis and Findings

In Section 3, we demonstrated that visual SSL models scale well with model size and training set size. These observations raise further questions about the generality and implications of these phenomena. To deepen our understanding, we investigate five key aspects, including whether scaling behavior extends to other vision-only models (Question 1), if SSL models also exhibit scaling behavior on smaller and more conventional data (Question 2), and whether SSL can retain competitive performance on classic vision tasks (Question 3). Additionally, we explore why scaling particularly enhances OCR & Chart performance (Question 4), and highlight emergent properties that arise via scaling visual SSL (Question 5). In this section, we provide a detailed analysis of these findings.

#### Question 1

Does the observed scaling behavior generalize to other visual SSL methods?

In previous sections, we derived our findings from DINOv2, a joint embedding visual SSL method. Here, we extend our analysis to a masked modelling based visual SSL method—Masked Autoencoder (MAE) (He et al., 2022). We train MAE on MC-2B (denoted as Web-MAE) using ViT models ranging from 1B to 5B parameters and compare the results with Web-DINO models in Figure 5.

Web-MAE models exhibit similar scaling behavior to Web-DINO models, with average VQA performance improving consistently as model size increases. Compared to joint embedding methods, Web-MAE models learn features that are particularly well-suited for OCR & Chart tasks but underperform in other

domains. These results suggest that the “scaling behavior” observed in VQA tasks generalizes across different visual SSL methods. We also note that different visual SSL approaches learn distinct representations even when trained under the same conditions, as demonstrated by Web-MAE’s OCR performance.

#### Question 2

Does visual SSL exhibit similar scaling behavior on smaller scale conventional data, such as ImageNet?

We pretrain Web-DINO 1B, 2B, and 3B models for 300 epochs on ImageNet-1k, a conventional pretraining dataset for SSL, following the recipe from (Oquab et al., 2023). We compare these variants to those trained on MC-2B. We evaluate their downstream VQA performance and ImageNet-1k linear probing results. As shown in Figure 6, models pretrained on ImageNet-1k exhibit consistently inferior performance across all the metrics. Moreover, unlike models trained on MC-2B, those trained on ImageNet-1k do not improve with increasing model sizes. This highlights the importance of training visual SSL on more diverse and larger datasets. This echoes recent findings that increasing dataset sizes and diversity drive LLM scaling (Kaplan et al., 2020; Hoffmann et al., 2023; Chowdhery et al., 2022), and also that pretraining data distribution is critical to downstream performance (Liu and He, 2025).

#### Question 3

How do scaled models perform on classic vision tasks?

We evaluate Web-DINO models, ranging from 1B

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

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
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| |
|---|

- Figure5 Web-MAEtrainedonMC-2B. Web-MAE also exhibits consistent scaling behavior as model size increases. Notably, Web-MAE demonstrates better performance in OCR & Chart tasks, achieving higher accuracy than Web-DINO across all model sizes.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

- Figure 6 Comparison of ImageNet-1k and MC-2B Pretraining. Increasing the diversity and scale of pretraining data improves model performance on VQA accuracy and ImageNet linear probing. Unlike MC-2B pretraining, training on ImageNet does not exhibit a clear scaling trend.

to 7B parameters, on classic vision benchmarks including linear probing on ImageNet-1k (Deng et al., 2009), semantic segmentation on ADE20K (Zhou et al., 2019), and depth estimation on NYUv2 (Silberman et al., 2012). Following the evaluation protocol of DINOv2 (Oquab et al., 2023), we freeze the vision encoder; see Appendix A for details. As shown in

- Figure 7, Web-DINO’s performance improves modestly with increasing model size. Web-DINO achieves strong performance across all benchmarks, outperforming MetaCLIP by a significant margin and remaining competitive with off-shelf DINOv2, even outperforming it on ADE20K +ms. Note that the comparison with off-shelf DINOv2 is not exactly apples-to-apples, as we do not use high-resolution adaptation (Oquab et al., 2023), in order to maintain the same input resolution as CLIP. Additionally, the DINOv2 training data has a higher correlation with these classic vision benchmarks, detailed further in Appendix E. These differences suggest that there remains considerable room for further improvement in our model’s classic vision performance.

classic benchmarks remain important, VQA provides a complementary view into model performance via offering a diverse set of tasks that are grounded in real-world perceptual challenges.

#### Question 4

Why does web-scale data improve OCR & Chart performance?

In Section 3, we observed that increasing model size and examples seen leads to unprecedented improvements in OCR & Chart performance for visual SSL models. This is surprising since current off-the-shelf visual SSL methods are notably poor at OCR & Chart understanding compared to language-supervised models (Tong et al., 2024a; Shi et al., 2024).

One possible explanation is that web-scale image datasets already contain a degree of textual information. Unlike object-centric datasets such as ImageNet, images from the web often contain text (e.g. labels, signs, diagrams, etc.). Larger capacity and more data might aid visual SSL models to extract and leverage this textual information.

However, we observe that the scaling behavior in classic vision tasks is less pronounced compared to VQA. This finding, along with insights from previous work (Tong et al., 2024a; Fini et al., 2024; Naeem et al., 2024), reinforces the value of VQA as a comprehensive vision model evaluation framework. While

To test this hypothesis, we apply an off-the-shelf MLLM—SmolVLM2 (Allal et al., 2025)—to identify images containing text. See Figure 8 for qualitative examples and Appendix A for details. This results in

Figure 7 Performance of Web-DINO models on classic vision tasks. All models achieve strong performance across ImageNet1k classification, ADE20K segmentation, and NYU Depth estimation, and all tasks experience moderate improvements from increasing model size from 1B to 7B parameters. Web-DINO outperforms MetaCLIP (HF) and is competitive with DINOv2 (HF). (HF) denotes the largest official Hugging Face released version.

VQA Evaluator Breakdown of OCR & Chart Tasks Method

% of MC-2B AVG General Knowledge

Vision Centric

OCR Chart ChartQA OCRBench TextVQA DocVQA

CLIP 2B 100% 53.0 72.2 48.8 55.0 36.1 32.8 32.9 52.6 26.0 Web-DINO 2B 100% 50.8 72.8 47.1 56.4 26.8 23.3 15.6 49.2 19.0 Web-DINO 2B 50.3% 53.4 (+2.6) 73.0 (+0.2) 51.7 (+4.6) 55.6 (-0.8) 33.2 (+6.4) 31.4 (+8.1) 27.3 (+11.7) 51.3 (+2.1) 23.0 (+4.0) Web-DINO 2B 1.3% 53.7 (+2.9) 70.7 (-2.1) 47.3 (+0.2) 56.2 (-0.2) 40.4 (+13.6) 47.5 (+24.2) 29.4 (+13.8) 52.8 (+3.6) 32.0 (+13.0)

- Table 2 Impact of data filtering on SSL model performance. We compare Web-DINO ViT-2B models trained on MC-2B with different levels of text filtering (full, 50.3%, and 1.3%) against CLIP ViT-2B trained on full MC-2B. OCR & Chart performance improves with progressively aggressive filtering, with the 1.3% filter achieving the best results. Despite receiving zero language supervision, SSL models can surpass CLIP in text-centric tasks while maintaining strong overall performance.

Raw Data Heavy Filter (1.3%)

Light Filter (50.3%)

[Figure 3]

[Figure 4]

[Figure 5]

“Does this image contain

“Does this image contain

charts, tables, or documents with readable text?

any readable text?”

- Figure8 ExamplesoffilteredMC-2Bimages. The Light filter (Middle) identifies images containing text, retaining 50.3% of the images. The Heavy filter (Right) identifies images explicitly containing charts and documents, retaining only 1.3% of MC-2B.

two curated datasets: (i) Light filter: retains 50.3% of Web-DINO and contains images with any textual content. (ii) Heavy filter: retains 1.3% of MC-2B and contains images with charts, tables, or documents.

We train Web-DINO ViT-2B models on these filtered datasets, with each experiment using 2 billion seen examples (meaning filtered datasets undergo multiple epochs). As shown in Table 2, the model trained on lightly filtered data outperforms the full data variant by +6.4% on OCR & Chart, while maintaining strong performance in other categories. The model trained on heavily filtered data performs better and

outperforms even the language-supervised CLIP ViT2B trained on full data by +4.3% on OCR & Chart. Likewise, heavy filtering also improves Average VQA performance, outperforming the full data Web-DINO ViT-2B by +2.6% and even the full data CLIP ViT2B by +0.7%. This means that is it possible for visual SSL models to outperform CLIP models of the same size, with only a fraction of the total data (in this case 1.3% of MC-2B).

The improvement in OCR & Chart from training on heavily filtered data is particularly pronounced for ChartQA (+24.2%), OCRBench (+13.8%), and DocVQA (+13.0%), while performance remains competitive in all other categories. These results demonstrate that self-supervised visual models, when trained on images containing more text in them, can develop high-quality text understanding capabilities without language supervision. It suggests that data composition—rather than purely scale or language supervision—is crucial for developing strong OCR & Chart understanding abilities.

Although it is not surprising that skewing the data in favor of OCR & Chart would improve OCR & Chart capabilities, it is surprising that simple data filtering can outperform language supervision on the full data. This simple proof of concept suggests that similar

[Figure 6]

0.180

0.177

0.175

###### Alignment Score

0.173

0.170

0.168

0.165

###### Train on MC 2B Increase Model Size Train on More Data

0.163

0.160

DINOv2 ViT 1B

Web DINO ViT 1B

Web DINO ViT 2B

Web DINO ViT 3B

Web DINO ViT 5B

Web DINO ViT 7B

Web DINO ViT 7B

Web DINO ViT 7B

Vision Models

Llama 3.1 8B

Llama 3.1 70B 2B Samples 4B Samples 8B Samples

| |
|---|

Figure 9 Alignment score between Web-DINO and LLMs.

Moving from DINOv2 to Web-DINO improves the alignment between the image and the corresponding text representations obtained by LLMs. Increasing model size from 1B to 7B parameters shows gradual improvement, while training on larger data quantities (4B/8B samples) yields the most significant alignment gains.

techniques may be used to help visual SSL bridge future gaps in other capabilities.

#### Question 5

Why can SSL learn strong visual representations for multimodal modeling, without language supervision?

Thus far, we have seen that visual SSL models can not only become competitive with CLIP models, but also that they can excel at tasks previously thought to require language. This raises an important question: why do vision-only models learn features that work well for multimodal models, even in the absence of language supervision?

We hypothesize that SSL models learn features increasingly aligned with language as model size and examples seen increases. Following Huh et al. (2024), we evaluate intrinsic representational alignment by computing a matching metric between the vision encoder and language model, using image-text pairs from the Wikipedia Captions dataset (Srinivasan et al., 2021). We use off-the-shelf DINOv2 (Oquab et al., 2023) and Web-DINO as vision encoders, and off-the-shelf Llama-3.1 8B and 70B (Touvron et al., 2023) as the language models, without any visual instruction tuning nor alignment procedure.

As shown in Figure 9, we observe three key trends: (1) training on more diverse data (MC-2B) improves alignment with LLMs (DINOv2 ViT-1B → WebDINO ViT-1B); (2) increasing the vision model size leads to slightly higher alignment (Web-DINO ViT-

1B → ViT-7B); and (3) seeing more training samples further enhances alignment (Web-DINO ViT-7B trained on 2B samples → 8B samples).

These findings suggest that as model size and, in particular, training samples scale, vision models naturally develop text-sensitive features and achieve strong alignment with LLMs and multimodal tasks, without explicit language supervision.

### 5 The Web-SSL Model Family

Next, we analyze the overall best performing vision encoders using both VQA and classic vision benchmarks. In Table 3, we show the best results of our vision encoders against recent off-the-shelf vision encoders, in terms of VQA and classic vision tasks.

For VQA, all vision encoders—including off-the-shelf models—are evaluated using the same visual instruction tuning setup detailed in Section 2.3, and mainly 224×224 input resolution for the purpose of fair comparison. Because the goal is not to produce a state-ofthe-art MLLM, we did not employ techniques such as unfreezing the vision encoder, resolution tiling (Liu et al., 2024b), and spatial visual aggregator (Tong et al., 2024a).

For classic vision, we follow the evaluation procedure from Oquab et al. (2023) and evaluate linear probe performance on ImageNet-1k (Deng et al., 2009), ADE20K (Zhou et al., 2019), and NYU Depth v2 (Silberman et al., 2012). The input resolution differs between classic vision tasks, but each model tested uses the same exact settings from Oquab et al. (2023). We emphasize that the primary motivation is still to provide controlled insights.

Performance at 224px. Web-DINO can outperform off-the-shelf MetaCLIP in both VQA and classic vision tasks. Web-DINO is even able to match the performance of SigLIP and SigLIP2 on VQA despite seeing 5× less data and receiving no language supervision. In general, Web-DINO outperforms all off-shelf language-supervised CLIP models at traditional vision benchmarks. Although our best Web-DINO model is 7B parameters, the results from Section 3.1 and Section 3.2 suggest that CLIP models saturate beyond moderate model and data sizes, while visual SSL improves progressively with increasing model and data size. Web-DINO also outperforms off-theshelf visual SSL methods, including DINOv2 (Oquab et al., 2023), in all VQA categories. Web-DINO is also competitive in traditional vision benchmarks.

Performance beyond 224px. Next, we discuss the performance of higher resolution models. Following

|Model<br><br>Method<br><br>Pretrain Data<br><br>Pretrain Samples<br><br>Seen Res|MLLM Evaluator<br><br>AVG<br><br>General<br><br>Knowledge<br><br>OCR&Chart<br><br>Vision-Centric|Classic Vision Tasks<br><br>IN1klin.<br><br>ADE20Klin.<br><br>ADE20Kms.<br><br>NYUdlin.1()↓<br><br>NYUdlin.4()↓|
|---|---|---|
|Language-Supervised Models<br><br>SigLIP ViT-SO400M WebLI 45.0B<br><br>224 384|55.4 74.4 48.7 39.5 58.9 60.0 76.3 50.4 53.5 59.7|86.5 36.5 38.0 0.607 0.525<br>87.3 39.5 47.2 0.582 0.438<br><br><br>|
|SigLIP2 ViT-SO400M WebLI 45.0B<br><br>224 384<br><br>|56.3 74.4 50.7 42.1 58.1 62.0 76.6 51.9 58.4 61.0|87.5 41.1 44.2 0.562 0.539<br>88.1 43.5 50.2 0.524 0.469<br>|
|MetaCLIP ViT-G MetaCLIP 12.8B 224<br><br>Visual Self-Supervised Models<br><br>MAE ViT-H ImageNet-1k 2.0B 224<br><br>|54.8 75.5 48.2 37.3 58.4<br><br>45.2 64.6 43.9 20.6 51.7|86.4 38.0 46.7 0.524 0.415<br><br>76.6 33.3 30.7 0.517 0.483|
|I-JEPA ViT-H ImageNet-22k 0.9B 224|44.7 65.4 43.9 21.2 48.4<br><br>|68.8 31.6 34.6 0.548 0.520|
|DINOv2 ViT-g LVD-142M 1.9B 518|47.9 70.2 45.0 21.2 55.3|86.0 49.0 53.0 0.344 0.298|
|224 Web-DINO ViT-7B MC-2B 8.0B 378<br><br>518|55.2 74.5 48.0 39.4 59.1 57.4 73.9 47.7 50.4 57.7 59.9 75.5 48.2 55.1 60.8<br><br>|86.5 42.1 52.6 0.491 0.376<br><br>86.3 42.3 53.1 0.498 0.366<br><br>86.4 42.6 52.8 0.490 0.362<br><br><br>|

NYUdlin.4()↓

NYUdlin.1()↓

Vision-Centric

OCR&Chart

ADE20Kms.

ADE20Klin.

Knowledge

IN1klin.

General

AVG

- Table 3 Comparison with other vision models. Web-DINO ViT-7B achieves competitive performance with CLIP models on VQA without language supervision and surpasses them on traditional vision tasks. Compared to other self-supervised models like DINOv2, Web-DINO significantly narrows the performance gap with CLIP on VQA tasks, particularly excelling in OCR & Chart understanding. These results demonstrate that SSL can effectively produce strong visual representations for both multimodal and classic vision tasks.

Oquab et al. (2023), we additionally fine-tune WebDINO for 20k steps. We do this for resolutions of 378 and 518, to compare against the higher-resolution off-shelf versions of SigLIP as well as DINO. See Appendix C for training details. From 224 to 378 to 518 resolution, Web-DINO improves steadily at average VQA, with notable gains in OCR & Chart performance. Classic vision performance improves modestly with higher resolution. At 384 resolution, Web-DINO trails behind SigLIP. At 518 resolution, Web-DINO is largely able to bridge the gap. The results suggest that Web-DINO may benefit from further increasing high-resolution adaptation.

- 6 Related Work

Visual self-supervised learning methods. Early visual SSL methods explored various pretext tasks for pretraining (Wang and Gupta, 2015; Doersch et al., 2015; Noroozi and Favaro, 2016; Zhang et al., 2016; Gidaris et al., 2018; Balestriero et al., 2023). More recently, research has converged on two primary approaches: joint embedding methods and masked image modeling. Joint embedding methods learn invariant features by aligning representations of different augmented views (He et al., 2019; Misra and Van Der Maaten, 2019; Chen et al., 2020a; Grill et al., 2020; Chen et al., 2020b; Chen and He, 2021; Chen

- et al., 2021; Caron et al., 2021; LeCun, 2022; Chen
- et al., 2022; Garrido et al., 2023), while masked modeling (Zhou et al., 2021; He et al., 2022; Wei et al.,

- 2022; Fan et al., 2023; Assran et al., 2023; Woo et al.,
- 2023; Bar et al., 2024; Bai et al., 2024; Carreira et al.,

2024) learns by predicting masked visual inputs.

Our work complements SSL research focused on pretraining algorithms, by taking off-the-shelf training code and training visual SSL at scale with a controlled experimental setup. In Question 1, we show that the observed scaling behavior generalizes across both joint embedding and masked modeling SSL methods, and is likely not a method-specific phenomena.

Data used to train vision models. Both supervised (He et al., 2016; Xie et al., 2016; Dosovitskiy et al., 2021; Liu et al., 2022) and SSL vision models have traditionally relied on standard datasets such as MNIST (LeCun, 1998), CIFAR-10 (Krizhevsky et al., 2009), and ImageNet (Deng et al., 2009; Ridnik et al., 2021). More recently, self-supervised methods have scaled to larger unlabeled datasets, such as YFCC (Thomee et al., 2016), LVD-142M (Oquab et al., 2023), and IG-3B (Singh et al., 2023); however, these methods still exhibit a significant performance gap compared to language-supervised models on VQA.

In contrast, language-supervised models (Radford

et al., 2021; Zhai et al., 2023; Sun et al., 2023, 2024; Xu et al., 2024b; Tang et al., 2025) leverage significantly larger image-text datasets, from WIT-400M (Radford et al., 2021) to billion-scale web data (Schuhmann et al., 2022; Fang et al., 2024; Xu et al., 2024b; Gadre et al., 2024), with some using up to 100B image-text pairs (Wang et al., 2025). Studies suggest that pretraining data distribution is more critical for downstream performance than specific training methodologies (Fang et al., 2022; Liu and He, 2025).

Our work bridges these paradigms by pretraining SSL models on web-scale data. Through controlled experiments (Section 3 and 4), we show that (1) visual SSL models are sensitive to the training distribution, (2) increasing data diversity and quantity significantly improves performance on a diverse range of VQA tasks, and (3) training on a higher concentration of images containing text is highly effective for improving OCR & Chart understanding.

Evaluating vision models. Classic works have primarily used image classification (LeCun, 1998; Krizhevsky et al., 2009; Deng et al., 2009; Bossard et al., 2014; Hendrycks et al., 2019, 2020) to evaluate learned representations. More recent SSL research has expanded evaluation to include image segmentation (Everingham et al., 2010; Cordts et al., 2016; He et al., 2017; Zhou et al., 2019), depth estimation (Silberman et al., 2012; Geiger et al., 2013; Song et al., 2015), and video classification (Soomro et al., 2012; Goyal et al., 2017a; Baruch et al., 2021). Languagesupervised models (Radford et al., 2021; Zhai et al., 2023), due to their two-tower encoder structure, commonly use zero-shot image classification to assess the quality of learned image and text features.

Our work follows recent proposals (Naeem et al., 2024; Fini et al., 2024; Tong et al., 2024a) to evaluate vision encoders on a broader range of VQA tasks (Goyal et al., 2017b; Yue et al., 2024a; Liu et al., 2024c; Fu et al., 2023; Tao and Xie, 2024; Yue et al., 2024b; xAI, 2024) using MLLMs. These VQA tasks complement traditional vision benchmarks by assessing visual features on a more diverse range of real-world perceptual challenges. As shown in Section 3 and Section 4, we find that visual SSL trained on web-scale data learns representations that continue to improve on VQA benchmarks, and—to a lesser degree—also on traditional vision benchmarks.

- 7 Limitations

In this work, we focus on training visual SSL models without using language. The main limitation of

vision-only models, compared to language-supervised models, is that they do not support zero-shot image classification out of the box. However, by integrating visual SSL models into MLLM frameworks through instruction tuning, we show they can achieve impressive downstream performance across classification and other tasks. Another way to achieve zero-shot image classification is to use LiT-style adaptation (Zhai et al., 2022; Jose et al., 2024), but this is outof-scope for our work as we do not use language supervision. To focus on comparing the vision encoder, we fixed the base LLM for visual instruction tuning to Llama-3 8B Instruct (AI@Meta, 2024). We hypothesize that the findings using other LLM backbones would be similar, however this is not in scope for our work. Additionally, while we demonstrate that visual SSL scales well on MetaCLIP data, we leave the exploration of even larger and/or uncurated datasets to future work.

- 8 Discussion

We show that large-scale visual encoders that are trained with self-supervised language-free objectives can produce high quality visual features for multimodal models. Our results echo the “bitter lesson” (Sutton, 2019) and suggest that imposing less supervision—including language—remains a promising direction for advancing the field of computer vision. We hope our work will inspire further exploration of vision-only approaches, which will enable the construction of next generation vision models that excel at both traditional vision and modern multimodal capabilities.

- 9 Acknowledgements

We thank Ellis Brown, John Nguyen, Junlin Han, Shengyi Qian, Tyler Zhu, Yuexiang Zhai, Druv Pai, Shusheng Yang, Jihan Yang, Muzi Tao, Boyang Zheng, and Anjali Gupta for reviewing this manuscript. We thank Hu Xu and the MetaCLIP paper authors for creating the MetaCLIP dataset. We thank Mido Assran, Mikael Henaff, Daniel Bolya, Hu Xu, Mark Ibrahim, Russ Howes, and Matthew Muckley for their insightful feedback. We thank Michaël Ramamonjisoa and Marc Szafraniec for their help with image segmentation and depth estimation evaluations. Lastly, we thank Ananya Saxena, Cody Olsen, Mack Ward, Maxwell Taylor, Kalyan Saladi, Dev Satpathy, Dinesh Kannappan, Xiaodong Ma, Jacob Kahn, Gabriel Synnaeve, and Shubho Sengupta for infrastructure support.

References

Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Devendra Chaplot, Jessica Chudnovsky, Saurabh Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024. 1

AI@Meta. Llama 3 model card. 2024. 1, 10, 16

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025. 6, 16

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, 2023. 9

Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan L Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. In CVPR, 2024. 9

Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Morcos, Shashank Shekhar, Tom Goldstein, Florian Bordes, Adrien Bardes, Gregoire Mialon, Yuandong Tian, et al. A cookbook of self-supervised learning. arXiv preprint

arXiv:2304.12210, 2023. 9

Amir Bar, Florian Bordes, Assaf Shocher, Mido Assran, Pascal Vincent, Nicolas Ballas, Trevor Darrell, Amir Globerson, and Yann LeCun. Stochastic positional embeddings improve masked image modeling. In ICML, 2024. 9

Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data. In NeurIPS, 2021. 10

Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 1

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In ECCV, 2014. 10

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 3

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 9

João Carreira, Dilara Gokay, Michael King, Chuhan Zhang, Ignacio Rocco, Aravindh Mahendran, Thomas Albert Keck, Joseph Heyward, Skanda Koppula, Etienne Pot, et al. Scaling 4d representations. arXiv preprint arXiv:2412.15212, 2024. 9

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020a. 1, 9

Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2021. 9

Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020b. 9

Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021. 9

Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual languageimage model. In ICLR, 2023. 2

Yubei Chen, Adrien Bardes, Zengyi Li, and Yann LeCun. Bag of image patch embedding behind the success of self-supervised learning. arXiv preprint arXiv:2206.08954, 2022. 9

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive languageimage learning. In CVPR, 2023. 16

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arxiv 2022. arXiv preprint arXiv:2204.02311, 10:1, 2022. 5

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In CVPR, 2016. 10

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 2, 3, 6, 8, 9, 10, 16, 20

Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015. 9

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 9

Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. IJCV, 2010. 10

David Fan, Jue Wang, Shuai Liao, Yi Zhu, Vimal Bhat, Hector Santos-Villalobos, Rohith MV, and Xinyu Li. Motion-guided masking for spatiotemporal representation learning. In CVPR, 2023. 9

Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pre-training (clip). In ICML, 2022. 10

Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. In ICLR, 2024. 10

Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, et al. Multimodal autoregressive pre-training of large vision encoders. arXiv preprint arXiv:2411.14402, 2024.

2, 6, 10

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: a comprehensive evaluation benchmark for multimodal large language models. corr abs/2306.13394 (2023), 2023. 10, 20

Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. In NeurIPS, 2024. 10

Quentin Garrido, Yubei Chen, Adrien Bardes, Laurent Najman, and Yann Lecun. On the duality between contrastive and non-contrastive self-supervised learning. In ICLR, 2023. 9

Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041, 2023. 20

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237, 2013. 10

Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018. 9

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In ICCV, 2017a. 10

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017b. 10

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. In NeurIPS, 2020. 9

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 9

Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In ICCV, 2017. 10

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. arxiv e-prints, art. In CVPR, 2019. 9

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 1, 3, 5, 9

Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Xiaodong Song. Natural adversarial examples. 2021 ieee. In CVPR, 2019. 10

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-ofdistribution generalization. 2021 ieee. In ICCV, 2020. 10

Tuomo Hiippala, Malihe Alikhani, Jonas Haverinen, Timo Kalliokoski, Evanfiya Logacheva, Serafina Orekhova, Aino Tuomainen, Matthew Stone, and John A Bateman. Ai2d-rst: A multimodal corpus of 1000 primary school science diagrams. Language Resources and Evaluation, 55:661–688, 2021. 20

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In NeurIPS, 2023. 5

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 20

Minyoung Huh, Brian Cheung, Tongzhou Wang, and

Phillip Isola. The platonic representation hypothesis. In ICML, 2024. 8

Cijo Jose, Théo Moutakanni, Dahyun Kang, Federico Baldassarre, Timothée Darcet, Hu Xu, Daniel Li, Marc Szafraniec, Michaël Ramamonjisoa, Maxime Oquab, et al. Dinov2 meets text: A unified framework for image-and pixel-level vision-language alignment. arXiv preprint arXiv:2412.16334, 2024. 10

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 3, 5

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 9, 10

Yann LeCun. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/, 1998. 9, 10

Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1): 1–62, 2022. 1, 9

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023a. 1

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024a. 1

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024b. 8

Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023b. 20

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024c. 10, 20

Zhuang Liu and Kaiming He. A decade’s battle on dataset bias: Are we there yet? In ICLR, 2025. 5, 10

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, 2022. 9

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter

Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022. 20

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2023. 20

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, 2022. 20

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021. 20

Ishan Misra and Laurens Van Der Maaten. Self-supervised learning of pretext-invariant representations. in 2020 ieee. In CVPR, 2019. 9

Muhammad Ferjad Naeem, Yongqin Xian, Xiaohua Zhai, Lukas Hoyer, Luc Van Gool, and Federico Tombari. Silc: Improving vision language pretraining with selfdistillation. In ECCV, 2024. 6, 10

Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016. 9

OpenAI. Chatgpt, 2022. 3

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. In TMLR, 2023. 1, 2, 3, 5, 6, 8, 9, 16, 17, 21

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 9, 10

Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses. arXiv preprint arXiv:2104.10972, 2021. 2, 9

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022. 2, 10, 16, 17

Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024. 1, 6

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 6, 8, 10, 16, 20

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019. 20

Mannat Singh, Quentin Duval, Kalyan Vasudev Alwala, Haoqi Fan, Vaibhav Aggarwal, Aaron Adcock, Armand Joulin, Piotr Dollár, Christoph Feichtenhofer, Ross Girshick, et al. The effectiveness of mae pre-pretraining for billion-scale pretraining. In ICCV, 2023. 9

Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In CVPR, 2015. 10

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 10

Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipediabased image text dataset for multimodal multilingual machine learning. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pages 2443–2449, 2021. 8

Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 10

Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, and Xinlong Wang. Evaclip-18b: Scaling clip to 18 billion parameters. arXiv preprint arXiv:2402.04252, 2024. 10

Richard Sutton. The bitter lesson. Incomplete Ideas (blog), 2019. 10

Zineng Tang, Long Lian, Seun Eisape, XuDong Wang, Roei Herzig, Adam Yala, Alane Suhr, Trevor Darrell, and David M. Chan. Tulip: Towards unified languageimage pretraining, 2025. Preprint. 10

Muzi Tao and Saining Xie. What does a visual formal analysis of the world’s 500 most famous paintings tell us about multimodal LLMs? In The Second Tiny Papers Track at ICLR 2024, 2024. 10

Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59 (2):64–73, 2016. 9

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration

of multimodal llms. In NeurIPS, 2024a. 1, 2, 3, 6, 8, 10, 16, 20

Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024b. 16

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, 2024c. 20

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971,

- 2023. 3, 8

Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, and Lucas Beyer. Image captioners are scalable vision learners too. In NeurIPS,

- 2024. 2, 3

Bo Wan, Michael Tschannen, Yongqin Xian, Filip Pavetic, Ibrahim Alabdulmohsin, Xiao Wang, André Susano Pinto, Andreas Steiner, Lucas Beyer, and Xiaohua Zhai. Locca: Visual pretraining with location-aware captioners. arXiv preprint arXiv:2403.19596, 2024. 2

Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. In ICCV,

2015. 9

Xiao Wang, Ibrahim Alabdulmohsin, Daniel Salz, Zhe Li, Keran Rong, and Xiaohua Zhai. Scaling pre-training to one hundred billion data for vision language models. arXiv preprint arXiv:2502.07617, 2025. 10

Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In CVPR, 2022. 9

Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders. In CVPR, 2023. 9

xAI. grok, 2024. 10, 20

Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. arXiv preprint arXiv:1611.05431, 2016. 9

Hu Xu, Po-Yao Huang, Xiaoqing Ellen Tan, Ching-Feng Yeh, Jacob Kahn, Christine Jou, Gargi Ghosh, Omer Levy, Luke Zettlemoyer, Wen-tau Yih, et al. Altogether: Image captioning via re-aligning alt-text. arXiv preprint arXiv:2410.17251, 2024a. 3

Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. In ICLR, 2024b. 2, 3, 10, 17, 21

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024a. 10, 20

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b. 10

Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18123–18133, 2022. 10

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In ICCV, 2023. 1, 10

Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016. 1, 9

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, ChienChin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023. 16

Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. IJCV, 2019. 3, 6, 8, 10, 16, 20

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021. 9

A Implementation Details

Training. For training Web-DINO, Web-MAE, and CLIP models, we closely follow the existing opensource codebases: the official DINOv2 and MAE repositories, and the MetaCLIP codebase which builds on top of the OpenCLIP codebase (Cherti et al., 2023). We use Fully Sharded Data Parallel (FSDP) (Zhao et al., 2023) for distributed training of larger models.

For Web-DINO and CLIP pretraining, we follow the exact recipe and hyperparameters from the original paper for their largest model. For MAE pretraining, we observe that training becomes more prone to divergence as model size increases. To mitigate this, we reduce the learning rate from 2.4e-3 to 1.6e-3 and extend the warmup period to 80K iterations. Table 4 provides a summary of the pretraining hyperparameters.

Model Batch Size Learning Rate Warmup Web-DINO 3072 3.5e-4 100K Web-MAE 4096 1.6e-3 80K CLIP 32768 4e-4 2K

- Table 4 Hyperparameters for Web-DINO, Web-MAE and CLIP.

VQA evaluation. For VQA evaluation, we follow Tong et al. (2024a,b) and use Cambrian-Alignment data for MLP projector training and Cambrian-7M for MLP and LLM fine-tuning. We finetune on top of Llama-3 8B Instruct (AI@Meta, 2024). The vision encoder is frozen throughout finetuning. We excluded LAION (Schuhmann et al., 2022) images from the Cambrian data to comply with safety standards. We first encode the images at the model’s original input resolution using the pretrained vision encoder. Next, we extract features from the final encoder layer. Following prior approaches (Tong et al., 2024a,b), we then resize the resulting token sequence to a fixed length of 576 tokens through bilinear interpolation. This ensures consistency across evaluations despite variations in input image resolutions. We report configurations in Table 5.

Classic vision evaluation. We follow the evaluation procedure in DINOv2 (Oquab et al., 2023) for all classic vision evaluation: linear probe on ImageNet1k (Deng et al., 2009), ADE20K (Zhou et al., 2019), and NYU Depth v2 (Silberman et al., 2012). For ImageNet-1k, we evaluate models with their pretrained image resolution; For ADE20K and NYU Depth v2, we use the settings from Oquab et al. (2023). For ADE20K, we follow DINOv2 and report

the linear and +ms setting. For NYU Depth v2, we report lin. 1 and lin. 4. See the original paper for additional details.

Model architectures. In Table 1, we defined the ViT architectures used in our study. To recap, we first borrowed the ViT-g architecture from Oquab et al. (2023) and named it ViT-1B for consistent notation. We then define 2B, 3B, 5B, and 7B architectures inspired by language model scaling. Specifically, the 2 - 7B architectures are wider than the 1B variant, inspired by language model recipes. Our 7B architecture is almost identical to the Llama-2 7B design, except for the patch embedding layer which is unique to ViTs.

Text filtering. In Question 4, we introduced the “Light” and “Heavy” filters which retain 50.3% and 1.3% of MC-2B respectively. Specifically, we use a small MLLM, SmolVLM2 (Allal et al., 2025), to identify images containing text, using prompts such as “Does this image contain any readable text?”. The intention is not to achieve perfect filtering, but rather to skew the data distribution in the general desired direction. See Figure 8 for a visualization of the filtering process and some examples. This results in two curated datasets:

- (i) Light filter: Retains 50.3% of the original data, primarily consisting of images with some textual content. Prompt used: “Does this image contain any readable text? Answer only yes or no.”
- (ii) Heavy filter: Retains only 1.3% of the data, focusing mainly on charts and documents. Prompt used: “Please think carefully before answering. Does this image contain charts, tables, or documents with readable text? Answer only yes or no.”

B Full Results

We include full results of all experiments presented in Section 3 and Section 4.

B.1 Web-DINO

Scaling up model sizes. We show quantitative results of scaling up the model under VQA evaluation in Table 6 and classic vision evaluation in Table 7. These are the numerical results for Section 3.1.

Scaling up data sizes. We show quantitative results of scaling up the number of data seen with WebDINO ViT-7B on VQA evaluation in Table 8 and

|Backbone LLM|Data Adapter Instruction Tuning<br><br>|Adapter LR WD BS|Instruction Tuning LR WD BS|
|---|---|---|---|
|Llama-3 8B Instruct|Cambrian Adapter Data Cambrian-7M|1.00e-5 0.0 512|4.00e-5 0 512|

- Table 5 Hyperparameters for all VQA experiments. We exclude LAION (Schuhmann et al., 2022) from Cambrian data.

|Vision Backbone<br><br>Model|Average<br><br>|General<br><br>PMME<br><br>MMB<br><br>ISEED<br><br>GQA|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench|
|---|---|---|---|---|---|
|Web-DINO ViT-1B<br>Web-DINO ViT-2B<br>Web-DINO ViT-3B Web-DINO ViT-5B Web-DINO ViT-7B<br>|49.01 50.77 51.71 52.83 53.87|1731.52 65.37 69.92 62.40 1760.80 68.98 71.29 62.89 1757.27 68.04 71.84 63.19 1840.81 70.01 72.39 63.56 1823.76 68.98 73.02 64.22|72.58 35.33 12.30 64.28<br>73.67 31.77 15.90 67.06<br><br><br>73.57 33.00 14.40 67.32 75.06 32.11 12.40 67.77<br>74.61 35.11 14.00 69.43<br>|19.20 9.40 47.41 17.00 23.30 15.60 49.20 19.00<br><br>25.68 17.10 50.45 20.00<br>26.96 22.10 50.64 21.00 28.80 23.59 51.10 22.00<br>|37.33 57.12 64.80 63.16<br>38.00 57.38 65.85 64.41 42.66 56.86 69.49 65.83 44.66 57.64 67.75 69.16 48.00 59.34 69.96 68.58<br>|

Average

PMME

MMB

ISEED

GQA

ISQA

VMMMU

MMathVista

AI2D

ChartQA

OCRBench

TextVQA

DocVQA

MMVP

RealWorldQA

2DCV-Bench

3DCV-Bench

- Table 6 VQA Evaluation: Web-DINO trained on MC-2B with 2 billion images seen.

classic vision evaluation in Table 9. These are the numerical results for Section 3.2.

Scaling down training data. We show VQA evaluation results from training Web-DINO on less diverse data–ImageNet-1k, in Table 10. These are the full results for scaling down training data experiments in Question 2.

- B.2 Web-MAE

We show VQA evaluation results from scaling up MAE trained on MC-2B, in Table 11. These are the full results for Question 1.

- B.3 Scaled CLIP Models

We show VQA evaluation results from scaling up MetaCLIP (Xu et al., 2024b) trained on MC-2B, in Table 12. These are the full results for Section 3.1. In contrast to visual SSL methods in Table 7 and Table 11, CLIP models do not exhibit clear scaling behavior.

- B.4 Text Filtered Models

We provide full results for Question 4. As shown in Table 13, SSL models learn features particularly well-suited for OCR & Chart tasks when trained on datasets with a higher concentration of text-rich images. This suggests that visual SSL is sensitive to the underlying training distribution and can be effectively steered toward specific downstream applications, such as OCR & Chart.

B.5 Baseline Models

In Table 14, we provide full VQA results for the reference off-shelf models that we evaluated in Section 5.

### C High Resolution Adaption of WebSSL

Following Oquab et al. (2023), we further fine-tune our model under higher resolution settings of 378×378 and 518×518 for 20k iterations. We use a batch size of 2048 and a correspondingly lower learning rate of 1.41e-5. All other parameters remain exactly the same as previously specified, including the learning rate warmup ratio, given the total of 10k iterations.

We also provided detailed benchmark results of highresolution adaptation of Web-DINO in Table 15.

### D Evaluation

Table 16 lists evaluation benchmarks used and their purposes.

### E Pretraining Dataset Cards

For reference, in Table 17 we include the data composition of LVD-142M, which was used to train the off-shelf DINOv2 model (Oquab et al., 2023). LVD142M is a carefully curated data mix closely aligned with downstream classic vision evaluation tasks. In comparison, we leverage MetaCLIP data, which is less curated and collected from 15 snapshots of CommonCrawl (CC).

Vision Backbone IN1k lin. ADE20K lin. ADE20K +ms. NYUd lin. 1 (↓) NYUd lin. 4 (↓)

- Web-DINO ViT-1B 84.70 46.60 50.97 0.364 0.345
- Web-DINO ViT-2B 85.16 50.55 52.32 0.351 0.335
- Web-DINO ViT-3B 85.66 50.17 53.12 0.348 0.328 Web-DINO ViT-5B 85.84 49.54 53.27 0.378 0.335 Web-DINO ViT-7B 86.00 49.08 54.65 0.380 0.339

- Table 7 Classic Vision Evaluation: Web-DINO trained on MC-2B with 2 billion images seen.

|Vision Backbone<br><br>Model|Average|General<br><br>PMME<br><br>MMB<br><br>ISEED<br><br>GQA|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench<br><br>|
|---|---|---|---|---|---|
|Web-DINO ViT-7B (1B Data)<br>Web-DINO ViT-7B (2B Data) Web-DINO ViT-7B (4B Data) Web-DINO ViT-7B (8B Data)<br>|51.02 53.87 54.37 55.24|1785.97 68.12 72.54 63.60 1823.76 68.98 73.02 64.22 1827.12 71.39 72.61 63.53 1811.05 71.30 72.14 64.04|73.87 32.88 12.70 66.58<br>74.61 35.11 14.00 69.43 72.73 34.00 18.90 67.09 72.43 35.66 15.20 68.52<br>|23.60 15.20 49.04 19.00 28.80 23.59 51.10 22.00 35.12 30.00 53.19 24.00 35.52 36.40 56.53 29.00|43.33 57.12 68.35 61.08 48.00 59.34 69.96 68.58<br><br>45.33 55.94 69.68 65.00<br>46.00 57.90 70.53 62.08<br>|

Average

PMME

MMB

ISEED

GQA

ISQA

VMMMU

MMathVista

AI2D

ChartQA

OCRBench

TextVQA

DocVQA

MMVP

RealWorldQA

2DCV-Bench

3DCV-Bench

- Table 8 VQA Evaluation: Web-DINO ViT-7B trained on MC-2B with increased number of images seen.

Vision Backbone IN1k lin. ADE20K lin. ADE20K +ms. NYUd lin. 1 (↓) NYUd lin. 4 (↓)

Web-DINO ViT-7B (2B Data) 86.00 49.08 54.65 0.380 0.339 Web-DINO ViT-7B (4B Data) 86.33 47.41 54.66 0.416 0.363 Web-DINO ViT-7B (8B Data) 86.52 42.14 52.55 0.491 0.376

- Table 9 Classic Vision Evaluation: Web-DINO ViT-7B trained on MC-2B with increased number of images seen.

|Vision Backbone<br><br>Model|Average|General<br><br>PMME<br><br>MMB<br><br>ISEED<br><br>GQA<br><br>|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench|
|---|---|---|---|---|---|
|Web-DINO ViT-1B<br>Web-DINO ViT-2B<br>Web-DINO ViT-3B Web-DINO ViT-5B<br>|46.39 45.99 46.43 46.28|1704.30 59.27 66.43 60.12 1666.01 60.13 66.64 60.19 1729.40 60.56 66.99 60.24 1661.25 59.27 67.24 61.10|71.29 32.77 18.70 63.40<br><br>68.71 34.88 12.10 62.07 70.50 31.88 11.70 62.30<br>69.41 31.55 10.90 61.46<br>|17.56 4.90 44.93 14.00<br>18.60 4.39 45.55 14.00<br><br><br>17.52 4.80 45.18 15.00<br>18.72 4.60 45.53 15.00<br>|32.00 52.41 62.81 56.41 32.66 52.67 62.07 57.83 31.33 53.20 62.77 62.50 34.00 53.07 64.57 61.08|

- Table 10 VQA Evaluation: Web-DINO trained on ImageNet-1k.

|Vision Backbone<br><br>Model|Average|General PMME<br><br>MMB<br><br>ISEED<br><br>GQA|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D<br><br>|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench|
|---|---|---|---|---|---|
|Web-MAE ViT-1B<br>Web-MAE ViT-2B<br>Web-MAE ViT-3B Web-MAE ViT-5B<br>|49.19 50.59 50.92 51.50|1736.22 62.02 68.38 60.05 1700.16 63.57 69.21 60.93 1723.85 64.69 69.71 60.94 1710.13 65.12 70.13 61.10|73.27 33.11 12.90 63.92 72.48 32.22 15.50 64.44 72.13 34.33 13.50 65.70 72.63 32.66 13.90 65.67|23.60 16.40 47.84 18.00<br><br>29.00 23.20 48.78 20.00<br>30.92 24.60 48.92 20.00 33.80 26.50 49.60 21.00<br>|36.66 52.81 70.42 60.83 38.00 55.16 67.98 63.91<br>37.33 54.64 64.15 66.91<br>38.00 53.72 66.69 67.91<br>|

- Table 11 VQA Evaluation: Web-MAE trained on MC-2B.

|Vision Backbone<br><br>Model|Average|General<br><br>PMME<br><br>MMB<br><br>ISEED<br><br>GQA<br><br>|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench|
|---|---|---|---|---|---|
|MetaCLIP ViT-1B<br>MetaCLIP ViT-2B<br>MetaCLIP ViT-3B MetaCLIP ViT-5B MetaCLIP ViT-7B<br>|52.30 53.03 53.22 52.52 52.97|1813.70 68.90 69.45 60.35 1787.39 68.81 69.54 61.08 1873.67 68.72 70.33 61.85 1779.03 70.10 70.26 61.53 1827.80 69.93 69.47 61.33|74.07 33.55 12.70 64.41<br>75.16 34.66 20.10 65.38 77.29 32.77 11.80 66.35 72.43 33.44 17.90 66.74 74.91 35.55 16.80 65.15<br>|33.20 34.59 52.15 26.00 32.80 32.90 52.55 26.00 32.16 34.40 54.58 26.00 30.04 32.20 52.49 25.00 32.12 32.10 52.07 25.00|37.33 52.15 65.47 61.83 37.33 52.94 65.19 64.67 35.33 55.55 65.57 65.08 39.33 54.50 64.22 61.16 39.33 54.11 65.08 63.16|

RealWorldQA

2DCV-Bench

3DCV-Bench

MMathVista

OCRBench

TextVQA

ChartQA

VMMMU

DocVQA

Average

MMVP

PMME

ISEED

MMB

AI2D

ISQA

GQA

- Table 12 VQA Evaluation: MetaCLIP trained on MC-2B with 2 billion images seen.

|Vision Backbone<br><br>Model|Average|General<br><br>PMME<br><br>MMB<br><br>ISEED<br><br>GQA<br><br>|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench|
|---|---|---|---|---|---|
|Web-DINO ViT-1B (No Filter)<br><br>Web-DINO ViT-1B (Light Filter)<br><br>Web-DINO ViT-1B (Heavy Filter)<br>Web-DINO ViT-2B (No Filter)<br><br><br>Web-DINO ViT-2B (Light Filter)<br><br><br>Web-DINO ViT-2B (Heavy Filter)<br>|49.01 50.73 49.44 50.77 53.38 53.65|1731.52 65.37 69.92 62.40 1690.89 65.54 70.68 62.63 1593.79 61.40 65.34 59.53 1760.80 68.98 71.29 62.89 1768.67 68.38 71.80 63.24 1743.56 65.29 69.28 61.19|72.58 35.33 12.30 64.28<br><br>70.99 33.89 17.80 63.69<br>71.19 31.33 14.90 64.83<br><br><br>73.67 31.77 15.90 67.06<br>74.16 33.88 31.40 67.38 74.86 32.22 14.50 67.42<br>|19.20 9.40 47.41 17.00 26.12 21.80 50.56 20.00 36.92 24.09 50.09 27.00 23.30 15.60 49.20 19.00 31.40 27.30 51.26 23.00 47.48 29.40 52.80 32.00|37.33 57.12 64.80 63.16 36.00 56.86 64.84 65.75 21.33 53.20 66.53 63.66<br>38.00 57.38 65.85 64.41<br>39.33 56.47 61.13 65.50<br>40.00 54.50 65.85 64.50<br>|

M

h

QA

2D

3D

- Table 13 VQA Evaluation: Web-DINO trained on text filtered MC-2B.

|Vision Backbone<br><br>Model|Average|General<br><br>PMME<br><br>MMB<br><br>ISEED<br><br>GQA|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench|
|---|---|---|---|---|---|
|CLIP Models MetaCLIP ViT-H224px SigLIP ViT-SO400M224px SigLIP ViT-SO400M384px SigLIP2 ViT-SO400M224px SigLIP2 ViT-SO400M384px SSL Models<br><br>DINOv2 ViT-g224px DINOv2 ViT-g378px DINOv2 ViT-g518px I-JEPA ViT-H 224px MAE ViT-H224px|54.91 55.36 59.97 56.32 61.98<br><br>49.25 47.94 47.91 44.78 45.21|1860.58 72.93 70.96 62.22 1807.30 72.76 71.83 62.68 1892.16 73.71 73.00 63.80 1789.26 73.36 72.20 62.60 1895.70 74.57 72.24 64.81<br><br>1785.25 64.86 70.89 62.89 1734.38 64.26 71.50 62.21 1694.08 62.45 70.64 62.87 1598.15 60.01 64.04 57.66 1697.06 56.87 56.41 60.51|77.88 36.88 15.00 67.32<br><br>76.74 35.44 14.00 68.65<br>77.83 33.88 20.00 69.78 74.96 35.55 22.40 69.85 79.27 36.33 19.90 72.24<br><br><br>72.03 32.11 12.40 62.37 71.04 33.11 9.60 63.08 71.29 33.55 11.80 63.37 68.91 34.55 10.20 62.07 70.74 32.11 11.50 61.30|35.60 33.40 55.10 29.00 33.08 40.20 56.61 28.00 54.24 46.40 63.53 50.00 35.76 42.00 59.68 31.00 59.68 52.90 67.15 54.00<br><br>17.96 5.50 47.06 15.00<br><br>17.76 5.00 45.59 15.00<br>18.32 5.10 46.27 15.00<br><br><br>16.72 4.00 42.99 14.00<br>17.40 5.50 45.38 14.00<br>|41.33 53.46 68.53 65.91 47.33 56.99 66.42 64.66<br><br>46.00 58.43 67.37 66.91 44.00 54.24 69.88 64.16 49.33 54.77 70.73 69.00<br>47.33 56.33 65.92 66.08 41.33 56.47 63.79 60.58 37.33 56.60 65.36 61.83 29.33 49.93 57.39 57.16 27.33 53.46 61.19 64.75<br>|

- Table 14 VQA Evaluation: Off-shelf CLIP and SSL models.

|Vision Backbone<br><br>Model|Average|General PMME<br><br>MMB<br><br>ISEED<br><br>GQA<br><br>|Knowledge<br><br>ISQA<br><br>VMMMU<br><br>MMathVista<br><br>AI2D|OCR & Chart<br><br>ChartQA<br><br>OCRBench<br><br>TextVQA<br><br>DocVQA|Vision-Centric<br><br>MMVP<br><br>RealWorldQA<br><br>2DCV-Bench<br><br>3DCV-Bench|
|---|---|---|---|---|---|
|Web-DINO224px Web-DINO378px Web-DINO518px|55.24 57.43 59.91|1811.05 71.30 72.14 64.04 1757.06 70.61 72.59 64.50 1807.08 73.79 72.92 64.78|72.43 35.66 15.20 68.52 72.53 35.11 16.10 67.09 74.36 34.66 14.50 69.43|35.52 36.40 56.53 29.00 52.04 42.19 61.51 46.00 57.28 45.70 64.48 53.00|46.00 57.90 70.53 62.08 38.00 59.08 66.55 67.16 43.33 60.52 70.08 69.41|

- Table 15 VQA Evaluation: Web-DINO ViT-7B adapted to different resolution

Benchmark Eval Citation GQA General VQA Hudson and Manning (2019) SEED General VQA Ge et al. (2023) MME General VQA Fu et al. (2023) MMBench General VQA Liu et al. (2024c) AI2D Knowledge VQA Hiippala et al. (2021) ScienceQA Knowledge VQA Lu et al. (2022) MathVista Knowledge VQA Lu et al. (2023) MMMU Knowledge VQA Yue et al. (2024a) TextVQA OCR & Chart VQA Singh et al. (2019) DocVQA OCR & Chart VQA Mathew et al. (2021) ChartQA OCR & Chart VQA Masry et al. (2022) OCRBench OCR & Chart VQA Liu et al. (2023b) MMVP Vision-Centric VQA Tong et al. (2024c) RealWorldQA Vision-Centric VQA xAI (2024)

- CVBench-2D Vision-Centric VQA Tong et al. (2024a)
- CVBench-3D Vision-Centric VQA Tong et al. (2024a) ImageNet-1k Image Classification Deng et al. (2009) ADE-20k Image Segmentation Zhou et al. (2019) NYU Depth v2 Depth Estimation Silberman et al. (2012)

- Table 16 List of benchmarks used

Task Dataset / Split Images Retrieval Retrieved Final classification ImageNet-22k / – 14,197,086 as is – 14,197,086 classification ImageNet-22k / – 14,197,086 sample 56,788,344 56,788,344 classification ImageNet-1k / train 1,281,167 sample 40,997,344 40,997,344 fine-grained classif. Caltech 101 / train 3,030 cluster 2,630,000 1,000,000 fine-grained classif. CUB-200-2011 / train 5,994 cluster 1,300,000 1,000,000 fine-grained classif. DTD / train1 1,880 cluster 1,580,000 1,000,000 fine-grained classif. FGVC-Aircraft / train 3,334 cluster 1,170,000 1,000,000 fine-grained classif. Flowers-102 / train 1,020 cluster 1,060,000 1,000,000 fine-grained classif. Food-101 / train 75,750 cluster 21,670,000 1,000,000 fine-grained classif. Oxford-IIIT Pet / trainval 3,680 cluster 2,750,000 1,000,000 fine-grained classif. Stanford Cars / train 8,144 cluster 7,220,000 1,000,000 fine-grained classif. SUN397 / train1 19,850 cluster 18,950,000 1,000,000 fine-grained classif. Pascal VOC 2007 / train 2,501 cluster 1,010,000 1,000,000 segmentation ADE20K / train 20,210 cluster 20,720,000 1,000,000 segmentation Cityscapes / train 2,975 cluster 1,390,000 1,000,000 segmentation Pascal VOC 2012 (seg.) / trainaug 1,464 cluster 10,140,000 1,000,000 depth estimation Mapillary SLS / train 1,434,262 as is – 1,434,262 depth estimation KITTI / train (Eigen) 23,158 cluster 3,700,000 1,000,000 depth estimation NYU Depth V2 / train 24,231 cluster 10,850,000 1,000,000 depth estimation SUN RGB-D / train 4,829 cluster 4,870,000 1,000,000 retrieval Google Landmarks v2 / train (clean) 1,580,470 as is – 1,580,470 retrieval Google Landmarks v2 / train (clean) 1,580,470 sample 6,321,880 6,321,880 retrieval AmsterTime / new 1,231 cluster 960,000 960,000 retrieval AmsterTime / old 1,231 cluster 830,000 830,000 retrieval Met / train 397,121 cluster 62,860,000 1,000,000 retrieval Revisiting Oxford / base 4,993 cluster 3,680,000 1,000,000 retrieval Revisiting Paris / base 6,322 cluster 3,660,000 1,000,000

142,109,386

- Table 17 LVD-142M Data Sources. In contrast to LVD-142M, which relies on highly curated data sources drawn from distributions closely aligned with various downstream evaluation tasks (see the table above from Oquab et al. (2023)), our data curation approach adopts the methodology from MetaCLIP (Xu et al., 2024b), utilizing web data collected from 15 snapshots of CommonCrawl (CC) spanning January 2021 through January 2023.

