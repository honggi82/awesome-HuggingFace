# arXiv:2409.14128v2[cs.CV]26Nov2024

## Present and Future Generalization of Synthetic Image Detectors

Pablo Bernabeu-P´erez Barcelona Supercomputing Center

pablo.bernabeu@bsc.es

Enrique Lopez-Cuena Barcelona Supercomputing Center

enrique.lopez@bsc.es

Dario Garcia-Gasulla Barcelona Supercomputing Center

dario.garcia@bsc.es

### Abstract

The continued release of increasingly realistic image generation models creates a demand for synthetic image detectors. To build effective detectors we must first understand how factors like data source diversity, training methodologies and image alterations affect their generalization capabilities. This work conducts a systematic analysis and uses its insights to develop practical guidelines for training robust synthetic image detectors. Model generalization capabilities are evaluated across different setups (e.g. scale, sources, transformations) including real-world deployment conditions. Through an extensive benchmarking of stateof-the-art detectors across diverse and recent datasets, we show that while current approaches excel in specific scenarios, no single detector achieves universal effectiveness. Critical flaws are identified in detectors, and workarounds are proposed to enable the deployment of real-world detector applications enhancing accuracy, reliability and robustness beyond the limitations of current systems.

### 1. Introduction

Synthetic image generation is presenting challenges regarding visual information integrity, mitigation of misinformation, and, trust and rights in digital environments. Due to these concerns, correctly attributing synthetic content has become a social demand and a top scientific priority. Recent legislation aligns with this context, mandating the identification and notification of synthetic digital content [52].

To address these needs, synthetic image detection (SID) has become locked in a race with synthetic image generation (SIG) [32]. SID aspires to win by developing universal detectors [11, 43], but their generalization capacity remains uncertain. Meanwhile, new SIG models join the race every month, moving forward in realism and posing new challenges to SID models. This work studies the relation be-

tween SIG and SID, by first analyzing the impact of training conditions on SID generalization (§4). The learnt lessons are applied to train a baseline for evaluating the generalization capacity on deployment conditions. This includes variations in data and model source (who used the SIG and which SIG was used) and scaling factors (§5). The last set of experiments (§6) include an updated benchmark on recent detectors, using synthetic data produced by the latest generators, under an optimized image scaling policy. Finally, the ethical considerations related to SID research and development, including when and how should detectors be publicly released, are discussed (§7).

Findings indicate current methods are insufficient for reliable SID, as no tested model generalizes universally. Factors like rescaling play a major role in detector performance, exposing a vector of attack for malicious actors. While some models suffer major degradations, others benefit from a resized input, emphasizing the importance of choosing the right preprocessing techniques. Lastly, detectors perform much worse on private models, like DALLE and Midjourney, compared to open models, highlighting the crucial role of open science for synthetic attribution. This work illustrates how, as of today, generalization should never be assumed in the field of SID.

### 2. Related Work

Previous work on SID has largely focused on GANgenerated content [8, 23, 24, 41, 53, 63], primarily due to their historical prevalence and relative speed. However, recent studies reveal that GAN-focused detectors often fail to identify content from modern diffusion models [39, 54]. While several recent works have addressed the detection of diffusion-based content [2, 14, 25, 36, 43, 45, 57, 60, 65, 66], which now produces the most perceptually convincing synthetic images, their generalization ability under different conditions remains mostly untested.

Across all families of detection methods, frequency

domain-based approaches are commonly used to detect synthetic content, revealing generation artifacts [9, 13]. Some methods leverage Fast Fourier Transform analysis to capture characteristic patterns [5, 50], while another recent work [18] explores wavelet-based features specifically tailored for diffusion outputs. Deep learning architectures such as CNNs [12, 47, 49] and Visual Transformers (ViTs) [2, 36] have been used to learn hierarchical synthetic patterns, with CLIP-based methods further enhancing detection through semantic [43] and intermediate feature analysis [30]. Models combining textual and visual features have also been adopted for SID; while [10] applies prompt tuning to detect deepfakes, by approaching detection as a visual question-answering problem, [58] performs contrastive learning via text guidance. Hybrid models combine multiple detection signals to improve generalization, such as dual-stream networks analyzing texture and frequency artifacts [59] and CLIP features fused with low-level image statistics [48]. Finally, local feature analysis is used to examine texture contrast patterns [64] and intrinsic dimensionality properties [39] to provide complementary signals.

AI-generated image detector models are usually trained using data from a single source and evaluated on datasets from multiple sources to assess their generalization capacity, [7, 11, 14, 25, 43, 44, 65, 66]. Among the various sources of bias that have been examined, image format and resolution stand out as key factors. In [14], authors highlight the impact of the resizing operation, a common practice in deep learning to adjust images to the network’s input resolution. The study presented in [25] highlights biases associated with JPEG compression and image size. Authors demonstrate a size bias affecting detector performance, with detectors generally performing better on natural images that differ significantly in size from the generated images used in training. This observation aligns with findings in [15], where it is demonstrated that dataset choice significantly impacts detection performance. Meanwhile, other relevant factors for generalization remain to be studied, including model family, model release date, and dataset source.

### 3. Methods

To examine detector biases arising from training methodology, we employ a fixed architecture (see §3.1), train it using six image datasets (see §3.2) and evaluate with fifteen additional datasets (see §3.3). To enable full reproducibility of the work, our codebase 1, training datasets 2 and model weights 3 for our best detector are publicly released.

- 1https://github.com/HPAI-BSC/SuSy
- 2https://huggingface.co/datasets/HPAI-BSC/SuSy-Dataset
- 3https://huggingface.co/HPAI-BSC/SuSy

#### 3.1. Architecture

The two popular architectural choices for building a SID are training a direct classifier, or using the features extracted from a pre-trained model. Both CNNs [25, 44, 66] and ViTs [15, 43, 60, 65] have been considered for those purposes, both of them performing competitively. For our experimentation, we choose a ResNet [28] trained as a direct classifier, as this has shown competitive and robust results [7, 25, 66] while being a lightweight architecture, allowing evaluation at scale. The staircase design proposed in [40] is used (12.7M parameters), which feeds features extracted at different blocks into a multi-layer perceptron (see Appendix A).

Detectors are commonly trained either on image patches or on downsampled images, as processing entire highresolution images is computationally intensive and the most discriminating features are typically low-level. For each image, we select five 224×224 patches exhibiting the highest contrast in their grey-level co-occurrence matrix [27]. These patches are individually processed through the network, producing per-patch predictions that must be aggregated to obtain image-level decisions. Various combination strategies and their impact on detection performance are studied in §5.2.

Regardless of evaluating at patch or image level, detection performance metrics are chosen based on the number of datasets being analyzed. We employ recall when evaluating performance on a single dataset (either authentic or synthetic), focusing exclusively on the model’s effectiveness in identifying the class at hand. This avoids misleading interpretations that could arise from metrics considering both positive and negative classes. For multi-dataset classification scenarios, we utilize macro accuracy, which provides an unweighted mean of per-class accuracy, ensuring fair evaluation across all classes regardless of sample size.

#### 3.2. Train Datasets

The training experiments detailed in §4 utilize two types of datasets: authentic real-world images sourced from COCO [34] and synthetic AI-generated images from DALLE3 [21], SD1.X [55], SDXL [20], MJ 1/2 [51] and MJ 5/6 [22]. These represent different versions of the three most popular image generators: DALLE, StableDiffusion and Midjourney. To ensure balanced class representation, COCO and SD1.X are undersampled to a maximum of 5,435 images. The pre-existing train, validation and test splits are respected, defaulting to a standard 60%-20%-20% random split distribution when such partitions are not available. For the SDXL dataset, the realistic-2.2 split was used for training and validation purposes, while the realistic-1 split was used for testing. Further details regarding release dates, image formats, resolutions and dataset split sizes are available in Appendix C.

[Figure 1]

Figure 1. Examples of the In-the-wild dataset.

#### 3.3. Benchmarking Datasets

To evaluate SID models we use fifteen datasets: eleven produced and gathered by others, two produced by others but gathered by us, and two produced by us. Image resolution distributions and visual samples are provided in Appendices G and H, respectively.

The datasets produced by others include two subsets of 5,000 randomly selected authentic images: scenes depicting people from Flickr30k [61] and natural and humanmade landmarks from GLDv2 [56]. Additionally, nine synthetic datasets from the Synthbuster superset [5] provide 1,000 images each, generated using a common set of prompts across both models included in our training (e.g. SDXL, DALLE3) and models outside our training set (e.g. DALLE2, Firefly).

The In-the-wild dataset, as shown in Figure 1, comprises both authentic and synthetic images gathered from online sources by the authors. The authentic split contains 121 images manually collected from Reddit (from communities that forbid AI content) and Flickr (from uploads prior to 2020), while the synthetic split consists of 99 photorealistic AI-generated images sourced from Civitai and Reddit’s synthetic content communities.

Finally, we generate two 8,192 synthetic image datasets: SD3, generated using Stable Diffusion 3-Medium [4], a MMDiT text-to-image model, and FLUX.1, created with FLUX.1-dev [31], a 12B parameter model that combines MMDiT and DiT [33] architectures. Additional details are provided in Appendix D.

| |DALLE3 SD1.X SDXL MJ 1/2 MJ 5/6<br><br>|Avg.|
|---|---|---|
|DALLE3 SD1.X SDXL MJ 1/2 MJ 5/6|97.70 27.59 70.19 50.73 97.02 49.76 98.30 68.23 39.65 40.36 51.27 33.45 97.57 59.14 67.97 31.39 17.63 73.14 99.07 51.51 91.76 26.13 64.75 62.69 99.25<br><br>|68.64 59.26 61.88 54.55 68.92|
|Avg.|64.38 40.62 74.78 62.26 71.22| |

Table 1. On each row, patch-level recall of single-class models for synthetic datasets. In bold, performance on the training dataset.

### 4. Train Experiments

This section examines how different training strategies affect model generalization. For consistent experimental comparison, all models share identical architecture (§3.1) and hardware setup (Appendix B). The training is capped at 20 epochs with a 2-epoch patience early stopping based on validation accuracy. The datasets described in §3.2 are augmented using horizontal flips with 50% probability, while additional transformations are analyzed in § 4.3.

#### 4.1. Single-class Models

We evaluate relationships between SIG models by training binary classifiers, using each synthetic dataset in §3.2 as a positive class and COCO as the negative class. These singleclass detectors are then tested on the remaining datasets to assess cross-model generalization (see Table 1).

While single-class detectors achieve excellent recall (over 97%) on their target class, performance drops substantially when tested on other datasets. SIG model age emerges as the dominant factor affecting generalization performance. When evaluating newer detectors on older generators, we observe severe performance degradation, as shown in the last row of Table 1, where detectors trained on SD1.X and MJ 1/2 (both from 2022) show the lowest average values. This pattern likely stems from older generators producing more pronounced artifacts, which newer detectors struggle to identify without specific training. Conversely, detectors trained on recent datasets show better cross-SIG generalization, as evidenced by the higher average values for DALLE3, SDXL and MJ 5/6 in the last column. Paradoxically, this suggests that more realistic generators enhance the robustness and reduce the bias of detectors. In contrast, SIG family has a weak effect on generalization. The detector trained on SDXL is below average when tested on SD1.X. Likewise, the SID model trained on MJ 1/2 is not particularly accurate on MJ 5/6. The effect of image format is also inconclusive.

#### 4.2. Multi-class Models

Multi-class detectors offer richer decision boundaries compared to single-class detectors, which tend to collapse [16] i.e. defaulting to predicting only one class. To explore the

| |Auth. DALLE3 SD1.X SDXL MJ 1/2 MJ 5/6|
|---|---|
|Single Binary 6 Class|- 97.70 98.30 97.57 99.07 99.25 94.85 99.64 98.98 99.22 99.60 99.74 97.39 99.76 97.89 99.30 99.91 99.97<br><br>|

Table 2. Patch level recall for Single: five models trained on each dataset separately (i.e. Table 1 diagonal). Binary: model trained with all synthetic datasets merged. Six-class: Multi-class model trained for the recognition task. Best in bold.

effects of this distinction on generalization, we train a binary classifier merging all synthetic data sources from §3.2 into a single synthetic class, including 14,323 synthetic images and an analogous amount drawn from COCO to compose the authentic class. We also train a six-class recognition model using the original splits defined in §3.2. To obtain binary classifications from the six-class model, we take argmax of the output probabilities, where all samples labeled as belonging to a synthetic class are considered equal predictions of the synthetic class. An alternative threshold mechanism was explored, with minimal impact on performance, and its results are reported in Appendix E.

Results in Table 2 show a good performance from both the binary and the six-way classifiers on all synthetic datasets. Better than single models, which means visual features of synthetic detectors are mutually beneficial for SID. In general, the six-way classifier outperforms all, with the only exception of one of the oldest and most distinct datasets (i.e. SD1.X) (lowest generalization in Table 1).

#### 4.3. Image Alteration Methods

Image transformations, while essential for storage optimization and transmission cost reduction, can significantly alter images and may be exploited by malicious actors to mask synthetic content. If image analysis models are not robust to these transformations, their utility in real-world scenarios becomes minimal. To evaluate this robustness, we test the six-class model from the previous section under several common transformations from the Albumentations library [6]: blur (AdvancedBlur and GaussianBlur), brightness and gamma alterations (RandomBrightnessContrast and RandomGamma), and JPEG compression, all using default parameters.

For a complete assessment, we train five multi-class models, each with a different transformation applied to its training set, and evaluate these alongside our original sixclass model across all transformations and unaltered images. The results are presented in Table 3 using multiclass macro accuracy, where any misclassification between synthetic classes is counted as an error. This metric was selected instead of the previously used binary metrics, as binary classification consistently achieved over 99% accuracy, limiting its ability to distinguish model performance

| |None Bright γ JPEG ABlur GBlur<br><br>|Avg.|
|---|---|---|
|6 Class Bright γ JPEG ABlur GBlur<br><br>|90.90 86.66 90.60 90.19 81.56 54.73<br>91.28 89.68 91.13 90.10 84.61 63.55 91.52 87.51 91.30 90.02 85.57 65.22<br><br><br>87.82 83.15 87.79 86.21 78.42 55.29 90.13 86.23 90.12 88.15 88.04 81.54<br>88.94 84.02 88.65 87.37 86.78 81.88<br>|82.44 85.06 85.19 79.78 87.37 86.27<br><br>|
|Avg.|90.10 86.21 89.93 88.67 84.16 67.04| |

Table 3. Patch-level accuracy of a six-class recognition model when trained on one alteration method and evaluated on all. In bold, performance on the alteration used for training. Last column: model average across all transformations. Bottom row: average performance of all models for each transformation.

in a multi-class context.

- Table 3 shows blur is the transformation that most im-

pacts detector performance. GaussianBlur, which causes drops in accuracy of over 7 points, is also the hardest transformation in training, showing the lowest diagonal score. However, both blur-trained models achieve the highest cross-transformation accuracies, demonstrating effective generalization and making blur a valuable addition to the training process.

5. Deployment Experiments

To study the impact of deployment factors on generalization, we use SuSy, a multi-class model trained with the setup described in §4. Training data from §3.2 is augmented with all transformations from §4.3, each applied with a 20% chance. Using this model generalization to images from new and external data sources is explored in §5.1. The aggregation of patch-level SID predictions into image-level decisions is considered in §5.2. Finally, §5.3 studies the impact of input resolution changes on model generalization.

5.1. Generalization to Source

The SuSy (Patch) column of Table 4 shows the results of evaluating SuSy under disjoint sets of data (see §3.3). For authentic datasets, three new sources of images are added, with varied results. The model generalizes well in Flickr30k, moderately in GLDv2 and poorly for In-the-wild images.

The middle section of Table 4 shows generalization on datasets from generators seen during the training stage (see §3.2). These datasets are from the same SIG models but generated by different users. Variations in SIG configurations, prompts and post-processing, may introduce significant biases. Nonetheless, generalization of SuSy is good, reaching a recall between 74% and 88% in all cases.

The third set of experiments, reported at the bottom of

- Table 4, considers datasets generated by models unseen during training. Performance in this set has a large variance,

##### Data SIG SuSy SuSy Source Model (Patch) (Image) Authentic Data Sources

Flickr30k None 91.81 94.48 GLDv2 None 68.37 71.80 In-the-wild None 30.91 27.27 Synthetic Data Sources (Models in train)

- Synthbuster SD 1.3 88.56 91.80

- Synthbuster SD 1.4 88.50 91.80 Synthbuster MJ V5 74.22 78.40 Synthbuster SD XL 79.22 83.80 Synthbuster DALLE-3 87.02 92.50 Synthetic Data Sources (Models not in train) Synthbuster Glide 52.78 53.20 Synthbuster SD 2 68.32 70.40 Synthbuster DALLE-2 24.50 19.70 Synthbuster Firefly 53.04 53.50 Authors SD 3 91.51 95.23 Authors FLUX.1 94.37 97.05 In-the-wild Unknown 90.51 91.92

- Table 4. Top: unseen authentic image datasets. Middle: unseen synthetic datasets produced by models seen during training. Bottom: synthetic datasets from unseen models. Recall at patch level and five-patch majority voting at image level. Best in bold.

with models reaching recalls as high as 94% and as low

- as 24%. The impact of model family on generalization is inconsistent: SuSy excels on SD3, performs adequately on SD2 and struggles with DALLE2, despite being trained on versions of Stable Diffusion and DALLE.

#### 5.2. Image Decision Boundary

While SID models operate on small image patches, realworld applications typically require whole-image predictions. To address this gap, we analyze the top five patches selected based on texture complexity, as described in §3.1.

We evaluated two aggregation strategies: majority voting of patch predictions and averaging patch logits before classification. Although both approaches demonstrate improvements over single-patch, majority voting consistently outperforms across datasets, with its results shown in the last column of Table 4. With this method, high-performing datasets showed further improvements, while poorly performing ones saw minimal gains, and those scoring below random chance experienced slight degradation. These findings highlight both the advantages and limitations of decision boundary tuning.

#### 5.3. Scale Generalization

Image resizing is a widespread image alteration, and almost impossible to prevent. It can alter frequency artifacts and defects that SID models rely on, decreasing their per-

[Figure 2]

Figure 2. Recall of SuSy on authentic and synthetic evaluation datasets, under different scaling factors.

formance. To assess the extent of this factor, we evaluate SuSy using images scaled at six different sizes (224 to 1440). First, if the image is not already square, equal padding is added on both sides of the shorter dimension to center it. Then the squared image is resized to the specified dimensions using bilinear interpolation. Using the evaluation datasets described in §3.3, which originally follow a diverse distribution of sizes (see Appendix G), the recall for each dataset is computed individually (see §3.1). Then, for each of the six image scales, the results for authentic and synthetic classes are averaged separately, allowing the monitoring of both accuracy and balance in detection. This experiment is reproduced in the benchmarking analysis of §6, for comparison with other SID models.

As shown in Figure 2, SuSy is not severely affected by resolution changes, only at higher rates. It achieves better combined results (around 140) at lower resolutions, with rescaling at 224, 400 and 512 being equally competitive. As resolution increases there is an increasing bias towards synthetic predictions. To ensure consistent performance in real-world applications, where images may have undergone prior resizing, we recommended including standardized rescaling in preprocessing pipelines.

#### 5.4. Human Evaluators

To measure the performance of SID models against human visual assessment, we use the In-the-wild dataset, containing both authentic and synthetic images (see §3.3 for details). We ask 10 volunteers aged 22-30 who have social media accounts and are likely to be exposed to digital

|In-the-wild|Authentic Synthetic<br><br>|
|---|---|
|Avg. Human Evaluator SuSy (Patch) SuSy (Image)<br><br>|72.82 69.14 72.56 64.24 75.21 69.70|

Table 5. In-the-wild recall by SuSy and human evaluators, best in bold. For SuSy, average performance at patch level, fivepatch majority voting at image level.

[Figure 3]

[Figure 4]

Figure 3. Recall of SID on authentic and synthetic evaluation datasets, under different scaling factors.

media and AI-generated content, to discriminate between both In-the-wild versions (authentic and synthetic). To ensure unbiased results, images were presented in random order and the evaluators were not informed about the distribution of the data. All participants viewed the images on the same IPS LCD display (1920x1200 resolution, 400 nits brightness) in a controlled lighting environment. Participants took an average of 15 minutes to label the 210 images, with no time constraints imposed. Results are reported in

- Table 5. SuSy outperforms the average human evaluator at image level, using the aggregation mechanism of §5.2 and the best resolution studied in §5.3.
- 6. Benchmarking Experiments

To complete this study on the generalization capacity of SID, we test the performance of ten different models (most made available through SIDBench [46]). Table 6 showcases the performance of the best six models (over 140 combined recall): LGrad [49], GramNet [38] and DIMD [30], which use CNNs as feature extractors, each with a unique emphasis on different image characteristics, together with Rine [30], DeFake [14] and FatFormer [35], based on transformer models. Further details on their architecture are given in Appendix F, which also includes results for the other tested detectors: CNNDetect [53], Dire [54], FreqDetect [23], and UnivFD [43].

#### 6.1. Rescaling

Given the crucial role of rescaling, the performance of current detectors is first assessed using the experimentation introduced in §5.3, providing insights on the generalization of SID under scale changes and pointing towards the ideal setup for each detector. The models displayed in the top row of Figure 3 are highly sensitive to any scale modifications, with their performance consistently deteriorating after rescaling (i.e. optimal performance without resizing). This sensitivity to scale changes creates a security vulnerability that malicious actors could exploit, compromising the detectors’ reliability. Moreover, these models demonstrate a notable bias toward the authentic class, achieving suboptimal recall scores for synthetic images (63% for two models, while the third performs below random chance).

In contrast, the detectors shown in the bottom row of Figure 3 demonstrate resilience to some scale variations (i.e. optimal performance includes resizing). DeFake and LGrad perform optimally at lower resolutions (224, 400 or 512), similarly to SuSy (see Figure 2), while GramNet excels at higher resolutions (768 or 1024). Their optimal rescaled input resolution enhances their resilience and enables the optimization of deployment pipelines. However, these models differ significantly in their prediction balance: DeFake shows stronger performance in synthetic class detection, LGrad excels in authentic class identification, while GramNet and SuSy achieve a more balanced performance across both categories.

SIG Model Year DIMD FatFormer Rine DeFake LGrad SuSy GramNet Avg. Best Resolution Native Native Native 224 400 400 768

Authentic Data Sources

Flickr30k - 2014 99.92 100.0 99.54 93.62 99.82 94.76 99.94 98.23 COCO - 2017 100.0 99.60 100.0 91.33 77.63 - 87.12 92.61 GLDv2 - 2020 96.54 99.92 77.42 78.32 98.66 82.62 100.0 90.50 In-the-wild - 2024 96.69 96.77 95.87 46.28 71.90 74.38 65.29 78.17

Average Authentic 98.29 99.07 93.21 77.39 87.00 83.92 88.09 Synthetic Data Sources

Synthbuster Glide 2021 6.10 68.10 83.60 86.50 53.50 53.30 68.80 59.99 mj-tti MJ V1/V2 2022 2.87 55.29 14.57 75.83 42.60 - 63.58 42.46

- Synthbuster SD 1.3 2022 100.0 88.20 99.90 86.20 81.10 87.00 90.00 90.34

- Synthbuster SD 1.4 2022 100.0 88.00 99.60 87.20 81.30 87.10 90.70 90.56 diffusiondb SD 1.X 2022 99.92 86.33 96.03 76.01 52.11 - 93.52 83.99 Synthbuster SD 2 2022 97.10 47.20 85.80 39.80 53.80 42.30 75.50 63.07

- Synthbuster DALLE-2 2022 0.40 45.80 70.80 47.30 76.00 20.70 93.10 50.59 Synthbuster MJ V5 2023 98.10 30.60 87.00 49.90 83.60 36.50 88.50 67.74 mj-images MJ V5/V6 2023 90.11 5.16 31.28 75.85 8.27 - 15.56 37.71 Synthbuster SDXL 2023 94.40 79.80 97.60 53.60 86.20 45.90 97.20 79.24 realisticSDXL SDXL 2023 97.65 28.64 82.17 91.82 69.77 - 75.61 74.28 Synthbuster Firefly 2023 18.10 81.40 43.30 54.00 66.40 24.50 83.00 52.96

- Synthbuster DALLE-3 2023 0.00 0.00 2.00 93.60 35.00 84.30 30.40 35.04 dalle3-images DALLE-3 2023 61.82 3.92 28.79 81.21 3.03 - 1.52 30.05 Authors’ SD3 SD 3 2024 99.24 59.02 85.05 89.42 70.74 78.44 89.03 81.56 Authors’ FLUX.1 Flux.1-dev 2024 62.89 23.08 54.72 81.43 63.92 85.40 92.99 66.35 In-the-wild Unknown 47.47 5.00 16.16 84.85 22.22 71.72 32.32 39.96 Average Synthetic 63.30 46.80 63.43 73.80 55.86 59.76 69.49

- Table 6. Center-patch recall of detector models evaluated with their best input resize resolution. Native denotes no resolution alteration. Top: Performance on authentic datasets. Bottom: Performance on synthetic datasets. Best recall in bold. Recalls below 50% (worse than random) underlined. Entries denoted by (-) in SuSy indicate datasets excluded from evaluation as they were used for training.

#### 6.2. Optimal Model Generalization

This final experiment evaluates the generalization capacity of existing detectors across benchmarking datasets after identifying their optimal input scales. Results in table 6 reveal a critical SID limitation: all detectors achieve less than 50% recall on at least four datasets. This demonstrates no universal detector exists, as all methods eventually perform worse than random chance. Performance metrics averaged across dataset types (authentic vs synthetic) consistently favor the authentic class. While this bias partially stems from the selection of optimal resolution (i.e. DeFake, GramNet and SuSy all have input resolutions where synthetic class

detection is higher, see Figure 3), it also reflects the more diverse and challenging distribution within the synthetic class. Additionally, a clear trade-off emerges across dataset types: DeFake is simultaneously the best synthetic detector and worst authentic detector, while FatFormer shows the opposite pattern.

While DeFake is the best synthetic detector on average and DIMD excels in 8 out of 17 synthetic datasets (including 6 out of 7 StableDiffusion variants), both suffer from high sensitivity to rescaling (see §6.1). This limitation makes them unsuitable for deployment scenarios with uncontrolled inputs. DIMD’s consistent performance across Stable Diffusion models stands as an exception, as

detectors generally show little consistency when handling different models within the same family. Performance varies even when two datasets are generated using the same model: the average recall difference between DALLE3 versions across SID models is 27.63%, while SDXL variations average 24.35%. While detectors may generalize to source changes under specific conditions (see §5.1), this is not universal. Despite the Synthbuster benchmarking datasets being generated using identical prompts and likely containing similar visual elements, detection performance varies substantially, as shown in Table 6.

Private models, like DALLE, Midjourney and Firefly, present generalization challenges. Detectors achieve only 45.22% average recall on closed SIG models, with the best closed dataset reaching 67.74%. In contrast, the same detectors achieve 76.60% average recall on open SIG models, with even the worst open dataset achieving 59.99% recall. These findings underscore the crucial role of open science in advancing the field.

The In-the-wild dataset, serving as a proxy for real-world performance, reveals additional limitations. No tested detector achieves above 50% recall for both authentic and synthetic versions across all input resolutions. Only SuSy demonstrates robust performance, achieving over 70% recall in both subsets, but specifically when operating

- at its optimal input resolution.

### 7. Conclusions

In a race equilibrium paradox, better generative models appear regularly, making the task harder for humans, while detectors trained on these newer generators are more reliable (see §4.1), keeping the race close.

The demand for detectors grows as society seeks to preserve social trust and digital rights while combating disinformation. Yet these detectors must improve their generalization capabilities to be truly effective. In that regard, the main lesson from this work is: never assume generalization in SID. Results in Table 6 indicate even within datasets produced by the same generative model, detection performance may largely vary, as a result of software and hardware setups and user bias. Similarly, generalization should not be assumed on synthetic images produced by older, less realistic generators either, even if these synthetic samples seem more obvious to the human eye. As shown in Table 1, samples from these models are hard to generalize (but not to train for) due to their stronger biases and distinct artifacts. In fact, even simple post-processing methods, like blur, can significantly reduce detector performance (see Table 3).

On top of this, image scale can dramatically affect the performance of most detectors, as well as the balance of their performance (i.e. authentic vs synthetic). Some detectors are highly sensitive to rescaling operations (see Figure 3), exposing a vulnerability to malicious inputs. At the

same time, other SID models work optimally when applied to data that has been scaled to a certain size (see Figure 3). This can be used to tune data for its detection, boosting performance on deployment settings (see Table 6).

The final contribution of this work, beyond the released SuSy, code and datasets, is a list of policies for the SID field as a whole, including an ethical risk assessment. First, our work emphasizes the importance of openness in the field of generative AI. Results from Table 6 indicate open generative models can be more easily detected (+20% combined recall points on average). While we are far from a universal detector (all detectors perform below random in some of our benchmarks), models trained for specific targets may be as good as humans at identifying synthetic content (see Table 5).

#### 7.1. Ethical Risks

Image detection systems pose significant ethical concerns, primarily due to their inherent fallibility. These systems produce both false positives and negatives (see Table 6), potentially misidentifying authentic images as synthetic and vice versa. Such errors could infringe on digital rights and enable censorship. Therefore, human expert oversight is crucial when these systems are used in contexts affecting individual rights, and their outputs should never serve as definitive evidence.

Additionally, model bias remains a critical challenge. Training datasets often contain inherent biases that can skew detection results (e.g. rural landscapes could be tagged as synthetic more often than urban images). Thorough evaluation across all relevant demographic and contextual factors is essential before deployment.

Furthermore, the datasets used for training may include samples with personal data. COCO contains images of real people, and synthetic datasets used could include realistic depictions of specific individuals. However, given the training objective and parameter size of SuSy, it is highly unlikely that any such information could be encoded within the weights released in this work.

A final risk of releasing a SID model is dual use, as it can be used as a training objective for generative models (e.g. adversarial training). To mitigate that, we add a specific clause in the terms of use of the model prohibiting such practice. Notice SuSy is not trained to be the best possible detector (not trained on all data), and should not be used as is in practice. We recommend any SID model produced for final use to be kept private, as long as its public release holds no special academic or social value.

#### 7.2. Future Work

The results of this work point towards four research directions that could improve SID robustness and adaptability. The complementary strengths of different detector mod-

els indicate potential benefits from ensemble methods that combine them. Exploring training data scaling laws could reveal further insights into data requirements and generalization capabilities. Given the impact of input resolution, developing multi-resolution architectures could provide inherent resilience against scaling-based evasion attempts. Lastly, extending detection capabilities to video content is crucial to address the increasing quality of video generation models. These advancements are critical to ensure SID keeps pace in the ongoing race with SIG.

### Acknowledgements

This work has been partly funded by the AI4Media and AI4Europe projects from the European Union’s Horizon 2020 programme (Grant Agreements Nº951911 and Nº101070000), and by a SGR-GRE grant from the Generalitat de Catalunya (code 2021 SGR 01187). The authors would like to acknowledge Mauro Achile, Eric Arean, Nura Mangado, Diego Rios and Daniel Pulido who contributed to motivating and contextualizing this work. Special thanks to the volunteers who participated in the human evaluation experiment.

### References

- [1] European Environment Agency. Greenhouse gas emission intensity of electricity generation in europe, 2024. 1
- [2] Agil Aghasanli, Dmitry Kangin, and Plamen Angelov. Interpretable-through-prototypes deepfake detection for diffusion models. In Proceedings of IEEE/CVF international conference on computer vision, pages 467–474, 2023. 1, 2
- [3] Stability AI. Stable diffusion xl, 2023. 2
- [4] Stability AI. Stable diffusion 3 medium, 2024. 3, 2
- [5] Quentin Bammey. Synthbuster: Towards detection of diffusion model generated images. IEEE Open Journal of Signal Processing, 2023. 2, 3
- [6] Alexander Buslaev, Vladimir I. Iglovikov, Eugene Khvedchenya, Alex Parinov, Mikhail Druzhinin, and Alexandr A. Kalinin. Albumentations: Fast and flexible image augmentations. Information, 11(2), 2020. 4
- [7] George Cazenavette, Avneesh Sud, Thomas Leung, and Ben Usman. Fakeinversion: Learning to detect images from unseen text-to-image models by inverting stable diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10759–10769, 2024. 2
- [8] Lucy Chai, David Bau, Ser-Nam Lim, and Phillip Isola. What makes fake images detectable? understanding properties that generalize. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVI 16, pages 103–120. Springer, 2020. 1
- [9] Keshigeyan Chandrasegaran, Ngoc-Trung Tran, and NgaiMan Cheung. A closer look at fourier spectrum discrepancies for cnn-generated images detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7200–7209, 2021. 2

- [10] You-Ming Chang, Chen Yeh, Wei-Chen Chiu, and Ning Yu. Antifakeprompt: Prompt-tuned vision-language models are fake image detectors. arXiv preprint arXiv:2310.17419,

2023. 2

- [11] Baoying Chen, Jishen Zeng, Jianquan Yang, and Rui Yang. Drct: Diffusion reconstruction contrastive training towards universal detection of diffusion generated images. In Fortyfirst International Conference on Machine Learning, 2024. 1, 2
- [12] Davide Alessandro Coccomini, Andrea Esuli, Fabrizio Falchi, Claudio Gennaro, and Giuseppe Amato. Detecting images generated by diffusers. PeerJ Computer Science, 10: e2127, 2024. 2
- [13] Riccardo Corvi, Davide Cozzolino, Giovanni Poggi, Koki Nagano, and Luisa Verdoliva. Intriguing properties of synthetic images: from generative adversarial networks to diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 973–982,

2023. 2

- [14] Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, and Luisa Verdoliva. On the detection of synthetic images generated by diffusion models. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 1, 2, 6, 3
- [15] Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva. Raising the bar of ai-generated image detection with clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4356–4366, 2024. 2
- [16] Pablo Del Moral, Sławomir Nowaczyk, and Sepideh Pashami. Why is multiclass classification hard? IEEE Access, 10:80448–80462, 2022. 3
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 3
- [18] Yufan Deng, Xin Deng, Yiping Duan, and Mai Xu. Diffusion-generated fake face detection by exploring wavelet domain forgery clues. In 2023 International Conference on Wireless Communications and Signal Processing (WCSP), pages 1–6. IEEE, 2023. 2
- [19] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 3
- [20] DucHaiten. realisticsdxl dataset, 2023. 2
- [21] ehristoforu. dalle-3-images dataset, 2024. 2, 1
- [22] ehristoforu. midjourney-images dataset, 2024. 2
- [23] Joel Frank, Thorsten Eisenhofer, Lea Sch¨onherr, Asja Fischer, Dorothea Kolossa, and Thorsten Holz. Leveraging frequency analysis for deep fake image recognition. In International conference on machine learning, pages 3247–3258. PMLR, 2020. 1, 6, 4
- [24] Oliver Giudice, Luca Guarnera, and Sebastiano Battiato. Fighting deepfakes by detecting gan dct anomalies. Journal of Imaging, 7(8):128, 2021. 1

- [25] Patrick Grommelt, Louis Weiss, Franz-Josef Pfreundt, and Janis Keuper. Fake or jpeg? revealing common biases in generated image detection datasets. arXiv preprint arXiv:2403.17608, 2024. 1, 2
- [26] Gustavosta. Stable-diffusion-prompts, 2023. 2
- [27] Robert M Haralick, Karthikeyan Shanmugam, and Its’ Hak Dinstein. Textural features for image classification. IEEE Transactions on systems, man, and cybernetics, (6):610–621,

1973. 2

- [28] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2, 1
- [29] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017. 3
- [30] Christos Koutlis and Symeon Papadopoulos. Leveraging representations from intermediate encoder-blocks for synthetic image detection. arXiv preprint arXiv:2402.19091, 2024. 2, 6
- [31] Black Forest Labs. Flux.1-dev, 2024. 3, 2
- [32] Linda Laurier, Ave Giulietta, Arlo Octavia, and Meade Cleti. The cat and mouse game: The ongoing arms race between diffusion models and detection methods. arXiv preprint arXiv:2410.18866, 2024. 1
- [33] Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. Dit: Self-supervised pre-training for document image transformer. In Proceedings of the 30th ACM International Conference on Multimedia, pages 3530–3539,

2022. 3

- [34] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Doll´ar. Microsoft coco: Common objects in context, 2015. 2
- [35] Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Yao Zhao, and Jingdong Wang. Forgery-aware adaptive transformer for generalizable synthetic image detection,

2023. 6, 3

- [36] Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Jingdong Wang, and Yao Zhao. Forgery-aware adaptive transformer for generalizable synthetic image detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10770–10780, 2024. 1, 2
- [37] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. arXiv preprint arXiv:2202.09778, 2022. 3
- [38] Zhengzhe Liu, Xiaojuan Qi, and Philip HS Torr. Global texture enhancement for fake face detection in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8060–8069, 2020. 6
- [39] Peter Lorenz, Ricard L Durall, and Janis Keuper. Detecting images generated by deep diffusion models using their local intrinsic dimensionality. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 448– 459, 2023. 1, 2
- [40] Enrique L´opez Cuena. Super-resolution assessment and detection, 2023. 2, 1

- [41] Francesco Marra, Diego Gragnaniello, Luisa Verdoliva, and Giovanni Poggi. Do gans leave artificial fingerprints? In 2019 IEEE conference on multimedia information processing and retrieval (MIPR), pages 506–511. IEEE, 2019. 1
- [42] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 3

- [43] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards universal fake image detectors that generalize across generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24480– 24489, 2023. 1, 2, 6
- [44] Jonas Ricker, Simon Damm, Thorsten Holz, and Asja Fischer. Towards the detection of diffusion model deepfakes. arXiv preprint arXiv:2210.14571, 2022. 2
- [45] Santosh, Li Lin, Irene Amerini, Xin Wang, and Shu Hu. Robust clip-based detector for exposing diffusion modelgenerated images. 2024 IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), pages 1–7, 2024. 1
- [46] Manos Schinas and Symeon Papadopoulos. Sidbench: A python framework for reliably assessing synthetic image detection methods. arXiv preprint arXiv:2404.18552, 2024. 6
- [47] Sergey Sinitsa and Ohad Fried. Deep image fingerprint: Towards low budget synthetic image detection and model lineage analysis. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4067– 4076, 2024. 2
- [48] Jiawei Song, Dengpan Ye, and Yunming Zhang. Trinity detector: text-assisted and attention mechanisms based spectral fusion for diffusion generation image detection. arXiv preprint arXiv:2404.17254, 2024. 2
- [49] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, and Yunchao Wei. Learning on gradients: Generalized artifacts representation for gan-generated images detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12105–12114, 2023. 2, 6
- [50] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Frequency-aware deepfake detection: Improving generalizability through frequency space learning. arXiv preprint arXiv:2403.07240, 2024. 2
- [51] Iulia Turc and Gaurav Nemade. Midjourney user prompts & generated images (250k), 2022. 2
- [52] European Union. Proposal for a regulation of the european parliament and of the council laying down harmonised rules on artificial intelligence (artificial intelligence act) and amending certain union legislative acts, 2024. 1
- [53] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot... for now. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8695–8704, 2020. 1, 6
- [54] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. Dire for diffusion-generated image detection. In Proceedings of the

- IEEE/CVF International Conference on Computer Vision, pages 22445–22455, 2023. 1, 6
- [55] Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. DiffusionDB: A large-scale prompt gallery dataset for text-toimage generative models. arXiv:2210.14896 [cs], 2022. 2
- [56] Tobias Weyand, Andre Araujo, Bingyi Cao, and Jack Sim. Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2575–2584, 2020. 3
- [57] Alexander Wißmann, Steffen Zeiler, Robert M Nickel, and Dorothea Kolossa. Whodunit: Detection and attribution of synthetic images by leveraging model-specific fingerprints. In Proceedings of the 3rd ACM International Workshop on Multimedia AI against Disinformation, pages 65–72, 2024. 1
- [58] Haiwei Wu, Jiantao Zhou, and Shile Zhang. Generalizable synthetic image detection via language-guided contrastive learning. arXiv preprint arXiv:2305.13800, 2023. 2
- [59] Ziyi Xi, Wenmin Huang, Kangkang Wei, Weiqi Luo, and Peijia Zheng. Ai-generated image detection using a crossattention enhanced dual-stream network. In 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 1463–1470. IEEE, 2023. 2
- [60] Qiang Xu, Hao Wang, Laijin Meng, Zhongjie Mi, Jianye Yuan, and Hong Yan. Exposing fake images generated by text-to-image diffusion models. Pattern Recognition Letters, 176:76–82, 2023. 1, 2
- [61] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. 3
- [62] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015. 3
- [63] Xu Zhang, Svebor Karaman, and Shih-Fu Chang. Detecting and simulating artifacts in gan fake images. In 2019 IEEE international workshop on information forensics and security (WIFS), pages 1–6. IEEE, 2019. 1
- [64] Nan Zhong, Yiran Xu, Sheng Li, Zhenxing Qian, and Xinpeng Zhang. Patchcraft: Exploring texture patch for efficient ai-generated image detection. arXiv preprint arXiv:2311.12397, pages 1–18, 2024. 2
- [65] Mingjian Zhu, Hanting Chen, Mouxiao Huang, Wei Li, Hailin Hu, Jie Hu, and Yunhe Wang. Gendet: Towards good generalizations for ai-generated image detection. arXiv preprint arXiv:2312.08880, 2023. 1, 2
- [66] Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, and Yunhe Wang. Genimage: A million-scale benchmark for detecting ai-generated image. Advances in Neural Information Processing Systems, 36, 2024. 1, 2

## Present and Future Generalization of Synthetic Image Detectors Supplementary Material

### A. Model Architecture

Figure 4 shows the detector architecture used for SuSy, based on the design proposed in [40]. The architecture combines CNN-based feature extraction with MLP classification in a staircase design. The model employs a ResNet18 [28] backbone for feature extraction, totaling 12.7M parameters, with the CNN feature extractor accounting for 12.5M parameters and the MLP classifier using 197K parameters.

The CNN feature extractor implements five sequential stages following the ResNet-18 architecture. Each stage’s output feeds into specialized bottleneck modules arranged in a staircase pattern. These bottleneck modules consist of three 2D convolutional layers that process and refine the extracted features. The staircase architecture creates a hierarchical feature processing system where each bottleneck level processes features from progressively later stages. The bottleneck modules combine inputs from their current stage and previous bottleneck module, except for the first module in each level. The staircase design enables the model to leverage features extracted at multiple depths, enhancing its ability to detect and classify synthetic content.

The classification component processes features through several steps. First, a 2D adaptive average pooling is applied to each bottleneck level output and stage 4. These pooled features are then concatenated to create a unified feature map. This feature map feeds into a three-layer MLP with dimensions of 512, 256 and 256 units, with dropout layers (at 0.5 probability) inserted between MLP layers to prevent overfitting.

[Figure 5]

Figure 4. Detector architecture used, based on a ResNet-18 from [40], including ResNet blocks (blue), bottlenecks (red), adaptative average pooling 2D (orange), concatenation (yellow) and an MLP (green).

[Figure 6]

Figure 5. Scalability analysis showing images processed per second using single-crop (green) and 5-crop (blue) approaches across different hardware configurations. Note the axis break highlighting the GPU acceleration gain.

### B. Experiment Setup

Experiments were conducted on the MareNostrum 5 supercomputer, hosted at the Barcelona Supercomputing Center (BSC). We utilize an Intel Xeon Platinum 8460Y processor and one NVIDIA Hopper H100 64GB GPU. Seventy-five training runs were conducted with this setup, totalling sixteen hours of computing time, while continuously monitoring GPU power usage. Using the European Union’s latest CO2 emission ratio [1], we estimated the carbon footprint of these experiments to be 0.63 kg of CO2.

Figure 5 presents a scalability analysis across different hardware configurations, ranging from 2 to 64 CPU cores, plus a hybrid setup combining 64 CPU cores with GPU acceleration. The evaluation compares processing speeds using single-crop and 5-crop approaches, measuring only the network’s forward pass time. The results demonstrate near-linear scaling with CPU cores, with the single-crop approach consistently outperforming the 5-crop strategy in CPU-only configurations due to its lower computational requirements. However, this performance gap becomes negligible in the hybrid CPU-GPU setup, where both approaches achieve similar throughput of approximately 3,000 images per second, caused by the GPU’s superior parallel processing capabilities for matrix operations. The improvement in processing speed with GPU acceleration, 7 to 34 times speedup compared with the fastest CPU-only setups, is highlighted by the broken y-axis in the plot.

### C. Training Datasets Details

The dalle3-images [21] dataset contains 1,647 unique, deduplicated images generated by DALLE3 (2023), encompassing both photorealistic and digital art styles. Another

Train Dataset Model Year Image Format Type Train Validation Test COCO - 2017 JPG Authentic 2,967 1,234 1,234 dalle3-images DALLE3 2023 JPG Synthetic 987 330 330 diffusiondb SD1.X 2022 PNG Synthetic 2,967 1,234 1,234

SDXL realisticSDXL 2023 PNG Synthetic 2,967 1,234 1,234 mj-tti MJ 1/2 2022 PNG Synthetic 2,718 906 906

mj-images MJ 5/6 2023 JPG Synthetic 1,845 617 617 Evaluation Dataset

Flickr30k - 2014 JPEG Authentic - - 31,655

GLDv2 - 2020 JPEG Authentic - - 5,000 In-the-wild - 2024 Mix Authentic - - 121 Synthbuster Many 2024 PNG Synthetic - - 9,000

SD3 SD 3 2024 PNG Synthetic - - 8,192 FLUX.1 FLUX.1-dev 2024 PNG Synthetic - - 8,192

In-the-wild ? 2024 PNG Synthetic - - 99

- Table 7. Datasets, including generative models included, release date, image format, authentic or synthetic, and number of samples within train, validation and test.

dataset, the diffusiondb [55], was created using models of the 1.x Stable Diffusion series, which were released in 2022. In this dataset, we filter samples making sure that ’photo’ appears in the prompt. Images from this dataset are of lower quality and visual detail than those of its successor SDXL [3], which was released in 2023. The associated dataset, SDXL [20] contains 5,435 images in the ’realistic’ subset.

Beyond DALL-E and Stable Diffusion, the third main provider of synthetic images is Midjourney. Its early iterations, the V1 and V2 models, date from early 2022, and were used to populate the mj-tti [51] dataset, which contains 4,530 images. Collage images and mosaics made of synthetic images were removed from this dataset. Later models, the V5 and V6 models from 2023 compose our last training dataset, mj-images [22], with 1,226 images. This dataset also had to be deduplicated.

### D. Generated Datasets

The input prompts used to produce the samples of SD3 and FLUX.1 are extracted from Gustavosta/Stable-DiffusionPrompts[26]. For each image, height and width were randomly selected from a uniform distribution over the set {512,768,1024,1344} pixels. The images were generated using the official models [4, 31] accessed through HuggingFace. We employed a consistent generation process across all images, utilizing 28 inference steps for each generation. To enhance the quality and realism of the generated images, we added a set of negative prompts: poorly rendered face, poor facial details, poorly rendered hands, low resolution, blurry image, oversaturated, extra fingers, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, extra foot. For FLUX-1.dev, inference

is run using torch.bfloat16 precision, a guidance scale of 3.5 and a maximum sequence length of 512.

### E. Threshold Study

To transform the six-class model predictions into binary classifications, a threshold-based approach is tested as an alternative to the traditional argmax method. This approach compares the probability score of the authentic class against a threshold value. Specifically, if the predicted probability for the authentic class is greater or equal than the threshold, the image is classified as authentic; conversely, if the probability falls below the threshold, the image is labeled as synthetic.

In the top part of Figure 6, the performance of SuSy with different thresholds is reported. The best performance is obtained with the threshold at 0.4, with the crossover point between authentic and synthetic accuracies being just above 0.2. Additionally, the threshold mechanism is tested with the different scale sizes of §5.3, as reported in the other plots of figure 6. With the scale changes, the crossover point moves progressively from above 0.9 at the smallest scale to approximately 0.01 at the largest scale. This behaviour mirrors the progression in class performance in Figure 2.

We extended our threshold analysis to the external datasets described in §3.3, as shown in Figure 7. The results reveal a similar inverse relationship between image dimensions and optimal threshold values. However, these experiments uncovered more pronounced performance disparities between authentic and synthetic classifications compared with the previous experiment.

[Figure 7]

- Figure 6. Classification accuracy curves for authentic (yellow), synthetic (blue) and green (average) test splits of §3.2 datasets across seven resize dimensions.

### F. Detector Models for Benchmarking

LGrad trains a ResNet-50 classifier using image gradients from a pre-trained CNN, with images generated by ProGAN and authentic images from Celeba-HQ [29]. Similarly, GramNet employs global image texture representations extracted at different levels from a ResNet-18, trained on StyleGAN-generated images and authentic Celeba-HQ images. Rine leverages image representations extracted by intermediate blocks of CLIP, with an additional trainable module. We use the checkpoint trained with Latent Diffusion Model [14] and ProGAN images. DIMD trains a ResNet-50 avoiding downsampling step, to preserve highfrequency fingerprints. We take the checkpoint trained on Latent Diffusion images. Training authentic images are taken from MSCOCO and LSUN. In DeFAKE, text and image encoders from a Visual-Language Model model are finetuned to detect synthetic images, using Latent Diffusion data. Dire uses the error between an input image

and its reconstruction by a pre-trained diffusion model for identification. A ResNet50 is trained as a classifier on their DiffusionForensics dataset. Synthetic images generated with, ADM [19], iDDPM [42] and PNDM [37], from LSUN-Bedroom [62] and Imagenet [17]. They hypothesize that diffusion-generated images can be approximately reconstructed by a diffusion model while real images cannot. FatFormer [35] adapts a pre-trained CLIP model by adding custom forgery-aware adapters to the image encoder to capture both low-level forgery artifacts and forgery traces in different frequency bands. It uses language-guided alignment, which leverages contrastive objectives between image features and text prompts.

The evaluation results in Table 8 reveal significant performance disparities across models when tested on authentic and synthetic datasets applying a center crop of size 224 × 224 to the input image. For CNNDetect, while it achieves high recall rates on authentic datasets (96.88% on average), its performance on synthetic datasets is poor, with

[Figure 8]

- Figure 7. Classification accuracy curves for authentic (yellow) and synthetic (blue) data across seven resize dimensions, showing optimal threshold values and corresponding accuracy scores. The average performance (green) reveals that higher resize dimensions lead to lower optimal threshold values.

an average recall of only 11.91%. This suggests a strong reliance on features specific to GAN-based models in the training set, which are largely absent in modern diffusionbased architectures. Similarly, FreqDetect [23] demonstrates recall values close to 100 for authentic datasets but fails to adapt to the frequency artifacts produced by diffusion-based generators. UnivFD shows more balanced performance across some diffusion-based datasets, suggesting better generalization. However, it still lags behind on more recent synthetic datasets. Dire stands out with notably low recall rates on both authentic (68.17%) and synthetic (42.98%) datasets. While it outperforms other models on a few synthetic datasets, its overall performance remains inconsistent.

### G. Benchmark Image Resolution Distribution

We calculate the resolution distribution for each of the evaluation datasets. Figure 8 contains information regarding the

size and aspect ratio of the datasets used. The top plot shows the width distribution of all samples, dataset-wise. The bottom plot shows the same information for height.

Figure 8 highlights distinct differences in resolution distribution between authentic and synthetic datasets. Authentic datasets, such as In-the-wild, exhibit a broad range of resolutions, with widths and heights reaching up to 7000 and 8000 pixels, respectively, reflecting real-world variability. Synthetic datasets demonstrate narrower resolution distributions that are similar to each other. Among synthetic datasets, Firefly contains the images with the highest resolutions.

### H. Benchmarking Dataset Image Samples

This section includes several sample images from the evaluation datasets utilized:

- • Figure 9: GLDv2 and Flickr30k.
- • Figure 10: In-the-wild

Model Year CNNDetect FreqDetect UnivFD Dire Best Resolution Native 768 1024 224

Authentic Data Sources

Flickr30k - 2014 98.28 98.80 93.26 65.38 COCO - 2017 98.14 97.89 94.98 66.45 GLDv2 - 2020 98.54 99.62 93.72 73.08 In-the-wild - 2024 92.56 95.87 98.35 67.77

Average Authentic 96.88 98.05 95.08 68.17 Synthetic Data Sources

Synthbuster Glide 2021 17.10 76.70 60.70 26.10 mj-tti MJ V1/2 2022 5.52 5.52 58.61 28.92

- Synthbuster SD-1.3 2022 7.50 1.80 66.10 76.10
- Synthbuster SD-1.4 2022 8.90 1.30 64.40 76.40 diffusiondb SD 1.X 2022 9.72 6.24 56.48 28.20 Synthbuster SD-2 2022 23.50 0.80 43.80 54.90

- Synthbuster DALLE-2 2022 18.00 6.80 72.70 49.00 Synthbuster MJ-V5 2023 11.40 2.10 1.90 29.40 mj-images MJ V5/6 2023 0.65 1.78 6.16 31.12 Synthbuster SD-XL 2023 35.90 0.90 29.30 39.10 SDXL SDXL 2023 18.40 6.16 4.21 27.88 Synthbuster Firefly 2023 20.80 6.20 33.80 38.20
- Synthbuster DALLE-3 2023 0.10 0.70 0.20 69.90 dalle3-images DALLE-3 2023 1.21 0.91 12.12 43.94 Authors SD-3 2024 10.89 12.57 6.70 28.56 Authors Flux.1-dev 2024 5.73 2.89 1.83 32.37 In-the-wild Unknown 7.07 3.03 2.02 50.51 Average Synthetic 11.91 8.02 30.65 42.98

- Table 8. Center-patch recall of other studied detector models across evaluation datasets. Each model is evaluated on the best resolution. Top: Performance for authentic images. Bottom: Performance on synthetic datasets. Recalls below 50%, worse than random chance, underlined. Best recall for each dataset in bold.

- • Figure 11: Synthbuster (GLIDE, SD1.3, SD1.4, MJ 5, Firefly, DALLE2, GLIDE, SD2, SDXL, DALLE3).
- • Figure 12: SD3 and FLUX.1.

[Figure 9]

###### Figure 8. Resolution distribution of images across various datasets. The plots display the distribution of image widths (top) and heights (bottom). SB indicates datasets sourced from Synthbuster, (test) refers to datasets derived from our test splits.

[Figure 10]

Figure 9. Sample authentic images from Flickr30k (top) and GLDv2 (bottom).

[Figure 11]

- Figure 10. Sample images from our In-the-wild dataset. Synthetic images (top) and authentic images (bottom)

[Figure 12]

[Figure 13]

###### Figure 11. Sample synthetic images from Synthbuster. Note that the same prompt is used for each column.

[Figure 14]

###### Figure 12. Sample synthetic images from our generated Stable Diffusion 3-Medium (top) and FLUX.1-dev (bottom) datasets.

