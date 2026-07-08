# arXiv:2406.19395v1[cs.CV]27Jun2024

## Dataset Size Recovery from LoRA Weights

Mohammad Salama Jonathan Kahana Eliahu Horwitz Yedid Hoshen School of Computer Science and Engineering The Hebrew University of Jerusalem, Israel

https://vision.huji.ac.il/dsire/ { mohammad.salama3, jonathan.kahana, eliahu.horwitz, yedid.hoshen}@mail.huji.ac.il

### Abstract

Model inversion and membership inference attacks aim to reconstruct and verify the data which a model was trained on. However, they are not guaranteed to find all training samples as they do not know the size of the training set. In this paper, we introduce a new task: dataset size recovery, that aims to determine the number of samples used to train a model, directly from its weights. We then propose DSiRe, a method for recovering the number of images used to fine-tune a model, in the common case where fine-tuning uses LoRA. We discover that both the norm and the spectrum of the LoRA matrices are closely linked to the fine-tuning dataset size; we leverage this finding to propose a simple yet effective prediction algorithm. To evaluate dataset size recovery of LoRA weights, we develop and release a new benchmark, LoRA-WiSE, consisting of over 25,000 weight snapshots from more than 2,000 diverse LoRA fine-tuned models. Our best classifier can predict the number of fine-tuning images with a mean absolute error of 0.36 images, establishing the feasibility of this attack.

### 1 Introduction

Data is the top factor for the success of machine learning models. Model inversion [13, 52, 14] and membership inference attacks [5, 42, 25] aim to reconstruct and verify the training data of a model, using its weights[14, 11, 36]. While these methods may discover some of the training data, they are not guaranteed to recover all training samples. One fundamental limit that prevents them from discovering the entirety of the training data is that they do not have a halting condition, as they do not know the size of the training set [14]. E.g., in membership inference, the attacker sequentially tests a set of images for membership in the training set but does not know when to halt the algorithm, effectively making its runtime infinite.

Discovering the size of a training dataset given the model weights is important, even without explicit reconstruction of the images themselves. A stock photography provider such as Getty or Shuterstock, may allow users to use their data for fine-tuning personalized generative models and charge them for the number of images actually used for training. Quantifying the dataset size would therefore be essential for billing. Understanding the number of images used to train or fine-tune models is also of great interest to researchers, who wish to understand the costs of replicating a model’s performance.

We therefore propose a new task: Dataset Size Recovery, which aims to recover the number of training images based on the model’s weights. We tackle the important special case of recovering the number of images used to fine-tune a model, where fine-tuning uses Low-Rank Adaption (LoRA)[21]. These LoRA personalized text-to-image foundation models [39] are among the most commonly trained models, as evident by the success of marketplaces for public sharing of them such as CivitAI and HuggingFace.

Preprint. Under review.

DSiRe

Finetuning

- 𝑠1

...

- 𝑠2

|SVD|
|---|

𝑟

|LoRA|
|---|

- 𝜎1

- 𝜎2

- 𝜎3 𝜎𝑟

|Pre-Trained Model|
|---|

∗

Majority Vote

𝑛

LoRA KNN

𝑑 Weights

...

𝑠𝐿

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

𝑛 samples

| |𝐿𝑀𝐴𝐸| |
|---|---|---|
| | | |

|𝒏|
|---|

2

1

Invisible to Attacker

- Figure 1: DSiRe: We introduce the task of dataset size recovery, which aims to recover the dataset size used to LoRA fine-tune a model based on its weights. DSiRe extracts the singular values of each LoRA matrix and treats them as features. These features are then used to train a set of layer-specific nearest-neighbor classifiers which predict the dataset size.

We first analyze the relationship between LoRA fine-tuning weights and their corresponding dataset size. Our observations suggest that the Frobenius norm of LoRA matrices is highly predictive of their fine-tuning dataset size. As a single scalar per weight matrix provides limited expressivity, singular values of each weight matrix serve as more expressive feature vectors. With this in mind, we present DSiRe (Dataset Size Recovery). The method recovers the fine-tuning dataset size from the LoRA weight spectrum. DSiRe classifies the dataset size using a trained classifier on top of the weight spectrum of the layer. In practice, we found that a very simple classifier suffices; in our experiments, DSiRe uses the nearest neighbor classifier. For an overview of the method see Fig 1.

To enable the evaluation of this work and encourage future research, we release a new, large-scale, and diverse dataset: LoRA-WiSE. The dataset comprises over 25,000 weights checkpoints drawn from more than 2,000 independent LoRA models, spanning different dataset sizes, backbones, ranks, and personalization sets. On LoRA-WiSE, DSiRe recovers the dataset sizes from LoRA weights with a Mean Absolute Error (MAE) of 0.36, demonstrating that our method is highly effective in realistic settings.

To summarize, our main contributions are:

- 1. Introducing the task of dataset size recovery.
- 2. Presenting DSiRe, a method for recovering dataset size for LoRA fine-tuning.
- 3. Releasing LoRA-WiSE, a comprehensive dataset size recovery evaluation suite based on Stable-Diffusion-fine-tuned LoRAs.

### 2 Related work

#### 2.1 Model Fine-Tuning

Model fine-tuning [58, 56, 3] adapts a model for a downstream task and is considered a cornerstone in machine learning. The emergence of large foundation models [37, 45, 4, 38] has made standard fine-tuning costly and unattainable without substantial resources. Parameter-Efficient Fine-Tuning (PEFT) methods were then proposed [21, 10, 20, 29, 28, 32, 16, 31, 26, 59, 47, 24], offering various ways to fine-tune models with fewer optimized parameters. Among these methods, LoRA [21] stands out, proposing to train additive low-rank weight matrices while keeping the pre-trained weights frozen. LoRA was found to be very effective across several modalities [48, 53, 2]. Recently, Horwitz et al. [19] identified a security issue in LoRA fine-tuning, demonstrating that multiple LoRAs can

be used to recover the original pre-trained weights. In this paper, we uncover a new use case of LoRA fine-tuning, specifically focusing on the recovery of the dataset size from text-to-image models fine-tuned via LoRA.

- 2.2 Membership Inference & Model Inversion Attacks

Two privacy vulnerabilities found in machine learning models are Membership Inference Attack (MIA) [41, 5, 23, 42, 25] and Model Inversion [13, 52, 17, 55, 14]. First presented by [44], MIAs aim to verify whether a certain image was in the training dataset of a given model. Typically, MIAs assumes that training samples are over-fitted proposing various membership criteria; either by looking for lower loss values [40, 54] or some other metrics [49, 5]. In generative models, MIAs have been extensively researched as well [18, 15, 6], including recent attacks against diffusion models [35, 22].

Model inversion is a similar attack, in a data-free setting. Introduced by [13], model inversion methods wish to generate training samples from scratch, instead of asking whether a known specific image was in the training set. Model inversion is also used for settings where data is unavailable, e.g., data-free quantization [7, 51, 30] and data-free distillation [33, 60, 57, 12, 43].

Haim et al. [14] emphasized the importance of recovering the training set size for model inversion applications. When this size is unknown, it prevents model inversion attacks from reconstructing the entire dataset a model was trained on, as it is unclear how many samples are sufficient. Our work specifically addresses this issue by uncovering a new vulnerability in fine-tuned models, which enables us to infer the size of the dataset used for fine-tuning.

- 3 Motivation

- 3.1 Background: LoRA fine-tuning.

Fine-tuning large foundation models can be a computationally expensive process, as it modifies each weight matrix of the pre-trained model W ∈ Rd×k, by an additive fine-tuning matrix ∆W, as follows:

W′ = W + ∆W (1)

Recently, Hu et al. [21] introduced Low-Rank Adaptation (LoRA) for efficient fine-tuning. In LoRA, the fine-tuning matrix ∆Wi is low-rank, i.e., we choose its rank r s.t. r ≪ min(d,k). An efficient and SGD amenable way to implement low-rank matrices is to parameterize them as the product of two rectangular matrices so that Bi ∈ Rd×r, Ai ∈ Rr×k. The fine-tuning matrix is therefore given by:

∆Wi = BiAi (2)

- 3.2 Analysing the LoRA Spectrum.

Our hypothesis is that the difference between pre-fine-tuning and post-fine-tuning weights, denoted as ∆Wi, encodes information about the size of the fine-tuning dataset. Here, we focus on the case where fine-tuning uses LoRA, i.e., where ∆Wi is low-rank, which is emerging as the most popular fine-tuning paradigm for foundation models.

We begin by considering a very simple statistic of each fine-tuning matrix, its Frobenius norm that we denote sF.

|∆Wij|2 (3)

sF =

ij

The norm of a weight matrix correlates with the function complexity the network can express. For example, weight decay, that effectively constrains the norm is a common way for regularizing models. We suggest a simple experiment to analyze the correlation between sF and the fine-tuning dataset size. We fine-tune Stable Diffusion (SD) 1.5 on 50 micro-datasets of sizes 1 to 6 images, keeping all hyper-parameters, apart from the dataset size, fixed. Fig. 2a shows the range of values of the Frobenius norm statistic sF for each dataset size. It is clear that sF is negatively correlated to the dataset size. We motivate this correlation by over-fitting, i.e., models over-fit faster to smaller dataset sizes, leading to larger sizes of sF.

To obtain a deeper understanding, we monitor another statistic of the fine-tuning matrix, its singular values spectrum. As we consider the case of LoRA fine-tuning, the spectrum of ∆W has at most r

[Figure 5]

[Figure 6]

(a) (b)

- Figure 2: Norm and Spectrum of Fine-Tuning Weights vs. Dataset Size. Analysis of 210 Stable Diffusion 1.5 models fine-tuned on datasets of sizes from 1 − 6. (a) Frobenius norm range per dataset size (b) Singular values per dataset size. There is a clear negative correlation between weight/spectrum magnitudes and the size of the fine-tuning dataset.

[Figure 7]

(a) (b)

[Figure 8]

- Figure 3: Spectrum Ranges of 2 Different Layers. Singular values distribution of two layers on opposite sides of Stable Diffusion 1.5 UNet, fine-tuned on datasets of sizes 1−6. (a) First down block (b) Last upper block. The last upper block shows greater separation of singular values compared to the first down block, highlighting that not all layers are born equally for dataset size recovery.

non-zero values (where r is the rank). We denote the mth spectral value as σm, where m ∈ [1,...,r]. The range of values of were plotted σm across different values of m and different dataset sizes in Fig. 2b. We note there is a better separation between different dataset sizes for the largest singular values. This suggests that the spectrum is more discriminative than the scalar Frobenius norm. Overall, both sF and the spectrum indicate larger values for small dataset sizes.

Finally, we analyzed how discriminative different layers are for predicting fine-tuning dataset size. We plot the spectra of two different layers - in the first down block and the last up block in Fig. 3. We can see that the layer is more discriminative than the former one. Indeed,the up layers are found to be more discriminative than the down ones. Without a clear explanation, it can hypothesize that the UNet decoder is more prone to over-fitting than the encoder. It is noteworthy that our experiments revealed that no single layer is discriminative for all models, suggesting that weighting the results from all layers is better.

- 4 Method

#### 4.1 Task Definition: Dataset Size Recovery

We introduce the task of Dataset Size Recovery for fine-tuned dataset, a new attack vector against fine-tuned models. Formally, given the fine-tuning weights of all layers of a model denoted as

∆W = [∆W1,∆W2...∆WL], our task is to recover the number of images n that the model was fine-tuned on. More formally, we wish to find a function f, such that:

n = f(∆W) (4)

The effectiveness of this attack was measured by the MAE between f(∆W) and n across a set of models.

#### 4.2 DSiRe

We propose DSiRe (Dataset Size Recovery), a supervised method for recovering dataset size from LoRA fine-tuned weights. Our approach first constructs a training dataset by fine-tuning multiple LoRA models on concept personalization sets across a range of dataset sizes (see Sec. 5). It then trains a predictor function f that acts on a set of fine-tuning weights of each model and outputs the predicted dataset size n. At test time, it generalizes to unseen models trained with different concepts. The method can be seen in Fig. 1

Training set synthesis. We first synthesize a training set by fine-tuning our model on each of Ntrain datasets, each containing a set of training images. The datasets span a range of sizes; in this paper, we

tested the ranges 1 − 6,1 − 50, and 1 − 1000. The result is a set of Ntrain models ∆Wm, each with a corresponding label of the dataset size nm.

Predictor training. Given the set of Ntrain labeled models, we wish to train a predictor that maps the fine-tuning weights Wm to dataset size nm. Motivated by the results of our analysis (see Sec. 3), we describe the weights of each fine-tuning layer using its spectrum Σ consisting of r singular values. Let us define the features of each model as the set of spectra of all its L layers f = [Σ1,Σ2..ΣL]. We tested many different predictors and ablated them in Tab. 5. Overall, the simple Nearest Neighbor (NN) ensemble performed the best. During inference, given a new model, for each layer, we retrieve the NN layer that has the most similar spectrum to the layer of the target model. Each layer votes for the dataset size of its NN layer. The overall prediction is the dataset size that most layers voted for.

### 5 LoRA WiSE Benchmark

We present the LoRA Weight Size Evaluation (LoRA-WiSE) benchmark, a comprehensive benchmark specifically designed to evaluate LoRA dataset size recovery methods, for generative models. More specifically, it features the weights of 2350 Stable Diffusion [38] models, which were LoRA fine-tuned by a standard, popular protocol [39, 1]. Our benchmark includes versions 1.5 and 2 of Stable Diffusion, having 2050 and 300 trained models for each version respectively.

We fine-tune the models using three different ranges of dataset size: (i) Low data range: 1 − 6 images. (ii) Medium data range: 1 − 50 images. (iii) High data range: 1 − 1000. For each range, we use a discrete set of fine-tuning dataset sizes. In the low and medium ranges, we also provide other versions of these benchmarks with different LoRA ranks and backbones. See Tab.1 for the precise benchmark details.

For our low data range set, we choose Concept101 [27], a previously collected set of micro-datasets (3 − 15 images) designed for personalization research. For our medium and high data ranges we use different classes of ImageNet [9] as the data source. This selection of datasets aims to ensure that the fine-tuned models are drawn from a diverse set of concepts, spanning various categories.

Each micro-dataset is used to fine-tune the models for each dataset size. The images are randomly selected from the micro-dataset. Each Stable Diffusion model consists of 132 adapted layers (pairs of Ai,Bi), including various layer types, such as self-attention, cross-attention, and MLPs. We save Ai, Bi separately, i.e., each model provides a total of 264 unique weight matrices. We then split each range of this new benchmark (low, medium, and high ranges) into train and test sets based on the micro-datasets. for more details see appendix 12.1

- Table 1: LoRA WiSE Benchmark Overview. The dataset comprises over 25,000 weights checkpoints drawn from more than 2000 independent LoRA models, spanning different dataset sizes, backbones, ranks, and personalization sets.

Data Range Dataset Sizes Source Backbone LoRA Rank # of Models

8 300 16 300 32 300

Low 1,2,3,4,5,6 Concept101 SD 1.5

16 300 32 300 64 300

SD 1.5

Medium 1,10,20,30,40,50 ImageNet

SD 2 32 300 High 1,50,100,500,1000 ImageNet SD 1.5 32 250

### 6 Experiments

- 6.1 Experimental Setup We evaluate DSiRe on the LoRA-WiSE benchmark. Unless mentioned otherwise, we use Stable

- Diffusion 1.5 as the pre-trained model, which we fine-tune using a LoRA of rank 32 (denoted as above ∆W). As for the training set, we use 15 different personalization sets, to train each ∆W model, resulting in a training set of 90 weight samples for low and medium dataset sizes, and 75 weight samples for our higher (1 − 1000) range. We evaluate DSiRe using 35 additional personalization sets. We note, We repeat this experiment 10 times, including subset sampling. The reported performance metrics are the average and standard deviation over the experiments.

Baseline. We compare DSiRe to a baseline, denoted as Frobenius-NN, which predicts the dataset size using a nearest neighbor classifier on top of the Frobenius norms of the layers’ LoRA weights. Similar to DSiRe, the Frobenius-NN is fitted separately to each layer, and then a majority vote rule is applied to select the prediction from all layer-wise predictions. The analysis in Sec 3 provides motivation for this baseline.

Evaluation metrics. As described in Sec. 4.1, our main evaluation metric is Mean Absolute Error (MAE). For completeness, we choose to report two complementary metrics as well: (i) Accuracy. (ii) Mean Absolute Percentage Error (MAPE). Since DSiRe predicts dataset sizes, simple accuracy does not adequately measure its effectiveness, e.g., predicting 4 when the true value is 5 is not as bad as predicting 1. We therefore provide MAPE scores as well, which compute the percentile from the ground truth that is equal to the absolute error.

- 6.2 Results

We begin by evaluating on 1 − 6 fine-tuning images. When using both the singular values as features for DSiRe and the Frobenius norm for Frobenius-NN, we find that they yield relatively successful results in recovering the dataset size.

These low data range results are presented in Table 2. As DSiRe outperforms Frobenius-NN by a small margin (> 3%), we conclude that the results align with our analysis (see Sec. 3.2), which demonstrate that both singular values and Frobenius norm are indeed predictive of the dataset size.

Mid-range fine-tuning dataset sizes (10 − 50 images) are common in artistic LoRA fine-tuning. We therefore present the results of our method using 1 − 50 fine-tuning images in Table 2, showing that DSiRe performs well with an MAE of 1.48. In this data range, the Frobenius-NN baseline achieves comparable results to DSiRe across all metrics, demonstrating good performance. While the absolute MAE value is larger than in the low data range case, it is relatively small compared to the range of data sizes. The accuracy and mean absolute percentage error (MAPE) scores of both methods further support this observation. Fig. 4 shows another favorable property of our approach: its mistakes are usually near hits, i.e., large errors between ground truth and predicted labels are rare.

- Table 2: Performance Comparison of Dataset Size Recovery Methods across Different Ranges. Performance of Frobenius-NN and DSiRe across different data ranges (1−6,1−50,1−1000) using Stable Diffusion 1.5. These decent results aligns with our analysis (see Sec 3), showed that both SVD and Frobenius norm are effective features for dataset size recovery. However, DSiRe outperfomrs the Frobenius-NN on all evaluation metrics.

Data Range Method MAE ↓ MAPE(%) ↓ Acc(%) ↑ 1-6

Frobenius-NN 0.43 ±0.04 15.14 ±2.12 65.29 ±2.42

- DSiRe 0.36 ±0.04 11.36 ±1.55 69.30 ±3.83

1-50

Frobenius-NN 1.56 ±0.19 4.16 ±0.75 85.33 ±1.81

- DSiRe 1.48 ±0.21 3.97 ±0.73 86.10 ±1.99

1-1000

Frobenius-NN 68.62 ±5.53 9.25 ±1.21 86.51 ±1.12 DSiRe 41.77 ±6.61 5.96 ±1.46 91.90 ±1.28

- Table 3: Robustness of Dataset Size Recovery Methods on Stable Diffusion 2.0. DSiRe recovers dataset size more effectively than Frobenius-NN for the medium data range (1 − 50) using Stable

- Diffusion 2.0. This supports the benefit from a more expressive representation given by the singular values, independent to the specific backbone model used.

Method MAE↓ MAPE(%) ↓ Acc(%) ↑

Frobenius-NN 2.95 ±0.28 11.99 ±3.93 73.90 ±2.21 DSiRe 2.51 ±0.22 7.46 ±0.95 77.43 ±1.70

At larger data quantities, dataset size recovery could aid in better understanding data collection quantities needed for fine-tuning. Therefore, we conducted an additional experiment using models trained with higher data ranges, having 1,50,100,500 and 1000 image samples per model (note that here we have 5 dataset size classes). Results, presented in Tab. 2, shows DSiRe is able to detect the dataset size with more than 90% accuracy, and a MAPE score of only 6%. Additionally, in Fig. 6 we show the confusion matrix generated by DSiRe, where we see that most of the errors happen between adjacent classes.

Other Backbone The LoRA fine-tuning technique is commonly used by popular text-to-image models. A desirable aspect of our paradigm is being robust to model architecture. In this part, We test the robustness of DSiRe to the backbone model by evaluating it on Stable Diffusion 2.0. We note that these models do not share pre-training weights, as Stable Diffusion 2.0 was not fine-tuned from a previous version. Tab. 3 shows that DSiRE performs well on Stable Diffusion 2.0, reaching around 77% accuracy. This provides evidence for the correlation between the singular values and dataset size is not specific to one backbone alone.

### 7 Ablation studies

#### 7.1 Number of Micro-Datasets

While our attack is data driven and requires access to the pre-trained model, we find that only a few examples are needed for DSiRe to perform well. E.g., in our medium data size range, our model can reach 86.4% accuracy using only 5 micro-datasets for training. The full results, presented in Fig. 5, showcases that while more samples (fine-tuned models) improves the accuracy of our predictor, even a single micro-dataset is sufficient to achieve around 80% accuracy. This shows that our method is robust to the number of micro-datasets used, even to very small numbers.

#### 7.2 Robustness to LoRA Hyper-Parameters

While DSiRe performs well in our settings, it is important to show that it remains robust to hyperparameters. In this section, we will ablate our method using the LoRA-WiSE benchmark by testing DSiRe with models fine-tuned using other recipes. We check two common variations of the

[Figure 9]

Figure 4: DSiRe Confusion Matrix for Medium Data Range in a single experiment. Illustrating DSiRes accuracy in the range of 1 − 50 samples, shows that most of the errors are near misses, highlighting DSiRe’s precision in dataset size recovery.

[Figure 10]

Figure 5: DSiRes Micro-Dataset Size vs. Accuracy, reportedonmediumdatasizerange(1−50). Even a single micro-dataset is sufficient for DSiRe to reach 80% accuracy. This demonstrates its effectiveness with limited training data.

- Table 4: DSiRe Performance with Different LoRA Ranks. Desire consistently achieves high accuracy across both low and medium ranges, indicating its robustness regardless of LoRA rank variations.

Data Range LoRA Rank MAE ↓ MAPE(%) ↓ Acc(%) ↑

8 0.43 ±0.04 14.8 ±2.3 66 ±3.08 16 0.42 ±0.03 12.4 ±1.11 67.7 ±2.3

1 − 6

- 32 0.36 ±0.04 11.36 ±1.55 69.30 ±3.83

1 − 50

16 1.67 ±0.17 4.32 ±0.46 84.04 ±1.85

- 32 1.48 ±0.21 3.97 ±0.73 86.10 ±1.99 64 1.41 ±0.39 3.90 ±1.30 86.58 ±3.45

LoRA models: (1) LoRA rank (2) seeding. For additional ablation studies on the training steps, batch size, and used classifier see appendix. 11.2.

LoRA Rank. Starting with the LoRa rank ablation, we train DSiRe for different varying LoRA ranks. Tab. 4 shows the results for medium and low data ranges. Our method is robust to the LoRA rank, achieving similar results in all 3 tested ranks for both ranges.

Seeding. While in the standard recipe, all models use seed = 0, we also tested the case where all seeds were selected randomly. Tab. 7 shows that the variation in seeds only reduces accuracy by around 4%, and that MAE decreases by less than 0.5. This is not a small change, given that the gap between possible dataset size values is 10.

#### 7.3 Choice of Classifier

We tested various predictors, including parametric and non-parametric classifiers, as shown in Table 5. Except the NN-full model baseline, our pipeline remains unchanged, all classifiers are fitted separately to each layer and a majority vote rule selects the label from all layer-wise predictions. In contrast, the NN-full model uses a kNN classifier which is fitted to all the layers simultaneously. The results show that the choice of classifier affects the performance significantly. Furthermore, these results confirm our hypothesis from Sec. 3: while each layer is predictive of the dataset size, the accuracy of individual layers is not sufficient. As the layer-wise NN predictor can pool information across layers, it helps DSiRe perform better than the other methods.

### 8 Discussion

Performance at Low Data Ranges. While our approach shows promising results, there is room for improvement in lower data regimes, where DSiRe reaches less than 80% accuracy. Improving these results will provide tighter upper bounds for membership inference and model inversion attacks.

- Table 5: Performance of various predictors, medium dataset size range (1 − 50). DSiRe with majority voting performs best by combining predictions from multiple layers, opposed to the nearest neighbor - full model baseline, which uses the spectra of all layers together as features for a single prediction.

Model MAE ↓ MAPE(%) ↓ Acc(%) ↑

NN - full model 7.33 ±0.75 33.37 ±7.71 48.61 ±3.53 Ridge Regression 8.05 ±0.08 38.95 ±6.24 37.09 ±0.63 GDA 2.89 ±0.48 7.87 ±1.30 74.14 ±3.65 DSiRe - average vote 3.57 ±0.36 19.05 ±3.37 64.62 ±3.58 DSiRe - majority vote 1.48 ±0.21 3.97 ±0.73 86.10 ±1.99

- Table 6: Defense against DSiRe, on medium data range (1 − 50). We show how the simplest data augmentation (horizontal flipping), can harm DSiRes performance, reducing its accuracy by more than 26%.

Ablation MAE ↓ MAPE(%) ↓ Acc(%) ↑

Defense 4.62 ±0.83 17.62 ±4.82 59.71 ±5.88 DSiRe 1.48 ±0.21 3.97 ±0.73 86.10 ±1.99

Data Driven Solution. Our method is data driven as it requires training multiple models from each dataset size. However, our analysis shows there is correlation between the Frobenius norm and dataset size (see Fig. 2a). This insight could be a stepping stone in developing a data-free solution.

Pre-training dataset size recovery. Another interesting application of dataset size recovery is for pre-training cases. Lower bounding the required number of training set samples for foundation models will have a substantial impact on the research community. Answering this question would require scaling up our method to much larger dataset sizes and weight matrix dimensions.

Defense against DSiRe. Data augmentation aims to artificially increase the size of the data set by generating more image transformations, such as flipping and rotation on existing training images. Motivated by this observation, we employ data augmentation during the fine-tuning process to inflate the actual number of unique images used for fine-tuning, adding more variation to every image in the dataset. To avoid hurting model convergence during fine-tuning or the quality of the generated images, we applied only the flip augmentation technique to each image in the dataset. This process reduced the correlation between the singular values and the true number of images, as illustrated in Fig. 6.

### 9 Social impact

Webelieveournewlyproposedtaskcanpositivelyimpactboththeresearchanddigitalartscommunities. When detecting small and mid range dataset sizes, our motivation is to establish an upper bound for membership inference attacks, promoting privacy aware deployment of LoRA models. For larger dataset sizes, it will increase the information on the resources requires to successfully train models. This is often for researchers that need to collect expensive datasets for new fine-tuning tasks e.g., [50] and [8].

### 10 Conclusion

We introduced the new task of dataset size recovery and proposed a method, DSiRe, for learning a predictor for this task for models that use LoRA fine-tuning. Our method showed promising results on a new large-scale dataset that we released. We believe our work can provide an upper bound for model inversion and membership inference attacks. It can also provide both researchers and stock photography owners with a quantitative analytic estimating the data cost for fine-tuning models.

### References

- [1] Dreambooth training readme. https://github.com/huggingface/diffusers/blob/ main/examples/dreambooth/README.md. Accessed: 01/02/24.
- [2] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. Break-ascene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers. Association for Computing Machinery, 2023.
- [3] Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, and Xi Yin. Spatext: Spatio-textual representation for controllable image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [5] Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and Privacy (SP), pages 1897–1914. IEEE, 2022.
- [6] Dingfan Chen, Ning Yu, Yang Zhang, and Mario Fritz. Gan-leaks: A taxonomy of membership inference attacks against generative models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pages 343–362, 2020.
- [7] Kanghyun Choi, Deokki Hong, Noseong Park, Youngsok Kim, and Jinho Lee. Qimera: Data-free quantization with synthetic boundary supporting samples. Advances in Neural Information Processing Systems, 34:14835–14847, 2021.
- [8] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [10] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.
- [11] Jinhao Duan, Fei Kong, Shiqi Wang, Xiaoshuang Shi, and Kaidi Xu. Are diffusion models vulnerable to membership inference attacks? In International Conference on Machine Learning, pages 8717–8730. PMLR, 2023.
- [12] Gongfan Fang, Kanya Mo, Xinchao Wang, Jie Song, Shitao Bei, Haofei Zhang, and Mingli Song. Up to 100x faster data-free knowledge distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 6597–6604, 2022.
- [13] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security, pages 1322–1333, 2015.
- [14] Niv Haim, Gal Vardi, Gilad Yehudai, Ohad Shamir, and Michal Irani. Reconstructing training data from trained neural networks. Advances in Neural Information Processing Systems, 35: 22911–22924, 2022.
- [15] Jamie Hayes, Luca Melis, George Danezis, and Emiliano De Cristofaro. Logan: Membership inference attacks against generative models. arXiv preprint arXiv:1705.07663, 2017.
- [16] Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a unified view of parameter-efficient transfer learning. ArXiv, abs/2110.04366, 2021. URL https://api.semanticscholar.org/CorpusID:238583580.

- [17] Zecheng He, Tianwei Zhang, and Ruby B Lee. Model inversion attacks against collaborative inference. In Proceedings of the 35th Annual Computer Security Applications Conference, pages 148–162, 2019.
- [18] Benjamin Hilprecht, Martin Härterich, and Daniel Bernau. Monte carlo and reconstruction membership inference attacks against generative models. Proceedings on Privacy Enhancing Technologies, 2019.
- [19] Eliahu Horwitz, Jonathan Kahana, and Yedid Hoshen. Recovering the pre-fine-tuning weights of generative models. arXiv preprint arXiv:2402.10208, 2024.
- [20] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019.
- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [22] Hailong Hu and Jun Pang. Membership inference of diffusion models. arXiv preprint arXiv:2301.09956, 2023.
- [23] Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S Yu, and Xuyun Zhang. Membership inference attacks on machine learning: A survey. ACM Computing Surveys (CSUR), 54(11s):1–37, 2022.
- [24] Nam Hyeon-Woo, Moon Ye-Bin, and Tae-Hyun Oh. Fedpara: Low-rank hadamard product for communication-efficient federated learning. arXiv preprint arXiv:2108.06098, 2021.
- [25] Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A Choquette-Choo, Nicholas Carlini, and Florian Tramer. Students parrot their teachers: Membership inference on model distillation. Advances in Neural Information Processing Systems, 36, 2024.
- [26] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision, pages 709–727. Springer, 2022.
- [27] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multiconcept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023.
- [28] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.
- [29] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.
- [30] Zhikai Li, Mengjuan Chen, Junrui Xiao, and Qingyi Gu. Psaq-vit v2: Toward accurate and general data-free quantization for vision transformers. IEEE Transactions on Neural Networks and Learning Systems, 2023.
- [31] Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950–1965, 2022.
- [32] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. AI Open, 2023.
- [33] Raphael Gontĳo Lopes, Stefano Fenu, and Thad Starner. Data-free knowledge distillation for deep neural networks. arXiv preprint arXiv:1710.07535, 2017.
- [34] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. https: //github.com/huggingface/peft, 2022.

- [35] Tomoya Matsumoto, Takayuki Miura, and Naoto Yanai. Membership inference attacks against diffusion models. In 2023 IEEE Security and Privacy Workshops (SPW), pages 77–83. IEEE, 2023.
- [36] Ngoc-Bao Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, and Ngai-Man Cheung. Re-thinking model inversion attacks against deep neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16384–16393, 2023.
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [39] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.
- [40] Alexandre Sablayrolles, Matthĳs Douze, Cordelia Schmid, Yann Ollivier, and Hervé Jégou. White-box vs black-box: Bayes optimal strategies for membership inference. In International Conference on Machine Learning, pages 5558–5567. PMLR, 2019.
- [41] Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, and Michael Backes. Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models. arXiv preprint arXiv:1806.01246, 2018.
- [42] Avital Shafran, Shmuel Peleg, and Yedid Hoshen. Membership inference attacks are easier on difficult problems. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14820–14829, 2021.
- [43] Renrong Shao, Wei Zhang, Jianhua Yin, and Jun Wang. Data-free knowledge distillation for fine-grained visual categorization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1515–1525, 2023.
- [44] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pages 3–18. IEEE, 2017.
- [45] HugoTouvron, ThibautLavril, GautierIzacard, XavierMartinet, Marie-AnneLachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [46] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022.
- [47] Zhen Wang, Rameswar Panda, Leonid Karlinsky, Rogerio Feris, Huan Sun, and Yoon Kim. Multitask prompt tuning enables parameter-efficient transfer learning. arXiv preprint arXiv:2303.02861, 2023.
- [48] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. ArXiv, abs/2305.16213, 2023. URL https://api.semanticscholar.org/CorpusID: 258887357.
- [49] Lauren Watson, Chuan Guo, Graham Cormode, and Alex Sablayrolles. On the importance of difficulty calibration in membership inference attacks. arXiv preprint arXiv:2111.08440, 2021.

- [50] Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion, 2024.
- [51] Shoukai Xu, Haokun Li, Bohan Zhuang, Jing Liu, Jiezhang Cao, Chuangrun Liang, and Mingkui Tan. Generative low-bitwidth data free quantization. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16, pages 1–17. Springer, 2020.
- [52] Ziqi Yang, Jiyi Zhang, Ee-Chien Chang, and Zhenkai Liang. Neural network inversion in adversarial setting via background knowledge alignment. In Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, pages 225–240, 2019.
- [53] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [54] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF), pages 268–282. IEEE, 2018.
- [55] Hongxu Yin, Pavlo Molchanov, Jose M Alvarez, Zhizhong Li, Arun Mallya, Derek Hoiem, Niraj K Jha, and Jan Kautz. Dreaming to distill: Data-free knowledge transfer via deepinversion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8715–8724, 2020.
- [56] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133, 2022.
- [57] Lin Zhang, Li Shen, Liang Ding, Dacheng Tao, and Ling-Yu Duan. Fine-tuning global model via data-free knowledge distillation for non-iid federated learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10174–10183, 2022.
- [58] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, October 2023.
- [59] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. arXiv preprint arXiv:2303.10512, 2023.
- [60] ZhuangdiZhu, JunyuanHong, andJiayuZhou. Data-freeknowledgedistillationforheterogeneous federated learning. In International conference on machine learning, pages 12878–12889. PMLR, 2021.

### 11 Appendix

#### 11.1 Additional Ablation Studies

We provide more ablation studies of our method. Specifically, we test the training steps, batch size, used classifier type and used LoRA matrices.

#### 11.2 Robustness to LoRA Hyper-Parameters

Batch Size. We ablate the batch size, results at shown in Tab. 7. Despite the change in batch size, DSiRe demonstrates robust performance, achieving a MAE score of 1.94 compared to the original 1.48. Additionally, the accuracy only decreases by less than 5%, indicating that our method maintains comparable effectiveness even with different batch sizes.

Table 7: DSiRe performance using different LoRA hyper-parameters. Medium data range Ablation MAE↓ MAPE(%) ↓ Acc(%) ↑

Batch size 1.94 ±0.26 9.35 ±1.34 81.50 ±2.55 Baseline 1.48 ±0.21 3.97 ±0.73 86.10 ±1.99

Training Steps. To train DSiRe, we first fine-tune a set of LoRA models. These models follow a certain recipe, with a specific amount of training steps. To evaluate robustness, we tested DSiRe on models fine-tuned at different steps, with 1200 steps as our baseline. As shown in Tab 8, DSiRe consistintly achieves comparable results across different fintuning steps. e.g. the MAE score ranges from 2.43 at 300 steps to 1.40 at 1400 steps, with accuracy variations within 10%.

- Table 8: DSiRe performance on different checkpoints of Stable Diffusion 1.5 rank 16 range 1 − 50 #Steps MAE↓ MAPE(%) ↓ Acc(%) ↑

300 2.43 ±0.20 6.82 ±0.78 77.90 ±1.49 400 2.39 ±0.20 6.72 ±0.76 78.38 ±1.49 500 2.05 ±0.15 5.55 ±0.59 81.33 ±1.60 600 1.86 ±0.10 4.59 ±0.34 82.76 ±0.86 700 1.89 ±0.21 5.13 ±0.77 82.00 ±2.01 800 1.71 ±0.29 4.59 ±0.89 83.67 ±2.68 900 1.60 ±0.22 4.21 ±0.69 85.14 ±2.04 1000 1.62 ±0.21 4.69 ±0.70 85.10 ±1.77 1100 1.58 ±0.19 4.50 ±0.90 84.48 ±1.32 1200 1.48 ±0.21 3.97 ±0.73 86.10 ±1.99 1300 1.46 ±0.15 3.84 ±0.51 86.29 ±1.55 1400 1.40 ±0.20 3.73 ±0.76 86.76 ±2.08

11.3 Choice of LoRA Matrices

Seeing in Sec. 3 that not all layers are similar in behavior, we test to see if different LoRA matrices also capture different information. In Tab. 9, we find that indeed different LoRA matrices capture different information, and lead to substantially other performances. Unsurprisingly, we also find that using all the LoRA matrices combined yields the best result.

- Table 9: DSiRe performance on different layers of LoRA of the UNet in Stable Diffusion 1.5, range 1 − 50:

# Layer Type MAE ↓ MAPE(%) ↓ Acc(%)

- A 1.9 ±0.29 5.63 ±1.26 82.52 ±2.20
- B 1.57 ±0.19 4.07 ±0.65 84.90 ±2.09 BA 1.61 ±0.16 4.22 ±0.47 85.00 ±1.45 full model 1.48 ±0.21 3.97 ±0.73 86.10 ±1.99

### 12 Higher Data Regimes Analysis

To better understand the results on higher data regimes we provide here the confusion matrix of DSiReusing 1 − 1000 training samples. We can see that most of the errors are in larger data classes.

[Figure 11]

Figure 6: DSiRe Confusion matrix in High data regime. Illustrating DSiRe’s accuracy in the range data size (1 − 1000) for a single experiment, showing that most predictions are correct or near misses, highlighting the DSiRe’s precision in dataset size recovery.

12.1 Implementations details 12.1.1 LoRA-WiSE. we now elaborate the implementations details of the LoRA-WiSE bench dataset.

Datasets in all ranges 1 − 6, 1 − 50, 1 − 1000. As the Pre-Ft models we use runwayml/stable-diffusion-v1-5 and stabilityai/stable-diffusion-2 [38]. We finetune the models using the PEFT library [34]. We use the script train_dreambooth_lora.py [39] with the diffusers library [46]. we use the standard recipe to fine-tune the models in all ranges[1] see tab11. we use batch size 8 for range 1-1000 for computational resources and 1000 training step. in the ablations we don’t change any hyper-parameter except the ablate one.

Each model took approximately 30-50 minutes to fine-tune. We used GPUs with 16-21GB of RAM, such as the NVIDIA RTX A5000. The DSiRe process, however, does not require GPUs and can run on CPUs.

Experiment Settings. In addition to the experiment settings described in Section 6, we used the following configurations for our models:

- For models in the ranges 1-6 and 1-50, we used the checkpoint at iteration 1200. - For models in the range 1-1000, we used the checkpoint at iteration 1000.

We used a fixed seed of 42 to split the train and test data for every experiment.

Layer weigh matrices In line with our analysis see Sec.3, given n example weights for each Ai,Bi we wish to build a separate classifier for each one. Knowing the relation between the singular values and the dataset size, we decompose each matrix using the singular value decomposition (SVD), and use the ordered set of singular values as features for our classifiers. Formally, we note the singular

Table 10: ranges 1-6 and 1-50 Name Value

lora_rank (r) r lr 1e − 4 batch_size 1 gradient_accumulation_steps 1 learning_rate_scheduler Constant training_steps 1400 warmup_ratio 0

imagenet[9] concept101[27]

dataset

seeds 0

Table 11: range 1-1000 Hyperparameters

Name Value

lora_rank (r) 32 lr 1e − 4 batch_size 8 gradient_accumulation_steps 1 learning_rate_scheduler Constant training_steps 1000 warmup_ratio 0 dataset imagenet seeds 0

values of Aij as ΣA

. We include the singular values of Bij · Aij denoted as ΣB

and the singular values of Bij as ΣB

ij

ij

ij·Aij. Additionally, our observations indicate that the product Bi · Ai also provides useful information for data size recovery. Thus, for each LoRA matrix, we obtain a dataset with n samples, where each sample is a vector of singular values Σij, paired with a corresponding label yj. Our method then trains three separate kNN-classifiers with K = 1 for each layer over (i) Ai (ii) Bi and (iii) BiAi. At inference time, the predictions from all classifiers are merged by majority voting.

