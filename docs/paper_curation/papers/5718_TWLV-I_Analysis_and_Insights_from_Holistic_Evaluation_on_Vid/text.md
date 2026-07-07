# arXiv:2408.11318v2[cs.CV]23Aug2024

## TWLV-I: Analysis and Insights from Holistic Evaluation on Video Foundation Models

### Twelve Labs

[Figure 1]

#### Abstract

In this work, we discuss evaluating video foundation models in a fair and robust manner. Unlike language or image foundation models, many video foundation models are evaluated with differing parameters (such as sampling rate, number of frames, pretraining steps, etc.), making fair and robust comparisons challenging. Therefore, we present a carefully designed evaluation framework for measuring two core capabilities of video comprehension: appearance and motion understanding. Our findings reveal that existing video foundation models, whether text-supervised like UMT or InternVideo2, or self-supervised like V-JEPA, exhibit limitations in at least one of these capabilities. As an alternative, we introduce TWLV-I, a new video foundation model that constructs robust visual representations for both motion- and appearance-based videos1. Based on the average top-1 accuracy of linear probing on five action recognition benchmarks, pretrained only on publicly accessible datasets2, our model shows a 4.6%p improvement compared to VJEPA (ViT-L) and a 7.7%p improvement compared to UMT (ViT-L). Even when compared to much larger models, our model demonstrates a 7.2%p improvement compared to DFN (ViT-H), a 2.7%p improvement compared to V-JEPA (ViTH) and a 2.8%p improvement compared to InternVideo2 (ViT-g). We provide embedding vectors obtained by TWLV-I from videos of several commonly used video benchmarks, along with evaluation source code that can directly utilize these embeddings. The code is available on https://github.com/twelvelabs-io/ video-embeddings-evaluation-framework.

#### 1 Introduction

Video is everyone’s first language. From the moment humans are born, they learn about the world by seeing videos even before using language. Therefore, similar to human languages, developing a video understanding system is essential to achieve the ability to perceive the world accurately. Since videos are sequences of images, it is crucial to recognize what appears in each frame (appearance). In addition to appearance, videos contain intrinsic characteristics not present in images: motion. In this technical report, we consolidate and improve existing evaluation methods for video understanding, while also introducing several new methodologies to propose a comprehensive and fair evaluation framework from the perspectives of appearance and motion. Along with the evaluation framework, we introduce a video foundation model that can comprehensively understand both the appearance and motion in videos.

A Foundation Model (FM) represents a model trained on large-scale data with diverse supervision, possessing the potential to be applied to various tasks within a domain rather than targeting a specific task [4]. Recently, Language Foundation Models (LFMs), represented by Large Language

- 1Please cite this paper as (Twelve Labs, 2024). Please see the Authorship Section at the end of this report for the full list of contributors.
- 2This work is strictly intended for academic purposes. Note that Twelve Labs’ Embedding API is powered by a model that excludes all non-commercial data.

[Figure 2]

[Figure 3]

- Figure 1: Comparison with video foundation models in the same scale. TWLV-I can transfer to various tasks and perform comparable or even superior to the state-of-the-art models. We present the best performance among the competitors, as well as the performance of our model. † denotes the model that uses the dataset in the pretraining stage. All compared models in this figure are on the ViT-L scale.

[Figure 4]

[Figure 5]

- Figure 2: Performance on appearance- vs. motion-centric benchmarks. Our model can handle both appearance- and motion-centric benchmarks reasonably well. † denotes that the pretraining dataset of the model includes the downstream dataset. V-JEPA uses Something-Something-v2 in the pretraining stage. InternVideo2 is pretrained on Moments-in-Time and Something-Something-v2.

Models (LLMs), have demonstrated that a single fixed model can effectively handle multiple tasks in general [1, 33, 6, 19, 7]. In the field of computer vision, Image Foundation Models (IFMs) follow this trend by proposing frameworks to encode general information obtainable from images into embedding vectors [29, 30]. Due to extensive research on large-scale model structures, large datasets, and appropriate training methodologies, fixed general models in the image domain have demonstrated superior or compatible performance across various tasks compared to expert models trained specifically for those tasks [37].

In the context of video analysis, one possible way to leverage the success of existing IFMs is to treat video as a sequence of images: sampling each frame, embedding them using image encoders, and then

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

(a) TWLV-I (b) V-JEPA (c) TWLV-I (d) InternVideo2

- Figure 3: (a), (b): t-SNE visualizations of embeddings obtained from the K400 validation set using TWLV-I and V-JEPA. (c), (d): LDA visualizations of embeddings from the ‘Moving something up’ class of the SSv2 validation set and their reversed versions using TWLV-I and InternVideo2. Details can be found in Figures 6, 9, 8, and Section 2.7. As seen from (a) and (b), V-JEPA lacks the capability to cluster embeddings of the same class when extracting embeddings from K400, where understanding the visual appearance of each frame is important. From (c) and (d), it is evident that InternVideo2 struggles to distinguish between videos with objects moving in a specific direction and their reversed versions, indicating a limitation in motion understanding capability. In contrast, TWLV-I demonstrates both of these capabilities.

aggregating the embeddings [26]. However, it is known that this method has limitations in capturing the detailed motion in videos [40] and our experiments also demonstrate this tendency, as shown in Figure 2. Additionally, previous work [36] observed that the models trained on the images tend to output similar embeddings for the contiguous frames. This approach is akin to dividing the frames into T isolated rooms, having T individuals enter each room to observe their respective frames, and then discussing to compile the information. Such a method makes it difficult to capture the detailed motion of objects within the video and the small changes between frames.

To address this issue, recent works propose utilizing the Vision Transformer [9] architecture to process all frames simultaneously. These methods can be categorized into two main approaches based on the training methodology. The first approach utilizes distillation, where high-performance IFMs, such as CLIP [30], are used as a teacher for distillation [24, 37]. The second approach is based on masked modeling, where the model is trained to predict the missing information of inputs from the given partial input [32, 35, 14, 3]. Since a video is essentially a sequence of images, it is crucial to understand the appearance of objects in the given data. Unlike images, videos also involve the time axis, making it essential to understand the motion of the objects. However, as shown in Figure 2 and Figure 3, distillation-based methods such as UMT and InternVideo2 struggle with motion-sensitive benchmarks such as Something-Something-v2 [15] and Diving-48 [25]. On the other hand, masked modeling-based methods (e.g., V-JEPA) underperform on appearance-centric benchmarks such as Kinetics-400 [21] and Moments-in-Time [28].

In this work, we introduce TWLV-I, a model that can provide an embedding vector for a video, capturing both appearance and motion. As shown in Figure 1, although TWLV-I is trained solely with publicly available datasets described in Table 1, demonstrates noticeable performance on both appearance- and motion-centric action recognition benchmark datasets. Furthermore, in addition to the action recognition task, TWLV-I achieves state-of-the-art performance on various video-centric tasks such as temporal action localization, spatiotemporal action localization, and temporal action segmentation, showing its strong spatial and temporal understanding capabilities.

To conduct a detailed analysis of various VFMs, including TWLV-I, we utilize several commonly used evaluation and analysis methods [40, 3]. However, given that these existing methods are insufficient for a comprehensive analysis of VFMs, we have improved some of them and proposed new analytical approaches. For example, in Section 2.7.2, we validate whether a VFM can distinguish videos based solely on the direction of motion, independent of appearance. This is achieved by visualizing the embeddings of original videos and their reversed versions to determine if their embedding distributions are separable.

In this technical report, we not only introduce TWLV-I but also highlight several important perspectives and evaluation methods that are essential for advancing the field of video understanding. Additionally, we propose key directions for future research to further the development of this area.

Table 1: Summary of datasets used for pretraining TWLV-I. Pretraining Dataset Type # of Clips (Images) Kinetics 710 Video 658K Howto360K Video 360K WebVid10M Video 10.73M COCO Image 113K SBU Captions Image 860K Visual Genome Image 100K CC3M Image 2.88M CC12M Image 11.00M

Table 2: Summary of benchmarks used for evaluation.

|Task<br><br>|Dataset # Videos (train / val) Clip Length Domain Note|
|---|---|
|AR<br><br>|Kinetics 400 235,693 / 19,165 10 seconds Web Appearance MiT 791,246 / 33,898 3 seconds Web Appearance SthSth-v2 168,913 / 24,777 2-6 seconds Crowd-source Motion Diving 48 15,027 / 1,970 5 seconds Web Motion Epic Kitchens 67,217 / 9,668 ∼10 seconds Crowd-source Ego-centric|
|TAL<br><br>|ActivityNet v1.3 10,024 / 4,926 5-10 minutes Web Temporal THUMOS14 200 / 212 4 minutes Web Temporal|
|STAL<br><br>|AVA v2.2 210,634 / 57,371 15 minutes Movie Spatio-temporal|
|TAS<br><br>|50Salads 40 / 10 5 minutes Web Temporal GTEA 21 / 7 1-3 minutes Web Ego-centric & Temporal Breakfast ∼1,284 / ∼428 2-3 minutes Web Temporal|

#### 2 TWLV-I & Video Foundation Model Evaluation Framework

In this section, we analyze VAM’s feature space compared to previous Video Foundation Models (VFMs). For this purpose, we focus on four representative tasks: Action Recognition (AR), Temporal Action Localization (TAL), Spatio-temporal Action Localization (STAL), and Temporal Action Segmentation (TAS). We evaluate the performance of each VFM across different settings, benchmarks, and evaluation methods. Details of the benchmarks used for evaluation are provided in Table 2. We compare our model with recent state-of-the-art VFMs, including Unmasked Teacher (UMT) [24], V-JEPA [3], and InternVideo2 [37]. Additionally, we include the Data Filtering Network (DFN) [12] as an appearance-centric model that processes video frame by frame, rather than handling it as a whole. To extract the embedding of a video, we apply a mean-pooling operation to the frame-wise embeddings. This strategy can be seen as a way to utilize an Image Foundation Model (IFM) in the video domain.

##### 2.1 Training Details and Frame Sampling

Architecture. We adopt a Vision Transformer (ViT) [9] architecture, specifically ViT-B (Base, 86M parameters) and ViT-L (Large, 307M parameters). An input video is tokenized into multiple patches and then processed through the transformer to produce patch-wise embeddings, which are subsequently pooled to obtain the overall embedding of the input.

Pretrain Dataset. For pretraining, we use Kinetics-710 [23], HowTo360K (a subset of HowTo100M [27]), WebVid10M [2], and a mixture of 15M publicly available image datasets to enhance image understanding capabilities. Details of the pretraining datasets can be found in Table 1.

Training Objective. Regarding the objective function, we leverage the strengths of existing distillation-based and masked modeling-based methods. Using the masked modeling schema as the baseline, we diversify the reconstruction targets to find the optimal objective for training a robust model that excels in both motion and appearance understanding. We train our model from scratch with this objective and the aforementioned datasets.

Frame Sampling. Because the computational complexity of the ViT architecture increases quadratically with the number of tokens, there is a constraint on the maximum number of frames that can be processed at once. Therefore, frames of an input video must be sub-sampled before being provided to the model. The most straightforward method is to sample the N frames with a uniform stride (at fixed

|Task Head<br><br>[Figure 10]|
|---|

|Task Head<br><br>[Figure 11]|
|---|

Clip-level Embeddings

Avg

Video-level Embedding (Multi-clip)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
|[Figure 12]<br><br>TWLV-I<br><br>[Figure 13]| | | | | | | | |

Multi-clip Sampling

- Figure 4: Overall architecture of the evaluation framework including TWLV-I. In Multi-Clip Embedding, the video is divided into multiple clips, and an embedding is produced from each clip. These clip-level embeddings are either all passed to the downstream task or averaged before being passed to the task head.

intervals), regardless of the video’s length (Uniform Embedding). However, as the video length increases, the time intervals between frames become longer, potentially losing temporal coherence between frames. To address this, for specific tasks, instead of Uniform Embedding, we fix the time length of a clip to T seconds and divide the video into M clips of T seconds each. Then, N frames are sampled from each clip to obtain M embeddings (Multi-Clip Embedding). In this setting, the number of embeddings increases as the video lengthens. When representing the entire video with a single embedding, we average the M embeddings. Research on finding better methods than averaging to aggregate M embeddings into a single video embedding could be an interesting future research topic. The differences between the overall evaluation pipeline and the two frame sampling methods are shown in Figure 4.

##### 2.2 Action Recognition

Action Recognition (AR) is a video classification task that aims to classify videos into predefined human action categories. For the action recognition task, we adopt five representative benchmarks: Kinetics-400 (K400) [21], Something-Something-v2 (SSv2) [15], Moments-in-Time (MiT) [28], Diving-48 (DV48) [25], and Epic-Kitchens (EK) [8]. We note that K400 and MiT are known to be appearance-focused, while SSv2 and DV48 are motion-centric datasets [40, 3]. We utilize the multi-view classification method commonly used in action recognition tasks. Specifically, we first spatially resize the input video so that its short side length matches the input resolution. Then, we uniformly sample m clips along the long side direction and n clips at equal intervals in the temporal direction. This results in a total of m × n clips. We then calculate the class probabilities for each of these clips and obtain the final output by averaging these probabilities.

##### 2.2.1 Linear Probing

Evaluation Settings. To evaluate the action recognition capability in terms of the clip-wise embeddings, we first adopt linear probing, which involves freezing the feature extractor (i.e., backbone model) and training a linear classifier on top of it. The linear classifier consists of a weight matrix that matches the embedding vector dimension to the number of classes. Details of the hyperparameters are provided in Table 4.

Results and Discussion. As shown in Table 3, UMT and InternVideo2 demonstrate limitations on motion-centric benchmarks such as SSv2 and DV48, which alignes with the discussion in Section 1.

- Table 3: Linear probing on recognition benchmarks. † denotes that the pretraining dataset for the model includes the downstream dataset. We note that all the pretraining dataset of video models includes Kinetics-400 (K400).

Method Arch.

K400 MiT SSv2 DV48 EK Avg. IN-1K Top1 Top5 Top1 Top5 Top1 Top5 Top1 Top5 Top1 Top5 Top1 Top1 Top5

- UMTs1 ViT-B 66.17 86.25 26.87 51.60 17.46 47.92 17.46 47.92 35.38 77.58 32.67 70.84 90.55

- UMTs2 ViT-B 69.47 88.65 29.79 56.17 18.22 48.63 18.22 48.63 37.41 76.67 34.62 74.75 93.17 TWLV-I ViT-B 74.17 91.52 30.59 56.89 37.19 66.79 30.41 67.82 42.05 80.03 42.88 61.14 83.95

[Figure 14]

- UMTs1 ViT-L 74.50 91.35 32.87 60.02 24.74 52.11 22.69 55.23 37.37 76.39 38.43 76.05 93.47

- UMTs2 ViT-L 76.60 92.48 35.69 63.71 31.02 61.25 21.42 54.92 39.81 79.28 40.91 79.24 95.12 V-JEPA ViT-L 70.77 90.00 28.83 54.41 52.50† 80.56† 22.86 56.84 45.34 77.07 44.06 56.67 80.37

[Figure 15]

TWLV-I ViT-L 80.17 92.83 36.29 64.79 46.41 75.79 32.84 70.36 47.34 80.14 48.61 72.98 92.13 DFN ViT-H 77.99 92.62 37.27 65.21 26.28 54.52 27.36 64.62 38.07 77.32 41.39 84.55 97.12 V-JEPA ViT-H 70.75 89.36 28.95 54.19 53.41† 81.18† 29.34 67.82 47.30 78.40 45.95 58.26 80.96

- InternVideo2s1 ViT-g 78.20 92.97 38.47† 66.84† 26.18† 54.72† 25.13 59.34 37.09 78.59 41.01 79.78 95.31

- InternVideo2s2 ViT-g 81.81 94.61 40.52† 69.33† 40.13† 72.30† 23.71 54.82 42.69 81.03 45.77 80.06 95.38 TWLV-I (+SSv2) ViT-L 80.73 94.47 37.19 65.37 48.14† 78.67† 31.07 71.52 46.58 81.97 48.74 73.64 92.54

[Figure 16]

- Table 4: Hyperparameters for linear probing (LP) and attentive probing (AP) evaluation. The hyperparameters are from V-JEPA [3] with minor changes.

Hyper-parameter K400 MiT SSv2 DV48 EK IN-1K

- LP AP LP AP LP AP LP AP LP AP LP AP train_num_clips 1 views (m × n) 3 × 4 3 × 4 3 × 1 3 × 4 3 × 4 num_frames 16 16 16 16 16 temporal_stride 4 1 1 4 1 horizontal_flip True False False False True True random_resize (0.3, 1.0) aspect_ratio 0.75, 1.33 0.5, 2.0 0.75, 1.33 0.5, 2.0 0.75, 1.33 0.5, 2.0 0.75, 1.33 0.5, 2.0 0.75, 1.33 0.5, 2.0 0.75, 1.33 0.5, 2.0 batch_size 1024 256 1024 256 1024 256 1024 256 1024 256 4096 512 epochs 150 20 50 20 50 20 300 50 150 50 90 20 warmup 10 0 10 0 10 0 10 0 10 0 0 0 scheduler cosine decay lr 0.1 0.001 0.1 0.001 0.075 0.001 0.02 0.001 0.1 0.001 0.1 0.001 final_lr 0.0 weight_decay 0 1e-5 0 1e-5 0 1e-5 0 1e-5 0 1e-5 0 0.001

Similarly, DFN, being an image encoder, lacks the capability to effectively understand these benchmarks. In contrast, V-JEPA shows superior performance on motion-centric datasets. However, it exhibits limitations on appearance-centric benchmarks such as K400 and MiT. While the V-JEPA model at the ViT-H scale performs better than the ViT-L scale on SSv2, DV48, and EK, its performance either improves very slightly or even decreases on appearance-centric benchmarks (K400 and MiT). This indicates that scaling up V-JEPA enhances motion understanding capability but does not improve its already limited appearance capability. Unlike its competitors, our model generally performs well across all benchmarks. Notably, TWLV-I achieves the best performance on almost all benchmarks within the same architecture scale (both ViT-B and ViT-L). For the SSv2 dataset, our model outperforms even the larger-scale models like ViT-H (DFN) and ViT-g (InternVideo2), except for V-JEPA, which uses SSv2 in the pretraining stage. Furthermore, our ViT-L model surpasses other larger-scale models on the EK and DV48 benchmarks.

##### 2.2.2 Attentive Probing

Evaluation Settings. While linear probing can evaluate the quality of the clip-wise embeddings, it cannot fully reflect the models’ capabilities, especially since models trained with patch-level supervision lack the frame-level descriptor [10]. To address this, we also validate the models using attentive probing. This method involves training a single attention layer with a learnable class token on top of the frozen models. The output class token is then passed through a linear classifier, and we measure the top-1 accuracy. We follow the same training schedule as V-JEPA [3], with detailed training parameters provided in Table 4.

Results and Discussion. As shown in Table 5, the attentive probing results confirm that our model achieves superior performance across both appearance- and motion-centric benchmarks compared to other models. Moreover, these results provide additional insights not captured by the linear probing results, such as the observation that the performance of some models in stage 2 is lower than in stage

- 1. Specifically, for UMT, a performance drop occurs in all benchmarks except MiT. For InternVideo2,

- Table 5: Attentive probing on recognition benchmarks. † denotes that the pretraining dataset for the model includes the downstream dataset. ∗ indicates that the results are from V-JEPA [3]. The accuracy of VideoPrism is from the original paper. We note that all the pretraining dataset of video models includes Kinetics-400 (K400).

Method Arch.

K400 MiT SSv2 DV48 EK Avg. IN-1K Top1 Top5 Top1 Top5 Top1 Top5 Top1 Top5 Top1 Top5 Top1 Top1 Top5

- UMTs1 ViT-B 74.32 91.46 33.12 60.81 51.64 80.03 48.88 86.60 54.76 87.04 52.54 78.16 94.75

- UMTs2 ViT-B 74.15 91.31 33.80 62.12 50.30 79.30 38.93 77.82 52.74 87.16 49.98 79.83 95.73 TWLV-I ViT-B 77.66 93.47 35.06 63.26 58.74 85.69 66.70 93.10 58.31 88.50 59.29 70.27 90.72

[Figure 17]

OmniMAE∗ ViT-L 65.6 - - - 60.6 - - - - - - 75.1 VideoMAE∗ ViT-L 77.8 - - - 65.5 - - - - - - 71.1 Hiera∗ Hiera-L 75.5 - - - 64.2 - - - - - - 68.9 MVD∗ ViT-L 79.4 - - - 66.5 - - - - - - 73.3 -

- UMTs1 ViT-L 81.50 95.03 39.25 67.58 58.68 85.13 56.60 89.39 57.91 88.32 58.79 82.64 96.77

- UMTs2 ViT-L 80.70 94.81 39.39 68.69 55.53 83.56 45.38 84.35 55.70 88.12 55.34 83.71 97.17 V-JEPA ViT-L 80.19 94.63 38.06 67.02 70.53† 92.97† 74.72 95.58 62.34 89.62 65.17 73.60 91.92

[Figure 18]

TWLV-I ViT-L 84.46 96.58 41.23 70.45 68.95 91.84 74.98 96.29 63.70 90.16 66.66 79.19 95.26 DFN ViT-H 81.96 95.15 41.30 70.06 45.03 76.31 62.18 93.30 51.85 87.84 56.46 87.54 98.32 V-JEPA ViT-H 80.75 95.03 39.60 68.18 72.48† 93.77† 80.41 97.97 63.94 89.57 67.44 73.49 91.66

- InternVideo2s1 ViT-g 83.96 95.97 42.39† 71.35† 61.76† 86.77† 64.01 91.73 59.04 89.27 62.23 85.77 97.98

- InternVideo2s2 ViT-g 84.39 96.27 43.18† 72.52† 61.54† 87.77† 51.88 85.18 56.87 89.36 59.57 85.28 97.75 VideoPrism ViT-g 87.2 - 45.5 - 68.5 - 71.3 - - - - - -

[Figure 19]

TWLV-I (+SSv2) ViT-L 84.54 96.43 42.00 71.38 71.12† 93.02† 74.57 96.50 65.17 90.35 66.88 80.84 95.88

- Table 6: K-Nearest Neighbors. † denotes that the pretraining dataset for the model includes the downstream dataset. Note that all the pretraining dataset of video models includes Kinetics-400 (K400).

K400 SSv2

Method Arch. Uniform Clip Video Uniform Clip Video UMTs2 ViT-B 50.95 55.76 54.64 12.00 12.46 11.92

TWLV-I ViT-B 57.51 63.64 62.59 19.82 19.67 18.48 UMTs2 ViT-L 60.68 64.68 63.76 14.47 14.76 14.24 V-JEPA ViT-L 39.95 47.62 46.00 25.82† 25.93† 23.23†

[Figure 20]

TWLV-I ViT-L 65.97 70.82 69.88 19.47 19.73 19.11 DFN ViT-H 63.78 68.07 66.97 15.30 16.21 15.62 V-JEPA ViT-H 34.46 42.45 40.68 24.67† 24.90† 21.54† InternVideo2 ViT-g 76.15 79.43 78.46 22.65 23.38 22.86

[Figure 21]

TWLV-I (+SSv2) ViT-L 67.83 72.14 71.21 21.99 22.33 21.94

[Figure 22]

the drop is observed in all benchmarks except K400 and MiT. Both UMT and InternVideo2 are trained with patch-level loss in stage 1, while state 2 involves aligning embedding vectors with other domains. This indicates that stage 2 weakens the patch-wise representation, which is particularly detrimental for motion-centric datasets, as evidenced by the significant performance drop in DV48. Our model shows even better performance in attentive probing than in linear probing, outperforming larger models like InternVideo2 not only on motion-centric benchmarks but also on appearance-centric ones. This suggests that our model has a richer patch-wise representation capability compared to other models. For the last row of VideoPrism [42], the reported numbers are taken directly from the corresponding paper. It is important to note that we follow V-JEPA’s attentive probing schedule, which differs from the training schedule of VideoPrism.

##### 2.2.3 K-Nearest Neighbors

Evaluation Settings. In the cases of linear probing and attentive probing, the optimization process uses stochastic gradient descent, which can result in varying performances due to differences in initialization and batch order. Additionally, as the embedding vector dimension increases, the number of learnable parameters also increases, potentially making the evaluation process unfair. Therefore, it is necessary to compare the quality of embedding vectors in a parameter-free manner. To this end, we adopt the K-Nearest Neighbors (KNN) classification task, one of the simplest and most representative non-parametric methods. We obtain two versions of embeddings through the methods introduced in Section 2.1: Uniform Embedding (‘Uniform’ in Table 6) and Multi-Clip Embedding. For Multi-Clip Embedding, we fix the length of each clip to 2 seconds. Unlike Uniform Embedding, Multi-Clip Embedding can yield more than one vector per video. In this case, there are two evaluation methods. One method is to average all the embeddings to create a single video embedding (‘Video’ in Table 6). The other method, similar to the multi-view classification introduced in

Section 2.2, sums the votes received by each clip to determine the final class (‘Clip’ in Table 6). We conduct three types of evaluations in total. The results are presented in Table 6.

Results and Discussion. Similar to the linear probing and attentive probing results, our model demonstrates generally good performance on both K400 and SSv2 compared to the models of the same scale. However, unlike the attentive probing results, our model performs worse than InternVideo2 on K400 and SSv2. This indicates that there is room for improvement in utilizing embeddings extracted from our model in a non-parametric manner. An interesting point observed in Table 6 is the difference between uniform sampling and clip sampling. Since K400 and SSv2 mostly consist of videos longer than 2 seconds, multiple clips are obtained from most videos. As a result, the ‘Clip’ performance, which utilizes each clip independently, shows the highest accuracy. The tendencies differ in the ‘Uniform’ and ‘Video’ columns. K400 videos are generally longer (around 10 seconds) than SSv2 videos, making them more vulnerable to uniform sampling (longer intervals between frames). However, as K400 videos are more appearance-centric, the information loss caused by averaging clip embeddings is minimal. In contrast, SSv2 videos are relatively shorter, resulting in a smaller performance drop due to uniform sampling. Since SSv2 contains many temporally dynamic videos, simply averaging clip embeddings does not boost and may even hurt performance compared to uniform sampling. This experiment highlights the need for further research and consideration on representing embeddings for videos longer than a single clip length, and even for videos lasting several hours.

##### 2.2.4 Pretraining with Something-Something-v2 Dataset.

Evaluation Settings. Unlike V-JEPA, we exclude SSv2 from our pretraining dataset. Because SSv2 contains classes such as ‘Moving something up’ and ‘Moving something down’ that cannot be distinguished by appearance alone from a single frame, we consider it better to exclude SSv2 from pretraining to precisely measure the motion understanding capability of the models. However, for a fair comparison with V-JEPA and to understand the impact of including SSv2 in the pretraining dataset, we also train an additional model that includes SSv2 in the pretraining phase. The results of our model with SSv2 are presented at the bottom of Table 3, Table 5, and Table 6.

Results and Discussion. Including SSv2 in the pretraining stage results in improved performance on SSv2 across linear probing, attentive probing, and KNN evaluations. Additionally, performance generally increases on other benchmarks as well. This gain can be attributed to the increased data scale used during pretraining. Notably, in attentive probing, our model trained with SSv2 outperforms V-JEPA with the same architecture, achieving the best performance among all competitors. Despite these improvements, the KNN performance is still not as strong as that of V-JEPA and InternVideo2. This indicates that our model has a relatively weaker video-level representation compared to its patch-level representation. Addressing this imbalance should be one of the directions for future enhancements.

##### 2.3 ImageNet Classification

To validate whether the foundation models trained on the video dataset can operate as vision models beyond just video models, we adopt the ImageNet classification task. The last columns of Table 3 and Table 5 show the results. In general, the models that peform well on appearance-centric action recognition tasks (e.g., K400) also exhibit strong performance on the ImageNet benchmark. However, this tendency changes in some respects. For example, although InternVideo2 significantly outperforms our model on ImageNet, the gap in the video action recognition task significantly decreases (or even reverses). This observation allows us to verify that the proposed method 1) relatively lacks the ability to process static images, and 2) has the capability to utilize motion information for understanding videos in addition to appearance. To advance beyond being just a video model, we need to find a better way to improve the model’s ability to understand single images.

##### 2.4 Temporal Action Localization

Temporal Action Localization (TAL) is a pivotal task for accurately identifying and understanding human activities within long and intricate videos. This task can be extended to various applications in domains such as autonomous driving, sports analysis, and content-based video retrieval. TAL aims to analyze an untrimmed video and precisely identify the temporal intervals and corresponding class

##### Table 7: Hyperparameters for TAL and TAS evaluation.

##### Table 8: Hyperparameters for STAL evaluation.

|Hyperparameter|TAL THUMOS14 ActivityNet-v1.3|TAS 50Salads GTEA Breakfast|
|---|---|---|
| | | |

Hyperparameters AVA v2.2 Decoder Architecture # of layers 6 hidden size 256 MLP dimension 2048 box head # layers 3 dropout rate 0.1 drop path 0.1 Training optimizer AdamW optimizer momemtum β1 = 0.9, β2 = 0.999 weight_decay 1e-5 learning rate 1e-4 batch_size 256 warmup epochs 10 epochs 50

Backbone Setting video FPS 30 30 30 15 15 embedding type Clip Embedding Clip Embedding clip length T (secs) 0.5 0.5 1.0 0.5 0.5 temporal stride Ts (secs) 0.125 0.25 0.25 0.125 0.125 feature post-processing crop resize - - feature (max) length 2304 192 variable Detector Setting batch_size 2 16 1 epochs 45 10 120 warm-up epochs 2 - - lr 0.001 1e-4 5e-4 5e-4 1e-4 lr_decay cosine annealing reduce on plateau weight_decay 0.05 1e-4

##### Table 9: Temporal action localization evaluation on THUMOS14.

|Method|Arch.<br><br>|Self-contained 0.3 0.5 0.7 Avg.| |w/ External Classifier 0.3 0.5 0.7 Avg.| |
|---|---|---|---|---|---|
| | | | | | |
|UMTs2 TWLV-I<br><br>[Figure 23]<br><br>|ViT-B ViT-B<br><br>|54.60 42.18 22.17 71.01 57.06 33.25<br><br>|40.34 54.83<br><br>|56.39 43.23 23.27 66.09 54.14 33.16<br><br>|41.66 52.05<br><br>|
|UMTs2 V-JEPA<br><br>[Figure 24]<br><br>TWLV-I|ViT-L ViT-L ViT-L<br><br>|64.17 51.79 29.61 62.89 52.31 31.43 73.61 61.21 38.40<br><br>|49.43 49.72 58.75<br><br>|61.53 49.22 29.32 66.54 54.50 33.60 66.83 55.54 35.88<br><br>|47.31 52.11 53.63<br><br>|
|DFN V-JEPA InternVideo2s2<br><br>|ViT-H ViT-H ViT-g<br><br>|64.54 50.79 26.59 62.98 52.48 32.19 81.09 69.94 44.40<br><br>|48.17 49.91 66.21<br><br>|58.11 44.44 24.40 65.31 53.79 32.94 68.51 58.98 37.60|43.21 51.54 55.81<br><br>|

labels of distinct actions occurring within the video. We expect that this task can evaluate the models from two perspectives: temporal sensitivity and instance discrimination. Temporal sensitivity requires the model to identify if an action of interest occurs at the current time step. Instance discrimination, on the other hand, requires the ability to distinguish or group frame-wise segments into a complete action instance. Although the task is designed to be motion-centric, we find that the two aspects can be achieved with both appearance and motion capabilities.

Evaluation Settings. For the TAL task, we use the ActivityNet-v1.3 [11] and THUMOS14 [20] datasets, and we adopt ActionFormer [41] as the detection head. We report the results using two evaluation methods indicated as ‘Self-contained’ and ‘w/ External Classifier’ in Table 10 and 9. ‘Self-contained’ means that the task requires the model to conduct both classification and regression without any external classifier. ‘w/ External Classifier’ indicates that the model performs binary classification while an external classifier predicts the action class. We adopt the external classifier used in [5]. We evaluate in both ways using various metrics to comprehensively understand VFMs. We extract temporal features following the Multi-Clip Embedding method described in Section 2.1. We divide each video into clips of T seconds with a stride of Ts, where Ts < T allows overlap for dense extraction. We then uniformly sample N frames within each clip, with N depending on the encoder configuration. Detailed hyperparameters for evaluation are provided in Table 7. Unless otherwise noted, we follow the configuration of ActionFormer.

Results and Discussion. The first point to note is the difference in performance tendencies between ActivityNet-v1.3 and THUMOS14. In ActivityNet-v1.3, the model’s performance with an external

Table 11: Spatio-temporal action localization on AVA v2.2.

Table 10: Temporal action localization on ActivityNet-v1.3. † indicates results of the models trained including ActivityNet-v1.3.

|Method<br><br>|Arch.|fAP@0.5|
|---|---|---|
|UMTs2 TWLV-I<br><br>[Figure 25]|ViT-B ViT-B<br><br>|21.88 20.50<br><br>|
|UMTs2 V-JEPA<br><br>[Figure 26]<br><br>TWLV-I|ViT-L ViT-L ViT-L<br><br>|26.38 22.57 27.39<br><br>|
|DFN V-JEPA InternVideo2s2|ViT-H ViT-H ViT-g<br><br>|23.41 22.59 28.68|

|Method|Arch.|Self-contained 0.5 0.75 0.95 Avg.| |w/ External Classifier 0.5 0.75 0.95 Avg.| |
|---|---|---|---|---|---|
| | | | | | |
|UMTs2 TWLV-I<br><br>[Figure 27]<br><br>|ViT-B ViT-B<br><br>|45.82 30.32 6.23 49.14 33.06 6.46<br><br>|29.86 32.34<br><br>|58.12 38.23 7.55 58.60 39.30 8.14<br><br>|37.73 38.56<br><br>|
|UMTs2 V-JEPA<br><br>[Figure 28]<br><br>TWLV-I|ViT-L ViT-L ViT-L<br><br>|52.06 34.62 7.24 42.56 28.65 6.07 52.96 36.07 7.88<br><br>|34.07 28.22 34.98<br><br>|58.98 39.40 8.09 57.88 38.77 7.51<br><br>59.56 40.47 9.06<br><br><br>|38.70 37.99 39.49<br><br>|
|DFN V-JEPA InternVideo2s2<br><br>|ViT-H ViT-H ViT-g|54.22 35.79 7.25 39.39 26.76 5.60 58.30† 38.85† 8.90†<br><br>|35.33 26.18<br><br>38.35†|58.62 39.34 7.61 58.49 39.33 8.23<br><br>61.36† 41.89† 9.01†<br><br>|38.36 38.46 40.82†|

classifier is higher than that of the self-contained evaluation, whereas in THUMOS14, the opposite trend is observed. This can be attributed to two main factors: the number of classes and the number of training samples. ActivityNet-v1.3 has 200 classes, which is ten times more than THUMOS14 (20 classes). Therefore, requiring a single model to perform both action boundary regression and classification simultaneously demands more representational power than the model’s capacity. On the other hand, THUMOS14 has a relatively small training dataset (refer to Table 2). As a result, the classification objective provides extra supervision that can compensate for the lack of training samples. V-JEPA uniquely exhibits a different trend from the other models, showing lower self-contained evaluation performance in both THUMOS14 and ActivityNet-v1.3. This can be interpreted as VJEPA’s limited ability to embed appearance information. In fact, the performance drop in ActivityNetv1.3 is larger than for the other models. Notably, for V-JEPA, even though the performance with the external classifier improves with the ViT-H scale model, the self-contained evaluation performance decreases. This indicates that V-JEPA’s motion understanding ability improves, but its appearance understanding ability does not as the model scale increases. This observation aligns with what was noted in Section 2.2. TWLV-I exhibits the best performance across all evaluation methods and benchmarks compared to models of the same scale. Remarkably, TWLV-I at the ViT-L scale even outperforms the larger scale models DFN and V-JEPA at the ViT-H scale. This demonstrates that our model possesses robust and generalizable capabilities as a backbone for temporal localization. Moreover, our model shows high performance at strict IoU thresholds, such as 0.95 in ActivityNet, demonstrating its strength in temporal sensitivity. Lastly, InternVideo2 performs best across all scales, especially in self-contained evaluation. This indicates that the TAL task requires a larger model scale than classification alone. Specifically, performing both classification and action boundary regression simultaneously demands more parameters. It is recommended to use a classifier to primarily evaluate motion understanding capability through the TAL task, while for a more comprehensive assessment, including appearance, it is better to use a self-contained approach.

##### 2.5 Spatio-Temporal Action Localization

Compared to the previous tasks where the input videos are trimmed or cropped, the spatio-temporal action recognition (STAL) task aims to evaluate the capability of the vision encoder to localize instances and recognize their actions [40]. This task requires the model to not only identify the action taking place but also accurately determine the spatial and temporal locations of the action within an untrimmed video.

Evaluation Settings. We evaluate the models on the classic STAL dataset AVA v2.2 [17]. The AVA v2.2 dataset consists of 430 movie clips, each 15 minutes long. Instead of annotating all frames, keyframes are annotated every second, resulting in 210,634 labeled frames in the training set and 57,371 in the validation set. There are 80 atomic actions labeled for every actor in the clip [17]. As a metric, we report the Frame Average Precision (fAP) at an IoU threshold of 0.5 using the latest v2.2 annotations [17]. In previous works [42, 38], external ROI headers such as Mask-RCNN [18] are used for extracting the bounding box of the instance in the patch-level representation. This process suggests that the ability to understand spatial information is limited by whether the patch token has a corresponding spatial feature, not the spatial localizing capacity itself. Therefore, we decide to adopt the end-to-end STAL framework [16], training the decoder while freezing the vision backbone. The detailed hyperparameters are shown in Table 8.

Results and Discussion. We report the evaluation results in Table 11. In the STAL task, UMT, our model, and InternVideo2 outperform DFN and V-JEPA. Even though DFN and V-JEPA show opposite trends in previous evaluations, they exhibit similar performance in this task. This is because the STAL task consists of two subtasks: localizing the instance and recognizing the action class. If a model localizes the instance incorrectly, it cannot perform the subsequent task of action recognition effectively. From this perspective, V-JEPA fails to localize the instance correctly, which hinders its ability to solve the recognition task. On the other hand, DFN can localize the instance due to its superior understanding of appearance. However, even if DFN successfully finds the instance, it fails to predict the action class due to an insufficient understanding of temporal information. Unlike DFN and V-JEPA, models such as UMT, ours, and InternVideo2 are validated to understand both appearance and motion reasonably well, allowing them to solve the problem at a similar level. From these results, we can verify that 1) the proposed model has the capability to understand spatio-temporal information, and 2) the STAL task in an end-to-end manner can be a proper measure to evaluate the ability to understand video comprehensively.

##### Table 12: Temporal action segmentation evaluation on three benchmarks.

|Method|Arch.<br><br>|50Salads mF1 Edit Acc.<br><br>|GTEA mF1 Edit Acc.|Breakfast mF1 Edit Acc.|
|---|---|---|---|---|
| | | | | |
|UMTs2 TWLV-I<br><br>[Figure 29]<br><br>|ViT-B ViT-B<br><br>|70.66 70.79 77.71 80.69 76.93 85.83<br><br>|80.75 83.22 75.92 88.26 87.70 82.94<br><br>|31.39 43.75 32.39 52.18 62.77 57.86<br><br>|
|UMTs2 V-JEPA<br><br>[Figure 30]<br><br>TWLV-I<br><br>|ViT-L ViT-L ViT-L<br><br>|71.20 69.23 76.60 60.85 61.17 70.30 80.60 77.19 84.75<br><br>|85.91 87.62 79.69<br><br>87.42 86.84 81.63<br><br>88.43 90.19 82.29<br><br><br>|45.25 58.11 49.04<br><br>46.96 58.63 52.66 50.66 62.80 54.17<br><br><br>|
|DFN V-JEPA InternVideo2s2<br><br>|ViT-H ViT-H ViT-g<br><br>|73.22 71.56 80.13 54.73 56.51 63.25 82.00 79.04 86.74|88.06 89.66 80.16 84.87 85.49 80.82 92.96 93.08 84.76<br><br>|41.73 55.47 46.40 45.55 57.07 51.92 57.38 66.98 62.38|

| |
|---|

| |
|---|

[Figure 31]

[Figure 32]

GT

GT

[Figure 33]

[Figure 34]

| |
|---|

| |
|---|

UMT

UMT

[Figure 35]

[Figure 36]

| |
|---|

| |
|---|

V-JEPA

V-JEPA

[Figure 37]

[Figure 38]

| |
|---|

| |
|---|

Ours

Ours

- Figure 5: Visualization of temporal action segmentation. The figure shows the qualitative results of two test samples from GTEA. Each action class corresponds to the different color and the x-axis represents time.

- 2.6 Temporal Action Segmentation

Temporal Action Segmentation (TAS) is an essential task for the comprehension and analysis of human activities within complex, extended videos, encompassing a diverse array of applications such

- as video surveillance, video summarization, and skill assessment. The TAS task aims to process an untrimmed video input and generate an action sequence of the class label for each frame.

Evaluation Settings. In this paper, we utilize three challenging benchmarks for TAS evaluation: 50Salads [31], GTEA [13], and Breakfast [22]. We employ ASFormer [39] as the detection head. The average F1 scores at thresholds of 10, 25, and 50, referred to as ‘mF1’, are reported. Similar to TAL, we extract features according to the Multi-Clip Embedding method described in Section 2.1 and aggregate them using spatial pooling to retain only the temporal axis. Further implementation details are provided in Table 7. Unless otherwise specified, we adhere to the configuration of ASFormer.

Results and Discussion. Table 12 presents the TAS evaluation results on three benchmarks. The appearance-centric models, such as UMT and DFN, perform better than the motion-centric model V-JEPA on 50Salads, but they exhibit worse performance on the GTEA and Breakfast datasets. In contrast, our model surpasses the baselines across all three datasets, demonstrating strong capabilities in both appearance and motion. Additionally, our model excels in Edit and mF1 scores, indicating that the sequence of predictions closely aligns with the ground-truth instances. Regarding camera views, 50Salads and GTEA primarily show human hands in top-down and ego-centric perspectives, while Breakfast features a wider view that mostly includes the human body. In this context, our model outperforms baselines across various views and performs comparably to InternVideo2 in top-down and egocentric views on the 50Salads and GTEA datasets. Figure 5 illustrates the visualization of two test samples from GTEA for large-scale models: UMT, V-JEPA, and ours. As depicted, our model produces precise segmentation, while UMT and V-JEPA generate more false positives in the sequence.

- 2.7 Embedding Visualization

- 2.7.1 T-SNE Visualization on Benchmark Datasets

Figures 6 and 7 visualize the embedding space of the K400 and SSv2 validation sets using t-SNE [34], with different colors representing different classes. In Figure 6, our model, along with UMT and InternVideo2, demonstrates clusters with clear decision boundaries for most classes. In contrast,

(a) TWLV-I (b) UMT (c) InternVideo2 (d) V-JEPA

- Figure 6: t-SNE visualization of Kinetics-400 validation set. TWLV-I, UMT, and InternVideo2 cluster the samples into several groups while V-JEPA does not provide a clear distinction between groups. We note that all the pretraining dataset of video models includes Kinetics-400 (K400).

[Figure 43]

(a) TWLV-I (b) UMT (c) InternVideo2 (d) V-JEPA†

[Figure 44]

[Figure 45]

[Figure 46]

- Figure 7: t-SNE visualization of Something-Something-v2 validation set. It shows uniformly sampled half of the points. All the models fail to provide clear groups because of their low absolute accuracy on SSv2. † denotes that the pretraining dataset for the model includes the SomethingSomething-v2 dataset.

V-JEPA fails to cluster the samples into distinct groups, which aligns with the analysis in Section 2.2. On the other hand, the results in Figure 7 indicate that none of the models, including V-JEPA, effectively cluster the classes. While Section 2.2 shows that our model and V-JEPA perform relatively better, the absolute performance of classification on the SSv2 dataset is significantly lower compared to the K400 dataset. This discrepancy suggests that none of the models achieve notable clustering results during visualization, highlighting the need for further research to improve performance on motion-centric benchmarks.

##### 2.7.2 Directional Motion Distinguishability

Since the visualization on the motion-centric benchmark dataset (SSv2; Figure 7) fails to properly provide information for understanding the models’ capabilities, we utilize specific classes within SSv2. Especially, classes like ‘Moving something up’, which require understanding the direction of an object’s motion. One way to assess whether a model accurately captures this directional motion is to see if it can distinguish between classes with opposite directions, such as ‘Moving something up’ and ‘Moving something down’. However, there is a limitation in that the appearance of videos in each class differs. To control the difference in appearance, we visualize the embeddings of the ‘Moving something up’ class and its reversed (temporally flipped) version. We then use Linear Discriminant Analysis (LDA) to visualize whether these two embeddings can be distinguished. This allows us to determine whether a model can encode the direction of motion in the embedding vector without relying only on appearance. Figure 8 presents the analysis for the ‘Moving something up’ class, and Figure 9 shows the analysis for the ‘Moving something down’ class. The results indicate that V-JEPA, trained on the SSv2 dataset, shows the highest distinguishability between forward and reverse videos. Our model also demonstrates a good separation between the two. In contrast, UMT and InternVideo2 provide non-separable embeddings, suggesting that while these models have strong appearance understanding capabilities, they fail to adequately distinguish between the forward and reverse versions of the same video. This conclusion indicates that UMT and InternVideo2 do not effectively capture the directional motion information.

(a) TWLV-I (b) UMT (c) InternVideo2 (d) V-JEPA†

- Figure 8: LDA visualization of Something-Something-v2 ‘Moving something up’ class videos and their reverse. TWLV-I and V-JEPA distinguish them well, but the appearance-centric models (UMT and InternVideo2) do not recognize the difference between the forward and reverse video. † denotes that the pretraining dataset for the model includes the Something-Something-v2 dataset.

[Figure 51]

(a) TWLV-I (b) UMT (c) InternVideo2 (d) V-JEPA†

[Figure 52]

[Figure 53]

[Figure 54]

- Figure 9: LDA visualization of Something-Something-v2 ‘Moving something down’ class videos and their reverse. TWLV-I and V-JEPA distinguish them well, but the appearance-centric models (UMT and InternVideo2) do not recognize the difference between the forward and reverse video. † denotes that the pretraining dataset for the model includes the Something-Something-v2 dataset.

#### 3 Future Directions

Scaling Up. In this technical report, we focus on two scales of models: ViT-B and ViT-L. Our model, at the size of ViT-L, already demonstrates performance comparable to larger scales such as ViT-H or ViT-g. Therefore, further scaling of the model has the potential to enhance performance even more. Along with increasing model size, another future direction involves using large-scale datasets collected in-house to develop more general and robust models.

Image Embedding. As mentioned in Section 2.3, our model currently exhibits limited image embedding capability. Since an image can be considered a single-frame video, and video understanding is an extension of image understanding, improving our model’s image embedding capacity is crucial. Enhancing this aspect will enable the model to transition from a video foundation model to a more versatile visual understanding model, thereby broadening its potential applications.

Expanding Modality In addition to the tasks covered in this technical report, there are various other video-related tasks, particularly, multimodal tasks such as retrieval and captioning. To align the model with the text domain while maintaining all its unimodal capabilities, further in-depth research is necessary. Additionally, improvements are needed to develop a more effective vision encoder for video-language models. Proper evaluation methodologies for these tasks must also be proposed to ensure comprehensive assessment and validation.

#### 4 Conclusion

In this technical report, we highlight appearance and motion as the two most critical elements of video understanding. Accordingly, we present methodologies for measuring each capability using various tasks and demonstrate through experiments that existing models have limitations in generally satisfying both capabilities. We then propose TWLV-I, a video foundation model that is robust in both motion and appearance. We expect our model and the embeddings obtained through it to be actively utilized and researched in various downstream tasks. Furthermore, we hope that the evaluation and analysis methods proposed in this technical report will be actively used in the video foundation model domain and will serve as a guiding direction for the video understanding field.

#### Authorship

This work was achieved through the combined efforts (equal contribution) of the core contributors, with significant support from the Twelve Labs ML Research and ML Data teams.

##### Core Contributors

Hyeongmin Lee, Research Scientist Jin-Young Kim, Research Scientist Kyungjune Baek, Research Scientist Jihwan Kim, Research Scientist Aiden Lee, CTO

##### Contributors 3

William (Hyojun) Go, Research Scientist Mars (Seongsu) Ha, Research Scientist Cooper (Seokjin) Han, Research Scientist Flynn (Jiho) Jang, Research Scientist Ray (Raehyuk) Jung, Research Scientist Leo (Daewoo) Kim, Research Scientist Daniel (GeunOh) Kim, ML Data Engineer Max (JongMok) Kim, Research Scientist Jeff (Jongseok) Kim, Research Scientist Jayden (Junwan) Kim, Research Intern Ian (Soonwoo) Kwon, Research Scientist Aaron (Jangwon) Lee, ML Data Intern Kyle (Seungjoon) Park, Research Scientist Calvin (Minjoon) Seo, Chief Scientist Jay Suh, ML Data Engineer Jay (Jaehyuk) Yi, Research Scientist

##### Acknowledgement

We thank Jae Lee (CEO) and the leadership team for the support in our research, and the product and go-to-market team at Twelve Labs for their inspiration and guidance. Special thanks to the engineering team for their essential support, and to Sangdoo Yun from Naver AI Lab for his valuable feedback.

3The author list sorted alphabetically.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2
- [2] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738, 2021. 4
- [3] Adrien Bardes, Quentin Garrido, Jean Ponce, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv:2404.08471, 2024. 3, 4, 5, 6, 7
- [4] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. 1
- [5] Guo Chen, Yifei Huang, Jilan Xu, Baoqi Pei, Zhe Chen, Zhiqi Li, Jiahao Wang, Kunchang Li, Tong Lu, and Limin Wang. Video mamba suite: State space model as a versatile alternative for video understanding. arXiv preprint arXiv:2403.09626, 2024. 9
- [6] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023. 2
- [7] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1– 113, 2023. 2
- [8] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European conference on computer vision (ECCV), pages 720–736, 2018. 5
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020. 3, 4
- [10] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. arXiv preprint arXiv:2401.08541, 2024. 6
- [11] Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 961–970, 2015. 9
- [12] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 4
- [13] Alireza Fathi, Xiaofeng Ren, and James M Rehg. Learning to recognize objects in egocentric activities. In CVPR 2011, pages 3281–3288. IEEE, 2011. 11
- [14] Christoph Feichtenhofer, Yanghao Li, Kaiming He, et al. Masked autoencoders as spatiotemporal learners. Advances in neural information processing systems, 35:35946–35958, 2022. 3
- [15] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842– 5850, 2017. 3, 5
- [16] Alexey A Gritsenko, Xuehan Xiong, Josip Djolonga, Mostafa Dehghani, Chen Sun, Mario Lucic, Cordelia Schmid, and Anurag Arnab. End-to-end spatio-temporal action localisation with video transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18373–18383, 2024. 10

- [17] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of spatio-temporally localized atomic visual actions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6047–6056, 2018. 10
- [18] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017. 10
- [19] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 2
- [20] Y.-G. Jiang, J. Liu, A. Roshan Zamir, G. Toderici, I. Laptev, M. Shah, and R. Sukthankar. THUMOS challenge: Action recognition with a large number of classes. http://crcv.ucf. edu/THUMOS14/, 2014. 9
- [21] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017. 3, 5
- [22] Hilde Kuehne, Ali Arslan, and Thomas Serre. The language of actions: Recovering the syntax and semantics of goal-directed human activities. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 780–787, 2014. 11
- [23] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, and Yu Qiao. Uniformerv2: Spatiotemporal learning by arming image vits with video uniformer, 2022. 4
- [24] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19948–19960, 2023. 3, 4
- [25] Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In Proceedings of the European Conference on Computer Vision (ECCV), pages 513–528, 2018. 3, 5
- [26] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. CLIP4Clip: An empirical study of clip for end to end video clip retrieval. arXiv preprint arXiv:2104.08860,

2021. 3

- [27] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019. 4
- [28] Mathew Monfort, Alex Andonian, Bolei Zhou, Kandan Ramakrishnan, Sarah Adel Bargal, Tom Yan, Lisa Brown, Quanfu Fan, Dan Gutfreund, Carl Vondrick, et al. Moments in time dataset: one million videos for event understanding. IEEE transactions on pattern analysis and machine intelligence, 42(2):502–508, 2019. 3, 5
- [29] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 3
- [31] Sebastian Stein and Stephen J McKenna. Combining embedded accelerometers with computer vision for recognizing food preparation activities. In Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing, pages 729–738, 2013. 11
- [32] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 3
- [33] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2

- [34] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008. 11
- [35] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14549–14560,

2023. 3

- [36] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, and Yu-Gang Jiang. Masked video distillation: Rethinking masked feature modeling for selfsupervised video representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6312–6322, 2023. 3
- [37] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024. 2, 3, 4
- [38] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 10
- [39] Fangqiu Yi, Hongyu Wen, and Tingting Jiang. Asformer: Transformer for action segmentation. arXiv preprint arXiv:2110.08568, 2021. 11
- [40] Liangzhe Yuan, Nitesh Bharadwaj Gundavarapu, Long Zhao, Hao Zhou, Yin Cui, Lu Jiang, Xuan Yang, Menglin Jia, Tobias Weyand, Luke Friedman, et al. Videoglue: Video general understanding evaluation of foundation models. arXiv preprint arXiv:2307.03166, 2023. 3, 5, 10
- [41] Chen-Lin Zhang, Jianxin Wu, and Yin Li. Actionformer: Localizing moments of actions with transformers. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IV, pages 492–510. Springer, 2022. 9
- [42] Long Zhao, Nitesh B Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, Jennifer J Sun, Luke Friedman, Rui Qian, Tobias Weyand, Yue Zhao, et al. Videoprism: A foundational visual encoder for video understanding. arXiv preprint arXiv:2402.13217, 2024. 7, 10

