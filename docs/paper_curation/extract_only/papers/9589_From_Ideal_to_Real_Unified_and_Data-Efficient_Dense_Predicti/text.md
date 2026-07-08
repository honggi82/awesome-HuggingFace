### From Ideal to Real: Unified and Data-Efficient Dense Prediction for Real-World Scenarios

Changliang Xia∗, Chengyou Jia∗, Zhuohang Dang, Minnan Luo†, Zhihui Li, and Xiaojun Chang .

## arXiv:2506.20279v2[cs.CV]30Sep2025

Abstract—Dense prediction tasks hold significant importance of computer vision, aiming to learn pixel-wise annotated labels for input images. Despite advances in this field, existing methods primarily focus on idealized conditions, exhibiting limited realworld generalization and struggling with the acute scarcity of real-world data in practical scenarios. To systematically study this problem, we first introduce DenseWorld, a benchmark spanning a broad set of 25 dense prediction tasks that correspond to urgent real-world applications, featuring unified evaluation across tasks. We then propose DenseDiT, which exploits generative models’ visual priors to perform diverse real-world dense prediction tasks through a unified strategy. DenseDiT combines a parameterreuse mechanism and two lightweight branches that adaptively integrate multi-scale context. This design enables DenseDiT to achieve efficient tuning with less than 0.1% additional parameters, activating the visual priors while effectively adapting to diverse real-world dense prediction tasks. Evaluations on DenseWorld reveal significant performance drops in existing general and specialized baselines, highlighting their limited real-world generalization. In contrast, DenseDiT achieves superior results using less than 0.01% training data of baselines, underscoring its practical value for real-world deployment.

Index Terms—Dense prediction, real-world, diffusion transformer, visual priors.

I. INTRODUCTION

# D

Ense prediction tasks [1], [2], such as segmentation and depth estimation, represent a class of fundamental

computer vision problems that aim to learn mappings from input images to pixel-wise annotated labels. These tasks are critical for numerous applications, such as medical imaging [3], autonomous driving [4], remote sensing [5], and others. While recent advances [6]–[8] have achieved strong performance through carefully designed strategies, they are mostly developed under idealized conditions [9], [10] featuring uniform lighting, minimal occlusion and easily accessible, relatively abundant data, resulting in limited generalization to practical dense prediction tasks [11], [12].

In contrast to idealized conditions, dense prediction in realworld scenarios offers more practical value. These scenarios

∗Equal Contribution. †Corresponding author: Minnan Luo.

Changliang Xia, Chengyou Jia, Zhuohang Dang and Minnan Luo, are with the School of Computer Science and Technology, Xi’an Jiaotong University, Xi’an 710049, China (e-mail: 202066@stu.xjtu.edu.cn; cp3jia@stu.xjtu.edu.cn; dangzhuohang@ stu.xjtu.edu.cn; minnluo@xjtu.edu.cn;).

Zhihui Li and Xiaojun Chang are with the School of Information Science and Technology, University of Science and Technology of China, Hefei 230026, China (e-mail: lizhihuics@ustc.edu.cn; xiaojun.chang@uts.edu.au.)

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

cover broad applications such as depth estimation in adverse weather to enhance autonomous driving safety, crack detection to enable proactive maintenance, and underwater image segmentation to advance deep-sea exploration. As illustrated in Fig. 1, real-world scenarios differ from idealized conditions in two aspects: inherent complexity and data scarcity. This motivates our core question: Can we develop a data-efficient dense prediction model that generalizes effectively across diverse real-world scenarios?

To this end, we first introduce DenseWorld, a benchmark to advance dense prediction in real-world scenarios. The core principle of DenseWorld is to cover a broad spectrum of realworld tasks that are both practical and challenging, while being difficult to collect and annotate at scale. We carefully curate 25 diverse dense prediction tasks, each corresponding to a distinct real-world scenario. These tasks span a wide range of applications, including ecology [13], healthcare [11], infrastructure [14], public safety [15], and industrial operations [16]. Unlike prior domain-specific benchmarks, DenseWorld establishes a unified evaluation protocol, enabling fair and holistic comparison across heterogeneous real-world tasks. Moreover, it intentionally reflects the inherent data scarcity of real-world applications, where few-shot dense prediction is the norm rather than the exception. In this sense, DenseWorld provides a unified and realistic platform to evaluate dense prediction methods under real-world complexity.

We further propose DenseDiT, a dense prediction framework built upon generative models. While prior works [17], [18] concatenate query and noisy tokens along the channel dimension, they inevitably modify the base model architecture, which compromises its pretrained visual priors. Moreover, such rigid concatenation restricts flexible interactions between query and noisy tokens. Instead, DenseDiT adopts a parameterreuse mechanism to process query tokens and noisy tokens independently within the intact latent space where the visual priors reside. This independence enables more expressive and flexible interactions via the Multi-Modal Attention (MMA) module [19] of the generative model. To further enhance generalization across diverse real-world scenarios, we introduce two lightweight branches: a prompt branch, which reuses the text encoder to inject task-specific semantic cues, and a demonstration branch, which reuses the latent encoder to align visual priors with complex scene distributions. Unlike approaches relying on full training [17], which are fundamentally prone to overfitting and hence impaired generalization given the inherent data scarcity of real-world scenarios, DenseDiT achieves robust performance and generalizes across diverse real-world dense prediction tasks with minimal adaptation.

##### Idealized Scenarios

##### Real-World Scenarios

[Figure 1]

[Figure 2]

(Well-lit with Minimal Occlusion)

(Challenging and Hard to Collect)

Outdoor depth estimation Depth Estimation in Foggy Scenes

Depth Estimation in Rainy Scenes

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Indoor depth estimation Crack Detection Urban Building Layout Analysis

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Outdoor image segmentation Nucleus Localization Marine Oil Spill Surveillance

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Indoor image segmentation Instant Fire Alert Underwater image segmentation

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

- Fig. 1. Comparison of idealized vs. real-world dense prediction. (Left) Idealized tasks under controlled conditions with uniform lighting, minimal occlusion, and abundant data. (Right) Real-world tasks exhibiting complex scenes, adverse conditions, and inherent data scarcity, presenting substantially greater challenges.

[7], [23]. However, most existing approaches are developed under idealized conditions, including bright and minimally occluded indoor scenarios [9] or outdoor scenarios with favorable weather [10], which are typically accompanied by easily obtainable and relatively abundant training data, as shown in the left part of Fig. 1. Recent few-shot dense prediction methods [8], [24] improve data efficiency via cross-image matching or prompt-based adaptation, but require extensive base-class training and are typically evaluated on controlled benchmark settings. This limits their practicality in diverse, complex, and data-scarce real-world scenarios. To mitigate these limitations, task-specific models [25], [26] have been proposed, but they lack generalizability across diverse dense prediction tasks. Corresponding datasets are also tailored to specific domains, such as OmniCrack30K [12] for crack segmentation and the Global-Scale [27] for road extraction. In this work, we tackle these limitations with a unified and dataefficient framework tailored for real-world dense prediction.

We conduct comprehensive experiments on the proposed DenseWorld benchmark and common benchmarks to evaluate various approaches. Extensive quantitative and qualitative results demonstrate that DenseDiT significantly outperforms both general-purpose and task-specific models across all tasks, showcasing strong capabilities for real-world dense prediction. Compared to existing generative-based dense prediction methods [17], [20], DenseDiT achieves strong performance in more challenging real-world scenarios under limited supervision, highlighting its practical alignment with the data-scarce nature of many real-world applications. Moreover, the effectiveness of DenseDiT’s branches reveals the strong flexible contextual modeling capacity of recent generative models, highlighting promising directions for future research. In summary, our contributions are as follows:

- • We introduce DenseWorld, a benchmark for unified evaluation of dense prediction across real-world scenarios. It covers 25 practical applications and faithfully reflects the inherent data scarcity common in real-world settings.
- • We propose DenseDiT, a unified framework that leverages a parameter-reuse architecture and branch-enhanced adaptation for data-efficient dense prediction across diverse real-world scenarios.
- • Extensive experiments on the proposed DenseWorld benchmark and common benchmarks demonstrate the superiority of DenseDiT over general-purpose and taskspecific baselines, revealing their limited generalization to real-world scenarios and highlighting DenseDiT’s potential for dense prediction in the wild.

B. Diffusion Models For Dense Prediction

Pretrained generative models have been shown to possess rich visual priors [28]. These priors have been leveraged for tasks like representation learning [29]. By redefining dense prediction as an image-to-image task, recent work [17], [30] has explored diffusion models [31], [32] for dense prediction, such as depth estimation and segmentation. In depth estimation, DepthGen [18] performs monocular depth estimation using diffusion models, addressing noisy depth data through step-unrolled denoising and infilling strategies. Marigold [17] fine-tunes a diffusion model on clean synthetic data to elegantly exploit its visual priors for depth estimation. In segmentation, RefLDM-Seg [33] formulates in-context segmentation as latent mask generation. However, these methods typically require modifying the backbone architecture, which

II. RELATED WORK A. Dense Prediction

Dense prediction tasks [21], [22] such as segmentation and depth estimation have made substantial progress through carefully designed architectures and training pipelines [6],

[Figure 27]

[Figure 28]

Construction pipeline

Sunrise

Est.

Overcast

Det.

[Figure 29]

[Figure 30]

[Figure 31]

Data Sources DenseWorld Benchmark

Depth

Search Filtering Pruning

[Figure 32]

Fog Depth Est.

[Figure 33]

25 Tasks

[Figure 34]

Depth Est.

[Figure 35]

Crack

Det.

#### ...

Depth Est.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Diversity and utility

Sunset

Auto. Pothole

Rain Depth Est.

[Figure 40]

[Figure 41]

Road

[Figure 42]

[Figure 43]

15 Train samples per task

Mon.

Pedestrian

Close-Range Seg.

[Figure 44]

Visual examples

Adverse Env.

Precise Road Mod.

[Figure 45]

[Figure 46]

Adverse Environment Perception Smart City Inspection

Smart City.

Food Cell Inspect.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Urban Layout Anal.

[Figure 51]

Safety Ctrl.

Fire Alert Sys.

Cardiac Screen.

[Figure 52]

[Figure 53]

Intelligent Medical Assistance Ecological Monitoring

Medical Assist.

Cigarette Det.

Retinal Vessel Anal.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Mask-Wear Mon.

Ecological Mon.

Spine

Map.

Morph. Ases.

Distrib.

Oocyte Det.

Track.

[Figure 58]

Nucleus Loc.

Water Body Map.

aC

ed

[Figure 59]

[Figure 60]

.otuA eer T g.a T

egalfuom

W

Spil

Safety Production Control

Oil

.te D

[Figure 61]

Unified Evaluation Support!

- Fig. 2. Overview of the DenseWorld benchmark. Upper left: the construction pipeline. Center left: examples of representative tasks across five real-world categories. Lower left: unified evaluation. Right: full taxonomy of 25 dense prediction tasks, each aligned with a practical application scenario.

compromises the visual priors. Moreover, they are mostly trained under idealized conditions, lacking generalization to real-world scenarios. In contrast, our work enables diffusion models to generalize across diverse real-world dense prediction tasks with minimal adaptation effort.

III. DENSEWORLD BENCHMARK

We introduce the DenseWorld benchmark, comprising diverse and practically valuable real-world dense prediction tasks that suffer from limited data, as shown in Fig. 2.

- A. Benchmark Construction

An overview of the benchmark construction pipeline is shown in the upper left of Fig. 2, comprising three stages: data collection, task filtering, and data pruning. To build DenseWorld, we collect data from various sources, including open platforms (e.g., Kaggle [34]), academic repositories (e.g., Papers With Code [35]), and domain-specific portals (e.g., NASA EarthData [36]). Unlike previous benchmarks that emphasize scale, we prioritize the diversity and utility of realworld scenarios. Notably, the benchmark intentionally reflects the inherent data scarcity common in practical applications, such as medical imaging where expert annotations are costly, or industrial monitoring where data acquisition is inherently difficult. This makes the few-shot setting not an artificial constraint, but a realistic simulation of deployment conditions. Built upon a cleaned collection of source data, DenseWorld covers 25 dense prediction tasks across five categories, each aligned with a critical real-world application.

- B. Unified Evaluation Metric

Real-world dense prediction includes diverse tasks such as depth estimation in autonomous driving and crack segmentation in infrastructure inspection, each traditionally evaluated using domain-specific metrics. This heterogeneity makes

cross-domain comparison and overall performance assessment challenging. To address this, DenseWorld introduces a unified evaluation framework centered around the D/S-Score, a composite metric that facilitates fair and consistent comparison across tasks. For regression tasks (e.g., depth estimation), the D-Score integrates five standard metrics (AbsRel, RMSE, SqRel, RMSE-log, Threshold Accuracy) following established protocols [37]. For classification tasks (e.g., segmentation), the S-Score combines IoU, PA, and Dice [38]. Normalization is essential for meaningful aggregation across these heterogeneous metrics. They differ significantly in scale and optimal direction. Without normalization, metrics with larger numerical ranges would dominate the aggregate, while opposing optimization directions would render cross-task comparison invalid. We therefore apply min-max normalization to project all metrics to a common [0,1] range with a consistent higheris-better convention:

 

m − mmin mmax − mmin

, for m ↑ mmax − m mmax − mmin

Normalized(m) =

(1)



, for m ↓

where m ↑ denotes that a higher value of metric m is better, and m ↓ denotes that a lower value is better.

The final D-Score and S-Score are computed as the arithmetic mean of their respective normalized metrics. This approach provides a standardized measure for evaluating model performance across diverse real-world dense prediction tasks.

C. Benchmark Analysis

DenseWorld offers notable advantages over existing benchmarks in real-world diversity and utility, as shown in Table I. Existing datasets such as Taskonomy [21], COCO [39], NYUv2 [9], KITTI [10], OmniCrack30k [12], and GlobalScale [27] are limited in real-world task coverage or collected

under ideal conditions. In contrast, DenseWorld covers 25 practical tasks, each aligned with a real-world application. Moreover, the benchmark intentionally reflects the inherent data scarcity of practical scenarios, where annotated data is naturally limited. The taxonomy is on the right of Fig. 2, with more details (e.g., test set size) in the supplementary material.

[Figure 62]

[Figure 63]

Parameter Reuse

[Figure 64]

task1

[Figure 65]

task2

task3

VAE

task4...

Shared DiT Block

TABLE I COMPARISON OF DENSE PREDICTION BENCHMARKS.

[Figure 66]

Practical Tasks

Training Samples

DataEfficiency?

Real-World Utility?

Unified Eval?

Dataset

Prompt Branch

- A [output format] of [real-world scene]

Noise

[Figure 67]

[Figure 68]

×n

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

LoRA

[Figure 73]

Query

Demo Branch

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

DAI

VAE

[Figure 82]

Learnable Frozen

[Figure 83]

Branch controller

Legends

Fig. 3. Overview of the DenseDiT architecture. The framework processes a query image through a parameter-reused VAE encoder, while lightweight prompt and demo branches provide contextual cues. These elements interact within the generative backbone, requiring only LoRA-based fine-tuning to achieve dense prediction in complex, data-scarce real-world scenarios.

generation, such as one-step diffusion [45], and is expected to gain further efficiency as these techniques mature. The inputs are jointly processed through MMA [19] modules for dense prediction. Details are provided in Section IV-C.

- B. Standardizing Task Representation

Diverse data formats in dense prediction pose challenges to unified processing. Inspired by [30], we standardize task representations into RGB format, as most vision models [31], [46], including DiT, are trained on RGB data, which is critical for leveraging visual priors. First, we align channel dimensions by duplicating single-channel data to ensure compatibility with RGB-based encoders. Then, we normalize pixel values to address scale variations across tasks. Specifically, we apply:

rnorm =

r − rmin

rmax − rmin − 0.5 × 2 (3) where rmax and rmin denote the maximum and minimum.

- C. DenseDiT

Taskonomy 1 4M COCO 2 118K NYUv2 1 795 KITTI 2 86K Omnicrack30k 1 22K Global-Scale 1 2K DenseWorld 25 15

IV. METHOD

Our goal is to reformulate generative models into dataefficient dense predictors capable of generalizing across diverse real-world scenarios. Section IV-A introduces the generative model with strong visual priors and our model overview.

- Section IV-B formalizes the dense prediction tasks. Finally,
- Section IV-C details our proposed method, DenseDiT.

A. Preliminary and Model Overview

Our approach builds upon the DiT architecture [40], adopted by recent state-of-the-art generative models [41], [42]. DiT uses a transformer-based denoising network to refine noisy latent iteratively. Each block includes a Multi-Modal Attention (MMA) module [19], enabling effective interaction between noisy latent and conditioning inputs.

DiT learns a velocity field [43] to map noisy samples to clean ones, optimized by minimizing the discrepancy between predicted and ground-truth velocities:

0,t,c ∥vθ(zt,t,c) − u(zt)∥22 (2)

Loss = Ez

Building upon the DiT architecture, we aim to tackle diverse real-world dense prediction tasks amid inherent data scarcity. As illustrated in Fig. 3, we convert the generative backbone into a real-world dense predictor with minimal architectural changes, preserving its pretrained strengths while enhancing its generalization capability across diverse real-world tasks.

where c is the conditioning information, z0 is the clean latent, and zt is the noisy latent at timestep t.

Fig. 3 illustrates our DenseDiT, a unified and data efficient framework for dense prediction. The unification in our approach reflects three key characteristics: unified evaluation across diverse tasks through D/S-Score (Section III-B), unified architecture that handles various real-world dense predictions without task-specific modules, and unified training that applies the same objective and optimization to all tasks, contrasting with prior methods requiring customized designs per task [44]. DenseDiT takes a query image, a textual prompt, and an optional demonstration pair as inputs. A parameterreused VAE encoder projects the query images into the latent space. Two lightweight auxiliary branches, the prompt and the demonstration branch, provide semantic and visual contextual cues, activated by a task attribute called DAI. This design is inherently compatible with future advances in efficient

1) Parameter-Reuse Mechanism: Visual priors in DiT are key to DenseDiT’s effectiveness. To preserve them, we perform all the processing in the latent space, where the visual priors are encoded. Specifically, DenseDiT reuses the VAE to encode the query image into the same latent space as noisy latent, ensuring compatibility with DiT blocks. DenseDiT then employs shared DiT blocks for processing. This parameterreuse mechanism ensures architectural simplicity and low parameter complexity, while maintaining strong compatibility with the pretrained generative backbone. Adaptation to the new input (the query image) is achieved solely through lightweight LoRA [47] modules, adding only 0.1% parameters.

TABLE II COMPARISON WITH GENERAL-PURPOSE MODELS: STRONG ZERO-SHOT BASELINES FOR DENSE PREDICTION

Training Samples

Adverse Env. (D-Score)

Smart City. (S-Score)

Medical Assist. (S-Score)

Ecological Mon. (S-Score)

Safety Ctrl. (S-Score) Avg.

Model

SAM 1.1B - 0.401 0.596 0.479 0.572 - / 0.512 CLIPSeg 345K - 0.562 0.484 0.498 0.499 - / 0.511 Grounded-SAM 1.1B - 0.423 0.447 0.403 0.478 - / 0.438 Marigold 74K 0.901 - - - - 0.901 / EcoDepth 48K 0.044 - - - - 0.044 / SQLdepth 24K 0.352 - - - - 0.352 / ZeoDepth+PF 19K 0.577 - - - - 0.577 / Depth-Anything+PF 19K 0.771 - - - - 0.771 / -

DenseDiT 15 0.944 0.734 0.825 0.749 0.669 0.944 / 0.744 w/ Mixed Training 15×25 0.904 0.624 0.683 0.606 0.624 0.904 / 0.634

2) Branch-Enhanced Task Understanding: The tasks in DenseWorld span diverse real-world scenarios, with inputs ranging from natural to non-natural images (e.g., medical images). Although DiT is pretrained on large-scale datasets, adapting its priors to specific tasks under limited supervision remains a challenge. To enhance task understanding with minimal cost, DenseDiT introduces two lightweight branches.

The first is the prompt branch, built directly on DiT’s text encoder. This is a natural design, as DiT is pretrained on image-text pairs, and its text encoder can be reused without further adaptation. For each task, we provide a prompt using the template: ”A [output format] of [real-world scene]”, injecting contextual cues for better task alignment.

The second is the demonstration branch, designed to mitigate domain gaps for more complex tasks. It employs another parameter-reused VAE to encode a demonstration pair [IQ;IT], where IQ is a query and IT is its dense label. The resulting latent token interacts with the query latent and prompt token via MMA [19]. To balance performance and computational efficiency, the demonstration branch is selectively activated only when necessary. A binary Distribution Alignment Indicator (DAI) controls demonstration branch activation, indicating whether a task’s data distribution aligns with DiT pretraining. This demonstration branch is enabled when DAI = No. DAI labels are automatically assigned using GPT-based classification [48]; specifically, the method evaluates whether a task’s imagery aligns with DiT’s pretraining distribution or represents a significant domain shift. To ensure robustness, we perform five independent queries and take the majority vote as the final DAI label. The prompt for constructing the DAI label is as follows.

Prompt for obtaining DAI labels

You are given a description and a pair of demonstration images of a dense prediction task. Please determine whether the typical images involved in this task are well-aligned with the training distribution of DiT (e.g., natural, diverse internet images from LAION-5B), or if they represent a significant distribution shift (e.g., medical scans, satellite imagery, infrared). Please respond with Yes if the task aligns with DiT’s training distribution, and No otherwise.

This design, combining parameter reuse and branch enhancement, enables DenseDiT to achieve strong multi-task

understanding and generalization with only minimal adaptation effort, rather than requiring full training like previous generation-based dense prediction approaches [17].

3) Input Representation and Training Objective: Our training objective is aligned with the DiT framework. Specifically, we optimize the model to predict the velocity field vθ, minimizing its discrepancy from the true velocity:

0,t ∥vθ(zt,z′,t,Cd,Cp) − u(zt)∥22 (4)

Loss = Ez

where zt is the noisy latent at timestep t, z′ is the latent of the query image, and Cd, Cp are the latent tokens from the demonstration and prompt branches, respectively.

V. EXPERIMENTS

- A. Implementation Details

DenseDiT is built upon FLUX.1-dev [41]. By default, we train a separate model for each task. Additionally, we train a single model on mixed-task data to further evaluate its multi-task learning capability. All training is performed using LoRA [47] fine-tuning. We optimize using Prodigy [49] with safeguard warmup, bias correction, and a weight decay of 0.01. Training is conducted on NVIDIA L40S GPUs with a batch size of 1 and gradient accumulation over 8 steps.

- B. Main results

1) Comparison with General-Purpose Models: To evaluate the generalizability of DenseDiT in various real-world scenarios, we compare its performance on the DenseWorld benchmark with several strong general-purpose models known for their strong zero-shot capabilities to perform various dense prediction tasks directly. For pixel-level regression tasks, we select five stable baselines: Marigold [17], EcoDepth [50], SQLdepth [51], ZeoDepth [52]+PF [53], and Depth-Anything [54]+PF. For pixel-level classification tasks, we include three established general-purpose baselines: SAM [55], CLIPSeg [56], and Grounded-SAM [57]. Table II presents the quantitative results. DenseDiT outperforms all baselines on all tasks in DenseWorld, despite training under limited supervision. DenseDiT achieves an average D-Score of 0.944 and an average S-Score of 0.744, surpassing the second-best models by 4.8% and 45.3%, respectively. These results highlight DenseDiT’s strong generalizability across

diverse real-world scenarios and its practicality for dataefficient dense prediction given the data scarcity inherent to real-world scenarios. Detailed metrics for two representative tasks are presented in Table III and Table IV to enhance the interpretability of evaluation results; results for all other tasks are in the supplementary material. These detailed metrics align consistently with the trends reflected in the D/S-Score.

We also train a single DenseDiT model on mixed-task data to assess multi-task learning capability. Although it is slightly weaker than per-task training, a result consistent with the known challenges of multi-task learning across disparate tasks and domains, it still outperforms all baselines. This demonstrates its robustness under limited supervision, a setting that closely reflects real-world application constraints. Certain tasks even benefit from mixed training, indicating inter-task learning potential not achieved in [30].

TABLE III EVALUATION RESULTS OF DETAILED METRICS FOR Fog Depth Est. TASK

Model δ1 ↑ δ2 ↑ δ3 ↑ REL ↓ Sq-rel ↓ RMS ↓ RMS log ↓

Marigold 0.762 0.938 0.978 0.163 1.187 6.114 0.101 ECoDepth 0.481 0.776 0.915 0.368 3.579 9.657 0.160 SQLdepth 0.344 0.670 0.889 0.408 3.945 10.563 0.189 ZoeDepth+PF 0.575 0.898 0.971 0.239 1.960 7.888 0.133 Depth-Anything+PF 0.588 0.932 0.984 0.223 1.702 7.804 0.110 DenseDiT 0.845 0.969 0.991 0.139 0.599 3.794 0.077 w/ Mixed Training 0.873 0.976 0.991 0.120 0.517 3.678 0.074

TABLE IV EVALUATION RESULTS OF DETAILED METRICS FOR Cardiac Screen TASK.

Model IoU ↑ PA ↑ DiCE ↑

SAM 0.535 0.732 0.652 CLIPSeg 0.439 0.515 0.481 Grounded-SAM 0.402 0.478 0.449 DenseDiT 0.550 0.667 0.617 w/ Mixed Training 0.693 0.800 0.792

- 2) Comparison with Task-Specific Models: We observe that

two tasks in DenseWorld, Road Crack Det. and Urban Layout Anal., have widely studied task-specific models due to their long-standing researches with curated datasets. Such models often perform well within their domains due to architectural customization and large-scale training data, but typically lack generalizability to broader scenarios. To examine this, we compare DenseDiT with state-of-the-art task-specific models on these tasks. For detailed evaluation, we adopt standard metrics from each domain. As shown in Table V and Table VI, DenseDiT outperforms task-specific models in both tasks, despite not relying on task-engineered architecture or large-scale training data. These results highlight the strong performance and generalization of DenseDiT in real-world applications.

- 3) Comparison with Fine-tuned Models: The baselines in

Table II are trained on large-scale datasets relevant to their respective tasks (e.g., depth or segmentation). This provides them with substantial prior task understanding and strong zero-shot capabilities to perform various dense prediction tasks directly. An intuitive yet nontrivial hypothesis is that further fine-tuning these powerful zero-shot models could yield additional performance gains; however, it could also impair their pre-trained task understanding (often referred to

TABLE V TASK-SPECIFIC COMPARISON ON Road Crack Det. TASK

Model IoU↑ PA↑ Dice↑ cIoU↑ [12]

CrackFormer [58] 0.512 0.729 0.058 0.198 TOPO [59] 0.569 0.594 0.625 0.489 nnUet [12] 0.732 0.791 0.811 0.818 CT-CrackSeg [25] 0.677 0.770 0.758 0.603

###### DenseDiT 0.774 0.863 0.855 0.844

TABLE VI TASK-SPECIFIC COMPARISON ON Urban Layout Anals. TASK

Model Recall↑ Precision↑ IoU↑ F1↑

D-LinkNet [60] 0.337 0.509 0.237 0.371 NL-LinkNet [61] 0.435 0.567 0.317 0.469 RCFSNet [62] 0.813 0.569 0.501 0.665 sam-road [63] 0.343 0.715 0.296 0.437

DenseDiT 0.619 0.746 0.512 0.672

as catastrophic forgetting [64]) or lead to overfitting, thus leading to compromised performance, especially given the extreme data scarcity inherent to real-world applications. To test this hypothesis and to thoroughly evaluate DenseDiT’s performance and generalization, we further compare it against fine-tuned versions of these powerful zero-shot baselines. Due to computational constraints, we perform full fine-tuning on the two strongest models, SAM [55] and Marigold [17], following their original setups but using 15 training samples per task from DenseWorld. We also include a recent few-shot model [24] under the same setting. As shown in Table VIII and Table VII, the results validate our hypothesis: although all fine-tuned models benefit from additional real-world data, zero-shot baselines fail to achieve robust generalization under extreme data scarcity. In contrast, DenseDiT outperforms them across all metrics. These results demonstrate DenseDiT’s exceptional capability in adapting to complex, data-scarce realworld scenarios with minimal data requirements.

TABLE VII FEW-SHOT COMPARISON ON THE Road Crack Det. TASK.

Model δ1↑ REL↓ Sq-rel↓ RMS↓ RMS log↓

Marigold (FT) 0.796 0.145 0.974 5.712 0.089 DenseDiT 0.845 0.139 0.599 3.749 0.077

TABLE VIII FEW-SHOT COMPARISON ON THE Fog Depth Est. TASK.

Model IoU↑ PA↑ Dice↑ SAM (FT) 0.622 0.738 0.688 Hossain et al. (FT) 0.625 0.655 0.689 DenseDiT 0.774 0.863 0.855

4) Comparison on Common Benchmarks: To evaluate DenseDiT beyond DenseWorld, we conduct experiments on three common benchmarks: KITTI [10], NYUv2 [9] and Omnicrack30k [12]. DenseDiT is trained on only 15 images from each benchmark’s training set. Table IX, Table X and Ta-

###### Input Images Marigold ECoDepth SQLDepth ZoeDepth+PF DA+PF DenseDiT Ground Truth

DepthEst. Sunset

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Fog

DepthEst. Rain

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

DepthEst. Overcast

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

DepthEst.

- Fig. 4. Qualitative comparisons on pixel-level regression. In the first and second row, DenseDiT successfully predicts occluded structures in fog or shadow, highlighting its capability for scene-level reasoning. The third row showcases DenseDiT’s ability to capture fine-grained details such as distant lampposts and layered foliage. The forth row emphasizes its sensitivity to abrupt depth transitions, producing sharper and more consistent boundaries than competing models.

Input Images SAM CLIPSeg Grounded-SAM DenseDiT

Cigarette

Detect. FireAlert

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

Sys. SpineMorph.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Assess. UrbanLayout

Anal.

Ground Truth

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

- Fig. 5. Qualitative comparisons on pixel-level classification. DenseDiT handles cluttered backgrounds (row 1), detects dynamic concepts like fire (row 2), and localizes fine structures in medical/satellite images (rows 3-4).

TABLE X QUANTITATIVE RESULTS ON NYUV2 BENCHMARK.

Training Samples δ1 ↑ REL ↓ Sq-Rel ↓ RMS ↓ RMS log ↓

Model

SQLdepth 24k 0.928 0.082 0.607 3.914 0.160 DenseDiT 15 0.851 0.129 0.589 3.918 0.072

TABLE XI QUANTITATIVE RESULTS ON OMNICRACK30K BENCHMARK.

Method IoU ↑ PA ↑ cIoU ↑

nnUnet 0.683 0.756 0.479 DenseDiT 0.608 0.635 0.245

baseline, its design prioritizes object-centric cues, leading to suboptimal results in structured or dynamic scenes. More visual comparisons are shown in the supplementary material.

ble XI show that DenseDiT achieves competitive results, compared to the state-of-the-art methods SQLdepth [51] (KITTI), ECoDepth [50] (NYUv2) and nnUnet [12] (Omnicrack30k).

C. Ablation studies

- 1) Effectiveness of the Demonstration Branch: As intro-

duced in Section IV-C2, DenseDiT employs a demonstration branch to enhance task understanding in complex scenarios, activated when DAI=No. We validate its effectiveness via ablations on two representative tasks: Spinal Morphology Assessment, involving intricate medical structures, and Instant Fire Alert, requiring dynamic visual pattern detection. These tasks exemplify real-world challenges where contextual cues are vital. As shown in Table XII, both tasks show notable performance gains with the branch activated, highlighting its impact on generalization in complex settings.

- 2) Training Loss: DenseDiT is a generative-based frame-

TABLE IX QUANTITATIVE RESULTS ON KITTI BENCHMARK.

Training Samples δ1 ↑ REL ↓ RMS ↓ REL log ↓

Model

ECoDepth 48k 0.978 0.059 0.218 0.026 DenseDiT 15 0.928 0.084 0.317 0.035

5) Visual Comparisons: Fig. 4 and Fig. 5 present qualitative comparisons. As shown in Fig. 4, prior methods often struggle with occlusions or low-visibility (e.g., fog, rain, overcast), while DenseDiT yields coherent depth maps by leveraging semantic and visual cues via its prompt and demonstration branches. Fig. 5 illustrates DenseDiT’s advantage in segmentation tasks. Notably, in the fourth row, DenseDiT preserves structured urban layouts in satellite imagery where other models struggle. While SAM is a strong segmentation

work for real-world dense prediction. While prior generativebased dense prediction methods [18] commonly use L1 loss for its robustness to noise in early benchmarks [10], we find that L2 loss (Eq.4), yields smoother convergence and a more stable plateau in DenseDiT, as shown in Fig. 6. We attribute

TABLE XII EFFECTIVENESS OF THE DEMONSTRATION BRANCH.

D/S-Score

[Figure 140]

Method Fire(D/S-ScoreAlert Sys.↑) Spine(D/S-ScoreMorph. Assess.↑)

Tasks with DAI=Yes

w/o demonstration branch 0.744 0.613 w/ demonstration branch 0.876 0.956

Tasks with DAI=No

0 10 20 30 40 50

# Inference Steps

this to two factors: (1) the high quality of DenseWorld reduces the need for noise-tolerant objectives, and (2) L2 loss provides stronger gradient signals for more effective optimization.

Fig. 8. Ablation study on inference steps.

D. Generalization Across Backbones

Loss

To validate the generality of our framework and unified training strategy across different backbones, we implement DenseDiT on SD 2.1, which utilizes a convolutional U-Net denoiser, in contrast to the Transformer-based denoiser used in DiT. We report results on two representative tasks: Automatic Tree Tagging and Fog Depth Estimation; similar trends are observed across other tasks in DenseWorld. As shown in Table XIII and Table XIV, DenseDiT (SD) achieves competitive performance, significantly outperforming all baselines. Although a slight performance gap exists compared to the DiTbased variant, which is expected given the Transformer’s superior long-range modeling capacity, these results consistently demonstrate the effectiveness of our framework and unified training strategy in complex, data-scarce real-world scenarios.

|[Figure 141]<br><br>L1 loss L2 loss<br><br>|
|---|

0.6 0.5 0.4 0.3 0.2 0.1

0 10K 20K 30K 40K

# Iterations

- Fig. 6. Ablation study on loss functions.

3) The Function of Prompt Branch: As introduced in Section IV-C2, DenseDiT includes a lightweight prompt branch using the native text encoder to deliver contextual cues via a text template. Since prompt quality is crucial for generative models [65], we evalute its impact in DenseDiT. Fig. 7 compares four settings: with prompt, without prompt, random prompt (#$%ˆ&*@), and richer prompt. We report results on two representative tasks under both DAI=No and DAI=Yes, with similar trends observed across other tasks in DenseWorld. Prompts consistently improve D/S-Score. In DAI=No tasks, removing or randomizing the prompt degrades performance. The effect is more evident in DAI=Yes tasks, where prompts are the only source of task semantics. Interestingly, richer prompts reduce performance compared to the text template. We attribute this to DenseDiT’s generative backbone, where overly detailed prompts trigger unnecessary imagination and distract from the core task, reducing prediction accuracy. These results highlight the importance of concise, task-aligned prompts for robust and generalizable dense prediction.

[Figure 142]

DAI=No task DAI=Yes task

- Fig. 7. Ablation study on prompt branch.

TABLE XIII QUANTITATIVE RESULTS ON Fog Depth Estimation TASK.

Model δ1 ↑ REL ↓ Sq-rel ↓ RMS ↓ RMS log ↓

Marigold 0.762 0.163 1.187 6.114 0.101 ECoDepth 0.481 0.368 3.579 9.657 0.160 SQLdepth 0.344 0.408 3.945 10.563 0.189 ZoeDepth+PF 0.575 0.239 1.960 7.888 0.133 Depth-Anything+PF 0.588 0.223 1.702 7.804 0.110 DenseDiT (DiT) 0.845 0.139 0.599 3.794 0.077 DenseDiT (SD) 0.842 0.146 0.926 4.745 0.079

TABLE XIV QUANTITATIVE RESULTS ON Automatic Tree Tagging TASK.

Model IoU ↑ PA ↑ DiCE ↑

SAM 0.765 0.840 0.809 CLIPSeg 0.463 0.722 0.579 Grounded-SAM 0.391 0.529 0.457 DenseDiT (DiT) 0.858 0.935 0.898 DenseDiT (SD) 0.845 0.916 0.885

VI. CONCLUSION

We introduced DenseWorld, a benchmark for unified evaluation of dense prediction across 25 diverse real-world tasks characterized by inherent data scarcity. To address the challenge of learning under such constraints, we proposed DenseDiT, a framework that leverages a parameter-reuse architecture and lightweight branches to achieve robust adaptation with minimal data and additional parameters. Extensive experiments demonstrate that DenseDiT attains strong generalization across complex real-world scenarios, significantly outperforming existing general-purpose and task-specific models. This work provides a significant step towards practical and scalable dense prediction for real-world applications.

4) Inference steps: Inference steps is key to generative model. More steps improve quality at higher computational cost. As shown in Fig. 8, we evaluate this effect on two tasks under DAI=No and DAI=Yes conditions, with similar trends across other tasks in DenseWorld. DenseDiT performs stably across a wide range of steps, with a noticeable improvement up to 20 steps. This indicates that 20 steps provide a balance between accuracy and efficiency. Notably, for DAI=Yes tasks, performance slightly declines beyond 30 steps, likely due to over-sampling effects degrading fine-grained details [66].

REFERENCES

- [1] Z. Zhou and Y. Zhu, “Rafpn: Relation-aware feature pyramid network for dense image prediction,” IEEE Transactions on Multimedia, vol. 26, pp. 7787–7800, 2024.
- [2] Y. Lu, S. Sirejiding, Y. Ding, C. Wang, and H. Lu, “Prompt guided transformer for multi-task dense prediction,” IEEE Transactions on Multimedia, vol. 26, pp. 6375–6385, 2024.
- [3] Y. Shu, H. Li, B. Xiao, X. Bi, and W. Li, “Cross-mix monitoring for medical image segmentation with limited supervision,” IEEE Transactions on Multimedia, vol. 25, pp. 1700–1712, 2022.
- [4] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11621–11631.
- [5] S. Zhao, H. Chen, X. Zhang, P. Xiao, L. Bai, and W. Ouyang, “Rs-mamba for large remote sensing image dense prediction,” IEEE Transactions on Geoscience and Remote Sensing, 2024.
- [6] L. Dong, W. Zhai, and Z.-J. Zha, “Unidense: Unleashing diffusion models with meta-routers for universal few-shot dense prediction,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 10525–10534.
- [7] D. Kim, S. Cho, S. Kim, C. Luo, and S. Hong, “Chameleon: A dataefficient generalist for dense visual prediction in the wild,” in European Conference on Computer Vision. Springer, 2024, pp. 422–441.
- [8] D. Kim, J. Kim, S. Cho, C. Luo, and S. Hong, “Universal few-shot learning of dense prediction tasks with visual token matching,” arXiv preprint arXiv:2303.14969, 2023.
- [9] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12. Springer, 2012, pp. 746– 760.
- [10] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” The international journal of robotics research, vol. 32, no. 11, pp. 1231–1237, 2013.
- [11] K. Jin, X. Huang, J. Zhou, Y. Li, Y. Yan, Y. Sun, Q. Zhang, Y. Wang, and J. Ye, “Fives: A fundus image dataset for artificial intelligence based vessel segmentation,” Scientific data, vol. 9, no. 1, p. 475, 2022.
- [12] C. Benz and V. Rodehorst, “Omnicrack30k: A benchmark for crack segmentation and the reasonable effectiveness of transfer learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 3876–3886.
- [13] S. Haug and J. Ostermann, “A crop/weed field image dataset for the evaluation of computer vision based precision agriculture tasks,” in Computer Vision - ECCV 2014 Workshops, 2015, pp. 105–116. [Online]. Available: http://dx.doi.org/10.1007/978-3-319-16220-1 8

- [14] V. Mnih, Machine learning for aerial image labeling. University of Toronto (Canada), 2013.
- [15] D. Y. Chino, L. P. Avalhais, J. F. Rodrigues, and A. J. Traina, “Bowfire: detection of fire in still images by integrating pixel color and texture analysis,” in 2015 28th SIBGRAPI conference on graphics, patterns and images. IEEE, 2015, pp. 95–102.
- [16] X. Qin, H. Dai, X. Hu, D.-P. Fan, L. Shao, and L. Van Gool, “Highly accurate dichotomous image segmentation,” in European Conference on Computer Vision. Springer, 2022, pp. 38–56.
- [17] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, “Repurposing diffusion-based image generators for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9492–9502.
- [18] S. Saxena, A. Kar, M. Norouzi, and D. J. Fleet, “Monocular depth estimation using diffusion models,” arXiv preprint arXiv:2302.14816, 2023.
- [19] Z. Pan, Z. Luo, J. Yang, and H. Li, “Multi-modal attention for speech emotion recognition,” arXiv preprint arXiv:2009.04107, 2020.
- [20] M.-Q. Le, T. V. Nguyen, T.-N. Le, T.-T. Do, M. N. Do, and M.-T. Tran, “Maskdiff: Modeling mask distribution with diffusion probabilistic model for few-shot instance segmentation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 3, 2024, pp. 2874– 2881.
- [21] A. R. Zamir, A. Sax, W. Shen, L. J. Guibas, J. Malik, and S. Savarese, “Taskonomy: Disentangling task transfer learning,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3712–3722.

- [22] C. Xia, X. Wang, F. Lv, X. Hao, and Y. Shi, “Vit-comer: Vision transformer with convolutional multi-scale feature interaction for dense predictions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 5493–5502.
- [23] H. Shi, M. Hayat, and J. Cai, “Unified open-vocabulary dense visual prediction,” IEEE Transactions on Multimedia, vol. 26, pp. 8704–8716, 2024.
- [24] M. R. I. Hossain, M. Siam, L. Sigal, and J. J. Little, “Visual prompting for generalized few-shot segmentation: A multi-scale approach,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 23470–23480.
- [25] H. Tao, B. Liu, J. Cui, and H. Zhang, “A convolutional-transformer network for crack segmentation with boundary awareness,” in 2023 IEEE International Conference on Image Processing (ICIP). IEEE, 2023, pp. 86–90.
- [26] S. Gasperini, N. Morbitzer, H. Jung, N. Navab, and F. Tombari, “Robust monocular depth estimation under challenging conditions,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 8177–8186.
- [27] P. Yin, K. Li, X. Cao, J. Yao, L. Liu, X. Bai, F. Zhou, and D. Meng, “Towards satellite image road graph extraction: A global-scale dataset and a novel method,” arXiv preprint arXiv:2411.16733, 2024.
- [28] A. Bhattad, D. McKee, D. Hoiem, and D. Forsyth, “Stylegan knows normal, depth, albedo, and more,” Advances in Neural Information Processing Systems, vol. 36, pp. 73082–73103, 2023.
- [29] J. Donahue and K. Simonyan, “Large scale adversarial representation learning,” Advances in neural information processing systems, vol. 32, 2019.
- [30] C. Zhao, M. Liu, H. Zheng, M. Zhu, Z. Zhao, H. Chen, T. He, and C. Shen, “Diception: A generalist diffusion model for visual perceptual tasks,” arXiv preprint arXiv:2502.17157, 2025.
- [31] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695.
- [32] H.-Y. Lee, H.-Y. Tseng, and M.-H. Yang, “Exploiting diffusion prior for generalizable dense prediction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7861–7871.
- [33] C. Wang, X. Li, H. Ding, L. Qi, J. Zhang, Y. Tong, C. C. Loy, and S. Yan, “Explore in-context segmentation via latent diffusion models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 7545–7553.
- [34] Kaggle, “Kaggle: Your machine learning and data science community,” https://www.kaggle.com, 2010.
- [35] Papers with Code, “Papers with code: State-of-the-art research and benchmarks,” https://paperswithcode.com, 2019.
- [36] NASA, “Nasa earthdata: Open access earth observation data,” https:// earthdata.nasa.gov, 1994.
- [37] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in neural information processing systems, vol. 27, 2014.
- [38] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 3213– 3223.
- [39] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13. Springer, 2014, pp. 740–755.
- [40] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.
- [41] B. F. Labs, “Flux,” https://github.com/black-forest-labs/flux, 2024.
- [42] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. M¨uller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” URL https://arxiv. org/abs/2403.03206, vol. 2, 2024.
- [43] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” Advances in neural information processing systems, vol. 35, pp. 26565–26577, 2022.
- [44] G. Xu, Y. Ge, M. Liu, C. Fan, K. Xie, Z. Zhao, H. Chen, and C. Shen, “What matters when repurposing diffusion models for general dense perception tasks?” arXiv preprint arXiv:2403.06090, 2024.

- [45] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park, “One-step diffusion with distribution matching distillation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 6613–6623.
- [46] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
- [47] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022.
- [48] OpenAI, “Chatgpt,” 2022, large language model. [Online]. Available: https://chat.openai.com/
- [49] K. Mishchenko and A. Defazio, “Prodigy: An expeditiously adaptive parameter-free learner,” arXiv preprint arXiv:2306.06101, 2023.
- [50] S. Patni, A. Agarwal, and C. Arora, “Ecodepth: Effective conditioning of diffusion models for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 28285–28295.
- [51] Y. Wang, Y. Liang, H. Xu, S. Jiao, and H. Yu, “Sqldepth: Generalizable self-supervised fine-structured monocular depth estimation,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 6, 2024, pp. 5713–5721.
- [52] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. M¨uller, “Zoedepth: Zero-shot transfer by combining relative and metric depth,” arXiv preprint arXiv:2302.12288, 2023.
- [53] Z. Li, S. F. Bhat, and P. Wonka, “Patchfusion: An end-to-end tile-based framework for high-resolution monocular metric depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10016–10025.
- [54] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10371–10381.
- [55] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, B. Caine, R. Mottaghi et al., “Segment anything,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21293–21312.
- [56] T. L¨uddecke and A. S. Ecker, “Image segmentation using text and image prompts,” arXiv preprint arXiv:2112.10003, 2022.
- [57] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.
- [58] H. Liu, X. Miao, C. Mertz, C. Xu, and H. Kong, “Crackformer: Transformer network for fine-grained crack detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 3783–3792.
- [59] B. G. Pantoja-Rosero, D. Oner, M. Kozinski, R. Achanta, P. Fua, F. P´erez-Cruz, and K. Beyer, “Topo-loss for continuity-preserving crack detection using deep learning,” Construction and Building Materials, vol. 344, p. 128264, 2022.
- [60] L. Zhou, C. Zhang, and M. Wu, “D-linknet: Linknet with pretrained encoder and dilated convolution for high resolution satellite imagery road extraction,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 182–186.
- [61] Y. Wang, J. Seo, and T. Jeon, “Nl-linknet: Toward lighter but more accurate road extraction with nonlocal operations,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2021.
- [62] Z. Yang, D. Zhou, Y. Yang, J. Zhang, and Z. Chen, “Road extraction from satellite imagery by road context and full-stage feature,” IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1–5, 2022.
- [63] C. Hetang, H. Xue, C. Le, T. Yue, W. Wang, and Y. He, “Segment anything model for road network graph extraction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2556–2566.
- [64] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.
- [65] C. Jia, C. Xia, Z. Dang, W. Wu, H. Qian, and M. Luo, “Chatgen: Automatic text-to-image generation from freestyle chatting,” arXiv preprint arXiv:2411.17176, 2024.
- [66] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in International conference on machine learning. PMLR, 2021, pp. 8162–8171.

