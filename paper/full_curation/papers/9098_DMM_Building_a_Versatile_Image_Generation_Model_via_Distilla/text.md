# arXiv:2504.12364v1[cs.CV]16Apr2025

## DMM: Building a Versatile Image Generation Model via Distillation-Based Model Merging

Tianhui Song† Nanjing University

Tiezheng Ge Alibaba Group

Weixin Feng Alibaba Group

Bo Zheng Alibaba Group

Shuai Wang† Nanjing University

Xubin Li Alibaba Group

Limin Wang* Nanjing University, Shanghai AI Lab

### https://github.com/MCG-NJU/DMM Abstract

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

The success of text-to-image (T2I) generation models has spurred a proliferation of numerous model checkpoints finetuned from the same base model on various specialized datasets. This overwhelming specialized model production introduces new challenges for high parameter redundancy and huge storage cost, thereby necessitating the development of effective methods to consolidate and unify the capabilities of diverse powerful models into a single one. A common practice in model merging adopts static linear interpolation in the parameter space to achieve the goal of style mixing. However, it neglects the features of T2I generation task that numerous distinct models cover sundry styles which may lead to incompatibility and confusion in the merged model. To address this issue, we introduce a style-promptable image generation pipeline which can accurately generate arbitrary-style images under the control of style vectors. Based on this design, we propose the score distillation based model merging paradigm (DMM), compressing multiple models into a single versatile T2I model. Moreover, we rethink and reformulate the model merging task in the context of T2I generation, by presenting new merging goals and evaluation protocols. Our experiments demonstrate that DMM can compactly reorganize the knowledge from multiple teacher models and achieve controllable arbitrary-style generation.

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

Figure 1. Examples of image generation. Our DMM is able to generate images with various expert styles (realistic style, Asian portrait, anime style, etc.) under the control of style prompts.

achieved great progress and numerous powerful diffusion models are released. These creation platforms make it handy for developers to fine-tune the powerful base models such as Stable Diffusion V1.5 on their specialized datasets to achieve customized generation with various styles and then upload and share the models. In spite of this fast development and prosperity, we are facing some dilemmas. First, for the thousands of personalized models, each contains billions of parameters and is saved as a large checkpoint binary file, leading to serious parameter waste and storage overhead. Second, due to limited data size and computational resources, these models are typically trained to achieve expertise in certain style domains and fail to cover a wide range of scenarios in the world. It brings many inconveniences in practical deployment when it requires different

### 1. Introduction

Diffusion models [18, 47] have steadily emerged as the predominant methods in text-to-image (T2I) generation task. Thanks to the open-source base models [11, 40, 43], tool libraries [3, 51], and communities [9, 28], this field has

†This work was done during interning at Alibaba Group.

*Corresponding author (lmwang.nju@gmail.com).

specific styles. For example, in commercial applications, each model needs to be deployed as an independent service on GPU clusters, which leads to the high overhead of computing resources due to the large model size. For personal users, when switching among different models, the disk and memory loading is also time-consuming, affecting the efficiency and experience. These issues can be alleviated if we can unify these expertise of different expert models into a single one. Currently, it is very challenging to build a versatile model, which is able to cover the knowledge of different models and supports steerable inference to accurately generate arbitrary-style images.

Model merging [49, 60] is a technique that attempts to mitigate the above problems. It has shown efficacy in many fields such as large language model (LLM), but has not been deeply discussed in T2I generation task. The existing prevalent practice for T2I models is to apply static weighted merging of model parameters [55], to achieve the goal of style mixing and enhance the outputs. However, this merging method still has some critical issues. First, this approach limits the range of source models to similar domains, because directly merging models of various styles will cause conflict and style confusion. For example, with a realistic-style merged model, if we continue to merge an animation-style model into it, it will be ambiguous with different patterns and output unexpected results, since it is under a meaningless intermediate state in the parameter space. Additionally, since the merge weights are usually manually set or by brute force search to obtain the best performance and parameters are fixed persistently, this scheme lacks flexibility in the style control during inference.

Therefore, we should rethink model merging for the nature of T2I diffusion models and design more reasonable goals and methodology. As mentioned above, one key feature we should pay attention to is there exist numerous distinct models based on diverse user creativity for different visual styles. This is relatively rare in other machine learning task territories, so direct parameter merging is insufficient to meet real application requirements. Instead, we need to devise innovative and specialized solutions to address the problem. According to our analysis, we summarize the requirements of a model merging system in the context of T2I generation: i) The merged model should preserve distinct capabilities of each source model without ambiguity, thereby truly attaining the substitution of multiple models with a singular and minimizing parameter redundancy and inefficiency. ii) The merged model should have a versatile and controllable inference mechanism to harness the knowledge of different domains, thus facilitating diverse stylistic generation and possibly generalizing to combination functionalities. iii) The training pipeline should be sustainable and scalable, supporting efficient and stable continual learning for new models to be incrementally merged.

Based on the above analysis, we introduce a stylepromptable image generation pipeline which can generate precise arbitrary-style images under the control of style vectors. Based on this style-promptable generation pipeline, we resort to knowledge distillation approach [16] and present a distillation-based model merging paradigm, abbreviated as DMM. As far as we know, we are the first to leverage knowledge distillation for the T2I diffusion model merging and building a versatile style-promptable T2I model. Additionally, we present a quantitative metric FIDt and validate our approach with the public state-of-theart T2I models like the family of Stable Diffusion [40, 43]. By merging eight different models, we train a versatile model that captures the capabilities of various styles concurrently with FIDt 77.51, while maintaining comparable performance of style mixing to previous merging methods.

To sum up, our contributions can be summarized as two folders:

- • We deeply analyze the model merging task in the field of text-to-image generation and rethink a new setting of model merging task objectives with practical value. A corresponding benchmark and quantitative metric are also proposed to measure the performance of model merging under this new setting.
- • We propose a new style-promptable T2I generation pipeline, which is simple to implement, steerable to different styles, and extensible to new export models. We design the first distillation-based model merging paradigm to unify the multiple expertise into a single versatile model, supporting flexible style control during inference.

### 2. Related Work

#### 2.1. Diffusion Models

Diffusion models [6, 18, 24, 39, 44, 47] have gradually become a fundamental approach in the field of generative modeling, surpassing preceding methods in the generation of diverse and high-fidelity images. Song [47] describes the generation process from the continuous-time perspective with stochastic differential equations (SDE), which iteratively denoise an initial noise leveraging the learned score of the data distribution to steer the process toward real data points. Injecting text conditions into the denoising procedure provides a more natural and user-friendly way to control image synthesis [6, 38, 42, 43]. LDM [43] performs the generation process in latent space to reduce computational costs and become the prototype of the most widespread application model Stable Diffusion (SD), which facilitates the nature of open-source AI generative models and spawned hundreds of various high quality models and innovations worldwide. These powerful community models are typically fine-tuned on specialized datasets, thus yielding distinct experts of different style domains.

- 2.2. Model Merging

Model merging is an effective technique that merges the parameters of multiple separate models with different capabilities to facilitate knowledge fusion and build a universal model. Despite its relative novelty, the field of model merging is experiencing rapid advancement and has been successfully applied across various domains [49, 60]. From the perspective of methodology, model merging can be implemented through linear interpolation in parameter space [21, 56, 59], leveraging mode connectivity [12], aligning features or parameters [2] and ensemble distillation [52]. With the emergence of large foundation models including large language models (LLM, [1, 4, 65]) and multi-modal large language models (MLLM, [5, 62]), the model merging method is also explored to enhance the performance and improve the efficiency [13].

In the field of text-to-image (T2I) generation, the potential of model merging has not been fully investigated. The popular practice in the community is to apply the weighted sum of parameters of multiple models [8, 35, 55], to achieve the effect of style mixing. As [7] analyzed, linearly merged diffusion models that have been fine-tuned on distinct stylized data fragments, can generate hybrid styles in a zeroshot learning context. [27] improves the faithfulness of T2I models by merging multiple skill-specific experts trained on synthesized datasets. Additionally, some recent works leverage ensemble learning to fuse multiple models. [53] propose an ensemble method, Adaptive Feature Aggregation (AFA), which dynamically adjusts the contributions of multiple models at the feature level according to various states, to enhance the generation quality. [37] also proposes a strategy of combining aligned features of multiple models, to handle different modality conditions. However, the ensemble approaches should involve multiple models simultaneously during inference, which is computationally and memory expensive. Besides, the disposal and simple merging mechanism limit the model candidates to similar style domains, since injecting completely different models will result in mode shift and confusion. As far as we know, we are the first to leverage distillation training to merge diffusion models and support flexible handling of various styles.

- 3. Method

- 3.1. Preliminary

A generative model is usually used to represent a cerntain data probability distribution pdata(x). Song [45, 47] proposed score-based generative modeling methods, whose key idea is to model the score function, which is defined as the gradient of the log probability density function: ∇x log p(x). To alleviate the difficulty of accurate score estimation in regions of low data density, we can perturb data points with noise and train score-based models on the

noisy data pt(xt) instead [45].

Leveraging the score function, the forward perturbation and backward sampling processes can be respectively described as stochastic differential equations (SDE) [47]:

dxt = f(xt,t)dt + g(t)dw, (1) dxt = f(xt,t) − g(t)2∇xt

log pt(xt) dt + g(t)dw¯ , (2)

where f(·,·) and g(·) denote the drift and diffusion coefficients respectively, and w,w¯ are the standard Wiener process. Moreover, a good property of this system is that there exists an ordinary differential equation (ODE), whose trajectories share the same marginal probability densities pt(xt) as the SDE, dubbed the Probability Flow (PF) ODE:

dxt = f(xt,t) −

- 1

- 2

g(t)2∇xt

log pt(xt) dt. (3)

Once we have trained a time-dependent score-based model s(xt,t;θ) ≈ ∇xt

log pt(xt), this is an instance of a neural ODE and clean images can be generated through solving it.

For training the score-based models, we can minimize the Fisher divergence between the model and the data distributions, which yields the score matching objective:

θ∗ = arg min

EtEx

t∼pt(xt)||s(xt,t;θ) − ∇xt

log pt(xt)||.

θ

(4) Since the regression target ∇xt

log pt(xt) is not tractable directly, many techniques [20, 46, 50] have been explored for optimizing score matching objectives. For example, denoising score matching [50] provides an equivalent but tractable optimization objectives:

θ∗ = arg min

EtEx

0∼p0(x0)Ex

t∼pt(xt|x0) ||s(xt,t;θ) − ∇xt

θ

log pt(xt|x0)||. (5)

This objective is consistent with Denoising Diffusion Probabilistic Model (DDPM, Ho et al. [18]). Specifically, under the formulation of DDPM with noise schedule give by α¯t:

xt = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,I), (6)

where a denoising model ϵ(xt,t;θ) is trained to predict the added noise of a noisy image xt with the objective:

θ∗ = arg min

θ

EtEx

0∼p0(x0)Eϵ∼N(0,I) [∥ϵ(xt,t;θ) − ϵ∥]. (7)

#### 3.2. Task Formulation

Given a set of N pre-trained isomorphic models {s(·;θi)}Ni=1, each parameterized with θi, and each model is trained on different datasets (such as realistic style, anime style, etc.), thus modeling different data distributions. We use {p0(i)(x0)}Ni=1 to represent the distributions

GPU Cluster

GPU Cluster

|GPU#1<br><br>Student UNet<br><br>New Teacher UNet<br><br>Initialize new embedding|
|---|

|GPU#0<br><br>Student UNet<br><br>Teacher UNet #0|
|---|

|GPU#1<br><br>Student UNet<br><br>Teacher UNet #1|
|---|

|GPU#m<br><br>Student UNet<br><br>Teacher UNet #(m%N)<br><br>|
|---|

|Codebook<br><br>Learnable Embeddings<br><br>…|
|---|

[Figure 17]

❄

Codebook

Frozen Student UNet

……

……

…

…

g

Randomly select

Student UNet

Weight Sharing

Extend the codebook

GPU#0

(a)

(b)

- Figure 2. Distributed Training Framework for DMM. (a) The model layout on a GPU cluster during training. Each node is assigned a specific teacher model to jointly supervise a student model with shared parameters. A set of learnable embeddings (style prompts) are maintained to provide hints and differentiate from each other. (b) Continual Learning. New teacher models are involved through initializing and adding new embeddings. The frozen pretrained student model serves as regularization with style prompts randomly selected.

Style-promptable generation. The student model shares the same UNet architecture as the base models, except for a few additional trainable parameters for handling the style index hint i suggesting which model’s style to trigger. Specifically, as shown in Fig. 2a and Fig. 3, we represent different model priors as a codebook of N learnable embeddings, named style prompts. The implementation details of injecting style prompts are stated in Supplementary Material. These prompts are used to specify the image style and modulate the student UNet to mimic the corresponding teacher model. Once the training is finished, the style prompts provide a flexible way to control the styles at test time, which will be discussed in Sec. 4.5.

corresponding to each model. Accordingly, each model predicts the corresponding score function: s(xt,t;θi) ≈ ∇x log p(ti)(xt). Our target is to merge them into one single model with parameter θ∗ while preserving the knowledge and capabilities of each individual expert model. Specifically, with the merged model θ∗, given an style index i, s(·,i;θ∗) should represent the corresponding data distribution p(0i)(x0). One solution is to train a versatile score-based model by distilling knowledge from multiple pre-trained expert models. Consequently, this gives a natural scoredistillation objective as follows:

N

θ∗ = arg min

EtEx

#### 3.4. Loss Function and Data Sampling

t∼p(ti)(xt)

θ

i=1

Score distillation. As analyzed in Sec. 3.2, we apply score distillation loss to learn different target probability distributions. Drawing the connection between DDPM and Score Matching, we can directly perform mean square error (MSE) loss on the outputs, of the ϵ-Prediction parameterized models (such as Stable Diffusion [43]).

log p(ti)(xt)|| ,

||s(xt,t,i;θ) − ∇xt

N

EtEx

t∼p(ti)(xt) [||s(xt,t,i;θ) − s(xt,t;θi)||].

≈ arg min

θ

i=1

(8) Our score distillation objective resorts to the original explicit score matching objective in Eq. (4), since now we have direct access to the target score term.

N

||ϵ(xt,t,i;θ) − ϵ(xt,t;θi)||22. (9)

Lscore(xt,t) =

i=1

#### 3.3. Distillation-based Model Merging Framework

Feature imitation. According to the observation that many previous works [54, 61] have explored, the intermediate features of the model contain rich style information. Therefore, we leverage feature supervision to facilitate knowledge transfer and style learning. Formally, let FSj,FTj ∈ Rh×w×c denote the feature map from student and teacher models, and the subscript represents the index of model layers. For feature imitation, we apply MSE loss:

Distillation-based model merging. We present a simple yet efficient distributed training framework of DMM to implement the score-distillation objective for model merging, as depicted in Fig. 2a. Our task is essentially a knowledge distillation task from multiple teacher models [14, 22, 36] to a single steerable student one. Besides, we propose three types of loss functions and a data sampling strategy to boost the performance. Since it is almost impossible to load all teacher models into a single GPU’s memory, we design to assign a specific teacher model to each GPU uniformly to efficiently utilize GPU memory.

N

FSj(xt,t,i;θ) − FTj(xt,t;θi) 22 ,

Lfeat(xt,t) =

i=1 j∈M

(10)

Teacher UNet

Predicted noise 𝜀

Predicted coarse 𝑥

[Figure 18]

[Figure 19]

Direct

generate

Noisy input 𝑥

[Figure 20]

Multi-Class Adversary Loss

Feature Imitation

Score Distillation discriminator

[Figure 21]

[Figure 22]

Style Prompt

Student UNet

- Figure 3. Style-promptable generation pipeline for disitllation-based model merging. Our proposed distillation objective incorporates three loss terms: Score Distillation, Feature Imitation, and Multi-Class Adversarial Loss.

uler as Eq. (6) [33, 58]:

where M denotes the set of layer indices that need to be supervised. Our experiment results demonstrate that the strategy of naively supervising all the layers’ features can lead to a substantial performance boost.

√1 − α¯tϵ(xt,t;θ) . (12)

1 √α¯t

xt −

g(xt,t;θ) =

Training data sampling. Herein, there is still a challenge in that we do not have direct access to the original training data of each teacher model, thus unable to sam-

Multi-class adversarial loss. To further enhance the model’s ability to discriminate and fit different data distributions, we incorporate an additional GAN objective into our training framework. Generative Adversarial Networks (GAN [45]) implicitly model the real data distribution by training a generator and a discriminator competitively. Essentially, it is optimizing Jensen–Shannon divergence [10] for matching two probability distributions. Considering we aim to learn multiple target data distributions simultaneously, we can naturally tailor a multi-class GAN to substitute the vanilla binary GAN. Specifically, given N teacher models, the total number of classification heads is 2N, where the first N classes represent N target styles and the last N classes represent fake ones. The discriminator is trained to not only distinguish between real and fake images but also to distinguish different style distributions. This yields our multi-class adversarial loss as below:

ple training data points x ∼ pt(i)(xt). Fortunately, due to the good property of direct access to the score target under distillation, this optimization process can be considered a conventional regression task, and the training data can be generalized to a common dataset. As depicted in Fig. 3, the final optimization objective is the weighted combination of the three loss terms as below:

Ltotal = EtEx

t∼pt(xt) [Lscore + λfeatLfeat + λadvLadv].

(13) Our experimental results show that sampling from a general common training dataset is enough to effectively distill knowledge from different teacher models. The intuition behind this is since we train the model on noise-perturbed data, the different noisy data distributions overlap with each other, especially at large timesteps. To compensate for regions of low data density at small timesteps, we use teacher models to synthesize a small number (hundreds) of images and fine-tune the model with a few thousand iterations. This stage can further refine the generation quality and is highly efficient, consuming only a few GPU hours.

N

Ladv(xt,t) = −

[log Di(g(xt,t,i;θ))

(11)

i=1

+ log DN+i(g(xt,t;θi))],

where Di(·) is the discriminator predicted probabilities for the i-th class, and g is the generation function for sampling clean images from model outputs. In order to sample efficiently, we leverage the formulation of predicting x0 directly from xt and ϵ(xt,t;θ) according to the noise sched-

#### 3.5. Incremental Learning with Regularization

Our proposed distillation framework supports flexible continual learning to incrementally merge new models into the

current checkpoint. In practice, we only need to extend the set of style prompts by adding new ones (randomly initialized), then fine-tune the model with supervision as discussed in Sec. 3.4 for them. However, only optimizing for new targets can lead to severe problem of catastrophically forgetting existing knowledge. To alleviate this issue, we propose an efficient incremental learning approach with regularization as illustrated in Fig. 2b. We adopt a self-supervised approach to preserve the knowledge of the learned models to achieve regularization. Specifically, we treat the pre-trained merged model as the teacher with its parameters frozen, to supervise the student model with corresponding style prompt inputs. The style prompt indices are randomly selected in each training iteration. This method allows for efficient device resource consumption that assigning one GPU is enough for regularization, instead of deploying N GPUs for all old teachers.

portrait photo of a girl, long golden hair, flowers, best quality

[Figure 23]

handsome boy, blue suit, upper body portrait

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

[Figure 37]

[Figure 38]

[Figure 39]

A white plate containing salmon, rice, and broccoli.

[Figure 40]

- Figure 5. Visual generation results with different style selections. In each group, the first line is our model’s results, and the second line is the corresponding results of the teacher models.

elf portrait,enchanting beauty,fantasy,ethereal glow,pointed ears,delicate facial features,long elegant hair,nature-themed attire,mystical ambiance,soft lighting,tranquil expression,harmonious with nature,subtle magical elements,serene,intricate jewelry,dreamlike quality,pastel colors,

[Figure 41]

- Figure 6. Multi-scale visual results of DMM. More examples are provided in the Supplementary Material.

[Figure 42]

- Figure 7. Results of DMM-SPLAM with only 4 inference steps.

### 4. Experiments

#### 4.1. Evaluation

Before presenting the experiments, we first introduce the evaluation protocol under our task setting. To quantitatively measure the performance of the model for learning different target model distributions, we propose an evaluation metric based on Fr´echet inception distance (FID). The FID [15] metric measures the distance between two probability distributions, the distributions of model predictions and reference images. Consequently, we sample images through the student model with different N style prompts and N different teacher models and calculate the FID score pairwisely,

[Figure 43]

[Figure 44]

70

which gives an FID matrix M ∈ RN×N satisfying:

70

- 9.63 11.53 10.55 10.13 18.31 63.66 14.89 21.96

11.81 9.84 13.23 12.49 15.51 58.30 12.73 18.61

- 10.67 12.85 9.57 11.37 19.15 65.54 16.11 23.28

- 9.47 12.08 10.54 10.15 20.10 71.18 15.85 24.11

12.03 9.51 13.53 12.84 16.86 65.54 13.24 20.37

- 10.48 13.40 9.23 11.70 20.90 72.87 16.86 25.41

01234567

01234567

60

60

M(i,j) = FID P ˆS(i),PT(j) , (14)

50

50

10.27 12.36 11.63 9.87 20.11 65.23 16.27 23.40

10.12 12.95 11.53 9.52 21.85 72.67 17.37 25.65

40

40

19.85 17.14 20.91 21.41 9.76 41.05 11.26 10.77

20.15 17.03 21.38 21.88 9.54 47.36 10.94 11.28

where PS/T(i) represents the distribution of generated images of student/teacher models with the i-th style. Ideally, we

30

30

70.82 66.53 72.58 72.38 50.07 9.05 54.70 45.93

70.76 65.59 73.19 72.10 47.34 8.21 53.37 43.48

15.46 13.43 16.79 16.73 10.45 46.76 9.78 12.51

15.78 13.35 17.20 17.19 10.94 53.52 9.70 13.62

20

20

hope the FID scores on the diagonal to be as small as possible, which indicates how well the model matches different target distributions respectively. Therefore, we define the metrics FIDt as the trace of M and use it to observe and measure the performance of model merging:

24.36 21.10 25.79 25.60 11.92 38.41 14.14 10.02

24.55 20.89 26.19 25.89 11.48 44.59 13.78 9.72

10

10

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7

Figure 4. Heatmap of the FID matrix. The left one is the result M of our model, and the right one is the reference matrix Mref.

|Methods|FIDt↓<br><br>|
|---|---|
|+ Score Distillation + Feature Imitation + Multi-Class Adversarial<br><br>+ Synthesized Finetune|80.69 79.27 78.38 77.51<br><br>|
|Teacher Reference<br><br>|74.91|

N

M(i,i). (15)

FIDt := Tr(M) =

i=1

Besides, for each style we leverage the teacher models to sample two batches of images with different random seeds and calculate their FID matrix Mref as above. We consider the Mref as a reference since it suggests the upper bound of model performance. In this paper, we sample 5k images per

- Table 1. Ablation Results. The first three lines are our proposed three types of losses. The fourth line is the fine-tuning stage with synthesized data. The last line is the reference upper bound.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Ours

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Weighted Merging

Figure 8. Visualization comparison with Weighted Merging.

batch using the text prompts from MS-COCO [30] validation set for FID calculation.

#### 4.2. Implementation Details

Our main experiments are based on SDv1.5 architecture and the student model is initialized from SDv1.5 weights. For base models to be merged, we select eight popular models with different styles from open-source model communities. We leverage JourneyDB [48] as our training dataset. The distillation training is conducted on 16 A100 GPUs, and the batch size is 320 with each GPU holding 20 samples. We train the model for 100k iterations, which takes about 32 GPU days. To reveal that our approach can generalize to a broader range of distinct styles and model formats, we also conduct experiments on SDXL architecture and LoRA models, and report results in the Supplementary Material, as well as more implementation details.

#### 4.3. Main Results

The FID score matrix M and reference matrix Mref are illustrated in Fig. 4, and the corresponding FIDt metric is presented in Tab. 1, where the default final result is highlighted in gray . Compared to the reference FIDt of 74.91, our approach achieves 77.51 FIDt which is quite close to the upper bound. Besides, we can observe that our result matrix M shows similar patterns to Mref. This demonstrates our method’s effectiveness in matching all different target model distributions, achieving all-in-one functionality, and contributing to a versatile model.

For visual qualitative results, we show generated images from our model with different style prompts in Fig. 5. We can see that our model can synthesize images that are highly consistent with the teacher models. This further illustrates that the domain knowledge and capabilities of different models have been compactly merged into one single model, thus significantly reducing parameter redundancy.

Simultaneously, it is worth noting that since some teacher models can synthesize images at multiple scales, our DMM can well inherit this ability owing to knowledge distillation even though it does not access the multi-scale data during training, as shown in Fig. 6.

a girl, red and yellow hairscarf

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

wearing sunglasses

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

- Figure 9. Visual results of DMM integrated with ControlNetCanny, IP-Adapter and character LoRA.

[Figure 73]

6 2

[Figure 74]

6 5

- Figure 10. The results of interpolation between two styles. The number on the side is the model index. The weight list of one ingredient is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].

[Figure 75]

- Figure 11. Panoramas results of DMM combined with Mixtureof-Diffusers, where various styles are well harmonized. 4.4. Ablation Study

Loss functions and synthesized data. Tab. 1 ablates the design of our loss functions: score distillation, feature imitation, and multi-class adversarial loss. It can be seen that feature imitation can increase the baseline performance by 1.5 FIDt, and adding multi-class adversarial loss further improves by 1.42 FIDt, which indicates the effectiveness of both feature imitation and adversarial learning for distribution matching in the model merging task. Furthermore, the fine-tuning stage utilizing synthesized data can enhance the FIDt by 0.87 at a minimal cost.

Incremental learning with regularization. To ascertain the efficacy of our proposed incremental learning mechanism, we conduct experiments in Tab. 2. #1 and #2 are two experiments with our method to merge four and eight mod-

|#|Stage<br><br>|#Model|Regular.<br><br>|1<br><br>|2|3<br><br>|4<br><br>|5|6<br><br>|7<br><br>|8|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0|Teacher<br><br>|8|-<br><br>|9.5<br><br>|9.5|9.2<br><br>|9.5|9.5<br><br>|8.2|9.7<br><br>|9.7|
|1<br><br>2<br><br><br>|Train Train|4 8<br><br>|-<br><br>|9.5 9.6|9.8 9.8<br><br>|9.5 9.6|9.7 9.9<br><br>|9.8<br><br>|9.0<br><br>|9.8|10.0<br><br>|
|3<br><br>4<br>|Fine-tune Fine-tune<br><br>|4 + 4 4 + 4<br><br>|✗ ✓<br><br>|21.0 9.7<br><br>|23.9 9.8<br><br>|26.7 9.6<br><br>|18.5 9.8<br><br>|9.7 9.9<br><br>|9.6 9.0<br><br>|9.7 9.8<br><br>|10.0 10.0<br><br>|

- Table 2. Incremental Learning. Each column is Experiment Number, Training Stage, Number of Merged Models, Regularization, and FID scores on the diagonal of M. ‘Teacher’ represents the reference upper-bound results of teacher models. ‘Train’ denotes the model is trained from the start, ‘Fine-tune’ denotes the model is fine-tuned from the checkpoint of #1.

|Method<br><br>|CLIP-Score|Aes-Score|Pick-Score<br><br>|
|---|---|---|---|
|WM DMM<br><br>|32.47 32.51<br><br>|5.58 5.58<br><br>|22.36 22.35<br><br>|

- Table 3. Quantitative comparison with Weighted Merging on metrics of CLIP-Score [41], Aesthetic-Score [26], and PickScore [25]. ‘WM’ denotes the Weighted Merging method.

els respectively, and #3,4 are experiments to fine-tune from #1 checkpoint with and without regularization to add the remaining four models. Experiment #3 shows that the first four FID scores have diverged without any regularization, revealing that the model severely suffers from catastrophic forgetting during continual learning. Experiment #4 crucially demonstrates that our regularization strategy can alleviate this phenomenon and reach parallel performance with full training, thus guaranteeing stable incremental learning.

- 4.5. Extension Applications

more flexible and can afford many style generation simultaneously. Additionally, to further illustrates that our mixing strategy really take effects, we perform interpolation on two styles and adjust the weights as shown in Fig. 10, which delivers a smooth and stable transition between them and verifies the semantics of the embedding space. More results are provided in the Supplementary Material.

Compatibility with plugins. Due to the model being trained based on Stable Diffusion, and the feature imitation module enabling the intermediate representation of hidden layers to be aligned with the base model, our DMM is seamlessly compatible with various downstream plugins such as ControlNet [64], LoRA [19], and IP-Adapter [61], without extra training. Besides, our approach can be easily adapted to techniques of integrating multiple diffusion processes and spatial control, such as Mixture-of-Diffusers [23] and MultiDiffusion [36]. Since DMM can leverage different styles flexibly through our proposed style prompts, it can further boost the diversity and versatility of integrated generation. We display some results with ControlNet, IPAdapter and LoRA in Fig. 9, from which we can see that these plug-and-play modules work on different styles with only a single versatile model. For integrated generation pipelines, we show the results of DMM combined with Mixture-of-Diffusers in Fig 11, the panoramas that harmonize different styles. These extensions greatly boost the efficiency of creativity and application. More results are in the Supplementary Material. These results show the general application potential of our DMM method.

Apart from triggering different generation styles, our framework’s significant advantage is its flexibility and scalability, which support many extension applications with excellent style control ability.

Style mixing. Benefiting from our proposed embeddingbased style prompts, our DMM is easy to tailor for style combinations during inference. Compared to the common practice of weight merging [55], which manually predetermines the weights to merge parameters of multiple SD models for mixed effects, our DMM can achieve the purpose more efficiently. Specifically, given the N prior embeddings {ei}Ni=1, we can represent the mixture of different model distributions through interpolation of them: e˜ = Ni=1 wiei,s.t. Ni=1 wi = 1, and feed it into the model. To verify the effectiveness of our proposed embedding interpolation approach for style mixing, we conduct experiments on the first four realistic-style merged models. Specifically, for DMM we directly average the first four style prompts during inference, and for the baseline method, we merge the parameters of the four models. We compare the results on the test set of COCO30K [30], as shown in Fig 8 and Tab. 3. We can see that DMM can achieve semantically and aesthetically comparable performance of style mixing to Weighted Merging (WM), while our approach is

Transferring to distillation-based acceleration method. Because our approach is based on distillation and the style prompt is lightweight to embed into the model, our merging mechanism can be naturally combined with many distillation-based acceleration methods [34, 57]. We illustrate the results in Fig. 7, as well as the implementation and more results in Supplementary Material.

### 5. Conclusion

In this paper, we have rethinked the model merging task in the realm of T2I diffusion models and built a versatile stylepromptable diffusion models for steerable image generation. Specifically, we present DMM, a simple yet effective

merging paradigm based on score distillation. DMM leverages three types of loss functions to boost the merging performance and perform regularization to support stable continual learning. With our designed embedding-based style control mechanism, users can operate the style prompts to execute various style combinations flexibly during inference. We design an evaluation benchmark with the new metric and the results demonstrate our merged model is able to well mimic the expert teacher model in image generation quality. We hope our DMM can facilitate the development of model merging in image generative models.

### Supplementary Material

### A. Implementation Details

This section provides a brief overview of the implementation details.

#### A.1. Style Prompts Design

As proposed in Sec. 3.3 in main paper, we represent the different style prompts as a set of trainable embeddings, forming a codebook. The dimension of embeddings is the same as that of timestep embedding in SDv1.5, which is 1280. These embeddings are randomly initialized before training and can be indexed to imply the mode. We adopt a simple strategy to inject the style prompts into the UNet model. Specifically, we first align the embeddings with an MLP and then add it to the timestep embedding, which is lightweight enough for plug-and-play purpose. The codebook and the MLP which contain two Rd×d linear projections are all the additional parameters.

#### A.2. Multi-Class GAN Classifier Design

The architecture design of our multi-class GAN classifier is inspired by DMD2 [63] and SDXL-Lightning [29]. Specifically, we attach a sequence of convolutions, group normalization, and SiLU activations on top of the middle block of the backbone UNet. The only difference is that the final classification projection dimension is 2N instead of a single scalar. Moreover, we diffuse the discriminator input images with random noise to improve the robustness.

#### A.3. Training Setting

SDv1.5 For the main experiment on SDv1.5 architecture, we merge a group of eight popular models from the opensource platform and list them in Tab. 4.

We leverage JourneyDB [48] as our training dataset. The distillation training is conducted on 16 NVIDIA A100 GPUs, and the batch size is 320 with each GPU holding 20 samples. We use AdamW optimizer and the learning rate is 10−5, for both the diffusion model and the discriminator. We train the model for 100k iterations, which costs about 32 GPU days. The loss weights in Eq. 13 are set

as λfeat = 0.001,λadv = 0.01. To support widely used Classifier-Free Guidance (CFG, [17]), we replace 10% text embeddings with null embeddings for training the unconditional model. For the synthesized fine-tuning stage, we synthesize 1.5k images per teacher and fine-tune the model with batch size 10 and 10k iterations, costing about 16 GPU hours.

SDXL We additionally conduct experiments on SDXL architecture and display the results in Sec. B. The teacher models are listed in Tab. 5.

The training hyper-parameter settings are the same as SDv1.5 experiments, except that the batch size is 10 per GPU to accommodate GPU memory usage.

SDv1.5-LoRA To verify that our DMM can also work on merging LoRAs, we conduct experiments on four LoRAs. In particular, we select four unique challenging styles to examine the performance. The model information is displayed in Tab. 6, and the results are shown in Fig. 16 and Fig. 17, which provides an interesting illustration and validates the feasibility of our approach in merging LoRAs.

SDv1.5-SPLAM As claimed in Sec. 4.5 in main paper, our approach can be transferred to distillation-based acceleration methods and obtain a fast version of the merged model. To train a DMM-SPLAM, we initialize the model with the checkpoint of vanilla DMM and replace the loss with SPLAM loss. The batch size is 50 per GPU, and since rapid convergence, we have trained DMM-SPLAM with 8 GPUs for 6k iterations.

#### A.4. DMM Training Algorithm

We present the overall training process in Algorithm 1. We omit some insignificant parts such as operations of VAE and text condition for simplicity.

#### A.5. Inference Setting

The prompts of generated samples are mainly drawn from MSCOCO [30], PartiPrompts [11], and open-source platforms. For SDv1.5 architecture, the generation resolutions contain 512x512, 512x768, and 768x512. For SDXL architecture, the generation resolutions contain 1024x1024, 1536x1024, and 1024x1536. The guidance (CFG) scale is set to 7 constantly, and we adopt a simple negative prompt ‘worst quality,low quality,normal quality,lowres,watermark,nsfw’. We use DPM-Solver++ [31, 32] scheduler, the number of inference steps is 25.

### B. Additional Results

This section provides more results of generated samples under different tasks to demonstrate our approach’s performance and flexibility.

- B.1. Main Results

We display more results of our DMM for text-to-image generation with different styles in Fig. 12 and Fig. 13. It is worth noting that since some teacher models can generate images at multiple scales, our DMM can well inherit this ability even though it does not access the multi-scale data during training, as shown in Fig. 14. The results on SDXL are provided in Fig. 15. The results on LoRAs are provided in Fig 16. The results on SPLAM are provided in Fig. 18.

- B.2. Results with Plugins

We display more results of our DMM equipped with various downstream plugins: ControlNet in Fig. 19 and LoRA in Fig. 20. It can be seen that our model maintains the power of these plugins while presenting different styles.

We show more results of DMM combined with Mixtureof-Diffusers in Fig 21, the panoramas that harmonize different styles.

- B.3. Results of Style Mixing

In this part, we illustrate the effectiveness of our proposed approach to style mixing. In Fig. 22, we interpolate the eight styles pairwisely with equal weights and show the grid of results. In Fig. 23 and Fig. 17, we perform interpolation on two styles and adjust the weights, delivering a smooth and natural transition between them.

### C. Limitation and Future Works

As far as we know, we are the first to comprehensively analyze and reorganize the model merging task in the context of diffusion generative models. We are also the first to propose a baseline training method for diffusion model merging based on knowledge distillation. While training brings the advantage of performance improvements, it also requires more computing resources. Our DMM currently consumes about 32 GPU days for 100k training iterations to reach optimal convergence, which is a relatively high overhead. We hope for more observation and investigation in the scenario of diffusion model merging and that more efficient approaches can be explored.

|No.<br><br>|Model Name|Style<br><br>|Source|
|---|---|---|---|
|1<br><br>|JuggernautReborn|Realistic<br><br>|https://civitai.com/models/46422|
|2|MajicmixRealisticV7<br><br>|Realistic, Asian|https://civitai.com/models/43331|
|3|EpicRealismV5<br><br>|Realistic<br><br>|https://civitai.com/models/25694|
|4<br><br>|RealisticVisionV5|Realistic<br><br>|https://civitai.com/models/4201|
|5<br><br>|MajicmixFantasyV3<br><br>|Anime|https://civitai.com/models/41865|
|6|MinimalismV2<br><br>|Illustration|https://www.liblib.art/modelinfo|
|7<br><br>|RealCartoon3dV17<br><br>|3D Cartoon<br><br>|https://civitai.com/models/94809|
|8|AWPaintingV1.4<br><br>|Anime|https://civitai.com/models/84476|

- Table 4. The information of all models to be merged on SDv1.5.

|No.<br><br>|Model Name<br><br>|Style|Source|
|---|---|---|---|
|1<br><br>|JuggernautXLV10|Realistic<br><br>|https://civitai.com/models/133005|
|2<br><br>|LEOSAMXLV7<br><br>|Realistic|https://www.liblib.art/modelinfo|
|3<br><br>|GhostXLV1<br><br>|Anime|https://civitai.com/models/312431|
|4|AnimagineV3.1<br><br>|Anime|https://civitai.com/models/260267|

- Table 5. The information of all models to be merged on SDXL.

|No.<br><br>|Model Name|Style|Source|
|---|---|---|---|
|1<br><br>|CJIllustration|Business flat illustrations<br><br>|https://www.liblib.art/modelinfo/|
|2<br><br>|MoXin|Traditional Chinese ink painting<br><br>|https://civitai.com/models/12597|
|3<br><br>|MPixel|Pixel art|https://civitai.com/models/44960|
|4<br><br>|SCHH<br><br>|Natural scenery|https://www.liblib.art/modelinfo|

- Table 6. The information of all models with rare and unique styles.

Algorithm 1 DMM Training Algorithm Require: Number of teacher models N. Number of GPU nodes M. Current GPU node index j. Student model ϵ(·;θ).

Teacher models {ϵ(·;θi)}Ni=1. Discriminator D(·;η).

- 1:
- 2: Get the teacher index of the current node i ← j%N + 1
- 3: repeat
- 4: Sample x0 from data distribution, ϵ ∼ N(0,I),t ∈ [0,T]
- 5: xt ← add noise(x0,ϵ,t)

- 6: ϵstu,Fstu ← ϵ(xt,t,i;θ) ▷ Get model outputs and intermediate features.
- 7: ϵtea,Ftea ← ϵ(xt,t;θi)
- 8: xˆ0,stu ← g(xt,t,ϵstu) ▷ Predict the clean image directly.
- 9: xˆ0,tea ← g(xt,t,ϵtea)
- 10: lstu ← D(xˆ0,stu) ▷ Predict the logits l ∈ R2N.
- 11: Ladv ← cross entrophy(lstu,i)

- 12: Lscore ← ∥ϵstu − ϵstu∥22
- 13: Lfeat ← ∥Fstu − Fstu∥22
- 14: Lgen ← Lscore + λfeatLfeat + λadvLadv
- 15: Update θ ← θ − ∂∂θLgen ▷ Generator backward.

- 16: lstu ← D(xˆ0,stu),ltea ← D(xˆ0,tea)
- 17: Ldis ← cross entrophy(lstu,N + i) + cross entrophy(ltea,i)

- 18: Update η ← η − ∂∂ηLdis ▷ Discriminator backward.

- 19: until converged

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 3

- [2] Samuel K Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. arXiv preprint arXiv:2209.04836, 2022. 3
- [3] AUTOMATIC1111. Stable Diffusion Web UI, 2022. 1
- [4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 3
- [6] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [7] Benjamin Biggs, Arjun Seshadri, Yang Zou, Achin Jain, Aditya Golatkar, Yusheng Xie, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto. Diffusion soup: Model merging for text-to-image diffusion models. arXiv preprint arXiv:2406.08431, 2024. 3
- [8] Chilloutmix. https://civitai.com/models/ 6424/chilloutmix. 3
- [9] Civitai. Civitai. 1
- [10] Dominik Maria Endres and Johannes E Schindelin. A new metric for probability distributions. IEEE Transactions on Information theory, 49(7):1858–1860, 2003. 5
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1, 9
- [12] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning, pages 3259–3269. PMLR, 2020. 3
- [13] Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. Arcee’s

- mergekit: A toolkit for merging large language models. arXiv preprint arXiv:2403.13257, 2024. 3
- [14] Yanan Gu, Cheng Deng, and Kun Wei. Class-incremental instance segmentation via multi-teacher networks. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1478–1486, 2021. 4
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [16] Geoffrey Hinton. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 2
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 9
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2, 3
- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 8
- [20] Aapo Hyv¨arinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005. 3
- [21] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089, 2022. 3
- [22] Yuxuan Jiang, Chen Feng, Fan Zhang, and David Bull. Mtkd: Multi-teacher knowledge distillation for image superresolution. arXiv preprint arXiv:2404.09571, 2024. 4
- [23] Alvaro´ Barbero Jim´enez. Mixture of diffusers for scene composition and high resolution image generation. arXiv preprint arXiv:2302.02412, 2023. 8
- [24] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 2
- [25] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36: 36652–36663, 2023. 8
- [26] LAION-AESTHETICS. https://laion.ai/blog/ laion-aesthetics/. 8
- [27] Jialu Li, Jaemin Cho, Yi-Lin Sung, Jaehong Yoon, and Mohit Bansal. Selma: Learning and merging skill-specific textto-image experts with auto-generated data. arXiv preprint arXiv:2403.06952, 2024. 3
- [28] LibLib. Liblib. 1
- [29] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 9
- [30] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In

A kangaroo wearing an orange hoodie and blue sunglasses standing on the grassin front of the Sydney Opera House

[Figure 76]

A brown teddy bear holding flowers in front of a grave

[Figure 77]

Figure 12. More text-to-image results of DMM compared with the teacher models. In each group, the first line is the results of DMM, and the second line is the results of teacher models.

Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 7, 8, 9

- [31] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787,

2022. 9

- [32] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 9
- [33] Guansong Lu, Yuanfan Guo, Jianhua Han, Minzhe Niu, Yihan Zeng, Songcen Xu, Zeyi Huang, Zhao Zhong, Wei Zhang, and Hang Xu. Pangu-draw: Advancing resource-efficient text-to-image synthesis with timedecoupled training and reusable coop-diffusion. arXiv preprint arXiv:2312.16486, 2023. 5
- [34] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 8
- [35] Majicmix. 3
- [36] Ze Meng, Xin Yao, and Lifeng Sun. Multi-task distillation: Towards mitigating the negative transfer in multi-task learning. In 2021 IEEE International Conference on Image Processing (ICIP), pages 389–393. IEEE, 2021. 4, 8
- [37] Nithin Gopalakrishnan Nair, Jeya Maria Jose Valanarasu, and Vishal M Patel. Maxfusion: Plug&play multi-modal generation in text-to-image diffusion models. arXiv preprint arXiv:2404.09977, 2024. 3

- [38] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [39] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 2

- [40] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 2
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 8
- [42] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 4
- [44] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using

1boy,handsome male,face,beard, beige shirt, white background

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

masterpiece,bestquality, 1girl, close up, colorful, cinematiclighting, bustshot, extremelydetailedCGunity8kwallpaper, redhair, solo,smile,intricateskirt,flyingpetal,Flowerymeadow sky, cloudy_sky, building, moonlight, moon, night, darktheme, light, fantasy,

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

World of ice and snow, Epic level scenery

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

a squirrel in a field

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

a boy and a penguin sitting on the moon

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Figure 13. More text-to-image results of DMM.

nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2

- [45] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 3, 5
- [46] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in Artificial Intelligence, pages 574–584. PMLR, 2020. 3
- [47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1, 2, 3
- [48] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024. 7, 9
- [49] Anke Tang, Li Shen, Yong Luo, Han Hu, Bo Do, and Dacheng Tao. Fusionbench: A comprehensive benchmark of deep model fusion. arXiv preprint arXiv:2406.03280, 2024. 2, 3
- [50] Pascal Vincent. A connection between score matching and

- denoising autoencoders. Neural computation, 23(7):1661– 1674, 2011. 3
- [51] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 1
- [52] Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, and Shuming Shi. Knowledge fusion of large language models. arXiv preprint arXiv:2401.10491, 2024. 3
- [53] Cong Wang, Kuan Tian, Yonghang Guan, Jun Zhang, Zhiwei Jiang, Fei Shen, Xiao Han, Qing Gu, and Wei Yang. Ensembling diffusion models via adaptive feature aggregation. arXiv preprint arXiv:2405.17082, 2024. 3
- [54] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 4
- [55] Weighted-Merging. https://github.com/hakomikan/sd-webui-supermerger. 2, 3, 8
- [56] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Ko-

masterpiece,best quality,igirl,solo,pinkeyes,stuffed bunny,white thighhighs,pinkhair,bangs,smile,looking at viewerlonghair,hair intakes,holding,dress,bareshoulders,closed mouth,virtual youtuber,offshoulder,holding stuffed toy,shoes,cowboyshot,ribbon,long sleeves,multicoloredhair,white jacket,pink ribbon,whitehair,bow,braid,white dress,hair betweeneyes, gradient background, absurdres,veryaesthetic,multicolored hair,amazingquality,detached sleeves

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

1girl, black hair with bangs, white sweater, orange background

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

A truck that is sitting in the grass.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

A young child using an electric sander on a long piece of wood.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Figure 14. More multi-scale text-to-image results of DMM.

rnblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR, 2022. 3

- [57] Chen Xu, Tianhui Song, Weixin Feng, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Accelerating image generation with sub-path linear approximation model. arXiv preprint arXiv:2404.13903, 2024. 8
- [58] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai

- Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36, 2024. 5
- [59] Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. Resolving interference when merging models. arXiv preprint arXiv:2306.01708, 1, 2023. 3
- [60] Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging

mysterious silhouette woman with hat,by Minjae Lee,Carne Griffiths,Emily Kell,Steve McCurry,Geoffroy Thoorens,Aaron Horkey,Jordan Grimmer,Greg Rutkowski,amazing depth,double exposure,surreal,geometric patterns,intricately detailed, bokeh,perfect balanced,deep fine borders,artistic photorealism,smooth

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

a beautiful female model, In the studio, beautiful flowers, depth of field, brightly

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

masterpiece,best quality,1girl,fractal art,ink wash painting art style,realistic art style

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

1boy, young man black hair putting hair, male k-pop idol, lovingly looking camera, levi ackerman, medium portrait soft light, chop, beautiful model, oval face, vivid

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

backlit portrait of a girl, soft sunlight halo around her head, wearing a thick blue scarf and a black coat, straight face, light leaking from an overexposed window, neutral, relaxed expression, 25 years old, masterpiece

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Figure 15. Text-to-image results of DMM on SDXL architecture.

tree,day,water,blue sky,forest,rock

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

protrait of a woman, full body, flowers

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

- Figure 16. Visual results of DMM with setting of four LoRAs in Tab. 6. In each group, the left part is the teacher models’ results, the right is DMM’s results.

[Figure 186]

2 1

[Figure 187]

4 1

[Figure 188]

- 2 4

[Figure 189]

- 3 2

- Figure 17. Results of interpolation between two styles with setting of four LoRAs in Tab. 6. The number on the side is the model index.

in llms, mllms, and beyond: Methods, theories, applications and opportunities. arXiv preprint arXiv:2408.07666, 2024. 2, 3

- [61] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 4, 8

- [62] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023. 3
- [63] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024. 9
- [64] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on

Computer Vision, pages 3836–3847, 2023. 8

[65] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. 3

1girl, pink hair

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

a handsome boy, yellow background, blue eyes

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Figure 18. Results of DMM combined with SPLAM with only 4 inference steps.

[Figure 206]

a boy on the country road, best quality

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

room

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

1girl in the kitchen

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

Figure 19. More results of DMM combined with ControlNet. We leverage the control conditions of OpenPose, MLSD, and Depth.

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

Figure 20. Results of DMM combined with different LoRAs. We use three open-source LoRAs respectively: Genshin Impact Furina, Pikachu, and Black Myth Wukong.

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Figure 21. More results of DMM combined with Mixture-of-Diffusers.

[Figure 264]

##### Figure 22. The result gird of pairwise interpolation. The interpolation weights are 0.5 and 0.5.

[Figure 265]

6 2

[Figure 266]

- 6 5

[Figure 267]

- 3 2

[Figure 268]

- 4 8

[Figure 269]

- 7 1

Figure 23. More results of interpolation between two styles. The number on the side is the model index. The weight list of one ingredient is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].

