# arXiv:2404.07987v4[cs.CV]19Nov2024

## ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback

##### Project Page: liming-ai.github.io/ControlNet_Plus_Plus

Ming Li1, Taojiannan Yang1, Huafeng Kuang2, Jie Wu2, Zhaoning Wang1, Xuefeng Xiao2, and Chen Chen1

1 Center for Research in Computer Vision, University of Central Florida 2 ByteDance

Abstract. To enhance the controllability of text-to-image diffusion models, existing efforts like ControlNet incorporated image-based conditional controls. In this paper, we reveal that existing methods still face significant challenges in generating images that align with the image conditional controls. To this end, we propose ControlNet++, a novel approach that improves controllable generation by explicitly optimizing pixel-level cycle consistency between generated images and conditional controls. Specifically, for an input conditional control, we use a pre-trained discriminative reward model to extract the corresponding condition of the generated images, and then optimize the consistency loss between the input conditional control and extracted condition. A straightforward implementation would be generating images from random noises and then calculating the consistency loss, but such an approach requires storing gradients for multiple sampling timesteps, leading to considerable time and memory costs. To address this, we introduce an efficient reward strategy that deliberately disturbs the input images by adding noise, and then uses the single-step denoised images for reward fine-tuning. This avoids the extensive costs associated with image sampling, allowing for more efficient reward fine-tuning. Extensive experiments show that ControlNet++ significantly improves controllability under various conditional controls. For example, it achieves improvements over ControlNet by 11.1% mIoU, 13.4% SSIM, and 7.6% RMSE, respectively, for segmentation mask, line-art edge, and depth conditions. All the code, models, demo and organized data have been open sourced on our Github Repo.

Keywords: Controllable Generation · Diffusion Model · ControlNet

### 1 Introduction

The emergence and improvements of diffusion models [12, 43, 50], along with the introduction of large-scale image-text datasets [48,49], has catalyzed significant strides in text-to-image generation. Nonetheless, as the proverb “an image is worth a thousand words” conveys, it’s challenging to depict an image accurately and in detail through language alone, and this dilemma also perplexes existing text-to-image diffusion models [43,46]. To this end, many studies focus on incorporating conditional controls such as segmentation mask into text-toimage diffusion models [22,30,37,62,63]. Despite the diversity in these methods, the core objective remains to facilitate more accurate and controllable image generation with explicit image-based conditional controls.

[Figure 1]

|[Figure 2]<br><br>1|[Figure 3]<br><br>[Figure 4]<br><br>2|[Figure 5]<br><br>[Figure 6]<br><br>3|
|---|---|---|
|[Figure 7]|[Figure 8]|[Figure 9]|

Less Details More Details

[Figure 10]

[Figure 11]

[Figure 12]

|[Figure 13]<br><br>1|
|---|

|[Figure 14]<br><br>2|
|---|

|[Figure 15]<br><br>3|
|---|

Line-Art Image Condition Text Prompt: Ryan Reynolds

(a) Inputs Conditions 1 2 3

(b) Ours (SSIM: 0.8714)

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

|[Figure 20]<br><br>1|[Figure 21]<br><br>[Figure 22]<br><br>2<br><br>[Figure 23]|[Figure 24]<br><br>3|
|---|---|---|
|[Figure 25]|[Figure 26]<br><br>|[Figure 27]|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

(c) T2I-Adapter-SDXL (SSIM: 0.6840)

(d) ControlNet v1.1 (SSIM: 0.7377)

- Fig. 1: (a) Given the same input image condition and text prompt, (b) the extracted conditions of our generated images are more consistent with the inputs, (c,d) while other methods fail to achieve accurate controllable generation. SSIM scores measure the similarity between all input edge conditions and the extracted edge conditions. All the line edges are extracted by the same line detection model used by ControlNet [63].

Achieving controllable image generation could involve retraining diffusion models from scratch [37,43], but this comes with high computational demands and a scarcity of large public datasets [63]. In light of this, a feasible strategy is fine-tuning pre-trained text-to-image models [23,61] or introducing trainable modules [30, 62, 63] like ControlNet [63]. However, despite these studies have explored the feasibility of controllability [30, 62, 63] in text-to-image diffusion models and expanded various applications [22,23,37], a significant gap remains in achieving precise and fine-grained control. As shown in Fig. 1, existing methods of controllable generation (e.g., ControlNet [63] and T2I-Adapter [30]) still struggle to accurately generate images that are consistent with the input image condition. For example, T2I-Adapter-SDXL consistently produced incorrect wrinkles on the forehead in all generated images, while ControlNet v1.1 introduced many wrong details. Regrettably, current efforts lack specific methods for improving controllability, which impedes progress in this research area.

To address this issue, we model image-based controllable generation as an image translation task [17] from input conditional controls to output generated images. Inspired by CycleGAN [71], we propose to employ pre-trained discriminative models to extract the condition from the generated images and directly optimize the cycle consistency loss for better controllability. The idea is that if we translate images from one domain to the other (condition cv → generated image x′0), and back again (generated image x′0 → condition c′v) we should arrive where we started (c′v = cv), as shown in Fig. 2. For example, given a segmentation mask as a conditional control, we can employ existing methods such as Control-

[Figure 34]

[Figure 35]

[Figure 36]

Generated Image Output

Condition

|[Figure 37]|[Figure 38]|
|---|---|
|[Figure 39]|[Figure 40]|

|[Figure 41]|[Figure 42]| |
|---|---|---|
|[Figure 43]|[Figure 44]| |

[Figure 45]

[Figure 46]

[Figure 47]

|Controllable Diffusions| |
|---|---|
| | |

|Pre-trained Reward Models| |
|---|---|
| | |

[Figure 48]

Hed Model Depth Model Canny Model Segmentation Model ······

Prompt: heart, mountains, and nature image

Cycle Consistency

- Fig. 2: Illustration of the cycle consistency. We first prompt the diffusion model G to generate an image x′0 based on the given image condition cv and text prompt ct, then extract the corresponding image condition cˆv from the generated image x′0 using pre-trained discriminative models D. The cycle consistency is defined as the similarity between the extracted condition cˆv and input condition cv.

Net [63] to generate corresponding images. Then the predicted segmentation masks of these generated images can be obtained by a pre-trained segmentation model. Ideally, the predicted segmentation masks and the input segmentation masks should be consistent. Hence, the cycle consistency loss can be formulated as the per-pixel classification loss between the input and predicted segmentation masks. Unlike existing related works [27, 30, 37, 63, 65] that implicitly achieve controllability by introducing conditional controls into the latent-space denoising process, our method explicitly optimizes controllability at the pixel-space for better performance, as demonstrated in Fig. 3.

To implement pixel-level loss within the context of diffusion models, an intuitive approach involves executing the diffusion model’s inference process, starting from random Gaussian noise and performing multiple sampling steps to obtain the final generated images, following recent works focusing on improving image quality with human feedback [11,36,60]. However, multiple samplings can lead to efficiency issues, and require the storage of gradients at every timestep and thus significant time and GPU memory consumption. We demonstrate that initiating sampling from random Gaussian noise is unnecessary. Instead, by directly adding noise to training images to disturb their consistency with input conditional controls and then using single-step denoised images to reconstruct the consistency, we can conduct more efficient reward fine-tuning. Our contributions are summarized as:

- – New Insight: We reveal that existing efforts in controllable generation still perform poorly in terms of controllability, with generated images significantly deviating from input conditions and lacking a clear strategy for improvement.

- – Consistency Reward Feedback: We show that pre-trained discriminative models can serve as powerful visual reward models to improve the controllability of controllable diffusion models in a cycle-consistency manner.

- – Efficient Reward Fine-tuning: We disrupt the consistency between input images and conditions, and enable the single-step denoising for efficient reward fine-tuning, avoiding time and memory overheads caused by image sampling.

- – Evaluation and Promising Results: We provide a unified and public evaluation of controllability under various conditional controls, and demonstrate that ControlNet++ comprehensively outperforms existing methods.

|Latent-space Denoising Loss<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>🔥<br><br>Encoder Diffusion<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>: A large building with a pointed roof and several chimneys.<br><br>[Figure 58]<br><br>|
|---|

|[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>🔥<br><br>Encoder Diffusion Decoder<br><br>[Figure 64]<br><br>[Figure 65]<br><br>Pixel-space Cycle Consistency Loss<br><br>|[Figure 66]<br><br>[Figure 67]| |
|---|---|
| | |
<br><br>| |[Figure 68]<br><br>[Figure 69]|
|---|---|
| | |
<br><br>|[Figure 70]| |
|---|---|
|[Figure 71]| |
<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]|
|---|

(a) Existing Methods (b) Our Solution

- Fig. 3: (a) Existing methods achieve implicit controllability by introducing image-

based conditional control cv into the denoising process of diffusion models, with the guidance of latent-space denoising loss. (b) We utilize discriminative reward models D to explicitly optimize the controllability of G via pixel-level cycle consistency loss.

### 2 Related Work

- 2.1 Diffusion-based Generative Models The diffusion probabilistic model presented in [50] has undergone substantial advancements [12,19,25], thanks to iterative refinements in training and sampling strategies [18,51,52]. To alleviate the computational demands for training diffusion models, Latent Diffusion [43] maps the pixel space diffusion process into the latent feature space. In the realm of text-to-image synthesis, diffusion models [31,35,40,41,43,46] integrate cross-attention mechanisms between UNet [44] denoisers and text embeddings from pre-trained language models like CLIP [38] and T5 [39] to facilitate reasonable text-to-image generation. Furthermore, diffusion models are employed across image editing tasks [3,14,24,29] by manipulating inputs [40], editing cross-attentions [16], and fine-tuning models [45]. Despite the astonishing capabilities of diffusion models, language is a sparse and highly semantic representation, unsuitable for describing dense, low-semantic images. Furthermore, existing methods [35,43] still struggle to understand detailed text prompts, posing a severe challenge to the controllable generation [63].
- 2.2 Controllable Text-to-Image Diffusion Models To achieve conditional control in pre-trained text-to-image diffusion models, ControlNet [63] and T2I-Adapter [30] introduce additional trainable modules for guided image generation. Furthermore, recent research employs various prompt engineering [27,61,64] and cross-attention constraints [6,23,58] for a more regulated generation. Some methods also explore multi-condition or multi-modal generation within a single diffusion model [21,37,65] or focus on the instancebased controllable generation [54,69]. However, despite these methods exploring feasibility and applications, there still lacks a clear approach to enhance controllability under various controls. Furthermore, existing works implicitly learn controllability by the denoising process of diffusion models, while our ControlNet++ achieves this in an explicit cycle-consistency manner, as shown in Fig. 3.
- 2.3 Linguistic and Visual Reward Models The reward model is trained to evaluate how well the results of generative models align with human expectations, and its quantified results will be used to

facilitate generative models for better and more controllable generation. It is usually trained with reinforcement learning from human feedback (RLHF) in NLP tasks [10,32,53], and has recently extended into the vision domain to improve the image quality for text-to-image diffusion models [1,11,13,36,56,60]. However, image quality is an exceedingly subjective metric, fraught with individual preferences, and requires the creation of new datasets with human preferences [26,55,56,60] and the training of reward models [36,55,60]. Diverging from the pursuit of global image quality with subjective human preference in current research, we target the more fine-grained and objective goal of controllability. Also, it’s more cost-effective to obtain AI feedback compared to human feedback.

### 3 Method

In this section, we first introduce the background of diffusion models in Sec. 3.1. In Sec. 3.2, we discuss how to design the cycle consistency loss for controllable diffusion models to enhance the controllability. Finally, in Sec. 3.3, we examine the efficiency issues with the straightforward solution and correspondingly propose an efficient reward strategy that utilizes the single-step denoised images for consistency loss, instead of sampling images from random noise.

#### 3.1 Preliminary

The diffusion models [18] define a Markovian chain of diffusion forward process q(xt|x0) by gradually adding noise to input data x0:

xt = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,I), (1)

where ϵ is a noise map sampled from a Gaussian distribution, and α¯t := ts=0 αs. αt = 1 − βt is a differentiable function of timestep t, which is determined by the denoising sampler such as DDPM [18]. To this end, the diffusion training loss can be represented by:

T

L(ϵθ) =

t=1

0∼q(x0),ϵ∼N(0,I) ϵθ √α¯tx0 + √1 − α¯tϵ − ϵ 22 . (2)

Ex

In the context of controllable generation [30,63], with given image condition cv and text prompt ct, the diffusion training loss at timestep t can be re-written as:

0,t,ct,cv,ϵ∼N(0,1) ∥ϵθ (xt,t,ct,cv) − ϵ∥22 . (3)

Ltrain = Ex

During the inference, given a random noise xT ∼ N(0,I), we can predict final denoised image x0 with the step-by-step denoising process [18]:

1 − αt √1 − α¯t

1 √αt

ϵθ (xt,t) + σtϵ, (4)

xt −

xt−1 =

where ϵθ refers to the predicted noise at timestep t by U-Net [44] with parameters θ, and σt = 1−α¯

1−α¯t βt is the variance of posterior Gaussian distribution pθ(x0).

t−1

#### 3.2 Reward Controllability with Consistency Feedback

As we model controllability as the consistency between input conditions and the generated images, we can naturally quantify this outcome through the discriminative reward models. Once we quantify the results of the generative model, we can perform further optimization for more controllable generation based on these quantified results in a unified manner for various conditional controls.

To be more specific, we minimize the consistency loss between the input condition cv and the corresponding output condition cˆv of the generated image x′0, as depicted in Fig. 2. The reward consistency loss can be formulated as:

Lreward = L(cv,cˆv)

= L(cv,D(x′0))

(5)

= L cv,D GT (ct,cv,xT,t) ,

where GT (ct,cv,xT,t) denotes the process that the model performs T denoising steps to generate the image x′0 from random noise xT, as shown in the Fig. 4 (a). Here, L is an abstract metric function that can take on different concrete forms for different visual conditions. For example, in the context of using segmentation mask as the input conditional control, L could be the per-pixel cross-entropy loss. The reward model D is also dependent on the condition, and we use the UperNet [57] for segmentation mask conditions. The details of loss functions and reward models are summarized in the supplementary material.

In addition to the reward loss, we also employ diffusion training loss in Eq. 3 to ensure that the original image generation capability is not compromised since they have different optimization goals. Finally, the total loss is the combination of Ltrain and Lreward:

Ltotal = Ltrain + λ · Lreward, (6)

where λ is a hyper-parameter to adjust the weight of the reward loss. Through this approach, the consistency loss can guide the diffusion model on how to sample at different timesteps to obtain images more consistent with the input controls, thereby enhancing controllability. Nonetheless, directly applying such reward consistency still poses challenges in efficiency in real-world settings.

#### 3.3 Efficient Reward Fine-tuning

To achieve the pixel-space consistency loss Lreward, it requires x0, the final diffused image, to calculate the reward consistency from the reward models. As modern diffusion models, such as Stable Diffusion [43], require multiple steps, e.g., 50 steps, to render a full image, directly using such a solution is impractical in realistic settings: (1) multiple time-consuming samplings are required to derive images from random noise. (2) to enable gradient backpropagation, we have to store gradients at each timestep, meaning the GPU memory usage will increase linearly with the number of time-steps. Taking ControlNet as an example,

|[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>…<br><br>…<br><br>[Figure 82]<br><br>|[Figure 83]||[Figure 87]|[Figure 88]|[Figure 89]|
|---|---|---|
| | | |
| | | |
|
|---|---|
| |…|
<br><br>…<br><br>|[Figure 87]|[Figure 88]|[Figure 89]|
|---|---|---|
| | | |
| | | |
<br><br>[Figure 90]<br><br>[Figure 91]<br><br>Eq. (5)<br><br>Eq. (4)<br><br>Multi-step Sampling (e.g., 50 steps)<br><br>Eq. (4) 50x Inference<br><br>Time & Memory<br><br>[Figure 92]|
|---|

||[Figure 93]|[Figure 94]|[Figure 95]|
|---|---|---|
| | | |
| | | |
<br><br>|[Figure 96]|
|---|
<br><br>|[Figure 97]|[Figure 98]|
|---|---|
| |Add Noise|
<br><br>[Figure 99]<br><br>[Figure 100]<br><br>Disturb Consistency Single-step Sampling<br><br>[Figure 101]<br><br>Eq. (8)<br><br>[Figure 102]<br><br>Eq. (1) Eq. (7) 1x Inference<br><br>[Figure 103]<br><br>Time & Memory|
|---|

(b) Efficient Reward Strategy (Ours)

(a) Default Reward Strategy

- Fig. 4: (a) Pipeline of default reward fine-tuning strategy. Reward fine-tuning requires sampling all the way to the full image. Such a method needs to keep all gradients for each timestep and the memory required is unbearable by current GPUs. (b) Pipeline

of our efficient reward strategy. We add a small noise (t ≤ tthre) to disturb the consistency between input images and conditions, then the single-step denoised image can be directly used for efficient reward fine-tuning.

when the batch size is 1 with FP16 mixed precision, the GPU memory required for a single denoising step and storing all training gradients is approximately 6.8GB. If we use the 50-step inference with the DDIM [51] scheduler, approximately 340GB of memory is needed to perform reward fine-tuning on a single sample, which is nearly impossible to achieve with current hardware capabilities. Although GPU memory consumption can be reduced by employing techniques such as Low-Rank Adaptation (LoRA) [11,20], gradient checkpointing [7,11], or stop-gradient [60], the efficiency degradation caused by the number of sampling steps required to generate images remains significant and cannot be overlooked. Therefore, an efficient reward fine-tuning approach is necessary.

In contrast to diffusing from random noise xT to obtain the final image x0, as illustrated in Fig. 4 (a), we instead propose an one-step efficient reward strategy. Specifically, instead of randomly sampling from noise, we add noise to the training images x0, thereby explicitly disturbing the consistency between the diffusion inputs x′t and their conditional controls cv, by performing diffusion forward process q(xt|x0) in Eq. 1. We demonstrate this process as the Disturb Consistency in Fig. 4 (b) , which is the same procedure as the standard diffusion training process. When the added noise ϵ is relatively small, we can predict the original image x′0 by performing single-step sampling3 on disturbed image x′t [18]:

√1 − αtϵθ (x′t,cv,ct,t) √αt

x′t −

x0 ≈ x′0 =

, (7)

and then we directly utilize the denoised image x′0 to perform reward fine-tuning:

Lreward = L(cv,cˆv) = L(cv,D(x′0)) = L(cv,D[G(ct,cv,x′t,t)]). (8)

Essentially, the process of adding noise destroys the consistency between the input image and its condition. Then the reward fine-tuning in Eq. 8 instructs the diffusion model to generate images that can reconstruct the consistency, thus enhancing its ability to follow conditions during generation.

- 3 We provide a more detailed proof in the supplementary material.

Please note that here we avoid the sampling process in Eq. 5. Finally, the loss is the combination of diffusion training loss and the reward loss:

Ltotal = Ltrain + λ · Lreward , if t ≤ tthre, Ltrain , otherwise,

(9)

where tthre denotes the timestep threshold, which is a hyper-parameter used to determine whether a noised image xt should be utilized for reward fine-tuning. We note that a small noise ϵ (i.e., a relatively small timestep t) can disturb the consistency and lead to effective reward fine-tuning. When the timestep t is large, xt is closer to the random noise xT, and predicting x′0 directly from xt results in severe image distortion. The advantage of our efficient rewarding is that xt can be employed both to train and reward the diffusion model without the need for time and GPU memory costs caused by multiple sampling, thereby significantly improving the efficiency during the reward fine-tuning stage.

During the reward fine-tuning phases, we freeze the pre-trained discriminative reward model and text-to-image model, and only update the ControlNet following its original implementation, which ensures the generative capabilities are not compromised. We also observe that using only the reward loss will lead to image distortion, aligning with the conclusions drawn in previous studies [60].

### 4 Experiments

#### 4.1 Experimental Setup

Condition Controls and Datasets. Given that existing text-image paired datasets for generative models are unable to provide accurate conditional control data pairs [48, 49], such as image-segmentation pairs, we endeavor to select specific datasets for different tasks that can offer more precise image-label data pairs. More specifically, ADE20K [67,68] and COCOStuff [4] are used for the segmentation mask condition following ControlNet [63]. For the canny edge map, hed edge map, lineart map, and depth map condition, we utilize the MultiGen-20M dataset proposed by UniControl [37], which is a subset of LAION-Aesthetics [48]. For the datasets without text caption such as ADE20K, we utilize MiniGPT-

- 4 [70] to generate the image caption with the instruction “Please briefly describe this image in one sentence”. The training and inference resolution is 512×512 for all datasets and methods. Details are provided in the supplementary material.

Evaluation and Metrics. We train ControlNet++ on the training set of each corresponding dataset and evaluate all methods on the validation dataset. All the experiments are evaluated under 512×512 resolution for fair comparison. For each condition, we evaluate the controllability by measuring the similarity between the input conditions and the extracted conditions from generated images of diffusion models. For semantic segmentation and depth map controls, we use mIoU and RMSE as evaluation metrics respectively, which is a common practice in related research fields. For the edge task, we use F1-Score for hard edges (canny

- Table 1: Controllability comparison with state-of-the-art methods under different conditional controls and datasets. ↑ denotes higher result is better, while ↓ means lower is better. ControlNet++ achieves significant controllability improvements. ‘-’ indicates that the method does not provide a public model for testing. We generate four groups of images in png format and report the average result to reduce random errors.

|Condition (Metric)|T2I Model|Seg. Mask (mIoU ↑)| |Canny Edge (F1 Score ↑)|Hed Edge (SSIM ↑)<br><br>|LineArt Edge (SSIM ↑)|Depth Map (RMSE ↓)|
|---|---|---|---|---|---|---|---|
|Dataset<br><br>| |ADE20K|COCO-Stuff<br><br>|MultiGen-20M|MultiGen-20M<br><br>|MultiGen-20M|MultiGen-20M<br><br>|
|ControlNet T2I-Adapter|SDXL SDXL<br><br>|-|-<br><br>|28.01|-<br><br>|0.6394|40.00 39.75<br><br>|
|T2I-Adapter Gligen Uni-ControlNet UniControl ControlNet Ours<br><br>|SD1.5<br><br>SD1.4<br>SD1.5 SD1.5 SD1.5 SD1.5<br>|12.61 23.78 19.39 25.44 32.55 43.64<br><br>|27.46 34.56|23.65 26.94 27.32 30.82 34.65 37.04<br><br>|0.5634 0.6910 0.7969 0.7621 0.8097<br><br>|0.7054 0.8399|48.40 38.83 40.65 39.18 35.90 28.32<br><br>|

edge) because it can be regarded as a binary classification problem of 0 (nonedge) and 1 (edge) and has a serious long-tail distribution, following the standard evaluation in edge detection [59]. The threshold used for evaluation is (100, 200) for OpenCV, and (0.1, 0.2) for Kornia implementation. The SSIM metric is used for the soft edges conditional controls (i.e., hed edge & lineart edge) following previous works [65]. For ControlNet++, we use the UniPC [66] sampler with 20 denoising steps to generate images with the original text prompt following ControlNet v1.1 [63], without any negative prompts. For other methods beyond ControlNet and ours, we utilized their open-source code to generate images and conducted fair evaluations under the same data, without changing their inference configures such as the number of inference steps or denoising sampler.

Baselines. Our evaluation primarily focuses on T2I-Adapter [30], ControlNet v1.1 [63], GLIGEN [27], Uni-ControlNet [65], and UniControl [37], as these methods are pioneering in the realm of controllable text-to-image diffusion models and offer public model weights for various image conditions. To ensure fairness of evaluation, all methods use the same image conditions and text prompts. While most methods employ the user-friendly SD1.5 as their text-to-image model for controllable generation, we have observed that recently there are a few models based on SDXL [35]. Therefore, we also report the controllability results for ControlNet-SDXL and T2I-Adapter-SDXL. Please note that ControlNet-SDXL mentioned here is not an officially released model as in ControlNet [63].

#### 4.2 Experimental Results

Comparison of Controllability with State-of-the-art Methods. The experimental results are shown in Tab. 1, which can be summarized as the following observations: (1) Existing methods still underperform in terms of controllability, struggling to achieve precise controlled generation. For instance, current methods (i.e., ControlNet) achieve only a 32.55 mIoU for images generated under the condition of segmentation masks, which is far from its performance on real datasets with a 50.7 mIoU, under the same evaluation from Mask2Former segmentation model [8]. (2) Our ControlNet++ significantly outperforms existing

- Table 2: FID (↓) comparison with state-of-the-art methods under different conditional controls and datasets. All the results are conducted on 512×512 image resolution with Clean-FID implementation [33] for fair comparisons. ‘-’ indicates that the method does not provide a public model for testing. We generate four groups of images in png format and report the average result to reduce random errors.

|Method<br><br>|T2I Model|Seg. Mask| |Canny Edge|Hed Edge<br><br>|LineArt Edge<br><br>|Depth Map|
|---|---|---|---|---|---|---|---|
| | |ADE20K|COCO<br><br>|MultiGen-20M|MultiGen-20M<br><br>|MultiGen-20M<br><br>|MultiGen-20M|
|Gligen T2I-Adapter UniControlNet UniControl ControlNet Ours|SD1.4<br><br>SD1.5<br><br><br>SD1.5 SD1.5 SD1.5 SD1.5<br><br>|33.02 39.15 39.70 46.34 33.28 29.49<br><br>|21.33 19.29|18.89 15.96 17.14 19.94 14.73 18.23<br><br>|17.08 15.99 15.41 15.01<br><br>|17.44 13.88<br><br>|18.36 22.52 20.27 18.66 17.76 16.66|

- Table 3: CLIP-score (↑) comparison with state-of-the-art methods under different conditional controls and datasets. ‘-’ indicates that the method does not provide a public model for testing. We generate four groups of images in png format and report the average result to reduce random errors.

|Method<br><br>|T2I Model|Seg. Mask| |Canny Edge|Hed Edge<br><br>|LineArt Edge<br><br>|Depth Map|
|---|---|---|---|---|---|---|---|
| | |ADE20K|COCO<br><br>|MultiGen-20M<br><br>|MultiGen-20M<br><br>|MultiGen-20M|MultiGen-20M|
|Gligen T2I-Adapter UniControlNet UniControl ControlNet Ours<br><br>|SD1.4<br>SD1.5 SD1.5 SD1.5 SD1.5 SD1.5<br>|31.12 30.65 30.59 30.92 31.53 31.96<br><br>|13.31 13.13|31.77 31.71 31.84 31.97 32.15 31.87<br><br>|31.94 32.02 32.33 32.05|32.46 31.95<br><br>|31.75 31.46 31.66 32.45 32.45 32.09|

works in terms of controllability across various conditional controls. For example, it achieves 11.1% RMSE improvements against previous state-of-the-art methods for the depth map condition; (3) For controllable diffusion models, the strength of the text-to-image backbone does not affect its controllability. As shown in the table, although SDXL-based [35] ControlNet and T2I-Adapter have better controllability on some specific tasks, the improvement is not large and is not significantly better than the counterparts with SD 1.5 [43].

Comparison of Image Quality with State-of-the-art Methods. To verify whether improving controllability leads to a decline in image quality, we reported the FID (Fréchet Inception Distance) metrics of different methods under various conditional generation tasks in Tab. 2. We discovered that, compared to existing methods, ControlNet++ generally exhibits superior FID values in most cases, indicating that our approach, while enhancing the controllability of conditional controls, does not result in a decrease in image quality. This can also be observed in Fig. 6. We provide more visual examples in the supplementary material.

Comparison of CLIP score with State-of-the-art Methods. Our ControlNet++ aims to improve the controllability of diffusion models using image-based conditions. Concerned about the potential adverse effects on text controllability, we evaluated various methods using CLIP-Score metrics across different datasets to measure the similarity between generated images and input text. As indicated in Tab. 3, ControlNet++ achieved comparable or superior CLIP-Score outcomes on several datasets relative to existing approaches. This suggests that our method not only markedly enhances conditional controllability but also preserves the original model’s text-to-image generation proficiency.

###### Generated Images Ground Truth Images GT + Generated Images

- Fig. 5: Training DeepLabv3 (MobileNetv2) from scratch with different images, including ground truth images from ADE20K, and the generated images from ControlNet and ours. All the labels (i.e., segmentation masks) are ground truth labels in ADE20K. Please note improvements here are non-trivial for semantic segmentation.

Effectiveness of Generated Images. To further validate our improvements in controllability and their impact, we use the generated images along with real human-annotated labels to create a new dataset for training discriminative models from scratch. Please note that the only difference from the original dataset used to train the discriminative model is that we have replaced the images with those generated by the controllable diffusion model while keeping the labels unchanged. If the generative model exhibits good controllability, the quality of the constructed dataset will be higher, thereby enabling to train a stronger model.

Specifically, we conduct experiments on the ADE20K [67, 68] dataset on DeepLabv3 with MobileNetv2 backbone [5]. We use the standard training dataset (20210 training samples) to train the discriminative model and the validation dataset (5000 evaluation samples) for evaluation. We show the experimental results in Fig. 5, the segmentation model trained on our images outperforms the baseline results (ControlNet) by 1.19 mIoU. Please note that this improvement is significant in segmentation tasks. For instance, Mask2Former [8] improves previous SOTA MaskFormer [9] with around 1.1 mIoU in semantic segmentation. In addition to conducting experiments solely on the generated dataset, we also combined generated data with real data to train the segmentation model. The experimental results indicate that augmenting real ground truth data with data generated by ControlNet does not yield additional performance improvements (34.11 v.s. 34.08). In contrast, augmenting real data with our generated data results in significant performance enhancements (+1.76 mIoU).

Qualitative Comparison. Figs. 6 and 7 provide a qualitative comparison between our ControlNet++ and previous state-of-the-art methods across different conditional controls. When given the same input text prompts and image-based conditional controls, we observe that existing methods often generate areas

Image & Condition Ours Uni-ControlNet UniControl ControlNet T2I-Adapter

Gligen

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

SegmentationDepth

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

[Figure 129]

[Figure 130]

[Figure 131]

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

Canny

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Image & Condition Ours Uni-ControlNet UniControl Gligen ControlNet

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

Hed

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

###### Fig. 6: Visualization comparison results in different conditional controls.

T2I-Adapter

ControlNet

Image & Condition Ours

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

inconsistent with the image conditions. For instance, in the segmentation mask generation task, other methods often produce extraneous picture frames on the walls, resulting in a mismatch between the segmentation masks extracted from the generated image and the inputs. A similar situation occurs under depth conditions, where other methods fail to accurately represent the depth of different fingers. In contrast, images generated by ControlNet++ maintain good consistency with the input depth map.

Line-Art

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

Fig. 7: Comparison on Line-Art Edge.

#### 4.3 Ablation Study

Loss Settings. In Fig. 8, we find that maintaining the original diffusion training process is crucial for preserving the quality and controllability of generated images. Relying solely on pixel-level consistency loss leads to severe image distortion, whereas training the model with both this loss and the diffusion training loss can enhance controllability without affecting image quality.

50 Steps 100 Steps 200 Steps 400 Steps 50 Steps 100 Steps 200 Steps 400 Steps

|[Figure 166]|
|---|
|[Figure 167]|

|[Figure 168]|
|---|
|[Figure 169]|

|[Figure 170]|
|---|
|[Figure 171]|

|[Figure 172]|
|---|
|[Figure 173]|

|[Figure 174]|
|---|
|[Figure 175]|

|[Figure 176]|
|---|
|[Figure 177]|

|[Figure 178]|
|---|
|[Figure 179]|

|[Figure 180]|
|---|
|[Figure 181]|

|[Figure 182]|
|---|

###### Input Condition

Prompt: A pelican gracefully takes off from the calm water.

Without Diffusion Training Loss With Diffusion Training Loss

- Fig. 8: Ablation study on different loss settings during training. Using only pixel-level consistency loss leads to severe image distortion, affecting both image quality and controllability. However, when combined with diffusion training loss, it is possible to gradually improve controllability without compromising image quality.

Generalizability of Efficient Reward Fine-tuning. Although the reward finetuning is used in a small subset of timesteps, it updates all the parameters of the ControlNet and therefore helps more timesteps to improve controllability during sampling. To prove this, we divide the sampling process into two parts: the unoptimized timesteps [T,tthre], and the optimized timesteps [tthre,1] and use ControlNet and our model for inference crossly, with 20-step sampling following ControlNet. Table 4 shows that our reward finetuning performed on a small number of timesteps [tthre,1] can be generalized to larger timesteps [T,tthre].

- Table 4: The impact of efficient reward fine-tuning on different timesteps.

Table 5: Stronger reward model (UperNetR50) leads to better controllability than the weaker reward model (DeepLabv3-MBv2).

|Unoptimized [T, tthre]|Optimized [tthre, 1]|ADE20K mIoU (↑)|
|---|---|---|
|ControlNet|ControlNet<br><br>|32.55|
|ControlNet<br><br>|Ours|38.03|
|Ours<br><br>|ControlNet<br><br>|41.46|
|Ours|Ours<br><br>|43.64|

Reward Model (RM) RM mIoU↑ Eval mIoU↑

- - 32.55 DeepLabv3-MBv2 34.02 31.96

FCN-R101 39.91 40.44 UperNet-R50 42.05 43.64

Choice of Different Reward Models. We demonstrate the effectiveness of different reward models in Tab. 5, all the evaluation results (i.e., Eval mIoU in the table) are evaluated by the most powerful segmentation model Mask2Former [8] with 56.01 mIoU, on ADE20K dataset. We experiment with three different reward models, including DeepLabv3 [5] with MobileNetv2 [47] backbone (DeepLabv3MBv2), FCN [28] with ResNet-101 [15] backbone (FCN-R101) and UperNet [57] with ResNet-50 backbone. The results demonstrate that a more powerful reward model leads to better controllability for controllable diffusion models.

- 5 Discussion How to make Hed/LineArt Edge extraction methods differentiable? The Hed and LineArt Edge extraction models are neural networks without non-differentiable operations. Differentiability can be achieved by simply modifying forward code. Some conditions (e.g., Box/Sketch/Pose) are not available. Our reward finetuning leverages a pre-trained ControlNet and a differentiable reward model. Currently, pre-trained ControlNet for object bounding boxes and differentiable reward models for sketches are lacking. In existing pose models, there are nondifferentiable operations such as the NMS and keypoints grouping. We leave the question of how to extend consistency reward to more conditions to future work. Influence of Text Prompt. We discuss how different types of text prompts (No Prompt, Conflicting Prompt, and Perfect Prompt) affect the final results. As shown in Fig. 9, when the text prompt is empty or there is a semantic conflict with the image conditional control, ControlNet often struggles to generate accurate content. In contrast, our ControlNet++ manages to generate images that comply with the input conditional controls under various text prompt scenarios.

No Prompt Conflict Prompt

“delicious cake”

Perfect Prompt

“a house, high-quality, extremely detailed, 4K”

|[Figure 183]<br><br>[Figure 184]| |[Figure 185]|
|---|---|---|
| | | |
|[Figure 186]| |[Figure 187]|
| | | |

|[Figure 188]|[Figure 189]|
|---|---|
|[Figure 190]|[Figure 191]|

|[Figure 192]|[Figure 193]|
|---|---|
|[Figure 194]|[Figure 195]|

OursControlNet

- Fig. 9: When the input text prompt is empty or conflicts with the image-based conditional controls (the segmentation map in the top left corner), ControlNet struggles to generate correct content (red boxes), whereas our method manages to generate it well.

- 6 Conclusion

In this paper, we demonstrate from both quantitative and qualitative perspectives that existing works focusing on controllable generation still fail to achieve precise conditional control, leading to inconsistency between generated images and input conditions. To address this issue, we introduce ControlNet++, it explicitly optimizes the consistency between input conditions and generated images using a pre-trained discriminative reward model in a cycle consistency manner, which is different from existing methods that implicitly achieve controllability through latent diffusion denoising. We also propose a novel and efficient reward strategy that calculates consistency loss by adding noise to input images followed by single-step denoising, thus avoiding the significant computational and memory costs associated with sampling from random Gaussian noise. Experimental results under multiple conditional controls show that ControlNet++ significantly improves controllability without compromising image quality and image-text alignment, offering new insights into controllable visual generation.

## ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback Supplementary Material

Ming Li1, Taojiannan Yang1, Huafeng Kuang2, Jie Wu2, Zhaoning Wang1, Xuefeng Xiao2, and Chen Chen1

1 Center for Research in Computer Vision, University of Central Florida 2 ByteDance

### 1 Overview of Supplementary

The supplementary material is organized into the following sections:

- – Section 2: Implementation details for all experiments.
- – Section 3: Proof for Eq.(7) in the main paper.
- – Section 4: More experiments and analysis.

- • Section 4.1: Effectiveness of conditioning scale of existing methods.
- • Section 4.2: Human evaluation on controllability, text guidance and image quaility.

- – Section 5: Discussion of broader impact and limitation.
- – Section 6: More visualization results.

### 2 Implementation Details

#### 2.1 Dataset Details

Considering that the training data for ControlNet [63] has not been publicly released, we need to construct our training dataset. In this paper, we adhere to the dataset construction principles of ControlNet [63], which endeavor to select datasets with more accurate conditional conditions wherever possible. Specifically, for the segmentation condition, previous works have provided datasets with accurately labeled segmentation masks [4,67,68]. Therefore, we opt to train our model using these accurately labeled datasets following ControlNet [63]. For the Hed, LineArt edge tasks, it is challenging to find datasets with real and accurate annotations. As a result, following ControlNet [63], we train the model using the MultiGen20M dataset [37], which is annotated by models, to address this issue. Regarding the depth task, existing datasets include masks of certain pixels

- as having unknown depth values, making them incompatible with the current ControlNet pipeline. Therefore, we also adapt the MultiGen20M depth dataset, which is similar to the dataset constructed by ControlNet [63]. In terms of the canny edge task, no human labels are required in the process, so we also adapt the MultiGen20M dataset. We provide details of the datasets in Table 1.

- Table 1: Dataset and evaluation details of different conditional controls. ↑ denotes higher is better, while ↓ means lower is better.

| |Segmentation Mask<br><br>|Canny Edge<br><br>|Hed Edge<br><br>|LineArt Edge|Depth Map|
|---|---|---|---|---|---|
|Dataset|ADE20K [67,68], COCOStuff [4]<br><br>|MultiGen20M [37]|MultiGen20M [37]<br><br>|MultiGen20M [37]|MultiGen20M [37]|
|Training Samples|20,210 & 118,287<br><br>|2,560,000|2,560,000<br><br>|2,560,000<br><br>|2,560,000|
|Evaluation Samples|2,000 & 5,000<br><br>|5,000|5,000<br><br>|5,000<br><br>|5,000|
|Evaluation Metric|mIoU ↑|F1 Score ↑|SSIM ↑<br><br>|SSIM ↑<br><br>|RMSE ↓|

- Table 2: Details of the reward model, evaluation model, and training loss under different conditional controls. ControlNet* denotes we use the same model to extract conditions as ControlNet [63]

| |Seg. Mask|Depth Edge<br><br>|Canny Edge<br><br>|Hed Edge<br><br>|LineArt Edge|
|---|---|---|---|---|---|
|Reward Model (RM)|UperNet-R50|DPT-Hybrid|Kornia Canny<br><br>|ControlNet*|ControlNet*|
|RM Performance<br><br>|ADE20K(mIoU): 42.05|NYU(AbsRel): 8.69|-<br><br>|-|-<br><br>|
|Evaluation Model (EM)|Mask2Former|DPT-Large|Kornia Canny<br><br>|ControlNet*|ControlNet*|
|EM Performance|ADE20K(mIoU): 56.01|NYU(AbsRel): 8.32<br><br>|-<br><br>|-<br><br>|-|
|Consistency Loss<br><br>|CrossEntropy Loss|MSE Loss<br><br>|MSE Loss<br><br>|MSE Loss|MSE Loss<br><br>|
|Loss Weight λ<br><br>|0.5|0.5<br><br>|1.0<br><br>|1.0|10|

.

#### 2.2 Reward Model and Evaluation Details

In general, we deliberately choose slightly weaker models as the reward model and opt for stronger models for evaluation. This practice not only ensures the fairness of the evaluation but also helps to determine whether performance improvements result from alignment with the reward model’s preferences or from a genuine enhancement in controllability. While such an approach is feasible for some tasks (Segmentation, Depth), it becomes challenging to implement for others (Hed, Canny, LineArt Edge) due to the difficulty in finding two distinct reward models. In such cases, we use the same model as both the reward model and the evaluation model. We utilize standard evaluation schemes from their respective research fields to evaluate the input conditions and extracted conditions from the generated images, as demonstrated in Section 4.1 of the main paper. We use the same Hed edge detection model and LineArt edge detection model as ControlNet [63]. We provide details of reward models and evaluation in Table 2.

#### 2.3 Training Details

The loss weight λ for reward consistency loss is different for each condition. Specifically, λ is 0.5, 0.5, 1.0, 1.0, and 10 for segmentation mask, depth, hed edge, canny edge, and LineArt edge condition, respectively. For all experiments, we first fine-tune the pre-trained ControlNet until convergence using a batch size of 256 and a learning rate of 1e-5. We then employ the same batch size and learning rate for 10k iterations reward fine-tuning. To this end, the valid training samples for reward fine-tuning is 256 × 10,000 = 2,560,000. We set threshold tthre = 200 of Eq.8 in the main paper for all experiments. Diverging from existing methods that use OpenCV’s [2] implementation of the canny algorithm, we have adopted Kornia’s [42] implementation to make it differentiable. Our codebase is based on the implementation in HuggingFace’s Diffusers [34], and we do not use classifier-free guidance during the reward fine-tuning process following diffusers.

Image Condition 1000 900 800 700 600 500 400 300 200 100

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Predicted at different timestep

- Fig. 1: Illustration of predicted image x′0 at different timesteps t. A small timestep t (i.e., small noise ϵt) leads to more precise estimation x′0 ≈ x0.

### 3 Proof of Equation 7 in the Main Paper

The diffusion models define a Markovian chain of diffusion forward process q(xt|x0) by gradually adding noise to input data x0:

xt = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,I), (1)

at any timestep t we can use the predicted ϵ(x′t,cv,ct,t − 1) to estimate the real noise ϵ in Eq. 1, and the above equation can be transformed through straightforward algebraic manipulation to the following form:

√α¯tx0 + √1 − α¯tϵθ (x′t,cv,ct,t − 1), x0 ≈ x′0 =

xt ≈

√1 − αtϵθ (x′t,cv,ct,t − 1) √αt

(2)

x′t −

.

To this end, we can obtain the predicted original image x′0 at any denoising timestep t and use it as the input for reward consistency loss. However, previous work demonstrates that this approximation only yields a smaller error when the time step t is relatively small [18]. Here we find similar results as shown in Figure 1, which illustrates the predicted x′0 is significantly different at different timesteps. We kindly encourage readers to refer to Section 4.3 and Figure 5 in the DDPM [18] paper for more experimental results.

### 4 More Experiments

In this section, we provide additional supplements to the experiments discussed in the main paper, including human evaluation on generated data samples on the Segmentation Mask condition in Sec. 4.2, analysis on conditioning scale of existing methods such as ControlNet [63] and T2I-Adapter [30] in Sec. 4.1.

Image & Condition Control Scale: 0.5 Control Scale: 1.0 Control Scale: 2.0 Control Scale: 3.0 Control Scale: 4.0 Control Scale: 10.0

18

a

|Li et<br><br>[Figure 203]|
|---|
|[Figure 204]|

|al.<br><br>[Figure 205]|[Figure 206]|[Figure 207]|[Figure 208]|[Figure 209]|[Figure 210]|
|---|---|---|---|---|---|
||[Figure 211]|
|---|
||[Figure 212]|
|---|
||[Figure 213]|
|---|
|[Figure 214]|[Figure 215]|[Figure 216]|

Generated

Images Extracted

ControlNet

Conditions Extracted

|[Figure 217]|
|---|
|[Figure 218]|

|[Figure 219]|[Figure 220]|[Figure 221]|[Figure 222]|[Figure 223]|[Figure 224]|
|---|---|---|---|---|---|
|[Figure 225]|[Figure 226]|[Figure 227]|[Figure 228]|[Figure 229]|[Figure 230]|

Conditions Generated

Images

T2I-Adapter-SDXL

- Fig. 2: Naively increasing the weight of image condition embedding compared to text condition embedding in exiting methods (i.e., ControlNet and T2I-Adapter) cannot improve controllability while ensuring image quality. The red boxes in the figures highlight areas where the generated image is inconsistent with the input conditions. Please note that we employ the same line detection model to extract conditions from images.

#### 4.1 Effectiveness of Conditioning Scale

To simultaneously achieve control based on text prompts and image conditions, existing controllable generation methods perform an addition operation between the image condition features and the text embedding features. The strength of different conditions can be adjusted through a weight value. Hence, an obvious question arises: can better controllability be achieved by increasing the weight of the image condition features? To answer this question, we conduct experiments under different control scales (The weight of image condition feature) in Figure 2. It demonstrates that naively increasing the control ratio of image conditions does not enhance controllability and may lead to severe image distortion.

#### 4.2 Human Evaluation

Following ControlNet, we use a single condition for human evaluation. We ask 20 users (12 in ControlNet paper) to select the best image based on three distinct criteria as shown in Table 3. Our ControlNet++ offers better controllability without sacrificing image quality or text guidance.

Table 3: Win rate on ADE20K validation dataset (Segmentation).

|20 annotators in total<br><br>|Ours|ControlNet|T2I-Adapter<br><br>|UniControl|
|---|---|---|---|---|
|Image-Mask Alignment|76.8%<br><br>|16.7%<br><br>|2.0%|4.5%<br><br>|
|Image Quality<br><br>|26.1%|25.8 %|23.6%|24.5 %|
|Image-Text Alignment<br><br>|25.3%|25.1%<br><br>|24.9%|24.7%|

### 5 Broader Impact and Limitation

In this paper, we use visual discriminative models to evaluate and improve the controllability of text-to-image models. However, we also realize that this work is still insufficient and discuss the following issues:

Conditions Expansion: While we have achieved notable improvements under six control conditions, our future work aims to broaden the scope by incorporating additional control conditions such as Human Pose and Scribbles. Ultimately, our objective is to control everything.

Beyond Controllability: While our current focus lies predominantly on controllability, we acknowledge the significance of quality and aesthetic appeal in the generated outputs. To address this, we plan to leverage human feedback to annotate controllability images. Subsequently, we will optimize the controllability model to simultaneously enhance both controllability and aesthetics.

Joint Optimization: To further enhance the overall performance, we intend to employ a larger set of controllable images for joint optimization of the control network and reward model. This holistic approach would facilitate their co-evolution, leading to further improvements in the final generated outputs. Through our research, we aspire to provide insightful contributions to controllability in text-to-image diffusion models. We hope that our work inspires and encourages more researchers to delve into this fascinating area.

Discussion on the necessity of controllability: Controllability is important since it allows users to modify image conditions to achieve more flexible and accurate generation. Take LineArt Edge as an example: (1) Freely generating in foreground may change the appearance (e.g., adding a beard for women) that we usually do not expect. (2) Freely generating in background will damage some applications (e.g., blur background). (3) Global free generating may destroy the overall artistic effect of the input image, such as lighting, composition, contrast, etc. Furthermore, we show in Fig.5 of the main paper that more controllable diffusion can in return improve the performance of discriminative models. Beyond image generation, the controllable conditional generation can also be combined with ID preserving methods to perform controllable image editing.

### 6 More Visualization

More visualization results across different conditional controls for our image generation are shown in Figures 3,4,5,6,7.

###### Image & Condition Generated Images & Extracted Conditions

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

- Fig. 3: More visualization results of our ControlNet++ (LineArt Edge)

###### Image & Condition Generated Images & Extracted Conditions

|[Figure 235]|
|---|
|[Figure 236]|
|[Figure 237]|
|[Figure 238]|

- Fig. 4: More visualization results of our ControlNet++ (Depth Map)

###### Image & Condition Generated Images & Extracted Conditions

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

- Fig. 5: More visualization results of our ControlNet++ (Hed Edge)

###### Image & Condition Generated Images & Extracted Conditions

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

- Fig. 6: More visualization results of our ControlNet++ (Canny Edge)

###### Image & Condition Generated Images & Extracted Conditions

|[Figure 247]|
|---|
|[Figure 248]|
|[Figure 249]|
|[Figure 250]|

Fig. 7: More visualization results of our ControlNet++ (Segmentation Mask)

### References

- 1. Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023)
- 2. Bradski, G.: The OpenCV Library. Dr. Dobb’s Journal of Software Tools (2000)
- 3. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023)
- 4. Caesar, H., Uijlings, J., Ferrari, V.: Coco-stuff: Thing and stuff classes in context. In: CVPR (2018)
- 5. Chen, L.C., Papandreou, G., Schroff, F., Adam, H.: Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587 (2017)
- 6. Chen, M., Laina, I., Vedaldi, A.: Training-free layout control with cross-attention guidance. arXiv preprint arXiv:2304.03373 (2023)
- 7. Chen, T., Xu, B., Zhang, C., Guestrin, C.: Training deep nets with sublinear memory cost. arXiv (2016)
- 8. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: CVPR (2022)
- 9. Cheng, B., Schwing, A., Kirillov, A.: Per-pixel classification is not all you need for semantic segmentation. NeurIPS (2021)
- 10. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., et al.: Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022)
- 11. Clark, K., Vicol, P., Swersky, K., Fleet, D.J.: Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400 (2023)
- 12. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. NeurIPS

(2021)

- 13. Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. NeurIPS (2023)
- 14. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. In: ICLR (2023)
- 15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016)
- 16. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-or, D.: Prompt-to-prompt image editing with cross-attention control. In: ICLR (2023)
- 17. Hertzmann, A., Jacobs, C.E., Oliver, N., Curless, B., Salesin, D.H.: Image analogies. In: SIGGRAPH (2001)
- 18. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. NeurIPS

(2020)

- 19. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)
- 20. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: ICLR (2022)
- 21. Hu, M., Zheng, J., Liu, D., Zheng, C., Wang, C., Tao, D., Cham, T.J.: Cocktail: Mixing multi-modality controls for text-conditional image generation. NeurIPS

(2023)

- 22. Huang, L., Chen, D., Liu, Y., Shen, Y., Zhao, D., Zhou, J.: Composer: Creative and controllable image synthesis with composable conditions. In: ICML (2015)

- 23. Ju, X., Zeng, A., Zhao, C., Wang, J., Zhang, L., Xu, Q.: Humansd: A native skeleton-guided diffusion model for human image generation. In: ICCV (2023)
- 24. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with diffusion models. In: CVPR (2023)
- 25. Kingma, D., Salimans, T., Poole, B., Ho, J.: Variational diffusion models. NeurIPS

(2021)

- 26. Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569 (2023)
- 27. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Open-set grounded text-to-image generation. In: CVPR (2023)
- 28. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: CVPR (2015)
- 29. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic differential equations. In: ICLR (2022)
- 30. Mou, C., Wang, X., Xie, L., Zhang, J., Qi, Z., Shan, Y., Qie, X.: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453 (2023)
- 31. Nichol, A.Q., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., Mcgrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In: ICML (2022)
- 32. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. NeurIPS (2022)
- 33. Parmar, G., Zhang, R., Zhu, J.Y.: On aliased resizing and surprising subtleties in gan evaluation. In: CVPR (2022)
- 34. von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., Wolf, T.: Diffusers: State-of-the-art diffusion models. https:// github.com/huggingface/diffusers (2022)
- 35. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 36. Prabhudesai, M., Goyal, A., Pathak, D., Fragkiadaki, K.: Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739

(2023)

- 37. Qin, C., Zhang, S., Yu, N., Feng, Y., Yang, X., Zhou, Y., Wang, H., Niebles, J.C., Xiong, C., Savarese, S., et al.: Unicontrol: A unified diffusion model for controllable visual generation in the wild. NeurIPS (2023)
- 38. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)
- 39. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR (2020)
- 40. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125

(2022)

- 41. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation. In: ICML (2021)
- 42. Riba, E., Mishkin, D., Ponsa, D., Rublee, E., Bradski, G.: Kornia: an open source differentiable computer vision library for pytorch. In: CVPR (2020)

- 43. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022)
- 44. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI (2015)
- 45. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: CVPR (2023)
- 46. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. NeurIPS (2022)
- 47. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: CVPR (2018)
- 48. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S., Crowson, K., Schmidt, L., Kaczmarczyk, R., Jitsev, J.: Laion-5b: An open large-scale dataset for training next generation image-text models. ArXiv (2022)
- 49. Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., Komatsuzaki, A.: Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. ArXiv (2021)
- 50. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: ICML (2015)
- 51. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: ICLR (2021)
- 52. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. In: ICLR

(2021)

- 53. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)
- 54. Wang, X., Darrell, T., Rambhatla, S.S., Girdhar, R., Misra, I.: Instancediffusion: Instance-level control for image generation (2024)
- 55. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023)
- 56. Wu, X., Sun, K., Zhu, F., Zhao, R., Li, H.: Better aligning text-to-image models with human preference. In: ICCV (2023)
- 57. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: ECCV (2018)
- 58. Xie, J., Li, Y., Huang, Y., Liu, H., Zhang, W., Zheng, Y., Shou, M.Z.: Boxdiff: Textto-image synthesis with training-free box-constrained diffusion. In: ICCV (2023)
- 59. Xie, S., Tu, Z.: Holistically-nested edge detection. In: ICCV (2015)
- 60. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. NeurIPS (2023)
- 61. Yang, Z., Wang, J., Gan, Z., Li, L., Lin, K., Wu, C., Duan, N., Liu, Z., Liu, C., Zeng, M., et al.: Reco: Region-controlled text-to-image generation. In: CVPR (2023)
- 62. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023)
- 63. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: ICCV (2023)

- 64. Zhang, T., Zhang, Y., Vineet, V., Joshi, N., Wang, X.: Controllable text-to-image generation with gpt-4. arXiv preprint arXiv:2305.18583 (2023)
- 65. Zhao, S., Chen, D., Chen, Y.C., Bao, J., Hao, S., Yuan, L., Wong, K.Y.K.: Unicontrolnet: All-in-one control to text-to-image diffusion models. NeurIPS (2023)
- 66. Zhao, W., Bai, L., Rao, Y., Zhou, J., Lu, J.: Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. arXiv preprint arXiv:2302.04867

(2023)

- 67. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ade20k dataset. In: CVPR (2017)
- 68. Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ade20k dataset. IJCV (2019)
- 69. Zhou, D., Li, Y., Ma, F., Yang, Z., Yang, Y.: Migc: Multi-instance generation controller for text-to-image synthesis. In: CVPR (2024)
- 70. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)
- 71. Zhu, J.Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using cycle-consistent adversarial networks. In: ICCV (2017)

