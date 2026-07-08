# arXiv:2503.19462v1[cs.CV]25Mar2025

## AccVideo: Accelerating Video Diffusion Model with Synthetic Dataset

Haiyu Zhang1,2*, Xinyuan Chen2, Yaohui Wang2, Xihui Liu3, Yunhong Wang1, Yu Qiao2† 1Beihang University 2Shanghai AI Laboratory 3The University of Hong Kong

1{zhyzhy,yhwang}@buaa.edu.cn 2{chenxinyuan,wangyaohui,qiaoyu}@pjlab.org.cn 3xihuiliu@eee.hku.hk https://aejion.github.io/accvideo

…womanwalks down a Tokyo street … wears a blackleather jacket…red dress, andblack boot…pedestrians walk about.

[Figure 1]

HunyuanVideo 3234s

|×8.5Faster|
|---|

[Figure 2]

A litter ofgolden retriever puppiesplaying in the snow. Their heads pop out of thesnow, covered in.

Figure 1. Video diffusion models can generate high-quality videos, but they require dozens of inference steps, resulting in slow generation process. For instance, HunyuanVideo [23] takes 3234 seconds to generate a 5-seconds, 720×1280, 24fps video on a single A100 GPU. In contrast, our method accelerates video diffusion models through distillation, achieving 8.5× improvements in generation speed while maintaining comparable quality.

### Abstract

Diffusion models have achieved remarkable progress in the field of video generation. However, their iterative denoising nature requires a large number of inference steps to generate a video, which is slow and computationally expen-

*Work done when Haiyu Zhang interned at Shanghai AI Laboratory. †Corresponding author

sive. In this paper, we begin with a detailed analysis of the challenges present in existing diffusion distillation methods and propose a novel efficient method, namely AccVideo, to reduce the inference steps for accelerating video diffusion models with synthetic dataset. We leverage the pretrained video diffusion model to generate multiple valid denoising trajectories as our synthetic dataset, which eliminates the use of useless data points during distillation. Based on

the synthetic dataset, we design a trajectory-based few-step guidance that utilizes key data points from the denoising trajectories to learn the noise-to-video mapping, enabling video generation in fewer steps. Furthermore, since the synthetic dataset captures the data distribution at each diffusion timestep, we introduce an adversarial training strategy to align the output distribution of the student model with that of our synthetic dataset, thereby enhancing the video quality. Extensive experiments demonstrate that our model achieves 8.5× improvements in generation speed compared to the teacher model while maintaining comparable performance. Compared to previous accelerating methods, our approach is capable of generating videos with higher quality and resolution, i.e., 5-seconds, 720×1280, 24fps.

### 1. Introduction

Video generation has garnered significant attention due to its ability to simulate the real physical world [1, 41], as well as its promising applications in entertainment, such as content creation [3, 4, 11, 20, 23, 38, 53, 61, 65], filmmaking [24, 41], video games [50, 60], and customized media generation [18, 24, 27, 39].

With advancements in data curation pipeline [1, 23] and scalable model architectures [42], diffusion models [15, 21] and flow matching [31, 32] have emerged as widely used frameworks in video generation, owing to their impressive generative capabilities. However, video diffusion models require iterative denoising of Gaussian noise to generate the final videos, making them typically demand dozens of inference steps. This process is both time-consuming and computationally intensive. For instance, HunyuanVideo [23] requires 3234s to generate a 5s, 720×1280, 24fps video on a single NVIDIA A100 GPU, as shown in Fig. 1.

Recently, significant progress has been made in the field of accelerating image diffusion models by distillation [2, 9, 29, 37, 46–48, 54, 58, 59, 62, 63]. However, distilling video diffusion models remains a challenge that requires further exploration. Although distillation methods for image diffusion models can be extended to video diffusion models, they require a large amount of data and computational resources [30, 64] due to the useless data points, which do not lie on the denoising trajectories of the teacher model. Moreover, the useless data points cause the teacher model to offer unreliable guidance for the student model, adversely affecting the video quality. The spatial-temporal complexity of videos exacerbates these issues.

To be concrete, the useless data points during distillation are caused by dataset mismatching or Gaussian noise mismatching. Dataset mismatching refers to the inconsistency between the dataset used for training the teacher model and the dataset utilized during the distillation process, which arises due to the difficulty in accessing the training dataset

of the teacher model. Gaussian noise mismatching refers to the misalignment between Gaussian noise and the data, a phenomenon that occurs during the forward diffusion operation, also called the flow operation.

In this paper, we begin with a detailed analysis of the challenges present in existing diffusion distillation models. Based on the analysis, we try to avoid the use of useless data points during the distillation process and propose a novel efficient distillation method, namely AccVideo, which aims to accelerate video diffusion models with synthetic dataset. Specifically, we first present a synthetic dataset, SynVid, comprising 110K denoising trajectories and high-quality videos generated by HunyuanVideo [23] with fine-grained text prompts. The data points on the denoising trajectories are intermediate results leading to the correct output, making them all valid and meaningful. Then, we design a trajectory-based few-step guidance that selects a few of data points from each denoising trajectory to construct a shorter noise-to-video mapping path, enabling the student model to generate videos in fewer steps. To further exploit the data distribution captured by our synthetic dataset, we propose an adversarial training strategy to align the output distribution of the student model with that of our synthetic dataset at each diffusion timestep, thereby enhancing the quality of generated videos. It is noteworthy that the data points used in our method deliver precise guidance for the student model, markedly enhancing training efficiency and reducing the number of data. Our model is trained using only 8 A100 GPUs with 38.4K synthetic data for 12 days, yet it is capable of generating high-quality 5-seconds, 720×1280, 24fps videos. Moreover, we obviate the complex regularization designs for the adversarial training [30] by combining the trajectory-based few-step guidance.

To summarize, our contributions are as follows:

- • We provide a detailed analysis of existing diffusion distillation models and propose a novel efficient method to accelerate video diffusion models, eliminating the presence of useless data points and enabling efficient distillation.
- • We present SynVid, a synthetic video dataset containing 110K high-quality synthetic videos, denoising trajectories, and corresponding fine-grained text prompts.
- • We design a trajectory-based few-step guidance that leverages key data points from the synthetic dataset to learn the noise-to-video mapping with fewer steps and an adversarial training strategy, which effectively utilizes the synthetic dataset, enhancing the performance.
- • Extensive experiments demonstrate that we achieve 8.5× improvements in generation speed compared to the teacher model while maintaining comparable performance. Moreover, we produce videos with higher quality and resolution, i.e., 5-seconds, 720×1280, 24fps, compared to previous accelerating methods.

### 2. Related Work

Video Diffusion Models. With the advancement of largescale video data and large-scale models, video diffusion models have achieved remarkable success. The pioneering work VDM [17] extends the 2D U-Net [45] used in image generation to 3D UNet and proposes joint training with both images and videos. [4, 11, 14, 52, 66] adopt the latent diffusion model (LDM) [44] to learn video data distribution in the latent space, significantly reducing computational complexity. To further increase the resolution of generated videos, [3, 16, 53] employ cascaded video diffusion models, which decomposes the video generation process into subtasks, such as key frame generation, frame interpolation, and super-resolution. With the impressive video generation capabilities shown by Sora [41], the diffusion transformer architecture (DiT) [42] has gradually become the mainstream backbone for video diffusion models. Latte [38] and GenTron [6] explore different variants of DiT for video generation. Snap Video [40], HunyuanVideo [23], and Cosmos [1] replace 1D+2D self-attention with 3D self-attention and leverage more careful data curation pipelines, larger-scale models, and more computational resources, achieving state-of-the-art video generation performance. For more details on video diffusion models, please refer to [57].

Accelerating Image Diffusion Models. Here, we introduce distillation techniques to accelerate image diffusion models. [2, 9, 46] progressively distill the teacher model, enabling the student model to learn the noise-to-image mapping with fewer inference steps. PeRFlow [59] divides the denoising process into several windows and learns the data mapping within each window. LCM [36] utilize consistency models [49] in the image latent space to accelerate pretrained image diffusion models. Inspired by Variation Score Distillation (VSD) [55] and Score Distillation Sampling (SDS) [43], [37, 48, 62, 63] propose the distribution matching loss, which aligns the real and fake image distributions by utilizing the score function derived from the diffusion model. Additionally, [29, 47, 54, 58] employ adversarial loss to train few-step or one-step image generators, using a diffusion model as the feature extractors. [33, 35] share similarities with our approach, they also utilize synthetic data to learn the noise-to-data mapping in fewer steps. However, they overlook the distribution information contained in the denoising trajectory, resulting blurry outputs.

Accelerating Video Diffusion Models. Recent studies primarily accelerate video diffusion models from three perspectives: efficient model architectures, high-compressionrate Variational Autoencoders (VAEs) [22], and distillation techniques. LinGen [51] employs the linear-complexity Mamba2 block [8], significantly reducing the training computational costs. LTX-Video [12] utilizes a carefully designed Video-VAE that achieves a high compression ratio,

reducing the input dimensions of the diffusion model and enhancing generation speed. AnimateDiff-Lightning [28] extends progressive adversarial distillation [29] to video diffusion models. T2V-Turbo [25] and T2V-Turbo-V2 [26] leverage refined consistency distillation loss [36] to reduce the number of inference steps and enhance video quality using reward models. Concurrent works, CausVid [64] and APT [30], leverage distribution matching loss and adversarial loss, respectively, to distill video diffusion models. They attempt to address the dataset mismatch by training the teacher model with their own video dataset before distillation. However, it requires substantial computational resources, especially when training models for highresolution video generation. Moreover, compared to them, our approach is capable of generating higher quality and resolution videos, i.e., 5-seconds, 720×1280, 24fps.

### 3. Preliminaries

#### 3.1. Flow Matching

In this section, we provide a brief overview about Flow Matching [31], which is commonly used in generative tasks. Flow Matching transforms a complex data distribution p0(x) into the simple standard normal distribution p1(x) = N(0,I) through a conditional probability paths pt(x|x0), where x0 ∼ p0(x) and t ∈ [0,1]. In [31], the conditional probability paths have the form:

pt(x|x0) = N(x|µt(x0),σt(x0)2I), (1)

where µt(x0) is the time-dependent mean, while σt(x0) describes the time-dependent scalar standard deviation (std). According to the Optimal Transport theory, the mean and the std are set to change linearly in time,

µt(x0) = (1 − t)x0,σt(x0) = t. (2)

Then, the sample xt = (1 − t)x0 + tx1 ∼ pt(x|x0) is obtained through forward diffusion operation. Here, x1 ∼ p1(x). A model v parameterized by θ is trained to predict the velocity ut(x|x0) = x1 − x0, which guides the sample xt towards the data x0. The training loss has the form:

LFM(θ) = Et,p

0(x),p1(x) vθ(xt,t) − ut(x|x0))

2

. (3)

After training, a sampled Gaussian noise x1 ∼ p1(x) can be denoised to data x0 ∼ p0(x) by integrating the predicted velocity vθ(xt,t) through the first-order Euler Ordinary Differential Equation (ODE) solver.

#### 3.2. Analysis of Previous Distillation Methods

Previous diffusion distillation methods can be broadly categorized into two classes: (1) knowledge distillation [2, 9, 25, 26, 36, 46, 59], where a student model is trained to

Gaussian Noise

[Figure 3]

[Figure 4]

T=1

T=1

|[Figure 5]|
|---|

Denoise

- T=0

- T=1

T=0

- T=0

Gaussian Noise

Data

|𝑥𝑡|
|---|

|𝑥𝑡′|
|---|

T=𝑡

T=𝑡′

b) Dataset or Gaussian Noise Mismatching

T=1

Gaussian Noise

Data

|𝑥𝑡|
|---|

c) Distribution Distillation

- T=1

Data

Learned ODE

a) Knowledge Distillation

[Figure 6]

Gaussian Noise

Gaussian Noise

[Figure 7]

[Figure 8]

[Figure 9]

𝑥𝑡𝑓𝑎𝑘𝑒 𝑥𝑡𝑟𝑒𝑎𝑙

T=𝑡

|𝑥𝑡𝑟𝑒𝑎𝑙|
|---|

𝑥𝑡𝑓𝑎𝑘𝑒

T=0

T=0

Data 𝑥0𝑓𝑎𝑘𝑒 𝑥0𝑟𝑒𝑎𝑙

𝑥0𝑓𝑎𝑘𝑒 𝑥0𝑟𝑒𝑎𝑙

Data

- e) Frequency of Useless Data Points

|𝑀|
|---|

Frequency(%)

26.93% 29.06%

17.43%

6.79%

0.78%

d) Dataset or Gaussian Noise Mismatching

- f) Distillation Results with Different 𝑀

Distillation Dataset 𝑝𝑑(𝑥)

Student Teacher

[Figure 10]

[Figure 11]

|[Figure 12]| | |
|---|---|---|
| |𝑀 = 3| |

𝑀 = 1.5

𝑀 = 0

- Figure 2. 1D Toy Experiment. We employ Flow Matching objective [31] to train the teacher model, which learns the ODE that maps Gaussian distribution to the data distribution. The data distribution consists of two data points. a) illustrates the knowledge distillation methods, where a student model is trained to mimic the teacher model’s denoising process. b) highlights the challenges posed by dataset or Gaussian noise mismatching in knowledge distillation, which can lead to unreliable guidance. c) demonstrates the distribution matching methods, which aims to align the output distribution of the student model with that of the teacher model. d) emphasizes the issue in distribution matching, which can result in inaccurate guidance. e) illustrates the frequency of useless data points in relation to M. f) shows the distillation results at various values of M.

mimic the denoising process of a teacher model with fewer inference steps, e.g., mapping xt to xt′ or x0 using one step, as shown in Fig. 2 a). The start data points xt are obtained through the forward diffusion operation. However, these methods inadvertently distill useless start data points xt, which do not lie on the denoising trajectories of the teacher model due to the dataset mismatching or Gaussian noise mismatching, as depicted in Fig. 2 b). When denoising such useless data points using the teacher model, it often produces inaccurate results, as highlighted by the red box in Fig. 2 b). Consequently, it can lead to unreliable guidance for the student model during distillation; (2) distribution distillation [30, 37, 47, 58, 62, 63], which leverages diffusion models to compute distribution matching loss or adversarial loss [54] to optimize the student model, e.g., feeding xfaket and xrealt into the pretrained diffusion model to derive meaningful guidance, as illustrated in Fig. 2 c). However, they may still rely on useless data points xfaket and xrealt due to the dataset mismatching or Gaussian noise mismatching, as shown in Fig. 2 d), which leads the diffu-

sion model to provide inaccurate guidance.

We count the frequency of useless data points in relation to the degree of data mismatching and Gaussian noise mismatching in a 1D toy experiment. We define the degree of mismatching between the distillation dataset pd(x) and the training dataset p(x) as follows,

min{|xd − x| for x ∈ p(x)}. (4)

##### M =

xd∈pd(x)

- Fig. 2 e) shows the results. Even without dataset mis-

matching, i.e., M = 0, the presence of Gaussian noise mismatching can still produce useless data points. As the degree of dataset mismatching M increases, it becomes evident that the frequency of useless data points grows significantly. Additionally, we conduct distillation experiments followed by [59] at different M. The results are illustrated in Fig. 2 f). As M increases, more useless data points are used during distillation, making the teacher model provide incorrect guidance. Consequently, this leads the generated data to deviate from the training dataset p(x) and the distillation dataset pd(x), demonstrating that useless data points are harmful to the distillation process.

4. Method

Our method aims to distill the pretrained video diffusion model [23] to reduce the number of inference steps, thereby accelerating video generation. Based on the analysis in Sec. 3.2, we avoid using useless data points during the distillation process. Specifically, we first introduce a synthetic video dataset, SynVid, which leverages the teacher model to generate high-quality synthetic videos and denoising trajectories (Sec. 4.1). Subsequently, we propose a trajectorybased few-step guidance that selects key data points from the denoising trajectories and learns the noise-to-video mapping based on these data points, enabling video generation with fewer inference steps (Sec. 4.2). Additionally, we introduce an adversarial training strategy that exploits the data distribution at each diffusion timestep captured by SynVid, further enhancing the model’s performance (Sec. 4.3).

- Fig. 3 illustrates the overview of our method. 4.1. SynVid: Synthetic Video Dataset

We employ HunyuanVideo [23] as our generator to produce synthetic data as shown in Fig. 4. HunyuanVideo vθ is a text-to-video diffusion model utilizing the DiT architecture [42], which operates in the latent space and is trained using the Flow Matching loss, as outlined in Eq. 3. We use the official code and settings* to generate our dataset.

Our SynVid Dsyn = {V i,lti

,Ti|i ∈ [1,N]} comprises high-quality synthetic videos Vi, denoising trajectories in latent space {lti

,lti

,...,lti

n−1

0

n

,lti

,...,lti

|tn = 1 >

n−1

n

0

*https://github.com/Tencent/HunyuanVideo

|𝑡𝑘′|
|---|

|𝑡𝑘−1′|
|---|

|𝑞(𝑙𝑡<br><br>𝑘<br><br>′)|
|---|

|𝑝(𝑙𝑡<br><br>𝑘<br><br>′<br><br>𝑔𝑒𝑛)|
|---|

|𝑝(𝑙𝑡<br><br>𝑘−1<br><br>′<br><br>𝑔𝑒𝑛)|
|---|

|𝑞(𝑙𝑡<br><br>𝑘−1<br><br>′ )|
|---|

Denoise Trajectory from SynVid

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

| | |
|---|---|
| | |
| | |

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

[Figure 40]

[Figure 41]

[Figure 42]

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

###### … …

###### … …

[Figure 70]

Student

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

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Model

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

| |
|---|

|Pretrained Video<br><br>Diffusion Model<br><br>|
|---|

|Pretrained Video<br><br>Diffusion Model<br><br>|
|---|

|ℒtraj|
|---|

Same

Many Steps

Many Steps

|Pretrained Video Diffusion Model<br><br>| | | | |
|---|---|---|---|---|
| | | | | |

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

Student

| | | | |
|---|---|---|---|
| |Projection Head 𝑯𝒌<br><br>| | |
| | | | |

[Figure 122]

[Figure 123]

Model

[Figure 124]

###### Projection Head 𝑯𝒌−𝟏

[Figure 125]

One Step

Fake or Real ? Fake or Real ?

(a) Trajectory-based Few-step Guidance

(b) Adversarial Training Strategy

- Figure 3. Method Overview. (a) Our method first designs a trajectory-based few-step guidance, which utilizes the key data points from the denoising trajectory to enable the student model to mimic the denoising process of the pretrained video diffusion model with fewer steps. (b) To fully exploit the data distribution at each diffusion timestep captured by our synthetic dataset, we propose an adversarial training strategy to align the output distribution of the student model with that captured by our synthetic dataset.

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

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

High-quality Videos

InternVL2.5

“Describe this video in detail”

… a young boy joyful moment as he flies a toy airplane … dressed in a

green jacket and a yellow bean … the airplane, a simple wooden model … His face is lit up with a broad smile … The background ... atmosphere of the video is ...

HunyuanVideo

|Synthetic Video 𝑉𝑖|
|---|

[Figure 143]

Denoising Trajectory {𝑙𝑡𝑖𝑛,…,𝑙𝑡𝑖0}

Text Prompt

|𝑇𝑖|
|---|

[Figure 144]

[Figure 145]

…

SynVid

Figure 4. The pipeline of constructing SynVid.

... > t0 = 0}, and their corresponding text prompts Ti, where N = 110K represents the number of data, n = 50 denotes the number of inference steps, and {tj|j ∈ [0,n]} denote the inference diffusion timesteps. The text prompts Ti are annotated by InternVL2.5 [7] with high-quality videos from the Internet. For a Gaussian noise lti

n

, the denoising trajectory can be solved as follows,

lti

j−1

= lti

j

+ (tj−1 − tj)vθ(lti

j

,tj,Ti), (5)

the synthetic video V i can be decoded by VAE using the clean latent lti

0

. For brevity, we omit the text prompt input T in following sections. It is noteworthy that we exclusively utilize the synthetic dataset Dsyn during the distillation process. The data points in Dsyn are intermediate results that lead to the correct output lti

0

, making them all valid and meaningful. This significantly aids in efficient distillation and reduces the demand for data.

- 4.2. Trajectory-based Few-step Guidance

|𝑡𝑘′ =0.9130|
|---|

|𝑡𝑘′ =0.8235|
|---|

|𝑡𝑘′ =0.6363|
|---|

|𝑡𝑘′ =0|
|---|

|𝑡𝑘′ =0.9651|
|---|

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

LayerLayerLayerLayer6020401

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

|[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]|[Figure 160]|
|---|---|
|[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]|[Figure 165]|

Figure 5. The illustration of features at different layers and diffusion timesteps of our feature extractor. The features within the red box are selected for discrimination.

model, i.e., the teacher model, to learn the noise-to-video mapping with fewer inference steps. The student model shares the same architecture as the teacher model and is parameterized by β, which is initialized using the parameters θ of the teacher model.

Specifically, we select m + 1 key diffusion timesteps {t′m = 1 > t′m−1 > ... > t′0 = 0} and obtain their corresponding latents {lti′m

} on each denoising trajectory. Then, we propose a trajectory-based loss to learn the denoising process of the teacher model with m steps,

,lti′

,...,lti′

m−1

0

lti′

− lti′

2

,t′k+1) −

Ltraj = Ei,k sβ(lti′

, (6)

k

k+1

)

t′

k − t′

k+1

k+1

Video diffusion models typically require a large number of inference steps to generate videos, which is time-consuming and computationally intensive. To accelerate the generation process, we design a student model sβ that utilizes denoising trajectories generated by the pretrained video diffusion

where k ∈ [0,m − 1]. By learning from these key latents, which construct a shorter path from Gaussian noise to video latent, our student model significantly reducing the number of inference steps. In our experiments, we set m = 5, which reduces the number of inference steps by a factor of

10 compared to the teacher model, greatly accelerating the generation process.

Algorithm 1: AccVideo Distillation Procedure

Input : Pretrained video diffusion model vθ, SynVid Dsyn, m. Output: Distilled m-step student model sβ, projection heads

{H0, ..., Hm−1}.

- 1 // Initialize student model from pretrained model
- 2 β ← θ
- 3 // Select m key timesteps on denoising trajectory
- 4 t′m = 1 > t′m−1 > ... > t′0 = 0
- 5 // Initialize latent queues with ∅
- 6 Q0, ..., Qm ← ∅
- 7 while train do

- 8 for k ← m − 1 to 0 do

- 9 // Sample key data points
- 10 {lt′m, lt′

m−1

, ..., lt′ 0

} ∼ Dsyn

- 11
- 12 // Update student model with Ltraj
- 13 Ltraj ← traj(sβ, lt′ k

, lt′

k+1

) // Eq. 6

- 14 β ← Update(β, Ltraj)
- 15 Sample z ∼ N(0, I)
- 16 Qm.push(z, lt′ m−1

, ..., lt′ 0

) if k = m − 1

- 17
- 18 // Compute latent ltgen′ k

∼ pβ(ltgen′

k

)

- 19 ltgen′ k+1

, lt′

m−1

, ..., lt′ 0

← Qk+1.pop()

- 20 ltgen′ k

← Solver(sβ, ltgen′

k+1

, t′k+1, t′k) // Eq. 5

- 21 Qk.push(ltgen′ k

, lt′

m−1

, ..., lt′ 0

)

- 22
- 23 // Update student model and Hk with Ladv
- 24 Ladv ← adv(ltgen′ k

, lt′

k

, t′k, Hk, vθ) // Eq. 7

- 25 β ← Update(β, λadvLadv)
- 26 Hk ← Update(Hk, λadvLadv)
- 27 end for
- 28 end while

#### 4.3. Adversarial Training Strategy

In addition to the one-to-one mapping between Gaussian noise and video latents, our synthetic dataset also contains many latents {lti′

|i ∈ [1,N]} at each diffusion timestep t′k, which implicitly represent the data distribution q(lt′

k

). To fully unleash the knowledge in our synthetic dataset and enhance the performance of our student model sβ, we propose an adversarial training strategy to minimize adversarial divergence Dadv q(lt′

k

)||p(ltgen′

) , where p(ltgen′

) denotes the

k

k

k

generated data distribution at diffusion timestep t′k of our student model sβ. The training objective follows the standard Generative Adversarial Network (GAN) [10]:

Ladv =min

max

Ek,q(l

),p(ltgen′ k

) log(D(lt′

t′ k

β

ϕ

,t′k)) + log(1 − D(ltgen′

,t′k)) ,

k

k

(7)

where D represents the discriminator parameterized by ϕ. Since the discriminator needs to distinguish noisy latents at different diffusion timesteps, we design our discriminator D to consist of a noise-aware feature extractor and m timestep-

aware projection heads {H0,...,Hm−1},

,t′k) = Hk vθ(lt′

,t′k) , D(ltgen′

D(lt′

k

k

,t′k) = Hk vθ(ltgen′

,t′k) ,

k

k

(8)

here, we leverage the frozen teacher model vθ as the feature extractor inspired by [29, 58, 62]. Fig. 5 shows the layer-wise behavior at different diffusion timesteps of vθ. When t′k > 0, the deeper layer concentrate on capturing high-frequency details, so we select the output at the final layer, i.e., the 60th layer for discrimination. When t′k = 0, we choose the output at the 40th layer for discrimination, which contains fine-grained clues, such as sticky notes on the table. Each projection head Hk is a 2D convolutional neural network used to output the label indicating where the latents lt′

and ltgen′

comes from. The projection heads are timestep-aware, effectively handling noisy features at different diffusion timesteps and thereby facilitating adversarial learning.

k

k

For latent ltgen′

∼ pβ(ltgen′

), it can be iteratively solved using sβ, similar to Eq. 5. However, it is time-consuming, which is not practical implementation. Therefore, we employ m+1 queues {Q0,...,Qm}, which maintain the latents generated by sβ at each diffusion timestep.

k

k

It should be noted that our adversarial training strategy avoids the forward diffusion operation commonly used in existing methods [58, 62], preventing the distillation of useless data points. This ensures that the teacher model provides accurate guidance for the student model. Moreover, the incorporation of our trajectory-based few-step guidance improves the stability of adversarial learning, eliminating the requirement for complex regularization designs [30].

Total Objective. The training parameters include β and the parameters of the projection heads. These parameters are trained using Ltraj and λadvLadv, where λadv is a hyperparameter. Algorithm 1 shows the distillation procedure.

### 5. Experiments

Implementation Details. Our student model and teacher model adopt the HunyuanVideo architecture [23] and are initialized using the officially released checkpoint. The teacher model remains frozen. The student model and projection heads are trained for 1200 iterations using the AdamW optimizer [34] with a learning rate of 5 × 10−6 for 12 days, utilizing 8 A100 GPUs and gradient accumulation to achieve a total batch size of 32. The hyperparameter λadv is set to 0.1. We use SynVid as described in Sec. 4.1 as our distillation dataset, where the resolution of the latents and videos is 68 × 120 × 24 and 544 × 960 × 93, respectively. After distillation, we can generate higher-resolution videos, i.e., 720 × 1280 × 129. The inference process in our experiments is conducted on a single A100 GPU.

[Figure 166]

[Figure 167]

T2V-Turbo-V2

[Figure 168]

[Figure 169]

LTX-Video

[Figure 170]

[Figure 171]

PyramidFlow

[Figure 172]

[Figure 173]

CogVideoX1.5

[Figure 174]

[Figure 175]

HunyuanVideo-720P

[Figure 176]

[Figure 177]

Ours-544P

[Figure 178]

[Figure 179]

Ours-720P

- Figure 6. Qualitative results on text-to-video. Left: a girl raises her left hand to cover her smiling mouth. Right: the camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope.

Evaluation. Our method is evaluated on VBench [19], a comprehensive benchmark that is commonly used to evaluate the video quality. Specifically, VBench evaluates video generation models across 16 dimensions using 946 prompts. Each prompt is sampled five times to reduce randomness.

Baselines. The baselines include VideoCrafter2, which is a 3DUNet-based model, as well as OpenSora [65], HunyuanVideo [23], and CogVideoX [61], which are DiT-based models. Additionally, we compare with faster models such

- as T2V-Turbo-V2 [26], which utilize distillation to accelerate video generation, and LTX-Video [12], which employs a high-compression VAE to speed up video generation, as well as PyramidFlow [20], which combines autoregressive and diffusion models for efficient video generation.

#### 5.1. Comparisons on Text-to-Video Generation

Inference Time. Tab. 1 presents the comparison of the inference time. Our model requires only 5 inference steps to generate videos, it achieves a 7.7-8.5× improvement in generation speed compared to the teacher model. Even compared to models with half the model size of ours, e.g.,

Table 1. Comparison of inference time. Here, we use a single A100 GPU to measure the time required for generating a video, which includes text encoding, VAE decoding, and diffusion time.

Resolution Method H × W × L Inference Time (s)

CogVideoX-5B 480 × 720 × 49 219 HunyuanVideo-544P 544 × 960 × 93 704 Ours-544P 544 × 960 × 93 91

Medium

CogVideoX1.5-5B 768 × 1360 × 81 926

High

HunyuanVideo-720P 720 × 1280 × 129 3234 Ours-720P 720 × 1280 × 129 380

CogVideoX-5B and CogVideoX1.5-5B [61], our model is still 2.7× faster.

Quantitative Evaluation. The evaluations on VBench are presented in Tab. 2. T2V-Turbo-V2 [26] achieves leading results in total score, primarily due to their highest performance in dynamic degree (improving by at least 13%). However, this may be attributed to its imperfect temporal consistency in local areas, as evidenced by its poor performance in temporal flickering. Moreover, it struggles to generate high-resolution videos, whereas our model re-

Table 2. Text-to-video comparisons on VBench [19]. The results marked with ∗ are tested by us, while the other results are obtained from the official VBench leaderboard. For detailed results across all 16 dimensions, please refer to our supplementary materials.

Resolution Method H × W × L Total Score Quality Score Semantic Score Temporal Flickering Dynamic Degree Object Class Color Spatial Relationship Low

VideoCrafter2 320 × 512 × 16 80.44% 82.20% 73.42% 98.41% 42.50% 92.55% 92.92% 35.86% T2V-Turbo-V2 320 × 512 × 16 83.52% 85.13% 77.12% 97.35% 90.00% 95.33% 92.53% 43.32%

LTX-Video 512 × 768 × 121 80.00% 82.30% 70.79% 99.34% 54.35% 83.45% 81.45% 65.43% CogVideoX-5B 480 × 720 × 49 81.61% 82.75% 77.04% 98.66% 70.97% 85.23% 82.81% 66.35% HunyuanVideo-544P∗ 544 × 960 × 93 82.67% 84.43% 75.59% 99.03% 76.38% 88.67% 92.07% 67.39% Ours-544P∗ 544 × 960 × 93 83.26% 84.58% 77.96% 99.18% 75.00% 92.99% 94.11% 75.70%

Medium

PyramidFlow 768 × 1280 × 121 81.72% 84.74% 69.62% 99.49% 64.63% 86.67% 82.87% 59.53% OpenSora V1.2 720 × 1280 × 204 79.76% 81.35% 73.39% 99.53% 42.39% 82.22% 90.08% 68.56% CogVideoX1.5-5B 768 × 1360 × 81 82.17% 82.78% 79.76% 98.88% 50.93% 87.47% 87.55% 80.25% HunyuanVideo-720P 720 × 1280 × 129 83.24% 85.09% 75.82% 99.44% 70.83% 86.10% 91.60% 68.68% Ours-720P∗ 720 × 1280 × 129 82.77% 84.38% 76.34% 99.01% 75.55% 89.71% 94.61% 71.74%

High

|[Figure 180]|
|---|

[Figure 181]

[Figure 182]

[Figure 183]

| |
|---|

(a) Teacher (5 steps)

|[Figure 184]|
|---|

[Figure 185]

[Figure 186]

[Figure 187]

| |
|---|

|(b) w/ 𝓛traj|
|---|

|[Figure 188]|
|---|

[Figure 189]

[Figure 190]

[Figure 191]

| |
|---|

|w/ 𝓛𝐭𝐫𝐚𝐣 and 𝓛𝐚𝐝𝐯 w/o Time-aware Heads|
|---|

(c)

|[Figure 192]|
|---|

[Figure 193]

[Figure 194]

[Figure 195]

| |
|---|

(d) Full

Frame 0 Frame 12 Frame 49

- Figure 7. Qualitative ablation study results. Please zoom in for details. Prompt: a toy robot wearing purple overalls and cowboy boots taking a pleasant stroll in Johannesburg South Africa during a beautiful sunset.

tains this ability. In addition to T2V-Turbo-V2 [26] and our teacher model, HunyuanVideo [23], our method surpasses all the other compared baselines at the same resolution level. Compared to our teacher model, we achieve comparable performance at high resolution and better performance

- at medium resolution. In particular, we achieve exceptional performance in color and spatial relationship, indicating that our model can effectively interpret text prompts and generate videos coherent with the text. This not only demonstrates the effectiveness of our method but also highlights that our synthetic dataset contains high-quality text prompts and data points.

Qualitative Evaluation. Fig. 6 presents the qualitative results. Compared to T2V-Turbo [26], LTX-Video [12], and PyramidFlow [20], our method generates videos with fewer artifacts, such as the hands of the little girl. Compared to CogVideoX1.5 [61], we produce higher-fidelity videos with better backgrounds. Compared to HunyuanVideo [23], our method better aligns with the text prompts, such as the vintage SUV. For more qualitative results, please refer to our supplementary materials.

#### 5.2. Ablation Studies

Trajectory-based Few-step Guidance. The trajectorybased few-step guidance is designed to guide the student model to learn the denoising trajectories of the teacher model with fewer steps. As illustrated in Fig. 7 (a) and

- (b), we observe that the teacher model struggles to generate clear videos with fewer inference steps, while distilling with our trajectory-based loss is capable of producing clearer videos in just 5 steps. We greatly accelerate video generation by reducing the number of inference steps,. Adversarial Training Strategy. As shown in Fig. 7 (b) and
- (c), although the trajectory-based few-step guidance accelerates video generation, the generated videos contain artifacts with unnatural motions. In contrast, when the adversarial training strategy is employed, we can generate higherquality videos. This further demonstrates its effectiveness. Timestep-Aware Projection Heads. We conduct an ablation study, which only uses one projection head to discriminate the extracted features. As illustrated in Fig. 7 (c) and
- (d), the generated videos better align with the text prompts and produce more coherent videos when using our timestepaware projection heads. We hypothesize that when a single projection head is used, it may struggle to distinguish features from different diffusion timesteps, i.e., different noise levels. In contrast, the timestep-aware projection heads can more effectively learn variance present in these features.

### 6. Conclusion

In this paper, we first analyze the challenges faced by existing diffusion distillation methods, which are caused by the use of useless data points. Based on this insight, we propose a novel efficient distillation method to accelerate video diffusion models. Specifically, we first construct a synthetic video dataset, SynVid, which contains valid and meaningful data points for distillation. Then, we introduce a trajectorybased few-step guidance that enables the student model to generate videos in just 5 steps, significantly accelerating the generation speed compared to the teacher model. To further enhance the video quality, we design an adversarial training strategy that leverages the data distribution captured by our synthetic video dataset. Our model achieves

- 8.5× improvements in generation speed compared to the teacher model while maintaining comparable performance.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2, 3
- [2] David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023. 2, 3
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, pages 22563–22575, 2023. 2, 3
- [4] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, pages 7310–7320, 2024. 2, 3
- [5] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. In ICLR, 2025. 12
- [6] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Delving deep into diffusion transformers for image and video generation. In CVPR,

2024. 3

- [7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 5, 12
- [8] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In ICML, 2024. 3
- [9] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In ICLR,

2025. 2, 3

- [10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, 2024. 6
- [11] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3
- [12] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime

video latent diffusion. arXiv preprint arXiv:2501.00103,

2024. 3, 7, 8, 12

- [13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 12
- [14] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 3

- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NIPS, pages 6840–6851, 2020. 2
- [16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [17] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NIPS, pages 8633–8646, 2022. 3
- [18] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In CVPR, pages 8153–8163, 2024. 2
- [19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024. 7, 8
- [20] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025. 2, 7, 8
- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. NIPS, 35:26565–26577, 2022. 2
- [22] Diederik P Kingma. Auto-encoding variational bayes. In ICLR, 2014. 3
- [23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 2, 3, 4, 6, 7, 8
- [24] Kuaishou. Kling, 2024. https : / / klingai . kuaishou.com. 2
- [25] Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. In NIPS, 2024. 3
- [26] Jiachen Li, Qian Long, Jian Zheng, Xiaofeng Gao, Robinson Piramuthu, Wenhu Chen, and William Yang Wang. T2vturbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design. In ICLR, 2025. 3, 7, 8
- [27] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025. 2

- [28] Shanchuan Lin and Xiao Yang. Animatediff-lightning: Cross-model diffusion distillation. arXiv preprint arXiv:2403.12706, 2024. 3
- [29] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 2, 3, 6
- [30] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316,

2025. 2, 3, 4, 6

- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 2, 3, 4
- [32] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 2
- [33] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusionbased text-to-image generation. In ICLR, 2023. 3
- [34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 6
- [35] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021. 3
- [36] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 3
- [37] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. In NIPS, 2024. 2, 3, 4
- [38] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2, 3
- [39] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In AAAI, pages 4117–4125, 2024. 2
- [40] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In CVPR, pages 7038–7048, 2024. 3
- [41] OpenAI. Sora, 2024. https://openai.com/sora. 2, 3
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 2, 3, 4
- [43] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR,

2023. 3

- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 3
- [45] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241, 2015. 3

- [46] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 2, 3
- [47] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia, pages 1–11, 2024. 3, 4
- [48] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In ECCV, pages 87–103, 2024. 2, 3
- [49] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. 3
- [50] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024. 2
- [51] Hongjie Wang, Chih-Yao Ma, Yen-Cheng Liu, Ji Hou, Tao Xu, Jialiang Wang, Felix Juefei-Xu, Yaqiao Luo, Peizhao Zhang, Tingbo Hou, et al. Lingen: Towards high-resolution minute-length text-to-video generation with linear computational complexity. arXiv preprint arXiv:2412.09856, 2024. 3, 13
- [52] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [53] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, pages 1–20, 2024. 2, 3
- [54] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. In ICLR, 2023. 2, 3, 4
- [55] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NIPS, 2024. 3
- [56] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. In ICLR, 2025. 13
- [57] Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. A survey on video diffusion models. ACM Computing Surveys, 57(2):1–42, 2024. 3
- [58] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In CVPR, pages 8196–8206, 2024. 2, 3, 4, 6
- [59] Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. In NIPS, 2024. 2, 3, 4
- [60] Mingyu Yang, Junyou Li, Zhongbin Fang, Sheng Chen, Yangbin Yu, Qiang Fu, Wei Yang, and Deheng Ye. Playable game generation. arXiv preprint arXiv:2412.00887, 2024. 2
- [61] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video

- diffusion models with an expert transformer. In ICLR, 2025. 2, 7, 8
- [62] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NIPS, 2024. 2, 3, 4, 6
- [63] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, pages 6613–6623, 2024. 2, 3, 4
- [64] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast causal video generators. arXiv preprint arXiv:2412.07772, 2024. 2, 3
- [65] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 2, 7
- [66] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 3

## AccVideo: Accelerating Video Diffusion Model with Synthetic Dataset Supplementary Material

BatchNorm

AvgPool1D

|Diffusion Time 𝑡|
|---|

###### PE

PE :Position Encoding

Residual

Conv1D

Block

FC

|1D Input 𝑥𝑡|
|---|

Velocity

BatchNorm

BatchNorm

BatchNorm

BatchNorm

Conv1D

Conv1D

Conv1D

Conv1D

Figure 8. Model architecture for the 1D toy experiment.

### A. Details about the 1D Toy Experiment

In Sec. 3.2, we conduct a 1D toy experiment to analyze existing diffusion distillation methods. Specifically, we set the training data as {−3,3}. The training loss is shown in Eq. (1). We use ResNet [13] to construct our model, as illustrated in Fig. 8. The model is trained using the Adam optimizer for 10000 iterations with a learning rate of 10−4. Although our dataset consists of only two data points, we achieve a batch size of 2048 by repeating the data.

### B. SynVid

Recent methods have demonstrated that fine-grained text prompts play a crucial role in video generation. Inspired by this, we leverage a multimodal large language model (MLLM), i.e., InternVL2.5-8B [7], to annotate real videos and obtain high-quality text prompts. Fig. 9 shows the length distribution of text prompts, most text lengths are concentrated between 100 and 150, demonstrating that the text prompts provide rich semantic information. As depicted in Fig. 10, the text prompts accurately describe the people and objects present in the videos, along with details about them, as well as dynamic motions. Moreover, they include descriptions of the atmosphere and background, which greatly aids in generating high-quality videos. The high-quality synthetic dataset facilitates the model distillation.

To promote community development and support future research, SynVid includes videos of different resolutions, i.e., medium and high resolution, with detailed information provided in Tab. 3.

### C. More Results

Full Comparisons on VBench. We evaluate our method on VBench benchmark. The full comparisons across 16 dimensions on VBench are shown in Tab. 4 and Tab. 5.

More Qualitative Results. Fig. 11 and Fig. 12 showcase

[Figure 196]

Length Distribution of Text Prompts

[Figure 197]

Number

| |
|---|

Text Length

Figure 9. The length distribution of text prompts.

In the video, a man is seendrinking from a water bottleon a red track. He is wearinga yellow and black shirt, and his face is wet, indicating that he has been exercising or running. The track is set against a backdrop of ablue sky and trees, suggesting an outdoor setting. The man's posture and the act of drinking water suggest that he is taking a break from his exercise routine. The overall style of the video is realistic and captures a common scene of someone engaging in physical activity.

[Figure 198]

The video begins with a close-up shot ofa person dressed in a clown costume, with their face paintedwhite and black, holding an array ofballoons, which are primarily black and red. The person is outdoors, and thebackground is blurry, showing trees with branches that suggest it might be autumn. As the frames progress, the clown's head moves slightly downwards. The balloons remain held high, with theclown adjusting themthroughout the sequence. There is a sense of stillness and quietness in the environment around the clown, emphasizing acalm atmosphere despite the vibrant balloons.

[Figure 199]

Figure 10. The illustration of our synthetic video dataset, SynVid. Table 3. Details resolutions about our synthetic dataset, SynVid.

Resolution H × W × L (Video) H × W × L (Latents) Number

High 720×1280×129 90×160×33 8303 Medium 544×960×93 68×120×24 105996

the results of our method in generating videos at high and medium resolutions, respectively.

### D. Limitations and Future Work

Our method focuses on accelerating video diffusion models by distillation technique, which reduces the number of inference steps. However, generation speed is also influenced by VAE, which are used to encode and decode videos, making the acceleration of the VAE another promising direction for exploration [5, 12]. Additionally, the DiT architecture contains many transformer blocks, and accelerating these

Table 4. Full comparison on VBench using all 16 metrics.

Aesthetic Quality Low

Total Score

Quality Score

Semantic Score

Subject Consistency

Background Consistency

Temporal Flickering

Motion Smoothness

Dynamic Degree

Resolution Method H × W × L

VideoCrafter2 320 × 512 × 16 80.44% 82.20% 73.42% 96.85% 98.22% 98.41% 97.73% 42.50% 63.13% T2V-Turbo-V2 320 × 512 × 16 83.52% 85.13% 77.12% 95.50% 96.71% 97.35% 97.07% 90.00% 62.61%

LTX-Video 512 × 768 × 121 80.00% 82.30% 70.79% 96.56% 97.20% 99.34% 98.96% 54.35% 59.81% CogVideoX 480 × 720 × 49 81.61% 82.75% 77.04% 96.23% 96.52% 98.66% 96.92% 70.97% 61.98% HunyuanVideo-544P 544 × 960 × 93 82.67% 84.43% 75.59% 94.33% 97.13% 99.03% 98.64% 76.38% 61.97% Ours-544P 544 × 960 × 93 83.26% 84.58% 77.96% 94.46% 97.45% 99.18% 98.79% 75.00% 62.08%

Medium

PyramidFlow 768 × 1280 × 121 81.72% 84.74% 69.62% 96.95% 98.06% 99.49% 99.12% 64.63% 63.26% OpenSora V1.2 720 × 1280 × 204 79.76% 81.35% 73.39% 96.75% 97.61% 99.53% 98.50% 42.39% 56.85% CogVideoX1.5 768 × 1360 × 81 82.17% 82.78% 79.76% 96.87% 97.35% 98.88% 98.31% 50.93% 62.79% HunyuanVideo-720P 720 × 1280 × 129 83.24% 85.09% 75.82% 97.37% 97.76% 99.44% 98.99% 70.83% 60.36% Ours-720P 720 × 1280 × 129 82.77% 84.38% 76.34% 94.20% 96.87% 99.01% 98.90% 75.55% 60.61%

High

Table 5. Full comparison on VBench using all 16 metrics.

Imaging Quality

Object Class

Multiple Objects

Human Action

Spatial Relationship

Overall Consistency Low

Appearance Style

Temporal Style

Resolution Method H × W × L

Color

Scene

VideoCrafter2 320 × 512 × 16 67.22% 92.55% 40.66% 95.00% 92.92% 35.86% 55.29% 25.13% 25.84% 28.23% T2V-Turbo-V2 320 × 512 × 16 71.78% 95.33% 61.49% 96.20% 92.53% 43.32% 56.40% 24.17% 27.06% 28.26%

LTX-Video 512 × 768 × 121 60.28% 83.45% 45.43% 92.80% 81.45% 65.43% 51.07% 21.47% 22.62% 25.19% CogVideoX 480 × 720 × 49 62.90% 85.23% 62.11% 99.40% 82.81% 66.35% 53.20% 24.91% 25.38% 27.59% HunyuanVideo-544P 544 × 960 × 93 65.57% 88.67% 67.69% 94.80% 92.07% 67.39% 51.23% 19.43% 23.96% 26.80% Ours-544P 544 × 960 × 93 65.64% 92.99% 67.33% 95.60% 94.11% 75.70% 54.72% 19.87% 23.71% 27.21%

Medium

PyramidFlow 768 × 1280 × 121 65.01% 86.67% 50.71% 85.60% 82.87% 59.53% 43.20% 20.91% 23.09% 26.23% OpenSora V1.2 720 × 1280 × 204 63.34% 82.22% 51.83% 91.20% 90.08% 68.56% 42.44% 23.95% 24.54% 26.85% CogVideoX1.5 768 × 1360 × 81 65.02% 87.47% 69.65% 97.20% 87.55% 80.25% 52.91% 24.89% 25.19% 27.30% HunyuanVideo-720P 720 × 1280 × 129 67.56% 86.10% 68.55% 94.40% 91.60% 68.68% 53.88% 19.80% 23.89% 26.44% Ours-720P 720 × 1280 × 129 66.70% 89.71% 66.35% 94.00% 94.61% 71.74% 50.75% 20.16% 23.44% 26.92%

High

###### transformer blocks is another area that warrants further investigation [51, 56].

In a realistic close-up shot with smooth camera movement,a charming woman is seen outdoors on a grassy lawn. She is wearing awhite

shirtpaired with awhite jacket, and she adorns a necklace and earrings, adding elegance to her appearance. The woman is gracefully

walking around an area enclosed bya wooden fence, moving in a gentle arc as she walks past the fence. The background features a lush

green lawn and tent-like structures, creating a serene and refreshing atmosphere. The lighting is ample, highlighting the natural beauty of the scene.

[Figure 200]

A middle-aged sadbald manbecomes happy as a wig ofcurly hair and sunglassesfall suddenly on his head.

[Figure 201]

A wide-angle view of adramatic cliffsideoverlooking the ocean,waves crashingagainst the rocks far below.

[Figure 202]

###### Adog wearing virtual reality goggles insunset.

[Figure 203]

A close-up of abutterfly's wings, showing the intricate patterns and vibrant colors in fine detail.

[Figure 204]

Acapybara relaxes in a wooden barrelfilled with steaming hot spring water, its serene gaze adding tranquility to the scene.

Perched atop its head is avibrant orange, adding a playful contrast to its soft brown fur.

[Figure 205]

Figure 11. Qualitative results on text-to-video with high resolution, i.e., 720×1280×129.

A westernprincess, withsunlight shining through the leaves on her face, facial close-up.

[Figure 206]

Abeautiful womanwalking on theschool playground. The sun shining on her face.

[Figure 207]

An extreme close-up of angray-haired manwitha beardin his 60s, he is deep in thought pondering the history of the universe as he sits at

a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in awool coat suit coatwith a

button-down shirt , he wears a brown beret andglasses and has a very professorial appearance, and the end he offers a subtle closed-

mouth smile as if he found the answer to the mystery of life, the lighting is very cinematic with the golden light and the Parisian streets and

city in the background, depth of field, cinematic 35mm film.

[Figure 208]

Awhite and orange tabby catis seen happily darting througha dense garden, as if chasing something. Its eyes are wide and happy as it

jogs forward, scanning the branches, flowers, and leaves as it walks.The path is narrowas it makes its way betweenall the plants. the

scene is captured from aground-level angle, following the cat closely, giving a low and intimate perspective. The image is cinematic with

warm tones and a grainy texture. The scattered daylight between the leaves and plants above creates a warm contrast, accentuating the

cat’s orange fur. The shot is clear and sharp, with a shallow depth of field.

[Figure 209]

Themonster stared at the foodwith wide eyes and open mouth. Its posture and expression convey a sense of innocence and playfulness.

[Figure 210]

Apanda in a scientist's lab coat, conducting experiments withbeakers and test tubes.

[Figure 211]

Figure 12. Qualitative results on text-to-video with medium resolution, i.e., 544×960×93.

