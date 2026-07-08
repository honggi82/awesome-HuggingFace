## DepthMaster: Taming Diffusion Models for Monocular Depth Estimation

Ziyang Song* , Zerong Wang* , Bo Li , Hao Zhang , Ruijie Zhu , Li Liu , Peng-Tao Jiang

†

†

, Tianzhu Zhang

### arXiv:2501.02576v2[cs.CV]23Apr2026

Abstract—Monocular depth estimation within the denoising diffusion paradigm demonstrates impressive generalization ability but suffers from low inference speed. Recent methods adopt a single-step deterministic paradigm to improve inference efficiency. However, uncritically applying the generative features from diffusion models to the perceptual depth estimation task leads to suboptimal results, due to the differences between image generation and dense prediction. In this work, we propose DepthMaster, a single-step diffusion model designed to tame diffusion models for enhanced performance and visual quality. First, to mitigate overfitting to texture details introduced by generative features, we propose a Feature Alignment module, which incorporates high-quality semantic features to enhance the denoising network’s representation capability. Second, to address the lack of fine-grained details in the single-step deterministic framework, we propose a Fourier Enhancement module to adaptively balance low-frequency structure and high-frequency details. We adopt a two-stage training strategy to fully leverage the potential of the two modules. In the first stage, we focus on learning the global scene structure with the Feature Alignment module, while in the second stage, we exploit the Fourier Enhancement module to improve the visual quality. Through these efforts, our model achieves state-of-the-art performance in terms of generalization and detail preservation, outperforming other diffusion-based methods across various datasets. Our project page can be found at https://indu1ge.github.io/DepthMaster page.

Index Terms—Monocular depth estimation, Zero-shot depth estimation, Diffusion models.

I. INTRODUCTION

# M

ONOCULAR depth estimation (MDE) has garnered considerable attention due to its simplicity, low cost,

and ease of deployment. Unlike traditional depth-sensing techniques such as LiDAR or stereo vision, MDE only requires a single RGB image as input, making it highly appealing for a wide range of applications, including autonomous driving [1]– [4], virtual reality [5], [6], and image synthesis [7], [8]. This versatility also presents a significant challenge: achieving exceptional generalization to effectively handle the diversity and complexity of broad-range application scenarios. However, this is a non-trivial task due to variants in scene layouts, depth distributions, lighting conditions, etc.

*Equal contribution. †Corresponding authors.

Ziyang Song, Ruijie Zhu, Li Liu, and Tianzhu Zhang are with School of Information Science and Technology, University of Science and Technology of China (USTC), Hefei 230026, P.R.China. Contact: {songziyang, ruijiezhu, liu li}@mail.ustc.edu.cn; {tzzhang}@ustc.edu.cn.

Zerong Wang, Bo Li, Hao Zhang, Peng-Tao Jiang are with vivo Mobile Communication Co., Ltd., Hangzhou 310030, P.R.China. Contact: {wangzerong}@live.com; {libra, haozhang, pt.jiang}@vivo.com.

This work was done during Ziyang Song’s internship at vivo.

Recent research on zero-shot monocular depth estimation has evolved into two main branches: data-driven [9]–[17] and model-driven [18]–[22]. The former relies on large-scale image-depth pairs to achieve generalization across various scenes, where data collection and training process is extremely time-consuming and resource-exhausting. In contrast, modeldriven approaches aim to adapt powerful pre-trained backbones with a small amount of annotated data and shorter training schedules, particularly in the context of Stable Diffusion models [23], [24]. For example, Marigold [18] reformulates depth estimation as a denoising diffusion process, achieving impressive performance in both generalization and detail preservation. However, the iterative denoising process results in a low inference speed. GenPercept [21] proposes a deterministic single-step paradigm, which directly inputs RGB images and outputs depth maps, improving inference efficiency while maintaining comparable performance. Despite these advancements in applying diffusion models to MDE, few works have thoroughly explored how to best adapt the generative features in diffusion models for the perceptual task.

In this work, we conduct an in-depth analysis of the feature representation within diffusion models. First, diffusion models are typically pre-trained to reconstruct clean images from noisy inputs, which is not capable to eliminate unnecessary texture details [25]–[27]. In contrast, monocular depth estimation is a perceptual task that requires more high-level structural and semantic understanding. Due to this fundamental mismatch, directly applying features from pre-trained diffusion models to depth estimation often leads to inaccurate and unrealistic textures in depth predictions, as illustrated in Fig. 1, Row (a). Therefore, how to enhance the model’s semantic representation capacity and reduce its reliance on irrelevant details is a key issue for taming diffusion models for depth estimation. Furthermore, the high visual quality of diffusion models’ outputs comes from the iterative refinement process. In the early steps, the model learns to recover the general structure, while in later steps, details are gradually refined [28]–[30]. When reformulated as a singlestep paradigm, the model is required to capture both primary structure and fine details in a single forward pass, which often results in blurry predictions [20]–[22], as shown in Fig. 1, Row (b). Thus, how to improve the fine-grained details in the single-step paradigm is another crucial challenge in leveraging the generative features for depth estimation.

To address the aforementioned challenges, we propose DepthMaster, a tamed single-step diffusion model designed to enhance the generalization and detail preservation abilities of

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

| |
|---|

| |
|---|

| |
|---|

(a)(b)

|[Figure 5]<br><br>| |
|---|
|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

| |
|---|

| |
|---|

| |
|---|

RGB Ground Truth Single-step Ours

- Fig. 1. Challenges in current single-step diffusion model-based methods. As shown in Row (a), limited by the feature representation capability of the denoising network, depth predictions of current single-step diffusion model-based methods tend to overfit texture details (shadow of the chair) and miss the real structure. Besides, due to the removal of the iterative process, current single-step methods suffer from blurry outputs, as shown in Row (b). By delicately adapting the features in diffusion models, our method achieves accurate and fine-grained depth predictions.

depth estimation models. First, to enhance the representational capacity of the diffusion model, we align its features with those from a high-quality external encoder via a Feature Alignment module. This alignment effectively injects semantic context and alleviates overfitting to texture details. Second, to alleviate the lack of fine-grained details due to the removal of the iterative process, we propose a Fourier Enhancement module. It adaptively balances low-frequency structure and high-frequency details in a single forward pass, effectively simulating the learning process of the multi-step denoising. To fully leverage the potential of the modules, we adopt a two-stage training strategy. The first stage leverages the Feature Alignment module to learn scene structure, while the second stage incorporates the Fourier Enhancement module to improve fine-grained details. Through tailoring generative features and exploiting the two-stage training strategy, our method achieves impressive zero-shot performance and superior detail preservation ability, outperforming existing state-ofthe-art diffusion model-based approaches, thereby narrowing the gap between data-driven and model-driven paradigms.

The main contributions of our work are as follows:

- • We propose DepthMaster, a novel approach that customizes generative features in diffusion models to suit the perceptual depth estimation task.
- • We introduce a Feature Alignment module to mitigate overfitting to texture details with high-quality external features and a Fourier Enhancement module to refine finegrained details in the frequency domain.
- • Our method exhibits state-of-the-art zero-shot performance and superior detail preservation ability, surpassing other diffusion-based methods across various datasets.

II. RELATED WORK A. Single-domain Monocular Depth Estimation

Monocular depth estimation plays a crucial role in 3D perception. Since Eigen et al. [31] introduce convolutional neural networks for end-to-end training, a series of learningbased methods have been proposed, which can be broadly categorized into three branches: regression-based, classificationbased, and denoising diffusion-based. Regression-based methods directly predict continuous per-pixel depth values from a single image. Early approaches [32], [33] rely on deep residual networks to extract depth features. To better leverage contextual information, Wang et al. [34], and Song et al. [35] introduce multi-scale fusion strategies. Later, recent works [12], [36], [37] incorporate Transformer to capture longrange dependencies, achieving notable improvements. With the advent of diffusion models, several methods [38]–[40] utilize pre-trained diffusion backbones to further enhance model performance. Beyond architecture innovations, recent research also explores improving model robustness in challenging scenes [41], [42] and improving depth representations through auxiliary cues and multi-task learning [43]–[48]. Classification-based approaches discretize the depth range into bins and classify each pixel accordingly. This idea is first introduced by Cao et al. [49]. DORN [50] further treats it as an ordinal regression problem to capture the inherent ordering of depth intervals. To improve flexibility, AdaBins [51] proposes an adaptive binning strategy that dynamically allocates depth intervals per image. BinsFormer [52] adopts a MaskFormerstyle [53] architecture for improved interaction between image features and bin representations. Building on this, LocalBins [54] further refines the binning process by assigning bins at the pixel level, enabling finer-grained predictions. Denoising diffusion-based methods iteratively denoise a noisy representation to recover the clean depth map. DepthGen [55]

and DDP [56] first introduce this paradigm, demonstrating its effectiveness even with sparse ground truth supervision. DDVM [57] extends the framework to broader dense prediction tasks such as optical flow. To enhance conditioning, DiffusionDepth [58] incorporates multi-scale features, leading to improved accuracy. Diffusion4RobustDepth [41] then leverages diffusion models to synthesize challenging images, thus enhancing model robustness. More recently, MonoDiffusion [59] brings this paradigm into the self-supervised setting by diffusing a pre-trained teacher’s predictions. Although these single-domain methods achieve impressive results, they struggle to generalize across diverse real-world scenarios.

- B. Zero-shot Monocular Depth Estimation

Estimating accurate depth maps for in-the-wild images is a crucial but challenging task due to variants in scene layouts, depth distributions, lighting conditions, etc. Some pioneering works have tried to solve this problem, which can be primarily divided into two main branches: data-driven and model-driven. The former focuses on collecting large-scale image-depth pairs to achieve the mapping from RGB images to depth maps. To maintain training stability, these approaches opt to predict relative depth, which can already represent the scene structure. For example, Diversedepth [16] and MiDaS [15] predict affineinvariant depth to jointly train on multiple datasets and achieve good generalization across various scenarios. On top of this, Omnidata [14] introduces a dataset that comprises roughly 14.5 million images and trains a robust depth estimation model following MiDaS to achieve zero-shot cross-dataset transfer. More recently, ZoeDepth [13] trains a relative depth estimation model and demonstrates its transferability to metric depth through fine-tuning on metric depth datasets. We follow this strategy to predict relative depth because it is not only practical but also can be converted to metric depth easily. DPT [12] further enhances MiDaS by replacing the original CNN with a Vision Transformer. Depth Anything [10], then, expands the training datasets with 62 million unlabeled images to further enlarge data coverage. However, these methods rely on largescale datasets, making the data collection and training process time-consuming and resource-exhausting.

The second branch of research seeks to improve model generalization by leveraging powerful image priors inherited in pre-trained models, especially in the context of Stable Diffusion models, which are trained on large-scale, highquality datasets. Marigold [18] first explores the potential of pre-trained latent diffusion models (LDMs) for monocular depth estimation by reformulating depth estimation as a conditional denoising diffusion process. GeoWizard [19] further improves it by incorporating normal estimation to enhance the ability to capture geometric details. To address the problem of low inference efficiency caused by iterative denoising, DepthFM [20] introduces flow matching to reduce the number of sampling steps at the cost of a slight performance degradation. More recently, GenPercept [21] offers a systematic analysis of fine-tuning protocols and proposes a single-step deterministic paradigm, where only the image latent is fed into the denoising network, and the noise latent

is output for depth prediction. Amazingly, it notably reduces the inference time with comparable performance. Lotus [22] also exploits a single-step denoising paradigm and proposes a detail preserver branch to improve visual quality. Despite these advancements in applying diffusion models to MDE, no work has thoroughly explored how to best adapt the generative features in diffusion models for the perceptual task. In this work, we bridge this gap by enhancing the denoising network’s representation capability and adaptively refining features in the frequency domain, leading to further improvements in both performance and visual quality.

III. METHOD

The overall framework is illustrated in Fig. 2. We begin with introducing the single-step deterministic paradigm, as detailed in Sec. III-A. Next, we provide an in-depth analysis of the Stable Diffusion model and introduce a Feature Alignment module to enhance the representation capability of the denoising network in Sec. III-B. To address the limitation of the single-step model’s inability to capture fine-grained details, we introduce a Fourier Enhancement module in Sec. III-C and a weighted multi-directional gradient loss in Sec. III-D. Finally, we present the two-stage training strategy in Sec. III-E.

A. Deterministic Paradigm

Our model is built upon Stable Diffusion v2 [23], which is pre-trained on the large-scale LAION-5B [60] dataset. The powerful image priors encoded in the model significantly assist in the depth estimation task. To reduce computational overhead, Stable Diffusion v2 operates in the latent space by employing an image-to-latent (I2L) encoder that compresses high-resolution image data into a latent representation at 18 the original resolution. The diffusion and denoising processes are performed within this compact latent space, and the denoised latent is subsequently decoded back into the image domain using an I2L decoder. To leverage the rich image priors learned by this model, we adopt the same paradigm and perform image perception in the latent space. Rather than reformulating the task to fit the denoising diffusion paradigm exploited by the diffusion model, we craft the model to better adapt to the task. Specifically, we employ a single-step deterministic transformation from image to depth, as illustrated in the upper part of Fig. 2. First, the image I ∈ RH×W×3 is encoded into the latent space using the I2L encoder, denoted as E, and obtain the image latent

##### zRGB = E (I), (1)

where zRGB ∈ Rh×w×4, h = H8 and w = W8 . The image latent is then fed into the U-Net model, denoted as ϵθ, which performs scene perception and generates the corresponding depth latent. The timestep is set to 1, ensuring a direct conversion from image latent to depth latent:

##### zpred = ϵθ (zRGB,1). (2)

Since the I2L encoder-decoder reconstructs depth maps with a negligible loss of accuracy, we only fine-tune the U-Net. Instead of predicting depth, we opt to predict square-root

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

U-Net

[Figure 14]

[Figure 15]

I2L Encoder

I2L Decoder

D1 D2 D3 M FE U1 U2 U3

𝒛𝑅𝐺𝐵 𝒛𝑝𝑟𝑒𝑑 𝐅𝑚𝑖𝑑

𝐃𝑝𝑟𝑒𝑑

RGB

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Spatial Pass

Conv

Conv

Feat Aligner

S

[Figure 20]

[Figure 21]

𝐅𝑚𝑖𝑑 𝐅෨𝑚𝑖𝑑

Fuse

ℒ𝑓𝑎

C

[Figure 22]

[Figure 23]

Modulator

[Figure 24]

𝐅෠𝑚𝑖𝑑

External Encoder

𝐅𝑚𝑖𝑑

iFFT

FFT

R

S

Frequency Pass

RGB 𝐅෨𝑒𝑥𝑡

[Figure 25]

[Figure 26]

Feature Alignment (FA) Module Stage1

Fourier Enhancement (FE) Module Stage2

- Fig. 2. The overall framework of DepthMaster. RGB is first projected into the latent space by the I2L Encoder to obtain zRGB. Next, the U-Net converts RGB latent to depth prediction latent zpred, which is decoded back to the depth map by the I2L Decoder. The Feature Alignment module is applied in the first stage to align the representation of the U-Net to that of the high-quality external encoder, introducing semantic information into the diffusion model. In the second stage, the Fourier Enhancement module adaptively balances low-frequency structure and high-frequency details to enhance the visual quality. S represents the Softmax operation, R refers to the reshaping operation, and C indicates concatenation along the channel dimension.

disparity. On the one hand, square-root disparity emphasizes the accuracy of nearby objects, which is desired by applications like autonomous driving. On the other hand, squareroot disparity leads to a more uniform distribution, thus fully releasing the capability of the input range. The preprocessed ground truth (GT) DGT is first normalized to [−1,1] to fit the input range of the I2L encoder. Then, we replicate it into three channels to simulate a gray-scale RGB image, which is passed through the I2L encoder to obtain the GT depth latent zGT. The training objective in the latent space is given as follows:

##### Llatent = (zGT − zpred)2. (3)

The final depth prediction Dpred is decoded from the depth latent zpred using the I2L decoder and postprocessed by averaging channels. The pixel-level supervision is defined as:

##### Lpixel = ∥DGT − Dpred∥2. (4)

By removing the iterative denoising process, the single-step deterministic paradigm significantly improves inference efficiency with comparable generalization performance.

- B. Feature Alignment Module

Stable Diffusion v2 consists of two components: the I2L encoder-decoder and the denoising U-Net. The I2L encoderdecoder is responsible for feature compression, which aims to reduce inference time and training costs. Trained with image reconstruction, it primarily captures low-level features. In contrast, the U-Net is responsible for recovering images from their noisy counterparts, enabling it to harvest scene perception and reasoning capabilities. However, since the U-Net is pretrained to reconstruct clean images from noisy inputs, it is not capable to eliminate unnecessary details, leading to “pseudotextures” rather than capturing true structure.

Inspired by REPA [25], we introduce semantic regularization to enhance the U-Net’s scene representation capabilities and prevent overfitting to superficial color information. Specifically, we incorporate a pre-trained external encoder f, which provides high-quality semantic feature representation. In this work, we use DINOv2 [61] as the external encoder. For a given RGB image I, the external feature representation is Fext = f(I) ∈ Rh×w×D, where h × w is the spatial size and D is the feature size. Simultaneously, the U-Net encoder extracts image representation at its middle block, denoted as Fmid ∈ Rh×w×C. Our goal is to align the image representation from the U-Net with the high-quality representation from the external encoder. Since the two representations lie in different spaces, we use a Multi-Layer Perceptron hϕ to project Fmid into the feature space of Fext, yielding the transformed representation F¯mid = hϕ(Fmid).

To enforce feature alignment, we minimize the distance between the two feature distributions. The feature alignment loss is defined as follows:

##### Lfa(Fext,F¯mid) = dist(Fext,F¯mid), (5)

where dist(·,·) measures the distance between the two feature distributions. In this work, we utilize the Kullback-Leibler (KL) divergence. Specifically, the features are first normalized along the feature dimension with the Softmax operation S, obtaining the distribution in the latent space. We then minimize the KL divergence between the two feature distributions:

##### dist(Fext,F¯mid) = kl div(F˜ext,F˜mid), (6)

where F˜ext and F˜mid represents the normalized features. Through feature alignment, the U-Net learns more semantically meaningful representation, improving its generalization ability while avoiding overfitting to low-level details. Notably,

the external encoder is a training-only component and removed at inference, incurring no additional test-time cost.

- C. Fourier Enhancement Module

The single-step paradigm effectively speeds up the inference process by avoiding multi-step iterations and multi-run integration. However, the fine-grained characteristic of the diffusion models’ outputs typically arises from the iterative refinement process, where details are gradually refined in the later steps. When reformulating in the single-step paradigm, models are observed to struggle in capturing both structure and details, leading to blurry depth maps [20]–[22]. Recent studies [29], [62], [63] demonstrate that frequency domain analysis provides an effective way to improve the detail quality of adapted diffusion models. Inspired by this line of research, we propose a Fourier Enhancement module, which operates in the frequency domain to simulate the iterative refinement process’s focus on different bands, as illustrated in the rightbottom of Fig. 2. Specifically, the Fourier Enhancement module is composed of two components: a spatial pass for general structure capture and a frequency pass for detail enhancement. In the frequency pass, the hidden state from the U-Net’s middle block Fmid ∈ Rh×w×C is first transformed into the frequency domain using a 2D Fast Fourier Transform (FFT), yielding Fmid

f

. To adaptively balance information across different frequency bands, a modulator comprised of convolution and activation layers is applied to Fmid

f

. The enhanced feature is then transformed back to the spatial domain using an inverse 2D Fast Fourier Transform (iFFT):

Ff = iFFT(σ(Conv(FFT(Fmid)))), (7)

where σ refers to the activation layer. Next, we concatenate the feature from the spatial pass Fs with that from the frequency pass Ff and perform a convolution operation to obtain the final enhanced feature Fˆmid:

Fˆmid = Conv(Fs∥Ff), (8)

where ∥ denotes the concatenation operator. By operating in the frequency domain, our model adaptively balances lowfrequency structural features and high-frequency detail features within a single forward pass, effectively improving the visual quality of depth predictions.

- D. Weighted Multi-directional Gradient Loss

To further enhance the sharpness of depth predictions, we propose a weighted multi-directional gradient loss function to capture detailed edge information on depth maps in all directions. Specifically, for the ground truth depth DGT and depth prediction Dpred, we compute gradients GGT ∈ RH×W×4 and Gpred ∈ RH×W×4 in horizontal, vertical and diagonal directions. At edges where the foreground and background meet, gradient values are typically much larger than those within local structures. These dominant differences can overwhelm the gradient loss, leading to unstable training and suboptimal

- Algorithm 1 The First Stage Training Process

- 1: Input: Training set D = {In,Dn}Nn=1, I2L encoder E, U-Net ϵθ, external encoder f, feature aligner hϕ, loss balancing factor λfa, first-stage training iteration Tstage1
- 2: Initialization:
- 3: Load E, ϵθ from Stable Diffusion v2
- 4: Randomly initialize hϕ
- 5: for iteration t ∈ {1, ···, Tstage1} do
- 6: (I,D) ← Sampler(D)
- 7: zpred,Fmid = ϵθ(E(I))
- 8: Fext = f(I)
- 9: zGT = E(D)
- 10: Lstage1 = Llatent(zpred,zGT) + λfaLfa(Fext,F¯mid)
- 11: Update ϵθ and hϕ by back-propagation
- 12: end for
- 13: Output: Optimized ϵθ

- Algorithm 2 The Second Stage Training Process

- 1: Input: Training set D = {In,Dn}Nn=1, I2L encoder E, I2L decoder D, U-Net with Fourier Enhancement module ϵψ, loss balancing factor λh, second-stage training iteration Tstage2
- 2: Initialization:
- 3: Load E, D from Stable Diffusion v2
- 4: Load ϵψ from the first stage model
- 5: for iteration t ∈ {1, ···, Tstage2} do
- 6: (I,D) ← Sampler(D)
- 7: Dpred = D(ϵψ(E(I)))
- 8: GGT = ∇(D)
- 9: Gpred = ∇(Dpred)
- 10: Lstage2 = Lpixel(Dpred,D) + λhLh(GGT,Gpred)
- 11: Update ϵψ by back-propagation
- 12: end for
- 13: Output: Optimized ϵψ

solutions. To mitigate this problem, we employ a modified Huber loss [64], defined as follows:

δ · |GGT − Gpred|, if |GGT − Gpred| ≤ δ,

Lh =

- 1

- 2 (GGT − Gpred)2 + 21δ2, otherwise,

(9) where δ controls the threshold at which the loss transitions from quadratic to linear, reducing the influence of outliers caused by large gradient differences at the foregroundbackground interface. Since the depth values are scaled to the interval [-1, 1], the gradient loss corresponding to most foreground-background interface points is multiplied by a value less than 1, thus reducing their proportion in the total gradient loss. This adjustment allows our model to focus on the fine details not only at foreground-background interfaces but also within local structures.

E. Two-stage Training Curriculum

Since the depth reconstruction accuracy of the I2L encoderdecoder is sufficiently high, we focus on fine-tuning the UNet. Experiments reveal that latent-space supervision helps the

TABLE I QUANTITATIVE COMPARISON WITH STATE-OF-THE-ART ZERO-SHOT AFFINE-INVARIANT MONOCULAR DEPTH ESTIMATION METHODS. THE UPPER PART LISTS DATA-DRIVEN METHODS AND THE LOWER PART PRESENTS THOSE BASED ON DIFFUSION MODELS. ALL METRICS ARE IN PERCENTAGE TERMS WITH “BOLD” BEST AND “UNDERLINE” SECOND BEST. “*” STANDS FOR THE RESULTS REPRODUCED BY LOTUS [22].

Training Data

FLOPs (G)

NYUv2 KITTI ETH3D ScanNet DIODE Avg. AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ Rank

Method Year

Data-driven methods DiverseDepth [16] arXiv’20 320K 207.8 11.7 87.5 19.0 70.4 22.8 69.4 10.9 88.2 37.6 63.1 11 MiDaS [15] TPAMI’20 2M 184.6 11.1 88.5 23.6 63.0 18.4 75.2 12.1 84.6 33.2 71.5 10.4 LeReS [65] CVPR’21 354K 190.4 9.0 91.6 14.9 78.4 17.1 77.7 9.1 91.7 27.1 76.6 7.8 Omnidata [14] ICCV’21 12.2M 195.4 7.4 94.5 14.9 83.5 16.6 77.8 7.5 93.6 33.9 74.2 7.2 DPT [12] ICCV’21 1.4M 449.2 9.8 90.3 10.0 90.1 7.8 94.6 8.2 93.4 18.2 75.8 6.1 HDN [66] NIPS’22 300K 195.4 6.9 94.8 11.5 86.7 12.1 83.3 8.0 93.9 24.6 78.0 5.3 MonoDiffusion [59] TCSVT’24 39K 38.8 24.5 55.0 16.4 73.4 20.0 65.1 22.4 61.3 34.6 50.5 11.4 Diffusion4RobustDepth [41] ECCV’24 30K 159.5 8.9 92.4 13.0 83.6 17.8 89.3 8.8 93.0 28.1 71.4 7.7 Metric3D [67] ICCV’23 8M 225.4 5.8 96.3 5.8 97.0 6.6 96.0 7.4 94.1 22.4 78.5 3.3 Depth Anything [10] CVPR’24 62M 586.0 4.3 98.1 7.6 94.7 6.5 97.2 4.2 98.0 27.7 75.9 3.3 Depth Anything V2 [11] NIPS’24 63.5M 1816.5 4.4 97.9 7.5 94.8 6.2 98.0 4.3 98.1 26.0 75.9 3.1 Metric3Dv2 [9] TPAMI’24 16M 520.6 4.3 98.1 4.3 98.2 4.2 98.3 2.2 99.4 13.6 89.5 1.0

Model-driven methods

Marigold [18] CVPR’24 74K 5196.5 5.5 96.4 9.9 91.6 6.5 96.0 6.4 95.1 30.8 77.3 4.3 GeoWizard [19] ECCV’25 280K 5247.3 5.2 96.6 9.7 92.1 6.4 96.1 6.1 95.3 29.7 79.2 2.9 DepthFM [20] AAAI’25 74K 2487.4 6.0 95.5 9.1 90.2 6.5 95.4 6.6 94.9 22.4 78.5 4.5 GenPercept [21] ICLR’25 74K 2142.8 5.6 96.0 9.9 90.4 6.2 95.8 6.2* 96.1* 35.7 75.6 4.4 Lotus [22] ICLR’25 59K 2142.8 5.3 96.7 9.3 92.8 6.8 95.3 6.0 96.3 22.8 73.8 3.5 DepthMaster (Ours) - 74K 2147.1 5.0 97.2 8.2 93.7 5.3 97.4 5.5 96.7 21.5 77.6 1.2

model to better capture the overall scene structure, while pixellevel supervision improves fine-grained details but introduces distortions in the global structure. Based on these observations, we propose a two-stage training strategy. In the first stage, our goal is to train a model that can robustly generalize across diverse scenarios. To achieve this, we apply constraints in the latent space and incorporate the Feature Alignment module to enhance the model’s scene perception capability,

- with the training process illustrated in Algorithm 1. The training objective of the first stage is as follows:

- Lstage1 = Llatent + λfaLfa, (10)

where λfa is set to 1. In the second stage, we aim to optimize the model’s performance on detail preservation. To balance structure and detail information, we incorporate the Fourier Enhancement module. After obtaining depth predictions, we apply constraints at the pixel level and introduce the weighted multi-directional gradient loss to enhance edge sharpness, with the training process illustrated in Algorithm 2. The total objective function for the second stage is as follows:

- Lstage2 = Lpixel + λhLh, (11)

where Lpixel is the pixel MSE loss and λh is set to 0.001. Benefiting from the two-stage training strategy, our model achieves both accurate structure capture and sharp edge preservation.

IV. EXPERIMENTS A. Implementation Details

Our model is based on Stable Diffusion v2 [23], with text conditioning disabled. In the first stage, we train our model for 20k iterations using the Adam [68] optimizer with a learning rate of 3 × 10−5. In the second stage, we reduce the learning rate to 3 × 10−6 and train for an additional 10k iterations. To achieve a batch size of 32, we exploit gradient accumulation.

Training the first stage takes approximately 30 hours, while fine-tuning the model in the second stage requires an additional 30 hours, both on a single NVIDIA H800 GPU. Additionally, we apply random horizontal flipping augmentation to enhance the diversity of training datasets.

B. Datasets

Training Datasets. We train our model on two synthetic datasets: Hypersim [69] and Virtual KITTI [70]. Hypersim is a high-fidelity dataset covering 461 indoor scenes with rich textures and geometry, generated using realistic 3D rendering techniques. We use the depth annotations and corresponding RGB images to train our model. Following Marigold [18], we transform the original depth values relative to the focal point into depth values relative to the focal plane. The official split with around 54K samples is used with the training resolution of 480×640. Virtual KITTI is a synthetic outdoor dataset that serves as a variant of the original KITTI [71] dataset, providing a wide range of road scenes under diverse lighting, weather, and traffic conditions. Unlike KITTI, Virtual KITTI is generated with a 3D simulator and provides dense depth annotations. We train on approximately 20k samples with a resolution of 1216 × 352 and set the far plane to 80 meters. The two datasets are mixed in a ratio of 9:1.

Evaluation Datasets. We evaluate our model’s zero-shot performance on 5 real datasets. NYU-Depth-V2 (NYUv2) [72] and ScanNet [73] are indoor datasets commonly used for evaluating depth estimation methods. We use the official test split with 654 images for NYUv2 and the split proposed by Marigold with 800 images for ScanNet. KITTI [71] is a street-scene dataset captured with equipment mounted on a moving vehicle. We follow the Eigen split [31], which consists of 652 images. ETH3D [74] and DIODE [75] are two real datasets containing both indoor and outdoor im-

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

KITTIDIODENYUv2ETH3DScanNet

|[Figure 33]<br><br>| |
|---|
|
|---|

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

RGB Ground Truth Marigold [18] DAv2 [11] Lotus [22] Ours

- Fig. 3. Qualitative comparison across different datasets with zero-shot monocular depth estimation methods. Our model demonstrates excellent detail preservation and structure capture capabilities. Benefiting from the Feature Alignment module, our model avoids overfitting to textures.

ages. For evaluation, we use the splits in Marigold to evaluate on 454 samples from ETH3D and 771 samples from DIODE. For zero-shot evaluation, we report absolute relative

error AbsRel = HW1 ΣHW |DGT−Dˆpred|

DGT and accuracy metric δ1 = HW1 ΣHW max D ˆDpred

#### , D

Dˆpred < 1.25 , where Dˆpred is the aligned depth prediction and [·] is Iverson bracket. For sharp boundary evaluation, we use the F1-score proposed by DepthPro [76], which computes the recall and precision of edges in depth predictions and ground truth depth maps. This metric reflects the visual quality of depth predictions to some extent, with higher values indicating higher visual quality.

GT

GT

- C. Qualitative and Quantitative Comparison

Table I presents a comparison of our method with other state-of-the-art (SOTA) zero-shot monocular depth estimation approaches. The upper part of the table lists data-driven methods, while the lower part focuses on diffusion modelbased methods. As shown in Table I, despite being trained on a relatively small amount of data, diffusion model-based methods already outperform many approaches that rely on large-scale datasets. This highlights the significant role of strong image priors encoded in diffusion models, which greatly enhance the generalization capabilities of depth estimation models. Our approach belongs to the diffusion model-based category. By incorporating the specially designed Feature Alignment module and Fourier Enhancement module, we achieve a 17.2% improvement over Marigold [18] in AbsRel on KITTI, effectively narrowing the performance gap between diffusion model-based methods and those reliant on large-scale

datasets. Besides, the single-step deterministic paradigm effectively improves the inference efficiency through the removal of the iterative denoising process, whose computation complexity is comparable to Depth Anything V2 [11]. To better illustrate the superiority of our approach, we provide qualitative results in Fig. 3. As highlighted in red boxes, the multi-step iterative diffusion model Marigold [18] demonstrates fine details in thin objects, whereas the single-step Lotus [22] often overlooks details such as thin wires, leading to relatively blurry outputs. With the proposed Fourier Enhancement Module and the two-stage training strategy, our method effectively alleviates this issue and achieves comparable detail preservation with Marigold, while avoiding texture overfitting commonly encountered in leveraging generative models. Additionally, Fig. 4 showcases predictions on in-the-wild examples, further demonstrating our model’s remarkable generalization ability in unconstrained real-world scenarios. These results emphasize the practical applicability and versatility of our approach, making it highly suitable for various real-world applications.

D. Ablation Studies

In this section, we conduct comprehensive experiments to validate the effectiveness of our designs. We first conduct a series of additive ablations on the learning paradigm, depth preprocessing strategies, model components and the feature alignment model and its integration location. Then we assess the role of the two-stage training strategy, the gradient loss Lh, and the Fourier Enhancement module in preserving details and maintaining performance.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Lotus[22]OursRGBMarigold[18]DAv2[11]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Fig. 4. Qualitative results on in-the-wild examples. Our model not only recovers correct scene structure, but also exhibits fine-grained details.

TABLE II ABLATION OF PARADIGM. “I2L” MEANS FEEDING DEPTH MAPS INTO THE IMAGE-TO-LATENT (I2L) ENCODER-DECODER AND OUTPUTTING RECONSTRUCTED ONES. “DENOISING” REFERS TO THE MULTI-STEP DENOISING DIFFUSION PARADIGM. “ITERATIVE” INDICATES ITERATIVE REFINEMENT BY PASSING THROUGH THE U-NET FOUR TIMES IN A DETERMINISTIC WAY. “SINGLE-STEP” DENOTES THE SINGLE-STEP DETERMINISTIC PARADIGM. “*” MARKS THE PARADIGM WE USE. “M.FULL” IS THE FINAL MODEL.

KITTI NYUv2 ScanNet ETH3D DIODE Hypersim Inference

Paradigm

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ F1↑ Time (s) I2L - - 1.1 99.5 0.9 99.7 - - 8.4 92.4 0.615 Denoising 10.4 90.2 5.7 96.0 6.9 94.6 6.4 95.7 30.9 76.8 0.274 12.91 Iterative 10.0 91.1 5.2 96.7 5.9 96.1 6.1 96.3 29.4 77.8 0.310 0.83 Single-step* (M.Single) 10.3 90.4 5.3 96.6 6.0 96.2 6.5 95.8 29.9 77.0 0.304 0.42

M.Full 8.2 93.7 5.0 97.2 5.3 97.4 5.5 96.7 21.5 77.6 0.337 0.42

TABLE III ABLATION OF DEPTH PREPROCESS. PREDICTING DISPARITY INSTEAD OF DEPTH RESULTS IN IMPROVED PERFORMANCE ON OUTDOOR DATASETS, WHILE USING SQUARE-ROOT DISPARITY LEADS TO CONSISTENT IMPROVEMENTS ACROSS ALL DATASETS.

Depth Preprocess Baseline: M.Single

KITTI NYUv2 ETH3D ScanNet DIODE

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

depth (D) 10.3 90.4 5.3 96.6 6.5 95.8 6.0 96.2 29.9 77.0

disparity (D1 ) 8.9 92.4 5.3 97.0 6.7 96.7 5.7 96.3 22.4 74.0 sqrt disp (√1D ) (M.Base) 8.7 93.1 5.1 97.3 5.5 97.2 5.8 96.4 21.8 77.2

Learning Paradigm. The ablation study of the learning paradigm is shown in Table II. “I2L” refers to feeding depth maps into the image-to-latent (I2L) encoder-decoder and outputting the reconstructed depth maps.1 As shown in

1Only datasets with dense depth maps are evaluated.

Table II, the reconstruction accuracy of the I2L encoderdecoder is sufficiently high. That is to say, in the paradigm of the diffusion model, the main performance bottleneck is in the U-Net part, which is also the focus of our work. “Denoising” refers to predicting depth maps from noise using the denoising diffusion paradigm, while “Single-step” indicates directly

TABLE IV ABLATION STUDIES OF MODEL COMPONENTS. “FA” INDICATES THE FEATURE ALIGNMENT MODULE. “2S” DENOTES THE TWO-STAGE TRAINING STRATEGY. “FE” REFERS TO THE FOURIER ENHANCEMENT MODULE. FA EFFECTIVELY IMPROVES GENERALIZATION, WHILE FE AND 2S JOINTLY ENHANCE DETAIL PRESERVATION, AS INDICATED BY HYPERSIM F1.

Model Baseline: M.Base

KITTI NYUv2 Scannet ETH3D DIODE HyperSim

FA 2S FE

AbsRel↓ δ1↑ AbsRel ↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ F1↑

M.Base 8.7 93.1 5.1 97.3 5.5 97.2 5.8 96.4 21.8 77.2 0.306

M.FA ✓ 8.3 93.7 5.0 97.3 5.3 97.4 5.5 96.7 21.6 77.5 0.309 M.Stage2 ✓ ✓ 8.4 93.5 5.0 97.2 5.3 97.4 5.5 96.6 21.6 77.5 0.330

M.Full ✓ ✓ ✓ 8.2 93.7 5.0 97.2 5.3 97.4 5.5 96.7 21.5 77.6 0.339

predicting depth from RGB images. Obviously, applying the diffusion model in a deterministic manner is better suited for the discriminative task, which not only enhances the model’s generalization capability but also significantly improves inference efficiency. Actually, in the denoising diffusion paradigm, noise is progressively added to ensure outputs’ diversity, which is not desirable in deterministic tasks, thus impairing the performance. When predicting depth in an iterative deterministic way, where the U-Net’s output is iteratively input to the UNet for 4 times, the performance of the model is further improved. This is because the iterative paradigm aligns with the multi-step denoising process used by diffusion models, therefore, better harnessing the prior knowledge inherent in diffusion models. However, the iterative process inevitably leads to a low inference speed, resulting in approximately twice the inference time. To address this issue, we adapt features in diffusion models using carefully designed modules, enabling a single-step deterministic approach that achieves zero-shot performance comparable to the iterative paradigm while maintaining nearly identical inference time with the single-step paradigm.

Depth Preprocess. We conduct ablation studies on three different depth preprocessing methods, including depth, disparity, and square-root disparity (sqrt disp) under the singlestep deterministic paradigm. To ensure compatibility with the input range of Stable Diffusion v2, the preprocessed depth maps are normalized to the range of [-1, 1] using percentiles. The results are shown in Table III. Switching from depth prediction to disparity prediction results in a notable performance improvement, particularly on outdoor and mixed indoor-outdoor datasets. This improvement can be attributed to the fact that disparity amplifies the foreground structure, helping the model focus more on nearby objects, which is desired by outdoor applications such as autonomous driving. Furthermore, predicting square-root disparity yields an additional performance boost, which we adopt as our baseline. This is because square-root disparity produces a more uniform depth distribution, as illustrated in Fig. 5, allowing for more efficient use of the depth range.

Model Components. To assess the effectiveness of the proposed modules, we evaluate four model variants, with their generalization performance and edge precision summarized in Table IV. The baseline model (M.Base) adopts a single-step deterministic paradigm, directly regressing square-root disparity from RGB images. Incorporating the Feature Alignment module (M.FA) yields notable improvements in generalization

[Figure 92]

vKITTI depth distribution

- Fig. 5. Depth distribution of different depth preprocess methods on Virtual KITTI. Square-root disparity exhibits the most uniform distribution.

[Figure 93]

| |
|---|

[Figure 94]

| |
|---|

[Figure 95]

| |
|---|

[Figure 96]

| |
|---|

[Figure 97]

| |
|---|

[Figure 98]

| |
|---|

[Figure 99]

| |
|---|

|[Figure 100]<br><br>| |
|---|
|
|---|

[Figure 101]

| |
|---|

RGB M.Base M.FA

- Fig. 6. Overfitting to texture details of single-step diffusion model’s predictions. Directly applying diffusion models in a single-step discriminative way (M.Base) suffers from overfitting textures and missing the real structure. The Feature Alignment module (M.FA) effectively alleviates this issue.

performance with a 4.6% reduction in AbsRel on KITTI. This is because the high-quality external semantic features guide the model to capture global scene structure, rather than local textures. Besides, as illustrated in Fig 6, directly applying generative features to the perceptual depth estimation task leads to overfitting to complex texture, since diffusion models are primarily trained for realistic image synthesis where

TABLE V ABLATION OF EXTERNAL MODEL TYPE IN FEATURE ALIGNMENT MODULE. INTRODUCING VARIOUS EXTERNAL ENCODERS CAN IMPROVE THE GENERALIZATION PERFORMANCE OF THE MODEL, AMONG WHICH DINOV2 YIELDS THE GREATEST PERFORMANCE IMPROVEMENT.

External Model Type Baseline: M.Base

KITTI NYUv2 ETH3D ScanNet DIODE

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

M.Base 8.7 93.1 5.1 97.3 5.5 97.2 5.8 96.4 21.8 77.2 OpenCLIP [77] 8.5 93.3 5.0 97.3 5.4 97.4 5.6 96.5 21.8 77.1

AIMv2 [78] 8.4 93.4 5.1 97.3 5.5 97.3 5.6 96.6 21.7 77.5

SAM [79] 8.3 93.5 5.0 97.3 5.3 97.5 5.5 96.7 21.7 77.2 DINOv2 [61](w. REPA-E [80]) 10.4 89.1 6.6 95.9 7.2 95.6 7.0 95.1 23.2 76.0

DINOv2 [61] (M.FA) 8.3 93.7 5.0 97.3 5.3 97.4 5.5 96.7 21.6 77.5

TABLE VI ABLATION OF FEATURE ALIGNMENT LOCATION. “D1”, “D2” REFER TO THE FIRST AND SECOND DOWN BLOCKS OF THE U-NET. “MID” MEANS THE MIDDLE BLOCK OF THE U-NET. THE EFFECTIVENESS OF THE FEATURE ALIGNMENT MODULE INCREASES AS THE ALIGNED LAYER GROWS DEEPER.

Location Baseline: M.Base

KITTI NYUv2 ETH3D ScanNet DIODE

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

M.Base 8.7 93.1 5.1 97.3 5.5 97.2 5.8 96.4 21.8 77.2 D1 8.5 93.5 5.0 97.3 5.3 97.5 5.6 96.6 21.8 77.4 D2 8.4 93.6 5.1 97.3 5.4 97.4 5.5 96.6 21.5 77.7

Mid (M.FA) 8.3 93.7 5.0 97.3 5.3 97.4 5.5 96.7 21.6 77.5

TABLE VII ABLATION OF DETAIL PRESERVATION. “PIXEL” INDICATES APPLYING CONSTRAINTS AT THE PIXEL LEVEL. “Lh” REFERS TO THE WEIGHTED MULTI-DIRECTIONAL GRADIENT LOSS. “FE” DENOTES THE FOURIER ENHANCEMENT MODULE. “2S” MEANS THE TWO-STAGE TRAINING STRATEGY. THE PROPOSED MODULES AND TRAINING STRATEGY EFFECTIVELY ENHANCE THE MODEL’S DETAIL PRESERVATION CAPABILITY.

Model Baseline: M.Base

KITTI NYUv2 ETH3D Scannet DIODE HyperSim

pixel Lh FE 2S

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ F1↑

M.Base 8.7 93.1 5.1 97.3 5.5 97.2 5.8 96.4 21.8 77.2 0.306 M.Pixel ✓ 8.7 93.0 5.2 97.2 5.5 97.1 5.8 96.5 21.8 77.1 0.307

M.Huber ✓ ✓ 8.5 93.0 5.0 97.2 5.5 97.1 5.5 96.9 21.6 77.4 0.308 M.FE Huber ✓ ✓ ✓ 8.3 93.5 5.1 97.2 5.3 97.2 5.5 96.7 21.6 77.4 0.314

M.Full ✓ ✓ ✓ ✓ 8.2 93.7 5.0 97.2 5.3 97.4 5.5 96.7 21.5 77.6 0.337

preserving texture details is crucial. The proposed Feature Alignment module (M.FA) effectively eliminates the effect of unnecessary texture details by enhancing the model’s semantic representation capacity. In the second stage, we fine-tune the first-stage model at the pixel level. This strategy effectively improves edge precision, with an increase of 6.8% in the F1score, demonstrating the effectiveness of the two-stage training strategy in detail preservation. However, due to the difficulty of learning both global structure and local details in a single forward pass, the model suffers from a slight performance drop and limited edge precision. To address this, we introduce the Fourier Enhancement module. Through adaptively balancing low-frequency structure and high-frequency details in the frequency domain, the Fourier Enhancement module achieves finer details and overall performance improvements.

Feature Alignment module. We conduct ablation studies on different external encoders and feature alignment locations in the Feature Alignment module. As shown in Table V, the high-quality features of these external encoders can effectively modulate the features of the diffusion model and bring gains in zero-shot performance. Among them, DINOv2 has consistent performance improvements on various datasets. We also provide comparison with end-to-end training with REPA-E [80], which leads to performance degradation. This may be caused by insufficient training data and the disruption of the VAE’s

depth encoding capability when fine-tuned only on RGB features. The results of the ablation study on feature alignment locations are shown in Table VI. The U-Net is an encoderdecoder structure with three downsampling blocks, one middle block, and three upsampling blocks. The prior knowledge and scene perception abilities are primarily stored in the encoder part. Therefore, we perform feature alignment between the DINOv2 feature and the features from the first and second downsampling blocks and the middle block, respectively. As shown in Table VI, the effectiveness of the Feature Alignment module increases with the depth of the layer where it is applied. This occurs because the shallow U-Net layers capture more local information and are rich in details, while the middle layer features have a better global perception, which matches the global nature of the DINOv2 features. When constraints are imposed on shallow layers, the detailed information will be compromised.

Detail preservation. To validate the detail-preserving capability of the components we propose, we conduct a series of ablation experiments, as shown in Table VII. “M.Base” represents our baseline. Directly applying constraints in the pixel space (M.pixel) and using the weighted multi-directional gradient loss (M.Huber) cannot improve the model’s detail preservation capability, as indicated by the F1 metric. Moreover, direct pixel-level supervision can lead to structural dis-

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

RGB M.Pixel M.Stage2

Fig. 7. Effect of pixel-level supervision at different stages. Applying pixellevel constraints in the first stage (M.Pixel) introduces structural distortion, while the two-stage strategy (M.Stage2) alleviates this issue.

[Figure 108]

[Figure 109]

[Figure 110]

| |
|---|

| |
|---|

| |
|---|

[Figure 111]

[Figure 112]

[Figure 113]

| |
|---|

| |
|---|

| |
|---|

RGB M.FA M.Full

Fig. 8. Visualization of predictions from two stages. With the Fourier Enhancement Module and the two-stage training strategy, the final model (M.Full) exhibits excellent detail preservation ability.

tortions, as shown in Fig 7. This is because, in the single-step paradigm, the model is required to learn both low-frequency structural information and high-frequency details in a single forward pass, which introduces confusion during the learning process. When the Fourier Enhancement module is applied to adaptively enhance features in the frequency domain, both the model’s generalization and fine-grained details are improved. This demonstrates that the Fourier Enhancement module effectively mimics the iterative process’s focus on different frequency bands. Additionally, with the proposed two-stage training strategy, the model’s fine-grained detail is significantly enhanced. The second stage only need to optimize details based on the first stage model, thus simplifying the problem of capturing both structure and deatils. Fig. 8 presents the qualitative results of the two stages, where the fine-tuned model demonstrates a remarkable detail preservation ability, highlighting the effectiveness of our strategy in improving the visual quality.

V. LIMITATIONS

Although our method achieves performance comparable to data-driven approaches and detail preservation ability comparable to diffusion-based methods, the model’s large parameter size limits its deployment on mobile devices. Through experimentation, we identify some redundant parameters in the UNet, and removing these layers does not significantly affect performance. Therefore, reducing the model’s computational cost through effective pruning and distillation techniques will be a key focus of our future work.

VI. CONCLUSION

In this work, we propose DepthMaster, a method that crafts diffusion models for depth estimation. By incorporating the Feature Alignment module, we effectively mitigate the overfitting to texture details. Additionally, the Fourier Enhancement module enhances fine-grained detail preservation ability bi operating in the frequency domain. Benefiting from the careful design, DepthMaster achieves a significant boost in zero-shot performance and inference efficiency. Extensive experiments validate the effectiveness of our approach, which achieves

state-of-the-art performance in terms of generalization and detail preservation, outperforming other diffusion-based methods across various datasets.

REFERENCES

- [1] M. Sch¨on, M. Buchholz, and K. Dietmayer, “Mgnet: Monocular geometric scene understanding for autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 15804–15815.
- [2] Y. Wang, W.-L. Chao, D. Garg, B. Hariharan, M. Campbell, and K. Q. Weinberger, “Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 8445–8453.
- [3] Y. You, Y. Wang, W.-L. Chao, D. Garg, G. Pleiss, B. Hariharan, M. Campbell, and K. Q. Weinberger, “Pseudo-lidar++: Accurate depth for 3d object detection in autonomous driving,” in International Conference on Learning Representations, 2019.
- [4] L. Kong, S. Xie, H. Hu, L. X. Ng, B. Cottereau, and W. T. Ooi, “Robodepth: Robust out-of-distribution depth estimation under corruptions,” Advances in Neural Information Processing Systems, vol. 36, pp. 21298–21342, 2023.
- [5] X. Luo, J.-B. Huang, R. Szeliski, K. Matzen, and J. Kopf, “Consistent video depth estimation,” ACM Transactions on Graphics, vol. 39, no. 4, pp. 71–1, 2020.
- [6] J. Noraky and V. Sze, “Low power depth estimation of rigid objects for time-of-flight imaging,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 6, pp. 1524–1534, 2019.
- [7] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3836–3847.
- [8] P. Esser, J. Chiu, P. Atighehchian, J. Granskog, and A. Germanidis, “Structure and content-guided video synthesis with diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7346–7356.
- [9] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen, “Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 10579–10596, 2024.
- [10] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10371–10381.
- [11] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in Neural Information Processing Systems, vol. 37, pp. 21875–21911, 2024.
- [12] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12179–12188.
- [13] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. M¨uller, “Zoedepth: Zero-shot transfer by combining relative and metric depth,” arXiv preprint arXiv:2302.12288, 2023.

- [14] A. Eftekhar, A. Sax, J. Malik, and A. Zamir, “Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10786–10796.
- [15] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 3, pp. 1623–1637, 2020.
- [16] W. Yin, X. Wang, C. Shen, Y. Liu, Z. Tian, S. Xu, C. Sun, and D. Renyin, “Diversedepth: Affine-invariant depth prediction using diverse data,” arXiv preprint arXiv:2002.00569, 2020.
- [17] R. Zhu, C. Wang, Z. Song, L. Liu, T. Zhang, and Y. Zhang, “Scaledepth: Decomposing metric depth estimation into scale prediction and relative depth estimation,” arXiv preprint arXiv:2407.08187, 2024.
- [18] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, “Repurposing diffusion-based image generators for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9492–9502.
- [19] X. Fu, W. Yin, M. Hu, K. Wang, Y. Ma, P. Tan, S. Shen, D. Lin, and X. Long, “Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image,” in European Conference on Computer Vision, 2025, pp. 241–258.
- [20] M. Gui, J. Schusterbauer, U. Prestel, P. Ma, D. Kotovenko, O. Grebenkova, S. A. Baumann, V. T. Hu, and B. Ommer, “Depthfm: Fast generative monocular depth estimation with flow matching,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, 2025, pp. 3203–3211.
- [21] G. Xu, Y. Ge, M. Liu, C. Fan, K. Xie, Z. Zhao, H. Chen, and C. Shen, “What matters when repurposing diffusion models for general dense perception tasks?” in International Conference on Learning Representations, 2025.
- [22] J. He, H. Li, W. Yin, Y. Liang, L. Li, K. Zhou, H. Liu, B. Liu, and Y.-C. Chen, “Lotus: Diffusion-based visual foundation model for high-quality dense prediction,” in International Conference on Learning Representations, 2025.
- [23] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10684–10695.
- [24] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. M¨uller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in International Conference on Machine Learning, 2024.
- [25] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, “Representation alignment for generation: Training diffusion transformers is easier than you think,” in International Conference on Learning Representations, 2025.
- [26] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 15619–15629.
- [27] Y. LeCun, “A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27,” Open Review, vol. 62, no. 1, pp. 1–62, 2022.
- [28] H. Lee, H. Lee, S. Gye, and J. Kim, “Beta sampling is all you need: Efficient image generation strategy for diffusion models using stepwise spectral analysis,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. IEEE, 2025, pp. 4215–4224.
- [29] X. Yang, D. Zhou, J. Feng, and X. Wang, “Diffusion probabilistic model made slim,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22552–22562.
- [30] J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, and S. Yoon, “Perception prioritized training of diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11472–11481.
- [31] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in Neural Information Processing Systems, vol. 2, pp. 2366–2374, 2014.
- [32] I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab, “Deeper depth prediction with fully convolutional residual networks,” in International Conference on 3D vision, 2016, pp. 239–248.
- [33] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 7132–7141.
- [34] C. Wang, S. Lucey, F. Perazzi, and O. Wang, “Web stereo video supervision for depth prediction from dynamic scenes,” in International Conference on 3D vision, 2019, pp. 348–357.

- [35] M. Song, S. Lim, and W. Kim, “Monocular depth estimation using laplacian pyramid-based depth residuals,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 11, pp. 4381–4393, 2021.
- [36] W. Yuan, X. Gu, Z. Dai, S. Zhu, and P. Tan, “Neural window fullyconnected crfs for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3916–3925.
- [37] L. Piccinelli, C. Sakaridis, and F. Yu, “idisc: Internal discretization for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21477–21487.
- [38] W. Zhao, Y. Rao, Z. Liu, B. Liu, J. Zhou, and J. Lu, “Unleashing text-to-image diffusion models for visual perception,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 5729–5739.
- [39] S. Patni, A. Agarwal, and C. Arora, “Ecodepth: Effective conditioning of diffusion models for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 28285–28295.
- [40] N. Kondapaneni, M. Marks, M. Knott, R. Guimar˜aes, and P. Perona, “Text-image alignment for diffusion-based perception,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13883–13893.
- [41] F. Tosi, P. Z. Ramirez, and M. Poggi, “Diffusion models for monocular depth estimation: Overcoming challenging conditions,” in European Conference on Computer Vision, 2024, pp. 236–257.
- [42] Z. Song, R. Zhu, C. Wang, J. Deng, J. He, and T. Zhang, “Ec-depth: Exploring the consistency of self-supervised monocular depth estimation in challenging scenes,” arXiv preprint arXiv:2310.08044, 2023.
- [43] X. Qi, R. Liao, Z. Liu, R. Urtasun, and J. Jia, “Geonet: Geometric neural network for joint depth and surface normal estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 283–291.
- [44] L. Liu, R. Zhu, J. Deng, Z. Song, W. Yang, and T. Zhang, “Plane2depth: Hierarchical adaptive plane guidance for monocular depth estimation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 2, pp. 1136–1149, 2024.
- [45] D. Xu, W. Ouyang, X. Wang, and N. Sebe, “Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing,” in Proceedings of the IEEE/CVF Conference on

- Computer Vision and Pattern Recognition, 2018, pp. 675–684.

[46] P.-Y. Chen, A. H. Liu, Y.-C. Liu, and Y.-C. F. Wang, “Towards scene understanding: Unsupervised monocular depth estimation with semanticaware representation,” in Proceedings of the IEEE/CVF Conference on

- Computer Vision and Pattern Recognition, 2019, pp. 2624–2632.

- [47] W. Yin, Y. Liu, C. Shen, and Y. Yan, “Enforcing geometric constraints of virtual normal for depth prediction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 5684–5693.
- [48] S. Shao, Z. Pei, W. Chen, X. Wu, and Z. Li, “Nddepth: Normal-distance assisted monocular depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7931–7940.
- [49] Y. Cao, Z. Wu, and C. Shen, “Estimating depth from monocular images as classification using deep fully convolutional residual networks,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 28, no. 11, pp. 3174–3182, 2017.
- [50] H. Fu, M. Gong, C. Wang, K. Batmanghelich, and D. Tao, “Deep ordinal regression network for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 2002–2011.
- [51] S. F. Bhat, I. Alhashim, and P. Wonka, “Adabins: Depth estimation using adaptive bins,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4009–4018.
- [52] Z. Li, X. Wang, X. Liu, and J. Jiang, “Binsformer: Revisiting adaptive bins for monocular depth estimation,” IEEE Transactions on Image Processing, vol. 33, pp. 3964–3976, 2024.
- [53] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked-attention mask transformer for universal image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, June 2022, pp. 1290–1299.
- [54] S. F. Bhat, I. Alhashim, and P. Wonka, “Localbins: Improving depth estimation by learning local distributions,” in European Conference on Computer Vision, 2022, pp. 480–496.
- [55] S. Saxena, A. Kar, M. Norouzi, and D. J. Fleet, “Monocular depth estimation using diffusion models,” arXiv preprint arXiv:2302.14816, 2023.
- [56] Y. Ji, Z. Chen, E. Xie, L. Hong, X. Liu, Z. Liu, T. Lu, Z. Li, and P. Luo, “Ddp: Diffusion model for dense visual prediction,” in Proceedings of

- the IEEE/CVF International Conference on Computer Vision, 2023, pp. 21741–21752.
- [57] S. Saxena, C. Herrmann, J. Hur, A. Kar, M. Norouzi, D. Sun, and D. J. Fleet, “The surprising effectiveness of diffusion models for optical flow and monocular depth estimation,” Advances in Neural Information Processing Systems, vol. 36, pp. 39443–39469, 2023.
- [58] Y. Duan, X. Guo, and Z. Zhu, “Diffusiondepth: Diffusion denoising approach for monocular depth estimation,” in European Conference on Computer Vision, 2024, pp. 432–449.
- [59] S. Shao, Z. Pei, W. Chen, D. Sun, P. C. Chen, and Z. Li, “Monodiffusion: self-supervised monocular depth estimation using diffusion model,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 4, pp. 3664–3678, 2024.
- [60] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion5b: An open large-scale dataset for training next generation image-text models,” Advances in Neural Information Processing Systems, vol. 35, pp. 25278–25294, 2022.
- [61] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.
- [62] C. Si, Z. Huang, Y. Jiang, and Z. Liu, “Freeu: Free lunch in diffusion u-net,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4733–4743.
- [63] Y. Ye, K. Xu, Y. Huang, R. Yi, and Z. Cai, “Diffusionedge: Diffusion probabilistic model for crisp edge detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 6675– 6683.
- [64] P. J. Huber, “Robust estimation of a location parameter,” in Breakthroughs in statistics: Methodology and distribution, 1992, pp. 492–518.
- [65] W. Yin, J. Zhang, O. Wang, S. Niklaus, L. Mai, S. Chen, and C. Shen, “Learning to recover 3d scene shape from a single image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 204–213.
- [66] C. Zhang, W. Yin, B. Wang, G. Yu, B. Fu, and C. Shen, “Hierarchical normalization for robust monocular depth estimation,” Advances in Neural Information Processing Systems, vol. 35, pp. 14128–14139, 2022.
- [67] W. Yin, C. Zhang, H. Chen, Z. Cai, G. Yu, K. Wang, X. Chen, and C. Shen, “Metric3d: Towards zero-shot metric 3d prediction from a single image,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9043–9053.
- [68] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2017.
- [69] M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar, M. A. Bautista, N. Paczan, R. Webb, and J. M. Susskind, “Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10912–10922.
- [70] Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” arXiv preprint arXiv:2001.10773, 2020.
- [71] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013.
- [72] P. K. Nathan Silberman, Derek Hoiem and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in Proceedings of the European Conference on Computer Vision, 2012, pp. 746–760.
- [73] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 5828–5839.
- [74] T. Schops, J. L. Schonberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys, and A. Geiger, “A multi-view stereo benchmark with high-resolution images and multi-camera videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 3260–3269.
- [75] I. Vasiljevic, N. Kolkin, S. Zhang, R. Luo, H. Wang, F. Z. Dai, A. F. Daniele, M. Mostajabi, S. Basart, M. R. Walter et al., “Diode: A dense indoor and outdoor depth dataset,” arXiv preprint arXiv:1908.00463, 2019.
- [76] A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. R. Richter, and V. Koltun, “Depth pro: Sharp monocular metric depth in less than a second,” in International Conference on Learning Representations, 2025.
- [77] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco, C. Gordon, C. Schuhmann, L. Schmidt, and J. Jitsev, “Reproducible

- scaling laws for contrastive language-image learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2818–2829.
- [78] E. Fini, M. Shukor, X. Li, P. Dufter, M. Klein, D. Haldimann, S. Aitharaju, V. G. T. da Costa, L. B´ethune, Z. Gan, A. T. Toshev, M. Eichner, M. Nabi, Y. Yang, J. M. Susskind, and A. El-Nouby, “Multimodal autoregressive pre-training of large vision encoders,” arXiv preprint arXiv:2411.14402, 2024.
- [79] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.
- [80] X. Leng, J. Singh, Y. Hou, Z. Xing, S. Xie, and L. Zheng, “Repa-e: Unlocking vae for end-to-end tuning of latent diffusion transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 18262–18272.

[Figure 114]

Ziyang Song received a bachelor’s degree in Control Science and Engineering from University of Science and Technology of China in 2022. She is now pursuing a master’s degree in Control Science and Engineering at University of Science and Technology of China. Her research interests include computer vision, especially depth estimation and multi-view stereo.

[Figure 115]

Zerong Wang is currently a researcher & engineer in the Quality Enhancement Center of vivo. Before that, He received a bachelor’s degree in Computer Science from Northwestern Polytechnical University, Xi’an, China, in 2015 and a master’s degree in Computer Application Technology from Sichuan University, Chengdu, China, in 2018. His research interests include computer vision, artificial intelligence and computational photography.

[Figure 116]

Bo Li received the BSc and PhD degrees from the Department of Computer Science, Nanjing University, China, in 2014 and 2019, respectively. From 2020 to 2023, he served as Senior Researcher of Youtu Lab in Tencent, China. He is currently a senior expert at vivo image algorithm research department, China. His research interests include computer vision, pattern recognition and artificial intelligence.

[Figure 117]

Hao Zhang received the BSc and Master degrees from the Beihang University (BUAA), China, in 2012 and 2015, respectively. At the same time, he also obtained the French General Engineer Diploma from Ecole Centrale de Pekin and Ecole Centrale de Lyon. After that, he served as Senior Researcher of Alipay and Tencent, China. He is currently a Senior Researcher at vivo Mobile Communication Co., Ltd., China. His research interests include computer vision, artificial intelligence and computational photography.

[Figure 118]

Ruijie Zhu received a bachelor’s degree in Computer Science from Northwestern Polytechnical University, Xi’an, China, in 2022. He is currently pursuing a master’s degree in Electronic Engineering and Information Science at the University of Science and Technology of China. His research interests include computer vision and machine learning, especially depth estimation, 3D reconstruction and neural rendering.

[Figure 119]

Li Liu received the bachelor’s degree in Computer Science from Hefei University of Technology, China, in 2022. He is currently pursuing a master’s degree in Electronic and Information Engineering at the University of Science and Technology of China. His research interests include computer vision and machine learning, especially depth estimation and stereo matching.

[Figure 120]

Peng-Tao Jiang is currently a lead researcher & engineer in the Quality Enhancement Center of vivo. Before that, He was a post-doc researcher at Zhejiang University, working with Prof. Chunhua Shen. He received his PhD at Nankai University, advised by Prof. Ming-Ming Cheng. His current research interests include diffusion, image restoration, multitask learning, and segmentation.

Tianzhu Zhang received a bachelor’s degree in communications and information technology from the Beijing Institute of Technology, Beijing, China, in 2006 and a Ph.D. in pattern recognition and intelligent systems from the Institute of Automation, Chinese Academy of Sciences, Beijing, China, in 2011. He is currently a Professor at the Department of Automation, School of Information Science and Technology, University of Science and Technology of China. His current research interests include computer vision and multimedia. He served/serves as the

[Figure 121]

Area Chair for CVPR 2020, ECCV 2020, ICCV 2019, ACM MM 2019, WACV 2018, ICPR 2018, and MVA 2017, the Associate Editor for IEEE T-CSVT and Neurocomputing.

