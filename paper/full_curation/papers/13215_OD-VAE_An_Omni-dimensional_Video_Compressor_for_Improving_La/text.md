## OD-VAE: An Omni-dimensional Video Compressor for Improving Latent Video Diffusion Model

# arXiv:2409.01199v2[cs.CV]9Sep2024

Liuhan Chen1,3,*, Zongjian Li1,3,*, Bin Lin1,3, Bin Zhu1,3, Qian Wang1,3, Shenghai Yuan1,3, Xing Zhou3, Xinhua Cheng1,3, Li Yuan1,2,†

1Peking University, 2Peng Cheng Laboratory, 3Rabbitpre Intelligence

liuhanchen@stu.pku.edu.cn, yuanli-ece@pku.edu.cn

### Abstract

Variational Autoencoder (VAE), compressing videos into latent representations, is a crucial preceding component of Latent Video Diffusion Models (LVDMs). With the same reconstruction quality, the more sufficient the VAE’s compression for videos is, the more efficient the LVDMs are. However, most LVDMs utilize 2D image VAE, whose compression for videos is only in the spatial dimension and often ignored in the temporal dimension. How to conduct temporal compression for videos in a VAE to obtain more concise latent representations while promising accurate reconstruction is seldom explored. To fill this gap, we propose an omni-dimension compression VAE, named OD-VAE, which can temporally and spatially compress videos. Although OD-VAE’s more sufficient compression brings a great challenge to video reconstruction, it can still achieve high reconstructed accuracy by our fine design. To obtain a better trade-off between video reconstruction quality and compression speed, four variants of OD-VAE are introduced and analyzed. In addition, a novel tail initialization is designed to train OD-VAE more efficiently, and a novel inference strategy is proposed to enable OD-VAE to handle videos of arbitrary length with limited GPU memory. Comprehensive experiments on video reconstruction and LVDMbased video generation demonstrate the effectiveness and efficiency of our proposed methods.1 2

### 1. Introduction

Video generation has gained significant attention in both academia and industry, especially after the announcement of OpenAI’s SORA [4]. Currently, Latent Video Diffusion Models (LVDMs), such as MagicTime [35], Video-

1Code: https://github.com/PKU-YuanGroup/Open-Sora-Plan 2Equal Contribution: * ; Corresponding author: †

Composer [29], AnimateDiff [11], Stable Video Diffusion (SVD) [3], HiGen [20], Latte [16], SORA [4], Open-Sora [39], Open-Sora-Plan [15], have been the dominators in video generation for their stability, effectiveness, and scalability. These LVDMs share the same workflow: Variational Autoencoders (VAEs) [14] compress origin videos into latent representations. Then, the denoisers are trained to predict the noise added to these compressed representations.

However, the most frequently used VAE by LVDMs, Stable Diffusion VAE (SD-VAE) [19, 21], is initially designed for spatially compressing images instead of videos. When compressing a video, it treats each frame as an individual image, completely ignoring the redundancy in the temporal dimension. This results in temporally redundant latent representation, which increases the input size of the following denoisers, leading to great hardware consumption for LVDMs. In addition, the frame-wise compression of a video ignores the temporal information beneficial to reconstruction, causing lower reconstruction accuracy and reducing the quality of LVDMs’ generated results. Although the exploitation of temporal information is considered in the decoder of Stable Video Diffusion VAE (SVD-VAE) [3], its compression of videos in the temporal dimension remains absent, which still brings a great hardware burden to LVDMs.

Furthermore, temporal compression for videos has been explored in some works about autoregressive-based video generation [9, 32–34]. They utilize VQ-VAEs [27] to tempspatially compress videos into discrete tokens and the following transformers are learned to predict these tokens. Although these VQ-VAEs can’t provide continuous latent representations for LVDMs, they still indicate the feasibility of temporal compression in LVDMs’ VAEs.

To relieve the hardware burden of LVDMs and enhance their video generation ability with limited resources, we propose an omni-dimensional compression VAE (OD-

VAE), which can temporally and spatially compress videos into concise latent representations. Since a high temporal correlation exists in video frames, by strong 3D-CausalCNN architecture [34], our OD-VAE can reconstruct video accurately with additional temporal compression. The sufficient compression and effective reconstruction of OD-VAE will greatly improve the efficiency of LVDMs. To achieve a better trade-off between video reconstruction quality and compression speed, significant to the video generation results of LVDMs and their training speeds, respectively, we introduce and analyze four model variants of OD-VAE. To train our OD-VAE more efficiently, we propose a novel tail initialization to exploit the weight of SD-VAE. Besides, we propose novel temporal tiling, a split but one-frame overlap inference strategy, enabling OD-VAE to handle videos of arbitrary length with limited GPU memory.

Our contributions are summarized as follows:

- • We propose OD-VAE, an omni-dimensional video compressor with a high reconstructed accuracy, which improves the efficiency of LVDMs.
- • To achieve a better trade-off between video reconstruction quality and compression speed, we introduce and analyze four model variants of OD-VAE.
- • To further improve the training efficiency and inference ability of our OD-VAE, we propose novel tile initialization and temporal tiling, respectively.
- • Extensive experiments and ablations on video reconstruction and LVDM-based Video generation demonstrate the effectiveness and efficiency of our methods.

### 2. Related Work

#### 2.1. Latent Video Diffusion Model

Latent Video Diffusion Models (LVDMs) is a significant task in artificial intelligence [28]. It first use VAEs to compress videos into latent representations and then utilize denoisers to predict the noise added to them, have been developing rapidly since last year. The OpenAI’s SORA [4] that can generate videos of 1080P resolution and one minute long, greatly shocks the world. LVDMs can be divided into two kinds in terms of the structures of their denoisers. The first kind uses U-net-based denoisers [6, 22, 36], such as MagicTime [35], AnimateDiff [11], and Stable Video Diffusion (SVD) [3]. While the second kind utilizes Transformer-based denoisers [18], such as Latte [16], SORA [4], Open-Sora [39], and Vidu [2]. Whatever the structures of the denoisers are, the VAEs determine the sizes of the inputs to denoisers and the reconstructed accuracy from latent representations to videos. Thus, VAEs that provide concise representation while maintaining high reconstruction quality will greatly improve the efficiency of LVDMs.

#### 2.2. Variational Autoencoder

Variational Autoencoder (VAE) is initially designed for generation tasks by maximizing the Evidence Lower Bound (ELBO) of date[14]. Gradually, it has become a preceding component of other generation models and can be divided into two types. The first is VQ-VAEs [27], which compress videos into discrete tokens and are used by autoregressive-based video generation models [9, 33, 34]. In these VQ-VAEs, temporal compressions for videos have existed, and the 3D-causal-CNN-based MAGVIT-v2 [34] achieves state-of-the-art video reconstruction. However, the discrete representations provided by VQ-VAEs are unsuitable for LVDMs. The second is continuous VAEs, which compress videos into continuous representations and are used by LVDMs. Among them, Stable Diffusion VAE (SDVAE) [21], and its decoder enhancement version, Stable Video Diffusion VAE (SVD-VAE) [3], are the most popular. However, they only spatially compress videos while ignoring the temporal redundancy of videos. Besides, we have discovered two works that are concurrent with ours. One is OPS-VAE [39], which utilizes two cascading VAEs to spatially and temporally compress videos, respectively. The other is CV-VAE [38], which proposes a temporally compressed VAE but focuses more on latent space alignment to SD-VAE. We will comprehensively compare our OD-VAE and them in the experiment.

### 3. Method

In this section, we first provide the overview of OD-VAE, shown in Fig. 1. Then, we discuss the four model variants of OD-VAE, shown in Fig. 2. Finally, we introduce the tail initialization and temporal tiling.

#### 3.1. Overview of OD-VAE

Our OD-VAE adopts 3D-causal-CNN architecture to temporally and spatially compress videos into concise latent representations and can reconstruct them accurately, as shown in Fig. 1. Since the structure of SD VAE is mature and stable, the basic design of our 3D-causal-CNN architecture is derived from it, which will be introduced in the next subsection. Let E and D denote the encoder and the decoder of our OD-VAE, respectively. A video containing N + 1 frames is denoted as X = [x1,x2,...,xN+1] ∈ R(N+1)×H×W×3, and the i-th frame of X is expressed as xi ∈ RH×W×3. The compressed latent representation of X is denoted as Z ∈ R(n+1)×h×w×c. When processing the video X, OD-VAE keeps the temporal independence of its first frame x1, and only spatially compresses it. In contrast, the following frames xi(i > 1) will be compressed in both the temporal and spatial dimensions. This can be formulated as:

###### Z = E(X). (1)

[Figure 1]

- Figure 1. The overview of our OD-VAE. It adopts 3D-causal-CNN architecture to temp-spatially compress videos into concise latent representations and can reconstruct them accurately. This greatly enhances the efficiency of LVDMs.

The reconstruction is the inverse of the compression. We use Xˆ ∈ R(N+1)×H×W×3 to express the reconstructed video and the process can be formulated as:

Xˆ = D(Z). (2) The temporal and spatial compression rates of OD-VAE are ct = Nn and cs = Hh = Ww , respectively. We set cs = 8 following SD-VAE and find ct = 4 will be a good trade-off between sufficient compression and accurate reconstruction.

#### 3.2. Model Variants of OD-VAE

Since video compression is necessary for the training of LVDMs, increasing the compression speed of our OD-VAE can greatly improve their training efficiency. Hence, we introduce and analyze four different model variants of our OD-VAE, aiming to achieve a better trade-off between the compression speed and video reconstruction quality.

- Variant 1. An easy way to extend SD VAE to our 3D-

causal-CNN-based OD-VAE is inflating all the 2D convolutions into 3D convolutions by adding a temporal dimension to all 2D kernels, shown in Fig. 2(a). The video reconstruction ability of variant 1 is the best since its full-3D architecture can completely exploit the temporal and spatial information in the video by making features temp-spatially interact at each convolution. However, numerous expensive 3D convolutions in the network lead to a slow compression speed, lowering the training efficiency of LVDMs.

- Variant 2. Since numerous 3D convolutions in variant

1 lead to a slow compression speed, we utilize an intuitive way to reduce expensive 3D convolutions. Specifically, we replace half of the 3D convolutions in variant 1 with 2D convolutions and obtain variant 2, shown in Fig. 2(b). In variant 2, half of its convolutions are limited to only conducting spatial transformation for the input features, lowering the computational consumption of compression. As half of the convolutions can still process the features omnidimensionally, abundant temporal and spatial information

in a video is still well utilized, guaranteeing its reconstruction ability.

- Variant 3. However, in variant 1, the consumption of

each 3D convolution is different. The 3D convolutions in the outer blocks process large-sized features with huge expense while those in the inner blocks process small-sized features with little expense. Hence, replacing a 3D convolution in an outer block leads to a greater reduction in consumption than replacing one in an inner block. Based on this, we utilize a more reasonable replacement strategy for variant 1 and obtain variant 3. Specifically, we replace all the 3D convolutions in some outer blocks with 2D convolutions while maintaining the other inner blocks unchanged, shown in Fig. 2(c). With this strategy, the compression speed of variant 3 will probably be faster than that of variant 2.

- Variant 4. Since the decoder of OD-VAE doesn’t partic-

ipate in video compression, the convolution replacement in the decoder can’t improve the training efficiency of LVDMs while lowering the reconstruction accuracy. Therefore, we keep the decoder of variant 1 unchanged and only replace the 3D convolutions in the outer blocks of the encoder with 2D convolutions and obtain variant 4, shown in Fig. 2(d). With a full 3D decoder, the video reconstruction ability of variant 4 will probably be better than that of variant 3.

#### 3.3. Tail Initialization and Temporal Tiling

Tail Initialization. Notably, when N = 0, the video X degrades as an image and our OD-VAE can be viewed as an image VAE. This brings the potential for OD-VAE to inherit the spatial compression and reconstruction ability of powerful SD VAE. With this inheritance of ability in the spatial dimension, the training efficiency of our OD-VAE is higher, since the spatial prior will accelerate the convergence of our model. Hence, for better inheritance, we design a special initialization method to utilize the weight of 2D SD-VAE perfectly, named tail initialization. Specifically, we denote

[Figure 2]

- Figure 2. Four variants of our OD-VAE. Variant 1: inflating all the 2D convolutions in SD VAE to 3D convolutions. Variant 2: replacing half of the 3D convolutions in variant 1 with 2D convolutions. Variant 3: replacing the 3D convolutions in the outer blocks of variant 1’s encoder and decoder with 2D convolutions. Variant 4: replacing the 3D convolutions in the outer blocks of variant 1’s encoder with 2D convolutions.

a 5 dimension 3D convolution kernel in the OD-VAE as K3D ∈ RI×O×T×H×W, and its corresponding 4 dimension 2D kernel in SD VAE as K2D ∈ RI×O×H×W. For K3D, we use the weight of K2D to initial its temporally last element and set other elements to 0, expressed as:

K3D[:,:,i,:,:] =

K2D, if i = −1. 0, else.

(3)

We use F3D and F2D to denote the input feature maps of K3D and K2D, respectively. With tail initialization, before training, our OD-VAE satisfies the following equation:

##### F3D ∗ K3D = F2D ∗ K2D. (4)

The equation means that our OD-VAE can compress an image into a latent representation and reconstruct it accurately as SD-VAE without learning. This indicates that the spatial compression and reconstruction ability of SD-VAE is completely transferred to our OD-VAE. The strong spatial prior accelerates the convergence of our OD-VAE, greatly enhancing the training efficiency.

Temporal Tiling. Since long video generation has been a main trend, enabling our OD-VAE to handle videos of arbitrary length with limited GPU memory is necessary. Hence, we design a split but one-frame overlap inference strategy, named temporal tiling. Specifically, we temporally split a video X into M groups, denoting as [X1,X2,...,XM]. The last frame of Xi and the first frame of Xi+1 are the same. We compress each group Xi into latent representation Zi individually. Then, we drop the first

frames of Zi when i > 1 and concatenate Zi(1 ≤ i ≤ M) along temporal dimension to obtain Z. We introduce the same grouping mechanism to the reconstructed video Xˆ that Xˆ = [Xˆ1,Xˆ2,...,XˆM]. To reconstruct Z as Xˆ, we first decode Zi into Xˆi individually. Then, we drop the first frames of Xˆi when i > 1 and concatenate Xˆi(1 ≤ i ≤ M) along temporal dimension. As a high temporal correlation exists in video frames, the overlap can connect each group well and greatly reduce compressed and reconstructed errors.

### 4. Experiment

In this section, we first introduce the experimental setting, including models, training strategy, and evaluation details. Then, comprehensive comparisons between OD-VAE and other baselines on video reconstruction and LVDM-based video generation are conducted to demonstrate the superiority of our OD-VAE. Finally, extensive ablations are provided to certify the effectiveness of our proposed methods.

#### 4.1. Experimental Setting

Models. To demonstrate the effectiveness and efficiency of our OD-VAE, we compare it with six other state-of-theart commonly used VAEs in terms of video reconstruction and LVDM-based video generation, including: (1) VQGAN [8]: a widely used image VQ-VAE. Following [38], we use its f8-8192 version in our experiment. (2) TATS [9]: a 3D video VQ-VAE applied to autoregressive-based video generation. (3) SD-VAE [21]: the most frequently used image VAE by LVDMs. Following [38], we use its nu-

[Figure 3]

- Figure 3. Video generation results of LVDMs with different VAEs on the SkyTimelapse dataset. As the figure shows, with OD-VAE, LVDM can generate more realistic and high-quality videos.

merically stable version, SD2.1-VAE. (4) SVD-VAE [3]: A video VAE obtained by enhancing the decoder of SDVAE. It shares the same encoder structure as SD-VAE. (5) CV-VAE [38]: a video VAEs contemporaneous with our research. (6) OPS-VAE [39]: another video VAEs also contemporaneous with our research. It first conducts spatial downsample then temporal downsample to an input video. As discrete VQGAN and TATS aren’t suitable for LVDMs, they are only used for experiments on video reconstruction. In the method section, we introduce four model variants of our OD-VAEs. We use variant 4 of our OD-VAE to compare to other baselines, since according to the ablations, variant 4 achieves the best trade-off between the video reconstruction quality and compression speed among all the variants.

Training strategy. We use Adma optimizer [13] to train our OD-VAE for 650k steps, with a constant learning rate 1 × 10−5 and batch size 8. The training dataset of our ODVAE contains 440k self-scrape internet videos and 220k videos from the K400 dataset [5]. During training, all the input videos are processed to clips of 25-frame length and 256 × 256 resolution. Following [8, 19], the loss function contains a reconstruction term, a KL term, and an adversarial term [10]. To obtain more stable training results, following [18, 19], we utilize an exponential moving average

(EMA) of OD-VAE weights over training with a decay of 0.999. Since SD2.1-VAE is numerically stable, We use its weights to initialize our OD-VAE, enhancing the training efficiency. The training is conducted on 8 NVIDIA 80G A100 GPUs with Pytorch [17].

Evaluation details. For evaluation on video reconstruction, we select two popular large open-domain video datasets, WebVid-10M [1] and Panda-70M [7]. we only use their validation sets for efficiency and fairness. For each video in these two validation sets, we transform it to a clip of 25-frame length and 256 × 256 resolution. To quantify models’ video reconstruction ability, we use three popular metrics, peak signal-to-noise ratio (PSNR) [12], structural similarity index measure (SSIM) [30], and Learned Perceptual Image Patch Similarity (LPIPS) [37]. We also use the video compression rate (VCPR) and the number of parameters (Params) to denote the video compression level and the network complexity of these VAEs, respectively. To evaluate these VAEs’ effect on LVDM-based video generation, we fix the structure of the denoiser and change its previous VAE. We select Latte’s denoiser [16], since it uses a novel SORA-like transformer-based structure and achieves excellent results in LVDM-based video generation. Following [9], we choose two public datasets, UCF101 [24] and

- Table 1. Video reconstruction results of VAEs on the WebVid-10M validation set and Panda-70M validation set, along with their numbers of parameters and video compression rate. For video reconstruction metrics, with the highest video compression rate, the best and second scores are indicated in bold and underlined.

|Method|VCPR Params<br><br>|WebVid-10M|Panda-70M|
|---|---|---|---|
| | |PSNR(↑) SSIM(↑) LPIPS(↓)|PSNR(↑) SSIM(↑) LPIPS(↓)|
|VQGAN SD-VAE SVD-VAE<br><br>|64(1 × 8 × 8)<br><br>69.00M 83.65M 97.74M<br><br>|26.26 0.7699 0.0906<br><br>30.19 0.8379 0.0568<br><br>31.15 0.8686 0.0547<br><br><br>|26.07 0.8295 0.0722<br><br>30.40 0.8894 0.0396<br>31.00 0.9058 0.0379<br>|
|TATS CV-VAE OPS-VAE OD-VAE|256(4 × 8 × 8)<br><br>52.19M 182.45M 393.34M 239.19M<br><br>|23.10 0.6758 0.2645<br><br>30.76 0.8566 0.0803<br><br>31.12 0.8569 0.1003<br><br><br>31.16 0.8694 0.0586<br><br>|21.77 0.6680 0.2858<br><br>29.57 0.8795 0.0673 31.06 0.8969 0.0666<br><br>30.49 0.8970 0.0454<br><br><br>|

- Table 2. Video generation results and the training efficiency of LVDMs with different VAEs on the UCF101 and SkyTimelapse dataset. The best and second scores are indicated in bold and underlined for all these metrics.

|Method|UCF101<br><br>|SkyTimelapse|
|---|---|---|
| |FVD(↓) KVD(↓) IS(↑) TMem(↓) TSpeed(↑)|FVD(↓) KVD(↓) TMem(↓) TSpeed(↑)|
|SD-VAE SVD-VAE CV-VAE OPS-VAE<br><br>|1685.29 116.10 33.00 74364MB 0.87it/s 1663.98 108.27 31.41 74364MB 0.87it/s 1380.43 129.29 61.11 30628MB 1.31it/s 1502.64 142.4 53.13 31220MB 1.52it/s<br><br>|325.00 26.28 74498MB 0.86it/s 285.23 25.10 74498MB 0.86it/s<br>326.86 23.57 30938MB 1.29it/s 312.22 24.47 31516MB 1.49it/s<br><br><br>|
|OD-VAE|1315.13 110.88 58.98 30520MB 1.80it/s<br><br>|294.31 20.76 30834MB 1.76it/s<br><br>|

- Table 3. The compression speed (CSpeed) of the four variants and the training speed (TSpeed) of LVDM with them.

Model Variant 1 Variant 2 Variant 3 Variant 4

CSpeed(↑) 2.15it/s 2.44it/s 4.13it/s 4.13it/s TSpeed(↑) 1.26it/s 1.38it/s 1.80it/s 1.80it/s

SkyTimelapse [31], for class-conditional and unconditional generation respectively. We use almost the same setting introduced in [16] to train Latte’s denoiser with these VAEs for 200k steps. The only difference is that we use longer video clips of 81-frame length and adjust the batch size to fit the memory limitation of a single GPU. To assess the quality of the generated videos on the two datasets, we employ two popular metrics, Frechet Video Distance (FVD) and Kernel Video Distance (KVD) [26]. In addition, we also report models’ Inception Score (IS) [23] on the UCF101 dataset, calculated by a trained C3D model [25]. These metrics are calculated based on 2048 samples. To measure LVDM’s efficiency with different VAEs, we list their training GPU memory consumption (TMem) and training speed (TSpeed) on the two datasets. These tests are conducted on NVIDIA 80G A100 GPUs.

- 4.2. Comparison with Other Baselines

compress videos, its reconstruction quality is not inferior to commonly used SD-VAE and SVD-VAE. For example, the PSNR and SSIM of our OD-VAE on the WebVid-10M validation set are 0.97 and 0.0315 higher than that of SD-VAE, respectively. Compared to SVD-VAE, although the overall performance of our OD-VAE is worse, its PSNR and SSIM on the WebVid-10M validation set are still slightly higher. This proves that our OD-VAE can fully exploit the temporal redundancy of video frames to obtain a more concise latent representation while maintaining high reconstructed quality. Furthermore, our OD-VAE behaves better than the two works concurrent with us, CV-VAE and OPSVAE, which proves the effectiveness of our model design and training strategy. For example, the SSIM of OD-VAE on the WebVid-10M validation set is 0.0128 and 0.0125 higher than that of CV-VAE and OPS-VAE, respectively. On the Panda-70M validation set, the LPIPS of our ODVAE is 0.0219 and 0.0211 lower than that of CV-VAE and OPS-VAE, respectively.

In Table. 2, we display the LVDM-based video generation results of our OD-VAE and other baselines. The results in Table. 2 show that, through 4× temporal compression of VAEs, the efficiency of LVDM is greatly improved. On the two datasets, the video generation results of our OD-VAE are better than that of SD-VAE and SVDVAE, while the training consumption is greatly reduced. For example, on the UCF101 dataset, with the same training steps, using our OD-VAE can achieve better FVD (370.16

We display the video reconstruction results of our ODVAE and other baselines in Table. 1. The results in Table. 1 reflect that although OD-VAE can 4× temporally

[Figure 4]

[Figure 5]

[Figure 6]

(a) PSNR of the four variants. (b) LPIPS of the four variants. (c) FVD of the four variants.

[Figure 7]

[Figure 8]

[Figure 9]

(d) PSNR of the three initialization methods. (e) LPIPS of the three initialization methods. (f) FVD of the three initialization methods.

- Figure 4. (a), (b) are the PSNR and LPIPS of the four variants on the WebVid-10M validation set. (c) is the FVD of the four variants on the UCF101 dataset. (d), (e) are the PSNR and LPIPS of the three initialization methods on the WebVid-10M validation set. (f) is the FVD of the three initialization methods on the UCF101 dataset.

lower than that of SD-VAE and 348.85 lower than that of SVD-VAE) and faster training speed (2.06× that of SDVAE and SVD-VAE). Furthermore, compared to CV-VAE and OPS-VAE, although the video compression rate is the same, our OD-VAE brings better video generation results and lower training consumption to LVDM. For example, on the SkyTimelapse dataset, with the same training steps, using our OD-VAE can obtain better FVD (32.55 lower than that of CV-VAE and 17.91 lower than that of OPS-VAE) and faster training speed (0.47it/s faster than that of SDVAE and 0.27it/s faster than that of OPS-VAE). Besides, we show some visual results of LVDM with different VAEs on the SkyTimelapse dataset in Fig. 3. According to Fig. 3, with OD-VAE, LVDM can generate more realistic and high-quality videos.

#### 4.3. Ablation Experiment

Model variant. To obtain the variant with the best tradeoff between video reconstruction quality and compression speed, we train the four model variants of OD-VAE for 150k steps with the same setting mentioned above. We show their PSNR and LPIPS on the WebVid-10M validation set in Fig. 4 (a) and (b). Besides, we use their final checkpoints to train Latte’s denoiser on the UCF101 dataset with the

same setting mentioned above and report their FVD in Fig. 4 (c). The compression speed (CSpeed) of the four variants, calculated by processing videos of 81-frame length and 256 × 256 resolution, along with the training speed (TSpeed) of LVDM with them on the UCF101 dataset, are listed in Table. 3. According to Fig. 4 (a), (b), the PSNR and SSIM of variant 4 are slightly worse than that of variant 1 but better than the other variants. Since the reconstruction abilities of the four variants are close, using them as the preceding components of LVDM causes similar results of video generation, shown in Fig. 4 (c). However, according to the Table. 3, the compression speed of variant 4 is much faster than that of variant 1 and variant 2, bringing extreme efficiency enhancement to the training of LVDM. Hence, our OD-VAE utilizes variant 4 as the final structure, achieving the best trade-off between video reconstruction quality and compression speed.

Initialization Method. To verify the effectiveness of our tail initialization, we compare it with two other initialization methods, average initialization, and random initialization. Average initialization can be expressed as:

##### K2D T

(1 ≤ i ≤ T). (5) The random initialization means we randomly initialize our

K3D[:,:,i,:,:] =

Table 4. The PSNR, LPIPS of our OD-VAE (w or w/o temporal tiling) on the WebVid-10 validation set and the FVD, IS of corresponding LVDM on the UCF101 dataset.

|Temporal Tiling|WebVid-10M<br><br>|UCF101|
|---|---|---|
| |PSNR↑ LPIPS↓|FVD↓ IS↑<br><br>|
|× ✓<br><br>|31.05 0.0589 30.98 0.0591|1315.13 58.98 1331.46 58.89<br><br>|

OD-VAE with Gaussian random numbers. We initialize our OD-VAE with the three methods and train the three versions for 150k steps with the same setting mentioned above, respectively. We show their PSNR and LPIPS on the WebVid10M validation set in Fig. 4 (d) and (e). Besides, we use their final checkpoints to train Latte’s denoiser on the UCF101 dataset with the same setting mentioned above and report their FVD in Fig. 4 (f). According to Fig. 4 (d), (e), and (f), with the same training steps, using tail initialization can greatly improve the video reconstruction ability of our OD-VAE and the video generation quality of LVDM.

Temporal Tiling When directly compressing and reconstructing a video of 256 × 256 resolution on an NVIDIA 80G A100 GPU, the maximum length of frames our ODVAE can process is 125. With temporal tiling, our OD-VAE can handle a video in groups and the original length limitation disappears. This enables LVDM to generate longer videos. To evaluate the effect of temporal tiling on video reconstruction and LVDM-based video generation, we conduct experiments on the WebVid-10M validation set and the UCF101 dataset with the same setting mentioned above, respectively. We fix the length of a group to 33 and increase the frame length of the WebVid-10M validation clips from 33 to 97. In Table. 4, we list the PSNR and LPIPS on the WebVid-10M validation set, and the FVD and IS on the UCF101 dataset. According to Table. 4, with temporal tiling, these metrics slightly decrease, which means temporal tiling will not do much harm to the video reconstruction ability of our OD-VAE and the video generation quality of corresponding LVDM.

### 5. Conclusion

In this work, we proposed a novel omni-dimensional compression VAE for improving LVDMs, termed OD-VAE. It utilized effective 3D-causal-CNN architecture to 4× temporally and 8× spatially compress videos into latent representations while maintaining high reconstructed accuracy. These more concise representations reduced the input size of LVDMs’ denoisers, greatly improving the efficiency of LVDMs. To achieve a better trade-off between video reconstruction quality and compression speed, we introduced and analyzed four variants of our OD-VAE. To train ODVAE more efficiently, we proposed a novel tail initializa-

tion to exploit the weight of SD-VAE perfectly. Besides, we proposed temporal tiling, a split but one-frame overlap inference strategy, enabling our OD-VAE to process videos of arbitrary length with limited GPU memory. Comprehensive experiments and ablations on video reconstruction and LVDM-based video generation demonstrated the effectiveness and efficiency of our proposed methods.

### References

- [1] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738,

2021. 5

- [2] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024. 2
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 2, 5
- [4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 1, 2

- [5] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 5
- [6] Liuhan Chen, Yirou Wang, and Yongyong Chen. End-to-end xy separation for single image blind deblurring. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1273–1282, 2023. 2
- [7] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 5
- [8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 4, 5
- [9] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022. 1, 2, 4, 5
- [10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 5

- [11] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. In The Twelfth International Conference on Learning Representations. 1, 2
- [12] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010. 5
- [13] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. International Conference on Learning Representations, 2015. 5
- [14] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. stat, 1050:1, 2014. 1, 2
- [15] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 1
- [16] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 1, 2, 5, 6
- [17] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 5
- [18] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2, 5

- [19] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 5
- [20] Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, and Nong Sang. Hierarchical spatio-temporal decoupling for text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6635–6645,

2024. 1

- [21] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 4
- [22] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 2
- [23] Masaki Saito, Shunta Saito, Masanori Koyama, and Sosuke Kobayashi. Train sparsely, generate densely: Memoryefficient unsupervised training of high-resolution temporal gan. International Journal of Computer Vision, 128(10): 2586–2606, 2020. 6
- [24] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. A dataset of 101 human action classes from videos in the

wild. Center for Research in Computer Vision, 2(11):1–7,

2012. 5

- [25] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 4489–4497,

2015. 6

- [26] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [27] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 1, 2
- [28] Shaodong Wang, Yunyang Ge, Liuhan Chen, Haiyang Zhou, Qian Wang, Xinhua Cheng, and Li Yuan. Prompt2poster: Automatically artistic chinese poster creation from prompt only. In ACM Multimedia 2024. 2
- [29] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024. 1
- [30] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5
- [31] Wei Xiong, Wenhan Luo, Lin Ma, Wei Liu, and Jiebo Luo. Learning to generate time-lapse videos using multi-stage dynamic generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2364–2373, 2018. 6
- [32] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 1
- [33] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 2
- [34] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. International Conference on Learning Representations, 2024. 1, 2
- [35] Shenghai Yuan, Jinfa Huang, Yujun Shi, Yongqi Xu, Ruijie Zhu, Bin Lin, Xinhua Cheng, Li Yuan, and Jiebo Luo. Magictime: Time-lapse video generation models as metamorphic simulators. arXiv preprint arXiv:2404.05014, 2024. 1, 2
- [36] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. arXiv preprint arXiv:2406.18522, 2024. 2
- [37] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of

- deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [38] Sijie Zhao, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Muyao Niu, Xiaoyu Li, Wenbo Hu, and Ying Shan. Cv-vae: A compatible video vae for latent generative video models. arXiv preprint arXiv:2405.20279, 2024. 2, 4, 5
- [39] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 1, 2, 5

